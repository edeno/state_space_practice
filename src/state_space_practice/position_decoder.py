"""Position decoding from population spike trains via Laplace-EKF.

Decodes the animal's 2D position from simultaneous spike trains of
neurons with known place fields. The latent state is position (and
optionally velocity), and observations are Poisson spike counts with
rates determined by each neuron's spatial tuning curve.

References
----------
[1] Brown, E.N., Frank, L.M., Tang, D., Quirk, M.C. & Wilson, M.A. (1998).
    A statistical paradigm for neural spike train decoding applied to
    position prediction from ensemble firing patterns of rat hippocampal
    place cells. J Neuroscience 18(18), 7411-7425.
[2] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
    Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
    Neural Computation 16, 971-998.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import _kalman_smoother_update, symmetrize
from state_space_practice.point_process_kalman import (
    _point_process_laplace_update,
    _safe_expected_count,
)

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveInflationConfig:
    """Configuration for innovation-based covariance inflation.

    When the Poisson score at the predicted state is larger than the
    local Fisher information predicts, the predicted covariance is
    inflated so the filter does not collapse onto a wrong trajectory.

    Parameters
    ----------
    enabled : bool
        Whether to apply adaptive inflation.
    gain : float
        Scaling factor *c* in alpha = clip(1 + c*(s - 1), 1, max_alpha).
        Larger values inflate more aggressively per unit excess score.
    max_alpha : float
        Maximum multiplicative inflation factor.
    epsilon : float
        Regularisation added to Fisher diagonal for inversion stability.
    min_fisher_trace : float
        Minimum Fisher trace below which inflation is skipped (no
        spikes → no information → nothing to be inconsistent with).
    """

    enabled: bool = True
    gain: float = 0.5
    max_alpha: float = 10.0
    epsilon: float = 1e-6
    min_fisher_trace: float = 1e-8

    def __post_init__(self):
        if self.gain < 0:
            raise ValueError(f"gain must be >= 0, got {self.gain}")
        if self.max_alpha < 1.0:
            raise ValueError(f"max_alpha must be >= 1.0, got {self.max_alpha}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.min_fisher_trace < 0:
            raise ValueError(
                f"min_fisher_trace must be >= 0, got {self.min_fisher_trace}"
            )


def build_position_dynamics(
    dt: float,
    q_pos: float = 1.0,
    q_vel: float = 10.0,
    include_velocity: bool = True,
) -> tuple[Array, Array]:
    """Build transition matrix and process noise for position dynamics.

    Parameters
    ----------
    dt : float
        Time bin width in seconds.
    q_pos : float
        Position process noise (cm^2/s).
    q_vel : float
        Velocity process noise (cm^2/s^3). Only used if include_velocity=True.
    include_velocity : bool, default=True
        If True, state is [x, y, vx, vy] with constant-velocity dynamics.
        If False, state is [x, y] with random walk dynamics.

    Returns
    -------
    A : Array, shape (n_state, n_state)
        Transition matrix.
    Q : Array, shape (n_state, n_state)
        Process noise covariance.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    if include_velocity:
        A = jnp.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        Q = jnp.diag(jnp.array([
            q_pos * dt,
            q_pos * dt,
            q_vel * dt,
            q_vel * dt,
        ]))
    else:
        A = jnp.eye(2)
        Q = jnp.eye(2) * q_pos * dt

    return A, Q


class PlaceFieldRateMaps:
    """Pre-computed place field rate maps for position decoding.

    Stores firing rate maps on a regular grid and provides fast
    interpolated evaluation at arbitrary positions. Computes Jacobians
    via finite differences for the Laplace-EKF.

    Parameters
    ----------
    rate_maps : np.ndarray, shape (n_neurons, n_grid_y, n_grid_x)
        Firing rate (Hz) for each neuron on the spatial grid.
    x_edges : np.ndarray, shape (n_grid_x,)
        X coordinates of the grid.
    y_edges : np.ndarray, shape (n_grid_y,)
        Y coordinates of the grid.
    """

    def __init__(
        self,
        rate_maps: ArrayLike,
        x_edges: ArrayLike,
        y_edges: ArrayLike,
        occupancy_mask: Optional[np.ndarray] = None,
        spike_histograms: Optional[np.ndarray] = None,
        occ_histogram: Optional[np.ndarray] = None,
        kde_sigma: Optional[float] = None,
    ):
        self.rate_maps = np.asarray(rate_maps)
        self.x_edges = np.asarray(x_edges)
        self.y_edges = np.asarray(y_edges)
        self.occupancy_mask = occupancy_mask

        if len(self.x_edges) < 2 or len(self.y_edges) < 2:
            raise ValueError("x_edges and y_edges must have at least 2 points")
        if self.rate_maps.ndim != 3:
            raise ValueError(
                f"rate_maps must be 3D (n_neurons, n_grid_y, n_grid_x), "
                f"got shape {self.rate_maps.shape}"
            )
        n_grid_y, n_grid_x = self.rate_maps.shape[1], self.rate_maps.shape[2]
        if n_grid_y != len(self.y_edges) or n_grid_x != len(self.x_edges):
            raise ValueError(
                f"rate_maps spatial dims ({n_grid_y}, {n_grid_x}) must match "
                f"y_edges ({len(self.y_edges)}) and x_edges ({len(self.x_edges)})"
            )

        self.n_neurons = self.rate_maps.shape[0]

        # Validate uniform grid spacing (bilinear interpolation assumes this)
        x_diffs = np.diff(self.x_edges)
        y_diffs = np.diff(self.y_edges)
        if not (np.allclose(x_diffs, x_diffs[0]) and np.allclose(y_diffs, y_diffs[0])):
            raise ValueError(
                "x_edges and y_edges must be uniformly spaced for "
                "bilinear interpolation"
            )

        # Precompute log rate maps (clamp rates below 1e-10 to avoid log(0))
        self._log_rate_maps = np.log(np.maximum(self.rate_maps, 1e-10))

        # Grid spacing
        self._dx = float(x_diffs[0])
        self._dy = float(y_diffs[0])

        # JAX arrays for JIT-compatible interpolation
        self._jax_log_rate_maps = jnp.array(self._log_rate_maps)
        self._jax_x_edges = jnp.array(self.x_edges)
        self._jax_y_edges = jnp.array(self.y_edges)

        # KDE sufficient statistics for analytical rate evaluation.
        # When available, rates and gradients are computed from kernel
        # sums over the gridded histograms rather than interpolating
        # the pre-smoothed rate map.  This preserves the full spatial
        # gradient information of the Gaussian kernel.
        self._kde_sigma = kde_sigma
        if spike_histograms is not None and occ_histogram is not None and kde_sigma is not None:
            # spike_histograms: (n_neurons, n_grid_y, n_grid_x) — raw spike counts per bin
            # occ_histogram: (n_grid_y, n_grid_x) — occupancy time per bin
            # Grid center coordinates as 2D meshgrid
            xx, yy = np.meshgrid(self.x_edges, self.y_edges)  # (ny, nx)
            self._jax_grid_xy = jnp.stack([xx.ravel(), yy.ravel()], axis=1)  # (ny*nx, 2)
            self._jax_spike_hists = jnp.array(
                spike_histograms.reshape(self.n_neurons, -1)  # (n_neurons, ny*nx)
            )
            self._jax_occ_hist = jnp.array(occ_histogram.ravel())  # (ny*nx,)
            self._jax_kde_sigma2 = jnp.array(kde_sigma ** 2)
            self._use_analytical = True
        else:
            self._use_analytical = False

    @classmethod
    def from_place_field_model(
        cls,
        model,
        n_grid: int = 50,
        time_slice: Optional[slice] = None,
    ) -> PlaceFieldRateMaps:
        """Construct rate maps from a fitted PlaceFieldModel.

        Parameters
        ----------
        model : PlaceFieldModel
            Fitted encoding model.
        n_grid : int
            Grid resolution per dimension.
        time_slice : slice or None
            Time window to average over. None = full session.

        Returns
        -------
        PlaceFieldRateMaps
        """
        grid, x_edges, y_edges = model.make_grid(n_grid)
        rate_maps = np.zeros((model.n_neurons, n_grid, n_grid))
        for neuron_idx in range(model.n_neurons):
            rate, _ = model.predict_rate_map(
                grid, time_slice=time_slice, neuron_idx=neuron_idx
            )
            rate_maps[neuron_idx] = rate.reshape(n_grid, n_grid)
        return cls(rate_maps=rate_maps, x_edges=x_edges, y_edges=y_edges)

    @classmethod
    def from_spike_position_data(
        cls,
        position: np.ndarray,
        spike_counts: np.ndarray,
        dt: float,
        n_grid: int = 50,
        sigma: float = 5.0,
        occupancy_tau: float = 1.0,
    ) -> PlaceFieldRateMaps:
        """Estimate rate maps from position and spike data using KDE.

        Uses occupancy-dependent shrinkage: at well-sampled locations
        the local KDE rate dominates; at poorly-sampled locations the
        estimate shrinks toward each neuron's session-wide mean rate.
        This prevents overconfident near-zero rates at low-occupancy
        positions, which would otherwise cause the Laplace-EKF Newton
        step to explode when a spike is observed.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
            Animal position in cm at each time bin.
        spike_counts : np.ndarray, shape (n_time,) or (n_time, n_neurons)
            Spike counts per time bin (and per neuron).
        dt : float
            Time bin width in seconds.
        n_grid : int
            Number of spatial bins per dimension.
        sigma : float
            Gaussian smoothing kernel width in cm.
        occupancy_tau : float
            Shrinkage timescale in seconds.  Controls how much
            occupancy is needed before the local rate estimate is
            trusted over the session-wide baseline.  At a location
            with ``occ`` seconds of data, the shrinkage weight toward
            the local estimate is ``occ / (occ + tau)``.  Default 1.0 s.
        Returns
        -------
        PlaceFieldRateMaps
        """
        from scipy.ndimage import gaussian_filter

        position = np.asarray(position)
        spike_counts = np.asarray(spike_counts)
        if spike_counts.ndim == 1:
            spike_counts = spike_counts[:, None]

        if position.shape[0] != spike_counts.shape[0]:
            raise ValueError(
                f"position and spike_counts must have the same number of "
                f"time bins, got {position.shape[0]} and {spike_counts.shape[0]}"
            )

        n_neurons = spike_counts.shape[1]

        # Guard against degenerate position ranges (e.g., stationary animal)
        x_min, x_max = position[:, 0].min(), position[:, 0].max()
        y_min, y_max = position[:, 1].min(), position[:, 1].max()
        if x_max - x_min < 1e-6:
            x_min, x_max = x_min - sigma, x_min + sigma
        if y_max - y_min < 1e-6:
            y_min, y_max = y_min - sigma, y_min + sigma

        # Extend the range by half a bin so that bin centers span the
        # full observed position range.  Without this, the interpolation
        # grid covers only [x_min + dx/2, x_max - dx/2], leaving a
        # half-bin gap on each side where observed positions get
        # clipped and the observation gradient is zero.
        dx = (x_max - x_min) / n_grid
        dy = (y_max - y_min) / n_grid
        x_bin_edges = np.linspace(x_min - dx / 2, x_max + dx / 2, n_grid + 1)
        y_bin_edges = np.linspace(y_min - dy / 2, y_max + dy / 2, n_grid + 1)

        # Compute occupancy
        occ, _, _ = np.histogram2d(
            position[:, 0], position[:, 1],
            bins=[x_bin_edges, y_bin_edges],
        )
        occ_time = occ * dt
        # Smooth occupancy (zero-pad at array boundaries to avoid
        # reflecting occupied-region data into the padding bins)
        sigma_bins = sigma / (x_bin_edges[1] - x_bin_edges[0])
        occ_smooth = gaussian_filter(
            occ_time.T, sigma_bins, mode="constant", cval=0
        )

        # Session-wide mean firing rate per neuron (baseline for shrinkage)
        total_time = position.shape[0] * dt
        baseline_rates = spike_counts.sum(axis=0) / total_time  # (n_neurons,)

        # Occupancy-dependent shrinkage weight: local estimate is trusted
        # in proportion to local occupancy, reverting to baseline where
        # data is sparse.  w = occ / (occ + tau), so:
        #   rate_shrunk = w * rate_local + (1 - w) * baseline
        #              = (spike_smooth + tau * baseline) / (occ_smooth + tau)
        rate_maps = np.zeros((n_neurons, n_grid, n_grid))
        for n in range(n_neurons):
            spike_map, _, _ = np.histogram2d(
                position[:, 0], position[:, 1],
                bins=[x_bin_edges, y_bin_edges],
                weights=spike_counts[:, n],
            )
            spike_smooth = gaussian_filter(
                spike_map.T, sigma_bins, mode="constant", cval=0
            )
            rate_maps[n] = (
                (spike_smooth + occupancy_tau * baseline_rates[n])
                / (occ_smooth + occupancy_tau)
            )

        # Bin centers now span [x_min, x_max] (the full data range)
        x_centers = 0.5 * (x_bin_edges[:-1] + x_bin_edges[1:])
        y_centers = 0.5 * (y_bin_edges[:-1] + y_bin_edges[1:])

        # Store occupancy mask based on raw (unsmoothed) occupancy.
        # A bin is on-track if the animal was ever there.  We use
        # smoothed occupancy with a relative threshold to fill in
        # bins that the animal traversed but didn't dwell in (due to
        # finite bin size).  The threshold is a fraction of the
        # median non-zero smoothed occupancy, making it robust to
        # both short and long recordings.
        nonzero_occ = occ_smooth[occ_smooth > 0]
        if len(nonzero_occ) > 0:
            occ_threshold = 0.1 * np.median(nonzero_occ)
        else:
            occ_threshold = 0.0
        occupancy_mask = occ_smooth > occ_threshold

        # Store raw histograms as KDE sufficient statistics so the
        # analytical rate evaluation can compute exact kernel-sum
        # rates and gradients at arbitrary positions.
        spike_histograms = np.zeros((n_neurons, n_grid, n_grid))
        for n in range(n_neurons):
            spike_map, _, _ = np.histogram2d(
                position[:, 0], position[:, 1],
                bins=[x_bin_edges, y_bin_edges],
                weights=spike_counts[:, n],
            )
            spike_histograms[n] = spike_map.T

        return cls(
            rate_maps=rate_maps,
            x_edges=x_centers,
            y_edges=y_centers,
            occupancy_mask=occupancy_mask,
            spike_histograms=spike_histograms,
            occ_histogram=occ_time.T,
            kde_sigma=sigma,
        )

    def log_rate(self, position: Array) -> Array:
        """Evaluate log firing rate for all neurons at a position.

        When KDE sufficient statistics are available (from
        ``from_spike_position_data``), uses analytical kernel-sum
        evaluation for exact rates.  Otherwise falls back to bicubic
        grid interpolation.

        Parameters
        ----------
        position : Array, shape (2,) or (4,)
            Position [x, y] or state [x, y, vx, vy].

        Returns
        -------
        log_rate : Array, shape (n_neurons,)
        """
        if self._use_analytical:
            return _kde_log_rate(
                position,
                self._jax_grid_xy,
                self._jax_spike_hists,
                self._jax_occ_hist,
                self._jax_kde_sigma2,
            )
        return _bicubic_log_rate(
            position, self._jax_log_rate_maps,
            self._jax_x_edges, self._jax_y_edges,
            self._dx, self._dy,
        )

    def log_rate_jacobian(self, position: Array) -> Array:
        """Jacobian of log firing rate w.r.t. position via jax.jacfwd.

        When KDE sufficient statistics are available, differentiates
        through the analytical kernel-sum evaluation.

        Parameters
        ----------
        position : Array, shape (2,) or (4,)

        Returns
        -------
        jacobian : Array, shape (n_neurons, 2)
            d(log_rate_n) / d(x, y) for each neuron.
        """
        if self._use_analytical:
            return _kde_log_rate_jacobian(
                position,
                self._jax_grid_xy,
                self._jax_spike_hists,
                self._jax_occ_hist,
                self._jax_kde_sigma2,
            )
        return _bicubic_log_rate_jacobian(
            position, self._jax_log_rate_maps,
            self._jax_x_edges, self._jax_y_edges,
            self._dx, self._dy,
        )


def _kde_log_rate(
    position: Array,
    encoding_positions: Array,
    encoding_spikes: Array,
    occ_weights: Array,
    sigma2: Array,
) -> Array:
    """Evaluate log firing rate via KDE kernel sums over gridded data.

    Computes the rate for each neuron by summing Gaussian kernels
    centered at grid positions, weighted by per-bin spike counts and
    occupancy times.

    rate_n(x) = sum_g[spike_{n,g} * K(x - pos_g)] / sum_g[occ_g * K(x - pos_g)]

    Parameters
    ----------
    position : Array, shape (2,) or (4,)
    encoding_positions : Array, shape (n_bins, 2)
        Grid centre positions.
    encoding_spikes : Array, shape (n_neurons, n_bins)
        Spike counts per grid bin per neuron.
    occ_weights : Array, shape (n_bins,)
        Occupancy time (seconds) per grid bin.
    sigma2 : Array, scalar
        Squared KDE bandwidth (sigma^2, cm^2).

    Returns
    -------
    log_rate : Array, shape (n_neurons,)
    """
    xy = position[:2]
    diff = encoding_positions - xy[None, :]  # (n_bins, 2)
    dist_sq = jnp.sum(diff ** 2, axis=1)  # (n_bins,)
    kernel = jnp.exp(-0.5 * dist_sq / sigma2)  # (n_bins,)

    occ_kernel = jnp.sum(occ_weights * kernel) + 1e-30
    spike_kernels = encoding_spikes @ kernel  # (n_neurons,)

    rates = spike_kernels / occ_kernel
    return jnp.log(jnp.maximum(rates, 1e-10))


def _kde_log_rate_jacobian(
    position: Array,
    encoding_positions: Array,
    encoding_spikes: Array,
    occ_weights: Array,
    sigma2: Array,
) -> Array:
    """Jacobian of KDE log-rate via jax.jacfwd."""
    pos_xy = position[:2]

    def _log_rate_xy(xy: Array) -> Array:
        return _kde_log_rate(xy, encoding_positions, encoding_spikes, occ_weights, sigma2)

    return jax.jacfwd(_log_rate_xy)(pos_xy)


def _bilinear_log_rate(
    position: Array,
    log_rate_maps: Array,
    x_edges: Array,
    y_edges: Array,
    dx: float,
    dy: float,
) -> Array:
    """JIT-compatible bilinear interpolation of log-rate maps.

    Parameters
    ----------
    position : Array, shape (2,) or (4,)
    log_rate_maps : Array, shape (n_neurons, n_grid_y, n_grid_x)
    x_edges : Array, shape (n_grid_x,)
    y_edges : Array, shape (n_grid_y,)
    dx, dy : float
        Grid spacing.

    Returns
    -------
    log_rate : Array, shape (n_neurons,)
    """
    x, y = position[0], position[1]

    # Normalized grid coordinates, clamped to valid range
    xi = jnp.clip((x - x_edges[0]) / dx, 0.0, x_edges.shape[0] - 1 - 1e-9)
    yi = jnp.clip((y - y_edges[0]) / dy, 0.0, y_edges.shape[0] - 1 - 1e-9)

    x0 = jnp.floor(xi).astype(jnp.int32)
    y0 = jnp.floor(yi).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, log_rate_maps.shape[2] - 1)
    y1 = jnp.minimum(y0 + 1, log_rate_maps.shape[1] - 1)

    fx = xi - x0
    fy = yi - y0

    # Vectorized over neurons (axis 0)
    val = (
        log_rate_maps[:, y0, x0] * (1 - fx) * (1 - fy)
        + log_rate_maps[:, y0, x1] * fx * (1 - fy)
        + log_rate_maps[:, y1, x0] * (1 - fx) * fy
        + log_rate_maps[:, y1, x1] * fx * fy
    )
    return val


def _catmull_rom(t: Array) -> Array:
    """Catmull-Rom cubic spline basis weights for offset t in [0, 1].

    Returns weights for the 4 grid points at indices -1, 0, 1, 2
    relative to the cell containing the query point.
    """
    t2 = t * t
    t3 = t2 * t
    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2
    return jnp.array([w0, w1, w2, w3])


def _bicubic_log_rate(
    position: Array,
    log_rate_maps: Array,
    x_edges: Array,
    y_edges: Array,
    dx: float,
    dy: float,
) -> Array:
    """JIT-compatible bicubic (Catmull-Rom) interpolation of log-rate maps.

    Uses a 4x4 grid neighborhood and C1-continuous cubic spline
    weights.  The smooth interpolation preserves spatial gradients
    much better than bilinear interpolation, yielding ~25x larger
    Fisher information when differentiated with jax.jacfwd.

    Parameters
    ----------
    position : Array, shape (2,) or (4,)
    log_rate_maps : Array, shape (n_neurons, n_grid_y, n_grid_x)
    x_edges, y_edges : Array
    dx, dy : float

    Returns
    -------
    log_rate : Array, shape (n_neurons,)
    """
    x, y = position[0], position[1]
    ny, nx = log_rate_maps.shape[1], log_rate_maps.shape[2]

    xi = jnp.clip((x - x_edges[0]) / dx, 0.0, nx - 1 - 1e-9)
    yi = jnp.clip((y - y_edges[0]) / dy, 0.0, ny - 1 - 1e-9)

    ix = jnp.floor(xi).astype(jnp.int32)
    iy = jnp.floor(yi).astype(jnp.int32)
    fx = xi - ix
    fy = yi - iy

    wx = _catmull_rom(fx)  # (4,)
    wy = _catmull_rom(fy)  # (4,)

    # Gather the 4x4 neighborhood, clamping indices at boundaries
    val = jnp.zeros(log_rate_maps.shape[0])
    for j in range(4):
        jj = jnp.clip(iy + j - 1, 0, ny - 1)
        for i in range(4):
            ii = jnp.clip(ix + i - 1, 0, nx - 1)
            val = val + log_rate_maps[:, jj, ii] * wx[i] * wy[j]

    return val


def _bicubic_log_rate_jacobian(
    position: Array,
    log_rate_maps: Array,
    x_edges: Array,
    y_edges: Array,
    dx: float,
    dy: float,
) -> Array:
    """Jacobian of bicubic log-rate via jax.jacfwd."""
    pos_xy = position[:2]

    def _log_rate_xy(xy: Array) -> Array:
        return _bicubic_log_rate(xy, log_rate_maps, x_edges, y_edges, dx, dy)

    return jax.jacfwd(_log_rate_xy)(pos_xy)


def _bilinear_log_rate_jacobian(
    position: Array,
    log_rate_maps: Array,
    x_edges: Array,
    y_edges: Array,
    dx: float,
    dy: float,
) -> Array:
    """JIT-compatible Jacobian of log-rate via jax.jacfwd.

    Uses automatic differentiation through the bilinear interpolation
    for exact gradients.
    """
    pos_xy = position[:2]

    def _log_rate_xy(xy: Array) -> Array:
        return _bilinear_log_rate(xy, log_rate_maps, x_edges, y_edges, dx, dy)

    # jacfwd gives shape (n_neurons, 2) — Jacobian of vector output w.r.t. 2D input
    return jax.jacfwd(_log_rate_xy)(pos_xy)


class DecoderResult:
    """Result of position decoding.

    Attributes
    ----------
    position_mean : Array, shape (n_time, n_state)
        Decoded position (and velocity) at each time step.
        Columns: [x, y, vx, vy] if include_velocity=True, else [x, y].
        Use the ``position_xy`` property for just the (x, y) coordinates.
    position_cov : Array, shape (n_time, n_state, n_state)
        Posterior covariance at each time step.
    marginal_log_likelihood : float
        Laplace approximation to the marginal log-evidence
        sum_t log p(y_t | y_{1:t-1}), where y_t are the spike counts.
        Includes the prior term and Laplace normalization correction.
        This is a sum over time bins, so longer recordings yield larger
        values. Normalize by n_time for cross-session comparison.
    """

    def __init__(
        self,
        position_mean: Array,
        position_cov: Array,
        marginal_log_likelihood: float,
    ):
        self.position_mean = position_mean
        self.position_cov = position_cov
        self.marginal_log_likelihood = marginal_log_likelihood

    @property
    def position_xy(self) -> Array:
        """Decoded (x, y) position, shape (n_time, 2)."""
        return self.position_mean[:, :2]

    @property
    def position_cov_xy(self) -> Array:
        """Position (x, y) covariance block, shape (n_time, 2, 2)."""
        return self.position_cov[:, :2, :2]

    def __repr__(self) -> str:
        n_time = self.position_mean.shape[0]
        return (
            f"DecoderResult(n_time={n_time}, "
            f"marginal_ll={self.marginal_log_likelihood:.2f})"
        )


def _build_track_penalty(
    occupancy_mask: np.ndarray,
    dx: float,
    dy: float,
    sigma_track: float = 5.0,
) -> Array:
    """Build a distance-to-track penalty field on the rate map grid.

    Positions off the track receive a quadratic penalty that grows
    with distance from the nearest occupied bin.  This provides a
    restoring gradient pushing the decoded position back toward valid
    track locations.

    Parameters
    ----------
    occupancy_mask : np.ndarray, shape (n_grid_y, n_grid_x)
        Boolean mask of bins the animal actually visited (True = on-track).
        Should be derived from raw or smoothed occupancy, **not** from
        the shrunk rate maps (which have positive rate everywhere).
    dx, dy : float
        Grid spacing in cm.
    sigma_track : float
        Width (cm) of the soft boundary.  Positions within
        ``sigma_track`` of the track edge receive modest penalty;
        positions farther away are penalized more steeply.

    Returns
    -------
    penalty_map : Array, shape (n_grid_y, n_grid_x)
        Quadratic penalty ``0.5 * (dist / sigma_track)^2`` at each
        grid point.  Zero on-track, increasing off-track.
    """
    from scipy.ndimage import distance_transform_edt

    dist_grid = distance_transform_edt(~occupancy_mask)
    dist_cm = dist_grid * max(dx, dy)

    penalty = 0.5 * (dist_cm / sigma_track) ** 2
    return jnp.array(penalty)


def position_decoder_filter(
    spikes: ArrayLike,
    rate_maps: PlaceFieldRateMaps,
    dt: float,
    q_pos: float = 1.0,
    q_vel: float = 10.0,
    include_velocity: bool = True,
    init_position: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
    track_penalty: Optional[Array] = None,
    sigma_track: float = 5.0,
    max_newton_iter: int = 1,
    adaptive_inflation: Optional[AdaptiveInflationConfig] = None,
) -> DecoderResult:
    """Decode position from spikes using Laplace-EKF filter.

    Parameters
    ----------
    spikes : ArrayLike, shape (n_time, n_neurons)
        Spike counts per time bin per neuron.
    rate_maps : PlaceFieldRateMaps
        Pre-estimated place fields for each neuron.
    dt : float
        Time bin width in seconds.
    q_pos : float
        Position process noise (cm^2/s).
    q_vel : float
        Velocity process noise (cm^2/s^3).
    include_velocity : bool
        Whether to include velocity in the state.
    init_position : ArrayLike or None
        Initial position estimate [x, y] or [x, y, vx, vy].
        If None, uses center of arena.
    init_cov : ArrayLike or None
        Initial covariance. If None, uses large diagonal.
    track_penalty : Array or None
        Pre-built penalty map from :func:`_build_track_penalty`.
        If None, built automatically from rate_maps.
    sigma_track : float
        Width (cm) of the off-track penalty boundary.  Only used
        when track_penalty is None.

    Returns
    -------
    DecoderResult
    """
    spikes_arr = jnp.asarray(spikes)
    if spikes_arr.ndim == 1:
        spikes_arr = spikes_arr[:, None]

    if spikes_arr.shape[1] != rate_maps.n_neurons:
        raise ValueError(
            f"spikes has {spikes_arr.shape[1]} columns but rate_maps "
            f"has {rate_maps.n_neurons} neurons"
        )

    A, Q = build_position_dynamics(dt, q_pos, q_vel, include_velocity)
    n_state = A.shape[0]

    if init_position is None:
        cx = float(rate_maps.x_edges.mean())
        cy = float(rate_maps.y_edges.mean())
        if include_velocity:
            init_position = jnp.array([cx, cy, 0.0, 0.0])
        else:
            init_position = jnp.array([cx, cy])

    if init_cov is None:
        init_cov = jnp.eye(n_state) * 100.0

    # Capture JAX arrays from rate_maps for JIT-compatible closures
    jax_log_rate_maps = rate_maps._jax_log_rate_maps
    jax_x_edges = rate_maps._jax_x_edges
    jax_y_edges = rate_maps._jax_y_edges
    grid_dx = rate_maps._dx
    grid_dy = rate_maps._dy
    n_neurons = rate_maps.n_neurons

    # Build distance-to-track penalty map
    if track_penalty is None:
        if rate_maps.occupancy_mask is not None:
            track_penalty = _build_track_penalty(
                rate_maps.occupancy_mask,
                rate_maps._dx, rate_maps._dy,
                sigma_track=sigma_track,
            )
        else:
            track_penalty = jnp.zeros(
                (len(rate_maps.y_edges), len(rate_maps.x_edges))
            )
    track_penalty = jnp.asarray(track_penalty)
    # Store as (1, ny, nx) for bilinear interpolation reuse
    jax_penalty_map = track_penalty[None, :, :]

    # Precompute constants used in every scan step
    _vel_pad = jnp.zeros((n_neurons, 2)) if include_velocity else None

    # Use analytical KDE evaluation when sufficient statistics are
    # available; fall back to bicubic grid interpolation otherwise.
    _use_kde = rate_maps._use_analytical
    if _use_kde:
        _grid_xy = rate_maps._jax_grid_xy
        _spike_hists = rate_maps._jax_spike_hists
        _occ_hist = rate_maps._jax_occ_hist
        _sigma2 = rate_maps._jax_kde_sigma2

    def log_intensity_func(state):
        if _use_kde:
            return _kde_log_rate(state, _grid_xy, _spike_hists, _occ_hist, _sigma2)
        return _bicubic_log_rate(
            state, jax_log_rate_maps, jax_x_edges, jax_y_edges, grid_dx, grid_dy
        )

    def grad_log_intensity_func(state):
        if _use_kde:
            jac_pos = _kde_log_rate_jacobian(
                state, _grid_xy, _spike_hists, _occ_hist, _sigma2
            )
        else:
            jac_pos = _bicubic_log_rate_jacobian(
                state, jax_log_rate_maps, jax_x_edges, jax_y_edges, grid_dx, grid_dy
            )
        if include_velocity:
            return jnp.concatenate([jac_pos, _vel_pad], axis=1)
        return jac_pos

    def _penalty_at(pos_xy):
        """Scalar track-distance penalty at (x, y)."""
        return _bilinear_log_rate(
            pos_xy, jax_penalty_map, jax_x_edges, jax_y_edges, grid_dx, grid_dy
        )[0]

    _penalty_grad_fn = jax.grad(_penalty_at)

    # Adaptive inflation configuration — capture as JAX-compatible
    # constants so they can be used inside the traced _step function.
    _inflate = adaptive_inflation is not None and adaptive_inflation.enabled
    if _inflate:
        assert adaptive_inflation is not None  # narrow for mypy
        _infl_gain = jnp.array(adaptive_inflation.gain)
        _infl_max = jnp.array(adaptive_inflation.max_alpha)
        _infl_eps = jnp.array(adaptive_inflation.epsilon)
        _infl_min_ft = jnp.array(adaptive_inflation.min_fisher_trace)
        # Normalise by the number of observed position dimensions (2),
        # not n_state.  The Fisher matrix has rank 2 regardless of
        # whether velocity is in the state, so the score statistic
        # has E[s_t]=1 only when divided by 2.
        _infl_d = jnp.array(2.0)

    # Run filter with jax.lax.scan
    def _step(carry, spike_t):
        mean_prev, cov_prev, total_ll = carry

        # Prediction
        one_step_mean = A @ mean_prev
        one_step_cov = A @ cov_prev @ A.T + Q
        one_step_cov = symmetrize(one_step_cov)

        # Incorporate distance-to-track prior into the prediction.
        # The penalty is a log-prior: log p_track(x) = -penalty(x).
        # We shift the predicted mean by a natural gradient step:
        #   new_mean = old_mean - cov @ grad_penalty
        # This pulls the prediction toward the track before the
        # Laplace observation update.
        pen_grad_xy = _penalty_grad_fn(one_step_mean[:2])
        if include_velocity:
            pen_grad = jnp.concatenate([pen_grad_xy, jnp.zeros(2)])
        else:
            pen_grad = pen_grad_xy
        one_step_mean = one_step_mean - one_step_cov @ pen_grad

        # Adaptive covariance inflation: if the Poisson score at the
        # predicted state is larger than the Fisher information predicts,
        # the filter is overconfident and we inflate the predicted
        # covariance before the Laplace update.
        if _inflate:
            log_lambda = log_intensity_func(one_step_mean)
            cond_int = _safe_expected_count(log_lambda, dt)
            jacobian = grad_log_intensity_func(one_step_mean)
            innovation = spike_t - cond_int
            score = jacobian.T @ innovation  # (n_state,)
            fisher = jacobian.T @ (cond_int[:, None] * jacobian)  # (n_state, n_state)
            fisher_trace = jnp.trace(fisher)
            # Standardised score statistic: s = g^T F^{-1} g / d
            fisher_reg = fisher + _infl_eps * jnp.eye(n_state)
            s_t = (score @ jnp.linalg.solve(fisher_reg, score)) / _infl_d
            # Inflate only when Fisher is non-trivial and score is large
            alpha_t = jnp.where(
                fisher_trace > _infl_min_ft,
                jnp.clip(1.0 + _infl_gain * (s_t - 1.0), 1.0, _infl_max),
                1.0,
            )
            one_step_cov = one_step_cov * alpha_t

        # Laplace update (spike observation model)
        post_mean, post_cov, ll = _point_process_laplace_update(
            one_step_mean,
            one_step_cov,
            spike_t,
            dt,
            log_intensity_func,
            grad_log_intensity_func=grad_log_intensity_func,
            include_laplace_normalization=True,
            max_newton_iter=max_newton_iter,
        )

        total_ll = total_ll + ll
        return (post_mean, post_cov, total_ll), (post_mean, post_cov)

    init_carry = (jnp.asarray(init_position), jnp.asarray(init_cov), jnp.array(0.0))
    (_, _, marginal_ll), (filtered_mean, filtered_cov) = jax.lax.scan(
        _step, init_carry, spikes_arr,
    )

    return DecoderResult(
        position_mean=filtered_mean,
        position_cov=filtered_cov,
        marginal_log_likelihood=float(marginal_ll),
    )


def position_decoder_smoother(
    spikes: ArrayLike,
    rate_maps: PlaceFieldRateMaps,
    dt: float,
    q_pos: float = 1.0,
    q_vel: float = 10.0,
    include_velocity: bool = True,
    init_position: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
    track_penalty: Optional[Array] = None,
    sigma_track: float = 5.0,
    max_newton_iter: int = 1,
    adaptive_inflation: Optional[AdaptiveInflationConfig] = None,
) -> DecoderResult:
    """Decode position from spikes using Laplace-EKF + RTS smoother.

    Runs the forward filter, then applies the Rauch-Tung-Striebel
    backward smoother to produce non-causal position estimates that
    use the entire spike train. Smoothed estimates have lower variance
    than filtered estimates.

    Parameters are the same as :func:`position_decoder_filter`.

    Returns
    -------
    DecoderResult
        Smoothed position estimates with the same marginal_log_likelihood
        as the forward filter (the smoother does not change it).
    """
    filter_result = position_decoder_filter(
        spikes, rate_maps, dt, q_pos, q_vel,
        include_velocity, init_position, init_cov,
        track_penalty=track_penalty, sigma_track=sigma_track,
        max_newton_iter=max_newton_iter,
        adaptive_inflation=adaptive_inflation,
    )

    A, Q = build_position_dynamics(dt, q_pos, q_vel, include_velocity)

    # RTS backward smoother
    def _smooth_step(carry, inputs):
        next_sm_mean, next_sm_cov = carry
        filt_mean, filt_cov = inputs

        sm_mean, sm_cov, _ = _kalman_smoother_update(
            next_sm_mean, next_sm_cov,
            filt_mean, filt_cov,
            Q, A,
        )
        return (sm_mean, sm_cov), (sm_mean, sm_cov)

    _, (sm_mean, sm_cov) = jax.lax.scan(
        _smooth_step,
        (filter_result.position_mean[-1], filter_result.position_cov[-1]),
        (filter_result.position_mean[:-1], filter_result.position_cov[:-1]),
        reverse=True,
    )

    smoother_mean = jnp.concatenate([sm_mean, filter_result.position_mean[-1:]])
    smoother_cov = jnp.concatenate([sm_cov, filter_result.position_cov[-1:]])

    return DecoderResult(
        position_mean=smoother_mean,
        position_cov=smoother_cov,
        marginal_log_likelihood=filter_result.marginal_log_likelihood,
    )


class PositionDecoder:
    """Position decoder from population spike trains.

    Two-step workflow:
    1. ``fit(position, spikes)``: estimate place field rate maps from
       training data (position + spikes during behavior).
    2. ``decode(spikes)``: decode position from spike trains alone,
       using the fitted rate maps.

    Parameters
    ----------
    dt : float
        Time bin width in seconds.
    q_pos : float, default=1.0
        Position process noise (cm^2/s).
    q_vel : float, default=10.0
        Velocity process noise (cm^2/s^3).
    include_velocity : bool, default=True
        Include velocity in the state.
    n_grid : int, default=50
        Grid resolution for rate map estimation.
    smoothing_sigma : float, default=5.0
        Gaussian smoothing kernel width (cm) for rate map estimation.

    Examples
    --------
    >>> decoder = PositionDecoder(dt=0.004)
    >>> decoder.fit(train_position, train_spikes)
    >>> result = decoder.decode(test_spikes)
    >>> decoded_xy = result.position_mean[:, :2]
    """

    def __init__(
        self,
        dt: float,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        include_velocity: bool = True,
        n_grid: int = 50,
        smoothing_sigma: float = 5.0,
        occupancy_tau: float = 1.0,
        max_newton_iter: int = 1,
        adaptive_inflation: Optional[AdaptiveInflationConfig] = None,
    ):
        self.dt = dt
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.include_velocity = include_velocity
        self.n_grid = n_grid
        self.smoothing_sigma = smoothing_sigma
        self.occupancy_tau = occupancy_tau
        self.max_newton_iter = max_newton_iter
        self.adaptive_inflation = adaptive_inflation

        self.rate_maps: Optional[PlaceFieldRateMaps] = None

    def __repr__(self) -> str:
        fitted = self.rate_maps is not None
        n_neurons = self.rate_maps.n_neurons if self.rate_maps is not None else "?"
        return (
            f"PositionDecoder(dt={self.dt}, n_neurons={n_neurons}, "
            f"fitted={fitted})"
        )

    def fit(
        self,
        position: np.ndarray,
        spikes: np.ndarray,
    ) -> None:
        """Estimate place field rate maps from training data via KDE.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
            Animal position in cm at each time bin.
        spikes : np.ndarray, shape (n_time,) or (n_time, n_neurons)
            Spike counts per time bin (and per neuron).
        """
        self.rate_maps = PlaceFieldRateMaps.from_spike_position_data(
            position=position,
            spike_counts=spikes,
            dt=self.dt,
            n_grid=self.n_grid,
            sigma=self.smoothing_sigma,
            occupancy_tau=self.occupancy_tau,
        )
        logger.info(
            "Fitted rate maps for %d neurons on %dx%d grid",
            self.rate_maps.n_neurons, self.n_grid, self.n_grid,
        )

    def fit_from_model(
        self,
        model,
        n_grid: Optional[int] = None,
        time_slice: Optional[slice] = None,
    ) -> None:
        """Use rate maps from a fitted PlaceFieldModel.

        Parameters
        ----------
        model : PlaceFieldModel
            Fitted encoding model.
        n_grid : int or None
            Grid resolution. If None, uses self.n_grid.
        time_slice : slice or None
            Time bin indices to average over. None = full session.
        """
        self.rate_maps = PlaceFieldRateMaps.from_place_field_model(
            model, n_grid=n_grid or self.n_grid, time_slice=time_slice,
        )
        logger.info(
            "Loaded rate maps for %d neurons on %dx%d grid from PlaceFieldModel",
            self.rate_maps.n_neurons,
            self.rate_maps.rate_maps.shape[1],
            self.rate_maps.rate_maps.shape[2],
        )

    def decode(
        self,
        spikes: ArrayLike,
        method: str = "smoother",
        init_position: Optional[ArrayLike] = None,
        init_cov: Optional[ArrayLike] = None,
    ) -> DecoderResult:
        """Decode position from spike trains.

        Parameters
        ----------
        spikes : ArrayLike, shape (n_time, n_neurons)
        method : str, "filter" or "smoother"
            "filter" for causal (online) decoding.
            "smoother" for non-causal (offline) decoding.
        init_position : ArrayLike or None
            Initial position estimate.
        init_cov : ArrayLike or None
            Initial covariance. If None, uses large diagonal.

        Returns
        -------
        DecoderResult
        """
        if self.rate_maps is None:
            raise RuntimeError(
                "PositionDecoder.decode() called before fitting. "
                "Call decoder.fit(position, spikes) or "
                "decoder.fit_from_model(model) first."
            )

        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]

        if spikes_arr.shape[1] != self.rate_maps.n_neurons:
            raise ValueError(
                f"spikes has {spikes_arr.shape[1]} neurons but rate maps "
                f"were estimated from {self.rate_maps.n_neurons} neurons. "
                f"Ensure test spikes use the same neuron ordering as "
                f"training data"
            )

        if method == "smoother":
            decode_func = position_decoder_smoother
        elif method == "filter":
            decode_func = position_decoder_filter
        else:
            raise ValueError(f"method must be 'filter' or 'smoother', got '{method}'")

        return decode_func(
            spikes=spikes_arr,
            rate_maps=self.rate_maps,
            dt=self.dt,
            q_pos=self.q_pos,
            q_vel=self.q_vel,
            include_velocity=self.include_velocity,
            init_position=init_position,
            init_cov=init_cov,
            max_newton_iter=self.max_newton_iter,
            adaptive_inflation=self.adaptive_inflation,
        )

    def plot_decoding(
        self,
        result: DecoderResult,
        true_position: Optional[np.ndarray] = None,
        ax=None,
    ):
        """Plot decoded vs true position trajectory.

        When called without ``ax``, creates a two-panel figure:
        left panel shows the 2D trajectory, right panel shows
        position error and uncertainty over time.

        Parameters
        ----------
        result : DecoderResult
        true_position : np.ndarray or None, shape (n_time, 2)
        ax : array of Axes or None
            If provided, must be an array of 2 Axes for both panels,
            or a single Axes for trajectory-only mode.

        Returns
        -------
        fig : Figure
        """
        import matplotlib.pyplot as plt

        decoded = np.array(result.position_xy)

        if ax is None:
            n_panels = 2 if true_position is not None else 1
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
            axes = np.atleast_1d(axes)
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        # Left: 2D trajectory
        if true_position is not None:
            axes[0].plot(
                true_position[:, 0], true_position[:, 1],
                "k-", alpha=0.3, label="True",
            )
        axes[0].plot(
            decoded[:, 0], decoded[:, 1],
            "r-", alpha=0.7, label="Decoded",
        )
        axes[0].set_xlabel("x (cm)")
        axes[0].set_ylabel("y (cm)")
        axes[0].set_title("Decoded Trajectory")
        axes[0].legend()
        axes[0].set_aspect("equal")

        # Right: error over time (if true position available)
        if true_position is not None and len(axes) > 1:
            error = np.linalg.norm(decoded - true_position, axis=1)
            t = np.arange(len(error)) * self.dt
            axes[1].plot(t, error, "k-", alpha=0.7)
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Position error (cm)")
            axes[1].set_title(f"Median error: {np.median(error):.1f} cm")
            pos_var = np.array(result.position_cov[:, :2, :2])
            # 1-sigma position uncertainty radius
            sigma_radius = np.sqrt(pos_var[:, 0, 0] + pos_var[:, 1, 1])
            axes[1].fill_between(
                t, 0, sigma_radius, alpha=0.2, color="blue",
                label=r"1$\sigma$ radius",
            )
            axes[1].legend()

        fig.tight_layout()
        return fig
