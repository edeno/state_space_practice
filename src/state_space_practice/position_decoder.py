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
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import _kalman_smoother_update, symmetrize
from state_space_practice.point_process_kalman import (
    _point_process_laplace_update,
)

logger = logging.getLogger(__name__)


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
    ):
        self.rate_maps = np.asarray(rate_maps)
        self.x_edges = np.asarray(x_edges)
        self.y_edges = np.asarray(y_edges)

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

        # Grid spacing for finite differences
        self._dx = float(x_diffs[0])
        self._dy = float(y_diffs[0])

        # JAX arrays for JIT-compatible interpolation
        self._jax_log_rate_maps = jnp.array(self._log_rate_maps)
        self._jax_x_edges = jnp.array(self.x_edges)
        self._jax_y_edges = jnp.array(self.y_edges)

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
    ) -> PlaceFieldRateMaps:
        """Estimate rate maps from position and spike data using KDE.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
        spike_counts : np.ndarray, shape (n_time,) or (n_time, n_neurons)
        dt : float
        n_grid : int
        sigma : float
            Gaussian smoothing kernel width in cm.

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

        # Use n_grid+1 bin edges so histogram2d produces n_grid bins,
        # making n_grid mean "number of spatial bins" consistently
        # with from_place_field_model.
        x_bin_edges = np.linspace(x_min, x_max, n_grid + 1)
        y_bin_edges = np.linspace(y_min, y_max, n_grid + 1)

        # Compute occupancy
        occ, _, _ = np.histogram2d(
            position[:, 0], position[:, 1],
            bins=[x_bin_edges, y_bin_edges],
        )
        occ_time = occ * dt
        # Smooth occupancy
        sigma_bins = sigma / (x_bin_edges[1] - x_bin_edges[0])
        occ_smooth = gaussian_filter(occ_time.T, sigma_bins)

        rate_maps = np.zeros((n_neurons, n_grid, n_grid))
        for n in range(n_neurons):
            spike_map, _, _ = np.histogram2d(
                position[:, 0], position[:, 1],
                bins=[x_bin_edges, y_bin_edges],
                weights=spike_counts[:, n],
            )
            spike_smooth = gaussian_filter(spike_map.T, sigma_bins)
            rate_maps[n] = np.where(
                occ_smooth > dt * 5, spike_smooth / occ_smooth, 0
            )

        # Use bin centers as grid coordinates
        x_centers = 0.5 * (x_bin_edges[:-1] + x_bin_edges[1:])
        y_centers = 0.5 * (y_bin_edges[:-1] + y_bin_edges[1:])

        return cls(rate_maps=rate_maps, x_edges=x_centers, y_edges=y_centers)

    def log_rate(self, position: Array) -> Array:
        """Evaluate log firing rate for all neurons at a position.

        JIT-compatible bilinear interpolation on the pre-computed grid.

        Parameters
        ----------
        position : Array, shape (2,) or (4,)
            Position [x, y] or state [x, y, vx, vy].

        Returns
        -------
        log_rate : Array, shape (n_neurons,)
        """
        return _bilinear_log_rate(
            position, self._jax_log_rate_maps,
            self._jax_x_edges, self._jax_y_edges,
            self._dx, self._dy,
        )

    def log_rate_jacobian(self, position: Array) -> Array:
        """Jacobian of log firing rate w.r.t. position via finite differences.

        JIT-compatible. Uses half-grid-spacing central differences.

        Parameters
        ----------
        position : Array, shape (2,) or (4,)

        Returns
        -------
        jacobian : Array, shape (n_neurons, 2)
            d(log_rate_n) / d(x, y) for each neuron.
        """
        return _bilinear_log_rate_jacobian(
            position, self._jax_log_rate_maps,
            self._jax_x_edges, self._jax_y_edges,
            self._dx, self._dy,
        )


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


def _bilinear_log_rate_jacobian(
    position: Array,
    log_rate_maps: Array,
    x_edges: Array,
    y_edges: Array,
    dx: float,
    dy: float,
) -> Array:
    """JIT-compatible Jacobian of log-rate via central finite differences."""
    eps_x = dx * 0.5
    eps_y = dy * 0.5

    # Use .at[].add() to perturb position without allocating new arrays
    pos_xy = position[:2]
    eps_vec_x = jnp.array([eps_x, 0.0])
    eps_vec_y = jnp.array([0.0, eps_y])

    rate_xp = _bilinear_log_rate(
        pos_xy + eps_vec_x, log_rate_maps, x_edges, y_edges, dx, dy
    )
    rate_xm = _bilinear_log_rate(
        pos_xy - eps_vec_x, log_rate_maps, x_edges, y_edges, dx, dy
    )
    rate_yp = _bilinear_log_rate(
        pos_xy + eps_vec_y, log_rate_maps, x_edges, y_edges, dx, dy
    )
    rate_ym = _bilinear_log_rate(
        pos_xy - eps_vec_y, log_rate_maps, x_edges, y_edges, dx, dy
    )

    dfdx = (rate_xp - rate_xm) / (2 * eps_x)
    dfdy = (rate_yp - rate_ym) / (2 * eps_y)

    return jnp.column_stack([dfdx, dfdy])


class DecoderResult:
    """Result of position decoding.

    Attributes
    ----------
    position_mean : Array, shape (n_time, n_state)
        Decoded position (and velocity) at each time step.
        Columns: [x, y, vx, vy] or [x, y].
    position_cov : Array, shape (n_time, n_state, n_state)
        Posterior covariance at each time step.
    marginal_log_likelihood : float
        Laplace approximation to the marginal log-evidence
        sum_t log p(y_t | y_{1:t-1}), where y_t are the spike counts.
        Includes the prior term and Laplace normalization correction.
        Suitable for model comparison and hyperparameter selection.
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

    def __repr__(self) -> str:
        n_time = self.position_mean.shape[0]
        return (
            f"DecoderResult(n_time={n_time}, "
            f"marginal_ll={self.marginal_log_likelihood:.2f})"
        )


def position_decoder_filter(
    spikes: ArrayLike,
    rate_maps: PlaceFieldRateMaps,
    dt: float,
    q_pos: float = 1.0,
    q_vel: float = 10.0,
    include_velocity: bool = True,
    init_position: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
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

    # Precompute constants used in every scan step
    _vel_pad = jnp.zeros((n_neurons, 2)) if include_velocity else None
    _zero_hess = jnp.zeros((n_neurons, n_state, n_state))

    def log_intensity_func(state):
        return _bilinear_log_rate(
            state, jax_log_rate_maps, jax_x_edges, jax_y_edges, grid_dx, grid_dy
        )

    def grad_log_intensity_func(state):
        jac_pos = _bilinear_log_rate_jacobian(
            state, jax_log_rate_maps, jax_x_edges, jax_y_edges, grid_dx, grid_dy
        )
        if include_velocity:
            return jnp.concatenate([jac_pos, _vel_pad], axis=1)
        return jac_pos

    def hess_log_intensity_func(state):
        return _zero_hess

    # Run filter with jax.lax.scan
    def _step(carry, spike_t):
        mean_prev, cov_prev, total_ll = carry

        # Prediction
        one_step_mean = A @ mean_prev
        one_step_cov = A @ cov_prev @ A.T + Q
        one_step_cov = symmetrize(one_step_cov)

        # Laplace update
        post_mean, post_cov, ll = _point_process_laplace_update(
            one_step_mean,
            one_step_cov,
            spike_t,
            dt,
            log_intensity_func,
            grad_log_intensity_func=grad_log_intensity_func,
            hess_log_intensity_func=hess_log_intensity_func,
            include_laplace_normalization=True,
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
) -> DecoderResult:
    """Decode position from spikes using Laplace-EKF + RTS smoother.

    Same parameters as ``position_decoder_filter``. Returns smoothed
    (non-causal) estimates that use the full spike train.

    Returns
    -------
    DecoderResult with smoothed position estimates.
    """
    filter_result = position_decoder_filter(
        spikes, rate_maps, dt, q_pos, q_vel,
        include_velocity, init_position, init_cov,
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
    ):
        self.dt = dt
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.include_velocity = include_velocity
        self.n_grid = n_grid
        self.smoothing_sigma = smoothing_sigma

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
        """Estimate place field rate maps from training data.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
        spikes : np.ndarray, shape (n_time,) or (n_time, n_neurons)
        """
        self.rate_maps = PlaceFieldRateMaps.from_spike_position_data(
            position=position,
            spike_counts=spikes,
            dt=self.dt,
            n_grid=self.n_grid,
            sigma=self.smoothing_sigma,
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
            raise RuntimeError("Must call fit() before decode().")

        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]

        if spikes_arr.shape[1] != self.rate_maps.n_neurons:
            raise ValueError(
                f"Expected {self.rate_maps.n_neurons} spike columns, "
                f"got {spikes_arr.shape[1]}"
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
        )

    def plot_decoding(
        self,
        result: DecoderResult,
        true_position: Optional[np.ndarray] = None,
        ax=None,
    ):
        """Plot decoded vs true position trajectory.

        Parameters
        ----------
        result : DecoderResult
        true_position : np.ndarray or None, shape (n_time, 2)
        ax : Axes or None

        Returns
        -------
        fig : Figure
        """
        import matplotlib.pyplot as plt

        decoded = np.array(result.position_mean[:, :2])

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
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
