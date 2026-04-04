"""Place field tracking model with point-process observations.

This module provides a model class for tracking time-varying 2D place fields
using the Laplace-EKF point-process filter/smoother with EM parameter estimation.

Model:
    x_t = A x_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t ~ Poisson(exp(Z_t @ x_t) * dt)

where:
    x_t ∈ R^{n_basis}: time-varying GLM weights on a 2D spatial spline basis
    Z_t ∈ R^{n_basis}: tensor-product B-spline basis evaluated at the animal's
        2D position at time t
    A: transition matrix (default: identity for random walk)
    Q: process noise covariance (diagonal, learned via EM)

The design matrix Z is built from 2D tensor-product B-splines (via patsy),
with knots placed at quantiles of the observed position data to concentrate
resolution where the animal spends time.

References
----------
[1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
    Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
    Neural Computation 16, 971-998.
"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from patsy import dmatrix

from state_space_practice.kalman import sum_of_outer_products, symmetrize
from state_space_practice.point_process_kalman import (
    get_confidence_interval,
    log_conditional_intensity,
    stochastic_point_process_filter,
    stochastic_point_process_smoother,
)
from state_space_practice.utils import check_converged

logger = logging.getLogger(__name__)


def build_2d_spline_basis(
    position: np.ndarray,
    n_interior_knots: int = 5,
    knots_x: Optional[np.ndarray] = None,
    knots_y: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Build a 2D tensor-product B-spline design matrix from position data.

    Constructs the design matrix by taking the tensor product of 1D B-spline
    bases for x and y coordinates. Knots are placed at quantiles of the
    position data by default, concentrating basis functions where the animal
    spends time.

    Parameters
    ----------
    position : np.ndarray, shape (n_time, 2)
        Animal position (x, y) at each time bin.
    n_interior_knots : int, default=5
        Number of interior knots per dimension. Total basis functions will be
        (n_interior_knots + 3)^2 for cubic B-splines. Ignored if knots_x
        and knots_y are provided.
    knots_x : np.ndarray or None, optional
        Explicit interior knot locations for x dimension.
    knots_y : np.ndarray or None, optional
        Explicit interior knot locations for y dimension.

    Returns
    -------
    design_matrix : np.ndarray, shape (n_time, n_basis)
        Evaluated 2D spline basis at each position.
    basis_info : dict
        Dictionary with keys needed to evaluate the basis at new positions:
        - "knots_x": interior knot locations for x
        - "knots_y": interior knot locations for y
        - "x_lo", "x_hi": x bounds
        - "y_lo", "y_hi": y bounds
        - "formula": patsy formula string
        - "n_basis": number of basis functions
        - "n_interior_knots": number of interior knots per dimension
    """
    position = np.asarray(position)
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError(
            f"position must be (n_time, 2), got shape {position.shape}"
        )

    x, y = position[:, 0], position[:, 1]

    if knots_x is None:
        knots_x = np.quantile(x, np.linspace(0.05, 0.95, n_interior_knots))
    if knots_y is None:
        knots_y = np.quantile(y, np.linspace(0.05, 0.95, n_interior_knots))

    x_lo, x_hi = float(x.min()), float(x.max())
    y_lo, y_hi = float(y.min()), float(y.max())

    formula = (
        "te(bs(x, knots=knots_x, lower_bound=x_lo, upper_bound=x_hi), "
        "   bs(y, knots=knots_y, lower_bound=y_lo, upper_bound=y_hi)) - 1"
    )
    env = {
        "knots_x": knots_x,
        "knots_y": knots_y,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
    }

    design_matrix = np.asarray(dmatrix(formula, {"x": x, "y": y, **env}))

    basis_info = {
        **env,
        "formula": formula,
        "n_basis": design_matrix.shape[1],
        "n_interior_knots": n_interior_knots,
    }

    return design_matrix, basis_info


def evaluate_basis(
    position: np.ndarray,
    basis_info: dict,
) -> np.ndarray:
    """Evaluate a previously constructed 2D spline basis at new positions.

    Positions outside the training data bounds are clipped to the boundary,
    since patsy B-splines do not support extrapolation.

    Parameters
    ----------
    position : np.ndarray, shape (n_points, 2)
        Positions (x, y) at which to evaluate the basis.
    basis_info : dict
        Basis specification from ``build_2d_spline_basis``.

    Returns
    -------
    design_matrix : np.ndarray, shape (n_points, n_basis)
    """
    position = np.asarray(position)
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError(
            f"position must be (n_points, 2), got shape {position.shape}"
        )
    x = np.clip(position[:, 0], basis_info["x_lo"], basis_info["x_hi"])
    y = np.clip(position[:, 1], basis_info["y_lo"], basis_info["y_hi"])
    env = {
        k: basis_info[k]
        for k in ("knots_x", "knots_y", "x_lo", "x_hi", "y_lo", "y_hi")
    }
    return np.asarray(dmatrix(basis_info["formula"], {"x": x, "y": y, **env}))


class PlaceFieldModel:
    """Point-process model for tracking time-varying 2D place fields.

    The latent state is a vector of GLM weights on a 2D spatial spline basis.
    The weights evolve via linear-Gaussian dynamics (default: random walk),
    and spikes are Poisson-distributed with log-linear intensity:

        log(lambda_t) = Z_t @ x_t

    where Z_t is the spline basis evaluated at the animal's position at time t.

    Parameters are estimated via EM: the E-step uses a Laplace-EKF
    filter/smoother, and the M-step updates the process noise covariance Q
    and initial conditions.

    Parameters
    ----------
    dt : float
        Time bin width in seconds.
    n_interior_knots : int, default=5
        Number of interior spline knots per spatial dimension. Determines
        spatial resolution. Total basis functions = (n_interior_knots + 3)^2.
    process_noise_structure : str, default="diagonal"
        Structure of the process noise covariance Q:
        - "diagonal": independent variance per basis function (learned via EM)
        - "isotropic": single scalar variance shared across all basis functions
    init_process_noise : float, default=1e-5
        Initial value for the diagonal of Q.
    update_transition_matrix : bool, default=False
        Whether to learn A. Default False (keeps A = I for random walk).
    update_process_cov : bool, default=True
        Whether to learn Q via EM.
    update_init_state : bool, default=True
        Whether to learn initial mean and covariance via EM.

    Attributes
    ----------
    smoother_mean : Array, shape (n_time, n_basis)
        Smoothed weight estimates after fitting.
    smoother_cov : Array, shape (n_time, n_basis, n_basis)
        Smoothed weight covariances after fitting.
    basis_info : dict
        Spline basis specification (knots, bounds, formula).
    log_likelihoods : list[float]
        Log-likelihood history from fitting.

    Examples
    --------
    >>> model = PlaceFieldModel(dt=0.004, n_interior_knots=5)
    >>> model.fit(position, spikes)
    >>> rate_map = model.predict_rate_map(grid_positions, time_slice=slice(0, 1000))
    """

    def __init__(
        self,
        dt: float,
        n_interior_knots: int = 5,
        process_noise_structure: str = "diagonal",
        init_process_noise: float = 1e-5,
        update_transition_matrix: bool = False,
        update_process_cov: bool = True,
        update_init_state: bool = True,
    ):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if n_interior_knots < 1:
            raise ValueError(
                f"n_interior_knots must be >= 1, got {n_interior_knots}"
            )
        if process_noise_structure not in ("diagonal", "isotropic"):
            raise ValueError(
                f"process_noise_structure must be 'diagonal' or 'isotropic', "
                f"got '{process_noise_structure}'"
            )

        self.dt = dt
        self.n_interior_knots = n_interior_knots
        self.process_noise_structure = process_noise_structure
        self.init_process_noise = init_process_noise
        self.update_transition_matrix = update_transition_matrix
        self.update_process_cov = update_process_cov
        self.update_init_state = update_init_state

        # Populated during fit
        self.basis_info: Optional[dict] = None
        self.n_basis: Optional[int] = None
        self.transition_matrix: Optional[Array] = None
        self.process_cov: Optional[Array] = None
        self.init_mean: Optional[Array] = None
        self.init_cov: Optional[Array] = None
        self.smoother_mean: Optional[Array] = None
        self.smoother_cov: Optional[Array] = None
        self.smoother_cross_cov: Optional[Array] = None
        self.log_likelihoods: list[float] = []

    def __repr__(self) -> str:
        fitted = self.smoother_mean is not None
        parts = [
            f"dt={self.dt}",
            f"n_interior_knots={self.n_interior_knots}",
            f"Q_structure={self.process_noise_structure}",
            f"fitted={fitted}",
        ]
        if fitted and self.process_cov is not None:
            q_mean = float(jnp.diag(self.process_cov).mean())
            parts.append(f"Q_diag_mean={q_mean:.2e}")
        if fitted and self.n_basis is not None:
            parts.append(f"n_basis={self.n_basis}")
        return f"<PlaceFieldModel: {', '.join(parts)}>"

    @classmethod
    def from_place_field_width(
        cls,
        dt: float,
        place_field_width: float,
        arena_range_x: tuple[float, float],
        arena_range_y: tuple[float, float],
        **kwargs,
    ) -> "PlaceFieldModel":
        """Create a model with knot spacing matched to place field size.

        Sets the number of interior knots so that knot spacing is approximately
        ``place_field_width / 3``, giving ~3 basis functions per field width.
        This ensures the spline basis can represent typical place fields
        without over- or under-fitting the spatial resolution.

        Typical place field widths (diameter at half-max):
        - CA1 open field: 30-40 cm
        - CA1 linear track: 20-30 cm
        - CA3: 40-60 cm

        Parameters
        ----------
        dt : float
            Time bin width in seconds.
        place_field_width : float
            Expected place field diameter in cm (full width at half-max).
        arena_range_x : tuple (x_min, x_max)
            Spatial extent of the arena in x (cm).
        arena_range_y : tuple (y_min, y_max)
            Spatial extent of the arena in y (cm).
        **kwargs
            Additional keyword arguments passed to ``PlaceFieldModel.__init__``
            (e.g., ``process_noise_structure``, ``init_process_noise``).

        Returns
        -------
        model : PlaceFieldModel

        Examples
        --------
        >>> # CA1 place cells in a 100x100 cm arena
        >>> model = PlaceFieldModel.from_place_field_width(
        ...     dt=0.004, place_field_width=30.0,
        ...     arena_range_x=(0, 100), arena_range_y=(0, 100),
        ... )
        >>> model.n_interior_knots
        10

        >>> # Plus maze spanning 30-230 cm
        >>> model = PlaceFieldModel.from_place_field_width(
        ...     dt=0.004, place_field_width=35.0,
        ...     arena_range_x=(30, 230), arena_range_y=(33, 223),
        ... )
        >>> model.n_interior_knots
        17
        """
        knot_spacing = place_field_width / 3.0
        extent_x = arena_range_x[1] - arena_range_x[0]
        extent_y = arena_range_y[1] - arena_range_y[0]
        extent = max(extent_x, extent_y)
        n_interior_knots = max(3, int(np.round(extent / knot_spacing)))

        return cls(dt=dt, n_interior_knots=n_interior_knots, **kwargs)

    def _build_design_matrix(
        self,
        position: np.ndarray,
        knots_x: Optional[np.ndarray] = None,
        knots_y: Optional[np.ndarray] = None,
    ) -> Array:
        """Build the spline design matrix from position data."""
        design_matrix, self.basis_info = build_2d_spline_basis(
            position,
            n_interior_knots=self.n_interior_knots,
            knots_x=knots_x,
            knots_y=knots_y,
        )
        self.n_basis = self.basis_info["n_basis"]
        return jnp.asarray(design_matrix)

    def _initialize_parameters(self) -> None:
        """Initialize model parameters."""
        n = self.n_basis
        self.transition_matrix = jnp.eye(n)
        self.process_cov = jnp.eye(n) * self.init_process_noise
        self.init_mean = jnp.zeros(n)
        self.init_cov = jnp.eye(n) * 1.0

    def _e_step(
        self, design_matrix: Array, spikes: Array
    ) -> float:
        """E-step: run filter and smoother."""
        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_ll,
        ) = stochastic_point_process_smoother(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spikes,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=log_conditional_intensity,
        )
        return float(marginal_ll)

    def _m_step(self) -> None:
        """M-step: update dynamics parameters from smoothed estimates.

        When A is fixed to identity (random walk), Q is computed directly as
        the expected variance of state increments under the posterior, rather
        than using the unconstrained ML formula from kalman_maximization_step.
        This avoids a small bias from the A-Q coupling in the general formula.

        The diagonal/isotropic constraint on Q is then applied, discarding
        off-diagonal covariance structure to keep the model tractable.
        """
        sm = self.smoother_mean
        sc = self.smoother_cov
        scc = self.smoother_cross_cov
        n_time = sm.shape[0]

        # Sufficient statistics: E[x_t x_t'], E[x_{t-1} x_t'], E[x_{t-1} x_{t-1}']
        gamma = jnp.sum(sc, axis=0) + sum_of_outer_products(sm, sm)
        gamma1 = gamma - jnp.outer(sm[-1], sm[-1]) - sc[-1]
        gamma2 = gamma - jnp.outer(sm[0], sm[0]) - sc[0]
        beta = (
            scc.sum(axis=0) + sum_of_outer_products(sm[:-1], sm[1:])
        ).T

        if self.update_transition_matrix:
            from state_space_practice.kalman import psd_solve

            A_new = psd_solve(gamma1, beta.T).T
            self.transition_matrix = A_new
            Q_new = (gamma2 - A_new @ beta.T) / (n_time - 1)
        else:
            # A = I: Q = E[(x_t - x_{t-1})(x_t - x_{t-1})']
            #       = (gamma2 - beta.T - beta + gamma1) / (T-1)
            Q_new = (gamma2 - beta.T - beta + gamma1) / (n_time - 1)

        Q_new = symmetrize(Q_new)

        if self.update_process_cov:
            if self.process_noise_structure == "diagonal":
                q_diag = jnp.maximum(jnp.diag(Q_new), 1e-10)
                self.process_cov = jnp.diag(q_diag)
            else:  # isotropic
                q_diag = jnp.maximum(jnp.diag(Q_new), 1e-10)
                self.process_cov = jnp.eye(self.n_basis) * jnp.mean(q_diag)

        if self.update_init_state:
            self.init_mean = sm[0]
            self.init_cov = symmetrize(sc[0])

    @staticmethod
    def bin_spike_times(
        spike_times: np.ndarray,
        time_bins: np.ndarray,
    ) -> np.ndarray:
        """Bin spike times into time bins.

        Convenience method for converting spike time arrays (as returned by
        most recording systems) into the binned spike counts expected by
        ``fit()``.

        Parameters
        ----------
        spike_times : np.ndarray, shape (n_spikes,)
            Times of individual spikes (in seconds or matching time_bins units).
        time_bins : np.ndarray, shape (n_time,)
            Left edges of time bins (e.g., from ``np.arange(t_start, t_end, dt)``).

        Returns
        -------
        spike_counts : np.ndarray, shape (n_time,)
            Number of spikes in each time bin.

        Examples
        --------
        >>> time_bins = np.arange(0, 600, 0.004)  # 600s at 4ms bins
        >>> spike_counts = PlaceFieldModel.bin_spike_times(spike_times, time_bins)
        >>> model.fit(position, spike_counts)
        """
        spike_counts = np.zeros(len(time_bins), dtype=int)
        bin_indices = np.searchsorted(time_bins, spike_times) - 1
        valid = (bin_indices >= 0) & (bin_indices < len(time_bins))
        np.add.at(spike_counts, bin_indices[valid], 1)
        return spike_counts

    def fit(
        self,
        position: np.ndarray,
        spikes: ArrayLike,
        max_iter: int = 10,
        tolerance: float = 1e-4,
        knots_x: Optional[np.ndarray] = None,
        knots_y: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> list[float]:
        """Fit the model to spike data and position using EM.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
            Animal position (x, y) at each time bin.
        spikes : ArrayLike, shape (n_time,)
            Spike counts per time bin. Use ``PlaceFieldModel.bin_spike_times``
            to convert spike time arrays to binned counts.
        max_iter : int, default=10
            Maximum number of EM iterations.
        tolerance : float, default=1e-4
            Convergence tolerance for relative log-likelihood change.
        knots_x : np.ndarray or None, optional
            Explicit interior knot locations for x dimension.
            If None, placed at quantiles of position data.
        knots_y : np.ndarray or None, optional
            Explicit interior knot locations for y dimension.
        verbose : bool, default=True
            Print progress to stdout during fitting.

        Returns
        -------
        log_likelihoods : list[float]
            Marginal log-likelihood at each EM iteration.
        """
        # Validate inputs
        position = np.asarray(position)
        spikes = jnp.asarray(spikes)
        if spikes.ndim != 1:
            raise ValueError(
                f"spikes must be 1D with shape (n_time,), got shape {spikes.shape}. "
                f"Multi-neuron input is not supported; fit one neuron at a time."
            )
        if position.shape[0] != spikes.shape[0]:
            raise ValueError(
                f"position and spikes must have the same number of time bins: "
                f"got position ({position.shape[0]},) vs spikes ({spikes.shape[0]},)"
            )

        # Build basis and initialize
        design_matrix = self._build_design_matrix(position, knots_x, knots_y)
        self._initialize_parameters()

        def _print(msg: str) -> None:
            if verbose:
                print(msg)

        _print(
            f"PlaceFieldModel: n_time={len(spikes)}, n_basis={self.n_basis}, "
            f"total_spikes={int(spikes.sum())}"
        )

        self.log_likelihoods = []

        for iteration in range(max_iter):
            ll = self._e_step(design_matrix, spikes)
            self.log_likelihoods.append(ll)

            _print(f"  EM iter {iteration + 1}/{max_iter}: LL = {ll:.1f}")

            if not jnp.isfinite(ll):
                _print(f"  WARNING: Non-finite LL at iteration {iteration + 1}")
                break

            if iteration > 0:
                is_converged, is_increasing = check_converged(
                    ll, self.log_likelihoods[-2], tolerance
                )
                if not is_increasing:
                    _print(
                        f"  WARNING: LL decreased: "
                        f"{self.log_likelihoods[-2]:.1f} -> {ll:.1f}"
                    )
                if is_converged and is_increasing:
                    _print(f"  Converged after {iteration + 1} iterations.")
                    break

            self._m_step()

        return self.log_likelihoods

    def predict_rate_map(
        self,
        grid_positions: np.ndarray,
        time_slice: Optional[slice] = None,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the firing rate map at given spatial positions.

        Evaluates the estimated place field at a grid of positions,
        averaging the smoothed weights over the specified time window.

        The confidence interval reflects the average per-time-step posterior
        uncertainty over the window, not the uncertainty of the time-averaged
        weight. This gives a representative sense of how uncertain the rate
        estimate is at each spatial location during the specified period.

        Parameters
        ----------
        grid_positions : np.ndarray, shape (n_grid_points, 2)
            Spatial positions (x, y) at which to evaluate the rate.
        time_slice : slice or None, optional
            Time window over which to average the smoothed weights.
            If None, uses the full session.
        alpha : float, default=0.05
            Significance level for confidence interval (0.05 for 95% CI).

        Returns
        -------
        rate : np.ndarray, shape (n_grid_points,)
            Estimated firing rate (Hz) at each grid position.
        rate_ci : np.ndarray, shape (n_grid_points, 2)
            Lower and upper confidence bounds on the rate (Hz).
        """
        if self.smoother_mean is None or self.basis_info is None:
            raise RuntimeError("Model has not been fitted yet.")

        Z_grid = evaluate_basis(grid_positions, self.basis_info)

        if time_slice is None:
            time_slice = slice(None)

        weights = np.array(self.smoother_mean[time_slice].mean(axis=0))
        cov = np.array(self.smoother_cov[time_slice].mean(axis=0))

        log_rate = Z_grid @ weights
        var_log_rate = np.sum(Z_grid @ cov * Z_grid, axis=1)
        std_log_rate = np.sqrt(np.maximum(var_log_rate, 0))

        z = float(jax.scipy.stats.norm.ppf(1 - alpha / 2))
        rate = np.exp(log_rate)
        rate_lo = np.exp(log_rate - z * std_log_rate)
        rate_hi = np.exp(log_rate + z * std_log_rate)

        return rate, np.column_stack([rate_lo, rate_hi])

    def predict_center(
        self,
        grid_positions: np.ndarray,
        n_blocks: int = 20,
    ) -> np.ndarray:
        """Estimate the place field center over time.

        Computes the weighted centroid of the estimated rate map in temporal
        blocks for sub-grid-resolution center estimates.

        Parameters
        ----------
        grid_positions : np.ndarray, shape (n_grid_points, 2)
            Spatial grid for evaluating the rate map.
        n_blocks : int, default=20
            Number of temporal blocks. Time steps are split as evenly as
            possible using ``np.array_split``, so no data is dropped.

        Returns
        -------
        centers : np.ndarray, shape (n_blocks, 2)
            Estimated place field center (x, y) in each block.
        """
        if self.smoother_mean is None or self.basis_info is None:
            raise RuntimeError("Model has not been fitted yet.")

        Z_grid = evaluate_basis(grid_positions, self.basis_info)
        n_time = self.smoother_mean.shape[0]
        if n_blocks < 1 or n_blocks > n_time:
            raise ValueError(
                f"n_blocks must be between 1 and n_time ({n_time}), got {n_blocks}"
            )

        block_indices = np.array_split(np.arange(n_time), n_blocks)
        centers = np.zeros((n_blocks, 2))
        for i, idx in enumerate(block_indices):
            weights = np.array(self.smoother_mean[idx].mean(axis=0))
            rate = np.exp(Z_grid @ weights)
            # Weighted centroid (more stable than argmax)
            rate_sum = rate.sum()
            if rate_sum > 0:
                centers[i] = (rate[:, None] * grid_positions).sum(axis=0) / rate_sum
            else:
                centers[i] = grid_positions.mean(axis=0)

        return centers

    def make_grid(self, n_grid: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a regular spatial grid within the training data bounds.

        Convenience method for constructing evaluation grids for
        ``predict_rate_map`` and ``predict_center``.

        Parameters
        ----------
        n_grid : int, default=50
            Number of grid points per dimension. Total points = n_grid^2.

        Returns
        -------
        grid_positions : np.ndarray, shape (n_grid^2, 2)
            Grid positions (x, y) as a flat array of coordinates.
        x_edges : np.ndarray, shape (n_grid,)
            X coordinates of the grid.
        y_edges : np.ndarray, shape (n_grid,)
            Y coordinates of the grid.

        Examples
        --------
        >>> grid, x_edges, y_edges = model.make_grid(n_grid=50)
        >>> rate, ci = model.predict_rate_map(grid)
        >>> plt.pcolormesh(x_edges, y_edges, rate.reshape(len(y_edges), len(x_edges)))
        """
        if self.basis_info is None:
            raise RuntimeError("Model has not been fitted yet.")
        x = np.linspace(self.basis_info["x_lo"], self.basis_info["x_hi"], n_grid)
        y = np.linspace(self.basis_info["y_lo"], self.basis_info["y_hi"], n_grid)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()]), x, y

    def score(
        self,
        position: np.ndarray,
        spikes: ArrayLike,
    ) -> float:
        """Compute the marginal log-likelihood on held-out data.

        Runs the filter (no smoothing or parameter updates) with the
        fitted model parameters on new position/spike data.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
            Animal position (x, y) at each time bin.
        spikes : ArrayLike, shape (n_time,)
            Spike counts per time bin.

        Returns
        -------
        log_likelihood : float
            Marginal log-likelihood of the held-out data.
        """
        if self.basis_info is None or self.init_mean is None:
            raise RuntimeError("Model has not been fitted yet.")

        design_matrix = jnp.asarray(
            evaluate_basis(np.asarray(position), self.basis_info)
        )
        spikes = jnp.asarray(spikes)

        _, _, marginal_ll = stochastic_point_process_filter(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spikes,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=log_conditional_intensity,
        )
        return float(marginal_ll)

    def get_state_confidence_interval(
        self, alpha: float = 0.05
    ) -> Array:
        """Get confidence intervals for the latent weight estimates.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI).

        Returns
        -------
        ci : Array, shape (n_time, n_basis, 2)
            Lower and upper bounds for each weight at each time step.
        """
        if self.smoother_mean is None or self.smoother_cov is None:
            raise RuntimeError("Model has not been fitted yet.")
        return get_confidence_interval(
            self.smoother_mean, self.smoother_cov, alpha=alpha
        )

    # ------------------------------------------------------------------
    # Drift analysis
    # ------------------------------------------------------------------

    def drift_summary(
        self,
        n_grid: int = 80,
        n_blocks: int = 20,
    ) -> dict:
        """Summarize place field drift over the session.

        Parameters
        ----------
        n_grid : int, default=80
            Grid resolution for center estimation.
        n_blocks : int, default=20
            Number of temporal blocks for center trajectory.

        Returns
        -------
        summary : dict with keys:
            centers : (n_blocks, 2) — place field center per block
            total_drift : float — Euclidean distance from first to last center (cm)
            cumulative_drift : float — total path length of center trajectory (cm)
            mean_rate_per_block : (n_blocks,) — peak rate in each block
            block_times : (n_blocks,) — center time of each block (in time steps)
        """
        if self.smoother_mean is None:
            raise RuntimeError("Model has not been fitted yet.")

        grid, _, _ = self.make_grid(n_grid)
        Z_grid = evaluate_basis(grid, self.basis_info)
        n_time = self.smoother_mean.shape[0]

        block_indices = np.array_split(np.arange(n_time), n_blocks)
        centers = np.zeros((n_blocks, 2))
        peak_rates = np.zeros(n_blocks)
        block_times = np.zeros(n_blocks)

        for i, idx in enumerate(block_indices):
            weights = np.array(self.smoother_mean[idx].mean(axis=0))
            rate = np.exp(Z_grid @ weights)
            rate_sum = rate.sum()
            if rate_sum > 0:
                centers[i] = (rate[:, None] * grid).sum(axis=0) / rate_sum
            else:
                centers[i] = grid.mean(axis=0)
            peak_rates[i] = rate.max()
            block_times[i] = idx.mean()

        displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)

        return {
            "centers": centers,
            "total_drift": float(np.linalg.norm(centers[-1] - centers[0])),
            "cumulative_drift": float(displacements.sum()),
            "mean_rate_per_block": peak_rates,
            "block_times": block_times * self.dt,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_rate_maps(
        self,
        n_time_bins: int = 3,
        n_grid: int = 50,
        ax: Optional[np.ndarray] = None,
    ):
        """Plot estimated rate maps in temporal bins.

        Parameters
        ----------
        n_time_bins : int, default=3
            Number of temporal bins (e.g., 3 for early/middle/late).
        n_grid : int, default=50
            Spatial grid resolution per dimension.
        ax : array of Axes or None
            Matplotlib axes to plot into. If None, creates a new figure.

        Returns
        -------
        fig : Figure
            The matplotlib figure.
        """
        import matplotlib.pyplot as plt

        if self.smoother_mean is None:
            raise RuntimeError("Model has not been fitted yet.")

        grid, x_edges, y_edges = self.make_grid(n_grid)
        n_time = self.smoother_mean.shape[0]
        block_indices = np.array_split(np.arange(n_time), n_time_bins)
        labels = {
            1: ["Full session"],
            2: ["First half", "Second half"],
            3: ["Early", "Middle", "Late"],
        }.get(n_time_bins, [f"Block {i+1}" for i in range(n_time_bins)])

        # Compute all rate maps first to get shared color scale
        rate_maps = []
        for idx in block_indices:
            rate, _ = self.predict_rate_map(grid, time_slice=slice(idx[0], idx[-1] + 1))
            rate_maps.append(rate.reshape(n_grid, n_grid))

        vmax = max(r.max() for r in rate_maps)

        if ax is None:
            fig, axes = plt.subplots(1, n_time_bins, figsize=(5 * n_time_bins, 4))
            if n_time_bins == 1:
                axes = [axes]
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        for i, (rate_map, label) in enumerate(zip(rate_maps, labels)):
            im = axes[i].pcolormesh(
                x_edges, y_edges, rate_map, cmap="hot", vmin=0, vmax=vmax
            )
            axes[i].set_title(label)
            axes[i].set_aspect("equal")
            axes[i].set_xlabel("x (cm)")
            if i == 0:
                axes[i].set_ylabel("y (cm)")
            plt.colorbar(im, ax=axes[i], label="Rate (Hz)")

        fig.tight_layout()
        return fig

    def plot_drift(
        self,
        n_blocks: int = 20,
        n_grid: int = 80,
        ax=None,
    ):
        """Plot the place field center trajectory over time.

        Colors the trajectory from blue (start) to red (end).

        Parameters
        ----------
        n_blocks : int, default=20
            Number of temporal blocks for center estimation.
        n_grid : int, default=80
            Spatial grid resolution for center estimation.
        ax : Axes or None
            Matplotlib axes to plot into. If None, creates a new figure.

        Returns
        -------
        fig : Figure
            The matplotlib figure.
        """
        import matplotlib.pyplot as plt

        summary = self.drift_summary(n_grid=n_grid, n_blocks=n_blocks)
        centers = summary["centers"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        colors = plt.cm.coolwarm(np.linspace(0, 1, n_blocks))
        for i in range(n_blocks - 1):
            ax.plot(
                centers[i : i + 2, 0],
                centers[i : i + 2, 1],
                "-o",
                color=colors[i],
                markersize=4,
            )
        ax.plot(centers[0, 0], centers[0, 1], "ko", markersize=10, label="Start")
        ax.plot(centers[-1, 0], centers[-1, 1], "k^", markersize=10, label="End")

        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(
            f"Place Field Center Drift\n"
            f"Total: {summary['total_drift']:.1f} cm, "
            f"Path: {summary['cumulative_drift']:.1f} cm"
        )
        ax.legend()
        ax.set_aspect("equal")

        fig.tight_layout()
        return fig
