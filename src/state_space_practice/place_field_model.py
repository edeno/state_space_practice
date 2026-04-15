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
[2] Brown, E.N., Frank, L.M., Tang, D., Quirk, M.C. & Wilson, M.A. (1998).
    A statistical paradigm for neural spike train decoding applied to position
    prediction from ensemble firing patterns of rat hippocampal place cells.
    J Neuroscience 18(18), 7411-7425.
[3] Ziv, Y., Burns, L.D., Cocker, E.D., Hamel, E.O., Ghosh, K.K.,
    Kitch, L.J., El Gamal, A. & Bhatt, M.J. (2013). Long-term dynamics
    of CA1 hippocampal place codes. Nature Neuroscience 16, 264-266.
"""

import logging
import warnings
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from patsy import dmatrix

from state_space_practice.kalman import psd_solve, sum_of_outer_products, symmetrize
from state_space_practice.point_process_kalman import (
    _detect_block_diagonal_problem,
    _safe_expected_count,
    _validate_filter_numerics,
    get_confidence_interval,
    log_conditional_intensity,
    stochastic_point_process_filter,
    stochastic_point_process_smoother,
)
from state_space_practice.sgd_fitting import SGDFittableMixin
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


class PlaceFieldModel(SGDFittableMixin):
    """Point-process model for tracking time-varying 2D place fields.

    The latent state is a vector of GLM weights on a 2D spatial spline basis.
    The weights evolve via linear-Gaussian dynamics (default: random walk),
    and spikes are Poisson-distributed with log-linear intensity:

        log(lambda_t) = Z_t @ x_t

    where Z_t is the spline basis evaluated at the animal's position at time t.

    Supports multiple neurons sharing a common spatial basis: each neuron
    has its own set of weights, concatenated into a single state vector.
    Pass spikes as ``(n_time, n_neurons)`` to enable multi-neuron mode.

    Parameters are estimated via EM: the E-step uses a Laplace-EKF
    filter/smoother, and the M-step updates the process noise covariance Q
    and initial conditions.

    Numerical precision
    -------------------
    The Laplace-EKF filter's covariance propagation requires **float64**
    for reliable long-sequence fits. In float32, accumulated roundoff can
    drive the posterior covariance below PSD after ~250-5000 time bins
    depending on ``init_cov`` conditioning, causing silent NaN. Enable
    float64 BEFORE importing this module::

        import jax
        jax.config.update("jax_enable_x64", True)
        from state_space_practice import PlaceFieldModel

    ``fit`` and ``fit_sgd`` validate ``init_cov`` at entry and raise if
    it is not PSD, or warn if the configuration is at risk of f32 NaN.

    Block-diagonal filter dispatch
    ------------------------------
    For multi-neuron fits (``n_neurons > 1``), ``fit`` and ``fit_sgd``
    automatically detect when the filter problem has block-diagonal
    structure (block-diagonal ``A``, ``Q``, ``init_cov``, shared
    ``Z_base`` across neurons) and dispatch to a specialized
    block-diagonal filter that runs per-neuron scans via ``jax.vmap``.
    On a problem with ``n_neurons=64`` and ``n_basis_per_neuron=36``,
    the per-step Cholesky cost drops by ~``n_neurons^2``.

    The dispatch is controlled by a ``force_dense`` keyword arg on
    both ``fit`` and ``fit_sgd``; the default (``False``) uses the
    auto-detected block path when applicable. Single-neuron fits never
    dispatch (no structural advantage).

    Numerical note: forward-pass equivalence between the dense and
    block paths is bit-identical (``atol=1e-9`` in the regression
    tests). Gradient equivalence is bit-identical for ``init_mean``,
    ``process_cov``, and ``transition_matrix``, but NOT for
    ``init_cov`` — the dense path's autodiff produces spurious
    non-zero gradients on off-block entries of ``init_cov`` that the
    block path never computes (because the block filter only reads
    diagonal blocks). As a result, multi-step ``fit_sgd`` trajectories
    can drift subtly between paths; the block path is the correct
    gradient for the block-diagonal parameterization, while the dense
    path has extra off-block noise that EM's M-step projects away.
    For ``fit`` (EM), the two paths agree to ``atol=1e-5``; for
    ``fit_sgd``, step-0 LL is bit-identical and subsequent steps may
    diverge by ~``1e-3`` in LL over tens of iterations without
    meaningful algorithmic difference.

    Parameters
    ----------
    dt : float
        Time bin width in seconds.
    n_interior_knots : int, default=5
        Number of interior spline knots per spatial dimension. Determines
        spatial resolution. Total basis functions = (n_interior_knots + 3)^2.
    process_noise_structure : str, default="diagonal"
        Structure of the process noise covariance Q:
        - "diagonal": independent variance per basis function (learned via EM).
          Use for most analyses; allows each spatial basis function to drift
          at its own rate.
        - "isotropic": single scalar variance shared across all basis functions.
          Faster to converge but assumes uniform drift across all spatial
          basis functions, which is rarely appropriate for place cells.
    init_process_noise : float, default=1e-6
        Initial value for the diagonal of Q. Chosen so that the cumulative
        random-walk drift over a typical session is small in log-rate space:
        ``sqrt(Q * T) ≈ 0.1`` nats for T=10,000 bins, i.e. the log-rate
        weights wander by ~0.1 nats (standard deviation) over the session.
        This is consistent with the small within-session place-field drift
        reported for CA1 in Ziv 2013 / Mankin 2012 — a few centimeters in
        arena space, which maps to a small log-rate perturbation on the
        spline-weight vector. EM (``fit``) updates Q from data, so this
        default only controls the initial scale; ``fit_sgd`` optimizes it
        via gradient descent.
    init_cov_scale : float, default=0.01
        Scale for the initial state covariance (``init_cov = I * init_cov_scale``)
        when ``warm_start=False``. Controls how uncertain the initial
        weight estimates are. Only used when warm-start is disabled; the
        default warm-start path replaces this with the Laplace posterior
        covariance of a stationary Poisson GLM, which is data-driven and
        far tighter than any scalar default. The scalar ``0.01``
        corresponds to prior std ~0.1 per weight — a reasonable
        "weakly informative" fallback when the warm-start is skipped.
    max_firing_rate_hz : float, default=500.0
        Physiologically motivated ceiling on the instantaneous firing rate.
        Used to set ``max_log_count = log(max_firing_rate_hz * dt)`` inside
        the Laplace-EKF filter's safe-exponentiation clip. Pathological bins
        (e.g. artifact counts, mis-binned spike floods) that would otherwise
        drive the Fisher step into a catastrophic log-likelihood region are
        clipped to this ceiling; the user is warned at the end of the fit
        if more than 0.1%% of bin/neuron combinations saturate, which usually
        indicates a data-quality issue upstream. The default of 500 Hz is
        conservative for CA1 pyramidal cells (<100 Hz peak) and generous
        enough not to clip fast-spiking interneurons under normal conditions.
    log_intensity_func : callable or None, default=None
        Custom log-intensity function with signature
        ``(design_matrix, params) -> log_rate``. If None, uses the default
        linear function ``log(lambda) = Z @ x``. For nonlinear functions,
        ``predict_rate_map`` uses the time-averaged weights which is an
        approximation; consider using ``smoother_mean`` directly.
    update_transition_matrix : bool, default=False
        Whether to learn A. Default False (keeps A = I for random walk).
    update_process_cov : bool, default=True
        Whether to learn Q via EM.
    update_init_state : bool, default=True
        Whether to learn initial mean and covariance via EM.

    Attributes (set after fit)
    --------------------------
    smoother_mean : Array, shape (n_time, n_state)
        Smoothed weight estimates. For multi-neuron, n_state = n_neurons * n_basis.
    smoother_cov : Array, shape (n_time, n_state, n_state)
        Smoothed weight covariances.
    filtered_mean : Array, shape (n_time, n_state)
        Filtered (causal) weight estimates.
    filtered_cov : Array, shape (n_time, n_state, n_state)
        Filtered weight covariances.
    basis_info : dict
        Spline basis specification (knots, bounds, formula).
    log_likelihoods : list[float]
        Log-likelihood history from fitting.
    n_neurons : int
        Number of neurons (detected from spikes during fit).

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
        init_process_noise: float = 1e-6,
        init_cov_scale: float = 0.01,
        log_intensity_func: Optional[Callable[[ArrayLike, ArrayLike], Array]] = None,
        update_transition_matrix: bool = False,
        update_process_cov: bool = True,
        update_init_state: bool = True,
        max_firing_rate_hz: float = 500.0,
        max_newton_iter: int = 1,
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
        if max_firing_rate_hz <= 0:
            raise ValueError(
                f"max_firing_rate_hz must be positive, got {max_firing_rate_hz}"
            )

        self.dt = dt
        self.n_interior_knots = n_interior_knots
        self.process_noise_structure = process_noise_structure
        self.init_process_noise = init_process_noise
        self.init_cov_scale = init_cov_scale
        self.max_firing_rate_hz = max_firing_rate_hz
        self.max_newton_iter = max_newton_iter
        self._log_intensity_func = (
            log_intensity_func if log_intensity_func is not None
            else log_conditional_intensity
        )
        self.update_transition_matrix = update_transition_matrix
        self.update_process_cov = update_process_cov
        self.update_init_state = update_init_state

        # Populated during fit
        self.basis_info: Optional[dict] = None
        self.n_basis_per_neuron: Optional[int] = None
        self.n_basis: Optional[int] = None  # total state dim
        self.n_neurons: int = 1
        self.transition_matrix: Optional[Array] = None
        self.process_cov: Optional[Array] = None
        self.init_mean: Optional[Array] = None
        self.init_cov: Optional[Array] = None
        self.smoother_mean: Optional[Array] = None
        self.smoother_cov: Optional[Array] = None
        self.smoother_cross_cov: Optional[Array] = None
        # Block-diagonal dispatch: populated at fit/fit_sgd entry via
        # _detect_block_structure. If both are ints, the filter/smoother
        # dispatch to the block-diagonal fast path. None means dense.
        self._block_n_neurons: Optional[int] = None
        self._block_size: Optional[int] = None
        self.filtered_mean: Optional[Array] = None
        self.filtered_cov: Optional[Array] = None
        self.log_likelihoods: list[float] = []
        self._total_spikes: int = 0
        self._n_time: int = 0

    def __repr__(self) -> str:
        fitted = self.smoother_mean is not None
        parts = [
            f"dt={self.dt}",
            f"n_interior_knots={self.n_interior_knots}",
            f"process_noise_structure={self.process_noise_structure}",
            f"fitted={fitted}",
        ]
        if fitted and self.process_cov is not None:
            q_mean = float(jnp.diag(self.process_cov).mean())
            parts.append(f"Q_diag_mean={q_mean:.2e}")
        if fitted and self.n_basis_per_neuron is not None:
            parts.append(f"n_basis={self.n_basis_per_neuron}")
        if fitted and self.n_neurons > 1:
            parts.append(f"n_neurons={self.n_neurons}")
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
        # Use average extent so non-square arenas don't over-parameterize
        # the short axis (same n_interior_knots is used for both dimensions).
        extent = (extent_x + extent_y) / 2.0
        n_interior_knots = max(3, int(np.round(extent / knot_spacing)))

        return cls(dt=dt, n_interior_knots=n_interior_knots, **kwargs)

    def _neuron_weights(
        self, neuron_idx: int = 0
    ) -> tuple[slice, int]:
        """Return the slice into the state vector for a given neuron."""
        if neuron_idx < 0 or neuron_idx >= self.n_neurons:
            raise ValueError(
                f"neuron_idx={neuron_idx} out of range for "
                f"n_neurons={self.n_neurons}"
            )
        assert self.n_basis_per_neuron is not None, "Model not initialized"
        nb = self.n_basis_per_neuron
        start = neuron_idx * nb
        return slice(start, start + nb), nb

    def _build_block_diagonal(self, Z_base: np.ndarray) -> Array:
        """Build block-diagonal design matrix for multi-neuron models.

        Parameters
        ----------
        Z_base : np.ndarray, shape (n_time, n_basis_per_neuron)

        Returns
        -------
        Array, shape (n_time, n_neurons, n_basis)
        """
        assert self.n_basis_per_neuron is not None, "Model not initialized"
        assert self.n_basis is not None, "Model not initialized"
        nb = self.n_basis_per_neuron
        n_time = Z_base.shape[0]
        Z_base_jnp = jnp.asarray(Z_base)
        # Z_full[t, j, :] = kron(e_j, Z_base[t]) — non-zero only at j*nb:(j+1)*nb
        eye_n = jnp.eye(self.n_neurons)  # (n_neurons, n_neurons)
        # (1, n_neurons, n_neurons, 1) * (n_time, 1, 1, nb) -> (n_time, n_neurons, n_neurons, nb)
        Z_full = (eye_n[None, :, :, None] * Z_base_jnp[:, None, None, :]).reshape(
            n_time, self.n_neurons, self.n_neurons * nb
        )
        return Z_full

    def _check_fitted(self, method_name: str) -> None:
        """Raise if the model has not been fitted."""
        if self.smoother_mean is None:
            raise RuntimeError(
                f"Model has not been fitted. "
                f"Call model.fit(position, spikes) before {method_name}()."
            )
        # These are always set before smoother_mean; assert for type narrowing
        assert self.basis_info is not None
        assert self.n_basis is not None
        assert self.n_basis_per_neuron is not None
        assert self.transition_matrix is not None
        assert self.process_cov is not None
        assert self.init_mean is not None
        assert self.init_cov is not None
        assert self.smoother_cov is not None

    def _build_spline_basis_matrix(
        self,
        position: np.ndarray,
        knots_x: Optional[np.ndarray] = None,
        knots_y: Optional[np.ndarray] = None,
    ) -> Array:
        """Build the base 2D spline basis matrix and cache basis metadata.

        Returns ``Z_base`` of shape ``(n_time, n_basis_per_neuron)`` before
        any multi-neuron block-diagonalization. Callers who need the full
        (possibly block-diagonal) design matrix should pass the result
        through ``_expand_to_block_diagonal``.

        This is factored out from ``_build_design_matrix`` so the warm-start
        helper (``_fit_stationary_glm``) can fit a per-neuron Poisson GLM
        on the shared base basis without reconstructing the block-diagonal
        form.
        """
        Z_base, self.basis_info = build_2d_spline_basis(
            position,
            n_interior_knots=self.n_interior_knots,
            knots_x=knots_x,
            knots_y=knots_y,
        )
        self.n_basis_per_neuron = self.basis_info["n_basis"]
        return jnp.asarray(Z_base)

    def _expand_to_block_diagonal(self, Z_base: Array) -> Array:
        """Expand a per-neuron spline basis into the full design matrix.

        For single-neuron, returns ``Z_base`` unchanged (shape
        ``(n_time, n_basis)``). For multi-neuron, block-diagonalizes into
        ``(n_time, n_neurons, n_neurons * n_basis)`` so each neuron's
        log-intensity row selects its own weight slice.

        Also sets ``self.n_basis``.
        """
        assert self.n_basis_per_neuron is not None
        if self.n_neurons == 1:
            self.n_basis = self.n_basis_per_neuron
            return Z_base

        self.n_basis = self.n_neurons * self.n_basis_per_neuron
        return self._build_block_diagonal(np.asarray(Z_base))

    def _build_design_matrix(
        self,
        position: np.ndarray,
        knots_x: Optional[np.ndarray] = None,
        knots_y: Optional[np.ndarray] = None,
    ) -> Array:
        """Build the spline design matrix from position data.

        For single-neuron: returns shape (n_time, n_basis).
        For multi-neuron: returns shape (n_time, n_neurons, n_neurons * n_basis)
        as a block-diagonal matrix where each neuron selects its own weight slice.
        """
        Z_base = self._build_spline_basis_matrix(position, knots_x, knots_y)
        return self._expand_to_block_diagonal(Z_base)

    def _initialize_parameters(self) -> None:
        """Initialize model parameters with scalar defaults (no warm-start)."""
        assert self.n_basis is not None, "Must call _build_design_matrix first"
        n = self.n_basis
        self.transition_matrix = jnp.eye(n)
        self.process_cov = jnp.eye(n) * self.init_process_noise
        self.init_mean = jnp.zeros(n)
        self.init_cov = jnp.eye(n) * self.init_cov_scale

    def _fit_stationary_glm(
        self,
        Z_base: Array,
        spikes: Array,
        window: Optional[slice] = None,
        max_iter: int = 15,
        prior_precision: float = 1.0,
    ) -> tuple[Array, Array]:
        """Fit a stationary Poisson GLM per neuron via Newton's method.

        Used by the warm-start path to set ``init_mean`` and ``init_cov`` to
        the MAP estimate and Laplace covariance of the stationary (no-drift)
        Poisson GLM. This gives the Kalman filter a dramatically better
        starting point than zero weights with an identity prior — on real
        CA1 data with 36-basis spline, the difference is ~7 orders of
        magnitude in first-step marginal log-likelihood.

        Model (per neuron j):
            log λ_t^j = Z_base[t] @ w^j
            y_t^j ~ Poisson(λ_t^j * dt)
            w^j ~ N(0, prior_precision^{-1} * I)    [weak prior]

        Fit: Newton-Raphson on the MAP NLL
            -LL(w) = -sum_t [y_t log(μ_t) - μ_t] + 0.5 * prior_precision * ||w||^2

        with gradient ``Z^T (μ - y) + prior_precision * w`` and Hessian
        ``Z^T diag(μ) Z + prior_precision * I``. The Hessian is PSD by
        construction (Fisher information of Poisson GLM plus identity),
        so Newton steps converge in ~10 iterations for well-conditioned
        problems.

        The returned ``init_cov`` is the inverse Hessian at the MAP —
        the Laplace approximation to the posterior covariance of the
        stationary GLM. For multi-neuron models, returns a block-diagonal
        covariance matching the block-diagonal design matrix.

        Parameters
        ----------
        Z_base : Array, shape (n_time, n_basis_per_neuron)
            Shared spline basis across neurons.
        spikes : Array, shape (n_time,) or (n_time, n_neurons)
            Binned spike counts.
        window : slice or None, default=None
            Time window to fit on. If None, uses the whole series. See the
            ``fit`` / ``fit_sgd`` ``warm_start_window`` parameter for the
            usual recommendation (whole dataset is typically best for
            statistical efficiency and spatial coverage).
        max_iter : int, default=15
            Newton iterations. The intercept-matching initial guess
            (see below) makes Newton converge to machine precision in
            ~8 iterations on well-conditioned problems, so 15 is
            comfortable insurance for poorly-conditioned data.
        prior_precision : float, default=1.0
            Weak Gaussian prior on the weights, added to the Hessian for
            numerical stability and to regularize basis functions whose
            support the animal never visited. With ``prior_precision=1.0``
            the Laplace covariance ``(Z' diag(mu) Z + I)^{-1}`` is bounded
            above by ``I`` (the old ``init_cov_scale=1.0`` default), and
            is strictly tighter wherever the data constrains the weights
            — so the warm-started ``init_cov`` is never looser than the
            old scalar default but becomes much tighter in well-sampled
            directions. Raise this to regularize more aggressively on
            very sparse spike data.

        Returns
        -------
        init_mean : Array, shape (n_basis,)
            MAP weight estimate, single neuron (shape ``(n_basis_per_neuron,)``)
            or concatenated across neurons (shape ``(n_neurons * nb,)``).
        init_cov : Array, shape (n_basis, n_basis)
            Laplace posterior covariance. Block-diagonal for multi-neuron.
        """
        assert self.n_basis_per_neuron is not None
        nb = self.n_basis_per_neuron
        Z_base = jnp.asarray(Z_base)
        spikes = jnp.asarray(spikes)

        if window is not None:
            Z_base = Z_base[window]
            spikes = spikes[window]

        single_neuron = spikes.ndim == 1
        spikes_2d = spikes[:, None] if single_neuron else spikes
        n_neurons = spikes_2d.shape[1]
        n_time = Z_base.shape[0]

        # Per-neuron Newton fit. Convex problem → well-defined MAP.
        eye = jnp.eye(nb)
        prior = prior_precision * eye

        # Intercept-matching initial guess: find the weight vector that best
        # represents a constant log-rate equal to the mean rate of the data.
        # For a constant target t, the least-squares solution in the column
        # space of Z_base is:
        #
        #     w_init(t) = (Z' Z + prior)^{-1} Z' (t * 1_T)
        #              = t * [(Z' Z + prior)^{-1} Z' 1_T]
        #              = t * intercept_direction
        #
        # ``intercept_direction`` depends only on Z_base and is shared across
        # neurons; we hoist it out of the per-neuron vmap.
        #
        # Borrowed from NeMoS's ``initialize_intercept_matching_mean_rate``
        # (flatironinstitute/nemos) but adapted to our no-explicit-intercept
        # spline parameterization by projecting the constant onto Z_base's
        # column space rather than fitting a separate intercept. Starting
        # Newton from this instead of zeros makes the first iteration refine
        # the spatial pattern rather than find the mean rate.
        #
        # Note: ``ZtZ_plus_prior`` here is the *linearization-free* Gram
        # matrix used only for this one-shot initializer. The Newton-step
        # Hessian below is the data-dependent Fisher information
        # ``Z' diag(mu) Z + prior`` and must not be replaced with this.
        ZtZ_plus_prior = Z_base.T @ Z_base + prior
        Zt_ones = Z_base.T @ jnp.ones(n_time, dtype=Z_base.dtype)
        intercept_direction = psd_solve(ZtZ_plus_prior, Zt_ones)  # (nb,)

        def _fit_one(y: Array) -> tuple[Array, Array]:
            y = y.astype(Z_base.dtype)

            def _newton_step(w: Array, _: None) -> tuple[Array, None]:
                log_rate = Z_base @ w
                # Safe exponentiation under the same ceiling the main filter
                # uses. Prevents Newton from diverging if the data has a
                # pathological outlier bin.
                mu = _safe_expected_count(
                    log_rate, self.dt, max_log_count=self._max_log_count
                )
                grad = Z_base.T @ (mu - y) + prior_precision * w
                hess = Z_base.T @ (mu[:, None] * Z_base) + prior
                delta = psd_solve(hess, grad)
                return w - delta, None

            # Initial log-rate targets the mean rate of this neuron, in log
            # Hz. Guard against zero-spike neurons (add one "phantom spike"
            # per session to avoid log(0)).
            mean_count_per_bin = (jnp.sum(y) + 1.0) / n_time
            target_log_rate = jnp.log(mean_count_per_bin / self.dt)
            w0 = target_log_rate * intercept_direction
            w_mle, _ = jax.lax.scan(_newton_step, w0, None, length=max_iter)

            # Laplace covariance at the MAP
            log_rate = Z_base @ w_mle
            mu = _safe_expected_count(
                log_rate, self.dt, max_log_count=self._max_log_count
            )
            hess = Z_base.T @ (mu[:, None] * Z_base) + prior
            cov = psd_solve(hess, eye)
            return w_mle, symmetrize(cov)

        # vmap over neurons: map axis=1 of spikes_2d (each column is one neuron).
        w_mles, covs = jax.vmap(_fit_one, in_axes=1)(spikes_2d)
        # w_mles: (n_neurons, nb); covs: (n_neurons, nb, nb)

        if single_neuron:
            # Single-neuron path: return raw vectors/matrices at the
            # (n_basis_per_neuron,) scale.
            return w_mles[0], covs[0]

        # Multi-neuron: concatenate means and block-diagonalize covariances.
        init_mean = w_mles.reshape(n_neurons * nb)
        # Block-diagonal from (n_neurons, nb, nb) -> (n_neurons*nb, n_neurons*nb)
        # using flat index scatter
        total = n_neurons * nb
        idx = jnp.arange(n_neurons)
        row_base = (idx * nb)[:, None, None] + jnp.arange(nb)[None, :, None]
        col_base = (idx * nb)[:, None, None] + jnp.arange(nb)[None, None, :]
        init_cov = jnp.zeros((total, total), dtype=Z_base.dtype)
        init_cov = init_cov.at[row_base, col_base].set(covs)
        return init_mean, symmetrize(init_cov)

    def _warm_start_parameters(
        self,
        Z_base: Array,
        spikes: Array,
        window: Optional[slice],
    ) -> None:
        """Warm-start ``init_mean`` and ``init_cov`` from a stationary GLM.

        Sets ``self.init_mean`` and ``self.init_cov`` to the Laplace
        approximation (MAP, inverse Hessian) of a stationary Poisson GLM
        fit on ``(Z_base, spikes[window])``. Also sets ``transition_matrix``
        and ``process_cov`` from the scalar defaults (these are not
        warm-started — the scalar defaults are fine because EM / SGD
        updates them during fitting).
        """
        assert self.n_basis_per_neuron is not None
        if self.n_neurons == 1:
            self.n_basis = self.n_basis_per_neuron
        else:
            self.n_basis = self.n_neurons * self.n_basis_per_neuron

        n = self.n_basis
        self.transition_matrix = jnp.eye(n)
        self.process_cov = jnp.eye(n) * self.init_process_noise

        init_mean, init_cov = self._fit_stationary_glm(
            Z_base, spikes, window=window
        )
        self.init_mean = init_mean
        self.init_cov = init_cov

    @property
    def _max_log_count(self) -> float:
        """Physiological ceiling converted to ``log(rate * dt)``.

        Used by the Laplace-EKF filter's safe-exponentiation clip. A posterior
        that saturates this value indicates a data-quality issue (e.g. a
        pathological outlier bin from mis-binned spikes).
        """
        # Plain NumPy is fine here: this is a scalar computed on concrete
        # Python floats, not a traced JAX value.
        return float(np.log(self.max_firing_rate_hz * self.dt))

    # Fraction of bin/neuron combinations allowed to saturate the firing-rate
    # ceiling before the saturation warning fires. Chosen so that a single
    # bad bin in a 1000-bin session does trigger the warning (one bin out of
    # ~1000 is ~0.1%) while random near-ceiling fluctuations in a well-behaved
    # session do not. Raise this if you have so many bad bins that you want
    # only the worst offenders to produce warnings.
    _SATURATION_WARN_FRAC = 1e-3

    def _detect_block_structure(
        self, design_matrix: Array, force_dense: bool = False
    ) -> tuple[Optional[int], Optional[int]]:
        """Detect block-diagonal structure of the current filter problem.

        Returns ``(n_neurons, block_size)`` integers if the problem
        satisfies the block-diagonal contract (multi-neuron, block-
        diagonal A/Q/init_cov, shared Z_base across neurons, identical
        per-neuron A and Q blocks), or ``(None, None)`` otherwise.

        These integers are Python constants (not traced values), safe
        to pass into ``stochastic_point_process_filter`` inside a
        jit-compiled loss function. Inside the jit boundary the filter
        uses static slicing to extract per-neuron factors from the
        traced init_mean/init_cov/A/Q/design_matrix arrays.

        Called at fit / fit_sgd entry time with concrete arrays.
        Re-detection is required after each EM M-step because the
        M-step can change transition_matrix or process_cov — if
        ``update_transition_matrix=True`` the new A may not be
        block-diagonal, in which case the next E-step falls back to
        the dense path.

        Parameters
        ----------
        design_matrix : Array
            Block-expanded design matrix as built by
            ``_expand_to_block_diagonal``.
        force_dense : bool, default=False
            If True, skip detection entirely and return ``(None, None)``.
            Used by the ``force_dense=True`` escape hatch on ``fit``
            and ``fit_sgd``.

        Returns
        -------
        (n_neurons, block_size) : tuple of int or tuple of None
        """
        if force_dense:
            return None, None
        structure = _detect_block_diagonal_problem(
            self.init_mean,
            self.init_cov,
            self.transition_matrix,
            self.process_cov,
            design_matrix,
        )
        if structure is None:
            return None, None
        return structure.n_neurons, structure.block_size

    def _warn_if_rate_saturated(
        self,
        design_matrix: Array,
        posterior_mean: Array,
        context: str,
    ) -> None:
        """Emit a warning if enough filtered bin/neuron log-counts saturate.

        Computes ``log_rate + log(dt)`` at the posterior mean for each time
        bin and checks how many entries reach the Laplace-EKF clip ceiling
        (``self._max_log_count``). If more than ``_SATURATION_WARN_FRAC``
        of entries saturate, emits a ``UserWarning`` naming the count so
        the user can investigate upstream data quality rather than silently
        accept a clipped fit.

        Parameters
        ----------
        design_matrix : Array, shape (n_time, n_basis) or (n_time, n_neurons, n_basis)
            Same design matrix passed to the filter/smoother.
        posterior_mean : Array, shape (n_time, n_state)
            Filtered (or smoothed) posterior means ``x_{k|k}``.
        context : str
            Method name for the warning message (``"fit"`` or ``"fit_sgd"``).
        """
        # log_intensity_func(Z_t, x_t) returns a scalar (single-neuron) or
        # (n_neurons,) (multi-neuron). vmap over time to get log-rates for
        # every bin, then flatten so the count logic is shape-agnostic.
        log_rates = jax.vmap(self._log_intensity_func)(
            design_matrix, posterior_mean
        )
        log_counts = log_rates + float(np.log(self.dt))
        log_counts_flat = jnp.ravel(log_counts)

        ceiling = self._max_log_count
        n_saturated = int(jnp.sum(log_counts_flat >= ceiling))
        n_total = int(log_counts_flat.size)
        if n_total == 0:
            return
        frac = n_saturated / n_total
        if frac > self._SATURATION_WARN_FRAC:
            msg = (
                f"PlaceFieldModel.{context}: {n_saturated}/{n_total} "
                f"({100 * frac:.2f}%) bin/neuron log-counts saturated the "
                f"max_firing_rate_hz={self.max_firing_rate_hz:g} Hz ceiling "
                f"(max_log_count={ceiling:g}). This usually indicates a "
                f"data-quality issue (e.g. mis-binned spikes, artifact "
                f"counts, or an overly tight ceiling). Check your spike "
                f"binning and consider raising max_firing_rate_hz if the "
                f"ceiling is genuinely too tight for your neuron population."
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    def _e_step(
        self, design_matrix: Array, spikes: Array
    ) -> float:
        """E-step: run filter and smoother."""
        assert self.init_mean is not None, "Model not initialized"
        assert self.init_cov is not None, "Model not initialized"
        assert self.transition_matrix is not None, "Model not initialized"
        assert self.process_cov is not None, "Model not initialized"
        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_ll,
            self.filtered_mean,
            self.filtered_cov,
        ) = stochastic_point_process_smoother(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spikes,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self._log_intensity_func,
            return_filtered=True,
            max_log_count=self._max_log_count,
            # fit() validates once at the top before EM starts; skip
            # per-iteration re-validation (eigvalsh is O(d^3)).
            validate_inputs=False,
            # Block-diagonal dispatch: None/None falls through to dense.
            # fit() re-detects after every M-step in case
            # update_transition_matrix changes A's structure.
            block_n_neurons=self._block_n_neurons,
            block_size=self._block_size,
            max_newton_iter=self.max_newton_iter,
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
        assert self.smoother_mean is not None, "E-step must run before M-step"
        assert self.smoother_cov is not None, "E-step must run before M-step"
        assert self.smoother_cross_cov is not None, "E-step must run before M-step"
        assert self.n_basis is not None, "Model not initialized"
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
            # Keep init_cov diagonal to match n_free_params count and
            # keep the model tractable for BIC/AIC comparisons.
            self.init_cov = jnp.diag(jnp.maximum(jnp.diag(sc[0]), 1e-10))

    @staticmethod
    def bin_spike_times(
        spike_times: np.ndarray,
        time_bins: np.ndarray,
        warn_on_drops: bool = True,
    ) -> np.ndarray:
        """Bin a single neuron's spike times into time bins.

        Thin single-neuron wrapper around
        :func:`state_space_practice.preprocessing.bin_spike_times`. For
        multi-neuron data, call ``preprocessing.bin_spike_times`` directly.

        Uses ``np.histogram`` semantics: bins are left-closed
        ``[t_i, t_{i+1})`` for all bins except the last, which is
        ``[t_{T-1}, t_{T-1} + dt]``. Spikes outside
        ``[time_bins[0], time_bins[-1] + dt]`` are silently discarded;
        a ``UserWarning`` is issued when any are dropped (pass
        ``warn_on_drops=False`` to suppress).

        Parameters
        ----------
        spike_times : np.ndarray, shape (n_spikes,)
            Times of individual spikes (in seconds or matching time_bins units).
        time_bins : np.ndarray, shape (n_time,)
            Left edges of time bins (e.g., from ``np.arange(t_start, t_end, dt)``).
        warn_on_drops : bool, default=True
            Emit a warning if spikes fall outside the bin window. Prevents
            the silent-funnel-to-last-bin failure mode that used to occur
            when callers passed a sub-window of the spike time range.

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
        from state_space_practice.preprocessing import (
            bin_spike_times as _bin_spike_times_multi,
        )

        spike_times = np.asarray(spike_times)
        # Pass _warn_stacklevel=3 so the warning points at the user's call
        # site rather than this wrapper's delegation line: user -> wrapper
        # -> canonical -> _warn_if_out_of_window.
        counts_2d = _bin_spike_times_multi(
            [spike_times],
            time_bins,
            warn_on_drops=warn_on_drops,
            _warn_stacklevel=3,
        )
        return counts_2d[:, 0]

    def fit(
        self,
        position: np.ndarray,
        spikes: ArrayLike,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        knots_x: Optional[np.ndarray] = None,
        knots_y: Optional[np.ndarray] = None,
        verbose: bool = True,
        warm_start: bool = True,
        warm_start_window: Optional[slice] = None,
        force_dense: bool = False,
    ) -> list[float]:
        """Fit the model to spike data and position using EM.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
            Animal position (x, y) at each time bin.
        spikes : ArrayLike, shape (n_time,)
            Spike counts per time bin. Use ``PlaceFieldModel.bin_spike_times``
            to convert spike time arrays to binned counts.
        max_iter : int, default=100
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
        warm_start : bool, default=True
            If True (default), initialize ``init_mean`` and ``init_cov``
            from the Laplace approximation of a stationary Poisson GLM
            fit on the same ``(position, spikes)``. On real CA1 data this
            improves the first-iteration marginal log-likelihood by ~7
            orders of magnitude vs. zero weights with an identity prior.
            Set to False to use the old scalar defaults (useful for
            ablation studies or when you want bit-identical behavior with
            pre-warm-start releases).
        warm_start_window : slice or None, default=None
            Time window to fit the stationary warm-start GLM on. If None
            (default), uses the whole session — this is usually best for
            both statistical efficiency (more spikes → better MAP estimate)
            and spatial coverage (the animal may only visit certain places
            later in the session). Use a ``slice`` (e.g.
            ``slice(0, T // 2)``) for sessions with substantial
            within-session drift where the initial field differs noticeably
            from the session-average field.
        force_dense : bool, default=False
            Skip the block-diagonal filter dispatch even when the
            problem structure would allow it. Used to compare block vs
            dense output numerically, or to bypass a suspected
            block-path bug. Default False: multi-neuron problems with
            block-diagonal A, Q, init_cov, and shared Z_base across
            neurons automatically dispatch to the per-neuron vmap
            filter for ~n_neurons^2 speedup.

        Returns
        -------
        log_likelihoods : list[float]
            Marginal log-likelihood at each EM iteration.
        """
        # Validate inputs
        position = np.asarray(position)
        spikes = jnp.asarray(spikes)
        if spikes.ndim == 1:
            spikes = spikes[:, None]  # (n_time,) -> (n_time, 1)
        if spikes.ndim != 2:
            raise ValueError(
                f"spikes must be 1D (n_time,) or 2D (n_time, n_neurons), "
                f"got shape {spikes.shape}."
            )
        if jnp.any(spikes < 0):
            raise ValueError(
                "spikes must be non-negative counts. Got negative values. "
                "If you have continuous rates, bin them first."
            )
        if position.shape[0] != spikes.shape[0]:
            raise ValueError(
                f"position and spikes must have the same number of time bins: "
                f"got position ({position.shape[0]},) vs spikes ({spikes.shape[0]},)"
            )

        self.n_neurons = spikes.shape[1]
        self._n_time = spikes.shape[0]
        self._total_spikes = int(spikes.sum())

        # Squeeze back to 1D for single-neuron (filter expects 1D)
        if self.n_neurons == 1:
            spikes = spikes.squeeze(axis=1)

        # Build basis, expand to block-diagonal for the filter, then
        # (optionally) warm-start. The warm-start only needs Z_base, so we
        # build the full design matrix once and reuse it.
        Z_base = self._build_spline_basis_matrix(position, knots_x, knots_y)
        design_matrix = self._expand_to_block_diagonal(Z_base)
        if warm_start:
            self._warm_start_parameters(Z_base, spikes, warm_start_window)
        else:
            self._initialize_parameters()

        # Numerical sanity check: validate init_cov is PSD and warn if
        # the configuration is at risk of f32 NaN during the scan. Runs
        # once here so the EM loop's _e_step calls can skip re-validation.
        _validate_filter_numerics(
            self.init_cov, n_time=design_matrix.shape[0]
        )

        # Block-diagonal dispatch: detect once before the EM loop.
        # Re-detected after each M-step in case the M-step writes back
        # a non-block-diagonal A (only possible when
        # update_transition_matrix=True).
        self._block_n_neurons, self._block_size = self._detect_block_structure(
            design_matrix, force_dense=force_dense
        )

        def _print(msg: str) -> None:
            if verbose:
                print(msg)

        neurons_str = f", n_neurons={self.n_neurons}" if self.n_neurons > 1 else ""
        block_str = (
            f", block_dispatch={self._block_n_neurons}x{self._block_size}"
            if self._block_n_neurons is not None
            else ""
        )
        _print(
            f"PlaceFieldModel: n_time={self._n_time}, "
            f"n_basis={self.n_basis_per_neuron}{neurons_str}, "
            f"total_spikes={self._total_spikes}{block_str}"
        )

        self.log_likelihoods = []

        for iteration in range(max_iter):
            # Save previous state so we can roll back on LL decrease
            prev_smoother_mean = self.smoother_mean
            prev_smoother_cov = self.smoother_cov
            prev_smoother_cross_cov = self.smoother_cross_cov
            prev_filtered_mean = self.filtered_mean
            prev_filtered_cov = self.filtered_cov

            ll = self._e_step(design_matrix, spikes)
            self.log_likelihoods.append(ll)

            _print(f"  EM iter {iteration + 1:>{len(str(max_iter))}}/{max_iter}: LL = {ll:.1f}")

            if not jnp.isfinite(ll):
                _print(f"  WARNING: Non-finite LL at iteration {iteration + 1}")
                break

            if iteration > 0:
                is_converged, is_increasing = check_converged(
                    ll, self.log_likelihoods[-2], tolerance
                )
                if not is_increasing:
                    # Roll back to the previous (better) state
                    self.smoother_mean = prev_smoother_mean
                    self.smoother_cov = prev_smoother_cov
                    self.smoother_cross_cov = prev_smoother_cross_cov
                    self.filtered_mean = prev_filtered_mean
                    self.filtered_cov = prev_filtered_cov
                    # Remove the bad LL so log_likelihoods[-1] matches
                    # the stored model state (used by bic/aic/summary).
                    bad_ll = self.log_likelihoods.pop()
                    msg = (
                        f"LL decreased: "
                        f"{self.log_likelihoods[-1]:.1f} -> {bad_ll:.1f}; "
                        f"stopping EM and rolling back to previous E-step."
                    )
                    _print(f"  WARNING: {msg}")
                    logger.warning(msg)
                    break
                if is_converged:
                    _print(f"  Converged after {iteration + 1} iterations.")
                    break

            self._m_step()
            # Re-detect block structure after the M-step, unconditionally.
            # The primary case where the M-step can break block-
            # diagonality is update_transition_matrix=True (where
            # kalman_maximization_step returns both a dense A_new and a
            # dense Q_new that are not diagonalized before being
            # written to self.process_cov). But unconditional re-
            # detection is cheap relative to the E-step (one host-side
            # _detect_block_diagonal_problem call per iteration) and
            # guards against any future M-step extension that breaks
            # structure without touching update_transition_matrix.
            self._block_n_neurons, self._block_size = (
                self._detect_block_structure(
                    design_matrix, force_dense=force_dense
                )
            )
        else:
            msg = (
                f"EM reached maximum iterations ({max_iter}) without "
                f"converging. Consider increasing max_iter or loosening "
                f"tolerance."
            )
            _print(f"  WARNING: {msg}")
            logger.warning(msg)

        # Saturation diagnostic: post-hoc check on the final filtered posterior.
        # Runs after fit completes so a substantial fraction of clipped bins
        # surfaces as a warning rather than a silently degraded fit.
        if self.filtered_mean is not None:
            self._warn_if_rate_saturated(
                design_matrix, self.filtered_mean, context="fit"
            )

        return self.log_likelihoods

    # --- SGDFittableMixin protocol ---

    def fit_sgd(  # type: ignore[override]
        self,
        position: ArrayLike,
        spikes: ArrayLike,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        warm_start: bool = True,
        warm_start_window: Optional[slice] = None,
        force_dense: bool = False,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        position : ArrayLike, shape (n_time, 2)
            2D positions at each time bin.
        spikes : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Spike counts per time bin.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps.
        convergence_tol : float or None
            If set, stop early when |ΔLL| < tol for 5 consecutive steps.
        warm_start : bool, default=True
            If True (default), initialize ``init_mean`` and ``init_cov``
            from the Laplace approximation of a stationary Poisson GLM
            fit on the same ``(position, spikes)``. On real CA1 data this
            improves the first-step marginal log-likelihood by orders of
            magnitude vs. zero weights with an identity prior. Set to
            False to use the old scalar defaults.
        warm_start_window : slice or None, default=None
            Time window to fit the stationary warm-start GLM on. If None
            (default), uses the whole session. See ``fit`` docstring for
            when to override.
        force_dense : bool, default=False
            Skip the block-diagonal filter dispatch even when the problem
            structure would allow it. See ``fit`` docstring for details.

        Returns
        -------
        log_likelihoods : list of float

        """
        position = np.asarray(position)
        spikes = jnp.asarray(spikes)
        if spikes.ndim == 1:
            spikes = spikes[:, None]

        self._sgd_n_time = spikes.shape[0]
        self.n_neurons = spikes.shape[1]

        # Squeeze back to 1D for single-neuron (filter expects 1D)
        if self.n_neurons == 1:
            spikes = spikes.squeeze(axis=1)

        # Build basis, expand to block-diagonal once, then (optionally)
        # warm-start. Warm-start only needs Z_base; the expanded
        # design_matrix feeds the filter.
        #
        # The ``elif self.init_mean is None`` guard on the cold-start path
        # is preserved from the pre-refactor behavior. Unlike ``fit`` (EM),
        # which always resets init state, repeated ``fit_sgd(..., warm_start=
        # False)`` calls on the same model reuse the existing init state —
        # this is intentional for users who want to resume optimization
        # from a previous fit_sgd result without a warm-start reset.
        Z_base = self._build_spline_basis_matrix(position)
        design_matrix = self._expand_to_block_diagonal(Z_base)
        if warm_start:
            self._warm_start_parameters(Z_base, spikes, warm_start_window)
        elif self.init_mean is None:
            self._initialize_parameters()

        # Numerical sanity check: validate init_cov is PSD and warn if
        # the configuration is at risk of f32 NaN during the scan. Runs
        # once here so the SGD loop's _sgd_loss_fn can skip re-validation
        # (the eigvalsh call is not jit-traceable anyway).
        _validate_filter_numerics(
            self.init_cov, n_time=design_matrix.shape[0]
        )

        # Block-diagonal dispatch: detect once before entering the SGD
        # loop. The detection result is stored on self and read by the
        # (jit'd) _sgd_loss_fn. Since n_neurons/block_size are Python
        # ints (not traced), passing them into the filter is safe.
        self._block_n_neurons, self._block_size = self._detect_block_structure(
            design_matrix, force_dense=force_dense
        )

        return super().fit_sgd(
            design_matrix, spikes,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    @property
    def _n_timesteps(self) -> int:
        return self._sgd_n_time

    def _check_sgd_initialized(self) -> None:
        if self.init_mean is None:
            raise RuntimeError(
                "Model parameters not initialized. "
                "Call fit_sgd(position, spikes) not super().fit_sgd() directly."
            )

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            POSITIVE,
            PSD_MATRIX,
            UNCONSTRAINED,
        )

        params: dict = {}
        spec: dict = {}

        if self.update_process_cov:
            if self.process_noise_structure == "diagonal":
                # Optimize per-basis-function variances
                params["process_diag"] = jnp.diag(self.process_cov)
                spec["process_diag"] = POSITIVE
            else:  # isotropic
                params["process_scalar"] = jnp.array(
                    jnp.mean(jnp.diag(self.process_cov))
                )
                spec["process_scalar"] = POSITIVE

        if self.update_transition_matrix:
            params["transition_matrix"] = self.transition_matrix
            spec["transition_matrix"] = UNCONSTRAINED

        if self.update_init_state:
            params["init_mean"] = self.init_mean
            spec["init_mean"] = UNCONSTRAINED
            params["init_cov"] = self.init_cov
            spec["init_cov"] = PSD_MATRIX

        return params, spec

    def _sgd_loss_fn(
        self, params: dict, design_matrix: Array, spikes: Array
    ) -> Array:
        A = params.get("transition_matrix", self.transition_matrix)
        m0 = params.get("init_mean", self.init_mean)
        P0 = params.get("init_cov", self.init_cov)

        if "process_diag" in params:
            Q = jnp.diag(params["process_diag"])
        elif "process_scalar" in params:
            Q = jnp.eye(self.n_basis) * params["process_scalar"]
        else:
            Q = self.process_cov

        _, _, marginal_ll = stochastic_point_process_filter(
            init_mean_params=m0,
            init_covariance_params=P0,
            design_matrix=design_matrix,
            spike_indicator=spikes,
            dt=self.dt,
            transition_matrix=A,
            process_cov=Q,
            log_conditional_intensity=self._log_intensity_func,
            max_log_count=self._max_log_count,
            # fit_sgd validates once at the top before the SGD loop;
            # skip per-step re-validation inside the jit'd loss fn
            # (the eigvalsh would break tracing anyway).
            validate_inputs=False,
            # Block-diagonal dispatch: n_neurons and block_size are
            # Python ints captured from self (not traced). None falls
            # through to the dense path.
            block_n_neurons=self._block_n_neurons,
            block_size=self._block_size,
            max_newton_iter=self.max_newton_iter,
        )
        return -marginal_ll

    def _store_sgd_params(self, params: dict) -> None:
        if "process_diag" in params:
            self.process_cov = jnp.diag(params["process_diag"])
        elif "process_scalar" in params:
            self.process_cov = jnp.eye(self.n_basis) * params["process_scalar"]
        if "transition_matrix" in params:
            self.transition_matrix = params["transition_matrix"]
        if "init_mean" in params:
            self.init_mean = params["init_mean"]
        if "init_cov" in params:
            self.init_cov = params["init_cov"]

    def _finalize_sgd(
        self, design_matrix: Array, spikes: Array
    ) -> None:
        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_ll,
            self.filtered_mean,
            self.filtered_cov,
        ) = stochastic_point_process_smoother(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spikes,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self._log_intensity_func,
            return_filtered=True,
            max_log_count=self._max_log_count,
            # fit_sgd validated at the top; skip per-call re-validation.
            validate_inputs=False,
            # Block-diagonal dispatch (None/None falls through to dense).
            block_n_neurons=self._block_n_neurons,
            block_size=self._block_size,
            max_newton_iter=self.max_newton_iter,
        )
        self.log_likelihoods = [float(marginal_ll)]
        # Saturation diagnostic: post-hoc check on the filtered posterior.
        # If a substantial fraction of bins saturate the physiological
        # ceiling, the filter output is unreliable.
        assert self.filtered_mean is not None
        self._warn_if_rate_saturated(
            design_matrix, self.filtered_mean, context="fit_sgd"
        )

    def predict_rate_map(
        self,
        grid_positions: np.ndarray,
        time_slice: Optional[slice] = None,
        alpha: float = 0.05,
        neuron_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the firing rate map at given spatial positions.

        Evaluates the estimated place field at a grid of positions,
        averaging the smoothed weights over the specified time window.

        Parameters
        ----------
        grid_positions : np.ndarray, shape (n_grid_points, 2)
            Spatial positions (x, y) at which to evaluate the rate.
        time_slice : slice or None, optional
            Time window over which to average the smoothed weights.
            If None, uses the full session.
        alpha : float, default=0.05
            Significance level for credible interval (0.05 for 95% CI).
        neuron_idx : int, default=0
            Which neuron's rate map to compute (for multi-neuron models).

        Returns
        -------
        rate : np.ndarray, shape (n_grid_points,)
            Estimated firing rate (Hz) at each grid position.
        rate_ci : np.ndarray, shape (n_grid_points, 2)
            Lower and upper credible bounds on the rate (Hz).

        Notes
        -----
        The credible interval reflects the average per-time-step marginal
        posterior uncertainty over the window, not the uncertainty of the
        time-averaged weight. This gives a representative sense of how
        uncertain the rate estimate is at each spatial location during the
        specified period.

        This method always uses the linear approximation ``exp(Z @ weights)``.
        If a nonlinear ``log_intensity_func`` was set at construction, this
        prediction is an approximation. Use ``smoother_mean`` directly with
        your intensity function for exact predictions.
        """
        self._check_fitted("predict_rate_map")
        assert self.basis_info is not None
        assert self.smoother_mean is not None
        assert self.smoother_cov is not None

        if self._log_intensity_func is not log_conditional_intensity:
            warnings.warn(
                "predict_rate_map uses the linear approximation exp(Z @ x). "
                "For the nonlinear log_intensity_func set on this model, "
                "use smoother_mean directly with your intensity function.",
                stacklevel=2,
            )

        Z_grid = evaluate_basis(grid_positions, self.basis_info)
        s, _ = self._neuron_weights(neuron_idx)

        if time_slice is None:
            time_slice = slice(None)

        weights = np.array(self.smoother_mean[time_slice, s].mean(axis=0))
        cov = np.array(self.smoother_cov[time_slice][:, s, s].mean(axis=0))

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
        neuron_idx: int = 0,
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
        neuron_idx : int, default=0
            Which neuron's center to estimate (for multi-neuron models).

        Returns
        -------
        centers : np.ndarray, shape (n_blocks, 2)
            Estimated place field center (x, y) in each block.
        """
        self._check_fitted("predict_center")
        assert self.basis_info is not None
        assert self.smoother_mean is not None

        Z_grid = evaluate_basis(grid_positions, self.basis_info)
        s, _ = self._neuron_weights(neuron_idx)
        n_time = self.smoother_mean.shape[0]
        if n_blocks < 1 or n_blocks > n_time:
            raise ValueError(
                f"n_blocks must be between 1 and n_time ({n_time}), got {n_blocks}"
            )

        block_indices = np.array_split(np.arange(n_time), n_blocks)
        centers = np.zeros((n_blocks, 2))
        for i, idx in enumerate(block_indices):
            weights = np.array(self.smoother_mean[idx][:, s].mean(axis=0))
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
        self._check_fitted("make_grid")
        assert self.basis_info is not None
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
        spikes : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Spike counts per time bin. Must match the number of neurons
            the model was fitted with.

        Returns
        -------
        log_likelihood : float
            Marginal log-likelihood of the held-out data.
        """
        self._check_fitted("score")
        assert self.basis_info is not None
        assert self.init_mean is not None
        assert self.init_cov is not None
        assert self.transition_matrix is not None
        assert self.process_cov is not None

        position = np.asarray(position)
        spikes = jnp.asarray(spikes)
        # Validate and normalize spikes shape
        if self.n_neurons == 1:
            if spikes.ndim == 2:
                spikes = spikes.squeeze(axis=1)
        else:
            if spikes.ndim == 1:
                raise ValueError(
                    f"Model was fitted with n_neurons={self.n_neurons} but "
                    f"received 1D spikes. Pass (n_time, {self.n_neurons}) array."
                )
            if spikes.ndim == 2 and spikes.shape[1] != self.n_neurons:
                raise ValueError(
                    f"Expected {self.n_neurons} spike columns (matching fitted "
                    f"n_neurons), got {spikes.shape[1]}."
                )
        if position.shape[0] != spikes.shape[0]:
            raise ValueError(
                f"position and spikes must have the same number of time bins: "
                f"got position ({position.shape[0]},) vs spikes ({spikes.shape[0]},)"
            )

        # Build design matrix matching the fitted n_neurons
        Z_base = evaluate_basis(position, self.basis_info)
        if self.n_neurons == 1:
            design_matrix = jnp.asarray(Z_base)
        else:
            design_matrix = self._build_block_diagonal(Z_base)

        _, _, marginal_ll = stochastic_point_process_filter(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spikes,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self._log_intensity_func,
            max_log_count=self._max_log_count,
            # init_cov was validated at fit time; skip O(d^3) eigvalsh
            # on every score() call.
            validate_inputs=False,
            # Reuse the block-diagonal dispatch decision made at fit time.
            block_n_neurons=self._block_n_neurons,
            block_size=self._block_size,
            max_newton_iter=self.max_newton_iter,
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
        self._check_fitted("get_state_confidence_interval")
        assert self.smoother_mean is not None
        assert self.smoother_cov is not None
        return get_confidence_interval(
            self.smoother_mean, self.smoother_cov, alpha=alpha
        )

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    @property
    def n_free_params(self) -> int:
        """Number of free parameters learned via EM.

        Counts diagonal entries of Q (or 1 for isotropic), A entries if
        learned, and initial mean + diagonal covariance if learned.
        """
        assert self.n_basis is not None, "Model not initialized"
        nb = self.n_basis
        n = 0
        if self.update_process_cov:
            n += nb if self.process_noise_structure == "diagonal" else 1
        if self.update_transition_matrix:
            n += nb ** 2
        if self.update_init_state:
            n += nb  # mean
            n += nb  # diagonal of covariance (effective)
        return n

    def bic(self) -> float:
        """Bayesian Information Criterion. Lower is better.

        BIC = -2 * log_likelihood + k * ln(n)

        where k is the number of free parameters and n is the number of
        time bins.

        Returns
        -------
        float
        """
        self._check_fitted("bic")
        assert self.smoother_mean is not None
        n = self.smoother_mean.shape[0]
        return float(-2.0 * self.log_likelihoods[-1] + self.n_free_params * np.log(n))

    def aic(self) -> float:
        """Akaike Information Criterion. Lower is better.

        AIC = -2 * log_likelihood + 2 * k

        where k is the number of free parameters.

        Returns
        -------
        float
        """
        self._check_fitted("aic")
        return -2.0 * self.log_likelihoods[-1] + 2.0 * self.n_free_params

    def summary(self) -> str:
        """Return a text summary of the fitted model.

        Returns
        -------
        str
            Multi-line summary including data statistics, fit quality,
            process noise, and drift metrics.
        """
        self._check_fitted("summary")
        assert self.smoother_mean is not None
        assert self.process_cov is not None

        n_time = self.smoother_mean.shape[0]
        session_duration = n_time * self.dt
        mean_rate = self._total_spikes / session_duration if session_duration > 0 else 0.0

        q_diag = jnp.diag(self.process_cov)

        lines = [
            "PlaceFieldModel Summary",
            "=" * 50,
            f"  dt:                       {self.dt:.4g} s",
            f"  n_interior_knots:         {self.n_interior_knots}",
            f"  n_basis_per_neuron:       {self.n_basis_per_neuron}",
            f"  n_neurons:                {self.n_neurons}",
            f"  process_noise_structure:  {self.process_noise_structure}",
            "",
            "Data",
            "-" * 50,
            f"  n_time_bins:              {n_time}",
            f"  session_duration:         {session_duration:.1f} s",
            f"  total_spikes:             {self._total_spikes}",
            f"  mean_rate:                {mean_rate:.2f} Hz",
            "",
            "Fit",
            "-" * 50,
            f"  EM iterations:            {len(self.log_likelihoods)}",
            f"  final log-likelihood:     {self.log_likelihoods[-1]:.2f}",
            f"  BIC:                      {self.bic():.2f}",
            f"  AIC:                      {self.aic():.2f}",
            f"  n_free_params:            {self.n_free_params}",
            "",
            "Process noise Q (diagonal)",
            "-" * 50,
            f"  mean:                     {float(q_diag.mean()):.2e}",
            f"  min:                      {float(q_diag.min()):.2e}",
            f"  max:                      {float(q_diag.max()):.2e}",
        ]

        try:
            drift = self.drift_summary(n_blocks=10)
            lines.extend([
                "",
                "Drift",
                "-" * 50,
                f"  total (start-to-end):     {drift['total_drift']:.2f} cm",
                f"  cumulative (path):        {drift['cumulative_drift']:.2f} cm",
            ])
        except (ValueError, RuntimeError) as e:
            logger.debug("drift_summary unavailable in summary: %s", e)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Drift analysis
    # ------------------------------------------------------------------

    def drift_summary(
        self,
        n_grid: int = 80,
        n_blocks: int = 20,
        neuron_idx: int = 0,
    ) -> dict:
        """Summarize place field drift over the session.

        Parameters
        ----------
        n_grid : int, default=80
            Grid resolution for center estimation.
        n_blocks : int, default=20
            Number of temporal blocks for center trajectory.
        neuron_idx : int, default=0
            Which neuron to summarize (for multi-neuron models).

        Returns
        -------
        summary : dict with keys:
            centers : (n_blocks, 2) — place field center per block
            total_drift : float — Euclidean distance from first to last center (cm)
            cumulative_drift : float — total path length of center trajectory (cm)
            peak_rate_per_block : (n_blocks,) — peak rate in each block (Hz)
            block_times : (n_blocks,) — center time of each block (seconds)
        """
        self._check_fitted("drift_summary")
        assert self.basis_info is not None
        assert self.smoother_mean is not None

        grid, _, _ = self.make_grid(n_grid)
        Z_grid = evaluate_basis(grid, self.basis_info)
        s, _ = self._neuron_weights(neuron_idx)
        n_time = self.smoother_mean.shape[0]

        block_indices = np.array_split(np.arange(n_time), n_blocks)
        centers = np.zeros((n_blocks, 2))
        peak_rates = np.zeros(n_blocks)
        block_times = np.zeros(n_blocks)

        for i, idx in enumerate(block_indices):
            weights = np.array(self.smoother_mean[idx][:, s].mean(axis=0))
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
            "peak_rate_per_block": peak_rates,
            "block_times": block_times * self.dt,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_rate_maps(
        self,
        n_time_bins: int = 3,
        n_grid: int = 50,
        neuron_idx: int = 0,
        ax: Optional[np.ndarray] = None,
    ):
        """Plot estimated rate maps in temporal bins.

        Parameters
        ----------
        n_time_bins : int, default=3
            Number of temporal bins (e.g., 3 for early/middle/late).
        n_grid : int, default=50
            Spatial grid resolution per dimension.
        neuron_idx : int, default=0
            Which neuron to plot (for multi-neuron models).
        ax : array of Axes or None
            Matplotlib axes to plot into. If None, creates a new figure.

        Returns
        -------
        fig : Figure
            The matplotlib figure.
        """
        import matplotlib.pyplot as plt

        self._check_fitted("plot_rate_maps")
        assert self.smoother_mean is not None

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
            rate, _ = self.predict_rate_map(
                grid, time_slice=slice(idx[0], idx[-1] + 1), neuron_idx=neuron_idx,
            )
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
                x_edges, y_edges, rate_map, cmap="viridis", vmin=0, vmax=vmax
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

        Colors the trajectory from dark (start) to bright (end).

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

        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, n_blocks))
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
