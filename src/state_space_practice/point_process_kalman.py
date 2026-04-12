"""Point-process Kalman filter and smoother for neural spike data.

This module implements state-space models with point-process (spike) observations
using the Laplace-EKF approach from Eden & Brown (2004).

The model is:
    x_k = A @ x_{k-1} + w_k,  w_k ~ N(0, Q)
    y_{n,k} ~ Poisson(exp(log_intensity_func(Z_k, x_k)[n]) * dt)

where x_k is the latent state, y_{n,k} is the spike count for neuron n at time k,
and log_intensity_func returns log firing rates for all neurons.

Multi-Neuron Support
--------------------
The filter supports multiple neurons sharing a common latent state:

- spike_indicator: (n_time, n_neurons) - spike counts for each neuron
- log_conditional_intensity(Z_k, x_k) returns (n_neurons,) log-intensities

For backwards compatibility, single-neuron inputs are automatically promoted:
- spike_indicator: (n_time,) is treated as (n_time, 1)
- scalar log-intensity output is wrapped to (1,)

References
----------
[1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
    Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
    Neural Computation 16, 971-998.
"""

import logging
import warnings
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import (
    _kalman_smoother_update,
    psd_solve,
    stabilize_covariance,
    sum_of_outer_products,
    symmetrize,
)
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import check_converged

logger = logging.getLogger(__name__)


def _validate_filter_numerics(
    init_covariance: Array,
    n_time: int,
    stacklevel: int = 3,
) -> None:
    """Validate init_cov positive definiteness + warn about f32 numerical risk.

    Raises
    ------
    ValueError
        If ``init_covariance`` is non-square, not symmetric, or has a
        non-positive minimum eigenvalue. The filter's Cholesky-based
        updates require *strict* positive definiteness — a rank-deficient
        or indefinite input will NaN on the first Cholesky, so there is
        no point running the forward pass.

    Warns
    -----
    UserWarning
        If ``init_covariance.dtype`` is ``float32`` AND the problem is
        long enough / ill-conditioned enough that accumulated covariance
        roundoff is likely to drive the predicted covariance below PSD
        during the scan. The warning names the three risk factors
        (T, n_state, cond) and tells the user the fix: enable float64
        BEFORE importing the library.

    Notes
    -----
    This helper runs at the top of each public entry point (``fit``,
    ``fit_sgd``) once per call, then inner call sites pass
    ``validate_inputs=False`` to skip re-validation. This matters for
    two reasons:

    1. The SGD loss function runs inside ``jax.jit``, which forbids the
       ``eigvalsh → float(...)`` conversion used here. Inner calls must
       bypass the check entirely.
    2. EM iterations re-run the E-step hundreds of times on unchanged
       ``init_cov``, so per-iteration re-validation wastes O(d^3) work
       per iteration.

    The pattern is: each model class's ``fit`` / ``fit_sgd`` validates
    explicitly at the top of the public API call, and threads
    ``validate_inputs=False`` through to inner ``_e_step``, ``_sgd_loss_fn``,
    and ``_finalize_sgd`` methods.
    """
    if init_covariance.ndim != 2 or init_covariance.shape[0] != init_covariance.shape[1]:
        raise ValueError(
            f"init_covariance must be a square 2D matrix, got shape "
            f"{init_covariance.shape}"
        )

    # Eigenvalue check. eigvalsh is O(d^3) but only runs once per filter
    # invocation, vs d^3 per scan step — negligible.
    init_cov_sym = symmetrize(init_covariance)
    eigs = jnp.linalg.eigvalsh(init_cov_sym)
    min_eig = float(eigs.min())
    max_eig = float(eigs.max())

    if min_eig <= 0:
        raise ValueError(
            f"init_covariance is not positive definite "
            f"(min eigenvalue {min_eig:g}). The filter's Cholesky-based "
            f"updates require strict positive definiteness; a rank-"
            f"deficient or indefinite matrix will NaN on the first step. "
            f"Check the warm-start or init_cov_scale parameter, or "
            f"supply a known-PD matrix."
        )

    # Condition number. Cap the denominator to avoid divide-by-zero on
    # an (already-filtered) perfectly-rank-deficient matrix.
    cond = max_eig / max(min_eig, 1e-300)
    n_state = int(init_covariance.shape[0])

    # Check precision: the filter's scan body inherits its dtype from
    # init_covariance (via jnp.asarray internally). If the caller passed
    # an f32 array — either because jax_enable_x64 is off, or because
    # they explicitly cast — the inner Cholesky solves and matrix
    # products accumulate f32 roundoff. f64 arrays do not have this
    # issue in practice.
    is_f32 = init_covariance.dtype == jnp.float32

    if is_f32:
        # Rough upper bound on per-bin absolute covariance roundoff from
        # the predict-step congruence (A @ P @ A^T + Q) and the Laplace
        # update (psd_solve'd precision). The constants come from
        # standard error-analysis bounds for Cholesky; we use a
        # conservative ~sqrt(n_state) factor and f32 machine epsilon
        # ~1.2e-7. Accumulated over n_time bins (random-walk model),
        # total roundoff ~ n_time * sqrt(n_state) * eps * max_eig.
        f32_eps = 1.2e-7
        worst_roundoff = (
            float((n_time * n_state) ** 0.5) * f32_eps * max_eig
        )
        if worst_roundoff > 0.5 * min_eig:
            warnings.warn(
                f"stochastic_point_process_filter running in float32 with "
                f"a long / ill-conditioned problem: "
                f"T={n_time}, n_state={n_state}, "
                f"init_cov condition number {cond:.1e}, "
                f"min_eig {min_eig:.2e}, max_eig {max_eig:.2e}. "
                f"Estimated accumulated covariance roundoff "
                f"({worst_roundoff:.2e}) exceeds half of min_eig, which "
                f"means the predict step's covariance is likely to lose "
                f"PSD during the scan and produce NaN. Enable float64 "
                f"BEFORE importing state_space_practice:\n"
                f"    import jax\n"
                f"    jax.config.update('jax_enable_x64', True)\n"
                f"    # now import state_space_practice models",
                UserWarning,
                stacklevel=stacklevel,
            )


def _is_block_diagonal(
    mat: Array,
    n_blocks: int,
    block_size: int,
    atol: float = 1e-10,
) -> bool:
    """Check whether a square matrix has block-diagonal structure.

    Returns True iff ``mat`` is a ``(n_blocks*block_size, n_blocks*block_size)``
    matrix whose off-diagonal blocks are all zero within ``atol``.

    Parameters
    ----------
    mat : Array, shape (n, n) where n = n_blocks * block_size
        Matrix to check.
    n_blocks : int
        Number of diagonal blocks.
    block_size : int
        Size of each block (matching block size across all blocks).
    atol : float, default=1e-10
        Absolute tolerance for zero off-diagonal blocks, applied to
        ``max|off_block_entries|``. The default is conservative for f64;
        callers on f32 should pass a larger value.

    Returns
    -------
    bool
        True if the off-diagonal blocks are all zero within tolerance.
        False otherwise (including shape mismatches — non-square or wrong
        total size returns False, not an error).
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    expected = n_blocks * block_size
    if mat.shape[0] != expected:
        return False
    if n_blocks == 1:
        # Trivially block-diagonal: single block IS the whole matrix.
        return True

    # Materialize an off-block mask (n, n) that is True outside the
    # block-diagonal structure. This is cheap vs the alternative of
    # iterating over all (n_blocks choose 2) off-block positions.
    #
    # jnp.kron rejects bool dtype, so we build the mask with int32
    # and convert to bool for jnp.where.
    eye_blocks = jnp.eye(n_blocks, dtype=jnp.int32)
    ones_block = jnp.ones((block_size, block_size), dtype=jnp.int32)
    # Kronecker-expand: entries are 1 where the block-diagonal structure
    # allows, 0 elsewhere. Convert to bool for masked-where.
    block_mask = jnp.kron(eye_blocks, ones_block).astype(bool)
    off_block_max = float(jnp.max(jnp.abs(jnp.where(block_mask, 0.0, mat))))
    return off_block_max <= atol


class BlockDiagonalStructure(NamedTuple):
    """Factored block-diagonal filter problem.

    Holds the per-neuron components of a point-process filter problem
    where ``A``, ``Q``, ``init_cov``, and the design matrix ``Z`` are all
    block-diagonal across neurons with the same per-neuron block size,
    AND the ``A`` and ``Q`` diagonal blocks are identical across neurons.

    The dense-form filter can be replaced by a vmapped per-neuron scan
    using these components — see ``_stochastic_point_process_filter_
    block_diagonal`` for the implementation. The vmap runs a d=block_size
    filter per neuron rather than a d=n_neurons*block_size dense filter,
    which is ~n_neurons^2 cheaper on the hot Cholesky path.

    Attributes
    ----------
    A_block : Array, shape (block_size, block_size)
        Per-neuron transition matrix (same across neurons — verified
        at detection time).
    Q_block : Array, shape (block_size, block_size)
        Per-neuron process noise covariance (same across neurons —
        verified at detection time).
    init_means_per_neuron : Array, shape (n_neurons, block_size)
        Per-neuron initial mean, sliced out of the concatenated state vector.
    init_covs_per_neuron : Array, shape (n_neurons, block_size, block_size)
        Per-neuron initial covariance, sliced out of the block-diagonal
        init_cov.
    Z_base : Array, shape (n_time, block_size)
        Shared spline basis. Identical across neurons because the
        upstream design_matrix was built by block-diagonalizing a
        single Z_base — verified at detection time.
    n_neurons : int
        Number of neurons.
    block_size : int
        Per-neuron state dimension.
    """

    A_block: Array
    Q_block: Array
    init_means_per_neuron: Array
    init_covs_per_neuron: Array
    Z_base: Array
    n_neurons: int
    block_size: int


def _detect_block_diagonal_problem(
    init_mean: Array,
    init_cov: Array,
    transition_matrix: Array,
    process_cov: Array,
    design_matrix: Array,
    atol: float = 1e-10,
) -> Optional[BlockDiagonalStructure]:
    """Detect whether a filter problem has block-diagonal structure.

    Returns a ``BlockDiagonalStructure`` with the per-neuron factors if
    the problem is block-diagonal, or ``None`` otherwise. Call sites use
    a non-None return as a green light to dispatch to the block-diagonal
    filter path.

    Requirements for detection to succeed:

    1. ``design_matrix`` has shape ``(n_time, n_neurons, n_state)`` with
       ``n_state = n_neurons * block_size`` — i.e., the multi-neuron
       block-diagonal shape produced by
       ``PlaceFieldModel._build_block_diagonal``.
    2. For every time bin, each neuron row ``design_matrix[t, j]`` is
       zero outside the slice ``[j*block_size : (j+1)*block_size]``, and
       the non-zero slice is identical *across neurons* (same shared
       spline basis used by every neuron).
    3. ``transition_matrix``, ``process_cov``, and ``init_cov`` are all
       block-diagonal with matching ``block_size``.
    4. All neurons share the same per-neuron transition and process
       matrices — i.e., every diagonal block of ``A`` (and of ``Q``) is
       equal to the block-0 slice. Heterogeneous per-neuron dynamics
       are not detected as block-diagonal even though they technically
       qualify — that case would require storing
       ``(n_neurons, block_size, block_size)`` A/Q arrays and is deferred
       until there's a caller that needs it.

    This is pure detection logic with no side effects — always safe to
    call, returns None on any shape mismatch or non-block-diagonal input.

    Tolerance model
    ---------------
    The off-block-zero and block-equality checks use a relative-absolute
    mix: ``effective_atol = atol * max(1, max|mat|)``. This scales the
    absolute ``atol`` (default ``1e-10``) up for matrices with large
    entries (e.g. transition matrices with O(1) entries still use the
    absolute floor, while covariance matrices with O(1e6) entries get a
    proportionally larger tolerance). For f32 callers, pass a larger
    ``atol`` — typical values are ``1e-6`` to ``1e-5``.

    Parameters
    ----------
    init_mean : Array, shape (n_state,)
    init_cov : Array, shape (n_state, n_state)
    transition_matrix : Array, shape (n_state, n_state)
    process_cov : Array, shape (n_state, n_state)
    design_matrix : Array
        If ndim == 2, treated as single-neuron ``(n_time, n_state)`` and
        returns ``None`` (no block structure to exploit).
        If ndim == 3, must be ``(n_time, n_neurons, n_state)``.
    atol : float, default=1e-10
        Absolute tolerance floor; scaled by ``max(1, max|mat|)`` per-check
        for relative robustness to matrix magnitude.

    Returns
    -------
    BlockDiagonalStructure or None
    """
    # Single-neuron design matrices are (n_time, n_state), not
    # (n_time, 1, n_state). The block-diagonal filter gives no benefit
    # at n_neurons=1 (block_size == n_state, same as the dense filter),
    # so return None to dispatch to dense.
    if design_matrix.ndim != 3:
        return None

    _n_time, n_neurons, n_state = design_matrix.shape
    if n_neurons < 2:
        # Genuinely single-neuron problem — dense filter is optimal.
        return None
    if n_state % n_neurons != 0:
        return None
    block_size = n_state // n_neurons

    def _scaled_atol(mat: Array) -> float:
        """Relative-absolute tolerance scaled by matrix magnitude."""
        max_abs = float(jnp.max(jnp.abs(mat))) if mat.size > 0 else 0.0
        return atol * max(1.0, max_abs)

    # Check state-level arrays are block-diagonal with the inferred size.
    if not _is_block_diagonal(
        init_cov, n_neurons, block_size, atol=_scaled_atol(init_cov)
    ):
        return None
    if not _is_block_diagonal(
        transition_matrix,
        n_neurons,
        block_size,
        atol=_scaled_atol(transition_matrix),
    ):
        return None
    if not _is_block_diagonal(
        process_cov, n_neurons, block_size, atol=_scaled_atol(process_cov)
    ):
        return None

    # Check the design matrix: for each neuron j, design_matrix[:, j, :]
    # must be zero outside the slice [j*block_size : (j+1)*block_size].
    dm_atol = _scaled_atol(design_matrix)
    for j in range(n_neurons):
        start = j * block_size
        end = start + block_size
        # All columns except the neuron's own slice
        if start > 0:
            left_cols = design_matrix[:, j, :start]
            if float(jnp.max(jnp.abs(left_cols))) > dm_atol:
                return None
        if end < n_state:
            right_cols = design_matrix[:, j, end:]
            if float(jnp.max(jnp.abs(right_cols))) > dm_atol:
                return None

    # Extract A_block and Q_block from the (0, 0) slice of the full matrices.
    A_block = transition_matrix[:block_size, :block_size]
    Q_block = process_cov[:block_size, :block_size]

    # CRITICAL: verify that ALL diagonal blocks of A and Q equal the block-0
    # slice. Without this check, the block-diagonal filter would silently
    # apply block-0's A and Q to every neuron, even if the EM M-step wrote
    # back per-neuron-different diagonal blocks (e.g. due to floating-point
    # non-associativity in the sufficient statistics). Such a mismatch
    # would not show up as "not block-diagonal" in the earlier check —
    # it would pass _is_block_diagonal and then produce wrong results in
    # the vmap.
    a_atol = _scaled_atol(transition_matrix)
    q_atol = _scaled_atol(process_cov)
    for j in range(1, n_neurons):
        s, e = j * block_size, (j + 1) * block_size
        a_block_j = transition_matrix[s:e, s:e]
        q_block_j = process_cov[s:e, s:e]
        if float(jnp.max(jnp.abs(a_block_j - A_block))) > a_atol:
            return None
        if float(jnp.max(jnp.abs(q_block_j - Q_block))) > q_atol:
            return None

    # Per-neuron init_mean: split the concatenated mean into n_neurons
    # slices of length block_size.
    init_means_per_neuron = init_mean.reshape(n_neurons, block_size)

    # Per-neuron init_cov: extract the diagonal blocks. Unlike A and Q,
    # init_cov is allowed to have DIFFERENT diagonal blocks per neuron
    # (the warm-start fits a separate stationary GLM per neuron and
    # produces a block-diagonal cov where each block is that neuron's
    # Laplace posterior).
    init_covs_per_neuron = jnp.stack(
        [
            init_cov[
                j * block_size : (j + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            for j in range(n_neurons)
        ]
    )

    # Z_base: neuron 0's own slice is (n_time, block_size). Verify that
    # all other neurons share the same basis. If they differ (e.g.,
    # per-neuron custom basis), fall back to the dense filter.
    Z_base = design_matrix[:, 0, 0:block_size]
    for j in range(1, n_neurons):
        neuron_slice = design_matrix[:, j, j * block_size : (j + 1) * block_size]
        if float(jnp.max(jnp.abs(neuron_slice - Z_base))) > dm_atol:
            return None

    return BlockDiagonalStructure(
        A_block=A_block,
        Q_block=Q_block,
        init_means_per_neuron=init_means_per_neuron,
        init_covs_per_neuron=init_covs_per_neuron,
        Z_base=Z_base,
        n_neurons=n_neurons,
        block_size=block_size,
    )


def _build_block_structure_from_traced(
    init_mean: Array,
    init_cov: Array,
    transition_matrix: Array,
    process_cov: Array,
    design_matrix: Array,
    n_neurons: int,
    block_size: int,
) -> BlockDiagonalStructure:
    """Build a BlockDiagonalStructure from traced arrays using pure slicing.

    Unlike ``_detect_block_diagonal_problem`` — which uses host-side
    ``float()`` conversions for tolerance scaling and thus cannot run
    inside ``jax.jit`` / ``jax.grad`` — this helper assumes the caller
    has ALREADY verified block-diagonal structure (at fit entry time,
    with concrete arrays) and simply extracts the per-neuron factors
    via static slicing. It runs safely inside the jit boundary because
    ``n_neurons`` and ``block_size`` are Python integers (not traced
    values) and all slicing indices are compile-time constants.

    Used by ``stochastic_point_process_filter``'s auto-dispatch path:
    the caller runs ``_detect_block_diagonal_problem`` once at fit
    entry to get ``n_neurons`` / ``block_size``, then passes those
    integers into the jit-compiled loss function. Inside the loss
    function, the traced (init_mean, init_cov, A, Q, Z) arrays may
    differ from the detection-time values (e.g., the SGD optimizer
    has updated them), but they're guaranteed to preserve the same
    block-diagonal structure as long as:
      - the caller uses a block-diagonal parameterization (e.g.,
        ``process_noise_structure="diagonal"``), AND
      - ``update_transition_matrix=False`` (no off-block entries
        can be introduced by EM's M-step).

    The model layer is responsible for calling _detect_block_diagonal_
    problem at BOTH fit entry AND after each EM M-step to catch any
    structural regression.

    Parameters
    ----------
    init_mean : Array, shape (n_neurons * block_size,)
    init_cov : Array, shape (n_state, n_state)
    transition_matrix : Array, shape (n_state, n_state)
    process_cov : Array, shape (n_state, n_state)
    design_matrix : Array, shape (n_time, n_neurons, n_state)
    n_neurons : int
    block_size : int

    Returns
    -------
    BlockDiagonalStructure
    """
    nb = block_size
    # A_block: extract the (0, 0) diagonal block of transition_matrix.
    # Inside jit this is a static slice on the traced matrix.
    A_block = transition_matrix[:nb, :nb]
    Q_block = process_cov[:nb, :nb]

    # Per-neuron init_mean: reshape the concatenated state vector.
    init_means_per_neuron = init_mean.reshape(n_neurons, nb)

    # Per-neuron init_cov: extract the diagonal blocks via static slicing.
    # ``n_neurons`` and ``nb`` are Python ints (compile-time constants),
    # so ``init_cov[j*nb:(j+1)*nb, ...]`` resolves to a static slice at
    # trace time. Using a Python for-loop over ``range(n_neurons)``
    # produces one slice op per neuron — fine for realistic n_neurons
    # (up to ~64) because the loop is unrolled at trace time.
    init_covs_per_neuron = jnp.stack(
        [
            init_cov[
                j * nb : (j + 1) * nb,
                j * nb : (j + 1) * nb,
            ]
            for j in range(n_neurons)
        ]
    )

    # Z_base: neuron 0's own slice is (n_time, block_size). All neurons
    # share the same Z_base by the block-diagonal contract, so neuron 0
    # is representative.
    Z_base = design_matrix[:, 0, 0:nb]

    return BlockDiagonalStructure(
        A_block=A_block,
        Q_block=Q_block,
        init_means_per_neuron=init_means_per_neuron,
        init_covs_per_neuron=init_covs_per_neuron,
        Z_base=Z_base,
        n_neurons=n_neurons,
        block_size=nb,
    )


def log_conditional_intensity(design_matrix: ArrayLike, params: ArrayLike) -> Array:
    """Computes the log conditional intensity for a point process.

    This is the default linear log-intensity function: log(λ) = Z @ x.

    For single-neuron models, this returns a scalar.
    For multi-neuron models, design_matrix should be (n_neurons, n_params)
    so this returns (n_neurons,) log-intensities.

    Parameters
    ----------
    design_matrix : ArrayLike, shape (n_params,) or (n_neurons, n_params)
        Design matrix (Z_k) used in the intensity function.
        For single neuron: (n_params,) row vector.
        For multi-neuron: (n_neurons, n_params) matrix.
    params : ArrayLike, shape (n_params,)
        Parameters (latent state) for the intensity function.

    Returns
    -------
    Array, shape () or (n_neurons,)
        Log conditional intensity (log(λ_k)).
        Scalar for single-neuron, (n_neurons,) for multi-neuron.
    """
    return jnp.asarray(design_matrix) @ jnp.asarray(params)


def _logdet_psd(mat: Array, diagonal_boost: float = 1e-9) -> Array:
    """Log-determinant of a PSD matrix via Cholesky (stabilized).

    Uses ``logdet(A) = 2 * sum(log(diag(chol(A))))`` rather than
    ``sum(log(eigvalsh(A)))``. Cholesky is ~3-5x faster than eigvalsh
    for small-to-moderate PSD matrices and dominates the runtime of
    the Laplace-EKF filter's inner scan body — benchmarks show ~1.5-1.7x
    overall filter speedup on T=10k, d=36.

    Stability is ensured by adding ``diagonal_boost * I`` before
    factoring (a uniform eigenvalue shift) rather than clipping
    individual eigenvalues — the two strategies are equivalent for
    well-conditioned matrices and the shift is slightly more conservative
    for pathological ones. The caller is expected to hand in a PSD
    matrix; Kalman-filter covariances and posterior-precision matrices
    are PSD by construction.

    Parameters
    ----------
    mat : Array, shape (n, n)
        Symmetric positive semi-definite matrix.
    diagonal_boost : float
        Uniform eigenvalue shift for numerical stability before Cholesky.

    Returns
    -------
    Array
        Scalar log-determinant.
    """
    n = mat.shape[-1]
    stabilized = symmetrize(mat) + diagonal_boost * jnp.eye(n, dtype=mat.dtype)
    chol = jnp.linalg.cholesky(stabilized)
    return 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))


def _safe_expected_count(
    log_rate: Array,
    dt: float,
    min_log_count: float = -20.0,
    max_log_count: float = 20.0,
) -> Array:
    """Convert log-rate in Hz to expected count per bin with overflow protection."""
    dt_array = jnp.asarray(dt, dtype=log_rate.dtype)
    log_count = log_rate + jnp.log(dt_array)
    return jnp.exp(jnp.clip(log_count, min_log_count, max_log_count))


def _point_process_laplace_update(
    one_step_mean: Array,
    one_step_cov: Array,
    spike_indicator_t: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
    diagonal_boost: float = 1e-9,
    grad_log_intensity_func: Optional[Callable[[Array], Array]] = None,
    include_laplace_normalization: bool = True,
    max_newton_iter: int = 1,
    line_search_beta: float = 0.5,
    max_log_count: float = 20.0,
) -> tuple[Array, Array, Array]:
    """Single point-process Laplace-EKF update for multiple neurons.

    Performs a Bayesian update of the latent state posterior given observed
    spike counts, using a Gaussian (Laplace) approximation to the posterior.
    The approximation is built via **Fisher scoring** (expected Hessian /
    statistical linearization) rather than full Newton-Raphson with the
    observed Hessian.

    This is the core math for point-process observation updates, factored
    out to be reusable by both the non-switching and switching filters.

    The observation model is:
        y_n ~ Poisson(exp(log_intensity_func(x)[n]) * dt)

    Fisher scoring vs full Newton
    -----------------------------
    The posterior precision is built as

        P_post = P_prior + J' diag(lambda * dt) J

    where ``J`` is the Jacobian of ``log_intensity_func`` w.r.t. the state
    and ``lambda`` is the conditional intensity. This is a sum of PSD
    matrices, so ``P_post`` is PSD by construction and requires no
    eigenvalue stabilization.

    Full Newton would additionally subtract the observed Hessian correction
    ``sum_n (y_n - lambda_n * dt) * d^2(log lambda_n)/dx^2``, which is
    indefinite in general and can produce wildly large steps at non-MAP
    points. For **linear** log-intensities (``log lambda = Z @ x``, the
    default via :func:`log_conditional_intensity`) the second derivative is
    zero and Fisher scoring is mathematically identical to full Newton.
    For **nonlinear** intensities (e.g. KDE rate maps in
    :class:`PositionDecoder`), Fisher scoring produces better-conditioned,
    more stable updates. This matches the approach used in dynamax and
    generalized linear model IRLS.

    Parameters
    ----------
    one_step_mean : Array, shape (n_latent,)
        Predicted mean from dynamics: A @ m_{t-1}
    one_step_cov : Array, shape (n_latent, n_latent)
        Predicted covariance: A @ P_{t-1} @ A.T + Q
    spike_indicator_t : Array, shape (n_neurons,)
        Spike counts at time t for all neurons
    dt : float
        Time bin width in seconds
    log_intensity_func : Callable[[Array], Array]
        Function mapping state (n_latent,) to log-intensities (n_neurons,).
        Should return log(lambda) where lambda is firing rate in Hz.
    diagonal_boost : float, default=1e-9
        Small value added to precision matrix diagonal for numerical stability.
    grad_log_intensity_func : Callable[[Array], Array] | None, optional
        Pre-computed gradient function (Jacobian) of log_intensity_func.
        If None, computed via jax.jacfwd(log_intensity_func).
        Passing pre-computed functions can improve compilation speed when
        this function is called repeatedly inside a JIT-compiled context.
    include_laplace_normalization : bool, default=True
        If True, include the Laplace normalization and prior terms to approximate
        log p(y_t | y_{1:t-1}). If False, return the plug-in log-likelihood
        at the posterior mode without normalization.
    max_newton_iter : int, default=1
        Maximum number of Fisher scoring iterations. Use > 1 with line search
        for numerical stability with large spike counts (e.g., many neurons).
        (Named ``max_newton_iter`` for backwards compatibility; the inner
        iterations are Fisher steps, not full Newton.)
    line_search_beta : float, default=0.5
        Step size reduction factor for backtracking line search. Only used
        when max_newton_iter > 1. At each iteration, step size is halved
        until the negative log-posterior decreases or minimum alpha reached.
    max_log_count : float, default=20.0
        Ceiling on ``log(rate * dt)`` applied inside ``_safe_expected_count``
        to prevent overflow when the Fisher step produces implausibly large
        intensities. Callers should set this to a physiologically motivated
        value, e.g. ``log(max_firing_rate_hz * dt)``. The default of 20.0
        corresponds to ~2.4e9 Hz at dt=0.2s, which is high enough to mask
        pathological outlier bins; tighter ceilings (e.g. ``log(500 * dt)``)
        catch them.
    Returns
    -------
    posterior_mean : Array, shape (n_latent,)
        Updated state mean after incorporating spike observations
    posterior_cov : Array, shape (n_latent, n_latent)
        Updated state covariance after incorporating spike observations
    log_likelihood : Array
        Approximate log p(y_t | y_{1:t-1}) using a Laplace expansion (scalar array).

    Notes
    -----
    The Laplace approximation uses the predicted mean as the expansion point
    for a single Fisher scoring step. For multiple neurons, the gradients
    and Jacobians are summed across neurons.

    For Poisson likelihood with log-link:
        log p(y | x) = sum_n [y_n * log(lambda_n * dt) - lambda_n * dt - log(y_n!)]
        gradient = sum_n [(y_n - lambda_n * dt) * d(log_lambda_n)/dx]
        Hessian = sum_n [(y_n - lambda_n * dt) * d^2(log_lambda_n)/dx^2
                        - lambda_n * dt * (d(log_lambda_n)/dx)^T @ (d(log_lambda_n)/dx)]

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    """
    # Compute gradient of log-intensity function. The Hessian is NOT used
    # because we use Fisher scoring (expected Hessian) rather than the full
    # observed Hessian. For Poisson with log link, the Fisher information is
    # J' diag(lambda * dt) J, which is PSD by construction — no stabilization
    # of the posterior precision is required. The observed Hessian adds a
    # (y - lambda*dt) * d^2(log lambda)/dx^2 correction that can be indefinite
    # at non-MAP points; dropping it is the standard approach used in
    # dynamax, glmnet-style IRLS, and textbook Fisher scoring.
    #
    # For linear log-intensity (log lambda = Z @ x), the second derivative is
    # zero so Fisher scoring is mathematically identical to full Newton.
    # For nonlinear intensities (e.g. KDE rate maps in PositionDecoder),
    # Fisher scoring produces better-conditioned updates because it
    # avoids inverting a precision whose PSD-ness relies on jitter.
    if grad_log_intensity_func is None:
        grad_log_intensity_func = jax.jacfwd(log_intensity_func)
    grad_log_intensity = grad_log_intensity_func

    # Prior precision via psd_solve for numerical stability
    n_latent = one_step_mean.shape[0]
    identity = jnp.eye(n_latent)
    prior_precision = psd_solve(one_step_cov, identity, diagonal_boost=diagonal_boost)

    def _neg_log_posterior(x: Array) -> Array:
        """Negative log-posterior for line search."""
        log_lambda = log_intensity_func(x)
        cond_int = _safe_expected_count(log_lambda, dt, max_log_count=max_log_count)
        # Poisson log-likelihood (ignoring constant log(y!) term)
        # No floor needed: _safe_expected_count guarantees cond_int >= exp(-20) > 0
        log_lik = jnp.sum(spike_indicator_t * jnp.log(cond_int) - cond_int)
        # Gaussian prior log-probability (ignoring constant)
        delta = x - one_step_mean
        log_prior = -0.5 * delta @ (prior_precision @ delta)
        return -(log_lik + log_prior)

    def _fisher_step_at(x: Array) -> tuple[Array, Array, Array]:
        """Compute Fisher-scoring step and posterior precision at point x.

        Uses the expected Hessian (Fisher information) rather than the
        observed Hessian:

            -E[H_log_likelihood] = J' diag(lambda * dt) J    [PSD]

        Combined with the prior precision this gives a posterior precision
        that is PSD by construction:

            post_prec = prior_precision + J' diag(lambda * dt) J
        """
        log_lambda = log_intensity_func(x)
        conditional_intensity = _safe_expected_count(
            log_lambda, dt, max_log_count=max_log_count
        )
        innovation = spike_indicator_t - conditional_intensity
        jacobian = grad_log_intensity(x)

        # Likelihood gradient (same as full Newton)
        likelihood_gradient = jacobian.T @ innovation

        # Prior gradient: -prior_precision @ (x - one_step_mean)
        prior_gradient = -prior_precision @ (x - one_step_mean)

        # Full posterior gradient
        gradient = likelihood_gradient + prior_gradient

        # Fisher information (expected negative Hessian of log-likelihood).
        # J' diag(cond_int) J is a sum of rank-1 PSD terms. Adding the PSD
        # prior precision gives a PSD posterior precision — no stabilization
        # of indefiniteness is required.
        fisher_info = jacobian.T @ (conditional_intensity[:, None] * jacobian)
        post_prec = symmetrize(prior_precision + fisher_info)

        # Fisher-scoring direction
        delta = psd_solve(post_prec, gradient, diagonal_boost=diagonal_boost)
        return delta, post_prec, gradient

    def _line_search_step(carry, _):
        """One iteration of Fisher scoring with backtracking line search."""
        x, _ = carry
        delta, _, _ = _fisher_step_at(x)
        current_loss = _neg_log_posterior(x)

        # Backtracking line search
        def _backtrack(alpha_carry, _):
            alpha, _ = alpha_carry
            new_x = x + alpha * delta
            new_loss = _neg_log_posterior(new_x)
            improved = new_loss < current_loss
            new_alpha = jnp.where(improved, alpha, alpha * line_search_beta)
            return (new_alpha, improved), None

        (final_alpha, _), _ = jax.lax.scan(
            _backtrack, (jnp.array(1.0), jnp.array(False)), None, length=10
        )
        new_x = x + final_alpha * delta

        # Recompute precision at accepted point for consistency
        _, new_post_prec, _ = _fisher_step_at(new_x)
        return (new_x, new_post_prec), None

    # Initialize at prior mean
    x = one_step_mean

    if max_newton_iter == 1:
        # Single-step Fisher scoring (no line search overhead).
        # Evaluate at prior mean (one_step_mean), so prior gradient is zero.
        log_lambda = log_intensity_func(one_step_mean)
        conditional_intensity = _safe_expected_count(
            log_lambda, dt, max_log_count=max_log_count
        )
        innovation = spike_indicator_t - conditional_intensity
        jacobian = grad_log_intensity(one_step_mean)
        # Likelihood gradient only; prior gradient = -P^{-1}(x - m) = 0 at x = m
        likelihood_gradient = jacobian.T @ innovation
        prior_gradient = jnp.zeros_like(likelihood_gradient)
        gradient = likelihood_gradient + prior_gradient
        fisher_info = jacobian.T @ (conditional_intensity[:, None] * jacobian)
        posterior_precision = symmetrize(prior_precision + fisher_info)
        posterior_mean = one_step_mean + psd_solve(
            posterior_precision, gradient, diagonal_boost=diagonal_boost
        )
    else:
        # Iterative Fisher scoring with line search
        (posterior_mean, posterior_precision), _ = jax.lax.scan(
            _line_search_step, (x, prior_precision), None, length=max_newton_iter
        )

    # Posterior covariance via psd_solve. No post-hoc stabilization needed
    # because posterior_precision is PSD by construction (sum of two PSD
    # matrices). psd_solve itself adds a small diagonal boost for Cholesky
    # conditioning.
    posterior_cov = symmetrize(psd_solve(
        posterior_precision, identity, diagonal_boost=diagonal_boost
    ))

    # Log-likelihood at posterior mode (approximate)
    log_lambda_mode = log_intensity_func(posterior_mean)
    conditional_intensity_mode = _safe_expected_count(
        log_lambda_mode, dt, max_log_count=max_log_count
    )
    log_likelihood = jnp.sum(
        jax.scipy.stats.poisson.logpmf(spike_indicator_t, conditional_intensity_mode)
    )

    if include_laplace_normalization:
        # Laplace correction: log p(y) ≈ log p(y|x*) + log p(x*) + 0.5 log|P_post|
        # Constant terms (d/2 * log 2π) are omitted since they cancel across states.
        delta = posterior_mean - one_step_mean
        quad = delta @ (prior_precision @ delta)
        logdet_prior = _logdet_psd(one_step_cov, diagonal_boost)
        logdet_post = _logdet_psd(posterior_cov, diagonal_boost)
        log_prior = -0.5 * quad - 0.5 * logdet_prior
        log_likelihood = log_likelihood + log_prior + 0.5 * logdet_post

    return posterior_mean, posterior_cov, log_likelihood


def stochastic_point_process_filter(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = True,
    max_log_count: float = 20.0,
    validate_inputs: bool = True,
    block_n_neurons: Optional[int] = None,
    block_size: Optional[int] = None,
    force_dense: bool = False,
) -> tuple[Array, Array, Array]:
    """Applies a Stochastic State Point Process Filter (SSPPF).

    This filter estimates a time-varying latent state ($x_k$) based on
    point process observations ($y_k$). It assumes a linear Gaussian state
    transition and a point process observation model where the conditional
    intensity $\\lambda_k$ depends on the state.

    $$ x_k = A x_{k-1} + w_k, \\quad w_k \\sim N(0, Q) $$
    $$ \\lambda_{n,k} = f(x_k, Z_k)_n $$
    $$ y_{n,k} \\sim \\text{Poisson}(\\lambda_{n,k} \\Delta t) $$

    The filter uses a local Gaussian approximation (Laplace-EKF approach)
    at each update step, utilizing the gradient and Hessian of the
    log-likelihood. It implements a single Newton-Raphson like step
    per time bin.

    Multi-Neuron Support
    --------------------
    The filter supports multiple neurons sharing a common latent state:

    - spike_indicator: (n_time, n_neurons) - spike counts for each neuron
    - log_conditional_intensity(Z_k, x_k) should return (n_neurons,)

    For backwards compatibility, single-neuron inputs work as before:
    - spike_indicator: (n_time,) is internally promoted to (n_time, 1)
    - scalar log-intensity output is wrapped to (1,)

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
        Initial mean of the latent state ($x_0$).
    init_covariance_params : ArrayLike, shape (n_params, n_params)
        Initial covariance of the latent state ($P_0$).
    design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_params)
        Design matrix ($Z_k$) used in the intensity function.
        Shape depends on the log_conditional_intensity function.
        For multi-neuron with default linear intensity, use (n_time, n_neurons, n_params).
    spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
        Observed spike counts or indicators ($y_k$).
        For single neuron: (n_time,)
        For multiple neurons: (n_time, n_neurons)
    dt : float
        Time step size ($\\Delta t$).
    transition_matrix : ArrayLike, shape (n_params, n_params)
        State transition matrix ($A$).
    process_cov : ArrayLike, shape (n_params, n_params)
        Process noise covariance ($Q$).
    log_conditional_intensity : callable
        Function `log_lambda(Z_k, x_k)` returning the log conditional
        intensity. Should return (n_neurons,) array for multi-neuron case,
        or scalar for single-neuron.
    include_laplace_normalization : bool, default=True
        If True, include Laplace normalization and prior terms in the
        marginal log-likelihood. If False, return the plug-in log-likelihood
        at the posterior mode without normalization.
    max_log_count : float, default=20.0
        Ceiling on ``log(rate * dt)`` used by the Laplace update's safe
        exponentiation to prevent overflow on pathological spike counts.
        Pass a physiologically motivated value (e.g.
        ``log(max_firing_rate_hz * dt)``) to catch outlier bins that would
        otherwise drive the Fisher step into a catastrophic region. The
        default of 20.0 corresponds to ~2.4e9 Hz at dt=0.2s and is kept
        for backwards compatibility with existing callers.

    Returns
    -------
    posterior_mean : Array, shape (n_time, n_params)
        Filtered posterior means ($x_{k|k}$).
    posterior_variance : Array, shape (n_time, n_params, n_params)
        Filtered posterior covariances ($P_{k|k}$).
    marginal_log_likelihood : Array
        Total log-likelihood of the observations given the model (scalar array).

    Notes
    -----
    For multiple neurons, the observation log-likelihood term is the sum
    of independent Poisson log-pmfs:
        log p(y_t | x_t) = sum_n log Poisson(y_{n,t} | lambda_{n,t} * dt)
    When ``include_laplace_normalization`` is True, prior and normalization
    terms are added to approximate the marginal log-likelihood.

    The filter aggregates information from all neurons to update the shared
    latent state. More neurons provide more information, reducing posterior
    uncertainty.

    Numerical precision
    -------------------
    This filter requires **float64** for reliable long-sequence fits.
    In float32, accumulated roundoff in the covariance propagation can
    drive the posterior covariance below PSD after ~250-5000 time bins
    (depending on ``init_covariance_params`` conditioning), producing
    silent NaN output. Enable float64 BEFORE importing this module::

        import jax
        jax.config.update("jax_enable_x64", True)

    The entry-point validation (``validate_inputs=True``, default) warns
    when the dtype-and-conditioning combination is at risk and raises
    when ``init_covariance_params`` is not positive definite.

    References
    ----------
    [1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
      Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
      Neural Computation 16, 971-998 (2004).
    """
    # Convert to arrays
    init_mean_params = jnp.asarray(init_mean_params)
    init_covariance_params = jnp.asarray(init_covariance_params)
    design_matrix = jnp.asarray(design_matrix)
    spike_indicator = jnp.asarray(spike_indicator)
    transition_matrix = jnp.asarray(transition_matrix)
    process_cov = jnp.asarray(process_cov)

    # Block-diagonal dispatch (opt-in via block_n_neurons / block_size).
    # The caller is responsible for calling _detect_block_diagonal_problem
    # ONCE at fit entry time (outside jax.jit / jax.grad) to determine
    # these integers. Inside the jit boundary they are Python constants,
    # not traced, so we can use them for static slicing of the traced
    # (init_mean, init_cov, A, Q, design_matrix) arrays into per-neuron
    # factors. This is safe because block extraction is pure array
    # slicing — no host-side float() conversions.
    if (
        block_n_neurons is not None
        and block_size is not None
        and not force_dense
    ):
        if spike_indicator.ndim == 1:
            raise ValueError(
                "block_n_neurons / block_size can only be passed for "
                "multi-neuron (2D spike_indicator) inputs."
            )
        # Shape guard: block_n_neurons * block_size must match the total
        # state dimension. A mismatch means the caller passed wrong
        # detection integers or the model state shape changed between
        # detection and dispatch.
        expected_state_dim = block_n_neurons * block_size
        if init_covariance_params.shape[-1] != expected_state_dim:
            raise ValueError(
                f"block dispatch shape mismatch: block_n_neurons="
                f"{block_n_neurons} * block_size={block_size} = "
                f"{expected_state_dim}, but init_cov has shape "
                f"{init_covariance_params.shape}. Re-run "
                f"_detect_block_diagonal_problem to refresh the "
                f"dispatch integers."
            )
        structure = _build_block_structure_from_traced(
            init_mean_params,
            init_covariance_params,
            transition_matrix,
            process_cov,
            design_matrix,
            block_n_neurons,
            block_size,
        )
        return _stochastic_point_process_filter_block_diagonal(
            structure,
            spike_indicator,
            dt,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
        )

    # Numerical sanity check. Raises on non-PSD init_cov, warns on f32 +
    # long T + ill-conditioned configurations. Gated behind
    # ``validate_inputs`` so tight inner loops (e.g. SGD) can pass False
    # after a single validation at the top of fit_sgd. stacklevel=4 so
    # the warning points at the user's call site: user -> fit_sgd ->
    # _sgd_loss_fn -> stochastic_point_process_filter -> _validate.
    if validate_inputs:
        _validate_filter_numerics(
            init_covariance_params, n_time=spike_indicator.shape[0], stacklevel=4
        )

    # Promote single-neuron spike_indicator to (n_time, 1) for consistent handling
    single_neuron = spike_indicator.ndim == 1
    if single_neuron:
        spike_indicator = spike_indicator[:, None]

    # Pre-compute gradient function outside the scan.
    # Parameterized by design_matrix_t to avoid recreating jax.jacfwd each step.
    def _log_intensity_with_design(design_matrix_t, x):
        log_lambda = log_conditional_intensity(design_matrix_t, x)
        return jnp.atleast_1d(log_lambda)

    # Gradient w.r.t. x (argnums=1), keeping design_matrix_t as parameter.
    # No Hessian is needed — Fisher scoring uses only first-order info.
    _grad_log_intensity = jax.jacfwd(_log_intensity_with_design, argnums=1)

    def _step(
        params_prev: tuple[Array, Array, Array],
        args: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array]]:
        """Point Process Adaptive Filter update step."""
        # Unpack previous parameters
        mean_prev, variance_prev, marginal_log_likelihood = params_prev
        design_matrix_t, spike_indicator_t = args

        # One-step prediction. Symmetrize variance_prev BEFORE the
        # congruence so any accumulated asymmetry from the previous step
        # isn't amplified by A @ ... @ A^T. Re-symmetrize AFTER the
        # addition to clean up any residual asymmetry introduced by
        # finite-precision arithmetic. This belt-and-suspenders approach
        # is cheap (two 0.5*(M + M.T) ops per step) and makes the
        # covariance propagation more robust to f32 roundoff.
        one_step_mean = transition_matrix @ mean_prev
        variance_prev_sym = symmetrize(variance_prev)
        one_step_covariance = symmetrize(
            transition_matrix @ variance_prev_sym @ transition_matrix.T
            + process_cov
        )

        # Create log_intensity_func that captures design_matrix_t
        def log_intensity_func(x):
            return _log_intensity_with_design(design_matrix_t, x)

        # Grad closure capturing design_matrix_t; uses the pre-computed
        # jacfwd function from outside the scan.
        def grad_log_intensity_func(x):
            return _grad_log_intensity(design_matrix_t, x)

        # Fisher-scoring Laplace update (no Hessian needed)
        posterior_mean, posterior_covariance, log_lik = _point_process_laplace_update(
            one_step_mean,
            one_step_covariance,
            spike_indicator_t,
            dt,
            log_intensity_func,
            grad_log_intensity_func=grad_log_intensity_func,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
        )

        marginal_log_likelihood += log_lik

        return (posterior_mean, posterior_covariance, marginal_log_likelihood), (
            posterior_mean,
            posterior_covariance,
        )

    marginal_log_likelihood = jnp.array(0.0)
    (_, _, marginal_log_likelihood), (
        filtered_mean,
        filtered_cov,
    ) = jax.lax.scan(
        _step,
        (init_mean_params, init_covariance_params, marginal_log_likelihood),
        (design_matrix, spike_indicator),
    )

    return filtered_mean, filtered_cov, marginal_log_likelihood


def _run_forward_block_diagonal(
    structure: BlockDiagonalStructure,
    spike_indicator: Array,
    dt: float,
    include_laplace_normalization: bool = True,
    max_log_count: float = 20.0,
) -> tuple[Array, Array, Array]:
    """Run the per-neuron forward Laplace-EKF filter in block form.

    Shared helper for ``_stochastic_point_process_filter_block_diagonal``
    and ``_stochastic_point_process_smoother_block_diagonal``. Both
    consumers need access to the per-neuron forward posteriors —
    the filter reassembles them into dense form immediately, while
    the smoother feeds them to the backward pass before reassembly.

    Parameters
    ----------
    structure : BlockDiagonalStructure
    spike_indicator : Array, shape (n_time, n_neurons)
    dt : float
    include_laplace_normalization : bool, default=True
    max_log_count : float, default=20.0

    Returns
    -------
    fwd_means : Array, shape (n_neurons, n_time, block_size)
        Per-neuron forward posterior means at each time step.
    fwd_covs : Array, shape (n_neurons, n_time, block_size, block_size)
        Per-neuron forward posterior covariances.
    lls_per_neuron : Array, shape (n_neurons,)
        Per-neuron marginal log-likelihood (sum over time steps).
    """
    A_block = structure.A_block
    Q_block = structure.Q_block

    def _step_one_neuron(carry, args):
        mean_prev, cov_prev, ll_acc = carry
        z_row_t, y_t = args
        one_step_mean = A_block @ mean_prev
        cov_prev_sym = symmetrize(cov_prev)
        one_step_cov = symmetrize(
            A_block @ cov_prev_sym @ A_block.T + Q_block
        )
        spike_as_vec = jnp.atleast_1d(y_t)

        def _lin(x_block: Array) -> Array:
            return jnp.atleast_1d(z_row_t @ x_block)

        def _grad(_x_block: Array) -> Array:
            # Analytical gradient: d(log_intensity)/dx = z_row_t.
            # Safe because _detect_block_diagonal_problem's contract
            # requires a Z_base-block design matrix, which produces a
            # strictly linear log_intensity. The block-diagonal path
            # cannot receive a nonlinear intensity (e.g. PositionDecoder's
            # KDE rate map) — detection rejects ndim==2 design matrices.
            return z_row_t[None, :]

        post_mean, post_cov, log_lik_step = _point_process_laplace_update(
            one_step_mean,
            one_step_cov,
            spike_as_vec,
            dt,
            _lin,
            grad_log_intensity_func=_grad,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
        )
        return (
            (post_mean, post_cov, ll_acc + log_lik_step),
            (post_mean, post_cov),
        )

    def _run_one_neuron(init_mean_j: Array, init_cov_j: Array, spikes_j: Array):
        init_carry = (init_mean_j, init_cov_j, jnp.array(0.0, dtype=init_mean_j.dtype))
        (_, _, ll_j), (means_j, covs_j) = jax.lax.scan(
            _step_one_neuron,
            init_carry,
            (structure.Z_base, spikes_j),
        )
        return means_j, covs_j, ll_j

    return jax.vmap(_run_one_neuron, in_axes=(0, 0, 1))(
        structure.init_means_per_neuron,
        structure.init_covs_per_neuron,
        spike_indicator,
    )


def _assemble_block_diagonal_covs(
    covs_per_neuron: Array,
    n_neurons: int,
    block_size: int,
) -> Array:
    """Reassemble (n_neurons, n_time, nb, nb) blocks into a dense
    (n_time, n_state, n_state) block-diagonal matrix via a single
    broadcasted scatter.

    Shared between the filter and smoother for cov / cross_cov reassembly.
    """
    n_state = n_neurons * block_size
    block_offsets = jnp.arange(n_neurons) * block_size
    within_block = jnp.arange(block_size)
    row_idx = block_offsets[:, None] + within_block[None, :]
    col_idx = row_idx

    def _assemble_one(covs_at_t: Array) -> Array:
        dense = jnp.zeros((n_state, n_state), dtype=covs_at_t.dtype)
        return dense.at[row_idx[:, :, None], col_idx[:, None, :]].set(covs_at_t)

    # Input: (n_neurons, n_time_dim, nb, nb). Transpose to put time first.
    covs_time_first = jnp.transpose(covs_per_neuron, (1, 0, 2, 3))
    return jax.vmap(_assemble_one)(covs_time_first)


def _stochastic_point_process_filter_block_diagonal(
    structure: BlockDiagonalStructure,
    spike_indicator: Array,
    dt: float,
    include_laplace_normalization: bool = True,
    max_log_count: float = 20.0,
) -> tuple[Array, Array, Array]:
    """Block-diagonal Laplace-EKF filter via vmapped per-neuron scans.

    Consumes a ``BlockDiagonalStructure`` (from
    ``_detect_block_diagonal_problem``) and runs ``n_neurons`` independent
    filters in parallel via ``jax.vmap``. Each per-neuron filter operates
    at ``d=block_size`` state dimension rather than ``d=n_neurons*block_size``,
    giving ~``n_neurons^2`` speedup on the per-step Cholesky.

    Mathematical equivalence
    ------------------------
    For a block-diagonal problem, the dense filter's posterior decomposes
    into ``n_neurons`` independent per-neuron posteriors. Specifically:

    - ``A @ P @ A^T + Q`` is block-diagonal → each block is
      ``A_block @ P_j @ A_block^T + Q_block`` computed independently.
    - ``J^T diag(rate*dt) J`` (Fisher info) for neuron j depends only
      on that neuron's basis slice and its own weights, so it contributes
      only to block j of the posterior precision.
    - The Laplace normalization's ``logdet(P_prior)`` for a block-
      diagonal matrix equals ``sum_j logdet(P_j_prior)``, so per-neuron
      logdet contributions sum correctly to the full-problem logdet.
    - Same for ``logdet(P_post)``.
    - The quadratic form ``(x* - m)^T P_prior^{-1} (x* - m)`` also
      decomposes into a sum over neurons because ``P_prior`` is
      block-diagonal.

    Therefore: per-neuron ``log p(y_t^j | y_{1:t-1})`` contributions
    sum to the full-problem marginal log-likelihood. This is mathematically
    identical (not approximately) to the dense filter's output on a
    block-diagonal problem.

    Output shape compatibility
    --------------------------
    To be a drop-in replacement for the dense filter, this function
    returns ``filtered_mean`` of shape ``(n_time, n_state)`` and
    ``filtered_cov`` of shape ``(n_time, n_state, n_state)`` — both
    reassembled from the per-neuron trajectories. This materializes the
    dense cov every time step, which is wasteful for very large
    ``n_neurons``. A future optimization could return a ``BlockCovariance``
    view that lazily materializes; for now the full dense form is
    returned so downstream callers (``PlaceFieldModel.predict_rate_map``,
    EM M-step, confidence intervals) work unchanged.

    Parameters
    ----------
    structure : BlockDiagonalStructure
        Per-neuron factored filter problem.
    spike_indicator : Array, shape (n_time, n_neurons)
        Spike counts per neuron per time bin. Must be 2D (not 1D);
        callers with single-neuron problems should dispatch to the
        dense filter, not this function.
    dt : float
        Time bin width.
    include_laplace_normalization : bool, default=True
    max_log_count : float, default=20.0

    Returns
    -------
    filtered_mean : Array, shape (n_time, n_neurons * block_size)
        Concatenated per-neuron posterior means.
    filtered_cov : Array, shape (n_time, n_neurons * block_size, n_neurons * block_size)
        Block-diagonal posterior covariance (dense-form for API compat).
    marginal_log_likelihood : Array, scalar
        Total log-likelihood summed across neurons.
    """
    n_neurons = structure.n_neurons
    nb = structure.block_size
    n_state = n_neurons * nb
    n_time = structure.Z_base.shape[0]

    # Forward pass via the shared per-neuron scan helper.
    means_per_neuron, covs_per_neuron, lls_per_neuron = _run_forward_block_diagonal(
        structure,
        spike_indicator,
        dt,
        include_laplace_normalization=include_laplace_normalization,
        max_log_count=max_log_count,
    )
    # means_per_neuron: (n_neurons, n_time, nb)
    # covs_per_neuron: (n_neurons, n_time, nb, nb)
    # lls_per_neuron: (n_neurons,)

    # Reassemble concatenated filtered_mean of shape (n_time, n_state).
    filtered_mean = jnp.transpose(means_per_neuron, (1, 0, 2)).reshape(
        n_time, n_state
    )

    # Reassemble dense block-diagonal filtered_cov via the shared scatter
    # helper. A future optimization could return a BlockCovariance view
    # to avoid the O(n_state^2) zero-padding per time step.
    filtered_cov = _assemble_block_diagonal_covs(covs_per_neuron, n_neurons, nb)

    # Marginal log-likelihood: sum across neurons. For block-diagonal
    # problems, log p(y_t | y_{1:t-1}) decomposes as a sum of per-neuron
    # contributions — the Laplace normalization's logdet of a block-
    # diagonal matrix equals the sum of its blocks' logdets, and the
    # quadratic form similarly decomposes.
    marginal_ll = jnp.sum(lls_per_neuron)

    return filtered_mean, filtered_cov, marginal_ll


def _stochastic_point_process_smoother_block_diagonal(
    structure: BlockDiagonalStructure,
    spike_indicator: Array,
    dt: float,
    include_laplace_normalization: bool = True,
    max_log_count: float = 20.0,
    return_filtered: bool = False,
) -> tuple[Array, ...]:
    """Block-diagonal RTS smoother via vmapped per-neuron backward pass.

    Runs the block-diagonal forward filter
    (``_stochastic_point_process_filter_block_diagonal``) to get
    per-neuron filtered posteriors, then runs the standard RTS backward
    pass INDEPENDENTLY per neuron via jax.vmap. The smoother is
    observation-model-agnostic — it operates only on Gaussian moments —
    so the block-diagonal decomposition propagates through the backward
    pass without any additional algebra.

    Mathematical equivalence
    ------------------------
    The RTS backward step ``_kalman_smoother_update`` uses only:

    - Per-time-step filter mean and covariance (per-neuron in the block form)
    - ``transition_matrix`` (per-neuron ``A_block``)
    - ``process_cov`` (per-neuron ``Q_block``)

    All of these are block-diagonal by construction. The smoother gain
    ``J_t = P_{t|t} A^T (A P_{t|t} A^T + Q)^{-1}`` has block-diagonal
    structure when ``P_{t|t}``, ``A``, and ``Q`` are all block-diagonal,
    so the backward update decomposes neuron-by-neuron. Running the
    backward pass per-neuron is mathematically identical to running it
    on the full dense state.

    Output shape compatibility
    --------------------------
    Returns dense ``(n_time, n_state, n_state)`` covariances and
    ``(n_time - 1, n_state, n_state)`` cross-covariances, reassembled
    from per-neuron blocks via the same scatter-into-zero-matrix pattern
    as the forward filter. Downstream callers (EM M-step, confidence
    intervals) see the same API as the dense smoother.

    Parameters
    ----------
    structure : BlockDiagonalStructure
    spike_indicator : Array, shape (n_time, n_neurons)
    dt : float
    include_laplace_normalization : bool, default=True
    max_log_count : float, default=20.0
    return_filtered : bool, default=False
        If True, also return the filtered mean and covariance.

    Returns
    -------
    smoother_mean : Array, shape (n_time, n_state)
    smoother_cov : Array, shape (n_time, n_state, n_state)
    smoother_cross_cov : Array, shape (n_time - 1, n_state, n_state)
    marginal_log_likelihood : Array, scalar
    filtered_mean, filtered_cov : Array (optional, if return_filtered=True)
    """
    n_neurons = structure.n_neurons
    nb = structure.block_size
    n_state = n_neurons * nb
    A_block = structure.A_block
    Q_block = structure.Q_block

    # Forward pass via the shared per-neuron helper. Returns per-neuron
    # moments (n_neurons, n_time, nb) that the backward pass can consume
    # directly, without going through the dense reassembly.
    fwd_means, fwd_covs, lls_per_neuron = _run_forward_block_diagonal(
        structure,
        spike_indicator,
        dt,
        include_laplace_normalization=include_laplace_normalization,
        max_log_count=max_log_count,
    )
    # fwd_means: (n_neurons, n_time, nb)
    # fwd_covs: (n_neurons, n_time, nb, nb)

    # Backward RTS smoother pass per neuron. _kalman_smoother_update is
    # observation-model-agnostic — it uses only A, Q, and the filtered
    # Gaussian moments. For block-diagonal A and Q, the backward pass
    # decomposes into independent per-neuron smoothers, exactly matching
    # the dense filter's backward pass on the block-diagonal problem.

    def _backward_step(carry, args):
        next_smoother_mean, next_smoother_cov = carry
        filter_mean, filter_cov = args
        sm, sc, scc = _kalman_smoother_update(
            next_smoother_mean,
            next_smoother_cov,
            filter_mean,
            filter_cov,
            Q_block,
            A_block,
        )
        return (sm, sc), (sm, sc, scc)

    def _run_backward_one_neuron(means_j, covs_j):
        # Initial carry: the last-time-step filtered posterior.
        (_, _), (sm_rev, sc_rev, scc_rev) = jax.lax.scan(
            _backward_step,
            (means_j[-1], covs_j[-1]),
            (means_j[:-1], covs_j[:-1]),
            reverse=True,
        )
        # Append the last time step's filter posterior (no backward update)
        sm_full = jnp.concatenate((sm_rev, means_j[-1][None]))
        sc_full = jnp.concatenate((sc_rev, covs_j[-1][None]))
        return sm_full, sc_full, scc_rev

    (
        smoother_means_per_neuron,
        smoother_covs_per_neuron,
        smoother_cross_covs_per_neuron,
    ) = jax.vmap(_run_backward_one_neuron)(fwd_means, fwd_covs)
    # smoother_means_per_neuron: (n_neurons, n_time, nb)
    # smoother_covs_per_neuron:  (n_neurons, n_time, nb, nb)
    # smoother_cross_covs_per_neuron: (n_neurons, n_time - 1, nb, nb)

    n_time = structure.Z_base.shape[0]

    # Reassemble concatenated smoother_mean (n_time, n_state).
    smoother_mean = jnp.transpose(smoother_means_per_neuron, (1, 0, 2)).reshape(
        n_time, n_state
    )

    # Reassemble block-diagonal covariances (n_time, n_state, n_state)
    # and cross-covariances (n_time-1, n_state, n_state) via the shared
    # scatter helper. Matches the dense smoother API.
    #
    # Cross-cov index convention: _kalman_smoother_update returns
    # ``J_t @ P_{t+1|T}``, which is the smoothed lag-one cross-cov
    # ``P_{t, t+1|T}`` (current-next). EM M-step consumers expect this
    # convention, so we preserve it in the block reassembly — see the
    # dense smoother's "# Lag-one cross covariance P_{t, t+1|T}" comment
    # in ``_kalman_smoother_update``.
    smoother_cov = _assemble_block_diagonal_covs(
        smoother_covs_per_neuron, n_neurons, nb
    )
    smoother_cross_cov = _assemble_block_diagonal_covs(
        smoother_cross_covs_per_neuron, n_neurons, nb
    )

    marginal_ll = jnp.sum(lls_per_neuron)

    if return_filtered:
        filtered_mean = jnp.transpose(fwd_means, (1, 0, 2)).reshape(n_time, n_state)
        filtered_cov = _assemble_block_diagonal_covs(fwd_covs, n_neurons, nb)
        return (
            smoother_mean,
            smoother_cov,
            smoother_cross_cov,
            marginal_ll,
            filtered_mean,
            filtered_cov,
        )
    return smoother_mean, smoother_cov, smoother_cross_cov, marginal_ll


def stochastic_point_process_smoother(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
    include_laplace_normalization: bool = True,
    return_filtered: bool = False,
    max_log_count: float = 20.0,
    validate_inputs: bool = True,
    block_n_neurons: Optional[int] = None,
    block_size: Optional[int] = None,
    force_dense: bool = False,
) -> tuple[Array, ...]:
    """Applies a Stochastic State Point Process Smoother (SSPPS).

    This smoother estimates a time-varying latent state ($x_k$) based on
    point process observations ($y_k$) using a Kalman smoother approach.
    It first applies a stochastic point process filter to obtain the filtered
    means and covariances, and then applies a Kalman smoother to refine these estimates.

    $$ x_k = A x_{k-1} + w_k, \\quad w_k \\sim N(0, Q) $$
    $$ \\lambda_{n,k} = f(x_k, Z_k)_n $$
    $$ y_{n,k} \\sim \\text{Poisson}(\\lambda_{n,k} \\Delta t) $$

    Multi-Neuron Support
    --------------------
    The smoother supports multiple neurons sharing a common latent state.
    See `stochastic_point_process_filter` for details on multi-neuron inputs.

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
        Initial mean of the latent state ($x_0$).
    init_covariance_params : ArrayLike, shape (n_params, n_params)
        Initial covariance of the latent state ($P_0$).
    design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_params)
        Design matrix ($Z_k$) used in the intensity function.
        Shape depends on the log_conditional_intensity function.
    spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
        Observed spike counts or indicators ($y_k$).
        For single neuron: (n_time,)
        For multiple neurons: (n_time, n_neurons)
    dt : float
        Time step size ($\\Delta t$).
    transition_matrix : ArrayLike, shape (n_params, n_params)
        State transition matrix ($A$).
    process_cov : ArrayLike, shape (n_params, n_params)
        Process noise covariance ($Q$).
    log_conditional_intensity : callable
        Function `log_lambda(Z_k, x_k)` returning the log conditional
        intensity. Should return (n_neurons,) for multi-neuron case.
    include_laplace_normalization : bool, default=True
        If True, include Laplace normalization and prior terms in the
        marginal log-likelihood. If False, return the plug-in log-likelihood
        at the posterior mode without normalization.

    Returns
    -------
    smoother_mean : Array, shape (n_time, n_params)
        Smoothed posterior means ($x_{k|T}$).
    smoother_cov : Array, shape (n_time, n_params, n_params)
        Smoothed posterior covariances ($P_{k|T}$).
    smoother_cross_cov : Array, shape (n_time - 1, n_params, n_params)
        Smoothed cross-covariances ($P_{k|T, k-1}$).
    marginal_log_likelihood : Array
        Total log-likelihood of the observations given the model (scalar array).

    Notes
    -----
    The smoother is observation-model agnostic - it operates only on the
    Gaussian posteriors from the filter. The multi-neuron handling is done
    entirely in the filter step.

    References
    ----------
    [1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998 (2004).
    """
    # Convert to arrays up-front so the dispatch branch can slice.
    init_mean_params = jnp.asarray(init_mean_params)
    init_covariance_params = jnp.asarray(init_covariance_params)
    design_matrix = jnp.asarray(design_matrix)
    spike_indicator = jnp.asarray(spike_indicator)
    transition_matrix = jnp.asarray(transition_matrix)
    process_cov = jnp.asarray(process_cov)

    # Block-diagonal dispatch: same opt-in contract as the filter.
    # See stochastic_point_process_filter's block-dispatch comment for
    # the full rationale. If both block_n_neurons and block_size are
    # provided (as Python ints) and force_dense is False, we short-
    # circuit to _stochastic_point_process_smoother_block_diagonal
    # which vmaps the forward and backward passes per-neuron.
    if (
        block_n_neurons is not None
        and block_size is not None
        and not force_dense
    ):
        if spike_indicator.ndim == 1:
            raise ValueError(
                "block_n_neurons / block_size can only be passed for "
                "multi-neuron (2D spike_indicator) inputs."
            )
        expected_state_dim = block_n_neurons * block_size
        if init_covariance_params.shape[-1] != expected_state_dim:
            raise ValueError(
                f"block dispatch shape mismatch: block_n_neurons="
                f"{block_n_neurons} * block_size={block_size} = "
                f"{expected_state_dim}, but init_cov has shape "
                f"{init_covariance_params.shape}. Re-run "
                f"_detect_block_diagonal_problem to refresh the "
                f"dispatch integers."
            )
        structure = _build_block_structure_from_traced(
            init_mean_params,
            init_covariance_params,
            transition_matrix,
            process_cov,
            design_matrix,
            block_n_neurons,
            block_size,
        )
        return _stochastic_point_process_smoother_block_diagonal(
            structure,
            spike_indicator,
            dt,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
            return_filtered=return_filtered,
        )

    filtered_mean, filtered_cov, marginal_log_likelihood = (
        stochastic_point_process_filter(
            init_mean_params,
            init_covariance_params,
            design_matrix,
            spike_indicator,
            dt,
            transition_matrix,
            process_cov,
            log_conditional_intensity,
            include_laplace_normalization=include_laplace_normalization,
            max_log_count=max_log_count,
            validate_inputs=validate_inputs,
        )
    )

    def _step(carry, args):
        (
            next_smoother_mean,
            next_smoother_cov,
        ) = carry

        filter_mean, filter_cov = args

        smoother_mean, smoother_cov, smoother_cross_cov = _kalman_smoother_update(
            next_smoother_mean,
            next_smoother_cov,
            filter_mean,
            filter_cov,
            process_cov,
            transition_matrix,
        )
        return (
            smoother_mean,
            smoother_cov,
        ), (
            smoother_mean,
            smoother_cov,
            smoother_cross_cov,
        )

    (_, _), (smoother_mean, smoother_cov, smoother_cross_cov) = jax.lax.scan(
        _step,
        (filtered_mean[-1], filtered_cov[-1]),
        (filtered_mean[:-1], filtered_cov[:-1]),
        reverse=True,
    )

    smoother_mean = jnp.concatenate((smoother_mean, filtered_mean[-1][None]))
    smoother_cov = jnp.concatenate((smoother_cov, filtered_cov[-1][None]))

    result = (smoother_mean, smoother_cov, smoother_cross_cov, marginal_log_likelihood)
    if return_filtered:
        return result + (filtered_mean, filtered_cov)
    return result


def kalman_maximization_step(
    smoother_mean: ArrayLike,
    smoother_cov: ArrayLike,
    smoother_cross_cov: ArrayLike,
) -> tuple[Array, Array, Array, Array]:
    """Maximization step for the Kalman filter.

    Parameters
    ----------
    smoother_mean : ArrayLike, shape (n_time, n_cont_states)
        smoother mean.
    smoother_cov : ArrayLike, shape (n_time, n_cont_states, n_cont_states)
        smoother covariance.
    smoother_cross_cov : ArrayLike, shape (n_time - 1, n_cont_states, n_cont_states)
        smoother cross-covariance.

    Returns
    -------
    transition_matrix : Array, shape (n_cont_states, n_cont_states)
        Transition matrix.
    process_cov : Array, shape (n_cont_states, n_cont_states)
        Process covariance.
    mean_init : Array, shape (n_cont_states,)
        Initial mean.
    cov_init : Array, shape (n_cont_states, n_cont_states)
        Initial covariance.

    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """
    smoother_mean = jnp.asarray(smoother_mean)
    smoother_cov = jnp.asarray(smoother_cov)
    smoother_cross_cov = jnp.asarray(smoother_cross_cov)

    n_time = smoother_mean.shape[0]

    # Compute intermediate expectation terms
    gamma = jnp.sum(smoother_cov, axis=0) + sum_of_outer_products(
        smoother_mean, smoother_mean
    )
    gamma1 = gamma - jnp.outer(smoother_mean[-1], smoother_mean[-1]) - smoother_cov[-1]
    gamma2 = gamma - jnp.outer(smoother_mean[0], smoother_mean[0]) - smoother_cov[0]
    beta = (
        smoother_cross_cov.sum(axis=0)
        + sum_of_outer_products(smoother_mean[:-1], smoother_mean[1:])
    ).T

    # Transition matrix
    transition_matrix = psd_solve(gamma1, beta.T).T

    # Process covariance
    process_cov = stabilize_covariance(
        (gamma2 - transition_matrix @ beta.T) / (n_time - 1),
        min_eigenvalue=1e-8,
    )

    # Initial mean and covariance
    init_mean = smoother_mean[0]
    init_cov = smoother_cov[0]

    return (
        transition_matrix,
        process_cov,
        init_mean,
        init_cov,
    )


def get_confidence_interval(
    posterior_mean: ArrayLike, posterior_covariance: ArrayLike, alpha: float = 0.01
) -> Array:
    """Get the confidence interval from the posterior covariance

    Parameters
    ----------
    posterior_mean : ArrayLike, shape (n_time, n_params)
    posterior_covariance : ArrayLike, shape (n_time, n_params, n_params)
    alpha : float, optional
        Confidence level, by default 0.01
    """
    posterior_mean = jnp.asarray(posterior_mean)
    posterior_covariance = jnp.asarray(posterior_covariance)
    z = jax.scipy.stats.norm.ppf(1 - alpha / 2)
    ci = z * jnp.sqrt(
        jnp.diagonal(posterior_covariance, axis1=-2, axis2=-1)
    )  # shape (n_time, n_params)

    return jnp.stack((posterior_mean - ci, posterior_mean + ci), axis=-1)


def steepest_descent_point_process_filter(
    init_mean_params: ArrayLike,
    x: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    epsilon: ArrayLike,
    log_receptive_field_model: Callable[[ArrayLike, ArrayLike], Array],
    max_log_count: float = 20.0,
) -> Array:
    """Steepest Descent Point Process Filter (SDPPF)

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
    x : ArrayLike, shape (n_time,)
        Continuous-valued input signal
    spike_indicator : ArrayLike, shape (n_time,)
        Spike count
    dt : float
        Time step
    epsilon : ArrayLike, shape (n_params, n_params)
        Learning rate
    log_receptive_field_model : callable
        Function that takes in `x` and parameters and returns the log spike rate
    max_log_count : float, default=20.0
        Ceiling on ``log(rate * dt)`` applied inside ``_safe_expected_count``
        to prevent overflow on pathological spike counts. Pass a physiologically
        motivated value (e.g. ``log(max_firing_rate_hz * dt)``) if you expect
        outlier bins. The default of 20.0 matches the Laplace-EKF filter's
        default for backwards compatibility.

    Returns
    -------
    posterior_mean : Array, shape (n_time, n_params)

    References
    ----------
    .. [1] Brown, E.N., Nguyen, D.P., Frank, L.M., Wilson, M.A., and Solo, V. (2001).
    An analysis of neural receptive field plasticity by point process adaptive filtering.
    Proceedings of the National Academy of Sciences 98, 12261–12266.
    https://doi.org/10.1073/pnas.201409398.

    .. [2] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
      Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
      Neural Computation 16, 971-998 (2004).

    Notes
    -----
    Equation in [1] is for the likelihood while in [2] it is for the log likelihood.
    This implementation follows the formulation in [2].

    """
    # Convert ArrayLike inputs to Array
    init_mean_params_arr: Array = jnp.asarray(init_mean_params)
    x_arr: Array = jnp.asarray(x)
    spike_indicator_arr: Array = jnp.asarray(spike_indicator)
    epsilon_arr: Array = jnp.asarray(epsilon)

    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)

    def _update(mean_prev: Array, args: tuple[Array, Array]) -> tuple[Array, Array]:
        """Steepest Descent Point Process Filter update step"""
        x_t, spike_indicator_t = args
        conditional_intensity = _safe_expected_count(
            log_receptive_field_model(x_t, mean_prev),
            dt,
            max_log_count=max_log_count,
        )
        innovation = spike_indicator_t - conditional_intensity
        one_step_grad = grad_log_receptive_field_model(x_t, mean_prev)
        posterior_mean = mean_prev + epsilon_arr @ one_step_grad * innovation

        return posterior_mean, posterior_mean

    return jax.lax.scan(_update, init_mean_params_arr, (x_arr, spike_indicator_arr))[1]


class PointProcessModel(SGDFittableMixin):
    """Point Process State-Space Model with EM fitting.

    Implements the Eden & Brown (2004) adaptive point process filter/smoother
    with EM algorithm for parameter estimation.

    Model:
        x_k = A @ x_{k-1} + w_k,  w_k ~ N(0, Q)
        n_{j,k} ~ Poisson(exp(log_intensity_func(Z_k, x_k)[j]) * dt)

    The EM algorithm estimates the state dynamics parameters (A, Q) while
    the latent states x_k are estimated via the E-step (filter/smoother).

    Multi-Neuron Support
    --------------------
    The model supports multiple neurons sharing a common latent state:

    - spike_indicator: (n_time, n_neurons) - spike counts for each neuron
    - log_intensity_func(Z_k, x_k) should return (n_neurons,) log-intensities

    For backwards compatibility, single-neuron inputs work as before:
    - spike_indicator: (n_time,) is internally promoted to (n_time, 1)
    - scalar log-intensity output is wrapped to (1,)

    Parameters
    ----------
    n_state_dims : int
        Dimension of the latent state.
    dt : float
        Time step size.
    transition_matrix : ArrayLike, optional
        Initial state transition matrix A. Default is identity (random walk).
    process_cov : ArrayLike, optional
        Initial process noise covariance Q.
    init_mean : ArrayLike, optional
        Initial state mean.
    init_cov : ArrayLike, optional
        Initial state covariance.
    log_intensity_func : callable, optional
        Function log_lambda(Z_k, x_k) returning log conditional intensity.
        Default is linear: Z_k @ x_k.
        For multi-neuron, should return (n_neurons,) array.
    update_transition_matrix : bool
        Whether to update A in M-step. Default True.
    update_process_cov : bool
        Whether to update Q in M-step. Default True.
    update_init_state : bool
        Whether to update initial state in M-step. Default True.

    Attributes
    ----------
    smoother_mean : Array
        Smoothed state estimates after fitting.
    smoother_cov : Array
        Smoothed state covariances after fitting.
    smoother_cross_cov : Array
        Smoothed cross-covariances after fitting.

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    """

    def __init__(
        self,
        n_state_dims: int,
        dt: float,
        transition_matrix: Optional[ArrayLike] = None,
        process_cov: Optional[ArrayLike] = None,
        init_mean: Optional[ArrayLike] = None,
        init_cov: Optional[ArrayLike] = None,
        log_intensity_func: Optional[Callable] = None,
        update_transition_matrix: bool = True,
        update_process_cov: bool = True,
        update_init_state: bool = True,
    ):
        self.n_state_dims = n_state_dims
        self.dt = dt

        # Initialize parameters
        if transition_matrix is None:
            self.transition_matrix = jnp.eye(n_state_dims)
        else:
            self.transition_matrix = jnp.asarray(transition_matrix)

        if process_cov is None:
            self.process_cov = jnp.eye(n_state_dims) * 1e-4
        else:
            self.process_cov = jnp.asarray(process_cov)

        if init_mean is None:
            self.init_mean = jnp.zeros(n_state_dims)
        else:
            self.init_mean = jnp.asarray(init_mean)

        if init_cov is None:
            self.init_cov = jnp.eye(n_state_dims)
        else:
            self.init_cov = jnp.asarray(init_cov)

        if log_intensity_func is None:
            self.log_intensity_func = log_conditional_intensity
        else:
            self.log_intensity_func = log_intensity_func

        # Update flags
        self.update_transition_matrix = update_transition_matrix
        self.update_process_cov = update_process_cov
        self.update_init_state = update_init_state

        # Results (populated after fit)
        self.smoother_mean: Optional[Array] = None
        self.smoother_cov: Optional[Array] = None
        self.smoother_cross_cov: Optional[Array] = None
        self.filtered_mean: Optional[Array] = None
        self.filtered_cov: Optional[Array] = None

    def _e_step(self, design_matrix: ArrayLike, spike_indicator: ArrayLike) -> float:
        """E-step: Run filter and smoother to estimate latent states.

        Parameters
        ----------
        design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_state_dims)
            Design matrix for the intensity function.
        spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Observed spike counts. Single neuron: (n_time,), multi-neuron: (n_time, n_neurons).

        Returns
        -------
        marginal_log_likelihood : float
        """
        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_log_likelihood,
        ) = stochastic_point_process_smoother(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self.log_intensity_func,
            # fit() validates once at entry; skip re-validation on each
            # EM iteration's E-step.
            validate_inputs=False,
        )

        # Also store filtered results
        self.filtered_mean, self.filtered_cov, _ = stochastic_point_process_filter(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self.log_intensity_func,
            validate_inputs=False,
        )

        return float(marginal_log_likelihood)

    def _m_step(self) -> None:
        """M-step: Update model parameters based on smoothed estimates."""
        if (
            self.smoother_mean is None
            or self.smoother_cov is None
            or self.smoother_cross_cov is None
        ):
            raise RuntimeError("Must run E-step before M-step")

        transition_matrix, process_cov, init_mean, init_cov = kalman_maximization_step(
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
        )

        if self.update_transition_matrix:
            self.transition_matrix = transition_matrix

        if self.update_process_cov:
            # kalman_maximization_step already applies stabilize_covariance
            # with min_eigenvalue=1e-8
            self.process_cov = process_cov

        if self.update_init_state:
            self.init_mean = init_mean
            self.init_cov = symmetrize(init_cov)

    def fit(
        self,
        design_matrix: ArrayLike,
        spike_indicator: ArrayLike,
        max_iter: int = 100,
        tolerance: float = 1e-4,
    ) -> list[float]:
        """Fit the model using the EM algorithm.

        Parameters
        ----------
        design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_state_dims)
            Design matrix for the intensity function.
            Shape depends on the log_intensity_func.
        spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Observed spike counts or indicators.
            For single neuron: (n_time,)
            For multiple neurons: (n_time, n_neurons)
        max_iter : int
            Maximum number of EM iterations.
        tolerance : float
            Convergence tolerance for relative change in log-likelihood.

        Returns
        -------
        log_likelihoods : list[float]
            Log-likelihood at each iteration.
        """
        design_matrix = jnp.asarray(design_matrix)
        spike_indicator = jnp.asarray(spike_indicator)

        # Numerical sanity check once at the top: validate PSD and warn
        # on f32 risk. Skips per-iteration re-validation in the E-step.
        _validate_filter_numerics(
            jnp.asarray(self.init_cov), n_time=spike_indicator.shape[0]
        )

        log_likelihoods: list[float] = []
        previous_log_likelihood = -jnp.inf

        for iteration in range(max_iter):
            # E-step
            current_log_likelihood = self._e_step(design_matrix, spike_indicator)
            log_likelihoods.append(current_log_likelihood)

            # Check convergence
            is_converged, is_increasing = check_converged(
                current_log_likelihood, previous_log_likelihood, tolerance
            )

            if not is_increasing:
                logger.warning(
                    f"Log-likelihood decreased at iteration {iteration + 1}!"
                )

            if is_converged:
                logger.info(f"Converged after {iteration + 1} iterations.")
                break

            # M-step
            self._m_step()

            logger.info(
                f"Iteration {iteration + 1}/{max_iter}\t"
                f"Log-Likelihood: {current_log_likelihood:.4f}\t"
                f"Change: {(current_log_likelihood - previous_log_likelihood):.6f}"
            )
            previous_log_likelihood = current_log_likelihood

        if len(log_likelihoods) == max_iter:
            logger.warning("Reached maximum iterations without converging.")

        return log_likelihoods

    # --- SGDFittableMixin protocol ---

    def fit_sgd(
        self,
        design_matrix: ArrayLike,
        spike_indicator: ArrayLike,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        design_matrix : ArrayLike
            Design matrix for the intensity function.
        spike_indicator : ArrayLike
            Observed spike counts.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps.
        convergence_tol : float or None
            If set, stop early when loss change < tol for 5 consecutive steps.

        Returns
        -------
        log_likelihoods : list of float
        """
        design_matrix = jnp.asarray(design_matrix)
        spike_indicator = jnp.asarray(spike_indicator)
        if spike_indicator.ndim == 1:
            spike_indicator = spike_indicator[:, None]
        self._sgd_n_time = spike_indicator.shape[0]
        self._sgd_design_matrix = design_matrix
        self._sgd_spike_indicator = spike_indicator

        # Numerical sanity check once at the top: validate PSD and warn
        # on f32 risk. The SGD loss_fn runs inside jax.jit, which cannot
        # call eigvalsh's host-side float() conversion, so per-step
        # re-validation is both wasteful and forbidden.
        _validate_filter_numerics(
            jnp.asarray(self.init_cov), n_time=spike_indicator.shape[0]
        )

        return super().fit_sgd(
            design_matrix, spike_indicator,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    @property
    def _n_timesteps(self) -> int:
        return self._sgd_n_time

    def _check_sgd_initialized(self) -> None:
        pass  # Parameters allocated at construction time

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            PSD_MATRIX,
            UNCONSTRAINED,
        )

        params: dict = {}
        spec: dict = {}

        if self.update_transition_matrix:
            params["transition_matrix"] = self.transition_matrix
            spec["transition_matrix"] = UNCONSTRAINED

        if self.update_process_cov:
            params["process_cov"] = self.process_cov
            spec["process_cov"] = PSD_MATRIX

        if self.update_init_state:
            params["init_mean"] = self.init_mean
            spec["init_mean"] = UNCONSTRAINED
            params["init_cov"] = self.init_cov
            spec["init_cov"] = PSD_MATRIX

        return params, spec

    def _sgd_loss_fn(
        self, params: dict, design_matrix: Array, spike_indicator: Array
    ) -> Array:
        A = params.get("transition_matrix", self.transition_matrix)
        Q = params.get("process_cov", self.process_cov)
        m0 = params.get("init_mean", self.init_mean)
        P0 = params.get("init_cov", self.init_cov)

        _, _, marginal_ll = stochastic_point_process_filter(
            init_mean_params=m0,
            init_covariance_params=P0,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=A,
            process_cov=Q,
            log_conditional_intensity=self.log_intensity_func,
            # fit_sgd validated once at the top; skip per-step
            # re-validation inside the jit'd loss fn.
            validate_inputs=False,
        )
        return -marginal_ll

    def _store_sgd_params(self, params: dict) -> None:
        if "transition_matrix" in params:
            self.transition_matrix = params["transition_matrix"]
        if "process_cov" in params:
            self.process_cov = params["process_cov"]
        if "init_mean" in params:
            self.init_mean = params["init_mean"]
        if "init_cov" in params:
            self.init_cov = params["init_cov"]

    def _finalize_sgd(
        self, design_matrix: Array, spike_indicator: Array
    ) -> None:
        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_ll,
        ) = stochastic_point_process_smoother(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self.log_intensity_func,
            validate_inputs=False,  # validated at fit_sgd entry
        )
        self.filtered_mean, self.filtered_cov, _ = stochastic_point_process_filter(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self.log_intensity_func,
            validate_inputs=False,  # validated at fit_sgd entry
        )
        self.log_likelihood_ = float(marginal_ll)

    def get_rate_estimate(
        self,
        design_matrix: ArrayLike,
        use_smoothed: bool = True,
        evaluate_at_all_positions: bool = True,
    ) -> Array:
        """Get the estimated firing rate using the model's log-intensity function.

        This method computes firing rates by evaluating the stored log_intensity_func,
        supporting both single-neuron and multi-neuron models with arbitrary
        (possibly nonlinear) intensity functions.

        Parameters
        ----------
        design_matrix : ArrayLike
            Design matrix to evaluate rate at. Shape depends on usage:
            - If evaluate_at_all_positions=True (default): (n_pos, n_state_dims)
              where n_pos can be n_time or any number of positions/conditions
            - If evaluate_at_all_positions=False: same shape as used during fit,
              e.g., (n_time, n_state_dims) for single-neuron or
              (n_time, n_neurons, n_state_dims) for multi-neuron

        use_smoothed : bool
            If True, use smoothed estimates; otherwise use filtered.

        evaluate_at_all_positions : bool, default=True
            If True, evaluate the rate at all positions in design_matrix for each
            time point, returning shape (n_time, n_pos) for single-neuron or
            (n_time, n_pos, n_neurons) for multi-neuron.
            If False, evaluate time-aligned: design_matrix[t] with state[t],
            returning shape (n_time,) for single-neuron or (n_time, n_neurons)
            for multi-neuron.

        Returns
        -------
        rate : Array
            Estimated firing rate in Hz. Shape depends on evaluate_at_all_positions
            and whether the model is single/multi-neuron (see above).

        Notes
        -----
        The rate is computed as exp(log_intensity_func(design_matrix, x)).
        This generalizes to arbitrary intensity functions, not just the default
        linear Z @ x.
        """
        if use_smoothed:
            if self.smoother_mean is None:
                raise RuntimeError("Model has not been fitted yet.")
            state_estimate = self.smoother_mean
        else:
            if self.filtered_mean is None:
                raise RuntimeError("Model has not been fitted yet.")
            state_estimate = self.filtered_mean

        design_matrix = jnp.asarray(design_matrix)

        if evaluate_at_all_positions:
            # Evaluate rate at all positions for each time point
            # For each (time, position) pair, compute log_intensity_func(design[pos], state[time])
            # vmap over positions (inner), then over times (outer)
            def rate_at_time(state_t):
                # For this time's state, evaluate at all positions
                return jax.vmap(lambda dm: self.log_intensity_func(dm, state_t))(
                    design_matrix
                )

            log_rate = jax.vmap(rate_at_time)(state_estimate)  # (n_time, n_pos, ...)
        else:
            # Time-aligned evaluation: design_matrix[t] with state_estimate[t]
            log_rate = jax.vmap(self.log_intensity_func)(design_matrix, state_estimate)

        rate = jnp.exp(log_rate)

        return rate

    def get_confidence_interval(
        self, alpha: float = 0.05, use_smoothed: bool = True
    ) -> Array:
        """Get confidence intervals for the state estimates.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CI).
        use_smoothed : bool
            If True, use smoothed estimates; otherwise use filtered.

        Returns
        -------
        ci : Array, shape (n_time, n_state_dims, 2)
            Lower and upper bounds of the confidence interval.
        """
        if use_smoothed:
            if self.smoother_mean is None or self.smoother_cov is None:
                raise RuntimeError("Model has not been fitted yet.")
            mean = self.smoother_mean
            cov = self.smoother_cov
        else:
            if self.filtered_mean is None or self.filtered_cov is None:
                raise RuntimeError("Model has not been fitted yet.")
            mean = self.filtered_mean
            cov = self.filtered_cov

        return get_confidence_interval(mean, cov, alpha=alpha)
