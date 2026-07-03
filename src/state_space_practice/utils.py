import warnings
from typing import Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from scipy.optimize import linear_sum_assignment

# Type alias for numeric values (scalars, numpy arrays, JAX arrays)
Numeric = Union[float, int, np.ndarray, jax.Array]


# ---------------------------------------------------------------------------
# Linear algebra utilities
# ---------------------------------------------------------------------------


def symmetrize(A: jax.Array) -> jax.Array:
    """Symmetrize one or more matrices by averaging each matrix with its transpose.

    Parameters
    ----------
    A : jax.Array
        A matrix or a batch of matrices to be symmetrized. The last two
        dimensions should be square matrices.

    Returns
    -------
    jax.Array
        The symmetrized matrix or batch of matrices, where each output matrix
        is (A + A.T) / 2.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> A = jnp.array([[1, 2], [3, 4]])
    >>> symmetrize(A)
    DeviceArray([[1. , 2.5],
                 [2.5, 4. ]], dtype=float32)

    """
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def psd_solve(
    A: jax.Array,
    b: jax.Array,
    diagonal_boost: float = 1e-9,
    relative_boost: float = 1e-12,
) -> jax.Array:
    """Solves a linear system Ax = b for positive semi-definite (PSD) matrices A.

    This function wraps a linear algebra solver, ensuring numerical stability
    by symmetrizing the input matrix A and adding a scaled diagonal shift
    before Cholesky. It is intended for use with PSD matrices, where
    ``assume_a="pos"`` can be safely set for performance.

    Stabilization shift
    -------------------
    The shift added to the diagonal is::

        effective = max(diagonal_boost, relative_boost * max|diag(A)|)

    Using ``max(absolute, relative * scale)`` is standard numerics
    practice. Rationale:

    - The absolute ``diagonal_boost`` handles matrices whose entries are
      O(1) or smaller; the ~1e-9 floor sits just above f64 cholesky
      precision and catches matrices that are numerically (but not
      structurally) positive semidefinite.
    - The ``relative_boost`` handles matrices whose entries span many
      orders of magnitude. A cov matrix with ``max_diag ~ 1e6`` needs a
      larger shift than ``1e-9`` to stabilize cholesky; a cov with
      ``max_diag ~ 1e-3`` would have ``1e-9`` be a relatively large
      perturbation of the signal. Relative scaling is scale-invariant
      and gives roughly f64-precision-level stability on any input.

    The default ``relative_boost=1e-12`` is approximately 1e4 * f64
    machine epsilon — small enough that the shift does not perturb
    near-singular f64 matrices (whose smallest eigenvalues may be at
    1e-10 to 1e-5) but still catches genuinely ill-conditioned matrices
    at large scales (where ``max_diag * 1e-12`` exceeds the absolute
    ``1e-9`` floor). Callers can raise ``relative_boost`` for aggressive
    regularization or lower it to 0.0 to disable relative scaling
    entirely and match the pre-adaptive behavior.

    Parameters
    ----------
    A : jax.Array
        The coefficient matrix, expected to be positive semi-definite.
    b : jax.Array
        The right-hand side vector or matrix.
    diagonal_boost : float, optional
        Absolute floor for the stabilization shift. Default is 1e-9.
    relative_boost : float, optional
        Coefficient of ``max|diag(A)|`` used as the relative component
        of the stabilization shift. Default is 1e-12 (~1e4 * f64 eps).
        Pass ``0.0`` to disable relative scaling entirely and match the
        pre-adaptive (absolute-only) behavior — useful for callers that
        need bit-for-bit reproducibility with older code.

    Returns
    -------
    jax.Array
        The solution x to the linear system Ax = b.

    """
    A_sym = symmetrize(A)
    n = A.shape[-1]
    # Use the maximum absolute diagonal entry as the reference scale for
    # the relative boost. For PSD matrices this is a tight bound on the
    # largest eigenvalue; for slightly non-PSD matrices (floating-point
    # drift) it is still a reasonable scale reference.
    max_diag = jnp.max(jnp.abs(jnp.diagonal(A_sym, axis1=-2, axis2=-1)))
    effective_boost = jnp.maximum(
        jnp.asarray(diagonal_boost, dtype=A.dtype),
        jnp.asarray(relative_boost, dtype=A.dtype) * max_diag,
    )
    return jax.scipy.linalg.solve(
        A_sym + effective_boost * jnp.eye(n, dtype=A.dtype),
        b,
        assume_a="pos",
    )


def project_psd(Q: jax.Array, min_eigenvalue: float = 1e-8) -> jax.Array:
    """Project a matrix onto the positive semi-definite cone.

    This function ensures the input matrix is positive semi-definite by:
    1. Computing its eigendecomposition
    2. Clipping eigenvalues to be at least `min_eigenvalue`
    3. Reconstructing the matrix from the clipped eigenvalues

    Parameters
    ----------
    Q : jax.Array
        A symmetric matrix to project onto the PSD cone. Shape (n, n).
    min_eigenvalue : float, optional
        Minimum eigenvalue to enforce. Default is 1e-8.

    Returns
    -------
    jax.Array
        The projected PSD matrix with all eigenvalues >= min_eigenvalue.
    """
    Q = symmetrize(Q)
    eigvals, eigvecs = jnp.linalg.eigh(Q)
    eigvals_clipped = jnp.maximum(eigvals, min_eigenvalue)
    projected = eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T
    return symmetrize(projected)


def stabilize_covariance(cov: jax.Array, min_eigenvalue: float = 1e-8) -> jax.Array:
    """Symmetrize a covariance-like matrix and project it to the PSD cone."""
    return project_psd(symmetrize(cov), min_eigenvalue=min_eigenvalue)


def shift_to_psd(cov: jax.Array, min_eigenvalue: float = 1e-8) -> jax.Array:
    r"""Lift a symmetric matrix to the PSD cone by a uniform eigenvalue shift.

    Returns ``cov + max(min_eigenvalue - lambda_min(cov), 0) * I`` -- the
    smallest isotropic diagonal shift that raises the minimum eigenvalue to at
    least ``min_eigenvalue``. For a matrix already PSD to within
    ``min_eigenvalue`` the shift is zero and this is the identity.

    Unlike :func:`stabilize_covariance` / :func:`project_psd`, this reads only
    ``lambda_min`` (via ``eigvalsh``) and never forms the eigenvector
    reconstruction ``V diag(f(lambda)) V^T``. That reconstruction has a gradient
    with ``1 / (lambda_i - lambda_j)`` terms that blow up to NaN when
    eigenvalues are degenerate -- which happens routinely for block-structured
    process covariances (e.g. the correlated-noise oscillator ``Q`` has paired
    eigenvalues). ``eigvalsh().min()`` keeps a finite gradient through such
    points, so this variant is safe to call **inside a differentiated SGD
    loss**; ``stabilize_covariance`` is for host-side (non-differentiated) use.

    The tradeoff is that the shift is isotropic (adds the same amount to every
    eigenvalue) rather than clipping only the offending ones, so it inflates the
    already-large eigenvalues too. Since it is exactly the identity whenever the
    matrix is PSD, this only affects the indefinite region, where it acts as a
    smooth barrier steering the optimizer back toward valid covariances.

    Parameters
    ----------
    cov : jax.Array
        A symmetric matrix. Shape (n, n).
    min_eigenvalue : float, optional
        Target lower bound on the minimum eigenvalue. Default is 1e-8.

    Returns
    -------
    jax.Array
        ``cov`` shifted so its minimum eigenvalue is at least
        ``min_eigenvalue``. Shape (n, n).
    """
    lambda_min = jnp.linalg.eigvalsh(cov).min()
    shift = jnp.maximum(min_eigenvalue - lambda_min, 0.0)
    return cov + shift * jnp.eye(cov.shape[-1], dtype=cov.dtype)


def debug_print_if(condition: jax.Array, fmt: str, **fmt_kwargs) -> None:
    """Fire ``jax.debug.print(fmt, **fmt_kwargs)`` only when ``condition`` is True.

    Wraps ``jax.lax.cond`` so callers don't have to spell out the
    ``(lambda: jax.debug.print(...), lambda: None)`` pattern at every
    silent-fallback site. The print branch fires when the predicate is
    True (i.e. when the *bad* condition holds), matching how the call
    site reads at the user's eye: "if `~is_valid`, print the warning."
    """
    jax.lax.cond(
        condition,
        lambda: jax.debug.print(fmt, **fmt_kwargs),
        lambda: None,
    )


def validate_choice_indices(choices: ArrayLike, n_options: int) -> None:
    """Host-side bounds check on discrete choice / category indices.

    JAX's out-of-range indexing is silent by default:
    ``jnp.zeros(K).at[i].set(1.0)`` and ``jax.nn.one_hot(i, K)`` both
    produce an all-zero vector when ``i`` is outside ``[0, K)`` rather
    than raising. Downstream filters / observation models then treat
    the step as "no observation" and leave the posterior unchanged,
    which looks like a normal result. This helper fails loudly before
    any JIT dispatch so out-of-range data is caught at the public API.
    """
    choices_np = np.asarray(choices)
    if choices_np.size == 0:
        return
    if not np.issubdtype(choices_np.dtype, np.number):
        raise ValueError("choices must be numeric category indices.")
    choices_float = choices_np.astype(float)
    if not np.all(np.isfinite(choices_float)):
        raise ValueError("choices must contain only finite category indices.")
    if not np.all(np.isclose(choices_float, np.round(choices_float))):
        raise ValueError("choices must contain integer-valued category indices.")
    if np.any(choices_np < 0) or np.any(choices_np >= n_options):
        raise ValueError(
            f"All choices must be in [0, {n_options}), "
            f"got range [{int(choices_float.min())}, {int(choices_float.max())}]. "
            f"JAX silently maps out-of-range indices to a zero indicator "
            f"vector (= no observation), so this would otherwise produce "
            f"a normal-looking result on bad data."
        )


def validate_count_array(
    counts: ArrayLike,
    name: str,
    *,
    allow_empty: bool = True,
) -> None:
    """Validate observed count data at public API boundaries.

    Count-valued observation models assume finite, non-negative integer
    counts.  Failing before JAX dispatch avoids silent float-to-int casts and
    invalid likelihood terms that otherwise look like normal model output.
    """
    counts_np = np.asarray(counts)
    if counts_np.size == 0:
        if allow_empty:
            return
        raise ValueError(f"{name} must contain at least one count.")
    if not np.issubdtype(counts_np.dtype, np.number):
        raise ValueError(f"{name} must be numeric count data.")
    counts_float = counts_np.astype(float)
    if not np.all(np.isfinite(counts_float)):
        raise ValueError(f"{name} must contain only finite counts.")
    if not np.all(counts_float >= 0):
        raise ValueError(f"{name} must contain non-negative counts.")
    if not np.all(np.isclose(counts_float, np.round(counts_float))):
        raise ValueError(f"{name} must contain integer-valued counts.")


def validate_scalar(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Validate a finite real scalar configuration value at a public boundary.

    Returns ``value`` coerced to ``float`` so callers can store the coerced
    result. Raises ``ValueError`` (rather than a bare ``TypeError``) on
    non-scalar, non-finite, or out-of-range input.
    """
    value_arr = np.asarray(value)
    if value_arr.shape != ():
        raise ValueError(f"{name} must be a scalar. Got shape {value_arr.shape}.")
    value_float = float(value_arr)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite. Got {value}.")
    if positive and value_float <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")
    if nonnegative and value_float < 0:
        raise ValueError(f"{name} must be non-negative. Got {value}.")
    return value_float


def _validate_filter_numerics(
    init_covariance: Array,
    n_time: int,
    stacklevel: int = 3,
    filter_name: str = "filter",
    measurement_cov: Optional[Array] = None,
    process_cov: Optional[Array] = None,
) -> None:
    """Validate covariance numerics + warn about f32 numerical risk.

    Shared by the Laplace-EKF point-process path
    (``stochastic_point_process_filter``) and the linear-Gaussian path
    (``kalman_filter`` / ``kalman_smoother``). Both families are Cholesky-
    based, so strictly-positive-definite ``init_cov`` and measurement covariance
    are hard requirements, process covariance must be positive semidefinite, and
    f32 + long-T is a documented risk.

    Raises
    ------
    ValueError
        If ``init_covariance`` is non-square, non-finite, not symmetric, or has
        a non-positive minimum eigenvalue. Same positive-definite check is
        applied to ``measurement_cov`` when supplied. ``process_cov`` is checked
        as positive semidefinite when supplied.

    Warns
    -----
    UserWarning
        If ``init_covariance.dtype`` is ``float32`` AND the problem is
        long enough / ill-conditioned enough that accumulated covariance
        roundoff is likely to drive the predicted covariance below PSD
        during the scan.

    Notes
    -----
    This helper runs at the top of each public entry point (``fit``,
    ``fit_sgd``, or the public filter/smoother wrappers) once per call,
    then inner call sites pass ``validate_inputs=False`` to skip
    re-validation. The ``eigvalsh → float(...)`` conversion used here
    is incompatible with ``jax.jit`` tracing, so inner calls must bypass
    the check entirely.
    """
    validate_covariance(
        init_covariance,
        name="init_covariance",
        require_positive_definite=True,
    )
    if measurement_cov is not None:
        validate_covariance(
            measurement_cov,
            name="measurement_cov",
            require_positive_definite=True,
        )
    if process_cov is not None:
        validate_covariance(
            process_cov,
            name="process_cov",
            require_positive_definite=False,
        )

    # Eigenvalue check. eigvalsh is O(d^3) but only runs once per filter
    # invocation, vs d^3 per scan step — negligible.
    init_cov_sym = symmetrize(init_covariance)
    eigs = jnp.linalg.eigvalsh(init_cov_sym)
    min_eig = float(eigs.min())
    max_eig = float(eigs.max())

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
        # the predict-step congruence (A @ P @ A^T + Q) and the Cholesky
        # update. Error-analysis constants: conservative ~sqrt(n_state)
        # factor, f32 machine epsilon ~1.2e-7. Accumulated over n_time
        # bins (random-walk model), total roundoff ~
        # n_time * sqrt(n_state) * eps * max_eig.
        f32_eps = 1.2e-7
        worst_roundoff = (
            float((n_time * n_state) ** 0.5) * f32_eps * max_eig
        )
        if worst_roundoff > 0.5 * min_eig:
            warnings.warn(
                f"{filter_name} running in float32 with "
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


def validate_covariance(
    covariance: Array,
    name: str = "covariance",
    *,
    require_positive_definite: bool = True,
    symmetry_atol: float = 1e-8,
    symmetry_rtol: float = 1e-6,
) -> None:
    """Validate a covariance matrix (or per-discrete-state stack) is symmetric PSD.

    Unlike :func:`_validate_filter_numerics` (which symmetrizes before its
    eigenvalue check and therefore cannot detect an asymmetric matrix), this
    validator checks symmetry on the *raw* matrix. It is the general-purpose
    covariance guard for public entry points that do not go through the
    kalman/place-field f32-warning path.

    Parameters
    ----------
    covariance : Array
        Either a single square matrix ``(d, d)`` or a stack of per-discrete-
        state matrices ``(d, d, n_states)`` (discrete-state axis last, per the
        project convention).
    name : str
        Field name used in error messages.
    require_positive_definite : bool, default True
        If True, require a strictly positive minimum eigenvalue (as the
        Cholesky-based filters need). If False, accept a positive-semidefinite
        matrix (minimum eigenvalue ``>= -symmetry_atol``).
    symmetry_atol, symmetry_rtol : float
        Absolute/relative tolerance for the ``C == C.T`` check.

    Raises
    ------
    ValueError
        If any slice is non-square, non-symmetric, or violates the eigenvalue
        floor. For a stacked input the offending discrete-state index is named.
    """
    arr = jnp.asarray(covariance)
    if arr.ndim == 2:
        slices: list[tuple[Optional[int], Array]] = [(None, arr)]
    elif arr.ndim == 3:
        slices = [(k, arr[..., k]) for k in range(arr.shape[-1])]
    else:
        raise ValueError(
            f"{name} must be a 2D matrix or a 3D per-state stack "
            f"(d, d, n_states), got shape {arr.shape}"
        )

    for state_ind, mat in slices:
        where = name if state_ind is None else f"{name}[..., {state_ind}]"
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"{where} must be a square 2D matrix, got shape {mat.shape}"
            )
        # Reject NaN/Inf up front: they slip past the symmetry check
        # (allclose treats inf == inf) and the eigenvalue floor (a NaN/Inf
        # eigenvalue is not <= 0), so a non-finite "covariance" would pass.
        if not bool(jnp.all(jnp.isfinite(mat))):
            raise ValueError(
                f"{where} has non-finite entries (NaN/Inf); a covariance "
                f"must be finite."
            )
        # Symmetry is checked on the RAW matrix: symmetrizing first would let a
        # non-symmetric "covariance" pass undetected (the exact defect that hid
        # in CorrelatedNoiseModel's process covariance).
        if not bool(
            jnp.allclose(mat, mat.T, rtol=symmetry_rtol, atol=symmetry_atol)
        ):
            asym = float(jnp.max(jnp.abs(mat - mat.T)))
            raise ValueError(
                f"{where} is not symmetric (max|C - C^T| = {asym:g}). A "
                f"covariance must be symmetric; an asymmetric matrix is not a "
                f"valid covariance and its Cholesky/eigen-decomposition is "
                f"ill-defined."
            )
        min_eig = float(jnp.linalg.eigvalsh(symmetrize(mat)).min())
        if require_positive_definite:
            invalid = min_eig <= 0.0
            kind = "positive definite"
        else:
            invalid = min_eig < -symmetry_atol
            kind = "positive semidefinite"
        if invalid:
            raise ValueError(
                f"{where} is not {kind} (min eigenvalue {min_eig:g}). "
                f"Covariance matrices in the Cholesky-based filters must be "
                f"{kind}; a rank-deficient or indefinite matrix will NaN on "
                f"the first step. Check the value you supplied."
            )


def validate_transition_matrix(
    transition_matrix: Array,
    name: str = "transition_matrix",
    *,
    atol: float = 1e-6,
) -> None:
    """Validate a row-stochastic transition matrix (rows non-negative, sum to 1).

    Parameters
    ----------
    transition_matrix : Array, shape (n_states, n_states)
        Discrete-state transition matrix, row-stochastic by convention
        (``T[i, j] = P(next = j | current = i)``).
    name : str
        Field name used in error messages.
    atol : float
        Absolute tolerance for the per-row sum-to-one check.

    Raises
    ------
    ValueError
        If the matrix is non-square, has negative entries, or has a row that
        does not sum to 1 within ``atol``.
    """
    arr = jnp.asarray(transition_matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be a square 2D matrix, got shape {arr.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise ValueError(f"{name} has non-finite entries (NaN/Inf).")
    if bool(jnp.any(arr < -atol)):
        raise ValueError(
            f"{name} has negative entries (min {float(arr.min()):g}); "
            f"transition probabilities must be non-negative."
        )
    row_sums = jnp.sum(arr, axis=1)
    if not bool(jnp.allclose(row_sums, 1.0, atol=atol)):
        worst = float(jnp.max(jnp.abs(row_sums - 1.0)))
        raise ValueError(
            f"{name} rows must sum to 1 (max deviation {worst:g}). Note the "
            f"row-stochastic convention T[i, j] = P(next=j | current=i); a "
            f"transposed matrix is a common cause of this error."
        )


def validate_probability_vector(
    probabilities: Array,
    name: str = "probabilities",
    *,
    atol: float = 1e-6,
) -> None:
    """Validate a probability vector (non-negative entries summing to 1).

    Parameters
    ----------
    probabilities : Array, shape (n_states,)
        Discrete probability vector.
    name : str
        Field name used in error messages.
    atol : float
        Absolute tolerance for the sum-to-one check.

    Raises
    ------
    ValueError
        If any entry is negative or the entries do not sum to 1 within ``atol``.
    """
    arr = jnp.asarray(probabilities)
    if not bool(jnp.all(jnp.isfinite(arr))):
        raise ValueError(f"{name} has non-finite entries (NaN/Inf).")
    if bool(jnp.any(arr < -atol)):
        raise ValueError(
            f"{name} has negative entries (min {float(arr.min()):g}); "
            f"probabilities must be non-negative."
        )
    total = float(jnp.sum(arr))
    if abs(total - 1.0) > atol:
        raise ValueError(
            f"{name} must sum to 1 (got {total:g}). Supply a normalized "
            f"probability vector."
        )


# ---------------------------------------------------------------------------
# Probability utilities
# ---------------------------------------------------------------------------

# Minimum probability threshold for numerical stability
_LOG_PROB_FLOOR = 1e-10
_LOG_FLOOR_VALUE = -23.0  # approximately log(1e-10)
_DISCRETE_PROB_STABILITY_FLOOR = 1e-10


def divide_safe(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """Divide two arrays, returning 0.0 where denominator is exactly 0.0.

    Guards against division-by-zero for exact floating-point zeros only
    (e.g., probability vectors with structural zeros). Does NOT guard
    against near-zero denominators, NaN, or Inf values.

    Parameters
    ----------
    numerator : jax.Array
    denominator : jax.Array

    Returns
    -------
    jax.Array
        ``numerator / denominator``, with 0.0 where ``denominator == 0.0``.
    """
    return jnp.where(denominator == 0.0, 0.0, numerator / denominator)


def safe_log(x: jax.Array) -> jax.Array:
    """Compute log(x) with numerical stability for small probabilities.

    Uses jnp.where to explicitly handle near-zero values rather than
    silently adding a small constant.

    Parameters
    ----------
    x : jax.Array
        Input array (typically probabilities).

    Returns
    -------
    jax.Array
        log(x) where x > _LOG_PROB_FLOOR, otherwise _LOG_FLOOR_VALUE.
    """
    return jnp.where(x > _LOG_PROB_FLOOR, jnp.log(x), _LOG_FLOOR_VALUE)


def stabilize_probability_vector(probabilities: jax.Array) -> jax.Array:
    """Prevent exact-zero probability lockout from numerical underflow.

    Applies a small floor to each element, then re-normalizes so the vector
    sums to 1. This ensures that no discrete state is permanently excluded
    once its probability underflows to zero.

    Parameters
    ----------
    probabilities : jax.Array, shape (n_states,)
        Probability vector (non-negative, ideally sums to 1).

    Returns
    -------
    jax.Array, shape (n_states,)
        Stabilized probability vector that sums to 1 with all entries
        >= ``_DISCRETE_PROB_STABILITY_FLOOR`` (before re-normalization).

    Notes
    -----
    If the input is all zeros (e.g. from complete underflow), every element
    is raised to the floor and re-normalization produces a uniform
    distribution. This is intentional for numerical robustness, but it is
    usually a sign of upstream numerical trouble — every iteration the
    filter's discrete-state posterior is reset toward uniform, erasing the
    evidence from observations. A ``jax.debug.print`` fires on the all-zero
    path so callers see it in filter/smoother logs without changing the
    function's return contract.
    """
    floor = jnp.asarray(_DISCRETE_PROB_STABILITY_FLOOR, dtype=probabilities.dtype)
    debug_print_if(
        jnp.all(probabilities <= 0),
        "utils.stabilize_probability_vector: input was all-zero (max={m}); "
        "falling back to uniform distribution. The filter's discrete-state "
        "posterior is being silently reset — expect the switching-model "
        "fit to ignore observations at this step.",
        m=jnp.max(probabilities),
    )
    stabilized = jnp.maximum(probabilities, floor)
    return stabilized / jnp.sum(stabilized)


def scale_likelihood(log_likelihood: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Scale the log likelihood to avoid numerical underflow.

    Parameters
    ----------
    log_likelihood : jax.Array
        Log likelihood values.

    Returns
    -------
    scaled_likelihood : jax.Array
        Scaled likelihood (exponentiated with max subtracted).
    ll_max : jax.Array
        Maximum log likelihood (scalar array).
    """
    ll_max = log_likelihood.max()
    ll_max = jnp.where(jnp.isfinite(ll_max), ll_max, 0.0)
    return jnp.exp(log_likelihood - ll_max), ll_max


def check_converged(
    log_likelihood: Numeric,
    previous_log_likelihood: Numeric,
    tolerance: float = 1e-4,
) -> tuple[bool, bool]:
    """We have converged if the slope of the log-likelihood function falls below 'tolerance',

    i.e., |f(t) - f(t-1)| / avg < tolerance,
    where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.

    Parameters
    ----------
    log_likelihood : float
        Current log likelihood
    previous_log_likelihood : float
        Previous log likelihood
    tolerance : float, optional
        threshold for similarity, by default 1e-4

    Returns
    -------
    is_converged : bool
        True if the relative change < tolerance.
    is_increasing : bool
        True if the relative decrease does not exceed tolerance.

    """
    # Handle infinite values (e.g., first iteration when previous is -inf)
    if not np.isfinite(previous_log_likelihood) or not np.isfinite(log_likelihood):
        # Can't be converged if either value is infinite
        # is_increasing is True if current is finite or greater
        is_increasing = log_likelihood >= previous_log_likelihood
        return False, bool(is_increasing)

    delta_log_likelihood = np.abs(log_likelihood - previous_log_likelihood)
    eps = np.finfo(float).eps
    avg_log_likelihood = (np.abs(log_likelihood) + np.abs(previous_log_likelihood) + eps) / 2

    relative_change = (log_likelihood - previous_log_likelihood) / avg_log_likelihood
    is_increasing = relative_change >= -tolerance
    is_converged = (delta_log_likelihood / avg_log_likelihood) < tolerance

    return bool(is_converged), bool(is_increasing)


def make_discrete_transition_matrix(
    diag: Array, n_discrete_states: int
) -> Array:
    """Build a row-stochastic transition matrix from diagonal values.

    Off-diagonal elements distribute the remaining probability mass
    equally among other states.

    Parameters
    ----------
    diag : Array, shape (n_discrete_states,)
        Diagonal (self-transition) probabilities for each state.
    n_discrete_states : int
        Number of discrete states.

    Returns
    -------
    transition_matrix : Array, shape (n_discrete_states, n_discrete_states)
        Row-stochastic transition matrix.
    """
    if n_discrete_states == 1:
        return jnp.array([[1.0]])

    transition_matrix = jnp.diag(diag)
    off_diag = (1.0 - diag) / (n_discrete_states - 1.0)
    transition_matrix = (
        transition_matrix
        + jnp.ones((n_discrete_states, n_discrete_states)) * off_diag[:, None]
        - jnp.diag(off_diag)
    )
    return transition_matrix / jnp.sum(transition_matrix, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Discrete-state inference utilities
# Viterbi algorithm adapted from dynamax (probml/dynamax), MIT License.
# https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/inference.py
# ---------------------------------------------------------------------------


@jax.jit
def hmm_viterbi(
    initial_probs: Array,
    transition_matrix: Array,
    log_likelihoods: Array,
) -> Array:
    """Find the most likely discrete state sequence (Viterbi algorithm).

    Uses a backward-forward decomposition that is compatible with
    ``jax.lax.scan`` and fully JIT-able.

    Parameters
    ----------
    initial_probs : Array, shape (K,)
        Prior probability of each discrete state at time 0.
    transition_matrix : Array, shape (K, K)
        Row-stochastic transition matrix where entry ``(i, j)`` is
        ``P(S_t = j | S_{t-1} = i)``.
    log_likelihoods : Array, shape (T, K)
        Per-state log observation likelihoods ``log p(y_t | S_t = k)``
        at each time step.

    Returns
    -------
    states : Array, shape (T,)
        Most likely state sequence (integer-valued).
    """
    num_timesteps, num_states = log_likelihoods.shape

    # Backward pass: accumulate best future scores and store argmax pointers
    def _backward_step(best_next_score, t):
        scores = jnp.log(transition_matrix) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    best_second_score, best_next_states = jax.lax.scan(
        _backward_step, jnp.zeros(num_states), jnp.arange(num_timesteps - 1), reverse=True
    )

    # Pick the best first state
    first_state = jnp.argmax(
        jnp.log(initial_probs) + log_likelihoods[0] + best_second_score
    )

    # Forward pass: trace through pointers
    def _forward_step(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    _, states = jax.lax.scan(_forward_step, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


# ---------------------------------------------------------------------------
# Discrete-state alignment utilities
# Adapted from dynamax (probml/dynamax), MIT License.
# https://github.com/probml/dynamax/blob/main/dynamax/utils/utils.py
# ---------------------------------------------------------------------------


def compute_state_overlap(
    z1: Array,
    z2: Array,
) -> Array:
    """Compute a matrix of state-wise overlap counts between two state sequences.

    Entry ``(i, j)`` counts the number of time steps where ``z1 == i`` and
    ``z2 == j``.

    Parameters
    ----------
    z1 : Int[Array, " num_timesteps"]
        First state sequence (integer-valued, non-negative).
    z2 : Int[Array, " num_timesteps"]
        Second state sequence (integer-valued, non-negative, same length).

    Returns
    -------
    overlap : Array, shape (K, K)
        Overlap matrix where ``K = max(z1.max(), z2.max()) + 1``.
    """
    K = jnp.maximum(z1.max(), z2.max()) + 1
    overlap = jnp.sum(
        (z1[:, None] == jnp.arange(K))[:, :, None]
        & (z2[:, None] == jnp.arange(K))[:, None, :],
        axis=0,
    )
    return overlap


def find_permutation(
    z1: Array,
    z2: Array,
) -> np.ndarray:
    """Find the permutation of labels in ``z1`` that best aligns with ``z2``.

    Uses the Hungarian algorithm on the negated overlap matrix to find the
    optimal assignment.

    Parameters
    ----------
    z1 : Int[Array, " num_timesteps"]
        First state sequence (integer-valued, non-negative).
    z2 : Int[Array, " num_timesteps"]
        Second state sequence (integer-valued, non-negative, same length).

    Returns
    -------
    permutation : np.ndarray, shape (K,)
        Permutation such that ``jnp.take(permutation, z1)`` best aligns
        with ``z2``.  ``K = max(z1.max(), z2.max()) + 1``.
    """
    overlap = compute_state_overlap(z1, z2)
    _, perm = linear_sum_assignment(-np.asarray(overlap))
    return perm
