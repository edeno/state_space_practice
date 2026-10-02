from __future__ import annotations

import functools
import logging
import operator
import warnings
from typing import cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.exceptions import StateSpaceWarning

logger = logging.getLogger(__name__)

# Type alias for numeric values (scalars, numpy arrays, JAX arrays)
Numeric = float | int | np.ndarray | jax.Array


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
    Array([[1. , 2.5],
           [2.5, 4. ]], dtype=float32)

    """
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


#: Default relative diagonal shift of :func:`psd_cholesky` / :func:`psd_solve`
#: for float64 matrices (~1e4 * f64 machine epsilon). Lower-precision dtypes
#: use their machine epsilon instead (see :func:`_default_relative_boost`).
DEFAULT_RELATIVE_BOOST = 1e-12


def _default_relative_boost(dtype) -> float:
    """Relative diagonal shift used when a caller passes ``relative_boost=None``.

    ``max(DEFAULT_RELATIVE_BOOST, eps(dtype))``: ``1e-12`` in float64, and the
    machine epsilon in float32 (``~1.2e-7``) where a ``1e-12`` relative shift
    would round away entirely.
    """
    return max(DEFAULT_RELATIVE_BOOST, float(jnp.finfo(dtype).eps))


#: Diagonal entries below this fraction of the largest one are shifted as if
#: they were this large (see :func:`_stabilizing_shift`).
_SHIFT_DIAGONAL_FLOOR_RATIO = 1e-8


def _stabilizing_shift(
    A_sym: jax.Array,
    diagonal_boost: float,
    relative_boost: float | None,
) -> jax.Array:
    """Per-diagonal-entry shift of :func:`psd_cholesky` (shape ``A.shape[:-1]``).

    ``shift_i = max(diagonal_boost, relative_boost * d_i, sqrt(tiny))`` with
    ``d_i = max(|A_ii|, 1e-8 * max_j |A_jj|)`` and ``tiny`` the smallest
    normal number of the dtype.

    The shift is proportional to each diagonal entry, i.e. ``A + rel *
    diag(A)``: Cholesky's roundoff is bounded componentwise by
    ``~n eps sqrt(A_ii A_jj)``, so this is the natural (diagonal-scaling
    invariant) stabilisation. It makes the factorization equivariant under
    any rescaling ``D A D`` of the coordinates (a state in cm next to one in
    cm/s is shifted by the same *relative* amount), and it is block-local: a
    block-diagonal matrix gets the same shift factored whole or block by
    block, so block-diagonal fast paths agree with their dense counterparts
    to roundoff. The ``1e-8 * max`` floor on ``d_i`` caps the precision of
    an exactly-degenerate coordinate (a zero diagonal entry) at
    ``~1e20 / max|diag|`` instead of ``1 / sqrt(tiny)``; ``sqrt(tiny)``
    (~1.5e-154 in float64) only keeps an all-zero matrix finite.
    """
    dtype = A_sym.dtype
    if relative_boost is None:
        relative_boost = _default_relative_boost(dtype)
    diag_abs = jnp.abs(jnp.diagonal(A_sym, axis1=-2, axis2=-1))
    max_diag = jnp.max(diag_abs, axis=-1, keepdims=True)
    scale = jnp.maximum(diag_abs, _SHIFT_DIAGONAL_FLOOR_RATIO * max_diag)
    floor = float(jnp.finfo(dtype).tiny) ** 0.5
    return jnp.maximum(
        jnp.maximum(
            jnp.asarray(diagonal_boost, dtype=dtype),
            jnp.asarray(relative_boost, dtype=dtype) * scale,
        ),
        jnp.asarray(floor, dtype=dtype),
    )


def psd_cholesky(
    A: jax.Array,
    diagonal_boost: float = 0.0,
    relative_boost: float | None = None,
) -> tuple[jax.Array, bool]:
    """Stabilized Cholesky factor of a PSD matrix.

    Symmetrizes ``A`` and adds the same diagonal shift as :func:`psd_solve`
    (entry ``i``: ``max(diagonal_boost, relative_boost * |A_ii|, sqrt(tiny))``,
    see :func:`_stabilizing_shift`) before factoring, then returns the ``(factor, lower)`` pair accepted by
    :func:`jax.scipy.linalg.cho_solve`. Sharing one factorization lets a
    caller reuse it for a linear solve, a quadratic form, and a
    log-determinant (:func:`psd_logdet`) instead of factoring the same
    matrix several times -- and guarantees all three see the *same*
    stabilized matrix (avoiding a boosted-solve / unboosted-``slogdet``
    mismatch on near-singular ``A``).

    See :func:`psd_solve` for the rationale behind the shift policy.

    Parameters
    ----------
    A : jax.Array
        Coefficient matrix or batch of matrices, expected positive
        semi-definite. Only the last two axes are treated as the matrix.
    diagonal_boost : float, optional
        Absolute floor for the stabilization shift. Default 0.0 (the shift
        is purely scale-relative).
    relative_boost : float or None, optional
        Coefficient of ``|A_ii|`` for the relative component of the shift of
        diagonal entry ``i``. Default None: ``1e-12`` for float64, machine
        epsilon for lower precision. Pass ``0.0`` to disable relative
        scaling.

    Returns
    -------
    tuple[jax.Array, bool]
        The ``(factor, lower)`` tuple from :func:`jax.scipy.linalg.cho_factor`.
        The stabilized matrix's log-determinant is
        ``2 * sum(log(abs(diag(factor))))``.
    """
    A_sym = symmetrize(A)
    shift = _stabilizing_shift(A_sym, diagonal_boost, relative_boost)
    n = A.shape[-1]
    idx = jnp.arange(n)
    A_stabilized = A_sym.at[..., idx, idx].add(shift)
    return jax.scipy.linalg.cho_factor(A_stabilized)


def psd_logdet(cho: tuple[jax.Array, bool]) -> jax.Array:
    """Log-determinant from a :func:`psd_cholesky` factor.

    ``2 * sum(log|diag(factor)|)`` for the stabilized matrix that was
    factored, so a caller reusing one factor for a solve and a
    log-determinant sees the same matrix in both.

    Parameters
    ----------
    cho : tuple[jax.Array, bool]
        The ``(factor, lower)`` tuple returned by :func:`psd_cholesky`;
        ``factor`` has shape ``(..., n, n)`` (a single matrix or a batch).

    Returns
    -------
    logdet : jax.Array, shape (...)
        Log-determinant of each stabilized matrix (a scalar array for a
        single ``(n, n)`` factor).
    """
    diag = jnp.diagonal(cho[0], axis1=-2, axis2=-1)
    return 2.0 * jnp.sum(jnp.log(jnp.abs(diag)), axis=-1)


def psd_solve(
    A: jax.Array,
    b: jax.Array,
    diagonal_boost: float = 0.0,
    relative_boost: float | None = None,
) -> jax.Array:
    """Solves a linear system Ax = b for positive semi-definite (PSD) matrices A.

    This function wraps a linear algebra solver, ensuring numerical stability
    by symmetrizing the input matrix A and adding a scaled diagonal shift
    before Cholesky. It is intended for use with PSD matrices, where
    ``assume_a="pos"`` can be safely set for performance.

    Stabilization shift
    -------------------
    Diagonal entry ``i`` of each matrix is shifted by::

        shift_i = max(diagonal_boost, relative_boost * d_i, sqrt(tiny))
        d_i     = max(|A_ii|, 1e-8 * max_j |A_jj|)

    (:func:`_stabilizing_shift`). For batched inputs the shift is computed
    independently for each matrix in the batch.

    By default the shift is **scale-relative**: ``diagonal_boost=0`` and
    ``relative_boost=1e-12`` in float64 (machine epsilon in float32, where
    ``1e-12`` would round away), i.e. ``A + 1e-12 * diag(A)``. Rescaling
    ``A -> s A`` rescales the shift by the same factor, so
    ``psd_solve(s A, b) == psd_solve(A, b) / s`` for any ``s > 0`` -- the
    solve is scale-equivariant, and so are the filters built on it (a
    covariance of scale ``1e-10`` is treated exactly like one of scale 1).
    Because the shift is proportional to each diagonal entry it is also
    equivariant under per-coordinate rescaling ``D A D`` and block-local
    (a block-diagonal matrix is shifted identically whether it is factored
    whole or block by block). ``sqrt(tiny)`` (``~1.5e-154`` in float64) is a
    last-resort floor that only keeps an all-zero matrix finite.

    An absolute floor (``diagonal_boost > 0``) is *not* scale-equivariant:
    ``diagonal_boost=1e-9`` dominates the shift for matrices below ~1e-3 in
    scale and swamps covariances below ~1e-7 entirely. Callers that want one
    anyway (e.g. to regularise a matrix that is structurally rank-deficient,
    or in fixed physical units) can still pass ``diagonal_boost``;
    ``diagonal_boost=0.0`` is the default.

    The default ``relative_boost=1e-12`` is approximately 1e4 * f64
    machine epsilon -- small enough that the shift does not perturb
    near-singular f64 matrices (whose smallest eigenvalues may be at
    1e-10 to 1e-5 of the largest) but large enough to absorb the
    ``O(n eps sqrt(A_ii A_jj))`` roundoff that makes a numerically-PSD
    matrix fail Cholesky. Pass ``relative_boost=0.0`` to disable relative
    scaling (with ``diagonal_boost=0.0`` too, only the ``sqrt(tiny)`` floor
    remains).

    Parameters
    ----------
    A : jax.Array
        The coefficient matrix or batch of coefficient matrices, expected to
        be positive semi-definite.
    b : jax.Array
        The right-hand side vector or matrix.
    diagonal_boost : float, optional
        Absolute floor for the stabilization shift. Default 0.0.
    relative_boost : float or None, optional
        Coefficient of ``|A_ii|`` used as the relative component of the
        shift of diagonal entry ``i``. Default None: ``1e-12`` (~1e4 * f64
        eps) for float64 and machine epsilon for lower-precision dtypes.
        Pass ``0.0`` to disable relative scaling entirely.

    Returns
    -------
    jax.Array
        The solution x to the linear system Ax = b.

    """
    # Factor the symmetrized, diagonally-shifted matrix once (see
    # :func:`psd_cholesky` for the shift). Solve via an explicit Cholesky
    # factor + cho_solve rather than jax.scipy.linalg.solve(assume_a="pos").
    # cho_solve deprecates a batch of 1D right-hand sides passed as b.ndim > 1
    # (ambiguous against a single 2D RHS), so give each system an explicit
    # trailing solve axis -- exactly the b[..., None] ... .squeeze(-1) pattern
    # the deprecation recommends -- and drop it again. The non-batched vector
    # path is numerically unchanged, and 2D matrix right-hand sides (all
    # internal callers) are untouched.
    cho = psd_cholesky(A, diagonal_boost=diagonal_boost, relative_boost=relative_boost)
    b = jnp.asarray(b)
    rhs_is_vector = b.ndim == A.ndim - 1
    b_mat = b[..., None] if rhs_is_vector else b
    x = jax.scipy.linalg.cho_solve(cho, b_mat)
    return x[..., 0] if rhs_is_vector else x


def clip_eigenvalues(
    mat: jax.Array,
    min_eigenvalue: float | None = None,
    max_eigenvalue: float | None = None,
) -> jax.Array:
    """Clip the eigenvalues of a symmetric matrix into ``[min, max]``.

    Symmetrizes ``mat``, eigendecomposes it, clips the spectrum (each bound is
    skipped when ``None``) and reconstructs the matrix. Used by the EM
    trust-region updates to keep process / initial covariances inside a
    configured eigenvalue range.

    Parameters
    ----------
    mat : jax.Array, shape (n, n)
        Matrix to clip; only its symmetric part is used.
    min_eigenvalue, max_eigenvalue : float or None
        Lower / upper bound on the eigenvalues; ``None`` leaves that side open.

    Returns
    -------
    clipped : jax.Array, shape (n, n)
        Symmetric matrix with the same eigenvectors as ``symmetrize(mat)`` and
        its eigenvalues clipped into ``[min_eigenvalue, max_eigenvalue]``.
    """
    eigvals, eigvecs = jnp.linalg.eigh(symmetrize(mat))
    if min_eigenvalue is not None:
        eigvals = jnp.maximum(eigvals, min_eigenvalue)
    if max_eigenvalue is not None:
        eigvals = jnp.minimum(eigvals, max_eigenvalue)
    return cast(jax.Array, eigvecs @ jnp.diag(eigvals) @ eigvecs.T)


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


#: Default relative eigenvalue floor for M-step covariance projections: an
#: eigenvalue below ``PSD_RELATIVE_FLOOR * max|eigenvalue|`` is raised to it.
PSD_RELATIVE_FLOOR = 1e-8
#: Absolute floor under the relative one; only binds for an (all-)zero matrix.
PSD_ABSOLUTE_FLOOR = 1e-30


def relative_psd_floor(
    eigenvalues: Array,
    relative_floor: float = PSD_RELATIVE_FLOOR,
    absolute_floor: float = PSD_ABSOLUTE_FLOOR,
) -> Array:
    """Scale-relative lower bound for the eigenvalues of a covariance.

    Returns ``max(absolute_floor, rel * max|eigenvalues|)`` with
    ``rel = max(relative_floor, 10 * eps)`` for the dtype, so the floor
    scales with the matrix: a covariance in volts (entries ~1e-10) is not
    clamped to the same absolute value as one in centimetres. The ``10 *
    eps`` term keeps the floor above the roundoff of an eigen-reconstruction
    in low precision.

    Parameters
    ----------
    eigenvalues : Array, shape (..., n)
        Eigenvalues (or the diagonal of a diagonal covariance).
    relative_floor : float, default=PSD_RELATIVE_FLOOR
        Floor as a fraction of the largest eigenvalue magnitude.
    absolute_floor : float, default=PSD_ABSOLUTE_FLOOR
        Lower bound on the floor itself (binds only for a ~zero matrix).

    Returns
    -------
    floor : Array, shape (...)
        The eigenvalue floor for each matrix.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    dtype = (
        eigenvalues.dtype
        if jnp.issubdtype(eigenvalues.dtype, jnp.floating)
        else jnp.result_type(float)
    )
    rel = max(float(relative_floor), 10.0 * float(jnp.finfo(dtype).eps))
    scale = jnp.max(jnp.abs(eigenvalues), axis=-1)
    return jnp.maximum(jnp.asarray(absolute_floor, dtype=dtype), rel * scale)


def _log_floored_eigenvalues(
    n_floored: Array,
    floor: Array,
    min_eigenvalue: Array | None = None,
    max_abs_eigenvalue: Array | None = None,
    *,
    name: str,
) -> None:
    """Host-side logger for :func:`project_psd_relative` (via debug callback)."""
    n = int(np.sum(np.asarray(n_floored)))
    if n == 0:
        return
    floor_value = float(np.min(np.asarray(floor)))
    if min_eigenvalue is not None and max_abs_eigenvalue is not None:
        min_eig = np.asarray(min_eigenvalue)
        max_abs = np.asarray(max_abs_eigenvalue)
        tolerance = np.sqrt(np.finfo(min_eig.dtype).eps) * max_abs
        if bool(np.any(min_eig < -tolerance)):
            logger.warning(
                "%s: the estimate is indefinite (minimum eigenvalue %.3g "
                "relative to a largest |eigenvalue| of %.3g, on the "
                "correlation scale); raised %d eigenvalue(s) to the PSD floor "
                "%.3g. A materially negative eigenvalue indicates inconsistent "
                "sufficient statistics (e.g. E-step moments that do not come "
                "from one posterior), not a degenerate latent dimension.",
                name,
                float(np.min(min_eig)),
                float(np.max(max_abs)),
                n,
                floor_value,
            )
            return
    logger.warning(
        "%s: raised %d eigenvalue(s) to the scale-relative PSD floor "
        "%.3g. The estimate is (numerically) rank deficient; check for "
        "a degenerate latent dimension or an over-parameterised model.",
        name,
        n,
        floor_value,
    )


def warn_if_floored(
    n_floored: Array,
    floor: Array,
    name: str,
    min_eigenvalue: Array | None = None,
    max_abs_eigenvalue: Array | None = None,
) -> None:
    """Log (host side, jit-safe) that ``n_floored`` eigenvalues were floored.

    Emits a ``logger.warning`` from :mod:`state_space_practice.utils` when
    ``n_floored > 0``. Works eagerly and inside ``jax.jit`` / ``lax.scan``
    (through :func:`jax.debug.callback`), so jitted M-steps can report a
    projection without a host sync of their own.

    A round-off-level negative (or zero) eigenvalue is reported as numerical
    rank deficiency. When the eigenvalue range is supplied and the minimum
    eigenvalue is materially negative (below ``-sqrt(eps) *
    max_abs_eigenvalue``), the matrix is reported as indefinite instead, which
    points at inconsistent sufficient statistics rather than a degenerate
    dimension.

    Parameters
    ----------
    n_floored : Array
        Number of eigenvalues (or variances) that were raised to the floor.
    floor : Array
        The floor that was applied.
    name : str
        Name of the quantity, used in the log message.
    min_eigenvalue, max_abs_eigenvalue : Array or None, optional
        Minimum eigenvalue and largest eigenvalue magnitude of the matrix
        before flooring (on the scale the floor was applied on).
    """
    # None is an empty pytree, so it reaches the host logger unchanged.
    jax.debug.callback(
        functools.partial(_log_floored_eigenvalues, name=name),
        n_floored,
        floor,
        min_eigenvalue,
        max_abs_eigenvalue,
    )


def clip_eigenvalues_relative(
    cov: Array,
    relative_floor: float = PSD_RELATIVE_FLOOR,
    absolute_floor: float = PSD_ABSOLUTE_FLOOR,
) -> tuple[Array, Array]:
    """Floor a covariance on the correlation scale and count what was raised.

    The silent core of :func:`project_psd_relative`, for callers that report
    the count themselves (e.g. accumulated over a ``lax.scan``). See
    :func:`project_psd_relative` for the projection.

    Parameters
    ----------
    cov : Array, shape (n, n)
        Covariance-like matrix; only its symmetric part is used.
    relative_floor, absolute_floor : float
        See :func:`relative_psd_floor`; applied to the eigenvalues of the
        correlation-scaled matrix.

    Returns
    -------
    projected : Array, shape (n, n)
        ``symmetrize(cov)`` exactly when no eigenvalue is floored, otherwise
        the floored matrix (positive definite).
    n_floored : Array
        Number of correlation-scale eigenvalues that were raised to the floor
        (int scalar).
    """
    projected, n_floored, _, _ = _clip_eigenvalues_relative(
        cov, relative_floor, absolute_floor
    )
    return projected, n_floored


def _correlation_scale(cov: Array) -> Array:
    """Per-coordinate scale ``sqrt(d_i)`` used to form ``D^{-1/2} C D^{-1/2}``.

    A nonpositive diagonal entry (a degenerate or indefinite coordinate) has
    no scale of its own and uses the largest diagonal entry instead; an
    all-nonpositive diagonal uses 1 (the absolute scale).

    Parameters
    ----------
    cov : Array, shape (..., n, n)
        Symmetric matrix.

    Returns
    -------
    scale : Array, shape (..., n)
        Positive scales.
    """
    diag = jnp.diagonal(cov, axis1=-2, axis2=-1)
    max_diag = jnp.max(diag, axis=-1, keepdims=True)
    fallback = jnp.where(max_diag > 0.0, max_diag, jnp.ones_like(max_diag))
    return jnp.sqrt(jnp.where(diag > 0.0, diag, fallback))


def _clip_eigenvalues_relative(
    cov: Array, relative_floor: float, absolute_floor: float
) -> tuple[Array, Array, Array, Array]:
    """:func:`clip_eigenvalues_relative` that also returns the floor used and
    the eigenvalues before flooring (both on the correlation scale)."""
    cov = symmetrize(jnp.asarray(cov))
    scale = _correlation_scale(cov)
    outer_scale = scale[..., :, None] * scale[..., None, :]
    eigvals, eigvecs = jnp.linalg.eigh(cov / outer_scale)
    floor = relative_psd_floor(eigvals, relative_floor, absolute_floor)
    floored = eigvals < floor
    raised = jnp.where(floored, floor, eigvals)
    projected = symmetrize(((eigvecs * raised[None, :]) @ eigvecs.T) * outer_scale)
    n_floored = jnp.sum(floored, dtype=jnp.int32)
    return jnp.where(n_floored > 0, projected, cov), n_floored, floor, eigvals


def project_psd_relative(
    cov: Array,
    relative_floor: float = PSD_RELATIVE_FLOOR,
    absolute_floor: float = PSD_ABSOLUTE_FLOOR,
    name: str = "covariance",
    warn: bool = True,
) -> Array:
    """Symmetrize a covariance and floor its eigenvalues on the correlation scale.

    The scale-aware covariance projection of the EM M-steps. With ``D`` the
    positive diagonal of ``C = symmetrize(cov)`` (see
    :func:`_correlation_scale` for nonpositive entries), the eigenvalues of
    the correlation-scaled matrix ``D^{-1/2} C D^{-1/2}`` below
    :func:`relative_psd_floor` of it are raised to it and the result is
    scaled back. The projection is therefore equivariant under any diagonal
    change of units, ``floor(S C S) = S floor(C) S``: a small variance next
    to a large one is floored relative to its own scale, not to the largest
    eigenvalue. A covariance with no correlation-scale eigenvalue below the
    floor is returned exactly (``symmetrize(cov)``).

    Parameters
    ----------
    cov : Array, shape (n, n)
        Covariance estimate; only its symmetric part is used.
    relative_floor, absolute_floor : float
        See :func:`relative_psd_floor`; applied to the correlation-scaled
        eigenvalues.
    name : str, default="covariance"
        Name used in the log message.
    warn : bool, default=True
        If True, log a warning (see :func:`warn_if_floored`) whenever the
        projection changes an eigenvalue.

    Returns
    -------
    projected : Array, shape (n, n)
        Symmetric positive-definite matrix.
    """
    projected, n_floored, floor, eigvals = _clip_eigenvalues_relative(
        cov, relative_floor, absolute_floor
    )
    if warn:
        warn_if_floored(
            n_floored,
            floor,
            name,
            min_eigenvalue=jnp.min(eigvals),
            max_abs_eigenvalue=jnp.max(jnp.abs(eigvals)),
        )
    return cast(Array, projected)


def floor_variances_relative(
    variances: Array,
    relative_floor: float = PSD_RELATIVE_FLOOR,
    absolute_floor: float = PSD_ABSOLUTE_FLOOR,
    name: str = "variances",
    warn: bool = True,
) -> Array:
    """Diagonal-covariance counterpart of :func:`project_psd_relative`.

    Parameters
    ----------
    variances : Array, shape (n,)
        Diagonal of a diagonal covariance.
    relative_floor, absolute_floor : float
        See :func:`relative_psd_floor`.
    name : str, default="variances"
        Name used in the log message.
    warn : bool, default=True
        If True, log a warning when any variance is raised to the floor.

    Returns
    -------
    floored : Array, shape (n,)
        ``maximum(variances, floor)``.
    """
    variances = jnp.asarray(variances)
    floor = relative_psd_floor(variances, relative_floor, absolute_floor)
    floored = variances < floor
    if warn:
        warn_if_floored(jnp.sum(floored, dtype=jnp.int32), floor, name)
    return jnp.where(floored, floor, variances)


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


def spectral_radius(matrix: ArrayLike) -> float:
    """Largest eigenvalue magnitude of a (possibly non-symmetric) square matrix.

    Computed on host with NumPy: ``jnp.linalg.eigvals`` (the general,
    non-symmetric eigendecomposition) has no GPU/TPU lowering, so calling it on
    an accelerator backend raises or forces an implicit host round-trip. Callers
    use this for eager, post-optimization stability checks -- not inside a JIT
    trace -- so a host computation is both safe and portable across backends.
    For symmetric matrices prefer ``jnp.linalg.eigvalsh``, which is accelerated.
    Inside a traced / differentiated computation use
    :func:`differentiable_spectral_radius` instead.

    Parameters
    ----------
    matrix : ArrayLike, shape (n, n)
        Square matrix (need not be symmetric).

    Returns
    -------
    float
        ``max_i |lambda_i(matrix)|``, the spectral radius.
    """
    eigenvalues = np.linalg.eigvals(np.asarray(matrix))
    return float(np.max(np.abs(eigenvalues)))


def _host_spectral_radius_and_gradient(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral radius and its gradient for a batch of real square matrices.

    For the dominant eigenvalue ``lambda = V[:, i]``-eigenpair of ``A = V D
    V^{-1}``, first-order perturbation gives ``d lambda = (V^{-1} dA V)_{ii}``
    and ``d|lambda| = Re(conj(lambda) d lambda) / |lambda|``, so
    ``d|lambda| / dA_{ab} = Re(conj(lambda) / |lambda| * Vinv[i, a] * V[b, i])``.
    The gradient is zeroed where it is undefined (``lambda == 0``) or not finite
    (defective / numerically non-diagonalizable ``A``).
    """
    matrices = np.asarray(matrices)
    batch_shape = matrices.shape[:-2]
    n = matrices.shape[-1]
    flat = matrices.reshape((-1, n, n))
    radii = np.zeros(len(flat), dtype=matrices.dtype)
    grads = np.zeros_like(flat)
    for k, A in enumerate(flat):
        if not np.all(np.isfinite(A)):
            radii[k] = np.nan
            continue
        eigenvalues, V = np.linalg.eig(A)
        i = int(np.argmax(np.abs(eigenvalues)))
        lam = eigenvalues[i]
        radii[k] = np.abs(lam)
        if radii[k] == 0.0:
            continue
        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            continue
        grad = np.real((np.conj(lam) / np.abs(lam)) * np.outer(V_inv[i, :], V[:, i]))
        if np.all(np.isfinite(grad)):
            grads[k] = grad
    return radii.reshape(batch_shape), grads.reshape(matrices.shape)


def _spectral_radius_callback(matrices: jax.Array) -> tuple[jax.Array, jax.Array]:
    dtype = matrices.dtype
    result_shape = (
        jax.ShapeDtypeStruct(matrices.shape[:-2], dtype),
        jax.ShapeDtypeStruct(matrices.shape, dtype),
    )
    return cast(
        tuple[jax.Array, jax.Array],
        jax.pure_callback(
            _host_spectral_radius_and_gradient,
            result_shape,
            matrices,
            vmap_method="sequential",
        ),
    )


@jax.custom_jvp
def differentiable_spectral_radius(matrices: ArrayLike) -> Array:
    """Exact, differentiable spectral radius usable inside JIT / ``grad``.

    The eigen-decomposition runs on host through ``jax.pure_callback`` (the
    general ``eigvals`` has no portable accelerator lowering), and a custom JVP
    supplies the first-order perturbation derivative of the dominant eigenvalue
    magnitude. Unlike a norm-based upper bound, the value is the actual
    ``max |lambda|``, so a clamp built on it engages only when the dynamics
    really exceed the bound.

    The derivative is exact wherever the dominant eigenvalue (or conjugate
    pair) is simple; at a crossing of two dominant moduli it is one
    subgradient, and it is zero for a defective matrix.

    Parameters
    ----------
    matrices : ArrayLike, shape (..., n, n)
        Real square matrix or batch of matrices.

    Returns
    -------
    Array, shape (...)
        Spectral radius of each matrix.
    """
    radii, _ = _spectral_radius_callback(jnp.asarray(matrices))
    return radii


@differentiable_spectral_radius.defjvp
def _differentiable_spectral_radius_jvp(primals, tangents):
    (matrices,) = primals
    (d_matrices,) = tangents
    matrices = jnp.asarray(matrices)
    radii, grads = _spectral_radius_callback(matrices)
    return radii, jnp.sum(grads * d_matrices, axis=(-2, -1))


def _strongly_connected_blocks(matrix: np.ndarray, block_size: int) -> list:
    """Group ``block_size`` diagonal blocks into strongly connected components.

    Block ``(i, j)`` is an edge when it has any nonzero entry. The spectrum of
    ``matrix`` is the union of the spectra of the principal submatrices of its
    strongly connected components (a permutation brings it to block-triangular
    form), so each component can be stabilized on its own.
    """
    from scipy.sparse.csgraph import connected_components

    n_blocks = matrix.shape[0] // block_size
    blocks = matrix.reshape(n_blocks, block_size, n_blocks, block_size)
    adjacency = np.any(blocks != 0.0, axis=(1, 3))
    n_components, labels = connected_components(
        adjacency, directed=True, connection="strong"
    )
    return [np.flatnonzero(labels == c) for c in range(n_components)]


def stabilize_transition_matrix(
    matrix: ArrayLike,
    max_spectral_radius: float = 0.99,
    block_size: int | None = None,
    warn: bool = True,
) -> Array:
    """Scale a transition matrix so its spectral radius is <= the bound.

    A linear transition ``x_t = A @ x_{t-1} + ...`` is stable only when every
    eigenvalue of ``A`` lies inside the unit circle. When the spectral radius
    exceeds ``max_spectral_radius`` the matrix is scaled by
    ``max_spectral_radius / spectral_radius``; otherwise it is returned
    unchanged.

    With ``block_size`` (e.g. ``2`` for oscillator blocks) the clamp is local:
    the ``block_size x block_size`` blocks are grouped into strongly connected
    components of the block coupling graph, whose principal submatrices carry
    disjoint parts of the spectrum, and only the components whose own spectral
    radius exceeds the bound are scaled. For uncoupled oscillators this rescales
    only the offending oscillator instead of damping every rhythm; for a fully
    coupled matrix it reduces to the uniform scale.

    Choosing ``max_spectral_radius``: an oscillator block with radius ``r`` at
    sampling rate ``fs`` has a spectral peak of half-power bandwidth
    ``Delta f ~= (1 - r) * fs / pi``. To keep rhythms as narrow as
    ``Delta f_min`` representable use ``max_spectral_radius >= 1 - pi *
    Delta f_min / fs`` (e.g. ``fs = 100`` Hz and ``Delta f_min = 0.3`` Hz gives
    ``0.99``; ``fs = 1000`` Hz needs ``0.999``).

    The spectral radius is computed on host (see :func:`spectral_radius`), so
    this is portable across accelerator backends but must be called eagerly,
    not inside a JIT trace.

    Parameters
    ----------
    matrix : ArrayLike, shape (n, n)
        Transition matrix (need not be symmetric). Integer and boolean inputs
        are promoted to the default floating dtype before scaling.
    max_spectral_radius : float, default=0.99
        Upper bound on the spectral radius.
    block_size : int or None, default=None
        Size of the structural diagonal blocks. ``None`` applies one uniform
        scale to the whole matrix. A positive ``block_size`` that does not
        divide ``n`` also falls back to the uniform scale (logged at WARNING
        when ``warn=True``).
    warn : bool, default=True
        Log a warning (``logging``, host-side) reporting the radius and applied
        scale whenever the clamp engages. Logging rather than
        ``warnings.warn`` because EM calls this every iteration.

    Returns
    -------
    Array, shape (n, n)
        The stabilized matrix, or the (dtype-promoted) input unchanged if
        already within the bound.
    """
    A = jnp.asarray(matrix)
    if not jnp.issubdtype(A.dtype, jnp.inexact):
        A = A.astype(jnp.result_type(float))
    A_host = np.asarray(A)
    n = A_host.shape[0]
    if block_size is None or block_size <= 0 or n % block_size != 0:
        if warn and block_size is not None and block_size > 0:
            logger.warning(
                "stabilize_transition_matrix: block_size=%d does not divide "
                "the matrix size %d; clamping with one uniform scale over the "
                "whole matrix instead of per block.",
                block_size,
                n,
            )
        components = [np.arange(n)]
    else:
        components = [
            (block_size * comp[:, None] + np.arange(block_size)[None, :]).ravel()
            for comp in _strongly_connected_blocks(A_host, block_size)
        ]

    stabilized = A_host.copy()
    applied = []
    for idx in components:
        sub = A_host[np.ix_(idx, idx)]
        radius = spectral_radius(sub)
        if radius > max_spectral_radius:
            scale = max_spectral_radius / radius
            stabilized[np.ix_(idx, idx)] = sub * scale
            applied.append((idx, radius, scale))
    if not applied:
        return A
    if warn:
        details = "; ".join(
            f"rows {idx.tolist()}: radius={radius:.6g}, scale={scale:.6g}"
            for idx, radius, scale in applied
        )
        logger.warning(
            "Transition matrix spectral radius exceeded max_spectral_radius="
            "%g; clamped (%s). If a narrow-band rhythm is expected, raise "
            "max_spectral_radius toward 1 - pi * bandwidth / sampling_freq.",
            max_spectral_radius,
            details,
        )
    return jnp.asarray(stabilized, dtype=A.dtype)


def contains_tracer(*values: object) -> bool:
    """Return True if any pytree leaf is being traced by JAX."""
    return any(
        isinstance(leaf, jax.core.Tracer)
        for value in values
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _warn_not_positive_definite_host(
    min_eigenvalue: np.ndarray, *, name: str, filter_name: str
) -> None:
    """Host side of :func:`warn_if_not_positive_definite_in_graph`."""
    min_eig = float(np.min(np.asarray(min_eigenvalue)))
    if not min_eig > 0.0:
        warnings.warn(
            f"{filter_name}: {name} is not positive definite (minimum "
            f"eigenvalue {min_eig:g}). The input was traced (jax.jit / "
            f"jax.grad / jax.vmap), so this could not be raised as an error; "
            f"the returned estimates and log-likelihood are invalid.",
            StateSpaceWarning,
            stacklevel=2,
        )


def warn_if_not_positive_definite_in_graph(
    cov: ArrayLike, *, name: str, filter_name: str
) -> None:
    """Report a non-positive-definite (possibly traced) covariance at run time.

    The in-graph counterpart of the host-side positive-definiteness check for
    inputs that are JAX tracers: the minimum eigenvalue of ``symmetrize(cov)``
    is computed inside the traced computation and handed to
    :func:`jax.debug.callback`, which emits a
    :class:`~state_space_practice.exceptions.StateSpaceWarning` naming it when
    it is not strictly positive (or not finite). Works under ``jax.jit``,
    ``jax.grad`` and ``jax.vmap``; the callback is asynchronous, so the warning
    is emitted when the computation runs (``jax.effects_barrier()`` waits for
    it).

    Parameters
    ----------
    cov : ArrayLike, shape (..., n, n)
        Covariance (or batch of covariances) to check.
    name : str
        Name of the covariance, used in the warning.
    filter_name : str
        Name of the public entry point, used in the warning.
    """
    cov = jnp.asarray(cov)
    min_eig = jnp.min(jnp.linalg.eigvalsh(symmetrize(cov)))
    jax.debug.callback(
        functools.partial(
            _warn_not_positive_definite_host, name=name, filter_name=filter_name
        ),
        min_eig,
    )


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


def validate_finite_array(name: str, value: ArrayLike) -> None:
    """Validate finite model parameters at public boundaries.

    Host-side check (it forces a device-to-host sync), so call it on concrete
    values before JIT dispatch, not inside traced code.

    Parameters
    ----------
    name : str
        Parameter name used in the error message.
    value : ArrayLike, any shape
        Values to check.

    Raises
    ------
    ValueError
        If any entry is NaN or infinite.
    """
    arr = jnp.asarray(value)
    if bool(jnp.any(~jnp.isfinite(arr))):
        raise ValueError(f"{name} must contain only finite values.")


def validate_nonnegative_array(name: str, value: ArrayLike) -> None:
    """Validate finite, non-negative model parameters at public boundaries.

    Host-side check; see :func:`validate_finite_array`.

    Parameters
    ----------
    name : str
        Parameter name used in the error message.
    value : ArrayLike, any shape
        Values to check.

    Raises
    ------
    ValueError
        If any entry is non-finite or negative.
    """
    arr = jnp.asarray(value)
    validate_finite_array(name, arr)
    if bool(jnp.any(arr < 0)):
        raise ValueError(f"{name} must be non-negative.")


def validate_unit_interval_array(name: str, value: ArrayLike) -> None:
    """Validate finite parameters constrained to the closed unit interval.

    Host-side check; see :func:`validate_finite_array`.

    Parameters
    ----------
    name : str
        Parameter name used in the error message.
    value : ArrayLike, any shape
        Values to check.

    Raises
    ------
    ValueError
        If any entry is non-finite or outside ``[0, 1]``.
    """
    arr = jnp.asarray(value)
    validate_finite_array(name, arr)
    if bool(jnp.any((arr < 0) | (arr > 1))):
        raise ValueError(f"{name} entries must lie in [0, 1].")


def validate_int(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    """Validate an integer configuration value at a public boundary.

    Accepts exactly what ``operator.index`` accepts: Python and NumPy
    integers, 0-d integer NumPy / JAX arrays, and ``bool`` (``True`` -> 1,
    since ``bool`` subclasses ``int``). Floats (even integer-valued ones) and
    arrays with ``ndim >= 1`` are rejected.

    Parameters
    ----------
    value : object
        Value to validate.
    name : str
        Parameter name used in the error message.
    positive : bool, default=False
        Require ``value > 0``.
    nonnegative : bool, default=False
        Require ``value >= 0``.

    Returns
    -------
    value_int : int
        ``value`` as a plain Python ``int``.

    Raises
    ------
    ValueError
        If ``value`` is not an integer (as above) or is outside the requested
        range.
    """
    kind = "positive" if positive else "non-negative" if nonnegative else "an"
    article = "a " if kind != "an" else ""
    message = f"{name} must be {article}{kind} integer."
    try:
        value_int = operator.index(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(message) from exc
    if (positive and value_int <= 0) or (nonnegative and value_int < 0):
        raise ValueError(message)
    return value_int


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
    measurement_cov: Array | None = None,
    process_cov: Array | None = None,
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
    re-validation. The ``eigvalsh → float(...)`` conversion used here needs
    concrete values. Concrete arrays are checked even inside an active trace
    (e.g. constants closed over by a jitted function): the checks run under
    :func:`jax.ensure_compile_time_eval`. The whole check is skipped when any
    covariance is a JAX tracer (a traced argument of ``jax.jit`` /
    ``jax.grad`` / ``jax.vmap``), since a traced value cannot be inspected
    host-side; the public filters then report a non-positive-definite
    ``init_cov`` at run time through
    :func:`warn_if_not_positive_definite_in_graph` instead.
    """
    if contains_tracer(init_covariance, measurement_cov, process_cov):
        return
    # Closed-over constants of a jitted caller are concrete, but any jnp op
    # on them inside the trace is staged; evaluate the checks eagerly so
    # they stay concrete.
    with jax.ensure_compile_time_eval():
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
            # factor, f32 machine epsilon ~1.2e-7. Under a random-walk
            # accumulation model over n_time bins, total roundoff ~
            # sqrt(n_time * n_state) * eps * max_eig.
            f32_eps = 1.2e-7
            worst_roundoff = float((n_time * n_state) ** 0.5) * f32_eps * max_eig
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
                    StateSpaceWarning,
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
        slices: list[tuple[int | None, Array]] = [(None, arr)]
    elif arr.ndim == 3:
        slices = [(k, arr[..., k]) for k in range(arr.shape[-1])]
    else:
        raise ValueError(
            f"{name} must be a 2D matrix or a 3D per-state stack "
            f"(d, d, n_states), got shape {arr.shape}"
        )

    # An empty input (0x0 matrix, or a stack with zero states / zero-dim
    # slices) would make every per-slice check below vacuously pass -- the
    # 3D case builds an empty `slices` list so the loop never runs, and the
    # 2D 0x0 case satisfies the symmetry/eigenvalue checks trivially. A
    # covariance with no entries is never a valid input; reject it loudly.
    if arr.size == 0:
        raise ValueError(
            f"{name} is empty (shape {arr.shape}); a covariance must have at "
            f"least one 1x1 slice."
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
        if not bool(jnp.allclose(mat, mat.T, rtol=symmetry_rtol, atol=symmetry_atol)):
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
        raise ValueError(f"{name} must be a square 2D matrix, got shape {arr.shape}")
    # A 0x0 matrix satisfies the square check and makes the non-negativity and
    # row-sum reductions vacuously pass (empty `any` is False, empty `allclose`
    # is True). A transition matrix with no states is never valid.
    if arr.shape[0] == 0:
        raise ValueError(
            f"{name} is empty (shape {arr.shape}); a transition matrix must "
            f"have at least one state."
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
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D probability vector, got shape {arr.shape}."
        )
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
_LOG_FLOOR_VALUE = float(np.log(_LOG_PROB_FLOOR))
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
    safe_denominator = jnp.where(denominator == 0.0, 1.0, denominator)
    return jnp.where(denominator == 0.0, 0.0, numerator / safe_denominator)


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
        log(x) where x > _LOG_PROB_FLOOR, otherwise _LOG_FLOOR_VALUE. A NaN
        argument is propagated as NaN (not floored): a NaN here signals genuine
        upstream divergence, and masking it to a finite value would defeat
        non-finite-based EM rollback. Finite non-positive inputs are treated as
        underflow and floored as before.
    """
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x, 1.0))
    floor = jnp.asarray(_LOG_PROB_FLOOR, dtype=x.dtype)
    floor_value = jnp.asarray(_LOG_FLOOR_VALUE, dtype=x.dtype)
    safe_x = jnp.where(x > floor, x, floor)
    result = jnp.where(x > floor, jnp.log(safe_x), floor_value)
    return jnp.where(jnp.isnan(x), jnp.nan, result)


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
    distribution. Non-finite entries are also sanitized before flooring so one
    bad scan step does not poison every later discrete posterior. These paths
    are intentional for numerical robustness, but they are usually a sign of
    upstream numerical trouble. A ``jax.debug.print`` fires on the fallback
    path so callers see it in filter/smoother logs without changing the
    function's return contract.
    """
    probabilities = jnp.asarray(probabilities)
    probabilities = probabilities.astype(jnp.result_type(probabilities, 1.0))
    floor = jnp.asarray(_DISCRETE_PROB_STABILITY_FLOOR, dtype=probabilities.dtype)
    has_nonfinite = jnp.any(~jnp.isfinite(probabilities))
    has_posinf = jnp.any(jnp.isposinf(probabilities))
    positive_infinity_mass = jnp.where(
        jnp.isposinf(probabilities),
        jnp.ones_like(probabilities),
        jnp.zeros_like(probabilities),
    )
    finite_probabilities = jnp.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    cleaned = jnp.where(has_posinf, positive_infinity_mass, finite_probabilities)
    all_nonpositive = jnp.all(cleaned <= 0)
    # A single negative-but-finite entry (a genuine upstream sign error) is
    # otherwise silently floored while a positive sibling survives, producing a
    # valid-looking distribution -- surface it too, not just the all-nonpositive
    # case.
    has_negative = jnp.any(cleaned < 0.0)
    debug_print_if(
        has_nonfinite | all_nonpositive | has_negative,
        "utils.stabilize_probability_vector: input was non-finite, "
        "all-nonpositive, or contained a negative entry (max={m}); "
        "sanitizing and flooring the discrete-state posterior. Check upstream "
        "likelihood scaling if this repeats.",
        m=jnp.max(probabilities),
    )
    stabilized = jnp.maximum(cleaned, floor)
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
    log_likelihood = jnp.asarray(log_likelihood)
    log_likelihood = log_likelihood.astype(jnp.result_type(log_likelihood, 1.0))
    has_nan = jnp.any(jnp.isnan(log_likelihood))
    has_posinf = jnp.any(jnp.isposinf(log_likelihood))

    finite_log_likelihood = jnp.where(
        jnp.isfinite(log_likelihood), log_likelihood, -jnp.inf
    )
    finite_ll_max = finite_log_likelihood.max()
    ll_max = jnp.where(jnp.isfinite(finite_ll_max), finite_ll_max, 0.0)
    scaled = jnp.exp(finite_log_likelihood - ll_max)

    posinf_scaled = jnp.asarray(log_likelihood == jnp.inf, dtype=scaled.dtype)
    scaled = jnp.where(has_posinf, posinf_scaled, scaled)
    ll_max = jnp.where(has_posinf, jnp.inf, ll_max)

    scaled = jnp.where(has_nan, jnp.full_like(scaled, jnp.nan), scaled)
    ll_max = jnp.where(has_nan, jnp.nan, ll_max)
    return scaled, ll_max


def check_converged(
    log_likelihood: Numeric,
    previous_log_likelihood: Numeric,
    tolerance: float = 1e-4,
    absolute_tolerance: float | None = None,
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
    absolute_tolerance : float, optional
        Absolute change threshold used when the likelihood scale is near zero.
        Defaults to ``tolerance``.

    Returns
    -------
    is_converged : bool
        True if the relative change < tolerance.
    is_increasing : bool
        True if the relative decrease does not exceed tolerance. Note: when the
        absolute change is below ``absolute_tolerance`` (the near-zero-scale
        regime) this returns ``True`` even for a genuine decrease within that
        tolerance -- intentional, to suppress spurious rollbacks on sub-tolerance
        EM fluctuations from the Laplace/GPB approximation.

    """
    # Handle infinite values (e.g., first iteration when previous is -inf)
    if not np.isfinite(previous_log_likelihood) or not np.isfinite(log_likelihood):
        # Can't be converged if either value is infinite
        # is_increasing is True if current is finite or greater
        is_increasing = log_likelihood >= previous_log_likelihood
        return False, bool(is_increasing)

    delta_log_likelihood = np.abs(log_likelihood - previous_log_likelihood)
    if absolute_tolerance is None:
        absolute_tolerance = tolerance
    if delta_log_likelihood < absolute_tolerance:
        return True, True

    eps = np.finfo(float).eps
    avg_log_likelihood = (
        np.abs(log_likelihood) + np.abs(previous_log_likelihood) + eps
    ) / 2

    relative_change = (log_likelihood - previous_log_likelihood) / avg_log_likelihood
    is_increasing = relative_change >= -tolerance
    is_converged = (delta_log_likelihood / avg_log_likelihood) < tolerance

    return bool(is_converged), bool(is_increasing)


def make_discrete_transition_matrix(diag: Array, n_discrete_states: int) -> Array:
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
    diag = jnp.asarray(diag)
    if n_discrete_states < 1:
        raise ValueError(
            f"n_discrete_states must be at least 1, got {n_discrete_states}."
        )
    if diag.shape != (n_discrete_states,):
        raise ValueError(
            f"diag must have shape ({n_discrete_states},), got {diag.shape}."
        )
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
    log_initial_probs = zero_preserving_log(initial_probs)
    log_transition_matrix = zero_preserving_log(transition_matrix)

    # Backward pass: accumulate best future scores and store argmax pointers
    def _backward_step(best_next_score, t):
        scores = log_transition_matrix + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    best_second_score, best_next_states = jax.lax.scan(
        _backward_step,
        jnp.zeros(num_states),
        jnp.arange(num_timesteps - 1),
        reverse=True,
    )

    # Pick the best first state
    first_state = jnp.argmax(log_initial_probs + log_likelihoods[0] + best_second_score)

    # Forward pass: trace through pointers
    def _forward_step(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    _, states = jax.lax.scan(_forward_step, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


def zero_preserving_log(probabilities: Array) -> Array:
    """Elementwise log of probabilities that keeps exact zeros at ``-inf``.

    There is no floor: an exact zero maps to ``-inf`` (an impossible state or
    transition stays impossible), and every positive value maps to its true
    log (e.g. ``1e-300 -> -690.8``; note XLA may flush subnormals to zero).
    Malformed inputs are not masked: NaN and negative entries give NaN, so
    they propagate (fail loud) instead of being mistaken for a structural zero.

    Parameters
    ----------
    probabilities : Array, shape (...)
        Probabilities (any shape).

    Returns
    -------
    log_probabilities : Array, shape (...)
        ``log(probabilities)`` with ``0 -> -inf`` and ``NaN``/negative -> NaN.
    """
    safe_probabilities = jnp.where(probabilities == 0, 1.0, probabilities)
    return jnp.where(probabilities == 0, -jnp.inf, jnp.log(safe_probabilities))


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

    Notes
    -----
    ``K`` is data-dependent, so this host-side helper is not compatible with
    ``jax.jit``.
    """
    z1 = jnp.asarray(z1)
    z2 = jnp.asarray(z2)
    if z1.shape != z2.shape:
        raise ValueError(
            f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}."
        )

    K = int(jnp.maximum(z1.max(), z2.max())) + 1
    overlap = jnp.zeros((K, K), dtype=jnp.int32)
    return overlap.at[z1, z2].add(1)


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
    from scipy.optimize import linear_sum_assignment  # deferred: slow import

    overlap = compute_state_overlap(z1, z2)
    _, perm = linear_sum_assignment(-np.asarray(overlap))
    return cast(np.ndarray, perm)
