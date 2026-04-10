from typing import Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
from jax import Array

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


def psd_solve(A: jax.Array, b: jax.Array, diagonal_boost: float = 1e-9) -> jax.Array:
    """Solves a linear system Ax = b for positive semi-definite (PSD) matrices A.

    This function wraps a linear algebra solver, ensuring numerical stability
    by symmetrizing the input matrix A and adding a small value to its
    diagonal (diagonal_boost). It is intended for use with PSD matrices,
    where 'assume_a="pos"' can be safely set for performance.

    Parameters
    ----------
    A : jax.Array
        The coefficient matrix, expected to be positive semi-definite.
    b : jax.Array
        The right-hand side vector or matrix.
    diagonal_boost : float, optional
        Small value added to the diagonal of A to improve numerical
        stability. Default is 1e-9.

    Returns
    -------
    jax.Array
        The solution x to the linear system Ax = b.

    """
    return jax.scipy.linalg.solve(
        symmetrize(A) + float(diagonal_boost) * jnp.eye(A.shape[-1], dtype=A.dtype),
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


# ---------------------------------------------------------------------------
# Probability utilities
# ---------------------------------------------------------------------------

# Minimum probability threshold for numerical stability
_LOG_PROB_FLOOR = 1e-10
_LOG_FLOOR_VALUE = -23.0  # approximately log(1e-10)
_DISCRETE_PROB_STABILITY_FLOOR = 1e-10


def divide_safe(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """Divides two arrays, while setting the result to 0.0
    if the denominator is 0.0.

    Parameters
    ----------
    numerator : jax.Array
    denominator : jax.Array
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
    distribution. This is intentional for numerical robustness, but callers
    should validate inputs upstream if they need to detect invalid priors.
    """
    floor = jnp.asarray(_DISCRETE_PROB_STABILITY_FLOOR, dtype=probabilities.dtype)
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
