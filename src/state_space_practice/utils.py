from typing import Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
from jax import Array
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
