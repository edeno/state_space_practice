from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# Type alias for numeric values (scalars, numpy arrays, JAX arrays)
Numeric = Union[float, int, np.ndarray, jax.Array]


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
