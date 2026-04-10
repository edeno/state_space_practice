"""Behavioral uncertainty helpers.

Pure functions for computing uncertainty summaries from behavioral model
posteriors. Used by MultinomialChoiceModel, CovariateChoiceModel,
ContingencyBeliefModel, and SwitchingChoiceModel.
"""

import jax.numpy as jnp
from jax import Array


def append_reference_option(values: Array) -> Array:
    """Prepend the reference option (value=0) to K-1 free values.

    Parameters
    ----------
    values : Array, shape (..., K-1)

    Returns
    -------
    Array, shape (..., K)
    """
    zeros = jnp.zeros(values.shape[:-1] + (1,))
    return jnp.concatenate([zeros, values], axis=-1)


def option_variances_from_covariances(covariances: Array) -> Array:
    """Extract per-option variances from covariance matrices.

    Adds zero variance for the reference option.

    Parameters
    ----------
    covariances : Array, shape (..., K-1, K-1)

    Returns
    -------
    Array, shape (..., K)
        Per-option variances with reference option variance = 0.
    """
    diag = jnp.diagonal(covariances, axis1=-2, axis2=-1)  # (..., K-1)
    zeros = jnp.zeros(diag.shape[:-1] + (1,))
    return jnp.concatenate([zeros, diag], axis=-1)


def categorical_entropy(probs: Array) -> Array:
    """Entropy of a categorical distribution.

    Parameters
    ----------
    probs : Array, shape (..., K)

    Returns
    -------
    Array, shape (...)
    """
    eps = 1e-10
    safe_probs = jnp.clip(probs, eps, 1.0)
    return -jnp.sum(safe_probs * jnp.log(safe_probs), axis=-1)


def belief_entropy(state_probs: Array) -> Array:
    """Entropy of the discrete state belief.

    Parameters
    ----------
    state_probs : Array, shape (T, S)

    Returns
    -------
    Array, shape (T,)
    """
    return categorical_entropy(state_probs)


def compute_surprise(predicted_probs: Array, choices: Array) -> Array:
    """Surprise: negative log predictive probability of actual choice.

    Parameters
    ----------
    predicted_probs : Array, shape (T, K)
        Predicted choice probabilities before observing the choice.
    choices : Array, shape (T,)
        Actual choices (0-indexed).

    Returns
    -------
    Array, shape (T,)
        -log P(actual choice | predicted). Higher = more surprising.
    """
    eps = 1e-10
    p = jnp.clip(
        predicted_probs[jnp.arange(len(choices)), choices], eps, 1.0
    )
    return -jnp.log(p)


def pairwise_change_point_probability(pairwise_state_prob: Array) -> Array:
    """True per-trial switch probability P(s_t != s_{t-1} | data).

    Marginalizes off-diagonal mass from the pairwise smoothed joint:

    .. math::
       P(s_t \\neq s_{t-1} \\mid y_{1:T}) = \\sum_{i \\neq j}
       P(s_{t-1}=i,\\, s_t=j \\mid y_{1:T})

    This is the correct quantity for locating putative contingency
    switch points in behavioral data.

    Parameters
    ----------
    pairwise_state_prob : Array, shape (T-1, S, S)
        Smoothed pairwise joint `P(s_{t-1}, s_t | data)` from an HMM
        smoother (e.g., ``contingency_belief_smoother``).
        ``pairwise_state_prob[t, i, j] = P(s_t=i, s_{t+1}=j | data)``.

    Returns
    -------
    Array, shape (T,)
        Switch probability per trial. The first entry is 0 because no
        previous trial exists, and subsequent entries are
        ``1 - sum_i P(s_{t-1}=i, s_t=i | data)`` (off-diagonal mass).
    """
    # Diagonal gives P(s_{t-1} = s_t) for each pair index
    n_states = pairwise_state_prob.shape[-1]
    diag = pairwise_state_prob[..., jnp.arange(n_states), jnp.arange(n_states)]
    stay_prob = jnp.sum(diag, axis=-1)  # (T-1,)
    switch_prob = 1.0 - stay_prob
    # Prepend 0 for trial 0 (no previous trial to switch from)
    return jnp.concatenate([jnp.zeros(1), switch_prob])


def bernoulli_mixture_mean_variance(
    state_probs: Array, reward_probs: Array
) -> tuple[Array, Array]:
    """Expected reward mean and variance under a discrete state mixture.

    Parameters
    ----------
    state_probs : Array, shape (T, S)
    reward_probs : Array, shape (S, K)
        P(reward=1 | state, option).

    Returns
    -------
    mean : Array, shape (T, K)
        Expected reward per option under the state mixture.
    variance : Array, shape (T, K)
        Reward variance per option (includes both Bernoulli variance
        and mixture uncertainty).
    """
    # E[r | option k] = sum_s P(s) * rho[s, k]
    mean = state_probs @ reward_probs  # (T, K)

    # E[r^2 | option k] = E[r | option k] (since r is binary)
    # Var[r | option k] = E_s[rho(1-rho)] + Var_s[rho]
    # = sum_s P(s) * rho_sk * (1 - rho_sk) + sum_s P(s) * (rho_sk - mean_k)^2
    bernoulli_var = state_probs @ (reward_probs * (1 - reward_probs))  # (T, K)
    mean_sq = state_probs @ (reward_probs ** 2)  # (T, K)
    mixture_var = mean_sq - mean ** 2  # Var_s[rho]
    variance = bernoulli_var + mixture_var  # Total variance

    return mean, variance
