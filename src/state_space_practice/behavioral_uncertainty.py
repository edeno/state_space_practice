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
    """Entropy of a categorical distribution, in nats.

    ``H(p) = -sum_k p_k log p_k`` with the convention ``0 log 0 = 0``, i.e.
    ``scipy.stats.entropy(p, axis=-1)`` for normalised ``p`` (this function
    does not renormalise). Entries ``<= 0`` contribute exactly zero, so a
    deterministic distribution has entropy exactly 0; the gradient stays finite
    at zero entries.

    Parameters
    ----------
    probs : Array, shape (..., K)
        Probabilities summing to one over the last axis.

    Returns
    -------
    Array, shape (...)
    """
    probs = jnp.asarray(probs)
    positive = probs > 0.0
    # double-where: log is never evaluated at 0, so gradients stay finite
    safe_probs = jnp.where(positive, probs, 1.0)
    return -jnp.sum(jnp.where(positive, probs * jnp.log(safe_probs), 0.0), axis=-1)


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
        -log P(actual choice | predicted). Higher = more surprising. The
        probability is floored at ``1e-10``, so the surprise of a choice the
        model deemed impossible is capped at ``-log(1e-10) ~ 23.03`` nats
        instead of ``inf``.
    """
    eps = 1e-10
    p = jnp.clip(predicted_probs[jnp.arange(len(choices)), choices], eps, 1.0)
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

    Notes
    -----
    By the law of total variance,
    ``Var[r] = E_s[rho (1 - rho)] + Var_s[rho]``, and since ``r`` is binary this
    sums to ``mean * (1 - mean)``: the marginal of a Bernoulli mixture is itself
    Bernoulli. The closed form is used directly, with ``mean`` clipped to
    ``[0, 1]`` against round-off, so unlike the two-term sum (whose
    ``E_s[rho^2] - mean^2`` cancels) it is never negative.
    """
    # E[r | option k] = sum_s P(s) * rho[s, k]
    mean = state_probs @ reward_probs  # (T, K)
    bounded_mean = jnp.clip(mean, 0.0, 1.0)  # mean can exceed 1 by an ulp
    variance = bounded_mean * (1.0 - bounded_mean)
    return mean, variance
