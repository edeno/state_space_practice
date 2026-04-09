"""Contingency-belief / latent task-state model.

An input-output HMM that infers hidden contingency or rule states from
bandit behavior (choices and rewards). The hidden state represents what
the animal believes the world rules are — not option values, not
behavioral strategy.

Model:
    s_t ~ Categorical(softmax(eta[s_{t-1}, :] + Gamma[s_{t-1}, :, :] @ h_t))
    r_t | s_t, a_t ~ Bernoulli(rho[s_t, a_t])
    a_t | s_t ~ Categorical(softmax(beta * V[s_t, :] + Theta @ z_t))

where:
    s_t: latent contingency/task state
    a_t: observed choice
    r_t: observed reward (0/1)
    h_t: transition covariates (e.g., session reset, surprise)
    z_t: choice-bias covariates (e.g., stay bias)

References
----------
[1] Smith, A.C. et al. (2004). Dynamic analysis of learning in
    behavioral experiments. J Neuroscience 24(2), 447-461.
"""

import logging
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

logger = logging.getLogger(__name__)


class ContingencyBeliefResult(NamedTuple):
    """Result container for contingency belief filter/smoother."""

    state_posterior: Array  # (n_trials, n_states)
    log_likelihood: Array  # scalar


def transition_logits_to_matrix(logits: Array) -> Array:
    """Convert unconstrained logits to a row-stochastic transition matrix.

    Parameters
    ----------
    logits : Array, shape (n_states, n_states)
        Unconstrained transition logits.

    Returns
    -------
    Array, shape (n_states, n_states)
        Row-stochastic transition matrix.
    """
    return jax.nn.softmax(logits, axis=1)


def compute_input_output_transition_matrix(
    baseline_logits: Array,
    transition_weights: Array,
    covariates_t: Array,
) -> Array:
    """Compute time-varying transition matrix from covariates.

    Parameters
    ----------
    baseline_logits : Array, shape (n_states, n_states)
        Baseline transition logits.
    transition_weights : Array, shape (n_states, n_states, d_h)
        Covariate weights for transitions.
    covariates_t : Array, shape (d_h,)
        Transition covariates at time t.

    Returns
    -------
    Array, shape (n_states, n_states)
        Row-stochastic transition matrix at time t.
    """
    # logits[i, j] = baseline[i, j] + weights[i, j, :] @ h_t
    logits = baseline_logits + jnp.einsum("ijk,k->ij", transition_weights, covariates_t)
    return jax.nn.softmax(logits, axis=1)


def compute_reward_log_likelihood(
    reward_t: ArrayLike,
    choice_t: ArrayLike,
    reward_probs: Array,
) -> Array:
    """Compute log P(reward | state, choice) for each state.

    Parameters
    ----------
    reward_t : int or Array
        Reward outcome (0 or 1).
    choice_t : int or Array
        Chosen option index.
    reward_probs : Array, shape (n_states, n_options)
        P(reward=1 | state, choice) for each state-option pair.

    Returns
    -------
    Array, shape (n_states,)
        Log-likelihood of the reward under each state.
    """
    p = reward_probs[:, choice_t]  # (n_states,)
    eps = 1e-10
    p = jnp.clip(p, eps, 1.0 - eps)
    return jnp.where(
        reward_t == 1,
        jnp.log(p),
        jnp.log(1.0 - p),
    )


def compute_choice_log_likelihood(
    choice_t: ArrayLike,
    state_values: Array,
    inverse_temperature: float,
    n_options: int,
) -> Array:
    """Compute log P(choice | state) for each state.

    Parameters
    ----------
    choice_t : int or Array
        Chosen option index.
    state_values : Array, shape (n_states, n_options)
        Value preferences per state-option pair.
    inverse_temperature : float
        Softmax temperature.
    n_options : int
        Number of options.

    Returns
    -------
    Array, shape (n_states,)
        Log-likelihood of the choice under each state.
    """
    logits = inverse_temperature * state_values  # (n_states, n_options)
    log_probs = jax.nn.log_softmax(logits, axis=1)  # (n_states, n_options)
    return log_probs[:, choice_t]  # (n_states,)


def contingency_belief_filter(
    choices: ArrayLike,
    rewards: ArrayLike,
    n_states: int,
    n_options: int,
    reward_probs: Array,
    state_values: Array,
    inverse_temperature: float = 1.0,
    transition_logits: Optional[Array] = None,
    transition_covariates: Optional[Array] = None,
    transition_weights: Optional[Array] = None,
    init_state_prob: Optional[Array] = None,
) -> ContingencyBeliefResult:
    """Forward filter for the contingency-belief HMM.

    Computes P(s_t | a_{1:t}, r_{1:t}) for each trial t.

    Parameters
    ----------
    choices : ArrayLike, shape (n_trials,)
        Observed choices (0-indexed).
    rewards : ArrayLike, shape (n_trials,)
        Observed rewards (0 or 1).
    n_states : int
        Number of latent contingency states.
    n_options : int
        Number of choice options.
    reward_probs : Array, shape (n_states, n_options)
        P(reward=1 | state, choice).
    state_values : Array, shape (n_states, n_options)
        Choice value preferences per state.
    inverse_temperature : float
        Softmax temperature for choice policy.
    transition_logits : Array or None, shape (n_states, n_states)
        Baseline transition logits. Default: zeros (uniform).
    transition_covariates : Array or None, shape (n_trials, d_h)
        Time-varying transition covariates.
    transition_weights : Array or None, shape (n_states, n_states, d_h)
        Weights for covariate-driven transitions.
    init_state_prob : Array or None, shape (n_states,)
        Initial state distribution. Default: uniform.

    Returns
    -------
    ContingencyBeliefResult
        state_posterior: shape (n_trials, n_states)
        log_likelihood: scalar
    """
    choices = jnp.asarray(choices, dtype=jnp.int32)
    rewards = jnp.asarray(rewards, dtype=jnp.int32)
    n_trials = choices.shape[0]

    if transition_logits is None:
        transition_logits = jnp.zeros((n_states, n_states))
    if init_state_prob is None:
        init_state_prob = jnp.ones(n_states) / n_states

    has_covariates = (
        transition_covariates is not None and transition_weights is not None
    )
    if not has_covariates:
        transition_covariates = jnp.zeros((n_trials, 1))
        transition_weights = jnp.zeros((n_states, n_states, 1))

    def _step(carry, trial_data):
        prev_belief, accum_ll = carry
        choice_t, reward_t, h_t = trial_data

        # Predict: P(s_t) = sum_i T(i→j|h_t) * P(s_{t-1}=i)
        trans = compute_input_output_transition_matrix(
            transition_logits, transition_weights, h_t
        )
        predicted = trans.T @ prev_belief  # (n_states,)

        # Update with reward observation
        reward_ll = compute_reward_log_likelihood(
            reward_t, choice_t, reward_probs
        )  # (n_states,)

        # Update with choice observation
        choice_ll = compute_choice_log_likelihood(
            choice_t, state_values, inverse_temperature, n_options
        )  # (n_states,)

        # Combined log-likelihood per state
        log_obs = reward_ll + choice_ll
        # Numerically stable update: work in log space
        log_joint = jnp.log(jnp.maximum(predicted, 1e-30)) + log_obs
        # Normalize
        log_norm = jax.nn.logsumexp(log_joint)
        posterior = jnp.exp(log_joint - log_norm)

        new_ll = accum_ll + log_norm
        return (posterior, new_ll), posterior

    init_carry = (init_state_prob, jnp.array(0.0))
    inputs = (choices, rewards, transition_covariates)
    (_, total_ll), posteriors = jax.lax.scan(_step, init_carry, inputs)

    return ContingencyBeliefResult(
        state_posterior=posteriors,
        log_likelihood=total_ll,
    )


class SmootherResult(NamedTuple):
    """Result container for contingency belief smoother."""

    smoothed_state_prob: Array  # (n_trials, n_states)
    pairwise_state_prob: Array  # (n_trials-1, n_states, n_states)
    log_likelihood: Array  # scalar


def contingency_belief_smoother(
    choices: ArrayLike,
    rewards: ArrayLike,
    n_states: int,
    n_options: int,
    reward_probs: Array,
    state_values: Array,
    inverse_temperature: float = 1.0,
    transition_logits: Optional[Array] = None,
    transition_covariates: Optional[Array] = None,
    transition_weights: Optional[Array] = None,
    init_state_prob: Optional[Array] = None,
) -> SmootherResult:
    """Forward-backward smoother for the contingency-belief HMM.

    Computes P(s_t | a_{1:T}, r_{1:T}) and P(s_{t-1}, s_t | data).

    Parameters
    ----------
    Same as contingency_belief_filter.

    Returns
    -------
    SmootherResult
        smoothed_state_prob: shape (n_trials, n_states)
        pairwise_state_prob: shape (n_trials-1, n_states, n_states)
        log_likelihood: scalar
    """
    choices = jnp.asarray(choices, dtype=jnp.int32)
    rewards = jnp.asarray(rewards, dtype=jnp.int32)
    n_trials = choices.shape[0]

    if transition_logits is None:
        transition_logits = jnp.zeros((n_states, n_states))
    if init_state_prob is None:
        init_state_prob = jnp.ones(n_states) / n_states

    has_covariates = (
        transition_covariates is not None and transition_weights is not None
    )
    if not has_covariates:
        transition_covariates = jnp.zeros((n_trials, 1))
        transition_weights = jnp.zeros((n_states, n_states, 1))

    # --- Forward pass: store filter beliefs and per-step info ---
    def _forward_step(carry, trial_data):
        prev_belief, accum_ll = carry
        choice_t, reward_t, h_t = trial_data

        trans = compute_input_output_transition_matrix(
            transition_logits, transition_weights, h_t
        )
        predicted = trans.T @ prev_belief

        reward_ll = compute_reward_log_likelihood(
            reward_t, choice_t, reward_probs
        )
        choice_ll = compute_choice_log_likelihood(
            choice_t, state_values, inverse_temperature, n_options
        )
        log_obs = reward_ll + choice_ll
        log_joint = jnp.log(jnp.maximum(predicted, 1e-30)) + log_obs
        log_norm = jax.nn.logsumexp(log_joint)
        posterior = jnp.exp(log_joint - log_norm)

        new_ll = accum_ll + log_norm
        return (posterior, new_ll), (posterior, predicted, trans)

    init_carry = (init_state_prob, jnp.array(0.0))
    inputs = (choices, rewards, transition_covariates)
    (_, total_ll), (filter_beliefs, predicted_beliefs, trans_matrices) = (
        jax.lax.scan(_forward_step, init_carry, inputs)
    )

    # --- Backward pass ---
    def _backward_step(beta_next, step_data):
        filter_t, predicted_tp1, trans_tp1 = step_data
        # beta_next = P(s_{t+1} | data) / P(s_{t+1} | data_{1:t})
        # smoothed_t = filter_t * sum_j T(i→j) * beta_next[j] / predicted[j]
        ratio = beta_next / jnp.maximum(predicted_tp1, 1e-30)
        beta_t = filter_t * (trans_tp1 @ ratio)
        # Normalize for stability
        beta_t = beta_t / jnp.maximum(beta_t.sum(), 1e-30)
        return beta_t, beta_t

    # Backward scan: from T-1 down to 0
    # step_data for backward step t uses filter[t], predicted[t+1], trans[t+1]
    backward_inputs = (
        filter_beliefs[:-1],      # filter[0:T-1]
        predicted_beliefs[1:],    # predicted[1:T]
        trans_matrices[1:],       # trans[1:T]
    )
    _, smoothed_interior = jax.lax.scan(
        _backward_step,
        filter_beliefs[-1],  # init: last filter belief = smoothed belief
        backward_inputs,
        reverse=True,
    )
    # Concatenate: smoothed[0:T-1] from backward + filter[-1] as last
    smoothed = jnp.concatenate(
        [smoothed_interior, filter_beliefs[-1:]], axis=0
    )

    # --- Pairwise state probabilities P(s_{t-1}=i, s_t=j | data) ---
    def _pairwise(step_data):
        smooth_t, predicted_tp1, trans_tp1, smooth_tp1 = step_data
        # P(s_t=i, s_{t+1}=j | data) = smooth[t,i] * T[i,j] * smooth[t+1,j] / predicted[t+1,j]
        ratio = smooth_tp1 / jnp.maximum(predicted_tp1, 1e-30)
        joint = smooth_t[:, None] * trans_tp1 * ratio[None, :]
        # Normalize
        joint = joint / jnp.maximum(joint.sum(), 1e-30)
        return joint

    pairwise_inputs = (
        smoothed[:-1],
        predicted_beliefs[1:],
        trans_matrices[1:],
        smoothed[1:],
    )
    pairwise = jax.vmap(_pairwise)(pairwise_inputs)

    return SmootherResult(
        smoothed_state_prob=smoothed,
        pairwise_state_prob=pairwise,
        log_likelihood=total_ll,
    )
