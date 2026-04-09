"""Switching choice model: strategy-dependent multi-armed bandit.

Discrete latent states represent behavioral strategies (e.g., exploit
vs explore) that control how continuous option values evolve and drive
choices. Combines the softmax observation model from multinomial_choice
with the GPB2 switching infrastructure from switching_kalman.

Model:
    s_t ~ Categorical(T[s_{t-1}, :])
    x_t = decay_{s_t} * x_{t-1} + B @ u_t + w_t,  w_t ~ N(0, q_{s_t} * I)
    c_t ~ softmax(beta_{s_t} * [0, x_t] + Theta @ z_t)

Per-state parameters: beta_s, Q_s, decay_s
Shared parameters: B (input gain), Theta (obs weights), init_mean

References
----------
[1] Linderman et al. (2017). Bayesian learning and inference in recurrent
    switching linear dynamical systems. AISTATS.
[2] Smith et al. (2004). Dynamic analysis of learning in behavioral
    experiments. J Neuroscience 24(2), 447-461.
"""

import logging
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.multinomial_choice import _softmax_update_core
from state_space_practice.switching_kalman import (
    _scale_likelihood,
    _stabilize_probability_vector,
    _update_discrete_state_probabilities,
    collapse_gaussian_mixture_per_discrete_state,
)

logger = logging.getLogger(__name__)


def _softmax_predict_and_update(
    prev_mean: Array,
    prev_cov: Array,
    choice: Array,
    transition_matrix: Array,
    process_cov: Array,
    n_options: int,
    inverse_temperature: float,
    input_gain: Array,
    covariates_t: Array,
    obs_offset: Array,
) -> tuple[Array, Array, Array]:
    """Predict + softmax update for one (prev_state_i, next_state_j) pair.

    Parameters
    ----------
    prev_mean : Array, shape (K-1,)
    prev_cov : Array, shape (K-1, K-1)
    choice : Array, scalar int
    transition_matrix : Array, shape (K-1, K-1) — A_j = decay_j * I
    process_cov : Array, shape (K-1, K-1) — Q_j
    n_options : int
    inverse_temperature : float — beta_j
    input_gain : Array, shape (K-1, d) — shared B
    covariates_t : Array, shape (d,)
    obs_offset : Array, shape (K,) — Theta @ z_t

    Returns
    -------
    post_mean : Array, shape (K-1,)
    post_cov : Array, shape (K-1, K-1)
    log_likelihood : Array, shape ()
    """
    # Predict
    pred_mean = transition_matrix @ prev_mean + input_gain @ covariates_t
    pred_cov = (
        transition_matrix @ prev_cov @ transition_matrix.T + process_cov
    )

    # Update via softmax Laplace-EKF
    post_mean, post_cov, _ll_prior = _softmax_update_core(
        pred_mean, pred_cov, choice, n_options, inverse_temperature,
        obs_offset=obs_offset,
    )

    # Recompute LL at the MAP mode (not prior mean) for correct
    # discrete-state weighting in the switching filter
    v_mode = jnp.concatenate([jnp.zeros(1), post_mean])
    _offset = obs_offset if obs_offset is not None else jnp.zeros(n_options)
    ll = jax.nn.log_softmax(inverse_temperature * v_mode + _offset)[choice]

    return post_mean, post_cov, ll


def _softmax_update_per_state_pair(
    prev_state_cond_mean: Array,
    prev_state_cond_cov: Array,
    choice: Array,
    transition_matrices: Array,
    process_covs: Array,
    n_options: int,
    inverse_temperatures: Array,
    input_gain: Array,
    covariates_t: Array,
    obs_offset: Array,
) -> tuple[Array, Array, Array]:
    """Per-state-pair softmax predict + update via double vmap.

    Computes pair-conditional posteriors for all (i, j) state pairs,
    following the switching_point_process.py pattern.

    Parameters
    ----------
    prev_state_cond_mean : Array, shape (K-1, S)
        State-conditional means from previous trial.
    prev_state_cond_cov : Array, shape (K-1, K-1, S)
        State-conditional covariances from previous trial.
    choice : Array, scalar int
    transition_matrices : Array, shape (K-1, K-1, S)
        Per-state A_j = decay_j * I.
    process_covs : Array, shape (K-1, K-1, S)
        Per-state Q_j.
    n_options : int
    inverse_temperatures : Array, shape (S,)
        Per-state beta_j.
    input_gain : Array, shape (K-1, d) — shared B
    covariates_t : Array, shape (d,)
    obs_offset : Array, shape (K,) — shared Theta @ z_t

    Returns
    -------
    pair_cond_mean : Array, shape (K-1, S_prev, S_next)
    pair_cond_cov : Array, shape (K-1, K-1, S_prev, S_next)
    pair_cond_ll : Array, shape (S_prev, S_next)
    """

    def _update_one_pair(prev_mean_i, prev_cov_i, A_j, Q_j, beta_j):
        """Update for one (i, j) pair."""
        return _softmax_predict_and_update(
            prev_mean_i, prev_cov_i, choice,
            A_j, Q_j, n_options, beta_j,
            input_gain, covariates_t, obs_offset,
        )

    # vmap over prev_state i (axis -1 of mean/cov)
    def _update_all_prev_for_next_j(A_j, Q_j, beta_j):
        # vmap over prev state i
        return jax.vmap(
            lambda m, c: _update_one_pair(m, c, A_j, Q_j, beta_j),
            in_axes=(1, 2),  # mean: axis 1, cov: axis 2
            out_axes=(1, 2, 0),  # mean: (K-1, S_prev), cov: (K-1, K-1, S_prev), ll: (S_prev,)
        )(prev_state_cond_mean, prev_state_cond_cov)

    # vmap over next_state j (axis -1 of A/Q, element of beta)
    pair_mean, pair_cov, pair_ll = jax.vmap(
        _update_all_prev_for_next_j,
        in_axes=(2, 2, 0),  # A: axis 2, Q: axis 2, beta: axis 0
        out_axes=(2, 3, 1),  # mean: (K-1, S_prev, S_next), cov: (K-1, K-1, S_prev, S_next), ll: (S_prev, S_next)
    )(transition_matrices, process_covs, inverse_temperatures)

    return pair_mean, pair_cov, pair_ll


class SwitchingChoiceFilterResult(NamedTuple):
    """Result container for switching choice filter."""

    filtered_values: Array        # (n_trials, K-1, S) state-conditional
    filtered_covs: Array          # (n_trials, K-1, K-1, S) state-conditional
    discrete_state_probs: Array   # (n_trials, S)
    marginal_log_likelihood: Array  # scalar
    pair_cond_means: Array        # (n_trials, K-1, S, S) for smoother
    pair_cond_covs: Array         # (n_trials, K-1, K-1, S, S) for smoother


def switching_choice_filter(
    choices: ArrayLike,
    n_options: int,
    n_discrete_states: int = 2,
    covariates: Optional[ArrayLike] = None,
    input_gain: Optional[ArrayLike] = None,
    obs_covariates: Optional[ArrayLike] = None,
    obs_weights: Optional[ArrayLike] = None,
    process_noises: Optional[ArrayLike] = None,
    inverse_temperatures: Optional[ArrayLike] = None,
    decays: Optional[ArrayLike] = None,
    discrete_transition_matrix: Optional[ArrayLike] = None,
    init_mean: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
    init_discrete_prob: Optional[ArrayLike] = None,
) -> SwitchingChoiceFilterResult:
    """Switching choice filter with GPB2 approximation.

    Parameters
    ----------
    choices : ArrayLike, shape (n_trials,)
        Observed choices (0-indexed).
    n_options : int
    n_discrete_states : int
    covariates : ArrayLike or None, shape (n_trials, d_dyn)
    input_gain : ArrayLike or None, shape (K-1, d_dyn)
    obs_covariates : ArrayLike or None, shape (n_trials, d_obs)
    obs_weights : ArrayLike or None, shape (K, d_obs)
    process_noises : ArrayLike or None, shape (S,) — per-state scalar Q
    inverse_temperatures : ArrayLike or None, shape (S,) — per-state beta
    decays : ArrayLike or None, shape (S,) — per-state decay
    discrete_transition_matrix : ArrayLike or None, shape (S, S)
    init_mean : ArrayLike or None, shape (K-1,) — shared across states
    init_cov : ArrayLike or None, shape (K-1, K-1)
    init_discrete_prob : ArrayLike or None, shape (S,)

    Returns
    -------
    SwitchingChoiceFilterResult
    """
    choices = jnp.asarray(choices, dtype=jnp.int32)
    n_trials = choices.shape[0]
    k_free = n_options - 1
    S = n_discrete_states

    # Defaults
    if process_noises is None:
        process_noises = jnp.ones(S) * 0.01
    else:
        process_noises = jnp.asarray(process_noises)
    if inverse_temperatures is None:
        inverse_temperatures = jnp.ones(S)
    else:
        inverse_temperatures = jnp.asarray(inverse_temperatures)
    if decays is None:
        decays = jnp.ones(S)
    else:
        decays = jnp.asarray(decays)
    if discrete_transition_matrix is None:
        discrete_transition_matrix = (
            0.9 * jnp.eye(S) + 0.1 / S * jnp.ones((S, S))
        )
    else:
        discrete_transition_matrix = jnp.asarray(discrete_transition_matrix)
    if init_mean is None:
        init_mean = jnp.zeros(k_free)
    else:
        init_mean = jnp.asarray(init_mean)
    if init_cov is None:
        init_cov = jnp.eye(k_free)
    else:
        init_cov = jnp.asarray(init_cov)
    if init_discrete_prob is None:
        init_discrete_prob = jnp.ones(S) / S
    else:
        init_discrete_prob = jnp.asarray(init_discrete_prob)

    # Covariates
    if covariates is not None:
        cov_arr = jnp.asarray(covariates)
        ig_arr = jnp.asarray(input_gain) if input_gain is not None else jnp.zeros((k_free, 1))
    else:
        if input_gain is not None:
            raise ValueError("input_gain provided but covariates is None")
        cov_arr = jnp.zeros((n_trials, 1))
        ig_arr = jnp.zeros((k_free, 1))

    if obs_covariates is not None and obs_weights is not None:
        obs_cov_arr = jnp.asarray(obs_covariates)
        ow_arr = jnp.asarray(obs_weights)
    else:
        if obs_covariates is not None or obs_weights is not None:
            raise ValueError(
                "obs_covariates and obs_weights must both be provided or both None"
            )
        obs_cov_arr = jnp.zeros((n_trials, 1))
        ow_arr = jnp.zeros((n_options, 1))

    # Build per-state transition and process cov matrices
    # A_j = decay_j * I, Q_j = q_j * I
    transition_matrices = jnp.stack(
        [d * jnp.eye(k_free) for d in decays], axis=-1
    )  # (K-1, K-1, S)
    process_covs = jnp.stack(
        [q * jnp.eye(k_free) for q in process_noises], axis=-1
    )  # (K-1, K-1, S)

    # Expand init_mean/cov to per-state (shared)
    init_state_cond_mean = jnp.stack([init_mean] * S, axis=-1)  # (K-1, S)
    init_state_cond_cov = jnp.stack([init_cov] * S, axis=-1)  # (K-1, K-1, S)

    init_discrete_prob = _stabilize_probability_vector(init_discrete_prob)

    # --- First timestep: update only (no dynamics prediction) ---
    def _first_update_for_state(prior_mean, prior_cov, beta):
        obs_offset_0 = ow_arr @ obs_cov_arr[0]
        post_mean, post_cov, _ll_prior = _softmax_update_core(
            prior_mean, prior_cov, choices[0], n_options, beta,
            obs_offset=obs_offset_0,
        )
        # LL at mode for discrete-state weighting
        v_mode = jnp.concatenate([jnp.zeros(1), post_mean])
        _off = obs_offset_0 if obs_offset_0 is not None else jnp.zeros(n_options)
        ll = jax.nn.log_softmax(beta * v_mode + _off)[choices[0]]
        return post_mean, post_cov, ll

    first_means, first_covs, first_lls = jax.vmap(
        _first_update_for_state,
        in_axes=(1, 2, 0),
        out_axes=(1, 2, 0),
    )(init_state_cond_mean, init_state_cond_cov, inverse_temperatures)
    # first_means: (K-1, S), first_covs: (K-1, K-1, S), first_lls: (S,)

    # Scale and update discrete probs for first timestep
    # Convert per-state LL to pair format for reuse of _update functions
    first_pair_ll = first_lls[None, :] * jnp.ones((S, 1))  # (S, S) broadcast
    first_ll_scaled, first_ll_max = _scale_likelihood(first_pair_ll)
    first_discrete_prob, _, first_pred_sum = _update_discrete_state_probabilities(
        first_ll_scaled, jnp.eye(S), init_discrete_prob,
    )
    first_marginal_ll = first_ll_max + jnp.log(first_pred_sum)

    # Pair-conditional for smoother: diagonal (no pair structure at t=0)
    first_pair_mean = jnp.stack([first_means] * S, axis=-1)  # (K-1, S, S)
    first_pair_cov = jnp.stack([first_covs] * S, axis=-1)  # (K-1, K-1, S, S)

    # --- Scan over t=1..T-1 ---
    def _step(carry, trial_data):
        prev_mean, prev_cov, prev_disc_prob, accum_ll = carry
        choice_t, u_t, z_t = trial_data

        obs_offset_t = ow_arr @ z_t  # (K,)

        # Per-state-pair predict + update
        pair_mean, pair_cov, pair_ll = _softmax_update_per_state_pair(
            prev_mean, prev_cov, choice_t,
            transition_matrices, process_covs, n_options,
            inverse_temperatures, ig_arr, u_t, obs_offset_t,
        )

        # Scale likelihood
        pair_ll_scaled, ll_max = _scale_likelihood(pair_ll)

        # Update discrete state probs
        disc_prob, backward_prob, pred_sum = _update_discrete_state_probabilities(
            pair_ll_scaled, discrete_transition_matrix, prev_disc_prob,
        )

        # Accumulate LL
        new_ll = accum_ll + ll_max + jnp.log(pred_sum)

        # Collapse mixtures
        state_mean, state_cov = collapse_gaussian_mixture_per_discrete_state(
            pair_mean, pair_cov, backward_prob,
        )

        return (state_mean, state_cov, disc_prob, new_ll), (
            state_mean, state_cov, disc_prob, pair_mean, pair_cov,
        )

    init_carry = (first_means, first_covs, first_discrete_prob, first_marginal_ll)
    scan_inputs = (choices[1:], cov_arr[1:], obs_cov_arr[1:])

    (_, _, _, total_ll), (
        rest_means, rest_covs, rest_disc_probs, rest_pair_means, rest_pair_covs,
    ) = jax.lax.scan(_step, init_carry, scan_inputs)

    # Concatenate first timestep
    filtered_values = jnp.concatenate(
        [first_means[None], rest_means], axis=0
    )
    filtered_covs = jnp.concatenate(
        [first_covs[None], rest_covs], axis=0
    )
    discrete_state_probs = jnp.concatenate(
        [first_discrete_prob[None], rest_disc_probs], axis=0
    )
    pair_cond_means = jnp.concatenate(
        [first_pair_mean[None], rest_pair_means], axis=0
    )
    pair_cond_covs = jnp.concatenate(
        [first_pair_cov[None], rest_pair_covs], axis=0
    )

    return SwitchingChoiceFilterResult(
        filtered_values=filtered_values,
        filtered_covs=filtered_covs,
        discrete_state_probs=discrete_state_probs,
        marginal_log_likelihood=total_ll,
        pair_cond_means=pair_cond_means,
        pair_cond_covs=pair_cond_covs,
    )
