"""Switching choice model: strategy-dependent multi-armed bandit.

Discrete latent states represent behavioral strategies (e.g., exploit
vs explore) that control how continuous option values evolve and drive
choices. Combines the softmax observation model from multinomial_choice
with the GPB1/IMM switching infrastructure from switching_kalman.

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
import functools
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.multinomial_choice import _softmax_update_core
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.switching_kalman import (
    _update_discrete_state_probabilities,
    collapse_gaussian_mixture_per_discrete_state,
)
from state_space_practice.utils import scale_likelihood as _scale_likelihood
from state_space_practice.utils import (
    stabilize_probability_vector as _stabilize_probability_vector,
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
    post_mean, post_cov, ll = _softmax_update_core(
        pred_mean, pred_cov, choice, n_options, inverse_temperature,
        obs_offset=obs_offset,
    )

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

    filtered_values: Array        # (n_trials, K-1, S) state-conditional posterior
    filtered_covs: Array          # (n_trials, K-1, K-1, S) state-conditional posterior
    predicted_values: Array       # (n_trials, K-1, S) state-conditional predicted (prior)
    predicted_covs: Array         # (n_trials, K-1, K-1, S) state-conditional predicted
    discrete_state_probs: Array   # (n_trials, S) filtered posterior
    marginal_log_likelihood: Array  # scalar
    pair_cond_means: Array        # (n_trials, K-1, S, S) for smoother
    pair_cond_covs: Array         # (n_trials, K-1, K-1, S, S) for smoother


@functools.partial(
    jax.jit,
    static_argnames=["n_options", "n_discrete_states"],
)
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
    """Switching choice filter with GPB1/IMM approximation.

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
    transition_matrices = decays[None, None, :] * jnp.eye(k_free)[:, :, None]  # (K-1, K-1, S)
    process_covs = process_noises[None, None, :] * jnp.eye(k_free)[:, :, None]  # (K-1, K-1, S)

    # Expand init_mean/cov to per-state (shared)
    init_state_cond_mean = jnp.stack([init_mean] * S, axis=-1)  # (K-1, S)
    init_state_cond_cov = jnp.stack([init_cov] * S, axis=-1)  # (K-1, K-1, S)

    init_discrete_prob = _stabilize_probability_vector(init_discrete_prob)

    # --- First timestep: predict + update (x₀ convention) ---
    # This uses the x₀ convention (predict+update at t=0) to match the
    # non-switching CovariateChoiceModel, NOT the x₁ convention used by
    # switching_kalman_filter and switching_point_process_filter (which
    # skip prediction at t=0). The x₀ convention means init_mean is
    # the prior BEFORE the first observation, not AT it. The smoother
    # still works because it only uses filter outputs, not the convention.
    def _first_update_for_state(prior_mean, prior_cov, beta, A_j, Q_j):
        pred_mean = A_j @ prior_mean + ig_arr @ cov_arr[0]
        pred_cov = A_j @ prior_cov @ A_j.T + Q_j

        obs_offset_0 = ow_arr @ obs_cov_arr[0]
        post_mean, post_cov, ll = _softmax_update_core(
            pred_mean, pred_cov, choices[0], n_options, beta,
            obs_offset=obs_offset_0,
        )
        return post_mean, post_cov, ll, pred_mean, pred_cov

    first_means, first_covs, first_lls, first_pred_means, first_pred_covs = jax.vmap(
        _first_update_for_state,
        in_axes=(1, 2, 0, 2, 2),
        out_axes=(1, 2, 0, 1, 2),
    )(init_state_cond_mean, init_state_cond_cov, inverse_temperatures,
      transition_matrices, process_covs)
    # first_means: (K-1, S), first_pred_means: (K-1, S), etc.

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

        # Compute per-state predicted (prior) means and covs
        def _predict_for_state_j(prev_m, prev_c, A_j, Q_j):
            pred_m = A_j @ prev_m + ig_arr @ u_t
            pred_c = A_j @ prev_c @ A_j.T + Q_j
            return pred_m, pred_c

        pred_means, pred_covs = jax.vmap(
            _predict_for_state_j,
            in_axes=(1, 2, 2, 2),
            out_axes=(1, 2),
        )(prev_mean, prev_cov, transition_matrices, process_covs)

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
            pred_means, pred_covs,
        )

    init_carry = (first_means, first_covs, first_discrete_prob, first_marginal_ll)
    scan_inputs = (choices[1:], cov_arr[1:], obs_cov_arr[1:])

    (_, _, _, total_ll), (
        rest_means, rest_covs, rest_disc_probs, rest_pair_means, rest_pair_covs,
        rest_pred_means, rest_pred_covs,
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
    predicted_values = jnp.concatenate(
        [first_pred_means[None], rest_pred_means], axis=0
    )
    predicted_covs = jnp.concatenate(
        [first_pred_covs[None], rest_pred_covs], axis=0
    )

    return SwitchingChoiceFilterResult(
        filtered_values=filtered_values,
        filtered_covs=filtered_covs,
        predicted_values=predicted_values,
        predicted_covs=predicted_covs,
        discrete_state_probs=discrete_state_probs,
        marginal_log_likelihood=total_ll,
        pair_cond_means=pair_cond_means,
        pair_cond_covs=pair_cond_covs,
    )


class SwitchingChoiceModel(SGDFittableMixin):
    """Switching multi-armed bandit with per-state learning dynamics.

    Discrete latent states represent behavioral strategies (e.g.,
    exploit vs explore) that control how option values evolve.

    Parameters
    ----------
    n_options : int
        Number of choice options K.
    n_discrete_states : int
        Number of discrete behavioral states S.
    n_covariates : int
        Number of dynamics covariates.
    n_obs_covariates : int
        Number of observation covariates.
    init_inverse_temperatures : Array or None, shape (S,)
        Per-state starting inverse temperatures.
    init_process_noises : Array or None, shape (S,)
        Per-state starting process noises.
    init_decays : Array or None, shape (S,)
        Per-state starting decays.
    """

    def __init__(
        self,
        n_options: int,
        n_discrete_states: int = 2,
        n_covariates: int = 0,
        n_obs_covariates: int = 0,
        init_inverse_temperatures: Optional[ArrayLike] = None,
        init_process_noises: Optional[ArrayLike] = None,
        init_decays: Optional[ArrayLike] = None,
    ):
        self.n_options = n_options
        self.n_discrete_states = n_discrete_states
        self.n_covariates = n_covariates
        self.n_obs_covariates = n_obs_covariates
        k_free = n_options - 1
        S = n_discrete_states

        # Per-state parameters
        if init_inverse_temperatures is not None:
            self.inverse_temperatures_ = jnp.asarray(init_inverse_temperatures)
        else:
            self.inverse_temperatures_ = jnp.ones(S)
        if init_process_noises is not None:
            self.process_noises_ = jnp.asarray(init_process_noises)
        else:
            self.process_noises_ = jnp.ones(S) * 0.01
        if init_decays is not None:
            self.decays_ = jnp.asarray(init_decays)
        else:
            self.decays_ = jnp.ones(S)

        # Shared parameters
        self.init_mean_ = jnp.zeros(k_free)
        self.init_cov_ = jnp.eye(k_free)
        self.discrete_transition_matrix_ = (
            0.9 * jnp.eye(S) + 0.1 / S * jnp.ones((S, S))
        )
        if n_covariates > 0:
            self.input_gain_ = jnp.zeros((k_free, n_covariates))
        else:
            self.input_gain_ = None
        if n_obs_covariates > 0:
            self.obs_weights_ = jnp.zeros((n_options, n_obs_covariates))
        else:
            self.obs_weights_ = None

        # Fitted state
        self._filter_result: Optional[SwitchingChoiceFilterResult] = None
        self.smoothed_discrete_probs_: Optional[Array] = None
        self.log_likelihood_: Optional[float] = None
        self.log_likelihood_history_: Optional[list[float]] = None
        self._n_trials: Optional[int] = None

        # Uncertainty summaries
        self.predicted_option_variances_: Optional[Array] = None
        self.smoothed_option_variances_: Optional[Array] = None
        self.predicted_choice_entropy_: Optional[Array] = None
        self.surprise_: Optional[Array] = None
        self.per_state_predicted_variances_: Optional[Array] = None

    @property
    def is_fitted(self) -> bool:
        return self._filter_result is not None

    def _populate_uncertainty(self, choices: Array) -> None:
        """Compute uncertainty summaries from filter result."""
        from state_space_practice.behavioral_uncertainty import (
            append_reference_option,
            categorical_entropy,
            compute_surprise,
        )

        if self._filter_result is None:
            return

        result = self._filter_result
        disc_probs = result.discrete_state_probs  # (T, S) — filtered posterior

        # Predicted (prior) discrete state probs: P(s_t | y_{1:t-1}).
        # Reconstruct from lagged filtered posterior + transition matrix.
        init_prob = jnp.ones(self.n_discrete_states) / self.n_discrete_states
        predicted_disc = jnp.concatenate([
            init_prob[None, :],
            (disc_probs[:-1] @ self.discrete_transition_matrix_),
        ], axis=0)  # (T, S)

        per_state_values = result.predicted_values  # (T, K-1, S) — prior, not posterior

        # Per-state predicted variances (diagonal of covariance).
        # predicted_covs has shape (T, K-1, K-1, S); we need (T, K-1, S).
        # jnp.diagonal appends the diagonal as the last axis, so calling it
        # with axis1=1, axis2=2 yields (T, S, K-1), not (T, K-1, S).
        per_state_vars = jnp.einsum(
            "tiis->tis", result.predicted_covs
        )  # (T, K-1, S)

        # Zero for reference option, then full K
        zero_ref = jnp.zeros((per_state_vars.shape[0], 1, per_state_vars.shape[2]))
        full_vars = jnp.concatenate([zero_ref, per_state_vars], axis=1)  # (T, K, S)
        self.per_state_predicted_variances_ = full_vars  # (T, K, S)

        # Per-state predicted means (full K with reference option)
        full_means = jnp.concatenate([
            jnp.zeros((per_state_values.shape[0], 1, per_state_values.shape[2])),
            per_state_values,
        ], axis=1)  # (T, K, S)

        # Law of total variance: Var(x) = E[Var(x|s)] + Var(E[x|s])
        # Use PREDICTED (prior) state probs for weighting.
        e_var = jnp.einsum("tks,ts->tk", full_vars, predicted_disc)  # E[Var(x|s)]
        e_mean = jnp.einsum("tks,ts->tk", full_means, predicted_disc)  # E[E[x|s]]
        e_mean_sq = jnp.einsum("tks,ts->tk", full_means ** 2, predicted_disc)
        var_mean = e_mean_sq - e_mean ** 2  # Var(E[x|s])
        self.predicted_option_variances_ = e_var + var_mean

        # Smoothed variances: law of total variance with smoother quantities
        if hasattr(self, '_smoother_state_cond_covs') and self._smoother_state_cond_covs is not None:
            # See note above on diagonal axis ordering; use einsum to get (T, K-1, S).
            smoother_diag = jnp.einsum(
                "tiis->tis", self._smoother_state_cond_covs
            )  # (T, K-1, S)
            zero_ref_sm = jnp.zeros((smoother_diag.shape[0], 1, smoother_diag.shape[2]))
            full_sm_vars = jnp.concatenate([zero_ref_sm, smoother_diag], axis=1)

            # Smoother state-conditional means
            smoother_means = getattr(self, '_smoother_state_cond_means', None)
            if smoother_means is not None:
                full_sm_means = jnp.concatenate([
                    jnp.zeros((smoother_means.shape[0], 1, smoother_means.shape[2])),
                    smoother_means,
                ], axis=1)  # (T, K, S)
                sm_disc = self.smoothed_discrete_probs_
                sm_e_var = jnp.einsum("tks,ts->tk", full_sm_vars, sm_disc)
                sm_e_mean = jnp.einsum("tks,ts->tk", full_sm_means, sm_disc)
                sm_e_mean_sq = jnp.einsum("tks,ts->tk", full_sm_means ** 2, sm_disc)
                sm_var_mean = sm_e_mean_sq - sm_e_mean ** 2
                self.smoothed_option_variances_ = sm_e_var + sm_var_mean
            else:
                # Fallback: no between-state term
                self.smoothed_option_variances_ = jnp.einsum(
                    "tks,ts->tk", full_sm_vars, self.smoothed_discrete_probs_
                )
        else:
            self.smoothed_option_variances_ = self.predicted_option_variances_

        # Obs offset: Theta @ z_t, shape (T, K) or zeros if no obs covariates
        if self.obs_weights_ is not None and self._obs_covariates is not None:
            obs_offsets = self._obs_covariates @ self.obs_weights_.T  # (T, K)
        else:
            obs_offsets = jnp.zeros((per_state_values.shape[0], self.n_options))

        per_state_probs = []
        for s in range(self.n_discrete_states):
            v = append_reference_option(per_state_values[:, :, s])
            p = jax.nn.softmax(self.inverse_temperatures_[s] * v + obs_offsets, axis=1)
            per_state_probs.append(p)
        per_state_probs = jnp.stack(per_state_probs, axis=-1)  # (T, K, S)
        predicted_probs = jnp.einsum("tks,ts->tk", per_state_probs, predicted_disc)

        self.predicted_choice_entropy_ = categorical_entropy(predicted_probs)
        self.surprise_ = compute_surprise(predicted_probs, choices)

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted else "not fitted"
        return (
            f"SwitchingChoiceModel(n_options={self.n_options}, "
            f"n_discrete_states={self.n_discrete_states}, {fitted})"
        )

    def _run_filter(self, choices, covariates=None, obs_covariates=None):
        """Run the switching choice filter with current parameters."""
        kwargs = dict(
            choices=choices,
            n_options=self.n_options,
            n_discrete_states=self.n_discrete_states,
            process_noises=self.process_noises_,
            inverse_temperatures=self.inverse_temperatures_,
            decays=self.decays_,
            discrete_transition_matrix=self.discrete_transition_matrix_,
            init_mean=self.init_mean_,
            init_cov=self.init_cov_,
        )
        if covariates is not None and self.input_gain_ is not None:
            kwargs["covariates"] = covariates
            kwargs["input_gain"] = self.input_gain_
        if obs_covariates is not None and self.obs_weights_ is not None:
            kwargs["obs_covariates"] = obs_covariates
            kwargs["obs_weights"] = self.obs_weights_
        return switching_choice_filter(**kwargs)

    def fit(
        self,
        choices: ArrayLike,
        covariates: Optional[ArrayLike] = None,
        obs_covariates: Optional[ArrayLike] = None,
        max_iter: int = 50,
        tolerance: float = 1e-4,
    ) -> list[float]:
        """Fit via simplified EM algorithm.

        EM updates per-state process_noises and discrete_transition_matrix.
        Per-state inverse_temperatures and decays are NOT updated (no
        closed-form M-step). Use fit_sgd() for full parameter learning.

        Parameters
        ----------
        choices : ArrayLike, shape (n_trials,)
        covariates : ArrayLike or None, shape (n_trials, d_dyn)
        obs_covariates : ArrayLike or None, shape (n_trials, d_obs)
        max_iter : int
        tolerance : float

        Returns
        -------
        log_likelihoods : list of float
        """
        choices = jnp.asarray(choices, dtype=jnp.int32)
        self._n_trials = int(choices.shape[0])
        self._covariates = jnp.asarray(covariates) if covariates is not None else None
        self._obs_covariates = jnp.asarray(obs_covariates) if obs_covariates is not None else None

        log_likelihoods: list[float] = []
        prev_ll = float("-inf")

        for iteration in range(max_iter):
            # E-step: filter + smoother
            result = self._run_filter(choices, self._covariates, self._obs_covariates)
            self._filter_result = result
            smoother_result = self._run_smoother(result)
            ll = float(result.marginal_log_likelihood)
            log_likelihoods.append(ll)

            if abs(ll - prev_ll) < tolerance and iteration > 0:
                logger.info(f"Converged at iteration {iteration + 1}")
                break
            prev_ll = ll

            # M-step
            self._m_step(choices, result, smoother_result)

        self.smoothed_discrete_probs_ = smoother_result[2]
        self._smoother_state_cond_means = smoother_result[5]  # (T, K-1, S)
        self._smoother_state_cond_covs = smoother_result[6]  # (T, K-1, K-1, S)
        self.log_likelihood_ = log_likelihoods[-1]
        self.log_likelihood_history_ = log_likelihoods
        self._populate_uncertainty(choices)
        return log_likelihoods

    def _run_smoother(self, filter_result):
        """Run the switching Kalman smoother on filter output."""
        from state_space_practice.switching_kalman import switching_kalman_smoother

        k_free = self.n_options - 1
        return switching_kalman_smoother(
            filter_mean=filter_result.filtered_values,
            filter_cov=filter_result.filtered_covs,
            filter_discrete_state_prob=filter_result.discrete_state_probs,
            last_filter_conditional_cont_mean=filter_result.pair_cond_means[-1],
            process_cov=jnp.stack(
                [q * jnp.eye(k_free) for q in self.process_noises_], axis=-1,
            ),
            continuous_transition_matrix=jnp.stack(
                [d * jnp.eye(k_free) for d in self.decays_], axis=-1,
            ),
            discrete_state_transition_matrix=self.discrete_transition_matrix_,
        )

    def _m_step(self, choices, filter_result, smoother_result):
        """M-step: update per-state Q and transition matrix.

        Uses smoother quantities throughout (approximate EM via GPB1/IMM):
        - smoother_joint_discrete_state_prob for transition matrix
        - state_cond_smoother_means/covs + cross-covs for Q

        Note: does NOT delegate to switching_kalman_maximization_step
        because the choice model uses scalar per-state parameters
        (decay, Q, beta) rather than full matrices. The maximization_step
        expects matrix A/Q and returns matrices. Per-state beta and decay
        have no closed-form M-step and are learned via SGD only.
        """
        gamma = smoother_result[2]  # smoothed discrete probs (T, S)
        joint = smoother_result[3]  # smoother joint (T-1, S, S)
        smoother_means = smoother_result[5]  # state-conditional smoother means (T, K-1, S)
        smoother_covs = smoother_result[6]   # state-conditional smoother covs (T, K-1, K-1, S)
        pair_cross_covs = smoother_result[7]  # pair-cond cross-covs (T-1, K-1, K-1, S, S)
        S = self.n_discrete_states
        k_free = self.n_options - 1
        eps = 1e-10

        # Deterministic input: B @ u_t for each t
        if self.input_gain_ is not None and self._covariates is not None:
            Bu = self._covariates @ self.input_gain_.T  # (T, K-1)
        else:
            Bu = jnp.zeros((smoother_means.shape[0], k_free))

        # Per-state process noise: E[||x_t - A_s x_{t-1} - B u_t||^2 | y_{1:T}]
        # weighted by P(S_t=s | y_{1:T}).
        #
        # The correct weight for Q_s aggregates over ALL previous states:
        #   w_t = sum_i P(S_{t-1}=i, S_t=s | y_{1:T})
        # and the cross-covariance terms must similarly aggregate over
        # (i, s) pairs.  See switching_kalman_maximization_step for the
        # reference implementation using full sufficient statistics.
        for s in range(S):
            # Weight: sum over previous states i of joint P(S_{t-1}=i, S_t=s)
            w = joint[:, :, s].sum(axis=1)  # (T-1,): sum_i P(S_{t-1}=i, S_t=s)
            w_sum = jnp.maximum(jnp.sum(w), eps)
            decay_s = self.decays_[s]

            # Mean residual: E[x_t|S_t=s] - decay_s * E[x_{t-1}|S_t=s] - B u_t
            # For the previous-state mean, marginalize over S_{t-1} using
            # the backward conditional P(S_{t-1}=i | S_t=s, y_{1:T}).
            # As an approximation consistent with the GPB1 collapsed means,
            # we use the state-conditional smoother mean at s directly.
            mean_resid = (
                smoother_means[1:, :, s]
                - decay_s * smoother_means[:-1, :, s]
                - Bu[1:]
            )

            # Covariance correction: aggregate cross-covs over all (i, s) pairs
            P_t = smoother_covs[1:, :, :, s]       # (T-1, K-1, K-1)
            P_tm1 = smoother_covs[:-1, :, :, s]    # (T-1, K-1, K-1)
            # Sum cross-cov over previous states: sum_i joint(i,s) * C(i,s)
            # pair_cross_covs shape: (T-1, K-1, K-1, S_prev, S_curr)
            C_t_weighted = jnp.einsum(
                "ti,tabi->tab", joint[:, :, s], pair_cross_covs[:, :, :, :, s]
            )  # (T-1, K-1, K-1)
            # Normalize to get expected cross-cov
            C_t = C_t_weighted / jnp.maximum(w[:, None, None], eps)

            cov_trace = (
                jnp.trace(P_t, axis1=1, axis2=2)
                - 2 * decay_s * jnp.trace(C_t, axis1=1, axis2=2)
                + decay_s**2 * jnp.trace(P_tm1, axis1=1, axis2=2)
            )  # (T-1,)

            mean_sq = jnp.sum(mean_resid**2, axis=1)  # (T-1,)
            q_hat = jnp.sum(w * (mean_sq + cov_trace)) / (w_sum * k_free)
            self.process_noises_ = self.process_noises_.at[s].set(
                jnp.maximum(q_hat, 1e-6)
            )

        # Transition matrix from smoother joint
        trans_counts = joint.sum(axis=0)  # (S, S)
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        self.discrete_transition_matrix_ = trans_counts / jnp.maximum(row_sums, eps)

    # --- SGDFittableMixin protocol ---

    def fit_sgd(
        self,
        choices: ArrayLike,
        covariates: Optional[ArrayLike] = None,
        obs_covariates: Optional[ArrayLike] = None,
        optimizer=None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol=None,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent."""
        choices = jnp.asarray(choices, dtype=jnp.int32)
        self._n_trials = int(choices.shape[0])
        self._covariates = jnp.asarray(covariates) if covariates is not None else None
        self._obs_covariates = jnp.asarray(obs_covariates) if obs_covariates is not None else None

        return super().fit_sgd(
            choices,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    @property
    def _n_timesteps(self) -> int:
        return self._n_trials

    def _check_sgd_initialized(self) -> None:
        pass

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            POSITIVE,
            STOCHASTIC_ROW,
            UNCONSTRAINED,
            UNIT_INTERVAL,
            positive_capped,
        )

        params = {
            "process_noises": self.process_noises_,
            "inverse_temperatures": self.inverse_temperatures_,
            "decays": self.decays_,
            "discrete_transition_matrix": self.discrete_transition_matrix_,
            "init_mean": self.init_mean_,
        }
        spec = {
            "process_noises": POSITIVE,
            # Cap inverse_temperature to prevent NaN in the Laplace-EKF
            # update: beta^2 * (diag(p) - outer(p,p)) becomes ill-conditioned
            # when beta > ~50.
            "inverse_temperatures": positive_capped(50.0),
            "decays": UNIT_INTERVAL,
            "discrete_transition_matrix": STOCHASTIC_ROW,
            "init_mean": UNCONSTRAINED,
        }
        if self.input_gain_ is not None:
            params["input_gain"] = self.input_gain_
            spec["input_gain"] = UNCONSTRAINED
        if self.obs_weights_ is not None:
            params["obs_weights"] = self.obs_weights_
            spec["obs_weights"] = UNCONSTRAINED
        return params, spec

    def _sgd_loss_fn(self, params: dict, choices: Array) -> Array:
        kwargs = dict(
            choices=choices,
            n_options=self.n_options,
            n_discrete_states=self.n_discrete_states,
            process_noises=params["process_noises"],
            inverse_temperatures=params["inverse_temperatures"],
            decays=params["decays"],
            discrete_transition_matrix=params["discrete_transition_matrix"],
            init_mean=params["init_mean"],
            init_cov=self.init_cov_,
        )
        if self._covariates is not None and "input_gain" in params:
            kwargs["covariates"] = self._covariates
            kwargs["input_gain"] = params["input_gain"]
        if self._obs_covariates is not None and "obs_weights" in params:
            kwargs["obs_covariates"] = self._obs_covariates
            kwargs["obs_weights"] = params["obs_weights"]

        result = switching_choice_filter(**kwargs)
        return -result.marginal_log_likelihood

    def _store_sgd_params(self, params: dict) -> None:
        self.process_noises_ = params["process_noises"]
        self.inverse_temperatures_ = params["inverse_temperatures"]
        self.decays_ = params["decays"]
        self.discrete_transition_matrix_ = params["discrete_transition_matrix"]
        self.init_mean_ = params["init_mean"]
        if "input_gain" in params:
            self.input_gain_ = params["input_gain"]
        if "obs_weights" in params:
            self.obs_weights_ = params["obs_weights"]

    def _finalize_sgd(self, choices: Array) -> None:
        result = self._run_filter(choices, self._covariates, self._obs_covariates)
        self._filter_result = result
        smoother_result = self._run_smoother(result)
        self.smoothed_discrete_probs_ = smoother_result[2]
        self._smoother_state_cond_means = smoother_result[5]  # (T, K-1, S)
        self._smoother_state_cond_covs = smoother_result[6]  # (T, K-1, K-1, S)
        self.log_likelihood_ = float(result.marginal_log_likelihood)
        self._populate_uncertainty(choices)


class SimulatedSwitchingChoiceData(NamedTuple):
    """Simulated switching choice data."""

    choices: Array        # (n_trials,)
    true_values: Array    # (n_trials, K-1)
    true_states: Array    # (n_trials,)
    true_probs: Array     # (n_trials, K)


def simulate_switching_choice_data(
    n_trials: int = 200,
    n_options: int = 3,
    n_discrete_states: int = 2,
    process_noises: Optional[ArrayLike] = None,
    inverse_temperatures: Optional[ArrayLike] = None,
    decays: Optional[ArrayLike] = None,
    transition_matrix: Optional[ArrayLike] = None,
    seed: int = 42,
) -> SimulatedSwitchingChoiceData:
    """Simulate switching multi-armed bandit choice data.

    Parameters
    ----------
    n_trials : int
    n_options : int
    n_discrete_states : int
    process_noises : ArrayLike or None, shape (S,)
    inverse_temperatures : ArrayLike or None, shape (S,)
    decays : ArrayLike or None, shape (S,)
    transition_matrix : ArrayLike or None, shape (S, S)
    seed : int

    Returns
    -------
    SimulatedSwitchingChoiceData
    """
    S = n_discrete_states
    k_free = n_options - 1
    key = jax.random.PRNGKey(seed)

    if process_noises is None:
        if S > 2:
            raise ValueError(
                f"Default process_noises only defined for S<=2, got S={S}. "
                "Provide explicit process_noises."
            )
        process_noises = jnp.array([0.001, 0.05][:S])
    else:
        process_noises = jnp.asarray(process_noises)
    if inverse_temperatures is None:
        if S > 2:
            raise ValueError(
                f"Default inverse_temperatures only defined for S<=2, got S={S}. "
                "Provide explicit inverse_temperatures."
            )
        inverse_temperatures = jnp.array([5.0, 0.5][:S])
    else:
        inverse_temperatures = jnp.asarray(inverse_temperatures)
    if decays is None:
        decays = jnp.ones(S)
    else:
        decays = jnp.asarray(decays)
    if transition_matrix is None:
        transition_matrix = 0.95 * jnp.eye(S) + 0.05 / S * jnp.ones((S, S))
    else:
        transition_matrix = jnp.asarray(transition_matrix)

    # Simulate via lax.scan for efficiency
    k1, k2, k3 = jax.random.split(key, 3)
    state_keys = jax.random.split(k1, n_trials)
    value_keys = jax.random.split(k2, n_trials)
    choice_keys = jax.random.split(k3, n_trials)

    # States via scan
    def _state_step(prev_state, key_t):
        next_state = jax.random.choice(key_t, S, p=transition_matrix[prev_state])
        next_state = jnp.int32(next_state)
        return next_state, next_state

    _, states = jax.lax.scan(_state_step, jnp.int32(0), state_keys[1:])
    states = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), states])

    # Values via scan
    def _value_step(prev_val, inputs):
        key_t, state_t = inputs
        noise = jax.random.normal(key_t, (k_free,)) * jnp.sqrt(process_noises[state_t])
        new_val = decays[state_t] * prev_val + noise
        return new_val, new_val

    _, values_rest = jax.lax.scan(
        _value_step, jnp.zeros(k_free), (value_keys[1:], states[1:])
    )
    values = jnp.concatenate([jnp.zeros((1, k_free)), values_rest], axis=0)

    # Choices via vmap (independent across trials)
    def _sample_choice(key_t, val_t, state_t):
        v = jnp.concatenate([jnp.zeros(1), val_t])
        probs = jax.nn.softmax(inverse_temperatures[state_t] * v)
        c = jax.random.choice(key_t, n_options, p=probs)
        return c, probs

    choices, all_probs = jax.vmap(_sample_choice)(choice_keys, values, states)

    return SimulatedSwitchingChoiceData(
        choices=choices,
        true_values=values,
        true_states=states,
        true_probs=all_probs,
    )
