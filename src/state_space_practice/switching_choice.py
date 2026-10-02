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

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.covariate_choice import covariate_predict
from state_space_practice.exceptions import NotFittedError
from state_space_practice.fitted_state import FittedAttribute, is_set
from state_space_practice.multinomial_choice import (
    _softmax_update_core,
    _warn_if_newton_unconverged,
)
from state_space_practice.parameter_transforms import (
    POSITIVE,
    STOCHASTIC_ROW,
    UNCONSTRAINED,
    UNIT_INTERVAL,
    positive_capped,
)
from state_space_practice.sgd_fitting import SGDFittableMixin, SGDParams, SGDParamSpec
from state_space_practice.switching_kalman import (
    SwitchingSmootherResult,
    _first_timestep_discrete_update,
    _normalize_initial_discrete_prob,
    _stabilize_probability_vector_preserving_zeros,
    _update_discrete_state_probabilities,
    collapse_gaussian_mixture,
    collapse_gaussian_mixture_per_discrete_state,
    switching_kalman_smoother,
)
from state_space_practice.utils import (
    typed_jit,
    validate_choice_indices,
)

if TYPE_CHECKING:
    import optax

logger = logging.getLogger(__name__)


def _softmax_predict_and_update(
    prev_mean: Array,
    prev_cov: Array,
    choice: Array,
    transition_matrix: Array,
    process_cov: Array,
    n_options: int,
    inverse_temperature: float | Array,
    input_gain: Array,
    covariates_t: Array,
    obs_offset: Array,
) -> tuple[Array, Array, Array, Array]:
    """Predict + softmax update for one (prev_state_i, next_state_j) pair.

    Parameters
    ----------
    prev_mean : Array, shape (K-1,)
    prev_cov : Array, shape (K-1, K-1)
    choice : Array, scalar int
    transition_matrix : Array, shape (K-1, K-1) — A_j = decay_j * I
    process_cov : Array, shape (K-1, K-1) — Q_j
    n_options : int
    inverse_temperature : float or Array, shape () — beta_j
    input_gain : Array, shape (K-1, d) — shared B
    covariates_t : Array, shape (d,)
    obs_offset : Array, shape (K,) — Theta @ z_t

    Returns
    -------
    post_mean : Array, shape (K-1,)
    post_cov : Array, shape (K-1, K-1)
    log_likelihood : Array, shape ()
    newton_gap : Array, shape ()
        Laplace mode-search convergence diagnostic (see
        ``multinomial_choice._softmax_update_core``).
    """
    # Predict
    pred_mean, pred_cov = covariate_predict(
        prev_mean, prev_cov, covariates_t, input_gain, transition_matrix, process_cov
    )

    # Update via softmax Laplace-EKF
    return _softmax_update_core(
        pred_mean,
        pred_cov,
        choice,
        n_options,
        inverse_temperature,
        obs_offset=obs_offset,
    )


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
) -> tuple[Array, Array, Array, Array]:
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
    pair_newton_gap : Array, shape (S_prev, S_next)
        Laplace mode-search convergence diagnostic per pair.
    """

    def _update_one_pair(
        prev_mean_i: Array, prev_cov_i: Array, A_j: Array, Q_j: Array, beta_j: Array
    ) -> tuple[Array, Array, Array, Array]:
        """Update for one (i, j) pair."""
        return _softmax_predict_and_update(
            prev_mean_i,
            prev_cov_i,
            choice,
            A_j,
            Q_j,
            n_options,
            beta_j,
            input_gain,
            covariates_t,
            obs_offset,
        )

    # vmap over prev_state i (axis -1 of mean/cov)
    def _update_all_prev_for_next_j(
        A_j: Array, Q_j: Array, beta_j: Array
    ) -> tuple[Array, Array, Array, Array]:
        # vmap over prev state i
        return jax.vmap(
            lambda m, c: _update_one_pair(m, c, A_j, Q_j, beta_j),
            in_axes=(1, 2),  # mean: axis 1, cov: axis 2
            # mean: (K-1, S_prev), cov: (K-1, K-1, S_prev), ll and gap: (S_prev,)
            out_axes=(1, 2, 0, 0),
        )(prev_state_cond_mean, prev_state_cond_cov)

    # vmap over next_state j (axis -1 of A/Q, element of beta)
    pair_mean, pair_cov, pair_ll, pair_gap = jax.vmap(
        _update_all_prev_for_next_j,
        in_axes=(2, 2, 0),  # A: axis 2, Q: axis 2, beta: axis 0
        # mean: (K-1, S_prev, S_next), cov: (K-1, K-1, S_prev, S_next),
        # ll and gap: (S_prev, S_next)
        out_axes=(2, 3, 1, 1),
    )(transition_matrices, process_covs, inverse_temperatures)

    return pair_mean, pair_cov, pair_ll, pair_gap


class SwitchingChoiceFilterResult(NamedTuple):
    """Result container for switching choice filter."""

    filtered_values: Array  # (n_trials, K-1, S) state-conditional posterior
    filtered_covs: Array  # (n_trials, K-1, K-1, S) state-conditional posterior
    predicted_values: Array  # (n_trials, K-1, S) state-conditional predicted (prior)
    predicted_covs: Array  # (n_trials, K-1, K-1, S) state-conditional predicted
    discrete_state_probs: Array  # (n_trials, S) filtered posterior
    marginal_log_likelihood: Array  # scalar
    pair_cond_means: Array  # (n_trials, K-1, S, S) for smoother
    pair_cond_covs: Array  # (n_trials, K-1, K-1, S, S) for smoother


def switching_choice_filter(
    choices: ArrayLike,
    n_options: int,
    n_discrete_states: int = 2,
    covariates: ArrayLike | None = None,
    input_gain: ArrayLike | None = None,
    obs_covariates: ArrayLike | None = None,
    obs_weights: ArrayLike | None = None,
    process_noises: ArrayLike | None = None,
    inverse_temperatures: ArrayLike | None = None,
    decays: ArrayLike | None = None,
    discrete_transition_matrix: ArrayLike | None = None,
    init_mean: ArrayLike | None = None,
    init_cov: ArrayLike | None = None,
    init_discrete_prob: ArrayLike | None = None,
) -> SwitchingChoiceFilterResult:
    """Validate choices and run the JIT-compiled switching choice filter.

    See :func:`_switching_choice_filter_jit` for parameters and returns.

    Warns
    -----
    StateSpaceWarning
        If a Laplace mode search (for any state pair) ends more than
        ``multinomial_choice.NEWTON_GAP_TOL`` nats (Newton estimate) below
        its mode.
    """
    validate_choice_indices(choices, n_options)
    return _switching_choice_filter_jit(
        choices,
        n_options,
        n_discrete_states,
        covariates,
        input_gain,
        obs_covariates,
        obs_weights,
        process_noises,
        inverse_temperatures,
        decays,
        discrete_transition_matrix,
        init_mean,
        init_cov,
        init_discrete_prob,
    )


@functools.partial(
    typed_jit,
    static_argnames=["n_options", "n_discrete_states"],
)
def _switching_choice_filter_jit(
    choices: ArrayLike,
    n_options: int,
    n_discrete_states: int = 2,
    covariates: ArrayLike | None = None,
    input_gain: ArrayLike | None = None,
    obs_covariates: ArrayLike | None = None,
    obs_weights: ArrayLike | None = None,
    process_noises: ArrayLike | None = None,
    inverse_temperatures: ArrayLike | None = None,
    decays: ArrayLike | None = None,
    discrete_transition_matrix: ArrayLike | None = None,
    init_mean: ArrayLike | None = None,
    init_cov: ArrayLike | None = None,
    init_discrete_prob: ArrayLike | None = None,
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
        discrete_transition_matrix = 0.9 * jnp.eye(S) + 0.1 / S * jnp.ones((S, S))
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

    # Covariates. Require covariates and input_gain both-or-neither (matching the
    # obs branch below); silently zeroing the covariates when input_gain is
    # omitted would drop the dynamics input without any signal.
    if covariates is not None:
        if input_gain is None:
            raise ValueError("covariates provided but input_gain is None")
        cov_arr = jnp.asarray(covariates)
        ig_arr = jnp.asarray(input_gain)
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
    transition_matrices = (
        decays[None, None, :] * jnp.eye(k_free)[:, :, None]
    )  # (K-1, K-1, S)
    process_covs = (
        process_noises[None, None, :] * jnp.eye(k_free)[:, :, None]
    )  # (K-1, K-1, S)

    # Expand init_mean/cov to per-state (shared)
    init_state_cond_mean = jnp.stack([init_mean] * S, axis=-1)  # (K-1, S)
    init_state_cond_cov = jnp.stack([init_cov] * S, axis=-1)  # (K-1, K-1, S)

    # Sanitize the caller-supplied prior exactly as the switching-Kalman and
    # switching point-process filters do: exact zeros are structural (an
    # impossible state stays impossible), NaN / negative entries are clamped to
    # 0, and the structural support S_1 is read from the sanitized prior.
    init_discrete_prob = _normalize_initial_discrete_prob(init_discrete_prob)
    first_support = init_discrete_prob > 0.0

    # --- First timestep: predict + update (x₀ convention) ---
    # This uses the x₀ convention (predict+update at t=0) to match the
    # non-switching CovariateChoiceModel, NOT the x₁ convention used by
    # switching_kalman_filter and switching_point_process_filter (which
    # skip prediction at t=0). The x₀ convention means init_mean is
    # the prior BEFORE the first observation, not AT it. The smoother
    # still works because it only uses filter outputs, not the convention.
    def _first_update_for_state(
        prior_mean: Array, prior_cov: Array, beta: Array, A_j: Array, Q_j: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        pred_mean, pred_cov = covariate_predict(
            prior_mean, prior_cov, cov_arr[0], ig_arr, A_j, Q_j
        )

        obs_offset_0 = ow_arr @ obs_cov_arr[0]
        post_mean, post_cov, ll, newton_gap = _softmax_update_core(
            pred_mean,
            pred_cov,
            choices[0],
            n_options,
            beta,
            obs_offset=obs_offset_0,
        )
        return post_mean, post_cov, ll, pred_mean, pred_cov, newton_gap

    (
        first_means,
        first_covs,
        first_lls,
        first_pred_means,
        first_pred_covs,
        first_newton_gaps,
    ) = jax.vmap(
        _first_update_for_state,
        in_axes=(1, 2, 0, 2, 2),
        out_axes=(1, 2, 0, 1, 2, 0),
    )(
        init_state_cond_mean,
        init_state_cond_cov,
        inverse_temperatures,
        transition_matrices,
        process_covs,
    )
    # first_means: (K-1, S), first_pred_means: (K-1, S), etc.

    # Log-space, support-masked discrete update for the first timestep, shared
    # with the other switching filters: a structural zero stays exactly 0, a
    # tiny prior is represented faithfully, and an all-zero prior fails loud.
    # There is no transition at t=0, so the support carried into the scan is S_1.
    first_discrete_prob, first_marginal_ll = _first_timestep_discrete_update(
        first_lls, init_discrete_prob
    )
    first_next_support = first_support

    # Pair-conditional for smoother: diagonal (no pair structure at t=0)
    first_pair_mean = jnp.stack([first_means] * S, axis=-1)  # (K-1, S, S)
    first_pair_cov = jnp.stack([first_covs] * S, axis=-1)  # (K-1, K-1, S, S)

    # --- Scan over t=1..T-1 ---
    def _step(
        carry: tuple[Array, Array, Array, Array, Array],
        trial_data: tuple[Array, Array, Array],
    ) -> tuple[
        tuple[Array, Array, Array, Array, Array],
        tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    ]:
        prev_mean, prev_cov, prev_disc_prob, accum_ll, prev_support = carry
        choice_t, u_t, z_t = trial_data

        obs_offset_t = ow_arr @ z_t  # (K,)

        # Compute per-state predicted (prior) means and covs
        def _predict_for_state_j(
            prev_m: Array, prev_c: Array, A_j: Array, Q_j: Array
        ) -> tuple[Array, Array]:
            return covariate_predict(prev_m, prev_c, u_t, ig_arr, A_j, Q_j)

        pred_means, pred_covs = jax.vmap(
            _predict_for_state_j,
            in_axes=(1, 2, 2, 2),
            out_axes=(1, 2),
        )(prev_mean, prev_cov, transition_matrices, process_covs)

        # Per-state-pair predict + update
        pair_mean, pair_cov, pair_ll, pair_gap = _softmax_update_per_state_pair(
            prev_mean,
            prev_cov,
            choice_t,
            transition_matrices,
            process_covs,
            n_options,
            inverse_temperatures,
            ig_arr,
            u_t,
            obs_offset_t,
        )

        # Log-space, support-masked discrete update (see switching_kalman).
        disc_prob, backward_prob, log_predictive, next_support = (
            _update_discrete_state_probabilities(
                pair_ll, discrete_transition_matrix, prev_disc_prob, prev_support
            )
        )

        # Accumulate LL
        new_ll = accum_ll + log_predictive

        # Collapse mixtures
        state_mean, state_cov = collapse_gaussian_mixture_per_discrete_state(
            pair_mean,
            pair_cov,
            backward_prob,
        )

        return (state_mean, state_cov, disc_prob, new_ll, next_support), (
            state_mean,
            state_cov,
            disc_prob,
            pair_mean,
            pair_cov,
            pred_means,
            pred_covs,
            pair_gap,
        )

    init_carry = (
        first_means,
        first_covs,
        first_discrete_prob,
        first_marginal_ll,
        first_next_support,
    )
    scan_inputs = (choices[1:], cov_arr[1:], obs_cov_arr[1:])

    (
        (_, _, _, total_ll, _),
        (
            rest_means,
            rest_covs,
            rest_disc_probs,
            rest_pair_means,
            rest_pair_covs,
            rest_pred_means,
            rest_pred_covs,
            rest_newton_gaps,
        ),
    ) = jax.lax.scan(_step, init_carry, scan_inputs)
    _warn_if_newton_unconverged(
        jnp.concatenate([first_newton_gaps, rest_newton_gaps.ravel()]),
        "switching_choice_filter",
    )

    # Concatenate first timestep
    filtered_values = jnp.concatenate([first_means[None], rest_means], axis=0)
    filtered_covs = jnp.concatenate([first_covs[None], rest_covs], axis=0)
    discrete_state_probs = jnp.concatenate(
        [first_discrete_prob[None], rest_disc_probs], axis=0
    )
    pair_cond_means = jnp.concatenate([first_pair_mean[None], rest_pair_means], axis=0)
    pair_cond_covs = jnp.concatenate([first_pair_cov[None], rest_pair_covs], axis=0)
    predicted_values = jnp.concatenate(
        [first_pred_means[None], rest_pred_means], axis=0
    )
    predicted_covs = jnp.concatenate([first_pred_covs[None], rest_pred_covs], axis=0)

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


@typed_jit
def switching_choice_smoother(
    filtered_values: ArrayLike,
    filtered_covs: ArrayLike,
    discrete_state_probs: ArrayLike,
    process_covs: ArrayLike,
    transition_matrices: ArrayLike,
    discrete_transition_matrix: ArrayLike,
    control_input: ArrayLike,
) -> SwitchingSmootherResult:
    """GPB1 switching RTS smoother for dynamics with a known control input.

    The latent dynamics of the switching choice model are
    ``x_t = A_{s_t} x_{t-1} + b_t + w_t`` with the known input
    ``b_t = B @ u_t``. :func:`switching_kalman.switching_kalman_smoother`
    assumes ``b_t = 0``: its backward step predicts ``A_k m_{t|t}`` and so,
    with covariates, compares the smoothed ``x_{t+1}`` against a prediction
    that is missing ``b_{t+1}``, which biases every smoothed mean.

    For each state pair the correct RTS mean update is
    ``m_t + J (m^s_{t+1} - A_k m_t - b_{t+1})``, which is the control-free
    update applied to the shifted next-step smoothed mean
    ``m^s_{t+1} - b_{t+1}``. The discrete-state recursion and every
    covariance (including the mixture-collapse spread terms, which are
    invariant to a common shift of the ``t+1`` means) do not depend on
    ``b``. This function therefore runs the library backward step one time
    step at a time, feeding it the shifted next-step means, and returns the
    same result as ``switching_kalman_smoother``. With ``control_input == 0``
    it reproduces ``switching_kalman_smoother`` exactly.

    Parameters
    ----------
    filtered_values : ArrayLike, shape (T, K-1, S)
    filtered_covs : ArrayLike, shape (T, K-1, K-1, S)
    discrete_state_probs : ArrayLike, shape (T, S)
    process_covs : ArrayLike, shape (K-1, K-1, S)
    transition_matrices : ArrayLike, shape (K-1, K-1, S)
    discrete_transition_matrix : ArrayLike, shape (S, S)
    control_input : ArrayLike, shape (T, K-1)
        ``control_input[t]`` is the known input ``b_t`` of the transition
        ``x_{t-1} -> x_t`` (row 0 is never read).

    Returns
    -------
    SwitchingSmootherResult
        With the shapes documented in
        :func:`switching_kalman.switching_kalman_smoother`.
    """
    filtered_values = jnp.asarray(filtered_values)
    filtered_covs = jnp.asarray(filtered_covs)
    discrete_state_probs = jnp.asarray(discrete_state_probs)
    control_input = jnp.asarray(control_input)

    def _backward_step(
        carry: tuple[Array, Array, Array],
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[Array, Array, Array], SwitchingSmootherResult]:
        next_means, next_covs, next_probs = carry
        filt_mean_t, filt_cov_t, filt_prob_t, b_next = inputs
        # A two-step call whose "last filter" slot is the smoothed t+1 state:
        # the library initialises its backward carry from that slot, so its
        # single backward step is exactly the GPB1 step at t.
        out = switching_kalman_smoother(
            filter_mean=jnp.stack([filt_mean_t, next_means - b_next[:, None]]),
            filter_cov=jnp.stack([filt_cov_t, next_covs]),
            filter_discrete_state_prob=jnp.stack([filt_prob_t, next_probs]),
            process_cov=process_covs,
            continuous_transition_matrix=transition_matrices,
            discrete_state_transition_matrix=discrete_transition_matrix,
        )
        step = SwitchingSmootherResult(*(o[0] for o in out))
        new_carry = (
            step.state_cond_smoother_means,
            step.state_cond_smoother_covs,
            _stabilize_probability_vector_preserving_zeros(
                step.smoother_discrete_state_prob
            ),
        )
        return new_carry, step

    init_carry = (
        filtered_values[-1],
        filtered_covs[-1],
        discrete_state_probs[-1],
    )
    _, steps = jax.lax.scan(
        _backward_step,
        init_carry,
        (
            filtered_values[:-1],
            filtered_covs[:-1],
            discrete_state_probs[:-1],
            control_input[1:],
        ),
        reverse=True,
    )
    last_mean, last_cov = collapse_gaussian_mixture(
        filtered_values[-1], filtered_covs[-1], discrete_state_probs[-1]
    )
    return steps._replace(
        overall_smoother_mean=jnp.concatenate(
            [steps.overall_smoother_mean, last_mean[None]], axis=0
        ),
        overall_smoother_cov=jnp.concatenate(
            [steps.overall_smoother_cov, last_cov[None]], axis=0
        ),
        smoother_discrete_state_prob=jnp.concatenate(
            [steps.smoother_discrete_state_prob, discrete_state_probs[-1:]], axis=0
        ),
        state_cond_smoother_means=jnp.concatenate(
            [steps.state_cond_smoother_means, filtered_values[-1:]], axis=0
        ),
        state_cond_smoother_covs=jnp.concatenate(
            [steps.state_cond_smoother_covs, filtered_covs[-1:]], axis=0
        ),
    )


def _between_state_variance(means: Array, probs: Array) -> Array:
    """Between-state term ``Var_s(E[x | s])`` of the law of total variance.

    Uses the centred form ``sum_s p_s (m_s - m_bar)^2`` rather than
    ``E[m^2] - E[m]^2``, which cancels catastrophically when the per-state
    means are large relative to their spread (the same reason
    ``collapse_gaussian_mixture`` centres its spread-of-means term).

    Parameters
    ----------
    means : Array, shape (n_time, n_options, n_discrete_states)
        Per-state conditional means.
    probs : Array, shape (n_time, n_discrete_states)
        Discrete-state mixing weights (each row sums to 1).

    Returns
    -------
    Array, shape (n_time, n_options)
        Non-negative variance of the per-state means under ``probs``.
    """
    mixture_mean = jnp.einsum("tks,ts->tk", means, probs)
    centred = means - mixture_mean[..., None]
    return jnp.einsum("tks,ts->tk", centred**2, probs)


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

    # Fitted state, set by fit() / fit_sgd(); reading one before raises
    # NotFittedError.
    _filter_result: FittedAttribute[SwitchingChoiceFilterResult] = FittedAttribute()
    # Smoothed discrete-state probabilities (T, S) and state-conditional
    # smoother means (T, K-1, S) and covariances (T, K-1, K-1, S).
    smoothed_discrete_probs_: FittedAttribute[Array] = FittedAttribute()
    _smoother_state_cond_means: FittedAttribute[Array] = FittedAttribute()
    _smoother_state_cond_covs: FittedAttribute[Array] = FittedAttribute()
    log_likelihood_: FittedAttribute[float] = FittedAttribute()
    _n_trials: FittedAttribute[int] = FittedAttribute()

    # Uncertainty summaries: (T, K) variances, (T,) entropy and surprise,
    # (T, K, S) per-state predicted variances.
    predicted_option_variances_: FittedAttribute[Array] = FittedAttribute()
    smoothed_option_variances_: FittedAttribute[Array] = FittedAttribute()
    predicted_choice_entropy_: FittedAttribute[Array] = FittedAttribute()
    surprise_: FittedAttribute[Array] = FittedAttribute()
    per_state_predicted_variances_: FittedAttribute[Array] = FittedAttribute()

    def __init__(
        self,
        n_options: int,
        n_discrete_states: int = 2,
        n_covariates: int = 0,
        n_obs_covariates: int = 0,
        init_inverse_temperatures: ArrayLike | None = None,
        init_process_noises: ArrayLike | None = None,
        init_decays: ArrayLike | None = None,
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

        # Per-state parameters have hard validity constraints that EM/SGD assume
        # but do not enforce for the fixed (non-learned) values; a negative
        # inverse temperature silently inverts choice preferences.
        if bool(jnp.any(self.inverse_temperatures_ <= 0)):
            raise ValueError(
                "init_inverse_temperatures must be strictly positive (a "
                "non-positive inverse temperature makes the softmax degenerate "
                "or inverts choice preferences); got min "
                f"{float(jnp.min(self.inverse_temperatures_))}."
            )
        if bool(jnp.any(self.process_noises_ < 0)):
            raise ValueError(
                "init_process_noises must be non-negative (Q_j = q_j * I must be "
                f"PSD); got min {float(jnp.min(self.process_noises_))}."
            )
        if bool(jnp.any((self.decays_ <= 0) | (self.decays_ > 1))):
            raise ValueError(
                "init_decays must lie in (0, 1] for stable latent dynamics; got "
                f"[{float(jnp.min(self.decays_))}, {float(jnp.max(self.decays_))}]."
            )

        # Shared parameters
        self.init_mean_ = jnp.zeros(k_free)
        self.init_cov_ = jnp.eye(k_free)
        self.discrete_transition_matrix_ = 0.9 * jnp.eye(S) + 0.1 / S * jnp.ones((S, S))
        self.input_gain_: Array | None
        if n_covariates > 0:
            self.input_gain_ = jnp.zeros((k_free, n_covariates))
        else:
            self.input_gain_ = None
        self.obs_weights_: Array | None
        if n_obs_covariates > 0:
            self.obs_weights_ = jnp.zeros((n_options, n_obs_covariates))
        else:
            self.obs_weights_ = None

        # Covariates bound by fit() / fit_sgd()
        self._covariates: Array | None = None
        self._obs_covariates: Array | None = None

    @property
    def is_fitted(self) -> bool:
        return is_set(self, "_filter_result")

    def _populate_uncertainty(self, choices: Array) -> None:
        """Compute uncertainty summaries from filter result."""
        from state_space_practice.behavioral_uncertainty import (
            append_reference_option,
            categorical_entropy,
            compute_surprise,
        )

        if not is_set(self, "_filter_result"):
            return

        result = self._filter_result
        disc_probs = result.discrete_state_probs  # (T, S) — filtered posterior

        # Predicted (prior) discrete state probs: P(s_t | y_{1:t-1}).
        # Reconstruct from lagged filtered posterior + transition matrix.
        init_prob = jnp.ones(self.n_discrete_states) / self.n_discrete_states
        predicted_disc = jnp.concatenate(
            [
                init_prob[None, :],
                (disc_probs[:-1] @ self.discrete_transition_matrix_),
            ],
            axis=0,
        )  # (T, S)

        per_state_values = result.predicted_values  # (T, K-1, S) — prior, not posterior

        # Per-state predicted variances (diagonal of covariance).
        # predicted_covs has shape (T, K-1, K-1, S); we need (T, K-1, S).
        # jnp.diagonal appends the diagonal as the last axis, so calling it
        # with axis1=1, axis2=2 yields (T, S, K-1), not (T, K-1, S).
        per_state_vars = jnp.einsum("tiis->tis", result.predicted_covs)  # (T, K-1, S)

        # Zero for reference option, then full K
        zero_ref = jnp.zeros((per_state_vars.shape[0], 1, per_state_vars.shape[2]))
        full_vars = jnp.concatenate([zero_ref, per_state_vars], axis=1)  # (T, K, S)
        self.per_state_predicted_variances_ = full_vars  # (T, K, S)

        # Per-state predicted means (full K with reference option)
        full_means = jnp.concatenate(
            [
                jnp.zeros((per_state_values.shape[0], 1, per_state_values.shape[2])),
                per_state_values,
            ],
            axis=1,
        )  # (T, K, S)

        # Law of total variance: Var(x) = E[Var(x|s)] + Var(E[x|s])
        # Use PREDICTED (prior) state probs for weighting.
        e_var = jnp.einsum("tks,ts->tk", full_vars, predicted_disc)  # E[Var(x|s)]
        var_mean = _between_state_variance(full_means, predicted_disc)
        self.predicted_option_variances_ = e_var + var_mean

        # Smoothed variances: law of total variance with smoother quantities
        if is_set(self, "_smoother_state_cond_covs"):
            # See note above on diagonal axis ordering; use einsum to get (T, K-1, S).
            smoother_diag = jnp.einsum(
                "tiis->tis", self._smoother_state_cond_covs
            )  # (T, K-1, S)
            zero_ref_sm = jnp.zeros((smoother_diag.shape[0], 1, smoother_diag.shape[2]))
            full_sm_vars = jnp.concatenate([zero_ref_sm, smoother_diag], axis=1)

            # Smoother state-conditional means
            if is_set(self, "_smoother_state_cond_means"):
                smoother_means = self._smoother_state_cond_means
                full_sm_means = jnp.concatenate(
                    [
                        jnp.zeros(
                            (smoother_means.shape[0], 1, smoother_means.shape[2])
                        ),
                        smoother_means,
                    ],
                    axis=1,
                )  # (T, K, S)
                sm_disc = self.smoothed_discrete_probs_
                sm_e_var = jnp.einsum("tks,ts->tk", full_sm_vars, sm_disc)
                sm_var_mean = _between_state_variance(full_sm_means, sm_disc)
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

        per_state_prob_list = []
        for s in range(self.n_discrete_states):
            v = append_reference_option(per_state_values[:, :, s])
            p = jax.nn.softmax(self.inverse_temperatures_[s] * v + obs_offsets, axis=1)
            per_state_prob_list.append(p)
        per_state_probs = jnp.stack(per_state_prob_list, axis=-1)  # (T, K, S)
        predicted_probs = jnp.einsum("tks,ts->tk", per_state_probs, predicted_disc)

        self.predicted_choice_entropy_ = categorical_entropy(predicted_probs)
        self.surprise_ = compute_surprise(predicted_probs, choices)

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted else "not fitted"
        return (
            f"SwitchingChoiceModel(n_options={self.n_options}, "
            f"n_discrete_states={self.n_discrete_states}, {fitted})"
        )

    def _run_filter(
        self,
        choices: Array,
        covariates: Array | None = None,
        obs_covariates: Array | None = None,
    ) -> SwitchingChoiceFilterResult:
        """Run the switching choice filter with current parameters."""
        kwargs: dict[str, Any] = {
            "choices": choices,
            "n_options": self.n_options,
            "n_discrete_states": self.n_discrete_states,
            "process_noises": self.process_noises_,
            "inverse_temperatures": self.inverse_temperatures_,
            "decays": self.decays_,
            "discrete_transition_matrix": self.discrete_transition_matrix_,
            "init_mean": self.init_mean_,
            "init_cov": self.init_cov_,
        }
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
        covariates: ArrayLike | None = None,
        obs_covariates: ArrayLike | None = None,
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
            Maximum EM iterations.
        tolerance : float
            Convergence threshold on the *absolute* log-likelihood change
            between successive E-steps, ``|LL_k - LL_{k-1}| < tolerance``
            (nats; not the relative criterion of
            :func:`state_space_practice.utils.check_converged`), so it should
            be scaled with the number of trials.

        Returns
        -------
        log_likelihoods : list of float
        """
        validate_choice_indices(choices, self.n_options)
        choices = jnp.asarray(choices, dtype=jnp.int32)
        self._n_trials = int(choices.shape[0])
        self._covariates = jnp.asarray(covariates) if covariates is not None else None
        self._obs_covariates = (
            jnp.asarray(obs_covariates) if obs_covariates is not None else None
        )

        log_likelihoods: list[float] = []
        prev_ll = float("-inf")
        converged = False

        for iteration in range(max_iter):
            # E-step: filter + smoother
            result = self._run_filter(choices, self._covariates, self._obs_covariates)
            self._filter_result = result
            smoother_result = self._run_smoother(result)
            ll = float(result.marginal_log_likelihood)
            log_likelihoods.append(ll)

            if abs(ll - prev_ll) < tolerance and iteration > 0:
                logger.info(f"Converged at iteration {iteration + 1}")
                converged = True
                break
            prev_ll = ll

            # M-step
            self._m_step(smoother_result)

        if not converged:
            # Final E-step with learned parameters so fitted summaries and
            # log_likelihood_ correspond to the model state after the last
            # M-step. On convergence no M-step followed the last E-step, so its
            # results are already at the final parameters.
            result = self._run_filter(choices, self._covariates, self._obs_covariates)
            self._filter_result = result
            smoother_result = self._run_smoother(result)

        self.smoothed_discrete_probs_ = smoother_result.smoother_discrete_state_prob
        self._smoother_state_cond_means = smoother_result.state_cond_smoother_means
        self._smoother_state_cond_covs = smoother_result.state_cond_smoother_covs
        self.log_likelihood_ = float(result.marginal_log_likelihood)
        # History is the per-iteration E-step LL trajectory; without
        # convergence its last entry predates the final M-step, so it
        # intentionally differs from log_likelihood_ above (re-evaluated at the
        # final parameters).
        self.log_likelihood_history_ = log_likelihoods
        self._populate_uncertainty(choices)
        self._finalize_convergence(converged, max_iter)

        return log_likelihoods

    def _run_smoother(
        self, filter_result: SwitchingChoiceFilterResult
    ) -> SwitchingSmootherResult:
        """Run the control-aware GPB1 switching smoother on filter output.

        The known dynamics input ``B @ u_t`` enters the smoother's one-step
        predictions (see :func:`switching_choice_smoother`), so the smoothed
        means stay consistent with the filter when covariates are present.
        """
        k_free = self.n_options - 1
        n_trials = filter_result.filtered_values.shape[0]
        if self.input_gain_ is not None and self._covariates is not None:
            control_input = self._covariates @ self.input_gain_.T  # (T, K-1)
        else:
            control_input = jnp.zeros((n_trials, k_free))
        return switching_choice_smoother(
            filter_result.filtered_values,
            filter_result.filtered_covs,
            filter_result.discrete_state_probs,
            self.process_noises_[None, None, :] * jnp.eye(k_free)[:, :, None],
            self.decays_[None, None, :] * jnp.eye(k_free)[:, :, None],
            self.discrete_transition_matrix_,
            control_input,
        )

    def _m_step(self, smoother_result: SwitchingSmootherResult) -> None:
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
        joint = smoother_result.smoother_joint_discrete_state_prob  # (T-1, S, S)
        smoother_means = smoother_result.state_cond_smoother_means  # (T, K-1, S)
        smoother_covs = smoother_result.state_cond_smoother_covs  # (T, K-1, K-1, S)
        # (T-1, K-1, K-1, S, S)
        pair_cross_covs = smoother_result.pair_cond_smoother_cross_covs
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
                smoother_means[1:, :, s] - decay_s * smoother_means[:-1, :, s] - Bu[1:]
            )

            # Covariance correction: aggregate cross-covs over all (i, s) pairs
            P_t = smoother_covs[1:, :, :, s]  # (T-1, K-1, K-1)
            P_tm1 = smoother_covs[:-1, :, :, s]  # (T-1, K-1, K-1)
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
        covariates: ArrayLike | None = None,
        obs_covariates: ArrayLike | None = None,
        optimizer: optax.GradientTransformation | None = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: float | None = None,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        choices : ArrayLike, shape (n_trials,)
            Observed choices, integers in ``[0, n_options)``.
        covariates : ArrayLike or None, shape (n_trials, n_covariates)
            Dynamics covariates driving the per-state value updates. None =
            no covariate drive.
        obs_covariates : ArrayLike or None, shape (n_trials, n_obs_covariates)
            Observation covariates biasing choice probabilities. None = no
            bias.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps (INFO level).
        convergence_tol : float or None
            Stop early when the relative LL change stays below this for 5
            consecutive steps.

        Returns
        -------
        log_likelihoods : list of float
            Marginal log-likelihood per optimization step.
        """
        validate_choice_indices(choices, self.n_options)
        choices = jnp.asarray(choices, dtype=jnp.int32)
        self._n_trials = int(choices.shape[0])
        self._covariates = jnp.asarray(covariates) if covariates is not None else None
        self._obs_covariates = (
            jnp.asarray(obs_covariates) if obs_covariates is not None else None
        )

        return super().fit_sgd(
            choices,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    @property
    def _n_timesteps(self) -> int:
        if not is_set(self, "_n_trials"):
            raise NotFittedError("Model must be fitted before accessing _n_timesteps.")
        return self._n_trials

    def _build_param_spec(self) -> tuple[SGDParams, SGDParamSpec]:
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

    def _sgd_loss_fn(self, params: SGDParams, choices: Array) -> Array:
        use_covariates = self._covariates is not None and "input_gain" in params
        use_obs_covariates = (
            self._obs_covariates is not None and "obs_weights" in params
        )

        # The jitted core, not the validating public wrapper: fit_sgd already
        # validated the choices, which are a traced jit argument here.
        result = _switching_choice_filter_jit(
            choices=choices,
            n_options=self.n_options,
            n_discrete_states=self.n_discrete_states,
            covariates=self._covariates if use_covariates else None,
            input_gain=params["input_gain"] if use_covariates else None,
            obs_covariates=self._obs_covariates if use_obs_covariates else None,
            obs_weights=params["obs_weights"] if use_obs_covariates else None,
            process_noises=params["process_noises"],
            inverse_temperatures=params["inverse_temperatures"],
            decays=params["decays"],
            discrete_transition_matrix=params["discrete_transition_matrix"],
            init_mean=params["init_mean"],
            init_cov=self.init_cov_,
        )
        return -result.marginal_log_likelihood

    def _store_sgd_params(self, params: SGDParams) -> None:
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
        self.smoothed_discrete_probs_ = smoother_result.smoother_discrete_state_prob
        self._smoother_state_cond_means = smoother_result.state_cond_smoother_means
        self._smoother_state_cond_covs = smoother_result.state_cond_smoother_covs
        self.log_likelihood_ = float(result.marginal_log_likelihood)
        self._populate_uncertainty(choices)


class SimulatedSwitchingChoiceData(NamedTuple):
    """Simulated switching choice data."""

    choices: Array  # (n_trials,)
    true_values: Array  # (n_trials, K-1)
    true_states: Array  # (n_trials,)
    true_probs: Array  # (n_trials, K)


def simulate_switching_choice_data(
    n_trials: int = 200,
    n_options: int = 3,
    n_discrete_states: int = 2,
    process_noises: ArrayLike | None = None,
    inverse_temperatures: ArrayLike | None = None,
    decays: ArrayLike | None = None,
    transition_matrix: ArrayLike | None = None,
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
    def _state_step(prev_state: Array, key_t: Array) -> tuple[Array, Array]:
        next_state = jax.random.choice(key_t, S, p=transition_matrix[prev_state])
        next_state = jnp.int32(next_state)
        return next_state, next_state

    _, states = jax.lax.scan(_state_step, jnp.int32(0), state_keys[1:])
    states = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), states])

    # Values via scan
    def _value_step(
        prev_val: Array, inputs: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        key_t, state_t = inputs
        noise = jax.random.normal(key_t, (k_free,)) * jnp.sqrt(process_noises[state_t])
        new_val = decays[state_t] * prev_val + noise
        return new_val, new_val

    _, values_rest = jax.lax.scan(
        _value_step, jnp.zeros(k_free), (value_keys[1:], states[1:])
    )
    values = jnp.concatenate([jnp.zeros((1, k_free)), values_rest], axis=0)

    # Choices via vmap (independent across trials)
    def _sample_choice(
        key_t: Array, val_t: Array, state_t: Array
    ) -> tuple[Array, Array]:
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
