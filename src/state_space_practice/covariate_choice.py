"""Covariate-driven choice model via Laplace-EKF.

Extends the multinomial choice model with input-driven value dynamics:
    x_t = x_{t-1} + B @ u_t + w_t,   w_t ~ N(0, q I)
    c_t ~ Categorical(softmax(beta * [0, x_t]))

where B is a learned input-gain matrix mapping trial covariates to value
updates. When covariates are absent, reduces to MultinomialChoiceModel.

Covariate indexing convention
-----------------------------
``covariates[t]`` drives the prediction at trial t, i.e. the transition
from x_{t-1} to x_t. In an RL context, the reward earned on trial t-1
should appear as ``covariates[t]`` so that it drives the value update
going into trial t. ``covariates[0]`` is typically zero (no prior reward).

References
----------
[1] Rescorla, R.A. & Wagner, A.R. (1972). A theory of Pavlovian conditioning.
[2] Piray, P. & Daw, N.D. (2021). A simple model for learning in volatile
    environments. PLoS Computational Biology 17(4), e1007963.
[3] Smith, A.C. et al. (2004). Dynamic analysis of learning in behavioral
    experiments. J Neuroscience 24(2), 447-461.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.multinomial_choice import (
    ChoiceFilterResult,
    ChoiceSmootherResult,
    MultinomialChoiceModel,
    _softmax_update_core,
)
from state_space_practice.parameter_transforms import (
    UNCONSTRAINED,
    UNIT_INTERVAL,
)
from state_space_practice.utils import psd_solve, symmetrize, validate_choice_indices

logger = logging.getLogger(__name__)


def covariate_predict(
    filt_mean: Array,
    filt_cov: Array,
    covariates_t: Array,
    input_gain: Array,
    transition_matrix: Array,
    process_noise_cov: Array,
) -> tuple[Array, Array]:
    """Prediction step with transition matrix and control input.

    Parameters
    ----------
    filt_mean : Array, shape (K-1,)
        Filtered state mean from previous trial.
    filt_cov : Array, shape (K-1, K-1)
        Filtered state covariance from previous trial.
    covariates_t : Array, shape (d,)
        Covariate vector for current trial.
    input_gain : Array, shape (K-1, d)
        Input-gain matrix B.
    transition_matrix : Array, shape (K-1, K-1)
        State transition matrix A. Identity = random walk.
        decay * I = mean-reverting (Ornstein-Uhlenbeck).
    process_noise_cov : Array, shape (K-1, K-1)
        Process noise covariance Q.

    Returns
    -------
    pred_mean : Array, shape (K-1,)
    pred_cov : Array, shape (K-1, K-1)
    """
    pred_mean = transition_matrix @ filt_mean + input_gain @ covariates_t
    pred_cov = transition_matrix @ filt_cov @ transition_matrix.T + process_noise_cov
    return pred_mean, pred_cov


@jax.jit
def m_step_input_gain(
    smoothed_values: Array,
    covariates: Array,
    decay: float = 1.0,
) -> Array:
    """Closed-form M-step for input-gain matrix B.

    B_hat = [sum_t r_t u_t'] @ [sum_t u_t u_t']^{-1}
    where r_t = m_{t|T} - decay * m_{t-1|T} is the dynamics residual.

    Parameters
    ----------
    smoothed_values : Array, shape (T, K-1)
        Smoothed state means from RTS smoother.
    covariates : Array, shape (T, d)
        Covariate matrix. Row t drives the prediction at trial t
        (transition x_{t-1} -> x_t). Uses covariates[1:] paired with
        the residual at each transition.
    decay : float, default 1.0
        AR decay coefficient. When decay != 1, the dynamics are
        x_t = decay * x_{t-1} + B u_t + w_t, so the target for B
        is m_t - decay * m_{t-1}, not m_t - m_{t-1}.

    Returns
    -------
    B_hat : Array, shape (K-1, d)

    Notes
    -----
    Since covariates are observed (not random), the cross-covariance
    terms from the smoother drop out and this simple regression is
    the exact EM M-step for B.
    """
    diff = smoothed_values[1:] - decay * smoothed_values[:-1]  # (T-1, K-1)
    u = covariates[1:]  # (T-1, d) — covariates[i+1] drives diff[i]

    # Cross term: sum delta_m_t u_t'
    cross = jnp.einsum("ti,tj->ij", diff, u)  # (K-1, d)
    # Covariate gram matrix: sum u_t u_t'
    gram = jnp.einsum("ti,tj->ij", u, u)  # (d, d)

    return psd_solve(gram.T, cross.T).T


@partial(jax.jit, static_argnames=("n_options", "max_newton_steps"))
def m_step_obs_weights(
    smoothed_values: Array,
    choices: Array,
    obs_covariates: Array,
    n_options: int,
    inverse_temperature: float,
    current_obs_weights: Array,
    max_newton_steps: int = 5,
) -> Array:
    """Newton M-step for observation weights Theta.

    Maximizes sum_t log softmax(beta * [0, m_t] + Theta @ z_t)[c_t]
    w.r.t. Theta, holding smoothed values m_t fixed. This is a
    standard multinomial logistic regression (concave in Theta).

    Parameters
    ----------
    smoothed_values : Array, shape (T, K-1)
    choices : Array, shape (T,) int
    obs_covariates : Array, shape (T, d_obs)
    n_options : int
        Number of options K (static under ``jax.jit``).
    inverse_temperature : float
    current_obs_weights : Array, shape (K, d_obs)
        Current Theta estimate (warm start).
    max_newton_steps : int
        Number of damped Newton steps (static under ``jax.jit``; the loop
        is unrolled at trace time).

    Returns
    -------
    Theta_hat : Array, shape (K, d_obs)
    """
    T = smoothed_values.shape[0]
    d_obs = obs_covariates.shape[1]
    K = n_options
    beta = inverse_temperature

    # Build full value vectors: [0, m_t] for each trial
    zeros = jnp.zeros((T, 1))
    full_values = jnp.concatenate([zeros, smoothed_values], axis=1)  # (T, K)

    # One-hot choices
    e_choices = jax.nn.one_hot(choices, K)  # (T, K)

    # Newton iteration on Theta (flattened to K*d_obs vector)
    theta = current_obs_weights.ravel()  # (K * d_obs,)

    for _ in range(max_newton_steps):
        Theta = theta.reshape(K, d_obs)
        # Logits: beta * v_t + Theta @ z_t
        offsets = obs_covariates @ Theta.T  # (T, K)
        logits = beta * full_values + offsets
        probs = jax.nn.softmax(logits, axis=1)  # (T, K)

        # Gradient: sum_t (e_t - p_t) ⊗ z_t → (K, d_obs)
        residuals = e_choices - probs  # (T, K)
        grad = jnp.einsum("tk,td->kd", residuals, obs_covariates)  # (K, d_obs)
        grad_flat = grad.ravel()

        # Block-diagonal Fisher approximation for Hessian: one (d_obs, d_obs)
        # block per option, weighted by p_k (1 - p_k).
        hess_blocks = jnp.einsum(
            "tk,td,te->kde", probs * (1 - probs), obs_covariates, obs_covariates
        )
        hess = jax.scipy.linalg.block_diag(*hess_blocks)

        # Damped Newton step (0.5 step size for stability)
        step = psd_solve(hess, grad_flat)
        theta = theta + 0.5 * step

    return theta.reshape(K, d_obs)


def covariate_choice_filter(
    choices: ArrayLike,
    n_options: int,
    covariates: ArrayLike | None = None,
    input_gain: ArrayLike | None = None,
    obs_covariates: ArrayLike | None = None,
    obs_weights: ArrayLike | None = None,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    decay: float = 1.0,
    init_mean: ArrayLike | None = None,
    init_cov: ArrayLike | None = None,
) -> ChoiceFilterResult:
    """Forward filter for covariate-driven choice model.

    Parameters
    ----------
    choices : ArrayLike, shape (n_trials,)
        Observed choices (0-indexed integers in [0, K)).
    n_options : int
        Total number of options K.
    covariates : ArrayLike or None, shape (n_trials, d_dyn)
        Dynamics covariates driving value updates. None = random walk.
    input_gain : ArrayLike or None, shape (K-1, d_dyn)
        Input-gain matrix B for dynamics covariates.
    obs_covariates : ArrayLike or None, shape (n_trials, d_obs)
        Observation covariates that bias choice probabilities without
        changing the latent value state (e.g., stay/switch, spatial bias).
    obs_weights : ArrayLike or None, shape (K, d_obs)
        Weights mapping observation covariates to logit offsets.
        Full K-dim (including reference option).
    process_noise : float
        Scalar process noise (Q = process_noise * I).
    inverse_temperature : float
        Softmax inverse temperature beta.
    decay : float
        Value decay rate. 1.0 = random walk (no decay).
        < 1.0 = mean-reverting toward zero (Ornstein-Uhlenbeck).
        Transition matrix is A = decay * I.
    init_mean : ArrayLike or None
        Initial state mean, shape (K-1,). Default: zeros.
    init_cov : ArrayLike or None
        Initial covariance, shape (K-1, K-1). Default: identity.

    Returns
    -------
    ChoiceFilterResult
    """
    validate_choice_indices(choices, n_options)
    choices_arr = jnp.asarray(choices, dtype=jnp.int32)
    k_free = n_options - 1

    if init_mean is None:
        init_mean = jnp.zeros(k_free)
    else:
        init_mean = jnp.asarray(init_mean)
    if init_cov is None:
        init_cov = jnp.eye(k_free)
    else:
        init_cov = jnp.asarray(init_cov)

    # Dynamics covariates
    if covariates is not None:
        covariates_arr = jnp.asarray(covariates)
        if covariates_arr.shape[0] != choices_arr.shape[0]:
            raise ValueError(
                f"covariates has {covariates_arr.shape[0]} rows but choices "
                f"has {choices_arr.shape[0]} trials"
            )
        if input_gain is None:
            input_gain = jnp.zeros((k_free, covariates_arr.shape[1]))
        input_gain_arr = jnp.asarray(input_gain)
    else:
        covariates_arr = jnp.zeros((choices_arr.shape[0], 1))
        input_gain_arr = jnp.zeros((k_free, 1))

    # Observation covariates
    if obs_covariates is not None:
        obs_cov_arr = jnp.asarray(obs_covariates)
        if obs_cov_arr.shape[0] != choices_arr.shape[0]:
            raise ValueError(
                f"obs_covariates has {obs_cov_arr.shape[0]} rows but choices "
                f"has {choices_arr.shape[0]} trials"
            )
        if obs_weights is None:
            obs_weights = jnp.zeros((n_options, obs_cov_arr.shape[1]))
        obs_weights_arr = jnp.asarray(obs_weights)
    else:
        obs_cov_arr = jnp.zeros((choices_arr.shape[0], 1))
        obs_weights_arr = jnp.zeros((n_options, 1))

    return _covariate_choice_filter_jit(
        choices_arr, n_options, covariates_arr, input_gain_arr,
        obs_cov_arr, obs_weights_arr,
        process_noise, inverse_temperature, decay, init_mean, init_cov,
    )


@partial(jax.jit, static_argnames=("n_options",))
def _covariate_choice_filter_jit(
    choices: Array,
    n_options: int,
    covariates: Array,
    input_gain: Array,
    obs_covariates: Array,
    obs_weights: Array,
    process_noise: float,
    inverse_temperature: float,
    decay: float,
    init_mean: Array,
    init_cov: Array,
) -> ChoiceFilterResult:
    """JIT-compiled filter core with decay, dynamics, and obs covariates."""
    k_free = n_options - 1
    Q = jnp.eye(k_free) * process_noise
    A = jnp.eye(k_free) * decay

    def _step(carry, inputs):
        filt_mean, filt_cov, total_ll = carry
        choice_t, u_t, z_t = inputs

        # Predict with transition and control input
        pred_mean = A @ filt_mean + input_gain @ u_t
        pred_cov = A @ filt_cov @ A.T + Q

        # Observation offset from obs covariates
        obs_offset = obs_weights @ z_t

        # Update via Laplace-EKF with obs offset
        post_mean, post_cov, ll = _softmax_update_core(
            pred_mean, pred_cov, choice_t,
            n_options, inverse_temperature,
            obs_offset=obs_offset,
        )

        total_ll = total_ll + ll
        return (post_mean, post_cov, total_ll), (
            post_mean, post_cov, pred_mean, pred_cov,
        )

    init_carry = (init_mean, init_cov, jnp.array(0.0))
    (_, _, marginal_ll), (filt_vals, filt_covs, pred_vals, pred_covs) = (
        jax.lax.scan(_step, init_carry, (choices, covariates, obs_covariates))
    )

    return ChoiceFilterResult(
        filtered_values=filt_vals,
        filtered_covariances=filt_covs,
        predicted_values=pred_vals,
        predicted_covariances=pred_covs,
        marginal_log_likelihood=marginal_ll,
    )


def _rts_smoother_pass_with_predictions(
    filtered_values: Array,
    filtered_covariances: Array,
    predicted_values: Array,
    predicted_covariances: Array,
    A: Array,
) -> tuple[Array, Array, Array]:
    """RTS backward smoother that consumes the filter's stored one-step predictions.

    Unlike :func:`kalman.rts_backward_scan`, which recomputes the
    one-step prediction as ``A @ m_filt`` (valid only for control-free dynamics),
    this uses ``predicted_values`` / ``predicted_covariances`` from the forward
    filter. Those already include the control input ``input_gain @ u_t``, so the
    smoothed means stay consistent with the filter when dynamics covariates are
    present. The gain/covariance recursion is the standard RTS update (the control
    input does not affect covariances), so with no control input this reduces
    exactly to ``rts_backward_scan``.
    """

    def _smooth_step(carry, inputs):
        next_sm_mean, next_sm_cov = carry
        f_mean, f_cov, p_mean_next, p_cov_next = inputs
        gain = psd_solve(p_cov_next, A @ f_cov).T
        sm_mean = f_mean + gain @ (next_sm_mean - p_mean_next)
        sm_cov = symmetrize(f_cov + gain @ (next_sm_cov - p_cov_next) @ gain.T)
        cross_cov = gain @ next_sm_cov
        return (sm_mean, sm_cov), (sm_mean, sm_cov, cross_cov)

    _, (sm_means, sm_covs, cross_covs) = jax.lax.scan(
        _smooth_step,
        (filtered_values[-1], filtered_covariances[-1]),
        (
            filtered_values[:-1],
            filtered_covariances[:-1],
            predicted_values[1:],
            predicted_covariances[1:],
        ),
        reverse=True,
    )
    return sm_means, sm_covs, cross_covs


def covariate_choice_smoother(
    choices: ArrayLike,
    n_options: int,
    covariates: ArrayLike | None = None,
    input_gain: ArrayLike | None = None,
    obs_covariates: ArrayLike | None = None,
    obs_weights: ArrayLike | None = None,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    decay: float = 1.0,
    init_mean: ArrayLike | None = None,
    init_cov: ArrayLike | None = None,
) -> ChoiceSmootherResult:
    """Forward filter + RTS backward smoother for covariate-driven choice model.

    Parameters are the same as :func:`covariate_choice_filter`.

    Returns
    -------
    ChoiceSmootherResult
    """
    filt = covariate_choice_filter(
        choices, n_options, covariates, input_gain,
        obs_covariates, obs_weights,
        process_noise, inverse_temperature, decay, init_mean, init_cov,
    )

    k_free = n_options - 1
    A = jnp.eye(k_free) * decay

    # Control-aware RTS: consume the filter's stored one-step predictions (which
    # include the control input input_gain @ u_t) instead of recomputing
    # A @ m_filt, so the smoothed means are consistent with the filter when
    # dynamics covariates are present.
    sm_means, sm_covs, cross_covs = _rts_smoother_pass_with_predictions(
        filt.filtered_values,
        filt.filtered_covariances,
        filt.predicted_values,
        filt.predicted_covariances,
        A,
    )

    smoothed_values = jnp.concatenate([sm_means, filt.filtered_values[-1:]])
    smoothed_covs = jnp.concatenate([sm_covs, filt.filtered_covariances[-1:]])

    return ChoiceSmootherResult(
        smoothed_values=smoothed_values,
        smoothed_covariances=smoothed_covs,
        smoother_cross_cov=cross_covs,
        marginal_log_likelihood=filt.marginal_log_likelihood,
    )


def _coerce_covariates(
    values: ArrayLike | None,
    n_expected: int,
    name: str,
    size_name: str,
    method: str,
) -> Array | None:
    """Validate a covariate matrix passed to ``fit`` / ``fit_sgd``.

    Returns None when the model declares no such covariates
    (``n_expected == 0``); an array passed in that case is ignored.
    """
    if n_expected == 0:
        return None
    if values is None:
        raise ValueError(
            f"Model has {size_name}={n_expected} but no {name} were passed to "
            f"{method}()"
        )
    arr = jnp.asarray(values)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D (n_trials, {size_name}) array, got shape "
            f"{arr.shape}"
        )
    if arr.shape[1] != n_expected:
        raise ValueError(
            f"{name} has {arr.shape[1]} columns but model expects "
            f"{size_name}={n_expected}"
        )
    return arr


class CovariateChoiceModel(MultinomialChoiceModel):
    """Multi-armed bandit with covariate-driven value dynamics and
    observation-level choice biases.

    Extends MultinomialChoiceModel with two types of covariates:

    1. **Dynamics covariates** (input-gain B): drive value updates
       ``x_t = x_{t-1} + B @ u_t + noise``
    2. **Observation covariates** (obs weights Theta): bias choice
       probabilities without changing the latent value state
       ``p(c_t) = softmax(beta * [0, x_t] + Theta @ z_t)``

    When n_covariates=0 and n_obs_covariates=0, reduces to
    MultinomialChoiceModel (pure random walk).

    The EM driver, uncertainty summaries, BIC and plots are inherited; this
    class overrides the hooks listed in :class:`MultinomialChoiceModel`
    (filter selection, dynamics, logit offsets, M-steps, final E-step).

    Parameters
    ----------
    n_options : int
        Number of choice options K.
    n_covariates : int
        Number of dynamics covariates d_dyn.
    n_obs_covariates : int
        Number of observation covariates d_obs.
    init_inverse_temperature : float
        Starting inverse temperature for EM.
    init_process_noise : float
        Starting process noise for EM.
    init_decay : float
        Starting value decay rate. 1.0 = random walk (no decay).
        < 1.0 = mean-reverting toward zero.
    learn_inverse_temperature : bool
        Whether to learn beta via EM.
    learn_process_noise : bool
        Whether to learn Q via EM.
    learn_decay : bool
        Whether to learn the decay parameter via EM.
    learn_obs_weights : bool
        Whether to learn Theta via EM.
    """

    def __init__(
        self,
        n_options: int,
        n_covariates: int = 0,
        n_obs_covariates: int = 0,
        init_inverse_temperature: float = 1.0,
        init_process_noise: float = 0.01,
        init_decay: float = 1.0,
        learn_inverse_temperature: bool = True,
        learn_process_noise: bool = True,
        learn_decay: bool = False,
        learn_obs_weights: bool = True,
    ):
        super().__init__(
            n_options,
            init_inverse_temperature=init_inverse_temperature,
            init_process_noise=init_process_noise,
            learn_inverse_temperature=learn_inverse_temperature,
            learn_process_noise=learn_process_noise,
        )
        if init_decay <= 0 or init_decay > 1:
            raise ValueError(
                "init_decay must lie in (0, 1] for stable latent dynamics, "
                f"got {init_decay}."
            )
        self.n_covariates = n_covariates
        self.n_obs_covariates = n_obs_covariates
        self.decay = init_decay
        self.learn_decay = learn_decay
        self.learn_obs_weights = learn_obs_weights

        k_free = n_options - 1
        self.input_gain_: Array = jnp.zeros((k_free, max(n_covariates, 1)))
        self.obs_weights_: Array = jnp.zeros((n_options, max(n_obs_covariates, 1)))

        # Covariates bound by fit() / fit_sgd()
        self._covariates: Array | None = None
        self._obs_covariates: Array | None = None

    def __repr__(self) -> str:
        fitted = self.is_fitted
        return (
            f"CovariateChoiceModel(n_options={self.n_options}, "
            f"n_covariates={self.n_covariates}, "
            f"beta={self.inverse_temperature:.3f}, "
            f"Q={self.process_noise:.4f}, "
            f"decay={self.decay:.4f}, fitted={fitted})"
        )

    # --- Hooks (see MultinomialChoiceModel) ---

    def _filter_kwargs(self) -> dict:
        kwargs = super()._filter_kwargs()
        kwargs.update(
            covariates=self._covariates,
            input_gain=self.input_gain_ if self.n_covariates > 0 else None,
            obs_covariates=self._obs_covariates,
            obs_weights=self.obs_weights_ if self.n_obs_covariates > 0 else None,
            decay=self.decay,
        )
        return kwargs

    def _run_filter(self, choices: Array, **overrides) -> ChoiceFilterResult:
        kwargs = {**self._filter_kwargs(), **overrides}
        return covariate_choice_filter(choices, self.n_options, **kwargs)

    def _run_smoother(self, choices: Array) -> ChoiceSmootherResult:
        return covariate_choice_smoother(
            choices, self.n_options, **self._filter_kwargs()
        )

    def _transition_decay(self) -> float:
        return self.decay

    def _control_input(self) -> Array | None:
        """``B u_t`` for each transition t-1 -> t, shape (T-1, K-1), or None."""
        if self.n_covariates > 0 and self._covariates is not None:
            return self._covariates[1:] @ self.input_gain_.T
        return None

    def _observation_logit_offsets(self) -> Array | None:
        """``Theta @ z_t`` per trial, shape (T, K); None without obs covariates."""
        if self._obs_covariates is None:
            return None
        return self._obs_covariates @ self.obs_weights_.T

    def _em_parameter_names(self) -> tuple[str, ...]:
        return ("input_gain_", "obs_weights_", "decay") + super()._em_parameter_names()

    def _em_progress(self) -> str:
        return super()._em_progress() + f", decay={self.decay:.4f}"

    def _m_step(
        self, smooth: ChoiceSmootherResult, choices: Array, beta_grid: Array
    ) -> None:
        # M-step for B (input gain)
        if self.n_covariates > 0 and self._covariates is not None:
            self.input_gain_ = m_step_input_gain(
                smooth.smoothed_values, self._covariates,
                decay=self.decay,
            )

        # M-step for Theta (observation weights)
        if (self.learn_obs_weights
                and self.n_obs_covariates > 0
                and self._obs_covariates is not None):
            self.obs_weights_ = m_step_obs_weights(
                smooth.smoothed_values, choices,
                self._obs_covariates, self.n_options,
                self.inverse_temperature, self.obs_weights_,
            )

        # M-step for decay
        if self.learn_decay:
            self.decay = self._m_step_decay(smooth)

        # M-steps for process noise Q and inverse temperature beta
        super()._m_step(smooth, choices, beta_grid)

    def _final_e_step(
        self,
        choices: Array,
        log_likelihoods: list[float],
        last_accepted: dict | None,
    ) -> float:
        """Final E-step, kept only if it did not decrease the log-likelihood.

        The final E-step syncs the smoother to the last M-step's parameters. It
        is accepted (its LL replacing or extending the history) when that LL is
        finite and not worse than the last accepted iterate; otherwise the last
        M-step is rolled back so the stored (params, smoother, LL) stay
        consistent.
        """
        final_ll = super()._final_e_step(choices, log_likelihoods, last_accepted)
        close = bool(log_likelihoods) and np.isclose(final_ll, log_likelihoods[-1])
        not_worse = (not log_likelihoods) or close or final_ll >= log_likelihoods[-1]
        if np.isfinite(final_ll) and not_worse:
            if close:
                log_likelihoods[-1] = final_ll
            else:
                log_likelihoods.append(final_ll)
        elif last_accepted is not None:
            self._restore_parameters(last_accepted)
            final_ll = super()._final_e_step(
                choices, log_likelihoods, last_accepted
            )
            logger.warning(
                "Final M-step decreased the log-likelihood; rolled back to the "
                "previous parameters."
            )
        return final_ll

    def _bind_covariates(
        self,
        choices: ArrayLike,
        covariates: ArrayLike | None,
        obs_covariates: ArrayLike | None,
        method: str,
    ) -> Array:
        """Validate the fit inputs, then store the covariates used by the fit.

        Everything (covariates, then choices through the shared
        ``_prepare_choices``) is validated before anything is assigned, so a
        call rejected by validation leaves a previous fit's covariates and
        trial count untouched. Returns the validated int32 choices.
        """
        covariates_arr = _coerce_covariates(
            covariates, self.n_covariates, "covariates", "n_covariates", method
        )
        obs_covariates_arr = _coerce_covariates(
            obs_covariates, self.n_obs_covariates, "obs_covariates",
            "n_obs_covariates", method,
        )
        choices_arr = self._prepare_choices(
            choices, "EM" if method == "fit" else "SGD"
        )
        self._covariates = covariates_arr
        self._obs_covariates = obs_covariates_arr
        return choices_arr

    def fit(
        self,
        choices: ArrayLike,
        covariates: ArrayLike | None = None,
        obs_covariates: ArrayLike | None = None,
        max_iter: int = 50,
        tolerance: float = 1e-4,
        verbose: bool = False,
        beta_grid: ArrayLike | None = None,
    ) -> list[float]:
        """Fit the model via EM algorithm.

        Parameters
        ----------
        choices : ArrayLike, shape (n_trials,)
            Observed choices (0-indexed integers in [0, K)).
        covariates : ArrayLike or None, shape (n_trials, d_dyn)
            Dynamics covariates driving value updates. None = random walk.
        obs_covariates : ArrayLike or None, shape (n_trials, d_obs)
            Observation covariates biasing choice probabilities
            (e.g., stay/switch indicator, spatial bias). None = no bias.
        max_iter : int
            Maximum EM iterations.
        tolerance : float
            Convergence tolerance on the relative log-likelihood change
            ``|LL_k - LL_{k-1}| / |LL_{k-1}| < tolerance`` (inherited from
            :meth:`MultinomialChoiceModel.fit`; normalized by the previous LL,
            not by the two-iterate average used by
            :func:`state_space_practice.utils.check_converged`).
        verbose : bool
            Print progress each iteration.
        beta_grid : ArrayLike or None
            Candidate inverse temperatures for grid search.

        Returns
        -------
        log_likelihoods : list of float
        """
        choices_arr = self._bind_covariates(
            choices, covariates, obs_covariates, "fit"
        )
        return self._fit_em(choices_arr, max_iter, tolerance, verbose, beta_grid)

    def fit_sgd(
        self,
        choices: ArrayLike,
        covariates: ArrayLike | None = None,
        obs_covariates: ArrayLike | None = None,
        optimizer: object | None = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: float | None = None,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        choices : ArrayLike, shape (n_trials,)
            Observed choices (0-indexed integers in [0, K)).
        covariates : ArrayLike or None, shape (n_trials, d_dyn)
            Dynamics covariates.
        obs_covariates : ArrayLike or None, shape (n_trials, d_obs)
            Observation covariates.
        optimizer : optax optimizer or None
            Gradient optimizer. Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps.
        convergence_tol : float or None
            If set, stop early when loss change < tol for 5 consecutive steps.

        Returns
        -------
        log_likelihoods : list of float
        """
        choices_arr = self._bind_covariates(
            choices, covariates, obs_covariates, "fit_sgd"
        )
        return self._fit_sgd_validated(
            choices_arr, optimizer, num_steps, verbose, convergence_tol
        )

    # --- SGDFittableMixin protocol ---

    def _build_param_spec(self) -> tuple[dict, dict]:
        params, spec = super()._build_param_spec()
        if self.learn_decay:
            params["decay"] = jnp.array(self.decay)
            spec["decay"] = UNIT_INTERVAL
        if self.n_covariates > 0:
            params["input_gain"] = self.input_gain_
            spec["input_gain"] = UNCONSTRAINED
        if self.n_obs_covariates > 0 and self.learn_obs_weights:
            params["obs_weights"] = self.obs_weights_
            spec["obs_weights"] = UNCONSTRAINED
        return params, spec

    def _sgd_loss_fn(self, params: dict, choices: Array) -> Array:
        # Model attributes are read only for parameters that are not being
        # optimized (see MultinomialChoiceModel._sgd_loss_fn).
        def _param(key: str, attr: str) -> Array:
            return params[key] if key in params else jnp.asarray(getattr(self, attr))

        k_free = self.n_options - 1

        if self._covariates is not None:
            cov_arr = self._covariates
            ig_arr = _param("input_gain", "input_gain_")
        else:
            cov_arr = jnp.zeros((self._n_trials, 1))
            ig_arr = jnp.zeros((k_free, 1))

        if self._obs_covariates is not None:
            obs_cov_arr = self._obs_covariates
            ow_arr = _param("obs_weights", "obs_weights_")
        else:
            obs_cov_arr = jnp.zeros((self._n_trials, 1))
            ow_arr = jnp.zeros((self.n_options, 1))

        result = _covariate_choice_filter_jit(
            choices, self.n_options,
            cov_arr, ig_arr,
            obs_cov_arr, ow_arr,
            _param("process_noise", "process_noise"),
            _param("inverse_temperature", "inverse_temperature"),
            _param("decay", "decay"),
            jnp.zeros(k_free),
            jnp.eye(k_free),
        )
        return -result.marginal_log_likelihood

    def _store_sgd_params(self, params: dict) -> None:
        super()._store_sgd_params(params)
        if "decay" in params:
            self.decay = float(params["decay"])
        if "input_gain" in params:
            self.input_gain_ = params["input_gain"]
        if "obs_weights" in params:
            self.obs_weights_ = params["obs_weights"]

    # --- M-steps specific to this model ---

    def _m_step_decay(self, smooth: ChoiceSmootherResult) -> float:
        """M-step: update scalar decay from smoother statistics.

        For x_t = a * x_{t-1} + B u_t + w_t with isotropic Q, maximizing the
        expected complete-data log-likelihood over the scalar a gives

            a = sum_t [(m_t - B u_t)' m_{t-1} + tr C_{t-1,t}]
                / sum_t [m_{t-1}' m_{t-1} + tr P_{t-1}],

        where ``E[x_t' x_{t-1}] = m_t' m_{t-1} + tr C_{t-1,t}`` and
        C_{t-1,t} = Cov(x_{t-1}, x_t | y_{1:T}) is the smoother's lag-one
        cross-covariance (``smoother_cross_cov[t-1]``; the trace is the same
        for either orientation). Dropping the ``tr C`` term biases the
        estimate towards zero.
        """
        m = smooth.smoothed_values  # (T, K-1)
        P = smooth.smoothed_covariances  # (T, K-1, K-1)
        C = smooth.smoother_cross_cov  # (T-1, K-1, K-1)

        target = m[1:]  # (T-1, K-1)
        control_input = self._control_input()
        if control_input is not None:
            target = target - control_input

        # Numerator: sum_t E[(x_t - B u_t)' x_{t-1}]
        numer = jnp.sum(target * m[:-1]) + jnp.sum(
            jnp.trace(C, axis1=1, axis2=2)
        )
        # Denominator: sum_t E[x_{t-1}' x_{t-1}]
        denom = jnp.sum(m[:-1] ** 2) + jnp.sum(
            jnp.trace(P[:-1], axis1=1, axis2=2)
        )
        return float(jnp.clip(numer / jnp.maximum(denom, 1e-10), 0.01, 1.0))

    @property
    def n_free_params(self) -> int:
        """Number of free parameters learned by EM."""
        n = super().n_free_params
        if self.n_covariates > 0:
            n += (self.n_options - 1) * self.n_covariates  # B matrix
        if self.n_obs_covariates > 0 and self.learn_obs_weights:
            # (K-1), not K: adding a constant to a full Theta column shifts every
            # option's logit equally (softmax-invariant), so one row per obs
            # covariate is non-identifiable and must not be counted.
            n += (self.n_options - 1) * self.n_obs_covariates  # Theta matrix
        if self.learn_decay:
            n += 1  # scalar decay
        return n

    def _summary_dimension_rows(self) -> list[tuple[str, object]]:
        rows = super()._summary_dimension_rows()
        return rows + [("n_covariates", self.n_covariates)]

    def summary(self) -> str:
        """Text summary of fitted model, including the input-gain matrix B."""
        lines = [super().summary()]
        if self.n_covariates > 0:
            lines.append("  input_gain (B):")
            B = np.array(self.input_gain_)
            for i in range(B.shape[0]):
                lines.append(
                    f"    option {i + 1}: "
                    + ", ".join(f"{B[i, j]:.4f}" for j in range(B.shape[1]))
                )
        return "\n".join(lines)

    def plot_input_gains(self, option_labels=None, covariate_labels=None, ax=None):
        """Bar plot of the learned input-gain matrix B.

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt

        self._check_fitted("plot_input_gains")

        B = np.array(self.input_gain_)
        k_free, d = B.shape

        if option_labels is None:
            option_labels = [f"Option {i + 1}" for i in range(k_free)]
        if covariate_labels is None:
            covariate_labels = [f"Cov {j}" for j in range(d)]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure

        x = np.arange(d)
        width = 0.8 / k_free
        for i in range(k_free):
            ax.bar(x + i * width, B[i], width, label=option_labels[i])

        ax.set_xticks(x + width * (k_free - 1) / 2)
        ax.set_xticklabels(covariate_labels)
        ax.set_ylabel("Input gain (B)")
        ax.set_title("Learned Input Gains")
        ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

        fig.tight_layout()
        return fig, ax

    def plot_summary(self, observed_choices=None, option_labels=None):
        """3-panel diagnostic: values, input gains, convergence.

        Returns
        -------
        fig, axes : array of 3 Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        self._check_fitted("plot_summary")

        option_labels = self._resolve_option_labels(option_labels)

        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(1, 3, figure=fig)

        # Panel 1: latent values
        ax0 = fig.add_subplot(gs[0, 0])
        self._plot_smoothed_values(ax0, option_labels, "Smoothed Values", 7)

        # Panel 2: input gains
        ax1 = fig.add_subplot(gs[0, 1])
        if self.n_covariates > 0:
            self.plot_input_gains(ax=ax1)
        else:
            ax1.text(0.5, 0.5, "No covariates", ha="center", va="center",
                     transform=ax1.transAxes)
            ax1.set_title("Input Gains")

        # Panel 3: convergence
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_convergence(ax=ax2)

        fig.tight_layout()
        return fig, np.array([ax0, ax1, ax2])


class SimulatedRLChoiceData(NamedTuple):
    """Simulated RL choice data with covariates.

    Attributes
    ----------
    choices : Array, shape (n_trials,)
    true_values : Array, shape (n_trials, K-1)
    true_probs : Array, shape (n_trials, K)
    covariates : Array, shape (n_trials, d)
    """
    choices: Array
    true_values: Array
    true_probs: Array
    covariates: Array


def simulate_rl_choice_data(
    n_trials: int = 200,
    n_options: int = 3,
    input_gain: ArrayLike | None = None,
    process_noise: float = 0.005,
    inverse_temperature: float = 2.0,
    reward_prob: float = 0.7,
    decay: float = 0.95,
    seed: int = 42,
) -> SimulatedRLChoiceData:
    """Simulate multi-armed bandit data with covariate-driven value evolution.

    Generates choices from a softmax model where option values evolve
    according to ``x_t = decay * x_{t-1} + B @ u_t + noise``. The ``decay < 1``
    forgetting factor keeps values bounded; without it a repeatedly-rewarded
    option's value would accumulate without bound (a near-deterministic runaway
    rather than a realistic bandit). Covariates follow
    the filter convention: ``u_t`` drives the prediction at trial t
    (i.e., the transition x_{t-1} -> x_t). For reward covariates, this
    means the reward earned on trial t-1 appears as ``u_t``.

    Parameters
    ----------
    n_trials : int
    n_options : int
    input_gain : ArrayLike, shape (K-1, K-1)
        Input-gain matrix B. Diagonal entries act as per-option learning rates.
    process_noise : float
        Residual process noise variance.
    inverse_temperature : float
        Softmax inverse temperature.
    reward_prob : float
        Probability of reward when an option is chosen.
    seed : int

    Returns
    -------
    SimulatedRLChoiceData
    """
    rng = np.random.default_rng(seed)
    k_free = n_options - 1

    if input_gain is None:
        input_gain = np.eye(k_free) * 0.5
    B = np.asarray(input_gain)
    d = B.shape[1]

    values = np.zeros((n_trials, k_free))
    choices = np.zeros(n_trials, dtype=int)
    probs = np.zeros((n_trials, n_options))
    covariates = np.zeros((n_trials, d))

    for t in range(n_trials):
        # Apply covariate-driven update: covariates[t] drives x[t-1] -> x[t]
        # (covariates[0] is zero — no prior reward before first trial)
        if t > 0:
            values[t] = (
                decay * values[t - 1]
                + B @ covariates[t]
                + rng.normal(0, np.sqrt(process_noise), k_free)
            )

        # Choice from softmax
        full_vals = np.concatenate([[0.0], values[t]])
        logits = inverse_temperature * full_vals
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        probs[t] = p
        choices[t] = rng.choice(n_options, p=p)

        # Generate reward covariate for *next* trial
        # Reward earned on trial t becomes covariates[t+1]
        if choices[t] > 0 and t < n_trials - 1:
            covariates[t + 1, choices[t] - 1] = float(
                rng.random() < reward_prob
            )

    return SimulatedRLChoiceData(
        choices=jnp.array(choices),
        true_values=jnp.array(values),
        true_probs=jnp.array(probs),
        covariates=jnp.array(covariates),
    )
