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
import math
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import psd_solve
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.multinomial_choice import (
    ChoiceFilterResult,
    ChoiceSmootherResult,
    _rts_smoother_pass,
    _softmax_update_core,
)

logger = logging.getLogger(__name__)

_DEFAULT_BETA_GRID = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


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


def m_step_input_gain(
    smoothed_values: Array,
    covariates: Array,
) -> Array:
    """Closed-form M-step for input-gain matrix B.

    B_hat = [sum_t delta_m_t u_t'] @ [sum_t u_t u_t']^{-1}
    where delta_m_t = m_{t|T} - m_{t-1|T}.

    Parameters
    ----------
    smoothed_values : Array, shape (T, K-1)
        Smoothed state means from RTS smoother.
    covariates : Array, shape (T, d)
        Covariate matrix. Row t drives the prediction at trial t
        (transition x_{t-1} -> x_t). Uses covariates[1:] paired with
        diff[i] = m[i+1] - m[i], since diff[i] corresponds to the
        transition driven by covariates[i+1].

    Returns
    -------
    B_hat : Array, shape (K-1, d)

    Notes
    -----
    Since covariates are observed (not random), the cross-covariance
    terms from the smoother drop out and this simple regression is
    the exact EM M-step for B.
    """
    diff = smoothed_values[1:] - smoothed_values[:-1]  # (T-1, K-1)
    u = covariates[1:]  # (T-1, d) — covariates[i+1] drives diff[i]

    # Cross term: sum delta_m_t u_t'
    cross = jnp.einsum("ti,tj->ij", diff, u)  # (K-1, d)
    # Covariate gram matrix: sum u_t u_t'
    gram = jnp.einsum("ti,tj->ij", u, u)  # (d, d)

    B_hat = psd_solve(gram.T, cross.T).T
    return B_hat


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
    inverse_temperature : float
    current_obs_weights : Array, shape (K, d_obs)
        Current Theta estimate (warm start).
    max_newton_steps : int

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

        # Block-diagonal Fisher approximation for Hessian
        hess_diag_blocks = []
        for k in range(K):
            w = probs[:, k] * (1 - probs[:, k])  # (T,)
            Hk = jnp.einsum("t,td,te->de", w, obs_covariates, obs_covariates)
            hess_diag_blocks.append(Hk)
        hess = jax.scipy.linalg.block_diag(*hess_diag_blocks)

        # Damped Newton step (0.5 step size for stability)
        step = psd_solve(hess, grad_flat)
        theta = theta + 0.5 * step

    return theta.reshape(K, d_obs)


def covariate_choice_filter(
    choices: ArrayLike,
    n_options: int,
    covariates: Optional[ArrayLike] = None,
    input_gain: Optional[ArrayLike] = None,
    obs_covariates: Optional[ArrayLike] = None,
    obs_weights: Optional[ArrayLike] = None,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    decay: float = 1.0,
    init_mean: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
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


def covariate_choice_smoother(
    choices: ArrayLike,
    n_options: int,
    covariates: Optional[ArrayLike] = None,
    input_gain: Optional[ArrayLike] = None,
    obs_covariates: Optional[ArrayLike] = None,
    obs_weights: Optional[ArrayLike] = None,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    decay: float = 1.0,
    init_mean: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
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
    Q = jnp.eye(k_free) * process_noise
    A = jnp.eye(k_free) * decay

    sm_means, sm_covs, cross_covs = _rts_smoother_pass(
        filt.filtered_values, filt.filtered_covariances, Q, A,
    )

    smoothed_values = jnp.concatenate([sm_means, filt.filtered_values[-1:]])
    smoothed_covs = jnp.concatenate([sm_covs, filt.filtered_covariances[-1:]])

    return ChoiceSmootherResult(
        smoothed_values=smoothed_values,
        smoothed_covariances=smoothed_covs,
        smoother_cross_cov=cross_covs,
        marginal_log_likelihood=filt.marginal_log_likelihood,
    )


class CovariateChoiceModel(SGDFittableMixin):
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
        if n_options < 2:
            raise ValueError(f"n_options must be >= 2, got {n_options}")
        self.n_options = n_options
        self.n_covariates = n_covariates
        self.n_obs_covariates = n_obs_covariates
        self.inverse_temperature = init_inverse_temperature
        self.process_noise = init_process_noise
        self.decay = init_decay
        self.learn_inverse_temperature = learn_inverse_temperature
        self.learn_process_noise = learn_process_noise
        self.learn_decay = learn_decay
        self.learn_obs_weights = learn_obs_weights

        k_free = n_options - 1
        self.input_gain_: Array = jnp.zeros((k_free, max(n_covariates, 1)))
        self.obs_weights_: Array = jnp.zeros((n_options, max(n_obs_covariates, 1)))

        # Fitted state
        self._smoother_result: Optional[ChoiceSmootherResult] = None
        self.log_likelihood_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.log_likelihood_history_: Optional[list[float]] = None
        self._n_trials: Optional[int] = None
        self._covariates: Optional[Array] = None
        self._obs_covariates: Optional[Array] = None

    def __repr__(self) -> str:
        fitted = self.is_fitted
        return (
            f"CovariateChoiceModel(n_options={self.n_options}, "
            f"n_covariates={self.n_covariates}, "
            f"beta={self.inverse_temperature:.3f}, "
            f"Q={self.process_noise:.4f}, "
            f"decay={self.decay:.4f}, fitted={fitted})"
        )

    @property
    def is_fitted(self) -> bool:
        return self._smoother_result is not None

    @property
    def smoothed_values(self) -> Array:
        """Smoothed option values, shape (n_trials, K-1)."""
        self._check_fitted("smoothed_values")
        return self._smoother_result.smoothed_values

    @property
    def smoothed_covariances(self) -> Array:
        """Smoothed covariances, shape (n_trials, K-1, K-1)."""
        self._check_fitted("smoothed_covariances")
        return self._smoother_result.smoothed_covariances

    def _check_fitted(self, method: str) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"CovariateChoiceModel.{method}() called before fitting. "
                f"Call model.fit(choices) first."
            )

    def fit(
        self,
        choices: ArrayLike,
        covariates: Optional[ArrayLike] = None,
        obs_covariates: Optional[ArrayLike] = None,
        max_iter: int = 50,
        tolerance: float = 1e-4,
        verbose: bool = False,
        beta_grid: Optional[ArrayLike] = None,
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
            Convergence tolerance on relative log-likelihood change.
        verbose : bool
            Print progress each iteration.
        beta_grid : ArrayLike or None
            Candidate inverse temperatures for grid search.

        Returns
        -------
        log_likelihoods : list of float
        """
        choices_arr = jnp.asarray(choices, dtype=jnp.int32)
        self._n_trials = int(choices_arr.shape[0])

        if self._n_trials < 2:
            raise ValueError(
                f"Need at least 2 trials for EM fitting, got {self._n_trials}"
            )

        choices_np = np.asarray(choices)
        if np.any(choices_np < 0) or np.any(choices_np >= self.n_options):
            raise ValueError(
                f"All choices must be in [0, {self.n_options}), "
                f"got range [{choices_np.min()}, {choices_np.max()}]"
            )

        # Dynamics covariates
        if self.n_covariates > 0 and covariates is not None:
            self._covariates = jnp.asarray(covariates)
            if self._covariates.shape[1] != self.n_covariates:
                raise ValueError(
                    f"covariates has {self._covariates.shape[1]} columns but "
                    f"model expects n_covariates={self.n_covariates}"
                )
        elif self.n_covariates > 0 and covariates is None:
            raise ValueError(
                f"Model has n_covariates={self.n_covariates} but no "
                f"covariates were passed to fit()"
            )
        else:
            self._covariates = None

        # Observation covariates
        if self.n_obs_covariates > 0 and obs_covariates is not None:
            self._obs_covariates = jnp.asarray(obs_covariates)
            if self._obs_covariates.shape[1] != self.n_obs_covariates:
                raise ValueError(
                    f"obs_covariates has {self._obs_covariates.shape[1]} columns "
                    f"but model expects n_obs_covariates={self.n_obs_covariates}"
                )
        elif self.n_obs_covariates > 0 and obs_covariates is None:
            raise ValueError(
                f"Model has n_obs_covariates={self.n_obs_covariates} but no "
                f"obs_covariates were passed to fit()"
            )
        else:
            self._obs_covariates = None

        if beta_grid is None:
            beta_grid = jnp.array(_DEFAULT_BETA_GRID)
        else:
            beta_grid = jnp.asarray(beta_grid)

        log_likelihoods = []

        for iteration in range(max_iter):
            # E-step: run smoother
            smooth = covariate_choice_smoother(
                choices_arr, self.n_options,
                covariates=self._covariates,
                input_gain=self.input_gain_ if self.n_covariates > 0 else None,
                obs_covariates=self._obs_covariates,
                obs_weights=self.obs_weights_ if self.n_obs_covariates > 0 else None,
                process_noise=self.process_noise,
                inverse_temperature=self.inverse_temperature,
                decay=self.decay,
            )
            ll = float(smooth.marginal_log_likelihood)
            log_likelihoods.append(ll)

            if verbose:
                logger.info(
                    "EM iter %d: LL=%.2f, beta=%.3f, Q=%.6f, decay=%.4f",
                    iteration + 1, ll,
                    self.inverse_temperature, self.process_noise, self.decay,
                )

            # Check convergence
            if len(log_likelihoods) > 1:
                prev_ll = log_likelihoods[-2]
                if abs(prev_ll) > 0:
                    rel_change = abs(ll - prev_ll) / abs(prev_ll)
                else:
                    rel_change = abs(ll - prev_ll)
                if rel_change < tolerance:
                    if verbose:
                        logger.info("Converged at iteration %d", iteration + 1)
                    break

            # M-step for B (input gain)
            if self.n_covariates > 0 and self._covariates is not None:
                self.input_gain_ = m_step_input_gain(
                    smooth.smoothed_values, self._covariates,
                )

            # M-step for Theta (observation weights)
            if (self.learn_obs_weights
                    and self.n_obs_covariates > 0
                    and self._obs_covariates is not None):
                self.obs_weights_ = m_step_obs_weights(
                    smooth.smoothed_values, choices_arr,
                    self._obs_covariates, self.n_options,
                    self.inverse_temperature, self.obs_weights_,
                )

            # M-step for decay
            if self.learn_decay:
                self.decay = self._m_step_decay(smooth)

            # M-step for process noise Q
            if self.learn_process_noise:
                self.process_noise = self._m_step_process_noise(smooth)

            # M-step for inverse temperature beta
            if self.learn_inverse_temperature:
                self.inverse_temperature = self._m_step_beta(
                    choices_arr, beta_grid,
                )

        # Final E-step
        self._smoother_result = covariate_choice_smoother(
            choices_arr, self.n_options,
            covariates=self._covariates,
            input_gain=self.input_gain_ if self.n_covariates > 0 else None,
            obs_covariates=self._obs_covariates,
            obs_weights=self.obs_weights_ if self.n_obs_covariates > 0 else None,
            process_noise=self.process_noise,
            inverse_temperature=self.inverse_temperature,
            decay=self.decay,
        )
        self.log_likelihood_ = float(self._smoother_result.marginal_log_likelihood)
        self.n_iter_ = len(log_likelihoods)
        self.log_likelihood_history_ = log_likelihoods

        return log_likelihoods

    def fit_sgd(
        self,
        choices: ArrayLike,
        covariates: Optional[ArrayLike] = None,
        obs_covariates: Optional[ArrayLike] = None,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
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
        # --- Validate inputs ---
        choices_arr = jnp.asarray(choices, dtype=jnp.int32)
        self._n_trials = int(choices_arr.shape[0])

        if self._n_trials < 2:
            raise ValueError(
                f"Need at least 2 trials for SGD fitting, got {self._n_trials}"
            )

        choices_np = np.asarray(choices)
        if np.any(choices_np < 0) or np.any(choices_np >= self.n_options):
            raise ValueError(
                f"All choices must be in [0, {self.n_options}), "
                f"got range [{choices_np.min()}, {choices_np.max()}]"
            )

        if self.n_covariates > 0 and covariates is not None:
            self._covariates = jnp.asarray(covariates)
        elif self.n_covariates > 0:
            raise ValueError(
                f"Model has n_covariates={self.n_covariates} but no "
                f"covariates were passed"
            )
        else:
            self._covariates = None

        if self.n_obs_covariates > 0 and obs_covariates is not None:
            self._obs_covariates = jnp.asarray(obs_covariates)
        elif self.n_obs_covariates > 0:
            raise ValueError(
                f"Model has n_obs_covariates={self.n_obs_covariates} but no "
                f"obs_covariates were passed"
            )
        else:
            self._obs_covariates = None

        return super().fit_sgd(
            choices_arr,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    # --- SGDFittableMixin protocol ---

    @property
    def _n_timesteps(self) -> int:
        return self._n_trials

    def _check_sgd_initialized(self) -> None:
        pass  # Parameters are allocated at construction time

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            POSITIVE,
            UNCONSTRAINED,
            UNIT_INTERVAL,
        )

        params: dict = {}
        spec: dict = {}
        if self.learn_process_noise:
            params["process_noise"] = jnp.array(self.process_noise)
            spec["process_noise"] = POSITIVE
        if self.learn_inverse_temperature:
            params["inverse_temperature"] = jnp.array(self.inverse_temperature)
            spec["inverse_temperature"] = POSITIVE
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
        k_free = self.n_options - 1

        if self._covariates is not None:
            cov_arr = self._covariates
            ig_arr = params.get("input_gain", self.input_gain_)
        else:
            cov_arr = jnp.zeros((self._n_trials, 1))
            ig_arr = jnp.zeros((k_free, 1))

        if self._obs_covariates is not None:
            obs_cov_arr = self._obs_covariates
            ow_arr = params.get("obs_weights", self.obs_weights_)
        else:
            obs_cov_arr = jnp.zeros((self._n_trials, 1))
            ow_arr = jnp.zeros((self.n_options, 1))

        result = _covariate_choice_filter_jit(
            choices, self.n_options,
            cov_arr, ig_arr,
            obs_cov_arr, ow_arr,
            params.get("process_noise", jnp.array(self.process_noise)),
            params.get("inverse_temperature", jnp.array(self.inverse_temperature)),
            params.get("decay", jnp.array(self.decay)),
            jnp.zeros(k_free),
            jnp.eye(k_free),
        )
        return -result.marginal_log_likelihood

    def _store_sgd_params(self, params: dict) -> None:
        if "process_noise" in params:
            self.process_noise = float(params["process_noise"])
        if "inverse_temperature" in params:
            self.inverse_temperature = float(params["inverse_temperature"])
        if "decay" in params:
            self.decay = float(params["decay"])
        if "input_gain" in params:
            self.input_gain_ = params["input_gain"]
        if "obs_weights" in params:
            self.obs_weights_ = params["obs_weights"]

    def _finalize_sgd(self, choices: Array) -> None:
        self._smoother_result = covariate_choice_smoother(
            choices, self.n_options,
            covariates=self._covariates,
            input_gain=self.input_gain_ if self.n_covariates > 0 else None,
            obs_covariates=self._obs_covariates,
            obs_weights=self.obs_weights_ if self.n_obs_covariates > 0 else None,
            process_noise=self.process_noise,
            inverse_temperature=self.inverse_temperature,
            decay=self.decay,
        )
        self.log_likelihood_ = float(self._smoother_result.marginal_log_likelihood)

    def _m_step_decay(self, smooth: ChoiceSmootherResult) -> float:
        """M-step: update scalar decay from smoother statistics.

        For x_t = a * x_{t-1} + B u_t + w_t, the M-step for scalar a is:
            a = sum_t [m_t - B u_t]' m_{t-1} / sum_t [m_{t-1}' m_{t-1} + tr(P_{t-1})]
        This is a weighted regression of (m_t - B u_t) on m_{t-1}.
        """
        m = smooth.smoothed_values  # (T, K-1)
        P = smooth.smoothed_covariances  # (T, K-1, K-1)

        target = m[1:]  # (T-1, K-1)
        if self.n_covariates > 0 and self._covariates is not None:
            u = self._covariates[1:]
            target = target - u @ self.input_gain_.T

        # Numerator: sum_t (target_t)' m_{t-1}
        numer = jnp.sum(target * m[:-1])
        # Denominator: sum_t (m_{t-1}' m_{t-1} + tr(P_{t-1}))
        denom = jnp.sum(m[:-1] ** 2) + jnp.sum(
            jnp.trace(P[:-1], axis1=1, axis2=2)
        )
        a = float(jnp.clip(numer / jnp.maximum(denom, 1e-10), 0.01, 1.0))
        return a

    def _m_step_process_noise(self, smooth: ChoiceSmootherResult) -> float:
        """M-step: update scalar process noise from smoother statistics.

        Accounts for transition matrix A and dynamics covariates B.
        Residual: m_t - A m_{t-1} - B u_t.
        """
        m = smooth.smoothed_values
        P = smooth.smoothed_covariances
        C = smooth.smoother_cross_cov
        a = self.decay

        T_minus_1 = m.shape[0] - 1
        diff = m[1:] - a * m[:-1]  # (T-1, K-1)

        # Subtract covariate contribution if present
        if self.n_covariates > 0 and self._covariates is not None:
            u = self._covariates[1:]
            diff = diff - u @ self.input_gain_.T

        # Q_hat with transition: E[(x_t - A x_{t-1})(...)'] =
        #   diff diff' + P_t + a^2 P_{t-1} - 2a C_{t-1,t}
        Q_hat = (
            jnp.einsum("ti,tj->ij", diff, diff)
            + jnp.sum(P[1:], axis=0)
            + a**2 * jnp.sum(P[:-1], axis=0)
            - 2 * a * jnp.sum(C, axis=0)
        ) / T_minus_1
        q = float(jnp.maximum(jnp.mean(jnp.diag(Q_hat)), 1e-8))
        return q

    def _m_step_beta(
        self,
        choices: Array,
        beta_grid: Array,
    ) -> float:
        """M-step: grid search + golden-section refinement for beta."""
        def _eval_beta(beta):
            result = covariate_choice_filter(
                choices, self.n_options,
                covariates=self._covariates,
                input_gain=self.input_gain_ if self.n_covariates > 0 else None,
                obs_covariates=self._obs_covariates,
                obs_weights=self.obs_weights_ if self.n_obs_covariates > 0 else None,
                process_noise=self.process_noise,
                inverse_temperature=beta,
                decay=self.decay,
            )
            return result.marginal_log_likelihood

        lls = jax.vmap(
            lambda b: _eval_beta(b),
            in_axes=0,
        )(beta_grid)

        best_idx = int(jnp.argmax(lls))
        best_beta = float(beta_grid[best_idx])

        lo_idx = max(0, best_idx - 1)
        hi_idx = min(len(beta_grid) - 1, best_idx + 1)
        lo = float(beta_grid[lo_idx])
        hi = float(beta_grid[hi_idx])

        if hi - lo < 1e-10:
            return best_beta

        gr = (math.sqrt(5) + 1) / 2
        for _ in range(10):
            c = hi - (hi - lo) / gr
            d = lo + (hi - lo) / gr
            ll_c = float(_eval_beta(c))
            ll_d = float(_eval_beta(d))
            if ll_c > ll_d:
                hi = d
            else:
                lo = c

        return (lo + hi) / 2

    def choice_probabilities(self) -> Array:
        """Softmax choice probabilities from smoothed values.

        Includes observation covariate offsets if present.

        Returns
        -------
        probs : Array, shape (n_trials, K)
        """
        self._check_fitted("choice_probabilities")
        zeros = jnp.zeros((self._smoother_result.smoothed_values.shape[0], 1))
        full_values = jnp.concatenate(
            [zeros, self._smoother_result.smoothed_values], axis=1,
        )
        logits = self.inverse_temperature * full_values
        if self.n_obs_covariates > 0 and self._obs_covariates is not None:
            logits = logits + self._obs_covariates @ self.obs_weights_.T
        return jax.nn.softmax(logits, axis=1)

    @property
    def n_free_params(self) -> int:
        """Number of free parameters learned by EM."""
        n = 0
        if self.learn_process_noise:
            n += 1
        if self.learn_inverse_temperature:
            n += 1
        if self.n_covariates > 0:
            n += (self.n_options - 1) * self.n_covariates  # B matrix
        if self.n_obs_covariates > 0 and self.learn_obs_weights:
            n += self.n_options * self.n_obs_covariates  # Theta matrix
        if self.learn_decay:
            n += 1  # scalar decay
        return n

    def bic(self) -> float:
        """Bayesian Information Criterion."""
        self._check_fitted("bic")
        return (
            -2.0 * self.log_likelihood_
            + self.n_free_params * math.log(self._n_trials)
        )

    def compare_to_null(self) -> dict:
        """Compare fitted model to a null (uniform 1/K) model."""
        self._check_fitted("compare_to_null")
        null_ll_per_trial = math.log(1.0 / self.n_options)
        null_ll = null_ll_per_trial * self._n_trials
        null_bic = -2.0 * null_ll

        model_bic = self.bic()
        delta_bic = null_bic - model_bic

        return {
            "model_ll": self.log_likelihood_,
            "null_ll": null_ll,
            "model_bic": model_bic,
            "null_bic": null_bic,
            "delta_bic": delta_bic,
            "learning_detected": delta_bic > 2.0,
        }

    def summary(self) -> str:
        """Text summary of fitted model."""
        self._check_fitted("summary")
        comparison = self.compare_to_null()
        lines = [
            "CovariateChoiceModel Summary",
            "=" * 40,
            f"  n_options:             {self.n_options}",
            f"  n_covariates:          {self.n_covariates}",
            f"  inverse_temperature:   {self.inverse_temperature:.4f}",
            f"  process_noise:         {self.process_noise:.6f}",
            f"  n_trials:              {self._n_trials}",
            f"  n_em_iterations:       {self.n_iter_}",
            f"  log_likelihood:        {self.log_likelihood_:.2f}",
            f"  BIC:                   {self.bic():.2f}",
            f"  null_ll (uniform):     {comparison['null_ll']:.2f}",
            f"  delta_BIC:             {comparison['delta_bic']:.2f}",
            f"  learning_detected:     {comparison['learning_detected']}",
        ]
        if self.n_covariates > 0:
            lines.append("  input_gain (B):")
            B = np.array(self.input_gain_)
            for i in range(B.shape[0]):
                lines.append(
                    f"    option {i + 1}: "
                    + ", ".join(f"{B[i, j]:.4f}" for j in range(B.shape[1]))
                )
        return "\n".join(lines)

    def plot_values(self, observed_choices=None, option_labels=None, ax=None):
        """Plot smoothed option values and choice probabilities.

        Returns
        -------
        fig, axes
        """
        import matplotlib.pyplot as plt

        self._check_fitted("plot_values")

        if option_labels is None:
            option_labels = [f"Option {i}" for i in range(self.n_options)]

        vals = np.array(self._smoother_result.smoothed_values)
        covs = np.array(self._smoother_result.smoothed_covariances)
        probs = np.array(self.choice_probabilities())
        trials = np.arange(vals.shape[0])

        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        for k in range(self.n_options - 1):
            std = np.sqrt(covs[:, k, k])
            axes[0].plot(trials, vals[:, k], label=option_labels[k + 1])
            axes[0].fill_between(
                trials, vals[:, k] - 1.96 * std, vals[:, k] + 1.96 * std,
                alpha=0.2,
            )
        axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5,
                         label=f"{option_labels[0]} (ref)")
        axes[0].set_ylabel("Relative value")
        axes[0].set_title("Smoothed Option Values")
        axes[0].legend(fontsize=8)

        axes[1].stackplot(trials, probs.T, labels=option_labels, alpha=0.7)
        if observed_choices is not None:
            choices_np = np.asarray(observed_choices)
            for k in range(self.n_options):
                chosen_trials = trials[choices_np == k]
                if len(chosen_trials) > 0:
                    axes[1].eventplot(
                        chosen_trials, lineoffsets=1.02 - k * 0.03,
                        linelengths=0.02, colors="k", alpha=0.4,
                    )
        axes[1].set_ylabel("Choice probability")
        axes[1].set_xlabel("Trial")
        axes[1].set_title("Choice Probabilities")
        axes[1].legend(fontsize=8, loc="upper right")

        fig.tight_layout()
        return fig, axes

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

    def plot_convergence(self, ax=None):
        """Plot EM log-likelihood convergence.

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt

        self._check_fitted("plot_convergence")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure

        ax.plot(range(1, len(self.log_likelihood_history_) + 1),
                self.log_likelihood_history_, "o-")
        ax.set_xlabel("EM Iteration")
        ax.set_ylabel("Log-Likelihood")
        ax.set_title("EM Convergence")

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

        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(1, 3, figure=fig)

        # Panel 1: latent values
        ax0 = fig.add_subplot(gs[0, 0])
        vals = np.array(self._smoother_result.smoothed_values)
        covs = np.array(self._smoother_result.smoothed_covariances)
        if option_labels is None:
            option_labels = [f"Option {i}" for i in range(self.n_options)]
        trials = np.arange(vals.shape[0])
        for k in range(self.n_options - 1):
            std = np.sqrt(covs[:, k, k])
            ax0.plot(trials, vals[:, k], label=option_labels[k + 1])
            ax0.fill_between(trials, vals[:, k] - 1.96 * std,
                             vals[:, k] + 1.96 * std, alpha=0.2)
        ax0.axhline(0, color="gray", linestyle="--", alpha=0.5,
                     label=f"{option_labels[0]} (ref)")
        ax0.set_ylabel("Relative value")
        ax0.set_title("Smoothed Values")
        ax0.legend(fontsize=7)

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
    input_gain: Optional[ArrayLike] = None,
    process_noise: float = 0.005,
    inverse_temperature: float = 2.0,
    reward_prob: float = 0.7,
    seed: int = 42,
) -> SimulatedRLChoiceData:
    """Simulate multi-armed bandit data with covariate-driven value evolution.

    Generates choices from a softmax model where option values evolve
    according to ``x_t = x_{t-1} + B @ u_t + noise``. Covariates follow
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
                values[t - 1]
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
