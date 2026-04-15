"""Multinomial choice learning model via Laplace-EKF.

Tracks evolving option values from a sequence of choices in a
multi-armed bandit task. The latent state x_t in R^{K-1} represents
relative values for options 1..K-1 (option 0 is the reference,
fixed at 0 for identifiability). Choices are modeled as
Categorical(softmax(beta * [0, x_t])).

References
----------
[1] Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J.
    (2006). Cortical substrates for exploratory decisions in humans.
    Nature 441, 876-879.
[2] Smith, A.C., Frank, L.M., Wirth, S. et al. (2004). Dynamic analysis
    of learning in behavioral experiments. J Neuroscience 24(2), 447-461.
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

from state_space_practice.sgd_fitting import SGDFittableMixin

from state_space_practice.kalman import (
    _kalman_smoother_update,
    psd_solve,
    symmetrize,
)
from state_space_practice.point_process_kalman import _logdet_psd

logger = logging.getLogger(__name__)


def _softmax_update_core(
    prior_mean: Array,
    prior_cov: Array,
    choice: Array,
    n_options: int,
    inverse_temperature: float,
    max_newton_steps: int = 3,
    obs_offset: Optional[Array] = None,
) -> tuple[Array, Array, Array]:
    """JIT-compatible Laplace-EKF update for softmax observation.

    All inputs must be JAX arrays (no Python-level validation).
    Use ``softmax_observation_update`` for the public API with validation.

    Parameters
    ----------
    max_newton_steps : int, default 3
        Number of Newton-Raphson iterations for Laplace mode-finding.
        These are unrolled at JIT compile time, so large values (> ~10)
        significantly increase compilation time and XLA graph size
        without proportional accuracy gains for well-conditioned problems.
    obs_offset : Array or None, shape (K,)
        Additive offset to the logits before softmax. Used for
        observation covariates (e.g., stay bias, spatial bias).
        These shift choice probabilities without changing the
        latent value state. None means no offset.
    """
    beta = inverse_temperature
    k_free = n_options - 1

    # Precompute constants
    e_k = jnp.zeros(n_options).at[choice].set(1.0)
    e_k_free = e_k[1:]
    eye_k = jnp.eye(k_free)
    zero_ref = jnp.zeros(1)
    beta_sq = beta**2
    _obs_offset = obs_offset if obs_offset is not None else jnp.zeros(n_options)

    # Prior precision
    prior_precision = psd_solve(prior_cov, eye_k)

    # Fixed Newton iterations (unrolled for JIT compatibility)
    x = prior_mean
    for _ in range(max_newton_steps):
        v = jnp.concatenate([zero_ref, x])
        p_free = jax.nn.softmax(beta * v + _obs_offset)[1:]

        gradient = beta * (e_k_free - p_free)
        neg_hessian = beta_sq * (jnp.diag(p_free) - jnp.outer(p_free, p_free))

        posterior_precision = prior_precision + neg_hessian
        rhs = gradient + prior_precision @ (prior_mean - x)
        x = x + psd_solve(posterior_precision, rhs)

    # Final posterior covariance at the mode
    v = jnp.concatenate([zero_ref, x])
    p_free = jax.nn.softmax(beta * v + _obs_offset)[1:]
    neg_hessian = beta_sq * (jnp.diag(p_free) - jnp.outer(p_free, p_free))
    posterior_precision = prior_precision + neg_hessian
    posterior_cov = symmetrize(psd_solve(posterior_precision, eye_k))

    # Laplace-approximated marginal log-likelihood log p(c_t | y_{1:t-1}):
    #   ≈ log p(c_t | x*) + log p(x* | y_{1:t-1}) + ½ log|Σ_post| + const
    # where x* is the posterior mode, and (k/2)log(2π) cancels.
    # See point_process_kalman._stochastic_point_process_filter_step for
    # the same derivation applied to Poisson observations.
    log_lik_at_mode = jax.nn.log_softmax(beta * v + _obs_offset)[choice]
    delta = x - prior_mean
    quad = delta @ (prior_precision @ delta)
    logdet_prior = _logdet_psd(prior_cov)
    logdet_post = _logdet_psd(posterior_cov)
    log_lik = log_lik_at_mode - 0.5 * quad - 0.5 * logdet_prior + 0.5 * logdet_post

    return x, posterior_cov, log_lik


def softmax_observation_update(
    prior_mean: Array,
    prior_cov: Array,
    choice: int,
    n_options: int,
    inverse_temperature: float = 1.0,
    max_newton_steps: int = 3,
) -> tuple[Array, Array, Array]:
    """Laplace-EKF update for a categorical observation with softmax link.

    The latent state x in R^{K-1} represents relative values for options
    1 through K-1. Option 0 is the reference (value fixed at 0).

    Parameters
    ----------
    prior_mean : Array, shape (K-1,)
        Prior state mean from prediction step.
    prior_cov : Array, shape (K-1, K-1)
        Prior state covariance from prediction step.
    choice : int
        Observed choice (0-indexed, 0 = reference option).
    n_options : int
        Total number of options K.
    inverse_temperature : float
        Softmax inverse temperature beta.
    max_newton_steps : int
        Maximum Newton iterations for Laplace mode-finding.

    Returns
    -------
    posterior_mean : Array, shape (K-1,)
    posterior_cov : Array, shape (K-1, K-1)
    log_likelihood : Array, scalar
        Log-likelihood log P(choice | prior_mean) evaluated at the
        prior mean (for EM monitoring).
    """
    if choice < 0 or choice >= n_options:
        raise ValueError(
            f"choice must be in [0, {n_options}), got {choice}"
        )
    return _softmax_update_core(
        prior_mean, prior_cov, jnp.int32(choice),
        n_options, inverse_temperature, max_newton_steps,
    )


class ChoiceFilterResult(NamedTuple):
    """Result of multinomial choice filtering.

    Attributes
    ----------
    filtered_values : Array, shape (n_trials, K-1)
        Posterior state means after each observation.
    filtered_covariances : Array, shape (n_trials, K-1, K-1)
        Posterior covariances after each observation.
    predicted_values : Array, shape (n_trials, K-1)
        Prior state means before each observation (for diagnostics).
    predicted_covariances : Array, shape (n_trials, K-1, K-1)
        Prior covariances before each observation.
    marginal_log_likelihood : Array
        Sum of per-trial log-likelihoods.
    """
    filtered_values: Array
    filtered_covariances: Array
    predicted_values: Array
    predicted_covariances: Array
    marginal_log_likelihood: Array


class ChoiceSmootherResult(NamedTuple):
    """Result of multinomial choice smoothing.

    Attributes
    ----------
    smoothed_values : Array, shape (n_trials, K-1)
    smoothed_covariances : Array, shape (n_trials, K-1, K-1)
    smoother_cross_cov : Array, shape (n_trials-1, K-1, K-1)
        Cross-covariance Cov(x_t, x_{t+1} | y_{1:T}).
    marginal_log_likelihood : Array
    """
    smoothed_values: Array
    smoothed_covariances: Array
    smoother_cross_cov: Array
    marginal_log_likelihood: Array


def multinomial_choice_filter(
    choices: ArrayLike,
    n_options: int,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    init_mean: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
) -> ChoiceFilterResult:
    """Forward filter for multinomial choice model.

    Parameters
    ----------
    choices : ArrayLike, shape (n_trials,)
        Observed choices (0-indexed integers in [0, K)).
    n_options : int
        Total number of options K.
    process_noise : float
        Scalar process noise (Q = process_noise * I).
    inverse_temperature : float
        Softmax inverse temperature beta.
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

    # Resolve defaults before JIT boundary
    if init_mean is None:
        init_mean = jnp.zeros(k_free)
    else:
        init_mean = jnp.asarray(init_mean)
    if init_cov is None:
        init_cov = jnp.eye(k_free)
    else:
        init_cov = jnp.asarray(init_cov)

    return _multinomial_choice_filter_jit(
        choices_arr, n_options, process_noise, inverse_temperature,
        init_mean, init_cov,
    )


@partial(jax.jit, static_argnames=("n_options",))
def _multinomial_choice_filter_jit(
    choices: Array,
    n_options: int,
    process_noise: float,
    inverse_temperature: float,
    init_mean: Array,
    init_cov: Array,
) -> ChoiceFilterResult:
    """JIT-compiled filter core."""
    k_free = n_options - 1
    Q = jnp.eye(k_free) * process_noise

    def _step(carry, choice_t):
        filt_mean, filt_cov, total_ll = carry

        # Predict (random walk: A = I)
        pred_mean = filt_mean
        pred_cov = filt_cov + Q

        # Update
        post_mean, post_cov, ll = _softmax_update_core(
            pred_mean, pred_cov, choice_t,
            n_options, inverse_temperature,
        )

        total_ll = total_ll + ll
        return (post_mean, post_cov, total_ll), (
            post_mean, post_cov, pred_mean, pred_cov
        )

    init_carry = (init_mean, init_cov, jnp.array(0.0))
    (_, _, marginal_ll), (filt_vals, filt_covs, pred_vals, pred_covs) = (
        jax.lax.scan(_step, init_carry, choices)
    )

    return ChoiceFilterResult(
        filtered_values=filt_vals,
        filtered_covariances=filt_covs,
        predicted_values=pred_vals,
        predicted_covariances=pred_covs,
        marginal_log_likelihood=marginal_ll,
    )


@jax.jit
def _rts_smoother_pass(
    filtered_values: Array,
    filtered_covariances: Array,
    Q: Array,
    A: Optional[Array] = None,
) -> tuple[Array, Array, Array]:
    """JIT-compiled RTS backward smoother.

    Parameters
    ----------
    A : Array or None
        Transition matrix. None defaults to identity (random walk).
    """
    A = A if A is not None else jnp.eye(Q.shape[0])

    def _smooth_step(carry, inputs):
        next_sm_mean, next_sm_cov = carry
        f_mean, f_cov = inputs

        sm_mean, sm_cov, cross_cov = _kalman_smoother_update(
            next_sm_mean, next_sm_cov,
            f_mean, f_cov,
            Q, A,
        )
        return (sm_mean, sm_cov), (sm_mean, sm_cov, cross_cov)

    _, (sm_means, sm_covs, cross_covs) = jax.lax.scan(
        _smooth_step,
        (filtered_values[-1], filtered_covariances[-1]),
        (filtered_values[:-1], filtered_covariances[:-1]),
        reverse=True,
    )
    return sm_means, sm_covs, cross_covs


def multinomial_choice_smoother(
    choices: ArrayLike,
    n_options: int,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    init_mean: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
) -> ChoiceSmootherResult:
    """Forward filter + RTS backward smoother for multinomial choice model.

    Parameters are the same as :func:`multinomial_choice_filter`.

    Returns
    -------
    ChoiceSmootherResult
    """
    filt = multinomial_choice_filter(
        choices, n_options, process_noise, inverse_temperature,
        init_mean, init_cov,
    )

    k_free = n_options - 1
    Q = jnp.eye(k_free) * process_noise

    sm_means, sm_covs, cross_covs = _rts_smoother_pass(
        filt.filtered_values, filt.filtered_covariances, Q,
    )

    # Append last filtered state (smoother[-1] == filter[-1])
    smoothed_values = jnp.concatenate([sm_means, filt.filtered_values[-1:]])
    smoothed_covs = jnp.concatenate([sm_covs, filt.filtered_covariances[-1:]])

    return ChoiceSmootherResult(
        smoothed_values=smoothed_values,
        smoothed_covariances=smoothed_covs,
        smoother_cross_cov=cross_covs,
        marginal_log_likelihood=filt.marginal_log_likelihood,
    )


_DEFAULT_BETA_GRID = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


class MultinomialChoiceModel(SGDFittableMixin):
    """Multi-armed bandit choice model with evolving option values.

    Tracks latent option values from a sequence of choices using a
    state-space model with softmax observation model. Uses EM to learn
    the drift rate (process noise) and exploration-exploitation tradeoff
    (inverse temperature).

    The latent state x_t in R^{K-1} represents the relative value of
    options 1 through K-1, with option 0 as the reference (value = 0).

    Typical workflow::

        model = MultinomialChoiceModel(n_options=4)
        model.fit(choices, verbose=True)
        print(model.summary())

    Parameters
    ----------
    n_options : int
        Number of choice options K.
    init_inverse_temperature : float
        Starting inverse temperature for EM.
    init_process_noise : float
        Starting process noise for EM.
    learn_inverse_temperature : bool
        Whether to learn beta via EM.
    learn_process_noise : bool
        Whether to learn Q via EM.
    """

    def __init__(
        self,
        n_options: int,
        init_inverse_temperature: float = 1.0,
        init_process_noise: float = 0.01,
        learn_inverse_temperature: bool = True,
        learn_process_noise: bool = True,
    ):
        if n_options < 2:
            raise ValueError(f"n_options must be >= 2, got {n_options}")
        self.n_options = n_options
        self.inverse_temperature = init_inverse_temperature
        self.process_noise = init_process_noise
        self.learn_inverse_temperature = learn_inverse_temperature
        self.learn_process_noise = learn_process_noise

        # Fitted state (populated by fit())
        self._smoother_result: Optional[ChoiceSmootherResult] = None
        self.log_likelihood_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.log_likelihood_history_: Optional[list[float]] = None
        self._n_trials: Optional[int] = None

        # Uncertainty summaries (populated after fitting)
        self.predicted_option_variances_: Optional[Array] = None
        self.smoothed_option_variances_: Optional[Array] = None
        self.predicted_choice_entropy_: Optional[Array] = None
        self.surprise_: Optional[Array] = None

    def __repr__(self) -> str:
        fitted = self.is_fitted
        return (
            f"MultinomialChoiceModel(n_options={self.n_options}, "
            f"beta={self.inverse_temperature:.3f}, "
            f"Q={self.process_noise:.4f}, fitted={fitted})"
        )

    @property
    def is_fitted(self) -> bool:
        return self._smoother_result is not None

    def _populate_uncertainty(self, choices: Array) -> None:
        """Compute uncertainty summaries from filter + smoother results.

        Note: runs the filter once to get predicted quantities (predicted_values,
        predicted_covariances) which are not stored on the smoother result.
        """
        from state_space_practice.behavioral_uncertainty import (
            append_reference_option,
            categorical_entropy,
            compute_surprise,
            option_variances_from_covariances,
        )

        filt = multinomial_choice_filter(
            choices, self.n_options,
            self.process_noise, self.inverse_temperature,
        )

        # Option values (full K with reference option appended)
        self.predicted_option_values_ = append_reference_option(filt.predicted_values)
        self.filtered_option_values_ = append_reference_option(filt.filtered_values)
        self.smoothed_option_values_ = append_reference_option(
            self._smoother_result.smoothed_values
        )

        # Option variances (full K options)
        self.predicted_option_variances_ = option_variances_from_covariances(
            filt.predicted_covariances
        )
        self.filtered_option_variances_ = option_variances_from_covariances(
            filt.filtered_covariances
        )
        self.smoothed_option_variances_ = option_variances_from_covariances(
            self._smoother_result.smoothed_covariances
        )

        # Predicted choice entropy
        pred_probs = jax.nn.softmax(
            self.inverse_temperature * self.predicted_option_values_, axis=1
        )
        self.predicted_choice_entropy_ = categorical_entropy(pred_probs)

        # Surprise
        self.surprise_ = compute_surprise(pred_probs, choices)

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
                f"MultinomialChoiceModel.{method}() called before fitting. "
                f"Call model.fit(choices) first."
            )

    def fit(
        self,
        choices: ArrayLike,
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
        max_iter : int
            Maximum EM iterations.
        tolerance : float
            Convergence tolerance on relative log-likelihood change.
        verbose : bool
            Print progress each iteration.
        beta_grid : ArrayLike or None
            Candidate inverse temperatures for grid search.
            Default: [0.1, 0.3, 0.5, 1, 2, 3, 5, 8, 12].

        Returns
        -------
        log_likelihoods : list of float
            Log-likelihood at each EM iteration.
        """
        choices_arr = jnp.asarray(choices, dtype=jnp.int32)
        self._n_trials = int(choices_arr.shape[0])

        if self._n_trials < 2:
            raise ValueError(
                f"Need at least 2 trials for EM fitting, got {self._n_trials}"
            )

        # Validate choices
        choices_np = np.asarray(choices)
        if np.any(choices_np < 0) or np.any(choices_np >= self.n_options):
            raise ValueError(
                f"All choices must be in [0, {self.n_options}), "
                f"got range [{choices_np.min()}, {choices_np.max()}]"
            )

        if beta_grid is None:
            beta_grid = jnp.array(_DEFAULT_BETA_GRID)
        else:
            beta_grid = jnp.asarray(beta_grid)

        log_likelihoods = []

        for iteration in range(max_iter):
            # E-step: run smoother with current parameters
            smooth = multinomial_choice_smoother(
                choices_arr, self.n_options,
                process_noise=self.process_noise,
                inverse_temperature=self.inverse_temperature,
            )
            ll = float(smooth.marginal_log_likelihood)
            log_likelihoods.append(ll)

            if verbose:
                logger.info(
                    "EM iter %d: LL=%.2f, beta=%.3f, Q=%.6f",
                    iteration + 1, ll,
                    self.inverse_temperature, self.process_noise,
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

            # M-step for process noise Q
            if self.learn_process_noise:
                self.process_noise = self._m_step_process_noise(smooth)

            # M-step for inverse temperature beta
            if self.learn_inverse_temperature:
                self.inverse_temperature = self._m_step_beta(
                    choices_arr, beta_grid,
                )

        # Final E-step with learned parameters
        self._smoother_result = multinomial_choice_smoother(
            choices_arr, self.n_options,
            process_noise=self.process_noise,
            inverse_temperature=self.inverse_temperature,
        )
        self.log_likelihood_ = float(self._smoother_result.marginal_log_likelihood)
        self.n_iter_ = len(log_likelihoods)
        self.log_likelihood_history_ = log_likelihoods
        self._populate_uncertainty(choices_arr)

        return log_likelihoods

    def fit_sgd(
        self,
        choices: ArrayLike,
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
        from state_space_practice.parameter_transforms import POSITIVE

        params: dict = {}
        spec: dict = {}
        if self.learn_process_noise:
            params["process_noise"] = jnp.array(self.process_noise)
            spec["process_noise"] = POSITIVE
        if self.learn_inverse_temperature:
            params["inverse_temperature"] = jnp.array(self.inverse_temperature)
            spec["inverse_temperature"] = POSITIVE
        return params, spec

    def _sgd_loss_fn(self, params: dict, choices: Array) -> Array:
        k_free = self.n_options - 1
        result = _multinomial_choice_filter_jit(
            choices, self.n_options,
            params.get("process_noise", jnp.array(self.process_noise)),
            params.get("inverse_temperature", jnp.array(self.inverse_temperature)),
            jnp.zeros(k_free),
            jnp.eye(k_free),
        )
        return -result.marginal_log_likelihood

    def _store_sgd_params(self, params: dict) -> None:
        if "process_noise" in params:
            self.process_noise = float(params["process_noise"])
        if "inverse_temperature" in params:
            self.inverse_temperature = float(params["inverse_temperature"])

    def _finalize_sgd(self, choices: Array) -> None:
        self._smoother_result = multinomial_choice_smoother(
            choices, self.n_options,
            process_noise=self.process_noise,
            inverse_temperature=self.inverse_temperature,
        )
        self.log_likelihood_ = float(self._smoother_result.marginal_log_likelihood)
        self._populate_uncertainty(choices)

    def _m_step_process_noise(self, smooth: ChoiceSmootherResult) -> float:
        """M-step: update scalar process noise from smoother statistics.

        Uses the standard EM formula for a random walk (A=I):
            Q_hat = (1/(T-1)) * sum_{t=1}^{T-1} [
                (m_t - m_{t-1})(m_t - m_{t-1})'
                + P_t + P_{t-1} - 2 * C_{t-1,t}
            ]
        where C_{t-1,t} = Cov(x_{t-1}, x_t | y_{1:T}) from the smoother.
        Convention: smoother_cross_cov[t] = Cov(x_t, x_{t+1} | y_{1:T}),
        which pairs with diff[t] = m[t+1] - m[t].
        """
        m = smooth.smoothed_values       # (T, K-1)
        P = smooth.smoothed_covariances  # (T, K-1, K-1)
        C = smooth.smoother_cross_cov    # (T-1, K-1, K-1)

        T_minus_1 = m.shape[0] - 1
        diff = m[1:] - m[:-1]  # (T-1, K-1)
        Q_hat = (
            jnp.einsum("ti,tj->ij", diff, diff)
            + jnp.sum(P[1:], axis=0)
            + jnp.sum(P[:-1], axis=0)
            - 2 * jnp.sum(C, axis=0)
        ) / T_minus_1
        # Scalar Q: mean of diagonal, clamped
        q = float(jnp.maximum(jnp.mean(jnp.diag(Q_hat)), 1e-8))
        return q

    def _m_step_beta(
        self,
        choices: Array,
        beta_grid: Array,
    ) -> float:
        """M-step: grid search + golden-section refinement for beta."""
        # Coarse grid search: evaluate filter LL at each candidate beta
        def _eval_beta(beta):
            result = multinomial_choice_filter(
                choices, self.n_options,
                process_noise=self.process_noise,
                inverse_temperature=beta,
            )
            return result.marginal_log_likelihood

        lls = jax.vmap(
            lambda b: _eval_beta(b),
            in_axes=0,
        )(beta_grid)

        best_idx = int(jnp.argmax(lls))
        best_beta = float(beta_grid[best_idx])

        # Golden-section refinement around the best grid point
        lo_idx = max(0, best_idx - 1)
        hi_idx = min(len(beta_grid) - 1, best_idx + 1)
        lo = float(beta_grid[lo_idx])
        hi = float(beta_grid[hi_idx])

        # If at grid edge, bracket collapses — skip refinement
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

        Returns
        -------
        probs : Array, shape (n_trials, K)
            Each row sums to 1.
        """
        self._check_fitted("choice_probabilities")
        # Build full value vectors: [0, x_t] for each trial
        zeros = jnp.zeros((self._smoother_result.smoothed_values.shape[0], 1))
        full_values = jnp.concatenate(
            [zeros, self._smoother_result.smoothed_values], axis=1
        )
        return jax.nn.softmax(self.inverse_temperature * full_values, axis=1)

    @property
    def n_free_params(self) -> int:
        """Number of free parameters actually learned by EM."""
        n = 0
        if self.learn_process_noise:
            n += 1  # Q scalar
        if self.learn_inverse_temperature:
            n += 1  # beta
        return n

    def bic(self) -> float:
        """Bayesian Information Criterion.

        Only counts parameters that are actually learned via EM.
        """
        self._check_fitted("bic")
        return (
            -2.0 * self.log_likelihood_
            + self.n_free_params * math.log(self._n_trials)
        )

    def compare_to_null(self) -> dict:
        """Compare fitted model to a null (uniform 1/K) model.

        Returns
        -------
        dict with keys: model_ll, null_ll, model_bic, null_bic,
        delta_bic, learning_detected.
        """
        self._check_fitted("compare_to_null")
        null_ll_per_trial = math.log(1.0 / self.n_options)
        null_ll = null_ll_per_trial * self._n_trials
        null_bic = -2.0 * null_ll  # 0 free params

        model_bic = self.bic()
        delta_bic = null_bic - model_bic  # positive favors learning model

        return {
            "model_ll": self.log_likelihood_,
            "null_ll": null_ll,
            "model_bic": model_bic,
            "null_bic": null_bic,
            "delta_bic": delta_bic,
            "learning_detected": delta_bic > 2.0,
        }

    def summary(self) -> str:
        """Text summary of fitted model including null comparison."""
        self._check_fitted("summary")
        comparison = self.compare_to_null()
        lines = [
            "MultinomialChoiceModel Summary",
            "=" * 40,
            f"  n_options:             {self.n_options}",
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
        return "\n".join(lines)

    def plot_values(self, observed_choices=None, option_labels=None, ax=None):
        """Plot smoothed option values and choice probabilities.

        Parameters
        ----------
        observed_choices : ArrayLike or None
            If provided, marks observed choices on the probability plot.
        option_labels : list of str or None
            Labels for each option. Default: ["Option 0", ...].
        ax : array of Axes or None
            Two Axes for values and probabilities panels.

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

        # Top: latent values with CI
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

        # Bottom: choice probabilities (stacked area)
        axes[1].stackplot(trials, probs.T, labels=option_labels, alpha=0.7)
        if observed_choices is not None:
            # Mark observed choices as tick marks along the top
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
        """3-panel diagnostic: values, convergence, probabilities.

        Returns
        -------
        fig, axes : array of 3 Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        self._check_fitted("plot_summary")

        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(1, 3, figure=fig)

        # Panel 1: latent values (single axis)
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

        # Panel 2: convergence
        ax1 = fig.add_subplot(gs[0, 1])
        self.plot_convergence(ax=ax1)

        # Panel 3: choice probabilities
        ax2 = fig.add_subplot(gs[0, 2])
        probs = np.array(self.choice_probabilities())
        ax2.stackplot(trials, probs.T, labels=option_labels, alpha=0.7)
        ax2.set_xlabel("Trial")
        ax2.set_ylabel("Probability")
        ax2.set_title("Choice Probabilities")
        ax2.legend(fontsize=7)

        fig.tight_layout()
        return fig, np.array([ax0, ax1, ax2])


class SimulatedChoiceData(NamedTuple):
    """Simulated multi-armed bandit data.

    Attributes
    ----------
    choices : Array, shape (n_trials,)
    true_values : Array, shape (n_trials, K-1)
    true_probs : Array, shape (n_trials, K)
    """
    choices: Array
    true_values: Array
    true_probs: Array


def simulate_choice_data(
    n_trials: int = 200,
    n_options: int = 4,
    process_noise: float = 0.05,
    inverse_temperature: float = 2.0,
    seed: int = 42,
) -> SimulatedChoiceData:
    """Simulate multi-armed bandit choice data with evolving values.

    Parameters
    ----------
    n_trials : int
    n_options : int
    process_noise : float
    inverse_temperature : float
    seed : int

    Returns
    -------
    SimulatedChoiceData
    """
    rng = np.random.default_rng(seed)
    k_free = n_options - 1

    # Generate latent values via random walk
    noise = rng.normal(0, np.sqrt(process_noise), (n_trials, k_free))
    true_values = np.cumsum(noise, axis=0)

    # Generate choices from softmax
    full_values = np.column_stack([np.zeros(n_trials), true_values])
    logits = inverse_temperature * full_values
    # Stable softmax
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)

    choices = np.array([
        rng.choice(n_options, p=probs[t]) for t in range(n_trials)
    ])

    return SimulatedChoiceData(
        choices=jnp.array(choices),
        true_values=jnp.array(true_values),
        true_probs=jnp.array(probs),
    )
