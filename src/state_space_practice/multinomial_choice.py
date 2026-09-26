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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import rts_backward_scan
from state_space_practice.parameter_transforms import POSITIVE
from state_space_practice.point_process_kalman import _logdet_psd
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import psd_cholesky, psd_solve, symmetrize
from state_space_practice.utils import validate_choice_indices as _validate_choices


def _softmax_update_core(
    prior_mean: Array,
    prior_cov: Array,
    choice: Array,
    n_options: int,
    inverse_temperature: float,
    max_newton_steps: int = 3,
    obs_offset: Array | None = None,
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
    post_cho = psd_cholesky(posterior_precision)
    posterior_cov = symmetrize(jax.scipy.linalg.cho_solve(post_cho, eye_k))

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
    init_mean: ArrayLike | None = None,
    init_cov: ArrayLike | None = None,
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

    Raises
    ------
    ValueError
        If any entry of ``choices`` is outside ``[0, n_options)``.
    """
    _validate_choices(choices, n_options)
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


def multinomial_choice_smoother(
    choices: ArrayLike,
    n_options: int,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    init_mean: ArrayLike | None = None,
    init_cov: ArrayLike | None = None,
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

    # Random-walk latent (identity transition); the shared RTS scan appends the
    # last filtered state itself (smoother[-1] == filter[-1]).
    smoothed_values, smoothed_covs, cross_covs = rts_backward_scan(
        filt.filtered_values, filt.filtered_covariances, jnp.eye(k_free), Q
    )

    return ChoiceSmootherResult(
        smoothed_values=smoothed_values,
        smoothed_covariances=smoothed_covs,
        smoother_cross_cov=cross_covs,
        marginal_log_likelihood=filt.marginal_log_likelihood,
    )


_DEFAULT_BETA_GRID = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)

# Tolerance below which a marginal-LL decrease between EM iterations is treated
# as an overshoot of the approximate (Laplace-EKF) M-step and triggers a rollback
# to the last accepted parameters.
_EM_MONOTONICITY_TOL = 1e-6


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

    Notes
    -----
    ``CovariateChoiceModel`` (``covariate_choice.py``) subclasses this model.
    The EM driver, the beta / process-noise M-steps, the SGD protocol, the
    uncertainty summaries, BIC and the plots live here; a subclass changes
    only what genuinely differs, through these hooks:

    - ``_run_filter()`` / ``_run_smoother()`` (which filter and smoother to
      run) and ``_filter_kwargs()`` (the keyword arguments selecting the
      current parameters that both pass on).
    - ``_transition_decay()`` / ``_control_input()``: the latent dynamics
      ``x_t = a x_{t-1} + b_t + w_t`` seen by the process-noise M-step.
    - ``_observation_logit_offsets()``: additive logit offsets applied to
      choice probabilities, predicted entropy and surprise.
    - ``_m_step()``, ``_em_parameter_names()``, ``_em_progress()`` and
      ``_final_e_step()``: the per-iteration parameter updates, the
      attributes the monotonicity guard snapshots, the verbose log line
      and the final E-step policy.
    - ``n_free_params`` and ``_summary_dimension_rows()``: parameter
      counting for BIC and the leading rows of ``summary()``.
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
        if init_inverse_temperature <= 0:
            raise ValueError(
                f"init_inverse_temperature must be > 0 (a non-positive value "
                f"inverts choice preferences), got {init_inverse_temperature}."
            )
        if init_process_noise < 0:
            raise ValueError(
                f"init_process_noise must be non-negative (Q = process_noise * I "
                f"must be PSD), got {init_process_noise}."
            )
        self.n_options = n_options
        self.inverse_temperature = init_inverse_temperature
        self.process_noise = init_process_noise
        self.learn_inverse_temperature = learn_inverse_temperature
        self.learn_process_noise = learn_process_noise

        # Fitted state (populated by fit())
        self._smoother_result: ChoiceSmootherResult | None = None
        self.log_likelihood_: float | None = None
        self.n_iter_: int | None = None
        self.converged_: bool | None = None
        self.log_likelihood_history_: list[float] | None = None
        self._n_trials: int | None = None

        # Uncertainty summaries (populated after fitting)
        self.predicted_option_variances_: Array | None = None
        self.smoothed_option_variances_: Array | None = None
        self.predicted_choice_entropy_: Array | None = None
        self.surprise_: Array | None = None

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

    def _check_fitted(self, method: str) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.{method}() called before fitting. "
                f"Call model.fit(choices) first."
            )

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

    # --- Hooks: which filter to run, with which parameters ---

    def _filter_kwargs(self) -> dict:
        """Keyword arguments selecting the current parameters for the filter.

        ``_run_filter`` and ``_run_smoother`` pass these on, after ``choices``
        and ``n_options``, to the filter / smoother functions they call.
        """
        return {
            "process_noise": self.process_noise,
            "inverse_temperature": self.inverse_temperature,
        }

    def _run_filter(self, choices: Array, **overrides) -> ChoiceFilterResult:
        """Forward filter at the current parameters; ``overrides`` replace any."""
        kwargs = {**self._filter_kwargs(), **overrides}
        return multinomial_choice_filter(choices, self.n_options, **kwargs)

    def _run_smoother(self, choices: Array) -> ChoiceSmootherResult:
        """Forward filter + RTS smoother at the current parameters."""
        return multinomial_choice_smoother(
            choices, self.n_options, **self._filter_kwargs()
        )

    # --- Hooks: latent dynamics and observation model ---

    def _transition_decay(self) -> float:
        """Scalar ``a`` of the dynamics ``x_t = a x_{t-1} + ...`` (1 = random walk)."""
        return 1.0

    def _control_input(self) -> Array | None:
        """Known input ``b_t`` of each transition t-1 -> t, (T-1, K-1), or None."""
        return None

    def _observation_logit_offsets(self) -> Array | None:
        """Additive per-trial logit offsets, shape (n_trials, K), or None."""
        return None

    # --- Hooks: EM loop ---

    def _em_parameter_names(self) -> tuple[str, ...]:
        """Attributes the EM monotonicity guard snapshots before each M-step."""
        return ("process_noise", "inverse_temperature")

    def _em_progress(self) -> str:
        """Current-parameter summary for the verbose per-iteration log line."""
        return f"beta={self.inverse_temperature:.3f}, Q={self.process_noise:.6f}"

    def _restore_parameters(self, snapshot: dict) -> None:
        for attr, value in snapshot.items():
            setattr(self, attr, value)

    def _m_step(
        self, smooth: ChoiceSmootherResult, choices: Array, beta_grid: Array
    ) -> None:
        """One EM M-step: update every learned parameter in place."""
        # M-step for process noise Q
        if self.learn_process_noise:
            self.process_noise = self._m_step_process_noise(smooth)

        # M-step for inverse temperature beta
        if self.learn_inverse_temperature:
            self.inverse_temperature = self._m_step_beta(choices, beta_grid)

    def _final_e_step(
        self,
        choices: Array,
        log_likelihoods: list[float],
        last_accepted: dict | None,
    ) -> float:
        """Sync ``_smoother_result`` / ``log_likelihood_`` to the final parameters.

        Runs once after the EM loop and returns the final log-likelihood.
        ``log_likelihoods`` is the per-iteration history and ``last_accepted``
        the parameter snapshot taken before the last M-step (None if no M-step
        ran); subclasses may use them to amend the history or roll a degrading
        final M-step back.
        """
        self._smoother_result = self._run_smoother(choices)
        self.log_likelihood_ = float(self._smoother_result.marginal_log_likelihood)
        return self.log_likelihood_

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

        filt = self._run_filter(choices)

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

        # Predicted choice entropy (including any observation-covariate offsets)
        logits = self.inverse_temperature * self.predicted_option_values_
        obs_offsets = self._observation_logit_offsets()
        if obs_offsets is not None:
            logits = logits + obs_offsets
        pred_probs = jax.nn.softmax(logits, axis=1)
        self.predicted_choice_entropy_ = categorical_entropy(pred_probs)

        # Surprise
        self.surprise_ = compute_surprise(pred_probs, choices)

    def _prepare_choices(self, choices: ArrayLike, method: str) -> Array:
        """Validate ``choices`` for fitting, record ``_n_trials``, return int32.

        The single validation point for ``fit`` / ``fit_sgd`` (subclasses
        route through it too). Nothing is recorded unless every check passes.

        Raises
        ------
        ValueError
            If ``choices`` is not 1-D, has fewer than 2 trials, or contains
            indices outside ``[0, n_options)``.
        """
        choices_np = np.asarray(choices)
        if choices_np.ndim != 1:
            raise ValueError(
                f"choices must be a 1-D array with one entry per trial, got "
                f"shape {choices_np.shape}."
            )
        n_trials = int(choices_np.shape[0])
        if n_trials < 2:
            raise ValueError(
                f"Need at least 2 trials for {method} fitting, got {n_trials}"
            )

        _validate_choices(choices_np, self.n_options)
        self._n_trials = n_trials
        return jnp.asarray(choices_np, dtype=jnp.int32)

    def fit(
        self,
        choices: ArrayLike,
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
        max_iter : int
            Maximum EM iterations.
        tolerance : float
            Convergence tolerance on the relative log-likelihood change
            ``|LL_k - LL_{k-1}| / |LL_{k-1}| < tolerance`` (the absolute change
            ``|LL_k - LL_{k-1}|`` when ``LL_{k-1} == 0``). This normalizes by the
            previous LL only, not by the average of the two as
            :func:`state_space_practice.utils.check_converged` does.
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
        choices_arr = self._prepare_choices(choices, "EM")
        return self._fit_em(choices_arr, max_iter, tolerance, verbose, beta_grid)

    def _fit_em(
        self,
        choices_arr: Array,
        max_iter: int,
        tolerance: float,
        verbose: bool,
        beta_grid: ArrayLike | None,
    ) -> list[float]:
        """EM loop on choices already validated by ``_prepare_choices``."""
        if beta_grid is None:
            beta_grid = jnp.array(_DEFAULT_BETA_GRID)
        else:
            beta_grid = jnp.asarray(beta_grid)

        # Subclasses log under their own module (as _finalize_convergence does).
        log = logging.getLogger(type(self).__module__)
        log_likelihoods: list[float] = []
        converged = False
        last_accepted: dict | None = None

        for iteration in range(max_iter):
            # E-step: run smoother with current parameters
            smooth = self._run_smoother(choices_arr)
            ll = float(smooth.marginal_log_likelihood)

            # GEM monotonicity guard: the approximate (Laplace-EKF) M-step can
            # decrease the marginal LL. On a decrease beyond tolerance, restore the
            # last accepted parameters and stop, so fit() returns the best iterate
            # and the LL history stays non-decreasing.
            if (
                log_likelihoods
                and last_accepted is not None
                and ll < log_likelihoods[-1] - _EM_MONOTONICITY_TOL
            ):
                self._restore_parameters(last_accepted)
                break

            log_likelihoods.append(ll)

            if verbose:
                log.info(
                    "EM iter %d: LL=%.2f, %s", iteration + 1, ll, self._em_progress()
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
                        log.info("Converged at iteration %d", iteration + 1)
                    converged = True
                    break

            # Snapshot the parameters this accepted E-step used, so the
            # monotonicity guard above can roll back a degrading M-step.
            last_accepted = {
                name: getattr(self, name) for name in self._em_parameter_names()
            }

            self._m_step(smooth, choices_arr, beta_grid)

        # Final E-step with learned parameters
        self._final_e_step(choices_arr, log_likelihoods, last_accepted)
        self.n_iter_ = len(log_likelihoods)
        self.log_likelihood_history_ = log_likelihoods
        self._populate_uncertainty(choices_arr)
        self._finalize_convergence(converged, max_iter)

        return log_likelihoods

    def fit_sgd(
        self,
        choices: ArrayLike,
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
        choices_arr = self._prepare_choices(choices, "SGD")
        return self._fit_sgd_validated(
            choices_arr, optimizer, num_steps, verbose, convergence_tol
        )

    def _fit_sgd_validated(
        self,
        choices_arr: Array,
        optimizer: object | None,
        num_steps: int,
        verbose: bool,
        convergence_tol: float | None,
    ) -> list[float]:
        """SGD fit on choices already validated by ``_prepare_choices``."""
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

    def _build_param_spec(self) -> tuple[dict, dict]:
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
        # Read the model attribute only when the parameter is not optimized
        # (not ``params.get(key, self.attr)``, which reads it regardless):
        # fit_sgd reuses a compiled step only while the attributes the loss
        # read at trace time are unchanged, and these are rewritten by every
        # fit.
        k_free = self.n_options - 1
        process_noise = (
            params["process_noise"]
            if "process_noise" in params
            else jnp.array(self.process_noise)
        )
        inverse_temperature = (
            params["inverse_temperature"]
            if "inverse_temperature" in params
            else jnp.array(self.inverse_temperature)
        )
        result = _multinomial_choice_filter_jit(
            choices, self.n_options,
            process_noise,
            inverse_temperature,
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
        self._smoother_result = self._run_smoother(choices)
        self.log_likelihood_ = float(self._smoother_result.marginal_log_likelihood)
        self._populate_uncertainty(choices)

    # --- M-steps ---

    def _m_step_process_noise(self, smooth: ChoiceSmootherResult) -> float:
        """M-step: update scalar process noise from smoother statistics.

        For dynamics x_t = a x_{t-1} + b_t + w_t with a = ``_transition_decay()``
        and known input b_t = ``_control_input()`` (the random walk a = 1,
        b_t = 0 for this model), the EM update is
            Q_hat = (1/(T-1)) * sum_{t=1}^{T-1} [
                r_t r_t' + P_t + a^2 P_{t-1} - 2 a C_{t-1,t}
            ],    r_t = m_t - a m_{t-1} - b_t,
        where C_{t-1,t} = Cov(x_{t-1}, x_t | y_{1:T}) from the smoother.
        Convention: smoother_cross_cov[t] = Cov(x_t, x_{t+1} | y_{1:T}),
        which pairs with diff[t] = m[t+1] - a * m[t].
        """
        m = smooth.smoothed_values       # (T, K-1)
        P = smooth.smoothed_covariances  # (T, K-1, K-1)
        C = smooth.smoother_cross_cov    # (T-1, K-1, K-1)
        a = self._transition_decay()

        T_minus_1 = m.shape[0] - 1
        diff = m[1:] - a * m[:-1]  # (T-1, K-1)

        # Subtract the known control input if present
        control_input = self._control_input()
        if control_input is not None:
            diff = diff - control_input

        Q_hat = (
            jnp.einsum("ti,tj->ij", diff, diff)
            + jnp.sum(P[1:], axis=0)
            + a**2 * jnp.sum(P[:-1], axis=0)
            - 2 * a * jnp.sum(C, axis=0)
        ) / T_minus_1
        # Scalar Q: mean of diagonal, clamped
        return float(jnp.maximum(jnp.mean(jnp.diag(Q_hat)), 1e-8))

    def _m_step_beta(
        self,
        choices: Array,
        beta_grid: Array,
    ) -> float:
        """M-step: grid search + golden-section refinement for beta."""
        def _eval_beta(beta):
            result = self._run_filter(choices, inverse_temperature=beta)
            return result.marginal_log_likelihood

        # One batched filter pass evaluates the LL at every candidate beta.
        _eval_betas = jax.vmap(_eval_beta)

        # Coarse grid search
        lls = _eval_betas(beta_grid)

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
            # Both candidates in one batched pass: a single device sync.
            ll_c, ll_d = np.asarray(_eval_betas(jnp.array([c, d])))
            if ll_c > ll_d:
                hi = d
            else:
                lo = c

        return (lo + hi) / 2

    def choice_probabilities(self) -> Array:
        """Softmax choice probabilities from smoothed values.

        Includes observation-covariate logit offsets when the model has them.

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
        logits = self.inverse_temperature * full_values
        obs_offsets = self._observation_logit_offsets()
        if obs_offsets is not None:
            logits = logits + obs_offsets
        return jax.nn.softmax(logits, axis=1)

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

    def _summary_dimension_rows(self) -> list[tuple[str, object]]:
        """``(label, value)`` rows describing the model's size, first in summary()."""
        return [("n_options", self.n_options)]

    def summary(self) -> str:
        """Text summary of fitted model including null comparison."""
        self._check_fitted("summary")
        comparison = self.compare_to_null()
        rows = self._summary_dimension_rows() + [
            ("inverse_temperature", f"{self.inverse_temperature:.4f}"),
            ("process_noise", f"{self.process_noise:.6f}"),
            ("n_trials", self._n_trials),
            ("n_em_iterations", self.n_iter_),
            ("log_likelihood", f"{self.log_likelihood_:.2f}"),
            ("BIC", f"{self.bic():.2f}"),
            ("null_ll (uniform)", f"{comparison['null_ll']:.2f}"),
            ("delta_BIC", f"{comparison['delta_bic']:.2f}"),
            ("learning_detected", comparison["learning_detected"]),
        ]
        lines = [f"{type(self).__name__} Summary", "=" * 40]
        lines += [f"  {label + ':':<23}{value}" for label, value in rows]
        return "\n".join(lines)

    # --- Plotting ---

    def _resolve_option_labels(self, option_labels):
        if option_labels is None:
            return [f"Option {i}" for i in range(self.n_options)]
        return option_labels

    def _plot_smoothed_values(self, ax, option_labels, title, legend_fontsize):
        """Draw smoothed relative values with 95% bands on ``ax``."""
        vals = np.array(self._smoother_result.smoothed_values)
        covs = np.array(self._smoother_result.smoothed_covariances)
        trials = np.arange(vals.shape[0])
        for k in range(self.n_options - 1):
            std = np.sqrt(covs[:, k, k])
            ax.plot(trials, vals[:, k], label=option_labels[k + 1])
            ax.fill_between(
                trials, vals[:, k] - 1.96 * std, vals[:, k] + 1.96 * std,
                alpha=0.2,
            )
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5,
                   label=f"{option_labels[0]} (ref)")
        ax.set_ylabel("Relative value")
        ax.set_title(title)
        ax.legend(fontsize=legend_fontsize)

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

        option_labels = self._resolve_option_labels(option_labels)
        probs = np.array(self.choice_probabilities())
        trials = np.arange(probs.shape[0])

        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        # Top: latent values with CI
        self._plot_smoothed_values(
            axes[0], option_labels, "Smoothed Option Values", 8
        )

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

        option_labels = self._resolve_option_labels(option_labels)

        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(1, 3, figure=fig)

        # Panel 1: latent values (single axis)
        ax0 = fig.add_subplot(gs[0, 0])
        self._plot_smoothed_values(ax0, option_labels, "Smoothed Values", 7)

        # Panel 2: convergence
        ax1 = fig.add_subplot(gs[0, 1])
        self.plot_convergence(ax=ax1)

        # Panel 3: choice probabilities
        ax2 = fig.add_subplot(gs[0, 2])
        probs = np.array(self.choice_probabilities())
        trials = np.arange(probs.shape[0])
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
