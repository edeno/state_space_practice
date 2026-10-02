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
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray

from state_space_practice.exceptions import NotFittedError, StateSpaceWarning
from state_space_practice.kalman import rts_backward_scan
from state_space_practice.parameter_transforms import POSITIVE
from state_space_practice.sgd_fitting import SGDFittableMixin, SGDParams, SGDParamSpec
from state_space_practice.utils import (
    _root_figure,
    psd_cholesky,
    psd_logdet,
    psd_solve,
    symmetrize,
    typed_jit,
)
from state_space_practice.utils import validate_choice_indices as _validate_choices

if TYPE_CHECKING:
    import optax
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Backtracking step sizes tried by every Newton iteration (largest first).
_NEWTON_STEP_SIZES = tuple(0.5**i for i in range(8))
# Armijo sufficient-increase constant for the backtracking line search.
_ARMIJO_C = 1e-4


def _armijo_slack(dtype: DTypeLike) -> float:
    """Relative round-off slack of the Armijo test in ``dtype``.

    ``1e-12`` in float64 and ``16 * eps`` in lower precision, where a
    ``1e-12`` relative change is below round-off.
    """
    return max(1e-12, 16 * float(jnp.finfo(dtype).eps))


# A Laplace mode search whose final iterate is more than this many nats below
# the mode (estimated as half the squared Newton decrement) is reported as
# unconverged.
NEWTON_GAP_TOL = 1e-6


def _warn_unconverged_host(
    n_unconverged: ArrayLike, max_gap: ArrayLike, *, solver: str, stacklevel: int
) -> None:
    """Emit the unconverged-Newton ``StateSpaceWarning`` (host side)."""
    n = int(np.sum(np.asarray(n_unconverged)))
    if n > 0:
        warnings.warn(
            f"{solver}: {n} Laplace mode search(es) stopped before "
            "convergence; the largest Newton estimate of the remaining "
            f"log-posterior gap is {float(np.max(np.asarray(max_gap))):.3g} "
            f"nats (tolerance {NEWTON_GAP_TOL:g}). The posterior mode, "
            "covariance and Laplace log-evidence of those updates are "
            "inaccurate.",
            StateSpaceWarning,
            stacklevel=stacklevel,
        )


def _warn_if_newton_unconverged(newton_gap: ArrayLike, solver: str) -> None:
    """Warn if any Laplace mode search ended short of its mode.

    Emits a :class:`~state_space_practice.exceptions.StateSpaceWarning` when
    any entry of ``newton_gap`` exceeds :data:`NEWTON_GAP_TOL`. NaN gaps come
    from NaN modes, which are visible in the output itself, and are not
    counted. Concrete inputs warn directly; inside ``jax.jit`` /
    ``lax.scan`` / ``jax.grad`` the reduced counts reach the host through one
    :func:`jax.debug.callback` per call.

    Parameters
    ----------
    newton_gap : ArrayLike, any shape
        Estimated log-posterior gap (nats) between each search's final iterate
        and its mode: half the squared Newton decrement
        ``g^T H^{-1} g / 2`` at the final iterate.
    solver : str
        Name of the public function, used in the warning message.
    """
    newton_gap = jnp.asarray(newton_gap)
    unconverged = newton_gap > NEWTON_GAP_TOL
    counts = (
        jnp.sum(unconverged),
        jnp.max(newton_gap, where=unconverged, initial=0.0),
    )
    if isinstance(newton_gap, jax.core.Tracer):
        jax.debug.callback(
            partial(_warn_unconverged_host, solver=solver, stacklevel=2), *counts
        )
    else:
        _warn_unconverged_host(*counts, solver=solver, stacklevel=4)


def _softmax_update_core(
    prior_mean: Array,
    prior_cov: Array,
    choice: Array,
    n_options: int,
    inverse_temperature: float | Array,
    max_newton_steps: int = 10,
    obs_offset: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """JIT-compatible Laplace-EKF update for softmax observation.

    All inputs must be JAX arrays (no Python-level validation).
    Use ``softmax_observation_update`` for the public API with validation.

    The mode of the (concave) log posterior is found by Newton's method with
    a line search: each iteration evaluates the steps ``1, 1/2, ..., 1/128``
    of the Newton direction and, among those passing the Armijo
    sufficient-increase test, takes the one with the highest log posterior
    (the largest one within round-off of it). A full Newton step is not safe
    here: when the prior mean sits on the saturated side of the softmax
    opposite the observed choice, the likelihood curvature is ~0 there and
    the full step overshoots to the other saturated side, after which Newton
    oscillates between the two (e.g. prior N(1.5, 1), beta=4, choice 0:
    iterates 1.5, -2.2, 1.5, ... around the true mode -0.2). Taking merely
    the largest Armijo-acceptable step still zigzags across the mode with
    slowly shrinking amplitude. With the line search every iteration ascends,
    and near the mode the full step is the best one, so well-behaved updates
    are unchanged.

    Parameters
    ----------
    max_newton_steps : int, default 10
        Number of line-searched Newton iterations for Laplace mode-finding:
        a fixed-length ``lax.scan`` with no early exit, so the update stays
        reverse-mode differentiable and the compile cost does not depend on
        it. Ten converge the mode in ordinary use (none of 42,000 simulated
        updates at beta in [0.5, 12] was left unconverged); three leave it
        unconverged at moderately large inverse temperatures (a 2.7-nat
        evidence error at beta=3 on 100 simulated trials). At a converged
        mode extra iterations take a zero-length step; ``newton_gap`` reports
        a mode that was not reached.
    obs_offset : Array or None, shape (K,)
        Additive offset to the logits before softmax. Used for
        observation covariates (e.g., stay bias, spatial bias).
        These shift choice probabilities without changing the
        latent value state. None means no offset.

    Returns
    -------
    posterior_mean : Array, shape (K-1,)
        Final Newton iterate (the posterior mode when converged).
    posterior_cov : Array, shape (K-1, K-1)
    log_likelihood : Array, shape ()
        Laplace evidence ``log p(choice | past)`` at the final iterate.
    newton_gap : Array, shape ()
        Half the squared Newton decrement at the final iterate: the
        estimated log-posterior gap (nats) to the mode, ~0 when converged.
        Pass it to :func:`_warn_if_newton_unconverged`.
    """
    # One floating dtype for every carried and constant array: integer inputs
    # are promoted to the default float, float32 inputs stay float32.
    float_inputs = [prior_mean, prior_cov, inverse_temperature]
    if obs_offset is not None:
        float_inputs.append(obs_offset)
    dtype = jnp.result_type(*float_inputs, 1.0)
    prior_mean = jnp.asarray(prior_mean, dtype=dtype)
    prior_cov = jnp.asarray(prior_cov, dtype=dtype)
    _obs_offset = (
        jnp.zeros(n_options, dtype=dtype)
        if obs_offset is None
        else jnp.asarray(obs_offset, dtype=dtype)
    )
    beta = jnp.asarray(inverse_temperature, dtype=dtype)
    k_free = n_options - 1

    # Precompute constants
    e_k = jnp.zeros(n_options, dtype=dtype).at[choice].set(1.0)
    e_k_free = e_k[1:]
    eye_k = jnp.eye(k_free, dtype=dtype)
    zero_ref = jnp.zeros(1, dtype=dtype)
    beta_sq = beta**2
    step_sizes = jnp.asarray(_NEWTON_STEP_SIZES, dtype=dtype)

    # Prior precision from one Cholesky factor of the prior covariance; the
    # same factor supplies log|P_prior| below, so the evidence sees exactly
    # the (scale-relatively shifted) matrix the update used.
    prior_cho = psd_cholesky(prior_cov)
    prior_precision = symmetrize(jax.scipy.linalg.cho_solve(prior_cho, eye_k))

    def log_posterior(x: Array) -> Array:
        """Unnormalised log posterior at x, shape (..., K-1) -> (...)."""
        zeros = jnp.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
        logits = beta * jnp.concatenate([zeros, x], axis=-1) + _obs_offset
        delta = x - prior_mean
        return jax.nn.log_softmax(logits, axis=-1)[..., choice] - 0.5 * jnp.einsum(
            "...i,ij,...j->...", delta, prior_precision, delta
        )

    def newton_iteration(x: Array, _: None) -> tuple[Array, None]:
        v = jnp.concatenate([zero_ref, x])
        p_free = jax.nn.softmax(beta * v + _obs_offset)[1:]

        gradient = beta * (e_k_free - p_free)
        neg_hessian = beta_sq * (jnp.diag(p_free) - jnp.outer(p_free, p_free))

        posterior_precision = prior_precision + neg_hessian
        rhs = gradient + prior_precision @ (prior_mean - x)  # d log post / dx
        direction = psd_solve(posterior_precision, rhs)

        # Armijo backtracking, evaluated for all candidate steps at once.
        candidates = x + step_sizes[:, None] * direction
        f_x = log_posterior(x)
        f_candidates = log_posterior(candidates)
        # The slack admits steps that change f only at round-off level, so
        # at a converged mode the (tiny) full Newton step is still taken:
        # rejecting it would freeze x, and reverse-mode gradients through
        # the scan would then miss the Newton map's contraction and carry
        # the error of an earlier, unconverged iterate.
        slack = _armijo_slack(dtype) * (1.0 + jnp.abs(f_x))
        sufficient = (
            f_candidates >= f_x + _ARMIJO_C * step_sizes * (rhs @ direction) - slack
        )
        # Of the acceptable steps, the one with the highest log posterior
        # (the largest within round-off of it, so a converged mode still takes
        # the full step). Taking the largest acceptable step instead lets the
        # iterates zigzag across the mode with slowly shrinking amplitude.
        # If no step qualifies (direction numerically useless), keep x rather
        # than risk a descent step.
        f_best = jnp.max(jnp.where(sufficient, f_candidates, -jnp.inf))
        first_ok = jnp.argmax(sufficient & (f_candidates >= f_best - slack))
        alpha = jnp.where(jnp.any(sufficient), step_sizes[first_ok], 0.0)
        return x + alpha * direction, None

    # Fixed number of Newton iterations: a scan (not a while loop) keeps the
    # update reverse-mode differentiable for fit_sgd, and (unlike an unrolled
    # Python loop) its compile cost does not grow with max_newton_steps.
    x, _ = jax.lax.scan(newton_iteration, prior_mean, None, length=max_newton_steps)

    # Final posterior covariance at the mode
    v = jnp.concatenate([zero_ref, x])
    p_free = jax.nn.softmax(beta * v + _obs_offset)[1:]
    neg_hessian = beta_sq * (jnp.diag(p_free) - jnp.outer(p_free, p_free))
    posterior_precision = prior_precision + neg_hessian
    post_cho = psd_cholesky(posterior_precision)
    posterior_cov = symmetrize(jax.scipy.linalg.cho_solve(post_cho, eye_k))
    # Convergence diagnostic: 0.5 g^T H^{-1} g at the final iterate, the
    # log-posterior increase a full Newton step would predict.
    final_gradient = beta * (e_k_free - p_free) + prior_precision @ (prior_mean - x)
    newton_gap = (
        0.5 * final_gradient @ jax.scipy.linalg.cho_solve(post_cho, final_gradient)
    )

    # Laplace-approximated marginal log-likelihood log p(c_t | y_{1:t-1}):
    #   ≈ log p(c_t | x*) + log p(x* | y_{1:t-1}) + ½ log|Σ_post| + const
    # where x* is the posterior mode, and (k/2)log(2π) cancels.
    # See point_process_kalman._stochastic_point_process_filter_step for
    # the same derivation applied to Poisson observations.
    log_lik_at_mode = jax.nn.log_softmax(beta * v + _obs_offset)[choice]
    delta = x - prior_mean
    quad = delta @ (prior_precision @ delta)
    # Both log-determinants come from the factors already computed (no extra
    # absolute jitter), which keeps the evidence invariant to the units of x:
    # log|Sigma_post| = -log|Lambda_post|.
    logdet_prior = psd_logdet(prior_cho)
    logdet_post = -psd_logdet(post_cho)
    log_lik = log_lik_at_mode - 0.5 * quad - 0.5 * logdet_prior + 0.5 * logdet_post

    return x, posterior_cov, log_lik, newton_gap


def softmax_observation_update(
    prior_mean: ArrayLike,
    prior_cov: ArrayLike,
    choice: int,
    n_options: int,
    inverse_temperature: float = 1.0,
    max_newton_steps: int = 10,
) -> tuple[Array, Array, Array]:
    """Laplace-EKF update for a categorical observation with softmax link.

    The latent state x in R^{K-1} represents relative values for options
    1 through K-1. Option 0 is the reference (value fixed at 0).

    Parameters
    ----------
    prior_mean : ArrayLike, shape (K-1,)
        Prior state mean from prediction step.
    prior_cov : ArrayLike, shape (K-1, K-1)
        Prior state covariance from prediction step.
    choice : int
        Observed choice (0-indexed, 0 = reference option).
    n_options : int
        Total number of options K.
    inverse_temperature : float
        Softmax inverse temperature beta.
    max_newton_steps : int, default 10
        Number of line-searched Newton iterations for the Laplace mode. This
        is a fixed-length ``lax.scan`` with no early exit: every call runs
        all of them (at a converged mode the extra iterations take a
        zero-length step). A ``StateSpaceWarning`` is emitted if the final
        iterate is still more than ``NEWTON_GAP_TOL`` nats (Newton estimate)
        below the mode.

    Returns
    -------
    posterior_mean : Array, shape (K-1,)
        Posterior mode.
    posterior_cov : Array, shape (K-1, K-1)
        Inverse negative Hessian of the log posterior at the mode.
    log_likelihood : Array, scalar
        Laplace approximation, at the posterior mode, of the evidence
        ``log p(choice | prior) = log E_{N(prior_mean, prior_cov)}[p(choice
        | x)]`` (for EM monitoring).

    Warns
    -----
    StateSpaceWarning
        If the Newton search stops short of the mode (see
        ``max_newton_steps``).
    """
    if choice < 0 or choice >= n_options:
        raise ValueError(f"choice must be in [0, {n_options}), got {choice}")
    prior_mean = jnp.asarray(prior_mean)
    prior_cov = jnp.asarray(prior_cov)
    post_mean, post_cov, log_lik, newton_gap = _softmax_update_core(
        prior_mean,
        prior_cov,
        jnp.int32(choice),
        n_options,
        inverse_temperature,
        max_newton_steps,
    )
    _warn_if_newton_unconverged(newton_gap, "softmax_observation_update")
    return post_mean, post_cov, log_lik


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

    Warns
    -----
    StateSpaceWarning
        If a Laplace mode search ends more than ``NEWTON_GAP_TOL`` nats
        (Newton estimate) below its mode.
    """
    _validate_choices(choices, n_options)
    choices_arr = jnp.asarray(choices, dtype=jnp.int32)
    k_free = n_options - 1

    # Resolve defaults before JIT boundary, in one floating dtype so the scan
    # carry keeps its type: integer inputs are promoted to the default float,
    # float32 inputs stay float32.
    float_inputs = [
        jnp.asarray(x)
        for x in (init_mean, init_cov, process_noise, inverse_temperature)
        if x is not None
    ]
    dtype = jnp.result_type(*float_inputs, 1.0)
    if init_mean is None:
        init_mean = jnp.zeros(k_free, dtype=dtype)
    else:
        init_mean = jnp.asarray(init_mean, dtype=dtype)
    if init_cov is None:
        init_cov = jnp.eye(k_free, dtype=dtype)
    else:
        init_cov = jnp.asarray(init_cov, dtype=dtype)

    return _multinomial_choice_filter_jit(
        choices_arr,
        n_options,
        process_noise,
        inverse_temperature,
        init_mean,
        init_cov,
    )


@partial(typed_jit, static_argnames=("n_options",))
def _multinomial_choice_filter_jit(
    choices: Array,
    n_options: int,
    process_noise: float | Array,
    inverse_temperature: float | Array,
    init_mean: Array,
    init_cov: Array,
) -> ChoiceFilterResult:
    """JIT-compiled filter core."""
    k_free = n_options - 1
    Q = jnp.eye(k_free, dtype=init_mean.dtype) * process_noise

    def _step(
        carry: tuple[Array, Array, Array], choice_t: Array
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array, Array, Array]]:
        filt_mean, filt_cov, total_ll = carry

        # Predict (random walk: A = I)
        pred_mean = filt_mean
        pred_cov = filt_cov + Q

        # Update
        post_mean, post_cov, ll, newton_gap = _softmax_update_core(
            pred_mean,
            pred_cov,
            choice_t,
            n_options,
            inverse_temperature,
        )

        total_ll = total_ll + ll
        return (post_mean, post_cov, total_ll), (
            post_mean,
            post_cov,
            pred_mean,
            pred_cov,
            newton_gap,
        )

    init_carry = (init_mean, init_cov, jnp.zeros((), dtype=init_mean.dtype))
    (_, _, marginal_ll), (filt_vals, filt_covs, pred_vals, pred_covs, newton_gaps) = (
        jax.lax.scan(_step, init_carry, choices)
    )
    _warn_if_newton_unconverged(newton_gaps, "multinomial_choice_filter")

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
        choices,
        n_options,
        process_noise,
        inverse_temperature,
        init_mean,
        init_cov,
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

    def _check_fitted(self, method: str) -> ChoiceSmootherResult:
        if self._smoother_result is None:
            raise NotFittedError(
                f"{type(self).__name__}.{method}() called before fitting. "
                f"Call model.fit(choices) first."
            )
        return self._smoother_result

    @property
    def smoothed_values(self) -> Array:
        """Smoothed option values, shape (n_trials, K-1)."""
        res = self._check_fitted("smoothed_values")
        return res.smoothed_values

    @property
    def smoothed_covariances(self) -> Array:
        """Smoothed covariances, shape (n_trials, K-1, K-1)."""
        res = self._check_fitted("smoothed_covariances")
        return res.smoothed_covariances

    # --- Hooks: which filter to run, with which parameters ---

    def _filter_kwargs(self) -> dict[str, Any]:
        """Keyword arguments selecting the current parameters for the filter.

        ``_run_filter`` and ``_run_smoother`` pass these on, after ``choices``
        and ``n_options``, to the filter / smoother functions they call.
        """
        return {
            "process_noise": self.process_noise,
            "inverse_temperature": self.inverse_temperature,
        }

    def _run_filter(self, choices: Array, **overrides: Any) -> ChoiceFilterResult:
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

    def _restore_parameters(self, snapshot: dict[str, Any]) -> None:
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
        last_accepted: dict[str, Any] | None,
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
        smoother = self._check_fitted("_populate_uncertainty")

        # Option values (full K with reference option appended)
        self.predicted_option_values_ = append_reference_option(filt.predicted_values)
        self.filtered_option_values_ = append_reference_option(filt.filtered_values)
        self.smoothed_option_values_ = append_reference_option(smoother.smoothed_values)

        # Option variances (full K options)
        self.predicted_option_variances_ = option_variances_from_covariances(
            filt.predicted_covariances
        )
        self.filtered_option_variances_ = option_variances_from_covariances(
            filt.filtered_covariances
        )
        self.smoothed_option_variances_ = option_variances_from_covariances(
            smoother.smoothed_covariances
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
        last_accepted: dict[str, Any] | None = None

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
        optimizer: optax.GradientTransformation | None = None,
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
        optimizer: optax.GradientTransformation | None,
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
        if self._n_trials is None:
            raise NotFittedError("Model must be fitted before accessing _n_timesteps.")
        return self._n_trials

    def _build_param_spec(self) -> tuple[SGDParams, SGDParamSpec]:
        params: SGDParams = {}
        spec: SGDParamSpec = {}
        if self.learn_process_noise:
            params["process_noise"] = jnp.array(self.process_noise)
            spec["process_noise"] = POSITIVE
        if self.learn_inverse_temperature:
            params["inverse_temperature"] = jnp.array(self.inverse_temperature)
            spec["inverse_temperature"] = POSITIVE
        return params, spec

    def _sgd_loss_fn(self, params: SGDParams, choices: Array) -> Array:
        # Read the model attribute only when the parameter is not optimized
        # (not ``params.get(key, self.attr)``, which reads it regardless):
        # fit_sgd reuses a compiled step only while the attributes the loss
        # read at trace time are unchanged, and these are rewritten by every
        # fit.
        def _param(key: str) -> Array:
            return params[key] if key in params else jnp.array(getattr(self, key))

        k_free = self.n_options - 1
        result = _multinomial_choice_filter_jit(
            choices,
            self.n_options,
            _param("process_noise"),
            _param("inverse_temperature"),
            jnp.zeros(k_free),
            jnp.eye(k_free),
        )
        return -result.marginal_log_likelihood

    def _store_sgd_params(self, params: SGDParams) -> None:
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
        m = smooth.smoothed_values  # (T, K-1)
        P = smooth.smoothed_covariances  # (T, K-1, K-1)
        C = smooth.smoother_cross_cov  # (T-1, K-1, K-1)
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
        """M-step: grid search + golden-section refinement for beta.

        Maximises the filter's marginal log-likelihood over beta (at the
        current process noise). Returns the best of the refined value, the
        best grid point and the current beta, so the marginal LL never
        decreases.
        """

        def _eval_beta(beta: Array) -> Array:
            result = self._run_filter(choices, inverse_temperature=beta)
            return result.marginal_log_likelihood

        # One batched filter pass evaluates the LL at every candidate beta.
        _eval_betas = jax.vmap(_eval_beta)

        # Coarse grid search
        lls = _eval_betas(beta_grid)

        best_idx = int(jnp.argmax(lls))
        best_beta = float(beta_grid[best_idx])

        # Golden-section refinement over the bracket around the best grid
        # point ([g0, g1] at a grid edge).
        lo_idx = max(0, best_idx - 1)
        hi_idx = min(len(beta_grid) - 1, best_idx + 1)
        lo = float(beta_grid[lo_idx])
        hi = float(beta_grid[hi_idx])
        current = float(self.inverse_temperature)
        candidates = [(float(lls[best_idx]), best_beta)]

        # The bracket is empty only for a one-point grid: nothing to refine,
        # but the grid point must still be compared with the current beta.
        if hi - lo < 1e-10:
            ll_current = float(_eval_betas(jnp.array([current]))[0])
            candidates.append((ll_current, current))
            return max(candidates, key=lambda pair: pair[0])[1]

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

        # The bracket midpoint can be worse than the best grid point (e.g. a
        # maximum at the edge of the grid, which golden section cannot
        # return) or than the current beta: keep whichever of the three has
        # the highest marginal LL, so the M-step never decreases it.
        refined = (lo + hi) / 2
        ll_refined, ll_current = np.asarray(_eval_betas(jnp.array([refined, current])))
        candidates += [(float(ll_refined), refined), (float(ll_current), current)]
        return max(candidates, key=lambda pair: pair[0])[1]

    def choice_probabilities(self) -> Array:
        """Softmax choice probabilities from smoothed values.

        Includes observation-covariate logit offsets when the model has them.

        Returns
        -------
        probs : Array, shape (n_trials, K)
            Each row sums to 1.
        """
        res = self._check_fitted("choice_probabilities")
        # Build full value vectors: [0, x_t] for each trial
        zeros = jnp.zeros((res.smoothed_values.shape[0], 1))
        full_values = jnp.concatenate([zeros, res.smoothed_values], axis=1)
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
        assert self.log_likelihood_ is not None
        assert self._n_trials is not None
        return -2.0 * self.log_likelihood_ + self.n_free_params * math.log(
            self._n_trials
        )

    def compare_to_null(self) -> dict[str, Any]:
        """Compare fitted model to a null (uniform 1/K) model.

        Returns
        -------
        dict with keys: model_ll, null_ll, model_bic, null_bic,
        delta_bic, learning_detected.
        """
        self._check_fitted("compare_to_null")
        assert self._n_trials is not None
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

    def _resolve_option_labels(self, option_labels: list[str] | None) -> list[str]:
        if option_labels is None:
            return [f"Option {i}" for i in range(self.n_options)]
        return option_labels

    def _plot_smoothed_values(
        self,
        ax: Axes,
        option_labels: list[str],
        title: str,
        legend_fontsize: float,
    ) -> None:
        """Draw smoothed relative values with 95% bands on ``ax``."""
        res = self._check_fitted("_plot_smoothed_values")
        vals = np.array(res.smoothed_values)
        covs = np.array(res.smoothed_covariances)
        trials = np.arange(vals.shape[0])
        for k in range(self.n_options - 1):
            std = np.sqrt(covs[:, k, k])
            ax.plot(trials, vals[:, k], label=option_labels[k + 1])
            ax.fill_between(
                trials,
                vals[:, k] - 1.96 * std,
                vals[:, k] + 1.96 * std,
                alpha=0.2,
            )
        ax.axhline(
            0,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"{option_labels[0]} (ref)",
        )
        ax.set_ylabel("Relative value")
        ax.set_title(title)
        ax.legend(fontsize=legend_fontsize)

    def plot_values(
        self,
        observed_choices: ArrayLike | None = None,
        option_labels: list[str] | None = None,
        ax: NDArray[np.object_] | None = None,
    ) -> tuple[Figure, NDArray[np.object_]]:
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
            fig = _root_figure(axes[0])

        # Top: latent values with CI
        self._plot_smoothed_values(axes[0], option_labels, "Smoothed Option Values", 8)

        # Bottom: choice probabilities (stacked area)
        axes[1].stackplot(trials, probs.T, labels=option_labels, alpha=0.7)
        if observed_choices is not None:
            # Mark observed choices as tick marks along the top
            choices_np = np.asarray(observed_choices)
            for k in range(self.n_options):
                chosen_trials = trials[choices_np == k]
                if len(chosen_trials) > 0:
                    axes[1].eventplot(
                        chosen_trials,
                        lineoffsets=1.02 - k * 0.03,
                        linelengths=0.02,
                        colors="k",
                        alpha=0.4,
                    )
        axes[1].set_ylabel("Choice probability")
        axes[1].set_xlabel("Trial")
        axes[1].set_title("Choice Probabilities")
        axes[1].legend(fontsize=8, loc="upper right")

        fig.tight_layout()
        return fig, axes

    def plot_convergence(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """Plot EM log-likelihood convergence.

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt

        self._check_fitted("plot_convergence")
        assert self.log_likelihood_history_ is not None

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = _root_figure(ax)

        ax.plot(
            range(1, len(self.log_likelihood_history_) + 1),
            self.log_likelihood_history_,
            "o-",
        )
        ax.set_xlabel("EM Iteration")
        ax.set_ylabel("Log-Likelihood")
        ax.set_title("EM Convergence")

        fig.tight_layout()
        return fig, ax

    def plot_summary(
        self,
        observed_choices: ArrayLike | None = None,
        option_labels: list[str] | None = None,
    ) -> tuple[Figure, NDArray[np.object_]]:
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

    choices = np.array([rng.choice(n_options, p=probs[t]) for t in range(n_trials)])

    return SimulatedChoiceData(
        choices=jnp.array(choices),
        true_values=jnp.array(true_values),
        true_probs=jnp.array(probs),
    )
