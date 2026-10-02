r"""Bayesian State-Space Model for Learning using Laplace Approximation.

This module implements a Bayesian filter and smoother designed to track a
latent learning state over trials, based on binomial (correct/incorrect)
observation data. It uses the approach described by:

    Smith, A. C., Frank, L. M., Wirth, S., Yanike, M., Hu, D., Kubota, Y.,
    Graybiel, A. M., Suzuki, W. A., & Brown, E. N. (2004).
    Dynamic analysis of learning in behavioral experiments.
    Journal of Neuroscience, 24(2), 447-461.

The model assumes:
1.  **Latent Learning State (x_k)**: Follows a Gaussian random walk, representing
    the underlying ability or knowledge at trial 'k'.
    $$
    x_k = x_{k-1} + w_k, \quad w_k \sim N(0, \sigma_\epsilon^2)
    $$
2.  **Observation Model**: The probability of a correct response ($p_k$) is
    linked to the latent state via a sigmoid (logit) function. The number
    of correct responses ($y_k$) in a trial follows a Binomial distribution.
    $$
    p_k = \frac{1}{1 + \exp(-(\mu + x_k))}
    $$
    $$
    y_k \sim \text{Binomial}(N_k, p_k)
    $$
    where $\mu$ is a bias term (often related to chance performance) and
    $N_k$ is the maximum possible correct responses in trial 'k'.

Due to the non-linear sigmoid link and non-Gaussian Binomial likelihood, the
posterior distribution is not Gaussian, and the standard Kalman filter cannot
be applied directly. This implementation addresses this by using the
**Laplace approximation** within the filter's update step. It approximates
the posterior at each step with a Gaussian distribution by finding its mode
and calculating the curvature (Hessian) at the mode.

The module provides:
- `_approximate_gaussian_newton`: The Laplace approximation used by the
  filter (fixed line-searched Newton steps; reverse-mode differentiable).
- `approximate_gaussian`: A general (multi-dimensional) BFGS-based Laplace
  approximation; not used by the filter and not reverse-mode differentiable.
- `_log_posterior_objective`: Defines the log-posterior for a single step.
- `smith_learning_filter`: Implements the forward-pass filter.
- `smith_learning_smoother`: Implements a backward-pass RTS smoother.

This implementation leverages JAX for automatic differentiation (gradient and
Hessian of the log posterior for the Newton mode search) and efficient
vectorized/scanned operations.

"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import optax
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np
import scipy.special
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.em_driver import (
    clear_attributes,
    restore_attributes,
    run_em,
    snapshot_attributes,
)
from state_space_practice.exceptions import NotFittedError, StateSpaceWarning
from state_space_practice.fitted_state import FittedAttribute, is_set
from state_space_practice.multinomial_choice import (
    _ARMIJO_C,
    _NEWTON_STEP_SIZES,
    _armijo_slack,
    _warn_if_newton_unconverged,
)
from state_space_practice.parameter_transforms import (
    POSITIVE,
    UNCONSTRAINED,
)
from state_space_practice.sgd_fitting import SGDFittableMixin, SGDParams, SGDParamSpec
from state_space_practice.utils import typed_jit, validate_count_array

logger = logging.getLogger(__name__)


class SmithFilterResult(NamedTuple):
    """Output of :func:`smith_learning_filter` (each shape ``(n_trials,)``)."""

    prob_correct_response: Array
    learning_state_mode: Array
    learning_state_variance: Array
    one_step_mode: Array
    one_step_variance: Array


# Default process noise standard deviation
# Smith et al. (2004) used variance of 0.05, so sigma = sqrt(0.05)
DEFAULT_PROCESS_NOISE_VARIANCE: float = 0.05
DEFAULT_SIGMA_EPSILON: float = math.sqrt(DEFAULT_PROCESS_NOISE_VARIANCE)


def approximate_gaussian(
    log_posterior_func: Callable[[ArrayLike], Array], x0: ArrayLike
) -> tuple[Array, Array]:
    """Approximate the posterior using Laplace approximation (BFGS).

    Finds the mode and covariance matrix using the Hessian of the
    negative log posterior. Adds regularization to the Hessian before
    inversion for numerical stability.

    This function uses ``jax.scipy.optimize.minimize(method="BFGS")``
    internally. It is **not** reverse-mode differentiable: ``jax.grad``
    cannot propagate through the BFGS solver, and jax.scipy's BFGS can stop
    early on a line-search failure. ``smith_learning_filter`` therefore
    uses the line-searched Newton solver ``_approximate_gaussian_newton``
    instead; this function is kept for general (multi-dimensional) use.

    Parameters
    ----------
    log_posterior_func : Callable
        Function computing the log posterior distribution.
        Takes one argument (state) and returns a scalar.
    x0 : ArrayLike, shape (1,)
        Initial guess for the mode.

    Returns
    -------
    mode : Array, shape (1,)
        The mode of the posterior distribution.
    covariance : Array, shape (1, 1)
        The covariance matrix (approximated) of the posterior distribution.
    """

    def neg_log_posterior(x: ArrayLike) -> Array:
        result: Array = -log_posterior_func(x)
        return result

    # Find the mode using BFGS optimization
    # Note: When called inside jax.lax.scan, result.success is a traced value,
    # so we cannot use Python conditionals on it. The optimization is assumed
    # to succeed for well-posed problems.
    x0_arr: Array = jnp.asarray(x0)
    result = jax.scipy.optimize.minimize(
        fun=neg_log_posterior, x0=x0_arr, method="BFGS"
    )

    mode = result.x
    hessian = jax.hessian(neg_log_posterior)(mode)

    # Add regularization for numerical stability
    # Use pseudo-inverse which is more robust and always works
    reg = 1e-6
    covariance = jnp.linalg.pinv(hessian + jnp.eye(hessian.shape[0]) * reg)

    return mode, covariance


def _approximate_gaussian_newton(
    log_posterior_func: Callable[[ArrayLike], Array],
    x0: ArrayLike,
    n_steps: int = 10,
) -> tuple[Array, Array, Array]:
    """Differentiable Laplace approximation using fixed Newton iterations.

    Unlike approximate_gaussian (which uses BFGS via lax.while_loop and is
    not reverse-mode differentiable), this version uses a fixed number of
    Newton-Raphson steps. This makes it compatible with jax.grad for SGD.

    Only supports 1D (scalar) state variables. Each step is a Newton step
    whose length is the Armijo-acceptable one (of 1, 1/2, ..., 1/128) with
    the highest log posterior, so the iteration ascends monotonically and
    neither oscillates nor zigzags across the mode; for the 1D Smith learning
    model, 10 steps reach the mode to machine precision. ``smith_learning_filter`` uses this solver
    for both its default and its ``differentiable`` path.

    Parameters
    ----------
    log_posterior_func : Callable
        Function computing the log posterior. Takes shape (1,) input.
    x0 : ArrayLike, shape (1,)
        Initial guess for the mode. Must be 1D (scalar state).
    n_steps : int
        Number of Newton-Raphson iterations.

    Returns
    -------
    mode : Array, shape (1,)
        Final Newton iterate (the posterior mode when converged).
    covariance : Array, shape (1, 1)
    newton_gap : Array, shape ()
        Half the squared Newton decrement ``g^2 / (2 h)`` at the final
        iterate: the estimated log-posterior gap (nats) to the mode, ~0 when
        converged (see ``multinomial_choice.NEWTON_GAP_TOL``).
    """
    x0_arr = jnp.asarray(x0)
    # Integer initial guesses are promoted to the default float (jax.grad
    # needs a floating input); float32 stays float32.
    x0_arr = x0_arr.astype(jnp.result_type(x0_arr, 1.0))
    if x0_arr.squeeze().ndim != 0:
        raise ValueError(
            "_approximate_gaussian_newton only supports 1D states, "
            f"got x0 with shape {x0_arr.shape}"
        )

    def neg_log_posterior(x: Array) -> Array:
        return -log_posterior_func(x)

    grad_fn = jax.grad(neg_log_posterior)
    hess_fn = jax.grad(grad_fn)
    step_sizes = jnp.asarray(_NEWTON_STEP_SIZES, dtype=x0_arr.dtype)
    batched_objective = jax.vmap(neg_log_posterior)

    def newton_step(x: Array, _: None) -> tuple[Array, None]:
        g = grad_fn(x)
        h = hess_fn(x)
        # Regularize Hessian for stability
        h_safe = jnp.maximum(h, 1e-6)
        direction = -g / h_safe
        # Armijo backtracking: a full Newton step overshoots when the prior
        # mean sits on the saturated side of the logistic opposite the data
        # (curvature ~ 0 there) and then oscillates between the two saturated
        # sides, e.g. N=10, y=0, prior N(3, 4): 3.0, -10.5, 3.0, ... Of the
        # steps in 1, 1/2, ..., 1/128 that decrease the objective
        # sufficiently, take the one with the lowest objective (the largest
        # within round-off of it); near the mode that is the full step. The
        # largest acceptable step instead zigzags across the mode with slowly
        # shrinking amplitude (N=1, y=0, prior N(5.16, 11.5): 5.16, -5.58,
        # 4.68, -5.21, ... still 4 nats short after ten iterations).
        f_candidates = batched_objective(x + step_sizes * direction)
        f_x = neg_log_posterior(x)
        # Round-off slack: at a converged mode the tiny full step must still
        # be taken, or gradients through the scan miss the Newton map's
        # contraction (see multinomial_choice._softmax_update_core).
        slack = _armijo_slack(x0_arr.dtype) * (1.0 + jnp.abs(f_x))
        sufficient = (
            f_candidates <= f_x + _ARMIJO_C * step_sizes * (g * direction) + slack
        )
        f_best = jnp.min(jnp.where(sufficient, f_candidates, jnp.inf))
        best_ok = jnp.argmax(sufficient & (f_candidates <= f_best + slack))
        alpha = jnp.where(jnp.any(sufficient), step_sizes[best_ok], 0.0)
        return x + alpha * direction, None

    mode, _ = jax.lax.scan(newton_step, jnp.squeeze(x0_arr), None, length=n_steps)

    # Compute covariance from Hessian at mode
    h = hess_fn(mode)
    h_safe = jnp.maximum(h, 1e-6)
    variance = 1.0 / h_safe
    newton_gap = 0.5 * grad_fn(mode) ** 2 / h_safe

    return (
        jnp.expand_dims(mode, 0),
        jnp.expand_dims(jnp.expand_dims(variance, 0), 0),
        newton_gap,
    )


def _chance_logit(prob_correct_by_chance: float) -> Array:
    """Observation-model bias ``mu = log(p / (1 - p))`` for chance level ``p``.

    Unlike ``SmithLearningModel._calculate_mu_bias`` this does not clamp
    ``p`` away from 0 and 1.
    """
    return jnp.log(prob_correct_by_chance / (1 - prob_correct_by_chance))


def _log_posterior_objective(
    learning_state: ArrayLike,
    learning_state_prev: ArrayLike,
    variance_prev: ArrayLike,
    n_correct_in_trial: ArrayLike,
    max_possible_correct: ArrayLike,
    bias: ArrayLike,
) -> Array:
    r"""Objective function for the log posterior distribution at one step.

    Parameters
    ----------
    learning_state : ArrayLike, shape (1,)
        Current latent learning state estimate, $x_k$.
    learning_state_prev : ArrayLike
        Previous latent learning state estimate, $x_{k-1}$.
    variance_prev : ArrayLike
        Previous state variance, $P_{k|k-1}$.
    n_correct_in_trial : ArrayLike
        Number of correct responses in the trial, $y_k$.
    max_possible_correct : ArrayLike
        Maximum number of correct responses, $N_k$.
    bias : ArrayLike
        Bias term ($\mu$) for the observation model.

    Returns
    -------
    log_posterior : Array, shape ()
        Scalar log posterior of the state estimate
    """
    prob_success = jax.nn.sigmoid(bias + learning_state)
    log_likelihood = jax.scipy.stats.binom.logpmf(
        k=n_correct_in_trial, n=max_possible_correct, p=prob_success
    )
    log_prior = jax.scipy.stats.norm.logpdf(
        x=learning_state, loc=learning_state_prev, scale=jnp.sqrt(variance_prev)
    )

    return jnp.squeeze(log_likelihood + log_prior)


def smith_laplace_log_likelihood(
    n_correct_responses: ArrayLike,
    max_possible_correct: ArrayLike,
    filtered_mode: ArrayLike,
    filtered_variance: ArrayLike,
    one_step_mode: ArrayLike,
    one_step_variance: ArrayLike,
    mu: ArrayLike,
) -> Array:
    r"""Per-trial Laplace approximation of ``log p(y_k | y_{1:k-1})``.

    Each filter update approximates the one-step posterior
    ``p(x_k | y_{1:k}) \propto p(y_k | x_k) N(x_k; m_{k|k-1}, P_{k|k-1})`` by a
    Gaussian at its mode ``x_k^*`` with variance ``P_{k|k}``. The same
    approximation of the normaliser gives

    .. math::

        \log p(y_k | y_{1:k-1}) \approx \log p(y_k | x_k^*)
            - \frac{(x_k^* - m_{k|k-1})^2}{2 P_{k|k-1}}
            - \tfrac12 \log P_{k|k-1} + \tfrac12 \log P_{k|k},

    the evidence used by the multinomial-choice and point-process filters.
    Unlike the plug-in ``log p(y_k | x = m_{k|k-1})`` it accounts for the
    predictive uncertainty ``P_{k|k-1}``; the plug-in is over-confident and
    biases a likelihood-based ``sigma_epsilon`` estimate downwards.

    Parameters
    ----------
    n_correct_responses : ArrayLike, shape (n_trials,)
        Number of correct responses ``y_k`` in each trial.
    max_possible_correct : ArrayLike, shape (n_trials,) or scalar
        Number of Bernoulli attempts ``N_k`` in each trial.
    filtered_mode : ArrayLike, shape (n_trials,)
        Posterior mode ``x_k^*`` of the learning state (the filter's
        ``learning_state_mode``).
    filtered_variance : ArrayLike, shape (n_trials,)
        Posterior variance ``P_{k|k}`` at the mode.
    one_step_mode : ArrayLike, shape (n_trials,)
        One-step prediction mean ``m_{k|k-1}``.
    one_step_variance : ArrayLike, shape (n_trials,)
        One-step prediction variance ``P_{k|k-1}``.
    mu : ArrayLike, scalar
        Logit bias of the observation model, ``p_k = sigmoid(mu + x_k)``
        (``log(p_chance / (1 - p_chance))``).

    Returns
    -------
    log_likelihood_terms : Array, shape (n_trials,)
    """
    y = jnp.asarray(n_correct_responses)
    n = jnp.asarray(max_possible_correct)
    mode = jnp.asarray(filtered_mode)
    eta = mu + mode
    log_binom_coef = (
        jax.scipy.special.gammaln(n + 1.0)
        - jax.scipy.special.gammaln(y + 1.0)
        - jax.scipy.special.gammaln(n - y + 1.0)
    )
    log_lik_at_mode = (
        log_binom_coef
        + y * jax.nn.log_sigmoid(eta)
        + (n - y) * jax.nn.log_sigmoid(-eta)
    )
    pred_var = jnp.asarray(one_step_variance)
    return (
        log_lik_at_mode
        - 0.5 * (mode - jnp.asarray(one_step_mode)) ** 2 / pred_var
        - 0.5 * jnp.log(pred_var)
        + 0.5 * jnp.log(jnp.asarray(filtered_variance))
    )


def smith_learning_filter(
    n_correct_responses: ArrayLike,
    init_learning_state: float | ArrayLike = 0.0,
    init_learning_variance: float | ArrayLike | None = None,
    sigma_epsilon: float | ArrayLike = DEFAULT_SIGMA_EPSILON,
    prob_correct_by_chance: float = 0.5,
    max_possible_correct: ArrayLike | None = None,
    differentiable: bool = False,
) -> SmithFilterResult:
    r"""Applies a non-linear Bayesian filter (Laplace approximation) for learning.

    Assumes a random walk model for the latent learning state ($x_k$) and
    a Binomial observation model with a sigmoid link function.

    $$ x_k = x_{k-1} + w_k, \quad w_k \sim N(0, \sigma_\epsilon^2) $$
    $$ p_k = \frac{1}{1 + \exp(-(\mu + x_k))} $$
    $$ y_k \sim \text{Binomial}(N_k, p_k) $$

    Parameters
    ----------
    n_correct_responses : ArrayLike, shape (n_trials,)
        Number of correct responses in each trial ($y_k$).
    init_learning_state : float, optional
        The subject's learning state at the beginning of the experiment ($x_0$).
        When None, it is set to 0.
    init_learning_variance : float, optional
        Initial learning state variance ($P_0$). Defaults to $\sigma_\epsilon^2$.
        Controls how fast the learning state is updated.
    sigma_epsilon : float, optional
        Standard deviation of process noise ($\sigma_\epsilon$), defaults to sqrt(0.05).
    prob_correct_by_chance : float, optional
        The probability of a correct response by chance in absence of any
        learning or experience, used to set the bias.
        Chance probability ($p_{chance}$), defaults to 0.5.
    max_possible_correct : int, ArrayLike, or None, optional
        Maximum number of correct responses in each trial ($N_k$).
        Can be a scalar int (applied to all trials) or an array of
        per-trial values. Defaults to max(n_correct_responses).
    differentiable : bool, optional
        If True, skip the host-side input validation so the filter can be
        traced by ``jax.grad`` (SGD fitting). Both settings find each
        Laplace mode with the same line-searched Newton iterations
        (reverse-mode differentiable), not the BFGS solver of
        :func:`approximate_gaussian`, which can terminate early on a
        line-search failure. Default False.

    Returns
    -------
    prob_correct_response : Array, shape (n_trials,)
        Posterior probability of a correct response ($p_k$).
    learning_state_mode : Array, shape (n_trials,)
        Posterior mode of the learning state ($x_{k|k}$).
    learning_state_variance : Array, shape (n_trials,)
        Posterior variance of the learning state ($P_{k|k}$).
    one_step_mode : Array, shape (n_trials,)
        One-step prediction mode ($x_{k|k-1}$).
    one_step_variance : Array, shape (n_trials,)
        One-step prediction variance ($P_{k|k-1}$).

    Warns
    -----
    StateSpaceWarning
        If a Laplace mode search ends more than
        ``multinomial_choice.NEWTON_GAP_TOL`` nats (Newton estimate) below
        its mode, e.g. for a prior deep in saturation opposite the data.
    """
    # Resolve concrete values before JIT boundary
    if not differentiable:
        validate_count_array(
            n_correct_responses, "n_correct_responses", allow_empty=False
        )
        if max_possible_correct is not None:
            validate_count_array(
                max_possible_correct, "max_possible_correct", allow_empty=False
            )
    n_correct_responses = jnp.asarray(n_correct_responses)
    mu = _chance_logit(prob_correct_by_chance)
    sigma_squared_epsilon = jnp.asarray(sigma_epsilon) ** 2

    init_var: Array
    if init_learning_variance is None:
        init_var = sigma_squared_epsilon
    else:
        init_var = jnp.asarray(init_learning_variance)

    max_correct_arr: Array
    if isinstance(max_possible_correct, (int, float)):
        max_possible_float = float(max_possible_correct)
        if max_possible_float <= 0:
            raise ValueError("max_possible_correct must be positive.")
        max_correct_arr = jnp.array(
            [max_possible_correct] * len(n_correct_responses), dtype=int
        )
    elif max_possible_correct is None:
        max_correct_arr = (
            jnp.ones_like(n_correct_responses, dtype=int) * n_correct_responses.max()
        )
    else:
        max_correct_arr = jnp.asarray(max_possible_correct)
    if not differentiable:
        if max_correct_arr.shape != n_correct_responses.shape:
            raise ValueError(
                f"max_possible_correct shape {max_correct_arr.shape} must match "
                f"n_correct_responses shape {n_correct_responses.shape}."
            )
        if bool(jnp.any(n_correct_responses > max_correct_arr)):
            raise ValueError(
                "n_correct_responses contains values exceeding max_possible_correct."
            )

    # One floating dtype for the scan carry: an integer initial state is
    # promoted to the default float, float32 inputs stay float32.
    init_state = jnp.asarray(init_learning_state)
    dtype = jnp.result_type(init_state, init_var, sigma_squared_epsilon, 1.0)
    return _smith_learning_filter_impl(
        n_correct_responses,
        max_correct_arr,
        init_state.astype(dtype),
        init_var.astype(dtype),
        sigma_squared_epsilon.astype(dtype),
        mu.astype(dtype),
    )


@typed_jit
def _smith_learning_filter_impl(
    n_correct_responses: Array,
    max_correct_arr: Array,
    init_learning_state: Array,
    init_var: Array,
    sigma_squared_epsilon: Array,
    mu: Array,
) -> SmithFilterResult:
    """JIT-compiled inner implementation of the Smith learning filter."""

    def _step(
        carry: tuple[Array, Array], trial_data: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array, Array, Array]]:
        """A single step of the non-linear filter."""
        mode_prev, variance_prev = carry
        n_correct_trial_k, max_possible_correct_trial_k = trial_data

        # 1. Prediction Step
        one_step_mode = mode_prev  # no transition matrix
        one_step_variance = variance_prev + sigma_squared_epsilon

        # 2. Update Step (using Laplace Approximation)
        log_objective_func = partial(
            _log_posterior_objective,
            learning_state_prev=one_step_mode,
            variance_prev=one_step_variance,
            n_correct_in_trial=n_correct_trial_k,
            max_possible_correct=max_possible_correct_trial_k,
            bias=mu,
        )
        # Find mode and covariance (variance). Both paths use the line-searched
        # Newton solver: jax.scipy's BFGS (``approximate_gaussian``) can stop
        # early on a line-search failure (e.g. a 0.06 mode error on a
        # binomial trial right after a run of saturated outcomes), while the
        # 1-D damped Newton reaches the mode to machine precision.
        posterior_mode, posterior_variance, newton_gap = _approximate_gaussian_newton(
            log_objective_func, x0=jnp.array([one_step_mode])
        )
        posterior_mode = jnp.squeeze(posterior_mode)
        posterior_variance = jnp.squeeze(posterior_variance)

        return (posterior_mode, posterior_variance), (
            posterior_mode,
            posterior_variance,
            one_step_mode,
            one_step_variance,
            newton_gap,
        )

    # Run the filter over all trials
    init_carry = (jnp.asarray(init_learning_state), init_var)
    inputs = (n_correct_responses, max_correct_arr)
    _, output = jax.lax.scan(
        _step,
        init_carry,
        inputs,
    )
    (
        learning_state_mode,
        learning_state_variance,
        one_step_mode,
        one_step_variance,
        newton_gaps,
    ) = output
    _warn_if_newton_unconverged(newton_gaps, "smith_learning_filter")

    # Compute probability of correct response
    prob_correct_response = jax.nn.sigmoid(mu + learning_state_mode)

    return SmithFilterResult(
        prob_correct_response=prob_correct_response,
        learning_state_mode=learning_state_mode,
        learning_state_variance=learning_state_variance,
        one_step_mode=one_step_mode,
        one_step_variance=one_step_variance,
    )


@typed_jit
def smith_learning_smoother(
    filtered_learning_state_mode: ArrayLike,
    filtered_learning_state_variance: ArrayLike,
    one_step_mode: ArrayLike,
    one_step_variance: ArrayLike,
    prob_correct_by_chance: float = 0.5,
) -> tuple[Array, Array, Array, Array]:
    """Smooth the filtered learning state estimates using the Kalman smoother.

    Parameters
    ----------
    filtered_learning_state_mode : ArrayLike, shape (n_trials,)
        Learning state mode estimates from the filter
    filtered_learning_state_variance : ArrayLike, shape (n_trials,)
        Learning state variance estimates from the filter
    one_step_mode : ArrayLike, shape (n_trials,)
        One-step prediction of the learning state
    one_step_variance : ArrayLike, shape (n_trials,)
        One-step prediction of the variance of the learning state
    prob_correct_by_chance : float, optional
        The probability of a correct response by chance in absence of any
        learning or experience.
        Default is 0.5.

    Returns
    -------
    learning_state_mode : Array, shape (n_trials,)
        Smoothed learning state mode estimates
    learning_state_variance : Array, shape (n_trials,)
        Smoothed learning state variance estimates
    prob_correct_response : Array, shape (n_trials,)
        Smoothed probability of a correct response
    smoother_gain : Array, shape (n_trials - 1,)
        Smoother gain estimates
    """
    filtered_learning_state_mode = jnp.asarray(filtered_learning_state_mode)
    filtered_learning_state_variance = jnp.asarray(filtered_learning_state_variance)
    one_step_mode = jnp.asarray(one_step_mode)
    one_step_variance = jnp.asarray(one_step_variance)
    n_trials: int = len(filtered_learning_state_mode)

    def _step(
        carry: tuple[Array, Array], k: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        """A single step of the RTS smoother."""
        mode_smoothed_next, variance_smoothed_next = carry
        smoother_gain = (
            filtered_learning_state_variance[k] / one_step_variance[k + 1]
        )  # smoother gain, A_k
        mode_smoothed = filtered_learning_state_mode[k] + smoother_gain * (
            mode_smoothed_next - one_step_mode[k + 1]
        )
        variance_smoothed = filtered_learning_state_variance[k] + smoother_gain**2 * (
            variance_smoothed_next - one_step_variance[k + 1]
        )

        return (mode_smoothed, variance_smoothed), (
            mode_smoothed,
            variance_smoothed,
            smoother_gain,
        )

    init_params = (
        filtered_learning_state_mode[-1],
        filtered_learning_state_variance[-1],
    )
    _, output = jax.lax.scan(
        _step,
        init_params,
        jnp.arange(n_trials - 1),
        reverse=True,
    )

    (
        learning_state_mode,
        learning_state_variance,
        smoother_gain,
    ) = output

    # Append the last state to the smoothed estimates
    learning_state_mode = jnp.concatenate(
        [learning_state_mode, filtered_learning_state_mode[-1:]]
    )
    learning_state_variance = jnp.concatenate(
        [learning_state_variance, filtered_learning_state_variance[-1:]]
    )

    mu_bias = _chance_logit(prob_correct_by_chance)
    prob_correct_response = jax.nn.sigmoid(mu_bias + learning_state_mode)

    return (
        learning_state_mode,
        learning_state_variance,
        prob_correct_response,
        smoother_gain,
    )


# Floor for the re-estimated initial-state variance: the M-step optimum
# P_{1|T} - sigma^2 is often <= 0, and a strictly positive value keeps the
# POSITIVE transform of fit_sgd finite.
_MIN_INIT_LEARNING_VARIANCE = 1e-8


@partial(typed_jit, static_argnames=["estimate_initial_variance"])
def maximization_step(
    smoothed_learning_state_mode: ArrayLike,
    smoothed_learning_state_variance: ArrayLike,
    smoother_gain: ArrayLike,
    estimate_initial_variance: bool = True,
) -> tuple[Array, Array, Array]:
    r"""Exact EM M-step for the process noise and the initial state.

    The filter draws ``x_1 ~ N(x_0, P_0 + \sigma^2)`` (one random-walk step
    from the initial state ``x_0`` with variance ``P_0``) and then
    ``x_{k+1} ~ N(x_k, \sigma^2)``. The expected complete-data
    log-likelihood in ``(\sigma^2, x_0, P_0)`` is

    .. math::

        -\tfrac12 \log(P_0 + \sigma^2)
        - \frac{(x_{1|T} - x_0)^2 + P_{1|T}}{2 (P_0 + \sigma^2)}
        - \tfrac{T-1}{2} \log \sigma^2 - \frac{S}{2 \sigma^2},
        \qquad S = \sum_k E[(x_{k+1} - x_k)^2 | y_{1:T}].

    Its maximiser is ``\sigma^2 = S / (T - 1)``, ``x_0 = x_{1|T}`` and
    ``P_0 = P_{1|T} - \sigma^2`` (floored at a small positive value), not
    ``P_0 = P_{1|T}``: the latter makes the prior on ``x_1``
    ``P_{1|T} + \sigma^2``, a generalised-EM step that does not maximise the
    objective.

    Parameters
    ----------
    smoothed_learning_state_mode : ArrayLike, shape (n_trials,)
        Smoothed learning state mode estimates.
    smoothed_learning_state_variance : ArrayLike, shape (n_trials,)
        Smoothed learning state variance estimates.
    smoother_gain : ArrayLike, shape (n_trials - 1,)
        Smoother gain estimates.
    estimate_initial_variance : bool, default True
        Maximise jointly over ``(sigma^2, x_0, P_0)`` (the default
        ``"reestimate_initial_from_data"`` method). False maximises the
        transition terms only (static under ``jax.jit``).

    Returns
    -------
    sigma_epsilon : Array, shape ()
        Estimated process noise standard deviation (scalar).
    init_learning_state : Array, shape ()
        Initial learning state estimate ``x_{1|T}`` (scalar).
    init_learning_variance : Array, shape ()
        Initial learning state variance estimate
        ``max(P_{1|T} - sigma_epsilon**2, 1e-8)`` (``P_{1|T}`` when
        ``estimate_initial_variance`` is False).
    """
    smoothed_learning_state_mode = jnp.asarray(smoothed_learning_state_mode)
    smoothed_learning_state_variance = jnp.asarray(smoothed_learning_state_variance)
    smoother_gain = jnp.asarray(smoother_gain)
    n_trials: int = len(smoothed_learning_state_mode)
    # E[(x_{k+1} - x_k)^2 | y_{1:T}] = (x_{k+1|T} - x_{k|T})^2 + P_{k+1|T} + P_{k|T}
    #                                  - 2 * Cov(x_{k+1}, x_k | y_{1:T})
    # where Cov(x_{k+1}, x_k | y_{1:T}) = A_k * P_{k+1|T}
    expected_squared_diff_terms = (
        (smoothed_learning_state_mode[1:] - smoothed_learning_state_mode[:-1]) ** 2
        + smoothed_learning_state_variance[1:]
        + smoothed_learning_state_variance[:-1]
        - 2.0
        * smoothed_learning_state_variance[1:]
        * smoother_gain  # Cov = A_k * P_{k+1|T}
    )

    expected_squared_diff = jnp.sum(expected_squared_diff_terms)
    sigma_epsilon_sq = expected_squared_diff / (n_trials - 1)
    first_variance = smoothed_learning_state_variance[0]
    if estimate_initial_variance:
        # P_0 = P_{1|T} - sigma^2 >= 0 binds: optimum at P_0 = 0, where the
        # prior term (x_1 - x_0)^2 / sigma^2 joins the transition terms.
        sigma_epsilon_sq = jnp.where(
            first_variance >= sigma_epsilon_sq,
            sigma_epsilon_sq,
            (expected_squared_diff + first_variance) / n_trials,
        )
    # Clamp to prevent NaN from negative values (can occur when smoother
    # gains exceed 1) and enforce a minimum floor to avoid degenerate estimates.
    sigma_epsilon_sq = jnp.maximum(sigma_epsilon_sq, 1e-12)
    sigma_epsilon = jnp.sqrt(sigma_epsilon_sq)

    init_learning_state = smoothed_learning_state_mode[0]
    if estimate_initial_variance:
        init_learning_variance = jnp.maximum(
            first_variance - sigma_epsilon_sq, _MIN_INIT_LEARNING_VARIANCE
        )
    else:
        init_learning_variance = first_variance

    return sigma_epsilon, init_learning_state, init_learning_variance


def calculate_probability_confidence_limits(
    key: Array,
    smoothed_learning_state_mode: ArrayLike,
    smoothed_learning_state_variance: ArrayLike,
    prob_correct_by_chance: float,
    n_samples: int = 10000,
    percentiles: ArrayLike | None = None,
    return_prob_above_chance: bool = False,
) -> tuple[Array, Array | None]:
    """Calculates confidence limits for the probability of a correct response.

    This is achieved by sampling from the smoothed posterior distribution of
    the learning state for each trial and transforming these samples through
    the sigmoid link function.

    Parameters
    ----------
    key : Array
        JAX PRNG key for random number generation.
    smoothed_learning_state_mode : ArrayLike, shape (n_trials,)
        Smoothed learning state means (x_{k|T}).
    smoothed_learning_state_variance : ArrayLike, shape (n_trials,)
        Smoothed learning state variances (P_{k|T}).
    prob_correct_by_chance : float
        Probability of a correct response by chance. Used to compute the
        bias term (mu) in the sigmoid function: p_k = sigmoid(mu + x_k).
        If ``return_prob_above_chance`` is True, also used as the threshold
        for computing prob_above_chance.
    n_samples : int, optional
        Number of Monte Carlo samples to draw per trial. Default is 10000.
    percentiles : ArrayLike, optional
        Array of percentiles to compute (e.g., jnp.array([5, 50, 95])).
        If None, defaults to jnp.array([5.0, 50.0, 95.0]).
    return_prob_above_chance : bool, optional
        If True, computes the certainty (prob_above_chance) that the true probability
        of a correct response is greater than the chance level. Default is False.

    Returns
    -------
    probability_percentiles : Array, shape (n_percentiles, n_trials)
        The computed percentile values for the probability of correct response
        for each trial.
    prob_above_chance : Optional[Array], shape (n_trials,)
        The certainty that p_k > prob_correct_by_chance for each trial.
        Returned if return_prob_above_chance is True.
    """
    mu_bias = math.log(prob_correct_by_chance / (1 - prob_correct_by_chance))
    smoothed_learning_state_mode = jnp.asarray(smoothed_learning_state_mode)
    smoothed_learning_state_variance = jnp.asarray(smoothed_learning_state_variance)
    if percentiles is None:
        percentiles = jnp.array([5.0, 50.0, 95.0])

    n_trials = smoothed_learning_state_mode.shape[0]

    epsilon = 1e-9
    smoothed_std_dev = jnp.sqrt(jnp.maximum(smoothed_learning_state_variance, epsilon))

    # Function to process a single trial
    def process_trial(
        key_trial: Array, mode_k: Array, std_dev_k: Array
    ) -> tuple[Array, Array | None]:
        # Generate samples from the Gaussian posterior of the learning state x_k
        # latent_state_samples will have shape (n_samples,)
        latent_state_samples = mode_k + std_dev_k * jax.random.normal(
            key_trial, shape=(n_samples,)
        )

        # Transform samples to probability of correct response
        # prob_samples will have shape (n_samples,)
        prob_samples = jax.nn.sigmoid(mu_bias + latent_state_samples)

        # Calculate requested percentiles for this trial
        trial_percentiles = jnp.percentile(prob_samples, percentiles)

        # Calculate prob_above_chance if requested
        if return_prob_above_chance:
            trial_prob_above_chance = jnp.mean(prob_samples > prob_correct_by_chance)
            return trial_percentiles, trial_prob_above_chance
        else:
            return trial_percentiles, None  # Or jnp.nan if a consistent shape is needed

    # Generate per-trial PRNG keys
    trial_keys = jax.random.split(key, n_trials)
    probability_percentiles = jax.vmap(process_trial)(
        trial_keys, smoothed_learning_state_mode, smoothed_std_dev
    )

    if return_prob_above_chance:
        return probability_percentiles[0].T, probability_percentiles[1]
    else:
        return probability_percentiles[0].T, None


def find_min_consecutive_successes(
    prob_correct_by_chance: float,
    critical_probability_threshold: float,
    sequence_length: int,
    min_run_length: int = 2,
    max_run_length: int = 35,
) -> int | None:
    """
    Finds the minimum number of consecutive successes (run_length) in a sequence of
    `sequence_length` Bernoulli trials (with success probability `prob_correct_by_chance`)
    such that the probability of observing at least one such run is less
    than `critical_probability_threshold`.

    This logic closely follows the MATLAB findj.m implementation, which is
    based on methods for calculating run probabilities.

    Parameters
    ----------
    prob_correct_by_chance : float
        Probability of a correct response under the null hypothesis (e.g., 0.5).
    critical_probability_threshold : float
        The critical p-value (e.g., 0.01 or 0.05).
    sequence_length : int
        The length of the trial sequence to consider.
    min_run_length : int, optional
        Minimum run length to test. Default is 2.
    max_run_length : int, optional
        Maximum run length to test. Default is 35 (based on findj.m).

    Returns
    -------
    Optional[int]
        The minimum number of consecutive successes (`final_run_length`) that meets
        the criterion. Returns None if no run_length in the range
        [min_run_length, max_run_length] satisfies the condition.
    """
    if not (0 < prob_correct_by_chance < 1):
        raise ValueError("prob_correct_by_chance must be between 0 and 1.")
    if not (0 < critical_probability_threshold < 1):
        raise ValueError("critical_probability_threshold must be between 0 and 1.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if min_run_length < 1 or max_run_length < min_run_length:
        raise ValueError(
            "Invalid min_run_length or max_run_length. "
            "Ensure min_run_length >= 1 and max_run_length >= min_run_length."
        )

    for current_run_length in range(min_run_length, max_run_length + 1):
        total_prob_at_least_one_run: float
        if current_run_length > sequence_length:
            total_prob_at_least_one_run = 0.0
        else:
            n_possible_ending_positions = sequence_length - current_run_length + 1
            prob_first_run_ends_at_idx = np.zeros(n_possible_ending_positions)

            prob_run_occurs = prob_correct_by_chance**current_run_length

            # Base case: first run ends at trial `current_run_length`
            prob_first_run_ends_at_idx[0] = prob_run_occurs

            if sequence_length <= 2 * current_run_length:
                if n_possible_ending_positions > 1:
                    prob_first_run_ends_at_idx[1:] = prob_run_occurs * (
                        1 - prob_correct_by_chance
                    )
            else:
                idx_simple_end = min(
                    current_run_length, n_possible_ending_positions - 1
                )
                if idx_simple_end >= 1:
                    prob_first_run_ends_at_idx[1 : idx_simple_end + 1] = (
                        prob_run_occurs * (1 - prob_correct_by_chance)
                    )

                # Recursive part for runs ending at trials > 2*current_run_length
                for current_f_idx in range(
                    current_run_length + 1, n_possible_ending_positions
                ):
                    sum_limit_exclusive = current_f_idx - current_run_length
                    sum_prev_f_values = np.sum(
                        prob_first_run_ends_at_idx[0:sum_limit_exclusive]
                    )
                    prob_first_run_ends_at_idx[current_f_idx] = (
                        prob_run_occurs
                        * (1 - prob_correct_by_chance)
                        * (1 - sum_prev_f_values)
                    )

            total_prob_at_least_one_run = float(np.sum(prob_first_run_ends_at_idx))

        if total_prob_at_least_one_run < critical_probability_threshold:
            return current_run_length

    return None  # No run_length in the specified range met the criterion


def simulate_learning_data(
    n_trials: int = 50,
    prob_success_init: float = 0.125,
    prob_success_final: float = 0.6,
    learning_rate: float = 0.2,
    inflection_point: float = 25.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Simulates learning data with a sigmoid probability curve.

    Generates binary outcomes (0 or 1) for a specified number of trials.
    The probability of success (outcome 1) for each trial follows a
    sigmoid curve, transitioning from an initial probability to a final
    probability.

    Parameters
    ----------
    n_trials : int, optional
        The total number of trials to simulate. Default is 50.
    prob_success_init : float, optional
        The initial probability of success at the beginning of learning.
        Should be between 0 and 1. Default is 0.125.
    prob_success_final : float, optional
        The final (asymptotic) probability of success after learning plateaus.
        Should be between 0 and 1. Default is 0.6.
    learning_rate : float, optional
        The rate of learning, controlling the steepness of the sigmoid curve.
        Higher values indicate faster learning. Default is 0.2.
    inflection_point : float, optional
        The trial number at which the learning curve has its inflection point
        (i.e., the point of steepest learning). Default is 25.0.
    seed : Optional[int], optional
        A seed for the random number generator to ensure reproducibility.
        If None, the generator is initialized without a fixed seed.
        Default is None.

    Returns
    -------
    simulated_outcomes : np.ndarray, shape (n_trials,)
        An array of simulated binary outcomes (0 or 1) for each trial.
    true_prob_success : np.ndarray, shape (n_trials,)
        An array of the true underlying probabilities of success for each trial.

    Notes
    -----
    The probability of success $P_k$ for trial $k$ is calculated as:
    $$
    P_k = P_{init} + (P_{final} - P_{init}) / (1 + \exp(-lr \cdot (k - infl)))
    $$
    where $P_{init}$ is `prob_success_init`, $P_{final}$ is `prob_success_final`,
    $lr$ is `learning_rate`, $k$ is the trial number (0-indexed), and $infl$
    is `inflection_point`.
    """
    if not (0 <= prob_success_init <= 1):
        raise ValueError("prob_success_init must be between 0 and 1.")
    if not (0 <= prob_success_final <= 1):
        raise ValueError("prob_success_final must be between 0 and 1.")

    trial_indices = np.arange(n_trials)
    sigmoid_component = scipy.special.expit(
        learning_rate * (trial_indices - inflection_point)
    )
    true_prob_success = (
        prob_success_init + (prob_success_final - prob_success_init) * sigmoid_component
    )
    true_prob_success = np.clip(true_prob_success, 0.0, 1.0)

    rng = np.random.default_rng(seed=seed)
    simulated_outcomes = rng.binomial(1, true_prob_success)

    return simulated_outcomes, true_prob_success


def _find_runs_of_value(
    data: jax.Array, value_to_find: int, min_length: int
) -> list[tuple[int, int]]:
    """
    Finds start and end indices of runs of a specific value of at least a minimum length.

    Parameters
    ----------
    data : jax.Array, 1D
        The input sequence of data.
    value_to_find : int
        The value for which to find runs (e.g., 1 for successes).
    min_length : int
        The minimum length of a run to be identified.

    Returns
    -------
    list[tuple[int, int]]
        A list of (start_index, end_index) tuples for each qualifying run.
        Indices are 0-based, and end_index is inclusive.
    """
    if min_length <= 0:
        return []

    # Create a boolean array where True indicates the presence of value_to_find
    is_value = data == value_to_find

    # Pad with False at both ends to correctly identify runs at the start/end
    padded_is_value = jnp.concatenate(
        [jnp.array([False]), is_value, jnp.array([False])]
    )

    # Find changes: 0 to 1 (run start), 1 to 0 (run end)
    diffs = jnp.diff(padded_is_value.astype(jnp.int32))

    # Start indices are where diff goes from 0 to 1 (original index)
    run_starts = jnp.where(diffs == 1)[0]
    # End indices are where diff goes from 1 to 0 (original index is one less)
    run_ends = jnp.where(diffs == -1)[0] - 1

    runs = []
    for start, end in zip(run_starts.tolist(), run_ends.tolist()):
        if (end - start + 1) >= min_length:
            runs.append((start, end))
    return runs


def compute_cross_covariance_matrix(
    smoothed_learning_state_variance: ArrayLike,
    smoother_gain: ArrayLike,
) -> Array:
    """Compute the full cross-covariance matrix for all trial pairs.

    Computes Cov(x_i, x_j | y_{1:T}) for all pairs i <= j using the formula:
        Cov(x_i, x_j | y_{1:T}) = A_i * A_{i+1} * ... * A_{j-1} * P_{j|T}

    This is a vectorized computation using cumulative products of smoother gains.

    Parameters
    ----------
    smoothed_learning_state_variance : jnp.ndarray, shape (n_trials,)
        Smoothed learning state variances (P_{k|T}).
    smoother_gain : jnp.ndarray, shape (n_trials - 1,)
        Smoother gain values (A_k).

    Returns
    -------
    cross_cov_matrix : jnp.ndarray, shape (n_trials, n_trials)
        Upper triangular matrix where entry [i, j] (i <= j) contains
        Cov(x_i, x_j | y_{1:T}). Diagonal contains variances P_{k|T}.
        Lower triangle is symmetric (Cov(x_i, x_j) = Cov(x_j, x_i)).
    """
    smoothed_var = jnp.asarray(smoothed_learning_state_variance)
    gain = jnp.asarray(smoother_gain)
    n_trials = smoothed_var.shape[0]

    # Compute cumulative product of smoother gains
    # cumgain[k] = A_0 * A_1 * ... * A_{k-1} for k >= 1, cumgain[0] = 1
    log_gains = jnp.log(jnp.clip(gain, 1e-10, None))
    cumsum_log_gains = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(log_gains)])

    # For Cov(x_i, x_j) where i < j:
    # = A_i * A_{i+1} * ... * A_{j-1} * P_j
    # = (cumgain[j] / cumgain[i]) * P_j
    # = exp(cumsum_log[j] - cumsum_log[i]) * P_j

    # Create index grids
    i_idx, j_idx = jnp.meshgrid(
        jnp.arange(n_trials), jnp.arange(n_trials), indexing="ij"
    )

    # Compute log of gain products: log(A_i * ... * A_{j-1}) = cumsum_log[j] - cumsum_log[i]
    log_gain_products = cumsum_log_gains[j_idx] - cumsum_log_gains[i_idx]

    # Cross-covariance = exp(log_gain_products) * P_j (for i <= j)
    cross_cov_matrix = jnp.exp(log_gain_products) * smoothed_var[j_idx]

    # For i > j, use symmetry: Cov(x_i, x_j) = Cov(x_j, x_i)
    # But our formula computes Cov(x_i, x_j) = product(A_i:A_{j-1}) * P_j
    # which is only valid for i <= j. For i > j, we need to use the transpose.
    upper_tri = jnp.triu(cross_cov_matrix)
    return upper_tri + upper_tri.T - jnp.diag(jnp.diag(upper_tri))


def _sample_trial_states(
    key: Array,
    mean: Array,
    cov: Array,
    n_samples: int,
    compare_probability: bool,
    prob_correct_by_chance: float | None,
) -> Array:
    """Draw joint Gaussian samples of learning states for trial comparisons.

    Samples ``N(mean, cov)`` through an eigendecomposition of ``cov`` (negative
    eigenvalues clipped to zero) and, if ``compare_probability``, maps them to
    probability space with ``sigmoid(mu + x)``.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    mean : Array, shape (n_compared,)
        Smoothed learning-state modes of the compared trials.
    cov : Array, shape (n_compared, n_compared)
        Joint posterior covariance of the compared trials.
    n_samples : int
        Number of Monte Carlo samples.
    compare_probability : bool
        If True, return sigmoid-transformed samples.
    prob_correct_by_chance : float or None
        Chance-level probability setting the bias ``mu``; required when
        ``compare_probability`` is True.

    Returns
    -------
    samples : Array, shape (n_samples, n_compared)

    Raises
    ------
    ValueError
        If ``compare_probability`` is True and ``prob_correct_by_chance`` is
        None.
    """
    # Eigendecomposition for numerical stability
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)  # Ensure non-negative
    sqrt_cov = eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues))

    z = jax.random.normal(key, shape=(n_samples, mean.shape[0]))
    samples: Array = mean + z @ sqrt_cov.T

    if compare_probability:
        if prob_correct_by_chance is None:
            raise ValueError(
                "prob_correct_by_chance is required when compare_probability=True"
            )
        mu_bias = math.log(prob_correct_by_chance / (1 - prob_correct_by_chance))
        samples = jax.nn.sigmoid(mu_bias + samples)
    return samples


def compute_trial_comparison_matrix(
    key: Array,
    smoothed_learning_state_mode: ArrayLike,
    smoothed_learning_state_variance: ArrayLike,
    smoother_gain: ArrayLike,
    n_samples: int = 10000,
    compare_probability: bool = False,
    prob_correct_by_chance: float | None = None,
) -> Array:
    """Compute pairwise comparison matrix for all trials (vectorized).

    Computes P(x_i > x_j | y_{1:T}) for all pairs of trials i < j.
    This implements the full trialtotrial.m functionality using vectorized
    JAX operations for efficiency.

    Parameters
    ----------
    key : Array
        JAX PRNG key for Monte Carlo sampling.
    smoothed_learning_state_mode : jnp.ndarray, shape (n_trials,)
        Smoothed learning state modes (x_{k|T}).
    smoothed_learning_state_variance : jnp.ndarray, shape (n_trials,)
        Smoothed learning state variances (P_{k|T}).
    smoother_gain : jnp.ndarray, shape (n_trials - 1,)
        Smoother gain values (A_k).
    n_samples : int, optional
        Number of Monte Carlo samples per comparison. Default is 10000.
    compare_probability : bool, optional
        If True, compare in probability space. Default is False.
    prob_correct_by_chance : Optional[float], optional
        Probability of correct response by chance. Required if compare_probability=True.
        Used to compute the bias term for sigmoid transformation.

    Returns
    -------
    comparison_matrix : jnp.ndarray, shape (n_trials, n_trials)
        Upper triangular matrix where entry [i, j] (i < j) contains
        P(x_i > x_j | y_{1:T}). Diagonal is 0.5, lower triangle is NaN.
    """
    mode = jnp.asarray(smoothed_learning_state_mode)
    n_trials = mode.shape[0]

    # Compute full cross-covariance matrix
    cross_cov_matrix = compute_cross_covariance_matrix(
        smoothed_learning_state_variance, smoother_gain
    )

    # Joint samples for all trials at once: shape (n_samples, n_trials)
    samples = _sample_trial_states(
        key,
        mode,
        cross_cov_matrix,
        n_samples,
        compare_probability,
        prob_correct_by_chance,
    )

    # Compute P(x_i > x_j) for all pairs using broadcasting
    # samples[:, :, None] has shape (n_samples, n_trials, 1)
    # samples[:, None, :] has shape (n_samples, 1, n_trials)
    # comparison has shape (n_samples, n_trials, n_trials)
    comparison = samples[:, :, None] > samples[:, None, :]

    # Mean over samples gives P(x_i > x_j)
    comparison_matrix = jnp.mean(comparison, axis=0)

    # Set diagonal to 0.5 and lower triangle to NaN
    comparison_matrix = jnp.where(
        jnp.eye(n_trials, dtype=bool),
        0.5,
        comparison_matrix,
    )
    return jnp.where(
        jnp.tril(jnp.ones((n_trials, n_trials), dtype=bool), k=-1),
        jnp.nan,
        comparison_matrix,
    )


def compare_two_trials(
    key: Array,
    smoothed_learning_state_mode: ArrayLike,
    smoothed_learning_state_variance: ArrayLike,
    smoother_gain: ArrayLike,
    trial1: int,
    trial2: int,
    n_samples: int = 10000,
    compare_probability: bool = False,
    prob_correct_by_chance: float | None = None,
) -> float:
    """Compute the probability that learning state at trial1 > trial2.

    This implements the trial-to-trial comparison from trialtotrial.m,
    computing P(x_{trial1} > x_{trial2} | y_{1:T}) or optionally
    P(p_{trial1} > p_{trial2} | y_{1:T}) for probability space.

    For comparing many pairs, use compute_trial_comparison_matrix() instead
    which is more efficient due to vectorization.

    Parameters
    ----------
    key : Array
        JAX PRNG key for Monte Carlo sampling.
    smoothed_learning_state_mode : jnp.ndarray, shape (n_trials,)
        Smoothed learning state modes (x_{k|T}).
    smoothed_learning_state_variance : jnp.ndarray, shape (n_trials,)
        Smoothed learning state variances (P_{k|T}).
    smoother_gain : jnp.ndarray, shape (n_trials - 1,)
        Smoother gain values (A_k).
    trial1 : int
        First trial index (0-based).
    trial2 : int
        Second trial index (0-based). Must be different from trial1.
    n_samples : int, optional
        Number of Monte Carlo samples. Default is 10000.
    compare_probability : bool, optional
        If True, compare in probability space (sigmoid-transformed).
        If False, compare raw latent states. Default is False.
    prob_correct_by_chance : Optional[float], optional
        Probability of correct response by chance. Required if compare_probability=True.
        Used to compute the bias term for sigmoid transformation.

    Returns
    -------
    posterior_probability : float
        Bayesian posterior probability P(x_{trial1} > x_{trial2} | y_{1:T}),
        or in probability space if requested. This is NOT a frequentist p-value.
        Values > 0.5 indicate trial1 has higher learning state than trial2.
        Values near 0.95 or 0.05 indicate statistically significant differences.
    """
    if trial1 == trial2:
        return 0.5  # Same trial, no difference

    # Sample from 2x2 marginal; avoids O(n² × n_samples) full comparison matrix
    cross_cov_matrix = compute_cross_covariance_matrix(
        smoothed_learning_state_variance, smoother_gain
    )
    indices = jnp.array([trial1, trial2])
    mode = jnp.asarray(smoothed_learning_state_mode)
    mean_2 = mode[indices]
    cov_2 = cross_cov_matrix[jnp.ix_(indices, indices)]

    # Sample from bivariate normal: shape (n_samples, 2)
    samples = _sample_trial_states(
        key, mean_2, cov_2, n_samples, compare_probability, prob_correct_by_chance
    )

    return float(jnp.mean(samples[:, 0] > samples[:, 1]))


def find_first_significant_trial(
    comparison_matrix: ArrayLike,
    reference_trial: int = 0,
    significance_level: float = 0.05,
) -> int | None:
    """Find the first trial significantly different from a reference trial.

    Parameters
    ----------
    comparison_matrix : jnp.ndarray, shape (n_trials, n_trials)
        Comparison matrix from compute_trial_comparison_matrix.
    reference_trial : int, optional
        The reference trial to compare against (usually 0 or 1). Default is 0.
    significance_level : float, optional
        Significance threshold (two-tailed). Default is 0.05.

    Returns
    -------
    first_significant : Optional[int]
        The first trial index that is significantly different from the reference,
        or None if no such trial exists.
    """
    # For trials after reference, check if P(ref > trial) < alpha/2
    # (i.e., trial is significantly HIGHER than reference). One host transfer
    # and a vectorized comparison instead of a device sync per trial.
    threshold_high = significance_level / 2  # e.g., 0.025

    comparison = np.asarray(comparison_matrix)
    first_candidate = reference_trial + 1
    if first_candidate >= comparison.shape[0]:
        return None
    row = comparison[reference_trial, first_candidate:]
    # P(ref > j) < 0.025 means j is significantly higher (NaN never qualifies)
    significant = np.flatnonzero(row < threshold_high)
    if significant.size == 0:
        return None
    return int(first_candidate + significant[0])


def calculate_latent_state_percentiles(
    key: Array,
    smoothed_learning_state_mode: ArrayLike,  # shape: (n_trials,)
    smoothed_learning_state_variance: ArrayLike,  # shape: (n_trials,)
    n_samples: int = 10000,
    percentiles: ArrayLike | None = None,
) -> Array:
    """Calculates confidence percentiles for the smoothed latent state.

    Samples from the smoothed posterior distribution of the learning state
    N(x_k|T, P_k|T) for each trial.

    Parameters
    ----------
    key : Array
        JAX PRNG key for random number generation.
    smoothed_learning_state_mode : jnp.ndarray, shape (n_trials,)
        Smoothed learning state means (x_{k|T}).
    smoothed_learning_state_variance : jnp.ndarray, shape (n_trials,)
        Smoothed learning state variances (P_{k|T}).
    n_samples : int, optional
        Number of Monte Carlo samples to draw per trial. Default is 10000.
    percentiles : jnp.ndarray, optional
        Array of percentiles to compute (e.g., jnp.array([5, 50, 95])).
        If None, defaults to jnp.array([5.0, 50.0, 95.0]).

    Returns
    -------
    latent_state_percentiles : jnp.ndarray, shape (n_percentiles, n_trials)
        The computed percentile values for the latent state for each trial.
    """
    if percentiles is None:
        percentiles = jnp.array([5.0, 50.0, 95.0])

    smoothed_learning_state_mode_arr = jnp.asarray(smoothed_learning_state_mode)
    smoothed_learning_state_variance_arr = jnp.asarray(smoothed_learning_state_variance)
    n_trials = smoothed_learning_state_mode_arr.shape[0]
    epsilon = 1e-9  # For numerical stability if variance is tiny
    smoothed_std_dev = jnp.sqrt(
        jnp.maximum(smoothed_learning_state_variance_arr, epsilon)
    )

    def process_trial_state(key_trial: Array, mode_k: Array, std_dev_k: Array) -> Array:
        latent_state_samples = mode_k + std_dev_k * jax.random.normal(
            key_trial, shape=(n_samples,)
        )
        return jnp.percentile(latent_state_samples, percentiles)

    trial_keys = jax.random.split(key, n_trials)
    # Use vmap for efficient per-trial processing
    # mapped_results will have shape (n_trials, n_percentiles)
    mapped_results: Array = jax.vmap(process_trial_state)(
        trial_keys, smoothed_learning_state_mode_arr, smoothed_std_dev
    )
    # Transpose to get (n_percentiles, n_trials)
    return mapped_results.T


VALID_INIT_METHODS = frozenset(
    {
        "reestimate_initial_from_data",
        "set_initial_to_zero",
        "set_initial_conservative_from_second_trial",
        "set_initial_direct_from_second_trial",
        "user_provided",
    }
)


class SmithLearningModel(SGDFittableMixin):
    """Bayesian state-space model for tracking learning from trial outcomes.

    Implements the Smith et al. (2004) algorithm for estimating a latent
    learning state from binomial (correct/incorrect) trial data. Uses a
    Laplace-approximated Kalman filter/smoother with EM parameter estimation.

    Typical workflow::

        model = SmithLearningModel(sigma_epsilon=0.22)
        log_likelihoods = model.fit(outcomes)
        key = jax.random.PRNGKey(0)
        prob_percentiles, _ = model.get_learning_curve(key)
        fig, ax = model.plot_learning_curve(key, observed_n_correct=outcomes)

    Attributes
    ----------
    sigma_epsilon : float
        Process noise standard deviation.
    prob_correct_by_chance : float
        Probability of correct response by chance.
    mu_bias : float
        Bias term derived from ``prob_correct_by_chance`` (logit of chance).
    initial_state_method : str
        Strategy for updating initial state during EM.
    max_possible_correct : Optional[int]
        Maximum correct responses per trial (``N_k``).
    init_learning_state : float
        Initial learning state estimate (updated during EM).
    init_learning_variance : float
        Initial learning state variance (updated during EM).
    filtered_prob_correct_response : jax.Array
        Filtered probability of correct response, shape ``(n_trials,)``.
    filtered_learning_state_mode : jax.Array
        Filtered learning state mode, shape ``(n_trials,)``.
    filtered_learning_state_variance : jax.Array
        Filtered learning state variance, shape ``(n_trials,)``.
    filtered_one_step_mode : jax.Array
        One-step-ahead predicted mode, shape ``(n_trials,)``.
    filtered_one_step_variance : jax.Array
        One-step-ahead predicted variance, shape ``(n_trials,)``.
    smoothed_learning_state_mode : jax.Array
        Smoothed learning state mode, shape ``(n_trials,)``.
    smoothed_learning_state_variance : jax.Array
        Smoothed learning state variance, shape ``(n_trials,)``.
    smoothed_prob_correct_response : jax.Array
        Smoothed probability of correct response, shape ``(n_trials,)``.
    smoother_gain : jax.Array
        Smoother gain, shape ``(n_trials - 1,)``.
    log_likelihood_ : float
        Final log-likelihood after fitting.
    n_iter_ : int or None
        Number of EM iterations performed; None before ``fit`` and after
        ``fit_sgd``.

    The filtered/smoothed estimates and fit diagnostics are set by ``fit`` /
    ``fit_sgd``; reading one before then raises ``NotFittedError``.

    References
    ----------
    Smith, A. C., Frank, L. M., Wirth, S., Yanike, M., Hu, D., Kubota, Y.,
    Graybiel, A. M., Suzuki, W. A., & Brown, E. N. (2004).
    Dynamic analysis of learning in behavioral experiments.
    Journal of Neuroscience, 24(2), 447-461.
    """

    # Filter/smoother outputs, set by each E-step.
    filtered_prob_correct_response: FittedAttribute[Array] = FittedAttribute()
    filtered_learning_state_mode: FittedAttribute[Array] = FittedAttribute()
    filtered_learning_state_variance: FittedAttribute[Array] = FittedAttribute()
    filtered_one_step_mode: FittedAttribute[Array] = FittedAttribute()
    filtered_one_step_variance: FittedAttribute[Array] = FittedAttribute()
    smoothed_learning_state_mode: FittedAttribute[Array] = FittedAttribute()
    smoothed_learning_state_variance: FittedAttribute[Array] = FittedAttribute()
    smoothed_prob_correct_response: FittedAttribute[Array] = FittedAttribute()
    smoother_gain: FittedAttribute[Array] = FittedAttribute()  # (n_trials - 1,)
    # The filter/smoother outputs above, cleared when a fit fails.
    _fit_output_attrs = (
        "filtered_prob_correct_response",
        "filtered_learning_state_mode",
        "filtered_learning_state_variance",
        "filtered_one_step_mode",
        "filtered_one_step_variance",
        "smoothed_learning_state_mode",
        "smoothed_learning_state_variance",
        "smoothed_prob_correct_response",
        "smoother_gain",
    )

    # Bound by the last fit.
    _n_trials_: FittedAttribute[int] = FittedAttribute()
    # ``max_possible_correct`` resolved against the data passed to fit_sgd.
    _resolved_max_correct: FittedAttribute[Array] = FittedAttribute()

    def __init__(
        self,
        init_learning_state: float = 0.0,
        init_learning_variance: float | None = None,
        sigma_epsilon: float = DEFAULT_SIGMA_EPSILON,
        prob_correct_by_chance: float = 0.5,
        max_possible_correct: int | None = None,
        initial_state_method: str = "reestimate_initial_from_data",
    ):
        """Initializes the Smith Learning Algorithm parameters.

        Parameters
        ----------
        init_learning_state : float, optional
            Initial learning state estimate (x_0). Default is 0.0.
        init_learning_variance : float, optional
            Initial learning state variance (P_0). Default is None, which sets it to sigma_epsilon^2.
        sigma_epsilon : float, optional
            Standard deviation of process noise (σ_ε). Default is sqrt(0.05).
        prob_correct_by_chance : float, optional
            Probability of a correct response by chance (p_chance). Default is 0.5.
        max_possible_correct : int, optional
            Maximum number of correct responses in each trial (N_k). Default is None.
        initial_state_method : str, optional
            Mode for learning the initial state.
            Options are:
            - "reestimate_initial_from_data": Re-estimates initial state from the data.
            - "set_initial_to_zero": Initial state is always 0.0 and initial variance is learned.
            - "set_initial_conservative_from_second_trial": Estimates initial state from the second trial's mode.
            - "set_initial_direct_from_second_trial": Uses the second trial's mode and variance directly.
            - "user_provided": Uses user-provided initial state and variance. No re-estimation.

        """
        if not isinstance(init_learning_state, (int, float)):
            raise TypeError("init_learning_state must be a float.")
        # init_learning_state (x_0) is in logit space, not a probability, so no [0,1] bound.

        if sigma_epsilon <= 0.0:
            raise ValueError("sigma_epsilon must be positive.")

        init_var: float
        if init_learning_variance is None:
            init_var = float(sigma_epsilon**2)
        elif not isinstance(init_learning_variance, (int, float)):
            raise TypeError(
                "init_learning_variance must be a non-negative float or None."
            )
        elif init_learning_variance < 0:
            raise ValueError("init_learning_variance must be non-negative.")
        else:
            init_var = float(init_learning_variance)

        if initial_state_method not in VALID_INIT_METHODS:
            raise ValueError(
                f"initial_state_method must be one of {sorted(VALID_INIT_METHODS)}, "
                f"got {initial_state_method!r}."
            )

        if not (0.0 < prob_correct_by_chance < 1.0):
            raise ValueError(
                "prob_correct_by_chance must be between 0 and 1 (exclusive)."
            )
        if max_possible_correct is not None:
            if not isinstance(max_possible_correct, (int, np.ndarray, jax.Array)):
                raise TypeError(
                    "max_possible_correct must be an int, NumPy array, or JAX array if provided."
                )
            if isinstance(max_possible_correct, int) and max_possible_correct <= 0:
                raise ValueError(
                    "max_possible_correct must be a positive integer if provided as scalar."
                )
            if not isinstance(max_possible_correct, int):
                validate_count_array(
                    max_possible_correct, "max_possible_correct", allow_empty=False
                )
                if np.any(np.asarray(max_possible_correct, dtype=float) <= 0):
                    raise ValueError(
                        "max_possible_correct must contain positive integers."
                    )

        self.init_learning_state = float(init_learning_state)
        self.init_learning_variance = init_var

        self.sigma_epsilon = sigma_epsilon
        self.prob_correct_by_chance = prob_correct_by_chance
        self.max_possible_correct = max_possible_correct
        self.mu_bias = self._calculate_mu_bias(self.prob_correct_by_chance)

        self.initial_state_method = initial_state_method

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted else "not fitted"
        return (
            f"SmithLearningModel("
            f"sigma_epsilon={self.sigma_epsilon:.4g}, "
            f"prob_correct_by_chance={self.prob_correct_by_chance:.4g}, "
            f"init_learning_state={self.init_learning_state:.4g}, "
            f"initial_state_method={self.initial_state_method!r}, "
            f"{fitted})"
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return (
            is_set(self, "smoothed_learning_state_mode")
            and is_set(self, "smoothed_learning_state_variance")
            and is_set(self, "smoother_gain")
        )

    def _calculate_mu_bias(self, prob_correct_by_chance: float) -> float:
        """Converts probability of chance performance to mu bias term (logit)."""
        epsilon = 1e-9
        p = max(epsilon, min(prob_correct_by_chance, 1.0 - epsilon))
        return math.log(p / (1.0 - p))

    def _resolve_max_possible_correct(
        self, n_correct_responses: jax.Array
    ) -> jax.Array:
        """Resolves max_possible_correct to an array matching n_correct_responses.

        If ``max_possible_correct`` was provided as an int, broadcasts to an
        array. If it was provided as an array, validates length. If None,
        infers from the maximum of ``n_correct_responses`` (stateless — no
        caching, so re-fitting with different data works correctly).
        """
        n_trials = len(n_correct_responses)
        mpc = self.max_possible_correct

        if mpc is not None:
            if isinstance(mpc, int):
                return jnp.full(n_trials, mpc, dtype=jnp.int32)
            # Array case
            mpc_arr = jnp.asarray(mpc, dtype=jnp.int32)
            if len(mpc_arr) != n_trials:
                raise ValueError(
                    f"max_possible_correct array length ({len(mpc_arr)}) "
                    f"doesn't match n_correct_responses length ({n_trials})."
                )
            validate_count_array(mpc, "max_possible_correct", allow_empty=False)
            if bool(jnp.any(mpc_arr <= 0)):
                raise ValueError("max_possible_correct must contain positive integers.")
            return mpc_arr

        # Infer from data
        val = int(jnp.max(n_correct_responses))
        if val <= 0:
            logger.warning(
                "All n_correct_responses are 0 or less; max_possible_correct inferred as 1."
            )
            val = 1
        logger.info(f"max_possible_correct not provided; inferred as {val} from data.")
        return jnp.full(n_trials, val, dtype=jnp.int32)

    def _e_step(self, n_correct_responses: jax.Array) -> float:
        """E-step of the EM algorithm.

        Computes the expected log-likelihood of the observed data given
        the current parameters. This is done by running the Smith learning filter.

        Parameters
        ----------
        n_correct_responses : jax.Array, shape (n_trials,)
            The sequence of correct responses.

        Returns
        -------
        log_likelihood : float
            Laplace-approximated marginal log-likelihood
            ``sum_k log p(y_k | y_{1:k-1})`` (see
            :func:`smith_laplace_log_likelihood`), which accounts for the
            predictive uncertainty of each one-step prediction.
        """
        validate_count_array(
            n_correct_responses, "n_correct_responses", allow_empty=False
        )
        resolved_trial_max_correct = self._resolve_max_possible_correct(
            n_correct_responses
        )
        if bool(jnp.any(n_correct_responses > resolved_trial_max_correct)):
            raise ValueError(
                "n_correct_responses contains values exceeding max_possible_correct. "
                "Check your data or provide max_possible_correct explicitly."
            )
        (
            self.filtered_prob_correct_response,
            self.filtered_learning_state_mode,
            self.filtered_learning_state_variance,
            self.filtered_one_step_mode,
            self.filtered_one_step_variance,
        ) = smith_learning_filter(
            n_correct_responses,
            init_learning_state=self.init_learning_state,
            init_learning_variance=self.init_learning_variance,
            sigma_epsilon=self.sigma_epsilon,
            prob_correct_by_chance=self.prob_correct_by_chance,
            max_possible_correct=resolved_trial_max_correct,
        )

        log_likelihood_terms = smith_laplace_log_likelihood(
            n_correct_responses,
            resolved_trial_max_correct,
            self.filtered_learning_state_mode,
            self.filtered_learning_state_variance,
            self.filtered_one_step_mode,
            self.filtered_one_step_variance,
            self.mu_bias,
        )
        log_likelihood = jnp.sum(log_likelihood_terms)

        (
            self.smoothed_learning_state_mode,
            self.smoothed_learning_state_variance,
            self.smoothed_prob_correct_response,
            self.smoother_gain,  # This has shape (n_trials-1,)
        ) = smith_learning_smoother(
            self.filtered_learning_state_mode,
            self.filtered_learning_state_variance,
            self.filtered_one_step_mode,
            self.filtered_one_step_variance,
            prob_correct_by_chance=self.prob_correct_by_chance,
        )

        return float(log_likelihood)

    def _m_step(self, n_correct_responses: jax.Array) -> None:
        """M-step of the EM algorithm.

        Updates the model parameters based on the current estimates of the
        latent variables. This is done by maximizing the expected log-likelihood
        computed in the E-step.

        Parameters
        ----------
        n_correct_responses : jax.Array, shape (n_trials,)
            The sequence of correct responses.
        """
        if not self.is_fitted:
            raise RuntimeError("Must run E-step before M-step")

        (
            sigma_epsilon_new,
            new_init_learning_state,
            new_init_learning_variance,
        ) = maximization_step(
            self.smoothed_learning_state_mode,
            self.smoothed_learning_state_variance,
            self.smoother_gain,
            estimate_initial_variance=(
                self.initial_state_method == "reestimate_initial_from_data"
            ),
        )
        self.sigma_epsilon = float(sigma_epsilon_new)

        # --- Apply initial_state_method for initial states ---
        if self.initial_state_method == "reestimate_initial_from_data":
            self.init_learning_state = float(new_init_learning_state)
            self.init_learning_variance = float(new_init_learning_variance)
        elif self.initial_state_method == "set_initial_to_zero":
            self.init_learning_state = 0.0
            self.init_learning_variance = float(self.sigma_epsilon**2)
        elif self.initial_state_method == "set_initial_conservative_from_second_trial":
            if len(self.smoothed_learning_state_mode) > 1:
                self.init_learning_state = float(
                    0.5 * self.smoothed_learning_state_mode[1]
                )  # x_{1|T}
            else:
                logger.warning(
                    "Not enough trials to use 'set_initial_conservative_from_second_trial' (need at least 2). "
                    "Falling back to 'reestimate_initial_from_data' for initial state x0."
                )
                self.init_learning_state = float(new_init_learning_state)
            self.init_learning_variance = float(
                self.sigma_epsilon**2
            )  # Use updated sigma_epsilon
        elif self.initial_state_method == "set_initial_direct_from_second_trial":
            # Heuristic: uses smoothed trial-1 estimate as initial condition.
            # Not from Smith et al. (2004); may not converge to same fixed point
            # as "reestimate_initial_from_data".
            if (
                len(self.smoothed_learning_state_mode) > 1
                and len(self.smoothed_learning_state_variance) > 1
            ):
                self.init_learning_state = float(
                    self.smoothed_learning_state_mode[1]
                )  # x_{1|T}
                self.init_learning_variance = float(
                    self.smoothed_learning_state_variance[1]
                )  # P_{1|T}
            else:
                logger.warning(
                    "Not enough trials/data to use 'set_initial_direct_from_second_trial' (need at least 2). "
                    "Falling back to 'reestimate_initial_from_data' for initial state (x0, P0)."
                )
                self.init_learning_state = float(new_init_learning_state)
                self.init_learning_variance = float(new_init_learning_variance)
        elif self.initial_state_method == "user_provided":
            pass  # No change to initial state, user must set it externally

    def fit(
        self,
        n_correct_responses: ArrayLike,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        verbose: bool = False,
    ) -> list[float]:
        """Fits the model to responses using the EM algorithm.

        Iteratively performs E-steps and M-steps until convergence or
        the maximum number of iterations is reached.

        Calling ``fit()`` again on an already-fitted model performs a
        **warm restart**: it continues from the current parameter values
        (``sigma_epsilon``, ``init_learning_state``, ``init_learning_variance``).
        To start fresh, create a new ``SmithLearningModel`` instance.

        Parameters
        ----------
        n_correct_responses : ArrayLike, shape (n_trials,)
            Number of correct responses at each trial. For binary
            (hit/miss) data, these are 0s and 1s. For multi-choice data,
            values range from 0 to ``max_possible_correct``.
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tolerance : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.
        verbose : bool, optional
            If True, log per-iteration progress and convergence at INFO level
            on the ``state_space_practice.smith_learning_algorithm`` logger
            (enable with e.g. ``logging.basicConfig(level=logging.INFO)``);
            otherwise per-iteration records are DEBUG. Default is False.

        Returns
        -------
        log_likelihoods : list[float]
            A list of marginal log-likelihoods at each iteration. On
            convergence, a final entry is appended from a post-convergence
            E-step that ensures stored results match the MLE parameters.
            A non-finite E-step is never recorded: EM warns, rolls the
            parameters and filter/smoother outputs back to the last accepted
            iteration and stops. If the very first E-step is non-finite the
            list is empty and the filter/smoother outputs are cleared, so
            ``is_fitted`` is False.
        """
        n_correct_responses = jnp.asarray(n_correct_responses)
        if n_correct_responses.ndim != 1:
            raise ValueError(
                f"n_correct_responses must be a 1D array, got shape {n_correct_responses.shape}."
            )
        if len(n_correct_responses) < 2:
            raise ValueError(
                f"n_correct_responses must have at least 2 trials, got {len(n_correct_responses)}."
            )
        validate_count_array(
            n_correct_responses, "n_correct_responses", allow_empty=False
        )

        # Snapshot keys: smoother + filter outputs the E-step produces, plus
        # the M-step parameters (sigma_epsilon and the initial-variance
        # derivative). On a rejected step both are restored so the stored
        # (params, smoother) pair stays consistent with the prior iteration.
        snapshot_keys = self._fit_output_attrs + (
            "sigma_epsilon",
            "init_learning_state",
            "init_learning_variance",
        )

        def _on_iteration(iteration: int, ll: float, change: float) -> None:
            # verbose=True surfaces per-iteration progress at INFO; otherwise DEBUG.
            logger.log(
                logging.INFO if verbose else logging.DEBUG,
                "Iteration %d/%d\tLog-Likelihood: %.4f\tChange: %+.4f",
                iteration + 1,
                max_iter,
                ll,
                change,
            )

        # On convergence the driver runs one more M-step (the MLE parameters)
        # and a synchronising E-step so the stored results match them.
        result = run_em(
            lambda: float(self._e_step(n_correct_responses)),
            lambda: self._m_step(n_correct_responses),
            lambda: snapshot_attributes(self, snapshot_keys),
            lambda state: restore_attributes(self, state),
            max_iter=max_iter,
            tol=tolerance,
            on_first_nonfinite="clear",
            clear_state=lambda: clear_attributes(self, self._fit_output_attrs),
            m_step_on_convergence=True,
            logger=logger,
            on_iteration=_on_iteration,
        )
        log_likelihoods = result.log_likelihoods
        if result.converged and verbose:
            logger.info("Converged. sigma_epsilon=%.4g", self.sigma_epsilon)

        self._record_fit_result(
            log_likelihoods, result.converged, n_iter=len(log_likelihoods)
        )
        self._n_trials_ = len(n_correct_responses)

        return log_likelihoods

    # --- SGDFittableMixin protocol ---

    def fit_sgd(
        self,
        n_correct_responses: ArrayLike,
        optimizer: optax.GradientTransformation | None = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: float | None = None,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        n_correct_responses : ArrayLike, shape (n_trials,)
            Number of correct responses at each trial.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
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
        return super().fit_sgd(
            n_correct_responses,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    def _prepare_sgd_data(
        self, n_correct_responses: ArrayLike
    ) -> tuple[tuple[Array], dict[str, Any]]:
        """Validate the ``fit_sgd`` responses and record the trial count.

        Runs after ``fit_sgd`` has validated its settings, so a call rejected
        for its settings or data leaves the model untouched.
        """
        n_correct_arr = jnp.asarray(n_correct_responses)
        if n_correct_arr.ndim != 1:
            raise ValueError(
                f"n_correct_responses must be 1D, got shape {n_correct_arr.shape}."
            )
        if len(n_correct_arr) < 2:
            raise ValueError(f"Need at least 2 trials, got {len(n_correct_arr)}.")
        validate_count_array(n_correct_arr, "n_correct_responses", allow_empty=False)
        resolved_max_correct = self._resolve_max_possible_correct(n_correct_arr)
        if bool(jnp.any(n_correct_arr > resolved_max_correct)):
            raise ValueError(
                "n_correct_responses contains values exceeding max_possible_correct."
            )

        self._n_trials_ = int(n_correct_arr.shape[0])
        self._resolved_max_correct = resolved_max_correct
        return (n_correct_arr,), {}

    @property
    def _n_timesteps(self) -> int:
        if not is_set(self, "_n_trials_"):
            raise NotFittedError("Model must be fitted before accessing _n_timesteps.")
        return self._n_trials_

    def _build_param_spec(self) -> tuple[SGDParams, SGDParamSpec]:
        params: SGDParams = {}
        spec: SGDParamSpec = {}

        # Process noise is always learnable
        params["sigma_epsilon"] = jnp.array(self.sigma_epsilon)
        spec["sigma_epsilon"] = POSITIVE

        # Initial state params depend on initial_state_method
        if self.initial_state_method in (
            "reestimate_initial_from_data",
            "user_provided",
        ):
            params["init_learning_state"] = jnp.array(self.init_learning_state)
            spec["init_learning_state"] = UNCONSTRAINED
            params["init_learning_variance"] = jnp.array(self.init_learning_variance)
            spec["init_learning_variance"] = POSITIVE
        # For other methods, init state is data-driven or fixed — freeze it.

        return params, spec

    def _sgd_loss_fn(self, params: SGDParams, n_correct_responses: Array) -> Array:
        sigma_eps = self._sgd_param(params, "sigma_epsilon")
        init_state = self._sgd_param(params, "init_learning_state")
        init_var = self._sgd_param(params, "init_learning_variance")

        (
            _prob_correct,
            learning_state_mode,
            learning_state_var,
            one_step_mode,
            one_step_var,
        ) = smith_learning_filter(
            n_correct_responses,
            init_learning_state=init_state,
            init_learning_variance=init_var,
            sigma_epsilon=sigma_eps,
            prob_correct_by_chance=self.prob_correct_by_chance,
            max_possible_correct=self._resolved_max_correct,
            differentiable=True,
        )

        log_likelihood_terms = smith_laplace_log_likelihood(
            n_correct_responses,
            self._resolved_max_correct,
            learning_state_mode,
            learning_state_var,
            one_step_mode,
            one_step_var,
            self.mu_bias,
        )
        return -jnp.sum(log_likelihood_terms)

    def _store_sgd_params(self, params: SGDParams) -> None:
        if "sigma_epsilon" in params:
            self.sigma_epsilon = float(params["sigma_epsilon"])
        if "init_learning_state" in params:
            self.init_learning_state = float(params["init_learning_state"])
        if "init_learning_variance" in params:
            self.init_learning_variance = float(params["init_learning_variance"])

    def _finalize_sgd(self, n_correct_responses: Array) -> float:
        # One E-step at the fitted parameters stores the filtered / smoothed
        # estimates, exactly as after EM, and returns the Laplace
        # log-likelihood (recorded as log_likelihood_).
        return self._e_step(n_correct_responses)

    def get_learning_curve(
        self,
        key: Array,
        n_samples: int = 10000,
        percentiles: ArrayLike | None = None,
        return_prob_above_chance: bool = False,
    ) -> tuple[jax.Array, jax.Array | None]:
        """
        Calculates the smoothed learning curve (probability of correct response)
        and its confidence limits.

        Must be called after `fit`.

        Parameters
        ----------
        key : Array
            JAX PRNG key for random number generation (for sampling).
        n_samples : int, optional
            Number of Monte Carlo samples to draw per trial for confidence limits.
            Default is 10000.
        percentiles : ArrayLike, optional
            Array of percentiles to compute for the probability (e.g., jnp.array([5, 50, 95])).
            If None, defaults to jnp.array([5.0, 50.0, 95.0]).
        return_prob_above_chance : bool, optional
            If True, also calculates and returns the certainty (prob_above_chance) that the true
            probability of a correct response is greater than `self.prob_correct_by_chance`.
            Default is False.

        Returns
        -------
        probability_percentiles : jnp.ndarray
            Shape (n_percentiles, n_trials). Computed percentile values for the
            probability of correct response for each trial.
        prob_above_chance : Optional[jnp.ndarray]
            Shape (n_trials,). Certainty p_k > p_chance. Returned if `return_prob_above_chance` is True.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet (i.e., smoothed estimates are not available).
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        return calculate_probability_confidence_limits(
            key=key,
            smoothed_learning_state_mode=self.smoothed_learning_state_mode,
            smoothed_learning_state_variance=self.smoothed_learning_state_variance,
            prob_correct_by_chance=self.prob_correct_by_chance,
            n_samples=n_samples,
            percentiles=percentiles,
            return_prob_above_chance=return_prob_above_chance,
        )

    def get_latent_state_percentiles(
        self,
        key: Array,
        n_samples: int = 10000,
        percentiles: ArrayLike | None = None,
    ) -> jax.Array:
        """
        Calculates confidence percentiles for the smoothed latent learning state x_k|T.

        Must be called after `fit`.

        Parameters
        ----------
        key : Array
            JAX PRNG key for random number generation.
        n_samples : int, optional
            Number of Monte Carlo samples per trial. Default is 10000.
        percentiles : ArrayLike, optional
            Percentiles to compute (e.g., jnp.array([5, 50, 95])).
            Defaults to [5.0, 50.0, 95.0].

        Returns
        -------
        jax.Array, shape (n_percentiles, n_trials)
            Computed percentile values for the latent state.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        return calculate_latent_state_percentiles(
            key=key,
            smoothed_learning_state_mode=self.smoothed_learning_state_mode,
            smoothed_learning_state_variance=self.smoothed_learning_state_variance,
            n_samples=n_samples,
            percentiles=percentiles,
        )

    def find_critical_run_length(
        self,
        sequence_length: int,
        prob_correct_by_chance: float | None = None,
        critical_probability_threshold: float = 0.05,
        min_run_length: int = 2,
        max_run_length: int = 35,
    ) -> int | None:
        """Determines the minimum length of a run of consecutive successes
        that would be statistically significant under a null hypothesis.

        This utilizes the module-level `find_min_consecutive_successes` function.

        Parameters
        ----------
        sequence_length : int
            The length of the trial sequence to consider for the criterion
            (e.g., total number of trials in an experiment block).
        prob_correct_by_chance : Optional[float], optional
            Probability of success under the null hypothesis (e.g., chance performance).
            If None, this method attempts to use the model's
            `self.prob_correct_by_chance`. Default is None.
        critical_probability_threshold : float, optional
            The critical p-value (alpha) for determining significance.
            Default is 0.05.
        min_run_length : int, optional
            Minimum run length to test. Default is 2.
        max_run_length : int, optional
            Maximum run length to test. Default is 35.

        Returns
        -------
        Optional[int]
            The minimum number of consecutive successes (`j_crit`) considered
            statistically significant. Returns None if no such run length is
            found within the specified range that meets the criterion.

        Notes
        -----
        When ``prob_correct_by_chance`` is None, uses ``self.prob_correct_by_chance``
        which is set at construction time (no fitting required).
        """
        prob_correct_by_chance_to_use: float
        if prob_correct_by_chance is None:
            prob_correct_by_chance_to_use = self.prob_correct_by_chance
            logger.info(
                f"Using prob_correct_by_chance ({prob_correct_by_chance_to_use:.3f})."
            )
        else:
            prob_correct_by_chance_to_use = prob_correct_by_chance

        return find_min_consecutive_successes(
            prob_correct_by_chance=prob_correct_by_chance_to_use,
            critical_probability_threshold=critical_probability_threshold,
            sequence_length=sequence_length,
            min_run_length=min_run_length,
            max_run_length=max_run_length,
        )

    def find_significant_runs(
        self,
        observed_binary_responses: ArrayLike,
        prob_correct_by_chance: float | None = None,
        critical_probability_threshold: float = 0.05,
        min_run_length_for_j_crit: int = 2,  # Parameter for j_crit calculation
        max_run_length_for_j_crit: int = 35,  # Parameter for j_crit calculation
    ) -> tuple[int | None, list[tuple[int, int]]]:
        """
        Identifies significant runs of successes in observed binary data.

        This method first determines a critical run length (`j_crit`) that
        is statistically unlikely to occur by chance (or a given null probability).
        It then scans the `observed_binary_responses` for all runs of successes
        (value 1) that meet or exceed this `j_crit`.

        Parameters
        ----------
        observed_binary_responses : jax.Array, shape (n_trials,)
            A 1D sequence of binary outcomes (1 for success, 0 for failure).
        prob_correct_by_chance : Optional[float], optional
            Probability of success under the null hypothesis used for determining `j_crit`.
            If None, defaults to the model's `self.prob_correct_by_chance`.
        critical_probability_threshold : float, optional
            The critical p-value (alpha) for determining `j_crit`. Default is 0.05.
        min_run_length_for_j_crit : int, optional
            Minimum run length to test when calculating `j_crit`. Default is 2.
        max_run_length_for_j_crit : int, optional
            Maximum run length to test when calculating `j_crit`. Default is 35.

        Returns
        -------
        tuple[Optional[int], list[tuple[int, int]]]
            - j_crit (Optional[int]): The determined critical run length.
            - significant_runs (list[tuple[int, int]]): A list of (start_index, end_index)
              tuples for each identified significant run of successes in the
              `observed_binary_responses`. Indices are 0-based and end_index is inclusive.

        Raises
        ------
        ValueError
            If `observed_binary_responses` is not 1D.

        Notes
        -----
        When ``prob_correct_by_chance`` is None, uses ``self.prob_correct_by_chance``
        which is set at construction time (no fitting required).
        """
        observed_binary_responses = jnp.asarray(observed_binary_responses)
        if observed_binary_responses.ndim != 1:
            raise ValueError("observed_binary_responses must be a 1D array.")
        # It's assumed observed_binary_responses contains 0s and 1s.

        sequence_length = len(observed_binary_responses)
        if sequence_length == 0:
            return None, []

        j_crit = self.find_critical_run_length(
            sequence_length=sequence_length,
            prob_correct_by_chance=prob_correct_by_chance,
            critical_probability_threshold=critical_probability_threshold,
            min_run_length=min_run_length_for_j_crit,
            max_run_length=max_run_length_for_j_crit,
        )

        significant_runs: list[tuple[int, int]] = []
        if j_crit is None:
            logger.info(
                "No critical run length (j_crit) could be determined "
                "with the given parameters. Cannot identify significant runs."
            )
            return None, significant_runs

        if j_crit > sequence_length:
            logger.info(
                f"Critical run length (j_crit={j_crit}) exceeds sequence length "
                f"({sequence_length}). No such runs possible."
            )
            return j_crit, significant_runs

        logger.info(f"Critical run length (j_crit) determined to be: {j_crit}")

        # Find all runs of successes (value 1) of length >= j_crit
        # Use the helper function _find_runs_of_value
        # Ensure observed_binary_responses is a JAX array for the helper
        significant_runs = _find_runs_of_value(
            data=observed_binary_responses,
            value_to_find=1,  # Assuming 1 represents success
            min_length=j_crit,
        )

        return j_crit, significant_runs

    def find_criterion_trial(
        self,
        key: Array,
        alpha: float = 0.05,
        n_samples: int = 10000,
    ) -> int | None:
        """Determines the first trial where learning is reliably above chance.

        Uses the Smith et al. (2004) criterion: finds the first trial k
        such that P(p_k > p_chance | y_{1:T}) >= 1 - alpha for all
        subsequent trials k' >= k. This is computed directly from the
        ``prob_above_chance`` posterior probability, not from percentile
        thresholds.

        Parameters
        ----------
        key : Array
            JAX PRNGKey for Monte Carlo sampling.
        alpha : float, optional
            Significance level. The criterion requires
            P(p_k > p_chance) >= 1 - alpha. Default is 0.05.
        n_samples : int, optional
            Number of Monte Carlo samples. Default is 10000.

        Returns
        -------
        Optional[int]
            The 0-indexed trial number of the first trial where performance
            is permanently above chance at the given significance level.
            Returns 0 if always above chance from the start.
            Returns None if the criterion is never met.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.

        Examples
        --------
        >>> model.fit(outcomes)
        >>> criterion = model.find_criterion_trial(jax.random.PRNGKey(0))
        >>> if criterion is not None:
        ...     print(f"Learning established at trial {criterion}")
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        # Compute P(p_k > p_chance | y_{1:T}) for each trial
        _, prob_above_chance = self.get_learning_curve(
            key=key,
            n_samples=n_samples,
            return_prob_above_chance=True,
        )
        assert prob_above_chance is not None

        threshold = 1.0 - alpha
        meets_criterion = jnp.asarray(prob_above_chance >= threshold)

        if bool(jnp.all(meets_criterion)):
            logger.info(
                f"P(p_k > chance) >= {threshold:.2f} for all trials. "
                f"Performance above chance from trial 0."
            )
            return 0

        if not bool(meets_criterion[-1]):
            logger.info(
                f"P(p_k > chance) < {threshold:.2f} at the last trial. "
                f"Learning criterion not met within the observed trials."
            )
            return None

        # Find the last trial that fails the criterion.
        # The criterion trial is the first trial after which the
        # condition holds permanently (Smith et al. 2004).
        fails_criterion = jnp.where(~meets_criterion)[0]
        last_fail = int(fails_criterion[-1])
        criterion_trial = last_fail + 1

        # Verify it holds for ALL subsequent trials
        if not bool(jnp.all(meets_criterion[criterion_trial:])):
            logger.info(
                "P(p_k > chance) oscillates above and below threshold; "
                "no sustained criterion trial found."
            )
            return None

        return criterion_trial

    def plot_learning_curve(
        self,
        key: Array,
        plot_type: str = "probability",
        observed_n_correct: ArrayLike | None = None,
        observed_max_possible: ArrayLike | None = None,
        confidence_bounds: tuple[float, float] = (5.0, 95.0),
        n_samples: int = 10000,
        title: str | None = None,
        xlabel: str = "Trial",
        ylabel_override: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plots the smoothed learning process with confidence intervals.

        This method visualizes either the probability of a correct response or
        the latent learning state over trials. Optionally, observed performance
        can be overlaid.

        Parameters
        ----------
        key : Array
            JAX PRNG key for random number generation, required for computing
            confidence intervals if they haven't been implicitly computed by
            prior calls that populate necessary attributes.
        plot_type : str, optional
            Type of plot to generate. Options are:
            - "probability": Plots the probability of a correct response (default).
            - "latent_state": Plots the latent learning state.
        observed_n_correct : Optional[ArrayLike], shape (n_trials,), optional
            Observed number of correct responses per trial to overlay on the plot.
            Default is None.
        observed_max_possible : Optional[ArrayLike], shape (n_trials,), optional
            Maximum possible correct responses for each trial corresponding to
            `observed_n_correct`. Required if `observed_n_correct` is provided
            and represents counts from multiple sub-trials. If `observed_n_correct`
            represents binary (0/1) outcomes, this can be omitted or set to ones.
            Default is None.
        confidence_bounds : tuple[float, float], optional
            Tuple of two floats representing the lower and upper percentile bounds
            for the confidence interval (e.g., (5.0, 95.0) for a 90% CI).
            Default is (5.0, 95.0).
        n_samples : int, optional
            Number of Monte Carlo samples used to compute confidence intervals.
            Default is 10000.
        title : Optional[str], optional
            Custom title for the plot. If None, a default title is generated.
            Default is None.
        xlabel : str, optional
            Label for the x-axis. Default is "Trial".
        ylabel_override : Optional[str], optional
            Custom label for the y-axis. If None, a default label is generated
            based on `plot_type`. Default is None.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure.
        ax : plt.Axes
            The matplotlib axes.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet (i.e., smoothed estimates
            are not available).
        ValueError
            If `plot_type` is invalid, or if `observed_n_correct` and
            `observed_max_possible` have inconsistent lengths.
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        import matplotlib.pyplot as plt

        n_trials = len(self.smoothed_learning_state_mode)
        trials_axis = jnp.arange(n_trials)

        lower_b, upper_b = sorted(confidence_bounds)
        # Percentiles for CI: lower, median (50th), upper
        plot_percentiles = jnp.array([lower_b, 50.0, upper_b])

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        current_ylabel: str

        if plot_type == "probability":
            prob_percentiles, _ = self.get_learning_curve(
                key=key,
                n_samples=n_samples,
                percentiles=plot_percentiles,
                return_prob_above_chance=False,
            )
            lower_ci = prob_percentiles[0, :]
            median_curve = prob_percentiles[1, :]
            upper_ci = prob_percentiles[2, :]
            current_ylabel = (
                ylabel_override if ylabel_override else "Probability Correct"
            )
            if title is None:
                title = "Learning Curve (Probability Correct)"
            ax.set_ylim((0, 1.05))  # Give a little space above 1.0
            # Draw chance-level reference line
            ax.axhline(
                self.prob_correct_by_chance,
                color="gray",
                linestyle="--",
                linewidth=1,
                label=f"Chance ({self.prob_correct_by_chance:.2g})",
            )
        elif plot_type == "latent_state":
            state_percentiles = self.get_latent_state_percentiles(
                key=key, n_samples=n_samples, percentiles=plot_percentiles
            )
            lower_ci = state_percentiles[0, :]
            median_curve = state_percentiles[1, :]
            upper_ci = state_percentiles[2, :]
            current_ylabel = (
                ylabel_override if ylabel_override else "Latent Learning State"
            )
            if title is None:
                title = "Learning Curve (Latent State)"
        else:
            plt.close(fig)  # Close figure if erroring out
            raise ValueError("plot_type must be 'probability' or 'latent_state'")

        ax.plot(
            trials_axis,
            median_curve,
            label=f"Smoothed Median ({plot_type.capitalize()})",
            color="blue",
            linewidth=2,
        )
        ax.fill_between(
            trials_axis,
            lower_ci,
            upper_ci,
            color="blue",
            alpha=0.2,
            label=f"{lower_b:.1f}-{upper_b:.1f}% Confidence Interval",
        )

        if observed_n_correct is not None:
            observed_n_correct = jnp.asarray(observed_n_correct)
            if len(observed_n_correct) != n_trials:
                plt.close(fig)
                raise ValueError(
                    f"observed_n_correct length ({len(observed_n_correct)}) "
                    f"must match number of trials ({n_trials})."
                )

            if observed_max_possible is not None:
                observed_max_possible = jnp.asarray(observed_max_possible)
                if len(observed_max_possible) != n_trials:
                    plt.close(fig)
                    raise ValueError(
                        f"observed_max_possible length ({len(observed_max_possible)}) "
                        f"must match number of trials ({n_trials})."
                    )
                # Avoid division by zero and ensure non-negative results
                fraction_correct = jnp.where(
                    observed_max_possible > 0,
                    jnp.clip(observed_n_correct / observed_max_possible, 0, 1),
                    jnp.nan,  # Represent invalid points as NaN
                )
                if plot_type == "probability":
                    ax.scatter(
                        trials_axis,
                        fraction_correct,
                        color="gray",
                        alpha=0.6,
                        label="Observed Fraction Correct",
                        s=20,  # size of scatter points
                        edgecolors="k",  # black edge color for points
                        linewidths=0.5,
                    )
                else:
                    warnings.warn(
                        "Observed fraction correct overlay is typically used with plot_type='probability'.",
                        StateSpaceWarning,
                        stacklevel=2,
                    )
            # If only observed_n_correct is given, assume binary if all are 0 or 1
            elif jnp.all((observed_n_correct == 0) | (observed_n_correct == 1)):
                if plot_type == "probability":
                    ax.scatter(
                        trials_axis,
                        observed_n_correct,
                        color="lightgray",  # Lighter for binary
                        alpha=0.7,
                        label="Observed Success (0/1)",
                        s=15,
                        marker="|",  # Use a different marker for binary
                    )
                else:
                    warnings.warn(
                        "Observed binary success overlay is typically used with plot_type='probability'.",
                        StateSpaceWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    "observed_n_correct provided without observed_max_possible, "
                    "and data is not strictly binary. Skipping observed data plot.",
                    StateSpaceWarning,
                    stacklevel=2,
                )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(current_ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="both", which="major", labelsize=10)

        return fig, ax

    def compare_trials(
        self,
        key: Array,
        trial1: int,
        trial2: int,
        n_samples: int = 10000,
        compare_probability: bool = False,
    ) -> float:
        """Compare two trials to determine if learning state differs.

        Computes P(x_{trial1} > x_{trial2} | y_{1:T}), the probability that
        the learning state at trial1 is greater than at trial2, given all
        observed data.

        Must be called after `fit`.

        Parameters
        ----------
        key : Array
            JAX PRNG key for Monte Carlo sampling.
        trial1 : int
            First trial index (0-based).
        trial2 : int
            Second trial index (0-based).
        n_samples : int, optional
            Number of Monte Carlo samples. Default is 10000.
        compare_probability : bool, optional
            If True, compare in probability space (sigmoid-transformed).
            If False, compare raw latent states. Default is False.

        Returns
        -------
        posterior_probability : float
            Bayesian posterior probability P(x_{trial1} > x_{trial2} | y_{1:T}).
            This is NOT a frequentist p-value.
            - Values > 0.975 indicate trial1 is significantly higher than trial2.
            - Values < 0.025 indicate trial1 is significantly lower than trial2.
            - Values near 0.5 indicate no significant difference.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        ValueError
            If trial indices are out of bounds.
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        n_trials = len(self.smoothed_learning_state_mode)
        if not (0 <= trial1 < n_trials and 0 <= trial2 < n_trials):
            raise ValueError(
                f"Trial indices must be in range [0, {n_trials - 1}]. "
                f"Got trial1={trial1}, trial2={trial2}."
            )

        prob_chance = self.prob_correct_by_chance if compare_probability else None

        return compare_two_trials(
            key=key,
            smoothed_learning_state_mode=self.smoothed_learning_state_mode,
            smoothed_learning_state_variance=self.smoothed_learning_state_variance,
            smoother_gain=self.smoother_gain,
            trial1=trial1,
            trial2=trial2,
            n_samples=n_samples,
            compare_probability=compare_probability,
            prob_correct_by_chance=prob_chance,
        )

    def get_trial_comparison_matrix(
        self,
        key: Array,
        n_samples: int = 10000,
        compare_probability: bool = False,
    ) -> Array:
        """Compute pairwise comparison matrix for all trials.

        For all pairs of trials i < j, computes P(x_i > x_j | y_{1:T}).
        This implements the trialtotrial.m functionality from the MATLAB code.

        Must be called after `fit`.

        Parameters
        ----------
        key : Array
            JAX PRNG key for Monte Carlo sampling.
        n_samples : int, optional
            Number of Monte Carlo samples per comparison. Default is 10000.
        compare_probability : bool, optional
            If True, compare in probability space. Default is False.

        Returns
        -------
        comparison_matrix : jnp.ndarray, shape (n_trials, n_trials)
            Upper triangular matrix where entry [i, j] (i < j) contains
            P(x_i > x_j | y_{1:T}). Diagonal is 0.5, lower triangle is NaN.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.

        Examples
        --------
        >>> model = SmithLearningModel()
        >>> model.fit(responses)
        >>> key = jax.random.PRNGKey(0)
        >>> matrix = model.get_trial_comparison_matrix(key)
        >>> # Check if trial 10 is significantly higher than trial 0
        >>> p_val = matrix[0, 10]
        >>> if p_val < 0.025:
        ...     print("Trial 10 is significantly higher than trial 0")
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        prob_chance = self.prob_correct_by_chance if compare_probability else None

        return compute_trial_comparison_matrix(
            key=key,
            smoothed_learning_state_mode=self.smoothed_learning_state_mode,
            smoothed_learning_state_variance=self.smoothed_learning_state_variance,
            smoother_gain=self.smoother_gain,
            n_samples=n_samples,
            compare_probability=compare_probability,
            prob_correct_by_chance=prob_chance,
        )

    def find_first_significant_improvement(
        self,
        key: Array,
        reference_trial: int = 0,
        significance_level: float = 0.05,
        n_samples: int = 10000,
        compare_probability: bool = False,
    ) -> int | None:
        """Find the first trial with significantly higher learning than reference.

        This identifies the earliest trial where the learning state is
        statistically significantly greater than a reference trial (typically
        the first trial).

        Must be called after `fit`.

        Parameters
        ----------
        key : Array
            JAX PRNG key for Monte Carlo sampling.
        reference_trial : int, optional
            The reference trial to compare against. Default is 0 (first trial).
        significance_level : float, optional
            Two-tailed significance threshold. Default is 0.05.
        n_samples : int, optional
            Number of Monte Carlo samples per comparison. Default is 10000.
        compare_probability : bool, optional
            If True, compare in probability space. Default is False.

        Returns
        -------
        first_significant : Optional[int]
            The 0-indexed trial number of the first trial significantly
            higher than the reference, or None if no such trial exists.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.

        Notes
        -----
        This corresponds to the analysis in trialtotrial.m that finds
        "Earliest trial signif above estimated start distribution".
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        comparison_matrix = self.get_trial_comparison_matrix(
            key=key,
            n_samples=n_samples,
            compare_probability=compare_probability,
        )

        return find_first_significant_trial(
            comparison_matrix=comparison_matrix,
            reference_trial=reference_trial,
            significance_level=significance_level,
        )

    def bic(self) -> float:
        """Bayesian Information Criterion for the fitted model.

        BIC = -2 * log_likelihood + k * ln(n)

        where k is the number of free parameters (sigma_epsilon, init_learning_state,
        init_learning_variance = 3) and n is the number of trials.

        Returns
        -------
        float
            BIC value. Lower is better.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if (
            not self.is_fitted
            or not is_set(self, "log_likelihood_")
            or not is_set(self, "_n_trials_")
        ):
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")
        n_params = 3  # sigma_epsilon, init_learning_state, init_learning_variance
        return -2.0 * self.log_likelihood_ + n_params * math.log(self._n_trials_)

    def compare_to_null(
        self,
        n_correct_responses: ArrayLike | None = None,
    ) -> dict[str, float | bool]:
        """Compare the fitted model to a null (no-learning) model.

        The null model assumes a constant probability of correct response
        equal to ``prob_correct_by_chance`` (i.e., sigma_epsilon=0, no
        learning state evolution). This is the simplest baseline: the
        subject performs at chance on every trial.

        Parameters
        ----------
        n_correct_responses : ArrayLike, shape (n_trials,), optional
            The observed data. If None, uses the data from the last ``fit()``
            call (requires that the model stores resolved max_possible_correct).

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"model_ll"``: Log-likelihood of the fitted model.
            - ``"null_ll"``: Log-likelihood of the null (chance) model.
            - ``"model_bic"``: BIC of the fitted model.
            - ``"null_bic"``: BIC of the null model.
            - ``"delta_bic"``: ``null_bic - model_bic``. Positive values
              favor the learning model. Values > 10 are "very strong"
              evidence (Kass & Raftery, 1995).
            - ``"learning_detected"``: True if delta_bic > 2.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        ValueError
            If ``n_correct_responses`` is not provided and cannot be inferred.
        """
        if not self.is_fitted or not is_set(self, "log_likelihood_"):
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        if n_correct_responses is None:
            raise ValueError(
                "n_correct_responses must be provided for null model comparison."
            )

        n_correct_responses = jnp.asarray(n_correct_responses)
        resolved_max = self._resolve_max_possible_correct(n_correct_responses)

        # Null model: constant probability = prob_correct_by_chance
        epsilon = 1e-9
        p_null = max(epsilon, min(self.prob_correct_by_chance, 1.0 - epsilon))
        null_ll = float(
            jnp.sum(
                jax.scipy.stats.binom.logpmf(
                    k=n_correct_responses, n=resolved_max, p=p_null
                )
            )
        )

        # BIC: null model has 0 free parameters (chance is fixed)
        model_bic = self.bic()
        null_bic = -2.0 * null_ll  # 0 params, so penalty term is 0

        delta_bic = null_bic - model_bic  # positive favors learning model

        return {
            "model_ll": self.log_likelihood_,
            "null_ll": null_ll,
            "model_bic": model_bic,
            "null_bic": null_bic,
            "delta_bic": delta_bic,
            "learning_detected": delta_bic > 2.0,
        }

    def summary(
        self,
        key: Array | None = None,
        n_correct_responses: ArrayLike | None = None,
    ) -> str:
        """Return a text summary of the fitted model.

        Parameters
        ----------
        key : Array, optional
            JAX PRNG key. If provided, computes the criterion trial.
        n_correct_responses : ArrayLike, optional
            Observed data. If provided along with ``key``, includes a null
            model comparison.

        Returns
        -------
        str
            Multi-line summary string.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        lines = [
            "SmithLearningModel Summary",
            "=" * 40,
            f"  sigma_epsilon:          {self.sigma_epsilon:.4g}",
            f"  prob_correct_by_chance:  {self.prob_correct_by_chance:.4g}",
            f"  init_learning_state:    {self.init_learning_state:.4g}",
            f"  init_learning_variance: {self.init_learning_variance:.4g}",
            f"  initial_state_method:   {self.initial_state_method}",
            "",
            f"  EM iterations:          {self.n_iter_}",
            f"  Log-likelihood:         {self.log_likelihood_:.4f}",
            f"  BIC:                    {self.bic():.4f}",
            f"  N trials:               {self._n_trials_}",
        ]

        if key is not None:
            criterion = self.find_criterion_trial(key)
            if criterion is not None:
                lines.append(f"  Criterion trial:        {criterion}")
            else:
                lines.append("  Criterion trial:        not met")

        if key is not None and n_correct_responses is not None:
            comparison = self.compare_to_null(n_correct_responses)
            lines.extend(
                [
                    "",
                    "  Null model comparison:",
                    f"    Null LL:              {comparison['null_ll']:.4f}",
                    f"    Null BIC:             {comparison['null_bic']:.4f}",
                    f"    Delta BIC:            {comparison['delta_bic']:.4f}",
                    f"    Learning detected:    {comparison['learning_detected']}",
                ]
            )

        return "\n".join(lines)

    def plot_trial_comparison_matrix(
        self,
        key: Array,
        n_samples: int = 10000,
        compare_probability: bool = False,
        significance_level: float = 0.05,
        title: str | None = None,
        cmap: str = "bone",
    ) -> tuple[Figure, Axes]:
        """Plot the trial-to-trial comparison matrix with significant points.

        Creates a heatmap visualization of pairwise trial comparisons,
        highlighting statistically significant differences. This replicates
        the visualization from trialtotrial.m.

        Must be called after `fit`.

        Parameters
        ----------
        key : Array
            JAX PRNG key for Monte Carlo sampling.
        n_samples : int, optional
            Number of Monte Carlo samples per comparison. Default is 10000.
        compare_probability : bool, optional
            If True, compare in probability space. Default is False.
        significance_level : float, optional
            Two-tailed significance threshold. Default is 0.05.
        title : Optional[str], optional
            Custom title. If None, auto-generated. Default is None.
        cmap : str, optional
            Colormap for heatmap. Default is "bone" (matches MATLAB).

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure.
        ax : plt.Axes
            The matplotlib axes.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        import matplotlib.pyplot as plt

        comparison_matrix = self.get_trial_comparison_matrix(
            key=key,
            n_samples=n_samples,
            compare_probability=compare_probability,
        )

        n_trials = comparison_matrix.shape[0]
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

        # Plot heatmap
        im = ax.imshow(comparison_matrix, cmap=cmap, vmin=0, vmax=1, origin="upper")
        plt.colorbar(im, ax=ax, label="P(trial_row > trial_col)")

        # Mark significant points using vectorized masks
        alpha = significance_level
        alpha_half = alpha / 2
        upper_mask = np.triu(np.ones((n_trials, n_trials), dtype=bool), k=1)
        mat_np = np.asarray(comparison_matrix)

        # Colorblind-safe palette (Wong 2011)
        color_higher = "#E69F00"  # orange
        color_lower = "#0072B2"  # blue

        # Significant: trial j higher than trial i
        sig_higher = upper_mask & (mat_np < alpha_half)
        rows, cols = np.where(sig_higher)
        if len(rows) > 0:
            ax.scatter(
                cols,
                rows,
                c=color_higher,
                marker="o",
                s=40,
                label="Sig. higher",
                zorder=3,
            )

        # Marginal: trial j higher
        marg_higher = upper_mask & (mat_np >= alpha_half) & (mat_np < alpha)
        rows, cols = np.where(marg_higher)
        if len(rows) > 0:
            ax.scatter(
                cols,
                rows,
                c=color_higher,
                marker="*",
                s=30,
                label="Marg. higher",
                zorder=3,
            )

        # Significant: trial i higher than trial j (unusual)
        sig_lower = upper_mask & (mat_np > 1 - alpha_half)
        rows, cols = np.where(sig_lower)
        if len(rows) > 0:
            ax.scatter(
                cols,
                rows,
                c=color_lower,
                marker="^",
                s=40,
                label="Sig. lower",
                zorder=3,
            )

        # Marginal: trial i higher
        marg_lower = upper_mask & (mat_np <= 1 - alpha_half) & (mat_np > 1 - alpha)
        rows, cols = np.where(marg_lower)
        if len(rows) > 0:
            ax.scatter(
                cols,
                rows,
                c=color_lower,
                marker="*",
                s=30,
                label="Marg. lower",
                zorder=3,
            )

        # Add diagonal line
        ax.plot([0, n_trials - 1], [0, n_trials - 1], "k-", linewidth=0.5)

        # Find first significant trial
        first_sig = find_first_significant_trial(
            comparison_matrix, reference_trial=0, significance_level=significance_level
        )

        if title is None:
            if first_sig is not None:
                title = f"Trial Comparisons (First sig. above start: {first_sig})"
            else:
                title = "Trial Comparisons (No trials sig. above start)"

        ax.set_xlabel("Trial Number", fontsize=12)
        ax.set_ylabel("Trial Number", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(-0.5, n_trials - 0.5)
        ax.set_ylim(n_trials - 0.5, -0.5)

        return fig, ax

    def plot_convergence(self) -> tuple[Figure, Axes]:
        """Plot the EM log-likelihood convergence trace.

        Must be called after ``fit()``.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure.
        ax : plt.Axes
            The matplotlib axes.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if not is_set(self, "log_likelihood_history_"):
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        iterations = range(1, len(self.log_likelihood_history_) + 1)
        ax.plot(iterations, self.log_likelihood_history_, "o-", markersize=4)
        ax.set_xlabel("EM Iteration", fontsize=12)
        ax.set_ylabel("Log-Likelihood", fontsize=12)
        ax.set_title("EM Convergence", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

        return fig, ax

    def plot_summary(
        self,
        key: Array,
        observed_n_correct: ArrayLike | None = None,
        n_samples: int = 10000,
    ) -> tuple[Figure, np.ndarray]:
        """Multi-panel diagnostic figure summarizing the fitted model.

        Creates a 3-panel figure:

        1. Learning curve (probability) with observed data and criterion trial
        2. Latent state trajectory with confidence interval
        3. EM convergence trace

        Must be called after ``fit()``.

        Parameters
        ----------
        key : Array
            JAX PRNG key for Monte Carlo sampling.
        observed_n_correct : ArrayLike, shape (n_trials,), optional
            Observed correct responses to overlay on the learning curve.
        n_samples : int, optional
            Number of Monte Carlo samples for confidence intervals.
            Default is 10000.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure.
        axes : np.ndarray of plt.Axes
            Array of axes objects for each panel.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if not self.is_fitted:
            raise NotFittedError("Model has not been fitted. Run .fit() method first.")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        n_trials = len(self.smoothed_learning_state_mode)
        trials_axis = jnp.arange(n_trials)
        plot_percentiles = jnp.array([5.0, 50.0, 95.0])

        # --- Panel 1: Learning curve (probability) ---
        ax = axes[0]
        key1, key2 = jax.random.split(key)
        prob_percentiles, _ = self.get_learning_curve(
            key=key1,
            n_samples=n_samples,
            percentiles=plot_percentiles,
        )
        ax.plot(
            trials_axis,
            prob_percentiles[1],
            color="blue",
            linewidth=2,
            label="Smoothed median",
        )
        ax.fill_between(
            trials_axis,
            prob_percentiles[0],
            prob_percentiles[2],
            color="blue",
            alpha=0.2,
            label="90% CI",
        )
        ax.axhline(
            self.prob_correct_by_chance,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Chance ({self.prob_correct_by_chance:.2g})",
        )

        if observed_n_correct is not None:
            observed_n_correct = jnp.asarray(observed_n_correct)
            if jnp.all((observed_n_correct == 0) | (observed_n_correct == 1)):
                ax.scatter(
                    trials_axis,
                    observed_n_correct,
                    color="lightgray",
                    alpha=0.7,
                    s=15,
                    marker="|",
                    label="Observed (0/1)",
                )

        # Mark criterion trial
        criterion = self.find_criterion_trial(key1, n_samples=n_samples)
        if criterion is not None and criterion > 0:
            ax.axvline(
                criterion,
                color="green",
                linestyle=":",
                linewidth=1.5,
                label=f"Criterion trial ({criterion})",
            )

        ax.set_xlabel("Trial", fontsize=11)
        ax.set_ylabel("P(Correct)", fontsize=11)
        ax.set_title("Learning Curve", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)

        # --- Panel 2: Latent state ---
        ax = axes[1]
        state_percentiles = self.get_latent_state_percentiles(
            key=key2,
            n_samples=n_samples,
            percentiles=plot_percentiles,
        )
        ax.plot(
            trials_axis,
            state_percentiles[1],
            color="blue",
            linewidth=2,
            label="Smoothed median",
        )
        ax.fill_between(
            trials_axis,
            state_percentiles[0],
            state_percentiles[2],
            color="blue",
            alpha=0.2,
            label="90% CI",
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Trial", fontsize=11)
        ax.set_ylabel("Latent State", fontsize=11)
        ax.set_title("Latent Learning State", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        # --- Panel 3: EM convergence ---
        ax = axes[2]
        if is_set(self, "log_likelihood_history_") and self.log_likelihood_history_:
            iterations = range(1, len(self.log_likelihood_history_) + 1)
            ax.plot(
                iterations,
                self.log_likelihood_history_,
                "o-",
                markersize=3,
                color="blue",
            )
        ax.set_xlabel("EM Iteration", fontsize=11)
        ax.set_ylabel("Log-Likelihood", fontsize=11)
        ax.set_title("EM Convergence", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(
            f"SmithLearningModel Summary "
            f"(\u03c3\u03b5={self.sigma_epsilon:.3g}, "
            f"BIC={self.bic():.1f})",
            fontsize=14,
            fontweight="bold",
        )

        return fig, axes
