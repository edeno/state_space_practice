"""Bayesian State-Space Model for Learning using Laplace Approximation.

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
- `approximate_gaussian`: Computes the Laplace approximation.
- `_log_posterior_objective`: Defines the log-posterior for a single step.
- `smith_learning_filter`: Implements the forward-pass filter.
- `smith_learning_smoother`: Implements a backward-pass RTS smoother.

This implementation leverages JAX for automatic differentiation (Hessian),
optimization (BFGS), and efficient vectorized/scanned operations.

"""

import warnings
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.optimize


def approximate_gaussian(
    log_posterior_func: Callable, x0: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate the posterior using Laplace approximation.

    Finds the mode and covariance matrix using the Hessian of the
    negative log posterior. Adds regularization to the Hessian before
    inversion for numerical stability.

    Parameters
    ----------
    log_posterior_func : Callable
        Function computing the log posterior distribution.
        Takes one argument (state) and returns a scalar.
    x0 : jnp.ndarray, shape (1,)
        Initial guess for the mode.

    Returns
    -------
    mode : jnp.ndarray, shape (1,)
        The mode of the posterior distribution.
    covariance : jnp.ndarray, shape (1, 1)
        The covariance matrix (approximated) of the posterior distribution.

    Raises
    ------
    RuntimeError
        If the optimization fails to converge.
    """
    neg_log_posterior = lambda x: -log_posterior_func(x)

    # Find the mode using BFGS optimization
    result = jax.scipy.optimize.minimize(fun=neg_log_posterior, x0=x0, method="BFGS")

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    mode = result.x
    hessian = jax.hessian(neg_log_posterior)(mode)

    # Add regularization for numerical stability
    reg = 1e-6
    try:
        covariance = jnp.linalg.inv(hessian + jnp.eye(hessian.shape[0]) * reg)
    except jnp.linalg.LinAlgError:
        warnings.warn("Hessian inversion failed, using pseudo-inverse.", RuntimeWarning)
        covariance = jnp.linalg.pinv(hessian + jnp.eye(hessian.shape[0]) * reg)

    return mode, covariance


@partial(jax.jit, static_argnames=("max_possible_correct", "bias"))
def _log_posterior_objective(
    learning_state: jax.Array,
    learning_state_prev: float,
    variance_prev: float,
    n_correct_in_trial: int,
    max_possible_correct: int,
    bias: float,
) -> jax.Array:
    """Objective function for the log posterior distribution at one step.

    Parameters
    ----------
    learning_state : jnp.ndarray, shape (1,)
        Current latent learning state estimate, $x_k$.
    learning_state_prev : float
        Previous latent learning state estimate, $x_{k-1}$.
    variance_prev : float
        Previous state variance, $P_{k|k-1}$.
    n_correct_in_trial : int
        Number of correct responses in the trial, $y_k$.
    max_possible_correct : int
        Maximum number of correct responses, $N_k$.
    bias : float
        Bias term ($\mu$) for the observation model.

    Returns
    -------
    log_posterior : jnp.ndarray, shape (n,)
        Log posterior of the state estimate vector
    """
    prob_success = jax.nn.sigmoid(
        bias + learning_state
    )  # can add in covariates @ learning_state
    log_likelihood = jax.scipy.stats.binom.logpmf(
        k=n_correct_in_trial, n=max_possible_correct, p=prob_success
    )
    log_prior = jax.scipy.stats.norm.logpdf(
        x=learning_state, loc=learning_state_prev, scale=jnp.sqrt(variance_prev)
    )

    return jnp.squeeze(log_likelihood + log_prior)


def smith_learning_filter(
    n_correct_responses: jax.Array,
    init_learning_state: float = 0.0,
    init_learning_variance: Optional[float] = None,
    sigma_epsilon: float = jnp.sqrt(0.05),
    prob_correct_by_chance: float = 0.5,
    max_possible_correct: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Applies a non-linear Bayesian filter (Laplace approximation) for learning.

    Assumes a random walk model for the latent learning state ($x_k$) and
    a Binomial observation model with a sigmoid link function.

    $$ x_k = x_{k-1} + w_k, \quad w_k \sim N(0, \sigma_\epsilon^2) $$
    $$ p_k = \frac{1}{1 + \exp(-(\mu + x_k))} $$
    $$ y_k \sim \text{Binomial}(N_k, p_k) $$

    Parameters
    ----------
    n_correct_responses : jnp.ndarray, shape (n_trials,)
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
    max_possible_correct : float, optional
        Maximum number of correct responses in each trial ($N_k$).
        Defaults to max(n_correct_responses).

    Returns
    -------
    prob_correct_response : jnp.ndarray, shape (n_trials,)
        Posterior probability of a correct response ($p_k$).
    learning_state_mode : jnp.ndarray, shape (n_trials,)
        Posterior mode of the learning state ($x_{k|k}$).
    learning_state_variance : jnp.ndarray, shape (n_trials,)
        Posterior variance of the learning state ($P_{k|k}$).
    one_step_mode : jnp.ndarray, shape (n_trials,)
        One-step prediction mode ($x_{k|k-1}$).
    one_step_variance : jnp.ndarray, shape (n_trials,)
        One-step prediction variance ($P_{k|k-1}$).
    """
    # Bias term based on chance probability of correct response
    mu: float = jnp.log(prob_correct_by_chance / (1 - prob_correct_by_chance))
    sigma_squared_epsilon: float = sigma_epsilon**2

    # Set initial variance if not provided
    if init_learning_variance is None:
        init_learning_variance = sigma_squared_epsilon

    if isinstance(max_possible_correct, (int, float)):
        max_possible_correct = jnp.array(
            [max_possible_correct] * len(n_correct_responses), dtype=int
        )

    if max_possible_correct is None:
        max_possible_correct = (
            jnp.ones_like(n_correct_responses, dtype=int) * n_correct_responses.max()
        )

    @jax.jit
    def _step(
        carry: Tuple[float, float], trial_data: Tuple[int, int]
    ) -> Tuple[Tuple[float, float], Tuple[float, float, float, float]]:
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
        # Find mode and covariance (variance)
        posterior_mode, posterior_variance = approximate_gaussian(
            log_objective_func, x0=jnp.array([one_step_mode])
        )
        posterior_mode = jnp.squeeze(posterior_mode)
        posterior_variance = jnp.squeeze(posterior_variance)

        return (posterior_mode, posterior_variance), (
            posterior_mode,
            posterior_variance,
            one_step_mode,
            one_step_variance,
        )

    # Run the filter over all trials
    init_carry = (init_learning_state, init_learning_variance)
    inputs = (n_correct_responses, max_possible_correct)
    _, output = jax.lax.scan(
        _step,
        init_carry,
        inputs,
    )
    (learning_state_mode, learning_state_variance, one_step_mode, one_step_variance) = (
        output
    )

    # Compute probability of correct response
    prob_correct_response = jax.nn.sigmoid(mu + learning_state_mode)

    return (
        prob_correct_response,
        learning_state_mode,
        learning_state_variance,
        one_step_mode,
        one_step_variance,
    )


def smith_learning_smoother(
    filtered_learning_state_mode: jax.Array,
    filtered_learning_state_variance: jax.Array,
    one_step_mode: jax.Array,
    one_step_variance: jax.Array,
    prob_correct_by_chance: float = 0.5,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Smooth the filtered learning state estimates using the Kalman smoother.

    Parameters
    ----------
    filtered_learning_state_mode : jax.Array, shape (n_trials,)
        Learning state mode estimates from the filter
    filtered_learning_state_variance : jax.Array, shape (n_trials,)
        Learning state variance estimates from the filter
    one_step_mode : jax.Array, shape (n_trials,)
        One-step prediction of the learning state
    one_step_variance : jax.Array, shape (n_trials,)
        One-step prediction of the variance of the learning state
    prob_correct_by_chance : float, optional
        The probability of a correct response by chance in absence of any
        learning or experience.
        Default is 0.5.

    Returns
    -------
    learning_state_mode : jax.Array, shape (n_trials,)
        Smoothed learning state mode estimates
    learning_state_variance : jax.Array, shape (n_trials,)
        Smoothed learning state variance estimates
    prob_correct_response : jax.Array, shape (n_trials,)
        Smoothed probability of a correct response
    smoother_gain : jax.Array, shape (n_trials,)
        Smoother gain estimates
    """
    n_trials: int = len(filtered_learning_state_mode)
    mu: float = jnp.log(prob_correct_by_chance / (1 - prob_correct_by_chance))

    @jax.jit
    def _step(
        carry: Tuple[float, float], k: int
    ) -> Tuple[Tuple[float, float], Tuple[float, float, float]]:
        """A single JIT-compiled step of the RTS smoother."""
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
    n_trials = len(filtered_learning_state_mode)
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

    mu = jnp.log(prob_correct_by_chance / (1 - prob_correct_by_chance))
    prob_correct_response = jax.nn.sigmoid(mu + learning_state_mode)

    return (
        learning_state_mode,
        learning_state_variance,
        prob_correct_response,
        smoother_gain,
    )


def maximization_step(
    smoothed_mode: jax.Array,
    smoothed_variance: jax.Array,
    smoother_gain: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Estimate process noise from smoothed estimates.

    Parameters
    ----------
    smoothed_mode : jax.Array, shape (n_trials,)
        Smoothed learning state mode estimates.
    smoothed_variance : jax.Array, shape (n_trials,)
        Smoothed learning state variance estimates.
    smoother_gain : jax.Array, shape (n_trials,)
        Smoother gain estimates.

    Returns
    -------
    sigma_epsilon : jax.Array, shape (n_trials,)
        Estimated process noise standard deviation.
    init_learning_state : jax.Array, shape (1,)
        Initial learning state estimate.
    init_learning_variance : jax.Array, shape (1,)
        Initial learning state variance estimate.
    """
    n_trials: int = len(smoothed_mode)
    expected_squared_diff_terms = (
        (smoothed_mode[1:] - smoothed_mode[:-1]) ** 2
        + smoothed_variance[1:]
        + smoothed_variance[:-1]
        - 2.0 * smoothed_variance[:-1] * smoother_gain
    )

    sigma_epsilon_sq = jnp.sum(expected_squared_diff_terms) / (n_trials - 1)
    sigma_epsilon = jnp.sqrt(sigma_epsilon_sq)

    init_learning_state = smoothed_mode[0]
    init_learning_variance = smoothed_variance[0]

    return sigma_epsilon, init_learning_state, init_learning_variance
