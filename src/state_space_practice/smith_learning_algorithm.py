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

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np


def approximate_gaussian(
    log_posterior_func: Callable, x0: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate the posterior distribution using Laplace approximation
    to find the mode and covariance matrix.

    This function finds the mode of the posterior distribution and
    computes the covariance matrix using the Hessian of the negative
    log posterior.

    Parameters
    ----------
    log_posterior_func : Callable
        A function that computes the log posterior distribution.
        It should take a single argument x and return a scalar value
    x0 : jnp.ndarray, shape (n,)
        Initial guess for the mode of the posterior distribution.

    Returns
    -------
    mode: jnp.ndarray, shape (n,)
        The mode of the posterior distribution.
    covariance: jnp.ndarray, shape (n, n)
        The covariance matrix of the posterior distribution.

    """
    # neg_log_posterior = lambda x: -log_posterior_func(x) / len(x)
    neg_log_posterior = lambda x: -log_posterior_func(x)
    mode = jax.scipy.optimize.minimize(fun=neg_log_posterior, x0=x0, method="BFGS").x
    hessian = jax.hessian(neg_log_posterior)(mode)
    try:
        covariance = jnp.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        try:
            covariance = jnp.linalg.pinv(hessian)
        except np.linalg.LinAlgError:
            return None, None

    return mode, covariance


@jax.jit
def log_posterior_objective(
    learning_state: float,
    learning_state_prev: float,
    sig_sq_old: float,
    n_correct_in_trial: jnp.ndarray,
    max_possible_correct: int,
    bias: float,
):
    """Objective function for the log posterior distribution.

    Parameters
    ----------
    learning_state : float
        Current latent learning state estimate, x_k
    learning_state_prev : float
        Previous latent learning state estimate vector, x_{k-1}
    sig_sq_old : float
        Previous state variance, \sigma_{k-1}^2
    n_correct_in_trial : jnp.ndarray, shape (n,)
        Number of correct responses in the trial
    max_possible_correct : int
        Maximum number of correct responses in the trial
    bias : float
        Bias term for the observation model

    Returns
    -------
    log_posterior : jnp.ndarray, shape (n,)
        Log posterior of the state estimate vector
    """
    prob_sucess = jax.nn.sigmoid(
        bias + learning_state
    )  # can add in covariates @ learning_state
    log_likelihood = jax.scipy.stats.binom.logpmf(
        k=n_correct_in_trial, n=max_possible_correct, p=prob_sucess
    )
    log_prior = jax.scipy.stats.norm.logpdf(
        x=learning_state,
        loc=learning_state_prev,
        scale=jnp.sqrt(sig_sq_old),
    )

    return jnp.squeeze(log_likelihood + log_prior)


def smith_learning_filter(
    n_correct_responses: jnp.ndarray,
    init_learning_state: float = 0.0,
    init_learning_variance: Optional[float] = None,
    sigma_epsilon: float = jnp.sqrt(0.05),
    prob_correct_by_chance: float = 0.5,
    max_possible_correct: Optional[int] = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    init_learning_variance : float
        Initial learning state variance ($P_0$). Defaults to $\sigma_\epsilon^2$.
        Controls how fast the learning state is updated.
    sigma_epsilon : float, optional
        Standard deviation of the process noise ($\sigma_\epsilon$),
    prob_correct_by_chance : float, optional
        The probability of a correct response by chance in absence of any
        learning or experience, used to set the bias.
        Default is 0.5.
    max_possible_correct : float, optional
        Maximum number of correct responses in each trial ($N_k$).
        When None, it is set to the maximum value in `n_correct`.

    Returns
    -------
    prob_correct_response : jnp.ndarray, shape (n_trials,)
        Posterior probability of a correct response ($p_k$).
    learning_state_mode : jnp.ndarray, shape (n_trials,)
        Posterior mode of the learning state ($x_{k|k}$).
    learning_state_variance : jnp.ndarray, shape (n_trials,)
        Posterior variance of the learning state ($P_{k|k}$).
    one_step_mode : jnp.ndarray, shape (n_trials,)
        One-step prediction of the learning state in each trial ($x_{k|k-1}$).
    one_step_variance : jnp.ndarray, shape (n_trials,)
        One-step prediction of the variance of the learning state in each trial ($P_{k|k-1}$).
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

    # @jax.jit
    def _step(
        params_prev: tuple[float, float], args: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
        """A single step of the non-linear filter."""
        mode_prev, variance_prev = params_prev
        n_correct_trial_k, max_possible_correct_trial_k = args

        # 1. Prediction Step
        one_step_mode = mode_prev  # no transition matrix
        one_step_variance = variance_prev + sigma_squared_epsilon

        # 2. Update Step (using Laplace Approximation)
        log_objective_func = partial(
            log_posterior_objective,
            learning_state_prev=one_step_mode,
            sig_sq_old=one_step_variance,
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
    learning_state_mode, learning_state_variance, one_step_mode, one_step_variance = (
        jax.lax.scan(
            _step,
            (init_learning_state, init_learning_variance),
            (n_correct_responses, max_possible_correct),
        )[1]
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
    filtered_learning_state_mode: jnp.ndarray,
    filtered_learning_state_variance: jnp.ndarray,
    one_step_mode: jnp.ndarray,
    one_step_variance: jnp.ndarray,
    prob_correct_by_chance: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Smooth the filtered learning state estimates using the Kalman smoother.

    Parameters
    ----------
    filtered_learning_state_mode : jnp.ndarray, shape (n_trials,)
        Learning state mode estimates from the filter
    filtered_learning_state_variance : jnp.ndarray, shape (n_trials,)
        Learning state variance estimates from the filter
    one_step_mode : jnp.ndarray, shape (n_trials,)
        One-step prediction of the learning state
    one_step_variance : jnp.ndarray, shape (n_trials,)
        One-step prediction of the variance of the learning state
    prob_correct_by_chance : float, optional
        The probability of a correct response by chance in absence of any
        learning or experience.
        Default is 0.5.

    Returns
    -------
    learning_state_mode : jnp.ndarray, shape (n_trials,)
        Smoothed learning state mode estimates
    learning_state_variance : jnp.ndarray, shape (n_trials,)
        Smoothed learning state variance estimates
    prob_correct_response : jnp.ndarray, shape (n_trials,)
        Smoothed probability of a correct response
    smoother_gain : jnp.ndarray, shape (n_trials,)
        Smoother gain estimates
    """

    def _step(params_prev, k):
        mode_smoothed_prev, variance_smoothed_prev = params_prev
        smoother_gain = (
            filtered_learning_state_variance[k] / one_step_variance[k + 1]
        )  # smoother gain, A_k
        mode_smoothed = filtered_learning_state_mode[k] + smoother_gain * (
            mode_smoothed_prev - one_step_mode[k + 1]
        )
        variance_smoothed = filtered_learning_state_variance[k] + smoother_gain**2 * (
            variance_smoothed_prev - one_step_variance[k + 1]
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
    (_, _), (learning_state_mode, learning_state_variance, smoother_gain) = (
        jax.lax.scan(
            _step,
            init_params,
            jnp.arange(n_trials - 1),
            reverse=True,
        )
    )

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
