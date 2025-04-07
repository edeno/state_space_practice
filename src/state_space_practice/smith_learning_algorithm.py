from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np


def approximate_gaussian(log_posterior_func: Callable, x0: jnp.ndarray) -> jnp.ndarray:
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
    n_correct_in_trial: bool,
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
    n_correct_in_trial : bool
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
    init_learning_variance: float = None,
    sigma_epsilon: float = jnp.sqrt(0.05),
    prob_correct_by_chance: float = 0.5,
    max_possible_correct: int = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """

    Parameters
    ----------
    n_correct_responses : jnp.ndarray, shape (n_trials,)
        Number of correct responses in each trial
    init_learning_state : float, optional
        The subject's learning state at the beginning of the experiment.
        When None, it is set to 0.
    init_learning_variance : float
        Initial variance of the learning state.
        Controls how fast the learning state is updated.
        When None, it is set to `sigma_epsilon`.
    sigma_epsilon : float, optional
        Standard deviation of the noise in the observation model.
    prob_correct_by_chance : float, optional
        The probability of a correct response by chance in absence of any
        learning or experience.
    max_possible_correct : float, optional
        Maximum number of correct responses in each trial.
        When None, it is set to the maximum value in `n_correct`.

    Returns
    -------
    prob_correct_response : jnp.ndarray, shape (n_trials,)
        Probability of a correct response in each trial
    learning_state_mode : jnp.ndarray, shape (n_trials,)
        Posterior mode of the learning state in each trial
    learning_state_variance : jnp.ndarray, shape (n_trials,)
        Posterior variance of the learning state in each trial
    one_step_mode : jnp.ndarray, shape (n_trials,)
        One-step prediction of the learning state in each trial
    one_step_variance : jnp.ndarray, shape (n_trials,)
        One-step prediction of the variance of the learning state in each trial
    """
    # Initial mode and variance
    mu = jnp.log(prob_correct_by_chance / (1 - prob_correct_by_chance))
    sigma_squared_epsilon = sigma_epsilon**2

    # @jax.jit
    def _step(
        params_prev: tuple[float, float], args: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:

        mode_prev, variance_prev = params_prev
        n_correct_trial_k, max_possible_correct_trial_k = args

        # one step prediction
        one_step_mode = mode_prev  # no transition matrix
        one_step_variance = variance_prev + sigma_squared_epsilon

        log_objective_func = partial(
            log_posterior_objective,
            learning_state_prev=one_step_mode,
            sig_sq_old=one_step_variance,
            n_correct_in_trial=n_correct_trial_k,
            max_possible_correct=max_possible_correct_trial_k,
            bias=mu,
        )
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

    if max_possible_correct is None:
        max_possible_correct = jnp.ones_like(n_correct) * n_correct.max()

    if init_learning_variance is None:
        init_learning_variance = sigma_squared_epsilon

    learning_state_mode, learning_state_variance, one_step_mode, one_step_variance = (
        jax.lax.scan(
            _step,
            (init_learning_state, init_learning_variance),
            (n_correct_responses, max_possible_correct),
        )[1]
    )

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
