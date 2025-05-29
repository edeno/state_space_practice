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

import logging
import warnings
from functools import partial
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np
import scipy.special

from state_space_practice.utils import check_converged

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def approximate_gaussian(
    log_posterior_func: Callable, x0: jnp.ndarray
) -> Tuple[jax.Array, jax.Array]:
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
    smoother_gain : jax.Array, shape (n_trials - 1,)
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


def calculate_probability_confidence_limits(
    key: jax.random.PRNGKey,
    smoothed_mode: jnp.ndarray,  # shape: (n_trials,)
    smoothed_variance: jnp.ndarray,  # shape: (n_trials,)
    mu_bias: float,
    n_samples: int = 10000,
    percentiles: jnp.ndarray = None,
    prob_correct_by_chance: Optional[float] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Calculates confidence limits for the probability of a correct response.

    This is achieved by sampling from the smoothed posterior distribution of
    the learning state for each trial and transforming these samples through
    the sigmoid link function.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX PRNG key for random number generation.
    smoothed_mode : jnp.ndarray, shape (n_trials,)
        Smoothed learning state means (x_{k|T}).
    smoothed_variance : jnp.ndarray, shape (n_trials,)
        Smoothed learning state variances (P_{k|T}).
    mu_bias : float
        Bias term (mu) in the sigmoid function: p_k = sigmoid(mu + x_k).
    n_samples : int, optional
        Number of Monte Carlo samples to draw per trial. Default is 10000.
    percentiles : jnp.ndarray, optional
        Array of percentiles to compute (e.g., jnp.array([5, 50, 95])).
        If None, defaults to jnp.array([5.0, 50.0, 95.0]).
    prob_correct_by_chance : Optional[float], optional
        If provided, computes the certainty (pcert) that the true probability
        of a correct response is greater than this chance level. Default is None.

    Returns
    -------
    probability_percentiles : jnp.ndarray, shape (n_percentiles, n_trials)
        The computed percentile values for the probability of correct response
        for each trial.
    pcert : Optional[jnp.ndarray], shape (n_trials,)
        The certainty that p_k > prob_correct_by_chance for each trial.
        Returned if prob_correct_by_chance is not None.
    """
    if percentiles is None:
        percentiles = jnp.array([5.0, 50.0, 95.0])

    n_trials = smoothed_mode.shape[0]

    epsilon = 1e-9
    smoothed_std_dev = jnp.sqrt(jnp.maximum(smoothed_variance, epsilon))

    # Function to process a single trial
    def process_trial(key_trial, mode_k, std_dev_k):
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

        # Calculate pcert if requested
        if prob_correct_by_chance is not None:
            trial_pcert = jnp.mean(prob_samples > prob_correct_by_chance)
            return trial_percentiles, trial_pcert
        else:
            return trial_percentiles, None  # Or jnp.nan if a consistent shape is needed

    # Generate per-trial PRNG keys
    trial_keys = jax.random.split(key, n_trials)
    probability_percentiles = jax.vmap(process_trial)(
        trial_keys, smoothed_mode, smoothed_std_dev
    )

    if prob_correct_by_chance is not None:
        return probability_percentiles[0].T, probability_percentiles[1]
    else:
        return probability_percentiles[0].T, None


def find_min_consecutive_successes(
    prob_success_null: float,
    critical_probability_threshold: float,
    sequence_length: int,
    min_run_length: int = 2,
    max_run_length: int = 35,
) -> Optional[int]:
    """
    Finds the minimum number of consecutive successes (run_length) in a sequence of
    `sequence_length` Bernoulli trials (with success probability `prob_success_null`)
    such that the probability of observing at least one such run is less
    than `critical_probability_threshold`.

    This logic closely follows the MATLAB findj.m implementation, which is
    based on methods for calculating run probabilities.

    Parameters
    ----------
    prob_success_null : float
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
    if not (0 < prob_success_null < 1):
        raise ValueError("prob_success_null must be between 0 and 1.")
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
            # A run of current_run_length cannot occur in sequence_length trials
            total_prob_at_least_one_run = 0.0
        else:
            # prob_first_run_ends_at_idx[i] stores P(first run of length `current_run_length`
            # ends at a trial corresponding to index i in this array).
            # The length of this array is `sequence_length - current_run_length + 1`,
            # representing the number of possible ending positions for the *first* run.
            n_possible_ending_positions = sequence_length - current_run_length + 1
            prob_first_run_ends_at_idx = jnp.zeros(n_possible_ending_positions)

            # Probability of run of length `current_run_length`
            prob_run_occurs = prob_success_null**current_run_length

            # Base case: first run ends at trial `current_run_length`
            # (corresponds to index 0 in prob_first_run_ends_at_idx)
            prob_first_run_ends_at_idx = prob_first_run_ends_at_idx.at[0].set(
                prob_run_occurs
            )

            # Case 1: Short sequence (sequence_length <= 2 * current_run_length)
            if sequence_length <= 2 * current_run_length:
                if n_possible_ending_positions > 1:
                    # For runs ending at trial k > current_run_length,
                    # they must be preceded by a failure.
                    # P(F S...S) = (1-p)p^j
                    prob_first_run_ends_at_idx = prob_first_run_ends_at_idx.at[1:].set(
                        prob_run_occurs * (1 - prob_success_null)
                    )
            # Case 2: Longer sequence (sequence_length > 2 * current_run_length)
            else:
                # For runs ending at trials > current_run_length and <= 2*current_run_length
                # (indices 1 to current_run_length in prob_first_run_ends_at_idx)
                # Example: if current_run_length is 3:
                # f[0] for run ending at trial 3 (SSS)
                # f[1] for run ending at trial 4 (FSSS)
                # f[2] for run ending at trial 5 (XFSSS)
                # f[current_run_length] for run ending at trial 2*current_run_length

                # Max index for this simple assignment part:
                # min(current_run_length, n_possible_ending_positions - 1)
                # This covers indices 1 up to current_run_length
                idx_simple_end = min(
                    current_run_length, n_possible_ending_positions - 1
                )
                if idx_simple_end >= 1:  # Ensure slice is valid
                    prob_first_run_ends_at_idx = prob_first_run_ends_at_idx.at[
                        1 : idx_simple_end + 1
                    ].set(prob_run_occurs * (1 - prob_success_null))

                # Recursive part for runs ending at trials > 2*current_run_length
                # Loop for current_f_idx from current_run_length + 1
                # up to n_possible_ending_positions - 1
                # This corresponds to runs ending at actual trial numbers > 2*current_run_length
                for current_f_idx in range(
                    current_run_length + 1, n_possible_ending_positions
                ):
                    # Sum P(first run ends at k) for k from current_run_length up to
                    # trial_index_of_current_f - current_run_length - 1.
                    # In terms of prob_first_run_ends_at_idx indices:
                    # Sum f[0]...f[current_f_idx - current_run_length - 1]
                    sum_limit_exclusive = current_f_idx - current_run_length

                    # Ensure sum_limit_exclusive is not negative for slicing
                    # Though sum of empty slice is 0.
                    sum_prev_f_values = jnp.sum(
                        prob_first_run_ends_at_idx[0:sum_limit_exclusive]
                    )

                    new_f_value = (
                        prob_run_occurs
                        * (1 - prob_success_null)
                        * (1 - sum_prev_f_values)
                    )
                    prob_first_run_ends_at_idx = prob_first_run_ends_at_idx.at[
                        current_f_idx
                    ].set(new_f_value)

            total_prob_at_least_one_run = jnp.sum(prob_first_run_ends_at_idx)

        if total_prob_at_least_one_run < critical_probability_threshold:
            return current_run_length

    return None  # No run_length in the specified range met the criterion


def simulate_learning_data(
    n_trials: int = 50,
    prob_success_init: float = 0.125,
    prob_success_final: float = 0.6,
    learning_rate: float = 0.2,
    inflection_point: float = 25.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulates learning data with a sigmoid probability curve.

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
) -> List[Tuple[int, int]]:
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
    List[Tuple[int, int]]
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


def calculate_latent_state_percentiles(
    key: jax.random.PRNGKey,
    smoothed_mode: jnp.ndarray,  # shape: (n_trials,)
    smoothed_variance: jnp.ndarray,  # shape: (n_trials,)
    n_samples: int = 10000,
    percentiles: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Calculates confidence percentiles for the smoothed latent state.

    Samples from the smoothed posterior distribution of the learning state
    N(x_k|T, P_k|T) for each trial.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX PRNG key for random number generation.
    smoothed_mode : jnp.ndarray, shape (n_trials,)
        Smoothed learning state means (x_{k|T}).
    smoothed_variance : jnp.ndarray, shape (n_trials,)
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

    n_trials = smoothed_mode.shape[0]
    epsilon = 1e-9  # For numerical stability if variance is tiny
    smoothed_std_dev = jnp.sqrt(jnp.maximum(smoothed_variance, epsilon))

    def process_trial_state(key_trial, mode_k, std_dev_k):
        latent_state_samples = mode_k + std_dev_k * jax.random.normal(
            key_trial, shape=(n_samples,)
        )
        return jnp.percentile(latent_state_samples, percentiles)

    trial_keys = jax.random.split(key, n_trials)
    # Use vmap for efficient per-trial processing
    # mapped_results will have shape (n_trials, n_percentiles)
    mapped_results = jax.vmap(process_trial_state)(
        trial_keys, smoothed_mode, smoothed_std_dev
    )
    # Transpose to get (n_percentiles, n_trials)
    return mapped_results.T


class SmithLearningAlgorithm:

    def __init__(
        self,
        init_learning_state: float = 0.0,
        init_learning_variance: Optional[float] = None,
        sigma_epsilon: float = jnp.sqrt(0.05),
        prob_correct_by_chance: float = 0.5,
        max_possible_correct: Optional[int] = None,
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

        if init_learning_variance is None:
            self.init_learning_variance = sigma_epsilon**2

        else:
            raise TypeError(
                "init_learning_variance must be a non-negative float or None."
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

        self.init_learning_state = float(init_learning_state)
        self.init_learning_variance = float(init_learning_variance)

        self.sigma_epsilon = sigma_epsilon
        self.prob_correct_by_chance = prob_correct_by_chance
        self.max_possible_correct = max_possible_correct
        self.mu_bias = self._calculate_mu_bias(self.prob_correct_by_chance)

        self.init_state_method = initial_state_method

        # Attributes to store filter/smoother outputs
        self.filtered_prob_correct_response: Optional[jax.Array] = None
        self.filtered_learning_state_mode: Optional[jax.Array] = None
        self.filtered_learning_state_variance: Optional[jax.Array] = None
        self.filtered_one_step_mode: Optional[jax.Array] = None
        self.filtered_one_step_variance: Optional[jax.Array] = None

        self.smoothed_learning_state_mode: Optional[jax.Array] = None
        self.smoothed_learning_state_variance: Optional[jax.Array] = None
        self.smoothed_prob_correct_response: Optional[jax.Array] = None
        self.smoother_gain: Optional[jax.Array] = None  # Has shape (n_trials-1,)

        self._max_possible_correct_val = max_possible_correct  # Store initial config
        self._is_max_possible_correct_resolved = isinstance(
            max_possible_correct, (np.ndarray, jax.Array)
        ) or (isinstance(max_possible_correct, int) and max_possible_correct > 0)

    def _calculate_mu_bias(self, prob_chance: float) -> float:
        """Converts probability of chance performance to mu bias term."""
        # Ensure prob_chance is not exactly 0 or 1 to avoid log(0) or division by zero
        epsilon = 1e-9  # Small epsilon
        prob_chance_clipped = jnp.clip(prob_chance, epsilon, 1.0 - epsilon)
        return jnp.log(prob_chance_clipped / (1.0 - prob_chance_clipped))

    def _resolve_max_possible_correct(
        self, n_correct_responses: jax.Array
    ) -> jax.Array:
        """
        Resolves max_possible_correct to an array if it's None or scalar.
        Stores it in self._max_possible_correct_val.
        """
        if (
            self._is_max_possible_correct_resolved
            and self._max_possible_correct_val is not None
        ):
            if isinstance(self._max_possible_correct_val, int):
                return jnp.array(
                    [self._max_possible_correct_val] * len(n_correct_responses),
                    dtype=jnp.int32,
                )
            # If already an array (checked in init or resolved before)
            if isinstance(self._max_possible_correct_val, (np.ndarray, jax.Array)):
                if len(self._max_possible_correct_val) != len(n_correct_responses):
                    raise ValueError(
                        "Provided max_possible_correct array has inconsistent length with n_correct_responses."
                    )
                return jnp.asarray(self._max_possible_correct_val, dtype=jnp.int32)

        # If None, infer from data (assuming constant N_k)
        val = jnp.max(n_correct_responses)
        if val <= 0:  # handle case where all n_correct_responses are 0
            logger.warning(
                "All n_correct_responses are 0 or less; max_possible_correct inferred as 1."
            )
            val = 1
        resolved_array = jnp.full_like(n_correct_responses, val, dtype=jnp.int32)
        self._max_possible_correct_val = resolved_array  # Store the array version
        self._is_max_possible_correct_resolved = True
        logger.info(
            f"max_possible_correct was not provided or was scalar; "
            f"resolved to constant value {val} for all trials."
        )
        return resolved_array

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
            The marginal log-likelihood of the observed data.
        """
        resolved_trial_max_correct = self._resolve_max_possible_correct(
            n_correct_responses
        )
        (
            self.prob_correct_response,
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

        prob_pred_success = jax.nn.sigmoid(self.mu_bias + self.filtered_one_step_mode)
        # Clip probabilities to avoid logpmf errors with values exactly 0 or 1
        epsilon = 1e-9
        prob_pred_success = jnp.clip(prob_pred_success, epsilon, 1.0 - epsilon)

        log_likelihood_terms = jax.scipy.stats.binom.logpmf(
            k=n_correct_responses,
            n=resolved_trial_max_correct,
            p=prob_pred_success,
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

        return float(log_likelihood) if log_likelihood is not None else None

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
        (
            self.sigma_epsilon,
            new_init_learning_state,
            new_init_learning_variance,
        ) = maximization_step(
            self.smoothed_learning_state_mode,
            self.smoothed_learning_state_variance,
            self.smoother_gain,
        )

        # --- Apply init_state_method for initial states ---
        if self.init_state_method == "reestimate_initial_from_data":
            self.init_learning_state = float(new_init_learning_state)
            self.init_learning_variance = float(new_init_learning_variance)
        elif self.init_state_method == "set_initial_to_zero":
            self.init_learning_state = 0.0
            self.init_learning_variance = self.sigma_epsilon**2
        elif self.init_state_method == "set_initial_conservative_from_second_trial":
            if len(self.smoothed_learning_state_mode) > 1:
                self.init_learning_state = (
                    0.5 * self.smoothed_learning_state_mode[1]
                )  # x_{1|T}
            else:
                logger.warning(
                    "Not enough trials to use 'set_initial_conservative_from_second_trial' (need at least 2). "
                    "Falling back to 'reestimate_initial_from_data' for initial state x0."
                )
                self.init_learning_state = float(new_init_learning_state)
            self.init_learning_variance = (
                self.sigma_epsilon**2
            )  # Use updated sigma_epsilon
        elif self.init_state_method == "set_initial_direct_from_second_trial":
            if (
                len(self.smoothed_learning_state_mode) > 1
                and len(self.smoothed_learning_state_variance) > 1
            ):
                self.init_learning_state = self.smoothed_learning_state_mode[
                    1
                ]  # x_{1|T}
                self.init_learning_variance = self.smoothed_learning_state_variance[
                    1
                ]  # P_{1|T}
            else:
                logger.warning(
                    "Not enough trials/data to use 'set_initial_direct_from_second_trial' (need at least 2). "
                    "Falling back to 'reestimate_initial_from_data' for initial state (x0, P0)."
                )
                self.init_learning_state = float(new_init_learning_state)
                self.init_learning_variance = float(new_init_learning_variance)
        elif self.init_state_method == "user_provided":
            pass  # No change to initial state, user must set it externally

    def fit(
        self,
        n_correct_responses: jax.Array,
        max_iter: int = 100,
        tolerance: float = 1e-4,
    ) -> list[float]:
        """Fits the model to responses using the EM algorithm.

        Iteratively performs E-steps and M-steps until convergence or
        the maximum number of iterations is reached.

        Parameters
        ----------
        responses : jax.Array, shape (n_trials,)
            The sequence of responses.
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tolerance : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.

        Returns
        -------
        log_likelihoods : list[float]
            A list of marginal log-likelihoods at each iteration.
        """
        log_likelihoods: List[Optional[float]] = []
        previous_log_likelihood: float = -jnp.inf

        for iteration in range(max_iter):
            # E-step
            current_log_likelihood = self._e_step(n_correct_responses)
            log_likelihoods.append(current_log_likelihood)

            # Check convergence
            is_converged, is_increasing = check_converged(
                current_log_likelihood, previous_log_likelihood, tolerance
            )

            if not is_increasing:
                logger.warning(
                    f"Log-likelihood decreased at iteration {iteration + 1}!"
                )

            if is_converged:
                logger.info(f"Converged after {iteration + 1} iterations.")
                break

            # M-step
            self._m_step(n_correct_responses)

            logger.info(
                f"Iteration {iteration + 1}/{max_iter}\t"
                f"Log-Likelihood: {current_log_likelihood:.4f}\t"
                f"Change: {(current_log_likelihood - previous_log_likelihood):.4f}"
            )
            previous_log_likelihood = current_log_likelihood

        if len(log_likelihoods) == max_iter:
            logger.warning("Reached maximum iterations without converging.")

        return log_likelihoods

    def get_learning_curve(
        self,
        key: jax.random.PRNGKey,
        n_samples: int = 10000,
        percentiles: Optional[jax.Array] = None,
        calculate_pcert: bool = False,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        """
        Calculates the smoothed learning curve (probability of correct response)
        and its confidence limits.

        Must be called after `fit`.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX PRNG key for random number generation (for sampling).
        n_samples : int, optional
            Number of Monte Carlo samples to draw per trial for confidence limits.
            Default is 10000.
        percentiles : jax.Array, optional
            Array of percentiles to compute for the probability (e.g., jnp.array([5, 50, 95])).
            If None, defaults to jnp.array([5.0, 50.0, 95.0]).
        calculate_pcert : bool, optional
            If True, also calculates and returns the certainty (pcert) that the true
            probability of a correct response is greater than `self.current_prob_correct_by_chance`.
            Default is False.

        Returns
        -------
        probability_percentiles : jnp.ndarray
            Shape (n_percentiles, n_trials). Computed percentile values for the
            probability of correct response for each trial.
        pcert : Optional[jnp.ndarray]
            Shape (n_trials,). Certainty p_k > p_chance. Returned if `calculate_pcert` is True.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet (i.e., smoothed estimates are not available).
        """
        if (
            self.smoothed_learning_state_mode is None
            or self.smoothed_learning_state_variance is None
        ):
            raise RuntimeError("Model has not been fitted. Run .fit() method first.")

        prob_chance_for_pcert = self.prob_correct_by_chance if calculate_pcert else None

        return calculate_probability_confidence_limits(
            key=key,
            smoothed_mode=self.smoothed_learning_state_mode,
            smoothed_variance=self.smoothed_learning_state_variance,
            mu_bias=self.mu_bias,
            n_samples=n_samples,
            percentiles=percentiles,
            prob_correct_by_chance=prob_chance_for_pcert,
        )

    def get_latent_state_percentiles(
        self,
        key: jax.random.PRNGKey,
        n_samples: int = 10000,
        percentiles: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Calculates confidence percentiles for the smoothed latent learning state x_k|T.

        Must be called after `fit`.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX PRNG key for random number generation.
        n_samples : int, optional
            Number of Monte Carlo samples per trial. Default is 10000.
        percentiles : jax.Array, optional
            Percentiles to compute (e.g., jnp.array([5, 50, 95])).
            Defaults to [5.0, 50.0, 95.0].

        Returns
        -------
        jax.Array, shape (n_percentiles, n_trials)
            Computed percentile values for the latent state.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if (
            self.smoothed_learning_state_mode is None
            or self.smoothed_learning_state_variance is None
        ):
            raise RuntimeError("Model has not been fitted. Run .fit() method first.")

        return calculate_latent_state_percentiles(
            key=key,
            smoothed_mode=self.smoothed_learning_state_mode,
            smoothed_variance=self.smoothed_learning_state_variance,
            n_samples=n_samples,
            percentiles=percentiles,
        )

    def find_critical_run_length(
        self,
        sequence_length: int,
        prob_success_null: Optional[float] = None,
        critical_probability_threshold: float = 0.05,
        min_run_length: int = 2,
        max_run_length: int = 35,
    ) -> Optional[int]:
        """Determines the minimum length of a run of consecutive successes
        that would be statistically significant under a null hypothesis.

        This utilizes the module-level `find_min_consecutive_successes` function.

        Parameters
        ----------
        sequence_length : int
            The length of the trial sequence to consider for the criterion
            (e.g., total number of trials in an experiment block).
        prob_success_null : Optional[float], optional
            Probability of success under the null hypothesis (e.g., chance performance).
            If None, this method attempts to use the model's fitted
            `self.current_prob_correct_by_chance`. Default is None.
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

        Raises
        ------
        RuntimeError
            If `prob_success_null` is None and the model has not been fitted yet
            (so `self.current_prob_correct_by_chance` is not available/fitted).
        """
        prob_success_null_to_use: float
        if prob_success_null is None:
            # Check if model has been fitted by looking at one of the E-step results
            if self.filtered_learning_state_mode is None:
                raise RuntimeError(
                    "Model must be fitted to use its estimate of "
                    "prob_correct_by_chance as prob_success_null. "
                    "Alternatively, provide prob_success_null directly."
                )
            prob_success_null_to_use = self.current_prob_correct_by_chance
            logger.info(
                f"Using fitted prob_correct_by_chance "
                f"({prob_success_null_to_use:.3f}) as prob_success_null."
            )
        else:
            prob_success_null_to_use = prob_success_null

        return find_min_consecutive_successes(
            prob_success_null=prob_success_null_to_use,
            critical_probability_threshold=critical_probability_threshold,
            sequence_length=sequence_length,
            min_run_length=min_run_length,
            max_run_length=max_run_length,
        )

    def identify_significant_runs_in_data(
        self,
        observed_binary_responses: jax.Array,  # Ensure this is a JAX or NumPy array
        prob_success_null: Optional[float] = None,
        critical_probability_threshold: float = 0.05,
        min_run_length_for_j_crit: int = 2,  # Parameter for j_crit calculation
        max_run_length_for_j_crit: int = 35,  # Parameter for j_crit calculation
    ) -> Tuple[Optional[int], List[Tuple[int, int]]]:
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
        prob_success_null : Optional[float], optional
            Probability of success under the null hypothesis used for determining `j_crit`.
            If None, defaults to the model's fitted `self.current_prob_correct_by_chance`.
        critical_probability_threshold : float, optional
            The critical p-value (alpha) for determining `j_crit`. Default is 0.05.
        min_run_length_for_j_crit : int, optional
            Minimum run length to test when calculating `j_crit`. Default is 2.
        max_run_length_for_j_crit : int, optional
            Maximum run length to test when calculating `j_crit`. Default is 35.

        Returns
        -------
        Tuple[Optional[int], List[Tuple[int, int]]]
            - j_crit (Optional[int]): The determined critical run length.
            - significant_runs (List[Tuple[int, int]]): A list of (start_index, end_index)
              tuples for each identified significant run of successes in the
              `observed_binary_responses`. Indices are 0-based and end_index is inclusive.

        Raises
        ------
        RuntimeError
            If `prob_success_null` is None and the model has not been fitted.
        TypeError
            If `observed_binary_responses` is not a JAX or NumPy array.
        ValueError
            If `observed_binary_responses` is not 1D.
        """
        if not isinstance(observed_binary_responses, (jax.Array, np.ndarray)):
            raise TypeError("observed_binary_responses must be a JAX or NumPy array.")
        if observed_binary_responses.ndim != 1:
            raise ValueError("observed_binary_responses must be a 1D array.")
        # It's assumed observed_binary_responses contains 0s and 1s.

        sequence_length = len(observed_binary_responses)
        if sequence_length == 0:
            return None, []

        j_crit = self.find_critical_run_length(
            sequence_length=sequence_length,
            prob_success_null=prob_success_null,
            critical_probability_threshold=critical_probability_threshold,
            min_run_length=min_run_length_for_j_crit,
            max_run_length=max_run_length_for_j_crit,
        )

        significant_runs: List[Tuple[int, int]] = []
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
        observed_binary_responses_jnp = jnp.asarray(observed_binary_responses)
        significant_runs = _find_runs_of_value(
            data=observed_binary_responses_jnp,
            value_to_find=1,  # Assuming 1 represents success
            min_length=j_crit,
        )

        return j_crit, significant_runs

    def find_criterion_trial(
        self,
        key: jax.random.PRNGKey,  # Needed if get_learning_curve not yet called
        lower_percentile_for_criterion: float = 5.0,  # e.g., for p05
        chance_level_override: Optional[float] = None,
        n_samples_for_ci: int = 10000,  # if CIs need to be recomputed
    ) -> Optional[int]:
        """Determines a trial index indicating when learning is reliably above chance.

        This method finds the last trial where the lower confidence bound
        (e.g., 5th percentile) of the estimated probability of a correct response
        is below a specified chance level. The trial *after* this index
        can be considered a point where performance is consistently above chance.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX PRNGKey, required if confidence intervals need to be (re)computed.
        lower_percentile_for_criterion : float, optional
            The lower percentile to use from the probability confidence interval
            (e.g., 5.0 for the 5th percentile, p05). Default is 5.0.
        chance_level_override : Optional[float], optional
            The probability of success considered as chance level.
            If None, uses the model's fitted `self.current_prob_correct_by_chance`.
            Default is None.
        n_samples_for_ci : int, optional
            Number of Monte Carlo samples if confidence intervals need to be recomputed.
            Default is 10000.

        Returns
        -------
        Optional[int]
            The 0-indexed trial number representing the last point where the lower
            confidence bound of success probability was below the chance level.
            Returns None if this condition is never met, or always met, or if
            the model is not fitted.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If `lower_percentile_for_criterion` is not found in computed CIs.
        """
        if self.smoothed_learning_state_mode is None:
            raise RuntimeError("Model has not been fitted. Run .fit() method first.")

        chance_p = (
            chance_level_override
            if chance_level_override is not None
            else self.prob_correct_by_chance
        )

        # Get the confidence intervals for the probability of correct response
        # Ensure percentiles passed to get_learning_curve includes the desired one
        target_percentiles = jnp.array(
            [
                lower_percentile_for_criterion,
                50.0,
                100.0 - lower_percentile_for_criterion,
            ]
        )

        prob_percentiles, _ = self.get_learning_curve(
            key=key,
            n_samples=n_samples_for_ci,
            percentiles=target_percentiles,
            calculate_pcert=False,  # pcert not needed for this specific calculation
        )

        # Assuming prob_percentiles is (n_percentiles, n_trials)
        # and the first row corresponds to lower_percentile_for_criterion
        # This requires knowing the order or finding the correct row.
        # If target_percentiles are sorted, prob_percentiles[0] is p_lower.
        p_lower_bound = prob_percentiles[0, :]

        # Find indices where the lower bound is less than chance_p
        # Note: MATLAB's find gives 1-based indices. Python gives 0-based.
        below_chance_indices = jnp.where(p_lower_bound < chance_p)[0]

        if below_chance_indices.shape[0] == 0:
            # Lower bound is never below chance (e.g., starts above chance or chance is very low)
            # Or, if learning is immediate, this might also be empty.
            # Consider what this means: performance is always reliably above chance from trial 0.
            # Smith et al. would return NaN, here maybe None or -1.
            logger.info(
                f"The {lower_percentile_for_criterion}th percentile of success probability "
                f"is never below the chance level of {chance_p:.3f}."
            )
            return None  # Or -1 to indicate learning from the start

        # `cback` in MATLAB is the last such trial.
        last_trial_below_chance = below_chance_indices[-1]

        # The MATLAB code has a check: if(cback(end) < size(I,2) ).
        # This means if the very last trial's p05 is still below chance,
        # it considers learning not "fully" established by this criterion.
        # Here, n_trials = len(p_lower_bound)
        if int(last_trial_below_chance) == (len(p_lower_bound) - 1):
            logger.info(
                f"The {lower_percentile_for_criterion}th percentile of success probability "
                f"is still below chance ({chance_p:.3f}) at the last trial. "
                f"Learning criterion not met within the observed trials."
            )
            return None  # Or last_trial_below_chance if definition differs slightly

        return int(last_trial_below_chance)
