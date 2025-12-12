"""Point-process Kalman filter and smoother for neural spike data.

This module implements state-space models with point-process (spike) observations
using the Laplace-EKF approach from Eden & Brown (2004).

The model is:
    x_k = A @ x_{k-1} + w_k,  w_k ~ N(0, Q)
    y_{n,k} ~ Poisson(exp(log_intensity_func(Z_k, x_k)[n]) * dt)

where x_k is the latent state, y_{n,k} is the spike count for neuron n at time k,
and log_intensity_func returns log firing rates for all neurons.

Multi-Neuron Support
--------------------
The filter supports multiple neurons sharing a common latent state:

- spike_indicator: (n_time, n_neurons) - spike counts for each neuron
- log_conditional_intensity(Z_k, x_k) returns (n_neurons,) log-intensities

For backwards compatibility, single-neuron inputs are automatically promoted:
- spike_indicator: (n_time,) is treated as (n_time, 1)
- scalar log-intensity output is wrapped to (1,)

References
----------
[1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
    Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
    Neural Computation 16, 971-998.
"""

import logging
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import (
    _kalman_smoother_update,
    psd_solve,
    sum_of_outer_products,
    symmetrize,
)
from state_space_practice.utils import check_converged

logger = logging.getLogger(__name__)


def log_conditional_intensity(design_matrix: ArrayLike, params: ArrayLike) -> Array:
    """Computes the log conditional intensity for a point process.

    This is the default linear log-intensity function: log(λ) = Z @ x.

    For single-neuron models, this returns a scalar.
    For multi-neuron models, design_matrix should be (n_neurons, n_params)
    so this returns (n_neurons,) log-intensities.

    Parameters
    ----------
    design_matrix : ArrayLike, shape (n_params,) or (n_neurons, n_params)
        Design matrix (Z_k) used in the intensity function.
        For single neuron: (n_params,) row vector.
        For multi-neuron: (n_neurons, n_params) matrix.
    params : ArrayLike, shape (n_params,)
        Parameters (latent state) for the intensity function.

    Returns
    -------
    Array, shape () or (n_neurons,)
        Log conditional intensity (log(λ_k)).
        Scalar for single-neuron, (n_neurons,) for multi-neuron.
    """
    return jnp.asarray(design_matrix) @ jnp.asarray(params)


def _point_process_laplace_update(
    one_step_mean: Array,
    one_step_cov: Array,
    spike_indicator_t: Array,
    dt: float,
    log_intensity_func: Callable[[Array], Array],
    diagonal_boost: float = 1e-9,
    grad_log_intensity_func: Optional[Callable[[Array], Array]] = None,
    hess_log_intensity_func: Optional[Callable[[Array], Array]] = None,
) -> tuple[Array, Array, float]:
    """Single point-process Laplace-EKF update for multiple neurons.

    Performs a Bayesian update of the latent state posterior given observed
    spike counts, using a Laplace (Gaussian) approximation to the posterior.

    This is the core math for point-process observation updates, factored
    out to be reusable by both the non-switching and switching filters.

    The observation model is:
        y_n ~ Poisson(exp(log_intensity_func(x)[n]) * dt)

    Parameters
    ----------
    one_step_mean : Array, shape (n_latent,)
        Predicted mean from dynamics: A @ m_{t-1}
    one_step_cov : Array, shape (n_latent, n_latent)
        Predicted covariance: A @ P_{t-1} @ A.T + Q
    spike_indicator_t : Array, shape (n_neurons,)
        Spike counts at time t for all neurons
    dt : float
        Time bin width in seconds
    log_intensity_func : Callable[[Array], Array]
        Function mapping state (n_latent,) to log-intensities (n_neurons,).
        Should return log(lambda) where lambda is firing rate in Hz.
    diagonal_boost : float, default=1e-9
        Small value added to precision matrix diagonal for numerical stability.
    grad_log_intensity_func : Callable[[Array], Array] | None, optional
        Pre-computed gradient function (Jacobian) of log_intensity_func.
        If None, computed via jax.jacfwd(log_intensity_func).
        Passing pre-computed functions can improve compilation speed when
        this function is called repeatedly inside a JIT-compiled context.
    hess_log_intensity_func : Callable[[Array], Array] | None, optional
        Pre-computed Hessian function of log_intensity_func.
        If None, computed via jax.jacfwd of the gradient function.

    Returns
    -------
    posterior_mean : Array, shape (n_latent,)
        Updated state mean after incorporating spike observations
    posterior_cov : Array, shape (n_latent, n_latent)
        Updated state covariance after incorporating spike observations
    log_likelihood : float
        Sum of Poisson log-pmfs across neurons

    Notes
    -----
    The Laplace approximation uses the predicted mean as the expansion point
    for a single Newton step update. For multiple neurons, the gradients and
    Hessians are summed across neurons.

    For Poisson likelihood with log-link:
        log p(y | x) = sum_n [y_n * log(lambda_n * dt) - lambda_n * dt - log(y_n!)]
        gradient = sum_n [(y_n - lambda_n * dt) * d(log_lambda_n)/dx]
        Hessian = sum_n [(y_n - lambda_n * dt) * d^2(log_lambda_n)/dx^2
                        - lambda_n * dt * (d(log_lambda_n)/dx)^T @ (d(log_lambda_n)/dx)]

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    """
    # Compute gradients and Hessians of log-intensity function
    if grad_log_intensity_func is None:
        grad_log_intensity_func = jax.jacfwd(log_intensity_func)
    if hess_log_intensity_func is None:
        hess_log_intensity_func = jax.jacfwd(grad_log_intensity_func)
    grad_log_intensity = grad_log_intensity_func
    hess_log_intensity = hess_log_intensity_func

    # Evaluate at the one-step predicted mean
    log_lambda = log_intensity_func(one_step_mean)  # (n_neurons,)
    conditional_intensity = jnp.exp(log_lambda) * dt  # Expected spike count

    # Innovation: observed - expected
    innovation = spike_indicator_t - conditional_intensity  # (n_neurons,)

    # Jacobian: (n_neurons, n_latent)
    jacobian = grad_log_intensity(one_step_mean)  # (n_neurons, n_latent)

    # Hessian: (n_neurons, n_latent, n_latent)
    hessian = hess_log_intensity(one_step_mean)  # (n_neurons, n_latent, n_latent)

    # Weighted gradient for mean update: jacobian.T @ innovation -> (n_latent,)
    gradient = jacobian.T @ innovation  # (n_latent,)

    # Fisher information: jacobian.T @ diag(conditional_intensity) @ jacobian
    fisher_info = jacobian.T @ (
        conditional_intensity[:, None] * jacobian
    )  # (n_latent, n_latent)

    # Hessian correction: sum_n innovation_n * hessian[n]
    hessian_correction = jnp.einsum(
        "n,nij->ij", innovation, hessian
    )  # (n_latent, n_latent)

    # Prior precision via psd_solve for numerical stability
    n_latent = one_step_mean.shape[0]
    identity = jnp.eye(n_latent)
    prior_precision = psd_solve(one_step_cov, identity, diagonal_boost=diagonal_boost)

    # Posterior precision: P^{-1} = P_prior^{-1} + Fisher - innovation * Hessian
    posterior_precision = prior_precision + fisher_info - hessian_correction

    # Posterior covariance via psd_solve
    posterior_cov = psd_solve(posterior_precision, identity, diagonal_boost=diagonal_boost)
    posterior_cov = symmetrize(posterior_cov)

    # Posterior mean: Newton step from prior
    posterior_mean = one_step_mean + psd_solve(
        posterior_precision, gradient, diagonal_boost=diagonal_boost
    )

    # Log-likelihood: sum of Poisson log-pmfs
    log_likelihood = jnp.sum(jax.scipy.stats.poisson.logpmf(spike_indicator_t, conditional_intensity))

    return posterior_mean, posterior_cov, log_likelihood


def stochastic_point_process_filter(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
) -> tuple[Array, Array, float]:
    """Applies a Stochastic State Point Process Filter (SSPPF).

    This filter estimates a time-varying latent state ($x_k$) based on
    point process observations ($y_k$). It assumes a linear Gaussian state
    transition and a point process observation model where the conditional
    intensity $\\lambda_k$ depends on the state.

    $$ x_k = A x_{k-1} + w_k, \\quad w_k \\sim N(0, Q) $$
    $$ \\lambda_{n,k} = f(x_k, Z_k)_n $$
    $$ y_{n,k} \\sim \\text{Poisson}(\\lambda_{n,k} \\Delta t) $$

    The filter uses a local Gaussian approximation (Laplace-EKF approach)
    at each update step, utilizing the gradient and Hessian of the
    log-likelihood. It implements a single Newton-Raphson like step
    per time bin.

    Multi-Neuron Support
    --------------------
    The filter supports multiple neurons sharing a common latent state:

    - spike_indicator: (n_time, n_neurons) - spike counts for each neuron
    - log_conditional_intensity(Z_k, x_k) should return (n_neurons,)

    For backwards compatibility, single-neuron inputs work as before:
    - spike_indicator: (n_time,) is internally promoted to (n_time, 1)
    - scalar log-intensity output is wrapped to (1,)

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
        Initial mean of the latent state ($x_0$).
    init_covariance_params : ArrayLike, shape (n_params, n_params)
        Initial covariance of the latent state ($P_0$).
    design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_params)
        Design matrix ($Z_k$) used in the intensity function.
        Shape depends on the log_conditional_intensity function.
        For multi-neuron with default linear intensity, use (n_time, n_neurons, n_params).
    spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
        Observed spike counts or indicators ($y_k$).
        For single neuron: (n_time,)
        For multiple neurons: (n_time, n_neurons)
    dt : float
        Time step size ($\\Delta t$).
    transition_matrix : ArrayLike, shape (n_params, n_params)
        State transition matrix ($A$).
    process_cov : ArrayLike, shape (n_params, n_params)
        Process noise covariance ($Q$).
    log_conditional_intensity : callable
        Function `log_lambda(Z_k, x_k)` returning the log conditional
        intensity. Should return (n_neurons,) array for multi-neuron case,
        or scalar for single-neuron.

    Returns
    -------
    posterior_mean : Array, shape (n_time, n_params)
        Filtered posterior means ($x_{k|k}$).
    posterior_variance : Array, shape (n_time, n_params, n_params)
        Filtered posterior covariances ($P_{k|k}$).
    marginal_log_likelihood : float
        Total log-likelihood of the observations given the model.

    Notes
    -----
    For multiple neurons, the log-likelihood at each timestep is the sum
    of independent Poisson log-pmfs:
        log p(y_t | x_t) = sum_n log Poisson(y_{n,t} | lambda_{n,t} * dt)

    The filter aggregates information from all neurons to update the shared
    latent state. More neurons provide more information, reducing posterior
    uncertainty.

    References
    ----------
    [1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
      Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
      Neural Computation 16, 971-998 (2004).
    """
    # Convert to arrays
    init_mean_params = jnp.asarray(init_mean_params)
    init_covariance_params = jnp.asarray(init_covariance_params)
    design_matrix = jnp.asarray(design_matrix)
    spike_indicator = jnp.asarray(spike_indicator)
    transition_matrix = jnp.asarray(transition_matrix)
    process_cov = jnp.asarray(process_cov)

    # Promote single-neuron spike_indicator to (n_time, 1) for consistent handling
    single_neuron = spike_indicator.ndim == 1
    if single_neuron:
        spike_indicator = spike_indicator[:, None]

    # Pre-compute gradient and Hessian functions outside the scan
    # These are parameterized by design_matrix_t to avoid recreating jax.jacfwd each step
    def _log_intensity_with_design(design_matrix_t, x):
        log_lambda = log_conditional_intensity(design_matrix_t, x)
        return jnp.atleast_1d(log_lambda)

    # Create grad/hess w.r.t. x (argnums=1), keeping design_matrix_t as parameter
    _grad_log_intensity = jax.jacfwd(_log_intensity_with_design, argnums=1)
    _hess_log_intensity = jax.jacfwd(_grad_log_intensity, argnums=1)

    def _step(
        params_prev: tuple[Array, Array, float],
        args: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array, float], tuple[Array, Array]]:
        """Point Process Adaptive Filter update step."""
        # Unpack previous parameters
        mean_prev, variance_prev, marginal_log_likelihood = params_prev
        design_matrix_t, spike_indicator_t = args

        # One-step prediction
        one_step_mean = transition_matrix @ mean_prev
        one_step_covariance = (
            transition_matrix @ variance_prev @ transition_matrix.T + process_cov
        )
        one_step_covariance = symmetrize(one_step_covariance)

        # Create log_intensity_func that captures design_matrix_t
        def log_intensity_func(x):
            return _log_intensity_with_design(design_matrix_t, x)

        # Create grad/hess functions that capture design_matrix_t
        # These use the pre-computed jacfwd functions from outside the scan
        def grad_log_intensity_func(x):
            return _grad_log_intensity(design_matrix_t, x)

        def hess_log_intensity_func(x):
            return _hess_log_intensity(design_matrix_t, x)

        # Use the generalized multi-neuron Laplace update
        posterior_mean, posterior_covariance, log_lik = _point_process_laplace_update(
            one_step_mean,
            one_step_covariance,
            spike_indicator_t,
            dt,
            log_intensity_func,
            grad_log_intensity_func=grad_log_intensity_func,
            hess_log_intensity_func=hess_log_intensity_func,
        )

        marginal_log_likelihood += log_lik

        return (posterior_mean, posterior_covariance, marginal_log_likelihood), (
            posterior_mean,
            posterior_covariance,
        )

    marginal_log_likelihood = jnp.array(0.0)
    (_, _, marginal_log_likelihood), (
        filtered_mean,
        filtered_cov,
    ) = jax.lax.scan(
        _step,
        (init_mean_params, init_covariance_params, marginal_log_likelihood),
        (design_matrix, spike_indicator),
    )

    return filtered_mean, filtered_cov, marginal_log_likelihood


def stochastic_point_process_smoother(
    init_mean_params: ArrayLike,
    init_covariance_params: ArrayLike,
    design_matrix: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    process_cov: ArrayLike,
    log_conditional_intensity: Callable[[ArrayLike, ArrayLike], Array],
) -> tuple[Array, Array, Array, float]:
    """Applies a Stochastic State Point Process Smoother (SSPPS).

    This smoother estimates a time-varying latent state ($x_k$) based on
    point process observations ($y_k$) using a Kalman smoother approach.
    It first applies a stochastic point process filter to obtain the filtered
    means and covariances, and then applies a Kalman smoother to refine these estimates.

    $$ x_k = A x_{k-1} + w_k, \\quad w_k \\sim N(0, Q) $$
    $$ \\lambda_{n,k} = f(x_k, Z_k)_n $$
    $$ y_{n,k} \\sim \\text{Poisson}(\\lambda_{n,k} \\Delta t) $$

    Multi-Neuron Support
    --------------------
    The smoother supports multiple neurons sharing a common latent state.
    See `stochastic_point_process_filter` for details on multi-neuron inputs.

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
        Initial mean of the latent state ($x_0$).
    init_covariance_params : ArrayLike, shape (n_params, n_params)
        Initial covariance of the latent state ($P_0$).
    design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_params)
        Design matrix ($Z_k$) used in the intensity function.
        Shape depends on the log_conditional_intensity function.
    spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
        Observed spike counts or indicators ($y_k$).
        For single neuron: (n_time,)
        For multiple neurons: (n_time, n_neurons)
    dt : float
        Time step size ($\\Delta t$).
    transition_matrix : ArrayLike, shape (n_params, n_params)
        State transition matrix ($A$).
    process_cov : ArrayLike, shape (n_params, n_params)
        Process noise covariance ($Q$).
    log_conditional_intensity : callable
        Function `log_lambda(Z_k, x_k)` returning the log conditional
        intensity. Should return (n_neurons,) for multi-neuron case.

    Returns
    -------
    smoother_mean : Array, shape (n_time, n_params)
        Smoothed posterior means ($x_{k|T}$).
    smoother_cov : Array, shape (n_time, n_params, n_params)
        Smoothed posterior covariances ($P_{k|T}$).
    smoother_cross_cov : Array, shape (n_time - 1, n_params, n_params)
        Smoothed cross-covariances ($P_{k|T, k-1}$).
    marginal_log_likelihood : float
        Total log-likelihood of the observations given the model.

    Notes
    -----
    The smoother is observation-model agnostic - it operates only on the
    Gaussian posteriors from the filter. The multi-neuron handling is done
    entirely in the filter step.

    References
    ----------
    [1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998 (2004).
    """
    filtered_mean, filtered_cov, marginal_log_likelihood = (
        stochastic_point_process_filter(
            init_mean_params,
            init_covariance_params,
            design_matrix,
            spike_indicator,
            dt,
            transition_matrix,
            process_cov,
            log_conditional_intensity,
        )
    )

    def _step(carry, args):
        (
            next_smoother_mean,
            next_smoother_cov,
        ) = carry

        filter_mean, filter_cov = args

        smoother_mean, smoother_cov, smoother_cross_cov = _kalman_smoother_update(
            next_smoother_mean,
            next_smoother_cov,
            filter_mean,
            filter_cov,
            process_cov,
            transition_matrix,
        )
        return (
            smoother_mean,
            smoother_cov,
        ), (
            smoother_mean,
            smoother_cov,
            smoother_cross_cov,
        )

    (_, _), (smoother_mean, smoother_cov, smoother_cross_cov) = jax.lax.scan(
        _step,
        (filtered_mean[-1], filtered_cov[-1]),
        (filtered_mean[:-1], filtered_cov[:-1]),
        reverse=True,
    )

    smoother_mean = jnp.concatenate((smoother_mean, filtered_mean[-1][None]))
    smoother_cov = jnp.concatenate((smoother_cov, filtered_cov[-1][None]))

    return smoother_mean, smoother_cov, smoother_cross_cov, marginal_log_likelihood


def kalman_maximization_step(
    smoother_mean: ArrayLike,
    smoother_cov: ArrayLike,
    smoother_cross_cov: ArrayLike,
) -> tuple[Array, Array, Array, Array]:
    """Maximization step for the Kalman filter.

    Parameters
    ----------
    smoother_mean : ArrayLike, shape (n_time, n_cont_states)
        smoother mean.
    smoother_cov : ArrayLike, shape (n_time, n_cont_states, n_cont_states)
        smoother covariance.
    smoother_cross_cov : ArrayLike, shape (n_time - 1, n_cont_states, n_cont_states)
        smoother cross-covariance.

    Returns
    -------
    transition_matrix : Array, shape (n_cont_states, n_cont_states)
        Transition matrix.
    process_cov : Array, shape (n_cont_states, n_cont_states)
        Process covariance.
    mean_init : Array, shape (n_cont_states,)
        Initial mean.
    cov_init : Array, shape (n_cont_states, n_cont_states)
        Initial covariance.

    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """
    smoother_mean = jnp.asarray(smoother_mean)
    smoother_cov = jnp.asarray(smoother_cov)
    smoother_cross_cov = jnp.asarray(smoother_cross_cov)

    n_time = smoother_mean.shape[0]

    # Compute intermediate expectation terms
    gamma = jnp.sum(smoother_cov, axis=0) + sum_of_outer_products(
        smoother_mean, smoother_mean
    )
    gamma1 = gamma - jnp.outer(smoother_mean[-1], smoother_mean[-1]) - smoother_cov[-1]
    gamma2 = gamma - jnp.outer(smoother_mean[0], smoother_mean[0]) - smoother_cov[0]
    beta = (
        smoother_cross_cov.sum(axis=0)
        + sum_of_outer_products(smoother_mean[:-1], smoother_mean[1:])
    ).T

    # Transition matrix
    transition_matrix = psd_solve(gamma1, beta.T).T

    # Process covariance
    process_cov = (gamma2 - transition_matrix @ beta.T) / (n_time - 1)
    process_cov = symmetrize(process_cov)

    # Initial mean and covariance
    init_mean = smoother_mean[0]
    init_cov = smoother_cov[0]

    return (
        transition_matrix,
        process_cov,
        init_mean,
        init_cov,
    )


def get_confidence_interval(
    posterior_mean: ArrayLike, posterior_covariance: ArrayLike, alpha: float = 0.01
) -> Array:
    """Get the confidence interval from the posterior covariance

    Parameters
    ----------
    posterior_mean : ArrayLike, shape (n_time, n_params)
    posterior_covariance : ArrayLike, shape (n_time, n_params, n_params)
    alpha : float, optional
        Confidence level, by default 0.01
    """
    posterior_mean = jnp.asarray(posterior_mean)
    posterior_covariance = jnp.asarray(posterior_covariance)
    z = jax.scipy.stats.norm.ppf(1 - alpha / 2)
    ci = z * jnp.sqrt(
        jnp.diagonal(posterior_covariance, axis1=-2, axis2=-1)
    )  # shape (n_time, n_params)

    return jnp.stack((posterior_mean - ci, posterior_mean + ci), axis=-1)


def steepest_descent_point_process_filter(
    init_mean_params: ArrayLike,
    x: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    epsilon: ArrayLike,
    log_receptive_field_model: Callable[[ArrayLike, ArrayLike], Array],
) -> Array:
    """Steepest Descent Point Process Filter (SDPPF)

    Parameters
    ----------
    init_mean_params : ArrayLike, shape (n_params,)
    x : ArrayLike, shape (n_time,)
        Continuous-valued input signal
    spike_indicator : ArrayLike, shape (n_time,)
        Spike count
    dt : float
        Time step
    epsilon : ArrayLike, shape (n_params, n_params)
        Learning rate
    log_receptive_field_model : callable
        Function that takes in `x` and parameters and returns the log spike rate

    Returns
    -------
    posterior_mean : Array, shape (n_time, n_params)

    References
    ----------
    .. [1] Brown, E.N., Nguyen, D.P., Frank, L.M., Wilson, M.A., and Solo, V. (2001).
    An analysis of neural receptive field plasticity by point process adaptive filtering.
    Proceedings of the National Academy of Sciences 98, 12261–12266.
    https://doi.org/10.1073/pnas.201409398.

    .. [2] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
      Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
      Neural Computation 16, 971-998 (2004).

    Notes
    -----
    Equation in [1] is for the likelihood while in [2] it is for the log likelihood.
    This implementation follows the formulation in [2].

    """
    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)

    def _update(
        mean_prev: Array, args: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        """Steepest Descent Point Process Filter update step"""
        x_t, spike_indicator_t = args
        conditional_intensity = jnp.exp(log_receptive_field_model(x_t, mean_prev)) * dt
        innovation = spike_indicator_t - conditional_intensity
        one_step_grad = grad_log_receptive_field_model(x_t, mean_prev)
        posterior_mean = mean_prev + epsilon @ one_step_grad * innovation

        return posterior_mean, posterior_mean

    return jax.lax.scan(_update, init_mean_params, (x, spike_indicator))[1]


class PointProcessModel:
    """Point Process State-Space Model with EM fitting.

    Implements the Eden & Brown (2004) adaptive point process filter/smoother
    with EM algorithm for parameter estimation.

    Model:
        x_k = A @ x_{k-1} + w_k,  w_k ~ N(0, Q)
        n_{j,k} ~ Poisson(exp(log_intensity_func(Z_k, x_k)[j]) * dt)

    The EM algorithm estimates the state dynamics parameters (A, Q) while
    the latent states x_k are estimated via the E-step (filter/smoother).

    Multi-Neuron Support
    --------------------
    The model supports multiple neurons sharing a common latent state:

    - spike_indicator: (n_time, n_neurons) - spike counts for each neuron
    - log_intensity_func(Z_k, x_k) should return (n_neurons,) log-intensities

    For backwards compatibility, single-neuron inputs work as before:
    - spike_indicator: (n_time,) is internally promoted to (n_time, 1)
    - scalar log-intensity output is wrapped to (1,)

    Parameters
    ----------
    n_state_dims : int
        Dimension of the latent state.
    dt : float
        Time step size.
    transition_matrix : ArrayLike, optional
        Initial state transition matrix A. Default is identity (random walk).
    process_cov : ArrayLike, optional
        Initial process noise covariance Q.
    init_mean : ArrayLike, optional
        Initial state mean.
    init_cov : ArrayLike, optional
        Initial state covariance.
    log_intensity_func : callable, optional
        Function log_lambda(Z_k, x_k) returning log conditional intensity.
        Default is linear: Z_k @ x_k.
        For multi-neuron, should return (n_neurons,) array.
    update_transition_matrix : bool
        Whether to update A in M-step. Default True.
    update_process_cov : bool
        Whether to update Q in M-step. Default True.
    update_init_state : bool
        Whether to update initial state in M-step. Default True.

    Attributes
    ----------
    smoother_mean : Array
        Smoothed state estimates after fitting.
    smoother_cov : Array
        Smoothed state covariances after fitting.
    smoother_cross_cov : Array
        Smoothed cross-covariances after fitting.

    References
    ----------
    [1] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
        Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
        Neural Computation 16, 971-998.
    """

    def __init__(
        self,
        n_state_dims: int,
        dt: float,
        transition_matrix: Optional[ArrayLike] = None,
        process_cov: Optional[ArrayLike] = None,
        init_mean: Optional[ArrayLike] = None,
        init_cov: Optional[ArrayLike] = None,
        log_intensity_func: Optional[Callable] = None,
        update_transition_matrix: bool = True,
        update_process_cov: bool = True,
        update_init_state: bool = True,
    ):
        self.n_state_dims = n_state_dims
        self.dt = dt

        # Initialize parameters
        if transition_matrix is None:
            self.transition_matrix = jnp.eye(n_state_dims)
        else:
            self.transition_matrix = jnp.asarray(transition_matrix)

        if process_cov is None:
            self.process_cov = jnp.eye(n_state_dims) * 1e-4
        else:
            self.process_cov = jnp.asarray(process_cov)

        if init_mean is None:
            self.init_mean = jnp.zeros(n_state_dims)
        else:
            self.init_mean = jnp.asarray(init_mean)

        if init_cov is None:
            self.init_cov = jnp.eye(n_state_dims)
        else:
            self.init_cov = jnp.asarray(init_cov)

        if log_intensity_func is None:
            self.log_intensity_func = log_conditional_intensity
        else:
            self.log_intensity_func = log_intensity_func

        # Update flags
        self.update_transition_matrix = update_transition_matrix
        self.update_process_cov = update_process_cov
        self.update_init_state = update_init_state

        # Results (populated after fit)
        self.smoother_mean: Optional[Array] = None
        self.smoother_cov: Optional[Array] = None
        self.smoother_cross_cov: Optional[Array] = None
        self.filtered_mean: Optional[Array] = None
        self.filtered_cov: Optional[Array] = None

    def _e_step(
        self, design_matrix: ArrayLike, spike_indicator: ArrayLike
    ) -> float:
        """E-step: Run filter and smoother to estimate latent states.

        Parameters
        ----------
        design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_state_dims)
            Design matrix for the intensity function.
        spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Observed spike counts. Single neuron: (n_time,), multi-neuron: (n_time, n_neurons).

        Returns
        -------
        marginal_log_likelihood : float
        """
        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_log_likelihood,
        ) = stochastic_point_process_smoother(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self.log_intensity_func,
        )

        # Also store filtered results
        self.filtered_mean, self.filtered_cov, _ = stochastic_point_process_filter(
            init_mean_params=self.init_mean,
            init_covariance_params=self.init_cov,
            design_matrix=design_matrix,
            spike_indicator=spike_indicator,
            dt=self.dt,
            transition_matrix=self.transition_matrix,
            process_cov=self.process_cov,
            log_conditional_intensity=self.log_intensity_func,
        )

        return float(marginal_log_likelihood)

    def _m_step(self) -> None:
        """M-step: Update model parameters based on smoothed estimates."""
        if self.smoother_mean is None or self.smoother_cov is None:
            raise RuntimeError("Must run E-step before M-step")

        transition_matrix, process_cov, init_mean, init_cov = kalman_maximization_step(
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
        )

        if self.update_transition_matrix:
            self.transition_matrix = transition_matrix

        if self.update_process_cov:
            # Ensure positive definiteness
            process_cov = symmetrize(process_cov)
            # Ensure eigenvalues are positive (numerical stability)
            eigvals, eigvecs = jnp.linalg.eigh(process_cov)
            eigvals = jnp.maximum(eigvals, 1e-10)
            process_cov = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
            self.process_cov = process_cov

        if self.update_init_state:
            self.init_mean = init_mean
            self.init_cov = symmetrize(init_cov)

    def fit(
        self,
        design_matrix: ArrayLike,
        spike_indicator: ArrayLike,
        max_iter: int = 100,
        tolerance: float = 1e-4,
    ) -> list[float]:
        """Fit the model using the EM algorithm.

        Parameters
        ----------
        design_matrix : ArrayLike, shape (n_time, ...) or (n_time, n_neurons, n_state_dims)
            Design matrix for the intensity function.
            Shape depends on the log_intensity_func.
        spike_indicator : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Observed spike counts or indicators.
            For single neuron: (n_time,)
            For multiple neurons: (n_time, n_neurons)
        max_iter : int
            Maximum number of EM iterations.
        tolerance : float
            Convergence tolerance for relative change in log-likelihood.

        Returns
        -------
        log_likelihoods : list[float]
            Log-likelihood at each iteration.
        """
        design_matrix = jnp.asarray(design_matrix)
        spike_indicator = jnp.asarray(spike_indicator)

        log_likelihoods: list[float] = []
        previous_log_likelihood = -jnp.inf

        for iteration in range(max_iter):
            # E-step
            current_log_likelihood = self._e_step(design_matrix, spike_indicator)
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
            self._m_step()

            logger.info(
                f"Iteration {iteration + 1}/{max_iter}\t"
                f"Log-Likelihood: {current_log_likelihood:.4f}\t"
                f"Change: {(current_log_likelihood - previous_log_likelihood):.6f}"
            )
            previous_log_likelihood = current_log_likelihood

        if len(log_likelihoods) == max_iter:
            logger.warning("Reached maximum iterations without converging.")

        return log_likelihoods

    def get_rate_estimate(
        self,
        design_matrix: ArrayLike,
        use_smoothed: bool = True,
        evaluate_at_all_positions: bool = True,
    ) -> Array:
        """Get the estimated firing rate using the model's log-intensity function.

        This method computes firing rates by evaluating the stored log_intensity_func,
        supporting both single-neuron and multi-neuron models with arbitrary
        (possibly nonlinear) intensity functions.

        Parameters
        ----------
        design_matrix : ArrayLike
            Design matrix to evaluate rate at. Shape depends on usage:
            - If evaluate_at_all_positions=True (default): (n_pos, n_state_dims)
              where n_pos can be n_time or any number of positions/conditions
            - If evaluate_at_all_positions=False: same shape as used during fit,
              e.g., (n_time, n_state_dims) for single-neuron or
              (n_time, n_neurons, n_state_dims) for multi-neuron

        use_smoothed : bool
            If True, use smoothed estimates; otherwise use filtered.

        evaluate_at_all_positions : bool, default=True
            If True, evaluate the rate at all positions in design_matrix for each
            time point, returning shape (n_time, n_pos) for single-neuron or
            (n_time, n_pos, n_neurons) for multi-neuron.
            If False, evaluate time-aligned: design_matrix[t] with state[t],
            returning shape (n_time,) for single-neuron or (n_time, n_neurons)
            for multi-neuron.

        Returns
        -------
        rate : Array
            Estimated firing rate in Hz. Shape depends on evaluate_at_all_positions
            and whether the model is single/multi-neuron (see above).

        Notes
        -----
        The rate is computed as exp(log_intensity_func(design_matrix, x)) / dt.
        This generalizes to arbitrary intensity functions, not just the default
        linear Z @ x.
        """
        if use_smoothed:
            if self.smoother_mean is None:
                raise RuntimeError("Model has not been fitted yet.")
            state_estimate = self.smoother_mean
        else:
            if self.filtered_mean is None:
                raise RuntimeError("Model has not been fitted yet.")
            state_estimate = self.filtered_mean

        design_matrix = jnp.asarray(design_matrix)

        if evaluate_at_all_positions:
            # Evaluate rate at all positions for each time point
            # For each (time, position) pair, compute log_intensity_func(design[pos], state[time])
            # vmap over positions (inner), then over times (outer)
            def rate_at_time(state_t):
                # For this time's state, evaluate at all positions
                return jax.vmap(lambda dm: self.log_intensity_func(dm, state_t))(
                    design_matrix
                )

            log_rate = jax.vmap(rate_at_time)(state_estimate)  # (n_time, n_pos, ...)
        else:
            # Time-aligned evaluation: design_matrix[t] with state_estimate[t]
            log_rate = jax.vmap(self.log_intensity_func)(design_matrix, state_estimate)

        rate = jnp.exp(log_rate) / self.dt

        return rate

    def get_confidence_interval(
        self, alpha: float = 0.05, use_smoothed: bool = True
    ) -> Array:
        """Get confidence intervals for the state estimates.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CI).
        use_smoothed : bool
            If True, use smoothed estimates; otherwise use filtered.

        Returns
        -------
        ci : Array, shape (n_time, n_state_dims, 2)
            Lower and upper bounds of the confidence interval.
        """
        if use_smoothed:
            if self.smoother_mean is None or self.smoother_cov is None:
                raise RuntimeError("Model has not been fitted yet.")
            mean = self.smoother_mean
            cov = self.smoother_cov
        else:
            if self.filtered_mean is None or self.filtered_cov is None:
                raise RuntimeError("Model has not been fitted yet.")
            mean = self.filtered_mean
            cov = self.filtered_cov

        return get_confidence_interval(mean, cov, alpha=alpha)
