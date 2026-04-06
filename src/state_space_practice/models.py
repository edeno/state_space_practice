from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import psd_solve, stabilize_covariance
from state_space_practice.point_process_kalman import _safe_expected_count


def log_receptive_field_model(position: ArrayLike, params: ArrayLike) -> Array:
    params_arr = jnp.asarray(params)
    log_max_rate, place_field_center, scale = params_arr
    result: Array = log_max_rate - (jnp.asarray(position) - place_field_center) ** 2 / (
        2 * scale**2
    )
    return result


# NOTE: most general form of the SSPPF accounts for multiple neurons which is not implemented here
def stochastic_point_process_filter(
    init_mode_params: ArrayLike,
    init_covariance_params: ArrayLike,
    x: ArrayLike,
    spike_indicator: ArrayLike,
    dt: float,
    transition_matrix: ArrayLike,
    latent_state_covariance: ArrayLike,
    log_receptive_field_model: Callable[[ArrayLike, ArrayLike], Array],
) -> tuple[Array, Array]:
    """Stochastic State Point Process Filter (SSPPF)

    Parameters
    ----------
    init_mode_params : ArrayLike, shape (n_params,)
        Initial mean parameters
    init_covariance_params : ArrayLike, shape (n_params, n_params)
        Initial covariance parameters
    x : ArrayLike, shape (n_time,)
        Continuous-valued input signal
    spike_indicator : ArrayLike, shape (n_time,)
        Spike count
    dt : float
        Time step
    transition_matrix : ArrayLike, shape (n_params, n_params)
    latent_state_covariance : ArrayLike, shape (n_params, n_params)
    log_receptive_field_model : callable
        Function that takes in `x` and parameters and returns the log spike rate

    Returns
    -------
    posterior_mode : Array, shape (n_time, n_params)
    posterior_covariance : Array, shape (n_time, n_params, n_params)

    References
    ----------
    .. [1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
      Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
      Neural Computation 16, 971-998 (2004).


    """
    # Convert ArrayLike inputs to Array for internal use
    init_mode_params_arr: Array = jnp.asarray(init_mode_params)
    init_covariance_params_arr: Array = jnp.asarray(init_covariance_params)
    x_arr: Array = jnp.asarray(x)
    spike_indicator_arr: Array = jnp.asarray(spike_indicator)
    transition_matrix_arr: Array = jnp.asarray(transition_matrix)
    latent_state_covariance_arr: Array = jnp.asarray(latent_state_covariance)

    # Compute the gradient and hessian of the log receptive field model
    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)
    hess_log_receptive_field_model = jax.hessian(log_receptive_field_model, argnums=1)

    # Define the update step
    def _update(
        params_prev: tuple[Array, Array],
        args: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        """Point Process Adaptive Filter update step

        F : transition matrix
        Q : covariance matrix
        \theta_{k | k-1} :
        W_{k | k-1}: one_step_variance_params
        \theta_{k | k} : posterior_mode
        W_{k | k} : posterior_variance
        """

        # Unpack previous parameters
        mode_prev, covariance_prev = params_prev
        x_t, spike_indicator_t = args

        # One-step prediction
        one_step_mean = transition_matrix_arr @ mode_prev
        one_step_variance = (
            transition_matrix_arr @ covariance_prev @ transition_matrix_arr.T
            + latent_state_covariance_arr
        )

        # Compute the conditional intensity and innovation
        conditional_intensity = _safe_expected_count(
            log_receptive_field_model(x_t, one_step_mean), dt
        )
        innovation = spike_indicator_t - conditional_intensity

        # Compute the posterior mean and variance
        one_step_grad = grad_log_receptive_field_model(x_t, one_step_mean)[None]
        one_step_hess = hess_log_receptive_field_model(x_t, one_step_mean)

        # sum over:
        # (one_step_grad.T * conditional_intensity @ one_step_grad) - innovation * one_step_hess
        # if multiple neurons
        identity = jnp.eye(one_step_variance.shape[0], dtype=one_step_variance.dtype)
        prior_precision = psd_solve(one_step_variance, identity)
        inverse_posterior_covariance = (
            prior_precision
            + (one_step_grad.T * conditional_intensity @ one_step_grad)
            - innovation * one_step_hess
        )
        inverse_posterior_covariance = stabilize_covariance(
            inverse_posterior_covariance, min_eigenvalue=1e-9
        )
        posterior_covariance = psd_solve(inverse_posterior_covariance, identity)
        posterior_covariance = stabilize_covariance(
            posterior_covariance, min_eigenvalue=1e-9
        )

        # sum over one_step_grad.squeeze() * innovation if multiple neurons
        posterior_mode = one_step_mean + posterior_covariance @ (
            one_step_grad.squeeze() * innovation
        )

        return (posterior_mode, posterior_covariance), (
            posterior_mode,
            posterior_covariance,
        )

    # Run the SSPPF
    return jax.lax.scan(
        _update,
        (init_mode_params_arr, init_covariance_params_arr),
        (x_arr, spike_indicator_arr),
    )[1]


def get_confidence_interval(
    posterior_mode: ArrayLike, posterior_covariance: ArrayLike, alpha: float = 0.01
) -> Array:
    """Get the confidence interval from the posterior covariance

    Parameters
    ----------
    posterior_mode : ArrayLike, shape (n_time, n_params)
    posterior_covariance : ArrayLike, shape (n_time, n_params, n_params)
    alpha : float, optional
        Confidence level, by default 0.01
    """
    posterior_mode = jnp.asarray(posterior_mode)
    posterior_covariance = jnp.asarray(posterior_covariance)
    z = jax.scipy.stats.norm.ppf(1 - alpha / 2)
    ci = z * jnp.sqrt(
        jnp.diagonal(posterior_covariance, axis1=-2, axis2=-1)
    )  # shape (n_time, n_params)

    return jnp.stack((posterior_mode - ci, posterior_mode + ci), axis=-1)


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
    posterior_mode : Array, shape (n_time, n_params)

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
    # Convert ArrayLike inputs to Array for internal use
    init_mean_params_arr: Array = jnp.asarray(init_mean_params)
    x_arr: Array = jnp.asarray(x)
    spike_indicator_arr: Array = jnp.asarray(spike_indicator)
    epsilon_arr: Array = jnp.asarray(epsilon)

    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)

    def _update(mode_prev: Array, args: tuple[Array, Array]) -> tuple[Array, Array]:
        """Steepest Descent Point Process Filter update step"""
        x_t, spike_indicator_t = args
        conditional_intensity = _safe_expected_count(
            log_receptive_field_model(x_t, mode_prev), dt
        )
        innovation = spike_indicator_t - conditional_intensity
        one_step_grad = grad_log_receptive_field_model(x_t, mode_prev)
        posterior_mode = mode_prev + epsilon_arr @ one_step_grad * innovation

        return posterior_mode, posterior_mode

    return jax.lax.scan(_update, init_mean_params_arr, (x_arr, spike_indicator_arr))[1]
