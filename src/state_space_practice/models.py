from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def log_receptive_field_model(position: ArrayLike, params: ArrayLike) -> Array:
    log_max_rate, place_field_center, scale = params
    return log_max_rate - (position - place_field_center) ** 2 / (2 * scale**2)


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
        one_step_mean = transition_matrix @ mode_prev
        one_step_variance = (
            transition_matrix @ covariance_prev @ transition_matrix.T
            + latent_state_covariance
        )

        # Compute the conditional intensity and innovation
        conditional_intensity = (
            jnp.exp(log_receptive_field_model(x_t, one_step_mean)) * dt
        )
        innovation = spike_indicator_t - conditional_intensity

        # Compute the posterior mean and variance
        one_step_grad = grad_log_receptive_field_model(x_t, one_step_mean)[None]
        one_step_hess = hess_log_receptive_field_model(x_t, one_step_mean)

        # sum over:
        # (one_step_grad.T * conditional_intensity @ one_step_grad) - innovation * one_step_hess
        # if multiple neurons
        inverse_posterior_covariance = (
            jnp.linalg.pinv(one_step_variance)
            + (one_step_grad.T * conditional_intensity @ one_step_grad)
            - innovation * one_step_hess
        )
        posterior_covariance = jnp.linalg.pinv(inverse_posterior_covariance)

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
        _update, (init_mode_params, init_covariance_params), (x, spike_indicator)
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
    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)

    def _update(
        mode_prev: Array, args: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        """Steepest Descent Point Process Filter update step"""
        x_t, spike_indicator_t = args
        conditional_intensity = jnp.exp(log_receptive_field_model(x_t, mode_prev)) * dt
        innovation = spike_indicator_t - conditional_intensity
        one_step_grad = grad_log_receptive_field_model(x_t, mode_prev)
        posterior_mode = mode_prev + epsilon @ one_step_grad * innovation

        return posterior_mode, posterior_mode

    return jax.lax.scan(_update, init_mean_params, (x, spike_indicator))[1]
