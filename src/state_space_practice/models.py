import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.point_process_kalman import (  # noqa: F401 -- re-exports
    _safe_expected_count,
    get_confidence_interval,
    steepest_descent_point_process_filter,
)
from state_space_practice.utils import psd_solve, stabilize_covariance


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
    """Stochastic State Point Process Filter (SSPPF).

    .. deprecated::
        This implementation uses the **observed Hessian** (not Fisher scoring)
        and may produce indefinite posterior precision matrices. Use
        :func:`point_process_kalman.stochastic_point_process_filter` instead,
        which uses Fisher scoring and is numerically more stable. This function
        will be removed in a future version.

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
    warnings.warn(
        "stochastic_point_process_filter in models.py uses the observed Hessian "
        "(not Fisher scoring) and may produce indefinite posterior precision. "
        "Use point_process_kalman.stochastic_point_process_filter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
