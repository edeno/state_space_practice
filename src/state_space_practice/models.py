import functools
import logging
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.point_process_kalman import (  # noqa: F401 -- re-exports
    _safe_expected_count,
    steepest_descent_point_process_filter,
)
from state_space_practice.point_process_kalman import (
    get_confidence_interval as _get_confidence_interval,
)
from state_space_practice.utils import (
    clip_eigenvalues_relative,
    contains_tracer,
    psd_solve,
)

logger = logging.getLogger(__name__)


def log_receptive_field_model(position: ArrayLike, params: ArrayLike) -> Array:
    """Log firing rate of a 1D Gaussian place field.

    ``log(rate) = log_max_rate - (position - center)**2 / (2 * scale**2)``.

    Parameters
    ----------
    position : ArrayLike, scalar or shape (n_time,)
        Position(s) at which to evaluate the field.
    params : ArrayLike, shape (3,)
        ``(log_max_rate, place_field_center, scale)``: log of the peak rate,
        field center (same units as ``position``) and Gaussian width.

    Returns
    -------
    log_rate : Array, same shape as ``position``
        Log firing rate at each position.
    """
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

    .. deprecated:: 0.1.0
        This implementation uses the **observed Hessian** (not Fisher scoring)
        and may produce indefinite posterior precision matrices. Use
        :func:`point_process_kalman.stochastic_point_process_filter` instead,
        which uses Fisher scoring and is numerically more stable. This function
        will be removed in version 0.2.0.

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
        "Use point_process_kalman.stochastic_point_process_filter instead. "
        "It will be removed in version 0.2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    posterior_mode, posterior_covariance, n_floored = (
        _stochastic_point_process_filter_impl(
            jnp.asarray(init_mode_params),
            jnp.asarray(init_covariance_params),
            jnp.asarray(x),
            jnp.asarray(spike_indicator),
            dt,
            jnp.asarray(transition_matrix),
            jnp.asarray(latent_state_covariance),
            log_receptive_field_model=log_receptive_field_model,
        )
    )
    if not contains_tracer(n_floored) and int(n_floored) > 0:
        logger.warning(
            "models.stochastic_point_process_filter: raised %d eigenvalue(s) "
            "of the posterior precision / covariance to the scale-relative "
            "PSD floor (the observed-Hessian update was indefinite). Use "
            "point_process_kalman.stochastic_point_process_filter instead.",
            int(n_floored),
        )
    return posterior_mode, posterior_covariance


@functools.partial(jax.jit, static_argnames=("log_receptive_field_model",))
def _stochastic_point_process_filter_impl(
    init_mode_params: Array,
    init_covariance_params: Array,
    x: Array,
    spike_indicator: Array,
    dt: float,
    transition_matrix: Array,
    latent_state_covariance: Array,
    *,
    log_receptive_field_model: Callable[[ArrayLike, ArrayLike], Array],
) -> tuple[Array, Array, Array]:
    """Jitted scan of the deprecated observed-Hessian SSPPF.

    ``log_receptive_field_model`` is static (hashed by identity), so repeated
    calls with the same model function reuse one compilation; ``dt`` and all
    arrays are traced.

    Returns
    -------
    posterior_mode : Array, shape (n_time, n_params)
    posterior_covariance : Array, shape (n_time, n_params, n_params)
    n_floored : Array
        Total number of eigenvalues raised to the relative PSD floor.
    """
    # Compute the gradient and hessian of the log receptive field model
    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)
    hess_log_receptive_field_model = jax.hessian(log_receptive_field_model, argnums=1)

    # Define the update step
    def _update(
        params_prev: tuple[Array, Array, Array],
        args: tuple[Array, Array],
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array]]:
        """Point Process Adaptive Filter update step

        F : transition matrix
        Q : covariance matrix
        \theta_{k | k-1} :
        W_{k | k-1}: one_step_variance_params
        \theta_{k | k} : posterior_mode
        W_{k | k} : posterior_variance
        """

        # Unpack previous parameters
        mode_prev, covariance_prev, n_floored = params_prev
        x_t, spike_indicator_t = args

        # One-step prediction
        one_step_mean = transition_matrix @ mode_prev
        one_step_variance = (
            transition_matrix @ covariance_prev @ transition_matrix.T
            + latent_state_covariance
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
        # Scale-relative eigenvalue floors (the observed Hessian can make the
        # precision indefinite); counted and reported once after the scan.
        inverse_posterior_covariance, n_prec = clip_eigenvalues_relative(
            inverse_posterior_covariance
        )
        posterior_covariance = psd_solve(inverse_posterior_covariance, identity)
        posterior_covariance, n_cov = clip_eigenvalues_relative(posterior_covariance)

        # sum over one_step_grad.squeeze() * innovation if multiple neurons
        posterior_mode = one_step_mean + posterior_covariance @ (
            one_step_grad.squeeze() * innovation
        )

        return (posterior_mode, posterior_covariance, n_floored + n_prec + n_cov), (
            posterior_mode,
            posterior_covariance,
        )

    # Run the SSPPF
    (_, _, n_floored), (posterior_mode, posterior_covariance) = jax.lax.scan(
        _update,
        (init_mode_params, init_covariance_params, jnp.zeros((), dtype=jnp.int32)),
        (x, spike_indicator),
    )
    return posterior_mode, posterior_covariance, n_floored


def get_confidence_interval(
    posterior_mode: ArrayLike, posterior_covariance: ArrayLike, alpha: float = 0.05
) -> Array:
    """Get the confidence interval from the posterior covariance

    Parameters
    ----------
    posterior_mode : ArrayLike, shape (n_time, n_params)
    posterior_covariance : ArrayLike, shape (n_time, n_params, n_params)
    alpha : float, optional
        Significance level in ``(0, 1)``, by default ``0.05``. Returns a
        ``1 - alpha`` confidence interval (i.e. the default 0.05 gives a
        95% CI, not a 5% CI). Matches the default in
        :func:`point_process_kalman.get_confidence_interval`, which computes it.

    Returns
    -------
    ci : Array, shape (n_time, n_params, 2)
        Lower and upper bounds, ``posterior_mode -/+ z * sqrt(diag(cov))``.
    """
    return _get_confidence_interval(posterior_mode, posterior_covariance, alpha=alpha)
