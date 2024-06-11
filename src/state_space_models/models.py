import jax
import jax.numpy as jnp


def log_receptive_field_model(position: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    log_max_rate, place_field_center, scale = params
    return log_max_rate - (position - place_field_center) ** 2 / (2 * scale**2)


def receptive_field_model(position: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(log_receptive_field_model(position, params))


def stochastic_point_process_filter(
    init_mean_params: jnp.ndarray,
    init_variance_params: jnp.ndarray,
    x: jnp.ndarray,
    spike_indicator: jnp.ndarray,
    dt: float,
    transition_matrix: jnp.ndarray,
    covariance_matrix: jnp.ndarray,
    log_receptive_field_model: callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stochastic State Point Process Filter (SSPPF)

    Parameters
    ----------
    init_mean_params : jnp.ndarray, shape (n_params,)
        Initial mean parameters
    init_variance_params : jnp.ndarray, shape (n_params, n_params)
        Initial variance parameters
    x : jnp.ndarray, shape (n_time,)
        Continuous-valued input signal
    spike_indicator : jnp.ndarray, shape (n_time,)
        Spike count
    dt : float
        Time step
    transition_matrix : jnp.ndarray, shape (n_params, n_params)
    covariance_matrix : jnp.ndarray, shape (n_params, n_params)
    log_receptive_field_model : callable
        Function that takes in `x` and parameters and returns the log spike rate

    Returns
    -------
    posterior_mean : jnp.ndarray, shape (n_time, n_params)
    posterior_variance : jnp.ndarray, shape (n_time, n_params, n_params)

    References
    ----------
    ...[1] Eden, U. T., Frank, L. M., Barbieri, R., Solo, V. & Brown, E. N.
      Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
      Neural Computation 16, 971-998 (2004).


    """
    # Compute the gradient and hessian of the log receptive field model
    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)
    hess_log_receptive_field_model = jax.hessian(log_receptive_field_model, argnums=1)

    # Define the update step
    def _update(
        params_prev: tuple[jnp.ndarray, jnp.ndarray],
        args: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
        """Point Process Adaptive Filter update step

        F : transition matrix
        Q : covariance matrix
        \theta_{k | k-1} :
        W_{k | k-1}: one_step_variance_params
        \theta_{k | k} : posterior_mean
        W_{k | k} : posterior_variance
        """

        # Unpack previous parameters
        mean_prev, variance_prev = params_prev
        x_t, spike_indicator_t = args

        # One-step prediction
        one_step_mean = transition_matrix @ mean_prev
        one_step_variance = (
            transition_matrix @ variance_prev @ transition_matrix.T + covariance_matrix
        )

        # Compute the conditional intensity and innovation
        conditional_intensity = (
            jnp.exp(log_receptive_field_model(x_t, one_step_mean)) * dt
        )
        innovation = spike_indicator_t - conditional_intensity

        # Compute the posterior mean and variance
        one_step_grad = grad_log_receptive_field_model(x_t, one_step_mean)[None]
        one_step_hess = hess_log_receptive_field_model(x_t, one_step_mean)

        inverse_posterior_variance = (
            jnp.linalg.pinv(one_step_variance)
            + (one_step_grad.T * conditional_intensity @ one_step_grad)
            - innovation * one_step_hess
        )
        posterior_variance = jnp.linalg.pinv(inverse_posterior_variance)
        posterior_mean = one_step_mean + posterior_variance @ (
            one_step_grad.squeeze() * innovation
        )

        return (posterior_mean, posterior_variance), (
            posterior_mean,
            posterior_variance,
        )

    # Run the SSPPF
    return jax.lax.scan(
        _update, (init_mean_params, init_variance_params), (x, spike_indicator)
    )[1]


def steepest_descent_point_process_filter(
    init_mean_params: jnp.ndarray,
    x: jnp.ndarray,
    spike_indicator: jnp.ndarray,
    dt: float,
    epsilon: jnp.ndarray,
    log_receptive_field_model: callable,
) -> jnp.ndarray:
    """Steepest Descent Point Process Filter (SDPPF)

    Parameters
    ----------
    init_mean_params : jnp.ndarray, shape (n_params,)
    x : jnp.ndarray, shape (n_time,)
        Continuous-valued input signal
    spike_indicator : jnp.ndarray, shape (n_time,)
        Spike count
    dt : float
        Time step
    epsilon : jnp.ndarray, shape (n_params, n_params)
        Learning rate
    log_receptive_field_model : callable
        Function that takes in `x` and parameters and returns the log spike rate

    Returns
    -------
    posterior_mean : jnp.ndarray, shape (n_time, n_params)
    """
    grad_log_receptive_field_model = jax.grad(log_receptive_field_model, argnums=1)

    def _update(mean_prev: jnp.ndarray, args: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Steepest Descent Point Process Filter update step"""

        x_t, spike_indicator_t = args
        conditional_intensity = jnp.exp(log_receptive_field_model(x_t, mean_prev)) * dt
        innovation = spike_indicator_t - conditional_intensity
        one_step_grad = grad_log_receptive_field_model(x_t, mean_prev)
        posterior_mean = mean_prev + epsilon @ one_step_grad * innovation

        return posterior_mean, posterior_mean

    return jax.lax.scan(_update, init_mean_params, (x, spike_indicator))[1]
