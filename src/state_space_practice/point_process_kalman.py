import jax
import jax.numpy as jnp

from state_space_practice.kalman import (
    _kalman_smoother_update,
    outer_sum,
    psd_solve,
    symmetrize,
)


def log_conditional_intensity(design_matrix, params):
    return design_matrix @ params


def stochastic_point_process_filter(
    init_mean_params: jnp.ndarray,
    init_covariance_params: jnp.ndarray,
    design_matrix: jnp.ndarray,
    spike_indicator: jnp.ndarray,
    dt: float,
    transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    log_conditional_intensity: callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stochastic State Point Process Filter (SSPPF)

    Parameters
    ----------
    init_mean_params : jnp.ndarray, shape (n_params,)
        Initial mean parameters
    init_covariance_params : jnp.ndarray, shape (n_params, n_params)
        Initial variance parameters
    design_matrix : jnp.ndarray, shape (n_time, n_params)
    spike_indicator : jnp.ndarray, shape (n_time,)
        Spike count
    dt : float
        Time step
    transition_matrix : jnp.ndarray, shape (n_params, n_params)
    process_cov : jnp.ndarray, shape (n_params, n_params)
    log_conditional_intensity : callable
        Function that takes in `design_matrix` and parameters and returns the log spike rate

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
    grad_log_conditional_intensity = jax.jacfwd(log_conditional_intensity, argnums=1)
    hess_log_conditional_intensity = jax.hessian(log_conditional_intensity, argnums=1)

    def _step(
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
        mean_prev, variance_prev, marginal_log_likelihood = params_prev
        design_matrix_t, spike_indicator_t = args

        # One-step prediction
        one_step_mean = transition_matrix @ mean_prev
        one_step_covariance = (
            transition_matrix @ variance_prev @ transition_matrix.T + process_cov
        )
        one_step_covariance = symmetrize(one_step_covariance)

        # Compute the conditional intensity and innovation
        conditional_intensity = (
            jnp.exp(log_conditional_intensity(design_matrix_t, one_step_mean)) * dt
        )
        innovation = spike_indicator_t - conditional_intensity

        # Compute the posterior mean and variance
        one_step_grad = grad_log_conditional_intensity(design_matrix_t, one_step_mean)[
            None
        ]
        one_step_hess = hess_log_conditional_intensity(design_matrix_t, one_step_mean)

        inverse_posterior_covariance = (
            jnp.linalg.pinv(one_step_covariance)
            + (one_step_grad.T * conditional_intensity @ one_step_grad)
            - innovation * one_step_hess
        )
        posterior_covariance = jnp.linalg.pinv(inverse_posterior_covariance)
        posterior_mean = one_step_mean + posterior_covariance @ (
            one_step_grad.squeeze() * innovation
        )
        marginal_log_likelihood += jax.scipy.stats.poisson.logpmf(
            k=spike_indicator_t, mu=conditional_intensity
        )

        return (posterior_mean, posterior_covariance, marginal_log_likelihood), (
            posterior_mean,
            posterior_covariance,
        )

    marginal_log_likelihood = 0.0
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
    init_mean_params: jnp.ndarray,
    init_covariance_params: jnp.ndarray,
    design_matrix: jnp.ndarray,
    spike_indicator: jnp.ndarray,
    dt: float,
    transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    log_conditional_intensity: callable,
):
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
    smoother_mean: jnp.ndarray,
    smoother_cov: jnp.ndarray,
    smoother_cross_cov: jnp.ndarray,
):
    """Maximization step for the Kalman filter.

    Parameters
    ----------
    smoother_mean : jnp.ndarray, shape (n_time, n_cont_states)
        smoother mean.
    smoother_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states)
        smoother covariance.
    smoother_cross_cov : jnp.ndarray, shape (n_time - 1, n_cont_states, n_cont_states)
        smoother cross-covariance.

    Returns
    -------
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Transition matrix.
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Process covariance.
    mean_init : jnp.ndarray, shape (n_cont_states,)
        Initial mean.
    cov_init : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Initial covariance.

    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """

    n_time = smoother_mean.shape[0]

    # Compute intermediate expectation terms
    gamma = jnp.sum(smoother_cov, axis=0) + outer_sum(smoother_mean, smoother_mean)
    gamma1 = gamma - jnp.outer(smoother_mean[-1], smoother_mean[-1]) - smoother_cov[-1]
    gamma2 = gamma - jnp.outer(smoother_mean[0], smoother_mean[0]) - smoother_cov[0]
    beta = smoother_cross_cov.sum(axis=0).T

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
    posterior_mean: jnp.ndarray, posterior_covariance: jnp.ndarray, alpha: float = 0.01
) -> jnp.ndarray:
    """Get the confidence interval from the posterior covariance

    Parameters
    ----------
    posterior_mean : jnp.ndarray, shape (n_time, n_params)
    posterior_covariance : jnp.ndarray, shape (n_time, n_params, n_params)
    alpha : float, optional
        Confidence level, by default 0.01
    """
    z = jax.scipy.stats.norm.ppf(1 - alpha / 2)
    ci = z * jnp.sqrt(
        jnp.diagonal(posterior_covariance, axis1=-2, axis2=-1)
    )  # shape (n_time, n_params)

    return jnp.stack((posterior_mean - ci, posterior_mean + ci), axis=-1)


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
        mean_prev: jnp.ndarray, args: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Steepest Descent Point Process Filter update step"""
        x_t, spike_indicator_t = args
        conditional_intensity = jnp.exp(log_receptive_field_model(x_t, mean_prev)) * dt
        innovation = spike_indicator_t - conditional_intensity
        one_step_grad = grad_log_receptive_field_model(x_t, mean_prev)
        posterior_mean = mean_prev + epsilon @ one_step_grad * innovation

        return posterior_mean, posterior_mean

    return jax.lax.scan(_update, init_mean_params, (x, spike_indicator))[1]
