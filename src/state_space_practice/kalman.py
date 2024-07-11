from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg


def _kalman_filter_update(
    mean_prev: jnp.ndarray,
    cov_prev: jnp.ndarray,
    obs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    measurement_cov: jnp.ndarray,
):
    """

    Parameters
    ----------
    mean_prev : jnp.ndarray, shape (n_cont_states,)
        Previous state mean.
    cov_prev : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Previous state covariance.
    obs : jnp.ndarray, shape (n_obs_dim,)
        Data observation. $y_t$
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State transition matrix. $A$
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State noise covariance. $\Sigma$
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states)
        Maps the observation to the state space. $H$
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance. $R$

    Returns
    -------
    posterior_mean : jnp.ndarray, shape (n_cont_states,)
        Posterior state mean.
    posterior_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Posterior state covariance.
    kalman_gain : jnp.ndarray, shape (n_cont_states, n_obs_dim)
        Kalman gain.
    marginal_log_likelihood : float
        Probability of the observation given the state.
    """

    # One step prediction
    one_step_mean = transition_matrix @ mean_prev
    one_step_cov = transition_matrix @ cov_prev @ transition_matrix.T + process_cov

    # Measurement update
    obs_mean = measurement_matrix @ one_step_mean
    # obs_cross_cov = one_step_cov @ measurement_matrix.T

    # project system uncertainty into measurement space
    obs_cov = measurement_matrix @ one_step_cov @ measurement_matrix.T + measurement_cov

    residual_error = obs - obs_mean  # innovation
    kalman_gain = jax.scipy.linalg.solve(
        obs_cov, measurement_matrix @ one_step_cov, assume_a="pos"
    ).T

    posterior_mean = one_step_mean + kalman_gain @ residual_error
    # posterior_cov = one_step_cov - kalman_gain @ obs_cov @ kalman_gain.T
    # subtraction could result in the diagonal matrix with negative values
    # More stable solution is P = (I-KH)P(I-KH)' + KRK' to ensure positive semidefinite
    # This is known as the Joseph form covariance update
    I_KH = jnp.eye(len(mean_prev)) - kalman_gain @ measurement_matrix
    posterior_cov = I_KH @ one_step_cov @ I_KH.T + kalman_gain @ (
        measurement_cov @ kalman_gain.T
    )

    marginal_log_likelihood = jax.scipy.stats.multivariate_normal.logpdf(
        x=residual_error, mean=jnp.zeros_like(residual_error), cov=obs_cov
    )

    return posterior_mean, posterior_cov, marginal_log_likelihood


def kalman_filter(
    init_mean,
    init_cov,
    obs,
    transition_matrix,
    process_cov,
    measurement_matrix,
    measurement_cov,
):
    def _step(carry, obs_t):
        mean_prev, cov_prev, marginal_log_likelihood = carry
        posterior_mean, posterior_cov, marginal_log_likelihood_t = (
            _kalman_filter_update(
                mean_prev,
                cov_prev,
                obs_t,
                transition_matrix,
                process_cov,
                measurement_matrix,
                measurement_cov,
            )
        )

        marginal_log_likelihood += marginal_log_likelihood_t

        return (posterior_mean, posterior_cov, marginal_log_likelihood), (
            posterior_mean,
            posterior_cov,
        )

    marginal_log_likelihood = 0.0
    (_, _, marginal_log_likelihood), (
        filtered_mean,
        filtered_cov,
    ) = jax.lax.scan(
        _step,
        (init_mean, init_cov, marginal_log_likelihood),
        obs,
    )

    return filtered_mean, filtered_cov, marginal_log_likelihood


def _kalman_smoother_update(
    next_smoother_mean,
    next_smoother_cov,
    filter_mean,
    filter_cov,
    process_cov,
    transition_matrix,
):
    one_step_mean = transition_matrix @ filter_mean
    one_step_cov = transition_matrix @ filter_cov @ transition_matrix.T + process_cov

    smoother_kalman_gain = psd_solve(one_step_cov, transition_matrix @ filter_cov).T
    # smoother_kalman_gain = jax.scipy.linalg.solve(
    #     one_step_cov, transition_matrix @ filter_cov, assume_a="pos"
    # ).T

    smoother_mean = filter_mean + smoother_kalman_gain @ (
        next_smoother_mean - one_step_mean
    )
    smoother_cov = (
        filter_cov
        + smoother_kalman_gain
        @ (next_smoother_cov - one_step_cov)
        @ smoother_kalman_gain.T
    )

    smoother_cross_cov = smoother_kalman_gain @ next_smoother_cov + jnp.outer(
        smoother_mean, next_smoother_mean
    )

    return smoother_mean, smoother_cov, smoother_cross_cov


def kalman_smoother(
    init_mean,
    init_cov,
    obs,
    transition_matrix,
    process_cov,
    measurement_matrix,
    measurement_cov,
):
    filtered_mean, filtered_cov, marginal_log_likelihood = kalman_filter(
        init_mean,
        init_cov,
        obs,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
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


def symmetrize(A):
    """Symmetrize one or more matrices."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def psd_solve(A, b, diagonal_boost=1e-9):
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = jax.scipy.linalg.cho_factor(A, lower=True)
    x = jax.scipy.linalg.cho_solve((L, lower), b)
    return x


# @jax.jit
def kalman_maximization_step(
    obs: jnp.ndarray,
    smoother_mean: jnp.ndarray,
    smoother_cov: jnp.ndarray,
    smoother_cross_cov: jnp.ndarray,
    outer_obs: Optional[jnp.ndarray] = None,
):
    """Maximization step for the Kalman filter.

    Parameters
    ----------
    obs : jnp.ndarray, shape (n_time, n_obs_dim)
        Observations.
    smoother_mean : jnp.ndarray, shape (n_time, n_cont_states)
        smoother mean.
    smoother_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states)
        smoother covariance.
    smoother_cross_cov : jnp.ndarray, shape (n_time - 1, n_cont_states, n_cont_states)
        smoother cross-covariance.
    outer_obs : jnp.ndarray, shape (n_obs_dim, n_obs_dim)
        Outer product of the observations. Default is None.

    Returns
    -------
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Transition matrix.
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states)
        Measurement matrix.
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Process covariance.
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim)
        Measurement covariance.
    mean_init : jnp.ndarray, shape (n_cont_states,)
        Initial mean.
    cov_init : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Initial covariance.

    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """
    if outer_obs is None:
        outer_obs = obs.T @ obs

    n_time = obs.shape[0]

    # outer products
    delta = obs.T @ smoother_mean
    # P_t = E[x_t, x_t]
    gamma = smoother_mean.T @ smoother_mean + jnp.sum(smoother_cov, axis=0)
    # P_{t, t-1} = E[x_t, x_{t-1}]
    beta = jnp.sum(smoother_cross_cov, axis=0)

    gamma1 = gamma - jnp.outer(smoother_mean[-1], smoother_mean[-1]) - smoother_cov[-1]
    gamma2 = gamma - jnp.outer(smoother_mean[0], smoother_mean[0]) - smoother_cov[0]

    # B = delta * gamma^-1 also sometimes C = delta * gamma^-1
    # measurement_matrix = delta @ jnp.linalg.inv(gamma)
    # measurement_matrix = jax.scipy.linalg.solve(gamma, delta.T).T
    measurement_matrix = psd_solve(gamma, delta.T).T

    # R = E[y_t, y_t] - C @ delta.T
    measurement_cov = (outer_obs - measurement_matrix @ delta.T) / n_time

    # A = beta * gamma1^-1
    # transition_matrix = beta @ jnp.linalg.inv(gamma1)
    # transition_matrix = jax.scipy.linalg.solve(gamma1, beta.T).T
    # transition_matrix = psd_solve(gamma1, beta.T).T

    # this matches the dynamax implementation
    transition_matrix = psd_solve(gamma1, beta).T
    # transition_matrix = jax.scipy.linalg.solve(gamma1.T, beta, assume_a="pos").T

    # Q = gamma2 - A @ beta.T
    # process_cov = (gamma2 - transition_matrix @ beta.T) / (n_time - 1)
    process_cov = (gamma2 - transition_matrix @ beta) / (
        n_time - 1
    )  # this matches the dynamax implementation

    mean_init = smoother_mean[0]
    cov_init = smoother_cov[0]

    return (
        transition_matrix,
        measurement_matrix,
        process_cov,
        measurement_cov,
        mean_init,
        cov_init,
    )
