"""Kalman filter and smoother.


References
----------
1. Sarkka, S. (2013). Bayesian Filtering and Smoothing
  (Cambridge University Press) https://doi.org/10.1017/CBO9781139344203.

"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg


def symmetrize(A):
    """
    Symmetrize one or more matrices by averaging each matrix with its transpose.

    Parameters
    ----------
    A : array_like
        A matrix or a batch of matrices to be symmetrized. The last two dimensions
        should be square matrices.

    Returns
    -------
    array_like
        The symmetrized matrix or batch of matrices, where each output matrix is
        (A + A.T) / 2.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> A = jnp.array([[1, 2], [3, 4]])
    >>> symmetrize(A)
    DeviceArray([[1., 2.5],
                 [2.5, 4.]], dtype=float32)
    """
    """Symmetrize one or more matrices."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def psd_solve(A, b, diagonal_boost=1e-9):
    """
    Solves a linear system Ax = b for positive semi-definite (PSD) matrices A.

    This function wraps a linear algebra solver, ensuring numerical stability by symmetrizing
    the input matrix A and adding a small value to its diagonal (diagonal_boost). It is intended
    for use with PSD matrices, where 'assume_a="pos"' can be safely set for performance.

    Parameters
    ----------
    A : jnp.ndarray
        The coefficient matrix, expected to be positive semi-definite.
    b : jnp.ndarray
        The right-hand side vector or matrix.
    diagonal_boost : float, optional
        Small value added to the diagonal of A to improve numerical stability. Default is 1e-9.

    Returns
    -------
    jnp.ndarray
        The solution x to the linear system Ax = b.
    """
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    return jax.scipy.linalg.solve(
        symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1]), b, assume_a="pos"
    )


def _kalman_filter_update(
    mean_prev: jnp.ndarray,
    cov_prev: jnp.ndarray,
    obs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    measurement_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
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
    obs_cov = symmetrize(
        measurement_matrix @ one_step_cov @ measurement_matrix.T + measurement_cov
    )

    residual_error = obs - obs_mean  # innovation
    kalman_gain = psd_solve(obs_cov, measurement_matrix @ one_step_cov).T

    posterior_mean = one_step_mean + kalman_gain @ residual_error
    # posterior_cov = one_step_cov - kalman_gain @ obs_cov @ kalman_gain.T
    # subtraction could result in the diagonal matrix with negative values
    # More stable solution is P = (I-KH)P(I-KH)' + KRK' to ensure positive semidefinite
    # This is known as the Joseph form covariance update
    n_cont = mean_prev.shape[0]
    I_KH = jnp.eye(n_cont) - kalman_gain @ measurement_matrix
    posterior_cov = symmetrize(
        I_KH @ one_step_cov @ I_KH.T + kalman_gain @ (measurement_cov @ kalman_gain.T)
    )

    marginal_log_likelihood = jax.scipy.stats.multivariate_normal.logpdf(
        x=obs, mean=obs_mean, cov=obs_cov
    )

    return posterior_mean, posterior_cov, marginal_log_likelihood


def kalman_filter(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    measurement_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """

    Parameters
    ----------
    init_mean : jnp.ndarray, shape (n_cont_states,)
    init_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
    obs : jnp.ndarray, shape (n_time, n_obs_dim)
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states)
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim)

    Returns
    -------
    filtered_mean : jnp.ndarray, shape (n_time, n_cont_states)
    filtered_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states)
    marginal_log_likelihood : float
        The log likelihood of the observations given the model parameters.
    """

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
    next_smoother_mean: jnp.ndarray,
    next_smoother_cov: jnp.ndarray,
    filter_mean: jnp.ndarray,
    filter_cov: jnp.ndarray,
    process_cov: jnp.ndarray,
    transition_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """

    Parameters
    ----------
    next_smoother_mean : jnp.ndarray, shape (n_cont_states,)
    next_smoother_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
    filter_mean : jnp.ndarray, shape (n_cont_states,)
    filter_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)

    Returns
    -------
    smoother_mean : jnp.ndarray, shape (n_cont_states,)
    smoother_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
    smoother_cross_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
    """
    one_step_mean = transition_matrix @ filter_mean
    one_step_cov = symmetrize(
        transition_matrix @ filter_cov @ transition_matrix.T + process_cov
    )

    smoother_kalman_gain = psd_solve(one_step_cov, transition_matrix @ filter_cov).T

    smoother_mean = filter_mean + smoother_kalman_gain @ (
        next_smoother_mean - one_step_mean
    )
    smoother_cov = symmetrize(
        filter_cov
        + smoother_kalman_gain
        @ (next_smoother_cov - one_step_cov)
        @ smoother_kalman_gain.T
    )
    # lag one cross covariance
    smoother_cross_cov = smoother_kalman_gain @ next_smoother_cov

    return smoother_mean, smoother_cov, smoother_cross_cov


def kalman_smoother(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    measurement_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """

    Parameters
    ----------
    init_mean : jnp.ndarray, shape (n_cont_states,)
    init_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
    obs : jnp.ndarray, shape (n_time, n_obs_dim)
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states)
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim)

    Returns
    -------
    smoother_mean : jnp.ndarray, shape (n_time, n_cont_states)
        Smoothed state mean.
    smoother_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states)
        Smoothed state covariance.
    smoother_cross_cov : jnp.ndarray, shape (n_time - 1, n_cont_states, n_cont_states)
        Smoothed cross-covariance between states at time t and t+1.
    marginal_log_likelihood : float
        The log likelihood of the observations given the model parameters.
    """
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


def outer_sum(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the sum of outer products between corresponding vectors in two sequences.

    Given two arrays `x` and `y` of shape (T, N) and (T, M) respectively, this function computes:
        S = sum_{t=1}^T x_t y_t^T
    where x_t and y_t are the t-th rows of `x` and `y`.

    Parameters
    ----------
    x : np.ndarray, shape (T, N)
    y : np.ndarray, shape (T, M)

    Returns
    -------
    outer_sum : np.ndarray, shape (N, M)
    """
    return x.T @ y


def kalman_maximization_step(
    obs: jnp.ndarray,
    smoother_mean: jnp.ndarray,
    smoother_cov: jnp.ndarray,
    smoother_cross_cov: jnp.ndarray,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
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

    n_time = obs.shape[0]

    # Compute intermediate expectation terms
    gamma = jnp.sum(smoother_cov, axis=0) + outer_sum(smoother_mean, smoother_mean)
    delta = outer_sum(obs, smoother_mean)
    alpha = outer_sum(obs, obs)
    gamma1 = gamma - jnp.outer(smoother_mean[-1], smoother_mean[-1]) - smoother_cov[-1]
    gamma2 = gamma - jnp.outer(smoother_mean[0], smoother_mean[0]) - smoother_cov[0]
    beta = (
        smoother_cross_cov.sum(axis=0)
        + outer_sum(smoother_mean[:-1], smoother_mean[1:])
    ).T

    # Measurement matrix and covariance
    measurement_matrix = psd_solve(gamma, delta.T).T
    measurement_cov = (alpha - measurement_matrix @ delta.T) / n_time
    measurement_cov = symmetrize(measurement_cov)

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
        measurement_matrix,
        process_cov,
        measurement_cov,
        init_mean,
        init_cov,
    )
