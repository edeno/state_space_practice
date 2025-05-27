"""Kalman filter and smoother for linear Gaussian state-space models.

Implements the Kalman filter, Rauch-Tung-Striebel (RTS) smoother, and
the Expectation-Maximization (EM) algorithm's M-step for parameter estimation.

The assumed state-space model is:
$$ x_t = A x_{t-1} + w_t, \quad w_t \sim N(0, \Sigma) $$
$$ y_t = H x_t + v_t, \quad v_t \sim N(0, R) $$

References
----------
1. Sarkka, S. (2013). Bayesian Filtering and Smoothing
  (Cambridge University Press) https://doi.org/10.1017/CBO9781139344203.
2. Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
   linear Gaussian models. Neural computation, 11(2), 305-345.

"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.stats.multivariate_normal
from jax import typing


def symmetrize(A: jnp.ndarray) -> jnp.ndarray:
    """Symmetrize one or more matrices by averaging each matrix with its transpose.

    Parameters
    ----------
    A : jnp.ndarray
        A matrix or a batch of matrices to be symmetrized. The last two
        dimensions should be square matrices.

    Returns
    -------
    jnp.ndarray
        The symmetrized matrix or batch of matrices, where each output matrix
        is (A + A.T) / 2.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> A = jnp.array([[1, 2], [3, 4]])
    >>> symmetrize(A)
    DeviceArray([[1. , 2.5],
                 [2.5, 4. ]], dtype=float32)

    """
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def psd_solve(
    A: jnp.ndarray, b: jnp.ndarray, diagonal_boost: float = 1e-9
) -> jnp.ndarray:
    """Solves a linear system Ax = b for positive semi-definite (PSD) matrices A.

    This function wraps a linear algebra solver, ensuring numerical stability
    by symmetrizing the input matrix A and adding a small value to its
    diagonal (diagonal_boost). It is intended for use with PSD matrices,
    where 'assume_a="pos"' can be safely set for performance.

    Parameters
    ----------
    A : jnp.ndarray
        The coefficient matrix, expected to be positive semi-definite.
    b : jnp.ndarray
        The right-hand side vector or matrix.
    diagonal_boost : float, optional
        Small value added to the diagonal of A to improve numerical
        stability. Default is 1e-9.

    Returns
    -------
    jnp.ndarray
        The solution x to the linear system Ax = b.

    """
    return jax.scipy.linalg.solve(
        symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1], dtype=A.dtype),
        b,
        assume_a="pos",
    )


@jax.jit
def _kalman_filter_update(
    mean_prev: jnp.ndarray,
    cov_prev: jnp.ndarray,
    obs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    measurement_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Performs a single update step of the Kalman filter.

    Parameters
    ----------
    mean_prev : jnp.ndarray, shape (n_cont_states,)
        Previous state mean, $$ m_{t-1} $$.
    cov_prev : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Previous state covariance, $$ P_{t-1} $$.
    obs : jnp.ndarray, shape (n_obs_dim,)
        Data observation, $$ y_t $$.
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \Sigma $$.
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states)
        Observation matrix, $$ H $$.
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance, $$ R $$.

    Returns
    -------
    posterior_mean : jnp.ndarray, shape (n_cont_states,)
        Posterior state mean, $$ m_t $$.
    posterior_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Posterior state covariance, $$ P_t $$.
    marginal_log_likelihood : float
        Log-likelihood of the observation, $$ \log p(y_t | y_{1:t-1}) $$.

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


@jax.jit
def kalman_filter(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    measurement_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Applies the Kalman filter to a sequence of observations.

    Parameters
    ----------
    init_mean : jnp.ndarray, shape (n_cont_states,)
        Initial state mean, $$ m_0 $$.
    init_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Initial state covariance, $$ P_0 $$.
    obs : jnp.ndarray, shape (n_time, n_obs_dim)
        Sequence of observations, $$ y_{1:T} $$.
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \Sigma $$.
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states)
        Observation matrix, $$ H $$.
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance, $$ R $$.

    Returns
    -------
    filtered_mean : jnp.ndarray, shape (n_time, n_cont_states)
        Filtered state means, $$ m_{1:T} $$.
    filtered_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states)
        Filtered state covariances, $$ P_{1:T} $$.
    marginal_log_likelihood : float
        Total log likelihood of the observations, $$ \sum_{t=1}^T \log p(y_t | y_{1:t-1}) $$.

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

    marginal_log_likelihood = jnp.array(0.0)
    (_, _, marginal_log_likelihood), (
        filtered_mean,
        filtered_cov,
    ) = jax.lax.scan(
        _step,
        (init_mean, init_cov, marginal_log_likelihood),
        obs,
    )

    return filtered_mean, filtered_cov, marginal_log_likelihood


@jax.jit
def _kalman_smoother_update(
    next_smoother_mean: jnp.ndarray,
    next_smoother_cov: jnp.ndarray,
    filter_mean: jnp.ndarray,
    filter_cov: jnp.ndarray,
    process_cov: jnp.ndarray,
    transition_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Performs a single backward update step of the RTS smoother.

    Parameters
    ----------
    next_smoother_mean : jnp.ndarray, shape (n_cont_states,)
        Smoothed mean from the next time step, $$ m_{t+1|T} $$.
    next_smoother_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Smoothed covariance from the next time step, $$ P_{t+1|T} $$.
    filter_mean : jnp.ndarray, shape (n_cont_states,)
        Filtered mean from the current time step, $$ m_{t|t} $$.
    filter_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Filtered covariance from the current time step, $$ P_{t|t} $$.
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \Sigma $$.
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.

    Returns
    -------
    smoother_mean : jnp.ndarray, shape (n_cont_states,)
        Smoothed state mean, $$ m_{t|T} $$.
    smoother_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Smoothed state covariance, $$ P_{t|T} $$.
    smoother_cross_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Smoothed cross-covariance, $$ P_{t, t+1|T} $$.

    """
    # Predicted mean m_{t+1|t}
    one_step_mean = transition_matrix @ filter_mean
    # Predicted covariance P_{t+1|t}
    one_step_cov = symmetrize(
        transition_matrix @ filter_cov @ transition_matrix.T + process_cov
    )

    # Smoother gain J_t
    smoother_kalman_gain = psd_solve(one_step_cov, transition_matrix @ filter_cov).T

    # Smoothed mean m_{t|T}
    smoother_mean = filter_mean + smoother_kalman_gain @ (
        next_smoother_mean - one_step_mean
    )
    # Smoothed covariance P_{t|T}
    smoother_cov = symmetrize(
        filter_cov
        + smoother_kalman_gain
        @ (next_smoother_cov - one_step_cov)
        @ smoother_kalman_gain.T
    )
    # Lag-one cross covariance P_{t, t+1|T}
    smoother_cross_cov = smoother_kalman_gain @ next_smoother_cov

    return smoother_mean, smoother_cov, smoother_cross_cov


@jax.jit
def kalman_smoother(
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs: jnp.ndarray,
    transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    measurement_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Applies the Rauch-Tung-Striebel (RTS) smoother.

    Parameters
    ----------
    init_mean : jnp.ndarray, shape (n_cont_states,)
        Initial state mean, $$ m_0 $$.
    init_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Initial state covariance, $$ P_0 $$.
    obs : jnp.ndarray, shape (n_time, n_obs_dim)
        Sequence of observations, $$ y_{1:T} $$.
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \Sigma $$.
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states)
        Observation matrix, $$ H $$.
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance, $$ R $$.

    Returns
    -------
    smoother_mean : jnp.ndarray, shape (n_time, n_cont_states)
        Smoothed state means, $$ m_{1:T|T} $$.
    smoother_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states)
        Smoothed state covariances, $$ P_{1:T|T} $$.
    smoother_cross_cov : jnp.ndarray, shape (n_time - 1, n_cont_states, n_cont_states)
        Smoothed cross-covariances, $$ P_{t, t+1|T} $$.
    marginal_log_likelihood : float
        Total log likelihood of the observations.

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

    def _step(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        args: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ]:
        """Helper function for `jax.lax.scan` backward pass."""
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

    init_carry = (filtered_mean[-1], filtered_cov[-1])
    (_, _), (smoother_mean, smoother_cov, smoother_cross_cov) = jax.lax.scan(
        _step,
        init_carry,
        (filtered_mean[:-1], filtered_cov[:-1]),
        reverse=True,
    )

    smoother_mean = jnp.concatenate((smoother_mean, filtered_mean[-1][None]))
    smoother_cov = jnp.concatenate((smoother_cov, filtered_cov[-1][None]))

    return smoother_mean, smoother_cov, smoother_cross_cov, marginal_log_likelihood


def sum_of_outer_products(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute the sum of outer products between corresponding vectors.

    Computes $$ S = \sum_{t=1}^T x_t y_t^T $$.

    Parameters
    ----------
    x : jnp.ndarray, shape (T, N)
        First sequence of vectors.
    y : jnp.ndarray, shape (T, M)
        Second sequence of vectors.

    Returns
    -------
    jnp.ndarray, shape (N, M)
        The sum of outer products.

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
    """Performs the Maximization (M) step of the EM algorithm for Kalman filters.

    Updates the model parameters based on the expected sufficient statistics
    derived from the E-step (Kalman smoother).

    Parameters
    ----------
    obs : jnp.ndarray, shape (n_time, n_obs_dim)
        Observations, $$ y_{1:T} $$.
    smoother_mean : jnp.ndarray, shape (n_time, n_cont_states)
        Smoothed means, $$ m_{1:T|T} $$.
    smoother_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states)
        Smoothed covariances, $$ P_{1:T|T} $$.
    smoother_cross_cov : jnp.ndarray, shape (n_time - 1, n_cont_states, n_cont_states)
        Smoothed cross-covariances, $$ P_{t, t+1|T} $$.

    Returns
    -------
    transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Updated transition matrix, $$ A $$.
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states)
        Updated measurement matrix, $$ H $$.
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Updated process covariance, $$ \Sigma $$.
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim)
        Updated measurement covariance, $$ R $$.
    init_mean : jnp.ndarray, shape (n_cont_states,)
        Updated initial mean, $$ m_1 $$.
    init_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        Updated initial covariance, $$ P_1 $$.

    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """

    n_time: int = obs.shape[0]

    # Compute intermediate expectation terms
    gamma = jnp.sum(smoother_cov, axis=0) + sum_of_outer_products(
        smoother_mean, smoother_mean
    )
    delta = sum_of_outer_products(obs, smoother_mean)
    alpha = sum_of_outer_products(obs, obs)
    gamma1 = gamma - jnp.outer(smoother_mean[-1], smoother_mean[-1]) - smoother_cov[-1]
    gamma2 = gamma - jnp.outer(smoother_mean[0], smoother_mean[0]) - smoother_cov[0]
    beta = (
        smoother_cross_cov.sum(axis=0)
        + sum_of_outer_products(smoother_mean[:-1], smoother_mean[1:])
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
