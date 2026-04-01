"""Kalman filter and smoother for linear Gaussian state-space models.

Implements the Kalman filter, Rauch-Tung-Striebel (RTS) smoother, and
the Expectation-Maximization (EM) algorithm's M-step for parameter estimation.

The assumed state-space model is:
$$ x_t = A x_{t-1} + w_t, \\quad w_t \\sim N(0, \\Sigma) $$
$$ y_t = H x_t + v_t, \\quad v_t \\sim N(0, R) $$

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


def symmetrize(A: jax.Array) -> jax.Array:
    """Symmetrize one or more matrices by averaging each matrix with its transpose.

    Parameters
    ----------
    A : jax.Array
        A matrix or a batch of matrices to be symmetrized. The last two
        dimensions should be square matrices.

    Returns
    -------
    jax.Array
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


def psd_solve(A: jax.Array, b: jax.Array, diagonal_boost: float = 1e-9) -> jax.Array:
    """Solves a linear system Ax = b for positive semi-definite (PSD) matrices A.

    This function wraps a linear algebra solver, ensuring numerical stability
    by symmetrizing the input matrix A and adding a small value to its
    diagonal (diagonal_boost). It is intended for use with PSD matrices,
    where 'assume_a="pos"' can be safely set for performance.

    Parameters
    ----------
    A : jax.Array
        The coefficient matrix, expected to be positive semi-definite.
    b : jax.Array
        The right-hand side vector or matrix.
    diagonal_boost : float, optional
        Small value added to the diagonal of A to improve numerical
        stability. Default is 1e-9.

    Returns
    -------
    jax.Array
        The solution x to the linear system Ax = b.

    """
    return jax.scipy.linalg.solve(
        symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1], dtype=A.dtype),
        b,
        assume_a="pos",
    )


def project_psd(Q: jax.Array, min_eigenvalue: float = 1e-4) -> jax.Array:
    """Project a matrix onto the positive semi-definite cone.

    This function ensures the input matrix is positive semi-definite by:
    1. Computing its eigendecomposition
    2. Clipping eigenvalues to be at least `min_eigenvalue`
    3. Reconstructing the matrix from the clipped eigenvalues

    This is the standard approach for handling M-step updates in EM algorithms
    with approximate E-steps (e.g., Laplace-EKF for point-process observations),
    where the raw M-step can produce non-PSD covariance matrices.

    Parameters
    ----------
    Q : jax.Array
        A symmetric matrix to project onto the PSD cone. Shape (n, n).
    min_eigenvalue : float, optional
        Minimum eigenvalue to enforce. Default is 1e-4. This represents the
        minimum allowable variance in any direction.

    Returns
    -------
    jax.Array
        The projected PSD matrix with all eigenvalues >= min_eigenvalue.

    Notes
    -----
    - The input should be symmetric; non-symmetric matrices may give
      unexpected results.
    - Uses `jnp.linalg.eigh` which assumes symmetric input.
    - The min_eigenvalue should be chosen based on the scale of the problem:
      - Too small (< 1e-8): May not prevent numerical instability
      - Too large (> 0.1): May dominate learned values

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> Q = jnp.array([[1.0, 0.5], [0.5, -0.1]])  # Non-PSD
    >>> Q_psd = project_psd(Q, min_eigenvalue=1e-4)
    >>> jnp.linalg.eigvalsh(Q_psd)  # All eigenvalues >= 1e-4

    """
    # Compute eigendecomposition (eigh assumes symmetric input)
    eigvals, eigvecs = jnp.linalg.eigh(Q)

    # Clip eigenvalues to minimum
    eigvals_clipped = jnp.maximum(eigvals, min_eigenvalue)

    # Reconstruct matrix: Q = V @ diag(lambda) @ V.T
    return eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T


@jax.jit
def _kalman_filter_update(
    mean_prev: jax.Array,
    cov_prev: jax.Array,
    obs: jax.Array,
    transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Performs a single update step of the Kalman filter.

    Parameters
    ----------
    mean_prev : jax.Array, shape (n_cont_states,)
        Previous state mean, $$ m_{t-1} $$.
    cov_prev : jax.Array, shape (n_cont_states, n_cont_states)
        Previous state covariance, $$ P_{t-1} $$.
    obs : jax.Array, shape (n_obs_dim,)
        Data observation, $$ y_t $$.
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \\Sigma $$.
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states)
        Observation matrix, $$ H $$.
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance, $$ R $$.

    Returns
    -------
    posterior_mean : jax.Array, shape (n_cont_states,)
        Posterior state mean, $$ m_t $$.
    posterior_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Posterior state covariance, $$ P_t $$.
    marginal_log_likelihood : jax.Array
        Log-likelihood of the observation, $$ \\log p(y_t | y_{1:t-1}) $$ (scalar array).

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

    marginal_log_likelihood = jnp.asarray(
        jax.scipy.stats.multivariate_normal.logpdf(x=obs, mean=obs_mean, cov=obs_cov)
    )

    return posterior_mean, posterior_cov, marginal_log_likelihood


@jax.jit
def kalman_filter(
    init_mean: jax.Array,
    init_cov: jax.Array,
    obs: jax.Array,
    transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Applies the Kalman filter to a sequence of observations.

    Parameters
    ----------
    init_mean : jax.Array, shape (n_cont_states,)
        Initial state mean, $$ m_0 $$.
    init_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Initial state covariance, $$ P_0 $$.
    obs : jax.Array, shape (n_time, n_obs_dim)
        Sequence of observations, $$ y_{1:T} $$.
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \\Sigma $$.
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states)
        Observation matrix, $$ H $$.
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance, $$ R $$.

    Returns
    -------
    filtered_mean : jax.Array, shape (n_time, n_cont_states)
        Filtered state means, $$ m_{1:T} $$.
    filtered_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
        Filtered state covariances, $$ P_{1:T} $$.
    marginal_log_likelihood : jax.Array
        Total log likelihood of the observations, $$ \\sum_{t=1}^T \\log p(y_t | y_{1:t-1}) $$ (scalar array).

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
    next_smoother_mean: jax.Array,
    next_smoother_cov: jax.Array,
    filter_mean: jax.Array,
    filter_cov: jax.Array,
    process_cov: jax.Array,
    transition_matrix: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Performs a single backward update step of the RTS smoother.

    Parameters
    ----------
    next_smoother_mean : jax.Array, shape (n_cont_states,)
        Smoothed mean from the next time step, $$ m_{t+1|T} $$.
    next_smoother_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Smoothed covariance from the next time step, $$ P_{t+1|T} $$.
    filter_mean : jax.Array, shape (n_cont_states,)
        Filtered mean from the current time step, $$ m_{t|t} $$.
    filter_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Filtered covariance from the current time step, $$ P_{t|t} $$.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \\Sigma $$.
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.

    Returns
    -------
    smoother_mean : jax.Array, shape (n_cont_states,)
        Smoothed state mean, $$ m_{t|T} $$.
    smoother_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Smoothed state covariance, $$ P_{t|T} $$.
    smoother_cross_cov : jax.Array, shape (n_cont_states, n_cont_states)
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
    init_mean: jax.Array,
    init_cov: jax.Array,
    obs: jax.Array,
    transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Applies the Rauch-Tung-Striebel (RTS) smoother.

    Parameters
    ----------
    init_mean : jax.Array, shape (n_cont_states,)
        Initial state mean, $$ m_0 $$.
    init_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Initial state covariance, $$ P_0 $$.
    obs : jax.Array, shape (n_time, n_obs_dim)
        Sequence of observations, $$ y_{1:T} $$.
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        State transition matrix, $$ A $$.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        State noise covariance, $$ \\Sigma $$.
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states)
        Observation matrix, $$ H $$.
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance, $$ R $$.

    Returns
    -------
    smoother_mean : jax.Array, shape (n_time, n_cont_states)
        Smoothed state means, $$ m_{1:T|T} $$.
    smoother_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
        Smoothed state covariances, $$ P_{1:T|T} $$.
    smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states)
        Smoothed cross-covariances, $$ P_{t, t+1|T} $$.
    marginal_log_likelihood : jax.Array
        Total log likelihood of the observations (scalar array).

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
        carry: tuple[jax.Array, jax.Array],
        args: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
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


def sum_of_outer_products(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute the sum of outer products between corresponding vectors.

    Computes $$ S = \\sum_{t=1}^T x_t y_t^T $$.

    Parameters
    ----------
    x : jax.Array, shape (T, N)
        First sequence of vectors.
    y : jax.Array, shape (T, M)
        Second sequence of vectors.

    Returns
    -------
    jax.Array, shape (N, M)
        The sum of outer products.

    """
    return x.T @ y


def kalman_maximization_step(
    obs: jax.Array,
    smoother_mean: jax.Array,
    smoother_cov: jax.Array,
    smoother_cross_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Performs the Maximization (M) step of the EM algorithm for Kalman filters.

    Updates the model parameters based on the expected sufficient statistics
    derived from the E-step (Kalman smoother).

    Parameters
    ----------
    obs : jax.Array, shape (n_time, n_obs_dim)
        Observations, $$ y_{1:T} $$.
    smoother_mean : jax.Array, shape (n_time, n_cont_states)
        Smoothed means, $$ m_{1:T|T} $$.
    smoother_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
        Smoothed covariances, $$ P_{1:T|T} $$.
    smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states)
        Smoothed cross-covariances, $$ P_{t, t+1|T} $$.

    Returns
    -------
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        Updated transition matrix, $$ A $$.
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states)
        Updated measurement matrix, $$ H $$.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Updated process covariance, $$ \\Sigma $$.
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
        Updated measurement covariance, $$ R $$.
    init_mean : jax.Array, shape (n_cont_states,)
        Updated initial mean, $$ m_1 $$.
    init_cov : jax.Array, shape (n_cont_states, n_cont_states)
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
