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

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.stats.multivariate_normal

from state_space_practice.utils import (  # noqa: F401 — re-exported for backward compat
    psd_solve,
    stabilize_covariance,
    symmetrize,
)


def woodbury_kalman_gain(
    prior_cov: jax.Array,
    emission_matrix: jax.Array,
    emission_cov_diag: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Kalman gain using the Woodbury identity for diagonal R.

    When the observation noise covariance R is diagonal, the standard
    Kalman gain computation is O(D_obs^3). The Woodbury identity reduces
    this to O(D_state^3), which is much faster when D_obs >> D_state
    (e.g. many neurons, low-dimensional latent state).

    Parameters
    ----------
    prior_cov : jax.Array, shape (D_state, D_state)
        Prior (predicted) state covariance P.
    emission_matrix : jax.Array, shape (D_obs, D_state)
        Observation matrix H.
    emission_cov_diag : jax.Array, shape (D_obs,)
        Diagonal of observation noise covariance R.

    Returns
    -------
    K : jax.Array, shape (D_state, D_obs)
        Kalman gain.
    S : jax.Array, shape (D_obs, D_obs)
        Innovation covariance H P H' + R (for log-likelihood).
    S_inv : jax.Array, shape (D_obs, D_obs)
        Inverse of innovation covariance (via Woodbury).
    """
    D = prior_cov.shape[0]
    I_D = jnp.eye(D)
    # U = H @ chol(P), shape (D_obs, D_state)
    U = emission_matrix @ jnp.linalg.cholesky(prior_cov)
    X = U / emission_cov_diag[:, None]  # R^{-1} U, shape (D_obs, D_state)
    # Woodbury: S^{-1} = R^{-1} - X (I + U' X)^{-1} X'
    S_inv = jnp.diag(1.0 / emission_cov_diag) - X @ psd_solve(I_D + U.T @ X, X.T)
    K = prior_cov @ emission_matrix.T @ S_inv
    S = jnp.diag(emission_cov_diag) + emission_matrix @ prior_cov @ emission_matrix.T
    return K, S, S_inv


def standard_kalman_gain(
    prior_cov: jax.Array,
    emission_matrix: jax.Array,
    emission_cov: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute Kalman gain using the standard formula.

    Parameters
    ----------
    prior_cov : jax.Array, shape (D_state, D_state)
        Prior (predicted) state covariance P.
    emission_matrix : jax.Array, shape (D_obs, D_state)
        Observation matrix H.
    emission_cov : jax.Array, shape (D_obs, D_obs)
        Observation noise covariance R.

    Returns
    -------
    K : jax.Array, shape (D_state, D_obs)
        Kalman gain.
    S : jax.Array, shape (D_obs, D_obs)
        Innovation covariance H P H' + R.
    """
    S = symmetrize(emission_matrix @ prior_cov @ emission_matrix.T + emission_cov)
    K = psd_solve(S, emission_matrix @ prior_cov).T
    return K, S


def joseph_form_update(
    prior_cov: jax.Array,
    kalman_gain: jax.Array,
    emission_matrix: jax.Array,
    emission_cov: jax.Array,
) -> jax.Array:
    """Joseph form covariance update: always PSD by construction.

    Computes ``P_post = (I - K H) P (I - K H)' + K R K'``, which is
    a sum of PSD terms and therefore guaranteed PSD regardless of
    floating-point rounding.

    Parameters
    ----------
    prior_cov : jax.Array, shape (D, D)
        Prior (predicted) state covariance.
    kalman_gain : jax.Array, shape (D, D_obs)
        Kalman gain K.
    emission_matrix : jax.Array, shape (D_obs, D)
        Observation matrix H.
    emission_cov : jax.Array, shape (D_obs, D_obs)
        Observation noise covariance R.

    Returns
    -------
    jax.Array, shape (D, D)
        Posterior covariance, guaranteed PSD.
    """
    D = prior_cov.shape[0]
    I_KH = jnp.eye(D) - kalman_gain @ emission_matrix
    return symmetrize(I_KH @ prior_cov @ I_KH.T + kalman_gain @ emission_cov @ kalman_gain.T)



@jax.jit
def kalman_measurement_update(
    prior_mean: jax.Array,
    prior_cov: jax.Array,
    obs: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Kalman measurement update (no prediction step).

    Parameters
    ----------
    prior_mean : jax.Array, shape (n_cont_states,)
        Prior state mean.
    prior_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Prior state covariance.
    obs : jax.Array, shape (n_obs_dim,)
        Observation.
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states)
        Observation matrix H.
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance R.

    Returns
    -------
    posterior_mean : jax.Array, shape (n_cont_states,)
    posterior_cov : jax.Array, shape (n_cont_states, n_cont_states)
    marginal_log_likelihood : jax.Array (scalar)
    """
    obs_mean = measurement_matrix @ prior_mean
    obs_cov = symmetrize(
        measurement_matrix @ prior_cov @ measurement_matrix.T + measurement_cov
    )

    residual_error = obs - obs_mean
    kalman_gain = psd_solve(obs_cov, measurement_matrix @ prior_cov).T

    posterior_mean = prior_mean + kalman_gain @ residual_error
    posterior_cov = joseph_form_update(
        prior_cov, kalman_gain, measurement_matrix, measurement_cov
    )

    marginal_log_likelihood = jnp.asarray(
        jax.scipy.stats.multivariate_normal.logpdf(x=obs, mean=obs_mean, cov=obs_cov)
    )

    return posterior_mean, posterior_cov, marginal_log_likelihood


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
    return kalman_measurement_update(
        one_step_mean, one_step_cov, obs, measurement_matrix, measurement_cov
    )


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


class _SmootherElement(NamedTuple):
    """Associative scan element for parallel RTS smoother.

    Each element (E, g, L) encodes one backward smoother step so that
    the composition of elements via the associative operator yields the
    full smoothed posterior.

    Attributes
    ----------
    E : jax.Array, shape (..., D, D)
        Smoother gain (analogous to J_t in the sequential formulation).
    g : jax.Array, shape (..., D)
        Bias term: m_{t|t} - J_t @ m_{t+1|t}.
    L : jax.Array, shape (..., D, D)
        Residual covariance: P_{t|t} - J_t @ P_{t+1|t} @ J_t.T.

    Note: shape prefix ``...`` indicates these may be batched by
    ``jax.lax.associative_scan``.
    """

    E: jax.Array
    g: jax.Array
    L: jax.Array


def parallel_kalman_smoother(
    filtered_means: jax.Array,
    filtered_covariances: jax.Array,
    transition_matrix: jax.Array,
    process_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """RTS smoother via parallel associative scan.

    Algebraically equivalent to the sequential ``kalman_smoother`` backward
    pass but runs in O(log T) parallel depth on GPU/TPU via
    ``jax.lax.associative_scan``.

    Parameters
    ----------
    filtered_means : jax.Array, shape (T, D)
        Filtered state means from the forward Kalman filter.
    filtered_covariances : jax.Array, shape (T, D, D)
        Filtered state covariances from the forward Kalman filter.
    transition_matrix : jax.Array, shape (D, D) or (T-1, D, D)
        State transition matrix. If 2-D, the same matrix is used at every
        time step. If 3-D, ``transition_matrix[t]`` is used for the
        transition from time ``t`` to ``t+1``.
    process_cov : jax.Array, shape (D, D) or (T-1, D, D)
        Process noise covariance. Broadcasting rules follow
        ``transition_matrix``.

    Returns
    -------
    smoothed_means : jax.Array, shape (T, D)
        Smoothed state means.
    smoothed_covariances : jax.Array, shape (T, D, D)
        Smoothed state covariances.
    cross_covariances : jax.Array, shape (T-1, D, D)
        Lag-one cross-covariances P_{t, t+1|T}.

    References
    ----------
    Särkkä, S. & García-Fernández, Á.F. (2021). Temporal parallelization
    of Bayesian smoothers. IEEE Trans. Automatic Control 66(1), 299-306.
    """
    T, D = filtered_means.shape

    # Broadcast time-invariant parameters to (T-1, D, D)
    if transition_matrix.ndim == 2:
        A = jnp.broadcast_to(transition_matrix, (T - 1, D, D))
    else:
        A = transition_matrix
    if process_cov.ndim == 2:
        Q = jnp.broadcast_to(process_cov, (T - 1, D, D))
    else:
        Q = process_cov

    # Build per-timestep smoother elements for t = 0, ..., T-2
    def _build_element(filt_mean, filt_cov, A_t, Q_t):
        pred_cov = symmetrize(A_t @ filt_cov @ A_t.T + Q_t)
        pred_mean = A_t @ filt_mean
        J = psd_solve(pred_cov, A_t @ filt_cov).T  # smoother gain
        g = filt_mean - J @ pred_mean
        L = symmetrize(filt_cov - J @ pred_cov @ J.T)
        return _SmootherElement(E=J, g=g, L=L)

    elements = jax.vmap(_build_element)(
        filtered_means[:-1], filtered_covariances[:-1], A, Q
    )

    # Terminal element for t = T-1: no dependence on future state
    terminal = _SmootherElement(
        E=jnp.zeros((D, D)),
        g=filtered_means[-1],
        L=filtered_covariances[-1],
    )

    # Concatenate elements with terminal at end
    all_elements = _SmootherElement(
        E=jnp.concatenate([elements.E, terminal.E[None]], axis=0),
        g=jnp.concatenate([elements.g, terminal.g[None]], axis=0),
        L=jnp.concatenate([elements.L, terminal.L[None]], axis=0),
    )

    # Associative operator vmapped over the batch dimension that
    # associative_scan introduces when combining sub-sequences.
    @jax.vmap
    def _operator(elem1, elem2):
        E1, g1, L1 = elem1
        E2, g2, L2 = elem2
        E = E2 @ E1
        g = E2 @ g1 + g2
        L = symmetrize(E2 @ L1 @ E2.T + L2)
        return _SmootherElement(E=E, g=g, L=L)

    scanned = jax.lax.associative_scan(
        _operator, all_elements, reverse=True
    )

    smoothed_means = scanned.g
    smoothed_covariances = scanned.L

    # Cross-covariances: P_{t,t+1|T} = J_t @ P_{t+1|T}
    smoother_gains = elements.E  # (T-1, D, D)
    cross_covariances = jnp.einsum("tij,tjk->tik", smoother_gains, smoothed_covariances[1:])

    return smoothed_means, smoothed_covariances, cross_covariances


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


@jax.jit
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
    measurement_cov = stabilize_covariance(
        (alpha - measurement_matrix @ delta.T) / n_time,
        min_eigenvalue=1e-8,
    )

    # Transition matrix
    transition_matrix = psd_solve(gamma1, beta.T).T

    # Process covariance
    process_cov = stabilize_covariance(
        (gamma2 - transition_matrix @ beta.T) / (n_time - 1),
        min_eigenvalue=1e-8,
    )

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
