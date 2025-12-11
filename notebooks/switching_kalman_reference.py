"""Reference implementation of the switching Kalman filter.

This module provides simple, unoptimized but clearly-correct implementations
of the switching Kalman filter, smoother, and M-step. It serves as ground truth
for testing the optimized vmapped implementations.

The implementation uses explicit loops over discrete states (no vmaps) with
clear variable names matching mathematical notation.

References
----------
1. Murphy, K.P. (1998). Switching kalman filters.
2. Shumway, R.H., and Stoffer, D.S. (1991). Dynamic Linear Models With Switching.
"""

import jax
import jax.numpy as jnp
import jax.scipy.stats.multivariate_normal

from state_space_practice.kalman import psd_solve, symmetrize


def _kalman_filter_predict(
    mean_prev: jax.Array,
    cov_prev: jax.Array,
    transition_matrix: jax.Array,
    process_cov: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Kalman filter prediction step.

    Parameters
    ----------
    mean_prev : jax.Array, shape (n_cont_states,)
        Previous state mean, m_{t-1|t-1}
    cov_prev : jax.Array, shape (n_cont_states, n_cont_states)
        Previous state covariance, P_{t-1|t-1}
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        State transition matrix, A
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Process noise covariance, Q

    Returns
    -------
    pred_mean : jax.Array, shape (n_cont_states,)
        Predicted state mean, m_{t|t-1}
    pred_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Predicted state covariance, P_{t|t-1}
    """
    pred_mean = transition_matrix @ mean_prev
    pred_cov = symmetrize(transition_matrix @ cov_prev @ transition_matrix.T + process_cov)
    return pred_mean, pred_cov


def _kalman_filter_update(
    pred_mean: jax.Array,
    pred_cov: jax.Array,
    obs: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, float]:
    """Kalman filter update step.

    Parameters
    ----------
    pred_mean : jax.Array, shape (n_cont_states,)
        Predicted state mean, m_{t|t-1}
    pred_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Predicted state covariance, P_{t|t-1}
    obs : jax.Array, shape (n_obs_dim,)
        Observation, y_t
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states)
        Measurement matrix, H
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
        Measurement noise covariance, R

    Returns
    -------
    post_mean : jax.Array, shape (n_cont_states,)
        Posterior state mean, m_{t|t}
    post_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Posterior state covariance, P_{t|t}
    log_likelihood : float
        Log-likelihood of the observation, log p(y_t | y_{1:t-1})
    """
    # Innovation
    obs_mean = measurement_matrix @ pred_mean
    obs_cov = symmetrize(measurement_matrix @ pred_cov @ measurement_matrix.T + measurement_cov)

    residual = obs - obs_mean

    # Kalman gain
    kalman_gain = psd_solve(obs_cov, measurement_matrix @ pred_cov).T

    # Update
    post_mean = pred_mean + kalman_gain @ residual

    # Joseph form for numerical stability
    n_cont = pred_mean.shape[0]
    I_KH = jnp.eye(n_cont) - kalman_gain @ measurement_matrix
    post_cov = symmetrize(
        I_KH @ pred_cov @ I_KH.T + kalman_gain @ measurement_cov @ kalman_gain.T
    )

    # Log-likelihood
    log_likelihood = jax.scipy.stats.multivariate_normal.logpdf(
        x=obs, mean=obs_mean, cov=obs_cov
    )

    return post_mean, post_cov, log_likelihood


def _divide_safe(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """Divides two arrays, while setting the result to 0.0 if the denominator is 0.0."""
    return jnp.where(denominator == 0.0, 0.0, numerator / denominator)


def _scale_likelihood(log_likelihood: jax.Array) -> tuple[jax.Array, float]:
    """Scale the log likelihood to avoid numerical underflow."""
    ll_max = log_likelihood.max()
    ll_max = jnp.where(jnp.isfinite(ll_max), ll_max, 0.0)
    return jnp.exp(log_likelihood - ll_max), ll_max


def switching_kalman_filter_reference(
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    obs: jax.Array,
    discrete_transition_matrix: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]:
    """Reference implementation of the switching Kalman filter using explicit loops.

    This implements the GPB1/IMM algorithm:
    For each time step t:
        1. For each pair (i, j) of (prev_state, curr_state):
           - Predict: x_{t|t-1}^{ij} = A_j @ x_{t-1|t-1}^i
           - Update with observation to get x_{t|t}^{ij}, P_{t|t}^{ij}, likelihood^{ij}
        2. Update discrete state probabilities using likelihoods
        3. Collapse mixture: merge over previous state i to get x_{t|t}^j, P_{t|t}^j

    Parameters
    ----------
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Initial means E[x_1 | S_1=j]
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Initial covariances Cov[x_1 | S_1=j]
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Initial discrete state probabilities P(S_1=j)
    obs : jax.Array, shape (n_time, n_obs_dim)
        Observations y_{1:T}
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Z(i, j) = P(S_t=j | S_{t-1}=i)
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Transition matrix A for each discrete state
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Process noise Q for each discrete state
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Measurement matrix H for each discrete state
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Measurement noise R for each discrete state

    Returns
    -------
    state_cond_filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        Filtered means E[x_t | y_{1:t}, S_t=j]
    state_cond_filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        Filtered covariances Cov[x_t | y_{1:t}, S_t=j]
    filter_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        Filtered discrete state probabilities P(S_t=j | y_{1:t})
    last_pair_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
        Last time step pair-conditional means E[x_T | y_{1:T}, S_{T-1}=i, S_T=j]
    marginal_log_likelihood : float
        Marginal log-likelihood of the observations
    """
    n_time, n_obs_dim = obs.shape
    n_cont_states = init_state_cond_mean.shape[0]
    n_discrete_states = init_discrete_state_prob.shape[0]

    # Initialize outputs
    state_cond_filter_mean = jnp.zeros((n_time, n_cont_states, n_discrete_states))
    state_cond_filter_cov = jnp.zeros(
        (n_time, n_cont_states, n_cont_states, n_discrete_states)
    )
    filter_discrete_state_prob = jnp.zeros((n_time, n_discrete_states))

    # Current state (to be updated)
    prev_mean = init_state_cond_mean  # shape (n_cont, n_disc)
    prev_cov = init_state_cond_cov  # shape (n_cont, n_cont, n_disc)
    prev_discrete_prob = init_discrete_state_prob  # shape (n_disc,)

    marginal_log_likelihood = 0.0
    last_pair_mean = None

    for t in range(n_time):
        obs_t = obs[t]

        # Step 1: For each pair (i, j), compute predict and update
        # pair_cond_mean[i, j] = E[x_t | y_{1:t}, S_{t-1}=i, S_t=j]
        pair_cond_mean = jnp.zeros((n_cont_states, n_discrete_states, n_discrete_states))
        pair_cond_cov = jnp.zeros(
            (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        )
        pair_log_likelihood = jnp.zeros((n_discrete_states, n_discrete_states))

        for i in range(n_discrete_states):
            for j in range(n_discrete_states):
                # Predict using transition from state j
                pred_mean, pred_cov = _kalman_filter_predict(
                    prev_mean[:, i],
                    prev_cov[:, :, i],
                    continuous_transition_matrix[:, :, j],
                    process_cov[:, :, j],
                )

                # Update using measurement model from state j
                post_mean, post_cov, log_lik = _kalman_filter_update(
                    pred_mean,
                    pred_cov,
                    obs_t,
                    measurement_matrix[:, :, j],
                    measurement_cov[:, :, j],
                )

                pair_cond_mean = pair_cond_mean.at[:, i, j].set(post_mean)
                pair_cond_cov = pair_cond_cov.at[:, :, i, j].set(post_cov)
                pair_log_likelihood = pair_log_likelihood.at[i, j].set(log_lik)

        # Step 2: Update discrete state probabilities
        # Scale likelihoods for numerical stability
        scaled_likelihood, ll_max = _scale_likelihood(pair_log_likelihood)

        # Joint probability P(S_{t-1}=i, S_t=j | y_{1:t})
        joint_prob = (
            scaled_likelihood  # L(i, j)
            * discrete_transition_matrix  # Z(i, j)
            * prev_discrete_prob[:, None]  # M_{t-1|t-1}(i)
        )
        total = jnp.sum(joint_prob)
        joint_prob = _divide_safe(joint_prob, total)

        # Marginal P(S_t=j | y_{1:t})
        curr_discrete_prob = jnp.sum(joint_prob, axis=0)

        # Backward conditional P(S_{t-1}=i | S_t=j, y_{1:t})
        backward_cond_prob = _divide_safe(joint_prob, curr_discrete_prob[None, :])

        # Update marginal log-likelihood
        marginal_log_likelihood += ll_max + jnp.log(total)

        # Step 3: Collapse mixture over previous state i for each current state j
        curr_mean = jnp.zeros((n_cont_states, n_discrete_states))
        curr_cov = jnp.zeros((n_cont_states, n_cont_states, n_discrete_states))

        for j in range(n_discrete_states):
            # Compute marginal mean E[x_t | y_{1:t}, S_t=j]
            weights = backward_cond_prob[:, j]  # shape (n_disc,)
            cond_means = pair_cond_mean[:, :, j]  # shape (n_cont, n_disc)
            cond_covs = pair_cond_cov[:, :, :, j]  # shape (n_cont, n_cont, n_disc)

            # Marginal mean
            marg_mean = cond_means @ weights  # shape (n_cont,)

            # Marginal covariance (law of total variance)
            diff = cond_means - marg_mean[:, None]
            marg_cov = cond_covs @ weights + (diff * weights) @ diff.T

            curr_mean = curr_mean.at[:, j].set(marg_mean)
            curr_cov = curr_cov.at[:, :, j].set(marg_cov)

        # Store results
        state_cond_filter_mean = state_cond_filter_mean.at[t].set(curr_mean)
        state_cond_filter_cov = state_cond_filter_cov.at[t].set(curr_cov)
        filter_discrete_state_prob = filter_discrete_state_prob.at[t].set(
            curr_discrete_prob
        )

        # Save for next iteration
        prev_mean = curr_mean
        prev_cov = curr_cov
        prev_discrete_prob = curr_discrete_prob
        last_pair_mean = pair_cond_mean

    return (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_state_prob,
        last_pair_mean,
        marginal_log_likelihood,
    )


def _kalman_smoother_update(
    next_smoother_mean: jax.Array,
    next_smoother_cov: jax.Array,
    filter_mean: jax.Array,
    filter_cov: jax.Array,
    process_cov: jax.Array,
    transition_matrix: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Kalman smoother backward update step.

    Parameters
    ----------
    next_smoother_mean : jax.Array, shape (n_cont_states,)
        Smoothed mean from next time step, m_{t+1|T}
    next_smoother_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Smoothed covariance from next time step, P_{t+1|T}
    filter_mean : jax.Array, shape (n_cont_states,)
        Filtered mean from current time step, m_{t|t}
    filter_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Filtered covariance from current time step, P_{t|t}
    process_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Process noise covariance, Q
    transition_matrix : jax.Array, shape (n_cont_states, n_cont_states)
        State transition matrix, A

    Returns
    -------
    smoother_mean : jax.Array, shape (n_cont_states,)
        Smoothed mean, m_{t|T}
    smoother_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Smoothed covariance, P_{t|T}
    smoother_cross_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Smoothed cross-covariance, Cov[x_{t+1}, x_t | y_{1:T}]
    """
    # Predicted mean and covariance m_{t+1|t}, P_{t+1|t}
    pred_mean = transition_matrix @ filter_mean
    pred_cov = symmetrize(transition_matrix @ filter_cov @ transition_matrix.T + process_cov)

    # Smoother gain J_t
    smoother_gain = psd_solve(pred_cov, transition_matrix @ filter_cov).T

    # Smoothed mean m_{t|T}
    smoother_mean = filter_mean + smoother_gain @ (next_smoother_mean - pred_mean)

    # Smoothed covariance P_{t|T}
    smoother_cov = symmetrize(
        filter_cov + smoother_gain @ (next_smoother_cov - pred_cov) @ smoother_gain.T
    )

    # Lag-one cross covariance Cov[x_{t+1}, x_t | y_{1:T}]
    smoother_cross_cov = smoother_gain @ next_smoother_cov

    return smoother_mean, smoother_cov, smoother_cross_cov


def switching_kalman_smoother_reference(
    filter_mean: jax.Array,
    filter_cov: jax.Array,
    filter_discrete_state_prob: jax.Array,
    last_filter_conditional_cont_mean: jax.Array,
    process_cov: jax.Array,
    continuous_transition_matrix: jax.Array,
    discrete_state_transition_matrix: jax.Array,
) -> tuple[
    jax.Array,  # overall_smoother_mean
    jax.Array,  # overall_smoother_cov
    jax.Array,  # smoother_discrete_state_prob
    jax.Array,  # smoother_joint_discrete_state_prob
    jax.Array,  # overall_smoother_cross_cov
    jax.Array,  # state_cond_smoother_means
    jax.Array,  # state_cond_smoother_covs
    jax.Array,  # pair_cond_smoother_cross_covs
]:
    """Reference implementation of the switching Kalman smoother.

    This implements the backward pass of the GPB1/IMM algorithm.

    Parameters
    ----------
    filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        Filtered means from forward pass
    filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        Filtered covariances from forward pass
    filter_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        Filtered discrete state probabilities from forward pass
    last_filter_conditional_cont_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
        Last time step pair-conditional means
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Process noise for each discrete state
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Transition matrix for each discrete state
    discrete_state_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Discrete state transition matrix Z(i, j) = P(S_t=j | S_{t-1}=i)

    Returns
    -------
    overall_smoother_mean : jax.Array, shape (n_time, n_cont_states)
        Marginal smoothed means E[x_t | y_{1:T}]
    overall_smoother_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
        Marginal smoothed covariances Cov[x_t | y_{1:T}]
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        Smoothed discrete state probabilities P(S_t=j | y_{1:T})
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time-1, n_discrete_states, n_discrete_states)
        Smoothed joint probabilities P(S_t=i, S_{t+1}=j | y_{1:T})
    overall_smoother_cross_cov : jax.Array, shape (n_time-1, n_cont_states, n_cont_states)
        Marginal smoothed cross-covariances Cov[x_{t+1}, x_t | y_{1:T}]
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        State-conditional smoothed means E[x_t | y_{1:T}, S_t=j]
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        State-conditional smoothed covariances Cov[x_t | y_{1:T}, S_t=j]
    pair_cond_smoother_cross_covs : jax.Array, shape (n_time-1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional cross-covariances Cov[x_{t+1}, x_t | y_{1:T}, S_t=i, S_{t+1}=j]
    """
    n_time = filter_mean.shape[0]
    n_cont_states = filter_mean.shape[1]
    n_discrete_states = filter_discrete_state_prob.shape[1]

    # Initialize outputs
    overall_smoother_mean = jnp.zeros((n_time, n_cont_states))
    overall_smoother_cov = jnp.zeros((n_time, n_cont_states, n_cont_states))
    smoother_discrete_state_prob = jnp.zeros((n_time, n_discrete_states))
    smoother_joint_discrete_state_prob = jnp.zeros(
        (n_time - 1, n_discrete_states, n_discrete_states)
    )
    overall_smoother_cross_cov = jnp.zeros((n_time - 1, n_cont_states, n_cont_states))
    state_cond_smoother_means = jnp.zeros((n_time, n_cont_states, n_discrete_states))
    state_cond_smoother_covs = jnp.zeros(
        (n_time, n_cont_states, n_cont_states, n_discrete_states)
    )
    pair_cond_smoother_cross_covs = jnp.zeros(
        (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    )

    # Initialize at last time step (smoother = filter at T)
    next_smoother_mean = filter_mean[-1]  # shape (n_cont, n_disc)
    next_smoother_cov = filter_cov[-1]  # shape (n_cont, n_cont, n_disc)
    next_smoother_discrete_prob = filter_discrete_state_prob[-1]
    next_pair_cond_mean = last_filter_conditional_cont_mean  # shape (n_cont, n_disc, n_disc)

    # Collapse last time step to overall mean/cov
    last_overall_mean = next_smoother_mean @ next_smoother_discrete_prob
    diff = next_smoother_mean - last_overall_mean[:, None]
    last_overall_cov = (
        next_smoother_cov @ next_smoother_discrete_prob
        + (diff * next_smoother_discrete_prob) @ diff.T
    )

    # Store last time step
    overall_smoother_mean = overall_smoother_mean.at[-1].set(last_overall_mean)
    overall_smoother_cov = overall_smoother_cov.at[-1].set(last_overall_cov)
    smoother_discrete_state_prob = smoother_discrete_state_prob.at[-1].set(
        next_smoother_discrete_prob
    )
    state_cond_smoother_means = state_cond_smoother_means.at[-1].set(next_smoother_mean)
    state_cond_smoother_covs = state_cond_smoother_covs.at[-1].set(next_smoother_cov)

    # Backward pass
    for t in range(n_time - 2, -1, -1):
        filt_mean_t = filter_mean[t]  # shape (n_cont, n_disc)
        filt_cov_t = filter_cov[t]  # shape (n_cont, n_cont, n_disc)
        filt_discrete_prob_t = filter_discrete_state_prob[t]

        # Step 1: Smooth for each pair (j, k) of (curr_state, next_state)
        pair_cond_smoother_mean = jnp.zeros(
            (n_cont_states, n_discrete_states, n_discrete_states)
        )
        pair_cond_smoother_cov = jnp.zeros(
            (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        )
        pair_cond_cross_cov = jnp.zeros(
            (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        )

        for j in range(n_discrete_states):
            for k in range(n_discrete_states):
                sm_mean, sm_cov, sm_cross = _kalman_smoother_update(
                    next_smoother_mean[:, k],
                    next_smoother_cov[:, :, k],
                    filt_mean_t[:, j],
                    filt_cov_t[:, :, j],
                    process_cov[:, :, k],
                    continuous_transition_matrix[:, :, k],
                )
                pair_cond_smoother_mean = pair_cond_smoother_mean.at[:, j, k].set(sm_mean)
                pair_cond_smoother_cov = pair_cond_smoother_cov.at[:, :, j, k].set(sm_cov)
                # sm_cross is Cov[x_t, x_{t+1}] matching the optimized implementation
                pair_cond_cross_cov = pair_cond_cross_cov.at[:, :, j, k].set(sm_cross)

        # Step 2: Compute discrete smoother probabilities
        # P(S_t=j | S_{t+1}=k, y_{1:t})
        backward_cond = filt_discrete_prob_t[:, None] * discrete_state_transition_matrix
        backward_cond = _divide_safe(backward_cond, jnp.sum(backward_cond, axis=0))

        # P(S_t=j, S_{t+1}=k | y_{1:T})
        joint_discrete_prob = backward_cond * next_smoother_discrete_prob

        # P(S_t=j | y_{1:T})
        smoother_discrete_prob_t = jnp.sum(joint_discrete_prob, axis=1)

        # P(S_{t+1}=k | S_t=j, y_{1:T})
        forward_cond = _divide_safe(joint_discrete_prob, smoother_discrete_prob_t[:, None])

        # Step 3: Collapse over next state k to get state-conditional smoothed estimates
        state_cond_mean_t = jnp.zeros((n_cont_states, n_discrete_states))
        state_cond_cov_t = jnp.zeros((n_cont_states, n_cont_states, n_discrete_states))

        for j in range(n_discrete_states):
            weights = forward_cond[j, :]  # P(S_{t+1}=k | S_t=j, y_{1:T})
            cond_means = pair_cond_smoother_mean[:, j, :]  # shape (n_cont, n_disc)
            cond_covs = pair_cond_smoother_cov[:, :, j, :]  # shape (n_cont, n_cont, n_disc)

            marg_mean = cond_means @ weights
            diff = cond_means - marg_mean[:, None]
            marg_cov = cond_covs @ weights + (diff * weights) @ diff.T

            state_cond_mean_t = state_cond_mean_t.at[:, j].set(marg_mean)
            state_cond_cov_t = state_cond_cov_t.at[:, :, j].set(marg_cov)

        # Step 4: Collapse to single overall mean and covariance
        overall_mean_t = state_cond_mean_t @ smoother_discrete_prob_t
        diff = state_cond_mean_t - overall_mean_t[:, None]
        overall_cov_t = (
            state_cond_cov_t @ smoother_discrete_prob_t
            + (diff * smoother_discrete_prob_t) @ diff.T
        )

        # Step 5: Compute overall cross-covariance
        # First collapse over S_t for each S_{t+1}
        # U^{j|k} = P(S_t=j | S_{t+1}=k, y_{1:T})
        smoother_backward_cond = _divide_safe(joint_discrete_prob, next_smoother_discrete_prob)

        state_cond_mean_tplus1 = jnp.zeros((n_cont_states, n_discrete_states))
        smoother_mean_t_cond_Stplus1 = jnp.zeros((n_cont_states, n_discrete_states))
        state_cond_cross_cov = jnp.zeros((n_cont_states, n_cont_states, n_discrete_states))

        for k in range(n_discrete_states):
            weights = smoother_backward_cond[:, k]  # P(S_t=j | S_{t+1}=k, y_{1:T})
            cond_means_t = pair_cond_smoother_mean[:, :, k]  # E[x_t | ..., S_t=j, S_{t+1}=k]
            cond_means_tplus1 = next_pair_cond_mean[:, :, k]  # E[x_{t+1} | ..., S_t=j, S_{t+1}=k]
            # cond_cross_covs[:,:,j] = Cov[x_t, x_{t+1} | S_t=j, S_{t+1}=k] where [a,b] = Cov[x_t[a], x_{t+1}[b]]
            cond_cross_covs = pair_cond_cross_cov[:, :, :, k]

            # Collapse over S_t to get Cov[x_{t+1}, x_t | S_{t+1}=k]
            # We want result[a,b] = Cov[x_{t+1}[a], x_t[b]]
            # cond_cross_covs[c,d,j] = Cov[x_t[c], x_{t+1}[d]], so transpose each slice
            marg_mean_t = cond_means_t @ weights
            marg_mean_tplus1 = cond_means_tplus1 @ weights
            diff_t = cond_means_t - marg_mean_t[:, None]
            diff_tplus1 = cond_means_tplus1 - marg_mean_tplus1[:, None]
            # Sum cond_cross_covs[:,:,j].T * weights[j] over j = sum Cov[x_{t+1}, x_t | j] * w[j]
            # einsum "abj,j->ab" gives result[a,b] = sum_j cond_cross_covs[a,b,j] * w[j]
            # But we need result[a,b] = sum_j cond_cross_covs[b,a,j] * w[j] for the transpose
            # This is einsum "baj,j->ab"
            marg_cross_cov = jnp.einsum("baj,j->ab", cond_cross_covs, weights) + (diff_tplus1 * weights) @ diff_t.T

            state_cond_mean_tplus1 = state_cond_mean_tplus1.at[:, k].set(marg_mean_tplus1)
            smoother_mean_t_cond_Stplus1 = smoother_mean_t_cond_Stplus1.at[:, k].set(marg_mean_t)
            state_cond_cross_cov = state_cond_cross_cov.at[:, :, k].set(marg_cross_cov)

        # Final collapse over S_{t+1}
        overall_mean_tplus1 = state_cond_mean_tplus1 @ next_smoother_discrete_prob
        overall_mean_t_from_cross = smoother_mean_t_cond_Stplus1 @ next_smoother_discrete_prob
        diff_tplus1 = state_cond_mean_tplus1 - overall_mean_tplus1[:, None]
        diff_t = smoother_mean_t_cond_Stplus1 - overall_mean_t_from_cross[:, None]
        # Sum state_cond_cross_cov[:,:,k] * prob[k] over k
        overall_cross_cov = (
            jnp.einsum("abk,k->ab", state_cond_cross_cov, next_smoother_discrete_prob)
            + (diff_tplus1 * next_smoother_discrete_prob) @ diff_t.T
        )

        # Store results
        overall_smoother_mean = overall_smoother_mean.at[t].set(overall_mean_t)
        overall_smoother_cov = overall_smoother_cov.at[t].set(overall_cov_t)
        smoother_discrete_state_prob = smoother_discrete_state_prob.at[t].set(
            smoother_discrete_prob_t
        )
        smoother_joint_discrete_state_prob = smoother_joint_discrete_state_prob.at[t].set(
            joint_discrete_prob
        )
        overall_smoother_cross_cov = overall_smoother_cross_cov.at[t].set(overall_cross_cov)
        state_cond_smoother_means = state_cond_smoother_means.at[t].set(state_cond_mean_t)
        state_cond_smoother_covs = state_cond_smoother_covs.at[t].set(state_cond_cov_t)
        pair_cond_smoother_cross_covs = pair_cond_smoother_cross_covs.at[t].set(
            pair_cond_cross_cov
        )

        # Update for next iteration
        next_smoother_mean = state_cond_mean_t
        next_smoother_cov = state_cond_cov_t
        next_smoother_discrete_prob = smoother_discrete_prob_t
        next_pair_cond_mean = pair_cond_smoother_mean

    return (
        overall_smoother_mean,
        overall_smoother_cov,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        overall_smoother_cross_cov,
        state_cond_smoother_means,
        state_cond_smoother_covs,
        pair_cond_smoother_cross_covs,
    )


def switching_kalman_maximization_step_reference(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
) -> tuple[
    jax.Array,  # continuous_transition_matrix
    jax.Array,  # measurement_matrix
    jax.Array,  # process_cov
    jax.Array,  # measurement_cov
    jax.Array,  # init_mean
    jax.Array,  # init_cov
    jax.Array,  # discrete_transition_matrix
    jax.Array,  # init_discrete_state_prob
]:
    """Reference implementation of the M-step for the switching Kalman EM.

    Parameters
    ----------
    obs : jax.Array, shape (n_time, n_obs_dim)
        Observations
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        State-conditional smoothed means
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        State-conditional smoothed covariances
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        Smoothed discrete state probabilities
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time-1, n_discrete_states, n_discrete_states)
        Smoothed joint discrete state probabilities
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time-1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional cross-covariances

    Returns
    -------
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
    init_mean : jax.Array, shape (n_cont_states, n_discrete_states)
    init_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
    """
    n_time = obs.shape[0]
    n_obs_dim = obs.shape[1]
    n_cont_states = state_cond_smoother_means.shape[1]
    n_discrete_states = smoother_discrete_state_prob.shape[1]

    # Sufficient statistic for each discrete state
    n_time_per_state = smoother_discrete_state_prob.sum(axis=0)
    n_time_1_per_state = smoother_discrete_state_prob[1:].sum(axis=0)

    # Initialize outputs
    continuous_transition_matrix = jnp.zeros(
        (n_cont_states, n_cont_states, n_discrete_states)
    )
    measurement_matrix = jnp.zeros((n_obs_dim, n_cont_states, n_discrete_states))
    process_cov = jnp.zeros((n_cont_states, n_cont_states, n_discrete_states))
    measurement_cov = jnp.zeros((n_obs_dim, n_obs_dim, n_discrete_states))

    for j in range(n_discrete_states):
        weights = smoother_discrete_state_prob[:, j]  # shape (n_time,)
        means = state_cond_smoother_means[:, :, j]  # shape (n_time, n_cont)
        covs = state_cond_smoother_covs[:, :, :, j]  # shape (n_time, n_cont, n_cont)

        # gamma: sum of E[x_t x_t^T | S_t=j] weighted by P(S_t=j)
        gamma = jnp.zeros((n_cont_states, n_cont_states))
        for t in range(n_time):
            outer = jnp.outer(means[t], means[t]) + covs[t]
            gamma = gamma + weights[t] * outer

        # delta: sum of y_t x_t^T weighted by P(S_t=j)
        delta = jnp.zeros((n_obs_dim, n_cont_states))
        for t in range(n_time):
            delta = delta + weights[t] * jnp.outer(obs[t], means[t])

        # alpha: sum of y_t y_t^T weighted by P(S_t=j)
        alpha = jnp.zeros((n_obs_dim, n_obs_dim))
        for t in range(n_time):
            alpha = alpha + weights[t] * jnp.outer(obs[t], obs[t])

        # Measurement matrix: H = delta @ gamma^{-1}
        H_j = psd_solve(gamma, delta.T).T
        measurement_matrix = measurement_matrix.at[:, :, j].set(H_j)

        # Measurement covariance: R = (alpha - H @ delta^T) / n
        R_j = symmetrize((alpha - H_j @ delta.T) / n_time_per_state[j])
        measurement_cov = measurement_cov.at[:, :, j].set(R_j)

        # For transition matrix, we need gamma1 (without last time step) weighted by joint prob
        # gamma1[a,b,j] = sum_{t=0}^{T-2} sum_i P(S_t=i, S_{t+1}=j) * E[x_t x_t^T | S_t=i]
        gamma1 = jnp.zeros((n_cont_states, n_cont_states))
        for t in range(n_time - 1):
            for i in range(n_discrete_states):
                weight = smoother_joint_discrete_state_prob[t, i, j]
                outer = (
                    jnp.outer(state_cond_smoother_means[t, :, i], state_cond_smoother_means[t, :, i])
                    + state_cond_smoother_covs[t, :, :, i]
                )
                gamma1 = gamma1 + weight * outer

        # gamma2 (without first time step)
        first_gamma = weights[0] * (jnp.outer(means[0], means[0]) + covs[0])
        gamma2 = gamma - first_gamma

        # beta: sum of E[x_{t+1} x_t^T | S_t=i, S_{t+1}=j] weighted by P(S_t=i, S_{t+1}=j)
        # cross_cov[a,b] = Cov[x_t[a], x_{t+1}[b]], so we need cross_cov.T to get Cov[x_{t+1}, x_t]
        beta = jnp.zeros((n_cont_states, n_cont_states))
        for t in range(n_time - 1):
            for i in range(n_discrete_states):
                weight = smoother_joint_discrete_state_prob[t, i, j]
                cross_cov = pair_cond_smoother_cross_cov[t, :, :, i, j]
                outer = jnp.outer(
                    state_cond_smoother_means[t + 1, :, j],
                    state_cond_smoother_means[t, :, i],
                )
                # E[x_{t+1} x_t^T] = Cov[x_{t+1}, x_t] + m_{t+1} m_t^T = cross_cov.T + outer
                beta = beta + weight * (cross_cov.T + outer)

        # Transition matrix: A = beta @ gamma1^{-1}
        A_j = psd_solve(gamma1, beta.T).T
        continuous_transition_matrix = continuous_transition_matrix.at[:, :, j].set(A_j)

        # Process covariance: Q = (gamma2 - A @ beta^T) / (n-1)
        Q_j = symmetrize((gamma2 - A_j @ beta.T) / n_time_1_per_state[j])
        process_cov = process_cov.at[:, :, j].set(Q_j)

    # Initial mean and covariance
    init_mean = state_cond_smoother_means[0]
    init_cov = state_cond_smoother_covs[0]

    # Discrete transition matrix: Z(i,j) = sum_t P(S_t=i, S_{t+1}=j) / sum_t P(S_t=i)
    discrete_transition = _divide_safe(
        smoother_joint_discrete_state_prob.sum(axis=0),
        smoother_discrete_state_prob[:-1].sum(axis=0)[:, None],
    )
    # Normalize rows to sum to 1
    discrete_transition = _divide_safe(
        discrete_transition, jnp.sum(discrete_transition, axis=1, keepdims=True)
    )

    # Initial discrete state probabilities
    init_discrete_state_prob = smoother_discrete_state_prob[0]
    init_discrete_state_prob = _divide_safe(
        init_discrete_state_prob, jnp.sum(init_discrete_state_prob)
    )

    return (
        continuous_transition_matrix,
        measurement_matrix,
        process_cov,
        measurement_cov,
        init_mean,
        init_cov,
        discrete_transition,
        init_discrete_state_prob,
    )
