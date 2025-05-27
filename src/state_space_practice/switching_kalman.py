"""Switching Kalman filter and smoother and EM algorithm.

References
----------
1. Shumway, R.H., and Stoffer, D.S. (1991). Dynamic Linear Models With Switching. 8.
2. Murphy, K.P. (1998). Switching kalman filters.
3. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2022). Switching Functional Network Models of Oscillatory Brain Dynamics. In 2022 56th Asilomar Conference on Signals, Systems, and Computers (IEEE), pp. 607–612. https://doi.org/10.1109/IEEECONF56349.2022.10052077.
4. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2024). Switching Models of Oscillatory Networks Greatly Improve Inference of Dynamic Functional Connectivity. Preprint at arXiv.
5. https://github.com/Stephen-Lab-BU/Switching_Oscillator_Networks
"""

from typing import Optional

import jax
import jax.numpy as jnp

from state_space_practice.kalman import (
    _kalman_filter_update,
    _kalman_smoother_update,
    psd_solve,
    symmetrize,
)

_kalman_filter_update_per_discrete_state_pair = jax.vmap(
    jax.vmap(
        _kalman_filter_update,
        in_axes=(-1, -1, None, None, None, None, None),
        out_axes=-1,
    ),
    in_axes=(None, None, None, -1, -1, -1, -1),
    out_axes=-1,
)  # shape (n_discrete_states, n_discrete_states, n_obs_dim)


@jax.jit
def collapse_gaussian_mixture(
    conditional_means_x: jnp.ndarray,
    conditional_cov: jnp.ndarray,
    mixing_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Collapse a mixture of Gaussians.

    Parameters
    ----------
    conditional_means_x : jnp.ndarray, shape (n_dims, n_discrete_states)
        E[X | S = j]
    conditional_cov : jnp.ndarray, shape (n_dims, n_dims, n_discrete_states)
        Cov[X | S = j]
    mixing_weights : jnp.ndarray, shape (n_discrete_states,)
        P[S = j]

    Returns
    -------
    unconditional_mean_x : jnp.ndarray, shape (n_dims,)
        E[X]
    unconditional_cov_x : jnp.ndarray, shape (n_dims, n_dims)
        Cov[X]
    """
    unconditional_mean_x = conditional_means_x @ mixing_weights  # E[X]
    diff_x = conditional_means_x - unconditional_mean_x[:, None]

    unconditional_cov_xx = (
        conditional_cov @ mixing_weights + (diff_x * mixing_weights) @ diff_x.T
    )  # E[XX]

    return unconditional_mean_x, unconditional_cov_xx


def collapse_gaussian_mixture_cross_covariance(
    conditional_means_x: jnp.ndarray,
    conditional_means_y: jnp.ndarray,
    conditional_cross_cov: jnp.ndarray,
    mixing_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute cross-covariance when collapsing a Gaussian mixture.

    Parameters
    ----------
    conditional_means_x : jnp.ndarray, shape (n_dims, n_discrete_states)
        E[X | S = j]
    conditional_means_y : jnp.ndarray, shape (n_dims, n_discrete_states)
        E[Y | S = j]
    conditional_cross_cov : jnp.ndarray, shape (n_dims, n_dims, n_discrete_states)
        E[X,Y^T | S = j], Conditional expectation of the outer product.
    mixing_weights : jnp.ndarray, shape (n_discrete_states,)
        P[S = j]

    Returns
    -------
    unconditional_mean_x : jnp.ndarray, shape (n_dims,)
        E[X]
    unconditional_mean_y : jnp.ndarray, shape (n_dims,)
        E[Y]
    unconditional_cov_xy : jnp.ndarray, shape (n_dims, n_dims)
        E[X,Y^T]
    """

    unconditional_mean_x = conditional_means_x @ mixing_weights  # E[X]
    unconditional_mean_y = conditional_means_y @ mixing_weights  # E[Y]

    diff_x = conditional_means_x - unconditional_mean_x[:, None]
    diff_y = conditional_means_y - unconditional_mean_y[:, None]

    unconditional_cov_xy = (
        conditional_cross_cov @ mixing_weights + (diff_x * mixing_weights) @ diff_y.T
    )  # E[XY]

    return unconditional_mean_x, unconditional_mean_y, unconditional_cov_xy


collapse_gaussian_mixture_per_discrete_state = jax.vmap(
    collapse_gaussian_mixture, in_axes=(-1, -1, -1), out_axes=(-1, -1)
)
collapse_gaussian_mixture_over_next_discrete_state = jax.vmap(
    collapse_gaussian_mixture, in_axes=(1, 2, 0), out_axes=(-1, -1)
)
collapse_cross_gaussian_mixture_across_states = jax.vmap(
    collapse_gaussian_mixture_cross_covariance, in_axes=(2, 2, 3, 1), out_axes=(1, 1, 2)
)


def _update_discrete_state_probabilities(
    pair_cond_marginal_likelihood_scaled: jnp.ndarray,
    discrete_transition_matrix: jnp.ndarray,
    prev_filter_discrete_prob: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Update the discrete state probabilities using the discrete transition matrix.

    Parameters
    ----------
    pair_cond_marginal_likelihood_scaled : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
    discrete_transition_matrix : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Z(i, j) = P(S_t=j | S_{t-1}=i)
    prev_filter_discrete_prob : jnp.ndarray, shape (n_discrete_states,)
        M_{t-1|t-1}(i) = Pr(S_{t-1}=i | y_{1:t-1})

    Returns
    -------
    filter_discrete_prob : jnp.ndarray, shape (n_discrete_states,)
        Updated discrete state probabilities, M_{t|t}(j) = Pr(S_t=j | y_{1:t})
    filter_backward_cond_prob : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Mixing weights for the discrete states, W^{i|j} = Pr(S_{t-1}=i | S_t=j, y_{1:t})
    predictive_likelihood_term_sum : float
        Scaled predictive likelihood sum
    """
    # joint discrete state prob between time steps
    # M_{t-1,t | t}(i, j) = P(S_{t-1}=i, S_t=j | y_{1:t})
    joint_discrete_state_prob = (
        pair_cond_marginal_likelihood_scaled  # L(i, j)
        * discrete_transition_matrix  # Z(i, j)
        * prev_filter_discrete_prob[:, None]  # M_{t-1|t-1}(i)
    )
    predictive_likelihood_term_sum = jnp.sum(joint_discrete_state_prob)
    joint_discrete_state_prob /= jnp.sum(joint_discrete_state_prob)

    # M_{t|t}(j) = Pr(S_t=j | y_{1:t})
    filter_discrete_prob = jnp.sum(joint_discrete_state_prob, axis=0)
    # W^{i|j} = Pr(S_{t-1}=i | S_t=j, y_{1:t})
    filter_backward_cond_prob = (
        joint_discrete_state_prob / filter_discrete_prob[None, :]
    )

    return (
        filter_discrete_prob,
        filter_backward_cond_prob,
        predictive_likelihood_term_sum,
    )


def _scale_likelihood(log_likelihood: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    """Scale the log likelihood to avoid numerical underflow.

    Parameters
    ----------
    log_likelihood : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Log likelihood of the discrete states.
    Returns
    -------
    scaled_likelihood : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Scaled log likelihood of the discrete states.
    ll_max : float
        Maximum log likelihood of the discrete states.
    """

    ll_max = log_likelihood.max()
    ll_max = jnp.where(jnp.isfinite(ll_max), ll_max, 0.0)
    return jnp.exp(log_likelihood - ll_max), ll_max


def switching_kalman_filter(
    init_state_cond_mean: jnp.ndarray,
    init_state_cond_cov: jnp.ndarray,
    init_discrete_state_prob: jnp.ndarray,
    obs: jnp.ndarray,
    discrete_transition_matrix: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    measurement_cov: jnp.ndarray,
) -> tuple[
    jnp.ndarray,  # Filtered mean of the continuous latent state
    jnp.ndarray,  # Filtered covariance of the continuous latent state
    jnp.ndarray,  # Filtered probability of the discrete states
    jnp.ndarray,  # Last filtered conditional mean of the continuous latent state
    float,  # Marginal log likelihood of the observations
]:
    """Switching Kalman filter for a linear Gaussian state space model with discrete states.

    Parameters
    ----------
    init_state_cond_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
        Initial value of the continuous latent state $x_1$
    init_state_cond_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
        Initial covariance of the continuous latent state $P_1$
    init_discrete_state_prob : jnp.ndarray, shape (n_discrete_states,)
        Initial probability of the discrete states $p(S_1)$
    obs : jnp.ndarray, shape (n_time, n_obs_dim)
        Observations $y_{1:T}$
    discrete_transition_matrix : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Transition matrix for the discrete states $B$
    continuous_transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
        Transition matrix for the continuous states $A$
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
        Process noise covariance matrix. $\\Sigma$
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Map observations to the continuous states $H$
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Measurement variance. $R$

    Returns
    -------
    state_cond_filter_mean : jnp.ndarray, shape (n_time, n_cont_states, n_discrete_states)
        Filtered mean of the continuous latent state
    state_cond_filter_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        Filtered covariance of the continuous latent state
    filter_discrete_state_prob : jnp.ndarray, shape (n_time, n_discrete_states)
        Filtered probability of the discrete states
    last_pair_cond_filter_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states, n_discrete_states)
        Last filtered conditional mean of the continuous latent state
    marginal_log_likelihood : float
        Marginal log likelihood of the observations

    """

    def _step(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], obs_t: jnp.ndarray
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float],  # Next carry
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],  # Stacked output
    ]:
        """One step of the switching Kalman filter.

        Parameters
        ----------
        carry : tuple
            prev_state_cond_filter_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
                Previous state mean.
            prev_state_cond_filter_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
                Previous state covariance.
            prev_filter_discrete_prob : jnp.ndarray, shape (n_discrete_states,)
                Previous discrete state probabilities
            pair_cond_marginal_log_likelihood : float
                Previous marginal log likelihood
        obs_t : jnp.ndarray, shape (n_obs_dim,)
            Observation at time t

        Returns
        -------
        carry : tuple
            prev_state_cond_filter_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
                Posterior state mean.
            prev_state_cond_filter_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
                Posterior state covariance.
            prev_filter_discrete_prob : jnp.ndarray, shape (n_discrete_states,)
                Posterior discrete state probabilities
            marginal_log_likelihood : float
                Posterior marginal log likelihood
        stack : tuple
            state_cond_filter_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
                Posterior state mean.
            state_cond_filter_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
                Posterior state covariance.
            filter_discrete_prob : jnp.ndarray, shape (n_discrete_states,)
                Posterior discrete state probabilities
            pair_cond_filter_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states, n_discrete_states)
                Conditional means of the continuous latent state
        """
        (
            prev_state_cond_filter_mean,
            prev_state_cond_filter_cov,
            prev_filter_discrete_prob,
            marginal_log_likelihood,
        ) = carry

        # Kalman update for each pair of discrete states
        # P(x_t | y_{1:t}, S_t = j, S_{t-1} = i)
        # vmap twice over the discrete states
        (
            pair_cond_filter_mean,  # x^{ij}_{t|t}
            pair_cond_filter_cov,  # V^{ij}_{t|t}
            pair_cond_marginal_log_likelihood,  # log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)
        ) = _kalman_filter_update_per_discrete_state_pair(
            prev_state_cond_filter_mean,  # x^i_{t-1|t-1}
            prev_state_cond_filter_cov,  # V^i_{t-1|t-1}
            obs_t,  # y_t
            continuous_transition_matrix,  # A
            process_cov,  # Sigma
            measurement_matrix,  # H
            measurement_cov,  # R
        )

        # Make sure the likelihood is normalized to max 1 for numerical stability
        pair_cond_marginal_likelihood_scaled, ll_max = _scale_likelihood(
            pair_cond_marginal_log_likelihood
        )

        (
            filter_discrete_prob,  # M_{t|t}(j) = P(S_t=j | y_{1:t})
            filter_backward_cond_prob,  # P(S_{t-1}=i | S_t=j, y_{1:t})
            predictive_likelihood_term_sum,  # Sum over i,j of unnormalized joint prob
        ) = _update_discrete_state_probabilities(
            pair_cond_marginal_likelihood_scaled,  # shape (n_discrete_states, n_discrete_states)
            discrete_transition_matrix,  # P(S_t=j | S_{t-1}=i)
            prev_filter_discrete_prob,  # M_{t-1|t-1}(i)
        )

        marginal_log_likelihood += ll_max + jnp.log(predictive_likelihood_term_sum)

        # Collapse pair-conditional Gaussians P(x_t | ..., S_t=j, S_{t-1}=i)
        # over S_{t-1}=i using weights P(S_{t-1}=i | S_t=j, y_{1:t})
        # to get state-conditional Gaussians P(x_t | ..., S_t=j)
        state_cond_filter_mean, state_cond_filter_cov = (
            collapse_gaussian_mixture_per_discrete_state(
                pair_cond_filter_mean,  # x^{ij}_{t|t}
                pair_cond_filter_cov,  # V^{ij}_{t|t}
                filter_backward_cond_prob,  # P(S_{t-1}=i | S_t=j, y_{1:t})
            )
        )

        return (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_prob,
            marginal_log_likelihood,
        ), (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_prob,
            pair_cond_filter_mean,
        )

    marginal_log_likelihood = jnp.array(0.0)
    (_, _, _, marginal_log_likelihood), (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_state_prob,
        pair_cond_filter_mean,
    ) = jax.lax.scan(
        _step,
        (
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            marginal_log_likelihood,
        ),
        obs,
    )

    return (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_state_prob,
        pair_cond_filter_mean[-1],
        marginal_log_likelihood,
    )


_kalman_smoother_update_per_discrete_state_pair = jax.vmap(
    jax.vmap(
        _kalman_smoother_update, in_axes=(None, None, -1, -1, None, None), out_axes=-1
    ),
    in_axes=(-1, -1, None, None, -1, -1),
    out_axes=-1,
)


def _update_smoother_discrete_probabilities(
    filter_discrete_prob: jnp.ndarray,
    discrete_state_transition_matrix: jnp.ndarray,
    next_smoother_discrete_prob: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """

    Parameters
    ----------
    filter_discrete_prob : jnp.ndarray, shape (n_discrete_states,)
        Pr(S_t=j | y_{1:t}), shape (n_discrete_states,), M_{t | t}(j)
    discrete_state_transition_matrix : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Pr(S_t=j | S_{t-1}=k), shape (n_discrete_states, n_discrete_states), Z(j, k)
    next_smoother_discrete_prob : jnp.ndarray, shape (n_discrete_states,)
        Pr(S_{t+1}=k | y_{1:T}), shape (n_discrete_states,) M_{t+1 | T}(k)

    Returns
    -------
    smoother_discrete_state_prob : jnp.ndarray, shape (n_discrete_states,)
        Pr(S_t=j | y_{1:T}), shape (n_discrete_states,),  M_{t | T}(j)
    smoother_backward_cond_prob : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Pr(S_t=j | S_{t+1}=k, y_{1:T}), shape (n_discrete_states, n_discrete_states), U^{j | k}_t
    joint_smoother_discrete_prob : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Pr(S_t=j, S_{t+1}=k | y_{1:T}), shape (n_discrete_states, n_discrete_states)
    smoother_forward_cond_prob : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Pr(S_{t+1}=k | S{t}=j, y_{1:T}), shape (n_discrete_states, n_discrete_states), W^{k | j}_t

    """
    # Discrete smoother prob
    # P(S_t = j, S_{t+1} = k | y_{1:T})
    smoother_backward_cond_prob = (
        filter_discrete_prob[:, None] * discrete_state_transition_matrix
    )
    smoother_backward_cond_prob /= jnp.sum(smoother_backward_cond_prob, axis=0)

    joint_smoother_discrete_prob = (
        smoother_backward_cond_prob * next_smoother_discrete_prob
    )
    # P(S_t = j | y_{1:T})
    smoother_discrete_state_prob = jnp.sum(joint_smoother_discrete_prob, axis=1)
    # P(S_{t+1} = k | S_t = j, y_{1:T})
    smoother_forward_cond_prob = (
        joint_smoother_discrete_prob / smoother_discrete_state_prob[:, None]
    )

    return (
        smoother_discrete_state_prob,
        smoother_backward_cond_prob,
        joint_smoother_discrete_prob,
        smoother_forward_cond_prob,
    )


def switching_kalman_smoother(
    filter_mean: jnp.ndarray,
    filter_cov: jnp.ndarray,
    filter_discrete_state_prob: jnp.ndarray,
    last_filter_conditional_cont_mean: jnp.ndarray,
    process_cov: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    discrete_state_transition_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Switching Kalman smoother for a linear Gaussian state space model with discrete states.

    Parameters
    ----------
    filter_mean : jnp.ndarray, shape (n_time, n_cont_states, n_discrete_states)
    filter_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    filter_discrete_state_prob : jnp.ndarray, shape (n_time, n_discrete_states)
    last_filter_conditional_cont_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states, n_discrete_states)
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
    continuous_transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
    discrete_state_transition_matrix : jnp.ndarray, shape (n_discrete_states, n_discrete_states)

    Returns
    -------
    overall_smoother_mean : jnp.ndarray, shape (n_time, n_cont_states)
    overall_smoother_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states)
    smoother_discrete_state_prob : jnp.ndarray, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jnp.ndarray, shape (n_time - 1, n_discrete_states, n_discrete_states)
    overall_smoother_cross_cov : jnp.ndarray, shape (n_time - 1, n_cont_states, n_cont_states)
    """

    def _step(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ]:
        """

        Parameters
        ----------
        carry : tuple
            next_smoother_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
            next_smoother_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
            next_discrete_state_prob : jnp.ndarray, shape (n_discrete_states,)
            next_conditional_cont_means : jnp.ndarray, shape (n_cont_states, n_discrete_states, n_discrete_states)
        args : tuple
            state_cond_filter_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
            state_cond_filter_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
            filter_discrete_prob : jnp.ndarray, shape (n_discrete_states,)

        Returns
        -------
        carry : tuple
            next_state_cond_smoother_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
            next_state_cond_smoother_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
            next_smoother_discrete_prob : jnp.ndarray, shape (n_discrete_states,)
            next_pair_cond_smoother_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states, n_discrete_states)
        args : tuple
            state_cond_filter_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
            state_cond_filter_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
            filter_discrete_prob : jnp.ndarray, shape (n_discrete_states,)
        """
        (
            next_state_cond_smoother_mean,
            next_state_cond_smoother_cov,
            next_smoother_discrete_prob,
            next_pair_cond_smoother_mean,
        ) = carry

        state_cond_filter_mean, state_cond_filter_cov, filter_discrete_prob = args

        # 1. Smooth for each discrete state pair
        (
            pair_cond_smoother_mean,  # E[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_discrete_states, n_discrete_states)
            pair_cond_smoother_covs,  # Cov[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
            pair_cond_smoother_cross_covs,  # Cov[X_{t+1}, X_t | y_{1:T}, S_{t+1}=j, S_{t}=k], shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        ) = _kalman_smoother_update_per_discrete_state_pair(
            next_state_cond_smoother_mean,  # E[X_{t+1} | y_{1:T}, S_t=k], shape (n_cont_states, n_discrete_states)
            next_state_cond_smoother_cov,  # Cov[X_{t+1} | y_{1:T}, S_t=k], shape (n_cont_states, n_cont_states, n_discrete_states)
            state_cond_filter_mean,  # E[X_t | y_{1:t}, S_t=j], shape (n_cont_states, n_discrete_states)
            state_cond_filter_cov,  # Cov[X_t | y_{1:t}, S_t=j], shape (n_cont_states, n_cont_states, n_discrete_states)
            process_cov,  # Cov[X_{t+1} | X_t], shape (n_cont_states, n_cont_states, n_discrete_states)
            continuous_transition_matrix,  # E[X_{t+1} | X_t], shape (n_cont_states, n_cont_states, n_discrete_states)
        )

        # 2. Compute discrete state intermediates
        (
            smoother_discrete_state_prob,  # Pr(S_t=j | y_{1:T}), shape (n_discrete_states,),  M_{t | T}(j)
            smoother_backward_cond_prob,  # Pr(S_t=j | S_{t+1}=k, y_{1:T}), shape (n_discrete_states, n_discrete_states), U^{j | k}_t
            joint_smoother_discrete_prob,  # Pr(S_t=j, S_{t+1}=k | y_{1:T}), shape (n_discrete_states, n_discrete_states)
            smoother_forward_cond_prob,  # Pr(S_{t+1}=k | S{t}=j, y_{1:T}), shape (n_discrete_states, n_discrete_states), W^{k | j}_t
        ) = _update_smoother_discrete_probabilities(
            filter_discrete_prob,  # Pr(S_t=j | y_{1:t}), shape (n_discrete_states,), M_{t | t}(j)
            discrete_state_transition_matrix,  # Pr(S_t=j | S_{t-1}=k), shape (n_discrete_states, n_discrete_states), Z(j, k)
            next_smoother_discrete_prob,  # Pr(S_{t+1}=k | y_{1:T}), shape (n_discrete_states,) M_{t+1 | T}(k)
        )

        # 3. Collapse conditional mean and covariance (n_states x n_states -> n_states)
        (
            state_cond_smoother_means,  # E[X_t | y_{1:T}, S_{t}=j], shape (n_cont_states, n_discrete_states)
            state_cond_smoother_covs,  # Cov[X_t | y_{1:T}, S_{t}=j], shape (n_cont_states, n_cont_states, n_discrete_states)
        ) = collapse_gaussian_mixture_over_next_discrete_state(
            pair_cond_smoother_mean,  # E[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_discrete_states, n_discrete_states)
            pair_cond_smoother_covs,  # Cov[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
            smoother_forward_cond_prob,  # Pr(S_{t+1} = k | S{t} = j, y_{1:T}), shape (n_discrete_states, n_discrete_states), W^{k | j}_t
        )

        # 4. Collapse to single mean and covariance (n_states -> 1)
        (
            overall_smoother_mean,  # E[X_t | y_{1:T}], shape (n_cont_states,), x_{t|T}
            overall_smoother_covs,  # Cov[X_t | y_{1:T}], shape (n_cont_states, n_cont_states)
        ) = collapse_gaussian_mixture(
            state_cond_smoother_means,  # E[X_t | y_{1:T}, S_{t}=j], shape (n_cont_states, n_discrete_states)
            state_cond_smoother_covs,  # Cov[X_t | y_{1:T}, S_{t}=j], shape (n_cont_states, n_cont_states, n_discrete_states)
            smoother_discrete_state_prob,  # Pr(S_t = j | y_{1:T}), shape (n_discrete_states,),  M_{t | T}(j)
        )

        # 5. Collapse cross covariance
        (
            state_cond_smoother_mean_tplus1,  # E[X_{t+1} | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_discrete_states), x^{()k}_{t+1 | T}
            smoother_mean_t_cond_Stplus1,  # E[X_t | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_discrete_states), x^{()k}_{t | T}
            state_cond_smoother_cross_cov,  # Cov(X_{t+1}, X_t | y_{1:T}, S_{t+1}=k), shape (n_cont_states, n_cont_states, n_discrete_states), V^k_{t+1, t | T}:
        ) = collapse_cross_gaussian_mixture_across_states(
            next_pair_cond_smoother_mean,  # E[X_{t+1} | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_discrete_states, n_discrete_states), x^{j(k)}_{t+1 | T}
            pair_cond_smoother_mean,  # E[X_t | y_{1:T}, S_t=j, S_{t+1}=k], shape (n_cont_states, n_discrete_states, n_discrete_states), x^{(j)k}_{t | T}
            pair_cond_smoother_cross_covs,  # Cov(X_{t+1}, X_t | y_{1:T}, S_t=j, S_{t+1}=k), shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states), V^{j(k)}_{t+1,t | T}
            smoother_backward_cond_prob,  # Pr(S_t=j | S_{t+1}=k, y_{1:T}), shape (n_discrete_states, n_discrete_states), U^{j | k}_t
        )

        # Cross collapse to a single Gaussian
        # overall_smoother_cross_cov, shape (n_cont_states, n_cont_states)
        _, _, overall_smoother_cross_cov = collapse_gaussian_mixture_cross_covariance(
            state_cond_smoother_mean_tplus1,  # E[X_{t+1} | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_discrete_states), x^{()k}_{t+1 | T}
            smoother_mean_t_cond_Stplus1,  # E[X_t | y_{1:T}, S_{t+1}=k], shape (n_cont_states, n_discrete_states), x^{()k}_{t | T}
            state_cond_smoother_cross_cov,  # V^k_{t+1, t | T}: state-conditional smoother cross covariance
            next_smoother_discrete_prob,  # Pr(S_{t+1} = k | y_{1:T}), M_{t+1 | T}(k)
        )

        return (
            state_cond_smoother_means,
            state_cond_smoother_covs,
            smoother_discrete_state_prob,
            pair_cond_smoother_mean,
        ), (
            overall_smoother_mean,
            overall_smoother_covs,
            smoother_discrete_state_prob,
            joint_smoother_discrete_prob,
            overall_smoother_cross_cov,
        )

    _, (
        overall_smoother_mean,
        overall_smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        overall_smoother_cross_cov,
    ) = jax.lax.scan(
        _step,
        (
            filter_mean[-1],  # shape (n_cont_states, n_discrete_states)
            filter_cov[-1],  # shape (n_cont_states, n_cont_states, n_discrete_states)
            filter_discrete_state_prob[-1],  # shape (n_discrete_states,)
            last_filter_conditional_cont_mean,  # shape (n_cont_states, n_discrete_states, n_discrete_states)
        ),
        (
            filter_mean[:-1],
            filter_cov[:-1],
            filter_discrete_state_prob[:-1],
        ),
        reverse=True,
    )

    last_smoother_mean, last_smoother_cov = collapse_gaussian_mixture(
        filter_mean[-1], filter_cov[-1], filter_discrete_state_prob[-1]
    )
    overall_smoother_mean = jnp.concatenate(
        [overall_smoother_mean, last_smoother_mean[None]], axis=0
    )
    overall_smoother_covs = jnp.concatenate(
        [overall_smoother_covs, last_smoother_cov[None]], axis=0
    )
    smoother_discrete_state_prob = jnp.concatenate(
        [smoother_discrete_state_prob, filter_discrete_state_prob[-1][None]], axis=0
    )

    return (
        overall_smoother_mean,
        overall_smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        overall_smoother_cross_cov,
    )


def weighted_sum_of_outer_products(
    x: jnp.ndarray, y: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    """Compute the weighted outer sum of two arrays.
    Parameters
    ----------
    x : jnp.ndarray, shape (n_time, x_dims, n_discrete_states)
        First array.
    y : jnp.ndarray, shape (n_time, y_dims, n_discrete_states)
        Second array.
    weights : jnp.ndarray, shape (n_time, n_discrete_states)
        Weights for the outer sum.

    Returns
    -------
    weighted_sum_of_outer_products: jnp.ndarray, shape (x_dims, y_dims, n_discrete_states)
        Weighted outer sum of x and y.
    """
    return jnp.einsum("tcm, tdm, tm -> cdm", x, y, weights)


psd_solve_per_discrete_state = jax.vmap(
    lambda x, y: psd_solve(x, y.T).T, in_axes=(-1, -1), out_axes=-1
)

cov_solve_per_discrete_state = jax.vmap(
    lambda x, y, z, n: symmetrize((x - y @ z.T) / n),
    in_axes=(-1, -1, -1, -1),
    out_axes=-1,
)


def switching_kalman_maximization_step(
    obs: jnp.ndarray,
    state_cond_smoother_means: jnp.ndarray,
    state_cond_smoother_covs: jnp.ndarray,
    smoother_discrete_state_prob: jnp.ndarray,
    smoother_joint_discrete_state_prob: jnp.ndarray,
    pair_cond_smoother_cross_cov: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Maximization step for the switching Kalman filter.

    Parameters
    ----------
    obs : jnp.ndarray, shape (n_time, n_obs_dim)
        Observations.
    state_cond_smoother_means : jnp.ndarray, shape (n_time, n_cont_states, n_discrete_states)
        smoother mean.
    state_cond_smoother_covs : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        smoother covariance.
    smoother_discrete_state_prob : jnp.ndarray, shape (n_time, n_discrete_states)
        smoother discrete state probabilities.
    smoother_joint_discrete_state_prob : jnp.ndarray, shape (n_time - 1, n_discrete_states, n_discrete_states)
        smoother joint discrete state probabilities.
    pair_cond_smoother_cross_cov : jnp.ndarray, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        smoother cross-covariance.

    Returns
    -------
    continuous_transition_matrix : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
        Transition matrix.
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Measurement matrix.
    process_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
        Process covariance.
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Measurement covariance.
    init_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
        Initial mean of the continuous latent state.
    init_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
        Initial covariance of the continuous latent state.
    discrete_transition_matrix : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Transition matrix for the discrete states.
    init_discrete_state_prob : jnp.ndarray, shape (n_discrete_states,)
        Initial discrete state probabilities.


    References
    ----------
    ... [1] Roweis, S. T., Ghahramani, Z., & Hinton, G. E. (1999). A unifying review of
    linear Gaussian models. Neural computation, 11(2), 305-345.
    """

    n_time = smoother_discrete_state_prob.sum(axis=0)
    n_time_1 = smoother_discrete_state_prob[1:].sum(axis=0)

    # Compute intermediate expectation terms
    gamma = jnp.sum(
        state_cond_smoother_covs * smoother_discrete_state_prob[:, None, None], axis=0
    ) + weighted_sum_of_outer_products(
        state_cond_smoother_means,
        state_cond_smoother_means,
        smoother_discrete_state_prob,
    )

    delta = weighted_sum_of_outer_products(
        obs[..., None], state_cond_smoother_means, smoother_discrete_state_prob
    )
    alpha = weighted_sum_of_outer_products(
        obs[..., None], obs[..., None], smoother_discrete_state_prob
    )

    last_gamma = (
        state_cond_smoother_covs[-1] * smoother_discrete_state_prob[-1, None, None]
    ) + weighted_sum_of_outer_products(
        state_cond_smoother_means[[-1]],
        state_cond_smoother_means[[-1]],
        smoother_discrete_state_prob[[-1]],
    )
    first_gamma = (
        state_cond_smoother_covs[0] * smoother_discrete_state_prob[0, None, None]
    ) + weighted_sum_of_outer_products(
        state_cond_smoother_means[[0]],
        state_cond_smoother_means[[0]],
        smoother_discrete_state_prob[[0]],
    )
    gamma1 = gamma - last_gamma
    gamma2 = gamma - first_gamma

    # beta = jnp.swapaxes(
    #     pair_cond_smoother_cross_cov * smoother_discrete_state_prob[1:, None, None],
    #     1,
    #     2,
    # ).sum(axis=0)

    beta = jnp.einsum(
        "tij,tcdij->cdi",
        smoother_joint_discrete_state_prob,
        pair_cond_smoother_cross_cov,
    )

    # gamma: (n_cont_states, n_cont_states, n_discrete_states)
    # beta: (n_cont_states, n_cont_states, n_discrete_states)
    # alpha: (n_obs_dim, n_obs_dim, n_discrete_states)
    # delta: (n_obs_dim, n_cont_states, n_discrete_states)

    # Measurement matrix and covariance
    measurement_matrix = psd_solve_per_discrete_state(gamma, delta)
    # measurement_matrix: shape (n_obs_dim, n_cont_states, n_discrete_states)
    measurement_cov = cov_solve_per_discrete_state(
        alpha, measurement_matrix, delta, n_time
    )

    # Transition matrix
    continuous_transition_matrix = psd_solve_per_discrete_state(gamma1, beta)

    # Process covariance
    process_cov = cov_solve_per_discrete_state(
        gamma2, continuous_transition_matrix, beta, n_time_1
    )
    # Initial mean and covariance
    init_state_cond_mean = state_cond_smoother_means[0]
    init_state_cond_cov = state_cond_smoother_covs[0]

    # Discrete transition matrix
    discrete_state_transition = (
        smoother_joint_discrete_state_prob.sum(
            axis=0,
        )
        / smoother_discrete_state_prob[:-1].sum(axis=0)[:, None]
    )
    # Ensure the discrete transition matrix is normalized
    # so that each row sums to 1
    discrete_state_transition /= discrete_state_transition.sum(axis=1, keepdims=True)

    init_discrete_state_prob = smoother_discrete_state_prob[0]
    init_discrete_state_prob /= jnp.sum(init_discrete_state_prob)

    return (
        continuous_transition_matrix,
        measurement_matrix,
        process_cov,
        measurement_cov,
        init_state_cond_mean,
        init_state_cond_cov,
        discrete_state_transition,
        init_discrete_state_prob,
    )
