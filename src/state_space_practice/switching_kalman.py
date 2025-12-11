"""Switching Kalman filter and smoother and EM algorithm.

References
----------
1. Shumway, R.H., and Stoffer, D.S. (1991). Dynamic Linear Models With Switching. 8.
2. Murphy, K.P. (1998). Switching kalman filters.
3. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2022). Switching Functional Network Models of Oscillatory Brain Dynamics. In 2022 56th Asilomar Conference on Signals, Systems, and Computers (IEEE), pp. 607–612. https://doi.org/10.1109/IEEECONF56349.2022.10052077.
4. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2024). Switching Models of Oscillatory Networks Greatly Improve Inference of Dynamic Functional Connectivity. Preprint at arXiv.
5. https://github.com/Stephen-Lab-BU/Switching_Oscillator_Networks
"""

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
    conditional_means_x: jax.Array,
    conditional_cov: jax.Array,
    mixing_weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Collapse a mixture of Gaussians.

    Parameters
    ----------
    conditional_means_x : jax.Array, shape (n_dims, n_discrete_states)
        E[X | S = j]
    conditional_cov : jax.Array, shape (n_dims, n_dims, n_discrete_states)
        Cov[X | S = j]
    mixing_weights : jax.Array, shape (n_discrete_states,)
        P[S = j]

    Returns
    -------
    unconditional_mean_x : jax.Array, shape (n_dims,)
        E[X]
    unconditional_cov_x : jax.Array, shape (n_dims, n_dims)
        Cov[X]
    """
    unconditional_mean_x = conditional_means_x @ mixing_weights  # E[X]
    diff_x = conditional_means_x - unconditional_mean_x[:, None]

    unconditional_cov_xx = (
        conditional_cov @ mixing_weights + (diff_x * mixing_weights) @ diff_x.T
    )  # E[XX]

    return unconditional_mean_x, unconditional_cov_xx


def collapse_gaussian_mixture_cross_covariance(
    conditional_means_x: jax.Array,
    conditional_means_y: jax.Array,
    conditional_cross_cov: jax.Array,
    mixing_weights: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute cross-covariance when collapsing a Gaussian mixture.

    Parameters
    ----------
    conditional_means_x : jax.Array, shape (n_dims, n_discrete_states)
        E[X | S = j]
    conditional_means_y : jax.Array, shape (n_dims, n_discrete_states)
        E[Y | S = j]
    conditional_cross_cov : jax.Array, shape (n_dims, n_dims, n_discrete_states)
        E[X,Y^T | S = j], Conditional expectation of the outer product.
    mixing_weights : jax.Array, shape (n_discrete_states,)
        P[S = j]

    Returns
    -------
    unconditional_mean_x : jax.Array, shape (n_dims,)
        E[X]
    unconditional_mean_y : jax.Array, shape (n_dims,)
        E[Y]
    unconditional_cov_xy : jax.Array, shape (n_dims, n_dims)
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


def _divide_safe(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """Divides two arrays, while setting the result to 0.0
    if the denominator is 0.0.

    Parameters
    ----------
    numerator : jax.Array
    denominator : jax.Array
    """
    return jnp.where(denominator == 0.0, 0.0, numerator / denominator)


# Minimum probability threshold for numerical stability
_LOG_PROB_FLOOR = 1e-10
_LOG_FLOOR_VALUE = -23.0  # approximately log(1e-10)


def _safe_log(x: jax.Array) -> jax.Array:
    """Compute log(x) with numerical stability for small probabilities.

    Uses jnp.where to explicitly handle near-zero values rather than
    silently adding a small constant. This makes the numerical treatment
    explicit and avoids potential issues where true zeros could silently
    produce finite values.

    Parameters
    ----------
    x : jax.Array
        Input array (typically probabilities).

    Returns
    -------
    jax.Array
        log(x) where x > _LOG_PROB_FLOOR, otherwise _LOG_FLOOR_VALUE.
    """
    return jnp.where(x > _LOG_PROB_FLOOR, jnp.log(x), _LOG_FLOOR_VALUE)


def _update_discrete_state_probabilities(
    pair_cond_marginal_likelihood_scaled: jax.Array,
    discrete_transition_matrix: jax.Array,
    prev_filter_discrete_prob: jax.Array,
) -> tuple[jax.Array, jax.Array, float]:
    """Update the discrete state probabilities using the discrete transition matrix.

    Parameters
    ----------
    pair_cond_marginal_likelihood_scaled : jax.Array, shape (n_discrete_states, n_discrete_states)
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Z(i, j) = P(S_t=j | S_{t-1}=i)
    prev_filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        M_{t-1|t-1}(i) = Pr(S_{t-1}=i | y_{1:t-1})

    Returns
    -------
    filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        Updated discrete state probabilities, M_{t|t}(j) = Pr(S_t=j | y_{1:t})
    filter_backward_cond_prob : jax.Array, shape (n_discrete_states, n_discrete_states)
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

    joint_discrete_state_prob = _divide_safe(
        joint_discrete_state_prob, predictive_likelihood_term_sum
    )

    # M_{t|t}(j) = Pr(S_t=j | y_{1:t})
    filter_discrete_prob = jnp.sum(joint_discrete_state_prob, axis=0)
    # W^{i|j} = Pr(S_{t-1}=i | S_t=j, y_{1:t})
    filter_backward_cond_prob = _divide_safe(
        joint_discrete_state_prob, filter_discrete_prob[None, :]
    )

    return (
        filter_discrete_prob,
        filter_backward_cond_prob,
        predictive_likelihood_term_sum,
    )


def _scale_likelihood(log_likelihood: jax.Array) -> tuple[jax.Array, float]:
    """Scale the log likelihood to avoid numerical underflow.

    Parameters
    ----------
    log_likelihood : jax.Array, shape (n_discrete_states, n_discrete_states)
        Log likelihood of the discrete states.
    Returns
    -------
    scaled_likelihood : jax.Array, shape (n_discrete_states, n_discrete_states)
        Scaled log likelihood of the discrete states.
    ll_max : float
        Maximum log likelihood of the discrete states.
    """

    ll_max = log_likelihood.max()
    ll_max = jnp.where(jnp.isfinite(ll_max), ll_max, 0.0)
    return jnp.exp(log_likelihood - ll_max), ll_max


def switching_kalman_filter(
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    obs: jax.Array,
    discrete_transition_matrix: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[
    jax.Array,  # Filtered mean of the continuous latent state
    jax.Array,  # Filtered covariance of the continuous latent state
    jax.Array,  # Filtered probability of the discrete states
    jax.Array,  # Last filtered conditional mean of the continuous latent state
    float,  # Marginal log likelihood of the observations
]:
    """Switching Kalman filter for a linear Gaussian state space model with discrete states.

    Parameters
    ----------
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Initial value of the continuous latent state $x_1$
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Initial covariance of the continuous latent state $P_1$
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Initial probability of the discrete states $p(S_1)$
    obs : jax.Array, shape (n_time, n_obs_dim)
        Observations $y_{1:T}$
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Transition matrix for the discrete states $B$
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Transition matrix for the continuous states $A$
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Process noise covariance matrix. $\\Sigma$
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Map observations to the continuous states $H$
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Measurement variance. $R$

    Returns
    -------
    state_cond_filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        Filtered mean of the continuous latent state
    state_cond_filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        Filtered covariance of the continuous latent state
    filter_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        Filtered probability of the discrete states
    last_pair_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
        Last filtered conditional mean of the continuous latent state
    marginal_log_likelihood : float
        Marginal log likelihood of the observations

    """

    def _step(
        carry: tuple[jax.Array, jax.Array, jax.Array, float], obs_t: jax.Array
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, float],  # Next carry
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],  # Stacked output
    ]:
        """One step of the switching Kalman filter.

        Parameters
        ----------
        carry : tuple
            prev_state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
                Previous state mean.
            prev_state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
                Previous state covariance.
            prev_filter_discrete_prob : jax.Array, shape (n_discrete_states,)
                Previous discrete state probabilities
            pair_cond_marginal_log_likelihood : float
                Previous marginal log likelihood
        obs_t : jax.Array, shape (n_obs_dim,)
            Observation at time t

        Returns
        -------
        carry : tuple
            prev_state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
                Posterior state mean.
            prev_state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
                Posterior state covariance.
            prev_filter_discrete_prob : jax.Array, shape (n_discrete_states,)
                Posterior discrete state probabilities
            marginal_log_likelihood : float
                Posterior marginal log likelihood
        stack : tuple
            state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
                Posterior state mean.
            state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
                Posterior state covariance.
            filter_discrete_prob : jax.Array, shape (n_discrete_states,)
                Posterior discrete state probabilities
            pair_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
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
    filter_discrete_prob: jax.Array,
    discrete_state_transition_matrix: jax.Array,
    next_smoother_discrete_prob: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """

    Parameters
    ----------
    filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        Pr(S_t=j | y_{1:t}), shape (n_discrete_states,), M_{t | t}(j)
    discrete_state_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Pr(S_t=j | S_{t-1}=k), shape (n_discrete_states, n_discrete_states), Z(j, k)
    next_smoother_discrete_prob : jax.Array, shape (n_discrete_states,)
        Pr(S_{t+1}=k | y_{1:T}), shape (n_discrete_states,) M_{t+1 | T}(k)

    Returns
    -------
    smoother_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Pr(S_t=j | y_{1:T}), shape (n_discrete_states,),  M_{t | T}(j)
    smoother_backward_cond_prob : jax.Array, shape (n_discrete_states, n_discrete_states)
        Pr(S_t=j | S_{t+1}=k, y_{1:T}), shape (n_discrete_states, n_discrete_states), U^{j | k}_t
    joint_smoother_discrete_prob : jax.Array, shape (n_discrete_states, n_discrete_states)
        Pr(S_t=j, S_{t+1}=k | y_{1:T}), shape (n_discrete_states, n_discrete_states)
    smoother_forward_cond_prob : jax.Array, shape (n_discrete_states, n_discrete_states)
        Pr(S_{t+1}=k | S{t}=j, y_{1:T}), shape (n_discrete_states, n_discrete_states), W^{k | j}_t

    """
    # Discrete smoother prob
    # P(S_t = j, S_{t+1} = k | y_{1:T})
    smoother_backward_cond_prob = (
        filter_discrete_prob[:, None] * discrete_state_transition_matrix
    )
    smoother_backward_cond_prob = _divide_safe(
        smoother_backward_cond_prob, jnp.sum(smoother_backward_cond_prob, axis=0)
    )

    joint_smoother_discrete_prob = (
        smoother_backward_cond_prob * next_smoother_discrete_prob
    )
    # P(S_t = j | y_{1:T})
    smoother_discrete_state_prob = jnp.sum(joint_smoother_discrete_prob, axis=1)
    # P(S_{t+1} = k | S_t = j, y_{1:T})
    smoother_forward_cond_prob = _divide_safe(
        joint_smoother_discrete_prob, smoother_discrete_state_prob[:, None]
    )

    return (
        smoother_discrete_state_prob,
        smoother_backward_cond_prob,
        joint_smoother_discrete_prob,
        smoother_forward_cond_prob,
    )


def switching_kalman_smoother(
    filter_mean: jax.Array,
    filter_cov: jax.Array,
    filter_discrete_state_prob: jax.Array,
    last_filter_conditional_cont_mean: jax.Array,
    process_cov: jax.Array,
    continuous_transition_matrix: jax.Array,
    discrete_state_transition_matrix: jax.Array,
) -> tuple[
    jax.Array,  # Overall smoother mean
    jax.Array,  # Overall smoother covariance
    jax.Array,  # Smoother discrete state probabilities
    jax.Array,  # Smoother joint discrete state probabilities
    jax.Array,  # Overall smoother cross covariance
    jax.Array,  # State conditional smoother means
    jax.Array,  # State conditional smoother covariances
    jax.Array,  # Pair conditional smoother cross covariances
    jax.Array,  # Pair conditional smoother means
]:
    """Switching Kalman smoother for a linear Gaussian state space model with discrete states.

    Parameters
    ----------
    filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    filter_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    last_filter_conditional_cont_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    discrete_state_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)

    Returns
    -------
    overall_smoother_mean : jax.Array, shape (n_time, n_cont_states)
    overall_smoother_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states)
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    overall_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states)
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    pair_cond_smoother_cross_covs : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    pair_cond_smoother_means : jax.Array, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=j, S_{t+1}=k] - needed for correct M-step beta computation.
    """

    def _step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        args: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        """

        Parameters
        ----------
        carry : tuple
            next_smoother_mean : jax.Array, shape (n_cont_states, n_discrete_states)
            next_smoother_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
            next_discrete_state_prob : jax.Array, shape (n_discrete_states,)
            next_conditional_cont_means : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
        args : tuple
            state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
            state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
            filter_discrete_prob : jax.Array, shape (n_discrete_states,)

        Returns
        -------
        carry : tuple
            next_state_cond_smoother_mean : jax.Array, shape (n_cont_states, n_discrete_states)
            next_state_cond_smoother_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
            next_smoother_discrete_prob : jax.Array, shape (n_discrete_states,)
            next_pair_cond_smoother_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
        args : tuple
            state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
            state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
            filter_discrete_prob : jax.Array, shape (n_discrete_states,)
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
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_covs,
            pair_cond_smoother_mean,  # E[X_t | y_{1:T}, S_t=j, S_{t+1}=k]
        )

    init_carry = (
        filter_mean[-1],  # shape (n_cont_states, n_discrete_states)
        filter_cov[-1],  # shape (n_cont_states, n_cont_states, n_discrete_states)
        filter_discrete_state_prob[-1],  # shape (n_discrete_states,)
        last_filter_conditional_cont_mean,  # shape (n_cont_states, n_discrete_states, n_discrete_states)
    )

    _, (
        overall_smoother_mean,
        overall_smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        overall_smoother_cross_cov,
        state_cond_smoother_means,
        state_cond_smoother_covs,
        pair_cond_smoother_cross_covs,
        pair_cond_smoother_means,
    ) = jax.lax.scan(
        _step,
        init_carry,
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
    state_cond_smoother_means = jnp.concatenate(
        [state_cond_smoother_means, filter_mean[-1][None]], axis=0
    )
    state_cond_smoother_covs = jnp.concatenate(
        [state_cond_smoother_covs, filter_cov[-1][None]], axis=0
    )

    return (
        overall_smoother_mean,
        overall_smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        overall_smoother_cross_cov,
        state_cond_smoother_means,
        state_cond_smoother_covs,
        pair_cond_smoother_cross_covs,
        pair_cond_smoother_means,
    )


def weighted_sum_of_outer_products(
    x: jax.Array, y: jax.Array, weights: jax.Array
) -> jax.Array:
    """Compute the weighted outer sum of two arrays.
    Parameters
    ----------
    x : jax.Array, shape (n_time, x_dims, n_discrete_states)
        First array.
    y : jax.Array, shape (n_time, y_dims, n_discrete_states)
        Second array.
    weights : jax.Array, shape (n_time, n_discrete_states)
        Weights for the outer sum.

    Returns
    -------
    weighted_sum_of_outer_products: jax.Array, shape (x_dims, y_dims, n_discrete_states)
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
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    pair_cond_smoother_means: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Maximization step for the switching Kalman filter.

    Parameters
    ----------
    obs : jax.Array, shape (n_time, n_obs_dim)
        Observations.
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        smoother mean.
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        smoother covariance.
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        smoother discrete state probabilities.
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
        smoother joint discrete state probabilities.
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        smoother cross-covariance.
    pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses the exact pair-conditional
        means for beta computation. If None, uses the approximate factored form.

    Returns
    -------
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Transition matrix.
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Measurement matrix.
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Process covariance.
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Measurement covariance.
    init_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Initial mean of the continuous latent state.
    init_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Initial covariance of the continuous latent state.
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)
        Transition matrix for the discrete states.
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
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

    first_gamma = (
        state_cond_smoother_covs[0] * smoother_discrete_state_prob[0, None, None]
    ) + weighted_sum_of_outer_products(
        state_cond_smoother_means[:1],
        state_cond_smoother_means[:1],
        smoother_discrete_state_prob[:1],
    )
    gamma2 = gamma - first_gamma

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

    # Compute beta and gamma1 for transition matrix estimation
    # These use joint probability P(S_t=i, S_{t+1}=j) weighting
    if pair_cond_smoother_means is not None:
        # Use exact pair-conditional means: E[x_t | S_t=i, S_{t+1}=j]
        # This is the correct formulation that guarantees EM monotonicity

        # gamma1[a,b,j] = sum_{t,i} P(S_t=i, S_{t+1}=j) * E[x_t x_t^T | S_t=i, S_{t+1}=j]
        # Note: We need pair-conditional covariances too, but we approximate with
        # state-conditional covariances (the covariance term is less sensitive)
        gamma1 = jnp.einsum(
            "tij, tabi -> abj",
            smoother_joint_discrete_state_prob,
            state_cond_smoother_covs[:-1],
        ) + jnp.einsum(
            "tij, taij, tbij -> abj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_means,  # E[x_t | S_t=i, S_{t+1}=j]
            pair_cond_smoother_means,  # E[x_t | S_t=i, S_{t+1}=j]
        )

        # beta[c,d,j] = sum_{t,i} P(S_t=i, S_{t+1}=j) * E[x_{t+1} x_t^T | S_t=i, S_{t+1}=j]
        # = sum_{t,i} P(S_t=i, S_{t+1}=j) * (Cov[x_{t+1}, x_t | ...] + E[x_{t+1}|...] E[x_t|...]^T)
        beta = jnp.einsum(
            "tij,tdcij->cdj",
            smoother_joint_discrete_state_prob,  # P(S_t=i, S_{t+1}=j)
            pair_cond_smoother_cross_cov,  # Cov[x_{t+1}, x_t | S_t=i, S_{t+1}=j]
        )
        # For the mean term, we need E[x_{t+1} | S_t=i, S_{t+1}=j] which we approximate
        # with E[x_{t+1} | S_{t+1}=j] (state_cond_smoother_means[1:, :, j])
        # beta[c,d,j] = sum_{t,i} P(S_t=i, S_{t+1}=j) * E[x_{t+1}|S_{t+1}=j]_c * E[x_t|S_t=i,S_{t+1}=j]_d
        beta += jnp.einsum(
            "tdij,tcj,tij->cdj",
            pair_cond_smoother_means,  # E[x_t | S_t=i, S_{t+1}=j], shape (T-1, d, i, j)
            state_cond_smoother_means[1:],  # E[x_{t+1} | S_{t+1}=j], shape (T-1, c, j)
            smoother_joint_discrete_state_prob,  # P(S_t=i, S_{t+1}=j), shape (T-1, i, j)
        )
    else:
        # Approximate factored form (original implementation)
        # gamma1[a,b,j] = sum_{t,i} P(S_t=i, S_{t+1}=j) * E[x_t x_t^T | S_t=i]
        gamma1 = jnp.einsum(
            "tij, tabi -> abj",
            smoother_joint_discrete_state_prob,
            state_cond_smoother_covs[:-1],
        ) + jnp.einsum(
            "tij, tai, tbi -> abj",
            smoother_joint_discrete_state_prob,
            state_cond_smoother_means[:-1],
            state_cond_smoother_means[:-1],
        )

        beta = jnp.einsum(
            "tij,tdcij->cdj",
            smoother_joint_discrete_state_prob,  # P(S_k=i, S_{k+1}=j)
            pair_cond_smoother_cross_cov,  # E[x_k x_{k+1}^T | S_k=i, S_{k+1}=j]
        )
        beta += jnp.einsum(
            "tdi,tcj,tij->cdj",
            state_cond_smoother_means[:-1],  # m_t^i (Shape T-1, c, i)
            state_cond_smoother_means[1:],  # m_{t+1}^j (Shape T-1, d, j)
            smoother_joint_discrete_state_prob,  # P(S_t=i, S_{t+1}=j) (Shape T-1, i, j)
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
    discrete_state_transition = _divide_safe(
        smoother_joint_discrete_state_prob.sum(axis=0),
        smoother_discrete_state_prob[:-1].sum(axis=0)[:, None],
    )
    # Ensure rows sum to 1
    discrete_state_transition = _divide_safe(
        discrete_state_transition,
        jnp.sum(discrete_state_transition, axis=1, keepdims=True),
    )

    # Ensure the initial discrete state probabilities sum to 1
    init_discrete_state_prob = smoother_discrete_state_prob[0]
    init_discrete_state_prob = _divide_safe(
        init_discrete_state_prob, jnp.sum(init_discrete_state_prob)
    )

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


def compute_expected_complete_log_likelihood(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
    discrete_transition_matrix: jax.Array,
) -> float:
    """Compute the expected complete-data log-likelihood E_q[log p(y, x, s | θ)].

    This is the Q-function that the EM algorithm maximizes. For variational EM,
    the ELBO = Q - entropy(q), and the Q-function should increase (or stay same)
    after each M-step.

    Parameters
    ----------
    obs : jax.Array, shape (n_time, n_obs_dim)
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)

    Returns
    -------
    expected_complete_ll : float
        E_q[log p(y, x, s | θ)]
    """
    n_time = obs.shape[0]
    n_discrete_states = smoother_discrete_state_prob.shape[1]
    n_cont_states = state_cond_smoother_means.shape[1]

    # 1. E_q[log p(s_1)] - initial discrete state
    log_init_discrete = jnp.sum(
        smoother_discrete_state_prob[0] * _safe_log(init_discrete_state_prob)
    )

    # 2. E_q[log p(x_1 | s_1)] - initial continuous state
    log_init_cont = 0.0
    for j in range(n_discrete_states):
        # E_q[log N(x_1; μ_0^j, Σ_0^j) | s_1=j]
        # = -0.5 * (log|Σ_0^j| + tr(Σ_0^j^{-1} E_q[(x_1 - μ_0^j)(x_1 - μ_0^j)^T | s_1=j]))
        mean_j = init_state_cond_mean[:, j]
        cov_j = init_state_cond_cov[:, :, j]

        # E_q[(x_1 - μ_0^j)(x_1 - μ_0^j)^T | s_1=j]
        smoother_mean_j = state_cond_smoother_means[0, :, j]
        smoother_cov_j = state_cond_smoother_covs[0, :, :, j]
        diff = smoother_mean_j - mean_j
        expected_outer = smoother_cov_j + jnp.outer(diff, diff)

        log_det = jnp.linalg.slogdet(cov_j)[1]
        trace_term = jnp.trace(psd_solve(cov_j, expected_outer))
        log_prob_j = -0.5 * (n_cont_states * jnp.log(2 * jnp.pi) + log_det + trace_term)
        log_init_cont += smoother_discrete_state_prob[0, j] * log_prob_j

    # 3. E_q[sum_t log p(s_t | s_{t-1})] - discrete state transitions
    log_discrete_trans = jnp.sum(
        smoother_joint_discrete_state_prob * _safe_log(discrete_transition_matrix)
    )

    # 4. E_q[sum_t log p(x_t | x_{t-1}, s_t)] - continuous state transitions
    log_cont_trans = 0.0
    for j in range(n_discrete_states):
        A_j = continuous_transition_matrix[:, :, j]
        Q_j = process_cov[:, :, j]
        log_det_Q = jnp.linalg.slogdet(Q_j)[1]

        for t in range(n_time - 1):
            # Sum over source states i weighted by P(s_t=i, s_{t+1}=j | y_{1:T})
            for i in range(n_discrete_states):
                weight = smoother_joint_discrete_state_prob[t, i, j]
                if weight < 1e-10:
                    continue

                # E_q[(x_{t+1} - A_j x_t)(x_{t+1} - A_j x_t)^T | s_t=i, s_{t+1}=j]
                # Need: E[x_{t+1} x_{t+1}^T], E[x_{t+1} x_t^T], E[x_t x_t^T]
                m_t_i = state_cond_smoother_means[t, :, i]
                m_t1_j = state_cond_smoother_means[t + 1, :, j]
                V_t_i = state_cond_smoother_covs[t, :, :, i]
                V_t1_j = state_cond_smoother_covs[t + 1, :, :, j]
                cross_cov_ij = pair_cond_smoother_cross_cov[t, :, :, i, j]

                # E[x_{t+1} x_{t+1}^T | ...]
                E_xt1_xt1 = V_t1_j + jnp.outer(m_t1_j, m_t1_j)
                # E[x_t x_t^T | ...]
                E_xt_xt = V_t_i + jnp.outer(m_t_i, m_t_i)
                # E[x_{t+1} x_t^T | ...]
                E_xt1_xt = cross_cov_ij + jnp.outer(m_t1_j, m_t_i)

                # E[(x_{t+1} - A x_t)(x_{t+1} - A x_t)^T]
                # = E[x_{t+1} x_{t+1}^T] - A E[x_t x_{t+1}^T] - E[x_{t+1} x_t^T] A^T + A E[x_t x_t^T] A^T
                expected_residual = (
                    E_xt1_xt1
                    - A_j @ E_xt1_xt.T
                    - E_xt1_xt @ A_j.T
                    + A_j @ E_xt_xt @ A_j.T
                )

                trace_term = jnp.trace(psd_solve(Q_j, expected_residual))
                log_prob = -0.5 * (
                    n_cont_states * jnp.log(2 * jnp.pi) + log_det_Q + trace_term
                )
                log_cont_trans += weight * log_prob

    # 5. E_q[sum_t log p(y_t | x_t, s_t)] - observations
    log_obs = 0.0
    for j in range(n_discrete_states):
        H_j = measurement_matrix[:, :, j]
        R_j = measurement_cov[:, :, j]
        log_det_R = jnp.linalg.slogdet(R_j)[1]
        n_obs = obs.shape[1]

        for t in range(n_time):
            weight = smoother_discrete_state_prob[t, j]
            if weight < 1e-10:
                continue

            m_t_j = state_cond_smoother_means[t, :, j]
            V_t_j = state_cond_smoother_covs[t, :, :, j]

            # E[(y_t - H x_t)(y_t - H x_t)^T | s_t=j]
            pred_mean = H_j @ m_t_j
            diff = obs[t] - pred_mean
            # E[x_t x_t^T | s_t=j]
            E_xt_xt = V_t_j + jnp.outer(m_t_j, m_t_j)
            # E[(y - Hx)(y - Hx)^T] = (y - H m)(y - H m)^T + H V H^T
            expected_residual = jnp.outer(diff, diff) + H_j @ V_t_j @ H_j.T

            trace_term = jnp.trace(psd_solve(R_j, expected_residual))
            log_prob = -0.5 * (n_obs * jnp.log(2 * jnp.pi) + log_det_R + trace_term)
            log_obs += weight * log_prob

    return log_init_discrete + log_init_cont + log_discrete_trans + log_cont_trans + log_obs


def compute_posterior_entropy(
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    state_cond_smoother_covs: jax.Array,
) -> float:
    """Compute the entropy of the approximate posterior H(q).

    For the switching Kalman filter with mixture collapse approximation:
    H(q) = H(q(s)) + E_q(s)[H(q(x|s))]

    Parameters
    ----------
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)

    Returns
    -------
    entropy : float
        H(q(x, s))
    """
    n_time = smoother_discrete_state_prob.shape[0]
    n_discrete_states = smoother_discrete_state_prob.shape[1]
    n_cont_states = state_cond_smoother_covs.shape[1]

    # 1. Entropy of discrete state sequence
    # H(q(s)) = -sum_t E_q[log q(s_t | s_{t-1})]
    # For t=1: -sum_j q(s_1=j) log q(s_1=j)
    discrete_entropy = -jnp.sum(
        smoother_discrete_state_prob[0]
        * _safe_log(smoother_discrete_state_prob[0])
    )

    # For t>1: -sum_{t,i,j} q(s_{t-1}=i, s_t=j) log q(s_t=j | s_{t-1}=i)
    # q(s_t=j | s_{t-1}=i) = q(s_{t-1}=i, s_t=j) / q(s_{t-1}=i)
    for t in range(n_time - 1):
        marginal_prev = smoother_discrete_state_prob[t]
        joint = smoother_joint_discrete_state_prob[t]
        cond = _divide_safe(joint, marginal_prev[:, None])
        discrete_entropy -= jnp.sum(joint * _safe_log(cond))

    # 2. Entropy of continuous states given discrete states
    # H(q(x|s)) = sum_t sum_j q(s_t=j) * H(q(x_t | s_t=j))
    # For Gaussian: H(N(μ, Σ)) = 0.5 * (k + k*log(2π) + log|Σ|)
    cont_entropy = 0.0
    for j in range(n_discrete_states):
        for t in range(n_time):
            weight = smoother_discrete_state_prob[t, j]
            if weight < 1e-10:
                continue
            cov_j = state_cond_smoother_covs[t, :, :, j]
            log_det = jnp.linalg.slogdet(cov_j)[1]
            gaussian_entropy = 0.5 * (
                n_cont_states * (1 + jnp.log(2 * jnp.pi)) + log_det
            )
            cont_entropy += weight * gaussian_entropy

    return discrete_entropy + cont_entropy


def compute_elbo(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
    discrete_transition_matrix: jax.Array,
) -> float:
    """Compute the Evidence Lower Bound (ELBO) for the switching Kalman filter.

    ELBO = E_q[log p(y, x, s | θ)] - E_q[log q(x, s)]
         = E_q[log p(y, x, s | θ)] + H(q)

    For variational EM with the GPB1/IMM approximation, the ELBO is guaranteed
    to increase (or stay the same) after each EM iteration.

    Parameters
    ----------
    obs : jax.Array, shape (n_time, n_obs_dim)
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    smoother_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
    discrete_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)

    Returns
    -------
    elbo : float
        The evidence lower bound
    """
    expected_ll = compute_expected_complete_log_likelihood(
        obs=obs,
        state_cond_smoother_means=state_cond_smoother_means,
        state_cond_smoother_covs=state_cond_smoother_covs,
        smoother_discrete_state_prob=smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
        pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
        init_state_cond_mean=init_state_cond_mean,
        init_state_cond_cov=init_state_cond_cov,
        init_discrete_state_prob=init_discrete_state_prob,
        continuous_transition_matrix=continuous_transition_matrix,
        process_cov=process_cov,
        measurement_matrix=measurement_matrix,
        measurement_cov=measurement_cov,
        discrete_transition_matrix=discrete_transition_matrix,
    )

    entropy = compute_posterior_entropy(
        smoother_discrete_state_prob=smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
        state_cond_smoother_covs=state_cond_smoother_covs,
    )

    return expected_ll + entropy


def compute_transition_sufficient_stats(
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    pair_cond_smoother_means: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute sufficient statistics for transition matrix estimation.

    Parameters
    ----------
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses exact pair-conditional means.

    Returns
    -------
    gamma1 : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        E[x_t x_t^T] weighted by joint probability P(S_t=i, S_{t+1}=j), summed over i.
    beta : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        E[x_t x_{t+1}^T] weighted by joint probability.
    """
    if pair_cond_smoother_means is not None:
        # Use exact pair-conditional means
        # gamma1[a,b,j] = sum_{t,i} P(S_t=i, S_{t+1}=j) * E[x_t x_t^T | S_t=i, S_{t+1}=j]
        gamma1 = jnp.einsum(
            "tij, tabi -> abj",
            smoother_joint_discrete_state_prob,
            state_cond_smoother_covs[:-1],
        ) + jnp.einsum(
            "tij, taij, tbij -> abj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_means,
            pair_cond_smoother_means,
        )

        # beta = E[x_t x_{t+1}^T] weighted by joint probability
        beta = jnp.einsum(
            "tij,tdcij->cdj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov,
        )
        # beta[c,d,j] = sum_{t,i} P(S_t=i, S_{t+1}=j) * E[x_{t+1}|S_{t+1}=j]_c * E[x_t|S_t=i,S_{t+1}=j]_d
        beta += jnp.einsum(
            "tdij,tcj,tij->cdj",
            pair_cond_smoother_means,  # E[x_t | S_t=i, S_{t+1}=j], shape (T-1, d, i, j)
            state_cond_smoother_means[1:],  # E[x_{t+1} | S_{t+1}=j], shape (T-1, c, j)
            smoother_joint_discrete_state_prob,  # P(S_t=i, S_{t+1}=j), shape (T-1, i, j)
        )
    else:
        # Approximate factored form (original implementation)
        # gamma1[a,b,j] = sum_{t,i} P(S_t=i, S_{t+1}=j) * E[x_t x_t^T | S_t=i]
        gamma1 = jnp.einsum(
            "tij, tabi -> abj",
            smoother_joint_discrete_state_prob,
            state_cond_smoother_covs[:-1],
        ) + jnp.einsum(
            "tij, tai, tbi -> abj",
            smoother_joint_discrete_state_prob,
            state_cond_smoother_means[:-1],
            state_cond_smoother_means[:-1],
        )

        # beta = E[x_t x_{t+1}^T] weighted by joint probability
        beta = jnp.einsum(
            "tij,tdcij->cdj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov,
        )
        beta += jnp.einsum(
            "tdi,tcj,tij->cdj",
            state_cond_smoother_means[:-1],
            state_cond_smoother_means[1:],
            smoother_joint_discrete_state_prob,
        )

    return gamma1, beta


def compute_transition_q_function(
    A: jax.Array,
    gamma1: jax.Array,
    beta: jax.Array,
) -> float:
    """Compute the Q-function contribution from transition matrix.

    The Q-function for A (ignoring constants and Q covariance) is:
        Q(A) ∝ -0.5 * tr(A^T A gamma1 - 2 A^T beta)

    We return the negative Q-function since we want to minimize.

    Parameters
    ----------
    A : jax.Array, shape (n_cont, n_cont)
        Transition matrix.
    gamma1 : jax.Array, shape (n_cont, n_cont)
        E[x_t x_t^T] summed over time, weighted by joint discrete state probs.
    beta : jax.Array, shape (n_cont, n_cont)
        E[x_t x_{t+1}^T] summed over time, weighted by joint discrete state probs.

    Returns
    -------
    float
        Negative Q-function value (to be minimized).
    """
    # We minimize: 0.5 * tr(A^T A gamma1) - tr(A^T beta)
    # Which is equivalent to maximizing the actual Q-function
    return 0.5 * jnp.trace(A.T @ A @ gamma1) - jnp.trace(A.T @ beta)


def compute_transition_q_from_params(
    damping: jax.Array,
    freq: jax.Array,
    coupling_strength: jax.Array,
    phase_diff: jax.Array,
    sampling_freq: float,
    gamma1: jax.Array,
    beta: jax.Array,
) -> float:
    """Compute Q-function from oscillator parameters.

    This is the objective to minimize in the reparameterized M-step.

    Parameters
    ----------
    damping : jax.Array, shape (n_oscillators,)
        Damping coefficients.
    freq : jax.Array, shape (n_oscillators,)
        Frequencies in Hz.
    coupling_strength : jax.Array, shape (n_oscillators, n_oscillators)
        Coupling strengths (0 on diagonal).
    phase_diff : jax.Array, shape (n_oscillators, n_oscillators)
        Phase differences (0 on diagonal).
    sampling_freq : float
        Sampling frequency.
    gamma1 : jax.Array, shape (n_cont, n_cont)
        Sufficient statistic.
    beta : jax.Array, shape (n_cont, n_cont)
        Sufficient statistic.

    Returns
    -------
    float
        Negative Q-function value (to be minimized).
    """
    from state_space_practice.oscillator_utils import (
        construct_directed_influence_transition_matrix,
    )

    A = construct_directed_influence_transition_matrix(
        freqs=freq,
        damping_coeffs=damping,
        coupling_strengths=coupling_strength,
        phase_diffs=phase_diff,
        sampling_freq=sampling_freq,
    )
    return compute_transition_q_function(A, gamma1, beta)


def optimize_dim_transition_params(
    gamma1: jax.Array,
    beta: jax.Array,
    init_params: dict,
    sampling_freq: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Optimize oscillator parameters to maximize Q-function.

    Uses JAX autodiff + BFGS optimizer.

    Parameters
    ----------
    gamma1 : jax.Array, shape (n_cont, n_cont)
        Sufficient statistic E[x_t x_t^T].
    beta : jax.Array, shape (n_cont, n_cont)
        Sufficient statistic E[x_t x_{t+1}^T].
    init_params : dict
        Initial parameter values (damping, freq, coupling_strength, phase_diff).
    sampling_freq : float
        Sampling frequency.
    max_iter : int
        Maximum optimization iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        Optimized parameters with keys: damping, freq, coupling_strength, phase_diff.
    """
    from jax.scipy.optimize import minimize

    n_osc = len(init_params["damping"])

    def pack_params(params: dict) -> jax.Array:
        """Flatten params to 1D array for optimizer."""
        return jnp.concatenate(
            [
                params["damping"],
                params["freq"],
                params["coupling_strength"].ravel(),
                params["phase_diff"].ravel(),
            ]
        )

    def unpack_params(flat: jax.Array) -> dict:
        """Unflatten 1D array to param dict."""
        idx = 0
        damping = flat[idx : idx + n_osc]
        idx += n_osc
        freq = flat[idx : idx + n_osc]
        idx += n_osc
        coupling = flat[idx : idx + n_osc * n_osc].reshape(n_osc, n_osc)
        idx += n_osc * n_osc
        phase = flat[idx : idx + n_osc * n_osc].reshape(n_osc, n_osc)
        return {
            "damping": damping,
            "freq": freq,
            "coupling_strength": coupling,
            "phase_diff": phase,
        }

    def loss(flat_params: jax.Array) -> float:
        params = unpack_params(flat_params)
        return compute_transition_q_from_params(
            damping=params["damping"],
            freq=params["freq"],
            coupling_strength=params["coupling_strength"],
            phase_diff=params["phase_diff"],
            sampling_freq=sampling_freq,
            gamma1=gamma1,
            beta=beta,
        )

    # Run optimizer
    init_flat = pack_params(init_params)
    result = minimize(loss, init_flat, method="BFGS", tol=tol)

    return unpack_params(result.x)