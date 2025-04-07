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
    """Collapse a mixture of Gaussians.

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
collapse_gaussian_mixture_per_previous_discrete_state = jax.vmap(
    collapse_gaussian_mixture, in_axes=(1, 2, 0), out_axes=(-1, -1)
)
collapse_cross_gaussian_mixture_across_states = jax.vmap(
    collapse_gaussian_mixture_cross_covariance, in_axes=(2, 2, 3, 1), out_axes=(1, 1, 2)
)


def _update_discrete_state_probabilities(
    conditional_marginal_likelihood: jnp.ndarray,
    discrete_transition_matrix: jnp.ndarray,
    discrete_state_prob_prev: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """Update the discrete state probabilities using the discrete transition matrix.

    Parameters
    ----------
    conditional_marginal_likelihood : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
    discrete_transition_matrix : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
    discrete_state_prob_prev : jnp.ndarray, shape (n_discrete_states,)

    Returns
    -------
    discrete_state_prob : jnp.ndarray, shape (n_discrete_states,)
        Updated discrete state probabilities, P(S_t)
    discrete_state_weights : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
        Mixing weights for the discrete states, P(S_{t-1} | S_t)
    predictive_likelihood_term_sum : float
        Scaled predictive likelihood sum
    """
    # joint discrete state prob between time steps
    # p(S_t, S_{t-1}) = p(y_t | S_t) * p(S_t | S_{t-1}) * p(S_{t-1})
    joint_discrete_state_prob = (
        conditional_marginal_likelihood
        * discrete_transition_matrix
        * discrete_state_prob_prev[:, None]
    )
    predictive_likelihood_term_sum = jnp.sum(joint_discrete_state_prob)
    joint_discrete_state_prob /= jnp.sum(joint_discrete_state_prob)

    # p(S_t)
    discrete_state_prob = jnp.sum(joint_discrete_state_prob, axis=0)
    # p(S_{t-1} | S_t) = p(S_t, S_{t-1}) / p(S_t)
    discrete_state_weights = joint_discrete_state_prob / discrete_state_prob[None, :]

    return discrete_state_prob, discrete_state_weights, predictive_likelihood_term_sum


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
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    init_discrete_state_prob: jnp.ndarray,
    obs: jnp.ndarray,
    discrete_transition_matrix: jnp.ndarray,
    continuous_transition_matrix: jnp.ndarray,
    process_cov: jnp.ndarray,
    measurement_matrix: jnp.ndarray,
    measurement_cov: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    float,
]:
    """Switching Kalman filter for a linear Gaussian state space model with discrete states.

    Parameters
    ----------
    init_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
        Initial value of the continuous latent state $x_1$
    init_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
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
        Process noise covariance matrix. $\Sigma$
    measurement_matrix : jnp.ndarray, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Map observations to the continuous states $H$
    measurement_cov : jnp.ndarray, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Measurement variance. $R$

    Returns
    -------
    filter_mean : jnp.ndarray, shape (n_time, n_cont_states, n_discrete_states)
        Filtered mean of the continuous latent state
    filter_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        Filtered covariance of the continuous latent state
    filter_discrete_state_prob : jnp.ndarray, shape (n_time, n_discrete_states)
        Filtered probability of the discrete states
    last_filter_conditional_cont_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states, n_discrete_states)
        Filtered conditional mean of the continuous latent state
    marginal_log_likelihood : float
        Marginal log likelihood of the observations

    """

    def _step(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], obs_t: jnp.ndarray
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ]:
        """One step of the switching Kalman filter.

        Parameters
        ----------
        carry : tuple
            mean_prev : jnp.ndarray, shape (n_cont_states, n_discrete_states)
                Previous state mean.
            cov_prev : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
                Previous state covariance.
            discrete_state_prob_prev : jnp.ndarray, shape (n_discrete_states,)
                Previous discrete state probabilities
            marginal_log_likelihood : float
                Previous marginal log likelihood
        obs_t : jnp.ndarray, shape (n_obs_dim,)
            Observation at time t

        Returns
        -------
        carry : tuple
            mean_prev : jnp.ndarray, shape (n_cont_states, n_discrete_states)
                Posterior state mean.
            cov_prev : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
                Posterior state covariance.
            discrete_state_prob_prev : jnp.ndarray, shape (n_discrete_states,)
                Posterior discrete state probabilities
            marginal_log_likelihood : float
                Posterior marginal log likelihood
        stack : tuple
            collapsed_means : jnp.ndarray, shape (n_cont_states, n_discrete_states)
                Posterior state mean.
            collapsed_covs : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
                Posterior state covariance.
            discrete_state_prob : jnp.ndarray, shape (n_discrete_states,)
                Posterior discrete state probabilities
            conditional_cont_means : jnp.ndarray, shape (n_cont_states, n_discrete_states, n_discrete_states)
                Conditional means of the continuous latent state
        """
        mean_prev, cov_prev, discrete_state_prob_prev, marginal_log_likelihood = carry

        # Kalman update for each pair of discrete states
        # P(x_t | y_{1:t}, S_t = j, S_{t-1} = i)
        # vmap twice over the discrete states
        (
            conditional_cont_means,
            conditional_cont_covs,
            conditional_marginal_log_likelihood,
        ) = _kalman_filter_update_per_discrete_state_pair(
            mean_prev,
            cov_prev,
            obs_t,
            continuous_transition_matrix,
            process_cov,
            measurement_matrix,
            measurement_cov,
        )

        # Make sure the likelihood is normalized to max 1 for numerical stability
        conditional_marginal_likelihood, ll_max = _scale_likelihood(
            conditional_marginal_log_likelihood
        )

        discrete_state_prob, discrete_state_weights, predictive_likelihood_term_sum = (
            _update_discrete_state_probabilities(
                conditional_marginal_likelihood,  # shape (n_discrete_states, n_discrete_states)
                discrete_transition_matrix,  # shape (n_discrete_states, n_discrete_states)
                discrete_state_prob_prev,  # shape (n_discrete_states,)
            )
        )

        marginal_log_likelihood += ll_max + jnp.log(predictive_likelihood_term_sum)

        # Collapse `n_discrete_states` x `n_discrete_states` Gaussians
        # to `n_discrete_states` Gaussians
        collapsed_means, collapsed_covs = collapse_gaussian_mixture_per_discrete_state(
            conditional_cont_means,  # shape (n_cont_states, n_discrete_states, n_discrete_states)
            conditional_cont_covs,  # shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
            discrete_state_weights,  # shape (n_discrete_states, n_discrete_states)
        )

        return (
            collapsed_means,
            collapsed_covs,
            discrete_state_prob,
            marginal_log_likelihood,
        ), (
            collapsed_means,
            collapsed_covs,
            discrete_state_prob,
            conditional_cont_means,
        )

    marginal_log_likelihood = 0.0
    (_, _, _, marginal_log_likelihood), (
        filter_mean,
        filter_cov,
        filter_discrete_state_prob,
        filter_conditional_cont_mean,
    ) = jax.lax.scan(
        _step,
        (init_mean, init_cov, init_discrete_state_prob, marginal_log_likelihood),
        obs,
    )

    return (
        filter_mean,
        filter_cov,
        filter_discrete_state_prob,
        filter_conditional_cont_mean[-1],
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
    filter_discrete_state_prob,
    discrete_state_transition_matrix,
    next_discrete_state_prob,
):
    # Discrete smoother prob
    # P(S_t = j, S_{t+1} = k | y_{1:T})
    smoother_conditional_discrete_state_prob = (
        filter_discrete_state_prob[:, None] * discrete_state_transition_matrix
    )
    smoother_conditional_discrete_state_prob /= jnp.sum(
        smoother_conditional_discrete_state_prob, axis=0
    )
    smoother_joint_discrete_state_prob = (
        smoother_conditional_discrete_state_prob * next_discrete_state_prob
    )
    # P(S_t = j | y_{1:T})
    smoother_discrete_state_prob = jnp.sum(smoother_joint_discrete_state_prob, axis=1)
    # P(S_{t+1} = k | S_t = j, y_{1:T})
    discrete_state_weights = (
        smoother_joint_discrete_state_prob / smoother_discrete_state_prob[:, None]
    )

    return (
        smoother_discrete_state_prob,
        smoother_conditional_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        discrete_state_weights,
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
    smoother_mean : jnp.ndarray, shape (n_time, n_cont_states)
    smoother_cov : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states)
    smoother_discrete_state_prob : jnp.ndarray, shape (n_time, n_discrete_states)
    smoother_joint_discrete_state_prob : jnp.ndarray, shape (n_time - 1, n_discrete_states, n_discrete_states)
    smoother_cross_cov : jnp.ndarray, shape (n_time - 1, n_cont_states, n_cont_states)
    """

    def _step(carry, args):
        """
        1. Smooth for each discrete state
            - x^k_{t+1 | T}: next state-conditional smoother mean, E[X_{t+1} | y_{1:T}, S_{t+1}=k]
            - V^k_{t+1 | T}: next state-conditional smoother covariance, Cov[X_{t+1} | y_{1:T}, S_{t+1}=k]
            - x^j_{t | t}: state-conditional filter mean, E[X_t | y_{1:t}, S_t=j]
            - V^j_{t | t}: state-conditional filter covariance, Cov[X_t | y_{1:t}, S_t=j]
            - V^k_{t+1 | t+1}: next state-conditional filter covariance
            - V^{jk}_{t+1,t | t+ 1}: next pair-conditional cross covariance, Cov[X_{t+1}, X_t | y_{1:t}, S_{t+1}=j, S_{t}=k]
            - F_k : transition matrix, E[X_{t+1} | X_t, S=k]
            - Q_k : process covariance

            - x^{jk}_{t | T}: pair-conditional smoother mean, E[X_t | y_{1:T}, S_t=j, S_{t+1}=k]
            - V^{jk}_{t | T}: pair-conditional smoother covariance, Cov[X_t | y_{1:T}, S_t=j, S_{t+1}=k]
            - V^{jk}_{t+1, t | T}: pair-conditional smoother cross covariance
        2. Compute discrete state intermediates
            - M_{t | t}(j): filter discrete prob, Pr(S_t = j | y_{1_t})
            - Z(j, k): discrete transition matrix, Pr(S_t = j | S_{t-1} = k)

            - U^{j | k}_t: Pr(S_t = j | S_{t+1} = k, y_{1:T}): smoother backward conditional probability

            - M_{t+1 | T}(k): next smoother discrete prob, Pr(S_{t+1} = k | y_{1:T})
            - M_{t, t+1|T}(j, k): joint smoother discrete prob, Pr(S_t = j, S_{t+1} = k | y_{1:T})

            - M_{t | T}(j): smoother discrete prob, Pr(S_t = j | y_{1:T})

            - W^{k | j}_t: smoother forward conditional probability, Pr(S_{t+1} = k | S{t} = j, y_{1:T})

        3. Collapse conditional mean and covariance (n_states x n_states -> n_states)
            - x^{jk}_{t | T}: pair-conditional smoother mean, E[X_t | y_{1:T}, S_t=j, S_{t+1}=k]
            - V^{jk}_{t | T}: pair-conditional smoother covariance, Cov[X_t | y_{1:T}, S_t=j, S_{t+1}=k]
            - W^{k | j}_t: smoother forward conditional probability, Pr(S_{t+1} = k | S{t} = j, y_{1:T})

            - x^j_{t|T}: state-conditional smoother mean, E[X_t | y_{1:T}, S_{t}=j]
            - V^j_{t|T}: state-conditional smoother covariance, Cov[X_t | y_{1:T}, S_{t} = j]
        4. Collapse to single mean and covariance (n_states -> 1)
            - x_{t|T}: overall smoother mean, E[X_t | y_{1:T}]
            - V_{t|T}: overall smoother covariance, Cov[X_t | y_{1:T}]
        5. Collapse cross covariance
            - x^k_{t+1 | T}: next state-conditional smoother mean

            - x^{jk}_{t+1 | T}: next pair-conditional smoother mean, E[X_{t+1} | y_{1:T}, S_{t+1}=k, S_{t}=j], why is this approx equal to above and why not just use the former

            - x^{jk}_{t+1 | T}: next pair-conditional smoother mean
            - x^{jk}_{t | T}: pair-conditional smoother mean
            - V^{jk}_{t+1, t | t+ 1}: next pair-conditional cross covariance, Cov[X_{t+1}, X_t | y_{1:t}, S_{t+1}=j, S_{t}=k]
            - U^{j | k}_t: Pr(S_t = j | S_{t+1} = k, y_{1:T}): smoother backward conditional probability

            - V^k_{t+1, t | T}: state-conditional smoother cross covariance

            - x^{jk}_{t | T}: pair-conditional smoother mean
            - U^{j | k}_t: Pr(S_t = j | S_{t+1} = k, y_{1:T}): smoother backward conditional probability

            - x^k_{t | T}: state-conditional smoother mean, E[X_t | y_{1:T}, S_{t+1}]

            - x^{jk}_{t+1 | T}: next pair-conditional smoother mean
            - x^k_{t | T}: state-conditional smoother mean, E[X_t | y_{1:T}, S_{t+1}]
            - V^k_{t+1, t | T}: state-conditional smoother cross covariance
            - M^k_{t+1 | T}: next smoother discrete prob, Pr(S_{t+1} = k | y_{1:T})

            - V_{t+1, t | T}: Overall smoother cross covariance

        Parameters
        ----------
        carry : tuple
            next_smoother_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
            next_smoother_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
            next_discrete_state_prob : jnp.ndarray, shape (n_discrete_states,)
            next_conditional_cont_means : jnp.ndarray, shape (n_cont_states, n_discrete_states, n_discrete_states)
        args : _type_
            filter_mean_t : jnp.ndarray, shape (n_cont_states, n_discrete_states)
            filter_cov_t : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
            filter_discrete_state_prob_t : jnp.ndarray, shape (n_discrete_states,)

        Returns
        -------
        carry : tuple
            smoother_mean : jnp.ndarray, shape (n_cont_states, n_discrete_states)
            smoother_cov : jnp.ndarray, shape (n_cont_states, n_cont_states, n_discrete_states)
            smoother_discrete_state_prob : jnp.ndarray, shape (n_discrete_states,)
            conditional_cont_means : jnp.ndarray, shape (n_cont_states, n_discrete_states, n_discrete_states)
        args : tuple
            single_collapsed_means : jnp.ndarray, shape (n_cont_states,)
            single_collapsed_covs : jnp.ndarray, shape (n_cont_states, n_cont_states)
            smoother_discrete_state_prob : jnp.ndarray, shape (n_discrete_states,)
            smoother_joint_discrete_state_prob : jnp.ndarray, shape (n_discrete_states, n_discrete_states)
            smoother_cross_cov : jnp.ndarray, shape (n_cont_states, n_cont_states)
        """
        (
            next_smoother_mean,
            next_smoother_cov,
            next_discrete_state_prob,
            next_conditional_cont_means,
        ) = carry

        filter_mean_t, filter_cov_t, filter_discrete_state_prob_t = args
        # filter_mean_t, shape (n_cont_states, n_discrete_states)
        # filter_cov_t, shape (n_cont_states, n_cont_states, n_discrete_states)
        # filter_discrete_state_prob_t, shape (n_discrete_states,)

        conditional_cont_means, conditional_cont_covs, conditional_cont_cross_covs = (
            _kalman_smoother_update_per_discrete_state_pair(
                next_smoother_mean,  # E[X_{t+1} | y_{1:T}], shape (n_cont_states, n_discrete_states)
                next_smoother_cov,  # Cov[X_{t+1} | y_{1:T}], shape (n_cont_states, n_cont_states, n_discrete_states)
                filter_mean_t,  # E[X_t | y_{1:T}], shape (n_cont_states, n_discrete_states)
                filter_cov_t,  # Cov[X_t | y_{1:T}], shape (n_cont_states, n_cont_states, n_discrete_states)
                process_cov,  # Cov[X_{t+1} | X_t], shape (n_cont_states, n_cont_states, n_discrete_states)
                continuous_transition_matrix,  # E[X_{t+1} | X_t], shape (n_cont_states, n_cont_states, n_discrete_states)
            )
        )
        # conditional_cont_means, shape (n_cont_states, n_discrete_states, n_discrete_states)
        # conditional_cont_covs, shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        # conditional_cont_cross_covs, shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)

        (
            smoother_discrete_state_prob,  # p(S_t | y_{1:T})
            smoother_conditional_discrete_state_prob,  # p(S_{t+1} | S_t, y_{1:T})
            smoother_joint_discrete_state_prob,  # p(S_t, S_{t+1} | y_{1:T})
            discrete_state_weights,  # p(S_{t+1} | S_t, y_{1:T})
        ) = _update_smoother_discrete_probabilities(
            filter_discrete_state_prob_t,  # p(S_t | y_{1:t})
            discrete_state_transition_matrix,  # p(S_{t+1} | S_t)
            next_discrete_state_prob,  # p(S_{t+1})
        )

        # smoother_discrete_state_prob, shape (n_discrete_states,)
        # smoother_conditional_discrete_state_prob, shape (n_discrete_states, n_discrete_states)
        # smoother_joint_discrete_state_prob, shape (n_discrete_states, n_discrete_states)
        # discrete_state_weights, shape (n_discrete_states, n_discrete_states)

        # Collapse `n_discrete_states` x `n_discrete_states` Gaussians
        # to `n_discrete_states` Gaussians
        collapsed_means, collapsed_covs = (
            collapse_gaussian_mixture_per_previous_discrete_state(
                conditional_cont_means,
                conditional_cont_covs,
                discrete_state_weights,
            )
        )

        # collapsed_means, shape (n_cont_states, n_discrete_states)
        # collapsed_covs, shape (n_cont_states, n_cont_states, n_discrete_states)

        # Collapse to a single Gaussian
        single_collapsed_means, single_collapsed_covs = collapse_gaussian_mixture(
            collapsed_means, collapsed_covs, smoother_discrete_state_prob
        )

        # single_collapsed_means, shape (n_cont_states,)
        # single_collapsed_covs, shape (n_cont_states, n_cont_states)

        # Collapse `n_discrete_states` x `n_discrete_states` Gaussians to `n_discrete_states` Gaussians
        (
            next_conditional_cont_cross_means,
            conditional_cont_cross_means,
            conditional_cont_cross_covs,
        ) = collapse_cross_gaussian_mixture_across_states(
            next_conditional_cont_means,  # Mean at time t+1
            conditional_cont_means,  # Mean at time t
            conditional_cont_cross_covs,  # Cross-covariance between time t and t+1
            smoother_conditional_discrete_state_prob,  # Weights at time t
        )

        # next_conditional_cont_means, shape (n_cont_states, n_discrete_states, n_discrete_states)
        # conditional_cont_means, shape (n_cont_states, n_discrete_states, n_discrete_states)
        # conditional_cont_cross_covs, shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        # smoother_conditional_discrete_state_prob, shape (n_discrete_states, n_discrete_states)f

        # conditional_cont_cross_covs, shape (n_cont_states, n_cont_states, n_discrete_states)

        # Cross collapse to a single Gaussian
        _, _, smoother_cross_cov = collapse_gaussian_mixture_cross_covariance(
            next_conditional_cont_cross_means,  # Mean at time t+1
            conditional_cont_cross_means,  # Mean at time t
            conditional_cont_cross_covs,  # Cross-covariance between time t and t+1
            next_discrete_state_prob,  # Weights at time t
        )

        # next_conditional_cont_means, shape (n_cont_states, n_discrete_states, n_discrete_states)
        # conditional_cont_means, shape (n_cont_states, n_discrete_states, n_discrete_states)
        # conditional_cont_cross_covs, shape (n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        # smoother_conditional_discrete_state_prob, shape (n_discrete_states, n_discrete_states)
        # smoother_cross_cov, shape (n_cont_states, n_cont_states)

        return (
            collapsed_means,
            collapsed_covs,
            smoother_discrete_state_prob,
            conditional_cont_means,
        ), (
            single_collapsed_means,
            single_collapsed_covs,
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            smoother_cross_cov,
        )

    (
        smoother_means,
        smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        smoother_cross_cov,
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
    )[
        1
    ]

    last_smoother_mean, last_smoother_cov = collapse_gaussian_mixture(
        filter_mean[-1], filter_cov[-1], filter_discrete_state_prob[-1]
    )
    smoother_means = jnp.concatenate([smoother_means, last_smoother_mean[None]], axis=0)
    smoother_covs = jnp.concatenate([smoother_covs, last_smoother_cov[None]], axis=0)
    smoother_discrete_state_prob = jnp.concatenate(
        [smoother_discrete_state_prob, filter_discrete_state_prob[-1][None]], axis=0
    )

    return (
        smoother_means,
        smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        smoother_cross_cov,
    )


def weighted_outer_sum(x: jnp.ndarray, y: jnp.ndarray, weights: jnp.ndarray):
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
    weighted_outer_sum: jnp.ndarray, shape (x_dims, y_dims, n_discrete_states)
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
    smoother_means,
    smoother_covs,
    smoother_discrete_state_prob,
    smoother_joint_discrete_state_prob,
    smoother_cross_cov,
):
    """Maximization step for the Kalman filter.

    Parameters
    ----------
    obs : jnp.ndarray, shape (n_time, n_obs_dim)
        Observations.
    smoother_means : jnp.ndarray, shape (n_time, n_cont_states, n_discrete_states)
        smoother mean.
    smoother_covs : jnp.ndarray, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        smoother covariance.
    smoother_discrete_state_prob : jnp.ndarray, shape (n_time, n_discrete_states)
        smoother discrete state probabilities.
    smoother_joint_discrete_state_prob : jnp.ndarray, shape (n_time - 1, n_discrete_states, n_discrete_states)
        smoother joint discrete state probabilities.
    smoother_cross_cov : jnp.ndarray, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states)
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
        smoother_covs * smoother_discrete_state_prob[:, None, None], axis=0
    ) + weighted_outer_sum(smoother_means, smoother_means, smoother_discrete_state_prob)

    delta = weighted_outer_sum(
        obs[..., None], smoother_means, smoother_discrete_state_prob
    )
    alpha = weighted_outer_sum(
        obs[..., None], obs[..., None], smoother_discrete_state_prob
    )

    last_gamma = (
        smoother_covs[-1] * smoother_discrete_state_prob[-1, None, None]
    ) + weighted_outer_sum(
        smoother_means[[-1]], smoother_means[[-1]], smoother_discrete_state_prob[[-1]]
    )
    first_gamma = (
        smoother_covs[0] * smoother_discrete_state_prob[0, None, None]
    ) + weighted_outer_sum(
        smoother_means[[0]], smoother_means[[0]], smoother_discrete_state_prob[[0]]
    )
    gamma1 = gamma - last_gamma
    gamma2 = gamma - first_gamma

    beta = jnp.swapaxes(
        smoother_cross_cov * smoother_discrete_state_prob[1:, None, None], 1, 2
    ).sum(axis=0)

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
    init_mean = smoother_means[0]
    init_cov = smoother_covs[0]

    # Discrete transition matrix
    discrete_state_transition = smoother_joint_discrete_state_prob.sum(
        axis=0
    ) / smoother_discrete_state_prob[:-1].sum(axis=0)

    init_discrete_state_prob = smoother_discrete_state_prob[0]

    return (
        continuous_transition_matrix,
        measurement_matrix,
        process_cov,
        measurement_cov,
        init_mean,
        init_cov,
        discrete_state_transition,
        init_discrete_state_prob,
    )
