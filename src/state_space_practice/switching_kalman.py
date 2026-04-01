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
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
    predictive_likelihood_term_sum : jax.Array
        Scaled predictive likelihood sum (scalar array)
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


def _scale_likelihood(log_likelihood: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Scale the log likelihood to avoid numerical underflow.

    Parameters
    ----------
    log_likelihood : jax.Array, shape (n_discrete_states, n_discrete_states)
        Log likelihood of the discrete states.
    Returns
    -------
    scaled_likelihood : jax.Array, shape (n_discrete_states, n_discrete_states)
        Scaled log likelihood of the discrete states.
    ll_max : jax.Array
        Maximum log likelihood of the discrete states (scalar array).
    """

    ll_max = log_likelihood.max()
    ll_max = jnp.where(jnp.isfinite(ll_max), ll_max, 0.0)
    return jnp.exp(log_likelihood - ll_max), ll_max


def _kalman_measurement_update_only(
    prior_mean: jax.Array,
    prior_cov: jax.Array,
    obs: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Kalman measurement update without prediction step.

    Used for the first timestep in the x₁ convention where init_state represents
    the prior for x₁ directly (no dynamics prediction needed).

    Parameters
    ----------
    prior_mean : jax.Array, shape (n_cont_states,)
        Prior state mean p(x₁ | S₁)
    prior_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Prior state covariance
    obs : jax.Array, shape (n_obs_dim,)
        Observation y₁
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states)
        Observation matrix H
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim)
        Observation noise covariance R

    Returns
    -------
    posterior_mean : jax.Array, shape (n_cont_states,)
        Posterior state mean p(x₁ | y₁, S₁)
    posterior_cov : jax.Array, shape (n_cont_states, n_cont_states)
        Posterior state covariance
    marginal_log_likelihood : jax.Array
        Log p(y₁ | S₁) (scalar array)
    """
    # Measurement update (no prediction step)
    obs_mean = measurement_matrix @ prior_mean
    obs_cov = symmetrize(
        measurement_matrix @ prior_cov @ measurement_matrix.T + measurement_cov
    )

    residual_error = obs - obs_mean
    kalman_gain = psd_solve(obs_cov, measurement_matrix @ prior_cov).T

    posterior_mean = prior_mean + kalman_gain @ residual_error

    # Joseph form covariance update for numerical stability
    n_cont = prior_mean.shape[0]
    I_KH = jnp.eye(n_cont) - kalman_gain @ measurement_matrix
    posterior_cov = symmetrize(
        I_KH @ prior_cov @ I_KH.T + kalman_gain @ (measurement_cov @ kalman_gain.T)
    )

    marginal_log_likelihood = jnp.asarray(
        jax.scipy.stats.multivariate_normal.logpdf(x=obs, mean=obs_mean, cov=obs_cov)
    )

    return posterior_mean, posterior_cov, marginal_log_likelihood


def _first_timestep_kalman_update(
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    obs_t: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Handle first timestep with x₁ convention (measurement update only).

    For the first observation y₁, we treat init_state_cond_mean/cov as p(x₁ | S₁)
    and apply only the measurement update (no dynamics prediction). This aligns
    with the EM M-step which sets init_state from smoother_means[0] = x₁|T.

    Parameters
    ----------
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Prior belief about x₁ given S₁, p(x₁ | S₁)
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Prior covariance of x₁ given S₁
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Prior probability p(S₁)
    obs_t : jax.Array, shape (n_obs_dim,)
        Observation at first timestep y₁
    measurement_matrix : jax.Array, shape (n_obs_dim, n_cont_states, n_discrete_states)
        Observation matrices H_j for each discrete state
    measurement_cov : jax.Array, shape (n_obs_dim, n_obs_dim, n_discrete_states)
        Observation noise covariances R_j for each discrete state

    Returns
    -------
    state_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Posterior state-conditional means p(x₁ | y₁, S₁=j)
    state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Posterior state-conditional covariances
    filter_discrete_prob : jax.Array, shape (n_discrete_states,)
        Posterior discrete state probabilities p(S₁ | y₁)
    pair_cond_filter_mean : jax.Array, shape (n_cont_states, n_discrete_states, n_discrete_states)
        For compatibility with smoother (diagonal structure at t=1)
    marginal_log_likelihood : jax.Array
        Log p(y₁) contribution (scalar array)
    """
    n_discrete_states = init_state_cond_mean.shape[-1]

    # Apply measurement update for each discrete state (no dynamics prediction)
    # vmap over discrete states j
    vmapped_update = jax.vmap(
        _kalman_measurement_update_only,
        in_axes=(-1, -1, None, -1, -1),
        out_axes=(-1, -1, -1),
    )
    state_cond_filter_mean, state_cond_filter_cov, state_cond_log_lik = vmapped_update(
        init_state_cond_mean,
        init_state_cond_cov,
        obs_t,
        measurement_matrix,
        measurement_cov,
    )

    # Update discrete state probabilities using observation likelihood
    # At t=1, there's no transition from S₀, so we just use:
    # p(S₁=j | y₁) ∝ p(y₁ | S₁=j) * p(S₁=j)
    scaled_lik, ll_max = _scale_likelihood(state_cond_log_lik)
    unnorm_prob = scaled_lik * init_discrete_state_prob
    norm_const = jnp.sum(unnorm_prob)
    filter_discrete_prob = _divide_safe(unnorm_prob, norm_const)

    # Marginal log-likelihood contribution
    marginal_log_likelihood = ll_max + jnp.log(norm_const)

    # For smoother compatibility: create pair_cond_filter_mean
    # At t=1, there's no S₀, so we create a diagonal structure
    pair_cond_filter_mean = jnp.broadcast_to(
        state_cond_filter_mean[:, None, :],
        (state_cond_filter_mean.shape[0], n_discrete_states, n_discrete_states),
    )

    return (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_prob,
        pair_cond_filter_mean,
        marginal_log_likelihood,
    )


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
    jax.Array,  # Marginal log likelihood of the observations (scalar array)
]:
    """Switching Kalman filter for a linear Gaussian state space model with discrete states.

    This filter uses the x₁ convention where init parameters represent the state
    at the first observation time (not before it). For the first observation y₁,
    only a measurement update is applied (no dynamics prediction). For subsequent
    observations y_t (t > 1), the standard predict-then-update cycle is used.

    This convention ensures consistency with the EM M-step, which sets
    init_state_cond_mean = smoother_means[0] = x₁|T.

    Parameters
    ----------
    init_state_cond_mean : jax.Array, shape (n_cont_states, n_discrete_states)
        Prior belief about x₁ given S₁, p(x₁ | S₁ = j) for each discrete state.
        This is the prior on the latent state *at* the first observation, before
        incorporating y₁.
    init_state_cond_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        Prior covariance of x₁ given S₁ for each discrete state.
    init_discrete_state_prob : jax.Array, shape (n_discrete_states,)
        Prior discrete state probabilities p(S₁ = j).
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
    marginal_log_likelihood : jax.Array
        Marginal log likelihood of the observations (scalar array)

    """

    def _step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], obs_t: jax.Array
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],  # Next carry
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
            pair_cond_marginal_log_likelihood : jax.Array
                Previous marginal log likelihood (scalar array)
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
            marginal_log_likelihood : jax.Array
                Posterior marginal log likelihood (scalar array)
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

    # Handle first timestep with x₁ convention: measurement update only (no dynamics)
    # init_state_cond_mean represents p(x₁ | S₁), the prior for the first observation
    (
        first_state_cond_mean,
        first_state_cond_cov,
        first_discrete_prob,
        first_pair_cond_mean,
        first_log_lik,
    ) = _first_timestep_kalman_update(
        init_state_cond_mean,
        init_state_cond_cov,
        init_discrete_state_prob,
        obs[0],
        measurement_matrix,
        measurement_cov,
    )

    # Run predict-then-update for t=2,...,T
    # jax.lax.scan handles empty inputs (obs[1:] when n_time=1) gracefully
    (_, _, _, marginal_log_likelihood), (
        rest_state_cond_filter_mean,
        rest_state_cond_filter_cov,
        rest_filter_discrete_state_prob,
        rest_pair_cond_filter_mean,
    ) = jax.lax.scan(
        _step,
        (
            first_state_cond_mean,
            first_state_cond_cov,
            first_discrete_prob,
            first_log_lik,
        ),
        obs[1:],
    )

    # Prepend first timestep results
    state_cond_filter_mean = jnp.concatenate(
        [first_state_cond_mean[None, ...], rest_state_cond_filter_mean], axis=0
    )
    state_cond_filter_cov = jnp.concatenate(
        [first_state_cond_cov[None, ...], rest_state_cond_filter_cov], axis=0
    )
    filter_discrete_state_prob = jnp.concatenate(
        [first_discrete_prob[None, ...], rest_filter_discrete_state_prob], axis=0
    )
    pair_cond_filter_mean = jnp.concatenate(
        [first_pair_cond_mean[None, ...], rest_pair_cond_filter_mean], axis=0
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

# GPB2 triple vmap: produces S³ triple-conditional smoother posteriors.
#
# For each (i, j, k) = (S_{t-1}, S_t, S_{t+1}):
#   carry: (L, S_j, S_k) = smoother at t+1 conditioned on (S_t=j, S_{t+1}=k)
#   filter: (L, S_i) = filter at t-1 conditioned on S_{t-1}=i
#   dynamics: A_j, Q_j indexed by S_t=j
#
# Innermost: vmap over i (filter axis)
# Middle: vmap over j (carry first axis + dynamics)
# Outer: vmap over k (carry second axis)
#
# Output shapes: mean (L, S_i, S_j, S_k), cov (L, L, S_i, S_j, S_k)
_gpb2_kalman_smoother_update_triple = jax.vmap(
    jax.vmap(
        jax.vmap(
            _kalman_smoother_update,
            in_axes=(None, None, -1, -1, None, None),
            out_axes=-1,
        ),
        in_axes=(1, 2, None, None, -1, -1),
        out_axes=-1,
    ),
    in_axes=(-1, -1, None, None, None, None),
    out_axes=-1,
)


def _collapse_triple_to_pair(
    triple_mean: jax.Array,
    triple_cov: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Collapse S³ triple-conditional Gaussians to S² pair-conditional.

    Marginalizes over the last conditioning variable (S_{t+1}=k) for each
    pair (S_{t-1}=i, S_t=j).

    Parameters
    ----------
    triple_mean : jax.Array, shape (n_latent, S_i, S_j, S_k)
    triple_cov : jax.Array, shape (n_latent, n_latent, S_i, S_j, S_k)
    weights : jax.Array, shape (S_j, S_k)
        P(S_{t+1}=k | S_t=j, y_{1:T}) — forward conditional probability.

    Returns
    -------
    pair_mean : jax.Array, shape (n_latent, S_i, S_j)
    pair_cov : jax.Array, shape (n_latent, n_latent, S_i, S_j)
    """
    # For each (i, j): collapse over k using weights[j, :]
    # collapse_gaussian_mixture expects: means (L, S_k), cov (L, L, S_k), weights (S_k,)
    #
    # After outer vmap slices j: triple_mean becomes (L, S_i, S_k)
    # Inner vmap iterates over S_i (axis -2), leaving (L, S_k) per slice — correct.
    _collapse_over_k_for_fixed_j = jax.vmap(
        collapse_gaussian_mixture,
        in_axes=(-2, -2, None),  # vmap over i (second-to-last), keep k as mixture
        out_axes=(-1, -1),
    )
    _collapse_over_k = jax.vmap(
        _collapse_over_k_for_fixed_j,
        in_axes=(-2, -2, 0),  # vmap over j (second-to-last of 4D), weights axis 0
        out_axes=(-1, -1),
    )
    return _collapse_over_k(triple_mean, triple_cov, weights)


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
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
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

        # 1b. Stabilize pair-conditional smoother outputs (GPB1 safety net).
        # Cap at 1e8x the filter covariance trace. This is large enough to
        # not interfere with normal EM dynamics (where smoother cov can
        # legitimately exceed filter cov by 10-100x from GPB1 collapse) but
        # catches the exponential blowup that leads to overflow on long
        # sequences (where growth reaches 10^100+).
        _COV_CAP_MULTIPLIER = 1e8
        _MAX_SMOOTHER_MEAN_ABS = 1e6

        # Compute max allowed trace from filter cov
        max_filter_trace = jnp.max(jax.vmap(jnp.trace, in_axes=-1)(state_cond_filter_cov))
        max_allowed_trace = max_filter_trace * _COV_CAP_MULTIPLIER + 1.0

        def _cap_pair_cov(pair_cov_jk):
            trace = jnp.trace(pair_cov_jk)
            ratio = trace / max_allowed_trace
            return jnp.where(ratio > 1.0, pair_cov_jk / ratio, pair_cov_jk)

        pair_cond_smoother_covs = jax.vmap(
            jax.vmap(_cap_pair_cov, in_axes=-1, out_axes=-1),
            in_axes=-1, out_axes=-1,
        )(pair_cond_smoother_covs)

        pair_cond_smoother_mean = jnp.clip(
            pair_cond_smoother_mean, -_MAX_SMOOTHER_MEAN_ABS, _MAX_SMOOTHER_MEAN_ABS,
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


def switching_kalman_smoother_gpb2(
    filter_mean: jax.Array,
    filter_cov: jax.Array,
    filter_discrete_state_prob: jax.Array,
    last_filter_conditional_cont_mean: jax.Array,
    process_cov: jax.Array,
    continuous_transition_matrix: jax.Array,
    discrete_state_transition_matrix: jax.Array,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
    jax.Array, jax.Array, jax.Array, jax.Array,
]:
    """GPB2 switching Kalman smoother — carries S² pair-conditional structure.

    Same interface and return types as switching_kalman_smoother (GPB1), but
    maintains pair-conditional (S_t, S_{t+1}) Gaussians through the backward
    pass instead of collapsing to state-conditional at each step. This
    provides better numerical stability for long sequences with sparse
    observations (smoother gain > 1) by avoiding the aggressive GPB1 collapse
    that introduces unbounded Var[E[x|S]] growth.

    Computational cost is ~2x GPB1 for S=2 (8 vs 4 RTS updates per step).
    """
    n_discrete_states = filter_mean.shape[-1]

    def _step(carry, args):
        (
            next_pair_cond_smoother_mean,   # (L, S_j, S_k) = (S_t, S_{t+1})
            next_pair_cond_smoother_cov,    # (L, L, S_j, S_k)
            next_smoother_discrete_prob,    # (S,) P(S_{t+1}=k | y_{1:T})
            _,  # placeholder for prev pair_cond_smoother_mean (from outputs)
        ) = carry

        state_cond_filter_mean, state_cond_filter_cov, filter_discrete_prob = args

        # 1. Triple-conditional smoother update: (L, S_i, S_j, S_k)
        # For each (i, j, k): RTS update with filter(i), carry(j,k), dynamics(j)
        (
            triple_mean,   # (L, S_i, S_j, S_k)
            triple_cov,    # (L, L, S_i, S_j, S_k)
            triple_cross,  # (L, L, S_i, S_j, S_k)
        ) = _gpb2_kalman_smoother_update_triple(
            next_pair_cond_smoother_mean,   # (L, S_j, S_k)
            next_pair_cond_smoother_cov,    # (L, L, S_j, S_k)
            state_cond_filter_mean,         # (L, S_i)
            state_cond_filter_cov,          # (L, L, S_i)
            process_cov,                    # (L, L, S_j) — dynamics indexed by S_t=j
            continuous_transition_matrix,   # (L, L, S_j)
        )

        # 1b. Stabilize triple-conditional outputs (safety net).
        _MAX_COV_TRACE = 1e6
        _MAX_MEAN_ABS = 1e4

        def _cap_triple_cov(cov_ijk):
            trace = jnp.trace(cov_ijk)
            ratio = trace / _MAX_COV_TRACE
            return jnp.where(ratio > 1.0, cov_ijk / ratio, cov_ijk)

        triple_cov = jax.vmap(jax.vmap(jax.vmap(
            _cap_triple_cov, in_axes=-1, out_axes=-1
        ), in_axes=-1, out_axes=-1), in_axes=-1, out_axes=-1)(triple_cov)

        triple_mean = jnp.clip(triple_mean, -_MAX_MEAN_ABS, _MAX_MEAN_ABS)
        triple_cross = jnp.clip(triple_cross, -_MAX_COV_TRACE, _MAX_COV_TRACE)

        # 2. Compute discrete state probabilities
        (
            smoother_discrete_state_prob,
            smoother_backward_cond_prob,
            joint_smoother_discrete_prob,
            smoother_forward_cond_prob,
        ) = _update_smoother_discrete_probabilities(
            filter_discrete_prob,
            discrete_state_transition_matrix,
            next_smoother_discrete_prob,
        )

        # 3. Collapse S³ → S² by marginalizing over S_{t+1}=k
        # For each (i, j): collapse over k using P(S_{t+1}=k | S_t=j, y_{1:T})
        (
            pair_cond_smoother_mean,   # (L, S_i, S_j)
            pair_cond_smoother_cov,    # (L, L, S_i, S_j)
        ) = _collapse_triple_to_pair(
            triple_mean, triple_cov, smoother_forward_cond_prob,
        )

        # 4. Extract state-conditional and overall outputs (same as GPB1)
        # Collapse (S_i, S_j) → S_j by marginalizing over S_{t-1}=i
        (
            state_cond_smoother_means,  # (L, S_j)
            state_cond_smoother_covs,   # (L, L, S_j)
        ) = collapse_gaussian_mixture_per_discrete_state(
            pair_cond_smoother_mean,    # (L, S_i, S_j)
            pair_cond_smoother_cov,     # (L, L, S_i, S_j)
            smoother_backward_cond_prob,  # P(S_{t-1}=i | S_t=j, y_{1:T})
        )

        # Overall smoother (collapse S_j → 1)
        (
            overall_smoother_mean,
            overall_smoother_covs,
        ) = collapse_gaussian_mixture(
            state_cond_smoother_means,
            state_cond_smoother_covs,
            smoother_discrete_state_prob,
        )

        # Cross-covariance for M-step.
        #
        # The M-step needs Cov[x_{t+1}, x_t | S_t=j, S_{t+1}=k] indexed by (j,k).
        # We have triple_cross[:,:,i,j,k] = Cov[x_t, x_{t+1} | i, j, k] (note: t, t+1 order)
        # and triple_mean[:,i,j,k] = E[x_t | i, j, k].
        #
        # To get (j,k)-conditional, marginalize over i using
        # smoother_backward_cond_prob[i, j] = P(S_{t-1}=i | S_t=j).
        #
        # For each (j,k):
        #   cross_jk = sum_i P(i|j) * cross_ijk + sum_i P(i|j) * outer(m_t_ijk - m_t_jk, m_t1_jk - ...)
        # The second term involves means of x_{t+1} given (i,j,k). Since the carry has
        # next_pair_cond_smoother_mean[:,j,k] = E[x_{t+1} | j, k] (already marginalized over i),
        # x_{t+1} doesn't depend on i. So the spread term only involves x_t means.
        #
        # Simplified: cross_jk = sum_i P(i|j) * cross_ijk
        #   (because E[x_{t+1}|i,j,k] = E[x_{t+1}|j,k] — independent of i)

        # Weighted sum over i for each (j,k): einsum is clearest here
        # triple_cross: (L, L, S_i, S_j, S_k)
        # backward_prob: (S_i, S_j) = P(S_{t-1}=i | S_t=j)
        pair_cond_smoother_cross_covs = jnp.einsum(
            "abijk,ij->abjk",
            triple_cross,
            smoother_backward_cond_prob,
        )

        # Collapse to overall cross-cov
        overall_smoother_cross_cov = jnp.einsum(
            "abjk,jk->ab",
            pair_cond_smoother_cross_covs,
            joint_smoother_discrete_prob,
        )

        return (
            pair_cond_smoother_mean,      # (L, S_i, S_j) — carry
            pair_cond_smoother_cov,       # (L, L, S_i, S_j) — carry
            smoother_discrete_state_prob,  # (S,) — carry
            pair_cond_smoother_mean,       # for output stacking
        ), (
            overall_smoother_mean,
            overall_smoother_covs,
            smoother_discrete_state_prob,
            joint_smoother_discrete_prob,
            overall_smoother_cross_cov,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_covs,
            pair_cond_smoother_mean,
        )

    # Initialize carry from last filter output
    # At t=T, expand state-conditional to pair-conditional by broadcasting
    last_pair_mean = last_filter_conditional_cont_mean  # (L, S, S) already pair-cond
    n_cont = filter_cov.shape[1]
    last_pair_cov = jnp.broadcast_to(
        filter_cov[-1][:, :, None, :],  # (L, L, 1, S_j) → (L, L, S_i, S_j)
        (n_cont, n_cont, n_discrete_states, n_discrete_states),
    )

    init_carry = (
        last_pair_mean,
        last_pair_cov,
        filter_discrete_state_prob[-1],
        last_pair_mean,
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
        (filter_mean[:-1], filter_cov[:-1], filter_discrete_state_prob[:-1]),
        reverse=True,
    )

    # Append last timestep (same as GPB1)
    last_overall_mean, last_overall_cov = collapse_gaussian_mixture(
        filter_mean[-1], filter_cov[-1], filter_discrete_state_prob[-1],
    )
    overall_smoother_mean = jnp.concatenate(
        [overall_smoother_mean, last_overall_mean[None]], axis=0
    )
    overall_smoother_covs = jnp.concatenate(
        [overall_smoother_covs, last_overall_cov[None]], axis=0
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

    # Initial mean and covariance (x₁ convention)
    # init_state_cond_mean represents p(x₁ | S₁), the prior for the first observation.
    # smoother_means[0] = x₁|T, which is the smoothed estimate of x at the first
    # observation time. This aligns with the filter's x₁ convention where
    # init_state is the prior for the first observation (measurement update only,
    # no dynamics prediction for y₁).
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
) -> jax.Array:
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
    expected_complete_ll : jax.Array
        E_q[log p(y, x, s | θ)] (scalar array)
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
) -> jax.Array:
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
    entropy : jax.Array
        H(q(x, s)) (scalar array)
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
) -> jax.Array:
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
    elbo : jax.Array
        The evidence lower bound (scalar array)
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
) -> jax.Array:
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
    jax.Array
        Negative Q-function value (to be minimized, scalar array).
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
) -> jax.Array:
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
    jax.Array
        Negative Q-function value (to be minimized, scalar array).
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

    def loss(flat_params: jax.Array) -> jax.Array:
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