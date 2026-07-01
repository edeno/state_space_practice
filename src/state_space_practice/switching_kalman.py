"""Switching Kalman filter and smoother and EM algorithm.

References
----------
1. Shumway, R.H., and Stoffer, D.S. (1991). Dynamic Linear Models With Switching. 8.
2. Murphy, K.P. (1998). Switching kalman filters.
3. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2022). Switching Functional Network Models of Oscillatory Brain Dynamics. In 2022 56th Asilomar Conference on Signals, Systems, and Computers (IEEE), pp. 607–612. https://doi.org/10.1109/IEEECONF56349.2022.10052077.
4. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2024). Switching Models of Oscillatory Networks Greatly Improve Inference of Dynamic Functional Connectivity. Preprint at arXiv.
5. https://github.com/Stephen-Lab-BU/Switching_Oscillator_Networks
"""

from functools import partial

import jax
import jax.numpy as jnp

from state_space_practice.kalman import (
    _kalman_filter_update,
    _kalman_smoother_update,
    kalman_measurement_update,
    psd_solve,
    stabilize_covariance,
)
from state_space_practice.utils import divide_safe as _divide_safe
from state_space_practice.utils import safe_log as _safe_log
from state_space_practice.utils import scale_likelihood as _scale_likelihood
from state_space_practice.utils import (
    stabilize_probability_vector as _stabilize_probability_vector,
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

    # Cov[X] via the law of total covariance:
    #   Cov[X] = E[Cov[X | S]] + Cov[E[X | S]]
    unconditional_cov_x = (
        conditional_cov @ mixing_weights + (diff_x * mixing_weights) @ diff_x.T
    )

    return unconditional_mean_x, unconditional_cov_x


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
        Cov[X, Y | S = j], conditional cross-covariance per discrete state.
    mixing_weights : jax.Array, shape (n_discrete_states,)
        P[S = j]

    Returns
    -------
    unconditional_mean_x : jax.Array, shape (n_dims,)
        E[X]
    unconditional_mean_y : jax.Array, shape (n_dims,)
        E[Y]
    unconditional_cov_xy : jax.Array, shape (n_dims, n_dims)
        Cov[X, Y]
    """

    unconditional_mean_x = conditional_means_x @ mixing_weights  # E[X]
    unconditional_mean_y = conditional_means_y @ mixing_weights  # E[Y]

    diff_x = conditional_means_x - unconditional_mean_x[:, None]
    diff_y = conditional_means_y - unconditional_mean_y[:, None]

    # Cov[X, Y] via the law of total covariance:
    #   Cov[X, Y] = E[Cov[X, Y | S]] + Cov[E[X | S], E[Y | S]]
    unconditional_cov_xy = (
        conditional_cross_cov @ mixing_weights + (diff_x * mixing_weights) @ diff_y.T
    )

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


def _cap_covariance_trace(
    cov: jax.Array,
    max_allowed_trace: jax.Array,
) -> jax.Array:
    """Cap a single covariance matrix so its trace does not exceed max_allowed_trace."""
    trace = jnp.trace(cov)
    ratio = trace / max_allowed_trace
    return jnp.where(ratio > 1.0, cov / ratio, cov)


_COV_CAP_MULTIPLIER = 1e8
_MAX_SMOOTHER_MEAN_ABS = 1e6


def _compute_max_allowed_trace(
    state_cond_filter_cov: jax.Array,
) -> jax.Array:
    """Compute the maximum allowed covariance trace from filter covariances.

    Parameters
    ----------
    state_cond_filter_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        State-conditional filter covariances. The last axis is the discrete
        state dimension, which is what the inner ``vmap`` broadcasts over.

    Returns
    -------
    jax.Array
        Scalar maximum trace, equal to ``max(trace(cov[:, :, k]) for k in states) * _COV_CAP_MULTIPLIER + 1.0``.
        The additive 1.0 prevents the cap from collapsing to zero when filter
        covariances are identically zero.
    """
    max_filter_trace = jnp.max(
        jax.vmap(jnp.trace, in_axes=-1)(state_cond_filter_cov)
    )
    return max_filter_trace * _COV_CAP_MULTIPLIER + 1.0


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
    prev_filter_discrete_prob = _stabilize_probability_vector(prev_filter_discrete_prob)

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


def _first_timestep_kalman_update(
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    obs_t: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
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
        For smoother compatibility. At t=1 there is no S₀, so this is broadcast
        to be constant across the S₀ axis (not diagonal); the GPB2 smoother
        relies on that constant-across-i property at the first timestep.
    marginal_log_likelihood : jax.Array
        Log p(y₁) contribution (scalar array)
    """
    n_discrete_states = init_state_cond_mean.shape[-1]

    # Apply measurement update for each discrete state (no dynamics prediction)
    # vmap over discrete states j
    vmapped_update = jax.vmap(
        kalman_measurement_update,
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

    init_discrete_state_prob = _stabilize_probability_vector(init_discrete_state_prob)

    # Update discrete state probabilities using observation likelihood
    # At t=1, there's no transition from S₀, so we just use:
    # p(S₁=j | y₁) ∝ p(y₁ | S₁=j) * p(S₁=j)
    scaled_lik, ll_max = _scale_likelihood(state_cond_log_lik)
    unnorm_prob = scaled_lik * init_discrete_state_prob
    norm_const = jnp.sum(unnorm_prob)
    filter_discrete_prob = _divide_safe(unnorm_prob, norm_const)

    # Marginal log-likelihood contribution
    marginal_log_likelihood = ll_max + jnp.log(norm_const)

    # For smoother compatibility: create pair_cond_filter_mean and cov
    # At t=1, there's no S₀, so broadcast to be constant across the S₀ axis
    pair_cond_filter_mean = jnp.broadcast_to(
        state_cond_filter_mean[:, None, :],
        (state_cond_filter_mean.shape[0], n_discrete_states, n_discrete_states),
    )
    pair_cond_filter_cov = jnp.broadcast_to(
        state_cond_filter_cov[:, :, None, :],
        (*state_cond_filter_cov.shape[:2], n_discrete_states, n_discrete_states),
    )

    return (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_prob,
        pair_cond_filter_mean,
        pair_cond_filter_cov,
        marginal_log_likelihood,
    )


@jax.jit
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
    jax.Array,  # Pair-conditional filter mean trajectory
    jax.Array,  # Pair-conditional filter covariance trajectory
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
    pair_cond_filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional filter mean trajectory E[x_t | S_{t-1}=i, S_t=j, y_{1:t}].
        The first timestep uses the x_1 convention (broadcast over the
        nonexistent S_0). GPB1 callers use the last timestep ``[-1]``; the GPB2
        smoother consumes the full trajectory.
    pair_cond_filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional filter covariance trajectory Cov[x_t | S_{t-1}=i, S_t=j, y_{1:t}].
    marginal_log_likelihood : jax.Array
        Marginal log likelihood of the observations (scalar array)

    """

    def _step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], obs_t: jax.Array
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],  # Next carry
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],  # Stacked output
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
            pair_cond_filter_cov,
        )

    # Handle first timestep with x₁ convention: measurement update only (no dynamics)
    # init_state_cond_mean represents p(x₁ | S₁), the prior for the first observation
    (
        first_state_cond_mean,
        first_state_cond_cov,
        first_discrete_prob,
        first_pair_cond_mean,
        first_pair_cond_cov,
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
    # jax.lax.scan handles empty inputs (obs[1:] when n_time=1) gracefully.
    # The scan now emits the full pair-conditional filter trajectory as stacked
    # outputs (the GPB2 smoother needs every timestep, not just the last).
    (_, _, _, marginal_log_likelihood), (
        rest_state_cond_filter_mean,
        rest_state_cond_filter_cov,
        rest_filter_discrete_state_prob,
        rest_pair_cond_filter_mean,
        rest_pair_cond_filter_cov,
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
    # Full pair-conditional filter trajectories, E[x_t | S_{t-1}=i, S_t=j, y_{1:t}]
    # and its covariance. The first timestep uses the x_1 convention (broadcast
    # over the nonexistent S_0). The GPB2 smoother needs the whole trajectory;
    # GPB1 callers take the last timestep, ``pair_cond_filter_mean[-1]``.
    pair_cond_filter_mean = jnp.concatenate(
        [first_pair_cond_mean[None, ...], rest_pair_cond_filter_mean], axis=0
    )
    pair_cond_filter_cov = jnp.concatenate(
        [first_pair_cond_cov[None, ...], rest_pair_cond_filter_cov], axis=0
    )

    return (
        state_cond_filter_mean,
        state_cond_filter_cov,
        filter_discrete_state_prob,
        pair_cond_filter_mean,
        pair_cond_filter_cov,
        marginal_log_likelihood,
    )


def switching_kalman_viterbi(
    init_state_cond_mean: jax.Array,
    init_state_cond_cov: jax.Array,
    init_discrete_state_prob: jax.Array,
    obs: jax.Array,
    discrete_transition_matrix: jax.Array,
    continuous_transition_matrix: jax.Array,
    process_cov: jax.Array,
    measurement_matrix: jax.Array,
    measurement_cov: jax.Array,
) -> jax.Array:
    """Find the most likely discrete state sequence for a switching Kalman model.

    Runs the GPB2 filter forward pass to collect pair-conditional
    log-likelihoods ``log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)``, then
    applies a pairwise Viterbi algorithm that accounts for the dependence
    of the emission on both the current and previous discrete state.

    Parameters are identical to :func:`switching_kalman_filter`.

    Returns
    -------
    states : jax.Array, shape (n_time,)
        Most likely discrete state sequence (integer-valued).
    """
    n_discrete_states = init_state_cond_mean.shape[-1]

    # --- First timestep: measurement update only (x₁ convention) -----------
    (
        first_state_cond_mean,
        first_state_cond_cov,
        first_discrete_prob,
        _,
        _,
        _,
    ) = _first_timestep_kalman_update(
        init_state_cond_mean,
        init_state_cond_cov,
        init_discrete_state_prob,
        obs[0],
        measurement_matrix,
        measurement_cov,
    )

    # --- Forward pass: collect pair log-likelihoods -----------------------
    def _step(carry, obs_t):
        prev_mean, prev_cov, prev_prob = carry

        # Pair-conditional Kalman update (same as in the filter)
        (
            pair_cond_filter_mean,
            pair_cond_filter_cov,
            pair_cond_log_lik,  # (K, K): log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)
        ) = _kalman_filter_update_per_discrete_state_pair(
            prev_mean, prev_cov, obs_t,
            continuous_transition_matrix, process_cov,
            measurement_matrix, measurement_cov,
        )

        # Collapse mixture (same as filter) to get state-conditional
        # means/covs for the next step
        pair_cond_lik_scaled, _ = _scale_likelihood(pair_cond_log_lik)
        (
            filter_prob,
            backward_cond_prob,
            _,
        ) = _update_discrete_state_probabilities(
            pair_cond_lik_scaled,
            discrete_transition_matrix,
            prev_prob,
        )

        state_cond_mean, state_cond_cov = (
            collapse_gaussian_mixture_per_discrete_state(
                pair_cond_filter_mean,
                pair_cond_filter_cov,
                backward_cond_prob,
            )
        )

        return (state_cond_mean, state_cond_cov, filter_prob), pair_cond_log_lik

    _, pair_log_liks = jax.lax.scan(
        _step,
        (first_state_cond_mean, first_state_cond_cov,
         _stabilize_probability_vector(first_discrete_prob)),
        obs[1:],
    )
    # pair_log_liks: (T-1, K, K) where entry (t, i, j) is
    # log p(y_{t+1} | y_{1:t}, S_t=i, S_{t+1}=j)

    # --- Pairwise Viterbi -------------------------------------------------
    # Backward pass: accumulate best future scores with pair log-likelihoods
    log_trans = jnp.log(discrete_transition_matrix)

    def _viterbi_backward(best_next_score, t):
        # scores[i, j] = log A(i,j) + pair_log_lik(t, i, j) + best_future(j)
        scores = log_trans + pair_log_liks[t] + best_next_score[None, :]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    n_rest = pair_log_liks.shape[0]
    best_second_score, best_next_states = jax.lax.scan(
        _viterbi_backward,
        jnp.zeros(n_discrete_states),
        jnp.arange(n_rest),
        reverse=True,
    )

    # Best first state: first_discrete_prob is already p(S_1 | y_1),
    # so the first-obs likelihood is already encoded — do not add it again.
    first_state = jnp.argmax(
        jnp.log(first_discrete_prob) + best_second_score
    )

    # Forward trace
    def _viterbi_forward(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    _, states = jax.lax.scan(_viterbi_forward, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


_kalman_smoother_update_per_discrete_state_pair = jax.vmap(
    jax.vmap(
        _kalman_smoother_update, in_axes=(None, None, -1, -1, None, None), out_axes=-1
    ),
    in_axes=(-1, -1, None, None, -1, -1),
    out_axes=-1,
)

# GPB2 triple vmap: produces S³ triple-conditional smoother posteriors of x_t
# indexed by (i, j, k) = (S_{t-1}, S_t, S_{t+1}).
#
# Each RTS update for the triple (i, j, k) combines
#   next smoother : E[x_{t+1} | S_t=j, S_{t+1}=k, y_{1:T}]   (carry, pair-cond)
#   filter        : E[x_t     | S_{t-1}=i, S_t=j, y_{1:t}]   (pair-cond filter)
#   dynamics      : A_k, Q_k  (the x_t -> x_{t+1} transition is governed by
#                              S_{t+1}=k, matching the filter/GPB1/M-step)
# The middle state S_t=j is SHARED between the carry (its first axis) and the
# pair-conditional filter (its second axis); the two are matched on j rather
# than treated as independent. Argument order of ``_kalman_smoother_update`` is
# (next_mean, next_cov, filter_mean, filter_cov, Q, A) with shapes
#   carry_mean (L, S_j, S_k), carry_cov (L, L, S_j, S_k),
#   filter_mean (L, S_i, S_j), filter_cov (L, L, S_i, S_j),
#   Q (L, L, S_k), A (L, L, S_k).
#
# Outer vmap: k = S_{t+1} (carry 2nd axis + dynamics)
# Middle vmap: j = S_t (carry 1st axis + filter 2nd axis, shared)
# Inner vmap: i = S_{t-1} (filter 1st axis)
#
# Output shapes: mean (L, S_i, S_j, S_k), cov/cross (L, L, S_i, S_j, S_k)
_gpb2_kalman_smoother_update_triple = jax.vmap(
    jax.vmap(
        jax.vmap(
            _kalman_smoother_update,
            in_axes=(None, None, 1, 2, None, None),
            out_axes=-1,
        ),
        in_axes=(1, 2, 2, 3, None, None),
        out_axes=-1,
    ),
    in_axes=(2, 3, None, None, 2, 2),
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
    # Stabilize filter input to prevent underflow propagation.
    # Note: next_smoother_discrete_prob is NOT stabilized here — it is
    # stabilized in the carry output so that the stored smoother_prob[t+1]
    # and the value used to compute joint_prob[t] are always identical.
    filter_discrete_prob = _stabilize_probability_vector(filter_discrete_prob)

    # Discrete smoother prob
    # P(S_t = j, S_{t+1} = k | y_{1:T})
    # Unnormalized joint P(S_t=j) * P(S_{t+1}=k | S_t=j)
    unnormalized = (
        filter_discrete_prob[:, None] * discrete_state_transition_matrix
    )
    # Normalize columns to get P(S_t=j | S_{t+1}=k, y_{1:t})
    smoother_backward_cond_prob = _divide_safe(
        unnormalized, jnp.sum(unnormalized, axis=0)
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


@jax.jit
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
    """GPB1/IMM approximate switching Kalman smoother.

    This is an approximate smoother: the forward pass collapses K^2 mixture
    components to K at each step, so the state-conditional quantities (means
    and covariances) are approximate. The pair-conditional cross-covariances
    and means are computed from these collapsed quantities. For exact
    pair-conditional structure, use ``switching_kalman_smoother_gpb2``.

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
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
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
        max_allowed_trace = _compute_max_allowed_trace(state_cond_filter_cov)

        _cap = partial(_cap_covariance_trace, max_allowed_trace=max_allowed_trace)
        pair_cond_smoother_covs = jax.vmap(
            jax.vmap(_cap, in_axes=-1, out_axes=-1),
            in_axes=-1,
            out_axes=-1,
        )(pair_cond_smoother_covs)

        pair_cond_smoother_mean = jnp.clip(
            pair_cond_smoother_mean,
            -_MAX_SMOOTHER_MEAN_ABS,
            _MAX_SMOOTHER_MEAN_ABS,
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

        # Stabilize smoother_discrete_state_prob in the carry only,
        # so it is consistent when used as next_smoother_discrete_prob
        # in the next backward step. The output arrays store the
        # un-stabilized version for exact joint/marginal consistency.
        stabilized_smoother_prob = _stabilize_probability_vector(
            smoother_discrete_state_prob
        )

        return (
            state_cond_smoother_means,
            state_cond_smoother_covs,
            stabilized_smoother_prob,
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


@jax.jit
def switching_kalman_smoother_gpb2(
    filter_mean: jax.Array,
    filter_cov: jax.Array,
    filter_discrete_state_prob: jax.Array,
    pair_cond_filter_mean: jax.Array,
    pair_cond_filter_cov: jax.Array,
    process_cov: jax.Array,
    continuous_transition_matrix: jax.Array,
    discrete_state_transition_matrix: jax.Array,
) -> tuple[
    jax.Array,  # overall_smoother_mean
    jax.Array,  # overall_smoother_covs
    jax.Array,  # smoother_discrete_state_prob
    jax.Array,  # smoother_joint_discrete_state_prob
    jax.Array,  # overall_smoother_cross_cov
    jax.Array,  # state_cond_smoother_means
    jax.Array,  # state_cond_smoother_covs
    jax.Array,  # pair_cond_smoother_cross_covs
    jax.Array,  # pair_cond_smoother_means
    jax.Array,  # pair_cond_smoother_covs_mstep
    jax.Array,  # next_pair_cond_smoother_means
]:
    """GPB2 (Kim second-order) switching Kalman smoother.

    Carries pair-conditional (S_t, S_{t+1}) Gaussians through the backward
    pass instead of collapsing to state-conditional at each step (GPB1). Each
    backward step combines the pair-conditional smoother of ``x_{t+1}`` (the
    carry) with the pair-conditional *filter* of ``x_t`` to form the
    triple-conditional posterior ``E[x_t | S_{t-1}=i, S_t=j, S_{t+1}=k,
    y_{1:T}]``, which is then marginalized two ways:

    * over ``S_{t+1}=k`` (weighted by ``P(S_{t+1}=k | S_t=j, y_{1:T})``) to give
      the pair-conditional smoother ``E[x_t | S_{t-1}=i, S_t=j]`` carried to the
      next backward step;
    * over ``S_{t-1}=i`` (weighted by ``P(S_{t-1}=i | S_t=j, y_{1:T})``) to give
      ``E[x_t | S_t=j, S_{t+1}=k]`` and its covariance / cross-covariance, the
      exact pair-conditional sufficient statistics the M-step consumes.

    The ``x_t -> x_{t+1}`` transition is governed by ``A[..., S_{t+1}]``,
    matching :func:`switching_kalman_filter`, :func:`switching_kalman_smoother`
    (GPB1), and the M-step. Unlike GPB1 this consumes the full pair-conditional
    filter trajectory, which is what makes the S² backward recursion
    well-defined.

    Parameters
    ----------
    filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
        State-conditional filter mean E[x_t | S_t=j, y_{1:t}].
    filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
        State-conditional filter covariance.
    filter_discrete_state_prob : jax.Array, shape (n_time, n_discrete_states)
        M_{t|t}(j) = P(S_t=j | y_{1:t}).
    pair_cond_filter_mean : jax.Array, shape (n_time, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional filter mean E[x_t | S_{t-1}=i, S_t=j, y_{1:t}], as
        returned (whole trajectory) by :func:`switching_kalman_filter`.
    pair_cond_filter_cov : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Pair-conditional filter covariance Cov[x_t | S_{t-1}=i, S_t=j, y_{1:t}].
    process_cov : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    continuous_transition_matrix : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
    discrete_state_transition_matrix : jax.Array, shape (n_discrete_states, n_discrete_states)

    Returns
    -------
    Eleven arrays, in the same layout as before: overall smoother mean/cov,
    smoother discrete marginal/joint probabilities, overall cross-covariance,
    state-conditional smoother means/covs, and the pair-conditional
    cross-covariances, means, covariances, and next-step means used by the
    M-step. See the type annotation above for the order.

    Computational cost is ~2x GPB1 for S=2 (8 vs 4 RTS updates per step).
    """

    def _step(carry, args):
        (
            next_pair_cond_smoother_mean,  # E[x_{t+1} | S_t=j, S_{t+1}=k], (L, Sj, Sk)
            next_pair_cond_smoother_cov,  # (L, L, Sj, Sk)
            next_smoother_discrete_prob,  # M_{t+1|T}(k), (S,)
        ) = carry

        (
            pair_filter_mean,  # E[x_t | S_{t-1}=i, S_t=j, y_{1:t}], (L, Si, Sj)
            pair_filter_cov,  # (L, L, Si, Sj)
            filter_discrete_prob,  # M_{t|t}(j), (S,)
            prev_filter_discrete_prob,  # M_{t-1|t-1}(i), (S,); placeholder at t=0
        ) = args

        # 1. Triple-conditional RTS update: E[x_t | S_{t-1}=i, S_t=j, S_{t+1}=k].
        # The middle S_t=j is shared between the carry (pair smoother of x_{t+1})
        # and the pair-conditional filter of x_t; dynamics use A_k, Q_k.
        (
            triple_mean,  # (L, Si, Sj, Sk)
            triple_cov,  # (L, L, Si, Sj, Sk)
            triple_cross,  # (L, L, Si, Sj, Sk) = Cov[x_t, x_{t+1} | i, j, k]
        ) = _gpb2_kalman_smoother_update_triple(
            next_pair_cond_smoother_mean,  # (L, Sj, Sk)
            next_pair_cond_smoother_cov,  # (L, L, Sj, Sk)
            pair_filter_mean,  # (L, Si, Sj)
            pair_filter_cov,  # (L, L, Si, Sj)
            process_cov,  # Q_k
            continuous_transition_matrix,  # A_k
        )

        # 1b. Stabilize triple-conditional outputs. Cap the covariance trace
        # relative to the filter covariance and clip means to prevent overflow
        # on long sequences with extreme damping contrasts, while leaving
        # well-conditioned problems untouched.
        pair_filter_traces = jax.vmap(
            jax.vmap(jnp.trace, in_axes=-1, out_axes=-1), in_axes=-1, out_axes=-1
        )(pair_filter_cov)  # (Si, Sj)
        max_allowed = jnp.max(pair_filter_traces) * _COV_CAP_MULTIPLIER + 1.0

        _cap = partial(_cap_covariance_trace, max_allowed_trace=max_allowed)
        triple_cov = jax.vmap(
            jax.vmap(
                jax.vmap(_cap, in_axes=-1, out_axes=-1), in_axes=-1, out_axes=-1
            ),
            in_axes=-1,
            out_axes=-1,
        )(triple_cov)

        triple_mean = jnp.clip(
            triple_mean, -_MAX_SMOOTHER_MEAN_ABS, _MAX_SMOOTHER_MEAN_ABS
        )
        triple_cross = jnp.clip(triple_cross, -max_allowed, max_allowed)

        # 2. Discrete probabilities for the (S_t=j, S_{t+1}=k) pair.
        (
            smoother_discrete_state_prob,  # M_{t|T}(j)
            _smoother_backward_cond_prob,  # P(S_t=j | S_{t+1}=k) — unused here
            joint_smoother_discrete_prob,  # P(S_t=j, S_{t+1}=k | y_{1:T})
            smoother_forward_cond_prob,  # W^{k|j} = P(S_{t+1}=k | S_t=j, y_{1:T})
        ) = _update_smoother_discrete_probabilities(
            filter_discrete_prob,
            discrete_state_transition_matrix,
            next_smoother_discrete_prob,
        )

        # 3. Discrete backward probability for the (S_{t-1}=i, S_t=j) pair, used
        # to marginalize the past state S_{t-1}: V^{i|j} = P(S_{t-1}=i | S_t=j).
        # At the earliest step S_0 does not exist, but the pair-conditional
        # filter is degenerate over S_{t-1} there, so any row-stochastic V is
        # fine (prev_filter_discrete_prob is a harmless placeholder).
        _, smoother_backward_cond_prob_prev, _, _ = (
            _update_smoother_discrete_probabilities(
                prev_filter_discrete_prob,
                discrete_state_transition_matrix,
                smoother_discrete_state_prob,
            )
        )  # V^{i|j}, shape (Si, Sj)

        # 4. Carry: pair-conditional smoother E[x_t | S_{t-1}=i, S_t=j] obtained
        # by marginalizing the triple over the future S_{t+1}=k with W^{k|j}.
        (
            carry_pair_cond_mean,  # (L, Si, Sj)
            carry_pair_cond_cov,  # (L, L, Si, Sj)
        ) = _collapse_triple_to_pair(
            triple_mean,
            triple_cov,
            smoother_forward_cond_prob,
        )

        # 5. M-step statistics: pair-conditional E[x_t | S_t=j, S_{t+1}=k] and
        # its covariance / cross-covariance, from marginalizing the triple over
        # the past S_{t-1}=i with V^{i|j}. Because E[x_{t+1} | i,j,k] does not
        # depend on i (the carry conditions only on j,k), the cross-covariance
        # has no i-spread term.
        mstep_pair_cond_means = jnp.einsum(
            "lijk,ij->ljk", triple_mean, smoother_backward_cond_prob_prev
        )
        # Cov[x_t | S_t=j, S_{t+1}=k] = E_i[Cov] + Var_i[E]
        mstep_pair_cond_covs = jnp.einsum(
            "abijk,ij->abjk", triple_cov, smoother_backward_cond_prob_prev
        )
        mean_diff = triple_mean - mstep_pair_cond_means[:, None, :, :]
        mstep_pair_cond_covs += jnp.einsum(
            "aijk,bijk,ij->abjk",
            mean_diff,
            mean_diff,
            smoother_backward_cond_prob_prev,
        )
        pair_cond_smoother_cross_covs = jnp.einsum(
            "abijk,ij->abjk", triple_cross, smoother_backward_cond_prob_prev
        )
        mstep_next_pair_cond_means = next_pair_cond_smoother_mean  # (L, Sj, Sk)

        # 6. State-conditional smoother E[x_t | S_t=j] by marginalizing the
        # (S_t=j, S_{t+1}=k) pair over the future with W^{k|j}.
        (
            state_cond_smoother_means,  # (L, Sj)
            state_cond_smoother_covs,  # (L, L, Sj)
        ) = collapse_gaussian_mixture_over_next_discrete_state(
            mstep_pair_cond_means,
            mstep_pair_cond_covs,
            smoother_forward_cond_prob,
        )

        # 7. Overall smoother (collapse S_j -> 1).
        (
            overall_smoother_mean,
            overall_smoother_covs,
        ) = collapse_gaussian_mixture(
            state_cond_smoother_means,
            state_cond_smoother_covs,
            smoother_discrete_state_prob,
        )

        # Overall lag-one cross covariance (diagnostic): E over (j,k) of the
        # pair-conditional cross covariance.
        overall_smoother_cross_cov = jnp.einsum(
            "abjk,jk->ab",
            pair_cond_smoother_cross_covs,
            joint_smoother_discrete_prob,
        )

        # Stabilize the discrete marginal in the carry only, for consistency
        # when it is reused as next_smoother_discrete_prob; the output array
        # stores the un-stabilized value for exact joint/marginal consistency.
        stabilized_smoother_prob = _stabilize_probability_vector(
            smoother_discrete_state_prob
        )

        return (
            carry_pair_cond_mean,  # E[x_t | S_{t-1}=i, S_t=j] — next carry
            carry_pair_cond_cov,
            stabilized_smoother_prob,
        ), (
            overall_smoother_mean,
            overall_smoother_covs,
            smoother_discrete_state_prob,
            joint_smoother_discrete_prob,
            overall_smoother_cross_cov,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_covs,
            mstep_pair_cond_means,  # E[x_t | S_t=j, S_{t+1}=k]
            mstep_pair_cond_covs,  # Cov[x_t | S_t=j, S_{t+1}=k]
            mstep_next_pair_cond_means,  # E[x_{t+1} | S_t=j, S_{t+1}=k]
        )

    # Initialize carry from the last timestep, where smoother == filter (no
    # future data). The pair-conditional filter at T-1 is exactly
    # E[x_{T-1} | S_{T-2}=j, S_{T-1}=k, y_{1:T-1}].
    init_carry = (
        pair_cond_filter_mean[-1],
        pair_cond_filter_cov[-1],
        filter_discrete_state_prob[-1],
    )

    # prev_filter_discrete_prob[t] = M_{t-1|t-1}; the earliest step (t=0) uses a
    # placeholder because S_0 does not exist (see note in _step). Sliced to the
    # scan length (n_time - 1) so it is empty when n_time == 1.
    n_time = filter_discrete_state_prob.shape[0]
    prev_filter_discrete_prob = jnp.concatenate(
        [filter_discrete_state_prob[:1], filter_discrete_state_prob], axis=0
    )[: n_time - 1]

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
        pair_cond_smoother_covs_mstep,
        next_pair_cond_smoother_means,
    ) = jax.lax.scan(
        _step,
        init_carry,
        (
            pair_cond_filter_mean[:-1],
            pair_cond_filter_cov[:-1],
            filter_discrete_state_prob[:-1],
            prev_filter_discrete_prob,
        ),
        reverse=True,
    )

    # Append last timestep (same as GPB1)
    last_overall_mean, last_overall_cov = collapse_gaussian_mixture(
        filter_mean[-1],
        filter_cov[-1],
        filter_discrete_state_prob[-1],
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
        pair_cond_smoother_covs_mstep,
        next_pair_cond_smoother_means,
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
    lambda x, y, z, n: stabilize_covariance(_divide_safe(x - y @ z.T, n)),
    in_axes=(-1, -1, -1, -1),
    out_axes=-1,
)


@jax.jit
def _switching_kalman_m_step_inner(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    gamma1: jax.Array,
    beta: jax.Array,
    transition_pseudo_counts: jax.Array,
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
    """JIT-compiled inner M-step. All arrays are concrete (no None branching).

    ``gamma1`` and ``beta`` are the transition sufficient statistics,
    pre-computed by the outer ``switching_kalman_maximization_step`` which
    resolves the Optional pair-conditional paths before calling this function.
    ``transition_pseudo_counts`` is zeros for ML or ``(alpha - 1)`` for MAP.
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

    # Measurement matrix and covariance
    measurement_matrix = psd_solve_per_discrete_state(gamma, delta)
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

    # Discrete transition matrix (MAP with optional Dirichlet prior)
    expected_counts = smoother_joint_discrete_state_prob.sum(axis=0)
    expected_counts = expected_counts + transition_pseudo_counts
    discrete_state_transition = _divide_safe(
        expected_counts,
        jnp.sum(expected_counts, axis=1, keepdims=True),
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
        init_state_cond_mean,
        init_state_cond_cov,
        discrete_state_transition,
        init_discrete_state_prob,
    )


def switching_kalman_maximization_step(
    obs: jax.Array,
    state_cond_smoother_means: jax.Array,
    state_cond_smoother_covs: jax.Array,
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    pair_cond_smoother_cross_cov: jax.Array,
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
    transition_prior: jax.Array | None = None,
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
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional
        means for transition sufficient statistics. If None, uses the approximate factored form.
    pair_cond_smoother_covs : jax.Array | None, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Cov[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional
        covariances for gamma1. If None, falls back to state-conditional covariances.
    next_pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_{t+1} | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional
        next-step means for beta. If None, falls back to state-conditional means.
    transition_prior : jax.Array | None, shape (n_discrete_states, n_discrete_states)
        Dirichlet prior alpha parameters for the discrete transition matrix.
        If provided, adds (alpha - 1) pseudo-counts to the expected transition
        counts (MAP estimate). Use ``get_transition_prior(concentration, stickiness,
        n_states)`` from ``contingency_belief`` to construct. If None, uses the
        standard ML estimate.

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

    # Resolve Optional args into concrete arrays before calling JIT inner.
    # Compute gamma1 and beta (transition sufficient statistics) here
    # because the code path depends on which Optional args are provided.

    # Compute beta and gamma1 for transition matrix estimation
    # These use joint probability P(S_t=i, S_{t+1}=j) weighting
    if pair_cond_smoother_means is not None:
        # Exact pair-conditional sufficient statistics for GPB2.
        if pair_cond_smoother_covs is not None:
            gamma1 = jnp.einsum(
                "tij, tabij -> abj",
                smoother_joint_discrete_state_prob,
                pair_cond_smoother_covs,
            )
        else:
            gamma1 = jnp.einsum(
                "tij, tabi -> abj",
                smoother_joint_discrete_state_prob,
                state_cond_smoother_covs[:-1],
            )
        gamma1 += jnp.einsum(
            "tij, taij, tbij -> abj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_means,
            pair_cond_smoother_means,
        )

        beta = jnp.einsum(
            "tij,tdcij->cdj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov,
        )
        if next_pair_cond_smoother_means is not None:
            beta += jnp.einsum(
                "tdij,tcij,tij->cdj",
                pair_cond_smoother_means,
                next_pair_cond_smoother_means,
                smoother_joint_discrete_state_prob,
            )
        else:
            beta += jnp.einsum(
                "tdij,tcj,tij->cdj",
                pair_cond_smoother_means,
                state_cond_smoother_means[1:],
                smoother_joint_discrete_state_prob,
            )
    else:
        # Approximate factored form (original implementation)
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
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov,
        )
        beta += jnp.einsum(
            "tdi,tcj,tij->cdj",
            state_cond_smoother_means[:-1],
            state_cond_smoother_means[1:],
            smoother_joint_discrete_state_prob,
        )

    # Transition prior pseudo-counts (zeros = no prior = ML estimate)
    n_discrete_states = smoother_discrete_state_prob.shape[1]
    if transition_prior is not None:
        transition_pseudo_counts = transition_prior - 1.0
    else:
        transition_pseudo_counts = jnp.zeros(
            (n_discrete_states, n_discrete_states)
        )

    return _switching_kalman_m_step_inner(
        obs,
        state_cond_smoother_means,
        state_cond_smoother_covs,
        smoother_discrete_state_prob,
        smoother_joint_discrete_state_prob,
        gamma1,
        beta,
        transition_pseudo_counts,
    )


def _compute_expected_complete_log_likelihood_reference(
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
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
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
    pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses exact pair-conditional
        quantities for the transition Q-function term.
    pair_cond_smoother_covs : jax.Array | None, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Cov[X_t | y_{1:T}, S_t=i, S_{t+1}=j].
    next_pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_{t+1} | y_{1:T}, S_t=i, S_{t+1}=j].

    Returns
    -------
    expected_complete_ll : jax.Array
        E_q[log p(y, x, s | θ)] (scalar array)

    Notes
    -----
    When GPB2 pair-conditional quantities are provided, the transition
    Q-function term uses exact pair-conditional means and covariances for
    x_t, but Cov[x_{t+1} | S_t, S_{t+1}] is still approximated with
    the state-conditional Cov[x_{t+1} | S_{t+1}] since the GPB2 smoother
    does not produce that quantity directly. This is a minor residual
    approximation affecting only the ELBO diagnostic, not parameter updates.
    """
    n_time = obs.shape[0]
    n_discrete_states = smoother_discrete_state_prob.shape[1]
    n_cont_states = state_cond_smoother_means.shape[1]

    # 1. E_q[log p(s_1)] - initial discrete state
    log_init_discrete = jnp.sum(
        smoother_discrete_state_prob[0] * _safe_log(init_discrete_state_prob)
    )

    # 2. E_q[log p(x_1 | s_1)] - initial continuous state
    log_init_cont = jnp.zeros(())
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
    log_cont_trans = jnp.zeros(())
    for j in range(n_discrete_states):
        A_j = continuous_transition_matrix[:, :, j]
        Q_j = process_cov[:, :, j]
        log_det_Q = jnp.linalg.slogdet(Q_j)[1]

        for t in range(n_time - 1):
            # Sum over source states i weighted by P(s_t=i, s_{t+1}=j | y_{1:T})
            for i in range(n_discrete_states):
                weight = smoother_joint_discrete_state_prob[t, i, j]

                # E_q[(x_{t+1} - A_j x_t)(x_{t+1} - A_j x_t)^T | s_t=i, s_{t+1}=j]
                # Use pair-conditional quantities when available (GPB2 exact),
                # otherwise fall back to state-conditional (GPB1 approximate).
                if pair_cond_smoother_means is not None:
                    m_t_ij = pair_cond_smoother_means[t, :, i, j]
                else:
                    m_t_ij = state_cond_smoother_means[t, :, i]

                if next_pair_cond_smoother_means is not None:
                    m_t1_ij = next_pair_cond_smoother_means[t, :, i, j]
                else:
                    m_t1_ij = state_cond_smoother_means[t + 1, :, j]

                if pair_cond_smoother_covs is not None:
                    V_t_ij = pair_cond_smoother_covs[t, :, :, i, j]
                else:
                    V_t_ij = state_cond_smoother_covs[t, :, :, i]

                # For V_{t+1}, we don't have pair-conditional Cov[x_{t+1} | S_t, S_{t+1}]
                # separately (only Cov[x_t | S_t, S_{t+1}]). Use state-conditional.
                V_t1_j = state_cond_smoother_covs[t + 1, :, :, j]

                cross_cov_ij = pair_cond_smoother_cross_cov[t, :, :, i, j]

                # E[x_{t+1} x_{t+1}^T | ...]
                E_xt1_xt1 = V_t1_j + jnp.outer(m_t1_ij, m_t1_ij)
                # E[x_t x_t^T | ...]
                E_xt_xt = V_t_ij + jnp.outer(m_t_ij, m_t_ij)
                # E[x_{t+1} x_t^T | ...]
                E_xt1_xt = cross_cov_ij + jnp.outer(m_t1_ij, m_t_ij)

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
                log_cont_trans += jnp.where(weight > 0, weight * log_prob, 0.0)

    # 5. E_q[sum_t log p(y_t | x_t, s_t)] - observations
    log_obs = jnp.zeros(())
    for j in range(n_discrete_states):
        H_j = measurement_matrix[:, :, j]
        R_j = measurement_cov[:, :, j]
        log_det_R = jnp.linalg.slogdet(R_j)[1]
        n_obs = obs.shape[1]

        for t in range(n_time):
            weight = smoother_discrete_state_prob[t, j]

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
            log_obs += jnp.where(weight > 0, weight * log_prob, 0.0)

    return (
        log_init_discrete
        + log_init_cont
        + log_discrete_trans
        + log_cont_trans
        + log_obs
    )


def _compute_posterior_entropy_reference(
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
        smoother_discrete_state_prob[0] * _safe_log(smoother_discrete_state_prob[0])
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
    cont_entropy = jnp.zeros(())
    for j in range(n_discrete_states):
        for t in range(n_time):
            weight = smoother_discrete_state_prob[t, j]
            cov_j = state_cond_smoother_covs[t, :, :, j]
            log_det = jnp.linalg.slogdet(cov_j)[1]
            gaussian_entropy = 0.5 * (
                n_cont_states * (1 + jnp.log(2 * jnp.pi)) + log_det
            )
            cont_entropy += jnp.where(weight > 0, weight * gaussian_entropy, 0.0)

    return discrete_entropy + cont_entropy


# ---------------------------------------------------------------------------
# Vectorized ELBO functions (JIT-compatible, no Python loops)
# ---------------------------------------------------------------------------


def _weighted_gaussian_log_prob(
    mean: jax.Array, cov: jax.Array, smoother_mean: jax.Array, smoother_cov: jax.Array,
    n_cont_states: int,
) -> jax.Array:
    """Log N(smoother_mean; mean, cov) including the expected covariance term."""
    diff = smoother_mean - mean
    expected_outer = smoother_cov + jnp.outer(diff, diff)
    log_det = jnp.linalg.slogdet(cov)[1]
    trace_term = jnp.trace(psd_solve(cov, expected_outer))
    return -0.5 * (n_cont_states * jnp.log(2 * jnp.pi) + log_det + trace_term)


@jax.jit
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
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
) -> jax.Array:
    """Vectorized expected complete-data log-likelihood E_q[log p(y, x, s | θ)].

    Equivalent to ``_compute_expected_complete_log_likelihood_reference`` but
    uses vectorized JAX operations instead of Python loops, making it
    JIT-compilable and significantly faster for long sequences.

    See ``_compute_expected_complete_log_likelihood_reference`` for full
    parameter documentation.
    """
    n_cont_states = state_cond_smoother_means.shape[1]
    n_obs = obs.shape[1]

    # 1. Initial discrete state
    log_init_discrete = jnp.sum(
        smoother_discrete_state_prob[0] * _safe_log(init_discrete_state_prob)
    )

    # 2. Initial continuous state — vmap over j
    log_probs_init = jax.vmap(
        lambda mean_j, cov_j, sm_j, sc_j: _weighted_gaussian_log_prob(
            mean_j, cov_j, sm_j, sc_j, n_cont_states
        )
    )(
        init_state_cond_mean.T,                          # (K, n)
        jnp.moveaxis(init_state_cond_cov, -1, 0),       # (K, n, n)
        state_cond_smoother_means[0].T,                  # (K, n)
        jnp.moveaxis(state_cond_smoother_covs[0], -1, 0),  # (K, n, n)
    )  # (K,)
    log_init_cont = jnp.sum(smoother_discrete_state_prob[0] * log_probs_init)

    # 3. Discrete state transitions
    log_discrete_trans = jnp.sum(
        smoother_joint_discrete_state_prob * _safe_log(discrete_transition_matrix)
    )

    # 4. Continuous state transitions — resolve Optional branching, then vectorize
    # Resolve pair-conditional means/covs to concrete arrays
    if pair_cond_smoother_means is not None:
        m_t = pair_cond_smoother_means  # (T-1, n, K_i, K_j)
    else:
        # Broadcast state-cond means: E[x_t | S_t=i] for all j
        m_t = jnp.broadcast_to(
            state_cond_smoother_means[:-1, :, :, None],
            smoother_joint_discrete_state_prob.shape[:1]
            + state_cond_smoother_means.shape[1:2]
            + smoother_joint_discrete_state_prob.shape[1:],
        )

    if next_pair_cond_smoother_means is not None:
        m_t1 = next_pair_cond_smoother_means  # (T-1, n, K_i, K_j)
    else:
        # Broadcast state-cond means: E[x_{t+1} | S_{t+1}=j] for all i
        m_t1 = jnp.broadcast_to(
            state_cond_smoother_means[1:, :, None, :],
            smoother_joint_discrete_state_prob.shape[:1]
            + state_cond_smoother_means.shape[1:2]
            + smoother_joint_discrete_state_prob.shape[1:],
        )

    if pair_cond_smoother_covs is not None:
        V_t = pair_cond_smoother_covs  # (T-1, n, n, K_i, K_j)
    else:
        # Broadcast state-cond covs: Cov[x_t | S_t=i] for all j
        V_t = jnp.broadcast_to(
            state_cond_smoother_covs[:-1, :, :, :, None],
            smoother_joint_discrete_state_prob.shape[:1]
            + state_cond_smoother_covs.shape[1:3]
            + smoother_joint_discrete_state_prob.shape[1:],
        )

    # V_{t+1} is always state-conditional
    V_t1 = jnp.broadcast_to(
        state_cond_smoother_covs[1:, :, :, None, :],
        smoother_joint_discrete_state_prob.shape[:1]
        + state_cond_smoother_covs.shape[1:3]
        + smoother_joint_discrete_state_prob.shape[1:],
    )

    def _cont_trans_log_prob_single(weight, m_t_ij, m_t1_ij, V_t_ij, V_t1_ij,
                                     cross_cov_ij, A_j, Q_j, log_det_Q):
        """Log-prob for a single (t, i, j) triple."""
        E_xt1_xt1 = V_t1_ij + jnp.outer(m_t1_ij, m_t1_ij)
        E_xt_xt = V_t_ij + jnp.outer(m_t_ij, m_t_ij)
        E_xt1_xt = cross_cov_ij + jnp.outer(m_t1_ij, m_t_ij)
        expected_residual = (
            E_xt1_xt1 - A_j @ E_xt1_xt.T - E_xt1_xt @ A_j.T + A_j @ E_xt_xt @ A_j.T
        )
        trace_term = jnp.trace(psd_solve(Q_j, expected_residual))
        log_prob = -0.5 * (n_cont_states * jnp.log(2 * jnp.pi) + log_det_Q + trace_term)
        return jnp.where(weight > 0, weight * log_prob, 0.0)

    # vmap over i (source state): weight(i,), mean(n,i)->axis -1, cov(n,n,i)->axis -1
    _over_i = jax.vmap(
        _cont_trans_log_prob_single,
        in_axes=(0, -1, -1, -1, -1, -1, None, None, None),
    )
    # vmap over t (time): everything has t as axis 0
    _over_ti = jax.vmap(
        _over_i,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )

    # For each j: compute over all (t, i) and sum
    def _sum_for_j(A_j, Q_j, weights_j, m_t_j, m_t1_j, V_t_j, V_t1_j, cross_cov_j):
        """Sum log-probs over (t, i) for a single destination state j.

        weights_j: (T-1, K_i)
        m_t_j:     (T-1, n, K_i)
        V_t_j:     (T-1, n, n, K_i)
        cross_cov_j: (T-1, n, n, K_i)
        """
        log_det_Q = jnp.linalg.slogdet(Q_j)[1]
        return jnp.sum(_over_ti(
            weights_j, m_t_j, m_t1_j, V_t_j, V_t1_j, cross_cov_j,
            A_j, Q_j, log_det_Q,
        ))

    # vmap over j (destination state)
    log_cont_trans = jnp.sum(jax.vmap(
        _sum_for_j,
        in_axes=(2, 2, 2, 3, 3, 4, 4, 4),
    )(
        continuous_transition_matrix,       # (:, :, j)
        process_cov,                        # (:, :, j)
        smoother_joint_discrete_state_prob, # (:, i, j) -> axis 1 for i-within-j
        m_t,                                # (:, :, i, j) -> axis 3
        m_t1,                               # (:, :, i, j) -> axis 3
        V_t,                                # (:, :, :, i, j) -> axis 4
        V_t1,                               # (:, :, :, i, j) -> axis 4
        pair_cond_smoother_cross_cov,       # (:, :, :, i, j) -> axis 4
    ))

    # 5. Observations — vmap over j, vectorize over t
    def _obs_log_prob_for_j(H_j, R_j, weights_j, means_j, covs_j):
        """Sum observation log-probs over t for a single state j."""
        log_det_R = jnp.linalg.slogdet(R_j)[1]

        def _single_t(weight, m_t_j, V_t_j, y_t):
            pred_mean = H_j @ m_t_j
            diff = y_t - pred_mean
            expected_residual = jnp.outer(diff, diff) + H_j @ V_t_j @ H_j.T
            trace_term = jnp.trace(psd_solve(R_j, expected_residual))
            log_prob = -0.5 * (n_obs * jnp.log(2 * jnp.pi) + log_det_R + trace_term)
            return jnp.where(weight > 0, weight * log_prob, 0.0)

        return jnp.sum(jax.vmap(_single_t)(weights_j, means_j, covs_j, obs))

    log_obs = jnp.sum(jax.vmap(
        _obs_log_prob_for_j,
        in_axes=(2, 2, 1, 2, 3),
    )(
        measurement_matrix,            # (n_obs, n_cont, K) -> axis 2
        measurement_cov,               # (n_obs, n_obs, K) -> axis 2
        smoother_discrete_state_prob,  # (T, K) -> axis 1
        state_cond_smoother_means,     # (T, n_cont, K) -> axis 2
        state_cond_smoother_covs,      # (T, n_cont, n_cont, K) -> axis 3
    ))

    return (
        log_init_discrete + log_init_cont + log_discrete_trans
        + log_cont_trans + log_obs
    )


@jax.jit
def compute_posterior_entropy(
    smoother_discrete_state_prob: jax.Array,
    smoother_joint_discrete_state_prob: jax.Array,
    state_cond_smoother_covs: jax.Array,
) -> jax.Array:
    """Vectorized posterior entropy H(q).

    Equivalent to ``_compute_posterior_entropy_reference`` but uses vectorized
    JAX operations instead of Python loops.

    See ``_compute_posterior_entropy_reference`` for full parameter documentation.
    """
    n_cont_states = state_cond_smoother_covs.shape[1]

    # 1. Discrete entropy: t=0
    discrete_entropy = -jnp.sum(
        smoother_discrete_state_prob[0] * _safe_log(smoother_discrete_state_prob[0])
    )

    # t>0: vectorize over all time steps at once
    marginal_prev = smoother_discrete_state_prob[:-1]  # (T-1, K)
    joint = smoother_joint_discrete_state_prob          # (T-1, K, K)
    cond = _divide_safe(joint, marginal_prev[:, :, None])  # (T-1, K, K)
    discrete_entropy -= jnp.sum(joint * _safe_log(cond))

    # 2. Continuous entropy: vmap slogdet over (T, K)
    # state_cond_smoother_covs: (T, n, n, K) -> need (T, K, n, n) for vmap
    covs_tk = jnp.moveaxis(state_cond_smoother_covs, -1, 1)  # (T, K, n, n)
    T, K = covs_tk.shape[:2]
    covs_flat = covs_tk.reshape(T * K, n_cont_states, n_cont_states)
    log_dets = jax.vmap(lambda c: jnp.linalg.slogdet(c)[1])(covs_flat)
    log_dets = log_dets.reshape(T, K)  # (T, K)

    gaussian_entropies = 0.5 * (
        n_cont_states * (1 + jnp.log(2 * jnp.pi)) + log_dets
    )  # (T, K)
    weights = smoother_discrete_state_prob  # (T, K)
    cont_entropy = jnp.sum(jnp.where(weights > 0, weights * gaussian_entropies, 0.0))

    return discrete_entropy + cont_entropy


@jax.jit
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
    pair_cond_smoother_means: jax.Array | None = None,
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
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
    pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. Optional, for GPB2 exact Q-function.
    pair_cond_smoother_covs : jax.Array | None, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Cov[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. Optional, for GPB2 exact Q-function.
    next_pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_{t+1} | y_{1:T}, S_t=i, S_{t+1}=j]. Optional, for GPB2 exact Q-function.

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
        pair_cond_smoother_means=pair_cond_smoother_means,
        pair_cond_smoother_covs=pair_cond_smoother_covs,
        next_pair_cond_smoother_means=next_pair_cond_smoother_means,
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
    pair_cond_smoother_covs: jax.Array | None = None,
    next_pair_cond_smoother_means: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute sufficient statistics for transition matrix estimation.

    Parameters
    ----------
    state_cond_smoother_means : jax.Array, shape (n_time, n_cont_states, n_discrete_states)
    state_cond_smoother_covs : jax.Array, shape (n_time, n_cont_states, n_cont_states, n_discrete_states)
    smoother_joint_discrete_state_prob : jax.Array, shape (n_time - 1, n_discrete_states, n_discrete_states)
    pair_cond_smoother_cross_cov : jax.Array, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
    pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional means.
    pair_cond_smoother_covs : jax.Array | None, shape (n_time - 1, n_cont_states, n_cont_states, n_discrete_states, n_discrete_states)
        Cov[X_t | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional covariances.
    next_pair_cond_smoother_means : jax.Array | None, shape (n_time - 1, n_cont_states, n_discrete_states, n_discrete_states)
        E[X_{t+1} | y_{1:T}, S_t=i, S_{t+1}=j]. If provided, uses pair-conditional next means.

    Returns
    -------
    gamma1 : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        E[x_t x_t^T] weighted by joint probability P(S_t=i, S_{t+1}=j), summed over i.
    beta : jax.Array, shape (n_cont_states, n_cont_states, n_discrete_states)
        E[x_t x_{t+1}^T] weighted by joint probability.
    """
    if pair_cond_smoother_means is not None:
        # gamma1[a,b,j] = sum_{t,i} w_t^{ij} * (Cov[x_t | i,j] + m_t^{ij} (m_t^{ij})^T)
        if pair_cond_smoother_covs is not None:
            gamma1 = jnp.einsum(
                "tij, tabij -> abj",
                smoother_joint_discrete_state_prob,
                pair_cond_smoother_covs,
            )
        else:
            gamma1 = jnp.einsum(
                "tij, tabi -> abj",
                smoother_joint_discrete_state_prob,
                state_cond_smoother_covs[:-1],
            )
        gamma1 += jnp.einsum(
            "tij, taij, tbij -> abj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_means,
            pair_cond_smoother_means,
        )

        # beta[c,d,j] = sum_{t,i} w_t^{ij} * (Cov[x_{t+1}, x_t | i,j] + m_{t+1}^{ij} (m_t^{ij})^T)
        beta = jnp.einsum(
            "tij,tdcij->cdj",
            smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov,
        )
        if next_pair_cond_smoother_means is not None:
            beta += jnp.einsum(
                "tdij,tcij,tij->cdj",
                pair_cond_smoother_means,
                next_pair_cond_smoother_means,
                smoother_joint_discrete_state_prob,
            )
        else:
            beta += jnp.einsum(
                "tdij,tcj,tij->cdj",
                pair_cond_smoother_means,
                state_cond_smoother_means[1:],
                smoother_joint_discrete_state_prob,
            )
    else:
        # Approximate factored form (original implementation)
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

    from state_space_practice.oscillator_utils import (
        construct_directed_influence_transition_matrix,
    )

    n_osc = len(init_params["damping"])

    # Optimize in transformed coordinates for stability:
    # - damping: sigmoid maps (-inf, inf) -> (0, max_damping)
    # - coupling_strength: tanh maps (-inf, inf) -> (-max_coupling, max_coupling)
    # - freq, phase_diff: unconstrained
    max_damping = 0.995
    max_coupling = 0.5

    def _sigmoid(x: jax.Array) -> jax.Array:
        return max_damping * jax.nn.sigmoid(x)

    def _inv_sigmoid(y: jax.Array) -> jax.Array:
        y_clipped = jnp.clip(y / max_damping, 1e-6, 1.0 - 1e-6)
        return jnp.log(y_clipped / (1.0 - y_clipped))

    def _bounded_coupling(x: jax.Array) -> jax.Array:
        return max_coupling * jnp.tanh(x)

    def _inv_bounded_coupling(y: jax.Array) -> jax.Array:
        y_clipped = jnp.clip(y / max_coupling, -1.0 + 1e-6, 1.0 - 1e-6)
        return jnp.arctanh(y_clipped)

    def pack_unconstrained(params: dict) -> jax.Array:
        """Map physical params to unconstrained coordinates."""
        return jnp.concatenate(
            [
                _inv_sigmoid(params["damping"]),
                params["freq"],
                _inv_bounded_coupling(params["coupling_strength"]).ravel(),
                params["phase_diff"].ravel(),
            ]
        )

    def unpack_constrained(flat: jax.Array) -> dict:
        """Map unconstrained coordinates to physical params."""
        idx = 0
        damping = _sigmoid(flat[idx : idx + n_osc])
        idx += n_osc
        freq = flat[idx : idx + n_osc]
        idx += n_osc
        coupling = _bounded_coupling(
            flat[idx : idx + n_osc * n_osc].reshape(n_osc, n_osc)
        )
        idx += n_osc * n_osc
        phase = flat[idx : idx + n_osc * n_osc].reshape(n_osc, n_osc)
        return {
            "damping": damping,
            "freq": freq,
            "coupling_strength": coupling,
            "phase_diff": phase,
        }

    def loss(flat_params: jax.Array) -> jax.Array:
        params = unpack_constrained(flat_params)
        return compute_transition_q_from_params(
            damping=params["damping"],
            freq=params["freq"],
            coupling_strength=params["coupling_strength"],
            phase_diff=params["phase_diff"],
            sampling_freq=sampling_freq,
            gamma1=gamma1,
            beta=beta,
        )

    # Run optimizer in unconstrained space
    init_flat = pack_unconstrained(init_params)
    result = minimize(loss, init_flat, method="BFGS", tol=tol)

    opt_params = unpack_constrained(result.x)

    # Post-check: verify spectral radius of resulting A matrix.
    # If unstable, uniformly scale damping AND coupling so the
    # spectral radius of the reconstructed A is <= 0.99.
    A_opt = construct_directed_influence_transition_matrix(
        freqs=opt_params["freq"],
        damping_coeffs=opt_params["damping"],
        coupling_strengths=opt_params["coupling_strength"],
        phase_diffs=opt_params["phase_diff"],
        sampling_freq=sampling_freq,
    )
    spectral_radius = jnp.max(jnp.abs(jnp.linalg.eigvals(A_opt)))
    safe_scale = jnp.where(spectral_radius > 0.99, 0.99 / spectral_radius, 1.0)
    opt_params["damping"] = opt_params["damping"] * safe_scale
    opt_params["coupling_strength"] = opt_params["coupling_strength"] * safe_scale

    return opt_params
