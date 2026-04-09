"""Switching choice model: strategy-dependent multi-armed bandit.

Discrete latent states represent behavioral strategies (e.g., exploit
vs explore) that control how continuous option values evolve and drive
choices. Combines the softmax observation model from multinomial_choice
with the GPB2 switching infrastructure from switching_kalman.

Model:
    s_t ~ Categorical(T[s_{t-1}, :])
    x_t = decay_{s_t} * x_{t-1} + B @ u_t + w_t,  w_t ~ N(0, q_{s_t} * I)
    c_t ~ softmax(beta_{s_t} * [0, x_t] + Theta @ z_t)

Per-state parameters: beta_s, Q_s, decay_s
Shared parameters: B (input gain), Theta (obs weights), init_mean

References
----------
[1] Linderman et al. (2017). Bayesian learning and inference in recurrent
    switching linear dynamical systems. AISTATS.
[2] Smith et al. (2004). Dynamic analysis of learning in behavioral
    experiments. J Neuroscience 24(2), 447-461.
"""

import logging

import jax
from jax import Array

from state_space_practice.multinomial_choice import _softmax_update_core

logger = logging.getLogger(__name__)


def _softmax_predict_and_update(
    prev_mean: Array,
    prev_cov: Array,
    choice: Array,
    transition_matrix: Array,
    process_cov: Array,
    n_options: int,
    inverse_temperature: float,
    input_gain: Array,
    covariates_t: Array,
    obs_offset: Array,
) -> tuple[Array, Array, Array]:
    """Predict + softmax update for one (prev_state_i, next_state_j) pair.

    Parameters
    ----------
    prev_mean : Array, shape (K-1,)
    prev_cov : Array, shape (K-1, K-1)
    choice : Array, scalar int
    transition_matrix : Array, shape (K-1, K-1) — A_j = decay_j * I
    process_cov : Array, shape (K-1, K-1) — Q_j
    n_options : int
    inverse_temperature : float — beta_j
    input_gain : Array, shape (K-1, d) — shared B
    covariates_t : Array, shape (d,)
    obs_offset : Array, shape (K,) — Theta @ z_t

    Returns
    -------
    post_mean : Array, shape (K-1,)
    post_cov : Array, shape (K-1, K-1)
    log_likelihood : Array, shape ()
    """
    # Predict
    pred_mean = transition_matrix @ prev_mean + input_gain @ covariates_t
    pred_cov = (
        transition_matrix @ prev_cov @ transition_matrix.T + process_cov
    )

    # Update via softmax Laplace-EKF
    post_mean, post_cov, ll = _softmax_update_core(
        pred_mean, pred_cov, choice, n_options, inverse_temperature,
        obs_offset=obs_offset,
    )
    return post_mean, post_cov, ll


def _softmax_update_per_state_pair(
    prev_state_cond_mean: Array,
    prev_state_cond_cov: Array,
    choice: Array,
    transition_matrices: Array,
    process_covs: Array,
    n_options: int,
    inverse_temperatures: Array,
    input_gain: Array,
    covariates_t: Array,
    obs_offset: Array,
) -> tuple[Array, Array, Array]:
    """Per-state-pair softmax predict + update via double vmap.

    Computes pair-conditional posteriors for all (i, j) state pairs,
    following the switching_point_process.py pattern.

    Parameters
    ----------
    prev_state_cond_mean : Array, shape (K-1, S)
        State-conditional means from previous trial.
    prev_state_cond_cov : Array, shape (K-1, K-1, S)
        State-conditional covariances from previous trial.
    choice : Array, scalar int
    transition_matrices : Array, shape (K-1, K-1, S)
        Per-state A_j = decay_j * I.
    process_covs : Array, shape (K-1, K-1, S)
        Per-state Q_j.
    n_options : int
    inverse_temperatures : Array, shape (S,)
        Per-state beta_j.
    input_gain : Array, shape (K-1, d) — shared B
    covariates_t : Array, shape (d,)
    obs_offset : Array, shape (K,) — shared Theta @ z_t

    Returns
    -------
    pair_cond_mean : Array, shape (K-1, S_prev, S_next)
    pair_cond_cov : Array, shape (K-1, K-1, S_prev, S_next)
    pair_cond_ll : Array, shape (S_prev, S_next)
    """

    def _update_one_pair(prev_mean_i, prev_cov_i, A_j, Q_j, beta_j):
        """Update for one (i, j) pair."""
        return _softmax_predict_and_update(
            prev_mean_i, prev_cov_i, choice,
            A_j, Q_j, n_options, beta_j,
            input_gain, covariates_t, obs_offset,
        )

    # vmap over prev_state i (axis -1 of mean/cov)
    def _update_all_prev_for_next_j(A_j, Q_j, beta_j):
        # vmap over prev state i
        return jax.vmap(
            lambda m, c: _update_one_pair(m, c, A_j, Q_j, beta_j),
            in_axes=(1, 2),  # mean: axis 1, cov: axis 2
            out_axes=(1, 2, 0),  # mean: (K-1, S_prev), cov: (K-1, K-1, S_prev), ll: (S_prev,)
        )(prev_state_cond_mean, prev_state_cond_cov)

    # vmap over next_state j (axis -1 of A/Q, element of beta)
    pair_mean, pair_cov, pair_ll = jax.vmap(
        _update_all_prev_for_next_j,
        in_axes=(2, 2, 0),  # A: axis 2, Q: axis 2, beta: axis 0
        out_axes=(2, 3, 1),  # mean: (K-1, S_prev, S_next), cov: (K-1, K-1, S_prev, S_next), ll: (S_prev, S_next)
    )(transition_matrices, process_covs, inverse_temperatures)

    return pair_mean, pair_cov, pair_ll
