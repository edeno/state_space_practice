"""
Tests for the switching Kalman filter and smoother implementations.

This module contains tests for the `state_space_practice.switching_kalman`
module, covering helper functions, core vmapped functions, and integration
tests comparing SKF to KF and verifying SKF behavior.
"""

import jax

# Enable 64-bit precision for tests that require it.
jax.config.update("jax_enable_x64", True)

from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random
from jax.nn import one_hot

from state_space_practice.kalman import (
    kalman_filter,
    kalman_maximization_step,
    kalman_smoother,
)
from state_space_practice.switching_kalman import (
    _kalman_filter_update_per_discrete_state_pair,
    _kalman_smoother_update_per_discrete_state_pair,
    _scale_likelihood,
    _update_discrete_state_probabilities,
    _update_smoother_discrete_probabilities,
    collapse_gaussian_mixture,
    collapse_gaussian_mixture_cross_covariance,
    collapse_gaussian_mixture_over_next_discrete_state,
    collapse_gaussian_mixture_per_discrete_state,
    switching_kalman_filter,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
    weighted_sum_of_outer_products,
)
from state_space_practice.tests.test_kalman import simple_1d_model

# --- Fixtures ---


@pytest.fixture(scope="module")
def simple_skf_model() -> (
    Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]
):
    """
    Provides parameters and simulated data for a 1D, 2-state SKF model.

    Returns
    -------
    init_mean : Array
        Initial means (n_cont_states, n_discrete_states).
    init_cov : Array
        Initial covariances (n_cont_states, n_cont_states, n_discrete_states).
    init_prob : Array
        Initial discrete probabilities (n_discrete_states,).
    obs : Array
        Simulated observations (n_time, n_obs_dim).
    Z : Array
        Discrete transition matrix (n_discrete_states, n_discrete_states).
    A : Array
        Continuous transition matrix (n_cont_states, n_cont_states, n_discrete_states).
    Q : Array
        Process noise (n_cont_states, n_cont_states, n_discrete_states).
    H : Array
        Observation matrix (n_obs_dim, n_cont_states, n_discrete_states).
    R : Array
        Observation noise (n_obs_dim, n_obs_dim, n_discrete_states).
    """
    n_cont_states = 1
    n_obs_dim = 1
    n_discrete_states = 2

    init_mean = jnp.array([[0.0], [5.0]]).T
    init_cov = jnp.array([[[0.5]], [[2.0]]]).T
    init_prob = jnp.array([0.8, 0.2])

    A = jnp.array([[[0.95]], [[1.05]]]).T
    Q = jnp.array([[[0.1]], [[0.5]]]).T
    H = jnp.array([[[1.0]], [[1.0]]]).T
    R = jnp.array([[[1.0]], [[3.0]]]).T
    Z = jnp.array([[0.9, 0.1], [0.2, 0.8]])  # P(S_t | S_{t-1})

    key = random.PRNGKey(42)
    n_time = 100
    key, s_key, x_key, y_key = random.split(key, 4)

    s_t = [int(random.choice(s_key, jnp.arange(n_discrete_states), p=init_prob))]
    for t in range(1, n_time):
        s_key, subkey = random.split(s_key)
        s_t.append(
            int(random.choice(subkey, jnp.arange(n_discrete_states), p=Z[s_t[-1]]))
        )
    s_t_arr = jnp.array(s_t)

    x_t = []
    x_key, subkey = random.split(x_key)
    x_t.append(
        random.multivariate_normal(
            subkey, init_mean[:, s_t_arr[0]], init_cov[..., s_t_arr[0]]
        )
    )
    for t in range(1, n_time):
        x_key, subkey_w, subkey_x = random.split(x_key, 3)
        w = random.multivariate_normal(
            subkey_w, jnp.zeros(n_cont_states), Q[..., s_t_arr[t]]
        )
        x_t.append(A[..., s_t_arr[t]] @ x_t[-1] + w)
        x_key = subkey_x
    x_t_arr = jnp.array(x_t)

    y_t = []
    for t in range(n_time):
        y_key, subkey = random.split(y_key)
        v = random.multivariate_normal(subkey, jnp.zeros(n_obs_dim), R[..., s_t_arr[t]])
        y_t.append(H[..., s_t_arr[t]] @ x_t_arr[t] + v)
    y_t_arr = jnp.array(y_t)

    return init_mean, init_cov, init_prob, y_t_arr, Z, A, Q, H, R


@pytest.fixture(scope="module")
def simple_2_state_params() -> Tuple[Array, Array, Array, Array, int, int, int]:
    """
    Provides parameters for a simple 1D, 2-state model without data.

    Returns
    -------
    A : Array
        Continuous state transition matrix (N, N, M).
    Q : Array
        Process noise covariance (N, N, M).
    H : Array
        Observation matrix (O, N, M).
    R : Array
        Observation noise covariance (O, O, M).
    N : int
        Number of continuous states.
    M : int
        Number of discrete states.
    O : int
        Number of observation dimensions.
    """
    N, M, O = 1, 2, 1
    A = jnp.array([[[0.9]], [[1.05]]]).T
    Q = jnp.array([[[0.1]], [[0.5]]]).T
    H = jnp.array([[[1.0]], [[1.0]]]).T
    R = jnp.array([[[1.0]], [[2.0]]]).T
    return A, Q, H, R, N, M, O


# --- Unit Tests: Helper Functions ---


def test_collapse_gaussian_mixture() -> None:
    """Tests collapsing a simple Gaussian mixture."""
    means = jnp.array([[1.0], [10.0]]).T
    covs = jnp.array([[[1.0]], [[2.0]]]).T
    weights = jnp.array([0.5, 0.5])

    expected_mean = jnp.array([5.5])
    expected_cov = jnp.array([[21.75]])

    mean, cov = collapse_gaussian_mixture(means, covs, weights)

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-6)
    np.testing.assert_allclose(cov, expected_cov, rtol=1e-6)


def test_scale_likelihood() -> None:
    """Tests likelihood scaling for numerical stability."""
    log_likelihood = jnp.array([[-1000.0, -1001.0], [-1002.0, -1000.5]])
    scaled, ll_max = _scale_likelihood(log_likelihood)

    np.testing.assert_allclose(ll_max, -1000.0)
    np.testing.assert_allclose(jnp.max(scaled), 1.0)
    assert jnp.all(scaled >= 0)


def test_update_discrete_state_probabilities() -> None:
    """Tests the discrete probability update step in the filter."""
    likelihood = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    transitions = jnp.array([[0.95, 0.05], [0.1, 0.9]])
    prev_probs = jnp.array([0.7, 0.3])

    joint = jnp.array([[0.5985, 0.0035], [0.006, 0.216]])
    total = jnp.sum(joint)
    joint_norm = joint / total
    expected_m_t = jnp.sum(joint_norm, axis=0)
    expected_w = joint_norm / expected_m_t[None, :]

    m_t, w, ll_sum = _update_discrete_state_probabilities(
        likelihood, transitions, prev_probs
    )

    np.testing.assert_allclose(jnp.sum(m_t), 1.0, rtol=1e-6)
    np.testing.assert_allclose(w, expected_w, rtol=1e-6)
    np.testing.assert_allclose(m_t, expected_m_t, rtol=1e-6)
    np.testing.assert_allclose(jnp.sum(w, axis=0), jnp.array([1.0, 1.0]), rtol=1e-6)
    np.testing.assert_allclose(ll_sum, total, rtol=1e-6)


def test_update_discrete_state_probabilities_zero_sum_check() -> None:
    """Tests _update_discrete_state_probabilities handles zero predictive sum."""
    likelihood = jnp.array([[0.0, 0.1], [0.2, 0.8]])
    transitions = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    prev_probs = jnp.array([1.0, 0.0])

    m_t, w, ll_sum = _update_discrete_state_probabilities(
        likelihood, transitions, prev_probs
    )

    assert not jnp.any(jnp.isnan(m_t)), "m_t contains NaN!"
    assert not jnp.any(jnp.isnan(w)), "w contains NaN!"
    np.testing.assert_allclose(m_t, jnp.array([0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(w, jnp.zeros_like(w), atol=1e-6)
    np.testing.assert_allclose(ll_sum, 0.0, atol=1e-6)


def test_update_smoother_discrete_probabilities() -> None:
    """Tests the calculation of smoother discrete probabilities."""
    filter_prob = jnp.array([0.6, 0.4])
    Z = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    next_smoother_prob = jnp.array([0.7, 0.3])

    Z_input = Z.T
    expected_smoother_prob = jnp.array([0.73354, 0.26646])
    expected_joint_prob = jnp.array([[0.65172, 0.08182], [0.04828, 0.21818]])

    (
        smoother_prob,
        smoother_backward,
        joint_prob,
        smoother_forward,
    ) = _update_smoother_discrete_probabilities(
        filter_prob, Z_input, next_smoother_prob
    )

    np.testing.assert_allclose(jnp.sum(smoother_prob), 1.0, rtol=1e-6)
    np.testing.assert_allclose(smoother_prob, expected_smoother_prob, rtol=1e-4)
    np.testing.assert_allclose(joint_prob, expected_joint_prob, rtol=1e-4)
    np.testing.assert_allclose(jnp.sum(joint_prob, axis=1), smoother_prob, rtol=1e-6)
    np.testing.assert_allclose(
        jnp.sum(joint_prob, axis=0), next_smoother_prob, rtol=1e-6
    )
    np.testing.assert_allclose(
        jnp.sum(smoother_backward, axis=0), jnp.array([1.0, 1.0]), rtol=1e-6
    )
    np.testing.assert_allclose(
        jnp.sum(smoother_forward, axis=1), jnp.array([1.0, 1.0]), rtol=1e-6
    )


# --- Unit Tests: Vmapped Core Functions ---


def test_kalman_filter_update_per_discrete_state_pair(
    simple_2_state_params: tuple,
) -> None:
    """Tests the vmapped Kalman filter update step."""
    A, Q, H, R, N, M, O = simple_2_state_params
    mean_prev = jnp.array([[0.0], [5.0]]).T
    cov_prev = jnp.array([[[1.0]], [[2.0]]]).T
    obs = jnp.array([1.0])

    pair_m, pair_c, pair_ll = _kalman_filter_update_per_discrete_state_pair(
        mean_prev, cov_prev, obs, A, Q, H, R
    )

    assert pair_m.shape == (N, M, M)
    assert pair_c.shape == (N, N, M, M)
    assert pair_ll.shape == (M, M)


def test_collapse_gaussian_mixture_per_discrete_state() -> None:
    """Tests vmapped collapse (per_discrete_state), summing over 'i'."""
    N, M = 1, 2
    means = jnp.array([[[1.0, 10.0], [2.0, 12.0]]])
    covs = jnp.array([[[[1.0, 1.1], [1.2, 1.3]]]])
    weights = jnp.array([[0.5, 0.6], [0.5, 0.4]])

    mean_out, cov_out = collapse_gaussian_mixture_per_discrete_state(
        means, covs, weights
    )

    assert mean_out.shape == (N, M)
    assert cov_out.shape == (N, N, M)
    np.testing.assert_allclose(mean_out[0, 0], 1.5, rtol=1e-6)
    np.testing.assert_allclose(mean_out[0, 1], 10.8, rtol=1e-6)


def test_collapse_gaussian_mixture_over_next_discrete_state() -> None:
    """Tests vmapped collapse (over_next_discrete_state), summing over 'k'."""
    N, M = 1, 2
    means = jnp.array([[[1.0, 10.0], [2.0, 12.0]]])
    covs = jnp.array([[[[1.0, 1.1], [1.2, 1.3]]]])
    weights = jnp.array([[0.5, 0.6], [0.5, 0.4]])

    mean_out, cov_out = collapse_gaussian_mixture_over_next_discrete_state(
        means, covs, weights
    )

    assert mean_out.shape == (N, M)
    assert cov_out.shape == (N, N, M)
    np.testing.assert_allclose(mean_out[0, 0], 6.5, rtol=1e-6)
    np.testing.assert_allclose(mean_out[0, 1], 5.8, rtol=1e-6)


def test_collapse_gaussian_mixture_cross_covariance() -> None:
    """Tests collapsing a cross-covariance mixture."""
    x_means = jnp.array([[1.0], [10.0]]).T
    y_means = jnp.array([[2.0], [12.0]]).T
    cross_covs_cond = jnp.array([[[0.5]], [[0.5]]]).T
    weights = jnp.array([0.5, 0.5])

    expected_cov_xy = jnp.array([[23.0]])
    expected_mean_x = jnp.array([5.5])
    expected_mean_y = jnp.array([7.0])

    mx, my, cxy = collapse_gaussian_mixture_cross_covariance(
        x_means, y_means, cross_covs_cond, weights
    )

    np.testing.assert_allclose(mx, expected_mean_x, rtol=1e-6)
    np.testing.assert_allclose(my, expected_mean_y, rtol=1e-6)
    np.testing.assert_allclose(cxy, expected_cov_xy, rtol=1e-6)


def test_kalman_smoother_update_per_discrete_state_pair(
    simple_2_state_params: tuple,
) -> None:
    """Tests the vmapped Kalman smoother update step."""
    A, Q, _, _, N, M, _ = simple_2_state_params
    next_smoother_mean = jnp.array([[1.0], [6.0]]).T
    next_smoother_cov = jnp.array([[[0.5]], [[1.5]]]).T
    filter_mean = jnp.array([[0.0], [5.0]]).T
    filter_cov = jnp.array([[[1.0]], [[2.0]]]).T

    (
        pair_sm,
        pair_sc,
        pair_scc,
    ) = _kalman_smoother_update_per_discrete_state_pair(
        next_smoother_mean,
        next_smoother_cov,
        filter_mean,
        filter_cov,
        Q,
        A,
    )

    assert pair_sm.shape == (N, M, M)
    assert pair_sc.shape == (N, N, M, M)
    assert pair_scc.shape == (N, N, M, M)


def test_weighted_sum_of_outer_products() -> None:
    """Tests the weighted sum of outer products helper."""
    T, N, M = 2, 1, 2
    x = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    y = jnp.array([[[5.0, 6.0]], [[7.0, 8.0]]])
    weights = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    expected = jnp.array([[[8.7, 26.8]]])

    result = weighted_sum_of_outer_products(x, y, weights)

    assert result.shape == (N, N, M)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

    expected = jnp.zeros((N, N, M))
    for t in range(T):
        for s in range(M):
            outer_prod = weights[t, s] * jnp.outer(x[t, :, s], y[t, :, s])
            expected = expected.at[:, :, s].add(outer_prod)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


# --- Integration Tests: Filter and Smoother ---


def test_switching_kalman_filter_shapes(simple_skf_model: tuple) -> None:
    """Tests SKF filter output shapes and basic properties."""
    (
        init_mean,
        init_cov,
        init_prob,
        obs,
        Z,
        A,
        Q,
        H,
        R,
    ) = simple_skf_model
    n_time, _ = obs.shape
    n_cont_states = init_mean.shape[0]
    n_discrete_states = init_prob.shape[0]

    (
        filt_m,
        filt_c,
        filt_p,
        last_pair_m,
        mll,
    ) = switching_kalman_filter(init_mean, init_cov, init_prob, obs, Z, A, Q, H, R)

    assert filt_m.shape == (n_time, n_cont_states, n_discrete_states)
    assert filt_c.shape == (
        n_time,
        n_cont_states,
        n_cont_states,
        n_discrete_states,
    )
    assert filt_p.shape == (n_time, n_discrete_states)
    assert last_pair_m.shape == (
        n_cont_states,
        n_discrete_states,
        n_discrete_states,
    )
    assert not jnp.isnan(mll)
    np.testing.assert_allclose(jnp.sum(filt_p, axis=1), 1.0, rtol=1e-5)


def test_skf_reduces_to_kf_single_state(simple_1d_model: tuple) -> None:
    """Tests that a single-state SKF matches the standard KF."""
    (
        init_mean,
        init_cov,
        obs,
        A,
        Q,
        H,
        R,
    ) = simple_1d_model

    kf_m, kf_c, kf_mll = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)

    skf_init_mean = init_mean[:, None]
    skf_init_cov = init_cov[..., None]
    skf_init_prob = jnp.array([1.0])
    skf_A = A[..., None]
    skf_Q = Q[..., None]
    skf_H = H[..., None]
    skf_R = R[..., None]
    skf_Z = jnp.array([[1.0]])

    (
        skf_m,
        skf_c,
        skf_p,
        _,
        skf_mll,
    ) = switching_kalman_filter(
        skf_init_mean,
        skf_init_cov,
        skf_init_prob,
        obs,
        skf_Z,
        skf_A,
        skf_Q,
        skf_H,
        skf_R,
    )

    np.testing.assert_allclose(kf_m.squeeze(), skf_m.squeeze(), rtol=1e-5)
    np.testing.assert_allclose(kf_c.squeeze(), skf_c.squeeze(), rtol=1e-5)
    np.testing.assert_allclose(kf_mll, skf_mll, rtol=1e-5)
    np.testing.assert_allclose(skf_p, 1.0, rtol=1e-5)


def test_skf_smoother_reduces_to_kf_smoother_single_state(
    simple_1d_model: tuple,
) -> None:
    """Tests that a single-state SKF smoother matches the standard KF smoother."""
    (
        init_mean,
        init_cov,
        obs,
        A,
        Q,
        H,
        R,
    ) = simple_1d_model

    kf_sm, kf_sc, kf_scc, kf_mll = kalman_smoother(init_mean, init_cov, obs, A, Q, H, R)

    skf_init_mean = init_mean[:, None]
    skf_init_cov = init_cov[..., None]
    skf_init_prob = jnp.array([1.0])
    skf_A = A[..., None]
    skf_Q = Q[..., None]
    skf_H = H[..., None]
    skf_R = R[..., None]
    skf_Z = jnp.array([[1.0]])

    (
        skf_fm,
        skf_fc,
        skf_fp,
        last_pair_m,
        skf_mll,
    ) = switching_kalman_filter(
        skf_init_mean,
        skf_init_cov,
        skf_init_prob,
        obs,
        skf_Z,
        skf_A,
        skf_Q,
        skf_H,
        skf_R,
    )
    (
        skf_sm,
        skf_sc,
        skf_sp,
        skf_sjp,
        skf_scc,
        skf_scsm,
        skf_sccs,
        skf_pcscc,
    ) = switching_kalman_smoother(
        filter_mean=skf_fm,
        filter_cov=skf_fc,
        filter_discrete_state_prob=skf_fp,
        last_filter_conditional_cont_mean=last_pair_m,
        process_cov=skf_Q,
        continuous_transition_matrix=skf_A,
        discrete_state_transition_matrix=skf_Z,
    )

    np.testing.assert_allclose(kf_mll, skf_mll, rtol=1e-5)
    np.testing.assert_allclose(kf_sm, skf_sm, rtol=1e-5)
    np.testing.assert_allclose(kf_sc, skf_sc, rtol=1e-5)
    np.testing.assert_allclose(kf_scc, skf_scc, rtol=1e-5)
    np.testing.assert_allclose(skf_sp, 1.0, rtol=1e-5)
    np.testing.assert_allclose(skf_sjp, 1.0, rtol=1e-5)
    np.testing.assert_allclose(skf_scsm.squeeze(), skf_sm.squeeze(), rtol=1e-5)
    np.testing.assert_allclose(skf_sccs.squeeze(), skf_sc.squeeze(), rtol=1e-5)
    np.testing.assert_allclose(skf_pcscc.squeeze(), skf_scc.squeeze(), rtol=1e-5)


def test_skf_deterministic_stay(simple_skf_model: tuple) -> None:
    """Tests SKF when transitions force staying in the initial state."""
    (
        init_mean,
        init_cov,
        _,
        obs,
        _,
        A,
        Q,
        H,
        R,
    ) = simple_skf_model

    Z_stay = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    init_prob_0 = jnp.array([1.0, 0.0])
    _, _, filt_p_0, _, _ = switching_kalman_filter(
        init_mean, init_cov, init_prob_0, obs, Z_stay, A, Q, H, R
    )
    expected_p_0 = jnp.zeros_like(filt_p_0).at[:, 0].set(1.0)
    np.testing.assert_allclose(filt_p_0, expected_p_0, atol=1e-6)

    init_prob_1 = jnp.array([0.0, 1.0])
    _, _, filt_p_1, _, _ = switching_kalman_filter(
        init_mean, init_cov, init_prob_1, obs, Z_stay, A, Q, H, R
    )
    expected_p_1 = jnp.zeros_like(filt_p_1).at[:, 1].set(1.0)
    np.testing.assert_allclose(filt_p_1, expected_p_1, atol=1e-6)


def test_skf_deterministic_switch(simple_skf_model: tuple) -> None:
    """Tests SKF when transitions force switching states."""
    (
        init_mean,
        init_cov,
        _,
        obs,
        _,
        A,
        Q,
        H,
        R,
    ) = simple_skf_model
    n_time = obs.shape[0]

    Z_switch = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    init_prob_0 = jnp.array([1.0, 0.0])

    _, _, filt_p, _, _ = switching_kalman_filter(
        init_mean, init_cov, init_prob_0, obs, Z_switch, A, Q, H, R
    )

    indices = (jnp.arange(n_time) + 1) % 2
    expected_p = one_hot(indices, 2, dtype=filt_p.dtype)

    np.testing.assert_allclose(filt_p, expected_p, atol=1e-2, rtol=1e-2)


def test_m_step_one_state(
    simple_1d_model: tuple,
) -> None:
    """Tests the M-step of the EM algorithm for a single-state model
    matches the standard KF M-step."""
    (
        init_mean,
        init_cov,
        obs,
        A,
        Q,
        H,
        R,
    ) = simple_1d_model

    kf_sm, kf_sc, kf_scc, kf_mll = kalman_smoother(init_mean, init_cov, obs, A, Q, H, R)

    (
        kf_new_A,
        kf_new_H,
        kf_new_Q,
        kf_new_R,
        kf_new_init_mean,
        kf_new_init_cov,
    ) = kalman_maximization_step(
        obs,
        kf_sm,
        kf_sc,
        kf_scc,
    )

    skf_init_mean = init_mean[:, None]
    skf_init_cov = init_cov[..., None]
    skf_init_prob = jnp.array([1.0])
    skf_A = A[..., None]
    skf_Q = Q[..., None]
    skf_H = H[..., None]
    skf_R = R[..., None]
    skf_Z = jnp.array([[1.0]])

    (
        skf_fm,
        skf_fc,
        skf_fp,
        last_pair_m,
        skf_mll,
    ) = switching_kalman_filter(
        skf_init_mean,
        skf_init_cov,
        skf_init_prob,
        obs,
        skf_Z,
        skf_A,
        skf_Q,
        skf_H,
        skf_R,
    )
    (
        skf_sm,
        skf_sc,
        skf_sdsp,
        skf_sjdsp,
        skf_scc,
        skf_scsm,
        skf_scsc,
        skf_pcscc,
    ) = switching_kalman_smoother(
        filter_mean=skf_fm,
        filter_cov=skf_fc,
        filter_discrete_state_prob=skf_fp,
        last_filter_conditional_cont_mean=last_pair_m,
        process_cov=skf_Q,
        continuous_transition_matrix=skf_A,
        discrete_state_transition_matrix=skf_Z,
    )

    (
        skf_new_A,
        skf_new_H,
        skf_new_Q,
        skf_new_R,
        skf_new_init_mean,
        skf_new_init_cov,
        _,
        _,
    ) = switching_kalman_maximization_step(
        obs,
        skf_scsm,
        skf_scsc,
        skf_sdsp,
        skf_sjdsp,
        skf_pcscc,
    )

    np.testing.assert_allclose(kf_new_A, skf_new_A.squeeze(), rtol=1e-5)
    np.testing.assert_allclose(kf_new_H, skf_new_H.squeeze(), rtol=1e-5)
    np.testing.assert_allclose(kf_new_Q, skf_new_Q.squeeze(), rtol=1e-5)
    np.testing.assert_allclose(kf_new_R, skf_new_R.squeeze(), rtol=1e-5)
    np.testing.assert_allclose(kf_new_init_mean, skf_new_init_mean.squeeze(), rtol=1e-5)
    np.testing.assert_allclose(kf_new_init_cov, skf_new_init_cov.squeeze(), rtol=1e-5)


def test_m_step_two_identical_states(
    simple_1d_model: tuple,
) -> None:
    """Tests the SKF M-step with two identical discrete states
    against the standard KF M-step."""
    (
        init_mean_kf,
        init_cov_kf,
        obs,
        A_kf,
        Q_kf,
        H_kf,
        R_kf,
    ) = simple_1d_model

    # 1. Run standard KF E-step and M-step
    kf_sm, kf_sc, kf_scc, _ = kalman_smoother(
        init_mean_kf, init_cov_kf, obs, A_kf, Q_kf, H_kf, R_kf
    )
    (
        kf_new_A,
        kf_new_H,
        kf_new_Q,
        kf_new_R,
        kf_new_init_mean,
        kf_new_init_cov,
    ) = kalman_maximization_step(
        obs,
        kf_sm,
        kf_sc,
        kf_scc,
    )

    # 2. Setup SKF parameters for two identical states
    n_discrete_states = 2
    n_cont_states = init_mean_kf.shape[0]
    n_obs_dim = H_kf.shape[0]  # Assuming H_kf is (n_obs_dim, n_cont_states)

    # Initial continuous states (mean and cov are identical for both discrete states)
    skf_init_mean = jnp.stack([init_mean_kf] * n_discrete_states, axis=-1)
    skf_init_cov = jnp.stack([init_cov_kf] * n_discrete_states, axis=-1)

    # Initial discrete state probabilities (e.g., equal)
    skf_init_prob = jnp.ones(n_discrete_states) / n_discrete_states

    # Continuous parameters (A, Q, H, R are identical for both discrete states)
    skf_A = jnp.stack([A_kf] * n_discrete_states, axis=-1)
    skf_Q = jnp.stack([Q_kf] * n_discrete_states, axis=-1)
    skf_H = jnp.stack([H_kf] * n_discrete_states, axis=-1)
    skf_R = jnp.stack([R_kf] * n_discrete_states, axis=-1)

    # Discrete state transition matrix (e.g., some mixing but could be anything)
    skf_Z = jnp.full((n_discrete_states, n_discrete_states), 1.0 / n_discrete_states)
    # A more robust Z might be one that allows staying or switching with some probability
    # For example: skf_Z = jnp.array([[0.9, 0.1], [0.1, 0.9]]) if n_discrete_states == 2

    # 3. Run SKF E-step (Filter and Smoother)
    (
        skf_fm,  # state_cond_filter_mean
        skf_fc,  # state_cond_filter_cov
        skf_fp,  # filter_discrete_state_prob
        last_pair_m,  # last_filter_conditional_cont_mean
        _,  # mll
    ) = switching_kalman_filter(
        skf_init_mean,
        skf_init_cov,
        skf_init_prob,
        obs,
        skf_Z,
        skf_A,
        skf_Q,
        skf_H,
        skf_R,
    )

    # Assuming your switching_kalman_smoother now returns these specific ESS
    # Adjust the unpack based on your smoother's actual return signature
    (
        skf_sm,
        skf_sc,
        skf_sdsp,
        skf_sjdsp,
        skf_scc,
        skf_scsm,
        skf_scsc,
        skf_pcscc,
    ) = switching_kalman_smoother(  # Unpack only the necessary outputs for M-step
        filter_mean=skf_fm,
        filter_cov=skf_fc,
        filter_discrete_state_prob=skf_fp,
        last_filter_conditional_cont_mean=last_pair_m,
        process_cov=skf_Q,
        continuous_transition_matrix=skf_A,
        discrete_state_transition_matrix=skf_Z,
    )

    # 4. Run SKF M-step
    (
        skf_new_A,  # Shape (N,N,M)
        skf_new_H,  # Shape (O,N,M)
        skf_new_Q,  # Shape (N,N,M)
        skf_new_R,  # Shape (O,O,M)
        skf_new_init_mean,  # Shape (N,M)
        skf_new_init_cov,  # Shape (N,N,M)
        _,  # skf_new_Z
        _,  # skf_new_init_prob
    ) = switching_kalman_maximization_step(
        obs,
        skf_scsm,
        skf_scsc,
        skf_sdsp,
        skf_sjdsp,
        skf_pcscc,
    )

    # 5. Compare parameters
    rtol = 1e-5
    # Compare parameters for the first discrete state of SKF with KF
    np.testing.assert_allclose(kf_new_A, skf_new_A[..., 0], rtol=rtol)
    np.testing.assert_allclose(kf_new_H, skf_new_H[..., 0], rtol=rtol)
    np.testing.assert_allclose(kf_new_Q, skf_new_Q[..., 0], rtol=rtol)
    np.testing.assert_allclose(kf_new_R, skf_new_R[..., 0], rtol=rtol)
    np.testing.assert_allclose(kf_new_init_mean, skf_new_init_mean[..., 0], rtol=rtol)
    np.testing.assert_allclose(kf_new_init_cov, skf_new_init_cov[..., 0], rtol=rtol)

    # Compare parameters for the second discrete state of SKF with KF
    np.testing.assert_allclose(kf_new_A, skf_new_A[..., 1], rtol=rtol)
    np.testing.assert_allclose(kf_new_H, skf_new_H[..., 1], rtol=rtol)
    np.testing.assert_allclose(kf_new_Q, skf_new_Q[..., 1], rtol=rtol)
    np.testing.assert_allclose(kf_new_R, skf_new_R[..., 1], rtol=rtol)
    np.testing.assert_allclose(kf_new_init_mean, skf_new_init_mean[..., 1], rtol=rtol)
    np.testing.assert_allclose(kf_new_init_cov, skf_new_init_cov[..., 1], rtol=rtol)

    # Optionally, check that parameters for state 0 and state 1 of SKF are close
    np.testing.assert_allclose(skf_new_A[..., 0], skf_new_A[..., 1], rtol=rtol)
    np.testing.assert_allclose(skf_new_H[..., 0], skf_new_H[..., 1], rtol=rtol)


def test_m_step_discrete_transition_matrix_simple():
    """
    A bare-bones two-state switching Kalman M-step test.
    We fix A=1, Q=0, H=1, R=0 so that continuous stats play no role,
    synthesize a discrete sequence with known Z, and verify z_est ≈ Z.
    """
    # 1) Model sizes
    n_time = 20_000
    n_cont = 1
    n_obs = 1
    n_disc = 2

    # 2) Ground-truth transition matrix
    #   State 0 → stay 80%, switch→1 with 20%
    #   State 1 → stay 90%, switch→0 with 10%
    Z_true = jnp.array([[0.80, 0.20], [0.10, 0.90]], dtype=jnp.float32)
    init_prob = jnp.array([0.5, 0.5], dtype=jnp.float32)

    # 3) Generate a long discrete path s_t ~ Markov(Z_true)
    key = random.PRNGKey(0)
    key, sub = random.split(key)
    s = jnp.zeros(n_time, dtype=jnp.int32)
    s = s.at[0].set(random.choice(sub, a=n_disc, p=init_prob))
    for t in range(1, n_time):
        key, sub = random.split(key)
        prev = int(s[t - 1])
        s = s.at[t].set(random.choice(sub, a=n_disc, p=Z_true[prev]))

    # 4) Build “perfect” smoother outputs:
    #    • gammaₜ(j)=P(Sₜ=j)=1 for the true j, else 0
    #    • ξₜ(i,j)=P(Sₜ=i,Sₜ₊₁=j)=1 on the true transition pair, else 0
    gamma = one_hot(s, n_disc, dtype=jnp.float32)  # (T, 2)
    ξ = jnp.zeros((n_time - 1, n_disc, n_disc), dtype=jnp.float32)
    for t in range(n_time - 1):
        i, j_ = int(s[t]), int(s[t + 1])
        ξ = ξ.at[t, i, j_].set(1.0)

    # 5) Continuous stats all zero/degenerate:
    means = jnp.zeros((n_time, n_cont, n_disc), dtype=jnp.float32)
    covs = jnp.zeros((n_time, n_cont, n_cont, n_disc), dtype=jnp.float32)
    cross = jnp.zeros((n_time - 1, n_cont, n_cont, n_disc, n_disc), dtype=jnp.float32)

    # 6) Dummy observations (never used in Z‐update)
    obs = jnp.zeros((n_time, n_obs), dtype=jnp.float32)

    # 7) Call just the M‐step
    _, _, _, _, _, _, Z_est, init_prob_est = switching_kalman_maximization_step(
        obs=obs,
        state_cond_smoother_means=means,
        state_cond_smoother_covs=covs,
        smoother_discrete_state_prob=gamma,
        smoother_joint_discrete_state_prob=ξ,
        pair_cond_smoother_cross_cov=cross,
    )

    # 8) Assert we recover Z_true (and roughly the init‐prob too)
    np.testing.assert_allclose(Z_true, Z_est, atol=1e-2)


def test_m_step_transition_matrix_deterministic():
    """
    Deterministic test for the discrete-transition M-step.
    We hand-construct a 3-step path 0 -> 1 -> 0, so that
      Z_true = [[0,1],
                [1,0]]
    and verify that the M-step recovers it exactly.
    """
    # 1) dimensions
    T = 3
    n_cont = 1
    n_obs = 1
    n_disc = 2

    # 2) perfect smoothing marginals gammaₜ(j)=P(Sₜ=j | Y)
    #    time 0: state=0, time 1: state=1, time 2: state=0
    gamma = jnp.array(
        [
            [1.0, 0.0],  # t=0
            [0.0, 1.0],  # t=1
            [1.0, 0.0],  # t=2
        ],
        dtype=jnp.float32,
    )  # shape (T, n_disc)

    # 3) perfect two-step joint ξₜ(i,j)=P(Sₜ=i, Sₜ₊₁=j | Y)
    #    at t=0: 0->1, at t=1: 1->0
    ξ = jnp.zeros((T - 1, n_disc, n_disc), dtype=jnp.float32)
    ξ = ξ.at[0, 0, 1].set(1.0)
    ξ = ξ.at[1, 1, 0].set(1.0)

    # 4) degenerate continuous statistics (so A,Q,H,R updates are trivial)
    means = jnp.zeros((T, n_cont, n_disc), dtype=jnp.float32)
    covs = jnp.zeros((T, n_cont, n_cont, n_disc), dtype=jnp.float32)
    cross = jnp.zeros((T - 1, n_cont, n_cont, n_disc, n_disc), dtype=jnp.float32)

    # 5) dummy observations (not used for Z)
    obs = jnp.zeros((T, n_obs), dtype=jnp.float32)

    # 6) run just the M-step
    _, _, _, _, _, _, Z_est, init_prob_est = switching_kalman_maximization_step(
        obs=obs,
        state_cond_smoother_means=means,
        state_cond_smoother_covs=covs,
        smoother_discrete_state_prob=gamma,
        smoother_joint_discrete_state_prob=ξ,
        pair_cond_smoother_cross_cov=cross,
    )

    # 7) check we recovered the “swap” transition matrix exactly
    Z_true = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)

    np.testing.assert_allclose(Z_est, Z_true, atol=0.0)


def test_m_step_continuous_transition_matrix_estimation():
    """
    Deterministic test for the continuous-transition M-step.
    We use one discrete state and no noise, so x_t = A_true @ x_{t-1}
    exactly, and the M-step should recover A_true.
    """
    # 1) dimensions
    T = 10  # Use a longer trajectory for better conditioning
    n_cont = 2
    n_obs = 1
    n_disc = 1

    # 2) Ground-truth A.
    # **FIX**: Use eigenvalues closer to 1 to prevent exploding state values,
    # which caused numerical instability with float32.
    A_true_flat = jnp.array([[1.1, 0.1], [-0.05, 1.2]], dtype=jnp.float32)
    A_true = A_true_flat[..., None, None].transpose(
        2, 3, 0, 1
    )  # Shape (1,1,2,2) -> (2,2,1,1) -> (2,2,1)
    A_true = A_true.reshape(n_cont, n_cont, n_disc)

    # 3) Build a perfect 2-D trajectory x_t
    x_t = jnp.zeros((T, n_cont))
    x_t = x_t.at[0].set(jnp.array([1.0, 1.0]))
    for t in range(1, T):
        x_t = x_t.at[t].set(A_true_flat @ x_t[t - 1])

    # 4) Dummy observations (not used for A-update)
    obs = jnp.zeros((T, n_obs), dtype=jnp.float32)

    # 5) “Perfect” smoother outputs:
    means = x_t[:, :, None]
    covs = jnp.zeros((T, n_cont, n_cont, n_disc), dtype=jnp.float32)
    # The cross term E[x_t x_{t-1}^T] must be provided.
    cross = jnp.zeros((T - 1, n_cont, n_cont, n_disc, n_disc), dtype=jnp.float32)

    gamma = jnp.ones((T, n_disc), dtype=jnp.float32)
    ξ = jnp.ones((T - 1, n_disc, n_disc), dtype=jnp.float32)

    # 6) Run only the M-step
    (A_est, _, _, _, _, _, _, _) = switching_kalman_maximization_step(
        obs=obs,
        state_cond_smoother_means=means,
        state_cond_smoother_covs=covs,
        smoother_discrete_state_prob=gamma,
        smoother_joint_discrete_state_prob=ξ,
        pair_cond_smoother_cross_cov=cross,
    )

    # 7) Assert A_est == A_true exactly
    np.testing.assert_allclose(A_est.squeeze(), A_true_flat, atol=2e-5)


def test_m_step_continuous_transition_scalar():
    """
    Deterministic test for the continuous-transition M-step.
    x_{t+1} = 2 x_t exactly, so A_est should be 2.
    """
    # 1) dimensions
    T = 5
    n_cont = 1
    n_obs = 1
    n_disc = 1

    # 2) True A = 2
    A_true = jnp.array([[[2.0]]], dtype=jnp.float32)  # shape (1,1,1)

    # 3) Trajectory: x = [1,2,4,8,16]
    x_t = jnp.array([[1.0], [2.0], [4.0], [8.0], [16.0]], dtype=jnp.float32)

    # 4) Dummy observations (not used in A update)
    obs = jnp.zeros((T, n_obs), dtype=jnp.float32)

    # 5) “Perfect” smoother outputs:
    #    • E[x_t|S=0] = x_t
    #    • Cov[x_t|S=0] = 0
    #    • Cov[x_t,x_{t+1}|S=0,S=0] = 0
    #    • gamma=1 and ξ=1 for the single state
    means = x_t[:, :, None]  # (T, n_cont, n_disc)
    covs = jnp.zeros((T, n_cont, n_cont, n_disc), dtype=jnp.float32)
    cross = jnp.zeros((T - 1, n_cont, n_cont, n_disc, n_disc), dtype=jnp.float32)
    gamma = jnp.ones((T, n_disc), dtype=jnp.float32)
    ξ = jnp.ones((T - 1, n_disc, n_disc), dtype=jnp.float32)

    # 6) Run only the M-step
    A_est, _, _, _, _, _, _, _ = switching_kalman_maximization_step(
        obs=obs,
        state_cond_smoother_means=means,
        state_cond_smoother_covs=covs,
        smoother_discrete_state_prob=gamma,
        smoother_joint_discrete_state_prob=ξ,
        pair_cond_smoother_cross_cov=cross,
    )

    # 7) Assert exact recovery
    np.testing.assert_allclose(A_est.squeeze(), 2.0, atol=0.0)


def test_m_step_discrete_transition_deterministic_switch():
    """
    Tests the discrete transition matrix M-step with a perfect,
    deterministic switching path (0 -> 1 -> 0 -> ...).

    This test bypasses the filter/smoother by constructing the exact
    smoothed probabilities (gamma and xi) that would result from a
    perfect observation of this path. It verifies that the M-step
    recovers the known transition matrix Z = [[0, 1], [1, 0]].
    """
    # 1. Define dimensions and the deterministic path
    n_time = 4
    n_cont_states = 1
    n_obs_dim = 1
    n_discrete_states = 2
    dtype = jnp.float32

    # The true, deterministic path: 0 -> 1 -> 0 -> 1
    true_states = jnp.array([0, 1, 0, 1])

    # The ground-truth transition matrix we expect to recover
    z_true = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)

    # 2. Construct "perfect" smoother outputs (gamma and xi)
    # gamma_t(j) = P(S_t=j | Y) is 1 for the true state, 0 otherwise.
    gamma = one_hot(true_states, n_discrete_states, dtype=dtype)

    # xi_t(i,j) = P(S_t=i, S_{t+1}=j | Y) is 1 for the true transition, 0 otherwise.
    xi = jnp.zeros((n_time - 1, n_discrete_states, n_discrete_states), dtype=dtype)
    for t in range(n_time - 1):
        i, j = true_states[t], true_states[t + 1]
        xi = xi.at[t, i, j].set(1.0)

    # 3. Create dummy inputs for all other continuous parameters
    dummy_obs = jnp.zeros((n_time, n_obs_dim), dtype=dtype)
    dummy_means = jnp.zeros((n_time, n_cont_states, n_discrete_states), dtype=dtype)
    dummy_covs = jnp.zeros(
        (n_time, n_cont_states, n_cont_states, n_discrete_states), dtype=dtype
    )
    dummy_cross_cov = jnp.zeros(
        (
            n_time - 1,
            n_cont_states,
            n_cont_states,
            n_discrete_states,
            n_discrete_states,
        ),
        dtype=dtype,
    )

    # 4. Run the M-step
    (
        _,
        _,
        _,
        _,
        _,
        _,
        z_est,
        _,
    ) = switching_kalman_maximization_step(
        obs=dummy_obs,
        state_cond_smoother_means=dummy_means,
        state_cond_smoother_covs=dummy_covs,
        smoother_discrete_state_prob=gamma,
        smoother_joint_discrete_state_prob=xi,
        pair_cond_smoother_cross_cov=dummy_cross_cov,
    )

    # 5. Assert exact recovery of the transition matrix
    np.testing.assert_allclose(z_est, z_true, atol=1e-7)


def test_m_step_two_state_continuous_transition():
    """
    Tests the M-step for the continuous transition matrix `A` in a
    two-state system.
    This test constructs a deterministic trajectory where the system spends
    the first half of its time in state 0 (with dynamics A_0) and the
    second half in state 1 (with different dynamics A_1). By providing the
    M-step with the exact, known smoother outputs for this path, we can
    definitively test if it correctly recovers both A_0 and A_1.
    """
    # 1. Define dimensions and use float64 for numerical stability
    n_time = 20
    n_cont_states = 2
    n_obs_dim = 1
    n_discrete_states = 2
    dtype = jnp.float64

    # 2. Define two different ground-truth dynamics
    a_true_0 = jnp.array([[0.9, 0.1], [-0.1, 0.9]], dtype=dtype)
    a_true_1 = jnp.array([[1.1, -0.05], [0.05, 1.1]], dtype=dtype)
    a_true_stacked = jnp.stack([a_true_0, a_true_1], axis=-1)

    # 3. Construct the deterministic path (10 steps in state 0, 10 in state 1)
    half_time = n_time // 2
    true_states = jnp.array(
        [0] * half_time + [1] * (n_time - half_time), dtype=jnp.int32
    )
    x_t = jnp.zeros((n_time, n_cont_states), dtype=dtype)
    x_t = x_t.at[0].set(jnp.array([1.0, 1.0], dtype=dtype))

    for t in range(n_time - 1):
        current_state_idx = true_states[t]
        a_matrix = a_true_stacked[..., current_state_idx]
        x_t = x_t.at[t + 1].set(a_matrix @ x_t[t])

    # 4. Construct "perfect" smoother outputs based on the known path
    # THIS SECTION IS THE FIX. We must be careful about the transition point.
    gamma = one_hot(true_states, n_discrete_states, dtype=dtype)

    # Create xi for stay-stay transitions. The switch 0->1 is handled by setting its xi to 0
    # so it does not contribute to the estimation of A_0 or A_1.
    xi = jnp.zeros((n_time - 1, n_discrete_states, n_discrete_states), dtype=dtype)
    for t in range(n_time - 1):
        i, j = true_states[t], true_states[t + 1]
        if i == j:  # Only include transitions where the state does not change
            xi = xi.at[t, i, j].set(1.0)

    means = jnp.zeros((n_time, n_cont_states, n_discrete_states), dtype=dtype)
    for t in range(n_time):
        means = means.at[t, :, true_states[t]].set(x_t[t])

    # All covariances are zero in a deterministic system.
    covs = jnp.zeros(
        (n_time, n_cont_states, n_cont_states, n_discrete_states), dtype=dtype
    )
    cross_cov = jnp.zeros(
        (
            n_time - 1,
            n_cont_states,
            n_cont_states,
            n_discrete_states,
            n_discrete_states,
        ),
        dtype=dtype,
    )
    dummy_obs = jnp.zeros((n_time, n_obs_dim), dtype=dtype)

    # 5. Run the M-step
    (
        a_est,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = switching_kalman_maximization_step(
        obs=dummy_obs,
        state_cond_smoother_means=means,
        state_cond_smoother_covs=covs,
        smoother_discrete_state_prob=gamma,
        smoother_joint_discrete_state_prob=xi,
        pair_cond_smoother_cross_cov=cross_cov,
    )

    # 6. Assert exact recovery of both transition matrices
    np.testing.assert_allclose(a_est, a_true_stacked, atol=1e-7)
