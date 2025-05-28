"""
Tests for the switching Kalman filter and smoother implementations.

This module contains tests for the `state_space_practice.switching_kalman`
module, covering helper functions, core vmapped functions, and integration
tests comparing SKF to KF and verifying SKF behavior.
"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random
from jax.nn import one_hot

from state_space_practice.kalman import kalman_filter, kalman_smoother
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
