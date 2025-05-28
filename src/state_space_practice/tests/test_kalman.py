from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array, random

from state_space_practice.kalman import (
    kalman_filter,
    kalman_maximization_step,
    kalman_smoother,
    psd_solve,
    sum_of_outer_products,
    symmetrize,
)

# --- Unit Tests ---


def test_symmetrize() -> None:
    """Tests the symmetrize function."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    expected = jnp.array([[1.0, 2.5], [2.5, 4.0]])
    np.testing.assert_allclose(symmetrize(A), expected, rtol=1e-6)

    A_batch = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    expected_batch = jnp.array([[[1.0, 2.5], [2.5, 4.0]], [[5.0, 6.5], [6.5, 8.0]]])
    np.testing.assert_allclose(symmetrize(A_batch), expected_batch, rtol=1e-6)


def test_psd_solve() -> None:
    """Tests the psd_solve function."""
    A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    b = jnp.array([1.0, 2.0])
    expected_x = jnp.linalg.solve(A, b)
    x = psd_solve(A, b, diagonal_boost=0.0)
    np.testing.assert_allclose(x, expected_x, rtol=1e-5)

    A_ns = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    b_ns = jnp.array([2.0, 2.0])
    x_ns = psd_solve(A_ns, b_ns, diagonal_boost=1e-6)
    assert not jnp.any(jnp.isnan(x_ns))


@pytest.fixture(scope="module")
def simple_1d_model() -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Provides parameters and data for a simple 1D random walk model."""
    key = random.PRNGKey(0)
    n_time = 50
    n_cont_states = 1
    n_obs_dim = 1

    init_mean = jnp.array([0.0])
    init_cov = jnp.eye(n_cont_states) * 1.0
    transition_matrix = jnp.eye(n_cont_states) * 1.0
    process_cov = jnp.eye(n_cont_states) * 0.1
    measurement_matrix = jnp.eye(n_obs_dim, n_cont_states)
    measurement_cov = jnp.eye(n_obs_dim) * 1.0

    true_states = [init_mean]
    obs = []
    k1, k2 = random.split(key)

    for t in range(1, n_time):
        w = random.multivariate_normal(k1, jnp.zeros(n_cont_states), process_cov)
        true_states.append(transition_matrix @ true_states[-1] + w)
        k1, _ = random.split(k1)

    for t in range(n_time):
        v = random.multivariate_normal(k2, jnp.zeros(n_obs_dim), measurement_cov)
        obs.append(measurement_matrix @ true_states[t] + v)
        k2, _ = random.split(k2)

    return (
        init_mean,
        init_cov,
        jnp.array(obs),
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    )


def test_kalman_filter_shapes_and_likelihood(simple_1d_model: tuple) -> None:
    """Tests Kalman filter output shapes and likelihood value."""
    (
        init_mean,
        init_cov,
        obs,
        A,
        Q,
        H,
        R,
    ) = simple_1d_model
    n_time, n_obs_dim = obs.shape
    n_cont_states = init_mean.shape[0]

    filtered_mean, filtered_cov, mll = kalman_filter(
        init_mean, init_cov, obs, A, Q, H, R
    )

    assert filtered_mean.shape == (n_time, n_cont_states)
    assert filtered_cov.shape == (n_time, n_cont_states, n_cont_states)
    assert isinstance(mll.item(), float)
    assert not jnp.isnan(mll)


def test_kalman_smoother_shapes(simple_1d_model: tuple) -> None:
    """Tests Kalman smoother output shapes."""
    (
        init_mean,
        init_cov,
        obs,
        A,
        Q,
        H,
        R,
    ) = simple_1d_model
    n_time, n_obs_dim = obs.shape
    n_cont_states = init_mean.shape[0]

    smoother_mean, smoother_cov, smoother_cross_cov, _ = kalman_smoother(
        init_mean, init_cov, obs, A, Q, H, R
    )

    assert smoother_mean.shape == (n_time, n_cont_states)
    assert smoother_cov.shape == (n_time, n_cont_states, n_cont_states)
    assert smoother_cross_cov.shape == (
        n_time - 1,
        n_cont_states,
        n_cont_states,
    )


def test_sum_of_outer_products() -> None:
    """Tests the sum_of_outer_products function."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    expected = x.T @ y
    out = sum_of_outer_products(x, y)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_kalman_filter_values() -> None:
    """Tests Kalman filter output values against a hand-calculated example."""
    init_mean = jnp.array([0.0])
    init_cov = jnp.eye(1) * 1.0
    A = jnp.eye(1) * 1.0
    Q = jnp.eye(1) * 0.1
    H = jnp.eye(1)
    R = jnp.eye(1) * 1.0
    obs = jnp.array([[0.5], [0.6]])

    # Expected values from manual calculation.
    expected_means = jnp.array([[0.2619], [0.3919]])
    expected_covs = jnp.array([[[0.5238]], [[0.3845]]])

    filtered_mean, filtered_cov, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)

    np.testing.assert_allclose(filtered_mean, expected_means, rtol=1e-3)
    np.testing.assert_allclose(filtered_cov, expected_covs, rtol=1e-3)


@pytest.fixture(scope="module")
def kalman_m_step_test_data() -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Provides parameters and simulated data for M-step testing."""
    key = random.PRNGKey(42)
    n_time = 200
    n_cont_states = 1
    n_obs_dim = 1

    true_init_mean = jnp.array([0.5])
    true_init_cov = jnp.eye(n_cont_states) * 0.8
    true_A = jnp.eye(n_cont_states) * 0.98
    true_Q = jnp.eye(n_cont_states) * 0.2
    true_H = jnp.eye(n_obs_dim, n_cont_states) * 1.1
    true_R = jnp.eye(n_obs_dim) * 1.2

    true_states = jnp.zeros((n_time, n_cont_states))
    obs = jnp.zeros((n_time, n_obs_dim))
    key_init, key_process, key_obs = random.split(key, 3)

    current_state = random.multivariate_normal(key_init, true_init_mean, true_init_cov)
    true_states = true_states.at[0].set(current_state)

    for t in range(n_time):
        if t > 0:
            process_noise = random.multivariate_normal(
                random.fold_in(key_process, t), jnp.zeros(n_cont_states), true_Q
            )
            current_state = true_A @ current_state + process_noise
            true_states = true_states.at[t].set(current_state)

        obs_noise = random.multivariate_normal(
            random.fold_in(key_obs, t), jnp.zeros(n_obs_dim), true_R
        )
        obs = obs.at[t].set(true_H @ current_state + obs_noise)

    return (
        true_init_mean,
        true_init_cov,
        obs,
        true_A,
        true_Q,
        true_H,
        true_R,
    )


def test_kalman_maximization_step_recovery(kalman_m_step_test_data: tuple) -> None:
    """Tests if the M-step can recover known parameters (approximately)."""
    (
        init_mean_true,
        init_cov_true,
        obs,
        A_true,
        Q_true,
        H_true,
        R_true,
    ) = kalman_m_step_test_data

    smoother_mean, smoother_cov, smoother_cross_cov, _ = kalman_smoother(
        init_mean_true, init_cov_true, obs, A_true, Q_true, H_true, R_true
    )

    (
        A_est,
        H_est,
        Q_est,
        R_est,
        init_mean_est,
        init_cov_est,
    ) = kalman_maximization_step(obs, smoother_mean, smoother_cov, smoother_cross_cov)

    rtol_params = 0.2
    rtol_covs = 0.5

    np.testing.assert_allclose(A_est, A_true, rtol=rtol_params, atol=0.05)
    np.testing.assert_allclose(H_est, H_true, rtol=rtol_params, atol=0.05)
    np.testing.assert_allclose(Q_est, Q_true, rtol=rtol_covs, atol=0.1)
    np.testing.assert_allclose(R_est, R_true, rtol=rtol_covs, atol=0.1)
    np.testing.assert_allclose(init_mean_est, smoother_mean[0], rtol=1e-5)
    np.testing.assert_allclose(init_cov_est, smoother_cov[0], rtol=1e-5)

    assert jnp.allclose(Q_est, Q_est.T)
    assert jnp.allclose(R_est, R_est.T)
    assert jnp.allclose(init_cov_est, init_cov_est.T)


@pytest.fixture(scope="module")
def multi_dim_model() -> Tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    """
    Provides parameters and data for a 2D state, 2D observation model.

    Returns
    -------
    init_mean : jnp.ndarray
        Initial state mean (N,).
    init_cov : jnp.ndarray
        Initial state covariance (N, N).
    obs : jnp.ndarray
        Simulated observations (T, O).
    A : jnp.ndarray
        Transition matrix (N, N).
    Q : jnp.ndarray
        Process noise covariance (N, N).
    H : jnp.ndarray
        Observation matrix (O, N).
    R : jnp.ndarray
        Observation noise covariance (O, O).
    """
    key = random.PRNGKey(123)
    n_time = 15
    n_cont_states = 2
    n_obs_dim = 2

    init_mean = jnp.array([0.0, 0.0])
    init_cov = jnp.eye(n_cont_states) * 1.0

    # Slightly damped system with some cross-coupling
    transition_matrix = jnp.array([[0.95, 0.1], [-0.05, 0.9]])
    process_cov = jnp.eye(n_cont_states) * 0.2

    # Observe both states, but maybe with different scaling/noise
    measurement_matrix = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    measurement_cov = jnp.eye(n_obs_dim) * 0.8

    # Simulate data
    true_states = [init_mean]
    obs = []
    k1, k2 = random.split(key)

    for t in range(1, n_time):
        w = random.multivariate_normal(
            random.fold_in(k1, t), jnp.zeros(n_cont_states), process_cov
        )
        true_states.append(transition_matrix @ true_states[-1] + w)

    for t in range(n_time):
        v = random.multivariate_normal(
            random.fold_in(k2, t), jnp.zeros(n_obs_dim), measurement_cov
        )
        obs.append(measurement_matrix @ true_states[t] + v)

    return (
        init_mean,
        init_cov,
        jnp.array(obs),
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    )


def test_kalman_smoother_values() -> None:
    """Tests Kalman smoother output values against a known example."""
    # Use the same simple 1D model as test_kalman_filter_values
    init_mean = jnp.array([0.0])
    init_cov = jnp.eye(1) * 1.0
    A = jnp.eye(1) * 1.0
    Q = jnp.eye(1) * 0.1
    H = jnp.eye(1)
    R = jnp.eye(1) * 1.0
    obs = jnp.array([[0.5], [0.6]])

    # Expected values calculated manually or via a reference implementation.
    # Filtered: m_0|0=0.2619, P_0|0=0.5238; m_1|1=0.3919, P_1|1=0.3845
    # Smoothed (t=1): m_1|1=0.3919, P_1|1=0.3845 (last step is same as filter)
    # Smoothed (t=0): m_0|1=0.3711, P_0|1=0.3551
    expected_means = jnp.array([[0.3711], [0.3919]])
    expected_covs = jnp.array([[[0.3551]], [[0.3845]]])
    # FIX: P_0,1|1 = P_1|1 * J_0^T = 0.3845 * 0.8397 = 0.32286
    expected_cross_cov = jnp.array([[[0.3229]]])  # Use 4dp for comparison

    smoother_mean, smoother_cov, smoother_cross_cov, _ = kalman_smoother(
        init_mean, init_cov, obs, A, Q, H, R
    )

    np.testing.assert_allclose(smoother_mean, expected_means, rtol=1e-3)
    np.testing.assert_allclose(smoother_cov, expected_covs, rtol=1e-3)
    np.testing.assert_allclose(smoother_cross_cov, expected_cross_cov, rtol=1e-3)


def test_kalman_filter_multi_dim(multi_dim_model: tuple) -> None:
    """Tests Kalman filter shapes and stability with multi-dimensional data."""
    (
        init_mean,
        init_cov,
        obs,
        A,
        Q,
        H,
        R,
    ) = multi_dim_model
    n_time, n_obs_dim = obs.shape
    n_cont_states = init_mean.shape[0]

    assert n_cont_states == 2
    assert n_obs_dim == 2

    filtered_mean, filtered_cov, mll = kalman_filter(
        init_mean, init_cov, obs, A, Q, H, R
    )

    assert filtered_mean.shape == (n_time, n_cont_states)
    assert filtered_cov.shape == (n_time, n_cont_states, n_cont_states)
    assert not jnp.isnan(mll)
    assert not jnp.any(jnp.isnan(filtered_mean))
    assert not jnp.any(jnp.isnan(filtered_cov))


def test_kalman_smoother_multi_dim(multi_dim_model: tuple) -> None:
    """Tests Kalman smoother shapes and stability with multi-dimensional data."""
    (
        init_mean,
        init_cov,
        obs,
        A,
        Q,
        H,
        R,
    ) = multi_dim_model
    n_time, n_obs_dim = obs.shape
    n_cont_states = init_mean.shape[0]

    assert n_cont_states == 2
    assert n_obs_dim == 2

    smoother_mean, smoother_cov, smoother_cross_cov, _ = kalman_smoother(
        init_mean, init_cov, obs, A, Q, H, R
    )

    assert smoother_mean.shape == (n_time, n_cont_states)
    assert smoother_cov.shape == (n_time, n_cont_states, n_cont_states)
    assert smoother_cross_cov.shape == (
        n_time - 1,
        n_cont_states,
        n_cont_states,
    )
    assert not jnp.any(jnp.isnan(smoother_mean))
    assert not jnp.any(jnp.isnan(smoother_cov))
    assert not jnp.any(jnp.isnan(smoother_cross_cov))
