from typing import Tuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import Array, random
from scipy.linalg import solve_discrete_are

from state_space_practice.kalman import (
    kalman_filter,
    kalman_maximization_step,
    kalman_smoother,
    psd_solve,
    sum_of_outer_products,
    symmetrize,
)
from state_space_practice.tests.conftest import (
    kalman_model_params,
    positive_definite_matrices,
    to_jax,
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

    expected = jnp.zeros((x.shape[1], y.shape[1]))
    for i in range(x.shape[0]):
        expected += jnp.outer(x[i], y[i])
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
    init_mean : jax.Array
        Initial state mean (N,).
    init_cov : jax.Array
        Initial state covariance (N, N).
    obs : jax.Array
        Simulated observations (T, O).
    A : jax.Array
        Transition matrix (N, N).
    Q : jax.Array
        Process noise covariance (N, N).
    H : jax.Array
        Observation matrix (O, N).
    R : jax.Array
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


# --- Property-Based Tests using Hypothesis ---


class TestSymmetrizeProperties:
    """Property-based tests for the symmetrize function."""

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_output_is_symmetric(self, n: int) -> None:
        """Symmetrized matrix should be exactly symmetric."""
        key = random.PRNGKey(42)
        A = random.normal(key, (n, n))
        result = symmetrize(A)
        np.testing.assert_allclose(result, result.T, rtol=1e-10)

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_symmetric_input_unchanged(self, n: int) -> None:
        """A symmetric matrix should be unchanged by symmetrize."""
        key = random.PRNGKey(123)
        A = random.normal(key, (n, n))
        A_sym = (A + A.T) / 2  # Make symmetric
        result = symmetrize(A_sym)
        np.testing.assert_allclose(result, A_sym, rtol=1e-10)

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_symmetrize_is_idempotent(self, n: int) -> None:
        """Applying symmetrize twice should give the same result."""
        key = random.PRNGKey(456)
        A = random.normal(key, (n, n))
        result1 = symmetrize(A)
        result2 = symmetrize(result1)
        np.testing.assert_allclose(result1, result2, rtol=1e-10)


class TestPsdSolveProperties:
    """Property-based tests for psd_solve."""

    @given(positive_definite_matrices(n=3))
    @settings(max_examples=30, deadline=None)
    def test_solution_satisfies_equation(self, A: np.ndarray) -> None:
        """Solution x should satisfy A @ x = b."""
        A_jax = jnp.array(A)
        key = random.PRNGKey(42)
        b = random.normal(key, (3,))

        x = psd_solve(A_jax, b)

        # Verify A @ x ≈ b (float32 precision ~1e-6)
        np.testing.assert_allclose(A_jax @ x, b, rtol=1e-3, atol=1e-5)

    @given(positive_definite_matrices(n=3))
    @settings(max_examples=30, deadline=None)
    def test_handles_identity_matrix_b(self, A: np.ndarray) -> None:
        """Solving A @ X = I should give the inverse of A."""
        A_jax = jnp.array(A)
        identity = jnp.eye(3)

        X = psd_solve(A_jax, identity)

        # A @ X should be close to identity (use atol for numerical precision)
        # Note: float32 precision limits accuracy to ~1e-5
        np.testing.assert_allclose(A_jax @ X, identity, rtol=1e-3, atol=2e-5)


class TestSumOfOuterProductsProperties:
    """Property-based tests for sum_of_outer_products."""

    @given(
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=30, deadline=None)
    def test_matches_matmul(self, n_time: int, n_x: int, n_y: int) -> None:
        """Result should equal X.T @ Y."""
        key = random.PRNGKey(42)
        x = random.normal(key, (n_time, n_x))
        y = random.normal(random.fold_in(key, 1), (n_time, n_y))

        result = sum_of_outer_products(x, y)
        expected = x.T @ y

        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestKalmanFilterProperties:
    """Property-based tests for the Kalman filter."""

    @given(kalman_model_params(n_cont_states=2, n_obs_dim=2))
    @settings(max_examples=20, deadline=None)
    def test_filter_covariances_are_positive_definite(self, params: dict) -> None:
        """Filter covariances should be positive definite at all time steps."""
        init_mean, init_cov, A, Q, H, R = to_jax(
            params["init_mean"],
            params["init_cov"],
            params["A"],
            params["Q"],
            params["H"],
            params["R"],
        )

        # Generate observations
        n_time = 10
        key = random.PRNGKey(0)
        obs = random.normal(key, (n_time, params["n_obs_dim"]))

        _, filter_cov, mll = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)

        # Check each covariance matrix is positive definite
        for t in range(n_time):
            eigenvalues = jnp.linalg.eigvalsh(filter_cov[t])
            assert jnp.all(
                eigenvalues > -1e-8
            ), f"Non-PD covariance at t={t}: {eigenvalues}"

    @given(kalman_model_params(n_cont_states=2, n_obs_dim=2))
    @settings(max_examples=20, deadline=None)
    def test_filter_covariances_are_symmetric(self, params: dict) -> None:
        """Filter covariances should be symmetric."""
        init_mean, init_cov, A, Q, H, R = to_jax(
            params["init_mean"],
            params["init_cov"],
            params["A"],
            params["Q"],
            params["H"],
            params["R"],
        )

        n_time = 10
        key = random.PRNGKey(1)
        obs = random.normal(key, (n_time, params["n_obs_dim"]))

        _, filter_cov, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)

        for t in range(n_time):
            np.testing.assert_allclose(
                filter_cov[t], filter_cov[t].T, rtol=1e-10, atol=1e-14
            )

    @given(kalman_model_params(n_cont_states=2, n_obs_dim=2))
    @settings(max_examples=20, deadline=None)
    def test_marginal_likelihood_is_finite(self, params: dict) -> None:
        """Marginal log-likelihood should be a finite number."""
        init_mean, init_cov, A, Q, H, R = to_jax(
            params["init_mean"],
            params["init_cov"],
            params["A"],
            params["Q"],
            params["H"],
            params["R"],
        )

        n_time = 10
        key = random.PRNGKey(2)
        obs = random.normal(key, (n_time, params["n_obs_dim"]))

        _, _, mll = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)

        assert jnp.isfinite(mll), f"MLL is not finite: {mll}"


class TestKalmanSmootherProperties:
    """Property-based tests for the Kalman smoother."""

    @given(kalman_model_params(n_cont_states=2, n_obs_dim=2))
    @settings(max_examples=20, deadline=None)
    def test_smoother_covariances_are_positive_definite(self, params: dict) -> None:
        """Smoother covariances should be positive definite."""
        init_mean, init_cov, A, Q, H, R = to_jax(
            params["init_mean"],
            params["init_cov"],
            params["A"],
            params["Q"],
            params["H"],
            params["R"],
        )

        n_time = 10
        key = random.PRNGKey(3)
        obs = random.normal(key, (n_time, params["n_obs_dim"]))

        smoother_mean, smoother_cov, _, _ = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        for t in range(n_time):
            eigenvalues = jnp.linalg.eigvalsh(smoother_cov[t])
            # Allow small negative eigenvalues due to numerical precision
            assert jnp.all(
                eigenvalues > -1e-6
            ), f"Non-PD smoother covariance at t={t}: {eigenvalues}"

    @given(kalman_model_params(n_cont_states=2, n_obs_dim=2))
    @settings(max_examples=20, deadline=None)
    def test_smoother_reduces_uncertainty(self, params: dict) -> None:
        """Smoother covariance trace should be <= filter covariance trace.

        The smoother uses more information (past + future) than the filter
        (past only), so it should have equal or lower uncertainty.
        """
        init_mean, init_cov, A, Q, H, R = to_jax(
            params["init_mean"],
            params["init_cov"],
            params["A"],
            params["Q"],
            params["H"],
            params["R"],
        )

        n_time = 10
        key = random.PRNGKey(4)
        obs = random.normal(key, (n_time, params["n_obs_dim"]))

        _, filter_cov, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)
        _, smoother_cov, _, _ = kalman_smoother(init_mean, init_cov, obs, A, Q, H, R)

        # Smoother should have strictly lower uncertainty than filter at interior times
        for t in range(n_time - 1):  # Exclude last step where they're equal
            filter_trace = jnp.trace(filter_cov[t])
            smoother_trace = jnp.trace(smoother_cov[t])
            assert smoother_trace <= filter_trace, (
                f"Smoother trace {smoother_trace:.6f} > filter trace "
                f"{filter_trace:.6f} at t={t}"
            )

    @given(kalman_model_params(n_cont_states=2, n_obs_dim=2))
    @settings(max_examples=20, deadline=None)
    def test_last_smoother_equals_last_filter(self, params: dict) -> None:
        """At the last time step, smoother = filter (no future info)."""
        init_mean, init_cov, A, Q, H, R = to_jax(
            params["init_mean"],
            params["init_cov"],
            params["A"],
            params["Q"],
            params["H"],
            params["R"],
        )

        n_time = 10
        key = random.PRNGKey(5)
        obs = random.normal(key, (n_time, params["n_obs_dim"]))

        filter_mean, filter_cov, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)
        smoother_mean, smoother_cov, _, _ = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        np.testing.assert_allclose(smoother_mean[-1], filter_mean[-1], rtol=1e-5)
        np.testing.assert_allclose(smoother_cov[-1], filter_cov[-1], rtol=1e-5)


class TestKalmanMaximizationStepProperties:
    """Property-based tests for the M-step."""

    @given(kalman_model_params(n_cont_states=2, n_obs_dim=2))
    @settings(max_examples=20, deadline=None)
    def test_estimated_covariances_are_positive_definite(self, params: dict) -> None:
        """Estimated Q and R should be positive definite."""
        init_mean, init_cov, A, Q, H, R = to_jax(
            params["init_mean"],
            params["init_cov"],
            params["A"],
            params["Q"],
            params["H"],
            params["R"],
        )

        n_time = 50  # Need more data for stable estimation
        key = random.PRNGKey(6)
        obs = random.normal(key, (n_time, params["n_obs_dim"]))

        smoother_mean, smoother_cov, smoother_cross_cov, _ = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        A_est, H_est, Q_est, R_est, _, _ = kalman_maximization_step(
            obs, smoother_mean, smoother_cov, smoother_cross_cov
        )

        # Check Q is positive definite
        Q_eigenvalues = jnp.linalg.eigvalsh(Q_est)
        assert jnp.all(
            Q_eigenvalues > -1e-6
        ), f"Q not PD: eigenvalues = {Q_eigenvalues}"

        # Check R is positive definite
        R_eigenvalues = jnp.linalg.eigvalsh(R_est)
        assert jnp.all(
            R_eigenvalues > -1e-6
        ), f"R not PD: eigenvalues = {R_eigenvalues}"

    @given(kalman_model_params(n_cont_states=2, n_obs_dim=2))
    @settings(max_examples=20, deadline=None)
    def test_estimated_covariances_are_symmetric(self, params: dict) -> None:
        """Estimated Q and R should be symmetric."""
        init_mean, init_cov, A, Q, H, R = to_jax(
            params["init_mean"],
            params["init_cov"],
            params["A"],
            params["Q"],
            params["H"],
            params["R"],
        )

        n_time = 50
        key = random.PRNGKey(7)
        obs = random.normal(key, (n_time, params["n_obs_dim"]))

        smoother_mean, smoother_cov, smoother_cross_cov, _ = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        _, _, Q_est, R_est, _, init_cov_est = kalman_maximization_step(
            obs, smoother_mean, smoother_cov, smoother_cross_cov
        )

        np.testing.assert_allclose(Q_est, Q_est.T, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(R_est, R_est.T, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(init_cov_est, init_cov_est.T, rtol=1e-10, atol=1e-14)

    def test_inconsistent_sufficient_statistics_still_return_psd_covariances(
        self,
    ) -> None:
        """M-step covariances should remain PSD under slight statistic inconsistency."""
        obs = jnp.ones((3, 1))
        smoother_mean = jnp.ones((3, 1))
        smoother_cov = jnp.array([[[-0.9]], [[-0.9]], [[-0.9]]])
        smoother_cross_cov = jnp.array([[[0.1]], [[0.1]]])

        _, _, Q_est, R_est, _, _ = kalman_maximization_step(
            obs, smoother_mean, smoother_cov, smoother_cross_cov
        )

        eigvals_q = jnp.linalg.eigvalsh(Q_est)
        eigvals_r = jnp.linalg.eigvalsh(R_est)

        assert jnp.all(jnp.isfinite(Q_est))
        assert jnp.all(jnp.isfinite(R_est))
        assert jnp.min(eigvals_q) >= -1e-8
        assert jnp.min(eigvals_r) >= -1e-8


# --- Boundary Tests ---


class TestKalmanFilterBoundary:
    """Boundary tests for Kalman filter edge cases."""

    def test_single_timestep(self) -> None:
        """Kalman filter should handle single timestep correctly."""
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1)
        Q = jnp.eye(1) * 0.1
        H = jnp.eye(1)
        R = jnp.eye(1)
        obs = jnp.array([[0.5]])  # Single observation

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert filtered_mean.shape == (1, 1)
        assert filtered_cov.shape == (1, 1, 1)
        assert jnp.isfinite(mll)

    def test_two_timesteps(self) -> None:
        """Kalman filter should handle two timesteps correctly."""
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1)
        Q = jnp.eye(1) * 0.1
        H = jnp.eye(1)
        R = jnp.eye(1)
        obs = jnp.array([[0.5], [0.6]])

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert filtered_mean.shape == (2, 1)
        assert filtered_cov.shape == (2, 1, 1)
        assert jnp.isfinite(mll)

    @pytest.mark.slow
    def test_long_sequence(self) -> None:
        """Kalman filter should handle long sequences without numerical issues."""
        n_time = 1000
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1) * 0.99  # Stable
        Q = jnp.eye(1) * 0.1
        H = jnp.eye(1)
        R = jnp.eye(1)

        key = random.PRNGKey(42)
        obs = random.normal(key, (n_time, 1))

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert filtered_mean.shape == (n_time, 1)
        assert not jnp.any(jnp.isnan(filtered_mean))
        assert not jnp.any(jnp.isnan(filtered_cov))
        assert jnp.isfinite(mll)

    def test_high_dimensional_state(self) -> None:
        """Kalman filter should handle higher dimensional states."""
        n_cont_states = 10
        n_obs_dim = 5
        n_time = 20

        key = random.PRNGKey(123)
        k1, k2, k3 = random.split(key, 3)

        init_mean = jnp.zeros(n_cont_states)
        init_cov = jnp.eye(n_cont_states)

        # Create stable transition matrix
        A = jnp.eye(n_cont_states) * 0.9
        Q = jnp.eye(n_cont_states) * 0.1

        H = random.normal(k1, (n_obs_dim, n_cont_states)) * 0.5
        R = jnp.eye(n_obs_dim)

        obs = random.normal(k2, (n_time, n_obs_dim))

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert filtered_mean.shape == (n_time, n_cont_states)
        assert filtered_cov.shape == (n_time, n_cont_states, n_cont_states)
        assert jnp.isfinite(mll)

    def test_very_small_process_noise(self) -> None:
        """Kalman filter should handle very small process noise."""
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1)
        Q = jnp.eye(1) * 1e-10  # Very small
        H = jnp.eye(1)
        R = jnp.eye(1)
        obs = jnp.array([[0.5], [0.6], [0.7]])

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert not jnp.any(jnp.isnan(filtered_mean))
        assert not jnp.any(jnp.isnan(filtered_cov))

    def test_very_large_measurement_noise(self) -> None:
        """Kalman filter should handle very large measurement noise."""
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1)
        Q = jnp.eye(1) * 0.1
        H = jnp.eye(1)
        R = jnp.eye(1) * 1e6  # Very large
        obs = jnp.array([[0.5], [0.6], [0.7]])

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        # With large R, filter should barely update from prior
        assert not jnp.any(jnp.isnan(filtered_mean))
        assert not jnp.any(jnp.isnan(filtered_cov))


class TestKalmanSmootherBoundary:
    """Boundary tests for Kalman smoother edge cases."""

    def test_single_timestep(self) -> None:
        """Kalman smoother should handle single timestep correctly."""
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1)
        Q = jnp.eye(1) * 0.1
        H = jnp.eye(1)
        R = jnp.eye(1)
        obs = jnp.array([[0.5]])  # Single observation

        smoother_mean, smoother_cov, smoother_cross_cov, mll = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert smoother_mean.shape == (1, 1)
        assert smoother_cov.shape == (1, 1, 1)
        assert smoother_cross_cov.shape == (0, 1, 1)  # No cross-cov for single step
        assert jnp.isfinite(mll)

    def test_two_timesteps(self) -> None:
        """Kalman smoother should handle two timesteps correctly."""
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1)
        Q = jnp.eye(1) * 0.1
        H = jnp.eye(1)
        R = jnp.eye(1)
        obs = jnp.array([[0.5], [0.6]])

        smoother_mean, smoother_cov, smoother_cross_cov, mll = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert smoother_mean.shape == (2, 1)
        assert smoother_cov.shape == (2, 1, 1)
        assert smoother_cross_cov.shape == (1, 1, 1)
        assert jnp.isfinite(mll)

    @pytest.mark.slow
    def test_long_sequence(self) -> None:
        """Kalman smoother should handle long sequences without numerical issues."""
        n_time = 1000
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1) * 0.99
        Q = jnp.eye(1) * 0.1
        H = jnp.eye(1)
        R = jnp.eye(1)

        key = random.PRNGKey(42)
        obs = random.normal(key, (n_time, 1))

        smoother_mean, smoother_cov, smoother_cross_cov, mll = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert smoother_mean.shape == (n_time, 1)
        assert not jnp.any(jnp.isnan(smoother_mean))
        assert not jnp.any(jnp.isnan(smoother_cov))
        assert jnp.isfinite(mll)


# --- Input Handling Tests ---


class TestKalmanFilterInputHandling:
    """Tests for Kalman filter behavior with edge case inputs."""

    def test_handles_zero_mean_observations(self) -> None:
        """Filter should handle observations centered at zero."""
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1)
        Q = jnp.eye(1) * 0.1
        H = jnp.eye(1)
        R = jnp.eye(1)
        obs = jnp.zeros((10, 1))

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert not jnp.any(jnp.isnan(filtered_mean))
        assert jnp.isfinite(mll)

    def test_handles_large_observations(self) -> None:
        """Filter should handle large observation values."""
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(1)
        A = jnp.eye(1)
        Q = jnp.eye(1) * 0.1
        H = jnp.eye(1)
        R = jnp.eye(1)
        obs = jnp.array([[1e6], [1e6], [1e6]])

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert not jnp.any(jnp.isnan(filtered_mean))
        # Mean should track towards large observations
        assert jnp.abs(filtered_mean[-1, 0]) > 1e4

    def test_handles_identity_observation_matrix(self) -> None:
        """Filter should work with identity observation matrix."""
        n_states = 3
        init_mean = jnp.zeros(n_states)
        init_cov = jnp.eye(n_states)
        A = jnp.eye(n_states) * 0.9
        Q = jnp.eye(n_states) * 0.1
        H = jnp.eye(n_states)
        R = jnp.eye(n_states)

        key = random.PRNGKey(0)
        obs = random.normal(key, (10, n_states))

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert filtered_mean.shape == (10, n_states)
        assert jnp.isfinite(mll)

    def test_handles_partial_observation(self) -> None:
        """Filter should work when observing fewer dimensions than state."""
        n_states = 4
        n_obs = 2
        init_mean = jnp.zeros(n_states)
        init_cov = jnp.eye(n_states)
        A = jnp.eye(n_states) * 0.9
        Q = jnp.eye(n_states) * 0.1
        H = jnp.zeros((n_obs, n_states))
        H = H.at[0, 0].set(1.0)
        H = H.at[1, 2].set(1.0)  # Observe states 0 and 2
        R = jnp.eye(n_obs)

        key = random.PRNGKey(0)
        obs = random.normal(key, (10, n_obs))

        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert filtered_mean.shape == (10, n_states)
        assert jnp.isfinite(mll)


# --- Mathematical Correctness Tests ---


def _make_asymmetric_stable_A(n: int, seed: int = 0) -> jnp.ndarray:
    """Create an asymmetric stable transition matrix with distinct eigenvalues."""
    key = random.PRNGKey(seed)
    # Random eigenvectors (non-orthogonal)
    V = random.normal(key, (n, n)) * 0.3 + jnp.eye(n)
    # Distinct eigenvalues in (0.5, 0.95)
    eigs = jnp.linspace(0.5, 0.95, n)
    A = V @ jnp.diag(eigs) @ jnp.linalg.inv(V)
    return A


def _simulate_from_model(
    A: jnp.ndarray,
    Q: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    n_time: int,
    seed: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate observations and states from a linear Gaussian model."""
    key = random.PRNGKey(seed)
    n_state = A.shape[0]
    n_obs = H.shape[0]

    states = np.zeros((n_time, n_state))
    obs = np.zeros((n_time, n_obs))

    key_init, key = random.split(key)
    states[0] = np.array(random.multivariate_normal(key_init, init_mean, init_cov))
    for t in range(n_time):
        if t > 0:
            key_proc, key = random.split(key)
            states[t] = np.array(A) @ states[t - 1] + np.array(
                random.multivariate_normal(key_proc, jnp.zeros(n_state), Q)
            )
        key_obs, key = random.split(key)
        obs[t] = np.array(H) @ states[t] + np.array(
            random.multivariate_normal(key_obs, jnp.zeros(n_obs), R)
        )

    return jnp.array(obs), jnp.array(states)


class TestKalmanFilterMathCorrectness:
    """Tests verifying mathematical correctness of the Kalman filter.

    Uses asymmetric A matrices and independent reference implementations
    to catch transpose, indexing, and formula errors.
    """

    def test_filter_steady_state_covariance_matches_dare(self) -> None:
        """Filter covariance should converge to the DARE solution (scipy)."""
        n_state, n_obs = 3, 2
        A = _make_asymmetric_stable_A(n_state, seed=0)
        Q = jnp.eye(n_state) * 0.1
        H = jnp.array([[1.0, 0.3, -0.1], [0.0, 0.8, 0.5]])
        R = jnp.eye(n_obs) * 0.5

        # scipy DARE: different algorithm (Schur decomposition of symplectic pencil)
        # Solves: P = A P A^T + Q - A P H^T (H P H^T + R)^{-1} H P A^T
        P_dare = jnp.array(
            solve_discrete_are(np.array(A).T, np.array(H).T, np.array(Q), np.array(R))
        )

        init_mean = jnp.zeros(n_state)
        init_cov = jnp.eye(n_state)
        obs = random.normal(random.PRNGKey(42), (500, n_obs))

        _, filtered_cov, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)

        # The filter's predicted covariance converges to the DARE solution.
        P_pred_last = A @ filtered_cov[-1] @ A.T + Q
        np.testing.assert_allclose(
            P_pred_last,
            P_dare,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Filter predicted covariance should converge to DARE solution",
        )

    def test_filter_matches_numpy_reference(self) -> None:
        """Filter should match an independent numpy implementation."""
        n_state, n_obs = 3, 2
        A = _make_asymmetric_stable_A(n_state, seed=1)
        Q = jnp.eye(n_state) * 0.2
        H = jnp.array([[1.0, -0.5, 0.2], [0.3, 1.0, -0.1]])
        R = jnp.eye(n_obs) * 0.5
        init_mean = jnp.zeros(n_state)
        init_cov = jnp.eye(n_state)
        obs = random.normal(random.PRNGKey(99), (20, n_obs))

        # Reference: plain numpy predict-update loop
        # Library convention: every step does predict-then-update, including t=0.
        # init_mean/init_cov represent state at t=0; first obs is at t=1.
        m = np.array(init_mean)
        P = np.array(init_cov)
        A_np, Q_np, H_np, R_np = (np.array(A), np.array(Q), np.array(H), np.array(R))
        obs_np = np.array(obs)
        ref_means, ref_covs = [], []
        for t in range(20):
            # Predict
            m = A_np @ m
            P = A_np @ P @ A_np.T + Q_np
            # Update
            S = H_np @ P @ H_np.T + R_np
            K = P @ H_np.T @ np.linalg.inv(S)
            v = obs_np[t] - H_np @ m
            m = m + K @ v
            IKH = np.eye(n_state) - K @ H_np
            P = IKH @ P @ IKH.T + K @ R_np @ K.T
            P = 0.5 * (P + P.T)
            ref_means.append(m.copy())
            ref_covs.append(P.copy())
        ref_means = np.array(ref_means)
        ref_covs = np.array(ref_covs)

        filtered_mean, filtered_cov, _ = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        np.testing.assert_allclose(
            filtered_mean,
            ref_means,
            rtol=1e-8,
            atol=1e-10,
            err_msg="Filter means should match numpy reference",
        )
        np.testing.assert_allclose(
            filtered_cov,
            ref_covs,
            rtol=1e-8,
            atol=1e-10,
            err_msg="Filter covariances should match numpy reference",
        )

    def test_1d_kalman_steady_state_analytical(self) -> None:
        """1D random walk: filter cov should converge to closed-form solution."""
        q, r = 0.1, 1.0
        # Steady-state filtered cov satisfies: P = (P + q) * r / (P + q + r)
        # Solving: P^2 + P*q - q*r = 0
        # P_filtered = (-q + sqrt(q^2 + 4*q*r)) / 2
        P_filtered_inf = (-q + np.sqrt(q**2 + 4 * q * r)) / 2

        A = jnp.array([[1.0]])
        Q = jnp.array([[q]])
        H = jnp.array([[1.0]])
        R = jnp.array([[r]])
        init_mean = jnp.array([0.0])
        init_cov = jnp.array([[1.0]])
        obs = random.normal(random.PRNGKey(0), (200, 1))

        _, filtered_cov, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)

        np.testing.assert_allclose(
            filtered_cov[-1, 0, 0],
            P_filtered_inf,
            rtol=1e-4,
            err_msg="1D filter should converge to analytical steady state",
        )

    def test_filter_innovations_are_white(self) -> None:
        """Normalized innovations should have near-zero lag-1 autocorrelation.

        Uses fixed seed and 5-sigma Bartlett bound to avoid flakiness.
        """
        n_state, n_obs = 2, 2
        A = _make_asymmetric_stable_A(n_state, seed=3)
        Q = jnp.eye(n_state) * 0.1
        H = jnp.array([[1.0, 0.3], [-0.2, 0.9]])
        R = jnp.eye(n_obs) * 0.5
        init_mean = jnp.zeros(n_state)
        init_cov = jnp.eye(n_state)
        n_time = 2000

        obs, _ = _simulate_from_model(A, Q, H, R, init_mean, init_cov, n_time, seed=0)
        filtered_mean, filtered_cov, _ = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        # Compute innovations: v_t = y_t - H @ m_{t|t-1}
        # Library predicts at every step: m_{t|t-1} = A @ m_{t-1|t-1}
        # For t=0: m_{0|-1} = A @ init_mean
        innovations = np.zeros((n_time, n_obs))
        pred_mean_0 = A @ init_mean
        innovations[0] = np.array(obs[0] - H @ pred_mean_0)
        for t in range(1, n_time):
            pred_mean = A @ filtered_mean[t - 1]
            innovations[t] = np.array(obs[t] - H @ pred_mean)

        # Normalize by innovation covariance
        norm_innov = np.zeros_like(innovations)
        P_pred_0 = np.array(A @ init_cov @ A.T + Q)
        S0 = np.array(H) @ P_pred_0 @ np.array(H).T + np.array(R)
        S0 = 0.5 * (S0 + S0.T)
        L0 = np.linalg.cholesky(S0)
        norm_innov[0] = np.linalg.solve(L0, innovations[0])
        for t in range(1, n_time):
            P_pred = np.array(A @ filtered_cov[t - 1] @ A.T + Q)
            S = np.array(H) @ P_pred @ np.array(H).T + np.array(R)
            S = 0.5 * (S + S.T)
            L = np.linalg.cholesky(S)
            norm_innov[t] = np.linalg.solve(L, innovations[t])

        # Lag-1 autocorrelation for each component.
        # Bartlett's formula: Var(r_k) ≈ 1/N for white noise.
        # Using 5-sigma bound (p < 3e-7) to avoid flaky tests.
        bartlett_bound = 5.0 / np.sqrt(n_time)
        for d in range(n_obs):
            v = norm_innov[:, d]
            autocorr = np.corrcoef(v[:-1], v[1:])[0, 1]
            assert abs(autocorr) < bartlett_bound, (
                f"Innovation dim {d} lag-1 autocorrelation {autocorr:.4f} "
                f"exceeds 5-sigma bound {bartlett_bound:.4f}"
            )


class TestKalmanSmootherMathCorrectness:
    """Tests verifying mathematical correctness of the RTS smoother."""

    def test_smoother_cross_cov_identity_asymmetric_A(self) -> None:
        """Cross-covariance should satisfy P_{t,t+1|T} = G_t @ P_{t+1|T}.

        G_t is recomputed independently from filter outputs, creating a
        three-way consistency check between filter_cov, smoother_cov, cross_cov.
        """
        n_state, n_obs = 3, 2
        A = _make_asymmetric_stable_A(n_state, seed=5)
        Q = jnp.eye(n_state) * 0.15
        H = jnp.array([[1.0, 0.3, -0.1], [0.0, 0.8, 0.5]])
        R = jnp.eye(n_obs) * 0.5
        init_mean = jnp.zeros(n_state)
        init_cov = jnp.eye(n_state)

        obs = random.normal(random.PRNGKey(123), (50, n_obs))
        filtered_mean, filtered_cov, _ = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )
        _, smoother_cov, smoother_cross_cov, _ = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        for t in range(49):
            P_pred = A @ filtered_cov[t] @ A.T + Q
            P_pred_sym = 0.5 * (P_pred + P_pred.T)
            G_t = filtered_cov[t] @ A.T @ jnp.linalg.inv(P_pred_sym)

            expected_cross_cov = G_t @ smoother_cov[t + 1]
            np.testing.assert_allclose(
                smoother_cross_cov[t],
                expected_cross_cov,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Cross-cov identity failed at t={t}",
            )

    def test_smoother_mean_satisfies_rts_recursion(self) -> None:
        """Smoother mean should satisfy the RTS backward recursion.

        m_{t|T} = m_{t|t} + G_t(m_{t+1|T} - A m_{t|t}).
        """
        n_state, n_obs = 3, 2
        A = _make_asymmetric_stable_A(n_state, seed=7)
        Q = jnp.eye(n_state) * 0.1
        H = jnp.array([[1.0, 0.0, 0.5], [-0.3, 1.0, 0.0]])
        R = jnp.eye(n_obs) * 0.3
        init_mean = jnp.zeros(n_state)
        init_cov = jnp.eye(n_state)

        obs = random.normal(random.PRNGKey(77), (30, n_obs))
        filtered_mean, filtered_cov, _ = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )
        smoother_mean, _, _, _ = kalman_smoother(init_mean, init_cov, obs, A, Q, H, R)

        for t in range(29):
            P_pred = A @ filtered_cov[t] @ A.T + Q
            P_pred_sym = 0.5 * (P_pred + P_pred.T)
            G_t = filtered_cov[t] @ A.T @ jnp.linalg.inv(P_pred_sym)

            expected = filtered_mean[t] + G_t @ (
                smoother_mean[t + 1] - A @ filtered_mean[t]
            )
            np.testing.assert_allclose(
                smoother_mean[t],
                expected,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"RTS recursion failed at t={t}",
            )


class TestKalmanEMMonotonicity:
    """Tests verifying EM log-likelihood monotonicity."""

    def test_em_log_likelihood_monotonic_asymmetric_A(self) -> None:
        """Linear Gaussian EM: log-likelihood must be non-decreasing."""
        n_state, n_obs = 2, 2
        A_true = _make_asymmetric_stable_A(n_state, seed=15)
        Q_true = jnp.eye(n_state) * 0.2
        H_true = jnp.array([[1.0, 0.3], [-0.2, 0.9]])
        R_true = jnp.eye(n_obs) * 0.5
        init_mean_true = jnp.zeros(n_state)
        init_cov_true = jnp.eye(n_state)

        obs, _ = _simulate_from_model(
            A_true, Q_true, H_true, R_true, init_mean_true, init_cov_true, 200, seed=42
        )

        # Start from perturbed parameters
        A = jnp.eye(n_state) * 0.5
        Q = jnp.eye(n_state)
        H = jnp.eye(n_obs, n_state)
        R = jnp.eye(n_obs) * 2.0
        init_mean = jnp.zeros(n_state)
        init_cov = jnp.eye(n_state) * 2.0

        log_likelihoods = []
        for _ in range(15):
            _, _, mll = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)
            log_likelihoods.append(float(mll))

            sm, sc, scc, _ = kalman_smoother(init_mean, init_cov, obs, A, Q, H, R)
            A, H, Q, R, init_mean, init_cov = kalman_maximization_step(obs, sm, sc, scc)

        for i in range(1, len(log_likelihoods)):
            assert log_likelihoods[i] >= log_likelihoods[i - 1] - 1e-6, (
                f"EM monotonicity violated: LL[{i}]={log_likelihoods[i]:.6f} "
                f"< LL[{i-1}]={log_likelihoods[i-1]:.6f}"
            )


class TestKalmanMStepMathCorrectness:
    """Tests verifying the EM M-step computes correct parameter estimates."""

    def test_mstep_A_satisfies_normal_equations(self) -> None:
        """M-step output A should satisfy A @ gamma1 = beta.

        gamma1 and beta are recomputed from smoother outputs using numpy.
        """
        n_state, n_obs = 3, 2
        A = _make_asymmetric_stable_A(n_state, seed=10)
        Q = jnp.eye(n_state) * 0.2
        H = jnp.array([[1.0, 0.5, -0.2], [0.0, 0.8, 0.3]])
        R = jnp.eye(n_obs) * 0.5
        init_mean = jnp.zeros(n_state)
        init_cov = jnp.eye(n_state)

        obs, _ = _simulate_from_model(A, Q, H, R, init_mean, init_cov, 200, seed=42)
        sm, sc, scc, _ = kalman_smoother(init_mean, init_cov, obs, A, Q, H, R)

        A_est, _, _, _, _, _ = kalman_maximization_step(obs, sm, sc, scc)

        # Recompute gamma1 and beta from smoother outputs (plain numpy)
        sm_np, sc_np, scc_np = np.array(sm), np.array(sc), np.array(scc)

        gamma1 = np.zeros((n_state, n_state))
        for t in range(199):
            gamma1 += sc_np[t] + np.outer(sm_np[t], sm_np[t])

        beta = np.zeros((n_state, n_state))
        for t in range(199):
            beta += scc_np[t].T + np.outer(sm_np[t + 1], sm_np[t])

        lhs = np.array(A_est) @ gamma1
        np.testing.assert_allclose(
            lhs,
            beta,
            rtol=1e-4,
            atol=1e-7,
            err_msg="M-step A should satisfy normal equations A @ gamma1 = beta",
        )


class TestKalmanNumericalStability:
    """Tests for numerical stability under adversarial conditions."""

    def test_filter_nearly_unobservable_state(self) -> None:
        """Nearly unobservable dimension should retain large uncertainty."""
        n_state, n_obs = 3, 2
        A = _make_asymmetric_stable_A(n_state, seed=20)
        Q = jnp.eye(n_state) * 0.1
        H = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1e-8]])
        R = jnp.eye(n_obs) * 0.5
        init_mean = jnp.zeros(n_state)
        init_cov = jnp.eye(n_state)

        obs = random.normal(random.PRNGKey(42), (100, n_obs))
        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert jnp.all(jnp.isfinite(filtered_mean)), "Means should be finite"
        assert jnp.all(jnp.isfinite(filtered_cov)), "Covariances should be finite"
        assert jnp.isfinite(mll), "MLL should be finite"

        for t in range(100):
            eigvals = jnp.linalg.eigvalsh(filtered_cov[t])
            assert jnp.all(eigvals > -1e-8), f"Cov not PSD at t={t}"

        assert (
            filtered_cov[-1, 2, 2] > filtered_cov[-1, 0, 0]
        ), "Unobserved state should have larger uncertainty"

    def test_filter_nearly_unstable_dynamics(self) -> None:
        """Spectral radius near 1: filter should stay finite for 500 steps."""
        n_state = 2
        V = jnp.array([[1.0, 0.3], [-0.1, 1.0]])
        A = V @ jnp.diag(jnp.array([0.999, 0.998])) @ jnp.linalg.inv(V)
        Q = jnp.eye(n_state) * 0.001
        H = jnp.eye(n_state)
        R = jnp.eye(n_state) * 1.0
        init_mean = jnp.zeros(n_state)
        init_cov = jnp.eye(n_state)

        obs = random.normal(random.PRNGKey(42), (500, n_state))
        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean, init_cov, obs, A, Q, H, R
        )

        assert jnp.all(jnp.isfinite(filtered_mean)), "Means should be finite"
        assert jnp.all(jnp.isfinite(filtered_cov)), "Covariances should be finite"
        assert jnp.isfinite(mll), "MLL should be finite"

        for t in range(500):
            eigvals = jnp.linalg.eigvalsh(filtered_cov[t])
            assert jnp.all(eigvals > -1e-8), f"Cov not PSD at t={t}"
