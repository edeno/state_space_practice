# ruff: noqa: E402

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import Array, random
from scipy.linalg import solve_discrete_are

from state_space_practice.exceptions import StateSpaceWarning
from state_space_practice.kalman import (
    InitialStatePrior,
    _gain_solve,
    _kalman_smoother_update,
    joseph_form_update,
    kalman_filter,
    kalman_maximization_step,
    kalman_measurement_update,
    kalman_smoother,
    measurement_cov_residual_form,
    parallel_kalman_smoother,
    process_cov_residual_form,
    psd_solve,
    smooth_initial_state,
    smooth_initial_state_with_cross_cov,
    standard_kalman_gain,
    sum_of_outer_products,
    symmetrize,
    woodbury_kalman_gain,
)
from state_space_practice.tests.conftest import (
    kalman_model_params,
    positive_definite_matrices,
    to_jax,
)

# The x_1-prior M-step (initial_state_prior=None) is deprecated; tests of its
# formulas assert the deprecation warning.
LEGACY_PRIOR = r"initial_state_prior=None"

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


def test_kalman_measurement_update_1d() -> None:
    """Tests kalman_measurement_update against analytic 1D scalar formulas."""
    prior_mean = jnp.array([2.0])
    prior_cov = jnp.array([[4.0]])
    obs = jnp.array([3.0])
    H = jnp.array([[1.0]])
    R = jnp.array([[1.0]])

    post_mean, post_cov, mll = kalman_measurement_update(
        prior_mean, prior_cov, obs, H, R
    )

    # Analytic: K = P*H' / (H*P*H' + R) = 4/(4+1) = 0.8
    # post_mean = prior + K*(obs - H*prior) = 2 + 0.8*(3-2) = 2.8
    # post_cov (Joseph) = (1-KH)*P*(1-KH)' + K*R*K' = 0.2*4*0.2 + 0.8*1*0.8 = 0.16+0.64 = 0.8
    np.testing.assert_allclose(post_mean, jnp.array([2.8]), atol=1e-6)
    np.testing.assert_allclose(post_cov, jnp.array([[0.8]]), atol=1e-6)

    # posterior_cov should be PSD
    assert jnp.all(jnp.linalg.eigvalsh(post_cov) > 0)

    # marginal_log_likelihood should match scipy
    from scipy.stats import multivariate_normal as mvn

    expected_mll = mvn.logpdf(
        np.array([3.0]), mean=np.array([2.0]), cov=np.array([[5.0]])
    )
    np.testing.assert_allclose(mll, expected_mll, atol=1e-5)


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


def test_sum_of_outer_products_rejects_mismatched_time_axis() -> None:
    x = jnp.ones((3, 2))
    y = jnp.ones((4, 2))

    with pytest.raises(ValueError, match="same leading time"):
        sum_of_outer_products(x, y)


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
def kalman_m_step_test_data() -> tuple[Array, Array, Array, Array, Array, Array, Array]:
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

    with pytest.warns(DeprecationWarning, match=LEGACY_PRIOR):
        (
            A_est,
            H_est,
            Q_est,
            R_est,
            init_mean_est,
            init_cov_est,
        ) = kalman_maximization_step(
            obs, smoother_mean, smoother_cov, smoother_cross_cov
        )

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


def test_kalman_maximization_step_rejects_one_timestep() -> None:
    """Transition dynamics are not identifiable from a single smoothed state."""
    obs = jnp.zeros((1, 2))
    smoother_mean = jnp.zeros((1, 2))
    smoother_cov = jnp.eye(2)[None]
    smoother_cross_cov = jnp.zeros((0, 2, 2))

    prior = InitialStatePrior(jnp.zeros(2), jnp.eye(2), jnp.eye(2), jnp.eye(2))
    with pytest.raises(ValueError, match="at least 2 time steps"):
        kalman_maximization_step(
            obs, smoother_mean, smoother_cov, smoother_cross_cov, prior
        )


@pytest.fixture(scope="module")
def multi_dim_model() -> tuple[
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
            assert jnp.all(eigenvalues > -1e-8), (
                f"Non-PD covariance at t={t}: {eigenvalues}"
            )

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
            assert jnp.all(eigenvalues > -1e-6), (
                f"Non-PD smoother covariance at t={t}: {eigenvalues}"
            )

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
            obs,
            smoother_mean,
            smoother_cov,
            smoother_cross_cov,
            InitialStatePrior(init_mean, init_cov, A, Q),
        )

        # Check Q is positive definite
        Q_eigenvalues = jnp.linalg.eigvalsh(Q_est)
        assert jnp.all(Q_eigenvalues > -1e-6), (
            f"Q not PD: eigenvalues = {Q_eigenvalues}"
        )

        # Check R is positive definite
        R_eigenvalues = jnp.linalg.eigvalsh(R_est)
        assert jnp.all(R_eigenvalues > -1e-6), (
            f"R not PD: eigenvalues = {R_eigenvalues}"
        )

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
            obs,
            smoother_mean,
            smoother_cov,
            smoother_cross_cov,
            InitialStatePrior(init_mean, init_cov, A, Q),
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

        with pytest.warns(DeprecationWarning, match=LEGACY_PRIOR):
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
            prior = InitialStatePrior(init_mean, init_cov, A, Q)
            A, H, Q, R, init_mean, init_cov = kalman_maximization_step(
                obs, sm, sc, scc, prior
            )

        for i in range(1, len(log_likelihoods)):
            assert log_likelihoods[i] >= log_likelihoods[i - 1] - 1e-6, (
                f"EM monotonicity violated: LL[{i}]={log_likelihoods[i]:.6f} "
                f"< LL[{i - 1}]={log_likelihoods[i - 1]:.6f}"
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

        with pytest.warns(DeprecationWarning, match=LEGACY_PRIOR):
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


# Gain-solve test matrices: well conditioned, and a float32 matrix singular
# to within 4 eps, whose Cholesky fails at an eps-relative shift.
_WELL_CONDITIONED_COV = [[3.0, 0.2], [0.2, 1.0]]
_F32_EPS = float(np.finfo(np.float32).eps)
_NEARLY_SINGULAR_F32_COV = [[1.0, 1.0], [1.0, 1.0 - 4 * _F32_EPS]]


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

        assert filtered_cov[-1, 2, 2] > filtered_cov[-1, 0, 0], (
            "Unobserved state should have larger uncertainty"
        )

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

    @pytest.mark.parametrize(
        "A, init_var, obs_var, obs",
        [
            (
                [[0.6, 0.8], [0.3, 0.4]],
                1.0,
                0.1,
                [[1.0, 2.0], [0.5, -1.0], [0.3, 0.2]],
            ),
            # Diagonals 50x apart: float32 round-off in the singular predicted
            # covariance exceeds an eps * A_ii shift of the small entry.
            ([[0.56, 0.56], [0.08, 0.08]], 2.0**-10, 2.0**-10, np.zeros((3, 2))),
        ],
    )
    def test_float32_smoother_rank_deficient_dynamics_zero_process_noise(
        self, A, init_var, obs_var, obs
    ) -> None:
        """A rank-1 A with Q = 0 makes every predicted covariance singular, so
        the RTS gain solve relies entirely on its stabilising shift. In float32
        the shift must survive rounding: the smoother must stay finite and
        agree with the float64 smoother to float32 accuracy."""
        args = (
            np.zeros(2),
            init_var * np.eye(2),
            np.asarray(obs, dtype=float),
            np.asarray(A),
            np.zeros((2, 2)),
            np.eye(2),
            obs_var * np.eye(2),
        )
        ref_mean, ref_cov, _, _ = kalman_smoother(*(jnp.asarray(a) for a in args))
        mean32, cov32, _, _ = kalman_smoother(
            *(jnp.asarray(a, dtype=jnp.float32) for a in args)
        )

        assert mean32.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(mean32)) and jnp.all(jnp.isfinite(cov32))
        scale = float(np.max(np.abs(ref_cov)))
        np.testing.assert_allclose(mean32, ref_mean, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(cov32, ref_cov, rtol=1e-3, atol=1e-3 * scale)

    def test_float32_smoother_gradient_through_stabilised_gain_solve(self) -> None:
        """The RTS gain solve needs its larger shift here; reverse mode must not
        pass through the factorization that failed. With a zero initial mean
        and zero observations the smoothed means are identically 0, so their
        gradient with respect to A is 0."""
        f32 = jnp.float32
        A = jnp.asarray([[0.56, 0.56], [0.08, 0.08]], dtype=f32)
        small = (2.0**-10) * jnp.eye(2, dtype=f32)

        def summed_means(A):
            means, _, _, _ = kalman_smoother(
                jnp.zeros(2, dtype=f32),
                small,
                jnp.zeros((3, 2), dtype=f32),
                A,
                jnp.zeros((2, 2), dtype=f32),
                jnp.eye(2, dtype=f32),
                small,
            )
            return jnp.sum(means)

        grad = jax.grad(summed_means)(A)
        assert jnp.all(jnp.isfinite(grad))
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)

    @pytest.mark.parametrize("needs_retry", [False, True])
    @pytest.mark.parametrize("vector_rhs", [False, True])
    def test_gain_solve_nonzero_derivatives(self, needs_retry, vector_rhs) -> None:
        """Scaling cov by exp(t) and rhs by exp(2t) scales the solution by exp(t).

        Both covariance and rhs derivatives must contribute, including the
        covariance-dependent diagonal shift and the retry under float32.
        """
        dtype = jnp.float32 if needs_retry else jnp.float64
        cov = jnp.asarray(
            _NEARLY_SINGULAR_F32_COV if needs_retry else _WELL_CONDITIONED_COV,
            dtype=dtype,
        )
        rhs = jnp.asarray(
            [1.0, 2.0] if vector_rhs else [[1.0, 2.0], [2.0, -1.0]], dtype=dtype
        )
        weights = jnp.arange(1, rhs.size + 1, dtype=dtype).reshape(rhs.shape)
        if needs_retry:
            # Guard: this case actually requires the larger shift.
            assert not jnp.all(
                jnp.isfinite(psd_solve(cov, rhs, relative_boost=_F32_EPS))
            )

        def scaled_solution(t):
            return _gain_solve(jnp.exp(t) * cov, jnp.exp(2 * t) * rhs)

        zero, one = jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype)
        solution, tangent = jax.jvp(scaled_solution, (zero,), (one,))
        np.testing.assert_allclose(tangent, solution, rtol=2e-3, atol=1e-6)

        def loss(t):
            return jnp.sum(weights * scaled_solution(t))

        value, gradient = jax.value_and_grad(loss)(zero)
        np.testing.assert_allclose(gradient, value, rtol=2e-3, atol=1e-6)
        np.testing.assert_allclose(
            jax.grad(jax.grad(loss))(zero), value, rtol=2e-3, atol=1e-6
        )

    def test_gain_solve_retry_gradients_under_vmap(self) -> None:
        """A failed factor in one batch entry cannot contaminate its gradients."""
        covs = jnp.asarray(
            [_NEARLY_SINGULAR_F32_COV, _WELL_CONDITIONED_COV], dtype=jnp.float32
        )
        rhs = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
        weights = jnp.asarray([2.0, -1.0], dtype=jnp.float32)

        def loss(t, cov):
            return weights @ _gain_solve(jnp.exp(t) * cov, rhs)

        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(0, 0)))(
            jnp.zeros(2, dtype=jnp.float32), covs
        )
        np.testing.assert_allclose(gradients, -values, rtol=2e-3, atol=1e-6)

    @pytest.mark.parametrize("wide", ["cov", "rhs"])
    def test_gain_solve_mixed_dtypes(self, wide) -> None:
        """A float32 / float64 mix solves in the promoted dtype, eagerly, under
        jit and under grad, and matches the all-float64 solve."""
        cov64 = jnp.asarray(_WELL_CONDITIONED_COV)
        rhs64 = jnp.asarray([[1.0, 2.0], [2.0, -1.0]])
        cov = cov64 if wide == "cov" else cov64.astype(jnp.float32)
        rhs = rhs64 if wide == "rhs" else rhs64.astype(jnp.float32)
        # Reference: the same (rounded) values, all in float64.
        expected = _gain_solve(cov.astype(jnp.float64), rhs.astype(jnp.float64))

        # A float32 cov is stabilised at float32 precision (an eps32-relative
        # shift), so it matches the float64 solve to ~1e-7, not round-off.
        rtol = 1e-12 if cov.dtype == jnp.float64 else 1e-6
        for solve in (_gain_solve, jax.jit(_gain_solve)):
            solution = solve(cov, rhs)
            assert solution.dtype == jnp.float64
            np.testing.assert_allclose(solution, expected, rtol=rtol)

        def loss(scale):
            return jnp.sum(_gain_solve(scale * cov, rhs))

        # d/ds sum((s cov)^{-1} rhs) at s = 1 is -sum(cov^{-1} rhs).
        gradient = jax.grad(loss)(jnp.asarray(1.0, dtype=cov.dtype))
        np.testing.assert_allclose(gradient, -jnp.sum(expected), rtol=1e-6)

    def test_m_step_nearly_singular_float32_moments_float64_obs(self) -> None:
        """Identical float32 smoothed means (second moment rank-1 up to 1e-10)
        with float64 observations. The regression is solved in float64, but
        the second moment carries float32 rounding, so its stabilising shift
        must follow float32: H and R stay finite and H m reproduces the mean
        observation (the only identified direction)."""
        f32 = jnp.float32
        mean = jnp.asarray([0.86, 0.245], dtype=f32)
        _, H, _, R, _, _ = kalman_maximization_step(
            jnp.asarray([[0.0], [1.0], [2.0]]),
            jnp.tile(mean, (3, 1)),
            1e-10 * jnp.tile(jnp.eye(2, dtype=f32), (3, 1, 1)),
            jnp.zeros((2, 2, 2), dtype=f32),
            initial_state_prior=InitialStatePrior(
                jnp.zeros(2, dtype=f32),
                jnp.eye(2, dtype=f32),
                jnp.eye(2, dtype=f32),
                jnp.eye(2, dtype=f32),
            ),
        )
        assert jnp.all(jnp.isfinite(H)) and jnp.all(jnp.isfinite(R))
        np.testing.assert_allclose(H @ mean, [1.0], rtol=1e-3)

    def test_measurement_update_float32_inputs_float64_measurement_cov(self) -> None:
        """float32 state / observation with a float64 R: eager, jit and grad."""
        f32 = jnp.float32
        args64 = (
            jnp.zeros(2),
            jnp.asarray([[2.0, 0.5], [0.5, 1.0]]),
            jnp.asarray([1.0, -0.5]),
            jnp.eye(2),
            0.5 * jnp.eye(2),
        )
        mixed = tuple(a.astype(f32) for a in args64[:4]) + (args64[4],)
        expected = kalman_measurement_update(*args64)

        for update in (kalman_measurement_update, jax.jit(kalman_measurement_update)):
            for got, want in zip(update(*mixed), expected, strict=True):
                np.testing.assert_allclose(got, want, rtol=1e-6)

        def posterior_mean_sum(*args):
            return jnp.sum(kalman_measurement_update(*args)[0])

        # Gradient with respect to the prior covariance.
        grad_mixed = jax.grad(posterior_mean_sum, argnums=1)(*mixed)
        grad64 = jax.grad(posterior_mean_sum, argnums=1)(*args64)
        assert jnp.all(jnp.isfinite(grad_mixed))
        np.testing.assert_allclose(grad_mixed, grad64, rtol=1e-5)


# --- Parallel Kalman Smoother Tests ---


class TestParallelKalmanSmoother:
    """Tests for the parallel RTS smoother via associative scan."""

    @pytest.mark.slow  # full sequential + parallel smoother, T=500 (~5 s)
    def test_matches_sequential_smoother(self) -> None:
        """Parallel smoother should match sequential smoother to high precision."""
        key = random.PRNGKey(42)
        T, D = 500, 4
        A = 0.95 * jnp.eye(D) + 0.02 * random.normal(key, (D, D))
        Q = jnp.eye(D) * 0.1
        H = jnp.eye(D)
        R = jnp.eye(D) * 0.5
        init_mean = jnp.zeros(D)
        init_cov = jnp.eye(D)

        obs = random.normal(random.PRNGKey(0), (T, D))

        # Sequential
        seq_mean, seq_cov, seq_cross, mll = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        # Parallel: first run filter, then parallel smoother
        filt_mean, filt_cov, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)
        par_mean, par_cov, par_cross = parallel_kalman_smoother(
            filt_mean, filt_cov, A, Q
        )

        np.testing.assert_allclose(par_mean, seq_mean, atol=1e-5)
        np.testing.assert_allclose(par_cov, seq_cov, atol=1e-5)
        np.testing.assert_allclose(par_cross, seq_cross, atol=1e-5)

    def test_output_shapes(self) -> None:
        """Output shapes should be correct."""
        T, D = 100, 3
        filt_mean = jnp.zeros((T, D))
        filt_cov = jnp.broadcast_to(jnp.eye(D), (T, D, D))
        A = jnp.eye(D)
        Q = jnp.eye(D) * 0.1

        sm, sc, sx = parallel_kalman_smoother(filt_mean, filt_cov, A, Q)

        assert sm.shape == (T, D)
        assert sc.shape == (T, D, D)
        assert sx.shape == (T - 1, D, D)

    def test_single_timestep(self) -> None:
        """T=1: smoother output should equal filter output."""
        D = 2
        filt_mean = jnp.array([[1.0, 2.0]])
        filt_cov = jnp.eye(D)[None] * 0.5
        A = jnp.eye(D)
        Q = jnp.eye(D) * 0.1

        sm, sc, sx = parallel_kalman_smoother(filt_mean, filt_cov, A, Q)

        np.testing.assert_allclose(sm, filt_mean, atol=1e-10)
        np.testing.assert_allclose(sc, filt_cov, atol=1e-10)
        assert sx.shape == (0, D, D)

    def test_rejects_zero_timesteps(self) -> None:
        """T=0 should fail clearly instead of broadcasting to negative shapes."""
        D = 2
        filt_mean = jnp.zeros((0, D))
        filt_cov = jnp.zeros((0, D, D))
        with pytest.raises(ValueError, match="at least one time step"):
            parallel_kalman_smoother(filt_mean, filt_cov, jnp.eye(D), jnp.eye(D))

    def test_two_timesteps(self) -> None:
        """T=2: verify against closed-form RTS update."""
        D = 1
        init_mean = jnp.array([0.0])
        init_cov = jnp.eye(D)
        A = jnp.eye(D)
        Q = jnp.eye(D) * 0.1
        H = jnp.eye(D)
        R = jnp.eye(D)
        obs = jnp.array([[0.5], [0.6]])

        # Get sequential answer
        seq_mean, seq_cov, seq_cross, _ = kalman_smoother(
            init_mean, init_cov, obs, A, Q, H, R
        )

        # Get parallel answer
        filt_mean, filt_cov, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)
        par_mean, par_cov, par_cross = parallel_kalman_smoother(
            filt_mean, filt_cov, A, Q
        )

        np.testing.assert_allclose(par_mean, seq_mean, atol=1e-6)
        np.testing.assert_allclose(par_cov, seq_cov, atol=1e-6)
        np.testing.assert_allclose(par_cross, seq_cross, atol=1e-6)

    @pytest.mark.slow  # Kalman filter + parallel smoother pipeline (~4 s)
    def test_time_varying_transition(self) -> None:
        """Should handle per-timestep transition matrices."""
        T, D = 50, 2
        key = random.PRNGKey(7)
        init_mean = jnp.zeros(D)
        init_cov = jnp.eye(D)

        # Time-varying A: slowly changing
        A_base = 0.9 * jnp.eye(D)
        perturbations = 0.01 * random.normal(key, (T - 1, D, D))
        A_tv = A_base[None] + perturbations  # (T-1, D, D)

        Q = jnp.eye(D) * 0.1
        H = jnp.eye(D)
        R = jnp.eye(D) * 0.5

        # Simulate observations using first A
        obs = random.normal(random.PRNGKey(99), (T, D))

        # Run filter with first A (filter doesn't support TV, so use constant)
        filt_mean, filt_cov, _ = kalman_filter(
            init_mean, init_cov, obs, A_base, Q, H, R
        )

        # Parallel smoother with time-varying A
        sm, sc, sx = parallel_kalman_smoother(filt_mean, filt_cov, A_tv, Q)

        assert sm.shape == (T, D)
        assert sc.shape == (T, D, D)
        assert sx.shape == (T - 1, D, D)
        assert jnp.all(jnp.isfinite(sm))
        assert jnp.all(jnp.isfinite(sc))

    def test_time_varying_transition_matches_sequential_reference(self) -> None:
        """Parallel smoother with A_t/Q_t should match explicit RTS recursion."""
        T, D = 17, 3
        key = random.PRNGKey(31)
        k_mean, k_cov, k_A = random.split(key, 3)
        filt_mean = random.normal(k_mean, (T, D))
        cov_factors = random.normal(k_cov, (T, D, D))
        filt_cov = jnp.einsum("tij,tkj->tik", cov_factors, cov_factors)
        filt_cov = filt_cov + 0.5 * jnp.eye(D)[None]
        A_tv = 0.75 * jnp.eye(D)[None] + 0.03 * random.normal(k_A, (T - 1, D, D))
        q_scales = jnp.linspace(0.05, 0.2, T - 1)
        Q_tv = q_scales[:, None, None] * jnp.eye(D)[None]

        par_mean, par_cov, par_cross = parallel_kalman_smoother(
            filt_mean, filt_cov, A_tv, Q_tv
        )

        next_mean = filt_mean[-1]
        next_cov = filt_cov[-1]
        seq_means = []
        seq_covs = []
        seq_cross = []
        for t in reversed(range(T - 1)):
            next_mean, next_cov, cross = _kalman_smoother_update(
                next_mean,
                next_cov,
                filt_mean[t],
                filt_cov[t],
                Q_tv[t],
                A_tv[t],
            )
            seq_means.append(next_mean)
            seq_covs.append(next_cov)
            seq_cross.append(cross)

        seq_mean = jnp.concatenate(
            [jnp.stack(seq_means[::-1]), filt_mean[-1][None]], axis=0
        )
        seq_cov = jnp.concatenate(
            [jnp.stack(seq_covs[::-1]), filt_cov[-1][None]], axis=0
        )
        seq_cross = jnp.stack(seq_cross[::-1])

        np.testing.assert_allclose(par_mean, seq_mean, atol=1e-8)
        np.testing.assert_allclose(par_cov, seq_cov, atol=1e-8)
        np.testing.assert_allclose(par_cross, seq_cross, atol=1e-8)

    def test_rejects_bad_time_varying_parameter_shapes(self) -> None:
        T, D = 5, 2
        filt_mean = jnp.zeros((T, D))
        filt_cov = jnp.broadcast_to(jnp.eye(D), (T, D, D))

        with pytest.raises(ValueError, match="transition_matrix.*leading time axis"):
            parallel_kalman_smoother(
                filt_mean,
                filt_cov,
                jnp.broadcast_to(jnp.eye(D), (T, D, D)),
                jnp.eye(D),
            )

        with pytest.raises(ValueError, match="process_cov.*leading time axis"):
            parallel_kalman_smoother(
                filt_mean,
                filt_cov,
                jnp.eye(D),
                jnp.broadcast_to(jnp.eye(D), (T, D, D)),
            )

    def test_rejects_bad_filtered_covariance_shape(self) -> None:
        T, D = 5, 2
        filt_mean = jnp.zeros((T, D))
        filt_cov = jnp.broadcast_to(jnp.eye(D), (T - 1, D, D))

        with pytest.raises(ValueError, match="filtered_covariances"):
            parallel_kalman_smoother(filt_mean, filt_cov, jnp.eye(D), jnp.eye(D))

    def test_smoothed_covariances_psd(self) -> None:
        """Smoothed covariances should be positive semi-definite."""
        key = random.PRNGKey(123)
        T, D = 200, 3
        A = 0.9 * jnp.eye(D)
        Q = jnp.eye(D) * 0.1
        H = jnp.eye(D)
        R = jnp.eye(D) * 0.5
        init_mean = jnp.zeros(D)
        init_cov = jnp.eye(D)

        obs = random.normal(key, (T, D))
        filt_mean, filt_cov, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)
        _, sc, _ = parallel_kalman_smoother(filt_mean, filt_cov, A, Q)

        for t in range(T):
            eigvals = jnp.linalg.eigvalsh(sc[t])
            assert jnp.all(eigvals > -1e-8), f"Smoothed cov not PSD at t={t}"


# --- Woodbury Kalman Gain Tests ---


class TestWoodburyKalmanGain:
    """Tests for Woodbury-optimized Kalman gain computation."""

    def test_matches_standard_gain(self) -> None:
        """Woodbury gain should match standard gain for diagonal R."""
        key = random.PRNGKey(42)
        D_state, D_obs = 4, 50
        P = random.normal(key, (D_state, D_state))
        P = P @ P.T + 0.1 * jnp.eye(D_state)  # PSD
        H = random.normal(random.PRNGKey(1), (D_obs, D_state))
        R_diag = jnp.abs(random.normal(random.PRNGKey(2), (D_obs,))) + 0.1

        K_w, S_w, _ = woodbury_kalman_gain(P, H, R_diag)
        K_s, S_s = standard_kalman_gain(P, H, jnp.diag(R_diag))

        np.testing.assert_allclose(K_w, K_s, atol=1e-6)
        np.testing.assert_allclose(S_w, S_s, atol=1e-6)

    def test_many_neurons(self) -> None:
        """Should work correctly with 200 neurons and 4-dim state."""
        key = random.PRNGKey(99)
        D_state, D_obs = 4, 200
        P = jnp.eye(D_state) * 0.5
        H = random.normal(key, (D_obs, D_state)) * 0.1
        R_diag = jnp.ones(D_obs) * 0.3

        K_w, S_w, _ = woodbury_kalman_gain(P, H, R_diag)
        K_s, S_s = standard_kalman_gain(P, H, jnp.diag(R_diag))

        np.testing.assert_allclose(K_w, K_s, atol=1e-5)

    def test_rank_deficient_prior_cov_matches_standard_gain(self) -> None:
        """PSD but singular P should not NaN in the Woodbury helper."""
        key = random.PRNGKey(11)
        D_state, D_obs = 4, 30
        P = jnp.diag(jnp.array([1.0, 0.4, 0.0, 0.0]))
        H = random.normal(key, (D_obs, D_state)) * 0.2
        R_diag = jnp.linspace(0.2, 1.0, D_obs)

        K_w, S_w, S_inv_w = woodbury_kalman_gain(P, H, R_diag)
        K_s, S_s = standard_kalman_gain(P, H, jnp.diag(R_diag))

        assert jnp.all(jnp.isfinite(K_w))
        assert jnp.all(jnp.isfinite(S_inv_w))
        np.testing.assert_allclose(K_w, K_s, atol=1e-6)
        np.testing.assert_allclose(S_w, S_s, atol=1e-10)
        np.testing.assert_allclose(S_inv_w @ S_w, jnp.eye(D_obs), atol=1e-8)

    def test_jit_compiles(self) -> None:
        """Value validation should not make the helper unusable in scans/jit."""
        key = random.PRNGKey(17)
        D_state, D_obs = 3, 12
        P = jnp.eye(D_state) * 0.5
        H = random.normal(key, (D_obs, D_state)) * 0.1
        R_diag = jnp.ones(D_obs) * 0.3

        K_w, S_w, S_inv_w = jax.jit(woodbury_kalman_gain)(P, H, R_diag)
        K_s, S_s = standard_kalman_gain(P, H, jnp.diag(R_diag))

        np.testing.assert_allclose(K_w, K_s, atol=1e-8)
        np.testing.assert_allclose(S_w, S_s, atol=1e-10)
        np.testing.assert_allclose(S_inv_w @ S_w, jnp.eye(D_obs), atol=1e-8)

    def test_log_likelihood_matches(self) -> None:
        """Innovation log-likelihood should match between methods."""
        key = random.PRNGKey(7)
        D_state, D_obs = 3, 20
        P = jnp.eye(D_state)
        H = random.normal(key, (D_obs, D_state))
        R_diag = jnp.ones(D_obs) * 0.5

        _, S_w, _ = woodbury_kalman_gain(P, H, R_diag)
        _, S_s = standard_kalman_gain(P, H, jnp.diag(R_diag))

        # Log-likelihood uses logdet(S) — should match
        sign_w, logdet_w = jnp.linalg.slogdet(S_w)
        sign_s, logdet_s = jnp.linalg.slogdet(S_s)
        np.testing.assert_allclose(logdet_w, logdet_s, atol=1e-5)

    def test_small_obs_dim(self) -> None:
        """Should still work when D_obs < D_state (no benefit, but correct)."""
        D_state, D_obs = 4, 2
        P = jnp.eye(D_state)
        H = jnp.zeros((D_obs, D_state)).at[0, 0].set(1.0).at[1, 1].set(1.0)
        R_diag = jnp.ones(D_obs) * 0.5

        K_w, _, _ = woodbury_kalman_gain(P, H, R_diag)
        K_s, _ = standard_kalman_gain(P, H, jnp.diag(R_diag))

        np.testing.assert_allclose(K_w, K_s, atol=1e-6)

    @pytest.mark.parametrize(
        "diag, match",
        [
            (jnp.array([0.0, 1.0]), "positive"),
            (jnp.array([-1.0, 1.0]), "positive"),
            (jnp.array([jnp.nan, 1.0]), "finite"),
        ],
    )
    def test_rejects_invalid_emission_cov_diag(self, diag, match) -> None:
        P = jnp.eye(2)
        H = jnp.ones((2, 2)) * 0.1
        with pytest.raises(ValueError, match=match):
            woodbury_kalman_gain(P, H, diag)

    def test_rejects_nonvector_emission_cov_diag(self) -> None:
        P = jnp.eye(2)
        H = jnp.ones((2, 2)) * 0.1
        with pytest.raises(ValueError, match="1D vector"):
            woodbury_kalman_gain(P, H, jnp.eye(2))


# --- Joseph Form Tests ---


class TestJosephFormUpdate:
    """Tests for the Joseph form covariance update."""

    def test_matches_standard_well_conditioned(self) -> None:
        """Joseph form should match P - K S K' on well-conditioned problems."""
        D = 3
        P = jnp.eye(D) * 2.0
        H = jnp.eye(D)
        R = jnp.eye(D) * 0.5
        K, S = standard_kalman_gain(P, H, R)

        # Standard: P_post = P - K S K'
        P_standard = P - K @ S @ K.T

        # Joseph form
        P_joseph = joseph_form_update(P, K, H, R)

        np.testing.assert_allclose(P_joseph, P_standard, atol=1e-10)

    def test_maintains_psd_ill_conditioned(self) -> None:
        """Joseph form should maintain PSD even with near-singular prior."""
        D = 4
        # Near-singular prior: one eigenvalue very small
        eigvals = jnp.array([1e-10, 0.1, 1.0, 10.0])
        V = jnp.linalg.qr(random.normal(random.PRNGKey(42), (D, D)))[0]
        P = V @ jnp.diag(eigvals) @ V.T

        H = random.normal(random.PRNGKey(1), (D, D))
        R = jnp.eye(D) * 0.1
        S = H @ P @ H.T + R
        K = psd_solve(S, H @ P).T

        P_joseph = joseph_form_update(P, K, H, R)

        # All eigenvalues should be non-negative
        eigs = jnp.linalg.eigvalsh(P_joseph)
        assert jnp.all(eigs >= -1e-12), f"Joseph form lost PSD: {eigs}"

    def test_symmetric_output(self) -> None:
        """Joseph form output should be exactly symmetric."""
        D = 3
        P = jnp.eye(D) + 0.1 * random.normal(random.PRNGKey(5), (D, D))
        P = P @ P.T
        H = random.normal(random.PRNGKey(6), (2, D))
        R = jnp.eye(2) * 0.5
        K, _ = standard_kalman_gain(P, H, R)

        P_joseph = joseph_form_update(P, K, H, R)

        np.testing.assert_allclose(P_joseph, P_joseph.T, atol=1e-14)


class TestKalmanInputValidation:
    """Guards on ``kalman_filter`` / ``kalman_smoother`` public entries.

    An indefinite ``init_cov`` or ``measurement_cov`` produces NaN
    mid-scan; callers can't distinguish "my covariance is bad" from
    "my data is pathological." The public wrappers eigenvalue-check
    both matrices host-side and raise ``ValueError`` before any work
    runs.
    """

    def _well_posed_args(self, D: int = 2, T: int = 10, seed: int = 0) -> tuple:
        key = random.PRNGKey(seed)
        k_obs, k_A, k_init = random.split(key, 3)
        A = jnp.eye(D) + 0.01 * random.normal(k_A, (D, D))
        Q = jnp.eye(D) * 0.1
        H = jnp.eye(D)
        R = jnp.eye(D)
        init_mean = jnp.zeros(D)
        init_cov = jnp.eye(D)
        obs = random.normal(k_obs, (T, D))
        return init_mean, init_cov, obs, A, Q, H, R

    def test_kalman_filter_rejects_negative_definite_init_cov(self) -> None:
        init_mean, _, obs, A, Q, H, R = self._well_posed_args()
        bad_init_cov = -jnp.eye(2)
        with pytest.raises(ValueError, match="not positive definite"):
            kalman_filter(init_mean, bad_init_cov, obs, A, Q, H, R)

    def test_kalman_filter_rejects_zero_init_cov(self) -> None:
        init_mean, _, obs, A, Q, H, R = self._well_posed_args()
        bad_init_cov = jnp.zeros((2, 2))
        with pytest.raises(ValueError, match="not positive definite"):
            kalman_filter(init_mean, bad_init_cov, obs, A, Q, H, R)

    def test_kalman_filter_rejects_asymmetric_init_cov(self) -> None:
        init_mean, _, obs, A, Q, H, R = self._well_posed_args()
        bad_init_cov = jnp.array([[1.0, 0.5], [0.0, 1.0]])
        with pytest.raises(ValueError, match="init_covariance.*not symmetric"):
            kalman_filter(init_mean, bad_init_cov, obs, A, Q, H, R)

    def test_kalman_filter_rejects_indefinite_measurement_cov(self) -> None:
        init_mean, init_cov, obs, A, Q, H, _ = self._well_posed_args()
        bad_R = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # eigs = 3, -1
        with pytest.raises(ValueError, match="measurement_cov.*not positive definite"):
            kalman_filter(init_mean, init_cov, obs, A, Q, H, bad_R)

    def test_kalman_filter_rejects_asymmetric_measurement_cov(self) -> None:
        init_mean, init_cov, obs, A, Q, H, _ = self._well_posed_args()
        bad_R = jnp.array([[1.0, 0.25], [0.0, 1.0]])
        with pytest.raises(ValueError, match="measurement_cov.*not symmetric"):
            kalman_filter(init_mean, init_cov, obs, A, Q, H, bad_R)

    @pytest.mark.parametrize(
        "bad_Q, match",
        [
            (jnp.array([[0.1, 0.5], [0.0, 0.1]]), "process_cov.*not symmetric"),
            (jnp.array([[0.1, 1.0], [1.0, 0.1]]), "positive semidefinite"),
            (jnp.array([[0.1, jnp.nan], [jnp.nan, 0.1]]), "non-finite"),
        ],
    )
    def test_kalman_filter_rejects_invalid_process_cov(self, bad_Q, match) -> None:
        init_mean, init_cov, obs, A, _, H, R = self._well_posed_args()
        with pytest.raises(ValueError, match=match):
            kalman_filter(init_mean, init_cov, obs, A, bad_Q, H, R)

    def test_kalman_filter_rejects_empty_observations(self) -> None:
        init_mean, init_cov, _, A, Q, H, R = self._well_posed_args()
        obs = jnp.zeros((0, 2))
        with pytest.raises(ValueError, match="at least one time step"):
            kalman_filter(init_mean, init_cov, obs, A, Q, H, R)

    def test_kalman_smoother_rejects_negative_definite_init_cov(self) -> None:
        init_mean, _, obs, A, Q, H, R = self._well_posed_args()
        bad_init_cov = -jnp.eye(2)
        with pytest.raises(ValueError, match="not positive definite"):
            kalman_smoother(init_mean, bad_init_cov, obs, A, Q, H, R)

    def test_kalman_smoother_rejects_indefinite_measurement_cov(self) -> None:
        init_mean, init_cov, obs, A, Q, H, _ = self._well_posed_args()
        bad_R = jnp.array([[0.5, 1.0], [1.0, 0.5]])  # eigs = 1.5, -0.5
        with pytest.raises(ValueError, match="measurement_cov.*not positive definite"):
            kalman_smoother(init_mean, init_cov, obs, A, Q, H, bad_R)

    def test_kalman_smoother_rejects_invalid_process_cov(self) -> None:
        init_mean, init_cov, obs, A, _, H, R = self._well_posed_args()
        bad_Q = jnp.array([[0.1, 0.5], [0.0, 0.1]])
        with pytest.raises(ValueError, match="process_cov.*not symmetric"):
            kalman_smoother(init_mean, init_cov, obs, A, bad_Q, H, R)

    def test_kalman_smoother_rejects_empty_observations(self) -> None:
        init_mean, init_cov, _, A, Q, H, R = self._well_posed_args()
        obs = jnp.zeros((0, 2))
        with pytest.raises(ValueError, match="at least one time step"):
            kalman_smoother(init_mean, init_cov, obs, A, Q, H, R)

    def test_validate_inputs_false_bypasses_check(self) -> None:
        """``validate_inputs=False`` must skip the eigvalsh call.

        Inner EM call sites that have already validated use this escape
        hatch to avoid O(d^3) re-work each iteration. With a bad
        ``init_cov`` and the bypass on, the filter runs (and typically
        produces NaN downstream); without the bypass, it would raise.
        """
        init_mean, _, obs, A, Q, H, R = self._well_posed_args()
        bad_init_cov = jnp.zeros((2, 2))
        filtered_mean, filtered_cov, mll = kalman_filter(
            init_mean,
            bad_init_cov,
            obs,
            A,
            Q,
            H,
            R,
            validate_inputs=False,
        )
        # We don't require NaN specifically (dtype / environment dependent),
        # only that no ValueError was raised host-side.
        assert filtered_mean.shape == obs.shape
        del filtered_cov, mll

    def test_well_posed_inputs_pass_validation(self) -> None:
        """Positive case: a well-posed PD ``init_cov`` / PD ``R`` must not raise."""
        init_mean, init_cov, obs, A, Q, H, R = self._well_posed_args()
        filtered_mean, _, _ = kalman_filter(init_mean, init_cov, obs, A, Q, H, R)
        assert filtered_mean.shape == obs.shape
        smoother_mean, _, _, _ = kalman_smoother(init_mean, init_cov, obs, A, Q, H, R)
        assert smoother_mean.shape == obs.shape


# --- Time-varying measurement noise R_t ---


def _time_varying_r_model():
    """Small 2-state, scalar-observation linear-Gaussian model for R_t tests."""
    init_mean = jnp.array([0.0, 0.0])
    init_cov = jnp.eye(2)
    A = jnp.array([[1.0, 0.1], [0.0, 0.95]])
    Q = 0.01 * jnp.eye(2)
    H = jnp.array([[1.0, 0.0]])
    obs = jnp.array([[0.5], [0.3], [-0.2], [0.8], [0.1], [-0.4]])
    return init_mean, init_cov, obs, A, Q, H


def _const_r_sequence(r_value: float, n_time: int) -> Array:
    return jnp.tile(jnp.array([[r_value]]), (n_time, 1, 1))


def test_filter_time_varying_R_matches_constant_when_all_equal():
    """A (T, 1, 1) R whose slices all equal R reproduces the constant-R filter."""
    init_mean, init_cov, obs, A, Q, H = _time_varying_r_model()
    R_const = jnp.array([[0.2]])
    R_seq = _const_r_sequence(0.2, obs.shape[0])

    m0, c0, ll0 = kalman_filter(init_mean, init_cov, obs, A, Q, H, R_const)
    m1, c1, ll1 = kalman_filter(init_mean, init_cov, obs, A, Q, H, R_seq)

    np.testing.assert_allclose(np.asarray(m1), np.asarray(m0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(c1), np.asarray(c0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(float(ll1), float(ll0), rtol=1e-10)


def test_smoother_time_varying_R_matches_constant_when_all_equal():
    """Time-varying-R smoother reproduces the constant-R smoother when R is constant."""
    init_mean, init_cov, obs, A, Q, H = _time_varying_r_model()
    R_const = jnp.array([[0.2]])
    R_seq = _const_r_sequence(0.2, obs.shape[0])

    sm0, sc0, scc0, ll0 = kalman_smoother(init_mean, init_cov, obs, A, Q, H, R_const)
    sm1, sc1, scc1, ll1 = kalman_smoother(init_mean, init_cov, obs, A, Q, H, R_seq)

    np.testing.assert_allclose(np.asarray(sm1), np.asarray(sm0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sc1), np.asarray(sc0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(scc1), np.asarray(scc0), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(float(ll1), float(ll0), rtol=1e-10)


def test_filter_large_R_at_one_bin_ignores_that_observation():
    """An (almost) infinite R at one bin drives the filtered mean to the prediction.

    With R -> infinity the Kalman gain -> 0, so the posterior mean at that bin
    equals the one-step prediction A @ m_{t-1|t-1}, i.e. the observation is
    ignored. This is the behavioral point of per-bin R.
    """
    init_mean, init_cov, obs, A, Q, H = _time_varying_r_model()
    n_time = obs.shape[0]
    R_seq = _const_r_sequence(0.2, n_time).at[2].set(jnp.array([[1e10]]))

    means, _covs, _ll = kalman_filter(init_mean, init_cov, obs, A, Q, H, R_seq)

    prediction_at_bin2 = A @ means[1]
    np.testing.assert_allclose(
        np.asarray(means[2]), np.asarray(prediction_at_bin2), rtol=1e-6, atol=1e-6
    )


def test_filter_time_varying_R_wrong_time_axis_raises():
    """A (T', 1, 1) R with T' != n_time is rejected by the new validation."""
    init_mean, init_cov, obs, A, Q, H = _time_varying_r_model()
    R_seq = _const_r_sequence(0.2, obs.shape[0] + 1)
    with pytest.raises(ValueError, match="leading time axis"):
        kalman_filter(init_mean, init_cov, obs, A, Q, H, R_seq)


def test_filter_time_varying_R_non_psd_slice_raises():
    """A correctly-shaped per-bin R with a negative slice is rejected."""
    init_mean, init_cov, obs, A, Q, H = _time_varying_r_model()
    R_seq = _const_r_sequence(0.2, obs.shape[0]).at[1].set(jnp.array([[-0.5]]))
    with pytest.raises(ValueError, match="positive definite.*time step 1"):
        kalman_filter(init_mean, init_cov, obs, A, Q, H, R_seq)


def test_filter_time_varying_R_asymmetric_slice_raises():
    """Per-bin covariance slices must be symmetric before eigvalue checks."""
    init_mean = jnp.zeros(2)
    init_cov = jnp.eye(2)
    obs = jnp.zeros((3, 2))
    A = jnp.eye(2)
    Q = 0.01 * jnp.eye(2)
    H = jnp.eye(2)
    R_seq = jnp.tile(jnp.eye(2)[None], (obs.shape[0], 1, 1))
    R_seq = R_seq.at[1].set(jnp.array([[1.0, 0.9], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="symmetric"):
        kalman_filter(init_mean, init_cov, obs, A, Q, H, R_seq)


# --- Initial-state M-step (smoothed x_0), residual forms, relative floors ---


@pytest.fixture(scope="module")
def contractive_1d_problem() -> dict:
    """1-D model with a contractive A (A=0.5, Q=0.1, R=0.05) and a short
    observation sequence offset from the prior mean, where updating the
    init prior to the smoothed x_1 (instead of x_0) decreases the LL."""
    rng = np.random.default_rng(0)
    return {
        "obs": jnp.asarray(rng.normal(size=(5, 1)) + 2.0),
        "A": jnp.array([[0.5]]),
        "Q": jnp.array([[0.1]]),
        "H": jnp.array([[1.0]]),
        "R": jnp.array([[0.05]]),
        "init_mean": jnp.zeros(1),
        "init_cov": jnp.eye(1),
    }


def _init_only_em(problem: dict, n_iter: int, use_prior: bool) -> list[float]:
    """EM that updates only (init_mean, init_cov); returns the LL history."""
    m0, P0 = problem["init_mean"], problem["init_cov"]
    A, Q, H, R = problem["A"], problem["Q"], problem["H"], problem["R"]
    lls = []
    for _ in range(n_iter):
        sm, sc, scc, ll = kalman_smoother(m0, P0, problem["obs"], A, Q, H, R)
        lls.append(float(ll))
        prior = InitialStatePrior(m0, P0, A, Q) if use_prior else None
        *_, m0, P0 = kalman_maximization_step(problem["obs"], sm, sc, scc, prior)
    return lls


def _dense_x0_posterior(problem: dict) -> tuple[float, float]:
    """E[x_0 | y] and Var[x_0 | y] for the 1-D model by dense Gaussian
    conditioning of the joint (x_0, ..., x_T, y_1, ..., y_T) -- independent
    of the RTS recursion."""
    a = float(problem["A"][0, 0])
    q = float(problem["Q"][0, 0])
    r = float(problem["R"][0, 0])
    m0 = float(problem["init_mean"][0])
    p0 = float(problem["init_cov"][0, 0])
    y = np.asarray(problem["obs"])[:, 0]
    T = y.shape[0]
    # x = L @ [x_0 - m0, w_1, ..., w_T] + mean, with x_t = a x_{t-1} + w_t.
    L = np.zeros((T + 1, T + 1))
    for t in range(T + 1):
        for s in range(t + 1):
            L[t, s] = a ** (t - s)
    noise_cov = np.diag([p0] + [q] * T)
    x_mean = np.array([a**t * m0 for t in range(T + 1)])
    x_cov = L @ noise_cov @ L.T
    y_mean = x_mean[1:]
    y_cov = x_cov[1:, 1:] + r * np.eye(T)
    cross = x_cov[0, 1:]
    gain = np.linalg.solve(y_cov, cross)
    return float(x_mean[0] + gain @ (y - y_mean)), float(x_cov[0, 0] - gain @ cross)


class TestInitialStateMStep:
    """The filter predicts before its first update, so ``init_mean`` /
    ``init_cov`` are the prior of x_0 and their EM update is the smoothed
    x_0 (one RTS step behind the smoother output, which starts at x_1)."""

    def test_init_only_em_is_monotone(self, contractive_1d_problem) -> None:
        lls = _init_only_em(contractive_1d_problem, n_iter=6, use_prior=True)
        assert np.all(np.diff(lls) >= -1e-10), lls
        # guard: the problem is one where the legacy x_1 update is not an EM
        # step -- its LL decreases after the first iteration.
        with pytest.warns(DeprecationWarning, match=LEGACY_PRIOR):
            legacy = _init_only_em(contractive_1d_problem, n_iter=6, use_prior=False)
        assert np.any(np.diff(legacy) < -1e-3), legacy

    def test_init_update_matches_grid_maximiser(self, contractive_1d_problem) -> None:
        """The M-step's (init_mean, init_cov) maximise the EM auxiliary
        function E[log N(x_0; m, P) | y], evaluated with the x_0 posterior
        from dense conditioning, over a (m, P) grid."""
        p = contractive_1d_problem
        sm, sc, scc, _ = kalman_smoother(
            p["init_mean"], p["init_cov"], p["obs"], p["A"], p["Q"], p["H"], p["R"]
        )
        prior = InitialStatePrior(p["init_mean"], p["init_cov"], p["A"], p["Q"])
        *_, m_new, P_new = kalman_maximization_step(p["obs"], sm, sc, scc, prior)

        post_mean, post_var = _dense_x0_posterior(p)
        m_grid = np.linspace(-1.0, 3.0, 2001)
        P_grid = np.linspace(0.01, 1.0, 991)
        M, P = np.meshgrid(m_grid, P_grid, indexing="ij")
        aux = -0.5 * np.log(P) - 0.5 * (post_var + (post_mean - M) ** 2) / P
        i, j = np.unravel_index(np.argmax(aux), aux.shape)
        np.testing.assert_allclose(float(m_new[0]), m_grid[i], atol=2e-3)
        np.testing.assert_allclose(float(P_new[0, 0]), P_grid[j], atol=1e-3)
        # guard: x_0's posterior differs from x_1's, which the legacy update
        # would install.
        assert abs(float(sm[0, 0]) - m_grid[i]) > 0.1

    def test_smooth_initial_state_is_rts_step_to_prior(self) -> None:
        """m_{0|T} = m_0 + J_0 (m_{1|T} - A m_0), P_{0|T} = P_0 + J_0 (P_{1|T} -
        P_{1|0}) J_0^T with J_0 = P_0 A^T P_{1|0}^{-1}, in 2-D."""
        A = jnp.array([[0.9, 0.2], [-0.1, 0.7]])
        Q = jnp.array([[0.3, 0.05], [0.05, 0.2]])
        m0 = jnp.array([1.0, -0.5])
        P0 = jnp.array([[2.0, 0.3], [0.3, 1.0]])
        m1, P1 = jnp.array([0.2, 0.4]), jnp.array([[0.5, 0.1], [0.1, 0.4]])
        m, P = smooth_initial_state(InitialStatePrior(m0, P0, A, Q), m1, P1)

        P_pred = np.asarray(A @ P0 @ A.T + Q)
        J = np.asarray(P0 @ A.T) @ np.linalg.inv(P_pred)
        # rtol covers psd_solve's 1e-9 diagonal jitter in the gain.
        np.testing.assert_allclose(
            m, np.asarray(m0) + J @ np.asarray(m1 - A @ m0), rtol=1e-7
        )
        np.testing.assert_allclose(
            P, np.asarray(P0) + J @ (np.asarray(P1) - P_pred) @ J.T, rtol=1e-7
        )

    def test_full_em_with_prior_is_monotone(self) -> None:
        """Full EM (all parameters) with the smoothed-x_0 init update keeps a
        non-decreasing LL on a 2-D model."""
        A_true = _make_asymmetric_stable_A(2, seed=15)
        H_true = jnp.array([[1.0, 0.3], [-0.2, 0.9]])
        obs, _ = _simulate_from_model(
            A_true,
            jnp.eye(2) * 0.2,
            H_true,
            jnp.eye(2) * 0.5,
            jnp.zeros(2),
            jnp.eye(2),
            200,
            seed=42,
        )
        A, Q, H, R = jnp.eye(2) * 0.5, jnp.eye(2), jnp.eye(2), jnp.eye(2) * 2.0
        m0, P0 = jnp.ones(2) * 3.0, jnp.eye(2) * 2.0
        lls = []
        for _ in range(10):
            sm, sc, scc, ll = kalman_smoother(m0, P0, obs, A, Q, H, R)
            lls.append(float(ll))
            prior = InitialStatePrior(m0, P0, A, Q)
            A, H, Q, R, m0, P0 = kalman_maximization_step(obs, sm, sc, scc, prior)
        assert np.all(np.diff(lls) >= -1e-6), lls
        assert lls[-1] > lls[0] + 1.0


class TestInitialTransitionInMStep:
    """With ``initial_state_prior`` the M-step's A and Q statistics include
    the x_0 -> x_1 transition (T transitions), using the smoothed x_0 and
    ``Cov(x_0, x_1 | y) = J_0 P_{1|T}``. Exactness against the dense oracle is
    in ``test_oracle_kalman.py``; these pin the formulas."""

    @pytest.fixture
    def problem(self) -> dict:
        A = jnp.array([[0.8, 0.2], [-0.1, 0.7]])
        Q = jnp.array([[0.3, 0.05], [0.05, 0.2]])
        H = jnp.array([[1.0, 0.4]])
        R = jnp.array([[0.2]])
        m0, P0 = jnp.array([0.5, -1.0]), jnp.array([[1.5, 0.2], [0.2, 0.8]])
        obs, _ = _simulate_from_model(A, Q, H, R, m0, P0, 12, seed=4)
        sm, sc, scc, _ = kalman_smoother(m0, P0, obs, A, Q, H, R)
        return {
            "obs": obs,
            "sm": sm,
            "sc": sc,
            "scc": scc,
            "prior": InitialStatePrior(m0, P0, A, Q),
        }

    def test_cross_cov_is_j0_times_p1(self, problem) -> None:
        prior = problem["prior"]
        m1, P1 = problem["sm"][0], problem["sc"][0]
        m00, P00, C01 = smooth_initial_state_with_cross_cov(prior, m1, P1)
        A, P0 = np.asarray(prior.transition_matrix), np.asarray(prior.init_cov)
        P_pred = A @ P0 @ A.T + np.asarray(prior.process_cov)
        J0 = P0 @ A.T @ np.linalg.inv(P_pred)
        np.testing.assert_allclose(C01, J0 @ np.asarray(P1), rtol=1e-10)
        m_b, P_b = smooth_initial_state(prior, m1, P1)
        np.testing.assert_array_equal(m_b, m00)
        np.testing.assert_array_equal(P_b, P00)

    def test_transition_statistics_span_x0_to_xT(self, problem) -> None:
        obs, sm, sc, scc = problem["obs"], problem["sm"], problem["sc"], problem["scc"]
        prior = problem["prior"]
        A, H, Q, R, m00, P00 = kalman_maximization_step(obs, sm, sc, scc, prior)
        _, _, C01 = smooth_initial_state_with_cross_cov(prior, sm[0], sc[0])
        means = np.concatenate([np.asarray(m00)[None], np.asarray(sm)])
        covs = np.concatenate([np.asarray(P00)[None], np.asarray(sc)])
        cross = np.concatenate([np.asarray(C01)[None], np.asarray(scc)])
        T = obs.shape[0]
        S = covs + np.einsum("ta,tb->tab", means, means)
        gamma1 = S[:-1].sum(0)
        gamma2 = S[1:].sum(0)
        beta = (cross + np.einsum("ta,tb->tab", means[:-1], means[1:])).sum(0).T
        A_ref = beta @ np.linalg.inv(gamma1)
        np.testing.assert_allclose(A, A_ref, rtol=1e-9)
        np.testing.assert_allclose(Q, (gamma2 - A_ref @ beta.T) / T, rtol=1e-8)
        # H and R still use the T observations of x_{1:T} only.
        with pytest.warns(DeprecationWarning, match=LEGACY_PRIOR):
            A_legacy, H_legacy, _, R_legacy, _, _ = kalman_maximization_step(
                obs, sm, sc, scc
            )
        np.testing.assert_allclose(H, H_legacy, rtol=1e-12)
        np.testing.assert_allclose(R, R_legacy, rtol=1e-12)
        # guard: including x_0 -> x_1 changes A here
        assert np.max(np.abs(np.asarray(A) - np.asarray(A_legacy))) > 1e-3


class TestResidualFormMStep:
    """R and Q use centred residual forms, PSD by construction."""

    @pytest.fixture
    def smoothed_3d(self) -> tuple:
        A = jnp.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.1], [0.0, 0.0, 0.7]])
        Q = jnp.eye(3) * 0.2
        H = jnp.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.3]])
        R = jnp.eye(2) * 0.5
        obs, _ = _simulate_from_model(A, Q, H, R, jnp.zeros(3), jnp.eye(3), 400, seed=3)
        sm, sc, scc, _ = kalman_smoother(jnp.zeros(3), jnp.eye(3), obs, A, Q, H, R)
        return obs, sm, sc, scc

    def test_residual_forms_equal_classical_forms(self, smoothed_3d) -> None:
        """At the exact solves H = delta gamma^{-1}, A = beta gamma1^{-1} the
        residual forms equal (alpha - H delta^T)/T and (gamma2 - A beta^T)/(T-1)."""
        obs, sm, sc, scc = (np.asarray(a) for a in smoothed_3d)
        T = obs.shape[0]
        gamma = sc.sum(0) + sm.T @ sm
        delta = obs.T @ sm
        alpha = obs.T @ obs
        gamma1 = gamma - np.outer(sm[-1], sm[-1]) - sc[-1]
        gamma2 = gamma - np.outer(sm[0], sm[0]) - sc[0]
        beta = (scc.sum(0) + sm[:-1].T @ sm[1:]).T
        H = np.linalg.solve(gamma, delta.T).T
        A = np.linalg.solve(gamma1, beta.T).T

        R_res = measurement_cov_residual_form(
            jnp.asarray(obs), jnp.asarray(sm), jnp.asarray(sc.sum(0)), jnp.asarray(H)
        )
        Q_res = process_cov_residual_form(
            jnp.asarray(sm),
            sum_next_cov=jnp.asarray(sc[1:].sum(0)),
            sum_prev_cov=jnp.asarray(sc[:-1].sum(0)),
            sum_cross_cov=jnp.asarray(scc.sum(0)),
            transition_matrix=jnp.asarray(A),
        )
        np.testing.assert_allclose(R_res, (alpha - H @ delta.T) / T, rtol=1e-12)
        np.testing.assert_allclose(Q_res, (gamma2 - A @ beta.T) / (T - 1), rtol=1e-12)

    def test_m_step_matches_classical_forms(self, smoothed_3d) -> None:
        """The M-step's R and Q agree with the classical shortcut forms at
        its own (jittered-solve) H and A to ~1e-10 relative on well-scaled
        data."""
        obs, sm, sc, scc = smoothed_3d
        with pytest.warns(DeprecationWarning, match=LEGACY_PRIOR):
            A, H, Q, R, _, _ = kalman_maximization_step(obs, sm, sc, scc)
        obs, sm, sc, scc = (np.asarray(a) for a in (obs, sm, sc, scc))
        T = obs.shape[0]
        gamma = sc.sum(0) + sm.T @ sm
        gamma2 = gamma - np.outer(sm[0], sm[0]) - sc[0]
        beta = (scc.sum(0) + sm[:-1].T @ sm[1:]).T
        R_old = (obs.T @ obs - np.asarray(H) @ (obs.T @ sm).T) / T
        Q_old = (gamma2 - np.asarray(A) @ beta.T) / (T - 1)
        np.testing.assert_allclose(R, R_old, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Q, Q_old, rtol=1e-10, atol=1e-12)

    def test_residual_form_is_accurate_with_a_large_offset(self) -> None:
        """Observations with a large constant offset (absorbed by a constant
        latent) make the uncentred alpha - H delta^T cancel catastrophically;
        the residual form keeps R accurate and PSD."""
        rng = np.random.default_rng(1)
        T, offset, r_true = 500, 1e7, 1e-4
        states = np.stack([np.full(T, offset), rng.normal(size=T)], axis=1)
        H = np.array([[1.0, 1.0]])
        obs = states @ H.T + rng.normal(size=(T, 1)) * np.sqrt(r_true)
        R_res = measurement_cov_residual_form(
            jnp.asarray(obs),
            jnp.asarray(states),
            jnp.zeros((2, 2)),
            jnp.asarray(H),
        )
        empirical = float(np.mean((obs - states @ H.T) ** 2))
        np.testing.assert_allclose(float(R_res[0, 0]), empirical, rtol=1e-8)
        # guard: the uncentred form really loses the answer here.
        R_old = (obs.T @ obs - H @ (obs.T @ states).T) / T
        assert abs(float(R_old[0, 0]) - empirical) > 10 * empirical


class TestRelativeEigenvalueFloor:
    """M-step covariance floors are relative to the matrix scale."""

    def test_m_step_recovers_tiny_measurement_noise(self) -> None:
        """At a 1e-3 signal scale with R = Q = 1e-10 (volts), exact state
        inputs give R, Q ~ 1e-10 instead of an absolute 1e-8 floor."""
        rng = np.random.default_rng(0)
        T, var = 4000, 1e-10
        x = np.zeros(T)
        x[0] = 1e-3
        for t in range(1, T):
            x[t] = 0.99 * x[t - 1] + rng.normal() * np.sqrt(var)
        obs = jnp.asarray((x + rng.normal(size=T) * np.sqrt(var))[:, None])
        sm = jnp.asarray(x[:, None])
        sc = jnp.zeros((T, 1, 1))
        scc = jnp.zeros((T - 1, 1, 1))
        with pytest.warns(DeprecationWarning, match=LEGACY_PRIOR):
            A, H, Q, R, _, _ = kalman_maximization_step(obs, sm, sc, scc)
        np.testing.assert_allclose(float(R[0, 0]), var, rtol=0.1)
        np.testing.assert_allclose(float(Q[0, 0]), var, rtol=0.1)

    def test_floor_logs_when_it_changes_an_eigenvalue(self, caplog) -> None:
        """A rank-deficient R (two identical noiseless channels) is floored,
        and the projection is reported through the logger."""
        rng = np.random.default_rng(2)
        T = 50
        sm = jnp.asarray(rng.normal(size=(T, 1)))
        obs = jnp.concatenate([sm, sm], axis=1)
        sc = jnp.full((T, 1, 1), 1e-2)
        scc = jnp.zeros((T - 1, 1, 1))
        with caplog.at_level("WARNING", logger="state_space_practice.utils"):
            with pytest.warns(DeprecationWarning, match=LEGACY_PRIOR):
                _, _, _, R, _, _ = kalman_maximization_step(obs, sm, sc, scc)
            jax.block_until_ready(R)
            jax.effects_barrier()
        eigs = np.linalg.eigvalsh(np.asarray(R))
        assert eigs.min() > 0.0
        assert eigs.min() < 1e-6 * eigs.max()
        assert any("measurement_cov" in rec.getMessage() for rec in caplog.records)


class TestKalmanTraceability:
    """Public filters trace under jit/grad with the default validate_inputs."""

    @pytest.fixture
    def problem(self) -> tuple:
        A = jnp.array([[0.9, 0.1], [0.0, 0.8]])
        Q = jnp.eye(2) * 0.1
        H = jnp.eye(2)
        R = jnp.eye(2) * 0.5
        obs, _ = _simulate_from_model(A, Q, H, R, jnp.zeros(2), jnp.eye(2), 30, seed=5)
        return jnp.zeros(2), jnp.eye(2), obs, A, Q, H, R

    def test_kalman_filter_jits_with_default_validation(self, problem) -> None:
        m0, P0, obs, A, Q, H, R = problem
        eager = kalman_filter(m0, P0, obs, A, Q, H, R)[2]
        jitted = jax.jit(lambda m: kalman_filter(m, P0, obs, A, Q, H, R)[2])(m0)
        np.testing.assert_allclose(jitted, eager, rtol=1e-12)

    def test_kalman_filter_grad_wrt_init_cov(self, problem) -> None:
        m0, P0, obs, A, Q, H, R = problem

        def loss(P):
            return kalman_filter(m0, P, obs, A, Q, H, R)[2]

        grad = jax.grad(loss)(P0)
        eps = 1e-6
        E = jnp.zeros((2, 2)).at[0, 0].set(1.0)
        fd = (loss(P0 + eps * E) - loss(P0 - eps * E)) / (2 * eps)
        np.testing.assert_allclose(grad[0, 0], fd, rtol=1e-5)
        assert abs(float(grad[0, 0])) > 1e-6

    def test_kalman_smoother_jits_with_default_validation(self, problem) -> None:
        m0, P0, obs, A, Q, H, R = problem
        eager = kalman_smoother(m0, P0, obs, A, Q, H, R)[0]
        jitted = jax.jit(lambda P: kalman_smoother(m0, P, obs, A, Q, H, R)[0])(P0)
        np.testing.assert_allclose(jitted, eager, rtol=1e-12)

    def test_eager_validation_still_raises(self, problem) -> None:
        m0, _, obs, A, Q, H, R = problem
        with pytest.raises(ValueError, match="not positive definite"):
            kalman_filter(m0, -jnp.eye(2), obs, A, Q, H, R)

    def test_float32_init_with_float64_params(self, problem) -> None:
        """A float32 init with float64 parameters promotes the scan carry
        instead of failing on a carry-type mismatch."""
        m0, P0, obs, A, Q, H, R = problem
        mean, cov, ll = kalman_filter(
            m0.astype(jnp.float32), P0.astype(jnp.float32), obs, A, Q, H, R
        )
        assert mean.dtype == jnp.float64 and cov.dtype == jnp.float64
        ref = kalman_filter(m0, P0, obs, A, Q, H, R)
        np.testing.assert_allclose(mean, ref[0], rtol=1e-12)
        np.testing.assert_allclose(ll, ref[2], rtol=1e-12)
        sm = kalman_smoother(
            m0.astype(jnp.float32), P0.astype(jnp.float32), obs, A, Q, H, R
        )[0]
        np.testing.assert_allclose(sm, kalman_smoother(m0, P0, obs, A, Q, H, R)[0])

    @pytest.mark.parametrize("as_numpy", [False, True])
    def test_closed_over_constants_are_validated_under_jit(
        self, problem, as_numpy
    ) -> None:
        """Concrete inputs closed over by a jitted function are still concrete:
        the value checks run (no TracerBoolConversionError) and still raise."""
        args = [np.asarray(a) if as_numpy else a for a in problem]
        eager = kalman_filter(*problem)[2]
        jitted = jax.jit(lambda: kalman_filter(*args)[2])()
        np.testing.assert_allclose(jitted, eager, rtol=1e-12)

        bad = list(args)
        bad[1] = -np.eye(2) if as_numpy else -jnp.eye(2)
        with pytest.raises(ValueError, match="not positive definite"):
            jax.jit(lambda: kalman_filter(*bad)[2])()

    @pytest.mark.parametrize("filter_fn", [kalman_filter, kalman_smoother])
    def test_traced_non_positive_definite_init_cov_warns(
        self, problem, filter_fn
    ) -> None:
        """A traced init_cov cannot be checked host-side; a negative-definite
        one must still be reported (it yields a finite but wrong result)."""
        m0, _, obs, A, Q, H, R = problem
        with pytest.warns(StateSpaceWarning, match=r"minimum eigenvalue -0\.5"):
            out = jax.jit(lambda P: filter_fn(m0, P, obs, A, Q, H, R)[-1])(
                -0.5 * jnp.eye(2)
            )
            jax.block_until_ready(out)
            jax.effects_barrier()

    def test_traced_positive_definite_init_cov_does_not_warn(self, problem) -> None:
        m0, P0, obs, A, Q, H, R = problem
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = jax.jit(lambda P: kalman_filter(m0, P, obs, A, Q, H, R)[2])(P0)
            jax.block_until_ready(out)
            jax.effects_barrier()
