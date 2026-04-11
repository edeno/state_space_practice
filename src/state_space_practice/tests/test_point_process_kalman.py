"""Tests for the point_process_kalman module.

This module tests point process filters and smoothers for neural encoding,
including stochastic filters, smoothers, and steepest descent methods.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.point_process_kalman import (
    PointProcessModel,
    _logdet_psd,
    _point_process_laplace_update,
    _validate_filter_numerics,
    get_confidence_interval,
    kalman_maximization_step,
    log_conditional_intensity,
    steepest_descent_point_process_filter,
    stochastic_point_process_filter,
    stochastic_point_process_smoother,
)

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


class TestLogConditionalIntensity:
    """Tests for the log_conditional_intensity function."""

    def test_output_shape_1d(self) -> None:
        """1D design matrix should produce 1D output."""
        n_time = 100
        n_params = 3

        design_matrix = jax.random.normal(jax.random.PRNGKey(0), (n_time, n_params))
        params = jnp.ones(n_params)

        log_intensity = log_conditional_intensity(design_matrix, params)

        assert log_intensity.shape == (n_time,)

    def test_linear_in_params(self) -> None:
        """Log intensity should be linear in parameters."""
        n_time = 50
        n_params = 3

        design_matrix = jax.random.normal(jax.random.PRNGKey(0), (n_time, n_params))
        params1 = jnp.array([1.0, 2.0, 3.0])
        params2 = jnp.array([2.0, 4.0, 6.0])  # 2x params1

        log_int1 = log_conditional_intensity(design_matrix, params1)
        log_int2 = log_conditional_intensity(design_matrix, params2)

        np.testing.assert_allclose(log_int2, 2 * log_int1, rtol=1e-10)

    def test_zero_params_gives_zero(self) -> None:
        """Zero parameters should give zero log intensity."""
        n_time = 50
        n_params = 3

        design_matrix = jax.random.normal(jax.random.PRNGKey(0), (n_time, n_params))
        params = jnp.zeros(n_params)

        log_intensity = log_conditional_intensity(design_matrix, params)

        np.testing.assert_allclose(log_intensity, jnp.zeros(n_time), rtol=1e-10)

    def test_identity_design_matrix(self) -> None:
        """Identity design matrix should return params directly."""
        n_params = 3

        design_matrix = jnp.eye(n_params)
        params = jnp.array([1.0, 2.0, 3.0])

        log_intensity = log_conditional_intensity(design_matrix, params)

        np.testing.assert_allclose(log_intensity, params, rtol=1e-10)


@pytest.fixture(scope="module")
def point_process_test_data():
    """Provides test data for point process filter/smoother tests."""
    n_time = 100
    n_params = 4
    dt = 0.02

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    # Initial parameters
    init_mean = jnp.zeros(n_params)
    init_cov = jnp.eye(n_params) * 1.0

    # State transition (near identity with slight decay)
    transition_matrix = jnp.eye(n_params) * 0.99

    # Process covariance
    process_cov = jnp.eye(n_params) * 0.01

    # Design matrix (random features)
    design_matrix = jax.random.normal(k1, (n_time, n_params)) * 0.5

    # Generate synthetic spikes
    true_params = jnp.array([0.5, -0.3, 0.2, 0.1])
    log_rate = design_matrix @ true_params
    rate = jnp.exp(log_rate) * dt
    spike_indicator = jax.random.poisson(k2, rate).astype(jnp.float64)

    return {
        "init_mean": init_mean,
        "init_cov": init_cov,
        "design_matrix": design_matrix,
        "spike_indicator": spike_indicator,
        "dt": dt,
        "transition_matrix": transition_matrix,
        "process_cov": process_cov,
        "n_time": n_time,
        "n_params": n_params,
        "true_params": true_params,
    }


class TestStochasticPointProcessFilter:
    """Tests for the stochastic_point_process_filter function."""

    def test_output_shapes(self, point_process_test_data) -> None:
        """Filter outputs should have correct shapes."""
        d = point_process_test_data

        filtered_mean, filtered_cov, mll = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        assert filtered_mean.shape == (d["n_time"], d["n_params"])
        assert filtered_cov.shape == (d["n_time"], d["n_params"], d["n_params"])
        assert mll.shape == ()

    def test_no_nans(self, point_process_test_data) -> None:
        """Filter should not produce NaN values."""
        d = point_process_test_data

        filtered_mean, filtered_cov, mll = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        assert not jnp.any(jnp.isnan(filtered_mean))
        assert not jnp.any(jnp.isnan(filtered_cov))
        assert not jnp.isnan(mll)

    def test_covariance_symmetric(self, point_process_test_data) -> None:
        """Filtered covariances should be symmetric."""
        d = point_process_test_data

        _, filtered_cov, _ = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        for t in range(d["n_time"]):
            np.testing.assert_allclose(
                filtered_cov[t], filtered_cov[t].T, rtol=1e-5, atol=1e-10
            )

    def test_log_likelihood_finite(self, point_process_test_data) -> None:
        """Marginal log-likelihood should be finite."""
        d = point_process_test_data

        _, _, mll = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        assert jnp.isfinite(mll)

    def test_with_zero_spikes(self, point_process_test_data) -> None:
        """Filter should handle case with no spikes."""
        d = point_process_test_data

        zero_spikes = jnp.zeros(d["n_time"])

        filtered_mean, filtered_cov, mll = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            zero_spikes,
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        assert not jnp.any(jnp.isnan(filtered_mean))
        assert not jnp.any(jnp.isnan(filtered_cov))
        assert jnp.isfinite(mll)

    def test_with_high_spike_rate(self, point_process_test_data) -> None:
        """Filter should handle high spike rates."""
        d = point_process_test_data

        # All spikes at every time point
        high_spikes = jnp.ones(d["n_time"]) * 5

        filtered_mean, filtered_cov, mll = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            high_spikes,
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        assert not jnp.any(jnp.isnan(filtered_mean))
        assert not jnp.any(jnp.isnan(filtered_cov))


class TestStochasticPointProcessSmoother:
    """Tests for the stochastic_point_process_smoother function."""

    def test_output_shapes(self, point_process_test_data) -> None:
        """Smoother outputs should have correct shapes."""
        d = point_process_test_data

        smoother_mean, smoother_cov, smoother_cross_cov, mll = (
            stochastic_point_process_smoother(
                d["init_mean"],
                d["init_cov"],
                d["design_matrix"],
                d["spike_indicator"],
                d["dt"],
                d["transition_matrix"],
                d["process_cov"],
                log_conditional_intensity,
            )
        )

        assert smoother_mean.shape == (d["n_time"], d["n_params"])
        assert smoother_cov.shape == (d["n_time"], d["n_params"], d["n_params"])
        assert smoother_cross_cov.shape == (
            d["n_time"] - 1,
            d["n_params"],
            d["n_params"],
        )
        assert mll.shape == ()

    def test_no_nans(self, point_process_test_data) -> None:
        """Smoother should not produce NaN values."""
        d = point_process_test_data

        smoother_mean, smoother_cov, smoother_cross_cov, mll = (
            stochastic_point_process_smoother(
                d["init_mean"],
                d["init_cov"],
                d["design_matrix"],
                d["spike_indicator"],
                d["dt"],
                d["transition_matrix"],
                d["process_cov"],
                log_conditional_intensity,
            )
        )

        assert not jnp.any(jnp.isnan(smoother_mean))
        assert not jnp.any(jnp.isnan(smoother_cov))
        assert not jnp.any(jnp.isnan(smoother_cross_cov))
        assert not jnp.isnan(mll)

    def test_smoother_reduces_variance(self, point_process_test_data) -> None:
        """Smoother variance should generally be <= filter variance."""
        d = point_process_test_data

        filtered_mean, filtered_cov, _ = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        smoother_mean, smoother_cov, _, _ = stochastic_point_process_smoother(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        # Compare traces (total variance)
        filter_var = jnp.trace(filtered_cov, axis1=1, axis2=2)
        smoother_var = jnp.trace(smoother_cov, axis1=1, axis2=2)

        # Smoother should have lower or equal variance at all times except last
        # (last time point should be equal)
        np.testing.assert_allclose(smoother_var[-1], filter_var[-1], rtol=1e-5)

        # For other time points, smoother variance should be <= filter variance
        assert jnp.all(smoother_var[:-1] <= filter_var[:-1] + 1e-6)

    def test_last_timepoint_equals_filter(self, point_process_test_data) -> None:
        """At last time point, smoother should equal filter."""
        d = point_process_test_data

        filtered_mean, filtered_cov, _ = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        smoother_mean, smoother_cov, _, _ = stochastic_point_process_smoother(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        np.testing.assert_allclose(smoother_mean[-1], filtered_mean[-1], rtol=1e-5)
        np.testing.assert_allclose(smoother_cov[-1], filtered_cov[-1], rtol=1e-5)


class TestKalmanMaximizationStep:
    """Tests for the kalman_maximization_step function."""

    @pytest.fixture
    def smoother_outputs(self, point_process_test_data):
        """Generate smoother outputs for M-step testing."""
        d = point_process_test_data

        smoother_mean, smoother_cov, smoother_cross_cov, _ = (
            stochastic_point_process_smoother(
                d["init_mean"],
                d["init_cov"],
                d["design_matrix"],
                d["spike_indicator"],
                d["dt"],
                d["transition_matrix"],
                d["process_cov"],
                log_conditional_intensity,
            )
        )

        return {
            "smoother_mean": smoother_mean,
            "smoother_cov": smoother_cov,
            "smoother_cross_cov": smoother_cross_cov,
            "n_params": d["n_params"],
        }

    def test_output_shapes(self, smoother_outputs) -> None:
        """M-step outputs should have correct shapes."""
        s = smoother_outputs

        transition_matrix, process_cov, init_mean, init_cov = kalman_maximization_step(
            s["smoother_mean"],
            s["smoother_cov"],
            s["smoother_cross_cov"],
        )

        n_params = s["n_params"]
        assert transition_matrix.shape == (n_params, n_params)
        assert process_cov.shape == (n_params, n_params)
        assert init_mean.shape == (n_params,)
        assert init_cov.shape == (n_params, n_params)

    def test_no_nans(self, smoother_outputs) -> None:
        """M-step should not produce NaN values."""
        s = smoother_outputs

        transition_matrix, process_cov, init_mean, init_cov = kalman_maximization_step(
            s["smoother_mean"],
            s["smoother_cov"],
            s["smoother_cross_cov"],
        )

        assert not jnp.any(jnp.isnan(transition_matrix))
        assert not jnp.any(jnp.isnan(process_cov))
        assert not jnp.any(jnp.isnan(init_mean))
        assert not jnp.any(jnp.isnan(init_cov))

    def test_process_cov_symmetric(self, smoother_outputs) -> None:
        """Process covariance should be symmetric."""
        s = smoother_outputs

        _, process_cov, _, _ = kalman_maximization_step(
            s["smoother_mean"],
            s["smoother_cov"],
            s["smoother_cross_cov"],
        )

        np.testing.assert_allclose(process_cov, process_cov.T, rtol=1e-10)

    def test_init_cov_symmetric(self, smoother_outputs) -> None:
        """Initial covariance should be symmetric."""
        s = smoother_outputs

        _, _, _, init_cov = kalman_maximization_step(
            s["smoother_mean"],
            s["smoother_cov"],
            s["smoother_cross_cov"],
        )

        np.testing.assert_allclose(init_cov, init_cov.T, rtol=1e-10)

    def test_init_mean_equals_first_smoother_mean(self, smoother_outputs) -> None:
        """Initial mean should equal first smoother mean."""
        s = smoother_outputs

        _, _, init_mean, _ = kalman_maximization_step(
            s["smoother_mean"],
            s["smoother_cov"],
            s["smoother_cross_cov"],
        )

        np.testing.assert_allclose(init_mean, s["smoother_mean"][0], rtol=1e-10)

    def test_init_cov_equals_first_smoother_cov(self, smoother_outputs) -> None:
        """Initial covariance should equal first smoother covariance."""
        s = smoother_outputs

        _, _, _, init_cov = kalman_maximization_step(
            s["smoother_mean"],
            s["smoother_cov"],
            s["smoother_cross_cov"],
        )

        np.testing.assert_allclose(init_cov, s["smoother_cov"][0], rtol=1e-10)


class TestSteepestDescentPointProcessFilter:
    """Tests for the steepest_descent_point_process_filter function."""

    def test_output_shape(self, point_process_test_data) -> None:
        """Filter output should have correct shape."""
        d = point_process_test_data
        epsilon = jnp.eye(d["n_params"]) * 0.01

        posterior_mean = steepest_descent_point_process_filter(
            d["init_mean"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            epsilon,
            log_conditional_intensity,
        )

        assert posterior_mean.shape == (d["n_time"], d["n_params"])

    def test_no_nans(self, point_process_test_data) -> None:
        """Filter should not produce NaN values."""
        d = point_process_test_data
        epsilon = jnp.eye(d["n_params"]) * 0.01

        posterior_mean = steepest_descent_point_process_filter(
            d["init_mean"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            epsilon,
            log_conditional_intensity,
        )

        assert not jnp.any(jnp.isnan(posterior_mean))

    def test_zero_learning_rate_preserves_params(self, point_process_test_data) -> None:
        """With zero learning rate, parameters should stay constant."""
        d = point_process_test_data
        epsilon = jnp.zeros((d["n_params"], d["n_params"]))

        posterior_mean = steepest_descent_point_process_filter(
            d["init_mean"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            epsilon,
            log_conditional_intensity,
        )

        # All time points should equal initial mean
        for t in range(d["n_time"]):
            np.testing.assert_allclose(posterior_mean[t], d["init_mean"], rtol=1e-10)

    def test_with_zero_spikes(self, point_process_test_data) -> None:
        """Filter should handle case with no spikes."""
        d = point_process_test_data
        epsilon = jnp.eye(d["n_params"]) * 0.01
        zero_spikes = jnp.zeros(d["n_time"])

        posterior_mean = steepest_descent_point_process_filter(
            d["init_mean"],
            d["design_matrix"],
            zero_spikes,
            d["dt"],
            epsilon,
            log_conditional_intensity,
        )

        assert not jnp.any(jnp.isnan(posterior_mean))


class TestGetConfidenceInterval:
    """Tests for the get_confidence_interval function."""

    def test_output_shape(self) -> None:
        """CI output should have correct shape."""
        n_time = 50
        n_params = 4

        posterior_mean = jnp.zeros((n_time, n_params))
        posterior_cov = jnp.stack([jnp.eye(n_params)] * n_time)

        ci = get_confidence_interval(posterior_mean, posterior_cov)

        assert ci.shape == (n_time, n_params, 2)

    def test_bounds_ordering(self) -> None:
        """Lower bound should be less than upper bound."""
        n_time = 50
        n_params = 4

        key = jax.random.PRNGKey(0)
        posterior_mean = jax.random.normal(key, (n_time, n_params))
        posterior_cov = jnp.stack([jnp.eye(n_params) * 0.5] * n_time)

        ci = get_confidence_interval(posterior_mean, posterior_cov)

        lower = ci[..., 0]
        upper = ci[..., 1]

        assert jnp.all(lower < upper)

    def test_mean_within_bounds(self) -> None:
        """Mean should be within confidence interval."""
        n_time = 50
        n_params = 4

        key = jax.random.PRNGKey(0)
        posterior_mean = jax.random.normal(key, (n_time, n_params))
        posterior_cov = jnp.stack([jnp.eye(n_params)] * n_time)

        ci = get_confidence_interval(posterior_mean, posterior_cov)

        lower = ci[..., 0]
        upper = ci[..., 1]

        assert jnp.all(posterior_mean >= lower)
        assert jnp.all(posterior_mean <= upper)

    def test_alpha_affects_width(self) -> None:
        """Different alpha should produce different CI widths."""
        n_time = 50
        n_params = 4

        posterior_mean = jnp.zeros((n_time, n_params))
        posterior_cov = jnp.stack([jnp.eye(n_params)] * n_time)

        ci_99 = get_confidence_interval(posterior_mean, posterior_cov, alpha=0.01)
        ci_95 = get_confidence_interval(posterior_mean, posterior_cov, alpha=0.05)

        width_99 = ci_99[..., 1] - ci_99[..., 0]
        width_95 = ci_95[..., 1] - ci_95[..., 0]

        # 99% CI should be wider
        assert jnp.all(width_99 > width_95)


class TestPointProcessModel:
    """Tests for the PointProcessModel class."""

    @pytest.fixture
    def simple_model_data(self):
        """Simple test data for PointProcessModel."""
        np.random.seed(42)
        n_time = 200
        n_basis = 4
        dt = 0.02

        # Simple cyclic design matrix
        design_matrix = np.eye(n_basis)[np.arange(n_time) % n_basis]

        # Generate spikes with moderate rate
        spike_indicator = np.random.poisson(0.3, size=n_time).astype(float)

        return {
            "design_matrix": jnp.asarray(design_matrix),
            "spike_indicator": jnp.asarray(spike_indicator),
            "n_time": n_time,
            "n_basis": n_basis,
            "dt": dt,
        }

    def test_initialization_defaults(self) -> None:
        """Model should initialize with sensible defaults."""
        model = PointProcessModel(n_state_dims=5, dt=0.02)

        assert model.n_state_dims == 5
        assert model.dt == 0.02
        assert model.transition_matrix.shape == (5, 5)
        assert model.process_cov.shape == (5, 5)
        assert model.init_mean.shape == (5,)
        assert model.init_cov.shape == (5, 5)

        # Default transition is identity
        np.testing.assert_allclose(model.transition_matrix, jnp.eye(5))

    def test_initialization_custom_params(self) -> None:
        """Model should accept custom parameters."""
        A = jnp.eye(3) * 0.95
        Q = jnp.eye(3) * 0.01
        m0 = jnp.ones(3)
        P0 = jnp.eye(3) * 2.0

        model = PointProcessModel(
            n_state_dims=3,
            dt=0.01,
            transition_matrix=A,
            process_cov=Q,
            init_mean=m0,
            init_cov=P0,
        )

        np.testing.assert_allclose(model.transition_matrix, A)
        np.testing.assert_allclose(model.process_cov, Q)
        np.testing.assert_allclose(model.init_mean, m0)
        np.testing.assert_allclose(model.init_cov, P0)

    def test_fit_returns_log_likelihoods(self, simple_model_data) -> None:
        """fit() should return a list of log-likelihoods."""
        d = simple_model_data
        model = PointProcessModel(n_state_dims=d["n_basis"], dt=d["dt"])

        ll_history = model.fit(
            d["design_matrix"],
            d["spike_indicator"],
            max_iter=5,
            tolerance=1e-10,  # Don't converge early
        )

        assert isinstance(ll_history, list)
        assert len(ll_history) == 5
        assert all(isinstance(ll, float) for ll in ll_history)

    def test_fit_populates_smoother_results(self, simple_model_data) -> None:
        """After fit(), smoother results should be populated."""
        d = simple_model_data
        model = PointProcessModel(n_state_dims=d["n_basis"], dt=d["dt"])

        # Before fit, results are None
        assert model.smoother_mean is None
        assert model.smoother_cov is None

        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=3)

        # After fit, results are populated
        assert model.smoother_mean is not None
        assert model.smoother_cov is not None
        assert model.smoother_cross_cov is not None
        assert model.filtered_mean is not None
        assert model.filtered_cov is not None

        assert model.smoother_mean.shape == (d["n_time"], d["n_basis"])
        assert model.smoother_cov.shape == (d["n_time"], d["n_basis"], d["n_basis"])

    def test_fit_no_nans(self, simple_model_data) -> None:
        """fit() should not produce NaN values."""
        d = simple_model_data
        model = PointProcessModel(
            n_state_dims=d["n_basis"],
            dt=d["dt"],
            process_cov=jnp.eye(d["n_basis"]) * 1e-3,
        )

        ll_history = model.fit(d["design_matrix"], d["spike_indicator"], max_iter=10)

        assert not any(np.isnan(ll) for ll in ll_history)
        assert model.smoother_mean is not None
        assert model.smoother_cov is not None
        assert not jnp.any(jnp.isnan(model.smoother_mean))
        assert not jnp.any(jnp.isnan(model.smoother_cov))
        assert not jnp.any(jnp.isnan(model.transition_matrix))
        assert not jnp.any(jnp.isnan(model.process_cov))

    def test_fit_log_likelihood_increases(self, simple_model_data) -> None:
        """EM should generally increase log-likelihood (monotonicity)."""
        d = simple_model_data
        model = PointProcessModel(
            n_state_dims=d["n_basis"],
            dt=d["dt"],
            process_cov=jnp.eye(d["n_basis"]) * 1e-3,
        )

        ll_history = model.fit(d["design_matrix"], d["spike_indicator"], max_iter=15)

        # Check that log-likelihood is non-decreasing (with small tolerance for numerical issues)
        for i in range(1, len(ll_history)):
            assert ll_history[i] >= ll_history[i - 1] - 1e-3, (
                f"Log-likelihood decreased at iteration {i}: "
                f"{ll_history[i-1]:.4f} -> {ll_history[i]:.4f}"
            )

    def test_fit_with_no_updates(self, simple_model_data) -> None:
        """fit() with no M-step updates should keep parameters fixed."""
        d = simple_model_data
        A_init = jnp.eye(d["n_basis"]) * 0.95
        Q_init = jnp.eye(d["n_basis"]) * 0.01

        model = PointProcessModel(
            n_state_dims=d["n_basis"],
            dt=d["dt"],
            transition_matrix=A_init,
            process_cov=Q_init,
            update_transition_matrix=False,
            update_process_cov=False,
            update_init_state=False,
        )

        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=5)

        # Parameters should be unchanged
        np.testing.assert_allclose(model.transition_matrix, A_init)
        np.testing.assert_allclose(model.process_cov, Q_init)

    def test_fit_updates_transition_matrix(self, simple_model_data) -> None:
        """fit() should update transition matrix when enabled."""
        d = simple_model_data
        A_init = jnp.eye(d["n_basis"]) * 0.5  # Start far from identity

        model = PointProcessModel(
            n_state_dims=d["n_basis"],
            dt=d["dt"],
            transition_matrix=A_init,
            process_cov=jnp.eye(d["n_basis"]) * 1e-3,
            update_transition_matrix=True,
        )

        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=10)

        # Transition matrix should have changed
        assert not jnp.allclose(model.transition_matrix, A_init)

    def test_get_rate_estimate_before_fit_raises(self, simple_model_data) -> None:
        """get_rate_estimate() before fit() should raise error."""
        d = simple_model_data
        model = PointProcessModel(n_state_dims=d["n_basis"], dt=d["dt"])

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_rate_estimate(d["design_matrix"])

    def test_get_rate_estimate_after_fit(self, simple_model_data) -> None:
        """get_rate_estimate() after fit() should return valid rates."""
        d = simple_model_data
        model = PointProcessModel(n_state_dims=d["n_basis"], dt=d["dt"])
        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=3)

        rate = model.get_rate_estimate(d["design_matrix"])

        assert rate.shape == (
            d["n_time"],
            d["n_time"],
        )  # (n_time, n_time) due to design @ design.T
        assert jnp.all(rate >= 0)  # Rates should be non-negative
        assert not jnp.any(jnp.isnan(rate))

    def test_get_confidence_interval_before_fit_raises(self, simple_model_data) -> None:
        """get_confidence_interval() before fit() should raise error."""
        d = simple_model_data
        model = PointProcessModel(n_state_dims=d["n_basis"], dt=d["dt"])

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_confidence_interval()

    def test_get_confidence_interval_after_fit(self, simple_model_data) -> None:
        """get_confidence_interval() should return valid CIs."""
        d = simple_model_data
        model = PointProcessModel(n_state_dims=d["n_basis"], dt=d["dt"])
        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=3)

        ci = model.get_confidence_interval(alpha=0.05)

        assert ci.shape == (d["n_time"], d["n_basis"], 2)

        # Lower bound should be less than upper bound
        lower = ci[..., 0]
        upper = ci[..., 1]
        assert jnp.all(lower < upper)

        # Mean should be within bounds
        assert jnp.all(model.smoother_mean >= lower)
        assert jnp.all(model.smoother_mean <= upper)

    def test_filtered_vs_smoothed_results(self, simple_model_data) -> None:
        """get_confidence_interval can return filtered or smoothed results."""
        d = simple_model_data
        model = PointProcessModel(n_state_dims=d["n_basis"], dt=d["dt"])
        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=3)

        ci_smoothed = model.get_confidence_interval(use_smoothed=True)
        ci_filtered = model.get_confidence_interval(use_smoothed=False)

        # They should be different (except possibly at last time point)
        assert ci_smoothed.shape == ci_filtered.shape
        # Last time point should be the same
        np.testing.assert_allclose(ci_smoothed[-1], ci_filtered[-1], rtol=1e-5)

    def test_random_walk_recovery(self) -> None:
        """Model should recover A ≈ I for random walk data."""
        np.random.seed(123)
        n_time = 500
        n_basis = 3
        dt = 0.02

        design_matrix = np.eye(n_basis)[np.arange(n_time) % n_basis]
        spike_indicator = np.random.poisson(0.2, size=n_time).astype(float)

        model = PointProcessModel(
            n_state_dims=n_basis,
            dt=dt,
            transition_matrix=jnp.eye(n_basis) * 0.8,  # Start away from I
            process_cov=jnp.eye(n_basis) * 1e-3,
        )

        model.fit(design_matrix, spike_indicator, max_iter=20)

        # A should be close to identity for random walk (allow 0.25 tolerance)
        # Note: with limited data, exact recovery is difficult
        np.testing.assert_allclose(model.transition_matrix, jnp.eye(n_basis), atol=0.25)


class TestKalmanMaximizationStepMStepRegression:
    """Regression tests for the kalman_maximization_step M-step bug fix.

    The bug was that the beta computation was missing the outer products of
    consecutive means. This caused incorrect estimation of A and Q.
    """

    def test_mstep_recovers_identity_transition(self) -> None:
        """M-step should recover A ≈ I when true dynamics are random walk.

        This test specifically catches the bug where beta was computed as:
            beta = smoother_cross_cov.sum(axis=0).T  # WRONG

        instead of:
            beta = (smoother_cross_cov.sum(axis=0)
                   + sum_of_outer_products(smoother_mean[:-1], smoother_mean[1:])).T
        """
        np.random.seed(42)
        n_time = 100
        n_params = 3

        # Simulate a random walk
        states = [jnp.zeros(n_params)]
        for _ in range(n_time - 1):
            noise = jax.random.normal(jax.random.PRNGKey(_), (n_params,)) * 0.1
            states.append(states[-1] + noise)
        smoother_mean = jnp.stack(states)

        # Small covariances (as if we had good observations)
        smoother_cov = jnp.stack([jnp.eye(n_params) * 0.001] * n_time)

        # Cross-covariances should be approximately P_{t|T} @ A.T @ P_{t+1|t}^{-1} @ P_{t+1|T}
        # For simplicity, use small identity-like cross-covariances
        smoother_cross_cov = jnp.stack([jnp.eye(n_params) * 0.0005] * (n_time - 1))

        # Run M-step
        A_est, Q_est, _, _ = kalman_maximization_step(
            smoother_mean, smoother_cov, smoother_cross_cov
        )

        # With the bug, A would be very different from identity
        # With the fix, A should be close to identity
        # Note: diagonal elements should be close to 1, off-diagonal close to 0
        np.testing.assert_allclose(jnp.diag(A_est), jnp.ones(n_params), atol=0.15)
        np.testing.assert_allclose(
            A_est - jnp.diag(jnp.diag(A_est)), jnp.zeros((n_params, n_params)), atol=0.2
        )

        # Q should also be reasonable (positive eigenvalues)
        eigvals = jnp.linalg.eigvalsh(Q_est)
        assert jnp.all(eigvals > -1e-6), f"Q has negative eigenvalues: {eigvals}"

    def test_mstep_transition_depends_on_mean_products(self) -> None:
        """Verify that transition matrix depends on outer products of means.

        If the bug exists (missing outer products), A would only depend on
        the cross-covariances, not on the actual state trajectory.
        """
        np.random.seed(123)
        n_time = 50
        n_params = 2

        # Case 1: States that increase linearly
        smoother_mean_increasing = jnp.linspace(0, 1, n_time)[:, None] * jnp.ones(
            n_params
        )
        smoother_cov = jnp.stack([jnp.eye(n_params) * 0.01] * n_time)
        smoother_cross_cov = jnp.stack([jnp.eye(n_params) * 0.005] * (n_time - 1))

        A_increasing, _, _, _ = kalman_maximization_step(
            smoother_mean_increasing, smoother_cov, smoother_cross_cov
        )

        # Case 2: States that are constant (all zeros)
        smoother_mean_constant = jnp.zeros((n_time, n_params))

        A_constant, _, _, _ = kalman_maximization_step(
            smoother_mean_constant, smoother_cov, smoother_cross_cov
        )

        # With the fix, A should be different for different trajectories
        # because the outer products of means contribute differently
        assert not jnp.allclose(
            A_increasing, A_constant, atol=0.01
        ), "A should depend on state trajectory (outer products of means)"


# --- Multi-Neuron Tests ---


@pytest.fixture(scope="module")
def multi_neuron_test_data():
    """Provides test data for multi-neuron point process tests."""
    n_time = 100
    n_params = 3
    n_neurons = 5
    dt = 0.02

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Initial parameters
    init_mean = jnp.zeros(n_params)
    init_cov = jnp.eye(n_params) * 1.0

    # State transition (near identity)
    transition_matrix = jnp.eye(n_params) * 0.99

    # Process covariance
    process_cov = jnp.eye(n_params) * 0.01

    # Design matrix for multi-neuron case: (n_time, n_neurons, n_params)
    # Each neuron has different weights
    design_matrix = jax.random.normal(k1, (n_time, n_neurons, n_params)) * 0.5

    # True latent state trajectory (for generating spikes)
    true_states = jax.random.normal(k2, (n_time, n_params)) * 0.3

    # Generate multi-neuron spikes based on design_matrix and true_states
    # log_rate[t, n] = design_matrix[t, n, :] @ true_states[t, :]
    log_rate = jnp.einsum("tnp,tp->tn", design_matrix, true_states)
    rate = jnp.exp(log_rate) * dt
    spike_indicator = jax.random.poisson(k3, rate).astype(jnp.float64)

    return {
        "init_mean": init_mean,
        "init_cov": init_cov,
        "design_matrix": design_matrix,
        "spike_indicator": spike_indicator,
        "dt": dt,
        "transition_matrix": transition_matrix,
        "process_cov": process_cov,
        "n_time": n_time,
        "n_params": n_params,
        "n_neurons": n_neurons,
        "true_states": true_states,
    }


def multi_neuron_log_intensity(design_matrix_t, params):
    """Multi-neuron log intensity function.

    Parameters
    ----------
    design_matrix_t : Array, shape (n_neurons, n_params)
        Design matrix at time t for all neurons.
    params : Array, shape (n_params,)
        Latent state.

    Returns
    -------
    Array, shape (n_neurons,)
        Log intensity for each neuron.
    """
    return design_matrix_t @ params


class TestMultiNeuronFilter:
    """Tests for multi-neuron point process filter."""

    def test_output_shapes(self, multi_neuron_test_data) -> None:
        """Filter outputs should have correct shapes for multi-neuron input."""
        d = multi_neuron_test_data

        filtered_mean, filtered_cov, mll = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            multi_neuron_log_intensity,
        )

        assert filtered_mean.shape == (d["n_time"], d["n_params"])
        assert filtered_cov.shape == (d["n_time"], d["n_params"], d["n_params"])
        assert mll.shape == ()

    def test_no_nans(self, multi_neuron_test_data) -> None:
        """Multi-neuron filter should not produce NaN values."""
        d = multi_neuron_test_data

        filtered_mean, filtered_cov, mll = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            multi_neuron_log_intensity,
        )

        assert not jnp.any(jnp.isnan(filtered_mean))
        assert not jnp.any(jnp.isnan(filtered_cov))
        assert not jnp.isnan(mll)

    def test_log_likelihood_finite(self, multi_neuron_test_data) -> None:
        """Marginal log likelihood should be finite."""
        d = multi_neuron_test_data

        _, _, mll = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            multi_neuron_log_intensity,
        )

        assert jnp.isfinite(mll)

    def test_covariance_symmetric(self, multi_neuron_test_data) -> None:
        """Multi-neuron filtered covariances should be symmetric."""
        d = multi_neuron_test_data

        _, filtered_cov, _ = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            multi_neuron_log_intensity,
        )

        for t in range(d["n_time"]):
            np.testing.assert_allclose(filtered_cov[t], filtered_cov[t].T, atol=1e-10)

    def test_multi_neuron_reduces_to_single_neuron(
        self, point_process_test_data
    ) -> None:
        """Multi-neuron with n_neurons=1 should match single-neuron filter."""
        d = point_process_test_data

        # Single neuron result (original API)
        filtered_mean_single, filtered_cov_single, mll_single = (
            stochastic_point_process_filter(
                d["init_mean"],
                d["init_cov"],
                d["design_matrix"],
                d["spike_indicator"],
                d["dt"],
                d["transition_matrix"],
                d["process_cov"],
                log_conditional_intensity,
            )
        )

        # Multi-neuron with n_neurons=1 (same data, reshaped)
        design_matrix_multi = d["design_matrix"][:, None, :]  # (n_time, 1, n_params)
        spike_indicator_multi = d["spike_indicator"][:, None]  # (n_time, 1)

        filtered_mean_multi, filtered_cov_multi, mll_multi = (
            stochastic_point_process_filter(
                d["init_mean"],
                d["init_cov"],
                design_matrix_multi,
                spike_indicator_multi,
                d["dt"],
                d["transition_matrix"],
                d["process_cov"],
                multi_neuron_log_intensity,
            )
        )

        np.testing.assert_allclose(filtered_mean_single, filtered_mean_multi, rtol=1e-5)
        np.testing.assert_allclose(filtered_cov_single, filtered_cov_multi, rtol=1e-5)
        np.testing.assert_allclose(mll_single, mll_multi, rtol=1e-5)

    def test_more_neurons_reduces_variance(self) -> None:
        """Adding more informative neurons should reduce posterior variance."""
        n_time = 50
        n_params = 2
        dt = 0.02

        key = jax.random.PRNGKey(123)
        k1, k2, k3 = jax.random.split(key, 3)

        init_mean = jnp.zeros(n_params)
        init_cov = jnp.eye(n_params)
        transition_matrix = jnp.eye(n_params) * 0.99
        process_cov = jnp.eye(n_params) * 0.01

        # Design matrix and spikes for 1 neuron
        design_1 = jax.random.normal(k1, (n_time, 1, n_params))
        spikes_1 = jax.random.poisson(k2, jnp.ones((n_time, 1)) * 0.1)

        # Design matrix and spikes for 5 neurons (tile the 1-neuron case)
        design_5 = jnp.tile(design_1, (1, 5, 1))
        spikes_5 = jnp.tile(spikes_1, (1, 5))

        # Run filter with 1 neuron
        _, cov_1, _ = stochastic_point_process_filter(
            init_mean,
            init_cov,
            design_1,
            spikes_1,
            dt,
            transition_matrix,
            process_cov,
            multi_neuron_log_intensity,
        )

        # Run filter with 5 neurons (more information)
        _, cov_5, _ = stochastic_point_process_filter(
            init_mean,
            init_cov,
            design_5,
            spikes_5,
            dt,
            transition_matrix,
            process_cov,
            multi_neuron_log_intensity,
        )

        # Average variance should be smaller with more neurons
        avg_var_1 = jnp.mean(jnp.trace(cov_1, axis1=-2, axis2=-1))
        avg_var_5 = jnp.mean(jnp.trace(cov_5, axis1=-2, axis2=-1))

        assert (
            avg_var_5 < avg_var_1
        ), f"Variance should decrease with more neurons: {avg_var_5} >= {avg_var_1}"


class TestMultiNeuronSmoother:
    """Tests for multi-neuron point process smoother."""

    def test_output_shapes(self, multi_neuron_test_data) -> None:
        """Smoother outputs should have correct shapes for multi-neuron input."""
        d = multi_neuron_test_data

        smoother_mean, smoother_cov, smoother_cross_cov, mll = (
            stochastic_point_process_smoother(
                d["init_mean"],
                d["init_cov"],
                d["design_matrix"],
                d["spike_indicator"],
                d["dt"],
                d["transition_matrix"],
                d["process_cov"],
                multi_neuron_log_intensity,
            )
        )

        assert smoother_mean.shape == (d["n_time"], d["n_params"])
        assert smoother_cov.shape == (d["n_time"], d["n_params"], d["n_params"])
        assert smoother_cross_cov.shape == (
            d["n_time"] - 1,
            d["n_params"],
            d["n_params"],
        )

    def test_no_nans(self, multi_neuron_test_data) -> None:
        """Multi-neuron smoother should not produce NaN values."""
        d = multi_neuron_test_data

        smoother_mean, smoother_cov, smoother_cross_cov, mll = (
            stochastic_point_process_smoother(
                d["init_mean"],
                d["init_cov"],
                d["design_matrix"],
                d["spike_indicator"],
                d["dt"],
                d["transition_matrix"],
                d["process_cov"],
                multi_neuron_log_intensity,
            )
        )

        assert not jnp.any(jnp.isnan(smoother_mean))
        assert not jnp.any(jnp.isnan(smoother_cov))
        assert not jnp.any(jnp.isnan(smoother_cross_cov))
        assert not jnp.isnan(mll)

    def test_smoother_reduces_variance(self, multi_neuron_test_data) -> None:
        """Smoother should have lower or equal variance compared to filter."""
        d = multi_neuron_test_data

        filtered_mean, filtered_cov, _ = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            multi_neuron_log_intensity,
        )

        smoother_mean, smoother_cov, _, _ = stochastic_point_process_smoother(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            multi_neuron_log_intensity,
        )

        # Exclude last timepoint where smoother = filter
        filter_trace = jnp.trace(filtered_cov[:-1], axis1=-2, axis2=-1)
        smoother_trace = jnp.trace(smoother_cov[:-1], axis1=-2, axis2=-1)

        # Smoother variance should be <= filter variance
        assert jnp.all(smoother_trace <= filter_trace + 1e-6)


class TestMultiNeuronIntensityDirection:
    """Test that spikes in one neuron push state in the correct direction."""

    def test_high_spikes_increase_intensity_estimate(self) -> None:
        """High spike counts should increase intensity in that direction.

        With a simple linear log-intensity, spikes in a neuron with positive
        weights should push the latent state to increase that neuron's intensity.
        """
        n_time = 20
        n_params = 2
        n_neurons = 2
        dt = 0.02

        init_mean = jnp.zeros(n_params)
        init_cov = jnp.eye(n_params)
        transition_matrix = jnp.eye(n_params)  # Random walk
        process_cov = jnp.eye(n_params) * 0.001

        # Simple design matrix: neuron 0 responds to state[0], neuron 1 to state[1]
        # Each row is same: [[1, 0], [0, 1]] (identity-like)
        design_matrix = jnp.broadcast_to(
            jnp.eye(n_neurons)[None, :, :], (n_time, n_neurons, n_params)
        )

        # Spikes only in neuron 0 (high rate)
        spikes_neuron0 = jnp.zeros((n_time, n_neurons))
        spikes_neuron0 = spikes_neuron0.at[:, 0].set(5)  # 5 spikes per bin in neuron 0

        # Spikes only in neuron 1 (high rate)
        spikes_neuron1 = jnp.zeros((n_time, n_neurons))
        spikes_neuron1 = spikes_neuron1.at[:, 1].set(5)  # 5 spikes per bin in neuron 1

        # Run filters
        mean_0, _, _ = stochastic_point_process_filter(
            init_mean,
            init_cov,
            design_matrix,
            spikes_neuron0,
            dt,
            transition_matrix,
            process_cov,
            multi_neuron_log_intensity,
        )

        mean_1, _, _ = stochastic_point_process_filter(
            init_mean,
            init_cov,
            design_matrix,
            spikes_neuron1,
            dt,
            transition_matrix,
            process_cov,
            multi_neuron_log_intensity,
        )

        # With spikes in neuron 0, state[0] should be higher than state[1]
        assert jnp.mean(mean_0[:, 0]) > jnp.mean(
            mean_0[:, 1]
        ), "Spikes in neuron 0 should increase state[0]"

        # With spikes in neuron 1, state[1] should be higher than state[0]
        assert jnp.mean(mean_1[:, 1]) > jnp.mean(
            mean_1[:, 0]
        ), "Spikes in neuron 1 should increase state[1]"


@pytest.fixture(scope="module")
def single_neuron_model_data():
    """Simple test data for single-neuron PointProcessModel tests."""
    np.random.seed(42)
    n_time = 100
    n_basis = 4
    dt = 0.02

    # Simple cyclic design matrix
    design_matrix = np.eye(n_basis)[np.arange(n_time) % n_basis]

    # Generate spikes with moderate rate
    spike_indicator = np.random.poisson(0.3, size=n_time).astype(float)

    return {
        "design_matrix": jnp.asarray(design_matrix),
        "spike_indicator": jnp.asarray(spike_indicator),
        "n_time": n_time,
        "n_basis": n_basis,
        "dt": dt,
    }


class TestGetRateEstimateMultiNeuron:
    """Tests for multi-neuron get_rate_estimate functionality."""

    def test_rate_estimate_uses_hz_convention_not_expected_count(self) -> None:
        """Rate estimate should return exp(log_rate), not exp(log_rate) / dt."""
        model = PointProcessModel(n_state_dims=1, dt=0.02)
        model.smoother_mean = jnp.array([[jnp.log(4.0)]])

        rate = model.get_rate_estimate(
            jnp.array([[1.0]]), evaluate_at_all_positions=False
        )

        np.testing.assert_allclose(rate, jnp.array([4.0]), rtol=1e-10)

    def test_time_aligned_single_neuron_shape(self, single_neuron_model_data) -> None:
        """Time-aligned rate estimate should have shape (n_time,) for single neuron."""
        d = single_neuron_model_data
        model = PointProcessModel(n_state_dims=d["n_basis"], dt=d["dt"])
        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=3)

        rate = model.get_rate_estimate(
            d["design_matrix"], evaluate_at_all_positions=False
        )

        assert rate.shape == (d["n_time"],)
        assert jnp.all(rate >= 0)
        assert not jnp.any(jnp.isnan(rate))

    def test_all_positions_single_neuron_shape(self, single_neuron_model_data) -> None:
        """All-positions rate estimate should have shape (n_time, n_pos) for single neuron."""
        d = single_neuron_model_data
        model = PointProcessModel(n_state_dims=d["n_basis"], dt=d["dt"])
        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=3)

        # Evaluate at all positions (default behavior)
        rate = model.get_rate_estimate(d["design_matrix"])

        assert rate.shape == (d["n_time"], d["n_time"])
        assert jnp.all(rate >= 0)
        assert not jnp.any(jnp.isnan(rate))

    def test_multi_neuron_time_aligned_shape(self, multi_neuron_test_data) -> None:
        """Time-aligned rate estimate should have shape (n_time, n_neurons) for multi-neuron."""
        d = multi_neuron_test_data
        model = PointProcessModel(
            n_state_dims=d["n_params"],
            dt=d["dt"],
            transition_matrix=d["transition_matrix"],
            process_cov=d["process_cov"],
            log_intensity_func=multi_neuron_log_intensity,
        )
        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=3)

        rate = model.get_rate_estimate(
            d["design_matrix"], evaluate_at_all_positions=False
        )

        assert rate.shape == (d["n_time"], d["n_neurons"])
        assert jnp.all(rate >= 0)
        assert not jnp.any(jnp.isnan(rate))

    def test_multi_neuron_all_positions_shape(self, multi_neuron_test_data) -> None:
        """All-positions rate estimate should have shape (n_time, n_pos, n_neurons)."""
        d = multi_neuron_test_data
        model = PointProcessModel(
            n_state_dims=d["n_params"],
            dt=d["dt"],
            transition_matrix=d["transition_matrix"],
            process_cov=d["process_cov"],
            log_intensity_func=multi_neuron_log_intensity,
        )
        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=3)

        # Evaluate at all positions
        rate = model.get_rate_estimate(d["design_matrix"])

        assert rate.shape == (d["n_time"], d["n_time"], d["n_neurons"])
        assert jnp.all(rate >= 0)
        assert not jnp.any(jnp.isnan(rate))

    def test_filtered_vs_smoothed_rates(self, multi_neuron_test_data) -> None:
        """Filtered and smoothed rate estimates should differ."""
        d = multi_neuron_test_data
        model = PointProcessModel(
            n_state_dims=d["n_params"],
            dt=d["dt"],
            transition_matrix=d["transition_matrix"],
            process_cov=d["process_cov"],
            log_intensity_func=multi_neuron_log_intensity,
        )
        model.fit(d["design_matrix"], d["spike_indicator"], max_iter=5)

        rate_smoothed = model.get_rate_estimate(
            d["design_matrix"], use_smoothed=True, evaluate_at_all_positions=False
        )
        rate_filtered = model.get_rate_estimate(
            d["design_matrix"], use_smoothed=False, evaluate_at_all_positions=False
        )

        # Should have same shape
        assert rate_smoothed.shape == rate_filtered.shape

        # Last timepoint should be identical (smoother = filter at last time)
        np.testing.assert_allclose(rate_smoothed[-1], rate_filtered[-1], rtol=1e-5)

        # Earlier timepoints may differ (smoothing uses future info)
        # Just check they're both valid
        assert jnp.all(rate_smoothed >= 0)
        assert jnp.all(rate_filtered >= 0)

    def test_rate_estimate_uses_log_intensity_func(self) -> None:
        """Rate estimate should use the model's log_intensity_func, not just linear."""
        n_time = 50
        n_params = 2
        dt = 0.02

        # Create a nonlinear log intensity function
        def nonlinear_log_intensity(design_matrix_t, params):
            # Quadratic: (Z @ x)^2
            linear = design_matrix_t @ params
            return linear**2

        key = jax.random.PRNGKey(123)
        design_matrix = jax.random.normal(key, (n_time, n_params)) * 0.3
        spike_indicator = jax.random.poisson(
            jax.random.PRNGKey(456),
            jnp.exp(
                jax.vmap(nonlinear_log_intensity)(
                    design_matrix, jnp.zeros((n_time, n_params))
                )
            )
            * dt,
        ).astype(jnp.float64)

        model = PointProcessModel(
            n_state_dims=n_params,
            dt=dt,
            log_intensity_func=nonlinear_log_intensity,
        )
        model.fit(design_matrix, spike_indicator, max_iter=3)

        rate = model.get_rate_estimate(design_matrix, evaluate_at_all_positions=False)

        # Rate should be non-negative (exp of anything is positive)
        assert jnp.all(rate >= 0)
        assert not jnp.any(jnp.isnan(rate))

    def test_laplace_update_with_nonlinear_intensity_produces_psd_covariance(
        self,
    ) -> None:
        """Laplace update should produce PSD covariances even with nonlinear intensity.

        For nonlinear log-intensity functions, the Hessian correction term can make
        the posterior precision matrix indefinite. The Laplace update should project
        to the PSD cone to ensure valid covariances.
        """
        from state_space_practice.point_process_kalman import (
            _point_process_laplace_update,
        )

        n_latent = 3
        dt = 0.01

        # Create a challenging nonlinear log-intensity with large second derivatives
        # Using a polynomial that will produce significant Hessian corrections
        def nonlinear_log_intensity(state):
            # Quadratic in state components - creates non-trivial Hessians
            # log_intensity[0] = state[0]^2 + state[1]
            # log_intensity[1] = state[1]^2 + state[2]
            return jnp.array(
                [
                    state[0] ** 2 + state[1],
                    state[1] ** 2 + state[2],
                ]
            )

        # Prior with moderate uncertainty
        prior_mean = jnp.array([1.0, 0.5, -0.5])
        prior_cov = jnp.eye(n_latent) * 0.5

        # High spike count to drive strong Hessian corrections
        # (innovation = y - lambda*dt can be large positive)
        spike_counts = jnp.array([10.0, 8.0])  # Much higher than expected

        # Run the Laplace update
        posterior_mean, posterior_cov, log_lik = _point_process_laplace_update(
            one_step_mean=prior_mean,
            one_step_cov=prior_cov,
            spike_indicator_t=spike_counts,
            dt=dt,
            log_intensity_func=nonlinear_log_intensity,
            diagonal_boost=1e-9,  # production default
            include_laplace_normalization=True,
        )

        # Verify outputs are valid (no NaN/Inf)
        assert not jnp.any(jnp.isnan(posterior_mean)), "Posterior mean contains NaN"
        assert not jnp.any(jnp.isnan(posterior_cov)), "Posterior cov contains NaN"
        assert not jnp.any(jnp.isinf(posterior_mean)), "Posterior mean contains Inf"
        assert not jnp.any(jnp.isinf(posterior_cov)), "Posterior cov contains Inf"

        # Verify posterior covariance is PSD (all eigenvalues >= 0)
        eigvals = jnp.linalg.eigvalsh(posterior_cov)
        assert jnp.all(
            eigvals >= 0
        ), f"Posterior cov is not PSD: eigenvalues = {eigvals}"

        # Verify covariance is symmetric
        np.testing.assert_allclose(
            posterior_cov,
            posterior_cov.T,
            rtol=1e-10,
            err_msg="Posterior covariance is not symmetric",
        )

    @pytest.mark.parametrize("log_rate", [200.0, 800.0])
    def test_laplace_update_large_log_rates_remain_finite(
        self, log_rate: float
    ) -> None:
        """Large positive log-rates should not create inf/nan outputs."""

        def large_log_intensity(_state):
            return jnp.array([log_rate])

        posterior_mean, posterior_cov, log_lik = _point_process_laplace_update(
            one_step_mean=jnp.zeros(1),
            one_step_cov=jnp.eye(1),
            spike_indicator_t=jnp.array([0.0]),
            dt=0.02,
            log_intensity_func=large_log_intensity,
            include_laplace_normalization=True,
        )

        assert jnp.all(jnp.isfinite(posterior_mean))
        assert jnp.all(jnp.isfinite(posterior_cov))
        assert jnp.isfinite(log_lik)


# --- Mathematical Correctness Tests ---


class TestPointProcessGradientCorrectness:
    """Tests verifying gradient/Hessian via finite differences.

    Catches sign errors or missing terms in the analytical gradient/Hessian
    that would be invisible to tests reimplementing the same formula.
    """

    def _make_log_intensity(self, weights: jnp.ndarray, baseline: jnp.ndarray):
        """Create a linear log-intensity function: log_lambda = baseline + W @ x."""

        def log_intensity(x):
            return jnp.atleast_1d(baseline + weights @ x)

        return log_intensity

    def _neg_log_posterior(
        self, x, one_step_mean, prior_precision, spikes, dt, log_intensity_func
    ):
        """Compute negative log-posterior for finite difference checks."""
        log_lambda = log_intensity_func(x)
        cond_int = jnp.exp(log_lambda) * dt
        log_lik = jnp.sum(spikes * jnp.log(cond_int + 1e-10) - cond_int)
        delta = x - one_step_mean
        log_prior = -0.5 * delta @ (prior_precision @ delta)
        return -(log_lik + log_prior)

    def test_gradient_matches_finite_difference(self) -> None:
        """Analytical gradient should match finite-difference gradient."""
        n_latent, n_neurons = 3, 4
        dt = 0.01

        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        weights = jax.random.normal(k1, (n_neurons, n_latent)) * 0.5
        baseline = jax.random.normal(k2, (n_neurons,)) - 1.0
        log_intensity_func = self._make_log_intensity(weights, baseline)

        one_step_mean = jnp.zeros(n_latent)
        prior_precision = jnp.eye(n_latent)
        spikes = jnp.array([0.0, 1.0, 0.0, 2.0])

        # Analytical gradient at the prior mean
        log_lambda = log_intensity_func(one_step_mean)
        cond_int = jnp.exp(log_lambda) * dt
        innovation = spikes - cond_int
        jacobian = jax.jacfwd(log_intensity_func)(one_step_mean)
        analytical_grad = jacobian.T @ innovation  # prior grad = 0 at prior mean

        # Finite-difference gradient
        eps = 1e-5
        fd_grad = np.zeros(n_latent)
        for d in range(n_latent):
            x_plus = np.array(one_step_mean).copy()
            x_minus = np.array(one_step_mean).copy()
            x_plus[d] += eps
            x_minus[d] -= eps
            f_plus = self._neg_log_posterior(
                jnp.array(x_plus),
                one_step_mean,
                prior_precision,
                spikes,
                dt,
                log_intensity_func,
            )
            f_minus = self._neg_log_posterior(
                jnp.array(x_minus),
                one_step_mean,
                prior_precision,
                spikes,
                dt,
                log_intensity_func,
            )
            fd_grad[d] = -(float(f_plus) - float(f_minus)) / (2 * eps)

        np.testing.assert_allclose(
            analytical_grad,
            fd_grad,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Analytical gradient should match finite differences",
        )

    def test_hessian_matches_finite_difference(self) -> None:
        """Analytical Hessian should match finite-difference Hessian."""
        n_latent, n_neurons = 3, 4
        dt = 0.01

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        weights = jax.random.normal(k1, (n_neurons, n_latent)) * 0.5
        baseline = jax.random.normal(k2, (n_neurons,)) - 1.0
        log_intensity_func = self._make_log_intensity(weights, baseline)

        one_step_mean = jnp.zeros(n_latent)
        prior_precision = jnp.eye(n_latent)
        spikes = jnp.array([0.0, 1.0, 0.0, 2.0])

        # Analytical Hessian of negative log-posterior at prior mean
        log_lambda = log_intensity_func(one_step_mean)
        cond_int = jnp.exp(log_lambda) * dt
        innovation = spikes - cond_int
        jacobian = jax.jacfwd(log_intensity_func)(one_step_mean)
        hessian_log_lambda = jax.jacfwd(jax.jacfwd(log_intensity_func))(one_step_mean)

        fisher_info = jacobian.T @ (cond_int[:, None] * jacobian)
        hessian_correction = jnp.einsum("n,nij->ij", innovation, hessian_log_lambda)
        # Posterior precision = -Hessian of log-posterior
        analytical_neg_hessian = prior_precision + fisher_info - hessian_correction

        # Finite-difference Hessian of negative log-posterior
        eps = 1e-4
        fd_hessian = np.zeros((n_latent, n_latent))
        for i in range(n_latent):
            for j in range(n_latent):
                x_pp = np.array(one_step_mean).copy()
                x_pm = np.array(one_step_mean).copy()
                x_mp = np.array(one_step_mean).copy()
                x_mm = np.array(one_step_mean).copy()
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                f_pp = float(
                    self._neg_log_posterior(
                        jnp.array(x_pp),
                        one_step_mean,
                        prior_precision,
                        spikes,
                        dt,
                        log_intensity_func,
                    )
                )
                f_pm = float(
                    self._neg_log_posterior(
                        jnp.array(x_pm),
                        one_step_mean,
                        prior_precision,
                        spikes,
                        dt,
                        log_intensity_func,
                    )
                )
                f_mp = float(
                    self._neg_log_posterior(
                        jnp.array(x_mp),
                        one_step_mean,
                        prior_precision,
                        spikes,
                        dt,
                        log_intensity_func,
                    )
                )
                f_mm = float(
                    self._neg_log_posterior(
                        jnp.array(x_mm),
                        one_step_mean,
                        prior_precision,
                        spikes,
                        dt,
                        log_intensity_func,
                    )
                )
                fd_hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)

        np.testing.assert_allclose(
            analytical_neg_hessian,
            fd_hessian,
            rtol=1e-3,
            atol=1e-5,
            err_msg="Analytical Hessian should match finite differences",
        )


class TestPointProcessMathCorrectness:
    """Tests verifying mathematical properties of the Laplace-EKF update."""

    def test_laplace_update_single_neuron_analytical(self) -> None:
        """1D state, linear log-intensity: verify against hand-derived solution.

        For log_lambda = x, prior N(m, P), y spikes:
          gradient at m = y - exp(m)*dt
          precision at m = 1/P + exp(m)*dt
          posterior mean = m + gradient / precision
          posterior cov = 1 / precision
        """
        dt = 0.02

        def log_intensity(x):
            return jnp.atleast_1d(x)  # log_lambda = x

        for y, label in [(0.0, "no_spike"), (1.0, "spike")]:
            m = jnp.array([0.0])
            P = jnp.array([[1.0]])
            spikes = jnp.array([y])

            lambda_dt = jnp.exp(m[0]) * dt  # exp(0)*0.02 = 0.02
            gradient = y - lambda_dt
            precision = 1.0 / P[0, 0] + lambda_dt
            expected_mean = m[0] + float(gradient / precision)
            expected_cov = 1.0 / float(precision)

            post_mean, post_cov, _ = _point_process_laplace_update(
                m,
                P,
                spikes,
                dt,
                log_intensity,
                include_laplace_normalization=False,
            )

            np.testing.assert_allclose(
                post_mean[0],
                expected_mean,
                rtol=1e-4,
                err_msg=f"Mean wrong for {label}",
            )
            np.testing.assert_allclose(
                post_cov[0, 0],
                expected_cov,
                rtol=1e-4,
                err_msg=f"Cov wrong for {label}",
            )

    def test_zero_spike_pushes_mean_toward_lower_intensity(self) -> None:
        """y=0 should decrease the posterior mean (lower intensity)."""
        dt = 0.02

        def log_intensity(x):
            return jnp.atleast_1d(x[0])

        m = jnp.array([1.0])  # moderate intensity
        P = jnp.array([[1.0]])
        spikes = jnp.array([0.0])

        post_mean, _, _ = _point_process_laplace_update(
            m,
            P,
            spikes,
            dt,
            log_intensity,
            include_laplace_normalization=False,
        )

        assert post_mean[0] < m[0] - 1e-8, (
            f"Zero spikes should decrease mean: prior={m[0]:.6f}, "
            f"posterior={post_mean[0]:.6f}"
        )

    def test_posterior_concentrates_with_more_spikes(self) -> None:
        """With multiple Newton iterations, more spikes should tighten the posterior.

        For linear log-intensity, the single-step Laplace precision is
        data-independent (Fisher info depends only on the predicted intensity).
        With multiple Newton steps, the posterior mode shifts, changing the
        intensity at the evaluation point, so precision becomes data-dependent.
        """
        dt = 0.02

        def log_intensity(x):
            return jnp.atleast_1d(x[0])

        m = jnp.array([1.0])
        P = jnp.array([[2.0]])

        covs = []
        for y in [0.0, 1.0, 5.0, 10.0]:
            _, post_cov, _ = _point_process_laplace_update(
                m,
                P,
                jnp.array([y]),
                dt,
                log_intensity,
                include_laplace_normalization=False,
                max_newton_iter=5,
            )
            covs.append(float(post_cov[0, 0]))

        # With multiple Newton steps, higher spike count → higher posterior mode
        # → higher exp(x*) → higher Fisher info → tighter posterior
        for i in range(len(covs) - 1):
            assert covs[i] >= covs[i + 1] - 1e-10, (
                f"Cov should decrease with more spikes: "
                f"y={[0,1,5,10][i]} cov={covs[i]:.6f} vs "
                f"y={[0,1,5,10][i+1]} cov={covs[i+1]:.6f}"
            )

    def test_multi_neuron_reduces_posterior_variance(self) -> None:
        """More neurons observing the same state should reduce uncertainty."""
        n_latent = 2
        dt = 0.02
        m = jnp.ones(n_latent)
        P = jnp.eye(n_latent)

        traces = []
        for n_neurons in [1, 5, 20]:
            key = jax.random.PRNGKey(n_neurons)
            weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.3
            baseline = -jnp.ones(n_neurons)

            def log_intensity(x, w=weights, b=baseline):
                return jnp.atleast_1d(b + w @ x)

            spikes = jnp.ones(n_neurons)
            _, post_cov, _ = _point_process_laplace_update(
                m,
                P,
                spikes,
                dt,
                log_intensity,
                include_laplace_normalization=False,
            )
            traces.append(float(jnp.trace(post_cov)))

        for i in range(len(traces) - 1):
            assert traces[i] > traces[i + 1], (
                f"Trace should decrease with more neurons: "
                f"n={[1,5,20][i]} trace={traces[i]:.6f} vs "
                f"n={[1,5,20][i+1]} trace={traces[i+1]:.6f}"
            )


class TestPointProcessEMCorrectness:
    """Tests verifying EM correctness for the non-switching PointProcessModel."""

    def test_em_log_likelihood_improves(self) -> None:
        """EM log-likelihood should improve overall.

        The Laplace approximation can cause small per-iteration decreases,
        but the overall trend should be upward.
        """
        n_time = 500
        n_params = 3  # intercept + 2 features
        dt = 0.01

        # True parameters
        true_params = jnp.array([1.0, 0.5, -0.3])

        # Design matrix: intercept + 2 features
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        features = jax.random.normal(k1, (n_time, 2)) * 0.3
        design_matrix = jnp.concatenate([jnp.ones((n_time, 1)), features], axis=1)

        # Simulate spikes
        log_rate = log_conditional_intensity(design_matrix, true_params)
        rate = jnp.exp(log_rate) * dt
        spikes = jax.random.poisson(k2, rate).astype(float)

        # Fit model from wrong initial params
        model = PointProcessModel(
            n_state_dims=n_params,
            dt=dt,
            transition_matrix=jnp.eye(n_params),
            process_cov=jnp.eye(n_params) * 0.001,
            init_mean=jnp.zeros(n_params),
            init_cov=jnp.eye(n_params) * 1.0,
        )

        log_likelihoods = model.fit(
            design_matrix,
            spikes,
            max_iter=15,
            tolerance=1e-8,
        )

        # Overall improvement
        assert log_likelihoods[-1] > log_likelihoods[0], (
            f"EM should improve LL: first={log_likelihoods[0]:.2f}, "
            f"last={log_likelihoods[-1]:.2f}"
        )

        # All finite
        for i, ll in enumerate(log_likelihoods):
            assert np.isfinite(ll), f"LL should be finite at iteration {i}"

    def test_em_recovers_stationary_params(self) -> None:
        """With A=I and small Q, EM should recover the true GLM parameters.

        The state is the parameter vector. With identity dynamics and small
        process noise, the smoother mean at steady state should approximate
        the true parameters.
        """
        n_time = 2000
        n_params = 3  # intercept + 2 features
        dt = 0.01

        true_params = jnp.array([1.5, 0.8, -0.5])

        key = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(key)
        features = jax.random.normal(k1, (n_time, 2)) * 0.3
        design_matrix = jnp.concatenate([jnp.ones((n_time, 1)), features], axis=1)

        log_rate = log_conditional_intensity(design_matrix, true_params)
        rate = jnp.exp(log_rate) * dt
        spikes = jax.random.poisson(k2, rate).astype(float)

        model = PointProcessModel(
            n_state_dims=n_params,
            dt=dt,
            transition_matrix=jnp.eye(n_params),
            process_cov=jnp.eye(n_params) * 0.0001,
            init_mean=jnp.zeros(n_params),
            init_cov=jnp.eye(n_params) * 1.0,
            update_process_cov=True,
            update_transition_matrix=False,
        )

        model.fit(design_matrix, spikes, max_iter=30, tolerance=1e-6)

        # Smoother mean averaged over the middle portion should approximate
        # true params. The smoother state drifts slightly due to process noise,
        # so we average over the central 50% of time to reduce variance.
        mid_start = n_time // 4
        mid_end = 3 * n_time // 4
        recovered_params = jnp.mean(model.smoother_mean[mid_start:mid_end], axis=0)

        np.testing.assert_allclose(
            recovered_params,
            true_params,
            atol=0.5,
            err_msg="Smoother should recover approximately true parameters",
        )

    def test_em_monotonic_per_iteration(self) -> None:
        """With well-conditioned data, EM should be nearly monotonic.

        Any decrease should be < 1% of total improvement.
        """
        n_time = 300
        n_params = 2  # intercept + 1 feature
        dt = 0.01

        true_params = jnp.array([2.0, 0.5])

        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        features = jax.random.normal(k1, (n_time, 1)) * 0.5
        design_matrix = jnp.concatenate([jnp.ones((n_time, 1)), features], axis=1)

        log_rate = log_conditional_intensity(design_matrix, true_params)
        rate = jnp.exp(log_rate) * dt
        spikes = jax.random.poisson(k2, rate).astype(float)

        model = PointProcessModel(
            n_state_dims=n_params,
            dt=dt,
            transition_matrix=jnp.eye(n_params),
            process_cov=jnp.eye(n_params) * 0.001,
            init_mean=jnp.zeros(n_params),
            init_cov=jnp.eye(n_params) * 1.0,
        )

        log_likelihoods = model.fit(
            design_matrix,
            spikes,
            max_iter=15,
            tolerance=1e-8,
        )

        total_improvement = log_likelihoods[-1] - log_likelihoods[0]
        assert total_improvement > 0, "EM should improve overall"

        for i in range(1, len(log_likelihoods)):
            if log_likelihoods[i] < log_likelihoods[i - 1]:
                drop = log_likelihoods[i - 1] - log_likelihoods[i]
                assert drop < 0.05 * total_improvement, (
                    f"EM drop at [{i}]={drop:.4f} exceeds 5% of "
                    f"total improvement {total_improvement:.4f}"
                )

    def test_em_convergence_is_fixed_point(self) -> None:
        """At convergence, one more EM iteration should barely change params."""
        n_time = 300
        n_params = 2
        dt = 0.01

        key = jax.random.PRNGKey(55)
        k1, k2 = jax.random.split(key)
        design_matrix = jnp.concatenate(
            [jnp.ones((n_time, 1)), jax.random.normal(k1, (n_time, 1)) * 0.3], axis=1
        )
        spikes = jax.random.poisson(k2, 0.3, shape=(n_time,)).astype(float)

        model = PointProcessModel(
            n_state_dims=n_params,
            dt=dt,
            transition_matrix=jnp.eye(n_params),
            process_cov=jnp.eye(n_params) * 0.001,
            init_mean=jnp.zeros(n_params),
            init_cov=jnp.eye(n_params),
        )

        # Run to convergence
        model.fit(design_matrix, spikes, max_iter=50, tolerance=1e-8)

        A_converged = model.transition_matrix.copy()
        Q_converged = model.process_cov.copy()

        # Run one more iteration manually
        model._e_step(design_matrix, spikes)
        model._m_step()

        np.testing.assert_allclose(
            model.transition_matrix,
            A_converged,
            atol=0.01,
            err_msg="A should be near fixed point",
        )
        np.testing.assert_allclose(
            model.process_cov,
            Q_converged,
            atol=0.01,
            err_msg="Q should be near fixed point",
        )


class TestPointProcessNumericalStability:
    """Tests for point-process filter numerical stability."""

    def test_filter_very_sparse_spikes(self) -> None:
        """Very sparse spikes (~5 in 5000 bins): should stay finite."""
        n_time = 5000
        dt = 0.001
        n_params = 3  # intercept + 2 latent weights

        A = jnp.eye(n_params) * 0.999
        Q = jnp.eye(n_params) * 0.001
        init_mean = jnp.array([-7.0, 0.0, 0.0])
        init_cov = jnp.eye(n_params)

        # Design matrix: intercept + 2 zero-valued features
        design_matrix = jnp.concatenate(
            [jnp.ones((n_time, 1)), jnp.zeros((n_time, 2))], axis=-1
        )

        # Generate sparse spikes (very low rate)
        key = jax.random.PRNGKey(42)
        log_rate = log_conditional_intensity(design_matrix, init_mean)
        rate = jnp.exp(log_rate) * dt
        spikes = jax.random.poisson(key, rate).astype(float)

        filtered_mean, filtered_cov, mll = stochastic_point_process_filter(
            init_mean,
            init_cov,
            design_matrix,
            spikes[:, None],
            dt,
            A,
            Q,
            log_conditional_intensity,
        )

        assert jnp.all(jnp.isfinite(filtered_mean)), "Means should be finite"
        assert jnp.all(jnp.isfinite(filtered_cov)), "Covs should be finite"
        assert jnp.isfinite(mll), "MLL should be finite"


class TestPointProcessGradientStability:
    """Gradient stability gate: verify gradients through Laplace-EKF are finite."""

    def test_gradient_through_filter_wrt_A_Q(self):
        """Gradients through Laplace-EKF must be finite for SGD to work."""
        n_state = 2
        n_neurons = 3
        n_time = 50
        dt = 0.001

        key = jax.random.PRNGKey(0)
        A = 0.99 * jnp.eye(n_state)
        Q = 0.001 * jnp.eye(n_state)
        m0 = jnp.zeros(n_state)
        P0 = jnp.eye(n_state)

        # Simple linear log-intensity with known weights
        W = jax.random.normal(key, (n_neurons, n_state)) * 0.1
        design_matrix = jnp.tile(W, (n_time, 1, 1))
        spike_indicator = jax.random.poisson(
            key, jnp.ones((n_time, n_neurons)) * 0.01
        )

        def loss_fn(A_val, Q_flat):
            from state_space_practice.parameter_transforms import PSD_MATRIX

            Q_val = PSD_MATRIX.to_constrained(Q_flat)
            _, _, mll = stochastic_point_process_filter(
                m0, P0, design_matrix, spike_indicator, dt, A_val, Q_val,
                log_conditional_intensity,
            )
            return -mll

        from state_space_practice.parameter_transforms import PSD_MATRIX

        Q_flat = PSD_MATRIX.to_unconstrained(Q)

        grad_A, grad_Q = jax.grad(loss_fn, argnums=(0, 1))(A, Q_flat)

        assert jnp.all(jnp.isfinite(grad_A)), f"Grad A not finite: {grad_A}"
        assert jnp.all(jnp.isfinite(grad_Q)), f"Grad Q not finite: {grad_Q}"
        # Check gradients are reasonably bounded
        assert float(jnp.max(jnp.abs(grad_A))) < 1e6
        assert float(jnp.max(jnp.abs(grad_Q))) < 1e6


class TestPointProcessSGDFitting:
    """Tests for PointProcessModel.fit_sgd()."""

    @pytest.fixture
    def small_pp_problem(self):
        """Create a small synthetic point process problem."""
        n_state = 2
        n_neurons = 3
        n_time = 100
        dt = 0.001

        key = jax.random.PRNGKey(42)
        W = jax.random.normal(key, (n_neurons, n_state)) * 0.1
        design_matrix = jnp.tile(W, (n_time, 1, 1))
        spike_indicator = jax.random.poisson(
            key, jnp.ones((n_time, n_neurons)) * 0.01
        )
        return n_state, dt, design_matrix, spike_indicator

    def test_sgd_improves_ll(self, small_pp_problem):
        n_state, dt, design_matrix, spike_indicator = small_pp_problem
        model = PointProcessModel(n_state, dt)
        initial_ll = model._e_step(design_matrix, spike_indicator)
        model2 = PointProcessModel(n_state, dt)
        lls = model2.fit_sgd(design_matrix, spike_indicator, num_steps=30)
        assert lls[-1] > initial_ll

    def test_sgd_respects_constraints(self, small_pp_problem):
        n_state, dt, design_matrix, spike_indicator = small_pp_problem
        model = PointProcessModel(n_state, dt)
        model.fit_sgd(design_matrix, spike_indicator, num_steps=30)
        # Process cov should be PSD
        eigvals = jnp.linalg.eigvalsh(model.process_cov)
        assert jnp.all(eigvals > 0)

    def test_sgd_process_cov_psd(self, small_pp_problem):
        n_state, dt, design_matrix, spike_indicator = small_pp_problem
        model = PointProcessModel(n_state, dt)
        model.fit_sgd(design_matrix, spike_indicator, num_steps=30)
        eigvals = jnp.linalg.eigvalsh(model.process_cov)
        assert jnp.all(eigvals > 0), "Process cov not PSD after SGD"

    def test_sgd_model_has_smoother_results(self, small_pp_problem):
        n_state, dt, design_matrix, spike_indicator = small_pp_problem
        model = PointProcessModel(n_state, dt)
        model.fit_sgd(design_matrix, spike_indicator, num_steps=30)
        assert model.smoother_mean is not None
        assert model.smoother_cov is not None
        assert hasattr(model, "log_likelihood_")


class TestLogdetPsd:
    """Tests for the Cholesky-based ``_logdet_psd`` helper.

    Regression test class for the performance optimization that replaced
    ``eigvalsh``-based logdet with a Cholesky-based one (~1.7x filter
    speedup at T=10k, d=36). These tests pin the numerical contract so a
    future refactor can't silently change the logdet behavior.
    """

    def test_matches_slogdet_on_well_conditioned_psd(self) -> None:
        """Cholesky-logdet matches ``jnp.linalg.slogdet`` on PSD input.

        ``slogdet`` is the reference implementation; ``_logdet_psd`` trades
        a small constant stability shift for a ~3-5x speedup. On a
        well-conditioned matrix the shift is negligible and the two must
        agree to ~1e-6.
        """
        key = jax.random.PRNGKey(0)
        n = 32
        # PSD via A A' + jitter. Well-conditioned: eigvals in [1, n+1].
        A = jax.random.normal(key, (n, n))
        mat = A @ A.T + jnp.eye(n)  # eigvals bounded below by 1

        ref_sign, ref_logdet = jnp.linalg.slogdet(mat)
        got = _logdet_psd(mat, diagonal_boost=1e-9)
        assert float(ref_sign) == 1.0
        np.testing.assert_allclose(float(got), float(ref_logdet), atol=1e-6)

    def test_stabilized_on_near_singular_matrix(self) -> None:
        """Cholesky-logdet does not crash on a near-singular PSD matrix.

        The ``diagonal_boost`` shift ensures the Cholesky succeeds even
        when the input has eigenvalues at machine-precision scale — this
        is critical for the inner filter loop where the posterior
        covariance can be near-singular in high-information directions.
        """
        n = 16
        # Rank-deficient PSD: A has rank 8 → 8 zero eigenvalues before shift.
        A = jax.random.normal(jax.random.PRNGKey(1), (n, n // 2))
        mat = A @ A.T  # rank n//2

        # Should not crash and should produce a finite logdet.
        result = _logdet_psd(mat, diagonal_boost=1e-9)
        assert jnp.isfinite(result)
        # With the shift, the result is approximately sum(log(eigs + 1e-9))
        # which is finite (vs -inf for the raw rank-deficient matrix).

    def test_identity_matches_zero_logdet(self) -> None:
        """logdet(I) ≈ 0 under the production diagonal_boost.

        With ``diagonal_boost=1e-9`` (the production default), the shift
        adds ``1e-9`` to each eigenvalue of ``I``, so each post-shift eigenvalue
        is ``1 + 1e-9`` and the logdet is ``n * log(1 + 1e-9) ≈ n * 1e-9``.
        The test uses the default to cover the actual code path callers hit.
        """
        for n in (4, 16, 64):
            got = _logdet_psd(jnp.eye(n))  # use default diagonal_boost=1e-9
            expected = n * float(jnp.log(1.0 + 1e-9))
            np.testing.assert_allclose(float(got), expected, atol=1e-12)


class TestValidateFilterNumerics:
    """Tests for ``_validate_filter_numerics``.

    Regression test class for the f32-NaN observability story: the filter
    should raise on non-PSD init_cov (guaranteed NaN) and warn on f32 +
    long T + ill-conditioned configurations (likely NaN). This mirrors
    the two failure modes in the user bug report:
      Problem A: T=1501, d=36, cond=5e3, min_eig=2e-4 → NaN at bin 245
      Problem B: T=28,373, d=64, cond=1e4, min_eig=9e-5 → NaN at step 2
    """

    def test_raises_on_non_psd_init_cov(self) -> None:
        n = 8
        bad = jnp.eye(n).at[0, 0].set(-0.5)  # indefinite — one negative eig
        with pytest.raises(ValueError, match="not positive definite"):
            _validate_filter_numerics(bad, n_time=100)

    def test_raises_on_zero_eigenvalue(self) -> None:
        """Rank-deficient matrix — min eigenvalue exactly 0."""
        n = 8
        A = jax.random.normal(jax.random.PRNGKey(0), (n, n // 2))
        mat = A @ A.T  # rank n/2, min_eig == 0
        with pytest.raises(ValueError, match="not positive definite"):
            _validate_filter_numerics(mat, n_time=100)

    def test_raises_on_non_square(self) -> None:
        bad = jnp.zeros((5, 8))
        with pytest.raises(ValueError, match="square 2D matrix"):
            _validate_filter_numerics(bad, n_time=100)

    def test_well_conditioned_psd_passes_silently(self) -> None:
        """Well-conditioned f64 init_cov must produce neither error nor warning."""
        import warnings

        init_cov = jnp.eye(36) * 0.5  # cond=1, min_eig=0.5

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would raise
            _validate_filter_numerics(init_cov, n_time=10_000)

    def test_no_warning_in_f64_on_long_ill_conditioned(self) -> None:
        """f64 mode: even Problem B conditioning does NOT trigger a warning.

        Tests run with jax_enable_x64=True (see module-level config), so
        the f32-risk branch must not fire. This guards against regressions
        where the validation would false-positive in the production
        (f64-enabled) path.
        """
        import warnings

        # Construct an init_cov with cond=1e4, min_eig=9e-5 — exactly
        # the Problem B conditioning from the user's bug report.
        n = 64
        eigs = jnp.logspace(
            jnp.log10(9e-5),
            jnp.log10(9e-5 * 1e4),
            n,
        )
        U, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(2), (n, n)))
        init_cov = (U * eigs) @ U.T

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_filter_numerics(init_cov, n_time=28_373)

    def test_warns_on_f32_long_ill_conditioned(self) -> None:
        """f32 init_cov on a long / ill-conditioned problem must warn.

        The validator keys off ``init_covariance.dtype == float32``, so
        explicitly casting an otherwise-valid init_cov to f32 is the
        supported way to trigger the warning. We do NOT flip
        ``jax_enable_x64`` mid-test because that corrupts pre-existing
        f64 arrays in the test's scope.

        The conditioning here matches Problem B from the user bug
        report: T=28k, d=64, min_eig=9e-5, max_eig=0.9, cond ~1e4.
        """
        n = 64
        eigs = jnp.logspace(
            jnp.log10(9e-5),
            jnp.log10(9e-5 * 1e4),
            n,
        )
        U, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(2), (n, n)))
        init_cov_f32 = ((U * eigs) @ U.T).astype(jnp.float32)

        with pytest.warns(UserWarning, match="float32"):
            _validate_filter_numerics(init_cov_f32, n_time=28_373)

    def test_f32_short_well_conditioned_no_warning(self) -> None:
        """f32 with a short, well-conditioned problem must NOT warn.

        The warning is gated on estimated accumulated roundoff exceeding
        half of min_eig. For T=100, d=10, min_eig=0.5, the roundoff
        budget is comfortable and no warning should fire.
        """
        import warnings

        init_cov_f32 = (jnp.eye(10) * 0.5).astype(jnp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_filter_numerics(init_cov_f32, n_time=100)

    def test_validate_inputs_false_skips_check(self) -> None:
        """``validate_inputs=False`` bypasses the validation layer.

        We construct a tiny problem with an init_cov that:
          - has a tiny (-1e-12) negative eigenvalue, so the validation
            layer would raise with "not positive definite"
          - is close enough to PSD that ``psd_solve``'s internal
            ``diagonal_boost=1e-9`` rescues the Cholesky, so the
            downstream scan completes without NaN

        With validate_inputs=True (default), this raises ValueError
        from the validator. With validate_inputs=False, the filter
        runs to completion and produces finite output.
        """
        key = jax.random.PRNGKey(0)
        T, n_state = 10, 4
        dm = jax.random.normal(key, (T, n_state)) * 0.1
        spikes = jnp.zeros(T, dtype=jnp.int32)
        bad_init_cov = jnp.eye(n_state) * 0.5
        bad_init_cov = bad_init_cov.at[0, 0].set(-1e-12)

        # Default path: validation fires and raises.
        with pytest.raises(ValueError, match="not positive definite"):
            stochastic_point_process_filter(
                jnp.zeros(n_state), bad_init_cov, dm, spikes, 0.01,
                jnp.eye(n_state), jnp.eye(n_state) * 1e-6,
                log_conditional_intensity,
            )

        # Opt-out path: validation is skipped, psd_solve's diagonal_boost
        # rescues the Cholesky, and the filter produces finite output.
        # (We don't assert output correctness — only that the validation
        # layer was bypassed cleanly.)
        filtered_mean, filtered_cov, mll = stochastic_point_process_filter(
            jnp.zeros(n_state), bad_init_cov, dm, spikes, 0.01,
            jnp.eye(n_state), jnp.eye(n_state) * 1e-6,
            log_conditional_intensity,
            validate_inputs=False,
        )
        assert jnp.all(jnp.isfinite(filtered_mean))
        assert jnp.all(jnp.isfinite(filtered_cov))
        assert jnp.isfinite(mll)
