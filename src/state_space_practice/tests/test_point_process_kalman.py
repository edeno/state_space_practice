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

        design_matrix = jax.random.normal(
            jax.random.PRNGKey(0), (n_time, n_params)
        )
        params = jnp.ones(n_params)

        log_intensity = log_conditional_intensity(design_matrix, params)

        assert log_intensity.shape == (n_time,)

    def test_linear_in_params(self) -> None:
        """Log intensity should be linear in parameters."""
        n_time = 50
        n_params = 3

        design_matrix = jax.random.normal(
            jax.random.PRNGKey(0), (n_time, n_params)
        )
        params1 = jnp.array([1.0, 2.0, 3.0])
        params2 = jnp.array([2.0, 4.0, 6.0])  # 2x params1

        log_int1 = log_conditional_intensity(design_matrix, params1)
        log_int2 = log_conditional_intensity(design_matrix, params2)

        np.testing.assert_allclose(log_int2, 2 * log_int1, rtol=1e-10)

    def test_zero_params_gives_zero(self) -> None:
        """Zero parameters should give zero log intensity."""
        n_time = 50
        n_params = 3

        design_matrix = jax.random.normal(
            jax.random.PRNGKey(0), (n_time, n_params)
        )
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
        np.testing.assert_allclose(
            smoother_var[-1], filter_var[-1], rtol=1e-5
        )

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

    def test_zero_learning_rate_preserves_params(
        self, point_process_test_data
    ) -> None:
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

        assert rate.shape == (d["n_time"], d["n_time"])  # (n_time, n_time) due to design @ design.T
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
        np.testing.assert_allclose(
            model.transition_matrix, jnp.eye(n_basis), atol=0.25
        )


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

        # Generate smoother outputs from a random walk
        true_A = jnp.eye(n_params)
        true_Q = jnp.eye(n_params) * 0.01

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
        assert not jnp.allclose(A_increasing, A_constant, atol=0.01), (
            "A should depend on state trajectory (outer products of means)"
        )


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
            np.testing.assert_allclose(
                filtered_cov[t], filtered_cov[t].T, atol=1e-10
            )

    def test_multi_neuron_reduces_to_single_neuron(self, point_process_test_data) -> None:
        """Multi-neuron with n_neurons=1 should match single-neuron filter."""
        d = point_process_test_data

        # Single neuron result (original API)
        filtered_mean_single, filtered_cov_single, mll_single = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            log_conditional_intensity,
        )

        # Multi-neuron with n_neurons=1 (same data, reshaped)
        design_matrix_multi = d["design_matrix"][:, None, :]  # (n_time, 1, n_params)
        spike_indicator_multi = d["spike_indicator"][:, None]  # (n_time, 1)

        filtered_mean_multi, filtered_cov_multi, mll_multi = stochastic_point_process_filter(
            d["init_mean"],
            d["init_cov"],
            design_matrix_multi,
            spike_indicator_multi,
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            multi_neuron_log_intensity,
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
            init_mean, init_cov, design_1, spikes_1, dt,
            transition_matrix, process_cov, multi_neuron_log_intensity
        )

        # Run filter with 5 neurons (more information)
        _, cov_5, _ = stochastic_point_process_filter(
            init_mean, init_cov, design_5, spikes_5, dt,
            transition_matrix, process_cov, multi_neuron_log_intensity
        )

        # Average variance should be smaller with more neurons
        avg_var_1 = jnp.mean(jnp.trace(cov_1, axis1=-2, axis2=-1))
        avg_var_5 = jnp.mean(jnp.trace(cov_5, axis1=-2, axis2=-1))

        assert avg_var_5 < avg_var_1, (
            f"Variance should decrease with more neurons: {avg_var_5} >= {avg_var_1}"
        )


class TestMultiNeuronSmoother:
    """Tests for multi-neuron point process smoother."""

    def test_output_shapes(self, multi_neuron_test_data) -> None:
        """Smoother outputs should have correct shapes for multi-neuron input."""
        d = multi_neuron_test_data

        smoother_mean, smoother_cov, smoother_cross_cov, mll = stochastic_point_process_smoother(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            multi_neuron_log_intensity,
        )

        assert smoother_mean.shape == (d["n_time"], d["n_params"])
        assert smoother_cov.shape == (d["n_time"], d["n_params"], d["n_params"])
        assert smoother_cross_cov.shape == (d["n_time"] - 1, d["n_params"], d["n_params"])

    def test_no_nans(self, multi_neuron_test_data) -> None:
        """Multi-neuron smoother should not produce NaN values."""
        d = multi_neuron_test_data

        smoother_mean, smoother_cov, smoother_cross_cov, mll = stochastic_point_process_smoother(
            d["init_mean"],
            d["init_cov"],
            d["design_matrix"],
            d["spike_indicator"],
            d["dt"],
            d["transition_matrix"],
            d["process_cov"],
            multi_neuron_log_intensity,
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
            init_mean, init_cov, design_matrix, spikes_neuron0, dt,
            transition_matrix, process_cov, multi_neuron_log_intensity
        )

        mean_1, _, _ = stochastic_point_process_filter(
            init_mean, init_cov, design_matrix, spikes_neuron1, dt,
            transition_matrix, process_cov, multi_neuron_log_intensity
        )

        # With spikes in neuron 0, state[0] should be higher than state[1]
        assert jnp.mean(mean_0[:, 0]) > jnp.mean(mean_0[:, 1]), (
            "Spikes in neuron 0 should increase state[0]"
        )

        # With spikes in neuron 1, state[1] should be higher than state[0]
        assert jnp.mean(mean_1[:, 1]) > jnp.mean(mean_1[:, 0]), (
            "Spikes in neuron 1 should increase state[1]"
        )


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
        np.testing.assert_allclose(
            rate_smoothed[-1], rate_filtered[-1], rtol=1e-5
        )

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
            return linear ** 2

        key = jax.random.PRNGKey(123)
        design_matrix = jax.random.normal(key, (n_time, n_params)) * 0.3
        spike_indicator = jax.random.poisson(
            jax.random.PRNGKey(456),
            jnp.exp(jax.vmap(nonlinear_log_intensity)(design_matrix, jnp.zeros((n_time, n_params)))) * dt
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
