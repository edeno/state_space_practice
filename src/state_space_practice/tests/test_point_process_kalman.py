"""Tests for the point_process_kalman module.

This module tests point process filters and smoothers for neural encoding,
including stochastic filters, smoothers, and steepest descent methods.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.point_process_kalman import (
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
