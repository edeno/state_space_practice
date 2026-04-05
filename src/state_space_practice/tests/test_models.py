"""Tests for the models module.

This module tests receptive field models and point process filters
used for neural encoding analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.models import (
    get_confidence_interval,
    log_receptive_field_model,
    steepest_descent_point_process_filter,
    stochastic_point_process_filter,
)

# Enable 64-bit precision for numerical stability in tests
jax.config.update("jax_enable_x64", True)


class TestLogReceptiveFieldModel:
    """Tests for the log_receptive_field_model function."""

    def test_output_is_scalar_for_scalar_input(self) -> None:
        """Scalar position input should produce scalar output."""
        position = jnp.array(150.0)
        params = jnp.array([jnp.log(10.0), 150.0, 10.0])
        log_rate = log_receptive_field_model(position, params)

        assert log_rate.shape == ()

    def test_max_log_rate_at_center(self) -> None:
        """Log rate should equal log_max_rate at the place field center."""
        center = 150.0
        log_max_rate = jnp.log(25.0)
        params = jnp.array([log_max_rate, center, 10.0])

        log_rate = log_receptive_field_model(jnp.array(center), params)
        np.testing.assert_allclose(log_rate, log_max_rate, rtol=1e-10)

    def test_gaussian_decay_in_log_space(self) -> None:
        """Log rate should decrease quadratically from center."""
        center = 150.0
        scale = 10.0
        log_max_rate = jnp.log(20.0)
        params = jnp.array([log_max_rate, center, scale])

        # At one standard deviation
        position = jnp.array(center + scale)
        log_rate = log_receptive_field_model(position, params)

        # Expected: log_max_rate - 0.5
        expected = log_max_rate - 0.5
        np.testing.assert_allclose(log_rate, expected, rtol=1e-10)

    def test_symmetric_around_center(self) -> None:
        """Log rate should be symmetric around center."""
        center = 150.0
        params = jnp.array([jnp.log(10.0), center, 15.0])

        positions_left = jnp.array([center - 10, center - 20, center - 30])
        positions_right = jnp.array([center + 10, center + 20, center + 30])

        # Use vmap to evaluate at multiple positions
        log_rates_left = jax.vmap(lambda p: log_receptive_field_model(p, params))(
            positions_left
        )
        log_rates_right = jax.vmap(lambda p: log_receptive_field_model(p, params))(
            positions_right
        )

        np.testing.assert_allclose(log_rates_left, log_rates_right, rtol=1e-10)

    def test_gradients_exist(self) -> None:
        """Function should be differentiable with respect to params."""
        position = jnp.array(150.0)
        params = jnp.array([jnp.log(10.0), 150.0, 10.0])

        grad_fn = jax.grad(log_receptive_field_model, argnums=1)
        grads = grad_fn(position, params)

        assert grads.shape == params.shape
        assert not jnp.any(jnp.isnan(grads))

    def test_hessian_exists(self) -> None:
        """Function should have valid Hessian."""
        position = jnp.array(150.0)
        params = jnp.array([jnp.log(10.0), 150.0, 10.0])

        hess_fn = jax.hessian(log_receptive_field_model, argnums=1)
        hess = hess_fn(position, params)

        assert hess.shape == (3, 3)
        assert not jnp.any(jnp.isnan(hess))


@pytest.fixture(scope="module")
def simple_point_process_model():
    """Provides parameters for a simple point process filter test."""
    n_time = 100
    n_params = 3
    dt = 0.02

    # Initial parameters
    init_mode = jnp.array([jnp.log(10.0), 150.0, 10.0])
    init_cov = jnp.eye(n_params) * 0.1

    # State transition (random walk with small drift)
    transition_matrix = jnp.eye(n_params)
    latent_state_cov = jnp.eye(n_params) * 0.001

    # Simulate position (back and forth on linear track)
    position = jnp.linspace(0, 300, n_time)

    # Simulate spikes based on true parameters
    key = jax.random.PRNGKey(42)
    true_rate = jnp.exp(
        init_mode[0] - (position - init_mode[1]) ** 2 / (2 * init_mode[2] ** 2)
    )
    spike_indicator = jax.random.poisson(key, true_rate * dt)

    return {
        "init_mode": init_mode,
        "init_cov": init_cov,
        "position": position,
        "spike_indicator": spike_indicator,
        "dt": dt,
        "transition_matrix": transition_matrix,
        "latent_state_cov": latent_state_cov,
        "n_time": n_time,
        "n_params": n_params,
    }


class TestStochasticPointProcessFilter:
    """Tests for the stochastic_point_process_filter function."""

    def test_output_shapes(self, simple_point_process_model) -> None:
        """Filter should output correct shapes."""
        m = simple_point_process_model

        posterior_mode, posterior_cov = stochastic_point_process_filter(
            m["init_mode"],
            m["init_cov"],
            m["position"],
            m["spike_indicator"],
            m["dt"],
            m["transition_matrix"],
            m["latent_state_cov"],
            log_receptive_field_model,
        )

        assert posterior_mode.shape == (m["n_time"], m["n_params"])
        assert posterior_cov.shape == (m["n_time"], m["n_params"], m["n_params"])

    def test_no_nans_in_output(self, simple_point_process_model) -> None:
        """Filter should not produce NaN values."""
        m = simple_point_process_model

        posterior_mode, posterior_cov = stochastic_point_process_filter(
            m["init_mode"],
            m["init_cov"],
            m["position"],
            m["spike_indicator"],
            m["dt"],
            m["transition_matrix"],
            m["latent_state_cov"],
            log_receptive_field_model,
        )

        assert not jnp.any(jnp.isnan(posterior_mode))
        assert not jnp.any(jnp.isnan(posterior_cov))

    def test_covariance_symmetric(self, simple_point_process_model) -> None:
        """Posterior covariances should be symmetric."""
        m = simple_point_process_model

        _, posterior_cov = stochastic_point_process_filter(
            m["init_mode"],
            m["init_cov"],
            m["position"],
            m["spike_indicator"],
            m["dt"],
            m["transition_matrix"],
            m["latent_state_cov"],
            log_receptive_field_model,
        )

        for t in range(m["n_time"]):
            np.testing.assert_allclose(
                posterior_cov[t], posterior_cov[t].T, rtol=1e-5, atol=1e-10
            )

    def test_with_no_spikes(self, simple_point_process_model) -> None:
        """Filter should handle case with no spikes."""
        m = simple_point_process_model

        no_spikes = jnp.zeros_like(m["spike_indicator"])

        posterior_mode, posterior_cov = stochastic_point_process_filter(
            m["init_mode"],
            m["init_cov"],
            m["position"],
            no_spikes,
            m["dt"],
            m["transition_matrix"],
            m["latent_state_cov"],
            log_receptive_field_model,
        )

        assert not jnp.any(jnp.isnan(posterior_mode))
        assert not jnp.any(jnp.isnan(posterior_cov))

    def test_identity_transition_preserves_params(self) -> None:
        """With identity transition and no process noise, params should be stable."""
        n_time = 50
        n_params = 3
        dt = 0.02

        init_mode = jnp.array([jnp.log(10.0), 150.0, 10.0])
        init_cov = jnp.eye(n_params) * 0.01
        transition_matrix = jnp.eye(n_params)
        latent_state_cov = jnp.zeros((n_params, n_params))  # No process noise

        # Constant position far from place field center (minimal spike influence)
        position = jnp.ones(n_time) * 0.0  # Far from center at 150
        spike_indicator = jnp.zeros(n_time, dtype=jnp.int32)

        posterior_mode, _ = stochastic_point_process_filter(
            init_mode,
            init_cov,
            position,
            spike_indicator,
            dt,
            transition_matrix,
            latent_state_cov,
            log_receptive_field_model,
        )

        # First mode should be close to init (only prediction step, small update)
        np.testing.assert_allclose(posterior_mode[0], init_mode, rtol=0.5)

    def test_large_log_rates_keep_posterior_covariance_finite_and_symmetric(
        self,
    ) -> None:
        """Legacy filter should keep covariances finite under large log-rates."""
        n_time = 3
        n_params = 3
        dt = 0.02

        def large_log_rate_model(position, params):
            del position
            return 800.0 + 0.0 * jnp.sum(params)

        _, posterior_cov = stochastic_point_process_filter(
            init_mode_params=jnp.zeros(n_params),
            init_covariance_params=jnp.eye(n_params) * 0.1,
            x=jnp.zeros(n_time),
            spike_indicator=jnp.zeros(n_time),
            dt=dt,
            transition_matrix=jnp.eye(n_params),
            latent_state_covariance=jnp.eye(n_params) * 1e-3,
            log_receptive_field_model=large_log_rate_model,
        )

        assert jnp.all(jnp.isfinite(posterior_cov))
        for time_ind in range(n_time):
            np.testing.assert_allclose(
                posterior_cov[time_ind],
                posterior_cov[time_ind].T,
                rtol=1e-10,
                atol=1e-14,
            )


class TestSteepestDescentPointProcessFilter:
    """Tests for the steepest_descent_point_process_filter function."""

    def test_output_shape(self, simple_point_process_model) -> None:
        """Filter should output correct shape."""
        m = simple_point_process_model
        epsilon = jnp.eye(m["n_params"]) * 0.001

        posterior_mode = steepest_descent_point_process_filter(
            m["init_mode"],
            m["position"],
            m["spike_indicator"],
            m["dt"],
            epsilon,
            log_receptive_field_model,
        )

        assert posterior_mode.shape == (m["n_time"], m["n_params"])

    def test_no_nans(self, simple_point_process_model) -> None:
        """Filter should not produce NaN values."""
        m = simple_point_process_model
        epsilon = jnp.eye(m["n_params"]) * 0.001

        posterior_mode = steepest_descent_point_process_filter(
            m["init_mode"],
            m["position"],
            m["spike_indicator"],
            m["dt"],
            epsilon,
            log_receptive_field_model,
        )

        assert not jnp.any(jnp.isnan(posterior_mode))

    def test_zero_learning_rate_preserves_params(
        self, simple_point_process_model
    ) -> None:
        """With zero learning rate, params should remain constant."""
        m = simple_point_process_model
        epsilon = jnp.zeros((m["n_params"], m["n_params"]))

        posterior_mode = steepest_descent_point_process_filter(
            m["init_mode"],
            m["position"],
            m["spike_indicator"],
            m["dt"],
            epsilon,
            log_receptive_field_model,
        )

        # All time points should have same params as init
        for t in range(m["n_time"]):
            np.testing.assert_allclose(posterior_mode[t], m["init_mode"], rtol=1e-10)

    def test_with_no_spikes(self, simple_point_process_model) -> None:
        """Filter should handle case with no spikes."""
        m = simple_point_process_model
        epsilon = jnp.eye(m["n_params"]) * 0.001
        no_spikes = jnp.zeros_like(m["spike_indicator"])

        posterior_mode = steepest_descent_point_process_filter(
            m["init_mode"],
            m["position"],
            no_spikes,
            m["dt"],
            epsilon,
            log_receptive_field_model,
        )

        assert not jnp.any(jnp.isnan(posterior_mode))

    def test_parameter_drift_with_spikes(self) -> None:
        """Parameters should drift toward spike locations."""
        n_time = 200
        dt = 0.02

        # Start with place field at 150
        init_mode = jnp.array([jnp.log(10.0), 150.0, 10.0])
        epsilon = jnp.diag(jnp.array([0.0, 0.1, 0.0]))  # Only update center

        # Position always at 200 (away from initial center)
        position = jnp.ones(n_time) * 200.0

        # Spikes at every time point
        spike_indicator = jnp.ones(n_time, dtype=jnp.int32)

        posterior_mode = steepest_descent_point_process_filter(
            init_mode,
            position,
            spike_indicator,
            dt,
            epsilon,
            log_receptive_field_model,
        )

        # Place field center should move toward 200
        final_center = posterior_mode[-1, 1]
        assert final_center > init_mode[1]  # Should have moved toward spike location


class TestGetConfidenceInterval:
    """Tests for the get_confidence_interval function."""

    def test_output_shape(self) -> None:
        """CI output should have correct shape."""
        n_time = 50
        n_params = 3

        posterior_mode = jnp.zeros((n_time, n_params))
        posterior_cov = jnp.stack([jnp.eye(n_params)] * n_time)

        ci = get_confidence_interval(posterior_mode, posterior_cov)

        assert ci.shape == (n_time, n_params, 2)

    def test_lower_bound_less_than_upper(self) -> None:
        """Lower CI bound should always be less than upper."""
        n_time = 50
        n_params = 3

        posterior_mode = jax.random.normal(jax.random.PRNGKey(0), (n_time, n_params))
        posterior_cov = jnp.stack([jnp.eye(n_params) * 0.5] * n_time)

        ci = get_confidence_interval(posterior_mode, posterior_cov)

        lower = ci[..., 0]
        upper = ci[..., 1]

        assert jnp.all(lower < upper)

    def test_mode_within_interval(self) -> None:
        """Posterior mode should be within confidence interval."""
        n_time = 50
        n_params = 3

        posterior_mode = jax.random.normal(jax.random.PRNGKey(0), (n_time, n_params))
        posterior_cov = jnp.stack([jnp.eye(n_params)] * n_time)

        ci = get_confidence_interval(posterior_mode, posterior_cov)

        lower = ci[..., 0]
        upper = ci[..., 1]

        assert jnp.all(posterior_mode >= lower)
        assert jnp.all(posterior_mode <= upper)

    def test_narrower_ci_with_smaller_variance(self) -> None:
        """Smaller variance should produce narrower CI."""
        n_time = 50
        n_params = 3

        posterior_mode = jnp.zeros((n_time, n_params))

        cov_large = jnp.stack([jnp.eye(n_params) * 1.0] * n_time)
        cov_small = jnp.stack([jnp.eye(n_params) * 0.1] * n_time)

        ci_large = get_confidence_interval(posterior_mode, cov_large)
        ci_small = get_confidence_interval(posterior_mode, cov_small)

        width_large = ci_large[..., 1] - ci_large[..., 0]
        width_small = ci_small[..., 1] - ci_small[..., 0]

        assert jnp.all(width_small < width_large)

    def test_different_alpha_levels(self) -> None:
        """Different alpha should produce different CI widths."""
        n_time = 50
        n_params = 3

        posterior_mode = jnp.zeros((n_time, n_params))
        posterior_cov = jnp.stack([jnp.eye(n_params)] * n_time)

        ci_99 = get_confidence_interval(posterior_mode, posterior_cov, alpha=0.01)
        ci_95 = get_confidence_interval(posterior_mode, posterior_cov, alpha=0.05)

        width_99 = ci_99[..., 1] - ci_99[..., 0]
        width_95 = ci_95[..., 1] - ci_95[..., 0]

        # 99% CI should be wider than 95% CI
        assert jnp.all(width_99 > width_95)

    def test_zero_variance_collapses_interval(self) -> None:
        """Zero variance should produce interval collapsed to point."""
        n_time = 10
        n_params = 3

        posterior_mode = jax.random.normal(jax.random.PRNGKey(0), (n_time, n_params))
        posterior_cov = jnp.zeros((n_time, n_params, n_params))

        ci = get_confidence_interval(posterior_mode, posterior_cov)

        lower = ci[..., 0]
        upper = ci[..., 1]

        np.testing.assert_allclose(lower, posterior_mode, rtol=1e-10)
        np.testing.assert_allclose(upper, posterior_mode, rtol=1e-10)
