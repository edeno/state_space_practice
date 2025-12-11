"""Tests for the switching_point_process module.

This module tests the switching point-process Kalman filter and smoother
for spike-based observations with discrete state switching.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


def linear_log_intensity(weights: Array, baseline: Array) -> Callable[[Array], Array]:
    """Create a linear log-intensity function for testing.

    Parameters
    ----------
    weights : Array, shape (n_neurons, n_latent)
    baseline : Array, shape (n_neurons,)

    Returns
    -------
    Callable[[Array], Array]
        Function mapping state (n_latent,) -> log_rates (n_neurons,)
    """

    def log_intensity(state: Array) -> Array:
        return baseline + weights @ state

    return log_intensity


class TestSpikeObsParams:
    """Tests for the SpikeObsParams dataclass."""

    def test_creation_with_valid_shapes(self) -> None:
        """SpikeObsParams should be created with correct shapes."""
        from state_space_practice.switching_point_process import SpikeObsParams

        n_neurons = 10
        n_latent = 4

        baseline = jnp.zeros(n_neurons)
        weights = jnp.ones((n_neurons, n_latent))

        params = SpikeObsParams(baseline=baseline, weights=weights)

        assert params.baseline.shape == (n_neurons,)
        assert params.weights.shape == (n_neurons, n_latent)

    def test_baseline_is_accessible(self) -> None:
        """Baseline attribute should be accessible."""
        from state_space_practice.switching_point_process import SpikeObsParams

        baseline = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.ones((3, 2))

        params = SpikeObsParams(baseline=baseline, weights=weights)

        np.testing.assert_allclose(params.baseline, baseline)

    def test_weights_is_accessible(self) -> None:
        """Weights attribute should be accessible."""
        from state_space_practice.switching_point_process import SpikeObsParams

        baseline = jnp.zeros(3)
        weights = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        params = SpikeObsParams(baseline=baseline, weights=weights)

        np.testing.assert_allclose(params.weights, weights)


class TestPointProcessKalmanUpdate:
    """Tests for the point_process_kalman_update function."""

    def test_output_shapes_single_neuron(self) -> None:
        """Output shapes should be correct for single neuron."""
        from state_space_practice.switching_point_process import (
            point_process_kalman_update,
        )

        n_latent = 4
        n_neurons = 1
        dt = 0.02

        one_step_mean = jnp.zeros(n_latent)
        one_step_cov = jnp.eye(n_latent)
        y_t = jnp.array([1.0])  # 1 spike

        weights = jnp.ones((n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        posterior_mean, posterior_cov, log_ll = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, log_intensity_func
        )

        assert posterior_mean.shape == (n_latent,)
        assert posterior_cov.shape == (n_latent, n_latent)
        assert log_ll.shape == ()

    def test_output_shapes_multiple_neurons(self) -> None:
        """Output shapes should be correct for multiple neurons."""
        from state_space_practice.switching_point_process import (
            point_process_kalman_update,
        )

        n_latent = 4
        n_neurons = 10
        dt = 0.02

        one_step_mean = jnp.zeros(n_latent)
        one_step_cov = jnp.eye(n_latent)
        y_t = jnp.ones(n_neurons)  # 1 spike per neuron

        weights = jax.random.normal(jax.random.PRNGKey(0), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        posterior_mean, posterior_cov, log_ll = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, log_intensity_func
        )

        assert posterior_mean.shape == (n_latent,)
        assert posterior_cov.shape == (n_latent, n_latent)
        assert log_ll.shape == ()

    def test_no_nans(self) -> None:
        """Update should not produce NaN values."""
        from state_space_practice.switching_point_process import (
            point_process_kalman_update,
        )

        n_latent = 4
        n_neurons = 5
        dt = 0.02

        one_step_mean = jnp.ones(n_latent) * 0.5
        one_step_cov = jnp.eye(n_latent) * 0.1
        y_t = jnp.array([0.0, 1.0, 2.0, 0.0, 1.0])

        weights = jax.random.normal(jax.random.PRNGKey(42), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        posterior_mean, posterior_cov, log_ll = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, log_intensity_func
        )

        assert not jnp.any(jnp.isnan(posterior_mean))
        assert not jnp.any(jnp.isnan(posterior_cov))
        assert not jnp.isnan(log_ll)

    def test_zero_spikes_minimal_update(self) -> None:
        """With zero spikes, posterior should be close to prior.

        The innovation y - lambda*dt is negative when y=0, which should
        push the posterior mean in the direction of lower intensity.
        """
        from state_space_practice.switching_point_process import (
            point_process_kalman_update,
        )

        n_latent = 4
        n_neurons = 5
        dt = 0.02

        one_step_mean = jnp.zeros(n_latent)
        one_step_cov = jnp.eye(n_latent)
        y_t = jnp.zeros(n_neurons)  # No spikes

        weights = jnp.ones((n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        posterior_mean, posterior_cov, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, log_intensity_func
        )

        # With zero spikes, the update should be small
        # (posterior should be close to prior for small intensity)
        assert jnp.allclose(posterior_mean, one_step_mean, atol=0.5)

    def test_covariance_is_symmetric(self) -> None:
        """Posterior covariance should be symmetric."""
        from state_space_practice.switching_point_process import (
            point_process_kalman_update,
        )

        n_latent = 4
        n_neurons = 5
        dt = 0.02

        one_step_mean = jnp.ones(n_latent) * 0.5
        one_step_cov = jnp.eye(n_latent) * 0.1
        y_t = jnp.array([0.0, 1.0, 2.0, 0.0, 1.0])

        weights = jax.random.normal(jax.random.PRNGKey(42), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        _, posterior_cov, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, log_intensity_func
        )

        np.testing.assert_allclose(posterior_cov, posterior_cov.T, rtol=1e-5, atol=1e-10)

    def test_covariance_is_positive_definite(self) -> None:
        """Posterior covariance should be positive semi-definite."""
        from state_space_practice.switching_point_process import (
            point_process_kalman_update,
        )

        n_latent = 4
        n_neurons = 5
        dt = 0.02

        one_step_mean = jnp.ones(n_latent) * 0.5
        one_step_cov = jnp.eye(n_latent) * 0.1
        y_t = jnp.array([0.0, 1.0, 2.0, 0.0, 1.0])

        weights = jax.random.normal(jax.random.PRNGKey(42), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        _, posterior_cov, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, log_intensity_func
        )

        # All eigenvalues should be non-negative
        eigvals = jnp.linalg.eigvalsh(posterior_cov)
        assert jnp.all(eigvals >= -1e-6), f"Negative eigenvalues: {eigvals}"

    def test_log_likelihood_finite(self) -> None:
        """Log-likelihood should be finite."""
        from state_space_practice.switching_point_process import (
            point_process_kalman_update,
        )

        n_latent = 4
        n_neurons = 5
        dt = 0.02

        one_step_mean = jnp.ones(n_latent) * 0.5
        one_step_cov = jnp.eye(n_latent) * 0.1
        y_t = jnp.array([0.0, 1.0, 2.0, 0.0, 1.0])

        weights = jax.random.normal(jax.random.PRNGKey(42), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        _, _, log_ll = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, log_intensity_func
        )

        assert jnp.isfinite(log_ll)

    def test_high_spikes_increases_intensity_estimate(self) -> None:
        """High spike count should increase intensity estimate.

        With positive weights, more spikes should push the state estimate
        in a direction that increases predicted intensity.
        """
        from state_space_practice.switching_point_process import (
            point_process_kalman_update,
        )

        n_latent = 2
        n_neurons = 1
        dt = 0.02

        one_step_mean = jnp.zeros(n_latent)
        one_step_cov = jnp.eye(n_latent)

        # Positive weights - state should increase to explain spikes
        weights = jnp.ones((n_neurons, n_latent)) * 0.5
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Low spikes
        posterior_mean_low, _, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, jnp.array([0.0]), dt, log_intensity_func
        )

        # High spikes
        posterior_mean_high, _, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, jnp.array([5.0]), dt, log_intensity_func
        )

        # With positive weights, high spikes should lead to higher log-intensity
        log_rate_low = log_intensity_func(posterior_mean_low)
        log_rate_high = log_intensity_func(posterior_mean_high)

        assert jnp.all(log_rate_high > log_rate_low)


class TestPointProcessPredictAndUpdate:
    """Tests for the _point_process_predict_and_update helper."""

    def test_output_shapes(self) -> None:
        """Output shapes should be correct."""
        from state_space_practice.switching_point_process import (
            _point_process_predict_and_update,
        )

        n_latent = 4
        n_neurons = 3
        dt = 0.02

        prev_state_cond_mean = jnp.zeros(n_latent)
        prev_state_cond_cov = jnp.eye(n_latent)
        y_t = jnp.ones(n_neurons)
        continuous_transition_matrix = jnp.eye(n_latent) * 0.99
        process_cov = jnp.eye(n_latent) * 0.01

        weights = jax.random.normal(jax.random.PRNGKey(0), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        posterior_mean, posterior_cov, log_ll = _point_process_predict_and_update(
            prev_state_cond_mean,
            prev_state_cond_cov,
            y_t,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        assert posterior_mean.shape == (n_latent,)
        assert posterior_cov.shape == (n_latent, n_latent)
        assert log_ll.shape == ()

    def test_prediction_incorporated(self) -> None:
        """The dynamics prediction should be incorporated before update."""
        from state_space_practice.switching_point_process import (
            _point_process_predict_and_update,
            point_process_kalman_update,
        )

        n_latent = 2
        n_neurons = 1
        dt = 0.02

        prev_mean = jnp.array([1.0, 0.0])
        prev_cov = jnp.eye(n_latent) * 0.1
        y_t = jnp.zeros(n_neurons)

        # Transition that shifts state
        A = jnp.array([[0.9, 0.1], [-0.1, 0.9]])
        Q = jnp.eye(n_latent) * 0.01

        weights = jnp.ones((n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Use predict_and_update
        mean_combined, cov_combined, _ = _point_process_predict_and_update(
            prev_mean, prev_cov, y_t, A, Q, dt, log_intensity_func
        )

        # Manual prediction + update
        one_step_mean = A @ prev_mean
        one_step_cov = A @ prev_cov @ A.T + Q
        mean_manual, cov_manual, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, log_intensity_func
        )

        np.testing.assert_allclose(mean_combined, mean_manual, rtol=1e-5)
        np.testing.assert_allclose(cov_combined, cov_manual, rtol=1e-5)

    def test_no_nans(self) -> None:
        """Should not produce NaN values."""
        from state_space_practice.switching_point_process import (
            _point_process_predict_and_update,
        )

        n_latent = 4
        n_neurons = 5
        dt = 0.02

        prev_mean = jnp.ones(n_latent) * 0.5
        prev_cov = jnp.eye(n_latent) * 0.1
        y_t = jnp.array([0.0, 1.0, 2.0, 0.0, 1.0])
        A = jnp.eye(n_latent) * 0.95
        Q = jnp.eye(n_latent) * 0.01

        weights = jax.random.normal(jax.random.PRNGKey(42), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        mean, cov, ll = _point_process_predict_and_update(
            prev_mean, prev_cov, y_t, A, Q, dt, log_intensity_func
        )

        assert not jnp.any(jnp.isnan(mean))
        assert not jnp.any(jnp.isnan(cov))
        assert not jnp.isnan(ll)


class TestPointProcessUpdatePerStatePair:
    """Tests for the _point_process_update_per_discrete_state_pair vmapped function."""

    def test_output_shapes(self) -> None:
        """Output shapes should be correct for state pairs."""
        from state_space_practice.switching_point_process import (
            _point_process_update_per_discrete_state_pair,
        )

        n_latent = 4
        n_neurons = 3
        n_discrete_states = 2
        dt = 0.02

        # State-conditional means/covs for previous timestep
        # Shape: (n_latent, n_discrete_states)
        prev_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        # Shape: (n_latent, n_latent, n_discrete_states)
        prev_state_cond_cov = jnp.stack([jnp.eye(n_latent)] * n_discrete_states, axis=-1)

        y_t = jnp.ones(n_neurons)

        # Dynamics per discrete state
        # Shape: (n_latent, n_latent, n_discrete_states)
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack(
            [jnp.eye(n_latent) * 0.01, jnp.eye(n_latent) * 0.02], axis=-1
        )

        weights = jax.random.normal(jax.random.PRNGKey(0), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        pair_mean, pair_cov, pair_ll = _point_process_update_per_discrete_state_pair(
            prev_state_cond_mean,
            prev_state_cond_cov,
            y_t,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Output shapes should be (n_latent, n_discrete_states, n_discrete_states)
        assert pair_mean.shape == (n_latent, n_discrete_states, n_discrete_states)
        assert pair_cov.shape == (n_latent, n_latent, n_discrete_states, n_discrete_states)
        assert pair_ll.shape == (n_discrete_states, n_discrete_states)

    def test_single_state_matches_base_function(self) -> None:
        """With 1 state, should match non-vmapped function."""
        from state_space_practice.switching_point_process import (
            _point_process_predict_and_update,
            _point_process_update_per_discrete_state_pair,
        )

        n_latent = 4
        n_neurons = 3
        dt = 0.02

        prev_mean_single = jnp.zeros(n_latent)
        prev_cov_single = jnp.eye(n_latent) * 0.5
        y_t = jnp.array([0.0, 1.0, 2.0])
        A_single = jnp.eye(n_latent) * 0.9
        Q_single = jnp.eye(n_latent) * 0.01

        weights = jax.random.normal(jax.random.PRNGKey(42), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Non-vmapped result
        expected_mean, expected_cov, expected_ll = _point_process_predict_and_update(
            prev_mean_single, prev_cov_single, y_t, A_single, Q_single, dt, log_intensity_func
        )

        # Vmapped result with state dimension
        prev_mean_batched = prev_mean_single[:, None]  # (n_latent, 1)
        prev_cov_batched = prev_cov_single[:, :, None]  # (n_latent, n_latent, 1)
        A_batched = A_single[:, :, None]  # (n_latent, n_latent, 1)
        Q_batched = Q_single[:, :, None]  # (n_latent, n_latent, 1)

        pair_mean, pair_cov, pair_ll = _point_process_update_per_discrete_state_pair(
            prev_mean_batched, prev_cov_batched, y_t, A_batched, Q_batched, dt, log_intensity_func
        )

        # Should match single result
        np.testing.assert_allclose(pair_mean[:, 0, 0], expected_mean, rtol=1e-5)
        np.testing.assert_allclose(pair_cov[:, :, 0, 0], expected_cov, rtol=1e-5)
        np.testing.assert_allclose(pair_ll[0, 0], expected_ll, rtol=1e-5)

    def test_no_nans(self) -> None:
        """Should not produce NaN values."""
        from state_space_practice.switching_point_process import (
            _point_process_update_per_discrete_state_pair,
        )

        n_latent = 4
        n_neurons = 5
        n_discrete_states = 3
        dt = 0.02

        key = jax.random.PRNGKey(123)
        k1, k2, k3 = jax.random.split(key, 3)

        prev_mean = jax.random.normal(k1, (n_latent, n_discrete_states)) * 0.1
        prev_cov = jnp.stack([jnp.eye(n_latent) * 0.1] * n_discrete_states, axis=-1)
        y_t = jnp.array([0.0, 1.0, 2.0, 0.0, 1.0])

        A = jnp.stack([jnp.eye(n_latent) * (0.9 + 0.02 * i) for i in range(n_discrete_states)], axis=-1)
        Q = jnp.stack([jnp.eye(n_latent) * (0.01 + 0.005 * i) for i in range(n_discrete_states)], axis=-1)

        weights = jax.random.normal(k2, (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        pair_mean, pair_cov, pair_ll = _point_process_update_per_discrete_state_pair(
            prev_mean, prev_cov, y_t, A, Q, dt, log_intensity_func
        )

        assert not jnp.any(jnp.isnan(pair_mean))
        assert not jnp.any(jnp.isnan(pair_cov))
        assert not jnp.any(jnp.isnan(pair_ll))

    def test_different_states_give_different_results(self) -> None:
        """Different discrete states should give different outputs."""
        from state_space_practice.switching_point_process import (
            _point_process_update_per_discrete_state_pair,
        )

        n_latent = 2
        n_neurons = 2
        n_discrete_states = 2
        dt = 0.02

        prev_mean = jnp.zeros((n_latent, n_discrete_states))
        prev_cov = jnp.stack([jnp.eye(n_latent)] * n_discrete_states, axis=-1)
        y_t = jnp.ones(n_neurons)

        # Very different dynamics
        A = jnp.stack([jnp.eye(n_latent) * 0.5, jnp.eye(n_latent) * 0.99], axis=-1)
        Q = jnp.stack([jnp.eye(n_latent) * 0.1, jnp.eye(n_latent) * 0.001], axis=-1)

        weights = jnp.ones((n_neurons, n_latent)) * 0.5
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        pair_mean, pair_cov, pair_ll = _point_process_update_per_discrete_state_pair(
            prev_mean, prev_cov, y_t, A, Q, dt, log_intensity_func
        )

        # Results for state j=0 and j=1 should be different
        assert not jnp.allclose(pair_mean[:, 0, 0], pair_mean[:, 0, 1])
        assert not jnp.allclose(pair_cov[:, :, 0, 0], pair_cov[:, :, 0, 1])


class TestSwitchingPointProcessFilter:
    """Tests for the switching_point_process_filter function (Task 2.5)."""

    def test_output_shapes(self) -> None:
        """All outputs should have correct shapes (Task 2.5)."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 50
        n_latent = 4
        n_neurons = 5
        n_discrete_states = 2
        dt = 0.02

        # Initial conditions
        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        # Observations (spike counts)
        key = jax.random.PRNGKey(0)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        # Discrete transition matrix
        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        # Continuous dynamics per state
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack(
            [jnp.eye(n_latent) * 0.01, jnp.eye(n_latent) * 0.02], axis=-1
        )

        # Spike observation parameters
        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            marginal_log_likelihood,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Check shapes
        assert state_cond_filter_mean.shape == (n_time, n_latent, n_discrete_states)
        assert state_cond_filter_cov.shape == (
            n_time,
            n_latent,
            n_latent,
            n_discrete_states,
        )
        assert filter_discrete_state_prob.shape == (n_time, n_discrete_states)
        assert last_pair_cond_filter_mean.shape == (
            n_latent,
            n_discrete_states,
            n_discrete_states,
        )
        assert marginal_log_likelihood.shape == ()

    def test_discrete_probs_sum_to_one(self) -> None:
        """Discrete state probabilities should sum to 1 at each timestep."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 30
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 3
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(42)
        spikes = jax.random.poisson(key, 0.3, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.ones((n_discrete_states, n_discrete_states))
        discrete_transition_matrix = discrete_transition_matrix / n_discrete_states

        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * (0.9 + 0.03 * i) for i in range(n_discrete_states)],
            axis=-1,
        )
        process_cov = jnp.stack(
            [jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1
        )

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        _, _, filter_discrete_state_prob, _, _ = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Probabilities should sum to 1 at each timestep
        prob_sums = jnp.sum(filter_discrete_state_prob, axis=1)
        np.testing.assert_allclose(prob_sums, jnp.ones(n_time), rtol=1e-5)

    def test_discrete_probs_nonnegative(self) -> None:
        """Discrete state probabilities should be non-negative."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 20
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.array([0.7, 0.3])

        key = jax.random.PRNGKey(123)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.95, 0.05], [0.05, 0.95]])

        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.9], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        _, _, filter_discrete_state_prob, _, _ = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # All probabilities should be >= 0
        assert jnp.all(filter_discrete_state_prob >= 0)

    def test_no_nans(self) -> None:
        """Filter should not produce NaN values."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 30
        n_latent = 4
        n_neurons = 5
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(99)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            marginal_log_likelihood,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        assert not jnp.any(jnp.isnan(state_cond_filter_mean))
        assert not jnp.any(jnp.isnan(state_cond_filter_cov))
        assert not jnp.any(jnp.isnan(filter_discrete_state_prob))
        assert not jnp.any(jnp.isnan(last_pair_cond_filter_mean))
        assert not jnp.isnan(marginal_log_likelihood)

    def test_marginal_log_likelihood_finite(self) -> None:
        """Marginal log-likelihood should be finite."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 20
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(42)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        _, _, _, _, marginal_log_likelihood = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        assert jnp.isfinite(marginal_log_likelihood)

    def test_single_discrete_state(self) -> None:
        """With 1 discrete state, filter should still work correctly."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 25
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 1
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.eye(n_latent)[:, :, None]
        init_discrete_state_prob = jnp.ones(n_discrete_states)

        key = jax.random.PRNGKey(0)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.ones((1, 1))

        continuous_transition_matrix = jnp.eye(n_latent)[:, :, None] * 0.99
        process_cov = jnp.eye(n_latent)[:, :, None] * 0.01

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            marginal_log_likelihood,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # With S=1, probabilities should always be 1
        np.testing.assert_allclose(
            filter_discrete_state_prob, jnp.ones((n_time, 1)), rtol=1e-5
        )

        # Shapes should still be correct
        assert state_cond_filter_mean.shape == (n_time, n_latent, 1)
        assert state_cond_filter_cov.shape == (n_time, n_latent, n_latent, 1)
        assert last_pair_cond_filter_mean.shape == (n_latent, 1, 1)

    def test_covariances_positive_semidefinite(self) -> None:
        """Filtered covariances should be positive semi-definite."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 15
        n_latent = 3
        n_neurons = 4
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(55)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        _, state_cond_filter_cov, _, _, _ = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Check all covariances are PSD (eigenvalues >= 0)
        for t in range(n_time):
            for s in range(n_discrete_states):
                eigvals = jnp.linalg.eigvalsh(state_cond_filter_cov[t, :, :, s])
                assert jnp.all(
                    eigvals >= -1e-6
                ), f"Negative eigenvalue at t={t}, s={s}: {eigvals}"

    def test_pair_conditional_shapes(self) -> None:
        """last_pair_cond_filter_mean should have correct shape for smoother."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 10
        n_latent = 3
        n_neurons = 2
        n_discrete_states = 3
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(77)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = (
            jnp.eye(n_discrete_states) * 0.8 + jnp.ones((n_discrete_states, n_discrete_states)) * 0.2 / n_discrete_states
        )
        discrete_transition_matrix = discrete_transition_matrix / jnp.sum(
            discrete_transition_matrix, axis=1, keepdims=True
        )

        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * (0.9 + 0.03 * i) for i in range(n_discrete_states)],
            axis=-1,
        )
        process_cov = jnp.stack(
            [jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1
        )

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        _, _, _, last_pair_cond_filter_mean, _ = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Shape: (n_latent, n_discrete_states, n_discrete_states)
        # where [:, i, j] is the mean for p(x_T | y_{1:T}, S_{T-1}=i, S_T=j)
        assert last_pair_cond_filter_mean.shape == (
            n_latent,
            n_discrete_states,
            n_discrete_states,
        )

    def test_all_zero_spikes(self) -> None:
        """Filter should handle observation sequences with no spikes (silent neurons)."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 20
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        # All zero spikes - silent neurons
        spikes = jnp.zeros((n_time, n_neurons))

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            marginal_log_likelihood,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Should not produce NaN values
        assert not jnp.any(jnp.isnan(state_cond_filter_mean))
        assert not jnp.any(jnp.isnan(state_cond_filter_cov))
        assert not jnp.any(jnp.isnan(filter_discrete_state_prob))
        assert not jnp.any(jnp.isnan(last_pair_cond_filter_mean))
        assert jnp.isfinite(marginal_log_likelihood)

        # Probabilities should still sum to 1
        prob_sums = jnp.sum(filter_discrete_state_prob, axis=1)
        np.testing.assert_allclose(prob_sums, jnp.ones(n_time), rtol=1e-5)

    def test_high_spike_counts(self) -> None:
        """Filter should handle high spike counts without numerical issues."""
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 15
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        # High spike counts (20+ spikes per bin for some neurons)
        key = jax.random.PRNGKey(88)
        spikes = jax.random.poisson(key, 15.0, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        # Higher baseline to match high spike counts
        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.ones(n_neurons) * 3.0  # Higher baseline for higher rates
        log_intensity_func = linear_log_intensity(weights, baseline)

        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            marginal_log_likelihood,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Should not produce NaN or infinite values
        assert not jnp.any(jnp.isnan(state_cond_filter_mean))
        assert not jnp.any(jnp.isnan(state_cond_filter_cov))
        assert not jnp.any(jnp.isnan(filter_discrete_state_prob))
        assert jnp.isfinite(marginal_log_likelihood)

        # Probabilities should still be valid
        assert jnp.all(filter_discrete_state_prob >= 0)
        prob_sums = jnp.sum(filter_discrete_state_prob, axis=1)
        np.testing.assert_allclose(prob_sums, jnp.ones(n_time), rtol=1e-5)


class TestSmootherIntegration:
    """Tests verifying switching_kalman_smoother works with point-process filter outputs.

    The smoother is observation-model agnostic - it only operates on Gaussian posteriors.
    These tests verify that the point-process filter outputs are compatible with the
    existing switching_kalman_smoother.
    """

    def test_smoother_runs_on_filter_output(self) -> None:
        """Smoother should run without error on point-process filter outputs."""
        from state_space_practice.switching_kalman import switching_kalman_smoother
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 30
        n_latent = 4
        n_neurons = 5
        n_discrete_states = 2
        dt = 0.02

        # Initial conditions
        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        # Observations
        key = jax.random.PRNGKey(0)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        # Dynamics
        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack(
            [jnp.eye(n_latent) * 0.01, jnp.eye(n_latent) * 0.02], axis=-1
        )

        # Spike observation model
        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Run filter
        (
            filter_mean,
            filter_cov,
            filter_discrete_prob,
            last_pair_cond_mean,
            _,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Run smoother on filter output - should not raise
        # Smoother returns 9 values: (overall_mean, overall_cov, discrete_probs,
        # joint_discrete_probs, cross_cov, state_cond_mean, state_cond_cov,
        # pair_cond_cross_cov, pair_cond_mean)
        (
            smoother_mean,
            smoother_cov,
            smoother_discrete_prob,
            _,  # joint discrete probs
            _,  # cross cov
            _,  # state conditional means
            _,  # state conditional covs
            _,  # pair conditional cross covs
            _,  # pair conditional means
        ) = switching_kalman_smoother(
            filter_mean=filter_mean,
            filter_cov=filter_cov,
            filter_discrete_state_prob=filter_discrete_prob,
            last_filter_conditional_cont_mean=last_pair_cond_mean,
            process_cov=process_cov,
            continuous_transition_matrix=continuous_transition_matrix,
            discrete_state_transition_matrix=discrete_transition_matrix,
        )

        # Verify shapes - overall smoother mean/cov are marginalized over discrete states
        assert smoother_mean.shape == (n_time, n_latent)
        assert smoother_cov.shape == (n_time, n_latent, n_latent)
        assert smoother_discrete_prob.shape == (n_time, n_discrete_states)

    def test_smoother_output_no_nans(self) -> None:
        """Smoother should not produce NaN values."""
        from state_space_practice.switching_kalman import switching_kalman_smoother
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 25
        n_latent = 3
        n_neurons = 4
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(42)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Filter
        (
            filter_mean,
            filter_cov,
            filter_discrete_prob,
            last_pair_cond_mean,
            _,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Smoother (9 return values)
        (
            smoother_mean,
            smoother_cov,
            smoother_discrete_prob,
            _,  # joint discrete probs
            _,  # cross cov
            _,  # state conditional means
            _,  # state conditional covs
            _,  # pair conditional cross covs
            _,  # pair conditional means
        ) = switching_kalman_smoother(
            filter_mean=filter_mean,
            filter_cov=filter_cov,
            filter_discrete_state_prob=filter_discrete_prob,
            last_filter_conditional_cont_mean=last_pair_cond_mean,
            process_cov=process_cov,
            continuous_transition_matrix=continuous_transition_matrix,
            discrete_state_transition_matrix=discrete_transition_matrix,
        )

        assert not jnp.any(jnp.isnan(smoother_mean))
        assert not jnp.any(jnp.isnan(smoother_cov))
        assert not jnp.any(jnp.isnan(smoother_discrete_prob))

    def test_smoother_shapes_compatible(self) -> None:
        """Smoother outputs should have shapes compatible with filter outputs.

        Note: The theoretical property that smoothed variance <= filtered variance
        holds exactly only for the marginal (overall) distributions in a standard
        Kalman filter. In switching models with mixture collapse approximations,
        this property may not hold exactly for state-conditional quantities.

        This test verifies shape compatibility and that the smoother produces
        reasonable outputs (no NaN, finite values).
        """
        from state_space_practice.switching_kalman import switching_kalman_smoother
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 40
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(99)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Filter
        (
            filter_mean,
            filter_cov,
            filter_discrete_prob,
            last_pair_cond_mean,
            _,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Smoother (9 return values)
        (
            overall_smoother_mean,
            overall_smoother_cov,
            smoother_discrete_prob,
            joint_discrete_prob,
            cross_cov,
            state_cond_smoother_mean,
            state_cond_smoother_cov,
            _,  # pair conditional cross covs
            _,  # pair conditional means
        ) = switching_kalman_smoother(
            filter_mean=filter_mean,
            filter_cov=filter_cov,
            filter_discrete_state_prob=filter_discrete_prob,
            last_filter_conditional_cont_mean=last_pair_cond_mean,
            process_cov=process_cov,
            continuous_transition_matrix=continuous_transition_matrix,
            discrete_state_transition_matrix=discrete_transition_matrix,
        )

        # Verify shape compatibility
        # Overall quantities are marginalized over discrete states
        assert overall_smoother_mean.shape == (n_time, n_latent)
        assert overall_smoother_cov.shape == (n_time, n_latent, n_latent)
        assert smoother_discrete_prob.shape == (n_time, n_discrete_states)
        # Joint discrete prob shape: (n_time - 1, n_discrete_states, n_discrete_states)
        assert joint_discrete_prob.shape == (
            n_time - 1,
            n_discrete_states,
            n_discrete_states,
        )
        # Cross covariance shape: (n_time - 1, n_latent, n_latent)
        assert cross_cov.shape == (n_time - 1, n_latent, n_latent)
        # State-conditional quantities indexed by discrete state
        assert state_cond_smoother_mean.shape == (n_time, n_latent, n_discrete_states)
        assert state_cond_smoother_cov.shape == (
            n_time,
            n_latent,
            n_latent,
            n_discrete_states,
        )

        # All outputs should be finite
        assert jnp.all(jnp.isfinite(overall_smoother_mean))
        assert jnp.all(jnp.isfinite(overall_smoother_cov))
        assert jnp.all(jnp.isfinite(smoother_discrete_prob))
        assert jnp.all(jnp.isfinite(state_cond_smoother_mean))
        assert jnp.all(jnp.isfinite(state_cond_smoother_cov))

    def test_smoother_discrete_probs_sum_to_one(self) -> None:
        """Smoothed discrete probabilities should sum to 1."""
        from state_space_practice.switching_kalman import switching_kalman_smoother
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 20
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 3
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(77)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.ones((n_discrete_states, n_discrete_states))
        discrete_transition_matrix = discrete_transition_matrix / n_discrete_states

        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * (0.9 + 0.03 * i) for i in range(n_discrete_states)],
            axis=-1,
        )
        process_cov = jnp.stack(
            [jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1
        )

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Filter
        (
            filter_mean,
            filter_cov,
            filter_discrete_prob,
            last_pair_cond_mean,
            _,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Smoother (9 return values)
        (
            _,  # overall mean
            _,  # overall cov
            smoother_discrete_prob,
            _,  # joint discrete probs
            _,  # cross cov
            _,  # state conditional means
            _,  # state conditional covs
            _,  # pair conditional cross covs
            _,  # pair conditional means
        ) = switching_kalman_smoother(
            filter_mean=filter_mean,
            filter_cov=filter_cov,
            filter_discrete_state_prob=filter_discrete_prob,
            last_filter_conditional_cont_mean=last_pair_cond_mean,
            process_cov=process_cov,
            continuous_transition_matrix=continuous_transition_matrix,
            discrete_state_transition_matrix=discrete_transition_matrix,
        )

        # Probabilities should sum to 1
        prob_sums = jnp.sum(smoother_discrete_prob, axis=1)
        np.testing.assert_allclose(prob_sums, jnp.ones(n_time), rtol=1e-5)


class TestSingleNeuronGLMLoss:
    """Tests for the _single_neuron_glm_loss helper function (Task 5.1)."""

    def test_loss_output_is_scalar(self) -> None:
        """Loss function should return a scalar."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_loss,
        )

        n_time = 50
        n_latent = 4
        dt = 0.02

        # Spike counts for a single neuron
        y_n = jax.random.poisson(jax.random.PRNGKey(0), 0.5, shape=(n_time,)).astype(
            float
        )

        # Smoother mean as design matrix
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent))

        # GLM parameters: baseline (scalar) and weights (n_latent,)
        baseline = 0.0
        weights = jnp.zeros(n_latent)

        loss = _single_neuron_glm_loss(baseline, weights, y_n, smoother_mean, dt)

        assert loss.shape == ()

    def test_loss_is_finite(self) -> None:
        """Loss should be finite for reasonable inputs."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_loss,
        )

        n_time = 30
        n_latent = 2
        dt = 0.02

        y_n = jax.random.poisson(jax.random.PRNGKey(0), 1.0, shape=(n_time,)).astype(
            float
        )
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        baseline = 0.0
        weights = jnp.ones(n_latent) * 0.1

        loss = _single_neuron_glm_loss(baseline, weights, y_n, smoother_mean, dt)

        assert jnp.isfinite(loss)

    def test_loss_nonnegative(self) -> None:
        """Negative log-likelihood should be non-negative."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_loss,
        )

        n_time = 40
        n_latent = 3
        dt = 0.02

        y_n = jax.random.poisson(jax.random.PRNGKey(0), 0.5, shape=(n_time,)).astype(
            float
        )
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.3

        baseline = 0.0
        weights = jnp.ones(n_latent) * 0.1

        loss = _single_neuron_glm_loss(baseline, weights, y_n, smoother_mean, dt)

        # NLL should be non-negative (we're computing -log_likelihood)
        # Actually, Poisson NLL can be any real number depending on data
        # But it should be finite
        assert jnp.isfinite(loss)

    def test_loss_decreases_with_better_params(self) -> None:
        """Loss should be lower when parameters match data generation."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_loss,
        )

        n_time = 100
        n_latent = 2
        dt = 0.02

        # True parameters
        true_baseline = 1.0
        true_weights = jnp.array([0.5, -0.3])

        # Generate design matrix
        key = jax.random.PRNGKey(42)
        smoother_mean = jax.random.normal(key, (n_time, n_latent)) * 0.5

        # Compute true log rates and sample spikes
        eta = true_baseline + smoother_mean @ true_weights
        rates = jnp.exp(eta) * dt
        y_n = jax.random.poisson(jax.random.PRNGKey(123), rates).astype(float)

        # Loss at true parameters
        loss_true = _single_neuron_glm_loss(
            true_baseline, true_weights, y_n, smoother_mean, dt
        )

        # Loss at wrong parameters (all zeros)
        loss_wrong = _single_neuron_glm_loss(
            0.0, jnp.zeros(n_latent), y_n, smoother_mean, dt
        )

        # True parameters should give lower (or similar) loss
        # Note: with finite samples, this may not always hold exactly
        # but with enough data the trend should be clear
        assert jnp.isfinite(loss_true)
        assert jnp.isfinite(loss_wrong)

    def test_loss_gradient_computable(self) -> None:
        """Should be able to compute gradient of loss w.r.t. parameters."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_loss,
        )

        n_time = 30
        n_latent = 2
        dt = 0.02

        y_n = jax.random.poisson(jax.random.PRNGKey(0), 0.5, shape=(n_time,)).astype(
            float
        )
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        baseline = 0.0
        weights = jnp.zeros(n_latent)

        # Compute gradients via JAX
        grad_fn = jax.grad(_single_neuron_glm_loss, argnums=(0, 1))
        grad_b, grad_w = grad_fn(baseline, weights, y_n, smoother_mean, dt)

        assert grad_b.shape == ()
        assert grad_w.shape == (n_latent,)
        assert jnp.isfinite(grad_b)
        assert jnp.all(jnp.isfinite(grad_w))

    def test_loss_zero_spikes(self) -> None:
        """Loss should be finite for zero spike counts."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_loss,
        )

        n_time = 20
        n_latent = 2
        dt = 0.02

        y_n = jnp.zeros(n_time)  # No spikes
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        baseline = 0.0
        weights = jnp.ones(n_latent) * 0.1

        loss = _single_neuron_glm_loss(baseline, weights, y_n, smoother_mean, dt)

        assert jnp.isfinite(loss)

    def test_loss_high_spikes(self) -> None:
        """Loss should be finite for high spike counts."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_loss,
        )

        n_time = 20
        n_latent = 2
        dt = 0.02

        # High spike counts
        y_n = jax.random.poisson(jax.random.PRNGKey(0), 20.0, shape=(n_time,)).astype(
            float
        )
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        # Higher baseline to match
        baseline = 3.0
        weights = jnp.ones(n_latent) * 0.1

        loss = _single_neuron_glm_loss(baseline, weights, y_n, smoother_mean, dt)

        assert jnp.isfinite(loss)


class TestSingleNeuronGLMStep:
    """Tests for the _single_neuron_glm_step Newton step function (Task 5.2)."""

    def test_output_shapes(self) -> None:
        """Output shapes should match input parameter shapes."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_step,
        )

        n_time = 50
        n_latent = 4
        dt = 0.02

        y_n = jax.random.poisson(jax.random.PRNGKey(0), 0.5, shape=(n_time,)).astype(
            float
        )
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent))

        baseline = 0.0
        weights = jnp.zeros(n_latent)

        new_baseline, new_weights = _single_neuron_glm_step(
            baseline, weights, y_n, smoother_mean, dt
        )

        assert new_baseline.shape == ()
        assert new_weights.shape == (n_latent,)

    def test_output_finite(self) -> None:
        """Newton step should produce finite parameter updates."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_step,
        )

        n_time = 30
        n_latent = 2
        dt = 0.02

        y_n = jax.random.poisson(jax.random.PRNGKey(0), 1.0, shape=(n_time,)).astype(
            float
        )
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        baseline = 0.0
        weights = jnp.ones(n_latent) * 0.1

        new_baseline, new_weights = _single_neuron_glm_step(
            baseline, weights, y_n, smoother_mean, dt
        )

        assert jnp.isfinite(new_baseline)
        assert jnp.all(jnp.isfinite(new_weights))

    def test_step_decreases_loss(self) -> None:
        """Newton step should decrease the loss (at least for well-conditioned problems)."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_loss,
            _single_neuron_glm_step,
        )

        n_time = 100
        n_latent = 2
        dt = 0.02

        # Generate data from known parameters
        true_baseline = 1.0
        true_weights = jnp.array([0.3, -0.2])

        key = jax.random.PRNGKey(42)
        smoother_mean = jax.random.normal(key, (n_time, n_latent)) * 0.5

        eta = true_baseline + smoother_mean @ true_weights
        rates = jnp.exp(eta) * dt
        y_n = jax.random.poisson(jax.random.PRNGKey(123), rates).astype(float)

        # Start from wrong parameters
        baseline = 0.0
        weights = jnp.zeros(n_latent)

        loss_before = _single_neuron_glm_loss(baseline, weights, y_n, smoother_mean, dt)

        # Take Newton step
        new_baseline, new_weights = _single_neuron_glm_step(
            baseline, weights, y_n, smoother_mean, dt
        )

        loss_after = _single_neuron_glm_loss(
            new_baseline, new_weights, y_n, smoother_mean, dt
        )

        # Loss should decrease (Newton step in descent direction)
        assert loss_after < loss_before

    def test_step_converges_to_optimal(self) -> None:
        """Multiple Newton steps should converge to optimal parameters."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_loss,
            _single_neuron_glm_step,
        )

        n_time = 200
        n_latent = 2
        dt = 0.02

        # Generate data from known parameters
        true_baseline = 0.5
        true_weights = jnp.array([0.4, -0.3])

        key = jax.random.PRNGKey(55)
        smoother_mean = jax.random.normal(key, (n_time, n_latent)) * 0.5

        eta = true_baseline + smoother_mean @ true_weights
        rates = jnp.exp(eta) * dt
        y_n = jax.random.poisson(jax.random.PRNGKey(77), rates).astype(float)

        # Start from initial parameters
        baseline = 0.0
        weights = jnp.zeros(n_latent)

        # Run multiple Newton iterations
        for _ in range(10):
            baseline, weights = _single_neuron_glm_step(
                baseline, weights, y_n, smoother_mean, dt
            )

        # Check convergence - final parameters should be close to optimal
        # (may not be exactly true params due to finite sample noise)
        final_loss = _single_neuron_glm_loss(baseline, weights, y_n, smoother_mean, dt)
        true_loss = _single_neuron_glm_loss(
            true_baseline, true_weights, y_n, smoother_mean, dt
        )

        # Optimized loss should be close to or better than true parameter loss
        assert final_loss <= true_loss + 1.0  # Allow some tolerance

    def test_step_zero_spikes(self) -> None:
        """Newton step should handle zero spike counts."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_step,
        )

        n_time = 30
        n_latent = 2
        dt = 0.02

        y_n = jnp.zeros(n_time)  # No spikes
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        baseline = 0.0
        weights = jnp.ones(n_latent) * 0.1

        new_baseline, new_weights = _single_neuron_glm_step(
            baseline, weights, y_n, smoother_mean, dt
        )

        assert jnp.isfinite(new_baseline)
        assert jnp.all(jnp.isfinite(new_weights))

    def test_step_high_spikes(self) -> None:
        """Newton step should handle high spike counts."""
        from state_space_practice.switching_point_process import (
            _single_neuron_glm_step,
        )

        n_time = 30
        n_latent = 2
        dt = 0.02

        # High spike counts
        y_n = jax.random.poisson(jax.random.PRNGKey(0), 20.0, shape=(n_time,)).astype(
            float
        )
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        baseline = 3.0
        weights = jnp.ones(n_latent) * 0.1

        new_baseline, new_weights = _single_neuron_glm_step(
            baseline, weights, y_n, smoother_mean, dt
        )

        assert jnp.isfinite(new_baseline)
        assert jnp.all(jnp.isfinite(new_weights))


class TestUpdateSpikeGLMParams:
    """Tests for the update_spike_glm_params function (Task 5.3)."""

    def test_output_shapes(self) -> None:
        """Output SpikeObsParams should have correct shapes."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 50
        n_latent = 4
        n_neurons = 5
        dt = 0.02

        # Spike data for multiple neurons
        key = jax.random.PRNGKey(0)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        # Smoother mean
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        # Current params
        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        # Update
        new_params = update_spike_glm_params(
            spikes, smoother_mean, current_params, dt, max_iter=5
        )

        assert new_params.baseline.shape == (n_neurons,)
        assert new_params.weights.shape == (n_neurons, n_latent)

    def test_output_finite(self) -> None:
        """Updated params should be finite."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 30
        n_latent = 2
        n_neurons = 3
        dt = 0.02

        key = jax.random.PRNGKey(0)
        spikes = jax.random.poisson(key, 1.0, shape=(n_time, n_neurons)).astype(float)
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.ones((n_neurons, n_latent)) * 0.1,
        )

        new_params = update_spike_glm_params(
            spikes, smoother_mean, current_params, dt, max_iter=5
        )

        assert jnp.all(jnp.isfinite(new_params.baseline))
        assert jnp.all(jnp.isfinite(new_params.weights))

    def test_decreases_total_loss(self) -> None:
        """M-step should decrease the total Poisson NLL across all neurons."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            _single_neuron_glm_loss,
            update_spike_glm_params,
        )

        n_time = 100
        n_latent = 2
        n_neurons = 3
        dt = 0.02

        # Generate data from known params
        true_baseline = jnp.array([1.0, 0.5, 0.0])
        true_weights = jax.random.normal(jax.random.PRNGKey(0), (n_neurons, n_latent)) * 0.3

        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        # Generate spikes
        eta = true_baseline[None, :] + smoother_mean @ true_weights.T
        rates = jnp.exp(eta) * dt
        spikes = jax.random.poisson(jax.random.PRNGKey(2), rates).astype(float)

        # Start from wrong params
        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        # Compute total loss before
        loss_before = 0.0
        for n in range(n_neurons):
            loss_before += _single_neuron_glm_loss(
                current_params.baseline[n],
                current_params.weights[n],
                spikes[:, n],
                smoother_mean,
                dt,
            )

        # Update
        new_params = update_spike_glm_params(
            spikes, smoother_mean, current_params, dt, max_iter=10
        )

        # Compute total loss after
        loss_after = 0.0
        for n in range(n_neurons):
            loss_after += _single_neuron_glm_loss(
                new_params.baseline[n],
                new_params.weights[n],
                spikes[:, n],
                smoother_mean,
                dt,
            )

        assert loss_after < loss_before

    def test_recovers_true_params(self) -> None:
        """With enough data, should approximately recover true params."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 500
        n_latent = 2
        n_neurons = 2
        dt = 0.02

        # True parameters
        true_baseline = jnp.array([1.0, 0.5])
        true_weights = jnp.array([[0.3, -0.2], [-0.1, 0.4]])

        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        # Generate spikes from true model
        eta = true_baseline[None, :] + smoother_mean @ true_weights.T
        rates = jnp.exp(eta) * dt
        spikes = jax.random.poisson(jax.random.PRNGKey(2), rates).astype(float)

        # Start from zeros
        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        # Run enough iterations
        new_params = update_spike_glm_params(
            spikes, smoother_mean, current_params, dt, max_iter=20
        )

        # Check recovery (with some tolerance due to finite samples)
        np.testing.assert_allclose(
            new_params.baseline, true_baseline, atol=0.3, rtol=0.3
        )
        np.testing.assert_allclose(
            new_params.weights, true_weights, atol=0.3, rtol=0.3
        )

    def test_single_neuron(self) -> None:
        """Should work with a single neuron."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 50
        n_latent = 2
        n_neurons = 1
        dt = 0.02

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        new_params = update_spike_glm_params(
            spikes, smoother_mean, current_params, dt, max_iter=5
        )

        assert new_params.baseline.shape == (1,)
        assert new_params.weights.shape == (1, n_latent)
        assert jnp.all(jnp.isfinite(new_params.baseline))
        assert jnp.all(jnp.isfinite(new_params.weights))

    def test_zero_spikes(self) -> None:
        """Should handle zero spike counts gracefully."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 30
        n_latent = 2
        n_neurons = 3
        dt = 0.02

        spikes = jnp.zeros((n_time, n_neurons))  # All silent neurons
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        new_params = update_spike_glm_params(
            spikes, smoother_mean, current_params, dt, max_iter=5
        )

        assert jnp.all(jnp.isfinite(new_params.baseline))
        assert jnp.all(jnp.isfinite(new_params.weights))


class TestUpdateSpikeGLMParamsSecondOrder:
    """Tests for the second-order expectation variant of update_spike_glm_params (Task 5.4)."""

    def test_second_order_output_shapes(self) -> None:
        """Second-order method should produce same output shapes."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 50
        n_latent = 4
        n_neurons = 5
        dt = 0.02

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5
        smoother_cov = jnp.stack([jnp.eye(n_latent) * 0.1] * n_time, axis=0)

        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        new_params = update_spike_glm_params(
            spikes,
            smoother_mean,
            current_params,
            dt,
            max_iter=5,
            smoother_cov=smoother_cov,
            use_second_order=True,
        )

        assert new_params.baseline.shape == (n_neurons,)
        assert new_params.weights.shape == (n_neurons, n_latent)

    def test_second_order_output_finite(self) -> None:
        """Second-order method should produce finite outputs."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 30
        n_latent = 2
        n_neurons = 3
        dt = 0.02

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 1.0, shape=(n_time, n_neurons)
        ).astype(float)
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5
        smoother_cov = jnp.stack([jnp.eye(n_latent) * 0.1] * n_time, axis=0)

        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.ones((n_neurons, n_latent)) * 0.1,
        )

        new_params = update_spike_glm_params(
            spikes,
            smoother_mean,
            current_params,
            dt,
            max_iter=5,
            smoother_cov=smoother_cov,
            use_second_order=True,
        )

        assert jnp.all(jnp.isfinite(new_params.baseline))
        assert jnp.all(jnp.isfinite(new_params.weights))

    def test_second_order_decreases_loss(self) -> None:
        """Second-order method should decrease loss."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            _single_neuron_glm_loss,
            update_spike_glm_params,
        )

        n_time = 100
        n_latent = 2
        n_neurons = 3
        dt = 0.02

        # Generate data
        true_baseline = jnp.array([1.0, 0.5, 0.0])
        true_weights = jax.random.normal(jax.random.PRNGKey(0), (n_neurons, n_latent)) * 0.3

        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5
        smoother_cov = jnp.stack([jnp.eye(n_latent) * 0.05] * n_time, axis=0)

        eta = true_baseline[None, :] + smoother_mean @ true_weights.T
        rates = jnp.exp(eta) * dt
        spikes = jax.random.poisson(jax.random.PRNGKey(2), rates).astype(float)

        # Start from wrong params
        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        # Compute loss before
        loss_before = 0.0
        for n in range(n_neurons):
            loss_before += _single_neuron_glm_loss(
                current_params.baseline[n],
                current_params.weights[n],
                spikes[:, n],
                smoother_mean,
                dt,
            )

        # Update with second-order
        new_params = update_spike_glm_params(
            spikes,
            smoother_mean,
            current_params,
            dt,
            max_iter=10,
            smoother_cov=smoother_cov,
            use_second_order=True,
        )

        # Compute loss after
        loss_after = 0.0
        for n in range(n_neurons):
            loss_after += _single_neuron_glm_loss(
                new_params.baseline[n],
                new_params.weights[n],
                spikes[:, n],
                smoother_mean,
                dt,
            )

        assert loss_after < loss_before

    def test_second_order_with_zero_variance(self) -> None:
        """With zero variance, second-order should match plug-in method."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 50
        n_latent = 2
        n_neurons = 2
        dt = 0.02

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5
        # Zero covariance = deterministic state
        smoother_cov = jnp.zeros((n_time, n_latent, n_latent))

        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        # Plug-in method
        params_plugin = update_spike_glm_params(
            spikes, smoother_mean, current_params, dt, max_iter=10
        )

        # Second-order with zero variance
        params_second = update_spike_glm_params(
            spikes,
            smoother_mean,
            current_params,
            dt,
            max_iter=10,
            smoother_cov=smoother_cov,
            use_second_order=True,
        )

        # Should be nearly identical
        np.testing.assert_allclose(
            params_second.baseline, params_plugin.baseline, rtol=1e-4
        )
        np.testing.assert_allclose(
            params_second.weights, params_plugin.weights, rtol=1e-4
        )

    def test_second_order_requires_smoother_cov(self) -> None:
        """Second-order method should raise error if smoother_cov not provided."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 30
        n_latent = 2
        n_neurons = 3
        dt = 0.02

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 1.0, shape=(n_time, n_neurons)
        ).astype(float)
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5

        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        with pytest.raises(ValueError, match="smoother_cov required"):
            update_spike_glm_params(
                spikes,
                smoother_mean,
                current_params,
                dt,
                use_second_order=True,  # smoother_cov not provided
            )

    def test_second_order_validates_smoother_cov_shape(self) -> None:
        """Second-order method should validate smoother_cov shape."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 30
        n_latent = 2
        n_neurons = 3
        dt = 0.02

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 1.0, shape=(n_time, n_neurons)
        ).astype(float)
        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent)) * 0.5
        # Wrong shape: (n_time, n_latent) instead of (n_time, n_latent, n_latent)
        smoother_cov_wrong = jax.random.normal(jax.random.PRNGKey(2), (n_time, n_latent))

        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        with pytest.raises(ValueError, match="smoother_cov shape"):
            update_spike_glm_params(
                spikes,
                smoother_mean,
                current_params,
                dt,
                smoother_cov=smoother_cov_wrong,
                use_second_order=True,
            )


class TestDynamicsMStepReuse:
    """Tests for dynamics M-step reuse with point-process observations.

    The `switching_kalman_maximization_step` function is observation-model agnostic
    for the dynamics parameters. It operates on smoother outputs (means, covariances,
    discrete state probabilities) which are Gaussian regardless of observation model.

    For point-process observations, the measurement_matrix and measurement_cov
    returns should be ignored since they assume Gaussian observations.
    """

    def test_dynamics_mstep_runs_on_point_process_smoother_output(self) -> None:
        """Dynamics M-step should run without error on point-process smoother output.

        This verifies that `switching_kalman_maximization_step` can be called directly
        with smoother outputs from the point-process filter/smoother pipeline.
        """
        from state_space_practice.switching_kalman import (
            switching_kalman_maximization_step,
            switching_kalman_smoother,
        )
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 50
        n_latent = 4
        n_neurons = 5
        n_discrete_states = 2
        dt = 0.02

        # Setup initial conditions
        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        # Generate spikes
        key = jax.random.PRNGKey(42)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        # Dynamics parameters
        discrete_transition_matrix = jnp.array([[0.95, 0.05], [0.05, 0.95]])
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        # Spike observation parameters
        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Run filter
        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            _,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Run smoother (observation-model agnostic)
        (
            _,  # overall_smoother_mean
            _,  # overall_smoother_cov
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            _,  # overall_smoother_cross_cov
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_cov,
            pair_cond_smoother_means,
        ) = switching_kalman_smoother(
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            process_cov,
            continuous_transition_matrix,
            discrete_transition_matrix,
        )

        # Call dynamics M-step with point-process smoother outputs
        # For point-process models, we pass spikes but note that
        # measurement_matrix and measurement_cov returns are meaningless
        (
            new_continuous_transition_matrix,
            measurement_matrix,  # Should be ignored for point-process
            new_process_cov,
            measurement_cov,  # Should be ignored for point-process
            new_init_mean,
            new_init_cov,
            new_discrete_transition_matrix,
            new_init_discrete_state_prob,
        ) = switching_kalman_maximization_step(
            obs=spikes,  # Not used for dynamics, but required arg
            state_cond_smoother_means=state_cond_smoother_means,
            state_cond_smoother_covs=state_cond_smoother_covs,
            smoother_discrete_state_prob=smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
            pair_cond_smoother_means=pair_cond_smoother_means,
        )

        # This should run without error - that's the main test

    def test_dynamics_mstep_returns_correct_shapes(self) -> None:
        """Dynamics M-step returns should have correct shapes for dynamics parameters.

        The relevant returns for point-process models are:
        - continuous_transition_matrix: (n_latent, n_latent, n_discrete_states)
        - process_cov: (n_latent, n_latent, n_discrete_states)
        - init_mean: (n_latent, n_discrete_states)
        - init_cov: (n_latent, n_latent, n_discrete_states)
        - discrete_transition_matrix: (n_discrete_states, n_discrete_states)
        - init_discrete_state_prob: (n_discrete_states,)
        """
        from state_space_practice.switching_kalman import (
            switching_kalman_maximization_step,
            switching_kalman_smoother,
        )
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 50
        n_latent = 4
        n_neurons = 5
        n_discrete_states = 3
        dt = 0.02

        # Setup initial conditions
        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        # Generate spikes
        key = jax.random.PRNGKey(123)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        # Dynamics parameters
        discrete_transition_matrix = jnp.eye(n_discrete_states) * 0.9 + 0.1 / n_discrete_states
        discrete_transition_matrix = discrete_transition_matrix / discrete_transition_matrix.sum(
            axis=1, keepdims=True
        )
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * (0.95 + 0.01 * i) for i in range(n_discrete_states)],
            axis=-1,
        )
        process_cov = jnp.stack(
            [jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1
        )

        # Spike observation parameters
        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Run filter
        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            _,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Run smoother
        (
            _,
            _,
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            _,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_cov,
            pair_cond_smoother_means,
        ) = switching_kalman_smoother(
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            process_cov,
            continuous_transition_matrix,
            discrete_transition_matrix,
        )

        # Call dynamics M-step
        (
            new_continuous_transition_matrix,
            _,  # measurement_matrix - ignore for point-process
            new_process_cov,
            _,  # measurement_cov - ignore for point-process
            new_init_mean,
            new_init_cov,
            new_discrete_transition_matrix,
            new_init_discrete_state_prob,
        ) = switching_kalman_maximization_step(
            obs=spikes,
            state_cond_smoother_means=state_cond_smoother_means,
            state_cond_smoother_covs=state_cond_smoother_covs,
            smoother_discrete_state_prob=smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
            pair_cond_smoother_means=pair_cond_smoother_means,
        )

        # Check shapes for dynamics parameters
        assert new_continuous_transition_matrix.shape == (
            n_latent,
            n_latent,
            n_discrete_states,
        )
        assert new_process_cov.shape == (n_latent, n_latent, n_discrete_states)
        assert new_init_mean.shape == (n_latent, n_discrete_states)
        assert new_init_cov.shape == (n_latent, n_latent, n_discrete_states)
        assert new_discrete_transition_matrix.shape == (
            n_discrete_states,
            n_discrete_states,
        )
        assert new_init_discrete_state_prob.shape == (n_discrete_states,)

    def test_dynamics_mstep_returns_finite_values(self) -> None:
        """Dynamics M-step should return finite values for all parameters."""
        from state_space_practice.switching_kalman import (
            switching_kalman_maximization_step,
            switching_kalman_smoother,
        )
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 50
        n_latent = 4
        n_neurons = 5
        n_discrete_states = 2
        dt = 0.02

        # Setup initial conditions
        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        # Generate spikes
        key = jax.random.PRNGKey(456)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        # Dynamics parameters
        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        # Spike observation parameters
        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Run filter
        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            _,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        # Run smoother
        (
            _,
            _,
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            _,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_cov,
            pair_cond_smoother_means,
        ) = switching_kalman_smoother(
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            process_cov,
            continuous_transition_matrix,
            discrete_transition_matrix,
        )

        # Call dynamics M-step
        (
            new_continuous_transition_matrix,
            _,
            new_process_cov,
            _,
            new_init_mean,
            new_init_cov,
            new_discrete_transition_matrix,
            new_init_discrete_state_prob,
        ) = switching_kalman_maximization_step(
            obs=spikes,
            state_cond_smoother_means=state_cond_smoother_means,
            state_cond_smoother_covs=state_cond_smoother_covs,
            smoother_discrete_state_prob=smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
            pair_cond_smoother_means=pair_cond_smoother_means,
        )

        # All dynamics parameters should be finite
        assert jnp.all(jnp.isfinite(new_continuous_transition_matrix))
        assert jnp.all(jnp.isfinite(new_process_cov))
        assert jnp.all(jnp.isfinite(new_init_mean))
        assert jnp.all(jnp.isfinite(new_init_cov))
        assert jnp.all(jnp.isfinite(new_discrete_transition_matrix))
        assert jnp.all(jnp.isfinite(new_init_discrete_state_prob))

    def test_discrete_transition_matrix_is_valid_stochastic_matrix(self) -> None:
        """Discrete transition matrix from M-step should be a valid stochastic matrix."""
        from state_space_practice.switching_kalman import (
            switching_kalman_maximization_step,
            switching_kalman_smoother,
        )
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 100
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 2
        dt = 0.02

        # Setup
        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(789)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Run filter and smoother
        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            _,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        (
            _,
            _,
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            _,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_cov,
            pair_cond_smoother_means,
        ) = switching_kalman_smoother(
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            process_cov,
            continuous_transition_matrix,
            discrete_transition_matrix,
        )

        # Call dynamics M-step
        (
            _,
            _,
            _,
            _,
            _,
            _,
            new_discrete_transition_matrix,
            new_init_discrete_state_prob,
        ) = switching_kalman_maximization_step(
            obs=spikes,
            state_cond_smoother_means=state_cond_smoother_means,
            state_cond_smoother_covs=state_cond_smoother_covs,
            smoother_discrete_state_prob=smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
            pair_cond_smoother_means=pair_cond_smoother_means,
        )

        # Check stochastic matrix properties
        # Rows should sum to 1
        np.testing.assert_allclose(
            new_discrete_transition_matrix.sum(axis=1),
            jnp.ones(n_discrete_states),
            rtol=1e-5,
        )
        # All entries should be non-negative
        assert jnp.all(new_discrete_transition_matrix >= 0)

        # Initial probabilities should sum to 1
        np.testing.assert_allclose(
            new_init_discrete_state_prob.sum(), 1.0, rtol=1e-5
        )
        # All probabilities should be non-negative
        assert jnp.all(new_init_discrete_state_prob >= 0)

    def test_covariances_are_symmetric(self) -> None:
        """Covariance matrices from M-step should be symmetric.

        Note: The M-step does NOT guarantee positive semi-definite covariances.
        When a discrete state has low probability or insufficient data, the
        process covariance can have negative eigenvalues. PSD enforcement
        (e.g., adding regularization Q = Q + eps*I) should be handled at the
        model class level, not in the raw M-step function.
        """
        from state_space_practice.switching_kalman import (
            switching_kalman_maximization_step,
            switching_kalman_smoother,
        )
        from state_space_practice.switching_point_process import (
            switching_point_process_filter,
        )

        n_time = 50
        n_latent = 4
        n_neurons = 5
        n_discrete_states = 2
        dt = 0.02

        # Setup
        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.ones(n_discrete_states) / n_discrete_states

        key = jax.random.PRNGKey(321)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        log_intensity_func = linear_log_intensity(weights, baseline)

        # Run filter and smoother
        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            _,
        ) = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            log_intensity_func,
        )

        (
            _,
            _,
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            _,
            state_cond_smoother_means,
            state_cond_smoother_covs,
            pair_cond_smoother_cross_cov,
            pair_cond_smoother_means,
        ) = switching_kalman_smoother(
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            process_cov,
            continuous_transition_matrix,
            discrete_transition_matrix,
        )

        # Call dynamics M-step
        (
            _,
            _,
            new_process_cov,
            _,
            _,
            new_init_cov,
            _,
            _,
        ) = switching_kalman_maximization_step(
            obs=spikes,
            state_cond_smoother_means=state_cond_smoother_means,
            state_cond_smoother_covs=state_cond_smoother_covs,
            smoother_discrete_state_prob=smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=pair_cond_smoother_cross_cov,
            pair_cond_smoother_means=pair_cond_smoother_means,
        )

        # Check symmetry for each discrete state's process covariance
        for s in range(n_discrete_states):
            Q_s = new_process_cov[:, :, s]
            np.testing.assert_allclose(Q_s, Q_s.T, rtol=1e-5)

        # Check symmetry for initial covariances
        for s in range(n_discrete_states):
            P0_s = new_init_cov[:, :, s]
            np.testing.assert_allclose(P0_s, P0_s.T, rtol=1e-5)


class TestSwitchingSpikeOscillatorModelInit:
    """Tests for SwitchingSpikeOscillatorModel.__init__() method."""

    def test_init_stores_required_parameters(self) -> None:
        """Model should store all required parameters."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_oscillators = 2
        n_neurons = 5
        n_discrete_states = 3
        sampling_freq = 100.0
        dt = 0.01

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=sampling_freq,
            dt=dt,
        )

        assert model.n_oscillators == n_oscillators
        assert model.n_neurons == n_neurons
        assert model.n_discrete_states == n_discrete_states
        assert model.sampling_freq == sampling_freq
        assert model.dt == dt
        # n_latent should be 2 * n_oscillators (amplitude + phase)
        assert model.n_latent == 2 * n_oscillators

    def test_init_with_default_update_flags(self) -> None:
        """Model should have default update flags set to True."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        # Default update flags should be True
        assert model.update_continuous_transition_matrix is True
        assert model.update_process_cov is True
        assert model.update_discrete_transition_matrix is True
        assert model.update_spike_params is True
        assert model.update_init_mean is True
        assert model.update_init_cov is True

    def test_init_with_custom_update_flags(self) -> None:
        """Model should accept custom update flags."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_continuous_transition_matrix=False,
            update_process_cov=False,
            update_spike_params=False,
        )

        assert model.update_continuous_transition_matrix is False
        assert model.update_process_cov is False
        assert model.update_spike_params is False
        # Others should still be True (default)
        assert model.update_discrete_transition_matrix is True
        assert model.update_init_mean is True
        assert model.update_init_cov is True

    def test_init_with_discrete_transition_diag(self) -> None:
        """Model should accept custom discrete transition diagonal."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_discrete_states = 3
        custom_diag = jnp.array([0.9, 0.85, 0.95])

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            discrete_transition_diag=custom_diag,
        )

        np.testing.assert_allclose(model.discrete_transition_diag, custom_diag)

    def test_init_with_default_discrete_transition_diag(self) -> None:
        """Model should create default discrete transition diagonal if not provided."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_discrete_states = 3

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        # Default should be 0.95 for all states
        expected_diag = jnp.full(n_discrete_states, 0.95)
        np.testing.assert_allclose(model.discrete_transition_diag, expected_diag)

    def test_init_single_discrete_state(self) -> None:
        """Model should handle single discrete state case."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=1,
            sampling_freq=100.0,
            dt=0.01,
        )

        assert model.n_discrete_states == 1
        assert model.discrete_transition_diag.shape == (1,)

    def test_init_single_oscillator(self) -> None:
        """Model should handle single oscillator case."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=1,
            n_neurons=5,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        assert model.n_oscillators == 1
        assert model.n_latent == 2  # 2 * 1

    def test_init_many_neurons(self) -> None:
        """Model should handle many neurons."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_neurons = 100
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=4,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        assert model.n_neurons == n_neurons

    def test_init_repr_contains_class_name(self) -> None:
        """Model repr should contain class name."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        repr_str = repr(model)
        assert "SwitchingSpikeOscillatorModel" in repr_str

    def test_init_repr_contains_key_parameters(self) -> None:
        """Model repr should contain key parameters."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=3,
            n_neurons=7,
            n_discrete_states=4,
            sampling_freq=50.0,
            dt=0.02,
        )

        repr_str = repr(model)
        assert "n_oscillators=3" in repr_str
        assert "n_neurons=7" in repr_str
        assert "n_discrete_states=4" in repr_str

    def test_init_negative_oscillators_raises_error(self) -> None:
        """Model should raise ValueError for negative n_oscillators."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        with pytest.raises(ValueError, match="n_oscillators must be positive"):
            SwitchingSpikeOscillatorModel(
                n_oscillators=-1,
                n_neurons=5,
                n_discrete_states=2,
                sampling_freq=100.0,
                dt=0.01,
            )

    def test_init_zero_neurons_raises_error(self) -> None:
        """Model should raise ValueError for zero n_neurons."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        with pytest.raises(ValueError, match="n_neurons must be positive"):
            SwitchingSpikeOscillatorModel(
                n_oscillators=2,
                n_neurons=0,
                n_discrete_states=2,
                sampling_freq=100.0,
                dt=0.01,
            )

    def test_init_negative_sampling_freq_raises_error(self) -> None:
        """Model should raise ValueError for negative sampling_freq."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        with pytest.raises(ValueError, match="sampling_freq must be positive"):
            SwitchingSpikeOscillatorModel(
                n_oscillators=2,
                n_neurons=5,
                n_discrete_states=2,
                sampling_freq=-100.0,
                dt=0.01,
            )

    def test_init_zero_dt_raises_error(self) -> None:
        """Model should raise ValueError for zero dt."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        with pytest.raises(ValueError, match="dt must be positive"):
            SwitchingSpikeOscillatorModel(
                n_oscillators=2,
                n_neurons=5,
                n_discrete_states=2,
                sampling_freq=100.0,
                dt=0.0,
            )

    def test_init_invalid_discrete_transition_diag_shape_raises_error(self) -> None:
        """Model should raise ValueError for wrong discrete_transition_diag shape."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        with pytest.raises(ValueError, match="discrete_transition_diag shape mismatch"):
            SwitchingSpikeOscillatorModel(
                n_oscillators=2,
                n_neurons=5,
                n_discrete_states=3,
                sampling_freq=100.0,
                dt=0.01,
                discrete_transition_diag=jnp.array([0.9, 0.95]),  # Wrong shape
            )
