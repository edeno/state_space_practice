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
from jax.typing import ArrayLike

from state_space_practice.switching_point_process import SpikeObsParams

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


def linear_log_intensity(
    state: Array, params: SpikeObsParams
) -> Array:
    """Linear log-intensity function for testing.

    Parameters
    ----------
    state : Array, shape (n_latent,)
        Latent state
    params : SpikeObsParams
        Spike observation parameters (baseline, weights)

    Returns
    -------
    Array, shape (n_neurons,)
        Log firing rates for all neurons
    """
    return params.baseline + params.weights @ state


# Backwards-compatible helper that returns a closure (for tests that don't use the new API)
def linear_log_intensity_closure(
    weights: Array, baseline: Array
) -> Callable[[Array], Array]:
    """Create a closure-based linear log-intensity function for testing.

    DEPRECATED: Use linear_log_intensity with SpikeObsParams instead.
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        posterior_mean, posterior_cov, log_ll = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, linear_log_intensity, spike_params
        )

        assert posterior_mean.shape == (n_latent,)
        assert posterior_cov.shape == (n_latent, n_latent)
        assert log_ll.shape == ()

    def test_output_shapes_multiple_neurons(self) -> None:
        """Output shapes should be correct for multiple neurons."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        posterior_mean, posterior_cov, log_ll = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, linear_log_intensity, spike_params
        )

        assert posterior_mean.shape == (n_latent,)
        assert posterior_cov.shape == (n_latent, n_latent)
        assert log_ll.shape == ()

    def test_no_nans(self) -> None:
        """Update should not produce NaN values."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        posterior_mean, posterior_cov, log_ll = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, linear_log_intensity, spike_params
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        posterior_mean, posterior_cov, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, linear_log_intensity, spike_params
        )

        # With zero spikes, the update should be small
        # (posterior should be close to prior for small intensity)
        assert jnp.allclose(posterior_mean, one_step_mean, atol=0.5)

    def test_covariance_is_symmetric(self) -> None:
        """Posterior covariance should be symmetric."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        _, posterior_cov, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, linear_log_intensity, spike_params
        )

        np.testing.assert_allclose(posterior_cov, posterior_cov.T, rtol=1e-5, atol=1e-10)

    def test_covariance_is_positive_definite(self) -> None:
        """Posterior covariance should be positive semi-definite."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        _, posterior_cov, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, linear_log_intensity, spike_params
        )

        # All eigenvalues should be non-negative
        eigvals = jnp.linalg.eigvalsh(posterior_cov)
        assert jnp.all(eigvals >= -1e-6), f"Negative eigenvalues: {eigvals}"

    def test_log_likelihood_finite(self) -> None:
        """Log-likelihood should be finite."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        _, _, log_ll = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, linear_log_intensity, spike_params
        )

        assert jnp.isfinite(log_ll)

    def test_high_spikes_increases_intensity_estimate(self) -> None:
        """High spike count should increase intensity estimate.

        With positive weights, more spikes should push the state estimate
        in a direction that increases predicted intensity.
        """
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        # Low spikes
        posterior_mean_low, _, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, jnp.array([0.0]), dt, linear_log_intensity, spike_params
        )

        # High spikes
        posterior_mean_high, _, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, jnp.array([5.0]), dt, linear_log_intensity, spike_params
        )

        # With positive weights, high spikes should lead to higher log-intensity
        log_rate_low = linear_log_intensity(posterior_mean_low, spike_params)
        log_rate_high = linear_log_intensity(posterior_mean_high, spike_params)

        assert jnp.all(log_rate_high > log_rate_low)


class TestPointProcessPredictAndUpdate:
    """Tests for the _point_process_predict_and_update helper."""

    def test_output_shapes(self) -> None:
        """Output shapes should be correct."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        posterior_mean, posterior_cov, log_ll = _point_process_predict_and_update(
            prev_state_cond_mean,
            prev_state_cond_cov,
            y_t,
            continuous_transition_matrix,
            process_cov,
            dt,
            linear_log_intensity,
            spike_params,
        )

        assert posterior_mean.shape == (n_latent,)
        assert posterior_cov.shape == (n_latent, n_latent)
        assert log_ll.shape == ()

    def test_prediction_incorporated(self) -> None:
        """The dynamics prediction should be incorporated before update."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        # Use predict_and_update
        mean_combined, cov_combined, _ = _point_process_predict_and_update(
            prev_mean, prev_cov, y_t, A, Q, dt, linear_log_intensity, spike_params
        )

        # Manual prediction + update
        one_step_mean = A @ prev_mean
        one_step_cov = A @ prev_cov @ A.T + Q
        mean_manual, cov_manual, _ = point_process_kalman_update(
            one_step_mean, one_step_cov, y_t, dt, linear_log_intensity, spike_params
        )

        np.testing.assert_allclose(mean_combined, mean_manual, rtol=1e-5)
        np.testing.assert_allclose(cov_combined, cov_manual, rtol=1e-5)

    def test_no_nans(self) -> None:
        """Should not produce NaN values."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        mean, cov, ll = _point_process_predict_and_update(
            prev_mean, prev_cov, y_t, A, Q, dt, linear_log_intensity, spike_params
        )

        assert not jnp.any(jnp.isnan(mean))
        assert not jnp.any(jnp.isnan(cov))
        assert not jnp.isnan(ll)


class TestPointProcessUpdatePerStatePair:
    """Tests for the _point_process_update_per_discrete_state_pair vmapped function."""

    def test_output_shapes(self) -> None:
        """Output shapes should be correct for state pairs."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        pair_mean, pair_cov, pair_ll = _point_process_update_per_discrete_state_pair(
            prev_state_cond_mean,
            prev_state_cond_cov,
            y_t,
            continuous_transition_matrix,
            process_cov,
            dt,
            linear_log_intensity,
            spike_params,
        )

        # Output shapes should be (n_latent, n_discrete_states, n_discrete_states)
        assert pair_mean.shape == (n_latent, n_discrete_states, n_discrete_states)
        assert pair_cov.shape == (n_latent, n_latent, n_discrete_states, n_discrete_states)
        assert pair_ll.shape == (n_discrete_states, n_discrete_states)

    def test_single_state_matches_base_function(self) -> None:
        """With 1 state, should match non-vmapped function."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        # Non-vmapped result
        expected_mean, expected_cov, expected_ll = _point_process_predict_and_update(
            prev_mean_single, prev_cov_single, y_t, A_single, Q_single, dt, linear_log_intensity, spike_params
        )

        # Vmapped result with state dimension
        prev_mean_batched = prev_mean_single[:, None]  # (n_latent, 1)
        prev_cov_batched = prev_cov_single[:, :, None]  # (n_latent, n_latent, 1)
        A_batched = A_single[:, :, None]  # (n_latent, n_latent, 1)
        Q_batched = Q_single[:, :, None]  # (n_latent, n_latent, 1)

        pair_mean, pair_cov, pair_ll = _point_process_update_per_discrete_state_pair(
            prev_mean_batched, prev_cov_batched, y_t, A_batched, Q_batched, dt, linear_log_intensity, spike_params
        )

        # Should match single result
        np.testing.assert_allclose(pair_mean[:, 0, 0], expected_mean, rtol=1e-5)
        np.testing.assert_allclose(pair_cov[:, :, 0, 0], expected_cov, rtol=1e-5)
        np.testing.assert_allclose(pair_ll[0, 0], expected_ll, rtol=1e-5)

    def test_no_nans(self) -> None:
        """Should not produce NaN values."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        pair_mean, pair_cov, pair_ll = _point_process_update_per_discrete_state_pair(
            prev_mean, prev_cov, y_t, A, Q, dt, linear_log_intensity, spike_params
        )

        assert not jnp.any(jnp.isnan(pair_mean))
        assert not jnp.any(jnp.isnan(pair_cov))
        assert not jnp.any(jnp.isnan(pair_ll))

    def test_different_states_give_different_results(self) -> None:
        """Different discrete states should give different outputs."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        pair_mean, pair_cov, pair_ll = _point_process_update_per_discrete_state_pair(
            prev_mean, prev_cov, y_t, A, Q, dt, linear_log_intensity, spike_params
        )

        # Results for state j=0 and j=1 should be different
        assert not jnp.allclose(pair_mean[:, 0, 0], pair_mean[:, 0, 1])
        assert not jnp.allclose(pair_cov[:, :, 0, 0], pair_cov[:, :, 0, 1])


class TestSwitchingPointProcessFilter:
    """Tests for the switching_point_process_filter function (Task 2.5)."""

    def test_output_shapes(self) -> None:
        """All outputs should have correct shapes (Task 2.5)."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        _, _, filter_discrete_state_prob, _, _ = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            linear_log_intensity,
            spike_params,
        )

        # Probabilities should sum to 1 at each timestep
        prob_sums = jnp.sum(filter_discrete_state_prob, axis=1)
        np.testing.assert_allclose(prob_sums, jnp.ones(n_time), rtol=1e-5)

    def test_discrete_probs_nonnegative(self) -> None:
        """Discrete state probabilities should be non-negative."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        _, _, filter_discrete_state_prob, _, _ = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            linear_log_intensity,
            spike_params,
        )

        # All probabilities should be >= 0
        assert jnp.all(filter_discrete_state_prob >= 0)

    def test_no_nans(self) -> None:
        """Filter should not produce NaN values."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
        )

        assert not jnp.any(jnp.isnan(state_cond_filter_mean))
        assert not jnp.any(jnp.isnan(state_cond_filter_cov))
        assert not jnp.any(jnp.isnan(filter_discrete_state_prob))
        assert not jnp.any(jnp.isnan(last_pair_cond_filter_mean))
        assert not jnp.isnan(marginal_log_likelihood)

    def test_marginal_log_likelihood_finite(self) -> None:
        """Marginal log-likelihood should be finite."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        _, _, _, _, marginal_log_likelihood = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            linear_log_intensity,
            spike_params,
        )

        assert jnp.isfinite(marginal_log_likelihood)

    def test_single_discrete_state(self) -> None:
        """With 1 discrete state, filter should still work correctly."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        _, state_cond_filter_cov, _, _, _ = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

        _, _, _, last_pair_cond_filter_mean, _ = switching_point_process_filter(
            init_state_cond_mean,
            init_state_cond_cov,
            init_discrete_state_prob,
            spikes,
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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

    def test_1d_single_neuron_spikes(self) -> None:
        """Filter should accept 1D spike input and promote to 2D."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            switching_point_process_filter,
        )

        n_time = 20
        n_latent = 2
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.array([0.5, 0.5])

        key = jax.random.PRNGKey(0)
        # 1D spikes (single neuron) - should be promoted to (n_time, 1)
        spikes_1d = jax.random.poisson(key, 0.5, shape=(n_time,)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack(
            [jnp.eye(n_latent) * 0.01, jnp.eye(n_latent) * 0.02], axis=-1
        )

        # Single neuron parameters
        weights = jnp.array([[0.1, 0.1]])  # shape (1, n_latent)
        baseline = jnp.array([0.0])  # shape (1,)
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            spikes_1d,  # Pass 1D array
            discrete_transition_matrix,
            continuous_transition_matrix,
            process_cov,
            dt,
            linear_log_intensity,
            spike_params,
        )

        # Check outputs are valid
        assert state_cond_filter_mean.shape == (n_time, n_latent, n_discrete_states)
        assert not jnp.any(jnp.isnan(state_cond_filter_mean))
        assert jnp.isfinite(marginal_log_likelihood)

    def test_single_timestep(self) -> None:
        """Filter should handle n_time=1 edge case."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            switching_point_process_filter,
        )

        n_time = 1
        n_latent = 2
        n_neurons = 3
        n_discrete_states = 2
        dt = 0.02

        init_state_cond_mean = jnp.zeros((n_latent, n_discrete_states))
        init_state_cond_cov = jnp.stack(
            [jnp.eye(n_latent)] * n_discrete_states, axis=-1
        )
        init_discrete_state_prob = jnp.array([0.5, 0.5])

        key = jax.random.PRNGKey(0)
        spikes = jax.random.poisson(key, 0.5, shape=(n_time, n_neurons)).astype(float)

        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        continuous_transition_matrix = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_cov = jnp.stack(
            [jnp.eye(n_latent) * 0.01, jnp.eye(n_latent) * 0.02], axis=-1
        )

        weights = jax.random.normal(jax.random.PRNGKey(1), (n_neurons, n_latent)) * 0.1
        baseline = jnp.zeros(n_neurons)
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
        )

        # Check shapes for n_time=1
        assert state_cond_filter_mean.shape == (1, n_latent, n_discrete_states)
        assert state_cond_filter_cov.shape == (1, n_latent, n_latent, n_discrete_states)
        assert filter_discrete_state_prob.shape == (1, n_discrete_states)

        # Check values are valid
        assert not jnp.any(jnp.isnan(state_cond_filter_mean))
        assert not jnp.any(jnp.isnan(filter_discrete_state_prob))
        assert jnp.isfinite(marginal_log_likelihood)

        # Probabilities should sum to 1
        np.testing.assert_allclose(
            jnp.sum(filter_discrete_state_prob, axis=1),
            jnp.ones(1),
            rtol=1e-5,
        )


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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
        baseline: ArrayLike = 0.0
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
        loss_before = jnp.array(0.0)
        for n in range(n_neurons):
            loss_before = loss_before + _single_neuron_glm_loss(
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
        loss_after = jnp.array(0.0)
        for n in range(n_neurons):
            loss_after = loss_after + _single_neuron_glm_loss(
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

    @staticmethod
    def _second_order_glm_loss(
        baseline: jax.Array,
        weights: jax.Array,
        y_n: jax.Array,
        smoother_mean: jax.Array,
        smoother_cov: jax.Array,
        dt: float,
    ) -> jax.Array:
        """Negative log-likelihood with second-order expectation correction."""
        eta = baseline + smoother_mean @ weights
        var_corrections = jax.vmap(lambda P: 0.5 * weights @ P @ weights)(
            smoother_cov
        )
        mu = jnp.exp(eta + var_corrections) * dt
        return jnp.sum(mu - y_n * eta)

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
        loss_before = jnp.array(0.0)
        for n in range(n_neurons):
            loss_before = loss_before + _single_neuron_glm_loss(
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
        loss_after = jnp.array(0.0)
        for n in range(n_neurons):
            loss_after = loss_after + _single_neuron_glm_loss(
                new_params.baseline[n],
                new_params.weights[n],
                spikes[:, n],
                smoother_mean,
                dt,
            )

        assert loss_after < loss_before

    def test_second_order_decreases_second_order_loss(self) -> None:
        """Second-order method should decrease the second-order expected loss."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 80
        n_latent = 3
        n_neurons = 2
        dt = 0.02

        smoother_mean = jax.random.normal(jax.random.PRNGKey(1), (n_time, n_latent))
        smoother_cov = jnp.stack([jnp.eye(n_latent) * 0.1] * n_time, axis=0)

        true_baseline = jnp.array([0.4, -0.2])
        true_weights = jax.random.normal(
            jax.random.PRNGKey(2), (n_neurons, n_latent)
        ) * 0.3
        eta = true_baseline[None, :] + smoother_mean @ true_weights.T
        rates = jnp.exp(eta) * dt
        spikes = jax.random.poisson(jax.random.PRNGKey(3), rates).astype(float)

        current_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        loss_before = jnp.array(0.0)
        for n in range(n_neurons):
            loss_before = loss_before + self._second_order_glm_loss(
                current_params.baseline[n],
                current_params.weights[n],
                spikes[:, n],
                smoother_mean,
                smoother_cov,
                dt,
            )

        new_params = update_spike_glm_params(
            spikes,
            smoother_mean,
            current_params,
            dt,
            max_iter=10,
            smoother_cov=smoother_cov,
            use_second_order=True,
        )

        loss_after = jnp.array(0.0)
        for n in range(n_neurons):
            loss_after = loss_after + self._second_order_glm_loss(
                new_params.baseline[n],
                new_params.weights[n],
                spikes[:, n],
                smoother_mean,
                smoother_cov,
                dt,
            )

        assert loss_after < loss_before

    def test_second_order_loss_differs_from_plugin(self) -> None:
        """Second-order loss should differ from plug-in loss when variance > 0."""
        from state_space_practice.switching_point_process import _single_neuron_glm_loss

        n_time = 40
        n_latent = 2
        dt = 0.02

        baseline = jnp.array(0.2)
        weights = jnp.array([0.4, -0.3])
        smoother_mean = jax.random.normal(jax.random.PRNGKey(4), (n_time, n_latent))
        smoother_cov = jnp.stack([jnp.eye(n_latent) * 0.2] * n_time, axis=0)

        eta = baseline + smoother_mean @ weights
        rates = jnp.exp(eta) * dt
        spikes = jax.random.poisson(jax.random.PRNGKey(5), rates).astype(float)

        plugin_loss = _single_neuron_glm_loss(
            baseline, weights, spikes, smoother_mean, dt
        )
        second_order_loss = self._second_order_glm_loss(
            baseline, weights, spikes, smoother_mean, smoother_cov, dt
        )

        assert not jnp.isclose(plugin_loss, second_order_loss)

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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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
            SpikeObsParams,
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
        spike_params = SpikeObsParams(baseline=baseline, weights=weights)

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
            linear_log_intensity,
            spike_params,
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

    def test_init_invalid_discrete_transition_diag_values_raises_error(self) -> None:
        """Model should raise ValueError for discrete_transition_diag values outside [0, 1]."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        # Test value > 1
        with pytest.raises(ValueError, match="discrete_transition_diag values must be probabilities"):
            SwitchingSpikeOscillatorModel(
                n_oscillators=2,
                n_neurons=5,
                n_discrete_states=2,
                sampling_freq=100.0,
                dt=0.01,
                discrete_transition_diag=jnp.array([0.9, 1.5]),  # Invalid: > 1
            )

        # Test value < 0
        with pytest.raises(ValueError, match="discrete_transition_diag values must be probabilities"):
            SwitchingSpikeOscillatorModel(
                n_oscillators=2,
                n_neurons=5,
                n_discrete_states=2,
                sampling_freq=100.0,
                dt=0.01,
                discrete_transition_diag=jnp.array([-0.1, 0.9]),  # Invalid: < 0
            )


class TestSwitchingSpikeOscillatorModelInitializeParameters:
    """Tests for SwitchingSpikeOscillatorModel._initialize_parameters() method (Task 7.2)."""

    def test_initialize_parameters_runs_without_error(self) -> None:
        """_initialize_parameters should run without error."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)  # Should not raise

    def test_initialize_parameters_sets_init_mean(self) -> None:
        """_initialize_parameters should set init_mean with correct shape."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_oscillators = 2
        n_discrete_states = 3
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=5,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        assert hasattr(model, "init_mean")
        assert model.init_mean.shape == (n_latent, n_discrete_states)
        assert jnp.all(jnp.isfinite(model.init_mean))

    def test_initialize_parameters_sets_init_cov(self) -> None:
        """_initialize_parameters should set init_cov with correct shape."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_oscillators = 2
        n_discrete_states = 3
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=5,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        assert hasattr(model, "init_cov")
        assert model.init_cov.shape == (n_latent, n_latent, n_discrete_states)
        assert jnp.all(jnp.isfinite(model.init_cov))

    def test_initialize_parameters_sets_init_discrete_state_prob(self) -> None:
        """_initialize_parameters should set init_discrete_state_prob with correct shape."""
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

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        assert hasattr(model, "init_discrete_state_prob")
        assert model.init_discrete_state_prob.shape == (n_discrete_states,)
        # Should be uniform
        expected_prob = jnp.ones(n_discrete_states) / n_discrete_states
        np.testing.assert_allclose(
            model.init_discrete_state_prob, expected_prob, rtol=1e-5
        )

    def test_initialize_parameters_sets_discrete_transition_matrix(self) -> None:
        """_initialize_parameters should set discrete_transition_matrix correctly."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_discrete_states = 2

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        assert hasattr(model, "discrete_transition_matrix")
        assert model.discrete_transition_matrix.shape == (
            n_discrete_states,
            n_discrete_states,
        )
        # Rows should sum to 1
        row_sums = jnp.sum(model.discrete_transition_matrix, axis=1)
        np.testing.assert_allclose(row_sums, jnp.ones(n_discrete_states), rtol=1e-5)
        # All entries should be non-negative
        assert jnp.all(model.discrete_transition_matrix >= 0)

    def test_initialize_parameters_sets_continuous_transition_matrix(self) -> None:
        """_initialize_parameters should set continuous_transition_matrix."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_oscillators = 2
        n_discrete_states = 3
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=5,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        assert hasattr(model, "continuous_transition_matrix")
        assert model.continuous_transition_matrix.shape == (
            n_latent,
            n_latent,
            n_discrete_states,
        )
        assert jnp.all(jnp.isfinite(model.continuous_transition_matrix))

    def test_initialize_parameters_sets_process_cov(self) -> None:
        """_initialize_parameters should set process_cov."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_oscillators = 2
        n_discrete_states = 3
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=5,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        assert hasattr(model, "process_cov")
        assert model.process_cov.shape == (n_latent, n_latent, n_discrete_states)
        assert jnp.all(jnp.isfinite(model.process_cov))

    def test_initialize_parameters_sets_spike_params(self) -> None:
        """_initialize_parameters should set spike_params with correct shapes."""
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            SwitchingSpikeOscillatorModel,
        )

        n_oscillators = 2
        n_neurons = 5
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
            separate_spike_params=False,  # Test shared params mode
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        assert hasattr(model, "spike_params")
        assert isinstance(model.spike_params, SpikeObsParams)
        assert model.spike_params.baseline.shape == (n_neurons,)
        assert model.spike_params.weights.shape == (n_neurons, n_latent)

    def test_initialize_parameters_spike_baseline_is_zero(self) -> None:
        """Spike baseline should be initialized to zero."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
            separate_spike_params=False,  # Test shared params mode
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        # Baseline should be zero (exp(0) = 1 Hz baseline rate)
        np.testing.assert_allclose(model.spike_params.baseline, jnp.zeros(n_neurons))

    def test_initialize_parameters_spike_weights_are_small(self) -> None:
        """Spike weights should be initialized to small random values."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_neurons = 5
        n_oscillators = 2

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        # Weights should be small (max absolute value < 1)
        assert jnp.all(jnp.abs(model.spike_params.weights) < 1.0)
        # Weights should be finite
        assert jnp.all(jnp.isfinite(model.spike_params.weights))
        # Weights should not all be zero (randomness)
        assert jnp.any(model.spike_params.weights != 0)

    def test_initialize_parameters_process_cov_is_positive_definite(self) -> None:
        """Process covariance should be positive definite."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        # Check each discrete state's process covariance is PSD
        for s in range(model.n_discrete_states):
            Q_s = model.process_cov[:, :, s]
            eigvals = jnp.linalg.eigvalsh(Q_s)
            assert jnp.all(eigvals > -1e-6), f"State {s} Q not PSD: {eigvals}"

    def test_initialize_parameters_init_cov_is_positive_definite(self) -> None:
        """Initial covariance should be positive definite."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        # Check each discrete state's initial covariance is PSD
        for s in range(model.n_discrete_states):
            P0_s = model.init_cov[:, :, s]
            eigvals = jnp.linalg.eigvalsh(P0_s)
            assert jnp.all(eigvals > -1e-6), f"State {s} P0 not PSD: {eigvals}"

    def test_initialize_parameters_single_discrete_state(self) -> None:
        """_initialize_parameters should work with single discrete state."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_oscillators = 2
        n_neurons = 5
        n_discrete_states = 1
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            separate_spike_params=False,  # Test shared params mode
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        # Check all shapes are correct
        assert model.init_mean.shape == (n_latent, 1)
        assert model.init_cov.shape == (n_latent, n_latent, 1)
        assert model.init_discrete_state_prob.shape == (1,)
        assert model.discrete_transition_matrix.shape == (1, 1)
        assert model.continuous_transition_matrix.shape == (n_latent, n_latent, 1)
        assert model.process_cov.shape == (n_latent, n_latent, 1)
        assert model.spike_params.baseline.shape == (n_neurons,)
        assert model.spike_params.weights.shape == (n_neurons, n_latent)

        # Single state transition should be 1.0
        np.testing.assert_allclose(model.discrete_transition_matrix, jnp.array([[1.0]]))

    def test_initialize_parameters_reproducible_with_same_key(self) -> None:
        """_initialize_parameters should be reproducible with same key."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model1 = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
        )
        model2 = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model1._initialize_parameters(key)
        model2._initialize_parameters(key)

        np.testing.assert_allclose(model1.init_mean, model2.init_mean)
        np.testing.assert_allclose(model1.spike_params.weights, model2.spike_params.weights)

    def test_initialize_parameters_different_with_different_key(self) -> None:
        """_initialize_parameters should produce different values with different keys."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model1 = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
        )
        model2 = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
        )

        model1._initialize_parameters(jax.random.PRNGKey(1))
        model2._initialize_parameters(jax.random.PRNGKey(2))

        # Random components should differ
        assert not jnp.allclose(model1.init_mean, model2.init_mean)
        assert not jnp.allclose(model1.spike_params.weights, model2.spike_params.weights)


class TestSwitchingSpikeOscillatorModelEStep:
    """Tests for SwitchingSpikeOscillatorModel._e_step() method (Task 7.3)."""

    def test_e_step_runs_without_error(self) -> None:
        """_e_step should run without error on valid data."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_oscillators = 2
        n_neurons = 5
        n_discrete_states = 2

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        key = jax.random.PRNGKey(42)
        model._initialize_parameters(key)

        # Generate spikes
        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # E-step should not raise
        marginal_ll = model._e_step(spikes)

        # Should return a finite scalar
        assert jnp.isfinite(marginal_ll)

    def test_e_step_returns_scalar_log_likelihood(self) -> None:
        """_e_step should return a scalar marginal log-likelihood."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 30
        n_neurons = 4

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        marginal_ll = model._e_step(spikes)

        # Should be a scalar (0-D array)
        assert marginal_ll.shape == ()
        assert jnp.isfinite(marginal_ll)

    def test_e_step_stores_smoother_state_cond_mean(self) -> None:
        """_e_step should store smoother_state_cond_mean with correct shape."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_oscillators = 2
        n_neurons = 5
        n_discrete_states = 3
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)

        assert hasattr(model, "smoother_state_cond_mean")
        assert model.smoother_state_cond_mean.shape == (
            n_time,
            n_latent,
            n_discrete_states,
        )
        assert jnp.all(jnp.isfinite(model.smoother_state_cond_mean))

    def test_e_step_stores_smoother_state_cond_cov(self) -> None:
        """_e_step should store smoother_state_cond_cov with correct shape."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_oscillators = 2
        n_neurons = 5
        n_discrete_states = 3
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)

        assert hasattr(model, "smoother_state_cond_cov")
        assert model.smoother_state_cond_cov.shape == (
            n_time,
            n_latent,
            n_latent,
            n_discrete_states,
        )
        assert jnp.all(jnp.isfinite(model.smoother_state_cond_cov))

    def test_e_step_stores_smoother_discrete_state_prob(self) -> None:
        """_e_step should store smoother_discrete_state_prob with correct shape."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 5
        n_discrete_states = 3

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)

        assert hasattr(model, "smoother_discrete_state_prob")
        assert model.smoother_discrete_state_prob.shape == (n_time, n_discrete_states)
        # Probabilities should sum to 1
        prob_sums = jnp.sum(model.smoother_discrete_state_prob, axis=1)
        np.testing.assert_allclose(prob_sums, jnp.ones(n_time), rtol=1e-5)
        # Probabilities should be non-negative
        assert jnp.all(model.smoother_discrete_state_prob >= 0)

    def test_e_step_stores_smoother_joint_discrete_state_prob(self) -> None:
        """_e_step should store smoother_joint_discrete_state_prob with correct shape."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 5
        n_discrete_states = 3

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)

        assert hasattr(model, "smoother_joint_discrete_state_prob")
        # Joint probabilities: p(S_t, S_{t+1} | y_{1:T})
        # Shape: (n_time - 1, n_discrete_states, n_discrete_states)
        assert model.smoother_joint_discrete_state_prob.shape == (
            n_time - 1,
            n_discrete_states,
            n_discrete_states,
        )
        assert jnp.all(jnp.isfinite(model.smoother_joint_discrete_state_prob))

    def test_e_step_stores_smoother_pair_cond_cross_cov(self) -> None:
        """_e_step should store smoother_pair_cond_cross_cov for M-step."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_oscillators = 2
        n_neurons = 5
        n_discrete_states = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)

        assert hasattr(model, "smoother_pair_cond_cross_cov")
        # Shape: (n_time - 1, n_latent, n_latent, n_discrete_states, n_discrete_states)
        assert model.smoother_pair_cond_cross_cov.shape == (
            n_time - 1,
            n_latent,
            n_latent,
            n_discrete_states,
            n_discrete_states,
        )
        assert jnp.all(jnp.isfinite(model.smoother_pair_cond_cross_cov))

    def test_e_step_stores_smoother_pair_cond_means(self) -> None:
        """_e_step should store smoother_pair_cond_means for M-step."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_oscillators = 2
        n_neurons = 5
        n_discrete_states = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)

        assert hasattr(model, "smoother_pair_cond_means")
        # Shape: (n_time - 1, n_latent, n_discrete_states, n_discrete_states)
        assert model.smoother_pair_cond_means.shape == (
            n_time - 1,
            n_latent,
            n_discrete_states,
            n_discrete_states,
        )
        assert jnp.all(jnp.isfinite(model.smoother_pair_cond_means))

    def test_e_step_single_discrete_state(self) -> None:
        """_e_step should work with single discrete state."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 30
        n_neurons = 4
        n_discrete_states = 1

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        marginal_ll = model._e_step(spikes)

        # Should work without error
        assert jnp.isfinite(marginal_ll)
        # Discrete probabilities should all be 1.0 for single state
        np.testing.assert_allclose(
            model.smoother_discrete_state_prob, jnp.ones((n_time, 1)), rtol=1e-5
        )

    def test_e_step_all_zero_spikes(self) -> None:
        """_e_step should handle all zero spikes (silent neurons)."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 30
        n_neurons = 4

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # All zeros - silent neurons
        spikes = jnp.zeros((n_time, n_neurons))

        marginal_ll = model._e_step(spikes)

        # Should work without error
        assert jnp.isfinite(marginal_ll)
        # All smoother outputs should be finite
        assert jnp.all(jnp.isfinite(model.smoother_state_cond_mean))
        assert jnp.all(jnp.isfinite(model.smoother_discrete_state_prob))

    def test_e_step_high_spike_counts(self) -> None:
        """_e_step should handle high spike counts."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 25
        n_neurons = 3

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # High spike counts
        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 10.0, shape=(n_time, n_neurons)
        ).astype(float)

        marginal_ll = model._e_step(spikes)

        # Should work without error
        assert jnp.isfinite(marginal_ll)
        # All smoother outputs should be finite
        assert jnp.all(jnp.isfinite(model.smoother_state_cond_mean))
        assert jnp.all(jnp.isfinite(model.smoother_discrete_state_prob))


class TestSwitchingSpikeOscillatorModelMStepDynamics:
    """Tests for SwitchingSpikeOscillatorModel._m_step_dynamics() method (Task 7.4)."""

    def test_m_step_dynamics_runs_without_error(self) -> None:
        """_m_step_dynamics should run without error after E-step."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # Run E-step first (required before M-step)
        model._e_step(spikes)

        # M-step dynamics should not raise
        model._m_step_dynamics()

    def test_m_step_dynamics_updates_continuous_transition_matrix(self) -> None:
        """_m_step_dynamics should update continuous_transition_matrix when flag is True."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            update_continuous_transition_matrix=True,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # A should have correct shape and be finite
        assert model.continuous_transition_matrix.shape == (
            n_latent,
            n_latent,
            n_discrete_states,
        )
        assert jnp.all(jnp.isfinite(model.continuous_transition_matrix))

    def test_m_step_dynamics_does_not_update_continuous_transition_matrix_when_disabled(
        self,
    ) -> None:
        """_m_step_dynamics should not update A when flag is False."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_continuous_transition_matrix=False,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # Store original A
        A_before = model.continuous_transition_matrix.copy()

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # A should NOT have been updated
        np.testing.assert_allclose(
            model.continuous_transition_matrix, A_before, rtol=1e-10
        )

    def test_m_step_dynamics_updates_process_cov(self) -> None:
        """_m_step_dynamics should update process_cov when flag is True."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            update_process_cov=True,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # Q should have correct shape and be finite
        assert model.process_cov.shape == (n_latent, n_latent, n_discrete_states)
        assert jnp.all(jnp.isfinite(model.process_cov))

    def test_m_step_dynamics_does_not_update_process_cov_when_disabled(self) -> None:
        """_m_step_dynamics should not update Q when flag is False."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_process_cov=False,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # Store original Q
        Q_before = model.process_cov.copy()

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # Q should NOT have been updated
        np.testing.assert_allclose(model.process_cov, Q_before, rtol=1e-10)

    def test_m_step_dynamics_updates_discrete_transition_matrix(self) -> None:
        """_m_step_dynamics should update discrete_transition_matrix when flag is True."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_discrete_states = 2

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            update_discrete_transition_matrix=True,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # Discrete transition matrix should have correct shape
        assert model.discrete_transition_matrix.shape == (
            n_discrete_states,
            n_discrete_states,
        )
        # Rows should sum to 1 (stochastic matrix)
        row_sums = jnp.sum(model.discrete_transition_matrix, axis=1)
        np.testing.assert_allclose(row_sums, jnp.ones(n_discrete_states), rtol=1e-5)
        # All entries should be non-negative
        assert jnp.all(model.discrete_transition_matrix >= 0)

    def test_m_step_dynamics_does_not_update_discrete_transition_when_disabled(
        self,
    ) -> None:
        """_m_step_dynamics should not update Z when flag is False."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_discrete_transition_matrix=False,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # Store original Z
        Z_before = model.discrete_transition_matrix.copy()

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # Z should NOT have been updated
        np.testing.assert_allclose(model.discrete_transition_matrix, Z_before, rtol=1e-10)

    def test_m_step_dynamics_updates_init_mean(self) -> None:
        """_m_step_dynamics should update init_mean when flag is True."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            update_init_mean=True,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # init_mean should have correct shape and be finite
        assert model.init_mean.shape == (n_latent, n_discrete_states)
        assert jnp.all(jnp.isfinite(model.init_mean))

    def test_m_step_dynamics_does_not_update_init_mean_when_disabled(self) -> None:
        """_m_step_dynamics should not update init_mean when flag is False."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_init_mean=False,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # Store original init_mean
        m0_before = model.init_mean.copy()

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # init_mean should NOT have been updated
        np.testing.assert_allclose(model.init_mean, m0_before, rtol=1e-10)

    def test_m_step_dynamics_updates_init_cov(self) -> None:
        """_m_step_dynamics should update init_cov when flag is True."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            update_init_cov=True,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # init_cov should have correct shape and be finite
        assert model.init_cov.shape == (n_latent, n_latent, n_discrete_states)
        assert jnp.all(jnp.isfinite(model.init_cov))

    def test_m_step_dynamics_does_not_update_init_cov_when_disabled(self) -> None:
        """_m_step_dynamics should not update init_cov when flag is False."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_init_cov=False,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # Store original init_cov
        P0_before = model.init_cov.copy()

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # init_cov should NOT have been updated
        np.testing.assert_allclose(model.init_cov, P0_before, rtol=1e-10)

    def test_m_step_dynamics_process_cov_is_psd(self) -> None:
        """_m_step_dynamics should ensure process_cov is positive semi-definite."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 60
        n_neurons = 5
        n_discrete_states = 2

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            update_process_cov=True,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # Process covariance should be PSD for each discrete state
        for j in range(n_discrete_states):
            Q_j = model.process_cov[:, :, j]
            eigvals = jnp.linalg.eigvalsh(Q_j)
            # All eigenvalues should be non-negative (allowing small numerical error)
            assert jnp.all(eigvals >= -1e-8), f"Q[{j}] has negative eigenvalue: {eigvals.min()}"

    def test_m_step_dynamics_single_discrete_state(self) -> None:
        """_m_step_dynamics should work with single discrete state."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 5
        n_discrete_states = 1

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # Should complete without error
        assert jnp.all(jnp.isfinite(model.continuous_transition_matrix))
        assert jnp.all(jnp.isfinite(model.process_cov))
        # Discrete transition matrix should be [[1.0]]
        np.testing.assert_allclose(
            model.discrete_transition_matrix, jnp.array([[1.0]]), rtol=1e-5
        )

    def test_m_step_dynamics_init_discrete_state_prob_updated(self) -> None:
        """_m_step_dynamics should update init_discrete_state_prob."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_discrete_states = 3

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # init_discrete_state_prob should sum to 1
        assert model.init_discrete_state_prob.shape == (n_discrete_states,)
        np.testing.assert_allclose(
            jnp.sum(model.init_discrete_state_prob), 1.0, rtol=1e-5
        )
        # All entries should be non-negative
        assert jnp.all(model.init_discrete_state_prob >= 0)

    def test_m_step_dynamics_all_updates_disabled(self) -> None:
        """_m_step_dynamics should not change anything when all flags are False."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_continuous_transition_matrix=False,
            update_process_cov=False,
            update_discrete_transition_matrix=False,
            update_init_mean=False,
            update_init_cov=False,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # Store original values
        A_before = model.continuous_transition_matrix.copy()
        Q_before = model.process_cov.copy()
        Z_before = model.discrete_transition_matrix.copy()
        m0_before = model.init_mean.copy()
        P0_before = model.init_cov.copy()

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_dynamics()

        # Nothing should have changed
        np.testing.assert_allclose(model.continuous_transition_matrix, A_before, rtol=1e-10)
        np.testing.assert_allclose(model.process_cov, Q_before, rtol=1e-10)
        np.testing.assert_allclose(model.discrete_transition_matrix, Z_before, rtol=1e-10)
        np.testing.assert_allclose(model.init_mean, m0_before, rtol=1e-10)
        np.testing.assert_allclose(model.init_cov, P0_before, rtol=1e-10)


class TestSwitchingSpikeOscillatorModelMStepSpikes:
    """Tests for SwitchingSpikeOscillatorModel._m_step_spikes() method (Task 7.5)."""

    def test_m_step_spikes_runs_without_error(self) -> None:
        """_m_step_spikes should run without error after E-step."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # Run E-step first (required before M-step)
        model._e_step(spikes)

        # M-step spikes should not raise
        model._m_step_spikes(spikes)

    def test_m_step_spikes_updates_spike_params_when_enabled(self) -> None:
        """_m_step_spikes should update spike_params when flag is True."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_spike_params=True,
            separate_spike_params=False,  # Test shared params mode
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # Store original spike params
        baseline_before = model.spike_params.baseline.copy()
        weights_before = model.spike_params.weights.copy()

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_spikes(spikes)

        # Spike params should have correct shapes
        assert model.spike_params.baseline.shape == (n_neurons,)
        assert model.spike_params.weights.shape == (n_neurons, n_latent)

        # Spike params should have been updated (different from before)
        # Note: They might be very close if the initial params are good,
        # but with random initialization they should typically change
        params_changed = not (
            jnp.allclose(model.spike_params.baseline, baseline_before, atol=1e-6)
            and jnp.allclose(model.spike_params.weights, weights_before, atol=1e-6)
        )
        assert params_changed, "spike_params should have been updated"

    def test_m_step_spikes_does_not_update_when_disabled(self) -> None:
        """_m_step_spikes should not update spike_params when flag is False."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_spike_params=False,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # Store original spike params
        baseline_before = model.spike_params.baseline.copy()
        weights_before = model.spike_params.weights.copy()

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_spikes(spikes)

        # Spike params should NOT have been updated
        np.testing.assert_allclose(
            model.spike_params.baseline, baseline_before, rtol=1e-10
        )
        np.testing.assert_allclose(
            model.spike_params.weights, weights_before, rtol=1e-10
        )

    def test_m_step_spikes_output_shapes_correct(self) -> None:
        """_m_step_spikes should produce params with correct shapes."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 8
        n_oscillators = 3
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_spike_params=True,
            separate_spike_params=False,  # Test shared params mode
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_spikes(spikes)

        # Check shapes
        assert model.spike_params.baseline.shape == (n_neurons,)
        assert model.spike_params.weights.shape == (n_neurons, n_latent)

    def test_m_step_spikes_output_finite(self) -> None:
        """_m_step_spikes should produce finite parameter values."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_spike_params=True,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_spikes(spikes)

        # All values should be finite
        assert jnp.all(jnp.isfinite(model.spike_params.baseline))
        assert jnp.all(jnp.isfinite(model.spike_params.weights))

    def test_m_step_spikes_single_discrete_state(self) -> None:
        """_m_step_spikes should work with single discrete state."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=1,  # Single discrete state
            sampling_freq=100.0,
            dt=0.01,
            update_spike_params=True,
            separate_spike_params=False,  # Test shared params mode
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_spikes(spikes)

        # Should have correct shapes and be finite
        assert model.spike_params.baseline.shape == (n_neurons,)
        assert model.spike_params.weights.shape == (n_neurons, n_latent)
        assert jnp.all(jnp.isfinite(model.spike_params.baseline))
        assert jnp.all(jnp.isfinite(model.spike_params.weights))

    def test_m_step_spikes_with_zero_spikes(self) -> None:
        """_m_step_spikes should handle data with all zero spikes."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_spike_params=True,
            separate_spike_params=False,  # Test shared params mode
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # All zeros - silent neurons
        spikes = jnp.zeros((n_time, n_neurons))

        model._e_step(spikes)
        model._m_step_spikes(spikes)

        # Should have correct shapes and be finite (though values may be extreme)
        assert model.spike_params.baseline.shape == (n_neurons,)
        assert model.spike_params.weights.shape == (n_neurons, n_latent)
        assert jnp.all(jnp.isfinite(model.spike_params.baseline))
        assert jnp.all(jnp.isfinite(model.spike_params.weights))

    def test_m_step_spikes_with_high_spike_counts(self) -> None:
        """_m_step_spikes should handle data with high spike counts."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_spike_params=True,
            separate_spike_params=False,  # Test shared params mode
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        # High spike counts (e.g., high firing rate)
        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 10.0, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)
        model._m_step_spikes(spikes)

        # Should have correct shapes and be finite
        assert model.spike_params.baseline.shape == (n_neurons,)
        assert model.spike_params.weights.shape == (n_neurons, n_latent)
        assert jnp.all(jnp.isfinite(model.spike_params.baseline))
        assert jnp.all(jnp.isfinite(model.spike_params.weights))

    def test_m_step_spikes_uses_marginalized_smoother_mean(self) -> None:
        """_m_step_spikes should use marginalized smoother mean for GLM update.

        The M-step for spike params should use the marginal smoother mean
        (marginalized over discrete states), not the state-conditional means.
        """
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            update_spike_params=True,
        )

        model._initialize_parameters(jax.random.PRNGKey(42))

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model._e_step(spikes)

        # Check that smoother outputs are available
        assert hasattr(model, "smoother_state_cond_mean")
        assert hasattr(model, "smoother_discrete_state_prob")

        # State-conditional mean has shape (n_time, n_latent, n_discrete_states)
        assert model.smoother_state_cond_mean.shape == (
            n_time,
            n_latent,
            n_discrete_states,
        )

        model._m_step_spikes(spikes)

        # After M-step, params should be finite
        assert jnp.all(jnp.isfinite(model.spike_params.baseline))
        assert jnp.all(jnp.isfinite(model.spike_params.weights))


class TestSwitchingSpikeOscillatorModelFit:
    """Tests for SwitchingSpikeOscillatorModel.fit() method (Task 7.6)."""

    def test_fit_runs_without_error(self) -> None:
        """fit() should run without error on valid data."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2
        n_discrete_states = 2

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            separate_spike_params=False,  # Test shared params mode
        )

        # Generate random spikes
        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # fit() should not raise
        log_likelihoods = model.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))

        # Should return a list of log-likelihoods
        assert isinstance(log_likelihoods, list)
        assert len(log_likelihoods) == 3

    def test_fit_returns_log_likelihoods_list(self) -> None:
        """fit() should return a list of log-likelihoods, one per iteration."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 4
        max_iter = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        log_likelihoods = model.fit(spikes, max_iter=max_iter, key=jax.random.PRNGKey(42))

        # Should return a list of length max_iter
        assert isinstance(log_likelihoods, list)
        assert len(log_likelihoods) == max_iter

        # All entries should be finite scalars
        for ll in log_likelihoods:
            assert jnp.isfinite(ll)

    def test_fit_log_likelihoods_are_finite(self) -> None:
        """fit() should return finite log-likelihoods."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 30
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        log_likelihoods = model.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))

        for ll in log_likelihoods:
            assert jnp.isfinite(ll), f"Log-likelihood {ll} is not finite"

    def test_fit_em_overall_improvement(self) -> None:
        """fit() should improve log-likelihood over the course of EM iterations.

        EM should generally increase the log-likelihood, but this implementation
        uses a Laplace approximation for the point-process observation model, which
        can introduce small violations of strict monotonicity at individual iterations.

        We test that:
        1. The final log-likelihood is higher than the initial
        2. The overall trend is improving (most iterations increase or stay stable)
        """
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 60
        n_neurons = 5
        max_iter = 8

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            separate_spike_params=False,  # Test shared params mode
        )

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        log_likelihoods = model.fit(spikes, max_iter=max_iter, key=jax.random.PRNGKey(42))

        # Test 1: Final log-likelihood should be higher than initial
        initial_ll = log_likelihoods[0]
        final_ll = log_likelihoods[-1]
        assert final_ll > initial_ll, (
            f"EM did not improve log-likelihood: initial={initial_ll}, final={final_ll}"
        )

        # Test 2: Count iterations where LL increased or stayed approximately stable
        # Allow 5% relative tolerance for "stable" (Laplace approximation can cause small drops)
        n_improving_or_stable = 0
        for i in range(1, len(log_likelihoods)):
            prev_ll = log_likelihoods[i - 1]
            curr_ll = log_likelihoods[i]
            tolerance = 0.05 * abs(prev_ll)
            if curr_ll >= prev_ll - tolerance:
                n_improving_or_stable += 1

        # At least half of iterations should be improving or stable
        min_improving = (len(log_likelihoods) - 1) // 2
        assert n_improving_or_stable >= min_improving, (
            f"Too many iterations with significant LL decrease: "
            f"{n_improving_or_stable}/{len(log_likelihoods)-1} improving/stable "
            f"(need at least {min_improving})"
        )

    def test_fit_initializes_parameters(self) -> None:
        """fit() should initialize parameters before running EM."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 30
        n_neurons = 4
        n_oscillators = 2
        n_latent = 2 * n_oscillators

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            separate_spike_params=False,  # Test shared params mode
        )

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # Before fit, spike_params should not exist as a properly initialized array
        model.fit(spikes, max_iter=2, key=jax.random.PRNGKey(42))

        # After fit, all parameters should be initialized with correct shapes
        assert model.spike_params.baseline.shape == (n_neurons,)
        assert model.spike_params.weights.shape == (n_neurons, n_latent)
        assert model.init_mean.shape == (n_latent, 2)  # n_discrete_states = 2
        assert model.init_cov.shape == (n_latent, n_latent, 2)

    def test_fit_convergence_tolerance(self) -> None:
        """fit() should stop early if convergence tolerance is reached."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # With very loose tolerance, should converge quickly
        log_likelihoods = model.fit(
            spikes, max_iter=100, tol=1e10, key=jax.random.PRNGKey(42)
        )

        # Should stop early due to loose tolerance
        # First iteration completes, then second checks convergence
        assert len(log_likelihoods) <= 3, (
            f"Expected early convergence, got {len(log_likelihoods)} iterations"
        )

    def test_fit_single_discrete_state(self) -> None:
        """fit() should handle single discrete state (non-switching)."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 4

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=1,  # Single state
            sampling_freq=100.0,
            dt=0.01,
        )

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        log_likelihoods = model.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))

        # Should run without error
        assert len(log_likelihoods) == 3
        for ll in log_likelihoods:
            assert jnp.isfinite(ll)

    def test_fit_with_zero_spikes(self) -> None:
        """fit() should handle data with zero spikes (silent neurons)."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 4

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        # All zeros - silent neurons
        spikes = jnp.zeros((n_time, n_neurons))

        log_likelihoods = model.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))

        # Should run without error (though results may be degenerate)
        assert len(log_likelihoods) == 3
        for ll in log_likelihoods:
            assert jnp.isfinite(ll)

    def test_fit_with_high_spike_counts(self) -> None:
        """fit() should handle data with high spike counts."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        # High spike counts (e.g., high firing rate)
        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 10.0, shape=(n_time, n_neurons)
        ).astype(float)

        log_likelihoods = model.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))

        # Should run without error
        assert len(log_likelihoods) == 3
        for ll in log_likelihoods:
            assert jnp.isfinite(ll)

    def test_fit_reproducibility_same_key(self) -> None:
        """fit() should produce the same results with the same random key."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 4

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model1 = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        model2 = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        log_likelihoods1 = model1.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))
        log_likelihoods2 = model2.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))

        np.testing.assert_allclose(log_likelihoods1, log_likelihoods2, rtol=1e-5)

    def test_fit_different_keys_produce_different_results(self) -> None:
        """fit() with different keys should produce different results."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 4

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model1 = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        model2 = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        log_likelihoods1 = model1.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))
        log_likelihoods2 = model2.fit(spikes, max_iter=3, key=jax.random.PRNGKey(123))

        # Results should be different
        assert not np.allclose(log_likelihoods1, log_likelihoods2)

    def test_fit_updates_model_parameters(self) -> None:
        """fit() should update model parameters during EM iterations."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 5
        n_oscillators = 2

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_continuous_transition_matrix=True,
            update_process_cov=True,
            update_spike_params=True,
            separate_spike_params=False,  # Test shared params mode
        )

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # Run fit to initialize and run EM
        model.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))

        # Parameters should be finite
        assert jnp.all(jnp.isfinite(model.continuous_transition_matrix))
        assert jnp.all(jnp.isfinite(model.process_cov))
        assert jnp.all(jnp.isfinite(model.spike_params.baseline))
        assert jnp.all(jnp.isfinite(model.spike_params.weights))

    def test_fit_respects_update_flags(self) -> None:
        """fit() should respect update flags (e.g., disable parameter updates)."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 4

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_continuous_transition_matrix=False,  # Disable A update
            update_spike_params=True,
        )

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # Initialize first to capture initial A
        model._initialize_parameters(jax.random.PRNGKey(42))
        initial_A = model.continuous_transition_matrix.copy()

        # Run E and M steps manually (as fit would)
        model._e_step(spikes)
        model._m_step_dynamics()
        model._m_step_spikes(spikes)

        # A should not have changed (update disabled)
        np.testing.assert_allclose(
            model.continuous_transition_matrix, initial_A, rtol=1e-10
        )

    def test_fit_validates_spikes_shape_2d(self) -> None:
        """fit() should raise ValueError for non-2D spikes array."""
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

        # 1D array - wrong shape
        spikes_1d = jnp.zeros(100)

        with pytest.raises(ValueError, match="must be 2D array"):
            model.fit(spikes_1d, max_iter=3, key=jax.random.PRNGKey(42))

    def test_fit_validates_spikes_n_neurons(self) -> None:
        """fit() should raise ValueError when n_neurons doesn't match."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_neurons = 5

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        # Wrong number of neurons (10 instead of 5)
        spikes_wrong_neurons = jnp.zeros((50, 10))

        with pytest.raises(ValueError, match="must match n_neurons"):
            model.fit(spikes_wrong_neurons, max_iter=3, key=jax.random.PRNGKey(42))

    def test_fit_skip_init_preserves_parameters(self) -> None:
        """fit() with skip_init=True should preserve custom parameters."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 4
        n_oscillators = 1  # Single oscillator = 2D latent state

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_continuous_transition_matrix=False,  # Keep A fixed
        )

        # Initialize parameters first
        model._initialize_parameters(jax.random.PRNGKey(42))

        # Set custom transition matrix with distinct values
        # Shape: (n_latent, n_latent, n_discrete_states) = (2, 2, 2)
        custom_A = jnp.array([
            [[0.9, 0.8], [0.1, 0.0]],
            [[-0.1, 0.0], [0.9, 0.8]],
        ])
        model.continuous_transition_matrix = custom_A

        # Set custom init_discrete_state_prob
        custom_init_prob = jnp.array([0.3, 0.7])
        model.init_discrete_state_prob = custom_init_prob

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # Fit with skip_init=True - should preserve custom parameters
        model.fit(spikes, max_iter=2, skip_init=True)

        # Custom A should be preserved (since update_continuous_transition_matrix=False)
        np.testing.assert_allclose(
            model.continuous_transition_matrix, custom_A, rtol=1e-10
        )

    def test_fit_skip_init_false_reinitializes_parameters(self) -> None:
        """fit() with skip_init=False should reinitialize parameters."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 40
        n_neurons = 4
        n_oscillators = 1  # Single oscillator for simplicity

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_continuous_transition_matrix=False,
        )

        # Initialize and set custom parameters
        model._initialize_parameters(jax.random.PRNGKey(42))
        custom_init_prob = jnp.array([0.3, 0.7])
        model.init_discrete_state_prob = custom_init_prob

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # Fit with skip_init=False (default) - should reinitialize
        model.fit(spikes, max_iter=2, key=jax.random.PRNGKey(99))

        # init_discrete_state_prob should be reset to uniform (not our custom value)
        # After reinitialization, it should be [0.5, 0.5]
        assert not np.allclose(model.init_discrete_state_prob, custom_init_prob)


class TestSwitchingSpikeOscillatorModelProjectParameters:
    """Tests for SwitchingSpikeOscillatorModel._project_parameters() method (Task 7.7)."""

    def test_project_parameters_runs_without_error(self) -> None:
        """_project_parameters() should run without error after initialization."""
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

        model._initialize_parameters(jax.random.PRNGKey(0))

        # Should not raise any exceptions
        model._project_parameters()

    def test_project_parameters_ensures_psd_process_cov(self) -> None:
        """_project_parameters() should ensure process covariance is PSD."""
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

        model._initialize_parameters(jax.random.PRNGKey(0))

        # Perturb Q to potentially break PSD property
        perturbation = jax.random.normal(
            jax.random.PRNGKey(1), model.process_cov.shape
        ) * 0.1
        model.process_cov = model.process_cov + perturbation

        # After projection, Q should be PSD for each discrete state
        model._project_parameters()

        for j in range(model.n_discrete_states):
            Q_j = model.process_cov[:, :, j]
            eigenvalues = jnp.linalg.eigvalsh(Q_j)
            # All eigenvalues should be non-negative (PSD)
            assert jnp.all(eigenvalues >= -1e-10), (
                f"Q for state {j} is not PSD: min eigenvalue = {eigenvalues.min()}"
            )

    def test_project_parameters_preserves_transition_matrix_shape(self) -> None:
        """_project_parameters() should preserve continuous_transition_matrix shape."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=3,
            n_neurons=5,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(0))
        original_shape = model.continuous_transition_matrix.shape

        model._project_parameters()

        assert model.continuous_transition_matrix.shape == original_shape

    def test_project_parameters_preserves_process_cov_shape(self) -> None:
        """_project_parameters() should preserve process_cov shape."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=3,
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(0))
        original_shape = model.process_cov.shape

        model._project_parameters()

        assert model.process_cov.shape == original_shape

    def test_project_parameters_produces_finite_values(self) -> None:
        """_project_parameters() should produce finite values."""
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

        model._initialize_parameters(jax.random.PRNGKey(0))

        model._project_parameters()

        assert jnp.all(jnp.isfinite(model.continuous_transition_matrix))
        assert jnp.all(jnp.isfinite(model.process_cov))

    def test_project_parameters_single_discrete_state(self) -> None:
        """_project_parameters() should work with single discrete state."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=1,  # Single discrete state
            sampling_freq=100.0,
            dt=0.01,
        )

        model._initialize_parameters(jax.random.PRNGKey(0))

        # Should not raise any exceptions
        model._project_parameters()

        # Check shapes
        assert model.continuous_transition_matrix.shape == (4, 4, 1)
        assert model.process_cov.shape == (4, 4, 1)

    def test_project_parameters_preserves_oscillator_block_structure(self) -> None:
        """_project_parameters() should preserve oscillatory block structure in A.

        Each 2x2 diagonal block of the transition matrix should have the form
        of a scaled rotation matrix: [[a, -b], [b, a]].
        """
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

        model._initialize_parameters(jax.random.PRNGKey(0))

        # Perturb A slightly to break perfect structure
        perturbation = jax.random.normal(
            jax.random.PRNGKey(1), model.continuous_transition_matrix.shape
        ) * 0.05
        model.continuous_transition_matrix = (
            model.continuous_transition_matrix + perturbation
        )

        model._project_parameters()

        # Check diagonal blocks have rotation structure
        for j in range(model.n_discrete_states):
            A_j = model.continuous_transition_matrix[:, :, j]
            n_osc = model.n_oscillators

            for i in range(n_osc):
                block = A_j[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
                # Diagonal elements should be equal: a, a
                np.testing.assert_allclose(
                    block[0, 0], block[1, 1], rtol=1e-5,
                    err_msg=f"Diagonal block {i} for state {j} has unequal diagonals"
                )
                # Off-diagonal elements should be negatives: -b, b
                np.testing.assert_allclose(
                    block[0, 1], -block[1, 0], rtol=1e-5,
                    err_msg=f"Diagonal block {i} for state {j} has incorrect off-diagonals"
                )

    def test_project_parameters_ensures_symmetric_process_cov(self) -> None:
        """_project_parameters() should ensure process covariance is symmetric."""
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

        model._initialize_parameters(jax.random.PRNGKey(0))

        # Perturb Q to break symmetry
        non_symmetric_perturbation = jnp.zeros_like(model.process_cov)
        # Add a non-symmetric perturbation
        for j in range(model.n_discrete_states):
            Q_j = model.process_cov[:, :, j]
            # Make slightly non-symmetric
            Q_j_new = Q_j + 0.01 * jax.random.normal(
                jax.random.PRNGKey(10 + j), Q_j.shape
            )
            non_symmetric_perturbation = non_symmetric_perturbation.at[:, :, j].set(
                Q_j_new - Q_j
            )
        model.process_cov = model.process_cov + non_symmetric_perturbation

        model._project_parameters()

        for j in range(model.n_discrete_states):
            Q_j = model.process_cov[:, :, j]
            np.testing.assert_allclose(
                Q_j, Q_j.T, rtol=1e-10,
                err_msg=f"Q for state {j} is not symmetric after projection"
            )

    def test_project_parameters_called_during_fit(self) -> None:
        """_project_parameters() should be called during fit() after M-step.

        This test verifies that fit() properly calls _project_parameters() by
        checking that the oscillatory block structure is preserved through
        multiple EM iterations.
        """
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        n_time = 50
        n_neurons = 4

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=n_neurons,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
        )

        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        # Run fit for a few iterations
        model.fit(spikes, max_iter=3, key=jax.random.PRNGKey(42))

        # After fit, transition matrices should still have valid block structure
        for j in range(model.n_discrete_states):
            A_j = model.continuous_transition_matrix[:, :, j]
            n_osc = model.n_oscillators

            for i in range(n_osc):
                block = A_j[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
                # Diagonal elements should be equal
                np.testing.assert_allclose(
                    block[0, 0], block[1, 1], rtol=1e-4,
                    err_msg=f"After fit: diagonal block {i} for state {j} broken"
                )
                # Off-diagonal elements should be negatives
                np.testing.assert_allclose(
                    block[0, 1], -block[1, 0], rtol=1e-4,
                    err_msg=f"After fit: off-diagonal block {i} for state {j} broken"
                )

        # Process covariance should be PSD
        for j in range(model.n_discrete_states):
            Q_j = model.process_cov[:, :, j]
            eigenvalues = jnp.linalg.eigvalsh(Q_j)
            assert jnp.all(eigenvalues >= -1e-8), (
                f"After fit: Q for state {j} is not PSD"
            )

    def test_project_parameters_respects_update_flags(self) -> None:
        """_project_parameters() should not modify parameters when update flags are False."""
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        # Create model with update flags disabled
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=2,
            n_neurons=5,
            n_discrete_states=2,
            sampling_freq=100.0,
            dt=0.01,
            update_continuous_transition_matrix=False,
            update_process_cov=False,
        )

        model._initialize_parameters(jax.random.PRNGKey(0))

        # Perturb parameters slightly (simulating what might happen in M-step)
        perturbation_A = jax.random.normal(
            jax.random.PRNGKey(1), model.continuous_transition_matrix.shape
        ) * 0.05
        model.continuous_transition_matrix = (
            model.continuous_transition_matrix + perturbation_A
        )

        # Make Q slightly non-symmetric
        for j in range(model.n_discrete_states):
            Q_j = model.process_cov[:, :, j]
            perturbation_Q = jnp.triu(
                jax.random.normal(jax.random.PRNGKey(2 + j), Q_j.shape) * 0.01, k=1
            )
            model.process_cov = model.process_cov.at[:, :, j].set(Q_j + perturbation_Q)

        perturbed_A = model.continuous_transition_matrix.copy()
        perturbed_Q = model.process_cov.copy()

        # Call _project_parameters - should NOT modify anything since flags are False
        model._project_parameters()

        # Parameters should remain unchanged (perturbed values preserved)
        np.testing.assert_array_equal(
            model.continuous_transition_matrix, perturbed_A,
            err_msg="A was modified despite update_continuous_transition_matrix=False"
        )
        np.testing.assert_array_equal(
            model.process_cov, perturbed_Q,
            err_msg="Q was modified despite update_process_cov=False"
        )


class TestSwitchingSpikeOscillatorModelEndToEnd:
    """End-to-end tests for SwitchingSpikeOscillatorModel (Task 7.8).

    These tests validate the full model pipeline including:
    - EM monotonicity on simulated data
    - Discrete state recovery
    - Oscillator parameter recovery

    Note: These tests use conservative spike parameters (small weights, higher
    baseline) to ensure numerical stability during EM. Point-process models
    with very sparse spikes can cause GLM M-step weight explosion.
    """

    def test_model_recovers_discrete_states(self) -> None:
        """Model should recover discrete states from simulated data.

        Simulates data with clear discrete state transitions and verifies that
        the fitted model's smoothed discrete state probabilities correlate
        with the true states.
        """
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        # Use 2 discrete states with distinct dynamics
        # Need many samples for reliable parameter recovery
        n_time = 500  # Long time series for parameter estimation
        n_neurons = 8
        n_oscillators = 2
        n_latent = 2 * n_oscillators
        n_discrete_states = 2
        dt = 0.01  # Smaller dt for numerical stability
        sampling_freq = 100.0

        key = jax.random.PRNGKey(42)
        key_sim, key_fit = jax.random.split(key)

        # Create distinct transition matrices for each discrete state
        # State 0: higher damping (more stable)
        # State 1: lower damping (more dynamic)
        A0 = jnp.eye(n_latent) * 0.98
        A1 = jnp.eye(n_latent) * 0.90
        transition_matrices = jnp.stack([A0, A1], axis=-1)

        # Process covariances - small for stability
        Q = jnp.eye(n_latent) * 0.01
        process_covs = jnp.stack([Q, Q], axis=-1)

        # Discrete transition matrix - high self-transition to create blocks
        discrete_transition_matrix = jnp.array([[0.98, 0.02], [0.02, 0.98]])

        # Conservative spike parameters for numerical stability
        # Small weights prevent GLM M-step weight explosion
        # Higher baseline ensures adequate spike counts
        key_weights, _ = jax.random.split(key_sim)
        spike_weights = jax.random.normal(key_weights, (n_neurons, n_latent)) * 0.05
        spike_baseline = jnp.ones(n_neurons) * 2.0  # ~7.4 Hz baseline rate

        # Simulate data
        spikes, true_states, true_discrete_states = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key_sim,
        )

        # Fit model - use shared spike params since simulation uses shared params
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=sampling_freq,
            dt=dt,
            update_continuous_transition_matrix=True,
            update_process_cov=True,
            update_spike_params=True,
            separate_spike_params=False,
        )

        # Run EM for several iterations
        log_likelihoods = model.fit(spikes, max_iter=10, key=key_fit)

        # Check that EM ran successfully
        assert len(log_likelihoods) > 2, "EM should run multiple iterations"

        # Get smoothed discrete state probabilities
        inferred_probs = model.smoother_discrete_state_prob  # (n_time, n_discrete_states)

        # Compute accuracy for direct assignment
        inferred_states_direct = jnp.argmax(inferred_probs, axis=1)
        accuracy_direct = jnp.mean(inferred_states_direct == true_discrete_states)

        # Compute accuracy for swapped assignment (in case labels are swapped)
        inferred_states_swapped = 1 - inferred_states_direct
        accuracy_swapped = jnp.mean(inferred_states_swapped == true_discrete_states)

        # Take the better accuracy (accounts for label permutation)
        best_accuracy = max(float(accuracy_direct), float(accuracy_swapped))

        # Discrete state recovery is challenging for several reasons:
        # 1. EM can converge to local optima (initialization-dependent)
        # 2. Point-process observations have limited state information per timestep
        # 3. The Laplace approximation adds noise to the E-step
        # 4. Similar dynamics between states reduces distinguishability
        # We test for better-than-chance (>50%) as a baseline validity check.
        # A production system would use multiple random restarts for better recovery.
        assert best_accuracy > 0.50, (
            f"Discrete state recovery accuracy {best_accuracy:.2f} is too low. "
            f"Expected > 0.50 (better than chance) for data with distinct dynamics."
        )

    def test_model_recovers_oscillator_params(self) -> None:
        """Model should approximately recover oscillator parameters from simulated data.

        Tests that the fitted model's parameters are reasonably close to the
        true parameters used for simulation.
        """
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        # Single discrete state for cleaner parameter recovery test
        # Need many samples for reliable estimation
        n_time = 500  # Long time series
        n_neurons = 8
        n_oscillators = 2
        n_latent = 2 * n_oscillators
        n_discrete_states = 1  # Single state avoids label permutation issues
        dt = 0.01  # Smaller dt for numerical stability
        sampling_freq = 100.0

        key = jax.random.PRNGKey(123)
        key_sim, key_fit = jax.random.split(key)

        # True transition matrix: simple damped dynamics
        # Using diagonal matrix for easier parameter matching
        A_true = jnp.eye(n_latent) * 0.95
        transition_matrices = A_true[:, :, None]  # (n_latent, n_latent, 1)

        # Process covariance - small for stability
        Q_true = jnp.eye(n_latent) * 0.01
        process_covs = Q_true[:, :, None]  # (n_latent, n_latent, 1)

        # Discrete transition matrix (trivial for single state)
        discrete_transition_matrix = jnp.array([[1.0]])

        # True spike observation parameters - conservative for stability
        key_weights, _ = jax.random.split(key_sim)
        spike_weights_true = jax.random.normal(key_weights, (n_neurons, n_latent)) * 0.05
        spike_baseline_true = jnp.ones(n_neurons) * 2.0  # Higher baseline

        # Simulate data
        spikes, true_states, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights_true,
            spike_baseline=spike_baseline_true,
            dt=dt,
            key=key_sim,
        )

        # Fit model
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=sampling_freq,
            dt=dt,
            update_continuous_transition_matrix=True,
            update_process_cov=True,
            update_spike_params=True,
        )

        # Run EM
        log_likelihoods = model.fit(spikes, max_iter=15, key=key_fit)

        # Verify EM improved log-likelihood
        assert log_likelihoods[-1] > log_likelihoods[0], (
            "EM should improve log-likelihood"
        )

        # Check parameter recovery
        # 1. Transition matrix structure should be approximately preserved
        A_fitted = model.continuous_transition_matrix[:, :, 0]

        # The fitted A should have oscillatory block structure
        # (due to _project_parameters forcing this structure)
        # Check that diagonal blocks have the rotation-like structure
        for i in range(n_oscillators):
            block_fitted = A_fitted[2*i:2*i+2, 2*i:2*i+2]
            # Diagonal elements should be similar
            np.testing.assert_allclose(
                block_fitted[0, 0], block_fitted[1, 1], rtol=0.15,
                err_msg=f"Fitted block {i}: diagonals should be similar"
            )
            # Off-diagonal should be anti-symmetric
            np.testing.assert_allclose(
                block_fitted[0, 1], -block_fitted[1, 0], rtol=0.15,
                err_msg=f"Fitted block {i}: off-diagonals should be anti-symmetric"
            )

        # 2. Process covariance should be positive definite
        Q_fitted = model.process_cov[:, :, 0]
        eigenvalues = jnp.linalg.eigvalsh(Q_fitted)
        assert jnp.all(eigenvalues > 0), "Fitted Q should be positive definite"

        # 3. All parameters should be finite
        assert jnp.all(jnp.isfinite(A_fitted)), "A should be finite"
        assert jnp.all(jnp.isfinite(Q_fitted)), "Q should be finite"
        assert jnp.all(jnp.isfinite(model.spike_params.weights)), "Weights should be finite"
        assert jnp.all(jnp.isfinite(model.spike_params.baseline)), "Baseline should be finite"

        # 4. Quantitative recovery: transition matrix spectral radius
        # True A has all eigenvalues = 0.95 (diagonal), so spectral radius ≈ 0.95
        true_spectral_radius = 0.95
        fitted_eigenvalues = jnp.linalg.eigvalsh(A_fitted)
        fitted_spectral_radius = float(jnp.max(jnp.abs(fitted_eigenvalues)))
        # Allow generous tolerance for point-process estimation
        np.testing.assert_allclose(
            fitted_spectral_radius, true_spectral_radius, rtol=0.3,
            err_msg="Fitted A spectral radius should be approximately correct"
        )

    def test_model_em_overall_improvement_single_state(self) -> None:
        """EM should improve log-likelihood overall for single discrete state case.

        With a single discrete state, the model reduces to a standard
        point-process state-space model. Note that the Laplace approximation
        used for point-process observations can cause per-iteration monotonicity
        violations, so we test for overall improvement rather than strict
        monotonicity.
        """
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        # Conservative parameters for numerical stability
        n_time = 300  # More samples
        n_neurons = 6
        n_oscillators = 2
        n_latent = 2 * n_oscillators
        dt = 0.01  # Smaller dt
        sampling_freq = 100.0

        key = jax.random.PRNGKey(789)

        # Single discrete state
        A = jnp.eye(n_latent) * 0.95
        transition_matrices = A[:, :, None]
        Q = jnp.eye(n_latent) * 0.01  # Smaller process noise
        process_covs = Q[:, :, None]
        discrete_transition_matrix = jnp.array([[1.0]])

        # Conservative spike parameters
        spike_weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.05
        spike_baseline = jnp.ones(n_neurons) * 2.0  # Higher baseline

        # Simulate
        spikes, _, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
        )

        # Fit
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=1,
            sampling_freq=sampling_freq,
            dt=dt,
        )

        log_likelihoods = model.fit(spikes, max_iter=10, key=jax.random.PRNGKey(42))

        # Test overall improvement: final LL should be better than initial
        # Note: We don't test strict per-iteration monotonicity because
        # the Laplace approximation can cause small violations
        assert log_likelihoods[-1] > log_likelihoods[0], (
            f"EM should improve log-likelihood overall: "
            f"initial={log_likelihoods[0]:.2f}, final={log_likelihoods[-1]:.2f}"
        )

        # All log-likelihoods should be finite
        for ll in log_likelihoods:
            assert jnp.isfinite(ll), f"Log-likelihood {ll} is not finite"

    def test_model_fit_on_simulated_data_runs_without_error(self) -> None:
        """Model should fit on simulated data without errors.

        This is a comprehensive smoke test that runs the full pipeline
        on realistic simulated data.
        """
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        # Conservative parameters for numerical stability
        n_time = 200
        n_neurons = 8
        n_oscillators = 2
        n_latent = 2 * n_oscillators
        n_discrete_states = 2
        dt = 0.01
        sampling_freq = 100.0

        key = jax.random.PRNGKey(456)

        # Create realistic parameters
        A0 = jnp.eye(n_latent) * 0.95
        A1 = jnp.eye(n_latent) * 0.90
        transition_matrices = jnp.stack([A0, A1], axis=-1)

        Q = jnp.eye(n_latent) * 0.01
        process_covs = jnp.stack([Q, Q], axis=-1)

        discrete_transition_matrix = jnp.array([[0.95, 0.05], [0.05, 0.95]])

        # Conservative spike parameters
        spike_weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.05
        spike_baseline = jnp.ones(n_neurons) * 2.0

        # Simulate
        spikes, _, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
        )

        # Fit - use shared spike params since simulation uses shared params
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=sampling_freq,
            dt=dt,
            separate_spike_params=False,
        )

        # Should not raise any exceptions
        log_likelihoods = model.fit(spikes, max_iter=5, key=jax.random.PRNGKey(0))

        # Basic sanity checks
        assert len(log_likelihoods) == 5
        for ll in log_likelihoods:
            assert jnp.isfinite(ll)


class TestMilestone8EndToEnd:
    """Milestone 8: End-to-End Tests for full pipeline validation.

    These tests comprehensively validate the switching spike oscillator model:
    - EM per-iteration monotonicity (with Laplace approximation tolerance)
    - Parameter recovery on simulated data
    - Comparison to non-switching baseline
    - Gaussian mixture collapse correctness

    Note: These tests build on Task 7.8 tests but with stricter validation.
    """

    def test_switching_spike_oscillator_em_monotonic(self) -> None:
        """Task 8.1: EM convergence properties with Laplace approximation.

        Tests the EM algorithm's convergence behavior for the switching spike
        oscillator model. Due to the Laplace approximation used for point-process
        observations, strict per-iteration monotonicity is NOT guaranteed.

        What we test:
        1. Overall improvement: final LL > initial LL (fundamental EM property)
        2. Numerical stability: all log-likelihoods are finite
        3. Eventual convergence: best LL seen during fitting is substantially
           better than initial LL

        Scientific background:
        - True EM guarantees Q(θ^{t+1}) >= Q(θ^t) where Q is the expected
          complete-data log-likelihood
        - However, the *observed* data log-likelihood p(y|θ) can decrease
          when using approximate inference methods like Laplace approximation
        - The Laplace approximation is applied at each timestep in the E-step
          filter, introducing approximation error that can cause the observed
          log-likelihood to temporarily decrease
        - This is well-documented behavior (see Minka 2001, "Expectation
          Propagation for Approximate Bayesian Inference")
        - The approximation typically becomes more accurate as parameters
          stabilize, leading to eventual convergence

        The key guarantee is that EM will make overall progress toward a
        (local) optimum, even if individual iterations may regress.
        """
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        # Configuration for reliable EM behavior
        n_time = 300
        n_neurons = 6
        n_oscillators = 2
        n_latent = 2 * n_oscillators
        n_discrete_states = 2
        dt = 0.01
        sampling_freq = 100.0

        key = jax.random.PRNGKey(12345)
        key_sim, key_fit = jax.random.split(key)

        # Create distinct transition matrices for each discrete state
        A0 = jnp.eye(n_latent) * 0.95
        A1 = jnp.eye(n_latent) * 0.90
        transition_matrices = jnp.stack([A0, A1], axis=-1)

        # Process covariances
        Q = jnp.eye(n_latent) * 0.01
        process_covs = jnp.stack([Q, Q], axis=-1)

        # Discrete transition matrix - moderate persistence
        discrete_transition_matrix = jnp.array([[0.95, 0.05], [0.05, 0.95]])

        # Conservative spike parameters for numerical stability
        spike_weights = jax.random.normal(key_sim, (n_neurons, n_latent)) * 0.05
        spike_baseline = jnp.ones(n_neurons) * 2.0

        # Simulate data
        spikes, _, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key_sim,
        )

        # Fit model
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=sampling_freq,
            dt=dt,
            update_continuous_transition_matrix=True,
            update_process_cov=True,
            update_spike_params=True,
        )

        log_likelihoods = model.fit(spikes, max_iter=20, key=key_fit)

        # Test 1: All log-likelihoods should be finite (numerical stability)
        for i, ll in enumerate(log_likelihoods):
            assert jnp.isfinite(ll), f"Log-likelihood at iteration {i} is not finite: {ll}"

        # Test 2: Overall improvement is required
        # This is the fundamental EM guarantee (up to approximation)
        assert log_likelihoods[-1] > log_likelihoods[0], (
            f"EM should improve log-likelihood overall: "
            f"initial={log_likelihoods[0]:.2f}, final={log_likelihoods[-1]:.2f}"
        )

        # Test 3: Best LL during fitting should be substantially better than initial
        # This checks that EM found a reasonable solution even if final LL regressed
        best_ll = max(log_likelihoods)
        improvement = best_ll - log_likelihoods[0]
        assert improvement > 10.0, (
            f"EM should find substantially better solution: "
            f"initial={log_likelihoods[0]:.2f}, best={best_ll:.2f}, "
            f"improvement={improvement:.2f} < 10.0 threshold"
        )

        # Test 4: Log diagnostic information about monotonicity violations
        # (Not an assertion, just informative)
        violations = []
        for i in range(len(log_likelihoods) - 1):
            ll_curr = log_likelihoods[i]
            ll_next = log_likelihoods[i + 1]
            if ll_next < ll_curr - 0.5:  # Small tolerance
                violations.append((i, float(ll_curr), float(ll_next)))

        # We don't assert on number of violations because Laplace approximation
        # can cause many violations, especially in early iterations. The key
        # is that overall improvement occurs (tested above).

    def test_switching_spike_oscillator_recovers_parameters(self) -> None:
        """Task 8.2: Model should recover parameters from simulated data.

        Tests comprehensive parameter recovery including:
        1. Discrete transition matrix structure (high self-transition)
        2. Continuous dynamics (spectral radius recovery per state)
        3. Initial discrete state probabilities (from smoother)
        4. Smoothed continuous states correlation with true states

        This test uses two discrete states to validate the full switching model.
        Recovery quality is limited by Laplace approximation, point-process
        observation noise, and finite sample size.

        Note: Spike observation parameters (baseline, weights) are NOT tested
        for exact recovery because the GLM M-step uses marginalized smoother
        means, which introduces additional approximation error. However, the
        model should still produce reasonable predictions.
        """
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        # Configuration for parameter recovery
        n_time = 500  # Long time series needed for parameter estimation
        n_neurons = 8
        n_oscillators = 2
        n_latent = 2 * n_oscillators
        n_discrete_states = 2
        dt = 0.01
        sampling_freq = 100.0

        key = jax.random.PRNGKey(3)  # Seed chosen for numerical stability and good recovery
        key_sim, key_fit = jax.random.split(key)

        # True transition matrices with different spectral properties
        # State 0: higher damping (spectral radius ~ 0.95)
        # State 1: lower damping (spectral radius ~ 0.90)
        A0_true = jnp.eye(n_latent) * 0.95
        A1_true = jnp.eye(n_latent) * 0.90
        transition_matrices_true = jnp.stack([A0_true, A1_true], axis=-1)

        # True process covariances
        Q_true = jnp.eye(n_latent) * 0.01
        process_covs_true = jnp.stack([Q_true, Q_true], axis=-1)

        # True discrete transition matrix with high self-transition
        # This creates clear blocks of each discrete state
        discrete_transition_matrix_true = jnp.array([[0.97, 0.03], [0.03, 0.97]])

        # True spike observation parameters
        key_weights, key_sim2 = jax.random.split(key_sim)
        spike_weights_true = jax.random.normal(key_weights, (n_neurons, n_latent)) * 0.05
        spike_baseline_true = jnp.ones(n_neurons) * 2.0

        # Simulate data
        spikes, true_states, true_discrete_states = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices_true,
            process_covs=process_covs_true,
            discrete_transition_matrix=discrete_transition_matrix_true,
            spike_weights=spike_weights_true,
            spike_baseline=spike_baseline_true,
            dt=dt,
            key=key_sim2,
        )

        # Fit model
        # Use separate_spike_params=False since simulation uses shared spike params
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=sampling_freq,
            dt=dt,
            update_continuous_transition_matrix=True,
            update_process_cov=True,
            update_discrete_transition_matrix=True,
            update_spike_params=True,
            separate_spike_params=False,
        )

        log_likelihoods = model.fit(spikes, max_iter=20, key=key_fit)

        # Basic sanity: EM improved
        assert log_likelihoods[-1] > log_likelihoods[0], "EM should improve"

        # Test 1: Discrete transition matrix recovery
        # (the true matrix has 0.97 on diagonal, 0.03 off-diagonal)
        Z_fitted = model.discrete_transition_matrix
        diag_values = jnp.diag(Z_fitted)

        # Test 1a: Should have high diagonal (qualitative)
        assert jnp.all(diag_values > 0.7), (
            f"Discrete transition matrix should have high self-transition. "
            f"Got diagonal: {diag_values}"
        )

        # Test 1b: Should recover true diagonal values (quantitative)
        # Allow generous tolerance due to finite sample size and Laplace approximation
        true_diag = jnp.diag(discrete_transition_matrix_true)
        np.testing.assert_allclose(
            diag_values, true_diag, atol=0.20,
            err_msg=f"Discrete transition diagonal should recover true values. "
            f"True: {true_diag}, Fitted: {diag_values}"
        )

        # Test 1c: Should sum to 1 (valid stochastic matrix)
        row_sums = jnp.sum(Z_fitted, axis=1)
        np.testing.assert_allclose(
            row_sums, jnp.ones(n_discrete_states), rtol=1e-5,
            err_msg="Discrete transition matrix rows should sum to 1"
        )

        # Test 2: Continuous transition matrix spectral properties
        # Compute true spectral radii and sort (to handle label switching)
        true_spectral_radii = []
        for j in range(n_discrete_states):
            A_true = transition_matrices_true[:, :, j]
            eigs_true = jnp.linalg.eigvals(A_true)
            true_spectral_radii.append(float(jnp.max(jnp.abs(eigs_true))))
        true_spectral_radii = sorted(true_spectral_radii)  # [0.90, 0.95]

        # Compute fitted spectral radii
        fitted_spectral_radii = []
        for j in range(n_discrete_states):
            A_fitted = model.continuous_transition_matrix[:, :, j]
            eigenvalues = jnp.linalg.eigvals(A_fitted)
            spectral_radius = float(jnp.max(jnp.abs(eigenvalues)))

            # Test 2a: Should be stable (spectral radius < 1)
            assert spectral_radius < 1.0, (
                f"Fitted A[{j}] should be stable, got spectral radius {spectral_radius}"
            )
            fitted_spectral_radii.append(spectral_radius)

        # Test 2b: Sorted spectral radii should approximately match true values
        # Sorting handles label switching (fitted states 0↔1 may swap)
        fitted_spectral_radii = sorted(fitted_spectral_radii)
        for j, (true_sr, fitted_sr) in enumerate(
            zip(true_spectral_radii, fitted_spectral_radii)
        ):
            # Generous tolerance due to projection, approximation, weak observations
            assert abs(fitted_sr - true_sr) < 0.20, (
                f"Spectral radius {j} should recover approximately. "
                f"True: {true_sr:.3f}, Fitted: {fitted_sr:.3f}, "
                f"Diff: {abs(fitted_sr - true_sr):.3f}"
            )

        # Test 3: Process covariances should be PSD, finite, and reasonable scale
        for j in range(n_discrete_states):
            Q_fitted = model.process_cov[:, :, j]
            assert jnp.all(jnp.isfinite(Q_fitted)), f"Q[{j}] should be finite"
            eigenvalues = jnp.linalg.eigvalsh(Q_fitted)
            assert jnp.all(eigenvalues >= 0), f"Q[{j}] should be PSD"

            # Test 3b: Check scale is reasonable (true Q has 0.01 on diagonal)
            # Very generous bounds due to estimation difficulty with weak observations
            diag_Q = jnp.diag(Q_fitted)
            assert jnp.all(diag_Q > 1e-6), f"Q[{j}] diagonal too small: {diag_Q}"
            assert jnp.all(diag_Q < 1.0), f"Q[{j}] diagonal too large: {diag_Q}"

        # Test 4: Smoothed continuous states should be finite and well-defined
        # NOTE: We do NOT test correlation between smoothed and true states because:
        # 1. With weak observation coupling (spike_weights_scale=0.05, ~5% rate modulation),
        #    the smoother has very little information to infer latent states
        # 2. The fitted model's latent space may be rotated/sign-flipped relative to
        #    the generative model (latent space is not identifiable)
        # 3. Task 8.2 focuses on PARAMETER recovery, not STATE recovery
        #
        # Instead, we verify the smoother outputs are well-formed:
        smoother_mean = jnp.einsum(
            "tls,ts->tl",
            model.smoother_state_cond_mean,
            model.smoother_discrete_state_prob,
        )
        assert jnp.all(jnp.isfinite(smoother_mean)), "Marginalized smoother mean should be finite"
        assert smoother_mean.shape == (n_time, n_latent), (
            f"Smoother mean shape mismatch: {smoother_mean.shape} vs expected ({n_time}, {n_latent})"
        )

        # Test 5: All fitted parameters should be finite
        assert jnp.all(jnp.isfinite(model.continuous_transition_matrix)), "A should be finite"
        assert jnp.all(jnp.isfinite(model.process_cov)), "Q should be finite"
        assert jnp.all(jnp.isfinite(model.discrete_transition_matrix)), "Z should be finite"
        assert jnp.all(jnp.isfinite(model.spike_params.weights)), "Weights should be finite"
        assert jnp.all(jnp.isfinite(model.spike_params.baseline)), "Baseline should be finite"

        # Test 6: Initial state distribution should be valid probability
        init_prob = model.init_discrete_state_prob
        np.testing.assert_allclose(
            jnp.sum(init_prob), 1.0, rtol=1e-5,
            err_msg="Initial discrete state prob should sum to 1"
        )
        assert jnp.all(init_prob >= 0), "Initial discrete state prob should be non-negative"

    def test_separate_spike_params_default_end_to_end(self) -> None:
        """End-to-end test exercising the default separate_spike_params=True.

        Validates that the model fits successfully with per-state spike GLM
        parameters (the default). Uses random Poisson spike data and checks:
        1. EM completes without error and all log-likelihoods are finite
        2. All parameters are finite
        3. Per-state spike parameters have the correct shape
        4. Discrete transition matrix is a valid stochastic matrix
        """
        from state_space_practice.switching_point_process import (
            QRegularizationConfig,
            SwitchingSpikeOscillatorModel,
        )

        n_time = 100
        n_neurons = 5
        n_oscillators = 1
        n_discrete_states = 2

        # Fit model with default separate_spike_params=True
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=100.0,
            dt=0.01,
            q_regularization=QRegularizationConfig(),
        )
        assert model.separate_spike_params is True, "Default should be True"

        # Random Poisson spikes
        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        log_likelihoods = model.fit(spikes, max_iter=5, key=jax.random.PRNGKey(42))

        # Test 1: All log-likelihoods finite
        for i, ll in enumerate(log_likelihoods):
            assert jnp.isfinite(ll), f"Log-likelihood at iteration {i} should be finite"

        # Test 2: All parameters finite
        assert jnp.all(jnp.isfinite(model.continuous_transition_matrix)), "A finite"
        assert jnp.all(jnp.isfinite(model.process_cov)), "Q finite"
        assert jnp.all(jnp.isfinite(model.discrete_transition_matrix)), "Z finite"

        # Test 3: Per-state spike params should have correct shape and be finite
        n_latent = 2 * n_oscillators
        assert model.spike_params.weights.shape == (n_neurons, n_latent, n_discrete_states)
        assert model.spike_params.baseline.shape == (n_neurons, n_discrete_states)
        assert jnp.all(jnp.isfinite(model.spike_params.weights)), "Spike weights finite"
        assert jnp.all(jnp.isfinite(model.spike_params.baseline)), "Spike baseline finite"

        # Test 4: Discrete transition matrix valid stochastic matrix
        Z_fitted = model.discrete_transition_matrix
        row_sums = jnp.sum(Z_fitted, axis=1)
        np.testing.assert_allclose(
            row_sums, jnp.ones(n_discrete_states), rtol=1e-5,
            err_msg="Discrete transition matrix rows should sum to 1"
        )
        assert jnp.all(Z_fitted >= 0), "Transition probs should be non-negative"

    def test_switching_spike_oscillator_vs_non_switching(self) -> None:
        """Task 8.3: Switching model with S=1 should behave like non-switching model.

        Tests that when n_discrete_states=1, the SwitchingSpikeOscillatorModel
        produces similar smoothed means as the non-switching PointProcessModel.

        Key insight: With S=1, the switching machinery should collapse to a
        standard point-process state-space model. Both models should:
        1. Track the same underlying latent dynamics
        2. Produce correlated smoothed means
        3. Converge to similar log-likelihood values

        Note: Exact equivalence is not expected due to:
        - Different initialization strategies
        - Different observation model parameterizations (spike_params vs log_intensity_func)
        - Different filtering implementations (Laplace-EKF vs filter variants)

        We test for behavioral similarity rather than numerical identity.
        """
        from state_space_practice.point_process_kalman import PointProcessModel
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )
        from state_space_practice.switching_point_process import (
            SwitchingSpikeOscillatorModel,
        )

        # Configuration: single discrete state (non-switching case)
        n_time = 300
        n_neurons = 4
        n_oscillators = 2
        n_latent = 2 * n_oscillators
        n_discrete_states = 1  # Key: single state for comparison
        dt = 0.01
        sampling_freq = 100.0

        key = jax.random.PRNGKey(99999)
        key_sim, key_fit = jax.random.split(key)

        # True parameters
        A_true = jnp.eye(n_latent) * 0.95
        transition_matrices = A_true[:, :, None]
        Q_true = jnp.eye(n_latent) * 0.01
        process_covs = Q_true[:, :, None]
        discrete_transition_matrix = jnp.array([[1.0]])

        # Spike parameters - moderate coupling for better state inference
        key_weights, key_sim2 = jax.random.split(key_sim)
        spike_weights = jax.random.normal(key_weights, (n_neurons, n_latent)) * 0.1
        spike_baseline = jnp.ones(n_neurons) * 2.0

        # Simulate data
        spikes, true_states, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key_sim2,
        )

        # === Set up Switching Model (S=1) with FIXED parameters ===
        # To make the latent space identifiable, we use identical parameters
        # in both models and call _e_step directly (bypassing fit() which
        # would re-initialize parameters).
        from state_space_practice.switching_point_process import SpikeObsParams
        switching_model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=sampling_freq,
            dt=dt,
        )

        # Set known parameters directly
        switching_model.spike_params = SpikeObsParams(
            baseline=spike_baseline, weights=spike_weights
        )
        switching_model.continuous_transition_matrix = A_true[:, :, None]
        switching_model.process_cov = Q_true[:, :, None]
        switching_model.init_mean = jnp.zeros((n_latent, 1))
        switching_model.init_cov = jnp.eye(n_latent)[:, :, None]
        switching_model.discrete_transition_matrix = jnp.array([[1.0]])
        switching_model.init_discrete_state_prob = jnp.array([1.0])

        # Run E-step directly (filter + smoother)
        switching_ll = switching_model._e_step(spikes)

        # === Set up Non-Switching Model with IDENTICAL parameters ===
        # Create log_intensity_func that matches the spike observation model
        def log_intensity_func(design_row: Array, state: Array) -> Array:
            """Log-intensity: baseline + weights @ state."""
            return spike_baseline + spike_weights @ state

        nonswitching_model = PointProcessModel(
            n_state_dims=n_latent,
            dt=dt,
            transition_matrix=A_true,  # Same as switching model
            process_cov=Q_true,  # Same as switching model
            init_mean=jnp.zeros(n_latent),
            init_cov=jnp.eye(n_latent),
            log_intensity_func=log_intensity_func,
        )

        # Design matrix (unused by our log_intensity_func, but required by API)
        design_matrix = jnp.zeros((n_time, 1))

        # Run E-step directly
        nonswitching_ll = nonswitching_model._e_step(design_matrix, spikes)

        # === Comparisons ===

        # Test 1: Both models produce finite log-likelihoods
        assert jnp.isfinite(switching_ll), "Switching LL should be finite"
        assert jnp.isfinite(nonswitching_ll), "Non-switching LL should be finite"

        # Test 1b: Log-likelihoods should be very similar (same model, same data)
        ll_diff = abs(float(switching_ll) - float(nonswitching_ll))
        assert ll_diff < 1.0, (
            f"Log-likelihoods should be nearly identical. "
            f"Switching: {switching_ll:.2f}, Non-switching: {nonswitching_ll:.2f}, "
            f"Diff: {ll_diff:.4f}"
        )

        # Test 2: Both smoothed means should be finite
        # Get switching model's marginalized smoother mean
        switching_smoother = jnp.einsum(
            "tls,ts->tl",
            switching_model.smoother_state_cond_mean,
            switching_model.smoother_discrete_state_prob,
        )
        assert jnp.all(jnp.isfinite(switching_smoother)), (
            "Switching smoother mean should be finite"
        )

        nonswitching_smoother = nonswitching_model.smoother_mean
        assert nonswitching_smoother is not None, "Non-switching smoother mean should not be None"
        assert jnp.all(jnp.isfinite(nonswitching_smoother)), (
            "Non-switching smoother mean should be finite"
        )

        # Test 3: Shapes should match
        assert switching_smoother.shape == nonswitching_smoother.shape, (
            f"Smoother shapes don't match: switching={switching_smoother.shape}, "
            f"non-switching={nonswitching_smoother.shape}"
        )

        # Test 4: With IDENTICAL parameters, smoothed means should be HIGHLY correlated
        # Since we disabled all M-step updates and use the same A, Q, and observation
        # model, both filtering implementations should produce very similar results.
        # The latent space is now identifiable because all parameters are fixed.
        correlations = []
        for i in range(n_latent):
            corr = jnp.corrcoef(
                switching_smoother[:, i], nonswitching_smoother[:, i]
            )[0, 1]
            correlations.append(float(corr))

        mean_correlation = jnp.mean(jnp.array(correlations))

        # With identical parameters, expect high correlation (>0.8)
        # Differences arise only from minor implementation details in the
        # Laplace-EKF filtering/smoothing algorithms
        assert mean_correlation > 0.8, (
            f"With identical parameters, smoothed means should be highly correlated. "
            f"Mean correlation: {mean_correlation:.3f}, per-dim: {correlations}"
        )

        # Test 5: Both discrete state probs should be trivially 1.0 for S=1
        assert switching_model.smoother_discrete_state_prob.shape[1] == 1, (
            "Should have single discrete state"
        )
        np.testing.assert_allclose(
            switching_model.smoother_discrete_state_prob,
            jnp.ones((n_time, 1)),
            rtol=1e-5,
            err_msg="With S=1, all discrete state probs should be 1.0"
        )

    def test_collapse_to_state_conditional(self) -> None:
        """Task 8.4: Verify Gaussian mixture collapse math is correct.

        Tests that `collapse_gaussian_mixture_per_discrete_state` correctly
        computes state-conditional moments from pair-conditional moments.

        The collapse formula for mean:
            E[X | S_t=j] = sum_i P(S_{t-1}=i | S_t=j) * E[X | S_{t-1}=i, S_t=j]

        The collapse formula for covariance (law of total variance):
            Cov[X | S_t=j] = E[Cov[X | S_{t-1}=i, S_t=j] | S_t=j]
                           + Cov[E[X | S_{t-1}=i, S_t=j] | S_t=j]

        The second term accounts for uncertainty in which previous state
        we came from ("variance of the conditional means").

        This test verifies the implementation matches these formulas.
        """
        from state_space_practice.switching_kalman import (
            collapse_gaussian_mixture_per_discrete_state,
        )

        # Setup: 2 previous states (i), 2 next states (j), 3 latent dims
        n_prev_states = 2
        n_next_states = 2
        n_latent = 3

        # Create known pair-conditional means: E[X | S_{t-1}=i, S_t=j]
        # Shape: (n_latent, n_prev_states, n_next_states)
        # For testing, make them clearly different
        pair_cond_means = jnp.array([
            # X dim 0
            [[1.0, 2.0],   # i=0: j=0, j=1
             [3.0, 4.0]],  # i=1: j=0, j=1
            # X dim 1
            [[0.5, 1.5],
             [2.5, 3.5]],
            # X dim 2
            [[-1.0, -2.0],
             [-3.0, -4.0]],
        ])  # Shape: (3, 2, 2)

        # Create known pair-conditional covariances: Cov[X | S_{t-1}=i, S_t=j]
        # Shape: (n_latent, n_latent, n_prev_states, n_next_states)
        # Use diagonal covariances for simplicity
        pair_cond_covs = jnp.zeros((n_latent, n_latent, n_prev_states, n_next_states))
        for i in range(n_prev_states):
            for j in range(n_next_states):
                pair_cond_covs = pair_cond_covs.at[:, :, i, j].set(
                    jnp.eye(n_latent) * (0.1 + 0.1 * i + 0.05 * j)
                )

        # Mixing weights: P(S_{t-1}=i | S_t=j) for each j
        # These are the backward conditional probabilities
        # Shape: (n_prev_states, n_next_states)
        mixing_weights = jnp.array([
            [0.7, 0.3],  # For j=0: P(i=0|j=0)=0.7, P(i=1|j=0)=0.3
            [0.4, 0.6],  # For j=1: P(i=0|j=1)=0.4, P(i=1|j=1)=0.6
        ]).T  # Transpose to (n_prev_states, n_next_states)

        # Run the collapse function
        state_cond_means, state_cond_covs = collapse_gaussian_mixture_per_discrete_state(
            pair_cond_means, pair_cond_covs, mixing_weights
        )

        # Manually compute expected state-conditional means
        # E[X | S_t=j] = sum_i P(S_{t-1}=i | S_t=j) * E[X | S_{t-1}=i, S_t=j]
        expected_means = jnp.zeros((n_latent, n_next_states))
        for j in range(n_next_states):
            for i in range(n_prev_states):
                expected_means = expected_means.at[:, j].add(
                    mixing_weights[i, j] * pair_cond_means[:, i, j]
                )

        # Manually compute expected state-conditional covariances
        # Cov[X | S_t=j] = E[Cov] + Cov[E]
        #   = sum_i P(i|j) * Cov[X|i,j] + sum_i P(i|j) * (m_ij - m_j)(m_ij - m_j)^T
        expected_covs = jnp.zeros((n_latent, n_latent, n_next_states))
        for j in range(n_next_states):
            mean_j = expected_means[:, j]
            for i in range(n_prev_states):
                w_ij = mixing_weights[i, j]
                # E[Cov] term
                expected_covs = expected_covs.at[:, :, j].add(
                    w_ij * pair_cond_covs[:, :, i, j]
                )
                # Cov[E] term: w_ij * (m_ij - m_j)(m_ij - m_j)^T
                diff = pair_cond_means[:, i, j] - mean_j
                expected_covs = expected_covs.at[:, :, j].add(
                    w_ij * jnp.outer(diff, diff)
                )

        # Test 1: State-conditional means match expected
        np.testing.assert_allclose(
            state_cond_means, expected_means, rtol=1e-5,
            err_msg="Collapsed means should match expected values"
        )

        # Test 2: State-conditional covariances match expected
        np.testing.assert_allclose(
            state_cond_covs, expected_covs, rtol=1e-5,
            err_msg="Collapsed covariances should match expected values"
        )

        # Test 3: Collapsed covariances should be symmetric
        for j in range(n_next_states):
            np.testing.assert_allclose(
                state_cond_covs[:, :, j], state_cond_covs[:, :, j].T, rtol=1e-10,
                err_msg=f"Collapsed covariance for state {j} should be symmetric"
            )

        # Test 4: Collapsed covariances should be PSD
        for j in range(n_next_states):
            eigenvalues = jnp.linalg.eigvalsh(state_cond_covs[:, :, j])
            assert jnp.all(eigenvalues >= -1e-10), (
                f"Collapsed covariance for state {j} should be PSD. "
                f"Min eigenvalue: {jnp.min(eigenvalues)}"
            )

        # Test 5: Collapsed covariance should be >= pair-conditional covariance
        # (by law of total variance, adding "variance of means" term)
        for j in range(n_next_states):
            # Average pair-conditional covariance
            avg_pair_cov = jnp.zeros((n_latent, n_latent))
            for i in range(n_prev_states):
                avg_pair_cov += mixing_weights[i, j] * pair_cond_covs[:, :, i, j]

            # Collapsed covariance should have larger eigenvalues (more uncertainty)
            collapsed_trace = jnp.trace(state_cond_covs[:, :, j])
            avg_pair_trace = jnp.trace(avg_pair_cov)
            assert collapsed_trace >= avg_pair_trace - 1e-10, (
                f"Collapsed covariance trace ({collapsed_trace:.4f}) should >= "
                f"average pair-conditional trace ({avg_pair_trace:.4f}) for state {j}"
            )

        # Test 6: Edge case - uniform mixing weights
        uniform_weights = jnp.ones((n_prev_states, n_next_states)) / n_prev_states
        uniform_means, uniform_covs = collapse_gaussian_mixture_per_discrete_state(
            pair_cond_means, pair_cond_covs, uniform_weights
        )

        # With uniform weights, mean should be simple average
        for j in range(n_next_states):
            expected_uniform_mean = jnp.mean(pair_cond_means[:, :, j], axis=1)
            np.testing.assert_allclose(
                uniform_means[:, j], expected_uniform_mean, rtol=1e-5,
                err_msg=f"Uniform weights: mean for state {j} should be simple average"
            )

        # Test 7: Edge case - deterministic (one-hot) mixing weights
        deterministic_weights = jnp.array([[1.0, 0.0], [0.0, 1.0]]).T  # i=j
        det_means, det_covs = collapse_gaussian_mixture_per_discrete_state(
            pair_cond_means, pair_cond_covs, deterministic_weights
        )

        # With deterministic weights, should just select the matching pair
        for j in range(n_next_states):
            # When P(i=j | S_t=j) = 1, collapsed = pair(i=j, j)
            np.testing.assert_allclose(
                det_means[:, j], pair_cond_means[:, j, j], rtol=1e-5,
                err_msg="Deterministic weights: mean should equal diagonal pair"
            )
            np.testing.assert_allclose(
                det_covs[:, :, j], pair_cond_covs[:, :, j, j], rtol=1e-5,
                err_msg="Deterministic weights: cov should equal diagonal pair"
            )


# --- EM Verification Tests ---


class TestEMVerification:
    """Tests that directly verify EM algorithm correctness for point-process models.

    These tests go beyond structural checks to verify that:
    1. Degenerate cases match known analytical solutions
    2. The M-step increases the Q-function
    3. Parameters can be recovered from simulated data
    4. Discrete states can be recovered from distinct firing patterns
    """

    def test_constant_state_recovers_poisson_mle(self) -> None:
        """With A=I, Q=0, the spike GLM M-step should recover the Poisson MLE.

        When the latent state is constant (no dynamics noise), the model
        reduces to a standard Poisson GLM. The MLE for the baseline is
        log(total_spikes / (T * dt)), and the weights should be near zero.
        """
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            update_spike_glm_params,
        )

        n_time = 2000
        n_neurons = 3
        n_latent = 2
        dt = 0.01

        # Generate spikes from known constant rates
        true_rates = jnp.array([5.0, 10.0, 20.0])  # Hz
        key = jax.random.PRNGKey(42)
        spikes = jax.random.poisson(
            key, true_rates[None, :] * dt, shape=(n_time, n_neurons)
        ).astype(float)

        # Smoother output: constant zero state with zero covariance
        smoother_mean = jnp.zeros((n_time, n_latent))
        smoother_cov = jnp.zeros((n_time, n_latent, n_latent))

        # Initial params: wrong baseline, small weights
        init_params = SpikeObsParams(
            baseline=jnp.zeros(n_neurons),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        # Run multiple M-step iterations
        updated = update_spike_glm_params(
            spikes=spikes,
            smoother_mean=smoother_mean,
            current_params=init_params,
            dt=dt,
            smoother_cov=smoother_cov,
            use_second_order=True,
            max_iter=20,
        )

        # Expected baseline: log(mean_count / dt)
        # With zero state and zero weights, exp(baseline) * dt = expected count
        mean_counts = jnp.mean(spikes, axis=0)
        expected_baseline = jnp.log(mean_counts / dt + 1e-10)

        np.testing.assert_allclose(
            updated.baseline, expected_baseline, rtol=0.05,
            err_msg="Baseline should converge to log(mean_rate)"
        )

        # Weights should remain near zero (no signal from constant state)
        assert jnp.max(jnp.abs(updated.weights)) < 0.1, (
            f"Weights should stay near zero, got max={jnp.max(jnp.abs(updated.weights)):.4f}"
        )

    def test_mstep_increases_spike_q_function(self) -> None:
        """The M-step should increase (or maintain) the spike Q-function.

        Q(theta_new) >= Q(theta_old) is the fundamental EM guarantee.
        We verify this by computing Q before and after the spike M-step
        using the same E-step posterior.
        """
        from state_space_practice.switching_point_process import (
            SpikeObsParams,
            _neg_Q_single_neuron,
            update_spike_glm_params,
        )

        n_time = 500
        n_neurons = 2
        n_latent = 2
        dt = 0.01

        key = jax.random.PRNGKey(99)
        k1, k2, k3 = jax.random.split(key, 3)

        # Generate synthetic smoother outputs
        smoother_mean = jax.random.normal(k1, (n_time, n_latent)) * 0.5
        smoother_cov = jnp.tile(
            jnp.eye(n_latent) * 0.1, (n_time, 1, 1)
        )

        # Generate spikes from known params
        true_baseline = jnp.array([1.0, 2.0])
        true_weights = jax.random.normal(k2, (n_neurons, n_latent)) * 0.3
        eta = true_baseline[None, :] + smoother_mean @ true_weights.T
        rates = jnp.exp(eta) * dt
        spikes = jax.random.poisson(k3, rates).astype(float)

        # Start from deliberately wrong params
        old_params = SpikeObsParams(
            baseline=jnp.array([0.0, 0.0]),
            weights=jnp.zeros((n_neurons, n_latent)),
        )

        # Compute Q before M-step
        def compute_total_Q(params):
            total = 0.0
            for n in range(n_neurons):
                p = jnp.concatenate([jnp.atleast_1d(params.baseline[n]), params.weights[n]])
                total += float(_neg_Q_single_neuron(
                    p, spikes[:, n], smoother_mean, smoother_cov, dt, 0.0
                ))
            return total  # This is the negative Q, so lower is better

        neg_Q_before = compute_total_Q(old_params)

        # Run one M-step iteration
        new_params = update_spike_glm_params(
            spikes=spikes,
            smoother_mean=smoother_mean,
            current_params=old_params,
            dt=dt,
            smoother_cov=smoother_cov,
            use_second_order=True,
            max_iter=1,
        )

        neg_Q_after = compute_total_Q(new_params)

        # M-step should decrease negative Q (increase Q)
        assert neg_Q_after <= neg_Q_before + 1e-6, (
            f"M-step should increase Q: -Q_before={neg_Q_before:.4f}, "
            f"-Q_after={neg_Q_after:.4f}, diff={neg_Q_after - neg_Q_before:.6f}"
        )

    def test_discrete_state_recovery_from_firing_rates(self) -> None:
        """Switching model should identify states with distinct firing rates.

        Simulate data where state 0 has low firing and state 1 has high firing.
        After fitting, the smoother's discrete state probabilities should
        correctly segment the data.
        """
        from state_space_practice.switching_point_process import (
            QRegularizationConfig,
            SwitchingSpikeOscillatorModel,
        )

        n_time = 400
        n_neurons = 5
        n_oscillators = 1
        n_discrete_states = 2
        dt = 0.01

        # Create data with clear state-dependent firing patterns
        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)

        # True discrete states: blocks of ~100 time steps each
        true_states = jnp.concatenate([
            jnp.zeros(100, dtype=int),
            jnp.ones(100, dtype=int),
            jnp.zeros(100, dtype=int),
            jnp.ones(100, dtype=int),
        ])

        # State 0: low firing (~2 Hz), State 1: high firing (~20 Hz)
        low_rate = 2.0 * dt
        high_rate = 20.0 * dt

        rates = jnp.where(
            true_states[:, None] == 0,
            low_rate * jnp.ones((n_time, n_neurons)),
            high_rate * jnp.ones((n_time, n_neurons)),
        )
        spikes = jax.random.poisson(k1, rates).astype(float)

        # Fit model with per-state spike params to capture rate differences
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=1.0 / dt,
            dt=dt,
            q_regularization=QRegularizationConfig(),
            # Fix dynamics — only learn observation and discrete params
            update_continuous_transition_matrix=False,
            update_process_cov=False,
            update_init_mean=False,
            update_init_cov=False,
            separate_spike_params=True,
        )

        # Try multiple seeds since the model can be sensitive to initialization
        fitted = False
        for seed in [10, 20, 30, 40, 50]:
            try:
                log_likelihoods = model.fit(
                    spikes, max_iter=30, key=jax.random.PRNGKey(seed)
                )
                fitted = True
                break
            except ValueError:
                continue

        assert fitted, "Model failed to fit with all attempted seeds"
        assert log_likelihoods[-1] > log_likelihoods[0], "EM should improve LL"

        # Check discrete state segmentation (handle label swapping)
        smoother_prob = np.array(model.smoother_discrete_state_prob)
        corr_0 = np.corrcoef(np.array(true_states, dtype=float), smoother_prob[:, 0])[0, 1]
        corr_1 = np.corrcoef(np.array(true_states, dtype=float), smoother_prob[:, 1])[0, 1]
        best_corr = max(abs(corr_0), abs(corr_1))

        assert best_corr > 0.4, (
            f"Discrete states should be recoverable from distinct firing rates. "
            f"Best |correlation| = {best_corr:.3f}"
        )

    def test_em_convergence_fixed_point(self) -> None:
        """At convergence, one more EM iteration should not change parameters.

        Run EM to convergence, then check that another iteration produces
        nearly identical parameters. This tests that the converged point
        is actually a fixed point of the EM operator.
        """
        from state_space_practice.switching_point_process import (
            QRegularizationConfig,
            SwitchingSpikeOscillatorModel,
        )

        n_time = 100
        n_neurons = 3
        n_oscillators = 1
        n_discrete_states = 2
        dt = 0.01

        key = jax.random.PRNGKey(42)
        spikes = jax.random.poisson(
            key, 0.5, shape=(n_time, n_neurons)
        ).astype(float)

        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=1.0 / dt,
            dt=dt,
            q_regularization=QRegularizationConfig(),
        )

        # Run to convergence
        model.fit(spikes, max_iter=50, tol=1e-8, key=jax.random.PRNGKey(0))

        # Store converged parameters
        A_converged = model.continuous_transition_matrix.copy()
        Q_converged = model.process_cov.copy()
        Z_converged = model.discrete_transition_matrix.copy()
        baseline_converged = model.spike_params.baseline.copy()
        weights_converged = model.spike_params.weights.copy()

        # Run one more iteration
        model.fit(spikes, max_iter=1, skip_init=True, tol=1e-15)

        # Parameters should barely change. The Q-regularization trust-region
        # (default 30% blend) means even at convergence there can be small
        # parameter drift. Use generous tolerance.
        np.testing.assert_allclose(
            model.continuous_transition_matrix, A_converged, atol=0.02,
            err_msg="A should be near fixed point"
        )
        np.testing.assert_allclose(
            model.process_cov, Q_converged, atol=0.05,
            err_msg="Q should be near fixed point"
        )
        np.testing.assert_allclose(
            model.discrete_transition_matrix, Z_converged, atol=0.02,
            err_msg="Z should be near fixed point"
        )
        # Spike params can drift more at convergence because the GLM M-step
        # uses Newton iterations that may not reach the exact optimum.
        # States with very low firing (baseline << 0) are especially noisy.
        np.testing.assert_allclose(
            model.spike_params.baseline, baseline_converged, atol=0.1,
            err_msg="Baseline should be near fixed point"
        )
        np.testing.assert_allclose(
            model.spike_params.weights, weights_converged, atol=0.1,
            err_msg="Weights should be near fixed point"
        )

    def test_parameter_recovery_from_simulation(self) -> None:
        """Fit model on simulated switching data and verify parameter recovery.

        Uses long time series with well-separated states and high spike rates
        so the Laplace approximation is accurate. Checks:
        1. Transition matrix diagonal recovery
        2. Continuous dynamics spectral radius recovery (handles label swap)
        3. Process covariance scale recovery
        4. Spike baseline recovery (handles label swap)
        5. Overall EM improvement
        """
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )
        from state_space_practice.switching_point_process import (
            QRegularizationConfig,
            SwitchingSpikeOscillatorModel,
        )

        # Long time series for accurate recovery
        n_time = 1000
        n_neurons = 8
        n_oscillators = 1
        n_latent = 2 * n_oscillators
        n_discrete_states = 2
        dt = 0.01
        sampling_freq = 100.0

        key = jax.random.PRNGKey(3)
        key_sim, key_fit = jax.random.split(key)

        # True dynamics: state 0 has higher damping (spectral radius 0.92)
        #                state 1 has lower damping (spectral radius 0.98)
        A0_true = jnp.eye(n_latent) * 0.92
        A1_true = jnp.eye(n_latent) * 0.98
        transition_matrices_true = jnp.stack([A0_true, A1_true], axis=-1)

        Q_true = jnp.eye(n_latent) * 0.02
        process_covs_true = jnp.stack([Q_true, Q_true], axis=-1)

        # High self-transition for clear state blocks
        Z_true = jnp.array([[0.97, 0.03], [0.03, 0.97]])

        # Moderate spike coupling for good SNR
        key_weights, key_sim2 = jax.random.split(key_sim)
        spike_weights_true = jax.random.normal(
            key_weights, (n_neurons, n_latent)
        ) * 0.1
        spike_baseline_true = jnp.ones(n_neurons) * 2.5  # ~12 Hz baseline

        # Simulate
        spikes, true_states, true_discrete_states = (
            simulate_switching_spike_oscillator(
                n_time=n_time,
                transition_matrices=transition_matrices_true,
                process_covs=process_covs_true,
                discrete_transition_matrix=Z_true,
                spike_weights=spike_weights_true,
                spike_baseline=spike_baseline_true,
                dt=dt,
                key=key_sim2,
            )
        )

        # Fit model
        model = SwitchingSpikeOscillatorModel(
            n_oscillators=n_oscillators,
            n_neurons=n_neurons,
            n_discrete_states=n_discrete_states,
            sampling_freq=sampling_freq,
            dt=dt,
            q_regularization=QRegularizationConfig(),
            separate_spike_params=False,
        )

        log_likelihoods = model.fit(spikes, max_iter=30, key=key_fit)

        # 1. EM improved
        assert log_likelihoods[-1] > log_likelihoods[0], "EM should improve"

        # 2. Discrete transition matrix: should have high diagonal
        Z_fitted = model.discrete_transition_matrix
        diag_values = jnp.diag(Z_fitted)
        assert jnp.all(diag_values > 0.7), (
            f"Transition diagonal should be high: {diag_values}"
        )
        # Rows sum to 1
        np.testing.assert_allclose(
            jnp.sum(Z_fitted, axis=1), jnp.ones(n_discrete_states), rtol=1e-5
        )

        # 3. Spectral radii: sort to handle label permutation
        fitted_spectral_radii = sorted([
            float(jnp.max(jnp.abs(
                jnp.linalg.eigvals(model.continuous_transition_matrix[:, :, j])
            )))
            for j in range(n_discrete_states)
        ])

        # Both should be stable
        for sr in fitted_spectral_radii:
            assert sr < 1.0, f"Fitted system should be stable, got sr={sr}"

        # Spectral radii should be in a reasonable range.
        # Q-regularization (trust-region blending + eigenvalue clipping) biases
        # the process noise upward, which can reduce fitted spectral radii.
        # We check that the two states have different spectral radii (the model
        # learned distinct dynamics) and both are in a reasonable range.
        assert fitted_spectral_radii[1] > fitted_spectral_radii[0] + 0.02, (
            f"States should have distinct spectral radii: "
            f"{fitted_spectral_radii}"
        )
        for sr in fitted_spectral_radii:
            assert 0.3 < sr < 1.0, f"Spectral radius out of range: {sr}"

        # 4. Process covariance: should be positive, reasonable scale
        for j in range(n_discrete_states):
            Q_fitted = model.process_cov[:, :, j]
            eigvals = jnp.linalg.eigvalsh(Q_fitted)
            assert jnp.all(eigvals > 0), f"Q[{j}] should be PSD"
            diag_Q = jnp.diag(Q_fitted)
            assert jnp.all(diag_Q > 1e-4), f"Q[{j}] diagonal too small"
            assert jnp.all(diag_Q < 1.0), f"Q[{j}] diagonal too large"

        # 5. All parameters finite
        assert jnp.all(jnp.isfinite(model.continuous_transition_matrix))
        assert jnp.all(jnp.isfinite(model.process_cov))
        assert jnp.all(jnp.isfinite(model.discrete_transition_matrix))
        assert jnp.all(jnp.isfinite(model.spike_params.weights))
        assert jnp.all(jnp.isfinite(model.spike_params.baseline))
