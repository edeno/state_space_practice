"""Tests for the switching_point_process module.

This module tests the switching point-process Kalman filter and smoother
for spike-based observations with discrete state switching.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
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
