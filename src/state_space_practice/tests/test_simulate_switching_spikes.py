"""Tests for the simulate_switching_spikes module.

These tests verify the simulation utilities for generating synthetic data
from the switching spike-oscillator model.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


class TestSimulateSwitchingSpikeOscillator:
    """Tests for the simulate_switching_spike_oscillator function (Task 4.4)."""

    def test_output_shapes(self) -> None:
        """Verify all output arrays have correct shapes."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 100
        n_neurons = 5
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = n_oscillators * 2
        dt = 0.02

        key = jax.random.PRNGKey(0)

        # Create simple dynamics parameters
        transition_matrices = jnp.stack(
            [jnp.eye(n_latent) * 0.99, jnp.eye(n_latent) * 0.95], axis=-1
        )
        process_covs = jnp.stack(
            [jnp.eye(n_latent) * 0.01, jnp.eye(n_latent) * 0.02], axis=-1
        )
        discrete_transition_matrix = jnp.array([[0.95, 0.05], [0.05, 0.95]])

        # Spike observation parameters
        spike_weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.1
        spike_baseline = jnp.zeros(n_neurons)

        # Initial conditions
        init_mean = jnp.zeros(n_latent)
        init_cov = jnp.eye(n_latent)
        init_discrete_prob = jnp.ones(n_discrete_states) / n_discrete_states

        spikes, true_states, true_discrete_states = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        assert spikes.shape == (n_time, n_neurons), f"Expected ({n_time}, {n_neurons}), got {spikes.shape}"
        assert true_states.shape == (n_time, n_latent), f"Expected ({n_time}, {n_latent}), got {true_states.shape}"
        assert true_discrete_states.shape == (n_time,), f"Expected ({n_time},), got {true_discrete_states.shape}"

    def test_spike_counts_nonnegative(self) -> None:
        """Spike counts should be non-negative integers."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 50
        n_neurons = 3
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = n_oscillators * 2
        dt = 0.02

        key = jax.random.PRNGKey(42)

        transition_matrices = jnp.stack([jnp.eye(n_latent) * 0.99] * n_discrete_states, axis=-1)
        process_covs = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)
        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        spike_weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.1
        spike_baseline = jnp.zeros(n_neurons)

        init_mean = jnp.zeros(n_latent)
        init_cov = jnp.eye(n_latent)
        init_discrete_prob = jnp.ones(n_discrete_states) / n_discrete_states

        spikes, _, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        # All spike counts should be non-negative
        assert jnp.all(spikes >= 0), "Spike counts should be non-negative"

        # Spike counts should be integers (within floating point tolerance)
        np.testing.assert_allclose(spikes, jnp.round(spikes), atol=1e-10)

    def test_discrete_states_valid_range(self) -> None:
        """Discrete states should be in valid range [0, n_discrete_states)."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 100
        n_neurons = 3
        n_oscillators = 2
        n_discrete_states = 3
        n_latent = n_oscillators * 2
        dt = 0.02

        key = jax.random.PRNGKey(123)

        transition_matrices = jnp.stack([jnp.eye(n_latent) * 0.99] * n_discrete_states, axis=-1)
        process_covs = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)
        discrete_transition_matrix = jnp.ones((n_discrete_states, n_discrete_states)) / n_discrete_states

        spike_weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.1
        spike_baseline = jnp.zeros(n_neurons)

        init_mean = jnp.zeros(n_latent)
        init_cov = jnp.eye(n_latent)
        init_discrete_prob = jnp.ones(n_discrete_states) / n_discrete_states

        _, _, true_discrete_states = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        # All discrete states should be in [0, n_discrete_states)
        assert jnp.all(true_discrete_states >= 0), "Discrete states should be >= 0"
        assert jnp.all(true_discrete_states < n_discrete_states), f"Discrete states should be < {n_discrete_states}"

        # Should be integers
        np.testing.assert_allclose(true_discrete_states, jnp.round(true_discrete_states), atol=1e-10)

    def test_single_discrete_state(self) -> None:
        """Should work with single discrete state (no switching)."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 50
        n_neurons = 3
        n_oscillators = 2
        n_discrete_states = 1
        n_latent = n_oscillators * 2
        dt = 0.02

        key = jax.random.PRNGKey(55)

        transition_matrices = jnp.eye(n_latent)[:, :, None] * 0.99
        process_covs = jnp.eye(n_latent)[:, :, None] * 0.01
        discrete_transition_matrix = jnp.ones((1, 1))

        spike_weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.1
        spike_baseline = jnp.zeros(n_neurons)

        init_mean = jnp.zeros(n_latent)
        init_cov = jnp.eye(n_latent)
        init_discrete_prob = jnp.ones(n_discrete_states)

        spikes, true_states, true_discrete_states = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        # All discrete states should be 0
        np.testing.assert_array_equal(true_discrete_states, jnp.zeros(n_time))

    def test_reproducibility_with_same_key(self) -> None:
        """Same random key should produce identical results."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 30
        n_neurons = 3
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = n_oscillators * 2
        dt = 0.02

        key = jax.random.PRNGKey(99)

        transition_matrices = jnp.stack([jnp.eye(n_latent) * 0.99] * n_discrete_states, axis=-1)
        process_covs = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)
        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        spike_weights = jax.random.normal(jax.random.PRNGKey(0), (n_neurons, n_latent)) * 0.1
        spike_baseline = jnp.zeros(n_neurons)

        init_mean = jnp.zeros(n_latent)
        init_cov = jnp.eye(n_latent)
        init_discrete_prob = jnp.ones(n_discrete_states) / n_discrete_states

        # First run
        spikes1, states1, discrete1 = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        # Second run with same key
        spikes2, states2, discrete2 = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        np.testing.assert_array_equal(spikes1, spikes2)
        np.testing.assert_array_equal(states1, states2)
        np.testing.assert_array_equal(discrete1, discrete2)

    def test_different_keys_produce_different_results(self) -> None:
        """Different random keys should produce different results."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 50
        n_neurons = 3
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = n_oscillators * 2
        dt = 0.02

        transition_matrices = jnp.stack([jnp.eye(n_latent) * 0.99] * n_discrete_states, axis=-1)
        process_covs = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)
        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        spike_weights = jax.random.normal(jax.random.PRNGKey(0), (n_neurons, n_latent)) * 0.1
        spike_baseline = jnp.zeros(n_neurons)

        init_mean = jnp.zeros(n_latent)
        init_cov = jnp.eye(n_latent)
        init_discrete_prob = jnp.ones(n_discrete_states) / n_discrete_states

        spikes1, _, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=jax.random.PRNGKey(1),
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        spikes2, _, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=jax.random.PRNGKey(2),
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        # Results should be different
        assert not jnp.allclose(spikes1, spikes2)

    def test_no_nans(self) -> None:
        """Simulation should not produce NaN values."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 100
        n_neurons = 5
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = n_oscillators * 2
        dt = 0.02

        key = jax.random.PRNGKey(42)

        transition_matrices = jnp.stack([jnp.eye(n_latent) * 0.99] * n_discrete_states, axis=-1)
        process_covs = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)
        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        spike_weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.1
        spike_baseline = jnp.zeros(n_neurons)

        init_mean = jnp.zeros(n_latent)
        init_cov = jnp.eye(n_latent)
        init_discrete_prob = jnp.ones(n_discrete_states) / n_discrete_states

        spikes, true_states, true_discrete_states = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        assert not jnp.any(jnp.isnan(spikes))
        assert not jnp.any(jnp.isnan(true_states))
        assert not jnp.any(jnp.isnan(true_discrete_states))

    def test_higher_baseline_produces_more_spikes(self) -> None:
        """Higher baseline should produce more spikes on average."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 200
        n_neurons = 5
        n_oscillators = 2
        n_discrete_states = 1
        n_latent = n_oscillators * 2
        dt = 0.02

        key = jax.random.PRNGKey(0)

        transition_matrices = jnp.eye(n_latent)[:, :, None] * 0.99
        process_covs = jnp.eye(n_latent)[:, :, None] * 0.01
        discrete_transition_matrix = jnp.ones((1, 1))

        spike_weights = jnp.zeros((n_neurons, n_latent))  # No state dependence

        init_mean = jnp.zeros(n_latent)
        init_cov = jnp.eye(n_latent)
        init_discrete_prob = jnp.ones(n_discrete_states)

        # Low baseline
        spike_baseline_low = jnp.ones(n_neurons) * (-2.0)  # log(rate) = -2
        spikes_low, _, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline_low,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        # High baseline
        spike_baseline_high = jnp.ones(n_neurons) * 2.0  # log(rate) = 2
        spikes_high, _, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline_high,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        # High baseline should produce more spikes
        assert jnp.mean(spikes_high) > jnp.mean(spikes_low)

    def test_state_conditioned_dynamics(self) -> None:
        """Continuous state dynamics should follow transition matrices."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 100
        n_neurons = 1
        n_oscillators = 1
        n_discrete_states = 2
        n_latent = n_oscillators * 2
        dt = 0.02

        key = jax.random.PRNGKey(77)

        # Very different dynamics for each state
        # State 0: stable (decay to 0)
        # State 1: unstable (grow)
        A_0 = jnp.eye(n_latent) * 0.5  # Fast decay
        A_1 = jnp.eye(n_latent) * 0.99  # Slow decay
        transition_matrices = jnp.stack([A_0, A_1], axis=-1)

        # Low noise to see dynamics clearly
        process_covs = jnp.stack([jnp.eye(n_latent) * 0.001] * n_discrete_states, axis=-1)

        # Force staying in one state
        discrete_transition_matrix = jnp.eye(n_discrete_states)

        spike_weights = jnp.zeros((n_neurons, n_latent))
        spike_baseline = jnp.ones(n_neurons) * (-5)  # Very low rate

        # Start with large initial state
        init_mean = jnp.ones(n_latent) * 5.0
        init_cov = jnp.eye(n_latent) * 0.001
        init_discrete_prob = jnp.array([1.0, 0.0])  # Start in state 0

        _, true_states, _ = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
            init_mean=init_mean,
            init_cov=init_cov,
            init_discrete_prob=init_discrete_prob,
        )

        # With decay dynamics and low noise, state should decrease over time
        initial_magnitude = jnp.sqrt(jnp.sum(true_states[0] ** 2))
        final_magnitude = jnp.sqrt(jnp.sum(true_states[-1] ** 2))

        # After many steps with A=0.5, state should be much smaller
        assert final_magnitude < initial_magnitude * 0.1


class TestSimulateSwitchingSpikeOscillatorConvenienceFunction:
    """Tests for convenience function with default parameters."""

    def test_works_with_minimal_params(self) -> None:
        """Should work with just required parameters."""
        from state_space_practice.simulate.simulate_switching_spikes import (
            simulate_switching_spike_oscillator,
        )

        n_time = 50
        n_neurons = 3
        n_oscillators = 2
        n_discrete_states = 2
        n_latent = n_oscillators * 2
        dt = 0.02

        key = jax.random.PRNGKey(0)

        transition_matrices = jnp.stack([jnp.eye(n_latent) * 0.99] * n_discrete_states, axis=-1)
        process_covs = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)
        discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])

        spike_weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.1
        spike_baseline = jnp.zeros(n_neurons)

        # Use defaults for initial conditions
        spikes, true_states, true_discrete_states = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=transition_matrices,
            process_covs=process_covs,
            discrete_transition_matrix=discrete_transition_matrix,
            spike_weights=spike_weights,
            spike_baseline=spike_baseline,
            dt=dt,
            key=key,
        )

        # Should work and produce valid output
        assert spikes.shape == (n_time, n_neurons)
        assert not jnp.any(jnp.isnan(spikes))
