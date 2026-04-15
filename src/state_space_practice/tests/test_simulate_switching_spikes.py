# ruff: noqa: E402
"""Tests for the simulate_switching_spikes module.

These tests verify the simulation utilities for generating synthetic data
from the switching spike-oscillator model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

from state_space_practice.simulate.simulate_switching_spikes import (
    simulate_switching_spike_oscillator,
)


@pytest.fixture
def switching_spike_params():
    """Standard parameters for switching spike-oscillator simulation tests."""
    n_neurons = 5
    n_oscillators = 2
    n_discrete_states = 2
    n_latent = n_oscillators * 2

    key = jax.random.PRNGKey(42)

    transition_matrices = jnp.stack(
        [jnp.eye(n_latent) * 0.99] * n_discrete_states, axis=-1
    )
    process_covs = jnp.stack(
        [jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1
    )
    discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.1, 0.9]])
    spike_weights = jax.random.normal(key, (n_neurons, n_latent)) * 0.1
    spike_baseline = jnp.zeros(n_neurons)
    init_mean = jnp.zeros(n_latent)
    init_cov = jnp.eye(n_latent)
    init_discrete_prob = jnp.ones(n_discrete_states) / n_discrete_states

    return {
        "n_neurons": n_neurons,
        "n_oscillators": n_oscillators,
        "n_discrete_states": n_discrete_states,
        "n_latent": n_latent,
        "transition_matrices": transition_matrices,
        "process_covs": process_covs,
        "discrete_transition_matrix": discrete_transition_matrix,
        "spike_weights": spike_weights,
        "spike_baseline": spike_baseline,
        "init_mean": init_mean,
        "init_cov": init_cov,
        "init_discrete_prob": init_discrete_prob,
    }


def _simulate(params, n_time=100, dt=0.02, key=None):
    """Helper to run simulation with standard params."""
    if key is None:
        key = jax.random.PRNGKey(42)
    return simulate_switching_spike_oscillator(
        n_time=n_time,
        transition_matrices=params["transition_matrices"],
        process_covs=params["process_covs"],
        discrete_transition_matrix=params["discrete_transition_matrix"],
        spike_weights=params["spike_weights"],
        spike_baseline=params["spike_baseline"],
        dt=dt,
        key=key,
        init_mean=params["init_mean"],
        init_cov=params["init_cov"],
        init_discrete_prob=params["init_discrete_prob"],
    )


class TestSimulateSwitchingSpikeOscillator:
    """Tests for the simulate_switching_spike_oscillator function."""

    def test_output_shapes(self, switching_spike_params) -> None:
        """Verify all output arrays have correct shapes."""
        p = switching_spike_params
        n_time = 100
        spikes, true_states, true_discrete_states = _simulate(p, n_time=n_time)

        assert spikes.shape == (n_time, p["n_neurons"])
        assert true_states.shape == (n_time, p["n_latent"])
        assert true_discrete_states.shape == (n_time,)

    def test_spike_counts_nonnegative(self, switching_spike_params) -> None:
        """Spike counts should be non-negative integers."""
        spikes, _, _ = _simulate(switching_spike_params, n_time=50)
        assert jnp.all(spikes >= 0)
        np.testing.assert_allclose(spikes, jnp.round(spikes), atol=1e-10)

    def test_discrete_states_valid_range(self) -> None:
        """Discrete states should be in valid range [0, n_discrete_states)."""
        n_discrete_states = 3
        n_latent = 4
        key = jax.random.PRNGKey(123)

        params = {
            "transition_matrices": jnp.stack(
                [jnp.eye(n_latent) * 0.99] * n_discrete_states, axis=-1
            ),
            "process_covs": jnp.stack(
                [jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1
            ),
            "discrete_transition_matrix": jnp.ones(
                (n_discrete_states, n_discrete_states)
            )
            / n_discrete_states,
            "spike_weights": jax.random.normal(key, (3, n_latent)) * 0.1,
            "spike_baseline": jnp.zeros(3),
            "init_mean": jnp.zeros(n_latent),
            "init_cov": jnp.eye(n_latent),
            "init_discrete_prob": jnp.ones(n_discrete_states) / n_discrete_states,
        }

        _, _, true_discrete_states = _simulate(params, n_time=100, key=key)

        assert jnp.all(true_discrete_states >= 0)
        assert jnp.all(true_discrete_states < n_discrete_states)
        np.testing.assert_allclose(
            true_discrete_states, jnp.round(true_discrete_states), atol=1e-10
        )

    def test_reproducibility_with_same_key(self, switching_spike_params) -> None:
        """Same random key should produce identical results."""
        key = jax.random.PRNGKey(99)
        s1, x1, d1 = _simulate(switching_spike_params, n_time=30, key=key)
        s2, x2, d2 = _simulate(switching_spike_params, n_time=30, key=key)

        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(d1, d2)

    def test_different_keys_produce_different_results(
        self, switching_spike_params
    ) -> None:
        """Different random keys should produce different results."""
        s1, _, _ = _simulate(
            switching_spike_params, n_time=50, key=jax.random.PRNGKey(1)
        )
        s2, _, _ = _simulate(
            switching_spike_params, n_time=50, key=jax.random.PRNGKey(2)
        )
        assert not jnp.allclose(s1, s2)

    def test_no_nans(self, switching_spike_params) -> None:
        """Simulation should not produce NaN values."""
        spikes, true_states, true_discrete_states = _simulate(
            switching_spike_params, n_time=100
        )
        assert not jnp.any(jnp.isnan(spikes))
        assert not jnp.any(jnp.isnan(true_states))
        assert not jnp.any(jnp.isnan(true_discrete_states))

    def test_higher_baseline_produces_more_spikes(self) -> None:
        """Higher baseline should produce more spikes on average."""
        n_time = 200
        n_neurons = 5
        n_latent = 4
        key = jax.random.PRNGKey(0)

        base_params = {
            "transition_matrices": jnp.eye(n_latent)[:, :, None] * 0.99,
            "process_covs": jnp.eye(n_latent)[:, :, None] * 0.01,
            "discrete_transition_matrix": jnp.ones((1, 1)),
            "spike_weights": jnp.zeros((n_neurons, n_latent)),
            "spike_baseline": None,  # set below
            "init_mean": jnp.zeros(n_latent),
            "init_cov": jnp.eye(n_latent),
            "init_discrete_prob": jnp.ones(1),
        }

        base_params["spike_baseline"] = jnp.ones(n_neurons) * (-2.0)
        spikes_low, _, _ = _simulate(base_params, n_time=n_time, key=key)

        base_params["spike_baseline"] = jnp.ones(n_neurons) * 2.0
        spikes_high, _, _ = _simulate(base_params, n_time=n_time, key=key)

        assert jnp.mean(spikes_high) > jnp.mean(spikes_low)

    def test_state_conditioned_dynamics(self) -> None:
        """Continuous state dynamics should follow transition matrices."""
        n_latent = 4
        n_discrete_states = 2
        key = jax.random.PRNGKey(77)

        params = {
            "transition_matrices": jnp.stack(
                [jnp.eye(n_latent) * 0.5, jnp.eye(n_latent) * 0.99], axis=-1
            ),
            "process_covs": jnp.stack(
                [jnp.eye(n_latent) * 0.001] * n_discrete_states, axis=-1
            ),
            "discrete_transition_matrix": jnp.eye(n_discrete_states),
            "spike_weights": jnp.zeros((1, n_latent)),
            "spike_baseline": jnp.ones(1) * (-5),
            "init_mean": jnp.ones(n_latent) * 5.0,
            "init_cov": jnp.eye(n_latent) * 0.001,
            "init_discrete_prob": jnp.array([1.0, 0.0]),
        }

        _, true_states, _ = _simulate(params, n_time=100, key=key)

        initial_magnitude = jnp.sqrt(jnp.sum(true_states[0] ** 2))
        final_magnitude = jnp.sqrt(jnp.sum(true_states[-1] ** 2))
        assert final_magnitude < initial_magnitude * 0.1

    def test_works_with_default_init_conditions(
        self, switching_spike_params
    ) -> None:
        """Should work with just required parameters (no init conditions)."""
        p = switching_spike_params
        n_time = 50
        key = jax.random.PRNGKey(0)

        spikes, true_states, true_discrete_states = simulate_switching_spike_oscillator(
            n_time=n_time,
            transition_matrices=p["transition_matrices"],
            process_covs=p["process_covs"],
            discrete_transition_matrix=p["discrete_transition_matrix"],
            spike_weights=p["spike_weights"],
            spike_baseline=p["spike_baseline"],
            dt=0.02,
            key=key,
        )

        assert spikes.shape == (n_time, p["n_neurons"])
        assert not jnp.any(jnp.isnan(spikes))
