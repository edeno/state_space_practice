"""Tests for SwitchingHamiltonianJointModel."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def switching_model():
    from state_space_practice.hamiltonian_switching import SwitchingHamiltonianJointModel
    return SwitchingHamiltonianJointModel(
        n_oscillators=1,
        n_discrete_states=2,
        n_lfp_sources=2,
        n_spike_sources=3,
        sampling_freq=100.0,
        hidden_dims=[8, 8],
        seed=0,
    )


@pytest.fixture
def synthetic_data(switching_model):
    key = jax.random.PRNGKey(1)
    n_time = 20
    k1, k2 = jax.random.split(key)
    lfp = jax.random.normal(k1, (n_time, switching_model.n_lfp))
    spikes = jax.random.poisson(k2, 0.5, (n_time, switching_model.n_spikes)).astype(jnp.float32)
    return lfp, spikes


@pytest.fixture
def params(switching_model):
    params, _ = switching_model._build_param_spec()
    return params


class TestSwitchingHamiltonianSmooth:
    """Smoke tests for SwitchingHamiltonianJointModel.smooth()."""

    def test_smooth_runs(self, switching_model, synthetic_data, params):
        """smooth() should run without error and return arrays of correct shape."""
        lfp, spikes = synthetic_data
        m_s, P_s = switching_model.smooth(lfp, spikes, params)
        n_time = lfp.shape[0]
        n_lat = switching_model.n_cont_states
        n_k = switching_model.n_discrete_states
        assert m_s.shape == (n_time, n_lat, n_k)
        assert P_s.shape == (n_time, n_lat, n_lat, n_k)

    def test_filter_runs(self, switching_model, synthetic_data, params):
        """filter() should run without error."""
        lfp, spikes = synthetic_data
        means, covs, probs, lls = switching_model.filter(lfp, spikes, params)
        n_time = lfp.shape[0]
        assert means.shape[0] == n_time
        assert probs.shape == (n_time, switching_model.n_discrete_states)

    def test_smooth_sensitive_to_transition_asymmetry(self, switching_model, synthetic_data, params):
        """Smoother output should change when transition matrix changes,
        confirming Jacobian weighting uses actual probabilities."""
        lfp, spikes = synthetic_data

        # Run with symmetric transitions
        params_sym = {**params, "Z": jnp.array([[0.5, 0.5], [0.5, 0.5]])}
        m_sym, _ = switching_model.smooth(lfp, spikes, params_sym)

        # Run with highly asymmetric transitions
        params_asym = {**params, "Z": jnp.array([[0.99, 0.01], [0.01, 0.99]])}
        m_asym, _ = switching_model.smooth(lfp, spikes, params_asym)

        # Results must differ — if Jacobian weighting were uniform (ignoring
        # transition probs), both runs would produce identical smoothed means.
        assert not jnp.allclose(m_sym, m_asym, atol=1e-6), (
            "Smoother produced identical output for symmetric vs asymmetric "
            "transition matrices — Jacobian weighting may be ignoring probabilities"
        )
