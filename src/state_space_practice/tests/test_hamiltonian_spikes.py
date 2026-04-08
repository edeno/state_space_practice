"""Tests for HamiltonianSpikeModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_spikes import HamiltonianSpikeModel


class TestHamiltonianSpikeModelSmoke:
    """Smoke tests: instantiation, filter, smooth all run without error."""

    @pytest.fixture
    def model_and_data(self):
        n_sources = 8
        n_time = 50
        model = HamiltonianSpikeModel(
            n_sources=n_sources, n_oscillators=1, hidden_dims=[16], seed=0,
            sampling_freq=1000.0,
        )
        key = jax.random.PRNGKey(42)
        spikes = jax.random.poisson(key, jnp.ones((n_time, n_sources)) * 0.5)
        return model, spikes

    def test_smooth_runs(self, model_and_data):
        """smooth() should not raise NameError from missing imports."""
        model, spikes = model_and_data
        params = {
            "mlp": model.mlp_params,
            "omega": model.omega,
            "C": model.C,
            "d": model.d,
            "init_mean": model.init_mean[:, 0],
        }
        m_s, P_s = model.smooth(spikes, params)
        assert jnp.all(jnp.isfinite(m_s)), "Smoothed means contain non-finite values"
        assert m_s.shape[0] == spikes.shape[0]


class TestHamiltonianSpikeMultiOscillator:
    """Tests for n_oscillators > 1."""

    def test_construction_n_oscillators_2(self):
        model = HamiltonianSpikeModel(
            n_sources=8, n_oscillators=2, hidden_dims=[16], seed=0, sampling_freq=1000.0
        )
        assert model.n_cont_states == 4
        assert model.init_mean.shape == (4, 1)
        # Verify blocked layout: first half should be q values (0.1), second half p values (0.0)
        m0 = model.init_mean[:, 0]
        assert jnp.allclose(m0[:2], 0.1)  # q values
        assert jnp.allclose(m0[2:], 0.0)  # p values

    def test_filter_n_oscillators_2(self):
        model = HamiltonianSpikeModel(
            n_sources=8, n_oscillators=2, hidden_dims=[16], seed=0, sampling_freq=1000.0
        )
        spikes = jax.random.poisson(jax.random.PRNGKey(0), jnp.ones((50, 8)) * 0.5)
        params = {
            "mlp": model.mlp_params,
            "omega": model.omega,
            "C": model.C,
            "d": model.d,
            "init_mean": model.init_mean[:, 0],
        }
        m_f, P_f, ll = model.filter(spikes, params)
        assert m_f.shape == (50, 4)
        assert jnp.all(jnp.isfinite(m_f))
