"""Smoke tests for JointHamiltonianModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_joint import JointHamiltonianModel


class TestJointHamiltonianSmoke:

    @pytest.fixture
    def model_and_data(self):
        n_lfp = 4
        n_spikes = 8
        n_time = 50
        model = JointHamiltonianModel(
            n_lfp_sources=n_lfp,
            n_spike_sources=n_spikes,
            n_oscillators=1,
            hidden_dims=[16],
            seed=0,
            sampling_freq=1000.0,
        )
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        lfp = jax.random.normal(k1, (n_time, n_lfp))
        spikes = jax.random.poisson(k2, jnp.ones((n_time, n_spikes)) * 0.5)
        return model, lfp, spikes

    def test_construction(self, model_and_data):
        model, _, _ = model_and_data
        assert model.n_cont_states == 2

    def test_filter_runs(self, model_and_data):
        model, lfp, spikes = model_and_data
        params, _ = model._build_param_spec()
        m_f, P_f, ll = model.filter(lfp, spikes, params)
        assert jnp.all(jnp.isfinite(m_f))
        assert m_f.shape[0] == lfp.shape[0]

    def test_smooth_runs(self, model_and_data):
        model, lfp, spikes = model_and_data
        params, _ = model._build_param_spec()
        m_s, P_s = model.smooth(lfp, spikes, params)
        assert jnp.all(jnp.isfinite(m_s))
        assert m_s.shape[0] == lfp.shape[0]


class TestJointHamiltonianMultiOscillator:

    def test_construction_n_oscillators_2(self):
        model = JointHamiltonianModel(
            n_lfp_sources=4, n_spike_sources=8, n_oscillators=2,
            hidden_dims=[16], seed=0, sampling_freq=1000.0,
        )
        assert model.n_cont_states == 4
        assert model.init_mean.shape == (4, 1)
        m0 = model.init_mean[:, 0]
        assert jnp.allclose(m0[:2], 0.1)  # q values
        assert jnp.allclose(m0[2:], 0.0)  # p values

    def test_filter_n_oscillators_2(self):
        model = JointHamiltonianModel(
            n_lfp_sources=4, n_spike_sources=8, n_oscillators=2,
            hidden_dims=[16], seed=0, sampling_freq=1000.0,
        )
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        lfp = jax.random.normal(k1, (50, 4))
        spikes = jax.random.poisson(k2, jnp.ones((50, 8)) * 0.5)
        params, _ = model._build_param_spec()
        m_f, P_f, ll = model.filter(lfp, spikes, params)
        assert m_f.shape == (50, 4)
        assert jnp.all(jnp.isfinite(m_f))
