"""Smoke tests for HamiltonianLFPModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel


class TestHamiltonianLFPSmoke:

    @pytest.fixture
    def model_and_data(self):
        n_sources = 4
        n_time = 50
        model = HamiltonianLFPModel(
            n_sources=n_sources, n_oscillators=1, hidden_dims=[16], seed=0,
            sampling_freq=1000.0,
        )
        lfp = jax.random.normal(jax.random.PRNGKey(42), (n_time, n_sources))
        return model, lfp

    def test_construction(self, model_and_data):
        model, _ = model_and_data
        assert model.n_cont_states == 2

    def test_filter_runs(self, model_and_data):
        model, lfp = model_and_data
        params, _ = model._build_param_spec()
        m_f, P_f, ll = model.filter(lfp, params)
        assert jnp.all(jnp.isfinite(m_f))
        assert m_f.shape[0] == lfp.shape[0]

    def test_smooth_runs(self, model_and_data):
        model, lfp = model_and_data
        params, _ = model._build_param_spec()
        m_s, P_s = model.smooth(lfp, params)
        assert jnp.all(jnp.isfinite(m_s))
        assert m_s.shape[0] == lfp.shape[0]


class TestHamiltonianLFPMultiOscillator:

    def test_construction_n_oscillators_2(self):
        model = HamiltonianLFPModel(
            n_sources=4, n_oscillators=2, hidden_dims=[16], seed=0,
            sampling_freq=1000.0,
        )
        assert model.n_cont_states == 4
        assert model.init_mean.shape == (4, 1)
        m0 = model.init_mean[:, 0]
        assert jnp.allclose(m0[:2], 0.1)  # q values
        assert jnp.allclose(m0[2:], 0.0)  # p values

    def test_filter_n_oscillators_2(self):
        model = HamiltonianLFPModel(
            n_sources=4, n_oscillators=2, hidden_dims=[16], seed=0,
            sampling_freq=1000.0,
        )
        lfp = jax.random.normal(jax.random.PRNGKey(0), (50, 4))
        params, _ = model._build_param_spec()
        m_f, P_f, ll = model.filter(lfp, params)
        assert m_f.shape == (50, 4)
        assert jnp.all(jnp.isfinite(m_f))
