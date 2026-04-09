"""Smoke tests for HamiltonianLFPModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel
from state_space_practice.nonlinear_dynamics import leapfrog_step, apply_mlp


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


class TestHamiltonianLFPBehavioral:
    """Behavioral test: smoother should recover latent trajectory from observations."""

    def test_smoother_recovers_oscillator_trajectory(self):
        """Generate data from a known harmonic oscillator, verify smoother
        estimates are closer to truth than the prior mean."""
        n_sources = 2
        n_time = 200
        dt = 0.01
        omega = 2 * jnp.pi  # ~1 Hz oscillator

        # Create model with known parameters
        model = HamiltonianLFPModel(
            n_sources=n_sources, n_oscillators=1, hidden_dims=[8], seed=0,
            sampling_freq=1.0 / dt,
        )

        # Zero out MLP weights so dynamics are a pure harmonic oscillator
        mlp_params = model.mlp_params
        for k in mlp_params:
            if k.startswith("w") or k.startswith("b"):
                mlp_params[k] = jnp.zeros_like(mlp_params[k])
        model.mlp_params = mlp_params
        model.omega = omega

        # Known observation matrix: observe q and p directly
        model.C = jnp.eye(n_sources)  # n_sources must equal n_cont_states=2
        model.d = jnp.zeros(n_sources)
        model.obs_noise_std = 0.3
        model.measurement_matrix = jnp.stack([model.C], axis=2)

        # Simulate ground truth trajectory
        trans_params = {**mlp_params, "omega": omega}
        x0 = jnp.array([1.0, 0.0])  # blocked: [q=1, p=0]

        def sim_step(x, key):
            x_next = leapfrog_step(x, trans_params, apply_mlp, dt)
            x_next = x_next + jax.random.normal(key, x.shape) * 1e-3  # tiny process noise
            return x_next, x_next

        keys = jax.random.split(jax.random.PRNGKey(42), n_time)
        _, x_true = jax.lax.scan(sim_step, x0, keys)  # (n_time, 2)

        # Generate noisy observations
        obs_noise = jax.random.normal(jax.random.PRNGKey(99), (n_time, n_sources)) * model.obs_noise_std
        lfp = x_true @ model.C.T + model.d + obs_noise

        # Run smoother
        params, _ = model._build_param_spec()
        # Override params with known values
        params["mlp"] = mlp_params
        params["omega"] = omega
        params["C"] = model.C
        params["d"] = model.d
        params["init_mean"] = x0

        m_s, P_s = model.smooth(lfp, params)

        # Smoother MSE should be much lower than prior (zero) MSE
        smoother_mse = jnp.mean((m_s - x_true) ** 2)
        prior_mse = jnp.mean(x_true ** 2)  # prior mean is near zero

        assert smoother_mse < prior_mse * 0.5, (
            f"Smoother MSE ({smoother_mse:.4f}) should be much less than "
            f"prior MSE ({prior_mse:.4f}), indicating trajectory recovery"
        )
        # Smoother should track the oscillation
        assert jnp.corrcoef(m_s[:, 0], x_true[:, 0])[0, 1] > 0.8, (
            "Smoothed position should be highly correlated with true position"
        )
