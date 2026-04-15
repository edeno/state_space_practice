"""Tests for JointHamiltonianModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_joint import JointHamiltonianModel
from state_space_practice.nonlinear_dynamics import apply_mlp, leapfrog_step
from state_space_practice.tests.recovery_helpers import (
    assert_ll_improves,
    simulate_harmonic_oscillator,
    simulate_lfp_observations,
    simulate_poisson_spikes,
)


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


@pytest.mark.slow
class TestJointHamiltonianBehavioral:
    """Behavioral test: joint model should recover latent trajectory."""

    def test_smoother_recovers_oscillator_from_joint_observations(self):
        """Given LFP + spikes from a known oscillator, smoother should
        produce estimates closer to truth than the prior mean."""
        n_lfp = 2
        n_spikes = 4
        n_time = 200
        dt = 0.01
        omega = 2 * jnp.pi

        model = JointHamiltonianModel(
            n_lfp_sources=n_lfp,
            n_spike_sources=n_spikes,
            n_oscillators=1,
            hidden_dims=[8],
            seed=0,
            sampling_freq=1.0 / dt,
        )

        # Zero out MLP for pure harmonic oscillator
        mlp_params = dict(model.mlp_params)
        for k in mlp_params:
            if k.startswith("w") or k.startswith("b"):
                mlp_params[k] = jnp.zeros_like(mlp_params[k])
        model.mlp_params = mlp_params
        model.omega = omega

        # Simulate ground truth
        trans_params = {**mlp_params, "omega": omega}
        x0 = jnp.array([1.0, 0.0])

        def sim_step(x, key):
            x_next = leapfrog_step(x, trans_params, apply_mlp, dt)
            x_next = x_next + jax.random.normal(key, x.shape) * 1e-3
            return x_next, x_next

        keys = jax.random.split(jax.random.PRNGKey(42), n_time)
        _, x_true = jax.lax.scan(sim_step, x0, keys)

        # Generate LFP observations
        C_lfp = model.C_lfp
        d_lfp = model.d_lfp
        obs_noise_std = model.obs_noise_std
        lfp_noise = jax.random.normal(
            jax.random.PRNGKey(99), (n_time, n_lfp)
        ) * obs_noise_std
        lfp = x_true @ C_lfp.T + d_lfp + lfp_noise

        # Generate spikes
        C_spike = model.C_spikes
        d_spike = model.d_spikes
        log_rates = x_true @ C_spike.T + d_spike
        rates = jnp.exp(jnp.clip(log_rates, -5, 3)) * dt
        spikes = jax.random.poisson(jax.random.PRNGKey(77), rates)

        # Run smoother
        params, _ = model._build_param_spec()
        params["mlp"] = mlp_params
        params["omega"] = omega
        params["init_mean"] = x0

        m_s, _ = model.smooth(lfp, spikes, params)

        # Smoother should be closer to truth than prior
        smoother_mse = float(jnp.mean((m_s[:, 0] - x_true[:, 0]) ** 2))
        prior_mse = float(jnp.mean(x_true[:, 0] ** 2))

        assert smoother_mse < prior_mse, (
            f"Joint smoother MSE ({smoother_mse:.4f}) should be less than "
            f"prior MSE ({prior_mse:.4f})"
        )


@pytest.mark.slow
class TestJointHamiltonianSGDRecovery:
    """Verify fit_sgd learns omega from joint LFP + spike observations."""

    @pytest.fixture(scope="class")
    def fitted(self):
        omega_true = 2 * jnp.pi
        dt = 0.01
        n_time = 300
        n_lfp = 2
        n_spikes = 4

        x_true, mlp_params = simulate_harmonic_oscillator(
            omega=omega_true, n_time=n_time, dt=dt,
            key=jax.random.PRNGKey(42), hidden_dims=[8],
        )

        C_lfp = jnp.eye(2)
        d_lfp = jnp.zeros(n_lfp)
        lfp = simulate_lfp_observations(
            x_true, C_lfp, d_lfp, noise_std=0.3,
            key=jax.random.PRNGKey(99),
        )

        C_spikes = jax.random.normal(jax.random.PRNGKey(10), (n_spikes, 2)) * 0.3
        d_spikes = -2.0 * jnp.ones(n_spikes)
        spikes = simulate_poisson_spikes(
            x_true, C_spikes, d_spikes, dt=dt,
            key=jax.random.PRNGKey(77),
        )

        model = JointHamiltonianModel(
            n_lfp_sources=n_lfp, n_spike_sources=n_spikes,
            n_oscillators=1, hidden_dims=[8], seed=0,
            sampling_freq=1.0 / dt,
        )
        model.omega = omega_true * 1.3

        lls = model.fit_sgd(
            lfp, spikes, key=jax.random.PRNGKey(1), num_steps=200,
        )
        return model, x_true, omega_true, lfp, spikes, lls

    def test_ll_improves(self, fitted):
        _, _, _, _, _, lls = fitted
        assert_ll_improves(lls, label="JointHamiltonian SGD")

    def test_frequency_recovery(self, fitted):
        model, _, omega_true, _, _, _ = fitted
        rel_error = float(abs(model.omega - omega_true) / omega_true)
        assert rel_error < 0.20, (
            f"Omega relative error {rel_error:.3f} >= 0.20 "
            f"(learned={float(model.omega):.3f}, true={float(omega_true):.3f})"
        )

    def test_smoother_tracks_truth(self, fitted):
        model, x_true, _, lfp, spikes, _ = fitted
        params, _ = model._build_param_spec()
        m_s, _ = model.smooth(lfp, spikes, params)
        corr = float(jnp.corrcoef(m_s[:, 0], x_true[:, 0])[0, 1])
        assert corr > 0.7, (
            f"Post-learning smoother correlation {corr:.3f} < 0.7"
        )
