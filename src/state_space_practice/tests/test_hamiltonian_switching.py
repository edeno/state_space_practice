"""Tests for SwitchingHamiltonianJointModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.nonlinear_dynamics import leapfrog_step, apply_mlp


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
        m_s, P_s, pi_s = switching_model.smooth(lfp, spikes, params)
        n_time = lfp.shape[0]
        n_lat = switching_model.n_cont_states
        n_k = switching_model.n_discrete_states
        assert m_s.shape == (n_time, n_lat, n_k)
        assert P_s.shape == (n_time, n_lat, n_lat, n_k)
        assert pi_s.shape == (n_time, n_k)

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
        m_sym, _, _ = switching_model.smooth(lfp, spikes, params_sym)

        # Run with highly asymmetric transitions
        params_asym = {**params, "Z": jnp.array([[0.99, 0.01], [0.01, 0.99]])}
        m_asym, _, _ = switching_model.smooth(lfp, spikes, params_asym)

        # Results must differ — if Jacobian weighting were uniform (ignoring
        # transition probs), both runs would produce identical smoothed means.
        assert not jnp.allclose(m_sym, m_asym, atol=1e-6), (
            "Smoother produced identical output for symmetric vs asymmetric "
            "transition matrices — Jacobian weighting may be ignoring probabilities"
        )


class TestSwitchingHamiltonianBehavioral:
    """Behavioral test: smoother should discriminate modes from observations."""

    def test_smoother_discriminates_modes_from_observations(self):
        """Two states with very different frequencies. Generate data from
        state 0 only. The smoother should assign high probability to state 0."""
        from state_space_practice.hamiltonian_switching import SwitchingHamiltonianJointModel

        n_lfp = 2
        n_spikes = 1  # minimal spike source with zero observations
        n_time = 100
        dt = 0.01

        model = SwitchingHamiltonianJointModel(
            n_oscillators=1,
            n_discrete_states=2,
            n_lfp_sources=n_lfp,
            n_spike_sources=n_spikes,
            sampling_freq=1.0 / dt,
            hidden_dims=[8],
            seed=0,
        )

        # State 0: slow oscillator (omega=1), State 1: fast oscillator (omega=10)
        # Zero out MLP weights
        for k in model.mlp_params:
            if isinstance(model.mlp_params[k], jnp.ndarray):
                if k.startswith("w") or k.startswith("b"):
                    model.mlp_params[k] = jnp.zeros_like(model.mlp_params[k])
        model.omega = jnp.array([1.0, 10.0])

        # LFP observes position and momentum directly
        model.C_lfp = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        model.d_lfp = jnp.zeros(n_lfp)
        model.obs_noise_std = 0.1

        # Spike readout: minimal (won't drive inference with zero spikes)
        model.C_spikes = jnp.zeros((n_spikes, 2))
        model.d_spikes = jnp.zeros(n_spikes)

        # Update measurement_matrix for BaseModel compliance
        C_full = jnp.concatenate([model.C_lfp, model.C_spikes], axis=0)
        model.measurement_matrix = jnp.stack([C_full, C_full], axis=2)

        # Simulate from STATE 0 (slow oscillator, omega=1)
        trans_params_0 = {**jax.tree_util.tree_map(lambda x: x[0], model.mlp_params), "omega": model.omega[0]}
        x0 = jnp.array([1.0, 0.0])

        def sim_step(x, key):
            x_next = leapfrog_step(x, trans_params_0, apply_mlp, dt)
            return x_next, x_next

        keys = jax.random.split(jax.random.PRNGKey(42), n_time)
        _, x_true = jax.lax.scan(sim_step, x0, keys)

        # Generate LFP observations + zero spikes
        obs_noise = jax.random.normal(jax.random.PRNGKey(99), (n_time, n_lfp)) * model.obs_noise_std
        lfp = x_true @ model.C_lfp.T + model.d_lfp + obs_noise
        spikes = jnp.zeros((n_time, n_spikes))

        # Strong persistence
        model.discrete_transition_matrix = jnp.array([[0.95, 0.05], [0.05, 0.95]])

        # Run smoother
        params, _ = model._build_param_spec()
        params["mlp"] = model.mlp_params
        params["omega"] = model.omega
        params["C_lfp"] = model.C_lfp
        params["d_lfp"] = model.d_lfp
        params["C_spikes"] = model.C_spikes
        params["d_spikes"] = model.d_spikes
        params["init_mean"] = jnp.stack([x0, x0], axis=1)
        params["Z"] = model.discrete_transition_matrix
        params["init_pi"] = jnp.array([0.5, 0.5])

        m_s, P_s, pi_s = model.smooth(lfp, spikes, params)

        # Data was generated from state 0 -- smoother should favor state 0
        mean_prob_state0 = jnp.mean(pi_s[:, 0])
        assert mean_prob_state0 > 0.7, (
            f"Mean smoothed probability of true state (0) is {mean_prob_state0:.3f}, "
            f"expected > 0.7 for observation-driven mode discrimination"
        )
