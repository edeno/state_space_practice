"""Tests for SwitchingHamiltonianJointModel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.nonlinear_dynamics import apply_mlp, leapfrog_step
from state_space_practice.tests.recovery_helpers import (
    assert_ll_improves,
    simulate_harmonic_oscillator,
    simulate_lfp_observations,
    state_segmentation_accuracy,
)


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


@pytest.mark.slow
class TestSwitchingHamiltonianSGDRecovery:
    """Verify fit_sgd learns distinguishable omegas and recovers state structure."""

    @pytest.fixture(scope="class")
    def fitted(self):
        from state_space_practice.hamiltonian_switching import SwitchingHamiltonianJointModel

        dt = 0.01
        n_time = 500
        n_lfp = 2
        n_spikes = 1
        omega_0 = 2 * jnp.pi   # ~1 Hz
        omega_1 = 6 * jnp.pi   # ~3 Hz

        # Generate switching state sequence (Z diagonal=0.95)
        Z = jnp.array([[0.95, 0.05], [0.05, 0.95]])
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)

        def markov_step(state, k):
            probs = Z[state]
            next_state = jax.random.choice(k, 2, p=probs).astype(jnp.int32)
            return next_state, next_state

        keys = jax.random.split(subkey, n_time)
        _, true_states = jax.lax.scan(markov_step, jnp.int32(0), keys)

        # Generate latent trajectory per state
        x_true_0, mlp_params = simulate_harmonic_oscillator(
            omega=omega_0, n_time=n_time, dt=dt,
            key=jax.random.PRNGKey(10), hidden_dims=[8],
        )
        x_true_1, _ = simulate_harmonic_oscillator(
            omega=omega_1, n_time=n_time, dt=dt,
            key=jax.random.PRNGKey(11), hidden_dims=[8],
        )

        # Select latent trajectory based on state
        x_true = jnp.where(
            true_states[:, None] == 0, x_true_0, x_true_1,
        )

        # Generate LFP observations
        C_lfp = jnp.eye(2)
        d_lfp = jnp.zeros(n_lfp)
        lfp = simulate_lfp_observations(
            x_true, C_lfp, d_lfp, noise_std=0.2,
            key=jax.random.PRNGKey(99),
        )
        spikes = jnp.zeros((n_time, n_spikes))

        model = SwitchingHamiltonianJointModel(
            n_oscillators=1, n_discrete_states=2,
            n_lfp_sources=n_lfp, n_spike_sources=n_spikes,
            sampling_freq=1.0 / dt, hidden_dims=[8], seed=0,
        )
        # Initialise both omegas close together so the model must learn
        # to separate them (avoids a vacuously true distinguishability test)
        mid_omega = (omega_0 + omega_1) / 2
        model.omega = jnp.array([mid_omega, mid_omega * 1.05])

        lls = model.fit_sgd(
            lfp, spikes, key=jax.random.PRNGKey(1), num_steps=200,
        )
        return model, true_states, lfp, spikes, lls

    def test_ll_improves(self, fitted):
        _, _, _, _, lls = fitted
        assert_ll_improves(lls, label="SwitchingHamiltonian SGD")

    def test_state_segmentation(self, fitted):
        model, true_states, lfp, spikes, _ = fitted
        params, _ = model._build_param_spec()
        _, _, pi_s = model.smooth(lfp, spikes, params)
        acc = state_segmentation_accuracy(
            np.array(true_states), np.array(pi_s),
        )
        assert acc >= 0.65, (
            f"State segmentation accuracy {acc:.3f} < 0.65"
        )

    def test_omegas_distinguishable(self, fitted):
        model, _, _, _, _ = fitted
        omega_gap = float(jnp.abs(model.omega[0] - model.omega[1]))
        assert omega_gap > 1.0, (
            f"Omega gap {omega_gap:.3f} < 1.0 "
            f"(learned: {model.omega}; true gap is ~12.6)"
        )
