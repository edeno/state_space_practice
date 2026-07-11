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
from state_space_practice.utils import divide_safe as _divide_safe
from state_space_practice.utils import scale_likelihood as _scale_likelihood


@pytest.fixture
def switching_model():
    from state_space_practice.hamiltonian_switching import (
        SwitchingHamiltonianJointModel,
    )

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
    spikes = jax.random.poisson(k2, 0.5, (n_time, switching_model.n_spikes)).astype(
        jnp.float32
    )
    return lfp, spikes


@pytest.fixture
def params(switching_model):
    params, _ = switching_model._build_param_spec()
    return params


class TestSwitchingHamiltonianSmooth:
    """Smoke tests for SwitchingHamiltonianJointModel.smooth()."""

    def test_combined_base_fields_initialized_for_every_state(self, switching_model):
        n_k = switching_model.n_discrete_states
        assert switching_model.measurement_matrix.shape == (
            switching_model.n_sources,
            switching_model.n_cont_states,
            n_k,
        )
        assert switching_model.measurement_cov.shape == (
            switching_model.n_sources,
            switching_model.n_sources,
            n_k,
        )
        for k in range(n_k):
            assert jnp.allclose(
                switching_model.measurement_matrix[: switching_model.n_lfp, :, k],
                switching_model.C_lfp,
            )
            assert jnp.allclose(
                switching_model.measurement_matrix[switching_model.n_lfp :, :, k],
                switching_model.C_spikes,
            )

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

    def test_smooth_empty_sequence_returns_empty(self, switching_model, params):
        """smooth() on a zero-length sequence returns empty trajectories.

        hamiltonian_core's shared RTS pass and gaussian_measurement_update
        handle T=0 (empty-in -> empty-out); the switching smoother's own
        backward pass must match, not IndexError while seeding the reverse scan
        from ``m_filt[-1]`` on an empty forward trajectory.
        """
        n_lat = switching_model.n_cont_states
        n_k = switching_model.n_discrete_states
        n_lfp, n_spikes = 2, 3
        lfp = jnp.zeros((0, n_lfp))
        spikes = jnp.zeros((0, n_spikes))

        m_s, P_s, pi_s = switching_model.smooth(lfp, spikes, params)

        assert m_s.shape == (0, n_lat, n_k)
        assert P_s.shape == (0, n_lat, n_lat, n_k)
        assert pi_s.shape == (0, n_k)

    def test_filter_runs(self, switching_model, synthetic_data, params):
        """filter() should run without error."""
        lfp, spikes = synthetic_data
        means, covs, probs, lls = switching_model.filter(lfp, spikes, params)
        n_time = lfp.shape[0]
        assert means.shape[0] == n_time
        assert probs.shape == (n_time, switching_model.n_discrete_states)

    def test_filter_covariance_defaults_are_not_stale(
        self, switching_model, synthetic_data, params
    ):
        """Mutating the per-state covariances between filter calls must take
        effect. On the old jit-with-static-self filter, R_lfp/process_cov/init_cov
        were baked into the compilation cache and silently reused."""
        lfp, spikes = synthetic_data
        n = switching_model.n_cont_states
        n_k = switching_model.n_discrete_states

        _, covs_before, _, _ = switching_model.filter(lfp, spikes, params)
        switching_model.R_lfp = jnp.eye(switching_model.n_lfp) * 10.0
        switching_model.process_cov = jnp.stack([jnp.eye(n) * 10.0] * n_k, axis=2)
        switching_model.init_cov = jnp.stack([jnp.eye(n) * 5.0] * n_k, axis=2)
        _, covs_after, _, _ = switching_model.filter(lfp, spikes, params)

        assert not jnp.allclose(covs_before, covs_after)

    @pytest.mark.parametrize(
        "bad_spikes, match",
        [
            (jnp.full((20, 3), -1.0), "non-negative"),
            (jnp.full((20, 3), 0.5), "integer-valued"),
        ],
    )
    def test_filter_validates_spike_counts(
        self, switching_model, synthetic_data, params, bad_spikes, match
    ):
        """The public filter must reject invalid spike counts loudly, like the
        non-switching siblings, rather than silently consuming them."""
        lfp, _ = synthetic_data
        with pytest.raises(ValueError, match=match):
            switching_model.filter(lfp, bad_spikes, params)

    def test_param_spec_drops_covariances(self, switching_model):
        """Q/init_cov/R_lfp are fixed model config for the switching model, not
        learnable params; a future accidental re-add would be silent."""
        params, spec = switching_model._build_param_spec()
        for key in ("Q", "init_cov", "R_lfp"):
            assert key not in params, f"{key} unexpectedly in switching params"
            assert key not in spec, f"{key} unexpectedly in switching spec"

    def test_smoother_final_probability_matches_filter(
        self, switching_model, synthetic_data, params
    ):
        """The last smoothed probability should equal the filtered probability."""
        lfp, spikes = synthetic_data
        params = {**params, "omega": jnp.array([1.0, 8.0])}
        switching_model.process_cov = jnp.stack(
            [
                jnp.eye(switching_model.n_cont_states) * 1e-4,
                jnp.array([[0.05, 0.01], [0.01, 0.02]]),
            ],
            axis=2,
        )

        _, _, pi_filt, _ = switching_model.filter(lfp, spikes, params)
        _, _, pi_smooth = switching_model.smooth(lfp, spikes, params)

        assert jnp.allclose(pi_smooth[-1], pi_filt[-1], atol=1e-6)

    def test_finalize_populates_decode_api(self, switching_model, synthetic_data):
        """fit/finalize should populate the attributes inherited decode() uses."""
        lfp, spikes = synthetic_data

        switching_model._finalize_sgd(lfp, spikes)

        decoded = switching_model.decode()
        probs = switching_model.predict_proba()
        assert decoded.shape == (lfp.shape[0],)
        assert probs.shape == (lfp.shape[0], switching_model.n_discrete_states)
        assert jnp.allclose(probs, switching_model.smoothed_discrete_probs_)

    def test_smooth_sensitive_to_transition_asymmetry(
        self, switching_model, synthetic_data, params
    ):
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

    def test_smoother_matches_pairwise_rts_reference(self):
        """The continuous smoother must update all (S_t, S_{t+1}) pairs."""
        from state_space_practice.hamiltonian_core import (
            gaussian_measurement_update,
            point_process_laplace_update,
        )
        from state_space_practice.hamiltonian_switching import (
            SwitchingHamiltonianJointModel,
        )
        from state_space_practice.nonlinear_dynamics import ekf_smooth_step
        from state_space_practice.switching_kalman import collapse_gaussian_mixture

        dt = 0.05
        model = SwitchingHamiltonianJointModel(
            n_oscillators=1,
            n_discrete_states=2,
            n_lfp_sources=2,
            n_spike_sources=1,
            sampling_freq=1.0 / dt,
            hidden_dims=[4],
            seed=0,
        )
        model.mlp_params = jax.tree_util.tree_map(jnp.zeros_like, model.mlp_params)
        model.omega = jnp.array([1.0, 4.0])
        model.C_lfp = jnp.eye(2)
        model.d_lfp = jnp.zeros(2)
        model.C_spikes = jnp.zeros((1, 2))
        model.d_spikes = jnp.array([-10.0])
        model.obs_noise_std = 0.25
        model.process_cov = jnp.stack(
            [jnp.eye(2) * 1e-3, jnp.array([[0.02, 0.004], [0.004, 0.015]])],
            axis=2,
        )

        params, _ = model._build_param_spec()
        params["mlp"] = model.mlp_params
        params["omega"] = model.omega
        params["C_lfp"] = model.C_lfp
        params["d_lfp"] = model.d_lfp
        params["C_spikes"] = model.C_spikes
        params["d_spikes"] = model.d_spikes
        params["init_mean"] = jnp.array([[0.8, -0.6], [0.1, 0.35]])
        params["Z"] = jnp.array([[0.05, 0.95], [0.85, 0.15]])
        params["init_pi"] = jnp.array([0.45, 0.55])

        lfp = jnp.array([[0.75, 0.05], [-0.35, 0.25]])
        spikes = jnp.zeros((2, 1))
        K = model.n_discrete_states

        def forward_reference():
            carry = (params["init_mean"], model.init_cov, params["init_pi"])
            outputs = []
            for y_lfp_t, y_spike_t in zip(lfp, spikes):
                m_prev, P_prev, pi_prev = carry
                (
                    m_pred,
                    P_pred,
                    F_pair,
                    joint_pi,
                    pi_pred,
                    m_pred_pair,
                    P_pred_pair,
                ) = model._per_state_pred_collapse(
                    m_prev,
                    P_prev,
                    pi_prev,
                    params["Z"],
                    params["mlp"],
                    params["omega"],
                    model.process_cov,
                    K,
                    with_jacobian=True,
                )
                m_posts = []
                P_posts = []
                lls = []
                for k in range(K):
                    m_mid, P_mid, ll_lfp = gaussian_measurement_update(
                        m_pred[:, k],
                        P_pred[:, :, k],
                        y_lfp_t,
                        params["C_lfp"],
                        params["d_lfp"],
                        model._r_lfp(),
                        include_normalization_const=False,
                    )
                    m_post, P_post, ll_spike = point_process_laplace_update(
                        m_mid,
                        P_mid,
                        y_spike_t,
                        params["C_spikes"],
                        params["d_spikes"],
                        model.dt,
                    )
                    m_posts.append(m_post)
                    P_posts.append(P_post)
                    lls.append(ll_lfp + ll_spike)

                m_filt = jnp.stack(m_posts, axis=1)
                P_filt = jnp.stack(P_posts, axis=2)
                lls = jnp.stack(lls)
                scaled_lik, _ = _scale_likelihood(lls)
                pi_filt = _divide_safe(
                    scaled_lik * pi_pred, jnp.sum(scaled_lik * pi_pred)
                )
                carry = (m_filt, P_filt, pi_filt)
                outputs.append(
                    (
                        m_filt,
                        P_filt,
                        m_pred,
                        P_pred,
                        m_pred_pair,
                        P_pred_pair,
                        F_pair,
                        joint_pi,
                        pi_pred,
                        pi_filt,
                    )
                )
            return tuple(jnp.stack(items) for items in zip(*outputs))

        (
            m_filt,
            P_filt,
            m_pred,
            P_pred,
            m_pred_pair,
            P_pred_pair,
            F_pair,
            joint_pi,
            pi_pred,
            pi_filt,
        ) = forward_reference()

        pi_s_next = pi_filt[-1]
        backward_cond = _divide_safe(joint_pi[1], pi_pred[1][None, :])
        joint_smooth = backward_cond * pi_s_next[None, :]
        pi_s_t0 = _divide_safe(jnp.sum(joint_smooth, axis=1), jnp.sum(joint_smooth))
        forward_cond = _divide_safe(joint_smooth, pi_s_t0[:, None])

        def smooth_reference_i(i):
            pair_means = []
            pair_covs = []
            for k in range(K):
                m_pair, P_pair = ekf_smooth_step(
                    m_filt[0, :, i],
                    P_filt[0, :, :, i],
                    m_pred_pair[1, i, k],
                    P_pred_pair[1, i, k],
                    m_filt[1, :, k],
                    P_filt[1, :, :, k],
                    F_pair[1, i, k],
                )
                pair_means.append(m_pair)
                pair_covs.append(P_pair)
            return collapse_gaussian_mixture(
                jnp.stack(pair_means, axis=1),
                jnp.stack(pair_covs, axis=2),
                forward_cond[i],
            )

        expected_m0, expected_P0 = jax.vmap(smooth_reference_i)(jnp.arange(K))

        def old_same_label_i(k):
            w_jk = _divide_safe(joint_pi[1, :, k], pi_pred[1, k])
            F_avg = jnp.sum(w_jk[:, None, None] * F_pair[1, :, k], axis=0)
            return ekf_smooth_step(
                m_filt[0, :, k],
                P_filt[0, :, :, k],
                m_pred[1, :, k],
                P_pred[1, :, :, k],
                m_filt[1, :, k],
                P_filt[1, :, :, k],
                F_avg,
            )

        old_m0, _ = jax.vmap(old_same_label_i)(jnp.arange(K))
        m_s, P_s, pi_s = model.smooth(lfp, spikes, params)

        assert not jnp.allclose(old_m0.T, expected_m0.T, atol=1e-4)
        assert jnp.allclose(m_s[0], expected_m0.T, atol=1e-8)
        assert jnp.allclose(P_s[0], expected_P0.transpose(1, 2, 0), atol=1e-8)
        assert jnp.allclose(pi_s[0], pi_s_t0, atol=1e-8)


class TestSwitchingHamiltonianBehavioral:
    """Behavioral test: smoother should discriminate modes from observations."""

    def test_smoother_discriminates_modes_from_observations(self):
        """Two states with very different frequencies. Generate data from
        state 0 only. The smoother should assign high probability to state 0."""
        from state_space_practice.hamiltonian_switching import (
            SwitchingHamiltonianJointModel,
        )

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
        trans_params_0 = {
            **jax.tree_util.tree_map(lambda x: x[0], model.mlp_params),
            "omega": model.omega[0],
        }
        x0 = jnp.array([1.0, 0.0])

        def sim_step(x, key):
            x_next = leapfrog_step(x, trans_params_0, apply_mlp, dt)
            return x_next, x_next

        keys = jax.random.split(jax.random.PRNGKey(42), n_time)
        _, x_true = jax.lax.scan(sim_step, x0, keys)

        # Generate LFP observations + zero spikes
        obs_noise = (
            jax.random.normal(jax.random.PRNGKey(99), (n_time, n_lfp))
            * model.obs_noise_std
        )
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
        from state_space_practice.hamiltonian_switching import (
            SwitchingHamiltonianJointModel,
        )

        dt = 0.01
        n_time = 500
        n_lfp = 2
        n_spikes = 1
        omega_0 = 2 * jnp.pi  # ~1 Hz
        omega_1 = 6 * jnp.pi  # ~3 Hz

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

        # Generate one continuous latent trajectory whose transition dynamics
        # are selected by the active state at each time bin. This matches the
        # model convention x_t = f_{S_t}(x_{t-1}) + noise.
        _, mlp_params = simulate_harmonic_oscillator(
            omega=omega_0,
            n_time=n_time,
            dt=dt,
            key=jax.random.PRNGKey(10),
            hidden_dims=[8],
        )
        trans_params_0 = {**mlp_params, "omega": omega_0}
        trans_params_1 = {**mlp_params, "omega": omega_1}

        def latent_step(x_prev, inputs):
            state_t, key_t = inputs
            x_next_0 = leapfrog_step(x_prev, trans_params_0, apply_mlp, dt)
            x_next_1 = leapfrog_step(x_prev, trans_params_1, apply_mlp, dt)
            x_next = jnp.where(state_t == 0, x_next_0, x_next_1)
            x_next = x_next + jax.random.normal(key_t, x_prev.shape) * 1e-3
            return x_next, x_next

        x0 = jnp.array([1.0, 0.0])
        noise_keys = jax.random.split(jax.random.PRNGKey(10), n_time)
        _, x_true = jax.lax.scan(latent_step, x0, (true_states, noise_keys))

        # Generate LFP observations
        C_lfp = jnp.eye(2)
        d_lfp = jnp.zeros(n_lfp)
        lfp = simulate_lfp_observations(
            x_true,
            C_lfp,
            d_lfp,
            noise_std=0.2,
            key=jax.random.PRNGKey(99),
        )
        spikes = jnp.zeros((n_time, n_spikes))

        model = SwitchingHamiltonianJointModel(
            n_oscillators=1,
            n_discrete_states=2,
            n_lfp_sources=n_lfp,
            n_spike_sources=n_spikes,
            sampling_freq=1.0 / dt,
            hidden_dims=[8],
            seed=0,
        )
        # Initialise both omegas close together so the model must learn
        # to separate them (avoids a vacuously true distinguishability test)
        mid_omega = (omega_0 + omega_1) / 2
        model.omega = jnp.array([mid_omega, mid_omega * 1.05])

        lls = model.fit_sgd(
            lfp,
            spikes,
            num_steps=200,
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
            np.array(true_states),
            np.array(pi_s),
        )
        assert acc >= 0.65, f"State segmentation accuracy {acc:.3f} < 0.65"

    def test_omegas_distinguishable(self, fitted):
        model, _, _, _, _ = fitted
        omega_gap = float(jnp.abs(model.omega[0] - model.omega[1]))
        assert omega_gap > 1.0, (
            f"Omega gap {omega_gap:.3f} < 1.0 "
            f"(learned: {model.omega}; true gap is ~12.6)"
        )


class TestSwitchingHamiltonianUseFilterGuard:
    """The switching model has no deterministic-rollout surrogate, so
    use_filter=False must fail loud rather than silently run the filter."""

    def test_sgd_loss_rejects_use_filter_false(
        self, switching_model, synthetic_data, params
    ):
        lfp, spikes = synthetic_data
        with pytest.raises(NotImplementedError, match="use_filter=True"):
            switching_model._sgd_loss_fn(params, lfp, spikes, use_filter=False)

    def test_fit_sgd_rejects_use_filter_false_through_jit(
        self, switching_model, synthetic_data
    ):
        # use_filter is a static Python bool, so the guard is a trace-time
        # branch. fit_sgd must surface a clean NotImplementedError from inside
        # the jitted train_step, NOT a TracerBoolConversionError — this asserts
        # the conditional is JAX-safe.
        lfp, spikes = synthetic_data
        with pytest.raises(NotImplementedError, match="use_filter=True"):
            switching_model.fit_sgd(lfp, spikes, num_steps=1, use_filter=False)
