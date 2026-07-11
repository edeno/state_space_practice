"""Smoke and recovery tests for HamiltonianLFPModel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel
from state_space_practice.kalman import kalman_filter, kalman_smoother
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    get_transition_jacobian,
    leapfrog_step,
)
from state_space_practice.tests.recovery_helpers import (
    assert_ll_improves,
    simulate_harmonic_oscillator,
    simulate_lfp_observations,
)


class TestHamiltonianLFPSmoke:
    @pytest.fixture
    def model_and_data(self):
        n_sources = 4
        n_time = 50
        model = HamiltonianLFPModel(
            n_sources=n_sources,
            n_oscillators=1,
            hidden_dims=[16],
            seed=0,
            sampling_freq=1000.0,
        )
        lfp = jax.random.normal(jax.random.PRNGKey(42), (n_time, n_sources))
        return model, lfp

    def test_construction(self, model_and_data):
        model, _ = model_and_data
        assert model.n_cont_states == 2

    def test_obs_noise_std_property_summarizes_measurement_cov(self, model_and_data):
        """Read-only scalar summary sqrt(mean(diag(R))); default 0.1 -> 0.1."""
        model, _ = model_and_data
        assert model.obs_noise_std == pytest.approx(0.1)
        # It is read-only (no isotropic-reset setter, unlike the joint model).
        with pytest.raises(AttributeError):
            model.obs_noise_std = 0.5

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

    def test_fit_raises_not_implemented(self, model_and_data):
        model, lfp = model_and_data
        with pytest.raises(NotImplementedError, match="fit_sgd"):
            model.fit(lfp, max_iter=1, skip_init=True)

    def test_store_sgd_params_resyncs_measurement_matrix(self, model_and_data):
        model, _ = model_and_data
        params, _ = model._build_param_spec()
        params["C"] = jnp.ones_like(model.C) * 0.25

        model._store_sgd_params(params)

        assert jnp.allclose(model.measurement_matrix[:, :, 0], model.C)

    def test_store_sgd_params_resyncs_measurement_covariance(self, model_and_data):
        model, _ = model_and_data
        params, _ = model._build_param_spec()
        params["R"] = jnp.eye(model.n_sources) * 0.25

        model._store_sgd_params(params)

        assert jnp.allclose(model.measurement_cov[:, :, 0], params["R"])

    @pytest.mark.parametrize(
        "lfp, match",
        [
            (jnp.zeros((5,)), "shape"),
            (jnp.zeros((5, 1)), "4 sources"),
            (jnp.full((5, 4), jnp.nan), "finite"),
            (jnp.full((5, 4), jnp.inf), "finite"),
            (jnp.empty((0, 4)), "at least one observation"),
        ],
    )
    def test_fit_sgd_rejects_invalid_lfp(self, model_and_data, lfp, match):
        model, _ = model_and_data
        with pytest.raises(ValueError, match=match):
            model.fit_sgd(lfp, num_steps=0)

    def test_filter_covariance_defaults_are_not_stale(self, model_and_data):
        model, lfp = model_and_data
        params, _ = model._build_param_spec()
        params.pop("R")
        params.pop("Q")
        params.pop("init_cov")

        _, covs_before, _ = model.filter(lfp, params)
        model.measurement_cov = jnp.stack([jnp.eye(model.n_sources) * 10.0], axis=2)
        model.process_cov = jnp.stack([jnp.eye(model.n_cont_states) * 10.0], axis=2)
        model.init_cov = jnp.stack([jnp.eye(model.n_cont_states) * 5.0], axis=2)
        _, covs_after, _ = model.filter(lfp, params)

        assert not jnp.allclose(covs_before, covs_after)

    def test_fit_sgd_rejects_removed_key_argument(self, model_and_data):
        model, lfp = model_and_data
        with pytest.raises(TypeError, match="unexpected keyword argument 'key'"):
            model.fit_sgd(lfp, key=jax.random.PRNGKey(1), num_steps=0)


class TestHamiltonianLFPMultiOscillator:
    def test_construction_n_oscillators_2(self):
        model = HamiltonianLFPModel(
            n_sources=4,
            n_oscillators=2,
            hidden_dims=[16],
            seed=0,
            sampling_freq=1000.0,
        )
        assert model.n_cont_states == 4
        assert model.init_mean.shape == (4, 1)
        m0 = model.init_mean[:, 0]
        assert jnp.allclose(m0[:2], 0.1)  # q values
        assert jnp.allclose(m0[2:], 0.0)  # p values

    def test_filter_n_oscillators_2(self):
        model = HamiltonianLFPModel(
            n_sources=4,
            n_oscillators=2,
            hidden_dims=[16],
            seed=0,
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
            n_sources=n_sources,
            n_oscillators=1,
            hidden_dims=[8],
            seed=0,
            sampling_freq=1.0 / dt,
            obs_noise_std=0.3,
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
        model.measurement_matrix = jnp.stack([model.C], axis=2)

        # Simulate ground truth trajectory
        trans_params = {**mlp_params, "omega": omega}
        x0 = jnp.array([1.0, 0.0])  # blocked: [q=1, p=0]

        def sim_step(x, key):
            x_next = leapfrog_step(x, trans_params, apply_mlp, dt)
            x_next = (
                x_next + jax.random.normal(key, x.shape) * 1e-3
            )  # tiny process noise
            return x_next, x_next

        keys = jax.random.split(jax.random.PRNGKey(42), n_time)
        _, x_true = jax.lax.scan(sim_step, x0, keys)  # (n_time, 2)

        # Generate noisy observations
        obs_noise = jax.random.normal(jax.random.PRNGKey(99), (n_time, n_sources)) * 0.3
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
        prior_mse = jnp.mean(x_true**2)  # prior mean is near zero

        assert smoother_mse < prior_mse * 0.5, (
            f"Smoother MSE ({smoother_mse:.4f}) should be much less than "
            f"prior MSE ({prior_mse:.4f}), indicating trajectory recovery"
        )
        # Smoother should track the oscillation
        assert jnp.corrcoef(m_s[:, 0], x_true[:, 0])[0, 1] > 0.8, (
            "Smoothed position should be highly correlated with true position"
        )


@pytest.mark.slow
class TestHamiltonianLFPSGDRecovery:
    """Verify fit_sgd learns omega and observation matrix from LFP data."""

    @pytest.fixture(scope="class")
    def fitted(self):
        omega_true = 2 * jnp.pi
        dt = 0.01
        n_time = 300

        x_true, mlp_params = simulate_harmonic_oscillator(
            omega=omega_true,
            n_time=n_time,
            dt=dt,
            key=jax.random.PRNGKey(42),
            hidden_dims=[8],
        )

        C_true = jnp.eye(2)
        d_true = jnp.zeros(2)
        lfp = simulate_lfp_observations(
            x_true,
            C_true,
            d_true,
            noise_std=0.3,
            key=jax.random.PRNGKey(99),
        )

        # Initialise model with perturbed omega
        model = HamiltonianLFPModel(
            n_sources=2,
            n_oscillators=1,
            hidden_dims=[8],
            seed=0,
            sampling_freq=1.0 / dt,
        )
        model.omega = omega_true * 1.3  # 30% off

        lls = model.fit_sgd(
            lfp,
            num_steps=200,
        )
        return model, x_true, omega_true, lfp, lls

    def test_ll_improves(self, fitted):
        _, _, _, _, lls = fitted
        assert_ll_improves(lls, label="HamiltonianLFP SGD")

    def test_frequency_recovery(self, fitted):
        model, _, omega_true, _, _ = fitted
        rel_error = float(abs(model.omega - omega_true) / omega_true)
        assert rel_error < 0.20, (
            f"Omega relative error {rel_error:.3f} >= 0.20 "
            f"(learned={float(model.omega):.3f}, true={float(omega_true):.3f})"
        )

    def test_smoother_tracks_truth(self, fitted):
        model, x_true, _, lfp, _ = fitted
        params, _ = model._build_param_spec()
        m_s, _ = model.smooth(lfp, params)
        corr = float(jnp.corrcoef(m_s[:, 0], x_true[:, 0])[0, 1])
        assert corr > 0.7, f"Post-learning smoother correlation {corr:.3f} < 0.7"

    def test_q_is_learned(self, fitted):
        """Process covariance Q should change from its initial value after SGD."""
        model, _, _, _, _ = fitted
        Q_learned = model.process_cov[:, :, 0]
        # Initial Q was 1e-4 * I; after learning it should differ
        Q_init = jnp.eye(model.n_cont_states) * 1e-4
        assert not jnp.allclose(Q_learned, Q_init, atol=1e-6), (
            "Q should have changed from its initial value after SGD"
        )
        # Q should remain PSD (positive eigenvalues)
        eigvals = jnp.linalg.eigvalsh(Q_learned)
        assert jnp.all(eigvals > 0), (
            f"Learned Q should be PSD, but has eigenvalues {eigvals}"
        )

    def test_r_is_learned(self, fitted):
        """Measurement covariance R should be learned and remain PSD."""
        model, _, _, _, _ = fitted
        R_learned = model.measurement_cov[:, :, 0]
        R_init = jnp.eye(model.n_sources) * 0.1**2
        assert not jnp.allclose(R_learned, R_init, atol=1e-6), (
            "R should have changed from its initial value after SGD"
        )
        eigvals = jnp.linalg.eigvalsh(R_learned)
        assert jnp.all(eigvals > 0), (
            f"Learned R should be PSD, but has eigenvalues {eigvals}"
        )


@pytest.mark.slow
class TestHamiltonianLFPLinearLimit:
    """With a zeroed MLP the dynamics are linear, so the EKF must reduce to the
    exact linear Kalman filter/smoother.

    The core F-alignment test validates ``ekf_rts_backward_pass`` for
    hand-constructed inputs; this pins the *model-level* forward-scan wiring --
    that ``filter``/``smooth`` actually produce correctly-aligned
    ``m_pred``/``P_pred``/``F`` per step. ``kalman_filter`` uses the same
    predict-then-update-from-init convention as the model, so the match is exact.
    """

    def test_matches_kalman_filter_and_smoother(self):
        model = HamiltonianLFPModel(
            n_oscillators=2, n_sources=3, sampling_freq=100.0, seed=0
        )
        params, _ = model._build_param_spec()
        # Zero the MLP -> linear (harmonic) leapfrog; zero offset d so the linear
        # observation y ~ N(C x, R) matches kalman_filter's (offset-free) model.
        params = dict(params)
        params["mlp"] = jax.tree_util.tree_map(jnp.zeros_like, params["mlp"])
        params["d"] = jnp.zeros_like(params["d"])

        # Constant transition matrix = the zeroed-MLP leapfrog Jacobian.
        trans_params = {**params["mlp"], "omega": params["omega"]}
        F = get_transition_jacobian(
            params["init_mean"], trans_params, apply_mlp, model.dt
        )

        lfp = jax.random.normal(jax.random.PRNGKey(1), (40, 3)) * 0.5
        init_mean = params["init_mean"]
        init_cov = model.init_cov[:, :, 0]
        Q = params["Q"]
        C = params["C"]
        R = params["R"]

        # sanity: the zeroed-MLP transition really is the linear map f(x) = F @ x.
        probe = jnp.array([0.3, -0.4, 0.2, 0.6])
        np.testing.assert_allclose(
            np.asarray(model.transition_func(probe, trans_params)),
            np.asarray(F @ probe),
            atol=1e-10,
        )

        m_filt, P_filt, lls = model.filter(lfp, params)
        m_smooth, P_smooth = model.smooth(lfp, params)

        ref_fm, ref_fc, ref_ll = kalman_filter(init_mean, init_cov, lfp, F, Q, C, R)
        ref_sm, ref_sc, _, _ = kalman_smoother(init_mean, init_cov, lfp, F, Q, C, R)

        np.testing.assert_allclose(
            np.asarray(m_filt), np.asarray(ref_fm), rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            np.asarray(P_filt), np.asarray(ref_fc), rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(float(jnp.sum(lls)), float(ref_ll), rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(m_smooth), np.asarray(ref_sm), rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            np.asarray(P_smooth), np.asarray(ref_sc), rtol=1e-6, atol=1e-8
        )
        # guard: the smoother genuinely differs from the filter (non-vacuous).
        assert float(jnp.max(jnp.abs(m_smooth - m_filt))) > 1e-3
