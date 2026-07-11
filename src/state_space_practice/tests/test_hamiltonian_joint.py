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


def test_sgd_surrogate_loss_finite_under_rate_overflow():
    """The (use_filter=False) SGD surrogate loss stays finite when a parameter
    drives the spike log-rate past exp() overflow.

    A large spike log-rate offset ``d_spikes`` makes ``log_lambda`` huge, so an
    unguarded ``exp(log_lambda)`` overflows to +inf and ``0*log(inf)`` /
    ``inf-inf`` NaN-poisons the loss.

    The guard must ALSO preserve the gradient: a hard clip (``jnp.clip``) has
    exactly zero gradient above the cap, so once SGD overshoots it the surrogate
    freezes with no signal to return -- a dead-gradient stall. Here the rate is
    far above the spike counts, so the correctly-signed Poisson-NLL gradient is
    strictly positive; the loss and gradient must be finite AND that restoring
    gradient must be nonzero.
    """
    model = JointHamiltonianModel(
        n_lfp_sources=4,
        n_spike_sources=8,
        n_oscillators=1,
        hidden_dims=[16],
        seed=0,
        sampling_freq=1000.0,
    )
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    lfp = jax.random.normal(k1, (20, 4))
    spikes = jax.random.poisson(k2, jnp.ones((20, 8)) * 0.5)
    params, _ = model._build_param_spec()
    params = {**params, "d_spikes": jnp.full_like(params["d_spikes"], 1000.0)}

    def loss_fn(p):
        return model._sgd_loss_fn(p, lfp, spikes, use_filter=False)

    assert bool(jnp.isfinite(loss_fn(params)))
    grad = jax.grad(loss_fn)(params)
    assert bool(jnp.all(jnp.isfinite(grad["d_spikes"])))
    # A hard clip would leave grad["d_spikes"] == 0 (dead-gradient stall); a
    # gradient-preserving cap keeps a nonzero, correctly-signed restoring push.
    assert bool(jnp.all(grad["d_spikes"] > 0.0))


def test_sgd_surrogate_preserves_gradient_under_rate_underflow():
    """Positive spike counts retain a restoring gradient at tiny predicted rates.

    A very negative ``d_spikes`` drives the predicted rate below float underflow,
    so ``exp(log_lambda)`` flushes to 0. Reading ``log(mu)`` as ``log(rate + eps)``
    would floor at ``log(eps)`` and kill the gradient (a dead-gradient stall); the
    analytic ``log(mu)`` keeps a finite, correctly-signed restoring push (SGD
    raises ``d_spikes`` to lift the rate).
    """
    model = JointHamiltonianModel(
        n_lfp_sources=4,
        n_spike_sources=8,
        n_oscillators=1,
        hidden_dims=[16],
        seed=0,
        sampling_freq=1000.0,
    )
    lfp = jax.random.normal(jax.random.PRNGKey(0), (20, 4))
    spikes = jnp.ones((20, 8))
    params, _ = model._build_param_spec()
    params = {
        **params,
        "C_spikes": jnp.zeros_like(params["C_spikes"]),
        "d_spikes": jnp.full_like(params["d_spikes"], -1000.0),
    }

    def loss_fn(p):
        return model._sgd_loss_fn(p, lfp, spikes, use_filter=False)

    grad = jax.grad(loss_fn)(params)

    assert bool(jnp.isfinite(loss_fn(params)))
    assert bool(jnp.all(jnp.isfinite(grad["d_spikes"])))
    assert bool(jnp.all(grad["d_spikes"] < 0.0))


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
        assert model.measurement_matrix.shape == (12, 2, 1)
        assert jnp.allclose(model.measurement_matrix[: model.n_lfp, :, 0], model.C_lfp)
        assert jnp.allclose(
            model.measurement_matrix[model.n_lfp :, :, 0], model.C_spikes
        )
        assert jnp.allclose(
            model.measurement_cov[: model.n_lfp, : model.n_lfp, 0], model.R_lfp
        )

    def test_obs_noise_std_setter_updates_covariances(self, model_and_data):
        """Setting the scalar rebuilds R_lfp and the measurement_cov LFP block."""
        model, _, _ = model_and_data
        model.obs_noise_std = 0.5
        assert model.obs_noise_std == 0.5
        assert jnp.allclose(model.R_lfp, jnp.eye(model.n_lfp) * 0.25)
        assert jnp.allclose(
            model.measurement_cov[: model.n_lfp, : model.n_lfp, 0], model.R_lfp
        )

    def test_obs_noise_std_rejects_nonpositive(self, model_and_data):
        model, _, _ = model_and_data
        with pytest.raises(ValueError, match="positive"):
            model.obs_noise_std = 0.0

    def test_obs_noise_std_setter_warns_when_clobbering_fitted_R(self, model_and_data):
        """Overwriting a non-isotropic (e.g. fitted) R_lfp must not be silent."""
        model, _, _ = model_and_data
        model.R_lfp = jnp.diag(jnp.arange(1.0, model.n_lfp + 1.0))  # anisotropic
        with pytest.warns(UserWarning, match="discards the current non-isotropic"):
            model.obs_noise_std = 0.1
        assert jnp.allclose(model.R_lfp, jnp.eye(model.n_lfp) * 0.01)

    def test_sgd_lfp_gaussian_nll_matches_hand_computation(self, model_and_data):
        """The use_filter=False LFP term is the exact multivariate-Gaussian NLL
        (pins the 0.5 factor, logdet sign, and 2*pi constant of the new surrogate)."""
        import numpy as np

        model, _, _ = model_and_data
        params, _ = model._build_param_spec()
        n_lfp = model.n_lfp
        n_time = 6
        rng = np.random.default_rng(0)
        lfp = jnp.asarray(rng.normal(size=(n_time, n_lfp)))
        spikes = jnp.zeros((n_time, model.n_spikes))
        R = jnp.diag(jnp.arange(1.0, n_lfp + 1.0))  # well-conditioned PSD
        params = {
            **params,
            "C_lfp": jnp.zeros_like(params["C_lfp"]),  # predicted LFP = 0
            "d_lfp": jnp.zeros_like(params["d_lfp"]),
            "C_spikes": jnp.zeros_like(params["C_spikes"]),
            "d_spikes": jnp.full_like(params["d_spikes"], -1000.0),  # rates -> 0
            "R_lfp": R,
        }
        loss = model._sgd_loss_fn(params, lfp, spikes, use_filter=False, l2_reg=0.0)

        # residual = lfp (predicted 0); spike NLL ~ 0 (rates underflow) and l2 off,
        # so the loss is the LFP Gaussian NLL alone.
        r = np.asarray(lfp)
        R_np = np.asarray(R)
        rinv = np.linalg.inv(R_np)
        quad = np.sum(r * (rinv @ r.T).T)
        logdet = np.log(np.linalg.det(R_np))
        expected = 0.5 * (quad + n_time * (logdet + n_lfp * np.log(2.0 * np.pi)))
        assert float(loss) == pytest.approx(expected, rel=1e-4)

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

    @pytest.mark.parametrize(
        "lfp, spikes, match",
        [
            (jnp.zeros((5, 1)), jnp.zeros((5, 8)), "lfp_data"),
            (jnp.zeros((5, 4)), jnp.zeros((5, 1)), "spike_data"),
            (jnp.zeros((5, 4)), jnp.zeros((4, 8)), "same number"),
            (jnp.full((5, 4), jnp.nan), jnp.zeros((5, 8)), "finite"),
            (jnp.zeros((5, 4)), jnp.full((5, 8), -1.0), "non-negative"),
            (jnp.zeros((5, 4)), jnp.full((5, 8), 0.5), "integer-valued"),
            (jnp.empty((0, 4)), jnp.empty((0, 8)), "at least one time row"),
        ],
    )
    def test_fit_sgd_rejects_invalid_joint_data(
        self, model_and_data, lfp, spikes, match
    ):
        model, _, _ = model_and_data
        with pytest.raises(ValueError, match=match):
            model.fit_sgd(lfp, spikes, num_steps=0)

    def test_filter_covariance_defaults_are_not_stale(self, model_and_data):
        model, lfp, spikes = model_and_data
        params, _ = model._build_param_spec()
        params.pop("R_lfp")
        params.pop("Q")
        params.pop("init_cov")

        _, covs_before, _ = model.filter(lfp, spikes, params)
        model.R_lfp = jnp.eye(model.n_lfp) * 10.0
        model.process_cov = jnp.stack([jnp.eye(model.n_cont_states) * 10.0], axis=2)
        model.init_cov = jnp.stack([jnp.eye(model.n_cont_states) * 5.0], axis=2)
        _, covs_after, _ = model.filter(lfp, spikes, params)

        assert not jnp.allclose(covs_before, covs_after)

    def test_store_sgd_params_resyncs_combined_fields(self, model_and_data):
        model, _, _ = model_and_data
        params, _ = model._build_param_spec()
        params["C_lfp"] = jnp.ones_like(params["C_lfp"]) * 0.25
        params["C_spikes"] = jnp.ones_like(params["C_spikes"]) * -0.25
        params["R_lfp"] = jnp.eye(model.n_lfp) * 0.5

        model._store_sgd_params(params)

        assert jnp.allclose(
            model.measurement_matrix[: model.n_lfp, :, 0], params["C_lfp"]
        )
        assert jnp.allclose(
            model.measurement_matrix[model.n_lfp :, :, 0], params["C_spikes"]
        )
        assert jnp.allclose(
            model.measurement_cov[: model.n_lfp, : model.n_lfp, 0],
            params["R_lfp"],
        )

    def test_fit_sgd_rejects_removed_key_argument(self, model_and_data):
        model, lfp, spikes = model_and_data
        with pytest.raises(TypeError, match="unexpected keyword argument 'key'"):
            model.fit_sgd(lfp, spikes, key=jax.random.PRNGKey(1), num_steps=0)


class TestJointHamiltonianMultiOscillator:
    def test_construction_n_oscillators_2(self):
        model = JointHamiltonianModel(
            n_lfp_sources=4,
            n_spike_sources=8,
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
        model = JointHamiltonianModel(
            n_lfp_sources=4,
            n_spike_sources=8,
            n_oscillators=2,
            hidden_dims=[16],
            seed=0,
            sampling_freq=1000.0,
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
        lfp_noise = (
            jax.random.normal(jax.random.PRNGKey(99), (n_time, n_lfp)) * obs_noise_std
        )
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
            omega=omega_true,
            n_time=n_time,
            dt=dt,
            key=jax.random.PRNGKey(42),
            hidden_dims=[8],
        )

        C_lfp = jnp.eye(2)
        d_lfp = jnp.zeros(n_lfp)
        lfp = simulate_lfp_observations(
            x_true,
            C_lfp,
            d_lfp,
            noise_std=0.3,
            key=jax.random.PRNGKey(99),
        )

        C_spikes = jax.random.normal(jax.random.PRNGKey(10), (n_spikes, 2)) * 0.3
        d_spikes = -2.0 * jnp.ones(n_spikes)
        spikes = simulate_poisson_spikes(
            x_true,
            C_spikes,
            d_spikes,
            dt=dt,
            key=jax.random.PRNGKey(77),
        )

        model = JointHamiltonianModel(
            n_lfp_sources=n_lfp,
            n_spike_sources=n_spikes,
            n_oscillators=1,
            hidden_dims=[8],
            seed=0,
            sampling_freq=1.0 / dt,
        )
        model.omega = omega_true * 1.3

        lls = model.fit_sgd(
            lfp,
            spikes,
            num_steps=200,
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
        assert corr > 0.7, f"Post-learning smoother correlation {corr:.3f} < 0.7"

    def test_r_lfp_is_learned(self, fitted):
        model, _, _, _, _, _ = fitted
        R_initial = jnp.eye(model.n_lfp) * 0.1**2
        assert not jnp.allclose(model.R_lfp, R_initial, atol=1e-6)
        assert jnp.all(jnp.linalg.eigvalsh(model.R_lfp) > 0.0)
