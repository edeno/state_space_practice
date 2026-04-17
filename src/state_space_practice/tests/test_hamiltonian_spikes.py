"""Tests for HamiltonianSpikeModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_spikes import HamiltonianSpikeModel
from state_space_practice.nonlinear_dynamics import apply_mlp, leapfrog_step
from state_space_practice.tests.recovery_helpers import (
    assert_ll_improves,
    simulate_harmonic_oscillator,
    simulate_poisson_spikes,
)


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


@pytest.mark.slow
class TestHamiltonianSpikeBehavioral:
    """Behavioral test: smoother should track state modulation in spike data."""

    def test_smoother_detects_rate_modulation(self):
        """When latent oscillation modulates spike rates, smoother should
        recover a trajectory correlated with the true latent state."""
        n_sources = 4
        n_time = 200
        dt = 0.01
        omega = 2 * jnp.pi  # ~1 Hz

        model = HamiltonianSpikeModel(
            n_sources=n_sources, n_oscillators=1, hidden_dims=[8], seed=0,
            sampling_freq=1.0 / dt,
        )

        # Zero out MLP so dynamics are pure harmonic oscillator
        mlp_params = dict(model.mlp_params)
        for k in mlp_params:
            if k.startswith("w") or k.startswith("b"):
                mlp_params[k] = jnp.zeros_like(mlp_params[k])
        model.mlp_params = mlp_params
        model.omega = omega

        # Simulate ground truth trajectory
        trans_params = {**mlp_params, "omega": omega}
        x0 = jnp.array([1.0, 0.0])

        def sim_step(x, key):
            x_next = leapfrog_step(x, trans_params, apply_mlp, dt)
            x_next = x_next + jax.random.normal(key, x.shape) * 1e-3
            return x_next, x_next

        keys = jax.random.split(jax.random.PRNGKey(42), n_time)
        _, x_true = jax.lax.scan(sim_step, x0, keys)

        # Generate spikes: rate = exp(C @ x + d)
        C = model.C
        d = model.d
        log_rates = x_true @ C.T + d
        rates = jnp.exp(jnp.clip(log_rates, -5, 3)) * dt
        spikes = jax.random.poisson(jax.random.PRNGKey(99), rates)

        # Run smoother
        params = {
            "mlp": mlp_params,
            "omega": omega,
            "C": C,
            "d": d,
            "init_mean": x0,
        }
        m_s, _ = model.smooth(spikes, params)

        # Smoother MSE on position (q) should be less than prior MSE
        smoother_mse = float(jnp.mean((m_s[:, 0] - x_true[:, 0]) ** 2))
        prior_mse = float(jnp.mean(x_true[:, 0] ** 2))

        assert smoother_mse < prior_mse, (
            f"Smoother MSE ({smoother_mse:.4f}) should be less than "
            f"prior MSE ({prior_mse:.4f})"
        )


@pytest.mark.slow
class TestHamiltonianSpikeSGDRecovery:
    """Verify fit_sgd learns omega from spike observations."""

    @pytest.fixture(scope="class")
    def fitted(self):
        omega_true = 2 * jnp.pi
        dt = 0.01
        n_time = 300
        n_sources = 4

        x_true, mlp_params = simulate_harmonic_oscillator(
            omega=omega_true, n_time=n_time, dt=dt,
            key=jax.random.PRNGKey(42), hidden_dims=[8],
        )

        C_true = jax.random.normal(jax.random.PRNGKey(10), (n_sources, 2)) * 0.3
        d_true = -2.0 * jnp.ones(n_sources)
        spikes = simulate_poisson_spikes(
            x_true, C_true, d_true, dt=dt,
            key=jax.random.PRNGKey(77),
        )

        model = HamiltonianSpikeModel(
            n_sources=n_sources, n_oscillators=1, hidden_dims=[8], seed=0,
            sampling_freq=1.0 / dt,
        )
        model.omega = omega_true * 1.3

        lls = model.fit_sgd(
            spikes, key=jax.random.PRNGKey(1), num_steps=200,
            use_filter=False,
        )
        return model, x_true, omega_true, spikes, lls

    def test_ll_improves(self, fitted):
        _, _, _, _, lls = fitted
        assert_ll_improves(lls, label="HamiltonianSpike SGD")

    def test_frequency_recovery(self, fitted):
        model, _, omega_true, _, _ = fitted
        rel_error = float(abs(model.omega - omega_true) / omega_true)
        assert rel_error < 0.25, (
            f"Omega relative error {rel_error:.3f} >= 0.25 "
            f"(learned={float(model.omega):.3f}, true={float(omega_true):.3f})"
        )

    def test_smoother_output_finite(self, fitted):
        """After SGD learning, smoother should produce finite output."""
        model, x_true, _, spikes, _ = fitted
        params, _ = model._build_param_spec()
        m_s, _ = model.smooth(spikes, params)
        assert jnp.all(jnp.isfinite(m_s)), "Smoother produced non-finite values"
        assert m_s.shape == x_true.shape


@pytest.mark.slow
class TestHamiltonianVsLinearBaseline:
    """Nonlinear Hamiltonian model should beat a linear random-walk
    PointProcessModel on spike data from nonlinear latent dynamics.

    Closes V1 plan success criterion 5 / Task 6
    (docs/plans/2026-04-08-hamiltonian-oscillator-state-space-model.md):
    'The nonlinear Hamiltonian model outperforms a linear random-walk
    or linear oscillator baseline on at least one nonlinear synthetic
    benchmark.'
    """

    def test_nonlinear_beats_linear_on_duffing_oscillator(self):
        from state_space_practice.point_process_kalman import PointProcessModel

        dt = 0.01
        n_time = 400
        n_sources = 6
        omega = 2 * jnp.pi  # ~1 Hz base oscillation
        alpha = 4.0  # cubic stiffness → Duffing nonlinearity

        # Symplectic Euler integration of Duffing Hamiltonian:
        # H = p^2/2 + omega^2 q^2/2 + alpha q^4/4
        def duffing_step(x, key_i):
            q, p = x[0], x[1]
            p_new = p + dt * (-(omega ** 2) * q - alpha * q ** 3)
            q_new = q + dt * p_new
            x_new = jnp.array([q_new, p_new])
            x_new = x_new + jax.random.normal(key_i, (2,)) * 1e-3
            return x_new, x_new

        x0 = jnp.array([1.0, 0.0])
        keys = jax.random.split(jax.random.PRNGKey(42), n_time)
        _, x_true = jax.lax.scan(duffing_step, x0, keys)

        # Linear Poisson readout (both models see identical spikes).
        C_true = jax.random.normal(jax.random.PRNGKey(10), (n_sources, 2)) * 0.5
        d_true = -1.5 * jnp.ones(n_sources)
        spikes = simulate_poisson_spikes(
            x_true, C_true, d_true, dt=dt, key=jax.random.PRNGKey(77),
        )

        # --- Nonlinear Hamiltonian fit ---
        # Warm-start observation params and init state to match what the linear
        # baseline gets for free via the design matrix. The comparison isolates
        # the dynamics prior — both models share the same readout knowledge.
        nonlinear = HamiltonianSpikeModel(
            n_sources=n_sources, n_oscillators=1, hidden_dims=[8], seed=0,
            sampling_freq=1.0 / dt,
        )
        nonlinear.C = C_true
        nonlinear.d = d_true
        nonlinear.init_mean = x0[:, None]
        nonlinear.omega = omega  # seed frequency near truth
        nonlinear.fit_sgd(
            spikes, key=jax.random.PRNGKey(1), num_steps=200,
            use_filter=False,
        )
        nonlinear_ll = float(nonlinear.log_likelihood_)

        # --- Linear random-walk fit ---
        # Intercept absorbed as a constant third latent dimension:
        # log_rate[t, n] = C_true[n, :] @ q_p_t + 1.0 * bias_t.
        design_matrix = jnp.concatenate([
            jnp.broadcast_to(C_true[None, :, :], (n_time, n_sources, 2)),
            jnp.ones((n_time, n_sources, 1)),
        ], axis=-1)
        linear = PointProcessModel(
            n_state_dims=3, dt=dt,
            init_mean=jnp.array([x0[0], x0[1], float(d_true[0])]),
            init_cov=jnp.eye(3) * 0.1,
            process_cov=jnp.eye(3) * 1e-3,
        )
        linear.fit_sgd(design_matrix, spikes, num_steps=200)
        linear_ll = float(linear.log_likelihood_)

        assert nonlinear_ll > linear_ll, (
            f"Nonlinear Hamiltonian LL ({nonlinear_ll:.2f}) should exceed "
            f"linear random-walk baseline LL ({linear_ll:.2f}) on "
            f"Duffing-oscillator spike data."
        )
