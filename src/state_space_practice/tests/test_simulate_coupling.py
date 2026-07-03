"""Tests for the Bernoulli-logistic coupling model and its simulator.

Behavioral, not shape-only: the simulator tests assert statistical properties
(base rate, phase locking, null controls), the spec for a correct generative
model.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import state_space_practice.coupling_model as coupling_model
from state_space_practice.circular_stats import (
    angular_distance,
    circular_mean,
    mean_resultant_length,
    rayleigh_test,
)
from state_space_practice.coupling_model import (
    CouplingModelParams,
    build_transition,
    deinterleave_coupling,
    interleave_coupling,
    logit,
    smooth_latent_from_lfp,
    validate_coupling_params,
)
from state_space_practice.simulate_coupling import simulate_coupling


def _spike_triggered_phase(sim, neuron: int, band: int) -> np.ndarray:
    """Latent phase of ``band`` at the bins where ``neuron`` spiked."""
    spikes = np.asarray(sim.spikes[:, neuron])
    re = np.asarray(sim.latent_true[:, 2 * band])
    im = np.asarray(sim.latent_true[:, 2 * band + 1])
    phase = np.arctan2(im, re)
    return phase[spikes > 0.5]


class TestModelMaps:
    def test_logit_matches_manual(self):
        """eta = baseline + beta_real @ re + beta_imag @ im, computed by hand."""
        params = CouplingModelParams(
            osc_frequencies=jnp.array([6.0, 10.0]),
            osc_decay=jnp.array([0.99, 0.99]),
            process_noise_var=jnp.array([0.02, 0.02]),
            beta_real=jnp.array([[3.0, 4.0]]),
            beta_imag=jnp.array([[5.0, 6.0]]),
            baseline=jnp.array([0.1]),
            dt=1e-3,
        )
        state = jnp.array([1.0, 0.5, -1.0, 2.0])  # band0=(1,0.5), band1=(-1,2)
        # 0.1 + (3*1 + 4*-1) + (5*0.5 + 6*2) = 0.1 - 1 + 14.5 = 13.6
        assert float(logit(state, params)[0]) == pytest.approx(13.6)

    def test_build_transition_rotates_and_decays(self):
        """A single oscillator block rotates by 2*pi*f*dt and scales by decay."""
        # freq chosen purely so the per-step rotation is exactly pi/2 (a clean
        # math check); 250 Hz is deliberately non-physiologic, not a real band.
        dt = 1e-3
        freq = 0.25 / dt  # 2*pi*f*dt = pi/2
        params = CouplingModelParams(
            osc_frequencies=jnp.array([freq]),
            osc_decay=jnp.array([0.5]),
            process_noise_var=jnp.array([0.01]),
            beta_real=jnp.zeros((1, 1)),
            beta_imag=jnp.zeros((1, 1)),
            baseline=jnp.zeros((1,)),
            dt=dt,
        )
        A, _ = build_transition(params)
        assert A.shape == (2, 2)
        # decay 0.5, rotate pi/2: (1, 0) -> 0.5 * (cos pi/2, sin pi/2) = (0, 0.5)
        np.testing.assert_allclose(
            np.asarray(A @ jnp.array([1.0, 0.0])), [0.0, 0.5], atol=1e-9
        )

    def test_build_transition_process_cov(self, coupling_params_small):
        """Process covariance is diag(repeat(process_noise_var, 2))."""
        _, Q = build_transition(coupling_params_small)
        var = np.asarray(coupling_params_small.process_noise_var)
        np.testing.assert_allclose(np.asarray(Q), np.diag(np.repeat(var, 2)))


class TestInterleaveCoupling:
    def test_interleave_layout(self):
        """(S,J) real/imag blocks -> [bR_0, bI_0, bR_1, bI_1, ...]."""
        design = interleave_coupling(
            beta_real=jnp.array([[1.0, 3.0]]), beta_imag=jnp.array([[2.0, 4.0]])
        )
        np.testing.assert_array_equal(np.asarray(design), [[1.0, 2.0, 3.0, 4.0]])

    def test_roundtrip(self):
        """deinterleave(interleave(bR, bI)) == (bR, bI)."""
        br = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bi = jnp.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
        out_r, out_i = deinterleave_coupling(interleave_coupling(br, bi))
        np.testing.assert_array_equal(np.asarray(out_r), np.asarray(br))
        np.testing.assert_array_equal(np.asarray(out_i), np.asarray(bi))

    def test_design_row_matches_logit_coupling(self, coupling_params_small):
        """interleave(beta) @ state equals the coupling term of logit().

        Pins the single-source-of-truth contract: the interleaved design row and
        the model logit use the same band-interleaved layout, so the EKF/PG
        design rows cannot drift from the simulator's logit.
        """
        params = coupling_params_small
        state = jnp.arange(1.0, 2 * params.osc_frequencies.shape[0] + 1)  # (2J,)
        design = interleave_coupling(params.beta_real, params.beta_imag)
        coupling_term = np.asarray(design @ state)
        expected = np.asarray(logit(state, params) - params.baseline)
        np.testing.assert_allclose(coupling_term, expected, atol=1e-12)


class TestSmoothLatentFromLFP:
    def test_smoothed_tracks_truth_better_than_raw_lfp(
        self, simulated_coupling_small, coupling_params_small
    ):
        """The LFP smoother denoises toward the true latent (lower MAE than raw LFP)."""
        sim = simulated_coupling_small
        smoothed = np.asarray(smooth_latent_from_lfp(sim.lfp, coupling_params_small))
        truth = np.asarray(sim.latent_true)
        mae_smoothed = np.mean(np.abs(smoothed - truth))
        mae_raw = np.mean(np.abs(np.asarray(sim.lfp) - truth))
        assert smoothed.shape == truth.shape
        assert mae_smoothed < mae_raw  # smoothing genuinely recovers the latent


class TestSimulatorBasics:
    def test_spikes_binary(self, simulated_coupling_small):
        assert set(np.unique(np.asarray(simulated_coupling_small.spikes))) <= {0.0, 1.0}

    def test_coupling_mask_matches_beta(self, simulated_coupling_small):
        # hand-written oracle: all 3 neurons couple to band 0, band 1 is control
        expected = np.array([[True, False], [True, False], [True, False]])
        np.testing.assert_array_equal(
            np.asarray(simulated_coupling_small.coupling_mask), expected
        )

    def test_lfp_is_latent_plus_noise(
        self, simulated_coupling_small, coupling_params_small
    ):
        """The LFP is the latent observed through N(0, lfp_noise_var) noise."""
        sim = simulated_coupling_small
        assert sim.lfp.shape == sim.latent_true.shape
        resid = np.asarray(sim.lfp) - np.asarray(sim.latent_true)
        np.testing.assert_allclose(resid.mean(axis=0), 0.0, atol=0.05)
        np.testing.assert_allclose(
            resid.var(axis=0), coupling_params_small.lfp_noise_var, rtol=0.15
        )

    def test_determinism(self, coupling_params_small):
        a = simulate_coupling(coupling_params_small, n_time=1000, seed=5)
        b = simulate_coupling(coupling_params_small, n_time=1000, seed=5)
        c = simulate_coupling(coupling_params_small, n_time=1000, seed=6)
        assert np.array_equal(np.asarray(a.spikes), np.asarray(b.spikes))
        np.testing.assert_array_equal(
            np.asarray(a.latent_true), np.asarray(b.latent_true)
        )
        # different seed -> different spikes (guard: simulation actually uses the seed)
        assert not np.array_equal(np.asarray(a.spikes), np.asarray(c.spikes))

    def test_history_kernel_not_implemented(self, coupling_params_small):
        params = coupling_params_small._replace(history_kernel=jnp.zeros((3, 2)))
        with pytest.raises(NotImplementedError):
            simulate_coupling(params, n_time=100, seed=0)

    def test_rate_matches_baseline_when_uncoupled(self, coupling_params_small):
        """With beta = 0 the spike rate is sigmoid(baseline) = 0.05 per bin."""
        params = coupling_params_small._replace(
            beta_real=jnp.zeros((3, 2)), beta_imag=jnp.zeros((3, 2))
        )
        sim = simulate_coupling(params, n_time=5000, seed=1)
        rate = float(np.mean(np.asarray(sim.spikes)))
        # binomial SE = sqrt(p(1-p)/N), N = 5000*3
        se = np.sqrt(0.05 * 0.95 / (5000 * 3))
        assert rate == pytest.approx(0.05, abs=4 * se)


@pytest.mark.slow
class TestPhaseLocking:
    def test_spikes_phase_lock_to_coupled_band(self, simulated_coupling_small):
        """Neuron 0's spikes concentrate in latent phase of the coupled band 0."""
        phases = _spike_triggered_phase(simulated_coupling_small, neuron=0, band=0)
        assert phases.size >= 50  # guard: enough spikes to test concentration
        _, pval = rayleigh_test(phases)
        assert pval < 0.01
        assert mean_resultant_length(phases) > 0.1

    def test_preferred_phase_matches_beta_phase(
        self, simulated_coupling_small, coupling_params_small
    ):
        """Preferred latent phase at spikes equals the coupling phase arctan2(bI, bR)."""
        for neuron in (0, 1):  # neuron 0 -> phase 0, neuron 1 -> phase pi/2
            phases = _spike_triggered_phase(
                simulated_coupling_small, neuron=neuron, band=0
            )
            assert phases.size >= 50
            true_phase = np.arctan2(
                float(coupling_params_small.beta_imag[neuron, 0]),
                float(coupling_params_small.beta_real[neuron, 0]),
            )
            assert float(angular_distance(circular_mean(phases), true_phase)) < 0.3

    def test_control_band_weaker_than_coupled(self, simulated_coupling_small):
        """Locking to the coupled band dominates locking to the control band.

        The control band's spike-triggered MRL is not ~0: the spikes are rhythmic
        (locked to the 6 Hz coupled band), so sampling the independent 10 Hz
        control band at that rhythm induces a stroboscopic (cross-frequency)
        concentration. The genuine, robust invariant is that the truly-coupled
        band dominates this artifact, not that the control MRL vanishes. The 2.5x
        margin is calibrated to ``simulated_coupling_small`` (seed 0, n_time 5000);
        a change to that fixture is a trigger to re-check it.
        """
        coupled = mean_resultant_length(
            _spike_triggered_phase(simulated_coupling_small, neuron=0, band=0)
        )
        control = mean_resultant_length(
            _spike_triggered_phase(simulated_coupling_small, neuron=0, band=1)
        )
        assert coupled > 0.4  # guard: genuinely, strongly locked to its band
        assert coupled > 2.5 * control  # dominates the stroboscopic floor

    def test_stronger_coupling_more_locking(
        self, simulated_coupling_small, coupling_params_small
    ):
        """Larger coupling magnitude yields stronger phase concentration."""
        weak = coupling_params_small._replace(
            beta_real=coupling_params_small.beta_real * 0.15,
            beta_imag=coupling_params_small.beta_imag * 0.15,
        )
        sim_weak = simulate_coupling(weak, n_time=5000, seed=0)
        mrl_strong = mean_resultant_length(
            _spike_triggered_phase(simulated_coupling_small, neuron=0, band=0)
        )
        mrl_weak = mean_resultant_length(
            _spike_triggered_phase(sim_weak, neuron=0, band=0)
        )
        assert mrl_strong > mrl_weak
        assert mrl_strong > 0.1  # guard: the strong case is genuinely locked


class TestSimulatorDynamics:
    def test_latent_follows_ar1_recursion(
        self, simulated_coupling_small, coupling_params_small
    ):
        """latent[k] = A @ latent[k-1] + noise: residuals are white with covariance Q.

        Directly verifies the scan core (catches a transposed/misapplied A or a
        dropped step), which the statistical phase-locking tests cannot.
        """
        A, Q = build_transition(coupling_params_small)
        latent = np.asarray(simulated_coupling_small.latent_true)
        # latent rows are states; A @ x for a batch is X @ A.T
        resid = latent[1:] - latent[:-1] @ np.asarray(A).T  # = the process noise
        np.testing.assert_allclose(resid.mean(axis=0), 0.0, atol=0.02)
        np.testing.assert_allclose(
            np.cov(resid, rowvar=False), np.asarray(Q), atol=0.01
        )

    @pytest.mark.slow
    def test_initial_state_is_stationary(self, coupling_params_small):
        """latent[0] is drawn from the stationary distribution, not a warm-up start."""
        params = coupling_params_small
        stationary_var = np.asarray(params.process_noise_var) / (
            1.0 - np.asarray(params.osc_decay) ** 2
        )
        first = np.array(
            [
                np.asarray(simulate_coupling(params, 2, s).latent_true[0])
                for s in range(200)
            ]
        )
        np.testing.assert_allclose(
            first.var(axis=0), np.repeat(stationary_var, 2), rtol=0.2
        )


class TestValidation:
    def test_accepts_canonical_params(self, coupling_params_small):
        # guard: the valid fixture passes (validation isn't rejecting good input)
        validate_coupling_params(coupling_params_small)

    def test_requires_jax_x64(self, coupling_params_small, monkeypatch):
        monkeypatch.setattr(coupling_model, "_coupling_x64_enabled", lambda: False)
        with pytest.raises(RuntimeError, match="x64"):
            validate_coupling_params(coupling_params_small)
        with pytest.raises(RuntimeError, match="x64"):
            build_transition(coupling_params_small)

    def test_rejects_decay_equal_one(self, coupling_params_small):
        # decay=1 has no stationary distribution -> would silently produce inf/NaN
        bad = coupling_params_small._replace(osc_decay=jnp.array([1.0, 0.99]))
        with pytest.raises(ValueError, match="osc_decay"):
            simulate_coupling(bad, n_time=100, seed=0)

    def test_rejects_negative_process_noise(self, coupling_params_small):
        bad = coupling_params_small._replace(process_noise_var=jnp.array([-1.0, 0.02]))
        with pytest.raises(ValueError, match="process_noise_var"):
            simulate_coupling(bad, n_time=100, seed=0)

    @pytest.mark.parametrize(
        "process_noise_var",
        [jnp.array([jnp.nan, 0.02]), jnp.array([jnp.inf, 0.02])],
    )
    def test_rejects_nonfinite_process_noise(
        self, coupling_params_small, process_noise_var
    ):
        bad = coupling_params_small._replace(process_noise_var=process_noise_var)
        with pytest.raises(ValueError, match="process_noise_var"):
            validate_coupling_params(bad)

    def test_rejects_shape_mismatch(self, coupling_params_small):
        # baseline length 1 but S = 3: previously broadcast silently
        bad = coupling_params_small._replace(baseline=jnp.zeros((1,)))
        with pytest.raises(ValueError, match="baseline"):
            simulate_coupling(bad, n_time=100, seed=0)

    def test_rejects_nonpositive_dt(self, coupling_params_small):
        with pytest.raises(ValueError, match="dt"):
            simulate_coupling(
                coupling_params_small._replace(dt=0.0), n_time=100, seed=0
            )

    def test_rejects_nonfinite_dt(self, coupling_params_small):
        with pytest.raises(ValueError, match="dt"):
            validate_coupling_params(coupling_params_small._replace(dt=float("inf")))

    def test_rejects_nonpositive_n_time(self, coupling_params_small):
        with pytest.raises(ValueError, match="n_time"):
            simulate_coupling(coupling_params_small, n_time=0, seed=0)

    def test_rejects_nonpositive_lfp_noise(self, coupling_params_small):
        with pytest.raises(ValueError, match="lfp_noise_var"):
            simulate_coupling(
                coupling_params_small._replace(lfp_noise_var=0.0), n_time=100, seed=0
            )

    def test_rejects_nonfinite_lfp_noise(self, coupling_params_small):
        with pytest.raises(ValueError, match="lfp_noise_var"):
            validate_coupling_params(
                coupling_params_small._replace(lfp_noise_var=float("inf"))
            )
