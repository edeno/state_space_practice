"""Tests for the Laplace-EKF coupling estimator (LFP smooth + Bernoulli regress)."""

import numpy as np
import pytest

from state_space_practice.coupling_ekf import fit_coupling_ekf
from state_space_practice.coupling_validation import (
    detection_metrics,
    phase_recovery_mae,
    wald_test,
)
from state_space_practice.simulate_coupling import simulate_coupling


class TestGuards:
    def test_rejects_spike_shape_mismatch(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        # spikes has fewer neurons than params -> would silently drop neurons
        with pytest.raises(ValueError, match="spikes"):
            fit_coupling_ekf(sim.spikes[:, :2], sim.lfp, coupling_params_small)

    def test_rejects_lfp_shape_mismatch(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        with pytest.raises(ValueError, match="lfp"):
            fit_coupling_ekf(sim.spikes, sim.lfp[:, :2], coupling_params_small)

    def test_rejects_invalid_params(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        bad = coupling_params_small._replace(lfp_noise_var=0.0)
        with pytest.raises(ValueError, match="lfp_noise_var"):
            fit_coupling_ekf(sim.spikes, sim.lfp, bad)


class TestMechanics:
    def test_returns_valid_posterior(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=500, seed=0)
        post = fit_coupling_ekf(sim.spikes, sim.lfp, coupling_params_small)
        n_neurons, n_bands = np.asarray(coupling_params_small.beta_real).shape
        assert post.beta_real_mean.shape == (n_neurons, n_bands)
        assert post.samples is None
        for arr in (
            post.beta_real_mean,
            post.beta_imag_mean,
            post.beta_real_var,
            post.beta_imag_var,
        ):
            assert np.all(np.isfinite(np.asarray(arr)))
        # variances are positive (the Wald test downstream needs this)
        assert np.all(np.asarray(post.beta_real_var) > 0)
        assert np.all(np.asarray(post.beta_imag_var) > 0)


@pytest.mark.slow
class TestRecovery:
    def test_recovers_coupling(self, coupling_params_small):
        """On a strong sim the EKF detects coupled bands, nulls controls, and
        recovers magnitude and phase."""
        sim = simulate_coupling(coupling_params_small, n_time=6000, seed=0)
        post = fit_coupling_ekf(sim.spikes, sim.lfp, coupling_params_small)
        mask = np.asarray(sim.coupling_mask)
        assert mask.any() and not mask.all()  # guard: coupled AND control bands exist

        _, pval = wald_test(post)
        det = detection_metrics(pval, mask)
        assert det["fp"] == 0  # no control band flagged
        assert det["f1"] == 1.0  # every coupled band detected

        mae = phase_recovery_mae(
            post,
            np.asarray(sim.beta_real_true),
            np.asarray(sim.beta_imag_true),
            mask,
        )
        assert mae < 0.3  # preferred phase recovered

        mag = np.sqrt(
            np.asarray(post.beta_real_mean) ** 2 + np.asarray(post.beta_imag_mean) ** 2
        )
        # coupled magnitude ~2 (the fixture's strong coupling); controls near zero
        np.testing.assert_allclose(mag[mask], 2.0, atol=0.4)
        assert np.all(mag[~mask] < 0.5)
