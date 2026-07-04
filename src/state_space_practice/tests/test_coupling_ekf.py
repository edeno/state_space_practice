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

    def test_rejects_history_kernel(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        bad = coupling_params_small._replace(history_kernel=np.zeros((3, 2)))
        with pytest.raises(NotImplementedError, match="history_kernel"):
            fit_coupling_ekf(sim.spikes, sim.lfp, bad)

    @pytest.mark.parametrize("bad_value", [0.5, -1.0, 2.0, np.nan])
    def test_rejects_invalid_spike_values(self, coupling_params_small, bad_value):
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        spikes = np.asarray(sim.spikes).copy()
        spikes[0, 0] = bad_value
        with pytest.raises(ValueError, match="0/1"):
            fit_coupling_ekf(spikes, sim.lfp, coupling_params_small)

    def test_rejects_nonfinite_lfp(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        lfp = np.asarray(sim.lfp).copy()
        lfp[0, 0] = np.nan
        with pytest.raises(ValueError, match="lfp"):
            fit_coupling_ekf(sim.spikes, lfp, coupling_params_small)

    @pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_invalid_sigma_beta(self, coupling_params_small, bad_value):
        # sigma_beta sets the prior covariance sigma_beta**2 * I; a non-positive
        # or non-finite value produces a degenerate / non-PSD prior.
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        with pytest.raises(ValueError, match="sigma_beta"):
            fit_coupling_ekf(
                sim.spikes, sim.lfp, coupling_params_small, sigma_beta=bad_value
            )

    @pytest.mark.parametrize("bad_value", [0, -1, 2.5])
    def test_rejects_invalid_max_newton_iter(
        self, coupling_params_small, bad_value
    ):
        # Must be a positive integer: 0/-1 give no Fisher-scoring steps, and a
        # float would silently truncate the iteration count.
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        with pytest.raises(ValueError, match="max_newton_iter"):
            fit_coupling_ekf(
                sim.spikes,
                sim.lfp,
                coupling_params_small,
                max_newton_iter=bad_value,
            )


class TestMechanics:
    def test_returns_valid_posterior(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=500, seed=0)
        post = fit_coupling_ekf(sim.spikes, sim.lfp, coupling_params_small)
        n_neurons, n_bands = np.asarray(coupling_params_small.beta_real).shape
        assert post.beta_real_mean.shape == (n_neurons, n_bands)
        assert post.beta_real_imag_cov.shape == (n_neurons, n_bands)
        assert post.samples is None
        for arr in (
            post.beta_real_mean,
            post.beta_imag_mean,
            post.beta_real_var,
            post.beta_imag_var,
            post.beta_real_imag_cov,
        ):
            assert np.all(np.isfinite(np.asarray(arr)))
        # variances are positive (the Wald test downstream needs this)
        var_r = np.asarray(post.beta_real_var)
        var_i = np.asarray(post.beta_imag_var)
        assert np.all(var_r > 0)
        assert np.all(var_i > 0)
        # The real-imag cross-covariance is the off-diagonal of the PSD
        # posterior covariance, so it must satisfy the Cauchy-Schwarz bound
        # cov^2 <= var_r * var_i. A shape/finiteness check alone would miss an
        # indexing or sign error that produced an inconsistent off-diagonal.
        cov_ri = np.asarray(post.beta_real_imag_cov)
        assert np.all(cov_ri**2 <= var_r * var_i + 1e-12)
        # Guard: the bound is only meaningful if the posterior actually has
        # real-imag correlation; an all-zero cross-cov (e.g. wrong slice) would
        # pass the bound vacuously.
        assert np.any(np.abs(cov_ri) > 1e-6)


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
        # alpha=0.01: with 3 control entries the family-wise false-positive rate at
        # 0.05 is ~14%, so a borderline control fluctuation (p~0.05) flips `fp` by
        # luck. 0.01 controls FWER here, while true couplings have p~0.
        det = detection_metrics(pval, mask, alpha=0.01)
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
