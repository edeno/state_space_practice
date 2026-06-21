"""Tests for the Polya-Gamma Gibbs coupling estimator."""

import numpy as np
import pytest
from polyagamma import random_polyagamma

from state_space_practice.coupling_ekf import fit_coupling_ekf
from state_space_practice.coupling_pg import fit_coupling_pg
from state_space_practice.coupling_validation import (
    detection_metrics,
    phase_recovery_mae,
    summarize_posterior,
    wald_test,
)
from state_space_practice.simulate_coupling import simulate_coupling


class TestPolyaGammaSampler:
    """Trust-but-verify the external sampler before relying on it."""

    @pytest.mark.parametrize("z", [0.0, 0.5, 2.0, -3.0])
    def test_moment_matches_theory(self, z):
        # E[PG(1, z)] = tanh(z/2) / (2z), and 1/4 at z = 0
        rng = np.random.default_rng(0)
        draws = random_polyagamma(h=1.0, z=z, size=100_000, random_state=rng)
        theory = 0.25 if z == 0 else np.tanh(z / 2) / (2 * z)
        assert float(draws.mean()) == pytest.approx(theory, rel=0.03)


class TestGuards:
    def test_rejects_spike_shape_mismatch(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        with pytest.raises(ValueError, match="spikes"):
            fit_coupling_pg(
                sim.spikes[:, :2], sim.lfp, coupling_params_small, n_iter=10, burn_in=5
            )

    def test_rejects_bad_burn_in(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        with pytest.raises(ValueError, match="burn_in"):
            fit_coupling_pg(
                sim.spikes, sim.lfp, coupling_params_small, n_iter=10, burn_in=10
            )

    def test_rejects_invalid_params(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=200, seed=0)
        bad = coupling_params_small._replace(lfp_noise_var=0.0)
        with pytest.raises(ValueError, match="lfp_noise_var"):
            fit_coupling_pg(sim.spikes, sim.lfp, bad, n_iter=10, burn_in=5)


class TestMechanics:
    def test_returns_posterior_with_samples(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=500, seed=0)
        post = fit_coupling_pg(
            sim.spikes, sim.lfp, coupling_params_small, n_iter=60, burn_in=30, seed=0
        )
        n_neurons, n_bands = np.asarray(coupling_params_small.beta_real).shape
        assert post.samples is not None
        assert post.samples.shape == (30, n_neurons, n_bands)
        assert np.iscomplexobj(post.samples)
        assert post.beta_real_mean.shape == (n_neurons, n_bands)
        for arr in (
            post.beta_real_mean,
            post.beta_imag_mean,
            post.beta_real_var,
            post.beta_imag_var,
        ):
            assert np.all(np.isfinite(np.asarray(arr)))
        assert np.all(np.asarray(post.beta_real_var) > 0)

    def test_mean_var_match_samples(self, coupling_params_small):
        """The mean/var fields are exactly the sample mean/var (reduction contract)."""
        sim = simulate_coupling(coupling_params_small, n_time=500, seed=0)
        post = fit_coupling_pg(
            sim.spikes, sim.lfp, coupling_params_small, n_iter=60, burn_in=30, seed=0
        )
        np.testing.assert_allclose(post.beta_real_mean, post.samples.real.mean(axis=0))
        np.testing.assert_allclose(post.beta_imag_mean, post.samples.imag.mean(axis=0))
        np.testing.assert_allclose(post.beta_real_var, post.samples.real.var(axis=0))
        np.testing.assert_allclose(post.beta_imag_var, post.samples.imag.var(axis=0))

    def test_deterministic(self, coupling_params_small):
        """Same seed -> identical samples; different seed -> different."""
        sim = simulate_coupling(coupling_params_small, n_time=400, seed=0)
        kw = dict(n_iter=40, burn_in=20)
        a = fit_coupling_pg(sim.spikes, sim.lfp, coupling_params_small, seed=7, **kw)
        b = fit_coupling_pg(sim.spikes, sim.lfp, coupling_params_small, seed=7, **kw)
        c = fit_coupling_pg(sim.spikes, sim.lfp, coupling_params_small, seed=8, **kw)
        np.testing.assert_array_equal(a.samples, b.samples)
        assert not np.array_equal(np.asarray(a.samples), np.asarray(c.samples))


@pytest.mark.slow
class TestRecovery:
    def test_recovers_and_agrees_with_ekf(self, coupling_params_small):
        sim = simulate_coupling(coupling_params_small, n_time=6000, seed=0)
        pg = fit_coupling_pg(
            sim.spikes, sim.lfp, coupling_params_small, n_iter=300, burn_in=150, seed=0
        )
        mask = np.asarray(sim.coupling_mask)
        assert mask.any() and not mask.all()  # guard: coupled AND control bands

        _, pval = wald_test(pg)
        # alpha=0.01: with 3 control entries the family-wise false-positive rate at
        # 0.05 is ~14%, so a borderline control fluctuation (p~0.05) flips `fp` by
        # luck. 0.01 controls FWER here, while true couplings have p~0.
        det = detection_metrics(pval, mask, alpha=0.01)
        assert det["fp"] == 0
        assert det["f1"] == 1.0

        mae = phase_recovery_mae(
            pg, np.asarray(sim.beta_real_true), np.asarray(sim.beta_imag_true), mask
        )
        assert mae < 0.3

        mag = np.sqrt(
            np.asarray(pg.beta_real_mean) ** 2 + np.asarray(pg.beta_imag_mean) ** 2
        )
        np.testing.assert_allclose(mag[mask], 2.0, atol=0.4)
        assert np.all(mag[~mask] < 0.5)

        # exact (sample-based) percentile credible intervals are available
        summary = summarize_posterior(pg)
        assert summary["beta_real_ci_lower"].shape == mask.shape

        # in this strong, near-Gaussian regime the Laplace and exact posteriors
        # should agree closely (Phase 5 will quantify where they diverge)
        ekf = fit_coupling_ekf(sim.spikes, sim.lfp, coupling_params_small)
        np.testing.assert_allclose(
            np.asarray(pg.beta_real_mean), np.asarray(ekf.beta_real_mean), atol=0.1
        )
        np.testing.assert_allclose(
            np.asarray(pg.beta_imag_mean), np.asarray(ekf.beta_imag_mean), atol=0.1
        )
        # and the posterior SPREAD agrees too (the sampler covariance is correct,
        # not just the mean): PG sample sd ~ EKF Laplace sd in this regime.
        np.testing.assert_allclose(
            np.sqrt(np.asarray(pg.beta_real_var)),
            np.sqrt(np.asarray(ekf.beta_real_var)),
            rtol=0.25,
        )
        np.testing.assert_allclose(
            np.sqrt(np.asarray(pg.beta_imag_var)),
            np.sqrt(np.asarray(ekf.beta_imag_var)),
            rtol=0.25,
        )

    def test_independent_chains_agree(self, coupling_params_small):
        """Two chains (different seeds) on the same data agree -> mixed, burn-in ok."""
        sim = simulate_coupling(coupling_params_small, n_time=6000, seed=0)
        kw = dict(n_iter=300, burn_in=150)
        a = fit_coupling_pg(sim.spikes, sim.lfp, coupling_params_small, seed=1, **kw)
        b = fit_coupling_pg(sim.spikes, sim.lfp, coupling_params_small, seed=2, **kw)
        np.testing.assert_allclose(
            np.asarray(a.beta_real_mean), np.asarray(b.beta_real_mean), atol=0.05
        )
        np.testing.assert_allclose(
            np.asarray(a.beta_imag_mean), np.asarray(b.beta_imag_mean), atol=0.05
        )
