"""Tests for the EKF-vs-PG cross-check harness."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from state_space_practice.coupling_crosscheck import (
    _mean_disagreement,
    _score,
    aggregate,
    run_crosscheck,
    scale_coupling,
)
from state_space_practice.coupling_ekf import fit_coupling_ekf
from state_space_practice.coupling_pg import fit_coupling_pg
from state_space_practice.coupling_validation import (
    CouplingPosterior,
    batch_means_mcse,
)
from state_space_practice.simulate_coupling import simulate_coupling


def _fake_sim(beta_real_true, beta_imag_true, mask):
    return SimpleNamespace(
        coupling_mask=np.asarray(mask, dtype=bool),
        beta_real_true=np.asarray(beta_real_true, dtype=float),
        beta_imag_true=np.asarray(beta_imag_true, dtype=float),
    )


class TestScore:
    def test_perfect_recovery(self):
        """Mean == truth -> zero bias, full coverage, expected Gaussian CI width."""
        post = CouplingPosterior(
            beta_real_mean=np.array([[1.0]]),
            beta_imag_mean=np.array([[0.0]]),
            beta_real_var=np.array([[0.01]]),
            beta_imag_var=np.array([[0.01]]),
            samples=None,
        )
        sim = _fake_sim([[1.0]], [[0.0]], [[True]])
        s = _score(post, sim)
        assert s["abs_bias"] == pytest.approx(0.0, abs=1e-9)
        assert s["coverage95"] == 1.0
        # Gaussian 95% width = 2 * 1.95996 * sd
        assert s["ci_width"] == pytest.approx(2 * 1.959964 * 0.1, rel=1e-3)
        assert s["phase_mae"] == pytest.approx(0.0, abs=1e-9)

    def test_coverage_detects_miss(self):
        """A real mean far from truth (tight CI) drops coverage to 0.5 (imag still ok)."""
        post = CouplingPosterior(
            beta_real_mean=np.array([[1.0]]),
            beta_imag_mean=np.array([[0.0]]),
            beta_real_var=np.array([[1e-4]]),  # tight: CI excludes the true 5.0
            beta_imag_var=np.array([[1e-4]]),
            samples=None,
        )
        sim = _fake_sim([[5.0]], [[0.0]], [[True]])
        s = _score(post, sim)
        assert s["coverage95"] == 0.5  # real component missed, imag covered

    def test_all_null_mask_is_quietly_undefined(self):
        """A zero-coupling sweep has no coupled-entry bias to average."""
        post = CouplingPosterior(
            beta_real_mean=np.zeros((2, 2)),
            beta_imag_mean=np.zeros((2, 2)),
            beta_real_var=np.ones((2, 2)),
            beta_imag_var=np.ones((2, 2)),
            samples=None,
        )
        sim = _fake_sim(
            beta_real_true=np.zeros((2, 2)),
            beta_imag_true=np.zeros((2, 2)),
            mask=np.zeros((2, 2), dtype=bool),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            score = _score(post, sim)
            out = aggregate(
                [
                    {
                        "coupling_mag": 0.0,
                        "replicate": 0,
                        "ekf": score,
                        "pg": score,
                        "ekf_pg_mean_maxdiff": 0.0,
                    }
                ]
            )
        assert np.isnan(score["abs_bias"])
        assert np.isnan(out[0.0]["ekf"]["abs_bias"])


class TestAggregate:
    def test_groups_and_averages(self):
        records = [
            {
                "coupling_mag": 1.0,
                "replicate": 0,
                "ekf": {"abs_bias": 0.1},
                "pg": {"abs_bias": 0.2},
                "ekf_pg_mean_maxdiff": 0.05,
                "latent_correlation": 0.8,
                "latent_rmse": 0.2,
                "latent_variance_ratio": 0.7,
            },
            {
                "coupling_mag": 1.0,
                "replicate": 1,
                "ekf": {"abs_bias": 0.3},
                "pg": {"abs_bias": 0.4},
                "ekf_pg_mean_maxdiff": 0.07,
                "latent_correlation": 0.9,
                "latent_rmse": 0.4,
                "latent_variance_ratio": 0.9,
            },
        ]
        agg = aggregate(records)
        assert agg[1.0]["n"] == 2
        assert agg[1.0]["ekf"]["abs_bias"] == pytest.approx(0.2)
        assert agg[1.0]["pg"]["abs_bias"] == pytest.approx(0.3)
        assert agg[1.0]["ekf_pg_mean_maxdiff"] == pytest.approx(0.06)
        assert agg[1.0]["latent_correlation"] == pytest.approx(0.85)
        assert agg[1.0]["latent_rmse"] == pytest.approx(0.3)
        assert agg[1.0]["latent_variance_ratio"] == pytest.approx(0.8)

    def test_rejects_empty_records(self):
        with pytest.raises(ValueError, match="records"):
            aggregate([])


class TestScaleCoupling:
    def test_scales_magnitude(self, coupling_params_small):
        scaled = scale_coupling(coupling_params_small, 0.5)
        np.testing.assert_allclose(
            np.asarray(scaled.beta_real),
            0.5 * np.asarray(coupling_params_small.beta_real),
        )
        # non-coupling fields untouched
        np.testing.assert_array_equal(
            np.asarray(scaled.osc_frequencies),
            np.asarray(coupling_params_small.osc_frequencies),
        )

    @pytest.mark.parametrize("scale", [-1.0, np.nan])
    def test_rejects_invalid_scale(self, coupling_params_small, scale):
        with pytest.raises(ValueError, match="scale"):
            scale_coupling(coupling_params_small, scale)


class TestRunCrosscheckGuards:
    def test_rejects_empty_scales(self, coupling_params_small):
        with pytest.raises(ValueError, match="scales"):
            run_crosscheck(
                coupling_params_small,
                scales=[],
                n_time=10,
                n_replicates=1,
                pg_n_iter=4,
                pg_burn_in=2,
            )

    def test_rejects_zero_replicates(self, coupling_params_small):
        with pytest.raises(ValueError, match="n_replicates"):
            run_crosscheck(
                coupling_params_small,
                scales=[1.0],
                n_time=10,
                n_replicates=0,
                pg_n_iter=4,
                pg_burn_in=2,
            )

    def test_rejects_negative_scales(self, coupling_params_small):
        with pytest.raises(ValueError, match="scales"):
            run_crosscheck(
                coupling_params_small,
                scales=[-1.0],
                n_time=10,
                n_replicates=1,
                pg_n_iter=4,
                pg_burn_in=2,
            )


@pytest.mark.slow
class TestIntegration:
    def test_runs_and_methods_agree_on_strong_cell(self, coupling_params_small):
        records = run_crosscheck(
            coupling_params_small,
            scales=[1.0],
            n_time=4000,
            n_replicates=1,
            pg_n_iter=200,
            pg_burn_in=100,
        )
        assert len(records) == 1
        rec = records[0]
        for method in ("ekf", "pg"):
            for value in rec[method].values():
                assert np.isfinite(value)
        for key in ("latent_correlation", "latent_rmse", "latent_variance_ratio"):
            assert np.isfinite(rec[key])
        agg = aggregate(records)
        agg_key = round(rec["coupling_mag"], 4)
        for key in ("latent_correlation", "latent_rmse", "latent_variance_ratio"):
            assert np.isfinite(agg[agg_key][key])
        # strong coupling: both detect perfectly and the two methods agree closely
        assert rec["ekf"]["detection_auc"] == 1.0
        assert rec["pg"]["detection_auc"] == 1.0
        assert rec["ekf_pg_mean_maxdiff"] < 0.15


def _posterior_from_samples(samples):
    return CouplingPosterior(
        beta_real_mean=samples.real.mean(0),
        beta_imag_mean=samples.imag.mean(0),
        beta_real_var=samples.real.var(0),
        beta_imag_var=samples.imag.var(0),
        samples=samples,
    )


class TestMeanDisagreement:
    def test_z_is_difference_over_batch_means_mcse(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=(400, 2, 1)) + 1j * rng.normal(size=(400, 2, 1))
        pg = _posterior_from_samples(samples)
        shift = np.array([[0.3], [-0.05]])
        ekf = pg._replace(beta_real_mean=pg.beta_real_mean + shift, samples=None)
        out = _mean_disagreement(ekf, pg)
        mcse_real = batch_means_mcse(samples.real, 20)
        mcse_imag = batch_means_mcse(samples.imag, 20)
        assert out["ekf_pg_mean_maxdiff"] == pytest.approx(0.3)
        assert out["pg_mean_mcse_max"] == pytest.approx(
            max(mcse_real.max(), mcse_imag.max())
        )
        # imaginary parts agree exactly, so the max z is the real part's
        assert out["ekf_pg_mean_max_z"] == pytest.approx(
            np.max(np.abs(shift) / mcse_real)
        )
        assert out["ekf_pg_mean_max_z"] > 4.0  # guard: 0.3 >> MCSE ~ 0.05

    def test_mcse_undefined_without_enough_samples(self):
        rng = np.random.default_rng(1)
        pg = _posterior_from_samples(rng.normal(size=(39, 1, 1)) + 0j)
        out = _mean_disagreement(pg, pg)
        assert out["ekf_pg_mean_maxdiff"] == 0.0
        assert np.isnan(out["pg_mean_mcse_max"])
        assert np.isnan(out["ekf_pg_mean_max_z"])


@pytest.mark.slow
class TestAgreementTracksWhereApproximationsCoincide:
    """The cross-check must pass where EKF == PG and fail where they differ.

    Both arms fit the *same* static model on the same design, so they differ
    only by the Laplace approximation. With lots of data (T = 4000, ~500 spikes
    per neuron) the posterior is Gaussian to within ~0.05 sd, so the EKF mean
    should sit within PG Monte Carlo error: each of the 12 per-component
    z = |EKF - PG| / MCSE is ~|N(0, 1)|, and max z > 4.5 has probability
    ~1e-4. With ~20 spikes per neuron (T = 150) the posterior is skewed; the
    mode (EKF) and the mean (PG) differ by ~0.1-0.3 sd, which a long chain
    resolves at many MCSE. Measured: max z 2.4 (maxdiff 0.009, MCSE 0.005) on
    the long static cell; max z 10.8 (maxdiff 0.24, MCSE 0.05) on the short one.
    """

    def test_static_long_data_agrees_within_monte_carlo_error(
        self, coupling_params_small
    ):
        (rec,) = run_crosscheck(
            coupling_params_small,
            scales=[1.0],
            n_time=4000,
            n_replicates=1,
            pg_n_iter=1000,
            pg_burn_in=200,
        )
        # guard: the MCSE is small, so "agreement" is a sharp statement
        assert rec["pg_mean_mcse_max"] < 0.01, rec
        assert rec["ekf_pg_mean_max_z"] < 4.5, rec

    def test_short_data_disagreement_exceeds_monte_carlo_error(
        self, coupling_params_small
    ):
        (rec,) = run_crosscheck(
            coupling_params_small,
            scales=[1.0],
            n_time=150,
            n_replicates=1,
            pg_n_iter=8000,
            pg_burn_in=200,
        )
        assert np.isfinite(rec["pg_mean_mcse_max"]), rec
        assert rec["ekf_pg_mean_max_z"] > 5.0, rec

    def test_record_reproduces_independent_fits(self, coupling_params_small):
        """Record == fits rerun with the documented seeds (cell seed = seed + r)."""
        (rec,) = run_crosscheck(
            coupling_params_small,
            scales=[1.0],
            n_time=300,
            n_replicates=1,
            seed=7,
            pg_n_iter=100,
            pg_burn_in=50,
        )
        sim = simulate_coupling(coupling_params_small, n_time=300, seed=7)
        ekf = fit_coupling_ekf(sim.spikes, sim.lfp, coupling_params_small)
        pg = fit_coupling_pg(
            sim.spikes, sim.lfp, coupling_params_small, n_iter=100, burn_in=50, seed=7
        )
        ref = _mean_disagreement(ekf, pg)
        for key, value in ref.items():
            assert rec[key] == pytest.approx(value, rel=1e-12), key
