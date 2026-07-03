"""Tests for the EKF-vs-PG cross-check harness."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from state_space_practice.coupling_crosscheck import (
    _score,
    aggregate,
    run_crosscheck,
    scale_coupling,
)
from state_space_practice.coupling_validation import CouplingPosterior


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
            out = aggregate([
                {
                    "coupling_mag": 0.0,
                    "replicate": 0,
                    "ekf": score,
                    "pg": score,
                    "ekf_pg_mean_maxdiff": 0.0,
                }
            ])
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
