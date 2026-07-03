"""Tests for the spike-field coupling recovery-validation harness.

All inputs are hand-constructed (no model or simulator), so these are fast and
exercise the metric formulas directly.
"""

import numpy as np
import pytest
from scipy import stats

from state_space_practice.coupling_validation import (
    detection_metrics,
    magnitude_recovery,
    phase_recovery_mae,
    roc_auc,
    summarize_posterior,
    wald_test,
)


class TestWaldTest:
    def test_significant_when_strong(self, make_coupling_posterior):
        """Large mean relative to small variance -> tiny p-value."""
        post = make_coupling_posterior(
            beta_real_mean=[[0.5]],
            beta_imag_mean=[[0.5]],
            beta_real_var=[[1e-3]],
            beta_imag_var=[[1e-3]],
        )
        W, pval = wald_test(post)
        # W = 0.25/1e-3 + 0.25/1e-3 = 500
        assert W[0, 0] == pytest.approx(500.0, rel=1e-6)  # guard: strong regime reached
        assert pval[0, 0] < 1e-3

    def test_null_when_zero(self, make_coupling_posterior):
        """Mean ~0 with finite variance -> p-value near 1."""
        post = make_coupling_posterior(
            beta_real_mean=[[0.0]],
            beta_imag_mean=[[0.0]],
            beta_real_var=[[1e-2]],
            beta_imag_var=[[1e-2]],
        )
        W, pval = wald_test(post)
        assert W[0, 0] < 0.5  # guard: genuinely in the null regime, not just p>0.5
        assert pval[0, 0] > 0.5

    def test_zero_variance_nonzero_mean_is_significant(self, make_coupling_posterior):
        """Zero variance floors denominator without erasing a strong signal."""
        post = make_coupling_posterior(
            beta_real_mean=[[0.5]],
            beta_imag_mean=[[0.5]],
            beta_real_var=[[0.0]],
            beta_imag_var=[[0.0]],
        )
        W, pval = wald_test(post)
        assert W[0, 0] > 1e9
        assert pval[0, 0] < 1e-12
        assert not np.isnan(pval[0, 0])

    def test_zero_variance_zero_mean_is_null(self, make_coupling_posterior):
        """Zero variance at exactly zero mean remains a null result."""
        post = make_coupling_posterior(
            beta_real_mean=[[0.0]],
            beta_imag_mean=[[0.0]],
            beta_real_var=[[0.0]],
            beta_imag_var=[[0.0]],
        )
        W, pval = wald_test(post)
        assert W[0, 0] == 0.0
        assert pval[0, 0] == 1.0
        assert not np.isnan(pval[0, 0])

    def test_rejects_negative_variance(self, make_coupling_posterior):
        post = make_coupling_posterior(
            beta_real_mean=[[0.0]],
            beta_imag_mean=[[0.0]],
            beta_real_var=[[-1.0]],
            beta_imag_var=[[1.0]],
        )
        with pytest.raises(ValueError, match="variance"):
            wald_test(post)


class TestDetectionMetrics:
    def test_known_confusion_matrix(self):
        """Hand-built pval/mask giving TP=2, FP=1, FN=1, TN=4."""
        pval = np.array(
            [
                [0.001, 0.001, 0.5, 0.5],  # TP, TP, TN, TN
                [0.001, 0.5, 0.5, 0.5],  # FP, FN, TN, TN
            ]
        )
        mask = np.array(
            [
                [True, True, False, False],
                [False, True, False, False],
            ]
        )
        m = detection_metrics(pval, mask, alpha=0.05)
        assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
        assert m["sensitivity"] == pytest.approx(2 / 3)
        assert m["specificity"] == pytest.approx(4 / 5)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(4 / 6)

    def test_band_view_any_neuron(self):
        """A band is detected if ANY neuron is significant for it."""
        pval = np.array([[0.5, 0.001], [0.5, 0.5]])  # band 1 significant via neuron 0
        mask = np.array([[False, True], [False, True]])
        m = detection_metrics(pval, mask, alpha=0.05)
        # band 0: not detected, not true -> TN; band 1: detected and true -> TP
        assert m["band_tp"] == 1
        assert m["band_tn"] == 1
        assert m["band_fp"] == 0
        assert m["band_fn"] == 0


class TestRocAuc:
    def test_perfect_separation(self):
        """Coupled entries get tiny p, controls p~1 -> AUC = 1."""
        pval = np.array([[1e-6, 0.9], [0.9, 0.9]])
        mask = np.array([[True, False], [False, False]])
        assert roc_auc(pval, mask) == pytest.approx(1.0)

    def test_single_class_returns_nan(self):
        """All-positive labels -> AUC undefined -> NaN (not a crash)."""
        pval = np.array([[1e-6, 1e-6]])
        mask = np.array([[True, True]])
        assert np.isnan(roc_auc(pval, mask))


class TestPhaseRecoveryMAE:
    def test_zero_when_exact(self, make_coupling_posterior, make_ground_truth):
        """Recovered beta == true beta -> phase MAE = 0."""
        br_true, bi_true, mask = make_ground_truth(
            beta_real_true=[[1.0, 0.0]],
            beta_imag_true=[[0.0, 1.0]],
            coupling_mask=[[True, True]],
        )
        post = make_coupling_posterior(
            beta_real_mean=br_true,
            beta_imag_mean=bi_true,
            beta_real_var=[[1e-3, 1e-3]],
            beta_imag_var=[[1e-3, 1e-3]],
        )
        assert int(mask.sum()) >= 1  # guard: at least one coupled entry scored
        assert phase_recovery_mae(post, br_true, bi_true, mask) == pytest.approx(0.0)

    def test_quarter_turn(self, make_coupling_posterior, make_ground_truth):
        """Recovered phase = true + pi/2 everywhere -> MAE = pi/2."""
        # true phases: band0 = 0 (1+0j), band1 = 0 (2+0j)
        br_true, bi_true, mask = make_ground_truth(
            beta_real_true=[[1.0, 2.0]],
            beta_imag_true=[[0.0, 0.0]],
            coupling_mask=[[True, True]],
        )
        # recovered rotated +pi/2: (r, 0) -> (0, r)
        post = make_coupling_posterior(
            beta_real_mean=[[0.0, 0.0]],
            beta_imag_mean=[[1.0, 2.0]],
            beta_real_var=[[1e-3, 1e-3]],
            beta_imag_var=[[1e-3, 1e-3]],
        )
        assert phase_recovery_mae(post, br_true, bi_true, mask) == pytest.approx(
            np.pi / 2, rel=1e-6
        )

    def test_only_coupled_entries_scored(
        self, make_coupling_posterior, make_ground_truth
    ):
        """Uncoupled (masked-out) entries must not contribute to the MAE."""
        br_true, bi_true, mask = make_ground_truth(
            beta_real_true=[[1.0, 1.0]],
            beta_imag_true=[[0.0, 0.0]],
            coupling_mask=[[True, False]],  # band 1 is a control
        )
        # band 0 exact (dist 0); band 1 wildly wrong but should be ignored
        post = make_coupling_posterior(
            beta_real_mean=[[1.0, -1.0]],
            beta_imag_mean=[[0.0, 0.0]],
            beta_real_var=[[1e-3, 1e-3]],
            beta_imag_var=[[1e-3, 1e-3]],
        )
        assert phase_recovery_mae(post, br_true, bi_true, mask) == pytest.approx(0.0)


class TestMagnitudeRecovery:
    def test_perfect_rank_correlation(self, make_coupling_posterior, make_ground_truth):
        """Recovered magnitudes monotonic in truth -> correlations near 1."""
        br_true, bi_true, mask = make_ground_truth(
            beta_real_true=[[0.1, 0.2, 0.3, 0.4]],
            beta_imag_true=[[0.0, 0.0, 0.0, 0.0]],
            coupling_mask=[[True, True, True, True]],
        )
        post = make_coupling_posterior(
            beta_real_mean=[[0.2, 0.4, 0.6, 0.8]],  # exactly 2x true magnitude
            beta_imag_mean=[[0.0, 0.0, 0.0, 0.0]],
            beta_real_var=[[1e-3, 1e-3, 1e-3, 1e-3]],
            beta_imag_var=[[1e-3, 1e-3, 1e-3, 1e-3]],
        )
        out = magnitude_recovery(post, br_true, bi_true, mask)
        assert out["n"] == 4
        assert out["pearson_r"] == pytest.approx(1.0, abs=1e-6)
        assert out["spearman_r"] == pytest.approx(1.0, abs=1e-6)

    def test_too_few_points_returns_nan(
        self, make_coupling_posterior, make_ground_truth
    ):
        """Fewer than 3 coupled entries -> correlation is NaN, not an error."""
        br_true, bi_true, mask = make_ground_truth(
            beta_real_true=[[0.1, 0.2]],
            beta_imag_true=[[0.0, 0.0]],
            coupling_mask=[[True, False]],  # only 1 coupled entry
        )
        post = make_coupling_posterior(
            beta_real_mean=[[0.2, 0.4]],
            beta_imag_mean=[[0.0, 0.0]],
            beta_real_var=[[1e-3, 1e-3]],
            beta_imag_var=[[1e-3, 1e-3]],
        )
        out = magnitude_recovery(post, br_true, bi_true, mask)
        assert out["n"] == 1
        assert np.isnan(out["pearson_r"])

    def test_constant_input_returns_nan(
        self, make_coupling_posterior, make_ground_truth
    ):
        """Constant magnitudes (zero variance) -> NaN, no warning escalated to error."""
        br_true, bi_true, mask = make_ground_truth(
            beta_real_true=[[0.3, 0.3, 0.3]],  # all-equal true magnitudes
            beta_imag_true=[[0.0, 0.0, 0.0]],
            coupling_mask=[[True, True, True]],
        )
        post = make_coupling_posterior(
            beta_real_mean=[[0.1, 0.5, 0.9]],
            beta_imag_mean=[[0.0, 0.0, 0.0]],
            beta_real_var=[[1e-3, 1e-3, 1e-3]],
            beta_imag_var=[[1e-3, 1e-3, 1e-3]],
        )
        out = magnitude_recovery(post, br_true, bi_true, mask)
        assert out["n"] == 3  # guard: the >=3-point branch is reached
        assert np.isnan(out["pearson_r"])
        assert np.isnan(out["spearman_r"])


class TestSummarizePosterior:
    def test_magnitude_and_phase(self, make_coupling_posterior):
        post = make_coupling_posterior(
            beta_real_mean=[[3.0]],
            beta_imag_mean=[[4.0]],
            beta_real_var=[[1e-3]],
            beta_imag_var=[[1e-3]],
        )
        summary = summarize_posterior(post)
        assert summary["magnitude"][0, 0] == pytest.approx(5.0)
        assert summary["phase"][0, 0] == pytest.approx(np.arctan2(4.0, 3.0))

    def test_gaussian_ci_matches_sample_ci(self, make_coupling_posterior):
        """Percentile CI from normal samples ~ Gaussian CI from (mean, var)."""
        rng = np.random.default_rng(0)
        mean_r, sd_r = 1.0, 0.2
        mean_i, sd_i = 0.5, 0.3
        n = 40_000
        real = rng.normal(mean_r, sd_r, size=n)
        imag = rng.normal(mean_i, sd_i, size=n)
        samples = (real + 1j * imag).reshape(n, 1, 1)

        post_samp = make_coupling_posterior(
            beta_real_mean=[[mean_r]],
            beta_imag_mean=[[mean_i]],
            beta_real_var=[[sd_r**2]],
            beta_imag_var=[[sd_i**2]],
            samples=samples,
        )
        post_gauss = make_coupling_posterior(
            beta_real_mean=[[mean_r]],
            beta_imag_mean=[[mean_i]],
            beta_real_var=[[sd_r**2]],
            beta_imag_var=[[sd_i**2]],
        )
        s_samp = summarize_posterior(post_samp, cred_mass=0.95)
        s_gauss = summarize_posterior(post_gauss, cred_mass=0.95)

        for key in (
            "beta_real_ci_lower",
            "beta_real_ci_upper",
            "beta_imag_ci_lower",
            "beta_imag_ci_upper",
        ):
            assert s_samp[key][0, 0] == pytest.approx(s_gauss[key][0, 0], abs=0.03)
        # guard: the interval is non-degenerate (the comparison is meaningful)
        assert s_gauss["beta_real_ci_upper"][0, 0] > s_gauss["beta_real_ci_lower"][0, 0]

    def test_gaussian_ci_formula(self, make_coupling_posterior):
        """Gaussian branch is exactly mean +/- z * sqrt(var), pinned independently."""
        post = make_coupling_posterior(
            beta_real_mean=[[1.0]],
            beta_imag_mean=[[2.0]],
            beta_real_var=[[0.04]],  # sd 0.2
            beta_imag_var=[[0.09]],  # sd 0.3
        )
        s = summarize_posterior(post, cred_mass=0.95)
        z = stats.norm.ppf(0.975)
        assert s["beta_real_ci_lower"][0, 0] == pytest.approx(1.0 - z * 0.2)
        assert s["beta_real_ci_upper"][0, 0] == pytest.approx(1.0 + z * 0.2)
        assert s["beta_imag_ci_lower"][0, 0] == pytest.approx(2.0 - z * 0.3)
        assert s["beta_imag_ci_upper"][0, 0] == pytest.approx(2.0 + z * 0.3)
