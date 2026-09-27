"""Tests for the spike-field coupling recovery-validation harness.

All inputs are hand-constructed (no model or simulator), so these are fast and
exercise the metric formulas directly.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from state_space_practice.coupling_validation import (
    CouplingPosterior,
    batch_means_mcse,
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

    def test_uses_real_imag_covariance(self, make_coupling_posterior):
        """The Wald statistic uses the full 2x2 covariance, not only marginals."""
        post = make_coupling_posterior(
            beta_real_mean=[[1.0]],
            beta_imag_mean=[[1.0]],
            beta_real_var=[[1.0]],
            beta_imag_var=[[1.0]],
            beta_real_imag_cov=[[0.5]],
        )
        W, pval = wald_test(post)
        assert W[0, 0] == pytest.approx(4.0 / 3.0)
        assert pval[0, 0] == pytest.approx(stats.chi2.sf(4.0 / 3.0, df=2))

    def test_rejects_negative_variance(self, make_coupling_posterior):
        post = make_coupling_posterior(
            beta_real_mean=[[0.0]],
            beta_imag_mean=[[0.0]],
            beta_real_var=[[-1.0]],
            beta_imag_var=[[1.0]],
        )
        with pytest.raises(ValueError, match="variance"):
            wald_test(post)

    def test_rejects_inconsistent_covariance(self, make_coupling_posterior):
        post = make_coupling_posterior(
            beta_real_mean=[[0.0]],
            beta_imag_mean=[[0.0]],
            beta_real_var=[[1.0]],
            beta_imag_var=[[1.0]],
            beta_real_imag_cov=[[2.0]],
        )
        with pytest.raises(ValueError, match="covariance"):
            wald_test(post)

    def test_rejects_non_2d_posterior_arrays(self, make_coupling_posterior):
        post = make_coupling_posterior(
            beta_real_mean=[0.0],
            beta_imag_mean=[0.0],
            beta_real_var=[1.0],
            beta_imag_var=[1.0],
        )
        with pytest.raises(ValueError, match="2D"):
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

    @pytest.mark.parametrize(
        "pval",
        [
            np.array([[np.nan]]),
            np.array([[-0.1]]),
            np.array([[1.1]]),
        ],
    )
    def test_rejects_invalid_pvalues(self, pval):
        with pytest.raises(ValueError, match="pval"):
            detection_metrics(pval, np.array([[True]]))

    def test_rejects_shape_mismatch(self):
        pval = np.array([[0.01], [0.5]])
        mask = np.array([[True]])
        with pytest.raises(ValueError, match="matching shapes"):
            detection_metrics(pval, mask)

    @pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5, np.nan])
    def test_rejects_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            detection_metrics(np.array([[0.01]]), np.array([[True]]), alpha=alpha)


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

    def test_rejects_invalid_pvalues(self):
        pval = np.array([[-0.1, 0.2]])
        mask = np.array([[True, False]])
        with pytest.raises(ValueError, match="pval"):
            roc_auc(pval, mask)


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

    @pytest.mark.parametrize("cred_mass", [-0.1, 0.0, 1.0, 1.5, np.nan])
    def test_rejects_invalid_cred_mass(self, make_coupling_posterior, cred_mass):
        post = make_coupling_posterior(
            beta_real_mean=[[1.0]],
            beta_imag_mean=[[2.0]],
            beta_real_var=[[0.04]],
            beta_imag_var=[[0.09]],
        )
        with pytest.raises(ValueError, match="cred_mass"):
            summarize_posterior(post, cred_mass=cred_mass)

    def test_rejects_negative_variance(self, make_coupling_posterior):
        post = make_coupling_posterior(
            beta_real_mean=[[1.0]],
            beta_imag_mean=[[2.0]],
            beta_real_var=[[-0.04]],
            beta_imag_var=[[0.09]],
        )
        with pytest.raises(ValueError, match="variance"):
            summarize_posterior(post)

    def test_rejects_sample_shape_mismatch(self, make_coupling_posterior):
        post = make_coupling_posterior(
            beta_real_mean=[[1.0]],
            beta_imag_mean=[[2.0]],
            beta_real_var=[[0.04]],
            beta_imag_var=[[0.09]],
            samples=np.ones((10, 2, 1), dtype=np.complex128),
        )
        with pytest.raises(ValueError, match="samples"):
            summarize_posterior(post)


# --- Hand computations and scipy.stats references ---------------------------
#
# Every metric is a pure function of small arrays, so Hypothesis draws the
# arrays and each test recomputes the metric from its definition with code
# that shares nothing with the implementation.

_shape = st.tuples(st.integers(1, 3), st.integers(1, 3))


@st.composite
def _gaussian_posteriors(draw):
    """Random (S, J) Gaussian posteriors with a valid real/imag covariance."""
    n_neurons, n_bands = draw(_shape)
    size = n_neurons * n_bands
    finite = dict(allow_nan=False, allow_infinity=False)

    def block(lo, hi):
        values = draw(
            st.lists(st.floats(lo, hi, **finite), min_size=size, max_size=size)
        )
        return np.asarray(values).reshape(n_neurons, n_bands)

    var_real, var_imag = block(1e-3, 10.0), block(1e-3, 10.0)
    corr = block(-0.95, 0.95)
    return CouplingPosterior(
        beta_real_mean=block(-5.0, 5.0),
        beta_imag_mean=block(-5.0, 5.0),
        beta_real_var=var_real,
        beta_imag_var=var_imag,
        beta_real_imag_cov=corr * np.sqrt(var_real * var_imag),
    )


class TestAgainstReferenceComputations:
    @given(_gaussian_posteriors())
    def test_wald_is_the_quadratic_form_with_chi2_2_tail(self, post):
        """W = m' Sigma^-1 m per entry; the chi2(2) tail is exactly exp(-W/2)."""
        W, pval = wald_test(post)
        for s_idx, j_idx in np.ndindex(W.shape):
            m = np.array(
                [post.beta_real_mean[s_idx, j_idx], post.beta_imag_mean[s_idx, j_idx]]
            )
            c = post.beta_real_imag_cov[s_idx, j_idx]
            sigma = np.array(
                [
                    [post.beta_real_var[s_idx, j_idx], c],
                    [c, post.beta_imag_var[s_idx, j_idx]],
                ]
            )
            w_ref = float(m @ np.linalg.solve(sigma, m))
            assert W[s_idx, j_idx] == pytest.approx(w_ref, rel=1e-8, abs=1e-12)
            assert pval[s_idx, j_idx] == pytest.approx(
                np.exp(-0.5 * w_ref), rel=1e-6, abs=1e-300
            )

    def test_wald_pvalues_uniform_under_null(self):
        """Means drawn from N(0, Sigma) give Uniform(0,1) p-values (KS test)."""
        rng = np.random.default_rng(0)
        n = 4000
        var_real, var_imag, rho = 0.5, 2.0, 0.7
        cov = rho * np.sqrt(var_real * var_imag)
        sigma = np.array([[var_real, cov], [cov, var_imag]])
        means = rng.multivariate_normal(np.zeros(2), sigma, size=n)
        post = CouplingPosterior(
            beta_real_mean=means[:, :1],
            beta_imag_mean=means[:, 1:],
            beta_real_var=np.full((n, 1), var_real),
            beta_imag_var=np.full((n, 1), var_imag),
            beta_real_imag_cov=np.full((n, 1), cov),
        )
        _, pval = wald_test(post)
        assert stats.kstest(pval.ravel(), "uniform").pvalue > 1e-3
        # power guard: dropping the covariance (diagonal Wald) is miscalibrated
        diag = post._replace(beta_real_imag_cov=None)
        _, pval_diag = wald_test(diag)
        assert stats.kstest(pval_diag.ravel(), "uniform").pvalue < 1e-6

    @given(_gaussian_posteriors(), st.floats(0.5, 0.99))
    def test_gaussian_interval_matches_scipy_norm_interval(self, post, cred_mass):
        summary = summarize_posterior(post, cred_mass=cred_mass)
        lo, hi = stats.norm.interval(
            cred_mass, loc=post.beta_real_mean, scale=np.sqrt(post.beta_real_var)
        )
        np.testing.assert_allclose(summary["beta_real_ci_lower"], lo, rtol=1e-10)
        np.testing.assert_allclose(summary["beta_real_ci_upper"], hi, rtol=1e-10)
        lo, hi = stats.norm.interval(
            cred_mass, loc=post.beta_imag_mean, scale=np.sqrt(post.beta_imag_var)
        )
        np.testing.assert_allclose(summary["beta_imag_ci_lower"], lo, rtol=1e-10)
        np.testing.assert_allclose(summary["beta_imag_ci_upper"], hi, rtol=1e-10)
        np.testing.assert_allclose(
            summary["magnitude"],
            np.abs(post.beta_real_mean + 1j * post.beta_imag_mean),
        )

    @given(st.integers(0, 2**32 - 1), st.floats(0.5, 0.95))
    def test_sample_interval_is_the_central_empirical_quantile(self, seed, cred_mass):
        """Percentile CI == sorted-sample order statistics (linear interpolation)."""
        rng = np.random.default_rng(seed)
        n = 201
        samples = rng.normal(size=(n, 2, 1)) + 1j * rng.gamma(2.0, size=(n, 2, 1))
        post = CouplingPosterior(
            beta_real_mean=samples.real.mean(0),
            beta_imag_mean=samples.imag.mean(0),
            beta_real_var=samples.real.var(0),
            beta_imag_var=samples.imag.var(0),
            samples=samples,
        )
        summary = summarize_posterior(post, cred_mass=cred_mass)
        tail = (1.0 - cred_mass) / 2.0
        for part, key in ((samples.real, "beta_real"), (samples.imag, "beta_imag")):
            ordered = np.sort(part, axis=0)
            for q, suffix in ((tail, "lower"), (1.0 - tail, "upper")):
                position = q * (n - 1)
                below = int(np.floor(position))
                frac = position - below
                ref = ordered[below] + frac * (
                    ordered[min(below + 1, n - 1)] - ordered[below]
                )
                np.testing.assert_allclose(
                    summary[f"{key}_ci_{suffix}"], ref, rtol=1e-12
                )

    @given(
        _shape.flatmap(
            lambda shp: st.tuples(
                st.lists(
                    st.booleans(), min_size=shp[0] * shp[1], max_size=shp[0] * shp[1]
                ),
                st.lists(
                    st.sampled_from([0.0, 0.001, 0.04, 0.05, 0.2, 1.0]),
                    min_size=shp[0] * shp[1],
                    max_size=shp[0] * shp[1],
                ),
                st.just(shp),
            )
        ),
        st.sampled_from([0.01, 0.05, 0.1]),
    )
    def test_detection_counts_match_loop(self, data, alpha):
        mask_list, pval_list, shp = data
        mask = np.asarray(mask_list).reshape(shp)
        pval = np.asarray(pval_list).reshape(shp)
        out = detection_metrics(pval, mask, alpha=alpha)
        tp = fp = fn = tn = 0
        for is_true, p in zip(mask_list, pval_list):
            hit = p < alpha
            tp += hit and is_true
            fp += hit and not is_true
            fn += (not hit) and is_true
            tn += (not hit) and not is_true
        assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (tp, fp, fn, tn)
        if tp + fp + fn > 0:
            assert out["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
        else:
            assert np.isnan(out["f1"])
        if tp + fn > 0:
            assert out["sensitivity"] == pytest.approx(tp / (tp + fn))
        band_hit = [
            any(pval[s, j] < alpha for s in range(shp[0])) for j in range(shp[1])
        ]
        band_true = [any(mask[s, j] for s in range(shp[0])) for j in range(shp[1])]
        assert out["band_tp"] == sum(h and t for h, t in zip(band_hit, band_true))
        assert out["band_fp"] == sum(h and not t for h, t in zip(band_hit, band_true))

    @given(st.integers(0, 2**32 - 1))
    def test_roc_auc_matches_mann_whitney_u(self, seed):
        """AUC = U / (n_pos n_neg), with U from scipy.stats.mannwhitneyu (ties 1/2)."""
        rng = np.random.default_rng(seed)
        n_neurons, n_bands = 4, 3
        mask = rng.random((n_neurons, n_bands)) < 0.5
        mask[0, 0], mask[0, 1] = True, False  # both classes present
        # coarse p-value alphabet -> ties are common and must count 1/2
        pval = rng.choice([1e-12, 0.001, 0.03, 0.3, 0.3, 1.0], size=mask.shape)
        auc = roc_auc(pval, mask)
        # score = -log10(p): larger score for coupled entries is a "win"
        score = -np.log10(pval)
        u_stat = stats.mannwhitneyu(score[mask], score[~mask]).statistic
        assert auc == pytest.approx(u_stat / (mask.sum() * (~mask).sum()), abs=1e-12)

    @given(
        st.lists(st.floats(-10.0, 10.0), min_size=3, max_size=6),
        st.lists(st.floats(-10.0, 10.0), min_size=3, max_size=6),
    )
    def test_phase_mae_is_mean_wrapped_distance(self, phases_hat, phases_true):
        n = min(len(phases_hat), len(phases_true))
        est = np.asarray(phases_hat[:n])[None, :]
        tru = np.asarray(phases_true[:n])[None, :]
        post = CouplingPosterior(
            beta_real_mean=np.cos(est),
            beta_imag_mean=np.sin(est),
            beta_real_var=np.ones_like(est),
            beta_imag_var=np.ones_like(est),
        )
        mae = phase_recovery_mae(
            post, 2.0 * np.cos(tru), 2.0 * np.sin(tru), np.ones_like(est, bool)
        )
        diff = np.mod(est - tru, 2.0 * np.pi)
        ref = float(np.mean(np.minimum(diff, 2.0 * np.pi - diff)))
        assert mae == pytest.approx(ref, abs=1e-9)

    @given(st.integers(0, 2**32 - 1))
    def test_magnitude_correlations_match_definition(self, seed):
        rng = np.random.default_rng(seed)
        mask = np.ones((2, 3), bool)
        mask[1, 2] = False
        est_r, est_i, tru_r, tru_i = rng.normal(size=(4, 2, 3))
        post = CouplingPosterior(
            beta_real_mean=est_r,
            beta_imag_mean=est_i,
            beta_real_var=np.ones((2, 3)),
            beta_imag_var=np.ones((2, 3)),
        )
        out = magnitude_recovery(post, tru_r, tru_i, mask)
        a = np.sqrt(est_r**2 + est_i**2)[mask]
        b = np.sqrt(tru_r**2 + tru_i**2)[mask]
        assert out["n"] == 5
        assert out["pearson_r"] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-10)
        ranks_a = np.argsort(np.argsort(a))
        ranks_b = np.argsort(np.argsort(b))
        assert out["spearman_r"] == pytest.approx(
            np.corrcoef(ranks_a, ranks_b)[0, 1], abs=1e-10
        )


class TestBatchMeansMCSE:
    def test_hand_computed_example(self):
        # batches [1,2,3,4] and [5,6,7,8] -> means 2.5, 6.5; sd = 2*sqrt(2);
        # MCSE = sd / sqrt(2) = 2. The trailing draw 9 is truncated.
        draws = np.arange(1.0, 10.0)
        assert float(batch_means_mcse(draws, n_batches=2)) == pytest.approx(2.0)

    def test_matches_iid_and_ar1_closed_forms(self):
        """iid: sd/sqrt(n). AR(1) rho: sd sqrt((1+rho)/((1-rho) n)) (tau = 19)."""
        rng = np.random.default_rng(1)
        n, rho = 200_000, 0.9
        iid = rng.normal(size=n)
        assert float(batch_means_mcse(iid, 40)) == pytest.approx(
            1.0 / np.sqrt(n), rel=0.3
        )
        noise = rng.normal(size=n) * np.sqrt(1.0 - rho**2)
        ar = np.empty(n)
        ar[0] = rng.normal()
        for k in range(1, n):
            ar[k] = rho * ar[k - 1] + noise[k]
        expected = np.sqrt((1.0 + rho) / (1.0 - rho) / n)
        got = float(batch_means_mcse(ar, 40))
        assert got == pytest.approx(expected, rel=0.3)
        # guard: the naive iid formula would be 4.4x too small here
        assert got > 3.0 / np.sqrt(n)

    def test_vectorised_over_trailing_axes(self):
        draws = np.random.default_rng(2).normal(size=(400, 2, 3))
        out = batch_means_mcse(draws, 20)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[1, 2], batch_means_mcse(draws[:, 1, 2], 20))

    @pytest.mark.parametrize("n_batches", [1, 0, 2.5, True])
    def test_rejects_bad_batch_count(self, n_batches):
        with pytest.raises(ValueError, match="n_batches"):
            batch_means_mcse(np.zeros(10), n_batches)

    def test_rejects_too_few_draws_and_nonfinite(self):
        with pytest.raises(ValueError, match="at least"):
            batch_means_mcse(np.zeros(3), 5)
        with pytest.raises(ValueError, match="finite"):
            batch_means_mcse(np.array([0.0, np.nan, 1.0, 2.0]), 2)
