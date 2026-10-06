"""Calibration and recovery statistics for the spike-field coupling estimators.

Three layers, kept apart because they fail for different reasons:

1. **Stage-2 calibration (conditional on the design).** Coupling vectors are
   drawn from the prior and spikes from the model *given the LFP-smoothed design
   the estimators condition on*. The exact posterior is then calibrated by
   construction (averaged over the prior), so pooled standardised errors must be
   N(0, 1) and 90% intervals must cover 90%, up to Monte Carlo error. This
   isolates the Laplace approximation (EKF arm) from the plug-in.
2. **End-to-end calibration against simulated truth.** Both estimators plug in
   the smoothed latent mean as a fixed design and ignore its uncertainty
   (errors-in-variables). The posterior mean stays essentially unbiased but the
   intervals are too narrow, increasingly so as ``lfp_noise_var`` grows; the
   tests pin where.
3. **Recovery sweeps.** Multi-seed, two-length error statistics on data from
   :func:`simulate_coupling`.

Replicates are packed as independent neurons of one fit (the estimators fit each
neuron separately on a shared design), which is statistically identical to
separate fits and an order of magnitude cheaper.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from state_space_practice.coupling_ekf import fit_coupling_ekf
from state_space_practice.coupling_model import (
    CouplingModelParams,
    smooth_latent_from_lfp,
)
from state_space_practice.coupling_pg import fit_coupling_pg
from state_space_practice.coupling_validation import summarize_posterior
from state_space_practice.simulate_coupling import simulate_coupling
from state_space_practice.tests.test_oracle_coupling import GridPosterior

Z90 = float(stats.norm.ppf(0.95))


def _one_band_params(
    n_neurons: int, base_rate: float, beta=(0.0, 0.0), lfp_noise_var: float = 0.25
) -> CouplingModelParams:
    return CouplingModelParams(
        osc_frequencies=jnp.array([8.0]),
        osc_decay=jnp.array([0.98]),
        process_noise_var=jnp.array([1.0 - 0.98**2]),
        beta_real=jnp.full((n_neurons, 1), beta[0]),
        beta_imag=jnp.full((n_neurons, 1), beta[1]),
        baseline=jnp.full((n_neurons,), float(special.logit(base_rate))),
        dt=1e-3,
        lfp_noise_var=lfp_noise_var,
    )


def _prior_predictive_replicates(n_time, base_rate, sigma_beta, n_fits, n_neurons):
    """Yield (params, lfp, spikes, beta_true) with beta ~ N(0, sigma_beta^2 I).

    One LFP (hence one design) per regime; each fit packs ``n_neurons``
    independent (beta, y) replicates drawn given that design.
    """
    params = _one_band_params(n_neurons, base_rate)
    sim = simulate_coupling(params, n_time=n_time, seed=0)
    design = np.asarray(smooth_latent_from_lfp(sim.lfp, params))
    offset = float(params.baseline[0])
    rng = np.random.default_rng(123)
    for _ in range(n_fits):
        beta = rng.normal(scale=sigma_beta, size=(n_neurons, 2))
        rate = special.expit(offset + design @ beta.T)  # (T, S)
        spikes = (rng.random(rate.shape) < rate).astype(float)
        yield params, sim.lfp, spikes, beta


def _standardised_errors(post, beta_true):
    mean = np.stack([post.beta_real_mean[:, 0], post.beta_imag_mean[:, 0]], 1)
    sd = np.sqrt(np.stack([post.beta_real_var[:, 0], post.beta_imag_var[:, 0]], 1))
    return (beta_true - mean) / sd


def _interval_hits(post, beta_true, cred_mass=0.90):
    s = summarize_posterior(post, cred_mass=cred_mass)
    real = (s["beta_real_ci_lower"][:, 0] <= beta_true[:, 0]) & (
        beta_true[:, 0] <= s["beta_real_ci_upper"][:, 0]
    )
    imag = (s["beta_imag_ci_lower"][:, 0] <= beta_true[:, 1]) & (
        beta_true[:, 1] <= s["beta_imag_ci_upper"][:, 0]
    )
    return np.stack([real, imag], 1)


def _binomial_band(p: float, n: int, k: float = 4.0) -> tuple[float, float]:
    half = k * np.sqrt(p * (1.0 - p) / n)
    return p - half, p + half


@pytest.mark.slow
class TestStage2Calibration:
    def test_laplace_calibrated_in_moderate_information_regime(self):
        """T = 1000, 20% base rate, sigma_beta = 1: z ~ N(0,1), 90% covers 90%.

        400 prior-predictive replicates (800 standardised errors). Bands use
        n = 400 (one effective draw per replicate) so they stay conservative if
        the real/imag errors of a replicate are correlated. Measured: mean z
        -0.03, var z 1.08, coverage 0.87 (within the n = 400 band 0.84-0.96).
        """
        zs, hits = [], []
        for params, lfp, spikes, beta in _prior_predictive_replicates(
            1000, 0.2, 1.0, n_fits=50, n_neurons=8
        ):
            post = fit_coupling_ekf(spikes, lfp, params, sigma_beta=1.0)
            zs.append(_standardised_errors(post, beta))
            hits.append(_interval_hits(post, beta))
        z = np.concatenate(zs).ravel()
        coverage = float(np.concatenate(hits).mean())
        n_eff = z.size // 2
        msg = f"mean z={z.mean():.3f} var z={z.var():.3f} coverage90={coverage:.3f}"
        assert abs(z.mean()) < 4.0 / np.sqrt(n_eff), msg
        lo, hi = stats.chi2.ppf([1e-4, 1 - 1e-4], n_eff) / n_eff
        assert lo < z.var() < hi, msg
        cov_lo, cov_hi = _binomial_band(0.9, n_eff)
        assert cov_lo < coverage < cov_hi, msg
        # Gaussian tails too: pooled z passes a KS test against N(0, 1)
        assert stats.kstest(z, "norm").pvalue > 1e-3, msg

    def test_laplace_undercovers_where_posterior_is_skewed(self):
        """T = 30, 5% base rate, sigma_beta = 5: pin the Laplace calibration gap.

        On the same 480 prior-predictive replicates, the exact posterior's 90%
        marginal intervals (grid quadrature, calibrated by construction) must
        cover within 4 binomial SE of 0.9 (measured 0.905) -- this validates the
        harness. The Laplace intervals under-cover (measured 0.80, z variance
        1.81: few
        spikes and a diffuse prior give skewed, near-separated posteriors whose
        long tail the Gaussian misses). Pins keep headroom on both sides.
        """
        ekf_z, ekf_hits, exact_hits = [], [], []
        for params, lfp, spikes, beta in _prior_predictive_replicates(
            30, 0.05, 5.0, n_fits=60, n_neurons=8
        ):
            ekf = fit_coupling_ekf(spikes, lfp, params, sigma_beta=5.0)
            ekf_z.append(_standardised_errors(ekf, beta))
            ekf_hits.append(_interval_hits(ekf, beta))
            design = np.asarray(smooth_latent_from_lfp(lfp, params))
            offset = float(params.baseline[0])
            for neuron in range(spikes.shape[1]):
                grid = GridPosterior(
                    design, spikes[:, neuron], offset, 5.0, n_points=121
                )
                exact_hits.append(
                    [
                        grid.quantile(a, 0.05)
                        <= beta[neuron, a]
                        <= grid.quantile(a, 0.95)
                        for a in range(2)
                    ]
                )
        z = np.concatenate(ekf_z).ravel()
        ekf_cov = float(np.concatenate(ekf_hits).mean())
        exact_cov = float(np.mean(exact_hits))
        n_eff = z.size // 2
        msg = (
            f"Laplace coverage={ekf_cov:.3f} var z={z.var():.3f}; "
            f"exact coverage={exact_cov:.3f}"
        )
        cov_lo, cov_hi = _binomial_band(0.9, n_eff)
        assert cov_lo < exact_cov < cov_hi, msg
        assert 0.7 < ekf_cov < 0.87, msg
        assert 1.3 < z.var() < 2.5, msg
        # paired on the same replicates: the gap is the Laplace approximation
        assert exact_cov - ekf_cov > 0.04, msg


@pytest.mark.slow
class TestPlugInDesignUncertainty:
    def test_coverage_against_truth_degrades_with_field_noise(self):
        """End-to-end: ignoring the smoother's uncertainty shrinks the intervals.

        Paired design: the same 60 seeds give the same latent and spikes at
        every ``lfp_noise_var`` (the LFP noise has its own RNG key), so only
        the design quality changes. Measured (T = 3000, |beta| = 1.8): pooled z
        variance 1.10 / 1.28 / 2.7 and 90% coverage 0.89 / 0.86 / 0.69 at
        lfp_noise_var = 0.01 / 0.25 / 4.0, with the magnitude unbiased (ratio
        1.00 / 1.00 / 1.01): the plug-in costs calibration (variance), not
        location. Pins keep headroom.
        """
        beta_true = np.array([1.5, -1.0])
        n_seeds = 60
        z_var, coverage, ratio = {}, {}, {}
        for noise in (0.01, 0.25, 4.0):
            params = _one_band_params(1, 0.1, beta=beta_true, lfp_noise_var=noise)
            z_all, ratios = [], []
            for seed in range(n_seeds):
                sim = simulate_coupling(params, n_time=3000, seed=seed)
                post = fit_coupling_ekf(sim.spikes, sim.lfp, params)
                z_all.append(_standardised_errors(post, beta_true[None, :]))
                ratios.append(
                    np.hypot(post.beta_real_mean[0, 0], post.beta_imag_mean[0, 0])
                    / np.hypot(*beta_true)
                )
            z = np.concatenate(z_all).ravel()
            z_var[noise] = float(np.mean(z**2))
            coverage[noise] = float(np.mean(np.abs(z) < Z90))
            ratio[noise] = float(np.mean(ratios))
        msg = f"E[z^2]={z_var} coverage={coverage} magnitude ratio={ratio}"
        # near-noiseless field: calibrated (4 binomial SE; chi2 band on E[z^2])
        assert coverage[0.01] > _binomial_band(0.9, n_seeds)[0], msg
        assert z_var[0.01] < stats.chi2.ppf(1 - 1e-4, n_seeds) / n_seeds, msg
        # more field noise => intervals too narrow, monotonically
        assert z_var[0.01] < z_var[0.25] < z_var[4.0], msg
        assert z_var[4.0] > 1.8 and coverage[4.0] < 0.8, msg
        # ... while the posterior mean stays unbiased for the magnitude
        for noise, value in ratio.items():
            assert abs(value - 1.0) < 0.04, (noise, msg)


@pytest.fixture(scope="module")
def recovery_params():
    """3 neurons x 2 bands, magnitude-2 coupling on band 0; band 1 is a control."""
    baseline_logit = float(np.log(0.05 / 0.95))
    return CouplingModelParams(
        osc_frequencies=jnp.array([6.0, 10.0]),
        osc_decay=jnp.array([0.99, 0.99]),
        process_noise_var=jnp.array([1.0 - 0.99**2, 1.0 - 0.99**2]),
        beta_real=jnp.array([[2.0, 0.0], [0.0, 0.0], [-2.0, 0.0]]),
        beta_imag=jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
        baseline=jnp.full((3,), baseline_logit),
        dt=1e-3,
    )


def _rmse(post, sim) -> float:
    err = np.concatenate(
        [
            (np.asarray(post.beta_real_mean) - np.asarray(sim.beta_real_true)).ravel(),
            (np.asarray(post.beta_imag_mean) - np.asarray(sim.beta_imag_true)).ravel(),
        ]
    )
    return float(np.sqrt(np.mean(err**2)))


@pytest.mark.slow
class TestRecoveryStatistics:
    @pytest.mark.parametrize("estimator", ["ekf", "pg"])
    def test_error_shrinks_with_more_data(self, recovery_params, estimator):
        """4 seeds x T in {1000, 8000}: coefficient RMSE falls with 8x the data.

        The sampling error shrinks by ~sqrt(8) = 2.8; the plug-in attenuation
        (a few percent of |beta| = 2) is a floor, so require the mean RMSE to
        drop by at least 40% and every seed to improve.
        """
        seeds = range(4)
        errors = {}
        for n_time in (1000, 8000):
            per_seed = []
            for seed in seeds:
                sim = simulate_coupling(recovery_params, n_time=n_time, seed=seed)
                if estimator == "ekf":
                    post = fit_coupling_ekf(sim.spikes, sim.lfp, recovery_params)
                else:
                    post = fit_coupling_pg(
                        sim.spikes,
                        sim.lfp,
                        recovery_params,
                        n_iter=300,
                        burn_in=100,
                        seed=seed,
                    )
                per_seed.append(_rmse(post, sim))
            errors[n_time] = np.asarray(per_seed)
        msg = (
            f"{estimator}: RMSE per seed T=1000 {np.round(errors[1000], 3)}, "
            f"T=8000 {np.round(errors[8000], 3)}"
        )
        assert np.all(errors[8000] < errors[1000]), msg
        assert errors[8000].mean() < 0.6 * errors[1000].mean(), msg
        # guard: the short runs carry real error (the comparison is not 0 < 0)
        assert errors[1000].mean() > 0.05, msg
