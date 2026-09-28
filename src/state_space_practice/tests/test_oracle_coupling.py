"""Independent-oracle tests for the spike-field coupling estimators.

Both estimators (:func:`fit_coupling_ekf`, :func:`fit_coupling_pg`) share stage 1
(the LFP-smoothed latent, a fixed design ``X``) and then fit, per neuron, the
*static* Bayesian Bernoulli-logit regression

    y_k ~ Bernoulli(sigmoid(b + X_k . beta)),   beta ~ N(0, sigma_beta^2 I),

with the intercept ``b`` known (``params.baseline``). The EKF arm returns the
Laplace approximation (MAP + inverse Hessian); the PG arm returns Gibbs draws
from the exact posterior. Every reference here is computed from scratch on the
*same* design the estimators condition on and shares no code with them:

- exact MAP / MLE and observed-information covariance by ``scipy.optimize``
  (trust-region Newton with the analytic gradient and Hessian);
- exact posterior moments and quantiles by grid quadrature (1-D and 2-D);
- the Polya-Gamma moments in closed form;
- a Geweke (2004) successive-conditional test of the Gibbs kernel's invariance.

Sampler tolerances are Monte Carlo standard errors from batch means (or i.i.d.
standard errors for exact draws), so every comparison is ``|diff| < k * MCSE``
with ``k`` stated, plus quadrature/optimizer error that is orders of magnitude
smaller.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from polyagamma import random_polyagamma
from scipy import optimize, special

from state_space_practice.coupling_ekf import fit_coupling_ekf
from state_space_practice.coupling_model import (
    CouplingModelParams,
    smooth_latent_from_lfp,
)
from state_space_practice.coupling_pg import fit_coupling_pg, pg_gibbs_sweep
from state_space_practice.coupling_validation import batch_means_mcse
from state_space_practice.simulate_coupling import simulate_coupling
from state_space_practice.tests.oracles import lgssm_dense_posterior

# ---------------------------------------------------------------------------
# Reference computations (NumPy / SciPy only)
# ---------------------------------------------------------------------------


def _one_band_params(
    beta: np.ndarray, base_rates: np.ndarray, lfp_noise_var: float = 0.25
) -> CouplingModelParams:
    """One 8 Hz band with unit stationary variance; ``beta`` is (S, 2) = (re, im)."""
    beta = np.atleast_2d(np.asarray(beta, dtype=float))
    base_rates = np.atleast_1d(np.asarray(base_rates, dtype=float))
    return CouplingModelParams(
        osc_frequencies=jnp.array([8.0]),
        osc_decay=jnp.array([0.98]),
        process_noise_var=jnp.array([1.0 - 0.98**2]),
        beta_real=jnp.asarray(beta[:, :1]),
        beta_imag=jnp.asarray(beta[:, 1:]),
        baseline=jnp.asarray(special.logit(base_rates)),
        dt=1e-3,
        lfp_noise_var=lfp_noise_var,
    )


def _neg_log_posterior(beta, design, y, offset, prior_precision):
    eta = offset + design @ beta
    loglik = y @ eta - np.logaddexp(0.0, eta).sum()
    return -loglik + 0.5 * prior_precision * beta @ beta


def _neg_log_posterior_hessian(beta, design, offset, prior_precision):
    p = special.expit(offset + design @ beta)
    return design.T @ ((p * (1.0 - p))[:, None] * design) + prior_precision * np.eye(
        design.shape[1]
    )


def logistic_map(design, y, offset, sigma_beta):
    """Exact MAP (``sigma_beta=None``: MLE) and inverse observed information.

    Closed-form Newton-Raphson with step halving on the penalised
    Bernoulli-logit negative log-likelihood (strictly convex for non-separable
    data), run to a step below 1e-13; the optimum is cross-checked with
    ``scipy.optimize.minimize`` (BFGS). Shares no code with the library's
    Fisher-scoring loop.
    """
    prior_precision = 0.0 if sigma_beta is None else 1.0 / sigma_beta**2
    args = (design, y, offset, prior_precision)

    def grad(beta):
        p = special.expit(offset + design @ beta)
        return -design.T @ (y - p) + prior_precision * beta

    beta = np.zeros(design.shape[1])
    for _ in range(200):
        step = np.linalg.solve(
            _neg_log_posterior_hessian(beta, design, offset, prior_precision),
            grad(beta),
        )
        alpha, current = 1.0, _neg_log_posterior(beta, *args)
        while _neg_log_posterior(beta - alpha * step, *args) > current + 1e-12:
            alpha *= 0.5
            assert alpha > 1e-10, "Newton line search failed"
        beta = beta - alpha * step
        if np.max(np.abs(alpha * step)) < 1e-13:
            break
    else:
        raise AssertionError("Newton iteration did not converge")
    check = optimize.minimize(
        _neg_log_posterior, np.zeros_like(beta), args=args, jac=lambda b, *_: grad(b)
    )
    assert np.max(np.abs(check.x - beta)) < 1e-4, (check.x, beta)
    hessian = _neg_log_posterior_hessian(beta, design, offset, prior_precision)
    return beta, np.linalg.inv(hessian)


def _log_posterior_on_points(points, design, y, offset, sigma_beta, chunk=4096):
    """Unnormalised log posterior at each row of ``points`` (G, P), chunked."""
    out = np.empty(points.shape[0])
    for start in range(0, points.shape[0], chunk):
        block = points[start : start + chunk]
        eta = offset + design @ block.T  # (T, g)
        loglik = y @ eta - np.logaddexp(0.0, eta).sum(axis=0)
        out[start : start + chunk] = loglik - 0.5 * (block**2).sum(1) / sigma_beta**2
    return out


class GridPosterior:
    """Exact posterior on a tensor grid centred on the MAP.

    The grid spans ``+- half`` Laplace standard deviations per axis and is
    widened (x2.5, up to four times) until the marginal mass on the boundary
    nodes is below 1e-8, so skewed, near-separated posteriors are covered too.
    Moments are midpoint-rule sums (spectrally accurate for smooth densities).
    """

    def __init__(self, design, y, offset, sigma_beta, half=12.0, n_points=161):
        mode, cov = logistic_map(design, y, offset, sigma_beta)
        self.mode = mode
        self.laplace_cov = cov
        for _ in range(5):
            self.axes = [
                m + np.sqrt(v) * np.linspace(-half, half, n_points)
                for m, v in zip(mode, np.diag(cov))
            ]
            mesh = np.meshgrid(*self.axes, indexing="ij")
            points = np.stack([g.ravel() for g in mesh], axis=1)
            logp = _log_posterior_on_points(points, design, y, offset, sigma_beta)
            weights = np.exp(logp - logp.max())
            weights /= weights.sum()
            self.weights = weights.reshape(mesh[0].shape)
            edge = max(
                self.marginal(a)[0] + self.marginal(a)[-1]
                for a in range(points.shape[1])
            )
            if edge < 1e-8:
                break
            half *= 2.5
        else:
            raise AssertionError(f"grid never captured the posterior (edge {edge})")
        self.mean = weights @ points
        centred = points - self.mean
        self.cov = (weights[:, None] * centred).T @ centred

    def marginal(self, axis: int) -> np.ndarray:
        other = tuple(a for a in range(self.weights.ndim) if a != axis)
        return self.weights.sum(axis=other)

    def quantile(self, axis: int, q: float) -> float:
        # Midpoint rule: node i carries the mass of the cell centred on it, so
        # the CDF *at* node i is the mass to its left plus half its own.
        marginal = self.marginal(axis)
        cdf = np.cumsum(marginal) - 0.5 * marginal
        return float(np.interp(q, cdf, self.axes[axis]))


def _pg_variance(b: float, c: float) -> float:
    """Closed-form Var[PG(b, c)] = b (sinh c - c) / (4 c^3 cosh^2(c/2)); b/24 at c=0."""
    if abs(c) < 1e-6:
        return b / 24.0
    return b * (np.sinh(c) - c) / (4.0 * c**3 * np.cosh(c / 2.0) ** 2)


def _pg_mean(b: float, c: float) -> float:
    """Closed-form E[PG(b, c)] = b tanh(c/2) / (2c); b/4 at c=0."""
    return b / 4.0 if c == 0 else b * np.tanh(c / 2.0) / (2.0 * c)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_neuron_data():
    """Two neurons (different base rates and couplings) on one band, T = 1000."""
    params = _one_band_params(
        beta=np.array([[1.0, -0.5], [-0.4, 0.8]]), base_rates=np.array([0.2, 0.08])
    )
    sim = simulate_coupling(params, n_time=1000, seed=1)
    design = np.asarray(smooth_latent_from_lfp(sim.lfp, params))
    return params, sim, design


# ---------------------------------------------------------------------------
# Stage 1 (shared by both estimators): the LFP smoother vs dense conditioning
# ---------------------------------------------------------------------------


class TestStageOneSmoother:
    def test_smoothed_design_equals_dense_gaussian_conditioning(self):
        """smooth_latent_from_lfp == E[x_t | lfp_{1:T}] from the dense joint.

        Two bands, T = 8. The transition is rebuilt by hand (decay times a
        rotation by 2 pi f dt per band), the prior is the stationary
        distribution, and the posterior mean comes from conditioning the full
        joint Gaussian (tests/oracles.py) -- no recursion shared with the
        library's Kalman smoother.
        """
        freqs, decay, q, dt, r = np.array([6.0, 30.0]), 0.9, 0.2, 1e-2, 0.5
        params = CouplingModelParams(
            osc_frequencies=jnp.asarray(freqs),
            osc_decay=jnp.full(2, decay),
            process_noise_var=jnp.full(2, q),
            beta_real=jnp.zeros((1, 2)),
            beta_imag=jnp.zeros((1, 2)),
            baseline=jnp.zeros(1),
            dt=dt,
            lfp_noise_var=r,
        )
        blocks = []
        for f in freqs:
            w = 2.0 * np.pi * f * dt
            blocks.append(
                decay * np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
            )
        transition = np.zeros((4, 4))
        transition[:2, :2], transition[2:, 2:] = blocks
        stationary = q / (1.0 - decay**2) * np.eye(4)
        lfp = np.random.default_rng(0).normal(size=(8, 4))
        dense = lgssm_dense_posterior(
            np.zeros(4),
            stationary,
            lfp,
            transition,
            q * np.eye(4),
            np.eye(4),
            r * np.eye(4),
        )
        smoothed = np.asarray(smooth_latent_from_lfp(lfp, params))
        np.testing.assert_allclose(smoothed, dense.smoothed_mean, atol=1e-10)
        # guard: smoothing is not the identity here (the prior pulls the mean)
        assert np.max(np.abs(smoothed - lfp)) > 0.1


# ---------------------------------------------------------------------------
# Task 2: regression limit -- stage 2 is exactly (penalised) logistic regression
# ---------------------------------------------------------------------------


class TestRegressionLimit:
    def test_design_reduces_to_lfp_as_field_noise_vanishes(self):
        """lfp_noise_var -> 0: the smoothed latent (the design) IS the LFP.

        The dynamics then play no role in stage 1, so the estimators are plain
        logistic regressions on the observed field.
        """
        params = _one_band_params(np.array([[1.0, -0.5]]), np.array([0.2]))
        sim = simulate_coupling(params, n_time=400, seed=0)
        noisy = np.asarray(smooth_latent_from_lfp(sim.lfp, params))
        exact = np.asarray(
            smooth_latent_from_lfp(sim.lfp, params._replace(lfp_noise_var=1e-10))
        )
        lfp = np.asarray(sim.lfp)
        # guard: at the default noise the smoother genuinely moves the design
        assert np.max(np.abs(noisy - lfp)) > 0.5
        np.testing.assert_allclose(exact, lfp, atol=1e-6)

    @pytest.mark.parametrize("sigma_beta", [1e4, 1.0, 0.3])
    def test_ekf_equals_exact_map_and_inverse_hessian(
        self, two_neuron_data, sigma_beta
    ):
        """EKF mean/cov == scipy MAP and inverse observed information, per neuron.

        ``sigma_beta = 1e4`` is the flat-prior limit (the MAP is the MLE and the
        covariance the inverse observed information, i.e. the textbook logistic
        regression standard errors); smaller values pin the exact prior
        precision ``1 / sigma_beta^2``.
        """
        params, sim, design = two_neuron_data
        post = fit_coupling_ekf(sim.spikes, sim.lfp, params, sigma_beta=sigma_beta)
        y_all = np.asarray(sim.spikes)
        for neuron in range(2):
            offset = float(params.baseline[neuron])
            reference = None if sigma_beta > 1e3 else sigma_beta
            mode, cov = logistic_map(design, y_all[:, neuron], offset, reference)
            ekf_mean = np.array(
                [post.beta_real_mean[neuron, 0], post.beta_imag_mean[neuron, 0]]
            )
            ekf_var = np.array(
                [post.beta_real_var[neuron, 0], post.beta_imag_var[neuron, 0]]
            )
            # The Fisher-scoring loop converges quadratically; the optimizer's
            # gtol=1e-11 puts both at ~1e-9 of each other (measured 2e-9).
            np.testing.assert_allclose(ekf_mean, mode, atol=1e-6)
            np.testing.assert_allclose(ekf_var, np.diag(cov), rtol=1e-6)
            np.testing.assert_allclose(
                post.beta_real_imag_cov[neuron, 0], cov[0, 1], rtol=1e-5, atol=1e-12
            )
            if sigma_beta == 0.3:
                # guard: the prior genuinely moves the estimate off the MLE, so
                # the comparison pins the prior precision, not just the data term
                mle, _ = logistic_map(design, y_all[:, neuron], offset, None)
                assert np.max(np.abs(mle - mode)) > 0.01

    @pytest.mark.slow
    def test_pg_posterior_approaches_map_under_flat_prior_and_long_data(self):
        """PG mean/cov -> MAP/inverse Hessian as the prior flattens and T grows.

        With ``sigma_beta = 1e3`` and T = 3000 the exact posterior mean differs
        from the MLE only by the O(1/T) skewness term, which the grid oracle
        computes; the PG sample mean must match the MLE within 4 batch-means
        MCSE plus that term. The same data at T = 100 (first 100 bins, grid
        oracle only) shows the mean-mode gap is genuinely larger with less data.
        A tight prior (``sigma_beta = 0.1``) must move the PG mean by the same
        amount it moves the MAP.
        """
        params = _one_band_params(np.array([[1.0, -0.5]]), np.array([0.2]))
        sim = simulate_coupling(params, n_time=3000, seed=2)
        design = np.asarray(smooth_latent_from_lfp(sim.lfp, params))
        y = np.asarray(sim.spikes)[:, 0]
        offset = float(params.baseline[0])

        grid_long = GridPosterior(design, y, offset, sigma_beta=1e3)
        grid_short = GridPosterior(design[:100], y[:100], offset, sigma_beta=1e3)
        gap_long = np.max(
            np.abs(grid_long.mean - grid_long.mode) / np.sqrt(np.diag(grid_long.cov))
        )
        gap_short = np.max(
            np.abs(grid_short.mean - grid_short.mode) / np.sqrt(np.diag(grid_short.cov))
        )
        # the mean-mode gap (in posterior sd) shrinks with data and is nonzero
        assert gap_short > 2.0 * gap_long, (gap_short, gap_long)
        assert gap_long > 1e-3

        results = {}
        for sigma_beta in (1e3, 0.1):
            pg = fit_coupling_pg(
                sim.spikes,
                sim.lfp,
                params,
                n_iter=4200,
                burn_in=200,
                sigma_beta=sigma_beta,
                seed=5,
            )
            draws = np.stack([pg.samples.real[:, 0, 0], pg.samples.imag[:, 0, 0]], 1)
            results[sigma_beta] = (draws.mean(0), batch_means_mcse(draws, 40), draws)

        flat_mean, flat_mcse, flat_draws = results[1e3]
        mle, inv_info = logistic_map(design, y, offset, None)
        tol = 4.0 * flat_mcse + np.abs(grid_long.mean - mle)
        assert np.all(np.abs(flat_mean - mle) < tol), (flat_mean, mle, tol)
        # covariance: relative MC error of a variance with n=4000 draws and an
        # integrated autocorrelation time of a few sweeps is ~0.05; allow 0.15.
        np.testing.assert_allclose(np.cov(flat_draws.T), inv_info, rtol=0.15, atol=0)

        tight_mean, tight_mcse, _ = results[0.1]
        map_tight, _ = logistic_map(design, y, offset, 0.1)
        shift_pg = tight_mean - flat_mean
        shift_map = map_tight - mle
        # guard: the prior shift is resolvable above the Monte Carlo error
        assert np.max(np.abs(shift_map)) > 10.0 * np.max(tight_mcse + flat_mcse)
        np.testing.assert_allclose(
            shift_pg, shift_map, atol=float(4.0 * np.max(tight_mcse + flat_mcse))
        )


# ---------------------------------------------------------------------------
# Task 3: Polya-Gamma sampler vs exact posterior; kernel invariance; PG draws
# ---------------------------------------------------------------------------


class TestPolyaGammaDraws:
    @pytest.mark.parametrize(
        ("b", "c"), [(1.0, 0.0), (1.0, 0.7), (1.0, -3.0), (1.0, 8.0), (2.0, 1.5)]
    )
    def test_mean_and_variance_match_closed_form(self, b, c):
        """E[PG(b,c)] = b tanh(c/2)/(2c), Var = b (sinh c - c)/(4 c^3 cosh^2(c/2))."""
        n = 200_000
        draws = random_polyagamma(
            h=b, z=c, size=n, random_state=np.random.default_rng(11)
        )
        mean_theory, var_theory = _pg_mean(b, c), _pg_variance(b, c)
        # i.i.d. draws: SE(mean) = sd/sqrt(n); SE(var) = sqrt((m4 - var^2)/n)
        se_mean = np.sqrt(var_theory / n)
        m4 = np.mean((draws - mean_theory) ** 4)
        se_var = np.sqrt((m4 - var_theory**2) / n)
        assert abs(draws.mean() - mean_theory) < 5.0 * se_mean
        assert abs(draws.var() - var_theory) < 5.0 * se_var
        # guard: the formulas are not trivially satisfied by the c = 0 value
        if c != 0.0:
            assert abs(mean_theory - b / 4.0) > 20.0 * se_mean

    def test_variance_formula_limit_is_continuous(self):
        """The closed-form variance is continuous at c = 0 (b/24)."""
        assert _pg_variance(1.0, 1e-3) == pytest.approx(1.0 / 24.0, rel=1e-5)


def _run_chain(design, y, offset, sigma_beta, n_iter, seed):
    """Iterate the library's Gibbs kernel from beta = 0; returns (n_iter, P)."""
    rng = np.random.default_rng(seed)
    kappa = y - 0.5
    prior_precision = np.eye(design.shape[1]) / sigma_beta**2
    beta = np.zeros(design.shape[1])
    out = np.empty((n_iter, design.shape[1]))
    for i in range(n_iter):
        beta = pg_gibbs_sweep(beta, design, kappa, offset, prior_precision, rng)
        out[i] = beta
    return out


class TestExactPosteriorOracle:
    @pytest.mark.slow
    def test_one_coefficient_kernel_matches_grid_posterior(self):
        """1-D: chain mean, variance and 90% interval match the exact posterior.

        Few spikes (T = 20, rate 0.2, 4 spikes) make the posterior visibly
        skewed (exact mean - MAP = 0.3 posterior sd, > 5 MCSE), so agreement
        with the exact posterior is distinguishable from agreement with a
        Laplace fit.
        """
        rng = np.random.default_rng(3)
        n_time = 20
        design = rng.normal(size=(n_time, 1))
        offset = float(special.logit(0.2))
        y = (rng.random(n_time) < special.expit(offset + 1.0 * design[:, 0])).astype(
            float
        )
        sigma_beta = 3.0
        grid = GridPosterior(design, y, offset, sigma_beta, half=16.0, n_points=8001)
        draws = _run_chain(design, y, offset, sigma_beta, n_iter=21_000, seed=0)[
            1000:, 0
        ]
        n_batches = 50
        mcse_mean = float(batch_means_mcse(draws, n_batches))
        # guard: the posterior is non-Gaussian enough that mean != mode
        assert abs(grid.mean[0] - grid.mode[0]) > 5.0 * mcse_mean

        assert abs(draws.mean() - grid.mean[0]) < 4.0 * mcse_mean
        sq = (draws - grid.mean[0]) ** 2
        assert abs(sq.mean() - grid.cov[0, 0]) < 4.0 * float(
            batch_means_mcse(sq, n_batches)
        )
        for q in (0.05, 0.95):
            below = (draws <= grid.quantile(0, q)).astype(float)
            assert abs(below.mean() - q) < 4.0 * float(
                batch_means_mcse(below, n_batches)
            ), (q, below.mean())

    @pytest.mark.slow
    def test_two_coefficient_fit_matches_grid_posterior(self):
        """2-D via the public API: marginal means, variances, 90% intervals.

        One band (real, imag) on the LFP-smoothed design fit_coupling_pg itself
        uses; T = 150 at a 5% base rate (~15 spikes) with a moderately tight
        prior. Tolerances: 4 batch-means MCSE (40 batches of 275 draws).
        """
        params = _one_band_params(np.array([[1.0, -0.5]]), np.array([0.05]))
        sim = simulate_coupling(params, n_time=150, seed=4)
        design = np.asarray(smooth_latent_from_lfp(sim.lfp, params))
        y = np.asarray(sim.spikes)[:, 0]
        sigma_beta = 1.5
        grid = GridPosterior(design, y, float(params.baseline[0]), sigma_beta)
        pg = fit_coupling_pg(
            sim.spikes,
            sim.lfp,
            params,
            n_iter=12_000,
            burn_in=1000,
            sigma_beta=sigma_beta,
            seed=1,
        )
        draws = np.stack([pg.samples.real[:, 0, 0], pg.samples.imag[:, 0, 0]], 1)
        n_batches = 40
        mcse = batch_means_mcse(draws, n_batches)
        # guard: the data moved the posterior well off the prior mean
        assert np.linalg.norm(grid.mean) > 10.0 * np.max(mcse)
        assert np.all(np.abs(draws.mean(0) - grid.mean) < 4.0 * mcse)
        sq = (draws - grid.mean) ** 2
        assert np.all(
            np.abs(sq.mean(0) - np.diag(grid.cov))
            < 4.0 * batch_means_mcse(sq, n_batches)
        )
        cross = np.prod(draws - grid.mean, axis=1)
        assert abs(cross.mean() - grid.cov[0, 1]) < 4.0 * float(
            batch_means_mcse(cross, n_batches)
        )
        for axis in range(2):
            for q in (0.05, 0.95):
                below = (draws[:, axis] <= grid.quantile(axis, q)).astype(float)
                assert abs(below.mean() - q) < 4.0 * float(
                    batch_means_mcse(below, n_batches)
                ), (axis, q, below.mean())


def _geweke_z(kernel_prior_scale: float) -> np.ndarray:
    """Geweke (2004) marginal-conditional vs successive-conditional z-scores.

    Marginal-conditional simulator: ``beta ~ prior``, ``y ~ p(y | beta)``,
    i.i.d. Successive-conditional simulator: alternate one library Gibbs sweep
    ``beta ~ K(. | beta, y)`` with a fresh ``y ~ p(y | beta)``. If ``K`` leaves
    ``p(beta | y)`` invariant, both simulators have the joint ``p(beta, y)`` as
    stationary law, so every test-function mean agrees. ``kernel_prior_scale``
    multiplies the prior precision handed to the kernel (1 = correct).
    """
    rng = np.random.default_rng(2024)
    n_time, n_coef = 25, 2
    design = rng.normal(size=(n_time, n_coef))
    offset = -0.5
    sigma_beta = 1.0
    n_draws = 20_000

    def test_functions(beta, y):
        # beta: (n, 2); y: (n, T)
        fitted = (y * (offset + beta @ design.T)).mean(axis=1)
        return np.column_stack(
            [
                beta[:, 0],
                beta[:, 1],
                beta[:, 0] ** 2,
                beta[:, 1] ** 2,
                beta[:, 0] * beta[:, 1],
                y.mean(axis=1),
                fitted,
            ]
        )

    def draw_y(beta):
        return (
            rng.random(beta.shape[:-1] + (n_time,))
            < special.expit(offset + beta @ design.T)
        ).astype(float)

    beta_mc = rng.normal(scale=sigma_beta, size=(n_draws, n_coef))
    g_mc = test_functions(beta_mc, draw_y(beta_mc))

    prior_precision = kernel_prior_scale * np.eye(n_coef) / sigma_beta**2
    beta = rng.normal(scale=sigma_beta, size=n_coef)
    y = draw_y(beta)
    betas = np.empty((n_draws, n_coef))
    ys = np.empty((n_draws, n_time))
    for i in range(n_draws):
        beta = pg_gibbs_sweep(beta, design, y - 0.5, offset, prior_precision, rng)
        y = draw_y(beta)
        betas[i], ys[i] = beta, y
    g_sc = test_functions(betas, ys)

    se_mc = g_mc.std(axis=0, ddof=1) / np.sqrt(n_draws)
    se_sc = batch_means_mcse(g_sc, 50)
    return (g_sc.mean(0) - g_mc.mean(0)) / np.sqrt(se_mc**2 + se_sc**2)


class TestGibbsKernelInvariance:
    @pytest.mark.slow
    def test_successive_conditional_matches_marginal_conditional(self):
        """The augmented Gibbs kernel leaves p(beta | y) invariant (Geweke test).

        Seven test functions (first and second beta moments, the spike rate and
        a beta-y cross moment); each z must be below 4 (Bonferroni 7 x 1e-4).
        Power check: the same test with the kernel given a 4x prior precision
        (a wrong target) fails by a wide margin.
        """
        z_correct = _geweke_z(kernel_prior_scale=1.0)
        assert np.max(np.abs(z_correct)) < 4.0, np.round(z_correct, 2)
        z_broken = _geweke_z(kernel_prior_scale=4.0)
        assert np.max(np.abs(z_broken)) > 8.0, np.round(z_broken, 2)


# ---------------------------------------------------------------------------
# Task 4: Laplace ("EKF") posterior vs the exact posterior
# ---------------------------------------------------------------------------


class TestLaplaceVsExactPosterior:
    @pytest.mark.slow
    def test_approximation_gap_shrinks_with_information(self):
        """|MAP - E[beta|y]| / sd and the variance error shrink as T grows.

        The coupling in this model is static, so "information per coefficient"
        grows with T. Averaged over 3 seeds (measured: mean gap 0.19, 0.06,
        0.023 posterior sd and variance error 4.4%, 1.1%, 0.2% at T = 100, 800,
        3200), each level must beat the previous one, the gap must be nonzero at
        T = 100, and the pins keep ~1.5-2.5x headroom.
        """
        lengths = (100, 800, 3200)
        mean_gaps, var_errors = [], []
        for n_time in lengths:
            gaps, errs = [], []
            for seed in range(3):
                params = _one_band_params(np.array([[1.0, -0.5]]), np.array([0.05]))
                sim = simulate_coupling(params, n_time=n_time, seed=seed)
                design = np.asarray(smooth_latent_from_lfp(sim.lfp, params))
                y = np.asarray(sim.spikes)[:, 0]
                grid = GridPosterior(
                    design, y, float(params.baseline[0]), 1.0, n_points=121
                )
                post = fit_coupling_ekf(sim.spikes, sim.lfp, params, sigma_beta=1.0)
                mean = np.array([post.beta_real_mean[0, 0], post.beta_imag_mean[0, 0]])
                var = np.array([post.beta_real_var[0, 0], post.beta_imag_var[0, 0]])
                sd = np.sqrt(np.diag(grid.cov))
                gaps.append(np.max(np.abs(mean - grid.mean) / sd))
                errs.append(np.max(np.abs(var / np.diag(grid.cov) - 1.0)))
            mean_gaps.append(float(np.mean(gaps)))
            var_errors.append(float(np.mean(errs)))
        msg = f"T={lengths} mean gaps={mean_gaps} var errors={var_errors}"
        assert mean_gaps[0] > mean_gaps[1] > mean_gaps[2], msg
        assert var_errors[0] > var_errors[1] > var_errors[2], msg
        assert mean_gaps[0] > 0.05 and var_errors[0] > 0.01, msg  # gap is real
        assert mean_gaps[0] < 0.3 and var_errors[0] < 0.08, msg
        assert mean_gaps[2] < 0.05 and var_errors[2] < 0.005, msg

    @pytest.mark.slow
    def test_static_model_cannot_follow_time_varying_coupling(self):
        """Pin the static-coupling assumption under a mid-session sign flip.

        Neither estimator has coupling dynamics (there is no filter over beta):
        when the true coupling flips from +c to -c halfway, the single static
        posterior sits near 0 and its 99% interval excludes both segment values.
        Control on the same latent: a constant +c is recovered.
        """
        params = _one_band_params(np.array([[0.0, 0.0]]), np.array([0.2]))
        sim = simulate_coupling(params, n_time=4000, seed=6)
        latent = np.asarray(sim.latent_true)
        rng = np.random.default_rng(6)
        c = np.array([1.2, 0.0])
        flip = np.where(np.arange(4000) < 2000, 1.0, -1.0)
        offset = float(params.baseline[0])

        def spikes_for(beta_t):
            rate = special.expit(offset + np.sum(latent * beta_t, axis=1))
            return (rng.random(4000) < rate).astype(float)[:, None]

        static = fit_coupling_ekf(spikes_for(c[None, :]), sim.lfp, params)
        flipped = fit_coupling_ekf(
            spikes_for(flip[:, None] * c[None, :]), sim.lfp, params
        )
        z99 = 2.576
        # control: the constant coupling is recovered (within 4 sd + the known
        # plug-in attenuation of a few percent)
        assert (
            abs(static.beta_real_mean[0, 0] - c[0])
            < 4.0 * np.sqrt(static.beta_real_var[0, 0]) + 0.05 * c[0]
        )
        m, sd = flipped.beta_real_mean[0, 0], np.sqrt(flipped.beta_real_var[0, 0])
        assert abs(m) < 0.3, m
        assert m + z99 * sd < c[0] and m - z99 * sd > -c[0], (m, sd)
