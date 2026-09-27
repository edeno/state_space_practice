"""Exact oracles for the Laplace-EKF point-process filter, smoother and EM.

Three kinds of independent evidence, each separating *approximation* from
*bug*:

1. **The code computes the approximation it claims.** An independent NumPy
   re-implementation of the Laplace recursion (pure Newton to machine
   precision for the converged update, one Newton step from the prior mean
   for ``max_newton_iter=1``) and a grid implementation of the RTS backward
   pass *given the filter's own Gaussians* must agree with
   :func:`stochastic_point_process_filter` / ``..._smoother`` to roundoff.
   Any disagreement here is a bug (a wrong sign, a missing term, a wrong
   log-likelihood normaliser), never "the approximation".

2. **How far the approximation is from the truth.** The exact posterior of
   a small Poisson-GLM state-space model (``T <= 6``, 1 or 2 latent
   dimensions, AR(1) dynamics) is computed by quadrature on a grid:
   filtered and smoothed means / variances and the exact marginal
   log-likelihood ``log p(y_{1:T})``. The Laplace outputs are compared in a
   near-Gaussian (high-rate) and a low-rate regime and the observed errors
   are pinned (with ~10x headroom for the converged update) so a regression
   is caught. The pins also assert the gap is *non-zero*, so a vacuous
   oracle (e.g. one that accidentally re-computes the Laplace answer) fails.
   The Laplace approximation must become exact as the data become
   informative (the posterior tends to a Gaussian): a missing constant in
   the normaliser would not vanish, so this also guards the log-likelihood.

   What is approximation (not a bug):

   * ``max_newton_iter=1`` is a single Fisher step from the prior mean (an
     EKF linearisation). When one bin's likelihood is much sharper than the
     predictive prior (high rates, broad prior) the step lands far from the
     mode; errors of 0.1-0.3 posterior sd -- and occasionally several sd --
     are expected there, and they do *not* shrink with more data per bin.
   * The converged update is a Laplace approximation at the posterior
     mode: its mean is the mode, not the mean (a skew offset of a few
     hundredths of a posterior sd here), and its variance the inverse
     curvature at the mode.
   * The smoother is an RTS pass over the filter's Gaussians. It inherits
     the filter's error and adds the error of treating the filtered
     densities as Gaussian when the backward message moves the posterior
     into their tails (up to ~6% in variance in the high-rate problems
     here, even when the filter is accurate to <1%).

3. **The EM M-step is exact.** The expected complete-data log-likelihood
   ``Q(theta)`` is written out explicitly below (per-time traces, no
   library sums) and its gradient must vanish at the parameters returned by
   :func:`dynamics_only_m_step` and ``PlaceFieldModel._m_step`` (finite
   differences, respecting the model's constraints), while ``Q`` does not
   decrease. The observation-side GLM fit (``_fit_stationary_glm``) is
   checked the same way against its penalised Poisson log-likelihood.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import gammaln

from state_space_practice.kalman import InitialStatePrior
from state_space_practice.place_field_model import PlaceFieldModel
from state_space_practice.point_process_kalman import (
    BlockDiagonalCovariance,
    dynamics_only_m_step,
    stochastic_point_process_smoother,
)

# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


def _affine_log_rate(design_t, x):
    """log lambda_n = b_n + w_n . x with ``design_t[n] = [b_n, w_n]``."""
    return design_t[:, 0] + design_t[:, 1:] @ x


class _Problem(NamedTuple):
    init_mean: np.ndarray  # (d,)  prior mean of x_0
    init_cov: np.ndarray  # (d, d)
    a_diag: np.ndarray  # (d,)  A = diag(a_diag)
    q_diag: np.ndarray  # (d,)  Q = diag(q_diag)
    design: np.ndarray  # (T, n_neurons, 1 + d)
    spikes: np.ndarray  # (T, n_neurons)
    dt: float


def _simulate_problem(
    n_latent: int,
    n_neurons: int,
    n_time: int,
    rate_hz: float,
    weight_scale: float,
    seed: int,
    a: float = 0.9,
    q: float = 0.05,
    dt: float = 0.01,
    init_var: float = 0.3,
) -> _Problem:
    """Draw x_0 ~ prior, AR(1) dynamics, Poisson spikes at baseline ``rate_hz``."""
    rng = np.random.default_rng(seed)
    init_mean = rng.normal(0.0, 0.3, n_latent)
    init_cov = init_var * np.eye(n_latent)
    if n_latent == 2:
        init_cov[0, 1] = init_cov[1, 0] = init_var / 3.0
    a_diag = np.full(n_latent, a)
    q_diag = np.full(n_latent, q)
    weights = rng.normal(0.0, weight_scale, (n_neurons, n_latent))
    baseline = np.full(n_neurons, np.log(rate_hz))
    design = np.tile(np.concatenate([baseline[:, None], weights], 1), (n_time, 1, 1))
    x = rng.multivariate_normal(init_mean, init_cov)
    spikes = []
    for _ in range(n_time):
        x = a_diag * x + rng.normal(0.0, np.sqrt(q_diag))
        spikes.append(rng.poisson(np.exp(baseline + weights @ x) * dt))
    return _Problem(
        init_mean, init_cov, a_diag, q_diag, design, np.asarray(spikes, float), dt
    )


def _run_laplace(problem: _Problem, max_newton_iter: int):
    """Library filter + smoother on ``problem`` (NumPy outputs)."""
    args = (
        jnp.asarray(problem.init_mean),
        jnp.asarray(problem.init_cov),
        jnp.asarray(problem.design),
        jnp.asarray(problem.spikes),
        problem.dt,
        jnp.diag(jnp.asarray(problem.a_diag)),
        jnp.diag(jnp.asarray(problem.q_diag)),
        _affine_log_rate,
    )
    sm, sc, scc, ll, fm, fc = stochastic_point_process_smoother(
        *args, max_newton_iter=max_newton_iter, return_filtered=True
    )
    return {
        "filt_mean": np.asarray(fm),
        "filt_cov": np.asarray(fc),
        "smooth_mean": np.asarray(sm),
        "smooth_cov": np.asarray(sc),
        "smooth_cross_cov": np.asarray(scc),
        "log_lik": float(ll),
    }


# ---------------------------------------------------------------------------
# Oracle 1: exact posterior by quadrature on a grid
# ---------------------------------------------------------------------------


def _gauss_kernel(grid: np.ndarray, a: float, q: float) -> np.ndarray:
    """``K[i, j] = p(x' = g_i | x = g_j) * dx`` for ``x' = a x + N(0, q)``."""
    dx = grid[1] - grid[0]
    return (
        np.exp(-0.5 * (grid[:, None] - a * grid[None, :]) ** 2 / q)
        / np.sqrt(2.0 * np.pi * q)
        * dx
    )


def _apply_kernels(density, kernels, transpose=False):
    """Apply a separable (diagonal-A, diagonal-Q) transition along each axis."""
    for axis, kernel in enumerate(kernels):
        k = kernel.T if transpose else kernel
        density = np.moveaxis(np.tensordot(k, density, axes=([1], [axis])), 0, axis)
    return density


def _grid_points(grids):
    mesh = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    return mesh.reshape(-1, len(grids)), mesh.shape[:-1]


def _gaussian_mass(points, shape, mean, cov, cell):
    r = points - mean
    log_pdf = -0.5 * np.einsum("ni,ij,nj->n", r, np.linalg.inv(cov), r)
    log_pdf -= 0.5 * np.linalg.slogdet(2.0 * np.pi * cov)[1]
    return np.exp(log_pdf).reshape(shape) * cell


def _moments(mass, points):
    w = mass.reshape(-1)
    mean = w @ points
    c = points - mean
    return mean, (w[:, None] * c).T @ c


def _grid_posterior(problem: _Problem, grids) -> dict:
    """Exact filtering / smoothing moments and log p(y_{1:T}) by quadrature.

    Densities are held as probability mass per grid cell. The transition is
    separable because A and Q are diagonal; the observation couples the
    dimensions, so the posterior itself is a full d-dimensional grid.
    """
    points, shape = _grid_points(grids)
    cell = np.prod([g[1] - g[0] for g in grids])
    A = np.diag(problem.a_diag)
    pred_mean = A @ problem.init_mean
    pred_cov = A @ problem.init_cov @ A.T + np.diag(problem.q_diag)
    pred = _gaussian_mass(points, shape, pred_mean, pred_cov, cell)
    kernels = [
        _gauss_kernel(g, a, q) for g, a, q in zip(grids, problem.a_diag, problem.q_diag)
    ]
    filtered, likelihoods = [], []
    log_lik = 0.0
    for design_t, y_t in zip(problem.design, problem.spikes):
        log_rate = design_t[:, 0][None, :] + points @ design_t[:, 1:].T
        mu = np.exp(log_rate) * problem.dt
        loglik = np.sum(y_t * np.log(mu) - mu - gammaln(y_t + 1.0), axis=1)
        loglik = loglik.reshape(shape)
        peak = loglik.max()
        lik = np.exp(loglik - peak)
        evidence = np.sum(lik * pred)
        log_lik += np.log(evidence) + peak
        post = lik * pred / evidence
        filtered.append(post)
        likelihoods.append(lik)
        pred = _apply_kernels(post, kernels)
    # Backward messages beta_t(x_t) = int p(x_{t+1}|x_t) L_{t+1} beta_{t+1}.
    n_time = len(filtered)
    smoothed = [None] * n_time
    smoothed[-1] = filtered[-1]
    beta = np.ones(shape)
    for t in range(n_time - 2, -1, -1):
        beta = _apply_kernels(likelihoods[t + 1] * beta, kernels, transpose=True)
        beta /= beta.max()
        s = filtered[t] * beta
        smoothed[t] = s / s.sum()
    fm = [_moments(p, points) for p in filtered]
    sm = [_moments(p, points) for p in smoothed]
    return {
        "filt_mean": np.array([m for m, _ in fm]),
        "filt_cov": np.array([c for _, c in fm]),
        "smooth_mean": np.array([m for m, _ in sm]),
        "smooth_cov": np.array([c for _, c in sm]),
        "log_lik": float(log_lik),
    }


def _standardized_errors(laplace: dict, exact: dict) -> dict:
    """Max over time/dims of |mean err| / exact sd and |var ratio - 1|."""
    out = {}
    for kind in ("filt", "smooth"):
        exact_var = np.diagonal(exact[f"{kind}_cov"], axis1=1, axis2=2)
        lap_var = np.diagonal(laplace[f"{kind}_cov"], axis1=1, axis2=2)
        mean_err = np.abs(laplace[f"{kind}_mean"] - exact[f"{kind}_mean"])
        out[f"{kind}_mean"] = float(np.max(mean_err / np.sqrt(exact_var)))
        out[f"{kind}_var"] = float(np.max(np.abs(lap_var / exact_var - 1.0)))
    out["log_lik"] = abs(laplace["log_lik"] - exact["log_lik"])
    return out


# ---------------------------------------------------------------------------
# Oracle 2: independent re-implementation of the Laplace recursion
# ---------------------------------------------------------------------------


def _log_evidence_terms(design_t, y_t, dt, x):
    """Poisson log-pmf, score and Fisher information of the affine model at x."""
    log_rate = design_t[:, 0] + design_t[:, 1:] @ x
    mu = np.exp(log_rate) * dt
    jac = design_t[:, 1:]
    logpmf = np.sum(y_t * np.log(mu) - mu - gammaln(y_t + 1.0))
    return logpmf, jac.T @ (y_t - mu), jac.T @ (mu[:, None] * jac)


def _laplace_filter_reference(problem: _Problem, converged: bool) -> dict:
    """NumPy Laplace filter: one Newton step (EKF) or Newton to convergence.

    For the affine log-link the observed and expected Hessians coincide, so
    pure Newton is the Fisher iteration the library runs. The per-step
    log-evidence is ``log p(y|x*) - 0.5 d' P^{-1} d - 0.5 log|P| + 0.5
    log|P_post|`` with ``d = x* - m_pred`` and ``P_post`` the inverse
    posterior precision the update returns.
    """
    A = np.diag(problem.a_diag)
    Q = np.diag(problem.q_diag)
    mean, cov = problem.init_mean, problem.init_cov
    means, covs, log_lik = [], [], 0.0
    for design_t, y_t in zip(problem.design, problem.spikes):
        m_pred = A @ mean
        P_pred = A @ cov @ A.T + Q
        prior_prec = np.linalg.inv(P_pred)
        x = m_pred.copy()
        _, score, fisher = _log_evidence_terms(design_t, y_t, problem.dt, x)
        post_prec = prior_prec + fisher
        x = x + np.linalg.solve(post_prec, score)
        if converged:
            for _ in range(200):
                _, score, fisher = _log_evidence_terms(design_t, y_t, problem.dt, x)
                post_prec = prior_prec + fisher
                step = np.linalg.solve(post_prec, score - prior_prec @ (x - m_pred))
                x = x + step
                if np.max(np.abs(step)) < 1e-14:
                    break
            _, _, fisher = _log_evidence_terms(design_t, y_t, problem.dt, x)
            post_prec = prior_prec + fisher
        logpmf, _, _ = _log_evidence_terms(design_t, y_t, problem.dt, x)
        d = x - m_pred
        log_lik += (
            logpmf
            - 0.5 * d @ prior_prec @ d
            - 0.5 * np.linalg.slogdet(P_pred)[1]
            - 0.5 * np.linalg.slogdet(post_prec)[1]
        )
        mean, cov = x, np.linalg.inv(post_prec)
        means.append(mean)
        covs.append(cov)
    return {
        "filt_mean": np.array(means),
        "filt_cov": np.array(covs),
        "log_lik": log_lik,
    }


def _grid_rts_given_gaussians(problem: _Problem, filt_mean, filt_cov, grid):
    """1D smoothed moments and lag-one cross-covariances by quadrature,
    treating the filtered densities as the given Gaussians.

    ``p(x_t, x_{t+1} | y) = f_t(x_t) K(x_{t+1}|x_t) s_{t+1}(x_{t+1}) /
    p_{t+1}(x_{t+1})`` with ``p_{t+1} = int K f_t`` -- the RTS recursion
    written as an integral, independent of the gain algebra.
    """
    a, q = problem.a_diag[0], problem.q_diag[0]
    dx = grid[1] - grid[0]
    kernel = _gauss_kernel(grid, a, q) / dx  # density p(x'|x)
    n_time = filt_mean.shape[0]

    def gauss(m, v):
        return np.exp(-0.5 * (grid - m) ** 2 / v) / np.sqrt(2 * np.pi * v)

    sm = [None] * n_time
    sv = [None] * n_time
    scc = [None] * (n_time - 1)
    s_next = gauss(filt_mean[-1, 0], filt_cov[-1, 0, 0])
    sm[-1], sv[-1] = filt_mean[-1, 0], filt_cov[-1, 0, 0]
    for t in range(n_time - 2, -1, -1):
        f_t = gauss(filt_mean[t, 0], filt_cov[t, 0, 0])
        pred_var = a**2 * filt_cov[t, 0, 0] + q
        p_next = gauss(a * filt_mean[t, 0], pred_var)
        ratio = np.where(p_next > 0, s_next / np.maximum(p_next, 1e-300), 0.0)
        joint = f_t[None, :] * kernel * ratio[:, None] * dx * dx  # [x_{t+1}, x_t]
        joint /= joint.sum()
        m_t = np.sum(joint.sum(0) * grid)
        m_next = np.sum(joint.sum(1) * grid)
        sm[t] = m_t
        sv[t] = np.sum(joint.sum(0) * (grid - m_t) ** 2)
        scc[t] = np.sum(joint * np.outer(grid - m_next, grid - m_t))
        s_next = joint.sum(0) / dx  # smoothed density of x_t (sum over x_{t+1})
    return np.array(sm), np.array(sv), np.array(scc)


# ---------------------------------------------------------------------------
# 1. The code computes the approximation it claims
# ---------------------------------------------------------------------------

# A high-rate problem (large per-bin updates, where the one-step and the
# converged updates differ substantially), and a low-rate one.
_HIGH_RATE = {"n_neurons": 4, "rate_hz": 3000.0, "weight_scale": 0.7}
_LOW_RATE = {"n_neurons": 2, "rate_hz": 20.0, "weight_scale": 0.7}


class TestLaplaceFilterComputesItsApproximation:
    """Disagreement with the NumPy recursion is a bug, not approximation."""

    @pytest.mark.parametrize("n_latent", [1, 2])
    @pytest.mark.parametrize("regime", [_HIGH_RATE, _LOW_RATE], ids=["high", "low"])
    @pytest.mark.parametrize(
        ("max_newton_iter", "converged"), [(1, False), (30, True)], ids=["N1", "Nconv"]
    )
    def test_filter_matches_independent_recursion(
        self, n_latent, regime, max_newton_iter, converged
    ) -> None:
        """One-step update to roundoff; the converged update to ~sqrt(eps).

        The library's iterated update accepts a Fisher step only when it
        strictly decreases the negative log-posterior. Within ~1e-8
        (relative) of the mode the decrease ``~ step^2 * curvature`` is
        below the roundoff of the loss (~eps * |loss|), so the iteration
        stops there: the converged mode is exact to ~sqrt(eps), not eps. The
        reference Newton runs to 1e-14. The one-step update has no line
        search and must match to roundoff.
        """
        problem = _simulate_problem(n_latent, n_time=6, seed=1, **regime)
        got = _run_laplace(problem, max_newton_iter)
        ref = _laplace_filter_reference(problem, converged=converged)
        # Guard: the updates are genuinely nonlinear (the reference mean moved
        # away from the prediction by more than roundoff).
        assert np.max(np.abs(np.diff(ref["filt_mean"], axis=0))) > 1e-3
        rtol = 1e-7 if converged else 1e-10
        np.testing.assert_allclose(
            got["filt_mean"], ref["filt_mean"], rtol=rtol, atol=rtol
        )
        np.testing.assert_allclose(
            got["filt_cov"], ref["filt_cov"], rtol=rtol, atol=1e-3 * rtol
        )
        np.testing.assert_allclose(got["log_lik"], ref["log_lik"], rtol=rtol, atol=rtol)


class TestSmootherIsExactRTSGivenFilteredGaussians:
    """The backward pass adds no error beyond the filter's Gaussians."""

    @pytest.mark.parametrize("regime", [_HIGH_RATE, _LOW_RATE], ids=["high", "low"])
    def test_1d_smoother_and_cross_cov_match_quadrature(self, regime) -> None:
        problem = _simulate_problem(1, n_time=6, seed=2, **regime)
        got = _run_laplace(problem, max_newton_iter=3)
        grid = np.linspace(-4.0, 4.0, 1601)
        sm, sv, scc = _grid_rts_given_gaussians(
            problem, got["filt_mean"], got["filt_cov"], grid
        )
        # Guard: smoothing changed the estimates (a no-op smoother would fail).
        assert (
            np.max(np.abs(got["smooth_mean"][:-1, 0] - got["filt_mean"][:-1, 0])) > 1e-3
        )
        np.testing.assert_allclose(got["smooth_mean"][:, 0], sm, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(got["smooth_cov"][:, 0, 0], sv, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(
            got["smooth_cross_cov"][:, 0, 0], scc, rtol=1e-6, atol=1e-9
        )


# ---------------------------------------------------------------------------
# 2. Approximation error against the exact posterior
# ---------------------------------------------------------------------------

# Near-Gaussian regime: high rates (~50 spikes / bin / neuron), a tight prior
# and slow dynamics, so every bin carries a lot of (nearly Gaussian) evidence.
_NEAR_GAUSSIAN = {
    "n_neurons": 4,
    "rate_hz": 5000.0,
    "weight_scale": 0.4,
    "init_var": 0.02,
    "a": 0.95,
    "q": 0.005,
}
_SEEDS = (0, 1, 2)
_METRICS = ("filt_mean", "filt_var", "smooth_mean", "smooth_var", "log_lik")


def _grids(n_latent: int) -> list[np.ndarray]:
    n_points = 1201 if n_latent == 1 else 241
    return [np.linspace(-3.0, 3.0, n_points)] * n_latent


def _regime_errors(n_latent: int, regime: dict) -> dict:
    """Standardised errors of N1 and N3 vs the exact posterior, per seed."""
    out = {1: [], 3: []}
    for seed in _SEEDS:
        problem = _simulate_problem(n_latent, n_time=6, seed=seed, **regime)
        exact = _grid_posterior(problem, _grids(n_latent))
        for n_iter in (1, 3):
            out[n_iter].append(
                _standardized_errors(_run_laplace(problem, n_iter), exact)
            )
    return out


def _worst(errors: list[dict], metric_prefix: str) -> float:
    return max(e[m] for e in errors for m in _METRICS if m.startswith(metric_prefix))


def _describe(errors: dict) -> str:
    return "; ".join(
        f"N{n}: " + ", ".join(f"{m}={max(e[m] for e in errs):.4f}" for m in _METRICS)
        for n, errs in errors.items()
    )


class TestLaplaceVsExactPosterior:
    """Pinned approximation error of the Laplace filter / smoother.

    Observed (worst over seeds 0-2 and 1-2 latent dims, T=6; mean errors in
    exact posterior sd, variance as |ratio - 1|, log-likelihood in nats):

    ============  ======  =========  ==========  ========
    regime        update  mean       variance    log-lik
    ============  ======  =========  ==========  ========
    near-Gauss    N3      0.047      0.029       0.031
    near-Gauss    N1      0.11       0.070       0.088
    low-rate      N3      0.134      0.080       0.025
    low-rate      N1      0.137      0.124       0.013
    ============  ======  =========  ==========  ========

    In the near-Gaussian regime the Newton-3 update is within a few
    hundredths of a posterior sd of the exact answer (pinned with ~10x
    headroom), while the one-step (N1) EKF update, which linearises at the
    prior mean, is 2-4x further off (pinned with ~2x headroom). In the
    low-rate regime the posterior is dominated by the prior, the likelihood
    (mostly zero counts) is log-concave and skewed, and both updates carry
    the same ~0.13 sd skew error (mode vs mean), pinned with ~2x headroom.
    Newton-3 is at least as close as Newton-1 on every metric in the
    near-Gaussian regime; in the low-rate regime only for the means -- the
    mode-based Laplace evidence is not a better estimate for a skewed
    posterior than the EKF's (2D log-likelihood error 0.025 for N3 vs 0.013
    for N1), which is approximation, not a bug.
    """

    @pytest.mark.parametrize("n_latent", [1, 2])
    def test_near_gaussian_regime_close_to_exact(self, n_latent) -> None:
        errors = _regime_errors(n_latent, _NEAR_GAUSSIAN)
        msg = _describe(errors)
        n3, n1 = errors[3], errors[1]
        # Pins (~10x headroom on N3, ~2x on N1).
        assert _worst(n3, "filt_mean") < 0.4 and _worst(n3, "smooth_mean") < 0.4, msg
        assert _worst(n3, "filt_var") < 0.25 and _worst(n3, "smooth_var") < 0.25, msg
        assert _worst(n3, "log_lik") < 0.3, msg
        assert _worst(n1, "filt_mean") < 0.25 and _worst(n1, "smooth_mean") < 0.25, msg
        assert _worst(n1, "filt_var") < 0.15 and _worst(n1, "smooth_var") < 0.15, msg
        assert _worst(n1, "log_lik") < 0.2, msg
        # The gap is real: the oracle resolves the approximation error ...
        assert _worst(n3, "filt_mean") > 1e-3, msg
        # ... and the one-step update is visibly worse than Newton-3 here.
        assert _worst(n1, "filt_mean") > 2.0 * _worst(n3, "filt_mean"), msg
        for e1, e3 in zip(n1, n3):
            for metric in _METRICS:
                assert e3[metric] <= e1[metric] + 1e-9, (metric, msg)

    @pytest.mark.parametrize("n_latent", [1, 2])
    def test_low_rate_regime_error_is_pinned(self, n_latent) -> None:
        errors = _regime_errors(n_latent, _LOW_RATE)
        msg = _describe(errors)
        for n_iter in (1, 3):
            errs = errors[n_iter]
            assert _worst(errs, "filt_mean") < 0.3, msg
            assert _worst(errs, "smooth_mean") < 0.3, msg
            assert _worst(errs, "filt_var") < 0.25, msg
            assert _worst(errs, "smooth_var") < 0.15, msg
            assert _worst(errs, "log_lik") < 0.06, msg
            # Non-zero gap: the low-rate skew error is resolved by the oracle.
            assert _worst(errs, "filt_mean") > 2e-3, msg
        for e1, e3 in zip(errors[1], errors[3]):
            assert e3["filt_mean"] <= e1["filt_mean"] + 1e-9, msg
            assert e3["smooth_mean"] <= e1["smooth_mean"] + 1e-9, msg


def _informative_problem(n_latent: int, info_scale: float, seed: int) -> _Problem:
    """T=3 problem whose expected counts scale with ``info_scale``.

    The spike counts are the rounded expected counts along a fixed latent
    path, so increasing ``info_scale`` sharpens the same likelihood rather
    than drawing new data.
    """
    rng = np.random.default_rng(seed)
    n_neurons, n_time = 4, 3
    weights = rng.normal(0.0, 0.5, (n_neurons, n_latent))
    baseline = np.full(n_neurons, np.log(5000.0))
    dt = 1e-3 * info_scale
    init_mean, init_cov = np.zeros(n_latent), 0.1 * np.eye(n_latent)
    a_diag, q_diag = np.full(n_latent, 0.95), np.full(n_latent, 0.01)
    x = rng.multivariate_normal(init_mean, init_cov)
    spikes = []
    for _ in range(n_time):
        x = a_diag * x + rng.normal(0.0, np.sqrt(q_diag))
        spikes.append(np.round(np.exp(baseline + weights @ x) * dt))
    design = np.tile(np.concatenate([baseline[:, None], weights], 1), (n_time, 1, 1))
    return _Problem(init_mean, init_cov, a_diag, q_diag, design, np.asarray(spikes), dt)


class TestLaplaceIsAsymptoticallyExact:
    """As the counts grow the converged Laplace answer tends to the truth at
    the textbook rates: mean error (in posterior sd) ~ n^{-1/2}, relative
    variance error ~ n^{-1}, log-evidence error ~ n^{-1}.

    Observed log-log slopes over expected counts x10 / x100 (seed 0, 1D /
    2D): mean -0.48 / -0.49, filtered variance -0.97 / -0.95, smoothed
    variance -0.92 / -0.82, log-likelihood -0.92 / -1.01. A missing
    or wrong constant in the Laplace normaliser, a wrong factor in the
    posterior precision or a biased mode would leave an error floor that
    does not vanish, flattening the slopes.
    """

    @pytest.mark.parametrize("n_latent", [1, 2])
    def test_error_vanishes_at_laplace_rates(self, n_latent) -> None:
        scales = np.array([1e3, 1e4, 1e5])
        errs = []
        for scale in scales:
            problem = _informative_problem(n_latent, scale, seed=0)
            laplace = _run_laplace(problem, max_newton_iter=30)
            sd = np.sqrt(np.diagonal(laplace["filt_cov"], axis1=1, axis2=2))
            lo = (laplace["filt_mean"] - 12 * sd).min(0)
            hi = (laplace["filt_mean"] + 12 * sd).max(0)
            n_points = 3001 if n_latent == 1 else 201
            grids = [np.linspace(lo[i], hi[i], n_points) for i in range(n_latent)]
            errs.append(_standardized_errors(laplace, _grid_posterior(problem, grids)))
        slope = {
            m: np.polyfit(np.log10(scales), np.log10([e[m] for e in errs]), 1)[0]
            for m in _METRICS
        }
        msg = f"slopes {slope}; errors {errs}"
        assert -0.6 < slope["filt_mean"] < -0.4, msg
        assert -0.6 < slope["smooth_mean"] < -0.4, msg
        assert -1.15 < slope["filt_var"] < -0.85, msg
        assert -1.25 < slope["smooth_var"] < -0.7, msg
        assert -1.2 < slope["log_lik"] < -0.75, msg
        # The errors at the smallest scale are genuinely resolved (not roundoff).
        assert errs[0]["filt_mean"] > 1e-3 and errs[0]["log_lik"] > 1e-4, msg


# ---------------------------------------------------------------------------
# 3. EM M-step exactness: the expected complete-data log-likelihood Q(theta)
# ---------------------------------------------------------------------------


def _smoothed_x0(init_mean, init_cov, A, Q, m1, P1):
    """Smoothed x_0 moments and Cov(x_0, x_1 | y), written out in NumPy.

    x_0 is unobserved, so p(x_0 | y) = int p(x_0 | x_1) p(x_1 | y) dx_1 with
    the Gaussian backward kernel of the prior: exact for any posterior on
    x_1 with the given mean and covariance.
    """
    P_pred = A @ init_cov @ A.T + Q
    J0 = init_cov @ A.T @ np.linalg.inv(P_pred)
    m0 = init_mean + J0 @ (m1 - A @ init_mean)
    P0 = init_cov + J0 @ (P1 - P_pred) @ J0.T
    return m0, P0, J0 @ P1


def _expected_log_normal(second_moment, cov):
    """E[log N(r; 0, cov)] when E[r r'] = second_moment."""
    d = cov.shape[0]
    return -0.5 * (
        d * np.log(2 * np.pi)
        + np.linalg.slogdet(cov)[1]
        + np.trace(np.linalg.solve(cov, second_moment))
    )


def _q_function(params, means, covs, cross):
    """Expected complete-data log-likelihood of the dynamics and x_0 prior.

    ``means`` / ``covs`` hold the smoothed moments of x_0 .. x_T and
    ``cross[t] = Cov(x_t, x_{t+1} | y)``, t = 0 .. T-1::

        Q = E[log N(x_0; m0, P0)] + sum_{t=1}^T E[log N(x_t; A x_{t-1}, Q)]

    (the observation term does not depend on these parameters).
    """
    A, Q, m0, P0 = params["A"], params["Q"], params["m0"], params["P0"]
    r0 = means[0] - m0
    total = _expected_log_normal(covs[0] + np.outer(r0, r0), P0)
    for t in range(1, means.shape[0]):
        e_xx = covs[t] + np.outer(means[t], means[t])
        e_pp = covs[t - 1] + np.outer(means[t - 1], means[t - 1])
        e_px = cross[t - 1] + np.outer(means[t - 1], means[t])  # E[x_{t-1} x_t']
        second = e_xx - A @ e_px - e_px.T @ A.T + A @ e_pp @ A.T
        total += _expected_log_normal(second, Q)
    return total


def _symmetric_directions(d, diagonal_only=False):
    dirs = []
    for i in range(d):
        for j in range(i, d):
            if diagonal_only and i != j:
                continue
            e = np.zeros((d, d))
            e[i, j] = e[j, i] = 1.0
            dirs.append(e)
    return dirs


def _unit_directions(shape):
    dirs = []
    for idx in np.ndindex(*shape):
        e = np.zeros(shape)
        e[idx] = 1.0
        dirs.append(e)
    return dirs


def _fd_gradient(f, params, directions, h=1e-6):
    """Central differences of f along each (name, direction) pair."""
    grads = []
    for name, direction in directions:
        plus = dict(params, **{name: params[name] + h * direction})
        minus = dict(params, **{name: params[name] - h * direction})
        grads.append((f(plus) - f(minus)) / (2 * h))
    return np.array(grads)


def _random_psd(rng, d, jitter):
    m = rng.normal(size=(d, d))
    return m @ m.T / d + jitter * np.eye(d)


def _random_smoother_moments(rng, n_time, d):
    """Marginal and lag-one blocks of a random joint Gaussian over x_1..x_T."""
    joint = _random_psd(rng, n_time * d, 0.05)
    blocks = joint.reshape(n_time, d, n_time, d)
    covs = np.stack([blocks[t, :, t, :] for t in range(n_time)])
    cross = np.array([blocks[t, :, t + 1, :] for t in range(n_time - 1)]).reshape(
        n_time - 1, d, d
    )
    return rng.normal(0.0, 1.0, (n_time, d)), covs, cross


def _augment_with_x0(prior, means, covs, cross):
    m0, P0, C01 = _smoothed_x0(
        prior["m0"], prior["P0"], prior["A"], prior["Q"], means[0], covs[0]
    )
    return (
        np.concatenate([m0[None], means]),
        np.concatenate([P0[None], covs]),
        np.concatenate([C01[None], cross]),
    )


class TestDynamicsMStepMaximisesQ:
    """dynamics_only_m_step with an initial_state_prior is exact EM."""

    @settings(deadline=None, max_examples=10)
    @given(
        seed=st.integers(0, 2**31 - 1),
        n_latent=st.integers(1, 3),
        n_time=st.integers(1, 5),
        fix_transition=st.booleans(),
    )
    def test_gradient_of_q_vanishes_at_update(
        self, seed, n_latent, n_time, fix_transition
    ) -> None:
        rng = np.random.default_rng(seed)
        means, covs, cross = _random_smoother_moments(rng, n_time, n_latent)
        old = {
            "A": 0.5 * np.eye(n_latent) + rng.uniform(-0.3, 0.3, (n_latent, n_latent)),
            "Q": _random_psd(rng, n_latent, 0.05),
            "m0": rng.normal(0.0, 1.0, n_latent),
            "P0": _random_psd(rng, n_latent, 0.1),
        }
        A_new, Q_new, m0_new, P0_new = dynamics_only_m_step(
            jnp.asarray(means),
            jnp.asarray(covs),
            jnp.asarray(cross),
            fixed_transition_matrix=jnp.asarray(old["A"]) if fix_transition else None,
            initial_state_prior=InitialStatePrior(
                jnp.asarray(old["m0"]),
                jnp.asarray(old["P0"]),
                jnp.asarray(old["A"]),
                jnp.asarray(old["Q"]),
            ),
        )
        new = {
            "A": np.asarray(A_new),
            "Q": np.asarray(Q_new),
            "m0": np.asarray(m0_new),
            "P0": np.asarray(P0_new),
        }
        # The E-step moments, x_0 included, are fixed by the OLD parameters.
        aug = _augment_with_x0(old, means, covs, cross)

        def q(params):
            return _q_function(params, *aug)

        directions = (
            [("Q", e) for e in _symmetric_directions(n_latent)]
            + [("m0", e) for e in _unit_directions((n_latent,))]
            + [("P0", e) for e in _symmetric_directions(n_latent)]
        )
        if fix_transition:
            np.testing.assert_array_equal(new["A"], old["A"])
        else:
            directions += [("A", e) for e in _unit_directions((n_latent, n_latent))]
        g_old = _fd_gradient(q, old, directions)
        g_new = _fd_gradient(q, new, directions)
        # Guard: the old parameters are not already stationary.
        assert np.max(np.abs(g_old)) > 1e-2
        assert np.max(np.abs(g_new)) < 1e-6 * max(1.0, np.max(np.abs(g_old))), (
            g_new,
            g_old,
        )
        assert q(new) >= q(old) - 1e-10 * abs(q(old))

    def test_legacy_path_without_prior_is_not_exact(self) -> None:
        """Without the prior the x_0 -> x_1 transition is dropped: the update
        is stationary for the T-1-transition objective, not the full Q."""
        rng = np.random.default_rng(3)
        means, covs, cross = _random_smoother_moments(rng, 4, 2)
        prior = {
            "A": 0.8 * np.eye(2),
            "Q": 0.2 * np.eye(2),
            "m0": np.zeros(2),
            "P0": np.eye(2),
        }
        A_new, Q_new, _, _ = dynamics_only_m_step(
            jnp.asarray(means), jnp.asarray(covs), jnp.asarray(cross)
        )
        aug = _augment_with_x0(prior, means, covs, cross)
        params = dict(prior, A=np.asarray(A_new), Q=np.asarray(Q_new))
        directions = [("A", e) for e in _unit_directions((2, 2))]

        def q_full(p):
            return _q_function(p, *aug)

        def q_observed_transitions(p):
            # drop the x_0 prior and the x_0 -> x_1 transition
            return _q_function(p, means, covs, cross) - _expected_log_normal(
                covs[0] + np.outer(means[0] - p["m0"], means[0] - p["m0"]), p["P0"]
            )

        g_legacy = _fd_gradient(q_observed_transitions, params, directions)
        g_full = _fd_gradient(q_full, params, directions)
        assert np.max(np.abs(g_legacy)) < 1e-6
        assert np.max(np.abs(g_full)) > 1e-3


def _place_field_model_with_moments(
    rng, n_neurons, block_size, n_time, structure, update_A, as_blocks
):
    """A PlaceFieldModel holding block-diagonal E-step output and parameters."""
    n_state = n_neurons * block_size
    blocks = [
        _random_smoother_moments(rng, n_time, block_size) for _ in range(n_neurons)
    ]
    means = np.concatenate([b[0] for b in blocks], axis=1)
    cov_blocks = np.stack([b[1] for b in blocks])  # (n_neurons, T, bs, bs)
    cross_blocks = np.stack([b[2] for b in blocks])
    model = PlaceFieldModel(
        dt=0.01,
        process_noise_structure=structure,
        update_transition_matrix=update_A,
    )
    model.n_neurons = n_neurons
    model.n_basis_per_neuron = block_size
    model.n_basis = n_state
    model.transition_matrix = jnp.eye(n_state)
    model.process_cov = jnp.diag(jnp.asarray(rng.uniform(0.05, 0.3, n_state)))
    model.init_mean = jnp.asarray(rng.normal(0.0, 1.0, n_state))
    model.init_cov = jnp.diag(jnp.asarray(rng.uniform(0.2, 1.0, n_state)))
    dense_covs = np.asarray(BlockDiagonalCovariance(cov_blocks).to_dense())
    dense_cross = np.asarray(BlockDiagonalCovariance(cross_blocks).to_dense())
    model.smoother_mean = jnp.asarray(means)
    if as_blocks:
        model.smoother_cov = BlockDiagonalCovariance(cov_blocks)
        model.smoother_cross_cov = BlockDiagonalCovariance(cross_blocks)
    else:
        model.smoother_cov = jnp.asarray(dense_covs)
        model.smoother_cross_cov = jnp.asarray(dense_cross)
    return model, means, dense_covs, dense_cross


class TestPlaceFieldMStepMaximisesQ:
    """PlaceFieldModel._m_step is exact EM under its constraints.

    Constraints: A = I unless ``update_transition_matrix`` (then A is a full
    matrix), Q diagonal (``"diagonal"``) or a multiple of I
    (``"isotropic"``), init_cov diagonal. The gradient of Q(theta) is taken
    only along those feasible directions.
    """

    @pytest.mark.parametrize("structure", ["diagonal", "isotropic"])
    @pytest.mark.parametrize("update_A", [False, True])
    @pytest.mark.parametrize("as_blocks", [False, True], ids=["dense", "blocks"])
    def test_gradient_of_q_vanishes_at_update(
        self, structure, update_A, as_blocks
    ) -> None:
        rng = np.random.default_rng(7)
        model, means, covs, cross = _place_field_model_with_moments(
            rng, 2, 2, 5, structure, update_A, as_blocks
        )
        n_state = means.shape[1]
        old = {
            "A": np.asarray(model.transition_matrix),
            "Q": np.asarray(model.process_cov),
            "m0": np.asarray(model.init_mean),
            "P0": np.asarray(model.init_cov),
        }
        aug = _augment_with_x0(old, means, covs, cross)
        model._m_step()
        new = {
            "A": np.asarray(model.transition_matrix),
            "Q": np.asarray(model.process_cov),
            "m0": np.asarray(model.init_mean),
            "P0": np.asarray(model.init_cov),
        }

        def q(params):
            return _q_function(params, *aug)

        q_dirs = (
            [np.eye(n_state)]
            if structure == "isotropic"
            else _symmetric_directions(n_state, diagonal_only=True)
        )
        directions = (
            [("Q", e) for e in q_dirs]
            + [("m0", e) for e in _unit_directions((n_state,))]
            + [("P0", e) for e in _symmetric_directions(n_state, diagonal_only=True)]
        )
        if update_A:
            directions += [("A", e) for e in _unit_directions((n_state, n_state))]
        else:
            np.testing.assert_array_equal(new["A"], np.eye(n_state))
        # Constraints hold.
        np.testing.assert_array_equal(new["Q"], np.diag(np.diag(new["Q"])))
        np.testing.assert_array_equal(new["P0"], np.diag(np.diag(new["P0"])))
        g_old = _fd_gradient(q, old, directions)
        g_new = _fd_gradient(q, new, directions)
        assert np.max(np.abs(g_old)) > 1e-2
        assert np.max(np.abs(g_new)) < 1e-6 * max(1.0, np.max(np.abs(g_old))), (
            g_new,
            g_old,
        )
        assert q(new) >= q(old)

    def test_block_container_matches_dense(self) -> None:
        """The block-diagonal container path gives the same update."""
        results = []
        for as_blocks in (False, True):
            model, *_ = _place_field_model_with_moments(
                np.random.default_rng(11), 3, 2, 4, "diagonal", True, as_blocks
            )
            model._m_step()
            results.append(
                [
                    np.asarray(x)
                    for x in (
                        model.transition_matrix,
                        model.process_cov,
                        model.init_mean,
                        model.init_cov,
                    )
                ]
            )
        for dense, block in zip(*results):
            np.testing.assert_allclose(block, dense, rtol=1e-12, atol=1e-14)


class TestStationaryGLMFitIsStationary:
    """The observation-side GLM fit (PlaceFieldModel warm start) returns the
    maximiser of its penalised Poisson log-likelihood and its Laplace
    covariance.

    ``L(w) = sum_t [y_t log mu_t - mu_t] - 0.5 * prior_precision * |w|^2``
    with ``mu_t = exp(Z_t w) dt``.
    """

    def test_map_is_stationary_and_covariance_is_inverse_hessian(self) -> None:
        rng = np.random.default_rng(0)
        n_time, n_basis, dt = 400, 5, 0.01
        Z = rng.uniform(0.0, 1.0, (n_time, n_basis))
        Z /= Z.sum(axis=1, keepdims=True)  # partition of unity, like B-splines
        w_true = np.log(20.0) + rng.normal(0.0, 0.7, (2, n_basis))
        spikes = rng.poisson(np.exp(Z @ w_true.T) * dt).astype(float)
        model = PlaceFieldModel(dt=dt)
        model.n_basis_per_neuron = n_basis
        w_all, cov_all = model._fit_stationary_glm(jnp.asarray(Z), jnp.asarray(spikes))
        w_all = np.asarray(w_all).reshape(2, n_basis)
        cov_all = np.asarray(cov_all)
        dirs = [("w", e) for e in _unit_directions((n_basis,))]
        for j in range(2):
            y = spikes[:, j]

            def objective(p, y=y):
                eta = Z @ p["w"]
                log_lik = np.sum(y * (eta + np.log(dt)) - np.exp(eta) * dt)
                return log_lik - 0.5 * np.sum(p["w"] ** 2)

            w = w_all[j]
            g = _fd_gradient(objective, {"w": w}, dirs)
            g_zero = _fd_gradient(objective, {"w": np.zeros(n_basis)}, dirs)
            assert np.max(np.abs(g_zero)) > 1.0  # guard
            assert np.max(np.abs(g)) < 1e-6 * np.max(np.abs(g_zero)), g
            assert objective({"w": w}) > objective({"w": np.zeros(n_basis)})
            for e in np.eye(n_basis):  # a maximum, not just a stationary point
                assert objective({"w": w}) >= objective({"w": w + 1e-3 * e})
                assert objective({"w": w}) >= objective({"w": w - 1e-3 * e})
            # Laplace covariance = inverse of -Hessian of L at the MAP:
            # -Hess = Z' diag(mu) Z + prior_precision * I.
            mu = np.exp(Z @ w) * dt
            neg_hess = Z.T @ (mu[:, None] * Z) + np.eye(n_basis)
            block = cov_all[
                j * n_basis : (j + 1) * n_basis, j * n_basis : (j + 1) * n_basis
            ]
            np.testing.assert_allclose(block, np.linalg.inv(neg_hess), rtol=1e-9)
        # Neurons are independent a posteriori: off-diagonal blocks are zero.
        assert np.all(cov_all[:n_basis, n_basis:] == 0.0)
