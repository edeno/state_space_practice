# ruff: noqa: E402
"""State-space Gaussian processes against dense-kernel ground truth.

A Matern-3/2 GP driven as the linear SDE in :mod:`state_space_practice.gp_ssm`
is *mathematically identical* to dense GP regression with the kernel

    k(tau) = s2 (1 + a|tau|) exp(-a|tau|),   a = sqrt(3) / lengthscale.

Every reference below is written from that kernel alone in plain NumPy (Gram
matrices, Cholesky solves, Newton on the dense log-posterior, 1-D quadrature)
and shares no code with the library, so agreement to ~1e-8 relative is
independent evidence that

* :func:`matern32_discretize` reproduces the kernel exactly, including on
  irregular grids (the SDE state is ``[f, f']``, so the derivative
  covariances ``dk/ds`` and ``-d2k/dtau2`` are checked too);
* the Kalman filter/smoother fed with those matrices returns the exact dense GP
  posterior mean, marginal variance, full posterior covariance and marginal
  log-likelihood;
* :func:`infer_log_rate` returns the exact dense GP-Laplace mode, Laplace
  variance and Laplace evidence (Rasmussen & Williams 2006, Alg. 3.1 and
  eq. 3.32) -- and the *approximation* gap of that Laplace evidence against the
  exact (quadrature) evidence is pinned, nonzero, and vanishes in the Gaussian
  (large-count) limit;
* :meth:`TemporalRateGP.fit_sgd` stops at a stationary point of the dense
  Laplace evidence (finite differences of the dense oracle).
"""

import jax

jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from scipy.special import gammaln, logsumexp

from state_space_practice.gp_ssm import matern32_continuous, matern32_discretize
from state_space_practice.kalman import kalman_filter, kalman_smoother
from state_space_practice.temporal_rate_gp import (
    TemporalRateGP,
    _infer_log_rate_traced,
    infer_log_rate,
    poisson_log_rate_site,
)
from state_space_practice.tests.oracles import (
    lgssm_dense_posterior,
    lgssm_joint_prior,
)

# Relative agreement demanded between the recursions and the dense references.
# Observed agreement is ~1e-12..1e-14 over the strategies below.
RTOL = 1e-8

# The Hypothesis sweeps call the library many times; eager execution re-traces
# every scan on each call (~2 s), so they go through jitted handles instead.
# ``infer_log_rate`` is a validating wrapper around ``_infer_log_rate_traced``
# (``test_jitted_core_equals_public_infer_log_rate`` pins the equivalence), and
# ``validate_inputs=False`` only skips host-side checks in the Kalman routines.
# A sampled handful of sequence lengths bounds the number of compilations.
_laplace_jit = jax.jit(_infer_log_rate_traced, static_argnums=(5, 6))
_kalman_filter_jit = jax.jit(partial(kalman_filter, validate_inputs=False))
_kalman_smoother_jit = jax.jit(partial(kalman_smoother, validate_inputs=False))
_N_TIME = st.sampled_from((3, 4, 9, 23, 40))


def _laplace(counts, dt, variance, lengthscale, mean, n_iter=60):
    return _laplace_jit(
        jnp.asarray(counts, dtype=float), dt, variance, lengthscale, mean, n_iter, 1e-9
    )


# ---------------------------------------------------------------------------
# Dense Matern-3/2 references (NumPy, from the kernel formula only)
# ---------------------------------------------------------------------------


def matern32_value_derivative_cov(
    times_a: np.ndarray, times_b: np.ndarray, variance: float, lengthscale: float
) -> np.ndarray:
    """Joint covariance of ``[f(t), f'(t)]`` at ``times_a`` against ``times_b``.

    With ``tau = t - s`` and ``a = sqrt(3)/l``:

    - ``Cov[f(t), f(s)]   = s2 (1 + a|tau|) e^{-a|tau|}``
    - ``Cov[f(t), f'(s)]  = d/ds k(t - s) = s2 a^2 tau e^{-a|tau|}``
    - ``Cov[f'(t), f(s)]  = d/dt k(t - s) = -s2 a^2 tau e^{-a|tau|}``
    - ``Cov[f'(t), f'(s)] = -k''(tau) = s2 a^2 (1 - a|tau|) e^{-a|tau|}``

    Returns an array of shape ``(2 * len(times_a), 2 * len(times_b))`` with
    ``[f, f']`` interleaved per time (the state-space block layout).
    """
    a = np.sqrt(3.0) / lengthscale
    tau = np.asarray(times_a)[:, None] - np.asarray(times_b)[None, :]
    abs_tau = np.abs(tau)
    decay = np.exp(-a * abs_tau)
    k_ff = variance * (1.0 + a * abs_tau) * decay
    k_fd = variance * a**2 * tau * decay
    k_df = -k_fd
    k_dd = variance * a**2 * (1.0 - a * abs_tau) * decay
    out = np.empty((2 * tau.shape[0], 2 * tau.shape[1]))
    out[0::2, 0::2] = k_ff
    out[0::2, 1::2] = k_fd
    out[1::2, 0::2] = k_df
    out[1::2, 1::2] = k_dd
    return out


def matern32_gram(times: np.ndarray, variance: float, lengthscale: float) -> np.ndarray:
    """Value-only Gram matrix ``K[i, j] = k(t_i - t_j)``."""
    return matern32_value_derivative_cov(times, times, variance, lengthscale)[
        0::2, 0::2
    ]


def dense_gp_regression(
    times: np.ndarray,
    y: np.ndarray,
    variance: float,
    lengthscale: float,
    noise_var: np.ndarray,
) -> dict:
    """Exact GP regression ``y_i = f(t_i) + e_i``, ``e_i ~ N(0, noise_var_i)``.

    Returns the posterior mean and full covariance of the interleaved state
    ``[f(t_i), f'(t_i)]``, the filtered moments (conditioning on the prefix
    ``y_{1:i}``) and the log marginal likelihood ``log N(y; 0, K + D)``.
    """
    n = times.size
    noise_var = np.broadcast_to(np.asarray(noise_var, dtype=float), (n,))
    k_state = matern32_value_derivative_cov(times, times, variance, lengthscale)
    k_state_y = k_state[:, 0::2]  # Cov[state, f] == Cov[state, y]
    k_yy = k_state[0::2, 0::2] + np.diag(noise_var)

    chol = np.linalg.cholesky(k_yy)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    v = np.linalg.solve(chol, k_state_y.T)
    post_mean = k_state_y @ alpha
    post_cov = k_state - v.T @ v
    log_ml = float(
        -0.5 * y @ alpha - np.sum(np.log(np.diag(chol))) - 0.5 * n * np.log(2 * np.pi)
    )

    filt_mean = np.zeros((n, 2))
    filt_cov = np.zeros((n, 2, 2))
    for i in range(n):
        k_prefix = k_yy[: i + 1, : i + 1]
        c = k_state_y[2 * i : 2 * i + 2, : i + 1]
        sol = np.linalg.solve(k_prefix, np.column_stack([y[: i + 1], c.T]))
        filt_mean[i] = c @ sol[:, 0]
        filt_cov[i] = k_state[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] - c @ sol[:, 1:]
    return {
        "mean": post_mean.reshape(n, 2),
        "cov": 0.5 * (post_cov + post_cov.T),
        "filtered_mean": filt_mean,
        "filtered_cov": filt_cov,
        "log_ml": log_ml,
    }


def dense_laplace_lgcp(
    counts: np.ndarray,
    dt: float,
    variance: float,
    lengthscale: float,
    mean: float = 0.0,
    min_weight: float = 1e-9,
    max_newton: int = 200,
) -> dict:
    """Dense GP-Laplace for ``y_i ~ Poisson(exp(mean + g_i) dt)``, ``g ~ GP``.

    Newton iteration in the numerically stable ``B = I + W^1/2 K W^1/2`` form
    of Rasmussen & Williams (2006) Algorithm 3.1 (no ``K^-1``), Fisher weights
    floored at ``min_weight`` exactly as documented for the library, with
    step halving on ``Psi(a) = log p(y | K a) - a^T K a / 2`` (R&W sec. 3.4.1:
    ``g = K a`` is linear in ``a``, so the line search needs no inverse).
    Returns the mode of ``g``, the Laplace marginal variance and the Laplace
    evidence (R&W eq. 3.32).
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.size
    offset = mean + np.log(dt)
    gram = matern32_gram(np.arange(n) * dt, variance, lengthscale)

    def psi(a_vec: np.ndarray) -> float:
        f = gram @ a_vec
        with np.errstate(over="ignore"):
            return float(
                np.sum(counts * (f + offset) - np.exp(f + offset)) - 0.5 * a_vec @ f
            )

    a = np.zeros(n)
    g = np.zeros(n)
    for _ in range(max_newton):
        rate = np.exp(g + offset)
        weight = np.maximum(rate, min_weight)
        sw = np.sqrt(weight)
        b_mat = np.eye(n) + sw[:, None] * gram * sw[None, :]
        chol = np.linalg.cholesky(b_mat)
        b = weight * g + (counts - rate)
        rhs = sw * (gram @ b)
        a_newton = b - sw * np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
        psi_old, step = psi(a), 1.0
        while psi(a + step * (a_newton - a)) < psi_old - 1e-12 * abs(psi_old):
            step *= 0.5
            if step < 1e-6:
                break
        a = a + step * (a_newton - a)
        g_new = gram @ a
        change = np.max(np.abs(g_new - g))
        g = g_new
        if change < 1e-14:
            break
    rate = np.exp(g + offset)
    weight = np.maximum(rate, min_weight)
    sw = np.sqrt(weight)
    b_mat = np.eye(n) + sw[:, None] * gram * sw[None, :]
    chol = np.linalg.cholesky(b_mat)
    v = np.linalg.solve(chol, sw[:, None] * gram)
    post_var = np.diag(gram) - np.sum(v * v, axis=0)
    log_lik = np.sum(counts * (g + offset) - rate - gammaln(counts + 1.0))
    evidence = -0.5 * a @ g + log_lik - np.sum(np.log(np.diag(chol)))
    return {"mode": g, "var": post_var, "evidence": float(evidence)}


def quadrature_poisson_normal(
    count: float, log_expected_offset: float, prior_var: float, n_grid: int = 40001
) -> dict:
    """Exact posterior of ``g ~ N(0, v)``, ``y ~ Poisson(exp(g + offset))``.

    A 1-D GP (a single time bin) is a single Gaussian variable, so its exact
    evidence, posterior mean and posterior variance are 1-D integrals, done
    here on a dense uniform grid spanning +-40 Laplace standard deviations
    around the mode (the integrand is super-exponentially small outside it).
    """
    # locate the mode by bisection on the (monotone) score
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        score = count - np.exp(mid + log_expected_offset) - mid / prior_var
        lo, hi = (mid, hi) if score > 0 else (lo, mid)
    mode = 0.5 * (lo + hi)
    local_sd = 1.0 / np.sqrt(np.exp(mode + log_expected_offset) + 1.0 / prior_var)
    half_width = 40.0 * local_sd
    grid = np.linspace(mode - half_width, mode + half_width, n_grid)
    log_joint = (
        count * (grid + log_expected_offset)
        - np.exp(grid + log_expected_offset)
        - gammaln(count + 1.0)
        - 0.5 * grid**2 / prior_var
        - 0.5 * np.log(2 * np.pi * prior_var)
    )
    h = grid[1] - grid[0]
    log_z = float(logsumexp(log_joint) + np.log(h))
    weights = np.exp(log_joint - log_z) * h
    post_mean = float(np.sum(weights * grid))
    post_var = float(np.sum(weights * (grid - post_mean) ** 2))
    return {"log_z": log_z, "mean": post_mean, "var": post_var, "mode": mode}


def _state_space_gp_regression(y, dt, variance, lengthscale, noise_var, jit=False):
    """Library path: Matern-3/2 SSM + Kalman filter/smoother (regular grid)."""
    filt, smooth = (
        (_kalman_filter_jit, _kalman_smoother_jit)
        if jit
        else (kalman_filter, kalman_smoother)
    )
    _F, _L, _Qc, h_vec, p_inf = matern32_continuous(variance, lengthscale)
    a_mat, q_mat = matern32_discretize(variance, lengthscale, dt)
    obs = jnp.asarray(y)[:, None]
    h_mat = h_vec[None, :]
    r_mat = jnp.array([[noise_var]])
    fm, fc, filt_ll = filt(jnp.zeros(2), p_inf, obs, a_mat, q_mat, h_mat, r_mat)
    sm, sc, cross, smooth_ll = smooth(
        jnp.zeros(2), p_inf, obs, a_mat, q_mat, h_mat, r_mat
    )
    return {
        "filtered_mean": np.asarray(fm),
        "filtered_cov": np.asarray(fc),
        "filter_ll": float(filt_ll),
        "mean": np.asarray(sm),
        "cov": np.asarray(sc),
        "cross": np.asarray(cross),
        "smoother_ll": float(smooth_ll),
    }


def _full_covariance_from_smoother(cov: np.ndarray, cross: np.ndarray) -> np.ndarray:
    """Rebuild the full joint posterior covariance of a Markov chain.

    With ``C_t = Cov[x_t, x_{t+1} | y]`` and ``P_t = Cov[x_t | y]`` the RTS gain
    is ``G_t = C_t P_{t+1}^{-1}`` and ``Cov[x_i, x_j | y] = G_i ... G_{j-1} P_j``
    for ``i < j``.
    """
    n_time, n = cov.shape[:2]
    gains = [np.linalg.solve(cov[t + 1].T, cross[t].T).T for t in range(n_time - 1)]
    full = np.zeros((n_time * n, n_time * n))
    for j in range(n_time):
        block = cov[j]
        full[j * n : (j + 1) * n, j * n : (j + 1) * n] = block
        for i in range(j - 1, -1, -1):
            block = gains[i] @ block
            full[i * n : (i + 1) * n, j * n : (j + 1) * n] = block
            full[j * n : (j + 1) * n, i * n : (i + 1) * n] = block.T
    return full


def _scale_tol(reference: np.ndarray) -> float:
    return RTOL * float(np.max(np.abs(reference)))


# Hyperparameter strategies. ``lengthscale / dt`` spans 0.3..30 (from nearly
# white to very smooth relative to the grid); the observation noise is at
# least 1e-2 of the signal variance so the dense Gram stays well conditioned.
_variance = st.floats(0.1, 10.0)
_lengthscale = st.floats(0.05, 3.0)
_noise_ratio = st.floats(1e-2, 2.0)


# ---------------------------------------------------------------------------
# 1. Gaussian likelihood: SSM == dense GP regression
# ---------------------------------------------------------------------------


def _check_regression_against_dense(
    variance, lengthscale, dt, noise_var, n_time, seed, jit
):
    rng = np.random.default_rng(seed)
    times = np.arange(n_time) * dt
    gram = matern32_gram(times, variance, lengthscale)
    f_true = np.linalg.cholesky(gram + 1e-12 * variance * np.eye(n_time)) @ (
        rng.standard_normal(n_time)
    )
    y = f_true + np.sqrt(noise_var) * rng.standard_normal(n_time)

    dense = dense_gp_regression(times, y, variance, lengthscale, noise_var)
    ssm = _state_space_gp_regression(y, dt, variance, lengthscale, noise_var, jit)

    for key in ("filtered_mean", "filtered_cov", "mean"):
        np.testing.assert_allclose(
            ssm[key], dense[key], rtol=RTOL, atol=_scale_tol(dense[key]), err_msg=key
        )
    dense_marginal_cov = np.stack(
        [dense["cov"][2 * t : 2 * t + 2, 2 * t : 2 * t + 2] for t in range(n_time)]
    )
    np.testing.assert_allclose(
        ssm["cov"], dense_marginal_cov, rtol=RTOL, atol=_scale_tol(dense_marginal_cov)
    )
    # The full covariance is rebuilt by chaining up to 39 RTS gains, each an
    # inverse of a posterior covariance, so it is compared 100x looser.
    full = _full_covariance_from_smoother(ssm["cov"], ssm["cross"])
    np.testing.assert_allclose(
        full, dense["cov"], rtol=1e2 * RTOL, atol=1e2 * _scale_tol(dense["cov"])
    )
    np.testing.assert_allclose(ssm["filter_ll"], dense["log_ml"], rtol=RTOL, atol=1e-9)
    np.testing.assert_allclose(
        ssm["smoother_ll"], dense["log_ml"], rtol=RTOL, atol=1e-9
    )
    # Guard: the data were informative (posterior variance strictly below prior).
    assert np.all(dense_marginal_cov[:, 0, 0] < variance)


def test_kalman_matern32_regression_equals_dense_gp():
    """One fixed case through the public (validating, eager) Kalman entry points."""
    _check_regression_against_dense(1.7, 0.35, 0.08, 0.3, 25, seed=11, jit=False)


@pytest.mark.slow
@settings(max_examples=60, deadline=None)
@given(
    variance=_variance,
    lengthscale=_lengthscale,
    dt_ratio=st.floats(1.0 / 30.0, 3.0),
    noise_ratio=_noise_ratio,
    n_time=_N_TIME,
    seed=st.integers(0, 2**31 - 1),
)
def test_kalman_matern32_regression_equals_dense_gp_sweep(
    variance, lengthscale, dt_ratio, noise_ratio, n_time, seed
):
    """Filtered/smoothed moments, full covariance and log-ML match the dense GP."""
    _check_regression_against_dense(
        variance,
        lengthscale,
        dt_ratio * lengthscale,
        noise_ratio * variance,
        n_time,
        seed,
        jit=True,
    )


def test_full_covariance_reconstruction_is_not_vacuous():
    """The dense full-covariance check sees off-diagonal (non-Markov) structure.

    The value-only posterior covariance of a Matern-3/2 GP is dense (its inverse
    is not banded in ``f`` alone), so a reconstruction that dropped the
    derivative state -- a first-order Markov (OU) approximation -- would fail.
    """
    variance, lengthscale, dt, noise_var, n_time = 1.3, 0.4, 0.1, 0.2, 12
    rng = np.random.default_rng(0)
    y = rng.standard_normal(n_time)
    ssm = _state_space_gp_regression(y, dt, variance, lengthscale, noise_var)
    dense = dense_gp_regression(np.arange(n_time) * dt, y, variance, lengthscale, 0.2)
    ff = dense["cov"][0::2, 0::2]
    # OU-style reconstruction from value-only marginals and lag-one covariances
    value_cov = ssm["cov"][:, 0, 0]
    value_cross = ssm["cross"][:, 0, 0]
    ou = np.diag(value_cov)
    for j in range(n_time):
        for i in range(j - 1, -1, -1):
            ou[i, j] = ou[j, i] = (value_cross[i] / value_cov[i + 1]) * ou[i + 1, j]
    assert np.max(np.abs(ou - ff)) > 1e-3
    full = _full_covariance_from_smoother(ssm["cov"], ssm["cross"])
    np.testing.assert_allclose(full[0::2, 0::2], ff, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Irregular grids: the discretised SDE prior == the dense kernel
# ---------------------------------------------------------------------------


def _irregular_ssm_matrices(times, variance, lengthscale):
    """Per-step (A_k, Q_k) from matern32_discretize for arbitrary spacings."""
    dts = np.diff(times)
    mats = [matern32_discretize(variance, lengthscale, float(d)) for d in dts]
    a_seq = np.stack([np.asarray(m[0]) for m in mats])
    q_seq = np.stack([np.asarray(m[1]) for m in mats])
    # prior_on_first_state=True replaces step 0 by the identity, so pad.
    a_seq = np.concatenate([np.eye(2)[None], a_seq])
    q_seq = np.concatenate([np.zeros((1, 2, 2)), q_seq])
    return a_seq, q_seq


@settings(max_examples=25, deadline=None)
@given(
    variance=_variance,
    lengthscale=_lengthscale,
    n_time=st.integers(3, 40),
    seed=st.integers(0, 2**31 - 1),
    log_spread=st.floats(-3.0, 1.0),
)
def test_irregular_grid_prior_equals_dense_kernel(
    variance, lengthscale, n_time, seed, log_spread
):
    """Joint prior of ``[f, f']`` from chained ``(A_k, Q_k)`` equals the kernel.

    The spacings are drawn log-uniformly over ~3 decades (down to ``1e-3 l``,
    which exercises the small-``dt`` series branch of ``Q``).
    """
    rng = np.random.default_rng(seed)
    dts = lengthscale * np.exp(rng.uniform(-7.0, log_spread, size=n_time - 1))
    times = np.concatenate([[0.0], np.cumsum(dts)])
    _F, _L, _Qc, _H, p_inf = matern32_continuous(variance, lengthscale)
    a_seq, q_seq = _irregular_ssm_matrices(times, variance, lengthscale)
    _mx, cov_x, _my, _cy, _cxy = lgssm_joint_prior(
        np.zeros(2),
        np.asarray(p_inf),
        a_seq,
        q_seq,
        np.array([[1.0, 0.0]]),
        np.array([[1.0]]),
        n_time,
        prior_on_first_state=True,
    )
    ssm_prior = cov_x[2:, 2:]  # drop the unused x_0 block
    dense_prior = matern32_value_derivative_cov(times, times, variance, lengthscale)
    np.testing.assert_allclose(
        ssm_prior, dense_prior, rtol=RTOL, atol=_scale_tol(dense_prior)
    )


@settings(max_examples=20, deadline=None)
@given(
    variance=_variance,
    lengthscale=_lengthscale,
    noise_ratio=_noise_ratio,
    n_time=st.integers(3, 30),
    seed=st.integers(0, 2**31 - 1),
)
def test_irregular_grid_posterior_equals_dense_gp(
    variance, lengthscale, noise_ratio, n_time, seed
):
    """Conditioning the irregular-grid SSM equals dense GP regression.

    The library's :func:`kalman_filter` takes one constant transition, so the
    irregular-grid SSM is conditioned with the (independent) dense LGSSM
    oracle; what is under test is ``matern32_discretize`` at every spacing.
    """
    rng = np.random.default_rng(seed)
    dts = lengthscale * np.exp(rng.uniform(-3.0, 1.0, size=n_time - 1))
    times = np.concatenate([[0.0], np.cumsum(dts)])
    noise_var = noise_ratio * variance
    y = rng.standard_normal(n_time) * np.sqrt(variance + noise_var)
    _F, _L, _Qc, _H, p_inf = matern32_continuous(variance, lengthscale)
    a_seq, q_seq = _irregular_ssm_matrices(times, variance, lengthscale)
    post = lgssm_dense_posterior(
        np.zeros(2),
        np.asarray(p_inf),
        y[:, None],
        a_seq,
        q_seq,
        np.array([[1.0, 0.0]]),
        np.array([[noise_var]]),
        prior_on_first_state=True,
    )
    dense = dense_gp_regression(times, y, variance, lengthscale, noise_var)
    np.testing.assert_allclose(
        post.smoothed_mean, dense["mean"], rtol=RTOL, atol=_scale_tol(dense["mean"])
    )
    np.testing.assert_allclose(
        post.joint_cov[2:, 2:], dense["cov"], rtol=RTOL, atol=_scale_tol(dense["cov"])
    )
    np.testing.assert_allclose(post.log_likelihood, dense["log_ml"], rtol=RTOL)


# ---------------------------------------------------------------------------
# 3. Poisson likelihood: iterated Laplace == dense GP-Laplace
# ---------------------------------------------------------------------------


@pytest.mark.slow
@settings(max_examples=60, deadline=None)
@given(
    variance=st.floats(0.1, 3.0),
    lengthscale=st.floats(0.05, 2.0),
    dt=st.floats(0.01, 0.2),
    log_rate=st.floats(np.log(2.0), np.log(60.0)),
    n_time=_N_TIME,
    seed=st.integers(0, 2**31 - 1),
)
def test_laplace_rate_gp_equals_dense_gp_laplace(
    variance, lengthscale, dt, log_rate, n_time, seed
):
    """Mode, Laplace variance and Laplace evidence match the dense GP-Laplace."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_time) * dt
    gram = matern32_gram(times, variance, lengthscale)
    g_true = np.linalg.cholesky(gram + 1e-10 * np.eye(n_time)) @ rng.standard_normal(
        n_time
    )
    counts = rng.poisson(np.exp(log_rate + g_true) * dt).astype(float)

    dense = dense_laplace_lgcp(counts, dt, variance, lengthscale, mean=log_rate)
    result = _laplace(counts, dt, variance, lengthscale, log_rate)
    assert float(result.max_abs_update) < 1e-10  # guard: at the mode
    np.testing.assert_allclose(
        np.asarray(result.log_rate_mean) - log_rate,
        dense["mode"],
        rtol=RTOL,
        atol=RTOL * max(1.0, np.max(np.abs(dense["mode"]))),
    )
    np.testing.assert_allclose(np.asarray(result.log_rate_var), dense["var"], rtol=RTOL)
    np.testing.assert_allclose(
        float(result.log_marginal_likelihood), dense["evidence"], rtol=RTOL
    )


def test_laplace_converges_when_prior_mean_is_far_below_data():
    """Regression: a baseline far below the data no longer derails Newton.

    Counts ~25 per bin against an expected 0.34 at ``g = 0`` made the undamped
    first Newton step land at ``g ~ 40``, after which it crawled back ~1 nat per
    iteration: at the default ``n_iter=25`` the mode was ~16 (true 4.1) and the
    evidence -4.6e6 (true -17.3). The line-searched iteration matches the
    dense oracle.
    """
    counts = np.array([26.0, 12.0, 24.0])
    dt, variance, lengthscale, mean = 0.125, 2.0, 2.0, 1.0
    dense = dense_laplace_lgcp(counts, dt, variance, lengthscale, mean=mean)
    result = infer_log_rate(counts, dt, variance, lengthscale, mean=mean)  # n_iter=25
    assert float(result.max_abs_update) < 1e-10
    np.testing.assert_allclose(
        np.asarray(result.log_rate_mean) - mean, dense["mode"], rtol=RTOL
    )
    np.testing.assert_allclose(
        float(result.log_marginal_likelihood), dense["evidence"], rtol=RTOL
    )
    assert np.all(dense["mode"] > 3.5)  # guard: the mode is far from g = 0


def test_jitted_core_equals_public_infer_log_rate():
    """The jitted traced core used by the sweeps equals the public entry point."""
    counts = np.array([0.0, 2.0, 1.0, 5.0, 0.0, 3.0, 1.0, 0.0, 4.0])
    public = infer_log_rate(counts, 0.05, 0.9, 0.2, mean=np.log(25.0), n_iter=60)
    core = _laplace(counts, 0.05, 0.9, 0.2, np.log(25.0))
    for a, b in zip(public, core):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=1e-13)
    # guard: this non-trivial case also agrees with the dense oracle
    dense = dense_laplace_lgcp(counts, 0.05, 0.9, 0.2, mean=np.log(25.0))
    np.testing.assert_allclose(
        float(public.log_marginal_likelihood), dense["evidence"], rtol=RTOL
    )


def test_laplace_posterior_is_exact_gp_regression_on_its_gaussian_sites():
    """Gaussian-likelihood limit of the Laplace step, exactly.

    At the mode the Laplace posterior *is* exact Gaussian-likelihood GP
    regression on the IRLS pseudo-data ``y~`` with heteroscedastic noise
    ``1/W``: its mean is the mode (a fixed point) and its covariance is the
    Laplace covariance. The reference is the dense GP-regression oracle.
    """
    rng = np.random.default_rng(3)
    n_time, dt, variance, lengthscale, mean = 30, 0.05, 0.8, 0.3, np.log(20.0)
    counts = rng.poisson(20.0 * dt, size=n_time).astype(float)
    result = infer_log_rate(counts, dt, variance, lengthscale, mean=mean, n_iter=60)
    g_mode = np.asarray(result.log_rate_mean) - mean
    working, site_var, _ = poisson_log_rate_site(g_mode, counts, mean + np.log(dt))
    dense = dense_gp_regression(
        np.arange(n_time) * dt,
        np.asarray(working),
        variance,
        lengthscale,
        np.asarray(site_var),
    )
    np.testing.assert_allclose(dense["mean"][:, 0], g_mode, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        np.diag(dense["cov"])[0::2], np.asarray(result.log_rate_var), rtol=1e-9
    )
    # Guard: the pseudo-data differ from the mode (non-trivial fixed point).
    assert np.max(np.abs(np.asarray(working) - g_mode)) > 0.1


# ---------------------------------------------------------------------------
# 4. Laplace approximation gap vs exact quadrature (1-D GP)
# ---------------------------------------------------------------------------

# (count, expected count exp(mean) * dt, prior variance) -> pinned upper bounds
# on |log Z_laplace - log Z_exact|, |mode - E[g|y]|, |var_L / Var[g|y] - 1|.
# Observed gaps (2026-09) are listed; bounds keep ~1.5-2x headroom.
_GAP_CASES = [
    # observed:           dlogZ    dmean    dvar_rel
    ((0.0, 0.5, 1.0), (0.019, 0.14, 0.060)),  # 0.0125  0.0950  0.0401
    ((1.0, 0.5, 1.0), (0.0135, 0.17, 0.030)),  # 0.0088  0.1153  0.0197
    ((3.0, 1.0, 2.0), (0.019, 0.21, 0.120)),  # 0.0126  0.1375  0.0804
    ((25.0, 20.0, 1.0), (0.0042, 0.028, 0.026)),  # 0.0028  0.0187  0.0170
]


@pytest.mark.parametrize(("case", "bounds"), _GAP_CASES)
def test_one_bin_laplace_gap_against_quadrature(case, bounds):
    """Pinned (nonzero) Laplace gap for a single-bin rate GP."""
    count, expected, prior_var = case
    dt = 0.1
    mean = np.log(expected / dt)
    result = _laplace([count], dt, prior_var, 1.0, mean)
    exact = quadrature_poisson_normal(count, np.log(expected), prior_var)
    g_mode = float(result.log_rate_mean[0]) - mean
    # The Laplace mode is the exact posterior mode (quadrature bisection).
    np.testing.assert_allclose(g_mode, exact["mode"], atol=1e-10)
    gaps = (
        abs(float(result.log_marginal_likelihood) - exact["log_z"]),
        abs(g_mode - exact["mean"]),
        abs(float(result.log_rate_var[0]) / exact["var"] - 1.0),
    )
    for gap, bound, name in zip(gaps, bounds, ("log Z", "mean", "var")):
        assert gap < bound, f"{name} gap {gap:.4g} exceeds pinned {bound}"
        # Every pin must be able to fail: the approximation is not exact here.
        assert gap > 0.1 * bound, f"{name} gap {gap:.4g} unexpectedly tiny"


def test_laplace_gap_vanishes_in_gaussian_limit():
    """The evidence gap shrinks ~1/count as the likelihood becomes Gaussian."""
    dt, prior_var = 0.1, 1.0
    gaps = []
    for count in (40.0, 400.0, 4000.0, 40000.0):
        mean = np.log(count / dt)
        result = _laplace([count], dt, prior_var, 1.0, mean)
        exact = quadrature_poisson_normal(count, np.log(count), prior_var)
        gaps.append(abs(float(result.log_marginal_likelihood) - exact["log_z"]))
    gaps = np.array(gaps)
    assert np.all(gaps > 0)
    assert np.all(np.diff(gaps) < 0)
    # a decade more counts buys ~a decade less error (O(1/count) gap);
    # observed slopes -0.96, -1.00, -1.00
    slopes = np.diff(np.log10(gaps))
    assert np.all((slopes < -0.85) & (slopes > -1.15)), slopes
    assert gaps[-1] < 1e-5


# ---------------------------------------------------------------------------
# 5. Posterior summaries of TemporalRateGP vs scipy.stats.lognorm
# ---------------------------------------------------------------------------


@settings(max_examples=15, deadline=None)
@given(
    mu=st.floats(-2.0, 4.0),
    var=st.floats(1e-4, 2.0),
    level=st.floats(0.05, 0.999),
)
def test_rate_summaries_match_lognormal(mu, var, level):
    """``predict_rate`` is the log-normal mean; ``credible_interval`` its quantiles."""
    model = TemporalRateGP(dt=0.1)
    model.log_rate_mean_ = jnp.array([mu, mu + 0.5])
    model.log_rate_var_ = jnp.array([var, 2.0 * var])
    dist = stats.lognorm(s=np.sqrt([var, 2.0 * var]), scale=np.exp([mu, mu + 0.5]))
    np.testing.assert_allclose(
        np.asarray(model.predict_rate()), dist.mean(), rtol=1e-12
    )
    lower, upper = model.credible_interval(level)
    np.testing.assert_allclose(
        np.asarray(lower), dist.ppf(0.5 * (1 - level)), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(upper), dist.ppf(0.5 * (1 + level)), rtol=1e-10
    )


# ---------------------------------------------------------------------------
# 6. Hyperparameter learning stops at a stationary point of the dense evidence
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_sgd_reaches_stationary_point_of_dense_evidence():
    """``fit_sgd`` converges to a zero of the dense-Laplace-evidence gradient.

    The gradient is taken by central finite differences of the *dense* oracle in
    the optimiser's own coordinates (log variance, log lengthscale, mean), and is
    required to be < 1e-5 of its size at the initial hyperparameters; the dense
    evidence must also be a local maximum along each coordinate.
    """
    rng = np.random.default_rng(7)
    n_time, dt = 40, 0.05
    times = np.arange(n_time) * dt
    g_true = np.linalg.cholesky(
        matern32_gram(times, 0.6, 0.3) + 1e-10 * np.eye(n_time)
    ) @ rng.standard_normal(n_time)
    counts = rng.poisson(np.exp(np.log(30.0) + g_true) * dt).astype(float)

    def dense_evidence(theta):
        return dense_laplace_lgcp(
            counts, dt, np.exp(theta[0]), np.exp(theta[1]), mean=theta[2]
        )["evidence"]

    def fd_grad(theta, eps=1e-5):
        grad = np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            grad[i] = (dense_evidence(theta + e) - dense_evidence(theta - e)) / (
                2 * eps
            )
        return grad

    theta_init = np.array([np.log(1.0), np.log(1.0), np.log(10.0)])
    model = TemporalRateGP(
        dt=dt, variance=1.0, lengthscale=1.0, mean=float(theta_init[2]), n_iter=12
    )
    schedule = optax.exponential_decay(0.1, transition_steps=60, decay_rate=0.5)
    model.fit_sgd(counts, num_steps=400, optimizer=optax.adam(schedule))
    theta_hat = np.array(
        [np.log(model.variance_), np.log(model.lengthscale_), model.mean_]
    )

    grad_init = fd_grad(theta_init)
    grad_hat = fd_grad(theta_hat)
    # observed: |grad_init| = 1.32, |grad_hat| = 4e-9
    assert np.linalg.norm(grad_init) > 0.5  # guard: started far from optimum
    assert np.linalg.norm(grad_hat) < 1e-5 * np.linalg.norm(grad_init), grad_hat
    # the library's stored evidence is the dense evidence at the optimum
    np.testing.assert_allclose(
        model.log_marginal_likelihood_, dense_evidence(theta_hat), rtol=1e-9
    )
    # local maximum along each coordinate
    for i in range(3):
        e = np.zeros(3)
        e[i] = 0.05
        assert dense_evidence(theta_hat + e) < dense_evidence(theta_hat)
        assert dense_evidence(theta_hat - e) < dense_evidence(theta_hat)
