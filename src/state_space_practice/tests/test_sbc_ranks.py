# ruff: noqa: E402
"""Rank-based simulation-based calibration (Talts et al. 2018).

For each replicate we draw the latent trajectory and the data from the
model's own generative process, run the smoother at the true parameters,
draw ``L = 19`` samples from the reported posterior marginal of one latent
coordinate at one time step and record the rank of the true value among
them (0..19). For a calibrated posterior the rank is uniform on its 20
values. One (time, coordinate) pair per replicate -- chosen by the
replicate index, independent of the data -- keeps the ranks independent
across replicates, so the rank histogram is multinomial and Pearson's
chi-square test with 19 degrees of freedom is exact in the large-sample
sense.

Significance level: ``alpha = 1e-3`` per test (critical value 43.8). With
fixed seeds the tests are deterministic; each "must pass" test also runs the
same check on a deliberately misspecified posterior and asserts that it
*fails* there, so the check has power at the replicate count used.

* Kalman smoother: exact posterior -- must pass.
* Switching Kalman smoother (GPB1) with identical per-state parameters,
  where the GPB collapse is exact -- must pass.
* Laplace-EKF point-process smoother at moderate rates -- passes (the
  z-score calibration of the same regime is in test_calibration_point_process).
* Multinomial choice (Laplace-EKF + RTS) smoother: calibrated at low
  inverse temperature, over-confident at high inverse temperature (the
  skewed one-step posterior of a saturating softmax; see the pinned
  z-variance / coverage numbers in
  ``test_calibration_behavioural.test_multinomial_smoother_calibration``:
  z variance ~3.0 and 90% coverage ~0.69 at beta = 5). The high-beta test
  pins the *failure*: the rank test must detect the miscalibration, with
  the U-shaped histogram of an over-confident posterior.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import softmax
from scipy.stats import chi2

from state_space_practice.kalman import kalman_smoother, rts_backward_scan
from state_space_practice.multinomial_choice import _multinomial_choice_filter_jit
from state_space_practice.point_process_kalman import stochastic_point_process_smoother
from state_space_practice.switching_kalman import (
    switching_kalman_filter,
    switching_kalman_smoother,
)
from state_space_practice.tests.oracles import random_spd_matrix, random_stable_matrix

N_DRAWS = 19  # posterior draws per replicate -> ranks 0..19
N_BINS = N_DRAWS + 1
ALPHA = 1e-3
CRITICAL = float(chi2.ppf(1.0 - ALPHA, N_BINS - 1))


def _ranks(rng, x_true, mean, var) -> np.ndarray:
    """Rank of each true value among N_DRAWS draws from N(mean, var)."""
    draws = mean[:, None] + np.sqrt(var)[:, None] * rng.normal(
        size=(x_true.shape[0], N_DRAWS)
    )
    return np.sum(draws < x_true[:, None], axis=1)


def _chi_square(ranks: np.ndarray) -> tuple[float, np.ndarray]:
    counts = np.bincount(ranks, minlength=N_BINS)
    expected = ranks.size / N_BINS
    return float(np.sum((counts - expected) ** 2 / expected)), counts


def _select(x, mean, var):
    """One (time, coordinate) per replicate: t = r mod T, d = (r // T) mod n.

    x, mean, var: (n_rep, T, n). Returns three (n_rep,) arrays."""
    n_rep, T, n = x.shape
    r = np.arange(n_rep)
    t, d = r % T, (r // T) % n
    return x[r, t, d], mean[r, t, d], var[r, t, d]


def _describe(stat, counts) -> str:
    return f"chi2 = {stat:.1f} (critical {CRITICAL:.1f}); counts {counts.tolist()}"


# ---------------------------------------------------------------------------
# Kalman smoother
# ---------------------------------------------------------------------------


_smooth_batch = jax.jit(
    jax.vmap(
        lambda y, m0, P0, A, Q, H, R: kalman_smoother(m0, P0, y, A, Q, H, R, False)[:2],
        in_axes=(0, None, None, None, None, None, None),
    )
)


def _kalman_replicates(seed: int, n_sets=4, n_rep=1000, T=8):
    rng = np.random.default_rng(seed)
    xs, means, vars_ = [], [], []
    for _ in range(n_sets):
        n, m = 2, 2
        m0, P0 = rng.normal(size=n), random_spd_matrix(rng, n)
        A, Q = random_stable_matrix(rng, n), random_spd_matrix(rng, n, scale=0.3)
        H, R = rng.normal(size=(m, n)), random_spd_matrix(rng, m, scale=0.5)
        x = m0 + rng.normal(size=(n_rep, n)) @ np.linalg.cholesky(P0).T
        xt, yt = [], []
        for _ in range(T):
            x = x @ A.T + rng.normal(size=(n_rep, n)) @ np.linalg.cholesky(Q).T
            xt.append(x)
            yt.append(x @ H.T + rng.normal(size=(n_rep, m)) @ np.linalg.cholesky(R).T)
        sm, sc = _smooth_batch(jnp.asarray(np.stack(yt, 1)), m0, P0, A, Q, H, R)
        xs.append(np.stack(xt, 1))
        means.append(np.asarray(sm))
        vars_.append(np.einsum("rtii->rti", np.asarray(sc)))
    return rng, np.concatenate(xs), np.concatenate(means), np.concatenate(vars_)


def test_kalman_smoother_ranks_are_uniform() -> None:
    """4 random models x 1000 replicates (4000 independent ranks)."""
    rng, x, mean, var = _kalman_replicates(0)
    xt, mt, vt = _select(x, mean, var)
    stat, counts = _chi_square(_ranks(rng, xt, mt, vt))
    assert stat < CRITICAL, _describe(stat, counts)  # observed 16.0
    # power guard: a posterior 25% too narrow in variance is rejected
    stat_bad, counts_bad = _chi_square(_ranks(rng, xt, mt, 0.75 * vt))
    assert stat_bad > CRITICAL, _describe(stat_bad, counts_bad)


# ---------------------------------------------------------------------------
# Switching Kalman smoother in its exact regime
# ---------------------------------------------------------------------------


@pytest.mark.slow  # switching filter + GPB1 smoother over 2000 replicates
def test_switching_kalman_smoother_ranks_are_uniform_in_exact_regime() -> None:
    """Identical per-state continuous parameters and a nontrivial Markov
    chain (the switching filter's convention: prior on x_1): the GPB1
    smoother's marginal is the exact posterior."""
    rng = np.random.default_rng(1)
    n, m, K, T, n_rep = 2, 2, 2, 6, 2000
    m0, P0 = rng.normal(size=n), random_spd_matrix(rng, n)
    A, Q = random_stable_matrix(rng, n), random_spd_matrix(rng, n, scale=0.3)
    H, R = rng.normal(size=(m, n)), random_spd_matrix(rng, m, scale=0.5)
    Zd = np.array([[0.8, 0.2], [0.3, 0.7]])
    pi = np.array([0.35, 0.65])
    x = m0 + rng.normal(size=(n_rep, n)) @ np.linalg.cholesky(P0).T
    xt, yt = [], []
    for t in range(T):
        if t > 0:
            x = x @ A.T + rng.normal(size=(n_rep, n)) @ np.linalg.cholesky(Q).T
        xt.append(x)
        yt.append(x @ H.T + rng.normal(size=(n_rep, m)) @ np.linalg.cholesky(R).T)
    x, y = np.stack(xt, 1), np.stack(yt, 1)

    def st(a):
        return jnp.asarray(np.stack([a] * K, -1))

    params = (st(m0), st(P0), jnp.asarray(pi))
    dyn = (jnp.asarray(Zd), st(A), st(Q), st(H), st(R))

    def smooth(obs):
        fm, fc, fp, *_ = switching_kalman_filter(*params, obs, *dyn)
        g = switching_kalman_smoother(fm, fc, fp, dyn[2], dyn[1], dyn[0])
        return g[0], g[1]

    # lax.map, not vmap: the filter's debug-print guards would fire per
    # element under vmap (see test_calibration_kalman).
    sm, sc = jax.jit(lambda ys: jax.lax.map(smooth, ys))(jnp.asarray(y))
    mean, var = np.asarray(sm), np.einsum("rtii->rti", np.asarray(sc))
    xt, mt, vt = _select(x, mean, var)
    stat, counts = _chi_square(_ranks(rng, xt, mt, vt))
    assert stat < CRITICAL, _describe(stat, counts)  # observed 14.0
    stat_bad, counts_bad = _chi_square(_ranks(rng, xt, mt + 0.3 * np.sqrt(vt), vt))
    assert stat_bad > CRITICAL, _describe(stat_bad, counts_bad)


# ---------------------------------------------------------------------------
# Laplace-EKF point-process smoother, moderate rates
# ---------------------------------------------------------------------------


def _affine_log_rate(design_t, x):
    return design_t[:, 0] + design_t[:, 1:] @ x


@pytest.mark.slow  # vmapped Laplace smoother over 2000 random models
def test_point_process_smoother_ranks_are_uniform_at_moderate_rates() -> None:
    """2000 random 2-latent / 3-neuron AR(1) models, rates 20-60 Hz at
    dt = 20 ms (0.4-1.2 expected spikes / bin / neuron), T = 20."""
    rng = np.random.default_rng(2)
    n_rep, T, d, n_neurons, dt = 2000, 20, 2, 3, 0.02
    a = rng.uniform(0.85, 0.98, (n_rep, d))
    q = rng.uniform(0.01, 0.05, (n_rep, d))
    m0 = rng.normal(0.0, 0.3, (n_rep, d))
    v0 = rng.uniform(0.1, 0.3, (n_rep, d))
    W = rng.normal(0.0, 0.7, (n_rep, n_neurons, d))
    b = np.log(rng.uniform(20.0, 60.0, (n_rep, n_neurons)))
    x = m0 + np.sqrt(v0) * rng.normal(size=(n_rep, d))
    xs, ys = [], []
    for _ in range(T):
        x = a * x + np.sqrt(q) * rng.normal(size=(n_rep, d))
        xs.append(x)
        ys.append(rng.poisson(np.exp(b + np.einsum("rnd,rd->rn", W, x)) * dt))
    x, y = np.stack(xs, 1), np.stack(ys, 1).astype(float)
    design = np.broadcast_to(
        np.concatenate([b[..., None], W], -1)[:, None], (n_rep, T, n_neurons, 1 + d)
    )

    def smooth(m0, v0, a, q, design, y):
        sm, sc, _, _ = stochastic_point_process_smoother(
            m0, jnp.diag(v0), design, y, dt, jnp.diag(a), jnp.diag(q), _affine_log_rate
        )
        return sm, jnp.diagonal(sc, axis1=-2, axis2=-1)

    sm, sv = jax.jit(jax.vmap(smooth))(
        *(jnp.asarray(v) for v in (m0, v0, a, q, np.ascontiguousarray(design), y))
    )
    assert y.mean() > 0.4  # guard: informative regime
    xt, mt, vt = _select(x, np.asarray(sm), np.asarray(sv))
    stat, counts = _chi_square(_ranks(rng, xt, mt, vt))
    # observed chi2 29.3 (p ~ 0.06)
    assert stat < CRITICAL, _describe(stat, counts)
    stat_bad, counts_bad = _chi_square(_ranks(rng, xt, mt, 0.75 * vt))
    assert stat_bad > CRITICAL, _describe(stat_bad, counts_bad)


# ---------------------------------------------------------------------------
# Multinomial choice smoother
# ---------------------------------------------------------------------------


def _multinomial_ranks(beta: float, seed: int, n_rep=2000, T=30, K=3, q=0.05):
    rng = np.random.default_rng(seed)
    k = K - 1
    x0 = rng.normal(size=(n_rep, k))  # x_0 ~ N(0, I), the model's prior
    x = x0[:, None, :] + np.cumsum(rng.normal(0, np.sqrt(q), (n_rep, T, k)), axis=1)
    logits = beta * np.concatenate([np.zeros((n_rep, T, 1)), x], axis=-1)
    cdf = np.cumsum(softmax(logits, axis=-1), axis=-1)
    choices = (rng.random((n_rep, T, 1)) > cdf).sum(-1)

    def smooth(c):
        f = _multinomial_choice_filter_jit(c, K, q, beta, jnp.zeros(k), jnp.eye(k))
        m, P, _ = rts_backward_scan(
            f.filtered_values, f.filtered_covariances, jnp.eye(k), q * jnp.eye(k)
        )
        return m, jnp.diagonal(P, axis1=-2, axis2=-1)

    m, v = jax.jit(jax.vmap(smooth))(jnp.asarray(choices))
    xt, mt, vt = _select(x, np.asarray(m), np.asarray(v))
    return rng, xt, mt, vt


@pytest.mark.slow  # vmapped Laplace filter compile (~6 s) + 2000 replicates
def test_multinomial_smoother_ranks_are_uniform_at_low_beta() -> None:
    rng, xt, mt, vt = _multinomial_ranks(beta=0.5, seed=3)
    stat, counts = _chi_square(_ranks(rng, xt, mt, vt))
    assert stat < CRITICAL, _describe(stat, counts)  # observed 18.0
    stat_bad, counts_bad = _chi_square(_ranks(rng, xt, mt, 0.75 * vt))
    assert stat_bad > CRITICAL, _describe(stat_bad, counts_bad)


@pytest.mark.slow
def test_multinomial_smoother_miscalibration_is_detected_at_high_beta() -> None:
    """Pinned failure: at beta = 5 the Laplace posterior is over-confident
    (see the module docstring and test_calibration_behavioural), so the rank
    test rejects uniformity by a wide margin and the histogram is U-shaped
    (the truth falls outside the posterior draws too often)."""
    rng, xt, mt, vt = _multinomial_ranks(beta=5.0, seed=4)
    stat, counts = _chi_square(_ranks(rng, xt, mt, vt))
    assert stat > 5 * CRITICAL, _describe(stat, counts)
    extreme = (counts[0] + counts[-1]) / counts.sum()
    # observed: chi2 382, 22% of ranks in the two extreme bins (10% uniform)
    assert extreme > 2.0 * (2 / N_BINS), _describe(stat, counts)
    # ... and the central bins are depleted (observed 16% vs 20% uniform)
    central = counts[8:12].sum() / counts.sum()
    assert central < 0.9 * (4 / N_BINS), _describe(stat, counts)
