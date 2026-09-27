# ruff: noqa: E402
"""Simulation-based calibration of the behavioural smoothers.

At the true parameters, a correct posterior is calibrated: the standardised
errors ``z = (x_true - m) / sqrt(P)`` pooled over replicates and trials are
~N(0, 1) and the central 90% interval covers the truth 90% of the time. We
simulate many small replicates *from each model's own generative prior*
(including ``x_0 ~ N(init_mean, init_cov)`` propagated once before the first
trial) and check this for the smoothed latent of the multinomial-choice,
Smith and switching-choice models.

Where the Laplace-EKF + RTS approximation is miscalibrated we pin the
observed statistics instead. The miscalibration is *not* in low-information
regimes -- with a weak likelihood the one-step posterior is nearly Gaussian
and the smoother is calibrated -- but in high-information / saturating ones
(large inverse temperature, large learning-state noise): the one-step
posterior of a saturating logistic / softmax likelihood is skewed, the
Laplace mode sits between the mean and the saturated side and its curvature
understates the spread, so the smoother is over-confident (z variance 2.6 at
beta=5 for two options). The exact quadrature posterior on the same
replicates is calibrated (z variance 0.97, coverage 0.91), which shows the
simulation and the test statistics are right and the gap is the Gaussian
approximation.

Coverage bands allow for the within-replicate correlation of z (effective
sample sizes of a few hundred).
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit, softmax

from state_space_practice.kalman import rts_backward_scan
from state_space_practice.multinomial_choice import _multinomial_choice_filter_jit
from state_space_practice.smith_learning_algorithm import (
    _smith_learning_filter_impl,
    smith_learning_smoother,
)
from state_space_practice.switching_choice import (
    _switching_choice_filter_jit,
    switching_choice_smoother,
)

_Z90 = 1.6448536269514722


def _z_summary(z):
    z = np.asarray(z).ravel()
    return z.mean(), z.var(), np.mean(np.abs(z) < _Z90)


def _fmt(stats):
    return "mean {:.3f}, var {:.3f}, 90% coverage {:.3f}".format(*stats)


# ---------------------------------------------------------------------------
# Multinomial choice
# ---------------------------------------------------------------------------


def _simulate_multinomial(n_rep, n_trials, n_options, q, beta, rng):
    x0 = rng.normal(size=(n_rep, n_options - 1))  # x_0 ~ N(0, I), the model prior
    x = x0[:, None, :] + np.cumsum(
        rng.normal(0, np.sqrt(q), (n_rep, n_trials, n_options - 1)), axis=1
    )
    logits = beta * np.concatenate([np.zeros((n_rep, n_trials, 1)), x], axis=-1)
    cdf = np.cumsum(softmax(logits, axis=-1), axis=-1)
    choices = (rng.random((n_rep, n_trials, 1)) > cdf).sum(-1)
    return x, choices


def _multinomial_smoother(n_options):
    k = n_options - 1

    def smooth(choices, q, beta):
        f = _multinomial_choice_filter_jit(
            choices, n_options, q, beta, jnp.zeros(k), jnp.eye(k)
        )
        m, P, _ = rts_backward_scan(
            f.filtered_values, f.filtered_covariances, jnp.eye(k), q * jnp.eye(k)
        )
        return m, P

    return jax.jit(jax.vmap(smooth, in_axes=(0, None, None)))


@pytest.mark.parametrize(
    "beta, bands",
    [
        # (mean, var interval, coverage interval)
        (0.5, (0.1, (0.85, 1.2), (0.86, 0.94))),  # observed -0.05, 1.01, 0.90
        (5.0, (0.15, (2.2, 3.9), (0.62, 0.76))),  # observed -0.03, 3.04, 0.69
    ],
)
def test_multinomial_smoother_calibration(beta, bands):
    """K=3: calibrated at beta=0.5, pinned over-confidence at beta=5."""
    q, n_rep, n_trials = 0.05, 300, 30
    x, choices = _simulate_multinomial(
        n_rep, n_trials, 3, q, beta, np.random.default_rng(0)
    )
    m, P = _multinomial_smoother(3)(jnp.asarray(choices), q, beta)
    z = (x - np.asarray(m)) / np.sqrt(np.einsum("rtii->rti", np.asarray(P)))
    stats = _z_summary(z)
    mean_tol, (var_lo, var_hi), (cov_lo, cov_hi) = bands
    assert abs(stats[0]) < mean_tol, _fmt(stats)
    assert var_lo < stats[1] < var_hi, _fmt(stats)
    assert cov_lo < stats[2] < cov_hi, _fmt(stats)


def test_multinomial_exact_posterior_is_calibrated_where_laplace_is_not():
    """K=2, beta=5: quadrature posterior calibrated, Laplace over-confident."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_oracle_choice import (
        _grid_forward_backward_1d,
        _moments_1d,
        _softmax_choice_log_liks_1d,
    )

    q, beta, n_rep, n_trials = 0.05, 5.0, 200, 12
    x, choices = _simulate_multinomial(
        n_rep, n_trials, 2, q, beta, np.random.default_rng(0)
    )
    m, P = _multinomial_smoother(2)(jnp.asarray(choices), q, beta)
    z_laplace = (x[..., 0] - np.asarray(m)[..., 0]) / np.sqrt(np.asarray(P)[..., 0, 0])

    grid = np.linspace(-9.0, 9.0, 451)
    prior = np.exp(-0.5 * grid**2 / (1 + q)) / np.sqrt(2 * np.pi * (1 + q))
    z_exact = []
    for r in range(n_rep):
        lls = _softmax_choice_log_liks_1d(grid, choices[r], beta)
        _, smooth, _ = _grid_forward_backward_1d(grid, prior, lls, q)
        sm, sv = _moments_1d(grid, smooth)
        z_exact.append((x[r, :, 0] - sm) / np.sqrt(sv))
    exact = _z_summary(np.array(z_exact))
    laplace = _z_summary(z_laplace)
    msg = f"exact: {_fmt(exact)}; Laplace: {_fmt(laplace)}"
    # Observed: exact var 0.97 / coverage 0.91; Laplace var 2.58 / 0.74.
    assert 0.85 < exact[1] < 1.15 and 0.87 < exact[2] < 0.95, msg
    assert 2.0 < laplace[1] < 3.3 and 0.68 < laplace[2] < 0.8, msg


# ---------------------------------------------------------------------------
# Smith learning model
# ---------------------------------------------------------------------------


@jax.jit
def _smith_smooth(y, sigma2, init_var):
    out = _smith_learning_filter_impl(
        y,
        jnp.ones_like(y),
        jnp.asarray(0.0),
        init_var,
        sigma2,
        jnp.asarray(0.0),
        differentiable=False,
    )
    sm, sv, _, _ = smith_learning_smoother(out[1], out[2], out[3], out[4])
    return sm, sv


@pytest.mark.slow  # jitted filter + smoother vmapped over 300 replicates
@pytest.mark.parametrize(
    "sigma2, bands",
    [
        (0.05, (0.12, (0.9, 1.25), (0.86, 0.93))),  # observed -0.07, 1.07, 0.89
        (1.0, (0.12, (1.35, 1.8), (0.8, 0.87))),  # observed -0.07, 1.56, 0.84
    ],
)
def test_smith_smoother_calibration(sigma2, bands):
    """Bernoulli outcomes: calibrated at small sigma, over-confident at large.

    ``x_0 ~ N(0, init_var)`` and ``x_1 = x_0 + w_1`` as in the filter.
    """
    n_rep, n_trials = 300, 40
    rng = np.random.default_rng(1)
    x = rng.normal(0, np.sqrt(sigma2), (n_rep, 1)) + np.cumsum(
        rng.normal(0, np.sqrt(sigma2), (n_rep, n_trials)), axis=1
    )
    y = (rng.random((n_rep, n_trials)) < expit(x)).astype(float)
    sm, sv = jax.vmap(_smith_smooth, in_axes=(0, None, None))(
        jnp.asarray(y), sigma2, sigma2
    )
    stats = _z_summary((x - np.asarray(sm)) / np.sqrt(np.asarray(sv)))
    mean_tol, (var_lo, var_hi), (cov_lo, cov_hi) = bands
    assert abs(stats[0]) < mean_tol, _fmt(stats)
    assert var_lo < stats[1] < var_hi, _fmt(stats)
    assert cov_lo < stats[2] < cov_hi, _fmt(stats)


# ---------------------------------------------------------------------------
# Switching choice model
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_switching_choice_smoother_calibration():
    """Continuous and discrete smoothed posteriors of the switching model.

    Two strategies (beta 0.5 / 3, q 0.02 / 0.2), two options. The continuous
    smoother inherits the Laplace over-confidence of the beta=3 state plus
    the GPB1 collapse (pinned); the discrete posterior is roughly reliable
    (per-bin empirical frequencies within 0.15 of the mean predicted
    probability except in the sparsely populated top bin, which is
    over-confident) and beats the prior's Brier score.
    """
    betas, qs = np.array([0.5, 3.0]), np.array([0.02, 0.2])
    Z = np.array([[0.9, 0.1], [0.15, 0.85]])
    pi0 = np.array([0.5, 0.5])
    n_rep, n_trials = 200, 25
    rng = np.random.default_rng(0)
    s = np.zeros((n_rep, n_trials), int)
    x = np.zeros((n_rep, n_trials))
    s[:, 0] = rng.random(n_rep) < pi0[1]  # no transition before trial 0
    x[:, 0] = rng.normal(size=n_rep) + rng.normal(size=n_rep) * np.sqrt(qs[s[:, 0]])
    for t in range(1, n_trials):
        s[:, t] = rng.random(n_rep) < Z[s[:, t - 1], 1]
        x[:, t] = x[:, t - 1] + rng.normal(size=n_rep) * np.sqrt(qs[s[:, t]])
    choices = (rng.random((n_rep, n_trials)) < expit(betas[s] * x)).astype(int)

    def run(c):
        f = _switching_choice_filter_jit(
            c,
            2,
            2,
            process_noises=jnp.asarray(qs),
            inverse_temperatures=jnp.asarray(betas),
            decays=jnp.ones(2),
            discrete_transition_matrix=jnp.asarray(Z),
            init_discrete_prob=jnp.asarray(pi0),
        )
        sm = switching_choice_smoother(
            f.filtered_values,
            f.filtered_covs,
            f.discrete_state_probs,
            jnp.asarray(qs)[None, None, :] * jnp.ones((1, 1, 2)),
            jnp.ones((1, 1, 2)),
            jnp.asarray(Z),
            jnp.zeros((c.shape[0], 1)),
        )
        return sm[0][:, 0], sm[1][:, 0, 0], sm[2][:, 1]

    # lax.map, not vmap: under vmap the smoother's debug_print_if guards
    # (lax.cond -> select) would print for every replicate.
    m, v, p1 = (np.asarray(a) for a in jax.jit(lambda c: jax.lax.map(run, c))(choices))
    stats = _z_summary((x - m) / np.sqrt(v))
    # Observed: mean 0.02, var 1.65, coverage 0.82.
    assert abs(stats[0]) < 0.12, _fmt(stats)
    assert 1.3 < stats[1] < 2.1, _fmt(stats)
    assert 0.76 < stats[2] < 0.87, _fmt(stats)

    p1, ind = p1.ravel(), (s == 1).ravel()
    edges = np.linspace(0, 1, 6)
    for lo, hi in zip(edges[:-2], edges[1:-1]):
        sel = (p1 >= lo) & (p1 < hi)
        if sel.sum() >= 200:
            assert abs(p1[sel].mean() - ind[sel].mean()) < 0.15, (lo, hi)
    brier = np.mean((p1 - ind) ** 2)
    brier_prior = np.mean((ind.mean() - ind) ** 2)
    assert brier < brier_prior - 0.01, (brier, brier_prior)
