# ruff: noqa: E402
"""Approximation gaps move in the direction theory predicts.

Each approximate method in the library (Laplace-EKF updates, GPB1 / GPB2
switching collapses) is compared with an exact reference (quadrature or
path enumeration, from the existing oracle modules) along a one-parameter
family of problems along which theory says the approximation becomes
exact. The tests assert a *monotone* decrease of the gap over 4-5 settings
and a clearly nonzero gap at the worst setting, so none can pass vacuously
(an oracle that re-computed the approximation would fail the nonzero
guard; an approximation with an error floor -- a missing normaliser
constant, a wrong collapse weight -- would fail the monotone decrease):

* Laplace point-process filter / smoother / evidence vs quadrature as the
  expected counts per bin grow (the posterior tends to a Gaussian);
* GPB1 / GPB2 vs exact path enumeration as the discrete states persist
  (the collapse mixes fewer switching paths) and as the per-state
  parameters become more similar (the collapsed components coincide);
* softmax (multinomial choice) Laplace vs quadrature as the inverse
  temperature falls, and -- equivalently, by the scale symmetry checked
  here -- as the prior tightens;
* the Smith (binomial) Laplace evidence as the number of trials per bin
  grows.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.multinomial_choice import (
    multinomial_choice_filter,
    multinomial_choice_smoother,
)
from state_space_practice.smith_learning_algorithm import (
    smith_laplace_log_likelihood,
    smith_learning_filter,
)
from state_space_practice.tests.test_oracle_choice import (
    _GRID_1D,
    _exact_smith,
    _grid_forward_backward_1d,
    _moments_1d,
    _softmax_choice_log_liks_1d,
)
from state_space_practice.tests.test_oracle_point_process import (
    _METRICS,
    _grid_posterior,
    _informative_problem,
    _run_laplace,
    _standardized_errors,
)
from state_space_practice.tests.test_oracle_switching_kalman import (
    _oracle,
    _random_switching_model,
    _run_library,
)


def _assert_decreasing(values, name: str, msg: str) -> None:
    values = np.asarray(values)
    assert np.all(np.diff(values) < 0), f"{name} not decreasing: {values}; {msg}"


# ---------------------------------------------------------------------------
# Laplace point-process filter vs quadrature
# ---------------------------------------------------------------------------


@pytest.mark.slow  # 5 quadrature posteriors + Laplace compile (~6 s)
def test_point_process_laplace_gap_shrinks_with_counts_per_bin() -> None:
    """Extends the three-scale slope check of test_oracle_point_process to a
    monotone trend over five scales (expected counts x1 .. x100 along the
    same latent path) for the filter, the smoother and the evidence.

    Observed (seed 0): smoothed mean error 6.7e-3 -> 6.6e-4 sd, smoothed
    variance 4.5e-3 -> 6.4e-5 (relative), log-evidence 3.8e-3 -> 5.5e-5
    nats. (Seed 1 is monotone until the log-evidence error reaches the
    ~1e-7 quadrature floor.)"""
    scales = [1e3, 3e3, 1e4, 3e4, 1e5]
    errs = []
    for scale in scales:
        problem = _informative_problem(1, scale, seed=0)
        laplace = _run_laplace(problem, max_newton_iter=30)
        sd = np.sqrt(laplace["filt_cov"][:, 0, 0])
        lo = (laplace["filt_mean"][:, 0] - 12 * sd).min()
        hi = (laplace["filt_mean"][:, 0] + 12 * sd).max()
        exact = _grid_posterior(problem, [np.linspace(lo, hi, 3001)])
        errs.append(_standardized_errors(laplace, exact))
    msg = "; ".join(f"{m}: " + str([f"{e[m]:.2e}" for e in errs]) for m in _METRICS)
    for metric in _METRICS:
        values = [e[metric] for e in errs]
        _assert_decreasing(values, metric, msg)
        assert values[0] > 5 * values[-1], msg
    # nonzero gap at the least informative scale
    assert errs[0]["smooth_mean"] > 1e-3 and errs[0]["log_lik"] > 1e-3, msg


# ---------------------------------------------------------------------------
# GPB1 / GPB2 vs exact enumeration
# ---------------------------------------------------------------------------

_GPB_METRICS = (
    "filter_ll",
    "filter_prob",
    "gpb1_prob",
    "gpb2_prob",
    "gpb1_mean",
    "gpb2_mean",
)


def _gpb_errors(model: dict) -> dict:
    exact = _oracle(model)
    out = _run_library(model)
    sd = np.sqrt(np.einsum("tii->ti", exact.smoothed_cov))

    def rms(a):
        return float(np.sqrt(np.mean(np.square(a))))

    return {
        "filter_ll": abs(out["filter"]["log_likelihood"] - exact.log_likelihood),
        "filter_prob": rms(out["filter"]["prob"] - exact.filtered_discrete_prob),
        "gpb1_prob": rms(out["gpb1"]["prob"] - exact.smoothed_discrete_prob),
        "gpb2_prob": rms(out["gpb2"]["prob"] - exact.smoothed_discrete_prob),
        "gpb1_mean": rms((out["gpb1"]["mean"] - exact.smoothed_mean) / sd),
        "gpb2_mean": rms((out["gpb2"]["mean"] - exact.smoothed_mean) / sd),
    }


def _coupled_simulation(model: dict, rho: float, seed: int, n_time: int = 5) -> dict:
    """Two-state model with symmetric switching probability ``rho``, data
    simulated with common random numbers across ``rho`` (and across the
    model's continuous parameters): the switch times are {t : u_t < rho},
    so lowering rho removes switches rather than drawing new data."""
    rng = np.random.default_rng(seed)
    n = model["continuous_transition_matrix"].shape[0]
    m = model["measurement_matrix"].shape[0]
    u = rng.uniform(size=n_time)
    zx, zy = rng.normal(size=(n_time, n)), rng.normal(size=(n_time, m))
    s = int(rng.uniform() < model["init_discrete_state_prob"][1])
    x = (
        model["init_state_cond_mean"][:, s]
        + np.linalg.cholesky(model["init_state_cond_cov"][..., s]) @ zx[0]
    )
    ys = []
    for t in range(n_time):
        if t > 0:
            s = 1 - s if u[t] < rho else s
            x = (
                model["continuous_transition_matrix"][..., s] @ x
                + np.linalg.cholesky(model["process_cov"][..., s]) @ zx[t]
            )
        ys.append(
            model["measurement_matrix"][..., s] @ x
            + np.linalg.cholesky(model["measurement_cov"][..., s]) @ zy[t]
        )
    out = dict(model)
    out["obs"] = np.stack(ys)
    out["discrete_transition_matrix"] = np.array([[1 - rho, rho], [rho, 1 - rho]])
    return out


def _blend(model: dict, eps: float) -> dict:
    """Per-state parameters pulled toward their across-state mean:
    theta_k(eps) = mean + eps (theta_k - mean) (convex, so covariances stay
    SPD); eps = 0 makes the states identical (the exact regime)."""
    out = dict(model)
    for key in (
        "continuous_transition_matrix",
        "process_cov",
        "measurement_matrix",
        "measurement_cov",
        "init_state_cond_mean",
        "init_state_cond_cov",
    ):
        v = model[key]
        mean = v.mean(-1, keepdims=True)
        out[key] = mean + eps * (v - mean)
    return out


def _gpb_table(settings, make_model, n_problems=12) -> dict:
    table = {k: np.zeros((len(settings), n_problems)) for k in _GPB_METRICS}
    for p in range(n_problems):
        base = _random_switching_model(np.random.default_rng(100 + p), 2, 1, 2, 5)
        for i, setting in enumerate(settings):
            errors = _gpb_errors(make_model(base, setting, p))
            for k in _GPB_METRICS:
                table[k][i, p] = errors[k]
    return {k: v.mean(axis=1) for k, v in table.items()}


def _describe(table: dict) -> str:
    return "; ".join(
        f"{k}: " + str([f"{v:.2e}" for v in vals]) for k, vals in table.items()
    )


@pytest.mark.slow  # 60 path-enumeration posteriors (~10 s)
def test_gpb_gap_shrinks_as_discrete_states_persist() -> None:
    """Switching probability rho = 0.3 .. 0.003 (data simulated from each
    model with common random numbers), mean over 12 random problems
    (n=2, m=1, K=2, T=5). As rho -> 0 the discrete path is constant and
    every GPB collapse is of a single component (exact at rho = 0).

    Observed means (rho = 0.3 -> 0.003): filter LL 6.0e-3 -> 3.3e-5 nats,
    filter P(S_t) 7.8e-4 -> 9.4e-6, smoothed P(S_t) GPB1 5.1e-2 -> 1.6e-3 and
    GPB2 1.5e-2 -> 6.9e-4, smoothed mean (sd) GPB1 0.11 -> 2.4e-3 and GPB2
    3.0e-2 -> 1.3e-3. Per problem the trend is monotone in 80-100% of the
    steps; the mean over problems is monotone for every metric.
    """
    rhos = [0.3, 0.1, 0.03, 0.01, 0.003]
    table = _gpb_table(rhos, lambda base, rho, p: _coupled_simulation(base, rho, p))
    msg = _describe(table)
    for k in _GPB_METRICS:
        _assert_decreasing(table[k], k, msg)
    for k in ("gpb1_prob", "gpb2_prob", "gpb1_mean", "gpb2_mean"):
        assert table[k][0] > 1e-2, msg  # nonzero gap at the worst setting
    assert table["filter_ll"][0] > 1e-3, msg
    # GPB2 is the better smoother at every setting (on average)
    assert np.all(table["gpb2_mean"] < table["gpb1_mean"]), msg
    assert np.all(table["gpb2_prob"] < table["gpb1_prob"]), msg


@pytest.mark.slow  # 48 path-enumeration posteriors (~8 s)
def test_gpb_gap_shrinks_as_states_become_similar() -> None:
    """Per-state parameters blended toward their mean, eps = 1 .. 1/64
    (rho = 0.2, common random numbers), mean over 12 problems. Observed
    (eps = 1 -> 1/64): filter LL 1.4e-3 -> 1e-6 nats, smoothed mean (sd)
    GPB1 0.10 -> 2.6e-4 and GPB2 1.8e-2 -> 6.2e-5."""
    eps = [1.0, 0.25, 0.0625, 0.015625]
    table = _gpb_table(
        eps, lambda base, e, p: _coupled_simulation(_blend(base, e), 0.2, p)
    )
    msg = _describe(table)
    for k in _GPB_METRICS:
        _assert_decreasing(table[k], k, msg)
        assert table[k][0] > 20 * table[k][-1], msg
    assert table["gpb1_mean"][0] > 1e-2 and table["filter_ll"][0] > 1e-4, msg


# ---------------------------------------------------------------------------
# Softmax Laplace vs quadrature
# ---------------------------------------------------------------------------

_CHOICES = [1, 1, 0, 1, 1, 1]


def _softmax_errors(beta: float, prior_var: float, q: float) -> dict:
    """Laplace-EKF filter / RTS smoother (K = 2) vs quadrature with prior
    x_0 ~ N(0, prior_var) and random-walk variance q."""
    v1 = prior_var + q
    prior = np.exp(-0.5 * _GRID_1D**2 / v1) / np.sqrt(2 * np.pi * v1)
    lls = _softmax_choice_log_liks_1d(_GRID_1D, _CHOICES, beta)
    filt, smooth, log_z = _grid_forward_backward_1d(_GRID_1D, prior, lls, q)
    fm, fv = _moments_1d(_GRID_1D, filt)
    sm, sv = _moments_1d(_GRID_1D, smooth)
    kw = dict(
        process_noise=q,
        inverse_temperature=beta,
        init_mean=jnp.zeros(1),
        init_cov=prior_var * jnp.eye(1),
    )
    f = multinomial_choice_filter(jnp.array(_CHOICES), 2, **kw)
    s = multinomial_choice_smoother(jnp.array(_CHOICES), 2, **kw)
    return {
        "filter_mean": float(
            np.max(np.abs(np.asarray(f.filtered_values[:, 0]) - fm) / np.sqrt(fv))
        ),
        "smoother_mean": float(
            np.max(np.abs(np.asarray(s.smoothed_values[:, 0]) - sm) / np.sqrt(sv))
        ),
        "smoother_var": float(
            np.max(np.abs(np.asarray(s.smoothed_covariances[:, 0, 0]) / sv - 1))
        ),
        "log_lik": abs(float(f.marginal_log_likelihood) - log_z),
        "laplace_ll": float(f.marginal_log_likelihood),
    }


_SOFTMAX_METRICS = ("filter_mean", "smoother_mean", "smoother_var", "log_lik")


@pytest.mark.slow  # 10 quadrature posteriors + choice filter compile (~5 s)
def test_softmax_laplace_gap_shrinks_as_beta_falls_or_prior_tightens() -> None:
    """beta = 4 .. 0.25 (prior N(0, 1), q = 0.1): the softmax likelihood
    becomes flatter over the posterior's width and the one-step posterior
    tends to a Gaussian. Observed (beta 4 -> 0.25): filter mean 0.48 ->
    1.0e-3 sd, smoother variance 0.33 -> 2.2e-3 (relative), log-evidence
    0.091 -> 6.9e-4 nats (the evidence error is not monotone between beta =
    4 and 2, where it peaks at 0.12, so the trend is asserted from beta = 2).

    Tightening the prior (init_cov = v, q = 0.1 v, v = 4 .. 1/64) at beta = 2
    is the same family: x -> x / sqrt(v) maps it to prior N(0, 1), q = 0.1 and
    inverse temperature 2 sqrt(v). That scale symmetry of the Laplace filter
    is checked exactly (identical evidence), and the trend asserted again.
    """
    betas = [4.0, 2.0, 1.0, 0.5, 0.25]
    by_beta = [_softmax_errors(b, 1.0, 0.1) for b in betas]
    msg = "; ".join(
        f"{m}: " + str([f"{e[m]:.2e}" for e in by_beta]) for m in _SOFTMAX_METRICS
    )
    for m in ("filter_mean", "smoother_mean", "smoother_var"):
        _assert_decreasing([e[m] for e in by_beta], m, msg)
    _assert_decreasing([e["log_lik"] for e in by_beta[1:]], "log_lik", msg)
    assert by_beta[0]["filter_mean"] > 0.1 and by_beta[1]["log_lik"] > 0.05, msg
    assert by_beta[-1]["filter_mean"] < 0.01 * by_beta[0]["filter_mean"], msg

    variances = [4.0, 1.0, 0.25, 0.0625, 0.015625]
    by_var = [_softmax_errors(2.0, v, 0.1 * v) for v in variances]
    for e_v, e_b in zip(by_var, by_beta):
        # 1e-7, not round-off: the update's log-determinants carry an
        # absolute 1e-9 Cholesky shift (see the xfail in test_invariances,
        # TestMultinomialChoiceInvariances::test_laplace_evidence_is_scale_equivariant)
        np.testing.assert_allclose(e_v["laplace_ll"], e_b["laplace_ll"], rtol=1e-7)
        for m in _SOFTMAX_METRICS:
            np.testing.assert_allclose(e_v[m], e_b[m], rtol=1e-4, atol=1e-8, err_msg=m)
    for m in ("filter_mean", "smoother_mean", "smoother_var"):
        _assert_decreasing([e[m] for e in by_var], m, msg)


# ---------------------------------------------------------------------------
# Smith Laplace evidence vs quadrature
# ---------------------------------------------------------------------------


@pytest.mark.slow  # 15 quadrature posteriors + Smith filter compile (~9 s)
def test_smith_laplace_evidence_bias_shrinks_with_trials_per_bin() -> None:
    """Binomial outcomes y_k = round(n p_k) along three fixed learning-state
    paths with p_k in [0.3, 0.7] (so the information per bin grows like n),
    sigma^2 = 0.3. The Laplace evidence underestimates the exact evidence
    by an amount that shrinks like ~n^-0.75 here: mean |error| 3.1e-2
    (n = 4) -> 4.9e-4 nats (n = 1024)."""
    paths = [
        np.array([-0.8, -0.4, 0.0, 0.3, 0.6, 0.8]),
        np.array([0.5, 0.2, -0.3, -0.6, -0.2, 0.4]),
        np.array([0.0, 0.3, 0.7, 0.4, 0.1, -0.2]),
    ]
    ns = [4, 16, 64, 256, 1024]
    sigma2 = 0.3
    errors = np.zeros((len(ns), len(paths)))
    for j, x in enumerate(paths):
        p = 1.0 / (1.0 + np.exp(-x))
        for i, n in enumerate(ns):
            y = np.round(n * p).astype(int)
            _, _, exact = _exact_smith(y, n, sigma2, 0.0, sigma2, 0.0)
            out = smith_learning_filter(
                jnp.asarray(y),
                init_learning_state=0.0,
                init_learning_variance=sigma2,
                sigma_epsilon=float(np.sqrt(sigma2)),
                prob_correct_by_chance=0.5,
                max_possible_correct=n,
            )
            laplace = float(
                jnp.sum(
                    smith_laplace_log_likelihood(
                        jnp.asarray(y), jnp.full(y.size, n), *out[1:], 0.0
                    )
                )
            )
            errors[i, j] = laplace - exact
    mean_abs = np.abs(errors).mean(axis=1)
    msg = f"errors (n x path) {errors.round(6).tolist()}; mean |error| {mean_abs}"
    _assert_decreasing(mean_abs, "mean |error|", msg)
    slope = np.polyfit(np.log(ns), np.log(mean_abs), 1)[0]
    assert slope < -0.5, (slope, msg)
    assert mean_abs[0] > 1e-2, msg  # nonzero gap at the worst setting
    # it is a bias: at small n every path's Laplace evidence is too low
    assert np.all(errors[0] < 0), msg
