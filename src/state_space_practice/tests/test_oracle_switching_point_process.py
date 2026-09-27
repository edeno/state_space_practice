# ruff: noqa: E402
"""Exact oracle for the switching point-process (spike-oscillator) E-step.

For ``T <= 5`` time bins and two discrete states the ``2**T`` discrete paths
can be enumerated.  Conditional on a path the model is a (non-switching)
linear-Gaussian state-space model with Poisson observations, so

* ``log p(y, s)`` per path is computed two ways:

  - **per-path Laplace** -- the library's own single-regime Laplace-EKF update
    (``point_process_kalman_update``) run along the path with no mixture
    collapse.  This isolates the *switching* bookkeeping (pair likelihoods,
    discrete updates, GPB collapse) from the Laplace approximation;
  - **quadrature** -- an independent grid forward recursion on a 1-d latent
    state (trapezoid rule on a fine grid; exact to ~1e-10 for these smooth,
    rapidly decaying integrands), which is the true model answer;

* normalising over paths gives the exact marginal likelihood and the exact
  filtered, smoothed and two-slice discrete posteriors.

What the model must reproduce, and the documented approximations:

1. Identical per-state parameters: the observations carry no information about
   the discrete path, so every discrete posterior equals the Markov-chain prior
   (to round-off) and the model log-likelihood equals the single-regime
   Laplace log-likelihood.
2. ``T = 2``: the GPB filter has not collapsed anything yet, so the filter
   log-likelihood and filtered posterior -- and the GPB2 smoother, which keeps
   the pair-conditional filter probabilities -- equal the per-path Laplace
   enumeration exactly.
3. The GPB1 smoother's discrete pass is Kim's backward recursion on the
   filtered probabilities, ``P(S_t=i | S_{t+1}=j, y_{1:T}) ~ P(S_t=i | S_{t+1}=j,
   y_{1:t})``.  It is reproduced to round-off; it is exact when the continuous
   state carries no memory (``A = 0``), which is checked at ``T = 5``.
4. With memory and distinct parameters the GPB mixture collapse and the
   Laplace approximation each leave a nonzero gap; those gaps are pinned with
   headroom (observed values are quoted at each assertion).
"""

import jax

jax.config.update("jax_enable_x64", True)

import itertools
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import gammaln, logsumexp

from state_space_practice.switching_kalman import (
    switching_kalman_smoother,
    switching_kalman_smoother_gpb2,
)
from state_space_practice.switching_point_process import (
    SpikeObsParams,
    SwitchingSpikeOscillatorModel,
    _linear_log_intensity,
    point_process_kalman_update,
    switching_point_process_filter,
)

N_STATES = 2
DT = 0.1
# Two-state chain shared by every case.
DISCRETE_TRANSITION = jnp.array([[0.8, 0.2], [0.3, 0.7]])
INIT_DISCRETE_PROB = jnp.array([0.6, 0.4])
QUAD_GRID = np.linspace(-10.0, 10.0, 2001)


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


def _params_1d(identical: bool = False, memoryless: bool = False) -> dict:
    """Per-state parameters for a 1-d latent state and 3 neurons."""
    A = jnp.array([[[0.9, 0.4]]])
    Q = jnp.array([[[0.3, 0.8]]])
    m0 = jnp.array([[0.4, -0.6]])
    P0 = jnp.array([[[0.8, 0.5]]])
    baseline = jnp.array([[1.0, 2.0], [1.5, 0.5], [0.8, 1.2]])
    weights = jnp.array([[[1.0, -0.8]], [[-0.7, 1.2]], [[0.5, 0.3]]])
    if identical:
        A, Q, m0, P0 = (jnp.repeat(p[..., :1], 2, axis=-1) for p in (A, Q, m0, P0))
        baseline = jnp.repeat(baseline[:, :1], 2, axis=-1)
        weights = jnp.repeat(weights[..., :1], 2, axis=-1)
    if memoryless:
        A = jnp.zeros_like(A)
    return {
        "A": A,
        "Q": Q,
        "m0": m0,
        "P0": P0,
        "spike_params": SpikeObsParams(baseline=baseline, weights=weights),
        "Z": DISCRETE_TRANSITION,
        "pi0": INIT_DISCRETE_PROB,
    }


def _simulate_spikes(params: dict, n_time: int, seed: int) -> jnp.ndarray:
    """Sample (s, x, y) from the generative model (x_1 convention)."""
    rng = np.random.default_rng(seed)
    A = np.asarray(params["A"])
    Q = np.asarray(params["Q"])
    m0 = np.asarray(params["m0"])
    P0 = np.asarray(params["P0"])
    base = np.asarray(params["spike_params"].baseline)
    W = np.asarray(params["spike_params"].weights)
    s = rng.choice(N_STATES, p=np.asarray(params["pi0"]))
    x = rng.multivariate_normal(m0[:, s], P0[:, :, s])
    spikes = []
    for t in range(n_time):
        if t > 0:
            s = rng.choice(N_STATES, p=np.asarray(params["Z"])[s])
            x = A[:, :, s] @ x + rng.multivariate_normal(
                np.zeros(x.shape[0]), Q[:, :, s]
            )
        rate = np.exp(base[:, s] + W[:, :, s] @ x) * DT
        spikes.append(rng.poisson(rate))
    return jnp.asarray(np.array(spikes, dtype=float))


def _all_paths(n_time: int) -> np.ndarray:
    return np.array(list(itertools.product(range(N_STATES), repeat=n_time)))


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("max_newton_iter",))
def _laplace_path_log_weights(
    paths, A, Q, m0, P0, baseline, weights, Z, pi0, spikes, max_newton_iter
):
    """Cumulative ``log p(y_{1:t}, s_{1:t})`` per path, per-path Laplace-EKF.

    Returns shape ``(n_paths, n_time)``; column ``t`` is the joint log weight
    of the path prefix ``s_{1:t+1}`` with ``y_{1:t+1}``.
    """

    def one_path(path):
        def update(mean, cov, t, j):
            params_j = SpikeObsParams(baseline=baseline[:, j], weights=weights[:, :, j])
            return point_process_kalman_update(
                mean,
                cov,
                spikes[t],
                DT,
                _linear_log_intensity,
                params_j,
                max_newton_iter=max_newton_iter,
            )

        mean, cov, ll = update(m0[:, path[0]], P0[:, :, path[0]], 0, path[0])
        log_w = jnp.log(pi0[path[0]]) + ll
        cumulative = [log_w]
        for t in range(1, spikes.shape[0]):
            j = path[t]
            A_j = A[:, :, j]
            mean, cov, ll = update(A_j @ mean, A_j @ cov @ A_j.T + Q[:, :, j], t, j)
            log_w = log_w + jnp.log(Z[path[t - 1], j]) + ll
            cumulative.append(log_w)
        return jnp.stack(cumulative)

    return jax.vmap(one_path)(paths)


def laplace_path_oracle(params: dict, spikes, max_newton_iter: int = 1):
    paths = _all_paths(spikes.shape[0])
    sp = params["spike_params"]
    log_w = _laplace_path_log_weights(
        jnp.asarray(paths),
        params["A"],
        params["Q"],
        params["m0"],
        params["P0"],
        sp.baseline,
        sp.weights,
        params["Z"],
        params["pi0"],
        spikes,
        max_newton_iter=max_newton_iter,
    )
    return paths, np.asarray(log_w)


def quadrature_path_oracle(params: dict, spikes, grid: np.ndarray = QUAD_GRID):
    """Exact ``log p(y_{1:T}, s_{1:T})`` per path by grid forward recursion (1-d)."""
    y = np.asarray(spikes)
    n_time = y.shape[0]
    dx = grid[1] - grid[0]
    A = np.asarray(params["A"])[0, 0]
    Q = np.asarray(params["Q"])[0, 0]
    m0 = np.asarray(params["m0"])[0]
    P0 = np.asarray(params["P0"])[0, 0]
    base = np.asarray(params["spike_params"].baseline)
    W = np.asarray(params["spike_params"].weights)[:, 0, :]
    Z = np.asarray(params["Z"])
    pi0 = np.asarray(params["pi0"])

    def log_normal(x, mean, var):
        return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (x - mean) ** 2 / var

    def log_obs(t, j):
        log_rate = base[:, j][:, None] + W[:, j][:, None] * grid[None, :]
        return np.sum(
            y[t][:, None] * (log_rate + np.log(DT))
            - np.exp(log_rate) * DT
            - gammaln(y[t] + 1.0)[:, None],
            axis=0,
        )

    obs = {(t, j): log_obs(t, j) for t in range(n_time) for j in range(N_STATES)}
    kernel = {
        j: np.exp(log_normal(grid[:, None], A[j] * grid[None, :], Q[j])) * dx
        for j in range(N_STATES)
    }
    paths = _all_paths(n_time)
    log_w = np.zeros(len(paths))
    for k, path in enumerate(paths):
        log_alpha = log_normal(grid, m0[path[0]], P0[path[0]]) + obs[0, path[0]]
        scale = log_alpha.max()
        alpha = np.exp(log_alpha - scale)
        log_prior = np.log(pi0[path[0]])
        for t in range(1, n_time):
            alpha = (kernel[path[t]] @ alpha) * np.exp(obs[t, path[t]])
            norm = alpha.sum()
            alpha /= norm
            scale += np.log(norm)
            log_prior += np.log(Z[path[t - 1], path[t]])
        log_w[k] = scale + np.log(alpha.sum() * dx) + log_prior
    return paths, log_w


def path_posterior(paths: np.ndarray, log_w: np.ndarray) -> dict:
    """Exact discrete posteriors from per-path joint log weights.

    ``log_w`` is ``(n_paths,)`` (full sequence) or ``(n_paths, n_time)``
    (cumulative prefixes, which also yields the filtered marginals).
    """
    n_paths, n_time = paths.shape
    full = log_w if log_w.ndim == 1 else log_w[:, -1]
    log_lik = float(logsumexp(full))
    w = np.exp(full - log_lik)
    marginal = np.zeros((n_time, N_STATES))
    joint = np.zeros((n_time - 1, N_STATES, N_STATES))
    for t in range(n_time):
        np.add.at(marginal[t], paths[:, t], w)
    for t in range(n_time - 1):
        np.add.at(joint[t], (paths[:, t], paths[:, t + 1]), w)
    out = {"log_lik": log_lik, "smoothed": marginal, "joint": joint}
    if log_w.ndim == 2:
        filtered = np.zeros((n_time, N_STATES))
        for t in range(n_time):
            w_t = np.exp(log_w[:, t] - logsumexp(log_w[:, t]))
            np.add.at(filtered[t], paths[:, t], w_t)
        out["filtered"] = filtered
    return out


def prior_chain_posteriors(n_time: int) -> tuple[np.ndarray, np.ndarray]:
    """Markov-chain prior marginals ``P(S_t)`` and two-slice ``P(S_t, S_{t+1})``."""
    Z = np.asarray(DISCRETE_TRANSITION)
    marginal = [np.asarray(INIT_DISCRETE_PROB)]
    for _ in range(n_time - 1):
        marginal.append(marginal[-1] @ Z)
    marginal = np.array(marginal)
    joint = marginal[:-1, :, None] * Z[None, :, :]
    return marginal, joint


def kim_smoother(filtered: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Kim (1994) backward recursion for the discrete marginals."""
    smoothed = np.zeros_like(filtered)
    smoothed[-1] = filtered[-1]
    for t in range(filtered.shape[0] - 2, -1, -1):
        predicted = filtered[t] @ Z
        smoothed[t] = filtered[t] * (Z @ (smoothed[t + 1] / predicted))
    return smoothed


def model_posteriors(params: dict, spikes, max_newton_iter: int = 1) -> dict:
    """The library's switching filter plus both GPB smoothers."""
    (fm, fc, fp, pm, pc, pp, log_lik) = switching_point_process_filter(
        params["m0"],
        params["P0"],
        params["pi0"],
        spikes,
        params["Z"],
        params["A"],
        params["Q"],
        DT,
        _linear_log_intensity,
        params["spike_params"],
        max_newton_iter=max_newton_iter,
    )
    gpb1 = switching_kalman_smoother(fm, fc, fp, params["Q"], params["A"], params["Z"])
    gpb2 = switching_kalman_smoother_gpb2(
        fm, fc, fp, pm, pc, pp, params["Q"], params["A"]
    )
    return {
        "log_lik": float(log_lik),
        "filtered": np.asarray(fp),
        "gpb1": np.asarray(gpb1[2]),
        "gpb1_joint": np.asarray(gpb1[3]),
        "gpb2": np.asarray(gpb2[2]),
        "gpb2_joint": np.asarray(gpb2[3]),
    }


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------------------
# Oracle self-checks
# ---------------------------------------------------------------------------


def test_quadrature_oracle_matches_closed_form_without_latent_coupling():
    """With zero spike weights the path likelihood is a closed-form Poisson product.

    Guards the quadrature oracle itself: the grid must integrate every
    Gaussian prior/transition kernel to one for this to hold.
    """
    params = _params_1d()
    sp = params["spike_params"]
    params["spike_params"] = SpikeObsParams(
        baseline=sp.baseline, weights=jnp.zeros_like(sp.weights)
    )
    spikes = _simulate_spikes(_params_1d(), n_time=4, seed=1)
    paths, log_w = quadrature_path_oracle(params, spikes)
    base = np.asarray(sp.baseline)
    y = np.asarray(spikes)
    for path, lw in zip(paths, log_w):
        rate = np.exp(base[:, path].T) * DT  # (T, N)
        expected = np.sum(y * np.log(rate) - rate - gammaln(y + 1.0))
        expected += np.log(INIT_DISCRETE_PROB[path[0]]) + np.sum(
            np.log(np.asarray(DISCRETE_TRANSITION)[path[:-1], path[1:]])
        )
        np.testing.assert_allclose(lw, expected, rtol=0, atol=1e-10)
    assert y.sum() > 0, "guard: spikes must be nonzero for the check to bite"


# ---------------------------------------------------------------------------
# 1. Identical per-state parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, pytest.param(11, marks=pytest.mark.slow)])
def test_identical_states_posterior_equals_prior(seed):
    """Uninformative switching: every discrete posterior is the chain prior."""
    params = _params_1d(identical=True)
    spikes = _simulate_spikes(params, n_time=5, seed=seed)
    model = model_posteriors(params, spikes)
    prior_marginal, prior_joint = prior_chain_posteriors(5)

    for name in ("gpb1", "gpb2"):
        np.testing.assert_allclose(model[name], prior_marginal, atol=1e-10)
        np.testing.assert_allclose(model[f"{name}_joint"], prior_joint, atol=1e-10)
    np.testing.assert_allclose(model["filtered"], prior_marginal, atol=1e-10)

    # The exact (quadrature) answer is the prior too -- the oracle agrees.
    exact = path_posterior(*quadrature_path_oracle(params, spikes))
    np.testing.assert_allclose(exact["smoothed"], prior_marginal, atol=1e-10)
    # Guard: the data are informative about x (so the check is not vacuous).
    assert float(jnp.sum(spikes)) >= 3


# Laplace LL gap pins |LL_laplace - LL_exact| over 5 bins, per Newton budget.
# Observed (seeds 3, 11): one Fisher-scoring step (the filter default) 0.010,
# 0.31; converged (10 steps) 0.028, 0.033.  The one-step update is expanded at
# the predicted mean, so an informative bin (seed 11 has 4 spikes in one bin)
# leaves a large gap that the converged mode removes.
@pytest.mark.slow
@pytest.mark.parametrize(("max_newton_iter", "upper"), [(1, 0.6), (10, 0.1)])
@pytest.mark.parametrize("seed", [3, 11])
def test_identical_states_log_likelihood_is_single_regime_laplace(
    seed, max_newton_iter, upper
):
    """Model LL = the single-regime Laplace LL; its gap to the exact LL is pinned."""
    params = _params_1d(identical=True)
    spikes = _simulate_spikes(params, n_time=5, seed=seed)
    model = model_posteriors(params, spikes, max_newton_iter)
    laplace = path_posterior(*laplace_path_oracle(params, spikes, max_newton_iter))
    exact = path_posterior(*quadrature_path_oracle(params, spikes))

    np.testing.assert_allclose(model["log_lik"], laplace["log_lik"], atol=1e-9)
    gap = abs(laplace["log_lik"] - exact["log_lik"])
    assert 1e-4 < gap < upper, f"Laplace LL gap {gap:.3e} outside (1e-4, {upper})"


# ---------------------------------------------------------------------------
# 2. T = 2: nothing collapsed yet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_newton_iter", [1, pytest.param(4, marks=pytest.mark.slow)]
)
def test_two_steps_filter_and_gpb2_match_per_path_laplace(max_newton_iter):
    params = _params_1d()
    spikes = _simulate_spikes(params, n_time=2, seed=3)
    model = model_posteriors(params, spikes, max_newton_iter)
    oracle = path_posterior(*laplace_path_oracle(params, spikes, max_newton_iter))

    np.testing.assert_allclose(model["log_lik"], oracle["log_lik"], atol=1e-10)
    np.testing.assert_allclose(model["filtered"], oracle["filtered"], atol=1e-10)
    np.testing.assert_allclose(model["gpb2"], oracle["smoothed"], atol=1e-10)
    np.testing.assert_allclose(model["gpb2_joint"], oracle["joint"], atol=1e-10)

    # GPB1's discrete pass is Kim's recursion, which drops y_2's information
    # about S_1 (carried through x_1): reproduced to round-off, and genuinely
    # different from the exact answer (observed gap 1.3e-3).
    np.testing.assert_allclose(
        model["gpb1"],
        kim_smoother(model["filtered"], np.asarray(params["Z"])),
        atol=1e-12,
    )
    kim_gap = _max_abs(model["gpb1"], oracle["smoothed"])
    assert 1e-5 < kim_gap < 1e-2, f"GPB1 (Kim) gap at T=2: {kim_gap:.3e}"


# ---------------------------------------------------------------------------
# 3. Memoryless dynamics: GPB1 is exact given Laplace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, pytest.param(5, marks=pytest.mark.slow)])
def test_memoryless_dynamics_gpb1_is_exact_given_laplace(seed):
    """With A = 0 the collapse and Kim's recursion lose nothing."""
    params = _params_1d(memoryless=True)
    spikes = _simulate_spikes(params, n_time=5, seed=seed)
    model = model_posteriors(params, spikes)
    oracle = path_posterior(*laplace_path_oracle(params, spikes))

    np.testing.assert_allclose(model["log_lik"], oracle["log_lik"], atol=1e-9)
    np.testing.assert_allclose(model["filtered"], oracle["filtered"], atol=1e-9)
    np.testing.assert_allclose(model["gpb1"], oracle["smoothed"], atol=1e-9)
    np.testing.assert_allclose(model["gpb1_joint"], oracle["joint"], atol=1e-9)
    # Guard: the discrete states are distinguishable from the data.
    prior_marginal, _ = prior_chain_posteriors(5)
    assert _max_abs(oracle["smoothed"], prior_marginal) > 0.05


# ---------------------------------------------------------------------------
# 4. Distinct parameters with memory: pinned approximation gaps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def distinct_t5_gaps() -> dict:
    """Per-seed gaps between the model, the per-path Laplace and the exact answer."""
    params = _params_1d()
    rows = []
    for seed in (3, 4, 5):
        spikes = _simulate_spikes(params, n_time=5, seed=seed)
        model = model_posteriors(params, spikes)
        laplace = path_posterior(*laplace_path_oracle(params, spikes))
        exact = path_posterior(*quadrature_path_oracle(params, spikes))
        rows.append(
            {
                "seed": seed,
                "ll_collapse": abs(model["log_lik"] - laplace["log_lik"]),
                "filtered_collapse": _max_abs(model["filtered"], laplace["filtered"]),
                "gpb1_vs_laplace": _max_abs(model["gpb1"], laplace["smoothed"]),
                "gpb2_vs_laplace": _max_abs(model["gpb2"], laplace["smoothed"]),
                "kim_identity": _max_abs(
                    model["gpb1"],
                    kim_smoother(model["filtered"], np.asarray(params["Z"])),
                ),
                "ll_laplace": abs(laplace["log_lik"] - exact["log_lik"]),
                "smoothed_laplace": _max_abs(laplace["smoothed"], exact["smoothed"]),
                "gpb1_vs_exact": _max_abs(model["gpb1"], exact["smoothed"]),
                "gpb2_vs_exact": _max_abs(model["gpb2"], exact["smoothed"]),
            }
        )
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# (key, lower bound that proves the approximation is active, upper pin).
# Observed per seed (3, 4, 5), T=5, one Newton step:
#   ll_collapse        |LL_model - LL_laplace|        7e-4, 0.026, 8.6e-3
#   filtered_collapse  max |filtered - exact|          5e-5, 0.015, 4.5e-3
#   gpb1_vs_laplace    max |GPB1 smoothed - laplace|   0.030, 0.021, 0.051
#   gpb2_vs_laplace    max |GPB2 smoothed - laplace|   0.011, 0.048, 0.036
#   ll_laplace         |LL_laplace - LL_exact|         3.8e-3, 0.20, 0.16
#   smoothed_laplace   max |laplace - exact| smoothed  9.6e-3, 0.026, 0.015
#   gpb1_vs_exact      max |GPB1 smoothed - exact|     0.035, 0.027, 0.053
#   gpb2_vs_exact      max |GPB2 smoothed - exact|     0.014, 0.041, 0.047
# Upper pins leave ~2-3x headroom; lower bounds assert the gap is real.
_GAP_PINS = [
    ("ll_collapse", 1e-4, 0.06),
    ("filtered_collapse", 1e-4, 0.04),
    ("gpb1_vs_laplace", 1e-3, 0.15),
    ("gpb2_vs_laplace", 1e-3, 0.15),
    ("ll_laplace", 1e-3, 0.5),
    ("smoothed_laplace", 1e-3, 0.08),
    ("gpb1_vs_exact", 1e-3, 0.15),
    ("gpb2_vs_exact", 1e-3, 0.15),
]


@pytest.mark.slow
@pytest.mark.parametrize(("key", "lower", "upper"), _GAP_PINS)
def test_distinct_states_approximation_gap_is_pinned(
    distinct_t5_gaps, key, lower, upper
):
    values = distinct_t5_gaps[key]
    per_seed = dict(zip(distinct_t5_gaps["seed"].tolist(), values.round(5).tolist()))
    assert values.max() < upper, f"{key} gap exceeds pin {upper}: {per_seed}"
    assert values.max() > lower, f"{key} gap unexpectedly vanished: {per_seed}"


@pytest.mark.slow
def test_distinct_states_gpb1_is_kim_recursion(distinct_t5_gaps):
    """Anything beyond Kim's approximation in the GPB1 discrete pass is a bug."""
    assert distinct_t5_gaps["kim_identity"].max() < 1e-12, distinct_t5_gaps[
        "kim_identity"
    ]


# ---------------------------------------------------------------------------
# 5. The model class's E-step (2-d oscillator latent state)
# ---------------------------------------------------------------------------


def _spike_oscillator_model(identical: bool) -> SwitchingSpikeOscillatorModel:
    model = SwitchingSpikeOscillatorModel(
        n_oscillators=1,
        n_neurons=3,
        n_discrete_states=N_STATES,
        sampling_freq=10.0,
        dt=DT,
    )
    model._initialize_parameters(jax.random.PRNGKey(0))
    model.discrete_transition_matrix = DISCRETE_TRANSITION
    model.init_discrete_state_prob = INIT_DISCRETE_PROB
    baseline = jnp.array([[1.0, 2.0], [1.5, 0.5], [0.8, 1.2]])
    weights = jnp.array(
        [
            [[0.8, -0.6], [0.3, 0.5]],
            [[-0.5, 0.9], [0.6, -0.4]],
            [[0.4, 0.2], [-0.7, 0.3]],
        ]
    )
    if identical:
        model.continuous_transition_matrix = jnp.repeat(
            model.continuous_transition_matrix[..., :1], N_STATES, axis=-1
        )
        model.process_cov = jnp.repeat(model.process_cov[..., :1], N_STATES, axis=-1)
        model.init_mean = jnp.repeat(model.init_mean[..., :1], N_STATES, axis=-1)
        baseline = jnp.repeat(baseline[:, :1], N_STATES, axis=-1)
        weights = jnp.repeat(weights[..., :1], N_STATES, axis=-1)
    model.process_cov = model.process_cov * 20.0  # informative dynamics
    model.spike_params = SpikeObsParams(baseline=baseline, weights=weights)
    return model


def _model_params(model: SwitchingSpikeOscillatorModel) -> dict:
    return {
        "A": model.continuous_transition_matrix,
        "Q": model.process_cov,
        "m0": model.init_mean,
        "P0": model.init_cov,
        "spike_params": model.spike_params,
        "Z": model.discrete_transition_matrix,
        "pi0": model.init_discrete_state_prob,
    }


@pytest.mark.slow
@pytest.mark.parametrize("smoother_type", ["gpb1", "gpb2"])
def test_model_e_step_identical_states_posterior_equals_prior(smoother_type):
    model = _spike_oscillator_model(identical=True)
    model.smoother_type = smoother_type
    spikes = _simulate_spikes(_model_params(model), n_time=5, seed=7)
    log_lik = model._e_step(spikes)
    prior_marginal, prior_joint = prior_chain_posteriors(5)
    np.testing.assert_allclose(
        model.smoother_discrete_state_prob, prior_marginal, atol=1e-8
    )
    np.testing.assert_allclose(
        model.smoother_joint_discrete_state_prob, prior_joint, atol=1e-8
    )
    # ... and the LL is the single-regime Laplace LL of the (common) path,
    # computed with the same number of Fisher steps the model uses.
    paths, log_w = laplace_path_oracle(
        _model_params(model), spikes, max_newton_iter=model.max_newton_iter
    )
    np.testing.assert_allclose(
        float(log_lik), path_posterior(paths, log_w)["log_lik"], atol=1e-10
    )
    assert float(jnp.sum(spikes)) >= 3


@pytest.mark.slow
def test_model_e_step_two_steps_matches_per_path_laplace():
    """The class E-step reproduces the 2-d per-path enumeration at T = 2."""
    model = _spike_oscillator_model(identical=False)
    model.smoother_type = "gpb2"
    params = _model_params(model)
    spikes = _simulate_spikes(params, n_time=2, seed=7)
    log_lik = model._e_step(spikes)
    oracle = path_posterior(
        *laplace_path_oracle(params, spikes, max_newton_iter=model.max_newton_iter)
    )
    np.testing.assert_allclose(float(log_lik), oracle["log_lik"], atol=1e-10)
    np.testing.assert_allclose(
        model.smoother_discrete_state_prob, oracle["smoothed"], atol=1e-10
    )
    np.testing.assert_allclose(
        model.smoother_joint_discrete_state_prob, oracle["joint"], atol=1e-10
    )
    prior_marginal, _ = prior_chain_posteriors(2)
    assert _max_abs(oracle["smoothed"], prior_marginal) > 1e-3
