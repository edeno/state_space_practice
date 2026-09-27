# ruff: noqa: E402
"""Switching Kalman filter / GPB smoothers / M-step against exact path enumeration.

:func:`~state_space_practice.tests.oracles.switching_lgssm_exact_posterior`
enumerates every discrete path ``s_{1:T}`` (``K ** T`` of them), solves each
path's linear-Gaussian model exactly by dense conditioning, and mixes the
per-path posteriors with the exact path posterior. It shares no code with the
library.

What is *approximation* and what would be a *bug*
--------------------------------------------------
``switching_kalman_filter`` collapses the ``K^2`` pair-conditional Gaussians
to ``K`` state-conditional ones per step (GPB2 / Kim filter). The GPB1
smoother additionally collapses to one Gaussian per ``S_{t+1}`` and uses Kim's
approximation ``P(S_t | S_{t+1}, y_{1:T}) ~ P(S_t | S_{t+1}, y_{1:t})``; the
GPB2 smoother carries pair-conditional Gaussians and pair probabilities.

* Exact regimes -- any deviation beyond round-off is a bug:

  - one discrete state, or identical per-state parameters (the continuous
    posterior does not depend on the discrete path, so every collapse is of
    identical Gaussians);
  - the filter at ``t = 1, 2`` for *any* parameters (the first collapse
    happens at ``t = 2`` and first affects the prediction for ``t = 3``);
  - the GPB2 smoother for ``T = 2`` and any parameters (its initial carry is
    the exact pair-conditional filter at ``t = 2`` and the backward step
    conditions on ``S_1``'s exact filter).

* Approximation regimes (distinct per-state parameters, ``T >= 3``): the gaps
  are measured and pinned below documented bounds, and asserted nonzero.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from state_space_practice.switching_kalman import (
    switching_kalman_filter,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
    switching_kalman_smoother_gpb2,
)
from state_space_practice.tests.oracles import (
    lgssm_dense_posterior,
    random_spd_matrix,
    random_stable_matrix,
    switching_lgssm_exact_posterior,
    switching_path_sufficient_statistics,
    switching_q_from_statistics,
)

RTOL = 1e-8

_MODEL_KEYS = (
    "init_state_cond_mean",
    "init_state_cond_cov",
    "init_discrete_state_prob",
    "obs",
    "discrete_transition_matrix",
    "continuous_transition_matrix",
    "process_cov",
    "measurement_matrix",
    "measurement_cov",
)


def _random_switching_model(
    rng: np.random.Generator,
    n_latent: int,
    n_obs: int,
    n_states: int,
    n_time: int,
    identical: bool = False,
) -> dict:
    """Random switching model and a trajectory simulated from it.

    With ``identical=True`` every per-state continuous parameter (including
    the initial moments) is shared, so the continuous posterior does not
    depend on the discrete path.
    """

    def per_state(draw):
        if identical:
            value = draw()
            return np.stack([value] * n_states, axis=-1)
        return np.stack([draw() for _ in range(n_states)], axis=-1)

    model = {
        "continuous_transition_matrix": per_state(
            lambda: random_stable_matrix(rng, n_latent)
        ),
        "process_cov": per_state(lambda: random_spd_matrix(rng, n_latent, scale=0.5)),
        "measurement_matrix": per_state(lambda: rng.normal(size=(n_obs, n_latent))),
        "measurement_cov": per_state(lambda: random_spd_matrix(rng, n_obs, scale=0.5)),
        "init_state_cond_mean": per_state(lambda: rng.normal(size=n_latent)),
        "init_state_cond_cov": per_state(lambda: random_spd_matrix(rng, n_latent)),
        "discrete_transition_matrix": 0.5 * np.eye(n_states)
        + 0.5 * rng.dirichlet(2.0 * np.ones(n_states), size=n_states),
        "init_discrete_state_prob": rng.dirichlet(2.0 * np.ones(n_states)),
    }
    # simulate s, x, y
    s = rng.choice(n_states, p=model["init_discrete_state_prob"])
    x = rng.multivariate_normal(
        model["init_state_cond_mean"][:, s], model["init_state_cond_cov"][:, :, s]
    )
    ys = []
    for t in range(n_time):
        if t > 0:
            s = rng.choice(n_states, p=model["discrete_transition_matrix"][s])
            x = model["continuous_transition_matrix"][:, :, s] @ x + (
                rng.multivariate_normal(
                    np.zeros(n_latent), model["process_cov"][..., s]
                )
            )
        ys.append(
            model["measurement_matrix"][:, :, s] @ x
            + rng.multivariate_normal(np.zeros(n_obs), model["measurement_cov"][..., s])
        )
    model["obs"] = np.stack(ys)
    return model


def _run_library(model: dict) -> dict:
    """Filter, GPB1 and GPB2 smoothers on ``model``."""
    args = [jnp.asarray(model[k]) for k in _MODEL_KEYS]
    (fm, fc, fp, pfm, pfc, pfp, ll) = switching_kalman_filter(*args)
    Q = jnp.asarray(model["process_cov"])
    A = jnp.asarray(model["continuous_transition_matrix"])
    Z = jnp.asarray(model["discrete_transition_matrix"])
    g1 = switching_kalman_smoother(fm, fc, fp, Q, A, Z)
    g2 = switching_kalman_smoother_gpb2(fm, fc, fp, pfm, pfc, pfp, Q, A)
    names = (
        "mean",
        "cov",
        "prob",
        "joint_prob",
        "cross_cov",
        "state_cond_mean",
        "state_cond_cov",
    )
    return {
        "filter": {
            "state_cond_mean": np.asarray(fm),
            "state_cond_cov": np.asarray(fc),
            "prob": np.asarray(fp),
            "log_likelihood": float(ll),
        },
        "gpb1": {k: np.asarray(v) for k, v in zip(names, g1[:7])},
        "gpb2": {k: np.asarray(v) for k, v in zip(names, g2[:7])},
        "gpb2_mstep": tuple(np.asarray(v) for v in g2[7:]),
    }


def _oracle(model: dict):
    return switching_lgssm_exact_posterior(**{k: model[k] for k in _MODEL_KEYS})


def _cov_scale(model: dict) -> float:
    return float(
        max(
            np.max(np.abs(model["init_state_cond_cov"])),
            np.max(np.abs(model["process_cov"])),
        )
    )


def _assert_smoother_exact(out: dict, oracle, model: dict, rtol: float) -> None:
    cov_atol = rtol * _cov_scale(model)
    mean_atol = rtol * max(1.0, float(np.max(np.abs(oracle.smoothed_mean))))
    np.testing.assert_allclose(
        out["mean"], oracle.smoothed_mean, rtol=rtol, atol=mean_atol
    )
    np.testing.assert_allclose(
        out["cov"], oracle.smoothed_cov, rtol=rtol, atol=cov_atol
    )
    np.testing.assert_allclose(
        out["cross_cov"], oracle.smoothed_cross_cov, rtol=rtol, atol=cov_atol
    )
    np.testing.assert_allclose(out["prob"], oracle.smoothed_discrete_prob, atol=rtol)
    np.testing.assert_allclose(
        out["joint_prob"], oracle.smoothed_joint_discrete_prob, atol=rtol
    )
    np.testing.assert_allclose(
        out["state_cond_mean"],
        oracle.state_cond_smoothed_mean,
        rtol=rtol,
        atol=mean_atol,
    )
    np.testing.assert_allclose(
        out["state_cond_cov"], oracle.state_cond_smoothed_cov, rtol=rtol, atol=cov_atol
    )


def _assert_filter_exact(
    out: dict, oracle, model: dict, rtol: float, upto=None
) -> None:
    sl = slice(None, upto)
    cov_atol = rtol * _cov_scale(model)
    np.testing.assert_allclose(
        out["prob"][sl], oracle.filtered_discrete_prob[sl], atol=rtol
    )
    np.testing.assert_allclose(
        out["state_cond_mean"][sl],
        oracle.state_cond_filtered_mean[sl],
        rtol=rtol,
        atol=rtol * max(1.0, float(np.max(np.abs(oracle.filtered_mean)))),
    )
    np.testing.assert_allclose(
        out["state_cond_cov"][sl],
        oracle.state_cond_filtered_cov[sl],
        rtol=rtol,
        atol=cov_atol,
    )


class TestSwitchingOracleSelfCheck:
    def test_single_state_reduces_to_dense_lgssm(self) -> None:
        model = _random_switching_model(np.random.default_rng(0), 2, 2, 1, 4)
        oracle = _oracle(model)
        dense = lgssm_dense_posterior(
            model["init_state_cond_mean"][:, 0],
            model["init_state_cond_cov"][..., 0],
            model["obs"],
            model["continuous_transition_matrix"][..., 0],
            model["process_cov"][..., 0],
            model["measurement_matrix"][..., 0],
            model["measurement_cov"][..., 0],
            prior_on_first_state=True,
        )
        np.testing.assert_allclose(oracle.log_likelihood, dense.log_likelihood)
        np.testing.assert_allclose(oracle.smoothed_mean, dense.smoothed_mean)
        np.testing.assert_allclose(oracle.smoothed_cross_cov, dense.smoothed_cross_cov)
        # prior on x_1: x_0 is x_1
        np.testing.assert_allclose(dense.init_smoothed_mean, dense.smoothed_mean[0])

    def test_probabilities_are_consistent(self) -> None:
        model = _random_switching_model(np.random.default_rng(1), 1, 1, 3, 4)
        oracle = _oracle(model)
        np.testing.assert_allclose(oracle.path_posterior_prob.sum(), 1.0)
        np.testing.assert_allclose(oracle.smoothed_discrete_prob.sum(axis=1), 1.0)
        np.testing.assert_allclose(
            oracle.smoothed_joint_discrete_prob.sum(axis=2),
            oracle.smoothed_discrete_prob[:-1],
        )
        np.testing.assert_allclose(
            oracle.smoothed_joint_discrete_prob.sum(axis=1),
            oracle.smoothed_discrete_prob[1:],
        )
        # evidence = log sum_s p(s) p(y | s)
        log_w = oracle.path_log_prior + oracle.path_log_likelihood
        np.testing.assert_allclose(
            oracle.log_likelihood, np.log(np.sum(np.exp(log_w))), rtol=1e-12
        )
        # guard: the discrete posterior is informative (not the prior)
        assert (
            np.max(
                np.abs(
                    oracle.smoothed_discrete_prob[0] - model["init_discrete_state_prob"]
                )
            )
            > 1e-3
        )


# --- exact regimes ----------------------------------------------------------


@settings(max_examples=6, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**31 - 1))
def test_identical_states_filter_and_smoothers_are_exact(seed: int) -> None:
    """Identical per-state parameters: the GPB collapses are exact, so the
    filter, GPB1 and GPB2 must equal path enumeration to round-off."""
    model = _random_switching_model(
        np.random.default_rng(seed), 2, 2, 3, 4, identical=True
    )
    oracle = _oracle(model)
    out = _run_library(model)
    _assert_filter_exact(out["filter"], oracle, model, RTOL)
    np.testing.assert_allclose(
        out["filter"]["log_likelihood"], oracle.log_likelihood, rtol=RTOL, atol=RTOL
    )
    _assert_smoother_exact(out["gpb1"], oracle, model, RTOL)
    _assert_smoother_exact(out["gpb2"], oracle, model, RTOL)
    # guard: the discrete posterior is nontrivial (varies over time/states)
    assert np.ptp(oracle.smoothed_discrete_prob) > 0.05


@pytest.mark.slow
def test_single_state_filter_and_smoothers_are_exact() -> None:
    model = _random_switching_model(np.random.default_rng(3), 2, 2, 1, 5)
    oracle = _oracle(model)
    out = _run_library(model)
    _assert_filter_exact(out["filter"], oracle, model, RTOL)
    np.testing.assert_allclose(
        out["filter"]["log_likelihood"], oracle.log_likelihood, rtol=RTOL
    )
    _assert_smoother_exact(out["gpb1"], oracle, model, RTOL)
    _assert_smoother_exact(out["gpb2"], oracle, model, RTOL)


def test_filter_is_exact_for_two_steps_with_distinct_states() -> None:
    """With distinct per-state parameters the filter is exact at t = 1, 2
    (no collapse has influenced a prediction yet) -- including the
    log-likelihood of y_{1:2} -- and only approximate from t = 3 on."""
    model = _random_switching_model(np.random.default_rng(4), 2, 2, 3, 4)
    oracle = _oracle(model)
    out = _run_library(model)
    _assert_filter_exact(out["filter"], oracle, model, RTOL, upto=2)
    # the two-step LL: rerun the filter on y_{1:2}
    two = dict(model, obs=model["obs"][:2])
    ll_two = float(
        switching_kalman_filter(*[jnp.asarray(two[k]) for k in _MODEL_KEYS])[-1]
    )
    np.testing.assert_allclose(ll_two, oracle.filtered_log_likelihood[1], rtol=RTOL)
    # guard: from t = 3 the collapse makes the filter approximate
    gap = np.max(
        np.abs(
            out["filter"]["state_cond_mean"][2:] - oracle.state_cond_filtered_mean[2:]
        )
    )
    assert gap > 1e-5


@pytest.mark.slow
def test_gpb2_smoother_is_exact_for_two_steps_gpb1_is_not() -> None:
    """For T = 2 and distinct per-state parameters the GPB2 smoother is exact;
    GPB1 is not (Kim's approximation of P(S_1 | S_2, y_{1:2}) by
    P(S_1 | S_2, y_1) drops the dependence of y_2's likelihood on S_1)."""
    model = _random_switching_model(np.random.default_rng(5), 2, 2, 3, 2)
    oracle = _oracle(model)
    out = _run_library(model)
    _assert_filter_exact(out["filter"], oracle, model, RTOL)
    np.testing.assert_allclose(
        out["filter"]["log_likelihood"], oracle.log_likelihood, rtol=RTOL
    )
    _assert_smoother_exact(out["gpb2"], oracle, model, RTOL)
    gpb1_gap = np.max(np.abs(out["gpb1"]["prob"] - oracle.smoothed_discrete_prob))
    assert gpb1_gap > 1e-3


def test_gpb1_discrete_smoother_is_kims_recursion() -> None:
    """The GPB1 discrete smoother is exactly Kim's backward recursion on the
    filtered probabilities -- so its gap to the exact posterior is Kim's
    approximation, not an implementation error::

        P(S_t=j | y) = sum_k P(S_{t+1}=k | y) F_t(j) Z[j, k] / sum_i F_t(i) Z[i, k]
    """
    model = _random_switching_model(np.random.default_rng(4), 2, 2, 3, 4)
    out = _run_library(model)
    F = out["filter"]["prob"]
    Z = model["discrete_transition_matrix"]
    kim = np.zeros_like(F)
    kim[-1] = F[-1]
    for t in range(F.shape[0] - 2, -1, -1):
        pred = F[t] @ Z
        kim[t] = F[t] * (Z @ (kim[t + 1] / pred))
    np.testing.assert_allclose(out["gpb1"]["prob"], kim, rtol=1e-10, atol=1e-12)
    # guard: Kim's answer is not the exact one here
    oracle = _oracle(model)
    assert np.max(np.abs(kim - oracle.smoothed_discrete_prob)) > 1e-3


# --- approximation regime: pinned gaps --------------------------------------

# 12 seeded problems simulated from their own model, T = 5, (n, m, K)
# alternating between the two shapes below. Observed maxima over these 12
# problems: filter LL 1.9e-3 nats, filter P(S_t | y_{1:t}) 3.9e-4; smoothed
# P(S_t | y) GPB1 0.245, GPB2 0.046; smoothed-mean error in posterior sd GPB1
# 0.45, GPB2 0.081 (minima 5e-7 .. 1e-3, i.e. never zero). Bounds are ~2x the
# observed maxima. In a wider 40-problem sweep GPB2's smoothed P(S_t) was the
# closer one on 36/40 problems (never worse by more than 15%) and its
# smoothed mean on 40/40.
_GAP_SHAPES = ((2, 1, 2), (1, 2, 3))
_GAP_BOUNDS = {
    "filter_ll": 4e-3,
    "filter_prob": 1e-3,
    "gpb1_prob": 0.5,
    "gpb2_prob": 0.1,
    "gpb1_mean_sd": 1.0,
    "gpb2_mean_sd": 0.2,
}


def _gap_problem_errors(seed: int) -> dict:
    n, m, K = _GAP_SHAPES[seed % 2]
    model = _random_switching_model(np.random.default_rng(100 + seed), n, m, K, 5)
    oracle = _oracle(model)
    out = _run_library(model)
    sd = np.sqrt(np.einsum("tii->ti", oracle.smoothed_cov))
    return {
        "filter_ll": abs(out["filter"]["log_likelihood"] - oracle.log_likelihood),
        "filter_prob": np.max(
            np.abs(out["filter"]["prob"] - oracle.filtered_discrete_prob)
        ),
        "gpb1_prob": np.max(
            np.abs(out["gpb1"]["prob"] - oracle.smoothed_discrete_prob)
        ),
        "gpb2_prob": np.max(
            np.abs(out["gpb2"]["prob"] - oracle.smoothed_discrete_prob)
        ),
        "gpb1_mean_sd": np.max(np.abs(out["gpb1"]["mean"] - oracle.smoothed_mean) / sd),
        "gpb2_mean_sd": np.max(np.abs(out["gpb2"]["mean"] - oracle.smoothed_mean) / sd),
    }


@pytest.mark.slow
def test_gpb_approximation_gaps_are_pinned() -> None:
    """Distinct per-state parameters, T = 5: the GPB approximation gaps stay
    below the documented bounds, are genuinely nonzero, and GPB2 is at least
    as close as GPB1 (per problem for the smoothed mean; on average for the
    discrete probabilities, where GPB2 is closer on most problems but not
    all -- see the sweep noted above)."""
    errors = [_gap_problem_errors(seed) for seed in range(12)]
    table = {k: np.array([e[k] for e in errors]) for k in _GAP_BOUNDS}
    for name, bound in _GAP_BOUNDS.items():
        assert np.max(table[name]) < bound, (name, table[name])
        # guard: the approximation really is an approximation here
        assert np.max(table[name]) > 1e-4, (name, table[name])
    assert np.all(table["gpb2_mean_sd"] <= table["gpb1_mean_sd"] + 1e-12)
    assert np.mean(table["gpb2_prob"]) < 0.5 * np.mean(table["gpb1_prob"])
    assert np.mean(table["gpb2_prob"] <= table["gpb1_prob"]) >= 0.75


# --- M-step against the exact switching Q ------------------------------------

_SYM = {"init_state_cond_cov", "process_cov", "measurement_cov"}
_CONT_NAMES = (
    "init_state_cond_mean",
    "init_state_cond_cov",
    "continuous_transition_matrix",
    "process_cov",
    "measurement_matrix",
    "measurement_cov",
)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _q(stats: dict, p: dict) -> float:
    return switching_q_from_statistics(
        stats,
        p["init_state_cond_mean"],
        p["init_state_cond_cov"],
        _softmax(p["init_logits"]),
        _softmax(p["transition_logits"], axis=1),
        p["continuous_transition_matrix"],
        p["process_cov"],
        p["measurement_matrix"],
        p["measurement_cov"],
    )


def _fd_grad(fun, params: dict, names, step: float = 1e-5) -> dict:
    grads = {}
    for name in names:
        value = np.asarray(params[name], dtype=float)
        grad = np.zeros_like(value)
        for idx in np.ndindex(value.shape):
            if name in _SYM and idx[0] > idx[1]:
                continue
            d = np.zeros_like(value)
            d[idx] = 1.0
            if name in _SYM:
                d[(idx[1], idx[0], *idx[2:])] = 1.0
            h = step * max(1.0, float(np.max(np.abs(value))))
            grad[idx] = (
                fun(dict(params, **{name: value + h * d}))
                - fun(dict(params, **{name: value - h * d}))
            ) / (2 * h)
        grads[name] = grad
    return grads


def _m_step_from_exact_stats(model: dict, oracle, **kwargs) -> dict:
    out = switching_kalman_maximization_step(
        obs=jnp.asarray(model["obs"]),
        state_cond_smoother_means=jnp.asarray(oracle.state_cond_smoothed_mean),
        state_cond_smoother_covs=jnp.asarray(oracle.state_cond_smoothed_cov),
        smoother_discrete_state_prob=jnp.asarray(oracle.smoothed_discrete_prob),
        smoother_joint_discrete_state_prob=jnp.asarray(
            oracle.smoothed_joint_discrete_prob
        ),
        pair_cond_smoother_cross_cov=jnp.asarray(oracle.pair_cond_smoothed_cross_cov),
        pair_cond_smoother_means=jnp.asarray(oracle.pair_cond_smoothed_mean),
        pair_cond_smoother_covs=jnp.asarray(oracle.pair_cond_smoothed_cov),
        next_pair_cond_smoother_means=jnp.asarray(oracle.next_pair_cond_smoothed_mean),
        **kwargs,
    )
    A, H, Q, R, m0, P0, Z, pi = (np.asarray(v) for v in out)
    return {
        "continuous_transition_matrix": A,
        "measurement_matrix": H,
        "process_cov": Q,
        "measurement_cov": R,
        "init_state_cond_mean": m0,
        "init_state_cond_cov": P0,
        "transition_logits": np.log(Z),
        "init_logits": np.log(pi),
    }


def _params_of(model: dict) -> dict:
    p = {k: np.asarray(model[k]) for k in _CONT_NAMES}
    p["transition_logits"] = np.log(model["discrete_transition_matrix"])
    p["init_logits"] = np.log(model["init_discrete_state_prob"])
    return p


@pytest.fixture(scope="module")
def mstep_problem():
    """Distinct-state model (n=2, m=1, K=2, T=5) with its exact posterior
    and path-enumerated sufficient statistics."""
    model = _random_switching_model(np.random.default_rng(21), 2, 1, 2, 5)
    oracle = _oracle(model)
    stats = switching_path_sufficient_statistics(oracle, model["obs"])
    return model, oracle, stats


class TestSwitchingMStepMaximisesExactQ:
    """Fed the *exact* posterior statistics (state-conditional and
    pair-conditional moments from path enumeration), the switching M-step
    must return the maximiser of the exact switching EM auxiliary function

        Q(theta) = sum_s P(s | y) E[log p(s, x, y | theta) | s, y],

    evaluated from per-path statistics (not from the moments the M-step
    consumes). This checks the update formulas *and* the reduction of the
    exact Q to the M-step's sufficient statistics (e.g. that the next-state
    second moment may be taken from the state-conditional smoother by total
    expectation). The initial-state update installs E[x_1 | S_1=j, y] and
    Cov[x_1 | S_1=j, y]: the switching filter places its prior on x_1 (no
    x_0 -> x_1 transition), so this is already the exact update."""

    def test_unconstrained_update_is_stationary(self, mstep_problem) -> None:
        model, oracle, stats = mstep_problem
        new = _m_step_from_exact_stats(model, oracle)
        names = (*_CONT_NAMES, "transition_logits", "init_logits")
        grads = _fd_grad(lambda p: _q(stats, p), new, names)
        rng = np.random.default_rng(0)
        old = _params_of(_random_switching_model(rng, 2, 1, 2, 5))
        g_old = max(
            np.max(np.abs(g))
            for g in _fd_grad(lambda p: _q(stats, p), old, names).values()
        )
        assert g_old > 1e-2  # guard
        for name, g in grads.items():
            np.testing.assert_allclose(g, 0.0, atol=1e-6, err_msg=name)
        q_new = _q(stats, new)
        assert q_new >= _q(stats, old)
        assert q_new >= _q(stats, _params_of(model))

    @pytest.mark.parametrize(
        "fixed", ["measurement_matrix", "continuous_transition_matrix"]
    )
    def test_fixed_matrix_update_is_constrained_maximiser(
        self, mstep_problem, fixed: str
    ) -> None:
        """Holding H (resp. A) fixed, the returned R (resp. Q) -- and every
        other parameter -- is stationary for Q restricted to that H (A)."""
        model, oracle, stats = mstep_problem
        rng = np.random.default_rng(1)
        fixed_value = np.asarray(model[fixed]) + 0.3 * rng.normal(
            size=np.shape(model[fixed])
        )
        kwarg = (
            "fixed_measurement_matrix"
            if fixed == "measurement_matrix"
            else "fixed_continuous_transition_matrix"
        )
        new = _m_step_from_exact_stats(
            model, oracle, **{kwarg: jnp.asarray(fixed_value)}
        )
        np.testing.assert_array_equal(new[fixed], fixed_value)
        names = [
            n for n in (*_CONT_NAMES, "transition_logits", "init_logits") if n != fixed
        ]
        for name, g in _fd_grad(lambda p: _q(stats, p), new, names).items():
            np.testing.assert_allclose(g, 0.0, atol=1e-6, err_msg=name)
        # guard: the unconstrained covariance differs, so the constrained
        # form matters here.
        free = _m_step_from_exact_stats(model, oracle)
        cov = "measurement_cov" if fixed == "measurement_matrix" else "process_cov"
        assert np.max(np.abs(free[cov] - new[cov])) > 1e-3

    def test_previous_params_gate_keeps_unoccupied_state(self) -> None:
        """A state with expected occupancy below n_latent + 1 keeps the
        previous per-state A/Q/H/R; the occupied state's update is still the
        exact (stationary) one."""
        rng = np.random.default_rng(31)
        model = _random_switching_model(rng, 2, 1, 2, 5)
        model["init_discrete_state_prob"] = np.array([0.97, 0.03])
        model["discrete_transition_matrix"] = np.array([[0.97, 0.03], [0.6, 0.4]])
        oracle = _oracle(model)
        stats = switching_path_sufficient_statistics(oracle, model["obs"])
        occupancy = oracle.smoothed_discrete_prob.sum(axis=0)
        assert occupancy[1] < 3.0 <= occupancy[0]  # guard: state 1 is gated
        previous = _params_of(_random_switching_model(rng, 2, 1, 2, 5))
        gated = (
            "continuous_transition_matrix",
            "process_cov",
            "measurement_matrix",
            "measurement_cov",
        )
        new = _m_step_from_exact_stats(
            model,
            oracle,
            previous_params={k: jnp.asarray(previous[k]) for k in gated},
        )
        for name in gated:
            np.testing.assert_array_equal(new[name][..., 1], previous[name][..., 1])
        # state 0's continuous parameters are stationary
        grads = _fd_grad(lambda p: _q(stats, p), new, _CONT_NAMES)
        for name, g in grads.items():
            np.testing.assert_allclose(g[..., 0], 0.0, atol=1e-6, err_msg=name)
