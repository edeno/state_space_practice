# ruff: noqa: E402
"""Simulation-based calibration of the Gaussian filter and smoothers.

If the latent trajectory is drawn from the model the filter/smoother assumes,
the posterior it reports must be *calibrated*: the whitened error
``L_t^{-1} (x_t - m_t)`` with ``P_t = L_t L_t^T`` is exactly N(0, I) (over the
joint draw of latents and data, for any fixed parameters), for the filter (``m_{t|t}``,
``P_{t|t}``) and the smoother (``m_{t|T}``, ``P_{t|T}``) alike.

Every check is per time step over independent replicates, so the sample
count is exact: with ``N`` iid N(0, 1) values the sample mean has sd
``1/sqrt(N)``, the sample variance ``~sqrt(2/N)``, and the empirical
coverage of the central 90% interval ``sqrt(0.09/N)``. Tolerances are
``4.5`` of those standard deviations (family-wise false-alarm probability
< 1e-3 over all the checks in this module; the seeds are fixed, so the tests
are deterministic). Each test also runs the same checks on a deliberately
misspecified posterior and asserts that they *fail* there.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from state_space_practice.kalman import kalman_filter, kalman_smoother
from state_space_practice.switching_kalman import (
    switching_kalman_filter,
    switching_kalman_smoother,
    switching_kalman_smoother_gpb2,
)
from state_space_practice.tests.oracles import random_spd_matrix, random_stable_matrix

N_SIGMA = 4.5
Z90 = float(norm.ppf(0.95))


def _whitened_errors(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """L^{-1} (x - mean) with cov = L L^T; arrays (..., n) / (..., n, n)."""
    chol = np.linalg.cholesky(cov)
    return np.linalg.solve(chol, (x - mean)[..., None])[..., 0]


def _calibration_violations(z: np.ndarray) -> list[str]:
    """Per-time checks on whitened errors ``z`` of shape (reps, T, n).

    Returns the failed checks (empty when calibrated). At each time the
    ``reps * n`` values are iid N(0, 1) under calibration (replicates are
    independent; whitening decorrelates the coordinates).
    """
    failures = []
    n_values = z.shape[0] * z.shape[2]
    tol_mean = N_SIGMA / np.sqrt(n_values)
    tol_var = N_SIGMA * np.sqrt(2.0 / n_values)
    tol_cov = N_SIGMA * np.sqrt(0.09 / n_values)
    for t in range(z.shape[1]):
        zt = z[:, t, :].reshape(-1)
        mean, var = zt.mean(), zt.var()
        coverage = np.mean(np.abs(zt) < Z90)
        if abs(mean) > tol_mean:
            failures.append(f"t={t}: mean {mean:.4f} (tol {tol_mean:.4f})")
        if abs(var - 1.0) > tol_var:
            failures.append(f"t={t}: variance {var:.4f} (tol {tol_var:.4f})")
        if abs(coverage - 0.9) > tol_cov:
            failures.append(f"t={t}: 90% coverage {coverage:.4f} (tol {tol_cov:.4f})")
    return failures


def _random_lgssm(rng: np.random.Generator, n: int, m: int) -> dict:
    return {
        "init_mean": rng.normal(size=n),
        "init_cov": random_spd_matrix(rng, n),
        "transition_matrix": random_stable_matrix(rng, n),
        "process_cov": random_spd_matrix(rng, n, scale=0.3),
        "measurement_matrix": rng.normal(size=(m, n)),
        "measurement_cov": random_spd_matrix(rng, m, scale=0.5),
    }


def _simulate_lgssm_batch(
    rng: np.random.Generator, p: dict, n_reps: int, n_time: int
) -> tuple[np.ndarray, np.ndarray]:
    """x_0 ~ N(m0, P0), x_t = A x_{t-1} + w_t, y_t = H x_t + v_t (t = 1..T).

    Returns x (reps, T, n) and y (reps, T, m).
    """
    n = p["init_mean"].shape[0]
    m = p["measurement_matrix"].shape[0]
    Lq = np.linalg.cholesky(p["process_cov"])
    Lr = np.linalg.cholesky(p["measurement_cov"])
    x = (
        p["init_mean"]
        + rng.normal(size=(n_reps, n)) @ np.linalg.cholesky(p["init_cov"]).T
    )
    xs, ys = [], []
    for _ in range(n_time):
        x = x @ p["transition_matrix"].T + rng.normal(size=(n_reps, n)) @ Lq.T
        xs.append(x)
        ys.append(x @ p["measurement_matrix"].T + rng.normal(size=(n_reps, m)) @ Lr.T)
    return np.stack(xs, axis=1), np.stack(ys, axis=1)


def _batched(fun):
    """vmap a (model..., obs) Kalman routine over a batch of observations."""
    return jax.jit(
        jax.vmap(
            lambda obs, m0, P0, A, Q, H, R: fun(m0, P0, obs, A, Q, H, R, False),
            in_axes=(0, None, None, None, None, None, None),
        )
    )


_batched_filter = _batched(kalman_filter)
_batched_smoother = _batched(kalman_smoother)


def _lgssm_errors(
    n_param_sets: int,
    n_reps: int,
    n_time: int,
    seed: int,
    process_cov_factor: float = 1.0,
    units: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Whitened filter and smoother errors pooled over random parameter sets.

    ``process_cov_factor`` != 1 runs the filter/smoother with a misspecified
    process covariance (data still simulated from the true one). ``units``
    rescales x and y (covariances by ``units**2``).
    """
    rng = np.random.default_rng(seed)
    z_filter, z_smoother = [], []
    for _ in range(n_param_sets):
        p = _random_lgssm(rng, 2, 2)
        p["init_mean"] = units * p["init_mean"]
        for key in ("init_cov", "process_cov", "measurement_cov"):
            p[key] = units**2 * p[key]
        x, y = _simulate_lgssm_batch(rng, p, n_reps, n_time)
        args = (
            jnp.asarray(p["init_mean"]),
            jnp.asarray(p["init_cov"]),
            jnp.asarray(p["transition_matrix"]),
            jnp.asarray(process_cov_factor * p["process_cov"]),
            jnp.asarray(p["measurement_matrix"]),
            jnp.asarray(p["measurement_cov"]),
        )
        f_mean, f_cov, _ = _batched_filter(jnp.asarray(y), *args)
        s_mean, s_cov, _, _ = _batched_smoother(jnp.asarray(y), *args)
        z_filter.append(_whitened_errors(x, np.asarray(f_mean), np.asarray(f_cov)))
        z_smoother.append(_whitened_errors(x, np.asarray(s_mean), np.asarray(s_cov)))
    return np.concatenate(z_filter), np.concatenate(z_smoother)


@pytest.fixture(scope="module")
def lgssm_errors() -> tuple[np.ndarray, np.ndarray]:
    # 4 random parameter sets x 500 replicates, T = 10 -> 4000 values per time.
    return _lgssm_errors(n_param_sets=4, n_reps=500, n_time=10, seed=0)


def test_kalman_smoother_is_calibrated(lgssm_errors) -> None:
    _, z_smoother = lgssm_errors
    assert _calibration_violations(z_smoother) == []


def test_kalman_filter_is_calibrated(lgssm_errors) -> None:
    z_filter, z_smoother = lgssm_errors
    assert _calibration_violations(z_filter) == []
    # guard: filter and smoother errors are different draws (the smoother
    # is sharper), so both checks carry information.
    assert np.mean(np.abs(z_filter - z_smoother)) > 0.05


def test_kalman_smoother_is_calibrated_in_small_units() -> None:
    """Covariances ~1e-10 (e.g. volts): calibration must not depend on the
    units. Note that calibration alone could not catch the former absolute
    1e-9 Cholesky shift in the gain solves: the Joseph-form covariance is the
    true error covariance of whatever gain is used, so a distorted gain gives
    a suboptimal but still calibrated posterior. That defect is caught by the
    exact-oracle test ``test_oracle_kalman.py::test_recursions_are_unit_equivariant``."""
    z_filter, z_smoother = _lgssm_errors(
        n_param_sets=4, n_reps=500, n_time=10, seed=0, units=1e-5
    )
    assert _calibration_violations(z_filter) == []
    assert _calibration_violations(z_smoother) == []


def test_calibration_check_detects_misspecified_posterior() -> None:
    """Power guard: a smoother run with the process covariance doubled
    reports over-wide posteriors, and the same checks flag it."""
    _, z_wrong = _lgssm_errors(
        n_param_sets=4, n_reps=500, n_time=10, seed=0, process_cov_factor=2.0
    )
    violations = _calibration_violations(z_wrong)
    assert any("variance" in v for v in violations), violations


# --- switching smoother in its exact regime ----------------------------------


def _switching_exact_regime_batch(
    rng: np.random.Generator, n_reps: int, n_time: int, n_states: int = 2
) -> dict:
    """Identical per-state continuous parameters (the GPB collapses are exact)
    and a nontrivial Markov chain; simulates (S, x, y) with the switching
    filter's convention (prior on x_1, measurement-only update at t = 1)."""
    n, m = 2, 2
    p = _random_lgssm(rng, n, m)
    Z = np.array([[0.8, 0.2], [0.3, 0.7]])
    pi = np.array([0.35, 0.65])

    Lq = np.linalg.cholesky(p["process_cov"])
    Lr = np.linalg.cholesky(p["measurement_cov"])
    x = (
        p["init_mean"]
        + rng.normal(size=(n_reps, n)) @ np.linalg.cholesky(p["init_cov"]).T
    )
    s = (rng.uniform(size=n_reps) > pi[0]).astype(int)
    xs, ys, ss = [], [], []
    for t in range(n_time):
        if t > 0:
            x = x @ p["transition_matrix"].T + rng.normal(size=(n_reps, n)) @ Lq.T
            s = (rng.uniform(size=n_reps) > Z[s, 0]).astype(int)
        xs.append(x)
        ss.append(s)
        ys.append(x @ p["measurement_matrix"].T + rng.normal(size=(n_reps, m)) @ Lr.T)

    def stack(a):
        return jnp.asarray(np.stack([a] * n_states, axis=-1))

    return {
        "x": np.stack(xs, axis=1),
        "s": np.stack(ss, axis=1),
        "y": np.stack(ys, axis=1),
        "args": (
            stack(p["init_mean"]),
            stack(p["init_cov"]),
            jnp.asarray(pi),
        ),
        "dyn": (
            jnp.asarray(Z),
            stack(p["transition_matrix"]),
            stack(p["process_cov"]),
            stack(p["measurement_matrix"]),
            stack(p["measurement_cov"]),
        ),
    }


def _switching_posterior(obs, m0, P0, pi, Z, A, Q, H, R):
    fm, fc, fp, pfm, pfc, pfp, _ = switching_kalman_filter(
        m0, P0, pi, obs, Z, A, Q, H, R
    )
    g1 = switching_kalman_smoother(fm, fc, fp, Q, A, Z)
    g2 = switching_kalman_smoother_gpb2(fm, fc, fp, pfm, pfc, pfp, Q, A)
    return g1[0], g1[1], g1[2], g2[0], g2[1], g2[2]


def _batched_switching(obs_batch, *params):
    """Loop (not vmap) over replicates: the switching filter/smoothers emit
    host-side warnings through ``lax.cond`` + ``jax.debug.print``, and under
    ``vmap`` the cond lowers to ``select`` and every warning fires on every
    element (the caveat noted in ``switching_kalman._cap_covariance_trace``)."""
    outs = [_switching_posterior(obs, *params) for obs in obs_batch]
    return tuple(np.stack([np.asarray(o[i]) for o in outs]) for i in range(6))


@pytest.mark.slow
def test_switching_smoothers_are_calibrated_in_exact_regime() -> None:
    """GPB1 and GPB2 smoothers with identical per-state parameters: the
    continuous posterior is calibrated, and the smoothed P(S_t = j | y)
    matches the empirical frequency of S_t = j (binomial tolerance)."""
    rng = np.random.default_rng(1)
    data = _switching_exact_regime_batch(rng, n_reps=2000, n_time=6)
    out = _batched_switching(jnp.asarray(data["y"]), *data["args"], *data["dyn"])
    assert out[0].shape[0] == data["x"].shape[0]
    g1_mean, g1_cov, g1_prob, g2_mean, g2_cov, g2_prob = (np.asarray(o) for o in out)
    for mean, cov, prob in ((g1_mean, g1_cov, g1_prob), (g2_mean, g2_cov, g2_prob)):
        assert _calibration_violations(_whitened_errors(data["x"], mean, cov)) == []
        # discrete: frequency of S_t = 1 vs the mean reported probability
        freq = np.mean(data["s"] == 1, axis=0)
        reported = prob[..., 1].mean(axis=0)
        tol = N_SIGMA * np.sqrt(reported * (1 - reported) / data["s"].shape[0])
        assert np.all(np.abs(freq - reported) < tol), (freq, reported)
    # guard: the discrete chain is not stationary at t = 1, so the reported
    # probability must change over time for the frequency check to bite.
    assert np.ptp(g1_prob[0, :, 1]) > 0.05
    # power guard: pretending the posterior is 20% narrower fails the check
    z_narrow = _whitened_errors(data["x"], g1_mean, 0.8 * g1_cov)
    assert any("variance" in v for v in _calibration_violations(z_narrow))
