# ruff: noqa: E402
"""The Kalman filter, RTS smoother and EM M-step against exact dense references.

The references in :mod:`state_space_practice.tests.oracles` build the joint
Gaussian over ``(x_0, ..., x_T, y_1, ..., y_T)`` explicitly and condition on
``y`` -- no recursion, no shared code with the library -- so agreement to
~1e-8 relative is independent evidence that the recursions compute the exact
posterior, and the M-step tests check the returned parameters against the
exact EM auxiliary function written out term by term.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from state_space_practice.kalman import (
    InitialStatePrior,
    kalman_filter,
    kalman_maximization_step,
    kalman_smoother,
    parallel_kalman_smoother,
    smooth_initial_state,
    smooth_initial_state_with_cross_cov,
)
from state_space_practice.tests.oracles import (
    lgssm_dense_posterior,
    lgssm_expected_complete_log_likelihood,
    random_spd_matrix,
    random_stable_matrix,
)

# Relative agreement demanded of the recursions. Observed agreement on
# well-conditioned problems is ~1e-13; the loosest cases (n_obs > n_latent
# with R ~ 1e-6, innovation covariance condition number ~1e7) reach ~2e-8.
RTOL = 1e-8
RTOL_ILL_CONDITIONED = 1e-6


def _simulate_lgssm(
    rng: np.random.Generator,
    n_latent: int,
    n_obs: int,
    n_time: int,
    r_scale: float = 1.0,
    rank_deficient_h: bool = False,
    scale: float = 1.0,
) -> dict:
    """Random stable model and a trajectory simulated from it.

    ``scale`` rescales the units of x and y (means by ``scale``, covariances
    by ``scale**2``); the exact posterior is equivariant under it.
    """
    A = random_stable_matrix(rng, n_latent)
    Q = random_spd_matrix(rng, n_latent, scale=0.5)
    P0 = random_spd_matrix(rng, n_latent)
    m0 = rng.normal(size=n_latent)
    H = rng.normal(size=(n_obs, n_latent))
    if rank_deficient_h:
        # rank 1: every row a multiple of the first
        H = np.outer(rng.normal(size=n_obs), H[0])
    R = random_spd_matrix(rng, n_obs, scale=r_scale)
    x = rng.multivariate_normal(m0, P0)
    ys = []
    for _ in range(n_time):
        x = A @ x + rng.multivariate_normal(np.zeros(n_latent), Q)
        ys.append(H @ x + rng.multivariate_normal(np.zeros(n_obs), R))
    return {
        "init_mean": scale * m0,
        "init_cov": scale**2 * P0,
        "obs": scale * np.stack(ys),
        "transition_matrix": A,
        "process_cov": scale**2 * Q,
        "measurement_matrix": H,
        "measurement_cov": scale**2 * R,
    }


def _args(model: dict) -> tuple:
    keys = (
        "init_mean",
        "init_cov",
        "obs",
        "transition_matrix",
        "process_cov",
        "measurement_matrix",
        "measurement_cov",
    )
    return tuple(jnp.asarray(model[k]) for k in keys)


def _oracle(model: dict):
    return lgssm_dense_posterior(**model)


def _mean_scale(oracle) -> float:
    """Natural absolute scale of the latent means (for atol)."""
    prior_sd = np.sqrt(np.max(np.diag(oracle.joint_cov)))
    return max(float(np.max(np.abs(oracle.smoothed_mean))), prior_sd, 1e-300)


def _cov_scale(model: dict) -> float:
    return float(
        max(np.max(np.abs(model["init_cov"])), np.max(np.abs(model["process_cov"])))
    )


def _assert_matches_oracle(model: dict, rtol: float):
    """Filter, smoother, x_0 smoothing and the log-likelihood vs the oracle."""
    oracle = _oracle(model)
    mean_atol = rtol * _mean_scale(oracle)

    def cov_close(actual, desired) -> None:
        # relative to the compared array's own scale (posterior covariances
        # are ~1e-7 when R ~ 1e-6), floored at the oracle's round-off level.
        atol = max(rtol * float(np.max(np.abs(desired))), 1e-13 * _cov_scale(model))
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)

    f_mean, f_cov, f_ll = kalman_filter(*_args(model))
    s_mean, s_cov, s_cross, s_ll = kalman_smoother(*_args(model))

    np.testing.assert_allclose(f_mean, oracle.filtered_mean, rtol=rtol, atol=mean_atol)
    cov_close(f_cov, oracle.filtered_cov)
    np.testing.assert_allclose(s_mean, oracle.smoothed_mean, rtol=rtol, atol=mean_atol)
    cov_close(s_cov, oracle.smoothed_cov)
    cov_close(s_cross, oracle.smoothed_cross_cov)
    # log-likelihood: an O(T) sum of O(1) terms; compare absolutely too.
    np.testing.assert_allclose(f_ll, oracle.log_likelihood, rtol=rtol, atol=rtol)
    np.testing.assert_allclose(s_ll, oracle.log_likelihood, rtol=rtol, atol=rtol)

    prior = InitialStatePrior(
        *(
            jnp.asarray(model[k])
            for k in ("init_mean", "init_cov", "transition_matrix", "process_cov")
        )
    )
    m00, P00, C01 = smooth_initial_state_with_cross_cov(prior, s_mean[0], s_cov[0])
    np.testing.assert_allclose(
        m00, oracle.init_smoothed_mean, rtol=rtol, atol=mean_atol
    )
    cov_close(P00, oracle.init_smoothed_cov)
    cov_close(C01, oracle.init_cross_cov)
    m00_b, P00_b = smooth_initial_state(prior, s_mean[0], s_cov[0])
    np.testing.assert_array_equal(m00_b, m00)
    np.testing.assert_array_equal(P00_b, P00)

    return oracle


class TestOracleSelfCheck:
    """The dense oracle itself against closed forms, so its agreement with the
    recursions cannot be a shared mistake."""

    def test_scalar_single_step_closed_form(self) -> None:
        a, q, r, m0, p0, y = 0.7, 0.3, 0.5, 1.0, 2.0, 0.4
        oracle = lgssm_dense_posterior(
            np.array([m0]),
            np.array([[p0]]),
            np.array([[y]]),
            np.array([[a]]),
            np.array([[q]]),
            np.array([[1.0]]),
            np.array([[r]]),
        )
        p_pred = a * a * p0 + q
        k = p_pred / (p_pred + r)
        np.testing.assert_allclose(
            oracle.filtered_mean[0, 0], a * m0 + k * (y - a * m0)
        )
        np.testing.assert_allclose(oracle.filtered_cov[0, 0, 0], (1 - k) * p_pred)
        s = p_pred + r
        ll = -0.5 * (np.log(2 * np.pi * s) + (y - a * m0) ** 2 / s)
        np.testing.assert_allclose(oracle.log_likelihood, ll, rtol=1e-14)
        # x_0 | y_1: joint (x_0, y_1) has Cov = a p0.
        g = a * p0 / s
        np.testing.assert_allclose(oracle.init_smoothed_mean[0], m0 + g * (y - a * m0))
        np.testing.assert_allclose(oracle.init_smoothed_cov[0, 0], p0 - g * a * p0)
        np.testing.assert_allclose(
            oracle.init_cross_cov[0, 0], a * p0 - g * s * k, rtol=1e-12
        )

    def test_filtered_last_equals_smoothed_last(self) -> None:
        model = _simulate_lgssm(np.random.default_rng(0), 2, 2, 4)
        oracle = _oracle(model)
        np.testing.assert_allclose(
            oracle.filtered_mean[-1], oracle.smoothed_mean[-1], rtol=1e-12
        )
        np.testing.assert_allclose(
            oracle.filtered_cov[-1], oracle.smoothed_cov[-1], rtol=1e-10
        )
        # guard: earlier smoothed moments differ from filtered ones
        assert not np.allclose(oracle.filtered_mean[0], oracle.smoothed_mean[0])


@settings(max_examples=10, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**31 - 1))
def test_recursions_match_oracle_fixed_shape(seed: int) -> None:
    """Random stable A, random SPD Q/R/P0, one shape (single compile)."""
    model = _simulate_lgssm(np.random.default_rng(seed), 2, 3, 5)
    oracle = _assert_matches_oracle(model, RTOL)
    # guard: smoothing moves x_1 well beyond the tolerance, so a smoother that
    # returned the filter output could not pass.
    assert np.max(np.abs(oracle.smoothed_mean[0] - oracle.filtered_mean[0])) > (
        1e4 * RTOL * _mean_scale(oracle)
    )


@pytest.mark.slow
@settings(max_examples=30, deadline=None, derandomize=True)
@given(
    seed=st.integers(0, 2**31 - 1),
    n_latent=st.integers(1, 3),
    n_obs=st.integers(1, 3),
    n_time=st.integers(2, 8),
    r_scale=st.sampled_from([1e-2, 1.0, 1e3]),
)
def test_recursions_match_oracle_random_shapes(
    seed: int, n_latent: int, n_obs: int, n_time: int, r_scale: float
) -> None:
    """Hypothesis over dims (n_latent 1-3, n_obs 1-3, T 2-8) and R scale."""
    model = _simulate_lgssm(
        np.random.default_rng(seed), n_latent, n_obs, n_time, r_scale=r_scale
    )
    _assert_matches_oracle(model, RTOL)


@pytest.mark.parametrize(
    ("n_latent", "n_obs", "n_time", "r_scale", "rank_deficient_h", "rtol"),
    [
        pytest.param(2, 3, 6, 1.0, False, RTOL, id="n_obs>n_latent"),
        pytest.param(3, 3, 5, 1.0, True, RTOL, id="rank_deficient_H"),
        pytest.param(3, 1, 6, 1e-6, False, RTOL, id="small_R"),
        pytest.param(
            2, 3, 6, 1e-6, False, RTOL_ILL_CONDITIONED, id="small_R_n_obs>n_latent"
        ),
        pytest.param(2, 2, 6, 1e3, False, RTOL, id="large_R"),
    ],
)
def test_recursions_match_oracle_degenerate(
    n_latent: int,
    n_obs: int,
    n_time: int,
    r_scale: float,
    rank_deficient_h: bool,
    rtol: float,
) -> None:
    """Degenerate observation models. With n_obs > n_latent and R ~ 1e-6 the
    innovation covariance has condition number ~1e7, so agreement degrades
    to ~2e-8 (still ~1e-5 posterior standard deviations)."""
    model = _simulate_lgssm(
        np.random.default_rng(7),
        n_latent,
        n_obs,
        n_time,
        r_scale=r_scale,
        rank_deficient_h=rank_deficient_h,
    )
    if rank_deficient_h:
        assert np.linalg.matrix_rank(model["measurement_matrix"]) == 1
    _assert_matches_oracle(model, rtol)


@pytest.mark.parametrize("scale", [1e-3, 1e-5])
def test_recursions_are_unit_equivariant(scale: float) -> None:
    """Rescaling x and y (covariances ~ scale**2) must not change the answer
    in posterior units. Regression: the gain solves used psd_solve's absolute
    1e-9 Cholesky shift, which at scale 1e-5 (covariances ~1e-10, e.g. volts)
    put posterior means >1 posterior standard deviation off and the
    covariances ~100% off; now the shift is scale-relative."""
    model = _simulate_lgssm(np.random.default_rng(11), 2, 2, 6, scale=scale)
    # guard: this is the regime where an absolute 1e-9 shift dominates.
    oracle = _oracle(model)
    assert np.max(np.abs(oracle.smoothed_cov)) < 1e2 * scale**2
    assert np.min(np.linalg.eigvalsh(oracle.filtered_cov)) < 1e-4 * 1e-2
    _assert_matches_oracle(model, RTOL)
    s_mean = np.asarray(kalman_smoother(*_args(model))[0])
    sd = np.sqrt(np.einsum("tii->ti", oracle.smoothed_cov))
    assert np.max(np.abs(s_mean - oracle.smoothed_mean) / sd) < 1e-6


def test_parallel_smoother_matches_oracle() -> None:
    model = _simulate_lgssm(np.random.default_rng(3), 3, 2, 7)
    oracle = _oracle(model)
    f_mean, f_cov, _ = kalman_filter(*_args(model))
    p_mean, p_cov, p_cross = parallel_kalman_smoother(
        f_mean, f_cov, model["transition_matrix"], model["process_cov"]
    )
    atol = RTOL * _cov_scale(model)
    np.testing.assert_allclose(
        p_mean, oracle.smoothed_mean, rtol=RTOL, atol=RTOL * _mean_scale(oracle)
    )
    np.testing.assert_allclose(p_cov, oracle.smoothed_cov, rtol=RTOL, atol=atol)
    np.testing.assert_allclose(p_cross, oracle.smoothed_cross_cov, rtol=RTOL, atol=atol)


# ---------------------------------------------------------------------------
# EM M-step: the returned parameters maximise the exact auxiliary function
# ---------------------------------------------------------------------------

_PARAM_NAMES = (
    "init_mean",
    "init_cov",
    "transition_matrix",
    "process_cov",
    "measurement_matrix",
    "measurement_cov",
)
_SYMMETRIC = {"init_cov", "process_cov", "measurement_cov"}


def _q_function(oracle, obs: np.ndarray, params: dict, include_x0: bool) -> float:
    return lgssm_expected_complete_log_likelihood(
        oracle.joint_mean,
        oracle.joint_cov,
        obs,
        params["init_mean"],
        params["init_cov"],
        params["transition_matrix"],
        params["process_cov"],
        params["measurement_matrix"],
        params["measurement_cov"],
        include_x0=include_x0,
    )


def _fd_gradient(fun, params: dict, names=_PARAM_NAMES, step: float = 1e-5) -> dict:
    """Central finite differences of ``fun(params)`` w.r.t. each entry of the
    named parameters; symmetric matrices are perturbed symmetrically (the
    derivative along E_ij + E_ji)."""
    grads = {}
    for name in names:
        value = np.asarray(params[name], dtype=float)
        grad = np.zeros_like(value)
        for idx in np.ndindex(value.shape):
            if name in _SYMMETRIC and idx[0] > idx[1]:
                continue
            direction = np.zeros_like(value)
            direction[idx] = 1.0
            if name in _SYMMETRIC:
                direction[idx[::-1]] = 1.0
            h = step * max(1.0, float(np.max(np.abs(value))))
            plus = dict(params, **{name: value + h * direction})
            minus = dict(params, **{name: value - h * direction})
            grad[idx] = (fun(plus) - fun(minus)) / (2 * h)
        grads[name] = grad
    return grads


def _random_params(rng: np.random.Generator, n: int, m: int) -> dict:
    return {
        "init_mean": rng.normal(size=n),
        "init_cov": random_spd_matrix(rng, n),
        "transition_matrix": random_stable_matrix(rng, n),
        "process_cov": random_spd_matrix(rng, n),
        "measurement_matrix": rng.normal(size=(m, n)),
        "measurement_cov": random_spd_matrix(rng, m),
    }


def _e_step_and_m_step(model: dict, use_prior: bool) -> tuple:
    s_mean, s_cov, s_cross, _ = kalman_smoother(*_args(model))
    prior = (
        InitialStatePrior(
            *(
                jnp.asarray(model[k])
                for k in ("init_mean", "init_cov", "transition_matrix", "process_cov")
            )
        )
        if use_prior
        else None
    )
    A, H, Q, R, m0, P0 = kalman_maximization_step(
        jnp.asarray(model["obs"]), s_mean, s_cov, s_cross, prior
    )
    return {
        "init_mean": np.asarray(m0),
        "init_cov": np.asarray(P0),
        "transition_matrix": np.asarray(A),
        "process_cov": np.asarray(Q),
        "measurement_matrix": np.asarray(H),
        "measurement_cov": np.asarray(R),
    }


class TestKalmanMStepMaximisesExactQ:
    """``kalman_maximization_step`` against the exact EM auxiliary function
    ``Q(theta) = E[log p(x_{0:T}, y_{1:T} | theta)]``, with the expectation
    under the dense-oracle posterior (independent of the smoother whose
    statistics the M-step consumes)."""

    @settings(max_examples=6, deadline=None, derandomize=True)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_gradient_vanishes_and_q_increases(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        model = _simulate_lgssm(rng, 2, 2, 6)
        oracle = _oracle(model)
        obs = model["obs"]
        new = _e_step_and_m_step(model, use_prior=True)

        def q(p):
            return _q_function(oracle, obs, p, include_x0=True)

        grad_new = _fd_gradient(q, new)
        old = _random_params(rng, 2, 2)
        grad_old = _fd_gradient(q, old)
        g_old = max(np.max(np.abs(g)) for g in grad_old.values())
        assert g_old > 1e-2  # guard: a random theta is far from stationary
        for name, g in grad_new.items():
            np.testing.assert_allclose(g, 0.0, atol=1e-6, err_msg=name)
        q_new = q(new)
        assert q_new >= q(old)
        assert q_new >= q({k: model[k] for k in _PARAM_NAMES})  # the E-step theta
        # local check: random small perturbations never increase Q
        for _ in range(5):
            name = _PARAM_NAMES[rng.integers(len(_PARAM_NAMES))]
            d = rng.normal(size=np.shape(new[name])) * 1e-3
            if name in _SYMMETRIC:
                d = d + d.T
            assert q(dict(new, **{name: new[name] + d})) <= q_new + 1e-12

    def test_x0_transition_is_needed_for_exactness(self) -> None:
        """With ``initial_state_prior`` the x_0 -> x_1 transition enters the A
        and Q statistics and the update is stationary for the filter's Q;
        the legacy (no-prior) update is stationary only for the model with
        the prior on x_1, and is not for the filter's model."""
        model = _simulate_lgssm(np.random.default_rng(5), 2, 2, 5)
        oracle = _oracle(model)
        obs = model["obs"]

        def q_full(p):
            return _q_function(oracle, obs, p, include_x0=True)

        def q_x1(p):
            return _q_function(oracle, obs, p, include_x0=False)

        dyn = ("transition_matrix", "process_cov")
        new = _e_step_and_m_step(model, use_prior=True)
        legacy = _e_step_and_m_step(model, use_prior=False)
        for name, g in _fd_gradient(q_full, new, names=dyn).items():
            np.testing.assert_allclose(g, 0.0, atol=1e-6, err_msg=name)
        for name, g in _fd_gradient(q_x1, legacy).items():
            np.testing.assert_allclose(g, 0.0, atol=1e-6, err_msg=name)
        # guard: the legacy A/Q are not stationary for the filter's model
        legacy_grad = _fd_gradient(q_full, legacy, names=dyn)
        assert max(np.max(np.abs(g)) for g in legacy_grad.values()) > 1e-2
        assert q_full(new) > q_full(legacy) + 1e-3

    def test_full_em_matches_dense_oracle_q_monotonically(self) -> None:
        """Three EM iterations with the prior: the exact log-likelihood from
        the dense oracle never decreases, and each M-step output is the
        maximiser of that iteration's exact Q (gradient check on A, Q)."""
        model = _simulate_lgssm(np.random.default_rng(9), 2, 2, 8)
        params = {k: model[k] for k in _PARAM_NAMES}
        lls = []
        for _ in range(3):
            current = dict(params, obs=model["obs"])
            oracle = _oracle(current)
            lls.append(oracle.log_likelihood)
            new = _e_step_and_m_step(current, use_prior=True)
            for name, g in _fd_gradient(
                lambda p, o=oracle: _q_function(o, model["obs"], p, include_x0=True),
                new,
                names=("transition_matrix", "process_cov"),
            ).items():
                np.testing.assert_allclose(g, 0.0, atol=1e-6, err_msg=name)
            params = new
        assert np.all(np.diff(lls) > 0), lls


# ---------------------------------------------------------------------------
# Independent library cross-check (optional dependency, not in the lock file)
# ---------------------------------------------------------------------------


def test_matches_dynamax_lgssm() -> None:
    """Filter, smoother, lag-one cross-covariance and log-likelihood against
    dynamax's ``lgssm_smoother``. dynamax puts its prior on x_1, so it gets
    the predicted prior ``N(A m_0, A P_0 A^T + Q)``; its
    ``smoothed_cross_covariances`` are second moments ``E[x_t x_{t+1}^T]``."""
    inference = pytest.importorskip("dynamax.linear_gaussian_ssm.inference")
    model = _simulate_lgssm(np.random.default_rng(13), 3, 2, 20)
    m0, P0, obs, A, Q, H, R = _args(model)
    n, m = A.shape[0], H.shape[0]
    params = inference.ParamsLGSSM(
        initial=inference.ParamsLGSSMInitial(mean=A @ m0, cov=A @ P0 @ A.T + Q),
        dynamics=inference.ParamsLGSSMDynamics(
            weights=A,
            bias=jnp.zeros(n),
            input_weights=jnp.zeros((n, 0)),
            cov=Q,
        ),
        emissions=inference.ParamsLGSSMEmissions(
            weights=H,
            bias=jnp.zeros(m),
            input_weights=jnp.zeros((m, 0)),
            cov=R,
        ),
    )
    post = inference.lgssm_smoother(params, obs)
    f_mean, f_cov, f_ll = kalman_filter(m0, P0, obs, A, Q, H, R)
    s_mean, s_cov, s_cross, s_ll = kalman_smoother(m0, P0, obs, A, Q, H, R)
    # dynamax's own psd_solve adds a small absolute diagonal shift, so agree
    # to ~1e-8 rather than round-off.
    rtol, atol = 1e-7, 1e-9
    np.testing.assert_allclose(f_mean, post.filtered_means, rtol=rtol, atol=atol)
    np.testing.assert_allclose(f_cov, post.filtered_covariances, rtol=rtol, atol=atol)
    np.testing.assert_allclose(s_mean, post.smoothed_means, rtol=rtol, atol=atol)
    np.testing.assert_allclose(s_cov, post.smoothed_covariances, rtol=rtol, atol=atol)
    dyn_cross = np.asarray(post.smoothed_cross_covariances) - np.einsum(
        "ta,tb->tab",
        np.asarray(post.smoothed_means[:-1]),
        np.asarray(post.smoothed_means[1:]),
    )
    np.testing.assert_allclose(s_cross, dyn_cross, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(f_ll, post.marginal_loglik, rtol=1e-10)
    np.testing.assert_allclose(s_ll, post.marginal_loglik, rtol=1e-10)
