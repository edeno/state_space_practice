# ruff: noqa: E402
"""Likelihood identities every filter / smoother must satisfy exactly.

These are cheap self-consistency checks, independent of any oracle: each
filter's reported marginal log-likelihood is recomputed from the filter's
*own* outputs with a NumPy formula written out here, and the smoothers'
outputs are checked against the structural identities of a Gaussian / HMM
posterior. All of them hold to round-off for a correct implementation --
whether the model is exact (Kalman, HMM) or approximate (Laplace-EKF, GPB),
because they test what the code claims to compute, not how close that is to
the truth:

(a) ``log p(y_{1:T}) = sum_t log p(y_t | y_{1:t-1})`` with each term formed
    from the filter's predictive moments (Gaussian models: the Gaussian
    predictive density; Laplace models: the per-step Laplace evidence
    ``log p(y_t | x*) - 0.5 d' P_pred^{-1} d - 0.5 log|P_pred| + 0.5
    log|P_post|`` at the filter's mode ``x*`` and posterior covariance;
    switching models: the mixture over state pairs);
(b) the smoother equals the filter at the last time step (bit-for-bit);
(c) the smoother's lag-one cross-covariance is the RTS one,
    ``Cov(x_t, x_{t+1} | y) = J_t P_{t+1|T}``, the joint covariance of
    ``(x_t, x_{t+1})`` is PSD, and its Schur complement is the backward
    conditional covariance ``P_{t|t} - J_t P_{t+1|t} J_t'``;
(d) discrete smoothed probabilities sum to one and the two-slice marginals
    sum to the one-slice marginals;
(e) the ELBO identity ``log p(y) = E_q[log p(x, y)] + H(q)`` at the E-step
    parameters, with q the (exact) smoother posterior;

plus the chain rule ``log p(y_{1:T}) = log p(y_{1:t}) + log p(y_{t+1:T} |
y_{1:t})`` by restarting each filter from its own filtered moments at t.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaln, log_softmax, logsumexp

from state_space_practice.contingency_belief import (
    centered_softmax,
    contingency_belief_filter,
    contingency_belief_smoother,
)
from state_space_practice.covariate_choice import covariate_choice_filter
from state_space_practice.kalman import (
    InitialStatePrior,
    kalman_filter,
    kalman_smoother,
    smooth_initial_state_with_cross_cov,
)
from state_space_practice.multinomial_choice import (
    multinomial_choice_filter,
    multinomial_choice_smoother,
)
from state_space_practice.place_field_model import PlaceFieldModel
from state_space_practice.point_process_kalman import (
    log_conditional_intensity,
    stochastic_point_process_filter,
    stochastic_point_process_smoother,
)
from state_space_practice.position_decoder import (
    PlaceFieldRateMaps,
    _position_decoder_filter_with_predictions,
    position_decoder_smoother,
)
from state_space_practice.smith_learning_algorithm import (
    SmithLearningModel,
    smith_learning_filter,
    smith_learning_smoother,
)
from state_space_practice.switching_choice import (
    switching_choice_filter,
    switching_choice_smoother,
)
from state_space_practice.switching_kalman import (
    compute_elbo,
    switching_kalman_filter,
    switching_kalman_smoother,
    switching_kalman_smoother_gpb2,
)
from state_space_practice.switching_point_process import (
    SpikeObsParams,
    _linear_log_intensity,
    switching_point_process_filter,
)
from state_space_practice.tests.oracles import (
    gaussian_logpdf,
    random_spd_matrix,
    random_stable_matrix,
)

RTOL = 1e-10


def _logdet(m: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(m)
    assert sign > 0
    return float(value)


def _assert_rts_cross_cov(
    filt_mean, filt_cov, smooth_cov, cross_cov, A, Q, pred_cov=None, rtol=1e-9
):
    """(c): C_t = J_t P_{t+1|T}, joint (x_t, x_{t+1}) PSD, Schur complement =
    backward conditional covariance. Returns the smallest joint eigenvalue
    relative to the joint scale (a guard that the joint is informative)."""
    n_time, n = filt_mean.shape
    worst = np.inf
    for t in range(n_time - 1):
        P_pred = (
            A @ filt_cov[t] @ A.T + Q
            if pred_cov is None
            else np.asarray(pred_cov[t + 1])
        )
        J = filt_cov[t] @ A.T @ np.linalg.inv(P_pred)
        scale = np.max(np.abs(smooth_cov[t]))
        np.testing.assert_allclose(
            cross_cov[t], J @ smooth_cov[t + 1], rtol=rtol, atol=rtol * scale
        )
        joint = np.block(
            [[smooth_cov[t], cross_cov[t]], [cross_cov[t].T, smooth_cov[t + 1]]]
        )
        eig = np.linalg.eigvalsh(0.5 * (joint + joint.T))
        assert eig[0] > -1e-12 * np.max(np.abs(joint)), (t, eig)
        worst = min(worst, eig[0] / eig[-1])
        schur = smooth_cov[t] - cross_cov[t] @ np.linalg.solve(
            smooth_cov[t + 1], cross_cov[t].T
        )
        backward = filt_cov[t] - J @ P_pred @ J.T
        np.testing.assert_allclose(schur, backward, rtol=1e-8, atol=1e-8 * scale)
    return worst


# ---------------------------------------------------------------------------
# Kalman filter / RTS smoother
# ---------------------------------------------------------------------------


def _lgssm(seed: int, n: int = 2, m: int = 3, n_time: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    p = {
        "init_mean": rng.normal(size=n),
        "init_cov": random_spd_matrix(rng, n),
        "transition_matrix": random_stable_matrix(rng, n),
        "process_cov": random_spd_matrix(rng, n, scale=0.4),
        "measurement_matrix": rng.normal(size=(m, n)),
        "measurement_cov": random_spd_matrix(rng, m, scale=0.5),
    }
    x = rng.multivariate_normal(p["init_mean"], p["init_cov"])
    ys = []
    for _ in range(n_time):
        x = p["transition_matrix"] @ x + rng.multivariate_normal(
            np.zeros(n), p["process_cov"]
        )
        ys.append(
            p["measurement_matrix"] @ x
            + rng.multivariate_normal(np.zeros(m), p["measurement_cov"])
        )
    p["obs"] = np.stack(ys)
    return p


def _kalman_args(p: dict, obs=None, init=None) -> tuple:
    m0, P0 = (p["init_mean"], p["init_cov"]) if init is None else init
    return tuple(
        jnp.asarray(v)
        for v in (
            m0,
            P0,
            p["obs"] if obs is None else obs,
            p["transition_matrix"],
            p["process_cov"],
            p["measurement_matrix"],
            p["measurement_cov"],
        )
    )


@pytest.fixture(scope="module")
def lgssm():
    p = _lgssm(0)
    f_mean, f_cov, f_ll = (np.asarray(v) for v in kalman_filter(*_kalman_args(p)))
    s_mean, s_cov, s_cross, s_ll = (
        np.asarray(v) for v in kalman_smoother(*_kalman_args(p))
    )
    return p, dict(
        f_mean=f_mean,
        f_cov=f_cov,
        f_ll=float(f_ll),
        s_mean=s_mean,
        s_cov=s_cov,
        s_cross=s_cross,
        s_ll=float(s_ll),
    )


class TestKalmanIdentities:
    def test_ll_is_sum_of_predictive_log_densities(self, lgssm) -> None:
        p, out = lgssm
        A, Q = p["transition_matrix"], p["process_cov"]
        H, R = p["measurement_matrix"], p["measurement_cov"]
        mean, cov = p["init_mean"], p["init_cov"]
        terms = []
        for t, y in enumerate(p["obs"]):
            m_pred, P_pred = A @ mean, A @ cov @ A.T + Q
            terms.append(gaussian_logpdf(y, H @ m_pred, H @ P_pred @ H.T + R))
            mean, cov = out["f_mean"][t], out["f_cov"][t]
        np.testing.assert_allclose(out["f_ll"], np.sum(terms), rtol=RTOL)
        np.testing.assert_allclose(out["s_ll"], out["f_ll"], rtol=0, atol=0)
        # guard: the terms vary (not a constant per step)
        assert np.ptp(terms) > 0.5

    def test_smoother_equals_filter_at_last_step(self, lgssm) -> None:
        _, out = lgssm
        np.testing.assert_array_equal(out["s_mean"][-1], out["f_mean"][-1])
        np.testing.assert_array_equal(out["s_cov"][-1], out["f_cov"][-1])
        assert not np.allclose(out["s_mean"][0], out["f_mean"][0])  # guard

    def test_cross_covariance_is_rts(self, lgssm) -> None:
        p, out = lgssm
        worst = _assert_rts_cross_cov(
            out["f_mean"],
            out["f_cov"],
            out["s_cov"],
            out["s_cross"],
            p["transition_matrix"],
            p["process_cov"],
        )
        assert worst > 1e-4  # guard: the joint is nondegenerate

    def test_elbo_identity(self, lgssm) -> None:
        """``E_q[log p(x_{0:T}, y)] + H(q) = log p(y)`` with q the smoother's
        Gaussian trajectory posterior (x_0 via the library's initial-state
        smoother). The trajectory entropy uses the Markov factorisation
        ``H(x_T) + sum_t H(x_t | x_{t+1})``, so every smoothed marginal and
        lag-one cross-covariance enters."""
        p, out = lgssm
        A, Q = p["transition_matrix"], p["process_cov"]
        H, R = p["measurement_matrix"], p["measurement_cov"]
        m0, P0 = p["init_mean"], p["init_cov"]
        prior = InitialStatePrior(*(jnp.asarray(v) for v in (m0, P0, A, Q)))
        x0m, x0P, C01 = (
            np.asarray(v)
            for v in smooth_initial_state_with_cross_cov(
                prior, out["s_mean"][0], out["s_cov"][0]
            )
        )
        means = np.concatenate([x0m[None], out["s_mean"]])
        covs = np.concatenate([x0P[None], out["s_cov"]])
        cross = np.concatenate([C01[None], out["s_cross"]])
        n = means.shape[1]

        def e_log_normal(second, cov):
            d = cov.shape[0]
            return -0.5 * (
                d * np.log(2 * np.pi)
                + _logdet(cov)
                + np.trace(np.linalg.solve(cov, second))
            )

        r0 = means[0] - m0
        q_val = e_log_normal(covs[0] + np.outer(r0, r0), P0)
        for t in range(1, means.shape[0]):
            exx = covs[t] + np.outer(means[t], means[t])
            epp = covs[t - 1] + np.outer(means[t - 1], means[t - 1])
            epx = cross[t - 1] + np.outer(means[t - 1], means[t])
            q_val += e_log_normal(exx - A @ epx - epx.T @ A.T + A @ epp @ A.T, Q)
            ry = p["obs"][t - 1] - H @ means[t]
            q_val += e_log_normal(np.outer(ry, ry) + H @ covs[t] @ H.T, R)

        def gauss_entropy(cov):
            return 0.5 * (n * (1 + np.log(2 * np.pi)) + _logdet(cov))

        entropy = gauss_entropy(covs[-1])
        for t in range(means.shape[0] - 1):
            cond = covs[t] - cross[t] @ np.linalg.solve(covs[t + 1], cross[t].T)
            entropy += gauss_entropy(cond)
        np.testing.assert_allclose(q_val + entropy, out["f_ll"], rtol=1e-9)
        # guard: the identity needs the cross terms (marginal entropies alone
        # over-count the trajectory entropy)
        naive = sum(gauss_entropy(c) for c in covs)
        assert naive - entropy > 1.0

    @pytest.mark.parametrize("split", [1, 5, 11])
    def test_chain_rule_by_restart(self, lgssm, split: int) -> None:
        p, out = lgssm
        head = kalman_filter(*_kalman_args(p, obs=p["obs"][:split]))
        tail = kalman_filter(
            *_kalman_args(
                p,
                obs=p["obs"][split:],
                init=(out["f_mean"][split - 1], out["f_cov"][split - 1]),
            )
        )
        np.testing.assert_allclose(
            float(head[2]) + float(tail[2]), out["f_ll"], rtol=RTOL
        )
        np.testing.assert_allclose(
            np.asarray(tail[0]), out["f_mean"][split:], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(tail[1]), out["f_cov"][split:], rtol=1e-12, atol=1e-12
        )


# ---------------------------------------------------------------------------
# Switching Kalman filter / GPB smoothers
# ---------------------------------------------------------------------------

_SWITCHING_KEYS = (
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


def _switching_model(seed: int, n=2, m=2, K=2, n_time=10, identical=False) -> dict:
    rng = np.random.default_rng(seed)

    def per_state(draw):
        if identical:
            v = draw()
            return np.stack([v] * K, axis=-1)
        return np.stack([draw() for _ in range(K)], axis=-1)

    p = {
        "init_state_cond_mean": per_state(lambda: rng.normal(size=n)),
        "init_state_cond_cov": per_state(lambda: random_spd_matrix(rng, n)),
        "init_discrete_state_prob": rng.dirichlet(3.0 * np.ones(K)),
        "discrete_transition_matrix": 0.6 * np.eye(K)
        + 0.4 * rng.dirichlet(2.0 * np.ones(K), size=K),
        "continuous_transition_matrix": per_state(lambda: random_stable_matrix(rng, n)),
        "process_cov": per_state(lambda: random_spd_matrix(rng, n, scale=0.4)),
        "measurement_matrix": per_state(lambda: rng.normal(size=(m, n))),
        "measurement_cov": per_state(lambda: random_spd_matrix(rng, m, scale=0.4)),
    }
    s = rng.choice(K, p=p["init_discrete_state_prob"])
    x = rng.multivariate_normal(
        p["init_state_cond_mean"][:, s], p["init_state_cond_cov"][..., s]
    )
    ys = []
    for t in range(n_time):
        if t > 0:
            s = rng.choice(K, p=p["discrete_transition_matrix"][s])
            x = p["continuous_transition_matrix"][..., s] @ x + rng.multivariate_normal(
                np.zeros(n), p["process_cov"][..., s]
            )
        ys.append(
            p["measurement_matrix"][..., s] @ x
            + rng.multivariate_normal(np.zeros(m), p["measurement_cov"][..., s])
        )
    p["obs"] = np.stack(ys)
    return p


def _run_switching(p: dict) -> dict:
    args = [jnp.asarray(p[k]) for k in _SWITCHING_KEYS]
    fm, fc, fp, pfm, pfc, pfp, ll = switching_kalman_filter(*args)
    A = jnp.asarray(p["continuous_transition_matrix"])
    Q = jnp.asarray(p["process_cov"])
    Z = jnp.asarray(p["discrete_transition_matrix"])
    g1 = switching_kalman_smoother(fm, fc, fp, Q, A, Z)
    g2 = switching_kalman_smoother_gpb2(fm, fc, fp, pfm, pfc, pfp, Q, A)
    return {
        "filter": tuple(np.asarray(v) for v in (fm, fc, fp, pfm, pfc, pfp)),
        "ll": float(ll),
        "gpb1": tuple(np.asarray(v) for v in g1),
        "gpb2": tuple(np.asarray(v) for v in g2),
    }


@pytest.fixture(scope="module")
def switching():
    p = _switching_model(1)
    return p, _run_switching(p)


def _switching_predictive_terms(p: dict, fm, fc, fp) -> np.ndarray:
    """log p(y_t | y_{1:t-1}) of the GPB2 (Kim) filter from its own collapsed
    state-conditional moments: a mixture over (S_{t-1}=i, S_t=j) of Gaussian
    predictives (t = 1: over S_1 with the prior on x_1)."""
    A, Q = p["continuous_transition_matrix"], p["process_cov"]
    H, R = p["measurement_matrix"], p["measurement_cov"]
    Z, K = p["discrete_transition_matrix"], p["discrete_transition_matrix"].shape[0]
    terms = []
    for t, y in enumerate(p["obs"]):
        logs = []
        for j in range(K):
            if t == 0:
                m, P = p["init_state_cond_mean"][:, j], p["init_state_cond_cov"][..., j]
                logs.append(
                    np.log(p["init_discrete_state_prob"][j])
                    + gaussian_logpdf(
                        y, H[..., j] @ m, H[..., j] @ P @ H[..., j].T + R[..., j]
                    )
                )
                continue
            for i in range(K):
                m = A[..., j] @ fm[t - 1][:, i]
                P = A[..., j] @ fc[t - 1][..., i] @ A[..., j].T + Q[..., j]
                logs.append(
                    np.log(fp[t - 1][i] * Z[i, j])
                    + gaussian_logpdf(
                        y, H[..., j] @ m, H[..., j] @ P @ H[..., j].T + R[..., j]
                    )
                )
        terms.append(logsumexp(logs))
    return np.array(terms)


@pytest.mark.slow  # compile-dominated (>3 s)
class TestSwitchingKalmanIdentities:
    def test_ll_is_sum_of_mixture_predictive_log_densities(self, switching) -> None:
        p, out = switching
        fm, fc, fp = out["filter"][:3]
        terms = _switching_predictive_terms(p, fm, fc, fp)
        np.testing.assert_allclose(out["ll"], terms.sum(), rtol=RTOL)
        assert np.ptp(terms) > 0.5  # guard

    def test_state_conditional_filter_is_collapse_of_pair_conditionals(
        self, switching
    ) -> None:
        """M_t(j) = sum_i P(i, j), and the state-conditional moments are the
        moment-matched collapse of the pair-conditional ones with weights
        P(S_{t-1}=i | S_t=j, y_{1:t})."""
        _, out = switching
        fm, fc, fp, pfm, pfc, pfp = out["filter"]
        np.testing.assert_allclose(fp.sum(axis=1), 1.0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(pfp.sum(axis=1), fp, rtol=1e-12, atol=1e-15)
        for t in range(1, fm.shape[0]):
            w = pfp[t] / fp[t][None, :]  # (i, j)
            for j in range(fp.shape[1]):
                mean = pfm[t][:, :, j] @ w[:, j]
                d = pfm[t][:, :, j] - mean[:, None]
                cov = (
                    np.einsum("abi,i->ab", pfc[t][..., j], w[:, j])
                    + (d * w[:, j]) @ d.T
                )
                np.testing.assert_allclose(fm[t][:, j], mean, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(fc[t][..., j], cov, rtol=1e-9, atol=1e-12)
        # guard: the collapse mixes distinct pair means
        assert np.max(np.abs(pfm[-1][:, 0, 0] - pfm[-1][:, 1, 0])) > 1e-2

    @pytest.mark.parametrize("smoother", ["gpb1", "gpb2"])
    def test_smoother_discrete_probabilities(self, switching, smoother) -> None:
        _, out = switching
        g = out[smoother]
        prob, joint = g[2], g[3]
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(joint.sum(axis=2), prob[:-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(joint.sum(axis=1), prob[1:], rtol=0, atol=1e-12)
        assert np.all(joint >= 0.0)
        assert np.ptp(prob[:, 0]) > 0.1  # guard: informative

    @pytest.mark.parametrize("smoother", ["gpb1", "gpb2"])
    def test_smoother_equals_filter_at_last_step(self, switching, smoother) -> None:
        _, out = switching
        fm, fc, fp = out["filter"][:3]
        g = out[smoother]
        np.testing.assert_allclose(g[2][-1], fp[-1], rtol=0, atol=1e-15)
        np.testing.assert_allclose(g[5][-1], fm[-1], rtol=0, atol=1e-14)
        np.testing.assert_allclose(g[6][-1], fc[-1], rtol=0, atol=1e-14)
        # the marginal is the collapse of the state-conditionals
        for t in range(fm.shape[0]):
            np.testing.assert_allclose(
                g[0][t], g[5][t] @ g[2][t], rtol=1e-10, atol=1e-12
            )
        assert not np.allclose(g[2][0], fp[0], atol=1e-3)  # guard

    @pytest.mark.parametrize("smoother", ["gpb1", "gpb2"])
    def test_marginal_joint_covariance_is_psd(self, switching, smoother) -> None:
        _, out = switching
        g = out[smoother]
        cov, cross = g[1], g[4]
        for t in range(cov.shape[0] - 1):
            joint = np.block([[cov[t], cross[t]], [cross[t].T, cov[t + 1]]])
            eig = np.linalg.eigvalsh(0.5 * (joint + joint.T))
            assert eig[0] > -1e-12 * eig[-1], (t, eig)
        assert np.max(np.abs(cross)) > 1e-2  # guard

    @pytest.mark.parametrize("smoother", ["gpb1", "gpb2"])
    def test_elbo_equals_ll_in_exact_regime(self, smoother) -> None:
        """Identical per-state parameters: the GPB posterior is exact and
        factorises as q(s) q(x) with q(s) the Markov prior, so the library's
        ELBO (``compute_elbo`` = expected complete-data LL + Markov trajectory
        entropy) equals the filter's log-likelihood."""
        p = _switching_model(2, identical=True, n_time=8)
        out = _run_switching(p)
        g = out[smoother]
        kwargs = {}
        if smoother == "gpb2":
            kwargs = dict(
                pair_cond_smoother_means=jnp.asarray(g[8]),
                pair_cond_smoother_covs=jnp.asarray(g[9]),
                next_pair_cond_smoother_means=jnp.asarray(g[10]),
            )
        else:
            kwargs = dict(pair_cond_smoother_means=jnp.asarray(g[8]))
        elbo = compute_elbo(
            obs=jnp.asarray(p["obs"]),
            state_cond_smoother_means=jnp.asarray(g[5]),
            state_cond_smoother_covs=jnp.asarray(g[6]),
            smoother_discrete_state_prob=jnp.asarray(g[2]),
            smoother_joint_discrete_state_prob=jnp.asarray(g[3]),
            pair_cond_smoother_cross_cov=jnp.asarray(g[7]),
            init_state_cond_mean=jnp.asarray(p["init_state_cond_mean"]),
            init_state_cond_cov=jnp.asarray(p["init_state_cond_cov"]),
            init_discrete_state_prob=jnp.asarray(p["init_discrete_state_prob"]),
            continuous_transition_matrix=jnp.asarray(p["continuous_transition_matrix"]),
            process_cov=jnp.asarray(p["process_cov"]),
            measurement_matrix=jnp.asarray(p["measurement_matrix"]),
            measurement_cov=jnp.asarray(p["measurement_cov"]),
            discrete_transition_matrix=jnp.asarray(p["discrete_transition_matrix"]),
            **kwargs,
        )
        np.testing.assert_allclose(float(elbo), out["ll"], rtol=1e-8)

    def test_elbo_differs_from_ll_outside_exact_regime(self) -> None:
        """Guard for the test above: with distinct, ambiguous states the GPB
        posterior is approximate and the identity fails (0.16 nats here), so
        it is not an artefact of ``compute_elbo``."""
        p = _switching_model(4)
        out = _run_switching(p)
        g = out["gpb2"]
        elbo = compute_elbo(
            obs=jnp.asarray(p["obs"]),
            state_cond_smoother_means=jnp.asarray(g[5]),
            state_cond_smoother_covs=jnp.asarray(g[6]),
            smoother_discrete_state_prob=jnp.asarray(g[2]),
            smoother_joint_discrete_state_prob=jnp.asarray(g[3]),
            pair_cond_smoother_cross_cov=jnp.asarray(g[7]),
            init_state_cond_mean=jnp.asarray(p["init_state_cond_mean"]),
            init_state_cond_cov=jnp.asarray(p["init_state_cond_cov"]),
            init_discrete_state_prob=jnp.asarray(p["init_discrete_state_prob"]),
            continuous_transition_matrix=jnp.asarray(p["continuous_transition_matrix"]),
            process_cov=jnp.asarray(p["process_cov"]),
            measurement_matrix=jnp.asarray(p["measurement_matrix"]),
            measurement_cov=jnp.asarray(p["measurement_cov"]),
            discrete_transition_matrix=jnp.asarray(p["discrete_transition_matrix"]),
            pair_cond_smoother_means=jnp.asarray(g[8]),
            pair_cond_smoother_covs=jnp.asarray(g[9]),
            next_pair_cond_smoother_means=jnp.asarray(g[10]),
        )
        assert abs(out["ll"] - float(elbo)) > 1e-2

    def test_filter_is_causal(self, switching) -> None:
        """Chain rule for the switching filter. The filter places its prior on
        x_1 (measurement-only first step), so it cannot be restarted from its
        own collapsed moments at t (that would drop a prediction step); the
        chain rule is checked as causality instead: the filter on y_{1:t}
        reproduces the first t outputs, and log p(y_{1:t}) plus the
        predictive terms after t is the full log-likelihood."""
        p, out = switching
        split = 4
        head = dict(p, obs=p["obs"][:split])
        res = switching_kalman_filter(*[jnp.asarray(head[k]) for k in _SWITCHING_KEYS])
        for a, b in zip(res[:3], out["filter"][:3]):
            np.testing.assert_allclose(np.asarray(a), b[:split], rtol=1e-12, atol=1e-14)
        terms = _switching_predictive_terms(p, *out["filter"][:3])
        np.testing.assert_allclose(
            float(res[-1]) + terms[split:].sum(), out["ll"], rtol=RTOL
        )


# ---------------------------------------------------------------------------
# Laplace-EKF point-process filter / smoother
# ---------------------------------------------------------------------------


def _affine_log_rate(design_t, x):
    return design_t[:, 0] + design_t[:, 1:] @ x


def _pp_problem(seed: int, n_latent=2, n_neurons=4, n_time=25, rate=30.0, dt=0.02):
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0.85, 0.97, n_latent))
    A[0, -1] = 0.05
    Q = random_spd_matrix(rng, n_latent, scale=0.02)
    m0 = rng.normal(0, 0.3, n_latent)
    P0 = random_spd_matrix(rng, n_latent, scale=0.2)
    W = rng.normal(0, 0.8, (n_neurons, n_latent))
    b = np.log(rng.uniform(0.5, 1.5, n_neurons) * rate)
    design = np.tile(np.concatenate([b[:, None], W], 1), (n_time, 1, 1))
    x = rng.multivariate_normal(m0, P0)
    ys = []
    for _ in range(n_time):
        x = A @ x + rng.multivariate_normal(np.zeros(n_latent), Q)
        ys.append(rng.poisson(np.exp(b + W @ x) * dt))
    return dict(
        m0=m0, P0=P0, A=A, Q=Q, W=W, b=b, design=design, y=np.array(ys, float), dt=dt
    )


def _laplace_terms(y, log_rate_at_mode, dt, mode, post_cov, pred_mean, pred_cov):
    mu = np.exp(log_rate_at_mode) * dt
    logpmf = np.sum(y * np.log(mu) - mu - gammaln(y + 1.0))
    d = mode - pred_mean
    return (
        logpmf
        - 0.5 * d @ np.linalg.solve(pred_cov, d)
        - 0.5 * _logdet(pred_cov)
        + 0.5 * _logdet(post_cov)
    )


class TestPointProcessIdentities:
    @pytest.mark.parametrize("max_newton_iter", [1, 3])
    def test_ll_is_sum_of_laplace_evidence_terms(self, max_newton_iter) -> None:
        """Per-step Laplace evidence at the filter's own mode and posterior
        covariance, with the predictive moments from its previous output."""
        pr = _pp_problem(0)
        fm, fc, ll = (
            np.asarray(v)
            for v in stochastic_point_process_filter(
                pr["m0"],
                pr["P0"],
                pr["design"],
                pr["y"],
                pr["dt"],
                pr["A"],
                pr["Q"],
                _affine_log_rate,
                max_newton_iter=max_newton_iter,
            )
        )
        mean, cov = pr["m0"], pr["P0"]
        terms = []
        for t in range(pr["y"].shape[0]):
            pm, pP = pr["A"] @ mean, pr["A"] @ cov @ pr["A"].T + pr["Q"]
            terms.append(
                _laplace_terms(
                    pr["y"][t],
                    pr["b"] + pr["W"] @ fm[t],
                    pr["dt"],
                    fm[t],
                    fc[t],
                    pm,
                    pP,
                )
            )
            mean, cov = fm[t], fc[t]
        np.testing.assert_allclose(float(ll), np.sum(terms), rtol=1e-10)
        assert pr["y"].sum() > 20 and np.ptp(terms) > 1.0  # guard

    def test_plugin_ll_without_normalisation(self) -> None:
        pr = _pp_problem(1)
        fm, _, ll = stochastic_point_process_filter(
            pr["m0"],
            pr["P0"],
            pr["design"],
            pr["y"],
            pr["dt"],
            pr["A"],
            pr["Q"],
            _affine_log_rate,
            include_laplace_normalization=False,
        )
        mu = np.exp(pr["b"] + np.asarray(fm) @ pr["W"].T) * pr["dt"]
        plug = np.sum(pr["y"] * np.log(mu) - mu - gammaln(pr["y"] + 1.0))
        np.testing.assert_allclose(float(ll), plug, rtol=1e-10)

    @pytest.mark.slow  # compile-dominated (>3 s)
    def test_block_diagonal_path_ll_identity(self) -> None:
        """The block-diagonal fast path (PlaceFieldModel E-step): the total is
        the sum of per-step Laplace evidence over the dense state."""
        rng = np.random.default_rng(3)
        n_neurons, nb, n_time, dt = 3, 3, 30, 0.02
        Z = rng.uniform(0, 1, (n_time, nb))
        Z /= Z.sum(1, keepdims=True)
        n = n_neurons * nb
        m0 = np.log(20.0) + rng.normal(0, 0.5, n)
        P0 = np.diag(rng.uniform(0.02, 0.1, n))
        Q = np.diag(rng.uniform(1e-4, 1e-3, n))
        A = np.eye(n)
        y = rng.poisson(20.0 * dt, (n_time, n_neurons)).astype(float)
        fm, fc, ll = stochastic_point_process_filter(
            m0,
            P0,
            Z,
            y,
            dt,
            A,
            Q,
            log_conditional_intensity,
            block_n_neurons=n_neurons,
            block_size=nb,
        )
        fm, fc = np.asarray(fm), np.asarray(fc)
        mean, cov = m0, P0
        total = 0.0
        for t in range(n_time):
            pm, pP = mean, cov + Q
            log_rate = fm[t].reshape(n_neurons, nb) @ Z[t]
            total += _laplace_terms(y[t], log_rate, dt, fm[t], fc[t], pm, pP)
            mean, cov = fm[t], fc[t]
        np.testing.assert_allclose(float(ll), total, rtol=1e-10)
        # guard: the block path was taken (off-block covariance exactly zero)
        assert np.all(fc[:, :nb, nb:] == 0.0)

    def test_smoother_last_step_and_cross_covariance(self) -> None:
        pr = _pp_problem(2)
        sm, sc, scc, s_ll, fm, fc = (
            np.asarray(v)
            for v in stochastic_point_process_smoother(
                pr["m0"],
                pr["P0"],
                pr["design"],
                pr["y"],
                pr["dt"],
                pr["A"],
                pr["Q"],
                _affine_log_rate,
                return_filtered=True,
            )
        )
        np.testing.assert_array_equal(sm[-1], fm[-1])
        np.testing.assert_array_equal(sc[-1], fc[-1])
        worst = _assert_rts_cross_cov(fm, fc, sc, scc, pr["A"], pr["Q"])
        assert worst > 1e-4
        assert not np.allclose(sm[0], fm[0])  # guard

    @pytest.mark.parametrize(
        "max_newton_iter", [1, pytest.param(3, marks=pytest.mark.slow)]
    )
    def test_chain_rule_by_restart(self, max_newton_iter) -> None:
        pr = _pp_problem(4)
        split = 9

        def run(m0, P0, sl):
            return stochastic_point_process_filter(
                m0,
                P0,
                pr["design"][sl],
                pr["y"][sl],
                pr["dt"],
                pr["A"],
                pr["Q"],
                _affine_log_rate,
                max_newton_iter=max_newton_iter,
            )

        fm, fc, ll = run(pr["m0"], pr["P0"], slice(None))
        head = run(pr["m0"], pr["P0"], slice(None, split))
        tail = run(fm[split - 1], fc[split - 1], slice(split, None))
        np.testing.assert_allclose(
            float(head[2]) + float(tail[2]), float(ll), rtol=1e-12
        )
        np.testing.assert_allclose(tail[0], fm[split:], rtol=1e-12, atol=1e-13)


@pytest.mark.slow  # compile-dominated (>3 s)
class TestSwitchingPointProcessIdentities:
    """GPB2 switching point-process filter: per-step LL is the mixture over
    state pairs of the pair-conditional Laplace evidences (formed from the
    pair-conditional modes / covariances the filter returns)."""

    @staticmethod
    def _problem():
        rng = np.random.default_rng(5)
        n, n_neurons, K, n_time, dt = 2, 3, 2, 15, 0.02
        A = np.stack([0.95 * np.eye(n), 0.8 * np.eye(n)], -1)
        A[0, 1, 1] = 0.2
        Q = np.stack([0.02 * np.eye(n), 0.05 * np.eye(n)], -1)
        m0 = np.stack([np.zeros(n), 0.3 * np.ones(n)], -1)
        P0 = np.stack([0.3 * np.eye(n), 0.5 * np.eye(n)], -1)
        pi = np.array([0.6, 0.4])
        Zd = np.array([[0.9, 0.1], [0.2, 0.8]])
        b = np.log(np.stack([np.full(n_neurons, 30.0), np.full(n_neurons, 15.0)], -1))
        W = rng.normal(0, 0.7, (n_neurons, n, K))
        y = rng.poisson(1.0, (n_time, n_neurons)).astype(float)
        return dict(A=A, Q=Q, m0=m0, P0=P0, pi=pi, Z=Zd, b=b, W=W, y=y, dt=dt)

    def test_ll_is_mixture_of_pair_laplace_evidences(self) -> None:
        pr = self._problem()
        fm, fc, fp, pfm, pfc, pfp, ll = (
            np.asarray(v)
            for v in switching_point_process_filter(
                jnp.asarray(pr["m0"]),
                jnp.asarray(pr["P0"]),
                jnp.asarray(pr["pi"]),
                jnp.asarray(pr["y"]),
                jnp.asarray(pr["Z"]),
                jnp.asarray(pr["A"]),
                jnp.asarray(pr["Q"]),
                pr["dt"],
                _linear_log_intensity,
                SpikeObsParams(jnp.asarray(pr["b"]), jnp.asarray(pr["W"])),
            )
        )
        K = pr["pi"].size
        total = 0.0
        for t in range(pr["y"].shape[0]):
            logs = []
            for j in range(K):
                sources = [None] if t == 0 else range(K)
                for i in sources:
                    if t == 0:
                        pm, pP, lw = (
                            pr["m0"][:, j],
                            pr["P0"][..., j],
                            np.log(pr["pi"][j]),
                        )
                        mode, pc = pfm[0][:, 0, j], pfc[0][..., 0, j]
                    else:
                        Aj = pr["A"][..., j]
                        pm = Aj @ fm[t - 1][:, i]
                        pP = Aj @ fc[t - 1][..., i] @ Aj.T + pr["Q"][..., j]
                        lw = np.log(fp[t - 1][i] * pr["Z"][i, j])
                        mode, pc = pfm[t][:, i, j], pfc[t][..., i, j]
                    log_rate = pr["b"][:, j] + pr["W"][..., j] @ mode
                    logs.append(
                        lw
                        + _laplace_terms(
                            pr["y"][t], log_rate, pr["dt"], mode, pc, pm, pP
                        )
                    )
            total += logsumexp(logs)
        np.testing.assert_allclose(float(ll), total, rtol=1e-10)
        np.testing.assert_allclose(fp.sum(1), 1.0, atol=1e-14)
        np.testing.assert_allclose(pfp[1:].sum(1), fp[1:], rtol=1e-12, atol=1e-15)
        assert np.ptp(fp[:, 0]) > 0.05  # guard


# ---------------------------------------------------------------------------
# Behavioural models
# ---------------------------------------------------------------------------


def _softmax_laplace_term(choice, beta, offset, mode, post_cov, pred_mean, pred_cov):
    logits = beta * np.concatenate([[0.0], mode]) + offset
    d = mode - pred_mean
    return (
        log_softmax(logits)[choice]
        - 0.5 * d @ np.linalg.solve(pred_cov, d)
        - 0.5 * _logdet(pred_cov)
        + 0.5 * _logdet(post_cov)
    )


class TestChoiceModelIdentities:
    CHOICES = np.array([0, 2, 2, 1, 2, 0, 0, 2, 1, 2, 2, 2])

    def test_multinomial_ll_is_sum_of_laplace_evidence(self) -> None:
        q, beta = 0.1, 1.5
        f = multinomial_choice_filter(
            self.CHOICES, 3, process_noise=q, inverse_temperature=beta
        )
        fm, fc = np.asarray(f.filtered_values), np.asarray(f.filtered_covariances)
        pm, pc = np.asarray(f.predicted_values), np.asarray(f.predicted_covariances)
        # the reported predictions are the random-walk predictions of the
        # previous filtered moments
        np.testing.assert_array_equal(pm[1:], fm[:-1])
        np.testing.assert_allclose(pc[1:], fc[:-1] + q * np.eye(2), rtol=1e-15)
        terms = [
            _softmax_laplace_term(c, beta, 0.0, fm[t], fc[t], pm[t], pc[t])
            for t, c in enumerate(self.CHOICES)
        ]
        np.testing.assert_allclose(
            float(f.marginal_log_likelihood), sum(terms), rtol=1e-9
        )

    def test_covariate_ll_is_sum_of_laplace_evidence(self) -> None:
        rng = np.random.default_rng(0)
        T = self.CHOICES.size
        u, z = rng.normal(size=(T, 2)), rng.normal(size=(T, 2))
        B, Wobs = rng.normal(size=(2, 2)), rng.normal(size=(3, 2))
        q, beta, decay = 0.1, 1.5, 0.9
        f = covariate_choice_filter(
            self.CHOICES,
            3,
            covariates=u,
            input_gain=B,
            obs_covariates=z,
            obs_weights=Wobs,
            process_noise=q,
            inverse_temperature=beta,
            decay=decay,
        )
        fm, fc = np.asarray(f.filtered_values), np.asarray(f.filtered_covariances)
        pm, pc = np.asarray(f.predicted_values), np.asarray(f.predicted_covariances)
        np.testing.assert_allclose(pm[1:], decay * fm[:-1] + u[1:] @ B.T, rtol=1e-13)
        terms = [
            _softmax_laplace_term(c, beta, Wobs @ z[t], fm[t], fc[t], pm[t], pc[t])
            for t, c in enumerate(self.CHOICES)
        ]
        np.testing.assert_allclose(
            float(f.marginal_log_likelihood), sum(terms), rtol=1e-9
        )

    @pytest.mark.slow  # compile-dominated (>3 s)
    def test_multinomial_smoother_last_step_cross_cov_and_chain_rule(self) -> None:
        q, beta = 0.1, 1.5
        s = multinomial_choice_smoother(
            self.CHOICES, 3, process_noise=q, inverse_temperature=beta
        )
        f = multinomial_choice_filter(
            self.CHOICES, 3, process_noise=q, inverse_temperature=beta
        )
        fm, fc = np.asarray(f.filtered_values), np.asarray(f.filtered_covariances)
        sm, sc = np.asarray(s.smoothed_values), np.asarray(s.smoothed_covariances)
        np.testing.assert_array_equal(sm[-1], fm[-1])
        np.testing.assert_array_equal(sc[-1], fc[-1])
        _assert_rts_cross_cov(
            fm, fc, sc, np.asarray(s.smoother_cross_cov), np.eye(2), q * np.eye(2)
        )
        split = 5
        head = multinomial_choice_filter(
            self.CHOICES[:split], 3, process_noise=q, inverse_temperature=beta
        )
        tail = multinomial_choice_filter(
            self.CHOICES[split:],
            3,
            process_noise=q,
            inverse_temperature=beta,
            init_mean=fm[split - 1],
            init_cov=fc[split - 1],
        )
        np.testing.assert_allclose(
            float(head.marginal_log_likelihood) + float(tail.marginal_log_likelihood),
            float(f.marginal_log_likelihood),
            rtol=1e-12,
        )

    @pytest.mark.slow  # compile-dominated (>3 s)
    def test_switching_choice_probabilities(self) -> None:
        Zd = np.array([[0.85, 0.15], [0.25, 0.75]])
        qs, betas = jnp.array([0.02, 0.3]), jnp.array([0.5, 3.0])
        f = switching_choice_filter(
            self.CHOICES,
            3,
            n_discrete_states=2,
            process_noises=qs,
            inverse_temperatures=betas,
            discrete_transition_matrix=jnp.asarray(Zd),
            init_discrete_prob=jnp.array([0.5, 0.5]),
        )
        T, k = self.CHOICES.size, 2
        sm = switching_choice_smoother(
            f.filtered_values,
            f.filtered_covs,
            f.discrete_state_probs,
            qs[None, None, :] * jnp.eye(k)[:, :, None],
            jnp.ones((k, k, 2)) * jnp.eye(k)[:, :, None],
            jnp.asarray(Zd),
            jnp.zeros((T, k)),
        )
        fp, prob, joint = (
            np.asarray(f.discrete_state_probs),
            np.asarray(sm[2]),
            np.asarray(sm[3]),
        )
        np.testing.assert_allclose(fp.sum(1), 1.0, atol=1e-14)
        np.testing.assert_allclose(prob.sum(1), 1.0, atol=1e-12)
        np.testing.assert_allclose(joint.sum(2), prob[:-1], atol=1e-12)
        np.testing.assert_allclose(joint.sum(1), prob[1:], atol=1e-12)
        np.testing.assert_allclose(prob[-1], fp[-1], atol=1e-15)
        np.testing.assert_allclose(
            np.asarray(sm[5])[-1], np.asarray(f.filtered_values)[-1], atol=1e-14
        )
        assert np.max(np.abs(prob - fp)) > 0.01  # guard: smoothing matters

        # (a) the log-likelihood is the mixture over (S_{t-1}, S_t) of the
        # pair-conditional Laplace evidences (t = 0: over S_1, with the
        # predict-then-update first step from the prior N(0, I)).
        fm, fc = np.asarray(f.filtered_values), np.asarray(f.filtered_covs)
        pm, pc = np.asarray(f.pair_cond_means), np.asarray(f.pair_cond_covs)
        qs_np, betas_np = np.asarray(qs), np.asarray(betas)
        total = 0.0
        for t, c in enumerate(self.CHOICES):
            logs = []
            for j in range(2):
                if t == 0:
                    logs.append(
                        np.log(0.5)
                        + _softmax_laplace_term(
                            c,
                            betas_np[j],
                            0.0,
                            fm[0][:, j],
                            fc[0][..., j],
                            np.zeros(k),
                            (1.0 + qs_np[j]) * np.eye(k),
                        )
                    )
                    continue
                for i in range(2):
                    logs.append(
                        np.log(fp[t - 1][i] * Zd[i, j])
                        + _softmax_laplace_term(
                            c,
                            betas_np[j],
                            0.0,
                            pm[t][:, i, j],
                            pc[t][..., i, j],
                            fm[t - 1][:, i],
                            fc[t - 1][..., i] + qs_np[j] * np.eye(k),
                        )
                    )
            total += logsumexp(logs)
        # 1e-8: the update's log-determinants carry an absolute 1e-9
        # Cholesky shift (see test_invariances, the xfail on the multinomial
        # evidence's scale equivariance)
        np.testing.assert_allclose(float(f.marginal_log_likelihood), total, rtol=1e-8)


@pytest.mark.slow  # compile-dominated (>3 s)
class TestContingencyBeliefIdentities:
    @staticmethod
    def _kwargs(choices, rewards, init):
        rng = np.random.default_rng(3)
        S, K = 3, 3
        return dict(
            choices=choices,
            rewards=rewards,
            n_states=S,
            n_options=K,
            reward_probs=jnp.asarray(rng.uniform(0.1, 0.9, (S, K))),
            state_values=jnp.asarray(rng.normal(size=(S, K))),
            inverse_temperature=1.7,
            transition_logits=jnp.asarray(rng.normal(size=(S, S - 1))),
            init_state_prob=jnp.asarray(init),
        )

    def test_hmm_identities(self) -> None:
        rng = np.random.default_rng(0)
        T = 10
        choices, rewards = rng.integers(0, 3, T), rng.integers(0, 2, T)
        init = np.array([0.5, 0.3, 0.2])
        kw = self._kwargs(choices, rewards, init)
        f = contingency_belief_filter(**kw)
        s = contingency_belief_smoother(**kw)
        fp = np.asarray(f.state_posterior)
        sp, pair = np.asarray(s.smoothed_state_prob), np.asarray(s.pairwise_state_prob)
        np.testing.assert_allclose(fp.sum(1), 1.0, atol=1e-14)
        np.testing.assert_allclose(sp.sum(1), 1.0, atol=1e-13)
        np.testing.assert_allclose(pair.sum(2), sp[:-1], atol=1e-13)
        np.testing.assert_allclose(pair.sum(1), sp[1:], atol=1e-13)
        np.testing.assert_allclose(sp[-1], fp[-1], atol=1e-14)
        np.testing.assert_allclose(
            float(s.log_likelihood), float(f.log_likelihood), rtol=1e-12
        )
        # chain rule: restart from the one-step prediction of the filter at t
        trans = np.asarray(centered_softmax(kw["transition_logits"]))
        split = 4
        head = contingency_belief_filter(
            **self._kwargs(choices[:split], rewards[:split], init)
        )
        tail = contingency_belief_filter(
            **self._kwargs(choices[split:], rewards[split:], fp[split - 1] @ trans)
        )
        np.testing.assert_allclose(
            float(head.log_likelihood) + float(tail.log_likelihood),
            float(f.log_likelihood),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(tail.state_posterior), fp[split:], atol=1e-13
        )
        assert np.max(np.abs(sp - fp)) > 0.05  # guard


class TestSmithIdentities:
    Y = np.array([1, 2, 2, 4, 5, 5, 3, 4, 5, 5])
    N, SIGMA2, PC = 5, 0.3, 0.25

    def _filter(self, y, init_state=0.0, init_var=None):
        return [
            np.asarray(v)
            for v in smith_learning_filter(
                jnp.asarray(y),
                init_learning_state=init_state,
                init_learning_variance=self.SIGMA2 if init_var is None else init_var,
                sigma_epsilon=float(np.sqrt(self.SIGMA2)),
                prob_correct_by_chance=self.PC,
                max_possible_correct=self.N,
            )
        ]

    @pytest.mark.slow  # compile-dominated (>3 s)
    def test_model_ll_is_sum_of_laplace_evidence(self) -> None:
        _, mode, var, pm, pv = self._filter(self.Y)
        np.testing.assert_allclose(pv[1:], var[:-1] + self.SIGMA2, rtol=1e-15)
        np.testing.assert_array_equal(pm[1:], mode[:-1])
        mu = np.log(self.PC / (1 - self.PC))
        p = 1 / (1 + np.exp(-(mu + mode)))
        y, n = self.Y, self.N
        log_lik = (
            gammaln(n + 1)
            - gammaln(y + 1)
            - gammaln(n - y + 1)
            + y * np.log(p)
            + (n - y) * np.log1p(-p)
        )
        terms = (
            log_lik - 0.5 * (mode - pm) ** 2 / pv - 0.5 * np.log(pv) + 0.5 * np.log(var)
        )
        model = SmithLearningModel(
            init_learning_variance=self.SIGMA2,
            sigma_epsilon=float(np.sqrt(self.SIGMA2)),
            prob_correct_by_chance=self.PC,
            max_possible_correct=self.N,
        )
        np.testing.assert_allclose(
            model._e_step(jnp.asarray(self.Y)), terms.sum(), rtol=1e-10
        )

    @pytest.mark.slow  # Smith filter compiles for two lengths (~5 s)
    def test_smoother_last_step_and_chain_rule(self) -> None:
        _, mode, var, pm, pv = self._filter(self.Y)
        sm, sv, _, gain = (
            np.asarray(v)
            for v in smith_learning_smoother(
                mode, var, pm, pv, prob_correct_by_chance=self.PC
            )
        )
        np.testing.assert_array_equal(sm[-1], mode[-1])
        np.testing.assert_array_equal(sv[-1], var[-1])
        np.testing.assert_allclose(gain, var[:-1] / pv[1:], rtol=1e-15)
        # joint (x_t, x_{t+1}) covariance [[sv_t, g sv_{t+1}], [., sv_{t+1}]] PSD
        cross = gain * sv[1:]
        assert np.all(sv[:-1] * sv[1:] - cross**2 > 0)
        split = 4
        _, t_mode, t_var, _, _ = self._filter(
            self.Y[split:],
            init_state=float(mode[split - 1]),
            init_var=float(var[split - 1]),
        )
        np.testing.assert_allclose(t_mode, mode[split:], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(t_var, var[split:], rtol=1e-12)


@pytest.mark.slow  # compile-dominated (>3 s)
class TestPositionDecoderIdentities:
    def test_ll_identity_and_smoother_last_step(self) -> None:
        """Nonlinear (bilinear log-rate) intensity: the Laplace evidence per
        bin from the decoder's own dynamics predictions and filtered
        moments sums to its marginal log-likelihood (the track penalty is
        zero inside an arena without an occupancy mask)."""
        rng = np.random.default_rng(0)
        edges = np.linspace(0.0, 40.0, 41)
        gx, gy = np.meshgrid(edges, edges)
        centers = rng.uniform(8, 32, (6, 2))
        d2 = (gx[None] - centers[:, 0, None, None]) ** 2 + (
            gy[None] - centers[:, 1, None, None]
        ) ** 2
        rates = 1.0 + 30.0 * np.exp(-0.5 * d2 / 8.0**2)
        maps = PlaceFieldRateMaps(rates, edges, edges)
        dt, q_pos, T = 0.02, 50.0, 30
        path = 20.0 + np.cumsum(np.sqrt(q_pos * dt) * rng.normal(size=(T, 2)), 0)
        log_rate = jax.vmap(maps.log_rate)
        spikes = rng.poisson(np.exp(np.asarray(log_rate(jnp.asarray(path)))) * dt)
        kw = dict(
            q_pos=q_pos,
            include_velocity=False,
            init_position=np.array([20.0, 20.0]),
            init_cov=4.0 * np.eye(2),
        )
        res, pm, pc = _position_decoder_filter_with_predictions(spikes, maps, dt, **kw)
        fm, fc = np.asarray(res.position_mean), np.asarray(res.position_cov)
        lr = np.asarray(log_rate(jnp.asarray(fm)))
        terms = [
            _laplace_terms(
                spikes[t], lr[t], dt, fm[t], fc[t], np.asarray(pm[t]), np.asarray(pc[t])
            )
            for t in range(T)
        ]
        # the decoder's absolute 1e-9 Cholesky shift perturbs the prior
        # log-determinant by ~1e-9 * tr(P^-1) per bin
        np.testing.assert_allclose(res.marginal_log_likelihood, sum(terms), rtol=1e-8)
        sm = position_decoder_smoother(spikes, maps, dt, **kw)
        np.testing.assert_array_equal(np.asarray(sm.position_mean)[-1], fm[-1])
        np.testing.assert_array_equal(np.asarray(sm.position_cov)[-1], fc[-1])
        assert spikes.sum() > 20  # guard


@pytest.mark.slow  # compile-dominated (>3 s)
class TestPlaceFieldModelIdentities:
    def test_e_step_ll_and_last_step(self) -> None:
        """PlaceFieldModel E-step (block path): smoother == filter at T and
        the per-neuron smoothed blocks satisfy the RTS cross-covariance."""
        rng = np.random.default_rng(1)
        n_neurons, nb, T, dt = 2, 3, 40, 0.02
        Z = rng.uniform(0, 1, (T, nb))
        Z /= Z.sum(1, keepdims=True)
        n = n_neurons * nb
        model = PlaceFieldModel(dt=dt)
        model.n_neurons, model.n_basis_per_neuron, model.n_basis = n_neurons, nb, n
        model.transition_matrix = jnp.eye(n)
        model.process_cov = 1e-3 * jnp.eye(n)
        model.init_mean = jnp.asarray(np.log(20.0) + rng.normal(0, 0.3, n))
        model.init_cov = 0.05 * jnp.eye(n)
        model._block_n_neurons, model._block_size = model._detect_block_structure()
        assert model._block_n_neurons == n_neurons  # guard: block path
        y = rng.poisson(20.0 * dt, (T, n_neurons)).astype(float)
        model._e_step(jnp.asarray(Z), jnp.asarray(y))
        sm, fm = np.asarray(model.smoother_mean), np.asarray(model.filtered_mean)
        sc = np.asarray(model.smoother_cov.to_dense())
        fc = np.asarray(model.filtered_cov.to_dense())
        scc = np.asarray(model.smoother_cross_cov.to_dense())
        np.testing.assert_array_equal(sm[-1], fm[-1])
        np.testing.assert_array_equal(sc[-1], fc[-1])
        _assert_rts_cross_cov(fm, fc, sc, scc, np.eye(n), 1e-3 * np.eye(n))
