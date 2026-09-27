# ruff: noqa: E402
"""Invariances and equivariances every model must satisfy exactly.

Each test transforms a problem by a symmetry of the model -- a change of
latent coordinates, of observation units, of time units, a relabelling of
channels / neurons / options / discrete states / oscillators, a translation
or axis swap of the arena, a time reversal -- runs the library on both
versions and checks that the log-likelihood is unchanged (or changes by the
exact Jacobian of the data transform) and that the posteriors transform
accordingly. These are identities, so they are checked to round-off
(``rtol ~ 1e-10``) wherever the computation is a fixed sequence of
arithmetic. Two documented exceptions:

* Laplace updates with ``max_newton_iter > 1`` (and the Newton solvers of
  the choice models) accept a line-search step only when the objective
  decreases; within ~1e-8 of the mode that comparison is decided by
  round-off, so two transformed runs may stop ~sqrt(eps) apart (see
  ``test_oracle_point_process``). Those cases use ``rtol ~ 1e-7`` and the
  exact-to-round-off check is made with ``max_newton_iter=1``.
* Where a model is genuinely *not* invariant (a parametrisation pins a
  reference, dynamics break exchangeability), the docstring says why and
  the test asserts the correct, weaker statement.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import expit

from state_space_practice.contingency_belief import (
    centered_softmax,
    contingency_belief_smoother,
)
from state_space_practice.covariate_choice import covariate_choice_filter
from state_space_practice.kalman import kalman_filter, kalman_smoother
from state_space_practice.multinomial_choice import (
    multinomial_choice_filter,
    multinomial_choice_smoother,
)
from state_space_practice.oscillator_models import (
    CommonOscillatorModel,
    CorrelatedNoiseModel,
    DirectedInfluenceModel,
)
from state_space_practice.place_field_model import PlaceFieldModel
from state_space_practice.point_process_kalman import (
    stochastic_point_process_smoother,
)
from state_space_practice.point_process_models import (
    CommonOscillatorPointProcessModel,
    CorrelatedNoisePointProcessModel,
    DirectedInfluencePointProcessModel,
)
from state_space_practice.position_decoder import (
    PlaceFieldRateMaps,
    position_decoder_smoother,
)
from state_space_practice.smith_learning_algorithm import (
    smith_learning_filter,
    smith_learning_smoother,
)
from state_space_practice.switching_choice import switching_choice_filter
from state_space_practice.switching_kalman import (
    switching_kalman_filter,
    switching_kalman_smoother,
    switching_kalman_smoother_gpb2,
)
from state_space_practice.switching_point_process import (
    SpikeObsParams,
    SwitchingSpikeOscillatorModel,
)
from state_space_practice.tests.oracles import random_spd_matrix, random_stable_matrix

RTOL = 1e-10


def _close(actual, desired, rtol=RTOL, err_msg=""):
    """Relative to the array's own scale (entries near 0 compared absolutely)."""
    desired = np.asarray(desired)
    atol = rtol * max(float(np.max(np.abs(desired))), 1e-300)
    np.testing.assert_allclose(
        np.asarray(actual), desired, rtol=rtol, atol=atol, err_msg=err_msg
    )


def _random_invertible(rng, n: int) -> np.ndarray:
    """Well-conditioned random invertible matrix (not orthogonal)."""
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return U @ np.diag(rng.uniform(0.3, 3.0, n)) @ V


def _rotation(phi: float) -> np.ndarray:
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def _block_permutation(perm) -> np.ndarray:
    """(Pi x) block i = x block perm[i] for 2-D oscillator blocks."""
    n = len(perm)
    Pi = np.zeros((2 * n, 2 * n))
    for i, j in enumerate(perm):
        Pi[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = np.eye(2)
    return Pi


def _permutation_matrix(perm) -> np.ndarray:
    return np.eye(len(perm))[list(perm)]


# ===========================================================================
# 1. Linear-Gaussian models
# ===========================================================================


def _lgssm(seed: int, n=2, m=3, n_time=10) -> dict:
    rng = np.random.default_rng(seed)
    p = {
        "m0": rng.normal(size=n),
        "P0": random_spd_matrix(rng, n),
        "A": random_stable_matrix(rng, n),
        "Q": random_spd_matrix(rng, n, scale=0.4),
        "H": rng.normal(size=(m, n)),
        "R": random_spd_matrix(rng, m, scale=0.5),
    }
    x = rng.multivariate_normal(p["m0"], p["P0"])
    ys = []
    for _ in range(n_time):
        x = p["A"] @ x + rng.multivariate_normal(np.zeros(n), p["Q"])
        ys.append(p["H"] @ x + rng.multivariate_normal(np.zeros(m), p["R"]))
    p["y"] = np.stack(ys)
    return p


def _kalman(p: dict) -> dict:
    args = [jnp.asarray(p[k]) for k in ("m0", "P0", "y", "A", "Q", "H", "R")]
    fm, fc, ll = kalman_filter(*args)
    sm, sc, scc, _ = kalman_smoother(*args)
    return {
        "fm": np.asarray(fm),
        "fc": np.asarray(fc),
        "sm": np.asarray(sm),
        "sc": np.asarray(sc),
        "scc": np.asarray(scc),
        "ll": float(ll),
    }


class TestKalmanInvariances:
    def test_latent_change_of_coordinates(self) -> None:
        """x -> L x (A -> L A L^-1, Q -> L Q L', H -> H L^-1, m0 -> L m0,
        P0 -> L P0 L'): same log-likelihood, posteriors transformed."""
        p = _lgssm(0)
        L = _random_invertible(np.random.default_rng(1), 2)
        Li = np.linalg.inv(L)
        q = dict(
            p,
            m0=L @ p["m0"],
            P0=L @ p["P0"] @ L.T,
            A=L @ p["A"] @ Li,
            Q=L @ p["Q"] @ L.T,
            H=p["H"] @ Li,
        )
        a, b = _kalman(p), _kalman(q)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=RTOL)
        for key in ("fm", "sm"):
            _close(b[key], a[key] @ L.T, err_msg=key)
        for key in ("fc", "sc", "scc"):
            _close(b[key], L @ a[key] @ L.T, err_msg=key)
        assert np.linalg.cond(L) > 2.0  # guard: a genuine (non-orthogonal) map

    @pytest.mark.parametrize("c", [1e-3, 7.0])
    def test_observation_units(self, c: float) -> None:
        """y -> c y (H -> c H, R -> c^2 R): posteriors unchanged, the
        log-likelihood shifts by the Jacobian -T m log|c|."""
        p = _lgssm(2)
        q = dict(p, y=c * p["y"], H=c * p["H"], R=c**2 * p["R"])
        a, b = _kalman(p), _kalman(q)
        T, m = p["y"].shape
        np.testing.assert_allclose(b["ll"], a["ll"] - T * m * np.log(c), rtol=RTOL)
        for key in ("fm", "fc", "sm", "sc", "scc"):
            _close(b[key], a[key], err_msg=key)

    def test_observation_mixing_and_channel_permutation(self) -> None:
        """y -> M y for any invertible M: log-likelihood shifts by
        -T log|det M|; a channel permutation (|det| = 1) leaves it exact."""
        p = _lgssm(3)
        rng = np.random.default_rng(4)
        T = p["y"].shape[0]
        for M in (_random_invertible(rng, 3), _permutation_matrix([2, 0, 1])):
            q = dict(p, y=p["y"] @ M.T, H=M @ p["H"], R=M @ p["R"] @ M.T)
            a, b = _kalman(p), _kalman(q)
            logdet = np.linalg.slogdet(M)[1]
            np.testing.assert_allclose(b["ll"], a["ll"] - T * logdet, rtol=RTOL)
            for key in ("sm", "sc", "scc"):
                _close(b[key], a[key], err_msg=key)

    @staticmethod
    def _stationary(seed: int, reversible: bool) -> dict:
        """Stationary AR model started in its stationary law. Reversible iff
        A Sigma is symmetric: A symmetric and Q = q I (Sigma commutes with A);
        a rotation-like A with Q = q I is stationary but not reversible."""
        rng = np.random.default_rng(seed)
        n, m, T = 2, 3, 10  # the shapes of _lgssm (shared compilation)
        if reversible:
            S = rng.normal(size=(n, n))
            A = S + S.T
            A *= 0.8 / np.max(np.abs(np.linalg.eigvalsh(A)))
        else:
            A = 0.85 * _rotation(0.6)
        Q = 0.3 * np.eye(n)
        Sigma = np.linalg.solve(np.eye(n * n) - np.kron(A, A), Q.ravel()).reshape(n, n)
        p = {"m0": np.zeros(n), "P0": Sigma, "A": A, "Q": Q}
        p["H"] = rng.normal(size=(m, n))
        p["R"] = random_spd_matrix(rng, m, scale=0.5)
        p["y"] = rng.normal(size=(T, m))
        return p

    def test_time_reversal_for_stationary_reversible_model(self) -> None:
        """Stationary, reversible model (x_0 ~ N(0, Sigma), so x_1..x_T is
        stationary and its law is invariant under time reversal): the
        smoother of the reversed sequence is the reversed smoother (the
        lag-one cross-covariances are reversed and transposed) and the
        log-likelihood is unchanged. Guard: with a stationary but
        non-reversible (rotational) A the reversed log-likelihood differs."""
        p = self._stationary(5, reversible=True)
        q = dict(p, y=p["y"][::-1])
        a, b = _kalman(p), _kalman(q)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=1e-12)
        _close(b["sm"], a["sm"][::-1], rtol=1e-10)
        _close(b["sc"], a["sc"][::-1], rtol=1e-10)
        _close(b["scc"], np.swapaxes(a["scc"][::-1], 1, 2), rtol=1e-10)
        # guard
        p = self._stationary(5, reversible=False)
        a, b = _kalman(p), _kalman(dict(p, y=p["y"][::-1]))
        assert abs(a["ll"] - b["ll"]) > 1e-2


# --- switching Kalman --------------------------------------------------------

_SWITCHING_KEYS = ("m0", "P0", "pi", "y", "Z", "A", "Q", "H", "R")


def _switching(seed: int, n=2, m=2, K=2, n_time=8) -> dict:
    rng = np.random.default_rng(seed)
    stack = lambda draw: np.stack([draw() for _ in range(K)], -1)  # noqa: E731
    p = {
        "m0": stack(lambda: rng.normal(size=n)),
        "P0": stack(lambda: random_spd_matrix(rng, n)),
        "pi": rng.dirichlet(3 * np.ones(K)),
        "Z": 0.6 * np.eye(K) + 0.4 * rng.dirichlet(2 * np.ones(K), size=K),
        "A": stack(lambda: random_stable_matrix(rng, n)),
        "Q": stack(lambda: random_spd_matrix(rng, n, scale=0.4)),
        "H": stack(lambda: rng.normal(size=(m, n))),
        "R": stack(lambda: random_spd_matrix(rng, m, scale=0.4)),
    }
    p["y"] = rng.normal(size=(n_time, m))
    return p


def _run_switching(p: dict) -> dict:
    fm, fc, fp, pfm, pfc, pfp, ll = switching_kalman_filter(
        *[jnp.asarray(p[k]) for k in _SWITCHING_KEYS]
    )
    A, Q, Z = (jnp.asarray(p[k]) for k in ("A", "Q", "Z"))
    g1 = switching_kalman_smoother(fm, fc, fp, Q, A, Z)
    g2 = switching_kalman_smoother_gpb2(fm, fc, fp, pfm, pfc, pfp, Q, A)
    out = {"ll": float(ll), "fp": np.asarray(fp), "fm": np.asarray(fm)}
    for name, g in (("g1", g1), ("g2", g2)):
        out[name] = {
            "mean": np.asarray(g[0]),
            "cov": np.asarray(g[1]),
            "prob": np.asarray(g[2]),
            "joint": np.asarray(g[3]),
            "cross": np.asarray(g[4]),
            "sc_mean": np.asarray(g[5]),
        }
    return out


def _per_state(f, arr):
    return np.stack([f(arr[..., j]) for j in range(arr.shape[-1])], -1)


@pytest.mark.slow  # switching filter + two smoothers compile (~8 s)
class TestSwitchingKalmanInvariances:
    def test_latent_change_of_coordinates(self) -> None:
        p = _switching(0)
        L = _random_invertible(np.random.default_rng(1), 2)
        Li = np.linalg.inv(L)
        q = dict(
            p,
            m0=L @ p["m0"],
            P0=_per_state(lambda P: L @ P @ L.T, p["P0"]),
            A=_per_state(lambda A: L @ A @ Li, p["A"]),
            Q=_per_state(lambda Q: L @ Q @ L.T, p["Q"]),
            H=_per_state(lambda H: H @ Li, p["H"]),
        )
        a, b = _run_switching(p), _run_switching(q)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=RTOL)
        _close(b["fp"], a["fp"])
        for g in ("g1", "g2"):
            _close(b[g]["prob"], a[g]["prob"], err_msg=g)
            _close(b[g]["joint"], a[g]["joint"], err_msg=g)
            _close(b[g]["mean"], a[g]["mean"] @ L.T, err_msg=g)
            _close(b[g]["cov"], L @ a[g]["cov"] @ L.T, err_msg=g)
            _close(b[g]["cross"], L @ a[g]["cross"] @ L.T, err_msg=g)
            _close(b[g]["sc_mean"], np.einsum("ab,tbk->tak", L, a[g]["sc_mean"]))

    def test_observation_units_and_channel_permutation(self) -> None:
        p = _switching(2, m=3)
        T = p["y"].shape[0]
        for M in (3e-2 * np.eye(3), _permutation_matrix([1, 2, 0])):
            q = dict(
                p,
                y=p["y"] @ M.T,
                H=_per_state(lambda H, M=M: M @ H, p["H"]),
                R=_per_state(lambda R, M=M: M @ R @ M.T, p["R"]),
            )
            a, b = _run_switching(p), _run_switching(q)
            logdet = np.linalg.slogdet(M)[1]
            np.testing.assert_allclose(b["ll"], a["ll"] - T * logdet, rtol=RTOL)
            for g in ("g1", "g2"):
                _close(b[g]["prob"], a[g]["prob"])
                _close(b[g]["mean"], a[g]["mean"])
                _close(b[g]["cov"], a[g]["cov"])

    def test_discrete_state_relabelling(self) -> None:
        p = _switching(3, K=3)
        perm = [2, 0, 1]
        q = dict(p, pi=p["pi"][perm], Z=p["Z"][np.ix_(perm, perm)])
        for key in ("m0", "P0", "A", "Q", "H", "R"):
            q[key] = p[key][..., perm]
        a, b = _run_switching(p), _run_switching(q)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=RTOL)
        for g in ("g1", "g2"):
            _close(b[g]["prob"], a[g]["prob"][:, perm])
            _close(b[g]["joint"], a[g]["joint"][:, perm][:, :, perm])
            _close(b[g]["mean"], a[g]["mean"])
            _close(b[g]["sc_mean"], a[g]["sc_mean"][..., perm])
        assert np.ptp(a["g1"]["prob"]) > 0.1  # guard: states are distinguishable

    def test_time_reversal_in_exact_regime(self) -> None:
        """Identical per-state continuous parameters from a stationary,
        reversible AR model (prior on x_1 = the stationary law), a symmetric
        (doubly stochastic, hence reversible under the uniform law) discrete
        chain started uniform: the joint law is time-reversible and the GPB
        smoothers are exact, so smoothing the reversed data gives the
        reversed posterior and the same log-likelihood. (With distinct
        per-state dynamics the model is not reversible -- x_t's transition is
        governed by the destination state S_t -- so no identity is tested.)"""
        rng = np.random.default_rng(9)
        n, m, K, T = 2, 2, 2, 8
        S_ = rng.normal(size=(n, n))
        A = S_ + S_.T
        A *= 0.8 / np.max(np.abs(np.linalg.eigvalsh(A)))
        Q = 0.3 * np.eye(n)
        Sigma = np.linalg.solve(np.eye(n * n) - np.kron(A, A), Q.ravel()).reshape(n, n)
        H, R = rng.normal(size=(m, n)), random_spd_matrix(rng, m, scale=0.5)

        def st(a):
            return np.stack([a] * K, -1)

        p = dict(
            m0=st(np.zeros(n)),
            P0=st(Sigma),
            pi=np.full(K, 1.0 / K),
            Z=np.array([[0.7, 0.3], [0.3, 0.7]]),
            A=st(A),
            Q=st(Q),
            H=st(H),
            R=st(R),
            y=rng.normal(size=(T, m)),
        )
        a, b = _run_switching(p), _run_switching(dict(p, y=p["y"][::-1]))
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=1e-12)
        for g in ("g1", "g2"):
            _close(b[g]["prob"], a[g]["prob"][::-1], rtol=1e-10)
            _close(b[g]["mean"], a[g]["mean"][::-1], rtol=1e-10)
            _close(b[g]["cov"], a[g]["cov"][::-1], rtol=1e-10)
            _close(b[g]["cross"], np.swapaxes(a[g]["cross"][::-1], 1, 2), rtol=1e-10)


# --- oscillator models (COM / CNM / DIM) --------------------------------------

_N_OSC = 3
_FREQS = np.array([6.0, 10.0, 17.0])
_DAMP = np.array([0.95, 0.9, 0.93])


def _cnm_pairs(K: int):
    """Mirrored CNM pair parameters (coupling symmetric, phase antisymmetric)."""
    rng = np.random.default_rng(7)
    c = np.zeros((_N_OSC, _N_OSC, K))
    ph = np.zeros((_N_OSC, _N_OSC, K))
    for i in range(_N_OSC):
        for j in range(i + 1, _N_OSC):
            c[i, j] = c[j, i] = rng.uniform(0.002, 0.01, K)
            ph[i, j] = rng.uniform(-np.pi, np.pi, K)
            ph[j, i] = -ph[i, j]
    return ph, c


def _dim_pairs(K: int):
    rng = np.random.default_rng(8)
    c = rng.uniform(0.0, 0.05, (_N_OSC, _N_OSC, K))
    ph = rng.uniform(-np.pi, np.pi, (_N_OSC, _N_OSC, K))
    idx = np.arange(_N_OSC)
    c[idx, idx] = 0.0
    ph[idx, idx] = 0.0
    return ph, c


def _gaussian_oscillator_model(kind: str, perm=(0, 1, 2)):
    """A 2-state oscillator model with its own initialised parameters, the
    oscillators relabelled by ``perm`` (new oscillator i = old perm[i])."""
    perm = list(perm)
    K = 2
    common = dict(n_oscillators=_N_OSC, n_discrete_states=K, sampling_freq=100.0)
    freqs, damp = jnp.asarray(_FREQS[perm]), jnp.asarray(_DAMP[perm])
    if kind == "COM":
        model = CommonOscillatorModel(
            n_sources=_N_OSC,
            freqs=freqs,
            damping_coef=damp,
            process_variance=jnp.asarray(np.array([0.01, 0.02, 0.015])[perm]),
            measurement_variance=0.1,
            **common,
        )
    elif kind == "CNM":
        ph, c = _cnm_pairs(K)
        model = CorrelatedNoiseModel(
            freqs=freqs,
            damping_coef=damp,
            process_variance=jnp.asarray(
                np.array([[0.01, 0.02], [0.02, 0.01], [0.015, 0.03]])[perm]
            ),
            measurement_variance=0.1,
            phase_difference=jnp.asarray(ph[np.ix_(perm, perm)]),
            coupling_strength=jnp.asarray(c[np.ix_(perm, perm)]),
            **common,
        )
    else:
        ph, c = _dim_pairs(K)
        model = DirectedInfluenceModel(
            freqs=freqs,
            damping_coef=damp,
            process_variance=jnp.asarray(np.array([0.01, 0.02, 0.015])[perm]),
            measurement_variance=0.1,
            phase_difference=jnp.asarray(ph[np.ix_(perm, perm)]),
            coupling_strength=jnp.asarray(c[np.ix_(perm, perm)]),
            **common,
        )
    model._initialize_parameters(jax.random.PRNGKey(0))
    # distinct discrete states: a state-specific measurement gain
    rng = np.random.default_rng(11)
    H = np.asarray(model.measurement_matrix).copy()
    if kind == "COM":
        H = rng.normal(size=H.shape)
    else:
        H[..., 1] *= 1.7
    model.measurement_matrix = jnp.asarray(H)
    model.init_mean = jnp.asarray(rng.normal(size=np.shape(model.init_mean)))
    return model


def _oscillator_obs(seed: int = 0, T: int = 40) -> np.ndarray:
    t = np.arange(T) / 100.0
    rng = np.random.default_rng(seed)
    return np.stack(
        [np.cos(2 * np.pi * f * t + k) for k, f in enumerate(_FREQS)], -1
    ) + 0.3 * rng.normal(size=(T, _N_OSC))


def _apply_latent_map(model, L, P_src=None):
    """Transform the model's latent coordinates by L (A, Q given) and,
    optionally, relabel the sources by P_src."""
    Li = np.linalg.inv(L)
    model.init_mean = jnp.asarray(L @ np.asarray(model.init_mean))
    model.init_cov = jnp.asarray(
        _per_state(lambda P: L @ P @ L.T, np.asarray(model.init_cov))
    )
    H = _per_state(lambda H: H @ Li, np.asarray(model.measurement_matrix))
    if P_src is not None:
        H = _per_state(lambda h: P_src @ h, H)
        model.measurement_cov = jnp.asarray(
            _per_state(lambda R: P_src @ R @ P_src.T, np.asarray(model.measurement_cov))
        )
    model.measurement_matrix = jnp.asarray(H)


@pytest.mark.slow  # switching filter + GPB1 smoother compile per model class
class TestOscillatorModelInvariances:
    @pytest.mark.parametrize("kind", ["COM", "CNM", "DIM"])
    def test_common_phase_shift(self, kind: str) -> None:
        """Rotating every oscillator's 2-D coordinates by the same angle
        (L = blockdiag R(phi)) commutes with the model's rotation-block
        transition matrices and rotation-invariant (CNM: scaled-rotation
        cross-block) process covariances, so only H, m0, P0 move; the
        E-step log-likelihood and discrete posterior are unchanged and the
        smoothed latent is rotated."""
        L = np.kron(np.eye(_N_OSC), _rotation(0.8))
        a_model = _gaussian_oscillator_model(kind)
        b_model = _gaussian_oscillator_model(kind)
        A = np.asarray(a_model.continuous_transition_matrix)
        Q = np.asarray(a_model.process_cov)
        # structural property the invariance rests on
        _close(_per_state(lambda M: L @ M @ L.T, A), A, rtol=1e-13)
        _close(_per_state(lambda M: L @ M @ L.T, Q), Q, rtol=1e-13)
        _apply_latent_map(b_model, L)
        y = jnp.asarray(_oscillator_obs())
        ll_a, ll_b = float(a_model._e_step(y)), float(b_model._e_step(y))
        np.testing.assert_allclose(ll_b, ll_a, rtol=RTOL)
        _close(
            b_model.smoother_discrete_state_prob, a_model.smoother_discrete_state_prob
        )
        _close(
            b_model.smoother_state_cond_mean,
            np.einsum("ab,tbk->tak", L, np.asarray(a_model.smoother_state_cond_mean)),
        )
        # guard: the rotation actually changed the latent coordinates
        assert not np.allclose(
            b_model.smoother_state_cond_mean,
            a_model.smoother_state_cond_mean,
            atol=1e-2,
        )

    @pytest.mark.parametrize("kind", ["COM", "CNM", "DIM"])
    def test_oscillator_relabelling(self, kind: str) -> None:
        """Relabelling oscillators (and the sources tied to them): the model
        constructed from permuted scientific parameters has the block-permuted
        A and Q (constructor equivariance), and the E-step is unchanged up
        to the relabelling."""
        perm = [2, 0, 1]
        Pi = _block_permutation(perm)
        P_src = _permutation_matrix(perm)
        a_model = _gaussian_oscillator_model(kind)
        b_model = _gaussian_oscillator_model(kind, perm)
        for attr in ("continuous_transition_matrix", "process_cov"):
            _close(
                getattr(b_model, attr),
                _per_state(lambda M: Pi @ M @ Pi.T, np.asarray(getattr(a_model, attr))),
                rtol=1e-12,
                err_msg=attr,
            )
        # the relabelled copy of a_model's free parameters
        b_model.init_mean = a_model.init_mean
        b_model.init_cov = a_model.init_cov
        b_model.measurement_matrix = a_model.measurement_matrix
        b_model.measurement_cov = a_model.measurement_cov
        _apply_latent_map(b_model, Pi, P_src)
        y = _oscillator_obs()
        ll_a = float(a_model._e_step(jnp.asarray(y)))
        ll_b = float(b_model._e_step(jnp.asarray(y @ P_src.T)))
        np.testing.assert_allclose(ll_b, ll_a, rtol=RTOL)
        _close(
            b_model.smoother_discrete_state_prob, a_model.smoother_discrete_state_prob
        )
        _close(
            b_model.smoother_state_cond_mean,
            np.einsum("ab,tbk->tak", Pi, np.asarray(a_model.smoother_state_cond_mean)),
        )
        assert np.ptp(np.asarray(a_model.smoother_discrete_state_prob)) > 0.05  # guard


# ===========================================================================
# 2. Point-process models
# ===========================================================================


def _affine_log_rate(design_t, x):
    return design_t[:, 0] + design_t[:, 1:] @ x


def _pp_problem(seed: int, n_latent=2, n_neurons=4, n_time=30, dt=0.02):
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0.85, 0.97, n_latent))
    A[0, -1] = 0.05
    return dict(
        m0=rng.normal(0, 0.3, n_latent),
        P0=random_spd_matrix(rng, n_latent, scale=0.2),
        A=A,
        Q=random_spd_matrix(rng, n_latent, scale=0.02),
        b=np.log(rng.uniform(15.0, 45.0, n_neurons)),
        W=rng.normal(0, 0.8, (n_neurons, n_latent)),
        y=rng.poisson(0.6, (n_time, n_neurons)).astype(float),
        dt=dt,
    )


def _pp_smooth(pr: dict, max_newton_iter: int) -> dict:
    T = pr["y"].shape[0]
    design = np.tile(np.concatenate([pr["b"][:, None], pr["W"]], 1), (T, 1, 1))
    sm, sc, scc, ll, fm, fc = stochastic_point_process_smoother(
        pr["m0"],
        pr["P0"],
        design,
        pr["y"],
        pr["dt"],
        pr["A"],
        pr["Q"],
        _affine_log_rate,
        max_newton_iter=max_newton_iter,
        return_filtered=True,
    )
    return dict(
        sm=np.asarray(sm),
        sc=np.asarray(sc),
        scc=np.asarray(scc),
        fm=np.asarray(fm),
        fc=np.asarray(fc),
        ll=float(ll),
    )


# N1 is a fixed arithmetic sequence (round-off); N3's line search compares
# losses that differ at round-off level near the mode (sqrt(eps)).
_NEWTON_RTOL = [(1, 1e-10), (3, 1e-7)]


class TestPointProcessFilterInvariances:
    @pytest.mark.parametrize(("max_newton_iter", "rtol"), _NEWTON_RTOL)
    def test_neuron_permutation(self, max_newton_iter, rtol) -> None:
        pr = _pp_problem(0)
        perm = [3, 1, 0, 2]
        q = dict(pr, b=pr["b"][perm], W=pr["W"][perm], y=pr["y"][:, perm])
        a, b = _pp_smooth(pr, max_newton_iter), _pp_smooth(q, max_newton_iter)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=rtol)
        for key in ("fm", "fc", "sm", "sc", "scc"):
            _close(b[key], a[key], rtol=rtol, err_msg=key)
        assert pr["y"].sum() > 30  # guard: informative spikes

    @pytest.mark.parametrize(("max_newton_iter", "rtol"), _NEWTON_RTOL)
    def test_change_of_time_units(self, max_newton_iter, rtol) -> None:
        """dt -> c dt with every rate -> rate / c (b -> b - log c): the
        expected counts lambda dt, hence the model, are unchanged (A and Q
        are per bin)."""
        pr = _pp_problem(1)
        c = 1e3  # e.g. seconds -> milliseconds
        q = dict(pr, dt=c * pr["dt"], b=pr["b"] - np.log(c))
        a, b = _pp_smooth(pr, max_newton_iter), _pp_smooth(q, max_newton_iter)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=rtol)
        for key in ("fm", "fc", "sm", "sc", "scc"):
            _close(b[key], a[key], rtol=rtol, err_msg=key)


def _place_field_model(n_neurons=3, nb=3, T=40, dt=0.02, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (T, nb))
    Z /= Z.sum(1, keepdims=True)
    n = n_neurons * nb
    model = PlaceFieldModel(dt=dt, max_newton_iter=1)
    model.n_neurons, model.n_basis_per_neuron, model.n_basis = n_neurons, nb, n
    model.transition_matrix = jnp.eye(n)
    model.process_cov = jnp.diag(jnp.asarray(rng.uniform(5e-4, 2e-3, n)))
    model.init_mean = jnp.asarray(np.log(20.0) + rng.normal(0, 0.4, n))
    model.init_cov = jnp.diag(jnp.asarray(rng.uniform(0.02, 0.1, n)))
    model._block_n_neurons, model._block_size = model._detect_block_structure()
    y = rng.poisson(20.0 * dt, (T, n_neurons)).astype(float)
    return model, Z, y


def _e_step_outputs(model) -> dict:
    def dense(v):
        return np.asarray(v.to_dense() if hasattr(v, "to_dense") else v)

    return dict(
        sm=np.asarray(model.smoother_mean),
        sc=dense(model.smoother_cov),
        scc=dense(model.smoother_cross_cov),
    )


@pytest.mark.slow  # block-diagonal E-step compile (~5 s)
class TestPlaceFieldModelEStepInvariances:
    """Neuron relabelling and latent rescaling of the PlaceFieldModel E-step.

    A change of time units (dt -> c dt, rates -> rates / c) is not tested
    here: it shifts every log-rate by -log c, and the spline basis has no
    intercept column (patsy's ``bs`` drops the first B-spline, so the basis
    rows do not sum to a constant), so the shift is not representable by the
    weights. It is tested for the filter itself and for the switching
    models, whose baselines are explicit."""

    def test_neuron_permutation(self) -> None:
        """Relabel neurons: spikes columns and the per-neuron weight blocks
        (block-diagonal E-step path)."""
        a_model, Z, y = _place_field_model()
        b_model, _, _ = _place_field_model()
        assert a_model._block_n_neurons == 3  # guard: block path
        perm = [2, 0, 1]
        nb = 3
        idx = np.concatenate([np.arange(j * nb, (j + 1) * nb) for j in perm])
        for attr in ("init_mean",):
            setattr(b_model, attr, getattr(a_model, attr)[idx])
        for attr in ("init_cov", "process_cov", "transition_matrix"):
            setattr(b_model, attr, getattr(a_model, attr)[np.ix_(idx, idx)])
        ll_a = a_model._e_step(jnp.asarray(Z), jnp.asarray(y))
        ll_b = b_model._e_step(jnp.asarray(Z), jnp.asarray(y[:, perm]))
        np.testing.assert_allclose(ll_b, ll_a, rtol=RTOL)
        a, b = _e_step_outputs(a_model), _e_step_outputs(b_model)
        _close(b["sm"], a["sm"][:, idx])
        _close(b["sc"], a["sc"][:, idx][:, :, idx])
        _close(b["scc"], a["scc"][:, idx][:, :, idx])

    @pytest.mark.parametrize("scale", [1e-3, 1e3])
    def test_latent_rescaling(self, scale: float) -> None:
        """x -> c x (design / c, init / Q scaled): the E-step
        (``max_newton_iter=1``) is equivariant to round-off."""
        a_model, Z, y = _place_field_model()
        b_model, _, _ = _place_field_model()
        b_model.init_mean = scale * a_model.init_mean
        b_model.init_cov = scale**2 * a_model.init_cov
        b_model.process_cov = scale**2 * a_model.process_cov
        ll_a = a_model._e_step(jnp.asarray(Z), jnp.asarray(y))
        ll_b = b_model._e_step(jnp.asarray(Z / scale), jnp.asarray(y))
        np.testing.assert_allclose(ll_b, ll_a, rtol=RTOL)
        a, b = _e_step_outputs(a_model), _e_step_outputs(b_model)
        _close(b["sm"] / scale, a["sm"])
        _close(b["sc"] / scale**2, a["sc"])
        _close(b["scc"] / scale**2, a["scc"])


# --- switching point-process models ------------------------------------------


def _pp_oscillator_model(kind: str, perm=(0, 1, 2), max_newton_iter=1, dt=0.01):
    perm = list(perm)
    K = 2
    common = dict(
        n_oscillators=_N_OSC,
        n_neurons=4,
        n_discrete_states=K,
        sampling_freq=1.0 / dt,
        dt=dt,
        max_newton_iter=max_newton_iter,
    )
    freqs, damp = jnp.asarray(_FREQS[perm]), jnp.asarray(_DAMP[perm])
    var = jnp.asarray(np.array([0.01, 0.02, 0.015])[perm])
    if kind == "SSO":
        model = SwitchingSpikeOscillatorModel(**common)
    elif kind == "COM":
        model = CommonOscillatorPointProcessModel(
            freqs=freqs, damping_coef=damp, process_variance=var, **common
        )
    elif kind == "CNM":
        ph, c = _cnm_pairs(K)
        model = CorrelatedNoisePointProcessModel(
            freqs=freqs,
            damping_coef=damp,
            process_variance=jnp.asarray(
                np.array([[0.01, 0.02], [0.02, 0.01], [0.015, 0.03]])[perm]
            ),
            phase_difference=jnp.asarray(ph[np.ix_(perm, perm)]),
            coupling_strength=jnp.asarray(c[np.ix_(perm, perm)]),
            **common,
        )
    else:
        ph, c = _dim_pairs(K)
        model = DirectedInfluencePointProcessModel(
            freqs=freqs,
            damping_coef=damp,
            process_variance=var,
            phase_difference=jnp.asarray(ph[np.ix_(perm, perm)]),
            coupling_strength=jnp.asarray(c[np.ix_(perm, perm)]),
            **common,
        )
    model._initialize_parameters(jax.random.PRNGKey(0))
    rng = np.random.default_rng(3)
    n_latent = 2 * _N_OSC
    model.spike_params = SpikeObsParams(
        baseline=jnp.asarray(np.log(rng.uniform(20, 60, (4, K)))),
        weights=jnp.asarray(rng.normal(0, 0.8, (4, n_latent, K))),
    )
    return model


def _pp_spikes(seed=0, T=40) -> np.ndarray:
    return np.random.default_rng(seed).poisson(0.5, (T, 4)).astype(float)


def _set_pp_latent_map(model, L, P_neu=None):
    Li = np.linalg.inv(L)
    model.init_mean = jnp.asarray(L @ np.asarray(model.init_mean))
    model.init_cov = jnp.asarray(
        _per_state(lambda P: L @ P @ L.T, np.asarray(model.init_cov))
    )
    model.continuous_transition_matrix = jnp.asarray(
        _per_state(lambda A: L @ A @ Li, np.asarray(model.continuous_transition_matrix))
    )
    model.process_cov = jnp.asarray(
        _per_state(lambda Q: L @ Q @ L.T, np.asarray(model.process_cov))
    )
    b = np.asarray(model.spike_params.baseline)
    W = _per_state(lambda w: w @ Li, np.asarray(model.spike_params.weights))
    if P_neu is not None:
        b, W = P_neu @ b, np.einsum("ij,jak->iak", P_neu, W)
    model.spike_params = SpikeObsParams(baseline=jnp.asarray(b), weights=jnp.asarray(W))


def _pp_estep(model, spikes) -> dict:
    ll = float(model._e_step(jnp.asarray(spikes)))
    return dict(
        ll=ll,
        prob=np.asarray(model.smoother_discrete_state_prob),
        mean=np.asarray(model.smoother_state_cond_mean),
        cov=np.asarray(model.smoother_state_cond_cov),
    )


@pytest.mark.slow  # one switching point-process filter compile per model class
class TestSwitchingPointProcessModelInvariances:
    """E-step of SwitchingSpikeOscillatorModel and the COM / CNM / DIM
    point-process oscillator models (``max_newton_iter=1``: round-off)."""

    @pytest.mark.parametrize("kind", ["SSO", "COM", "CNM", "DIM"])
    def test_neuron_permutation(self, kind: str) -> None:
        a_model = _pp_oscillator_model(kind)
        b_model = _pp_oscillator_model(kind)
        perm = [3, 1, 0, 2]
        _set_pp_latent_map(b_model, np.eye(2 * _N_OSC), _permutation_matrix(perm))
        y = _pp_spikes()
        a, b = _pp_estep(a_model, y), _pp_estep(b_model, y[:, perm])
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=RTOL)
        for key in ("prob", "mean", "cov"):
            _close(b[key], a[key], err_msg=key)
        assert np.ptp(a["prob"]) > 0.05  # guard: informative discrete posterior

    @pytest.mark.parametrize("kind", ["SSO", "DIM"])
    def test_latent_rescaling(self, kind: str) -> None:
        c = 1e-3
        a_model = _pp_oscillator_model(kind)
        b_model = _pp_oscillator_model(kind)
        _set_pp_latent_map(b_model, c * np.eye(2 * _N_OSC))
        y = _pp_spikes(1)
        a, b = _pp_estep(a_model, y), _pp_estep(b_model, y)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=RTOL)
        _close(b["prob"], a["prob"])
        _close(b["mean"] / c, a["mean"])
        _close(b["cov"] / c**2, a["cov"])

    @pytest.mark.parametrize("kind", ["SSO", "CNM"])
    def test_change_of_time_units(self, kind: str) -> None:
        """dt -> c dt, baselines -> baseline - log c (the dynamics are per
        bin, so the oscillator frequencies are held in cycles per bin)."""
        c = 1e3
        a_model = _pp_oscillator_model(kind)
        b_model = _pp_oscillator_model(kind)
        b_model.dt = c * a_model.dt
        b_model.spike_params = SpikeObsParams(
            baseline=a_model.spike_params.baseline - np.log(c),
            weights=a_model.spike_params.weights,
        )
        y = _pp_spikes(2)
        a, b = _pp_estep(a_model, y), _pp_estep(b_model, y)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=RTOL)
        for key in ("prob", "mean", "cov"):
            _close(b[key], a[key], err_msg=key)

    @pytest.mark.parametrize("kind", ["COM", "CNM", "DIM"])
    def test_common_phase_shift(self, kind: str) -> None:
        """Common rotation of all oscillator coordinates: A and Q are
        invariant (structure), only the spike weights, m0 and P0 move."""
        L = np.kron(np.eye(_N_OSC), _rotation(-1.1))
        a_model = _pp_oscillator_model(kind)
        b_model = _pp_oscillator_model(kind)
        _close(
            _per_state(
                lambda M: L @ M @ L.T, np.asarray(a_model.continuous_transition_matrix)
            ),
            a_model.continuous_transition_matrix,
            rtol=1e-13,
        )
        _set_pp_latent_map(b_model, L)
        _close(b_model.process_cov, a_model.process_cov, rtol=1e-13)
        y = _pp_spikes(3)
        a, b = _pp_estep(a_model, y), _pp_estep(b_model, y)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=RTOL)
        _close(b["prob"], a["prob"])
        _close(b["mean"], np.einsum("ab,tbk->tak", L, a["mean"]))

    @pytest.mark.parametrize("kind", ["COM", "CNM", "DIM"])
    def test_oscillator_relabelling(self, kind: str) -> None:
        perm = [1, 2, 0]
        Pi = _block_permutation(perm)
        a_model = _pp_oscillator_model(kind)
        b_model = _pp_oscillator_model(kind, perm)
        for attr in ("continuous_transition_matrix", "process_cov"):
            _close(
                getattr(b_model, attr),
                _per_state(lambda M: Pi @ M @ Pi.T, np.asarray(getattr(a_model, attr))),
                rtol=1e-12,
                err_msg=attr,
            )
        # a_model's free parameters, relabelled; A and Q come from b's
        # constructor (checked above)
        b_model.init_mean = jnp.asarray(Pi @ np.asarray(a_model.init_mean))
        b_model.init_cov = jnp.asarray(
            _per_state(lambda P: Pi @ P @ Pi.T, np.asarray(a_model.init_cov))
        )
        b_model.spike_params = SpikeObsParams(
            baseline=a_model.spike_params.baseline,
            weights=jnp.asarray(
                _per_state(lambda w: w @ Pi.T, np.asarray(a_model.spike_params.weights))
            ),
        )
        y = _pp_spikes(4)
        a, b = _pp_estep(a_model, y), _pp_estep(b_model, y)
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=RTOL)
        _close(b["prob"], a["prob"])
        _close(b["mean"], np.einsum("ab,tbk->tak", Pi, a["mean"]))

    def test_neuron_permutation_default_newton(self) -> None:
        """The default ``max_newton_iter=3`` (line-searched): invariant to
        ~sqrt(eps)."""
        a_model = _pp_oscillator_model("SSO", max_newton_iter=3)
        b_model = _pp_oscillator_model("SSO", max_newton_iter=3)
        perm = [1, 0, 3, 2]
        _set_pp_latent_map(b_model, np.eye(2 * _N_OSC), _permutation_matrix(perm))
        y = _pp_spikes(5)
        a, b = _pp_estep(a_model, y), _pp_estep(b_model, y[:, perm])
        np.testing.assert_allclose(b["ll"], a["ll"], rtol=1e-7)
        _close(b["mean"], a["mean"], rtol=1e-6)


# ===========================================================================
# 3. Choice / learning models
# ===========================================================================

_CHOICES_K4 = np.array([0, 3, 3, 1, 2, 3, 0, 3, 1, 3, 2, 3])
_CHOICES_K3 = np.array([0, 2, 2, 1, 2, 0, 0, 2, 1, 2, 2, 2])


def _mc_filter(choices, K, q, beta, m0=None, P0=None):
    f = multinomial_choice_filter(
        choices, K, process_noise=q, inverse_temperature=beta, init_mean=m0, init_cov=P0
    )
    return (
        float(f.marginal_log_likelihood),
        np.asarray(f.filtered_values),
        np.asarray(f.filtered_covariances),
    )


class TestMultinomialChoiceInvariances:
    """The latent is the value of options 1..K-1 relative to option 0 (the
    reference, whose value is pinned at 0)."""

    @pytest.mark.slow  # filter + smoother compile (~4 s)
    def test_non_reference_option_relabelling(self) -> None:
        """Permuting options 1..K-1 permutes the latent coordinates; the
        default prior N(0, I) and the isotropic process noise are invariant
        under it, so the log-likelihood is unchanged and the posteriors
        are permuted (the smoother too)."""
        perm = np.array([0, 3, 1, 2])  # new option k = old option perm[k]
        inv = np.argsort(perm)
        idx = perm[1:] - 1
        q, beta = 0.1, 1.5
        ll_a, m_a, P_a = _mc_filter(_CHOICES_K4, 4, q, beta)
        ll_b, m_b, P_b = _mc_filter(inv[_CHOICES_K4], 4, q, beta)
        np.testing.assert_allclose(ll_b, ll_a, rtol=1e-10)
        _close(m_b, m_a[:, idx], rtol=1e-9)
        _close(P_b, P_a[:, idx][:, :, idx], rtol=1e-9)
        s_a = multinomial_choice_smoother(
            _CHOICES_K4, 4, process_noise=q, inverse_temperature=beta
        )
        s_b = multinomial_choice_smoother(
            inv[_CHOICES_K4], 4, process_noise=q, inverse_temperature=beta
        )
        _close(s_b.smoothed_values, np.asarray(s_a.smoothed_values)[:, idx], rtol=1e-9)

    def test_reference_change_with_static_values(self) -> None:
        """Making option 2 the reference (swap labels 0 <-> 2) maps the
        relative values linearly, x' = L x with L = [[1, -1], [0, -1]]. The
        Laplace update is affine-equivariant (Newton's method, the Laplace
        mode / curvature and the evidence's log-det ratio all are), so with
        static values (q = 0) and the transformed prior the filter is
        invariant. With q > 0 it is not: the scalar process noise q I on the
        relative values is not preserved by L (it would have to become
        q L L'), i.e. the parametrisation pins the reference option; the
        guard asserts that."""
        swap = np.array([2, 1, 0])
        L = np.array([[1.0, -1.0], [0.0, -1.0]])
        m0, P0 = np.array([0.3, -0.4]), np.array([[1.0, 0.3], [0.3, 0.8]])
        beta = 1.5
        ll_a, m_a, P_a = _mc_filter(_CHOICES_K3, 3, 0.0, beta, m0, P0)
        ll_b, m_b, P_b = _mc_filter(
            swap[_CHOICES_K3], 3, 0.0, beta, L @ m0, L @ P0 @ L.T
        )
        np.testing.assert_allclose(ll_b, ll_a, rtol=1e-9)
        _close(m_b, m_a @ L.T, rtol=1e-8)
        _close(P_b, L @ P_a @ L.T, rtol=1e-8)
        # guard: with process noise the reference choice matters
        ll_a, *_ = _mc_filter(_CHOICES_K3, 3, 0.1, beta, m0, P0)
        ll_b, *_ = _mc_filter(swap[_CHOICES_K3], 3, 0.1, beta, L @ m0, L @ P0 @ L.T)
        assert abs(ll_a - ll_b) > 1e-3

    @pytest.mark.slow  # eager Newton scan (~5 s)
    def test_laplace_evidence_is_scale_equivariant(self) -> None:
        """x -> x / sqrt(v) maps prior N(m, v), inverse temperature beta to
        prior N(m / sqrt(v), 1), inverse temperature beta sqrt(v): the mode,
        covariance and evidence of the Laplace update must transform
        exactly (the mode and covariance do)."""
        from state_space_practice.multinomial_choice import _softmax_update_core

        def update(v):
            x, P, ll, _ = _softmax_update_core(
                jnp.array([0.3 * np.sqrt(v)]),
                jnp.array([[v]]),
                jnp.int32(0),
                2,
                2.0 / np.sqrt(v),
            )
            return float(x[0]) / np.sqrt(v), float(P[0, 0]) / v, float(ll)

        ref, small = update(1.0), update(1e-8)
        np.testing.assert_allclose(small[:2], ref[:2], rtol=1e-10)
        np.testing.assert_allclose(small[2], ref[2], rtol=1e-9)


class TestCovariateChoiceInvariances:
    @staticmethod
    def _run(choices, u, B, z, W):
        f = covariate_choice_filter(
            choices,
            3,
            covariates=u,
            input_gain=B,
            obs_covariates=z,
            obs_weights=W,
            process_noise=0.1,
            inverse_temperature=1.5,
            decay=0.9,
        )
        return float(f.marginal_log_likelihood), np.asarray(f.filtered_values)

    @staticmethod
    def _inputs():
        rng = np.random.default_rng(0)
        T = _CHOICES_K3.size
        return (
            rng.normal(size=(T, 2)),
            rng.normal(size=(2, 2)),
            rng.normal(size=(T, 2)),
            rng.normal(size=(3, 2)),
        )

    def test_covariate_rescaling_with_inverse_gain(self) -> None:
        """u -> u D with B -> B D^-1 (dynamics), z -> z E with W -> W E^-1
        (observation covariates): B u and W z are unchanged."""
        u, B, z, W = self._inputs()
        D, E = np.diag([1e-3, 50.0]), np.diag([20.0, 1e-2])
        ll_a, m_a = self._run(_CHOICES_K3, u, B, z, W)
        ll_b, m_b = self._run(
            _CHOICES_K3, u @ D, B @ np.linalg.inv(D), z @ E, W @ np.linalg.inv(E)
        )
        np.testing.assert_allclose(ll_b, ll_a, rtol=1e-10)
        _close(m_b, m_a, rtol=1e-9)

    def test_common_logit_offset_is_unidentified(self) -> None:
        """The observation weights have a row for every option, including
        the reference; adding the same row to all options shifts every logit
        equally and leaves the softmax (and the model) unchanged."""
        u, B, z, W = self._inputs()
        ll_a, m_a = self._run(_CHOICES_K3, u, B, z, W)
        ll_b, m_b = self._run(_CHOICES_K3, u, B, z, W + np.array([[3.0, -2.0]]))
        np.testing.assert_allclose(ll_b, ll_a, rtol=1e-10)
        _close(m_b, m_a, rtol=1e-9)

    def test_non_reference_option_relabelling(self) -> None:
        u, B, z, W = self._inputs()
        swap = np.array([0, 2, 1])
        ll_a, m_a = self._run(_CHOICES_K3, u, B, z, W)
        ll_b, m_b = self._run(swap[_CHOICES_K3], u, B[[1, 0]], z, W[swap])
        np.testing.assert_allclose(ll_b, ll_a, rtol=1e-10)
        _close(m_b, m_a[:, [1, 0]], rtol=1e-9)


@pytest.mark.slow  # switching choice filter compile (~8 s)
class TestSwitchingChoiceInvariances:
    @staticmethod
    def _run(choices, qs, betas, Zd, pi):
        f = switching_choice_filter(
            choices,
            3,
            n_discrete_states=len(qs),
            process_noises=jnp.asarray(qs),
            inverse_temperatures=jnp.asarray(betas),
            discrete_transition_matrix=jnp.asarray(Zd),
            init_discrete_prob=jnp.asarray(pi),
        )
        return (
            float(f.marginal_log_likelihood),
            np.asarray(f.discrete_state_probs),
            np.asarray(f.filtered_values),
        )

    QS, BETAS = np.array([0.02, 0.3, 0.1]), np.array([0.5, 3.0, 1.5])
    Z = np.array([[0.8, 0.15, 0.05], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    PI = np.array([0.5, 0.3, 0.2])

    def test_discrete_state_relabelling(self) -> None:
        perm = [2, 0, 1]
        a = self._run(_CHOICES_K3, self.QS, self.BETAS, self.Z, self.PI)
        b = self._run(
            _CHOICES_K3,
            self.QS[perm],
            self.BETAS[perm],
            self.Z[np.ix_(perm, perm)],
            self.PI[perm],
        )
        np.testing.assert_allclose(b[0], a[0], rtol=1e-10)
        _close(b[1], a[1][:, perm], rtol=1e-9)
        _close(b[2], a[2][..., perm], rtol=1e-9)
        assert np.ptp(a[1]) > 0.1  # guard

    def test_non_reference_option_relabelling(self) -> None:
        swap = np.array([0, 2, 1])
        a = self._run(_CHOICES_K3, self.QS, self.BETAS, self.Z, self.PI)
        b = self._run(swap[_CHOICES_K3], self.QS, self.BETAS, self.Z, self.PI)
        np.testing.assert_allclose(b[0], a[0], rtol=1e-10)
        _close(b[1], a[1], rtol=1e-9)
        _close(b[2], a[2][:, [1, 0]], rtol=1e-9)


class TestContingencyBeliefInvariances:
    """Options have no reference in this parametrisation (per-option
    rewards / values / observation weights), so every relabelling of
    options is a symmetry. The discrete states are parametrised by
    centered-softmax transition logits with the *last* state as the
    reference, so a state relabelling must re-express the logits relative
    to the new last state."""

    @staticmethod
    def _kwargs():
        rng = np.random.default_rng(4)
        S, K, T = 3, 3, 9
        return dict(
            choices=rng.integers(0, K, T),
            rewards=rng.integers(0, 2, T),
            n_states=S,
            n_options=K,
            reward_probs=rng.uniform(0.1, 0.9, (S, K)),
            state_values=rng.normal(size=(S, K)),
            inverse_temperature=1.7,
            transition_logits=rng.normal(size=(S, S - 1)),
            init_state_prob=np.array([0.5, 0.3, 0.2]),
            obs_design_matrix=rng.normal(size=(T, 2)),
            obs_weights=rng.normal(size=(K, 2)),
        )

    @staticmethod
    def _run(kw):
        s = contingency_belief_smoother(
            **{
                k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
                for k, v in kw.items()
            }
        )
        return (
            float(s.log_likelihood),
            np.asarray(s.smoothed_state_prob),
            np.asarray(s.pairwise_state_prob),
        )

    def test_option_relabelling(self) -> None:
        kw = self._kwargs()
        perm = np.array([2, 0, 1])
        inv = np.argsort(perm)
        kb = dict(
            kw,
            choices=inv[kw["choices"]],
            reward_probs=kw["reward_probs"][:, perm],
            state_values=kw["state_values"][:, perm],
            obs_weights=kw["obs_weights"][perm],
        )
        a, b = self._run(kw), self._run(kb)
        np.testing.assert_allclose(b[0], a[0], rtol=1e-12)
        _close(b[1], a[1], rtol=1e-11)
        _close(b[2], a[2], rtol=1e-11)

    def test_state_relabelling(self) -> None:
        kw = self._kwargs()
        perm = [1, 2, 0]
        T_old = np.asarray(centered_softmax(jnp.asarray(kw["transition_logits"])))
        T_new = T_old[np.ix_(perm, perm)]
        logits_new = np.log(T_new[:, :-1]) - np.log(T_new[:, -1:])
        kb = dict(
            kw,
            reward_probs=kw["reward_probs"][perm],
            state_values=kw["state_values"][perm],
            transition_logits=logits_new,
            init_state_prob=kw["init_state_prob"][perm],
        )
        a, b = self._run(kw), self._run(kb)
        np.testing.assert_allclose(b[0], a[0], rtol=1e-12)
        _close(b[1], a[1][:, perm], rtol=1e-10)
        _close(b[2], a[2][:, perm][:, :, perm], rtol=1e-10)
        # guard: naively permuting the logits (ignoring the reference) is wrong
        kc = dict(kb, transition_logits=kw["transition_logits"][perm])
        assert abs(self._run(kc)[0] - a[0]) > 1e-3


class TestSmithStaticLimit:
    """Trial order is not a symmetry of the Smith model (the learning state
    is a random walk). With process noise 0 the model is static,
    x_k = x_0 ~ N(x0, P0), and the fitted curve must reduce to a static
    logistic (binomial) fit: the RTS gain is 1, so the smoothed state and
    variance are constant (to round-off) and equal to the final filtered
    ones. That constant is the sequential-Laplace approximation of the
    static MAP; it is not exactly the batch MAP (each step's Gaussian
    approximation of a non-Gaussian posterior carries forward), but the gap
    is a small fraction of a posterior sd and shrinks with the number of
    trials."""

    @staticmethod
    def _run(y, n, P0=1.0):
        out = smith_learning_filter(
            jnp.asarray(y),
            init_learning_state=0.0,
            init_learning_variance=P0,
            sigma_epsilon=0.0,
            prob_correct_by_chance=0.5,
            max_possible_correct=n,
        )
        sm, sv, _, _ = smith_learning_smoother(*out[1:])
        return np.asarray(sm), np.asarray(sv), np.asarray(out[1]), np.asarray(out[2])

    @staticmethod
    def _map(y, n, P0=1.0):
        def neg(x):
            p = expit(x)
            return -(np.sum(y * np.log(p) + (n - y) * np.log1p(-p)) - 0.5 * x**2 / P0)

        x = minimize_scalar(
            neg, bounds=(-10, 10), method="bounded", options={"xatol": 1e-12}
        ).x
        p = expit(x)
        return x, 1.0 / (n * len(y) * p * (1 - p) + 1.0 / P0)

    @pytest.mark.slow  # three Smith filter compiles (~5 s)
    def test_zero_process_noise_is_static_logistic_fit(self) -> None:
        rng = np.random.default_rng(0)
        gaps, var_gaps = {}, {}
        for T in (10, 640):
            y = rng.binomial(5, expit(0.7), T)
            sm, sv, fm, fv = self._run(y, 5)
            np.testing.assert_allclose(sm, fm[-1], rtol=0, atol=1e-14)
            np.testing.assert_allclose(sv, fv[-1], rtol=0, atol=1e-14)
            x_map, v_map = self._map(y, 5)
            gaps[T] = abs(sm[0] - x_map)
            assert gaps[T] < 0.1 * np.sqrt(v_map), (T, gaps[T], v_map)
            var_gaps[T] = abs(sv[0] / v_map - 1.0)
        # observed 0.062 (T=10) and 0.010 (T=640)
        assert var_gaps[10] < 0.15 and var_gaps[640] < 0.5 * var_gaps[10], var_gaps
        assert gaps[640] < 0.2 * gaps[10], gaps
        assert gaps[10] > 1e-4  # guard: the sequential approximation is visible
        # guard: with process noise the smoothed state is not constant
        out = smith_learning_filter(
            jnp.asarray(y), sigma_epsilon=0.3, max_possible_correct=5
        )
        sm_noisy = np.asarray(smith_learning_smoother(*out[1:])[0])
        assert np.ptp(sm_noisy) > 0.1


# ===========================================================================
# 4. Position decoder
# ===========================================================================


def _decoder_problem(seed=0):
    rng = np.random.default_rng(seed)
    x_edges, y_edges = np.linspace(0.0, 50.0, 51), np.linspace(0.0, 30.0, 31)
    gx, gy = np.meshgrid(x_edges, y_edges)
    centers = np.column_stack([rng.uniform(8, 42, 7), rng.uniform(6, 24, 7)])
    d2 = (gx[None] - centers[:, 0, None, None]) ** 2 + (
        gy[None] - centers[:, 1, None, None]
    ) ** 2
    rates = 1.0 + 30.0 * np.exp(-0.5 * d2 / 7.0**2)
    dt, q_pos, T = 0.02, 40.0, 30
    path = np.array([25.0, 15.0]) + np.cumsum(
        np.sqrt(q_pos * dt) * rng.normal(size=(T, 2)), 0
    )
    maps = PlaceFieldRateMaps(rates, x_edges, y_edges)
    spikes = rng.poisson(
        np.exp(np.asarray(jax.vmap(maps.log_rate)(jnp.asarray(path)))) * dt
    ).astype(float)
    return dict(
        rates=rates,
        x_edges=x_edges,
        y_edges=y_edges,
        spikes=spikes,
        dt=dt,
        q_pos=q_pos,
        init=np.array([25.0, 15.0]),
        init_cov=np.array([[4.0, 0.5], [0.5, 2.0]]),
    )


def _decode(pr, rates, x_edges, y_edges, init, init_cov):
    res = position_decoder_smoother(
        pr["spikes"],
        PlaceFieldRateMaps(rates, x_edges, y_edges),
        pr["dt"],
        q_pos=pr["q_pos"],
        include_velocity=False,
        init_position=init,
        init_cov=init_cov,
    )
    return (
        np.asarray(res.position_mean),
        np.asarray(res.position_cov),
        res.marginal_log_likelihood,
    )


@pytest.mark.slow  # decoder compile (~6 s)
class TestPositionDecoderInvariances:
    def test_rigid_translation(self) -> None:
        """Shifting the arena (grid edges), the start and hence every decoded
        position by d leaves the likelihood and covariances unchanged."""
        pr = _decoder_problem()
        d = np.array([13.7, -5.2])
        a = _decode(
            pr, pr["rates"], pr["x_edges"], pr["y_edges"], pr["init"], pr["init_cov"]
        )
        b = _decode(
            pr,
            pr["rates"],
            pr["x_edges"] + d[0],
            pr["y_edges"] + d[1],
            pr["init"] + d,
            pr["init_cov"],
        )
        np.testing.assert_allclose(b[2], a[2], rtol=1e-9)
        np.testing.assert_allclose(b[0], a[0] + d, rtol=0, atol=1e-8)
        _close(b[1], a[1], rtol=1e-7)
        assert pr["spikes"].sum() > 20  # guard

    def test_axis_swap(self) -> None:
        """Swapping x and y (rate maps transposed, edges / start / prior
        covariance swapped; the random-walk prior is isotropic)."""
        pr = _decoder_problem(1)
        S = np.array([[0.0, 1.0], [1.0, 0.0]])
        a = _decode(
            pr, pr["rates"], pr["x_edges"], pr["y_edges"], pr["init"], pr["init_cov"]
        )
        b = _decode(
            pr,
            np.swapaxes(pr["rates"], 1, 2),
            pr["y_edges"],
            pr["x_edges"],
            S @ pr["init"],
            S @ pr["init_cov"] @ S,
        )
        np.testing.assert_allclose(b[2], a[2], rtol=1e-10)
        _close(b[0], a[0][:, ::-1], rtol=1e-10)
        _close(b[1], S @ a[1] @ S, rtol=1e-9)
