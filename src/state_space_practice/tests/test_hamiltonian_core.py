"""Tests for shared Hamiltonian EKF/Laplace helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from state_space_practice import hamiltonian_core
from state_space_practice.hamiltonian_core import (
    HamiltonianModelBase,
    ekf_rts_backward_pass,
    gaussian_measurement_update,
    hamiltonian_ekf_filter,
    hamiltonian_ekf_smoother,
    point_process_laplace_update,
)
from state_space_practice.hamiltonian_joint import JointHamiltonianModel
from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel
from state_space_practice.hamiltonian_spikes import HamiltonianSpikeModel
from state_space_practice.hamiltonian_switching import SwitchingHamiltonianJointModel
from state_space_practice.oscillator_models import BaseModel, OscillatorParameterBase
from state_space_practice.point_process_kalman import glm_laplace_update, poisson_family
from state_space_practice.sgd_fitting import SGDFittableMixin


def test_point_process_laplace_covariance_matches_information_form():
    m_pred = jnp.array([0.2, -0.4])
    P_pred = jnp.array([[3.0, 1.2], [1.2, 1.0]])
    y = jnp.array([0.0, 1.0, 2.0])
    C = jnp.array([[1.0, 2.0], [-0.3, 1.5], [2.0, -1.0]])
    d = jnp.array([0.1, -0.2, 0.3])
    dt = 0.1

    _, P_post, _ = point_process_laplace_update(m_pred, P_pred, y, C, d, dt)

    rate_pred = jnp.exp(C @ m_pred + d) * dt
    H_lik = C.T @ (rate_pred[:, None] * C)
    # Independent reference: the Laplace posterior covariance is the information
    # form (P_pred^-1 + H_lik)^-1, computed here via explicit inverses so the
    # test does not merely mirror the implementation's psd_solve expression.
    expected = jnp.linalg.inv(jnp.linalg.inv(P_pred) + H_lik)

    assert jnp.allclose(P_post, expected, rtol=1e-6, atol=1e-6)


def test_point_process_laplace_matches_glm_poisson_for_counts_above_one():
    m_pred = jnp.array([0.2, -0.4, 0.1])
    P_pred = jnp.array(
        [
            [1.5, 0.2, -0.1],
            [0.2, 0.9, 0.15],
            [-0.1, 0.15, 1.2],
        ]
    )
    y = jnp.array([0.0, 2.0, 5.0, 3.0])
    C = jnp.array(
        [
            [1.0, 0.3, -0.2],
            [-0.3, 1.5, 0.4],
            [0.2, -0.1, 1.1],
            [1.2, -0.4, 0.5],
        ]
    )
    d = jnp.array([0.1, -0.2, 0.3, -0.4])
    dt = 0.05

    actual = point_process_laplace_update(m_pred, P_pred, y, C, d, dt)
    expected = glm_laplace_update(
        m_pred,
        P_pred,
        y,
        lambda x: C @ x + d,
        poisson_family(dt),
        grad_eta_func=lambda _x: C,
    )

    for actual_arr, expected_arr in zip(actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual_arr),
            np.asarray(expected_arr),
            rtol=1e-10,
            atol=1e-10,
        )


def test_point_process_laplace_overflow_path_returns_finite_values():
    m_pred = jnp.array([1000.0, -1000.0])
    P_pred = jnp.array([[2.0, 0.1], [0.1, 1.5]])
    y = jnp.array([1.0, 4.0])
    C = jnp.array([[1.0, 0.2], [0.3, -0.8]])
    d = jnp.array([500.0, 700.0])
    dt = 0.1

    m_post, P_post, ll = point_process_laplace_update(m_pred, P_pred, y, C, d, dt)

    assert bool(jnp.all(jnp.isfinite(m_post)))
    assert bool(jnp.all(jnp.isfinite(P_post)))
    assert bool(jnp.isfinite(ll))


def test_point_process_laplace_compute_log_likelihood_false_keeps_update():
    m_pred = jnp.array([0.2, -0.4])
    P_pred = jnp.array([[1.3, 0.2], [0.2, 0.7]])
    y = jnp.array([0.0, 3.0])
    C = jnp.array([[1.0, -0.5], [0.2, 0.8]])
    d = jnp.array([0.1, -0.2])
    dt = 0.1

    m_expected, P_expected, _ = point_process_laplace_update(
        m_pred,
        P_pred,
        y,
        C,
        d,
        dt,
    )
    m_actual, P_actual, ll = point_process_laplace_update(
        m_pred,
        P_pred,
        y,
        C,
        d,
        dt,
        compute_log_likelihood=False,
    )

    np.testing.assert_allclose(
        np.asarray(m_actual), np.asarray(m_expected), rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(P_actual), np.asarray(P_expected), rtol=1e-10, atol=1e-10
    )
    assert float(ll) == 0.0


def test_point_process_laplace_can_be_jitted_with_static_configuration():
    m_pred = jnp.array([0.2, -0.4])
    P_pred = jnp.array([[1.3, 0.2], [0.2, 0.7]])
    y = jnp.array([0.0, 3.0])
    C = jnp.array([[1.0, -0.5], [0.2, 0.8]])
    d = jnp.array([0.1, -0.2])
    dt = 0.1

    expected = point_process_laplace_update(m_pred, P_pred, y, C, d, dt)
    jitted_update = jax.jit(
        point_process_laplace_update,
        static_argnames=("dt", "compute_log_likelihood"),
    )
    actual = jitted_update(m_pred, P_pred, y, C, d, dt=dt)

    for actual_arr, expected_arr in zip(actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual_arr),
            np.asarray(expected_arr),
            rtol=1e-10,
            atol=1e-10,
        )


def test_gaussian_measurement_update_matches_independent_reference():
    m_pred = jnp.array([0.5, -1.0, 0.2])
    prior_root = jnp.array([[1.2, 0.3, -0.2], [0.3, 0.9, 0.1], [-0.2, 0.1, 1.5]])
    P_pred = prior_root @ prior_root.T
    C = jnp.array([[1.0, 0.5, -0.3], [0.2, -0.8, 1.1]])
    d = jnp.array([0.1, -0.4])
    R = jnp.array([[0.7, 0.1], [0.1, 0.5]])
    y = jnp.array([0.3, -0.6])

    m_post, P_post, ll = gaussian_measurement_update(m_pred, P_pred, y, C, d, R)

    # Independent reference: textbook Kalman update via explicit inverses, and
    # the Gaussian marginal log-likelihood from scipy (not the module's split
    # quadratic-form/logdet expression).
    S = np.asarray(C @ P_pred @ C.T + R)
    K = np.asarray(P_pred @ C.T) @ np.linalg.inv(S)
    innovation = np.asarray(y - (C @ m_pred + d))
    m_ref = np.asarray(m_pred) + K @ innovation
    P_ref = (np.eye(3) - K @ np.asarray(C)) @ np.asarray(P_pred)
    ll_ref = multivariate_normal.logpdf(
        np.asarray(y), mean=np.asarray(C @ m_pred + d), cov=S
    )

    np.testing.assert_allclose(np.asarray(m_post), m_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(P_post), P_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(ll), float(ll_ref), rtol=1e-6, atol=1e-6)

    # Dropping the normalization constant adds back exactly n_obs/2 * log(2π)
    # (the constant lowers the normalized log-density).
    _, _, ll_no_const = gaussian_measurement_update(
        m_pred, P_pred, y, C, d, R, include_normalization_const=False
    )
    expected_gap = 0.5 * y.shape[0] * np.log(2 * np.pi)
    np.testing.assert_allclose(
        float(ll_no_const) - float(ll), expected_gap, rtol=1e-6, atol=1e-6
    )


def test_gaussian_measurement_update_with_no_observations_is_identity():
    m_pred = jnp.array([0.5, -1.0])
    P_pred = jnp.array([[1.2, 0.3], [0.3, 0.9]])
    y = jnp.empty((0,), dtype=m_pred.dtype)
    C = jnp.empty((0, 2), dtype=m_pred.dtype)
    d = jnp.empty((0,), dtype=m_pred.dtype)
    R = jnp.empty((0, 0), dtype=m_pred.dtype)

    m_post, P_post, ll = jax.jit(gaussian_measurement_update)(
        m_pred, P_pred, y, C, d, R
    )

    np.testing.assert_array_equal(np.asarray(m_post), np.asarray(m_pred))
    np.testing.assert_array_equal(np.asarray(P_post), np.asarray(P_pred))
    assert float(ll) == 0.0


def test_ekf_rts_backward_pass_alignment_matches_textbook_recursion():
    """Lock the F[t+1]/m_filt[t] alignment against an independent RTS recursion.

    Uses a *time-varying* F so a one-step index shift (e.g. F[:-1] instead of
    F[1:]) would change every smoother gain and fail the comparison — the exact
    silent-misalignment failure the smoother convention is meant to prevent.
    """
    rng = np.random.default_rng(0)
    T, n = 6, 3

    def rand_psd():
        root = rng.standard_normal((n, n))
        return root @ root.T + n * np.eye(n)

    m_filt = rng.standard_normal((T, n))
    P_filt = np.stack([rand_psd() for _ in range(T)])
    m_pred = rng.standard_normal((T, n))
    P_pred = np.stack([rand_psd() for _ in range(T)])
    F = rng.standard_normal((T, n, n))

    m_s, P_s = ekf_rts_backward_pass(
        jnp.asarray(m_filt),
        jnp.asarray(P_filt),
        jnp.asarray(m_pred),
        jnp.asarray(P_pred),
        jnp.asarray(F),
    )

    # Independent textbook RTS backward recursion with explicit gains.
    m_ref = m_filt.astype(float).copy()
    P_ref = P_filt.astype(float).copy()
    for t in range(T - 2, -1, -1):
        gain = P_filt[t] @ F[t + 1].T @ np.linalg.inv(P_pred[t + 1])
        m_ref[t] = m_filt[t] + gain @ (m_ref[t + 1] - m_pred[t + 1])
        P_ref[t] = P_filt[t] + gain @ (P_ref[t + 1] - P_pred[t + 1]) @ gain.T

    np.testing.assert_allclose(np.asarray(m_s), m_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(P_s), P_ref, rtol=1e-6, atol=1e-6)
    # Terminal step is the un-smoothed filter posterior.
    np.testing.assert_allclose(np.asarray(m_s[-1]), m_filt[-1])
    np.testing.assert_allclose(np.asarray(P_s[-1]), P_filt[-1])


def test_ekf_rts_backward_pass_returns_empty_trajectory_for_zero_steps():
    n = 3
    m_filt = jnp.empty((0, n))
    P_filt = jnp.empty((0, n, n))
    m_pred = jnp.empty((0, n))
    P_pred = jnp.empty((0, n, n))
    F = jnp.empty((0, n, n))

    m_smooth, P_smooth = jax.jit(ekf_rts_backward_pass)(
        m_filt, P_filt, m_pred, P_pred, F
    )

    assert m_smooth.shape == (0, n)
    assert P_smooth.shape == (0, n, n)


@pytest.mark.parametrize(
    "model_cls",
    [
        HamiltonianLFPModel,
        HamiltonianSpikeModel,
        JointHamiltonianModel,
        SwitchingHamiltonianJointModel,
    ],
)
def test_hamiltonian_models_do_not_inherit_the_em_layer(model_cls):
    """SGD-only family: parameter base + SGDFittableMixin, no EM hooks or fit()."""
    # Guard: BaseModel is still the EM layer, so the absence checks below mean
    # something.
    assert BaseModel.__abstractmethods__, "BaseModel declares no abstract hooks"
    assert hasattr(BaseModel, "fit") and hasattr(BaseModel, "_e_step")

    assert issubclass(model_cls, HamiltonianModelBase)
    assert issubclass(model_cls, OscillatorParameterBase)
    assert issubclass(model_cls, SGDFittableMixin)
    assert not issubclass(model_cls, BaseModel)
    for name in ("fit", "_e_step", "_m_step", *BaseModel.__abstractmethods__):
        assert not hasattr(model_cls, name), f"{model_cls.__name__} exposes {name}"


def _lfp_case(seed, sampling_freq):
    model = HamiltonianLFPModel(
        n_oscillators=1,
        n_sources=2,
        sampling_freq=sampling_freq,
        hidden_dims=[4],
        seed=seed,
    )
    return model, (jax.random.normal(jax.random.PRNGKey(3), (6, 2)),)


def _spike_case(seed, sampling_freq):
    model = HamiltonianSpikeModel(
        n_oscillators=1,
        n_sources=3,
        sampling_freq=sampling_freq,
        hidden_dims=[4],
        seed=seed,
    )
    return model, (jax.random.poisson(jax.random.PRNGKey(3), 0.5, (6, 3)),)


def _joint_case(seed, sampling_freq):
    model = JointHamiltonianModel(
        n_oscillators=1,
        n_lfp_sources=2,
        n_spike_sources=3,
        sampling_freq=sampling_freq,
        hidden_dims=[4],
        seed=seed,
    )
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    return model, (
        jax.random.normal(k1, (6, 2)),
        jax.random.poisson(k2, 0.5, (6, 3)),
    )


@pytest.mark.parametrize("make_case", [_lfp_case, _spike_case, _joint_case])
def test_jitted_cores_are_shared_across_model_instances(make_case, monkeypatch):
    """A second instance with the same shapes and dt reuses the compilation.

    The cores are module-level ``jax.jit`` functions keyed on array shapes plus
    the static ``dt`` / observation-model name, not on the model object. A
    never-seen ``dt`` must trace again, which keeps the reuse check from being
    vacuous.

    Compilations are observed as traces (``_observation_updates`` runs only
    while a core is traced), not as absolute ``_cache_size()`` counts: those
    depend on what earlier tests compiled, and JAX's jit cache is a global
    LRU whose evictions can hide a new entry in a long run.
    """
    traces: list = []
    original = hamiltonian_core._observation_updates

    def counting(*args, **kwargs):
        traces.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(hamiltonian_core, "_observation_updates", counting)
    # Sampling rates no other test uses, so the first run must compile (the
    # parametrized cases differ in the static observation-model name).
    sampling_freq, other_sampling_freq = 7919.0, 7927.0

    def run(model, data):
        params = model._build_param_spec()[0]
        model.filter(*data, params)
        model.smooth(*data, params)

    run(*make_case(seed=0, sampling_freq=sampling_freq))
    n_filter = hamiltonian_ekf_filter._cache_size()
    n_smooth = hamiltonian_ekf_smoother._cache_size()
    # Guard: a dt no earlier test used really was compiled here.
    assert len(traces) == 2  # filter + smoother

    traces.clear()
    run(*make_case(seed=1, sampling_freq=sampling_freq))
    assert traces == []
    assert hamiltonian_ekf_filter._cache_size() == n_filter
    assert hamiltonian_ekf_smoother._cache_size() == n_smooth

    run(*make_case(seed=0, sampling_freq=other_sampling_freq))
    assert len(traces) == 2


@pytest.mark.parametrize("make_case", [_lfp_case, _spike_case, _joint_case])
def test_single_regime_models_reject_discrete_state_decoding(make_case):
    """A single-regime model has no discrete-state posterior to decode, so after
    a completed fit decode/predict_proba must say that (and point at the
    switching model) rather than ask for a fit that already happened."""
    model, data = make_case(seed=0, sampling_freq=200.0)
    model._finalize_sgd(*data)
    # Guard: the fit's finalize step actually populated the smoother output.
    assert bool(jnp.all(jnp.isfinite(model.smoothed_means_)))

    for method in (model.decode, model.predict_proba):
        with pytest.raises(NotImplementedError, match="SwitchingHamiltonianJointModel"):
            method()


def test_unfitted_decode_message_does_not_name_a_missing_fit_method():
    """The Hamiltonian family has no ``fit``; the not-yet-fitted error inherited
    from the parameter base must not tell the user to call it."""
    model = SwitchingHamiltonianJointModel(
        n_oscillators=1,
        n_discrete_states=2,
        n_lfp_sources=2,
        n_spike_sources=3,
        sampling_freq=100.0,
        hidden_dims=[4],
        seed=0,
    )
    assert not hasattr(model, "fit")
    for method in (model.decode, model.predict_proba):
        with pytest.raises(RuntimeError, match="No smoother posteriors") as excinfo:
            method()
        assert "fit()" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# EKF vs exact / Monte Carlo references on a one-oscillator problem
# ---------------------------------------------------------------------------


class _OneOscillator:
    """A 2-D (q, p) Hamiltonian SSM with a clearly non-quadratic potential.

    The MLP weights are scaled x3 so H(q, 0) is asymmetric (H(-1.5) = 7.3,
    H(1.5) = 1.7) and the leapfrog map is visibly nonlinear on the prior's
    support.
    """

    dt = 0.1
    omega = 2.0
    n_time = 12

    def __init__(self, linear=False):
        from state_space_practice.nonlinear_dynamics import (
            apply_mlp,
            init_mlp_params,
            leapfrog_step,
        )

        mlp = init_mlp_params(1, [8], jax.random.PRNGKey(3))
        scale = 0.0 if linear else 3.0
        self.mlp = {k: (scale * v if k.startswith("w") else v) for k, v in mlp.items()}
        trans = {**self.mlp, "omega": self.omega}
        self.step = jax.jit(
            jax.vmap(lambda x: leapfrog_step(x, trans, apply_mlp, self.dt))
        )
        self.Q = np.diag([0.01, 0.01])
        self.m0 = np.array([0.8, 0.0])
        self.P0 = 0.05 * np.eye(2)
        self.C_lfp = np.array([[1.0, 0.0]])
        self.R = np.array([[0.01]])
        self.C_spk = np.array([[1.5, 0.0], [-1.5, 0.0]])
        self.d_spk = np.full(2, 1.0 + np.log(5.0))

    def params(self, observation_model):
        base = dict(
            mlp=self.mlp,
            omega=self.omega,
            init_mean=jnp.asarray(self.m0),
            init_cov=jnp.asarray(self.P0),
            Q=jnp.asarray(self.Q),
        )
        if observation_model == "gaussian":
            return {
                **base,
                "C": jnp.asarray(self.C_lfp),
                "d": jnp.zeros(1),
                "R": jnp.asarray(self.R),
            }
        return {**base, "C": jnp.asarray(self.C_spk), "d": jnp.asarray(self.d_spk)}

    def simulate(self, observation_model, rng, n_rep=1):
        x = self.m0 + rng.multivariate_normal(np.zeros(2), self.P0, n_rep)
        xs, ys = [], []
        for _ in range(self.n_time):
            x = np.asarray(self.step(jnp.asarray(x)))
            x = x + rng.multivariate_normal(np.zeros(2), self.Q, n_rep)
            xs.append(x)
            if observation_model == "gaussian":
                ys.append(x @ self.C_lfp.T + 0.1 * rng.normal(size=(n_rep, 1)))
            else:
                rate = np.exp(x @ self.C_spk.T + self.d_spk) * self.dt
                ys.append(rng.poisson(rate).astype(float))
        return np.stack(xs, 1), np.stack(ys, 1)  # (n_rep, T, 2), (n_rep, T, n_obs)

    def log_lik(self, observation_model, X, y):
        if observation_model == "gaussian":
            r = y[0] - X[:, 0]
            return -0.5 * r**2 / self.R[0, 0] - 0.5 * np.log(2 * np.pi * self.R[0, 0])
        from scipy.special import gammaln

        lam = np.exp(X @ self.C_spk.T + self.d_spk) * self.dt
        return np.sum(y * np.log(lam) - lam - gammaln(y + 1), axis=1)

    def particle_filter(self, observation_model, ys, n_particles, seed, keep=False):
        """Bootstrap PF: weighted moments before resampling, log-evidence."""
        rng = np.random.default_rng(seed)
        X = self.m0 + rng.multivariate_normal(np.zeros(2), self.P0, n_particles)
        L = np.linalg.cholesky(self.Q)
        means, covs, store, log_ev = [], [], [], 0.0
        for t in range(self.n_time):
            X = (
                np.asarray(self.step(jnp.asarray(X)))
                + rng.normal(size=(n_particles, 2)) @ L.T
            )
            lw = self.log_lik(observation_model, X, ys[t])
            w = np.exp(lw - lw.max())
            log_ev += lw.max() + np.log(w.mean())
            w /= w.sum()
            mu = w @ X
            means.append(mu)
            covs.append(((X - mu).T * w) @ (X - mu))
            if keep:
                store.append((X.copy(), w.copy()))
            X = X[rng.choice(n_particles, n_particles, p=w)]
        return np.array(means), np.array(covs), log_ev, store

    def ffbsm(self, store):
        """Forward-filtering backward-smoothing marginal reweighting (O(N^2))."""
        Qi = np.linalg.inv(self.Q)
        X, w = store[-1]
        means, covs = [None] * self.n_time, [None] * self.n_time
        means[-1] = w @ X
        covs[-1] = ((X - means[-1]).T * w) @ (X - means[-1])
        ws = w
        for t in range(self.n_time - 2, -1, -1):
            Xt, wt = store[t]
            Xn = store[t + 1][0]
            diff = Xn[None] - np.asarray(self.step(jnp.asarray(Xt)))[:, None]
            logk = -0.5 * np.einsum("ijk,kl,ijl->ij", diff, Qi, diff)
            K = np.exp(logk - logk.max())
            ws = wt * (K @ (ws / (wt @ K)))
            ws /= ws.sum()
            means[t] = ws @ Xt
            covs[t] = ((Xt - means[t]).T * ws) @ (Xt - means[t])
        return np.array(means), np.array(covs)


def test_ekf_is_exact_kalman_filter_for_linear_gaussian_case():
    """Zero MLP output: leapfrog is linear, so the EKF must be exact.

    The reference is the joint Gaussian density of the stacked observations
    ``y_{1:T}`` (built from the linear map, no recursion), and an explicit
    information-form posterior of x_T.
    """
    osc = _OneOscillator(linear=True)
    _, ys = osc.simulate("gaussian", np.random.default_rng(0))
    ys = ys[0]
    A = np.asarray(
        jax.jacfwd(lambda x: osc.step(x[None])[0])(jnp.zeros(2))
    )  # exact linear map
    T = osc.n_time
    # x_t = A^{t+1} x_0 + sum_{s<=t} A^{t-s} w_s  (w_s ~ N(0, Q))
    powers = [np.linalg.matrix_power(A, k) for k in range(T + 1)]
    mean_x = np.stack([powers[t + 1] @ osc.m0 for t in range(T)])
    cov_x = np.zeros((T, T, 2, 2))
    for t in range(T):
        for u in range(T):
            c = powers[t + 1] @ osc.P0 @ powers[u + 1].T
            for s in range(min(t, u) + 1):
                c = c + powers[t - s] @ osc.Q @ powers[u - s].T
            cov_x[t, u] = c
    C = osc.C_lfp
    mean_y = mean_x @ C.T
    cov_y = np.einsum("ab,tubc,dc->tuad", C, cov_x, C).reshape(T, T) + osc.R[
        0, 0
    ] * np.eye(T)
    exact_ll = multivariate_normal(mean_y[:, 0], cov_y).logpdf(ys[:, 0])
    # Exact posterior of x_T given y_{1:T} by Gaussian conditioning.
    # Cov(x_T, y_t) = Cov(x_T, x_t) C^T = cov_x[t, T-1]^T C^T.
    cross = np.einsum("tba,cb->tac", cov_x[:, T - 1], C)[:, :, 0].T  # (2, T)
    gain = cross @ np.linalg.inv(cov_y)
    post_mean = mean_x[-1] + gain @ (ys[:, 0] - mean_y[:, 0])
    post_cov = cov_x[T - 1, T - 1] - gain @ cross.T

    means, covs, lls = hamiltonian_ekf_filter(
        jnp.asarray(ys), osc.params("gaussian"), dt=osc.dt, observation_model="gaussian"
    )
    np.testing.assert_allclose(float(jnp.sum(lls)), exact_ll, atol=1e-8)
    np.testing.assert_allclose(means[-1], post_mean, atol=1e-8)
    np.testing.assert_allclose(covs[-1], post_cov, atol=1e-8)
    # Guard: the data moved the posterior away from the prior.
    assert np.linalg.norm(post_mean - mean_x[-1]) > 0.05


@pytest.mark.slow
@pytest.mark.parametrize(
    "observation_model, pins",
    [
        # (filter mean, smoother mean) errors in posterior sd, LL error.
        # Observed: gaussian 0.035 / 0.059 / 0.007, poisson 0.138 / 0.159 / 0.015.
        ("gaussian", (0.15, 0.2, 0.25)),
        ("poisson", (0.35, 0.45, 0.25)),
    ],
)
def test_ekf_against_particle_filter_and_smoother(observation_model, pins):
    """EKF / Laplace-EKF filter and RTS smoother vs a many-particle reference.

    Reference: bootstrap PF with 40k particles (filter moments, evidence) and
    FFBSm with 2k particles (smoother moments). For the Gaussian LFP readout
    the EKF error is at the Monte Carlo noise level; the Poisson readout's
    single-Fisher-step Laplace update leaves a measurable mean error, which
    is pinned from both sides.
    """
    osc = _OneOscillator()
    _, ys = osc.simulate(observation_model, np.random.default_rng(0))
    ys = ys[0]
    params = osc.params(observation_model)
    fm, fc, lls = hamiltonian_ekf_filter(
        jnp.asarray(ys), params, dt=osc.dt, observation_model=observation_model
    )
    sm, sc = hamiltonian_ekf_smoother(
        jnp.asarray(ys), params, dt=osc.dt, observation_model=observation_model
    )
    pm, pc, p_ll, _ = osc.particle_filter(observation_model, ys, 40_000, seed=1)
    _, _, _, store = osc.particle_filter(
        observation_model, ys, 2_000, seed=2, keep=True
    )
    rm, rc = osc.ffbsm(store)

    sd_f = np.sqrt(np.einsum("tii->ti", pc))
    sd_s = np.sqrt(np.einsum("tii->ti", rc))
    err_f = np.max(np.abs(np.asarray(fm) - pm) / sd_f)
    err_s = np.max(np.abs(np.asarray(sm) - rm) / sd_s)
    ratio_f = np.sqrt(np.einsum("tii->ti", np.asarray(fc))) / sd_f
    ratio_s = np.sqrt(np.einsum("tii->ti", np.asarray(sc))) / sd_s
    err_ll = abs(float(jnp.sum(lls)) - p_ll)
    msg = (
        f"{observation_model}: filter mean {err_f:.3f} sd, smoother mean "
        f"{err_s:.3f} sd, filter sd ratio [{ratio_f.min():.3f}, "
        f"{ratio_f.max():.3f}], smoother sd ratio [{ratio_s.min():.3f}, "
        f"{ratio_s.max():.3f}], log-evidence {err_ll:.3f}"
    )
    assert err_f < pins[0], msg
    assert err_s < pins[1], msg
    assert err_ll < pins[2], msg
    assert np.all((ratio_f > 0.9) & (ratio_f < 1.1)), msg
    assert np.all((ratio_s > 0.85) & (ratio_s < 1.15)), msg
    if observation_model == "poisson":
        # Above the ~0.02-sd Monte Carlo error of the 40k-particle filter.
        assert err_f > 0.05, msg
    # The smoother uses future data: it is closer to the reference smoother
    # than the filter is (excluding the shared last step).
    assert np.linalg.norm(np.asarray(sm)[:-1] - rm[:-1]) < np.linalg.norm(
        np.asarray(fm)[:-1] - rm[:-1]
    ), msg


@pytest.mark.slow
def test_ekf_smoother_is_calibrated_on_simulated_replicates():
    """Pooled standardised smoother errors at the true parameters ~ N(0, 1).

    60 replicates of the nonlinear one-oscillator LFP model; the z-scores of
    both coordinates at every time step are pooled (720 values, correlated
    within a replicate, so the bands allow for an effective sample of ~200).
    """
    osc = _OneOscillator()
    xs, ys = osc.simulate("gaussian", np.random.default_rng(11), n_rep=60)
    params = osc.params("gaussian")
    smooth = jax.vmap(
        lambda y: hamiltonian_ekf_smoother(
            y, params, dt=osc.dt, observation_model="gaussian"
        )
    )
    sm, sc = smooth(jnp.asarray(ys))
    z = (xs - np.asarray(sm)) / np.sqrt(np.einsum("rtii->rti", np.asarray(sc)))
    z = z.ravel()
    coverage = np.mean(np.abs(z) < 1.6449)
    msg = f"mean {z.mean():.3f}, var {z.var():.3f}, 90% coverage {coverage:.3f}"
    assert abs(z.mean()) < 0.15, msg
    assert 0.8 < z.var() < 1.25, msg
    assert 0.85 < coverage < 0.95, msg


@pytest.mark.slow
def test_switching_discrete_posterior_is_exact_for_identical_regimes():
    """Identical per-state dynamics: every per-step approximation is exact.

    All regime-conditional Gaussians coincide, so the Kim filter's collapse is
    exact and every discrete path has the same likelihood. Exact enumeration
    over all 2^T paths then gives the Markov chain's prior marginals, and the
    evidence equals the single-regime joint model's. (Like the continuous
    state, ``init_pi`` is the regime distribution one transition *before* the
    first observation.)
    """
    import itertools

    from state_space_practice.hamiltonian_switching import (
        switching_hamiltonian_filter,
        switching_hamiltonian_smoother,
    )

    rng = np.random.default_rng(4)
    T = 6
    model = SwitchingHamiltonianJointModel(
        n_oscillators=1,
        n_discrete_states=2,
        n_lfp_sources=2,
        n_spike_sources=2,
        sampling_freq=20.0,
        hidden_dims=[4],
    )
    params = model._complete_filter_params(model._build_param_spec()[0])
    # Make the two regimes identical (state 1 copies state 0).
    params["mlp"] = jax.tree_util.tree_map(
        lambda v: jnp.stack([v[0], v[0]]), params["mlp"]
    )
    params["omega"] = jnp.stack([params["omega"][0]] * 2)
    params["Q"] = jnp.stack([params["Q"][..., 0]] * 2, axis=-1)
    Z = jnp.array([[0.8, 0.2], [0.3, 0.7]])
    params["Z"] = Z
    params["init_pi"] = jnp.array([0.9, 0.1])
    lfp = 0.1 * rng.normal(size=(T, 2))
    spikes = rng.poisson(0.5, size=(T, 2)).astype(float)
    obs = (jnp.asarray(lfp), jnp.asarray(spikes))
    means, covs, probs, lls = switching_hamiltonian_filter(obs, params, dt=model.dt)

    # Enumeration with a path-independent likelihood.
    Zn, pi0 = np.asarray(Z), np.array([0.9, 0.1])
    filt = np.zeros((T, 2))
    for t in range(T):
        # Paths (s_0, s_1, ..., s_{t+1}); s_0 ~ init_pi is never observed.
        for path in itertools.product(range(2), repeat=t + 2):
            p = pi0[path[0]] * np.prod(
                [Zn[path[i - 1], path[i]] for i in range(1, t + 2)]
            )
            filt[t, path[-1]] += p
    np.testing.assert_allclose(probs, filt, atol=1e-12)
    np.testing.assert_allclose(means[..., 0], means[..., 1], atol=1e-12)

    single = JointHamiltonianModel(
        n_oscillators=1,
        n_lfp_sources=2,
        n_spike_sources=2,
        sampling_freq=20.0,
        hidden_dims=[4],
    )
    single_params = {
        "mlp": jax.tree_util.tree_map(lambda v: v[0], params["mlp"]),
        "omega": params["omega"][0],
        "C_lfp": params["C_lfp"],
        "d_lfp": params["d_lfp"],
        "R_lfp": params["R_lfp"],
        "C_spikes": params["C_spikes"],
        "d_spikes": params["d_spikes"],
        "init_mean": params["init_mean"][..., 0],
        "init_cov": params["init_cov"][..., 0],
        "Q": params["Q"][..., 0],
    }
    _, _, single_lls = hamiltonian_ekf_filter(
        obs, single_params, dt=single.dt, observation_model="joint"
    )
    np.testing.assert_allclose(
        float(jnp.sum(lls)), float(jnp.sum(single_lls)), atol=1e-9
    )

    smoothed = switching_hamiltonian_smoother(obs, params, dt=model.dt)
    sm_probs = np.asarray(smoothed[2]) if len(smoothed) > 2 else None
    assert sm_probs is not None
    np.testing.assert_allclose(sm_probs, filt, atol=1e-10)
