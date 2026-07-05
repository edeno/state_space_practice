"""Covariance-hardening and input-validation checks for nonlinear_dynamics.

These complement ``test_nonlinear_dynamics_symplectic.py`` (which covers the
leapfrog symplecticity contract). Here we pin two behaviours:

- the EKF predict/smoother steps return *symmetric* covariances even when the
  incoming carry has drifted out of symmetry (as accumulated float roundoff
  produces over a long scan), and
- the leapfrog / MLP entry points reject odd-length states with a clear
  boundary error instead of a confusing reshape error deep inside ``grad``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.kalman import kalman_filter, kalman_smoother
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    ekf_predict_step,
    ekf_predict_step_with_jacobian,
    ekf_smooth_step,
    get_transition_jacobian,
    init_mlp_params,
    leapfrog_step,
)
from state_space_practice.utils import psd_solve


def _asymmetry(matrix: jax.Array) -> jax.Array:
    """Max absolute deviation from symmetry, ``max|A - A.T|``."""
    return jnp.max(jnp.abs(matrix - matrix.T))


def _params() -> dict:
    return init_mlp_params(input_dim=4, hidden_dims=[4], key=jax.random.PRNGKey(0))


# An asymmetric-but-diagonally-dominant covariance, standing in for a filtered
# covariance that has drifted out of symmetry over a long forward scan.
_DRIFTED_COV = jnp.array(
    [
        [2.0, 0.30, 0.10, 0.00],
        [0.20, 1.5, 0.00, 0.05],
        [0.10, 0.00, 1.2, 0.15],
        [0.00, 0.05, 0.25, 1.1],
    ]
)


def test_drifted_cov_fixture_is_actually_asymmetric():
    # Guard: the fixture must carry real asymmetry, otherwise the symmetrize
    # tests below would pass vacuously.
    assert _asymmetry(_DRIFTED_COV) > 1e-3


def test_ekf_predict_step_symmetrizes_drifted_covariance():
    params = _params()
    m = jnp.array([0.2, -0.3, 0.5, -0.1])
    Q = 0.01 * jnp.eye(4)

    # Guard: the raw (un-symmetrized) propagation is genuinely asymmetric, so
    # this test fails if the symmetrize() call is removed.
    F = get_transition_jacobian(m, params, apply_mlp, dt=0.1)
    raw = F @ _DRIFTED_COV @ F.T + Q
    assert _asymmetry(raw) > 1e-3

    _, P_pred = ekf_predict_step(m, _DRIFTED_COV, params, apply_mlp, Q, dt=0.1)

    assert _asymmetry(P_pred) < 1e-12


def test_ekf_smooth_step_symmetrizes_drifted_covariance():
    n = 4
    m_filt = jnp.zeros(n)
    P_filt = jnp.eye(n)
    m_pred_next = jnp.zeros(n)
    P_pred_next = _DRIFTED_COV
    m_smooth_next = jnp.zeros(n)
    P_smooth_next = jnp.eye(n)
    F_next = jnp.eye(n)

    # Guard: the raw smoother covariance update is genuinely asymmetric.
    G = psd_solve(P_pred_next, F_next @ P_filt).T
    raw = P_filt + G @ (P_smooth_next - P_pred_next) @ G.T
    assert _asymmetry(raw) > 1e-3

    _, P_smooth = ekf_smooth_step(
        m_filt,
        P_filt,
        m_pred_next,
        P_pred_next,
        m_smooth_next,
        P_smooth_next,
        F_next,
    )

    assert _asymmetry(P_smooth) < 1e-12


def test_leapfrog_step_rejects_odd_dimension():
    params = _params()
    x = jnp.array([0.1, 0.2, 0.3])  # odd length: cannot split into (q, p)
    with pytest.raises(ValueError, match="even"):
        leapfrog_step(x, params, apply_mlp, dt=0.1)


def test_apply_mlp_rejects_odd_dimension():
    params = _params()
    x = jnp.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="even"):
        apply_mlp(params, x)


def _zero_mlp_harmonic_params(dim: int, omega: float, key: jax.Array) -> dict:
    """Params for which ``apply_mlp`` reduces to H = p^2/2 + omega^2 q^2/2.

    Zeroing every MLP weight/bias makes the residual identically zero, leaving
    only the separable quadratic prior -- i.e. an exact linear harmonic
    oscillator, whose leapfrog flow has a closed form.
    """
    params = init_mlp_params(input_dim=dim, hidden_dims=[4], key=key)
    params = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {**params, "omega": jnp.array(omega)}


def test_leapfrog_tracks_harmonic_oscillator_analytic():
    """Zero-MLP leapfrog integrates H = p^2/2 + omega^2 q^2/2 to its closed form.

    Symplecticity (see ``test_nonlinear_dynamics_symplectic.py``) only pins that
    the Jacobian preserves the canonical form; it does *not* pin that leapfrog
    integrates the right dynamics -- a swapped q/p or a sign error in the
    kick/drift would remain symplectic yet trace the wrong trajectory. Comparing
    to the analytic harmonic flow catches that.
    """
    omega = 1.3
    dt = 0.01
    n_steps = 400
    params = _zero_mlp_harmonic_params(dim=2, omega=omega, key=jax.random.PRNGKey(0))

    q0, p0 = 1.0, 0.0
    x0 = jnp.array([q0, p0])

    def rollout(x, _):
        x_next = leapfrog_step(x, params, apply_mlp, dt)
        return x_next, x_next

    _, rest = jax.lax.scan(rollout, x0, None, length=n_steps)
    traj = np.asarray(jnp.concatenate([x0[None], rest], axis=0))  # (n_steps + 1, 2)

    t = np.arange(n_steps + 1) * dt
    q_true = q0 * np.cos(omega * t) + (p0 / omega) * np.sin(omega * t)
    p_true = p0 * np.cos(omega * t) - q0 * omega * np.sin(omega * t)

    # Guard: the analytic trajectory completes a full swing (q and p both go
    # strongly negative), so agreement is non-vacuous rather than "both ~constant".
    assert q_true.min() < -0.5
    assert p_true.min() < -0.9

    np.testing.assert_allclose(traj[:, 0], q_true, atol=1e-3)
    np.testing.assert_allclose(traj[:, 1], p_true, atol=1e-3)


@pytest.mark.slow
def test_ekf_backward_matches_linear_kalman_smoother():
    """On a linear (zero-MLP) system the EKF equals the exact RTS smoother.

    Zero-MLP leapfrog is an exact constant linear map F, so
    ``ekf_predict_step_with_jacobian`` + ``ekf_smooth_step`` must reproduce the
    trusted ``kalman.py`` RTS smoother. A wrong transition Jacobian or RTS gain
    would diverge from this linear-Gaussian reference -- pinning EKF-machinery
    *correctness*, which the symmetrization tests do not.
    """
    dim = 4
    omega = 1.5
    dt = 0.05
    params = _zero_mlp_harmonic_params(dim=dim, omega=omega, key=jax.random.PRNGKey(3))

    F = get_transition_jacobian(jnp.zeros(dim), params, apply_mlp, dt)
    Q = 0.02 * jnp.eye(dim)
    H = jnp.eye(dim)
    R = 0.1 * jnp.eye(dim)
    init_mean = jnp.zeros(dim)
    init_cov = jnp.eye(dim)

    # Sanity: the zero-MLP leapfrog really is the linear map f(x) = F @ x, so the
    # EKF is exact here and comparison to the linear Kalman smoother is valid.
    x_probe = jnp.array([0.3, -0.4, 0.2, 0.6])
    np.testing.assert_allclose(
        np.asarray(leapfrog_step(x_probe, params, apply_mlp, dt)),
        np.asarray(F @ x_probe),
        atol=1e-10,
    )

    # Simulate a linear-Gaussian sequence x_k = F x_{k-1} + w, y_k = x_k + v.
    chol_Q = jnp.linalg.cholesky(Q)
    chol_R = jnp.linalg.cholesky(R)

    def sim_step(x_prev, k):
        kx, ky = jax.random.split(k)
        x_next = F @ x_prev + chol_Q @ jax.random.normal(kx, (dim,))
        obs_k = x_next + chol_R @ jax.random.normal(ky, (dim,))
        return x_next, obs_k

    n_time = 25
    step_keys = jax.random.split(jax.random.PRNGKey(11), n_time)
    _, obs = jax.lax.scan(sim_step, init_mean, step_keys)

    # Trusted reference smoother (returns means, covs, cross-covs, log-likelihood).
    ref_means, ref_covs, _, _ = kalman_smoother(init_mean, init_cov, obs, F, Q, H, R)

    # EKF path: the trusted forward filter (a linear system => the EKF filter is
    # identical), then an RTS backward pass built only from this module's steps.
    filt_means, filt_covs, _ = kalman_filter(init_mean, init_cov, obs, F, Q, H, R)

    ekf_means: list = [None] * n_time
    ekf_covs: list = [None] * n_time
    ekf_means[-1] = filt_means[-1]
    ekf_covs[-1] = filt_covs[-1]
    for k in range(n_time - 2, -1, -1):
        m_pred, P_pred, F_next = ekf_predict_step_with_jacobian(
            filt_means[k], filt_covs[k], params, apply_mlp, Q, dt
        )
        ekf_means[k], ekf_covs[k] = ekf_smooth_step(
            filt_means[k],
            filt_covs[k],
            m_pred,
            P_pred,
            ekf_means[k + 1],
            ekf_covs[k + 1],
            F_next,
        )
    ekf_means = jnp.stack(ekf_means)
    ekf_covs = jnp.stack(ekf_covs)

    # Guard: smoothing actually moved the estimate off the filtered mean, so an
    # exact match to the reference is a real agreement, not a no-op.
    assert float(jnp.max(jnp.abs(ekf_means - filt_means))) > 1e-3

    np.testing.assert_allclose(np.asarray(ekf_means), np.asarray(ref_means), atol=1e-8)
    np.testing.assert_allclose(np.asarray(ekf_covs), np.asarray(ref_covs), atol=1e-8)
