"""Tests for shared Hamiltonian EKF/Laplace helpers."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal

from state_space_practice.hamiltonian_core import (
    _BaseModelStubs,
    ekf_rts_backward_pass,
    gaussian_measurement_update,
    point_process_laplace_update,
)
from state_space_practice.point_process_kalman import glm_laplace_update, poisson_family


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


def test_basemodel_stubs_cover_all_abstract_hooks():
    """_BaseModelStubs must stub every BaseModel abstract hook (catches renames)."""
    from state_space_practice.oscillator_models import BaseModel

    stubbed = {name for name in vars(_BaseModelStubs) if not name.startswith("__")}
    # Guard: the check is only meaningful if BaseModel actually declares hooks.
    assert BaseModel.__abstractmethods__, "BaseModel declares no abstract hooks"
    missing = set(BaseModel.__abstractmethods__) - stubbed
    assert not missing, f"_BaseModelStubs is missing stubs for {missing}"
