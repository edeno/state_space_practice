"""Tests for shared Hamiltonian EKF/Laplace helpers."""

import jax.numpy as jnp
import numpy as np

from state_space_practice.hamiltonian_core import point_process_laplace_update
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
