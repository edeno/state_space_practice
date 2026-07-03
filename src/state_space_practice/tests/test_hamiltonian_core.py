"""Tests for shared Hamiltonian EKF/Laplace helpers."""

import jax.numpy as jnp

from state_space_practice.hamiltonian_core import point_process_laplace_update
from state_space_practice.kalman import psd_solve, symmetrize


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
    eye = jnp.eye(P_pred.shape[0])
    expected = symmetrize(psd_solve(psd_solve(P_pred, eye) + H_lik, eye))

    assert jnp.allclose(P_post, expected, rtol=1e-6, atol=1e-6)
