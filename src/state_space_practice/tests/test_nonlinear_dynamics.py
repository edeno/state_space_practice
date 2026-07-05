"""Covariance-hardening checks for nonlinear_dynamics EKF steps.

These complement ``test_nonlinear_dynamics_symplectic.py`` (which covers the
leapfrog symplecticity contract). Here we pin that the EKF predict/smoother
steps return *symmetric* covariances even when the incoming carry has drifted
out of symmetry (as accumulated float roundoff produces over a long scan).
"""

import jax
import jax.numpy as jnp

from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    ekf_predict_step,
    ekf_smooth_step,
    get_transition_jacobian,
    init_mlp_params,
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
