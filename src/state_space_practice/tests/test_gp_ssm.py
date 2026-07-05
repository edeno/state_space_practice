"""Tests for the Matern-3/2 Gaussian-process-as-SDE state-space primitive.

The discrete-time ``(A, Q)`` returned by :func:`matern32_discretize` is checked
against three independent oracles:

- ``A`` must equal ``expm(F * dt)`` (the definition of the discretized drift).
- ``Q`` must equal ``Pinf - A @ Pinf @ A.T`` at moderate ``dt`` (the discrete
  process noise that keeps a *stationary* process stationary).
- ``Pinf`` must solve the continuous Lyapunov equation
  ``F @ Pinf + Pinf @ F.T + L @ (Qc * L.T) = 0``.

None of these oracles reuses the closed-form expressions under test, so a wrong
coefficient in the closed form fails at least one of them.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm
from numpy.testing import assert_allclose

from state_space_practice.gp_ssm import matern32_continuous, matern32_discretize

VARIANCE = 2.0
LENGTHSCALE = 0.5


def test_transition_matches_matrix_exponential():
    """A(dt) == expm(F dt) for the Matern-3/2 drift."""
    F, _L, _Qc, _H, _Pinf = matern32_continuous(VARIANCE, LENGTHSCALE)
    for dt in (1e-3, 1e-2, 0.1, 0.5, 1.0):
        A, _Q = matern32_discretize(VARIANCE, LENGTHSCALE, dt)
        assert_allclose(
            np.asarray(A), np.asarray(expm(np.asarray(F) * dt)), rtol=1e-9, atol=1e-12
        )


def test_process_noise_matches_stationary_difference():
    """Q(dt) == Pinf - A Pinf A.T at moderate dt (stationary discrete noise)."""
    _F, _L, _Qc, _H, Pinf = matern32_continuous(VARIANCE, LENGTHSCALE)
    for dt in (1e-2, 0.1, 0.5, 1.0, 2.0):
        A, Q = matern32_discretize(VARIANCE, LENGTHSCALE, dt)
        expected = Pinf - A @ Pinf @ A.T
        assert_allclose(np.asarray(Q), np.asarray(expected), rtol=1e-8, atol=1e-12)


def test_process_noise_is_psd_across_dt():
    """Q stays positive semidefinite from very small to large dt."""
    for dt in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0):
        _A, Q = matern32_discretize(VARIANCE, LENGTHSCALE, dt)
        Q_sym = np.asarray((Q + Q.T) / 2)
        min_eig = float(np.linalg.eigvalsh(Q_sym).min())
        assert min_eig >= -1e-12, f"Q not PSD at dt={dt}: min eigenvalue {min_eig}"


def test_stationary_covariance_solves_lyapunov():
    """Pinf solves F Pinf + Pinf F.T + L Qc L.T = 0."""
    F, L, Qc, _H, Pinf = matern32_continuous(VARIANCE, LENGTHSCALE)
    L_col = jnp.reshape(L, (-1, 1))
    residual = F @ Pinf + Pinf @ F.T + L_col @ (Qc * L_col.T)
    assert_allclose(np.asarray(residual), np.zeros((2, 2)), atol=1e-10)


def test_marginal_prior_variance_equals_variance():
    """H Pinf H.T recovers the marginal prior variance of f = H x."""
    _F, _L, _Qc, H, Pinf = matern32_continuous(VARIANCE, LENGTHSCALE)
    assert_allclose(float(H @ Pinf @ H), VARIANCE, rtol=1e-12)


def test_longer_lengthscale_gives_slower_decay():
    """A larger lengthscale correlates f over longer times: A[0,0] decays slower."""
    dt = 0.1
    A_short, _ = matern32_discretize(VARIANCE, 0.2, dt)
    A_long, _ = matern32_discretize(VARIANCE, 2.0, dt)
    # A[0, 0] is the one-step retention of the f-component; longer lengthscale
    # (slower dynamics) retains more of the previous value.
    assert float(A_long[0, 0]) > float(A_short[0, 0])


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_rejects_nonpositive_hyperparameters(bad_value):
    """variance and lengthscale must be strictly positive."""
    with pytest.raises(ValueError):
        matern32_continuous(bad_value, LENGTHSCALE)
    with pytest.raises(ValueError):
        matern32_continuous(VARIANCE, bad_value)
