r"""Gaussian-process priors as linear-Gaussian state-space models.

A stationary temporal Gaussian process with a Matern kernel is equivalent to a
linear time-invariant stochastic differential equation (SDE)

.. math::

    \mathrm{d}\mathbf{x}(t) = F\,\mathbf{x}(t)\,\mathrm{d}t + L\,\mathrm{d}\beta(t),
    \qquad f(t) = H\,\mathbf{x}(t),

driven by white noise with spectral density ``Qc`` (Hartikainen & Sarkka, 2010;
Sarkka & Solin, 2019, ch. 12). Sampling the SDE on a grid of spacing ``dt`` gives
a discrete linear-Gaussian model ``x_k = A x_{k-1} + w_k``, ``w_k ~ N(0, Q)`` that
plugs directly into the Kalman filter/smoother in :mod:`state_space_practice.kalman`.

This module provides the Matern-3/2 kernel, whose sample paths are once
mean-square differentiable. Its state is two-dimensional -- the value and
derivative of ``f`` -- and every discretized quantity is available in closed
form, so no matrix exponential or covariance integral is evaluated at run time.

References
----------
Hartikainen, J. & Sarkka, S. (2010). Kalman Filtering and Smoothing Solutions to
    Temporal Gaussian Process Regression Models. IEEE MLSP.
Sarkka, S. & Solin, A. (2019). Applied Stochastic Differential Equations.
    Cambridge University Press. (Ch. 12.)
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.utils import validate_scalar

# The value of ``f`` is the first state component: f = H @ x.
MATERN32_MEASUREMENT_VECTOR = jnp.array([1.0, 0.0])


def _matern32_rate(lengthscale: Array) -> Array:
    r"""Return the SDE rate ``lambda = sqrt(3) / lengthscale`` for Matern-3/2."""
    return jnp.sqrt(3.0) / lengthscale


def matern32_continuous(
    variance: ArrayLike,
    lengthscale: ArrayLike,
    validate: bool = True,
) -> tuple[Array, Array, Array, Array, Array]:
    r"""Continuous-time SDE representation of a Matern-3/2 Gaussian process.

    The kernel is
    ``k(tau) = variance * (1 + sqrt(3)|tau|/l) * exp(-sqrt(3)|tau|/l)``. Its
    state-space form has rate ``lambda = sqrt(3) / lengthscale`` and

    - ``F = [[0, 1], [-lambda^2, -2 lambda]]`` (drift),
    - ``L = [0, 1]`` (noise-effect vector),
    - ``Qc = 4 * variance * lambda^3`` (white-noise spectral density),
    - ``H = [1, 0]`` (``f`` is the first component),
    - ``Pinf = diag(variance, lambda^2 * variance)`` (stationary covariance).

    ``Pinf`` is the unique solution of the Lyapunov equation
    ``F Pinf + Pinf F.T + L Qc L.T = 0`` and is the correct prior covariance for
    the initial state ``x_0``.

    Parameters
    ----------
    variance : ArrayLike
        Marginal prior variance of ``f`` (strictly positive).
    lengthscale : ArrayLike
        Correlation time of ``f`` (strictly positive).

    Returns
    -------
    F : Array, shape (2, 2)
        Drift matrix.
    L : Array, shape (2,)
        Noise-effect vector.
    Qc : Array, shape ()
        White-noise spectral density (scalar array).
    H : Array, shape (2,)
        Measurement vector, ``[1, 0]``.
    Pinf : Array, shape (2, 2)
        Stationary state covariance.

    Notes
    -----
    Pass ``validate=False`` to skip the host-side scalar checks when calling
    from inside a ``jax.jit`` / ``jax.grad`` trace (where ``variance`` and
    ``lengthscale`` are tracers).
    """
    if validate:
        validate_scalar(variance, "variance", positive=True)
        validate_scalar(lengthscale, "lengthscale", positive=True)
    variance = jnp.asarray(variance, dtype=float)
    lengthscale = jnp.asarray(lengthscale, dtype=float)

    lam = _matern32_rate(lengthscale)
    F = jnp.array([[0.0, 1.0], [-(lam**2), -2.0 * lam]])
    L = jnp.array([0.0, 1.0])
    Qc = 4.0 * variance * lam**3
    Pinf = jnp.array([[variance, 0.0], [0.0, lam**2 * variance]])
    return F, L, Qc, MATERN32_MEASUREMENT_VECTOR, Pinf


def matern32_discretize(
    variance: ArrayLike,
    lengthscale: ArrayLike,
    dt: ArrayLike,
    validate: bool = True,
) -> tuple[Array, Array]:
    r"""Closed-form discrete transition ``A`` and process noise ``Q``.

    For a step of size ``dt`` the drift ``F`` has the repeated eigenvalue
    ``-lambda``, so ``A = expm(F dt) = exp(-lambda dt) (I + (F + lambda I) dt)``:

    .. math::

        A = e^{-\lambda\,dt}
            \begin{bmatrix} 1 + \lambda\,dt & dt \\
                            -\lambda^2\,dt & 1 - \lambda\,dt \end{bmatrix}.

    The process noise ``Q = Pinf - A Pinf A.T`` (the increment that keeps the
    stationary process stationary) evaluates in closed form to the expressions
    below. Because ``Q`` is written analytically rather than as a subtractive
    ``Pinf - A Pinf A.T``, it stays positive semidefinite as ``dt -> 0`` instead
    of losing PSD to cancellation.

    Parameters
    ----------
    variance : ArrayLike
        Marginal prior variance of ``f`` (strictly positive).
    lengthscale : ArrayLike
        Correlation time of ``f`` (strictly positive).
    dt : ArrayLike
        Time step (strictly positive).

    Returns
    -------
    A : Array, shape (2, 2)
        Discrete transition matrix.
    Q : Array, shape (2, 2)
        Discrete process-noise covariance.

    Notes
    -----
    Pass ``validate=False`` to skip the host-side scalar checks when calling
    from inside a ``jax.jit`` / ``jax.grad`` trace.
    """
    if validate:
        validate_scalar(variance, "variance", positive=True)
        validate_scalar(lengthscale, "lengthscale", positive=True)
        validate_scalar(dt, "dt", positive=True)
    variance = jnp.asarray(variance, dtype=float)
    lengthscale = jnp.asarray(lengthscale, dtype=float)
    dt = jnp.asarray(dt, dtype=float)

    lam = _matern32_rate(lengthscale)
    lam_dt = lam * dt
    decay = jnp.exp(-lam_dt)
    decay_sq = decay * decay

    A = decay * jnp.array([[1.0 + lam_dt, dt], [-(lam**2) * dt, 1.0 - lam_dt]])

    # Q = Pinf - A Pinf A.T, expanded analytically for the Matern-3/2 state.
    # The q11 term is O((lambda dt)^3); the closed form subtracts nearly equal
    # O(1) quantities at high sampling rates, so switch to its Taylor series in
    # that regime. q22 is written with expm1 so its O(lambda dt) leading term is
    # not lost to cancellation.
    r = lam_dt
    r2 = r * r
    q11_closed = variance * (1.0 - decay_sq * (1.0 + 2.0 * r + 2.0 * r2))
    q11_poly = 16.0 / 405.0
    for coeff in (-2.0 / 15.0, 8.0 / 21.0, -8.0 / 9.0, 8.0 / 5.0, -2.0, 4.0 / 3.0):
        q11_poly = coeff + r * q11_poly
    q11_series = variance * r**3 * q11_poly
    q11 = jnp.where(r <= 5e-2, q11_series, q11_closed)
    q12 = 2.0 * variance * lam * r2 * decay_sq
    q22_base = -jnp.expm1(-2.0 * r) + decay_sq * (2.0 * r - 2.0 * r2)
    q22 = variance * lam**2 * q22_base
    Q = jnp.array([[q11, q12], [q12, q22]])
    return A, Q
