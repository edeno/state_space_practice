# ruff: noqa: E402
"""Leapfrog dynamics against an ODE-solver oracle and geometric invariants.

:func:`state_space_practice.nonlinear_dynamics.leapfrog_step` is the
Stormer-Verlet (kick-drift-kick) scheme. For a separable Hamiltonian
``H(q, p) = T(p) + V(q)`` it must

1. converge to the true flow with global order 2 -- checked against
   :func:`scipy.integrate.solve_ivp` (DOP853 at ``rtol = atol = 1e-12``) over a
   step-size sweep, via the log-log slope of the error;
2. be time-reversible to round-off (``N`` steps with ``dt`` followed by ``N``
   steps with ``-dt`` return the initial state);
3. keep the energy error bounded, with no secular growth, over long horizons;
4. preserve phase-space volume and the symplectic form (``det J = 1`` and
   ``J^T Omega J = Omega`` for the Jacobian of the ``N``-step map, obtained by
   :func:`jax.jacfwd`).

Each geometric property is contrasted with classical RK4 -- accurate (order 4)
but not symplectic -- to show the checks are not vacuous: RK4 fails 2-4 by
orders of magnitude more than round-off.

The Hamiltonians are the library's own :func:`apply_mlp` with a random
(anharmonic) potential network, and a nonlinear pendulum.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    get_transition_jacobian,
    init_mlp_params,
    leapfrog_step,
)

# ---------------------------------------------------------------------------
# Hamiltonians and reference integrators
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlp_system():
    """Anharmonic separable H = |p|^2/2 + omega^2 |q|^2/2 + MLP(q) - MLP(0)."""
    params = init_mlp_params(input_dim=2, hidden_dims=[8], key=jax.random.PRNGKey(3))
    # scale the residual up so the dynamics are clearly non-linear
    params = {k: 1.5 * v for k, v in params.items()}
    params = {**params, "omega": jnp.array(1.1)}
    x0 = jnp.array([0.8, -0.3, 0.1, 0.6])
    return _System(params, apply_mlp, x0)


def _pendulum_h(_params, state):
    q, p = state[0], state[1]
    return 0.5 * p**2 - jnp.cos(q)


@pytest.fixture(scope="module")
def pendulum_system():
    return _System({}, _pendulum_h, jnp.array([2.0, 0.0]))  # large-amplitude swing


def _vector_field(params, h_fn):
    """Hamilton's equations dq/dt = dH/dp, dp/dt = -dH/dq."""
    grad_h = jax.grad(lambda s: jnp.squeeze(h_fn(params, s)))

    def field(state):
        g = grad_h(state)
        n = state.shape[0] // 2
        return jnp.concatenate([g[n:], -g[:n]])

    return field


def _rk4_step(field, x, dt):
    k1 = field(x)
    k2 = field(x + 0.5 * dt * k1)
    k3 = field(x + 0.5 * dt * k2)
    k4 = field(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rollout(step_fn, x0, n_steps):
    """All states x_0..x_N of an ``x -> step_fn(x)`` iteration (jit + scan)."""

    @jax.jit
    def run(x):
        def body(carry, _):
            nxt = step_fn(carry)
            return nxt, nxt

        _, rest = jax.lax.scan(body, x, None, length=n_steps)
        return jnp.concatenate([x[None], rest], axis=0)

    return run(x0)


def _leapfrog_map(params, h_fn):
    """Jitted ``(x, dt, n_steps) -> Phi_dt^n_steps(x)``; one compile per system."""

    @jax.jit
    def n_step_map(x, dt, n_steps):
        return jax.lax.fori_loop(
            0, n_steps, lambda _, s: leapfrog_step(s, params, h_fn, dt), x
        )

    return n_step_map


def _rk4_map(params, h_fn):
    """Jitted ``(x, dt, n_steps) -> RK4 flow``; one compile per system."""
    field = _vector_field(params, h_fn)

    @jax.jit
    def n_step_map(x, dt, n_steps):
        return jax.lax.fori_loop(0, n_steps, lambda _, s: _rk4_step(field, s, dt), x)

    return n_step_map


class _System:
    """A Hamiltonian system with compiled leapfrog / RK4 n-step maps."""

    def __init__(self, params, h_fn, x0):
        self.params, self.h_fn, self.x0 = params, h_fn, x0
        self.leapfrog = _leapfrog_map(params, h_fn)
        self.rk4 = _rk4_map(params, h_fn)

    def __iter__(self):  # unpack as (params, h_fn, x0)
        return iter((self.params, self.h_fn, self.x0))


def _solve_ivp_reference(params, h_fn, x0, t_final):
    field = jax.jit(_vector_field(params, h_fn))
    sol = solve_ivp(
        lambda _t, y: np.asarray(field(jnp.asarray(y))),
        (0.0, t_final),
        np.asarray(x0),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    assert sol.success
    return sol.y[:, -1]


def _symplectic_form(dim):
    n = dim // 2
    return np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])


# ---------------------------------------------------------------------------
# 1. Convergence to the exact flow (solve_ivp oracle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", ["mlp_system", "pendulum_system"])
def test_leapfrog_converges_to_solve_ivp_with_order_two(system, request):
    """Global error vs a 1e-12 DOP853 solution falls as dt^2 (slope 2 +- 0.1)."""
    sys_ = request.getfixturevalue(system)
    params, h_fn, x0 = sys_
    t_final = 3.0
    reference = _solve_ivp_reference(params, h_fn, x0, t_final)

    n_steps_sweep = np.array([30, 60, 120, 240, 480, 960])
    errors = np.array(
        [
            float(
                np.max(
                    np.abs(np.asarray(sys_.leapfrog(x0, t_final / n, n)) - reference)
                )
            )
            for n in n_steps_sweep
        ]
    )
    dts = t_final / n_steps_sweep

    slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
    assert 1.9 < slope < 2.1, (slope, errors)  # observed 2.001 / 1.9997
    # guard: the finest error is far above the 1e-12 reference tolerance, so
    # the slope measures the integrator, not the oracle
    assert errors[-1] > 1e3 * 1e-12
    # and the error really is dominated by dt^2 (successive ratios ~4)
    np.testing.assert_allclose(errors[:-1] / errors[1:], 4.0, rtol=0.1)


def test_rk4_reference_has_order_four(mlp_system):
    """The non-symplectic comparator is itself correct (order 4 vs solve_ivp)."""
    params, h_fn, x0 = mlp_system
    t_final = 3.0
    reference = _solve_ivp_reference(params, h_fn, x0, t_final)
    n_steps_sweep = np.array([15, 30, 60, 120])
    errors = np.array(
        [
            float(
                np.max(
                    np.abs(np.asarray(mlp_system.rk4(x0, t_final / n, n)) - reference)
                )
            )
            for n in n_steps_sweep
        ]
    )
    slope = np.polyfit(np.log(t_final / n_steps_sweep), np.log(errors), 1)[0]
    assert 3.8 < slope < 4.2, (slope, errors)


def test_transition_jacobian_matches_jacfwd(mlp_system):
    """The EKF Jacobian equals autodiff of one leapfrog step."""
    params, h_fn, x0 = mlp_system
    dt = 0.07
    expected = jax.jit(jax.jacfwd(lambda s: leapfrog_step(s, params, h_fn, dt)))(x0)
    actual = get_transition_jacobian(x0, params, h_fn, dt)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-14)
    assert np.max(np.abs(np.asarray(expected) - np.eye(4))) > 1e-2  # non-trivial


# ---------------------------------------------------------------------------
# 2. Time reversibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", ["mlp_system", "pendulum_system"])
def test_leapfrog_is_time_reversible_to_roundoff(system, request):
    """N steps forward then N steps with -dt return x0 to round-off; RK4 does not."""
    sys_ = request.getfixturevalue(system)
    x0 = sys_.x0
    dt, n_steps = 0.1, 2000

    def there_and_back(n_step_map):
        x_far = n_step_map(x0, dt, n_steps)
        return x_far, n_step_map(x_far, -dt, n_steps)

    x_far, x_back = there_and_back(sys_.leapfrog)
    assert float(jnp.max(jnp.abs(x_far - x0))) > 0.1  # guard: travelled far
    leapfrog_error = float(jnp.max(jnp.abs(x_back - x0)))
    assert leapfrog_error < 1e-11

    # Equivalent statement via the momentum-flip involution R(q, p) = (q, -p):
    # Phi_dt^N o R o Phi_dt^N = R.
    n = x0.shape[0] // 2
    flip = jnp.concatenate([jnp.ones(n), -jnp.ones(n)])
    x_flip_back = sys_.leapfrog(flip * x_far, dt, n_steps)
    assert float(jnp.max(jnp.abs(flip * x_flip_back - x0))) < 1e-11

    _, rk4_back = there_and_back(sys_.rk4)
    rk4_error = float(jnp.max(jnp.abs(rk4_back - x0)))
    assert rk4_error > 1e3 * max(leapfrog_error, 1e-15)


# ---------------------------------------------------------------------------
# 3. Long-horizon energy behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", ["mlp_system", "pendulum_system"])
def test_leapfrog_energy_error_is_bounded_while_rk4_drifts(system, request):
    """Leapfrog energy error stays O(dt^2) with no trend; RK4's grows secularly.

    Over 20 000 steps (~170 periods) the leapfrog energy error in the
    last tenth of the run is no larger than in the first tenth, and the whole
    band is O(dt^2). RK4 at the same step has a much smaller per-step error but
    a monotone drift: its error at the end is ~10x its error after a tenth.
    """
    params, h_fn, x0 = request.getfixturevalue(system)
    dt, n_steps = 0.05, 20_000

    def energy(traj):
        h = jax.vmap(lambda s: jnp.squeeze(h_fn(params, s)))(traj)
        return np.asarray(h - h[0])

    lf = energy(_rollout(lambda s: leapfrog_step(s, params, h_fn, dt), x0, n_steps))
    field = _vector_field(params, h_fn)
    rk = energy(_rollout(lambda s: _rk4_step(field, s, dt), x0, n_steps))

    tenth = n_steps // 10
    lf_early = np.max(np.abs(lf[1 : tenth + 1]))
    lf_late = np.max(np.abs(lf[-tenth:]))
    assert lf_early > 0  # guard: finite step => nonzero energy error
    assert lf_late < 1.2 * lf_early, (lf_early, lf_late)
    # bounded, O(dt^2); observed 2.9e-3 (MLP) and 6.8e-4 (pendulum) = 1.2 dt^2
    assert np.max(np.abs(lf)) < 2.0 * dt**2

    # Halving dt quarters the leapfrog band (the modified-Hamiltonian bound).
    lf_half = energy(
        _rollout(lambda s: leapfrog_step(s, params, h_fn, dt / 2), x0, 2 * tenth)
    )
    ratio = np.max(np.abs(lf[: tenth + 1])) / np.max(np.abs(lf_half))
    assert 3.8 < ratio < 4.2, ratio  # observed 4.001

    # RK4: secular drift -- growth over the run, of a single sign.
    rk_at_tenth = abs(rk[tenth])
    rk_at_end = abs(rk[-1])
    assert rk_at_end > 5.0 * rk_at_tenth, (rk_at_tenth, rk_at_end)
    trend = np.polyfit(np.arange(rk.size), rk, 1)[0] * rk.size
    assert abs(trend) > 0.5 * rk_at_end


# ---------------------------------------------------------------------------
# 4. Phase-space volume / symplectic form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", ["mlp_system", "pendulum_system"])
def test_leapfrog_map_preserves_volume_and_symplectic_form(system, request):
    """det(d Phi^N / dx) = 1 and J^T Omega J = Omega to round-off; RK4 is off."""
    sys_ = request.getfixturevalue(system)
    x0 = sys_.x0
    dt, n_steps = 0.1, 200
    dim = x0.shape[0]
    omega = _symplectic_form(dim)

    jac_lf = np.asarray(jax.jacfwd(sys_.leapfrog)(x0, dt, n_steps))
    jac_rk = np.asarray(jax.jacfwd(sys_.rk4)(x0, dt, n_steps))

    # guard: the map is far from the identity (and strongly shearing)
    assert np.max(np.abs(jac_lf - np.eye(dim))) > 0.5

    det_error_lf = abs(np.linalg.det(jac_lf) - 1.0)
    sympl_error_lf = np.max(np.abs(jac_lf.T @ omega @ jac_lf - omega))
    assert det_error_lf < 1e-11
    assert sympl_error_lf < 1e-11

    det_error_rk = abs(np.linalg.det(jac_rk) - 1.0)
    sympl_error_rk = np.max(np.abs(jac_rk.T @ omega @ jac_rk - omega))
    assert det_error_rk > 1e3 * max(det_error_lf, 1e-15)
    assert sympl_error_rk > 1e3 * max(sympl_error_lf, 1e-15)
