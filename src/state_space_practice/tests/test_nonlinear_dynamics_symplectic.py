"""Symplecticity checks for Hamiltonian leapfrog dynamics."""

import jax
import jax.numpy as jnp

from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    get_transition_jacobian,
    init_mlp_params,
)


def _canonical_symplectic_form(dim: int):
    n = dim // 2
    zeros = jnp.zeros((n, n))
    eye = jnp.eye(n)
    return jnp.block([[zeros, eye], [-eye, zeros]])


def _symplectic_error(jacobian):
    symplectic_form = _canonical_symplectic_form(jacobian.shape[0])
    residual = jacobian.T @ symplectic_form @ jacobian - symplectic_form
    return jnp.max(jnp.abs(residual))


def _full_state_nonseparable_hamiltonian(_params, state):
    n = state.shape[0] // 2
    q, p = state[:n], state[n:]
    return q[0] * p[0]


def test_default_random_mlp_transition_jacobian_is_symplectic():
    params = init_mlp_params(
        input_dim=4,
        hidden_dims=[5, 4],
        key=jax.random.PRNGKey(0),
    )
    params = {**params, "omega": jnp.array(1.4)}
    x = jnp.array([0.2, -0.35, 0.7, -0.4])

    jacobian = get_transition_jacobian(x, params, apply_mlp, dt=0.07)

    assert _symplectic_error(jacobian) < 1e-8


def test_zero_mlp_transition_jacobian_is_symplectic():
    params = init_mlp_params(
        input_dim=4,
        hidden_dims=[5, 4],
        key=jax.random.PRNGKey(1),
    )
    params = jax.tree_util.tree_map(jnp.zeros_like, params)
    params = {**params, "omega": jnp.array(2.0)}
    x = jnp.array([0.6, -0.25, 0.15, 0.9])

    jacobian = get_transition_jacobian(x, params, apply_mlp, dt=0.11)

    assert _symplectic_error(jacobian) < 1e-8


def test_full_state_nonseparable_hamiltonian_fails_symplectic_check():
    x = jnp.array([0.6, -0.25])

    jacobian = get_transition_jacobian(
        x,
        params={},
        h_apply_fn=_full_state_nonseparable_hamiltonian,
        dt=0.3,
    )

    assert _symplectic_error(jacobian) > 1e-3
