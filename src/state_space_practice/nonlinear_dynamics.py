"""Core Nonlinear Dynamics for Hamiltonian State-Space Models.

This module provides the 'Physics Engine' for latent trajectories,
independent of the observation model.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Callable
from jax import Array
from state_space_practice.kalman import psd_solve

def leapfrog_step(
    x: Array, 
    params: Dict[str, Array], 
    h_apply_fn: Callable[[Dict[str, Array], Array], Array], 
    dt: float = 0.1
) -> Array:
    """Symplectic Leapfrog Integrator (Shared Engine)."""
    n = x.shape[0] // 2
    q, p = x[:n], x[n:]
    
    def H_func(state: Array) -> Array:
        return jnp.squeeze(h_apply_fn(params, state))

    # Compute gradient function once, reuse for all three evaluations
    grad_H = jax.grad(H_func)

    def get_grads(state: Array) -> Tuple[Array, Array]:
        grads = grad_H(state)
        return grads[:n], grads[n:]

    # 1. p(t + dt/2)
    dH_dq, _ = get_grads(x)
    p_mid = p - (dt / 2.0) * dH_dq.reshape(p.shape)

    # 2. q(t + dt)
    state_mid = jnp.concatenate([q, p_mid])
    _, dH_dp = get_grads(state_mid)
    q_next = q + dt * dH_dp.reshape(q.shape)

    # 3. p(t + dt)
    state_next_mid = jnp.concatenate([q_next, p_mid])
    dH_dq_next, _ = get_grads(state_next_mid)
    p_next = p_mid - (dt / 2.0) * dH_dq_next.reshape(p.shape)
    
    return jnp.concatenate([q_next, p_next])

def get_transition_jacobian(
    x: Array,
    params: Dict[str, Array],
    h_apply_fn: Callable,
    dt: float
) -> Array:
    """Compute local Jacobian F = df/dx for EKF-style covariance propagation.

    Uses reverse-mode AD (jacrev) which is more efficient than forward-mode
    for square Jacobians since it computes all rows in a single backward pass.
    """
    def step_fn(state):
        return leapfrog_step(state, params, h_apply_fn, dt)
    return jax.jacrev(step_fn)(x)

def init_mlp_params(
    input_dim: int,
    hidden_dims: List[int],
    key: Array
) -> Dict[str, Array]:
    params = {}
    dims = [input_dim] + hidden_dims + [1]
    for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:])):
        k1, k2, key = jax.random.split(key, 3)
        params[f"w{i}"] = jax.random.normal(k1, (n_in, n_out)) * jnp.sqrt(2 / (n_in + n_out))
        params[f"b{i}"] = jnp.zeros((n_out,))
    return params

def ekf_predict_step(
    m_prev: Array,
    P_prev: Array,
    params: Dict[str, Array],
    h_apply_fn: Callable,
    Q: Array,
    dt: float
) -> Tuple[Array, Array]:
    """EKF Prediction Step: x_t = f(x_{t-1}) + w_t."""
    m_pred, F = _leapfrog_step_and_jacobian(m_prev, params, h_apply_fn, dt)
    P_pred = F @ P_prev @ F.T + Q
    return m_pred, P_pred


def ekf_predict_step_with_jacobian(
    m_prev: Array,
    P_prev: Array,
    params: Dict[str, Array],
    h_apply_fn: Callable,
    Q: Array,
    dt: float,
) -> Tuple[Array, Array, Array]:
    """EKF Prediction Step that also returns the transition Jacobian.

    Use this in smoother forward passes to avoid recomputing the Jacobian.
    """
    m_pred, F = _leapfrog_step_and_jacobian(m_prev, params, h_apply_fn, dt)
    P_pred = F @ P_prev @ F.T + Q
    return m_pred, P_pred, F


def _leapfrog_step_and_jacobian(
    x: Array,
    params: Dict[str, Array],
    h_apply_fn: Callable,
    dt: float,
) -> Tuple[Array, Array]:
    """Compute leapfrog step and its Jacobian in a single pass."""
    def step_fn(state):
        return leapfrog_step(state, params, h_apply_fn, dt)
    m_pred = step_fn(x)
    F = jax.jacrev(step_fn)(x)
    return m_pred, F

def ekf_smooth_step(
    m_filt: Array,
    P_filt: Array,
    m_pred_next: Array,
    P_pred_next: Array,
    m_smooth_next: Array,
    P_smooth_next: Array,
    F_next: Array
) -> Tuple[Array, Array]:
    """EKF RTS Smoother Step (Backward Pass)."""
    G = psd_solve(P_pred_next, F_next @ P_filt).T
    m_smooth = m_filt + G @ (m_smooth_next - m_pred_next)
    P_smooth = P_filt + G @ (P_smooth_next - P_pred_next) @ G.T
    return m_smooth, P_smooth

def apply_mlp(params: Dict[str, Array], x: Array) -> Array:
    """Apply the MLP to compute scalar energy H(x).
    
    Includes a learnable quadratic prior and centering.
    """
    # Check if params is batched
    if "w0" in params and params["w0"].ndim > 2:
        return jax.vmap(apply_mlp, in_axes=(0, None))(params, x)

    weight_keys = sorted([k for k in params.keys() if k.startswith("w")])
    omega = params.get("omega", 1.0)
    
    n = x.shape[0] // 2
    q, p = x[:n], x[n:]
    
    def mlp_forward(input_vec):
        curr = input_vec
        for i in range(len(weight_keys) - 1):
            curr = jnp.dot(curr, params[f"w{i}"]) + params[f"b{i}"]
            curr = jax.nn.tanh(curr)
        curr = jnp.dot(curr, params[f"w{len(weight_keys)-1}"]) + params[f"b{len(weight_keys)-1}"]
        return jnp.squeeze(curr)

    h_prior = 0.5 * jnp.sum(p**2) + 0.5 * (omega**2) * jnp.sum(q**2)
    h_mlp = mlp_forward(x) - mlp_forward(jnp.zeros_like(x))
    
    return h_prior + h_mlp
