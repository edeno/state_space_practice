"""Core nonlinear dynamics for Hamiltonian state-space models.

This module provides the 'Physics Engine' for latent trajectories,
independent of the observation model.
"""

import operator
from typing import Callable, Dict, List, Tuple, cast

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.kalman import psd_solve
from state_space_practice.utils import symmetrize


def _validate_state_vector(x: Array) -> Array:
    """Return a floating one-dimensional canonical ``[q, p]`` state."""
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"State must be a one-dimensional vector; got {x.shape}.")
    if x.shape[0] == 0 or x.shape[0] % 2 != 0:
        raise ValueError(
            "State dimension must be positive and even to split into (q, p); "
            f"got {x.shape[0]}."
        )
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.result_type(x, 1.0))
    return x


def leapfrog_step(
    x: Array,
    params: Dict[str, Array],
    h_apply_fn: Callable[[Dict[str, Array], Array], Array],
    dt: float,
) -> Array:
    """Symplectic leapfrog integrator (shared engine).

    Symplectic **only for separable** Hamiltonians ``H(q, p) = T(p) + V(q)``.
    The explicit kick-drift-kick scheme below evaluates ``dH/dq`` and ``dH/dp``
    at frozen partner coordinates; for a non-separable ``H`` it is only some
    explicit approximation with no symplecticity guarantee (a symplectic scheme
    would require an implicit solve). ``apply_mlp`` is constructed to be
    separable, so the shipped pairing is symplectic.
    """
    x = _validate_state_vector(x)
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
    dt: float,
) -> Array:
    """Compute local Jacobian F = df/dx for EKF-style covariance propagation.

    Uses ``jax.linearize`` so callers that also need the predicted state can
    reuse the same primal evaluation. The returned linear map is applied to a
    basis with ``vmap`` to materialize the square Jacobian.
    """
    _, jacobian = _leapfrog_step_and_jacobian(x, params, h_apply_fn, dt)
    return jacobian


def init_mlp_params(
    input_dim: int, hidden_dims: List[int], key: Array
) -> Dict[str, Array]:
    """Initialize a scalar MLP over position coordinates."""
    try:
        input_dim = operator.index(input_dim)
        hidden_dims = [operator.index(width) for width in hidden_dims]
    except TypeError as exc:
        raise ValueError("MLP dimensions must be integers.") from exc
    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")
    if any(width <= 0 for width in hidden_dims):
        raise ValueError("hidden_dims must contain only positive integers.")

    params = {}
    dims = [input_dim] + hidden_dims + [1]
    for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:])):
        k1, key = jax.random.split(key, 2)
        params[f"w{i}"] = jax.random.normal(k1, (n_in, n_out)) * jnp.sqrt(
            2 / (n_in + n_out)
        )
        params[f"b{i}"] = jnp.zeros((n_out,))
    return params


def ekf_predict_step(
    m_prev: Array,
    P_prev: Array,
    params: Dict[str, Array],
    h_apply_fn: Callable,
    Q: Array,
    dt: float,
) -> Tuple[Array, Array]:
    """EKF Prediction Step: x_t = f(x_{t-1}) + w_t."""
    m_pred, P_pred, _ = ekf_predict_step_with_jacobian(
        m_prev, P_prev, params, h_apply_fn, Q, dt
    )
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
    P_pred = symmetrize(F @ P_prev @ F.T + Q)
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

    x = _validate_state_vector(x)
    m_pred, jvp = jax.linearize(step_fn, x)
    # vmap stores J @ e_i along axis 0; transpose those columns into the
    # conventional output-by-input Jacobian layout.
    F = jax.vmap(jvp)(jnp.eye(x.shape[0], dtype=x.dtype)).T
    return m_pred, F


def ekf_smooth_step(
    m_filt: Array,
    P_filt: Array,
    m_pred_next: Array,
    P_pred_next: Array,
    m_smooth_next: Array,
    P_smooth_next: Array,
    F_next: Array,
) -> Tuple[Array, Array]:
    """EKF RTS Smoother Step (Backward Pass)."""
    G = psd_solve(P_pred_next, F_next @ P_filt).T
    m_smooth = m_filt + G @ (m_smooth_next - m_pred_next)
    P_smooth = symmetrize(P_filt + G @ (P_smooth_next - P_pred_next) @ G.T)
    return m_smooth, P_smooth


def _mlp_layer_count(params: Dict[str, Array], input_dim: int) -> int:
    """Validate a contiguous scalar-output MLP and return its layer count."""
    weight_indices = sorted(
        int(key[1:]) for key in params if key.startswith("w") and key[1:].isdigit()
    )
    if not weight_indices:
        raise ValueError("MLP parameters must contain at least w0 and b0.")
    expected = list(range(len(weight_indices)))
    if weight_indices != expected:
        raise ValueError(
            f"MLP weight keys must be contiguous w0...wN; got {weight_indices}."
        )
    bias_indices = sorted(
        int(key[1:]) for key in params if key.startswith("b") and key[1:].isdigit()
    )
    if bias_indices != expected:
        raise ValueError(
            "MLP bias keys must pair one-to-one with weights b0...bN; "
            f"got {bias_indices}."
        )

    previous_width = input_dim
    for i in expected:
        weight_key, bias_key = f"w{i}", f"b{i}"
        if bias_key not in params:
            raise ValueError(f"MLP parameters are missing {bias_key}.")
        weight, bias = params[weight_key], params[bias_key]
        if weight.ndim != 2 or bias.ndim != 1:
            raise ValueError(
                f"{weight_key} must be 2D and {bias_key} must be 1D; "
                f"got {weight.shape} and {bias.shape}."
            )
        if weight.shape[0] != previous_width or weight.shape[1] != bias.shape[0]:
            raise ValueError(
                f"Incompatible shapes for {weight_key}/{bias_key}: "
                f"expected input width {previous_width}, got "
                f"{weight.shape} and {bias.shape}."
            )
        previous_width = weight.shape[1]
    if previous_width != 1:
        raise ValueError(f"Final MLP layer must have one output; got {previous_width}.")
    return len(expected)


def apply_mlp(params: Dict[str, Array], x: Array) -> Array:
    """Apply the MLP to compute a separable scalar Hamiltonian H(q, p).

    The quadratic kinetic term depends on momentum, while the MLP residual is
    restricted to the position coordinates so explicit leapfrog remains
    symplectic. The potential MLP therefore has ``len(q)`` inputs; momentum
    coordinates are not represented by structurally dead weights.

    Batch over an ensemble of parameter sets at the call site with
    ``jax.vmap(apply_mlp, in_axes=(0, None))``; this function itself takes a
    single parameter set and returns a scalar.
    """
    x = _validate_state_vector(x)
    omega = params.get("omega", 1.0)

    n = x.shape[0] // 2
    q, p = x[:n], x[n:]
    n_layers = _mlp_layer_count(params, input_dim=n)

    def mlp_forward(input_vec):
        curr = input_vec
        for i in range(n_layers - 1):
            curr = jnp.dot(curr, params[f"w{i}"]) + params[f"b{i}"]
            curr = jax.nn.tanh(curr)
        curr = jnp.dot(curr, params[f"w{n_layers - 1}"]) + params[f"b{n_layers - 1}"]
        return jnp.squeeze(curr)

    h_prior = 0.5 * jnp.sum(p**2) + 0.5 * (omega**2) * jnp.sum(q**2)
    h_mlp = mlp_forward(q) - mlp_forward(jnp.zeros_like(q))

    return cast(Array, h_prior + h_mlp)
