"""Hamiltonian Point-Process Model with Laplace-EKF Filtering.

This module implements the full nonlinear state-space model:
- Symplectic latent dynamics (Hamiltonian).
- Gaussian process noise (Q).
- Point-process observation model (Poisson/Log-linear).
- Laplace-EKF filtering and RTS smoothing.
"""

from typing import Dict, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.hamiltonian_models import (
    symplectic_transition, transition_jacobian
)
from state_space_practice.point_process_kalman import _point_process_laplace_update
from state_space_practice.kalman import _kalman_smoother_update, symmetrize

class HamiltonianFilterResult(NamedTuple):
    filtered_mean: Array
    filtered_cov: Array
    linearization_matrices: Array
    marginal_log_likelihood: Array

def nonlinear_predict(
    mean_prev: Array, 
    cov_prev: Array, 
    h_params: Dict[str, Array], 
    process_cov: Array, 
    dt: float
) -> Tuple[Array, Array, Array]:
    """Nonlinear prediction using symplectic integration and local Jacobian."""
    mean_pred = symplectic_transition(mean_prev, h_params, dt)
    A_t = transition_jacobian(mean_prev, h_params, dt)
    # P_t = A_t * P_{t-1} * A_t^T + Q
    cov_pred = symmetrize(A_t @ cov_prev @ A_t.T + process_cov)
    return mean_pred, cov_pred, A_t

@jax.jit
def hamiltonian_point_process_filter(
    init_mean: Array,
    init_cov: Array,
    process_cov: Array,
    hamiltonian_params: Dict[str, Array],
    C: Array,
    d: Array,
    spike_indicator: Array,
    dt: float,
) -> HamiltonianFilterResult:
    """End-to-end nonlinear point-process filter."""

    # Pre-compute derivatives of the log-intensity function for the Laplace update
    # log_lambda = C @ x + d
    def log_intensity_func(x):
        return jnp.dot(C, x) + d
    grad_log_intensity_func = jax.jacfwd(log_intensity_func)
    # Hessian is zero for linear intensity, but we pass it anyway
    hess_log_intensity_func = jax.jacfwd(grad_log_intensity_func)

    def step(carry, y_t):
        m_prev, P_prev = carry
        
        # 1. Predict
        m_pred, P_pred, A_t = nonlinear_predict(m_prev, P_prev, hamiltonian_params, process_cov, dt)
        
        # 2. Update (Laplace)
        m_post, P_post, log_lik_t = _point_process_laplace_update(
            m_pred, 
            P_pred, 
            y_t, 
            dt, 
            log_intensity_func,
            grad_log_intensity_func=grad_log_intensity_func,
            hess_log_intensity_func=hess_log_intensity_func
        )
        
        return (m_post, P_post), (m_post, P_post, A_t, log_lik_t)

    _, (m_f, P_f, A_all, lls) = jax.lax.scan(step, (init_mean, init_cov), spike_indicator)

    return HamiltonianFilterResult(
        filtered_mean=m_f,
        filtered_cov=P_f,
        linearization_matrices=A_all,
        marginal_log_likelihood=jnp.sum(lls)
    )

@jax.jit
def hamiltonian_point_process_smoother(
    filtered_mean: Array,
    filtered_cov: Array,
    linearization_matrices: Array,
    process_cov: Array,
) -> Tuple[Array, Array]:
    """Nonlinear RTS smoother using locally linearized transition matrices."""
    
    def backward_step(carry, inputs):
        m_s_next, P_s_next = carry
        m_f_t, P_f_t, A_next = inputs
        
        # _kalman_smoother_update(next_m_s, next_P_s, m_f, P_f, Q, A)
        m_s, P_s, _ = _kalman_smoother_update(
            m_s_next, P_s_next, m_f_t, P_f_t, process_cov, A_next
        )
        return (m_s, P_s), (m_s, P_s)

    # Align inputs for backward scan
    # m_f[t], P_f[t] with m_s[t+1], P_s[t+1] and A[t+1]
    bw_inputs = (
        filtered_mean[:-1], 
        filtered_cov[:-1], 
        linearization_matrices[1:]
    )
    
    last_state = (filtered_mean[-1], filtered_cov[-1])
    _, (m_s_rev, P_s_rev) = jax.lax.scan(backward_step, last_state, bw_inputs, reverse=True)
    
    m_s = jnp.concatenate([m_s_rev, filtered_mean[-1:]], axis=0)
    P_s = jnp.concatenate([P_s_rev, filtered_cov[-1:]], axis=0)
    
    return m_s, P_s
