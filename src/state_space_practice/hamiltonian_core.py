"""Shared EKF / Laplace-EKF kernel for the Hamiltonian model family.

Conventions
-----------
- State, covariance, and observation arguments are JAX arrays. Configuration
  arguments such as ``dt`` and likelihood-control booleans are static Python
  values; declare them static when JIT-compiling a helper directly.
- Helpers return JAX arrays and do not capture Python state; closures over
  external arrays inside ``jax.lax.scan`` bodies must come from the caller,
  not from this module.
- Log-likelihoods returned here are the per-step contribution. Callers
  accumulate via the scan ``carry`` or via ``jnp.sum`` over the scan
  output, depending on filter/smoother convention.

See docs/hamiltonian_architecture.md for the broader rationale (why the
Hamiltonian family is standalone — no linear-Gaussian EM integration,
SGD-only fitting).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import Array

from state_space_practice.kalman import joseph_form_update
from state_space_practice.nonlinear_dynamics import ekf_smooth_step
from state_space_practice.point_process_kalman import (
    glm_laplace_update,
    poisson_family,
)
from state_space_practice.utils import psd_cholesky


def gaussian_measurement_update(
    m_pred: Array,
    P_pred: Array,
    y: Array,
    C: Array,
    d: Array,
    R: Array,
    *,
    include_normalization_const: bool = True,
) -> Tuple[Array, Array, Array]:
    """Standard Kalman update for a linear-Gaussian observation y ~ N(C x + d, R).

    Returns
    -------
    m_post : (n,)
    P_post : (n, n) — Joseph-form for PSD preservation
    log_likelihood : ()
        Per-step Gaussian marginal log-likelihood. Set
        ``include_normalization_const=False`` to drop the
        ``n_obs * log(2π)`` constant — useful for *relative* likelihoods
        (e.g. discrete-state softmax in switching models) where the
        constant cancels in normalization.
    """
    # The Gaussian density over an empty observation vector is the empty
    # product: it contributes zero log-likelihood and leaves the prior
    # unchanged. Besides being mathematically natural, this avoids asking
    # psd_cholesky to reduce over a 0 x 0 innovation covariance.
    if y.shape[0] == 0:
        ll_dtype = jnp.result_type(m_pred, P_pred, 0.0)
        return m_pred, P_pred, jnp.zeros((), dtype=ll_dtype)

    err = y - (C @ m_pred + d)
    S = C @ P_pred @ C.T + R
    # One stabilized Cholesky of S serves the gain solve, the quadratic form,
    # and the log-determinant, so all three see the same symmetrized + boosted
    # matrix (avoids the boosted-solve / unboosted-slogdet mismatch on
    # near-singular S) and S is factored once rather than three times.
    S_cho = psd_cholesky(S)
    K = jax.scipy.linalg.cho_solve(S_cho, C @ P_pred).T
    m_post = m_pred + K @ err
    P_post = joseph_form_update(P_pred, K, C, R)

    logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(S_cho[0]))))
    ll = -0.5 * (err @ jax.scipy.linalg.cho_solve(S_cho, err) + logdet)
    if include_normalization_const:
        n_obs = err.shape[0]
        ll = ll - 0.5 * n_obs * jnp.log(2 * jnp.pi)
    return m_post, P_post, ll


def point_process_laplace_update(
    m_pred: Array,
    P_pred: Array,
    y: Array,
    C: Array,
    d: Array,
    dt: float,
    *,
    compute_log_likelihood: bool = True,
) -> Tuple[Array, Array, Array]:
    """Single-Fisher-step Laplace update for Poisson observations.

    Observation model: ``y[n] ~ Poisson(exp(C[n] @ x + d[n]) * dt)``.
    Delegates to the shared GLM Laplace update with ``poisson_family(dt)`` so
    Hamiltonian point-process likelihoods use the same normalized Poisson
    log-PMF and expected-count clipping as the generic point-process filter.

    ``dt`` and ``compute_log_likelihood`` are configuration values and must be
    declared static when this helper is JIT-compiled directly, for example
    ``jax.jit(point_process_laplace_update,
    static_argnames=("dt", "compute_log_likelihood"))``. Hamiltonian model
    methods already capture ``self.dt`` statically through their static model
    instance.

    Returns
    -------
    m_post : (n,)
    P_post : (n, n) — symmetrised
    log_likelihood : ()
        Laplace-approximated marginal ``log p(y | y_{1:t-1})``.
        With ``compute_log_likelihood=False`` returns ``jnp.array(0.0)``
        and, because the log-likelihood is discarded, skips the Laplace
        normalization (two Cholesky log-determinants per step) inside the
        GLM update — the intended saving for the smoother forward pass.
    """

    def eta_func(x: Array) -> Array:
        return C @ x + d

    def grad_eta_func(_x: Array) -> Array:
        return C

    # The posterior mean/covariance do not depend on the normalization
    # constant, so dropping it when the ll is discarded changes nothing but
    # the (unused) return value while avoiding two Cholesky log-determinants.
    m_post, P_post, ll = glm_laplace_update(
        m_pred,
        P_pred,
        y,
        eta_func,
        poisson_family(dt),
        grad_eta_func=grad_eta_func,
        include_laplace_normalization=compute_log_likelihood,
    )
    if not compute_log_likelihood:
        return m_post, P_post, jnp.array(0.0)
    return m_post, P_post, ll


def ekf_rts_backward_pass(
    m_filt: Array,
    P_filt: Array,
    m_pred: Array,
    P_pred: Array,
    F: Array,
) -> Tuple[Array, Array]:
    """EKF-RTS backward smoother given a forward pass's filtered + predicted state.

    Parameters
    ----------
    m_filt, P_filt : (T, n) and (T, n, n)
        Filtered means and covariances.
    m_pred, P_pred : (T, n) and (T, n, n)
        One-step-ahead predicted means and covariances. The
        ``backward_step`` consumes ``m_pred[t+1]``, ``P_pred[t+1]`` and
        ``F[t+1]`` while smoothing position ``t``.
    F : (T, n, n)
        Transition Jacobian at each forward step (``∂f/∂x`` evaluated
        at the previous filtered mean, returned by
        ``ekf_predict_step_with_jacobian``). The correct alignment is the
        one that satisfies ``P_pred[t+1] == F[t+1] @ P_filt[t] @ F[t+1].T
        + Q`` — i.e. ``F[t+1]`` is the Jacobian *used to produce*
        ``P_pred[t+1]``, evaluated at ``m_filt[t]``. Storing ``F`` shifted
        by one step silently corrupts every smoother gain.

    Notes
    -----
    Index 0 of ``m_pred``, ``P_pred`` and ``F`` is never read (the scan
    slices ``[1:]``), because position 0 has no predecessor to smooth
    against. Callers may leave those slots as any placeholder.

    Returns
    -------
    m_smooth, P_smooth : (T, n) and (T, n, n)
        The final time step is not re-smoothed (``m_smooth[-1] == m_filt[-1]``).
    """
    # A zero-length filtered trajectory has no terminal state from which to
    # initialize the reverse scan. Its smoother is therefore the same empty
    # trajectory. The time dimension is static, so this branch is JIT-safe.
    if m_filt.shape[0] == 0:
        return m_filt, P_filt

    def backward_step(carry, inputs):
        m_s_next, P_s_next = carry
        m_f_t, P_f_t, m_p_next, P_p_next, F_next = inputs
        m_s, P_s = ekf_smooth_step(
            m_f_t,
            P_f_t,
            m_p_next,
            P_p_next,
            m_s_next,
            P_s_next,
            F_next,
        )
        return (m_s, P_s), (m_s, P_s)

    init_smooth = (m_filt[-1], P_filt[-1])
    bw_inputs = (m_filt[:-1], P_filt[:-1], m_pred[1:], P_pred[1:], F[1:])
    _, (m_s_rev, P_s_rev) = jax.lax.scan(
        backward_step,
        init_smooth,
        bw_inputs,
        reverse=True,
    )
    m_smooth = jnp.concatenate([m_s_rev, m_filt[-1:]], axis=0)
    P_smooth = jnp.concatenate([P_s_rev, P_filt[-1:]], axis=0)
    return m_smooth, P_smooth


def mlp_l2_penalty(mlp_params: Dict[str, Any]) -> Array:
    """Sum of squared MLP weights (entries whose key starts with 'w').

    The Hamiltonian models all penalise weights ``w*`` but not biases
    ``b*``; this helper centralises the convention.
    """
    return jnp.sum(
        jnp.array([jnp.sum(v**2) for k, v in mlp_params.items() if k.startswith("w")])
    )


def default_init_mean(n_oscillators: int) -> Array:
    """Default Hamiltonian latent state at t=0: positions=0.1, momenta=0."""
    return jnp.concatenate(
        [jnp.full((n_oscillators,), 0.1), jnp.zeros((n_oscillators,))]
    )


class _BaseModelStubs:
    """Mixin providing no-op implementations of BaseModel's abstract hooks.

    The Hamiltonian family is SGD-only — the linear-Gaussian EM hooks
    (``_initialize_measurement_matrix`` etc.) are not used by any
    Hamiltonian model. Each class previously defined six identical
    one-line ``pass`` stubs; this mixin defines them once.
    """

    def _initialize_measurement_matrix(self, key=None) -> None:
        return

    def _initialize_measurement_covariance(self) -> None:
        return

    def _initialize_continuous_transition_matrix(self) -> None:
        return

    def _initialize_process_covariance(self) -> None:
        return

    def _project_parameters(self) -> None:
        return

    def _check_sgd_initialized(self) -> None:
        # Safe no-op: unlike the lazily-initialized oscillator models (whose
        # _check_sgd_initialized guards the "constructed but never initialized"
        # state), every Hamiltonian model fully populates its parameters in
        # __init__, so there is no uninitialized state to detect here.
        return
