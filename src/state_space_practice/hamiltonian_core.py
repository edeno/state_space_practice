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

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import Array

from state_space_practice.kalman import joseph_form_update
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    ekf_predict_step,
    ekf_predict_step_with_jacobian,
    ekf_smooth_step,
)
from state_space_practice.point_process_kalman import (
    _soft_expected_count_and_log,
    glm_laplace_update,
    poisson_family,
)
from state_space_practice.utils import psd_cholesky, psd_logdet


def gaussian_measurement_update(
    m_pred: Array,
    P_pred: Array,
    y: Array,
    C: Array,
    d: Array,
    R: Array,
    *,
    include_normalization_const: bool = True,
) -> tuple[Array, Array, Array]:
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

    logdet = psd_logdet(S_cho)
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
) -> tuple[Array, Array, Array]:
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
) -> tuple[Array, Array]:
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


def mlp_l2_penalty(mlp_params: dict[str, Any]) -> Array:
    """Sum of squared MLP weights (entries whose key starts with 'w').

    The Hamiltonian models all penalise weights ``w*`` but not biases
    ``b*``; this helper centralises the convention.
    """
    return jnp.sum(
        jnp.array([jnp.sum(v**2) for k, v in mlp_params.items() if k.startswith("w")])
    )


def run_ekf_filter(
    observations: Any,
    init_mean: Array,
    init_cov: Array,
    trans_params: dict[str, Any],
    process_cov: Array,
    dt: float,
    update_fn: Callable[[Array, Array, Any], tuple[Array, Array, Array]],
) -> tuple[Array, Array, Array]:
    """Scan an EKF (Hamiltonian predict + ``update_fn``) over ``observations``.

    ``observations`` is any pytree whose leaves share a leading time axis (one
    array, or a tuple of arrays for multi-modality models); ``update_fn(m_pred,
    P_pred, y_t)`` returns ``(m_post, P_post, log_likelihood_t)``.
    """

    def step(carry, y_t):
        m_prev, P_prev = carry
        m_pred, P_pred = ekf_predict_step(
            m_prev, P_prev, trans_params, apply_mlp, process_cov, dt
        )
        m_post, P_post, ll = update_fn(m_pred, P_pred, y_t)
        return (m_post, P_post), (m_post, P_post, ll)

    _, (means, covs, lls) = jax.lax.scan(step, (init_mean, init_cov), observations)
    return means, covs, lls


def run_ekf_smoother(
    observations: Any,
    init_mean: Array,
    init_cov: Array,
    trans_params: dict[str, Any],
    process_cov: Array,
    dt: float,
    update_fn: Callable[[Array, Array, Any], tuple[Array, Array]],
) -> tuple[Array, Array]:
    """EKF forward pass (with Jacobians) followed by the RTS backward pass.

    Same conventions as :func:`run_ekf_filter`, except ``update_fn`` returns
    only ``(m_post, P_post)`` (no log-likelihood is needed for smoothing).
    """

    def forward_step(carry, y_t):
        m_prev, P_prev = carry
        m_pred, P_pred, F_t = ekf_predict_step_with_jacobian(
            m_prev, P_prev, trans_params, apply_mlp, process_cov, dt
        )
        m_post, P_post = update_fn(m_pred, P_pred, y_t)
        return (m_post, P_post), (m_post, P_post, m_pred, P_pred, F_t)

    _, (m_f, P_f, m_p, P_p, F) = jax.lax.scan(
        forward_step, (init_mean, init_cov), observations
    )
    return ekf_rts_backward_pass(m_f, P_f, m_p, P_p, F)


def poisson_rollout_nll(log_lambda: Array, spikes: Array, dt: float) -> Array:
    """Overflow-safe, gradient-preserving Poisson negative log-likelihood.

    An unclipped ``exp`` overflows to ``+inf`` on a divergent rollout
    (``0 * log(inf)`` / ``inf - inf`` give NaN); a hard clip would zero the
    gradient above the cap and freeze SGD. ``_soft_expected_count_and_log``
    continues ``exp`` logarithmically past the cap and returns ``log(mu)``
    analytically under that same cap: ``log_rates`` equals ``log(rates)``
    wherever ``rates`` is representable and positive, and stays finite (not
    ``-inf``) where ``rates`` underflows to 0, so a positive spike count keeps
    a finite restoring gradient there that ``log(rates + eps)`` would kill.
    """
    rates, log_rates = _soft_expected_count_and_log(log_lambda, dt)
    return jnp.sum(rates - spikes * log_rates + jax.scipy.special.gammaln(spikes + 1.0))


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

    def _rollout_trajectory(self, params: dict[str, Any], n_time: int) -> Array:
        """Deterministic Hamiltonian rollout of ``n_time`` steps from ``init_mean``.

        Used by the ``use_filter=False`` surrogate SGD loss, which scores the
        noise-free trajectory against the observations (no process prior, no
        latent uncertainty) to warm-start the dynamics.
        """
        trans_params = {**params["mlp"], "omega": params["omega"]}

        def scan_fn(x_prev, _):
            x_next = self.transition_func(x_prev, trans_params)
            return x_next, x_next

        _, x_traj = jax.lax.scan(scan_fn, params["init_mean"], None, length=n_time)
        return x_traj

    def fit(self, *args, **kwargs):
        """Hamiltonian models do not support linear EM."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support the linear EM path "
            "(fit()). Please use fit_sgd() for non-linear optimization."
        )

    def _finalize_sgd(self, *data, **kwargs):
        """Run filter + smoother to populate fitted states after SGD.

        ``data`` is whatever ``fit_sgd`` passed positionally (one observation
        array for the single-modality models, LFP and spikes for the joint
        model); ``filter`` and ``smooth`` take the same positional layout.
        """
        params = self._build_param_spec()[0]
        means, covs, lls = self.filter(*data, params)
        self.filtered_means_ = means
        self.filtered_covs_ = covs
        self.log_likelihood_ = float(jnp.sum(lls))
        sm_means, sm_covs = self.smooth(*data, params)
        self.smoothed_means_ = sm_means
        self.smoothed_covs_ = sm_covs

