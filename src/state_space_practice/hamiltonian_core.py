"""Shared EKF / Laplace-EKF machinery for the Hamiltonian model family.

Contents
--------
- Stateless per-step helpers: :func:`gaussian_measurement_update`,
  :func:`point_process_laplace_update`, :func:`ekf_rts_backward_pass`, the
  scan drivers :func:`run_ekf_filter` / :func:`run_ekf_smoother`, and the
  loss helpers :func:`mlp_l2_penalty` / :func:`poisson_rollout_nll`.
- Module-level jitted cores :func:`hamiltonian_ekf_filter` /
  :func:`hamiltonian_ekf_smoother` for the single-regime models. They take
  every mutable quantity in a ``params`` dict and only ``dt`` and the
  observation-model name as static arguments, so model instances with the
  same configuration share one compilation.
- :class:`HamiltonianModelBase`, the scaffold the LFP, spike, joint and
  switching models inherit: ``OscillatorParameterBase`` containers plus
  ``SGDFittableMixin``, with the shared ``fit_sgd`` hooks.

Conventions
-----------
- State, covariance, and observation arguments are JAX arrays. Configuration
  arguments such as ``dt`` and likelihood-control booleans are static Python
  values; declare them static when JIT-compiling a helper directly.
- Helpers return JAX arrays and do not capture Python state; closures over
  external arrays inside ``jax.lax.scan`` bodies must come from the caller,
  not from this module.
- Log-likelihoods returned by the per-step helpers are that step's
  contribution. Callers accumulate via the scan ``carry`` or via ``jnp.sum``
  over the scan output, depending on filter/smoother convention.

See docs/hamiltonian_architecture.md for the design rationale (no
linear-Gaussian EM integration, SGD-only fitting).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import joseph_form_update
from state_space_practice.nonlinear_dynamics import (
    apply_mlp,
    ekf_predict_step,
    ekf_predict_step_with_jacobian,
    ekf_smooth_step,
    leapfrog_step,
)
from state_space_practice.oscillator_models import OscillatorParameterBase
from state_space_practice.point_process_kalman import (
    _soft_expected_count_and_log,
    glm_laplace_update,
    poisson_family,
)
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import psd_cholesky, psd_logdet, stabilize_covariance


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

    Parameters
    ----------
    m_pred : Array, shape (n,)
        Predicted (prior) state mean.
    P_pred : Array, shape (n, n)
        Predicted (prior) state covariance.
    y : Array, shape (n_obs,)
        Observation at this step. ``n_obs = 0`` is allowed and returns the
        prior unchanged with zero log-likelihood.
    C : Array, shape (n_obs, n)
        Observation matrix.
    d : Array, shape (n_obs,)
        Observation offset.
    R : Array, shape (n_obs, n_obs)
        Observation noise covariance (positive definite).
    include_normalization_const : bool, default True
        Whether to include the ``-0.5 * n_obs * log(2π)`` term in the
        log-likelihood.

    Returns
    -------
    m_post : Array, shape (n,)
    P_post : Array, shape (n, n)
        Joseph-form update, for PSD preservation.
    log_likelihood : Array, shape ()
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
    static_argnames=("dt", "compute_log_likelihood"))``. The Hamiltonian
    models pass ``dt`` as a static argument of the module-level jitted cores
    (:func:`hamiltonian_ekf_filter` and friends), which bind it here.

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

    Parameters
    ----------
    observations : pytree of Array
        Leaves share a leading time axis of length ``n_time``: one
        ``(n_time, n_obs)`` array, or a tuple of such arrays for the
        multi-modality models (e.g. ``(lfp, spikes)``).
    init_mean : Array, shape (n_latent,)
        Prior mean of the state before the first prediction.
    init_cov : Array, shape (n_latent, n_latent)
        Prior covariance of the state before the first prediction.
    trans_params : dict
        MLP parameters plus ``"omega"`` for the leapfrog transition.
    process_cov : Array, shape (n_latent, n_latent)
        Additive process-noise covariance ``Q``.
    dt : float
        Leapfrog step (static when jitted).
    update_fn : callable
        ``update_fn(m_pred, P_pred, y_t) -> (m_post, P_post,
        log_likelihood_t)`` with ``y_t`` the time-``t`` slice of
        ``observations``.

    Returns
    -------
    filtered_means : Array, shape (n_time, n_latent)
    filtered_covs : Array, shape (n_time, n_latent, n_latent)
    log_likelihoods : Array, shape (n_time,)
        Per-step contributions ``log p(y_t | y_{1:t-1})`` as returned by
        ``update_fn``.
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

    Parameters
    ----------
    observations, init_mean, init_cov, trans_params, process_cov, dt
        As in :func:`run_ekf_filter`.
    update_fn : callable
        ``update_fn(m_pred, P_pred, y_t) -> (m_post, P_post)``; no
        log-likelihood is needed for smoothing.

    Returns
    -------
    smoothed_means : Array, shape (n_time, n_latent)
    smoothed_covs : Array, shape (n_time, n_latent, n_latent)
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


OBSERVATION_MODELS = ("gaussian", "poisson", "joint")


FilterUpdate = Callable[[Array, Array, Any], tuple[Array, Array, Array]]
SmootherUpdate = Callable[[Array, Array, Any], tuple[Array, Array]]


def _observation_updates(
    observation_model: str, params: dict[str, Any], dt: float
) -> tuple[FilterUpdate, SmootherUpdate]:
    """Build the per-step ``(filter_update, smoother_update)`` closures.

    ``observation_model`` names the readout of a single-regime Hamiltonian
    model: ``"gaussian"`` (LFP; reads ``C``, ``d``, ``R``), ``"poisson"``
    (spikes; ``C``, ``d``) or ``"joint"`` (an LFP update followed by a spike
    update on the same latent state; ``C_lfp``, ``d_lfp``, ``R_lfp``,
    ``C_spikes``, ``d_spikes``). The filter update returns
    ``(m_post, P_post, log_likelihood)``; the smoother update returns
    ``(m_post, P_post)`` and skips the likelihood normalization constants,
    which the RTS pass never reads.
    """
    if observation_model == "gaussian":
        C, d, R = params["C"], params["d"], params["R"]

        def filter_update(m, P, y):
            return gaussian_measurement_update(m, P, y, C, d, R)

        def smoother_update(m, P, y):
            return gaussian_measurement_update(
                m, P, y, C, d, R, include_normalization_const=False
            )[:2]

    elif observation_model == "poisson":
        C, d = params["C"], params["d"]

        def filter_update(m, P, y):
            return point_process_laplace_update(m, P, y, C, d, dt)

        def smoother_update(m, P, y):
            return point_process_laplace_update(
                m, P, y, C, d, dt, compute_log_likelihood=False
            )[:2]

    elif observation_model == "joint":
        C_l, d_l, R_l = params["C_lfp"], params["d_lfp"], params["R_lfp"]
        C_s, d_s = params["C_spikes"], params["d_spikes"]

        def filter_update(m, P, y):
            # Sequential: LFP update first, then point-process update on the
            # LFP posterior. The two log-likelihoods sum to the joint marginal
            # because the observations are conditionally independent given x_t.
            y_lfp, y_spike = y
            m_mid, P_mid, ll_lfp = gaussian_measurement_update(
                m, P, y_lfp, C_l, d_l, R_l
            )
            m_post, P_post, ll_spike = point_process_laplace_update(
                m_mid, P_mid, y_spike, C_s, d_s, dt
            )
            return m_post, P_post, ll_lfp + ll_spike

        def smoother_update(m, P, y):
            y_lfp, y_spike = y
            m_mid, P_mid, _ = gaussian_measurement_update(
                m, P, y_lfp, C_l, d_l, R_l, include_normalization_const=False
            )
            m_post, P_post, _ = point_process_laplace_update(
                m_mid, P_mid, y_spike, C_s, d_s, dt, compute_log_likelihood=False
            )
            return m_post, P_post

    else:
        raise ValueError(
            f"observation_model must be one of {OBSERVATION_MODELS}, "
            f"got {observation_model!r}."
        )
    return filter_update, smoother_update


@partial(jax.jit, static_argnames=("dt", "observation_model"))
def hamiltonian_ekf_filter(
    observations: Any,
    params: dict[str, Any],
    *,
    dt: float,
    observation_model: str,
) -> tuple[Array, Array, Array]:
    """JIT-compiled EKF filter shared by the single-regime Hamiltonian models.

    ``observations`` is the per-step observation pytree (one array, or an
    ``(lfp, spikes)`` tuple for ``"joint"``). ``params`` carries every
    mutable quantity: ``mlp``, ``omega``, the readout parameters named in
    :func:`_observation_updates`, ``init_mean``, ``init_cov`` and ``Q``.
    Only ``dt`` and ``observation_model`` are static, so the compile cache is
    keyed on array shapes/dtypes plus those two hashable scalars: model
    instances with the same configuration share one compilation, and nothing
    read from a model object can go stale inside the cache.

    Returns ``(filtered_means, filtered_covs, log_likelihoods)``, each with a
    leading time axis.
    """
    trans_params = {**params["mlp"], "omega": params["omega"]}
    filter_update, _ = _observation_updates(observation_model, params, dt)
    return run_ekf_filter(
        observations,
        params["init_mean"],
        params["init_cov"],
        trans_params,
        params["Q"],
        dt,
        filter_update,
    )


@partial(jax.jit, static_argnames=("dt", "observation_model"))
def hamiltonian_ekf_smoother(
    observations: Any,
    params: dict[str, Any],
    *,
    dt: float,
    observation_model: str,
) -> tuple[Array, Array]:
    """JIT-compiled EKF-RTS smoother shared by the single-regime models.

    Same conventions and caching behaviour as :func:`hamiltonian_ekf_filter`.
    Returns ``(smoothed_means, smoothed_covs)``.
    """
    trans_params = {**params["mlp"], "omega": params["omega"]}
    _, smoother_update = _observation_updates(observation_model, params, dt)
    return run_ekf_smoother(
        observations,
        params["init_mean"],
        params["init_cov"],
        trans_params,
        params["Q"],
        dt,
        smoother_update,
    )


def poisson_rollout_nll(log_lambda: Array, spikes: Array, dt: float) -> Array:
    """Overflow-safe, gradient-preserving Poisson negative log-likelihood.

    Computes ``sum(mu - y * log(mu) + log(y!))`` with expected counts
    ``mu = exp(log_lambda) * dt``, i.e. the full normalized Poisson NLL
    (``log(y!)`` enters as ``gammaln(y + 1)``), so it is on the same scale as
    the Gaussian NLL it is added to in the joint surrogate.

    An unclipped ``exp`` overflows to ``+inf`` on a divergent rollout
    (``0 * log(inf)`` / ``inf - inf`` give NaN); a hard clip would zero the
    gradient above the cap and freeze SGD. ``_soft_expected_count_and_log``
    continues ``exp`` logarithmically past the cap and returns ``log(mu)``
    analytically under that same cap: ``log_rates`` equals ``log(rates)``
    wherever ``rates`` is representable and positive, and stays finite (not
    ``-inf``) where ``rates`` underflows to 0, so a positive spike count keeps
    a finite restoring gradient there that ``log(rates + eps)`` would kill.

    Parameters
    ----------
    log_lambda : Array, shape (n_time, n_neurons)
        Log firing rate in Hz.
    spikes : Array, shape (n_time, n_neurons)
        Spike counts per bin.
    dt : float
        Bin width in seconds.

    Returns
    -------
    nll : Array, shape ()
        Negative log-likelihood summed over time and neurons.
    """
    rates, log_rates = _soft_expected_count_and_log(log_lambda, dt)
    return jnp.sum(rates - spikes * log_rates + jax.scipy.special.gammaln(spikes + 1.0))


def default_init_mean(n_oscillators: int) -> Array:
    """Default Hamiltonian latent state at t=0: positions=0.1, momenta=0."""
    return jnp.concatenate(
        [jnp.full((n_oscillators,), 0.1), jnp.zeros((n_oscillators,))]
    )


class HamiltonianModelBase(OscillatorParameterBase, SGDFittableMixin):
    """Shared scaffold of the SGD-only Hamiltonian model family.

    Combines the oscillator parameter containers with ``SGDFittableMixin``
    and nothing else: no EM layer, no abstract initialization hooks and no
    ``fit`` -- ``fit_sgd`` is the only fitting entry point. Concrete models
    add their observation head(s), the validating ``filter`` / ``smooth``
    wrappers, ``_build_param_spec``, ``_sgd_loss_fn`` and
    ``_validate_fit_data``, and name their readout in ``_observation_model``.

    Each concrete model also declares a thin public ``fit_sgd`` with its own
    named data arguments (``observations`` or ``lfp_obs, spike_obs``) followed
    by ``optimizer, num_steps, verbose, convergence_tol, use_filter, l2_reg``,
    so the data can be passed by name and the optimizer positionally. It
    only forwards to ``SGDFittableMixin.fit_sgd``; validation happens once in
    ``_prepare_sgd_data``.
    """

    #: Readout name passed to the module-level jitted cores.
    _observation_model: ClassVar[str]

    #: Whether the model infers a discrete-state posterior. The single-regime
    #: models have one dynamical regime and nothing for ``decode`` /
    #: ``predict_proba`` to report; only the switching model sets this.
    _has_discrete_states: ClassVar[bool] = False

    _sgd_param_attrs = {"mlp": "mlp_params", "omega": "omega"}

    def __init__(
        self,
        n_oscillators: int,
        n_discrete_states: int,
        n_sources: int,
        sampling_freq: float,
        hidden_dims: list[int] | None,
        seed: int,
    ):
        super().__init__(
            n_oscillators=n_oscillators,
            n_discrete_states=n_discrete_states,
            n_sources=n_sources,
            sampling_freq=sampling_freq,
        )
        # A plain float: ``dt`` is a hashable static argument of the jitted
        # cores, so every instance with the same rate shares their cache.
        self.dt = 1.0 / float(sampling_freq)
        self.hidden_dims = hidden_dims or [32, 32]
        self.key = jax.random.PRNGKey(seed)

    def transition_func(self, x: Array, params: dict[str, Array]) -> Array:
        """Deterministic Hamiltonian transition."""
        return leapfrog_step(x, params, apply_mlp, self.dt)

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
        return cast(Array, x_traj)

    def _discrete_state_posterior(self, caller: str) -> Array:
        """Reject discrete-state decoding on a single-regime model.

        Raises
        ------
        NotImplementedError
            If the model has a single dynamical regime.
        """
        if not self._has_discrete_states:
            raise NotImplementedError(
                f"{type(self).__name__} has a single dynamical regime and no "
                f"discrete-state posterior, so {caller}() is undefined. Use "
                "SwitchingHamiltonianJointModel to infer regime switches."
            )
        return super()._discrete_state_posterior(caller)

    def _filter_jit(
        self, observations: Any, params: dict[str, Any]
    ) -> tuple[Array, Array, Array]:
        """Run the jitted filter core on validated data and completed params."""
        return cast(
            tuple[Array, Array, Array],
            hamiltonian_ekf_filter(
                observations,
                params,
                dt=self.dt,
                observation_model=self._observation_model,
            ),
        )

    def _smooth_jit(
        self, observations: Any, params: dict[str, Any]
    ) -> tuple[Array, Array]:
        """Run the jitted smoother core on validated data and completed params."""
        return cast(
            tuple[Array, Array],
            hamiltonian_ekf_smoother(
                observations,
                params,
                dt=self.dt,
                observation_model=self._observation_model,
            ),
        )

    def _validate_fit_data(self, *data: Any, **kwargs: Any) -> tuple[Array, ...]:
        """Validate the observation arrays given to ``fit_sgd`` (non-empty).

        Each model declares its own positional layout (``observations`` or
        ``lfp_obs, spike_obs``) and returns the validated arrays as a tuple.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _validate_fit_data."
        )

    def _prepare_sgd_data(
        self,
        *data: ArrayLike,
        use_filter: bool = True,
        l2_reg: float = 1e-4,
    ) -> tuple[tuple[Array, ...], dict[str, Any]]:
        """Validate the data for ``fit_sgd`` and record its length.

        ``use_filter`` and ``l2_reg`` are forwarded to ``_sgd_loss_fn``. The
        Hamiltonian models allocate their parameters in ``__init__`` and take
        no initialization ``key``; the public ``fit_sgd`` signatures reject it
        with a ``TypeError``.
        """
        validated = self._validate_fit_data(*data)
        self._sgd_n_time = validated[0].shape[0]
        return validated, {"use_filter": use_filter, "l2_reg": l2_reg}

    def _store_sgd_params(self, params: dict) -> None:
        """Store the plain keys, then the single-regime derived ones.

        ``init_mean`` is optimized as the ``(n_cont_states,)`` slice of the
        single discrete state and written back into slot 0; ``Q`` is the
        single-state process covariance, re-stabilized after the unconstrained
        round trip. Subclasses resync their observation containers on top.
        """
        super()._store_sgd_params(params)
        if "init_mean" in params:
            self.init_mean = self.init_mean.at[:, 0].set(params["init_mean"])
        if "Q" in params:
            self.process_cov = jnp.stack([stabilize_covariance(params["Q"])], axis=2)

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
