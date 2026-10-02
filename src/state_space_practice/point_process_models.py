"""Switching oscillator models with point-process (spike) observations.

This module provides structured model classes for switching state-space models
with spike observations, mirroring the Gaussian observation hierarchy in
``oscillator_models.py``:

- Common Oscillator Model (COM-PP): spike observation params switch
- Correlated Noise Model (CNM-PP): process noise covariance switches
- Directed Influence Model (DIM-PP): transition matrix switches

All models use the Laplace-EKF approach for point-process observations and
EM for parameter estimation.

References
----------
1. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2022). Switching Functional
   Network Models of Oscillatory Brain Dynamics. In 2022 56th Asilomar
   Conference on Signals, Systems, and Computers (IEEE), pp. 607-612.
2. Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
   Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
   Neural Computation 16, 971-998.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.em_driver import (
    restore_attributes,
    run_em,
    snapshot_attributes,
)
from state_space_practice.exceptions import NonFiniteLikelihoodError, NotFittedError
from state_space_practice.fitted_state import FittedAttribute, is_set
from state_space_practice.oscillator_utils import (
    DirectedInfluenceDynamicsMixin,
    canonicalize_correlated_noise_pair_parameters,
    constrain_correlated_noise_process_covariance,
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix_stack,
    construct_correlated_noise_process_covariance,
    construct_correlated_noise_process_covariance_stack,
    construct_stable_directed_influence_transition_stack,
    extract_correlated_noise_params_from_covariance_stack,
    optimize_dim_transition_params_joint_until_stationary,
    project_correlated_noise_process_covariance,
)
from state_space_practice.parameter_transforms import POSITIVE, UNCONSTRAINED
from state_space_practice.sgd_fitting import SGDParams, SGDParamSpec
from state_space_practice.switching_kalman import (
    compute_process_covariance_sufficient_stats,
    compute_transition_sufficient_stats,
    minimum_state_occupancy,
    optimize_dim_transition_params_joint,
    switching_kalman_maximization_step,
    warn_low_occupancy_states,
)
from state_space_practice.switching_point_process import (
    QRegularizationConfig,
    SwitchingPointProcessBase,
)
from state_space_practice.utils import (
    clip_eigenvalues,
    shift_to_psd,
    symmetrize,
    validate_count_array,
    validate_finite_array,
    validate_nonnegative_array,
    validate_unit_interval_array,
)

if TYPE_CHECKING:
    import numpy as np
    import optax

    from state_space_practice.oscillator_regularization import (
        OscillatorPenaltyConfig,
    )

logger = logging.getLogger(__name__)

# Bounds for the M-step init_cov eigenvalue clip in switching point-process
# models, *relative to the latent scale* (the mean per-dimension variance
# ``trace(init_cov) / n_latent`` of the initial init_cov of the fit; 1 for the
# default identity init_cov). Lower bound prevents the smoother at t=0 from
# collapsing to a point estimate; upper bound prevents a positive feedback loop
# with sparse observations (large init_cov → diffuse filter → larger smoother
# uncertainty → larger init_cov). Relative bounds keep the clip meaning the
# same whatever units the latent state is expressed in.
_INIT_COV_EIGVAL_MIN = 1e-4
_INIT_COV_EIGVAL_MAX = 2.0


def _validate_oscillator_parameters(
    freqs: Array, damping_coef: Array, process_variance: Array
) -> None:
    """Validate shared oscillator parameters at public model boundaries.

    Mirrors the Gaussian oscillator models (``CommonOscillatorModel``,
    ``CorrelatedNoiseModel``, ``DirectedInfluenceModel``): frequencies must be
    finite, damping must be finite and lie in ``[0, 1]``, and process variance
    must be finite and non-negative. Otherwise these flow into a NaN or
    unstable transition/process matrix undetected. Shape validation is left to
    the caller (it is model-specific).
    """
    validate_finite_array("freqs", freqs)
    validate_unit_interval_array("damping_coef", damping_coef)
    validate_nonnegative_array("process_variance", process_variance)


class BaseSwitchingPointProcessModel(SwitchingPointProcessBase):
    """Abstract base class for switching oscillator models with spike observations.

    The core EM machinery for switching linear dynamical systems observed
    through point-process (spike) observations -- the Laplace-EKF filter and
    GPB smoother E-step, the dynamics and spike-GLM M-steps, initialization
    and the SGD protocol -- is shared with ``SwitchingSpikeOscillatorModel``
    through ``switching_point_process.SwitchingPointProcessBase``.
    This class adds the structured-model EM driver: GMM warm initialization of
    the discrete states, best-iterate tracking with rollback, multi-restart
    fitting, clipped initial-state updates and an optional sticky transition
    prior. Subclasses must implement methods to initialize model-specific
    parameters (transition matrix, process covariance) and project them onto
    valid spaces.

    The observation model is a Poisson point-process with log-linear intensity:
        log(lambda_n(t)) = baseline_n + weights_n @ x_t

    Parameters
    ----------
    n_oscillators : int
        Number of latent oscillators. State dimension is 2 * n_oscillators.
    n_neurons : int
        Number of observed neurons (spike trains).
    n_discrete_states : int
        Number of discrete network states.
    sampling_freq : float
        Sampling frequency in Hz.
    dt : float
        Time bin width in seconds.
    discrete_transition_diag : ArrayLike | None, optional
        Diagonal of discrete transition matrix. Defaults to a ~1 s expected
        dwell time at ``sampling_freq``.
    stickiness : float, default=0.0
        Strength of the sticky Dirichlet prior on the discrete transition
        matrix (0 disables the prior).
    update_continuous_transition_matrix : bool, default=True
        Update A during M-step.
    update_process_cov : bool, default=True
        Update Q during M-step.
    update_discrete_transition_matrix : bool, default=True
        Update Z during M-step.
    update_spike_params : bool, default=True
        Update spike GLM params during M-step.
    separate_spike_params : bool, default=True
        If True, fit separate spike GLM per discrete state.
    update_init_mean : bool, default=True
        Update initial mean during M-step.
    update_init_cov : bool, default=True
        Update initial covariance during M-step.
    q_regularization : QRegularizationConfig | None, optional
        Trust-region and eigenvalue clipping for Q updates.
    spike_weight_l2 : float, default=0.01
        L2 regularization on spike GLM weights.
    spike_baseline_prior_l2 : float, default=0.0
        L2 regularization shrinking baselines toward empirical log-rate.
    max_newton_iter : int, default=3
        Newton iterations per Laplace-EKF update.
    line_search_beta : float, default=0.5
        Armijo line search parameter.
    smoother_type : str, default="gpb1"
        Smoother algorithm: "gpb1" or "gpb2".
    """

    _REPR_UPDATE_FLAG_LABELS = {
        "update_continuous_transition_matrix": "transition",
        "update_process_cov": "process_cov",
        "update_discrete_transition_matrix": "discrete_transition",
        "update_spike_params": "spike_params",
        "update_init_mean": "init_mean",
        "update_init_cov": "init_cov",
    }

    # Parameters and posteriors captured for EM rollback. Every JAX-array and
    # immutable-container attribute is *reassigned* (never mutated in place)
    # by the E/M steps, so a plain reference is a valid snapshot.
    _EM_SNAPSHOT_KEYS = (
        "init_mean",
        "init_cov",
        "init_discrete_state_prob",
        "discrete_transition_matrix",
        "continuous_transition_matrix",
        "process_cov",
        "spike_params",
        "smoother_state_cond_mean",
        "smoother_state_cond_cov",
        "smoother_discrete_state_prob",
        "smoother_joint_discrete_state_prob",
        "smoother_pair_cond_cross_cov",
        "smoother_pair_cond_means",
        "smoother_pair_cond_covs",
        "smoother_next_pair_cond_means",
        "freqs",
        "damping_coef",
        "process_variance",
        "phase_difference",
        "coupling_strength",
        "_current_osc_params",
        "_transition_suff_stats",
    )

    #: Marginal log-likelihood of the final EM iterate; set by ``fit``.
    log_likelihood_: FittedAttribute[float] = FittedAttribute()

    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        n_discrete_states: int,
        sampling_freq: float,
        dt: float,
        discrete_transition_diag: ArrayLike | None = None,
        stickiness: float = 0.0,
        update_continuous_transition_matrix: bool = True,
        update_process_cov: bool = True,
        update_discrete_transition_matrix: bool = True,
        update_spike_params: bool = True,
        separate_spike_params: bool = True,
        update_init_mean: bool = True,
        update_init_cov: bool = True,
        q_regularization: QRegularizationConfig | None = None,
        spike_weight_l2: float = 0.01,
        spike_baseline_prior_l2: float = 0.0,
        max_newton_iter: int = 3,
        line_search_beta: float = 0.5,
        smoother_type: str = "gpb1",
    ) -> None:
        super().__init__(
            n_oscillators,
            n_neurons,
            n_discrete_states,
            sampling_freq,
            dt,
            discrete_transition_diag=discrete_transition_diag,
            update_continuous_transition_matrix=update_continuous_transition_matrix,
            update_process_cov=update_process_cov,
            update_discrete_transition_matrix=update_discrete_transition_matrix,
            update_spike_params=update_spike_params,
            separate_spike_params=separate_spike_params,
            update_init_mean=update_init_mean,
            update_init_cov=update_init_cov,
            q_regularization=q_regularization,
            spike_weight_l2=spike_weight_l2,
            spike_baseline_prior_l2=spike_baseline_prior_l2,
            max_newton_iter=max_newton_iter,
            line_search_beta=line_search_beta,
            smoother_type=smoother_type,
        )

        # Dirichlet prior for transition matrix (sticky prior)
        from state_space_practice.contingency_belief import get_transition_prior

        self.transition_prior = (
            get_transition_prior(
                concentration=1.0,
                stickiness=stickiness,
                n_states=n_discrete_states,
            )
            if stickiness > 0
            else None
        )

    def _default_discrete_transition_diag(self) -> Array:
        """~1 s expected dwell: ``p_stay = 1 - 1 / (dwell_sec * sampling_freq)``."""
        expected_dwell_sec = 1.0
        p_stay = 1.0 - 1.0 / (expected_dwell_sec * self.sampling_freq)
        if not 0.0 <= p_stay <= 1.0:
            raise ValueError(
                f"Computed default self-transition probability "
                f"p_stay={p_stay:g} is outside [0, 1] (from "
                f"sampling_freq={self.sampling_freq} Hz and a "
                f"{expected_dwell_sec}s expected dwell time). This occurs when "
                f"sampling_freq < 1 / expected_dwell_sec; pass "
                f"discrete_transition_diag explicitly for low sampling rates."
            )
        return jnp.full((self.n_discrete_states,), p_stay)

    def decode(self) -> Array:
        """Return the most likely discrete state at each time step.

        Returns
        -------
        states : Array, shape (n_time,)
            Argmax of smoother discrete state probabilities.

        Raises
        ------
        RuntimeError
            If called before fit() or fit_sgd().
        """
        if not is_set(self, "smoother_discrete_state_prob"):
            raise NotFittedError("Call fit() or fit_sgd() before decode().")
        return jnp.argmax(self.smoother_discrete_state_prob, axis=1)

    def predict_proba(self) -> Array:
        """Return smoothed discrete state probabilities.

        Returns
        -------
        probs : Array, shape (n_time, n_discrete_states)
            Posterior probability of each discrete state at each time step.

        Raises
        ------
        RuntimeError
            If called before fit() or fit_sgd().
        """
        if not is_set(self, "smoother_discrete_state_prob"):
            raise NotFittedError("Call fit() or fit_sgd() before predict_proba().")
        return self.smoother_discrete_state_prob

    # ------------------------------------------------------------------
    # Warm initialization
    # ------------------------------------------------------------------

    def _warm_initialize_states(self, spikes: Array) -> None:
        """Warm-initialize discrete state probabilities from spike statistics.

        Uses a Gaussian mixture model on windowed per-neuron spike features
        (mean rate + rate variance) to segment data into approximate discrete
        states. This captures both rate changes (COM-PP) and variance changes
        (CNM-PP) for symmetry breaking.

        Parameters
        ----------
        spikes : Array, shape (n_time, n_neurons)
            Observed spike counts.
        """
        import numpy as np_cpu

        n_time = spikes.shape[0]
        n_states = self.n_discrete_states
        spikes_np = np_cpu.array(spikes)

        window, n_windows = self._warm_init_windows(n_time)
        if n_windows < n_states * 2:
            # Not enough windows — fall back to uniform
            self.smoother_discrete_state_prob = jnp.ones((n_time, n_states)) / n_states
            self.smoother_joint_discrete_state_prob = (
                jnp.ones((n_time - 1, n_states, n_states)) / n_states**2
            )
            return

        # Reshape into windows and compute features
        trimmed = spikes_np[: n_windows * window]
        windowed = trimmed.reshape(n_windows, window, -1)
        # Features: per-neuron mean rate, per-neuron variance,
        # and windowed spectral features (power at oscillator frequencies)
        means = windowed.mean(axis=1)  # (n_windows, n_neurons)
        variances = windowed.var(axis=1)  # (n_windows, n_neurons)

        features = np_cpu.concatenate([means, variances], axis=1)
        self._set_state_probs_from_window_features(features, window, n_time)

    def _warm_init_windows(self, n_time: int) -> tuple[int, int]:
        """Window length and count for warm-init feature extraction.

        ~50 timesteps (0.5 s at 100 Hz) balances temporal resolution against
        statistical stability and is short enough to resolve 8-25 Hz
        oscillators; the floor of 10 keeps very short recordings usable.
        """
        window = max(min(50, n_time // (2 * self.n_discrete_states)), 10)
        return window, n_time // window

    def _set_state_probs_from_window_features(
        self, features: np.ndarray, window: int, n_time: int
    ) -> None:
        """Cluster windowed features with a GMM and store per-timestep state probs.

        Window-level responsibilities are expanded to every timestep in the
        window (the last window's row is tiled over any remainder), softened
        to a minimum probability of ``0.05`` for numerical safety, and the
        joint adjacent-timestep probabilities are set from the marginals.
        """
        import numpy as np_cpu
        from sklearn.mixture import GaussianMixture

        n_states = self.n_discrete_states
        n_windows = features.shape[0]
        gmm = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            n_init=5,
            random_state=0,
        )
        gmm.fit(features)
        window_probs = gmm.predict_proba(features)  # (n_windows, n_states)

        probs_np = np_cpu.repeat(window_probs, window, axis=0)
        if n_time > n_windows * window:
            remainder = n_time - n_windows * window
            probs_np = np_cpu.concatenate(
                [probs_np, np_cpu.tile(window_probs[-1], (remainder, 1))]
            )
        probs = jnp.array(probs_np[:n_time])
        probs = probs * 0.9 + 0.05 / n_states
        probs = probs / probs.sum(axis=1, keepdims=True)

        self.smoother_discrete_state_prob = probs
        joint = probs[:-1, :, None] * probs[1:, None, :]
        joint = joint / jnp.sum(joint, axis=(1, 2), keepdims=True)
        self.smoother_joint_discrete_state_prob = joint

    # ------------------------------------------------------------------
    # M-step regularization of the initial-state estimates
    # ------------------------------------------------------------------

    def _regularize_init_mean_update(self, init_mean: Array) -> Array:
        """Clip initial mean updates to a plausible latent-state range."""
        return jnp.clip(init_mean, -10.0, 10.0)

    def _init_cov_latent_scale(self) -> float:
        """Latent scale the init_cov clip bounds are relative to.

        The mean per-dimension variance ``trace(init_cov) / n_latent`` of the
        init_cov recorded when parameters were last initialised (a
        ``fit`` without ``skip_init``). Warm restarts (``skip_init=True``)
        keep that scale, so a clipped init_cov cannot raise its own ceiling.
        When nothing has been recorded yet, the current init_cov is recorded.
        """
        scale = getattr(self, "_init_cov_reference_scale", None)
        if scale is None:
            scale = self._record_init_cov_latent_scale()
        return scale

    def _record_init_cov_latent_scale(self) -> float:
        """Record ``mean_j trace(init_cov_j) / n_latent`` as the latent scale."""
        traces = jnp.trace(jnp.asarray(self.init_cov), axis1=0, axis2=1)
        scale = float(jnp.mean(traces)) / self.n_latent
        if not (scale > 0.0 and jnp.isfinite(scale)):
            logger.warning(
                "init_cov has a non-finite or non-positive mean trace (%s); "
                "using latent scale 1.0 for the M-step init_cov clip bounds.",
                scale,
            )
            scale = 1.0
        self._init_cov_reference_scale = scale
        return scale

    def _regularize_init_cov_update(self, init_cov: Array) -> Array:
        """Clip initial covariance eigenvalues for sparse-spike EM stability.

        The bounds are ``[_INIT_COV_EIGVAL_MIN, _INIT_COV_EIGVAL_MAX]`` times
        the latent scale (:meth:`_init_cov_latent_scale`), so the clip is
        invariant to the units of the latent state. A warning is logged when
        the clip changes an eigenvalue.
        """
        latent_scale = self._init_cov_latent_scale()
        eig_min = _INIT_COV_EIGVAL_MIN * latent_scale
        eig_max = _INIT_COV_EIGVAL_MAX * latent_scale
        clip_init_cov_eigenvalues = functools.partial(
            clip_eigenvalues,
            min_eigenvalue=eig_min,
            max_eigenvalue=eig_max,
        )

        def _per_state_eigrange(P: Array) -> tuple[Array, Array]:
            eigs = jnp.linalg.eigvalsh(symmetrize(P))
            return jnp.min(eigs), jnp.max(eigs)

        per_state_min, per_state_max = jax.vmap(_per_state_eigrange, in_axes=-1)(
            init_cov
        )
        raw_min = float(jnp.min(per_state_min))
        raw_max = float(jnp.max(per_state_max))
        init_cov = jax.vmap(clip_init_cov_eigenvalues, in_axes=-1, out_axes=-1)(
            init_cov
        )
        if raw_min < eig_min or raw_max > eig_max:
            logger.warning(
                "M-step init_cov eigenvalues clipped to [%.2e, %.2e] "
                "(= [%.0e, %.1f] x latent scale %.3g; raw range across "
                "discrete states: [%.2e, %.2e]). Frequent triggering indicates "
                "the smoother at t=0 is numerically diffuse -- consider "
                "tightening the prior or shortening the sequence.",
                eig_min,
                eig_max,
                _INIT_COV_EIGVAL_MIN,
                _INIT_COV_EIGVAL_MAX,
                latent_scale,
                raw_min,
                raw_max,
            )
        return init_cov

    # ------------------------------------------------------------------
    # EM state snapshots
    # ------------------------------------------------------------------

    def _snapshot_em_state(self) -> dict[str, object]:
        """Snapshot parameters and posteriors for EM rollback.

        Restoring a reference snapshot is unaffected by later reassignment
        (see ``_EM_SNAPSHOT_KEYS``). Only ``_current_osc_params`` (a mutable
        ``dict | None`` warm-start cache for the reparameterized M-step) is
        deep-copied, so an in-place edit of the live dict cannot leak into the
        snapshot.
        """
        return snapshot_attributes(
            self, self._EM_SNAPSHOT_KEYS, deepcopy_keys=("_current_osc_params",)
        )

    def _restore_em_state(self, state: dict[str, object]) -> None:
        """Restore a snapshot produced by ``_snapshot_em_state``."""
        restore_attributes(self, self._EM_SNAPSHOT_KEYS, state)

    # ------------------------------------------------------------------
    # EM loop
    # ------------------------------------------------------------------

    def fit(
        self,
        spikes: ArrayLike,
        max_iter: int = 50,
        tol: float = 1e-4,
        key: Array | None = None,
        skip_init: bool = False,
        n_restarts: int = 1,
    ) -> list[float]:
        """Fit the model to spike data using EM.

        Parameters
        ----------
        spikes : ArrayLike, shape (n_time, n_neurons)
            Observed spike counts.
        max_iter : int, default=50
            Maximum EM iterations.
        tol : float, default=1e-4
            Convergence tolerance for relative log-likelihood change.
        key : Array | None, optional
            JAX random key for initialization. Defaults to PRNGKey(0).
        skip_init : bool, default=False
            If True, skip initialization (use existing parameters).
        n_restarts : int, default=1
            Number of random restarts. Each restart uses a different random
            key. The run with the best final log-likelihood is kept.

        Returns
        -------
        log_likelihoods : list[float]
            Marginal log-likelihood at each iteration (from the best restart).
        """
        spikes = jnp.asarray(spikes)

        if spikes.ndim != 2:
            raise ValueError(
                f"spikes must be 2D with shape (n_time, n_neurons), "
                f"got {spikes.ndim}D with shape {spikes.shape}"
            )
        if spikes.shape[1] != self.n_neurons:
            raise ValueError(
                f"spikes shape[1] must match n_neurons={self.n_neurons}, "
                f"got shape {spikes.shape}"
            )
        validate_count_array(spikes, "spikes")

        if key is None:
            key = jax.random.PRNGKey(0)

        if n_restarts > 1 and not skip_init:
            return self._fit_multi_restart(spikes, max_iter, tol, key, n_restarts)

        return self._fit_single(spikes, max_iter, tol, key, skip_init)

    def _fit_single(
        self,
        spikes: Array,
        max_iter: int,
        tol: float,
        key: Array,
        skip_init: bool = False,
    ) -> list[float]:
        """Single EM run with warm initialization."""
        if not skip_init:
            self._initialize_parameters(key)
            self._warm_initialize_states(spikes)
            self._record_init_cov_latent_scale()
            # Set placeholder smoother outputs needed by spike M-step
            n_time = spikes.shape[0]
            self.smoother_state_cond_mean = jnp.zeros(
                (n_time, self.n_latent, self.n_discrete_states)
            )
            self.smoother_state_cond_cov = jnp.stack(
                [jnp.eye(self.n_latent)] * self.n_discrete_states, axis=2
            )[None].repeat(n_time, axis=0)
            self.smoother_pair_cond_cross_cov = jnp.zeros(
                (
                    n_time - 1,
                    self.n_latent,
                    self.n_latent,
                    self.n_discrete_states,
                    self.n_discrete_states,
                )
            )
            # Not computed until the first E-step; drop any from a prior fit.
            del self.smoother_pair_cond_means
            del self.smoother_pair_cond_covs
            del self.smoother_next_pair_cond_means
            self._m_step_spikes(spikes)
        else:
            self._init_cov_latent_scale()

        def _m_step() -> None:
            self._m_step_dynamics()
            self._m_step_spikes(spikes)
            self._project_parameters()

        # Approximate (GPB) EM: a decrease beyond ``tol`` is logged but
        # iteration continues, and the best accepted state is restored at the
        # end if the final iterate is worse. A non-finite first E-step means
        # the initial parameters are unusable. ``require_increase_to_converge``
        # is not passed: with ``decrease_tol == tol`` it is a no-op (a step
        # within ``tol`` can never be a decrease beyond ``tol``), so omitting it
        # states what the call actually does without changing behavior.
        result = run_em(
            lambda: float(self._e_step(spikes)),
            _m_step,
            self._snapshot_em_state,
            self._restore_em_state,
            max_iter=max_iter,
            tol=tol,
            on_first_nonfinite="raise",
            stop_on_decrease=False,
            track_best=True,
            logger=logger,
        )
        self.converged_ = result.converged
        log_likelihoods = result.log_likelihoods
        if log_likelihoods:
            self.log_likelihood_ = float(log_likelihoods[-1])

        return log_likelihoods

    def _fit_multi_restart(
        self,
        spikes: Array,
        max_iter: int,
        tol: float,
        key: Array,
        n_restarts: int,
    ) -> list[float]:
        """Run EM with multiple random restarts, keep the best.

        Each restart uses a different random key for initialization.
        The run with the highest final log-likelihood is kept, and the
        model's parameters are set to those of the best run.
        """
        best_lls: list[float] | None = None
        best_final_ll = -float("inf")
        best_state: dict[str, object] | None = None

        keys = jax.random.split(key, n_restarts)

        for restart in range(n_restarts):
            try:
                lls = self._fit_single(spikes, max_iter, tol, keys[restart])
                final_ll = float(
                    getattr(self, "log_likelihood_", lls[-1] if lls else -float("inf"))
                )

                if final_ll > best_final_ll:
                    best_final_ll = final_ll
                    best_lls = lls
                    best_state = self._snapshot_em_state()

                logger.info(
                    f"Restart {restart + 1}/{n_restarts}: final LL={final_ll:.4f}"
                )
            except NonFiniteLikelihoodError:
                logger.warning(
                    f"Restart {restart + 1}/{n_restarts}: failed (non-finite LL)"
                )
                continue

        if best_state is None or best_lls is None:
            raise ValueError(
                f"All {n_restarts} restarts failed with non-finite log-likelihood."
            )

        # Restore best model state
        self._restore_em_state(best_state)

        # Re-run E-step to populate all smoother outputs for the best params
        self.log_likelihood_ = float(self._e_step(spikes))

        return best_lls


# ==========================================================================
# Common Oscillator Model (COM-PP)
# ==========================================================================


class CommonOscillatorPointProcessModel(BaseSwitchingPointProcessModel):
    """Common Oscillator Model with point-process observations (COM-PP).

    The **spike observation parameters** (baseline, weights) switch across
    discrete states, while the dynamics (A) and process noise (Q) are
    constant. This is the point-process analog of ``CommonOscillatorModel``.

    Different discrete states represent different ways the shared oscillators
    drive neural spiking — e.g., a neuron may be strongly modulated by theta
    in one state but not another.

    Parameters
    ----------
    n_oscillators : int
        Number of latent oscillators.
    n_neurons : int
        Number of observed neurons.
    n_discrete_states : int
        Number of discrete network states.
    sampling_freq : float
        Sampling frequency in Hz.
    dt : float
        Time bin width in seconds.
    freqs : ArrayLike, shape (n_oscillators,)
        Intrinsic oscillation frequencies in Hz.
    damping_coef : ArrayLike, shape (n_oscillators,)
        Damping coefficients for each oscillator (0 to 1).
    process_variance : ArrayLike, shape (n_oscillators,)
        Process noise variance for each oscillator.
    """

    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        n_discrete_states: int,
        sampling_freq: float,
        dt: float,
        freqs: ArrayLike,
        damping_coef: ArrayLike,
        process_variance: ArrayLike,
        **kwargs: Any,
    ) -> None:
        # Force COM-specific update flags
        kwargs["update_continuous_transition_matrix"] = False
        kwargs["update_process_cov"] = False
        kwargs.setdefault("separate_spike_params", True)
        super().__init__(
            n_oscillators, n_neurons, n_discrete_states, sampling_freq, dt, **kwargs
        )

        freqs = jnp.asarray(freqs)
        damping_coef = jnp.asarray(damping_coef)
        process_variance = jnp.asarray(process_variance)
        if freqs.shape != (n_oscillators,):
            raise ValueError(f"freqs shape {freqs.shape} != ({n_oscillators},)")
        if damping_coef.shape != (n_oscillators,):
            raise ValueError(
                f"damping_coef shape {damping_coef.shape} != ({n_oscillators},)"
            )
        if process_variance.shape != (n_oscillators,):
            raise ValueError(
                f"process_variance shape {process_variance.shape} != ({n_oscillators},)"
            )
        _validate_oscillator_parameters(freqs, damping_coef, process_variance)

        self.freqs = freqs
        self.damping_coef = damping_coef
        self.process_variance = process_variance

    def _initialize_continuous_transition_matrix(self) -> None:
        """A is constant across states: uncoupled oscillators."""
        self.continuous_transition_matrix = (
            construct_common_oscillator_transition_matrix_stack(
                self.freqs,
                self.damping_coef,
                self.sampling_freq,
                self.n_discrete_states,
            )
        )

    def _initialize_process_covariance(self) -> None:
        """Q is constant across states: block-diagonal from process_variance."""
        process_cov = construct_common_oscillator_process_covariance(
            variance=self.process_variance,
        )
        self.process_cov = jnp.stack([process_cov] * self.n_discrete_states, axis=2)

    def _warm_initialize_states(self, spikes: Array) -> None:
        """Warm-initialize using spectral features.

        COM-PP states differ in which oscillator modulates the neurons
        (e.g., theta vs beta), so windowed spectral band power ratios
        are better features than rate alone.
        """
        import numpy as np_cpu

        n_time = spikes.shape[0]
        n_states = self.n_discrete_states
        spikes_np = np_cpu.array(spikes)

        window, n_windows = self._warm_init_windows(n_time)
        if n_windows < n_states * 2:
            super()._warm_initialize_states(spikes)
            return

        trimmed = spikes_np[: n_windows * window]
        windowed = trimmed.reshape(n_windows, window, -1)

        # Per-neuron mean rates
        means = windowed.mean(axis=1)

        # Spectral: band power in oscillator frequency ranges
        total_per_window = windowed.sum(axis=2)
        total_per_window = total_per_window - total_per_window.mean(
            axis=1, keepdims=True
        )
        freqs_fft = np_cpu.fft.rfftfreq(window, d=1.0 / self.sampling_freq)
        power = np_cpu.abs(np_cpu.fft.rfft(total_per_window, axis=1)) ** 2
        power_sum = power.sum(axis=1, keepdims=True) + 1e-10
        power_norm = power / power_sum

        # Band power at each oscillator's frequency ± 2Hz
        spectral_features = []
        for freq in np_cpu.array(self.freqs):
            mask = (freqs_fft >= freq - 2) & (freqs_fft <= freq + 2)
            if mask.any():
                spectral_features.append(power_norm[:, mask].sum(axis=1, keepdims=True))
        if spectral_features:
            spectral = np_cpu.concatenate(spectral_features, axis=1)
            features = np_cpu.concatenate([means, spectral], axis=1)
        else:
            features = means

        self._set_state_probs_from_window_features(features, window, n_time)

    def _project_parameters(self) -> None:
        """No projection needed — A and Q are not updated."""
        pass

    def _build_param_spec(self) -> tuple[SGDParams, SGDParamSpec]:
        return self._shared_sgd_param_spec()


# ==========================================================================
# Correlated Noise Model (CNM-PP)
# ==========================================================================


class CorrelatedNoisePointProcessModel(BaseSwitchingPointProcessModel):
    """Correlated Noise Model with point-process observations (CNM-PP).

    The **process noise covariance (Q)** switches across discrete states,
    while the dynamics (A) are constant. This is the point-process analog
    of ``CorrelatedNoiseModel``.

    Different discrete states represent different patterns of shared
    stochastic drive between oscillators, implying functional connectivity
    changes without direct dynamical coupling.

    Parameters
    ----------
    n_oscillators : int
        Number of latent oscillators.
    n_neurons : int
        Number of observed neurons.
    n_discrete_states : int
        Number of discrete network states.
    sampling_freq : float
        Sampling frequency in Hz.
    dt : float
        Time bin width in seconds.
    freqs : ArrayLike, shape (n_oscillators,)
        Intrinsic oscillation frequencies in Hz.
    damping_coef : ArrayLike, shape (n_oscillators,)
        Damping coefficients for each oscillator.
    process_variance : ArrayLike, shape (n_oscillators, n_discrete_states)
        Process noise variance per oscillator per state.
    phase_difference : ArrayLike, shape (n_oscillators, n_oscillators, n_discrete_states)
        Phase differences for noise correlation. Each oscillator pair may be
        supplied in the strict upper triangle, strict lower triangle, or both
        triangles if the two entries are opposite phases; values are stored
        canonically in the strict upper triangle.
    coupling_strength : ArrayLike, shape (n_oscillators, n_oscillators, n_discrete_states)
        Coupling strengths for noise correlation. Each oscillator pair may be
        supplied in the strict upper triangle, strict lower triangle, or both
        triangles if the two entries agree; values are stored canonically in the
        strict upper triangle.
    use_reparameterized_mstep : bool, default=True
        If True (default), use the exact joint constrained CNM covariance
        M-step. It updates variance, coupling, and phase together from the
        fixed-transition residual covariance while guaranteeing CNM block
        structure and positive semidefiniteness. If False, use the generic
        covariance M-step (evaluated at the fixed ``A``) followed by structural
        projection.
    """

    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        n_discrete_states: int,
        sampling_freq: float,
        dt: float,
        freqs: ArrayLike,
        damping_coef: ArrayLike,
        process_variance: ArrayLike,
        phase_difference: ArrayLike,
        coupling_strength: ArrayLike,
        use_reparameterized_mstep: bool = True,
        **kwargs: Any,
    ) -> None:
        # Force CNM-specific update flags
        kwargs["update_continuous_transition_matrix"] = False
        kwargs["update_process_cov"] = True
        super().__init__(
            n_oscillators, n_neurons, n_discrete_states, sampling_freq, dt, **kwargs
        )

        freqs = jnp.asarray(freqs)
        damping_coef = jnp.asarray(damping_coef)
        process_variance = jnp.asarray(process_variance)
        phase_difference = jnp.asarray(phase_difference)
        coupling_strength = jnp.asarray(coupling_strength)
        if freqs.shape != (n_oscillators,):
            raise ValueError(f"freqs shape {freqs.shape} != ({n_oscillators},)")
        if damping_coef.shape != (n_oscillators,):
            raise ValueError(
                f"damping_coef shape {damping_coef.shape} != ({n_oscillators},)"
            )
        if process_variance.shape != (n_oscillators, n_discrete_states):
            raise ValueError(
                f"process_variance shape {process_variance.shape} "
                f"!= ({n_oscillators}, {n_discrete_states})"
            )
        if phase_difference.shape != (
            n_oscillators,
            n_oscillators,
            n_discrete_states,
        ):
            raise ValueError(
                f"phase_difference shape {phase_difference.shape} "
                f"!= ({n_oscillators}, {n_oscillators}, {n_discrete_states})"
            )
        if coupling_strength.shape != (
            n_oscillators,
            n_oscillators,
            n_discrete_states,
        ):
            raise ValueError(
                f"coupling_strength shape {coupling_strength.shape} "
                f"!= ({n_oscillators}, {n_oscillators}, {n_discrete_states})"
            )
        _validate_oscillator_parameters(freqs, damping_coef, process_variance)

        phase_difference, coupling_strength = (
            canonicalize_correlated_noise_pair_parameters(
                phase_difference, coupling_strength
            )
        )

        self.freqs = freqs
        self.damping_coef = damping_coef
        self.process_variance = process_variance
        self.phase_difference = phase_difference
        self.coupling_strength = coupling_strength
        self.use_reparameterized_mstep = use_reparameterized_mstep

    def _initialize_continuous_transition_matrix(self) -> None:
        """A is constant across states: uncoupled oscillators."""
        self.continuous_transition_matrix = (
            construct_common_oscillator_transition_matrix_stack(
                self.freqs,
                self.damping_coef,
                self.sampling_freq,
                self.n_discrete_states,
            )
        )

    def _initialize_process_covariance(self) -> None:
        """Q varies across states: correlated noise structure."""
        self.process_cov = construct_correlated_noise_process_covariance_stack(
            self.process_variance, self.phase_difference, self.coupling_strength
        )

    def _project_parameters(self) -> None:
        """Enforce the PSD CNM covariance family after the generic M-step.

        The default path structurally projects the generic covariance and syncs
        the public scientific parameters. The opt-in path replaces that estimate
        with the exact fixed-A constrained covariance update.
        """
        if not self.update_process_cov:
            return

        if self.use_reparameterized_mstep:
            self._m_step_constrained_process_covariance()
            return

        self.process_cov = jnp.stack(
            [
                project_correlated_noise_process_covariance(self.process_cov[:, :, j])
                for j in range(self.n_discrete_states)
            ],
            axis=-1,
        )
        self._sync_process_covariance_params()

    def _m_step_constrained_process_covariance(self) -> None:
        """Install the exact fixed-A, PSD, jointly constrained CNM Q update."""
        residual_scatter, state_counts = compute_process_covariance_sufficient_stats(
            continuous_transition_matrix=self.continuous_transition_matrix,
            state_cond_smoother_means=self.smoother_state_cond_mean,
            state_cond_smoother_covs=self.smoother_state_cond_cov,
            smoother_discrete_state_prob=self.smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
            pair_cond_smoother_means=self.smoother_pair_cond_means,
            pair_cond_smoother_covs=getattr(self, "smoother_pair_cond_covs", None),
            next_pair_cond_smoother_means=getattr(
                self, "smoother_next_pair_cond_means", None
            ),
        )

        previous = construct_correlated_noise_process_covariance_stack(
            self.process_variance, self.phase_difference, self.coupling_strength
        )
        # A state with fewer than n_cont_states + 1 expected transitions has an
        # unidentified residual covariance: keep its previous Q (and warn).
        min_count = minimum_state_occupancy(residual_scatter.shape[0])
        warn_low_occupancy_states(
            state_counts,
            min_count,
            "CorrelatedNoisePointProcessModel constrained Q M-step",
            "their process covariance kept its previous value",
        )
        updated = []
        cfg = self.q_regularization
        for j in range(self.n_discrete_states):
            count = state_counts[j]
            target = residual_scatter[..., j] / jnp.maximum(count, 1e-12)
            constrained = constrain_correlated_noise_process_covariance(target)
            candidate = jnp.where(count >= min_count, constrained, previous[..., j])

            if cfg.enabled:
                candidate = (
                    cfg.trust_region_weight * candidate
                    + (1.0 - cfg.trust_region_weight) * previous[..., j]
                )
                if cfg.min_eigenvalue is not None or cfg.max_eigenvalue is not None:
                    candidate = clip_eigenvalues(
                        candidate, cfg.min_eigenvalue, cfg.max_eigenvalue
                    )
                candidate = constrain_correlated_noise_process_covariance(candidate)
            candidate = jnp.where(count >= min_count, candidate, previous[..., j])
            updated.append(candidate)

        self.process_cov = jnp.stack(updated, axis=-1)
        self._sync_process_covariance_params()

    def _sync_process_covariance_params(self) -> None:
        """Synchronize public CNM parameters from the structured Q stack."""
        params = extract_correlated_noise_params_from_covariance_stack(
            self.process_cov, self.n_oscillators
        )
        self.process_variance = params["variance"]
        self.phase_difference = params["phase_difference"]
        self.coupling_strength = params["coupling_strength"]

    # --- SGDFittableMixin: CNM-PP specific ---

    def _build_param_spec(self) -> tuple[SGDParams, SGDParamSpec]:
        params, spec = self._shared_sgd_param_spec()

        if self.update_process_cov:
            params["process_variance"] = self.process_variance
            spec["process_variance"] = POSITIVE
            params["phase_difference"] = self.phase_difference
            spec["phase_difference"] = UNCONSTRAINED
            params["coupling_strength"] = self.coupling_strength
            spec["coupling_strength"] = UNCONSTRAINED

        return params, spec

    def _sgd_loss_fn(self, params: SGDParams, spikes: jax.Array) -> jax.Array:
        # Reconstruct per-state Q from scientific params
        proc_var = params.get("process_variance", self.process_variance)
        phase_diff = params.get("phase_difference", self.phase_difference)
        coupling = params.get("coupling_strength", self.coupling_strength)

        # Vectorize Q construction over discrete states (last axis)
        Q = jax.vmap(
            construct_correlated_noise_process_covariance,
            in_axes=(-1, -1, -1),
            out_axes=-1,
        )(proc_var, phase_diff, coupling)
        # coupling_strength is UNCONSTRAINED, so SGD can propose a coupling that
        # makes the reconstructed Q indefinite, which NaN-poisons the filter and
        # the gradient. shift_to_psd is a gradient-safe barrier: identity while Q
        # is PSD, a smooth lift back to the cone otherwise.
        Q = jax.vmap(shift_to_psd, in_axes=-1, out_axes=-1)(Q)

        # Inject reconstructed Q into the base loss function via params
        params_with_Q = dict(params)
        params_with_Q["_Q"] = Q
        return super()._sgd_loss_fn(params_with_Q, spikes)

    def _store_sgd_params(self, params: SGDParams) -> None:
        super()._store_sgd_params(params)
        if "process_variance" in params:
            self.process_variance = params["process_variance"]
        if "phase_difference" in params:
            self.phase_difference = params["phase_difference"]
        if "coupling_strength" in params:
            self.coupling_strength = params["coupling_strength"]
        if any(k in params for k in ("phase_difference", "coupling_strength")):
            self.phase_difference, self.coupling_strength = (
                canonicalize_correlated_noise_pair_parameters(
                    self.phase_difference, self.coupling_strength
                )
            )
        # Reconstruct Q from updated params, applying the same gradient-safe PSD
        # shift the SGD loss used (coupling_strength is UNCONSTRAINED, so the raw
        # reconstruction can be indefinite). Matching the loss's projection keeps
        # the stored process_cov identical to what the optimizer evaluated.
        if any(
            k in params
            for k in ("process_variance", "phase_difference", "coupling_strength")
        ):
            Q_list = []
            for j in range(self.n_discrete_states):
                Q_j = construct_correlated_noise_process_covariance(
                    variance=self.process_variance[:, j],
                    phase_difference=self.phase_difference[..., j],
                    coupling_strength=self.coupling_strength[..., j],
                )
                Q_list.append(shift_to_psd(Q_j))
            self.process_cov = jnp.stack(Q_list, axis=-1)
            self._sync_process_covariance_params()


# ==========================================================================
# Directed Influence Model (DIM-PP)
# ==========================================================================


class DirectedInfluencePointProcessModel(
    DirectedInfluenceDynamicsMixin, BaseSwitchingPointProcessModel
):
    """Directed Influence Model with point-process observations (DIM-PP).

    The **continuous transition matrix (A)** switches across discrete states,
    while the process noise (Q) is constant. This is the point-process analog
    of ``DirectedInfluenceModel``.

    Different discrete states represent different patterns of directed
    dynamical coupling between oscillators — e.g., CA1 driving PFC at theta
    in one state, PFC driving CA1 in another.

    Parameters
    ----------
    n_oscillators : int
        Number of latent oscillators.
    n_neurons : int
        Number of observed neurons.
    n_discrete_states : int
        Number of discrete network states.
    sampling_freq : float
        Sampling frequency in Hz.
    dt : float
        Time bin width in seconds.
    freqs : ArrayLike, shape (n_oscillators,)
        Intrinsic oscillation frequencies in Hz.
    damping_coef : ArrayLike, shape (n_oscillators,)
        Damping coefficients for each oscillator.
    process_variance : ArrayLike, shape (n_oscillators,)
        Process noise variance (constant across states).
    phase_difference : ArrayLike, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial coupling phase differences.
    coupling_strength : ArrayLike, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial coupling strengths.
    use_reparameterized_mstep : bool, default=False
        If True, optimize oscillator parameters directly (guarantees valid
        oscillator structure). If False, use standard M-step with projection.
    max_spectral_radius : float, default=0.99
        Target upper bound on the spectral radius of each state's transition
        matrix. The differentiable stability scale shrinks damping and coupling
        so the largest spectral radius across states stays at or below this
        value (stable parameters are left unchanged). A
        larger radius (closer to one) permits longer memory and a narrower
        spectral peak: the resolvable half-power bandwidth is
        ``delta_f ~= (1 - radius) * fs / pi``, so at ``fs = 1 kHz`` the default
        ``0.99`` floors the bandwidth near ``3.2 Hz`` -- too broad to isolate a
        slow, narrow-band rhythm such as delta. Raise it toward ``0.999`` to
        resolve such rhythms; lowering it increases damping (broader, more
        overdamped bands). Must lie in ``(0, 1)``.
    max_damping : float, default=0.995
        Upper bound on the intrinsic per-oscillator damping used by the
        reparameterized M-step's bounded optimizer. Must lie in ``(0, 1)``.
    """

    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        n_discrete_states: int,
        sampling_freq: float,
        dt: float,
        freqs: ArrayLike,
        damping_coef: ArrayLike,
        process_variance: ArrayLike,
        phase_difference: ArrayLike,
        coupling_strength: ArrayLike,
        use_reparameterized_mstep: bool = False,
        max_spectral_radius: float = 0.99,
        max_damping: float = 0.995,
        **kwargs: Any,
    ) -> None:
        # Force DIM-specific update flags
        kwargs["update_continuous_transition_matrix"] = True
        kwargs["update_process_cov"] = False
        super().__init__(
            n_oscillators, n_neurons, n_discrete_states, sampling_freq, dt, **kwargs
        )

        freqs = jnp.asarray(freqs)
        damping_coef = jnp.asarray(damping_coef)
        process_variance = jnp.asarray(process_variance)
        phase_difference = jnp.asarray(phase_difference)
        coupling_strength = jnp.asarray(coupling_strength)
        if freqs.shape != (n_oscillators,):
            raise ValueError(f"freqs shape {freqs.shape} != ({n_oscillators},)")
        if damping_coef.shape != (n_oscillators,):
            raise ValueError(
                f"damping_coef shape {damping_coef.shape} != ({n_oscillators},)"
            )
        if process_variance.shape != (n_oscillators,):
            raise ValueError(
                f"process_variance shape {process_variance.shape} != ({n_oscillators},)"
            )

        _validate_oscillator_parameters(freqs, damping_coef, process_variance)

        if phase_difference.shape != (
            n_oscillators,
            n_oscillators,
            n_discrete_states,
        ):
            raise ValueError(
                f"phase_difference shape {phase_difference.shape} "
                f"!= ({n_oscillators}, {n_oscillators}, {n_discrete_states})"
            )
        if coupling_strength.shape != (
            n_oscillators,
            n_oscillators,
            n_discrete_states,
        ):
            raise ValueError(
                f"coupling_strength shape {coupling_strength.shape} "
                f"!= ({n_oscillators}, {n_oscillators}, {n_discrete_states})"
            )

        if not bool(jnp.all(jnp.isfinite(phase_difference))) or not bool(
            jnp.all(jnp.isfinite(coupling_strength))
        ):
            raise ValueError(
                "phase_difference and coupling_strength must contain only finite "
                "values."
            )
        diag_idx = jnp.arange(n_oscillators)
        diag_phase = phase_difference[diag_idx, diag_idx, :]
        diag_coupling = coupling_strength[diag_idx, diag_idx, :]
        if bool(jnp.any(jnp.abs(diag_phase) > 1e-8)):
            raise ValueError(
                "DIM-PP phase_difference diagonal entries are ignored by the "
                "transition model and must be zero."
            )
        if bool(jnp.any(jnp.abs(diag_coupling) > 1e-8)):
            raise ValueError(
                "DIM-PP coupling_strength diagonal entries are ignored by the "
                "transition model and must be zero."
            )

        self.freqs = freqs
        self.damping_coef = damping_coef
        self.process_variance = process_variance
        self.phase_difference = phase_difference.at[diag_idx, diag_idx, :].set(0.0)
        self.coupling_strength = coupling_strength.at[diag_idx, diag_idx, :].set(0.0)
        self.use_reparameterized_mstep = use_reparameterized_mstep
        self._current_osc_params: dict[str, Array] | None = None

        # Stability bounds applied when rebuilding transition matrices.
        if not 0.0 < max_spectral_radius < 1.0:
            raise ValueError("max_spectral_radius must lie in (0, 1).")
        if not 0.0 < max_damping < 1.0:
            raise ValueError("max_damping must lie in (0, 1).")
        self.max_spectral_radius = max_spectral_radius
        self.max_damping = max_damping

    def _initialize_process_covariance(self) -> None:
        """Q is constant across states: block-diagonal from process_variance."""
        process_cov = construct_common_oscillator_process_covariance(
            variance=self.process_variance,
        )
        self.process_cov = jnp.stack([process_cov] * self.n_discrete_states, axis=2)

    def _m_step_dynamics(self) -> None:
        """M-step with optional reparameterized transition update."""
        if self.use_reparameterized_mstep:
            self._m_step_reparameterized()
            return

        if self.update_continuous_transition_matrix:
            self._remember_pre_m_step_dynamics()
        super()._m_step_dynamics()

    def _m_step_reparameterized(self) -> None:
        """M-step using reparameterized optimization for A.

        Optimizes oscillator parameters (damping, freq, coupling_strength,
        phase_diff) directly, guaranteeing valid oscillator structure.
        """
        # Standard M-step for non-A parameters
        n_time = self.smoother_state_cond_mean.shape[0]
        dummy_obs = jnp.zeros((n_time, 1))

        (
            _,  # A — computed below via reparameterized optimization
            _,  # measurement_matrix
            _,  # Q — not updated for DIM
            _,  # measurement_cov
            new_init_mean,
            new_init_cov,
            new_discrete_transition,
            new_init_discrete_prob,
        ) = switching_kalman_maximization_step(
            obs=dummy_obs,
            state_cond_smoother_means=self.smoother_state_cond_mean,
            state_cond_smoother_covs=self.smoother_state_cond_cov,
            smoother_discrete_state_prob=self.smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
            pair_cond_smoother_means=self.smoother_pair_cond_means,
            pair_cond_smoother_covs=getattr(self, "smoother_pair_cond_covs", None),
            next_pair_cond_smoother_means=getattr(
                self, "smoother_next_pair_cond_means", None
            ),
            transition_prior=self.transition_prior,
            fixed_continuous_transition_matrix=self.continuous_transition_matrix,
            estimate_measurement_params=False,
        )

        if self.update_discrete_transition_matrix:
            self.discrete_transition_matrix = new_discrete_transition
        if self.update_init_mean:
            self.init_mean = self._regularize_init_mean_update(new_init_mean)
        if self.update_init_cov:
            self.init_cov = self._regularize_init_cov_update(new_init_cov)
        self.init_discrete_state_prob = new_init_discrete_prob

        # Reparameterized optimization for A. Forward GPB2's pair-conditioned
        # covariance and next-step means when available; otherwise
        # compute_transition_sufficient_stats falls back to the state-conditioned
        # approximation. Omitting them would silently discard GPB2's accuracy
        # here (the standard M-step and the Gaussian DIM both pass them).
        gamma1, beta = compute_transition_sufficient_stats(
            state_cond_smoother_means=self.smoother_state_cond_mean,
            state_cond_smoother_covs=self.smoother_state_cond_cov,
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
            pair_cond_smoother_means=self.smoother_pair_cond_means,
            pair_cond_smoother_covs=getattr(self, "smoother_pair_cond_covs", None),
            next_pair_cond_smoother_means=getattr(
                self, "smoother_next_pair_cond_means", None
            ),
        )

        # Initialize the joint warm start from the current shared/public
        # scientific parameters (matching the Gaussian DirectedInfluenceModel).
        if self._current_osc_params is None:
            self._current_osc_params = self._intrinsic_osc_params()

        # Jointly optimize the shared frequency/damping and per-state
        # coupling/phase in ONE objective, rather than optimizing each state
        # independently and averaging the inconsistent shared parameters.
        self._current_osc_params = (
            optimize_dim_transition_params_joint_until_stationary(
                gamma1=gamma1,
                beta=beta,
                init_params=self._current_osc_params,
                sampling_freq=self.sampling_freq,
                process_cov=self.process_cov,
                max_spectral_radius=self.max_spectral_radius,
                max_damping=self.max_damping,
                optimizer=optimize_dim_transition_params_joint,
            )
        )
        self._update_public_oscillator_params()

        # Build A from the effective (globally stabilized) parameters -- the same
        # differentiable stability scale the joint objective used. Public
        # freq/damping/coupling stay intrinsic; A re-applies the scale, so it is
        # reconstructable via compute_directed_influence_stability_scale.
        self._rebuild_stable_transition_matrix()

    # --- SGDFittableMixin: DIM-PP specific ---

    def _build_param_spec(self) -> tuple[SGDParams, SGDParamSpec]:
        params, spec = self._shared_sgd_param_spec()

        if self.update_continuous_transition_matrix:
            params["phase_difference"] = self.phase_difference
            spec["phase_difference"] = UNCONSTRAINED
            params["coupling_strength"] = self.coupling_strength
            spec["coupling_strength"] = UNCONSTRAINED

        return params, spec

    def fit_sgd(
        self,
        spikes: ArrayLike,
        key: Array | None = None,
        optimizer: optax.GradientTransformation | None = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: float | None = None,
        connectivity_penalty: OscillatorPenaltyConfig | None = None,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        SGD optimizes ``coupling_strength`` and ``phase_difference``
        (and optionally spike params, discrete transition, init params).
        Frequencies (``freqs``) and damping (``damping_coef``) are frozen
        during SGD and used as constants.

        Parameters
        ----------
        spikes : ArrayLike, shape (n_time, n_neurons)
        key : Array or None
        optimizer : optax optimizer or None
        num_steps : int
        verbose : bool
        convergence_tol : float or None
        connectivity_penalty : OscillatorPenaltyConfig or None
            If provided, adds structured sparsity penalties on
            coupling_strength during SGD optimization.

        Returns
        -------
        log_likelihoods : list of float
        """
        self._connectivity_penalty = connectivity_penalty
        return super().fit_sgd(
            spikes,
            key=key,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    def _sgd_loss_fn(self, params: SGDParams, spikes: jax.Array) -> jax.Array:
        phase_diff = params.get("phase_difference", self.phase_difference)
        coupling = params.get("coupling_strength", self.coupling_strength)

        # Apply the shared differentiable stability scale so SGD optimizes only
        # over transition matrices that honor max_spectral_radius, matching the
        # Gaussian DIM SGD loss and the EM path. Damping is fixed during SGD
        # (only coupling/phase are free), so the scale depends on the free
        # coupling and the fixed damping.
        params_with_A = dict(params)
        params_with_A["_A"] = construct_stable_directed_influence_transition_stack(
            self.freqs,
            self.damping_coef,
            coupling,
            phase_diff,
            self.sampling_freq,
            max_spectral_radius=self.max_spectral_radius,
        )
        base_loss = super()._sgd_loss_fn(params_with_A, spikes)

        # Add connectivity penalty if configured
        penalty_config = getattr(self, "_connectivity_penalty", None)
        if penalty_config is not None:
            from state_space_practice.oscillator_regularization import (
                total_connectivity_penalty,
            )

            coupling_transposed = jnp.moveaxis(coupling, -1, 0)
            base_loss = base_loss + total_connectivity_penalty(
                coupling_transposed,
                penalty_config,
                n_timesteps=self._n_timesteps,
            )

        return base_loss

    def _store_sgd_params(self, params: SGDParams) -> None:
        super()._store_sgd_params(params)
        if "phase_difference" in params:
            self.phase_difference = params["phase_difference"]
        if "coupling_strength" in params:
            self.coupling_strength = params["coupling_strength"]
        if "phase_difference" in params or "coupling_strength" in params:
            # Rebuild through the shared stability scale so stored matrices honor
            # max_spectral_radius and stay reconstructable from the public params.
            self._rebuild_stable_transition_matrix()
