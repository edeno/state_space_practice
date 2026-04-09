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

import logging
from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.kalman import symmetrize
from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
    construct_correlated_noise_process_covariance,
    construct_directed_influence_transition_matrix,
    extract_dim_params_from_matrix,
    project_coupled_transition_matrix,
    project_matrix_blockwise,
)
from state_space_practice.switching_kalman import (
    compute_transition_q_function,
    compute_transition_sufficient_stats,
    optimize_dim_transition_params,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
    switching_kalman_smoother_gpb2,
)
from state_space_practice.switching_point_process import (
    QRegularizationConfig,
    SpikeObsParams,
    switching_point_process_filter,
    update_spike_glm_params,
)
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import check_converged, make_discrete_transition_matrix

logger = logging.getLogger(__name__)


class BaseSwitchingPointProcessModel(ABC, SGDFittableMixin):
    """Abstract base class for switching oscillator models with spike observations.

    This class provides the core EM machinery for switching linear dynamical
    systems observed through point-process (spike) observations. Subclasses
    must implement methods to initialize model-specific parameters
    (transition matrix, process covariance) and project them onto valid spaces.

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
    discrete_transition_diag : Array | None, optional
        Diagonal of discrete transition matrix. Defaults to 0.95.
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
    max_newton_iter : int, default=1
        Newton iterations per Laplace-EKF update.
    line_search_beta : float, default=0.5
        Armijo line search parameter.
    smoother_type : str, default="gpb1"
        Smoother algorithm: "gpb1" or "gpb2".
    """

    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        n_discrete_states: int,
        sampling_freq: float,
        dt: float,
        discrete_transition_diag: Array | None = None,
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
        max_newton_iter: int = 1,
        line_search_beta: float = 0.5,
        smoother_type: str = "gpb1",
    ) -> None:
        if n_oscillators <= 0:
            raise ValueError(f"n_oscillators must be positive. Got {n_oscillators}.")
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive. Got {n_neurons}.")
        if n_discrete_states <= 0:
            raise ValueError(
                f"n_discrete_states must be positive. Got {n_discrete_states}."
            )
        if sampling_freq <= 0:
            raise ValueError(f"sampling_freq must be positive. Got {sampling_freq}.")
        if dt <= 0:
            raise ValueError(f"dt must be positive. Got {dt}.")
        if discrete_transition_diag is not None:
            discrete_transition_diag = jnp.asarray(discrete_transition_diag)
            if discrete_transition_diag.shape != (n_discrete_states,):
                raise ValueError(
                    f"discrete_transition_diag shape mismatch: expected "
                    f"({n_discrete_states},), got {discrete_transition_diag.shape}."
                )
            if not jnp.all(
                (discrete_transition_diag >= 0) & (discrete_transition_diag <= 1)
            ):
                raise ValueError(
                    "discrete_transition_diag values must be probabilities in [0, 1]."
                )
        if spike_weight_l2 < 0:
            raise ValueError(
                f"spike_weight_l2 must be non-negative. Got {spike_weight_l2}."
            )
        if spike_baseline_prior_l2 < 0:
            raise ValueError(
                f"spike_baseline_prior_l2 must be non-negative, got {spike_baseline_prior_l2}"
            )
        if q_regularization is not None:
            if not (0.0 <= q_regularization.trust_region_weight <= 1.0):
                raise ValueError(
                    "q_regularization.trust_region_weight must be in [0, 1]."
                )
        if smoother_type not in ("gpb1", "gpb2"):
            raise ValueError(
                f"smoother_type must be 'gpb1' or 'gpb2', got '{smoother_type}'"
            )

        self.n_oscillators = n_oscillators
        self.n_neurons = n_neurons
        self.n_discrete_states = n_discrete_states
        self.sampling_freq = sampling_freq
        self.dt = dt
        self.n_latent = 2 * n_oscillators

        if discrete_transition_diag is None:
            self.discrete_transition_diag = jnp.full(
                (n_discrete_states,), 0.95, dtype=jnp.float32
            )
        else:
            self.discrete_transition_diag = discrete_transition_diag

        self.update_continuous_transition_matrix = update_continuous_transition_matrix
        self.update_process_cov = update_process_cov
        self.update_discrete_transition_matrix = update_discrete_transition_matrix
        self.update_spike_params = update_spike_params
        self.separate_spike_params = separate_spike_params
        self.update_init_mean = update_init_mean
        self.update_init_cov = update_init_cov

        self.spike_weight_l2 = spike_weight_l2
        self.spike_baseline_prior_l2 = spike_baseline_prior_l2
        self.q_regularization = q_regularization or QRegularizationConfig()

        self.max_newton_iter = max_newton_iter
        self.line_search_beta = line_search_beta
        self.smoother_type = smoother_type

        # Placeholders for model parameters (initialized by _initialize_parameters)
        self.init_mean: Array
        self.init_cov: Array
        self.init_discrete_state_prob: Array
        self.discrete_transition_matrix: Array
        self.continuous_transition_matrix: Array
        self.process_cov: Array
        self.spike_params: SpikeObsParams

        # Placeholders for smoother results (computed in E-step)
        self.smoother_state_cond_mean: Array
        self.smoother_state_cond_cov: Array
        self.smoother_discrete_state_prob: Array
        self.smoother_joint_discrete_state_prob: Array
        self.smoother_pair_cond_cross_cov: Array
        self.smoother_pair_cond_means: Array

    def __repr__(self) -> str:
        params = [
            f"n_oscillators={self.n_oscillators}",
            f"n_neurons={self.n_neurons}",
            f"n_discrete_states={self.n_discrete_states}",
            f"sampling_freq={self.sampling_freq}",
            f"dt={self.dt}",
        ]

        update_flags = {
            "A": self.update_continuous_transition_matrix,
            "Q": self.update_process_cov,
            "Z": self.update_discrete_transition_matrix,
            "spike": self.update_spike_params,
            "m0": self.update_init_mean,
            "P0": self.update_init_cov,
        }

        flags_str = ", ".join(f"Update({k})={v}" for k, v in update_flags.items())

        return f"<{self.__class__.__name__}: {', '.join(params)}, [{flags_str}]>"

    # ------------------------------------------------------------------
    # Initialization methods
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
        from sklearn.mixture import GaussianMixture

        n_time = spikes.shape[0]
        n_states = self.n_discrete_states
        spikes_np = np_cpu.array(spikes)

        # Window size: ~50 timesteps (0.5s at 100Hz) for good balance
        # between temporal resolution and statistical stability.
        # Short enough to resolve oscillator frequencies (8-25Hz).
        window = min(50, n_time // (2 * n_states))
        window = max(window, 10)

        # Compute windowed features: per-neuron mean and variance
        n_windows = n_time // window
        if n_windows < n_states * 2:
            # Not enough windows — fall back to uniform
            self.smoother_discrete_state_prob = (
                jnp.ones((n_time, n_states)) / n_states
            )
            self.smoother_joint_discrete_state_prob = (
                jnp.ones((n_time - 1, n_states, n_states)) / n_states**2
            )
            return

        # Reshape into windows and compute features
        trimmed = spikes_np[: n_windows * window]
        windowed = trimmed.reshape(n_windows, window, -1)
        # Features: per-neuron mean rate, per-neuron variance,
        # and windowed spectral features (power at oscillator frequencies)
        means = windowed.mean(axis=1)      # (n_windows, n_neurons)
        variances = windowed.var(axis=1)   # (n_windows, n_neurons)

        features = np_cpu.concatenate([means, variances], axis=1)

        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            n_init=5,
            random_state=0,
        )
        gmm.fit(features)
        window_probs = gmm.predict_proba(features)  # (n_windows, n_states)

        # Expand window probabilities to per-timestep
        probs_np = np_cpu.repeat(window_probs, window, axis=0)
        # Handle remainder timesteps
        if n_time > n_windows * window:
            remainder = n_time - n_windows * window
            probs_np = np_cpu.concatenate(
                [probs_np, np_cpu.tile(window_probs[-1], (remainder, 1))]
            )
        probs = jnp.array(probs_np[:n_time])

        # Soften to avoid numerical issues (min prob 0.05)
        probs = probs * 0.9 + 0.05 / n_states
        probs = probs / probs.sum(axis=1, keepdims=True)

        self.smoother_discrete_state_prob = probs

        # Joint probabilities from adjacent timestep marginals
        joint = probs[:-1, :, None] * probs[1:, None, :]
        joint = joint / jnp.sum(joint, axis=(1, 2), keepdims=True)
        self.smoother_joint_discrete_state_prob = joint

    def _initialize_parameters(self, key: Array) -> None:
        """Initialize all model parameters."""
        k1, k2 = jax.random.split(key)
        self._initialize_discrete_state_prob()
        self._initialize_discrete_transition_matrix()
        self._initialize_continuous_state(k1)
        self._initialize_continuous_transition_matrix()
        self._initialize_process_covariance()
        self._initialize_spike_params(k2)
        self._validate_parameter_shapes()

    def _initialize_discrete_state_prob(self) -> None:
        """Initialize uniform discrete state probabilities."""
        self.init_discrete_state_prob = (
            jnp.ones(self.n_discrete_states) / self.n_discrete_states
        )

    def _initialize_discrete_transition_matrix(self) -> None:
        """Initialize discrete state transition matrix from diagonal."""
        self.discrete_transition_matrix = make_discrete_transition_matrix(
            self.discrete_transition_diag, self.n_discrete_states
        )

    def _initialize_continuous_state(self, key: Array) -> None:
        """Initialize continuous state mean (per-state random) and covariance (identity)."""
        keys = jax.random.split(key, self.n_discrete_states)
        means = jax.vmap(
            lambda k: jax.random.multivariate_normal(
                key=k, mean=jnp.zeros(self.n_latent), cov=jnp.eye(self.n_latent)
            )
        )(keys)
        self.init_mean = means.T  # (n_latent, n_discrete_states)
        self.init_cov = jnp.stack(
            [jnp.eye(self.n_latent)] * self.n_discrete_states, axis=2
        )

    @abstractmethod
    def _initialize_continuous_transition_matrix(self) -> None:
        """Initialize the continuous state transition matrix (A).

        Subclasses define whether A is constant across states or varies.
        """

    @abstractmethod
    def _initialize_process_covariance(self) -> None:
        """Initialize the process noise covariance (Q).

        Subclasses define whether Q is constant across states or varies.
        """

    def _initialize_spike_params(self, key: Array) -> None:
        """Initialize spike observation parameters (baseline and weights)."""
        if self.separate_spike_params:
            baseline = jnp.zeros((self.n_neurons, self.n_discrete_states))
            weights = (
                jax.random.normal(
                    key, (self.n_neurons, self.n_latent, self.n_discrete_states)
                )
                * 0.1
            )
        else:
            baseline = jnp.zeros(self.n_neurons)
            weights = jax.random.normal(key, (self.n_neurons, self.n_latent)) * 0.1

        self.spike_params = SpikeObsParams(baseline=baseline, weights=weights)

    def _validate_parameter_shapes(self) -> None:
        """Validate that all parameters have correct shapes."""
        if self.init_mean.shape != (self.n_latent, self.n_discrete_states):
            raise ValueError(
                f"init_mean shape mismatch: expected "
                f"({self.n_latent}, {self.n_discrete_states}), "
                f"got {self.init_mean.shape}."
            )
        if self.init_cov.shape != (
            self.n_latent,
            self.n_latent,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"init_cov shape mismatch: expected "
                f"({self.n_latent}, {self.n_latent}, {self.n_discrete_states}), "
                f"got {self.init_cov.shape}."
            )
        if self.init_discrete_state_prob.shape != (self.n_discrete_states,):
            raise ValueError(
                f"init_discrete_state_prob shape mismatch: expected "
                f"({self.n_discrete_states},), "
                f"got {self.init_discrete_state_prob.shape}."
            )
        if self.discrete_transition_matrix.shape != (
            self.n_discrete_states,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"discrete_transition_matrix shape mismatch: expected "
                f"({self.n_discrete_states}, {self.n_discrete_states}), "
                f"got {self.discrete_transition_matrix.shape}."
            )
        if self.continuous_transition_matrix.shape != (
            self.n_latent,
            self.n_latent,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"continuous_transition_matrix shape mismatch: expected "
                f"({self.n_latent}, {self.n_latent}, {self.n_discrete_states}), "
                f"got {self.continuous_transition_matrix.shape}."
            )
        if self.process_cov.shape != (
            self.n_latent,
            self.n_latent,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"process_cov shape mismatch: expected "
                f"({self.n_latent}, {self.n_latent}, {self.n_discrete_states}), "
                f"got {self.process_cov.shape}."
            )
        if self.separate_spike_params:
            expected_baseline = (self.n_neurons, self.n_discrete_states)
            expected_weights = (
                self.n_neurons,
                self.n_latent,
                self.n_discrete_states,
            )
        else:
            expected_baseline = (self.n_neurons,)
            expected_weights = (self.n_neurons, self.n_latent)

        if self.spike_params.baseline.shape != expected_baseline:
            raise ValueError(
                f"spike_params.baseline shape mismatch: expected "
                f"{expected_baseline}, got {self.spike_params.baseline.shape}."
            )
        if self.spike_params.weights.shape != expected_weights:
            raise ValueError(
                f"spike_params.weights shape mismatch: expected "
                f"{expected_weights}, got {self.spike_params.weights.shape}."
            )

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(self, spikes: Array) -> Array:
        """E-step: run filter and smoother, store posterior statistics.

        Parameters
        ----------
        spikes : Array, shape (n_time, n_neurons)
            Observed spike counts.

        Returns
        -------
        marginal_log_likelihood : Array, shape ()
        """

        def log_intensity_func(state: Array, params: SpikeObsParams) -> Array:
            return params.baseline + params.weights @ state

        (
            state_cond_filter_mean,
            state_cond_filter_cov,
            filter_discrete_state_prob,
            last_pair_cond_filter_mean,
            last_pair_cond_filter_cov,
            marginal_log_likelihood,
        ) = switching_point_process_filter(
            init_state_cond_mean=self.init_mean,
            init_state_cond_cov=self.init_cov,
            init_discrete_state_prob=self.init_discrete_state_prob,
            spikes=spikes,
            discrete_transition_matrix=self.discrete_transition_matrix,
            continuous_transition_matrix=self.continuous_transition_matrix,
            process_cov=self.process_cov,
            dt=self.dt,
            log_intensity_func=log_intensity_func,
            spike_params=self.spike_params,
            max_newton_iter=self.max_newton_iter,
            line_search_beta=self.line_search_beta,
        )

        smoother_args = dict(
            filter_mean=state_cond_filter_mean,
            filter_cov=state_cond_filter_cov,
            filter_discrete_state_prob=filter_discrete_state_prob,
            last_filter_conditional_cont_mean=last_pair_cond_filter_mean,
            process_cov=self.process_cov,
            continuous_transition_matrix=self.continuous_transition_matrix,
            discrete_state_transition_matrix=self.discrete_transition_matrix,
        )

        if self.smoother_type == "gpb2":
            (
                _,
                _,
                smoother_discrete_state_prob,
                smoother_joint_discrete_state_prob,
                _,
                state_cond_smoother_means,
                state_cond_smoother_covs,
                pair_cond_smoother_cross_covs,
                pair_cond_smoother_means,
                pair_cond_smoother_covs,
                next_pair_cond_smoother_means,
            ) = switching_kalman_smoother_gpb2(
                **smoother_args,
                last_filter_conditional_cont_cov=last_pair_cond_filter_cov,
            )
        else:
            (
                _,
                _,
                smoother_discrete_state_prob,
                smoother_joint_discrete_state_prob,
                _,
                state_cond_smoother_means,
                state_cond_smoother_covs,
                pair_cond_smoother_cross_covs,
                pair_cond_smoother_means,
            ) = switching_kalman_smoother(**smoother_args)
            pair_cond_smoother_covs = None
            next_pair_cond_smoother_means = None

        self.smoother_state_cond_mean = state_cond_smoother_means
        self.smoother_state_cond_cov = state_cond_smoother_covs
        self.smoother_discrete_state_prob = smoother_discrete_state_prob
        self.smoother_joint_discrete_state_prob = smoother_joint_discrete_state_prob
        self.smoother_pair_cond_cross_cov = pair_cond_smoother_cross_covs
        self.smoother_pair_cond_means = pair_cond_smoother_means
        self.smoother_pair_cond_covs = pair_cond_smoother_covs
        self.smoother_next_pair_cond_means = next_pair_cond_smoother_means

        return marginal_log_likelihood

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _m_step_dynamics(self) -> None:
        """M-step for dynamics parameters: A, Q, Z, initial state.

        Calls ``switching_kalman_maximization_step`` and applies Q regularization.
        The measurement_matrix and measurement_cov returns are ignored since
        they assume Gaussian observations.
        """
        n_time = self.smoother_state_cond_mean.shape[0]
        dummy_obs = jnp.zeros((n_time, 1))

        (
            new_A,
            _,  # measurement_matrix — ignored for point-process
            new_Q,
            _,  # measurement_cov — ignored for point-process
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
            pair_cond_smoother_covs=getattr(
                self, "smoother_pair_cond_covs", None
            ),
            next_pair_cond_smoother_means=getattr(
                self, "smoother_next_pair_cond_means", None
            ),
        )

        if self.update_continuous_transition_matrix:
            self.continuous_transition_matrix = new_A

        if self.update_process_cov:
            cfg = self.q_regularization
            if cfg.enabled:
                Q_blended = (
                    cfg.trust_region_weight * new_Q
                    + (1 - cfg.trust_region_weight) * self.process_cov
                )

                def clip_eigenvalues(Q: Array) -> Array:
                    Q = symmetrize(Q)
                    eigvals, eigvecs = jnp.linalg.eigh(Q)
                    if cfg.min_eigenvalue is not None:
                        eigvals = jnp.maximum(eigvals, cfg.min_eigenvalue)
                    if cfg.max_eigenvalue is not None:
                        eigvals = jnp.minimum(eigvals, cfg.max_eigenvalue)
                    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T

                Q_clipped = jax.vmap(clip_eigenvalues, in_axes=-1, out_axes=-1)(
                    Q_blended
                )
                self.process_cov = Q_clipped
            else:
                self.process_cov = new_Q

        if self.update_discrete_transition_matrix:
            self.discrete_transition_matrix = new_discrete_transition

        if self.update_init_mean:
            # Clip init_mean to prevent divergence. With sparse spike data,
            # the smoother at t=0 is poorly constrained and can produce
            # extreme estimates. The latent state is centered, so values
            # beyond a few standard deviations are unphysical.
            self.init_mean = jnp.clip(new_init_mean, -10.0, 10.0)

        if self.update_init_cov:
            # Regularize init_cov eigenvalues. The smoother at t=0 can have
            # enormous uncertainty with sparse observations, causing a
            # positive feedback loop: large init_cov → diffuse filter →
            # even larger smoother uncertainty → larger init_cov.
            def clip_init_cov_eigenvalues(P: Array) -> Array:
                P = symmetrize(P)
                eigvals, eigvecs = jnp.linalg.eigh(P)
                eigvals = jnp.clip(eigvals, 1e-4, 2.0)
                return eigvecs @ jnp.diag(eigvals) @ eigvecs.T

            new_init_cov = jax.vmap(
                clip_init_cov_eigenvalues, in_axes=-1, out_axes=-1
            )(new_init_cov)
            self.init_cov = new_init_cov

        self.init_discrete_state_prob = new_init_discrete_prob

    def _m_step_spikes(self, spikes: Array) -> None:
        """M-step for spike observation parameters: baseline and weights."""
        if not self.update_spike_params:
            return

        if self.separate_spike_params:
            MIN_STATE_WEIGHT = 1e-8

            if self.spike_baseline_prior_l2 > 0:
                mean_counts = jnp.mean(spikes, axis=0)
                baseline_prior = jnp.log(mean_counts / self.dt + 1e-10)
            else:
                baseline_prior = None

            new_baselines = []
            new_weights = []
            for j in range(self.n_discrete_states):
                state_weights = self.smoother_discrete_state_prob[:, j]
                total_weight = jnp.sum(state_weights)

                if total_weight < MIN_STATE_WEIGHT:
                    new_baselines.append(self.spike_params.baseline[:, j])
                    new_weights.append(self.spike_params.weights[:, :, j])
                    continue

                current_params = SpikeObsParams(
                    baseline=self.spike_params.baseline[:, j],
                    weights=self.spike_params.weights[:, :, j],
                )
                updated = update_spike_glm_params(
                    spikes=spikes,
                    smoother_mean=self.smoother_state_cond_mean[:, :, j],
                    current_params=current_params,
                    dt=self.dt,
                    time_weights=state_weights,
                    weight_l2=self.spike_weight_l2,
                    smoother_cov=self.smoother_state_cond_cov[:, :, :, j],
                    use_second_order=True,
                    baseline_prior=baseline_prior,
                    baseline_prior_l2=self.spike_baseline_prior_l2,
                )
                new_baselines.append(updated.baseline)
                new_weights.append(updated.weights)

            self.spike_params = SpikeObsParams(
                baseline=jnp.stack(new_baselines, axis=-1),
                weights=jnp.stack(new_weights, axis=-1),
            )
            return

        # Shared spike params: marginalize over discrete states
        smoother_mean = jnp.einsum(
            "tls,ts->tl",
            self.smoother_state_cond_mean,
            self.smoother_discrete_state_prob,
        )

        expected_cov = jnp.einsum(
            "tlks,ts->tlk",
            self.smoother_state_cond_cov,
            self.smoother_discrete_state_prob,
        )
        mean_deviation = (
            self.smoother_state_cond_mean - smoother_mean[:, :, None]
        )
        var_of_mean = jnp.einsum(
            "tls,tks,ts->tlk",
            mean_deviation,
            mean_deviation,
            self.smoother_discrete_state_prob,
        )
        smoother_cov = expected_cov + var_of_mean

        max_cov_eigenvalue = 1.0

        def clip_cov_eigenvalues(cov: Array) -> Array:
            cov = symmetrize(cov)
            eigvals, eigvecs = jnp.linalg.eigh(cov)
            eigvals_clipped = jnp.clip(eigvals, 0.0, max_cov_eigenvalue)
            return eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T

        smoother_cov_clipped = jax.vmap(clip_cov_eigenvalues)(smoother_cov)

        self.spike_params = update_spike_glm_params(
            spikes=spikes,
            smoother_mean=smoother_mean,
            current_params=self.spike_params,
            dt=self.dt,
            weight_l2=self.spike_weight_l2,
            smoother_cov=smoother_cov_clipped,
            use_second_order=True,
        )

    @abstractmethod
    def _project_parameters(self) -> None:
        """Project parameters onto valid spaces after M-step.

        Subclasses define model-specific projections (e.g., oscillatory
        structure for A, PSD for Q).
        """

    # ------------------------------------------------------------------
    # EM loop
    # ------------------------------------------------------------------

    def fit(
        self,
        spikes: Array,
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

        if key is None:
            key = jax.random.PRNGKey(0)

        if n_restarts > 1 and not skip_init:
            return self._fit_multi_restart(
                spikes, max_iter, tol, key, n_restarts
            )

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
            self.smoother_pair_cond_means = None
            self.smoother_pair_cond_covs = None
            self.smoother_next_pair_cond_means = None
            self._m_step_spikes(spikes)

        import copy

        log_likelihoods: list[float] = []
        best_ll = -float("inf")
        best_params: dict | None = None

        for iteration in range(max_iter):
            marginal_ll = self._e_step(spikes)
            log_likelihoods.append(float(marginal_ll))

            if not jnp.isfinite(marginal_ll):
                raise ValueError(
                    f"Non-finite log-likelihood at iteration {iteration}: "
                    f"{marginal_ll}. This may indicate numerical instability."
                )

            # Track best parameters seen (approximate EM can decrease LL)
            if log_likelihoods[-1] > best_ll:
                best_ll = log_likelihoods[-1]
                best_params = {
                    "init_mean": self.init_mean,
                    "init_cov": self.init_cov,
                    "init_discrete_state_prob": self.init_discrete_state_prob,
                    "discrete_transition_matrix": self.discrete_transition_matrix,
                    "continuous_transition_matrix": self.continuous_transition_matrix,
                    "process_cov": self.process_cov,
                    "spike_params": copy.deepcopy(self.spike_params),
                    "smoother_discrete_state_prob": self.smoother_discrete_state_prob,
                }

            if iteration > 0:
                is_converged, is_increasing = check_converged(
                    log_likelihood=log_likelihoods[-1],
                    previous_log_likelihood=log_likelihoods[-2],
                    tolerance=tol,
                )
                if not is_increasing:
                    logger.warning(
                        f"Log-likelihood decreased at iteration {iteration + 1}: "
                        f"{log_likelihoods[-2]:.4f} -> {log_likelihoods[-1]:.4f}"
                    )
                # Only declare convergence if LL is not decreasing
                if is_converged and is_increasing:
                    logger.info(f"Converged after {iteration + 1} iterations.")
                    break

            self._m_step_dynamics()
            self._m_step_spikes(spikes)
            self._project_parameters()

            logger.info(
                f"Iteration {iteration + 1}/{max_iter}\t"
                f"Log-Likelihood: {log_likelihoods[-1]:.4f}"
            )

        # Restore best parameters if LL decreased at any point
        if best_params is not None and log_likelihoods[-1] < best_ll:
            logger.info(
                f"Restoring best params from LL={best_ll:.4f} "
                f"(final was {log_likelihoods[-1]:.4f})"
            )
            for attr, value in best_params.items():
                setattr(self, attr, value)
            # Re-run E-step to populate all smoother outputs consistently
            # with the restored parameters
            self._e_step(spikes)

        if len(log_likelihoods) == max_iter:
            logger.warning("Reached maximum iterations without converging.")

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
        import copy

        best_lls: list[float] | None = None
        best_final_ll = -float("inf")
        best_state: dict | None = None

        keys = jax.random.split(key, n_restarts)

        for restart in range(n_restarts):
            try:
                lls = self._fit_single(spikes, max_iter, tol, keys[restart])
                final_ll = lls[-1] if lls else -float("inf")

                if final_ll > best_final_ll:
                    best_final_ll = final_ll
                    best_lls = lls
                    # Save model state
                    best_state = {
                        "init_mean": self.init_mean,
                        "init_cov": self.init_cov,
                        "init_discrete_state_prob": self.init_discrete_state_prob,
                        "discrete_transition_matrix": self.discrete_transition_matrix,
                        "continuous_transition_matrix": self.continuous_transition_matrix,
                        "process_cov": self.process_cov,
                        "spike_params": copy.deepcopy(self.spike_params),
                        "smoother_discrete_state_prob": self.smoother_discrete_state_prob,
                    }

                logger.info(
                    f"Restart {restart + 1}/{n_restarts}: "
                    f"final LL={final_ll:.4f}"
                )
            except ValueError:
                logger.warning(
                    f"Restart {restart + 1}/{n_restarts}: failed (non-finite LL)"
                )
                continue

        if best_state is None:
            raise ValueError(
                f"All {n_restarts} restarts failed with non-finite log-likelihood."
            )

        # Restore best model state
        for attr, value in best_state.items():
            setattr(self, attr, value)

        # Re-run E-step to populate all smoother outputs for the best params
        self._e_step(spikes)

        return best_lls

    # --- SGDFittableMixin protocol (shared by all switching PP subclasses) ---

    def fit_sgd(
        self,
        spikes: Array,
        key: Optional[Array] = None,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        spikes : Array, shape (n_time, n_neurons)
            Observed spike counts.
        key : Array or None
            JAX random key for initialization. Required on first call.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps.
        convergence_tol : float or None
            If set, stop early when |ΔLL| < tol for 5 consecutive steps.

        Returns
        -------
        log_likelihoods : list of float
        """
        spikes = jnp.asarray(spikes)
        self._sgd_n_time = spikes.shape[0]

        if not self._is_initialized():
            if key is None:
                raise ValueError("key required for initialization on first call")
            self._initialize_parameters(key)
            self._warm_initialize_states(spikes)

        return super().fit_sgd(
            spikes,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    def _is_initialized(self) -> bool:
        return (
            hasattr(self, "continuous_transition_matrix")
            and self.continuous_transition_matrix is not None
            and hasattr(self, "spike_params")
            and self.spike_params is not None
        )

    @property
    def _n_timesteps(self) -> int:
        return self._sgd_n_time

    def _check_sgd_initialized(self) -> None:
        if not self._is_initialized():
            raise RuntimeError(
                "Call fit_sgd(spikes, key=...) to initialize parameters."
            )

    def _store_sgd_params(self, params: dict) -> None:
        if "discrete_transition_matrix" in params:
            self.discrete_transition_matrix = params["discrete_transition_matrix"]
        if "init_mean" in params:
            self.init_mean = params["init_mean"]
        if "spike_baseline" in params:
            self.spike_params = SpikeObsParams(
                baseline=params["spike_baseline"],
                weights=params.get("spike_weights", self.spike_params.weights),
            )
        if "spike_weights" in params and "spike_baseline" not in params:
            self.spike_params = SpikeObsParams(
                baseline=self.spike_params.baseline,
                weights=params["spike_weights"],
            )
        # Per-state arrays reconstructed by subclasses

    def _finalize_sgd(self, spikes: Array) -> None:
        self._e_step(spikes)

    def _sgd_loss_fn(self, params: dict, spikes: Array) -> Array:
        """Compute negative marginal LL for SGD. Subclasses can override."""
        Z = params.get("discrete_transition_matrix", self.discrete_transition_matrix)
        m0 = params.get("init_mean", self.init_mean)
        A = params.get("_A", self.continuous_transition_matrix)
        Q = params.get("_Q", self.process_cov)

        baseline = params.get("spike_baseline", self.spike_params.baseline)
        weights = params.get("spike_weights", self.spike_params.weights)
        sp = SpikeObsParams(baseline=baseline, weights=weights)

        P0 = self._reconstruct_per_state_array(
            params, "init_cov", self.init_cov
        )

        def log_int(state, p):
            return p.baseline + p.weights @ state

        result = switching_point_process_filter(
            init_state_cond_mean=m0,
            init_state_cond_cov=P0,
            init_discrete_state_prob=self.init_discrete_state_prob,
            spikes=spikes,
            discrete_transition_matrix=Z,
            continuous_transition_matrix=A,
            process_cov=Q,
            dt=self.dt,
            log_intensity_func=log_int,
            spike_params=sp,
        )
        return -result[5]

    def _reconstruct_per_state_array(
        self, params: dict, prefix: str, fallback: Array
    ) -> Array:
        """Reconstruct a (…, n_discrete_states) array from per-state params."""
        if not any(k.startswith(f"{prefix}_") for k in params):
            return fallback
        return jnp.stack(
            [params.get(f"{prefix}_{j}", fallback[..., j])
             for j in range(self.n_discrete_states)],
            axis=-1,
        )


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
    freqs : Array, shape (n_oscillators,)
        Intrinsic oscillation frequencies in Hz.
    auto_regressive_coef : Array, shape (n_oscillators,)
        Damping coefficients for each oscillator (0 to 1).
    process_variance : Array, shape (n_oscillators,)
        Process noise variance for each oscillator.
    """

    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        n_discrete_states: int,
        sampling_freq: float,
        dt: float,
        freqs: jax.Array,
        auto_regressive_coef: jax.Array,
        process_variance: jax.Array,
        **kwargs,
    ):
        # Force COM-specific update flags
        kwargs["update_continuous_transition_matrix"] = False
        kwargs["update_process_cov"] = False
        kwargs.setdefault("separate_spike_params", True)
        super().__init__(
            n_oscillators, n_neurons, n_discrete_states, sampling_freq, dt, **kwargs
        )

        if freqs.shape != (n_oscillators,):
            raise ValueError(
                f"freqs shape {freqs.shape} != ({n_oscillators},)"
            )
        if auto_regressive_coef.shape != (n_oscillators,):
            raise ValueError(
                f"auto_regressive_coef shape {auto_regressive_coef.shape} "
                f"!= ({n_oscillators},)"
            )
        if process_variance.shape != (n_oscillators,):
            raise ValueError(
                f"process_variance shape {process_variance.shape} "
                f"!= ({n_oscillators},)"
            )

        self.freqs = freqs
        self.auto_regressive_coef = auto_regressive_coef
        self.process_variance = process_variance

    def _initialize_continuous_transition_matrix(self) -> None:
        """A is constant across states: uncoupled oscillators."""
        transition_matrix = construct_common_oscillator_transition_matrix(
            freqs=self.freqs,
            auto_regressive_coef=self.auto_regressive_coef,
            sampling_freq=self.sampling_freq,
        )
        self.continuous_transition_matrix = jnp.stack(
            [transition_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_process_covariance(self) -> None:
        """Q is constant across states: block-diagonal from process_variance."""
        process_cov = construct_common_oscillator_process_covariance(
            variance=self.process_variance,
        )
        self.process_cov = jnp.stack(
            [process_cov] * self.n_discrete_states, axis=2
        )

    def _warm_initialize_states(self, spikes: Array) -> None:
        """Warm-initialize using spectral features.

        COM-PP states differ in which oscillator modulates the neurons
        (e.g., theta vs beta), so windowed spectral band power ratios
        are better features than rate alone.
        """
        import numpy as np_cpu
        from sklearn.mixture import GaussianMixture

        n_time = spikes.shape[0]
        n_states = self.n_discrete_states
        spikes_np = np_cpu.array(spikes)

        window = min(50, n_time // (2 * n_states))
        window = max(window, 10)
        n_windows = n_time // window
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
                spectral_features.append(
                    power_norm[:, mask].sum(axis=1, keepdims=True)
                )
        if spectral_features:
            spectral = np_cpu.concatenate(spectral_features, axis=1)
            features = np_cpu.concatenate([means, spectral], axis=1)
        else:
            features = means

        gmm = GaussianMixture(
            n_components=n_states, covariance_type="full",
            n_init=5, random_state=0,
        )
        gmm.fit(features)
        window_probs = gmm.predict_proba(features)

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

    def _project_parameters(self) -> None:
        """No projection needed — A and Q are not updated."""
        pass

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            PSD_MATRIX,
            STOCHASTIC_ROW,
            UNCONSTRAINED,
        )

        params: dict = {}
        spec: dict = {}

        if self.update_spike_params:
            params["spike_baseline"] = self.spike_params.baseline
            spec["spike_baseline"] = UNCONSTRAINED
            params["spike_weights"] = self.spike_params.weights
            spec["spike_weights"] = UNCONSTRAINED

        if self.update_discrete_transition_matrix:
            params["discrete_transition_matrix"] = self.discrete_transition_matrix
            spec["discrete_transition_matrix"] = STOCHASTIC_ROW

        if self.update_init_mean:
            params["init_mean"] = self.init_mean
            spec["init_mean"] = UNCONSTRAINED

        if self.update_init_cov:
            for j in range(self.n_discrete_states):
                k = f"init_cov_{j}"
                params[k] = self.init_cov[..., j]
                spec[k] = PSD_MATRIX

        return params, spec

    def _store_sgd_params(self, params: dict) -> None:
        super()._store_sgd_params(params)
        self.init_cov = self._reconstruct_per_state_array(
            params, "init_cov", self.init_cov
        )


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
    freqs : Array, shape (n_oscillators,)
        Intrinsic oscillation frequencies in Hz.
    auto_regressive_coef : Array, shape (n_oscillators,)
        Damping coefficients for each oscillator.
    process_variance : Array, shape (n_oscillators, n_discrete_states)
        Process noise variance per oscillator per state.
    phase_difference : Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Phase differences for noise correlation.
    coupling_strength : Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Coupling strengths for noise correlation.
    """

    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        n_discrete_states: int,
        sampling_freq: float,
        dt: float,
        freqs: jax.Array,
        auto_regressive_coef: jax.Array,
        process_variance: jax.Array,
        phase_difference: jax.Array,
        coupling_strength: jax.Array,
        **kwargs,
    ):
        # Force CNM-specific update flags
        kwargs["update_continuous_transition_matrix"] = False
        kwargs["update_process_cov"] = True
        super().__init__(
            n_oscillators, n_neurons, n_discrete_states, sampling_freq, dt, **kwargs
        )

        if freqs.shape != (n_oscillators,):
            raise ValueError(
                f"freqs shape {freqs.shape} != ({n_oscillators},)"
            )
        if auto_regressive_coef.shape != (n_oscillators,):
            raise ValueError(
                f"auto_regressive_coef shape {auto_regressive_coef.shape} "
                f"!= ({n_oscillators},)"
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

        self.freqs = freqs
        self.auto_regressive_coef = auto_regressive_coef
        self.process_variance = process_variance
        self.phase_difference = phase_difference
        self.coupling_strength = coupling_strength

    def _initialize_continuous_transition_matrix(self) -> None:
        """A is constant across states: uncoupled oscillators."""
        transition_matrix = construct_common_oscillator_transition_matrix(
            freqs=self.freqs,
            auto_regressive_coef=self.auto_regressive_coef,
            sampling_freq=self.sampling_freq,
        )
        self.continuous_transition_matrix = jnp.stack(
            [transition_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_process_covariance(self) -> None:
        """Q varies across states: correlated noise structure."""
        self.process_cov = jnp.stack(
            [
                construct_correlated_noise_process_covariance(
                    variance=self.process_variance[:, state_ind],
                    phase_difference=self.phase_difference[:, :, state_ind],
                    coupling_strength=self.coupling_strength[:, :, state_ind],
                )
                for state_ind in range(self.n_discrete_states)
            ],
            axis=2,
        )

    def _project_parameters(self) -> None:
        """Project Q to preserve oscillatory block structure and ensure PSD."""
        if not self.update_process_cov:
            return

        projected_Q_list = []
        for j in range(self.n_discrete_states):
            Q_j = project_matrix_blockwise(self.process_cov[:, :, j])
            # Ensure PSD
            Q_j = symmetrize(Q_j)
            eigenvalues, eigenvectors = jnp.linalg.eigh(Q_j)
            eigenvalues_clipped = jnp.maximum(eigenvalues, 1e-8)
            Q_j_psd = eigenvectors @ jnp.diag(eigenvalues_clipped) @ eigenvectors.T
            Q_j_psd = symmetrize(Q_j_psd)
            projected_Q_list.append(Q_j_psd)

        self.process_cov = jnp.stack(projected_Q_list, axis=-1)


# ==========================================================================
# Directed Influence Model (DIM-PP)
# ==========================================================================


class DirectedInfluencePointProcessModel(BaseSwitchingPointProcessModel):
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
    freqs : Array, shape (n_oscillators,)
        Intrinsic oscillation frequencies in Hz.
    auto_regressive_coef : Array, shape (n_oscillators,)
        Damping coefficients for each oscillator.
    process_variance : Array, shape (n_oscillators,)
        Process noise variance (constant across states).
    phase_difference : Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial coupling phase differences.
    coupling_strength : Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial coupling strengths.
    use_reparameterized_mstep : bool, default=False
        If True, optimize oscillator parameters directly (guarantees valid
        oscillator structure). If False, use standard M-step with projection.
    """

    def __init__(
        self,
        n_oscillators: int,
        n_neurons: int,
        n_discrete_states: int,
        sampling_freq: float,
        dt: float,
        freqs: jax.Array,
        auto_regressive_coef: jax.Array,
        process_variance: jax.Array,
        phase_difference: jax.Array,
        coupling_strength: jax.Array,
        use_reparameterized_mstep: bool = False,
        **kwargs,
    ):
        # Force DIM-specific update flags
        kwargs["update_continuous_transition_matrix"] = True
        kwargs["update_process_cov"] = False
        super().__init__(
            n_oscillators, n_neurons, n_discrete_states, sampling_freq, dt, **kwargs
        )

        if freqs.shape != (n_oscillators,):
            raise ValueError(
                f"freqs shape {freqs.shape} != ({n_oscillators},)"
            )
        if auto_regressive_coef.shape != (n_oscillators,):
            raise ValueError(
                f"auto_regressive_coef shape {auto_regressive_coef.shape} "
                f"!= ({n_oscillators},)"
            )
        if process_variance.shape != (n_oscillators,):
            raise ValueError(
                f"process_variance shape {process_variance.shape} "
                f"!= ({n_oscillators},)"
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

        self.freqs = freqs
        self.auto_regressive_coef = auto_regressive_coef
        self.process_variance = process_variance
        self.phase_difference = phase_difference
        self.coupling_strength = coupling_strength
        self.use_reparameterized_mstep = use_reparameterized_mstep
        self._current_osc_params: Optional[list[dict]] = None

    def _initialize_continuous_transition_matrix(self) -> None:
        """A varies across states: directed influence coupling structure."""
        self.continuous_transition_matrix = jnp.stack(
            [
                construct_directed_influence_transition_matrix(
                    freqs=self.freqs,
                    damping_coeffs=self.auto_regressive_coef,
                    coupling_strengths=self.coupling_strength[:, :, state_ind],
                    phase_diffs=self.phase_difference[:, :, state_ind],
                    sampling_freq=self.sampling_freq,
                )
                for state_ind in range(self.n_discrete_states)
            ],
            axis=2,
        )

    def _initialize_process_covariance(self) -> None:
        """Q is constant across states: block-diagonal from process_variance."""
        process_cov = construct_common_oscillator_process_covariance(
            variance=self.process_variance,
        )
        self.process_cov = jnp.stack(
            [process_cov] * self.n_discrete_states, axis=2
        )

    def _m_step_dynamics(self) -> None:
        """M-step with optional reparameterized transition update.

        Caches the unconstrained A and sufficient statistics (gamma1, beta)
        so that _project_parameters can check whether projection worsens
        the Q-function.
        """
        if self.use_reparameterized_mstep:
            self._m_step_reparameterized()
            return

        super()._m_step_dynamics()

        # Cache sufficient statistics for Q-function-aware projection
        self._transition_suff_stats = compute_transition_sufficient_stats(
            state_cond_smoother_means=self.smoother_state_cond_mean,
            state_cond_smoother_covs=self.smoother_state_cond_cov,
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
            pair_cond_smoother_means=self.smoother_pair_cond_means,
        )

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
            pair_cond_smoother_covs=getattr(
                self, "smoother_pair_cond_covs", None
            ),
            next_pair_cond_smoother_means=getattr(
                self, "smoother_next_pair_cond_means", None
            ),
        )

        if self.update_discrete_transition_matrix:
            self.discrete_transition_matrix = new_discrete_transition
        if self.update_init_mean:
            self.init_mean = new_init_mean
        if self.update_init_cov:
            self.init_cov = new_init_cov
        self.init_discrete_state_prob = new_init_discrete_prob

        # Reparameterized optimization for A
        gamma1, beta = compute_transition_sufficient_stats(
            state_cond_smoother_means=self.smoother_state_cond_mean,
            state_cond_smoother_covs=self.smoother_state_cond_cov,
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
            pair_cond_smoother_means=self.smoother_pair_cond_means,
        )

        if self._current_osc_params is None:
            self._current_osc_params = []
            for j in range(self.n_discrete_states):
                params = extract_dim_params_from_matrix(
                    self.continuous_transition_matrix[:, :, j],
                    self.sampling_freq,
                    self.n_oscillators,
                )
                self._current_osc_params.append(params)

        A_list = []
        for j in range(self.n_discrete_states):
            opt_params = optimize_dim_transition_params(
                gamma1=gamma1[:, :, j],
                beta=beta[:, :, j],
                init_params=self._current_osc_params[j],
                sampling_freq=self.sampling_freq,
            )
            self._current_osc_params[j] = opt_params

            A_j = construct_directed_influence_transition_matrix(
                freqs=opt_params["freq"],
                damping_coeffs=opt_params["damping"],
                coupling_strengths=opt_params["coupling_strength"],
                phase_diffs=opt_params["phase_diff"],
                sampling_freq=self.sampling_freq,
            )
            A_list.append(A_j)

        self.continuous_transition_matrix = jnp.stack(A_list, axis=-1)
        self._update_public_oscillator_params()

    def _update_public_oscillator_params(self) -> None:
        """Sync optimized oscillator params to public attributes."""
        if self._current_osc_params is None:
            return

        freqs_list = [p["freq"] for p in self._current_osc_params]
        damping_list = [p["damping"] for p in self._current_osc_params]

        self.freqs = jnp.mean(jnp.stack(freqs_list, axis=-1), axis=-1)
        self.auto_regressive_coef = jnp.mean(
            jnp.stack(damping_list, axis=-1), axis=-1
        )

        coupling_list = [p["coupling_strength"] for p in self._current_osc_params]
        phase_list = [p["phase_diff"] for p in self._current_osc_params]

        self.coupling_strength = jnp.stack(coupling_list, axis=-1)
        self.phase_difference = jnp.stack(phase_list, axis=-1)

    def _project_parameters(self) -> None:
        """Project A to oscillatory block structure, preserving EM monotonicity.

        The unconstrained M-step maximizes the Q-function. Block projection
        can worsen the Q-function, breaking EM monotonicity. We only accept
        the projected A if it does not increase the Q-function loss. If
        projection worsens the objective, we keep the unconstrained A.

        After projection (or not), we enforce spectral radius < 1 for
        stability, again only accepting the rescaled A if it doesn't
        worsen the Q-function.
        """
        if self.use_reparameterized_mstep:
            return  # Already valid by construction

        if not self.update_continuous_transition_matrix:
            return

        suff_stats = getattr(self, "_transition_suff_stats", None)

        projected = []
        for j in range(self.n_discrete_states):
            A_unc_j = self.continuous_transition_matrix[:, :, j]
            A_proj_j = project_coupled_transition_matrix(A_unc_j)

            # Only accept projection if it doesn't worsen Q-function
            if suff_stats is not None:
                gamma1_j = suff_stats[0][:, :, j]
                beta_j = suff_stats[1][:, :, j]
                q_unc = compute_transition_q_function(A_unc_j, gamma1_j, beta_j)
                q_proj = compute_transition_q_function(A_proj_j, gamma1_j, beta_j)
                # compute_transition_q_function returns negative Q (to minimize)
                # so q_proj > q_unc means projection worsened the objective
                A_j = jnp.where(q_proj <= q_unc, A_proj_j, A_unc_j)
            else:
                A_j = A_proj_j

            # Enforce spectral radius < 1 for stability (unconditional).
            # Stability is a hard physical constraint: an unstable A causes
            # state divergence and invalidates the E-step posteriors. Unlike
            # the block structure projection above, this is not optional.
            max_spectral_radius = 0.99
            eigvals = jnp.linalg.eigvals(A_j)
            spectral_radius = jnp.max(jnp.abs(eigvals))
            scale = jnp.where(
                spectral_radius > max_spectral_radius,
                max_spectral_radius / spectral_radius,
                1.0,
            )
            A_j = A_j * scale

            projected.append(A_j)
        self.continuous_transition_matrix = jnp.stack(projected, axis=-1)

    # --- SGDFittableMixin: DIM-PP specific ---

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            PSD_MATRIX,
            STOCHASTIC_ROW,
            UNCONSTRAINED,
        )

        params: dict = {}
        spec: dict = {}

        if self.update_continuous_transition_matrix:
            params["phase_difference"] = self.phase_difference
            spec["phase_difference"] = UNCONSTRAINED
            params["coupling_strength"] = self.coupling_strength
            spec["coupling_strength"] = UNCONSTRAINED

        if self.update_spike_params:
            params["spike_baseline"] = self.spike_params.baseline
            spec["spike_baseline"] = UNCONSTRAINED
            params["spike_weights"] = self.spike_params.weights
            spec["spike_weights"] = UNCONSTRAINED

        if self.update_discrete_transition_matrix:
            params["discrete_transition_matrix"] = self.discrete_transition_matrix
            spec["discrete_transition_matrix"] = STOCHASTIC_ROW

        if self.update_init_mean:
            params["init_mean"] = self.init_mean
            spec["init_mean"] = UNCONSTRAINED

        if self.update_init_cov:
            for j in range(self.n_discrete_states):
                k = f"init_cov_{j}"
                params[k] = self.init_cov[..., j]
                spec[k] = PSD_MATRIX

        return params, spec

    def fit_sgd(
        self,
        spikes,
        key=None,
        optimizer=None,
        num_steps=200,
        verbose=False,
        convergence_tol=None,
        connectivity_penalty=None,
    ):
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        spikes : Array, shape (n_time, n_neurons)
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
            spikes, key=key, optimizer=optimizer, num_steps=num_steps,
            verbose=verbose, convergence_tol=convergence_tol,
        )

    def _sgd_loss_fn(self, params: dict, spikes: jax.Array) -> jax.Array:
        phase_diff = params.get("phase_difference", self.phase_difference)
        coupling = params.get("coupling_strength", self.coupling_strength)

        A_list = []
        for j in range(self.n_discrete_states):
            A_j = construct_directed_influence_transition_matrix(
                freqs=self.freqs,
                damping_coeffs=self.auto_regressive_coef,
                phase_diffs=phase_diff[..., j],
                coupling_strengths=coupling[..., j],
                sampling_freq=self.sampling_freq,
            )
            A_list.append(A_j)

        # Inject reconstructed A into the base loss function via params
        params_with_A = dict(params)
        params_with_A["_A"] = jnp.stack(A_list, axis=-1)
        base_loss = super()._sgd_loss_fn(params_with_A, spikes)

        # Add connectivity penalty if configured
        penalty_config = getattr(self, "_connectivity_penalty", None)
        if penalty_config is not None:
            from state_space_practice.oscillator_regularization import (
                total_connectivity_penalty,
            )
            coupling_transposed = jnp.moveaxis(coupling, -1, 0)
            base_loss = base_loss + total_connectivity_penalty(
                coupling_transposed, penalty_config,
                n_timesteps=self._n_timesteps,
            )

        return base_loss

    def _store_sgd_params(self, params: dict) -> None:
        super()._store_sgd_params(params)
        if "phase_difference" in params:
            self.phase_difference = params["phase_difference"]
        if "coupling_strength" in params:
            self.coupling_strength = params["coupling_strength"]
        if "phase_difference" in params or "coupling_strength" in params:
            A_list = []
            for j in range(self.n_discrete_states):
                A_j = construct_directed_influence_transition_matrix(
                    freqs=self.freqs,
                    damping_coeffs=self.auto_regressive_coef,
                    phase_diffs=self.phase_difference[..., j],
                    coupling_strengths=self.coupling_strength[..., j],
                    sampling_freq=self.sampling_freq,
                )
                A_list.append(A_j)
            self.continuous_transition_matrix = jnp.stack(A_list, axis=-1)
        self.init_cov = self._reconstruct_per_state_array(
            params, "init_cov", self.init_cov
        )
