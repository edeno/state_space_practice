r"""Implementation of the switching oscillator models

References
----------
1. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2022). Switching Functional
   Network Models of Oscillatory Brain Dynamics. In 2022 56th Asilomar
   Conference on Signals, Systems, and Computers (IEEE), pp. 607-612.
   https://doi.org/10.1109/IEEECONF56349.2022.10052077.
2. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2024). Switching Models of
   Oscillatory Networks Greatly Improve Inference of Dynamic Functional
   Connectivity. https://doi.org/10.48550/arXiv.2404.18854
3. https://github.com/Stephen-Lab-BU/Switching_Oscillator_Networks

Variables
----------
- `x`: continuous latent state, shape (n_time, 2 * n_oscillators, n_discrete_states)
- `S`: discrete latent state, shape (n_time, n_discrete_states)
- `B`: measurement/observation matrix, shape (n_sources, 2 * n_oscillators, n_discrete_states)
- `A`: continuous transition matrix, shape (2 * n_oscillators, 2 * n_oscillators, n_discrete_states)
- `Z`: discrete transition matrix, shape (n_discrete_states, n_discrete_states)
- `\Sigma`: process noise covariance matrix, shape (2 * n_oscillators, 2 * n_oscillators, n_discrete_states)
- `R`: measurement noise covariance matrix, shape (n_sources, n_sources)
- `y`: observations, shape (n_time, n_sources)

Models
------
- Common Oscillator Model (COM)
    - measurement matrix depends on discrete latent state
- Correlated Noise Model (CNM)
    - process noise covariance depends on discrete latent state
- Directed Influence Model (DIM)
    - continuous transition matrix depends on discrete latent state

"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.oscillator_utils import (
    canonicalize_correlated_noise_pair_parameters,
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
    construct_correlated_noise_measurement_matrix,
    construct_correlated_noise_process_covariance,
    compute_directed_influence_stability_scale,
    construct_directed_influence_measurement_matrix,
    construct_directed_influence_transition_matrix,
    extract_correlated_noise_params_from_covariance,
    extract_dim_params_from_matrix,
    get_block_slice,
    project_correlated_noise_process_covariance,
    project_coupled_transition_matrix,
)
from state_space_practice.switching_kalman import (
    compute_transition_sufficient_stats,
    optimize_dim_transition_params_joint,
    switching_kalman_filter,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
    switching_kalman_smoother_gpb2,
)
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import (
    check_converged,
    make_discrete_transition_matrix,
    shift_to_psd,
    stabilize_transition_matrix,
    validate_covariance,
)

logger = logging.getLogger(__name__)


def _validate_finite_array(name: str, value: ArrayLike) -> None:
    """Validate finite model parameters at public boundaries."""
    arr = jnp.asarray(value)
    if bool(jnp.any(~jnp.isfinite(arr))):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_nonnegative_array(name: str, value: ArrayLike) -> None:
    """Validate finite, non-negative model parameters at public boundaries."""
    arr = jnp.asarray(value)
    _validate_finite_array(name, arr)
    if bool(jnp.any(arr < 0)):
        raise ValueError(f"{name} must be non-negative.")


def _validate_unit_interval_array(name: str, value: ArrayLike) -> None:
    """Validate finite parameters constrained to the closed unit interval."""
    arr = jnp.asarray(value)
    _validate_finite_array(name, arr)
    if bool(jnp.any((arr < 0) | (arr > 1))):
        raise ValueError(f"{name} entries must lie in [0, 1].")


def _validate_positive_scalar(name: str, value: ArrayLike) -> None:
    """Validate a finite, strictly positive real scalar.

    Accepts Python, NumPy, and 0-d JAX scalars (not just ``float``/``int``), so
    a ``np.float32`` or 0-d array passes rather than being rejected on type.
    """
    arr = jnp.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be a scalar. Got shape {arr.shape}.")
    try:
        numeric = float(arr)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real scalar. Got {value!r}.") from None
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be positive and finite. Got {value}.")


def _dim_stability_scale(
    freqs: ArrayLike,
    damping_coef: ArrayLike,
    coupling_strength: ArrayLike,
    sampling_freq: float,
    max_spectral_radius: float = 0.99,
) -> Array:
    """Compatibility wrapper for the shared differentiable DIM stability bound."""
    return compute_directed_influence_stability_scale(
        freqs,
        damping_coef,
        coupling_strength,
        sampling_freq,
        max_spectral_radius,
    )


# Update flags each oscillator subclass fixes by model definition (which of the
# continuous transition matrix A, measurement matrix H, and process covariance Q
# are learned). COM, CNM, and DIM differ only in which single one is state-
# dependent, so all three hard-set exactly these three flags.
_FORCED_UPDATE_FLAGS = (
    "update_continuous_transition_matrix",
    "update_measurement_matrix",
    "update_process_cov",
)


def _reject_forced_update_flags(model_name: str, kwargs: dict) -> None:
    """Reject update flags a subclass fixes by model definition.

    Each subclass hard-sets which of A, H, and Q are learned. Silently
    accepting and then discarding a user-supplied override (e.g.
    ``update_process_cov=True`` on a model that fixes Q) is worse than
    refusing it, so raise instead.
    """
    conflicting = sorted(k for k in _FORCED_UPDATE_FLAGS if k in kwargs)
    if conflicting:
        raise ValueError(
            f"{model_name} fixes {', '.join(conflicting)} by model definition; "
            "these update flags cannot be overridden."
        )


def _stabilize_transition_matrix(
    A: ArrayLike, max_spectral_radius: float = 0.99
) -> Array:
    """Scale a transition matrix so its spectral radius is <= the bound.

    Thin wrapper over :func:`state_space_practice.utils.stabilize_transition_matrix`;
    kept for the eager per-state stability clamp in oscillator model M-steps.
    """
    return stabilize_transition_matrix(A, max_spectral_radius=max_spectral_radius)


class BaseModel(ABC, SGDFittableMixin):
    """Abstract base class for switching oscillator models.

    This class provides the core structure for Expectation-Maximization (EM)
    based fitting of switching linear Gaussian state-space models,
    specifically tailored for oscillator dynamics. Subclasses must implement
    methods to initialize model-specific parameters and project them onto
    valid spaces after M-steps.

    Parameters
    ----------
    n_oscillators : int
        Number of latent oscillators.
    n_discrete_states : int
        Number of discrete latent states (network configurations).
    n_sources : int
        Number of observed sources or channels.
    sampling_freq : float
        Sampling frequency of the observations.
    discrete_transition_diag : Optional[jax.Array], default=None
        Diagonal elements of the discrete transition matrix (Z).
        If None, initializes Z with a default based on `n_discrete_states`.
    stickiness : float, default=0.0
        Self-transition bias of a symmetric sticky Dirichlet prior on the
        discrete transition matrix. If > 0, this prior regularizes the M-step
        update of Z toward staying in the current state (concentration is fixed
        at 1.0); 0.0 disables the prior entirely.
    update_discrete_transition_matrix : bool, default=True
        Flag to update Z during M-step.
    update_continuous_transition_matrix : bool, default=True
        Flag to update A during M-step.
    update_measurement_matrix : bool, default=True
        Flag to update H during M-step.
    update_process_cov : bool, default=True
        Flag to update Sigma (Q) during M-step.
    update_measurement_cov : bool, default=True
        Flag to update R during M-step.
    update_init_mean : bool, default=True
        Flag to update initial mean during M-step.
    update_init_cov : bool, default=True
        Flag to update initial covariance during M-step.
    smoother_type : {"gpb1", "gpb2"}, default="gpb1"
        Approximate switching smoother. GPB2 retains pair-conditioned
        continuous-state statistics and is more accurate but more expensive.

    Attributes
    ----------
    init_mean : jax.Array
        Initial mean (n_cont_states, n_discrete_states).
    init_cov : jax.Array
        Initial covariances (n_cont_states, n_cont_states, n_discrete_states).
    init_discrete_state_prob : jax.Array
        Initial discrete probabilities (n_discrete_states,).
    discrete_transition_matrix : jax.Array
        Discrete transition matrix (n_discrete_states, n_discrete_states).
    continuous_transition_matrix : jax.Array
        Continuous transition matrix (n_cont_states, n_cont_states, n_discrete_states).
    process_cov : jax.Array
        Process noise (n_cont_states, n_cont_states, n_discrete_states).
    measurement_matrix : jax.Array
        Observation matrix (n_obs_dim, n_cont_states, n_discrete_states).
    measurement_cov : jax.Array
        Observation noise (n_obs_dim, n_obs_dim, n_discrete_states).

    """

    _EM_SNAPSHOT_KEYS = (
        "smoother_state_cond_mean",
        "smoother_state_cond_cov",
        "smoother_discrete_state_prob",
        "smoother_joint_discrete_state_prob",
        "smoother_pair_cond_cross_cov",
        "smoother_pair_cond_means",
        "smoother_pair_cond_covs",
        "smoother_next_pair_cond_means",
        "continuous_transition_matrix",
        "process_cov",
        "measurement_matrix",
        "measurement_cov",
        "init_mean",
        "init_cov",
        "init_discrete_state_prob",
        "discrete_transition_matrix",
        "freqs",
        "damping_coef",
        "process_variance",
        "phase_difference",
        "coupling_strength",
        "_current_osc_params",
    )

    def __init__(
        self,
        n_oscillators: int,
        n_discrete_states: int,
        n_sources: int,
        sampling_freq: float,
        discrete_transition_diag: Optional[jax.Array] = None,
        stickiness: float = 0.0,
        update_discrete_transition_matrix: bool = True,
        update_continuous_transition_matrix: bool = True,
        update_measurement_matrix: bool = True,
        update_process_cov: bool = True,
        update_measurement_cov: bool = True,
        update_init_mean: bool = True,
        update_init_cov: bool = True,
        smoother_type: str = "gpb1",
    ):
        _validate_positive_scalar("sampling_freq", sampling_freq)
        if smoother_type not in ("gpb1", "gpb2"):
            raise ValueError(
                f"smoother_type must be 'gpb1' or 'gpb2', got {smoother_type!r}."
            )
        self.n_oscillators = n_oscillators
        self.n_discrete_states = n_discrete_states
        self.n_sources = n_sources
        self.sampling_freq = sampling_freq
        self.smoother_type = smoother_type
        self.n_cont_states = 2 * n_oscillators

        # Set default diagonal if not provided.
        # Default: ~1s expected dwell time, computed from sampling_freq.
        # p_stay = 1 - 1/(dwell_seconds * sampling_freq)
        if discrete_transition_diag is None:
            expected_dwell_sec = 1.0
            if sampling_freq < 1.0 / expected_dwell_sec:
                raise ValueError(
                    f"sampling_freq={sampling_freq} Hz is too low for "
                    f"expected_dwell_sec={expected_dwell_sec}. "
                    f"Require sampling_freq >= {1.0 / expected_dwell_sec} Hz."
                )
            p_stay = 1.0 - 1.0 / (expected_dwell_sec * sampling_freq)
            # Cap just below 1 to avoid a numerically absorbing state at very
            # high sampling rates. The formula already keeps p_stay in [0, 1)
            # for valid sampling_freq, so this preserves the ~1s dwell target
            # (e.g. p_stay ~= 0.999 at 1000 Hz) instead of flattening every
            # rate above 20 Hz to 0.95 (a 20-sample = 20 ms dwell at 1000 Hz).
            p_stay = min(p_stay, 1.0 - 1e-6)
            self.discrete_transition_diag = jnp.full((n_discrete_states,), p_stay)
        else:
            diag = jnp.asarray(discrete_transition_diag)
            if diag.shape != (n_discrete_states,):
                raise ValueError(
                    "discrete_transition_diag must have shape "
                    f"({n_discrete_states},), got {diag.shape}."
                )
            if bool(jnp.any(~jnp.isfinite(diag))):
                raise ValueError(
                    "discrete_transition_diag must contain only finite values."
                )
            if bool(jnp.any((diag < 0) | (diag > 1))):
                raise ValueError("discrete_transition_diag entries must lie in [0, 1].")
            self.discrete_transition_diag = diag

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

        # Store update flags
        self.update_discrete_transition_matrix = update_discrete_transition_matrix
        self.update_continuous_transition_matrix = update_continuous_transition_matrix
        self.update_measurement_matrix = update_measurement_matrix
        self.update_process_cov = update_process_cov
        self.update_measurement_cov = update_measurement_cov
        self.update_init_mean = update_init_mean
        self.update_init_cov = update_init_cov

        # Placeholder for parameters - to be initialized by subclasses
        self.init_mean: jax.Array
        self.init_cov: jax.Array
        self.init_discrete_state_prob: jax.Array
        self.discrete_transition_matrix: jax.Array
        self.continuous_transition_matrix: jax.Array
        self.process_cov: jax.Array
        self.measurement_matrix: jax.Array
        self.measurement_cov: jax.Array

        # Placeholders for smoother results (Expected Sufficient Statistics - ESS).
        # Optional because a failed E-step reset (see _clear_smoother_state)
        # sets them to None so decode()/predict_proba() guards fire.
        self.smoother_state_cond_mean: Optional[jax.Array]
        self.smoother_state_cond_cov: Optional[jax.Array]
        self.smoother_discrete_state_prob: Optional[jax.Array]
        self.smoother_joint_discrete_state_prob: Optional[jax.Array]
        self.smoother_pair_cond_cross_cov: Optional[jax.Array]
        self.smoother_pair_cond_means: Optional[jax.Array]  # E[x|S_t=i,S_{t+1}=j]
        self.smoother_pair_cond_covs: Optional[jax.Array]
        self.smoother_next_pair_cond_means: Optional[jax.Array]

    def _snapshot_em_state(self) -> dict:
        """Capture parameters and smoother outputs for EM rollback.

        Snapshotted values are JAX arrays (immutable), so a reference copy is
        sufficient -- except ``_current_osc_params``, a dict that the
        reparameterized M-step mutates in place, which needs a deep copy.
        """
        import copy

        snapshot: dict = {}
        for key in self._EM_SNAPSHOT_KEYS:
            if not hasattr(self, key):
                continue
            value = getattr(self, key)
            snapshot[key] = (
                copy.deepcopy(value) if key == "_current_osc_params" else value
            )
        return snapshot

    def _restore_em_state(self, state: dict) -> None:
        """Restore a state captured by _snapshot_em_state."""
        for key, value in state.items():
            setattr(self, key, value)

    def _clear_smoother_state(self) -> None:
        """Drop smoother posteriors so decode()/predict_proba() fail loudly.

        Used when EM cannot produce a usable posterior (e.g. a non-finite
        first E-step with no earlier accepted state to roll back to). Without
        this the NaN posteriors installed by the failed E-step would remain,
        and decode()/predict_proba() -- whose guards catch a missing/None
        attribute but not a NaN-filled array -- would silently return garbage
        (argmax of NaN). Setting the attributes to None makes those guards fire.
        """
        self.smoother_discrete_state_prob = None
        self.smoother_joint_discrete_state_prob = None
        self.smoother_state_cond_mean = None
        self.smoother_state_cond_cov = None
        self.smoother_pair_cond_cross_cov = None
        self.smoother_pair_cond_means = None
        self.smoother_pair_cond_covs = None
        self.smoother_next_pair_cond_means = None

    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the model.

        Returns
        -------
        str
            A string showing the model class name and its core parameters.
        """
        # Collect the core parameters
        params = [
            f"n_oscillators={self.n_oscillators}",
            f"n_discrete_states={self.n_discrete_states}",
            f"n_sources={self.n_sources}",
            f"sampling_freq={self.sampling_freq}",
            f"smoother_type={self.smoother_type!r}",
        ]

        # Add update flags for clarity
        update_flags = {
            "discrete_transition": self.update_discrete_transition_matrix,
            "transition": self.update_continuous_transition_matrix,
            "measurement": self.update_measurement_matrix,
            "process_cov": self.update_process_cov,
            "measurement_cov": self.update_measurement_cov,
            "init_mean": self.update_init_mean,
            "init_cov": self.update_init_cov,
        }

        flags_str = ", ".join(f"Update({k})={v}" for k, v in update_flags.items())

        return f"<{self.__class__.__name__}: {', '.join(params)}, [{flags_str}]>"

    def decode(self) -> jax.Array:
        """Return the most likely discrete state at each time step.

        Returns
        -------
        states : jax.Array, shape (n_time,)
            Argmax of smoother discrete state probabilities.

        Raises
        ------
        RuntimeError
            If called before fit() or fit_sgd().
        """
        if (
            not hasattr(self, "smoother_discrete_state_prob")
            or self.smoother_discrete_state_prob is None
        ):
            raise RuntimeError(
                "No smoother posteriors available. Call fit() or fit_sgd() and "
                "ensure it produced a finite log-likelihood before decode()."
            )
        return jnp.argmax(self.smoother_discrete_state_prob, axis=1)

    def predict_proba(self) -> jax.Array:
        """Return smoothed discrete state probabilities.

        Returns
        -------
        probs : jax.Array, shape (n_time, n_discrete_states)
            Posterior probability of each discrete state at each time step.

        Raises
        ------
        RuntimeError
            If called before fit() or fit_sgd().
        """
        if (
            not hasattr(self, "smoother_discrete_state_prob")
            or self.smoother_discrete_state_prob is None
        ):
            raise RuntimeError(
                "No smoother posteriors available. Call fit() or fit_sgd() and "
                "ensure it produced a finite log-likelihood before "
                "predict_proba()."
            )
        return self.smoother_discrete_state_prob

    def _warm_initialize_states(self, observations: ArrayLike) -> None:
        """Warm-initialize discrete state priors and per-state init_mean.

        Uses windowed cross-covariance features clustered with a Gaussian
        mixture model to break symmetry before the first E-step.  Sets
        ``init_discrete_state_prob`` from GMM mixing weights and scales
        ``init_cov`` to match data variance.  For models with a single shared
        measurement matrix it also sets per-state ``init_mean`` via the H
        pseudo-inverse; models with a state-dependent H (e.g. COM) keep their
        cold ``init_mean`` (see ``_apply_warm_init``).

        Subclasses may override to use model-specific features.  The shared
        post-GMM logic lives in ``_apply_warm_init``.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
        """
        import numpy as np_cpu

        obs_np = np_cpu.array(observations)
        n_sources = obs_np.shape[1]
        n_states = self.n_discrete_states

        if n_states <= 1:
            return

        window, n_windows, windowed = self._prepare_windows(obs_np, n_states)
        if windowed is None:
            return

        # Windowed cross-covariance features (upper triangle)
        triu_idx = np_cpu.triu_indices(n_sources)
        features = np_cpu.stack(
            [np_cpu.cov(windowed[w].T)[triu_idx] for w in range(n_windows)]
        )

        self._apply_warm_init(features, obs_np, windowed, n_states)

    @staticmethod
    def _prepare_windows(obs_np, n_states):
        """Compute windowed observations for warm init.

        Returns (window, n_windows, windowed) or (None, None, None)
        if there is not enough data.
        """

        n_time = obs_np.shape[0]
        window = min(50, n_time // (2 * n_states))
        window = max(window, 10)
        n_windows = n_time // window

        if n_windows < n_states * 2:
            logger.debug(
                "Warm init skipped: only %d windows for %d states.",
                n_windows,
                n_states,
            )
            return None, None, None

        trimmed = obs_np[: n_windows * window]
        windowed = trimmed.reshape(n_windows, window, -1)
        return window, n_windows, windowed

    def _apply_warm_init(self, features, obs_np, windowed, n_states) -> None:
        """Shared GMM clustering and parameter setting for warm init.

        Called by ``_warm_initialize_states`` (and subclass overrides)
        after model-specific features have been constructed.

        Requires ``self.measurement_matrix`` to be initialized. The per-state
        ``init_mean`` step assumes a single shared H (as in CNM and DIM); when
        H is state-dependent (COM, whose H is small and random at init),
        ``pinv(H[..., 0])`` is arbitrary and would amplify the cluster means
        through the inverse of noise, so that step is skipped and ``init_mean``
        is left at its cold value.
        """
        import numpy as np_cpu
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            n_init=5,
            random_state=0,
        )
        gmm.fit(features)
        labels = gmm.predict(features)

        # Per-state init_mean via H pseudo-inverse -- only meaningful when H is
        # shared across discrete states. For state-dependent H (COM), skip it.
        H_all = np_cpu.array(self.measurement_matrix)
        h_is_shared = bool(np_cpu.allclose(H_all, H_all[:, :, :1]))
        if h_is_shared:
            H_pinv = np_cpu.linalg.pinv(H_all[:, :, 0])
            per_state_means = []
            for j in range(n_states):
                mask = labels == j
                if mask.any():
                    state_obs_mean = windowed[mask].mean(axis=(0, 1))
                else:
                    state_obs_mean = obs_np.mean(axis=0)
                per_state_means.append(H_pinv @ state_obs_mean)

            self.init_mean = jnp.stack([jnp.array(m) for m in per_state_means], axis=1)

        # Data-informed init_cov scaled to observation variance
        obs_var = np_cpu.var(obs_np, axis=0).mean()
        scale = max(float(obs_var), 1e-6)
        self.init_cov = jnp.stack(
            [scale * jnp.identity(self.n_cont_states)] * n_states, axis=2
        )

        # init_discrete_state_prob from GMM mixing weights (smoothed)
        eps = 0.1
        weights = np_cpu.array(gmm.weights_)
        smoothed = weights * (1.0 - eps) + eps / n_states
        smoothed = smoothed / smoothed.sum()
        self.init_discrete_state_prob = jnp.array(smoothed)

    def _initialize_discrete_state_prob(self) -> None:
        """Initializes the starting probability for each discrete state."""
        self.init_discrete_state_prob = (
            jnp.ones(self.n_discrete_states) / self.n_discrete_states
        )

    def _initialize_discrete_transition_matrix(self) -> None:
        """Initializes the discrete state transition matrix (Z).

        Constructs Z based on the `discrete_transition_diag`. Off-diagonal
        elements are set to distribute the remaining probability mass equally
        among other states.
        """
        self.discrete_transition_matrix = make_discrete_transition_matrix(
            self.discrete_transition_diag, self.n_discrete_states
        )

    def _initialize_continuous_state(self, key: Array) -> None:
        """Initializes the initial continuous state mean (m0) and covariance (P0).

        m0 is drawn from a multivariate normal distribution. P0 is set to identity
        and stacked for each discrete state.

        Parameters
        ----------
        key : Array
            JAX random number generator key.
        """
        mean = jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros((self.n_cont_states,)),
            cov=jnp.identity(self.n_cont_states),
        )
        self.init_mean = jnp.stack([mean] * self.n_discrete_states, axis=1)
        self.init_cov = jnp.stack(
            [jnp.identity(self.n_cont_states)] * self.n_discrete_states, axis=2
        )

    @abstractmethod
    def _initialize_measurement_matrix(self, key: Array | None = None):
        """Abstract method to initialize the measurement matrix (H)."""
        pass

    @abstractmethod
    def _initialize_measurement_covariance(self):
        """Abstract method to initialize the measurement covariance (R)."""
        pass

    @abstractmethod
    def _initialize_continuous_transition_matrix(self):
        """Abstract method to initialize the continuous transition matrix (A)."""
        pass

    @abstractmethod
    def _initialize_process_covariance(self):
        """Abstract method to initialize the process covariance (Q)."""
        pass

    @abstractmethod
    def _project_parameters(self):
        """Abstract method to project estimated parameters onto valid spaces."""
        pass

    def _initialize_parameters(self, key: Array) -> None:
        """Initializes all model parameters by calling specific methods.

        Parameters
        ----------
        key : Array
            JAX random number generator key.
        """
        k1, k2 = jax.random.split(key)
        self._initialize_discrete_state_prob()
        self._initialize_discrete_transition_matrix()
        self._initialize_continuous_state(k1)
        self._initialize_measurement_matrix(k2)
        self._initialize_measurement_covariance()
        self._initialize_continuous_transition_matrix()
        self._initialize_process_covariance()

        if self.init_mean.shape != (self.n_cont_states, self.n_discrete_states):
            raise ValueError(
                f"init_mean shape mismatch: expected ({self.n_cont_states}, {self.n_discrete_states}), "
                f"got {self.init_mean.shape}."
            )
        if self.init_cov.shape != (
            self.n_cont_states,
            self.n_cont_states,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"init_cov shape mismatch: expected ({self.n_cont_states}, {self.n_cont_states}, {self.n_discrete_states}), "
                f"got {self.init_cov.shape}."
            )
        if self.init_discrete_state_prob.shape != (self.n_discrete_states,):
            raise ValueError(
                f"init_discrete_state_prob shape mismatch: expected ({self.n_discrete_states},), "
                f"got {self.init_discrete_state_prob.shape}."
            )
        if self.discrete_transition_matrix.shape != (
            self.n_discrete_states,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"discrete_transition_matrix shape mismatch: expected ({self.n_discrete_states}, {self.n_discrete_states}), "
                f"got {self.discrete_transition_matrix.shape}."
            )
        if self.continuous_transition_matrix.shape != (
            self.n_cont_states,
            self.n_cont_states,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"continuous_transition_matrix shape mismatch: expected ({self.n_cont_states}, {self.n_cont_states}, {self.n_discrete_states}), "
                f"got {self.continuous_transition_matrix.shape}."
            )
        if self.process_cov.shape != (
            self.n_cont_states,
            self.n_cont_states,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"process_cov shape mismatch: expected ({self.n_cont_states}, {self.n_cont_states}, {self.n_discrete_states}), "
                f"got {self.process_cov.shape}."
            )
        if self.measurement_matrix.shape != (
            self.n_sources,
            self.n_cont_states,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"measurement_matrix shape mismatch: expected ({self.n_sources}, {self.n_cont_states}, {self.n_discrete_states}), "
                f"got {self.measurement_matrix.shape}."
            )
        if self.measurement_cov.shape != (
            self.n_sources,
            self.n_sources,
            self.n_discrete_states,
        ):
            raise ValueError(
                f"measurement_cov shape mismatch: expected ({self.n_sources}, {self.n_sources}, {self.n_discrete_states}), "
                f"got {self.measurement_cov.shape}."
            )

    def _e_step(self, observations: ArrayLike) -> jax.Array:
        """Performs the Expectation (E) step of the approximate EM algorithm.

        Runs the switching Kalman filter and the configured GPB1 or GPB2
        approximate smoother to compute sufficient statistics and the marginal
        log-likelihood. Both smoothers use moment matching, so exact EM
        monotonicity does not hold.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.

        Returns
        -------
        marginal_log_likelihood : jax.Array
            The log-likelihood of the observations given the current parameters (scalar array).
        """
        obs_arr: jax.Array = jnp.asarray(observations)
        (
            filter_mean,
            filter_cov,
            filter_discrete_state_prob,
            pair_cond_filter_mean,
            pair_cond_filter_cov,
            pair_cond_filter_prob,
            marginal_log_likelihood,
        ) = switching_kalman_filter(
            init_state_cond_mean=self.init_mean,
            init_state_cond_cov=self.init_cov,
            init_discrete_state_prob=self.init_discrete_state_prob,
            obs=obs_arr,
            discrete_transition_matrix=self.discrete_transition_matrix,
            continuous_transition_matrix=self.continuous_transition_matrix,
            process_cov=self.process_cov,
            measurement_matrix=self.measurement_matrix,
            measurement_cov=self.measurement_cov,
        )

        smoother_args = dict(
            filter_mean=filter_mean,
            filter_cov=filter_cov,
            filter_discrete_state_prob=filter_discrete_state_prob,
            process_cov=self.process_cov,
            continuous_transition_matrix=self.continuous_transition_matrix,
        )

        if self.smoother_type == "gpb2":
            (
                _,  # smoother_mean (marginal)
                _,  # smoother_cov (marginal)
                self.smoother_discrete_state_prob,
                self.smoother_joint_discrete_state_prob,
                _,  # smoother_cross_cov (marginal)
                self.smoother_state_cond_mean,
                self.smoother_state_cond_cov,
                self.smoother_pair_cond_cross_cov,
                self.smoother_pair_cond_means,
                self.smoother_pair_cond_covs,
                self.smoother_next_pair_cond_means,
            ) = switching_kalman_smoother_gpb2(
                **smoother_args,
                pair_cond_filter_mean=pair_cond_filter_mean,
                pair_cond_filter_cov=pair_cond_filter_cov,
                pair_cond_filter_prob=pair_cond_filter_prob,
            )
        else:
            (
                _,  # smoother_mean (marginal)
                _,  # smoother_cov (marginal)
                self.smoother_discrete_state_prob,
                self.smoother_joint_discrete_state_prob,
                _,  # smoother_cross_cov (marginal)
                self.smoother_state_cond_mean,
                self.smoother_state_cond_cov,
                self.smoother_pair_cond_cross_cov,
                self.smoother_pair_cond_means,
            ) = switching_kalman_smoother(
                **smoother_args,
                last_filter_conditional_cont_mean=pair_cond_filter_mean[-1],
                discrete_state_transition_matrix=self.discrete_transition_matrix,
            )
            self.smoother_pair_cond_covs = None
            self.smoother_next_pair_cond_means = None

        return jnp.asarray(marginal_log_likelihood)

    def _m_step(self, observations: ArrayLike) -> None:
        """Performs the Maximization (M) step of the EM algorithm.

        Updates the model parameters using the Expected Sufficient Statistics
        computed in the E-step.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        """
        obs_arr: jax.Array = jnp.asarray(observations)
        # The M-step always follows a populated E-step, so the smoother ESS are
        # non-None here (the failed-EM reset only runs before any M-step).
        assert self.smoother_state_cond_mean is not None
        assert self.smoother_state_cond_cov is not None
        assert self.smoother_discrete_state_prob is not None
        assert self.smoother_joint_discrete_state_prob is not None
        assert self.smoother_pair_cond_cross_cov is not None
        (
            A,
            H,
            Q,
            R,
            m0,
            P0,
            Z,
            pi0,
        ) = switching_kalman_maximization_step(
            obs=obs_arr,
            state_cond_smoother_means=self.smoother_state_cond_mean,
            state_cond_smoother_covs=self.smoother_state_cond_cov,
            smoother_discrete_state_prob=self.smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
            pair_cond_smoother_means=self.smoother_pair_cond_means,
            pair_cond_smoother_covs=self.smoother_pair_cond_covs,
            next_pair_cond_smoother_means=self.smoother_next_pair_cond_means,
            transition_prior=self.transition_prior,
        )

        # Update parameters based on flags
        if self.update_continuous_transition_matrix:
            self.continuous_transition_matrix = A
        if self.update_measurement_matrix:
            self.measurement_matrix = H
        if self.update_process_cov:
            self.process_cov = Q
        if self.update_measurement_cov:
            self.measurement_cov = self._pool_measurement_covariance(R)
        if self.update_init_mean:
            self.init_mean = m0
        if self.update_init_cov:
            self.init_cov = P0
        if self.update_discrete_transition_matrix:
            self.discrete_transition_matrix = Z
        # Always update init_prob based on the first smoother probability
        self.init_discrete_state_prob = pi0

    def _stack_shared_measurement_covariance(self, measurement_cov: Array) -> Array:
        """Return a per-state stack from one shared observation covariance."""
        cov_arr = jnp.asarray(measurement_cov)
        if cov_arr.ndim == 3:
            if cov_arr.shape[-1] != self.n_discrete_states:
                raise ValueError(
                    "measurement_cov state axis mismatch: expected "
                    f"{self.n_discrete_states}, got {cov_arr.shape[-1]}."
                )
            return cov_arr
        if cov_arr.shape != (self.n_sources, self.n_sources):
            raise ValueError(
                "shared measurement_cov must have shape "
                f"({self.n_sources}, {self.n_sources}), got {cov_arr.shape}."
            )
        return jnp.stack([cov_arr] * self.n_discrete_states, axis=-1)

    def _measurement_covariance_from_params(self, params: dict) -> Array:
        """Get the shared observation covariance stack for SGD losses."""
        if "measurement_cov" in params:
            return self._stack_shared_measurement_covariance(params["measurement_cov"])
        return self.measurement_cov

    def _store_shared_measurement_covariance(self, params: dict) -> None:
        """Store a shared observation covariance optimized by SGD."""
        if "measurement_cov" in params:
            self.measurement_cov = self._stack_shared_measurement_covariance(
                params["measurement_cov"]
            )

    def _pool_measurement_covariance(self, per_state_cov: Array) -> Array:
        """Pool per-state Kalman M-step covariances into one shared R.

        Eq. 2.18 of Hsin et al. (2024) estimates a single observation-noise
        covariance, not one covariance per switching state.  The generic
        switching Kalman M-step returns state-specific ``R_j`` values; weighting
        them by state responsibilities recovers the pooled update.
        """
        # Called from the M-step, after a populated E-step.
        assert self.smoother_discrete_state_prob is not None
        state_weights = jnp.sum(self.smoother_discrete_state_prob, axis=0)
        total_weight = jnp.maximum(jnp.sum(state_weights), 1e-12)
        pooled = jnp.einsum("j,abj->ab", state_weights, per_state_cov) / total_weight
        pooled = 0.5 * (pooled + pooled.T)
        return self._stack_shared_measurement_covariance(pooled)

    def fit(
        self,
        observations: ArrayLike,
        key: Array | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        skip_init: bool = False,
    ) -> list[float]:
        """Fits the model to observations using the EM algorithm.

        Iteratively performs E-steps and M-steps until convergence or
        the maximum number of iterations is reached.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        key : Array or None, optional
            JAX random key for initialization. Defaults to PRNGKey(0).
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tol : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.
        skip_init : bool, default=False
            If True, skip initialization and warm start (use existing
            parameters). Useful for resuming fitting or providing
            custom initial parameters.

        Returns
        -------
        log_likelihoods : list[float]
            Marginal log-likelihood at each iteration.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        observations = jnp.asarray(observations)
        if not skip_init:
            self._initialize_parameters(key)
            self._warm_initialize_states(observations)
        log_likelihoods: list[float] = []
        last_accepted_state: Optional[dict] = None
        needs_final_e_step = False

        for iteration in range(max_iter):
            current_log_likelihood = float(self._e_step(observations))
            log_likelihoods.append(current_log_likelihood)
            needs_final_e_step = False

            if not jnp.isfinite(current_log_likelihood):
                bad_ll = log_likelihoods.pop()
                if last_accepted_state is not None:
                    self._restore_em_state(last_accepted_state)
                    logger.warning(
                        f"Non-finite log-likelihood at iteration {iteration + 1} "
                        f"({bad_ll}); rolling back to previous E-step and "
                        f"stopping EM."
                    )
                else:
                    # No earlier accepted state to roll back to: the failed
                    # E-step already installed NaN posteriors, so drop them
                    # rather than let decode()/predict_proba() return garbage.
                    self._clear_smoother_state()
                    logger.warning(
                        f"Non-finite log-likelihood at iteration {iteration + 1} "
                        f"with no usable previous state; clearing posteriors and "
                        f"stopping EM."
                    )
                self.converged_ = False
                break

            if iteration > 0:
                # The GPB1 E-step is approximate, so exact monotonicity is not
                # guaranteed. We deliberately stop on the first LL decrease:
                # within-tol decreases fall through to the is_converged check
                # below (treated as a plateau), while larger decreases roll
                # back to the last accepted state. Continuing past a real
                # decrease would risk drifting away from a good iterate.
                is_converged, is_increasing = check_converged(
                    current_log_likelihood, log_likelihoods[-2], tol
                )

                if not is_increasing:
                    bad_ll = log_likelihoods.pop()
                    if last_accepted_state is not None:
                        self._restore_em_state(last_accepted_state)
                    logger.warning(
                        f"LL decreased: {log_likelihoods[-1]:.4f} -> "
                        f"{bad_ll:.4f}; rolling back to previous E-step "
                        f"and stopping EM."
                    )
                    self.converged_ = False
                    break

                if is_converged:
                    logger.info(f"Converged after {iteration + 1} iterations.")
                    self.converged_ = True
                    break

            last_accepted_state = self._snapshot_em_state()
            self._m_step(observations)
            self._project_parameters()
            needs_final_e_step = True

            change = (
                current_log_likelihood - log_likelihoods[-2]
                if iteration > 0
                else float("nan")
            )
            logger.info(
                f"Iteration {iteration + 1}/{max_iter}\t"
                f"Log-Likelihood: {current_log_likelihood:.4f}\t"
                f"Change: {change:.4f}"
            )
        else:
            self.converged_ = False
            logger.warning("Reached maximum iterations without converging.")

        # Final E-step to sync smoother results with current parameters.
        # Without this, the stored posteriors correspond to the previous
        # iteration's parameters after the last M-step. If the final refresh
        # reveals that the last M-step was bad, roll back to the last accepted
        # E-step state instead of returning inconsistent parameters.
        if needs_final_e_step:
            final_ll = float(self._e_step(observations))
            if not jnp.isfinite(final_ll):
                if last_accepted_state is not None:
                    self._restore_em_state(last_accepted_state)
                logger.warning(
                    "Final E-step produced non-finite log-likelihood; "
                    "rolling back to previous E-step."
                )
                self.converged_ = False
            elif log_likelihoods:
                _, is_increasing = check_converged(final_ll, log_likelihoods[-1], tol)
                if is_increasing:
                    log_likelihoods.append(final_ll)
                else:
                    if last_accepted_state is not None:
                        self._restore_em_state(last_accepted_state)
                    logger.warning(
                        f"Final E-step decreased LL: {log_likelihoods[-1]:.4f} -> "
                        f"{final_ll:.4f}; rolling back to previous E-step."
                    )
                    self.converged_ = False
            else:
                log_likelihoods.append(final_ll)

        return log_likelihoods

    # --- SGDFittableMixin protocol (shared by all oscillator subclasses) ---

    def fit_sgd(  # type: ignore[override]  # concrete signature vs mixin *args/**kwargs
        self,
        observations: ArrayLike,
        key: Array | None = None,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        skip_init: bool = False,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        key : Array or None
            JAX random key for parameter initialization.
            If None, defaults to ``jax.random.PRNGKey(0)``.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps.
        convergence_tol : float or None
            If set, stop early when |ΔLL| < tol for 5 consecutive steps.
        skip_init : bool, default=False
            If True, skip initialization and warm start.

        Returns
        -------
        log_likelihoods : list of float
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        observations = jnp.asarray(observations)
        if not skip_init:
            self._initialize_parameters(key)
            self._warm_initialize_states(observations)
        self._sgd_n_time = observations.shape[0]

        return super().fit_sgd(
            observations,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    @property
    def _n_timesteps(self) -> int:
        return self._sgd_n_time

    def _check_sgd_initialized(self) -> None:
        if (
            not hasattr(self, "continuous_transition_matrix")
            or self.continuous_transition_matrix is None
        ):
            raise RuntimeError(
                "Call fit_sgd(observations, key=...) to initialize parameters."
            )

    def _store_sgd_params(self, params: dict) -> None:
        if "measurement_matrix" in params:
            self.measurement_matrix = params["measurement_matrix"]
        if "measurement_cov" in params:
            self.measurement_cov = params["measurement_cov"]
        if "discrete_transition_matrix" in params:
            self.discrete_transition_matrix = params["discrete_transition_matrix"]
        if "init_mean" in params:
            self.init_mean = params["init_mean"]
        if "init_cov" in params:
            self.init_cov = params["init_cov"]
        if "init_discrete_state_prob" in params:
            self.init_discrete_state_prob = params["init_discrete_state_prob"]
        # Subclasses may store additional params (coupling, etc.)

    def _finalize_sgd(self, observations: ArrayLike) -> None:
        self._e_step(observations)

    def _reconstruct_per_state_array(
        self, params: dict, prefix: str, fallback: Array
    ) -> Array:
        """Reconstruct a (…, n_discrete_states) array from per-state PSD params.

        Used by subclass _sgd_loss_fn and _store_sgd_params to reassemble
        arrays like measurement_cov and init_cov from per-state keys
        (e.g. "measurement_cov_0", "measurement_cov_1", ...).
        """
        if not any(k.startswith(f"{prefix}_") for k in params):
            return fallback
        return jnp.stack(
            [
                params.get(f"{prefix}_{j}", fallback[..., j])
                for j in range(self.n_discrete_states)
            ],
            axis=-1,
        )

    # Subclasses must implement _build_param_spec and _sgd_loss_fn


class CommonOscillatorModel(BaseModel):
    """Common Oscillator Model (COM).

    In this model, the **measurement matrix (H)** depends on the discrete
    latent state. This allows different network states to represent
    different ways oscillators contribute to observed signals.
    A, Q, and R are assumed to be constant across discrete states.

    Parameters
    ----------
    freqs : jax.Array, shape (n_oscillators,)
        Intrinsic frequencies of the oscillators.
    damping_coef : jax.Array, shape (n_oscillators,)
        Damping coefficients for each oscillator.
    process_variance : jax.Array, shape (n_oscillators,)
        Process noise variance for each oscillator.
    measurement_variance : float
        Variance of the measurement noise (assumed isotropic and constant).
    """

    def __init__(
        self,
        n_oscillators: int,
        n_discrete_states: int,
        n_sources: int,
        sampling_freq: float,
        freqs: jax.Array,
        damping_coef: jax.Array,
        process_variance: jax.Array,
        measurement_variance: float,
        **kwargs,
    ):
        _reject_forced_update_flags("CommonOscillatorModel", kwargs)
        super().__init__(
            n_oscillators, n_discrete_states, n_sources, sampling_freq, **kwargs
        )
        if freqs.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: freqs {freqs.shape} vs n_oscillators {n_oscillators}"
            )
        _validate_finite_array("freqs", freqs)
        self.freqs = freqs

        if damping_coef.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: damping_coef {damping_coef.shape} vs n_oscillators {n_oscillators}"
            )
        _validate_unit_interval_array("damping_coef", damping_coef)
        self.damping_coef = damping_coef

        if process_variance.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: process_variance {process_variance.shape} vs n_oscillators {n_oscillators}"
            )
        _validate_nonnegative_array("process_variance", process_variance)
        self.process_variance = process_variance

        _validate_positive_scalar("measurement_variance", measurement_variance)
        self.measurement_variance = measurement_variance

        # COM specific M-step update flags
        self.update_continuous_transition_matrix = False
        self.update_process_cov = False
        self.update_measurement_matrix = True

    def _initialize_measurement_matrix(self, key: Array | None = None):
        """Initializes H with small random values, varying across discrete states."""
        if key is None:
            raise ValueError("A JAX PRNGKey must be provided for COM initialization.")
        self.measurement_matrix = jax.random.uniform(
            key,
            (self.n_sources, self.n_cont_states, self.n_discrete_states),
            minval=0.0,
            maxval=0.1,
        )

    def _initialize_measurement_covariance(self):
        """Initializes R as isotropic, constant across discrete states."""
        measurement_cov = jnp.identity(self.n_sources) * self.measurement_variance
        self.measurement_cov = jnp.stack(
            [measurement_cov] * self.n_discrete_states, axis=2
        )

    def _initialize_continuous_transition_matrix(self):
        """Initializes A based on freqs/damping, constant across discrete states."""
        transition_matrix = construct_common_oscillator_transition_matrix(
            freqs=self.freqs,
            damping_coef=self.damping_coef,
            sampling_freq=self.sampling_freq,
        )
        self.continuous_transition_matrix = jnp.stack(
            [transition_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_process_covariance(self):
        """Initializes Q based on variance, constant across discrete states."""
        process_cov = construct_common_oscillator_process_covariance(
            variance=self.process_variance,
        )
        self.process_cov = jnp.stack([process_cov] * self.n_discrete_states, axis=2)

    def _project_parameters(self):
        """No specific projection needed for COM beyond M-step updates."""
        pass  # H is typically unconstrained in COM

    def get_oscillator_influence_on_node(self) -> Array:
        """Calculates the influence of each oscillator on each source.

        This is computed as the L2 norm of the (2x1) block in H
        corresponding to an oscillator-source pair.

        Returns
        -------
        Array, shape (n_sources, n_oscillators, n_discrete_states)
            The influence magnitude.
        """
        n_sources, n_cont_states, n_discrete_states = self.measurement_matrix.shape
        n_oscillators = n_cont_states // 2

        return jnp.sqrt(
            jnp.sum(
                self.measurement_matrix.reshape(
                    (n_sources, n_oscillators, 2, n_discrete_states),
                )
                ** 2,
                axis=2,
            )
        )

    def get_phase_difference(
        self, node1_ind: int, node2_ind: int, oscillator_ind: int
    ) -> Array:
        """Calculates the phase difference between two sources for one oscillator.

        Parameters
        ----------
        node1_ind : int
            Index of the first source.
        node2_ind : int
            Index of the second source.
        oscillator_ind : int
            Index of the oscillator.

        Returns
        -------
        Array, shape (n_discrete_states,)
            The phase difference in radians for each discrete state.
        """
        _, col = get_block_slice(oscillator_ind, oscillator_ind)
        coef1 = self.measurement_matrix[node1_ind, col, :]  # Shape (2, M)
        coef2 = self.measurement_matrix[node2_ind, col, :]  # Shape (2, M)

        phase1 = jnp.arctan2(coef1[1], coef1[0])  # Shape (M,)
        phase2 = jnp.arctan2(coef2[1], coef2[0])  # Shape (M,)

        return phase1 - phase2

    def fit(
        self,
        observations: ArrayLike,
        key: Array | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        skip_init: bool = False,
    ) -> list[float]:
        """Fits the model to observations using the EM algorithm.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        key : Array or None, optional
            JAX random key for initialization. Defaults to PRNGKey(0).
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tol : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.
        skip_init : bool, default=False
            If True, skip initialization and warm start.

        Returns
        -------
        log_likelihoods : list[float]
            Marginal log-likelihood at each iteration.
        """
        observations = jnp.asarray(observations)
        if observations.shape[1] != self.n_sources:
            raise ValueError(
                f"observations must have {self.n_sources} sources, "
                f"got {observations.shape[1]}."
            )
        return super().fit(observations, key, max_iter, tol, skip_init=skip_init)

    # --- SGDFittableMixin: COM-specific param spec and loss ---

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            PSD_MATRIX,
            STOCHASTIC_ROW,
            UNCONSTRAINED,
        )

        params: dict = {}
        spec: dict = {}

        if self.update_measurement_matrix:
            params["measurement_matrix"] = self.measurement_matrix
            spec["measurement_matrix"] = UNCONSTRAINED
        if self.update_measurement_cov:
            params["measurement_cov"] = self.measurement_cov[..., 0]
            spec["measurement_cov"] = PSD_MATRIX
        if self.update_discrete_transition_matrix:
            params["discrete_transition_matrix"] = self.discrete_transition_matrix
            spec["discrete_transition_matrix"] = STOCHASTIC_ROW
        if self.update_init_mean:
            params["init_mean"] = self.init_mean
            spec["init_mean"] = UNCONSTRAINED
        if self.update_init_cov:
            for j in range(self.n_discrete_states):
                key = f"init_cov_{j}"
                params[key] = self.init_cov[..., j]
                spec[key] = PSD_MATRIX

        return params, spec

    def _sgd_loss_fn(self, params: dict, observations: Array) -> Array:
        H = params.get("measurement_matrix", self.measurement_matrix)
        Z = params.get("discrete_transition_matrix", self.discrete_transition_matrix)
        m0 = params.get("init_mean", self.init_mean)
        R = self._measurement_covariance_from_params(params)
        P0 = self._reconstruct_per_state_array(params, "init_cov", self.init_cov)

        result = switching_kalman_filter(
            init_state_cond_mean=m0,
            init_state_cond_cov=P0,
            init_discrete_state_prob=self.init_discrete_state_prob,
            obs=observations,
            discrete_transition_matrix=Z,
            continuous_transition_matrix=self.continuous_transition_matrix,
            process_cov=self.process_cov,
            measurement_matrix=H,
            measurement_cov=R,
        )
        return -jnp.asarray(result[6])  # scalar marginal_ll

    def _store_sgd_params(self, params: dict) -> None:
        super()._store_sgd_params(params)
        self._store_shared_measurement_covariance(params)
        self.init_cov = self._reconstruct_per_state_array(
            params, "init_cov", self.init_cov
        )


class CorrelatedNoiseModel(BaseModel):
    """Correlated Noise Model (CNM).

    In this model, the **process noise covariance (Q)** depends on the discrete
    latent state. This allows different network states to represent
    different patterns of shared noise or input, implying functional connectivity
    changes. A, H, and R are constant. n_sources must equal n_oscillators.

    Parameters
    ----------
    freqs : jax.Array, shape (n_oscillators,)
        Intrinsic frequencies of the oscillators.
    damping_coef : jax.Array, shape (n_oscillators,)
        Damping coefficients for each oscillator.
    process_variance : jax.Array, shape (n_oscillators, n_discrete_states)
        Process noise variance for each oscillator and state.
    measurement_variance : float
        Variance of the measurement noise.
    phase_difference : jax.Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial phase differences for noise correlation. Each oscillator pair may
        be supplied in the strict upper triangle, strict lower triangle, or both
        triangles if the two entries are opposite phases; values are stored
        canonically in the strict upper triangle.
    coupling_strength : jax.Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial coupling strengths for noise correlation. Each oscillator pair
        may be supplied in the strict upper triangle, strict lower triangle, or
        both triangles if the two entries agree; values are stored canonically in
        the strict upper triangle.
    """

    def __init__(
        self,
        n_oscillators: int,
        n_discrete_states: int,
        sampling_freq: float,
        freqs: jax.Array,
        damping_coef: jax.Array,
        process_variance: jax.Array,
        measurement_variance: float,
        phase_difference: jax.Array,
        coupling_strength: jax.Array,
        **kwargs,
    ):
        # n_sources is fixed to n_oscillators for CNM (passed to super below).
        _reject_forced_update_flags("CorrelatedNoiseModel", kwargs)
        super().__init__(
            n_oscillators,
            n_discrete_states,
            n_oscillators,
            sampling_freq,
            **kwargs,
        )
        if freqs.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: freqs {freqs.shape} vs n_oscillators {n_oscillators}"
            )
        if damping_coef.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: damping_coef {damping_coef.shape} vs n_oscillators {n_oscillators}"
            )
        if process_variance.shape != (n_oscillators, n_discrete_states):
            raise ValueError(
                "process_variance must have shape (n_oscillators, n_discrete_states)."
                f" Got {process_variance.shape}."
            )
        _validate_finite_array("freqs", freqs)
        _validate_unit_interval_array("damping_coef", damping_coef)
        _validate_nonnegative_array("process_variance", process_variance)
        if phase_difference.shape != (n_oscillators, n_oscillators, n_discrete_states):
            raise ValueError(
                "phase_difference must have shape (n_oscillators, n_oscillators, n_discrete_states)."
                f" Got {phase_difference.shape}."
            )
        if coupling_strength.shape != (n_oscillators, n_oscillators, n_discrete_states):
            raise ValueError(
                "coupling_strength must have shape (n_oscillators, n_oscillators, n_discrete_states)."
                f" Got {coupling_strength.shape}."
            )
        _validate_positive_scalar("measurement_variance", measurement_variance)
        phase_difference, coupling_strength = (
            canonicalize_correlated_noise_pair_parameters(
                phase_difference, coupling_strength
            )
        )

        self.freqs = freqs
        self.damping_coef = damping_coef
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.phase_difference = phase_difference
        self.coupling_strength = coupling_strength

        # CNM specific M-step update flags
        self.update_continuous_transition_matrix = False
        self.update_measurement_matrix = False  # H is fixed in CNM
        self.update_process_cov = True

    def _initialize_measurement_matrix(self, key: Array | None = None):
        """Initializes H as block-diagonal [1, 0], constant across states."""
        measurement_matrix = construct_correlated_noise_measurement_matrix(
            self.n_sources,
        )
        self.measurement_matrix = jnp.stack(
            [measurement_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_measurement_covariance(self):
        """Initializes R as isotropic, constant across discrete states."""
        measurement_cov = jnp.identity(self.n_sources) * self.measurement_variance
        self.measurement_cov = jnp.stack(
            [measurement_cov] * self.n_discrete_states, axis=2
        )

    def _initialize_continuous_transition_matrix(self):
        """Initializes A based on freqs/damping, constant across discrete states."""
        transition_matrix = construct_common_oscillator_transition_matrix(
            freqs=self.freqs,
            damping_coef=self.damping_coef,
            sampling_freq=self.sampling_freq,
        )
        self.continuous_transition_matrix = jnp.stack(
            [transition_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_process_covariance(self):
        """Initializes Q based on initial params, varying across discrete states."""
        self.process_cov = jnp.stack(
            [
                construct_correlated_noise_process_covariance(
                    variance=self.process_variance[..., state_ind],
                    phase_difference=self.phase_difference[..., state_ind],
                    coupling_strength=self.coupling_strength[..., state_ind],
                )
                for state_ind in range(self.n_discrete_states)
            ],
            axis=2,
        )
        # Needs shape (n_cont_states, n_cont_states, n_discrete_states).
        # The constructor guarantees symmetry; reject a too-strong coupling that
        # makes Q symmetric-but-indefinite (it would reach the switching filter on
        # EM iteration 0, before _project_parameters floors the eigenvalues).
        validate_covariance(
            self.process_cov, "process_cov", require_positive_definite=False
        )

    def _project_parameters(self):
        """Project each per-state Q to the CNM covariance family.

        The generic switching Kalman M-step estimates an unconstrained
        covariance. CNM requires scalar-identity diagonal oscillator blocks and
        symmetric scaled-rotation cross-blocks, so project back to that
        structure and update the public scientific parameters accordingly.
        """
        self.process_cov = jnp.stack(
            [
                project_correlated_noise_process_covariance(self.process_cov[..., j])
                for j in range(self.n_discrete_states)
            ],
            axis=-1,
        )
        self._sync_process_covariance_params()

    def _sync_process_covariance_params(self) -> None:
        """Sync CNM scientific parameters from the structured Q stack."""
        variances = []
        phases = []
        couplings = []
        for j in range(self.n_discrete_states):
            params = extract_correlated_noise_params_from_covariance(
                self.process_cov[..., j],
                self.n_oscillators,
            )
            variances.append(params["variance"])
            phases.append(params["phase_difference"])
            couplings.append(params["coupling_strength"])

        self.process_variance = jnp.stack(variances, axis=-1)
        self.phase_difference = jnp.stack(phases, axis=-1)
        self.coupling_strength = jnp.stack(couplings, axis=-1)

    def fit(
        self,
        observations: ArrayLike,
        key: Array | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        skip_init: bool = False,
    ) -> list[float]:
        """Fits the model to observations using the EM algorithm.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        key : Array or None, optional
            JAX random key for initialization. Defaults to PRNGKey(0).
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tol : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.
        skip_init : bool, default=False
            If True, skip initialization and warm start.

        Returns
        -------
        log_likelihoods : list[float]
            Marginal log-likelihood at each iteration.
        """
        observations = jnp.asarray(observations)
        if observations.shape[1] != self.n_sources:
            raise ValueError(
                f"observations must have {self.n_sources} sources, "
                f"got {observations.shape[1]}."
            )
        return super().fit(observations, key, max_iter, tol, skip_init=skip_init)

    # --- SGDFittableMixin: CNM-specific param spec and loss ---

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import (
            PSD_MATRIX,
            POSITIVE,
            STOCHASTIC_ROW,
            UNCONSTRAINED,
        )

        params: dict = {}
        spec: dict = {}

        if self.update_process_cov:
            params["process_variance"] = self.process_variance
            spec["process_variance"] = POSITIVE
            params["phase_difference"] = self.phase_difference
            spec["phase_difference"] = UNCONSTRAINED
            params["coupling_strength"] = self.coupling_strength
            spec["coupling_strength"] = UNCONSTRAINED

        if self.update_discrete_transition_matrix:
            params["discrete_transition_matrix"] = self.discrete_transition_matrix
            spec["discrete_transition_matrix"] = STOCHASTIC_ROW

        if self.update_measurement_cov:
            params["measurement_cov"] = self.measurement_cov[..., 0]
            spec["measurement_cov"] = PSD_MATRIX

        if self.update_init_mean:
            params["init_mean"] = self.init_mean
            spec["init_mean"] = UNCONSTRAINED

        if self.update_init_cov:
            for j in range(self.n_discrete_states):
                k = f"init_cov_{j}"
                params[k] = self.init_cov[..., j]
                spec[k] = PSD_MATRIX

        return params, spec

    def _sgd_loss_fn(self, params: dict, observations: Array) -> Array:
        Z = params.get("discrete_transition_matrix", self.discrete_transition_matrix)
        m0 = params.get("init_mean", self.init_mean)
        R = self._measurement_covariance_from_params(params)
        P0 = self._reconstruct_per_state_array(params, "init_cov", self.init_cov)

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

        result = switching_kalman_filter(
            init_state_cond_mean=m0,
            init_state_cond_cov=P0,
            init_discrete_state_prob=self.init_discrete_state_prob,
            obs=observations,
            discrete_transition_matrix=Z,
            continuous_transition_matrix=self.continuous_transition_matrix,
            process_cov=Q,
            measurement_matrix=self.measurement_matrix,
            measurement_cov=R,
        )
        return -jnp.asarray(result[6])

    def _store_sgd_params(self, params: dict) -> None:
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
            Q_raw = jax.vmap(
                construct_correlated_noise_process_covariance,
                in_axes=(-1, -1, -1),
                out_axes=-1,
            )(self.process_variance, self.phase_difference, self.coupling_strength)
            self.process_cov = jax.vmap(shift_to_psd, in_axes=-1, out_axes=-1)(Q_raw)
            self._sync_process_covariance_params()
        self._store_shared_measurement_covariance(params)
        self.init_cov = self._reconstruct_per_state_array(
            params, "init_cov", self.init_cov
        )


class DirectedInfluenceModel(BaseModel):
    """Directed Influence Model (DIM).

    In this model, the **continuous transition matrix (A)** depends on the
    discrete latent state. This allows different network states to
    represent different patterns of direct influence or coupling between
    oscillators. H, Q, and R are constant. n_sources must equal n_oscillators.

    Parameters
    ----------
    freqs : jax.Array, shape (n_oscillators,)
        Intrinsic frequencies of the oscillators.
    damping_coef : jax.Array, shape (n_oscillators,)
        Damping coefficients for each oscillator.
    process_variance : jax.Array, shape (n_oscillators,)
        Process noise variance (constant across states).
    measurement_variance : float
        Variance of the measurement noise.
    phase_difference : jax.Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial phase differences for coupling.
    coupling_strength : jax.Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial coupling strengths.
    use_reparameterized_mstep : bool, default=False
        If True, use reparameterized M-step that directly optimizes oscillator
        parameters (damping, freq, coupling_strength, phase_diff) to maximize
        the Q-function. This guarantees valid oscillator structure and monotonic
        log-likelihood increase. If False, use standard M-step with projection.
    max_spectral_radius : float, default=0.99
        Target upper bound on the spectral radius of each state's transition
        matrix. The differentiable stability scale shrinks damping and coupling
        so the block-row operator-norm bound stays at or below this value. A
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
        n_discrete_states: int,
        sampling_freq: float,
        freqs: jax.Array,
        damping_coef: jax.Array,
        process_variance: jax.Array,
        measurement_variance: float,
        phase_difference: jax.Array,
        coupling_strength: jax.Array,
        use_reparameterized_mstep: bool = False,
        max_spectral_radius: float = 0.99,
        max_damping: float = 0.995,
        **kwargs,
    ):
        # n_sources is fixed to n_oscillators for DIM (passed to super below).
        _reject_forced_update_flags("DirectedInfluenceModel", kwargs)
        super().__init__(
            n_oscillators,
            n_discrete_states,
            n_oscillators,
            sampling_freq,
            **kwargs,
        )

        if freqs.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: freqs {freqs.shape} vs n_oscillators {n_oscillators}"
            )
        _validate_finite_array("freqs", freqs)
        self.freqs = freqs
        if damping_coef.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: damping_coef {damping_coef.shape} vs n_oscillators {n_oscillators}"
            )
        _validate_unit_interval_array("damping_coef", damping_coef)
        self.damping_coef = damping_coef
        if process_variance.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: process_variance {process_variance.shape} vs n_oscillators {n_oscillators}"
            )
        _validate_nonnegative_array("process_variance", process_variance)
        self.process_variance = process_variance

        _validate_positive_scalar("measurement_variance", measurement_variance)
        self.measurement_variance = measurement_variance

        if phase_difference.shape != (n_oscillators, n_oscillators, n_discrete_states):
            raise ValueError(
                "phase_difference must have shape (n_oscillators, n_oscillators, n_discrete_states)."
                f" Got {phase_difference.shape}."
            )

        if coupling_strength.shape != (n_oscillators, n_oscillators, n_discrete_states):
            raise ValueError(
                "coupling_strength must have shape (n_oscillators, n_oscillators, n_discrete_states)."
                f" Got {coupling_strength.shape}."
            )
        phase_difference = jnp.asarray(phase_difference)
        coupling_strength = jnp.asarray(coupling_strength)
        _validate_finite_array("phase_difference", phase_difference)
        _validate_finite_array("coupling_strength", coupling_strength)
        diag_idx = jnp.arange(n_oscillators)
        diag_phase = phase_difference[diag_idx, diag_idx, :]
        diag_coupling = coupling_strength[diag_idx, diag_idx, :]
        if bool(jnp.any(jnp.abs(diag_phase) > 1e-8)):
            raise ValueError(
                "DIM phase_difference diagonal entries are ignored by the "
                "transition model and must be zero."
            )
        if bool(jnp.any(jnp.abs(diag_coupling) > 1e-8)):
            raise ValueError(
                "DIM coupling_strength diagonal entries are ignored by the "
                "transition model and must be zero."
            )
        self.phase_difference = phase_difference.at[diag_idx, diag_idx, :].set(0.0)
        self.coupling_strength = coupling_strength.at[diag_idx, diag_idx, :].set(0.0)

        # DIM specific M-step update flags
        self.update_continuous_transition_matrix = True
        self.update_measurement_matrix = False  # H is fixed in DIM
        self.update_process_cov = False  # Q is fixed in DIM

        # Reparameterized M-step option
        self.use_reparameterized_mstep = use_reparameterized_mstep
        # Store current oscillator params for warm-starting optimizer
        self._current_osc_params: Optional[dict] = None

        # Stability bounds applied when rebuilding transition matrices.
        if not 0.0 < max_spectral_radius < 1.0:
            raise ValueError("max_spectral_radius must lie in (0, 1).")
        if not 0.0 < max_damping < 1.0:
            raise ValueError("max_damping must lie in (0, 1).")
        self.max_spectral_radius = max_spectral_radius
        self.max_damping = max_damping

    def _initialize_measurement_matrix(self, key: Array | None = None):
        """Initializes H with [1/sqrt(2), 1/sqrt(2)] blocks, constant across states."""
        measurement_matrix = construct_directed_influence_measurement_matrix(
            self.n_sources,
        )
        self.measurement_matrix = jnp.stack(
            [measurement_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_measurement_covariance(self):
        """Initializes R as isotropic, constant across discrete states."""
        measurement_cov = jnp.identity(self.n_sources) * self.measurement_variance
        self.measurement_cov = jnp.stack(
            [measurement_cov] * self.n_discrete_states, axis=2
        )

    def _initialize_continuous_transition_matrix(self):
        """Initializes A based on initial params, varying across discrete states."""
        self._rebuild_stable_transition_matrix()

    def _effective_dim_scale(self) -> Array:
        """Global stability scale for the current (intrinsic) DIM parameters.

        ``continuous_transition_matrix`` is built from ``damping_coef * scale``
        and ``coupling_strength * scale``; reconstructing it from the public
        parameters requires re-applying this same scale.
        """
        return _dim_stability_scale(
            self.freqs,
            self.damping_coef,
            self.coupling_strength,
            self.sampling_freq,
            max_spectral_radius=self.max_spectral_radius,
        )

    def _rebuild_stable_transition_matrix(self) -> None:
        """Rebuild stable DIM matrices from the intrinsic scientific params.

        The stability scale is applied only to the *effective* damping and
        coupling used to build ``A``; it is deliberately NOT written back onto
        ``self.damping_coef`` / ``self.coupling_strength``. Keeping the public
        parameters as the intrinsic source makes the rebuild idempotent --
        re-stabilizing already-stable params is a no-op -- so damping does not
        drift toward zero across successive fits with strong coupling (the scale
        would otherwise compound into the accumulating public damping). ``A`` is
        reconstructable from the public params by re-applying the same
        :func:`_dim_stability_scale` (see ``_effective_dim_scale``).
        """
        scale = self._effective_dim_scale()
        effective_damping = jnp.asarray(self.damping_coef) * scale
        effective_coupling = jnp.asarray(self.coupling_strength) * scale
        self.continuous_transition_matrix = jax.vmap(
            lambda phase, coupling: construct_directed_influence_transition_matrix(
                freqs=self.freqs,
                damping_coeffs=effective_damping,
                coupling_strengths=coupling,
                phase_diffs=phase,
                sampling_freq=self.sampling_freq,
            ),
            in_axes=(-1, -1),
            out_axes=-1,
        )(self.phase_difference, effective_coupling)

        if self._current_osc_params is not None:
            self._current_osc_params = {
                "freq": self.freqs,
                "damping": self.damping_coef,
                "coupling_strength": self.coupling_strength,
                "phase_diff": self.phase_difference,
            }

    def _initialize_process_covariance(self):
        """Initializes Q based on variance, constant across discrete states."""
        process_cov = construct_common_oscillator_process_covariance(
            variance=self.process_variance,
        )
        self.process_cov = jnp.stack([process_cov] * self.n_discrete_states, axis=2)

    def _m_step(self, observations: ArrayLike) -> None:
        """Performs the M-step, with optional reparameterized transition update.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        """
        if self.use_reparameterized_mstep:
            self._m_step_reparameterized(observations)
        else:
            super()._m_step(observations)

    def _m_step_reparameterized(self, observations: ArrayLike) -> None:
        """M-step using reparameterized optimization for A.

        Instead of solving for A directly and projecting, this optimizes
        the underlying oscillator parameters (damping, freq, coupling_strength,
        phase_diff) to maximize the Q-function. This guarantees valid oscillator
        structure by construction.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        """
        obs_arr: jax.Array = jnp.asarray(observations)
        # The M-step always follows a populated E-step, so the smoother ESS are
        # non-None here (the failed-EM reset only runs before any M-step).
        assert self.smoother_state_cond_mean is not None
        assert self.smoother_state_cond_cov is not None
        assert self.smoother_discrete_state_prob is not None
        assert self.smoother_joint_discrete_state_prob is not None
        assert self.smoother_pair_cond_cross_cov is not None
        # First, run the standard M-step for all parameters except A
        (
            _,  # A - we'll compute this ourselves
            H,
            Q,
            R,
            m0,
            P0,
            Z,
            pi0,
        ) = switching_kalman_maximization_step(
            obs=obs_arr,
            state_cond_smoother_means=self.smoother_state_cond_mean,
            state_cond_smoother_covs=self.smoother_state_cond_cov,
            smoother_discrete_state_prob=self.smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
            pair_cond_smoother_means=self.smoother_pair_cond_means,
            pair_cond_smoother_covs=self.smoother_pair_cond_covs,
            next_pair_cond_smoother_means=self.smoother_next_pair_cond_means,
            transition_prior=self.transition_prior,
        )

        # Update non-A parameters based on flags (same as standard M-step)
        if self.update_measurement_matrix:
            self.measurement_matrix = H
        if self.update_process_cov:
            self.process_cov = Q
        if self.update_measurement_cov:
            self.measurement_cov = self._pool_measurement_covariance(R)
        if self.update_init_mean:
            self.init_mean = m0
        if self.update_init_cov:
            self.init_cov = P0
        if self.update_discrete_transition_matrix:
            self.discrete_transition_matrix = Z
        self.init_discrete_state_prob = pi0

        # Now optimize all state-specific A matrices in one shared parameter space.
        if self.update_continuous_transition_matrix:
            # Compute sufficient statistics for transition matrix
            gamma1, beta = compute_transition_sufficient_stats(
                state_cond_smoother_means=self.smoother_state_cond_mean,
                state_cond_smoother_covs=self.smoother_state_cond_cov,
                smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
                pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
                pair_cond_smoother_means=self.smoother_pair_cond_means,
                pair_cond_smoother_covs=self.smoother_pair_cond_covs,
                next_pair_cond_smoother_means=self.smoother_next_pair_cond_means,
            )

            # Initialize the joint warm start from the current shared/public
            # scientific parameters. The joint objective applies the same
            # effective stability scale used to construct A.
            if self._current_osc_params is None:
                self._current_osc_params = {
                    "freq": self.freqs,
                    "damping": self.damping_coef,
                    "coupling_strength": self.coupling_strength,
                    "phase_diff": self.phase_difference,
                }

            self._current_osc_params = optimize_dim_transition_params_joint(
                gamma1=gamma1,
                beta=beta,
                init_params=self._current_osc_params,
                sampling_freq=self.sampling_freq,
                process_cov=self.process_cov,
                max_spectral_radius=self.max_spectral_radius,
                max_damping=self.max_damping,
            )

            # Sync the one joint solution directly; no post-hoc averaging of
            # independently optimized state-specific frequency/damping values.
            self._update_public_oscillator_params()

            # Rebuild from the shared intrinsic parameters and enforce one
            # global stability scale, preserving exact reconstructability.
            self._rebuild_stable_transition_matrix()

    def _update_public_oscillator_params(self) -> None:
        """Update public oscillator attributes from optimized parameters.

        After the joint reparameterized M-step, syncs the private optimizer
        result to the public attributes. Frequency/damping are already shared;
        coupling/phase retain their discrete-state axis.
        """
        if self._current_osc_params is None:
            return

        self.freqs = self._current_osc_params["freq"]
        self.damping_coef = self._current_osc_params["damping"]
        self.coupling_strength = self._current_osc_params["coupling_strength"]
        self.phase_difference = self._current_osc_params["phase_diff"]

    def _project_parameters(self):
        """Project transition matrices to the DIM oscillator family.

        With reparameterized M-step, A is already valid by construction.
        With standard M-step, the generic Kalman update is unconstrained; the
        paper's M-step therefore projects each 2x2 block back to scaled-rotation
        structure and then enforces stability.
        """
        if self.use_reparameterized_mstep:
            return

        projected = []
        for j in range(self.n_discrete_states):
            A_unc_j = self.continuous_transition_matrix[..., j]
            A_j = project_coupled_transition_matrix(A_unc_j)

            # Enforce spectral radius < 1 for stability (unconditional).
            # Stability is a hard physical constraint: an unstable A causes
            # state divergence and invalidates the E-step posteriors. Unlike
            # the block structure projection above, this is not optional.
            A_j = _stabilize_transition_matrix(
                A_j, max_spectral_radius=self.max_spectral_radius
            )

            projected.append(A_j)
        self.continuous_transition_matrix = jnp.stack(projected, axis=-1)

        # Sync all scientific parameters, then rebuild A so the public shared
        # frequency/damping plus per-state coupling/phase exactly represent it.
        self._sync_coupling_from_transition_matrix()

    def _sync_coupling_from_transition_matrix(self) -> None:
        """Synchronize scientific parameters from the current transition matrix.

        Called after standard EM projection so shared frequency/damping and
        per-state coupling/phase all reflect the fitted A rather than the
        initial values.
        """
        frequency_list = []
        damping_list = []
        coupling_list = []
        phase_list = []
        for j in range(self.n_discrete_states):
            params = extract_dim_params_from_matrix(
                self.continuous_transition_matrix[..., j],
                self.sampling_freq,
                self.n_oscillators,
            )
            frequency_list.append(params["freq"])
            damping_list.append(params["damping"])
            coupling_list.append(params["coupling_strength"])
            phase_list.append(params["phase_diff"])
        self.freqs = jnp.mean(jnp.stack(frequency_list, axis=-1), axis=-1)
        self.damping_coef = jnp.mean(jnp.stack(damping_list, axis=-1), axis=-1)
        self.coupling_strength = jnp.stack(coupling_list, axis=-1)
        self.phase_difference = jnp.stack(phase_list, axis=-1)
        self._rebuild_stable_transition_matrix()

    def fit(
        self,
        observations: ArrayLike,
        key: Array | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        skip_init: bool = False,
    ) -> list[float]:
        """Fits the model to observations using the EM algorithm.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        key : Array or None, optional
            JAX random key for initialization. Defaults to PRNGKey(0).
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tol : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.
        skip_init : bool, default=False
            If True, skip initialization and warm start.

        Returns
        -------
        log_likelihoods : list[float]
            Marginal log-likelihood at each iteration.
        """
        observations = jnp.asarray(observations)
        if observations.shape[1] != self.n_sources:
            raise ValueError(
                f"observations must have {self.n_sources} sources, "
                f"got {observations.shape[1]}."
            )
        return super().fit(observations, key, max_iter, tol, skip_init=skip_init)

    # --- SGDFittableMixin: DIM-specific param spec and loss ---

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

        if self.update_measurement_cov:
            params["measurement_cov"] = self.measurement_cov[..., 0]
            spec["measurement_cov"] = PSD_MATRIX

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

    def fit_sgd(  # type: ignore[override]  # concrete signature vs mixin *args/**kwargs
        self,
        observations: ArrayLike,
        key: Array | None = None,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        connectivity_penalty: Optional[object] = None,
        skip_init: bool = False,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        SGD optimizes ``coupling_strength`` and ``phase_difference``
        (and optionally discrete transition, init params, measurement cov).
        Frequencies (``freqs``) and damping (``damping_coef``) are not direct
        SGD variables and are kept as intrinsic parameters. When a candidate
        would otherwise be unstable, a global stability scale reduces the
        *effective* damping and coupling used to build the transition matrix
        ``A`` (bounding its spectral radius), but the public ``damping_coef`` /
        ``coupling_strength`` are left at their intrinsic values -- reconstruct
        ``A`` by re-applying :meth:`_effective_dim_scale`.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
        key : Array or None
            JAX random key for parameter initialization.
            If None, defaults to ``jax.random.PRNGKey(0)``.
        optimizer : optax optimizer or None
        num_steps : int
        verbose : bool
        convergence_tol : float or None
        connectivity_penalty : OscillatorPenaltyConfig or None
            If provided, adds structured sparsity penalties on
            coupling_strength during SGD optimization.
        skip_init : bool, default=False
            If True, skip initialization and warm start.

        Returns
        -------
        log_likelihoods : list of float
        """
        self._connectivity_penalty = connectivity_penalty
        return super().fit_sgd(
            observations,
            key=key,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
            skip_init=skip_init,
        )

    def _sgd_loss_fn(self, params: dict, observations) -> jax.Array:
        phase_diff = params.get("phase_difference", self.phase_difference)
        coupling = params.get("coupling_strength", self.coupling_strength)
        stability_scale = _dim_stability_scale(
            self.freqs,
            self.damping_coef,
            coupling,
            self.sampling_freq,
            max_spectral_radius=self.max_spectral_radius,
        )
        effective_damping = self.damping_coef * stability_scale
        effective_coupling = coupling * stability_scale

        # Vectorize A construction over discrete states (last axis)
        A = jax.vmap(
            lambda pd, cs: construct_directed_influence_transition_matrix(
                freqs=self.freqs,
                damping_coeffs=effective_damping,
                phase_diffs=pd,
                coupling_strengths=cs,
                sampling_freq=self.sampling_freq,
            ),
            in_axes=(-1, -1),
            out_axes=-1,
        )(phase_diff, effective_coupling)

        Z = params.get("discrete_transition_matrix", self.discrete_transition_matrix)
        m0 = params.get("init_mean", self.init_mean)
        R = self._measurement_covariance_from_params(params)
        P0 = self._reconstruct_per_state_array(params, "init_cov", self.init_cov)

        result = switching_kalman_filter(
            init_state_cond_mean=m0,
            init_state_cond_cov=P0,
            init_discrete_state_prob=self.init_discrete_state_prob,
            obs=observations,
            discrete_transition_matrix=Z,
            continuous_transition_matrix=A,
            process_cov=self.process_cov,
            measurement_matrix=self.measurement_matrix,
            measurement_cov=R,
        )
        base_loss = -result[6]

        # Add connectivity penalty if configured
        penalty_config = getattr(self, "_connectivity_penalty", None)
        if penalty_config is not None:
            from state_space_practice.oscillator_regularization import (
                total_connectivity_penalty,
            )

            # coupling shape: (n_osc, n_osc, n_states) → (n_states, n_osc, n_osc)
            coupling_transposed = jnp.moveaxis(effective_coupling, -1, 0)
            base_loss = base_loss + total_connectivity_penalty(
                coupling_transposed,
                penalty_config,
                n_timesteps=self._n_timesteps,
            )

        return jnp.asarray(base_loss)

    def _store_sgd_params(self, params: dict) -> None:
        super()._store_sgd_params(params)
        if "phase_difference" in params:
            self.phase_difference = params["phase_difference"]
        if "coupling_strength" in params:
            self.coupling_strength = params["coupling_strength"]
        diag_idx = jnp.arange(self.n_oscillators)
        self.phase_difference = self.phase_difference.at[diag_idx, diag_idx, :].set(0.0)
        self.coupling_strength = self.coupling_strength.at[diag_idx, diag_idx, :].set(
            0.0
        )
        # Reconstruct A from updated scientific params using the same global
        # stability scale evaluated by the SGD loss.
        if "phase_difference" in params or "coupling_strength" in params:
            self._rebuild_stable_transition_matrix()
        self._store_shared_measurement_covariance(params)
        self.init_cov = self._reconstruct_per_state_array(
            params, "init_cov", self.init_cov
        )
