"""Implementation of the switching oscillator models

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
from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
    construct_correlated_noise_measurement_matrix,
    construct_correlated_noise_process_covariance,
    construct_directed_influence_measurement_matrix,
    construct_directed_influence_transition_matrix,
    extract_dim_params_from_matrix,
    get_block_slice,
    project_coupled_transition_matrix,
    project_matrix_blockwise,
)
from state_space_practice.switching_kalman import (
    compute_transition_sufficient_stats,
    optimize_dim_transition_params,
    switching_kalman_filter,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
)
from state_space_practice.utils import check_converged

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
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

    def __init__(
        self,
        n_oscillators: int,
        n_discrete_states: int,
        n_sources: int,
        sampling_freq: float,
        discrete_transition_diag: Optional[jax.Array] = None,
        update_discrete_transition_matrix: bool = True,
        update_continuous_transition_matrix: bool = True,
        update_measurement_matrix: bool = True,
        update_process_cov: bool = True,
        update_measurement_cov: bool = True,
        update_init_mean: bool = True,
        update_init_cov: bool = True,
    ):
        self.n_oscillators = n_oscillators
        self.n_discrete_states = n_discrete_states
        self.n_sources = n_sources
        self.sampling_freq = sampling_freq
        self.n_cont_states = 2 * n_oscillators

        # Set default diagonal if not provided
        if discrete_transition_diag is None:
            self.discrete_transition_diag = jnp.full(
                (n_discrete_states,), 0.95, dtype=jnp.float32
            )
        else:
            self.discrete_transition_diag = discrete_transition_diag

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

        # Placeholders for smoother results (Expected Sufficient Statistics - ESS)
        self.smoother_state_cond_mean: jax.Array
        self.smoother_state_cond_cov: jax.Array
        self.smoother_discrete_state_prob: jax.Array
        self.smoother_joint_discrete_state_prob: jax.Array
        self.smoother_pair_cond_cross_cov: jax.Array
        self.smoother_pair_cond_means: jax.Array  # E[x_t | S_t=i, S_{t+1}=j]

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
        ]

        # Add update flags for clarity
        update_flags = {
            "Z": self.update_discrete_transition_matrix,
            "A": self.update_continuous_transition_matrix,
            "H": self.update_measurement_matrix,
            "Q": self.update_process_cov,
            "R": self.update_measurement_cov,
            "m0": self.update_init_mean,
            "P0": self.update_init_cov,
        }

        # Filter for flags that are False (since True is default/expected)
        # Or show all for full clarity - let's show all for now.
        flags_str = ", ".join(f"Update({k})={v}" for k, v in update_flags.items())

        return (
            f"<{self.__class__.__name__}: " f"{', '.join(params)}, " f"[{flags_str}]>"
        )

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
        diag = self.discrete_transition_diag
        transition_matrix = jnp.diag(diag)

        if self.n_discrete_states == 1:
            # Single state: transition matrix is just [[1.0]]
            self.discrete_transition_matrix = jnp.array([[1.0]])
            return

        # Compute off-diagonal values for each row
        off_diag = (1.0 - diag) / (self.n_discrete_states - 1.0)

        # Add off-diagonal elements, avoiding the diagonal
        transition_matrix = transition_matrix + jnp.ones(
            (self.n_discrete_states, self.n_discrete_states)
        ) * off_diag[:, None] - jnp.diag(off_diag)

        # Ensure rows sum to 1 (handle potential floating point issues)
        self.discrete_transition_matrix = transition_matrix / jnp.sum(
            transition_matrix, axis=1, keepdims=True
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
        """Performs the Expectation (E) step of the EM algorithm.

        Runs the switching Kalman filter and smoother to compute the
        Expected Sufficient Statistics (ESS) and the marginal log-likelihood.

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
            last_cond_cont_mean,
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

        (
            _,  # smoother_mean (marginal)
            _,  # smoother_cov (marginal)
            self.smoother_discrete_state_prob,
            self.smoother_joint_discrete_state_prob,
            _,  # smoother_cross_cov (marginal)
            self.smoother_state_cond_mean,
            self.smoother_state_cond_cov,
            self.smoother_pair_cond_cross_cov,
            self.smoother_pair_cond_means,  # E[x_t | S_t=i, S_{t+1}=j]
        ) = switching_kalman_smoother(
            filter_mean=filter_mean,
            filter_cov=filter_cov,
            filter_discrete_state_prob=filter_discrete_state_prob,
            last_filter_conditional_cont_mean=last_cond_cont_mean,
            process_cov=self.process_cov,
            continuous_transition_matrix=self.continuous_transition_matrix,
            discrete_state_transition_matrix=self.discrete_transition_matrix,
        )

        return marginal_log_likelihood

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
        )

        # Update parameters based on flags
        if self.update_continuous_transition_matrix:
            self.continuous_transition_matrix = A
        if self.update_measurement_matrix:
            self.measurement_matrix = H
        if self.update_process_cov:
            self.process_cov = Q
        if self.update_measurement_cov:
            self.measurement_cov = R
        if self.update_init_mean:
            self.init_mean = m0
        if self.update_init_cov:
            self.init_cov = P0
        if self.update_discrete_transition_matrix:
            self.discrete_transition_matrix = Z
        # Always update init_prob based on the first smoother probability
        self.init_discrete_state_prob = pi0

    def fit(
        self,
        observations: ArrayLike,
        key: Array,
        max_iter: int = 100,
        tolerance: float = 1e-4,
    ) -> list[jax.Array]:
        """Fits the model to observations using the EM algorithm.

        Iteratively performs E-steps and M-steps until convergence or
        the maximum number of iterations is reached.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        key : Array
            JAX random number generator key for initialization.
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tolerance : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.

        Returns
        -------
        log_likelihoods : list[jax.Array]
            A list of marginal log-likelihoods at each iteration (scalar arrays).
        """
        observations = jnp.asarray(observations)
        self._initialize_parameters(key)
        log_likelihoods: list[jax.Array] = []
        previous_log_likelihood: jax.Array = jnp.array(-jnp.inf)

        for iteration in range(max_iter):
            # E-step
            current_log_likelihood = self._e_step(observations)
            log_likelihoods.append(current_log_likelihood)

            # Check convergence
            is_converged, is_increasing = check_converged(
                current_log_likelihood, previous_log_likelihood, tolerance
            )

            if not is_increasing:
                logger.warning(
                    f"Log-likelihood decreased at iteration {iteration + 1}!"
                )

            if is_converged:
                logger.info(f"Converged after {iteration + 1} iterations.")
                break

            # M-step
            self._m_step(observations)

            # Projection step
            self._project_parameters()

            logger.info(
                f"Iteration {iteration + 1}/{max_iter}\t"
                f"Log-Likelihood: {current_log_likelihood:.4f}\t"
                f"Change: {(current_log_likelihood - previous_log_likelihood):.4f}"
            )
            previous_log_likelihood = current_log_likelihood

        if len(log_likelihoods) == max_iter:
            logger.warning("Reached maximum iterations without converging.")

        return log_likelihoods


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
    auto_regressive_coef : jax.Array, shape (n_oscillators,)
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
        auto_regressive_coef: jax.Array,
        process_variance: jax.Array,
        measurement_variance: float,
        **kwargs,
    ):
        super().__init__(
            n_oscillators, n_discrete_states, n_sources, sampling_freq, **kwargs
        )
        if freqs.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: freqs {freqs.shape} vs n_oscillators {n_oscillators}"
            )
        self.freqs = freqs

        if auto_regressive_coef.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: auto_regressive_coef {auto_regressive_coef.shape} vs n_oscillators {n_oscillators}"
            )
        self.auto_regressive_coef = auto_regressive_coef

        if process_variance.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: process_variance {process_variance.shape} vs n_oscillators {n_oscillators}"
            )
        self.process_variance = process_variance

        if not isinstance(measurement_variance, (float, int)):
            raise ValueError(
                "measurement_variance must be a scalar (float or int)."
                f" Got {type(measurement_variance)}."
            )
        if measurement_variance <= 0:
            raise ValueError(
                "measurement_variance must be positive. " f"Got {measurement_variance}."
            )
        self.measurement_variance = measurement_variance

        if sampling_freq <= 0:
            raise ValueError("sampling_freq must be positive. " f"Got {sampling_freq}.")

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
            dtype=jnp.float32,
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
            auto_regressive_coef=self.auto_regressive_coef,
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
        key: Array,
        max_iter: int = 100,
        tolerance: float = 1e-4,
    ) -> list[jax.Array]:
        """Fits the model to observations using the EM algorithm.

        Iteratively performs E-steps and M-steps until convergence or
        the maximum number of iterations is reached.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        key : Array
            JAX random number generator key for initialization.
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tolerance : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.

        Returns
        -------
        log_likelihoods : list[jax.Array]
            A list of marginal log-likelihoods at each iteration (scalar arrays).
        """
        observations = jnp.asarray(observations)
        if observations.shape[1] != self.n_sources:
            raise ValueError(
                f"observations must have {self.n_sources} sources, "
                f"got {observations.shape[1]}."
            )
        return super().fit(observations, key, max_iter, tolerance)


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
    auto_regressive_coef : jax.Array, shape (n_oscillators,)
        Damping coefficients for each oscillator.
    process_variance : jax.Array, shape (n_oscillators, n_discrete_states)
        Process noise variance for each oscillator and state.
    measurement_variance : float
        Variance of the measurement noise.
    phase_difference : jax.Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial phase differences for noise correlation.
    coupling_strength : jax.Array, shape (n_oscillators, n_oscillators, n_discrete_states)
        Initial coupling strengths for noise correlation.
    """

    def __init__(
        self,
        n_oscillators: int,
        n_discrete_states: int,
        sampling_freq: float,
        freqs: jax.Array,
        auto_regressive_coef: jax.Array,
        process_variance: jax.Array,
        measurement_variance: float,
        phase_difference: jax.Array,
        coupling_strength: jax.Array,
        **kwargs,
    ):
        # n_sources must equal n_oscillators for CNM/DIM
        super().__init__(
            n_oscillators,
            n_discrete_states,
            n_oscillators,
            sampling_freq,
            **kwargs,
        )
        if n_oscillators != self.n_sources:
            raise ValueError("For CNM, n_sources must equal n_oscillators.")
        if freqs.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: freqs {freqs.shape} vs n_oscillators {n_oscillators}"
            )
        if auto_regressive_coef.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: auto_regressive_coef {auto_regressive_coef.shape} vs n_oscillators {n_oscillators}"
            )
        if process_variance.shape != (n_oscillators, n_discrete_states):
            raise ValueError(
                "process_variance must have shape (n_oscillators, n_discrete_states)."
                f" Got {process_variance.shape}."
            )
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
        if not isinstance(measurement_variance, (float, int)):
            raise ValueError(
                "measurement_variance must be a scalar (float or int)."
                f" Got {type(measurement_variance)}."
            )
        if measurement_variance <= 0:
            raise ValueError(
                "measurement_variance must be positive. " f"Got {measurement_variance}."
            )
        if sampling_freq <= 0:
            raise ValueError("sampling_freq must be positive. " f"Got {sampling_freq}.")
        self.freqs = freqs
        self.auto_regressive_coef = auto_regressive_coef
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
            auto_regressive_coef=self.auto_regressive_coef,
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
        # Needs shape (n_cont_states, n_cont_states, n_discrete_states)

    def _project_parameters(self):
        """Projects each Q_j to preserve its oscillatory/coupling structure.

        Uses `project_matrix_blockwise` to project each 2x2 block to its
        closest (scaled) rotation matrix.
        """
        self.process_cov = jnp.stack(
            [
                project_matrix_blockwise(self.process_cov[..., j])
                for j in range(self.n_discrete_states)
            ],
            axis=-1,
        )

    def fit(
        self,
        observations: ArrayLike,
        key: Array,
        max_iter: int = 100,
        tolerance: float = 1e-4,
    ) -> list[jax.Array]:
        """Fits the model to observations using the EM algorithm.

        Iteratively performs E-steps and M-steps until convergence or
        the maximum number of iterations is reached.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        key : Array
            JAX random number generator key for initialization.
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tolerance : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.

        Returns
        -------
        log_likelihoods : list[jax.Array]
            A list of marginal log-likelihoods at each iteration (scalar arrays).
        """
        observations = jnp.asarray(observations)
        if observations.shape[1] != self.n_sources:
            raise ValueError(
                f"observations must have {self.n_sources} sources, "
                f"got {observations.shape[1]}."
            )
        return super().fit(observations, key, max_iter, tolerance)


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
    auto_regressive_coef : jax.Array, shape (n_oscillators,)
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
    """

    def __init__(
        self,
        n_oscillators: int,
        n_discrete_states: int,
        sampling_freq: float,
        freqs: jax.Array,
        auto_regressive_coef: jax.Array,
        process_variance: jax.Array,
        measurement_variance: float,
        phase_difference: jax.Array,
        coupling_strength: jax.Array,
        use_reparameterized_mstep: bool = False,
        **kwargs,
    ):
        super().__init__(
            n_oscillators,
            n_discrete_states,
            n_oscillators,
            sampling_freq,
            **kwargs,
        )
        if n_oscillators != self.n_sources:
            raise ValueError("For DIM, n_sources must equal n_oscillators.")

        if freqs.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: freqs {freqs.shape} vs n_oscillators {n_oscillators}"
            )
        self.freqs = freqs
        if auto_regressive_coef.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: auto_regressive_coef {auto_regressive_coef.shape} vs n_oscillators {n_oscillators}"
            )
        self.auto_regressive_coef = auto_regressive_coef
        if process_variance.shape != (n_oscillators,):
            raise ValueError(
                f"Shape mismatch: process_variance {process_variance.shape} vs n_oscillators {n_oscillators}"
            )
        self.process_variance = process_variance

        if not isinstance(measurement_variance, (float, int)):
            raise ValueError(
                "measurement_variance must be a scalar (float or int)."
                f" Got {type(measurement_variance)}."
            )
        if measurement_variance <= 0:
            raise ValueError(
                "measurement_variance must be positive. " f"Got {measurement_variance}."
            )
        self.measurement_variance = measurement_variance

        if phase_difference.shape != (n_oscillators, n_oscillators, n_discrete_states):
            raise ValueError(
                "phase_difference must have shape (n_oscillators, n_oscillators, n_discrete_states)."
                f" Got {phase_difference.shape}."
            )
        self.phase_difference = phase_difference

        if coupling_strength.shape != (n_oscillators, n_oscillators, n_discrete_states):
            raise ValueError(
                "coupling_strength must have shape (n_oscillators, n_oscillators, n_discrete_states)."
                f" Got {coupling_strength.shape}."
            )
        self.coupling_strength = coupling_strength

        # DIM specific M-step update flags
        self.update_continuous_transition_matrix = True
        self.update_measurement_matrix = False  # H is fixed in DIM
        self.update_process_cov = False  # Q is fixed in DIM

        # Reparameterized M-step option
        self.use_reparameterized_mstep = use_reparameterized_mstep
        # Store current oscillator params for warm-starting optimizer
        self._current_osc_params: Optional[list[dict]] = None

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
        self.continuous_transition_matrix = jnp.stack(
            [
                construct_directed_influence_transition_matrix(
                    freqs=self.freqs,
                    damping_coeffs=self.auto_regressive_coef,
                    coupling_strengths=self.coupling_strength[..., state_ind],
                    phase_diffs=self.phase_difference[..., state_ind],
                    sampling_freq=self.sampling_freq,
                )
                for state_ind in range(self.n_discrete_states)
            ],
            axis=2,
        )

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
        )

        # Update non-A parameters based on flags (same as standard M-step)
        if self.update_measurement_matrix:
            self.measurement_matrix = H
        if self.update_process_cov:
            self.process_cov = Q
        if self.update_measurement_cov:
            self.measurement_cov = R
        if self.update_init_mean:
            self.init_mean = m0
        if self.update_init_cov:
            self.init_cov = P0
        if self.update_discrete_transition_matrix:
            self.discrete_transition_matrix = Z
        self.init_discrete_state_prob = pi0

        # Now optimize A in parameter space for each discrete state
        if self.update_continuous_transition_matrix:
            # Compute sufficient statistics for transition matrix
            gamma1, beta = compute_transition_sufficient_stats(
                state_cond_smoother_means=self.smoother_state_cond_mean,
                state_cond_smoother_covs=self.smoother_state_cond_cov,
                smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
                pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
                pair_cond_smoother_means=self.smoother_pair_cond_means,
            )

            # Initialize current_osc_params if not already done
            if self._current_osc_params is None:
                self._current_osc_params = []
                for j in range(self.n_discrete_states):
                    params = extract_dim_params_from_matrix(
                        self.continuous_transition_matrix[..., j],
                        self.sampling_freq,
                        self.n_oscillators,
                    )
                    self._current_osc_params.append(params)

            # Optimize for each discrete state
            A_list = []
            for j in range(self.n_discrete_states):
                # Get initial params (warm start from previous iteration)
                init_params = self._current_osc_params[j]

                # Optimize
                opt_params = optimize_dim_transition_params(
                    gamma1=gamma1[..., j],
                    beta=beta[..., j],
                    init_params=init_params,
                    sampling_freq=self.sampling_freq,
                )

                # Store for warm start
                self._current_osc_params[j] = opt_params

                # Reconstruct A from optimized parameters
                A_j = construct_directed_influence_transition_matrix(
                    freqs=opt_params["freq"],
                    damping_coeffs=opt_params["damping"],
                    coupling_strengths=opt_params["coupling_strength"],
                    phase_diffs=opt_params["phase_diff"],
                    sampling_freq=self.sampling_freq,
                )
                A_list.append(A_j)

            self.continuous_transition_matrix = jnp.stack(A_list, axis=-1)

            # Update public oscillator attributes to reflect optimized values
            self._update_public_oscillator_params()

    def _update_public_oscillator_params(self) -> None:
        """Update public oscillator attributes from optimized parameters.

        After the reparameterized M-step, syncs the private _current_osc_params
        to the public attributes (freqs, auto_regressive_coef, coupling_strength,
        phase_difference) so downstream code can inspect the learned network.
        """
        if self._current_osc_params is None:
            return

        # Stack per-state values into arrays with state as last dimension
        # freqs and damping: shape (n_osc,) averaged across states
        # (they should be similar across states, so we average)
        freqs_list = [p["freq"] for p in self._current_osc_params]
        damping_list = [p["damping"] for p in self._current_osc_params]

        # For freqs and damping, take mean across states (they're per-oscillator params)
        self.freqs = jnp.mean(jnp.stack(freqs_list, axis=-1), axis=-1)
        self.auto_regressive_coef = jnp.mean(jnp.stack(damping_list, axis=-1), axis=-1)

        # coupling_strength and phase_diff: shape (n_osc, n_osc, n_discrete_states)
        coupling_list = [p["coupling_strength"] for p in self._current_osc_params]
        phase_list = [p["phase_diff"] for p in self._current_osc_params]

        self.coupling_strength = jnp.stack(coupling_list, axis=-1)
        self.phase_difference = jnp.stack(phase_list, axis=-1)

    def _project_parameters(self):
        """Projects parameters to valid space.

        With reparameterized M-step, A is already valid by construction,
        so no projection is needed. With standard M-step, projects each A_j
        to preserve oscillatory/coupling structure.
        """
        if not self.use_reparameterized_mstep:
            # Only project if using standard M-step
            self.continuous_transition_matrix = jnp.stack(
                [
                    project_coupled_transition_matrix(
                        self.continuous_transition_matrix[..., j]
                    )
                    for j in range(self.n_discrete_states)
                ],
                axis=-1,
            )

    def fit(
        self,
        observations: ArrayLike,
        key: Array,
        max_iter: int = 100,
        tolerance: float = 1e-4,
    ) -> list[jax.Array]:
        """Fits the model to observations using the EM algorithm.

        Iteratively performs E-steps and M-steps until convergence or
        the maximum number of iterations is reached.

        Parameters
        ----------
        observations : ArrayLike, shape (n_time, n_sources)
            The sequence of observations.
        key : Array
            JAX random number generator key for initialization.
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tolerance : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.

        Returns
        -------
        log_likelihoods : list[jax.Array]
            A list of marginal log-likelihoods at each iteration (scalar arrays).
        """
        observations = jnp.asarray(observations)
        if observations.shape[1] != self.n_sources:
            raise ValueError(
                f"observations must have {self.n_sources} sources, "
                f"got {observations.shape[1]}."
            )
        return super().fit(observations, key, max_iter, tolerance)
