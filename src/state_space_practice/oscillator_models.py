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

from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
    construct_correlated_noise_measurement_matrix,
    construct_correlated_noise_process_covariance,
    construct_directed_influence_measurement_matrix,
    construct_directed_influence_transition_matrix,
    get_block_slice,
    project_coupled_transition_matrix,
    project_matrix_blockwise,
)
from state_space_practice.switching_kalman import (
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
        transition_matrix = diag * jnp.identity(self.n_discrete_states)

        if self.n_discrete_states == 1:
            off_diag = 0.0  # Or 1.0, but 0.0 makes more sense
        else:
            off_diag = (1.0 - diag) / (self.n_discrete_states - 1.0)

        # Add off-diagonal elements, avoiding the diagonal
        transition_matrix += jnp.ones(
            (self.n_discrete_states, self.n_discrete_states)
        ) * off_diag[:, None] - off_diag[:, None] * jnp.identity(self.n_discrete_states)

        # Ensure rows sum to 1 (handle potential floating point issues)
        self.discrete_transition_matrix = transition_matrix / jnp.sum(
            transition_matrix, axis=1, keepdims=True
        )

    def _initialize_continuous_state(self, key: jax.random.PRNGKey) -> None:
        """Initializes the initial continuous state mean (m0) and covariance (P0).

        m0 is drawn from a multivariate normal distribution. P0 is set to identity
        and stacked for each discrete state.

        Parameters
        ----------
        key : jax.random.PRNGKey
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
    def _initialize_measurement_matrix(self, key: Optional[jax.random.PRNGKey] = None):
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

    def _initialize_parameters(self, key: jax.random.PRNGKey) -> None:
        """Initializes all model parameters by calling specific methods.

        Parameters
        ----------
        key : jax.random.PRNGKey
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

    def _e_step(self, observations: jnp.ndarray) -> float:
        """Performs the Expectation (E) step of the EM algorithm.

        Runs the switching Kalman filter and smoother to compute the
        Expected Sufficient Statistics (ESS) and the marginal log-likelihood.

        Parameters
        ----------
        observations : jnp.ndarray, shape (n_time, n_sources)
            The sequence of observations.

        Returns
        -------
        marginal_log_likelihood : float
            The log-likelihood of the observations given the current parameters.
        """
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
            obs=observations,
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

    def _m_step(self, observations: jnp.ndarray) -> None:
        """Performs the Maximization (M) step of the EM algorithm.

        Updates the model parameters using the Expected Sufficient Statistics
        computed in the E-step.

        Parameters
        ----------
        observations : jnp.ndarray, shape (n_time, n_sources)
            The sequence of observations.
        """
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
            obs=observations,
            state_cond_smoother_means=self.smoother_state_cond_mean,
            state_cond_smoother_covs=self.smoother_state_cond_cov,
            smoother_discrete_state_prob=self.smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,
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
        observations: jnp.ndarray,
        key: jax.random.PRNGKey,
        max_iter: int = 100,
        tolerance: float = 1e-4,
    ) -> list[float]:
        """Fits the model to observations using the EM algorithm.

        Iteratively performs E-steps and M-steps until convergence or
        the maximum number of iterations is reached.

        Parameters
        ----------
        observations : jnp.ndarray, shape (n_time, n_sources)
            The sequence of observations.
        key : jax.random.PRNGKey
            JAX random number generator key for initialization.
        max_iter : int, optional
            Maximum number of EM iterations, by default 100.
        tolerance : float, optional
            Convergence tolerance for log-likelihood, by default 1e-4.

        Returns
        -------
        log_likelihoods : list[float]
            A list of marginal log-likelihoods at each iteration.
        """
        self._initialize_parameters(key)
        log_likelihoods = []
        previous_log_likelihood = -jnp.inf

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

            previous_log_likelihood = current_log_likelihood
            logger.info(
                f"Iteration {iteration + 1}/{max_iter}, "
                f"Log-Likelihood: {current_log_likelihood:.4f}"
            )

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
        self.freqs = freqs
        self.auto_regressive_coef = auto_regressive_coef
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # COM specific M-step update flags
        self.update_continuous_transition_matrix = False
        self.update_process_cov = False
        self.update_measurement_matrix = True

    def _initialize_measurement_matrix(self, key: Optional[jax.random.PRNGKey] = None):
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

    def get_oscillator_influence_on_node(self) -> jnp.ndarray:
        """Calculates the influence of each oscillator on each source.

        This is computed as the L2 norm of the (2x1) block in H
        corresponding to an oscillator-source pair.

        Returns
        -------
        jnp.ndarray, shape (n_sources, n_oscillators, n_discrete_states)
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
    ) -> jnp.ndarray:
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
        jnp.ndarray, shape (n_discrete_states,)
            The phase difference in radians for each discrete state.
        """
        _, col = get_block_slice(oscillator_ind, oscillator_ind)
        coef1 = self.measurement_matrix[node1_ind, col, :]  # Shape (2, M)
        coef2 = self.measurement_matrix[node2_ind, col, :]  # Shape (2, M)

        phase1 = jnp.arctan2(coef1[1], coef1[0])  # Shape (M,)
        phase2 = jnp.arctan2(coef2[1], coef2[0])  # Shape (M,)

        return phase1 - phase2


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

    def _initialize_measurement_matrix(self, key: Optional[jax.random.PRNGKey] = None):
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
        super().__init__(
            n_oscillators,
            n_discrete_states,
            n_oscillators,
            sampling_freq,
            **kwargs,
        )
        if n_oscillators != self.n_sources:
            raise ValueError("For DIM, n_sources must equal n_oscillators.")
        self.freqs = freqs
        self.auto_regressive_coef = auto_regressive_coef
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.phase_difference = phase_difference
        self.coupling_strength = coupling_strength

        # DIM specific M-step update flags
        self.update_continuous_transition_matrix = True
        self.update_measurement_matrix = False  # H is fixed in DIM
        self.update_process_cov = False  # Q is fixed in DIM

    def _initialize_measurement_matrix(self, key: Optional[jax.random.PRNGKey] = None):
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

    def _project_parameters(self):
        """Projects each A_j to preserve its oscillatory/coupling structure.

        Uses `project_coupled_transition_matrix` which applies specific
        projections to diagonal and off-diagonal blocks.
        """
        self.continuous_transition_matrix = jnp.stack(
            [
                project_coupled_transition_matrix(
                    self.continuous_transition_matrix[..., j]
                )
                for j in range(self.n_discrete_states)
            ],
            axis=-1,
        )
