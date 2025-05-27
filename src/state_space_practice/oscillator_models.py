"""Implementation of the switching oscillator models

References
----------
1. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2022). Switching Functional Network Models of Oscillatory Brain Dynamics. In 2022 56th Asilomar Conference on Signals, Systems, and Computers (IEEE), pp. 607–612. https://doi.org/10.1109/IEEECONF56349.2022.10052077.
2. Hsin, W.-C., Eden, U.T., and Stephen, E.P. (2024). Switching Models of Oscillatory Networks Greatly Improve Inference of Dynamic Functional Connectivity. Preprint at arXiv.
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
)
from state_space_practice.switching_kalman import (
    switching_kalman_filter,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
)
from state_space_practice.utils import check_converged


class BaseModel:

    def __init__(
        self,
    ):
        pass

    # init_mean: jnp.ndarray,
    # init_cov: jnp.ndarray,
    # init_discrete_state_prob: jnp.ndarray,
    # obs: jnp.ndarray,
    # discrete_transition_matrix: jnp.ndarray,
    # continuous_transition_matrix: jnp.ndarray,
    # process_cov: jnp.ndarray,
    # measurement_matrix: jnp.ndarray,
    # measurement_cov: jnp.ndarray,

    def _initialize_discrete_state_prob(self):
        self.init_discrete_state_prob = (
            jnp.ones(self.n_discrete_states) / self.n_discrete_states
        )

    def _initialize_discrete_transition_matrix(self):
        diag = self.discrete_transition_diag
        transition_matrix = diag * jnp.identity(self.n_discrete_states)
        if self.n_discrete_states == 1:
            off_diag = 1.0
        else:
            off_diag = ((1.0 - diag) / (self.n_discrete_states - 1.0))[:, None]

        transition_matrix += jnp.ones(
            (self.n_discrete_states, self.n_discrete_states)
        ) * off_diag - off_diag * jnp.identity(self.n_discrete_states)

        self.discrete_transition_matrix = transition_matrix

    def _initialize_continuous_state(self, key: jax.random.PRNGKey):
        self.continuous_state_mean = jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros((2 * self.n_oscillators,)),
            cov=jnp.identity(2 * self.n_oscillators),
        )

    def _e_step(self, observations: jnp.ndarray):
        filter_mean, filter_cov, filter_discrete_state_prob, last_cond_cont_mean = (
            switching_kalman_filter(
                init_mean=self.init_mean,
                init_cov=self.init_cov,
                init_discrete_state_prob=self.init_discrete_state_prob,
                obs=observations,
                discrete_transition_matrix=self.discrete_transition_matrix,
                continuous_transition_matrix=self.continuous_transition_matrix,
                process_cov=self.process_cov,
                measurement_matrix=self.measurement_matrix,
                measurement_cov=self.measurement_cov,
            )
        )
        (
            smoother_means,
            smoother_covs,
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            smoother_cross_cov,
        ) = switching_kalman_smoother(
            filter_mean,
            filter_cov,
            filter_discrete_state_prob,
            last_cond_cont_mean,
            process_cov=self.process_cov,
            continuous_transition_matrix=self.continuous_transition_matrix,
            discrete_state_transition_matrix=self.discrete_transition_matrix,
        )

        return (
            smoother_means,
            smoother_covs,
            smoother_discrete_state_prob,
            smoother_joint_discrete_state_prob,
            smoother_cross_cov,
        )

    def _m_step(self, observations: jnp.ndarray):
        (
            self.discrete_transition_matrix,
            self.continuous_transition_matrix,
            self.measurement_matrix,
            self.process_cov,
            self.measurement_cov,
            self.init_mean,
            self.init_cov,
        ) = switching_kalman_maximization_step(
            obs=observations,
            smoother_mean=self.smoother_mean,
            smoother_cov=self.smoother_cov,
            smoother_cross_cov=self.smoother_cross_cov,
            weights=self.smoother_discrete_state_prob,
            update_discrete_transition_matrix=self.update_discrete_transition_matrix,
            update_continuous_transition_matrix=self.update_continuous_transition_matrix,
            update_measurement_matrix=self.update_measurement_matrix,
            update_process_cov=self.update_process_cov,
            update_measurement_cov=self.update_measurement_cov,
            update_init_mean=self.update_init_mean,
            update_init_cov=self.update_init_cov,
        )


class CommonOscillatorModel(BaseModel):
    """Common Oscillator Model (COM)

    The measurement matrix depends on the discrete latent state.
    """

    def __init__(self):
        super().__init__()

    def _initialize__measurement_matrix(self, key: jax.random.PRNGKey):
        self.measurement_matrix = jax.random.uniform(
            key,
            (self.n_sources, 2 * self.n_oscillators, self.n_discrete_states),
            dtype=jnp.float32,
            minval=0.0,
            maxval=0.1,
        )

    def _initialize_measurement_covariance(self):
        measurement_cov = jnp.identity(self.n_sources) * self.measurement_variance
        self.measurement_cov = jnp.stack(
            [measurement_cov] * self.n_discrete_states, axis=2
        )

    def _initialize_continuous_transition_matrix(self):
        transition_matrix = construct_common_oscillator_transition_matrix(
            freqs=self.freqs,
            auto_regressive_coef=self.auto_regressive_coef,
            sampling_freq=self.sampling_freq,
        )
        self.continuous_transition_matrix = jnp.stack(
            [transition_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_process_covariance(self):
        process_cov = construct_common_oscillator_process_covariance(
            variance=self.process_variance,
        )
        self.process_cov = jnp.stack([process_cov] * self.n_discrete_states, axis=2)

    def get_oscillator_influence_on_node(self) -> jnp.ndarray:
        """Get the influence of an oscillator on a node

        Returns
        -------
        influence : jnp.ndarray, shape (n_sources, n_oscillators, n_discrete_states)
            The influence of the oscillator on the node for discrete states

        """
        n_sources, n_oscillators_x2, n_discrete_states = self.measurement_matrix.shape
        n_oscillators = n_oscillators_x2 // 2

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
        """Get the phase difference between nodes for a given oscillator

        Returns
        -------
        phase_difference : float
            The phase difference node1_ind and node2_ind for oscillator_ind

        """
        _, col = get_block_slice(oscillator_ind, oscillator_ind)
        coef1 = self.measurement_matrix[node1_ind, col, :]
        coef2 = self.measurement_matrix[node2_ind, col, :]

        phase1 = jnp.arctan2(coef1[1], coef1[0])
        phase2 = jnp.arctan2(coef2[1], coef2[0])

        return phase1 - phase2


class CorrelatedNoiseModel(BaseModel):
    """Correlated Noise Model (CNM)

    The process noise covariance depends on the discrete latent state.
    The number of nodes is equal to the number of latent oscillators.
    """

    def __init__(self):
        super().__init__()

    def _initialize_measurement_matrix(self):
        measurement_matrix = construct_correlated_noise_measurement_matrix(
            self.n_sources,
        )
        self.measurement_matrix = jnp.stack(
            [measurement_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_measurement_covariance(self):
        measurement_cov = jnp.identity(self.n_sources) * self.measurement_variance
        self.measurement_cov = jnp.stack(
            [measurement_cov] * self.n_discrete_states, axis=2
        )

    def _initialize_continuous_transition_matrix(self):
        transition_matrix = construct_common_oscillator_transition_matrix(
            freqs=self.freqs,
            auto_regressive_coef=self.auto_regressive_coef,
            sampling_freq=self.sampling_freq,
        )
        self.continuous_transition_matrix = jnp.stack(
            [transition_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_process_covariance(self):
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


class DirectedInfluenceModel(BaseModel):
    """Directed Influence Model (DIM)

    The continuous transition matrix depends on the discrete latent state.
    The number of nodes is equal to the number of latent oscillators.
    """

    def __init__(self):
        super().__init__()

    def _initialize_measurement_matrix(self):
        measurement_matrix = construct_directed_influence_measurement_matrix(
            self.n_sources,
        )
        self.measurement_matrix = jnp.stack(
            [measurement_matrix] * self.n_discrete_states, axis=2
        )

    def _initialize_continuous_transition_matrix(self):
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
