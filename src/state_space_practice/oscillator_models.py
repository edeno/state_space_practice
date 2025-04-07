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

from state_space_practice.switching_kalman import (
    switching_kalman_filter,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
)
from state_space_practice.utils import check_converged

IDENTITY_2x2 = jnp.identity(2)
ZEROS_2x2 = jnp.zeros((2, 2))


def get_block_slice(from_oscillator, to_oscillator) -> tuple:
    """Get the indices for a 2x2 block in a 2n_oscillator matrix

    Parameters
    ----------
    from_oscillator : int
    to_oscillator : int

    Returns
    -------
    rows : slice
        The slice for the rows of the block
    cols : slice
        The slice for the columns of the block

    """
    row_slice = slice(2 * from_oscillator, 2 * (from_oscillator + 1))
    col_slice = slice(2 * to_oscillator, 2 * (to_oscillator + 1))
    return row_slice, col_slice


def _get_rotation_matrix(rotation_frequency: float) -> jnp.ndarray:
    """Get the rotation matrix for a given frequency

    The rotation matrix is a 2x2 matrix that rotates a vector.

    Parameters
    ----------
    rotation_frequency : float
        The frequency in radians

    Returns
    -------
    rotation_matrix : jnp.ndarray
        The rotation matrix
    """
    cos_rot = jnp.cos(rotation_frequency)
    sin_rot = jnp.sin(rotation_frequency)

    return jnp.ndarray(
        [
            [cos_rot, -sin_rot],
            [sin_rot, cos_rot],
        ],
    )


def _compute_intrinsic_oscillation_block(
    oscillation_freq: float, auto_regressive_coef: float, sampling_freq: float = 1
) -> jnp.ndarray:
    """Compute the rotation matrix for a given frequency and auto-regressive coefficient

    Parameters
    ----------
    oscillation_freq : float
        Oscillation frequency in Hz
    auto_regressive_coef : float
        Controls the damping of the oscillation. A value of 1 corresponds to a
        pure oscillation, while a value of 0 corresponds to no oscillation.
    sampling_freq : float, optional
        Samples per second, by default 1

    Returns
    -------
    jnp.ndarray, shape (2, 2)
        The transition matrix at the specified frequency

    Raises
    ------
    ValueError
        If the auto_regressive_coef is not between 0 and 1

    """
    if jnp.logical_or(auto_regressive_coef > 1, auto_regressive_coef < 0):
        raise ValueError("Auto-regressive coefficient must be between 0 and 1")

    return auto_regressive_coef * _get_rotation_matrix(
        2 * jnp.pi * oscillation_freq / sampling_freq
    )


def _compute_coupled_oscillator_block(
    freq: float,
    auto_regressive_coef: float,
    sum_incoming_coupling_strength: float,
    sampling_freq: float = 1,
) -> jnp.ndarray:
    """Compute the diagonal block of the transition matrix for the coupled model

    Parameters
    ----------
    freq : float
        Oscillation frequency in Hz
    auto_regressive_coef : float
        Controls the damping of the oscillation. A value of 1 corresponds to a
        pure oscillation, while a value of 0 corresponds to no oscillation.
    sum_incoming_coupling_strength : float
        Sum of all incoming coupling strengths to this oscillator from all other
        oscillators.
    sampling_freq : float, optional
        Samples per second, by default 1

    Returns
    -------
    diagonal_block : jnp.ndarray, shape (2, 2)
        The transition matrix at the specified frequency

    """
    return (
        _compute_intrinsic_oscillation_block(freq, auto_regressive_coef, sampling_freq)
        - sum_incoming_coupling_strength * IDENTITY_2x2
    )


def _compute_coupling_transition_block(
    phase_difference: float,
    coupling_strength: float,
) -> jnp.ndarray:
    """Compute the off-diagonal block transition matrix for a coupling model

    Parameters
    ----------
    phase_difference : float
        Phase difference between the two oscillators
    coupling_strength : float
        Strength of the coupling between the two oscillators

    Returns
    -------
    jnp.ndarray, shape (2, 2)
        The transition matrix between the two oscillators

    """
    if jnp.isclose(coupling_strength, 0.0):
        return ZEROS_2x2
    else:
        return coupling_strength * _get_rotation_matrix(phase_difference)


def construct_common_oscillator_transition_matrix(
    freqs: jnp.ndarray,
    auto_regressive_coef: jnp.ndarray,
    sampling_freq: float = 1.0,
) -> jnp.ndarray:
    """Constructs the transition matrix for a common oscillator model.

    The transition matrix is a block diagonal matrix with each block
    corresponding to a single oscillator.

    Also used for the correlated noise model.

    Parameters
    ----------
    freqs : jnp.ndarray, shape (n_oscillators,)
        Array of oscillation frequencies (fk) for each oscillator.
    auto_regressive_coef : jnp.ndarray, shape (n_oscillators,)
        Array of auto-regressive coefficients (alpha_j^k) for each oscillator k.
    sampling_freq : float, optional
        Sampling frequency (Fs) in Hz, by default 1.0.

    Returns
    -------
    transition_matrix : jnp.ndarray, shape (2 * n_oscillators, 2 * n_oscillators)

    Raises
    ------
    ValueError
        If input array dimensions do not match the inferred number of oscillators.
    """
    n_oscillators = freqs.shape[0]
    if not auto_regressive_coef.shape == (n_oscillators,):
        raise ValueError(
            "auto_regressive_coef must be a 1D array of shape (n_oscillators,)"
        )
    diag_blocks = [
        _compute_intrinsic_oscillation_block(
            freq=freqs[k],
            auto_regressive_coef=auto_regressive_coef[k],
            sampling_freq=sampling_freq,
        )
        for k in range(n_oscillators)
    ]

    return jax.scipy.linalg.block_diag(*diag_blocks)


def construct_common_oscillator_process_covariance(
    variance: jnp.ndarray,
) -> jnp.ndarray:
    pass
    """Constructs the process covariance matrix for a common oscillator model.

    The process covariance matrix (\Sigma) is a block diagonal matrix with each block
    corresponding to a single oscillator.

    Parameters
    ----------
    variance : jnp.ndarray, shape (n_oscillators,)
        Array of process noise variances (sigma_j) for each oscillator.

    Returns
    -------
    process_covariance : jnp.ndarray, shape (2 * n_oscillators, 2 * n_oscillators)
        The process covariance matrix for the common oscillator model.

    """
    n_oscillators = variance.shape[0]
    diag_blocks = [variance[k] * IDENTITY_2x2 for k in range(n_oscillators)]

    return jax.scipy.linalg.block_diag(*diag_blocks)


def construct_correlated_noise_process_covariance(
    variance: jnp.ndarray,
    phase_difference: jnp.ndarray,
    coupling_strength: jnp.ndarray,
) -> jnp.ndarray:
    """

    Parameters
    ----------
    variance : jnp.ndarray, shape (n_oscillators,)
        Array of process noise variances (sigma_j) for each oscillator.
    phase_difference : jnp.ndarray, shape (n_oscillators, n_oscillators)
        Matrix where phase_diffs[n1, n2] is the phase difference for
        coupling from oscillator n2 to oscillator n1 (phi_j^{n1,n2}).
    coupling_strength : jnp.ndarray, shape (n_oscillators, n_oscillators)
        Matrix where coupling_strengths[n1, n2] is the coupling strength

    Returns
    -------
    process_covariance : jnp.ndarray, shape (2 * n_oscillators, 2 * n_oscillators)
        The process covariance matrix for the correlated noise model.
    """
    n_oscillators = variance.shape[0]

    return jnp.block(
        [
            [
                (
                    variance[from_oscillator] * IDENTITY_2x2
                    if from_oscillator == to_oscillator
                    else _compute_coupling_transition_block(
                        phase_difference[from_oscillator, to_oscillator],
                        coupling_strength[from_oscillator, to_oscillator],
                    )
                )
                for to_oscillator in range(n_oscillators)
            ]
            for from_oscillator in range(n_oscillators)
        ]
    )


def construct_correlated_noise_measurement_matrix(
    n_sources: int,
) -> jnp.ndarray:
    """Constructs the measurement matrix for a correlated noise model.

    The measurement matrix is a block diagonal matrix

    Parameters
    ----------
    n_sources : int
        Number of oscillators in the model.

    Returns
    -------
    measurement_matrix : jnp.ndarray, shape (n_sources, 2 * n_oscillators)
        The measurement matrix for the correlated noise model.
    """
    n_oscillators = n_sources  # Each node is influenced by one oscillator

    measurement_matrix = jnp.zeros((n_oscillators, 2 * n_oscillators))
    for node_ind in range(n_sources):
        for oscillator_ind in range(n_oscillators):
            if node_ind == oscillator_ind:
                _, col = get_block_slice(oscillator_ind, oscillator_ind)
                measurement_matrix = measurement_matrix.at[node_ind, col].set(
                    [1.0, 0.0]
                )

    return measurement_matrix


def construct_directed_influence_transition_matrix(
    freqs: jnp.ndarray,
    damping_coeffs: jnp.ndarray,
    coupling_strengths: jnp.ndarray,
    phase_diffs: jnp.ndarray,
    sampling_freq: float = 1.0,
) -> jnp.ndarray:
    """Constructs the full state transition matrix Aj using jnp.block.

    Based on Equation 2.11 for coupled oscillators. The final matrix will
    have shape (2 * n_oscillators, 2 * n_oscillators).

    Parameters
    ----------
    freqs : jnp.ndarray, shape (n_oscillators,)
        Array of oscillation frequencies (fk) for each oscillator.
    damping_coeffs : jnp.ndarray, shape (n_oscillators,)
        Array of damping coefficients (alpha_j^k) for each oscillator k.
    coupling_strengths : jnp.ndarray, shape (n_oscillators, n_oscillators)
        Matrix where coupling_strengths[n1, n2] is the coupling strength
        from oscillator n2 to oscillator n1 (alpha_j^{n1,n2}).
        Diagonal elements are ignored. A value of 0 indicates no direct coupling.
    phase_diffs : jnp.ndarray, shape (n_oscillators, n_oscillators)
        Matrix where phase_diffs[n1, n2] is the phase difference for
        coupling from oscillator n2 to oscillator n1 (phi_j^{n1,n2}).
        Diagonal elements are ignored.
    sampling_freq : float, optional
        Sampling frequency (Fs) in Hz, by default 1.0.

    Returns
    -------
    transition_matrix : jnp.ndarray, shape (2 * n_oscillators, 2 * n_oscillators)

    Raises
    ------
    ValueError
        If input array dimensions do not match the inferred number of oscillators.
    """
    # Use more descriptive name n_oscillators instead of K
    n_oscillators = freqs.shape[0]
    if not (
        damping_coeffs.shape == (n_oscillators,)
        and coupling_strengths.shape == (n_oscillators, n_oscillators)
        and phase_diffs.shape == (n_oscillators, n_oscillators)
    ):
        raise ValueError(
            "Input array dimensions do not match n_oscillators "
            f"derived from freqs ({n_oscillators})."
        )

    block_rows = []  # List to hold the rows of blocks

    for from_oscillator in range(n_oscillators):
        current_row_blocks = []
        for to_oscillator in range(n_oscillators):
            if from_oscillator == to_oscillator:
                # --- Diagonal Block (k=n1) ---
                # Calculate sum of incoming couplings for oscillator k=n1
                # Sum strengths alpha_j^{n1, other_n2} where other_n2 != n1
                mask = jnp.arange(n_oscillators) != from_oscillator
                sum_incoming_coupling = jnp.sum(
                    coupling_strengths[from_oscillator, mask]
                )

                current_row_blocks.append(
                    _compute_coupled_oscillator_block(
                        freq=freqs[from_oscillator],
                        auto_regressive_coef=damping_coeffs[from_oscillator],
                        sum_incoming_coupling=sum_incoming_coupling,
                        sampling_freq=sampling_freq,
                    )
                )

            else:
                # --- Off-Diagonal Block ---
                current_row_blocks.append(
                    _compute_coupling_transition_block(
                        phase_difference=phase_diffs[from_oscillator, to_oscillator],
                        coupling_strength=coupling_strengths[
                            from_oscillator, to_oscillator
                        ],
                    )
                )

        # Add the completed row of blocks to the list of rows
        block_rows.append(current_row_blocks)

    # Assemble the full matrix from the list of lists of blocks
    return jnp.block(block_rows)


def construct_directed_influence_measurement_matrix(
    n_sources: int,
) -> jnp.ndarray:
    n_oscillators = n_sources  # Each node is influenced by one oscillator
    measurement_matrix = jnp.zeros((n_sources, 2 * n_oscillators))
    block_coefficients = 1 / jnp.sqrt(2)

    for node_ind in range(n_sources):
        for oscillator_ind in range(n_oscillators):
            if node_ind == oscillator_ind:
                _, col = get_block_slice(oscillator_ind, oscillator_ind)
                measurement_matrix = measurement_matrix.at[node_ind, col].set(
                    block_coefficients
                )

    return measurement_matrix


def _get_scaling_factor(s: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Get the scaling factor for the singular values

    Parameters
    ----------
    s : jnp.ndarray, shape (2,)
        The singular values of the matrix
    eps : float, optional
        The tolerance for the singular values, by default 1e-12

    Returns
    -------
    float
        The scaling factor for the singular values
    """
    s = jnp.maximum(s, eps)
    return jnp.sqrt(s[0] * s[1])  # geometric mean


def _project_to_closest_rotation(matrix: jnp.ndarray) -> jnp.ndarray:
    """Project a matrix to the closest rotation matrix using SVD

    Parameters
    ----------
    matrix : jnp.ndarray, shape (2, 2)
        The matrix to project, must be square and 2D

    Returns
    -------
    jnp.ndarray, shape (2, 2)
        The closest rotation matrix to the input matrix

    Raises
    ------
    ValueError
        If the input matrix is not square or not 2D
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")
    try:
        U, s, Vh = jnp.linalg.svd(matrix)

        # Instead of scaling each direction by the singular value,
        # we scale both directions of the matrix by the same geometric mean
        # i.e. there is no shearing in the rotation matrix
        scale_factor = _get_scaling_factor(s)

        # The rotation matrix is U @ Vh
        projected = scale_factor * (U @ Vh)

    except jnp.linalg.LinAlgError as e:
        projected = ZEROS_2x2

    return projected


def project_coupled_transition_matrix(transition_matrix: jnp.ndarray) -> jnp.ndarray:
    """Projects each 2x2 oscillator block of the transition matrix to the closest
    rotation matrix.

    The diagonal blocks are adjusted to ensure the projected transition matrix
    is a valid transition matrix.

    Parameters
    ----------
    transition_matrix : jnp.ndarray, shape (2 * n_oscillators, 2 * n_oscillators)

    Returns
    -------
    projected_transition_matrix : jnp.ndarray, shape (2 * n_oscillators, 2 * n_oscillators)

    Raises
    ------
    ValueError
        If the input matrix dimensions are not even or not square.
    """
    dim = transition_matrix.shape[0]
    if dim % 2 != 0 or transition_matrix.shape != (dim, dim):
        raise ValueError("Input transition_matrix must be square with even dimensions.")
    n_oscillators = dim // 2

    # --- Pass 1: Calculate scaling factors for off-diagonal blocks ---
    scaling_factors = jnp.zeros(
        (n_oscillators, n_oscillators), dtype=transition_matrix.dtype
    )
    for from_oscillator in range(n_oscillators):
        for to_oscillator in range(n_oscillators):
            if from_oscillator != to_oscillator:
                block = transition_matrix[
                    get_block_slice(from_oscillator, to_oscillator)
                ]
                try:
                    scaling_factors = scaling_factors.at[
                        from_oscillator, to_oscillator
                    ].set(_get_scaling_factor(jnp.linalg.svd(block)[1]))
                except jnp.linalg.LinAlgError:
                    pass  # Leave scaling factor as 0 if SVD fails

    # --- Calculate row sums required for diagonal projection ---
    row_sum_scaling = jnp.sum(scaling_factors, axis=1)  # shape (n_oscillators,)

    # --- Pass 2: Compute projected blocks and assemble ---
    projected_block_rows = []

    for from_oscillator in range(n_oscillators):  # Row index of the block
        current_row_blocks = []
        for to_oscillator in range(n_oscillators):  # Column index of the block
            rows, cols = get_block_slice(from_oscillator, to_oscillator)
            if from_oscillator == to_oscillator:
                # --- Diagonal Block ---
                projected_modified_block = _project_to_closest_rotation(
                    transition_matrix[rows, cols]
                    + row_sum_scaling[from_oscillator] * IDENTITY_2x2
                )
                # Subtract the adjustment term
                # Need to double check!
                projected_block = (
                    projected_modified_block
                    - row_sum_scaling[from_oscillator] * IDENTITY_2x2
                )
            else:
                # --- Off-Diagonal ---
                projected_block = _project_to_closest_rotation(
                    transition_matrix[rows, cols]
                )

            # Append the correctly projected block
            current_row_blocks.append(projected_block)

        projected_block_rows.append(current_row_blocks)

    # Assemble the final projected matrix
    return jnp.block(projected_block_rows)


def project_matrix_blockwise(transition_matrix: jnp.ndarray) -> jnp.ndarray:
    """Projects each 2x2 oscillator block of the transition matrix to the closest
    rotation matrix.

    Parameters
    ----------
    transition_matrix : jnp.ndarray, shape (2 * n_oscillators, 2 * n_oscillators)

    Returns
    -------
    projected_transition_matrix, jnp.ndarray, shape (2 * n_oscillators, 2 * n_oscillators)

    Raises
    ------
    ValueError
        If the input matrix dimensions are not even or not square.
    """
    dim = transition_matrix.shape[0]
    if dim % 2 != 0 or transition_matrix.shape != (dim, dim):
        raise ValueError("Input transition_matrix must be square with even dimensions.")
    n_oscillators = dim // 2

    return jnp.block(
        [
            [
                _project_to_closest_rotation(
                    transition_matrix[*get_block_slice(from_oscillator, to_oscillator)]
                )
                for to_oscillator in range(n_oscillators)
            ]
            for from_oscillator in range(n_oscillators)
        ]
    )


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

    def initialize_discrete_state_prob(self):
        self.init_discrete_state_prob = (
            jnp.ones(self.n_discrete_states) / self.n_discrete_states
        )

    def initialize_discrete_transition_matrix(self):
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

    def initialize_continuous_state(self, key: jax.random.PRNGKey):
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
        ) = kalman_maximization_step(
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

    # def fit(self, observations: jnp.ndarray):
    # log_likelihoods = []
    # is_converged = False

    # for i in range(self.n_iter):
    #     # E-step
    #     (
    #         smoother_means,
    #         smoother_covs,
    #         smoother_discrete_state_prob,
    #         smoother_joint_discrete_state_prob,
    #         smoother_cross_cov,
    #     ) = self._e_step(observations)
    #     print(f"Iteration {i}: log-likelihood = {marginal_log_likelihood:.2f}")
    #     log_likelihoods.append(marginal_log_likelihood)
    #     # M-step
    #     (
    #         transition_matrix,
    #         _,
    #         process_cov,
    #         measurement_cov,
    #         init_mean,
    #         init_cov,
    #     ) = kalman_maximization_step(
    #         obs=observations,
    #         smoother_mean=smoother_means,
    #         smoother_cov=smoother_covs,
    #         smoother_cross_cov=smoother_cross_cov,
    #     )

    #     is_converged, _ = check_converged(
    #         log_likelihoods[-1],
    #         log_likelihoods[-2] if len(log_likelihoods) > 1 else 0,
    #     )
    #     print(
    #         f"Diff: {log_likelihoods[-1] - log_likelihoods[-2] if len(log_likelihoods) > 1 else 0:.2f}"
    #     )

    #     if is_converged:
    #         print(
    #             f"Converged at iteration {i} with log-likelihood = {log_likelihoods[-1]:.2f}"
    #         )
    #         break


class CommonOscillatorModel(BaseModel):
    """Common Oscillator Model (COM)

    The measurement matrix depends on the discrete latent state.
    """

    def __init__(self):
        super().__init__()

    def initialize__measurement_matrix(self, key: jax.random.PRNGKey):
        self.measurement_matrix = jax.random.uniform(
            key,
            (self.n_sources, 2 * self.n_oscillators, self.n_discrete_states),
            dtype=jnp.float32,
            minval=0.0,
            maxval=0.1,
        )

    def initialize_measurement_covariance(self):
        measurement_cov = jnp.identity(self.n_sources) * self.measurement_variance
        self.measurement_cov = jnp.stack(
            [measurement_cov] * self.n_discrete_states, axis=2
        )

    def initialize_continuous_transition_matrix(self):
        transition_matrix = construct_common_oscillator_transition_matrix(
            freqs=self.freqs,
            auto_regressive_coef=self.auto_regressive_coef,
            sampling_freq=self.sampling_freq,
        )
        self.continuous_transition_matrix = jnp.stack(
            [transition_matrix] * self.n_discrete_states, axis=2
        )

    def initialize_process_covariance(self):
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

    def initialize_measurement_matrix(self):
        measurement_matrix = construct_correlated_noise_measurement_matrix(
            self.n_sources,
        )
        self.measurement_matrix = jnp.stack(
            [measurement_matrix] * self.n_discrete_states, axis=2
        )

    def initialize_measurement_covariance(self):
        measurement_cov = jnp.identity(self.n_sources) * self.measurement_variance
        self.measurement_cov = jnp.stack(
            [measurement_cov] * self.n_discrete_states, axis=2
        )

    def initialize_continuous_transition_matrix(self):
        transition_matrix = construct_common_oscillator_transition_matrix(
            freqs=self.freqs,
            auto_regressive_coef=self.auto_regressive_coef,
            sampling_freq=self.sampling_freq,
        )
        self.continuous_transition_matrix = jnp.stack(
            [transition_matrix] * self.n_discrete_states, axis=2
        )

    def initialize_process_covariance(self):
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

    def initialize_measurement_matrix(self):
        measurement_matrix = construct_directed_influence_measurement_matrix(
            self.n_sources,
        )
        self.measurement_matrix = jnp.stack(
            [measurement_matrix] * self.n_discrete_states, axis=2
        )

    def initialize_continuous_transition_matrix(self):
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
