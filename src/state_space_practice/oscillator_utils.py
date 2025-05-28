import jax
import jax.numpy as jnp

IDENTITY_2x2 = jnp.identity(2)
ZEROS_2x2 = jnp.zeros((2, 2))


def get_block_slice(from_oscillator: int, to_oscillator: int) -> tuple:
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


def _get_rotation_matrix(rotation_frequency: float) -> jax.Array:
    """Get the rotation matrix for a given frequency

    The rotation matrix is a 2x2 matrix that rotates a vector.

    Parameters
    ----------
    rotation_frequency : float
        The frequency in radians

    Returns
    -------
    rotation_matrix : jax.Array
        The rotation matrix
    """
    cos_rot = jnp.cos(rotation_frequency)
    sin_rot = jnp.sin(rotation_frequency)

    return jnp.array(
        [
            [cos_rot, -sin_rot],
            [sin_rot, cos_rot],
        ],
    )


def _compute_intrinsic_oscillation_block(
    oscillation_freq: float, auto_regressive_coef: float, sampling_freq: float = 1.0
) -> jax.Array:
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
    jax.Array, shape (2, 2)
        The transition matrix at the specified frequency

    Raises
    ------
    ValueError
        If the auto_regressive_coef is not between 0 and 1

    """
    return auto_regressive_coef * _get_rotation_matrix(
        2 * jnp.pi * oscillation_freq / sampling_freq
    )


def _compute_coupled_oscillator_block(
    freq: float,
    auto_regressive_coef: float,
    sum_incoming_coupling_strength: float,
    sampling_freq: float = 1.0,
) -> jax.Array:
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
    diagonal_block : jax.Array, shape (2, 2)
        The transition matrix at the specified frequency

    """
    return (
        _compute_intrinsic_oscillation_block(freq, auto_regressive_coef, sampling_freq)
        - sum_incoming_coupling_strength * IDENTITY_2x2
    )


def _compute_coupling_transition_block(
    phase_difference: float,
    coupling_strength: float,
) -> jax.Array:
    """Compute the off-diagonal block transition matrix for a coupling model

    Parameters
    ----------
    phase_difference : float
        Phase difference between the two oscillators
    coupling_strength : float
        Strength of the coupling between the two oscillators

    Returns
    -------
    jax.Array, shape (2, 2)
        The transition matrix between the two oscillators

    """
    # Calculate the potentially non-zero block
    scaled_rotation = coupling_strength * _get_rotation_matrix(phase_difference)
    return jnp.where(jnp.isclose(coupling_strength, 0.0), ZEROS_2x2, scaled_rotation)


def construct_common_oscillator_transition_matrix(
    freqs: jax.Array,
    auto_regressive_coef: jax.Array,
    sampling_freq: float = 1.0,
) -> jax.Array:
    """Constructs the transition matrix for a common oscillator model.

    The transition matrix is a block diagonal matrix with each block
    corresponding to a single oscillator.

    Also used for the correlated noise model.

    Parameters
    ----------
    freqs : jax.Array, shape (n_oscillators,)
        Array of oscillation frequencies (fk) for each oscillator.
    auto_regressive_coef : jax.Array, shape (n_oscillators,)
        Array of auto-regressive coefficients (alpha_j^k) for each oscillator k.
    sampling_freq : float, optional
        Sampling frequency (Fs) in Hz, by default 1.0.

    Returns
    -------
    transition_matrix : jax.Array, shape (2 * n_oscillators, 2 * n_oscillators)

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
            oscillation_freq=freqs[k],
            auto_regressive_coef=auto_regressive_coef[k],
            sampling_freq=sampling_freq,
        )
        for k in range(n_oscillators)
    ]

    return jax.scipy.linalg.block_diag(*diag_blocks)


def construct_common_oscillator_process_covariance(
    variance: jax.Array,
) -> jax.Array:
    pass
    """Constructs the process covariance matrix for a common oscillator model.

    The process covariance matrix ($$ \\Sigma $$) is a block diagonal matrix with each block
    corresponding to a single oscillator.

    Parameters
    ----------
    variance : jax.Array, shape (n_oscillators,)
        Array of process noise variances (sigma_j) for each oscillator.

    Returns
    -------
    process_covariance : jax.Array, shape (2 * n_oscillators, 2 * n_oscillators)
        The process covariance matrix for the common oscillator model.

    """
    n_oscillators = variance.shape[0]
    diag_blocks = [variance[k] * IDENTITY_2x2 for k in range(n_oscillators)]

    return jax.scipy.linalg.block_diag(*diag_blocks)


def construct_correlated_noise_process_covariance(
    variance: jax.Array,
    phase_difference: jax.Array,
    coupling_strength: jax.Array,
) -> jax.Array:
    """

    Parameters
    ----------
    variance : jax.Array, shape (n_oscillators,)
        Array of process noise variances (sigma_j) for each oscillator.
    phase_difference : jax.Array, shape (n_oscillators, n_oscillators)
        Matrix where phase_diffs[n1, n2] is the phase difference for
        coupling from oscillator n2 to oscillator n1 (phi_j^{n1,n2}).
    coupling_strength : jax.Array, shape (n_oscillators, n_oscillators)
        Matrix where coupling_strengths[n1, n2] is the coupling strength

    Returns
    -------
    process_covariance : jax.Array, shape (2 * n_oscillators, 2 * n_oscillators)
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
) -> jax.Array:
    """Constructs the measurement matrix for a correlated noise model.

    The measurement matrix is a block diagonal matrix

    Parameters
    ----------
    n_sources : int
        Number of oscillators in the model.

    Returns
    -------
    measurement_matrix : jax.Array, shape (n_sources, 2 * n_oscillators)
        The measurement matrix for the correlated noise model.
    """
    n_oscillators = n_sources  # Each node is influenced by one oscillator

    measurement_matrix = jnp.zeros((n_sources, 2 * n_oscillators))

    # Get the row indices (0 to n_sources-1)
    row_indices = jnp.arange(n_sources)
    # Get the column indices (0, 2, 4, ...)
    col_indices = jnp.arange(0, 2 * n_oscillators, 2)

    # Set the [1, 0] blocks
    measurement_matrix = measurement_matrix.at[row_indices, col_indices].set(1.0)

    return measurement_matrix


def construct_directed_influence_transition_matrix(
    freqs: jax.Array,
    damping_coeffs: jax.Array,
    coupling_strengths: jax.Array,
    phase_diffs: jax.Array,
    sampling_freq: float = 1.0,
) -> jax.Array:
    """Constructs the full state transition matrix Aj.

    Based on Equation 2.11 for coupled oscillators. The final matrix will
    have shape (2 * n_oscillators, 2 * n_oscillators).

    Parameters
    ----------
    freqs : jax.Array, shape (n_oscillators,)
        Array of oscillation frequencies (fk) for each oscillator.
    damping_coeffs : jax.Array, shape (n_oscillators,)
        Array of damping coefficients (alpha_j^k) for each oscillator k.
    coupling_strengths : jax.Array, shape (n_oscillators, n_oscillators)
        Matrix where coupling_strengths[n1, n2] is the coupling strength
        from oscillator n2 to oscillator n1 (alpha_j^{n1,n2}).
        Diagonal elements are ignored. A value of 0 indicates no direct coupling.
    phase_diffs : jax.Array, shape (n_oscillators, n_oscillators)
        Matrix where phase_diffs[n1, n2] is the phase difference for
        coupling from oscillator n2 to oscillator n1 (phi_j^{n1,n2}).
        Diagonal elements are ignored.
    sampling_freq : float, optional
        Sampling frequency (Fs) in Hz, by default 1.0.

    Returns
    -------
    transition_matrix : jax.Array, shape (2 * n_oscillators, 2 * n_oscillators)

    Raises
    ------
    ValueError
        If input array dimensions do not match the inferred number of oscillators.
    """
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

    # 1. Calculate sum_incoming_coupling (vectorized)
    # We need to exclude the diagonal before summing.
    sum_incoming_coupling = jnp.sum(
        coupling_strengths, axis=1, where=~jnp.eye(n_oscillators, dtype=bool)
    )

    # 2. Vmap _compute_coupling_transition_block for off-diagonals
    # We create a function that computes A_j^{n1, n2}
    coupling_row = jax.vmap(
        _compute_coupling_transition_block, in_axes=(0, 0)
    )  # Vmap over columns (n2)
    coupling_all = jax.vmap(coupling_row, in_axes=(0, 0))  # Vmap over rows (n1)

    all_coupling_blocks = coupling_all(phase_diffs, coupling_strengths)
    # Shape: (n_oscillators, n_oscillators, 2, 2)

    # 3. Vmap _compute_coupled_oscillator_block for diagonals
    diag = jax.vmap(_compute_coupled_oscillator_block, in_axes=(0, 0, 0, None))
    all_diag_blocks = diag(freqs, damping_coeffs, sum_incoming_coupling, sampling_freq)
    # Shape: (n_oscillators, 2, 2)

    # 4. Combine: Replace diagonal blocks in all_coupling_blocks
    # Get indices for the diagonal blocks
    diag_indices = jnp.arange(n_oscillators)
    all_blocks = all_coupling_blocks.at[diag_indices, diag_indices].set(all_diag_blocks)
    # Shape: (n_oscillators, n_oscillators, 2, 2)

    # 5. Reshape and transpose to final matrix form
    # (n1, n2, 2, 2) -> (n1, 2, n2, 2) -> (2 * n1, 2 * n2)
    transition_matrix = all_blocks.swapaxes(1, 2).reshape(
        2 * n_oscillators, 2 * n_oscillators
    )

    return transition_matrix


def construct_directed_influence_measurement_matrix(
    n_sources: int,
) -> jax.Array:
    """Constructs the measurement matrix for a directed influence model.

    The measurement matrix ($$ H $$) creates an observation by averaging the 'x'
    and 'y' components of each oscillator's state, scaled by 1/sqrt(2).
    It has a shape of (n_sources, 2 * n_oscillators).

    Parameters
    ----------
    n_sources : int
        Number of sources, equal to the number of oscillators.

    Returns
    -------
    measurement_matrix : jax.Array
        The measurement matrix, shape (n_sources, 2 * n_sources).
    """
    n_oscillators = n_sources
    measurement_matrix = jnp.zeros((n_sources, 2 * n_oscillators))
    block_coefficient = 1.0 / jnp.sqrt(2.0)

    # Get the row indices (0 to n_sources-1)
    row_indices = jnp.arange(n_sources)
    # Get the 'x' column indices (0, 2, 4, ...)
    col_indices_x = jnp.arange(0, 2 * n_oscillators, 2)
    # Get the 'y' column indices (1, 3, 5, ...)
    col_indices_y = jnp.arange(1, 2 * n_oscillators, 2)

    # Set the [coeff, coeff] blocks
    measurement_matrix = measurement_matrix.at[row_indices, col_indices_x].set(
        block_coefficient
    )
    measurement_matrix = measurement_matrix.at[row_indices, col_indices_y].set(
        block_coefficient
    )

    return measurement_matrix


def _get_scaling_factor(s: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Get the scaling factor for the singular values

    Parameters
    ----------
    s : jax.Array, shape (2,)
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


def _project_to_closest_rotation(matrix: jax.Array) -> jax.Array:
    """Project a matrix to the closest rotation matrix using SVD

    Parameters
    ----------
    matrix : jax.Array, shape (2, 2)
        The matrix to project, must be square and 2D

    Returns
    -------
    jax.Array, shape (2, 2)
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


def project_coupled_transition_matrix(transition_matrix: jax.Array) -> jax.Array:
    """Projects each 2x2 oscillator block of the transition matrix to the closest
    rotation matrix.

    The diagonal blocks are adjusted to ensure the projected transition matrix
    is a valid transition matrix.

    Parameters
    ----------
    transition_matrix : jax.Array, shape (2 * n_oscillators, 2 * n_oscillators)

    Returns
    -------
    projected_transition_matrix : jax.Array, shape (2 * n_oscillators, 2 * n_oscillators)

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


def project_matrix_blockwise(transition_matrix: jax.Array) -> jax.Array:
    """Projects each 2x2 oscillator block of the transition matrix to the closest
    rotation matrix.

    Parameters
    ----------
    transition_matrix : jax.Array, shape (2 * n_oscillators, 2 * n_oscillators)

    Returns
    -------
    projected_transition_matrix, jax.Array, shape (2 * n_oscillators, 2 * n_oscillators)

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
                    transition_matrix[get_block_slice(from_oscillator, to_oscillator)]
                )
                for to_oscillator in range(n_oscillators)
            ]
            for from_oscillator in range(n_oscillators)
        ]
    )
