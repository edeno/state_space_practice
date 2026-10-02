from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from state_space_practice.exceptions import StateSpaceWarning
from state_space_practice.utils import (
    contains_tracer,
    debug_print_if,
    differentiable_spectral_radius,
    stabilize_transition_matrix,
    symmetrize,
)

logger = logging.getLogger(__name__)

IDENTITY_2x2 = jnp.identity(2)
# Relative tolerance above ``max_spectral_radius`` tolerated by the DIM stability
# scale before it engages (absorbs round-off of an already clamped matrix).
_STABILITY_SCALE_RTOL = 1e-10
ZEROS_2x2 = jnp.zeros((2, 2))


def _wrap_angle(angle: jax.Array) -> jax.Array:
    """Wrap angles to [-pi, pi) for phase-equivalence checks."""
    return jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi


def _scatter_block_diagonal(blocks: jax.Array) -> jax.Array:
    """Assemble an array of 2x2 blocks into a block-diagonal matrix.

    Parameters
    ----------
    blocks : jax.Array, shape (n_blocks, 2, 2)

    Returns
    -------
    jax.Array, shape (2 * n_blocks, 2 * n_blocks)
    """
    n = blocks.shape[0]
    size = 2 * n
    # Flatten each 2x2 block and scatter 4 elements per block
    flat = blocks.reshape(n, 4)  # (n, 4)
    offsets = jnp.array([0, 1, size, size + 1])  # within-block offsets in flat matrix
    base = 2 * jnp.arange(n) * (size + 1)  # diagonal starting positions
    indices = base[:, None] + offsets[None, :]  # (n, 4)
    return (
        jnp.zeros(size * size).at[indices.ravel()].set(flat.ravel()).reshape(size, size)
    )


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


def _get_rotation_matrix(rotation_frequency: ArrayLike) -> jax.Array:
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
    oscillation_freq: ArrayLike,
    damping_coef: ArrayLike,
    sampling_freq: float | jax.Array = 1.0,
) -> jax.Array:
    """Compute the rotation matrix for a given frequency and auto-regressive coefficient

    Parameters
    ----------
    oscillation_freq : float
        Oscillation frequency in Hz
    damping_coef : float
        Controls the damping of the oscillation. A value of 1 corresponds to a
        pure oscillation, while a value of 0 corresponds to no oscillation.
        Values outside ``[0, 1]`` are accepted here so callers can decide
        whether to validate, clip, or optimize unconstrained parameters.
    sampling_freq : float, optional
        Samples per second, by default 1

    Returns
    -------
    jax.Array, shape (2, 2)
        The transition matrix at the specified frequency

    """
    result: jax.Array = jnp.asarray(damping_coef) * _get_rotation_matrix(
        2 * jnp.pi * oscillation_freq / sampling_freq
    )
    return result


def _compute_coupled_oscillator_block(
    freq: ArrayLike,
    damping_coef: ArrayLike,
    sum_incoming_coupling_strength: ArrayLike,
    sampling_freq: float | jax.Array = 1.0,
) -> jax.Array:
    """Compute the diagonal block of the transition matrix for the coupled model.

    This block represents the intrinsic oscillation of an oscillator,
    adjusted by the sum of incoming coupling strengths from other oscillators.

    This is similar to partial directed coherence.

    Parameters
    ----------
    freq : float
        Oscillation frequency in Hz
    damping_coef : float
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
    eye = jnp.eye(
        2, dtype=jnp.result_type(freq, damping_coef, sum_incoming_coupling_strength)
    )
    return (
        _compute_intrinsic_oscillation_block(freq, damping_coef, sampling_freq)
        - sum_incoming_coupling_strength * eye
    )


def _compute_coupling_transition_block(
    phase_difference: ArrayLike,
    coupling_strength: ArrayLike,
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
    safe_phase = jnp.where(
        (coupling_strength == 0.0) & ~jnp.isfinite(phase_difference),
        0.0,
        phase_difference,
    )
    return jnp.asarray(coupling_strength) * _get_rotation_matrix(safe_phase)


def construct_common_oscillator_transition_matrix(
    freqs: jax.Array,
    damping_coef: jax.Array,
    sampling_freq: float = 1.0,
) -> jax.Array:
    """Constructs the transition matrix for a common oscillator model.

    The "common oscillator" model represents a set of independent (uncoupled)
    oscillators, each with its own frequency and damping.

    Accordingly, the transition matrix is a block diagonal matrix with each block
    corresponding to a single oscillator.

    Also used for the correlated noise model.

    Parameters
    ----------
    freqs : jax.Array, shape (n_oscillators,)
        Array of oscillation frequencies (fk) for each oscillator.
    damping_coef : jax.Array, shape (n_oscillators,)
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
    if not damping_coef.shape == (n_oscillators,):
        raise ValueError("damping_coef must be a 1D array of shape (n_oscillators,)")
    # Warn and clamp out-of-range values (JIT-compatible via pre-trace check)
    if not contains_tracer(damping_coef):
        if jnp.any(jnp.logical_or(damping_coef > 1, damping_coef < 0)):
            warnings.warn(
                "damping_coef values outside [0, 1] will be clipped",
                StateSpaceWarning,
                stacklevel=2,
            )
    damping_coef = jnp.clip(damping_coef, 0.0, 1.0)

    # Vectorized: compute all 2x2 blocks at once, then scatter into block-diagonal
    blocks = jax.vmap(_compute_intrinsic_oscillation_block, in_axes=(0, 0, None))(
        freqs, damping_coef, sampling_freq
    )
    # blocks: (n_oscillators, 2, 2)
    return _scatter_block_diagonal(blocks)


def construct_common_oscillator_process_covariance(
    variance: jax.Array,
) -> jax.Array:
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
    # Each oscillator contributes a diagonal 2x2 block: variance[k] * I
    # This is equivalent to diag(v0, v0, v1, v1, ..., vn, vn)
    return jnp.diag(jnp.repeat(variance, 2))


def canonicalize_correlated_noise_pair_parameters(
    phase_difference: jax.Array,
    coupling_strength: jax.Array,
    *,
    atol: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """Canonicalize CNM pair parameters to strict-upper-triangle storage.

    The correlated-noise model has one undirected noise-correlation parameter per
    oscillator pair. For convenience at public API boundaries, this accepts any
    of three equivalent user inputs for each pair and discrete state:

    - strict upper triangle only,
    - strict lower triangle only, or
    - both triangles, if they describe the same covariance block
      (equal coupling and opposite phase).

    The returned arrays zero the diagonal/lower triangle and store the canonical
    pair parameters in the strict upper triangle. If both directions are supplied
    but disagree, a ``ValueError`` is raised instead of silently ignoring one.

    This is a host-side API helper, not a differentiated loss primitive.
    """
    phase = jnp.asarray(phase_difference)
    coupling = jnp.asarray(coupling_strength)
    if phase.shape != coupling.shape:
        raise ValueError(
            "phase_difference and coupling_strength must have the same shape; "
            f"got {phase.shape} and {coupling.shape}."
        )
    if phase.ndim not in (2, 3) or phase.shape[0] != phase.shape[1]:
        raise ValueError(
            "phase_difference and coupling_strength must be square pair "
            f"matrices with optional discrete-state axis, got shape {phase.shape}."
        )
    if not bool(jnp.all(jnp.isfinite(phase))) or not bool(
        jnp.all(jnp.isfinite(coupling))
    ):
        raise ValueError(
            "phase_difference and coupling_strength must contain only finite "
            "values (no NaN/Inf)."
        )

    n_oscillators = phase.shape[0]
    canonical_phase = jnp.zeros_like(phase)
    canonical_coupling = jnp.zeros_like(coupling)

    diag_idx = jnp.arange(n_oscillators)
    diag_coupling = coupling[diag_idx, diag_idx]
    if bool(jnp.any(jnp.abs(diag_coupling) > atol)):
        max_diag = float(jnp.max(jnp.abs(diag_coupling)))
        raise ValueError(
            "coupling_strength diagonal entries are ignored by CNM and must be "
            f"zero (max |diag| = {max_diag:g}). Put pair couplings in an "
            "off-diagonal entry instead."
        )

    for i in range(n_oscillators):
        for j in range(i + 1, n_oscillators):
            upper_c = coupling[i, j]
            lower_c = coupling[j, i]
            upper_p = phase[i, j]
            lower_p = phase[j, i]
            has_upper = jnp.abs(upper_c) > atol
            has_lower = jnp.abs(lower_c) > atol
            has_both = has_upper & has_lower

            coupling_conflict = has_both & (jnp.abs(upper_c - lower_c) > atol)
            phase_conflict = has_both & (jnp.abs(_wrap_angle(upper_p + lower_p)) > atol)
            if bool(jnp.any(coupling_conflict | phase_conflict)):
                raise ValueError(
                    "Conflicting correlated-noise pair parameters for pair "
                    f"({i}, {j}). CNM covariance has one undirected pair "
                    "parameter: supply only one triangle, or set "
                    "coupling_strength[i,j] == coupling_strength[j,i] and "
                    "phase_difference[i,j] == -phase_difference[j,i] (mod 2*pi)."
                )

            use_lower = (~has_upper) & has_lower
            canon_c = jnp.where(use_lower, lower_c, upper_c)
            canon_p = jnp.where(use_lower, -lower_p, upper_p)
            canonical_coupling = canonical_coupling.at[i, j].set(canon_c)
            canonical_phase = canonical_phase.at[i, j].set(canon_p)

    return canonical_phase, canonical_coupling


def construct_correlated_noise_process_covariance(
    variance: jax.Array,
    phase_difference: jax.Array,
    coupling_strength: jax.Array,
) -> jax.Array:
    """Symmetric process covariance for correlated oscillator noise.

    Lets the noise driving different oscillators be correlated, representing
    shared unobserved inputs -- distinct from directed dynamic coupling in the
    transition matrix (directed influence belongs in the DirectedInfluenceModel,
    not in a covariance).

    Because a covariance is symmetric, this low-level constructor reads the
    STRICT UPPER TRIANGLE (i < j) of ``phase_difference`` /
    ``coupling_strength``: the (i, j) cross-block is
    ``coupling_strength[i, j] * R(phase_difference[i, j])`` and the (j, i)
    block is its transpose. Model constructors call
    ``canonicalize_correlated_noise_pair_parameters`` first, so user-facing
    inputs may be upper-only, lower-only, or mirrored full-pair values.

    Parameters
    ----------
    variance : jax.Array, shape (n_oscillators,)
        Process-noise variance for each oscillator; sets the diagonal 2x2 blocks
        to ``variance[j] * I``.
    phase_difference : jax.Array, shape (n_oscillators, n_oscillators)
        Per-pair phase of the noise correlation in canonical strict-upper form;
        ``phase_difference[i, j]`` sets the phase of the (i, j) cross-block, and
        the (j, i) block is its transpose.
    coupling_strength : jax.Array, shape (n_oscillators, n_oscillators)
        Per-pair noise-correlation magnitude in canonical strict-upper form.

    Returns
    -------
    process_covariance : jax.Array, shape (2 * n_oscillators, 2 * n_oscillators)
        Symmetric process covariance. Symmetric by construction, but NOT
        guaranteed positive semidefinite for large coupling -- the model's
        ``_project_parameters`` / entry-point validation enforce PSD.
    """
    n_oscillators = variance.shape[0]

    # Compute all (n, n) coupling blocks at once via nested vmap
    coupling_row = jax.vmap(_compute_coupling_transition_block, in_axes=(0, 0))
    all_blocks = jax.vmap(coupling_row, in_axes=(0, 0))(
        phase_difference, coupling_strength
    )  # (n_oscillators, n_oscillators, 2, 2)

    # Enforce covariance symmetry: the (j, i) cross-block must equal the (i, j)
    # cross-block transposed. Take the strict upper triangle (i < j) as the
    # source of truth and mirror it to the lower triangle as its transpose, so
    # the assembled Q is symmetric by construction. This gives one correlation
    # magnitude and one (signed) phase per oscillator pair -- the identifiable
    # parametrization of a phase-coupled noise covariance -- instead of the
    # independent directed blocks that made Q non-symmetric. Only the strict
    # upper triangle of ``phase_difference`` / ``coupling_strength`` is used.
    lower_source = jnp.swapaxes(jnp.swapaxes(all_blocks, 0, 1), -1, -2)
    upper_mask = (
        jnp.arange(n_oscillators)[:, None] < jnp.arange(n_oscillators)[None, :]
    )  # (n, n), True where i < j
    all_blocks = jnp.where(upper_mask[..., None, None], all_blocks, lower_source)

    # Replace diagonal blocks with variance * I
    diag_blocks = variance[:, None, None] * IDENTITY_2x2[None, :, :]
    diag_idx = jnp.arange(n_oscillators)
    all_blocks = all_blocks.at[diag_idx, diag_idx].set(diag_blocks)

    # Reshape (n, n, 2, 2) -> (2n, 2n)
    return all_blocks.swapaxes(1, 2).reshape(2 * n_oscillators, 2 * n_oscillators)


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
    return measurement_matrix.at[row_indices, col_indices].set(1.0)


def construct_directed_influence_transition_matrix(
    freqs: jax.Array,
    damping_coeffs: jax.Array,
    coupling_strengths: jax.Array,
    phase_diffs: jax.Array,
    sampling_freq: float | jax.Array = 1.0,
) -> jax.Array:
    """Constructs the full state transition matrix Aj.

    Based on Equation 2.11 for coupled oscillators. The final matrix will
    have shape (2 * n_oscillators, 2 * n_oscillators).

    The "directed influence" model represents a system where oscillators
    directly affect each other's dynamics with specific phase lags and strengths.

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
    return all_blocks.swapaxes(1, 2).reshape(2 * n_oscillators, 2 * n_oscillators)


def compute_directed_influence_stability_scale(
    freqs: ArrayLike,
    damping_coef: ArrayLike,
    coupling_strength: ArrayLike,
    sampling_freq: float | jax.Array,
    max_spectral_radius: float | jax.Array = 0.99,
    *,
    phase_difference: ArrayLike,
) -> jax.Array:
    """Return a differentiable global scale guaranteeing stable DIM dynamics.

    Scaling damping and coupling by the same scalar ``s`` scales the complete
    directed-influence transition matrix by ``s`` (frequency and phase are
    unchanged), so its spectral radius scales by ``s`` too. The returned scale
    is ``min(1, max_spectral_radius / max_j rho(A_j))`` with ``rho`` the
    *actual* spectral radius of each state's unscaled matrix
    (:func:`~state_space_practice.utils.differentiable_spectral_radius`): it is
    exactly ``1`` -- and has zero gradient -- whenever every ``A_j`` already
    honors the bound, so stable parameters are a fixed point of the
    construction. (A norm-based upper bound would engage for stable matrices
    and over-damp them.) ``coupling_strength`` / ``phase_difference`` may be
    one matrix or a stack with a final discrete-state axis; one shared scale is
    returned for the full stack.

    Parameters
    ----------
    freqs : ArrayLike, shape (n_oscillators,)
    damping_coef : ArrayLike, shape (n_oscillators,)
    coupling_strength : ArrayLike, shape (n_osc, n_osc[, n_discrete_states])
    sampling_freq : float
    max_spectral_radius : float, default=0.99
    phase_difference : ArrayLike, same shape as ``coupling_strength``
        Required: the spectral radius depends on the coupling phases.

    Returns
    -------
    jax.Array, scalar
        Scale in ``(0, 1]``.
    """
    freqs_arr = jnp.asarray(freqs)
    damping_arr = jnp.asarray(damping_coef)
    coupling_arr = jnp.asarray(coupling_strength)
    phase_arr = jnp.asarray(phase_difference)
    if coupling_arr.ndim == 2:
        coupling_arr = coupling_arr[..., None]
    if phase_arr.ndim == 2:
        phase_arr = phase_arr[..., None]
    phase_arr = jnp.broadcast_to(phase_arr, coupling_arr.shape)

    transition_stack = jax.vmap(
        lambda coupling, phase: construct_directed_influence_transition_matrix(
            freqs=freqs_arr,
            damping_coeffs=damping_arr,
            coupling_strengths=coupling,
            phase_diffs=phase,
            sampling_freq=sampling_freq,
        ),
        in_axes=(-1, -1),
    )(coupling_arr, phase_arr)
    radius = jnp.max(differentiable_spectral_radius(transition_stack))
    tiny = jnp.finfo(jnp.result_type(radius, 1.0)).tiny
    # Relative slack so an extract -> construct round-trip of a matrix already
    # clamped to exactly the bound is not re-scaled by floating-point noise.
    threshold = max_spectral_radius * (1.0 + _STABILITY_SCALE_RTOL)
    return jnp.where(
        radius > threshold,
        max_spectral_radius / jnp.maximum(radius, tiny),
        jnp.ones_like(radius),
    )


def construct_directed_influence_measurement_matrix(
    n_sources: int,
) -> jax.Array:
    """Constructs the measurement matrix for a directed influence model.

    The measurement matrix ($$ H $$) creates an observation by averaging the 'x'
    and 'y' components of each oscillator's state, scaled by 1/sqrt(2).
    It has a shape of (n_sources, 2 * n_oscillators).

    Setting the real and imaginary parts to 1/sqrt(2) ensures that the signal
    at each node is determined equally by the real and imaginary components
    of the oscillator. A measurement matrix that takes only the real parts
    allows the imaginary component of each oscillator to be unconstrained.
    Equal parts allows a mixing of oscillators in the measurement space.

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
    return measurement_matrix.at[row_indices, col_indices_y].set(block_coefficient)


def _get_scaling_factor(s: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Get the scaling factor for the singular values.

    This avoids shearing in the rotation matrix by scaling both directions
    by the geometric mean of the singular values.

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
    """Project a matrix to the closest rotation matrix using SVD.

    Instead of scaling each direction by the singular value,
    scale both directions of the matrix by the same geometric mean
    to prevent shearing in the rotation matrix.

    Parameters
    ----------
    matrix : jax.Array, shape (2, 2)
        The matrix to project, must be square and 2D

    Returns
    -------
    jax.Array, shape (2, 2)
        The closest scaled rotation matrix to the input matrix.
        Falls back to a damped identity if SVD produces non-finite values.

    Raises
    ------
    ValueError
        If the input matrix is not square or not 2D
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")

    U, s, Vh = jnp.linalg.svd(matrix)

    # Instead of scaling each direction by the singular value,
    # we scale both directions of the matrix by the same geometric mean
    # i.e. there is no shearing in the rotation matrix
    scale_factor = _get_scaling_factor(s)

    # Orthogonal Procrustes with det +1. Plain U @ Vh may be a reflection when
    # det(U @ Vh) < 0, which is outside the oscillator scaled-rotation family.
    det_uv = jnp.linalg.det(U @ Vh)
    det_sign = jnp.where(det_uv < 0.0, -1.0, 1.0)
    correction = jnp.diag(jnp.array([1.0, det_sign], dtype=matrix.dtype))
    rotation = U @ correction @ Vh
    projected = scale_factor * rotation

    # If SVD produced NaN/Inf (numerical failure), fall back to a damped
    # identity (zero rotation). Returning the original matrix would defeat
    # the purpose of projection since it is the problematic input. The
    # Frobenius norm / sqrt(2) gives the RMS singular value, which is the
    # ideal scale for a 2x2 scaled rotation. Downstream spectral radius
    # clamping will correct the scale if needed.
    #
    # This helper is PURE (no telemetry): it runs under ``jax.vmap`` (see
    # ``project_coupled_transition_matrix``), where ``lax.cond`` lowers to
    # ``select`` and both branches execute, so a per-block ``debug_print_if``
    # here fires on every lane regardless of the predicate. The fallback signal
    # is emitted once, reduced with ``jnp.any``, at the non-vmapped call sites.
    is_valid = jnp.all(jnp.isfinite(projected))
    frob_norm = jnp.linalg.norm(matrix, "fro")
    fallback_scale = jnp.where(jnp.isfinite(frob_norm), frob_norm / jnp.sqrt(2.0), 0.5)
    fallback = fallback_scale * jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    return jnp.where(is_valid, projected, fallback)


def _project_to_scaled_rotation_matrix(matrix: jax.Array) -> jax.Array:
    """Project a 2x2 matrix to the closest scaled rotation block.

    The oscillator models use blocks of the form ``scale * R(angle)``, i.e.
    ``[[a, -b], [b, a]]``.  This is a 2D linear subspace of 2x2 matrices, so
    the Frobenius projection has the closed form below.  This differs from the
    generic orthogonal Procrustes/SVD projection in
    ``_project_to_closest_rotation``: that routine forces a proper rotation
    (det +1) via a determinant correction, while this closed form additionally
    constrains the block to the exact ``scale * R(angle)`` form.
    """
    if matrix.shape != (2, 2):
        raise ValueError("Scaled-rotation projection expects a 2x2 matrix.")

    a = 0.5 * (matrix[0, 0] + matrix[1, 1])
    b = 0.5 * (matrix[1, 0] - matrix[0, 1])
    projected = jnp.array([[a, -b], [b, a]], dtype=matrix.dtype)

    is_valid = jnp.all(jnp.isfinite(projected))
    frob_norm = jnp.linalg.norm(matrix, "fro")
    fallback_scale = jnp.where(jnp.isfinite(frob_norm), frob_norm / jnp.sqrt(2.0), 0.5)
    fallback = fallback_scale * IDENTITY_2x2.astype(matrix.dtype)
    return jnp.where(is_valid, projected, fallback)


def _warn_if_rotation_projection_degenerate(transition_matrix: jax.Array) -> None:
    """Emit ONE fallback warning if any 2x2 block would fail rotation projection.

    The projection helpers fall back to a damped identity (zero rotation,
    silently nulling an oscillation component) exactly when their input block is
    non-finite. The reduced predicate is therefore ``any(~isfinite(input))``,
    computed here in the NON-vmapped caller so ``debug_print_if``'s ``lax.cond``
    behaves conditionally instead of firing on every block.
    """
    debug_print_if(
        jnp.any(~jnp.isfinite(transition_matrix)),
        "oscillator_utils: transition matrix has non-finite entries; the "
        "affected 2x2 oscillator block(s) were replaced by a damped-identity "
        "(zero rotation) fallback, so those components are nulled until the "
        "input is regularized.",
    )


def _extract_scale_and_angle(block: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Extract scaling factor and rotation angle from a 2x2 oscillator block.

    Decomposes ``block ≈ scale * R(angle)`` where R is a rotation matrix.

    Parameters
    ----------
    block : jax.Array, shape (2, 2)

    Returns
    -------
    scale : jax.Array
        Coupling magnitude of the closest scaled-rotation block.
    angle : jax.Array
        Rotation angle in radians.
    """
    projected = _project_to_scaled_rotation_matrix(block)
    scale = jnp.sqrt(projected[0, 0] ** 2 + projected[1, 0] ** 2)
    angle = jnp.where(
        scale > 0.0,
        jnp.arctan2(projected[1, 0], projected[0, 0]),
        0.0,
    )
    return scale, angle


def project_coupled_transition_matrix(transition_matrix: jax.Array) -> jax.Array:
    """Project each 2x2 block to the closest scaled-rotation oscillator block.

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

    # Non-vmapped: emit one reduced fallback signal for the whole matrix.
    _warn_if_rotation_projection_degenerate(transition_matrix)

    # Reshape to (n_oscillators, 2, n_oscillators, 2) for block access
    blocks = transition_matrix.reshape(n_oscillators, 2, n_oscillators, 2)
    # Transpose to (n_oscillators, n_oscillators, 2, 2) - (from, to, row, col)
    blocks = blocks.transpose(0, 2, 1, 3)

    project_all = jax.vmap(jax.vmap(_project_to_scaled_rotation_matrix, in_axes=0))
    projected_blocks = project_all(blocks)

    # Reshape back to (2*n_oscillators, 2*n_oscillators)
    # (from, to, row, col) -> (from, row, to, col) -> (2*n_osc, 2*n_osc)
    return projected_blocks.transpose(0, 2, 1, 3).reshape(dim, dim)


def _matrix_to_oscillator_blocks(matrix: jax.Array) -> jax.Array:
    """Reshape a ``(2n, 2n)`` matrix to ``(n, n, 2, 2)`` blocks."""
    dim = matrix.shape[0]
    if dim % 2 != 0 or matrix.shape != (dim, dim):
        raise ValueError("Input matrix must be square with even dimensions.")
    n_oscillators = dim // 2
    return matrix.reshape(n_oscillators, 2, n_oscillators, 2).transpose(0, 2, 1, 3)


def _oscillator_blocks_to_matrix(blocks: jax.Array) -> jax.Array:
    """Assemble ``(n, n, 2, 2)`` oscillator blocks into a matrix."""
    n_oscillators = blocks.shape[0]
    return blocks.transpose(0, 2, 1, 3).reshape(2 * n_oscillators, 2 * n_oscillators)


def _cnm_structured_projection(
    process_covariance: jax.Array, min_eigenvalue: float
) -> tuple[jax.Array, jax.Array]:
    """Project a covariance onto the CNM block family, block by block.

    Returns ``(structured, diag_only)``: ``structured`` has ``variance_k * I``
    diagonal blocks (variance floored at ``min_eigenvalue``) and symmetric
    scaled-rotation off-diagonal blocks; ``diag_only`` keeps only the
    diagonal blocks. Every block is projected in one vectorised pass (no
    per-block ``.at[].set`` chain), so the function is a handful of fused
    ops whether run eagerly or under ``jit``.
    """
    cov = symmetrize(process_covariance)
    blocks = _matrix_to_oscillator_blocks(cov)  # (n, n, 2, 2)
    n_oscillators = blocks.shape[0]
    eye = jnp.eye(n_oscillators, dtype=cov.dtype)

    # Diagonal blocks: variance_k * I, variance floored at min_eigenvalue.
    variance = jnp.maximum(
        0.5 * jnp.trace(blocks, axis1=-2, axis2=-1).diagonal(), min_eigenvalue
    )
    diag_blocks = (eye * variance[:, None])[:, :, None, None] * IDENTITY_2x2.astype(
        cov.dtype
    )

    # Off-diagonal blocks: scaled-rotation projection of every upper block,
    # mirrored (transposed) into the lower triangle.
    projected = jax.vmap(jax.vmap(_project_to_scaled_rotation_matrix))(blocks)
    upper = jnp.triu(jnp.ones((n_oscillators, n_oscillators), dtype=cov.dtype), k=1)
    upper_blocks = upper[:, :, None, None] * projected
    structured_blocks = (
        diag_blocks + upper_blocks + jnp.transpose(upper_blocks, (1, 0, 3, 2))
    )

    return (
        _oscillator_blocks_to_matrix(structured_blocks),
        _oscillator_blocks_to_matrix(diag_blocks),
    )


def _cnm_psd_shrink_factor(
    structured: jax.Array,
    diag_only: jax.Array,
    min_eigenvalue: float,
    safety_margin: float | None = None,
) -> jax.Array:
    """Largest ``t`` in ``[0, 1]`` with ``diag_only + t * O >= min_eigenvalue * I``.

    ``O = structured - diag_only`` holds the linkage blocks. Write
    ``S = diag_only - min_eigenvalue * I``: diagonal, with entries ``>= 0``
    because the block variances are floored at ``min_eigenvalue``. For
    ``S > 0`` the constraint is a congruence away from an eigenvalue bound::

        S + t O >= 0   <=>   I + t M >= 0,   M = S^{-1/2} O S^{-1/2}

    which holds iff ``t * lambda_max(-M) <= 1``. So ``t* = 1 /
    lambda_max(-M)`` when that eigenvalue is positive and ``t* = 1`` (no
    shrink needed) otherwise -- one symmetric eigendecomposition, traceable
    under ``jit``.

    A coordinate whose variance sits exactly at the floor has ``S_ii = 0``.
    It drops out of ``M`` (its scale factor is set to 0) when its linkage
    row is zero -- it is decoupled -- and otherwise forces ``t = 0``, since
    any ``t > 0`` would make ``S + t O`` indefinite in that coordinate.

    ``safety_margin`` shrinks ``t*`` by that relative amount so that the
    floating-point minimum eigenvalue of the result stays at or above
    ``min_eigenvalue``: the exact ``t*`` puts it there only up to the
    eigen-solver's roundoff. (The minimum eigenvalue is concave in ``t``, so
    the margin lifts it by at least ``safety_margin * min(S_ii)``.) It also
    applies when the computed ``t*`` lands just above 1, where roundoff can
    hide a structured covariance sitting just past the floor. The default,
    ``max(1e-9, 64 * eps)`` for the input dtype, is ``1e-9`` in float64 and
    about ``7.6e-6`` in float32, where ``1 - 1e-9`` rounds to 1.
    """
    dtype = structured.dtype
    off_diag = structured - diag_only
    shift = jnp.diagonal(diag_only) - jnp.asarray(min_eigenvalue, dtype=dtype)
    positive = shift > 0
    inv_sqrt = jnp.where(positive, 1.0 / jnp.sqrt(jnp.where(positive, shift, 1.0)), 0.0)
    scaled = -(inv_sqrt[:, None] * off_diag * inv_sqrt[None, :])
    lam_max = jnp.max(jnp.linalg.eigvalsh(symmetrize(scaled)))
    needs_shrink = lam_max > 0
    if safety_margin is None:
        safety_margin = max(1e-9, 64.0 * float(jnp.finfo(dtype).eps))
    t_exact = 1.0 / jnp.where(needs_shrink, lam_max, 1.0)
    t = jnp.where(needs_shrink, t_exact * (1.0 - safety_margin), 1.0)
    pinned = jnp.any((~positive) & (jnp.max(jnp.abs(off_diag), axis=1) > 0))
    t = jnp.where(pinned, 0.0, t)
    return jnp.clip(t, 0.0, 1.0)


@jax.jit
def project_correlated_noise_process_covariance(
    process_covariance: jax.Array,
    min_eigenvalue: float = 1e-8,
) -> jax.Array:
    """Project a covariance to the CNM block structure while preserving PSD.

    The Correlated Noise Model requires diagonal oscillator blocks
    ``sigma_k * I`` and symmetric off-diagonal scaled-rotation blocks.  A
    generic Kalman M-step produces an arbitrary covariance, so this projects the
    blocks back to the model family and, if needed, shrinks only the off-diagonal
    linkage blocks toward zero -- by the closed-form factor of
    :func:`_cnm_psd_shrink_factor` -- until the covariance is positive
    semidefinite with minimum eigenvalue ``min_eigenvalue``.  Fully traceable
    (one eigendecomposition, no host syncs) and jit-compiled, so per-EM-iteration
    callers pay one compile per covariance shape.
    """
    structured, diag_only = _cnm_structured_projection(
        process_covariance, min_eigenvalue
    )
    shrink = _cnm_psd_shrink_factor(structured, diag_only, min_eigenvalue)
    return jnp.where(
        shrink >= 1.0, structured, diag_only + shrink * (structured - diag_only)
    )


def constrain_correlated_noise_process_covariance(
    process_covariance: jax.Array,
    min_eigenvalue: float = 1e-8,
) -> jax.Array:
    """Return the exact PSD CNM projection of a residual covariance.

    CNM covariances are the real representation of a proper complex covariance:
    every diagonal 2x2 block is ``variance * I`` and every upper cross-block is
    ``coupling * R(phase)``, with its transpose in the lower block.  The
    orthogonal projection onto that linear family is the group average

    ``0.5 * (S + J @ S @ J.T)``,

    where ``J`` is block diagonal with a 90-degree rotation in each oscillator
    block.  Because this is an average of two PSD matrices, it preserves PSD
    whenever ``S`` is PSD.  It is also the exact constrained covariance M-step:
    for a CNM covariance ``Q``, ``Q**-1`` is in the same linear family, so the
    Gaussian covariance objective depends on ``S`` only through this projection.

    A final isotropic lift handles roundoff or slightly indefinite approximate
    GPB sufficient statistics without leaving the CNM family.
    """
    cov = jnp.asarray(process_covariance)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] % 2:
        raise ValueError(
            "process_covariance must be a square 2D matrix with even dimensions."
        )
    if min_eigenvalue < 0.0:
        raise ValueError("min_eigenvalue must be non-negative.")

    cov = symmetrize(cov)
    n_oscillators = cov.shape[0] // 2
    quarter_turn = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=cov.dtype)
    complex_structure = jnp.kron(jnp.eye(n_oscillators, dtype=cov.dtype), quarter_turn)
    constrained = 0.5 * (cov + complex_structure @ cov @ complex_structure.T)
    constrained = symmetrize(constrained)

    min_eig = jnp.min(jnp.linalg.eigvalsh(constrained))
    lift = jnp.maximum(
        jnp.asarray(min_eigenvalue, dtype=cov.dtype) - min_eig,
        jnp.asarray(0.0, dtype=cov.dtype),
    )
    return constrained + lift * jnp.eye(constrained.shape[0], dtype=cov.dtype)


def extract_correlated_noise_params_from_covariance(
    process_covariance: jax.Array,
    n_oscillators: int,
) -> dict:
    """Extract CNM scientific parameters from a structured process covariance."""
    blocks = _matrix_to_oscillator_blocks(process_covariance)
    if blocks.shape[0] != n_oscillators:
        raise ValueError(
            "n_oscillators does not match process_covariance shape: "
            f"{n_oscillators} vs {blocks.shape[0]}."
        )

    variance = jnp.zeros((n_oscillators,), dtype=process_covariance.dtype)
    phase_difference = jnp.zeros(
        (n_oscillators, n_oscillators), dtype=process_covariance.dtype
    )
    coupling_strength = jnp.zeros_like(phase_difference)

    for i in range(n_oscillators):
        variance = variance.at[i].set(0.5 * jnp.trace(blocks[i, i]))
        for j in range(i + 1, n_oscillators):
            scale, angle = _extract_scale_and_angle(blocks[i, j])
            coupling_strength = coupling_strength.at[i, j].set(scale)
            phase_difference = phase_difference.at[i, j].set(angle)

    return {
        "variance": variance,
        "phase_difference": phase_difference,
        "coupling_strength": coupling_strength,
    }


def extract_dim_params_from_matrix(
    A: jax.Array,
    sampling_freq: float,
    n_oscillators: int,
) -> dict:
    """Extract oscillator parameters from a DIM transition matrix.

    This function extracts the underlying oscillator parameters (damping, frequency,
    coupling strength, phase difference) from a transition matrix that has been
    constructed using construct_directed_influence_transition_matrix().

    The transition matrix has structure:
    - Diagonal blocks: damping * R(2π*freq/fs) - sum_incoming_coupling * I
    - Off-diagonal blocks: coupling_strength * R(phase_diff)

    Parameters
    ----------
    A : jax.Array, shape (2*n_osc, 2*n_osc)
        Transition matrix (assumed to have rotation block structure).
    sampling_freq : float
        Sampling frequency in Hz.
    n_oscillators : int
        Number of oscillators.

    Returns
    -------
    dict with keys:
        - damping: (n_osc,) - damping coefficients in (0, 1)
        - freq: (n_osc,) - frequencies in Hz
        - coupling_strength: (n_osc, n_osc) - coupling strengths (0 on diagonal)
        - phase_diff: (n_osc, n_osc) - phase differences (0 on diagonal)
    """
    # Reshape A into (n, n, 2, 2) blocks
    blocks = A.reshape(n_oscillators, 2, n_oscillators, 2).transpose(0, 2, 1, 3)

    # Extract scale and angle from all blocks at once via nested vmap
    _extract_row = jax.vmap(_extract_scale_and_angle)
    _extract_all = jax.vmap(_extract_row)
    all_scales, all_angles = _extract_all(blocks)
    # all_scales, all_angles: (n_oscillators, n_oscillators)

    # Off-diagonal entries are coupling parameters; zero out diagonal
    diag_idx = jnp.arange(n_oscillators)
    coupling_strength = all_scales.at[diag_idx, diag_idx].set(0.0)
    phase_diff = all_angles.at[diag_idx, diag_idx].set(0.0)

    # Diagonal blocks: damping * R(2π*freq/fs) - sum_incoming_coupling * I
    # Add back sum of incoming coupling to recover the intrinsic rotation
    sum_incoming = jnp.sum(coupling_strength, axis=1)  # (n_oscillators,)
    diag_blocks = blocks[diag_idx, diag_idx]  # (n_oscillators, 2, 2)
    adjusted = diag_blocks + sum_incoming[:, None, None] * IDENTITY_2x2[None, :, :]

    damping, angles = jax.vmap(_extract_scale_and_angle)(adjusted)

    # Convert angles to signed frequencies in [-fs/2, fs/2]. Keeping the sign is
    # necessary for construct(extract(A)) to preserve clockwise rotations.
    freq = angles * sampling_freq / (2 * jnp.pi)

    return {
        "damping": damping,
        "freq": freq,
        "coupling_strength": coupling_strength,
        "phase_diff": phase_diff,
    }


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

    _warn_if_rotation_projection_degenerate(transition_matrix)

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


def construct_stable_directed_influence_transition_stack(
    freqs: ArrayLike,
    damping_coef: ArrayLike,
    coupling_strength: ArrayLike,
    phase_difference: ArrayLike,
    sampling_freq: float,
    max_spectral_radius: float = 0.99,
) -> jax.Array:
    """Build every discrete state's DIM transition matrix under one stability scale.

    The global scale from :func:`compute_directed_influence_stability_scale` is
    applied to the *effective* damping and coupling used to build each
    ``A_j``; the intrinsic inputs are left untouched, so the construction is
    idempotent and ``A_j`` is reconstructable from the intrinsic parameters by
    re-applying the same scale.

    Parameters
    ----------
    freqs : ArrayLike, shape (n_oscillators,)
    damping_coef : ArrayLike, shape (n_oscillators,)
    coupling_strength : ArrayLike, shape (n_osc, n_osc, n_discrete_states)
    phase_difference : ArrayLike, shape (n_osc, n_osc, n_discrete_states)
    sampling_freq : float
    max_spectral_radius : float, default=0.99
        Target upper bound on each ``A_j``'s spectral radius.

    Returns
    -------
    jax.Array, shape (2 * n_oscillators, 2 * n_oscillators, n_discrete_states)
    """
    freqs_arr = jnp.asarray(freqs)
    scale = compute_directed_influence_stability_scale(
        freqs_arr,
        damping_coef,
        coupling_strength,
        sampling_freq,
        max_spectral_radius=max_spectral_radius,
        phase_difference=phase_difference,
    )
    effective_damping = jnp.asarray(damping_coef) * scale
    effective_coupling = jnp.asarray(coupling_strength) * scale
    return jax.vmap(
        lambda phase, coupling: construct_directed_influence_transition_matrix(
            freqs=freqs_arr,
            damping_coeffs=effective_damping,
            coupling_strengths=coupling,
            phase_diffs=phase,
            sampling_freq=sampling_freq,
        ),
        in_axes=(-1, -1),
        out_axes=-1,
    )(jnp.asarray(phase_difference), effective_coupling)


def project_transition_matrix_stack(
    transition_matrices: jax.Array, max_spectral_radius: float = 0.99
) -> jax.Array:
    """Project each state's transition matrix onto the coupled-oscillator family.

    Every ``A_j`` is projected block-wise to scaled-rotation structure
    (:func:`project_coupled_transition_matrix`) and then its spectral radius is
    clamped to at most ``max_spectral_radius``
    (:func:`~state_space_practice.utils.stabilize_transition_matrix` with
    ``block_size=2``). The clamp is local to the strongly connected components
    of the oscillator coupling graph: for uncoupled (or one-directionally
    coupled) oscillators only the offending oscillator block is rescaled, so a
    single unstable rhythm does not damp every other rhythm; for fully coupled
    oscillators it is the uniform scale. The clamp logs a warning (``logging``)
    reporting the radius and scale whenever it engages. It is computed on host
    (``eigvals`` has no accelerator lowering), so this runs eagerly.

    Parameters
    ----------
    transition_matrices : jax.Array, shape (n_latent, n_latent, n_discrete_states)
        Per-state transition matrices, ``n_latent = 2 * n_oscillators``.
    max_spectral_radius : float, default=0.99
        See :func:`~state_space_practice.utils.stabilize_transition_matrix` for
        how to choose it from the sampling rate and the narrowest bandwidth
        that must remain representable.

    Returns
    -------
    jax.Array, shape (2 * n_oscillators, 2 * n_oscillators, n_discrete_states)
    """
    return jnp.stack(
        [
            stabilize_transition_matrix(
                project_coupled_transition_matrix(transition_matrices[..., j]),
                max_spectral_radius=max_spectral_radius,
                block_size=2,
            )
            for j in range(transition_matrices.shape[-1])
        ],
        axis=-1,
    )


def extract_dim_params_from_matrix_stack(
    transition_matrices: jax.Array, sampling_freq: float, n_oscillators: int
) -> dict:
    """Extract shared DIM oscillator parameters from a per-state matrix stack.

    Frequency and damping are shared across discrete states in the directed
    influence model, so their per-state extractions are averaged; coupling
    strength and phase difference keep their trailing discrete-state axis.

    Parameters
    ----------
    transition_matrices : jax.Array, shape (n_latent, n_latent, n_discrete_states)
        Per-state transition matrices, ``n_latent = 2 * n_oscillators``.
    sampling_freq : float
    n_oscillators : int

    Returns
    -------
    dict
        ``freq`` and ``damping`` of shape ``(n_oscillators,)``;
        ``coupling_strength`` and ``phase_diff`` of shape
        ``(n_oscillators, n_oscillators, n_discrete_states)``.
    """
    per_state = [
        extract_dim_params_from_matrix(
            transition_matrices[..., j], sampling_freq, n_oscillators
        )
        for j in range(transition_matrices.shape[-1])
    ]
    stacked = {
        key: jnp.stack([p[key] for p in per_state], axis=-1)
        for key in ("freq", "damping", "coupling_strength", "phase_diff")
    }
    return {
        "freq": jnp.mean(stacked["freq"], axis=-1),
        "damping": jnp.mean(stacked["damping"], axis=-1),
        "coupling_strength": stacked["coupling_strength"],
        "phase_diff": stacked["phase_diff"],
    }


def optimize_dim_transition_params_joint_until_stationary(
    gamma1: jax.Array,
    beta: jax.Array,
    init_params: dict,
    sampling_freq: float,
    process_cov: jax.Array | None,
    max_spectral_radius: float,
    max_damping: float,
    max_restarts: int = 5,
    param_tol: float = 1e-8,
    optimizer: Callable[..., dict] | None = None,
) -> dict:
    """Run the joint DIM optimizer, restarting BFGS until it stops moving.

    :func:`~state_space_practice.switching_kalman.optimize_dim_transition_params_joint`
    runs one ``jax.scipy`` BFGS solve, which terminates as soon as a line
    search fails (``status=3``) -- this happens well before the gradient
    tolerance on ordinary DIM problems (e.g. after 13 iterations with an
    objective gradient of 67 in the damping coordinate), leaving the M-step
    at a non-stationary point.  Restarting from the returned point resets
    the inverse-Hessian approximation and the line-search bracket; each call
    never accepts a worse objective than its start (the optimizer backtracks),
    so the restarts can only improve the M-step objective.  Iteration stops
    once a restart moves no parameter by more than ``param_tol``.

    Parameters
    ----------
    gamma1, beta, init_params, sampling_freq, process_cov, max_spectral_radius,
    max_damping
        As for ``optimize_dim_transition_params_joint``.
    max_restarts : int, default=5
        Additional BFGS solves after the first.
    param_tol : float, default=1e-8
        Largest parameter change of a solve that counts as converged (a first
        solve that does not move needs no restart).
    optimizer : callable, optional
        The single-solve optimizer, called with the keyword arguments of
        ``optimize_dim_transition_params_joint`` (the default).

    Returns
    -------
    dict
        Shared ``damping``/``freq`` and state-specific
        ``coupling_strength``/``phase_diff``.
    """
    if optimizer is None:
        from state_space_practice.switching_kalman import (
            optimize_dim_transition_params_joint,
        )

        optimizer = optimize_dim_transition_params_joint
    solve: Callable[..., dict] = optimizer

    params = dict(init_params)
    for _ in range(max_restarts + 1):
        updated = solve(
            gamma1=gamma1,
            beta=beta,
            init_params=params,
            sampling_freq=sampling_freq,
            process_cov=process_cov,
            max_spectral_radius=max_spectral_radius,
            max_damping=max_damping,
        )
        change = max(
            float(
                jnp.max(jnp.abs(jnp.asarray(updated[key]) - jnp.asarray(params[key])))
            )
            for key in ("freq", "damping", "coupling_strength", "phase_diff")
        )
        params = updated
        if change <= param_tol:
            break
    return params


class DirectedInfluenceDynamicsMixin:
    """Transition-matrix machinery shared by the directed influence models.

    ``DirectedInfluenceModel`` (Gaussian observations) and
    ``DirectedInfluencePointProcessModel`` (spike observations) keep the same
    per-state coupled-oscillator transition matrices ``A_j`` in sync with the
    same public scientific parameters. The host class owns those parameters
    (``freqs``, ``damping_coef``, ``coupling_strength``, ``phase_difference``),
    the stability bound ``max_spectral_radius``, the
    ``use_reparameterized_mstep`` / ``update_continuous_transition_matrix``
    flags and the joint-optimizer warm-start cache ``_current_osc_params``;
    this mixin provides the shared rebuild, projection and synchronization
    steps that keep ``continuous_transition_matrix`` consistent with them.

    The public ``damping_coef`` / ``coupling_strength`` are *intrinsic*: the
    stability scale is applied only to the effective values used to build
    ``A``, never written back, so rebuilding is idempotent and damping does not
    drift toward zero across successive fits with strong coupling. (The
    standard-EM sync is the exception: there the public values are estimates
    extracted from ``A`` itself, see :meth:`_sync_coupling_from_transition_matrix`.)
    """

    freqs: jax.Array
    damping_coef: jax.Array
    coupling_strength: jax.Array
    phase_difference: jax.Array
    continuous_transition_matrix: jax.Array
    sampling_freq: float
    max_spectral_radius: float
    n_oscillators: int
    n_discrete_states: int
    use_reparameterized_mstep: bool
    update_continuous_transition_matrix: bool
    _current_osc_params: dict | None
    _pre_m_step_dynamics: dict | None
    process_cov: jax.Array

    _PUBLIC_DYNAMICS_ATTRS = (
        "continuous_transition_matrix",
        "freqs",
        "damping_coef",
        "coupling_strength",
        "phase_difference",
    )

    def _initialize_continuous_transition_matrix(self) -> None:
        """Build the per-state A from the intrinsic params via the stability scale.

        The initial matrices therefore already honor ``max_spectral_radius``
        before the first E-step runs.
        """
        self._rebuild_stable_transition_matrix()

    def _effective_dim_scale(self) -> jax.Array:
        """Global stability scale for the current (intrinsic) DIM parameters.

        ``continuous_transition_matrix`` is built from ``damping_coef * scale``
        and ``coupling_strength * scale``; reconstructing it from the public
        parameters requires re-applying this same scale.
        """
        return compute_directed_influence_stability_scale(
            self.freqs,
            self.damping_coef,
            self.coupling_strength,
            self.sampling_freq,
            max_spectral_radius=self.max_spectral_radius,
            phase_difference=self.phase_difference,
        )

    def _intrinsic_osc_params(self) -> dict:
        """The public scientific parameters in the joint optimizer's layout."""
        return {
            "freq": self.freqs,
            "damping": self.damping_coef,
            "coupling_strength": self.coupling_strength,
            "phase_diff": self.phase_difference,
        }

    def _rebuild_stable_transition_matrix(self) -> None:
        """Rebuild A from the intrinsic public params via the shared scale.

        See :func:`construct_stable_directed_influence_transition_stack`. Also
        refreshes the joint-optimizer warm-start cache so a later
        reparameterized M-step does not warm-start (or, on BFGS fallback,
        restore) stale pre-rebuild dynamics after the public params changed
        via SGD or standard EM. The cache is left untouched (``None``) before
        the first joint solve.
        """
        self.continuous_transition_matrix = (
            construct_stable_directed_influence_transition_stack(
                self.freqs,
                self.damping_coef,
                self.coupling_strength,
                self.phase_difference,
                self.sampling_freq,
                max_spectral_radius=self.max_spectral_radius,
            )
        )
        scale = float(self._effective_dim_scale())
        if scale < 1.0:
            # Logged (not warnings.warn): EM and SGD rebuild A every step.
            logger.warning(
                "DIM transition matrix exceeded max_spectral_radius=%g "
                "(radius=%.6g); effective damping and coupling were scaled by "
                "%.6g. The public parameters stay intrinsic; raise "
                "max_spectral_radius toward 1 - pi * bandwidth / sampling_freq "
                "for narrow-band rhythms.",
                self.max_spectral_radius,
                self.max_spectral_radius / scale,
                scale,
            )
        if self._current_osc_params is not None:
            self._current_osc_params = self._intrinsic_osc_params()

    def _update_public_oscillator_params(self) -> None:
        """Sync the joint optimizer solution to the public attributes.

        Frequency/damping are already shared (one joint solution); coupling and
        phase retain their discrete-state axis. No post-hoc averaging.
        """
        if self._current_osc_params is None:
            return

        self.freqs = self._current_osc_params["freq"]
        self.damping_coef = self._current_osc_params["damping"]
        self.coupling_strength = self._current_osc_params["coupling_strength"]
        self.phase_difference = self._current_osc_params["phase_diff"]

    def _project_parameters(self) -> None:
        """Project A onto the coupled-oscillator family and enforce stability.

        The unconstrained switching Kalman M-step can leave the directed
        influence model family, so this projection is a hard structural
        constraint (see :func:`project_transition_matrix_stack`). Returns
        without changes when ``A`` is not being learned
        (``update_continuous_transition_matrix`` is False) or when the
        reparameterized M-step already produced a valid ``A`` by construction.
        """
        previous = getattr(self, "_pre_m_step_dynamics", None)
        self._pre_m_step_dynamics = None

        if self.use_reparameterized_mstep:
            return

        if not self.update_continuous_transition_matrix:
            return

        self.continuous_transition_matrix = project_transition_matrix_stack(
            self.continuous_transition_matrix, self.max_spectral_radius
        )
        self._sync_coupling_from_transition_matrix()
        if previous is not None:
            self._keep_previous_dynamics_if_objective_decreased(previous)

    # --- Generalized-EM safeguard for the standard (projected) M-step -------

    def _remember_pre_m_step_dynamics(self) -> None:
        """Record ``A`` and its public parameters before a standard M-step.

        The next :meth:`_project_parameters` compares the projected ``A``
        against this previous iterate on the M-step objective (see
        :meth:`_keep_previous_dynamics_if_objective_decreased`).
        """
        self._pre_m_step_dynamics = {
            name: getattr(self, name) for name in self._PUBLIC_DYNAMICS_ATTRS
        }

    def _transition_objective(self, transition_matrix: jax.Array) -> float:
        """A-dependent part of the expected complete-data log-likelihood.

        ``-1/2 sum_j tr(Q_j^{-1} (A_j Gamma1_j A_j^T - A_j Beta_j^T - Beta_j A_j^T))``
        with the current E-step's transition statistics (the ``A``-free
        ``Gamma2`` and ``log|Q|`` terms are omitted; ``Q`` is fixed in DIM).
        """
        from state_space_practice.switching_kalman import (
            compute_transition_q_function,
            compute_transition_sufficient_stats,
        )

        gamma1, beta = compute_transition_sufficient_stats(
            state_cond_smoother_means=self.smoother_state_cond_mean,  # type: ignore[attr-defined]
            state_cond_smoother_covs=self.smoother_state_cond_cov,  # type: ignore[attr-defined]
            smoother_joint_discrete_state_prob=self.smoother_joint_discrete_state_prob,  # type: ignore[attr-defined]
            pair_cond_smoother_cross_cov=self.smoother_pair_cond_cross_cov,  # type: ignore[attr-defined]
            pair_cond_smoother_means=self.smoother_pair_cond_means,  # type: ignore[attr-defined]
            pair_cond_smoother_covs=getattr(self, "smoother_pair_cond_covs", None),
            next_pair_cond_smoother_means=getattr(
                self, "smoother_next_pair_cond_means", None
            ),
        )
        negative = jax.vmap(compute_transition_q_function, in_axes=-1)(
            transition_matrix, gamma1, beta, self.process_cov
        )
        return -float(jnp.sum(negative))

    def _keep_previous_dynamics_if_objective_decreased(self, previous: dict) -> None:
        """Make the standard DIM M-step a generalized EM step.

        The standard M-step solves the unconstrained ``A*`` and then projects
        it (Frobenius-closest scaled rotations, spectral clamp, shared
        frequency/damping averaged across states). That projection ignores
        the ``Q^{-1} (x) Gamma1`` metric of the objective, so the projected
        ``A`` can score *below* the previous iterate -- e.g. by 0.2-0.8 nats
        when EM starts from the constrained optimum. When it does, the
        previous (already valid) dynamics are kept, so the M-step never
        decreases the expected complete-data log-likelihood.
        """
        new_objective = self._transition_objective(self.continuous_transition_matrix)
        old_objective = self._transition_objective(
            previous["continuous_transition_matrix"]
        )
        slack = 1e-10 * max(1.0, abs(old_objective))
        if new_objective >= old_objective - slack:
            return
        logger.warning(
            "DIM standard M-step: the projected transition matrix lowers the "
            "M-step objective (%.6g -> %.6g); keeping the previous dynamics. "
            "If this repeats every iteration the dynamics never move; consider "
            "use_reparameterized_mstep=True.",
            old_objective,
            new_objective,
        )
        for name, value in previous.items():
            setattr(self, name, value)
        if self._current_osc_params is not None:
            self._current_osc_params = self._intrinsic_osc_params()

    def _sync_coupling_from_transition_matrix(self) -> None:
        """Sync all four scientific params from the current transition matrix.

        Called after the standard EM projection so the public
        frequency/damping/coupling/phase reflect the fitted A -- not just the
        initial values. ``A`` is then rebuilt from the public params so it
        stays reconstructable from them. Extracting only coupling/phase would
        leave frequency/damping stale, so ``A`` could not be reconstructed.

        The projection has already clamped each state's *actual* spectral
        radius and the extract -> construct round-trip of a single state is
        exact, so ``A`` is not re-scaled here. Only when averaging the shared
        frequency/damping across disagreeing states pushes a rebuilt ``A_j``
        past the bound are the extracted damping and coupling scaled by the
        exact factor -- written into the public parameters, which here are
        estimates of ``A`` rather than user-supplied intrinsic values -- so the
        public parameters reconstruct ``A`` with a stability scale of 1.
        """
        params = extract_dim_params_from_matrix_stack(
            self.continuous_transition_matrix, self.sampling_freq, self.n_oscillators
        )
        self.freqs = params["freq"]
        self.damping_coef = params["damping"]
        self.coupling_strength = params["coupling_strength"]
        self.phase_difference = params["phase_diff"]
        scale = self._effective_dim_scale()
        self.damping_coef = self.damping_coef * scale
        self.coupling_strength = self.coupling_strength * scale
        self._rebuild_stable_transition_matrix()
