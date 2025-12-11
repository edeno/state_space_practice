import jax
import jax.numpy as jnp
import pytest

from state_space_practice.oscillator_utils import (
    IDENTITY_2x2,
    ZEROS_2x2,
    _compute_coupled_oscillator_block,
    _compute_coupling_transition_block,
    _compute_intrinsic_oscillation_block,
    _get_rotation_matrix,
    _get_scaling_factor,
    _project_to_closest_rotation,
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


def test_get_block_slice():
    rows, cols = get_block_slice(1, 2)
    assert rows == slice(2, 4)
    assert cols == slice(4, 6)


def test__get_rotation_matrix_identity():
    mat = _get_rotation_matrix(0.0)
    expected = jnp.eye(2)
    assert jnp.allclose(mat, expected, atol=1e-7)


def test__get_rotation_matrix_pi_over_2():
    mat = _get_rotation_matrix(jnp.pi / 2)
    expected = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    assert jnp.allclose(mat, expected, atol=1e-7)


def test__compute_intrinsic_oscillation_block_valid():
    block = _compute_intrinsic_oscillation_block(0.0, 1.0)
    assert jnp.allclose(block, jnp.eye(2), atol=1e-7)


def test__compute_coupled_oscillator_block():
    block = _compute_coupled_oscillator_block(0.0, 1.0, 0.5)
    expected = jnp.eye(2) * 0.5
    assert jnp.allclose(block, expected, atol=1e-7)


def test__compute_coupling_transition_block_zero_strength():
    block = _compute_coupling_transition_block(0.0, 0.0)
    assert jnp.allclose(block, jnp.zeros((2, 2)), atol=1e-7)


def test__compute_coupling_transition_block_nonzero():
    block = _compute_coupling_transition_block(jnp.pi, 2.0)
    expected = 2.0 * jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    assert jnp.allclose(block, expected, atol=1e-6)


def test_construct_common_oscillator_transition_matrix():
    freqs = jnp.array([0.0, 0.0])
    coefs = jnp.array([1.0, 0.5])
    mat = construct_common_oscillator_transition_matrix(freqs, coefs)
    expected = jax.scipy.linalg.block_diag(jnp.eye(2), 0.5 * jnp.eye(2))
    assert jnp.allclose(mat, expected, atol=1e-7)


def test_construct_common_oscillator_transition_matrix_shape_error():
    freqs = jnp.array([0.0, 0.0])
    coefs = jnp.array([1.0])
    with pytest.raises(ValueError):
        construct_common_oscillator_transition_matrix(freqs, coefs)


def test_construct_common_oscillator_process_covariance():
    var = jnp.array([2.0, 3.0])
    mat = construct_common_oscillator_process_covariance(var)
    expected = jax.scipy.linalg.block_diag(2.0 * jnp.eye(2), 3.0 * jnp.eye(2))
    assert jnp.allclose(mat, expected, atol=1e-7)


def test_construct_correlated_noise_process_covariance():
    var = jnp.array([1.0, 2.0])
    phase = jnp.zeros((2, 2))
    coupling = jnp.zeros((2, 2))
    mat = construct_correlated_noise_process_covariance(var, phase, coupling)
    expected = jnp.block(
        [[1.0 * jnp.eye(2), jnp.zeros((2, 2))], [jnp.zeros((2, 2)), 2.0 * jnp.eye(2)]]
    )
    assert jnp.allclose(mat, expected, atol=1e-7)


def test_construct_correlated_noise_measurement_matrix():
    mat = construct_correlated_noise_measurement_matrix(2)
    expected = jnp.zeros((2, 4)).at[0, 0:2].set([1.0, 0.0]).at[1, 2:4].set([1.0, 0.0])
    assert jnp.allclose(mat, expected, atol=1e-7)


def test_construct_directed_influence_transition_matrix_shape_error():
    freqs = jnp.array([1.0, 2.0])
    coupling = jnp.zeros((2, 2))
    phase = jnp.zeros((2, 2))
    # Wrong shape for damping
    with pytest.raises(ValueError):
        construct_directed_influence_transition_matrix(
            freqs, jnp.array([0.9]), coupling, phase
        )


def test_construct_directed_influence_measurement_matrix():
    mat = construct_directed_influence_measurement_matrix(2)
    coeff = 1 / jnp.sqrt(2)
    expected = jnp.zeros((2, 4)).at[0, 0:2].set(coeff).at[1, 2:4].set(coeff)
    assert jnp.allclose(mat, expected, atol=1e-7)


def test__get_scaling_factor():
    s = jnp.array([4.0, 9.0])
    scale = _get_scaling_factor(s)
    assert jnp.isclose(scale, 6.0)


def test__project_to_closest_rotation_identity():
    mat = jnp.eye(2)
    projected = _project_to_closest_rotation(mat)
    assert jnp.allclose(projected, mat, atol=1e-7)


def test__project_to_closest_rotation_general():
    # A matrix with scaling and some non-rotation
    mat = jnp.array([[1.5, 0.5], [-0.5, 1.0]])
    U, s, Vh = jnp.linalg.svd(mat)
    scale = jnp.sqrt(s[0] * s[1])
    expected = scale * (U @ Vh)
    projected = _project_to_closest_rotation(mat)
    assert jnp.allclose(projected, expected, atol=1e-7)


def test__project_to_closest_rotation_pure_rotation():
    """
    Tests that a pure rotation matrix projects to itself (with scale=1).
    """
    mat = _get_rotation_matrix(jnp.pi / 3)
    projected = _project_to_closest_rotation(mat)
    assert jnp.allclose(projected, mat, atol=1e-7)


def test_project_matrix_blockwise():
    mat = jnp.eye(4)
    projected = project_matrix_blockwise(mat)
    assert jnp.allclose(projected, mat, atol=1e-7)


def test_project_matrix_blockwise_general():
    """
    Tests the blockwise projection with a non-identity 4x4 matrix.
    """
    block1 = jnp.array([[1.5, 0.5], [-0.5, 1.0]])  # General matrix
    block2 = jnp.array([[0.0, -2.0], [2.0, 0.0]])  # Scaled rotation (scale=2)

    # Construct a block matrix (off-diagonals are zero)
    mat = jnp.block([[block1, ZEROS_2x2], [ZEROS_2x2, block2]])

    # Calculate expected projection
    p1 = _project_to_closest_rotation(block1)
    p2 = _project_to_closest_rotation(block2)
    expected = jnp.block([[p1, ZEROS_2x2], [ZEROS_2x2, p2]])

    # Project using the function
    projected = project_matrix_blockwise(mat)

    assert jnp.allclose(projected, expected, atol=1e-7)


def test_project_coupled_transition_matrix_shape_error():
    mat = jnp.eye(3)
    with pytest.raises(ValueError):
        project_coupled_transition_matrix(mat)


def test_project_coupled_transition_matrix_simple_case():
    """
    Tests the coupled projection function with a simple case
    where the matrix is already constructed in a way that its blocks
    are scaled rotations or close to it, and checks if the algorithm
    behaves predictably (in this case, it should return the original).
    NOTE: This test primarily verifies the mechanics for a known case;
    it doesn't validate the algorithm's general correctness.
    """
    freqs = jnp.array([0.0, 0.0])
    damping_coeffs = jnp.array([1.0, 1.0])
    coupling_strengths = jnp.array([[0.0, 0.1], [0.2, 0.0]])
    phase_diffs = jnp.array([[0.0, 0.0], [jnp.pi / 2, 0.0]])

    # Construct the matrix
    mat = construct_directed_influence_transition_matrix(
        freqs, damping_coeffs, coupling_strengths, phase_diffs
    )

    # In this specific case, the blocks are already scaled rotations,
    # and the projection algorithm as written should return the original matrix.
    # Block (0, 0) = 0.9 * I. R1 = 0.1. P(0.9*I + 0.1*I) - 0.1*I = P(I) - 0.1*I = I - 0.1*I = 0.9*I
    # Block (1, 1) = 0.8 * I. R2 = 0.2. P(0.8*I + 0.2*I) - 0.2*I = P(I) - 0.2*I = I - 0.2*I = 0.8*I
    # Off-diagonal blocks are already scaled rotations, so P(Aij) = Aij.
    expected_matrix = mat

    # Project using the function
    projected = project_coupled_transition_matrix(mat)

    assert jnp.allclose(projected, expected_matrix, atol=1e-7)


def test_directed_influence_reduces_to_common_when_uncoupled():
    """
    Tests that the directed influence matrix equals the common oscillator
    matrix when all coupling strengths are zero.
    """
    n_oscillators = 3
    sampling_freq = 100.0
    key = jax.random.PRNGKey(42)

    # Generate some plausible random parameters
    freqs = jax.random.uniform(key, (n_oscillators,), minval=5.0, maxval=20.0)
    damping_coeffs = jax.random.uniform(key, (n_oscillators,), minval=0.9, maxval=0.99)

    # Set coupling to zero
    coupling_strengths = jnp.zeros((n_oscillators, n_oscillators))
    # Phase differences don't matter when coupling is zero, but set to zero
    phase_diffs = jnp.zeros((n_oscillators, n_oscillators))

    # Calculate using the directed influence function
    mat_directed = construct_directed_influence_transition_matrix(
        freqs, damping_coeffs, coupling_strengths, phase_diffs, sampling_freq
    )

    # Calculate using the common (uncoupled) function
    mat_common = construct_common_oscillator_transition_matrix(
        freqs, damping_coeffs, sampling_freq
    )

    # The two matrices should be identical
    assert jnp.allclose(mat_directed, mat_common, atol=1e-7)


# Define test cases: (name, freqs, damping, coupling, phase, expected_func)
test_cases = [
    (
        "Simple 2-Osc Case (from previous)",
        jnp.array([0.0, 0.0]),  # freqs
        jnp.array([1.0, 1.0]),  # damping
        jnp.array([[0.0, 0.1], [0.2, 0.0]]),  # coupling
        jnp.array([[0.0, 0.0], [jnp.pi / 2, 0.0]]),  # phase
        lambda: jnp.block(  # expected
            [
                [0.9 * IDENTITY_2x2, 0.1 * IDENTITY_2x2],
                [
                    0.2 * jnp.array([[0.0, -1.0], [1.0, 0.0]]),
                    0.8 * IDENTITY_2x2,
                ],
            ]
        ),
    ),
    (
        "Single Oscillator (No Coupling)",
        jnp.array([10.0]),
        jnp.array([0.95]),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
        lambda: _compute_coupled_oscillator_block(10.0, 0.95, 0.0, 1.0),
    ),
    (
        "Two Oscillators - One Way Coupling",
        jnp.array([5.0, 5.0]),
        jnp.array([0.9, 0.9]),
        jnp.array([[0.0, 0.1], [0.0, 0.0]]),  # Only 2 -> 1 coupling
        jnp.array([[0.0, jnp.pi / 4], [0.0, 0.0]]),
        lambda: jnp.block(
            [
                [
                    _compute_coupled_oscillator_block(5.0, 0.9, 0.1, 1.0),
                    _compute_coupling_transition_block(jnp.pi / 4, 0.1),
                ],
                [
                    _compute_coupling_transition_block(0.0, 0.0),
                    _compute_coupled_oscillator_block(5.0, 0.9, 0.0, 1.0),
                ],
            ]
        ),
    ),
    # Add more complex cases as needed
]


@pytest.mark.parametrize(
    "name, freqs, damping, coupling, phase, expected_func", test_cases
)
def test_directed_influence_parametrized(
    name, freqs, damping, coupling, phase, expected_func
):
    """
    Tests construct_directed_influence_transition_matrix with various
    parameter sets.
    """
    sampling_freq = 1.0  # Keep it simple for these tests or add as param

    # Calculate using the function
    mat_calculated = construct_directed_influence_transition_matrix(
        freqs, damping, coupling, phase, sampling_freq
    )

    # Get the expected result
    mat_expected = expected_func()

    assert mat_calculated.shape == mat_expected.shape
    assert jnp.allclose(mat_calculated, mat_expected, atol=1e-5)


# ============================================================================
# Tests for extract_dim_params_from_matrix
# ============================================================================


def test_extract_dim_params_roundtrip_simple():
    """
    Tests that extract_dim_params_from_matrix can recover parameters
    used to construct a DIM transition matrix (no coupling case).
    """
    n_osc = 2
    sampling_freq = 100.0
    freqs = jnp.array([8.0, 12.0])
    damping = jnp.array([0.95, 0.90])
    coupling = jnp.zeros((n_osc, n_osc))
    phase = jnp.zeros((n_osc, n_osc))

    # Construct matrix
    A = construct_directed_influence_transition_matrix(
        freqs, damping, coupling, phase, sampling_freq
    )

    # Extract params
    params = extract_dim_params_from_matrix(A, sampling_freq, n_osc)

    # Check roundtrip
    assert jnp.allclose(params["damping"], damping, atol=1e-4)
    assert jnp.allclose(params["freq"], freqs, atol=0.5)  # freq recovery is less precise
    assert jnp.allclose(params["coupling_strength"], coupling, atol=1e-4)


def test_extract_dim_params_roundtrip_with_coupling():
    """
    Tests parameter extraction with non-zero coupling.
    """
    n_osc = 2
    sampling_freq = 100.0
    freqs = jnp.array([10.0, 15.0])
    damping = jnp.array([0.95, 0.95])
    coupling = jnp.array([[0.0, 0.1], [0.05, 0.0]])
    phase = jnp.array([[0.0, jnp.pi / 4], [jnp.pi / 2, 0.0]])

    # Construct matrix
    A = construct_directed_influence_transition_matrix(
        freqs, damping, coupling, phase, sampling_freq
    )

    # Extract params
    params = extract_dim_params_from_matrix(A, sampling_freq, n_osc)

    # Check coupling strength recovery (off-diagonal)
    assert jnp.allclose(params["coupling_strength"][0, 1], coupling[0, 1], atol=1e-4)
    assert jnp.allclose(params["coupling_strength"][1, 0], coupling[1, 0], atol=1e-4)

    # Diagonal should be zero
    assert params["coupling_strength"][0, 0] == 0.0
    assert params["coupling_strength"][1, 1] == 0.0


def test_extract_dim_params_reconstructs_matrix():
    """
    Tests that extracted parameters can reconstruct the original matrix.
    """
    n_osc = 2
    sampling_freq = 100.0
    freqs = jnp.array([8.0, 12.0])
    damping = jnp.array([0.95, 0.92])
    coupling = jnp.array([[0.0, 0.05], [0.08, 0.0]])
    phase = jnp.array([[0.0, jnp.pi / 6], [jnp.pi / 3, 0.0]])

    # Construct original matrix
    A_original = construct_directed_influence_transition_matrix(
        freqs, damping, coupling, phase, sampling_freq
    )

    # Extract params
    params = extract_dim_params_from_matrix(A_original, sampling_freq, n_osc)

    # Reconstruct matrix
    A_reconstructed = construct_directed_influence_transition_matrix(
        params["freq"],
        params["damping"],
        params["coupling_strength"],
        params["phase_diff"],
        sampling_freq,
    )

    # Should be close
    assert jnp.allclose(A_reconstructed, A_original, atol=1e-4)
