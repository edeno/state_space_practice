import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.oscillator_utils import (
    DirectedInfluenceDynamicsMixin,
    IDENTITY_2x2,
    ZEROS_2x2,
    _cnm_psd_shrink_factor,
    _cnm_structured_projection,
    _compute_coupled_oscillator_block,
    _compute_coupling_transition_block,
    _compute_intrinsic_oscillation_block,
    _get_rotation_matrix,
    _get_scaling_factor,
    _project_to_closest_rotation,
    _scatter_block_diagonal,
    canonicalize_correlated_noise_pair_parameters,
    compute_directed_influence_stability_scale,
    constrain_correlated_noise_process_covariance,
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
    construct_correlated_noise_measurement_matrix,
    construct_correlated_noise_process_covariance,
    construct_directed_influence_measurement_matrix,
    construct_directed_influence_transition_matrix,
    construct_stable_directed_influence_transition_stack,
    extract_dim_params_from_matrix,
    extract_dim_params_from_matrix_stack,
    get_block_slice,
    project_correlated_noise_process_covariance,
    project_coupled_transition_matrix,
    project_matrix_blockwise,
    project_transition_matrix_stack,
)
from state_space_practice.utils import stabilize_transition_matrix


def test_constrain_correlated_noise_covariance_is_structured_psd() -> None:
    """The exact CNM constraint keeps PSD and yields reconstructable blocks."""
    raw_factor = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.3, 0.8, 0.0, 0.0],
            [0.2, -0.1, 0.7, 0.0],
            [0.4, 0.2, -0.3, 0.6],
        ]
    )
    residual_cov = raw_factor @ raw_factor.T
    constrained = constrain_correlated_noise_process_covariance(residual_cov)

    np.testing.assert_allclose(constrained, constrained.T, atol=1e-12)
    assert float(jnp.min(jnp.linalg.eigvalsh(constrained))) >= 1e-8 - 1e-12

    blocks = constrained.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
    for i in range(2):
        np.testing.assert_allclose(
            blocks[i, i], 0.5 * jnp.trace(blocks[i, i]) * jnp.eye(2), atol=1e-12
        )
    upper = blocks[0, 1]
    np.testing.assert_allclose(upper[0, 0], upper[1, 1], atol=1e-12)
    np.testing.assert_allclose(upper[0, 1], -upper[1, 0], atol=1e-12)
    np.testing.assert_allclose(blocks[1, 0], upper.T, atol=1e-12)


def test_get_block_slice():
    rows, cols = get_block_slice(1, 2)
    assert rows == slice(2, 4)
    assert cols == slice(4, 6)


@pytest.mark.parametrize("n_blocks", [1, 2, 3])
def test_scatter_block_diagonal_matches_block_diag(n_blocks):
    """_scatter_block_diagonal must produce the same result as jax.scipy.linalg.block_diag."""
    key = jax.random.PRNGKey(42)
    blocks = jax.random.normal(key, (n_blocks, 2, 2))
    result = _scatter_block_diagonal(blocks)
    expected = jax.scipy.linalg.block_diag(*[blocks[k] for k in range(n_blocks)])
    np.testing.assert_allclose(result, expected, atol=1e-14)
    # Off-diagonal blocks must be zero
    for i in range(n_blocks):
        for j in range(n_blocks):
            if i != j:
                block = result[2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1)]
                assert jnp.allclose(block, 0.0), f"Off-diag block ({i},{j}) non-zero"


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


def test__compute_coupling_transition_block_has_gradient_at_zero_strength():
    """Zero coupling should still be able to learn away from zero under SGD."""
    phase = 0.7

    def block_entry(coupling):
        return _compute_coupling_transition_block(phase, coupling)[0, 0]

    grad_at_zero = jax.grad(block_entry)(0.0)
    np.testing.assert_allclose(grad_at_zero, np.cos(phase), atol=1e-7)


def test__compute_coupling_transition_block_zero_strength_masks_nan_phase():
    """An ignored phase on an exactly zero coupling should not create NaNs."""
    block = _compute_coupling_transition_block(jnp.nan, 0.0)
    assert jnp.all(jnp.isfinite(block))
    np.testing.assert_allclose(block, jnp.zeros((2, 2)), atol=0.0)


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


def test_construct_correlated_noise_process_covariance_nonzero_coupling():
    """Q is symmetric: upper cross-block = coupling*R(phase), lower = its transpose.

    The directed inputs are asymmetric (``phase[i,j] != phase[j,i]``), but a
    covariance must be symmetric, so the constructor uses only the strict upper
    triangle and mirrors it as the transpose.
    """
    key = jax.random.PRNGKey(99)
    n_osc = 3
    var = jax.random.uniform(key, (n_osc,), minval=0.5, maxval=2.0)
    phase = jax.random.uniform(key, (n_osc, n_osc), minval=0, maxval=jnp.pi)
    coupling = jax.random.uniform(key, (n_osc, n_osc), minval=0.1, maxval=0.5)

    mat = construct_correlated_noise_process_covariance(var, phase, coupling)

    # Diagonal blocks are variance * I.
    for i in range(n_osc):
        block = mat[2 * i : 2 * (i + 1), 2 * i : 2 * (i + 1)]
        np.testing.assert_allclose(block, var[i] * jnp.eye(2), atol=1e-13)

    # Off-diagonal blocks: the strict upper triangle is coupling * R(phase); the
    # lower triangle is the transpose of its upper partner (tie-blocks symmetry).
    from state_space_practice.oscillator_utils import _compute_coupling_transition_block

    for i in range(n_osc):
        for j in range(n_osc):
            if i == j:
                continue
            block = mat[2 * i : 2 * (i + 1), 2 * j : 2 * (j + 1)]
            u, v = min(i, j), max(i, j)
            upper = _compute_coupling_transition_block(phase[u, v], coupling[u, v])
            expected = upper if i < j else upper.T
            np.testing.assert_allclose(block, expected, atol=1e-13)

    # The assembled covariance must be symmetric (this fails on the old
    # directed-block constructor, which used phase[i,j] and phase[j,i]
    # independently).
    np.testing.assert_allclose(mat, mat.T, atol=1e-13)


def test_canonicalize_correlated_noise_pair_parameters_accepts_lower_only():
    phase = jnp.zeros((2, 2, 2)).at[1, 0, :].set(-0.7)
    coupling = jnp.zeros((2, 2, 2)).at[1, 0, :].set(0.2)

    canon_phase, canon_coupling = canonicalize_correlated_noise_pair_parameters(
        phase, coupling
    )

    np.testing.assert_allclose(canon_phase[0, 1, :], 0.7, atol=1e-12)
    np.testing.assert_allclose(canon_coupling[0, 1, :], 0.2, atol=1e-12)
    np.testing.assert_allclose(canon_phase[1, 0, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(canon_coupling[1, 0, :], 0.0, atol=1e-12)


def test_canonicalize_correlated_noise_pair_parameters_accepts_mirrored_full_pair():
    phase = jnp.zeros((2, 2)).at[0, 1].set(0.7).at[1, 0].set(-0.7)
    coupling = jnp.zeros((2, 2)).at[0, 1].set(0.2).at[1, 0].set(0.2)

    canon_phase, canon_coupling = canonicalize_correlated_noise_pair_parameters(
        phase, coupling
    )

    np.testing.assert_allclose(canon_phase, jnp.array([[0.0, 0.7], [0.0, 0.0]]))
    np.testing.assert_allclose(canon_coupling, jnp.array([[0.0, 0.2], [0.0, 0.0]]))


def test_canonicalize_correlated_noise_pair_parameters_rejects_conflict():
    phase = jnp.zeros((2, 2)).at[0, 1].set(0.7).at[1, 0].set(0.4)
    coupling = jnp.zeros((2, 2)).at[0, 1].set(0.2).at[1, 0].set(0.2)

    with pytest.raises(ValueError, match="Conflicting correlated-noise"):
        canonicalize_correlated_noise_pair_parameters(phase, coupling)


def test_canonicalize_correlated_noise_pair_parameters_rejects_diagonal_coupling():
    phase = jnp.zeros((2, 2))
    coupling = jnp.eye(2) * 0.2

    with pytest.raises(ValueError, match="diagonal"):
        canonicalize_correlated_noise_pair_parameters(phase, coupling)


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


def test_construct_directed_influence_transition_matrix_accepts_numpy_inputs():
    freqs = np.array([5.0, 9.0])
    damping = np.array([0.95, 0.9])
    coupling = np.array([[0.0, 0.1], [0.05, 0.0]])
    phase = np.array([[0.0, 0.4], [-0.7, 0.0]])

    from_numpy = construct_directed_influence_transition_matrix(
        freqs, damping, coupling, phase, sampling_freq=100.0
    )
    from_jax = construct_directed_influence_transition_matrix(
        jnp.asarray(freqs),
        jnp.asarray(damping),
        jnp.asarray(coupling),
        jnp.asarray(phase),
        sampling_freq=100.0,
    )

    assert isinstance(from_numpy, jax.Array)
    # Guard: the coupled off-diagonal blocks are populated, so the comparison
    # covers the coupling and phase inputs, not just the intrinsic blocks.
    assert float(jnp.max(jnp.abs(from_jax[:2, 2:]))) > 0.0
    np.testing.assert_array_equal(np.asarray(from_numpy), np.asarray(from_jax))


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
    det_sign = jnp.where(jnp.linalg.det(U @ Vh) < 0.0, -1.0, 1.0)
    correction = jnp.diag(jnp.array([1.0, det_sign], dtype=mat.dtype))
    expected = scale * (U @ correction @ Vh)
    projected = _project_to_closest_rotation(mat)
    assert jnp.allclose(projected, expected, atol=1e-7)


def test__project_to_closest_rotation_rejects_reflection_solution():
    """The SVD projection must stay in scaled rotations, not reflections."""
    reflection = jnp.array([[1.0, 0.0], [0.0, -1.0]])
    projected = _project_to_closest_rotation(reflection)

    assert float(jnp.linalg.det(projected)) >= 0.0
    np.testing.assert_allclose(projected[0, 0], projected[1, 1], atol=1e-7)
    np.testing.assert_allclose(projected[0, 1], -projected[1, 0], atol=1e-7)


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


def test_project_coupled_transition_matrix_uses_scaled_rotation_blocks():
    """Projection must not return SVD reflections inside DIM oscillator blocks."""
    matrix = jnp.array(
        [
            [0.8, 0.4, 0.2, -0.7],
            [0.1, 0.6, 0.9, 0.3],
            [-0.5, 0.8, 0.7, -0.2],
            [0.4, -0.6, 0.5, 0.9],
        ]
    )

    projected = np.asarray(project_coupled_transition_matrix(matrix))

    for row in range(2):
        for col in range(2):
            block = projected[2 * row : 2 * row + 2, 2 * col : 2 * col + 2]
            np.testing.assert_allclose(block[0, 0], block[1, 1], atol=1e-8)
            np.testing.assert_allclose(block[0, 1], -block[1, 0], atol=1e-8)


def test_project_coupled_transition_matrix_matches_independent_block_projection():
    """The coupled projection is the per-block scaled-rotation projection."""
    matrix = jnp.array(
        [
            [0.8, -0.2, 0.4, 0.1],
            [0.2, 0.8, -0.1, 0.4],
            [0.3, -0.2, 0.7, 0.5],
            [0.2, 0.3, -0.5, 0.7],
        ]
    )
    projected = project_coupled_transition_matrix(matrix)
    blocks = matrix.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
    a = 0.5 * (blocks[..., 0, 0] + blocks[..., 1, 1])
    b = 0.5 * (blocks[..., 1, 0] - blocks[..., 0, 1])
    expected_blocks = jnp.stack(
        [
            jnp.stack([a, -b], axis=-1),
            jnp.stack([b, a], axis=-1),
        ],
        axis=-2,
    )
    expected = expected_blocks.transpose(0, 2, 1, 3).reshape(4, 4)

    np.testing.assert_allclose(projected, expected, atol=1e-7)


def test_project_correlated_noise_process_covariance_preserves_structure_and_psd():
    """CNM Q projection keeps scalar diagonal blocks and symmetric pair blocks."""
    covariance = jnp.array(
        [
            [0.2, 0.04, 0.6, -0.5],
            [0.04, 0.5, 0.2, 0.7],
            [0.6, 0.2, 0.3, -0.02],
            [-0.5, 0.7, -0.02, 0.4],
        ]
    )

    projected = np.asarray(project_correlated_noise_process_covariance(covariance))

    np.testing.assert_allclose(projected, projected.T, atol=1e-10)
    assert np.linalg.eigvalsh(projected).min() >= -1e-8

    for osc in range(2):
        block = projected[2 * osc : 2 * osc + 2, 2 * osc : 2 * osc + 2]
        np.testing.assert_allclose(block[0, 0], block[1, 1], atol=1e-10)
        np.testing.assert_allclose(block[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(block[1, 0], 0.0, atol=1e-10)

    upper = projected[0:2, 2:4]
    lower = projected[2:4, 0:2]
    np.testing.assert_allclose(upper[0, 0], upper[1, 1], atol=1e-10)
    np.testing.assert_allclose(upper[0, 1], -upper[1, 0], atol=1e-10)
    np.testing.assert_allclose(lower, upper.T, atol=1e-10)


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
    assert jnp.allclose(
        params["freq"], freqs, atol=0.5
    )  # freq recovery is less precise
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


def test_extract_dim_params_reconstructs_negative_frequency_matrix():
    """Signed recovered frequencies must preserve clockwise rotation blocks."""
    n_osc = 2
    sampling_freq = 100.0
    freqs = jnp.array([-8.0, 12.0])
    damping = jnp.array([0.95, 0.92])
    coupling = jnp.array([[0.0, 0.05], [0.08, 0.0]])
    phase = jnp.array([[0.0, jnp.pi / 6], [jnp.pi / 3, 0.0]])

    A_original = construct_directed_influence_transition_matrix(
        freqs, damping, coupling, phase, sampling_freq
    )
    params = extract_dim_params_from_matrix(A_original, sampling_freq, n_osc)
    A_reconstructed = construct_directed_influence_transition_matrix(
        params["freq"],
        params["damping"],
        params["coupling_strength"],
        params["phase_diff"],
        sampling_freq,
    )

    np.testing.assert_allclose(params["freq"][0], freqs[0], atol=1e-5)
    np.testing.assert_allclose(A_reconstructed, A_original, atol=1e-6)


def test_extract_dim_params_canonicalizes_negative_coupling_but_reconstructs_matrix():
    """Negative authored coupling is represented as magnitude plus shifted phase."""
    n_osc = 2
    sampling_freq = 100.0
    freqs = jnp.array([8.0, -12.0])
    damping = jnp.array([0.95, 0.92])
    coupling = jnp.array([[0.0, -0.05], [0.08, 0.0]])
    phase = jnp.array([[0.0, jnp.pi / 6], [jnp.pi / 3, 0.0]])

    A_original = construct_directed_influence_transition_matrix(
        freqs, damping, coupling, phase, sampling_freq
    )
    params = extract_dim_params_from_matrix(A_original, sampling_freq, n_osc)
    A_reconstructed = construct_directed_influence_transition_matrix(
        params["freq"],
        params["damping"],
        params["coupling_strength"],
        params["phase_diff"],
        sampling_freq,
    )

    assert jnp.all(params["coupling_strength"] >= 0.0)
    np.testing.assert_allclose(A_reconstructed, A_original, atol=1e-6)


# ============================================================================
# Tests for SVD fallback in _project_to_closest_rotation
# ============================================================================


class TestProjectToClosestRotationFallback:
    """Tests for the SVD failure fallback in _project_to_closest_rotation."""

    def test_nan_input_returns_damped_identity(self) -> None:
        """A matrix with NaN entries should produce a finite damped identity."""
        matrix = jnp.array([[jnp.nan, 0.0], [0.0, jnp.nan]])
        result = _project_to_closest_rotation(matrix)

        assert jnp.all(jnp.isfinite(result)), f"Result should be finite: {result}"
        # Should be a scaled identity (damped, no rotation)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[1, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[0, 0], result[1, 1], atol=1e-10)

    def test_inf_input_returns_finite_fallback(self) -> None:
        """A matrix with Inf entries should produce a finite result."""
        matrix = jnp.array([[jnp.inf, 0.0], [0.0, 1.0]])
        result = _project_to_closest_rotation(matrix)

        assert jnp.all(jnp.isfinite(result)), f"Result should be finite: {result}"

    def test_all_nan_uses_default_scale(self) -> None:
        """All-NaN matrix can't compute Frobenius norm; uses default 0.5."""
        matrix = jnp.full((2, 2), jnp.nan)
        result = _project_to_closest_rotation(matrix)

        assert jnp.all(jnp.isfinite(result))
        # Scale should be 0.5 (the default fallback)
        np.testing.assert_allclose(result[0, 0], 0.5, atol=1e-10)

    def test_well_conditioned_input_unchanged(self) -> None:
        """A valid scaled rotation should pass through SVD without fallback."""
        # [[a, -b], [b, a]] is already a scaled rotation
        matrix = jnp.array([[0.9, -0.3], [0.3, 0.9]])
        result = _project_to_closest_rotation(matrix)

        np.testing.assert_allclose(result, matrix, atol=1e-6)

    def test_fallback_produces_valid_block_structure(self) -> None:
        """Fallback identity has scaled rotation structure [[a, 0], [0, a]]."""
        matrix = jnp.array([[jnp.nan, 1.0], [1.0, jnp.nan]])
        result = _project_to_closest_rotation(matrix)

        # Identity is a special case of [[a, -b], [b, a]] with b=0
        np.testing.assert_allclose(result[0, 1], -result[1, 0], atol=1e-10)
        np.testing.assert_allclose(result[0, 0], result[1, 1], atol=1e-10)


class TestProjectCoupledTransitionMatrixPathological:
    """Tests for project_coupled_transition_matrix with pathological inputs."""

    def test_unstable_matrix_produces_finite_output(self) -> None:
        """An unstable transition matrix should still project to finite result."""
        # Large diagonal = spectral radius > 1
        A = jnp.array(
            [
                [2.0, -0.5, 0.1, 0.0],
                [0.5, 2.0, 0.0, 0.1],
                [0.1, 0.0, 1.5, -0.3],
                [0.0, 0.1, 0.3, 1.5],
            ]
        )
        result = project_coupled_transition_matrix(A)
        assert jnp.all(jnp.isfinite(result))

    def test_near_singular_matrix_produces_finite_output(self) -> None:
        """A nearly singular matrix should project without NaN."""
        A = jnp.array(
            [
                [1e-15, -1e-15, 0.0, 0.0],
                [1e-15, 1e-15, 0.0, 0.0],
                [0.0, 0.0, 1e-15, -1e-15],
                [0.0, 0.0, 1e-15, 1e-15],
            ]
        )
        result = project_coupled_transition_matrix(A)
        assert jnp.all(jnp.isfinite(result))

    def test_finite_input_emits_no_false_fallback_warning(self, capfd) -> None:
        """A finite input must not emit the non-finite fallback telemetry.

        The per-block ``debug_print_if`` used to live inside
        ``_project_to_closest_rotation``, which runs under ``jax.vmap`` where
        ``lax.cond`` lowers to ``select`` and both branches execute -- so the
        "non-finite" warning fired for every block on every call, even on
        perfectly finite input. Capture at the fd level (``capfd``) because
        ``jax.debug.print`` writes past ``sys.stdout``.
        """
        A = jnp.array(
            [
                [0.9, -0.3, 0.05, 0.0],
                [0.3, 0.9, 0.0, 0.05],
                [0.05, 0.0, 0.8, -0.2],
                [0.0, 0.05, 0.2, 0.8],
            ]
        )
        assert jnp.all(jnp.isfinite(A))  # guard: input is finite
        capfd.readouterr()  # drop anything buffered before the call
        result = project_coupled_transition_matrix(A)
        result.block_until_ready()
        jax.effects_barrier()  # flush pending jax.debug.print effects
        captured = capfd.readouterr()
        assert "non-finite" not in (captured.out + captured.err).lower(), (
            "finite input should not emit any non-finite fallback warning; "
            f"got: {captured.out + captured.err!r}"
        )

    def test_nonfinite_input_emits_fallback_warning(self, capfd) -> None:
        """A non-finite input must still emit exactly the fail-loud signal."""
        A = jnp.eye(4).at[0, 0].set(jnp.nan)
        capfd.readouterr()
        result = project_coupled_transition_matrix(A)
        result.block_until_ready()
        jax.effects_barrier()
        captured = capfd.readouterr()
        assert "non-finite" in (captured.out + captured.err).lower(), (
            "non-finite input should emit the damped-identity fallback warning"
        )


# ---------------------------------------------------------------------------
# Shared directed-influence transition-stack helpers
# ---------------------------------------------------------------------------


def _spectral_radius(A) -> float:
    return float(jnp.max(jnp.abs(jnp.linalg.eigvals(A))))


def _assert_scaled_rotation_blocks(A, n_oscillators: int, atol: float) -> None:
    for row in range(n_oscillators):
        for col in range(n_oscillators):
            block = A[2 * row : 2 * row + 2, 2 * col : 2 * col + 2]
            np.testing.assert_allclose(block[0, 0], block[1, 1], atol=atol)
            np.testing.assert_allclose(block[0, 1], -block[1, 0], atol=atol)


def test_project_transition_matrix_stack_projects_blocks_and_clamps_each_state():
    """Every state is projected to scaled-rotation blocks and clamped to the bound;
    a state already inside the family and the bound passes through unchanged."""
    n_osc = 2
    bound = 0.9
    raw = 3.0 * jax.random.normal(jax.random.PRNGKey(0), (2 * n_osc, 2 * n_osc, 3))
    stable = construct_common_oscillator_transition_matrix(
        freqs=jnp.array([8.0, 12.0]),
        damping_coef=jnp.array([0.8, 0.7]),
        sampling_freq=100.0,
    )
    raw = raw.at[..., 2].set(stable)
    # Guard: the clamp is exercised on the random states (non-vacuous below).
    for j in range(2):
        assert _spectral_radius(project_coupled_transition_matrix(raw[..., j])) > bound

    projected = project_transition_matrix_stack(raw, max_spectral_radius=bound)

    assert projected.shape == raw.shape
    for j in range(3):
        assert _spectral_radius(projected[..., j]) <= bound + 1e-8
        _assert_scaled_rotation_blocks(projected[..., j], n_osc, atol=1e-10)
        expected = stabilize_transition_matrix(
            project_coupled_transition_matrix(raw[..., j]),
            max_spectral_radius=bound,
            block_size=2,
            warn=False,
        )
        np.testing.assert_allclose(projected[..., j], expected, atol=1e-12)
    np.testing.assert_allclose(projected[..., 2], stable, atol=1e-10)


def test_construct_stable_directed_influence_transition_stack_applies_one_scale():
    """Each state is the DIM construction under the shared stability scale, so the
    stack honors the bound and is reconstructable from the intrinsic params."""
    freqs = jnp.array([8.0, 12.0])
    damping = jnp.array([0.95, 0.9])
    fs = 100.0
    strong = jnp.zeros((2, 2, 2)).at[0, 1, :].set(2.0).at[1, 0, 0].set(1.5)
    phase = jnp.zeros((2, 2, 2)).at[0, 1, :].set(0.4)
    bound = 0.9

    stack = construct_stable_directed_influence_transition_stack(
        freqs, damping, strong, phase, fs, max_spectral_radius=bound
    )
    scale = compute_directed_influence_stability_scale(
        freqs, damping, strong, fs, max_spectral_radius=bound, phase_difference=phase
    )

    assert float(scale) < 1.0  # guard: stabilization engaged
    assert stack.shape == (4, 4, 2)
    for j in range(2):
        assert _spectral_radius(stack[..., j]) <= bound + 1e-6
        expected = construct_directed_influence_transition_matrix(
            freqs, damping * scale, strong[..., j] * scale, phase[..., j], fs
        )
        np.testing.assert_allclose(stack[..., j], expected, atol=1e-12)

    # Weak coupling under the default bound (0.99 > max damping 0.95) needs no
    # scaling: the stack is the plain construction.
    weak = 0.01 * strong
    assert (
        float(
            compute_directed_influence_stability_scale(
                freqs, damping, weak, fs, phase_difference=phase
            )
        )
        == 1.0
    )
    stack_weak = construct_stable_directed_influence_transition_stack(
        freqs, damping, weak, phase, fs
    )
    for j in range(2):
        expected = construct_directed_influence_transition_matrix(
            freqs, damping, weak[..., j], phase[..., j], fs
        )
        np.testing.assert_allclose(stack_weak[..., j], expected, atol=1e-12)


def test_extract_dim_params_from_matrix_stack_recovers_shared_and_per_state_params():
    """Shared freq/damping come back averaged over states; coupling/phase keep
    their state axis."""
    freqs = jnp.array([6.0, 11.0])
    damping = jnp.array([0.9, 0.8])
    fs = 100.0
    coupling = jnp.zeros((2, 2, 2)).at[0, 1, 0].set(0.05).at[1, 0, 1].set(0.08)
    phase = jnp.zeros((2, 2, 2)).at[0, 1, 0].set(0.3).at[1, 0, 1].set(-0.7)
    stack = construct_stable_directed_influence_transition_stack(
        freqs, damping, coupling, phase, fs
    )

    params = extract_dim_params_from_matrix_stack(stack, fs, 2)

    np.testing.assert_allclose(params["freq"], freqs, atol=1e-8)
    np.testing.assert_allclose(params["damping"], damping, atol=1e-8)
    np.testing.assert_allclose(params["coupling_strength"], coupling, atol=1e-8)
    np.testing.assert_allclose(params["phase_diff"], phase, atol=1e-8)

    # Disagreeing per-state matrices: the shared params are the state average.
    other = construct_directed_influence_transition_matrix(
        freqs + 2.0, damping - 0.1, coupling[..., 1], phase[..., 1], fs
    )
    params_avg = extract_dim_params_from_matrix_stack(
        stack.at[..., 1].set(other), fs, 2
    )
    np.testing.assert_allclose(params_avg["freq"], freqs + 1.0, atol=1e-8)
    np.testing.assert_allclose(params_avg["damping"], damping - 0.05, atol=1e-8)


class _DIMHost(DirectedInfluenceDynamicsMixin):
    """Minimal host exposing the attributes the mixin documents as required."""

    def __init__(self, coupling, use_reparameterized_mstep=False):
        self.freqs = jnp.array([8.0, 12.0])
        self.damping_coef = jnp.array([0.95, 0.9])
        self.coupling_strength = coupling
        self.phase_difference = jnp.zeros_like(coupling)
        self.sampling_freq = 100.0
        self.max_spectral_radius = 0.9
        self.n_oscillators = 2
        self.n_discrete_states = coupling.shape[-1]
        self.use_reparameterized_mstep = use_reparameterized_mstep
        self.update_continuous_transition_matrix = True
        self._current_osc_params = None


def test_rejected_projected_dynamics_are_restored_and_logged_at_warning(caplog):
    """When the projected A lowers the M-step objective the previous dynamics
    are kept; if that happens every iteration A never moves, so it must be
    visible at the default log level."""
    host = _DIMHost(jnp.zeros((2, 2, 1)))
    old_A = jnp.eye(4)[..., None] * 0.5
    host.continuous_transition_matrix = jnp.eye(4)[..., None] * 0.8
    # Objective prefers the previous matrix.
    host._transition_objective = lambda A: -float(jnp.sum((A - old_A) ** 2))
    with caplog.at_level("WARNING", logger="state_space_practice.oscillator_utils"):
        host._keep_previous_dynamics_if_objective_decreased(
            {"continuous_transition_matrix": old_A}
        )
    np.testing.assert_array_equal(host.continuous_transition_matrix, old_A)
    records = [r for r in caplog.records if "keeping the previous" in r.getMessage()]
    assert records and records[0].levelname == "WARNING"


def test_directed_influence_mixin_projection_syncs_public_params_and_rebuilds():
    """Standard-EM projection leaves A reconstructable from the synced public
    params; the reparameterized path leaves A untouched."""
    strong = jnp.zeros((2, 2, 2)).at[0, 1, :].set(2.0).at[1, 0, :].set(2.0)
    host = _DIMHost(strong)
    host._initialize_continuous_transition_matrix()
    assert float(host._effective_dim_scale()) < 1.0  # guard: scale engaged
    for j in range(2):
        assert _spectral_radius(host.continuous_transition_matrix[..., j]) <= 0.9 + 1e-6

    # A generic (out-of-family) M-step estimate must be projected and synced.
    host.continuous_transition_matrix = jnp.stack(
        [
            construct_directed_influence_transition_matrix(
                jnp.array([2.0 + j, 4.0 + j]),
                jnp.array([0.55 + 0.05 * j, 0.65]),
                jnp.array([[0.0, 0.02], [0.03, 0.0]]),
                jnp.zeros((2, 2)),
                100.0,
            )
            for j in range(2)
        ],
        axis=-1,
    )
    host._project_parameters()
    np.testing.assert_allclose(host.freqs, [2.5, 4.5], atol=1e-8)
    np.testing.assert_allclose(host.damping_coef, [0.575, 0.65], atol=1e-8)
    for j in range(2):
        reconstructed = construct_directed_influence_transition_matrix(
            host.freqs,
            host.damping_coef,
            host.coupling_strength[..., j],
            host.phase_difference[..., j],
            host.sampling_freq,
        )
        np.testing.assert_allclose(
            host.continuous_transition_matrix[..., j], reconstructed, atol=1e-9
        )

    reparam = _DIMHost(strong, use_reparameterized_mstep=True)
    reparam._initialize_continuous_transition_matrix()
    unconstrained = jnp.full((4, 4, 2), 0.3)
    reparam.continuous_transition_matrix = unconstrained
    reparam._project_parameters()
    np.testing.assert_array_equal(reparam.continuous_transition_matrix, unconstrained)


def test_directed_influence_mixin_rebuild_refreshes_optimizer_cache_only_once_set():
    strong = jnp.zeros((2, 2, 2)).at[0, 1, :].set(0.5)
    host = _DIMHost(strong)
    host._rebuild_stable_transition_matrix()
    assert host._current_osc_params is None

    host._current_osc_params = {
        "freq": host.freqs,
        "damping": host.damping_coef,
        "coupling_strength": jnp.zeros_like(strong),
        "phase_diff": jnp.zeros_like(strong),
    }
    host._rebuild_stable_transition_matrix()
    np.testing.assert_allclose(host._current_osc_params["coupling_strength"], strong)

    host._current_osc_params["freq"] = jnp.array([5.0, 9.0])
    host._update_public_oscillator_params()
    np.testing.assert_allclose(host.freqs, [5.0, 9.0])


# CNM covariance projection: closed-form PSD shrink factor vs. bisection
# ---------------------------------------------------------------------------


def _bisection_shrink_factor(structured, diag_only, min_eigenvalue=1e-8, max_iter=60):
    """Reference: the pre-closed-form host-side bisection on the min eigenvalue."""
    if float(jnp.linalg.eigvalsh(structured).min()) >= min_eigenvalue:
        return 1.0
    off_diag = structured - diag_only
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        candidate = diag_only + mid * off_diag
        if float(jnp.linalg.eigvalsh(candidate).min()) >= min_eigenvalue:
            lo = mid
        else:
            hi = mid
    return lo


def _random_cnm_covariance(rng, n_oscillators, linkage_scale):
    """Random symmetric covariance with inflated linkage blocks.

    Inflating the off-diagonal blocks makes the block-wise CNM projection
    indefinite for most draws, so the shrink factor is genuinely exercised.
    """
    dim = 2 * n_oscillators
    factor = rng.normal(size=(dim, dim))
    cov = factor @ factor.T / dim
    scale = np.full((dim, dim), linkage_scale)
    for i in range(n_oscillators):
        scale[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = 1.0
    return jnp.asarray(cov * scale)


def test_cnm_shrink_factor_matches_bisection_on_random_covariances():
    """The closed-form shrink factor agrees with the 60-step bisection it replaced.

    Also pins the PSD guarantee: the shrunk covariance's minimum eigenvalue
    stays at or above ``min_eigenvalue`` (the bisection accepted only
    candidates that passed that floating-point check; the closed form's
    relative safety margin keeps the same property).
    """
    min_eigenvalue = 1e-8
    rng = np.random.default_rng(0)
    n_shrunk = 0
    for _ in range(30):
        n_oscillators = int(rng.integers(2, 6))
        cov = _random_cnm_covariance(rng, n_oscillators, rng.uniform(1.5, 4.0))
        structured, diag_only = _cnm_structured_projection(cov, min_eigenvalue)
        t_ref = _bisection_shrink_factor(structured, diag_only, min_eigenvalue)
        t_new = float(_cnm_psd_shrink_factor(structured, diag_only, min_eigenvalue))
        np.testing.assert_allclose(t_new, t_ref, rtol=1e-6)
        shrunk = diag_only + t_new * (structured - diag_only)
        assert float(jnp.linalg.eigvalsh(shrunk).min()) >= min_eigenvalue
        n_shrunk += t_ref < 1.0
    # Guard: the comparison is only meaningful if shrinking actually happened.
    assert n_shrunk >= 10


def test_cnm_projection_matches_bisection_reference_and_is_jittable():
    """The public projection equals ``diag + t_bisect * off`` and traces under jit."""
    rng = np.random.default_rng(1)
    cov = _random_cnm_covariance(rng, n_oscillators=4, linkage_scale=3.0)
    structured, diag_only = _cnm_structured_projection(cov, 1e-8)
    t_ref = _bisection_shrink_factor(structured, diag_only, 1e-8)
    assert t_ref < 1.0  # guard: this draw needs shrinking
    expected = np.asarray(diag_only + t_ref * (structured - diag_only))

    projected = np.asarray(project_correlated_noise_process_covariance(cov))
    np.testing.assert_allclose(projected, expected, rtol=1e-6, atol=1e-9)
    assert np.linalg.eigvalsh(projected).min() >= 1e-8

    # Inside a larger jitted computation (no host syncs to break the trace).
    jitted = np.asarray(
        jax.jit(lambda c: 2.0 * project_correlated_noise_process_covariance(c))(cov)
    )
    np.testing.assert_allclose(jitted, 2.0 * projected, rtol=1e-12, atol=1e-14)

    # An already-PSD structured covariance is returned unshrunk.
    diag_strong = np.asarray(diag_only) * 50.0
    structured_psd = jnp.asarray(diag_strong) + (structured - diag_only)
    assert float(jnp.linalg.eigvalsh(structured_psd).min()) >= 1e-8  # guard
    np.testing.assert_allclose(
        np.asarray(project_correlated_noise_process_covariance(structured_psd)),
        np.asarray(structured_psd),
        rtol=1e-12,
        atol=1e-14,
    )


def _near_singular_cnm_covariance(rng, n_oscillators):
    """CNM covariance whose pairwise coherences sit near +/-1, plus small noise.

    Coherences within ``1e-7``-``1e-1`` of +/-1 put the structured projection
    at or just past the PSD boundary, where the shrink factor is most sensitive
    to roundoff.
    """
    dim = 2 * n_oscillators
    variance = np.exp(rng.uniform(-2.0, 2.0, n_oscillators))
    cov = np.kron(np.diag(variance), np.eye(2))
    for i in range(n_oscillators):
        for j in range(i + 1, n_oscillators):
            rho = rng.choice([-1.0, 1.0]) * (1.0 - 10.0 ** rng.uniform(-7.0, -1.0))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            rotation = np.array(
                [[np.cos(phase), -np.sin(phase)], [np.sin(phase), np.cos(phase)]]
            )
            block = rho * np.sqrt(variance[i] * variance[j]) * rotation
            cov[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = block
            cov[2 * j : 2 * j + 2, 2 * i : 2 * i + 2] = block.T
    return cov + 1e-3 * rng.normal(size=(dim, dim))


@pytest.mark.parametrize("min_eigenvalue", [1e-8, 1e-4])
def test_cnm_projection_keeps_eigenvalue_floor_in_float32(min_eigenvalue):
    """In float32 the shrunk covariance still has min eigenvalue >= the floor.

    The safety margin on the closed-form shrink factor must exceed float32
    roundoff (``1 - 1e-9 == 1`` in float32). The eigenvalues are those of the
    stored float32 matrix, computed in float64 so the check itself does not add
    float32 roundoff; the floor is compared at its float32 value.
    """
    rng = np.random.default_rng(3)
    covariances = [
        np.asarray(_random_cnm_covariance(rng, int(rng.integers(2, 5)), 3.0))
        for _ in range(40)
    ] + [_near_singular_cnm_covariance(rng, int(rng.integers(2, 5))) for _ in range(40)]
    floor = float(np.float32(min_eigenvalue))
    n_shrunk = 0
    for cov in covariances:
        cov32 = jnp.asarray(cov, dtype=jnp.float32)
        projected = project_correlated_noise_process_covariance(cov32, min_eigenvalue)
        assert projected.dtype == jnp.float32
        structured, _ = _cnm_structured_projection(cov32, min_eigenvalue)
        n_shrunk += not np.array_equal(np.asarray(projected), np.asarray(structured))
        min_eig = np.linalg.eigvalsh(np.asarray(projected, dtype=np.float64)).min()
        assert min_eig >= floor
    # Guard: the floor is only at risk when the linkage was actually shrunk.
    assert n_shrunk >= 40


def test_cnm_shrink_factor_handles_variance_at_the_floor():
    """Blocks whose variance sits exactly at the floor cannot carry linkage.

    ``S = diag_only - min_eigenvalue * I`` then has a zero diagonal entry: a
    non-zero linkage row there forces ``t = 0``; a zero row is decoupled and
    the factor is that of the remaining oscillators.
    """
    min_eigenvalue = 1e-8
    rng = np.random.default_rng(2)
    n_oscillators = 4
    cov = np.asarray(_random_cnm_covariance(rng, n_oscillators, 3.0))
    # Zero out oscillator 1 entirely: its variance is floored, its linkage vanishes.
    decoupled = cov.copy()
    decoupled[2:4, :] = 0.0
    decoupled[:, 2:4] = 0.0
    structured, diag_only = _cnm_structured_projection(
        jnp.asarray(decoupled), min_eigenvalue
    )
    assert float(diag_only[2, 2]) == min_eigenvalue  # guard: floored block
    t_decoupled = float(_cnm_psd_shrink_factor(structured, diag_only, min_eigenvalue))
    keep = [0, 1, 4, 5, 6, 7]
    reduced = jnp.asarray(decoupled[np.ix_(keep, keep)])
    structured_r, diag_only_r = _cnm_structured_projection(reduced, min_eigenvalue)
    t_reduced = float(_cnm_psd_shrink_factor(structured_r, diag_only_r, min_eigenvalue))
    assert t_reduced < 1.0  # guard: the remaining oscillators need shrinking
    np.testing.assert_allclose(t_decoupled, t_reduced, rtol=1e-10)

    # Floored block WITH linkage: no positive t keeps the covariance PSD.
    pinned = decoupled.copy()
    pinned[2, 4] = pinned[4, 2] = 0.3
    structured_p, diag_only_p = _cnm_structured_projection(
        jnp.asarray(pinned), min_eigenvalue
    )
    t_pinned = float(_cnm_psd_shrink_factor(structured_p, diag_only_p, min_eigenvalue))
    assert t_pinned == 0.0
    np.testing.assert_allclose(
        np.asarray(project_correlated_noise_process_covariance(jnp.asarray(pinned))),
        np.asarray(diag_only_p),
        atol=1e-15,
    )


# ---------------------------------------------------------------------------
# Tight DIM stability scale and block-local spectral-radius clamp
# ---------------------------------------------------------------------------


def _dim_pair(coupling: float, freqs=(8.0, 20.0), damping=0.95, n_states=2):
    """Two oscillators with one-directional coupling that reverses by state."""
    coupling_stack = jnp.zeros((2, 2, n_states))
    coupling_stack = coupling_stack.at[1, 0, 0].set(coupling)
    if n_states > 1:
        coupling_stack = coupling_stack.at[0, 1, 1].set(coupling)
    return (
        jnp.asarray(freqs),
        jnp.full(2, damping),
        coupling_stack,
        jnp.zeros((2, 2, n_states)),
    )


def test_stability_scale_leaves_stable_dim_matrices_untouched() -> None:
    """A stable DIM network is rebuilt exactly: the scale is 1, not a loose bound.

    The previous block-row operator-norm bound returned ~0.82 for this pair
    (actual radius 0.85), over-damping every rebuilt matrix by up to 0.16.
    """
    freqs, damping, coupling, phase = _dim_pair(0.3)
    fs = 100.0
    plain = jnp.stack(
        [
            construct_directed_influence_transition_matrix(
                freqs, damping, coupling[..., j], phase[..., j], fs
            )
            for j in range(2)
        ],
        axis=-1,
    )
    radius = max(_spectral_radius(plain[..., j]) for j in range(2))
    assert radius < 0.99  # guard: genuinely stable under the default bound

    scale = compute_directed_influence_stability_scale(
        freqs, damping, coupling, fs, phase_difference=phase
    )
    assert float(scale) == 1.0
    stack = construct_stable_directed_influence_transition_stack(
        freqs, damping, coupling, phase, fs
    )
    np.testing.assert_allclose(stack, plain, atol=1e-14)


def test_stability_scale_clamps_actual_radius_exactly_and_is_differentiable() -> None:
    """When engaged, the scale puts the actual radius exactly on the bound and its
    gradient matches finite differences; below the bound its gradient is zero."""
    freqs, damping, coupling, phase = _dim_pair(1.5, damping=0.85)
    fs = 100.0
    bound = 0.9

    def scale_of(c):
        return compute_directed_influence_stability_scale(
            freqs, damping, c, fs, max_spectral_radius=bound, phase_difference=phase
        )

    scale = float(scale_of(coupling))
    assert scale < 1.0  # guard: engaged
    stack = construct_stable_directed_influence_transition_stack(
        freqs, damping, coupling, phase, fs, max_spectral_radius=bound
    )
    radii = [_spectral_radius(stack[..., j]) for j in range(2)]
    np.testing.assert_allclose(max(radii), bound, rtol=1e-10)

    direction = jnp.zeros_like(coupling).at[1, 0, 0].set(1.0)
    grad = jax.jit(jax.grad(scale_of))(coupling)
    h = 1e-6
    fd = (
        float(scale_of(coupling + h * direction))
        - float(scale_of(coupling - h * direction))
    ) / (2 * h)
    np.testing.assert_allclose(float(jnp.sum(grad * direction)), fd, rtol=1e-5)

    weak = _dim_pair(0.05, damping=0.85)[2]
    grad_weak = jax.grad(scale_of)(weak)
    assert float(scale_of(weak)) == 1.0
    np.testing.assert_array_equal(np.asarray(grad_weak), 0.0)


def test_dim_scenario_truth_is_a_fixed_point_of_init_and_projection(caplog) -> None:
    """The repo's DIM scenario parameters survive init and the standard-EM
    projection/sync unchanged (the old loose scale shrank them to 0.76x)."""
    from state_space_practice.simulate.scenarios import simulate_dim_scenario

    params = simulate_dim_scenario(n_time=10)["params"]
    host = _DIMHost(jnp.asarray(params["coupling_strength"]))
    host.freqs = jnp.asarray(params["freqs"])
    host.damping_coef = jnp.asarray(params["damping"])
    host.phase_difference = jnp.asarray(params["phase_difference"])
    host.sampling_freq = params["sampling_freq"]
    host.max_spectral_radius = 0.999  # true radius is 0.9962
    A_true = np.asarray(params["A"])

    host._initialize_continuous_transition_matrix()
    assert np.max(np.abs(np.asarray(host.continuous_transition_matrix) - A_true)) < 1e-8

    host.continuous_transition_matrix = jnp.asarray(A_true)
    with caplog.at_level("WARNING"):
        host._project_parameters()
    assert "exceeded max_spectral_radius" not in caplog.text  # no clamp engaged
    assert np.max(np.abs(np.asarray(host.continuous_transition_matrix) - A_true)) < 1e-8
    np.testing.assert_allclose(host.damping_coef, params["damping"], atol=1e-10)
    np.testing.assert_allclose(
        host.coupling_strength, params["coupling_strength"], atol=1e-10
    )


def test_project_stack_clamps_only_the_unstable_uncoupled_block(caplog) -> None:
    """For uncoupled oscillators the clamp rescales only the offending block,
    leaves the stable rhythm untouched, and reports what it did."""
    unstable = 1.2 * _get_rotation_matrix(2.0 * jnp.pi * 8.0 / 100.0)
    stable = 0.5 * _get_rotation_matrix(2.0 * jnp.pi * 20.0 / 100.0)
    A = jnp.zeros((4, 4)).at[:2, :2].set(unstable).at[2:, 2:].set(stable)
    bound = 0.95

    with caplog.at_level("WARNING"):
        projected = project_transition_matrix_stack(A[..., None], bound)[..., 0]
    assert "rows [0, 1]: radius=1.2, scale=0.791667" in caplog.text
    assert "rows [2, 3]" not in caplog.text

    np.testing.assert_allclose(projected[2:, 2:], stable, atol=1e-14)
    np.testing.assert_allclose(projected[:2, :2], unstable * (bound / 1.2), atol=1e-14)
    np.testing.assert_array_equal(np.asarray(projected[:2, 2:]), 0.0)
    np.testing.assert_allclose(_spectral_radius(projected), bound, rtol=1e-12)


def test_stabilize_transition_matrix_block_clamp_follows_coupling_graph(
    caplog,
) -> None:
    """One-directional coupling keeps the blocks' spectra separate, so only the
    unstable source block is scaled; mutual coupling falls back to one scale."""
    source = 1.1 * _get_rotation_matrix(0.3)
    target = 0.6 * _get_rotation_matrix(0.9)
    link = 0.2 * _get_rotation_matrix(0.1)
    one_way = jnp.zeros((4, 4)).at[:2, :2].set(source).at[2:, 2:].set(target)
    one_way = one_way.at[2:, :2].set(link)

    with caplog.at_level("WARNING"):
        clamped = stabilize_transition_matrix(one_way, 0.9, block_size=2)
    assert "rows [0, 1]: radius=1.1" in caplog.text
    np.testing.assert_allclose(clamped[2:, 2:], target, atol=1e-14)
    np.testing.assert_allclose(clamped[2:, :2], link, atol=1e-14)
    np.testing.assert_allclose(_spectral_radius(clamped), 0.9, rtol=1e-12)

    mutual = one_way.at[:2, 2:].set(link)
    radius = _spectral_radius(mutual)
    clamped_mutual = stabilize_transition_matrix(mutual, 0.9, block_size=2)
    np.testing.assert_allclose(clamped_mutual, mutual * (0.9 / radius), atol=1e-14)

    # Already stable: returned unchanged and silent.
    caplog.clear()
    with caplog.at_level("WARNING"):
        unchanged = stabilize_transition_matrix(0.5 * one_way, 0.9, block_size=2)
    np.testing.assert_array_equal(np.asarray(unchanged), np.asarray(0.5 * one_way))
    assert caplog.text == ""


def test_differentiable_spectral_radius_matches_numpy_under_jit_and_vmap() -> None:
    from state_space_practice.utils import differentiable_spectral_radius

    mats = jax.random.normal(jax.random.PRNGKey(3), (3, 5, 5))
    expected = [np.max(np.abs(np.linalg.eigvals(np.asarray(m)))) for m in mats]
    np.testing.assert_allclose(
        jax.jit(differentiable_spectral_radius)(mats), expected, rtol=1e-12
    )
    np.testing.assert_allclose(
        jax.vmap(differentiable_spectral_radius)(mats), expected, rtol=1e-12
    )
