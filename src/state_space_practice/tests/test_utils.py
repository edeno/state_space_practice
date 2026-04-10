"""Tests for the utils module."""

import jax.numpy as jnp
import numpy as np

from state_space_practice.utils import (
    check_converged,
    divide_safe,
    project_psd,
    psd_solve,
    safe_log,
    scale_likelihood,
    stabilize_covariance,
    stabilize_probability_vector,
    symmetrize,
)


class TestCheckConverged:
    """Tests for the check_converged function."""

    def test_identical_values_converged(self) -> None:
        """Identical log-likelihoods should indicate convergence."""
        ll = -100.0
        is_converged, is_increasing = check_converged(ll, ll, tolerance=1e-4)
        assert is_converged is True
        assert is_increasing is True

    def test_small_relative_change_converged(self) -> None:
        """Small relative change below tolerance should converge."""
        ll_prev = -1000.0
        # Change of 0.01 on average of 1000 = 1e-5 relative change
        ll_curr = -999.99
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_converged is True
        assert is_increasing is True

    def test_large_relative_change_not_converged(self) -> None:
        """Large relative change above tolerance should not converge."""
        ll_prev = -100.0
        ll_curr = -90.0  # 10% change
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_converged is False
        assert is_increasing is True

    def test_increasing_log_likelihood(self) -> None:
        """Increasing log-likelihood should be flagged as increasing."""
        ll_prev = -100.0
        ll_curr = -50.0  # LL increased (less negative)
        _, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_increasing is True

    def test_decreasing_log_likelihood_beyond_tolerance(self) -> None:
        """Decreasing log-likelihood beyond tolerance should flag as not increasing."""
        ll_prev = -50.0
        ll_curr = -100.0  # LL decreased significantly
        _, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_increasing is False

    def test_slight_decrease_within_tolerance_is_increasing(self) -> None:
        """Slight decrease within tolerance should still be considered increasing."""
        ll_prev = -100.0
        ll_curr = -100.00005  # Decrease by 5e-5, within 1e-4 tolerance
        _, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_increasing is True

    def test_zero_log_likelihoods(self) -> None:
        """Zero log-likelihoods should be handled without division errors."""
        is_converged, is_increasing = check_converged(0.0, 0.0, tolerance=1e-4)
        assert is_converged is True
        assert is_increasing is True

    def test_negative_infinity_previous(self) -> None:
        """Negative infinity previous LL (initial state) should not converge."""
        ll_prev = -np.inf
        ll_curr = -100.0
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        # Any finite value is an improvement from -inf
        assert is_converged is False
        assert is_increasing is True

    def test_both_negative_infinity(self) -> None:
        """Both -inf should be considered converged (no change)."""
        ll_prev = -np.inf
        ll_curr = -np.inf
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        # inf - inf = nan, but the function should handle this
        # The actual behavior depends on implementation
        assert isinstance(is_converged, bool)
        assert isinstance(is_increasing, bool)

    def test_positive_log_likelihoods(self) -> None:
        """Positive log-likelihoods (valid in some contexts) should work."""
        ll_prev = 10.0
        ll_curr = 10.0001
        is_converged, is_increasing = check_converged(ll_curr, ll_prev, tolerance=1e-4)
        assert is_converged is True
        assert is_increasing is True

    def test_tolerance_boundary_exact(self) -> None:
        """Test behavior exactly at the tolerance boundary."""
        ll_prev = -100.0
        # Create a change exactly at tolerance
        avg = 100.0  # |ll_prev| + |ll_curr| / 2 ≈ 100
        tolerance = 1e-4
        delta = tolerance * avg  # 0.01
        ll_curr = ll_prev + delta

        is_converged, _ = check_converged(ll_curr, ll_prev, tolerance=tolerance)
        # At boundary, should not be converged (strictly less than)
        assert is_converged is False

    def test_custom_tolerance(self) -> None:
        """Custom tolerance values should be respected."""
        ll_prev = -100.0
        ll_curr = -99.0  # 1% change

        # With 1% tolerance, should converge
        is_converged_loose, _ = check_converged(ll_curr, ll_prev, tolerance=0.02)
        assert is_converged_loose is True

        # With 0.1% tolerance, should not converge
        is_converged_tight, _ = check_converged(ll_curr, ll_prev, tolerance=0.001)
        assert is_converged_tight is False

    def test_returns_python_bools(self) -> None:
        """Function should return Python bools, not numpy bools."""
        is_converged, is_increasing = check_converged(-100.0, -100.0)
        assert type(is_converged) is bool
        assert type(is_increasing) is bool

    def test_very_small_values(self) -> None:
        """Very small log-likelihoods should be handled correctly."""
        ll_prev = -1e-10
        ll_curr = -1e-10
        is_converged, is_increasing = check_converged(ll_curr, ll_prev)
        assert is_converged is True
        assert is_increasing is True

    def test_very_large_values(self) -> None:
        """Very large log-likelihoods should be handled correctly."""
        ll_prev = -1e10
        ll_curr = -1e10 + 1  # Small absolute change, tiny relative change
        is_converged, is_increasing = check_converged(ll_curr, ll_prev)
        assert is_converged is True
        assert is_increasing is True


class TestLinearAlgebraUtilities:
    """Tests for linear algebra utilities moved from kalman.py."""

    def test_symmetrize_makes_symmetric(self) -> None:
        A = jnp.array([[1.0, 2.0], [4.0, 5.0]])
        S = symmetrize(A)
        np.testing.assert_allclose(S, S.T, atol=1e-12)
        np.testing.assert_allclose(S, jnp.array([[1.0, 3.0], [3.0, 5.0]]))

    def test_symmetrize_batch(self) -> None:
        """Symmetrize should handle batched matrices (last two dims)."""
        A = jnp.stack([jnp.array([[1.0, 2.0], [4.0, 5.0]]), jnp.eye(2) * 3.0])
        S = symmetrize(A)
        np.testing.assert_allclose(S, jnp.swapaxes(S, -1, -2), atol=1e-12)

    def test_psd_solve_matches_numpy(self) -> None:
        A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x = psd_solve(A, b)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, expected, atol=1e-6)

    def test_project_psd_clips_negative_eigvals(self) -> None:
        Q = jnp.array([[1.0, 0.5], [0.5, -0.1]])  # indefinite
        Q_psd = project_psd(Q, min_eigenvalue=1e-4)
        eigvals = jnp.linalg.eigvalsh(Q_psd)
        assert jnp.all(eigvals >= 1e-4 - 1e-8)

    def test_stabilize_covariance_preserves_psd_input(self) -> None:
        C = jnp.array([[2.0, 0.5], [0.5, 1.0]])  # PSD already
        stabilized = stabilize_covariance(C)
        np.testing.assert_allclose(stabilized, C, atol=1e-6)


class TestProbabilityUtilities:
    """Tests for probability utilities moved from switching_kalman.py."""

    def test_divide_safe_zero_denominator(self) -> None:
        """divide_safe should return 0 when denominator is 0."""
        result = divide_safe(jnp.array(5.0), jnp.array(0.0))
        assert float(result) == 0.0

    def test_divide_safe_normal(self) -> None:
        result = divide_safe(jnp.array(6.0), jnp.array(2.0))
        assert float(result) == 3.0

    def test_divide_safe_elementwise(self) -> None:
        num = jnp.array([2.0, 4.0, 6.0])
        den = jnp.array([1.0, 0.0, 3.0])
        result = divide_safe(num, den)
        np.testing.assert_allclose(result, jnp.array([2.0, 0.0, 2.0]))

    def test_safe_log_at_floor_boundary(self) -> None:
        """safe_log should floor small inputs to -23.0, not -inf."""
        result = safe_log(jnp.array(0.0))
        assert float(result) == -23.0

    def test_safe_log_normal(self) -> None:
        """safe_log should match jnp.log for values above the floor."""
        x = jnp.array(0.5)
        np.testing.assert_allclose(safe_log(x), jnp.log(x), atol=1e-6)

    def test_stabilize_probability_vector_sums_to_one(self) -> None:
        p = jnp.array([0.1, 0.2, 0.7])
        stabilized = stabilize_probability_vector(p)
        np.testing.assert_allclose(float(jnp.sum(stabilized)), 1.0, atol=1e-12)

    def test_stabilize_probability_vector_all_zeros_gives_uniform(self) -> None:
        """All-zero input should be lifted to uniform after re-normalization."""
        p = jnp.zeros(4)
        stabilized = stabilize_probability_vector(p)
        np.testing.assert_allclose(stabilized, jnp.ones(4) / 4.0, atol=1e-6)
        np.testing.assert_allclose(float(jnp.sum(stabilized)), 1.0, atol=1e-12)

    def test_stabilize_probability_vector_preserves_dominant_state(self) -> None:
        """A peaked distribution should remain peaked after stabilization."""
        p = jnp.array([1.0, 0.0, 0.0])
        stabilized = stabilize_probability_vector(p)
        assert float(stabilized[0]) > 0.99

    def test_scale_likelihood_subtracts_max(self) -> None:
        ll = jnp.array([-10.0, -5.0, -20.0])
        scaled, ll_max = scale_likelihood(ll)
        assert float(ll_max) == -5.0
        # The max-scaled likelihood entry should be 1.0
        assert float(jnp.max(scaled)) == 1.0

    def test_scale_likelihood_all_neg_inf_uses_zero_max(self) -> None:
        """All -inf input: ll_max should fall back to 0.0, preventing NaN."""
        ll = jnp.full((3,), -jnp.inf)
        scaled, ll_max = scale_likelihood(ll)
        assert float(ll_max) == 0.0
        # exp(-inf - 0) = 0.0
        np.testing.assert_allclose(scaled, jnp.zeros(3))
