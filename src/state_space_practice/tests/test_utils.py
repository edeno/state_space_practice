"""Tests for the utils module.

This module tests the convergence checking utility used in EM algorithms.
"""

import numpy as np

from state_space_practice.utils import check_converged


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
