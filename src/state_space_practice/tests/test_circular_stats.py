"""Tests for circular statistics utilities."""

import numpy as np
import pytest

from state_space_practice.circular_stats import (
    angular_distance,
    circular_correlation,
    circular_mean,
    circular_std,
    compute_phase_histogram,
    compute_preferred_phase,
    mean_resultant_length,
    rayleigh_test,
    wrap_to_pi,
)


class TestCircularMean:
    """Tests for circular_mean function."""

    def test_zero_mean(self):
        """Test phases with zero circular mean."""
        phases = np.array([np.pi / 4, -np.pi / 4])
        result = circular_mean(phases)
        assert np.isclose(result, 0, atol=1e-10)

    def test_known_mean(self):
        """Test phases with known circular mean."""
        phases = np.array([0, 0, 0])
        result = circular_mean(phases)
        assert np.isclose(result, 0, atol=1e-10)

    def test_wrapped_mean(self):
        """Test that mean handles wrapping correctly."""
        # Mean of pi and -pi should be pi (or -pi, they're equivalent)
        phases = np.array([np.pi - 0.1, -np.pi + 0.1])
        result = circular_mean(phases)
        assert np.isclose(np.abs(result), np.pi, atol=0.2)


class TestCircularStd:
    """Tests for circular_std function."""

    def test_concentrated_distribution(self):
        """Test std of highly concentrated distribution."""
        phases = np.array([0.0, 0.01, -0.01, 0.02, -0.02])
        result = circular_std(phases)
        # Should be small for concentrated distribution
        assert result < 0.1

    def test_uniform_distribution(self):
        """Test std of uniform distribution."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(-np.pi, np.pi, 1000)
        result = circular_std(phases)
        # Should be large for uniform distribution
        assert result > 1.0

    def test_handles_extreme_concentration(self):
        """Test numerical stability for identical phases."""
        phases = np.zeros(100)
        result = circular_std(phases)
        # Should be very small but not NaN or inf
        assert np.isfinite(result)
        assert result < 0.01


class TestMeanResultantLength:
    """Tests for mean_resultant_length function."""

    def test_identical_phases(self):
        """Test MRL of identical phases is 1."""
        phases = np.zeros(100)
        result = mean_resultant_length(phases)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_uniform_phases(self):
        """Test MRL of uniform distribution approaches 0."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(-np.pi, np.pi, 10000)
        result = mean_resultant_length(phases)
        # Should be close to 0 for large uniform sample
        assert result < 0.1

    def test_range(self):
        """Test MRL is in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.vonmises(0, rng.uniform(0, 5), 100)
            result = mean_resultant_length(phases)
            assert 0 <= result <= 1


class TestRayleighTest:
    """Tests for rayleigh_test function."""

    def test_uniform_distribution_not_significant(self):
        """Test uniform distribution gives high p-value."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(-np.pi, np.pi, 1000)
        R, p_value = rayleigh_test(phases)

        assert p_value > 0.05  # Should not be significant

    def test_phase_locked_significant(self):
        """Test concentrated distribution gives low p-value."""
        rng = np.random.default_rng(42)
        phases = rng.vonmises(0, 2, 1000)  # High concentration
        R, p_value = rayleigh_test(phases)

        assert p_value < 0.05  # Should be significant

    def test_small_sample_correction(self):
        """Test that small sample correction is applied."""
        rng = np.random.default_rng(42)
        phases = rng.vonmises(0, 2, 10)  # Small sample
        R, p_value = rayleigh_test(phases)

        # Just check it returns valid values
        assert 0 <= R <= 1
        assert 0 <= p_value <= 1

    def test_p_value_range(self):
        """Test p-value is in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.vonmises(0, rng.uniform(0, 3), rng.integers(5, 100))
            R, p_value = rayleigh_test(phases)
            assert 0 <= p_value <= 1


class TestCircularCorrelation:
    """Tests for circular_correlation function."""

    def test_identical_phases(self):
        """Test correlation of identical phases is 1."""
        phases = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
        result = circular_correlation(phases, phases)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_independent_phases(self):
        """Test correlation of independent uniform phases is ~0."""
        rng = np.random.default_rng(42)
        phases1 = rng.uniform(-np.pi, np.pi, 1000)
        phases2 = rng.uniform(-np.pi, np.pi, 1000)
        result = circular_correlation(phases1, phases2)
        assert np.abs(result) < 0.1

    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError):
            circular_correlation(np.array([0, 1]), np.array([0]))


class TestComputePhaseHistogram:
    """Tests for compute_phase_histogram function."""

    def test_basic_histogram(self):
        """Test basic phase histogram computation."""
        spike_times = np.array([0.25, 0.75])  # Spikes at specific times
        time_axis = np.arange(0, 1, 0.01)  # 100 time points
        # Phase increases linearly from -pi to pi
        inferred_phase = np.linspace(-np.pi, np.pi, len(time_axis))

        hist, bin_centers = compute_phase_histogram(
            spike_times, inferred_phase, time_axis, n_bins=36
        )

        assert len(hist) == 36
        assert len(bin_centers) == 36
        assert hist.sum() == 2  # Two spikes total

    def test_with_mask(self):
        """Test histogram with mask."""
        spike_times = np.array([0.1, 0.5, 0.9])
        time_axis = np.arange(0, 1, 0.01)
        inferred_phase = np.zeros(len(time_axis))  # All phase 0
        mask = time_axis > 0.5  # Only include second half

        hist, _ = compute_phase_histogram(
            spike_times, inferred_phase, time_axis, mask=mask, n_bins=36
        )

        # Only spike at 0.9 should be included (0.5 is at boundary)
        assert hist.sum() == 1

    def test_empty_spikes(self):
        """Test histogram with no spikes."""
        spike_times = np.array([])
        time_axis = np.arange(0, 1, 0.01)
        inferred_phase = np.zeros(len(time_axis))

        hist, _ = compute_phase_histogram(
            spike_times, inferred_phase, time_axis, n_bins=36
        )

        assert hist.sum() == 0


class TestComputePreferredPhase:
    """Tests for compute_preferred_phase function."""

    def test_basic_preferred_phase(self):
        """Test basic preferred phase computation."""
        # Spikes all at t=0.25 where phase = -pi/2
        spike_times = np.array([0.24, 0.25, 0.26])
        time_axis = np.arange(0, 1, 0.01)
        # Phase from -pi to pi
        inferred_phase = np.linspace(-np.pi, np.pi, len(time_axis))

        pref_phase, mrl, p_value = compute_preferred_phase(
            spike_times, inferred_phase, time_axis
        )

        # Should have high MRL (concentrated spikes)
        assert mrl > 0.8
        # Should be significant
        assert p_value < 0.05

    def test_too_few_spikes(self):
        """Test returns NaN with too few spikes."""
        spike_times = np.array([0.5])
        time_axis = np.arange(0, 1, 0.01)
        inferred_phase = np.zeros(len(time_axis))

        pref_phase, mrl, p_value = compute_preferred_phase(
            spike_times, inferred_phase, time_axis
        )

        assert np.isnan(pref_phase)
        assert np.isnan(mrl)
        assert np.isnan(p_value)


class TestAngularDistance:
    """Tests for angular_distance function."""

    def test_zero_distance(self):
        """Test distance between same phase is 0."""
        result = angular_distance(0.5, 0.5)
        assert np.isclose(result, 0, atol=1e-10)

    def test_opposite_phases(self):
        """Test distance between opposite phases is pi."""
        result = angular_distance(0, np.pi)
        assert np.isclose(result, np.pi, atol=1e-10)

    def test_wrapping(self):
        """Test distance handles wrapping correctly."""
        # Distance between pi and -pi should be 0 (they're the same point)
        result = angular_distance(np.pi, -np.pi)
        assert np.isclose(result, 0, atol=1e-10)

    def test_array_input(self):
        """Test with array inputs."""
        phases1 = np.array([0, np.pi / 2, np.pi])
        phases2 = np.array([np.pi / 4, np.pi / 2, 0])
        result = angular_distance(phases1, phases2)

        expected = np.array([np.pi / 4, 0, np.pi])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestWrapToPi:
    """Tests for wrap_to_pi function."""

    def test_no_wrap_needed(self):
        """Test phases already in range stay unchanged."""
        phases = np.array([-np.pi / 2, 0, np.pi / 2])
        result = wrap_to_pi(phases)
        np.testing.assert_allclose(result, phases, atol=1e-10)

    def test_positive_wrap(self):
        """Test positive angles wrap correctly."""
        phases = np.array([np.pi + 0.1, 2 * np.pi, 3 * np.pi])
        result = wrap_to_pi(phases)
        # All should be in [-pi, pi]
        assert np.all(result >= -np.pi)
        assert np.all(result <= np.pi)

    def test_negative_wrap(self):
        """Test negative angles wrap correctly."""
        phases = np.array([-np.pi - 0.1, -2 * np.pi, -3 * np.pi])
        result = wrap_to_pi(phases)
        # All should be in [-pi, pi]
        assert np.all(result >= -np.pi)
        assert np.all(result <= np.pi)
