"""Tests for preprocessing utilities."""

import numpy as np

from state_space_practice.preprocessing import (
    bin_spike_times,
    binned_to_spike_times,
    clip_spike_times_to_window,
    compute_firing_rates,
    create_behavioral_labels,
    get_spike_times_subset,
    identify_behavioral_bouts,
    interpolate_to_new_times,
    select_units,
)


class TestBinSpikeTimes:
    """Tests for bin_spike_times function."""

    def test_basic_binning(self):
        """Test basic spike binning."""
        spike_times = [np.array([0.1, 0.3, 0.5]), np.array([0.2, 0.4])]
        time_bins = np.arange(0, 1.0, 0.2)  # 0, 0.2, 0.4, 0.6, 0.8

        counts = bin_spike_times(spike_times, time_bins)

        assert counts.shape == (5, 2)
        # Neuron 0: spikes at 0.1 (bin 0), 0.3 (bin 1), 0.5 (bin 2)
        assert counts[0, 0] == 1
        assert counts[1, 0] == 1
        assert counts[2, 0] == 1
        # Neuron 1: spikes at 0.2 (bin 1), 0.4 (bin 2)
        assert counts[1, 1] == 1
        assert counts[2, 1] == 1

    def test_empty_units(self):
        """Test with empty spike trains."""
        spike_times = [np.array([]), np.array([0.5])]
        time_bins = np.arange(0, 1.0, 0.2)

        counts = bin_spike_times(spike_times, time_bins)

        assert counts.shape == (5, 2)
        assert counts[:, 0].sum() == 0  # First neuron has no spikes
        assert counts[:, 1].sum() == 1  # Second neuron has one spike

    def test_multiple_spikes_per_bin(self):
        """Test multiple spikes in same bin."""
        spike_times = [np.array([0.1, 0.15, 0.18])]  # All in first bin (0.0-0.2)
        time_bins = np.arange(0, 1.0, 0.2)

        counts = bin_spike_times(spike_times, time_bins)

        assert counts[0, 0] == 3

    def test_preserves_total_spikes(self):
        """Test that binning preserves total spike count."""
        rng = np.random.default_rng(42)
        spike_times = [rng.uniform(0, 10, 100) for _ in range(5)]
        time_bins = np.arange(0, 10, 0.01)

        counts = bin_spike_times(spike_times, time_bins)

        total_original = sum(len(st) for st in spike_times)
        total_binned = counts.sum()
        assert total_binned == total_original


class TestComputeFiringRates:
    """Tests for compute_firing_rates function."""

    def test_basic_firing_rates(self):
        """Test basic firing rate computation."""
        spike_times = [np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5])]
        rates = compute_firing_rates(spike_times, start_time=0, end_time=4)

        # Neuron 0: 3 spikes / 4 s = 0.75 Hz
        assert np.isclose(rates[0], 0.75)
        # Neuron 1: 2 spikes / 4 s = 0.5 Hz
        assert np.isclose(rates[1], 0.5)

    def test_empty_units(self):
        """Test with empty spike trains."""
        spike_times = [np.array([]), np.array([1.0, 2.0])]
        rates = compute_firing_rates(spike_times, start_time=0, end_time=4)

        assert rates[0] == 0.0
        assert np.isclose(rates[1], 0.5)

    def test_all_empty(self):
        """Test with all empty spike trains."""
        spike_times = [np.array([]), np.array([])]
        rates = compute_firing_rates(spike_times)

        assert np.all(rates == 0.0)

    def test_auto_time_window(self):
        """Test automatic time window from spike times."""
        spike_times = [np.array([1.0, 3.0])]
        rates = compute_firing_rates(spike_times)

        # Duration = 3.0 - 1.0 = 2.0 s, 2 spikes -> 1.0 Hz
        assert np.isclose(rates[0], 1.0)


class TestSelectUnits:
    """Tests for select_units function."""

    def test_filter_by_rate(self):
        """Test unit selection by firing rate."""
        # Create spike trains with known rates
        spike_times = [
            np.arange(0, 10, 0.1),  # 10 Hz
            np.arange(0, 10, 1.0),  # 1 Hz
            np.arange(0, 10, 0.01),  # 100 Hz
        ]

        selected = select_units(spike_times, min_rate=5.0, max_rate=50.0)

        assert len(selected) == 1
        assert selected[0] == 0  # Only 10 Hz unit selected

    def test_all_selected(self):
        """Test when all units pass criteria."""
        spike_times = [np.arange(0, 10, 0.1) for _ in range(3)]
        selected = select_units(spike_times, min_rate=0.0, max_rate=100.0)

        assert len(selected) == 3


class TestIdentifyBehavioralBouts:
    """Tests for identify_behavioral_bouts function."""

    def test_basic_bouts(self):
        """Test basic bout identification."""
        speed = np.array([1, 2, 10, 15, 12, 3, 2, 8, 9, 10, 1])
        bouts = identify_behavioral_bouts(speed, speed_threshold=5.0, min_duration=2)

        # Two running bouts: indices 2-5 and 7-10
        assert len(bouts) == 2
        assert bouts[0] == (2, 5)
        assert bouts[1] == (7, 10)

    def test_below_threshold(self):
        """Test finding bouts below threshold."""
        speed = np.array([10, 10, 1, 1, 1, 10, 10])
        bouts = identify_behavioral_bouts(
            speed, speed_threshold=5.0, min_duration=2, above_threshold=False
        )

        assert len(bouts) == 1
        assert bouts[0] == (2, 5)

    def test_min_duration_filter(self):
        """Test minimum duration filtering."""
        speed = np.array([10, 1, 10, 10, 10])  # Only 1 sample below threshold
        bouts = identify_behavioral_bouts(
            speed, speed_threshold=5.0, min_duration=2, above_threshold=False
        )

        assert len(bouts) == 0  # Too short


class TestCreateBehavioralLabels:
    """Tests for create_behavioral_labels function."""

    def test_basic_labels(self):
        """Test basic behavioral labeling."""
        speed = np.array([1, 3, 8, 10, 4, 1])
        labels = create_behavioral_labels(speed)

        # 1 < 2: immobility (0)
        # 3 in (2, 5): transition (2)
        # 8 > 5: running (1)
        # 10 > 5: running (1)
        # 4 in (2, 5): transition (2)
        # 1 < 2: immobility (0)
        expected = np.array([0, 2, 1, 1, 2, 0])
        np.testing.assert_array_equal(labels, expected)

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        speed = np.array([1, 5, 10])
        labels = create_behavioral_labels(
            speed, running_threshold=8.0, immobility_threshold=3.0
        )

        expected = np.array([0, 2, 1])
        np.testing.assert_array_equal(labels, expected)


class TestInterpolateToNewTimes:
    """Tests for interpolate_to_new_times function."""

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        values = np.array([0, 2, 4])
        original_times = np.array([0, 1, 2])
        new_times = np.array([0.5, 1.5])

        result = interpolate_to_new_times(values, original_times, new_times)

        np.testing.assert_allclose(result, [1, 3])

    def test_multidimensional(self):
        """Test interpolation of multi-dimensional data."""
        values = np.array([[0, 0], [2, 4], [4, 8]])
        original_times = np.array([0, 1, 2])
        new_times = np.array([0.5, 1.5])

        result = interpolate_to_new_times(values, original_times, new_times)

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, [[1, 2], [3, 6]])


class TestGetSpikeTimesSubset:
    """Tests for get_spike_times_subset function."""

    def test_basic_subset(self):
        """Test extracting spike time subset."""
        spike_times = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        subset = get_spike_times_subset(spike_times, [0, 2])

        assert len(subset) == 2
        np.testing.assert_array_equal(subset[0], [1, 2])
        np.testing.assert_array_equal(subset[1], [5, 6])


class TestClipSpikeTimesToWindow:
    """Tests for clip_spike_times_to_window function."""

    def test_basic_clipping(self):
        """Test basic spike time clipping."""
        spike_times = [np.array([0.5, 1.5, 2.5, 3.5])]
        clipped = clip_spike_times_to_window(spike_times, 1.0, 3.0)

        np.testing.assert_array_equal(clipped[0], [1.5, 2.5])


class TestBinnedToSpikeTimes:
    """Tests for binned_to_spike_times function."""

    def test_single_neuron(self):
        """Test extracting spike times for single neuron."""
        binned = np.array([[1, 0], [0, 2], [1, 1]])
        time_bins = np.array([0.0, 0.1, 0.2])

        spike_times = binned_to_spike_times(binned, time_bins, neuron_idx=0)

        np.testing.assert_array_equal(spike_times, [0.0, 0.2])

    def test_all_neurons(self):
        """Test extracting spike times for all neurons."""
        binned = np.array([[1, 0], [0, 2], [1, 1]])
        time_bins = np.array([0.0, 0.1, 0.2])

        spike_times = binned_to_spike_times(binned, time_bins)

        assert len(spike_times) == 2
        np.testing.assert_array_equal(spike_times[0], [0.0, 0.2])
        # Neuron 1: 2 spikes at t=0.1, 1 spike at t=0.2
        np.testing.assert_array_equal(spike_times[1], [0.1, 0.1, 0.2])

    def test_roundtrip(self):
        """Test binning then unbinning preserves spike counts."""
        original_spike_times = [np.array([0.05, 0.35, 0.55])]
        time_bins = np.arange(0, 1.0, 0.2)

        binned = bin_spike_times(original_spike_times, time_bins)
        recovered = binned_to_spike_times(binned, time_bins)

        assert len(recovered[0]) == len(original_spike_times[0])
