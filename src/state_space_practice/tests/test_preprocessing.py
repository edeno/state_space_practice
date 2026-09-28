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
        import warnings as _w

        rng = np.random.default_rng(42)
        # uniform(0, 10) returns values in [0, 10), and we give explicit
        # headroom past 10 in time_bins so the last bin fully covers the
        # spike range and no drop-warning can fire regardless of seed.
        spike_times = [rng.uniform(0, 10, 100) for _ in range(5)]
        time_bins = np.arange(0, 10.02, 0.01)  # last bin ends at ~10.02

        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning would raise
            counts = bin_spike_times(spike_times, time_bins)

        total_original = sum(len(st) for st in spike_times)
        total_binned = counts.sum()
        assert total_binned == total_original

    def test_out_of_window_warning_and_no_funnel(self):
        """Spikes past time_bins[-1] + dt must be dropped (not funneled
        into the last bin) and must trigger a UserWarning.

        Regression test for the silent-funnel bug that caused catastrophic
        log-likelihoods in PlaceFieldModel.fit_sgd on real data where a
        sub-window of spike times was being binned. Before the fix, all
        out-of-window spikes got assigned to bin T-1, producing one
        pathological count that drove a single Fisher step to O(100)
        step magnitude and log-likelihoods of -1e8.
        """
        import pytest

        time_bins = np.arange(0, 5, 1.0)  # dt=1, t_end=5
        spike_times = [
            np.array([0.5, 1.5, 2.5]),  # all in-window
            np.concatenate([np.array([3.5]), np.full(1000, 99.9)]),  # 1 in, 1000 out
        ]
        with pytest.warns(UserWarning, match="1000 spike"):
            counts = bin_spike_times(spike_times, time_bins)
        # Neuron 0: all 3 in-window spikes counted
        assert counts[:, 0].sum() == 3
        # Neuron 1: only the one in-window spike counted; 1000 out-of-window dropped
        assert counts[:, 1].sum() == 1
        # Critically: bin T-1 (the last bin) does NOT contain the out-of-window flood
        assert counts[-1, 1] == 0

    def test_warn_on_drops_suppression(self):
        """warn_on_drops=False suppresses the out-of-window warning."""
        import warnings

        time_bins = np.arange(0, 5, 1.0)
        spike_times = [np.array([100.0])]  # out-of-window
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would raise
            counts = bin_spike_times(spike_times, time_bins, warn_on_drops=False)
        assert counts.sum() == 0

    def test_accepts_single_1d_array(self):
        """Shorthand: passing a single 1D array treats it as one unit."""
        time_bins = np.arange(0, 5, 1.0)
        counts = bin_spike_times(np.array([0.5, 1.5, 2.5]), time_bins)
        assert counts.shape == (5, 1)
        assert counts.sum() == 3


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


# ---------------------------------------------------------------------------
# Property tests (Hypothesis)
# ---------------------------------------------------------------------------

import warnings  # noqa: E402

import pytest  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import arrays  # noqa: E402

from state_space_practice.exceptions import StateSpaceWarning  # noqa: E402


@st.composite
def _binning_problem(draw, max_spikes: int = 80):
    """Uniform bins plus spike times that include exact edges and out-of-window."""
    n_bins = draw(st.integers(2, 40))
    t0 = draw(st.floats(-5.0, 5.0))
    dt = draw(st.floats(1e-3, 1.0))
    time_bins = t0 + dt * np.arange(n_bins)
    t_end = float(time_bins[-1]) + float(time_bins[1] - time_bins[0])
    edges = np.append(time_bins, t_end)
    n_spikes = draw(st.integers(0, max_spikes))
    inside = (
        draw(arrays(np.float64, n_spikes, elements=st.floats(0.0, 1.0))) * (t_end - t0)
        + t0
    )
    n_edge = draw(st.integers(0, 5))
    on_edges = edges[draw(arrays(np.int64, n_edge, elements=st.integers(0, n_bins)))]
    n_out = draw(st.integers(0, 3))
    outside = np.concatenate(
        [
            t0 - dt * draw(arrays(np.float64, n_out, elements=st.floats(1e-3, 3.0))),
            t_end + dt * draw(arrays(np.float64, n_out, elements=st.floats(1e-3, 3.0))),
        ]
    )
    spikes = np.concatenate([inside, on_edges, outside])
    return time_bins, edges, spikes


def _brute_force_counts(spikes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """[e_k, e_{k+1}) for every bin, the last one closed on the right."""
    counts = np.zeros(edges.size - 1, dtype=int)
    for t in spikes:
        for k in range(edges.size - 1):
            last = k == edges.size - 2
            if edges[k] <= t and (t < edges[k + 1] or (last and t == edges[k + 1])):
                counts[k] += 1
                break
    return counts


class TestBinningProperties:
    @settings(deadline=None, max_examples=60)
    @given(problem=_binning_problem())
    def test_counts_match_brute_force_and_are_conserved(self, problem):
        time_bins, edges, spikes = problem
        counts = bin_spike_times(spikes, time_bins, warn_on_drops=False)[:, 0]
        np.testing.assert_array_equal(counts, _brute_force_counts(spikes, edges))
        in_window = np.sum((spikes >= edges[0]) & (spikes <= edges[-1]))
        assert counts.sum() == in_window

    @settings(deadline=None, max_examples=40)
    @given(problem=_binning_problem(), seed=st.integers(0, 2**31 - 1))
    def test_invariant_to_permuting_spikes_and_equivariant_in_units(
        self, problem, seed
    ):
        time_bins, _edges, spikes = problem
        rng = np.random.default_rng(seed)
        units = [spikes, spikes[: spikes.size // 2], rng.permutation(spikes)]
        counts = bin_spike_times(units, time_bins, warn_on_drops=False)
        np.testing.assert_array_equal(counts[:, 0], counts[:, 2])
        order = rng.permutation(3)
        permuted = bin_spike_times(
            [units[i] for i in order], time_bins, warn_on_drops=False
        )
        np.testing.assert_array_equal(permuted, counts[:, order])

    @settings(deadline=None, max_examples=40)
    @given(problem=_binning_problem())
    def test_warns_exactly_when_spikes_are_dropped(self, problem):
        time_bins, edges, spikes = problem
        n_out = int(np.sum((spikes < edges[0]) | (spikes > edges[-1])))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bin_spike_times(spikes, time_bins)
        relevant = [w for w in caught if issubclass(w.category, StateSpaceWarning)]
        assert len(relevant) == (1 if n_out else 0)
        if n_out:
            assert f"{n_out} spike(s)" in str(relevant[0].message)

    @settings(deadline=None, max_examples=40)
    @given(problem=_binning_problem())
    def test_clip_then_bin_equals_bin_without_warning(self, problem):
        time_bins, edges, spikes = problem
        clipped = clip_spike_times_to_window([spikes], edges[0], edges[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from_clipped = bin_spike_times(clipped, time_bins)
        np.testing.assert_array_equal(
            from_clipped, bin_spike_times([spikes], time_bins, warn_on_drops=False)
        )

    @settings(deadline=None, max_examples=40)
    @given(
        n_bins=st.integers(2, 40),
        t0=st.floats(-5.0, 5.0),
        dt=st.floats(1e-3, 1.0),
        data=st.data(),
    )
    def test_counts_to_times_to_counts_round_trip(self, n_bins, t0, dt, data):
        time_bins = t0 + dt * np.arange(n_bins)
        counts = data.draw(arrays(np.int64, (n_bins, 3), elements=st.integers(0, 6)))
        times = binned_to_spike_times(counts, time_bins)
        assert [t.size for t in times] == list(counts.sum(axis=0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # left edges are never out of window
            np.testing.assert_array_equal(bin_spike_times(times, time_bins), counts)

    @settings(deadline=None, max_examples=40)
    @given(problem=_binning_problem())
    def test_times_to_counts_to_times_floors_to_left_edges(self, problem):
        time_bins, edges, spikes = problem
        counts = bin_spike_times(spikes, time_bins, warn_on_drops=False)
        recovered = binned_to_spike_times(counts, time_bins, neuron_idx=0)
        kept = np.sort(spikes[(spikes >= edges[0]) & (spikes <= edges[-1])])
        assert recovered.size == kept.size
        index = np.searchsorted(time_bins, kept, side="right") - 1
        np.testing.assert_array_equal(np.sort(recovered), time_bins[index])
        assert np.all(recovered <= kept)

    def test_boundary_bins(self):
        """First edge -> bin 0; interior edge -> the bin it opens; right end of
        the window -> last bin; one ulp outside either end -> dropped + warned."""
        time_bins = np.array([0.0, 0.25, 0.5, 0.75])
        t_end = 1.0
        counts = bin_spike_times(np.array([0.0, 0.25, 0.75, t_end]), time_bins)
        np.testing.assert_array_equal(counts[:, 0], [1, 1, 0, 2])
        for outside in (np.nextafter(0.0, -1.0), np.nextafter(t_end, 2.0)):
            with pytest.warns(StateSpaceWarning, match="1 spike"):
                counts = bin_spike_times(np.array([outside, 0.5]), time_bins)
            np.testing.assert_array_equal(counts[:, 0], [0, 0, 1, 0])


class TestRateAndInterpolationProperties:
    @settings(deadline=None, max_examples=40)
    @given(problem=_binning_problem())
    def test_firing_rate_times_duration_is_the_binned_count(self, problem):
        time_bins, edges, spikes = problem
        rates = compute_firing_rates([spikes], edges[0], edges[-1])
        counts = bin_spike_times(spikes, time_bins, warn_on_drops=False)
        np.testing.assert_allclose(
            rates[0] * (edges[-1] - edges[0]), counts.sum(), rtol=1e-12
        )

    @settings(deadline=None, max_examples=30)
    @given(
        n=st.integers(5, 30),
        a=st.floats(-3.0, 3.0),
        b=st.floats(-3.0, 3.0),
        kind=st.sampled_from(["linear", "nearest", "cubic"]),
        seed=st.integers(0, 2**31 - 1),
    )
    def test_interpolation_is_linear_and_exact_at_knots(self, n, a, b, kind, seed):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0.0, 10.0, n)) + 1e-3 * np.arange(n)
        v1 = rng.normal(size=(n, 2))
        v2 = rng.normal(size=(n, 2))
        new_t = rng.uniform(t[0], t[-1], 25)
        combo = interpolate_to_new_times(a * v1 + b * v2, t, new_t, kind=kind)
        separate = a * interpolate_to_new_times(
            v1, t, new_t, kind=kind
        ) + b * interpolate_to_new_times(v2, t, new_t, kind=kind)
        np.testing.assert_allclose(combo, separate, atol=1e-9)
        np.testing.assert_allclose(
            interpolate_to_new_times(v1, t, t, kind=kind), v1, atol=1e-9
        )

    @settings(deadline=None, max_examples=30)
    @given(slope=st.floats(-5.0, 5.0), intercept=st.floats(-5.0, 5.0))
    def test_linear_interpolation_reproduces_affine_functions(self, slope, intercept):
        t = np.array([0.0, 0.3, 1.1, 2.0, 3.7])
        new_t = np.linspace(-1.0, 5.0, 13)  # includes linear extrapolation
        values = slope * t + intercept
        np.testing.assert_allclose(
            interpolate_to_new_times(values, t, new_t),
            slope * new_t + intercept,
            atol=1e-12,
        )


class TestBoutProperties:
    @settings(deadline=None, max_examples=60)
    @given(
        speed=arrays(np.float64, st.integers(0, 60), elements=st.floats(0.0, 10.0)),
        threshold=st.floats(0.0, 10.0),
        min_duration=st.integers(1, 5),
        above=st.booleans(),
    )
    def test_bouts_are_exactly_the_long_runs(
        self, speed, threshold, min_duration, above
    ):
        mask = speed > threshold if above else speed < threshold
        bouts = identify_behavioral_bouts(speed, threshold, min_duration, above)
        covered = np.zeros(speed.size, dtype=bool)
        for start, end in bouts:
            assert end - start >= min_duration
            assert np.all(mask[start:end])
            assert start == 0 or not mask[start - 1]  # maximal on the left
            assert end == speed.size or not mask[end]  # maximal on the right
            covered[start:end] = True
        # every masked sample outside a bout lies in a run shorter than min
        runs = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts, ends = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
        long_runs = [(s, e) for s, e in zip(starts, ends) if e - s >= min_duration]
        assert bouts == [(int(s), int(e)) for s, e in long_runs]
        assert covered.sum() == sum(e - s for s, e in long_runs)
