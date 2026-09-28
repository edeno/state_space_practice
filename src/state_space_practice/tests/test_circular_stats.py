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


# ---------------------------------------------------------------------------
# Oracles: scipy.stats and closed forms
# ---------------------------------------------------------------------------

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import arrays  # noqa: E402
from scipy import stats  # noqa: E402


def _angles(min_size: int = 1, max_size: int = 60):
    return st.integers(min_size, max_size).flatmap(
        lambda n: arrays(np.float64, n, elements=st.floats(-10.0, 10.0))
    )


def _legacy_circular_std(phases: np.ndarray) -> float:
    """The hand-written implementation replaced by the scipy delegation."""
    resultant = float(np.abs(np.mean(np.exp(1j * phases))))
    resultant = np.clip(resultant, 1e-10, 1.0)
    return float(np.sqrt(-2 * np.log(resultant)))


def _std_roundoff(std: float) -> float:
    """Round-off scale of sqrt(-2 ln R) given R known to a few ulp.

    d std / d R = -1 / (R std), so a few-ulp error in R becomes ~1e-15 / std,
    saturating at the sqrt(2 eps) ~ 2e-8 floor for (numerically) identical
    phases.
    """
    return min(3e-8, 1e-14 / max(std, 1e-300)) + 1e-12 * std


class TestCircularMeanOracle:
    @settings(deadline=None, max_examples=50)
    @given(phases=_angles())
    def test_matches_scipy_circmean_on_the_circle(self, phases):
        resultant = mean_resultant_length(phases)
        assume(resultant > 1e-6)  # the mean direction is undefined at R = 0
        ours = circular_mean(phases)
        theirs = stats.circmean(phases, high=np.pi, low=-np.pi)
        assert -np.pi < ours <= np.pi
        assert angular_distance(ours, theirs) < 1e-13 / resultant

    def test_branch_cut_convention_differs_from_scipy(self):
        """Why circular_mean is *not* delegated to scipy.stats.circmean.

        np.angle returns (-pi, pi]; scipy's wrap ``(x - low) % period + low``
        returns [-pi, pi) and adds ulp(pi)-sized absolute round-off, so a
        mean at +pi becomes -pi and a tiny mean loses all relative precision.
        """
        assert circular_mean(np.array([np.pi])) == np.pi
        assert stats.circmean(np.array([np.pi]), high=np.pi, low=-np.pi) == -np.pi
        assert circular_mean(np.array([1e-20])) == 1e-20
        assert stats.circmean(np.array([1e-20]), high=np.pi, low=-np.pi) == 0.0


class TestCircularStdOracle:
    @settings(deadline=None, max_examples=50)
    @given(phases=_angles())
    def test_matches_scipy_circstd_and_legacy(self, phases):
        ours = circular_std(phases)
        legacy = _legacy_circular_std(phases)
        with np.errstate(divide="ignore"):
            theirs = float(stats.circstd(phases))
        tol = _std_roundoff(legacy)
        assert abs(ours - legacy) <= tol, (ours, legacy)
        if mean_resultant_length(phases) > 1e-9:  # away from the legacy floor
            assert abs(ours - theirs) <= tol, (ours, theirs)

    @settings(deadline=None, max_examples=30)
    @given(half_angle=st.floats(0.0, 1.5))
    def test_two_point_closed_form(self, half_angle):
        """For phases {+a, -a}: R = cos a and std = sqrt(-2 ln cos a)."""
        phases = np.array([half_angle, -half_angle])
        expected = np.sqrt(-2.0 * np.log(np.cos(half_angle)))
        assert abs(circular_std(phases) - expected) <= _std_roundoff(expected)

    def test_uniform_phases_hit_the_documented_floor(self):
        """R ~ 1e-17 is floored at 1e-10 (scipy alone would return 8.6)."""
        phases = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        assert mean_resultant_length(phases) < 1e-10  # guard
        assert circular_std(phases) == pytest.approx(np.sqrt(-2 * np.log(1e-10)))
        assert float(stats.circstd(phases)) > circular_std(phases) + 1.0


class TestMeanResultantLengthOracle:
    @settings(deadline=None, max_examples=50)
    @given(phases=_angles())
    def test_matches_scipy(self, phases):
        ours = mean_resultant_length(phases)
        # 1 - circvar is the only angle-level scipy route (absolute accuracy)
        assert abs(ours - (1.0 - stats.circvar(phases))) < 1e-14
        with np.errstate(invalid="ignore", divide="ignore"):
            vectors = np.column_stack([np.cos(phases), np.sin(phases)])
            ds = stats.directional_stats(vectors).mean_resultant_length
        assert abs(ours - ds) < 1e-14

    @settings(deadline=None, max_examples=30)
    @given(n=st.integers(2, 40), offset=st.floats(-np.pi, np.pi))
    def test_equally_spaced_phases_have_zero_resultant(self, n, offset):
        phases = offset + 2 * np.pi * np.arange(n) / n
        assert mean_resultant_length(phases) < 1e-14
        assert mean_resultant_length(phases[:1]) == pytest.approx(1.0, abs=1e-15)


def _phases_with_statistic(n: int, z: float) -> np.ndarray:
    """Phases whose Rayleigh statistic n * R^2 equals ``z`` (n even or odd)."""
    target = np.sqrt(z / n)
    pairs = n // 2
    if n % 2 == 0:
        half = np.arccos(target)
        return np.concatenate([np.full(pairs, half), np.full(pairs, -half)])
    half = np.arccos((n * target - 1.0) / (n - 1))
    return np.concatenate([[0.0], np.full(pairs, half), np.full(pairs, -half)])


class TestRayleighOracle:
    def test_statistic_construction(self):
        for n, z in ((10, 3.0), (49, 6.0), (50, 6.0)):
            phases = _phases_with_statistic(n, z)
            assert phases.size == n
            assert n * mean_resultant_length(phases) ** 2 == pytest.approx(z)

    @pytest.mark.parametrize("z", [1.0, 3.0, 6.0])
    def test_p_value_matches_monte_carlo_null(self, z):
        """p = P(n R^2 >= z) under uniformity, vs 400k simulated samples (n=10).

        The O(1/n) correction matters here: the plain exp(-z) is off by more
        than 5 Monte-Carlo standard errors at z = 3 and 6 (guard).
        """
        n, reps = 10, 400_000
        rng = np.random.default_rng(1234)
        theta = rng.uniform(-np.pi, np.pi, size=(reps, n))
        stat = n * np.abs(np.mean(np.exp(1j * theta), axis=1)) ** 2
        p_mc = float(np.mean(stat >= z))
        se = np.sqrt(p_mc * (1 - p_mc) / reps)
        _, p = rayleigh_test(_phases_with_statistic(n, z))
        assert abs(p - p_mc) < 4 * se + 2e-4, (p, p_mc, se)
        if z > 1.0:
            assert abs(np.exp(-z) - p_mc) > 5 * se

    @pytest.mark.parametrize("z", [3.0, 6.0])
    def test_p_value_is_continuous_across_sample_sizes(self, z):
        """Regression: the O(1/n) correction used to stop at n = 50.

        At z = 6 the p-value jumped by +14% between n = 49 and n = 50 (0.00217
        -> 0.00248; Monte Carlo at n = 50: 0.00223 +- 0.00005).
        """
        p = {n: rayleigh_test(_phases_with_statistic(n, z))[1] for n in (49, 50, 51)}
        assert abs(p[50] / p[49] - 1.0) < 0.005
        assert abs(p[51] / p[50] - 1.0) < 0.005

    def test_large_n_reduces_to_exp_minus_z(self):
        _, p = rayleigh_test(_phases_with_statistic(10_000, 3.0))
        assert p == pytest.approx(np.exp(-3.0), rel=1e-3)


class TestCircularCorrelationOracle:
    @settings(deadline=None, max_examples=40)
    @given(
        phases=_angles(min_size=5),
        shift=st.floats(-np.pi, np.pi),
        seed=st.integers(0, 2**31 - 1),
    )
    def test_matches_jammalamadaka_sengupta_formula(self, phases, shift, seed):
        """r = sum sin(a - abar) sin(b - bbar) / sqrt(sum sin^2 * sum sin^2)."""
        other = phases + np.random.default_rng(seed).normal(0.0, 0.5, phases.size)
        assume(mean_resultant_length(phases) > 1e-3)
        assume(mean_resultant_length(other) > 1e-3)
        sa = np.sin(phases - stats.circmean(phases))
        sb = np.sin(other - stats.circmean(other))
        assume(np.sum(sa**2) > 1e-6 and np.sum(sb**2) > 1e-6)
        expected = np.sum(sa * sb) / np.sqrt(np.sum(sa**2) * np.sum(sb**2))
        r = circular_correlation(phases, other)
        assert abs(r - expected) < 1e-9
        # rotation invariance of either variable
        assert abs(circular_correlation(phases + shift, other) - r) < 1e-9
        # exact co-rotation gives +1, reflection -1
        assert circular_correlation(phases, phases + shift) == pytest.approx(1.0)
        assert circular_correlation(phases, shift - phases) == pytest.approx(-1.0)


class TestWrapAndDistanceOracle:
    @settings(deadline=None, max_examples=50)
    @given(x=arrays(np.float64, 20, elements=st.floats(-1e3, 1e3)))
    def test_wrap_to_pi_is_the_2pi_representative(self, x):
        w = wrap_to_pi(x)
        assert np.all((w > -np.pi - 1e-12) & (w <= np.pi + 1e-12))
        turns = (x - w) / (2 * np.pi)
        np.testing.assert_allclose(turns, np.round(turns), atol=1e-9)

    @settings(deadline=None, max_examples=50)
    @given(
        a=arrays(np.float64, 20, elements=st.floats(-50.0, 50.0)),
        b=arrays(np.float64, 20, elements=st.floats(-50.0, 50.0)),
        c=arrays(np.float64, 20, elements=st.floats(-50.0, 50.0)),
    )
    def test_angular_distance_is_a_metric_on_the_circle(self, a, b, c):
        d_ab = angular_distance(a, b)
        d = np.mod(a - b, 2 * np.pi)
        np.testing.assert_allclose(d_ab, np.minimum(d, 2 * np.pi - d), atol=1e-11)
        np.testing.assert_allclose(d_ab, angular_distance(b, a), atol=1e-12)
        assert np.all(d_ab <= angular_distance(a, c) + angular_distance(c, b) + 1e-11)
        assert np.all((d_ab >= 0) & (d_ab <= np.pi))
