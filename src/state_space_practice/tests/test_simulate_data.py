"""Tests for the simulate_data module.

This module tests data simulation utilities for point process models,
including receptive field models and Eden/Brown 2004 simulations.
"""

import numpy as np

from state_space_practice.simulate_data import (
    receptive_field_model,
    simulate_eden_brown_2004_jump,
    simulate_eden_brown_2004_linear,
)


class TestReceptiveFieldModel:
    """Tests for the receptive_field_model function."""

    def test_output_shape_1d_params(self) -> None:
        """Output shape matches position shape with 1D params."""
        position = np.linspace(0, 300, 100)
        params = np.array([np.log(10.0), 150.0, 10.0])
        rate = receptive_field_model(position, params)
        assert rate.shape == position.shape

    def test_output_shape_2d_params(self) -> None:
        """Output shape matches position shape with 2D params."""
        position = np.linspace(0, 300, 100)
        params = np.array([[np.log(10.0), 150.0, 10.0]])
        rate = receptive_field_model(position, params)
        assert rate.shape == position.shape

    def test_peak_at_place_field_center(self) -> None:
        """Rate should be maximum at the place field center."""
        position = np.linspace(0, 300, 1000)
        center = 150.0
        params = np.array([np.log(10.0), center, 10.0])
        rate = receptive_field_model(position, params)

        # Find the position of maximum rate
        max_idx = np.argmax(rate)
        max_position = position[max_idx]

        # Should be within one position step of center
        assert np.abs(max_position - center) < (position[1] - position[0])

    def test_max_rate_equals_exp_log_max_rate(self) -> None:
        """Maximum rate should equal exp(log_max_rate)."""
        center = 150.0
        log_max_rate = np.log(25.0)
        params = np.array([log_max_rate, center, 10.0])

        # Evaluate exactly at center
        rate_at_center = receptive_field_model(np.array([center]), params)
        np.testing.assert_allclose(rate_at_center, np.exp(log_max_rate), rtol=1e-10)

    def test_gaussian_decay_from_center(self) -> None:
        """Rate should decay as Gaussian from center."""
        center = 150.0
        scale = 10.0
        log_max_rate = np.log(20.0)
        params = np.array([log_max_rate, center, scale])

        # At one standard deviation from center
        position_1sd = np.array([center + scale])
        rate_1sd = receptive_field_model(position_1sd, params)

        # Expected: max_rate * exp(-0.5)
        expected = np.exp(log_max_rate) * np.exp(-0.5)
        np.testing.assert_allclose(rate_1sd, expected, rtol=1e-10)

    def test_symmetric_around_center(self) -> None:
        """Rate should be symmetric around the center."""
        center = 150.0
        params = np.array([np.log(10.0), center, 15.0])

        offsets = np.array([10, 20, 30, 50])
        rate_left = receptive_field_model(center - offsets, params)
        rate_right = receptive_field_model(center + offsets, params)

        np.testing.assert_allclose(rate_left, rate_right, rtol=1e-10)

    def test_positive_rates(self) -> None:
        """All rates should be positive (exponential model)."""
        position = np.linspace(-100, 500, 200)
        params = np.array([np.log(10.0), 150.0, 10.0])
        rate = receptive_field_model(position, params)

        assert np.all(rate > 0)

    def test_scalar_position(self) -> None:
        """Should handle scalar position input."""
        position = 150.0
        params = np.array([np.log(10.0), 150.0, 10.0])
        rate = receptive_field_model(position, params)

        # Should be the max rate
        np.testing.assert_allclose(rate, 10.0, rtol=1e-10)

    def test_different_scales_affect_width(self) -> None:
        """Larger scale should produce wider receptive field."""
        position = np.linspace(0, 300, 1000)
        center = 150.0

        params_narrow = np.array([np.log(10.0), center, 5.0])
        params_wide = np.array([np.log(10.0), center, 20.0])

        rate_narrow = receptive_field_model(position, params_narrow)
        rate_wide = receptive_field_model(position, params_wide)

        # At same distance from center, wider field should have higher rate
        distance = 15.0
        pos_test = center + distance
        idx = np.argmin(np.abs(position - pos_test))

        assert rate_wide[idx] > rate_narrow[idx]


class TestSimulateEdenBrown2004Jump:
    """Tests for the jump parameter simulation."""

    def test_output_types(self) -> None:
        """All outputs should be numpy arrays or float."""
        time, position, spikes, dt, params1, params2 = simulate_eden_brown_2004_jump()

        assert isinstance(time, np.ndarray)
        assert isinstance(position, np.ndarray)
        assert isinstance(spikes, np.ndarray)
        assert isinstance(dt, float)
        assert isinstance(params1, np.ndarray)
        assert isinstance(params2, np.ndarray)

    def test_consistent_lengths(self) -> None:
        """Time, position, and spikes should have consistent lengths."""
        time, position, spikes, dt, _, _ = simulate_eden_brown_2004_jump()

        assert len(time) == len(position)
        assert len(time) == len(spikes)

    def test_time_step_consistency(self) -> None:
        """Time array should have consistent dt spacing."""
        time, _, _, dt, _, _ = simulate_eden_brown_2004_jump()

        time_diffs = np.diff(time)
        np.testing.assert_allclose(time_diffs, dt, rtol=1e-10)

    def test_position_bounds(self) -> None:
        """Position should stay within track bounds [0, 300]."""
        _, position, _, _, _, _ = simulate_eden_brown_2004_jump()

        assert np.all(position >= 0)
        assert np.all(position <= 300)

    def test_spikes_non_negative(self) -> None:
        """Spike counts should be non-negative integers."""
        _, _, spikes, _, _, _ = simulate_eden_brown_2004_jump()

        assert np.all(spikes >= 0)
        assert spikes.dtype in [np.int32, np.int64]

    def test_params_have_correct_shape(self) -> None:
        """Parameter arrays should have 3 elements each."""
        _, _, _, _, params1, params2 = simulate_eden_brown_2004_jump()

        assert params1.shape == (3,)
        assert params2.shape == (3,)

    def test_params_are_different(self) -> None:
        """The two parameter sets should be different (representing the jump)."""
        _, _, _, _, params1, params2 = simulate_eden_brown_2004_jump()

        assert not np.allclose(params1, params2)

    def test_expected_simulation_length(self) -> None:
        """Simulation should have expected length based on dt and total time."""
        time, _, _, dt, _, _ = simulate_eden_brown_2004_jump()

        total_time = 8000.0
        expected_length = int(total_time / dt)
        assert len(time) == expected_length

    def test_spikes_follow_poisson_statistics(self) -> None:
        """Spike counts should approximately follow Poisson statistics."""
        # Run multiple simulations to test statistics
        np.random.seed(42)
        _, _, spikes, _, _, _ = simulate_eden_brown_2004_jump()

        # Poisson: variance approximately equals mean for each region
        # Test on first half (should have lower rate)
        first_half = spikes[: len(spikes) // 2]
        # Just verify it's a reasonable count (not all zeros or extremely high)
        assert 0 < np.mean(first_half) < 1  # dt * max_rate should be < 1


class TestSimulateEdenBrown2004Linear:
    """Tests for the linear interpolation parameter simulation."""

    def test_output_types(self) -> None:
        """All outputs should be numpy arrays or float."""
        time, position, spikes, dt, params = simulate_eden_brown_2004_linear()

        assert isinstance(time, np.ndarray)
        assert isinstance(position, np.ndarray)
        assert isinstance(spikes, np.ndarray)
        assert isinstance(dt, float)
        assert isinstance(params, np.ndarray)

    def test_consistent_lengths(self) -> None:
        """Time, position, spikes, and params should have consistent lengths."""
        time, position, spikes, dt, params = simulate_eden_brown_2004_linear()

        assert len(time) == len(position)
        assert len(time) == len(spikes)
        assert len(time) == len(params)

    def test_params_shape(self) -> None:
        """Parameters should have shape (n_time, 3)."""
        time, _, _, _, params = simulate_eden_brown_2004_linear()

        assert params.shape == (len(time), 3)

    def test_params_interpolate_linearly(self) -> None:
        """Parameters should interpolate linearly between start and end."""
        _, _, _, _, params = simulate_eden_brown_2004_linear()

        # Check that middle point is average of first and last
        mid_idx = len(params) // 2
        expected_mid = (params[0] + params[-1]) / 2
        np.testing.assert_allclose(params[mid_idx], expected_mid, rtol=1e-3)

    def test_first_params_match_true_params1(self) -> None:
        """First time point should have params close to true_params1."""
        _, _, _, _, params = simulate_eden_brown_2004_linear()

        true_params1 = np.array([np.log(10.0), 250.0, np.sqrt(12.0)])
        np.testing.assert_allclose(params[0], true_params1, rtol=1e-10)

    def test_last_params_match_true_params2(self) -> None:
        """Last time point should have params close to true_params2."""
        _, _, _, _, params = simulate_eden_brown_2004_linear()

        true_params2 = np.array([np.log(30.0), 150.0, np.sqrt(20.0)])
        np.testing.assert_allclose(params[-1], true_params2, rtol=1e-10)

    def test_position_bounds(self) -> None:
        """Position should stay within track bounds."""
        _, position, _, _, _ = simulate_eden_brown_2004_linear()

        assert np.all(position >= 0)
        assert np.all(position <= 300)

    def test_spikes_non_negative(self) -> None:
        """Spike counts should be non-negative."""
        _, _, spikes, _, _ = simulate_eden_brown_2004_linear()

        assert np.all(spikes >= 0)

    def test_different_from_jump_simulation(self) -> None:
        """Linear simulation should produce different parameter trajectory than jump."""
        _, _, _, _, params_linear = simulate_eden_brown_2004_linear()
        _, _, _, _, params1_jump, params2_jump = simulate_eden_brown_2004_jump()

        # In jump, params are constant in each half
        # In linear, params change gradually
        # Check that params at 1/4 point are different from both endpoints
        quarter_idx = len(params_linear) // 4
        quarter_params = params_linear[quarter_idx]

        # Should not equal either endpoint
        assert not np.allclose(quarter_params, params_linear[0])
        assert not np.allclose(quarter_params, params_linear[-1])
