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

    def test_peak_at_place_field_center(self) -> None:
        """Rate should be maximum at the place field center."""
        position = np.linspace(0, 300, 1000)
        center = 150.0
        params = np.array([np.log(10.0), center, 10.0])
        rate = receptive_field_model(position, params)

        max_idx = np.argmax(rate)
        max_position = position[max_idx]

        assert np.abs(max_position - center) < (position[1] - position[0])

    def test_max_rate_equals_exp_log_max_rate(self) -> None:
        """Maximum rate should equal exp(log_max_rate)."""
        center = 150.0
        log_max_rate = np.log(25.0)
        params = np.array([log_max_rate, center, 10.0])

        rate_at_center = receptive_field_model(np.array([center]), params)
        np.testing.assert_allclose(rate_at_center, np.exp(log_max_rate), rtol=1e-10)

    def test_gaussian_decay_from_center(self) -> None:
        """Rate should decay as Gaussian from center."""
        center = 150.0
        scale = 10.0
        log_max_rate = np.log(20.0)
        params = np.array([log_max_rate, center, scale])

        position_1sd = np.array([center + scale])
        rate_1sd = receptive_field_model(position_1sd, params)

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

    def test_different_scales_affect_width(self) -> None:
        """Larger scale should produce wider receptive field."""
        center = 150.0
        distance = 15.0

        params_narrow = np.array([np.log(10.0), center, 5.0])
        params_wide = np.array([np.log(10.0), center, 20.0])

        rate_narrow = receptive_field_model(np.array([center + distance]), params_narrow)
        rate_wide = receptive_field_model(np.array([center + distance]), params_wide)

        assert rate_wide > rate_narrow


class TestSimulateEdenBrown2004Jump:
    """Tests for the jump parameter simulation."""

    def test_consistent_lengths(self) -> None:
        """Time, position, and spikes should have consistent lengths."""
        time, position, spikes, dt, _, _ = simulate_eden_brown_2004_jump()

        assert len(time) == len(position) == len(spikes)

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

    def test_params_are_different(self) -> None:
        """The two parameter sets should be different (representing the jump)."""
        _, _, _, _, params1, params2 = simulate_eden_brown_2004_jump()

        assert not np.allclose(params1, params2)

    def test_spike_rate_consistent_with_model(self) -> None:
        """Mean spike count should be roughly consistent with the receptive field model."""
        rng = np.random.default_rng(42)
        _, position, spikes, dt, params1, _ = simulate_eden_brown_2004_jump(rng=rng)

        # First half uses params1
        n_half = len(spikes) // 2
        first_half_spikes = spikes[:n_half]
        first_half_position = position[:n_half]

        # Expected rate from model
        expected_rate = receptive_field_model(first_half_position, params1)
        expected_mean_count = np.mean(expected_rate * dt)
        observed_mean_count = np.mean(first_half_spikes)

        # Observed mean should be within an order of magnitude of expected
        np.testing.assert_allclose(
            observed_mean_count, expected_mean_count, rtol=1.0,
            err_msg="Observed spike rate differs from model prediction by >2x",
        )


class TestSimulateEdenBrown2004Linear:
    """Tests for the linear interpolation parameter simulation."""

    def test_params_interpolate_linearly(self) -> None:
        """Parameters should interpolate linearly between start and end."""
        _, _, _, _, params = simulate_eden_brown_2004_linear()

        mid_idx = len(params) // 2
        expected_mid = (params[0] + params[-1]) / 2
        np.testing.assert_allclose(params[mid_idx], expected_mid, rtol=1e-3)

    def test_first_and_last_params_match_ground_truth(self) -> None:
        """First and last time points should match the known true parameters."""
        _, _, _, _, params = simulate_eden_brown_2004_linear()

        true_params1 = np.array([np.log(10.0), 250.0, np.sqrt(12.0)])
        true_params2 = np.array([np.log(30.0), 150.0, np.sqrt(20.0)])
        np.testing.assert_allclose(params[0], true_params1, rtol=1e-10)
        np.testing.assert_allclose(params[-1], true_params2, rtol=1e-10)

    def test_position_bounds(self) -> None:
        """Position should stay within track bounds."""
        _, position, _, _, _ = simulate_eden_brown_2004_linear()

        assert np.all(position >= 0)
        assert np.all(position <= 300)

    def test_params_change_gradually(self) -> None:
        """Linear simulation should produce gradually changing parameters."""
        _, _, _, _, params = simulate_eden_brown_2004_linear()

        # Params at 1/4 point should differ from both endpoints
        quarter_idx = len(params) // 4
        quarter_params = params[quarter_idx]

        assert not np.allclose(quarter_params, params[0])
        assert not np.allclose(quarter_params, params[-1])
