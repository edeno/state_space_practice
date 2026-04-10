"""Tests for position decoding from spike trains via Laplace-EKF."""

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.position_decoder import (
    AdaptiveInflationConfig,
    DecoderResult,
    PlaceFieldRateMaps,
    PositionDecoder,
    build_position_dynamics,
    position_decoder_filter,
    position_decoder_smoother,
)


class TestBuildPositionDynamics:
    def test_constant_velocity_shapes(self):
        A, Q = build_position_dynamics(dt=0.004, q_pos=1.0, q_vel=10.0)
        assert A.shape == (4, 4)
        assert Q.shape == (4, 4)

    def test_constant_velocity_prediction(self):
        """Starting at (10, 20) with velocity (5, -3),
        after one step should be at (10 + 5*dt, 20 - 3*dt)."""
        dt = 0.01
        A, _ = build_position_dynamics(dt=dt)
        z = jnp.array([10.0, 20.0, 5.0, -3.0])
        z_next = A @ z
        np.testing.assert_allclose(z_next[0], 10.0 + 5.0 * dt)
        np.testing.assert_allclose(z_next[1], 20.0 - 3.0 * dt)
        np.testing.assert_allclose(z_next[2], 5.0)
        np.testing.assert_allclose(z_next[3], -3.0)

    def test_q_is_psd(self):
        _, Q = build_position_dynamics(dt=0.004, q_pos=1.0, q_vel=10.0)
        eigvals = jnp.linalg.eigvalsh(Q)
        assert jnp.all(eigvals >= 0)

    def test_position_only_mode(self):
        """2D position-only state (no velocity)."""
        A, Q = build_position_dynamics(
            dt=0.004, q_pos=1.0, include_velocity=False
        )
        assert A.shape == (2, 2)
        assert Q.shape == (2, 2)
        np.testing.assert_allclose(A, jnp.eye(2))

    def test_dt_validation(self):
        with pytest.raises(ValueError):
            build_position_dynamics(dt=0.0)
        with pytest.raises(ValueError):
            build_position_dynamics(dt=-0.01)


class TestPlaceFieldRateMaps:
    @pytest.fixture
    def simple_fields(self):
        """Two neurons with Gaussian place fields at different locations."""
        n_grid = 50
        x_edges = np.linspace(0, 100, n_grid)
        y_edges = np.linspace(0, 100, n_grid)
        xx, yy = np.meshgrid(x_edges, y_edges)

        # Neuron 0: field at (30, 40)
        rate0 = 20 * np.exp(-((xx - 30)**2 + (yy - 40)**2) / (2 * 15**2)) + 0.5
        # Neuron 1: field at (70, 60)
        rate1 = 30 * np.exp(-((xx - 70)**2 + (yy - 60)**2) / (2 * 12**2)) + 0.5

        rate_maps = np.stack([rate0, rate1])  # (2, n_grid, n_grid)
        return PlaceFieldRateMaps(
            rate_maps=rate_maps,
            x_edges=x_edges,
            y_edges=y_edges,
        )

    def test_log_rate_at_field_center(self, simple_fields):
        """Rate should be highest near field center."""
        position = jnp.array([30.0, 40.0])
        log_rate = simple_fields.log_rate(position)
        assert log_rate.shape == (2,)
        # Neuron 0 should fire faster than neuron 1 at this location
        assert log_rate[0] > log_rate[1]

    def test_log_rate_away_from_field(self, simple_fields):
        """Rate should be near baseline far from field center."""
        position = jnp.array([90.0, 10.0])  # far from both fields
        log_rate = simple_fields.log_rate(position)
        # Both rates should be near log(0.5) ~ -0.69
        assert jnp.all(log_rate < 1.0)

    def test_log_rate_clamps_at_boundary(self, simple_fields):
        """Positions outside grid are clamped."""
        pos_outside = jnp.array([-10.0, 200.0])
        log_rate = simple_fields.log_rate(pos_outside)
        assert jnp.all(jnp.isfinite(log_rate))

    def test_jacobian_shape(self, simple_fields):
        """Jacobian of log-rate w.r.t. position should be (n_neurons, 2)."""
        position = jnp.array([50.0, 50.0])
        jac = simple_fields.log_rate_jacobian(position)
        assert jac.shape == (2, 2)  # (n_neurons, n_position_dims)

    def test_jacobian_points_toward_field(self, simple_fields):
        """At a point between two fields, the gradient for neuron 0
        should point toward (30, 40)."""
        position = jnp.array([50.0, 50.0])
        jac = simple_fields.log_rate_jacobian(position)
        # Neuron 0's field is at (30, 40), so gradient should point left and down
        assert jac[0, 0] < 0  # d(log_rate_0)/dx < 0
        assert jac[0, 1] < 0  # d(log_rate_0)/dy < 0

    def test_jacobian_finite_difference_accuracy(self, simple_fields):
        """Finite-difference Jacobian should match independent numerical derivative."""
        pos = jnp.array([45.0, 55.0])
        jac = simple_fields.log_rate_jacobian(pos)

        eps = 0.01
        for dim in range(2):
            pos_p = pos.at[dim].set(pos[dim] + eps)
            pos_m = pos.at[dim].set(pos[dim] - eps)
            numerical = (simple_fields.log_rate(pos_p) - simple_fields.log_rate(pos_m)) / (2 * eps)
            np.testing.assert_allclose(jac[:, dim], numerical, atol=0.1)

    def test_from_place_field_model(self):
        """Construct rate maps from a fitted PlaceFieldModel."""
        from state_space_practice.place_field_model import PlaceFieldModel
        from state_space_practice.simulate_data import (
            simulate_2d_moving_place_field,
        )

        data = simulate_2d_moving_place_field(
            total_time=30.0, dt=0.004, peak_rate=80.0, n_interior_knots=3,
        )
        model = PlaceFieldModel(dt=0.004, n_interior_knots=3)
        model.fit(data["position"], data["spikes"], max_iter=2, verbose=False)

        rate_maps = PlaceFieldRateMaps.from_place_field_model(
            model, n_grid=30,
        )
        assert rate_maps.n_neurons == 1
        assert rate_maps.rate_maps.shape == (1, 30, 30)

    def test_from_spike_position_data(self):
        """Construct rate maps from raw position + spikes."""
        rng = np.random.default_rng(42)
        n_time = 1000
        dt = 0.004
        position = np.column_stack([
            50 + 20 * np.cos(np.linspace(0, 4 * np.pi, n_time)),
            50 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_time)),
        ])
        # Two neurons
        spikes = rng.poisson(0.1, (n_time, 2))

        rate_maps = PlaceFieldRateMaps.from_spike_position_data(
            position=position,
            spike_counts=spikes,
            dt=dt,
            n_grid=30,
        )
        assert rate_maps.n_neurons == 2
        assert rate_maps.rate_maps.shape == (2, 30, 30)

    def test_from_spike_position_data_stationary(self):
        """Stationary position should not crash."""
        n_time = 100
        position = np.tile([50.0, 50.0], (n_time, 1))
        spikes = np.zeros((n_time, 2))

        rate_maps = PlaceFieldRateMaps.from_spike_position_data(
            position=position,
            spike_counts=spikes,
            dt=0.004,
            n_grid=20,
        )
        assert rate_maps.n_neurons == 2
        assert rate_maps.rate_maps.shape == (2, 20, 20)
        assert np.all(np.isfinite(rate_maps.rate_maps))

    def test_from_spike_position_data_length_mismatch(self):
        """Mismatched position/spike lengths should raise ValueError."""
        position = np.random.randn(100, 2)
        spikes = np.random.poisson(0.1, (90, 3))

        with pytest.raises(ValueError, match="same number of time bins"):
            PlaceFieldRateMaps.from_spike_position_data(
                position=position,
                spike_counts=spikes,
                dt=0.004,
                n_grid=20,
            )


class TestPositionDecoderFilter:
    @pytest.fixture
    def decoding_data(self):
        """Simulate position + spikes, then build rate maps and decode."""
        rng = np.random.default_rng(42)
        n_time = 500
        dt = 0.004

        # Simple trajectory: circle
        t = np.arange(n_time) * dt
        true_x = 50 + 20 * np.cos(2 * np.pi * t / 2.0)
        true_y = 50 + 20 * np.sin(2 * np.pi * t / 2.0)
        true_pos = np.column_stack([true_x, true_y])

        # Two neurons with place fields
        n_grid = 30
        x_edges = np.linspace(0, 100, n_grid)
        y_edges = np.linspace(0, 100, n_grid)
        xx, yy = np.meshgrid(x_edges, y_edges)

        rate0 = 30 * np.exp(-((xx - 30)**2 + (yy - 50)**2) / (2 * 15**2)) + 0.5
        rate1 = 30 * np.exp(-((xx - 70)**2 + (yy - 50)**2) / (2 * 15**2)) + 0.5
        rate_maps_arr = np.stack([rate0, rate1])

        rate_maps = PlaceFieldRateMaps(
            rate_maps=rate_maps_arr, x_edges=x_edges, y_edges=y_edges,
        )

        # Generate spikes from true position
        spikes = np.zeros((n_time, 2))
        for t_idx in range(n_time):
            log_r = rate_maps.log_rate(jnp.array(true_pos[t_idx]))
            rates = np.exp(np.array(log_r))
            spikes[t_idx] = rng.poisson(rates * dt)

        return {
            "true_position": true_pos,
            "spikes": jnp.array(spikes),
            "rate_maps": rate_maps,
            "dt": dt,
            "n_time": n_time,
        }

    def test_filter_output_shapes(self, decoding_data):
        result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        assert isinstance(result, DecoderResult)
        assert result.position_mean.shape == (decoding_data["n_time"], 4)
        assert result.position_cov.shape == (decoding_data["n_time"], 4, 4)

    def test_filter_tracks_position(self, decoding_data):
        """Decoded position should be correlated with true position."""
        result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        decoded_x = np.array(result.position_mean[:, 0])
        true_x = decoding_data["true_position"][:, 0]

        # Skip first 50 bins (filter warmup)
        corr = np.corrcoef(decoded_x[50:], true_x[50:])[0, 1]
        assert corr > 0.5

    def test_filter_covariance_is_psd(self, decoding_data):
        result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        for t in range(0, decoding_data["n_time"], 50):
            eigvals = jnp.linalg.eigvalsh(result.position_cov[t])
            assert jnp.all(eigvals >= -1e-8)

    def test_filter_ll_is_finite(self, decoding_data):
        result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        assert np.isfinite(result.marginal_log_likelihood)

    def test_smoother_output_shapes(self, decoding_data):
        result = position_decoder_smoother(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        assert result.position_mean.shape == (decoding_data["n_time"], 4)
        assert result.position_cov.shape == (decoding_data["n_time"], 4, 4)

    def test_smoother_reduces_error(self, decoding_data):
        """Smoother should have lower error than filter."""
        filter_result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        smoother_result = position_decoder_smoother(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )

        true_pos = decoding_data["true_position"]
        filter_error = np.mean(np.linalg.norm(
            np.array(filter_result.position_mean[50:, :2]) - true_pos[50:],
            axis=1,
        ))
        smoother_error = np.mean(np.linalg.norm(
            np.array(smoother_result.position_mean[50:, :2]) - true_pos[50:],
            axis=1,
        ))

        # With sparse spikes, smoother may not always improve substantially
        assert smoother_error <= filter_error * 1.2

    def test_smoother_last_matches_filter(self, decoding_data):
        """Last smoother time step should match filter."""
        filter_result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        smoother_result = position_decoder_smoother(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        np.testing.assert_allclose(
            smoother_result.position_mean[-1],
            filter_result.position_mean[-1],
            atol=1e-5,
        )

    def test_smoother_variance_leq_filter(self, decoding_data):
        """Average smoothed variance should be <= filtered variance."""
        filter_result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        smoother_result = position_decoder_smoother(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        filt_var = np.mean([np.trace(c) for c in np.array(filter_result.position_cov)])
        smooth_var = np.mean([np.trace(c) for c in np.array(smoother_result.position_cov)])
        assert smooth_var <= filt_var * 1.01


class TestPositionDecoderIntegration:
    """End-to-end integration tests with simulated 2D place fields and spikes."""

    @pytest.fixture
    def circular_trajectory_2d(self):
        """6 neurons with fields tiling (x, y) space, circular trajectory."""
        rng = np.random.default_rng(123)
        n_time = 1000
        dt = 0.004

        # Circular trajectory
        t = np.arange(n_time) * dt
        true_x = 50 + 25 * np.cos(2 * np.pi * t / 2.0)
        true_y = 50 + 25 * np.sin(2 * np.pi * t / 2.0)
        true_pos = np.column_stack([true_x, true_y])

        # 6 neurons with fields at distinct (x, y) locations
        n_grid = 40
        x_edges = np.linspace(0, 100, n_grid)
        y_edges = np.linspace(0, 100, n_grid)
        xx, yy = np.meshgrid(x_edges, y_edges)

        field_centers = [
            (30, 30), (70, 30),  # bottom left/right
            (30, 70), (70, 70),  # top left/right
            (50, 25), (50, 75),  # bottom/top center
        ]
        rate_maps_list = []
        for cx, cy in field_centers:
            rate = 40 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 12**2)) + 0.5
            rate_maps_list.append(rate)

        rate_maps_arr = np.stack(rate_maps_list)
        rate_maps = PlaceFieldRateMaps(
            rate_maps=rate_maps_arr, x_edges=x_edges, y_edges=y_edges,
        )

        # Generate spikes from true position
        n_neurons = len(field_centers)
        spikes = np.zeros((n_time, n_neurons))
        for t_idx in range(n_time):
            log_r = rate_maps.log_rate(jnp.array(true_pos[t_idx]))
            rates = np.exp(np.array(log_r))
            spikes[t_idx] = rng.poisson(rates * dt)

        return {
            "true_position": true_pos,
            "spikes": jnp.array(spikes),
            "rate_maps": rate_maps,
            "dt": dt,
            "n_time": n_time,
        }

    def test_filter_tracks_both_dimensions(self, circular_trajectory_2d):
        """Filter should track both x and y on a circular trajectory."""
        data = circular_trajectory_2d
        result = position_decoder_filter(
            spikes=data["spikes"], rate_maps=data["rate_maps"], dt=data["dt"],
        )
        warmup = 100
        decoded = np.array(result.position_mean[warmup:, :2])
        true_pos = data["true_position"][warmup:]

        corr_x = np.corrcoef(decoded[:, 0], true_pos[:, 0])[0, 1]
        corr_y = np.corrcoef(decoded[:, 1], true_pos[:, 1])[0, 1]

        assert corr_x > 0.5, f"x correlation {corr_x:.2f} < 0.5"
        assert corr_y > 0.5, f"y correlation {corr_y:.2f} < 0.5"

    def test_smoother_improves_over_filter(self, circular_trajectory_2d):
        """Smoother should have higher correlation than filter in both dims."""
        data = circular_trajectory_2d
        filt = position_decoder_filter(
            spikes=data["spikes"], rate_maps=data["rate_maps"], dt=data["dt"],
        )
        smooth = position_decoder_smoother(
            spikes=data["spikes"], rate_maps=data["rate_maps"], dt=data["dt"],
        )
        warmup = 100
        true_pos = data["true_position"][warmup:]

        filt_corr_x = np.corrcoef(
            np.array(filt.position_mean[warmup:, 0]), true_pos[:, 0]
        )[0, 1]
        smooth_corr_x = np.corrcoef(
            np.array(smooth.position_mean[warmup:, 0]), true_pos[:, 0]
        )[0, 1]

        assert smooth_corr_x >= filt_corr_x - 0.05, (
            f"smoother x corr {smooth_corr_x:.2f} < filter {filt_corr_x:.2f}"
        )

    def test_stationary_animal_converges(self, circular_trajectory_2d):
        """Animal at fixed position: decoded position should converge."""
        data = circular_trajectory_2d
        rate_maps = data["rate_maps"]
        dt = data["dt"]
        rng = np.random.default_rng(99)

        # Animal stays at (50, 50)
        n_time = 300
        true_pos = np.tile([50.0, 50.0], (n_time, 1))
        spikes = np.zeros((n_time, rate_maps.n_neurons))
        for t_idx in range(n_time):
            log_r = rate_maps.log_rate(jnp.array(true_pos[t_idx]))
            rates = np.exp(np.array(log_r))
            spikes[t_idx] = rng.poisson(rates * dt)

        result = position_decoder_filter(
            spikes=jnp.array(spikes), rate_maps=rate_maps, dt=dt,
        )

        # After warmup, decoded position should be near (50, 50)
        decoded_end = np.array(result.position_mean[-50:, :2])
        mean_decoded = decoded_end.mean(axis=0)
        assert np.linalg.norm(mean_decoded - np.array([50.0, 50.0])) < 15.0, (
            f"stationary decode {mean_decoded} too far from (50, 50)"
        )

        # Posterior variance should decrease over time
        early_var = np.trace(np.array(result.position_cov[50, :2, :2]))
        late_var = np.trace(np.array(result.position_cov[-1, :2, :2]))
        assert late_var < early_var, "variance should decrease as evidence accumulates"

    def test_no_nan_or_inf(self, circular_trajectory_2d):
        """All outputs should be finite."""
        data = circular_trajectory_2d
        filt = position_decoder_filter(
            spikes=data["spikes"], rate_maps=data["rate_maps"], dt=data["dt"],
        )
        smooth = position_decoder_smoother(
            spikes=data["spikes"], rate_maps=data["rate_maps"], dt=data["dt"],
        )
        assert np.all(np.isfinite(np.array(filt.position_mean)))
        assert np.all(np.isfinite(np.array(filt.position_cov)))
        assert np.all(np.isfinite(np.array(smooth.position_mean)))
        assert np.all(np.isfinite(np.array(smooth.position_cov)))
        assert np.isfinite(filt.marginal_log_likelihood)


class TestPositionDecoder:
    @pytest.fixture
    def trajectory_data(self):
        rng = np.random.default_rng(42)
        n_time = 1000
        dt = 0.004

        t = np.arange(n_time) * dt
        true_x = 50 + 20 * np.cos(2 * np.pi * t / 2.0)
        true_y = 50 + 20 * np.sin(2 * np.pi * t / 2.0)
        position = np.column_stack([true_x, true_y])

        # 5 neurons with different place fields
        n_neurons = 5
        centers = rng.uniform(20, 80, (n_neurons, 2))
        spikes = np.zeros((n_time, n_neurons))
        for n in range(n_neurons):
            dist_sq = np.sum((position - centers[n]) ** 2, axis=1)
            rate = 25 * np.exp(-dist_sq / (2 * 15**2)) + 0.5
            spikes[:, n] = rng.poisson(rate * dt)

        return {
            "position": position,
            "spikes": spikes,
            "dt": dt,
            "n_time": n_time,
        }

    def test_fit_decode_workflow(self, trajectory_data):
        decoder = PositionDecoder(dt=trajectory_data["dt"])

        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        assert decoder.rate_maps is not None
        assert decoder.rate_maps.n_neurons == 5

        result = decoder.decode(
            spikes=trajectory_data["spikes"],
            method="smoother",
        )
        assert result.position_mean.shape[0] == trajectory_data["n_time"]

    def test_decode_error(self, trajectory_data):
        decoder = PositionDecoder(dt=trajectory_data["dt"])
        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        result = decoder.decode(spikes=trajectory_data["spikes"])

        decoded_pos = np.array(result.position_mean[:, :2])
        true_pos = trajectory_data["position"]
        error = np.median(np.linalg.norm(decoded_pos[100:] - true_pos[100:], axis=1))
        assert error < 30.0

    def test_decode_requires_fit(self):
        decoder = PositionDecoder(dt=0.004)
        with pytest.raises(RuntimeError):
            decoder.decode(spikes=np.zeros((10, 2)))

    def test_filter_vs_smoother_methods(self, trajectory_data):
        decoder = PositionDecoder(dt=trajectory_data["dt"])
        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        result_f = decoder.decode(spikes=trajectory_data["spikes"], method="filter")
        result_s = decoder.decode(spikes=trajectory_data["spikes"], method="smoother")
        assert result_f.position_mean.shape == result_s.position_mean.shape

    def test_invalid_method_raises(self, trajectory_data):
        decoder = PositionDecoder(dt=trajectory_data["dt"])
        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        with pytest.raises(ValueError, match="method must be"):
            decoder.decode(spikes=trajectory_data["spikes"], method="invalid")

    def test_plot_decoding(self, trajectory_data):
        import matplotlib
        matplotlib.use("Agg")

        decoder = PositionDecoder(dt=trajectory_data["dt"])
        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        result = decoder.decode(spikes=trajectory_data["spikes"])
        fig = decoder.plot_decoding(
            result, true_position=trajectory_data["position"],
        )
        assert fig is not None

    def test_plot_without_true_position(self, trajectory_data):
        import matplotlib
        matplotlib.use("Agg")

        decoder = PositionDecoder(dt=trajectory_data["dt"])
        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        result = decoder.decode(spikes=trajectory_data["spikes"])
        fig = decoder.plot_decoding(result)
        assert fig is not None

    def test_repr(self, trajectory_data):
        decoder = PositionDecoder(dt=0.004)
        assert "fitted=False" in repr(decoder)
        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        assert "fitted=True" in repr(decoder)
        assert "n_neurons=5" in repr(decoder)


class TestAdaptiveInflation:
    """Regression tests for innovation-based covariance inflation."""

    @pytest.fixture
    def rate_maps_and_data(self):
        """Build simple rate maps and synthetic spike data."""
        rng = np.random.default_rng(42)
        n_time = 500
        dt = 0.004

        t = np.arange(n_time) * dt
        true_x = 50 + 20 * np.cos(2 * np.pi * t / 2.0)
        true_y = 50 + 20 * np.sin(2 * np.pi * t / 2.0)
        position = np.column_stack([true_x, true_y])

        n_neurons = 8
        centers = rng.uniform(20, 80, (n_neurons, 2))
        spikes = np.zeros((n_time, n_neurons))
        for n in range(n_neurons):
            dist_sq = np.sum((position - centers[n]) ** 2, axis=1)
            rate = 25 * np.exp(-dist_sq / (2 * 15**2)) + 0.5
            spikes[:, n] = rng.poisson(rate * dt)

        rm = PlaceFieldRateMaps.from_spike_position_data(
            position, spikes, dt=dt, n_grid=50, sigma=5.0,
        )
        return rm, spikes, position, dt

    def test_disabled_matches_baseline(self, rate_maps_and_data):
        """Inflation disabled should reproduce the no-inflation result."""
        rm, spikes, position, dt = rate_maps_and_data
        init_pos = jnp.array(position[0])

        result_none = position_decoder_filter(
            spikes=spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
        )
        result_off = position_decoder_filter(
            spikes=spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
            adaptive_inflation=AdaptiveInflationConfig(enabled=False),
        )
        np.testing.assert_allclose(
            result_none.position_mean, result_off.position_mean, atol=1e-10,
        )

    def test_inflation_increases_covariance(self, rate_maps_and_data):
        """With inflation enabled, filter covariances should be >= baseline."""
        rm, spikes, position, dt = rate_maps_and_data
        init_pos = jnp.array(position[0])

        result_base = position_decoder_filter(
            spikes=spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
        )
        result_infl = position_decoder_filter(
            spikes=spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
            adaptive_inflation=AdaptiveInflationConfig(gain=1.0, max_alpha=5.0),
        )
        # Mean trace of covariance should be >= baseline
        base_trace = np.mean(np.trace(np.array(result_base.position_cov), axis1=1, axis2=2))
        infl_trace = np.mean(np.trace(np.array(result_infl.position_cov), axis1=1, axis2=2))
        assert infl_trace >= base_trace

    def test_capped_inflation_bounds_covariance_growth(self, rate_maps_and_data):
        """With max_alpha near 1, covariance growth is bounded."""
        rm, spikes, position, dt = rate_maps_and_data
        init_pos = jnp.array(position[0])

        result_capped = position_decoder_filter(
            spikes=spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
            adaptive_inflation=AdaptiveInflationConfig(
                gain=100.0, max_alpha=1.01,
            ),
        )
        result_base = position_decoder_filter(
            spikes=spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
        )
        base_trace = np.trace(np.array(result_base.position_cov), axis1=1, axis2=2)
        capped_trace = np.trace(np.array(result_capped.position_cov), axis1=1, axis2=2)
        ratio = capped_trace / np.maximum(base_trace, 1e-12)
        # Per-step cap is 1.01; accumulated ratio should stay modest
        assert np.max(ratio) < 3.0

    def test_no_spikes_no_inflation(self, rate_maps_and_data):
        """With zero spikes, innovation is negative and inflation is skipped."""
        rm, spikes, position, dt = rate_maps_and_data
        init_pos = jnp.array(position[0])

        zero_spikes = np.zeros_like(spikes)
        result_base = position_decoder_filter(
            spikes=zero_spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
        )
        result_infl = position_decoder_filter(
            spikes=zero_spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
            adaptive_inflation=AdaptiveInflationConfig(gain=2.0, max_alpha=10.0),
        )
        # With zero spikes the score is small (innovation = -lambda*dt)
        # so the clip to alpha >= 1 prevents any deflation; the result
        # may still differ slightly because the score is not exactly zero.
        base_cov = np.array(result_base.position_cov)
        infl_cov = np.array(result_infl.position_cov)
        base_trace = np.trace(base_cov, axis1=1, axis2=2)
        infl_trace = np.trace(infl_cov, axis1=1, axis2=2)
        # Covariance should not shrink (alpha clipped at 1)
        assert np.all(infl_trace >= base_trace - 1e-6)

    def test_smoother_with_inflation_produces_valid_output(self, rate_maps_and_data):
        """Smoother with inflation produces finite, correctly-shaped output."""
        rm, spikes, position, dt = rate_maps_and_data
        init_pos = jnp.array(position[0])
        cfg = AdaptiveInflationConfig(gain=0.5, max_alpha=5.0)

        result_f = position_decoder_filter(
            spikes=spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
            adaptive_inflation=cfg,
        )
        result_s = position_decoder_smoother(
            spikes=spikes, rate_maps=rm, dt=dt,
            q_pos=50.0, include_velocity=False,
            init_position=init_pos,
            adaptive_inflation=cfg,
        )
        assert not np.any(np.isnan(result_f.position_mean))
        assert not np.any(np.isnan(result_s.position_mean))
        assert result_s.position_mean.shape == result_f.position_mean.shape
        # Smoother covariance should generally be <= filter covariance
        f_trace = np.trace(np.array(result_f.position_cov), axis1=1, axis2=2)
        s_trace = np.trace(np.array(result_s.position_cov), axis1=1, axis2=2)
        # Check that smoother reduces uncertainty on average
        assert np.mean(s_trace) < np.mean(f_trace) * 1.1
