"""Tests for the PlaceFieldModel class and supporting functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.place_field_model import (
    PlaceFieldModel,
    build_2d_spline_basis,
    evaluate_basis,
)
from state_space_practice.simulate_data import simulate_2d_moving_place_field

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------
# build_2d_spline_basis / evaluate_basis
# ------------------------------------------------------------------


class TestBuild2dSplineBasis:
    """Tests for the 2D tensor-product B-spline basis construction."""

    @pytest.fixture()
    def position(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.uniform(0, 100, (500, 2))

    def test_output_shapes(self, position: np.ndarray) -> None:
        dm, info = build_2d_spline_basis(position, n_interior_knots=4)
        n_basis = (4 + 3) ** 2  # cubic B-spline: n_knots + degree
        assert dm.shape == (500, n_basis)
        assert info["n_basis"] == n_basis

    def test_basis_info_keys(self, position: np.ndarray) -> None:
        _, info = build_2d_spline_basis(position, n_interior_knots=3)
        required = {
            "knots_x", "knots_y", "x_lo", "x_hi",
            "y_lo", "y_hi", "formula", "n_basis", "n_interior_knots",
        }
        assert required.issubset(info.keys())

    def test_evaluate_roundtrip(self, position: np.ndarray) -> None:
        dm, info = build_2d_spline_basis(position, n_interior_knots=3)
        dm2 = evaluate_basis(position, info)
        np.testing.assert_allclose(dm, dm2, atol=1e-10)

    def test_evaluate_clips_out_of_bounds(self, position: np.ndarray) -> None:
        _, info = build_2d_spline_basis(position, n_interior_knots=3)
        oob = np.array([[-10.0, -10.0], [200.0, 200.0]])
        result = evaluate_basis(oob, info)
        assert result.shape == (2, info["n_basis"])
        assert np.all(np.isfinite(result))

    def test_invalid_position_shape(self) -> None:
        with pytest.raises(ValueError, match="must be .* 2"):
            build_2d_spline_basis(np.zeros((10, 3)))

    def test_evaluate_invalid_shape(self) -> None:
        pos = np.random.default_rng(0).uniform(0, 100, (50, 2))
        _, info = build_2d_spline_basis(pos, n_interior_knots=3)
        with pytest.raises(ValueError, match="must be .* 2"):
            evaluate_basis(np.zeros((10,)), info)


# ------------------------------------------------------------------
# PlaceFieldModel
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_data() -> dict:
    """Short simulation for fast tests."""
    return simulate_2d_moving_place_field(
        total_time=30.0,
        dt=0.020,
        arena_size=80.0,
        peak_rate=25.0,
        background_rate=1.0,
        n_interior_knots=3,
        rng=np.random.default_rng(42),
    )


class TestPlaceFieldModelInit:
    """Tests for construction and validation."""

    def test_default_construction(self) -> None:
        m = PlaceFieldModel(dt=0.004)
        assert m.dt == 0.004
        assert m.n_interior_knots == 5

    def test_invalid_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            PlaceFieldModel(dt=-1.0)

    def test_invalid_knots(self) -> None:
        with pytest.raises(ValueError, match="n_interior_knots"):
            PlaceFieldModel(dt=0.004, n_interior_knots=0)

    def test_invalid_noise_structure(self) -> None:
        with pytest.raises(ValueError, match="process_noise_structure"):
            PlaceFieldModel(dt=0.004, process_noise_structure="full")

    def test_from_place_field_width(self) -> None:
        m = PlaceFieldModel.from_place_field_width(
            dt=0.004,
            place_field_width=30.0,
            arena_range_x=(0, 100),
            arena_range_y=(0, 100),
        )
        assert m.n_interior_knots == 10

    def test_repr_unfitted(self) -> None:
        m = PlaceFieldModel(dt=0.004)
        r = repr(m)
        assert "fitted=False" in r
        assert "process_noise_structure=" in r

    def test_init_cov_scale(self) -> None:
        m = PlaceFieldModel(dt=0.004, init_cov_scale=5.0)
        assert m.init_cov_scale == 5.0


class TestPlaceFieldModelFit:
    """Tests for the EM fitting procedure."""

    def test_fit_smoke(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        lls = model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=5,
            verbose=False,
        )
        assert len(lls) >= 1
        assert all(np.isfinite(ll) for ll in lls)

    def test_ll_increases(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        lls = model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=10,
            verbose=False,
        )
        # EM guarantees non-decreasing LL (within numerical tolerance).
        # The EM loop breaks if LL decreases, so all recorded pairs must
        # be non-decreasing within the relative tolerance used by check_converged.
        for i in range(1, len(lls)):
            assert lls[i] >= lls[i - 1] - 1e-3

    def test_smoother_populated(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        n_time = len(sim_data["spikes"])
        n_basis = model.n_basis
        assert model.smoother_mean.shape == (n_time, n_basis)
        assert model.smoother_cov.shape == (n_time, n_basis, n_basis)
        assert not jnp.any(jnp.isnan(model.smoother_mean))

    def test_mismatched_lengths(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        with pytest.raises(ValueError, match="same number of time bins"):
            model.fit(sim_data["position"][:10], sim_data["spikes"])

    def test_multidim_spikes_rejected(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        bad_spikes = np.zeros((len(sim_data["spikes"]), 2))
        with pytest.raises(ValueError, match="1D"):
            model.fit(sim_data["position"], bad_spikes)

    def test_max_iter_warning(self, sim_data: dict, caplog) -> None:
        import logging

        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        with caplog.at_level(logging.WARNING):
            # max_iter=1: single iteration, no previous LL to compare, so
            # convergence/decrease checks are never reached -> else clause fires
            model.fit(
                sim_data["position"],
                sim_data["spikes"],
                max_iter=1,
                verbose=False,
            )
        assert "maximum iterations" in caplog.text.lower()

    def test_repr_fitted(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        r = repr(model)
        assert "fitted=True" in r
        assert "n_basis=" in r

    def test_fit_isotropic(self, sim_data: dict) -> None:
        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            process_noise_structure="isotropic",
        )
        lls = model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        assert len(lls) >= 1
        # isotropic: all diagonal elements should be equal
        diag = jnp.diag(model.process_cov)
        np.testing.assert_allclose(diag, diag[0])

    def test_fit_update_transition_matrix(self, sim_data: dict) -> None:
        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            update_transition_matrix=True,
        )
        lls = model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        assert len(lls) >= 1
        # Transition matrix should no longer be identity
        assert not jnp.allclose(model.transition_matrix, jnp.eye(model.n_basis))

    def test_negative_spikes_rejected(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        bad_spikes = -1 * np.ones(len(sim_data["spikes"]))
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(sim_data["position"], bad_spikes)


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------


class TestPlaceFieldModelPredict:
    """Tests for prediction methods."""

    @pytest.fixture()
    def fitted_model(self, sim_data: dict) -> PlaceFieldModel:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=5,
            verbose=False,
        )
        return model

    def test_predict_rate_map_shapes(self, fitted_model: PlaceFieldModel) -> None:
        grid, x, y = fitted_model.make_grid(n_grid=10)
        rate, ci = fitted_model.predict_rate_map(grid)
        assert rate.shape == (100,)
        assert ci.shape == (100, 2)
        assert np.all(np.isfinite(rate))
        assert np.all(rate >= 0)
        assert np.all(ci[:, 0] <= ci[:, 1])

    def test_predict_rate_map_with_time_slice(
        self, fitted_model: PlaceFieldModel
    ) -> None:
        grid, _, _ = fitted_model.make_grid(n_grid=10)
        rate, ci = fitted_model.predict_rate_map(grid, time_slice=slice(0, 100))
        assert rate.shape == (100,)
        assert np.all(np.isfinite(rate))

    def test_predict_center_shapes(self, fitted_model: PlaceFieldModel) -> None:
        grid, _, _ = fitted_model.make_grid(n_grid=10)
        centers = fitted_model.predict_center(grid, n_blocks=5)
        assert centers.shape == (5, 2)
        assert np.all(np.isfinite(centers))

    def test_make_grid_shapes(self, fitted_model: PlaceFieldModel) -> None:
        grid, x, y = fitted_model.make_grid(n_grid=20)
        assert grid.shape == (400, 2)
        assert x.shape == (20,)
        assert y.shape == (20,)

    def test_not_fitted_raises(self) -> None:
        model = PlaceFieldModel(dt=0.004)
        with pytest.raises(RuntimeError, match="Call model.fit"):
            model.predict_rate_map(np.zeros((10, 2)))
        with pytest.raises(RuntimeError, match="Call model.fit"):
            model.predict_center(np.zeros((10, 2)))
        with pytest.raises(RuntimeError, match="Call model.fit"):
            model.make_grid()
        with pytest.raises(RuntimeError, match="Call model.fit"):
            model.get_state_confidence_interval()


# ------------------------------------------------------------------
# score
# ------------------------------------------------------------------


class TestPlaceFieldModelScore:
    """Tests for held-out scoring."""

    def test_score_returns_finite(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        ll = model.score(sim_data["position"], sim_data["spikes"])
        assert np.isfinite(ll)

    def test_score_not_fitted(self) -> None:
        model = PlaceFieldModel(dt=0.004)
        with pytest.raises(RuntimeError, match="Call model.fit"):
            model.score(np.zeros((10, 2)), np.zeros(10))

    def test_score_mismatched_lengths(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        with pytest.raises(ValueError, match="same number of time bins"):
            model.score(sim_data["position"][:10], sim_data["spikes"])


# ------------------------------------------------------------------
# bin_spike_times
# ------------------------------------------------------------------


class TestBinSpikeTimes:
    """Tests for the spike binning utility."""

    def test_known_spike_train(self) -> None:
        time_bins = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        spike_times = np.array([0.5, 0.7, 2.3, 4.0])
        counts = PlaceFieldModel.bin_spike_times(spike_times, time_bins)
        # bin 0: [0, 1) -> 0.5, 0.7 = 2 spikes
        # bin 1: [1, 2) -> 0 spikes
        # bin 2: [2, 3) -> 2.3 = 1 spike
        # bin 3: [3, 4) -> 4.0 = 1 spike (searchsorted returns 4, -1 = bin 3)
        # bin 4: empty
        np.testing.assert_array_equal(counts, [2, 0, 1, 1, 0])

    def test_spike_at_session_start(self) -> None:
        """Spike exactly at time_bins[0] gets searchsorted index 0, -1 = -1, discarded."""
        time_bins = np.arange(0, 5, 1.0)
        counts = PlaceFieldModel.bin_spike_times(np.array([0.0]), time_bins)
        # Spike at t=0 is discarded by the (t_i, t_{i+1}] convention
        assert counts.sum() == 0

    def test_empty_spike_train(self) -> None:
        time_bins = np.arange(0, 10, 1.0)
        counts = PlaceFieldModel.bin_spike_times(np.array([]), time_bins)
        np.testing.assert_array_equal(counts, np.zeros(len(time_bins), dtype=int))


# ------------------------------------------------------------------
# drift_summary
# ------------------------------------------------------------------


class TestDriftSummary:
    """Tests for the drift analysis method."""

    def test_drift_summary_keys(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        summary = model.drift_summary(n_grid=10, n_blocks=5)
        assert "centers" in summary
        assert "total_drift" in summary
        assert "cumulative_drift" in summary
        assert "peak_rate_per_block" in summary
        assert "block_times" in summary
        assert summary["centers"].shape == (5, 2)
        assert np.isfinite(summary["total_drift"])


# ------------------------------------------------------------------
# get_state_confidence_interval
# ------------------------------------------------------------------


class TestGetStateConfidenceInterval:
    """Tests for the state CI method."""

    def test_ci_shapes(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        ci = model.get_state_confidence_interval()
        n_time = len(sim_data["spikes"])
        assert ci.shape == (n_time, model.n_basis, 2)
        assert jnp.all(ci[..., 0] <= ci[..., 1])
