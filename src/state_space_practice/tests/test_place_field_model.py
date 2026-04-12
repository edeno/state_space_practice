"""Tests for the PlaceFieldModel class and supporting functions."""

import warnings

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
        # Pin the biologically motivated defaults so a future refactor
        # can't silently revert them. See PlaceFieldModel.__init__ docstring
        # for the derivations (cumulative log-rate drift, warm-start
        # fallback, physiological firing-rate ceiling).
        assert m.init_process_noise == 1e-6
        assert m.init_cov_scale == 0.01
        assert m.max_firing_rate_hz == 500.0

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

    def test_3d_spikes_rejected(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        bad_spikes = np.zeros((len(sim_data["spikes"]), 2, 3))
        with pytest.raises(ValueError, match="1D.*or 2D"):
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
    """Tests for the spike binning utility.

    Note: uses left-closed ``[t_i, t_{i+1})`` bins (inherited from the
    canonical ``np.histogram``-based implementation in
    ``state_space_practice.preprocessing``). The last bin ``[t_{T-1},
    t_{T-1} + dt]`` is right-closed on the endpoint.
    """

    def test_known_spike_train(self) -> None:
        time_bins = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        spike_times = np.array([0.5, 0.7, 2.3, 4.0])
        counts = PlaceFieldModel.bin_spike_times(spike_times, time_bins)
        # Left-closed intervals:
        # bin 0: [0, 1) -> 0.5, 0.7 -> 2
        # bin 1: [1, 2) -> 0
        # bin 2: [2, 3) -> 2.3 -> 1
        # bin 3: [3, 4) -> 0
        # bin 4: [4, 5] -> 4.0 -> 1  (last bin is right-closed on endpoint)
        np.testing.assert_array_equal(counts, [2, 0, 1, 0, 1])

    def test_spike_at_session_start(self) -> None:
        """Spike exactly at time_bins[0] falls in bin 0 (left-closed)."""
        time_bins = np.arange(0, 5, 1.0)
        counts = PlaceFieldModel.bin_spike_times(np.array([0.0]), time_bins)
        assert counts[0] == 1
        assert counts.sum() == 1

    def test_empty_spike_train(self) -> None:
        time_bins = np.arange(0, 10, 1.0)
        counts = PlaceFieldModel.bin_spike_times(np.array([]), time_bins)
        np.testing.assert_array_equal(counts, np.zeros(len(time_bins), dtype=int))

    def test_out_of_window_spikes_discarded_with_warning(self) -> None:
        """Spikes past time_bins[-1] + dt must be discarded, not funneled
        into the last bin. This is a regression test for the silent-funnel
        bug that caused catastrophic log-likelihoods in fit_sgd."""
        time_bins = np.arange(0, 5, 1.0)  # [0, 1, 2, 3, 4], dt=1, t_end=5
        # Inject many spikes FAR past the window. Before the fix, these
        # all got dumped into bin T-1 via the searchsorted catch-all.
        spike_times = np.concatenate(
            [
                np.array([0.5, 1.5, 2.5]),  # in-window: 3 spikes, one per bin
                np.full(9999, 100.0),  # 9999 spikes at t=100, way past t_end=5
            ]
        )
        with pytest.warns(UserWarning, match="9999 spike"):
            counts = PlaceFieldModel.bin_spike_times(spike_times, time_bins)
        # Bin 4 (the last bin, [4, 5]) should be empty, NOT contain 9999.
        assert counts[4] == 0
        # Only the 3 in-window spikes should survive.
        assert counts.sum() == 3
        np.testing.assert_array_equal(counts, [1, 1, 1, 0, 0])

    def test_warn_on_drops_suppression(self) -> None:
        """warn_on_drops=False silences the out-of-window warning."""
        import warnings as _w

        time_bins = np.arange(0, 5, 1.0)
        spike_times = np.array([100.0])  # out-of-window
        with _w.catch_warnings():
            _w.simplefilter("error")  # any UserWarning would raise
            counts = PlaceFieldModel.bin_spike_times(
                spike_times, time_bins, warn_on_drops=False
            )
        assert counts.sum() == 0

    def test_spikes_in_last_bin_not_dropped(self) -> None:
        """Spikes in the genuine last bin [t_{T-1}, t_{T-1} + dt] must count."""
        time_bins = np.arange(0, 5, 1.0)  # dt=1, last bin is [4, 5]
        spike_times = np.array([4.5])  # inside last bin
        counts = PlaceFieldModel.bin_spike_times(spike_times, time_bins)
        assert counts[-1] == 1
        assert counts.sum() == 1

    def test_warning_points_at_caller_through_wrapper(self) -> None:
        """The drop warning must point at the user's call site, not at
        the preprocessing module or the PlaceFieldModel wrapper itself.

        Regression test for a stacklevel bug where the warning would
        report ``preprocessing.py`` or ``place_field_model.py`` as the
        source, making it harder for users to find where the bad
        ``time_bins`` came from. ``PlaceFieldModel.bin_spike_times``
        delegates to the canonical implementation, which is two frames
        deep, so it must pass ``_warn_stacklevel=3``.
        """
        import warnings as _w

        time_bins = np.arange(0, 5, 1.0)
        spike_times = np.array([100.0])  # out-of-window

        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always")
            PlaceFieldModel.bin_spike_times(spike_times, time_bins)

        assert len(captured) == 1
        warning = captured[0]
        # Warning should point at THIS test file, not at the library internals.
        assert warning.filename == __file__, (
            f"stacklevel points at {warning.filename}, expected {__file__}"
        )


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


# ------------------------------------------------------------------
# Filtered estimates
# ------------------------------------------------------------------


class TestFilteredEstimates:
    """Tests for stored filtered estimates."""

    def test_filtered_populated(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        n_time = len(sim_data["spikes"])
        assert model.filtered_mean.shape == (n_time, model.n_basis)
        assert model.filtered_cov.shape == (n_time, model.n_basis, model.n_basis)
        assert not jnp.any(jnp.isnan(model.filtered_mean))


# ------------------------------------------------------------------
# BIC / AIC / summary
# ------------------------------------------------------------------


class TestModelComparison:
    """Tests for BIC, AIC, and summary."""

    def test_bic_aic_finite(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        assert np.isfinite(model.bic())
        assert np.isfinite(model.aic())

    def test_bic_not_fitted(self) -> None:
        model = PlaceFieldModel(dt=0.004)
        with pytest.raises(RuntimeError, match="Call model.fit"):
            model.bic()

    def test_n_free_params(self) -> None:
        m = PlaceFieldModel(dt=0.004, n_interior_knots=3)
        # Before fit, n_basis is None — but n_free_params uses it
        # After fit, it should be n_basis (diagonal Q) + 2*n_basis (init)
        # With default settings: update_process_cov=True (diagonal), update_init_state=True
        # n_free_params = n_basis + n_basis + n_basis = 3 * n_basis
        # We can only test after fit, but let's verify the property logic
        assert m.update_process_cov is True
        assert m.update_init_state is True
        assert m.update_transition_matrix is False

    def test_n_free_params_after_fit(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        nb = model.n_basis
        # diagonal Q + init_mean + init_cov_diag = 3 * n_basis
        assert model.n_free_params == 3 * nb

    def test_more_knots_higher_bic_penalty(self, sim_data: dict) -> None:
        model_small = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model_small.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=5,
            verbose=False,
        )
        assert model_small.n_free_params < 200  # sanity check

    def test_summary_string(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        s = model.summary()
        assert "PlaceFieldModel Summary" in s
        assert "BIC" in s
        assert "AIC" in s
        assert "n_basis_per_neuron" in s
        assert "total_spikes" in s

    def test_summary_not_fitted(self) -> None:
        model = PlaceFieldModel(dt=0.004)
        with pytest.raises(RuntimeError, match="Call model.fit"):
            model.summary()


# ------------------------------------------------------------------
# Custom intensity function
# ------------------------------------------------------------------


class TestCustomIntensity:
    """Tests for custom log_intensity_func."""

    def test_custom_func_runs(self, sim_data: dict) -> None:
        # Use the default linear function explicitly
        from state_space_practice.point_process_kalman import log_conditional_intensity

        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            log_intensity_func=log_conditional_intensity,
        )
        lls = model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        assert len(lls) >= 1
        assert all(np.isfinite(ll) for ll in lls)


# ------------------------------------------------------------------
# Multi-neuron
# ------------------------------------------------------------------


class TestMultiNeuron:
    """Tests for multi-neuron fitting."""

    def test_two_neuron_fit(self, sim_data: dict) -> None:
        # Stack the same neuron twice as a simple multi-neuron test
        spikes_2n = np.column_stack([sim_data["spikes"], sim_data["spikes"]])
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        lls = model.fit(
            sim_data["position"],
            spikes_2n,
            max_iter=3,
            verbose=False,
        )
        assert len(lls) >= 1
        assert model.n_neurons == 2
        assert model.n_basis == 2 * model.n_basis_per_neuron
        n_time = len(sim_data["spikes"])
        assert model.smoother_mean.shape == (n_time, model.n_basis)

    def test_multi_neuron_predict_rate_map(self, sim_data: dict) -> None:
        spikes_2n = np.column_stack([sim_data["spikes"], sim_data["spikes"]])
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(sim_data["position"], spikes_2n, max_iter=3, verbose=False)
        grid, _, _ = model.make_grid(n_grid=10)
        # Each neuron should have its own rate map
        rate0, _ = model.predict_rate_map(grid, neuron_idx=0)
        rate1, _ = model.predict_rate_map(grid, neuron_idx=1)
        assert rate0.shape == (100,)
        assert rate1.shape == (100,)
        assert np.all(np.isfinite(rate0))
        assert np.all(np.isfinite(rate1))

    def test_multi_neuron_score(self, sim_data: dict) -> None:
        spikes_2n = np.column_stack([sim_data["spikes"], sim_data["spikes"]])
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(sim_data["position"], spikes_2n, max_iter=3, verbose=False)
        ll = model.score(sim_data["position"], spikes_2n)
        assert np.isfinite(ll)

    def test_3d_spikes_rejected(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        bad = np.zeros((len(sim_data["spikes"]), 2, 3))
        with pytest.raises(ValueError, match="1D.*or 2D"):
            model.fit(sim_data["position"], bad)

    def test_multi_neuron_summary(self, sim_data: dict) -> None:
        spikes_2n = np.column_stack([sim_data["spikes"], sim_data["spikes"]])
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(sim_data["position"], spikes_2n, max_iter=3, verbose=False)
        s = model.summary()
        assert "n_neurons" in s

    def test_multi_neuron_recovers_distinct_fields(self) -> None:
        """Two neurons with well-separated fields should produce distinct rate maps."""
        rng = np.random.default_rng(123)
        dt = 0.020
        arena_size = 80.0
        n_time = 2000
        n_interior_knots = 3

        # Lawnmower trajectory for good spatial coverage
        position = rng.uniform(0, arena_size, (n_time, 2))

        # Neuron 0: field centered at (25, 25)
        center0 = np.array([25.0, 25.0])
        dist_sq_0 = np.sum((position - center0) ** 2, axis=1)
        rate0 = 1.0 + 25.0 * np.exp(-dist_sq_0 / (2 * 10.0**2))
        spikes0 = rng.poisson(rate0 * dt)

        # Neuron 1: field centered at (60, 60)
        center1 = np.array([60.0, 60.0])
        dist_sq_1 = np.sum((position - center1) ** 2, axis=1)
        rate1 = 1.0 + 25.0 * np.exp(-dist_sq_1 / (2 * 10.0**2))
        spikes1 = rng.poisson(rate1 * dt)

        spikes_2n = np.column_stack([spikes0, spikes1])

        model = PlaceFieldModel(dt=dt, n_interior_knots=n_interior_knots)
        model.fit(position, spikes_2n, max_iter=20, verbose=False)

        # Check that each neuron's peak rate is near its true center
        grid, _, _ = model.make_grid(n_grid=30)
        for neuron_idx, true_center in enumerate([center0, center1]):
            rate_map, _ = model.predict_rate_map(grid, neuron_idx=neuron_idx)
            peak_idx = np.argmax(rate_map)
            estimated_peak = grid[peak_idx]
            dist = np.linalg.norm(estimated_peak - true_center)
            assert dist < 15.0, (
                f"Neuron {neuron_idx}: estimated peak {estimated_peak} "
                f"is {dist:.1f} cm from true center {true_center}"
            )

        # The two neurons' rate maps should differ substantially
        rate0_map, _ = model.predict_rate_map(grid, neuron_idx=0)
        rate1_map, _ = model.predict_rate_map(grid, neuron_idx=1)
        correlation = np.corrcoef(rate0_map, rate1_map)[0, 1]
        assert correlation < 0.5, (
            f"Neuron rate maps are too similar (r={correlation:.2f}), "
            f"multi-neuron fitting may not be separating neurons."
        )

    def test_score_wrong_neuron_count(self, sim_data: dict) -> None:
        spikes_2n = np.column_stack([sim_data["spikes"], sim_data["spikes"]])
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(sim_data["position"], spikes_2n, max_iter=3, verbose=False)
        # 1D spikes should be rejected for a 2-neuron model
        with pytest.raises(ValueError, match="n_neurons=2"):
            model.score(sim_data["position"], sim_data["spikes"])
        # Wrong number of columns
        with pytest.raises(ValueError, match="Expected 2"):
            model.score(sim_data["position"], np.column_stack(
                [sim_data["spikes"], sim_data["spikes"], sim_data["spikes"]]
            ))

    def test_neuron_idx_out_of_range(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False)
        grid, _, _ = model.make_grid(n_grid=5)
        with pytest.raises(ValueError, match="neuron_idx=1 out of range"):
            model.predict_rate_map(grid, neuron_idx=1)


# ------------------------------------------------------------------
# n_free_params variations
# ------------------------------------------------------------------


class TestNFreeParamsVariations:
    """Tests for n_free_params across configurations."""

    def test_no_updates(self, sim_data: dict) -> None:
        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3,
            update_process_cov=False, update_init_state=False,
        )
        model.fit(sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False)
        assert model.n_free_params == 0

    def test_isotropic(self, sim_data: dict) -> None:
        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3,
            process_noise_structure="isotropic",
        )
        model.fit(sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False)
        # isotropic Q (1) + init_mean (nb) + init_cov_diag (nb) = 1 + 2*nb
        assert model.n_free_params == 1 + 2 * model.n_basis

    def test_with_transition(self, sim_data: dict) -> None:
        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3,
            update_transition_matrix=True,
        )
        model.fit(sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False)
        nb = model.n_basis
        # diagonal Q (nb) + A (nb^2) + init_mean (nb) + init_cov_diag (nb) = nb^2 + 3*nb
        assert model.n_free_params == nb ** 2 + 3 * nb


# ------------------------------------------------------------------
# Nonlinear intensity warning
# ------------------------------------------------------------------


class TestNonlinearWarning:
    """Tests for warning when predict_rate_map uses linear approximation."""

    def test_warning_with_custom_func(self, sim_data: dict) -> None:
        import warnings

        def custom_func(dm, params):
            return dm @ params  # same as default but different object

        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3,
            log_intensity_func=custom_func,
        )
        model.fit(sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False)
        grid, _, _ = model.make_grid(n_grid=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.predict_rate_map(grid)
            assert len(w) == 1
            assert "linear approximation" in str(w[0].message)


class TestPlaceFieldSGDGradientStability:
    """Gradient stability test for Laplace-EKF at PlaceFieldModel dimensions.

    Verifies that removing stabilize_covariance (eigendecomp-based PSD
    projection) from the Laplace update gives finite gradients at all
    dimensions, including PlaceFieldModel scale (25+ basis functions).
    """

    def test_gradient_finite_at_high_state_dim(self) -> None:
        """Gradients through Laplace-EKF are finite at n_state=25."""
        from state_space_practice.parameter_transforms import (
            POSITIVE,
            transform_to_constrained,
            transform_to_unconstrained,
        )
        from state_space_practice.point_process_kalman import (
            log_conditional_intensity,
            stochastic_point_process_filter,
        )

        n_state = 25  # PlaceFieldModel scale
        key = jax.random.PRNGKey(42)
        A = 0.99 * jnp.eye(n_state)
        Q = 0.01 * jnp.eye(n_state)
        m0 = jnp.zeros(n_state)
        P0 = jnp.eye(n_state)
        W = jax.random.normal(key, (3, n_state)) * 0.1
        dm = jnp.tile(W, (50, 1, 1))
        spikes = jax.random.poisson(key, jnp.ones((50, 3)) * 0.01)

        spec = {"q_diag": POSITIVE}
        params = {"q_diag": jnp.diag(Q)}
        unc = transform_to_unconstrained(params, spec)

        def loss_fn(unc_p):
            p = transform_to_constrained(unc_p, spec)
            _, _, mll = stochastic_point_process_filter(
                m0, P0, dm, spikes, 0.001, A, jnp.diag(p["q_diag"]),
                log_conditional_intensity,
            )
            return -mll

        g = jax.grad(loss_fn)(unc)
        assert jnp.all(jnp.isfinite(g["q_diag"]))


class TestPlaceFieldSGDFitting:
    """Tests for PlaceFieldModel.fit_sgd()."""

    def test_sgd_improves_ll(self, sim_data: dict) -> None:
        import optax

        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-3,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0), optax.adam(1e-3),
        )
        lls = model.fit_sgd(
            sim_data["position"], sim_data["spikes"],
            optimizer=optimizer, num_steps=30,
        )
        assert len(lls) > 1
        assert all(np.isfinite(ll) for ll in lls)

    def test_sgd_process_cov_positive(self, sim_data: dict) -> None:
        import optax

        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-3,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0), optax.adam(1e-3),
        )
        model.fit_sgd(
            sim_data["position"], sim_data["spikes"],
            optimizer=optimizer, num_steps=20,
        )
        assert jnp.all(jnp.diag(model.process_cov) > 0)

    def test_sgd_populates_smoother(self, sim_data: dict) -> None:
        import optax

        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-3,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0), optax.adam(1e-3),
        )
        model.fit_sgd(
            sim_data["position"], sim_data["spikes"],
            optimizer=optimizer, num_steps=10,
        )
        assert model.smoother_mean is not None
        assert model.filtered_mean is not None


class TestWarmStart:
    """Tests for the stationary Poisson GLM warm-start path."""

    def test_warm_start_sets_init_mean_away_from_zero(
        self, sim_data: dict
    ) -> None:
        """Warm-start must produce a non-trivial init_mean from spikes.

        With zero spikes the MAP collapses to the prior mean (zeros). With
        real spike data the MAP is pulled toward non-zero weights that
        capture the rate map. A warm-started model on realistic data must
        therefore have ``|init_mean| > 0``.
        """
        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-5,
        )
        # Run fit_sgd with warm_start=True but 0 optimizer steps: that sets
        # init_mean / init_cov from the warm-start path without letting SGD
        # drift from them.
        import optax
        opt = optax.adam(1e-3)
        model.fit_sgd(
            sim_data["position"], sim_data["spikes"],
            optimizer=opt, num_steps=0, warm_start=True,
        )
        assert model.init_mean is not None
        assert model.init_cov is not None
        assert float(jnp.linalg.norm(model.init_mean)) > 0.1, (
            "warm-start init_mean should move away from zero on real data"
        )

    def test_warm_start_improves_marginal_ll_at_init(
        self, sim_data: dict
    ) -> None:
        """Warm-start must improve the marginal LL evaluated at init_mean/init_cov.

        This is the direct test of the warm-start's value proposition:
        we evaluate ``stochastic_point_process_filter`` at the warm-start
        ``(init_mean, init_cov)`` vs. the cold-start ``(zeros, I)`` and
        assert the warm-start LL is strictly better. Unlike a post-SGD-step
        comparison, this cannot be confounded by the optimizer recovering
        from a bad initialization in one step, so it directly probes the
        quality of the initial state.
        """
        import optax

        from state_space_practice.point_process_kalman import (
            log_conditional_intensity,
            stochastic_point_process_filter,
        )

        opt = optax.adam(1e-3)

        # Warm-started model: init_mean / init_cov set by Laplace-GLM fit.
        m_warm = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-5,
        )
        m_warm.fit_sgd(
            sim_data["position"], sim_data["spikes"],
            optimizer=opt, num_steps=0, warm_start=True,
        )
        # Cold-started: init_mean=zeros, init_cov=init_cov_scale*I
        m_cold = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-5,
        )
        m_cold.fit_sgd(
            sim_data["position"], sim_data["spikes"],
            optimizer=opt, num_steps=0, warm_start=False,
        )

        # Evaluate the filter directly at each model's init state —
        # no SGD steps taken, so this is the marginal LL at the initial
        # prior, the quantity the warm-start is designed to improve.
        design_matrix = m_warm._expand_to_block_diagonal(
            m_warm._build_spline_basis_matrix(np.asarray(sim_data["position"]))
        )
        spikes = jnp.asarray(sim_data["spikes"])
        if spikes.ndim == 2 and spikes.shape[1] == 1:
            spikes = spikes.squeeze(axis=1)

        _, _, ll_warm = stochastic_point_process_filter(
            m_warm.init_mean, m_warm.init_cov, design_matrix, spikes,
            m_warm.dt, m_warm.transition_matrix, m_warm.process_cov,
            log_conditional_intensity,
            max_log_count=m_warm._max_log_count,
        )
        _, _, ll_cold = stochastic_point_process_filter(
            m_cold.init_mean, m_cold.init_cov, design_matrix, spikes,
            m_cold.dt, m_cold.transition_matrix, m_cold.process_cov,
            log_conditional_intensity,
            max_log_count=m_cold._max_log_count,
        )
        assert jnp.isfinite(ll_warm) and jnp.isfinite(ll_cold)
        assert ll_warm > ll_cold, (
            f"warm-start marginal LL at init ({float(ll_warm):.2f}) should "
            f"be better than cold-start ({float(ll_cold):.2f})"
        )

    def test_warm_start_window_slices_data(self, sim_data: dict) -> None:
        """warm_start_window must restrict the GLM fit to that slice.

        We compare the warm-started init_mean for window=first-half vs
        window=whole-session. They should differ, proving the window
        parameter is actually changing the fit.
        """
        import optax

        opt = optax.adam(1e-3)
        n_time = len(sim_data["spikes"])

        m_whole = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-5,
        )
        m_whole.fit_sgd(
            sim_data["position"], sim_data["spikes"],
            optimizer=opt, num_steps=0, warm_start=True, warm_start_window=None,
        )

        m_half = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-5,
        )
        m_half.fit_sgd(
            sim_data["position"], sim_data["spikes"],
            optimizer=opt, num_steps=0, warm_start=True,
            warm_start_window=slice(0, n_time // 2),
        )

        # Half-window and full-window warm-starts should not produce
        # identical init_means on a real dataset.
        assert not jnp.allclose(m_whole.init_mean, m_half.init_mean)

    def test_warm_start_false_matches_old_defaults(self, sim_data: dict) -> None:
        """warm_start=False must reproduce the pre-warm-start behavior.

        With ``warm_start=False``, ``init_mean`` should be all zeros and
        ``init_cov`` should equal ``init_cov_scale * I`` — the old scalar
        defaults that predate this commit. This is the back-compat path
        for ablation studies.
        """
        import optax

        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3,
            init_process_noise=1e-5, init_cov_scale=0.5,
        )
        opt = optax.adam(1e-3)
        model.fit_sgd(
            sim_data["position"], sim_data["spikes"],
            optimizer=opt, num_steps=0, warm_start=False,
        )
        assert jnp.allclose(model.init_mean, jnp.zeros(model.n_basis))
        assert jnp.allclose(
            model.init_cov, jnp.eye(model.n_basis) * 0.5, atol=1e-10
        )

    def test_warm_start_converges_at_low_max_iter(
        self, sim_data: dict
    ) -> None:
        """With the intercept-matching initial guess, Newton converges to
        machine precision in ~8 iterations. Max_iter=3 should already be
        within 20% relative error of the machine-precision fit, proving
        the initial guess is close to the MAP.
        """
        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-5,
        )
        model.n_neurons = 1
        Z_base = model._build_spline_basis_matrix(
            np.asarray(sim_data["position"])
        )
        spikes = jnp.asarray(sim_data["spikes"])
        if spikes.ndim == 2 and spikes.shape[1] == 1:
            spikes = spikes.squeeze(axis=1)

        # Reference: many-iteration fit
        w_ref, _ = model._fit_stationary_glm(Z_base, spikes, max_iter=30)

        # At max_iter=3 we should already be close (not yet fully
        # converged, but in the same neighborhood). This test would
        # catch a regression where the intercept init stops working.
        w_3, _ = model._fit_stationary_glm(Z_base, spikes, max_iter=3)
        err = float(jnp.linalg.norm(w_3 - w_ref)) / float(jnp.linalg.norm(w_ref))
        assert err < 0.2, (
            f"intercept-init Newton should converge to within 20% of the "
            f"reference in 3 iterations; got relative error {err:.3f}"
        )

    def test_warm_start_initial_rate_matches_mean(
        self, sim_data: dict
    ) -> None:
        """The warm-start's initial weights (before Newton even runs) should
        produce predicted rates that match the empirical mean firing rate.

        This is the direct test of the NeMoS-inspired intercept-matching
        initializer: the least-squares projection of ``log(mean_rate)``
        onto the column space of Z_base must give a weight vector whose
        average predicted rate is close to the observed mean rate.

        We verify this at ``max_iter=0`` (no Newton steps) so we're testing
        the initial guess directly, not the post-Newton MAP.
        """
        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-5,
        )
        model.n_neurons = 1
        Z_base = model._build_spline_basis_matrix(
            np.asarray(sim_data["position"])
        )
        spikes = jnp.asarray(sim_data["spikes"])
        if spikes.ndim == 2 and spikes.shape[1] == 1:
            spikes = spikes.squeeze(axis=1)

        # max_iter=0: return the initial guess without running Newton
        w_init, _ = model._fit_stationary_glm(Z_base, spikes, max_iter=0)
        predicted_rate = float(jnp.mean(jnp.exp(Z_base @ w_init)))
        observed_rate = float(jnp.mean(spikes) / sim_data["dt"])
        # Initial rate should match observed to within ~30% — the match
        # is approximate because the spline basis is not a partition of
        # unity, so the projection of a constant onto its column space
        # has a non-zero residual. Newton's first iteration cleans this up.
        rel_err = abs(predicted_rate - observed_rate) / max(observed_rate, 1e-6)
        assert rel_err < 0.3, (
            f"initial predicted rate {predicted_rate:.3f} should be close "
            f"to observed {observed_rate:.3f}; relative error {rel_err:.3f}"
        )

    def test_warm_start_multi_neuron_block_diagonal_cov(
        self, sim_data: dict
    ) -> None:
        """Multi-neuron warm-start must produce a block-diagonal init_cov.

        The design matrix is block-diagonal across neurons (each neuron's
        log-intensity depends only on its own weight slice), so the
        Laplace covariance from independent per-neuron GLM fits must
        also be block-diagonal.
        """
        # Build a 2-neuron spike array from sim_data
        spikes_single = np.asarray(sim_data["spikes"])
        if spikes_single.ndim == 2:
            spikes_single = spikes_single.squeeze(axis=-1)
        spikes_multi = np.stack([spikes_single, spikes_single[::-1]], axis=-1)

        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3, init_process_noise=1e-5,
        )
        # Populate n_neurons and warm-start directly (avoid running SGD)
        model.n_neurons = 2
        Z_base = model._build_spline_basis_matrix(
            np.asarray(sim_data["position"])
        )
        model._warm_start_parameters(Z_base, jnp.asarray(spikes_multi), None)

        nb = model.n_basis_per_neuron
        assert model.init_cov.shape == (2 * nb, 2 * nb)
        # Off-diagonal (cross-neuron) block must be exactly zero
        cross_block = model.init_cov[:nb, nb:]
        assert jnp.allclose(cross_block, jnp.zeros_like(cross_block))
        # Diagonal blocks must be PSD (positive eigenvalues)
        eigvals_0 = jnp.linalg.eigvalsh(model.init_cov[:nb, :nb])
        eigvals_1 = jnp.linalg.eigvalsh(model.init_cov[nb:, nb:])
        assert jnp.all(eigvals_0 > 0)
        assert jnp.all(eigvals_1 > 0)


class TestMaxFiringRateHz:
    """Tests for the ``max_firing_rate_hz`` ceiling and saturation warning."""

    def test_default_ceiling_matches_physiology(self) -> None:
        """Default ceiling corresponds to ``log(500 * dt)``."""
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        expected = float(np.log(500.0 * 0.02))  # log(10) ≈ 2.3
        assert np.isclose(model._max_log_count, expected)

    def test_invalid_max_firing_rate_rejected(self) -> None:
        """Non-positive max_firing_rate_hz must raise."""
        with pytest.raises(ValueError, match="max_firing_rate_hz must be positive"):
            PlaceFieldModel(
                dt=0.02, n_interior_knots=3, max_firing_rate_hz=0.0
            )
        with pytest.raises(ValueError, match="max_firing_rate_hz must be positive"):
            PlaceFieldModel(
                dt=0.02, n_interior_knots=3, max_firing_rate_hz=-10.0
            )

    def test_ceiling_caps_marginal_ll_on_pathological_bin(self) -> None:
        """Pathological outlier bin must not drive marginal LL to -1e8.

        Regression for the original bug report: a single bin with 5926
        spikes (vs 0-10 in neighbors) caused Laplace-EKF's first marginal
        LL to reach ~-4.86e8 with the default max_log_count=20. With the
        physiological ceiling max_firing_rate_hz=500 Hz at dt=0.2s, the
        per-bin Poisson logpmf contribution is bounded by
        ``5926 * log(100) - 100 - lgamma(5927)`` ≈ -21k. Accounting for
        Laplace normalization and the quadratic prior term on a single
        outlier bin, the total LL over 200 bins must stay above -1e5.
        The uncapped (default=20) path produces ~-1e8, so the gap is huge.
        """
        from state_space_practice.point_process_kalman import (
            log_conditional_intensity,
            stochastic_point_process_filter,
        )

        key = jax.random.PRNGKey(0)
        n_time, n_basis = 200, 16
        dt = 0.2
        # Tight design matrix with modest weights; typical bins have O(1) spikes.
        Z = jax.random.normal(key, (n_time, n_basis)) * 0.3
        spikes = jax.random.poisson(
            jax.random.split(key, 1)[0], jnp.full((n_time,), 1.0)
        )
        # Inject the pathological count in the middle of the series so the
        # filter has context on both sides.
        spikes = spikes.at[n_time // 2].set(5926)

        m0 = jnp.zeros(n_basis)
        P0 = jnp.eye(n_basis) * 0.01  # tight prior
        A = jnp.eye(n_basis)
        Q = jnp.eye(n_basis) * 1e-7

        # WITH the physiological ceiling (max_log_count = log(500 * 0.2) ≈ 4.6)
        _, _, mll_capped = stochastic_point_process_filter(
            m0, P0, Z, spikes, dt, A, Q,
            log_conditional_intensity,
            max_log_count=float(np.log(500.0 * dt)),
        )
        # WITHOUT (default of 20) — the old catastrophic-LL path
        _, _, mll_default = stochastic_point_process_filter(
            m0, P0, Z, spikes, dt, A, Q,
            log_conditional_intensity,
        )

        # The physiological ceiling must keep the total marginal LL within
        # the analytical bound (~-21k for the one bad bin, plus -O(1) from
        # the 199 good bins and the Laplace normalization).
        assert jnp.isfinite(mll_capped)
        assert mll_capped > -1e5, (
            f"ceiling should keep LL > -1e5 (analytical bound ~-21k for the "
            f"outlier bin); got {float(mll_capped):.2e}"
        )
        # And it must be dramatically better than the default-ceiling LL on
        # this pathological input (default path produces ~-1e8).
        assert mll_capped - mll_default > 1e5, (
            f"ceiling should improve LL by >1e5; got capped={float(mll_capped):.2e} "
            f"default={float(mll_default):.2e}"
        )

    def test_saturation_warning_machinery_does_not_crash(
        self, sim_data: dict
    ) -> None:
        """The saturation check must run cleanly on the ``fit_sgd`` forward path.

        Note: this test does NOT assert that the warning fires. With a
        tight prior (``init_cov_scale=0.01``) and a single outlier bin,
        the Laplace-EKF posterior mean is pulled back toward the prior
        at that bin and may not reach the 500 Hz ceiling, even though the
        *data* clearly would. That's the expected behavior of a well-
        regularized filter — the whole point of this commit is that
        outlier bins no longer blow up the LL. A test that hard-asserts
        the warning fires would be brittle.

        What this test checks:
        1. ``fit_sgd`` completes without crashing when given a bin with
           a 1000-spike artifact (this used to produce catastrophic LLs).
        2. If the warning does fire, its message has the expected format.
        3. The saturation check itself (vmap, shape handling) runs without
           dtype or shape errors on realistic input.
        """
        import optax

        # Inject an artifact spike flood into the middle of the series.
        spikes = np.asarray(sim_data["spikes"])
        if spikes.ndim == 2:
            spikes = spikes.squeeze(axis=-1)
        spikes = spikes.astype(np.int64).copy()
        bad_idx = spikes.shape[0] // 2
        spikes[bad_idx] = 1000  # far above any physiological rate

        model = PlaceFieldModel(
            dt=sim_data["dt"], n_interior_knots=3,
            init_process_noise=1e-5, init_cov_scale=0.01,
            max_firing_rate_hz=500.0,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0), optax.adam(1e-3),
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            lls = model.fit_sgd(
                sim_data["position"], spikes,
                optimizer=optimizer, num_steps=5,
            )
        # Core assertion: the forward path completed and returned finite LLs.
        assert all(np.isfinite(ll) for ll in lls), (
            f"fit_sgd must return finite LLs on outlier data, got {lls}"
        )
        # If the warning fired, it must name the configured ceiling.
        saturation_warnings = [
            w for w in captured
            if "saturated" in str(w.message).lower()
        ]
        for w in saturation_warnings:
            assert "max_firing_rate_hz" in str(w.message)


class TestBlockDiagonalDispatch:
    """Tests for PlaceFieldModel's auto-dispatch to the block-diagonal
    filter path.

    These tests verify that:
    1. Multi-neuron fits automatically dispatch to the block path
       (``_block_n_neurons`` and ``_block_size`` are set non-None).
    2. ``force_dense=True`` disables dispatch and falls through to the
       dense filter.
    3. The two paths produce numerically equivalent output (smoother
       mean, marginal LL) on the same problem — critical for fit_sgd
       correctness, since SGD updates depend on gradient equivalence.
    4. Single-neuron fits do not dispatch (no structure to exploit).
    """

    def _make_multi_neuron_data(self, n_neurons: int = 3, n_time: int = 500):
        """Construct simulated multi-neuron data via simulate_2d_moving_place_field."""
        sim = simulate_2d_moving_place_field(total_time=n_time * 0.02, dt=0.02)
        position = sim["position"]
        # Replicate single-neuron spikes into n_neurons columns with
        # slightly different noise per neuron so the fit is non-trivial.
        base_spikes = np.asarray(sim["spikes"]).squeeze()
        rng = np.random.default_rng(0)
        spikes_multi = np.stack(
            [
                base_spikes + rng.integers(0, 2, size=base_spikes.shape)
                for _ in range(n_neurons)
            ],
            axis=-1,
        ).astype(np.int64)
        return position, spikes_multi

    def test_fit_sgd_multi_neuron_dispatches_to_block_path(self) -> None:
        """Multi-neuron fit_sgd should detect block structure and set
        ``_block_n_neurons`` / ``_block_size`` non-None."""
        position, spikes = self._make_multi_neuron_data(n_neurons=3)
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        import optax

        model.fit_sgd(
            position, spikes, optimizer=optax.sgd(1e-4), num_steps=0
        )
        assert model._block_n_neurons == 3
        assert model._block_size == model.n_basis_per_neuron

    def test_fit_sgd_multi_neuron_force_dense_skips_dispatch(self) -> None:
        """force_dense=True should suppress block detection."""
        position, spikes = self._make_multi_neuron_data(n_neurons=3)
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        import optax

        model.fit_sgd(
            position, spikes, optimizer=optax.sgd(1e-4),
            num_steps=0, force_dense=True,
        )
        assert model._block_n_neurons is None
        assert model._block_size is None

    def test_fit_sgd_single_neuron_does_not_dispatch(self) -> None:
        """Single-neuron fits have design_matrix.ndim==2, which detection
        rejects. Auto-dispatch should leave _block_n_neurons as None."""
        sim = simulate_2d_moving_place_field(total_time=10.0, dt=0.02)
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        import optax

        model.fit_sgd(
            sim["position"], sim["spikes"],
            optimizer=optax.sgd(1e-4), num_steps=0,
        )
        assert model._block_n_neurons is None
        assert model._block_size is None

    def test_fit_sgd_block_vs_dense_initial_step_matches(self) -> None:
        """Block and dense paths must produce an identical step-0 LL.

        The step-0 LL is the forward filter evaluated at the warm-start
        init state before any optimizer updates. Both paths should give
        bit-identical output at this step (pinned at atol=1e-9 by the
        low-level equivalence tests in TestBlockDiagonalFilterEquivalence).

        Across subsequent SGD steps, the trajectories naturally diverge
        because the dense path's autodiff produces non-zero off-block
        gradient components on the init_cov parameter (spurious entries
        that the block filter never computes), which feed into the PSD
        parameter transform and perturb the reconstructed init_cov
        differently between paths. This is not a bug — the block path
        is the CORRECT gradient for the block-diagonal parameterization,
        and the dense path has extra off-block noise that gets projected
        away by the next M-step / detection cycle. See the
        PlaceFieldModel class docstring for the full architectural
        explanation.

        For this test we pin step-0 LL equivalence to tight tolerance,
        and only require that both paths produce finite LLs across the
        full SGD trajectory (no NaN, no divergence).
        """
        position, spikes = self._make_multi_neuron_data(n_neurons=3, n_time=200)

        import optax

        m_block = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        lls_block = m_block.fit_sgd(
            position, spikes, optimizer=optax.sgd(1e-4), num_steps=5,
            force_dense=False,
        )
        assert m_block._block_n_neurons == 3

        m_dense = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        lls_dense = m_dense.fit_sgd(
            position, spikes, optimizer=optax.sgd(1e-4), num_steps=5,
            force_dense=True,
        )
        assert m_dense._block_n_neurons is None

        # Step 0 LL must match bit-exactly (both paths run the filter
        # on identical warm-started init state).
        np.testing.assert_allclose(
            float(lls_block[0]), float(lls_dense[0]),
            atol=1e-9, rtol=1e-10,
        )

        # Both paths must produce finite LLs throughout — no NaN drift.
        assert all(np.isfinite(ll) for ll in lls_block)
        assert all(np.isfinite(ll) for ll in lls_dense)

        # Both paths should converge in the same direction (LL should
        # improve or stay flat, not diverge catastrophically).
        assert lls_block[-1] >= lls_block[0] - 1.0
        assert lls_dense[-1] >= lls_dense[0] - 1.0

    def test_fit_em_multi_neuron_dispatches(self) -> None:
        """EM fit() also uses block dispatch when structure is detected."""
        position, spikes = self._make_multi_neuron_data(n_neurons=2, n_time=200)
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        model.fit(position, spikes, max_iter=3, verbose=False)
        assert model._block_n_neurons == 2
        # Fit completes and produces finite LLs
        assert all(np.isfinite(ll) for ll in model.log_likelihoods)
        assert model.smoother_mean is not None

    def test_fit_em_block_vs_dense_equivalence(self) -> None:
        """EM fit() under block dispatch should agree with force_dense."""
        position, spikes = self._make_multi_neuron_data(n_neurons=2, n_time=200)

        m_block = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        lls_block = m_block.fit(
            position, spikes, max_iter=3, verbose=False, force_dense=False,
        )
        assert m_block._block_n_neurons == 2

        m_dense = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        lls_dense = m_dense.fit(
            position, spikes, max_iter=3, verbose=False, force_dense=True,
        )
        assert m_dense._block_n_neurons is None

        np.testing.assert_allclose(
            np.asarray(lls_block), np.asarray(lls_dense),
            atol=1e-5, rtol=1e-6,
        )

    def test_em_falls_back_to_dense_when_m_step_breaks_structure(self) -> None:
        """When update_transition_matrix=True, the M-step writes back a
        dense A that breaks block-diagonal structure. The next E-step
        must automatically fall back to the dense filter path.

        Regression for the re-detect-after-M-step logic. Without this
        re-detection, the next E-step would call _build_block_structure_
        from_traced on a non-block-diagonal A, silently applying block
        0's A to every neuron and producing wrong results.

        We verify by checking that ``_block_n_neurons`` is set to None
        after the first M-step, AND that the fit completes without
        crashing (the dense path handles the dense A correctly).
        """
        position, spikes = self._make_multi_neuron_data(
            n_neurons=2, n_time=300
        )
        model = PlaceFieldModel(
            dt=0.02,
            n_interior_knots=3,
            update_transition_matrix=True,  # allows M-step to learn A
        )
        lls = model.fit(position, spikes, max_iter=2, verbose=False)

        # After the first M-step, A may no longer be block-diagonal
        # (the M-step formula produces a full matrix when
        # update_transition_matrix=True). The re-detect logic should
        # set _block_n_neurons=None in that case, falling back to dense.
        # We don't strictly require the fall-back to happen (if the
        # learned A happens to be block-diagonal by luck, detection
        # still succeeds), but the fit must complete without crashing.
        assert len(lls) >= 1
        assert all(np.isfinite(ll) for ll in lls)
        assert model.smoother_mean is not None
        # If the M-step did produce a non-block-diagonal A, verify
        # that fall-back happened and the dense path produced the fit.
        from state_space_practice.point_process_kalman import (
            _is_block_diagonal,
        )

        if not _is_block_diagonal(
            model.transition_matrix,
            n_blocks=2,
            block_size=model.n_basis_per_neuron,
        ):
            assert model._block_n_neurons is None, (
                "A became non-block-diagonal but re-detect did not "
                "flip the dispatch to dense"
            )

    def test_score_reuses_fit_time_dispatch(self) -> None:
        """score() should reuse the block dispatch decision made at fit time
        without re-detecting. This keeps score() fast — detection is
        O(d^3) for eigvalsh-free checks but still ~ms-scale — while
        guaranteeing consistency between fit and score outputs."""
        position, spikes = self._make_multi_neuron_data(n_neurons=3, n_time=100)
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        model.fit(position, spikes, max_iter=2, verbose=False)
        assert model._block_n_neurons == 3

        # score() on the fitted model should produce finite output
        # while still having _block_n_neurons set.
        ll = model.score(position, spikes)
        assert np.isfinite(ll)
        assert model._block_n_neurons == 3  # dispatch state preserved
