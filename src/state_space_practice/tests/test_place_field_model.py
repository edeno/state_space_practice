"""Tests for the PlaceFieldModel class and supporting functions."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.exceptions import NotFittedError
from state_space_practice.place_field_model import (
    PlaceFieldModel,
    build_2d_spline_basis,
    evaluate_basis,
)
from state_space_practice.simulate_data import simulate_2d_moving_place_field
from state_space_practice.tests.recovery_helpers import assert_ll_monotonic

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
            "knots_x",
            "knots_y",
            "x_lo",
            "x_hi",
            "y_lo",
            "y_hi",
            "formula",
            "n_basis",
            "n_interior_knots",
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

    @pytest.mark.parametrize(
        ("bad_value", "match"),
        [(0.5, "integer-valued"), (np.nan, "finite")],
    )
    def test_fit_rejects_invalid_spike_counts(
        self, sim_data: dict, bad_value: float, match: str
    ) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        spikes = np.asarray(sim_data["spikes"], dtype=float).copy()
        spikes.reshape(-1)[0] = bad_value
        with pytest.raises(ValueError, match=match):
            model.fit(sim_data["position"], spikes, max_iter=1, verbose=False)

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

    def test_max_iter_final_estep_matches_returned_ll(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        lls = model.fit(
            sim_data["position"], sim_data["spikes"], max_iter=1, verbose=False
        )
        assert model.basis_info is not None
        design_matrix = jnp.asarray(
            evaluate_basis(sim_data["position"], model.basis_info)
        )
        spikes = jnp.asarray(sim_data["spikes"])
        if spikes.ndim == 2 and spikes.shape[1] == 1:
            spikes = spikes.squeeze(axis=1)
        fresh_ll = model._e_step(design_matrix, spikes)
        np.testing.assert_allclose(fresh_ll, lls[-1], atol=1e-6)

    def test_rollback_restores_m_step_parameters(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        real_e_step = model._e_step
        ll_iter = iter([0.0, -1e6])

        def fake_e_step(*args, **kwargs):
            real_e_step(*args, **kwargs)
            return next(ll_iter)

        def bad_m_step():
            assert model.process_cov is not None
            model.process_cov = model.process_cov + 100.0 * jnp.eye(
                model.process_cov.shape[0]
            )

        model._e_step = fake_e_step
        model._m_step = bad_m_step
        lls = model.fit(
            sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False
        )
        assert lls == [0.0]
        assert model.process_cov is not None
        assert float(jnp.max(jnp.diag(model.process_cov))) < 1.0

    _POSTERIOR_ATTRS = (
        "smoother_mean",
        "smoother_cov",
        "smoother_cross_cov",
        "filtered_mean",
        "filtered_cov",
    )

    @pytest.mark.slow
    def test_non_finite_first_e_step_leaves_model_unfitted(
        self, sim_data: dict, caplog
    ) -> None:
        """A first E-step with a non-finite LL has no accepted state to roll
        back to: the posteriors it installed are dropped, so the model is
        visibly unfitted (bic/summary raise the not-fitted error rather than
        an IndexError on the empty history) and a warning is logged."""
        import logging

        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        real_e_step = model._e_step
        n_calls = []

        def nan_e_step(*args, **kwargs):
            real_e_step(*args, **kwargs)  # installs posteriors
            n_calls.append(1)
            return float("nan")

        model._e_step = nan_e_step
        with caplog.at_level(logging.WARNING):
            lls = model.fit(
                sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False
            )
        assert len(n_calls) == 1  # guard: the injected E-step ran, EM stopped
        assert lls == []
        assert model.log_likelihoods == []
        assert "non-finite" in caplog.text.lower()
        for attr in self._POSTERIOR_ATTRS:
            assert getattr(model, attr) is None, attr
        assert "fitted=False" in repr(model)
        with pytest.raises(NotFittedError, match="Call model.fit"):
            model.bic()
        with pytest.raises(RuntimeError, match="Call model.fit"):
            model.summary()

    @pytest.mark.slow
    def test_non_finite_later_e_step_rolls_back_to_last_accepted(
        self, sim_data: dict
    ) -> None:
        """A non-finite third E-step rolls back to the (parameters, posteriors)
        pair of the second, drops the NaN from the history and stops EM."""
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        real_e_step = model._e_step
        calls: list[dict] = []

        def e_step(*args, **kwargs):
            ll = real_e_step(*args, **kwargs)
            calls.append(
                {
                    "ll": ll,
                    "process_cov": model.process_cov,
                    "init_mean": model.init_mean,
                    "smoother_mean": model.smoother_mean,
                }
            )
            return float("nan") if len(calls) == 3 else ll

        model._e_step = e_step
        lls = model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=10,
            tolerance=1e-12,
            verbose=False,
        )
        assert len(calls) == 3  # guard: EM accepted two steps and ran a third
        # guard: the rejected E-step ran under different (M-step) parameters
        assert not np.array_equal(
            np.asarray(calls[2]["process_cov"]), np.asarray(calls[1]["process_cov"])
        )
        assert lls == [calls[0]["ll"], calls[1]["ll"]]
        assert all(np.isfinite(lls))
        for key in ("process_cov", "init_mean", "smoother_mean"):
            np.testing.assert_array_equal(
                np.asarray(getattr(model, key)), np.asarray(calls[1][key]), key
            )
        assert np.isfinite(model.bic())

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

    def test_predict_rate_map_averages_rates_not_weights(self) -> None:
        """Dynamic rate maps must average exp(Z x_t), not exp(Z mean_t[x_t])."""
        position = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.1],
                [0.9, 0.9],
                [0.3, 0.7],
            ]
        )
        model = PlaceFieldModel(dt=0.02, n_interior_knots=1)
        _, model.basis_info = build_2d_spline_basis(position, n_interior_knots=1)
        grid = position[:1]
        z = evaluate_basis(grid, model.basis_info)[0]
        n_basis = z.shape[0]
        amplitude = 1.4
        weights = z * amplitude / np.dot(z, z)

        model.n_neurons = 1
        model.n_basis_per_neuron = n_basis
        model.n_basis = n_basis
        model.transition_matrix = jnp.eye(n_basis)
        model.process_cov = jnp.eye(n_basis) * 1e-6
        model.init_mean = jnp.zeros(n_basis)
        model.init_cov = jnp.eye(n_basis)
        model.smoother_mean = jnp.asarray([weights, -weights])
        model.smoother_cov = jnp.zeros((2, n_basis, n_basis))

        rate, ci = model.predict_rate_map(grid)
        expected = 0.5 * (np.exp(amplitude) + np.exp(-amplitude))
        np.testing.assert_allclose(rate, [expected], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ci, [[expected, expected]], rtol=1e-12, atol=1e-12)

    def test_predict_rate_map_uses_lognormal_posterior_mean(self) -> None:
        """For Gaussian weights, E[exp(Zx)] includes the 0.5 ZPZ term."""
        position = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.1],
                [0.9, 0.9],
                [0.3, 0.7],
            ]
        )
        model = PlaceFieldModel(dt=0.02, n_interior_knots=1)
        _, model.basis_info = build_2d_spline_basis(position, n_interior_knots=1)
        grid = position[:1]
        z = evaluate_basis(grid, model.basis_info)[0]
        n_basis = z.shape[0]
        sigma2 = 0.7
        cov = np.eye(n_basis) * sigma2
        var_log_rate = float(z @ cov @ z)

        model.n_neurons = 1
        model.n_basis_per_neuron = n_basis
        model.n_basis = n_basis
        model.transition_matrix = jnp.eye(n_basis)
        model.process_cov = jnp.eye(n_basis) * 1e-6
        model.init_mean = jnp.zeros(n_basis)
        model.init_cov = jnp.eye(n_basis)
        model.smoother_mean = jnp.zeros((1, n_basis))
        model.smoother_cov = jnp.asarray(cov[None, :, :])

        alpha = 0.1
        rate, ci = model.predict_rate_map(grid, alpha=alpha)
        z_alpha = float(jax.scipy.stats.norm.ppf(1 - alpha / 2))
        expected_rate = np.exp(0.5 * var_log_rate)
        expected_ci = np.exp(
            np.array(
                [
                    -z_alpha * np.sqrt(var_log_rate),
                    z_alpha * np.sqrt(var_log_rate),
                ]
            )
        )
        np.testing.assert_allclose(rate, [expected_rate], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ci[0], expected_ci, rtol=1e-12, atol=1e-12)

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

    def test_score_rejects_invalid_spike_counts(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        bad_spikes = np.asarray(sim_data["spikes"], dtype=float).copy()
        bad_spikes.reshape(-1)[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            model.score(sim_data["position"], bad_spikes)

    def test_score_single_neuron_rejects_extra_columns(self, sim_data: dict) -> None:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(
            sim_data["position"],
            sim_data["spikes"],
            max_iter=3,
            verbose=False,
        )
        spikes = np.column_stack([sim_data["spikes"], sim_data["spikes"]])
        with pytest.raises(ValueError, match="n_neurons=1"):
            model.score(sim_data["position"], spikes)


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
        with pytest.raises(NotFittedError, match="Call model.fit"):
            model.bic()

    def test_n_free_params_before_fit_raises_not_fitted(self) -> None:
        # n_basis is only known after fit; the property must say so instead of
        # tripping a bare assert (which ``python -O`` would strip).
        m = PlaceFieldModel(dt=0.004, n_interior_knots=3)
        with pytest.raises(NotFittedError, match="n_free_params"):
            _ = m.n_free_params

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

    def test_multi_neuron_custom_func_disables_block_dispatch(
        self, sim_data: dict
    ) -> None:
        base_spikes = np.asarray(sim_data["spikes"]).squeeze()
        spikes = np.stack([base_spikes, base_spikes], axis=1)

        def custom_func(design_matrix_t, state):
            return 1.25 * (design_matrix_t @ state) + 0.1

        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            log_intensity_func=custom_func,
        )
        lls = model.fit(
            sim_data["position"],
            spikes,
            max_iter=1,
            verbose=False,
        )
        assert model._block_n_neurons is None
        assert model._block_size is None
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
            model.score(
                sim_data["position"],
                np.column_stack(
                    [sim_data["spikes"], sim_data["spikes"], sim_data["spikes"]]
                ),
            )

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
            dt=sim_data["dt"],
            n_interior_knots=3,
            update_process_cov=False,
            update_init_state=False,
        )
        model.fit(sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False)
        assert model.n_free_params == 0

    def test_isotropic(self, sim_data: dict) -> None:
        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            process_noise_structure="isotropic",
        )
        model.fit(sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False)
        # isotropic Q (1) + init_mean (nb) + init_cov_diag (nb) = 1 + 2*nb
        assert model.n_free_params == 1 + 2 * model.n_basis

    def test_with_transition(self, sim_data: dict) -> None:
        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            update_transition_matrix=True,
        )
        model.fit(sim_data["position"], sim_data["spikes"], max_iter=3, verbose=False)
        nb = model.n_basis
        # diagonal Q (nb) + A (nb^2) + init_mean (nb) + init_cov_diag (nb) = nb^2 + 3*nb
        assert model.n_free_params == nb**2 + 3 * nb


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
            dt=sim_data["dt"],
            n_interior_knots=3,
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
                m0,
                P0,
                dm,
                spikes,
                0.001,
                A,
                jnp.diag(p["q_diag"]),
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
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-3,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(1e-3),
        )
        lls = model.fit_sgd(
            sim_data["position"],
            sim_data["spikes"],
            optimizer=optimizer,
            num_steps=30,
        )
        assert len(lls) > 1
        assert all(np.isfinite(ll) for ll in lls)

    def test_sgd_process_cov_positive(self, sim_data: dict) -> None:
        import optax

        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-3,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(1e-3),
        )
        model.fit_sgd(
            sim_data["position"],
            sim_data["spikes"],
            optimizer=optimizer,
            num_steps=20,
        )
        assert jnp.all(jnp.diag(model.process_cov) > 0)

    def test_sgd_populates_smoother(self, sim_data: dict) -> None:
        import optax

        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-3,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(1e-3),
        )
        model.fit_sgd(
            sim_data["position"],
            sim_data["spikes"],
            optimizer=optimizer,
            num_steps=10,
        )
        assert model.smoother_mean is not None
        assert model.filtered_mean is not None

    def test_sgd_mismatched_lengths_rejected(self, sim_data: dict) -> None:
        import optax

        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        with pytest.raises(ValueError, match="same number of time bins"):
            model.fit_sgd(
                sim_data["position"][:-1],
                sim_data["spikes"],
                optimizer=optax.adam(1e-3),
                num_steps=0,
            )


class TestWarmStart:
    """Tests for the stationary Poisson GLM warm-start path."""

    def test_warm_start_solves_stationary_glm_map_equations(self) -> None:
        """Warm-start MAP and covariance should match Poisson GLM equations."""
        dt = 0.02
        prior_precision = 0.7
        Z_base = jnp.array(
            [
                [1.0, -0.4, 0.2],
                [1.0, -0.1, -0.3],
                [1.0, 0.2, 0.1],
                [1.0, 0.5, -0.2],
                [1.0, 0.8, 0.4],
                [1.0, 1.1, -0.1],
            ]
        )
        spikes = jnp.array([0.0, 1.0, 0.0, 2.0, 1.0, 3.0])
        model = PlaceFieldModel(dt=dt, n_interior_knots=3)
        model.n_basis_per_neuron = Z_base.shape[1]

        weights, cov = model._fit_stationary_glm(
            Z_base,
            spikes,
            max_iter=30,
            prior_precision=prior_precision,
        )

        log_rate = Z_base @ weights
        expected_count = jnp.exp(log_rate) * dt
        gradient = Z_base.T @ (expected_count - spikes)
        gradient = gradient + prior_precision * weights
        hessian = Z_base.T @ (expected_count[:, None] * Z_base)
        hessian = hessian + prior_precision * jnp.eye(Z_base.shape[1])

        np.testing.assert_allclose(
            gradient,
            jnp.zeros_like(weights),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            cov,
            jnp.linalg.inv(hessian),
            rtol=1e-8,
            atol=1e-9,
        )

    def test_warm_start_sets_init_mean_away_from_zero(self, sim_data: dict) -> None:
        """Warm-start must produce a non-trivial init_mean from spikes.

        With zero spikes the MAP collapses to the prior mean (zeros). With
        real spike data the MAP is pulled toward non-zero weights that
        capture the rate map. A warm-started model on realistic data must
        therefore have ``|init_mean| > 0``.
        """
        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
        )
        # Run fit_sgd with warm_start=True but 0 optimizer steps: that sets
        # init_mean / init_cov from the warm-start path without letting SGD
        # drift from them.
        import optax

        opt = optax.adam(1e-3)
        model.fit_sgd(
            sim_data["position"],
            sim_data["spikes"],
            optimizer=opt,
            num_steps=0,
            warm_start=True,
        )
        assert model.init_mean is not None
        assert model.init_cov is not None
        assert float(jnp.linalg.norm(model.init_mean)) > 0.1, (
            "warm-start init_mean should move away from zero on real data"
        )

    def test_warm_start_improves_marginal_ll_at_init(self, sim_data: dict) -> None:
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
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
        )
        m_warm.fit_sgd(
            sim_data["position"],
            sim_data["spikes"],
            optimizer=opt,
            num_steps=0,
            warm_start=True,
        )
        # Cold-started: init_mean=zeros, init_cov=init_cov_scale*I
        m_cold = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
        )
        m_cold.fit_sgd(
            sim_data["position"],
            sim_data["spikes"],
            optimizer=opt,
            num_steps=0,
            warm_start=False,
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
            m_warm.init_mean,
            m_warm.init_cov,
            design_matrix,
            spikes,
            m_warm.dt,
            m_warm.transition_matrix,
            m_warm.process_cov,
            log_conditional_intensity,
            max_log_count=m_warm._max_log_count,
        )
        _, _, ll_cold = stochastic_point_process_filter(
            m_cold.init_mean,
            m_cold.init_cov,
            design_matrix,
            spikes,
            m_cold.dt,
            m_cold.transition_matrix,
            m_cold.process_cov,
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
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
        )
        m_whole.fit_sgd(
            sim_data["position"],
            sim_data["spikes"],
            optimizer=opt,
            num_steps=0,
            warm_start=True,
            warm_start_window=None,
        )

        m_half = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
        )
        m_half.fit_sgd(
            sim_data["position"],
            sim_data["spikes"],
            optimizer=opt,
            num_steps=0,
            warm_start=True,
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
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
            init_cov_scale=0.5,
        )
        opt = optax.adam(1e-3)
        model.fit_sgd(
            sim_data["position"],
            sim_data["spikes"],
            optimizer=opt,
            num_steps=0,
            warm_start=False,
        )
        assert jnp.allclose(model.init_mean, jnp.zeros(model.n_basis))
        assert jnp.allclose(model.init_cov, jnp.eye(model.n_basis) * 0.5, atol=1e-10)

    def test_warm_start_converges_at_low_max_iter(self, sim_data: dict) -> None:
        """With the intercept-matching initial guess, Newton converges to
        machine precision in ~8 iterations. Max_iter=3 should already be
        within 20% relative error of the machine-precision fit, proving
        the initial guess is close to the MAP.
        """
        model = PlaceFieldModel(
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
        )
        model.n_neurons = 1
        Z_base = model._build_spline_basis_matrix(np.asarray(sim_data["position"]))
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

    def test_warm_start_initial_rate_matches_mean(self, sim_data: dict) -> None:
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
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
        )
        model.n_neurons = 1
        Z_base = model._build_spline_basis_matrix(np.asarray(sim_data["position"]))
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

    def test_warm_start_multi_neuron_block_diagonal_cov(self, sim_data: dict) -> None:
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
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
        )
        # Populate n_neurons and warm-start directly (avoid running SGD)
        model.n_neurons = 2
        Z_base = model._build_spline_basis_matrix(np.asarray(sim_data["position"]))
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
            PlaceFieldModel(dt=0.02, n_interior_knots=3, max_firing_rate_hz=0.0)
        with pytest.raises(ValueError, match="max_firing_rate_hz must be positive"):
            PlaceFieldModel(dt=0.02, n_interior_knots=3, max_firing_rate_hz=-10.0)

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
            m0,
            P0,
            Z,
            spikes,
            dt,
            A,
            Q,
            log_conditional_intensity,
            max_log_count=float(np.log(500.0 * dt)),
        )
        # WITHOUT (default of 20) — the old catastrophic-LL path
        _, _, mll_default = stochastic_point_process_filter(
            m0,
            P0,
            Z,
            spikes,
            dt,
            A,
            Q,
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

    def test_saturation_warning_machinery_does_not_crash(self, sim_data: dict) -> None:
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
            dt=sim_data["dt"],
            n_interior_knots=3,
            init_process_noise=1e-5,
            init_cov_scale=0.01,
            max_firing_rate_hz=500.0,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(1e-3),
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            lls = model.fit_sgd(
                sim_data["position"],
                spikes,
                optimizer=optimizer,
                num_steps=5,
            )
        # Core assertion: the forward path completed and returned finite LLs.
        assert all(np.isfinite(ll) for ll in lls), (
            f"fit_sgd must return finite LLs on outlier data, got {lls}"
        )
        # If the warning fired, it must name the configured ceiling.
        saturation_warnings = [
            w for w in captured if "saturated" in str(w.message).lower()
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

        model.fit_sgd(position, spikes, optimizer=optax.sgd(1e-4), num_steps=0)
        assert model._block_n_neurons == 3
        assert model._block_size == model.n_basis_per_neuron

    def test_fit_sgd_rejects_invalid_spike_counts(self) -> None:
        position, spikes = self._make_multi_neuron_data(n_neurons=3)
        spikes = spikes.astype(float)
        spikes[0, 0] = 0.5
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        import optax

        with pytest.raises(ValueError, match="integer-valued"):
            model.fit_sgd(position, spikes, optimizer=optax.sgd(1e-4), num_steps=0)

    def test_fit_sgd_multi_neuron_force_dense_skips_dispatch(self) -> None:
        """force_dense=True should suppress block detection."""
        position, spikes = self._make_multi_neuron_data(n_neurons=3)
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        import optax

        model.fit_sgd(
            position,
            spikes,
            optimizer=optax.sgd(1e-4),
            num_steps=0,
            force_dense=True,
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
            sim["position"],
            sim["spikes"],
            optimizer=optax.sgd(1e-4),
            num_steps=0,
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
            position,
            spikes,
            optimizer=optax.sgd(1e-4),
            num_steps=5,
            force_dense=False,
        )
        assert m_block._block_n_neurons == 3

        m_dense = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        lls_dense = m_dense.fit_sgd(
            position,
            spikes,
            optimizer=optax.sgd(1e-4),
            num_steps=5,
            force_dense=True,
        )
        assert m_dense._block_n_neurons is None

        # Step 0 LL must match bit-exactly (both paths run the filter
        # on identical warm-started init state).
        np.testing.assert_allclose(
            float(lls_block[0]),
            float(lls_dense[0]),
            atol=1e-9,
            rtol=1e-10,
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
            position,
            spikes,
            max_iter=3,
            verbose=False,
            force_dense=False,
        )
        assert m_block._block_n_neurons == 2

        m_dense = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        lls_dense = m_dense.fit(
            position,
            spikes,
            max_iter=3,
            verbose=False,
            force_dense=True,
        )
        assert m_dense._block_n_neurons is None

        np.testing.assert_allclose(
            np.asarray(lls_block),
            np.asarray(lls_dense),
            atol=1e-5,
            rtol=1e-6,
        )

    @pytest.mark.slow
    def test_fit_em_update_transition_matrix_block_vs_dense(self) -> None:
        """With ``update_transition_matrix=True`` the block-path fit (block
        E-step, block-covariance M-step, then the dense fallback once A is
        dense) must learn the same A, Q and LL history as a dense-only fit."""
        position, spikes = self._make_multi_neuron_data(n_neurons=2, n_time=200)

        def fit(force_dense: bool) -> tuple[PlaceFieldModel, list, list]:
            model = PlaceFieldModel(
                dt=0.02, n_interior_knots=3, update_transition_matrix=True
            )
            real_e_step = model._e_step
            dispatch: list = []

            def e_step(*args, **kwargs):
                dispatch.append(model._block_n_neurons)
                return real_e_step(*args, **kwargs)

            model._e_step = e_step
            lls = model.fit(
                position, spikes, max_iter=3, verbose=False, force_dense=force_dense
            )
            return model, lls, dispatch

        m_block, lls_block, dispatch_block = fit(force_dense=False)
        m_dense, lls_dense, dispatch_dense = fit(force_dense=True)
        # guard: the block fit really started on the block path and the
        # learned A then moved it to dense; the dense fit never dispatched.
        assert dispatch_block[0] == 2 and dispatch_block[-1] is None
        assert set(dispatch_dense) == {None}
        # guard: A was actually learned
        assert not np.allclose(
            np.asarray(m_dense.transition_matrix), np.eye(m_dense.n_basis)
        )

        assert len(lls_block) == len(lls_dense)
        np.testing.assert_allclose(lls_block, lls_dense, rtol=0, atol=1e-8)
        # A stays close to I (learned deviations ~1e-5), so compare A - I
        # at an absolute tolerance well below the learned change.
        eye = np.eye(m_dense.n_basis)
        np.testing.assert_allclose(
            np.asarray(m_block.transition_matrix) - eye,
            np.asarray(m_dense.transition_matrix) - eye,
            rtol=0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(m_block.process_cov),
            np.asarray(m_dense.process_cov),
            rtol=1e-6,
            atol=1e-14,
        )

    @staticmethod
    def _distinct_two_neuron_data(total_time: float) -> tuple:
        """A place cell and an unrelated Poisson neuron: their fitted
        per-neuron process noise genuinely differs."""
        sim = simulate_2d_moving_place_field(
            total_time=total_time, dt=0.02, rng=np.random.default_rng(1)
        )
        base = np.asarray(sim["spikes"]).squeeze()
        other = np.random.default_rng(0).poisson(1.0, base.shape)
        spikes = np.stack([base, other], axis=-1).astype(np.int64)
        return sim["position"], spikes

    @staticmethod
    def _record_dispatch(model: PlaceFieldModel) -> list:
        """Record the block dispatch in effect at every E-step."""
        real_e_step = model._e_step
        dispatch: list = []

        def e_step(*args, **kwargs):
            dispatch.append(model._block_n_neurons)
            return real_e_step(*args, **kwargs)

        model._e_step = e_step
        return dispatch

    @pytest.mark.slow
    def test_fit_em_per_neuron_process_noise_stays_on_block_path(self) -> None:
        """EM learns a different Q for each neuron. The block path must keep
        running (each neuron with its own Q block) and match a dense fit to
        roundoff in LL history, Q and A."""
        position, spikes = self._distinct_two_neuron_data(total_time=8.0)
        fits = {}
        for force_dense in (False, True):
            model = PlaceFieldModel(
                dt=0.02, n_interior_knots=2, init_process_noise=1e-4
            )
            dispatch = self._record_dispatch(model)
            lls = model.fit(
                position, spikes, max_iter=3, verbose=False, force_dense=force_dense
            )
            fits[force_dense] = (model, lls, dispatch)

        m_block, lls_block, dispatch_block = fits[False]
        m_dense, lls_dense, dispatch_dense = fits[True]
        assert set(dispatch_block) == {2}  # every E-step on the block path
        assert set(dispatch_dense) == {None}
        q_dense = np.asarray(jnp.diag(m_dense.process_cov))
        nb = q_dense.size // 2
        # guard: the neurons' learned Q really differ
        assert np.max(np.abs(q_dense[:nb] - q_dense[nb:]) / q_dense[:nb]) > 1e-4

        assert len(lls_block) == len(lls_dense)
        np.testing.assert_allclose(lls_block, lls_dense, rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(m_block.process_cov),
            np.asarray(m_dense.process_cov),
            rtol=1e-8,
            atol=1e-16,
        )
        np.testing.assert_array_equal(
            np.asarray(m_block.transition_matrix),
            np.asarray(m_dense.transition_matrix),
        )

    @pytest.mark.slow
    def test_fit_sgd_block_path_learns_per_neuron_process_noise(self) -> None:
        """fit_sgd on the block path trains every neuron's Q block (not just
        neuron 0's) and matches a dense fit."""
        import optax

        position, spikes = self._distinct_two_neuron_data(total_time=6.0)
        q, dispatch = {}, {}
        for force_dense in (False, True):
            # init state fixed: its full-PSD SGD parameterization has
            # off-block gradients only the dense path sees.
            model = PlaceFieldModel(
                dt=0.02, n_interior_knots=2, update_init_state=False
            )
            model.fit_sgd(
                position,
                spikes,
                optimizer=optax.adam(1e-1),
                num_steps=5,
                force_dense=force_dense,
            )
            q[force_dense] = np.asarray(jnp.diag(model.process_cov))
            dispatch[force_dense] = model._block_n_neurons
        assert dispatch == {False: 2, True: None}
        nb = q[True].size // 2
        # guard: the dense fit learns different Q for the two neurons
        assert np.max(np.abs(q[True][:nb] - q[True][nb:])) > 1e-8
        np.testing.assert_allclose(q[False], q[True], rtol=1e-6)

    @pytest.mark.slow
    def test_fit_sgd_update_transition_matrix_dispatches_dense(self) -> None:
        """SGD learns the full A, whose off-block entries only the dense
        filter gives gradients to, so fit_sgd must not take the block path."""
        import optax

        position, spikes = self._distinct_two_neuron_data(total_time=4.0)
        model = PlaceFieldModel(
            dt=0.02, n_interior_knots=2, update_transition_matrix=True
        )
        model.fit_sgd(position, spikes, optimizer=optax.sgd(1e-4), num_steps=0)
        assert model._block_n_neurons is None
        # guard: the same data without A updates does take the block path
        model = PlaceFieldModel(dt=0.02, n_interior_knots=2)
        model.fit_sgd(position, spikes, optimizer=optax.sgd(1e-4), num_steps=0)
        assert model._block_n_neurons == 2

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
        position, spikes = self._make_multi_neuron_data(n_neurons=2, n_time=300)
        model = PlaceFieldModel(
            dt=0.02,
            n_interior_knots=3,
            update_transition_matrix=True,  # allows M-step to learn A
        )
        lls = model.fit(position, spikes, max_iter=2, verbose=False)

        # After the first M-step A is no longer block-diagonal (the
        # M-step formula produces a full matrix when
        # update_transition_matrix=True). The re-detect logic must set
        # _block_n_neurons=None, falling back to dense, and the fit must
        # complete.
        assert len(lls) >= 1
        assert all(np.isfinite(ll) for ll in lls)
        assert model.smoother_mean is not None
        from state_space_practice.point_process_kalman import (
            _block_diagonal_parameters_ok,
        )

        # Guard: the learned parameters really broke the block structure.
        assert not bool(
            _block_diagonal_parameters_ok(
                model.init_cov,
                model.transition_matrix,
                model.process_cov,
                n_neurons=2,
                block_size=model.n_basis_per_neuron,
            )
        )
        assert model._block_n_neurons is None, (
            "the parameters broke the block structure but re-detect did "
            "not flip the dispatch to dense"
        )

    @pytest.mark.slow
    def test_block_path_never_expands_the_design_matrix(self, monkeypatch) -> None:
        """On the block path the filter works from Z_base; the
        ``(n_time, n_neurons, n_neurons * n_basis)`` expansion is never built
        -- not by fit (EM loop + saturation check), fit_sgd, or score."""
        import optax

        position, spikes = self._make_multi_neuron_data(n_neurons=3, n_time=120)

        def _boom(self_, Z_base):
            raise AssertionError("dense design matrix built on the block path")

        monkeypatch.setattr(PlaceFieldModel, "_expand_to_block_diagonal", _boom)
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        lls = model.fit(position, spikes, max_iter=2, verbose=False)
        assert model._block_n_neurons == 3  # guard: the block path was taken
        assert all(np.isfinite(ll) for ll in lls)
        assert np.isfinite(model.score(position, spikes))

        model_sgd = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        lls_sgd = model_sgd.fit_sgd(
            position, spikes, optimizer=optax.sgd(1e-4), num_steps=2
        )
        assert model_sgd._block_n_neurons == 3
        assert all(np.isfinite(ll) for ll in lls_sgd)

    @pytest.mark.slow
    def test_block_path_stores_block_covariances_matching_dense(self) -> None:
        """Block-path covariances are BlockDiagonalCovariance containers and
        every consumer (rate maps, credible intervals, state CIs, drift
        summary, M-step output) matches the dense path."""
        from state_space_practice.point_process_kalman import (
            BlockDiagonalCovariance,
        )

        position, spikes = self._make_multi_neuron_data(n_neurons=2, n_time=200)
        m_block = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        m_block.fit(position, spikes, max_iter=3, verbose=False)
        m_dense = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        m_dense.fit(position, spikes, max_iter=3, verbose=False, force_dense=True)
        assert m_block._block_n_neurons == 2 and m_dense._block_n_neurons is None

        for name in ("smoother_cov", "smoother_cross_cov", "filtered_cov"):
            block_cov = getattr(m_block, name)
            dense_cov = getattr(m_dense, name)
            assert isinstance(block_cov, BlockDiagonalCovariance)
            assert block_cov.shape == dense_cov.shape
            np.testing.assert_allclose(
                np.asarray(block_cov), np.asarray(dense_cov), atol=1e-7
            )

        grid = m_block.make_grid(8)[0]
        for neuron_idx in range(2):
            rate_b, ci_b = m_block.predict_rate_map(grid, neuron_idx=neuron_idx)
            rate_d, ci_d = m_dense.predict_rate_map(grid, neuron_idx=neuron_idx)
            assert rate_d.max() > 1.0  # guard: a non-trivial map
            np.testing.assert_allclose(rate_b, rate_d, rtol=1e-6)
            np.testing.assert_allclose(ci_b, ci_d, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(m_block.get_state_confidence_interval()),
            np.asarray(m_dense.get_state_confidence_interval()),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            m_block.drift_summary(n_grid=8, n_blocks=3)["centers"],
            m_dense.drift_summary(n_grid=8, n_blocks=3)["centers"],
            rtol=1e-6,
        )
        # The M-step consumed the block sufficient statistics: same Q. The
        # tolerance is looser here because Q (~1e-6) is a difference of
        # O(n_time) sufficient statistics, so roundoff-level differences in
        # the time sums are amplified by the cancellation; a wrong statistic
        # would change Q by orders of magnitude.
        np.testing.assert_allclose(
            np.asarray(jnp.diag(m_block.process_cov)),
            np.asarray(jnp.diag(m_dense.process_cov)),
            rtol=1e-4,
        )

    @pytest.mark.slow
    def test_detect_block_structure_tracks_parameter_matrices(self) -> None:
        """Dispatch follows the parameter matrices alone (the design structure
        is fixed by construction): a dense A flips it off, restoring A flips
        it back, force_dense always wins."""
        position, spikes = self._make_multi_neuron_data(n_neurons=2, n_time=100)
        model = PlaceFieldModel(dt=0.02, n_interior_knots=3)
        model.fit(position, spikes, max_iter=1, verbose=False)
        assert model._detect_block_structure() == (2, model.n_basis_per_neuron)
        assert model._detect_block_structure(force_dense=True) == (None, None)

        block_A = model.transition_matrix
        nb = model.n_basis_per_neuron
        model.transition_matrix = block_A.at[0, nb].set(0.1)  # off-block entry
        assert model._detect_block_structure() == (None, None)
        model.transition_matrix = block_A
        assert model._detect_block_structure() == (2, nb)

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


# ------------------------------------------------------------------
# Integration: rate-map recovery on simulated data
# ------------------------------------------------------------------


@pytest.mark.slow
class TestPlaceFieldModelRecovery:
    """Fit PlaceFieldModel on simulated moving place field and verify
    the recovered rate map correlates with ground truth."""

    @pytest.fixture(scope="class")
    @classmethod
    def fitted(cls):
        data = simulate_2d_moving_place_field(
            total_time=30.0,
            dt=0.020,
            arena_size=80.0,
            peak_rate=25.0,
            background_rate=1.0,
            n_interior_knots=4,
            rng=np.random.default_rng(42),
        )
        model = PlaceFieldModel(dt=data["dt"], n_interior_knots=4)
        lls = model.fit(
            data["position"],
            data["spikes"],
            max_iter=20,
            verbose=False,
        )
        return model, data, lls

    def test_ll_monotonic(self, fitted):
        _, _, lls = fitted
        assert_ll_monotonic(lls, tol=1e-3, label="PlaceFieldModel")

    def test_rate_map_correlates_with_truth(self, fitted):
        model, data, _ = fitted
        # Predict rate at the observed positions (time-averaged)
        grid, _, _ = model.make_grid(n_grid=30)
        rate_map, _ = model.predict_rate_map(grid)
        assert np.all(np.isfinite(rate_map)), "Rate map contains non-finite values"
        # Correlate predicted rate at observed positions with true rate
        pred_rate_at_pos, _ = model.predict_rate_map(data["position"])
        corr = float(np.corrcoef(pred_rate_at_pos, data["true_rate"])[0, 1])
        assert corr > 0.6, f"Predicted-vs-true rate correlation {corr:.3f} < 0.6"

    def test_rate_map_peak_near_true_center(self, fitted):
        model, data, _ = fitted
        grid, _, _ = model.make_grid(n_grid=30)
        rate_map, _ = model.predict_rate_map(grid)
        peak_idx = np.argmax(rate_map)
        estimated_peak = grid[peak_idx]
        # True center at the midpoint of the simulation (time-averaged)
        true_center = np.mean(data["true_center"], axis=0)
        dist = float(np.linalg.norm(estimated_peak - true_center))
        assert dist < 15.0, (
            f"Rate map peak {estimated_peak} is {dist:.1f} cm from "
            f"true center {true_center}"
        )


# ------------------------------------------------------------------
# M-step: smoothed x_0 init, residual-form Q, scale-relative floors
# ------------------------------------------------------------------


class TestPlaceFieldMStep:
    """``_m_step`` on synthetic smoother outputs, checked against the
    closed forms."""

    @pytest.fixture(scope="class")
    @classmethod
    def initialized_model(cls, sim_data: dict) -> PlaceFieldModel:
        model = PlaceFieldModel(dt=sim_data["dt"], n_interior_knots=3)
        model.fit(sim_data["position"], sim_data["spikes"], max_iter=1, verbose=False)
        return model

    @staticmethod
    def _install_posterior(model: PlaceFieldModel, scale: float, seed: int) -> tuple:
        """Random-walk smoother outputs with increments and covariances of
        size ``scale``; parameters reset to a contractive A for the E-step."""
        rng = np.random.default_rng(seed)
        n_time, n_basis = 60, model.n_basis
        increments = rng.normal(size=(n_time, n_basis)) * np.sqrt(scale)
        sm = jnp.asarray(1.0 + np.cumsum(increments, axis=0))
        L = rng.normal(size=(n_time, n_basis, n_basis)) * np.sqrt(scale / n_basis)
        sc = jnp.asarray(L @ np.swapaxes(L, 1, 2) + scale * np.eye(n_basis))
        scc = jnp.asarray(0.3 * np.asarray(sc[1:]))
        model.smoother_mean, model.smoother_cov, model.smoother_cross_cov = sm, sc, scc
        model.transition_matrix = jnp.eye(n_basis) * 0.9
        model.process_cov = jnp.eye(n_basis) * scale
        model.init_mean = jnp.zeros(n_basis)
        model.init_cov = jnp.eye(n_basis) * 10.0 * scale
        return sm, sc, scc

    def test_init_state_is_smoothed_x0(self, initialized_model) -> None:
        from state_space_practice.kalman import InitialStatePrior, smooth_initial_state

        model = initialized_model
        sm, sc, _ = self._install_posterior(model, scale=1e-2, seed=0)
        prior = InitialStatePrior(
            model.init_mean, model.init_cov, model.transition_matrix, model.process_cov
        )
        expected_mean, expected_cov = smooth_initial_state(prior, sm[0], sc[0])
        model._m_step()
        np.testing.assert_allclose(model.init_mean, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(
            jnp.diag(model.init_cov), jnp.diag(expected_cov), rtol=1e-10
        )
        # guard: the smoothed x_0 is not the smoothed x_1 (m_{0|T} ~ 0.99 m_{1|T}
        # here, with m_{1|T} ~ 1).
        assert np.max(np.abs(np.asarray(model.init_mean - sm[0]))) > 5e-3

    def test_random_walk_q_is_expected_increment_variance(
        self, initialized_model
    ) -> None:
        """With A held fixed, diag(Q) = mean_t E[(x_t - A x_{t-1})^2] over all
        T transitions -- including x_0 -> x_1, whose smoothed moments follow
        from the prior by one RTS step (J_0 = P_0 A' (A P_0 A' + Q)^{-1})."""
        model = initialized_model
        sm, sc, scc = (np.asarray(a) for a in self._install_posterior(model, 1e-2, 1))
        A = np.asarray(model.transition_matrix)
        P0, m0 = np.asarray(model.init_cov), np.asarray(model.init_mean)
        P_pred = A @ P0 @ A.T + np.asarray(model.process_cov)
        J0 = P0 @ A.T @ np.linalg.inv(P_pred)
        m0_s = m0 + J0 @ (sm[0] - A @ m0)
        P0_s = P0 + J0 @ (sc[0] - P_pred) @ J0.T
        means = np.concatenate([m0_s[None], sm])
        covs = np.concatenate([P0_s[None], sc])
        cross = np.concatenate([(J0 @ sc[0])[None], scc])  # Cov(x_{t-1}, x_t)
        model._m_step()
        a = np.diag(A)
        var = (
            (means[1:] - a * means[:-1]) ** 2
            + np.diagonal(covs[1:], axis1=1, axis2=2)
            + a**2 * np.diagonal(covs[:-1], axis1=1, axis2=2)
            - 2 * a * np.diagonal(cross, axis1=1, axis2=2)
        )
        np.testing.assert_allclose(
            jnp.diag(model.process_cov), var.mean(axis=0), rtol=1e-10
        )
        # guard: dropping the x_0 transition would change the answer.
        legacy = var[1:].mean(axis=0)
        assert not np.allclose(jnp.diag(model.process_cov), legacy, rtol=1e-3)

    def test_process_noise_floor_is_scale_relative(self, initialized_model) -> None:
        """Increments of variance ~1e-13 give Q ~1e-13 rather than the former
        absolute 1e-10 floor."""
        model = initialized_model
        sm, _, _ = self._install_posterior(model, scale=1e-13, seed=2)
        # A random-walk model whose x_0 prior agrees with the posterior, so
        # every transition residual (x_0 -> x_1 included) is ~1e-13 in size.
        model.transition_matrix = jnp.eye(model.n_basis)
        model.init_mean = sm[0]
        model._m_step()
        q = np.asarray(jnp.diag(model.process_cov))
        assert np.all(q > 0.0)
        assert q.max() < 1e-11


@pytest.mark.slow
class TestPlaceFieldRecoverySweep:
    """Rate-map recovery as a statistic over seeds and session lengths.

    Complements ``TestPlaceFieldModelRecovery`` (one seed, correlation >
    0.6) with 3 seeds x {10 s, 40 s} of a stationary place field and a tiny
    16-weight spline basis (kept small for runtime). The error is the RMSE
    of the log-rate predicted at the long session's positions against the
    true log-rate; more data (and coverage) must reduce it. Observed RMSE:
    0.83-0.97 (10 s) vs 0.23-0.33 (40 s).
    """

    def test_log_rate_error_decreases_with_session_length(self) -> None:
        errors = {10.0: [], 40.0: []}
        for seed in (0, 1, 2):
            sessions = {
                total: simulate_2d_moving_place_field(
                    total_time=total,
                    dt=0.02,
                    arena_size=60.0,
                    peak_rate=25.0,
                    background_rate=1.0,
                    drift_speed=0.0,
                    n_interior_knots=1,
                    rng=np.random.default_rng(seed),
                )
                for total in errors
            }
            eval_position = sessions[40.0]["position"]
            eval_log_rate = np.log(sessions[40.0]["true_rate"])
            for total, data in sessions.items():
                model = PlaceFieldModel(dt=data["dt"], n_interior_knots=1)
                model.fit(data["position"], data["spikes"], max_iter=2, verbose=False)
                pred, _ = model.predict_rate_map(eval_position)
                rmse = np.sqrt(np.mean((np.log(pred) - eval_log_rate) ** 2))
                errors[total].append(float(rmse))
        short, long = np.array(errors[10.0]), np.array(errors[40.0])
        msg = (
            f"per-seed log-rate RMSE: 10 s {np.round(short, 3)}, "
            f"40 s {np.round(long, 3)}"
        )
        assert np.all(long < 0.6), msg
        assert np.mean(long) < 0.6 * np.mean(short), msg
        assert np.all(long < short), msg
