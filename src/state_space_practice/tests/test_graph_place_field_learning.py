"""Behavioral checks of the supported profile/gradient graph estimator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from state_space_practice.exceptions import NotFittedError
from state_space_practice.fitted_state import is_set
from state_space_practice.graph_place_field import GraphPlaceFieldModel
from state_space_practice.tests.graph_math_reference import joint_poisson_laplace

neurospatial = pytest.importorskip("neurospatial")
from neurospatial import Environment  # noqa: E402


@pytest.fixture(scope="module")
def learning_env():
    return Environment.from_samples(np.linspace(0, 10, 201)[:, None], bin_size=2.0)


@pytest.fixture(scope="module")
def scalar_data(learning_env):
    times = np.arange(24) * 0.1
    positions = np.repeat(learning_env.bin_centers[:1], len(times), axis=0)
    positions[[3, 9, 10, 17]] = 1e6
    counts = np.array(
        [0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 0, 1, 2, 1, 4, 3, 3, 0, 4, 5, 4, 3, 5, 4],
        dtype=float,
    )
    return times, positions, counts


def scalar_model(env, start):
    model = GraphPlaceFieldModel(
        env,
        0.1,
        rank=1,
        inference_method="joint",
        update_drift_scale=True,
        update_amplitude=False,
        update_kappa2=False,
        update_init_mean=False,
        init_drift_scale=start,
    )
    phi = float(model.basis.eigvecs[0, 0])
    model.init_mean = jnp.array([[np.log(10) / phi]])
    model.tau2 = 0.08 / phi**2
    return model


@pytest.mark.slow
@pytest.mark.parametrize("start", [0.0, 0.001, 0.3])
def test_profile_fit_matches_independent_dense_evidence(
    learning_env, scalar_data, start
):
    times, positions, counts = scalar_data
    model = scalar_model(learning_env, start)
    z, spikes, valid = model._design_and_spikes(*scalar_data)

    def objective(q):
        return -joint_poisson_laplace(
            np.asarray(model.init_mean[0]),
            np.asarray(model.prior_cov()),
            np.asarray(model.drift_cov(q)),
            np.asarray(z),
            np.asarray(spikes[0]),
            np.asarray(valid),
            0.1,
        ).log_evidence

    grid = np.geomspace(1e-8, 10, 41)
    values = [objective(q) for q in grid]
    best = int(np.argmin(values))
    opt = minimize_scalar(
        lambda x: objective(np.exp(x)),
        bounds=np.log(grid[[max(0, best - 1), min(len(grid) - 1, best + 1)]]),
        method="bounded",
    )
    expected = min(objective(0), opt.fun)
    model.fit_mle(times, positions, counts, warm_start=False)
    assert float(model.drift_scale[0]) > 1e-3  # informative positive example
    assert -model.log_likelihood_ == pytest.approx(expected, abs=0.001)
    assert model.converged_
    assert model.drift_profiles_[0].loss == pytest.approx(expected, abs=0.001)


@pytest.mark.slow
def test_static_boundary_and_refit_resume(learning_env, scalar_data):
    times, positions, _ = scalar_data
    model = scalar_model(learning_env, 0.3)
    model.fit_mle(times, positions, np.ones(len(times)), warm_start=False)
    assert float(model.drift_scale[0]) == 0
    assert model.drift_profiles_[0].scale == 0
    first = model.log_likelihood_
    model.fit_mle(times, positions, np.ones(len(times)), warm_start=False)
    assert model.log_likelihood_ == pytest.approx(first, abs=1e-8)
    assert float(model.drift_scale[0]) == 0
    assert not model._mle_field_coordinates


@pytest.mark.slow
def test_joint_learning_masks_counts_and_constrains_prior_means(learning_env):
    rng = np.random.default_rng(22)
    times = np.arange(200) * 0.1
    positions = learning_env.bin_centers[
        rng.integers(learning_env.n_bins, size=200)
    ].copy()
    missing = np.arange(200) % 10 == 3
    positions[missing] = 1e6
    counts = rng.poisson(np.exp(np.linspace(-1, 2, 200)))
    results = []
    for poison, initial_q in ((0, 0.0), (0, 1e-3), (10000, 1e-3)):
        model = GraphPlaceFieldModel(
            learning_env,
            0.1,
            rank=3,
            update_drift_scale=True,
            inference_method="joint",
            init_drift_scale=initial_q,
        )
        model.fit_mle(times, positions, np.where(missing, poison, counts))
        results.append(model)
        assert np.all(np.asarray(model.init_mean)[:, model.basis.n_components :] == 0)
        assert float(model.drift_scale[0]) > 0
        assert model.converged_
        assert model.smoother_diagnostics_.relative_newton_step.max() < 1e-8
        factor = (
            np.sum(np.asarray(model._spectral_shape_current())) / learning_env.n_bins
        )
        np.testing.assert_allclose(
            model.field_drift_scale_, np.asarray(model.drift_scale) * factor
        )
    for result in results[1:]:
        assert results[0].log_likelihood_ == pytest.approx(
            result.log_likelihood_, abs=1e-8
        )
        np.testing.assert_allclose(
            results[0].smoother_mean, result.smoother_mean, atol=1e-8
        )


@pytest.mark.slow
def test_fixed_scales_and_failed_refit_clear_diagnostics(learning_env, scalar_data):
    model = GraphPlaceFieldModel(
        learning_env,
        0.1,
        rank=2,
        update_drift_scale=False,
        init_drift_scale=0.02,
        inference_method="joint",
    )
    model.fit_mle(*scalar_data)
    assert float(model.drift_scale[0]) == 0.02
    assert model.drift_profiles_ == ()
    assert -model.optimizer_result_.loss * len(scalar_data[0]) == pytest.approx(
        model.log_likelihood_, abs=1e-8
    )
    with pytest.raises(ValueError, match="profile"):
        model.fit_mle(*scalar_data, profile_points=1)
    assert not is_set(model, "optimizer_result_")
    assert not is_set(model, "parameter_bound_hits_")
    assert not model._mle_field_coordinates
    with pytest.raises(NotFittedError):
        model.predict_log_rate_trajectory()


@pytest.mark.slow
def test_field_coordinate_gradients_include_kappa_dependence(learning_env, scalar_data):
    model = GraphPlaceFieldModel(
        learning_env, 0.1, rank=3, update_drift_scale=True, inference_method="joint"
    )
    args, _ = model._prepare_sgd_data(*scalar_data, warm_start=True)
    model._mle_field_coordinates = True
    model._mle_baseline_mean = True
    model.init_mean = model.init_mean.at[:, 1:].set(0)
    params, _ = model._build_param_spec()

    def loss(theta):
        updated = dict(
            params,
            kappa2=jnp.exp(theta[0]),
            tau2=jnp.exp(theta[1]),
            drift_scale=jnp.exp(theta[2:3]),
        )
        return model._sgd_loss_fn(updated, *args)

    point = jnp.array([0.0, 0.0, -4.0])
    grad = np.asarray(jax.grad(loss)(point))
    assert np.linalg.norm(grad) > 0.1
    for i in range(3):
        step = np.eye(3)[i] * 1e-5
        fd = (float(loss(point + step)) - float(loss(point - step))) / 2e-5
        assert grad[i] == pytest.approx(fd, rel=2e-5, abs=1e-7)


@pytest.mark.slow
def test_constructor_defaults_use_automatic_fit_and_score(learning_env):
    """A plain constructor and fit learn scales without a favorable start/budget."""
    rng = np.random.default_rng(52)
    times = np.arange(300) * 0.1
    positions = learning_env.bin_centers[rng.integers(learning_env.n_bins, size=300)]
    counts = rng.poisson(np.exp(np.linspace(-1, 2, 300)))
    normal = GraphPlaceFieldModel(learning_env, 0.1)
    history = normal.fit(times, positions, counts)
    assert normal.update_drift_scale
    assert normal.inference_method == "joint"
    assert normal.rank == learning_env.n_bins
    assert normal.converged_
    assert float(normal.drift_scale[0]) > 0
    assert not np.isclose(
        float(normal.drift_scale[0]), normal.init_drift_scale, rtol=0.1
    )
    assert normal.n_iter_ is None
    assert normal.log_likelihood_history_ == history
    assert normal.log_likelihood_ == history[-1]
    state = np.asarray(normal.smoother_mean).copy()
    assert normal.score(times, positions, counts) == pytest.approx(
        history[-1], abs=1e-8
    )
    np.testing.assert_array_equal(normal.smoother_mean, state)
    alias = GraphPlaceFieldModel(learning_env, 0.1)
    alias.fit_mle(times, positions, counts)
    assert alias.log_likelihood_ == pytest.approx(normal.log_likelihood_, abs=1e-8)
    normal.fit_sgd(times, positions, counts, num_steps=1, warm_start=False)
    assert not is_set(normal, "drift_profiles_")
    assert not is_set(normal, "parameter_bound_hits_")
    assert not is_set(normal, "optimizer_result_")


@pytest.mark.slow
def test_unconverged_joint_evidence_is_rejected(learning_env):
    model = scalar_model(learning_env, 0.03)
    model.max_smoother_iter = 1
    model.init_mean = jnp.zeros((1, 1))
    model.tau2 = 6.0
    times = np.arange(2) * 0.1
    positions = np.repeat(learning_env.bin_centers[:1], 2, axis=0)
    with pytest.raises(ValueError, match="finite likelihood"):
        model.fit(times, positions, np.array([8.0, 0.0]), warm_start=False)
    assert not model._is_fitted
    assert not is_set(model, "smoother_diagnostics_")
