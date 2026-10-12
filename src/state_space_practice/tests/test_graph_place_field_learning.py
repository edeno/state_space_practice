"""Behavioral checks of the supported profile/gradient graph estimator."""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import state_space_practice.graph_place_field as gpf
from state_space_practice.exceptions import (
    NonFiniteLikelihoodError,
    NotFittedError,
    StateSpaceWarning,
)
from state_space_practice.fitted_state import is_set
from state_space_practice.graph_place_field import GraphPlaceFieldModel
from state_space_practice.tests.graph_helpers import line_env
from state_space_practice.tests.graph_math_reference import joint_poisson_laplace

neurospatial = pytest.importorskip("neurospatial")


@pytest.fixture(scope="module")
def learning_env():
    return line_env()


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


@pytest.mark.slow
def test_score_raises_when_joint_inference_is_unconverged(learning_env, scalar_data):
    model = scalar_model(learning_env, 0.03)
    model.update_drift_scale = False
    model.fit_mle(*scalar_data, warm_start=False)
    assert np.isfinite(model.score(*scalar_data))
    model.max_smoother_iter = 2
    with pytest.raises(NonFiniteLikelihoodError, match="max_smoother_iter"):
        model.score(*scalar_data)


@pytest.mark.slow
def test_profile_warns_about_unconverged_scales(learning_env, scalar_data):
    model = scalar_model(learning_env, 0.03)
    model.max_smoother_iter = 4
    with pytest.warns(StateSpaceWarning, match="drift profile"):
        model.fit_mle(*scalar_data, warm_start=False)
    assert model.drift_profiles_[0].n_nonfinite > 0


@pytest.mark.slow
def test_fit_warns_when_optimizer_budget_is_exhausted(learning_env, scalar_data):
    model = GraphPlaceFieldModel(learning_env, 0.1, rank=2)
    with pytest.warns(StateSpaceWarning, match="did not converge"):
        model.fit(*scalar_data, max_iter=1, profile_rounds=1, profile_points=5)
    assert not model.converged_


@pytest.mark.slow
def test_failed_fit_restores_starting_hyperparameters(learning_env, scalar_data):
    model = GraphPlaceFieldModel(learning_env, 0.1, rank=2)
    model.max_smoother_iter = 2
    start = (model.kappa2, model.tau2, model.init_mean, model.drift_scale)
    with pytest.raises(ValueError, match="finite likelihood"):
        model.fit(*scalar_data, profile_points=5, profile_rounds=1)
    assert (model.kappa2, model.tau2, model.init_mean, model.drift_scale) == start
    assert not model._is_fitted


@pytest.mark.slow
def test_unusable_static_candidate_keeps_positive_fit(
    learning_env, scalar_data, monkeypatch
):
    model = scalar_model(learning_env, 0.03)
    optimize = model._fit_lbfgs_prepared

    def static_candidate_fails(*args, **kwargs):
        mask = model._mle_static_mask
        if mask is not None and bool(jnp.all(mask)):
            # Same contract as an unusable evidence evaluation.
            model._clear_fit_state()
            raise NonFiniteLikelihoodError("L-BFGS initial objective is non-finite.")
        return optimize(*args, **kwargs)

    monkeypatch.setattr(model, "_fit_lbfgs_prepared", static_candidate_fails)
    with pytest.warns(StateSpaceWarning, match="zero-drift candidate"):
        model.fit_mle(*scalar_data, warm_start=False)
    assert float(model.drift_scale[0]) > 1e-3
    assert model.converged_
    assert np.isfinite(model.score(*scalar_data))


@pytest.mark.slow
def test_joint_fit_skips_the_causal_filter(learning_env, scalar_data, caplog):
    """Joint inference never reads the causal filter, so it does not run it."""
    fitted = {}
    for inference_method in ("joint", "sequential"):
        model = GraphPlaceFieldModel(
            learning_env,
            0.1,
            rank=1,
            max_newton_iter=2,  # leaves the causal filter unconverged here
            init_drift_scale=0.03,
            inference_method=inference_method,
            update_drift_scale=False,
            update_amplitude=False,
            update_kappa2=False,
            update_init_mean=False,
        )
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="state_space_practice"):
            model.fit(*scalar_data, warm_start=False)
            jax.effects_barrier()
        fitted[inference_method] = [
            r.getMessage()
            for r in caplog.records
            if "Newton updates did not converge" in r.getMessage()
        ]
        if inference_method == "joint":
            assert not is_set(model, "filtered_mean")
            assert not is_set(model, "filtered_cov")
            assert bool(model.smoother_diagnostics_.converged.all())
        else:
            assert is_set(model, "filtered_mean") and is_set(model, "filtered_cov")
    assert fitted["sequential"]  # guard: the filter really is unconverged here
    assert not fitted["joint"]


@pytest.mark.slow
def test_static_candidate_replaces_positive_profile_choice(learning_env, monkeypatch):
    """Re-optimized zero drift wins over the profile's positive scale here.

    The single-round profile at the warm-start nuisance parameters prefers a
    small positive q, but L-BFGS of the static candidate reaches at least the
    positive-drift evidence, so the fit must store exact zero drift.
    """
    rng = np.random.default_rng(0)
    n_time = 100
    bins = rng.integers(learning_env.n_bins, size=n_time)
    positions = np.asarray(learning_env.bin_centers)[bins]
    rate_hz = 20 * np.exp(-((np.arange(learning_env.n_bins) - 2.5) ** 2) / 4)
    counts = rng.poisson(rate_hz[bins] * 0.1).astype(float)  # stationary field
    model = GraphPlaceFieldModel(learning_env, 0.1, rank=2)

    profile_scales, candidates = [], []
    profile, optimize = model._profile_drift, model._fit_lbfgs_prepared

    def recording_profile(*args, **kwargs):
        profiles = profile(*args, **kwargs)
        profile_scales.append([p.scale for p in profiles])
        return profiles

    def recording_optimize(*args, **kwargs):
        static = bool(jnp.all(model._mle_static_mask))
        result = optimize(*args, **kwargs)
        candidates.append((static, model.log_likelihood_))
        return result

    monkeypatch.setattr(model, "_profile_drift", recording_profile)
    monkeypatch.setattr(model, "_fit_lbfgs_prepared", recording_optimize)
    model.fit_mle(
        np.arange(n_time) * 0.1,
        positions,
        counts,
        profile_rounds=1,
        profile_points=9,
    )

    # Guard: the round profile chose positive drift, so the positive-drift
    # optimization ran first and the static candidate then competed with it.
    assert profile_scales[0][0] > 0
    assert [static for static, _ in candidates] == [False, True]
    (_, positive_ll), (_, static_ll) = candidates
    assert static_ll >= positive_ll - 1e-6

    assert float(model.drift_scale[0]) == 0.0
    assert bool(jnp.all(model._mle_static_mask))
    assert "drift_scale" not in model.parameter_bound_hits_
    assert model.converged_
    assert model.log_likelihood_ == static_ll
    assert model.drift_profiles_[0].scale == 0  # final diagnostic profile agrees


@pytest.mark.slow
def test_partitioned_static_neurons_match_batched_evidence(learning_env, scalar_data):
    """Grouping static and drifting neurons leaves the objective and its gradient."""
    times, positions, counts = scalar_data
    spikes = np.stack([counts, counts[::-1], np.ones_like(counts)], axis=1)
    model = GraphPlaceFieldModel(learning_env, 0.1, rank=2)
    args, _ = model._prepare_sgd_data(times, positions, spikes, warm_start=True)
    drift = jnp.array([0.0, 0.03, 0.0])
    params = dict(model._current_params(), drift_scale=drift)

    def loss(q, static_neurons):
        return gpf._graph_forward_loss(
            dict(params, drift_scale=q),
            *args,
            jnp.asarray(model.basis.eigvals),
            model.transition_matrix,
            dt=model.dt,
            alpha=model.alpha,
            max_log_count=model._max_log_count,
            max_newton_iter=model.max_newton_iter,
            inference_method="joint",
            max_smoother_iter=model.max_smoother_iter,
            static_neurons=static_neurons,
        )

    batched = loss(drift, None)
    assert np.isfinite(float(batched))
    partitioned = loss(drift, (True, False, True))
    assert float(partitioned) == pytest.approx(float(batched), rel=1e-12)
    grad_batched = jax.grad(loss)(drift, None)
    grad_partitioned = jax.grad(loss)(drift, (True, False, True))
    np.testing.assert_allclose(grad_partitioned[1], grad_batched[1], rtol=1e-10)
    np.testing.assert_array_equal(np.asarray(grad_partitioned)[[0, 2]], 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("inference_method", ["joint", "sequential"])
def test_drift_profile_losses_equal_pointwise_objective(
    learning_env, scalar_data, inference_method
):
    times, positions, counts = scalar_data
    spikes = np.stack([counts, counts[::-1]], axis=1)
    model = GraphPlaceFieldModel(
        learning_env, 0.1, rank=2, inference_method=inference_method
    )
    args, _ = model._prepare_sgd_data(times, positions, spikes, warm_start=True)
    profiles = model._profile_drift(*args, (1e-8, 10.0), 5)
    base = model._current_params()
    factor = float(model._field_shape_factor())
    Z, spk, valid = args
    for cell, profile in enumerate(profiles):
        assert np.all(np.isfinite(profile.losses))
        for q, loss in zip(profile.scales, profile.losses, strict=True):
            params = dict(
                base,
                init_mean=base["init_mean"][cell : cell + 1],
                drift_scale=jnp.atleast_1d(q / factor),
            )
            expected = float(
                model._forward_loss(params, Z, spk[cell : cell + 1], valid, (q == 0,))
            )
            assert loss == pytest.approx(expected, rel=1e-10, abs=1e-9)


@pytest.mark.slow
def test_neuron_chunking_matches_unchunked(learning_env, scalar_data, monkeypatch):
    times, positions, counts = scalar_data
    spikes = np.stack(
        [counts, counts[::-1], np.ones_like(counts), counts, counts[::-1]], axis=1
    )
    model = GraphPlaceFieldModel(learning_env, 0.1, rank=2)
    args, _ = model._prepare_sgd_data(times, positions, spikes, warm_start=True)
    drift = jnp.array([0.0, 0.03, 0.0, 0.01, 0.02])
    static = tuple(bool(d == 0) for d in np.asarray(drift))
    model.drift_scale = drift
    params = model._current_params()
    chunked_calls = []
    lax_map = jax.lax.map

    def counting_map(*a, **k):
        chunked_calls.append(k.get("batch_size"))
        return lax_map(*a, **k)

    monkeypatch.setattr(jax.lax, "map", counting_map)
    results = {}
    for batch in (None, 2):  # 2 leaves an uneven last chunk in each noise group
        chunked_calls.clear()
        model.neuron_batch_size = batch
        value, grad = jax.value_and_grad(
            lambda q: model._forward_loss(dict(params, drift_scale=q), *args, static)
        )(drift)
        evidence = model._e_step(*args)
        results[batch] = (value, grad, evidence, np.asarray(model.smoother_mean))
        # Guard: chunking is traced only when a batch size is set.
        assert (2 in chunked_calls) == (batch == 2)
    assert np.isfinite(results[None][0])
    for expected, actual in zip(results[None], results[2], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("inference_method", "drift", "blocked_compiles"),
    [
        # Blocks of 2 in neuron order would see 4 patterns: (T, F), (F, T),
        # (T, F) and a short (F,).
        ("joint", [0.0, 0.03, 0.01, 0.0, 0.02, 0.0, 0.01, 0.04, 0.03], 2),
        ("joint", [0.01, 0.03, 0.01, 0.02, 0.02, 0.01, 0.01, 0.04, 0.03], 1),
        ("sequential", [0.0, 0.03, 0.01, 0.0, 0.02, 0.0, 0.01, 0.04, 0.03], 1),
    ],
    ids=["joint-mixed", "joint-drifting", "sequential"],
)
def test_neuron_blocks_compile_once_per_noise_group(
    learning_env, scalar_data, inference_method, drift, blocked_compiles
):
    """Blocked inference compiles one program per noise group, not one per
    block shape or static/drifting pattern, and matches the unblocked result."""
    times, positions, counts = scalar_data
    spikes = np.stack([counts, counts[::-1], np.ones_like(counts)] * 3, axis=1)
    model = GraphPlaceFieldModel(
        learning_env, 0.1, rank=2, inference_method=inference_method
    )
    args, _ = model._prepare_sgd_data(times, positions, spikes, warm_start=True)
    model.drift_scale = jnp.array(drift)
    results = {}
    for batch in (None, 2):
        model.neuron_batch_size = batch
        gpf._graph_posterior.clear_cache()
        evidence = model._e_step(*args)
        results[batch] = (
            evidence,
            np.asarray(model.smoother_mean),
            np.asarray(model.smoother_cov),
        )
        expected_compiles = 1 if batch is None else blocked_compiles
        assert gpf._graph_posterior._cache_size() == expected_compiles
    assert np.isfinite(results[None][0])
    for expected, actual in zip(results[None], results[2], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.slow
def test_graph_parallel_fit_matches_sequential(learning_env, scalar_data):
    fitted = {}
    for parallel in (False, True):
        model = GraphPlaceFieldModel(learning_env, 0.1, rank=2, parallel=parallel)
        model.fit(*scalar_data, profile_points=9, profile_rounds=1)
        assert model.converged_
        fitted[parallel] = model
    assert fitted[True].log_likelihood_ == pytest.approx(
        fitted[False].log_likelihood_, abs=1e-6
    )
    # The two passes differ in the last bit, so L-BFGS stops at slightly
    # different points within its gradient tolerance; the evidence agrees.
    np.testing.assert_allclose(
        fitted[True].drift_scale, fitted[False].drift_scale, rtol=1e-3, atol=1e-12
    )


def test_neuron_chunks_rematerialise_under_differentiation(learning_env, scalar_data):
    """Chunking bounds gradient memory only if each chunk is recomputed backward."""
    times, positions, counts = scalar_data
    spikes = np.stack([counts, counts[::-1], counts], axis=1)
    model = GraphPlaceFieldModel(learning_env, 0.1, rank=2)
    args, _ = model._prepare_sgd_data(times, positions, spikes, warm_start=True)
    params = model._current_params()

    def primitives(batch):
        model.neuron_batch_size = batch
        jaxpr = jax.make_jaxpr(
            jax.grad(lambda q: model._forward_loss(dict(params, drift_scale=q), *args))
        )(params["drift_scale"])
        return str(jaxpr)

    assert "remat" not in primitives(None)  # jax.checkpoint prints as remat
    assert "remat" in primitives(2)


def test_neuron_chunks_do_not_recompute_the_newton_iteration(learning_env, scalar_data):
    """With the static/drifting pattern known (the L-BFGS fit path), chunk
    checkpoints save the converged mode, so the backward pass replays only the
    exact Newton map and final linear pass, not the adaptive iteration."""
    times, positions, counts = scalar_data
    spikes = np.stack([counts, counts[::-1], counts], axis=1)
    model = GraphPlaceFieldModel(learning_env, 0.1, rank=2)
    args, _ = model._prepare_sgd_data(times, positions, spikes, warm_start=True)
    params = model._current_params()
    static = (False, True, False)

    def newton_loops(batch):
        model.neuron_batch_size = batch
        jaxpr = jax.make_jaxpr(
            jax.grad(
                lambda q: model._forward_loss(
                    dict(params, drift_scale=q), *args, static
                )
            )
        )(params["drift_scale"])
        return str(jaxpr).count("while[")

    assert newton_loops(None) >= 1  # guard: the joint iteration is present
    assert newton_loops(2) == newton_loops(None)


@pytest.mark.slow
def test_compact_posterior_fit_matches_full(learning_env, scalar_data):
    """Without stored covariances the fit, means and rate maps are unchanged."""
    times, positions, counts = scalar_data
    spikes = np.stack([counts, counts[::-1]], axis=1)
    fitted = {}
    for store in (True, False):
        model = GraphPlaceFieldModel(learning_env, 0.1, rank=2, store_covariances=store)
        model.fit(times, positions, spikes, profile_points=9, profile_rounds=1)
        fitted[store] = model
    full, compact = fitted[True], fitted[False]
    assert compact.log_likelihood_ == pytest.approx(full.log_likelihood_, rel=1e-12)
    np.testing.assert_allclose(compact.drift_scale, full.drift_scale, rtol=1e-10)
    np.testing.assert_allclose(
        compact.smoother_mean, full.smoother_mean, rtol=1e-10, atol=1e-12
    )
    for name in ("smoother_cov", "smoother_cross_cov"):
        assert is_set(full, name) and not is_set(compact, name), name
    for neuron in (0, 1):
        np.testing.assert_allclose(
            compact.predict_rate_map(neuron), full.predict_rate_map(neuron), rtol=1e-10
        )
    for model in (full, compact):
        diagnostics = model.smoother_diagnostics_
        assert not hasattr(diagnostics, "covariance")
        for field in diagnostics._fields:
            assert np.shape(getattr(diagnostics, field)) == (2,), field
        assert bool(np.all(diagnostics.converged))


@pytest.mark.slow
def test_compact_posterior_em_matches_full(learning_env, scalar_data):
    """EM's M-step needs only per-neuron summaries, so both storage modes agree."""
    fitted = {}
    for store in (True, False):
        model = GraphPlaceFieldModel(learning_env, 0.1, rank=2, store_covariances=store)
        model.fit(*scalar_data, method="em", max_iter=3)
        fitted[store] = model
    full, compact = fitted[True], fitted[False]
    assert compact.tau2 == pytest.approx(full.tau2, rel=1e-10)
    np.testing.assert_allclose(compact.drift_scale, full.drift_scale, rtol=1e-10)
    np.testing.assert_allclose(
        compact.init_mean, full.init_mean, rtol=1e-10, atol=1e-12
    )


def test_compact_posterior_storage_scales_with_rank(learning_env, scalar_data):
    model = GraphPlaceFieldModel(learning_env, 0.1, rank=3, store_covariances=False)
    args, _ = model._prepare_sgd_data(*scalar_data, warm_start=True)
    model._e_step(*args)
    n_time = args[0].shape[0]
    assert np.shape(model.smoother_mean) == (1, n_time, 3)
    for name in ("smoother_cov", "smoother_cross_cov", "filtered_mean", "filtered_cov"):
        assert not is_set(model, name), name
