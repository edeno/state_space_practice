"""Mathematical contracts checked against independent dense/integral references.

Exact algebra and numerical identities have tight tolerances. Poisson posterior
moments are only required to be close in a declared benign regime; deliberately
skewed cases demonstrate, rather than conceal, Laplace approximation error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, logsumexp
from scipy.stats import norm, poisson

from state_space_practice.graph_place_field import (
    GraphPlaceFieldModel,
    _masked_graph_point_process_filter,
    fit_static_graph_glm,
    static_log_evidence,
)
from state_space_practice.kalman import rts_backward_scan
from state_space_practice.parameter_transforms import (
    transform_to_constrained,
    transform_to_unconstrained,
)
from state_space_practice.tests.graph_math_reference import (
    expected_gaussian_prior_nll,
    gaussian_filtered_moments,
    gaussian_joint_posterior,
    interval_mass,
    poisson_grid_posterior,
)

neurospatial = pytest.importorskip("neurospatial")
from neurospatial import Environment  # noqa: E402


def _legacy_graph_model(*args, **kwargs):
    """Keep the original sequential/EM regression targets explicit.

    Automatic constructor defaults are tested in test_graph_place_field_learning.
    """
    kwargs.setdefault("inference_method", "sequential")
    kwargs.setdefault("update_drift_scale", False)
    return GraphPlaceFieldModel(*args, **kwargs)


@pytest.fixture(scope="module")
def math_env():
    return Environment.from_samples(np.linspace(0, 10, 201)[:, None], bin_size=2.0)


@pytest.fixture(scope="module", params=["connected", "disconnected"])
def topology_env(request, math_env):
    if request.param == "connected":
        return math_env
    samples = np.r_[np.linspace(0, 4, 101), np.linspace(20, 24, 101)]
    return Environment.from_samples(samples[:, None], bin_size=2.0)


@pytest.mark.parametrize("convention", ["distance", "inverse_distance"])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_field_prior_equals_dense_graph_resolvent(topology_env, convention, alpha):
    """The prior in field coordinates equals a matrix function of the raw graph."""
    env = topology_env
    model = _legacy_graph_model(
        env, dt=0.1, kappa2=0.7, tau2=2.3, alpha=alpha, laplacian_convention=convention
    )
    laplacian = np.zeros((env.n_bins, env.n_bins))
    for i, j, data in env.connectivity.edges(data=True):
        distance = float(data["distance"])
        weight = distance if convention == "distance" else 1 / distance
        laplacian[i, i] += weight
        laplacian[j, j] += weight
        laplacian[i, j] -= weight
        laplacian[j, i] -= weight
    shifted = laplacian + 0.7 * np.eye(env.n_bins)
    inverse = np.linalg.solve(shifted, np.eye(env.n_bins))
    expected_shape = (
        inverse
        if alpha == 1
        else inverse @ inverse
        if alpha == 2
        else fractional_matrix_power(shifted, -alpha)
    )
    phi = np.asarray(model.basis.eigvecs)
    np.testing.assert_allclose(
        phi @ np.asarray(model.prior_cov()) @ phi.T,
        2.3 * expected_shape,
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        phi @ np.asarray(model.drift_cov(0.04)) @ phi.T,
        0.04 * expected_shape,
        atol=1e-12,
        rtol=1e-11,
    )
    w = np.random.default_rng(35).normal(size=model.rank)
    field = phi @ w
    edge_energy = sum(
        (
            float(data["distance"])
            if convention == "distance"
            else 1 / float(data["distance"])
        )
        * (field[i] - field[j]) ** 2
        for i, j, data in env.connectivity.edges(data=True)
    )
    assert edge_energy > 0.1
    assert float(w @ (model.basis.eigvals * w)) == pytest.approx(edge_energy, rel=1e-11)


@pytest.mark.parametrize("n_neurons", [1, 2])
def test_static_mode_curvature_and_evidence_match_independent_objective(
    math_env, n_neurons
):
    model = _legacy_graph_model(math_env, dt=0.1, rank=3, tau2=1.7, kappa2=0.6)
    phi = np.asarray(model.basis.eigvecs)
    occupancy = np.geomspace(0.1, 15.0, math_env.n_bins)
    occupancy[1] = 0.0
    counts = np.random.default_rng(36).poisson(
        occupancy[:, None]
        * np.exp(phi @ np.array([[2.0, 1.2], [-0.8, 0.6], [0.4, -0.5]]))
    )[:, :n_neurons]
    precision = (0.6 + np.asarray(model.basis.eigvals)) / 1.7
    weights, covariances = fit_static_graph_glm(
        counts, occupancy, phi, precision, max_iter=40, tol=1e-10
    )
    independent_evidence = 0.0
    for y, w, covariance in zip(
        counts.T, np.asarray(weights), np.asarray(covariances), strict=True
    ):

        def objective(x, y=y):
            mu = occupancy * np.exp(phi @ x)
            return float(-poisson.logpmf(y, mu).sum() + 0.5 * np.dot(precision * x, x))

        optimum = minimize(
            objective, np.zeros(3), method="BFGS", jac="3-point", options={"gtol": 1e-8}
        )
        assert np.linalg.norm(optimum.x) > 0.2
        np.testing.assert_allclose(w, optimum.x, atol=2e-6, rtol=2e-6)
        mu = occupancy * np.exp(phi @ w)
        gradient = phi.T @ (mu - y) + precision * w
        hessian = phi.T @ (mu[:, None] * phi) + np.diag(precision)
        np.testing.assert_allclose(gradient, 0, atol=1e-7)
        np.testing.assert_allclose(
            covariance, np.linalg.inv(hessian), atol=1e-10, rtol=1e-9
        )
        # Check the returned curvature against differences of the independent
        # objective, rather than only evaluating the same Hessian formula.
        direction = np.array([0.3, -0.4, 0.2])
        epsilon = 1e-3
        curvature = (
            objective(w + epsilon * direction)
            - 2 * objective(w)
            + objective(w - epsilon * direction)
        ) / epsilon**2
        assert curvature == pytest.approx(
            float(direction @ np.linalg.solve(covariance, direction)),
            rel=2e-6,
            abs=2e-6,
        )
        log_prior_normalizer = 0.5 * np.log(precision).sum() - 1.5 * np.log(2 * np.pi)
        laplace = (
            -objective(w)
            + log_prior_normalizer
            + 1.5 * np.log(2 * np.pi)
            - 0.5 * np.linalg.slogdet(hessian)[1]
        )
        independent_evidence += laplace + gammaln(y + 1).sum()
    actual = static_log_evidence(
        counts, occupancy, phi, model.basis.eigvals, tau2=1.7, kappa2=0.6
    )
    assert actual == pytest.approx(independent_evidence, abs=1e-8)


@pytest.fixture(scope="module")
def gaussian_problem(math_env):
    model = _legacy_graph_model(math_env, dt=0.1, rank=3, tau2=1.3, kappa2=0.7)
    indices = [0, math_env.n_bins - 1, 1, math_env.n_bins // 2, 0]
    return dict(
        initial_mean=np.array([0.3, -0.2, 0.1]),
        initial_covariance=np.asarray(model.prior_cov()),
        process_covariance=np.asarray(model.drift_cov(0.15)),
        design=np.asarray(model.basis.eigvecs)[indices],
        observations=np.array([1.0, -0.5, 0.8, 1.2, -0.3]),
        observation_variance=np.array([0.2, 0.4, 0.3, 0.25, 0.5]),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "valid", [[True, False, True, True, False], [False, True, True, False, True]]
)
def test_rts_all_moments_match_dense_gaussian_conditioning(gaussian_problem, valid):
    problem = dict(gaussian_problem, valid=np.array(valid))
    reference = gaussian_joint_posterior(**problem)
    filtered_mean, filtered_covariance = gaussian_filtered_moments(**problem)
    mean, covariance, cross = rts_backward_scan(
        filtered_mean, filtered_covariance, np.eye(3), problem["process_covariance"]
    )
    assert (
        np.max(
            np.abs(
                reference.lag_covariance - reference.lag_covariance.transpose(0, 2, 1)
            )
        )
        > 1e-4
    )
    np.testing.assert_allclose(mean, reference.mean, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(covariance, reference.covariance, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(cross, reference.lag_covariance, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("valid", [[True, True], [False, True], [True, False]])
def test_grid_oracle_matches_independent_two_state_gauss_hermite(valid):
    """Validate the oracle itself by integration in independent noise coordinates."""
    mean, variance, q, dt = np.log(3), 0.3, 0.05, 0.2
    design, counts, mask = np.array([1.0, 0.8]), np.array([1.0, 2.0]), np.array(valid)
    reference = poisson_grid_posterior(mean, variance, q, design, counts, mask, dt)
    nodes, weights = hermgauss(96)
    x0 = mean + np.sqrt(2 * variance) * nodes[:, None]
    x1 = x0 + np.sqrt(2 * q) * nodes[None, :]
    states = np.stack([np.broadcast_to(x0, x1.shape), x1], axis=-1)
    log_weight = np.log(weights[:, None] * weights[None, :] / np.pi)
    for t in range(2):
        if mask[t]:
            log_weight += poisson.logpmf(
                counts[t], dt * np.exp(design[t] * states[..., t])
            )
    log_evidence = float(logsumexp(log_weight))
    mass = np.exp(log_weight - log_evidence)
    posterior_mean = np.sum(mass[..., None] * states, axis=(0, 1))
    centered = states - posterior_mean
    posterior_variance = np.sum(mass[..., None] * centered**2, axis=(0, 1))
    lag = np.sum(mass * centered[..., 0] * centered[..., 1])
    np.testing.assert_allclose(reference.smoothed_mean, posterior_mean, atol=1e-8)
    np.testing.assert_allclose(
        reference.smoothed_variance, posterior_variance, atol=1e-8
    )
    np.testing.assert_allclose(reference.lag_covariance, [lag], atol=1e-8)
    assert reference.log_evidence == pytest.approx(log_evidence, abs=1e-9)


@pytest.fixture(scope="module")
def benign_poisson_chain():
    return dict(
        initial_mean=np.log(25.0),
        initial_variance=0.03,
        process_variance=0.006,
        design=np.ones(6),
        counts=np.array([1.0, 0.0, 1.0, 2.0, 0.0, 1.0]),
        valid=np.array([True, True, False, True, False, True]),
        dt=0.02,
    )


@pytest.mark.slow
def test_grid_posterior_is_converged_under_refinement_and_domain_expansion(
    benign_poisson_chain,
):
    fine = poisson_grid_posterior(**benign_poisson_chain, n_grid=2001)
    coarse = poisson_grid_posterior(**benign_poisson_chain, n_grid=1001)
    expanded = poisson_grid_posterior(**benign_poisson_chain, n_grid=2401, radius_sd=12)
    longer_tails = poisson_grid_posterior(
        **benign_poisson_chain, n_grid=2001, transition_radius_sd=16
    )
    for other in (coarse, expanded, longer_tails):
        for field in (
            "filtered_mean",
            "filtered_variance",
            "smoothed_mean",
            "smoothed_variance",
            "lag_covariance",
        ):
            np.testing.assert_allclose(
                getattr(other, field), getattr(fine, field), atol=1e-9, rtol=1e-8
            )
        assert other.log_evidence == pytest.approx(fine.log_evidence, abs=1e-9)
    assert fine.maximum_boundary_mass < 1e-10
    assert fine.maximum_transition_mass_error < 1e-10


@pytest.mark.slow
@pytest.mark.parametrize("newton_steps", [1, 25])
def test_graph_chain_matches_grid_posterior_in_declared_benign_regime(
    math_env, benign_poisson_chain, newton_steps
):
    p = benign_poisson_chain
    model = _legacy_graph_model(
        math_env,
        dt=p["dt"],
        rank=1,
        max_newton_iter=newton_steps,
        update_amplitude=False,
        update_init_mean=False,
    )
    phi = float(model.basis.eigvecs[0, 0])
    model.tau2 = p["initial_variance"] / phi**2
    model.init_mean = jnp.array([[p["initial_mean"] / phi]])
    model.drift_scale = jnp.array([p["process_variance"] / phi**2])
    times = np.arange(6) * p["dt"]
    trajectory = np.repeat(math_env.bin_centers[:1], 6, axis=0)
    trajectory[~p["valid"]] = 1e6
    z, spikes, valid = model._design_and_spikes(times, trajectory, p["counts"])
    log_evidence = model._e_step(z, spikes, valid)
    reference = poisson_grid_posterior(**p, n_grid=2001)
    for mean, covariance, exact_mean, exact_variance in (
        (
            model.filtered_mean[0],
            model.filtered_cov[0],
            reference.filtered_mean,
            reference.filtered_variance,
        ),
        (
            model.smoother_mean[0],
            model.smoother_cov[0],
            reference.smoothed_mean,
            reference.smoothed_variance,
        ),
    ):
        standardized_error = np.abs(
            np.asarray(mean[:, 0]) * phi - exact_mean
        ) / np.sqrt(exact_variance)
        assert standardized_error.max() < 0.1
        np.testing.assert_allclose(
            np.asarray(covariance[:, 0, 0]) * phi**2, exact_variance, rtol=0.05
        )
    np.testing.assert_allclose(
        np.asarray(model.smoother_cross_cov[0, :, 0, 0]) * phi**2,
        reference.lag_covariance,
        rtol=0.06,
        atol=1e-4,
    )
    assert log_evidence == pytest.approx(reference.log_evidence, abs=0.03)
    smoothed_mean = np.asarray(model.smoother_mean[0, :, 0]) * phi
    smoothed_sd = np.sqrt(np.asarray(model.smoother_cov[0, :, 0, 0]) * phi**2)
    masses = interval_mass(
        reference,
        smoothed_mean - 1.96 * smoothed_sd,
        smoothed_mean + 1.96 * smoothed_sd,
    )
    assert np.all((masses > 0.92) & (masses < 0.98))


@pytest.mark.slow
def test_one_step_and_converged_laplace_have_distinct_mathematical_targets():
    """One Fisher step is not required to equal the MAP or posterior mean."""
    count, dt = 8.0, 0.1
    reference = poisson_grid_posterior(
        0.0,
        1.0,
        0.0,
        np.ones(1),
        np.array([count]),
        np.ones(1, dtype=bool),
        dt,
        n_grid=2001,
    )

    def objective(x):
        return float(-poisson.logpmf(count, dt * np.exp(x)) - norm.logpdf(x))

    normalizer = quad(
        lambda x: np.exp(-objective(x)), -12, 12, epsabs=1e-14, epsrel=1e-11
    )[0]
    assert reference.log_evidence == pytest.approx(np.log(normalizer), abs=1e-9)
    mode = minimize_scalar(
        objective, bounds=(-2, 8), method="bounded", options={"xatol": 1e-12}
    ).x
    expected_variance = 1 / (1 + dt * np.exp(mode))
    outcomes = {}
    for iterations in (1, 25):
        mean, covariance, ll = _masked_graph_point_process_filter(
            jnp.zeros(1),
            jnp.eye(1),
            jnp.ones((1, 1)),
            jnp.array([count]),
            jnp.ones(1, dtype=bool),
            jnp.eye(1),
            jnp.zeros((1, 1)),
            dt=dt,
            max_log_count=20.0,
            max_newton_iter=iterations,
        )
        outcomes[iterations] = float(mean[0, 0]), float(covariance[0, 0, 0]), float(ll)
    one_mean, one_variance, one_ll = outcomes[1]
    iterative_mean, iterative_variance, iterative_ll = outcomes[25]
    assert one_mean == pytest.approx((count - dt) / (1 + dt), abs=1e-10)
    assert one_variance == pytest.approx(1 / (1 + dt), abs=1e-10)
    assert iterative_mean == pytest.approx(mode, abs=1e-7)
    assert iterative_variance == pytest.approx(expected_variance, rel=1e-7)
    expected_laplace = -objective(mode) + 0.5 * np.log(2 * np.pi * expected_variance)
    assert iterative_ll == pytest.approx(expected_laplace, abs=1e-8)
    # A guard against pretending all three quantities are identical: the true
    # posterior mean differs from the converged mode in this skewed example.
    assert abs(reference.smoothed_mean[0] - mode) > 0.02
    assert abs(one_mean - reference.smoothed_mean[0]) > 2
    assert abs(iterative_ll - reference.log_evidence) < 0.05
    assert abs(one_ll - reference.log_evidence) > 20


@pytest.mark.slow
def test_default_newton_budget_reaches_unexpected_count_mode(math_env, caplog):
    """The constructor default must fix the documented one-step overshoot."""
    model = _legacy_graph_model(
        math_env,
        dt=0.1,
        rank=1,
        init_drift_scale=0.0,
        max_firing_rate_hz=1e50,
        update_amplitude=False,
        update_init_mean=False,
    )
    phi = float(model.basis.eigvecs[0, 0])
    model.tau2 = 1 / phi**2
    model.init_mean = jnp.zeros((1, 1))
    model.drift_scale = jnp.zeros(1)
    times = np.arange(2) * model.dt
    trajectory = np.repeat(math_env.bin_centers[:1], 2, axis=0)
    trajectory[1] = 1e6
    z, counts, mask = model._design_and_spikes(times, trajectory, [8.0, 1e6])
    with caplog.at_level("WARNING"):
        ll = model._e_step(z, counts, mask)
        jax.effects_barrier()

    def objective(eta):
        return 0.1 * np.exp(eta) - 8 * eta + 0.5 * eta**2

    optimum = minimize_scalar(objective, bounds=(-2, 8), method="bounded").x
    eta = float(model.smoother_mean[0, 0, 0]) * phi
    variance = float(model.smoother_cov[0, 0, 0, 0]) * phi**2
    assert abs(eta - (8 - 0.1) / 1.1) > 2  # the one-step result must fail
    assert eta == pytest.approx(optimum, abs=1e-6)
    assert variance == pytest.approx(1 / (1 + 0.1 * np.exp(optimum)), rel=1e-6)
    assert np.isfinite(ll)
    assert not any(
        "Newton updates did not converge" in r.message for r in caplog.records
    )


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["eager", "jit", "grad"])
@pytest.mark.parametrize("observed", [0, 1])
def test_graph_newton_budget_diagnostic_counts_only_observed_rows(
    mode, observed, caplog
):
    """Accepted steps can still exhaust the budget without line-search failure."""

    def log_evidence(mean):
        return _masked_graph_point_process_filter(
            mean,
            jnp.eye(1),
            jnp.ones((20, 1)),
            jnp.full(20, 8.0),
            jnp.arange(20) < observed,
            jnp.eye(1),
            jnp.zeros((1, 1)),
            dt=0.1,
            max_log_count=20.0,
            max_newton_iter=2,
        )[2]

    function = {
        "eager": log_evidence,
        "jit": jax.jit(log_evidence),
        "grad": jax.grad(log_evidence),
    }[mode]
    with caplog.at_level("WARNING", logger="state_space_practice.graph_place_field"):
        result = function(jnp.zeros(1))
        jax.block_until_ready(result)
        jax.effects_barrier()
    assert np.all(np.isfinite(result))
    messages = [
        r.message
        for r in caplog.records
        if "Newton updates did not converge" in r.message
    ]
    if observed:
        assert any("1/1 observed time bins" in message for message in messages)
        assert not any("line search rejected" in r.message for r in caplog.records)
    else:
        assert not messages


@pytest.mark.slow
def test_multimode_poisson_update_matches_independent_scalar_projection(math_env):
    """A rank-one likelihood permits an independent multi-dimensional oracle.

    For eta=z.T w under a Gaussian prior, w conditional on eta is Gaussian.
    Integrating only eta gives the exact full-state mean/covariance, while
    optimizing eta gives an independent full-state MAP and Laplace normalizer.
    """
    z = np.asarray(_legacy_graph_model(math_env, dt=0.2, rank=3).basis.eigvecs[0])
    mean = np.array([0.3, -0.1, 0.2])
    chol = np.array([[0.8, 0, 0], [0.2, 0.7, 0], [-0.1, 0.15, 0.6]])
    prior = chol @ chol.T
    eta_mean, eta_variance = float(z @ mean), float(z @ prior @ z)
    count, dt = 3.0, 0.2
    exact = poisson_grid_posterior(
        eta_mean,
        eta_variance,
        0.0,
        np.ones(1),
        np.array([count]),
        np.ones(1, dtype=bool),
        dt,
        n_grid=2001,
    )

    def objective(eta):
        return float(
            -poisson.logpmf(count, dt * np.exp(eta))
            - norm.logpdf(eta, eta_mean, np.sqrt(eta_variance))
        )

    radius = 10 * np.sqrt(eta_variance)
    mode = minimize_scalar(
        objective,
        bounds=(eta_mean - radius, eta_mean + radius),
        method="bounded",
        options={"xatol": 1e-12},
    ).x
    gain = prior @ z / eta_variance
    expected_mode = mean + gain * (mode - eta_mean)
    expected_covariance = np.linalg.inv(
        np.linalg.inv(prior) + dt * np.exp(mode) * np.outer(z, z)
    )
    expected_log_evidence = -objective(mode) + 0.5 * np.log(
        2 * np.pi / (1 / eta_variance + dt * np.exp(mode))
    )
    actual_mean, actual_covariance, actual_evidence = (
        _masked_graph_point_process_filter(
            jnp.asarray(mean),
            jnp.asarray(prior),
            jnp.asarray(z[None]),
            jnp.array([count]),
            jnp.ones(1, dtype=bool),
            jnp.eye(3),
            jnp.eye(3) * 0.04,
            dt=dt,
            max_log_count=20.0,
            max_newton_iter=25,
        )
    )
    np.testing.assert_allclose(actual_mean[0], expected_mode, atol=1e-7)
    np.testing.assert_allclose(actual_covariance[0], expected_covariance, atol=1e-8)
    assert float(actual_evidence) == pytest.approx(expected_log_evidence, abs=1e-8)
    exact_mean = mean + gain * (exact.smoothed_mean[0] - eta_mean)
    exact_covariance = (
        prior
        - eta_variance * np.outer(gain, gain)
        + exact.smoothed_variance[0] * np.outer(gain, gain)
    )
    assert np.linalg.norm(np.asarray(actual_mean[0]) - exact_mean) > 1e-3
    # Directions orthogonal to the prior's measurement covariance retain their
    # distribution; this checks covariance structure beyond the scalar variance.
    nuisance = np.cross(prior @ z, np.array([1.0, 0, 0]))
    assert np.linalg.norm(nuisance) > 0.01
    assert float(nuisance @ actual_covariance[0] @ nuisance) == pytest.approx(
        float(nuisance @ prior @ nuisance), abs=1e-10
    )
    assert float(nuisance @ exact_covariance @ nuisance) == pytest.approx(
        float(nuisance @ prior @ nuisance), abs=1e-10
    )


@pytest.mark.parametrize("update_mean", [False, True])
def test_em_updates_maximize_independent_expected_complete_density(
    math_env, gaussian_problem, update_mean
):
    model = _legacy_graph_model(
        math_env,
        dt=0.1,
        rank=3,
        kappa2=0.7,
        tau2=1.3,
        update_drift_scale=True,
        update_init_mean=update_mean,
    )
    model.n_neurons = 2
    means = np.array([[0.3, -0.2, 0.1], [-0.4, 0.6, -0.1]])
    q_initial = np.array([0.15, 0.3])
    shape = (0.7 + model.basis.eigvals) ** -1
    posteriors = [
        gaussian_joint_posterior(
            **dict(
                gaussian_problem,
                initial_mean=means[c],
                process_covariance=np.diag(q_initial[c] * shape),
                observations=gaussian_problem["observations"] + c * 0.8,
                valid=np.array([True, False, True, True, True]),
            )
        )
        for c in range(2)
    ]
    model.init_mean = jnp.asarray(means)
    model.drift_scale = jnp.asarray(q_initial)
    model.smoother_mean = jnp.asarray(np.array([p.mean for p in posteriors]))
    model.smoother_cov = jnp.asarray(np.array([p.covariance for p in posteriors]))
    model.smoother_cross_cov = jnp.asarray(
        np.array([p.lag_covariance for p in posteriors])
    )
    start = (
        np.r_[np.log(q_initial), np.log(1.3), means.ravel()]
        if update_mean
        else np.r_[np.log(q_initial), np.log(1.3)]
    )

    def auxiliary_nll(parameters):
        qs, tau = np.exp(parameters[:2]), np.exp(parameters[2])
        m0 = parameters[3:].reshape(2, 3) if update_mean else means
        return sum(
            expected_gaussian_prior_nll(
                p, m0[c], np.diag(tau * shape), np.diag(qs[c] * shape)
            )
            for c, p in enumerate(posteriors)
        )

    optimum = minimize(
        auxiliary_nll,
        start,
        method="BFGS",
        jac="3-point",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    model._m_step()
    actual = np.r_[np.log(np.asarray(model.drift_scale)), np.log(model.tau2)]
    if update_mean:
        actual = np.r_[actual, np.asarray(model.init_mean).ravel()]
    else:
        np.testing.assert_array_equal(model.init_mean, means)
    assert auxiliary_nll(actual) < auxiliary_nll(start) - 0.01
    assert auxiliary_nll(actual) == pytest.approx(float(optimum.fun), abs=1e-8)
    np.testing.assert_allclose(actual, optimum.x, atol=1e-5, rtol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("newton_steps", [1, 5])
@pytest.mark.parametrize("coordinates", ["log", "sgd"])
def test_graph_hyperparameter_gradients_match_directional_finite_differences(
    math_env, newton_steps, coordinates
):
    model = _legacy_graph_model(
        math_env, dt=0.1, rank=3, max_newton_iter=newton_steps, update_drift_scale=True
    )
    model.n_neurons = 2
    model.tau2, model.kappa2 = 1.2, 0.7
    model.init_mean = jnp.array([[1.0, -0.2, 0.3], [0.4, 0.1, -0.2]])
    model.drift_scale = jnp.array([0.04, 0.08])
    times = np.arange(7) * 0.1
    trajectory = np.asarray(math_env.bin_centers)[np.arange(7) % math_env.n_bins].copy()
    trajectory[[0, 3]] = 1e6
    counts = np.array(
        [[0, 2], [1, 0], [2, 1], [8, 4], [0, 1], [1, 2], [0, 0]], dtype=float
    )
    z, spk, valid = model._design_and_spikes(times, trajectory, counts)
    point = jnp.r_[
        np.log(1.2), np.log(0.7), np.log([0.04, 0.08]), model.init_mean.ravel()
    ]
    constrained, spec = model._build_param_spec()
    if coordinates == "sgd":
        unconstrained = transform_to_unconstrained(constrained, spec)
        point = jnp.r_[
            unconstrained["tau2"],
            unconstrained["kappa2"],
            unconstrained["drift_scale"],
            unconstrained["init_mean"].ravel(),
        ]

    def loss(parameters):
        params = dict(
            tau2=parameters[0],
            kappa2=parameters[1],
            drift_scale=parameters[2:4],
            init_mean=parameters[4:].reshape(2, 3),
        )
        if coordinates == "sgd":
            params = transform_to_constrained(params, spec)
        else:
            params = dict(
                params,
                tau2=jnp.exp(params["tau2"]),
                kappa2=jnp.exp(params["kappa2"]),
                drift_scale=jnp.exp(params["drift_scale"]),
            )
        return model._sgd_loss_fn(params, z, spk, valid)

    gradient = np.asarray(jax.grad(loss)(point))
    for i in range(len(point)):
        direction = np.eye(len(point))[i]
        h = 1e-5
        finite_difference = (
            float(loss(point + h * direction)) - float(loss(point - h * direction))
        ) / (2 * h)
        assert abs(finite_difference) > 1e-7
        assert gradient[i] == pytest.approx(finite_difference, rel=2e-5, abs=1e-7)


@pytest.mark.slow
def test_basis_coordinate_changes_preserve_field_posterior_and_evidence(topology_env):
    env = topology_env
    model = _legacy_graph_model(env, dt=0.1, rank=3)
    model.init_mean = jnp.array([[0.4, -0.1, 0.2]])
    model.drift_scale = jnp.array([0.03])
    times = np.arange(8) * 0.1
    trajectory = np.asarray(env.bin_centers)[np.arange(8) % env.n_bins]
    counts = np.array([0, 1, 0, 2, 1, 0, 0, 1], dtype=float)
    z, spk, valid = model._design_and_spikes(times, trajectory, counts)
    evidence = model._e_step(z, spk, valid)
    field = np.asarray(model.smoother_mean[0]) @ model.basis.eigvecs.T
    phi = np.asarray(model.basis.eigvecs)
    variance = np.einsum("br,trs,bs->tb", phi, np.asarray(model.smoother_cov[0]), phi)
    change = np.diag([-1.0, 1.0, -1.0])
    if model.basis.n_components == 2:
        angle = 0.6
        change = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    assert not np.allclose(change, np.eye(3))
    model.basis = model.basis._replace(eigvecs=phi @ change)
    model.init_mean = model.init_mean @ change
    z, spk, valid = model._design_and_spikes(times, trajectory, counts)
    changed_evidence = model._e_step(z, spk, valid)
    changed_phi = np.asarray(model.basis.eigvecs)
    changed_variance = np.einsum(
        "br,trs,bs->tb", changed_phi, np.asarray(model.smoother_cov[0]), changed_phi
    )
    np.testing.assert_allclose(
        np.asarray(model.smoother_mean[0]) @ changed_phi.T, field, atol=1e-10
    )
    np.testing.assert_allclose(changed_variance, variance, atol=1e-10)
    assert changed_evidence == pytest.approx(evidence, abs=1e-10)


@pytest.mark.slow
def test_process_variance_scales_per_step_to_preserve_physical_time(math_env):
    results = []
    for dt, n_time in ((0.05, 9), (0.1, 5)):
        model = _legacy_graph_model(math_env, dt=dt, rank=3, init_drift_scale=0.04 * dt)
        covariance = _masked_graph_point_process_filter(
            jnp.zeros(3),
            model.prior_cov(),
            jnp.zeros((n_time, 3)),
            jnp.zeros(n_time),
            jnp.zeros(n_time, dtype=bool),
            jnp.eye(3),
            model.drift_cov(model.init_drift_scale),
            dt=dt,
            max_log_count=20.0,
            max_newton_iter=1,
        )[1]
        results.append(np.asarray(covariance))
    np.testing.assert_allclose(results[0][::2], results[1], atol=1e-12, rtol=1e-12)
    assert np.trace(results[1][-1]) > np.trace(results[1][0])
