"""Joint-mode/evidence contracts against an independent dense Newton oracle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice import laplace_smoothing
from state_space_practice.laplace_smoothing import (
    _iterate_mode,
    poisson_laplace_smoother,
)
from state_space_practice.tests.graph_math_reference import joint_poisson_laplace


@pytest.fixture
def joint_problem():
    return dict(
        mean0=jnp.array([0.3, -0.2]),
        covariance0=jnp.array([[1.0, 0.2], [0.2, 0.6]]),
        transition=jnp.array([[0.9, 0.05], [0.0, 0.8]]),
        design=jnp.array([[1.0, 0.4], [0.2, 1.0], [1.0, -0.3], [0.7, 0.9], [0.1, 1.0]]),
        counts=jnp.array([8.0, 1e6, 0.0, 2.0, 1.0]),
        valid=jnp.array([True, False, True, True, True]),
        dt=0.1,
    )


@pytest.mark.parametrize("q", [0.0, 0.03, 1e-8])
def test_joint_mode_curvature_lag_and_evidence_match_dense_reference(joint_problem, q):
    args = dict(joint_problem, noise=q * jnp.eye(2))
    result = poisson_laplace_smoother(**args)
    reference = joint_poisson_laplace(
        np.asarray(args["mean0"]),
        np.asarray(args["covariance0"]),
        np.asarray(args["noise"]),
        np.asarray(args["design"]),
        np.asarray(args["counts"]),
        np.asarray(args["valid"]),
        args["dt"],
        np.asarray(args["transition"]),
    )
    assert result.relative_newton_step < 1e-8
    assert result.n_rejected_steps == 0
    assert bool(result.converged)
    np.testing.assert_allclose(result.mean, reference.mean, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(
        result.covariance, reference.covariance, atol=1e-7, rtol=1e-7
    )
    np.testing.assert_allclose(
        result.cross_covariance, reference.cross_covariance, atol=1e-7, rtol=1e-7
    )
    assert float(result.log_evidence) == pytest.approx(reference.log_evidence, abs=1e-7)


@pytest.mark.slow
@pytest.mark.parametrize("q", [0.0, 0.03])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_explicit_float32_inputs_with_x64_enabled(joint_problem, q, mixed_precision):
    args = {
        name: value.astype(jnp.float32)
        if isinstance(value, jax.Array) and name != "valid"
        else value
        for name, value in joint_problem.items()
    }
    args["noise"] = q * jnp.eye(2, dtype=jnp.float32)
    if mixed_precision:
        args["covariance0"] = args["covariance0"].astype(jnp.float64)
    expected_dtype = jnp.float64 if mixed_precision else jnp.float32
    tolerance = 1e-8 if mixed_precision else 1e-4
    result = poisson_laplace_smoother(**args, tolerance=tolerance)
    reference = joint_poisson_laplace(
        np.asarray(args["mean0"]),
        np.asarray(args["covariance0"]),
        np.asarray(args["noise"]),
        np.asarray(args["design"]),
        np.asarray(args["counts"]),
        np.asarray(args["valid"]),
        args["dt"],
        np.asarray(args["transition"]),
    )
    assert result.relative_newton_step <= tolerance
    assert result.n_rejected_steps == 0
    for name in ("mean", "covariance", "cross_covariance", "log_evidence"):
        actual = getattr(result, name)
        assert actual.dtype == expected_dtype
        np.testing.assert_allclose(
            actual, getattr(reference, name), atol=5e-5, rtol=5e-5
        )
    assert result.relative_newton_step.dtype == expected_dtype


@pytest.mark.slow
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_implicit_evidence_gradients_match_finite_differences(joint_problem, q):
    def loss(theta):
        args = dict(
            joint_problem,
            mean0=joint_problem["mean0"] + theta[0],
            covariance0=jnp.exp(theta[1]) * joint_problem["covariance0"],
            noise=q * jnp.exp(theta[2]) * jnp.eye(2),
        )
        return poisson_laplace_smoother(**args).log_evidence

    point = jnp.array([0.0, 0.0, 0.0])
    gradient = np.asarray(jax.grad(loss)(point))
    for i in range(3):
        direction = np.eye(3)[i] * 1e-5
        fd = (float(loss(point + direction)) - float(loss(point - direction))) / 2e-5
        assert gradient[i] == pytest.approx(fd, rel=2e-5, abs=1e-7)
    assert np.linalg.norm(gradient) > 1


@pytest.mark.slow
def test_static_evidence_is_order_invariant_and_continuous_at_zero():
    def infer(y, q):
        return poisson_laplace_smoother(
            jnp.zeros(1),
            jnp.eye(1),
            jnp.eye(1),
            q * jnp.eye(1),
            jnp.ones((2, 1)),
            jnp.asarray(y),
            jnp.ones(2, dtype=bool),
            dt=0.1,
        )

    first, second = infer([8.0, 0.0], 0.0), infer([0.0, 8.0], 0.0)
    assert float(first.log_evidence) == pytest.approx(
        float(second.log_evidence), abs=1e-10
    )
    np.testing.assert_allclose(first.mean, second.mean, atol=1e-10)
    assert float(infer([8.0, 0.0], 1e-8).log_evidence) == pytest.approx(
        float(first.log_evidence), abs=1e-7
    )


@pytest.mark.slow
def test_mixed_static_dynamic_vmap_gradients_match_unbatched(joint_problem):
    """Batching static and dynamic neurons must not change either gradient."""

    def evidence(q, shift):
        args = dict(
            joint_problem,
            mean0=joint_problem["mean0"] + shift,
            noise=q * jnp.eye(2),
        )
        return poisson_laplace_smoother(**args).log_evidence

    scales = jnp.array([0.0, 0.03])
    shifts = jnp.array([0.1, -0.2])
    batched_q, batched_shift = jax.grad(
        lambda q, m: jax.vmap(evidence)(q, m).sum(), argnums=(0, 1)
    )(scales, shifts)
    for i in range(2):
        q_grad, shift_grad = jax.grad(evidence, argnums=(0, 1))(scales[i], shifts[i])
        assert float(batched_q[i]) == pytest.approx(float(q_grad), rel=1e-10, abs=1e-12)
        assert float(batched_shift[i]) == pytest.approx(
            float(shift_grad), rel=1e-10, abs=1e-12
        )
        # Guard: both the static and the dynamic neuron carry real gradients.
        assert abs(float(shift_grad)) > 1e-3
    assert abs(float(batched_q[1])) > 1e-3


def test_missing_observations_leave_full_markov_prior(joint_problem):
    args = dict(joint_problem, noise=0.03 * jnp.eye(2), valid=jnp.zeros(5, dtype=bool))
    result = poisson_laplace_smoother(**args)
    mean, covariance = np.asarray(args["mean0"]), np.asarray(args["covariance0"])
    for t in range(5):
        np.testing.assert_allclose(result.mean[t], mean, atol=1e-10)
        np.testing.assert_allclose(result.covariance[t], covariance, atol=1e-10)
        mean = np.asarray(args["transition"]) @ mean
        covariance = np.asarray(args["transition"]) @ covariance @ np.asarray(
            args["transition"]
        ).T + 0.03 * np.eye(2)
    assert float(result.log_evidence) == 0


@pytest.mark.slow
def test_iteration_exhaustion_is_reported(joint_problem):
    result = poisson_laplace_smoother(
        **dict(joint_problem, noise=0.03 * jnp.eye(2)), max_iter=1
    )
    assert result.n_iterations == 1
    assert result.n_rejected_steps == 0
    assert result.relative_newton_step > 1e-4
    assert not bool(result.converged)


def test_iterate_mode_stops_on_fully_rejected_line_search():
    """A line search with no usable trial keeps the path and stops iterating."""
    initial = jnp.zeros(2)

    def target(path):
        # Newton target of the concave merit -|path - 1|^2 below.
        return jnp.ones_like(path)

    def unusable_merit(path):
        # Finite only at the current point; every trial step (>= 0.5**19) is
        # non-finite, so no Armijo step can be accepted.
        finite = -jnp.sum((path - 1.0) ** 2)
        return jnp.where(jnp.max(jnp.abs(path)) < 1e-9, finite, jnp.nan)

    path, step, n_iterations, n_rejected = _iterate_mode(
        initial, target, unusable_merit, max_iter=50, tolerance=1e-8
    )
    np.testing.assert_array_equal(path, initial)
    assert int(n_rejected) == 1
    assert int(n_iterations) == 1  # stopped by the rejection, not by max_iter
    assert float(step) == pytest.approx(1.0)  # the unaccepted Newton step remains

    # Guard: with a finite merit the same Newton step is accepted and converges.
    path, step, n_iterations, n_rejected = _iterate_mode(
        initial, target, lambda p: -jnp.sum((p - 1.0) ** 2), max_iter=50, tolerance=1e-8
    )
    assert int(n_rejected) == 0
    np.testing.assert_allclose(path, jnp.ones(2))
    assert float(step) == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_rejected_line_search_is_never_reported_converged(
    joint_problem, q, monkeypatch
):
    """A rejection makes the result unconverged even at a small Newton step."""
    iterate = laplace_smoothing._iterate_mode
    calls = []

    def iterate_with_one_rejection(*args, **kwargs):
        path, step, n, rejected = iterate(*args, **kwargs)
        calls.append(None)
        return path, step, n, rejected + 1

    monkeypatch.setattr(laplace_smoothing, "_iterate_mode", iterate_with_one_rejection)
    # The unjitted body re-traces with the patched iteration; the compiled
    # cache is neither used nor polluted.
    result = poisson_laplace_smoother.__wrapped__(
        **dict(joint_problem, noise=q * jnp.eye(2))
    )
    assert calls  # guard: the patched iteration was used
    assert float(result.relative_newton_step) < 1e-8  # the mode itself converged
    assert int(result.n_rejected_steps) == 1
    assert not bool(result.converged)


@pytest.mark.slow
def test_joint_core_agrees_with_temporal_rate_gp():
    """The common Poisson core agrees with the existing iterated GP machinery."""
    from state_space_practice.gp_ssm import matern32_continuous, matern32_discretize
    from state_space_practice.temporal_rate_gp import infer_log_rate

    counts = jnp.array([0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 2.0])
    dt, variance, lengthscale = 0.1, 0.8, 0.4
    _f, _l, _q, h, p = matern32_continuous(variance, lengthscale)
    a, q = matern32_discretize(variance, lengthscale, dt)
    common = poisson_laplace_smoother(
        jnp.zeros(2),
        p,
        a,
        q,
        jnp.broadcast_to(h, (len(counts), 2)),
        counts,
        jnp.ones(len(counts), dtype=bool),
        dt=dt,
    )
    existing = infer_log_rate(counts, dt, variance, lengthscale, mean=0.0, n_iter=60)
    assert common.relative_newton_step < 1e-8
    np.testing.assert_allclose(common.mean @ h, existing.log_rate_mean, atol=1e-7)
    np.testing.assert_allclose(
        jnp.einsum("r,trs,s->t", h, common.covariance, h),
        existing.log_rate_var,
        atol=1e-7,
    )
    assert float(common.log_evidence) == pytest.approx(
        float(existing.log_marginal_likelihood), abs=1e-7
    )
