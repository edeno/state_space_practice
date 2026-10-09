"""Behavioral checks for the reusable likelihood optimizers."""

import numpy as np
import pytest

from state_space_practice.likelihood_optimization import (
    minimize_likelihood,
    profile_grid,
    profile_scale,
)


def test_profile_includes_exact_static_limit():
    result = profile_scale(lambda q: np.log1p(q) + q**2)
    assert result.scale == 0
    assert result.loss == 0
    assert result.loss < np.min(result.losses[result.scales > 0])


def test_profile_compares_multiple_basins_and_endpoints():
    def objective(q):
        if not q:
            return 2.0
        x = np.log(q)
        return float(
            -3 * np.exp(-((x + 3) ** 2) / 0.2) - 7 * np.exp(-((x - 1) ** 2) / 0.2)
        )

    result = profile_scale(objective, bounds=(0.001, 100))
    assert np.log(result.scale) == pytest.approx(1, abs=1e-4)
    assert result.loss < -6.99
    endpoint = profile_scale(lambda q: -q, bounds=(0.001, 100))
    assert endpoint.scale == 100
    assert endpoint.at_upper_bound


def test_flat_profile_is_reported_and_prefers_static():
    result = profile_scale(lambda q: 3.0)
    assert result.flat
    assert result.scale == 0


@pytest.mark.parametrize("unusable", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    ("finite_interval", "minimum"),
    [((0.6, 2.3), 1.4), ((1.7, 3.4), 2.6), ((1.6, 2.3), 1.8)],
)
def test_profile_refines_basin_next_to_unusable_samples(
    unusable, finite_interval, minimum
):
    def objective(q):
        if q == 0:
            return 0.1
        x = np.log(q)
        if finite_interval[0] < x < finite_interval[1]:
            return float((x - minimum) ** 2)
        return unusable

    result = profile_scale(objective, bounds=(1, np.exp(4)), n_grid=5)
    assert np.log(result.scale) == pytest.approx(minimum, abs=1e-5)
    assert result.loss == pytest.approx(0, abs=1e-10)
    # Diagnostics retain the caller's raw failures rather than the search penalty.
    np.testing.assert_equal(
        result.losses[:6], [objective(q) for q in result.scales[:6]]
    )


@pytest.mark.parametrize("unusable", [np.nan, np.inf, -np.inf])
def test_profile_ignores_unusable_bounded_refinement_trials(unusable):
    unusable_trials = []

    def objective(q):
        if q == 0:
            return 0.1
        x = np.log(q)
        if 0.2 < x < 0.9:
            unusable_trials.append(q)
            return unusable
        return float((x - 1.4) ** 2)

    result = profile_scale(objective, bounds=(1, np.exp(4)), n_grid=5)
    assert unusable_trials  # Every initial grid point is finite.
    assert np.log(result.scale) == pytest.approx(1.4, abs=1e-5)
    assert result.loss == pytest.approx(0, abs=1e-10)


def test_profile_refinement_preserves_small_improvements_at_large_loss():
    offset = 1e8

    def objective(q):
        return offset + (0.1 if q == 0 else (np.log(q) - 1.4) ** 2)

    result = profile_scale(objective, bounds=(1, np.exp(4)), n_grid=5)
    assert np.log(result.scale) == pytest.approx(1.4, abs=2e-4)
    assert result.loss == pytest.approx(offset, rel=0, abs=1e-7)


@pytest.mark.parametrize("unusable", [np.nan, np.inf, -np.inf])
def test_profile_rejects_entirely_unusable_search(unusable):
    with pytest.raises(ValueError, match="No finite likelihood"):
        profile_scale(lambda q: unusable)


def test_bounded_optimizer_checks_projected_gradient():
    result = minimize_likelihood(
        lambda x: (float((x[0] - 2) ** 2), np.array([2 * (x[0] - 2)])),
        np.array([0.5]),
        bounds=[(0, 1)],
    )
    assert result.parameters[0] == 1
    assert result.loss == 1
    assert result.converged
    assert result.gradient_norm == 0


def test_budget_exhaustion_is_not_likelihood_convergence():
    def objective(x):
        a, b = x
        return (
            float(100 * (b - a * a) ** 2 + (1 - a) ** 2),
            np.array([-400 * a * (b - a * a) - 2 * (1 - a), 200 * (b - a * a)]),
        )

    result = minimize_likelihood(objective, np.array([-1.0, 1.0]), max_iter=1)
    assert result.loss < objective(np.array([-1.0, 1.0]))[0]
    assert result.gradient_norm > 1e-3
    assert not result.converged


def test_profile_counts_unusable_evaluations():
    def objective(q):
        return np.nan if q > 1e-3 else (q - 1e-3) ** 2 + 1

    result = profile_scale(objective, bounds=(1e-8, 10.0), n_grid=9)
    assert result.n_nonfinite > 0
    assert result.n_nonfinite == int(np.sum(result.scales > 1e-3))
    assert profile_scale(lambda q: q**2 + 1).n_nonfinite == 0


def test_optimizer_history_records_accepted_iterates_only():
    """Line-search trials, including unusable ones, are not likelihood history."""

    def objective(x):
        if x[0] > 2.5:
            return np.inf, np.array([2 * (x[0] - 3)])
        return float((x[0] - 3) ** 2), np.array([2 * (x[0] - 3)])

    result = minimize_likelihood(objective, np.array([0.0]))
    history = np.asarray(result.loss_history)
    assert np.all(np.isfinite(history))
    assert np.all(np.diff(history) <= 0)
    assert len(history) == result.n_iter + 1
    assert history[0] == 9.0
    assert history[-1] == result.loss


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(bounds=(1.0, 1.0)), "increasing"),
        (dict(bounds=(10.0, 1.0)), "increasing"),
        (dict(bounds=(0.0, 1.0)), "lower scale bound must be positive"),
        (dict(bounds=(-1.0, 1.0)), "lower scale bound must be positive"),
        (dict(bounds=(np.nan, 1.0)), "lower scale bound must be finite"),
        (dict(bounds=(1e-3, np.inf)), "upper scale bound must be finite"),
        (dict(n_grid=2), "at least three"),
        (dict(n_grid=0), "n_grid must be a positive integer"),
        (dict(n_grid=5.0), "n_grid must be a positive integer"),
        (dict(objective_tolerance=-1.0), "objective_tolerance must be non-negative"),
        (dict(objective_tolerance=np.nan), "objective_tolerance must be finite"),
    ],
)
def test_profile_rejects_invalid_search_settings(kwargs, match):
    calls = []

    def objective(q):
        calls.append(q)
        return (q - 1.0) ** 2

    with pytest.raises(ValueError, match=match):
        profile_scale(objective, **kwargs)
    assert calls == []  # rejected before any likelihood evaluation


@pytest.mark.parametrize(
    ("initial", "kwargs", "match"),
    [
        (np.array([np.nan]), {}, "finite nonempty vector"),
        (np.array([np.inf, 0.0]), {}, "finite nonempty vector"),
        (np.array([]), {}, "finite nonempty vector"),
        (np.zeros((1, 1)), {}, "finite nonempty vector"),
        (np.zeros(1), dict(max_iter=0), "max_iter must be a positive integer"),
        (np.zeros(1), dict(max_iter=2.0), "max_iter must be a positive integer"),
        (np.zeros(1), dict(gradient_tolerance=0.0), "gradient_tolerance must be"),
        (np.zeros(1), dict(gradient_tolerance=np.nan), "gradient_tolerance must be"),
    ],
)
def test_optimizer_rejects_invalid_start_and_settings(initial, kwargs, match):
    calls = []

    def objective(x):
        calls.append(x)
        return float(np.sum(x**2)), 2 * np.asarray(x, dtype=float)

    with pytest.raises(ValueError, match=match):
        minimize_likelihood(objective, initial, **kwargs)
    assert calls == []


def test_bounded_optimizer_projects_gradient_at_lower_bound():
    # The unconstrained minimum (-2) lies below the box, so the solution is the
    # lower bound with a strictly positive raw gradient pushing outward.
    result = minimize_likelihood(
        lambda x: (float((x[0] + 2) ** 2), np.array([2 * (x[0] + 2)])),
        np.array([0.5]),
        bounds=[(0, 1)],
    )
    assert result.parameters[0] == 0
    assert result.loss == 4
    assert result.converged
    assert result.gradient_norm == 0  # the raw gradient (4) is projected away


def test_profile_uses_supplied_grid_losses_without_reevaluating():
    def objective(q):
        return float((np.log(q + 1e-3) + 2) ** 2)

    reference = profile_scale(objective, n_grid=9)
    _, points = profile_grid((1e-8, 10.0), 9)
    calls = []

    def counted(q):
        calls.append(q)
        return objective(q)

    result = profile_scale(
        counted, n_grid=9, grid_losses=[objective(q) for q in points]
    )
    assert calls  # refinement still evaluates the objective
    assert not set(calls) & set(points.tolist())
    assert result.scale == reference.scale
    np.testing.assert_array_equal(result.scales, reference.scales)
    np.testing.assert_array_equal(result.losses, reference.losses)


def test_profile_rejects_misaligned_grid_losses():
    with pytest.raises(ValueError, match="grid_losses"):
        profile_scale(lambda q: q, n_grid=9, grid_losses=np.zeros(9))
