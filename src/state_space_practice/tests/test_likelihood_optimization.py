"""Behavioral checks for the reusable likelihood optimizers."""

import numpy as np
import pytest

from state_space_practice.likelihood_optimization import (
    minimize_likelihood,
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
