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
