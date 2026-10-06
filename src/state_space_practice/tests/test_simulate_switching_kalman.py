"""Tests for the simulate_switching_kalman module's random-state handling."""

import importlib

import numpy as np

from state_space_practice.simulate import simulate_switching_kalman
from state_space_practice.simulate.simulate_switching_kalman import (
    simdata_settings,
    simulate_model,
)


def _global_rng_state() -> tuple:
    return np.random.get_state(legacy=True)


def _assert_same_rng_state(a: tuple, b: tuple) -> None:
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2:] == b[2:]


def test_import_does_not_reseed_global_numpy_rng() -> None:
    np.random.seed(123)
    np.random.random(5)  # move away from a freshly seeded state
    before = _global_rng_state()
    importlib.reload(simulate_switching_kalman)
    _assert_same_rng_state(before, _global_rng_state())


def test_simulation_does_not_touch_global_numpy_rng() -> None:
    np.random.seed(7)
    before = _global_rng_state()
    simdata_settings()
    simulate_model(T=50)
    _assert_same_rng_state(before, _global_rng_state())


def test_simulate_model_is_deterministic_given_its_arguments() -> None:
    """Consecutive calls (and an unrelated draw from the global RNG in
    between) give the same initial state and data."""
    first = simulate_model(T=200, blnSimS=True)
    np.random.random(10)
    second = simulate_model(T=200, blnSimS=True)
    # X0, s, y, x
    for idx in (13, 16, 17, 18):
        np.testing.assert_array_equal(first[idx], second[idx])


def test_seeds_select_the_realization() -> None:
    base = simulate_model(T=200)
    other_init = simulate_model(T=200, init_seed=1)
    other_noise = simulate_model(T=200, noise_seed=15)
    assert not np.array_equal(base[13], other_init[13])  # X0
    np.testing.assert_array_equal(base[13], other_noise[13])
    assert not np.array_equal(base[17], other_noise[17])  # y


def test_none_seed_draws_fresh_initial_states() -> None:
    x0_a = simdata_settings(init_seed=None)[13]
    x0_b = simdata_settings(init_seed=None)[13]
    assert not np.array_equal(x0_a, x0_b)
