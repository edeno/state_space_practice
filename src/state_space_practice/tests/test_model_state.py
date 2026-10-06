"""Self-tests for the ``model_state`` snapshot helpers used by the fit tests."""

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.tests.model_state import (
    assert_model_state_unchanged,
    snapshot_model_state,
)


class _Posterior(NamedTuple):
    mean: Any
    label: str


class _Model:
    """Stand-in model holding each kind of state the snapshot unpacks."""

    def __init__(self) -> None:
        self.weights = jnp.arange(3.0)
        self.settings = {"inner": {"scale": np.array([1.0, np.nan])}, "n": 2}
        self.posterior = _Posterior(mean=jnp.ones(2), label="a")
        self.name = "model"


@pytest.fixture
def model() -> _Model:
    return _Model()


def test_untouched_model_passes(model: _Model) -> None:
    before = snapshot_model_state(model)

    # Rebinding attributes to equal copies is not a change (NaN equals NaN).
    model.weights = jnp.arange(3.0)
    model.settings = {"inner": {"scale": np.array([1.0, np.nan])}, "n": 2}

    assert_model_state_unchanged(model, before)


def _set_array_value(model: _Model) -> None:
    model.weights = model.weights.at[0].set(9.0)


def _mutate_nested_dict_in_place(model: _Model) -> None:
    # In place: the snapshot must hold a copy, not the live array.
    model.settings["inner"]["scale"][0] = 2.0


def _replace_namedtuple_field(model: _Model) -> None:
    model.posterior = model.posterior._replace(mean=jnp.zeros(2))


def _add_attribute(model: _Model) -> None:
    model.extra = 1


def _delete_attribute(model: _Model) -> None:
    del model.name


@pytest.mark.parametrize(
    "mutate",
    [
        _set_array_value,
        _mutate_nested_dict_in_place,
        _replace_namedtuple_field,
        _add_attribute,
        _delete_attribute,
    ],
)
def test_every_change_is_detected(model: _Model, mutate) -> None:
    before = snapshot_model_state(model)

    mutate(model)

    with pytest.raises(AssertionError):
        assert_model_state_unchanged(model, before)
