"""Tests for the FittedAttribute descriptor."""

import copy
import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from state_space_practice.exceptions import NotFittedError
from state_space_practice.fitted_state import (
    FittedAttribute,
    UnfittedAttributeError,
    is_set,
)


class _Model:
    smoother_mean: FittedAttribute[Array] = FittedAttribute()
    log_likelihood_: FittedAttribute[float] = FittedAttribute()

    def fit(self) -> None:
        self.smoother_mean = jnp.arange(3.0)
        self.log_likelihood_ = -1.5


@pytest.fixture
def fitted() -> _Model:
    model = _Model()
    model.fit()
    return model


def test_unset_read_raises_not_fitted_with_attribute_name() -> None:
    with pytest.raises(NotFittedError, match=r"_Model\.smoother_mean"):
        _ = _Model().smoother_mean


def test_unset_attribute_is_also_attribute_error() -> None:
    model = _Model()
    assert issubclass(UnfittedAttributeError, AttributeError)
    assert not hasattr(model, "smoother_mean")
    assert getattr(model, "smoother_mean", "default") == "default"
    assert not is_set(model, "smoother_mean")


def test_set_value_round_trips(fitted: _Model) -> None:
    np.testing.assert_array_equal(fitted.smoother_mean, [0.0, 1.0, 2.0])
    assert fitted.log_likelihood_ == -1.5
    assert is_set(fitted, "smoother_mean")


def test_delete_returns_to_unset(fitted: _Model) -> None:
    del fitted.smoother_mean
    with pytest.raises(NotFittedError):
        _ = fitted.smoother_mean
    # Deleting an unset attribute is a no-op rather than an error.
    del fitted.smoother_mean


def test_instances_do_not_share_state(fitted: _Model) -> None:
    assert not is_set(_Model(), "smoother_mean")
    assert is_set(fitted, "smoother_mean")


def test_copy_deepcopy_and_pickle_preserve_values(fitted: _Model) -> None:
    for clone in (
        copy.copy(fitted),
        copy.deepcopy(fitted),
        pickle.loads(pickle.dumps(fitted)),
    ):
        np.testing.assert_array_equal(clone.smoother_mean, fitted.smoother_mean)
        assert clone.log_likelihood_ == fitted.log_likelihood_
    assert not is_set(copy.deepcopy(_Model()), "smoother_mean")


def test_value_stored_under_public_name(fitted: _Model) -> None:
    # Plain __dict__ storage keeps vars() and older pickles compatible.
    assert set(vars(fitted)) == {"smoother_mean", "log_likelihood_"}


def test_class_access_returns_descriptor() -> None:
    assert isinstance(_Model.__dict__["smoother_mean"], FittedAttribute)
    assert isinstance(_Model.smoother_mean, FittedAttribute)
