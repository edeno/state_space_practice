"""Snapshot a model's instance state and assert a call left it unchanged.

Used by the tests that check a rejected call (e.g. ``fit_sgd`` with invalid
settings) raises before mutating the model.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import jax
import networkx as nx
import numpy as np
import scipy.sparse as sp


def _snapshot(value: Any) -> Any:
    """Deep copy of ``value`` with arrays, dataclasses and containers unpacked."""
    if isinstance(value, jax.Array) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    ):  # typed PRNG keys cannot become NumPy arrays; compare their key data
        return ("prng_key", np.array(jax.random.key_data(value)))
    if isinstance(value, (jax.Array, np.ndarray)):
        return np.array(value)
    if sp.issparse(value):
        return (type(value), value.toarray())
    if isinstance(value, nx.Graph):
        return (
            type(value),
            _snapshot(value.graph),
            _snapshot(dict(value.nodes(data=True))),
            _snapshot(nx.to_dict_of_dicts(value)),
        )
    if type(value).__module__.startswith("neurospatial.layout"):
        return (type(value), _snapshot(vars(value)))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            {
                f.name: _snapshot(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if hasattr(value, f.name)
            },
        )
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # NamedTuple
        return (type(value), {f: _snapshot(getattr(value, f)) for f in value._fields})
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(v) for v in value)
    return copy.deepcopy(value)


def _assert_same_snapshot(before: Any, after: Any, path: str) -> None:
    """Assert two ``_snapshot`` results are identical (NaN equals NaN)."""
    if isinstance(before, np.ndarray):
        assert isinstance(after, np.ndarray), path
        assert (before.shape, before.dtype) == (after.shape, after.dtype), path
        equal_nan = np.issubdtype(before.dtype, np.inexact)
        assert np.array_equal(before, after, equal_nan=equal_nan), path
    elif isinstance(before, dict):
        assert isinstance(after, dict), path
        assert before.keys() == after.keys(), (path, before.keys() ^ after.keys())
        for key in before:
            _assert_same_snapshot(before[key], after[key], f"{path}.{key}")
    elif isinstance(before, (list, tuple)):
        assert type(after) is type(before) and len(after) == len(before), path
        for i, (b, a) in enumerate(zip(before, after, strict=True)):
            _assert_same_snapshot(b, a, f"{path}[{i}]")
    else:
        assert before == after, path


def snapshot_model_state(model: object) -> Any:
    """Deep copy of every instance attribute of ``model``, fitted state included."""
    return _snapshot(vars(model))


def assert_model_state_unchanged(model: object, before: Any) -> None:
    """Every instance attribute of ``model`` still equals its snapshot."""
    _assert_same_snapshot(before, _snapshot(vars(model)), type(model).__name__)


def assert_snapshots_equal(first: Any, second: Any) -> None:
    """Two :func:`snapshot_model_state` results are identical."""
    _assert_same_snapshot(first, second, "model")
