"""Shared constructors for the graph place-field test modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from state_space_practice.graph_place_field import GraphPlaceFieldModel


def legacy_graph_model(*args: Any, **kwargs: Any) -> GraphPlaceFieldModel:
    """Keep the original sequential/EM regression targets explicit.

    Automatic constructor defaults are tested in test_graph_place_field_learning.
    """
    kwargs.setdefault("inference_method", "sequential")
    kwargs.setdefault("update_drift_scale", False)
    return GraphPlaceFieldModel(*args, **kwargs)


def line_env() -> Any:
    """Six-bin 1-D track on [0, 10] with 2-unit bins."""
    from neurospatial import Environment

    return Environment.from_samples(np.linspace(0, 10, 201)[:, None], bin_size=2.0)
