"""Shared constructors for the graph place-field test modules."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from state_space_practice.graph_place_field import (
    GraphPlaceFieldModel,
    _summarize_posterior,
)


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


def set_smoothed_posterior(
    model: GraphPlaceFieldModel,
    mean: ArrayLike,
    covariance: ArrayLike,
    cross_covariance: ArrayLike,
) -> None:
    """Set hand-built smoothed moments and the M-step summaries derived from them.

    ``mean`` ``(n_neurons, n_time, rank)``, ``covariance`` ``(n_neurons, n_time,
    rank, rank)``, ``cross_covariance`` ``(n_neurons, n_time - 1, rank, rank)``.
    """
    model.smoother_mean = jnp.asarray(mean)
    model.smoother_cov = jnp.asarray(covariance)
    model.smoother_cross_cov = jnp.asarray(cross_covariance)
    model._posterior_summary = jax.vmap(_summarize_posterior, in_axes=(0, 0, 0, None))(
        model.smoother_mean,
        model.smoother_cov,
        model.smoother_cross_cov,
        model.transition_matrix,
    )
