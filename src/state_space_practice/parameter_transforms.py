"""Lightweight parameter constraint transforms for SGD optimization.

Maps between constrained parameter spaces (positive reals, unit interval,
PSD matrices) and unconstrained reals for gradient-based optimization.
No TFP dependency -- pure JAX.

Usage::

    unc_params = transform_to_unconstrained(params, param_spec)
    # ... optimize unc_params with optax ...
    params = transform_to_constrained(unc_params, param_spec)
"""

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class ParameterTransform(NamedTuple):
    """Maps between constrained and unconstrained parameter spaces."""

    to_unconstrained: Callable[[Array], Array]
    to_constrained: Callable[[Array], Array]


def _inverse_softplus(x: Array) -> Array:
    """Inverse of softplus: log(exp(x) - 1), numerically stable."""
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(x)))


# softplus chosen over exp to prevent overflow for large unconstrained values.
# softplus(x) = log(1 + exp(x)) grows linearly, not exponentially.
POSITIVE = ParameterTransform(
    to_unconstrained=_inverse_softplus,
    to_constrained=jax.nn.softplus,
)

UNIT_INTERVAL = ParameterTransform(
    to_unconstrained=lambda x: jnp.log(x / (1 - x)),  # logit
    to_constrained=jax.nn.sigmoid,
)

UNCONSTRAINED = ParameterTransform(
    to_unconstrained=lambda x: x,
    to_constrained=lambda x: x,
)


def _psd_to_real(P: Array) -> Array:
    """Flatten a PSD matrix to unconstrained reals via Cholesky + log-diagonal.

    A small jitter (1e-9) is added before Cholesky to prevent NaN on
    near-singular matrices during optimization. This introduces a bounded
    roundtrip error of ~1e-9.
    """
    # Add jitter for numerical stability on near-singular matrices
    n = P.shape[0]
    L = jnp.linalg.cholesky(P + 1e-9 * jnp.eye(n))
    # Replace diagonal with log(diagonal) so it's unconstrained
    L = L.at[jnp.diag_indices_from(L)].set(jnp.log(jnp.diag(L)))
    return L[jnp.tril_indices_from(L)]


def _real_to_psd(flat: Array) -> Array:
    """Reconstruct a PSD matrix from unconstrained reals."""
    # Solve n*(n+1)/2 = len(flat) for n
    # Use math.sqrt (not jnp.sqrt) so n is a concrete int under jax.jit
    n = int((-1 + math.sqrt(1 + 8 * flat.shape[0])) / 2)
    L = jnp.zeros((n, n)).at[jnp.tril_indices(n)].set(flat)
    # Exponentiate diagonal to ensure positivity
    L = L.at[jnp.diag_indices(n)].set(jnp.exp(jnp.diag(L)))
    return L @ L.T


PSD_MATRIX = ParameterTransform(
    to_unconstrained=_psd_to_real,
    to_constrained=_real_to_psd,
)


def transform_to_unconstrained(params: dict, spec: dict) -> dict:
    """Transform a dict of constrained parameters to unconstrained space."""
    return {k: spec[k].to_unconstrained(v) for k, v in params.items()}


def transform_to_constrained(unc_params: dict, spec: dict) -> dict:
    """Transform a dict of unconstrained parameters back to constrained space."""
    return {k: spec[k].to_constrained(v) for k, v in unc_params.items()}
