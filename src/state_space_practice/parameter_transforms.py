"""Lightweight parameter constraint transforms for SGD optimization.

Maps between constrained parameter spaces (positive reals, unit interval,
PSD matrices, row-stochastic matrices) and unconstrained reals for
gradient-based optimization. No TFP dependency -- pure JAX.

Usage::

    unc_params = transform_to_unconstrained(params, param_spec)
    # ... optimize unc_params with optax ...
    params = transform_to_constrained(unc_params, param_spec)
"""

import math
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class ParameterTransform:
    """Maps between constrained and unconstrained parameter spaces."""

    to_unconstrained: Callable[[Array], Array]
    to_constrained: Callable[[Array], Array]
    trainable: bool = True


def _check_array(condition: Array, message: str) -> None:
    """Validate concrete arrays while leaving traced JAX values jittable."""
    try:
        ok = bool(jnp.all(condition))
    except jax.errors.TracerBoolConversionError:
        return
    if not ok:
        raise ValueError(message)


def _check_finite(name: str, x: Array) -> None:
    _check_array(jnp.isfinite(x), f"{name} must contain only finite values.")


def _inverse_softplus(x: Array) -> Array:
    """Inverse of softplus: log(exp(x) - 1), numerically stable."""
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(x)))


def _positive_to_unconstrained(x: Array) -> Array:
    x = jnp.asarray(x)
    _check_finite("POSITIVE", x)
    _check_array(x > 0.0, "POSITIVE values must be strictly positive.")
    return _inverse_softplus(x)


# softplus chosen over exp to prevent overflow for large unconstrained values.
# softplus(x) = log(1 + exp(x)) grows linearly, not exponentially.
POSITIVE = ParameterTransform(
    to_unconstrained=_positive_to_unconstrained,
    to_constrained=jax.nn.softplus,
)

def positive_capped(max_val: float = 50.0) -> ParameterTransform:
    """Positive transform smoothly saturating in ``(0, max_val)``.

    Maps unconstrained reals to ``(0, max_val)`` via ``max_val * sigmoid``.
    Unlike a hard ``min(softplus(x), max_val)`` clip -- whose gradient is
    exactly zero once the cap is reached, so SGD cannot move a parameter back
    off the boundary and it freezes there with no diagnostic -- this map has a
    strictly positive gradient everywhere. Useful for inverse-temperature
    parameters where very large values cause ill-conditioned Hessians in the
    Laplace-EKF update.
    """
    if not math.isfinite(max_val) or max_val <= 0.0:
        raise ValueError("max_val must be positive and finite.")

    def _to_constrained(x: Array) -> Array:
        return max_val * jax.nn.sigmoid(x)

    def _to_unconstrained(x: Array) -> Array:
        x = jnp.asarray(x)
        _check_finite("positive_capped", x)
        _check_array(
            (x >= 0.0) & (x <= max_val),
            "positive_capped values must be between 0 and max_val.",
        )
        # logit(x / max_val), clamped away from the open-interval endpoints so
        # a value stored at exactly 0 or max_val does not map to +/-inf.
        ratio = jnp.clip(x / max_val, 1e-6, 1.0 - 1e-6)
        return jnp.log(ratio) - jnp.log1p(-ratio)

    return ParameterTransform(
        to_unconstrained=_to_unconstrained,
        to_constrained=_to_constrained,
    )


def _unit_interval_to_unconstrained(x: Array) -> Array:
    x = jnp.asarray(x)
    _check_finite("UNIT_INTERVAL", x)
    _check_array(
        (x >= 0.0) & (x <= 1.0),
        "UNIT_INTERVAL values must be between 0 and 1.",
    )
    # Clamp away from the open-interval endpoints so a value stored at exactly
    # 0 or 1 (e.g. a decay of 1.0 = no forgetting) maps to a finite logit rather
    # than +/-inf, matching positive_capped's handling of its endpoints.
    ratio = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jnp.log(ratio / (1 - ratio))


UNIT_INTERVAL = ParameterTransform(
    to_unconstrained=_unit_interval_to_unconstrained,
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
    P = jnp.asarray(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("PSD_MATRIX values must be square 2D matrices.")
    _check_finite("PSD_MATRIX", P)
    _check_array(
        jnp.isclose(P, P.T, rtol=1e-7, atol=1e-9),
        "PSD_MATRIX values must be symmetric.",
    )
    # Add jitter for numerical stability on near-singular matrices
    n = P.shape[0]
    jitter = 1e-9
    P_jittered = P + jitter * jnp.eye(n)
    _check_array(
        jnp.linalg.eigvalsh(P_jittered) > 0.0,
        "PSD_MATRIX values must be positive semidefinite.",
    )
    L = jnp.linalg.cholesky(P_jittered)
    # Replace diagonal with log(diagonal) so it's unconstrained
    L = L.at[jnp.diag_indices_from(L)].set(jnp.log(jnp.diag(L)))
    return L[jnp.tril_indices_from(L)]


def _real_to_psd(flat: Array) -> Array:
    """Reconstruct a PSD matrix from unconstrained reals."""
    flat = jnp.asarray(flat)
    if flat.ndim != 1:
        raise ValueError("PSD_MATRIX unconstrained values must be a 1D vector.")
    _check_finite("PSD_MATRIX unconstrained", flat)
    # Solve n*(n+1)/2 = len(flat) for n
    # Use math.sqrt (not jnp.sqrt) so n is a concrete int under jax.jit
    n = int((-1 + math.sqrt(1 + 8 * flat.shape[0])) / 2)
    if n * (n + 1) // 2 != flat.shape[0]:
        raise ValueError(
            "PSD_MATRIX unconstrained vector length must be triangular "
            "(n * (n + 1) / 2)."
        )
    L = jnp.zeros((n, n)).at[jnp.tril_indices(n)].set(flat)
    # Exponentiate diagonal to ensure positivity
    L = L.at[jnp.diag_indices(n)].set(jnp.exp(jnp.diag(L)))
    return L @ L.T


PSD_MATRIX = ParameterTransform(
    to_unconstrained=_psd_to_real,
    to_constrained=_real_to_psd,
)


def _stochastic_to_real(Z: Array) -> Array:
    """Row-stochastic matrix to unconstrained logits (drop last column)."""
    Z = jnp.asarray(Z)
    if Z.shape[-1] == 0:
        raise ValueError("STOCHASTIC_ROW values must have at least one column.")
    _check_finite("STOCHASTIC_ROW", Z)
    _check_array(Z >= 0.0, "STOCHASTIC_ROW values must be non-negative.")
    _check_array(
        jnp.isclose(Z.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-8),
        "STOCHASTIC_ROW rows must sum to one.",
    )
    # Jitter to avoid log(0)
    Z_safe = jnp.maximum(Z, 1e-10)
    return jnp.log(Z_safe[..., :-1]) - jnp.log(Z_safe[..., -1:])


def _real_to_stochastic(logits: Array) -> Array:
    """Unconstrained logits to row-stochastic matrix via softmax."""
    logits = jnp.asarray(logits)
    if logits.shape[-1] == 0:
        return jnp.ones(logits.shape[:-1] + (1,), dtype=logits.dtype)
    full_logits = jnp.concatenate(
        [logits, jnp.zeros_like(logits[..., :1])], axis=-1
    )
    return jax.nn.softmax(full_logits, axis=-1)


STOCHASTIC_ROW = ParameterTransform(
    to_unconstrained=_stochastic_to_real,
    to_constrained=_real_to_stochastic,
)


def frozen(transform: ParameterTransform) -> ParameterTransform:
    """Return a copy of the transform with trainable=False."""
    return ParameterTransform(
        to_unconstrained=transform.to_unconstrained,
        to_constrained=transform.to_constrained,
        trainable=False,
    )


def transform_to_unconstrained(params: dict, spec: dict) -> dict:
    """Transform a dict of constrained parameters to unconstrained space."""
    return {k: spec[k].to_unconstrained(v) for k, v in params.items()}


def transform_to_constrained(unc_params: dict, spec: dict) -> dict:
    """Transform a dict of unconstrained parameters back to constrained space.

    Applies jax.lax.stop_gradient for parameters marked as not trainable.
    """
    result = {}
    for k, v in unc_params.items():
        value = spec[k].to_constrained(v)
        if not spec[k].trainable:
            value = jax.lax.stop_gradient(value)
        result[k] = value
    return result
