"""Lightweight parameter constraint transforms for SGD optimization.

Maps between constrained parameter spaces (positive reals, unit interval,
PSD matrices, row-stochastic matrices) and unconstrained reals for
gradient-based optimization. No TFP dependency -- pure JAX.

Validation is host-side: concrete invalid inputs raise ``ValueError`` at
pack/unpack boundaries, while traced values skip these checks so the transforms
remain compatible with ``jax.jit`` and ``jax.vmap``. Callers should validate
initial constrained parameters before entering compiled optimization loops.

Usage::

    unc_params = transform_to_unconstrained(params, param_spec)
    # ... optimize unc_params with optax ...
    params = transform_to_constrained(unc_params, param_spec)
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class ParameterTransform:
    """Maps between constrained and unconstrained parameter spaces."""

    to_unconstrained: Callable[[Array], Array]
    to_constrained: Callable[[Array], Array]
    trainable: bool = True


_OPEN_INTERVAL_EPS = 1e-6


def _as_float_array(x: Array) -> Array:
    """Convert numeric inputs to a floating JAX array without needless widening."""
    arr = jnp.asarray(x)
    return arr.astype(jnp.result_type(arr, 1.0))


def _dtype_tiny(x: Array) -> Array:
    """Smallest normal positive number for ``x``'s floating dtype."""
    return jnp.asarray(jnp.finfo(x.dtype).tiny, dtype=x.dtype)


def _check_array(condition: Array, message: str) -> None:
    """Validate concrete arrays while leaving traced JAX values jittable."""
    try:
        ok = bool(jnp.all(condition))
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerBoolConversionError,
    ):
        return
    if not ok:
        raise ValueError(message)


def _check_finite(name: str, x: Array) -> None:
    _check_array(jnp.isfinite(x), f"{name} must contain only finite values.")


def _inverse_softplus(x: Array) -> Array:
    """Inverse of softplus: log(exp(x) - 1), numerically stable."""
    x = _as_float_array(x)
    x = jnp.maximum(x, _dtype_tiny(x))
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(x)))


def _positive_from_real(x: Array) -> Array:
    """Map unconstrained reals to strictly positive values without overflow."""
    x = _as_float_array(x)
    return jax.nn.softplus(x) + _dtype_tiny(x)


def _positive_to_unconstrained(x: Array) -> Array:
    x = _as_float_array(x)
    _check_finite("POSITIVE", x)
    _check_array(x > 0.0, "POSITIVE values must be strictly positive.")
    shifted = jnp.maximum(x - _dtype_tiny(x), _dtype_tiny(x))
    return _inverse_softplus(shifted)


def _logit(x: Array, eps: float = _OPEN_INTERVAL_EPS) -> Array:
    """Numerically stable logit with finite endpoint clamping."""
    x = _as_float_array(x)
    eps_arr = jnp.asarray(eps, dtype=x.dtype)
    ratio = jnp.clip(x, eps_arr, 1.0 - eps_arr)
    return jnp.log(ratio) - jnp.log1p(-ratio)


# softplus chosen over exp to prevent overflow for large unconstrained values.
# softplus(x) = log(1 + exp(x)) grows linearly, not exponentially.
POSITIVE = ParameterTransform(
    to_unconstrained=_positive_to_unconstrained,
    to_constrained=_positive_from_real,
)


def positive_capped(max_val: float = 50.0) -> ParameterTransform:
    """Positive transform smoothly saturating in ``(0, max_val)``.

    Maps unconstrained reals to ``(0, max_val)`` via ``max_val * sigmoid``.
    Stored endpoint values 0 and ``max_val`` are accepted by
    ``to_unconstrained`` and clamped just inside the open interval so packing
    finite boundary initial values does not produce infinities.
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
        x = _as_float_array(x)
        _check_finite("positive_capped", x)
        _check_array(
            (x >= 0.0) & (x <= max_val),
            "positive_capped values must be between 0 and max_val inclusive.",
        )
        # logit(x / max_val), clamped away from the open-interval endpoints so
        # a value stored at exactly 0 or max_val does not map to +/-inf.
        return _logit(x / max_val)

    return ParameterTransform(
        to_unconstrained=_to_unconstrained,
        to_constrained=_to_constrained,
    )


def _unit_interval_to_unconstrained(x: Array) -> Array:
    x = _as_float_array(x)
    _check_finite("UNIT_INTERVAL", x)
    _check_array(
        (x >= 0.0) & (x <= 1.0),
        "UNIT_INTERVAL values must be between 0 and 1 inclusive.",
    )
    # Clamp away from the open-interval endpoints so a value stored at exactly
    # 0 or 1 (e.g. a decay of 1.0 = no forgetting) maps to a finite logit rather
    # than +/-inf, matching positive_capped's handling of its endpoints.
    return _logit(x)


UNIT_INTERVAL = ParameterTransform(
    to_unconstrained=_unit_interval_to_unconstrained,
    to_constrained=jax.nn.sigmoid,
)

UNCONSTRAINED = ParameterTransform(
    to_unconstrained=lambda x: x,
    to_constrained=lambda x: x,
)


def _psd_to_real(P: Array) -> Array:
    """Flatten a PSD matrix to unconstrained reals via Cholesky + softplus diagonal.

    A small jitter (1e-9) is added before Cholesky to prevent NaN on
    near-singular matrices during optimization. This introduces a bounded
    roundtrip error of ~1e-9.
    """
    P = _as_float_array(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("PSD_MATRIX values must be square 2D matrices.")
    if P.shape[0] == 0:
        raise ValueError("PSD_MATRIX values must be non-empty.")
    _check_finite("PSD_MATRIX", P)
    _check_array(
        jnp.isclose(P, P.T, rtol=1e-7, atol=1e-9),
        "PSD_MATRIX values must be symmetric.",
    )
    # Add jitter for numerical stability on near-singular matrices
    n = P.shape[0]
    jitter = jnp.asarray(1e-9, dtype=P.dtype)
    P_jittered = P + jitter * jnp.eye(n, dtype=P.dtype)
    _check_array(
        jnp.linalg.eigvalsh(P_jittered) > 0.0,
        "PSD_MATRIX values must be positive semidefinite.",
    )
    L = jnp.linalg.cholesky(P_jittered)
    # Replace diagonal with inverse-softplus coordinates so the constrained
    # direction can use softplus rather than exp and avoid optimizer overflow.
    diag = jnp.maximum(jnp.diag(L) - _dtype_tiny(L), _dtype_tiny(L))
    L = L.at[jnp.diag_indices_from(L)].set(_inverse_softplus(diag))
    return L[jnp.tril_indices_from(L)]


def _real_to_psd(flat: Array) -> Array:
    """Reconstruct a PSD matrix from unconstrained reals."""
    flat = _as_float_array(flat)
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
    if n == 0:
        raise ValueError("PSD_MATRIX unconstrained vector must be non-empty.")
    L = jnp.zeros((n, n), dtype=flat.dtype).at[jnp.tril_indices(n)].set(flat)
    # Softplus grows linearly for large inputs, avoiding exp overflow while
    # still guaranteeing a strictly positive Cholesky diagonal.
    L = L.at[jnp.diag_indices(n)].set(_positive_from_real(jnp.diag(L)))
    return L @ L.T


PSD_MATRIX = ParameterTransform(
    to_unconstrained=_psd_to_real,
    to_constrained=_real_to_psd,
)


def _stochastic_to_real(Z: Array) -> Array:
    """Row-stochastic matrix to unconstrained logits (drop last column)."""
    Z = _as_float_array(Z)
    if Z.shape[-1] == 0:
        raise ValueError("STOCHASTIC_ROW values must have at least one column.")
    _check_finite("STOCHASTIC_ROW", Z)
    _check_array(Z >= 0.0, "STOCHASTIC_ROW values must be non-negative.")
    _check_array(Z > 0.0, "STOCHASTIC_ROW values must be strictly positive.")
    _check_array(
        jnp.isclose(Z.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-8),
        "STOCHASTIC_ROW rows must sum to one.",
    )
    return jnp.log(Z[..., :-1]) - jnp.log(Z[..., -1:])


def _real_to_stochastic(logits: Array) -> Array:
    """Unconstrained logits to row-stochastic matrix via softmax."""
    logits = _as_float_array(logits)
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


def _validate_matching_keys(
    values: dict,
    spec: dict,
    *,
    values_name: str,
    allow_missing_non_trainable: bool = False,
    static_params: Optional[dict] = None,
) -> None:
    value_keys = set(values)
    spec_keys = set(spec)
    static_keys = set(static_params or {})
    extra = (value_keys | static_keys) - spec_keys
    missing = spec_keys - value_keys - static_keys
    if allow_missing_non_trainable:
        missing = {k for k in missing if spec[k].trainable}
    if extra or missing:
        parts = []
        if missing:
            parts.append(f"missing keys required by spec: {sorted(missing)}")
        if extra:
            parts.append(f"keys not present in spec: {sorted(extra)}")
        raise ValueError(f"{values_name} key mismatch; " + "; ".join(parts) + ".")


def transform_to_unconstrained(
    params: dict,
    spec: dict,
    *,
    include_non_trainable: bool = True,
) -> dict:
    """Transform constrained parameters to unconstrained optimizer coordinates.

    Parameters marked ``trainable=False`` are included by default for backwards
    compatibility. Pass ``include_non_trainable=False`` when building an
    optimizer pytree; frozen constrained values can then be supplied to
    :func:`transform_to_constrained` via ``static_params``.
    """
    _validate_matching_keys(params, spec, values_name="params")
    return {
        k: spec[k].to_unconstrained(v)
        for k, v in params.items()
        if include_non_trainable or spec[k].trainable
    }


def transform_to_constrained(
    unc_params: dict,
    spec: dict,
    *,
    static_params: Optional[dict] = None,
) -> dict:
    """Transform a dict of unconstrained parameters back to constrained space.

    ``static_params`` supplies already-constrained parameters omitted from
    ``unc_params`` (typically frozen parameters excluded from an optimizer
    pytree). Applies ``jax.lax.stop_gradient`` for parameters marked as not
    trainable.
    """
    static_params = {} if static_params is None else static_params
    overlap = set(unc_params) & set(static_params)
    if overlap:
        raise ValueError(
            f"Parameters cannot appear in both unc_params and static_params: "
            f"{sorted(overlap)}."
        )
    _validate_matching_keys(
        unc_params,
        spec,
        values_name="unc_params",
        static_params=static_params,
    )
    result = {}
    for k, v in unc_params.items():
        value = spec[k].to_constrained(v)
        if not spec[k].trainable:
            value = jax.lax.stop_gradient(value)
        result[k] = value
    for k, v in static_params.items():
        value = _as_float_array(v)
        if not spec[k].trainable:
            value = jax.lax.stop_gradient(value)
        result[k] = value
    return result
