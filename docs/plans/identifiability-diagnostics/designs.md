# Designs

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Complete code for the parts of the implementation that are not obvious. Phases
reference sections by anchor; this file does not repeat task lists. Contracts (signatures,
report fields, option defaults) live in [shared-contracts.md](shared-contracts.md) and are
not restated here — the code below implements them.

Sections:

1. [Module header and coordinate charts](#module-header-and-coordinate-charts)
2. [Hessian methods](#hessian-methods)
3. [Spectrum analysis](#spectrum-analysis)
4. [Worst-direction slice](#worst-direction-slice)
5. [Report text](#report-text)
6. [`identifiability_report_from_loss` and `fisher_information`](#entry-points)
7. [Model entry point and mixin wiring (phase 2)](#model-entry-point)
8. [The Hessian is only meaningful at a stationary point](#the-hessian-is-only-meaningful-at-a-stationary-point)
9. [Why the observed Hessian suffices at initialisation](#why-the-observed-hessian-suffices-at-initialisation)
10. [Test losses: spike-only and LFP-anchored coupling](#test-losses-spike-only-and-lfp-anchored-coupling)
11. [Test helper: polishing to a stationary point](#test-helper-polishing-to-a-stationary-point)
12. [Test helper: symmetry generators by finite difference](#test-helper-symmetry-generators-by-finite-difference)
13. [Literature](#literature)

## Module header and coordinate charts

```python
"""Identifiability diagnostics from the observed information of a fitted model.

The observed information is the Hessian of the negative log-likelihood (the SGD
loss) with respect to the model's *scientific* parameters -- variances,
loadings, couplings -- in their own units. Its eigen-structure tells which
parameters, or which linear combinations of parameters, the data determine:
an eigenvalue at zero is a direction the objective does not curve in, so the
fit along it is arbitrary; a coordinate whose whole row is zero is a parameter
the objective is flat in at this point, so a gradient method cannot move it.

Two caveats the report carries with it. First, the Hessian describes flat
directions of the objective only at a stationary point (along a curved orbit of
an exact symmetry ``gen' H gen = -g . theta''``), so every report states whether
the point is stationary. Second, the report describes the objective the model
optimises: penalties and priors count as curvature and can hide a degeneracy of
the likelihood alone.

The report is built by :func:`identifiability_report_from_loss` for any
differentiable loss over a dict of constrained parameters, and by
:func:`identifiability_report` / ``model.identifiability_report()`` for
``SGDFittableMixin`` models. Requires float64 (``jax_enable_x64``) for the
eigenvalue tolerances to be meaningful.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike

from state_space_practice.exceptions import NotFittedError, StateSpaceWarning
from state_space_practice.parameter_transforms import (
    PSD_MATRIX,
    STOCHASTIC_ROW,
    ParameterTransform,
)

logger = logging.getLogger(__name__)

HessianMethod = Literal["auto", "autodiff", "finite_difference"]
ChartKind = Literal["identity", "psd", "stochastic_row"]

#: Floor on a coordinate's magnitude when a *relative* step is formed from it
#: (finite-difference steps, slice widths). Below this, steps are absolute.
_RELATIVE_SCALE_FLOOR = 1e-2
#: Smallest |coefficient| printed for a collinear near-null direction.
_MIN_PRINTED_COEFFICIENT = 0.05


def _as_float(x: Any) -> Array:
    arr = jnp.asarray(x)
    return arr.astype(jnp.result_type(arr, 1.0))


def _chart_kind(transform: ParameterTransform | None) -> ChartKind:
    if transform is None:
        return "identity"
    if transform.to_constrained is PSD_MATRIX.to_constrained:
        return "psd"
    if transform.to_constrained is STOCHASTIC_ROW.to_constrained:
        return "stochastic_row"
    return "identity"


def _to_chart(value: Array, kind: ChartKind) -> Array:
    """Constrained value -> minimal scientific coordinates."""
    if kind == "psd":
        return value[jnp.tril_indices(value.shape[0])]
    if kind == "stochastic_row":
        return value[..., :-1]
    return value


def _from_chart(coords: Array, kind: ChartKind, template: Array) -> Array:
    """Inverse of :func:`_to_chart` (``template`` supplies the shape)."""
    if kind == "psd":
        n = template.shape[0]
        lower = jnp.zeros((n, n), dtype=coords.dtype).at[jnp.tril_indices(n)].set(coords)
        return lower + lower.T - jnp.diag(jnp.diag(lower))
    if kind == "stochastic_row":
        last = 1.0 - jnp.sum(coords, axis=-1, keepdims=True)
        return jnp.concatenate([coords, last], axis=-1)
    return coords


def _leaf_label(key: str, path: tuple[Any, ...]) -> str:
    """``key.sub.leaf`` for a nested pytree leaf; ``key`` for a bare array."""
    parts = [key]
    for entry in path:
        if isinstance(entry, jax.tree_util.DictKey):
            parts.append(str(entry.key))
        elif isinstance(entry, jax.tree_util.SequenceKey):
            parts.append(str(entry.idx))
        elif isinstance(entry, jax.tree_util.GetAttrKey):
            parts.append(entry.name)
        else:  # pragma: no cover - other key types are not produced by our params
            parts.append(str(entry))
    return ".".join(parts)


def _coordinate_names(label: str, kind: ChartKind, template: Array) -> list[str]:
    if kind == "psd":
        rows, cols = np.tril_indices(template.shape[0])
        return [f"{label}[{i},{j}]" for i, j in zip(rows, cols, strict=True)]
    shape = tuple(template.shape)
    if kind == "stochastic_row":
        shape = shape[:-1] + (shape[-1] - 1,)
    if shape == ():
        return [label]
    return [label + "[" + ",".join(map(str, idx)) + "]" for idx in np.ndindex(*shape)]


@dataclass(frozen=True)
class ScientificCoordinates:
    """Flattened minimal coordinates of the trainable parameters.

    Attributes
    ----------
    names : tuple of str, length P
        One label per coordinate, e.g. ``"coupling_strength[0,1,0]"``,
        ``"init_cov[1,0]"`` (lower-triangular entry of a PSD matrix),
        ``"mlp.layer_0.w[3,1]"`` (nested pytree leaf).
    values : Array, shape (P,)
        The coordinates at the given parameters.
    unravel : callable
        Maps a ``(P,)`` vector back to the full constrained parameter dict,
        frozen (non-trainable) entries included.
    """

    names: tuple[str, ...]
    values: Array
    unravel: Callable[[Array], dict[str, Any]]


def scientific_coordinates(
    params: Mapping[str, Any],
    transforms: Mapping[str, ParameterTransform] | None = None,
) -> ScientificCoordinates:
    """Flatten constrained parameters into minimal scientific coordinates.

    Parameters
    ----------
    params : mapping key -> constrained value (array or pytree of arrays)
    transforms : mapping key -> ParameterTransform, optional
        Entries with ``trainable=False`` are held fixed. ``PSD_MATRIX`` values
        are represented by their lower triangle (``n(n+1)/2`` coordinates),
        ``STOCHASTIC_ROW`` values by all but their last column; every other
        transform (or none) uses the value itself. Keys absent from
        ``transforms`` are trainable with the identity chart.
    """
    transforms = {} if transforms is None else dict(transforms)
    extra = set(transforms) - set(params)
    if extra:
        raise ValueError(f"transforms has keys not present in params: {sorted(extra)}.")

    fixed: dict[str, Any] = {}
    coords: dict[str, Any] = {}
    kinds: dict[str, ChartKind] = {}
    templates: dict[str, Any] = {}
    names: list[str] = []
    for key in sorted(params):  # ravel_pytree flattens dicts in sorted-key order
        value = jax.tree_util.tree_map(_as_float, params[key])
        transform = transforms.get(key)
        if transform is not None and not transform.trainable:
            fixed[key] = value
            continue
        kind = _chart_kind(transform)
        leaves_with_path = jax.tree_util.tree_flatten_with_path(value)[0]
        if kind != "identity" and (len(leaves_with_path) != 1 or leaves_with_path[0][0]):
            raise ValueError(
                f"{key!r}: a PSD_MATRIX / STOCHASTIC_ROW parameter must be a single array."
            )
        kinds[key], templates[key] = kind, value
        coords[key] = jax.tree_util.tree_map(lambda leaf: _to_chart(leaf, kind), value)
        for path, leaf in leaves_with_path:
            names.extend(_coordinate_names(_leaf_label(key, path), kind, leaf))

    flat, unravel_coords = ravel_pytree(coords)
    if flat.size == 0:
        raise ValueError("No trainable parameters to analyse.")

    def unravel(vector: Array) -> dict[str, Any]:
        chart_tree = unravel_coords(vector)
        full = {
            key: jax.tree_util.tree_map(
                lambda c, t, k=kinds[key]: _from_chart(c, k, t),
                chart_tree[key],
                templates[key],
            )
            for key in chart_tree
        }
        full.update(fixed)
        return full

    return ScientificCoordinates(tuple(names), flat, unravel)
```

Note `zip(..., strict=True)`: ruff `B905` is enforced for new modules (`pyproject.toml`
per-file ignores exempt only the listed legacy modules).

## Hessian methods

```python
def _hessian_finite_difference(
    f: Callable[[Array], Array], x: Array, fd_step: float
) -> Array:
    """Richardson-extrapolated central differences of the exact gradient.

    ``(4 D(h/2) - D(h)) / 3`` with ``D(h) = (g(x + h e_i) - g(x - h e_i)) / 2h`` has
    truncation error O(h^4); with ``h_i = fd_step * max(|x_i|, 1e-2)`` and
    ``fd_step = 1e-3`` the round-off term is ~1e-10 relative in float64, which
    resolves exact null directions to ~1e-8 of the largest eigenvalue. Costs
    ``4 P`` gradient evaluations. Positive coordinates stay positive because the
    step is relative.
    """
    grad = jax.jit(jax.grad(f))
    steps = fd_step * jnp.maximum(jnp.abs(x), _RELATIVE_SCALE_FLOOR)
    columns = []
    for i in range(int(x.size)):
        unit = jnp.zeros_like(x).at[i].set(1.0)

        def central(h: Array) -> Array:
            return (grad(x + h * unit) - grad(x - h * unit)) / (2.0 * h)

        h = steps[i]
        columns.append((4.0 * central(0.5 * h) - central(h)) / 3.0)
    return jnp.stack(columns, axis=1)


def _is_callback_jvp_error(err: BaseException) -> bool:
    """The error jax raises when differentiating through a ``pure_callback`` JVP rule."""
    return isinstance(err, ValueError) and "callback" in str(err).lower()


def _loss_hessian(
    f: Callable[[Array], Array], x: Array, method: HessianMethod, fd_step: float
) -> tuple[Array, Literal["autodiff", "finite_difference"]]:
    if method not in ("auto", "autodiff", "finite_difference"):
        raise ValueError(
            f"hessian_method must be 'auto', 'autodiff' or 'finite_difference', got {method!r}."
        )
    if method == "finite_difference":
        return _hessian_finite_difference(f, x, fd_step), "finite_difference"
    try:
        return jax.hessian(f)(x), "autodiff"
    except ValueError as err:
        if method == "autodiff" or not _is_callback_jvp_error(err):
            raise
        # differentiable_spectral_radius (utils.py) has a custom JVP built on a
        # host callback; its rule cannot be differentiated again. The exact
        # gradient exists, so difference it.
        logger.info(
            "Second-order autodiff is unavailable for this loss (%s); using finite "
            "differences of the exact gradient.",
            err,
        )
        return _hessian_finite_difference(f, x, fd_step), "finite_difference"
```

`x` is the `(P,)` coordinate vector and `f(x) = loss_fn(coords.unravel(x))`; the data are
closed over as constants, so no data tracing issue arises (the concreteness problems
`sgd_fitting.py:294-301` guards against concern *data* as jit arguments, which we never
do).

## Spectrum analysis

```python
@dataclass(frozen=True)
class NullDirection:
    kind: Literal["zero_curvature", "collinear"]
    eigenvalue: float
    coefficients: dict[str, float]

    def describe(self) -> str:
        if self.kind == "zero_curvature":
            (name,) = self.coefficients
            return f"{name} (zero curvature, H_ii = {self.eigenvalue:.3g})"
        terms = " ".join(
            f"{c:+.2f}*{n}"
            for n, c in sorted(self.coefficients.items(), key=lambda kv: -abs(kv[1]))
        )
        return f"lambda = {self.eigenvalue:.3g}: {terms}"


def _analyse_spectrum(
    names: tuple[str, ...],
    grad: npt.NDArray[np.float64],
    hessian: npt.NDArray[np.float64],
    *,
    near_null_tol: float,
    zero_curvature_tol: float,
    stationary_tol: float,
) -> dict[str, Any]:
    """Raw and normalised eigen-analysis, flags and stationarity."""
    H = 0.5 * (hessian + hessian.T)
    diag = np.diag(H)
    row_scale = np.max(np.abs(H), axis=1)
    matrix_scale = float(row_scale.max())
    # A zero diagonal alone does not imply a flat coordinate at a saddle.
    flat = row_scale <= zero_curvature_tol * matrix_scale
    active = ~flat

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Positive congruence on ALL coordinates preserves inertia, including rows
    # flagged as numerically flat. Use a row scale when a diagonal is zero.
    scale2 = np.where(np.abs(diag) > 0.0, np.abs(diag), row_scale)
    full_scale = np.sqrt(np.where(scale2 > 0.0, scale2, 1.0))
    full_normalized = H / np.outer(full_scale, full_scale)
    full_values, full_vectors = np.linalg.eigh(full_normalized)
    full_radius = float(np.max(np.abs(full_values)))
    negative = full_values < -near_null_tol * full_radius

    d = full_scale[active]
    normalized = H[np.ix_(active, active)] / np.outer(d, d)
    if normalized.size:
        n_eigenvalues, n_eigenvectors = np.linalg.eigh(normalized)
        spectral_radius = float(np.max(np.abs(n_eigenvalues)))
    else:
        n_eigenvalues = np.zeros(0)
        n_eigenvectors = np.zeros((0, 0))
        spectral_radius = 0.0
    near_null = np.abs(n_eigenvalues) < near_null_tol * spectral_radius
    active_names = tuple(n for n, a in zip(names, active, strict=True) if a)

    directions: list[NullDirection] = [
        NullDirection("zero_curvature", float(diag[i]), {names[i]: 1.0})
        for i in np.flatnonzero(flat)
    ]
    for k in np.flatnonzero(near_null):
        vec = n_eigenvectors[:, k]
        vec = vec * np.sign(vec[np.argmax(np.abs(vec))])  # deterministic sign
        coefficients = {
            n: float(c)
            for n, c in zip(active_names, vec, strict=True)
            if abs(c) >= _MIN_PRINTED_COEFFICIENT
        }
        directions.append(NullDirection("collinear", float(n_eigenvalues[k]), coefficients))

    # Stationarity: distance to the optimum along each coordinate in units of that
    # coordinate's own curvature scale (dimensionless), plus "flat coordinates have
    # no slope" (a slope along a flat coordinate means the objective is unbounded
    # along it, not stationary).
    grad_abs = np.abs(grad)
    scaled = grad_abs[active] / d if active.any() else np.zeros(0)
    scaled_gradient_max = float(scaled.max()) if scaled.size else 0.0
    flat_slope_ok = bool(
        np.all(grad_abs[flat] <= stationary_tol * max(float(grad_abs.max()), 0.0))
    )
    is_stationary = scaled_gradient_max < stationary_tol and flat_slope_ok

    def _ratio(values: npt.NDArray[np.float64]) -> float:
        if values.size == 0 or values[0] <= 0.0:
            return float("inf")
        return float(values[-1] / values[0])

    try:
        chol = np.linalg.cholesky(H)
        inv_diag = np.sum(np.linalg.inv(chol) ** 2, axis=0)
        wald = dict(zip(names, np.sqrt(inv_diag).tolist(), strict=True))
    except np.linalg.LinAlgError:
        wald = None

    return dict(
        hessian=H,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        zero_curvature_parameters=tuple(names[i] for i in np.flatnonzero(flat)),
        normalized_eigenvalues=n_eigenvalues,
        normalized_eigenvectors=n_eigenvectors,
        active_parameter_names=active_names,
        condition_number=_ratio(n_eigenvalues),
        sloppiness_ratio=_ratio(eigenvalues),
        n_negative_eigenvalues=int(negative.sum()),
        near_null_directions=tuple(directions),
        wald_standard_errors=wald,
        scaled_gradient_max=scaled_gradient_max,
        is_stationary=is_stationary,
        _flat_mask=flat,
        _active_scale=d,
        _full_scale=full_scale,
        _full_eigenvectors=full_vectors,
    )
```

Why two detectors: normalising a tiny, isolated positive diagonal produces an
isolated unit eigenvalue, which can hide a numerically flat coordinate. Measured
on the spike-only coupling loss at β = 0: raw `H_QQ =
1.08e-11` against `max |H_jj| = 162.7` (ratio 6.7e-14), cross terms exactly 0, and the
normalised spectrum `[..., 0.369, 0.369, 1.0]` — the `1.0` is the flat coordinate. Hence
the full-row annotation with `zero_curvature_tol = 1e-10`. This annotation is
unit-dependent and must never remove a coordinate from the negative-curvature
check. For `H = [[0, 1], [1, 0]]`, neither row is flat and the full congruence
has eigenvalues `[-1, 1]`; a stationary saddle must be reported. A tiny negative
isolated diagonal is likewise retained in this check even if its row is flagged.
For nonzero diagonal entries and an unchanged active set, correlation
normalisation is invariant to positive rescaling of parameter units; the raw
row threshold and zero-diagonal fallback do not have that invariance. Relative
eigenvalue tolerances are numerical decisions, not exact inertia identities.

`inv_diag` computes `diag(H^{-1})` from the Cholesky factor `L` as the squared column
norms of `L^{-1}` (`H^{-1} = L^{-T} L^{-1}`).

## Worst-direction slice

```python
def _worst_direction(analysis: dict[str, Any]) -> npt.NDArray[np.float64]:
    """Unit raw-coordinate direction of least curvature."""
    flat = analysis["_flat_mask"]
    n_total = flat.size
    direction = np.zeros(n_total)
    if analysis["n_negative_eigenvalues"]:
        direction = analysis["_full_eigenvectors"][:, 0] / analysis["_full_scale"]
        return direction / np.linalg.norm(direction)
    if flat.any():
        direction[int(np.flatnonzero(flat)[0])] = 1.0
        return direction
    # Smallest normalised eigenvector w corresponds to the raw direction D^{-1/2} w.
    direction[~flat] = analysis["normalized_eigenvectors"][:, 0] / analysis["_active_scale"]
    return direction / np.linalg.norm(direction)


def _slice_along(
    f: Callable[[Array], Array],
    x: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    *,
    half_width: float,
    n_points: int,
) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None]:
    """Loss along ``x + t * delta`` for ``t`` in ``linspace(-1, 1, n_points)``.

    ``delta`` is ``direction`` scaled so that the largest *relative* change of any
    coordinate at ``t = 1`` equals ``half_width`` (relative to ``max(|x_i|, 1e-2)``).
    Non-finite losses (a step outside the parameter domain) are kept as NaN.
    """
    scale = np.maximum(np.abs(x), _RELATIVE_SCALE_FLOOR)
    delta = half_width * direction / float(np.max(np.abs(direction) / scale))
    if n_points <= 0:
        return delta, None
    t = np.linspace(-1.0, 1.0, n_points)
    values = np.array([float(f(jnp.asarray(x + ti * delta))) for ti in t])
    return delta, (t, values)
```

## Report text

```python
    def __str__(self) -> str:
        lines = [
            f"Identifiability report: {len(self.parameter_names)} parameters, "
            f"loss = {self.loss:.6g}, |grad| = {self.gradient_norm:.3g}, "
            f"Hessian by {self.hessian_method_used}",
            (
                f"  stationary: {'yes' if self.is_stationary else 'NO'} "
                f"(max |g_i|/d_i = {self.scaled_gradient_max:.3g}; "
                f"near-null directions are only flat directions at a stationary point)"
            ),
            (
                f"  negative eigenvalues: {self.n_negative_eigenvalues}"
                + ("  <- not a local minimum" if self.n_negative_eigenvalues else "")
            ),
            (
                f"  condition number (normalised): {self.condition_number:.3g}; "
                f"sloppiness ratio (raw, unit-dependent): {self.sloppiness_ratio:.3g}"
            ),
        ]
        if self.near_null_directions:
            lines.append(
                f"  NOT IDENTIFIABLE: {len(self.near_null_directions)} near-null direction(s) "
                f"(|lambda| < {self.near_null_tol:g} lambda_max, "
                f"max_j |H_ij| <= {self.zero_curvature_tol:g} max_jk |H_jk|):"
            )
            lines.extend(f"    {d.describe()}" for d in self.near_null_directions)
        else:
            lines.append("  identifiable: no near-null direction at these tolerances")
        if self.wald_standard_errors is None:
            lines.append("  Wald SEs: unavailable (Hessian not positive definite)")
        else:
            note = " (inflated along near-null directions)" if self.near_null_directions else ""
            lines.append(f"  Wald SEs{note}:")
            lines.extend(
                f"    {n} = {v:.6g} +/- {self.wald_standard_errors[n]:.3g}"
                for n, v in zip(self.parameter_names, self.parameter_values, strict=True)
            )
        if self.worst_direction_slice is not None:
            t, vals = self.worst_direction_slice
            lines.append(
                "  slice along worst direction (loss - loss(0); not a profile, other "
                "parameters are held fixed): "
                + ", ".join(f"{ti:+.2f}:{v - self.loss:+.3g}" for ti, v in zip(t, vals, strict=True))
            )
        return "\n".join(lines)
```

## Entry points

```python
def _flat_loss(
    loss_fn: Callable[[dict], ArrayLike], coords: ScientificCoordinates
) -> Callable[[Array], Array]:
    def f(x: Array) -> Array:
        return jnp.reshape(jnp.asarray(loss_fn(coords.unravel(x))), ())

    return f


def fisher_information(
    loss_fn: Callable[[dict], ArrayLike],
    params: Mapping[str, Any],
    transforms: Mapping[str, ParameterTransform] | None = None,
    *,
    hessian_method: HessianMethod = "auto",
    fd_step: float = 1e-3,
) -> Array:
    """Observed information: Hessian of ``loss_fn`` in scientific coordinates.

    ... (NumPy docstring: parameters as in shared-contracts; Returns ``Array,
    shape (P, P)``, symmetrised; row ``i`` is ``scientific_coordinates(params,
    transforms).names[i]``. Notes: this is the *observed* information; it equals
    the expected Fisher information only in expectation, or at the MLE of an
    exponential-family model. Cost: ``P`` forward-over-reverse passes
    (``autodiff``) or ``4P`` gradient evaluations (``finite_difference``).)
    """
    coords = scientific_coordinates(params, transforms)
    hessian, _ = _loss_hessian(_flat_loss(loss_fn, coords), coords.values, hessian_method, fd_step)
    return 0.5 * (hessian + hessian.T)


def identifiability_report_from_loss(
    loss_fn: Callable[[dict], ArrayLike],
    params: Mapping[str, Any],
    transforms: Mapping[str, ParameterTransform] | None = None,
    *,
    near_null_tol: float = 1e-6,
    zero_curvature_tol: float = 1e-10,
    stationary_tol: float = 1e-2,
    hessian_method: HessianMethod = "auto",
    fd_step: float = 1e-3,
    n_slice_points: int = 9,
    slice_half_width: float = 0.5,
    warn: bool = True,
) -> IdentifiabilityReport:
    """Identifiability report of a differentiable loss at ``params``. ..."""
    for name, value in (
        ("near_null_tol", near_null_tol),
        ("zero_curvature_tol", zero_curvature_tol),
        ("stationary_tol", stationary_tol),
        ("fd_step", fd_step),
        ("slice_half_width", slice_half_width),
    ):
        if not (value > 0.0) or not np.isfinite(value):
            raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    if n_slice_points < 0:
        raise ValueError(f"n_slice_points must be non-negative, got {n_slice_points}.")

    coords = scientific_coordinates(params, transforms)
    f = _flat_loss(loss_fn, coords)
    x = coords.values
    loss = float(f(x))
    grad = np.asarray(jax.grad(f)(x), dtype=float)
    if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
        bad = [n for n, g in zip(coords.names, grad, strict=True) if not np.isfinite(g)]
        raise ValueError(
            f"loss ({loss}) or gradient is not finite at the given parameters "
            f"(non-finite gradient entries: {bad}); the report needs a finite point."
        )
    hessian_jax, method_used = _loss_hessian(f, x, hessian_method, fd_step)
    hessian = np.asarray(hessian_jax, dtype=float)
    if not np.all(np.isfinite(hessian)):
        rows = sorted({coords.names[i] for i in np.flatnonzero(~np.isfinite(hessian).all(axis=1))})
        raise ValueError(f"Hessian has non-finite entries in rows {rows}.")

    analysis = _analyse_spectrum(
        coords.names, grad, hessian,
        near_null_tol=near_null_tol,
        zero_curvature_tol=zero_curvature_tol,
        stationary_tol=stationary_tol,
    )
    x_np = np.asarray(x, dtype=float)
    delta, slice_ = _slice_along(
        f, x_np, _worst_direction(analysis), half_width=slice_half_width, n_points=n_slice_points
    )
    report = IdentifiabilityReport(
        parameter_names=coords.names,
        parameter_values=x_np,
        loss=loss,
        gradient=grad,
        gradient_norm=float(np.linalg.norm(grad)),
        hessian_method_used=method_used,
        worst_direction=delta,
        worst_direction_slice=slice_,
        near_null_tol=near_null_tol,
        zero_curvature_tol=zero_curvature_tol,
        stationary_tol=stationary_tol,
        **{k: v for k, v in analysis.items() if not k.startswith("_")},
    )
    if warn and not report.is_identifiable:
        warnings.warn(
            "Parameters are not identifiable at these tolerances: "
            + "; ".join(d.describe() for d in report.near_null_directions)
            + (
                "" if report.is_stationary
                else " (the point is not stationary, so these directions may reflect "
                "the gradient rather than the likelihood surface)"
            ),
            StateSpaceWarning,
            stacklevel=2,
        )
    return report
```

## Model entry point

Phase 2. In `identifiability.py`:

```python
class _SGDLossModel(Protocol):
    """The part of the ``SGDFittableMixin`` protocol the report needs."""

    def _check_sgd_initialized(self) -> None: ...
    def _prepare_sgd_data(
        self, *args: Any, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def _build_param_spec(self) -> tuple[dict, dict]: ...
    def _sgd_loss_fn(self, params: dict, *args: Any, **kwargs: Any) -> Array: ...


_REPORT_OPTIONS = (
    "near_null_tol", "zero_curvature_tol", "stationary_tol", "hessian_method",
    "fd_step", "n_slice_points", "slice_half_width", "warn",
)


def identifiability_report(
    model: _SGDLossModel, *args: Any, **kwargs: Any
) -> IdentifiabilityReport:
    """Identifiability report of an ``SGDFittableMixin`` model at its current parameters.

    ... (docstring: the report options are keyword-only and taken from ``kwargs``;
    what remains is data. With no data, the data of the last ``fit_sgd`` is
    reused; a model fitted by EM (``fit``) must be given its data, in the form
    ``_sgd_loss_fn`` takes -- for most models the arrays ``fit_sgd`` takes, for
    ``PlaceFieldModel`` the design matrix and spikes of ``_finalize_sgd``. To
    diagnose the *initial* point of a model that initialises inside ``fit_sgd``,
    call ``fit_sgd(..., num_steps=0)`` first. The loss is the total (not
    per-time-step) objective the model optimises, penalties included, so
    penalised parameters can look identified when the likelihood alone would
    not determine them.)
    """
    options = {name: kwargs.pop(name) for name in _REPORT_OPTIONS if name in kwargs}
    if args or kwargs:
        args, kwargs = model._prepare_sgd_data(*args, **kwargs)
    else:
        stored = getattr(model, "_sgd_data_", None)
        if stored is None:
            raise NotFittedError(
                f"{type(model).__name__}.identifiability_report needs data: call "
                "fit_sgd first (the report then reuses its data) or pass the data "
                "arguments that _sgd_loss_fn takes."
            )
        args, kwargs = stored
    model._check_sgd_initialized()
    params, spec = model._build_param_spec()

    def loss_fn(p: dict) -> Array:
        return model._sgd_loss_fn(p, *args, **kwargs)

    return identifiability_report_from_loss(loss_fn, params, spec, **options)
```

In `sgd_fitting.py`, inside `SGDFittableMixin` (after `fit_sgd`):

```python
    def identifiability_report(self, *args: Any, **kwargs: Any) -> "IdentifiabilityReport":
        """Observed-information identifiability report at the current parameters.

        See :func:`state_space_practice.identifiability.identifiability_report`
        for the data arguments, the keyword options and the caveats.
        """
        from state_space_practice.identifiability import identifiability_report

        return identifiability_report(self, *args, **kwargs)
```

(The import is local so `sgd_fitting`, imported by every model module, does not import
the diagnostics module eagerly; `IdentifiabilityReport` goes under `TYPE_CHECKING`.)
And at `sgd_fitting.py:544`:

```python
        args, kwargs = self._prepare_sgd_data(*args, **kwargs)
        # The prepared data is exactly what _sgd_loss_fn consumes; keep it so
        # identifiability_report() can re-evaluate the objective without the
        # caller re-supplying (and re-preparing) the data.
        self._sgd_data_ = (args, kwargs)
```

## The Hessian is only meaningful at a stationary point

If the loss is exactly invariant along a one-parameter family `θ(c)` with `θ(1) = θ*`,
differentiating `f(θ(c)) = const` twice at `c = 1` gives

    gen' H gen = − g · θ''(1),      gen = θ'(1).

So `H gen = 0` (a null eigenvector) follows only where the gradient `g` vanishes, or
where the orbit is a straight line (`θ'' = 0`, e.g. a linear reparameterisation
degeneracy). For the multiplicative symmetries of state-space models — latent scale
`(β/c, c²Q)`, oscillator phase `(H R(−φ), R P0 R', R m0)` — the orbit is curved.

Measured in the planning session on the spike-only coupling loss (Bernoulli Laplace-EKF
marginal, `T = 300`): `f(θ(c)) − f(θ)` ∈ {0, ±3e-14, ±6e-14} for `c ∈ {0.5, 0.9, 1.1,
2}`; `g·gen/(|g||gen|) = 1.4e-10`; but at the truth (`|g| = 10.8`) `‖H gen‖/‖H‖ = 2.7e-3`
and `|cos(v_min, gen)| = 0.82`. At β = 0 (where `g = 2.6e-12`) the null eigenvalue was
`1.1e-11` against `λ_max = 242`.

Consequences encoded in the plan: the report's `is_stationary`/`scaled_gradient_max`;
the warning text notes non-stationarity; acceptance tests that assert a null eigenvector
first polish to `|g| ≲ 1e-8` ([polishing helper](#test-helper-polishing-to-a-stationary-point));
one test asserts the confound itself (the same loss at a non-stationary point shows
`|cos| < 0.95`), so the stationarity flag cannot be removed without a failing test.

## Why the observed Hessian suffices at initialisation

The findings note describes the joint estimator's trap: with logit `η = β·x` and the
initialisation `x = β = 0`, the Jacobian of `η` is zero so a *joint* (latent-as-parameter)
estimator has no first-order information. For the *marginal* objectives this library
optimises (the latent integrated out by the Laplace-EKF), the same trap appears as:

- at `β = 0` the marginal likelihood does not depend on the latent's dynamics at all
  (`p(y | β = 0, Q) = p(y | baseline)`), so the gradient and the whole Hessian row of
  every dynamics parameter are exactly zero → `zero_curvature_parameters`;
- the gradient with respect to `β` is `Σ_t (y_t − μ_t) x̂_t'` with the predicted latent
  `x̂_t = A^t m0 = 0`, so it is exactly zero too → the point is *stationary*: Adam with
  zero gradients never moves, `fit_sgd` reports a constant LL and "converges";
- the curvature along `β` is negative when the data are coupled (β = 0 is a maximum of
  the NLL along β) → `n_negative_eigenvalues > 0`: a saddle, not a fit.

All three are read off the observed Hessian and gradient; no expected (Gauss–Newton)
information is needed. Measured at β = 0: `|g| = 2.6e-12`, `H_QQ = 1.1e-11`, raw
eigenvalues `[−242, −242, −79, −79, 1e-11, 6.2, 6.2]`.

## Test losses: spike-only and LFP-anchored coupling

Lives in `tests/test_identifiability.py` (phase 1). Both losses are the Laplace-EKF
marginal negative log-likelihood of the Bernoulli-logit coupling model of
`coupling_model.py`; the anchored one additionally conditions on the LFP each step with
the exact Gaussian update (`lfp_t = x_t + N(0, σ² I)`, `H = I`, `σ²` fixed — the
observation that pins the latent's scale, `coupling_model.py:58-64`).

```python
import scipy.optimize
from jax.flatten_util import ravel_pytree

from state_space_practice.coupling_model import (
    CouplingModelParams, build_transition, interleave_coupling,
)
from state_space_practice.kalman import kalman_measurement_update
from state_space_practice.parameter_transforms import (
    POSITIVE, UNCONSTRAINED, transform_to_constrained, transform_to_unconstrained,
)
from state_space_practice.point_process_kalman import (
    BERNOULLI_LOGIT_FAMILY, glm_laplace_update,
)
from state_space_practice.simulate_coupling import simulate_coupling

COUPLING_SPEC = {
    "beta_real": UNCONSTRAINED,
    "beta_imag": UNCONSTRAINED,
    "process_noise_var": POSITIVE,
}


def coupling_params(n_neurons=3, base_rate=0.05, lfp_noise_var=0.25):
    """One 8 Hz band; neurons 0/2 couple in-phase/anti-phase, neuron 1 in quadrature."""
    beta_real = jnp.zeros((n_neurons, 1)).at[0, 0].set(1.5).at[2 % n_neurons, 0].set(-1.0)
    beta_imag = jnp.zeros((n_neurons, 1)).at[1 % n_neurons, 0].set(1.5)
    return CouplingModelParams(
        osc_frequencies=jnp.array([8.0]),
        osc_decay=jnp.array([0.98]),
        process_noise_var=jnp.array([1.0 - 0.98**2]),   # stationary variance 1
        beta_real=beta_real,
        beta_imag=beta_imag,
        baseline=jnp.full((n_neurons,), float(np.log(base_rate / (1 - base_rate)))),
        dt=1e-3,
        lfp_noise_var=lfp_noise_var,
    )


def coupling_marginal_nll(base, spikes, lfp):
    """Return ``loss(theta)`` over ``COUPLING_SPEC`` keys; ``lfp=None`` -> spike-only."""

    def loss(theta):
        p = base._replace(
            beta_real=theta["beta_real"],
            beta_imag=theta["beta_imag"],
            process_noise_var=theta["process_noise_var"],
        )
        A, Q = build_transition(p)
        B = interleave_coupling(p.beta_real, p.beta_imag)            # (S, 2J)
        n_latent = A.shape[0]
        P0 = jnp.diag(jnp.repeat(p.process_noise_var / (1.0 - p.osc_decay**2), 2))
        H = jnp.eye(n_latent)
        R = p.lfp_noise_var * H

        def step(carry, obs):
            m, P = carry
            y, field = obs
            m_pred, P_pred = A @ m, A @ P @ A.T + Q
            ll = jnp.zeros(())
            if lfp is not None:
                m_pred, P_pred, ll_field = kalman_measurement_update(m_pred, P_pred, field, H, R)
                ll = ll + ll_field
            m_new, P_new, evidence = glm_laplace_update(
                m_pred, P_pred, y, lambda x: p.baseline + B @ x,
                BERNOULLI_LOGIT_FAMILY, max_newton_iter=3,
            )
            return (m_new, P_new), ll + evidence

        fields = lfp if lfp is not None else jnp.zeros((spikes.shape[0], n_latent))
        _, lls = jax.lax.scan(step, (jnp.zeros(n_latent), P0), (spikes, fields))
        return -jnp.sum(lls)

    return loss


def coupling_theta(params, *, zero_coupling=False):
    zeros = jnp.zeros_like(params.beta_real)
    return {
        "beta_real": zeros if zero_coupling else params.beta_real,
        "beta_imag": zeros if zero_coupling else params.beta_imag,
        "process_noise_var": params.process_noise_var,
    }


def scale_latent(theta, c):
    """The exact symmetry of the spike-only marginal: x -> c x."""
    return {
        "beta_real": theta["beta_real"] / c,
        "beta_imag": theta["beta_imag"] / c,
        "process_noise_var": theta["process_noise_var"] * c**2,
    }
```

`kalman_measurement_update(prior_mean, prior_cov, obs, measurement_matrix,
measurement_cov)` returns `(mean, cov, log_likelihood)` (`kalman.py:427-433`);
`glm_laplace_update(...)` returns `(mean, cov, log_evidence)` when
`return_line_search_failures=False` (`point_process_kalman.py:1333-1345, 1372-1376`).
`build_transition` calls `require_coupling_x64()` (a host check, fine under tracing) and
builds `A`, `Q` from `p` (`coupling_model.py:262-282`); `validate_coupling_params` must
*not* be called inside the loss (host-side, needs concrete values) — call it once on the
fixed `base` params.

Scale invariance and spectrum at β = 0 were verified in the planning session with exactly
this code (see the two sections above).

## Test helper: polishing to a stationary point

```python
def polish_to_stationary(loss_fn, params, spec, *, gtol=1e-9, maxiter=2000):
    """L-BFGS in unconstrained coordinates; returns constrained params at the optimum.

    Unconstrained coordinates keep positive / PSD parameters in their domain during
    the line search. ``gtol`` is scipy's projected-gradient tolerance in those
    coordinates; the report's ``scaled_gradient_max`` is checked afterwards.
    """
    unc0 = transform_to_unconstrained(params, spec)
    flat0, unravel = ravel_pytree(unc0)

    def f(u):
        return loss_fn(transform_to_constrained(unravel(u), spec))

    value_and_grad = jax.jit(jax.value_and_grad(f))

    def fun(x):
        value, grad = value_and_grad(jnp.asarray(x))
        return float(value), np.asarray(grad, dtype=float)

    result = scipy.optimize.minimize(
        fun, np.asarray(flat0, dtype=float), jac=True, method="L-BFGS-B",
        options={"gtol": gtol, "maxiter": maxiter},
    )
    assert np.isfinite(result.fun), result
    return transform_to_constrained(unravel(jnp.asarray(result.x)), spec)
```

For a model (phase 2): `params, spec = model._build_param_spec()`; `loss_fn = lambda p:
model._sgd_loss_fn(p, *args, **kwargs)` with `(args, kwargs)` from `capture_sgd_problem`
or `model._sgd_data_`; then `model._store_sgd_params(polished)` and
`model.identifiability_report()`. Note `_store_sgd_params` may re-stabilise (DIM zeroes
the coupling diagonal, `oscillator_models.py:2321-2325`; the Hamiltonian base
re-stabilises `Q`, `hamiltonian_core.py:702-703`), so re-check `report.is_stationary`
rather than assuming it.

## Test helper: symmetry generators by finite difference

The analytic generator of a symmetry `S_c(θ)` at the identity is
`gen = d/dc S_c(θ)|_{c=1}`; build it numerically in the report's own coordinates so PSD
(`vech`) and stochastic-row charts are handled by the same code that names the report:

```python
def symmetry_generator(symmetry, theta, spec, eps=1e-6):
    """Unit vector d/dc symmetry(theta, c) at c = 1 in scientific coordinates."""
    from state_space_practice.identifiability import scientific_coordinates

    plus = scientific_coordinates(symmetry(theta, 1.0 + eps), spec).values
    minus = scientific_coordinates(symmetry(theta, 1.0 - eps), spec).values
    gen = np.asarray(plus - minus) / (2 * eps)
    return gen / np.linalg.norm(gen)


def cosine_with_raw_null_space(report, gen, n_directions):
    """|projection| of ``gen`` onto the span of the ``n_directions`` smallest raw eigenvectors."""
    V = report.eigenvectors[:, :n_directions]
    return float(np.linalg.norm(V.T @ gen))
```

For the `CommonOscillatorModel` phase symmetry (phase 2), `x → R(φ) x` per oscillator
with `R(φ)` the `2×2` rotation; the *joint* rotation of all oscillators by the same φ:

```python
def rotate_oscillator_phase(params, phi, n_oscillators):
    c, s = np.cos(phi), np.sin(phi)
    block = jnp.array([[c, -s], [s, c]])
    R = jax.scipy.linalg.block_diag(*([block] * n_oscillators))       # (2n, 2n)
    out = dict(params)
    if "measurement_matrix" in out:   # (n_sources, 2n, n_states): y = H x = (H R') (R x)
        out["measurement_matrix"] = jnp.einsum("slj,kl->skj", params["measurement_matrix"], R)
    if "init_mean" in out:            # (2n, n_states)
        out["init_mean"] = R @ params["init_mean"]
    for key in params:
        if key.startswith("init_cov_"):
            out[key] = R @ params[key] @ R.T
    return out
```

with the symmetry passed to `symmetry_generator` as `lambda theta, c:
rotate_oscillator_phase(theta, c - 1.0, n_oscillators)`. `A` (block rotation-decay,
`construct_common_oscillator_transition_matrix`) and `Q` (isotropic per oscillator)
commute with `R` and are not SGD parameters of COM (`oscillator_models.py:1286-1288`
sets `update_process_cov = False`; `:1420-1443` spec; `:1444-1462` loss). Each
oscillator has its own phase, so there are `n_oscillators` independent null directions
and the joint generator lies in their span.

## Literature

Each entry says which claim in this plan it supports.

- Auger-Méthé, M., Field, C., Albertsen, C. M., Derocher, A. E., Lewis, M. A., Jonsen,
  I. D., Mills Flemming, J. (2016). *State-space models' dirty little secrets: even
  simple linear Gaussian models can have estimation problems.* Scientific Reports 6,
  26677. — Motivates the linear-Gaussian SSM tests: parameter estimation in simple
  linear-Gaussian SSMs can fail (ridges / weak identifiability), so a diagnostic is
  needed even outside the point-process models.
- Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U.,
  Timmer, J. (2009). *Structural and practical identifiability analysis of partially
  observed dynamical models by exploiting the profile likelihood.* Bioinformatics
  25(15), 1923–1929. — The profile likelihood is the reference method for practical
  identifiability; this is why the report's linear cut is named a *slice* and the profile
  is Open Question 4.
- Transtrum, M. K., Machta, B. B., Brown, K. S., Daniels, B. C., Myers, C. R., Sethna,
  J. P. (2015). *Perspective: Sloppiness and emergent theories in physics, biology, and
  beyond.* J. Chem. Phys. 143, 010901. — Hessian eigenvalue spectra spanning many
  decades ("sloppiness"); the `sloppiness_ratio` field and the log-parameter convention
  deferred in Open Question 3.
- Brun, R., Reichert, P., Künsch, H. R. (2001). *Practical identifiability analysis of
  large environmental simulation models.* Water Resources Research 37(4), 1015–1030. —
  Column-normalised sensitivity/information matrix and the collinearity index
  `1/sqrt(λ_min)`; the basis for deciding near-null directions on `D^{-1/2} H D^{-1/2}`.
- Viallefont, A., Lebreton, J.-D., Reboulet, A.-M., Gory, G. (1998). *Parameter
  identifiability and model selection in capture-recapture models: a numerical
  approach.* Biometrical Journal 40(3), 313–325. — The numerical Hessian-rank check at
  the MLE for parameter redundancy; the origin of "count near-zero eigenvalues of the
  observed information".
- Cole, D. J. (2020). *Parameter Redundancy and Identifiability.* Chapman & Hall/CRC. —
  Reference text on parameter redundancy, including the Hessian method above and its
  limits (rank at one point, tolerance choice), which is why the tolerances are options.
