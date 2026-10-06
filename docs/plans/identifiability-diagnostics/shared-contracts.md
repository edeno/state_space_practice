# Shared contracts

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Contracts used by both phases. Each appears exactly once, here.

- [Public API of `state_space_practice.identifiability`](#public-api)
- [Scientific coordinates](#scientific-coordinates)
- [`IdentifiabilityReport` and `NullDirection`](#identifiabilityreport)
- [Report options and defaults](#report-options)
- [Mixin data contract: `_sgd_data_`](#mixin-data-contract)

## Public API

Module `src/state_space_practice/identifiability.py` (phase 1 unless noted).

```python
def scientific_coordinates(
    params: Mapping[str, Any],
    transforms: Mapping[str, ParameterTransform] | None = None,
) -> ScientificCoordinates: ...

def fisher_information(
    loss_fn: Callable[[dict], ArrayLike],
    params: Mapping[str, Any],
    transforms: Mapping[str, ParameterTransform] | None = None,
    *,
    hessian_method: HessianMethod = "auto",
    fd_step: float = 1e-3,
) -> Array: ...
    # Observed information: the symmetrised Hessian of ``loss_fn`` (a negative
    # log-likelihood) with respect to the scientific coordinates of the trainable
    # parameters, shape (P, P). Row/column ``i`` is ``scientific_coordinates(params,
    # transforms).names[i]``. Equals the expected Fisher information only in
    # expectation / at the MLE of an exponential-family model; the name follows the
    # brief and the docstring says this.

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
) -> IdentifiabilityReport: ...

def identifiability_report(model, *data, **kwargs) -> IdentifiabilityReport: ...   # phase 2
    # ``model`` is any SGDFittableMixin-protocol object. The report options above
    # are popped from ``kwargs``; the remaining ``(data, kwargs)`` are the loss's
    # data arguments in the form ``_sgd_loss_fn`` receives (they go through
    # ``model._prepare_sgd_data``). With no data, ``model._sgd_data_`` (see below)
    # is used; if absent, ``NotFittedError``.

HessianMethod = Literal["auto", "autodiff", "finite_difference"]
```

`loss_fn` takes a **constrained** parameter dict (the same contract as
`_sgd_loss_fn(params, ...)` with the data closed over) and returns a scalar. It receives
*all* keys of `params` — trainable ones rebuilt from the coordinates, frozen ones
verbatim.

`SGDFittableMixin.identifiability_report(self, *data, **kwargs)` (phase 2, in
`sgd_fitting.py`) is a one-line delegation to `identifiability_report(self, *data,
**kwargs)`.

Top-level lazy exports (phase 2): `identifiability_report`,
`identifiability_report_from_loss`, `fisher_information`, `IdentifiabilityReport`.

## Scientific coordinates

The coordinates the Hessian is taken in. Rules (do not weaken — the tests in phase 1 pin
each one):

| `transforms[key]` | Chart (value → coordinates) | Inverse | Coordinate names |
| --- | --- | --- | --- |
| `None`, `UNCONSTRAINED`, `POSITIVE`, `UNIT_INTERVAL`, `positive_capped(...)`, any unknown transform | identity, every leaf of the (possibly nested) pytree value | identity | `key[i,j,...]`; nested leaves `key.sub.leaf[i]`; scalar `key` |
| `PSD_MATRIX` (or `frozen(PSD_MATRIX)` when trainable) | `vech`: `value[jnp.tril_indices(n)]` — `n(n+1)/2` entries | `L + L.T − diag(diag(L))` | `key[i,j]` for `i ≥ j`, in `tril_indices` order |
| `STOCHASTIC_ROW` | `value[..., :-1]` | `concat([free, 1 − free.sum(-1, keepdims=True)], -1)` | `key[i,j]` for `j < K−1` (or `key[j]` for a vector) |
| any transform with `trainable=False` | not a coordinate; held fixed and passed to the loss | — | — |

Detection is by identity of the constrained map: `transform.to_constrained is
PSD_MATRIX.to_constrained` / `is STOCHASTIC_ROW.to_constrained` (survives `frozen(...)`,
which copies the callables — `parameter_transforms.py:255-261`).

Additional invariants:

- Keys are processed in **sorted order** and leaves in `jax.tree_util` flatten order,
  which is exactly the order `jax.flatten_util.ravel_pytree` produces, so `names[i]`
  labels `values[i]`.
- Zero-size leaves (e.g. a `STOCHASTIC_ROW` `(1, 1)` matrix → `(1, 0)` free
  coordinates) contribute no coordinates and no names.
- A `PSD_MATRIX` / `STOCHASTIC_ROW` key whose value is not a single array raises
  `ValueError`.
- All coordinates are float (`jnp.result_type(x, 1.0)`); with x64 enabled, float64.
- `P == 0` raises `ValueError("No trainable parameters to analyse.")` (mirrors
  `sgd_fitting.py:549-550`).

```python
@dataclass(frozen=True)
class ScientificCoordinates:
    names: tuple[str, ...]                 # length P
    values: Array                          # shape (P,)
    unravel: Callable[[Array], dict]       # (P,) -> full constrained params dict, frozen keys included
```

## IdentifiabilityReport

```python
@dataclass(frozen=True)
class NullDirection:
    kind: Literal["zero_curvature", "collinear"]
    eigenvalue: float          # raw diagonal H_ii for zero_curvature; normalised eigenvalue for collinear
    coefficients: dict[str, float]
    # collinear: entries of the unit eigenvector of the normalised matrix, i.e. the
    # direction in units of each coordinate's own curvature scale 1/d_i (sqrt(abs(H_ii)), with a row-scale fallback at zero); only
    # entries with |c| >= 0.05 are kept. zero_curvature: {name: 1.0}.

@dataclass(frozen=True)
class IdentifiabilityReport:
    parameter_names: tuple[str, ...]                 # P
    parameter_values: npt.NDArray[np.float64]        # (P,)
    loss: float
    gradient: npt.NDArray[np.float64]                # (P,)
    gradient_norm: float
    scaled_gradient_max: float                       # max_i |g_i| / d_i over curvature-bearing coordinates; same fallback as spectrum
    is_stationary: bool                              # scaled_gradient_max < stationary_tol and flat coordinates have ~zero gradient
    hessian: npt.NDArray[np.float64]                 # (P, P) symmetrised, raw scientific coordinates
    hessian_method_used: Literal["autodiff", "finite_difference"]
    eigenvalues: npt.NDArray[np.float64]             # (P,) ascending, raw
    eigenvectors: npt.NDArray[np.float64]            # (P, P) columns, raw
    zero_curvature_parameters: tuple[str, ...]
    normalized_eigenvalues: npt.NDArray[np.float64]  # (P_active,) ascending, of D^{-1/2} H D^{-1/2} over non-flat coordinates
    normalized_eigenvectors: npt.NDArray[np.float64] # (P_active, P_active)
    active_parameter_names: tuple[str, ...]          # the P_active names, in normalised-matrix order
    condition_number: float                          # normalised lambda_max / lambda_min; inf if lambda_min <= 0
    sloppiness_ratio: float                          # raw lambda_max / lambda_min; inf if lambda_min <= 0 (unit-dependent)
    n_negative_eigenvalues: int
    near_null_directions: tuple[NullDirection, ...]  # zero_curvature entries first, then collinear
    wald_standard_errors: dict[str, float] | None    # sqrt(diag(H^{-1})) when Cholesky of H succeeds, else None
    worst_direction: npt.NDArray[np.float64]         # (P,) raw-coordinate step delta used by the slice
    worst_direction_slice: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None
    # (t, loss(theta + t * worst_direction)) for t in linspace(-1, 1, n_slice_points); None if n_slice_points == 0
    near_null_tol: float
    zero_curvature_tol: float
    stationary_tol: float

    @property
    def is_identifiable(self) -> bool:
        return not self.near_null_directions   # no zero-curvature coordinate and no collinear near-null direction

    @property
    def is_local_minimum(self) -> bool:
        return self.is_stationary and self.n_negative_eigenvalues == 0

    def __str__(self) -> str: ...   # see designs.md#report-text
```

Semantics that must hold (tests pin them):

- `is_identifiable` is the *only* verdict the warning is keyed on; it is false whenever
  `near_null_directions` is non-empty.
- Negative eigenvalues are counted on a positive diagonal congruence of the **full**
  Hessian, before any flat-row removal. A zero diagonal with a nonzero row uses
  that row's maximum absolute entry as its squared scale; an exactly zero row
  uses scale 1. Flat annotations require the entire row to be small.
- Sylvester's law preserves exact inertia under this congruence; numerical counts
  also depend on tolerance. The active correlation spectrum is unit-invariant
  for nonzero diagonals and an unchanged active set. Raw row thresholds and
  zero-diagonal fallback scales remain unit-dependent and must be documented.
- `wald_standard_errors` is `None` exactly when `np.linalg.cholesky(hessian)` fails; when
  near-null directions exist and the Cholesky still succeeds, the SEs are returned
  (they will be large along those directions) and `__str__` says so.
- The report holds NumPy arrays and Python floats only (no tracers, no JAX arrays), so it
  prints and pickles.

## Report options

| Option | Default | Meaning |
| --- | --- | --- |
| `near_null_tol` | `1e-6` | active normalised `|λ| < near_null_tol · max(abs(λ))` → collinear near-null; negative count uses `λ < −near_null_tol · max(abs(λ))` on the full congruence |
| `zero_curvature_tol` | `1e-10` | raw `max_j(abs(H_ij)) ≤ zero_curvature_tol · max_jk(abs(H_jk))` → numerically flat row annotation; never excludes a row from the negative-curvature check |
| `stationary_tol` | `1e-2` | `is_stationary` iff `scaled_gradient_max < stationary_tol` and every flat coordinate has `|g_i| ≤ stationary_tol · max_j |g_j|` |
| `hessian_method` | `"auto"` | `"autodiff"`: `jax.hessian`, error propagates; `"finite_difference"`: Richardson central differences of `jax.grad`; `"auto"`: autodiff, falling back to FD only on the `pure_callback` JVP `ValueError` |
| `fd_step` | `1e-3` | relative FD step: `h_i = fd_step · max(|θ_i|, 1e-2)` |
| `n_slice_points` | `9` | loss evaluations along the worst direction; `0` disables the slice |
| `slice_half_width` | `0.5` | the largest relative move of any coordinate at `t = ±1` (relative to `max(|θ_i|, 1e-2)`) |
| `warn` | `True` | emit `StateSpaceWarning` when `not is_identifiable` |

## Mixin data contract

Phase 2, `sgd_fitting.py`. Immediately after `args, kwargs = self._prepare_sgd_data(*args,
**kwargs)` (`sgd_fitting.py:544`), `SGDFittableMixin.fit_sgd` sets

```python
self._sgd_data_ = (args, kwargs)
```

— the prepared positional and keyword data that `_sgd_loss_fn` and `_finalize_sgd`
receive. It is set before `_check_sgd_initialized` / the "No learnable parameters" check
so a failed fit still leaves the data available for diagnosis. `identifiability_report`
reads it via `getattr(model, "_sgd_data_", None)`. No other code reads or clears it;
`_store_sgd_params`/`_finalize_sgd` do not touch it.
