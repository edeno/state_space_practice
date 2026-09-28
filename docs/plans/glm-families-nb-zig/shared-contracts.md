# Shared contracts

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Contracts referenced by both phases. Each appears once; phases link by anchor.
"Do not weaken" clauses are the cross-phase invariants.

- [C1 — `GLMFamily` extended contract](#c1-glmfamily-extended-contract)
- [C2 — `family=` on the filter and smoother](#c2-family-on-the-filter-and-smoother)
- [C3 — `PointProcessModel` family kinds](#c3-pointprocessmodel-family-kinds)
- [C4 — Stacked-predictor layout](#c4-stacked-predictor-layout)

---

## C1 — `GLMFamily` extended contract

Defined at `src/state_space_practice/point_process_kalman.py:1215-1239`; the
optional fields are appended with defaults (reuse fields already added by masks/WoLF) so every existing constructor call
(`poisson_family` `:1242-1265`, `BERNOULLI_LOGIT_FAMILY` `:1292-1297`,
`hamiltonian_core.py:179-191`, `coupling_ekf.py:96-105`) is untouched.

```python
class GLMFamily(NamedTuple):
    mean: Callable[[Array], Array]                                  # eta (n_eta,) -> mu (n_eta,)
    fisher_weight: Callable[[Array, Array], Array]                  # (eta, mu) -> w (n_eta,), w >= 0, EXPECTED information per predictor
    loglik_plugin: Callable[[Array, Array, Array], Array]           # (y, eta, mu) -> scalar, x-dependent part (line search)
    loglik_normalized: Callable[[Array, Array, Array], Array]       # (y, eta, mu) -> scalar, full log-likelihood (marginal LL)
    score: Callable[[Array, Array, Array], Array] | None = None     # Phase 1: (y, eta, mu) -> d loglik / d eta (n_eta,); None => y - mu
    loglik_per_obs: Callable[[Array, Array, Array], Array] | None = None  # normalized terms (n_obs,), shared with masks/WoLF
    validate_observations: Callable[[ArrayLike, str], None] | None = None  # Phase 2: host-side check of y; None => validate_count_array
```

Semantics and invariants:

1. `n_eta` may exceed `n_obs` (multi-predictor families, C4); `y` is always
   `(n_obs,)`. The update never compares the two lengths.
2. `fisher_weight` is the **expected** negative second derivative of the
   log-likelihood with respect to each predictor (Fisher scoring). The
   information identity `w = E_y[score^2]` holds for every family; the
   canonical-link identity `w = d mean / d eta` holds only for
   `poisson_family` / `BERNOULLI_LOGIT_FAMILY` and its test must not be
   parametrised over the new families.
3. `loglik_plugin` must contain every `x`-dependent term of
   `loglik_normalized` (its `eta`-gradient equals `score`); it may drop terms
   that depend only on `y` or on fixed family parameters.
4. **Do not weaken:** with `score is None` and `validate_observations is None`
   the traced program of `glm_laplace_update`, the filters and
   `PointProcessModel` is byte-identical to today's. Guarded by
   `tests/test_glm_laplace.py:36-68` (parity) and the `point_process_glm` /
   `common_oscillator_pp` / `directed_influence_pp` cases in
   `tests/test_em_golden_regression.py`.
5. `loglik_per_obs` returns normalized terms of shape `(n_obs,)`, whose sum
   equals `loglik_normalized`; every built-in family provides it. Masked calls
   use `where(mask, terms, 0).sum()` for both line search and evidence, keeping
   scalar callback signatures unchanged. Unmasked calls do not require the
   optional callback; custom families lacking it raise `ValueError` only when
   a mask is requested. Predictor-major masks tile over predictor groups for
   score/information, but remain observation-sized for likelihood terms.
6. Families are static under `jit`: every field is a function (hashable by
   identity). Never put arrays in fields; close over them. Each factory call is
   a new cache key — hoist the family out of loops.

---

## C2 — `family=` on the filter and smoother

`stochastic_point_process_filter(..., family: GLMFamily | None = None)` and
`stochastic_point_process_smoother(..., family: GLMFamily | None = None)`
(signatures at `point_process_kalman.py:1467-1484` and `:2392-2410`).

1. `family=None` (default) runs the original Poisson update
   `_point_process_laplace_update` (`:927-1212`) unchanged, on both the dense
   and the block-diagonal path.
2. `family` given ⇒ the dense scan calls `glm_laplace_update` with that family
   (design [§4](designs.md#4-threading-family-through-the-filter-and-smoother)).
   The block-diagonal path is Poisson-only: `use_block_dispatch` requires
   `family is None`, so `block_n_neurons`/`block_size` with a family silently
   take the dense path (documented; no error).
3. `max_log_count` applies only to `family=None`; passing a non-default value
   together with a family emits a `StateSpaceWarning`. Families own their
   overflow ceilings (`negative_binomial_family(..., max_log_count=)`).
4. Observation validation at the entry points (`_validate_public_inputs`,
   `:96-145`) uses `family.validate_observations` when set, else
   `validate_count_array`. With masks, zero-fill masked observations first and
   pass the sanitized array to this same family-specific validator.
5. Inside `_stochastic_point_process_filter_impl` (`:1737-1866`) `family` is a
   static argument; a family closing over a traced array (SGD) is allowed and
   is traced once per outer compile.
6. **Do not weaken:** `family=None` must never route through
   `glm_laplace_update` in this plan (retiring the legacy update is a separate
   decision — [overview Open Question 2](overview.md#open-questions)).

---

## C3 — `PointProcessModel` family kinds

`PointProcessModel.__init__` (`point_process_kalman.py:2993-3006`) gains
keyword arguments; attributes and behaviour per kind:

| `family=` | Extra kwargs | `family_kind` | `_family` passed to the smoother | SGD-learnable family params |
|---|---|---|---|---|
| `"poisson"` (default) | none allowed | `"poisson"` | `None` (legacy path) | — |
| `"negative_binomial"` | `dispersion` (required; shape `()` or `(n_neurons,)`, `> 0`), `update_dispersion=False` | `"negative_binomial"` | `negative_binomial_family(dt, dispersion)` | `dispersion` via `POSITIVE` when `update_dispersion` |
| `"zero_inflated_gamma"` (Phase 2) | `gamma_shape` (required, `> 0`, `()`/`(n_neurons,)`), `gamma_loc=0.0` (`>= 0`) | `"zero_inflated_gamma"` | `zero_inflated_gamma_family(gamma_shape, gamma_loc)` | none |
| a `GLMFamily` instance | none allowed | `"custom"` | the instance | none |

1. Kwargs that do not belong to the chosen kind raise `ValueError` at
   construction; a per-neuron array whose length disagrees with the neuron
   axis raises `ValueError` at `fit`/`fit_sgd`.
2. `_family` is rebuilt only when its parameters change
   (`_store_sgd_params`), so EM iterations reuse one compiled filter.
3. `log_intensity_func` keeps its name; for non-Poisson kinds it is the
   linear-predictor function `eta_func(design_t, x)` and its output length is
   `n_neurons * n_eta_per_neuron` (C4).
4. `dispersion` is **fixed under EM** (`fit`); it is learned only by `fit_sgd`
   when `update_dispersion=True`.
5. **Do not weaken:** the default construction and `fit`/`fit_sgd` behaviour is
   unchanged (golden case `tests/test_em_golden_regression.py:239-248`).

---

## C4 — Stacked-predictor layout

For a family with `n_eta_per_neuron = k > 1` predictors per observation
(ZIG: `k = 2`), the single flat predictor vector is ordered **predictor-major**:

```
eta = concat([eta_1 (n_obs,), eta_2 (n_obs,), ..., eta_k (n_obs,)])
mean(eta), fisher_weight(eta, mu), score(y, eta, mu)  use the same layout
```

For ZIG: `eta[:n_obs] = logit q`, `eta[n_obs:] = log theta` (gamma scale), and
`mean(eta) = concat([q, theta])`.

With the default linear `log_conditional_intensity` (`point_process_kalman.py:523-547`)
this means a design matrix of shape `(n_time, k * n_neurons, n_state)` whose
first `n_neurons` rows drive `eta_1`, the next `n_neurons` rows `eta_2`, etc.
Simulators and tests that use an affine predictor put the intercept in column
0 (`tests/test_oracle_point_process.py:76-78` style) and the same row order.

**Do not weaken:** families with `k = 1` (`n_eta == n_obs`) are unaffected; the
layout is a convention between a family and its callers, not something the
update inspects.
