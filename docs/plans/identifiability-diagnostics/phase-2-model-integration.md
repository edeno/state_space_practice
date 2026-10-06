# Phase 2 — `model.identifiability_report()` for every `SGDFittableMixin` model

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#model-entry-point)

Wires the phase-1 report to the models: `fit_sgd` keeps the prepared data, the mixin
gains `identifiability_report()`, the four public names are exported at package level,
and README/CHANGELOG describe the feature. The model-level acceptance tests cover every
SGD loss in the library, an exact phase symmetry of the oscillator models, and the
spike-only latent-oscillator trap at the model level.

**Inputs to read first:**

- `src/state_space_practice/identifiability.py` (from phase 1) — extend with
  `identifiability_report(model, ...)`; read `identifiability_report_from_loss` and
  `scientific_coordinates` first.
- `src/state_space_practice/sgd_fitting.py:19-37` (imports), `:354-373` (class
  docstring — add the new method to the listed API), `:384-397` (`_prepare_sgd_data`
  default), `:506-560` (`fit_sgd` up to the frozen split; the insertion point is `:544`),
  `:703-721` (why parameters are re-read from `_build_param_spec()` after a fit).
- How each public `fit_sgd` prepares data before `super()` — this is why the stored
  prepared data is the only model-agnostic source: `place_field_model.py:1478-1534`
  (position → design matrix, warm start, block detection),
  `oscillator_models.py:1146-1190` (`_initialize_parameters` unless `skip_init`,
  `_sgd_n_time`), `switching_point_process.py:3313-3359` (initialises on first call),
  `multinomial_choice.py:954-1001` (`_prepare_choices`), `covariate_choice.py:805-843`
  (`_bind_covariates` sets `_covariates`/`_obs_covariates`), `hamiltonian_core.py:674-689`
  (`_prepare_sgd_data` returns `{"use_filter", "l2_reg"}` kwargs).
- `src/state_space_practice/oscillator_models.py:1286-1288` (COM fixes `Q`, frees `H`),
  `:1420-1443` (COM spec), `:1444-1462` (COM loss: `switching_kalman_filter` with `H`,
  `Z`, `m0`, `R`, `P0` from params), `:2169-2197, 2253-2313` (DIM spec/loss — contains
  the spectral-radius callback), `:2315-2329` (DIM store re-zeroes the diagonal).
- `src/state_space_practice/switching_point_process.py:147-160` (`SpikeObsParams`),
  `:192-200` (`_linear_log_intensity`: `log λ = baseline + weights @ x`), `:3379-3409`
  (shared spec: `spike_baseline`, `spike_weights`, `init_mean`, `init_cov_j`),
  `:3411-3470` (loss; L2 at `:3455-3457`, baseline prior at `:3463-3468`), `:4011-4039`
  (`A_blocks_j` scaled-rotation params, `Q_j`), `:3610` (`SwitchingSpikeOscillatorModel.__init__`,
  `spike_weight_l2=100.0` default, `update_discrete_transition_matrix`).
- `src/state_space_practice/point_process_models.py:1477-1494`
  (`DirectedInfluencePointProcessModel._sgd_loss_fn` also uses the callback → FD path).
- `src/state_space_practice/tests/test_gradients.py:38` (`pytestmark = slow`), `:47-69`
  (`capture_sgd_problem(monkeypatch, model, *data, **kwargs)`), `:185-448` (the 15
  model constructions and data sizes to reuse verbatim), `:402-430` (oscillator models
  from `simulate.scenarios`).
- `src/state_space_practice/tests/test_sgd_fitting.py:103-146` (`_ToyModel`: loss
  `(scale − target)² · 100`, `POSITIVE`; `_UnconstrainedToyModel`).
- `src/state_space_practice/tests/test_package.py:35-48` (the lazy-API tests are
  parametrised over `_LAZY_API`, so new names are covered automatically).
- `src/state_space_practice/__init__.py:33-52, 54-75, 77-101`; `README.md:41-58` (float64
  section — the new section goes right after it), `:60-67` (Package layout);
  `CHANGELOG.md` — the phase-1 entry.

**Contracts referenced:**

- [Public API](shared-contracts.md#public-api) — `identifiability_report(model, *data,
  **kwargs)` and the mixin delegation.
- [Mixin data contract](shared-contracts.md#mixin-data-contract) — `_sgd_data_` is set
  at `sgd_fitting.py:544` and read by `identifiability_report`; nothing else touches it.
- [Report options](shared-contracts.md#report-options) — popped from `kwargs` by name.

**Designs referenced:** [designs.md §7](designs.md#model-entry-point) (entry point and
mixin code), [§11](designs.md#test-helper-polishing-to-a-stationary-point) and
[§12](designs.md#test-helper-symmetry-generators-by-finite-difference) (test helpers,
`rotate_oscillator_phase`), [§9](designs.md#why-the-observed-hessian-suffices-at-initialisation).

## Tasks

- **Store the prepared data in `SGDFittableMixin.fit_sgd`**: after `sgd_fitting.py:544`
  add `self._sgd_data_ = (args, kwargs)` with the comment from
  [designs §7](designs.md#model-entry-point). Nothing else in the loop changes.
- **Add `identifiability_report(model, *args, **kwargs)` and `_SGDLossModel` to
  `identifiability.py`** exactly as in [designs §7](designs.md#model-entry-point); add
  `"identifiability_report"` to the module `__all__`. `NotFittedError` when neither
  data nor `_sgd_data_` is available; the message names both remedies.
- **Add `SGDFittableMixin.identifiability_report`** (`sgd_fitting.py`, after `fit_sgd`)
  as the one-line delegation in [designs §7](designs.md#model-entry-point), with a
  local import and a `TYPE_CHECKING` import of `IdentifiabilityReport` for the return
  annotation. Extend the class docstring (`sgd_fitting.py:354-373`) with one sentence
  on the method and the module docstring list (`:7-16`) with the stored data. Keep
  `sgd_fitting.py` ruff-clean; it is not in the mypy file list and need not be added.
- **Export at package level**: in `__init__.py` add to `_LAZY_API`
  `"identifiability_report": "identifiability"`, `"identifiability_report_from_loss":
  "identifiability"`, `"fisher_information": "identifiability"`,
  `"IdentifiabilityReport": "identifiability"`; add the four names to `__all__` and the
  `TYPE_CHECKING` block. `test_package.py` covers them automatically (do not add
  module-specific tests there).
- **README**: after the float64 section (`README.md:41-58`) add a short section
  "Checking a fit is identifiable" — one paragraph and one code block:

  ```python
  model.fit_sgd(observations, num_steps=500)
  report = model.identifiability_report()   # reuses the data fit_sgd prepared
  print(report)                              # named near-null directions, Wald SEs, stationarity
  if not report.is_identifiable:
      ...  # do not interpret parameters along report.near_null_directions
  ```

  plus one sentence each on: the report is meaningful at a stationary point
  (`report.is_stationary`); penalties in the objective count as information; for a
  hand-written loss use `identifiability_report_from_loss(loss_fn, params, transforms)`.
  Add `identifiability_report` to the entry-point list in Package layout
  (`README.md:62-66`).
- **CHANGELOG**: extend the phase-1 entry with: `SGDFittableMixin.identifiability_report()`
  and `identifiability.identifiability_report(model, ...)` (reuse the data of the last
  `fit_sgd` or pass the loss's data; `fit_sgd` now keeps the prepared data on the model
  as `_sgd_data_`; `DirectedInfluenceModel` / `DirectedInfluencePointProcessModel` use
  the finite-difference Hessian because their stability scale has no second derivative);
  and the four top-level lazy exports.
- **Write `src/state_space_practice/tests/test_identifiability_models.py`** (module
  header as in `test_gradients.py`; import `capture_sgd_problem` from
  `state_space_practice.tests.test_gradients` and `polish_to_stationary`,
  `symmetry_generator`, `cosine_with_raw_null_space` from
  `state_space_practice.tests.test_identifiability`; `_ToyModel` from
  `state_space_practice.tests.test_sgd_fitting`). Contents per the validation slice.
  For the all-models test, build a `MODEL_PROBLEMS` list of `(id, factory)` pairs
  reproducing the 15 constructions of `test_gradients.py:185-448` (same sizes) so both
  files stay in step; `EXPECTED_FD = {"dim", "dim_pp"}` documents which losses need the
  finite-difference path.

## Deliberately not in this phase

- Per-model overrides of `identifiability_report` that accept the *public* data form
  (e.g. `PlaceFieldModel(position, spikes)`): the stored-data path covers the common
  case and the explicit path is documented as "the form `_sgd_loss_fn` takes". Revisit
  if a consumer plan needs to report on an EM-fitted `PlaceFieldModel` without a
  preceding `fit_sgd`.
- Changing any model's default penalties (`spike_weight_l2=100.0`) or spec.
- A second derivative for `differentiable_spectral_radius` (would remove the FD path for
  DIM / DIM-PP) — out of scope; trigger: a consumer needs autodiff-precision nulls on DIM.
- Profile likelihood, log-parameter scales — [overview Open Questions 3–4](overview.md#open-questions).

## Validation slice

| Test | Asserts |
| --- | --- |
| `TestMixinReport::test_toy_model_explicit_data_matches_analytic_hessian` (fast) | `_ToyModel(scale=2.0).identifiability_report(jnp.array(5.0))`: `parameter_names == ("scale",)`, `hessian == [[200.0]]` to 1e-10, `gradient == [200·(2−5)]`, `wald_standard_errors["scale"] == 1/sqrt(200)`, `is_stationary is False`; at `scale=5.0`: `is_stationary`, `is_identifiable`, `hessian_method_used == "autodiff"` |
| `TestMixinReport::test_report_without_data_requires_fit` (fast) | `_ToyModel().identifiability_report()` raises `NotFittedError` whose message contains `fit_sgd` and `_sgd_loss_fn` |
| `TestMixinReport::test_stored_data_path_matches_explicit_path` (slow: calls `fit_sgd`) | after `fit_sgd(jnp.array(5.0), num_steps=5)`: `_sgd_data_ == ((array(5.0),), {})`; `identifiability_report()` and `identifiability_report(jnp.array(5.0))` give identical `hessian`, `gradient`, `loss`; report options passed as kwargs (`n_slice_points=0`, `warn=False`) are honoured and not forwarded to `_prepare_sgd_data` (a subclass whose `_prepare_sgd_data` rejects unknown kwargs is used as the guard) |
| `TestMixinReport::test_frozen_params_are_held_fixed` (fast) | the `_FrozenBoundaryModel`-style toy (`test_sgd_fitting.py:239-261` pattern: `scale` POSITIVE, `decay` frozen): names `== ("scale",)`, and the loss value in the report equals the toy loss evaluated with `decay=1.0` |
| `TestAllModels::test_every_sgd_loss_produces_a_report` (slow, parametrised over the 15 problems) | `model.identifiability_report(*args, **kwargs, warn=False, n_slice_points=0)` returns finite `hessian` (symmetric to 1e-8 relative), `len(parameter_names) == P == hessian.shape[0]`, `P` equals the number of trainable scientific coordinates implied by `_build_param_spec()` (computed in the test from the spec: `n(n+1)/2` per PSD, `K−1` per stochastic row, size otherwise); `hessian_method_used == ("finite_difference" if id in EXPECTED_FD else "autodiff")`; behavioural check: for one random unit direction `d`, `hessian @ d` matches the central finite difference of `jax.grad` of the same flat loss (`h=1e-4·max(|θ|,1e-2)`) to `rtol=1e-4, atol=1e-6·max|H|` |
| `TestAllModels::test_dim_autodiff_is_refused_and_auto_falls_back` (slow) | `DirectedInfluenceModel` problem: `hessian_method="autodiff"` raises `ValueError` mentioning `callback`; default `"auto"` succeeds with `hessian_method_used == "finite_difference"` — pins that the FD path is load-bearing for a shipped model |
| `TestOscillatorPhaseSymmetry::test_common_oscillator_free_measurement_matrix_has_n_oscillator_null_directions` (slow) | `simulate_com_scenario(n_time=150, seed=9)`; `CommonOscillatorModel` as in `test_gradients.py:402-430`; `fit_sgd(obs, key, num_steps=30)`, then `polish_to_stationary` on `(model._build_param_spec(), model._sgd_loss_fn)` and `_store_sgd_params`; `report = model.identifiability_report()`: `is_stationary`; `len([d for d in near_null_directions if d.kind == "collinear"]) == n_oscillators`; `cosine_with_raw_null_space(report, symmetry_generator(rotate_by_phase, θ̂, spec), n_oscillators) > 0.99`; every near-null direction has coefficients on `measurement_matrix[...]` and none on `measurement_cov[...]` or `discrete_transition_matrix[...]`; `str(report)` names `measurement_matrix` |
| `TestOscillatorPhaseSymmetry::test_fixed_measurement_matrix_is_identified` (slow) | same data, `update_measurement_matrix=False`, polished: `is_identifiable`, `n_negative_eigenvalues == 0`, `condition_number < 1e4` — the guard that the previous test's flags are not vacuous |
| `TestSpikeOscillatorTrap::test_zero_loadings_leave_dynamics_uninformed` (slow: calls `fit_sgd(num_steps=0)`) | `SwitchingSpikeOscillatorModel(n_oscillators=1, n_neurons=3, n_discrete_states=1, sampling_freq=100.0, dt=0.01, update_discrete_transition_matrix=False, spike_weight_l2=0.0)`; Poisson spikes `(60, 3)` at rate 0.2 with key 0; `fit_sgd(spikes, key=key, num_steps=0)`; then `model.spike_params = SpikeObsParams(baseline=..., weights=jnp.zeros_like(...))`, `model.init_mean = jnp.zeros_like(model.init_mean)`; `report = model.identifiability_report(warn=False)`: every name starting with `Q_0[`, `init_cov_0[`, `A_blocks_0[`, `init_mean[` is in `zero_curvature_parameters`; no name starting with `spike_baseline[` is; `gradient` is `< 1e-10` on every `spike_weights[` and dynamics coordinate (the optimiser cannot move them: the marginal-likelihood form of the `x = β = 0` trap); `is_identifiable is False`; guard: with the model's own initial (nonzero) weights restored, `"Q_0[0,0]" not in zero_curvature_parameters` |
| `TestSpikeOscillatorTrap::test_penalty_masks_the_degeneracy` (slow) | same model with default `spike_weight_l2=100.0` at the zero-loading point: `"spike_weights[0,0,0]"` is *not* zero-curvature (the L2 term supplies curvature) while the dynamics coordinates still are — pins the documented caveat that the report describes the optimised objective |
| `test_package.py::TestPublicAPI` (existing, fast) | passes with the four new names (`test_lazy_name_resolves_to_defining_module` asserts `obj.__module__ == "state_space_practice.identifiability"`) |

Mark slow / integration tests explicitly: everything that calls `fit_sgd`,
`capture_sgd_problem` (its patched `fit_sgd` is a fit call in the AST) or compiles a
model loss; use `pytestmark = pytest.mark.slow` on the classes `TestAllModels`,
`TestOscillatorPhaseSymmetry`, `TestSpikeOscillatorTrap` and on
`test_stored_data_path_matches_explicit_path`. The three `TestMixinReport` toy tests are
fast.

## Fixtures

- The 15 model problems: rebuilt in `test_identifiability_models.py` from the
  constructions in `test_gradients.py:185-448` (same data sizes, seeds and keyword
  arguments), captured with `capture_sgd_problem` so the loss sees exactly what
  `fit_sgd` would. No data is checked in.
- `simulate.scenarios.simulate_com_scenario(n_time=150, seed=9)` for the oscillator
  phase-symmetry tests (returns `{"obs", "params", ...}`; parameters are read as in
  `test_gradients.py:407-426`).
- Poisson spikes for the spike-oscillator trap generated with `jax.random.poisson(key,
  jnp.ones((60, 3)) * 0.2)` (the pattern of `test_gradients.py:438-439`).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Additionally for this phase: `fit_sgd`'s behaviour is unchanged apart from the stored
  attribute (`test_sgd_fitting.py` passes untouched); `uv run mypy` is clean
  (`identifiability.py` is in the file list; `sgd_fitting.py` is not and stays out);
  `import state_space_practice` still imports only JAX
  (`test_package.py::TestImportSideEffects::test_top_level_import_is_lazy`).
