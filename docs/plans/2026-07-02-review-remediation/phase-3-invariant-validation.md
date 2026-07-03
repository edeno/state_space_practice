# Phase 3 — Consistent invariant validation (Theme B)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

The library owns the right validation vocabulary but applies it unevenly: only `init_cov`,
and only in `PlaceFieldModel`, is ever PSD-checked; `process_cov`/`measurement_cov` never
are; several probability/positivity invariants are enforced only during SGD
reparameterization, never at construction or in the bare filter functions. Add a small
shared validator layer and call it at every public fit/filter entry, so invalid inputs
fail loudly instead of producing wrong results or NaN.

**Inputs to read first:**

- `src/state_space_practice/utils.py` — locate `_validate_filter_numerics` and the
  constraint helpers (`PSD_MATRIX`, `STOCHASTIC_ROW`, `POSITIVE`, `UNIT_INTERVAL`,
  `stabilize_probability_vector`) to reuse; this is where the new shared validators belong.
- `src/state_space_practice/coupling_model.py:113-162` (`validate_coupling_params`) — the
  model to copy: cross-field shape agreement + range invariants each tied to a concrete
  numerical failure, with field-naming error messages.
- `src/state_space_practice/place_field_model.py:339-353, 1181, 1372` — the one class that
  already wires validation into both `fit` and `fit_sgd`; mirror this pattern.
- The gap sites: `switching_choice.py:239-244, 359-361, 443-487, 643-645`,
  `contingency_belief.py:425-426, 741-761`, `multinomial_choice.py:415-429`,
  `point_process_models.py:203-208, 397-431, 905, 1082`,
  `position_decoder.py:67-82, 994, 1437-1449`,
  `oscillator_models.py` (CNM `__init__`).

## Tasks

- **Add a shared symmetric-PSD covariance validator** in `utils.py`, e.g.
  `validate_covariance(cov, name, *, per_state=False)` that raises `ValueError` (naming
  `name` and, for stacked `(d,d,S)` inputs, the offending discrete-state index) when a
  slice is non-symmetric (`max|C−Cᵀ| > tol`) or non-PSD (`min eigvalsh(C) < -tol` after
  the symmetry check passes). Host-side (not traced): call it at public entry points, not
  inside `jit`/`scan`. Reuse it from `PlaceFieldModel` too, replacing the ad-hoc
  `_validate_filter_numerics` call if it subsumes it (delete the old path if so; otherwise
  leave a one-line note why both remain).
- **Validate covariances at the switching / point-process / position entries.** Apply
  `validate_covariance` to `init_cov`, `process_cov`, and `measurement_cov` in:
  `point_process_models._fit_single` (`:905`) and `fit_sgd` (`:1082`);
  `position_decoder_filter` (`:994`) and `PositionDecoder.decode` (`:1437`);
  `CorrelatedNoiseModel` construction (consolidating the phase-2 hook). Add a finiteness
  guard to `point_process_models.fit_sgd`'s loss so a non-finite marginal LL raises rather
  than silently driving all params to NaN.
- **Validate probability-simplex and stochastic-matrix inputs.**
  `switching_choice_filter` (`:359`): reject a `discrete_transition_matrix` whose rows are
  not non-negative and sum to 1 (reuse `STOCHASTIC_ROW`); `contingency_belief_filter`
  (`:491`): normalize/validate `init_state_prob` via `stabilize_probability_vector` (as
  `switching_choice` already does for `init_discrete_prob`) instead of the bare
  `maximum(·, 1e-30)` floor.
- **Validate positivity of scalar hyperparameters at construction.** In
  `SwitchingChoiceModel.__init__` (`:443`), `MultinomialChoiceModel.__init__` (`:415`),
  `ContingencyBeliefModel.__init__` (`:741`): reject non-positive `inverse_temperature`
  and `process_noise`, and `decay ∉ (0, 1]`, at construction (the SGD spec already
  declares these constraints — enforce them for the EM/`learn_*=False` paths too). A
  negative `inverse_temperature` currently inverts preferences silently.
- **Bound `AdaptiveInflationConfig` above.** In `__post_init__`
  (`position_decoder.py:72-82`), add upper bounds on `max_alpha` and `gain` (with a
  documented default ceiling), rejecting values that would let per-bin inflation diverge.
- **Guard the computed `p_stay` default.** `point_process_models.py:203-208`: apply the
  same `[0, 1]` check to the *computed* default (`1 − 1/(expected_dwell_sec·sampling_freq)`)
  that already guards the user-supplied branch, raising for `sampling_freq < 1 Hz` regimes
  that produce a negative transition probability.
- **Docs:** add a short "Input validation" subsection to the relevant public docstrings
  (or a note in `CLAUDE.md`'s numerical-precision section) stating that covariance/simplex
  inputs are validated at entry — and correct the current CLAUDE.md claim that "the filter
  validates `init_cov` at every public entry point", which becomes true only after this
  phase.

## Deliberately not in this phase

- No telemetry/warnings for *internal* divergence during fitting — that is
  [phase 4](phase-4-divergence-telemetry.md). This phase is about rejecting invalid
  *inputs* at the boundary.
- Do not add validation inside `jit`/`scan` bodies (not traceable); host-side only.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_validate_covariance_*` | Raises on non-symmetric and on indefinite slices; passes valid; names the bad state index |
| `test_*_rejects_non_psd_init_cov` (per entry point) | `ValueError` from `position_decoder`, `point_process_models`, CNM on a symmetric-indefinite `init_cov` |
| `test_switching_choice_rejects_non_stochastic_transition` | `ValueError` on rows not summing to 1 |
| `test_choice_models_reject_nonpositive_inverse_temperature` | `ValueError` at construction for `inverse_temperature ≤ 0` |
| `test_adaptive_inflation_config_upper_bound` | `ValueError` for `max_alpha`/`gain` above ceiling |
| `test_p_stay_default_guard` | `ValueError` for `sampling_freq < 1 Hz` computed default |
| existing suites | Unchanged — valid inputs still pass |

## Fixtures

Small synthesized invalid inputs per test (indefinite covariance = symmetric with a
negative eigenvalue; non-stochastic matrix; negative scalar). No real data.

## Review

Dispatch `code-reviewer` against the diff. Confirm:
- Each validator is tied to a concrete numerical failure and names the offending field
  (not a generic "invalid input").
- Validators run at public entry points only, never inside traced code.
- If `_validate_filter_numerics` is subsumed, the old path is removed (no orphan).
- CLAUDE.md's `init_cov`-validation claim is corrected to match reality.
- Every new rejection has a test; no previously-passing test regresses.
