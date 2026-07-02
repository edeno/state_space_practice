# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

Findings and evidence live in the review report
([`docs/reviews/2026-07-02-comprehensive-review.md`](../../reviews/2026-07-02-comprehensive-review.md));
this file covers only what the executor needs to sequence and scope the fixes.

## Current codebase integration points

Phase 1 (test-only):
- `src/state_space_practice/tests/test_position_decoder.py:1127-1156` —
  `test_capped_inflation_bounds_covariance_growth`: rewrite the assertion; the impl under
  test (`position_decoder.py:955-964`) is **unchanged**.
- `src/state_space_practice/tests/test_position_decoder.py:184-185` — replace bare
  `np.random.*` with the seeded `rng`.

Phase 2 (`CorrelatedNoiseModel`):
- `src/state_space_practice/oscillator_utils.py:253-293`
  (`construct_correlated_noise_process_covariance`) — the source of the asymmetric `Q`;
  changed. Called by both `_initialize_process_covariance` and the SGD reconstruction path.
- `src/state_space_practice/oscillator_utils.py:149-170`
  (`_compute_coupling_transition_block`) — the per-block builder; unchanged (reused).
- `src/state_space_practice/oscillator_models.py:1255-1282`
  (`_initialize_process_covariance`, `_project_parameters`) — `_project_parameters` gains
  a symmetric-PSD projection of the assembled `Q`.
- `src/state_space_practice/tests/test_oscillator_models.py:48-69, 689-698, 877-896` —
  fixture + symmetry/PSD tests that currently hardcode zero coupling; changed.

Phase 3 (shared validators) touches, at their public fit/filter entry points:
`switching_choice.py`, `contingency_belief.py`, `multinomial_choice.py`,
`point_process_models.py`, `position_decoder.py`, `oscillator_models.py` (CNM),
and adds one shared helper (see phase file for home).

Phase 4 (telemetry): `switching_kalman.py:133-168, 2604-2623`,
`switching_point_process.py:2455-2509`, `place_field_model.py:1005-1015`,
`position_decoder.py:955-964, 1201-1231`, and the four EM loops
(`covariate_choice.py:654`, `multinomial_choice.py:566`, `switching_choice.py:667`,
`contingency_belief.py:1006-1067`).

Phase 5: `coupling_validation.py:90-96`, `contingency_belief.py:764`,
`circular_stats.py:132-135`, `parameter_transforms.py:55`.

Phase 6: `switching_kalman.py:796, 955, 994, 1652-1732`,
`switching_point_process.py:758-823, 955-1043, 1487-1498, 2545-2557`,
`place_field_model.py:969, 1258-1264`, `coupling_ekf.py:13-14`,
`contingency_belief.py:471-474`, `covariate_choice.py:953-999` +
`multinomial_choice.py:744-788`, `oscillator_models.py` (`_initialize_measurement_covariance`).

## Scope and dependency policy

### Goals

- Restore a green fast suite and keep it green (`pytest -m "not slow"`).
- Make `CorrelatedNoiseModel` produce a valid (symmetric-PSD) process covariance, and
  make the tests able to catch the class of defect that hid it.
- Convert silently-wrong-result paths into loud ones (warnings, `converged_` flags,
  raised errors on invalid input) — consistent with CLAUDE.md's stance and the pattern
  already used in `place_field_model.fit` / `sgd_fitting.fit_sgd`.
- Apply covariance-PSD and probability-simplex/positivity validation uniformly at public
  entry points.
- Clear the statistical-validity, documentation, and simplification backlog.

### Non-Goals

- **No changes to verified-correct numerical algorithms.** Kalman/RTS/EM, Laplace-EKF
  Poisson, GPB1/GPB2 moment-matching, PG augmentation, softmax/multinomial Laplace were
  hand-verified; do not "optimize" or restructure them.
- No new models, no API surface beyond validators/flags, no performance work.
- The `smith_learning_algorithm.py:178` "negative-Hessian clamp" is **not** in scope — it
  was reviewed and is correct (convex objective).

### Dependency policy

No new runtime dependencies. Validators use `jnp.linalg.eigvalsh`/`eigvals` and existing
helpers (`stabilize_covariance`, `symmetrize`, `validate_coupling_params`).

## Metrics

- Fast suite green: `pytest -m "not slow"` → 0 failed.
- Phase 2: a nonzero-coupling `CorrelatedNoiseModel` has `max|Q−Qᵀ| < 1e-10` and
  `min Re(eigvals(Q)) ≥ -1e-8` for every discrete state; the new tests fail on the
  pre-fix constructor (guard-assert this by construction).
- Phase 3: every public fit/filter entry rejects a non-PSD covariance / non-stochastic
  transition / non-positive `inverse_temperature` with a clear `ValueError`; a regression
  test exercises each rejection.
- Phase 4: each clamp/floor/optimizer-failure path emits a warning when it fires, and
  each of the four EM loops exposes a `converged_` attribute and warns on max-iter; tests
  assert the warning via `caplog`/`pytest.warns`.
- Numerical parity: after every phase, previously-passing tests still pass unchanged
  (these are additive guards, not algorithm changes).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| New validators raise on inputs previously accepted, breaking a user notebook/experiment | Intended: those inputs produced silently-wrong results. Validators target genuinely-invalid values (non-PSD cov, negative inverse-temperature, non-stochastic rows), not marginal ones. Messages name the offending field. See Rollout. |
| Phase 2 changes `CorrelatedNoiseModel` fitted values (symmetrization alters `Q`) | The pre-fix `Q` was invalid, so there is no correct baseline to preserve. Verify recovery tests (`test_scenario_recovery.py:93-113`) still meet their segmentation-accuracy threshold with the corrected `Q`; if they regress, that is signal the old behavior depended on the bug. |
| Adding telemetry inside `jax.jit`/`lax.scan` bodies (Phase 4) is constrained (`jax.debug.print`, no host-side raise) | Prefer host-side checks at the public entry points where possible; inside traced code use `jax.debug.print` on a computed flag, following the existing pattern in `utils.stabilize_probability_vector`. |
| Phase 6 "simplifications" silently change numerics | Both simplifier passes already excluded numeric-touching changes; each task states why it is behavior-preserving. Re-run the full module's tests before/after each simplification and require an independent review at the phase boundary. |

## Rollout Strategy

Replace-in-place per CLAUDE.md default ("just change the code"). No deprecation window:
the new validation converts silent-wrong-result inputs into loud `ValueError`s, which is
strictly safer and is the intended behavior; the telemetry additions are warnings that do
not change return values on valid inputs. There is no published downstream package
depending on these internals (single-author research library). If any checked-in notebook
under `notebooks/` relies on a now-rejected input, that reliance was masking a wrong
result and should be fixed in the notebook, not accommodated.

## Open Questions

1. **How to make `CorrelatedNoiseModel`'s `Q` symmetric — tie blocks, or symmetrize the
   assembled matrix?**
   *Recommended answer (baked into phase 2): tie blocks.* A process-noise cross-covariance
   between oscillator *i* and oscillator *j* is a 2×2 block `C_ij`, and covariance symmetry
   *requires* block `(j,i) = C_ij.T`. Build the off-diagonal block once per unordered pair
   (from the `i<j` entries of `phase_difference`/`coupling_strength`) and set the `(j,i)`
   block to its transpose. This is the correct covariance semantics; the current directed
   two-parameter-per-pair form is ill-defined for a covariance. The alternative
   (`0.5·(Q+Qᵀ)` on the assembled matrix) preserves the directed-parameter API but keeps a
   meaningless directed interpretation and still needs a PSD guard. Decide before starting
   phase 2; the phase file assumes "tie blocks".
2. **Should a too-large coupling that makes the symmetric `Q` indefinite be PSD-projected
   (silently corrected) or raise?** *Recommended: project inside `_project_parameters`
   (the method whose job is "map params into the valid space", used during fitting) AND
   validate-and-raise at the public `__init__`/`fit` entry (phase 3), so a user who
   *constructs* an indefinite `Q` gets a loud error, while the optimizer stays on the
   valid manifold during fitting.*

## Estimated Effort

Small-to-moderate, additive. Rough diff sizing: phase 1 ~30 LOC (tests); phase 2 ~60 LOC
src + ~80 LOC tests; phase 3 ~150 LOC (one shared validator + call sites + tests);
phase 4 ~120 LOC (warnings/flags + tests); phase 5 ~80 LOC; phase 6 net-negative
(~-200 LOC from dedup/relocation) plus doc edits. No algorithm rewrites.
