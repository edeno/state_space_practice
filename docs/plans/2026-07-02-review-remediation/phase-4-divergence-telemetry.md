# Phase 4 — Surface silent numerical-divergence handling (Theme A)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Several paths clamp, clip, or floor their way out of *documented* numerical blow-ups with
no signal, and four EM loops never report non-convergence — so a diverged fit returns
finite, bounded, monotone-LL output that is scientifically wrong. Make each firing
observable, following the pattern already used in `place_field_model.fit`
(`for…else` + `caplog`) and `sgd_fitting.fit_sgd` (NaN-rollback warns). This phase adds
warnings/flags only; it does not change return values on well-conditioned inputs.

**Inputs to read first:**

- `src/state_space_practice/place_field_model.py:1224-1287` — the good pattern: warns on
  non-finite LL, logs LL-decrease rollback, `for…else` max-iter warning.
- `src/state_space_practice/utils.py` (`stabilize_probability_vector`) — the in-`scan`
  telemetry pattern using `jax.debug.print` on a computed flag; reuse for traced sites.
- `src/state_space_practice/tests/conftest.py:555`
  (`assert_em_rolls_back_on_ll_decrease`) — the caplog-based helper to extend to the
  switching / contingency / covariate loops.
- Firing sites: `switching_kalman.py:133-168` (`_cap_covariance_trace`,
  `_compute_max_allowed_trace`, mean clip), `:2604-2623`
  (`optimize_dim_transition_params`); `switching_point_process.py:2455-2509`
  (`_m_step_dynamics` eigenvalue clips); `place_field_model.py:1005-1015` (negative
  variance floor); `position_decoder.py:955-964, 1201-1231` (inflation NaN + NaN-blind
  divergence guard).
- EM loops without signaling: `covariate_choice.py:654-735`,
  `multinomial_choice.py:566-616`, `switching_choice.py:667-689`,
  `contingency_belief.py:1006-1067`.

## Tasks

- **Signal when the smoother covariance/mean cap fires.** `switching_kalman.py`: when
  `_cap_covariance_trace` rescales (ratio > 1) or the ±`1e6` mean clip engages, emit a
  `jax.debug.print` on the computed "cap fired" flag (host-side raising is impossible in
  the jitted smoother). Document in the smoother docstring that a printed cap-warning means
  the returned posterior and EM sufficient statistics are unreliable. Do the same for the
  GPB2 triple-stat clip.
- **Check optimizer success in `optimize_dim_transition_params`**
  (`switching_kalman.py:2604`). After `minimize(...)`, warn (host-side) when
  `result.success` is False or the spectral-radius rescale (`:2619-2621`) actually
  triggered, so the caller knows the returned "optimized" params may be an unconverged or
  post-modified iterate.
- **Signal the switching-PP M-step eigenvalue clips** (`switching_point_process.py:2455`).
  Emit a `jax.debug.print` when the process-cov / init-cov / smoother-cov eigenvalue clip
  changes the spectrum by more than a documented tolerance (i.e. the clip actually bit),
  since the code comment itself attributes this to a GPB1 collapse feedback loop.
- **Warn on the place-field negative-variance floor** (`place_field_model.py:1005`). When
  `jnp.diag(Q_new)`/`init_cov` has a negative entry before the `maximum(·, 1e-10)` floor,
  log a warning (the neighboring rollback path already warns; close this gap).
- **Make the inflation path NaN-aware** (`position_decoder.py:955-964, 1201-1231`). Detect
  a non-finite `alpha_t`/covariance during the scan and surface it; fix the divergence
  guard so it does not silently pass on NaN (NaN `</>` comparisons return `False`, so the
  current guard is NaN-blind) — check `jnp.isnan`/`jnp.isinf` explicitly and warn.
- **Add non-convergence signaling to the four EM loops.** For
  `CovariateChoiceModel`, `MultinomialChoiceModel`, `SwitchingChoiceModel`,
  `ContingencyBeliefModel`: set a `converged_` attribute, warn on max-iter exhaustion
  (`for…else`), and warn on an LL decrease between iterations (mirror
  `smith_learning_algorithm` / `place_field_model`). Also check the inner-BFGS
  `result.success` in `contingency_belief.py:1063` and warn when it is False.

## Deliberately not in this phase

- Do not change the *values* the clamps/clips produce, or the algorithms — only add
  observability. (Whether a too-large coupling should raise is [phase 3](phase-3-invariant-validation.md).)
- Do not add input validation here (that is phase 3). This phase is about *internal*
  divergence during fitting.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_switching_em_warns_on_ll_decrease` (extend `conftest.py:555` helper) | Warning emitted when a crafted iteration decreases LL |
| `test_<model>_sets_converged_flag` (×4 EM models) | `converged_ is False` + max-iter warning when `max_iter` too small; `converged_ is True` on an easy problem |
| `test_optimize_dim_transition_params_warns_on_failure` | Warns when BFGS does not converge / rescale triggers |
| `test_place_field_warns_on_negative_variance_floor` | Warning when M-step produces a negative variance pre-floor |
| `test_inflation_nan_is_surfaced` | Divergence guard fires (warns) on a NaN-injecting input instead of passing silently |
| existing suites | Unchanged on well-conditioned inputs (no spurious warnings — add must-NOT-warn companions) |

## Fixtures

Crafted ill-conditioned / max-iter-too-small inputs synthesized per test. For the
`jax.debug.print` traced sites, assert via captured stdout or a host-side flag exposed on
the result where feasible; where not, assert the host-side `converged_`/warning wrappers.

## Review

Dispatch `code-reviewer` against the diff. Confirm:
- No return value changes on valid inputs (must-NOT-warn companion tests pass).
- Traced-code telemetry uses `jax.debug.print` on a real computed condition, not an
  unconditional print.
- All four EM loops now expose `converged_` consistently; the flag is documented.
- The inflation divergence guard is no longer NaN-blind.
