# Comprehensive Library Review — 2026-07-02

**Scope:** entire `src/state_space_practice/` library source (~18k lines, 35 modules).
**Method:** 13 specialized review agents (6 subsystem code-reviewers, 2 silent-failure
hunters, 2 report-only simplifiers, test-quality, comment-accuracy, type/invariant
analyzers) plus a baseline fast-test run (`pytest -m "not slow"`).
**Mode:** report-only — no source files were modified during the review.

**Remediation plan:** [`docs/plans/2026-07-02-review-remediation/PLAN.md`](../plans/2026-07-02-review-remediation/PLAN.md)

---

## Bottom line

The core math is correct and the library is unusually careful. Reviewers hand-verified
the load-bearing algorithms against reference formulas — Kalman/RTS/Joseph-form, EM
M-steps, Laplace-EKF Poisson likelihood, GPB1/GPB2 moment-matching (including the
recently-fixed axis indexing), Polya-Gamma augmentation, softmax/multinomial Laplace —
several via brute-force numerical cross-checks. The point-process, switching, coupling,
and core-Kalman subsystems came back clean. Findings are about **robustness,
error-surfacing, and hygiene**, not algorithmic correctness. They cluster into four
themes plus one real (latent) bug and one brittle test.

Baseline fast suite: **1 failed, 1323 passed, 164 slow deselected** (~7 min). The slow
suite (EM / full-pipeline tests) was not run.

---

## Status tracker

Remediation shipped on branch `fix/review-remediation` (phases 1, 3, 4, 5, 6,
each reviewed clean). ✅ done · ◑ partial (rest deferred, see notes) · ⏸ paused.

| # | Finding | Severity | Phase | Status |
|---|---------|----------|-------|--------|
| 1 | Brittle red test `test_capped_inflation_bounds_covariance_growth` | Build-red | 1 | ✅ done (rewritten to per-step-cap invariant; RNG seeded) |
| 2 | `CorrelatedNoiseModel` non-symmetric/non-PSD `Q` | Critical | 2 | ⏸ paused — needs tie-blocks-vs-symmetrize decision |
| 3 | Theme A — silent numerical-divergence guards (no telemetry) | High | 4 | ◑ EM `converged_` flags + GPB cap telemetry; other in-jit clamps deferred |
| 4 | Theme B — inconsistent invariant/input validation | High | 3 | ◑ host-side covariance/prob/scalar validators; jitted filter-level deferred |
| 5 | Statistical-validity notes (Wald, reward-init, Rayleigh, capped-transform) | Medium | 5 | ◑ positive_capped gradient + reward-init fixed; Wald deferred; Rayleigh no-change |
| 6 | Doc/comment accuracy hazards | Low | 6 | ✅ done (except one self-consistent time-index comment left as-is) |
| 7 | Report-only simplifications (dedup / dead / test-only code) | Low | 6 | ◑ switching_kalman dedup + contingency dead-branch; `_m_step_beta`/hoist deferred |
| 8 | Test-quality gaps (valid-only projection tests, no EM-guard tests, unseeded RNG) | Medium | 1,3,4 | ◑ inflation test + validator/guard/converged tests added; broader gaps remain |

---

## 1. Build state — the fast suite is RED (1 failing test)

`src/state_space_practice/tests/test_position_decoder.py:1156` —
`test_capped_inflation_bounds_covariance_growth` deterministically fails (seed 42):
trace ratio **42.7×** vs assertion `< 6.0`.

**Verdict (evidence-backed): the test is brittle — rewrite the assertion; do not relax
the threshold a 4th time or "fix" the implementation.** Reproduced: the trace ratio is
*non-monotonic* in `max_alpha` (1.001→23×, 1.01→43×, 1.5→22×) and is a single-bin
(t=419) artifact where the *baseline* trace is transiently near-zero (that is what the
`np.maximum(base_trace, 1e-12)` guard betrays). The `<6.0` bound was never a property of
the per-step cap; its history was relaxed 3→5→6 chasing the same fixture artifact. A
correct test asserts the real invariant: `capped_predicted_cov ≤ max_alpha ·
base_predicted_cov` per step (before the measurement update mixes the covariances),
with a guard that some bin actually hit the cap.

Related test-hygiene: `test_position_decoder.py:184-185` uses bare, unseeded
`np.random.randn` / `np.random.poisson` (the suite's only unseeded RNG) despite a seeded
`rng` at line 147.

---

## 2. Critical correctness bug — non-symmetric "covariance" in `CorrelatedNoiseModel`

`src/state_space_practice/oscillator_utils.py:253`
(`construct_correlated_noise_process_covariance`) builds off-diagonal blocks from
*independent* directed params — block `(i,j) = coupling[i,j]·R(phase[i,j])`, block
`(j,i)` from separate `coupling[j,i]`/`phase[j,i]` — so the assembled `Q` is generically
**not symmetric and not PSD**. `src/state_space_practice/oscillator_models.py:1255`
(`_initialize_process_covariance`) feeds it straight into the switching Kalman filter's
`slogdet(Q)` / `psd_solve(Q, ·)`, which assume symmetric PSD.
`_project_parameters` (`oscillator_models.py:1270`) projects each 2×2 block to a rotation
via `project_matrix_blockwise` (`oscillator_utils.py:772`) but does **not** restore
cross-block transpose symmetry.

Reproduced end-to-end through `CorrelatedNoiseModel.process_cov` with nonzero coupling:
`max|Q − Qᵀ| = 0.47`, `min Re(λ) = −0.34` (complex eigenvalues).

**Why every test misses it** — two independent blind spots:
- The symmetry test `test_oscillator_models.py:877`
  (`test_process_cov_symmetric_for_all_states`) hardcodes
  `coupling_strength=zeros`, `phase_difference=zeros` (lines 889-890), so `Q` is diagonal
  and trivially symmetric. The shared fixture `correlated_noise_params`
  (`test_oscillator_models.py:48`) also zeroes coupling.
- Every PSD check (`test_oscillator_models.py:696`, and siblings) uses
  `jnp.linalg.eigvalsh`, which reads one triangle and implicitly symmetrizes — it
  **cannot** detect asymmetry.

Consequence: a non-symmetric/indefinite `Q` in `A P Aᵀ + Q` corrupts every predicted
covariance → wrong Kalman gain / likelihood, or silent NaN. Scoped to
`CorrelatedNoiseModel`; the core Kalman/point-process/switching/coupling paths are clean.
Telling inconsistency: DIM enforces spectral-radius < 1 on its transition matrix as "a
hard physical constraint" (`oscillator_models.py:1729`), but CNM has no PSD guard on `Q`.

---

## 3. Theme A — silent numerical-divergence handling

The switching smoothers/M-steps and place-field/inflation paths clamp, clip, and floor
their way out of *documented* numerical blow-ups with **no telemetry**, so a diverged fit
returns finite, bounded, monotone-LL output that is scientifically wrong with no signal.
This violates the repo's stated "loud failures over silently-wrong results" stance
(CLAUDE.md). The correct pattern already exists in `place_field_model.fit`
(`for…else` + `caplog` warnings) and `sgd_fitting.fit_sgd` (NaN-rollback warns); it is
applied inconsistently.

| Site | What is swallowed |
|---|---|
| `switching_kalman.py:133` (`_cap_covariance_trace`, consts `:143-144`) | Rescales smoother cov at `1e8×` filter trace + clips means to ±`1e6`; corrupts EM sufficient stats, no warning |
| `switching_point_process.py:2455` (`_m_step_dynamics`) | M-step eigenvalue clips band-aid a *documented* "GPB1 collapse positive-feedback loop", no telemetry |
| `switching_kalman.py:2604` (`optimize_dim_transition_params`) | Ignores BFGS `result.success` + silently rescales the result |
| `place_field_model.py:1005` (`_m_step`) | Negative EM-estimated variances floored to `1e-10` silently (neighboring rollback path *does* warn) |
| `position_decoder.py:955` | Adaptive inflation can inject NaN; the divergence guard (`:1201-1231`) compares NaN with `</>` → `False`, so even the warning is suppressed |

Plus **four EM fitters never signal non-convergence or LL-decrease**:
`covariate_choice.py:654`, `multinomial_choice.py:566`, `switching_choice.py:667`,
`contingency_belief.py:1006` (and `contingency_belief.py:1063` discards inner BFGS
`success`) — all inconsistent with the Smith / place-field / oscillator loops that warn.

**Cross-checked and downgraded (not defects):**
- `smith_learning_algorithm.py:178` "negative-Hessian clamp inverts the Laplace
  approximation" — that model's neg-log-posterior (Bernoulli-logit + Gaussian prior) is
  provably convex, so the Hessian is always ≥ prior precision > 0; the floor only catches
  roundoff. Not a real breakdown.
- `position_decoder.py` "KDE fabricates rates at unvisited positions" — by-design
  Bayesian shrinkage-to-baseline, not a defect.

---

## 4. Theme B — inconsistent invariant / input validation

The library owns the right constraint vocabulary (`PSD_MATRIX`, `STOCHASTIC_ROW`,
`_validate_filter_numerics`, `validate_coupling_params`) but enforces it on only some
surfaces. Only `init_cov`, and only in `PlaceFieldModel`, is ever PSD-checked;
`process_cov` / `measurement_cov` never are, and the bare filter functions bypass
validation. CLAUDE.md's claim that "the filter validates `init_cov` at every public entry
point" is true only for `PlaceFieldModel`.

Highest-value gaps (invalid value → wrong result, not just a clean error):
- `switching_choice.py:443` — a **negative `inverse_temperature` silently inverts
  preferences** (softmax favors the worst option); EM has no beta M-step to correct it.
- `switching_choice.py:239` / `contingency_belief.py:425` — non-row-stochastic transition
  / un-normalized prior accepted (`init_discrete_prob` *is* stabilized, the transition
  matrix isn't); a transposed-but-plausible matrix → silently wrong posterior.
- `position_decoder.py:994` and `point_process_models.py:397` — non-PSD `init_cov`
  accepted with zero validation → all-NaN output; `fit_sgd` has no finiteness guard so
  NaN gradients poison all params.
- `position_decoder.py:67` — `AdaptiveInflationConfig` bounds `max_alpha`/`gain` below but
  **not above** (the ceiling that would prevent the divergence the config exists to
  prevent; ties into finding 1).
- `point_process_models.py:203` — computed default `p_stay` goes negative for
  `sampling_freq < 1 Hz` (the `[0,1]` check guards only the user-supplied branch).

Single highest-leverage fix: one shared symmetric-PSD validator applied to every
`init_cov`/`process_cov`/`measurement_cov` slice at every public entry — closes the
finding-2 blast path and the covariance gaps at once. Positive examples to copy:
`validate_coupling_params` (`coupling_model.py:113`), `PlaceFieldRateMaps.__init__`
(`position_decoder.py:201`), `PlaceFieldModel.__init__` (`place_field_model.py:339`),
`MultinomialChoiceModel._validate_choices` (`multinomial_choice.py:156`).

---

## 5. Statistical-validity notes

- `coupling_validation.py:90` — the Wald degenerate-variance guard
  (`var < 1e-12 → W=0, p=1`) can report the **strongest** couplings as null: near-perfect
  separation drives variance→0 while the mean grows, so the most significant entries are
  exactly the ones zeroed. Add a per-entry flag distinguishing "no coupling" from "test
  degenerated."
- `contingency_belief.py:764` — `reward_probs_` initialized identically across states
  (only `state_values_` is broken out of symmetry), risking a degenerate EM fixed point
  for the model's stated purpose (reward-contingency states).
- `circular_stats.py:132` — Rayleigh small-sample correction is non-monotonic in `z` and
  only ad-hoc clipped to `[0,1]`; can inflate p-values for strongly phase-locked small
  samples.
- `parameter_transforms.py:55` — `positive_capped` uses a hard `min(softplus(u),
  max_val)` whose gradient is exactly 0 past the cap → a parameter optimized to the
  boundary is frozen there with no diagnostic.

---

## 6. Doc / comment accuracy hazards (code correct, comments bug-bait)

- `switching_kalman.py:796` **and** `switching_kalman.py:994` — transition docstring
  reverses the conditioning direction (`Z(j,k)=P(S_t=j|S_{t-1}=k)` vs the correct
  row-stochastic `Z[j,k]=P(S_{t+1}=k|S_t=j)`). Two spots, same defect.
- `switching_kalman.py:955` — cross-covariance axis label transposed vs its own siblings
  (lines 953-954, 1026 are correct).
- `switching_point_process.py:1487` — filter time-index doc is self-contradictory
  (off-by-one: output 0 is `p(x₁|y₁)`, not `p(x_{t+1}|y_{1:t+1})`).
- `place_field_model.py:969` — comments justify block re-detection via a function
  (`dynamics_only_m_step`) never called on that path; `n_basis` documented with two
  meanings across `place_field_model.py:564`/586 vs 500.
- Stale "future work" notes describing already-shipped features: `coupling_ekf.py:13`;
  `switching_point_process.py:2545` references a nonexistent
  `update_spike_glm_params_second_order`.

---

## 7. Report-only simplifications (behavior-preserving)

Both simplifier passes were told never to touch numerics and correctly *rejected* several
tempting merges (Poisson-vs-generic Laplace bit-for-bit contract; vmap-boundary
smoothers). Clean, low-risk wins:
- `switching_kalman.py:1652` — ~55 lines of inlined gamma1/beta stats are **byte-identical**
  to the existing public `compute_transition_sufficient_stats`; call it instead.
- `switching_point_process.py:758` — `_single_neuron_glm_loss` and `_neg_Q_single_neuron`
  (~155 lines) are **test-only** (nothing in `src/` calls them) — relocate or mark.
- `contingency_belief.py:471` — `_compute_obs_offset` has two token-identical branches;
  collapse (and drop the dead `per_state_obs` local).
- `_m_step_beta` duplicated near-verbatim across `covariate_choice.py:953` and
  `multinomial_choice.py:744`; the binom-LL block appears 3× in
  `smith_learning_algorithm.py`; identical `_initialize_measurement_covariance` in all
  three oscillator subclasses could hoist to `BaseModel`.

---

## 8. Test-quality gaps (beyond finding 1)

1. **Clamp/projection routines tested only on already-valid input** —
   `test_switching_point_process.py:5831` (`test_project_parameters_*`) feeds valid params
   and asserts only shape/finiteness; a no-op projection passes. Same weakness that hid
   finding 2.
2. **Switching EM has no divergence/LL-decrease guard test** — the
   `assert_em_rolls_back_on_ll_decrease` helper (`conftest.py:555`) is wired into
   point-process and Smith loops but has **zero** hits in the switching tests, the most
   numerically fragile models.
3. `test_position_decoder.py:184` — the suite's only genuinely unseeded randomness.
4. Value-critical Kalman/GPB updates tested shape-only; `test_gpb2_output_shapes_match_gpb1`
   compares only shapes despite its name; several tautological echo-back assertions and a
   loose `<1.0`-nat "convergence" check (`test_smith_learning_algorithm.py:774`).

---

## What is strong (keep as models)

- EM monotonicity coverage in `test_switching_kalman.py:1256-1712` (single-state exact,
  two-identical-state, distinguishable-state, ELBO).
- f32/numerical-guard tests in `test_point_process_kalman.py:2427` (`pytest.warns(match=
  "float32")` with a companion must-NOT-warn test).
- `conftest.py:555` `assert_em_rolls_back_on_ll_decrease` (caplog-based) — the template to
  extend to the switching / contingency / covariate loops.
- Recovery/statistical suites use real statistics (correlation > 0.7, segmentation
  accuracy ≥ 0.70, `assert_allclose(atol=1e-4)` M-step recovery), not range-bound
  antipatterns.
- `validate_coupling_params`, `PlaceFieldRateMaps.__init__`, `PlaceFieldModel.__init__`,
  `MultinomialChoiceModel._validate_choices` — consequence-tied construction-time
  validation done right.
