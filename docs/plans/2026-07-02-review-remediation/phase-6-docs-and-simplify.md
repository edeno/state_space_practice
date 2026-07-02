# Phase 6 — Doc/comment accuracy + behavior-preserving simplifications

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Two low-risk cleanups grouped into one PR: (a) correct comments/docstrings that misstate
shapes, axis order, or conditioning direction (bug-bait in numerical code, though the code
is correct), and (b) apply the behavior-preserving simplifications both simplifier passes
endorsed. Every change here must be provably output-preserving — re-run each touched
module's tests before/after.

**Inputs to read first:**

- Comment sites: `switching_kalman.py:796, 994` (reversed conditioning), `:955`
  (transposed cross-cov axis label), `switching_point_process.py:1487-1498` (off-by-one
  filter-time-index doc; also `:2545-2557` stale future-work + nonexistent
  `update_spike_glm_params_second_order`), `place_field_model.py:969, 1258-1264`
  (uncalled `dynamics_only_m_step` reference; `n_basis` overloaded at `:564, 586` vs
  `:500`), `coupling_ekf.py:13-14` (stale "future" framing), `oscillator_utils.py:760-762`
  (frequency-fold comment).
- Simplification sites: `switching_kalman.py:1652-1732` (inlined gamma1/beta) vs
  `:2350-2416` (`compute_transition_sufficient_stats`); `switching_point_process.py:758-823,
  955-1043` (test-only functions); `contingency_belief.py:471-474` (dead branch);
  `covariate_choice.py:953-999` + `multinomial_choice.py:744-788` (`_m_step_beta` dup);
  `oscillator_models.py` (`_initialize_measurement_covariance` in COM/CNM/DIM).

## Tasks

**Doc/comment corrections (code unchanged):**

- Fix the reversed transition-conditioning docstring in **both**
  `switching_kalman.py:796` and `:994` to the correct row-stochastic form
  `Z[j,k] = P(S_{t+1}=k | S_t=j)`.
- Fix the transposed cross-covariance axis label at `switching_kalman.py:955` to match its
  correct siblings (lines 953-954, 1026): second-to-last axis = current state `S_t=j`,
  last = next state `S_{t+1}=k`.
- Fix the self-contradictory filter time-index doc at `switching_point_process.py:1487`
  (output index 0 is `p(x₁|y₁)`, not `p(x_{t+1}|y_{1:t+1})`).
- Correct `place_field_model.py:969, 1258-1264` (remove/adjust the reference to the
  uncalled `dynamics_only_m_step`; the `_m_step` computes `A_new`/`Q_new` inline) and the
  `n_basis` docstrings at `:564, 586` to state per-neuron vs total consistently with the
  actual array shapes.
- Update the stale "future work" notes now shipped: `coupling_ekf.py:13-14` (PG
  cross-check exists); `switching_point_process.py:2545-2557` (second-order correction is
  implemented via `use_second_order=True`; remove the reference to the nonexistent
  `update_spike_glm_params_second_order`).
- Clarify the `oscillator_utils.py:760-762` frequency-fold comment (adds `fs/2`, folding
  into `[0, fs/2)` — not a period unwrap); state the intended assumption.

**Behavior-preserving simplifications (re-run tests before/after each):**

- `switching_kalman.py:1652`: replace the ~55 inlined gamma1/beta lines in
  `switching_kalman_maximization_step` with a call to the existing public
  `compute_transition_sufficient_stats` (`:2350`) — verified byte-identical logic. Confirm
  identical outputs on the existing switching-EM tests.
- `switching_point_process.py:758, 955`: `_single_neuron_glm_loss` and
  `_neg_Q_single_neuron` are test-only (no `src/` caller). Relocate them to the test module
  (or the test-support module) with a note, or, if kept, add a comment marking them as test
  fixtures. Requires a human call — do not blind-delete (they back real tests).
- `contingency_belief.py:471-474`: collapse the two token-identical branches of
  `_compute_obs_offset` to a single `return obs_weights @ obs_dm_t` and drop the now-dead
  `per_state_obs` local.
- Extract the duplicated `_m_step_beta` grid + golden-section routine shared by
  `covariate_choice.py:953` and `multinomial_choice.py:744` into one helper taking the
  per-model `_eval_beta` closure (the numeric closures stay per-class, untouched).
- Hoist the byte-identical `_initialize_measurement_covariance` from the three oscillator
  subclasses to `BaseModel` (`oscillator_models.py`) — verify no subclass diverges.

## Deliberately not in this phase

- Do **not** merge the Poisson-vs-generic Laplace update or the dense/block-diagonal
  smoothers — both simplifier passes explicitly rejected these (bit-for-bit roundoff
  contract / vmap-boundary), and they are numerics-adjacent.
- Do not rename for taste or reformat unrelated code (ruff formatting on touched lines is
  fine).

## Validation slice

| Test | Asserts |
| --- | --- |
| touched-module suites (switching, contingency, choice, oscillator, coupling) | All pass unchanged before and after each simplification |
| `test_switching_kalman` EM/maximization tests | Identical `A`/`Q` sufficient stats after the `compute_transition_sufficient_stats` swap |
| `git grep update_spike_glm_params_second_order` | No references remain (nonexistent symbol removed from docs) |

## Fixtures

None new; reuse existing module fixtures. Simplifications are covered by the modules'
current tests (that they pass unchanged is the correctness evidence).

## Review

Dispatch `code-reviewer` against the diff. Confirm:
- No numerical output changed (state that the touched-module suites pass unchanged; for
  the `compute_transition_sufficient_stats` swap, show equal outputs).
- Comment fixes match what the code actually does (spot-check each against the code).
- Test-only functions were relocated/annotated, not blind-deleted, and their tests still
  import them.
- No parallel v1/v2 left behind by the dedups.
