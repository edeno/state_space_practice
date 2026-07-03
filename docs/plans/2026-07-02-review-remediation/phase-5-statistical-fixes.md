# Phase 5 — Statistical-validity fixes

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Four independent, small correctness improvements to statistical routines. None is a
crash; each can produce a misleading statistic in a specific regime.

**Inputs to read first:**

- `src/state_space_practice/coupling_validation.py:90-96` — Wald degenerate-variance
  guard (`var < 1e-12 → W=0, p=1`).
- `src/state_space_practice/contingency_belief.py:764-765` — `reward_probs_` /
  `state_values_` initialization.
- `src/state_space_practice/circular_stats.py:132-135` — Rayleigh small-sample
  (Mardia–Jupp) correction and its `max(0, min(1, ·))` clip.
- `src/state_space_practice/parameter_transforms.py:44-64` — `positive_capped` and the
  `UNIT_INTERVAL` logit transform.

## Tasks

- **Distinguish "no coupling" from "test degenerated" in the Wald test**
  (`coupling_validation.py:90`). Near-perfect separation drives the Laplace posterior
  variance → 0 while the mean grows, so the strongest real couplings are exactly the ones
  whose `var < 1e-12` term is currently zeroed and reported as `p=1` (a false negative).
  Return a per-entry flag (or NaN p-value) marking degenerate entries instead of silently
  emitting `p=1`, and count them separately in `detection_metrics` rather than as
  not-detected. Document the regime in the function docstring.
- **Break reward-emission symmetry at init** (`contingency_belief.py:764`).
  `reward_probs_` is initialized identically across states (`ones(...)/2`) while
  `state_values_` gets a random perturbation; for the model's stated purpose (states that
  differ in reward contingency) this risks a degenerate EM fixed point. Add a small
  per-state random (seeded) perturbation to `reward_probs_`, mirroring the `state_values_`
  init, or document that identification relies entirely on the value channel.
- **Fix / bound the Rayleigh small-sample correction** (`circular_stats.py:132`). The
  Mardia–Jupp bracket `(1 + (2z − z²)/(4n) − …/(288 n²))` makes the corrected p-value
  non-monotonic in `z` for moderate `z`, so it is only rescued by the `[0,1]` clip and can
  *inflate* the p-value (report less significance than reality) for strongly phase-locked
  small samples. Validate the implementation against a reference (e.g. a table or
  `scipy`/`pycircstat` value) and either correct the formula or fall back to the
  uncorrected `exp(-z)` where the correction would exceed 1. Add a goodness-of-fit test on
  known-concentration samples.
- **Give `positive_capped` a non-zero gradient at the cap**
  (`parameter_transforms.py:55`). The hard `min(softplus(u), max_val)` has exactly zero
  gradient once `softplus(u) > max_val`, so a parameter optimized to the boundary is
  frozen there with no diagnostic. Replace with a smooth saturating map (e.g. a scaled
  sigmoid `max_val · sigmoid(u)`, or `max_val · tanh`-style) that preserves a positive
  gradient throughout, and make the inverse transform consistent. Verify SGD can move a
  parameter *off* the boundary after this change.

## Deliberately not in this phase

- No changes to the coupling estimators' core likelihood/augmentation (verified correct);
  only the `wald_test` post-processing.
- No new statistical models.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_wald_flags_degenerate_variance` | A near-separated component is flagged degenerate (not silently `p=1`); a genuinely-null component still returns `p≈1` |
| `test_contingency_reward_init_breaks_symmetry` | Post-init `reward_probs_` differ across states; a 2-state reward-contingency recovery no longer collapses |
| `test_rayleigh_pvalue_monotone_small_n` | Corrected p-value is monotone non-increasing in `z` and matches a reference within tolerance |
| `test_positive_capped_gradient_nonzero_at_cap` | `grad` of the transform is > 0 at/above the cap; SGD moves a boundary-initialized param inward |

## Fixtures

Small synthesized inputs per test (a near-separated coupling design; known-concentration
circular samples; a boundary-initialized parameter). No real data.

## Review

Dispatch `code-reviewer` against the diff. Confirm:
- Each statistical change is validated against a reference or a proper goodness-of-fit
  test, not just a range bound (per CLAUDE.md testing standards).
- The `positive_capped` change keeps the forward map monotone and the inverse consistent;
  existing transform round-trip tests still pass for interior values.
- The Wald change does not alter results for well-conditioned (non-degenerate) inputs.
