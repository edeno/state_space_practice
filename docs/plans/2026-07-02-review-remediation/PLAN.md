# Review-Remediation Implementation Plan

**Status:** Mostly shipped in code. Fast suite green on 2026-07-08
(`1777 passed, 184 deselected` with `uv run pytest -m "not slow" --tb=short`).
This plan is retained as the historical phase plan; remaining items are deferred
hygiene or in-jit guard follow-ups, not active blockers.

Tracks remediation for the findings from the 2026-07-02 comprehensive library review
([report](../../reviews/2026-07-02-comprehensive-review.md)). The library's core math
was verified correct; this work restored a green test suite, fixed one latent
correctness bug (`CorrelatedNoiseModel` built an invalid covariance), made silent
numerical-divergence handling louder, applied covariance/probability invariant
validation more consistently, and cleared a backlog of statistical-validity,
documentation, and behavior-preserving simplification items. The phase files
remain useful for audit context and deferred cleanup.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is self-contained
   (inputs to read, tasks, validation slice, fixtures, review checklist).
2. **Need broader scope / risks / the CorrelatedNoiseModel design decision?**
   [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, integration points, risks, rollout/back-compat, open questions
- Phases (each ships as a separable PR):
  - [phase-1-restore-green.md](phase-1-restore-green.md) — rewrite the brittle inflation test to its real invariant; seed the one unseeded RNG (test-only, unblocks CI)
  - [phase-2-correlated-noise-psd.md](phase-2-correlated-noise-psd.md) — fix `CorrelatedNoiseModel` to build a symmetric-PSD `Q`; add asymmetry-catching tests
  - [phase-3-invariant-validation.md](phase-3-invariant-validation.md) — shared covariance/probability validators applied at every public entry (Theme B)
  - [phase-4-divergence-telemetry.md](phase-4-divergence-telemetry.md) — surface silent clamps/clips and EM non-convergence (Theme A)
  - [phase-5-statistical-fixes.md](phase-5-statistical-fixes.md) — Wald degenerate flag, reward-init symmetry, Rayleigh correction, capped-transform gradient
  - [phase-6-docs-and-simplify.md](phase-6-docs-and-simplify.md) — doc/comment accuracy fixes + behavior-preserving simplifications
