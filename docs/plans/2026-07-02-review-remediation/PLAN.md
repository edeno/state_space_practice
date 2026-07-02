# Review-Remediation Implementation Plan

**Status:** Not started.

Fixes the findings from the 2026-07-02 comprehensive library review
([report](../../reviews/2026-07-02-comprehensive-review.md)). The library's core math
was verified correct; this work restores a green test suite, fixes one latent
correctness bug (`CorrelatedNoiseModel` builds an invalid covariance), makes silent
numerical-divergence handling loud, applies covariance/probability invariant validation
consistently, and clears a backlog of statistical-validity, documentation, and
behavior-preserving simplification items. Each phase ships as an independent PR.

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
