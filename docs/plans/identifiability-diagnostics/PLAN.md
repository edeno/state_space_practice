# Identifiability Diagnostics Implementation Plan

**Status:** Not started.

Adds `state_space_practice.identifiability`: a report built from the observed
information (the Hessian of a model's negative log-likelihood in its scientific
parameter coordinates) that tells a user, before a scientific claim rests on a fit,
which parameters — or which linear combinations of parameters — the data do not
determine. It works on any `SGDFittableMixin` model through
`model.identifiability_report()` and on any differentiable loss through
`fisher_information` / `identifiability_report_from_loss`. It flags exactly-degenerate
directions (an unobserved latent's scale traded against its loading, an oscillator's
arbitrary phase against a free measurement matrix), coordinates the objective is flat
in at the current point (the spike-only "x = β = 0" bootstrap trap), saddle points
masquerading as fits, and reports Wald standard errors, a condition number and a slice
of the objective along the worst direction. Two other plans
(`docs/plans/volatile-kalman-choice/`, `docs/plans/theta-sweep-amplitude/`) use it as a
gate for their models.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, decided defaults and what the codebase forced us to change, risks, metrics, open questions
- [shared-contracts.md](shared-contracts.md) — public API signatures, `IdentifiabilityReport` fields, the scientific-coordinate chart rules, the `_sgd_data_` attribute the mixin keeps
- [designs.md](designs.md) — complete code for the coordinate charts, the Hessian methods (autodiff + finite-difference fallback), the spectrum analysis, the slice, the report text, and the acceptance-test losses
- Phases (each ships as a separable PR):
  - [phase-1-core-diagnostics.md](phase-1-core-diagnostics.md) — `identifiability.py` functional API + report; unit tests against analytic information matrices and exact symmetries; the spike-only vs LFP-anchored coupling acceptance tests
  - [phase-2-model-integration.md](phase-2-model-integration.md) — `SGDFittableMixin.identifiability_report()` (the mixin must first keep the prepared data), model-level acceptance tests (every SGD loss; oscillator phase symmetry; spike-only latent-oscillator trap), top-level exports, README and CHANGELOG
