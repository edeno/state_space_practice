# GLM Families: Negative Binomial and Zero-Inflated Gamma — Implementation Plan

**Status:** Not started.

Adds two observation families to the Laplace-EKF point-process machinery so the
same filter, smoother and `PointProcessModel` fit over-dispersed spike counts
(negative binomial with log link, per-neuron dispersion learnable by SGD) and
deconvolved calcium-imaging traces (zero-inflated gamma after Wei et al. 2020,
with latent-dependent non-zero probability and gamma scale). Users opt in with
`family=negative_binomial_family(dt, r)` / `zero_inflated_gamma_family(k)` on the
filters or `PointProcessModel(family="negative_binomial" | "zero_inflated_gamma", ...)`;
everything they use today is unchanged, number for number.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points (file:line), goals/non-goals, dependency policy, metrics, risks, rollout, open questions, effort
- [shared-contracts.md](shared-contracts.md) — the extended `GLMFamily` contract, the `family=` filter/smoother keyword, `PointProcessModel` family kinds, the stacked-predictor layout
- [designs.md](designs.md) — derivations (scores, expected information, log-likelihoods), the stable log-gamma ratio, design alternatives for ZIG and the chosen one, full code for the families, the filter/model plumbing, simulators, and the oracle/calibration harness extension
- Phases (each ships as a separable PR):
  - [phase-1-negative-binomial.md](phase-1-negative-binomial.md) — `GLMFamily.score` hook, `negative_binomial_family`, `family=` on the filters, `PointProcessModel(family="negative_binomial", dispersion=, update_dispersion=)`, NB simulator, parity/oracle/calibration/recovery tests
  - [phase-2-zero-inflated-gamma.md](phase-2-zero-inflated-gamma.md) — `GLMFamily.validate_observations` hook, `zero_inflated_gamma_family` + `zero_inflated_gamma_mean`, `PointProcessModel(family="zero_inflated_gamma", gamma_shape=, gamma_loc=)`, calcium-like simulator, ZIG math/oracle/recovery/calibration tests, regression guard
