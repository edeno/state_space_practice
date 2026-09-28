# Multi-Map Place Fields Implementation Plan

**Status:** Not started.

Adds `MultiMapPlaceFieldModel`: a population-level switching model in which a shared hidden
map state picks, at every time bin, which of K spatial maps generates the recorded spikes at
the animal's actual position. Users hand it a spatial design matrix (2D spline or graph
basis) and binned spikes, and get back per-map rate maps, the smoothed/filtered map
posterior, the Viterbi map sequence, a per-transition map-switch probability, helpers to
align switching with behaviour (e.g. running speed), BIC / held-out likelihood for choosing
K, and both EM and SGD fitting. K = 1 is an ordinary penalised place-field GLM fit. A gated
follow-on phase makes the map-switch hazard depend on behavioural covariates.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, non-goals, integration points into the existing code, sibling-plan dependencies, metrics, risks, open questions, the deferred drift follow-on
- [shared-contracts.md](shared-contracts.md) — the HMM forward–backward interface (constant or time-varying transitions) and the model's fitted-attribute names/shapes
- [designs.md](designs.md) — complete code for the log-space forward–backward, per-map likelihoods, weighted penalised GLM M-step, k-means initialisation, model wiring, simulator, exact test oracle, behaviour helpers, spline penalty; drift sketch
- Phases (each ships as a separable PR):
  - [phase-1-static-maps-hmm.md](phase-1-static-maps-hmm.md) — the complete static-maps model: `utils` HMM helpers, `MultiMapPlaceFieldModel` (EM + SGD), simulator, model selection, behaviour helpers, tests, docs, real-data smoke script
  - [phase-1b-speed-gated-transitions.md](phase-1b-speed-gated-transitions.md) — covariate-dependent (speed-gated) map transitions through the recurrent-transitions interface; gated on `docs/plans/recurrent-switching-transitions/` phase 1
