# Multi-Map Place Fields Implementation Plan

**Status:** Not started. Scope updated 2026-10-12 for complex environments,
Neuropixels-scale chronic recordings, directional fields, recurrence versus drift,
and an unknown/changing observed map repertoire.

Adds `MultiMapPlaceFieldModel`: a population-level switching model in which a shared hidden
map state picks, at every time bin, which of K spatial maps generates the recorded spikes at
the animal's actual position. Users hand it a spatial design matrix (2D spline or graph
basis) and binned spikes, and get back per-map rate maps, the smoothed/filtered map
posterior, the Viterbi map sequence, a per-transition map-switch probability, helpers to
align switching with behaviour (e.g. running speed), BIC / held-out likelihood for choosing
K, and both EM and SGD fitting. K = 1 is an ordinary penalised place-field GLM fit.

**Primary operating target:** branching/looping multi-room environments and long
chronic recordings with thousands of simultaneous units, on 24 GB GPUs and A100s.
The [complex-environment and scale design](complex-environments-and-scale.md) makes
geometry resolution, out-of-core fitting, exact chunk boundaries and measured resource
budgets explicit deliverables. A graph input and a small single-session demo do not
alone meet this target. Recording durations in its benchmark ladder are provisional.

The scientific target is whether changing place fields reflect continuous drift,
returns to previously expressed population maps, or both. Long recordings supply
repeated visits for estimating shared maps and testing their return on later data.
Learning and barrier insertion/removal provide controlled tests of those changes.
Unequal coverage requires separate analyses of remapping on shared position/direction
support and prediction in newly visited conditions; first observation is not field
formation. Observed travel direction can change firing within one map, and that
directional tuning may itself develop with experience.

The delivery path is **phase 1 → phase 1a → phase 2 → phase 3 → phase 4**:
fixed-K HMM, stable directional fields, scalable masked multi-session fitting, comparisons
including evolving directionality, then an unknown-count/sequential-discovery prototype.
Phase 3 also includes a small joint drift/switching comparison before phase 4's
biological repertoire interpretation: persistent states can still discretise drift.
Phase 2's mask work can proceed alongside 1a; both gate the scientific comparison.
Choose among fixed K with blocked prediction first. Only after the confound/uncertainty
benchmarks are validated, test a capped Bayesian library and later recruitment of
unused capacity. Changing usage, first supported expression and biological map birth
are separate claims. Phase 1b adds learning/event transitions on a separate branch.
Production-scale switching-plus-drift inference and indefinite online map creation
remain deferred pending the small joint prototype and resource/approximation evidence.

Novelty is a separate evidence gate. The [contribution audit](novelty-and-evidence.md)
identifies close existing methods and a candidate target: calibrated discrimination
of reusable maps, drift and local field turnover in complex environments under
incomplete chronic sampling at thousands-of-units scale, with measured accuracy/cost.
Run the small comparison before expanding model complexity or claiming a new method.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — goals, non-goals, integration points into the existing code, sibling-plan dependencies, metrics, risks, open questions, the deferred drift follow-on
- [shared-contracts.md](shared-contracts.md) — HMM interface, directional/map semantics, repertoire counts, chronic-recording boundaries/masks and scientific comparison rules
- [complex-environments-and-scale.md](complex-environments-and-scale.md) — geometry and direction requirements, out-of-core architecture, 24 GB/A100 workload ladder and separate statistical/computational acceptance gates
- [sampling-and-barriers.md](sampling-and-barriers.md) — unvisited versus silent fields, coverage and first-visit prediction, changing geometry, learning/intervention benchmark and uncertainty gates
- [drift-and-discrete-states.md](drift-and-discrete-states.md) — neuroscience HDP-HMM/hybrid precedents, low-dimensional joint drift/switching prototype and evidence needed to distinguish drift from additional maps
- [novelty-and-evidence.md](novelty-and-evidence.md) — closest prior work, provisional contribution, local field-change competitors and go/no-go criteria for scientific novelty
- [designs.md](designs.md) — complete code for the log-space forward–backward, per-map likelihoods, weighted penalised GLM M-step, k-means initialisation, model wiring, simulator, exact test oracle, behaviour helpers, spline penalty; drift sketch
- Phases (each ships as a separable PR):
  - [phase-1-static-maps-hmm.md](phase-1-static-maps-hmm.md) — the complete static-maps model: `utils` HMM helpers, `MultiMapPlaceFieldModel` (EM + SGD), simulator, model selection, behaviour helpers, tests, docs, real-data smoke script
  - [phase-1a-directional-fields.md](phase-1a-directional-fields.md) — shared spatial fields plus position-dependent directional contrasts, partial pooling, direction-conditioned predictions and confounding checks
  - [phase-1b-speed-gated-transitions.md](phase-1b-speed-gated-transitions.md) — learning- and event-dependent transitions (speed, encounter, reward/error history, experience); gated on `docs/plans/recurrent-switching-transitions/` phase 1
  - [phase-2-long-recording-support.md](phase-2-long-recording-support.md) — shared static maps across visits/sessions, masks, exact chunked inference, disk-backed inputs/outputs and measured fixed-K population scale; consumes the masks plan's shared helpers
  - [phase-3-recurrence-versus-drift.md](phase-3-recurrence-versus-drift.md) — stable/evolving directional, drifting, switching and small joint comparisons; held-out-neuron recurrence and learning/barrier tests, shared-support versus first-visit evaluation
  - [phase-4-unknown-map-repertoire.md](phase-4-unknown-map-repertoire.md) — prototype posterior uncertainty over map count, capacity/prior sensitivity, then sequential discovery versus reactivation; no automatic biological birth/death interpretation
