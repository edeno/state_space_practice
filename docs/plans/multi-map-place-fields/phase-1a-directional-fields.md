# Phase 1a — Directional fields within a shared map

[← back to PLAN.md](PLAN.md) · [contracts](shared-contracts.md#direction-and-map-identity) · [sampling](sampling-and-barriers.md#sampling-contract)

**Status: not started.** Make observed travel direction an explicit part of the
emission model before interpreting additional latent states as additional maps.
Deliver a small design-matrix adapter, direction-conditioned predictions and a
validated comparison recipe. The HMM still represents population map identity.

## Entry gates and scope

- [Phase 1](phase-1-static-maps-hmm.md) supplies the basis-agnostic static-map HMM,
  matrix-valued GLM penalty and rate prediction from a supplied design matrix.
- Read [designs C](designs.md#c-weighted-penalised-poisson-glm-m-step) and the
  [model attributes](shared-contracts.md#model-attributes) before implementation.
  Keep the default spatial-only path unchanged; no new switching solver is needed.
- Begin with bidirectional tracks/routes with a declared physical orientation.
  Travel direction, head direction and eventual route destination are different
  measurements. Record which is used and when it becomes available.
- This phase fits stable directional fields. Experience-dependent directional
  change is the small single-map prototype in
  [phase 3](phase-3-recurrence-versus-drift.md#experience-dependent-directional-fields).
  A small [joint drift/switching reference](drift-and-discrete-states.md) is also
  owned by phase 3. This bidirectional adapter is the first reference, not the full
  complex-environment target. Phase 3 adds a small local edge-direction or circular
  heading interaction adapter under the
  [complex-environment design](complex-environments-and-scale.md#complex-environments-are-an-inference-requirement).
  Unrestricted heading interactions and production-scale hybrid fields remain later
  extensions; they are not prerequisites for this first adapter.

## Observation model

For observed direction `d_t ∈ {-1,+1}`, use

```text
log rate[n,t,k] = Z(position[t]) @ a[n,:,k]
                 + d[t] * Z(position[t]) @ u[n,:,k]
```

The shared component `a` and directional contrast `u` allow the two directions to
have different field amplitudes and shapes within one map. `a` is the midpoint of
the two log-rate fields, not an occupancy-averaged firing rate. Direction does not
double K or determine the map label. A simple direction intercept cannot capture
position-dependent selectivity and is an insufficient comparison by itself.

Use augmented rows `[Z, d*Z]`, weights `[a, u]`, and a block-diagonal penalty
`diag(P_shared, P_direction)`. Shrinking `u` toward zero partially pools directions;
setting `u=0` recovers the nondirectional field. Tune relative shrinkage on training/
validation data. Proper priors, including component intercepts, are required when
interpreting uncertainty in unsupported position/direction combinations. With only
one direction sampled locally, the decomposition can be prior-driven; report that
limitation even when the fitted total rate is well supported.

## Tasks

1. **Construct and preserve the design.** Add a stateless adapter beside the planned
   multi-map helpers, with the direction encoding, physical basis dimension and
   coefficient slices available to prediction/analysis. For the underlying model,
   `n_basis_` is the full augmented column count; weights retain `(N, 2B, K)` shape.
   Build the penalty explicitly; do not apply a spatial difference operator across
   the boundary between shared and directional coefficient blocks.
2. **Define direction validity.** Use a declared speed/quality threshold and a
   past-only direction estimate for prospective predictions. Before masks land,
   fit valid contiguous intervals; after phase 2, retain invalid bins with neural
   emissions masked and finite placeholder rows. Use identical observed rows for
   all comparison arms. Setting an unknown direction to zero is not marginalising
   over directions. A genuine stationary category or uncertain-direction mixture
   requires an explicit additional observation model.
3. **Predict and diagnose by direction.** Evaluate `[Z_grid,+Z_grid]` and
   `[Z_grid,-Z_grid]` separately through the existing rate-map interface. Save rates,
   directional log-rate contrasts, observed seconds and independent traversals.
   A pooled display must state its direction weights. Report identifiability on
   position × direction × neuron × map support, separately by geometry.
4. **Compare fixed K with and without directional fields.** Reuse blocked model
   selection and multiple initialisations. Compare K=1 directional fields against
   multiple nondirectional maps before claiming the latter reflect distinct maps.
   Fit multiple directional maps only where data can distinguish direction effects
   from map identity. If inferred maps occur exclusively in different directions,
   report confounding rather than asserting that the separation is identified.
5. **Document the adapter and smoke example.** Extend the planned multi-map module,
   its focused tests and analysis recipe; no changes to generic HMM recursion.
   Reuse phase 2 for masked/multi-session integration and phase 3 for biology.

## Validation and exit gates

Extend the [shared simulation harness](simulation-and-validation.md) with stable
directional tuning, unequal direction/route sampling and late first traversals.
Reuse its truth separation, independent evaluation replicates and uncertainty on
false-map rates; these controls become inputs to phase 3's learning comparison.

| Check | Required evidence |
| --- | --- |
| Algebra and parity | Augmented predictions match the explicit shared-plus-contrast formula; fixed `u=0` reproduces the spatial-only likelihood/posteriors. K=1 matches an independent penalised GLM with the same design/penalty. |
| Direction relabelling | Reversing the physical direction code and contrast sign preserves rates and map posteriors. |
| Stable directional map | Alternating directions with distinct fields is explained by K=1 directional tuning; quantify excess-map calls from the nondirectional fit across seeds. |
| Genuine recurrence | Direction-balanced A–B–A directional maps are recoverable under one global map-label permutation. |
| Incomplete support | Visiting a position only in one direction does not mark the opposite direction supported. A late first traversal in that direction is flagged as extrapolation. |
| Missing direction | No transition is fabricated across excluded gaps; with phase-2 masks, changes to invalid rows do not affect emission likelihoods. |
| Barrier control | A barrier that changes only direction/route sampling changes pooled fields but leaves the planted direction-conditioned fields stable. |

Calibrate model-selection tolerances on simulations before biological analysis;
finite samples need not select K=1 on every replicate. Report false extra-map rates.
[Navratilova et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3282921/) motivates
the experience-dependent follow-on: directional selectivity itself can develop with
experience. This phase supplies the stable directional baseline for that test.
