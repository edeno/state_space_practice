# Spatial sampling, learning and barrier manipulations

[← back to PLAN.md](PLAN.md) · [phase 1b](phase-1b-speed-gated-transitions.md) · [phase 2](phase-2-long-recording-support.md) · [phase 3](phase-3-recurrence-versus-drift.md)

**Decision, 2026-10-12:** learning and controlled changes to the environment are
scientific targets of the multi-map comparison. Unequal spatial coverage is part of
the inference problem. A field first observed after a barrier is removed may have
existed before; its first observation does not identify when it formed.
The same applies to first sampling an already visited position in a new direction.

## Sampling contract

Keep these cases separate in the data model and every report:

| Case | Likelihood and interpretation |
| --- | --- |
| Visited location, observed neuron, zero spikes | Positive exposure and zero counts: evidence about a low rate, with uncertainty depending on exposure. |
| Location never visited in the relevant condition/map | Zero exposure, hence no direct likelihood information about its field. This is not a zero firing rate. |
| Position visited only in the opposite direction | No direct observation of this direction's field; shared-field assumptions may transfer information but do not create exposure. |
| Location visited but neuron unobserved | That neuron's exposure is zero; other observed neurons still contribute. |
| Location or route inaccessible under the current geometry | Record physical accessibility separately. No exposure does not imply an absent place field or low intrinsic rate. |
| Previously sampled location revisited much later | Historical field information exists, but confidence depends on the assumed stability/drift model and time since observation. |

Use time-bin likelihoods conditioned on actual positions. If aggregating stationary
Poisson observations, use observed seconds as exposure; a zero-exposure bin contributes
zero to the log-likelihood. Do not add a zero-count observation with positive exposure,
an occupancy epsilon, or a fabricated visit to make division/log offsets convenient.

Track raw exposure, independent visits/traversals, geometry version, observed direction
and cell availability. Also report effective per-map exposure for neuron `n`, map `k`,
geometry `c`, position bin `r` and direction category `v`:

`E[n,k,c,r,v] = sum_t dt * obs_mask[t,n] * responsibility[t,k] * 1(geometry[t]=c, position_bin[t]=r, direction[t]=v)`.

This is a support diagnostic/EM weight, not an independent observed Poisson count.
For prospective evaluation compute support using training/past data only; descriptive
retrospective support may use smoothed responsibilities but must be labelled as such.
Whole-session occupancy is insufficient when maps, direction or available cells differ.
Choose exposure/visit thresholds using validation or simulations, not held-out firing.
Retain pooled spatial exposure as a separate summary; it does not certify support in
each direction. Invalid direction follows the
[phase-1a observation rule](phase-1a-directional-fields.md), with identical evaluated
rows across comparison models. Report support for the chosen direction/route coding.

## Places first visited later

1. **Represent the physical domain independently of visitation.** When the apparatus
   layout is known, declare a common grid/node registry from its geometry, including
   accessible but unvisited locations. Do not build the training domain from the union
   of future animal trajectories. Geometry metadata supplied by the experiment is
   distinct from evidence that the animal has learned it. If the physical domain is
   genuinely discovered later, version its expansion; extend the prior and posterior
   with consistent means, cross-covariances and nonzero uncertainty, or refit a clearly
   retrospective model. Never initialize a new location as known silent.
2. **Predict with explicit uncertainty.** A proper spatial prior can borrow information
   from observed locations along allowed paths. Label estimates as directly supported,
   spatially interpolated, or prior-dominated. Unvisited disconnected components have
   no spatial data support; a population-level prior may still inform their scale.
   Do not report a narrow plug-in rate or map posterior as calibrated evidence there.
   Proper priors are required for unsupported coefficients, including disconnected
   component intercepts; an unpenalized unobserved intercept is unidentified.
3. **Score before updating.** For a first-visit experiment, save the predictive
   distribution based on earlier data, score the newly observed counts, then update
   on that declared adaptation block and evaluate on later blocks. A frozen test of
   recurrence never updates from final-test targets. The adaptive experiment is a
   separately reported prequential analysis with the same update schedule for every
   model. Earlier predictions/coverage flags must not change when future data change.
4. **Retain uncertainty in map attribution.** A position/direction sampled under map A
   can still be unsupported under B. Use neural evidence at locations/cells with established
   map distinctions to anchor identity; report ambiguous attribution on unsupported
   excursions. Lack of coverage alone cannot justify adding a new map. Map priors
   and transition covariates may influence predictions there, but that is not new
   neural evidence for switching. Resume testing map identity on return to shared
   support, and use subsequent revisits to learn the new region's tuning.
5. **Separate current estimates from past evidence.** Retrospective fits may use a
   later visit to estimate what an earlier unobserved field could have been under a
   static-map assumption. They cannot establish that the field existed, was absent,
   or changed at a particular time during the unobserved interval. When location and
   experimental condition are perfectly confounded, the remapping effect there is
   not identified without additional sampling or explicit, testable assumptions.

## Field uncertainty is an additional deliverable

Phase 1's forward–backward posterior is exact **conditional on fitted map weights**.
It does not integrate uncertainty in those weights. This distinction matters most in
unvisited regions. Phase 2 must expose support diagnostics; phase 3's first-visit
prediction arm additionally needs a validated predictive-uncertainty prototype.

Start with proper Gaussian spatial priors and a small reference calculation that
integrates static map-weight uncertainty; assess a Laplace approximation against
quadrature or posterior sampling on tiny problems. Report approximation/calibration
failures under sparse exposure and uncertain map assignments. A GLM covariance
conditional on soft labels omits label uncertainty and must not be treated as exact.
Hyperparameter and label sensitivity need separate checks; multiple optimizer seeds
alone are not a posterior over unsupported fields.

If using parameter draws, retain one draw of the static weights for the **whole
sequence**, run an HMM per draw, and integrate sequence likelihoods. Averaging emissions
independently at each time invents weights that are redrawn each bin. For the
held-out-neuron score, take the log ratio of integrated joint and conditioning-data
likelihoods, not the mean of per-draw log ratios. Reuse training-only draws/priors and
match sequence boundaries in both terms. Exact HMM inference per draw does not make
the approximation to the weight posterior exact.

The shared-support recurrence benchmark can ship first. Predictions in new space
must remain labelled as extrapolation until this uncertainty gate passes; a fitted
field plot or a ridge penalty by itself does not pass it.

## Geometry and barrier changes

- Retain stable physical node IDs and a timestamped accessibility/edge mask for each
  geometry. Barrier insertion/removal and the animal's first encounter are different
  events. Removing an edge changes connectivity; an unvisited edge is not necessarily
  blocked. Geometry comes from the apparatus/manipulation record, not absence of moves.
- A geometry-aware smoothing prior must not connect directly across a blocked edge.
  Influence can still travel around an available route or through an explicitly
  declared shared prior. Changed behavior, altered geometry and learned map state
  are separate variables; do not hard-code a map switch at insertion time.
- The current graph basis is fixed to its environment. Do not silently replace its
  eigenvectors while reusing coefficient means/covariances, or compare raw coefficients
  from different graphs. Compare predicted fields on registered physical coordinates.
  Barrier support is a new adapter/prototype, not an existing model capability.
- Start the geometry comparison on a small common node grid with proper priors built
  from each known geometry's Laplacian. Fit known-geometry epochs and compare on shared
  physical support. This tests sensitivity to geometry-aware regularisation and local
  field changes; it is not yet a model of learning the barrier or carrying a posterior
  through changing graph bases. Any later low-rank/online adapter must transform the
  full distribution and pass invariance tests before use in the HMM.
- A fixed common-coordinate basis is useful as a control. If it smooths through a wall,
  label that assumption; include a geometry-aware alternative before interpreting a
  near-barrier residual as neural remapping. Match observations and physical resolution
  across alternatives. Exclude inaccessible regions from measured before/after effects.
- An empirical movement/SR basis is a separate experience-dependent hypothesis. Estimate
  it from training/past trajectories with geometry-aware adjacency, and retain the
  [symmetrised-movement control](../successor-representation-basis/PLAN.md). A gain from
  sampling a new route is not by itself evidence for predictive coding or neural learning.
- A barrier can change the proportions of inbound/outbound travel while each
  direction-conditioned field stays fixed. Test this null explicitly. A changed
  pooled rate map is insufficient evidence for remapping; after matching direction,
  compare evolving directional contrasts as well as local geometry and map switches.

## Learning and intervention benchmark

Use baseline → insertion → repeated detours → removal → delayed reintroduction, with
unchanged/sham blocks and counterbalanced timing/order where feasible. Record insertion,
first encounter, first successful detour, reward/errors, route choice/efficiency, trial
number, cumulative exposure and elapsed time separately. [Dupret et al. 2010](https://doi.org/10.1038/nn.2599)
motivates learning-related reorganisation; [Alvernhe et al. 2011](https://doi.org/10.1111/j.1460-9568.2011.07653.x)
motivates a local barrier response rather than assuming global remapping;
[Geva et al. 2023](https://doi.org/10.1016/j.neuron.2023.05.005) motivates separating
elapsed time from experience.

Estimate learning from behavior alone first, carry uncertainty, and test whether past
neural state adds prediction of subsequent choices beyond behavioral history. Use
filtered estimates for precedence claims. A joint learning/neural model is a later
comparison, not independent evidence of a relationship its shared state assumes.
Phase 1b tests event/history-dependent transitions; the
[covariate-driven drift plan](../2026-04-04-covariate-driven-drift.md) supplies the gradual
change alternative. Neither transition covariates nor graph changes alone control
behavioral effects on firing. Distinguish randomized manipulation effects from the
stronger claim that remapping causes learning.

Report two separate evaluation targets:

| Target | Design |
| --- | --- |
| Remapping/recurrence on shared support | Restrict contrasts to position/direction/cell combinations sampled sufficiently in both conditions/maps; match or stratify speed, route and task phase. Freeze the comparison rule without target spikes. Use a declared common reference distribution and report how much data/domain it excludes. No overlap means no identified contrast. |
| Generalisation and adaptation in new space or directions | Evaluate untouched first-visit predictions and uncertainty, then improvement on later visits after declared updates. Report separately from shared-support scores; improvements here cannot establish a before/after field change in an unsampled position/direction combination. |

Use natural exposure in the generative likelihood; occupancy differences already affect
estimation precision. Balancing/reweighting is for a declared comparison estimand or
sensitivity analysis, not a replacement for missing visits. Cap any importance weights,
report effective sample size, and do not attempt weighting where exposure is zero.
Repeated probe visits to common locations provide anchors when experimentally feasible;
without them, retain the restricted conclusion. Learning may alter route sampling, so
report both the shared-support conditional comparison and overall predictive performance
instead of silently conditioning away all behavior changes.

## Ownership and validation gates

| Owner | Deliverable |
| --- | --- |
| Phase 1a | Directional emissions, explicit direction validity/coding, partial pooling and direction-conditioned predictions. |
| Phase 1b | Learning/event covariates, valid event timing, history-only predictive features, uncertainty sensitivity and event/behavior controls. |
| Phase 2 | Position/direction exposure/support metadata, distinction between masks and unvisited conditions, training-only support, shared physical coordinates and geometry version IDs. |
| Phase 3 | Shared-support and first-visit analyses, field-uncertainty, evolving-directionality, small joint drift/switching and geometry prototypes, controlled learning/barrier benchmark and resource report. |
| Phase 4 | Unknown-count and sequential-discovery prototypes; novelty versus reactivation checked against support/uncertainty and directional-learning alternatives. |
| Covariate-driven drift / SR plans | Gradual-change and movement-basis alternatives with the same support/split rules; no automatic full joint or online SR implementation. |

Add synthetic cases with (a) stationary fields and expanding route coverage, (b) a
previously unsampled high-rate region first visited late, (c) local changes near a
barrier, (d) true A–B–A returns, (e) gradual learning on unchanged geometry, and (f) no
shared spatial support. Cross them with zero-spike visits, neuron dropout, disconnected
components and irregular revisits. Add first visits in the opposite direction,
developing directional selectivity, and barriers that change only direction sampling.
Require:

- Zero exposure contributes no rate likelihood; positive exposure with zero spikes
  does. Per-neuron/per-map/per-direction support differs from pooled occupancy as expected.
- Future counts/trajectories cannot change a saved first-visit prediction, event feature
  or training-only support mask. Changing graph eigenvector signs/order leaves aligned
  field predictions and transformed uncertainty unchanged.
- With fixed independent component priors and hyperparameters, unobserved components
  retain prior uncertainty; assess predictive intervals by support stratum. Report
  any information borrowed through learned/shared hyperparameters separately. Extra
  observations improve calibration/precision in aggregate, not by requiring every
  realised interval to shrink monotonically.
- No shared support produces an explicit non-identifiable comparison. A stationary
  field with expanded sampling is included in the predeclared false-remapping rate;
  merely discovering its high-rate part must not count as evidence that a field formed.
  Apply the same rule to a first traversal in a new direction. Phase 4 must measure
  false new-map discoveries on these nulls before interpreting repertoire growth.
- Barrier effects are compared to coverage-matched stationary/local-change controls;
  geometry alone never forces a neural map transition. First-versus-repeat encounters
  test reinstatement and behavioral adaptation on held-out visits/animals.
