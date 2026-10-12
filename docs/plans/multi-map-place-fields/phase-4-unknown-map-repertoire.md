# Phase 4 — Unknown map count and sequential discovery

[← back to PLAN.md](PLAN.md) · [contracts](shared-contracts.md#map-count-and-repertoire) · [phase 3](phase-3-recurrence-versus-drift.md)

**Status: not started; prototype first.** Estimate uncertainty about the number of
maps supported by the recording, then test whether later observations support a
previously unexpressed map. These are separate steps. A first supported expression
does not date biological map creation, and an unused map need not have been lost.

## Entry gates and inputs

- Phases [1](phase-1-static-maps-hmm.md), [1a](phase-1a-directional-fields.md) and
  [2](phase-2-long-recording-support.md) supply fixed-K directional maps, observation
  masks, shared identities and independent sequence boundaries.
- [Phase 3](phase-3-recurrence-versus-drift.md) supplies the blocked fixed-K reference
  and validated nulls for evolving directionality, ordinary drift, gains, changed
  coverage and geometry. A positive biological recurrence finding is not required
  to run this prototype; these alternatives must be available to assess it.
- Read [drift versus discrete states](drift-and-discrete-states.md), including the
  neuroscience precedents. Its small joint drift/switching prototype gates biological
  repertoire interpretation: stickiness alone can leave long-lived staircase states
  that approximate continuous change. A static-emission count is conditional on that
  emission assumption, even if its Bayesian posterior appears concentrated.
- The [field-uncertainty gate](sampling-and-barriers.md#field-uncertainty-is-an-additional-deliverable)
  must pass before interpreting novel/unsupported observations as evidence for an
  extra map. A fitted-weight HMM posterior is insufficient for this comparison.
- The [local field-change comparison](novelty-and-evidence.md#smallest-decisive-benchmark)
  must also be available for repertoire interpretation. A count can describe
  combinations of changing fields/assemblies rather than reusable whole maps.
- Read the current HMM code, prior/penalty contracts and phase-3 validation reports.
  Keep this work in reproducible analysis prototypes until its inference, resource
  costs and calibration justify a separately reviewed production API.
  The [24 GB/A100 scale targets](complex-environments-and-scale.md) do not imply this
  full Bayesian hybrid already fits thousands of units. Measure capacity/prior sweep
  and parameter-uncertainty costs separately, with explicit limits on any approximation.

## Quantities to keep separate

| Quantity | Meaning and limit |
| --- | --- |
| Fixed K selected by prediction | A useful approximation chosen from a declared grid; report tied/near-tied alternatives. |
| K used in the fitted recording | Uncertainty over the number of distinct states expressed in the observed sequences, conditional on the model and prior. |
| Maps used within a time window | May increase or decrease even with a fixed library. State occupancy and its uncertainty depend on the window. |
| Cumulative discovered repertoire | Supported map identities encountered up to the current analysis time; retain dormant identities for return tests. |
| Biological availability, birth or loss | Not identified merely by first/last observed use. Requires an explicit availability/birth/death model and independent constraints. |

At each bin this HMM has one active map. “Several active maps” refers to usage over
a window, not simultaneous map states. A fixed library already allows A early,
B later and A returning; variable usage alone is not evidence that K must grow.

## Stage A — A capped Bayesian library

Start with a finite approximation motivated by the
[sticky HDP-HMM of Fox et al. 2011](https://ics.uci.edu/~sudderth/papers/aoas11hdphmm.pdf).
Neuroscience precedents include [Linderman et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4801699/)
for hippocampal HDP-HMM inference and sticky autoregressive HMMs for behavior;
their state meanings and the continuous-variation lesson are distinguished in the
[literature note](drift-and-discrete-states.md#neuroscience-precedents-and-their-scope).
Its persistence prior addresses redundant, rapidly switching states; it is not a
model of when a neural map is created. Use a declared capacity `K_max`:

```text
beta ~ Dirichlet(gamma / K_max, ..., gamma / K_max)
transition[i,:] ~ Dirichlet(alpha * beta + kappa * unit_vector(i))
initial_prob ~ Dirichlet(alpha_initial * beta)
field coefficients[k] ~ proper spatial/directional Gaussian priors
```

`gamma`, `alpha`, `alpha_initial` and `kappa` have declared priors or train/validation
selection. `K_max` is capacity, not the estimated count. Shared global weights favor
reuse, concentration controls how broadly states are occupied, and persistence
discourages needless switches. Their influence must be measured, not assumed to
discover a unique true K. Begin with constant within-visit transitions; combining
this prior with phase-1b covariate logits is a separate extension.

Tasks:

1. **Build a small posterior reference.** Start with fixed field parameters and
   tiny K/T where path enumeration checks discrete inference, then a small Bayesian
   sampler that alternates state paths, transition parameters and nonconjugate
   Poisson-field coefficients. Check mixing across initialisations and label-invariant
   summaries. Validate any faster variational/Laplace approximation against this
   reference; neither EM restarts nor a low-occupancy threshold is a posterior over K.
2. **Report distributions, not a single count.** For each sampled state path, count
   distinct states over the actual sequence lengths to estimate `P(K_used | data)`;
   also report window occupancy, pairwise co-assignment, near-duplicate fields and
   dependence on prior/cap. Counts include statistical states, not proven biological
   maps. A “supported discovery” criterion additionally needs repeat/predictive evidence
   and shared support; it is distinct from any sampled path using a state once.
3. **Vary capacity and priors.** Use at least two larger capacities where feasible,
   holding total concentration conventions fixed. Record posterior mass at the cap,
   changes in predictive scores/counts, and runtime/peak memory. Mass near the cap
   or sensitivity to its enlargement means the count remains unresolved. Low mass
   at the cap alone does not establish reliable count inference.
4. **Compare on phase-3 targets.** Evaluate the same held-out-neuron and later-visit
   predictions as the fixed-K grid and directional/drifting baselines. Integrate
   parameter uncertainty coherently over whole sequences, including transition
   parameters; match information access and sequence priors in conditional scores.
   Report whether complexity buys predictive improvement and calibration.
5. **Repeat the count comparison with drift allowed.** Extend the validated small
   fixed-K joint reference to the capped prior before biological repertoire claims.
   Examine count uncertainty and predictions as drift scale/rank, persistence and
   concentration vary together. Extra states that lose support once within-map
   deformation is allowed are evidence of model misspecification, not confirmed
   new maps. An unresolved tradeoff must remain explicit.

Stage A can finish with a well-supported interval of counts or an inconclusive
result. It does not require selecting a unique K or finding growing repertoires.

## Stage B — Sequential discovery and reactivation

Start only after Stage A's calibration, joint drift comparison and resource report.
Use a chronological
score-before-update experiment with the same observation/update schedule for all
models; retain phase 3's frozen recurrence test as a separate result.

- Start from earlier visits, retain posterior uncertainty and dormant map identities,
  and permit later blocks to recruit previously unused capacity. Preserve earlier
  predictions and discovery records. Retrospective smoothing is a separately labelled
  estimate and cannot revise what was known at the earlier time.
- Compare an old map returning, a direction-specific field being learned, gradual
  change within a map and a candidate additional map. Score each incoming block
  before adaptation. Fit a candidate on a declared discovery block, then test it on
  subsequent blocks and neurons whose evaluation spikes were not used to choose or
  fit the candidate. Unknown fields at newly visited positions/directions alone
  cannot pass the discovery gate; seek return visits to shared support.
- Maintain persistent analysis identifiers. Occupancy ordering remains a display
  convention for a completed fit, not an identity tracker across prefixes. Align
  posterior draws/fits on supported field predictions and common cells/directions;
  retain ambiguity when identities cannot be matched. Do not drop or renumber a
  dormant map simply because it was unused in a recent window.
- Report first-supported-expression intervals, uncertainty in novel-versus-return
  attribution, false discovery rate, detection delay and recurrence after long
  absences. When intervening space/time was unobserved, report the gap rather than
  assigning a creation or disappearance time within it.
- Preallocate capacity for the first prototype. If capacity is exhausted, flag the
  limit and evaluate a larger-capacity refit as a new analysis. Exact online capacity
  expansion changes the prior/posterior representation and needs its own validated
  inference design; appending zero-weight states does not provide it.

## Validation and exit gates

Extend the [shared simulation workstream](simulation-and-validation.md) rather than
introducing a separate favorable count-recovery demo. Distinguish known-parameter
state calibration, learned-parameter Bayesian calibration and mechanism-specific
false-discovery/delay measurements. Retain ambiguous attribution and undefined true
map counts for continuous/local generators; do not score an arbitrary number of
approximation states as a known biological repertoire.

Cross the phase-3 nulls with: unknown fixed K, a rare old map returning after a long
absence, a late genuinely distinct map, gradual development of directionality,
first travel in the opposite direction, and non-overlapping spatial/cell support.
Use prior-predictive calibration of count/occupancy uncertainty on small problems,
plus mechanism-specific simulations that measure false discoveries and detection
delay. Include drift-only staircase segmentation, switching plus drift and dormant
map returns after long gaps, following the joint prototype's simulation grid.
Include seed/mixing, drift/persistence tradeoffs, prior and capacity sensitivity.

Completion is a reproducible inference/comparison report with explicit failure
cases. Claims of a newly supported map require later predictive confirmation beyond
the declared alternatives; inaccessible or unvisited conditions remain unresolved.
Explicit birth/death processes, indefinite online growth, non-geometric dwell times
and a production-scale switching-plus-drifting directional model each require a follow-on design
justified by residual failures of these prototypes.
