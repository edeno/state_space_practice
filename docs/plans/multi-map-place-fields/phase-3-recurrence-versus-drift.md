# Phase 3 — Test recurrence versus drift on long recordings

[← back to PLAN.md](PLAN.md) · [overview](overview.md#scientific-target-and-literature) · [contracts](shared-contracts.md#scientific-comparison-contract)

**Status: not started.** Deliver a reproducible comparison and evidence report, including
negative or inconclusive results. The target is whether earlier population maps return
and how much apparent field change is explained by changing map occupancy. Selecting
`K > 1` alone does not establish recurrence, attractor dynamics or memory retrieval.
Learning and barrier insertion/removal are explicit intervention benchmarks. Their
sampling, geometry and uncertainty requirements are specified in
[sampling and barriers](sampling-and-barriers.md); read that design before analysis.

**Primary application:** complex environments and chronic recordings with thousands
of simultaneous units. Follow the
[geometry and scale design](complex-environments-and-scale.md): validate small
references first, then measure larger comparisons on 24 GB GPUs and A100s. A fixed-K
scale demonstration alone does not establish a large-data drift-versus-map result.

## Entry gates and inputs

- Multi-map [phases 1](phase-1-static-maps-hmm.md),
  [1a](phase-1a-directional-fields.md) and [2](phase-2-long-recording-support.md)
  have merged. Phase 1b is optional for basic
  recurrence and required for fitted learning/event-dependent within-visit transitions.
- The masks plan's [phase 5](../masks-and-multi-sequence/phase-5-multi-sequence-point-process.md)
  supplies the masked, multi-sequence `PlaceFieldModel` baseline. A graph-basis arm
  may use `GraphPlaceFieldModel` only after an adapter provides equivalent observation
  support and boundary semantics; its current trajectory-validity mask does not
  establish arbitrary per-neuron missingness support. Neither existing estimator
  automatically supplies the augmented directional design. Provide and validate an
  analysis adapter or a matched low-level point-process reference before using it
  for the directional comparison; ordinary spatial drift alone is not that comparator.
- Identify an available dataset with repeated visits, shared spatial coordinates,
  tracked-cell metadata, behavior and enough repeated coverage. A single J16 session
  is a smoke test, not evidence of recurrence across days. If no chronic dataset is
  available, finish synthetic validation and record the real-data gate as unmet.
- Inspect [identifiability diagnostics](../identifiability-diagnostics/PLAN.md).
  Report near-duplicate/low-occupancy maps and seed sensitivity even before its full
  API exists. Any curvature report must use verified derivatives; the graph joint
  Laplace evidence currently guarantees first derivatives, not an autodiff Hessian.
- Read the [contribution/related-work audit](novelty-and-evidence.md). Map recurrence,
  fluctuating representations and abrupt field changes have close precedents;
  implement/adapt the relevant comparators before claiming a methodological advance.

## Comparison design

Use the same observations, spatial design within a fixed geometry, bin width, count
likelihood and evaluation targets for the comparison ladder. Geometry alternatives
compare predictions on a common physical grid and matched support/resolution, not
coefficients from different eigenbases:

| Arm | Explanation being tested |
| --- | --- |
| One static nondirectional map | A simple reference for the cost of omitting direction. |
| One static directional map | Stable direction-specific tuning within a single population map. |
| One map with evolving directional tuning | Directional selectivity developing with experience, without additional maps; prototype below. |
| One drifting directional map | Continuous changes in tuning using a direction-compatible point-process/graph reference. |
| K reusable static directional maps | Coordinated switches and returns beyond stable direction effects, using `MultiMapPlaceFieldModel`. |
| Local field changes | Independent and small-group field changes can generate population drift without a single reusable whole-map state; use a matched predictive spike model. |
| Small K directional maps with low-dimensional drift | Joint discrete and continuous change; analysis prototype required before phase-4 repertoire interpretation. |

Fit a declared grid of K for the last arm. Report a range of predictively comparable
counts when justified; BIC or a single best seed cannot resolve an ambiguous count.
The nondirectional arm is a diagnostic. The directional arms use the same measured
direction features and support; different temporal constraints are the hypotheses.
The [joint drift/switching design](drift-and-discrete-states.md) defines the joint
arm and its validation gate. Comparing only drift-only and switching-only models
does not test whether extra static states are approximating drift within real maps.
Likewise, shared HMM states can encode combinations of local changes. The
[local-change benchmark](novelty-and-evidence.md#smallest-decisive-benchmark) supplies
an alternative with independent and correlated field processes and specifies fair
observation-model adaptations of the closest published representation models.

Independent per-session drift fits test **within-session** dynamics. They do not model
drift over days. For a cross-day drift-versus-switching conclusion, also require a
validated elapsed-time baseline from the
[cross-session drift plan](../2026-04-03-cross-session-drift.md), or a documented
equivalent. Its summary extraction and uncertainty must pass that plan's integration
gates first. Resetting a within-session model at every day cannot substitute for it.

Add a fixed-map-per-visit reference for the Sheintuch-style timescale: each visit has
one latent map and contributes `logsumexp(log(pi[k]) + sum_t ell[t,k])`. This analysis
reference can reuse weighted GLM updates without adding a new production model class.
Compare it to bin-wise switching; do not force between-visit changes into a measured
within-visit switch hazard. Visit maps share parameters but have independent priors.
For interventions between visits, extend this analysis reference with a regularised
initial-map regression `pi_s = softmax(a + D h_s)` fitted on training visits, with
features available at visit start. Compare against shared `pi`; report initial
occupancy effects separately from within-visit switch hazards. This is an analysis
prototype, not permission to add transitions across session boundaries.

For barrier/learning data, include a local geometry-aware field-change alternative
and a gradual covariate-driven drift alternative. A shared map state can accommodate
partial remapping; many neurons need not change. Test whether a full extra map adds
prediction beyond those more local explanations. Geometry and field-uncertainty
prototypes are additional gates for the new-space analysis, not assumed capabilities
of the current fixed-basis graph model or phase-1 MAP-weight HMM.

## Experience-dependent directional fields

Phase 1a's bidirectional example is the reference case. For branching tracks and
arenas, first validate the scale design's local edge-direction or small circular
heading interaction adapter, with explicit junction/low-speed semantics and identical
observed covariates across arms. Its spatial design must preserve topology and resolve
localized fields at the chosen rank. Then apply the evolving-contrast comparison
below to those direction features. Avoid a global inbound/outbound code when the
geometry has no such orientation; route destination must obey prediction-time
availability. Complex-environment validation is required for the target application.

[Navratilova et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3282921/) motivates
testing directional selectivity that develops with experience.
[Mehta et al. 2000](https://pubmed.ncbi.nlm.nih.gov/10774737/) motivates a distinct
experience-dependent field-shape/asymmetry comparison. Treat these as possible
within-map changes, not automatic evidence for additional maps.

Start with a small K=1 prototype: keep the shared spatial coefficients `a` fixed
and let directional coefficients `u_j` evolve across trials/visits `j` under a
regularised random walk, using the phase-1a emission `Z a + d Z u_j`. Begin with
a declared low-rank spatial contrast; validate exact static nesting at zero process
variance and predictive uncertainty on tiny simulated problems. Compare a gain-only
directional contrast with spatially varying contrasts to distinguish rate changes
from shape changes. Test elapsed time versus accumulated experience as the evolution
clock, with scales selected on training/validation data. A zero-mean random walk
permits change but does not assert systematic learning; a past-behavior-driven mean
change is the separately gated covariate-drift alternative.

Use a validated low-level point-process inference adapter; audit approximation,
mask and sequence-boundary semantics rather than assuming the existing high-level
spatial model accepts this design. Treat directional coefficients as static within
the declared trial/visit, and propagate their uncertainty between blocks according
to the chosen clock. This does not create HMM transition edges across visits.
For prospective prediction, propagate from earlier blocks without fitting the
future contrast using target spikes. Keep filtered forecasting separate from
retrospective smoothing, and match information access in conditional neuron scores.

The scientific question is whether extra static maps predict beyond this simpler
directional-learning explanation. Then use the separately specified small
[joint drift/switching prototype](drift-and-discrete-states.md) to test coexistence.
This is part of the scientific comparison before phase-4 repertoire claims; it
does not require a prior positive static-map recurrence result. Full-scale hybrid
inference remains a separate resource/approximation decision.

## Tasks and evaluation protocol

1. **Build a common data manifest and split.** Save cell IDs/matching confidence,
   visit/session boundaries, timestamps, position registration, observation masks,
   occupancy, speed, direction, reward/task phase where available, and exclusions.
   For manipulations, add geometry/edge availability, insertion, first encounter,
   first successful detour, removal/reintroduction and sampling-support histories.
   Define chronological training visits, validation visits for all tuning, and final
   held-out visits. For fewer visits, use blocked folds with a declared separation
   between training and test. Fit transformations, field weights, K, regularisation
   and drift hyperparameters without final-test spikes. Do not randomly split bins.
   Predeclare shared-support versus first-visit targets using
   [the sampling contract](sampling-and-barriers.md#sampling-contract). Use geometry
   metadata to define the domain; future trajectories cannot silently define its basis.
2. **Fit and diagnose the core arms.** Use multiple initial seeds and a declared,
   comparable tuning budget. Examine convergence, map separation, effective observed
   occupancy per neuron/map/direction, joint position-direction coverage and seed
   agreement. Use blocked predictive performance to compare K and retain near-tied
   alternatives; BIC remains descriptive because its nominal
   parameter/sample counts do not capture all regularisation and temporal dependence.
3. **Validate returns without reusing the target spikes for state assignment.** Learn
   maps for all eligible cells on early visits. Partition tracked cells into a state
   inference set A and a target set B. During a later visit, infer map probabilities
   using A alone with frozen parameters, and predict B's counts at the observed
   positions. B must have training observations; this is held-out activity from known
   cells, not prediction of entirely unseen cells. Rotate neuron groups and report
   performance for each session/animal. Later visits must not refit or relabel maps.
   Restrict the remapping contrast to adequate shared spatial/behavioral support.
   Predictions in newly visited regions are reported separately and require field
   uncertainty. An optional adaptive experiment scores each new block before updating,
   with its own later evaluation blocks; it does not replace this frozen recurrence test.
4. **Use a common predictive target.** For the switching model, the exact joint
   conditional sequence score is `log p(y_A,y_B) - log p(y_A)` at fixed parameters and
   identical initial priors/boundaries, computed by two HMM passes. This integrates
   map uncertainty rather than scoring a Viterbi assignment. Sanitize target spikes
   out of the A-only pass. Test the score against path enumeration and verify A-only
   posteriors are unchanged when B changes. For drift models, implement a corresponding
   masked predictive calculation and validate its approximation on small simulations.
   Label retrospective conditional prediction separately from forward forecasting;
   whole-sequence smoothed fits or approximate evidence from different algorithms
   are not automatically comparable predictive scores.
   The HMM ratio above conditions on fitted weights. For uncertain fields use the
   [integrated predictive calculation](sampling-and-barriers.md#field-uncertainty-is-an-additional-deliverable)
   and validate it before claiming calibrated generalisation into new space.
5. **Test alternative explanations.** Include a static spatial GLM with observed
   speed/task covariates and the position-dependent directional fields above, with
   matched nuisance terms in the other arms. Test developing directionality rather
   than controlling direction only as a fixed additive covariate. Check a
   population-gain control, matched spatial/behavioral coverage,
   stricter cell-registration/unit-quality thresholds and removal of suspect cells.
   Transition covariates alone do not control direct behavioral effects on firing.
   Masking cannot repair a falsely matched cell. Disconnected cell-overlap groups
   cannot identify shared map labels without additional assumptions.
   Similarly, non-overlapping visited locations cannot identify a before/after field
   change there. Report non-identifiability rather than extrapolating a remapping
   effect. Match support by position/direction/route and report exclusions, exposure
   and independent visits; weighting cannot repair zero exposure.
6. **Summarise what long recordings add.** Report repeated-map prediction, occupancy
   by visit/day, recurrence intervals in calendar time, within-visit dwell/switch
   summaries and uncertainty. Mark boundary-censored dwell times and unobserved gaps;
   never assign a switch timestamp inside an overnight gap. Compare daily averages
   with position-conditioned mixtures of the learned maps to assess changes in map
   usage, and examine residual within-map change on matched position/direction
   support. Separate directional-contrast trajectories, direction/route sampling
   changes, map-occupancy changes and unresolved extra-map evidence. Ordinary
   HMM dwell times are geometric; do not interpret them as a fitted general duration law.
   Follow the [learning/intervention protocol](sampling-and-barriers.md#learning-and-intervention-benchmark)
   to test neural prediction of subsequent learning, local versus broad reorganisation,
   recovery after barrier removal and reinstatement on delayed reintroduction.
7. **Save a reviewable report.** Add a reproducible analysis script and a results
   directory under `docs/validation/multi-map-place-fields/` with dataset provenance,
   split/seed/configuration manifests, model-comparison tables, uncertainty, exclusions,
   runtime/peak memory and failure cases. Keep raw recordings out of git. Bootstrap
   at the animal/session level appropriate to the claim, not over correlated bins.
8. **Validate complex geometry and population scale.** Use the
   [workload and fixture ladder](complex-environments-and-scale.md#workload-ladder-and-acceptance-evidence).
   Cross branches/loops/rooms/barriers and uneven coverage with neuron/time scale.
   Compare physical resolution versus basis rank; validate local or multiresolution
   alternatives if needed. Carry the small-reference uncertainty/calibration checks
   into larger, validated approximate comparators. Report full fitting, scoring,
   compile/I/O, disk use, host RSS and peak GPU memory separately for 24 GB and A100
   configurations. Record a missing or infeasible comparator as an unmet scientific
   gate; static-model throughput alone cannot resolve drift/local changes at scale.

## Synthetic validation and scientific exit gates

Use and extend the [simulation workstream](simulation-and-validation.md) established
in phase 1. Its generator/observation/truth separation, independent model-mismatch
scenarios, final evaluation ensemble and metric/replication rules are authoritative.
Report performance surfaces and unresolved regions across sampling, mechanism strength,
geometry and resources, not only successful example segmentations. The benchmark
includes available competing models with comparable tuning budgets; missing ones stay
explicit. Freeze decisions on validation simulations before final synthetic/biological
evaluation, and count numerical failures/timeouts in the report.

Use a fixed multi-seed suite with static fields, smooth drift without recurrence,
abrupt A–B–A returns, changing map occupancy, and a mixed switching-plus-drift case.
Add behavioral gain changes, tracking gaps, unit dropout and registration errors.
Add excess count variability and residual shared neural fluctuations; test whether
the independent-Poisson model invents extra maps or becomes overconfident as neuron
count grows. Include these observation-model failures in the calibration report,
and justify any richer likelihood/gain alternative before using it as a comparator.
Add asynchronous and correlated local field changes without a reusable whole-map
state; test whether the shared-map model incorrectly reports repertoire growth.
Add one stable directional map with changing travel-direction proportions, one map
whose directional contrast gradually develops, and a first late traversal in the
opposite direction. Cross these with genuine A–B–A directional maps and barrier
detours. Measure false extra-map/recurrence calls when only directionality or its
sampling changes; direction-confounded maps must be reported as unresolved.
Include the [sampling/geometry validation cases](sampling-and-barriers.md#ownership-and-validation-gates):
stationary fields with expanding coverage, late discovery of a high-rate region,
disconnected unobserved space, local barrier responses and zero shared support.
Match position/behavior sampling across mechanisms where possible. Before examining
the biological result, fix the split, effect-size tolerance and uncertainty procedure
using these simulations and record false recurrence calls as well as power. Hybrid
data should expose residual misfit; it need not be well fit by the static-map model.

Completion requires a reproducible comparison, including uncertainty and resource
measurements. **Evidence supporting recurrence additionally requires** later returns
that predict held-out target neurons, improve over the declared competing baselines
outside the predeclared uncertainty/parsimony tolerance, and survive behavioral,
sampling and registration controls. An ambiguous result is an acceptable analysis
outcome and must be reported as such. No recurrence claim passes if the dataset or
necessary comparator is missing.

The small joint prototype can begin on simulations once mask/uncertainty semantics
and the simpler comparisons are validated. Before interpreting phase 4's counts as
maps, pass its [drift-versus-state gate](drift-and-discrete-states.md#inference-and-validation-gates).
Propose production-scale switching-plus-drift inference only after the small study
shows predictive value and a measured approximation/memory strategy is feasible,
as described in [designs K](designs.md#k-slow-per-map-drift--sketch-not-scheduled).
