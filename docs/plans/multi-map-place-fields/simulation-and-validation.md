# Simulation as the model-development workstream

[← back to PLAN.md](PLAN.md) · [scale targets](complex-environments-and-scale.md) · [scientific comparison](phase-3-recurrence-versus-drift.md)

**Priority, 2026-10-12:** simulated data are a primary way to determine what the
models recover, when they fail, how trustworthy their uncertainty is, and whether
they remain useful at chronic Neuropixels scale. Develop the simulator and evaluation
harness alongside the first inference primitives. Each later phase extends the same
benchmark rather than producing an unrelated favorable recovery example.

The deliverable is a reproducible map of performance versus observation conditions,
including unresolved mechanisms and failures. Passing a matched-model simulation
does not establish biological validity. Real-data predictive checks remain a separate
gate; the benchmark should guide model design and experimental sampling before that gate.

## Three distinct kinds of evidence

| Layer | Question | Reference and limit |
| --- | --- | --- |
| Numerical correctness | Does the implementation calculate the stated quantity? | Tiny independent path enumeration, direct likelihoods, derivative checks and dense solves. Agreement does not establish model adequacy. |
| Recovery and calibration | Can the fitted model recover planted structure and represent its uncertainty? | Repeated simulations within the declared model family, with parameters learned as in the intended analysis. Known-parameter checks and learned-parameter results are separate. |
| Robustness and mechanism discrimination | What happens when the data differ from the fitted assumptions? | Independent field/behavior/noise generators, local changes and mixed mechanisms. Report false discoveries, predictive failures and ambiguity rather than forcing a map count. |

Resource measurements accompany all layers at appropriate sizes. A fast run is not
a recovery result, and accurate inference on 16 neurons is not a population-scale result.

## A composable generator with a truth record

Keep scenario configuration, observations, and evaluator-only truth separate.
The fitter receives the same information it would receive in an experiment: spikes,
measured position/direction, timestamps, observation masks, supplied unit identities,
visit boundaries, and recorded geometry/events. It never receives true maps, true
drift trajectories, corruption identities or future information through initialization,
feature construction or model selection. Explicit oracle fits must be labelled.

Build independently configurable components:

1. **Geometry and movement.** Use fixed physical node coordinates with branches,
   loops, rooms, narrow passages and walls; allow timestamped changes in connectivity.
   Trajectories include uneven route use, direction imbalance, pauses, gradual
   exploration and repeated probe visits. Separate barrier insertion from encounter
   and adaptation. Unknown direction and position-tracking errors have explicit rules.
2. **Neural fields and dynamics.** Generate stable maps, coordinated returns,
   continuous field deformation, developing directional tuning, independent or
   group-level local changes, and combinations. Specify elapsed-time versus experience
   clocks and behavior during dormant/unobserved periods. Include partial remapping,
   unequal map occupancy, weak separation and non-geometric dwell times.
3. **Spiking and shared variability.** Begin with Poisson counts; then introduce
   excess count variance, shared gain fluctuations and temporally structured spiking
   through separately documented generators. If using a renewal/history model, record
   its conditional intensity and distinguish it from a marginal mean rate. These are
   robustness scenarios, not a requirement to add every observation model to production.
4. **Recording and unit identity.** Generate neuron dropout, changing observed
   populations, tracking gaps and known registration mistakes; optionally model unit
   splits/merges and contamination. Preserve the underlying cell identities only in
   the truth record. Distinguish observation failures from biological field turnover.

Use separate deterministic random streams for mechanism parameters, movement, spikes
and observation failures. Pair selected conditions on the same trajectory, fields
and masks to isolate a mechanism; also include scenarios where learning changes
movement. Never create an impossible paired trajectory that crosses a blocked edge.

Include both basis-matched fields and fields generated independently on the physical
domain, such as localized geodesic fields and deformations not confined to the fitted
low-rank basis. Test finer spatial detail than the fitted representation and multiple
field shapes. The general benchmark cannot consist solely of data drawn by calling
the model's own prediction routine; numerical oracles also need independent calculations.

The truth record includes rates or a regenerable field/intensity evaluator, latent
histories where defined, local events, support/accessibility, true cell IDs, sampling
and corruption parameters, and complete seed/version metadata. Some generators have
no shared discrete map state. Mark that quantity undefined rather than inventing a
"true K" for continuous or asynchronous local processes. Hidden states with identical
observable fields are also not necessarily distinguishable from the observations.

## Mechanisms and confounds to compare

| Scenario | What the evaluation must reveal |
| --- | --- |
| Stable fields with changing route/direction coverage | How often sampling changes create a false remapping or recurrence claim. |
| Coordinated A–B–A maps, including partial remapping | Recovery of returns and later-neuron prediction, with sensitivity to separation and map occupancy. |
| Continuous drift without a reusable map library | Whether static states merely segment a continuous trajectory; evaluate the scientific claim, not an invented true count. |
| Directional selectivity developing within one map | Whether learning a direction contrast is mistaken for additional maps. |
| Independent and correlated local field changes | Whether combinations of field/assembly changes masquerade as shared population maps. |
| Switching plus within-map drift | Tradeoffs among map count, drift flexibility and predictive performance. |
| Late first visits, inaccessible regions and zero shared support | Appropriate extrapolation uncertainty and abstention from unsupported before/after claims. |
| Barrier insertion/removal/reintroduction | Local responses, coverage-only effects, genuine map changes and returns are separately planted. |
| Dormant map returns and later distinct maps | False discoveries, detection delay, missed returns and uncertainty in return-versus-new attribution. |
| Non-Poisson/shared variability and recording corruption | Whether more neurons or longer recordings create confident errors under misspecification. |

Vary signal separation, fraction of changing cells, field width/shape, firing rates,
drift speed/rank, dwell durations, revisit intervals, neuron overlap, missingness and
sampling coverage. Cross those with geometry complexity, direction, neuron count,
recording duration and fitted spatial rank. Use staged sweeps and targeted interactions
near failure boundaries rather than an unaffordable full Cartesian product. Publish
the sampled parameter ranges and avoid generalizing beyond them.

## Evaluation protocol and metrics

**Separate development from evaluation.** Use development scenarios/seeds for debugging,
validation scenarios/seeds for model/threshold/regularization choices, and a frozen
final ensemble of new seeds plus predeclared held-out parameter ranges or generator
variants. Within each recording retain chronological train/validation/test visits
and held-out-neuron targets as in phase 3. A new random seed at the same parameters
tests repeatability; a held-out mechanism or parameter regime tests broader robustness.
Changing the method after seeing final results requires a new versioned evaluation.

For each scenario/configuration, fit all applicable competitors with the same observed
data, information availability, splits and comparable tuning budgets. Compare the
same saved simulated recording rather than fresh draws for each algorithm. Keep
explicit missing-comparator flags. Early fixed-K fits may approximate drift with extra
states; recurrence claims are judged only once the required alternatives exist.

Report these distinct outputs:

- **Fields and rates:** error in predicted firing rates on a declared physical grid,
  using a common reference weighting, plus held-out count log scores. Separate direct
  support, interpolation and first-visit extrapolation; correlation alone misses rate
  amplitude errors. Match spatial and temporal resolution across algorithms.
- **States and events where defined:** state error, return identification, switch
  precision/recall and timing error, including missed switches and false alarms.
  Declare event matching tolerance before evaluation. Use a single global label
  alignment for descriptive recovery, never a new permutation for every day. For
  actual later-visit recurrence predictions, freeze training labels and parameters.
  Include label-invariant occupancy/co-assignment summaries on sampled pairs; avoid
  constructing an all-time-pairs matrix for large recordings.
- **Mechanism decisions:** false positive rates under each null, power on alternatives,
  and the fraction reported unresolved. If discoveries are counted in mixed scenarios,
  define the discovery unit and report false discovery proportion separately from a
  null false positive rate. Report error among decisive calls as well as abstention;
  neither always choosing a mechanism nor always abstaining demonstrates usefulness.
- **Uncertainty:** map-probability reliability/Brier scores, predictive count interval
  coverage and width, and field/count uncertainty when that inference is available.
  Known-parameter HMM calibration does not validate learned-field uncertainty. Check
  the Bayesian prototype with prior-drawn parameter/data replicates and label-invariant
  quantities; use a declared discrete/rank procedure where needed. Assess robustness
  under misspecification separately from calibration under its own prior/model.
  A MAP fit does not acquire a field posterior from optimizer restarts.
- **Resources and failure:** full-fit time, convergence/initialization sensitivity,
  memory, disk/I/O and approximation error. Report failed fits and timeouts as outcomes;
  do not drop them from the denominator and report accuracy only on survivors.

Choose independent recording/animal replicates and their count from the desired
Monte Carlo precision, with intervals on false positive rates, power and score
differences. Time bins and correlated neurons are not independent simulation
replicates. Paired comparisons retain pairing when estimating uncertainty. For
expensive large-data runs report the smaller replication budget explicitly; do not
infer reliable rare-error rates from a handful of seeds or from zero observed errors.

Finite-sample performance need not worsen monotonically on each individual seed as
signal decreases. Evaluate trends over replicates and report their uncertainty.
Controlled deterministic fixtures remain useful regression checks but cannot replace
this ensemble evidence. Add deliberately degraded predictions to check that the
evaluation detects miscalibration or poor recovery.

## First executable benchmark and progressive delivery

Start the harness at the beginning of phase 1. Use a small branching/looping environment
with wall-separated nearby paths and 32–64 neurons, plus smaller oracle fixtures.
The first four scenarios are stable fields with changed coverage, coordinated A–B–A
returns, continuous drift, and local field changes. Supply planted truth and a common
evaluation report before enlarging the model. Existing stationary/graph fits can be
initial comparison arms; unavailable multi-map or scientific competitors are marked
pending, not represented by placeholder results.

Then extend it with each phase:

- **Phase 1:** numerical oracles, matched-model recovery, initial negative controls,
  and the fixed-K EM comparison. Resolve objective/derivative consistency before
  numerical claims. A two-map recovery example alone does not complete validation.
- **Phase 1a:** stable directional fields, route/direction imbalance and first travel
  in the opposite direction. Phase 1b adds event timing and past-only history checks.
- **Phase 2:** dropout, identity overlap/errors, gaps, independent visits, chunk and
  restart equivalence, and streamed population/chronic workloads.
- **Phase 3:** the full mechanism comparison, developing directionality, local
  direction/heading, geometry changes, field uncertainty and the small joint model.
  Produce performance/failure surfaces before interpreting the corresponding real-data
  result, including the observation regimes where mechanisms cannot be distinguished.
- **Phase 4:** calibrated count inference, dormant returns versus candidate discoveries,
  prior/capacity sensitivity and discovery delay, repeating comparisons with drift.

Organize reusable generator components under the existing `simulate/` package after
checking its conventions; retain the simple `simulate_multi_map_session` fixture API.
Put small independent numerical references in the test oracle infrastructure. Keep
large ensembles in a reproducible analysis runner with explicit configuration, separate
from routine unit tests. This is a planned implementation split, not an existing API.

Large simulations must obey the [scale architecture](complex-environments-and-scale.md):
generate/consume chunks without a full T×N×K rate or truth tensor. Preserve sequential
generator state and deterministic random-stream indexing across chunks/restarts;
changing computational batch size must not change the simulated experiment. Store
compact latent histories/parameters and evaluate truth on demand. Validate generator
chunk parity before using its output to evaluate inference chunk parity. Report
generation cost separately from model fitting and avoid storing raw ensembles in git.

Save manifests, commands, generator/fit revisions, seed roles, parameter ranges,
split rules, metric definitions, convergence/failure records, replicate-level scores,
uncertainty and summary figures under `docs/validation/multi-map-place-fields/` when
the runner is implemented. Reuse configurations across model versions. Include
performance surfaces for map separation versus coverage, drift versus revisitation,
neuron overlap versus return accuracy, and spatial rank versus resource cost; show
uncertainty and unresolved regions rather than only successful example traces.

Real trajectories can later drive synthetic spikes to stress realistic coverage;
use only trajectories available to the declared split and preserve truth separation.
Calibrate generator ranges from training recordings where possible and document the
remaining assumptions. Posterior predictive checks on real recordings should expose
which observed features the simulation suite still misses. Neither these checks nor
simulation success alone establish the biological mechanism.
