# Complex environments and chronic population scale

[← back to PLAN.md](PLAN.md) · [sampling and barriers](sampling-and-barriers.md) · [phase 2](phase-2-long-recording-support.md) · [phase 3](phase-3-recurrence-versus-drift.md)

**Priority, 2026-10-12:** complex environments and long chronic recordings with
thousands of simultaneously recorded units are primary design targets. The user
specified 24 GB GPUs and A100s. Benchmark a single 24 GB GPU and an A100 separately;
record the A100's actual memory capacity rather than assuming all A100s are equal.
Recording hours, day count, host RAM and turnaround requirements remain to be set.
The workload ladder below is provisional for those dimensions, not measured capability.

The candidate contribution is reliable, geometry-aware discrimination of reusable
population maps, drift and local field changes at this scale under incomplete
sampling. Neither accepting a graph design matrix nor accelerating a small example
establishes that contribution. The [novelty audit](novelty-and-evidence.md) must test
both statistical capability and accuracy at a measured resource cost.

## Complex environments are an inference requirement

1. **Preserve topology and physical coordinates.** Include branching paths, loops,
   rooms joined by narrow passages, nearby locations separated by walls, disconnected
   components and timestamped barrier configurations. A global one-dimensional track
   coordinate is not a general representation of these environments. Use stable
   physical node IDs; derive connectivity from geometry, not missing traversals.
2. **Separate spatial resolution from computational rank.** Let G be physical nodes
   and B the fitted design dimension, including direction/nuisance columns. Reuse the
   existing graph basis as the first candidate. Measure whether its retained modes
   resolve localized fields at bottlenecks and on different branches. Compare rank
   ladders and a sparse/local or multiresolution basis prototype if the spectral basis
   requires excessive rank. These alternatives need explicit priors and validation;
   they are not already supplied by the multi-map API. More graph nodes alone do not
   establish finer effective field resolution.
3. **Represent observed direction locally.** Phase 1a's two-direction track model is
   the reference fixture. The phase-3 complex-environment adapter must define local
   edge travel direction or spatial interactions with a small circular heading basis,
   as appropriate to the apparatus. Specify behavior at junctions and during low speed;
   do not impose one global inbound/outbound sign on a branching arena. Keep travel
   direction, head direction and route history distinct. A route destination observed
   only later cannot be an earlier prospective predictor. Unsupported direction and
   route combinations retain uncertainty rather than creating new map labels.
4. **Compare alternatives where observations overlap.** Track position, direction,
   geometry and neuron support per map. Complex routes can make state, direction and
   location inseparable. Repeated common probe locations provide anchors; no overlap
   means the relevant map-change claim is unresolved. Follow the separate shared-support
   and first-visit evaluations in the sampling contract.
5. **Treat changed geometry explicitly.** A fixed complex graph is supported by the
   initial basis interface; changing connectivity requires the phase-3 adapter. Begin
   with the common-node geometry-epoch comparison already specified in the sampling
   design. Do not transfer coefficient vectors between unrelated eigenbases or
   interpret a basis change as a neural map change.

Required fixtures include a wall-separated pair of nearby paths, a loop with unequal
route use, a branching multi-room environment, a rarely visited arm, and a barrier
that changes connectivity or only changes direction/route sampling. Preserve field
resolution and predictive calibration as G, B and geometry complexity increase.

## Architecture for thousands of units and long recordings

Write T for recorded time bins (excluding overnight gaps), N for simultaneously
observed units, K for maps, and C and N_b for time and neuron batch sizes. The registered
union of neurons across days may exceed N; benchmark that dimension and overlap too.
The dense phase-1 implementation remains a small-data reference. Phase 2 must add an
out-of-core execution path; padded all-session arrays cannot be the required large-data
input representation. Define its public reader/storage contract before implementation.

- **Read bounded blocks.** Accept replayable, disk-backed counts or spike-time blocks,
  positions, unit IDs, validity and geometry metadata. Construct design rows per block.
  Sparse spike storage does not remove zero-spike exposure terms from the likelihood.
  Keep masks and exclusions synchronized with binning; avoid materializing all T×N
  counts or T×B design rows on the GPU or as a mandatory host allocation.
- **Reduce neurons into map evidence.** Sum per-neuron log likelihoods over neuron
  batches into C×K emission blocks. Do not allocate T×N×K rate arrays. With fixed map
  parameters, observed neurons contribute additively; this batching is exact up to
  numerical reduction order. Distinguish neural representational drift from unit
  instability or recording drift; spike sorting and cross-day identity remain upstream.
- **Carry HMM messages across compute chunks.** A chunk boundary inside a visit
  preserves the forward state and likelihood normalization. Exact offline smoothing
  uses a backward pass with stored checkpoints and recomputation or disk-backed
  messages. Include the pairwise transition across each internal chunk boundary once.
  Reset only at declared independent visits. Fixed-lag smoothing is an approximation
  and must have a separate accuracy comparison; it cannot silently replace this path.
- **Make posterior storage selectable.** Keep the existing dense posterior attributes
  for the small-data API. The scalable path needs an explicit summaries/disk-output
  interface: sum transition counts without keeping T×K×K pairwise probabilities,
  and stream requested map probabilities and switch summaries. Full posterior access
  must be opt-in and declare disk/RAM cost. Viterbi decoding similarly needs disk-backed
  backpointers or validated recomputation if a full path is requested. Initialization,
  ordering, diagnostics and exposure summaries must obey the same memory budget.
- **Accumulate GLM updates in blocks.** Freeze the E-step responsibilities through
  each M-step. Stream weighted gradients/objectives and batch neuron/map Hessians;
  account for extra data passes for Newton updates and line search. For repeated
  identical design rows, weighted spike counts and observed exposure can be aggregated
  exactly. Aggregate only when the complete design, geometry and covariates match;
  discretizing continuous direction or changing position resolution is a model change.
  Avoid one dense Hessian across all neurons and maps. Benchmark Hessian-vector or
  sparse solvers if B makes per-neuron dense solves the limiting cost.
- **Keep optimization and inference consistent.** Summed independent-visit objectives
  retain their own initial priors; model priors enter once. Arbitrary minibatches of
  time are not independent HMM likelihoods. A scalable gradient fit needs a validated
  checkpointed gradient or another explicit approximation. Validate EM first if that
  is the affordable full-data route; do not infer SGD scalability from forward timing.
- **Bound working memory, account for total work.** Separate persistent parameters
  (N_union×B×K), basis storage (G×B for dense eigenvectors), local emission/HMM buffers,
  solver batches, checkpoints and optional disk outputs. Audit array lifetimes,
  automatic differentiation and compiler temporaries. Dense emissions cost roughly
  O(TNBK), HMM recursion O(TK²), and full dense GLM curvature roughly O(TNKB²) per
  sweep before reuse/aggregation; eliminating a large tensor does not eliminate compute.

Longer recordings may increase disk use, recomputation and runtime. With fixed batch
sizes, active GPU working memory should not require a term linear in the full T.
Parameters, spatial rank and selected output requests still affect memory. Sparse or
low-rank approximations must report their predictive and uncertainty error against
the reference, not only throughput. Increasing bin width, dropping most units, or
reducing spatial resolution cannot silently satisfy a failed scale target.

The existing parallel Laplace core may accelerate continuous-state comparisons;
it is not the discrete HMM implementation, and parallelism can increase memory.
Benchmark sequential/chunked and parallel choices independently on the two GPU tiers.
Full per-time, per-unit, per-map coefficient covariance histories are not the default
architecture for a chronic hybrid. Start from the small low-rank/coarse-time reference
in [drift and discrete states](drift-and-discrete-states.md), then validate any scalable
approximation. Preserve correlations induced by uncertain map histories in the
reference; conditional neuron independence does not justify discarding them silently.

## Workload ladder and acceptance evidence

The [shared simulation workstream](simulation-and-validation.md) supplies controlled
mechanisms, trajectories, masks and evaluator-only truth for this ladder. Its recovery,
calibration and failure metrics accompany scale measurements. Generate chunks with
preserved stochastic state and deterministic random-stream indexing; changing batch
size must not change the synthetic experiment. Avoid a dense truth-rate tensor and
record simulation cost separately from model inference/fitting.

The following are **proposed benchmarks**, not descriptions of an available dataset.
Use 20 ms bins as one declared setting and a smaller-bin sensitivity case. Report
recorded hours separately from calendar span. Spatial nodes G and design dimension B
are independent benchmark axes; B includes all fitted direction/nuisance columns.

| Tier | Units per recording | Recorded duration | Spatial/design size | Maps | Purpose |
| --- | --- | --- | --- | --- | --- |
| Exact reference | 4–32 | Seconds to minutes | Small complex graph, B=8–32 | 1–3 | Dense/oracle comparison, masks, chunk boundaries and gradients. |
| Population | 1,000 and 3,000 | 1 and 6 hours | G=1,000–10,000; B=64–256 | 2, 4, 8 | Full fixed-K fit and scoring on a 24 GB GPU and an A100. |
| Chronic | 3,000, plus changing registered union | 30 visits/days × 6 hours | Same topology/rank ladder, uneven coverage | 2, 4, 8 | Out-of-core shared-map fit, dropout, recurrence and restartability. |
| Stress | 5,000 | Extend duration/rank separately | Larger G or B at fixed physical resolution | Extend K separately | Find and report failure boundaries rather than assert arbitrary scale. |

For perspective, the provisional 3,000-unit chronic workload has 32.4 million bins.
A float64 T×N×K array at K=6 would occupy about **4.67 TB** (decimal). Even uint16
T×N counts require about **194 GB**, before masks or raw data. These are arithmetic
storage estimates, not peak-memory measurements; they motivate disk-backed inputs and
block reductions. Waveforms and sorting outputs are outside these estimates.

Acceptance evidence:

1. At small sizes, chunked and dense likelihoods, filtered/smoothed probabilities,
   transition sums and GLM updates agree within declared float64 tolerances. Test
   arbitrary chunk sizes, missing interior bins, real visit boundaries and a switch
   straddling a chunk boundary. Frozen E-step responsibilities remain identical during
   M-step replay. Restarting from a checkpoint preserves the same fitted objective.
2. At larger sizes, measure end-to-end convergence and held-out scores, not only one
   forward pass. Report initialization, compilation, each E/M step, optimization
   iterations/restarts, I/O, checkpoint disk space, host peak RSS and peak allocated
   GPU memory. Confirm the 24 GB configuration actually fits that device, including
   temporaries; do not substitute a large-memory A100 result for this requirement.
3. Fix an acceptable turnaround budget and host-memory budget with the dataset owner
   before interpreting the benchmark as operationally usable. Report failures and
   unfinished runs. A100 throughput and 24 GB feasibility are separate results;
   multi-GPU hardware is not a prerequisite for the first fixed-K scale target.
4. Cross size with complex geometry, unit dropout, late first visits, direction/route
   imbalance, true recurrence, smooth drift and local changes. Include excess count
   variability and residual shared fluctuations so large populations do not create
   falsely certain map assignments under a misspecified Poisson observation model.
   Use matched predictive targets and simulation-calibrated false discovery rates.
5. Compare algorithms at matched spatial/temporal resolution and accuracy. Validate
   any approximate drift/local-change comparator against its small reference before
   scaling it. If only the static HMM reaches the large workload, report a software
   scale result; the large-data mechanism comparison remains unmet.

## Delivery and claim boundaries

- **Phase 1:** establish the small exact discrete reference and a graph-based complex
  geometry fixture; profile before committing to large-data interfaces. The numerical
  objective/derivative issue identified in review must be reconciled before reuse.
- **Phase 1a:** implement the bidirectional reference; preserve direction metadata for
  the later complex-environment adapter.
- **Phase 2:** deliver replayable input, batched likelihood/GLM operations, exact
  chunked HMM processing, optional posterior storage, restart support and fixed-K
  resource measurements. Shared mask helpers alone do not deliver these capabilities.
- **Phase 3:** deliver local direction/heading and geometry adapters, representation
  resolution tests, calibrated uncertainty and comparisons on complex environments.
  Add validated scalable comparators as a measured milestone after small-reference
  validation. Separate fixed-K production capacity from prototype hybrid capacity.
- **Phase 4:** enlarge the Bayesian repertoire prototype only after its cost and
  accuracy are established. Every K_max/prior sweep adds cost; do not imply that a
  3,000-unit static fit makes full Bayesian drift/count inference production-ready.

Re-estimate implementation effort for these deliverables; the original source-line
estimates cover neither out-of-core execution nor the complex-environment comparison.
Scientific novelty requires a capability audit of geometry, chronic missingness,
spatial resolution, scale, uncertainty and held-out prediction against prior methods.
