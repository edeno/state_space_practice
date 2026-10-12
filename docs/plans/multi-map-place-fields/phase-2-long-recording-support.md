# Phase 2 — Shared maps across visits and sessions

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md#long-recording-inputs-and-boundaries)

Fit one set of static maps to repeated visits with missing observations and changing
neuron availability. This phase supplies the data semantics needed for chronic
recordings; it does not introduce drift within maps or infer activity during overnight
gaps. **Status: not started.**

**Scale target:** thousands of simultaneous units on 24 GB GPUs and A100s, across long
recordings and repeated days. Read the
[complex-environment and scale design](complex-environments-and-scale.md) for the
provisional workloads and architecture. Masked padded arrays remain the small-data
reference; a replayable out-of-core path is an explicit deliverable here.

## Entry gates and inputs

- Multi-map [phase 1](phase-1-static-maps-hmm.md) has merged.
- [Phase 1a](phase-1a-directional-fields.md) can proceed alongside the mask adapter.
  Before the phase-3 comparison, integrate its design metadata, direction validity
  and direction-conditioned prediction recipe with the sequence inputs below.
- The masks plan's [C1–C4](../masks-and-multi-sequence/shared-contracts.md#c1--obs_mask-argument)
  and [C6–C8](../masks-and-multi-sequence/shared-contracts.md#c6--multi-sequence-inputs-padding-and-sequence-lengths)
  are implemented: `validate_observation_mask`, sequence canonicalisation, padding,
  lengths and reduction weights. These are currently delivered in its phases 1/3/4.
  A separately reviewed extraction of those helpers can satisfy this gate; unrelated
  oscillator integrations are not scientific prerequisites.
- Read the implemented multi-map likelihood, GLM updates, initialisation, fit/SGD,
  score and posterior accessors before editing. Historical file:line references in
  phase 1 must be rechecked against the merged tree.
- [Phase 1b](phase-1b-speed-gated-transitions.md) is optional. If already present,
  extend its covariate validation and transition statistics to sequences too.
- Read [sampling and barriers](sampling-and-barriers.md#sampling-contract): unvisited
  space, inaccessible space, unobserved neurons and observed zero spikes have different
  meanings. This phase supplies support diagnostics; uncertainty/geometry prototypes
  for first-visit scientific predictions are owned by phase 3.

The multi-map adapter is owned here. The masks plan does not automatically make this
new model accept masked or batched inputs. Its phase 5 is needed separately for the
`PlaceFieldModel` comparison in phase 3.

## Tasks

1. **Accept masks throughout the public model path.** Add keyword-only `obs_mask=None`
   to `fit`, `fit_sgd`, `score` and `predict_map_posterior`; pass it to initialisation,
   emission likelihoods and weighted GLM refits. Missing entries contribute no Poisson
   term or GLM exposure. Sanitize them before arithmetic. A fully unobserved interior
   bin still has a map state and a transition; its emission log-likelihood is zero
   for every map. The filter is predict-only there; smoothing can use later evidence.
   Preserve exact HMM inference conditional on the fitted parameters.
2. **Add a leading sequence axis.** Accept `(S,T,B)` design matrices and `(S,T,N)`
   counts with the sibling plan's padding convention. Run each HMM from the shared
   `init_map_prob_`; sum sequence log-likelihoods. Pool observation and within-sequence
   transition statistics for EM, average initial responsibilities over sequences,
   and apply each parameter prior once to the total objective. SGD must optimise the
   same objective. There is no transition from the end of one visit to the next.
3. **Make all statistics mask-aware.** Initialisation windows stay within a sequence,
   use observed exposure, and exclude unsupported neuron/location estimates instead
   of filling them with zero firing. Fit feature scaling on training data only.
   Freeze and flag a neuron/map GLM with inadequate observed effective occupancy;
   global map occupancy alone is insufficient when units disappear. Ignore padding
   in map ordering, transition counts, occupancy, BIC sample counts and SGD scaling;
   report observed exposure separately from latent-state occupancy. Implement the
   per-neuron/map/geometry/location/direction exposure diagnostic and independent-visit counts
   from [the sampling contract](sampling-and-barriers.md#sampling-contract), computed
   from training data for prospective evaluation. Use requested bins/sparse summaries
   or per-sequence accumulation, not a time-by-neuron-by-map-by-location tensor.
4. **Preserve posterior boundaries.** Batched fitted posterior attributes retain the
   sequence axis and expose `sequence_lengths_`; methods that return a path or time
   series accept `sequence=`. Trim padding in reported paths and switch summaries.
   Single-sequence calls keep their phase-1 shapes and values. Map labels are shared
   across all sequences and ordered once globally, never independently per day.
5. **Add a chronic-recording input recipe.** Use a common neuron-ID axis and spatial
   coordinate system/basis across fixed-geometry sessions. Record session/visit IDs,
   geometry version, physical accessibility, first visits, timestamps,
   travel-direction code/validity, cumulative direction-specific traversals,
   matching confidence and unit-quality exclusions alongside the arrays. Absent or
   uncertainly matched cells are masked, not encoded as zero counts. Cell matching
   itself stays upstream; report sensitivity to stricter registration thresholds.
   Invalid position masks all neural emissions at that bin, with a finite design-row
   placeholder. Never concatenate valid rows across an interior tracking gap.
   Unvisited space has no extra observation row to mask; it has zero spatial exposure.
   Store it in a geometry-defined domain with a proper prior rather than a fabricated
   zero rate. A changed graph needs phase 3's geometry adapter; do not silently replace
   a fitted eigenbasis or require identical graphs across a barrier manipulation.
6. **Bound memory.** Accumulate per-sequence likelihoods and sufficient statistics
   without constructing an `(S,T,N,K)` rate tensor. Batch both time and neurons;
   evaluate graph/directional rows from blocks and stream input counts/masks.
   Define the replayable reader and output contracts before implementation; neither
   a full padded recording nor its dense design is a required host/GPU allocation.
7. **Preserve inference across compute chunks.** Carry forward messages and likelihood
   normalizers within visits. Use checkpoint/recompute or disk messages for exact
   backward smoothing. Accumulate pairwise counts including internal chunk edges;
   reset only at real visit boundaries. Provide optional posterior/path disk output
   and summary-only execution. Extend the dense attribute contract explicitly;
   never return missing arrays as if a requested full posterior had been stored.
8. **Stream fitting and diagnostics.** Keep E-step responsibilities fixed throughout
   the M-step; accumulate weighted GLM objectives/derivatives over blocks and batch
   neuron/map solves. Replay data as needed for Newton steps and line search. Bound
   initialization and support-summary memory too. Validate exact aggregation only
   for identical complete design rows. A gradient-based large-data path additionally
   needs validated checkpointed derivatives; random time chunks are not independent
   HMM objectives. Full-data EM is the first scale gate; report SGD capacity separately.
9. **Measure resources and restartability.** Run the scale design's population/chronic
   ladder on a 24 GB GPU and an A100; record actual hardware, compilation, I/O, host
   RSS, GPU peak memory, disk output/checkpoints, full-fit iterations and held-out
   scores. Keep parameter/optimizer/configuration and iteration checkpoints to resume
   a long fit reproducibly. Set duration and turnaround targets with the dataset owner;
   the ladder's day/hour counts are provisional. Do not call independent chunks of a
   continuous visit separate sessions merely to reduce memory.

## Validation and exit gates

Extend the [shared simulation harness](simulation-and-validation.md) with controlled
dropout, unit-identity errors/overlap, gaps and independent visits. Generate the
population/chronic workload in deterministic chunks, retaining compact evaluator-only
truth. Verify simulator chunk/restart parity separately from inference parity, and
report generation cost separately from fitting. Large runs supplement the independent
small-replicate recovery/calibration ensemble rather than replacing it.

| Check | Required evidence |
| --- | --- |
| Masked exact oracle | Tiny HMM path enumeration with neuron dropout, a fully missing interior bin, and NaN placeholders matches posteriors and observed-data likelihood to `rtol=1e-10`. Changing masked count values has no effect. |
| Independent sequences | Joint likelihood equals the sum of individual likelihoods at fixed parameters (`rtol=1e-12`); transition/initial sufficient statistics equal their explicit per-sequence sums/averages. A constructed discontinuity makes naive concatenation disagree. |
| Padding | Appending masked padding leaves real-bin posteriors, objectives and M-step updates unchanged; reported switch probabilities contain no padding or cross-session edge. |
| Partial availability | Masking a neuron differs from observing zero spikes; an unsupported neuron/map estimate stays unchanged and is flagged. Supported cells still recover planted maps. |
| Spatial/directional exposure | An unvisited location differs from a visited zero-spike location; a position visited only under A remains unsupported under B. Exposure in one direction does not mark the opposite direction supported. Future visits do not modify saved training-only support. |
| Recurrence | A planted A–B–A visit sequence with overlapping tracked cells recovers the return to A under one global label permutation. Disjoint observed populations are flagged as inadequate evidence for cross-session map identity. |
| Compatibility | Unmasked 2-D inputs reproduce phase 1; single-sequence batching matches within numerical reduction tolerance; EM and SGD use the same masked objective. Phase-1a directional designs retain their coefficient blocks and prediction semantics across sequences. |
| Resources | Record runtime and peak memory, dimensions, bin width, missingness and sequence lengths for the benchmark; no unmeasured claim of chronic-scale feasibility. |
| Chunked parity | Across arbitrary chunk sizes, match the dense likelihood, posteriors, boundary pairwise counts and GLM updates, including a switch at a chunk edge and missing interior bins. |
| Out-of-core fitting | Complete declared thousand-unit fits with bounded working memory on 24 GB hardware; account for initialization, all fitting passes, I/O and optional posterior storage. Report A100 capacity/throughput separately. |
| Restart | Restoring a checkpoint preserves the objective and updates under a fixed configuration; no skipped/repeated observations or priors, changed labels, or fabricated sequence boundaries. |

Run the targeted multi-map, HMM and sequence tests, then the existing unmasked
regressions affected by the adapters. Mark fitting tests slow. This phase is complete
when the semantics and resource checks pass; a biological recurrence claim belongs
to [phase 3](phase-3-recurrence-versus-drift.md).

## Exclusions

No automated unit registration, hierarchical session-specific map priors, explicit
dwell-time distribution, continuous-time transition across recording gaps, or hybrid
switching/drifting fields. Calendar timestamps support recurrence summaries; the
within-visit transition matrix does not model overnight dynamics.
