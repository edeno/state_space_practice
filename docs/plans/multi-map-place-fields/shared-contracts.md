# Shared contracts — multi-map place fields

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Contracts referenced by more than one phase. Each appears once; phases link by anchor and
must not weaken them.

- [HMM forward–backward](#hmm-forward-backward) — used by phase 1 (constant transitions) and phase 1b (time-varying stack)
- [Model attributes](#model-attributes) — the fitted-state names and shapes phase 1 defines and phase 1b extends
- [Direction and map identity](#direction-and-map-identity) — observed directional tuning is distinct from latent map identity
- [Map count and repertoire](#map-count-and-repertoire) — fixed-K selection, count uncertainty, usage and first supported expression
- [Drift versus discrete states](drift-and-discrete-states.md) — joint prototype, neuroscience precedents and the repertoire interpretation gate
- [Long-recording inputs and boundaries](#long-recording-inputs-and-boundaries) — phase 2 masks, shared identities and sequence boundaries
- [Scientific comparison contract](#scientific-comparison-contract) — phase 3 validation and limits on interpretation
- [Spatial sampling and intervention contract](#spatial-sampling-and-intervention-contract) — shared support, first visits and learning/barrier timing

## HMM forward-backward

`state_space_practice.utils.hmm_filter(initial_probs, transition_matrix, log_likelihoods)`
and `state_space_practice.utils.hmm_forward_backward(...)` (full code in
[designs A](designs.md#a-exact-hmm-forwardbackward-in-log-space)).

| Argument / field | Shape | Semantics |
| --- | --- | --- |
| `initial_probs` | `(n_states,)` | `P(s_1 = k)`; exact zeros are structural and stay zero |
| `transition_matrix` | `(n_states, n_states)` **or** `(n_time - 1, n_states, n_states)` | row-stochastic `Z[i, j] = P(s_{t+1} = j \| s_t = i)`; in a stack, index `t` moves the chain from bin `t` to bin `t + 1` |
| `log_likelihoods` | `(n_time, n_states)` | `log p(y_t \| s_t = k)`, fully normalised |
| `HMMPosterior.smoothed_prob` | `(n_time, n_states)` | `P(s_t \| y_{1:T})` |
| `HMMPosterior.pairwise_prob` | `(n_time - 1, n_states, n_states)` | `P(s_t = i, s_{t+1} = j \| y_{1:T})`; slice `t` sums over `j` to `smoothed_prob[t]` and over `i` to `smoothed_prob[t + 1]` |
| `HMMPosterior.filtered_prob` | `(n_time, n_states)` | `P(s_t \| y_{1:t})` |
| `HMMPosterior.log_likelihood` / `hmm_filter(...)[1]` | `()` | `log p(y_{1:T})`, differentiable in all arguments |

Invariants (do not weaken):

- Exact: matches path enumeration to round-off (`rtol 1e-10`) for both the constant and the
  stacked transition input. Phase 1 pins this; phase 1b relies on the stacked branch being
  already exact and adds no new approximation.
- Discrete-state axis trails, time axis leads (project convention).
- Both functions are `jax.jit`-compiled and accept a 3-D stack without any change of
  signature — this is what makes the recurrent-transitions extension additive.
- Structural zeros in `transition_matrix` or `initial_probs` never receive posterior mass;
  an impossible observation (every reachable state at `-inf`) yields NaN (fail loud, no
  silent dynamics fallback).

`hmm_viterbi` (`src/state_space_practice/utils.py:1769-1824`) takes a `(K, K)` matrix in
phase 1. Phase 1b consumes the recurrent-transitions plan's `(n_time, K, K)`
branch: entry t is the transition into bin t and entry 0 is ignored. Its
backward step uses `[t + 1]`. The model pads its `(n_time - 1, K, K)` edge
stack with an ignored identity row before this call; the 2-D path stays
bit-identical. Reject a stack with the wrong leading length.

## Model attributes

`state_space_practice.multi_map_place_field.MultiMapPlaceFieldModel`, after `fit` /
`fit_sgd` (map axis trails everywhere). Accessing any of these before a fit raises
`NotFittedError` through `_check_fitted` for the methods; the attributes themselves are
`None`. `from_parameters(...)` sets the parameter block and leaves the posterior block
`None`.

| Attribute | Shape / type | Meaning |
| --- | --- | --- |
| `weights_` | `(n_neurons, n_basis, n_maps)` | per-neuron, per-map GLM weights; `exp(Z @ weights_[n, :, k])` is neuron `n`'s rate (Hz) under map `k` |
| `transition_matrix_` | `(n_maps, n_maps)` | row-stochastic map transitions (phase 1b: the *baseline* matrix at zero covariates) |
| `init_map_prob_` | `(n_maps,)` | `P(s_1)` |
| `smoothed_map_prob_` | `(n_time, n_maps)` | `P(s_t \| y)` |
| `filtered_map_prob_` | `(n_time, n_maps)` | `P(s_t \| y_{1:t})` |
| `pairwise_map_prob_` | `(n_time - 1, n_maps, n_maps)` | `P(s_t, s_{t+1} \| y)` |
| `switch_probability_` | `(n_time - 1,)` | `1 - trace(pairwise_map_prob_[t])`: probability that the map changes between bins `t` and `t + 1` |
| `log_likelihood_` | `float` | unpenalised marginal log-likelihood of the final parameters on the training data |
| `log_likelihood_history_` | `list[float]` | the objective the fitter maximised per accepted iteration/step: marginal log-likelihood plus the log prior (weights penalty, Dirichlet pseudo-counts); equals the marginal log-likelihood when `penalty=0` and the transition prior is flat |
| `converged_` | `bool` | EM relative-change criterion met / SGD stall criterion met |
| `n_neurons_`, `n_basis_` | `int` | recorded at fit; a refit with different shapes raises `ValueError` |
| `viterbi_path()` | `(n_time,)` int | most likely map sequence on the training data |

Label convention: map 0 is the most-occupied map (`smoothed_map_prob_.sum(0)` descending,
ties by original index). Phase 1b adds `transition_logits_` `(n_maps, n_maps - 1)` and
`transition_weights_` `(n_maps, n_maps - 1, d_h)` and keeps every row above unchanged.
Phase 1a expands the design columns, not the map axis; `n_basis_` includes directional
columns. In phase 4's sequential analysis, persistent identifiers are separate from
this per-fit display order; never infer a new identity from an occupancy rank change.

## Direction and map identity

- Observed travel direction enters the emission model. The shared hidden state
  indexes maps; neither reversing direction nor first sampling a direction forces
  a map transition. Head direction and route destination require their own measured
  definitions and availability times.
- [Phase 1a](phase-1a-directional-fields.md) uses a shared spatial log-rate field
  plus a position-dependent directional contrast with shrinkage toward zero.
  Evaluate and compare fields on position × direction support. A direction intercept
  or a transition covariate alone does not provide this observation model.
- Directional contrasts may develop with experience. [Phase 3](phase-3-recurrence-versus-drift.md#experience-dependent-directional-fields)
  tests a single-map evolving-directionality prototype before interpreting additional
  maps as a changing repertoire. This does not schedule full per-map drift inference.
- Unvisited position/direction combinations are unsupported, even when the position
  was visited in another direction. Unknown direction is masked or explicitly
  marginalised, not set to a zero contrast and called an average. Confounded direction
  and map identity must be reported as unresolved.

## Map count and repertoire

- Phase 1 accepts a supplied K; phase 3 compares a fixed-K grid using blocked
  predictive scores and reports near-tied choices. This is not posterior uncertainty
  over K, and `fit` does not silently choose a count.
- A fixed library already permits changing usage and long absences followed by
  recurrence. Distinguish total states used in the recording, windowed usage, and
  cumulative supported discoveries; each depends on observational support.
- [Phase 4](phase-4-unknown-map-repertoire.md) first prototypes a capped Bayesian
  library with sparsity/persistence and field uncertainty. Report capacity and prior
  sensitivity and label-invariant count uncertainty. Next test sequential recruitment
  and reactivation with preserved dormant identities and score-before-update evaluation.
- Persistence does not explain drift. Before biological repertoire/growth claims,
  compare counts with within-map continuous deformation in the
  [joint prototype](drift-and-discrete-states.md). Assess drift scale/rank, persistence
  and concentration together; report ambiguity when extra states and continuous
  change predict comparably. A static-emission posterior count remains conditional
  on static fields, not a model-independent biological estimate.
- First supported expression is not biological birth; last use is not death.
  New-map evidence must beat directional learning, ordinary drift, nuisance and
  sampling alternatives on subsequent held-out data with shared support. Surprise
  in an unsupported region/direction alone cannot establish a new map.

## Long-recording inputs and boundaries

Phase 2 adopts the masks plan's [C1–C4](../masks-and-multi-sequence/shared-contracts.md#c1--obs_mask-argument)
and [C6–C8](../masks-and-multi-sequence/shared-contracts.md#c6--multi-sequence-inputs-padding-and-sequence-lengths).
The unmasked phase-1 input path remains unchanged. Multi-map-specific requirements:

- `obs_mask=True` means observed. A missing neuron contributes neither a spike count
  nor a zero-spike exposure. Sanitize masked observations before validation/arithmetic;
  a fully masked emission row is zero for every map. GLM weights are posterior map
  responsibility times that neuron's observation mask.
- Single-sequence shapes are `Z: (T,B)`, `y: (T,N)`; batched shapes are `(S,T,B)` and
  `(S,T,N)`. Masks follow the sibling whole-bin/per-entry convention. Spatial bases
  are common across fixed-geometry sequences; the neuron-ID and physical coordinate
  axes are common throughout. Changing geometry requires the adapter specified in
  [sampling and barriers](sampling-and-barriers.md#geometry-and-barrier-changes).
  Mask uncertain cell matches;
  matching neurons or registering environments is upstream work.
- Retain interior missing bins on the regular time grid. Invalid position masks all
  neural emissions and uses a finite placeholder design row. No valid-row compression
  across gaps. Before phase 2, restrict fits to contiguous fully observed intervals.
- Each sequence starts from the shared `init_map_prob_`; no transition statistic,
  switch probability or Viterbi edge spans separate visits. Total log-likelihood is
  the sum of sequence likelihoods and parameter priors enter only once. Sequence
  boundaries are supplied explicitly, not inferred from neuron dropout.
- Batched posteriors prepend `(S,)` to the phase-1 shapes. Expose `sequence_lengths_`
  and trim padding in reported time series. Initial probabilities pool one draw per
  sequence; map labels are ordered globally using non-padding occupancy. Keep and
  report observed exposure separately, including unsupported neuron/map combinations.
- Record calendar times for recurrence analyses. Independent sequence priors make
  no assertion about the path between visits. Trailing all-missing bins follow the
  sibling padding rule and supply no measured dwell or switch information.

## Scientific comparison contract

- Use identical evaluation observations, spatial coordinates, direction encoding,
  likelihood and masks
  across static, drifting and switching arms; document differences in boundary priors
  and approximation. A restarted session model does not estimate between-day drift.
- Choose transformations, hyperparameters, K and label alignment on training/validation
  data. Reserve contiguous visits/blocks for evaluation. For recurrence, infer the map
  from one neuron group and predict another group's activity on later visits using
  previously learned weights. Do not assign a test map using the target neurons.
- Distinguish conditional reconstruction from forward prediction. Compare normalized
  predictive scores with matched information access, not training fit or incompatible
  approximate evidence values. BIC is supplementary; report seed sensitivity,
  low occupancy, near-identical maps and spatial/behavioral support.
- Shared statistical states do not establish attractors, subjective contexts, memory
  retrieval, or changes in represented position. A larger K may approximate continuous
  drift or developing directional selectivity. [Phase 3](phase-3-recurrence-versus-drift.md)
  owns controls, false-recurrence simulations and the scientific evidence gate.

## Spatial sampling and intervention contract

[Sampling and barriers](sampling-and-barriers.md) is authoritative for these additions:

- No visit means zero exposure and no direct rate observation. It is distinct from
  observed silence, a masked cell and a physically inaccessible location. Preserve
  per-neuron/map/location/direction/geometry exposure and independent-visit diagnostics.
- Primary remapping contrasts use adequate shared spatial and behavioral support.
  First visits test extrapolation and adaptation separately, with predictions saved
  before seeing the new spikes. No overlap means the local change is unidentified.
- Known geometry may define unvisited domain nodes and proper spatial priors. No
  unsupported field or component intercept is known zero. Predictions must expose
  field uncertainty; the phase-1 conditional HMM posterior alone does not provide it.
- A changed graph does not imply a map switch. Keep geometry/accessibility metadata
  separate from learned state, and compare fields in physical coordinates rather
  than raw coefficients across bases. Geometry-aware priors respect blocked edges.
- Learning/event predictors have recorded availability times. Use past-only behavior
  and filtered neural estimates for precedence/prediction claims; insertion, encounter
  and successful adaptation are separate events. Between-visit condition changes
  affect initial-map/occupancy analyses, not invented within-visit transition edges.
