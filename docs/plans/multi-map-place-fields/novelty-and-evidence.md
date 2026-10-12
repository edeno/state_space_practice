# Candidate contribution and novelty checks

[← back to PLAN.md](PLAN.md) · [scientific comparison](phase-3-recurrence-versus-drift.md) · [drift and discrete states](drift-and-discrete-states.md)

**Status, 2026-10-12:** targeted literature audit; novelty is not established.
The HMM, HDP prior, switching/continuous dynamics, directional fields and map
recurrence each have substantial precedent. Their combination is a useful research
tool, but a methods contribution requires a demonstrated capability beyond the
closest alternatives. Do not describe this plan as the first model of multiple
place-field maps or the first distinction between drift and switching.

## Closest work and implications

| Work | Established overlap | Consequence for our claim/comparison |
| --- | --- | --- |
| [Fuhs et al. 2005](https://journals.physiology.org/doi/full/10.1152/jn.00132.2005) | Poisson-HMM reconstruction used reference population maps to infer which environment/map generated test activity. | Inferring a map sequence from place-cell spikes is not itself new. Include a reference-template decoder where applicable. |
| [Linderman et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4801699/) | Hippocampal HDP-HMM with Poisson observations and unknown state count; states describe spatial location/topology. | Nonparametric count inference is established. Our observed-position/map-identity distinction must be explicit. |
| [Farhoodi, Plitt, Giocomo and Eden 2020](https://papers.nips.cc/paper_files/paper/2020/hash/7a8b8402b2f0fc78cf726ee484a0a2b7-Abstract.html) | State-space inference of fluctuating representations, multiple firing patterns per environment and uncertainty in CA1 morph experiments. | A close methods comparator. Audit its position/history-dependent Gamma observation model, trial-pattern clustering and environment-state assumptions before adapting it to spikes. Label any adaptation. |
| [Sheintuch et al. 2020](https://doi.org/10.1016/j.cub.2020.02.018) | Multiple CA1 maps recur across visits over weeks. | Recurrence in long recordings alone is not the new biological result. |
| [Costacurta et al. 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/96b3aa81a9e593ca5e9b184756034a43-Abstract-Conference.html) | Continuous behavioral variation can create duplicate ARHMM states; a warped model addresses this. | Discrete-versus-continuous confounding is established beyond place fields. Show what changes for chronic field inference with incomplete sampling. |
| [Zheng et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11485410/) | Change-point/model comparisons distinguish abrupt field changes from gradual population decorrelation, with co-switching and field reappearance. | Add local field/assembly change alternatives. A shared map state must predict beyond combinations of smaller changes. |
| [Chen et al., online 2025; 2026 issue](https://pmc.ncbi.nlm.nih.gov/articles/PMC12884766/) | Longitudinal field retention/re-expression models, including HMM comparisons and an extension with inter-field interactions. | Compare against history-dependent local field evolution; chronic data and field return prediction alone are insufficient novelty. |

These entries identify overlap, not an exhaustive list of capabilities absent from
each paper. The current audit searched combinations of place fields, remapping,
HMM/state-space models, drift, sampling, recurrence and longitudinal field evolution,
and checked the linked primary papers. Before making a priority claim, complete
backward/forward citation tracing, inspect supplements and released implementations,
and update the search through the manuscript's submission date. Record publication
versus preprint versions and separate verified absence from unexamined features.

## Proposed narrow contribution

**Candidate:** scalable, geometry-aware inference of reusable population maps versus
continuous drift and local field turnover in complex environments, from chronic
recordings with thousands of units, incomplete position/direction sampling and
changing neuron availability.

Complex geometry and population/chronic scale are primary differentiators to test,
not optional application examples. The
[complex-environment and scale design](complex-environments-and-scale.md) defines
branches/loops/rooms/barriers, resolution and direction checks, a provisional chronic
workload ladder, and separate 24 GB GPU/A100 benchmarks. A graph basis, a GPU kernel,
or an increased neuron count is not alone a verified methods advance. Audit prior
geometry-aware and scalable inference methods as well as the place-field studies
above before claiming uniqueness; that expanded capability audit is still pending.

This is a hypothesis about a useful methodological gap, not a verified first claim.
It should produce decisions with uncertainty and an explicit unresolved category,
not merely another segmentation. A scientific application could test whether repeated
barrier exposure recruits an earlier population map, beyond local adaptation and
changed sampling, and whether that inference predicts subsequent behavior. That
biological result is contingent on data; a manipulation alone does not establish it.

Separate three possible contributions:

- **Method:** demonstrably better calibration or discrimination of these mechanisms
  under realistic incomplete observations, with necessary inference innovations
  identified precisely. An existing method with a new application should be described
  as an adaptation unless the comparisons establish a methodological advance.
- **Biology:** a new, reproducible finding about map reuse, local reorganization or
  learning, with later-visit/animal prediction and the competing mechanisms tested.
- **Software:** a reliable, efficient implementation and reproducible benchmark.
  This has value even when a new model or biological claim does not pass; runtime
  improvements require matched accuracy and measured resources.
  Report end-to-end fitting, memory/I/O and uncertainty quality at matched physical
  resolution. A scalable calibrated approximation may be a methods contribution
  if the capability comparison demonstrates an advance; label pure implementation
  improvements separately.

## Smallest decisive benchmark

Before expanding to a large adaptive library, compare a small fixed-K joint model
against static maps, continuous drift, the close state-space representation baseline,
and local field-change alternatives. Use the same spike observations, positions,
direction encoding, temporal splits and scoring targets. A comparison to an imaging
method needs a justified observation adapter; do not compare Gamma fluorescence
likelihoods directly to Poisson spike likelihoods.

Generate independent and correlated field changes, map recurrence, continuous drift,
directional learning and mixed mechanisms. Cross these with late first visits,
barrier-driven coverage changes, missing cells and long gaps. Include simulations
outside our own fitted model family. Report false map-discovery rates, power at
matched false-positive rates, predictive scores, uncertainty calibration and runtime.
Map the observation regimes where the alternatives cannot be distinguished.

Then scale validated comparisons across neurons, recorded time, days, physical graph
size and basis rank independently. Include branching multi-room and wall-separated
paths, loops, bottlenecks and rarely sampled arms. Preserve the same physical field
resolution and scientific targets as size grows. Run on a 24 GB GPU and an A100 with
declared host/disk budgets. A large static-map run with missing drift/local-change
comparators is a software feasibility result, not the completed mechanism comparison.

Local change is a substantive competitor, not just a null with no population
coordination. Start with independent field-level amplitude/on-off processes and a
small correlated/group-level alternative, with field supports/groups learned only
from training data. On/off states change field expression, not neuron observation
masks. Give these models valid predictive spike likelihoods and the same behavioral
covariates. Do not simply count detected change points as a competing likelihood.
The first benchmark does not require a general-purpose factorial HMM implementation.

For recurrence, learn all cell fields on earlier visits, infer current state from
one neuron subset and predict another subset's later activity. Rotate subsets and
test recurrence across spatial/directional anchors; inspect whether support for
maps comes from only one field or a small group. Correlation between changes alone
cannot distinguish a whole-map state from a changing assembly.

## Decisions before claiming novelty

1. Write one contribution sentence and a capability matrix for the closest methods,
   supported by method/supplement/code references. Identify the exact inference or
   experimental difference; a list of familiar components is insufficient.
   Include topology/geometry changes, local direction, spatial resolution, chronic
   missingness, unit/time scale, peak memory, uncertainty and later-visit prediction.
   Record untested capabilities as unknown rather than absent.
2. Freeze the small benchmark, primary metric, acceptable calibration and practical
   improvement tolerance before comparing final results. Allow competing methods a
   comparable tuning budget and reproduce at least one published result or trusted
   fixture per implemented baseline. Mark missing comparators explicitly.
3. Ablate continuous drift, local changes, directionality and coverage/field uncertainty.
   Establish which component creates a useful advantage and where it fails. Defer
   unknown K unless count uncertainty addresses a demonstrated failure of fixed K.
4. Use an available chronic dataset for a prespecified predictive question; reserve
   later visits/animals and event repetitions. If the needed recordings do not exist,
   report a methods benchmark, not an unperformed biological validation.
5. Update the literature matrix before major implementation expansion and manuscript
   submission. Prepare a concrete method/figure comparison for expert feedback;
   this plan does not authorize contacting authors or collaborators.

**Go/no-go:** pursue a methods novelty claim only if the capability audit and fair
benchmark establish a material advance. If existing methods match it, narrow the
claim to a biological finding, benchmark or software contribution, or revise the
question. A negative result can be informative; novelty is not guaranteed by model
complexity, a favorable training fit, or failure to find an identical paper title.
