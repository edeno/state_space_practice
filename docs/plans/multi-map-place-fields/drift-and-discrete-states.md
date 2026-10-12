# Drift within maps versus additional discrete states

[← back to PLAN.md](PLAN.md) · [phase 3](phase-3-recurrence-versus-drift.md) · [phase 4](phase-4-unknown-map-repertoire.md)

**Decision, 2026-10-12:** a sticky prior controls persistence but does not distinguish
continuous field change from additional maps. A sequence of long-lived static states
can approximate a drifting field. Add a small joint drift/switching prototype before
interpreting phase 4's state count or discoveries as a neural map repertoire. The
production-scale hybrid remains separately gated; evidence for static-map recurrence
is not a prerequisite for studying this confound on simulations.

## Neuroscience precedents and their scope

- [Linderman, Johnson, Wilson and Chen 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4801699/)
  applies an HDP-HMM with Poisson observations to rat hippocampal population activity.
  Its latent states represent spatial location/topology; our position is observed
  and a state indexes an entire directional population map. It discusses sticky
  HDP-HMMs as an extension, rather than using the sticky variant as its main model.
- [Wiltschko et al. 2015](https://datta.hms.harvard.edu/wp-content/uploads/2018/01/pub_23.pdf)
  uses an HDP prior with a sticky bias in an autoregressive HMM for mouse behavior.
  This provides neuroscience precedent for persistence and unknown state count.
- [Linderman et al. 2017](https://proceedings.mlr.press/v54/linderman17a.html)
  develops recurrent switching linear dynamical systems that combine discrete
  regimes and continuous dynamics. This is a relevant inference/modeling lineage,
  not an existing solution for drifting place-field coefficients or unknown map count.
- [Costacurta et al. 2022, with Linderman](https://papers.neurips.cc/paper_files/paper/2022/hash/96b3aa81a9e593ca5e9b184756034a43-Abstract-Conference.html)
  shows that omitting continuous behavioral variability can produce duplicate
  ARHMM states; warped ARHMMs capture that variability with fewer states. This
  motivates our confound test by analogy, not as direct evidence about place fields.

The [contribution audit](novelty-and-evidence.md) adds close place-field state-space
and local-change precedents. The target comparison includes local field/assembly
switching as well as continuous drift: combinations of local discrete changes can
also appear as additional states in a shared whole-map HMM.

## Minimal joint prototype

Keep phase 1's static HMM as the first implementation. In phase 3, add a small
fixed-K analysis reference that allows both map switches and within-map drift:

```text
X[t] = [Z(position[t]), direction[t] * Z(position[t])]
w[n,k,j] = template[n,k] + D @ delta[n,k,j]
log rate[n,t] = X[t] @ w[n,s[t],block(t)] + nuisance[n,t]
delta[n,k,j+1] = delta[n,k,j] + epsilon[n,k,j]
epsilon[n,k,j] ~ Normal(0, Q[k] * elapsed_clock[j])
```

`j` indexes a declared trial/visit or other coarse drift interval. `D` is a fixed,
small set of spatial/directional coefficient modes, chosen without test spikes;
start with one or two modes and small neuron/map counts. `s[t]` remains a shared
population map state with independent initial draws at visit boundaries. Use proper
template/innovation priors and an anchored initial deviation (zero in the first
reference prototype) so an arbitrary constant cannot move between template and drift.
These conventions define a statistical decomposition, not biological identifiability.

Allow direction contrast and shared spatial coefficients to drift. Compare gain-only
and spatial/directional deformation modes; a gain-only hybrid cannot exclude field
shape drift. K=1 gives drift only, Q=0 gives static switching, and K>1 with Q>0 gives
both. Phase 4 can first repeat this small reference at a finite capacity with its
sticky HDP approximation; scaling to all coefficients and chronic durations is a
separate inference/resource task. A shared drift across all maps is a useful cheaper
control but cannot substitute for the map-specific alternative when residuals differ.

Time/experience clocks and inactive maps are model choices. With calendar-time drift,
propagate uncertainty during unobserved visits and dormant periods without inventing
discrete transitions across gaps. With an experience clock, declare what advances
it; if activity depends on the latent map, inference must include that dependence.
Freezing inactive fields asserts zero drift there and is not just a memory optimization.
After long gaps, a returning map may be poorly distinguished from a new one.

## What counts as evidence for an extra state

1. **Give the existing maps a fair continuous alternative.** Compare drift only,
   static switching and joint switching/drift on identical data and position/direction
   support. Tune or infer Q, persistence and concentration together using training/
   validation data. Do not fit an overly rigid drift model and assign all residual
   variation to new states, or subtract a fitted drift trajectory before clustering
   as if that trajectory were known exactly.
2. **Integrate uncertainty in the prediction.** Compare continuation of each old
   map, return to a dormant map and an additional map using predictive distributions
   over fields and state histories. Use phase 3's held-out-neuron score and phase 4's
   later confirmation blocks. A large residual under a plug-in field estimate or
   crossing an arbitrary map-distance threshold is insufficient.
3. **Look for reproducible population structure.** An extra state gains support when
   a distinct population pattern predicts targets beyond plausible continuous
   deformation. Coordinated changes across cells and A–B–A reinstatement on matched
   support are informative. Recurrence is not required for a genuinely novel state;
   later predictive confirmation is. Abruptness alone can also reflect local field
   plasticity, changed gain or geometry and must face those alternatives.
4. **Measure the tradeoff rather than conceal it.** Report scores and count/switch
   uncertainty across Q, persistence, concentration, drift rank and temporal resolution.
   Increasing Q can absorb jumps; decreasing Q can create staircase states. If both
   explanations predict comparably, report drift/state ambiguity instead of forcing
   a map count. Evidence is conditional on the tested continuous family; failure of
   a very restricted drift basis does not rule out continuous change generally.
5. **Use repeated coverage in long recordings.** Repeated visits to the same
   positions and directions estimate ordinary within-map variability and test
   returns. Examine whether apparent states simply partition recording time along
   a continuous trajectory. More states with more data is a diagnostic of possible
   misspecification, not proof of biological repertoire growth. Never impute a
   transition time or map birth inside an unobserved interval.

## Inference and validation gates

The exact phase-1 HMM conditions on a known field trajectory. Marginalizing continuous
drift creates dependence across time; it cannot be implemented by independently
inflating each bin's emission variance. Retain joint trajectory uncertainty in
prediction. A low-rank drift basis reduces cost but does not make joint inference exact.

Start with very small K, neuron count, rank and sequence length. Check discrete
path enumeration at fixed continuous trajectories, then validate integrated
likelihoods/predictions against quadrature on tiny feasible fixtures or independent
posterior sampling with quantified Monte Carlo error. Check Q=0 and K=1 nesting.
Measure approximation error, posterior mixing/calibration and memory before any
large-data claim; use [designs K](designs.md#k-slow-per-map-drift--sketch-not-scheduled)
for the full-covariance/block-approximation questions. A continuous Poisson model
does not become exact merely by enumerating its discrete paths.

Cross drift rate/shape, switch size, revisit interval, directional learning, firing
rate and sampling coverage in simulations. Include drift-only, switching-only,
switching plus drift, abrupt local plasticity without a new recurring map, and long
gaps/dormant returns. Report false state discoveries, missed switches, predictive
calibration and a region of parameter/sampling settings where separation is unresolved.
Check sensitivity to coarse drift blocks and geometric HMM dwell assumptions; do not
interpret segmentation caused by a poorly specified timescale as map discovery.

**Gate:** phase 3's simpler comparisons can be reported first. Phase 4's static-field
posterior count must be labelled conditional on static emissions. Biological
repertoire/growth interpretations additionally require the joint prototype and
adequate predictive performance on these confound tests. Failure or insufficient
resources leaves that interpretation unresolved; it is not solved by stronger stickiness.
