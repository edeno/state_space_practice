# Recurrent Switching Transitions Implementation Plan

**Status:** Not started.

Lets the discrete-state transition probabilities of the three switching families
(`switching_kalman` Gaussian, `switching_point_process` spikes,
`switching_choice` behaviour) depend on observed covariates such as running
speed, reward or theta power (phase 1: an input-output HMM in the sense of
Bengio & Frasconi 1995) and on the continuous latent state itself (phase 2: the
recurrent SLDS of Linderman et al. 2017). Both are additive keyword-only options
on the existing filters, smoothers, Viterbi decoders and model classes; with the
options left at `None` the fixed-matrix code path runs exactly as it does today.
One shared primitive (`state_space_practice.discrete_transitions`) builds the
time-varying transition matrices, fits their logit coefficients in the M-step,
and is used by all three families and their simulators.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, cross-plan dependencies, risks, rollout (back-compat policy), open questions
- [shared-contracts.md](shared-contracts.md) — the transition-logit contract shared by the three families: parametrisation, shapes, time alignment, keyword names, None-path invariant, M-step and validation contracts, sibling-mirroring checklist
- [designs.md](designs.md) — full code for the shared primitive, filter/smoother/Viterbi integration, the logit M-step, the oracle extension, the phase-2 collapse approximation and its exact grid oracle, the relabelling transform, and the simulators
- Phases (each ships as a separable PR):
  - [phase-1a-covariate-transitions-gaussian.md](phase-1a-covariate-transitions-gaussian.md) — shared primitive; covariate-dependent transitions in the Gaussian switching filter / GPB1 smoother / Viterbi / `hmm_viterbi`; logit M-step; `BaseModel` (COM/CNM/DIM) EM + SGD; Gaussian simulator; exact oracle with time-varying transitions
  - [phase-1b-covariate-transitions-point-process.md](phase-1b-covariate-transitions-point-process.md) — mirror in the switching point-process filter, `SwitchingPointProcessBase` / `SwitchingSpikeOscillatorModel` / COM-PP / CNM-PP / DIM-PP, and the spike simulator
  - [phase-1c-covariate-transitions-choice.md](phase-1c-covariate-transitions-choice.md) — mirror in the switching choice filter / control-aware smoother / `SwitchingChoiceModel`, and the choice simulator
  - [phase-2a-state-dependent-transitions-gaussian.md](phase-2a-state-dependent-transitions-gaussian.md) — `transition_state_weights=` (rSLDS) in the primitive and the Gaussian family with the collapsed-mean plug-in, its approximation-trend tests and a 1-D grid oracle
  - [phase-2b-state-dependent-transitions-siblings.md](phase-2b-state-dependent-transitions-siblings.md) — the same option in the point-process and choice families and their simulators
