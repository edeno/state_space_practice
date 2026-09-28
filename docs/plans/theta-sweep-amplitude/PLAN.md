# Theta-Sweep Amplitude Implementation Plan

**Status:** Not started.

Adds `state_space_practice.theta_sweep`: a state-space estimator, with uncertainty, of how far ahead of or behind the animal the hippocampal population representation sweeps within each theta cycle on a 1-D linearised track. Given binned spikes, linear position, signed running velocity, an LFP theta phase and fixed 1-D rate maps, `ThetaSweepModel.fit` returns per-bin sweep coefficients with posterior covariances, the derived sweep amplitude and phase offset with credible intervals, the represented position, the marginal log-likelihood and a score against the no-sweep null. Two follow-on PRs add a per-theta-cycle random offset state (the cycle-to-cycle "sampling" variance) and EM for the process and cycle-offset variances with a likelihood-ratio test of cycle-to-cycle variability.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).
5. **Need upstream-repo line refs / on-disk format details?** Not applicable — there is no appendix; every reference is into this repository and is listed in the phase files.

## Files

- [overview.md](overview.md) — scientific background and literature, integration points into the existing Laplace-EKF code, goals/non-goals, cross-plan dependency policy, metrics, risks, rollout, open questions
- [shared-contracts.md](shared-contracts.md) — the design-matrix layout, `LinearRateMaps`, `theta_sweep_smoother`, `ThetaSweepResult`, `ThetaSweepModel` and the simulator's output schema, each with the fields later phases add
- [designs.md](designs.md) — derivations and complete code for the observation model and its Jacobian, posterior summaries, the null model and score, bout handling, phase shuffling, 1-D rate maps, the simulator, the oracle generalisation, cycle boundaries, time-varying dynamics, the M-steps and the cycle-variance test
- Phases (each ships as a separable PR):
  - [phase-1-sweep-amplitude.md](phase-1-sweep-amplitude.md) — the module, `ThetaSweepModel` with a random-walk prior on the sweep coefficients through `stochastic_point_process_smoother`, the simulator, recovery / null / shuffle / calibration / oracle tests, docs and a real-data smoke script
  - [phase-2a-cycle-offset-state.md](phase-2a-cycle-offset-state.md) — theta-cycle boundaries from phase, the per-cycle random-offset state with a reset at cycle starts, and the time-varying forward scan that carries it (the constant-dynamics path is re-routed through it and pinned to the library smoother)
  - [phase-2b-hyperparameter-em-and-sampling-test.md](phase-2b-hyperparameter-em-and-sampling-test.md) — exact EM for the process variance and the cycle-offset variance through `run_em`, and the likelihood-ratio test of cycle-to-cycle sampling variance (`sigma_c^2 = 0` vs free)
