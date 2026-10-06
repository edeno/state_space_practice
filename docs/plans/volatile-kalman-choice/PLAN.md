# Volatile Kalman Choice Implementation Plan

**Status:** Not started.

Gives the bandit choice models a trial-varying, inferred learning rate: a
volatile Kalman filter (Piray & Daw 2020) transcribed from the authors'
reference code, a reward-learning `VolatileKalmanChoiceModel` whose
volatility rises after reward-contingency changes, an additive
`CovariateChoiceModel(dynamics="volatile")` that feeds that volatility into
the Laplace-EKF choice filter as its process noise, and a comparison harness
(plus a real-data script) that pits the volatile explanation against the
repo's fixed-process-noise and discrete-regime models. Existing defaults are
unchanged bit-for-bit.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need a per-component design?** [designs.md](designs.md).
3. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points (file:line), goals / non-goals, cross-plan dependencies and literature, metrics, risks, rollout, open questions, effort
- [designs.md](designs.md) — transcribed VKF recursions (Gaussian and binary), the masked/shared JAX core, NumPy oracle and hand trace, `VolatileKalmanChoiceModel`, block-change simulator, the hybrid filter's exact substitutions, identifiability gate, comparison harness and generative agents
- Phases (each ships as a separable PR):
  - [phase-1-volatile-kalman.md](phase-1-volatile-kalman.md) — `volatile_kalman.py`: VKF recursions under `lax.scan`, `VolatileKalmanChoiceModel` (SGD), simulator, reference-parity and recovery tests, export and docs
  - [phase-2-volatile-dynamics.md](phase-2-volatile-dynamics.md) — `CovariateChoiceModel(dynamics="volatile")`: volatility as the process noise of the Laplace-EKF choice filter, bit-for-bit default preservation, identifiability gate
  - [phase-3-model-comparison.md](phase-3-model-comparison.md) — `choice_model_comparison.py`: fit VKF / fixed-q / switching to one sequence, BIC and held-out LL, confusion-matrix test, real-data script
