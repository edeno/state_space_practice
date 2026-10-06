# Observation Masks and Multi-Sequence Fitting Implementation Plan

**Status:** Not started.

Adds two things users of the Gaussian oscillator models and the point-process
models (`PointProcessModel`, `PlaceFieldModel`) currently cannot do: (a) fit and
evaluate with **missing data** — dropped LFP channels, tracking loss, excluded
epochs — by passing a boolean `obs_mask` so those bins/channels are handled as
predict-only steps with a correct log-likelihood; and (b) fit **several
trials, epochs or sessions at once with shared parameters** by passing
observations with a leading `n_sequences` axis (padded, with the mask carrying
the padding). Both are additive keyword options; existing calls run the
existing code bit-for-bit.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points (file:line), goals / non-goals, metrics, risks, rollout, open questions, effort
- [shared-contracts.md](shared-contracts.md) — `obs_mask` semantics, masked-bin E-/M-step rules, `validate_observation_mask`, bit-identity policy, padding / sequence-length rules, the `sequences` helper API, `_n_timesteps`, where batching lives
- [designs.md](designs.md) — static-shape mask trick, imputation M-step, masked Laplace update, two-pass batched statistics, measured memory, oracle extensions, block-diagonal container with a sequence axis
- Phases (each ships as a separable PR):
  - [phase-1-gaussian-filter-masks.md](phase-1-gaussian-filter-masks.md) — `obs_mask` on `kalman_measurement_update` / `kalman_filter` / `kalman_smoother`; dense-oracle and LL-identity tests
  - [phase-2-gaussian-em-masks.md](phase-2-gaussian-em-masks.md) — masked `kalman_maximization_step`, masked `switching_kalman_filter` / M-step, `obs_mask` on the oscillator models' `fit` / `fit_sgd`
  - [phase-3-point-process-masks.md](phase-3-point-process-masks.md) — masked Laplace-EKF updates, point-process filters/smoothers (dense + block), `PointProcessModel` / `PlaceFieldModel` `fit` / `fit_sgd` / `score`
  - [phase-4-multi-sequence-oscillators.md](phase-4-multi-sequence-oscillators.md) — `sequences` module, batched sufficient statistics, multi-sequence oscillator fits, memory smoke test
  - [phase-5-multi-sequence-point-process.md](phase-5-multi-sequence-point-process.md) — sequence axis for `BlockDiagonalCovariance`, batched `dynamics_only_m_step`, multi-sequence `PointProcessModel` / `PlaceFieldModel`
