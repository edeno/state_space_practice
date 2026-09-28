# WoLF Robust Measurement Updates Implementation Plan

**Status:** Not started.

Adds outlier-robust measurement updates to the library's filters via generalised Bayes: the weighted observation likelihood filter (WoLF) of Duran-Martin et al. (ICML 2024). Every filter, smoother and single-step update gains an additive keyword-only option `robust_weight=` (default `None`, which reproduces today's output bit-for-bit). With `robust_weight=imq_weight(c=...)` each observation's log-likelihood is tempered by a weight computed from its standardised residual at the *prior predictive*, giving a closed-form update with provably bounded influence. The Gaussian path (`kalman.py`, then the switching filter behind the LFP oscillator models) is delivered first; the Laplace-EKF point-process path (`glm_laplace_update`, the point-process filters, `PointProcessModel` / `PlaceFieldModel` / `PositionDecoder`) follows. EM fits driven by `run_em` monitor the generalised-Bayes objective instead of the raw log-likelihood when a robust weight is active, and the measurement-covariance M-step becomes the weighted residual form.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, cross-plan dependencies, risks, metrics, rollout, open questions, literature.
- [shared-contracts.md](shared-contracts.md) — the `RobustWeight` protocol, `imq_weight` / `IMQWeight`, `RobustOutput`, the return-arity rule, the two objective definitions and the cross-phase invariants.
- [designs.md](designs.md) — derivations and complete code: the weighted Gaussian update and generalised evidence, the weighted M-step and its bias analysis, the shared-weight switching filter, the deviance-weighted GLM Laplace update, the stop-gradient rule for SGD.
- Phases (each ships as a separable PR; phases 2 and 3 both depend only on phase 1 and can proceed in parallel):
  - [phase-1-kalman-gaussian.md](phase-1-kalman-gaussian.md) — `utils.imq_weight` + protocol; `robust_weight=` on `kalman_measurement_update` / `kalman_filter` / `kalman_smoother`; weighted `measurement_cov_residual_form` / `kalman_maximization_step`; oracle, bounded-influence and contamination tests.
  - [phase-2-switching-oscillators.md](phase-2-switching-oscillators.md) — `robust_weight=` on `switching_kalman_filter` and the switching M-step; opt-in on `CommonOscillatorModel` / `CorrelatedNoiseModel` / `DirectedInfluenceModel` (EM objective, weighted R, SGD loss); LFP-artifact simulation tests.
  - [phase-3-point-process.md](phase-3-point-process.md) — deviance-weighted `glm_laplace_update` / `_point_process_laplace_update`; `robust_weight=` on `stochastic_point_process_filter` / `_smoother` (dense and block-diagonal); opt-in on `PointProcessModel`, `PlaceFieldModel`, `PositionDecoder`; burst-injection tests.
