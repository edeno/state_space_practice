# Claude Code Execution Roadmap

> For Claude: Execute plans in this exact order within each track. Do not start a downstream plan until all entry and exit gates for upstream dependencies pass.

> **Reconciled 2026-07-08:** This roadmap now reflects the checked-in source and
> the 2026-06-21 spike-field identifiability finding. Completed plans are listed
> as existing capabilities, not active work. Spike-only latent-oscillator coupling
> plans are blocked as scientific estimators unless an observed field such as LFP
> anchors the latent.

## Purpose

This roadmap provides the authoritative implementation queue for all plans in docs/plans/, organized into two tracks:

- **Infrastructure track:** Core models and utilities that form the computational foundation
- **Scientific track:** Neuroscience-specific models that build on the infrastructure to test scientific hypotheses about CA1/mPFC replay, theta sequences, and value-modulated neural coding

Both tracks share the same dependency constraints. The scientific track has a "minimum publishable path" — the shortest route to a novel scientific claim.

## Global Rules

1. Run all commands through `uv run` from the repository root. Older plan snippets
   may still say `conda run -n state_space_practice`; treat those as historical
   and translate them to `uv run`.
2. Use test-first sequencing inside each plan task.
3. Never run two high-risk plans in parallel.
4. If any gate fails, stop and fix before continuing.
5. For speculative plans, complete prototype scope before full model integration.

## Status

| Plan | Status |
|---|---|
| Numerical Stability | **DONE** (kalman.py, switching_kalman.py, point_process_kalman.py stabilized) |
| Position Decoding | **DONE** (position_decoder.py, 35 tests) |
| Multinomial Choice | **DONE** (multinomial_choice.py, 43 tests) |
| RL Covariates | **DONE** (covariate_choice.py, 55 tests — dynamics covariates, obs covariates, decay) |
| Computational Improvements | **DONE** (parallel smoother, Woodbury, Joseph form, parameter constraints, SGD for choice models) |
| SGD Fitting All Models | **DONE** (fit_sgd() on all model classes, 80+ SGD tests, eigendecomp→Cholesky PSD fix) |
| Contingency Belief | **DONE** (`contingency_belief.py`, EM + SGD + uncertainty summaries) |
| Uncertainty-Aware Behavioral Modeling | **DONE** (`behavioral_uncertainty.py`; summaries/surprise exposed on behavioral models) |
| Switching Choice | **DONE** (`switching_choice.py`, per-state beta/Q/decay, EM + SGD) |
| Regularized Oscillator Connectivity | **DONE** (`oscillator_regularization.py`, DIM SGD penalties) |
| Hamiltonian Oscillator SSM | **DONE** (hamiltonian_spikes/lfp/joint/switching.py, 34 tests, nonlinear>linear baseline verified) |
| Hamiltonian Review Fixes | **DONE** (all 7 phases: runtime crashes, state layout, PRNG split, Phase 4 weighted-Jacobian smoother, smoke tests, API cleanup) |
| Review Remediation | **PARTIAL/DONE IN CODE** (fast suite green; remaining items are deferred hygiene, not blockers) |
| Spike-field coupling cross-check | **DONE** (`coupling_*` modules; LFP-conditioned two-stage path validated) |
| Cross-region oscillator coupling | **BLOCKED AS SPECIFIED** (spike-only latent + loading inference is degenerate; requires LFP-conditioned rewrite) |
| Switching spike oscillator plan | **CODED, SCIENCE CAVEAT** (numerically tested, but spike-only joint latent/loading interpretation is not identifiable) |
| Remaining additive modules | Not started unless listed below (`cross_session_drift.py`, `adaptive_decoder.py`, `ca1_represented_state.py`, `value_gated_sequence.py`, `dynamic_spike_coupling.py`, etc. do not exist yet) |

## Current Priority Queue

The previous P1/P2/P2.5/P3/P6 queue has shipped. The current queue should start
from genuinely missing modules and from plans de-risked by the latest findings:

| Priority | Plan | Feasibility | Risk | Depends On | Notes |
|---|---|---|---|---|---|
| **N1** | dynamic-neuron-coupling.md | READY-ish | Medium | PointProcessModel + SGD (DONE) | Safe from bilinear latent degeneracy because regressors are observed lagged spikes; reuse `glm_laplace_update`. |
| **N2** | ca1-represented-state-switching.md | READY-ish | Medium | Position decoder + switching Kalman + point-process update (DONE) | Best first step toward replay/value scientific claims. |
| **N3** | gaussian-sum-filter-replay.md | PLANNED | Med-High | Position decoder (DONE) | Strong replay decoder extension; bigger implementation due to GSF smoother/mixture management. |
| **N4** | cross-session-drift.md | READY | Low-Med | PlaceFieldModel + Kalman (DONE) | Additive module for long-timescale remapping. |
| **N5** | sgd-v2-improvements.md | PLANNED | Low-Med | SGDFittableMixin (DONE) | Do `log_det_jacobian` first; benchmark-gate chunked `lax.scan`. |
| **N6** | adaptive-decoder.md | READY-ish | Medium | Position decoder + PlaceFieldModel (DONE) | Useful for long recordings; implement semi-supervised MVP only. |
| **N7** | joint-learning-drift.md | SPECULATIVE | High | Smith + PlaceField + switching_kalman (DONE) | Prototype first; high integration complexity. |

### Parallelism Opportunities

N1, N4, and N5 are mostly independent. N2 should precede S2/S3/S4 scientific
plans. N3 can proceed in parallel with N2 if replay decoding is the immediate
priority. Avoid running N2 and N3 in the same work session unless the task is
only shared test/data scaffolding.

### Suggested Execution Strategy

- **Phase A:** Pick one concrete missing module: dynamic spike-history coupling
  (N1) if the target is functional connectivity, or CA1 represented-state
  switching (N2) if the target is replay/value content.
- **Phase B:** If replay events are the bottleneck, implement Gaussian-sum replay
  decoding (N3) after or alongside N2 with strict K=1 parity tests.
- **Phase C:** Land cross-session drift (N4) before any covariate-driven drift or
  hierarchical multi-timescale work.
- **Phase D:** Add `log_det_jacobian` from SGD v2 (N5). Only implement chunked
  `lax.scan` SGD if benchmarks show meaningful speedup.
- **Phase E:** Keep joint-learning/drift and dual-latent CA1-mPFC plans
  prototype-first until their prerequisites exist as checked-in modules.

## Dependency Notes

- Behavioral foundations (`multinomial_choice.py`, `covariate_choice.py`,
  `contingency_belief.py`, `behavioral_uncertainty.py`, `switching_choice.py`)
  are complete.
- Replay/value work now starts with `ca1-represented-state-switching.md`, then
  `value-gated-sequence-expression.md`, then dual-latent or hierarchical plans.
- Long-timescale remapping should start with `cross-session-drift.md` before
  covariate-driven drift or hierarchical multi-timescale modeling.
- Dynamic spike-history coupling is safe to pursue because its regressors are
  observed lagged spikes. Spike-only latent oscillator coupling is blocked as a
  scientific estimator unless an observed field anchors the latent.
- Gaussian-sum replay decoding is additive to the existing position decoder and
  should keep a K=1 parity gate against `position_decoder.py`.

## Ordered Queue

### Foundation (Order 0)

| Order | Plan | Feasibility | Risk | Effort | Depends On |
|---|---|---|---|---|---|
| 0 | numerical-stability-remediation.md | **DONE** | Low | 1 week | None |
| 0.5 | computational-improvements.md | **DONE** | Low-Med | 2-3 weeks | None |
| 0.6 | sgd-fitting-all-models.md | **DONE** | — | — | 0.5 |

Order 0: Fixed correctness bugs in kalman.py, switching_kalman.py, and point_process_kalman.py.

Order 0.5: **DONE.** Parallel smoother (associative scan), Woodbury-optimized updates, Joseph form covariance, parameter constraints, SGD fitting for choice models.

Order 0.6: **DONE.** SGD fitting mixin for all model classes. SGDFittableMixin with STOCHASTIC_ROW transform, trainable flag, stop_gradient for frozen params, optax.masked. Covers MultinomialChoiceModel, CovariateChoiceModel, SmithLearningModel, PointProcessModel, PlaceFieldModel, CommonOscillatorModel, CorrelatedNoiseModel, DirectedInfluenceModel, all switching point-process models, and SwitchingSpikeOscillatorModel. Bonus: replaced eigendecomp-based PSD projection with Cholesky-based `_ensure_psd` in Laplace-EKF update to fix gradient NaN at high state dimensions.

### Infrastructure Track

| Order | Plan | Feasibility | Risk | Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 1 | position-decoding.md | READY | Low-Med | 1-2 weeks | None | **DONE** |
| 2 | cross-session-drift.md | READY | Low-Med | 1-2 weeks | None | Not started (deferred) |
| 3 | multinomial-choice-model.md | READY | Low-Med | 2-3 weeks | None | **DONE** |
| 3.5 | rl-state-space-covariates.md | READY | Low-Med | 1-2 weeks | Multinomial Choice | **DONE** |
| 3.6 | contingency-belief-latent-task-state.md | DONE | — | — | Multinomial Choice | **DONE** |
| 3.65 | uncertainty-aware-behavioral-modeling.md | DONE | — | — | RL Covariates + Contingency Belief | **DONE** |
| 3.7 | switching-choice-model.md | DONE | — | — | RL Covariates + Switching Kalman | **DONE** |
| 4 | joint-belief-state-decoder.md | PARTIAL | Medium | 2-3 weeks | Multinomial Choice | Not started (deferred) |
| 5 | adaptive-decoder.md | READY-ish | Medium | 2 weeks | Position Decoding | Not started |
| 6 | covariate-driven-drift.md | PARTIAL | Medium | 2-3 weeks | Cross-Session Drift | Not started (deferred) |
| 7 | spatial-value-model.md | PARTIAL | Med-High | 2-3 weeks | Multinomial Choice + Joint Belief | Not started (deferred) |
| 8 | joint-learning-drift.md | SPECULATIVE | High | 3+ weeks | Smith + PlaceField + Switching Kalman | **P4** |
| 9 | cross-region-coupling.md | BLOCKED AS SPECIFIED | High | rewrite needed | LFP-conditioned coupling stack | Spike-only version is degenerate; require per-region LFP/two-stage rewrite. |

### Oscillator Track

| Order | Plan | Feasibility | Risk | Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| O1 | regularized-oscillator-connectivity.md | **DONE** | — | — | SGD Fitting (**DONE**) | **P6** — complete |
| O2 | dynamic-neuron-coupling.md | READY-ish | Med | 2-3 weeks | SGD Fitting (**DONE**) | Not started — unblocked and de-risked by observed regressors |
| O3 | principled-stabilization-refactor.md | PARTIAL | Med | 2-3 weeks | None | Partly superseded by 2026-07 remediation; defer remaining in-jit guard work |

### Hamiltonian Track (standalone, nonlinear dynamics)

| Order | Plan | Feasibility | Risk | Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| H1 | hamiltonian-oscillator-state-space-model.md | **DONE** | — | — | SGD Fitting (**DONE**), point_process_kalman | Four modules (spikes/lfp/joint/switching) plus `nonlinear_dynamics.py`. Implementation diverged from plan's single-module skeleton. |
| H2 | hamiltonian-review-fixes.md | **DONE** | — | — | H1 | All 7 phases complete: runtime crashes, multi-osc state layout, PRNG split, weighted-Jacobian smoother, smoke tests for LFP/Joint, API cleanup, full-suite verification. |
| H3 | square-root-filter-investigation.md | **DEFERRED** | — | — | None | Un-defer only on a concrete trigger (real f32 failure, GPU memory pressure). See plan §"When to un-defer". |

The Hamiltonian family is standalone from `BaseModel` EM by design — see
[docs/hamiltonian_architecture.md](../hamiltonian_architecture.md) for the
rationale. Fitting is SGD-only (`.fit_sgd()`); `.fit()` raises `NotImplementedError`.

### Scientific Track (CA1/mPFC replay and value)

| Order | Plan | Feasibility | Risk | Effort | Depends On |
|---|---|---|---|---|---|
| S1 | ca1-represented-state-switching.md | PARTIAL | Medium | 2-3 weeks | Position Decoding (DONE) |
| S2 | value-gated-sequence-expression.md | PARTIAL | Med-High | 2-3 weeks | S1 + Multinomial Choice (DONE) |
| S3 | coupled-dual-latent-ca1-mpfc.md | PARTIAL | High | 3+ weeks | S1 + Joint Belief Decoder |
| S4 | hierarchical-multi-timescale.md | SPECULATIVE | High | 3+ weeks | S1 + Multinomial Choice + Cross-Session Drift |

### Minimum Publishable Path

The shortest route to a novel scientific claim (currently deferred in favor of priority queue above):

1. Position Decoding — **DONE**
2. Multinomial Choice — **DONE**
3. CA1 Represented-State Switching (S1) — deferred
4. Value-Gated Sequence Expression (S2) — the novel result

Claim: "CA1 alternates between local and nonlocal represented content, and latent value inferred from behavior biases nonlocal sequence expression."

## Why This Order

- **Order 0** fixes shared infrastructure bugs before anything builds on top.
- **Orders 1-3** (DONE) established the core decoding and behavioral models.
- **Order 3.5** (RL covariates) strengthens the behavioral model with mechanistic value updates, supporting all downstream value-related plans.
- **P1 (SGD Fitting)** is **DONE** — all model classes have `fit_sgd()`.
- **P6 (Regularized Oscillator)** is **DONE** — edge L1, area group L2, state-shared group L2 penalties on DIM coupling.
- **P2 (Contingency Belief)** is **DONE** — input-output HMM with centered softmax, Dirichlet prior, design-matrix transitions, EM + SGD.
- **P2.5 (Uncertainty-Aware Behavioral Modeling)** is **DONE** — uncertainty summaries and surprise are already first-class outputs.
- **P3 (Switching Choice)** is **DONE** — the behavioral switching layer is available for downstream plans.
- **P4 (Joint Learning+Drift)** is SPECULATIVE but scientifically important — links learning rate and representational drift through shared discrete states.
- **P5 (Adaptive Decoder)** enables long-duration decoding without retraining.
- **P6 (Regularized Oscillator)** is **DONE** — it is no longer blocked.
- **H1/H2 (Hamiltonian Track)** are **DONE**. Symplectic nonlinear latent dynamics (leapfrog + EKF linearization) with Gaussian and point-process observations, plus a switching variant. Standalone from `BaseModel` EM; SGD-only fitting. No downstream plan currently depends on this track, so it ran on its own schedule outside the priority queue.
- **S1-S2** form the minimum publishable path using infrastructure already built (currently deferred).
- **S3-S4** are ambitious cross-region and multi-timescale models for later.

## Shared Code Opportunities

The following plans share identical mathematical patterns and should use shared helpers:

- **Input-gain M-step** (B matrix from smoother statistics): Used by both RL Covariates (Order 3.5) and Covariate-Driven Drift (Order 6). Extract `m_step_input_gain(smoothed_increments, covariates)` into `kalman.py`.
- **Model comparison utilities** (BIC, held-out LL, cross-validation): Used by Multinomial Choice, RL Covariates, Joint Belief Decoder, and Value-Gated Sequences. Consider a shared `model_comparison.py` utility module.

## Plan Entry/Exit Gates

### Order 0: Numerical Stability

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_kalman.py src/state_space_practice/tests/test_switching_kalman.py src/state_space_practice/tests/test_point_process_kalman.py -v
```

Exit gate:

```bash
# Same tests must still pass, plus new stability regression tests
uv run pytest src/state_space_practice/tests/ -v
uv run ruff check src/state_space_practice
```

Stop condition:

- Any existing test regresses after stabilization changes

### Order 1: Position Decoding — DONE

35 tests passing. Exit gate satisfied.

### Order 2: Cross-Session Drift (deferred)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_kalman.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_cross_session_drift.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- session summary extraction API cannot provide stable means/covariances

### Order 3: Multinomial Choice — DONE

43 tests passing. Exit gate satisfied. K=2 Smith consistency verified.

### Order 3.5: RL State-Space Covariates — DONE

55 tests passing. Exit gate satisfied.

### P1 / Order 0.6: SGD Fitting All Models

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/ -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/ -v
uv run ruff check src/state_space_practice
```

Stop conditions:

- SGD fit diverges on any model where EM converges
- Gradient stability gate fails for point-process or oscillator models

### P2 / Order 3.6: Contingency Belief

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_contingency_belief.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- posterior state occupancy does not distinguish synthetic block-structured contingencies

### P3 / Order 3.7: Switching Choice Model

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_switching_choice.py -v
uv run ruff check src/state_space_practice
```

Stop conditions:

- S=1 parity with CovariateChoiceModel fails
- Synthetic two-state recovery does not distinguish exploit vs. explore

### P4 / Order 8: Joint Learning + Drift (Prototype First)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_smith_learning_algorithm.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_state_dependent_learning.py src/state_space_practice/tests/test_joint_discrete_state.py src/state_space_practice/tests/test_state_dependent_drift.py src/state_space_practice/tests/test_joint_learning_drift_model.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- single-state equivalence to baseline Smith model fails

### P5 / Order 5: Adaptive Decoder

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_position_decoder.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_adaptive_decoder.py src/state_space_practice/tests/test_position_decoder.py src/state_space_practice/tests/test_place_field_model.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- position and weight updates cannot remain stable in alternating loop

### P6: Regularized Oscillator Connectivity

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v
# AND: P1 oscillator SGD tasks 5-6 must be complete
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_oscillator_regularization.py src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- penalty causes synthetic recovery to collapse relative to unpenalized SGD baseline

### Order 4: Joint Belief-State Decoder (deferred)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- joint model does not improve held-out choice log-likelihood over behavior-only baseline

### Order 6: Covariate-Driven Drift (deferred)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_cross_session_drift.py src/state_space_practice/tests/test_place_field_model.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_covariate_drift.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- covariate alignment to time bins is ambiguous or inconsistent

### Order 7: Spatial-Value Model (deferred)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_spatial_value_model.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- value-only and spatial-only baseline parity checks fail

### Order 9: Cross-Region Coupling (deferred)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_switching_point_process.py src/state_space_practice/tests/test_oscillator_utils.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_multi_region_model.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- finite-difference Jacobian checks for multi-region spike observation fail

### Hamiltonian Track Gates

#### H1: Hamiltonian Oscillator State-Space Model — DONE

Exit gate (passes as of 2026-04-16):

```bash
uv run pytest src/state_space_practice/tests/test_hamiltonian_spikes.py src/state_space_practice/tests/test_hamiltonian_lfp.py src/state_space_practice/tests/test_hamiltonian_joint.py -v
```

34 tests pass, including `TestHamiltonianVsLinearBaseline::test_nonlinear_beats_linear_on_duffing_oscillator` which closes V1 plan success criterion 5.

#### H2: Hamiltonian Review Fixes — DONE

Exit gate (passes as of 2026-04-16):

```bash
uv run pytest src/state_space_practice/tests/test_hamiltonian_switching.py -v
```

All Phase 1–6 items verified: runtime-crash fixes, multi-oscillator state layout, PRNG split, probability-weighted Jacobian in switching smoother (verified by `test_smooth_sensitive_to_transition_asymmetry`), smoke tests for LFP/Joint, API cleanup.

### Scientific Track Gates

#### S1: CA1 Represented-State Switching (deferred)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_position_decoder.py src/state_space_practice/tests/test_switching_kalman.py src/state_space_practice/tests/test_point_process_kalman.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_ca1_represented_state.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- discrete state posterior does not distinguish local vs. nonlocal content on synthetic data with known replay events

#### S2: Value-Gated Sequence Expression (deferred)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_value_gated_sequence.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- value modulation of nonlocal content is not detectable on synthetic data with known value-destination coupling

#### S3: Coupled Dual-Latent CA1-mPFC (deferred)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_joint_belief_decoder.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_dual_latent_coupling.py -v
uv run ruff check src/state_space_practice
```

#### S4: Hierarchical Multi-Timescale (deferred)

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_hierarchical_timescale.py -v
uv run ruff check src/state_space_practice
```

## Completion Definition

A plan is complete only when all are true:

1. Entry gate passed before implementation.
2. Targeted tests pass.
3. Neighbor regression tests pass.
4. Ruff passes.
5. MVP smoke check is stable and finite.
6. Deferred items remain deferred (no scope creep).
