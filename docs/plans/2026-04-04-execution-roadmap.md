# Claude Code Execution Roadmap

> For Claude: Execute plans in this exact order within each track. Do not start a downstream plan until all entry and exit gates for upstream dependencies pass.

## Purpose

This roadmap provides the authoritative implementation queue for all plans in docs/plans/, organized into two tracks:

- **Infrastructure track:** Core models and utilities that form the computational foundation
- **Scientific track:** Neuroscience-specific models that build on the infrastructure to test scientific hypotheses about CA1/mPFC replay, theta sequences, and value-modulated neural coding

Both tracks share the same dependency graph. The scientific track has a "minimum publishable path" — the shortest route to a novel scientific claim.

## Global Rules

1. Run all commands in the conda environment state_space_practice.
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
| All others | Not started |

## Current Priority Queue

The following six plans are the current implementation priorities, ordered by interest and respecting dependencies:

| Priority | Plan | Feasibility | Risk | Depends On | Notes |
|---|---|---|---|---|---|
| **P1** | sgd-fitting-all-models.md | READY | Med | Computational Improvements (DONE) | Unblocks P6; highest priority |
| **P2** | contingency-belief-latent-task-state.md | PARTIAL | Med | Multinomial Choice (DONE) | Can start in parallel with P1 |
| **P3** | switching-choice-model.md | READY | Med | RL Covariates (DONE) + switching_kalman | Can start in parallel with P1-P2 |
| **P4** | joint-learning-drift.md | SPECULATIVE | High | Smith + PlaceField + switching_kalman (all exist) | Prototype first; high integration complexity |
| **P5** | adaptive-decoder.md | PARTIAL | Med | Position Decoding (DONE) | Can start in parallel with P1-P3 |
| **P6** | regularized-oscillator-connectivity.md | HIGH | Low | **SGD Fitting tasks 5-6** (P1) | Blocked until P1 oscillator SGD is done |

### Parallelism Opportunities

P1 through P5 have no mutual dependencies and can proceed in parallel:

```
                    ┌─────────────────────────┐
                    │  Completed Foundation    │
                    │  (Stability, Comp Impr,  │
                    │   Pos Dec, Choice, RL)   │
                    └────────┬────────────────┘
                             │
     ┌───────────┬───────────┼───────────┬────────────┐
     │           │           │           │            │
     ▼           ▼           ▼           ▼            ▼
  ┌──────┐  ┌──────┐   ┌──────┐   ┌──────┐     ┌──────┐
  │ P1   │  │ P2   │   │ P3   │   │ P4   │     │ P5   │
  │ SGD  │  │Contin│   │Switch│   │Joint │     │Adapt │
  │Fittin│  │gency │   │Choice│   │Learn │     │Decode│
  │  g   │  │Belief│   │Model │   │+Drift│     │  r   │
  └──┬───┘  └──────┘   └──────┘   └──────┘     └──────┘
     │
     │ (after oscillator SGD tasks 5-6)
     ▼
  ┌──────┐
  │ P6   │
  │Regul.│
  │Oscil.│
  └──────┘
```

### Suggested Execution Strategy

- **Phase A (parallel):** Start P1 (SGD Fitting) and P2 (Contingency Belief) simultaneously. P1 is the largest infrastructure investment and unblocks P6. P2 is a self-contained new module.
- **Phase B (after P1 tasks 0-1 or P2):** Start P3 (Switching Choice) — benefits from seeing the SGD mixin pattern established.
- **Phase C (after Phase A/B):** Start P5 (Adaptive Decoder) — independent but lower priority.
- **Phase D (after P1 tasks 5-6):** Start P6 (Regularized Oscillator) — blocked until oscillator SGD exists.
- **Phase E (after Phase A/B validation):** Start P4 (Joint Learning+Drift) — SPECULATIVE, prototype-first. Benefits from infrastructure maturity.

## Dependency Graph (Full)

```
                    ┌─────────────────────────┐
                    │  0. Numerical Stability  │
                    │     (infrastructure)     │
                    └────────┬────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                     │
        ▼                    ▼                     ▼
  ┌──────────┐      ┌──────────────┐      ┌──────────────┐
  │ 1. Pos   │      │ 2. Cross-    │      │ 3. Multinomial│
  │ Decoding │      │ Session Drift│      │ Choice       │
  │ ✅ DONE  │      │              │      │ ✅ DONE      │
  └────┬─────┘      └──────┬───────┘      └──┬───────────┘
       │                   │                  │
       │                   │          ┌───────┼────────────┐
       │                   │          │       │            │
       ▼                   │          ▼       ▼            ▼
  ┌──────────┐             │   ┌──────────┐ ┌──────┐ ┌──────────┐
  │P5 Adapt. │             │   │ 3.5 RL   │ │P2    │ │ 4. Joint │
  │ Decoder  │             │   │ Covariates│ │Contin│ │ Belief   │
  └──────────┘             │   │ ✅ DONE  │ │gency │ │ Decoder  │
       │                   │   └─────┬────┘ └──────┘ └──────┬───┘
       │                   │         │                       │
       │                   │    ┌────┴────┐                  │
       │                   │    │         │                  │
       │                   │    ▼         ▼                  │
       │                   │  ┌──────┐ ┌──────┐              │
       │                   │  │P3    │ │P4    │              │
       │                   │  │Switch│ │Joint │              │
       │                   │  │Choice│ │Learn │              │
       │                   │  └──────┘ │+Drift│              │
       │                   │           └──────┘              │
       │                   ▼                                 │
       │            ┌────────────┐                           │
       │            │ 6. Covar.  │                           │
       │            │ Drift      │                           │
       │            └────────────┘                           │
       │                   │                                 │
       ▼                   ▼              ▼                  ▼
  ┌─────────────────────────────────────────────────────────┐
  │              Integration Plans (7-9)                     │
  │  7. Spatial-Value  8. Joint Learn+Drift  9. X-Region    │
  └─────────────────────────────────────────────────────────┘

  Infrastructure / Oscillator Track:

  0.5 Comp. Improvements ──► P1 SGD Fitting ──► P6 Regularized Oscillator
       ✅ DONE

  Scientific Track (builds on infrastructure):

  1. Pos Decoding ──► CA1 Represented-State Switching
       ✅ DONE              │
                             ▼
  3. Multinomial ──► Value-Gated Sequence Expression
     Choice                  │
     ✅ DONE                 ▼
                     Coupled Dual-Latent CA1-mPFC
                             │
                             ▼
                     Hierarchical Multi-Timescale
```

## Ordered Queue

### Foundation (Order 0)

| Order | Plan | Feasibility | Risk | Effort | Depends On |
|---|---|---|---|---|---|
| 0 | numerical-stability-remediation.md | **DONE** | Low | 1 week | None |
| 0.5 | computational-improvements.md | **DONE** | Low-Med | 2-3 weeks | None |
| 0.6 | sgd-fitting-all-models.md | READY | Med | 3-4 weeks | 0.5 |

Order 0: Fixed correctness bugs in kalman.py, switching_kalman.py, and point_process_kalman.py.

Order 0.5: **DONE.** Parallel smoother (associative scan), Woodbury-optimized updates, Joseph form covariance, parameter constraints, SGD fitting for choice models.

Order 0.6: **P1 — current top priority.** SGD fitting mixin for all model classes. Adds STOCHASTIC_ROW transform, trainable flag with stop_gradient, shared SGDFittableMixin. Covers SmithLearningModel, PointProcessModel, PlaceFieldModel, all oscillator models, and all switching point-process models.

### Infrastructure Track

| Order | Plan | Feasibility | Risk | Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 1 | position-decoding.md | READY | Low-Med | 1-2 weeks | None | **DONE** |
| 2 | cross-session-drift.md | READY | Low-Med | 1-2 weeks | None | Not started (deferred) |
| 3 | multinomial-choice-model.md | READY | Low-Med | 2-3 weeks | None | **DONE** |
| 3.5 | rl-state-space-covariates.md | READY | Low-Med | 1-2 weeks | Multinomial Choice | **DONE** |
| 3.6 | contingency-belief-latent-task-state.md | PARTIAL | Medium | 2-3 weeks | Multinomial Choice | **P2** |
| 3.7 | switching-choice-model.md | READY | Medium | 1-2 weeks | RL Covariates + Switching Kalman | **P3** |
| 4 | joint-belief-state-decoder.md | PARTIAL | Medium | 2-3 weeks | Multinomial Choice | Not started (deferred) |
| 5 | adaptive-decoder.md | PARTIAL | Medium | 2 weeks | Position Decoding | **P5** |
| 6 | covariate-driven-drift.md | PARTIAL | Medium | 2-3 weeks | Cross-Session Drift | Not started (deferred) |
| 7 | spatial-value-model.md | PARTIAL | Med-High | 2-3 weeks | Multinomial Choice + Joint Belief | Not started (deferred) |
| 8 | joint-learning-drift.md | SPECULATIVE | High | 3+ weeks | Smith + PlaceField + Switching Kalman | **P4** |
| 9 | cross-region-coupling.md | SPECULATIVE | High | 3+ weeks | Switching point-process + oscillator | Not started (deferred) |

### Oscillator Track

| Order | Plan | Feasibility | Risk | Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| O1 | regularized-oscillator-connectivity.md | HIGH | Low | 1-2 weeks | SGD Fitting (oscillator tasks 5-6) | **P6** |
| O2 | dynamic-neuron-coupling.md | PARTIAL | Med | 2-3 weeks | SGD Fitting (point-process task 3) | Not started (deferred) |
| O3 | principled-stabilization-refactor.md | PARTIAL | Med | 2-3 weeks | None | Not started (deferred) |

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
- **P1 (SGD Fitting)** is the top current priority because it provides a unified optimization path for all model classes and unblocks P6 (regularized oscillator connectivity).
- **P2 (Contingency Belief)** adds an explicit hidden-world-state model, complementing both continuous value inference and planned strategy-state switching.
- **P3 (Switching Choice)** is the first switching behavioral model with per-state value dynamics.
- **P4 (Joint Learning+Drift)** is SPECULATIVE but scientifically important — links learning rate and representational drift through shared discrete states.
- **P5 (Adaptive Decoder)** enables long-duration decoding without retraining.
- **P6 (Regularized Oscillator)** adds structured sparsity penalties to oscillator coupling — blocked until P1 oscillator tasks complete.
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
conda run -n state_space_practice pytest src/state_space_practice/tests/test_kalman.py src/state_space_practice/tests/test_switching_kalman.py src/state_space_practice/tests/test_point_process_kalman.py -v
```

Exit gate:

```bash
# Same tests must still pass, plus new stability regression tests
conda run -n state_space_practice pytest src/state_space_practice/tests/ -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- Any existing test regresses after stabilization changes

### Order 1: Position Decoding — DONE

35 tests passing. Exit gate satisfied.

### Order 2: Cross-Session Drift (deferred)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py -v
conda run -n state_space_practice ruff check src/state_space_practice
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
conda run -n state_space_practice pytest src/state_space_practice/tests/ -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/ -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop conditions:

- SGD fit diverges on any model where EM converges
- Gradient stability gate fails for point-process or oscillator models

### P2 / Order 3.6: Contingency Belief

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- posterior state occupancy does not distinguish synthetic block-structured contingencies

### P3 / Order 3.7: Switching Choice Model

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_choice.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop conditions:

- S=1 parity with CovariateChoiceModel fails
- Synthetic two-state recovery does not distinguish exploit vs. explore

### P4 / Order 8: Joint Learning + Drift (Prototype First)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_smith_learning_algorithm.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_state_dependent_learning.py src/state_space_practice/tests/test_joint_discrete_state.py src/state_space_practice/tests/test_state_dependent_drift.py src/state_space_practice/tests/test_joint_learning_drift_model.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- single-state equivalence to baseline Smith model fails

### P5 / Order 5: Adaptive Decoder

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_adaptive_decoder.py src/state_space_practice/tests/test_position_decoder.py src/state_space_practice/tests/test_place_field_model.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- position and weight updates cannot remain stable in alternating loop

### P6: Regularized Oscillator Connectivity

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v
# AND: P1 oscillator SGD tasks 5-6 must be complete
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_regularization.py src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- penalty causes synthetic recovery to collapse relative to unpenalized SGD baseline

### Order 4: Joint Belief-State Decoder (deferred)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- joint model does not improve held-out choice log-likelihood over behavior-only baseline

### Order 6: Covariate-Driven Drift (deferred)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py src/state_space_practice/tests/test_place_field_model.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_drift.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- covariate alignment to time bins is ambiguous or inconsistent

### Order 7: Spatial-Value Model (deferred)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_spatial_value_model.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- value-only and spatial-only baseline parity checks fail

### Order 9: Cross-Region Coupling (deferred)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_point_process.py src/state_space_practice/tests/test_oscillator_utils.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multi_region_model.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- finite-difference Jacobian checks for multi-region spike observation fail

### Scientific Track Gates

#### S1: CA1 Represented-State Switching (deferred)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py src/state_space_practice/tests/test_switching_kalman.py src/state_space_practice/tests/test_point_process_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- discrete state posterior does not distinguish local vs. nonlocal content on synthetic data with known replay events

#### S2: Value-Gated Sequence Expression (deferred)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_value_gated_sequence.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- value modulation of nonlocal content is not detectable on synthetic data with known value-destination coupling

#### S3: Coupled Dual-Latent CA1-mPFC (deferred)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_joint_belief_decoder.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_dual_latent_coupling.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

#### S4: Hierarchical Multi-Timescale (deferred)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_hierarchical_timescale.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

## Completion Definition

A plan is complete only when all are true:

1. Entry gate passed before implementation.
2. Targeted tests pass.
3. Neighbor regression tests pass.
4. Ruff passes.
5. MVP smoke check is stable and finite.
6. Deferred items remain deferred (no scope creep).
