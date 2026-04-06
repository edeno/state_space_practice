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
| All others | Not started |

## Dependency Graph

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
       │                   │          ┌───────┴────────┐
       │                   │          │                │
       │                   │          ▼                ▼
       │                   │   ┌────────────┐  ┌─────────────┐
       ▼                   │   │ 3.5 RL     │  │ 4. Joint    │
  ┌──────────┐             │   │ Covariates │  │ Belief      │
  │ 5. Adapt.│             │   └─────┬──────┘  │ Decoder     │
  │ Decoder  │             │         │         └──────┬──────┘
  └──────────┘             │         │                │
       │                   ▼         │                │
       │            ┌────────────┐   │                │
       │            │ 6. Covar.  │   │                │
       │            │ Drift      │   │                │
       │            └────────────┘   │                │
       │                   │         │                │
       ▼                   ▼         ▼                ▼
  ┌─────────────────────────────────────────────────────┐
  │              Integration Plans (7-9)                 │
  │  7. Spatial-Value  8. Joint Learn+Drift  9. X-Region│
  └─────────────────────────────────────────────────────┘

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

Fixed correctness bugs in kalman.py, switching_kalman.py, and point_process_kalman.py.

### Infrastructure Track

| Order | Plan | Feasibility | Risk | Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 1 | position-decoding.md | READY | Low-Med | 1-2 weeks | None | **DONE** |
| 2 | cross-session-drift.md | READY | Low-Med | 1-2 weeks | None | Not started |
| 3 | multinomial-choice-model.md | READY | Low-Med | 2-3 weeks | None | **DONE** |
| 3.5 | rl-state-space-covariates.md | READY | Low-Med | 1-2 weeks | Multinomial Choice | Not started |
| 4 | joint-belief-state-decoder.md | PARTIAL | Medium | 2-3 weeks | Multinomial Choice | Not started |
| 5 | adaptive-decoder.md | PARTIAL | Medium | 2 weeks | Position Decoding | Not started |
| 6 | covariate-driven-drift.md | PARTIAL | Medium | 2-3 weeks | Cross-Session Drift | Not started |
| 7 | spatial-value-model.md | PARTIAL | Med-High | 2-3 weeks | Multinomial Choice + Joint Belief | Not started |
| 8 | joint-learning-drift.md | SPECULATIVE | High | 3+ weeks | Adaptive + Covariate + Multinomial | Not started |
| 9 | cross-region-coupling.md | SPECULATIVE | High | 3+ weeks | Switching point-process + oscillator | Not started |

### Scientific Track (CA1/mPFC replay and value)

| Order | Plan | Feasibility | Risk | Effort | Depends On |
|---|---|---|---|---|---|
| S1 | ca1-represented-state-switching.md | PARTIAL | Medium | 2-3 weeks | Position Decoding (DONE) |
| S2 | value-gated-sequence-expression.md | PARTIAL | Med-High | 2-3 weeks | S1 + Multinomial Choice (DONE) |
| S3 | coupled-dual-latent-ca1-mpfc.md | PARTIAL | High | 3+ weeks | S1 + Joint Belief Decoder |
| S4 | hierarchical-multi-timescale.md | SPECULATIVE | High | 3+ weeks | S1 + Multinomial Choice + Cross-Session Drift |

### Minimum Publishable Path

The shortest route to a novel scientific claim:

1. Position Decoding — **DONE**
2. Multinomial Choice — **DONE**
3. CA1 Represented-State Switching (S1) — next scientific target
4. Value-Gated Sequence Expression (S2) — the novel result

Claim: "CA1 alternates between local and nonlocal represented content, and latent value inferred from behavior biases nonlocal sequence expression."

## Why This Order

- **Order 0** fixes shared infrastructure bugs before anything builds on top.
- **Orders 1-3** (DONE) established the core decoding and behavioral models.
- **Order 3.5** (RL covariates) strengthens the behavioral model with mechanistic value updates, supporting all downstream value-related plans.
- **Orders 4-7** are dependency-driven integrations.
- **Orders 8-9** are speculative integration efforts — only after lower-risk pieces are validated.
- **S1-S2** form the minimum publishable path using infrastructure already built.
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

### Order 2: Cross-Session Drift

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

### Order 3.5: RL State-Space Covariates

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop conditions:

- No-covariate mode does not reproduce MultinomialChoiceModel exactly
- Rescorla-Wagner equivalence test fails (K=2, reward covariate, Q=0)

### Order 4: Joint Belief-State Decoder

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

### Order 5: Adaptive Decoder

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_adaptive_decoder.py src/state_space_practice/tests/test_position_decoder.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- position and weight updates cannot remain stable in alternating loop

### Order 6: Covariate-Driven Drift

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

### Order 7: Spatial-Value Model

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

### Order 8: Joint Learning + Drift (Prototype First)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_adaptive_decoder.py src/state_space_practice/tests/test_covariate_drift.py src/state_space_practice/tests/test_multinomial_choice.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_learning_drift_model.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- single-state equivalence to baseline Smith model fails

### Order 9: Cross-Region Coupling (Two-Region MVP First)

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

#### S1: CA1 Represented-State Switching

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

#### S2: Value-Gated Sequence Expression

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

#### S3: Coupled Dual-Latent CA1-mPFC

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_joint_belief_decoder.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_dual_latent_coupling.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

#### S4: Hierarchical Multi-Timescale

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_hierarchical_timescale.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

## Suggested Weekly Milestones

- Week 1: Order 0 (numerical stability)
- Week 2-3: Order 3.5 (RL covariates) — immediate next step
- Week 3-4: S1 (CA1 represented-state switching) — min publishable path
- Week 5-6: S2 (value-gated sequences) — the novel scientific result
- Week 7+: Orders 2, 4-7 as needed; S3-S4 as stretch goals

## Completion Definition

A plan is complete only when all are true:

1. Entry gate passed before implementation.
2. Targeted tests pass.
3. Neighbor regression tests pass.
4. Ruff passes.
5. MVP smoke check is stable and finite.
6. Deferred items remain deferred (no scope creep).
