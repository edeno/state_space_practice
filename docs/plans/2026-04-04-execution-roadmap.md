# Claude Code Execution Roadmap for Plan Set

> For Claude: Execute plans in this exact order. Do not start a downstream plan until all entry and exit gates for upstream dependencies pass.

## Purpose

This roadmap provides one strict implementation queue for all ten plans in docs/plans, with:

- dependency order
- feasibility/risk level
- estimated implementation effort
- objective entry gates and exit gates
- stop conditions for unsafe progression

## Global Rules

1. Run all commands in the conda environment state_space_practice.
2. Use test-first sequencing inside each plan task.
3. Never run two high-risk plans in parallel.
4. If any gate fails, stop and fix before continuing.
5. For speculative plans, complete prototype scope before full model integration.

## Ordered Queue

| Order | Plan | Feasibility | Risk | Effort | Depends On |
|---|---|---|---|---|---|
| 1 | 2026-04-03-position-decoding.md + 2026-04-03-position-decoding-tasks.md | READY | Low-Med | 1-2 weeks | None |
| 2 | 2026-04-03-cross-session-drift.md + 2026-04-03-cross-session-drift-tasks.md | READY | Low-Med | 1-2 weeks | None |
| 3 | 2026-04-04-multinomial-choice-model.md + 2026-04-04-multinomial-choice-tasks.md | READY | Low-Med | 2-3 weeks | None |
| 4 | 2026-04-04-joint-belief-state-decoder.md + 2026-04-04-joint-belief-state-decoder-tasks.md | PARTIAL | Medium | 2-3 weeks | Multinomial Choice complete |
| 5 | 2026-04-03-adaptive-decoder.md | PARTIAL | Medium | 2 weeks | Position Decoding |
| 6 | 2026-04-04-covariate-driven-drift.md | PARTIAL | Medium | 2-3 weeks | Cross-Session Drift baseline APIs |
| 7 | 2026-04-04-spatial-value-model.md | PARTIAL | Medium-High | 2-3 weeks | Multinomial Choice + Joint Belief Decoder complete |
| 8 | 2026-04-03-joint-learning-drift.md | SPECULATIVE | High | 3+ weeks | Adaptive + Covariate + Multinomial pieces |
| 9 | 2026-04-03-cross-region-coupling.md | SPECULATIVE | High | 3+ weeks | Switching point-process stability + oscillator block abstractions |

## Why This Order

- Orders 1-3 maximize fast wins using existing infrastructure and reduce unknowns.
- Orders 4-7 are dependency-driven integrations that become tractable after the core models exist.
- Orders 8-9 are research-grade integration efforts and should only start once lower-risk blocks are validated.

## Plan Entry/Exit Gates

### Order 1: Position Decoding

Entry gate:

- point-process and place-field regression baseline passes

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_point_process_kalman.py src/state_space_practice/tests/test_place_field_model.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- synthetic decode smoke run has unstable or non-finite trajectories

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

### Order 3: Multinomial Choice (Design + Tasks)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_smith_learning_algorithm.py src/state_space_practice/tests/test_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- K=2 consistency check against Smith-style behavior fails

### Order 4: Joint Belief-State Decoder (Design + Tasks)

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

Exit gate (prototype):

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_state_dependent_learning.py src/state_space_practice/tests/test_joint_discrete_state.py src/state_space_practice/tests/test_state_dependent_drift.py src/state_space_practice/tests/test_joint_learning_drift_model.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- single-state equivalence to baseline Smith model fails

### Order 9: Cross-Region Coupling (Two-Region MVP First)

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_point_process.py src/state_space_practice/tests/test_oscillator_utils.py -v
```

Exit gate (two-region MVP):

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multi_region_oscillator.py src/state_space_practice/tests/test_multi_region_spike_obs.py src/state_space_practice/tests/test_multi_region_model.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- finite-difference Jacobian checks for multi-region spike observation fail

## Suggested Branching Strategy

Use one branch per order block:

- roadmap-order-1-position-decoding
- roadmap-order-2-cross-session-drift
- roadmap-order-3-multinomial-choice
- roadmap-order-4-joint-belief-decoder
- roadmap-order-5-adaptive-decoder
- roadmap-order-6-covariate-drift
- roadmap-order-7-spatial-value
- roadmap-order-8-joint-learning-drift
- roadmap-order-9-cross-region-coupling

Do not stack more than one unfinished speculative branch.

## Suggested Weekly Milestones

- Week 1-2: Orders 1-2
- Week 3-5: Orders 3-4
- Week 6-10: Orders 5-7
- Week 11+: Orders 8-9 prototypes

## Completion Definition

A plan is complete only when all are true:

1. Entry gate passed before implementation.
2. Targeted tests pass.
3. Neighbor regression tests pass.
4. Ruff passes.
5. MVP smoke check is stable and finite.
6. Deferred items remain deferred (no scope creep).
