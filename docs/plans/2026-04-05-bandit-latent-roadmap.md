# Spatial Bandit Latent Modeling Roadmap

> For Claude: Use this roadmap for CA1-mPFC spatial bandit work. It supplements the generic execution roadmap and is the preferred queue when the scientific target is replay, theta sequences, value modulation, or cross-region coordination.

## Purpose

This roadmap reorganizes the next modeling steps around what the repository already supports:

- continuous latent filtering and smoothing in `kalman.py`
- switching discrete-state inference in `switching_kalman.py`
- point-process observation updates in `point_process_kalman.py`
- CA1 encoding and decoding foundations in `place_field_model.py` and `position_decoder.py`
- bandit belief or value inference in `multinomial_choice.py` and `covariate_choice.py`

## Codebase Reality Check

Already implemented and reviewed:

- `src/state_space_practice/position_decoder.py`: fixed-place-field decoding of physical position from spikes
- `src/state_space_practice/multinomial_choice.py`: latent option-value filtering and smoothing from behavior
- `src/state_space_practice/covariate_choice.py`: covariate-driven value dynamics (reward → value updates)
- `src/state_space_practice/place_field_model.py`: drifting spatial encoding weights on spline bases
- `src/state_space_practice/switching_kalman.py`: generic discrete-continuous switching inference
- `src/state_space_practice/switching_point_process.py`: switching point-process filtering for spike observations
- `src/state_space_practice/oscillator_utils.py` and `src/state_space_practice/oscillator_models.py`: block-structured switching oscillator infrastructure

Not yet implemented:

- SGD fitting mixin for all model classes (P1)
- contingency-belief or hidden-world-state model (P2)
- switching behavioral strategy model (P3)
- joint learning + representational drift model (P4)
- adaptive decoder with drifting tuning curves (P5)
- regularized oscillator connectivity penalties (P6)
- represented-content or replay state separate from physical position (S1)
- value-dependent switching into local versus nonlocal sequence modes (S2)
- heterogeneous CA1 and mPFC latent blocks in one coupled model (S3)
- explicit fast or slow hierarchy tying sequence content, trial belief, and session drift (S4)

## Current Priority Queue

The current priorities focus on infrastructure and behavioral modeling:

| Priority | Plan | Track | Status |
|---|---|---|---|
| P1 | `sgd-fitting-all-models.md` | Infrastructure | **DONE** — all model classes have fit_sgd() |
| P2 | `contingency-belief-latent-task-state.md` | Behavioral | Not started — next priority |
| P3 | `switching-choice-model.md` | Behavioral | Not started |
| P4 | `joint-learning-drift.md` | Integration | Not started |
| P5 | `adaptive-decoder.md` | Neural | Not started |
| P6 | `regularized-oscillator-connectivity.md` | Oscillator | Not started — unblocked by P1 |

### Behavioral Modeling Sub-Roadmap

The three behavioral plans (P2, P3, P4) form a conceptual progression:

```
MultinomialChoiceModel (DONE) ──► CovariateChoiceModel (DONE)
        │                                    │
        ▼                                    ▼
  P2: Contingency Belief              P3: Switching Choice
  "What does the animal think         "How is the animal
   the world rules are?"               learning right now?"
        │                                    │
        └────────────┬───────────────────────┘
                     ▼
            P4: Joint Learning + Drift
            "Do learning rate and place field
             drift share the same brain state?"
```

P2 and P3 are independent and answer different questions. P4 is the ambitious integration.

## Deferred Scientific Track

The minimum publishable path (CA1 replay + value gating) is deferred in favor of the current priorities but remains the shortest route to a novel scientific claim:

| Order | Plan | Status | Depends On |
|---|---|---|---|
| S1 | `ca1-represented-state-switching.md` | Deferred | Position Decoding (DONE) |
| S2 | `value-gated-sequence-expression.md` | Deferred | S1 + Multinomial Choice (DONE) |
| S3 | `coupled-dual-latent-ca1-mpfc.md` | Deferred | S1 + Joint Belief Decoder |
| S4 | `hierarchical-multi-timescale.md` | Deferred | S1 + Multinomial Choice + Cross-Session Drift |

Claim when S1-S2 are complete: "CA1 alternates between local and nonlocal represented content, and latent value inferred from behavior biases nonlocal sequence expression."

## Deferred Infrastructure

| Plan | Depends On | Notes |
|---|---|---|
| `cross-session-drift.md` | None | Useful for S4 and long-timescale remapping |
| `joint-belief-state-decoder.md` | Multinomial Choice (DONE) | Neural + behavioral value fusion |
| `covariate-driven-drift.md` | Cross-Session Drift | Reward/context-driven remapping |
| `spatial-value-model.md` | Multinomial Choice + Joint Belief | Mixed-selectivity decomposition |
| `cross-region-coupling.md` | Switching PP + Oscillator | Multi-region block dynamics |
| `dynamic-neuron-coupling.md` | SGD Fitting (PP task 3) | Time-varying spike-history GLM |
| `principled-stabilization-refactor.md` | None | Replace ad hoc numerical guards |

## Entry and Exit Gates

### P1: SGD Fitting All Models

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/ -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/ -v
conda run -n state_space_practice ruff check src/state_space_practice
```

### P2: Contingency Belief

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

### P3: Switching Choice Model

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_choice.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

### P4: Joint Learning + Drift

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_smith_learning_algorithm.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_state_dependent_learning.py src/state_space_practice/tests/test_joint_discrete_state.py src/state_space_practice/tests/test_state_dependent_drift.py src/state_space_practice/tests/test_joint_learning_drift_model.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

### P5: Adaptive Decoder

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_adaptive_decoder.py src/state_space_practice/tests/test_position_decoder.py src/state_space_practice/tests/test_place_field_model.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

### P6: Regularized Oscillator Connectivity (unblocked)

Entry gate:

```bash
# P1 oscillator SGD is DONE
conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v -k "sgd or SGD"
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_regularization.py src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

### S1: CA1 Represented-State Switching (deferred)

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

- local-mode decoding cannot recover the standard position decoder on synthetic data

### S2: Value-Gated Sequence Expression (deferred)

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

- value features do not change local versus nonlocal occupancy or nonlocal represented destination on held-out data

## Why This Queue

- **P1 (SGD Fitting)** is **DONE** — all model classes have `fit_sgd()`, P6 is unblocked.
- **P2 (Contingency Belief)** models what the animal believes about the world, complementing P3's strategy model. Later serves as mPFC belief latent for S3.
- **P3 (Switching Choice)** is the behavioral strategy-switching model — directly interpretable and the first model that infers exploit/explore without the experimenter's contingency table.
- **P4 (Joint Learning+Drift)** is the ambitious integration linking behavioral learning and neural remapping. SPECULATIVE but scientifically important — implements prototype-first.
- **P5 (Adaptive Decoder)** enables multi-hour decoding, needed for any long-session neural analysis.
- **P6 (Regularized Oscillator)** is unblocked — DIM SGD over `coupling_strength`/`phase_difference` is ready for penalty terms.
- The scientific track (S1-S4) is deferred but all dependencies are already met for S1. It can be reactivated at any time.
