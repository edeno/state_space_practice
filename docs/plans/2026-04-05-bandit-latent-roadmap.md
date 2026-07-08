# Spatial Bandit Latent Modeling Roadmap

> For Claude: Use this roadmap for CA1-mPFC spatial bandit work. It supplements the generic execution roadmap and is the preferred queue when the scientific target is replay, theta sequences, value modulation, or cross-region coordination.

> **Reconciled 2026-07-08:** P1/P2/P2.5/P3/P6 have shipped. The active
> scientific queue now starts from missing replay/value modules and from
> coupling plans that avoid the 2026-06-21 spike-only latent identifiability
> failure.

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
- `src/state_space_practice/sgd_fitting.py`: SGD fitting mixin across model classes
- `src/state_space_practice/contingency_belief.py`: hidden contingency/task-state model
- `src/state_space_practice/behavioral_uncertainty.py`: uncertainty and surprise summaries for behavioral posteriors
- `src/state_space_practice/switching_choice.py`: switching behavioral strategy model
- `src/state_space_practice/oscillator_regularization.py`: regularized directed influence connectivity penalties

Not yet implemented:

- joint learning + representational drift model (P4)
- adaptive decoder with drifting tuning curves (P5)
- cross-session drift and covariate-driven representational drift
- joint belief-state decoder
- represented-content or replay state separate from physical position (S1)
- value-dependent switching into local versus nonlocal sequence modes (S2)
- heterogeneous CA1 and mPFC latent blocks in one coupled model (S3)
- explicit fast or slow hierarchy tying sequence content, trial belief, and session drift (S4)
- Gaussian-sum replay decoder
- dynamic spike-history coupling with observed lagged-spike regressors

## Current Priority Queue

The previous infrastructure/behavioral queue is complete. Current priorities
should focus on missing replay/value modules and coupling plans that are
scientifically identifiable:

| Priority | Plan | Track | Status |
|---|---|---|---|
| N1 | `dynamic-neuron-coupling.md` | Coupling | Not started — de-risked by observed spike-history regressors |
| N2 | `ca1-represented-state-switching.md` | Replay/value | Not started — best first step for CA1 represented-content claims |
| N3 | `gaussian-sum-filter-replay.md` | Replay decoder | Planned — additive extension with K=1 parity against `position_decoder.py` |
| N4 | `cross-session-drift.md` | Neural drift | Not started — prerequisite for long-timescale remapping |
| N5 | `sgd-v2-improvements.md` | Infrastructure | Planned — start with `log_det_jacobian`; benchmark-gate chunked SGD |
| N6 | `adaptive-decoder.md` | Neural | Not started — useful for long recordings after drift basics |
| N7 | `joint-learning-drift.md` | Integration | Not started — speculative; prototype first |

### Behavioral Modeling Sub-Roadmap

The behavioral plans (P2, P2.5, P3, P4) form a conceptual progression:

```
MultinomialChoiceModel (DONE) ──► CovariateChoiceModel (DONE)
        │                                    │
        ▼                                    ▼
  P2: Contingency Belief              P3: Switching Choice
  "What does the animal think         "How is the animal
   the world rules are?"               learning right now?"
      │                                    ▲
      ▼                                    │
 P2.5: Uncertainty-Aware Modeling ───────────┘
 "How uncertain are the posteriors,
  and when are they surprising?"
        │                                    │
        └────────────┬───────────────────────┘
                     ▼
            P4: Joint Learning + Drift
            "Do learning rate and place field
             drift share the same brain state?"
```

P2 and P3 are independent and answer different questions. P2.5 adds first-class
uncertainty summaries and surprise terms on top of the existing behavioral
posteriors, providing the cleanest bridge from behavioral posteriors to later
neural analyses. P4 remains the ambitious integration.

## Scientific Track

The minimum publishable path (CA1 replay + value gating) remains the shortest
route to a novel scientific claim. S1 is also listed in the current queue because
it is the required first step for this track:

| Order | Plan | Status | Depends On |
|---|---|---|---|
| S1 | `ca1-represented-state-switching.md` | Current queue N2 | Position Decoding (DONE) |
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
| `cross-region-coupling.md` | LFP-conditioned coupling stack | Spike-only latent version is blocked as specified; rewrite around observed field anchors |
| `dynamic-neuron-coupling.md` | SGD Fitting (DONE) | Time-varying spike-history GLM with observed lagged-spike regressors |
| `principled-stabilization-refactor.md` | None | Partly superseded by 2026-07 remediation; remaining in-jit guard work can stay deferred |

## Entry and Exit Gates

### P1: SGD Fitting All Models

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/ -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/ -v
uv run ruff check src/state_space_practice
```

### P2: Contingency Belief

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

### P3: Switching Choice Model

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_switching_choice.py -v
uv run ruff check src/state_space_practice
```

### P2.5: Uncertainty-Aware Behavioral Modeling

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_contingency_belief.py -q
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_behavioral_uncertainty.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_contingency_belief.py -v
uv run ruff check src/state_space_practice
```

Stop condition:

- uncertainty summaries are unstable, uninterpretable, or do not remain aligned to the existing posterior quantities under regression tests

### P4: Joint Learning + Drift

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_smith_learning_algorithm.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_switching_kalman.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_state_dependent_learning.py src/state_space_practice/tests/test_joint_discrete_state.py src/state_space_practice/tests/test_state_dependent_drift.py src/state_space_practice/tests/test_joint_learning_drift_model.py -v
uv run ruff check src/state_space_practice
```

### P5: Adaptive Decoder

Entry gate:

```bash
uv run pytest src/state_space_practice/tests/test_position_decoder.py -v
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_adaptive_decoder.py src/state_space_practice/tests/test_position_decoder.py src/state_space_practice/tests/test_place_field_model.py -v
uv run ruff check src/state_space_practice
```

### P6: Regularized Oscillator Connectivity (unblocked)

Entry gate:

```bash
# P1 oscillator SGD is DONE
uv run pytest src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v -k "sgd or SGD"
```

Exit gate:

```bash
uv run pytest src/state_space_practice/tests/test_oscillator_regularization.py src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v
uv run ruff check src/state_space_practice
```

### S1: CA1 Represented-State Switching (deferred)

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

- local-mode decoding cannot recover the standard position decoder on synthetic data

### S2: Value-Gated Sequence Expression (deferred)

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

- value features do not change local versus nonlocal occupancy or nonlocal represented destination on held-out data

## Why This Queue

- **P1 (SGD Fitting)** is **DONE** — all model classes have `fit_sgd()`, P6 is unblocked.
- **P2 (Contingency Belief)** models what the animal believes about the world, complementing P3's strategy model. Later serves as mPFC belief latent for S3.
- **P2.5 (Uncertainty-Aware Behavioral Modeling)** exposes value variance, belief entropy, predictive uncertainty, and surprise as first-class behavioral outputs.
- **P3 (Switching Choice)** is the behavioral strategy-switching model — directly interpretable and the first model that infers exploit/explore without the experimenter's contingency table.
- **P4 (Joint Learning+Drift)** is the ambitious integration linking behavioral learning and neural remapping. SPECULATIVE but scientifically important — implements prototype-first.
- **P5 (Adaptive Decoder)** enables multi-hour decoding, needed for any long-session neural analysis.
- **P6 (Regularized Oscillator)** is **DONE** — edge L1, area group L2, state-shared group L2 penalties on DIM coupling.
- The scientific track (S1-S4) is deferred, but all dependencies are already met for S1. It can be reactivated at any time; S1 should precede S2/S3/S4.
- Dynamic spike-history coupling is now the safest coupling plan to run first. Cross-region latent oscillator coupling should wait for the LFP-conditioned rewrite.
