# Spatial Bandit Latent Modeling Roadmap

> For Claude: Use this roadmap for CA1-mPFC spatial bandit work. It supplements the generic execution roadmap and is the preferred queue when the scientific target is replay, theta sequences, value modulation, or cross-region coordination.

## Purpose

This roadmap reorganizes the next modeling steps around what the repository already supports:

- continuous latent filtering and smoothing in `kalman.py`
- switching discrete-state inference in `switching_kalman.py`
- point-process observation updates in `point_process_kalman.py`
- CA1 encoding and decoding foundations in `place_field_model.py` and `position_decoder.py`
- bandit belief or value inference in `multinomial_choice.py`

The main missing layer is represented content: there is no checked-in model that distinguishes the rat's physical location from the location currently represented by CA1 activity. The roadmap below adds that layer first, then builds value gating, timescale hierarchy, and CA1-mPFC coupling on top of it.

## Codebase Reality Check

Already implemented and reviewed:

- `src/state_space_practice/position_decoder.py`: fixed-place-field decoding of physical position from spikes
- `src/state_space_practice/multinomial_choice.py`: latent option-value filtering and smoothing from behavior
- `src/state_space_practice/place_field_model.py`: drifting spatial encoding weights on spline bases
- `src/state_space_practice/switching_kalman.py`: generic discrete-continuous switching inference
- `src/state_space_practice/switching_point_process.py`: switching point-process filtering for spike observations
- `src/state_space_practice/oscillator_utils.py` and `src/state_space_practice/oscillator_models.py`: block-structured switching oscillator infrastructure

Not yet implemented:

- represented-content or replay state separate from physical position
- value-dependent switching into local versus nonlocal sequence modes, plus destination bias within nonlocal content
- heterogeneous CA1 and mPFC latent blocks in one coupled model
- explicit fast or slow hierarchy tying sequence content, trial belief, and session drift together

## Recommended Queue

| Order | Plan | Why It Comes Here | Depends On |
|---|---|---|---|
| 1 | `2026-04-03-position-decoding.md` | Reusable CA1 rate-map and decoding foundation | None |
| 2 | `2026-04-04-multinomial-choice-model.md` | Reusable latent belief or value foundation | None |
| 3 | `2026-04-05-ca1-represented-state-switching.md` | First explicit model of local vs nonlocal represented content | Position decoding |
| 4 | `2026-04-05-value-gated-sequence-expression.md` | First novel test of whether latent value changes CA1 nonlocal expression or destination bias | Represented-state switching + multinomial choice |
| 5 | `2026-04-04-joint-belief-state-decoder.md` | Behavior plus neural belief-state foundation, best interpreted as mPFC or task-variable neural model | Multinomial choice |
| 6 | `2026-04-05-coupled-dual-latent-ca1-mpfc.md` | First content-level cross-region model with distinct CA1 and mPFC latents | Represented-state switching + joint belief |
| 7 | `2026-04-04-spatial-value-model.md` | Mixed-selectivity decomposition after represented-content foundation exists | Multinomial choice + place-field infrastructure |
| 8 | `2026-04-05-hierarchical-multi-timescale.md` | Later synthesis layer linking fast sequence content, trial belief, and slow drift | Represented-state switching + multinomial choice + cross-session drift foundation |
| 9 | `2026-04-03-cross-region-coupling.md` | Optional later oscillator-level communication model after content-level coupling is interpretable | Switching point-process + oscillator abstractions |

Parallel infrastructure, not on the shortest scientific path:

- `2026-04-03-cross-session-drift.md`
- `2026-04-03-adaptive-decoder.md`
- `2026-04-04-covariate-driven-drift.md`
- `2026-04-03-joint-learning-drift.md`
- `2026-04-06-contingency-belief-latent-task-state.md`

These remain useful, especially for long-timescale remapping, but they are not the first route to replay, theta-sequence, and value-modulated content in CA1.

## Minimum Publishable Path

If the goal is to be pragmatic but still say something genuinely new, the shortest path is:

1. `2026-04-03-position-decoding.md`
2. `2026-04-04-multinomial-choice-model.md`
3. `2026-04-05-ca1-represented-state-switching.md`
4. `2026-04-05-value-gated-sequence-expression.md`

This four-plan path is the recommended first milestone because it supports a concrete scientific claim:

- CA1 alternates between local and nonlocal represented content
- latent value inferred from behavior biases nonlocal sequence expression or destination content

That claim is more novel than standard place-field or value-regression analyses, but it still builds almost entirely on checked-in numerical infrastructure.

## Second-Stage Path

Only after the minimum publishable path is working should you add:

5. `2026-04-04-joint-belief-state-decoder.md`
6. `2026-04-05-coupled-dual-latent-ca1-mpfc.md`

This is the first defensible route to a cross-region claim, because it asks whether an mPFC-like belief latent improves explanation of CA1 nonlocal content beyond behavior alone.

## Later or Optional Extensions

These are useful, but they should not be prerequisites for the first novel result:

- `2026-04-04-spatial-value-model.md`
- `2026-04-05-hierarchical-multi-timescale.md`
- `2026-04-03-cross-region-coupling.md`
- `2026-04-03-cross-session-drift.md`
- `2026-04-03-adaptive-decoder.md`
- `2026-04-04-covariate-driven-drift.md`
- `2026-04-03-joint-learning-drift.md`

## Entry and Exit Gates

### Order 3: CA1 Represented-State Switching

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

### Order 4: Value-Gated Sequence Expression

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

### Order 5: Joint Belief-State Decoder

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

- neural observations do not improve held-out belief or choice inference over behavior-only baseline

### Order 6: Coupled Dual-Latent CA1-mPFC

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_joint_belief_decoder.py src/state_space_practice/tests/test_switching_point_process.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_dual_latent_ca1_mpfc.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- one region's latent can be removed without reducing held-out prediction of the other region or behavior

### Order 8: Hierarchical Multi-Timescale

Entry gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py -v
```

Exit gate:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_hierarchical_multi_timescale.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Stop condition:

- the hierarchy collapses to one timescale or cannot produce stable posterior summaries across fast and slow layers

## Why This Queue

- The repository already has strong support for physical-position decoding and behavior-driven value inference.
- The missing scientific object is represented content, so that layer should be added before any cross-region story.
- Value-gated sequence expression now comes before neural belief coupling because it already has a pragmatic behavior-first input path and gives the shortest novel result.
- A contingency-belief or latent-task-state model is a useful parallel behavior-side extension, especially for later mPFC interpretation, but it is not required for the shortest CA1-first publishable path.
- Cross-region latent coupling should come only after the CA1-only content model and the mPFC-like belief model both work independently.
- A modular multi-timescale hierarchy is best treated as a later synthesis layer rather than part of the first publishable result.
- The existing oscillator cross-region plan remains useful, but only after the content-level models are in place.