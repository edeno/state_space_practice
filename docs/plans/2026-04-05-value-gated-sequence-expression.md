# Value-Gated Sequence Expression Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a model where latent belief or value modulates when CA1 enters local or nonlocal sequence-expression modes, and biases the content of nonlocal representations, rather than affecting firing-rate gain alone.

**Architecture:** Start from the represented-state switching CA1 model and add value-dependent transition logits for the discrete mode process. In the MVP, this is an input-output switching model: the input is a behavior-derived latent value state from `multinomial_choice.py`, and the output is CA1 spike data explained through local versus nonlocal represented content. The key object of inference is whether value changes nonlocal occupancy, represented destination, or both.

**Tech Stack:** JAX, `multinomial_choice.py`, `ca1_represented_state.py`, `switching_kalman.py`, `point_process_kalman.py`.

---

## Codebase Reality Check

- `src/state_space_practice/multinomial_choice.py` already gives a reusable latent value trajectory from behavior.
- The represented-state CA1 layer is not yet in the repository and must land first.
- There is no checked-in input-output HMM or covariate-conditioned switching model in the current codebase, so the transition-logit machinery is genuinely new.

## MVP Scope Lock

- Use behavior-derived value trajectories as the only gating covariate in the first version.
- Treat the first version explicitly as an input-output switching model with fixed value inputs driving hidden-state transitions.
- Modulate discrete mode transitions before attempting to modulate represented-state velocity or dwell time.
- Use one-step lagged value features and arm-specific value differences; keep the gating design matrix small.
- Keep dwell times Markovian for the MVP. Semi-Markov extensions are deferred.

## Defer Until Post-MVP

- Ripple-specific or theta-cycle-specific compressed replay timing
- Hidden semi-Markov dwell times
- Full joint learning of value and sequence-expression parameters in one EM loop
- Cross-region coupling to mPFC spikes

## Verification Gates

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_value_gated_sequence.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

## Mathematical Model

```text
Value state from behavior:
    v_t in R^(K-1)

CA1 represented-content mode:
    s_t in {local, nonlocal}

Covariate-dependent transitions:
    P(s_t = j | s_{t-1} = i, v_t, p_t) = softmax(eta_ij + gamma_ij^T g(v_t, p_t))

Represented-position dynamics:
    r_t follows the mode-specific model from the represented-state plan

Spike observation:
    y_{n,t} ~ Poisson(lambda_n(r_t) * dt)
```

The crucial question is whether `gamma_ij` meaningfully changes transitions into the nonlocal mode, and whether latent value biases where nonlocal content points, not merely whether spikes scale with value.

## Task 1: Value-Gated Transition Helpers

**Files:**
- Create: `src/state_space_practice/value_gated_sequence.py`
- Create: `src/state_space_practice/tests/test_value_gated_sequence.py`

**Step 1:** Write failing tests for transition-probability normalization, covariate-shape validation, and reduction to a standard Markov model when gating weights are zero.

**Step 2:** Run the tests and confirm failure.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_value_gated_sequence.py -v
```

**Step 3:** Implement minimal helpers:

- `build_value_features(...)`
- `compute_value_gated_transition_matrix(...)`
- `ValueGatedSequenceResult`

**Step 4:** Re-run targeted tests until the transition helper contracts pass.

## Task 2: Integrate Value Gating Into the CA1 Switching Model

**Files:**
- Modify: `src/state_space_practice/value_gated_sequence.py`
- Modify: `src/state_space_practice/tests/test_value_gated_sequence.py`

**Step 1:** Add failing tests that check:

- zero gating weights match the base represented-state model
- positive gating toward a target arm increases posterior nonlocal occupancy and shifts the inferred nonlocal represented position toward that arm in synthetic data

**Step 2:** Implement a filter wrapper that injects value-conditioned transition matrices into the CA1 represented-state model using value trajectories from `multinomial_choice.py`.

**Step 3:** Re-run targeted tests and the neighboring represented-state and multinomial regressions.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_value_gated_sequence.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py -v
```

## Task 3: Ablations and Held-Out Validation

**Files:**
- Modify: `src/state_space_practice/value_gated_sequence.py`
- Modify: `src/state_space_practice/tests/test_value_gated_sequence.py`
- Optionally create: `notebooks/value_gated_sequence_smoke.py`

**Step 1:** Add failing tests for ablation flags and held-out predictive metrics.

**Step 2:** Implement:

- ungated vs gated comparison helpers
- held-out nonlocal-occupancy or represented-destination prediction metrics
- summary plots of nonlocal occupancy and nonlocal represented destination as a function of latent value

**Step 3:** Run targeted tests, neighbor regressions, and lint.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_value_gated_sequence.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

## Acceptance Criteria

- Zero gating weights recover the ungated represented-state model.
- On synthetic data, higher latent value for an arm increases posterior nonlocal occupancy and or shifts nonlocal represented content toward that arm.
- The fitted model demonstrates a held-out improvement over an ungated baseline on at least one benchmark split.