# Joint Belief-State Decoder Implementation Plan - Task Breakdown

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Implement the joint belief-state decoder described in `docs/plans/2026-04-04-joint-belief-state-decoder.md`, where one latent value state explains both multi-armed bandit choices and neural spike observations.

**Architecture:** This task breakdown follows the companion design doc. The latent state uses the `K-1` reference-option parameterization, with a shared filter/smoother core and two optional observation updates per step (choice and spikes). The model must support behavior-only, spikes-only, and joint modes under one API.

**Tech Stack:** JAX, `multinomial_choice` for softmax updates, `place_field_model` and point-process update patterns, and smoother utilities from `kalman.py`.

**Prerequisite Gates:**

- Read and follow `docs/plans/2026-04-04-joint-belief-state-decoder.md` before implementation.
- Require multinomial choice tests to pass before Task 1.
- Verify point-process and place-field helper APIs before wiring spike updates.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v`
- Lint after each completed task: `conda run -n state_space_practice ruff check src/state_space_practice`
- Before declaring completion, run targeted tests plus neighbor regressions and verify held-out choice log-likelihood gain over behavior-only on at least one benchmark split.

**Feasibility Status:** PARTIAL (integration-heavy but directly supported by existing components)

**Codebase Reality Check:**

- Reusable behavior update path exists in `src/state_space_practice/multinomial_choice.py`.
- Reusable neural/spatial components exist in `src/state_space_practice/place_field_model.py` and `src/state_space_practice/point_process_kalman.py`.
- Planned new implementation/test files remain required: `src/state_space_practice/joint_belief_decoder.py` and `src/state_space_practice/tests/test_joint_belief_decoder.py`.

**Claude Code Execution Notes:**

- Treat this checklist as the runbook; do not reorder tasks unless a gate fails.
- Preserve one shared latent state trajectory across all modes.
- Add finite-value checks after each update path and before smoother steps.
- Keep a behavior-only reference path available in the same module for parity tests.

**MVP Scope Lock (implement now):**

- Implement K=3 as the primary case (general `K>2` support allowed if clean).
- Use scalar process noise (`Q = q * I`) with fixed per-neuron value loading in MVP.
- Implement one update schedule: predict -> choice update -> spike update.
- Require three checks: behavior-only parity, neural-only finite/stability, and joint held-out choice gain.

**Defer Until Post-MVP:**

- Hierarchical priors and rich process-noise structures.
- Fully learned per-neuron value loading during full joint EM.
- Cross-region and multi-session coupling in the same model.

**Design doc:** `docs/plans/2026-04-04-joint-belief-state-decoder.md`

**Key files:**

- Create: `src/state_space_practice/joint_belief_decoder.py`
- Create: `src/state_space_practice/tests/test_joint_belief_decoder.py`
- Reference: `src/state_space_practice/multinomial_choice.py`
- Reference: `src/state_space_practice/place_field_model.py`
- Reference: `src/state_space_practice/point_process_kalman.py`
- Reference: `src/state_space_practice/kalman.py`

**Critical design decisions:**

- Keep option 0 as fixed reference value for identifiability; latent state is `x_t in R^(K-1)`.
- Support missing-observation steps: some time bins may have spikes without choice events.
- Keep update ordering deterministic and documented (choice update before spike update).
- Preserve ablation mode symmetry: same return schema whether using behavior-only, spikes-only, or both.

---

## Task 1: API Skeleton + Failing Contracts

Create the module shell and fail-fast tests for API contracts.

### Step 1.1: Write failing tests

Create `src/state_space_practice/tests/test_joint_belief_decoder.py` with tests for:

- filter output shapes for means/covariances
- required argument validation
- behavior-only mode executes with no spike input
- schema consistency across mode flags

### Step 1.2: Run tests and confirm FAIL

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v
```

Expected: FAIL for missing module/functions.

### Step 1.3: Implement minimal skeleton

Create `src/state_space_practice/joint_belief_decoder.py` with stubs:

- `joint_belief_filter(...)`
- `joint_belief_smoother(...)`
- `fit_joint_belief_decoder(...)`
- result container(s) with stable field names

### Step 1.4: Re-run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v
```

Expected: shape/schema tests pass; math tests still fail.

### Step 1.5: Commit

```bash
git add src/state_space_practice/joint_belief_decoder.py src/state_space_practice/tests/test_joint_belief_decoder.py

git commit -m "feat: add joint belief decoder API skeleton"
```

---

## Task 2: Behavior Path Integration (Parity Gate)

Wire choice updates from `multinomial_choice` and enforce behavior-only parity.

### Step 2.1: Add failing parity tests

Add tests:

- behavior-only mode matches multinomial choice filter/smoother within tolerance
- K=2 reduced case remains consistent with expected binary behavior trends

### Step 2.2: Run parity tests and confirm FAIL

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py::test_behavior_only_matches_multinomial -v
```

Expected: FAIL due to missing integration.

### Step 2.3: Implement choice update integration

Call existing choice update/filter primitives from `multinomial_choice.py` inside the shared loop.

### Step 2.4: Re-run parity tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py::test_behavior_only_matches_multinomial -v
```

Expected: PASS.

### Step 2.5: Commit

```bash
git add src/state_space_practice/joint_belief_decoder.py src/state_space_practice/tests/test_joint_belief_decoder.py

git commit -m "feat: integrate behavior update path in joint belief decoder"
```

---

## Task 3: Spike Path Integration + Stability Checks

Integrate point-process spike update against the same latent state.

### Step 3.1: Add failing stability tests

Add tests:

- finite posterior means/covariances under moderate synthetic spikes
- deterministic outputs under fixed seed
- invalid spike shape raises clear ValueError

### Step 3.2: Run stability tests and confirm FAIL

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py::test_spike_update_finite -v
```

Expected: FAIL until spike path is wired.

### Step 3.3: Implement spike update integration

Use Laplace-style point-process update in latent value space and apply `symmetrize`/PSD-safe solves.

### Step 3.4: Re-run targeted tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v
```

Expected: PASS for stability/shape checks.

### Step 3.5: Commit

```bash
git add src/state_space_practice/joint_belief_decoder.py src/state_space_practice/tests/test_joint_belief_decoder.py

git commit -m "feat: integrate spike update path in joint belief decoder"
```

---

## Task 4: Smoother + Ablations + Held-Out Metric

Finish smoother and evaluation gates for the joint model.

### Step 4.1: Add failing tests

Add tests:

- smoother returns finite trajectories and valid covariance shapes
- last smoother state matches last filter state
- mode flags (`use_choice`, `use_spikes`) preserve output schema
- held-out choice log-likelihood helper returns finite scalar

### Step 4.2: Run tests and confirm FAIL

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py::test_smoother_and_ablations -v
```

Expected: FAIL until smoother/eval helpers are complete.

### Step 4.3: Implement smoother and evaluation helper

- Call smoother utilities from `kalman.py`.
- Add held-out choice log-likelihood function for behavior-only and joint mode comparison.

### Step 4.4: Run full verification gates

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

Expected:

- all targeted and neighbor tests pass
- lint passes
- held-out choice log-likelihood improvement is positive for at least one benchmark split

### Step 4.5: Commit

```bash
git add src/state_space_practice/joint_belief_decoder.py src/state_space_practice/tests/test_joint_belief_decoder.py

git commit -m "feat: complete joint belief decoder smoother and evaluation"
```

---

## Completion Checklist

- [ ] Behavior-only parity gate passes against multinomial choice model.
- [ ] Spike path is finite and numerically stable.
- [ ] Shared latent-state invariant is preserved in all update modes.
- [ ] Smoother outputs are finite with valid covariance structure.
- [ ] Joint model shows held-out choice gain over behavior-only in benchmark split.
- [ ] Targeted tests, neighbor regressions, and ruff all pass.
