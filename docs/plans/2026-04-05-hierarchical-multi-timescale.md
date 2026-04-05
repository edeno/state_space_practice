# Hierarchical Multi-Timescale Bandit Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a modular hierarchical model that links fast CA1 represented content, intermediate trial-level belief or value, and slow representational drift, so that each timescale can inform the others without immediately collapsing into one intractable joint latent system.

**Architecture:** Treat the hierarchy as three connected latent layers. The fast layer is the CA1 represented-state switching model, the intermediate layer is the trial-level belief or value model from behavior and mPFC-like signals, and the slow layer is a drift process over place-field or sequence-expression parameters across trials or sessions. The MVP uses posterior summaries from lower layers as observations for higher layers, then tightens coupling only after those summaries are validated.

**Tech Stack:** JAX, `ca1_represented_state.py`, `multinomial_choice.py`, `place_field_model.py`, `cross_session_drift` plan outputs, `switching_kalman.py`.

---

## Codebase Reality Check

- The repository already has separate infrastructure for fast spatial inference, trial-level behavioral state inference, and slow place-field drift.
- What is missing is a checked-in hierarchy that connects those layers explicitly while preserving their different timescales.
- A modular hierarchy is more compatible with the current codebase than a monolithic single-filter model.

## MVP Scope Lock

- Use posterior summaries from lower-level models as inputs to the next layer; do not start with one giant joint ELBO or fully coupled EM loop.
- Start with three timescales: within-theta-window or short-bin represented content, trial-level value, and session-level drift.
- Limit the slow layer to low-dimensional summaries such as prospective occupancy, represented-position bias, or place-field drift magnitude.
- Allow the higher-level model to modulate priors on lower-level parameters, but not vice versa in the first version.

## Defer Until Post-MVP

- Fully joint inference over all timescales
- Continuous-time replay timing models
- Cross-region dual-latent coupling in the same fit
- Rich hierarchical priors over neuron-specific parameters

## Verification Gates

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hierarchical_multi_timescale.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

## Mathematical Structure

```text
Fast layer:
    z_t^{fast} = represented position and sequence mode summaries

Intermediate layer:
    v_k = trial-level belief or value state

Slow layer:
    d_m = session-level or block-level drift state

Coupling examples:
    p(z_t^{fast} | v_k, d_m)
    p(v_k | d_m)
    p(d_m | d_{m-1})
```

The hierarchy should answer questions like: when value changes across trials, does prospective sequence content change immediately or only after slower remapping? Does slow drift explain shifts in sequence expression over days?

## Task 1: Shared Summary Interfaces

**Files:**
- Create: `src/state_space_practice/hierarchical_multi_timescale.py`
- Create: `src/state_space_practice/tests/test_hierarchical_multi_timescale.py`

**Step 1:** Write failing tests for summary-adapter dataclasses and shape validation.

**Step 2:** Run the tests and confirm failure.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_hierarchical_multi_timescale.py -v
```

**Step 3:** Implement minimal containers and adapters:

- `FastContentSummary`
- `TrialBeliefSummary`
- `SlowDriftSummary`
- `assemble_multi_timescale_dataset(...)`

**Step 4:** Re-run targeted tests until the basic interfaces pass.

## Task 2: Hierarchical Dynamics and Recovery Tests

**Files:**
- Modify: `src/state_space_practice/hierarchical_multi_timescale.py`
- Modify: `src/state_space_practice/tests/test_hierarchical_multi_timescale.py`

**Step 1:** Add failing tests for:

- one-layer ablations reducing to lower-level baselines
- synthetic recovery of a slow drift variable that modulates fast prospective occupancy

**Step 2:** Implement a first-pass hierarchical model using low-dimensional Gaussian state dynamics over summary statistics.

**Step 3:** Re-run targeted tests and neighboring regressions.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_hierarchical_multi_timescale.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py -v
```

## Task 3: Scientist-Facing Diagnostics

**Files:**
- Modify: `src/state_space_practice/hierarchical_multi_timescale.py`
- Modify: `src/state_space_practice/tests/test_hierarchical_multi_timescale.py`
- Optionally create: `notebooks/hierarchical_multi_timescale_smoke.py`

**Step 1:** Add failing tests for summary plots or reporting helpers.

**Step 2:** Implement diagnostics that expose:

- fast prospective occupancy over trials
- latent belief trajectories aligned to blocks or choices
- slow drift trajectories with uncertainty
- lag or coupling summaries across timescales

**Step 3:** Run targeted tests, neighbor regressions, and lint.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_hierarchical_multi_timescale.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_multinomial_choice.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

## Acceptance Criteria

- The model cleanly reduces to lower-level summaries when higher layers are ablated.
- Synthetic data with known slow and fast couplings is recoverable.
- Scientist-facing outputs describe fast, intermediate, and slow latent trajectories with uncertainty.