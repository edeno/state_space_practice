# Coupled Dual-Latent CA1-mPFC Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a coupled model with distinct latent systems for CA1 represented content and mPFC belief or value, so that coordination can be studied at the level of latent content rather than only through oscillator or correlation structure.

**Architecture:** The CA1 side uses a represented-state switching latent with local or nonlocal mode probabilities and represented positions. The mPFC side uses a low-dimensional belief or value latent derived from behavior and mPFC spikes. Coupling is introduced through cross-latent terms in either the mPFC state dynamics, the CA1 mode-transition logits, or both. The first version should be asymmetric and interpretable before attempting a fully bidirectional model.

**Tech Stack:** JAX, `ca1_represented_state.py`, `joint_belief_decoder.py`, `multinomial_choice.py`, `switching_kalman.py`, `switching_point_process.py`.

---

## Codebase Reality Check

- The repository has strong switching and point-process infrastructure, but no existing model with heterogeneous CA1 and mPFC latent blocks.
- The current cross-region coupling plan is oscillator-oriented and not a good first content-level coordination model for this dataset.
- The joint belief decoder and represented-state CA1 model should be treated as prerequisites, not embedded assumptions.

## MVP Scope Lock

- Start with unidirectional coupling from mPFC belief state to CA1 mode transitions.
- Keep CA1 represented-position dynamics unchanged in the first pass; only modulate local versus nonlocal occupancy.
- Use behavior plus optional mPFC spikes to estimate the mPFC latent. Keep CA1 spikes as the only observation for represented content.
- Require ablations that remove the coupling and remove one region entirely.

## Defer Until Post-MVP

- Fully bidirectional continuous-state coupling
- Relay-cell mixture priors
- Oscillator-level communication subspace modeling
- One-shot joint EM over every parameter block

## Verification Gates

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_dual_latent_ca1_mpfc.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_joint_belief_decoder.py src/state_space_practice/tests/test_switching_point_process.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

## Mathematical Structure

```text
CA1 latent:
    r_t = represented position
    s_t = local or nonlocal represented-content mode

mPFC latent:
    v_t = belief or value state

Coupled transitions:
    P(s_t | s_{t-1}, v_t, p_t)
    v_t = A_v v_{t-1} + B_v h(s_{t-1}, r_{t-1}) + noise

Observations:
    y_t^{CA1} ~ Poisson(lambda^{CA1}(r_t) * dt)
    y_t^{mPFC}, c_t ~ belief-linked spike and choice observations
```

The initial scientific target is directional influence on latent content: does mPFC belief or policy state increase the probability of nonlocal CA1 sequence expression, and does CA1 represented content in turn improve choice prediction?

## Task 1: Coupled Data Structures and Synthetic Simulator

**Files:**
- Create: `src/state_space_practice/dual_latent_ca1_mpfc.py`
- Create: `src/state_space_practice/tests/test_dual_latent_ca1_mpfc.py`

**Step 1:** Write failing tests for coupled result containers, simulator output shapes, and ablation flag validation.

**Step 2:** Run the tests and confirm failure.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_dual_latent_ca1_mpfc.py -v
```

**Step 3:** Implement minimal scaffolding:

- `DualLatentResult`
- `simulate_dual_latent_bandit_data(...)`
- `fit_dual_latent_model(...)`

**Step 4:** Re-run targeted tests until the API contract passes.

## Task 2: Unidirectional mPFC-to-CA1 Coupling

**Files:**
- Modify: `src/state_space_practice/dual_latent_ca1_mpfc.py`
- Modify: `src/state_space_practice/tests/test_dual_latent_ca1_mpfc.py`

**Step 1:** Add failing tests for:

- zero coupling reducing to independent fits
- positive coupling increasing nonlocal CA1 mode occupancy and shifting represented content toward a target arm in synthetic data when the mPFC latent favors that arm

**Step 2:** Implement a wrapper that feeds mPFC latent summaries into CA1 mode-transition logits.

**Step 3:** Re-run targeted tests and neighbor regressions.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_dual_latent_ca1_mpfc.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_joint_belief_decoder.py -v
```

## Task 3: Bidirectional Diagnostics and Held-Out Benchmarks

**Files:**
- Modify: `src/state_space_practice/dual_latent_ca1_mpfc.py`
- Modify: `src/state_space_practice/tests/test_dual_latent_ca1_mpfc.py`
- Optionally create: `notebooks/dual_latent_ca1_mpfc_smoke.py`

**Step 1:** Add failing tests for held-out choice prediction and held-out CA1 mode prediction with and without coupling.

**Step 2:** Implement:

- coupling ablation utilities
- held-out metrics for choice log-likelihood and CA1 local versus nonlocal mode occupancy
- scientist-facing summaries of directed latent influence

**Step 3:** Run targeted tests, neighbor regressions, and lint.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_dual_latent_ca1_mpfc.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py src/state_space_practice/tests/test_joint_belief_decoder.py src/state_space_practice/tests/test_switching_point_process.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

## Acceptance Criteria

- With zero coupling, the model matches separate CA1 and mPFC fits.
- On synthetic data, coupling improves recovery of nonlocal CA1 occupancy, destination bias, or belief-linked choice predictions.
- Scientist-facing outputs isolate distinct CA1 and mPFC latents rather than collapsing them into one shared state.