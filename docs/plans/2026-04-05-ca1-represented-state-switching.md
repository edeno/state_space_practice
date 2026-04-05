# CA1 Represented-State Switching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a CA1 state-space model that distinguishes the rat's physical position from the position currently represented by hippocampal population activity, while inferring whether the representation is local or nonlocal.

**Architecture:** Use a continuous latent represented position `r_t` and a discrete mode `s_t`. Spikes are generated from place fields evaluated at `r_t`, not necessarily the animal's observed position. The local mode is tethered to tracked position, while the nonlocal mode allows represented content to deviate from tracked position without hard-coding whether that content is past- or future-oriented. Filtering uses point-process Laplace updates plus switching-state inference from the existing Kalman stack.

**Tech Stack:** JAX, `position_decoder.PlaceFieldRateMaps`, `_point_process_laplace_update` from `point_process_kalman.py`, `switching_kalman.py`, `place_field_model.py`.

---

## Codebase Reality Check

- `src/state_space_practice/position_decoder.py` already provides a validated physical-position decoder and reusable rate-map interpolation.
- `src/state_space_practice/point_process_kalman.py` already provides the nonlinear spike update needed for represented-position inference.
- `src/state_space_practice/switching_kalman.py` already provides the discrete-state machinery needed to infer local vs nonlocal modes.
- No checked-in module currently separates represented position from tracked position, and there is no CA1 replay or theta-sequence model in `src/state_space_practice`.

## MVP Scope Lock

- Use fixed place fields or rate maps learned upstream; do not jointly relearn place fields in this plan.
- Start with exactly two modes: `local` and `nonlocal`.
- Use the same bin width as the existing decoder; do not add compressed replay timing or semi-Markov dwell times yet.
- Classify nonlocal represented content as past-like, future-like, or goal-directed only in downstream analyses, not as hard-coded latent states.
- Support CA1 spikes only. mPFC coupling is deferred to a later plan.

## Defer Until Post-MVP

- Ripple-specific replay compression
- Hidden semi-Markov dwell times
- Value-dependent transition logits
- Cross-region latent coupling

## Verification Gates

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py src/state_space_practice/tests/test_point_process_kalman.py src/state_space_practice/tests/test_switching_kalman.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

## Mathematical Model

```text
Observed position:
    p_t in R^2

Represented state:
    r_t in R^2

Discrete mode:
    s_t in {local, nonlocal}

Mode-specific dynamics:
    r_t = A^{s_t} r_{t-1} + b^{s_t}(p_{1:t}) + w_t^{s_t}

Spike observation:
    y_{n,t} ~ Poisson(lambda_n(r_t) * dt)

Local-mode tether:
    p(r_t | s_t = local, p_t) favors r_t near p_t

Nonlocal-mode prior:
    p(r_t | s_t = nonlocal, p_{1:t}) allows departure from p_t without
    precommitting to retrospective or prospective semantics
```

The key scientific output is not just decoded position. It is the posterior over local versus nonlocal represented content and the posterior trajectory of the represented location when it departs from the animal's actual location. Whether nonlocal content is retrospective, prospective, or goal-directed is analyzed afterward from the inferred represented trajectory.

## Task 1: API Skeleton and Synthetic Fixtures

**Files:**
- Create: `src/state_space_practice/ca1_represented_state.py`
- Create: `src/state_space_practice/tests/test_ca1_represented_state.py`
- Optionally create: `notebooks/ca1_represented_state_smoke.py`

**Step 1:** Write failing tests for result shapes, mode-probability normalization, and validation errors.

**Step 2:** Run the test file and confirm failure.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py -v
```

**Step 3:** Implement minimal dataclasses and function skeletons:

- `RepresentedStateResult`
- `simulate_represented_state_data(...)`
- `ca1_represented_state_filter(...)`
- `ca1_represented_state_smoother(...)`

**Step 4:** Re-run the targeted tests until the API contract passes.

## Task 2: Local-Mode Equivalence and Nonlocal Switching

**Files:**
- Modify: `src/state_space_practice/ca1_represented_state.py`
- Modify: `src/state_space_practice/tests/test_ca1_represented_state.py`

**Step 1:** Add a failing regression test that forces all posterior mass into the local mode and checks equivalence to the existing position decoder within tolerance.

**Step 2:** Add a failing recovery test on synthetic data with known local vs nonlocal segments.

**Step 3:** Implement represented-position prediction and per-mode spike updates using existing rate-map interpolation from `position_decoder.py` and discrete-state updating from `switching_kalman.py`.

**Step 4:** Re-run targeted tests and the neighboring decoder regression suite.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py src/state_space_practice/tests/test_point_process_kalman.py -v
```

## Task 3: Tethering, Smoother, and Diagnostics

**Files:**
- Modify: `src/state_space_practice/ca1_represented_state.py`
- Modify: `src/state_space_practice/tests/test_ca1_represented_state.py`

**Step 1:** Add failing tests for local-mode tethering, finite smoother outputs, and mode-switch interpretability helpers.

**Step 2:** Implement:

- local tether penalty or pseudo-observation toward tracked position
- RTS-style smoother pass over represented position
- summary helpers for nonlocal occupancy and represented-position deviation
- post hoc utilities that compare inferred nonlocal content to recent and upcoming paths without making those categories part of the latent state

**Step 3:** Run targeted tests, neighbor regressions, and lint.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_ca1_represented_state.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_kalman.py src/state_space_practice/tests/test_position_decoder.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

## Acceptance Criteria

- In forced-local mode, the model reproduces the standard position decoder to within a small tolerance.
- On synthetic data, the model detects nonlocal represented segments above chance.
- Posterior represented positions remain finite and interpretable during local and nonlocal periods.
- The output API exposes both represented trajectories and mode probabilities.