# Joint Belief-State Decoder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Build a shared latent belief-state model for multi-armed bandit behavior where the same latent value state explains trial choices and task-variable neural spiking activity, serving as the mPFC or behavior-linked latent foundation rather than the CA1 represented-content model.

**Architecture:** Use a single latent value state `x_t` (relative option values) with random-walk dynamics. Apply two observation updates at each step: (1) multinomial choice update via softmax likelihood and (2) optional neural spike update using position-conditioned firing with value modulation. Use Laplace-EKF style updates with existing Kalman smoothing utilities. This plan is for belief or value inference, not for replay or theta-sequence content; CA1 represented-content switching belongs in a separate model layer.

**Tech Stack:** JAX, `multinomial_choice` softmax update/filter/smoother APIs, `place_field_model` basis utilities, point-process update patterns from `point_process_kalman.py`, smoother primitives in `kalman.py`.

**Prerequisite Gates:**

- Require passing multinomial choice implementation and tests before starting this plan.
- Verify place-field basis and point-process updates are available and numerically stable.
- If API naming differs from assumptions below, update this plan before writing code.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v`
- Lint after each completed task: `conda run -n state_space_practice ruff check src/state_space_practice`
- Before declaring completion, verify that held-out choice log-likelihood for the joint model beats a behavior-only baseline on at least one benchmark split.

**Feasibility Status:** PARTIAL (depends on multinomial-choice completion; moderate integration risk)

**Codebase Reality Check:**

- Reusable components exist for softmax choice updates (`multinomial_choice`) and point-process spatial encoding (`place_field_model`, `point_process_kalman`).
- Shared filtering/smoothing math is already available in `kalman.py`.
- Planned new module is required: `src/state_space_practice/joint_belief_decoder.py`.
- Planned new tests are required: `src/state_space_practice/tests/test_joint_belief_decoder.py`.

**Claude Code Execution Notes:**

- Keep one shared latent state object as the design invariant; do not fork behavior and neural latent trajectories.
- Add finite-value assertions after each update (choice and spike) and before smoother passes.
- Implement a behavior-only mode in the same API for parity checks and ablations.
- Do not overload this plan with CA1 replay, retrospective or prospective sequence modes, or represented-position switching. Those belong to the represented-state and value-gated sequence plans.

**MVP Scope Lock (implement now):**

- Support K=3 options as the primary case with option-0 reference identifiability.
- Use scalar process noise (`Q = q * I`) and fixed per-neuron value loading during initial integration.
- Implement one update schedule per time step: predict -> choice update (if trial event) -> spike update (if spikes observed).
- Prioritize behavior plus mPFC-like or task-variable neural signals; treat CA1 spike inputs here as optional ablations, not the primary replay model.
- Require three acceptance checks: behavior-only parity, neural-only sanity, and joint model held-out choice gain over behavior-only.

**Defer Until Post-MVP:**

- Rich process-noise structures and hierarchical priors over value dynamics.
- Adaptive or learned per-neuron value loadings during full joint EM.
- CA1 represented-content switching, replay or theta-sequence modes, and value-gated sequence expression.
- Multi-session hierarchical coupling and cross-region latent coupling in the same model.

**References:**

- Smith, A.C., Frank, L.M., Wirth, S. et al. (2004). Dynamic analysis of learning in behavioral experiments. J Neuroscience 24(2), 447-461.
- Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J. (2006). Cortical substrates for exploratory decisions in humans. Nature 441, 876-879.
- Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004). Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering. Neural Computation 16, 971-998.

---

## Mathematical Model

### Latent state and dynamics

```text
x_t = x_{t-1} + w_t,   w_t ~ N(0, q I)

x_t in R^(K-1), with option 0 fixed as reference value 0.
```

### Choice observation model

```text
v_t = [0, x_t] in R^K
p(c_t = k | x_t) = softmax(beta * v_t)_k
c_t ~ Categorical(p_t)
```

### Spike observation model (per neuron n)

```text
log lambda_{n,t} = b_n + w_n^T Z(p_t) + alpha_n^T x_t

Z(p_t): known spatial basis evaluated at position p_t
alpha_n: value loading onto latent belief state
```

Spikes are modeled as point-process counts/rates conditioned on `lambda_{n,t}` in the existing point-process framework.

### Inference schedule (single step)

```text
1. Predict x_t from x_{t-1}
2. Apply choice likelihood update when a choice event is present
3. Apply spike likelihood update when neural observations are present
4. Save filtered moments for smoothing
```

---

## Task 1: Skeleton API and Failing Tests

**Files:**
- Create: `src/state_space_practice/joint_belief_decoder.py`
- Create: `src/state_space_practice/tests/test_joint_belief_decoder.py`
- Modify: `src/state_space_practice/__init__.py` (if needed for exports)

**Step 1: Write failing tests for API shape and basic contracts**

Add tests for:
- output shapes for filtered means/covariances
- required-key validation errors for missing observations
- behavior-only mode execution without spike input

**Step 2: Run test to verify it fails**

Run:
`conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v`

Expected:
- FAIL for missing module/functions

**Step 3: Implement minimal module skeleton**

Add stubs for:
- `joint_belief_filter(...)`
- `joint_belief_smoother(...)`
- `fit_joint_belief_decoder(...)`

**Step 4: Run test to verify first test(s) pass**

Run:
`conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v`

Expected:
- partial pass; remaining tests fail for unimplemented math

**Step 5: Commit**

```bash
git add src/state_space_practice/joint_belief_decoder.py src/state_space_practice/tests/test_joint_belief_decoder.py

git commit -m "feat: add joint belief decoder API skeleton"
```

## Task 2: Choice Update Integration and Behavior-Only Parity

**Files:**
- Modify: `src/state_space_practice/joint_belief_decoder.py`
- Modify: `src/state_space_practice/tests/test_joint_belief_decoder.py`

**Step 1: Write failing parity test vs behavior-only multinomial choice path**

Add test that when spike observations are disabled, the joint decoder matches multinomial choice filtering behavior within tolerance.

**Step 2: Run test to verify it fails**

Run:
`conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py::test_behavior_only_matches_multinomial -v`

Expected:
- FAIL with mismatch beyond tolerance

**Step 3: Implement choice update call-through**

Use existing `multinomial_choice` update/filter primitives directly from the joint filter loop.

**Step 4: Re-run parity test**

Run:
`conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py::test_behavior_only_matches_multinomial -v`

Expected:
- PASS

**Step 5: Commit**

```bash
git add src/state_space_practice/joint_belief_decoder.py src/state_space_practice/tests/test_joint_belief_decoder.py

git commit -m "feat: integrate multinomial choice updates in joint belief decoder"
```

## Task 3: Spike Update Integration and Numerical Stability

**Files:**
- Modify: `src/state_space_practice/joint_belief_decoder.py`
- Modify: `src/state_space_practice/tests/test_joint_belief_decoder.py`

**Step 1: Write failing tests for spike update stability**

Add tests for:
- finite filtered means/covariances under moderate spike rates
- deterministic output under fixed seed/synthetic data

**Step 2: Run test to verify it fails**

Run:
`conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py::test_spike_update_finite -v`

Expected:
- FAIL with non-finite values or not-yet-implemented update

**Step 3: Implement spike likelihood update in the same latent state**

Use a Laplace-style point-process update and guard Hessian solves with existing PSD/symmetrization utilities.

**Step 4: Re-run targeted tests**

Run:
`conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v`

Expected:
- PASS for stability and shape contracts

**Step 5: Commit**

```bash
git add src/state_space_practice/joint_belief_decoder.py src/state_space_practice/tests/test_joint_belief_decoder.py

git commit -m "feat: add spike update to joint belief decoder"
```

## Task 4: Smoother, Ablations, and Held-Out Evaluation

**Files:**
- Modify: `src/state_space_practice/joint_belief_decoder.py`
- Modify: `src/state_space_practice/tests/test_joint_belief_decoder.py`
- Optionally create: `notebooks/joint_belief_decoder_smoke.py`

**Step 1: Write failing tests for smoother and ablations**

Add tests that:
- smoother returns finite state trajectories
- ablation flags (`use_choice`, `use_spikes`) run without changing API schema

**Step 2: Run tests to verify failure**

Run:
`conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py::test_smoother_and_ablations -v`

Expected:
- FAIL until smoother/flags are implemented

**Step 3: Implement smoother integration and ablation flags**

Call existing smoother update logic and add a helper to compute held-out choice log-likelihood.

**Step 4: Run full verification gates**

Run:
`conda run -n state_space_practice pytest src/state_space_practice/tests/test_joint_belief_decoder.py -v`

Run:
`conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v`

Run:
`conda run -n state_space_practice ruff check src/state_space_practice`

Expected:
- all tests pass
- no lint errors

**Step 5: Commit**

```bash
git add src/state_space_practice/joint_belief_decoder.py src/state_space_practice/tests/test_joint_belief_decoder.py

git commit -m "feat: complete joint belief decoder smoother and evaluation"
```

## Completion Checklist

- Shared latent-state invariant maintained in all update paths.
- Behavior-only parity check passes.
- Joint model improves held-out choice log-likelihood over behavior-only baseline in smoke evaluation.
- Numerical stability checks (finite means/covariances) pass for filter and smoother.

## Handoff

Plan complete and saved to `docs/plans/2026-04-04-joint-belief-state-decoder.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?