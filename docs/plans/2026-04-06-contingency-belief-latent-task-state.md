# Contingency-Belief / Latent Task-State Model Implementation Plan — COMPLETE

> **Status:** DONE as of 2026-04-09. All tasks complete, 37 tests, smoke check passes.

**Goal:** Build a behavior-first latent task-state model that infers hidden contingency or rule states from bandit behavior, supports input-output transitions driven by trial covariates, and provides a reusable latent foundation for later mPFC-linked neural models.

**Architecture:** Start with an input-output HMM whose hidden state represents latent contingency or task state and whose observations are reward outcomes and choices. The first version is discrete-only and focuses on belief over hidden world state. A later extension adds a continuous value state conditioned on the discrete state so the model can bridge to the existing choice-state-space stack without forcing all task uncertainty into continuous values.

**Tech Stack:** JAX, `multinomial_choice.py`, `covariate_choice.py`, `switching_kalman.py`, `kalman.py`, pytest.

---

## Why This Plan Exists

The repository currently has two strong behavior-side abstractions:

- continuous latent option values from `multinomial_choice.py`
- continuous latent values with covariate-driven dynamics from `covariate_choice.py`

It also has a planned strategy-switching model in `2026-04-05-switching-choice-model.md`.

What is still missing is a latent for **what the animal thinks the world is**, rather than only:

- what value each option has
- which behavioral strategy is active

This plan fills that gap by modeling hidden contingency or latent task state directly.

Typical scientific uses:

1. Separate latent world-state inference from latent strategy switching.
2. Ask whether mPFC better reflects contingency belief than scalar value.
3. Test whether CA1 nonlocal content aligns with inferred hidden task state.
4. Compare hidden-state belief against the Frank Lab HMM-style interpretation without importing their task-specific code directly.

## Scientific Scope

This plan is intentionally scoped to remain inside the repository's current boundary:

- It is **not** a task-structured stem/leaf behavior model.
- It is **not** a replay-summary interface plan.
- It **is** a latent-state behavior model that can later serve as an mPFC-like belief latent.

## Mathematical Model

### MVP: discrete latent task state with input-output transitions

Let:

- `s_t in {1, ..., S}` be the latent contingency or task state
- `a_t in {0, ..., K-1}` be the observed choice
- `r_t in {0, 1}` or `R` be the observed reward outcome
- `h_t` be observed transition covariates (e.g. session reset, trial index, surprise proxy)
- `z_t` be observed choice-bias covariates

Generative model:

```text
State transitions:
    P(s_t = j | s_{t-1} = i, h_t)
      = softmax_j(eta[i, j] + Gamma[i, j, :] @ h_t)

Reward emission:
    r_t | s_t, a_t ~ Bernoulli(rho[s_t, a_t])

Choice policy:
    a_t | s_t ~ Categorical(softmax(beta * V[s_t, :] + Theta @ z_t))
```

Parameters:

- `eta`: baseline transition logits, shape `(S, S)`
- `Gamma`: transition-covariate weights, shape `(S, S, d_h)`
- `rho`: state-specific reward probabilities, shape `(S, K)`
- `V`: state-specific choice preferences or expected values, shape `(S, K)`
- `beta`: inverse temperature
- `Theta`: optional observation-bias weights

The posterior object of interest is:

```text
alpha_t(j) = P(s_t = j | a_{1:t}, r_{1:t})
```

This is the trial-by-trial latent task-state belief.

### Phase 2: hybrid discrete-continuous extension

Once the discrete-only model works, add a continuous value latent:

```text
s_t ~ input-output Markov chain
x_t = A_{s_t} x_{t-1} + B_{s_t} u_t + w_t
a_t ~ Categorical(softmax(beta * [0, x_t] + Theta @ z_t))
r_t ~ p(r_t | s_t, a_t)
```

This turns the latent task-state model into a hybrid belief/value model and connects naturally to `switching_choice_model.md`, but it is explicitly out of scope for the MVP.

## Codebase Reality Check

- Reusable softmax observation machinery already exists in `src/state_space_practice/multinomial_choice.py`.
- Covariate-conditioned policy offsets already exist in `src/state_space_practice/covariate_choice.py`.
- Generic switching utilities exist for discrete-continuous models in `src/state_space_practice/switching_kalman.py`, but they do not currently provide a discrete-only input-output HMM API.
- A new module is required: `src/state_space_practice/contingency_belief.py`.
- New tests are required: `src/state_space_practice/tests/test_contingency_belief.py`.

## Feasibility Status

**PARTIAL**

Reasoning:

- The discrete filtering and softmax observation pieces are straightforward.
- Input-output transition logits are new infrastructure and should be implemented carefully.
- The MVP is still an integration task, but it is not as plug-and-play as the current switching-choice plan.

## Verification Gates

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_switching_kalman.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

## MVP Scope Lock

- Discrete-only latent task state in the first version.
- Reward and choice observations only.
- Input-output transitions with a small covariate design matrix.
- No task-specific stem/leaf action hierarchy.
- No neural observations in the first version.
- No semi-Markov dwell times in the first version.

## Defer Until Post-MVP

- Continuous value latent conditioned on discrete state.
- Neural observation models for mPFC spikes.
- Semi-Markov dwell times.
- Full joint replay-belief coupling.
- Group-level hierarchical priors across subjects.

## Task 1: Create Transition-Logit Helpers and Result Containers

**Files:**
- Create: `src/state_space_practice/contingency_belief.py`
- Create: `src/state_space_practice/tests/test_contingency_belief.py`
- Modify: `src/state_space_practice/__init__.py`

**Step 1: Write the failing tests**

Add tests for:

- transition-logit shape validation
- row-normalized transition matrices
- reduction to a stationary transition matrix when covariates and weights are zero
- result container shape contracts

Example tests:

```python
def test_transition_matrix_rows_sum_to_one():
    logits = jnp.zeros((3, 3))
    trans = transition_logits_to_matrix(logits)
    np.testing.assert_allclose(trans.sum(axis=1), 1.0)


def test_zero_covariates_recover_baseline_transition():
    baseline = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    weights = jnp.zeros((2, 2, 3))
    h_t = jnp.zeros(3)
    trans = compute_input_output_transition_matrix(baseline, weights, h_t)
    expected = jax.nn.softmax(baseline, axis=1)
    np.testing.assert_allclose(trans, expected)
```

**Step 2: Run the test to verify failure**

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -v`

Expected:

- FAIL with import errors or missing symbols.

**Step 3: Implement the minimal helpers**

In `src/state_space_practice/contingency_belief.py`, add:

- `ContingencyBeliefResult`
- `transition_logits_to_matrix(logits)`
- `compute_input_output_transition_matrix(baseline_logits, transition_weights, covariates_t)`
- `compute_reward_log_likelihood(reward_t, choice_t, reward_probs)`

Use JAX arrays throughout. The helper should be JIT-compatible and deterministic.

**Step 4: Run the targeted tests again**

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -v`

Expected:

- PASS for helper and container tests.

**Step 5: Commit**

```bash
git add src/state_space_practice/contingency_belief.py src/state_space_practice/tests/test_contingency_belief.py src/state_space_practice/__init__.py
git commit -m "feat: add contingency belief transition helpers"
```

## Task 2: Implement the Discrete Forward Filter

**Files:**
- Modify: `src/state_space_practice/contingency_belief.py`
- Modify: `src/state_space_practice/tests/test_contingency_belief.py`

**Step 1: Write the failing filter tests**

Add tests for:

- posterior state probabilities have shape `(n_trials, n_states)`
- each posterior row sums to one
- session reset covariate can shift state occupancy
- obvious synthetic contingency blocks are partially recovered

Example synthetic recovery test:

```python
def test_block_switch_changes_posterior_mass():
    choices = jnp.array([0] * 40 + [1] * 40, dtype=jnp.int32)
    rewards = jnp.array([1] * 40 + [1] * 40, dtype=jnp.int32)
    covariates = jnp.zeros((80, 1))
    result = contingency_belief_filter(
        choices=choices,
        rewards=rewards,
        n_states=2,
        n_options=2,
        reward_probs=jnp.array([[0.8, 0.2], [0.2, 0.8]]),
        state_values=jnp.array([[2.0, 0.0], [0.0, 2.0]]),
        transition_logits=jnp.array([[3.0, 0.0], [0.0, 3.0]]),
        transition_covariates=covariates,
        transition_weights=jnp.zeros((2, 2, 1)),
    )
    assert result.state_posterior.shape == (80, 2)
```

**Step 2: Run the test to verify failure**

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py::test_block_switch_changes_posterior_mass -v`

Expected:

- FAIL until the filter exists.

**Step 3: Implement the filter**

Add:

- `contingency_belief_filter(...)`
- forward recursion in log-space or stabilized probability space
- per-trial choice log-likelihood from state-conditioned softmax
- per-trial reward log-likelihood from `reward_probs[state, choice_t]`

Use `jax.lax.scan` for the recursion. Keep the carry minimal:

- previous belief over states
- accumulated log-likelihood

**Step 4: Run targeted tests again**

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -v`

Expected:

- PASS for forward-filter tests.

**Step 5: Commit**

```bash
git add src/state_space_practice/contingency_belief.py src/state_space_practice/tests/test_contingency_belief.py
git commit -m "feat: add contingency belief forward filter"
```

## Task 3: Add Backward Smoothing and Joint State Probabilities

**Files:**
- Modify: `src/state_space_practice/contingency_belief.py`
- Modify: `src/state_space_practice/tests/test_contingency_belief.py`

**Step 1: Write the failing smoother tests**

Add tests for:

- smoothed state probabilities with shape `(n_trials, n_states)`
- pairwise smoothed probabilities with shape `(n_trials - 1, n_states, n_states)`
- each smoothed posterior row sums to one
- smoother is at least as sharp as filter on a block-switch synthetic example

**Step 2: Run the test to verify failure**

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py::test_smoother_shapes -v`

Expected:

- FAIL until the smoother exists.

**Step 3: Implement the smoother**

Add:

- `contingency_belief_smoother(...)`
- backward recursion using stored forward beliefs and time-varying transitions
- smoothed pair probabilities for later M-step updates

Use the same stabilization style as `switching_kalman.py` for probability vectors.

**Step 4: Run targeted tests again**

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -v`

Expected:

- PASS for smoother tests.

**Step 5: Commit**

```bash
git add src/state_space_practice/contingency_belief.py src/state_space_practice/tests/test_contingency_belief.py
git commit -m "feat: add contingency belief smoother"
```

## Task 4: Add a Minimal Model Class and EM Parameter Updates

**Files:**
- Modify: `src/state_space_practice/contingency_belief.py`
- Modify: `src/state_space_practice/tests/test_contingency_belief.py`

**Step 1: Write the failing model-class tests**

Add tests for:

- `ContingencyBeliefModel.fit(...)`
- learned reward probabilities move toward the synthetic ground truth
- zero transition weights recover a stationary latent task-state model
- `predict_state_posterior()` works after fitting

**Step 2: Run the test to verify failure**

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py::test_model_fit_recovers_reward_preferences -v`

Expected:

- FAIL until the model class exists.

**Step 3: Implement the minimal model class**

Add:

- `ContingencyBeliefModel`
- `fit(...)`
- `predict_state_posterior(...)`
- `state_posterior_` and `smoothed_state_posterior_`

For the MVP M-step:

- update `reward_probs` from smoothed expected successes/failures per state and option
- keep `transition_weights` fixed or optional in the first pass
- update baseline transition logits from smoothed joint state counts
- keep `beta` fixed or grid-searched on a small set, not fully optimized at once

Do **not** try to learn every parameter block simultaneously in the first version.

**Step 4: Run targeted and neighbor tests**

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -v`

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_switching_kalman.py -v`

Expected:

- PASS for targeted tests and no regressions in neighbors.

**Step 5: Commit**

```bash
git add src/state_space_practice/contingency_belief.py src/state_space_practice/tests/test_contingency_belief.py
git commit -m "feat: add contingency belief model"
```

## Task 5: Document the Bridge to Later Neural and Hybrid Models

**Files:**
- Modify: `docs/plans/2026-04-06-contingency-belief-latent-task-state.md`
- Optionally create: `notebooks/contingency_belief_smoke.py`

**Step 1: Write the failing scientist-facing tests or smoke checks**

Add a smoke test or notebook check that:

- fits the model on synthetic block-structured behavior
- plots posterior state occupancy over trials
- compares latent task-state belief against reward contingencies

**Step 2: Implement diagnostics**

Expose:

- posterior state occupancy over trials
- per-state reward templates
- state-transition summaries
- optional comparison to a continuous-value baseline

**Step 3: Run final verification gates**

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -v`

Run:

`conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_switching_kalman.py -v`

Run:

`conda run -n state_space_practice ruff check src/state_space_practice`

Expected:

- PASS across all targeted and neighbor gates.

## Acceptance Criteria

- The model infers a sensible trial-by-trial posterior over latent contingency states.
- On synthetic data with block-structured reward templates, posterior state occupancy tracks the hidden block sequence better than chance.
- The model reduces to a stationary latent-state model when input-output transition weights are zero.
- The plan provides a reusable latent-task-state foundation for later mPFC-oriented models without requiring a task-structured stem/leaf action model.

## Relationship to Existing Plans

- `switching-choice-model.md` remains the strategy-state plan.
- This plan is the contingency-belief or hidden-world-state plan.
- `joint-belief-state-decoder.md` can later use this latent instead of or in addition to scalar value.
- `coupled-dual-latent-ca1-mpfc.md` can later test whether mPFC latent task-state belief predicts CA1 nonlocal occupancy or represented destination.

Plan complete and saved to `docs/plans/2026-04-06-contingency-belief-latent-task-state.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**