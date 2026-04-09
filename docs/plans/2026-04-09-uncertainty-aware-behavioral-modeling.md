# Uncertainty-Aware Behavioral Modeling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Add first-class uncertainty summaries and trial-by-trial surprise to the current behavioral models so scientists can directly correlate model-derived uncertainty quantities with neural activity.

**Architecture:** Expose trial-aligned uncertainty summaries from the existing models without changing their latent-state definitions: option-value variance, predictive uncertainty, and trial-by-trial surprise for the continuous choice models; belief entropy, predictive reward uncertainty, and change-point probability for the contingency-belief model. Also expose these quantities on SwitchingChoiceModel. All quantities are computed from existing filter/smoother outputs — no new inference is needed.

**Tech Stack:** JAX, `lax.scan`, `vmap`, `multinomial_choice.py`, `covariate_choice.py`, `contingency_belief.py`, `SGDFittableMixin`, new uncertainty helper module in `src/state_space_practice/behavioral_uncertainty.py`.

---

**Prerequisite Gates:**

- Current targeted tests must be green before touching model outputs:
  - `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_contingency_belief.py -q`
- The uncertainty-aware policy MVP is non-switching only. Do not touch the planned switching-choice model in this plan.
- The uncertainty-aware policy learns its weights through `fit_sgd()` only. Do not add a fake EM M-step for uncertainty weights.

**Verification Gates:**

- Targeted tests:
  - `conda run -n state_space_practice pytest src/state_space_practice/tests/test_behavioral_uncertainty.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_contingency_belief.py -v`
- Neighbor regression:
  - `conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_kalman.py -q`
- Lint:
  - `conda run -n state_space_practice ruff check src/state_space_practice`

**Feasibility Status:** READY

**Why this plan is worth doing now:**

- The current filters and smoothers already carry the raw ingredients for many useful uncertainty summaries.
- The uncertainty-aware policy can be implemented as an additive logit term computed inside the JAX recursion, which preserves the current model family and takes advantage of `scan`.
- The outputs from this plan will be immediately useful for scientist-facing alignment to neural data, even before joint neural-behavioral inference exists.

**MVP Scope Lock:**

- First-class uncertainty summaries for `MultinomialChoiceModel`, `CovariateChoiceModel`, `ContingencyBeliefModel`, and `SwitchingChoiceModel`
- Trial-by-trial surprise (negative log predictive probability of actual choice)
- Change-point probability for ContingencyBeliefModel
- Scientist-facing stored attributes on model objects
- No new inference or policy terms — all quantities derived from existing posteriors

**Defer Until Post-MVP:**

- Uncertainty-aware choice policy (additive logit term from predicted covariance) — defer until it can be per-state in the switching model
- Separate latent volatility state or ambiguity state
- Internal decomposition into epistemic vs. aleatoric uncertainty
- Joint neural-behavioral fitting
- UCB-like exploration bonus as a policy term

---

## Target Outputs

By the end of this plan, the following model outputs should exist after fitting:

### Continuous choice models (Multinomial, Covariate, Switching)

- `predicted_option_variances_`: per-option prior variance at each trial, shape (T, K)
- `smoothed_option_variances_`: smoothed variance after backward pass, shape (T, K)
- `predicted_choice_entropy_`: entropy of the predicted choice distribution, shape (T,)
- `surprise_`: negative log predictive probability of the actual choice, shape (T,)

### Contingency belief model

- `belief_entropy_`: entropy of the posterior discrete-state belief, shape (T,)
- `predicted_reward_mean_`: expected reward per option under current belief, shape (T, K)
- `predicted_reward_variance_`: reward uncertainty per option under current belief, shape (T, K)
- `surprise_`: negative log predictive probability of the actual choice, shape (T,)
- `change_point_probability_`: 1 - max(state_posterior), shape (T,) — proxy for "a switch just happened"

### Switching choice model (additional)

- `per_state_predicted_variances_`: per-state per-option variance, shape (T, K, S)
- All continuous-model quantities above, marginalized over discrete states

---

## Task 1: Add Shared Uncertainty Helper Module

**Files:**
- Create: `src/state_space_practice/behavioral_uncertainty.py`
- Create: `src/state_space_practice/tests/test_behavioral_uncertainty.py`

### Step 1: Write failing tests for uncertainty helpers

```python
import numpy as np
import jax.numpy as jnp

from state_space_practice.behavioral_uncertainty import (
    append_reference_option,
    option_variances_from_covariances,
    categorical_entropy,
    belief_entropy,
    bernoulli_mixture_mean_variance,
    compute_surprise,
    change_point_probability,
)


def test_append_reference_option_prepends_zero_column():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = append_reference_option(x)
    np.testing.assert_allclose(out[:, 0], 0.0)
    np.testing.assert_allclose(out[:, 1:], x)


def test_option_variances_from_covariances_adds_reference_zero():
    cov = jnp.array([
        [[0.2, 0.0], [0.0, 0.5]],
        [[0.1, 0.0], [0.0, 0.3]],
    ])
    out = option_variances_from_covariances(cov)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 0], 0.0)
    np.testing.assert_allclose(out[:, 1:], jnp.array([[0.2, 0.5], [0.1, 0.3]]))


def test_categorical_entropy_matches_uniform_case():
    probs = jnp.array([[0.25, 0.25, 0.25, 0.25]])
    ent = categorical_entropy(probs)
    np.testing.assert_allclose(ent, np.log(4.0), atol=1e-6)


def test_belief_entropy_zero_for_deterministic_state():
    probs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    ent = belief_entropy(probs)
    np.testing.assert_allclose(ent, 0.0, atol=1e-8)


def test_bernoulli_mixture_mean_variance_has_expected_shape():
    state_probs = jnp.array([[0.7, 0.3], [0.1, 0.9]])
    reward_probs = jnp.array([[0.8, 0.2, 0.1], [0.2, 0.4, 0.9]])
    mean, var = bernoulli_mixture_mean_variance(state_probs, reward_probs)
    assert mean.shape == (2, 3)
    assert var.shape == (2, 3)
    assert np.all(np.asarray(var) >= 0.0)


def test_surprise_is_positive_for_unlikely_choice():
    probs = jnp.array([[0.9, 0.05, 0.05]])
    choices = jnp.array([2])  # unlikely choice
    surp = compute_surprise(probs, choices)
    assert surp[0] > 1.0  # -log(0.05) ≈ 3.0


def test_change_point_probability_is_zero_for_certain_state():
    probs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    cp = change_point_probability(probs)
    np.testing.assert_allclose(cp, 0.0, atol=1e-8)
```

### Step 2: Run tests to verify failure

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_behavioral_uncertainty.py -v`

Expected: FAIL because the module does not exist.

### Step 3: Implement the helper module

Implementation requirements:

- `append_reference_option(values)` prepends the reference option value `0`
- `option_variances_from_covariances(covariances)` returns full K-option variances with zero variance for the reference option
- `categorical_entropy(probs)` computes rowwise entropy with numerical clipping
- `belief_entropy(state_prob)` is a thin alias around `categorical_entropy`
- `bernoulli_mixture_mean_variance(state_probs, reward_probs)` computes per-option mean and variance under the discrete-state mixture
- `compute_surprise(predicted_probs, choices)` returns -log P(actual choice | predicted), shape (T,)
- `change_point_probability(state_probs)` returns 1 - max(state_posterior), shape (T,)

Suggested implementation sketch:

```python
def option_variances_from_covariances(covariances: Array) -> Array:
    diag = jnp.diagonal(covariances, axis1=-2, axis2=-1)
    zeros = jnp.zeros(diag.shape[:-1] + (1,))
    return jnp.concatenate([zeros, diag], axis=-1)


def compute_surprise(predicted_probs: Array, choices: Array) -> Array:
    eps = 1e-10
    p = jnp.clip(predicted_probs[jnp.arange(len(choices)), choices], eps, 1.0)
    return -jnp.log(p)


def change_point_probability(state_probs: Array) -> Array:
    return 1.0 - jnp.max(state_probs, axis=-1)
```

### Step 4: Re-run tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_behavioral_uncertainty.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/behavioral_uncertainty.py src/state_space_practice/tests/test_behavioral_uncertainty.py
git commit -m "feat: add behavioral uncertainty helpers"
```

---

## Task 2: Expose Uncertainty Summaries in Multinomial and Covariate Choice Models

**Files:**
- Modify: `src/state_space_practice/multinomial_choice.py`
- Modify: `src/state_space_practice/covariate_choice.py`
- Modify: `src/state_space_practice/tests/test_multinomial_choice.py`
- Modify: `src/state_space_practice/tests/test_covariate_choice.py`

### Step 1: Write failing tests for model-level uncertainty summaries

Add tests covering:

```python
def test_multinomial_choice_model_exposes_uncertainty_summaries(simulated_choices):
    model = MultinomialChoiceModel(n_options=3)
    model.fit(simulated_choices, max_iter=3)
    assert model.predicted_option_variances_ is not None
    assert model.smoothed_option_variances_ is not None
    assert model.predicted_choice_entropy_ is not None
    assert model.predicted_option_variances_.shape[1] == 3


def test_covariate_choice_model_exposes_uncertainty_summaries(simulated_data):
    model = CovariateChoiceModel(n_options=3, n_covariates=1)
    model.fit(simulated_data.choices, simulated_data.covariates, max_iter=3)
    assert model.predicted_option_variances_ is not None
    assert model.filtered_option_variances_ is not None
    assert model.smoothed_option_variances_ is not None
    assert model.predicted_choice_entropy_.shape[0] == len(simulated_data.choices)
```

### Step 2: Run targeted tests to verify failure

Run:

- `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py -k uncertainty -v`
- `conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py -k uncertainty -v`

Expected: FAIL because the summary attributes do not exist.

### Step 3: Implement summary storage on model objects

Implementation requirements:

- Add these optional attributes to both model classes:
  - `predicted_option_values_`
  - `predicted_option_variances_`
  - `filtered_option_values_`
  - `filtered_option_variances_`
  - `smoothed_option_values_`
  - `smoothed_option_variances_`
  - `predicted_choice_entropy_`
- Populate them in both `fit()` and `fit_sgd()` finalization paths
- Do not change the public `ChoiceFilterResult` or `ChoiceSmootherResult` tuple shapes in the MVP; compute the summaries from the existing result objects

Suggested helper use inside model finalization:

```python
from state_space_practice.behavioral_uncertainty import (
    append_reference_option,
    option_variances_from_covariances,
    categorical_entropy,
)

full_pred_values = append_reference_option(filter_result.predicted_values)
full_pred_vars = option_variances_from_covariances(filter_result.predicted_covariances)
logits = self.inverse_temperature_ * full_pred_values
pred_probs = jax.nn.softmax(logits, axis=1)
pred_entropy = categorical_entropy(pred_probs)
```

### Step 4: Re-run tests

Run:

- `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py -k uncertainty -v`
- `conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py -k uncertainty -v`

Expected: PASS

### Step 5: Run neighboring regressions

Run:

- `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py -q`

Expected: PASS

### Step 6: Commit

```bash
git add src/state_space_practice/multinomial_choice.py src/state_space_practice/covariate_choice.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py
git commit -m "feat: expose uncertainty summaries for choice models"
```

---

## Task 3: Expose Belief and Reward Uncertainty in ContingencyBeliefModel

**Files:**
- Modify: `src/state_space_practice/contingency_belief.py`
- Modify: `src/state_space_practice/tests/test_contingency_belief.py`

### Step 1: Write failing tests for belief uncertainty summaries

Add tests covering:

```python
def test_contingency_belief_model_exposes_entropy_and_reward_uncertainty():
    choices, rewards, _, _ = _simulate_block_bandit(n_trials=80)
    model = ContingencyBeliefModel(n_states=2, n_options=3)
    model.fit(choices, rewards, max_iter=3)
    assert model.belief_entropy_ is not None
    assert model.predicted_reward_mean_ is not None
    assert model.predicted_reward_variance_ is not None
    assert model.belief_entropy_.shape == (80,)
    assert model.predicted_reward_mean_.shape == (80, 3)
    assert model.predicted_reward_variance_.shape == (80, 3)
```

### Step 2: Run focused tests to verify failure

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -k uncertainty -v`

Expected: FAIL because the attributes do not exist.

### Step 3: Implement uncertainty summaries

Implementation requirements:

- Add these optional attributes to `ContingencyBeliefModel`:
  - `belief_entropy_`
  - `predicted_reward_mean_`
  - `predicted_reward_variance_`
- Compute them from the final posterior arrays using the helper functions from Task 1
- Use the smoothed state posterior for the initial MVP summaries; if both filtered and smoothed are useful later, add the filtered versions in a follow-up

Suggested implementation sketch:

```python
from state_space_practice.behavioral_uncertainty import (
    belief_entropy,
    bernoulli_mixture_mean_variance,
)

self.belief_entropy_ = belief_entropy(self.smoothed_state_posterior_)
mean, var = bernoulli_mixture_mean_variance(
    self.smoothed_state_posterior_,
    self.reward_probs_,
)
self.predicted_reward_mean_ = mean
self.predicted_reward_variance_ = var
```

### Step 4: Re-run focused tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -k uncertainty -v`

Expected: PASS

### Step 5: Run full contingency-belief regression suite

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_contingency_belief.py -q`

Expected: PASS

### Step 6: Commit

```bash
git add src/state_space_practice/contingency_belief.py src/state_space_practice/tests/test_contingency_belief.py
git commit -m "feat: add belief uncertainty summaries"
```

---

## Task 4: Expose Uncertainty Summaries on SwitchingChoiceModel

**Files:**
- Modify: `src/state_space_practice/switching_choice.py`
- Modify: `src/state_space_practice/tests/test_switching_choice.py`

### Step 1: Write failing tests

```python
def test_switching_choice_model_exposes_uncertainty_summaries():
    model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
    model.fit_sgd(choices, num_steps=10)
    assert model.predicted_option_variances_ is not None
    assert model.surprise_ is not None
    assert model.predicted_choice_entropy_ is not None
    assert model.predicted_option_variances_.shape == (T, 3)
    assert model.surprise_.shape == (T,)
```

### Step 2: Implement

Compute uncertainty summaries from the filter result's state-conditional
covariances, marginalized over discrete states using discrete_state_probs.
Surprise uses the predicted choice probabilities (marginalized over states).

### Step 3: Run tests and commit

---

## Task 5: Smoke Check Script

**Files:**
- Create: `notebooks/uncertainty_summaries_smoke.py`

Demonstrate all uncertainty outputs on synthetic data:
- Value variance over trials (continuous models)
- Surprise time series showing spikes at contingency changes
- Belief entropy and change-point probability (contingency model)
- Predicted choice entropy comparing exploit vs explore phases (switching model)

---

## Follow-On: Uncertainty-Aware Choice Policy

Deferred until it can be per-state in the switching model. The idea:

$$
\pi(a_t \mid s_t=j) \propto \exp\left(\beta_j Q_t(a) + W^{(j)}_u \phi_u(P_{t|t-1}) + \Theta z_t\right)
$$

where `phi_u` extracts uncertainty features from the predicted covariance.
This requires per-state policy weights in `SwitchingChoiceModel`, not
a shared uncertainty term in the non-switching model.

---

## JAX-Specific Implementation Notes

- All uncertainty helpers are pure functions (arrays in, arrays out)
- Surprise and entropy computed post-hoc from stored filter/smoother outputs
- No new inference needed — just extraction from existing posteriors

---

## Final Verification Command Bundle

Run these before claiming the work complete:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_behavioral_uncertainty.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_contingency_belief.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_kalman.py -q
conda run -n state_space_practice ruff check src/state_space_practice
```

Expected: all tests pass, no new lint errors in modified files.