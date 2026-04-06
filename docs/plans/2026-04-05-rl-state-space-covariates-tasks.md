# RL in State-Space Form — Task Breakdown

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Implement the covariate-driven choice model described in `docs/plans/2026-04-05-rl-state-space-covariates.md`.

**Design doc:** `docs/plans/2026-04-05-rl-state-space-covariates.md`

**Key files:**

- Create: `src/state_space_practice/covariate_choice.py`
- Create: `src/state_space_practice/tests/test_covariate_choice.py`
- Reference: `src/state_space_practice/multinomial_choice.py` (softmax update, filter/smoother patterns)
- Reference: `src/state_space_practice/kalman.py` (psd_solve, symmetrize, _kalman_smoother_update)

**Prerequisite Gates:**

- Verify `multinomial_choice.py` passes all tests before starting.
- Verify `_softmax_update_core` signature and `_kalman_smoother_update` return convention.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py -v`
- Neighbor regression: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_kalman.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

**Critical design decisions:**

- The covariate matrix `u_t` is `(n_trials, d)` where `d` is the number of covariates. It is always provided by the user — no automatic covariate construction.
- The input-gain matrix `B` is `(K-1, d)`. Each column maps one covariate to value updates for all non-reference options.
- The prediction step is `pred_mean = filt_mean + B @ u_t`, not `pred_mean = A @ filt_mean + B @ u_t`. The transition matrix is always identity (random walk with input). This simplifies the M-step.
- The M-step for B is closed-form (linear regression of smoothed value increments on covariates). Do NOT use gradient-based optimization.
- When `covariates=None`, the model reduces exactly to `MultinomialChoiceModel` (random walk). This is the parity test.
- `@jax.jit` the filter core with `n_options` as `static_argnames`.
- Use `jax.vmap` over the beta grid in the M-step (same pattern as `MultinomialChoiceModel`).

**MVP Scope Lock:**

- Dynamics covariates only (B matrix). No observation covariates (Θ).
- Scalar residual process noise `Q = q * I`.
- EM learns B, Q, and β jointly.
- Require: no-covariate parity with MultinomialChoiceModel, Rescorla-Wagner equivalence, K=2 Smith parity.

**Defer:**

- Observation covariates (Θ in softmax).
- Per-option process noise.
- Regularization on B.

---

## Task 1: Covariate-Driven Prediction Step

Implement the prediction step with control input and the M-step for B.

### Tests to write:

```python
class TestCovariatePrediction:
    def test_prediction_with_covariates():
        # pred_mean = filt_mean + B @ u_t
        # B = [[0.5, 0], [0, 0.3]] (2 options, 2 covariates)
        # u_t = [1, 0] → pred_mean = filt_mean + [0.5, 0]

    def test_prediction_without_covariates():
        # B = None or zeros → pred_mean = filt_mean (random walk)

    def test_b_mstep_recovers_known_input():
        # Simulate: x_t = x_{t-1} + B_true @ u_t (no noise)
        # Run M-step → B_hat should be close to B_true

    def test_b_mstep_shapes():
        # B should be (K-1, d)

    def test_q_mstep_with_covariates():
        # Residual Q should be smaller when B explains the variance
```

### Implementation:

```python
def covariate_predict(
    filt_mean: Array,      # (K-1,)
    filt_cov: Array,       # (K-1, K-1)
    covariates_t: Array,   # (d,)
    input_gain: Array,     # (K-1, d)
    process_noise_cov: Array,  # (K-1, K-1)
) -> tuple[Array, Array]:
    """Prediction step with control input."""
    pred_mean = filt_mean + input_gain @ covariates_t
    pred_cov = filt_cov + process_noise_cov
    return pred_mean, pred_cov

def m_step_input_gain(
    smoothed_values: Array,    # (T, K-1)
    covariates: Array,         # (T, d) — u_1..u_T, where u_t drives x_{t-1} → x_t
) -> Array:
    """Closed-form M-step for input-gain matrix B.

    B_hat = [Σ_t Δm_t u_t'] @ [Σ_t u_t u_t']^{-1}
    where Δm_t = m_{t|T} - m_{t-1|T}.
    """
    diff = smoothed_values[1:] - smoothed_values[:-1]  # (T-1, K-1)
    u = covariates[1:]  # (T-1, d) — u_t at trial t drives Δm_t

    # Cross term: Σ Δm_t u_t'
    cross = jnp.einsum("ti,tj->ij", diff, u)   # (K-1, d)
    # Covariate gram: Σ u_t u_t'
    gram = jnp.einsum("ti,tj->ij", u, u)       # (d, d)

    B_hat = psd_solve(gram.T, cross.T).T
    return B_hat
```

### Verification checkpoint:

- [ ] All tests pass
- [ ] `ruff check` passes
- [ ] B_hat recovers known B_true on noiseless synthetic data

### Commit:

```bash
git commit -m "Add covariate-driven prediction step and B M-step"
```

---

## Task 2: Covariate Choice Filter and Smoother

Build the forward filter and RTS smoother with covariate-driven dynamics.

### Tests to write:

```python
class TestCovariateChoiceFilter:
    def test_output_shapes():
        # 100 trials, 3 options, 2 covariates

    def test_reward_covariate_increases_chosen_value():
        # Always choose option 1, reward option 1 → value[0] should
        # increase faster than without covariates

    def test_no_covariate_parity_with_multinomial():
        # With B=zeros and same Q/β, filter output should match
        # MultinomialChoiceModel exactly

    def test_marginal_ll_is_finite():

class TestCovariateChoiceSmoother:
    def test_output_shapes():
    def test_smoother_reduces_variance():
    def test_last_trial_matches_filter():
    def test_smoother_cross_cov_shape():
```

### Implementation:

```python
def covariate_choice_filter(
    choices: ArrayLike,
    n_options: int,
    covariates: Optional[ArrayLike] = None,
    input_gain: Optional[ArrayLike] = None,
    process_noise: float = 0.01,
    inverse_temperature: float = 1.0,
    init_mean: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
) -> ChoiceFilterResult:
```

Use `jax.lax.scan`. Each step:
1. `pred_mean = filt_mean + B @ u_t` (or `filt_mean` if no covariates)
2. `pred_cov = filt_cov + Q`
3. `softmax_observation_update(pred_mean, pred_cov, choice_t, ...)`

Reuse `ChoiceFilterResult` and `ChoiceSmootherResult` from `multinomial_choice.py`.
Reuse `_rts_smoother_pass` from `multinomial_choice.py` (smoother doesn't change — it uses the stored filtered values).

### Verification checkpoint:

- [ ] All tests pass
- [ ] No-covariate parity: filter output matches `multinomial_choice_filter` exactly
- [ ] `ruff check` passes

### Commit:

```bash
git commit -m "Add covariate-driven choice filter and smoother"
```

---

## Task 3: CovariateChoiceModel Class with EM

Wrap into a model class with EM for B, Q, and β.

### Tests to write:

```python
class TestCovariateChoiceModel:
    def test_init_and_repr():
    def test_fit_returns_log_likelihoods():
    def test_is_fitted():

    def test_fit_learns_reward_sensitivity():
        # Reward option 1 → B[0, reward_col] should be positive

    def test_fit_without_covariates_matches_multinomial():
        # KEY PARITY: same data, no covariates, same params →
        # same log-likelihood as MultinomialChoiceModel

    def test_fit_with_covariates_improves_ll():
        # On data generated with known B, covariate model should
        # have higher LL than no-covariate model

    def test_residual_q_smaller_with_covariates():
        # When B explains the drift, residual Q should shrink

    def test_em_log_likelihood_non_decreasing():

    def test_bic_comparison():
        # Covariate model has more params — BIC should still
        # favor it when covariates are informative

    def test_rescorla_wagner_equivalence():
        # KEY VALIDATION: K=2, single reward covariate, Q=0 (fixed).
        # B should recover the RL learning rate α.
        # Generate data with known α, fit model, check B ≈ α.

    def test_choice_probabilities():
    def test_summary():
    def test_compare_to_null():

    def test_input_gain_matrix_shape():
        # After fit, model.input_gain_ should be (K-1, d)
```

### Implementation:

```python
class CovariateChoiceModel:
    """Multi-armed bandit with covariate-driven value dynamics.

    Extends MultinomialChoiceModel with input-driven value updates:
        x_t = x_{t-1} + B @ u_t + noise

    where B is a learned input-gain matrix mapping trial covariates
    to value updates. When covariates=None, reduces to
    MultinomialChoiceModel (pure random walk).

    Typical workflow::

        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        model.fit(choices, covariates=reward_matrix, verbose=True)
        print(model.summary())
        print("Reward sensitivity:", model.input_gain_)
    """
```

EM loop:
1. E-step: run smoother with current B, Q, β
2. M-step for B: `m_step_input_gain(smoothed_values, covariates)`
3. M-step for Q: same formula as MultinomialChoiceModel but using residuals `Δm - B @ u`
4. M-step for β: grid search + golden-section (same as MultinomialChoiceModel)

### Verification checkpoint — CRITICAL:

- [ ] All tests pass, `ruff check` passes
- [ ] **No-covariate parity:** identical LL to MultinomialChoiceModel on same data
- [ ] **Rescorla-Wagner equivalence:** K=2, reward covariate, Q=0, B ≈ known learning rate
- [ ] **Covariate improvement:** LL improves over random walk on data with known B
- [ ] EM monotonic (within tolerance)
- [ ] B has correct sign: rewarding option k → B[k-1, reward_col] > 0

### Commit:

```bash
git commit -m "Add CovariateChoiceModel with EM for input-gain matrix"
```

---

## Task 4: Simulation, Diagnostics, and Rescorla-Wagner Comparison

### Tests to write:

```python
class TestSimulateRLChoiceData:
    def test_output_shapes():
    def test_choices_are_valid():
    def test_reward_driven_choices():
        # High B → choices track rewarded option
    def test_seed_reproducibility():

class TestCovariateChoiceModelPlotting:
    def test_plot_values_with_covariates():
    def test_plot_input_gains():
        # Bar plot of B matrix columns
    def test_plot_convergence():
    def test_plot_summary():

class TestRescorlaWagnerComparison:
    def test_rw_equivalence_recovers_learning_rate():
        # Generate data with known α from standard RW update rule
        # Fit CovariateChoiceModel with reward covariate, Q=0
        # Check: B ≈ α (within tolerance)

    def test_rw_with_multiple_learning_rates():
        # K=3, different learning rates per option
        # B should recover the per-option rates
```

### Implementation:

- `simulate_rl_choice_data(n_trials, n_options, input_gain, covariates, ...)` — generates synthetic bandit data with known B and covariate-driven value evolution.
- `plot_input_gains(option_labels, covariate_labels)` — bar plot of B matrix.
- `plot_values(...)` — reuse pattern from MultinomialChoiceModel.
- `plot_summary(...)` — 3-panel diagnostic.

### Verification checkpoint:

- [ ] ALL tests pass
- [ ] `ruff check` passes
- [ ] Full integration: simulate with known B → fit → recover B → compare to null
- [ ] Rescorla-Wagner comparison passes
- [ ] Existing multinomial_choice tests still pass
- [ ] Existing kalman tests still pass

### Commit:

```bash
git commit -m "Add RL simulation, Rescorla-Wagner comparison, and diagnostics"
```
