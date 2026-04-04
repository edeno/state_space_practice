# Multinomial Choice Model — Implementation Tasks

> **For Claude:** Implement these tasks sequentially. Each task has verification
> steps — do NOT proceed to the next task until all verifications pass. Use the
> `jax` skill for JAX best practices. Use `test-driven-development` skill.

**Design doc:** `docs/plans/2026-04-04-multinomial-choice-model.md`

**Key files:**

- Create: `src/state_space_practice/multinomial_choice.py`
- Create: `src/state_space_practice/tests/test_multinomial_choice.py`
- Reference: `src/state_space_practice/kalman.py` (for `psd_solve`, `symmetrize`, `_kalman_smoother_update`)
- Reference: `src/state_space_practice/smith_learning_algorithm.py` (for API patterns)

**Critical design decisions:**

- The latent state is `x_t ∈ R^{K-1}`, NOT `R^K`. Option 0 is the reference
  (fixed at value 0) for identifiability. The full value vector for softmax is
  `v_t = [0, x_t]`.
- Use iterative Newton (2-3 steps with convergence check) for the Laplace
  update. The softmax log-likelihood is log-concave, so Newton converges
  reliably, but a single step may not reach the mode for large β.
- Process noise Q is scalar × I (diagonal, same drift for all options).
  `init_cov` is fixed (identity, not learned via EM).
- Follow `SmithLearningModel` patterns: `__repr__`, `is_fitted`, `summary()`,
  `bic()`, `compare_to_null()`, `plot_*()` methods, `verbose` in `fit()`.
- Task 3 tests use manually constructed choice arrays (e.g., `jnp.ones`,
  `np.random.default_rng`), NOT `simulate_choice_data` (which is Task 4).
- Apply `@jax.jit` to `multinomial_choice_filter` and `multinomial_choice_smoother`.
- Use `jax.vmap` over the β grid in the M-step for performance.

---

## Task 1: Softmax Observation Update

Build the Laplace-EKF update step for a categorical observation with softmax link.

### Step 1.1: Create test file with failing tests

Create `src/state_space_practice/tests/test_multinomial_choice.py` with:

```python
class TestSoftmaxObservationUpdate:
    def test_output_shapes(self):
        # 4 options → 3 free params. prior_mean shape (3,), prior_cov (3,3)
        # choice=2 (0-indexed, option 2 of 4)
        # Returns: post_mean (3,), post_cov (3,3), log_likelihood scalar

    def test_chosen_option_value_increases(self):
        # 3 options (2 free params). Choose option 1 (free param index 0).
        # post_mean[0] should increase relative to prior_mean[0].

    def test_unchosen_option_value_decreases(self):
        # Choose option 1. Free param for option 2 should decrease or stay.

    def test_reference_option_choice_decreases_all(self):
        # Choose option 0 (reference). All free values should decrease
        # (since reference is fixed at 0, choosing it means others are overvalued).

    def test_high_temperature_weak_update(self):
        # inverse_temperature=0.1 should produce smaller update than 5.0.

    def test_posterior_covariance_shrinks(self):
        # trace(post_cov) < trace(prior_cov).

    def test_log_likelihood_is_finite(self):
        # Should be finite and negative.

    def test_posterior_covariance_is_psd(self):
        # All eigenvalues >= 0.

    def test_invalid_choice_raises(self):
        # choice >= n_options or choice < 0 should raise ValueError
```

### Step 1.2: Run tests — verify they FAIL with ImportError

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py -v
```

### Step 1.3: Implement `softmax_observation_update`

Create `src/state_space_practice/multinomial_choice.py` with module docstring and:

```python
def softmax_observation_update(
    prior_mean: Array,          # shape (K-1,)
    prior_cov: Array,           # shape (K-1, K-1)
    choice: int,                # 0-indexed option (0 = reference)
    n_options: int,             # K total options
    inverse_temperature: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Laplace-EKF update for a categorical observation with softmax link.

    The latent state x ∈ R^{K-1} represents relative values for options
    1 through K-1. Option 0 is the reference (value fixed at 0).

    Returns posterior_mean, posterior_cov, log_likelihood.
    """
    # 1. Validate: 0 <= choice < n_options
    # 2. Build full value vector: v = [0, x] ∈ R^K (concatenate 0 + prior_mean)
    # 3. Compute softmax probabilities: p = softmax(β * v) ∈ R^K
    # 4. Log-likelihood: log(p[choice])
    # 5. Gradient w.r.t. x (K-1 dims):
    #    IMPORTANT: Build the full K-dim one-hot e_k first, THEN slice [1:]
    #    e_k = jnp.zeros(n_options).at[choice].set(1.0)
    #    gradient = β * (e_k[1:] - p[1:])
    #    When choice=0 (reference): e_k[1:] = zeros, so gradient = -β * p[1:]
    #    (all free values decrease — correct behavior)
    # 6. Neg-Hessian w.r.t. x (K-1 × K-1):
    #    β² * (diag(p[1:]) - outer(p[1:], p[1:]))
    # 7. Iterative Newton (2-3 steps with convergence check):
    #    x = prior_mean
    #    for _ in range(max_newton_steps):
    #        Recompute v=[0,x], p=softmax(β*v), gradient, neg_hessian at x
    #        posterior_precision = prior_precision + neg_hessian
    #        newton_step = solve(posterior_precision, gradient + prior_precision @ (prior_mean - x))
    #        x_new = x + newton_step
    #        if converged (|x_new - x| < tol): break
    #        x = x_new
    # 8. Posterior covariance = inv(posterior_precision) at final x
```

Import `psd_solve` and `symmetrize` from `state_space_practice.kalman`.

### Step 1.4: Run tests — verify they PASS

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py::TestSoftmaxObservationUpdate -v
```

### Step 1.5: Verification checkpoint

- [ ] All tests pass
- [ ] `ruff check` passes on the new file
- [ ] The gradient and Hessian formulas are correct for the K-1 parameterization
- [ ] Choosing the reference option (choice=0) decreases all free values
- [ ] Log-likelihood matches `jax.nn.log_softmax` at the prior mean

### Step 1.6: Commit

```bash
git commit -m "Add softmax observation update for multinomial choice model"
```

---

## Task 2: Multinomial Choice Filter and Smoother

Build the full forward filter and RTS backward smoother.

### Step 2.1: Write failing tests

```python
from typing import NamedTuple

class ChoiceFilterResult(NamedTuple):
    filtered_values: Array          # (n_trials, K-1)
    filtered_covariances: Array     # (n_trials, K-1, K-1)
    predicted_values: Array         # (n_trials, K-1) — for diagnostics
    predicted_covariances: Array    # (n_trials, K-1, K-1) — for diagnostics
    marginal_log_likelihood: Array  # scalar

class ChoiceSmootherResult(NamedTuple):
    smoothed_values: Array          # (n_trials, K-1)
    smoothed_covariances: Array     # (n_trials, K-1, K-1)
    smoother_cross_cov: Array       # (n_trials-1, K-1, K-1)
    marginal_log_likelihood: Array  # scalar

class TestMultinomialChoiceFilter:
    def test_output_shapes(self):
        # 100 trials, 4 options → filtered_values shape (100, 3)

    def test_preferred_option_has_highest_value(self):
        # 200 trials, option 1 chosen 80%. Final filtered_values[:,0] should
        # be highest (option 1 = free param index 0).

    def test_switching_preference_tracked(self):
        # First 100 trials: always choose option 1. Next 100: always option 2.
        # At trial 50: value[0] > value[1]. At trial 180: value[1] > value[0].

    def test_marginal_ll_is_finite(self):
        # Should be finite and negative.

    def test_predicted_values_stored(self):
        # predicted_values and predicted_covariances should have correct shapes.

class TestMultinomialChoiceSmoother:
    def test_output_shapes(self):
        # Same data. smoothed_values shape (100, 3).

    def test_smoother_reduces_variance(self):
        # mean trace(smoothed_cov) <= mean trace(filtered_cov)

    def test_last_trial_matches_filter(self):
        # smoothed_values[-1] == filtered_values[-1]

    def test_smoother_cross_cov_shape(self):
        # shape (n_trials-1, K-1, K-1)
```

### Step 2.2: Run tests — verify they FAIL

### Step 2.3: Implement filter

```python
def multinomial_choice_filter(
    choices: Array,                     # (n_trials,) ints
    n_options: int,                     # K
    process_noise: float = 0.01,        # scalar × I
    inverse_temperature: float = 1.0,
    init_mean: Optional[Array] = None,  # (K-1,) default zeros
    init_cov: Optional[Array] = None,   # (K-1,K-1) default identity
) -> ChoiceFilterResult:
```

Use `jax.lax.scan`. Each step:
1. Predict: `pred_mean = filt_mean`, `pred_cov = filt_cov + Q` (random walk, A=I)
2. Update: call `softmax_observation_update(pred_mean, pred_cov, choice, n_options, β)`
3. Accumulate log-likelihood
4. Store filtered values/covariances (smoother consumes these, NOT the predicted ones)
5. Also store predicted values/covariances for diagnostics

### Step 2.4: Implement smoother

```python
def multinomial_choice_smoother(
    choices: Array,
    n_options: int,
    ...same params as filter...
) -> ChoiceSmootherResult:
```

1. Run the filter
2. Use `_kalman_smoother_update` from `kalman.py` in a reverse `lax.scan`
3. The smoother consumes `filter_mean`, `filter_cov`, `process_cov`, `transition_matrix`
   — transition_matrix is identity for random walk
4. `_kalman_smoother_update` internally recomputes the predicted covariance
   from `A @ filter_cov @ A.T + Q`, so we do NOT need to pass predicted values
5. Convention: `smoother_cross_cov[t]` = `Cov(x_t, x_{t+1} | y_{1:T})`
   where index t is the earlier trial. The M-step must be consistent with this.
6. **IMPORTANT:** The reverse `lax.scan` over `filtered[:-1]` produces
   `(n_trials-1, K-1)` arrays. You MUST append the last filtered state to
   get `(n_trials, K-1)` smoothed output (same pattern as `kalman_smoother`
   in kalman.py lines 445-446). The `test_last_trial_matches_filter` test
   will catch this if forgotten.

### Step 2.5: Run tests — verify they PASS

### Step 2.6: Verification checkpoint

- [ ] All filter and smoother tests pass
- [ ] Smoother variance <= filter variance on average
- [ ] Last smoothed trial matches last filtered trial
- [ ] Marginal log-likelihood is finite and reasonable (not -inf)
- [ ] With deterministic choices (always option 1), the value for option 1
      increases monotonically in the filter output
- [ ] `ruff check` passes

### Step 2.7: Commit

```bash
git commit -m "Add multinomial choice filter and smoother"
```

---

## Task 3: MultinomialChoiceModel Class with EM

Wrap into a model class following `SmithLearningModel` patterns.

### Step 3.1: Write failing tests

```python
class TestMultinomialChoiceModel:
    def test_init_and_repr(self):
        # model = MultinomialChoiceModel(n_options=4)
        # repr should show n_options, β, Q, fitted status

    def test_fit_returns_log_likelihoods(self):
        # 300 trials, 4 options with biased choices
        # fit(choices, max_iter=5) returns list of finite floats

    def test_is_fitted(self):
        # False before fit, True after

    def test_fit_learns_from_deterministic_choices(self):
        # Always choose option 1 → high inverse_temperature after fit

    def test_fit_learns_process_noise(self):
        # Switching preferences should yield higher process_noise than stable

    def test_fit_verbose(self, capsys):
        # verbose=True should print progress

    def test_choice_probabilities(self):
        # After fit, shape (n_trials, K), rows sum to 1

    def test_bic_is_finite(self):
        # After fit, bic() returns finite value

    def test_summary_returns_string(self):
        # After fit, summary() contains key info

    def test_compare_to_null(self):
        # With biased choices, model should beat null (uniform chance = 1/K)
        # Note: null model uses uniform 1/K, not frequency-matched proportions.
        # This is the simplest baseline (no information about preferences).

    def test_em_log_likelihood_non_decreasing(self):
        # EM log-likelihood should be monotonically non-decreasing
        # (within tolerance, since β grid search is discrete).
        # Allow small decreases (< 0.1) due to grid discretization.

    def test_invalid_choice_raises(self):
        # choice >= n_options or choice < 0 should raise ValueError

    def test_two_options_consistent_with_smith(self):
        # KEY VALIDATION: With K=2 binary choices, compare multinomial vs Smith.
        # Setup: fix β=1, prob_chance=0.5, same sigma_epsilon=sqrt(Q).
        # Both models: learn_inverse_temperature=False, learn_process_noise=False
        # (fix parameters to be identical, only compare filter/smoother output).
        # Use the SAME data (binary outcomes → choices for multinomial).
        # Check: correlation > 0.95 AND mean absolute error < 0.2 in logit units.
        # The models won't be exactly equal (iterative Newton vs BFGS, K-1 vs
        # scalar parameterization) but should be very close with fixed params.
```

### Step 3.2: Run tests — verify they FAIL

### Step 3.3: Implement `MultinomialChoiceModel`

```python
class MultinomialChoiceModel:
    """Multi-armed bandit choice model with evolving option values.

    Tracks latent option values from a sequence of choices using a
    state-space model with softmax observation model. Uses EM to learn
    the drift rate (process noise) and exploration-exploitation tradeoff
    (inverse temperature).

    The latent state x_t ∈ R^{K-1} represents the relative value of
    options 1 through K-1, with option 0 as the reference (value = 0).
    This ensures identifiability of the softmax.

    Typical workflow::

        model = MultinomialChoiceModel(n_options=4)
        model.fit(choices, verbose=True)
        print(model.summary())
        fig, axes = model.plot_summary(observed_choices=choices)

    Parameters
    ----------
    n_options : int
    init_inverse_temperature : float, default=1.0
    init_process_noise : float, default=0.01
    learn_inverse_temperature : bool, default=True
    learn_process_noise : bool, default=True
    """
```

Methods to implement:

- `__init__`, `__repr__`, `is_fitted` property
- `fit(choices, max_iter, tolerance, verbose, beta_grid=None)` — EM loop:
  - E-step: run smoother
  - M-step for Q (multivariate formula, see design doc):
    ```
    Q_hat = (1/(T-1)) * sum_t [
        (m_{t|T} - m_{t-1|T})(m_{t|T} - m_{t-1|T})'
        + P_{t|T} + P_{t-1|T}
        - 2 * smoother_cross_cov[t-1]
    ]
    q = mean(diag(Q_hat)), clamped >= 1e-8
    ```
  - M-step for β: two-phase optimization:
    (a) Coarse grid search (default `[0.1, 0.3, 0.5, 1, 2, 3, 5, 8, 12]`,
    configurable via `beta_grid`). Use `jax.vmap` over grid for performance.
    (b) Golden-section refinement around best grid point (bracket =
    neighboring grid values, ~10 steps). This handles β >> 12.
  - EM iteration order: run smoother (for Q M-step), then grid search
    filters (for β M-step). This means T + N_β * T forward passes per iter.
  - Store `log_likelihood_`, `n_iter_`, `log_likelihood_history_`
- `choice_probabilities()` — softmax(β * [0, smoothed_values])
- `bic()` — parameter count: Q (1 scalar) + β (1 scalar) + initial mean (K-1 values)
  = K+1 total free parameters
- `compare_to_null(choices)` — null = uniform 1/K on every trial
- `summary()` — text overview

### Step 3.4: Run tests — verify they PASS

### Step 3.5: Verification checkpoint — CRITICAL

This is the most important checkpoint. Verify:

- [ ] All tests pass, `ruff check` passes
- [ ] **K=2 consistency**: Fit both `MultinomialChoiceModel(n_options=2)` and
      `SmithLearningModel` on the same binary data. The smoothed values should
      be highly correlated (r > 0.9). If not, there is a bug in the K-1
      parameterization or the gradient/Hessian formulas.
- [ ] Deterministic choices (always option k) produce high β and value[k] >> others
- [ ] Random uniform choices produce β near the low end of the grid and
      values near 0 (no learning)
- [ ] Switching preferences produce higher Q than stable preferences
- [ ] EM log-likelihood is monotonically non-decreasing (within tolerance)
- [ ] `compare_to_null()` detects learning on biased data, does not detect
      learning on uniform random data
- [ ] `summary()` output is readable and complete

### Step 3.6: Commit

```bash
git commit -m "Add MultinomialChoiceModel class with EM fitting"
```

---

## Task 4: Simulation, Plotting, and Polish

Add data simulation, diagnostic plots, and final polish.

### Step 4.1: Write failing tests

```python
class TestSimulateChoiceData:
    def test_output_shapes(self):
        # simulate_choice_data(n_trials=100, n_options=4, seed=42)
        # returns (choices, true_values, true_probs)

    def test_choices_are_valid(self):
        # All in [0, K)

    def test_biased_simulation(self):
        # With one option having high value, it should be chosen most often

    def test_seed_reproducibility(self):
        # Same seed → same output

class TestMultinomialChoiceModelPlotting:
    def test_plot_values_returns_fig(self):
        # After fit, returns (fig, axes) with 2 panels

    def test_plot_convergence_returns_fig(self):

    def test_plot_summary_returns_fig(self):
        # 3-panel figure

    def test_plot_requires_fit(self):
        # Should raise RuntimeError
```

### Step 4.2: Implement

- `simulate_choice_data(n_trials, n_options, process_noise, inverse_temperature, seed)`
  — generates synthetic multi-armed bandit data with known ground truth.
  Returns a `SimulatedChoiceData` NamedTuple with fields: `choices`, `true_values`,
  `true_probs` (consistent with `ChoiceFilterResult`/`ChoiceSmootherResult` pattern).
- `plot_values(observed_choices=None, option_labels=None)` — 2-panel:
  top = latent values with CI, bottom = choice probabilities (stacked area)
- `plot_convergence()` — EM log-likelihood trace
- `plot_summary(observed_choices=None)` — 3-panel diagnostic

### Step 4.3: Run all tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py -v
```

### Step 4.4: Final verification checkpoint

- [ ] ALL tests pass
- [ ] `ruff check` passes
- [ ] Run full integration test on simulated data:
  ```python
  from state_space_practice.multinomial_choice import (
      MultinomialChoiceModel, simulate_choice_data
  )
  choices, true_values, true_probs = simulate_choice_data(
      n_trials=200, n_options=4, process_noise=0.05, seed=42
  )
  model = MultinomialChoiceModel(n_options=4)
  model.fit(choices, verbose=True)
  print(model.summary(choices=choices))
  fig, axes = model.plot_summary(observed_choices=choices)
  ```
  Verify: smoothed values track true values (correlation > 0.7 per option),
  `compare_to_null()` detects learning, plots look correct.
- [ ] Run K=2 cross-validation against SmithLearningModel one more time
- [ ] Existing smith_learning_algorithm tests still pass:
  ```bash
  conda run -n state_space_practice pytest src/state_space_practice/tests/test_smith_learning_algorithm.py -q
  ```

### Step 4.5: Commit

```bash
git commit -m "Add simulation, plotting, and diagnostics to MultinomialChoiceModel"
```
