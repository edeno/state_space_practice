# Switching Choice Model — Task Breakdown

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Implement the switching choice model described in `docs/plans/2026-04-05-switching-choice-model.md`.

**Design doc:** `docs/plans/2026-04-05-switching-choice-model.md`

**Key files:**

- Create: `src/state_space_practice/switching_choice.py`
- Create: `src/state_space_practice/tests/test_switching_choice.py`
- Reference: `src/state_space_practice/switching_kalman.py` (GPB2 machinery)
- Reference: `src/state_space_practice/switching_point_process.py` (implementation pattern)
- Reference: `src/state_space_practice/multinomial_choice.py` (`_softmax_update_core`)
- Reference: `src/state_space_practice/covariate_choice.py` (covariate prediction, M-steps)

**Prerequisite Gates:**

- Covariate choice tests pass.
- Switching Kalman tests pass.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_switching_kalman.py -q
```

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_choice.py -v`
- Neighbor regression: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py src/state_space_practice/tests/test_multinomial_choice.py src/state_space_practice/tests/test_switching_kalman.py -q`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

**Critical design decisions:**

- Follow the `switching_point_process.py` architecture exactly. The per-state-pair
  update uses `_softmax_update_core` in place of `_point_process_laplace_update`.
  Everything else (mixture collapse, discrete state update, likelihood scaling,
  smoother) is reused from `switching_kalman.py`.
- Per-state scalar parameters: `β_s`, `Q_s`, `decay_s`. These are indexed by the
  next discrete state `j` in the pair-conditional update.
- Shared `B` and `Θ` across states (MVP scope). Per-state B and Θ are deferred.
- The `_softmax_update_core` already accepts `obs_offset` — pass per-state offsets
  (shared Θ @ z_t for all states in MVP).
- Use `jax.vmap` over (prev_state, next_state) pairs, same as
  `_point_process_update_per_discrete_state_pair`.
- JIT compilation time will be significant (S² × Newton steps unrolled). Start
  with S=2 to validate, test S=3 for publication use.

**MVP Scope Lock:**

- S=2 discrete states (exploit vs. explore).
- Per-state: β_s (scalar), Q_s (scalar), decay_s (scalar).
- Shared: B (input gain), Θ (obs weights), init_mean.
- EM learns: per-state Q, per-state β (weighted grid search), per-state decay,
  transition matrix T, shared B and Θ.
- Require: S=1 parity with CovariateChoiceModel, two-state synthetic recovery,
  improved LL over non-switching on switching data.

**Defer:**

- Per-state B and Θ.
- S > 3.
- Joint neural-behavioral switching.

---

## Task 1: Per-State-Pair Softmax Update

Build the core observation update that runs the softmax Laplace-EKF for each
(prev_state, next_state) pair, following `_point_process_update_per_discrete_state_pair`.

### Step 1.1: Write failing tests

```python
# tests/test_switching_choice.py

class TestSoftmaxUpdatePerStatePair:
    def test_output_shapes(self):
        # S=2 discrete states, K=3 options (2 free params)
        # prev_state_cond_mean: (2, 2) — 2 free params, 2 discrete states
        # prev_state_cond_cov: (2, 2, 2)
        # Returns: pair_cond_mean (2, 2, 2), pair_cond_cov (2, 2, 2, 2),
        #          pair_cond_ll (2, 2)

    def test_single_state_matches_softmax_update(self):
        # S=1 → output should match _softmax_update_core directly

    def test_different_states_give_different_posteriors(self):
        # With different β per state, posteriors should differ

    def test_log_likelihood_is_finite(self):
        # All S² log-likelihoods should be finite

    def test_covariance_is_psd(self):
        # All S² posterior covariances should be PSD
```

### Step 1.2: Implement

```python
# switching_choice.py

def _softmax_predict_and_update(
    prev_mean: Array,           # (K-1,) — previous state-conditional mean
    prev_cov: Array,            # (K-1, K-1)
    choice: Array,              # scalar int
    transition_matrix: Array,   # (K-1, K-1) — A_j = decay_j * I
    process_cov: Array,         # (K-1, K-1) — Q_j
    n_options: int,
    inverse_temperature: float, # β_j
    input_gain: Array,          # (K-1, d) — shared B
    covariates_t: Array,        # (d,)
    obs_offset: Array,          # (K,) — Θ @ z_t (shared)
) -> tuple[Array, Array, Array]:
    """Predict + softmax update for one (prev_state_i, next_state_j) pair."""
    # Predict
    pred_mean = transition_matrix @ prev_mean + input_gain @ covariates_t
    pred_cov = transition_matrix @ prev_cov @ transition_matrix.T + process_cov

    # Update
    post_mean, post_cov, ll = _softmax_update_core(
        pred_mean, pred_cov, choice, n_options, inverse_temperature,
        obs_offset=obs_offset,
    )
    return post_mean, post_cov, ll


def _softmax_update_per_state_pair(
    prev_state_cond_mean: Array,   # (K-1, S)
    prev_state_cond_cov: Array,    # (K-1, K-1, S)
    choice: Array,
    transition_matrices: Array,    # (K-1, K-1, S) — per-state A
    process_covs: Array,           # (K-1, K-1, S) — per-state Q
    n_options: int,
    inverse_temperatures: Array,   # (S,) — per-state β
    input_gain: Array,             # (K-1, d)
    covariates_t: Array,           # (d,)
    obs_offset: Array,             # (K,)
) -> tuple[Array, Array, Array]:
    """Double-vmapped update over all (i, j) state pairs.

    Returns pair-conditional posteriors:
        mean: (K-1, S_prev, S_next)
        cov: (K-1, K-1, S_prev, S_next)
        ll: (S_prev, S_next)
    """
    # Inner vmap over prev state i (axis -1 of mean/cov)
    # Outer vmap over next state j (axis -1 of A/Q, element of β)
    # Same pattern as switching_point_process.py
```

### Step 1.3: Run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_choice.py::TestSoftmaxUpdatePerStatePair -v
```

### Step 1.4: Commit

```bash
git commit -m "Add per-state-pair softmax update for switching choice model"
```

---

## Task 2: Switching Choice Filter with GPB2

### Step 2.1: Write failing tests

```python
class TestSwitchingChoiceFilter:
    def test_output_shapes(self):
        # 200 trials, K=3 options, S=2 states
        # filtered_values: (200, 2) — K-1 free params
        # filtered_covs: (200, 2, 2)
        # discrete_state_probs: (200, 2)
        # marginal_ll: scalar

    def test_single_state_matches_covariate_filter(self):
        # KEY PARITY: S=1 → output == covariate_choice_filter exactly
        # Same choices, same params → same filtered_values, same LL

    def test_discrete_state_probs_sum_to_one(self):
        # At every trial, sum over states == 1

    def test_discrete_state_probs_nonnegative(self):

    def test_marginal_ll_is_finite(self):

    def test_two_state_switching_detected(self):
        # First 100 trials: always choose option 1 (exploit-like)
        # Next 100 trials: random choices (explore-like)
        # At trial 50: p(exploit) > 0.7
        # At trial 150: p(explore) > 0.5
        # (With appropriate per-state β)

    def test_covariance_is_psd(self):
        # All filtered covariances should be PSD
```

### Step 2.2: Implement

```python
def switching_choice_filter(
    choices: ArrayLike,             # (T,) int
    n_options: int,
    n_discrete_states: int = 2,
    covariates: Optional[ArrayLike] = None,
    input_gain: Optional[ArrayLike] = None,
    obs_covariates: Optional[ArrayLike] = None,
    obs_weights: Optional[ArrayLike] = None,
    process_noises: Optional[ArrayLike] = None,     # (S,) per-state Q
    inverse_temperatures: Optional[ArrayLike] = None, # (S,) per-state β
    decays: Optional[ArrayLike] = None,              # (S,) per-state decay
    discrete_transition_matrix: Optional[ArrayLike] = None,  # (S, S)
    init_mean: Optional[ArrayLike] = None,
    init_cov: Optional[ArrayLike] = None,
    init_discrete_prob: Optional[ArrayLike] = None,
) -> SwitchingChoiceFilterResult:
```

Architecture (following `switching_point_process_filter`):
1. Initial timestep: per-state softmax update (vmap over states)
2. Scan over trials t=2..T:
   a. Per-state-pair predict + softmax update (double vmap)
   b. Scale likelihoods (`_scale_likelihood` from switching_kalman.py)
   c. Update discrete state probs (`_update_discrete_state_probabilities`)
   d. Collapse mixtures (`collapse_gaussian_mixture_per_discrete_state`)
3. Return filtered values, covs, discrete probs, marginal LL

Reuse from `switching_kalman.py`:
- `_scale_likelihood`
- `_update_discrete_state_probabilities`
- `collapse_gaussian_mixture_per_discrete_state`

### Step 2.3: Run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_choice.py::TestSwitchingChoiceFilter -v
```

### Step 2.4: Commit

```bash
git commit -m "Add switching choice filter with GPB2 approximation"
```

---

## Task 3: SwitchingChoiceModel Class with EM

### Step 3.1: Write failing tests

```python
class TestSwitchingChoiceModel:
    def test_init_and_repr(self):
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        assert "n_discrete_states=2" in repr(model)

    def test_fit_returns_log_likelihoods(self):
        # 200 trials, S=2 → list of finite floats

    def test_is_fitted(self):

    def test_single_state_parity(self):
        # KEY PARITY: S=1 SwitchingChoiceModel == CovariateChoiceModel
        # Same data, same params → same LL, same smoothed values

    def test_two_state_learns_different_betas(self):
        # Exploit phase (deterministic) + explore phase (random)
        # Per-state β should differ: β_exploit > β_explore

    def test_two_state_learns_different_Q(self):
        # Stable phase + volatile phase
        # Q_volatile > Q_stable

    def test_transition_matrix_learned(self):
        # Sticky transitions (high self-transition prob)
        # Diagonal of T should be > 0.5

    def test_discrete_state_posterior(self):
        # After fit, smoothed_discrete_probs shape (T, S), sum to 1

    def test_em_log_likelihood_non_decreasing(self):
        # Within tolerance for grid discretization

    def test_bic_comparison_to_non_switching(self):
        # On switching data: switching model BIC < non-switching BIC
        # On non-switching data: switching model BIC >= non-switching BIC

    def test_with_covariates(self):
        # Reward covariates + switching → should work

    def test_with_obs_covariates(self):
        # Stay bias + switching → should work

    def test_choice_probabilities_per_state(self):
        # Shape (T, K) using smoothed values and smoothed discrete probs
```

### Step 3.2: Implement

```python
class SwitchingChoiceModel:
    """Switching multi-armed bandit with per-state learning dynamics.

    Discrete latent states represent behavioral strategies (e.g.,
    exploit vs. explore) that control how option values evolve.

    Parameters
    ----------
    n_options : int
    n_discrete_states : int
    n_covariates : int
    n_obs_covariates : int
    ...per-state initial parameters...
    """
```

EM loop:
1. E-step: run switching filter + reuse `switching_kalman_smoother`
   (smoother is observation-model agnostic — same as for spike models)
2. M-step for per-state Q: weighted by smoothed discrete state probs
3. M-step for per-state decay: weighted regression
4. M-step for per-state β: per-state grid search (weight filter LL by state prob)
5. M-step for transition matrix: standard from joint smoothed probs
   (reuse `switching_kalman_maximization_step` for this part)
6. M-step for shared B: regression weighted by total prob (=1 per trial)
7. M-step for shared Θ: Newton weighted by total prob

### Step 3.3: Run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_choice.py::TestSwitchingChoiceModel -v
```

### Step 3.4: Verification checkpoint — CRITICAL

- [ ] All tests pass
- [ ] **S=1 parity:** identical LL and smoothed values to CovariateChoiceModel
- [ ] **Two-state recovery:** β_exploit > β_explore on synthetic switching data
- [ ] **BIC comparison:** switching model wins on switching data, loses on non-switching
- [ ] EM monotonic (within tolerance)
- [ ] `ruff check` passes
- [ ] Existing covariate_choice and switching_kalman tests still pass

### Step 3.5: Commit

```bash
git commit -m "Add SwitchingChoiceModel with per-state EM fitting"
```

---

## Task 4: Simulation, Diagnostics, and Model Comparison

### Step 4.1: Write failing tests

```python
class TestSimulateSwitchingChoiceData:
    def test_output_shapes(self):
        # choices (T,), true_values (T, K-1), true_states (T,), true_probs (T, K)

    def test_choices_valid(self):
    def test_states_valid(self):
    def test_seed_reproducibility(self):

    def test_state_transitions_match_matrix(self):
        # Empirical transition frequencies ≈ specified T matrix

class TestSwitchingChoiceModelPlotting:
    def test_plot_values_with_states(self):
        # Values colored/shaded by discrete state posterior

    def test_plot_discrete_states(self):
        # Posterior discrete state probabilities over trials

    def test_plot_convergence(self):
    def test_plot_summary(self):
        # 4-panel: values, states, convergence, probabilities

class TestModelComparison:
    def test_switching_beats_nonswitching_on_switching_data(self):
        # Simulate with known state switches
        # Fit switching model and non-switching model
        # Switching model has better LL and BIC

    def test_nonswitching_preferred_on_stationary_data(self):
        # Simulate without switches
        # Non-switching BIC should be better (fewer params)
```

### Step 4.2: Implement

```python
class SimulatedSwitchingChoiceData(NamedTuple):
    choices: Array
    true_values: Array
    true_states: Array
    true_probs: Array

def simulate_switching_choice_data(
    n_trials, n_options, n_discrete_states=2,
    process_noises=None, inverse_temperatures=None, decays=None,
    transition_matrix=None, input_gain=None, covariates=None,
    seed=42,
) -> SimulatedSwitchingChoiceData:
```

Plotting methods:
- `plot_values(observed_choices=None)` — values with state-shaded background
- `plot_discrete_states()` — posterior probabilities over trials
- `plot_convergence()` — EM LL trace
- `plot_summary()` — combined diagnostic

### Step 4.3: Run all tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_choice.py -v
```

### Step 4.4: Final verification

- [ ] ALL tests pass
- [ ] `ruff check` passes
- [ ] Full integration: simulate switching data → fit → recover states and params
- [ ] S=1 parity still holds
- [ ] Neighbor regression tests pass
- [ ] Model comparison: switching wins on switching data, loses on stationary data

### Step 4.5: Commit

```bash
git commit -m "Add switching choice simulation, plotting, and model comparison"
```
