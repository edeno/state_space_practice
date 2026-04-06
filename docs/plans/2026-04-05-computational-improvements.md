# Computational and Numerical Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Add parallel inference, SGD fitting, Woodbury-optimized updates, and numerical stability improvements to the shared infrastructure. These are model-agnostic improvements that benefit every current and future model in the library.

**Scientific motivation:** The current codebase works correctly but leaves performance and stability on the table. Long sessions (~500K time bins at 4ms) are slow with sequential scan on GPU. Many-neuron models (50+ neurons) waste computation on large matrix inversions that the Woodbury identity avoids. EM is the only fitting option, but SGD via gradient descent on the marginal log-likelihood would eliminate the β grid search hack and enable fitting models where the M-step has no closed form. These improvements are prerequisite infrastructure for scaling to real CA1/mPFC data.

**Architecture:** All improvements are additive — existing code continues to work unchanged. New capabilities are exposed via:
1. Alternative filter/smoother backends (parallel, information form)
2. A `fit_sgd` method alongside existing `fit_em`
3. An optimized observation update for diagonal-covariance observations
4. A lightweight parameter constraint system for unconstrained optimization

**Tech Stack:** JAX (`jax.lax.associative_scan` for parallel inference, `optax` for SGD), existing `kalman.py` infrastructure. No TFP or dynamax dependency.

**Prerequisite Gates:**

- All existing tests must pass before starting (this plan touches shared infrastructure).
- Each improvement is independently testable — no ordering dependency between tasks.

**Verification Gates:**

- All existing tests must continue to pass after each task (strict regression).
- New tests verify numerical equivalence: parallel filter output == sequential filter output.
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

**Feasibility Status:** READY (well-understood algorithms, reference implementations in dynamax)

**Codebase Reality Check:**

- `kalman.py` (555 lines): filter, smoother, M-step — gets parallel backend and info form
- `point_process_kalman.py` (1172 lines): Laplace-EKF — gets Woodbury optimization
- `position_decoder.py` (871 lines): bilinear interpolation + Laplace-EKF — gets Woodbury
- `multinomial_choice.py`, `covariate_choice.py`: model classes — get `fit_sgd`
- All model classes: get parameter constraint system for SGD

**Claude Code Execution Notes:**

- Each task is independent. Implement in any order based on what's most needed.
- The parallel smoother is the highest-impact single item for GPU workloads.
- The SGD fitting requires the parameter constraint system (Task 4 before Task 5).
- Test numerical equivalence with tight tolerances (atol=1e-5) — these should be algebraically identical, not just "close."

**MVP Scope Lock (implement now):**

- Tasks 1-5 as described below.
- Parallel inference for linear Gaussian only (nonlinear filter must remain sequential).
- SGD for `CovariateChoiceModel` as proof of concept.
- Woodbury for `_point_process_laplace_update` where n_neurons > state_dim.

**Defer Until Post-MVP:**

- Parallel EKF (requires custom associative scan elements for nonlinear observations).
- Automatic backend selection (CPU → sequential, GPU → parallel).
- Second-order SGD (natural gradient, Fisher preconditioning).
- Distributed multi-GPU training.

**References:**

- Särkkä, S. & García-Fernández, Á.F. (2021). Temporal parallelization of Bayesian smoothers. IEEE Transactions on Automatic Control 66(1), 299-306.
- Rauch, H.E., Tung, F. & Striebel, C.T. (1965). Maximum likelihood estimates of linear dynamic systems. AIAA Journal 3(8), 1445-1450.
- Kingma, D.P. & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.

---

## Task 1: Parallel Kalman Smoother via Associative Scan

The RTS smoother backward pass can be parallelized using `jax.lax.associative_scan` with `reverse=True`. This gives O(log T) parallel depth on GPU instead of O(T) sequential depth. The forward filter for nonlinear observations (Laplace-EKF) must remain sequential, but the backward smoother operates on pre-computed filtered Gaussians and is always linear.

**Files:**
- Modify: `src/state_space_practice/kalman.py`
- Test: `src/state_space_practice/tests/test_kalman.py`

### What to implement

```python
def parallel_kalman_smoother(
    filtered_means: Array,      # (T, D)
    filtered_covariances: Array, # (T, D, D)
    transition_matrix: Array,    # (D, D) or (T-1, D, D)
    process_cov: Array,          # (D, D) or (T-1, D, D)
) -> tuple[Array, Array, Array]:
    """RTS smoother via parallel associative scan.

    Algebraically equivalent to the sequential smoother but runs
    in O(log T) parallel depth on GPU/TPU via jax.lax.associative_scan.

    Uses the formulation from Särkkä & García-Fernández (2021):
    each smoother element is (E, g, L) where the smoother backward
    pass composes these elements associatively.
    """
```

The associative scan elements for the RTS smoother are:
```
E_t = J_t  (smoother gain)
g_t = m_{t|t} - J_t @ m_{t+1|t}  (bias)
L_t = P_{t|t} - J_t @ P_{t+1|t} @ J_t.T  (residual covariance)

Composition: (E1, g1, L1) ∘ (E2, g2, L2) =
    (E1 @ E2, E1 @ g2 + g1, E1 @ L2 @ E1.T + L1)

Smoothed mean: m_{t|T} = E_t @ m_{t+1|T} + g_t
Smoothed cov: P_{t|T} = E_t @ P_{t+1|T} @ E_t.T + L_t
```

### Tests

- `test_parallel_smoother_matches_sequential`: same output (atol=1e-5) on synthetic data
- `test_parallel_smoother_shapes`: correct output shapes
- `test_parallel_smoother_cross_cov`: smoother cross-covariance matches sequential
- `test_parallel_smoother_time_varying_params`: works with (T-1, D, D) transition matrices

### Verification

- [ ] Sequential and parallel smoother agree to atol=1e-5 on 1000-step synthetic data
- [ ] All existing kalman.py tests still pass
- [ ] `ruff check` passes

---

## Task 2: Woodbury-Optimized Observation Update

When the observation noise is diagonal (independent neurons), the standard Kalman gain computation is O(D_obs³) where D_obs is the number of neurons. The Woodbury identity reduces this to O(D_state³), which is much smaller when you have many neurons and a low-dimensional latent state.

**Files:**
- Modify: `src/state_space_practice/kalman.py` (add `woodbury_kalman_gain`)
- Modify: `src/state_space_practice/point_process_kalman.py` (use Woodbury when beneficial)
- Test: add to existing test files

### What to implement

```python
def woodbury_kalman_gain(
    prior_cov: Array,          # (D_state, D_state)
    emission_matrix: Array,    # (D_obs, D_state)
    emission_cov_diag: Array,  # (D_obs,) — diagonal only
) -> tuple[Array, Array]:
    """Compute Kalman gain using Woodbury identity for diagonal R.

    Standard:  K = P H' (H P H' + R)^{-1}     O(D_obs^3)
    Woodbury:  K = P H' [R^{-1} - R^{-1} U (I + U' R^{-1} U)^{-1} U' R^{-1}]
               where U = H @ chol(P)            O(D_state^3)

    Returns K and the innovation covariance S for log-likelihood.
    """
```

### When to use

The Woodbury optimization is beneficial when `n_neurons > 2 * state_dim`. For the position decoder (4-dim state, typically 20+ neurons), this is always the case. For the choice model (K-1 dim state, 1 observation), it's never needed.

Add a helper that selects the faster path:
```python
def _compute_kalman_gain(P, H, R):
    """Auto-select standard or Woodbury based on dimensions."""
    if R.ndim == 1 and R.shape[0] > 2 * P.shape[0]:
        return woodbury_kalman_gain(P, H, R)
    else:
        return standard_kalman_gain(P, H, R)
```

### Tests

- `test_woodbury_matches_standard`: same Kalman gain (atol=1e-6) for various dimension ratios
- `test_woodbury_many_neurons`: correct result with 100 neurons, 4-dim state
- `test_woodbury_log_likelihood`: innovation log-likelihood matches standard

---

## Task 3: Joseph Form Covariance Update

The standard covariance update `P_post = P - K S K'` can lose PSD-ness due to floating-point arithmetic. The Joseph form is algebraically equivalent but maintains PSD-ness by construction:

```
P_post = (I - K H) P (I - K H)' + K R K'
```

This is a sum of PSD terms, so the result is always PSD regardless of rounding.

**Files:**
- Modify: `src/state_space_practice/kalman.py` (add option to `_kalman_update`)
- Test: `src/state_space_practice/tests/test_kalman.py`

### What to implement

Add a `use_joseph_form: bool = False` parameter to the Kalman update. When True, use the Joseph form. Default False for backward compatibility and performance (the Joseph form requires one extra matrix multiply).

The switching Kalman filter (`switching_kalman.py`) would benefit most, since it accumulates numerical errors through the GPB2 mixture collapse.

### Tests

- `test_joseph_form_matches_standard`: same result on well-conditioned problems
- `test_joseph_form_psd_ill_conditioned`: PSD is maintained when standard form fails
- `test_joseph_form_switching`: stability improvement in GPB2 filter

---

## Task 4: Lightweight Parameter Constraint System

For SGD fitting, parameters must be optimized in unconstrained space and transformed back. This requires a mapping between constrained and unconstrained representations.

**Files:**
- Create: `src/state_space_practice/parameter_transforms.py`
- Test: `src/state_space_practice/tests/test_parameter_transforms.py`

### What to implement

```python
from typing import NamedTuple

class ParameterTransform(NamedTuple):
    """Maps between constrained and unconstrained parameter spaces."""
    to_unconstrained: Callable[[Array], Array]
    to_constrained: Callable[[Array], Array]

# Built-in transforms
POSITIVE = ParameterTransform(
    to_unconstrained=jnp.log,           # x > 0 → log(x) ∈ R
    to_constrained=jnp.exp,
)

UNIT_INTERVAL = ParameterTransform(
    to_unconstrained=lambda x: jnp.log(x / (1 - x)),  # logit
    to_constrained=jax.nn.sigmoid,
)

UNCONSTRAINED = ParameterTransform(
    to_unconstrained=lambda x: x,
    to_constrained=lambda x: x,
)

PSD_MATRIX = ParameterTransform(
    to_unconstrained=_psd_to_real,      # cholesky → log diag → flatten
    to_constrained=_real_to_psd,
)
```

No TFP dependency — pure JAX implementations.

### Parameter specs for our models

```python
COVARIATE_CHOICE_PARAMS = {
    "process_noise": POSITIVE,
    "inverse_temperature": POSITIVE,
    "decay": UNIT_INTERVAL,
    "input_gain": UNCONSTRAINED,
    "obs_weights": UNCONSTRAINED,
}
```

### Tests

- `test_positive_roundtrip`: constrain → unconstrain → constrain = identity
- `test_unit_interval_bounds`: sigmoid output always in (0, 1)
- `test_psd_roundtrip`: PSD matrix survives round-trip through Cholesky parameterization
- `test_gradient_flows`: gradients through transforms are finite and nonzero

---

## Task 5: SGD Fitting via Optax

Add `fit_sgd` as an alternative to `fit_em` for any model class. SGD optimizes the negative marginal log-likelihood directly using gradient descent, eliminating grid search for β and handling models where M-steps have no closed form.

**Files:**
- Modify: `src/state_space_practice/covariate_choice.py` (add `fit_sgd`)
- Modify: `src/state_space_practice/multinomial_choice.py` (add `fit_sgd`)
- Reference: `src/state_space_practice/parameter_transforms.py` (from Task 4)
- Test: add to existing test files

### What to implement

```python
def fit_sgd(
    self,
    choices: ArrayLike,
    covariates: Optional[ArrayLike] = None,
    obs_covariates: Optional[ArrayLike] = None,
    optimizer: Optional[Any] = None,  # optax optimizer
    num_steps: int = 200,
    verbose: bool = False,
) -> list[float]:
    """Fit the model by minimizing negative marginal log-likelihood via SGD.

    Uses optax for optimization. Parameters are transformed to
    unconstrained space, optimized, and transformed back.

    Advantages over fit_em:
    - No grid search for beta (gradient-based)
    - Handles models without closed-form M-steps
    - Can use momentum, learning rate schedules, etc.

    Disadvantages vs fit_em:
    - May find local optima (EM has monotonic LL guarantee)
    - Requires tuning learning rate
    - No smoothed estimates during fitting (only at convergence)
    """
```

The loss function:
```python
def _loss(unc_params, choices, covariates, obs_covariates):
    params = transform_to_constrained(unc_params, param_spec)
    result = covariate_choice_filter(
        choices, n_options, covariates, params.input_gain,
        obs_covariates, params.obs_weights,
        params.process_noise, params.inverse_temperature,
        params.decay,
    )
    return -result.marginal_log_likelihood
```

Default optimizer: `optax.adam(1e-2)` with gradient clipping.

### Tests

- `test_sgd_improves_ll`: LL improves from initial params
- `test_sgd_matches_em`: SGD and EM converge to similar parameters on synthetic data (loose tolerance)
- `test_sgd_learns_beta_without_grid`: β is learned smoothly, no grid artifacts
- `test_sgd_respects_constraints`: process_noise > 0, 0 < decay < 1 after fitting

### Key consideration

The marginal log-likelihood must be differentiable w.r.t. all parameters. This works because:
- The filter is a `jax.lax.scan` over differentiable operations
- The softmax update uses `jax.nn.softmax` and `psd_solve` which are differentiable
- The only non-differentiable operation was the Python-level β grid search, which SGD eliminates

---

## Task 6: Information Form Filter (Optional)

The information form Kalman filter represents the state using precision-weighted mean (η = Λ @ m) and precision matrix (Λ = P⁻¹) instead of mean and covariance. The update step becomes additive (no matrix inverse), which is more numerically stable for:
- Very uncertain initial states (large P → small Λ, well-conditioned)
- Very informative observations (large Fisher information, directly added to Λ)
- Switching models where mixture collapse can create ill-conditioned covariances

**Files:**
- Create: `src/state_space_practice/info_kalman.py`
- Test: `src/state_space_practice/tests/test_info_kalman.py`

### What to implement

```python
def info_kalman_filter(
    init_eta: Array,           # Λ @ m (precision-weighted mean)
    init_precision: Array,     # Λ = P^{-1}
    obs: Array,
    transition_matrix: Array,
    process_precision: Array,  # Q^{-1}
    measurement_matrix: Array,
    measurement_precision: Array,  # R^{-1}
) -> InfoFilterResult:
    """Kalman filter in information form.

    Update step (no matrix inverse needed):
        Λ_post = Λ_prior + H' R^{-1} H
        η_post = η_prior + H' R^{-1} y

    Prediction step (requires one inverse):
        Λ_pred = (F Λ^{-1} F' + Q)^{-1}
        η_pred = Λ_pred F Λ^{-1} η

    Conversion to moment form for output:
        m = Λ^{-1} η
        P = Λ^{-1}
    """
```

### Tests

- `test_info_form_matches_moment_form`: same filtered means/covs (atol=1e-5)
- `test_info_form_large_initial_uncertainty`: stable when P_0 = 1e6 * I
- `test_info_form_high_snr`: stable when observation is very informative

### Note

This is marked optional because it's most valuable for the switching models (Task 3 of Joseph form helps more immediately). Implement if switching model stability is still a concern after Joseph form is added.
