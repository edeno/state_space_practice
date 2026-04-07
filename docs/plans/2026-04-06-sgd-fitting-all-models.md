# SGD Fitting for All Model Classes

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Add `fit_sgd()` to every model class in the library via a shared mixin, providing a gradient-based alternative to EM that eliminates grid search, supports constrained optimization, and enables differentiable regularization.

**Relationship to differentiable-em-optimization.md:** This plan is a follow-on to the differentiable EM optimization plan. It depends on the parameter transform infrastructure from Order 0.5 (partially implemented: `POSITIVE`, `UNIT_INTERVAL`, `PSD_MATRIX`, `UNCONSTRAINED` exist; `STOCHASTIC_ROW` and `trainable` flag are new work in Task 0). The differentiable-EM plan's phasing is respected: linear-Gaussian and choice models first (choice models done, linear-Gaussian baseline in differentiable-EM plan Task 2-3), then point-process only after gradient stability is verified. Oscillator and switching models are last, with explicit stability gates. `optax` must be added to `pyproject.toml` (currently only in `environment.yml`).

**Scientific motivation:** EM has closed-form M-steps for linear Gaussian models but requires grid search for inverse temperature (choice models) and post-hoc eigenvalue/stability projection (oscillator models). SGD eliminates grid search for beta via softplus-constrained gradient descent. For point-process models, SGD can unify dynamics and observation parameter learning into a single gradient step. For oscillator models, SGD over the model's scientific parameterization (frequency, damping, coupling) would replace the current project-and-hope pattern with proper constrained optimization.

**Design reference:** Dynamax (probml/dynamax) uses a base-class `fit_sgd` + model-specific `marginal_log_prob`, with TFP bijectors for constraints and `stop_gradient` for freezing parameters. We adopt the same architecture but with pure-JAX transforms (no TFP dependency) and a mixin class (since our models don't share a single base class).

## Architecture

```
SGDFittableMixin (sgd_fitting.py)
├── fit_sgd(data, ..., optimizer, num_steps, verbose)
│   ├── calls self._build_param_spec() → (params, spec)
│   ├── calls self._sgd_loss_fn(params, data) → scalar
│   ├── unconstrain → optimize → constrain
│   ├── calls self._store_sgd_params(params)
│   └── calls self._finalize_sgd(*args, **kwargs)  ← model-specific post-opt
│
parameter_transforms.py (extended)
├── POSITIVE (softplus) — existing
├── UNIT_INTERVAL (sigmoid) — existing
├── PSD_MATRIX (Cholesky log-diag) — existing
├── UNCONSTRAINED — existing
├── STOCHASTIC_ROW — NEW: softmax per row for transition matrices Z
└── transform_to_constrained() applies stop_gradient for frozen params
```

Each model implements:
- `_build_param_spec() → tuple[dict, dict]` — returns `(params, spec)` with current parameter values and their transforms. Must respect model-specific flags (e.g., `learn_process_noise`, `initial_state_method`).
- `_sgd_loss_fn(params: dict, *args) → Array` — returns **raw** negative marginal log-likelihood (NOT normalized). The mixin divides by `_n_timesteps` internally. Receives **constrained** parameters. Must not read mutable model attributes — all data must come through arguments (required for `lax.scan` JIT compatibility).
- `_store_sgd_params(params: dict) → None` — writes optimized parameters back to model attributes.
- `_finalize_sgd(*args, **kwargs) → None` — runs final smoother pass and populates fitted-state diagnostics (smoothed values, covariances, `log_likelihood_`, `is_fitted`, etc.).

**Tech Stack:** JAX, optax. `optax` must be added to `pyproject.toml` dependencies before Task 0 (currently in `environment.yml` but not `pyproject.toml`).

**Prerequisite Gates:**

- All existing tests pass before starting.
- `optax` declared in `pyproject.toml`.
- Existing `MultinomialChoiceModel.fit_sgd` and `CovariateChoiceModel.fit_sgd` pass as validation of the approach.

**Verification Gates (per task):**

- Targeted tests for new functionality pass.
- ALL existing tests still pass (strict regression).
- `conda run -n state_space_practice ruff check src/state_space_practice`

## Key Design Decisions

1. **Transition matrix A parameterization varies by model family.** Choice models and Smith: no A (random walk, A=I). PointProcessModel: UNCONSTRAINED A (following dynamax — initialize near identity, let data maintain stability). Oscillator models: SGD optimizes the model's own scientific parameterization (frequency via POSITIVE, damping via UNIT_INTERVAL for [0,1] constraint, coupling strengths via UNCONSTRAINED), NOT raw A. The transition matrix is reconstructed from these parameters inside `_sgd_loss_fn` via `oscillator_utils`, guaranteeing valid oscillator structure by construction. The existing `_project_parameters()` methods are not needed during SGD.

2. **Discrete transition matrix Z uses softmax rows.** `STOCHASTIC_ROW` transform applies softmax per row, mapping unconstrained reals to row-stochastic matrices. This is equivalent to dynamax's `SoftmaxCentered` bijector.

3. **`stop_gradient` for frozen parameters.** Instead of excluding non-learnable parameters from the dict, include all parameters and apply `jax.lax.stop_gradient()` during `transform_to_constrained()` for params marked as frozen. This keeps the PyTree structure constant.

4. **Loss normalized by sequence length.** The loss is `-marginal_log_likelihood / n_timesteps`, making learning rates transferable across datasets.

5. **Python loop for verbose, lax.scan for non-verbose.** Following dynamax's pattern: Python loop for outer training (allows logging), lax.scan for JIT-compiled fast path.

6. **`_finalize_sgd` hook for model-specific post-optimization.** The existing choice-model `fit_sgd` implementations run a final smoother pass and populate `_smoother_result`, `log_likelihood_`, and `is_fitted` after optimization. The mixin must call a model-specific finalization hook to preserve this contract.

7. **Initialization contract for oscillator and switching models.** The mixin does NOT handle parameter initialization. For models that require keyed initialization (oscillator family, switching point-process), callers must call `model.initialize_parameters(key=key, ...)` before `fit_sgd`. The mixin calls `self._check_sgd_initialized()` at the top of `fit_sgd`, which raises `RuntimeError` if parameters are not allocated. For choice models and SmithLearningModel, parameters are allocated at construction time, so no separate initialization is needed. This avoids duplicating the warm-start logic (spectral clustering, placeholder smoother stats, initial spike M-step) that exists in each model's `fit()` method — SGD only needs valid starting parameters, not the full EM warm-start procedure.

8. **Point-process gradient stability gate.** Before exposing SGD for point-process models, verify that gradients through the Laplace-EKF are finite on small synthetic problems. If gradients are unstable, defer that model and document the issue (following the differentiable-EM plan's guidance).

9. **SmithLearningModel respects initial_state_method.** The `_build_param_spec` must branch on `initial_state_method` to decide which initial-state parameters are learnable vs fixed.

---

## Phase 1: Infrastructure and Choice Model Refactor

### Task 0: Infrastructure — Parameter Transforms and Mixin

#### Step 0.0: Add optax to pyproject.toml

Add `optax` to the project dependencies in `pyproject.toml`, matching `environment.yml`.

#### Step 0.1: Add STOCHASTIC_ROW transform

Add to `src/state_space_practice/parameter_transforms.py`:

```python
def _stochastic_to_real(Z: Array) -> Array:
    """Row-stochastic matrix to unconstrained logits (drop last column)."""
    return jnp.log(Z[..., :-1]) - jnp.log(Z[..., -1:])

def _real_to_stochastic(logits: Array) -> Array:
    """Unconstrained logits to row-stochastic matrix via softmax."""
    full_logits = jnp.concatenate(
        [logits, jnp.zeros_like(logits[..., :1])], axis=-1
    )
    return jax.nn.softmax(full_logits, axis=-1)

STOCHASTIC_ROW = ParameterTransform(
    to_unconstrained=_stochastic_to_real,
    to_constrained=_real_to_stochastic,
)
```

**Numerical safety:** The inverse transform `_stochastic_to_real` uses `jnp.log(Z)`, which diverges for near-zero probabilities. Add a jitter: `jnp.log(jnp.maximum(Z[..., :-1], 1e-10)) - jnp.log(jnp.maximum(Z[..., -1:], 1e-10))`. Alternatively, initialize from logits directly rather than inverting a probability matrix.

#### Step 0.2: Add trainable flag and stop_gradient support

Extend `ParameterTransform` with a `trainable` field:

```python
class ParameterTransform(NamedTuple):
    to_unconstrained: Callable[[Array], Array]
    to_constrained: Callable[[Array], Array]
    trainable: bool = True
```

Update `transform_to_constrained` to apply `stop_gradient` for frozen params:

```python
def transform_to_constrained(unc_params: dict, spec: dict) -> dict:
    result = {}
    for k, v in unc_params.items():
        value = spec[k].to_constrained(v)
        if not spec[k].trainable:
            value = jax.lax.stop_gradient(value)
        result[k] = value
    return result
```

**Breaking change check:** Adding a third field to a `NamedTuple` with a default value requires care:

1. All existing constants (`POSITIVE`, `UNIT_INTERVAL`, `UNCONSTRAINED`, `PSD_MATRIX`) must be updated to include `trainable=True` explicitly: `POSITIVE = ParameterTransform(to_unconstrained=..., to_constrained=..., trainable=True)`.
2. New constants like `STOCHASTIC_ROW` must also include `trainable=True` explicitly — omitting it creates a 2-tuple that will fail.
3. Any external code using positional construction `ParameterTransform(fn1, fn2)` will still work (gets `trainable=True` default), but code that pattern-matches on `len(transform)` or `._fields` needs updating.
4. Task 0.4 tests must verify: (a) existing transforms still have 3 fields, (b) `trainable` defaults to `True`, (c) `ParameterTransform(fn1, fn2)` still works without explicit `trainable`.

#### Step 0.3: Create SGDFittableMixin

Create `src/state_space_practice/sgd_fitting.py`:

```python
import logging
from typing import Optional

import jax
from jax import Array

logger = logging.getLogger(__name__)


class SGDFittableMixin:
    """Mixin providing fit_sgd() for state-space models.

    Models must implement:
    - _build_param_spec() -> tuple[dict, dict]
    - _sgd_loss_fn(params, *args, **kwargs) -> Array
    - _store_sgd_params(params: dict) -> None
    - _finalize_sgd(*args, **kwargs) -> None
    - _check_sgd_initialized() -> None  (raise if params not allocated)
    - _n_timesteps: int (property or attribute)
    """

    def fit_sgd(
        self,
        *args,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        **kwargs,
    ) -> list[float]:
        """Fit by minimizing negative marginal LL via gradient descent.

        Parameters
        ----------
        *args, **kwargs
            Passed to _sgd_loss_fn and _finalize_sgd.
        optimizer : optax optimizer or None
            Default: adam(1e-2) with gradient clipping.
        num_steps : int
            Number of optimization steps.
        verbose : bool
            Log progress every 10 steps.

        Returns
        -------
        log_likelihoods : list of float
        """
        import optax
        from state_space_practice.parameter_transforms import (
            transform_to_constrained,
            transform_to_unconstrained,
        )

        self._check_sgd_initialized()
        params, param_spec = self._build_param_spec()

        if not param_spec:
            raise ValueError("No learnable parameters — nothing to optimize.")

        unc_params = transform_to_unconstrained(params, param_spec)
        n_timesteps = float(self._n_timesteps)

        @jax.value_and_grad
        def loss_fn(unc_p):
            p = transform_to_constrained(unc_p, param_spec)
            return self._sgd_loss_fn(p, *args, **kwargs) / n_timesteps

        if optimizer is None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(10.0),
                optax.adam(1e-2),
            )
        opt_state = optimizer.init(unc_params)

        def _train_step(carry, _):
            unc_p, o_state = carry
            loss, grads = loss_fn(unc_p)
            updates, new_o_state = optimizer.update(grads, o_state, unc_p)
            new_unc_p = optax.apply_updates(unc_p, updates)
            return (new_unc_p, new_o_state), loss

        if verbose:
            log_likelihoods: list[float] = []
            for step in range(num_steps):
                (unc_params, opt_state), loss = _train_step(
                    (unc_params, opt_state), None,
                )
                ll = -float(loss) * self._n_timesteps
                log_likelihoods.append(ll)
                if step % 10 == 0 or step == num_steps - 1:
                    logger.info("SGD step %d: LL=%.2f", step, ll)
        else:
            (unc_params, opt_state), losses = jax.lax.scan(
                _train_step, (unc_params, opt_state), None, length=num_steps,
            )
            log_likelihoods = (-losses * n_timesteps).tolist()

        final_params = transform_to_constrained(unc_params, param_spec)
        self._store_sgd_params(final_params)
        self.log_likelihood_history_ = log_likelihoods
        self._finalize_sgd(*args, **kwargs)

        return log_likelihoods
```

Note: `_sgd_loss_fn` returns the **raw** negative marginal LL (not normalized). The mixin divides by `n_timesteps`. This avoids each model having to know about normalization.

#### Step 0.4: Tests

Tests for STOCHASTIC_ROW, trainable flag, and a minimal mixin smoke test using a toy model.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_parameter_transforms.py -v
conda run -n state_space_practice pytest src/state_space_practice/tests/test_sgd_fitting.py -v
conda run -n state_space_practice ruff check src/state_space_practice
```

---

### Task 1: Refactor Choice Models to Use Mixin

#### Step 1.1: Migrate MultinomialChoiceModel

Add `SGDFittableMixin` to `MultinomialChoiceModel`. Implement:

- `_build_param_spec`: branches on `learn_process_noise` and `learn_inverse_temperature`
- `_sgd_loss_fn(params, choices)`: calls `_multinomial_choice_filter_jit`, returns `-marginal_log_likelihood`
- `_store_sgd_params`: writes `process_noise`, `inverse_temperature`
- `_finalize_sgd(choices)`: runs final smoother pass, sets `_smoother_result`, `log_likelihood_`, `is_fitted`
- `_n_timesteps` property: returns `self._n_trials`

Delete the existing `fit_sgd` body. The mixin's `fit_sgd` replaces it. Input validation (choice bounds, minimum trials) moves to a `_validate_sgd_inputs` called from `_sgd_loss_fn` or from an overridden `fit_sgd` that calls `super().fit_sgd(...)`.

#### Step 1.2: Migrate CovariateChoiceModel

Same pattern with additional params: `decay` (UNIT_INTERVAL), `input_gain` (UNCONSTRAINED), `obs_weights` (UNCONSTRAINED).

#### Step 1.3: Verify all existing SGD tests pass unchanged

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_multinomial_choice.py -v -k "sgd"
conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py -v -k "sgd"
```

**Stop condition:** If any existing SGD test fails, the mixin design needs revision before proceeding.

---

## Phase 2: Simple Models (Tier 1)

### Task 2: SmithLearningModel

#### What it learns
- `sigma_epsilon` (process noise scalar) — POSITIVE
- `init_learning_state` — UNCONSTRAINED (only when `initial_state_method != "set_initial_to_zero"`)
- `init_learning_variance` — POSITIVE (only when `initial_state_method != "set_initial_to_zero"`)

#### Step 2.1: Add mixin, implement four methods

**Critical:** `_build_param_spec` must branch on `self.initial_state_method` (4 modes in `smith_learning_algorithm.py`):
- `"set_initial_to_zero"`: only `sigma_epsilon` is learnable. Init state frozen at zero.
- `"user_provided"`: `sigma_epsilon` + `init_learning_state` (UNCONSTRAINED) + `init_learning_variance` (POSITIVE).
- `"set_initial_conservative_from_second_trial"`: only `sigma_epsilon` is learnable. Init state is computed from data each EM iteration — for SGD, freeze init params (matching the data-driven EM behavior, since SGD cannot replicate the data-dependent recomputation).
- `"set_initial_direct_from_second_trial"`: same as conservative — freeze init params.
- `"reestimate_initial_from_data"`: `sigma_epsilon` + `init_learning_state` + `init_learning_variance` (same as `"user_provided"` — EM re-estimates these from smoother stats).

`_sgd_loss_fn(params, choices, ...)`: calls the Smith learning filter, returns `-marginal_log_likelihood`.

`_finalize_sgd(choices, ...)`: runs final smoother pass.

#### Step 2.2: Tests

```python
class TestSmithSGDFitting:
    def test_sgd_improves_ll(self): ...
    def test_sgd_respects_constraints(self): ...
    def test_sgd_model_is_fitted(self): ...
    def test_sgd_matches_em_approximately(self): ...
    def test_sgd_respects_initial_state_method(self):
        """When initial_state_method='set_initial_to_zero', init params are frozen."""
```

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_smith_learning_algorithm.py -v -k "sgd"
```

---

### Task 3: PointProcessModel

#### What it learns
- `transition_matrix` (A) — UNCONSTRAINED (initialize near identity, let data maintain stability)
- `process_cov` (Q) — PSD_MATRIX
- `init_mean` — UNCONSTRAINED
- `init_cov` — PSD_MATRIX

#### Step 3.1: Gradient stability gate

**Before implementing fit_sgd**, verify that gradients through the Laplace-EKF are finite:

```python
def test_point_process_gradient_stability():
    """Gradients through Laplace-EKF must be finite for SGD to work."""
    # Create small synthetic problem (4 latent, 10 neurons, 100 timesteps)
    # Compute grad of -marginal_ll w.r.t. A, Q
    # Assert all gradients are finite and reasonably bounded
```

**Stop condition:** If gradients are NaN or > 1e6, defer this task and document the issue.

#### Step 3.2: Add mixin, implement four methods

`_sgd_loss_fn(params, spikes, dt, log_intensity_func)`: calls point-process filter.

#### Step 3.3: Tests

```python
class TestPointProcessSGDFitting:
    def test_sgd_improves_ll(self): ...
    def test_sgd_respects_constraints(self): ...
    def test_sgd_process_cov_psd(self): ...
    def test_sgd_matches_em_approximately(self): ...
```

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_point_process_kalman.py -v -k "sgd"
```

---

### Task 4: PlaceFieldModel

#### What it learns

Learnable parameters depend on `update_*` flags:

- `process_cov` (Q diagonal) — POSITIVE (per-basis-function variance). Always learned.
- `init_mean` — UNCONSTRAINED. Always learned.
- `init_cov` — PSD_MATRIX. Always learned.
- `transition_matrix` (A) — UNCONSTRAINED. Only learned when `self.update_transition_matrix=True` (default `False`). When frozen, A stays at its initialized value.

`_build_param_spec` must check `self.update_transition_matrix` and conditionally include A.

#### Step 4.1: Add mixin, implement four methods

Similar to PointProcessModel but with spatial basis functions for the observation model.

#### Step 4.2: Tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_place_field_model.py -v -k "sgd"
```

---

## Phase 3: Switching Linear Gaussian Models (Tier 2)

### Task 5: BaseModel (Oscillator ABC) + CommonOscillatorModel

#### Design principle: SGD optimizes the existing scientific parameterization

SGD is an alternative optimizer for the parameters each class already exposes — not a new parameter API. The transition matrix A is never optimized directly; it is reconstructed from scientific parameters inside `_sgd_loss_fn`. The existing `_project_parameters()` methods are NOT called during SGD because the transforms enforce validity by construction.

#### CommonOscillatorModel: what it actually learns

In COM, A, Q, and R are **fixed** (constant across states). The primary learned parameter is the **measurement matrix H**, which varies per discrete state — it determines how oscillators contribute to observed signals. The EM M-step updates H via `switching_kalman_maximization_step()` with `update_measurement_matrix=True` and `update_continuous_transition_matrix=False`.

Learnable parameters for SGD (matching EM's `update_*` flags):

- `measurement_matrix` (H) — shape `(n_sources, 2*n_oscillators, n_discrete_states)`, UNCONSTRAINED. **This is the main scientifically relevant parameter** (`update_measurement_matrix=True`).
- `measurement_cov` (R) — shape `(n_sources, n_sources, n_discrete_states)`, PSD_MATRIX per state. EM updates R by default (`update_measurement_cov` is not set to `False` in COM, so `BaseModel._m_step()` updates it via `switching_kalman_maximization_step`).
- `discrete_transition_matrix` (Z) — shape `(n_discrete_states, n_discrete_states)`, STOCHASTIC_ROW.
- `init_mean` (per state) — UNCONSTRAINED.
- `init_cov` (per state) — PSD_MATRIX.
- `init_prob` — shape `(n_discrete_states,)`, STOCHASTIC_ROW (1 row).

Fixed parameters (frozen — matching EM flags):

- `continuous_transition_matrix` (A) — fixed (`update_continuous_transition_matrix=False`). Constructed at initialization from `freqs` and `auto_regressive_coef`.
- `process_cov` (Q) — fixed (`update_process_cov=False`). Constructed from `process_variance`.
- `freqs`, `auto_regressive_coef`, `process_variance`, `measurement_variance` — scientific initialization parameters used to construct A, Q, R. Not directly optimized.

Note: The `_build_param_spec` must respect the `update_*` flags on `BaseModel`. If a caller sets `update_continuous_transition_matrix=True` or `update_process_cov=True`, those parameters should become learnable (UNCONSTRAINED for A, PSD_MATRIX for Q). The default COM behavior freezes A and Q.

#### Step 5.1: Add mixin to BaseModel

```python
class BaseModel(ABC, SGDFittableMixin):
    ...
```

BaseModel provides:
- `_check_sgd_initialized()`: verifies that `initialize_parameters(key=...)` has been called (checks that `continuous_transition_matrix`, `measurement_matrix`, etc. are allocated). Raises `RuntimeError("Call model.initialize_parameters(key=...) before fit_sgd()")` if not.
- `_finalize_sgd`: runs the switching Kalman smoother to populate fitted-state diagnostics.

Each subclass implements `_build_param_spec` and `_sgd_loss_fn` with its own learned parameters.

**Usage pattern for oscillator models:**
```python
model = CommonOscillatorModel(n_oscillators=2, n_discrete_states=3, ...)
model.initialize_parameters(key=jax.random.PRNGKey(0), obs=obs)
model.fit_sgd(obs, num_steps=200)
```

This mirrors the existing EM pattern where `fit()` calls `initialize_parameters()` internally. For SGD, initialization is explicit because SGD does not need the full EM warm-start procedure (spectral clustering, etc.) — it only needs valid starting parameters.

#### Step 5.2: Implement on CommonOscillatorModel first

`_sgd_loss_fn` must:
1. Receive constrained parameters (primarily H, plus Z, init)
2. Use the already-initialized A, Q, R (fixed at construction time from `freqs`, `auto_regressive_coef`, `process_variance`, `measurement_variance`)
3. Call the switching Kalman filter with the learned H and fixed A, Q, R
4. Return `-marginal_log_likelihood`

This mirrors the existing EM behavior where only H and Z are updated.

#### Step 5.3: Per-state parameter handling

Parameters stored as `(D, D, n_states)` need transforms applied per-state. Transpose to `(n_states, D, D)` and apply PSD_MATRIX via `jax.vmap` for init_cov.

#### Step 5.3: Tests

```python
class TestOscillatorSGDFitting:
    def test_sgd_improves_ll(self): ...
    def test_sgd_process_cov_psd(self): ...
    def test_sgd_discrete_transitions_stochastic(self): ...
    def test_sgd_measurement_cov_psd(self): ...
```

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_models.py -v -k "sgd"
```

---

### Task 6: Remaining Oscillator Subclasses

#### Step 6.1: CorrelatedNoiseModel

Override `_build_param_spec` to expose:

- `freqs` — POSITIVE (shared with CommonOscillator)
- `auto_regressive_coef` — UNIT_INTERVAL (shared)
- `process_variance` — shape `(n_oscillators, n_discrete_states)`, POSITIVE. Per-oscillator, per-state.
- `measurement_variance` — scalar, POSITIVE
- `phase_difference` — shape `(n_oscillators, n_oscillators, n_discrete_states)`, UNCONSTRAINED. Cross-oscillator noise phase.
- `coupling_strength` — shape `(n_oscillators, n_oscillators, n_discrete_states)`, UNCONSTRAINED. Cross-oscillator noise coupling.
- Z, init_mean, init_cov, init_prob — same transforms as CommonOscillator

`_sgd_loss_fn` reconstructs the structured Q matrix from `process_variance`, `phase_difference`, and `coupling_strength` before calling the filter.

#### Step 6.2: DirectedInfluenceModel

DIM learns A (varies per state via coupling) and R, but NOT Q. Override `_build_param_spec`:

Learnable (matching EM flags `update_continuous_transition_matrix=True`, `update_measurement_cov=True`):

- `phase_difference` — shape `(n_oscillators, n_oscillators, n_discrete_states)`, UNCONSTRAINED. Per-state coupling phase in A.
- `coupling_strength` — shape `(n_oscillators, n_oscillators, n_discrete_states)`, UNCONSTRAINED. Per-state coupling strength in A.
- `measurement_cov` (R) — PSD_MATRIX per state (or frozen if `update_measurement_cov=False`).
- Z, init_mean, init_cov, init_prob — same transforms as COM.

Fixed (matching EM flags `update_process_cov=False`):

- `freqs` — fixed at initialization. Used to construct A but not optimized.
- `auto_regressive_coef` — fixed. UNIT_INTERVAL domain.
- `process_variance` — fixed. Q is constant across states and never updated in EM for DIM.

Note: `phase_difference` entries are rotation angles — mathematically unconstrained. PSD-ness of the reconstructed Q (for CNM) or A structure (for DIM) is guaranteed by the construction functions in `oscillator_utils.py`, not by the transform.

`_sgd_loss_fn` reconstructs per-state A from `freqs`, `auto_regressive_coef`, `phase_difference`, and `coupling_strength` via `oscillator_utils`, guaranteeing valid oscillator structure.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_models.py -v -k "sgd"
```

---

## Phase 4: Switching Point-Process Models (Tier 3)

### Task 7: Gradient Stability Gate for Switching Point-Process

**Before implementing any switching point-process SGD**, verify gradient stability:

```python
def test_switching_point_process_gradient_stability():
    """Gradients through switching Laplace-EKF must be finite."""
    # Small problem: 2 discrete states, 4 latent, 10 neurons, 100 timesteps
    # Compute grad of -marginal_ll w.r.t. A, Q, Z, spike params
    # Assert all gradients finite
```

The existing codebase uses trust-region blending, Armijo line search, eigenvalue floors, and best-parameter rollback in the EM loop for these models. These safeguards exist because the optimization landscape is difficult. SGD must demonstrate basic gradient stability before we proceed.

**Stop condition:** If gradients are NaN or highly sensitive, defer Tasks 7-9 and document the blocker. Consider a generalized-EM approach (differentiable M-step penalties as in differentiable-em-optimization.md Task 5) as an alternative.

---

### Task 8: BaseSwitchingPointProcessModel + Subclasses

#### What it learns (in addition to oscillator params)
- `baseline_` — spike GLM baseline (UNCONSTRAINED)
- `spike_weights_` — spike GLM weights (UNCONSTRAINED)

#### Step 8.1: Add mixin to BaseSwitchingPointProcessModel ABC

```python
class BaseSwitchingPointProcessModel(ABC, SGDFittableMixin):
    ...
```

`_check_sgd_initialized()` must verify that parameters are allocated AND that the spike GLM parameters (`self.spike_params.baseline`, `self.spike_params.weights` — stored as a `SpikeObsParams` struct, not as `baseline_`/`spike_weights_` attributes) are initialized. The existing `fit()` path runs warm initialization (spectral clustering of discrete states, placeholder smoother statistics, initial spike M-step) before the main EM loop. For SGD, only parameter allocation is required — the warm-start heuristics are EM-specific. Callers must call `model.initialize_parameters(key=..., obs=...)` before `fit_sgd()`.

The `_build_param_spec` for spike GLM parameters should use keys like `"spike_baseline"` and `"spike_weights"` in the param dict, and `_store_sgd_params` must write back to `self.spike_params.baseline` and `self.spike_params.weights` (or reconstruct a new `SpikeObsParams`).

#### Step 8.2: Implement on CommonOscillatorPointProcessModel first

Proof of concept on simplest subclass.

#### Step 8.3: Extend to CorrelatedNoise and DirectedInfluence subclasses

Override `_build_param_spec` for structured parameters.

#### Step 8.4: Tests

```python
class TestSwitchingPointProcessSGDFitting:
    def test_sgd_improves_ll(self): ...
    def test_sgd_spike_params_finite(self):
        """Verify spike_params.baseline and spike_params.weights are finite."""
    def test_sgd_discrete_transitions_stochastic(self): ...
    def test_sgd_process_cov_psd(self): ...
```

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_point_process_models.py -v -k "sgd"
```

---

### Task 9: SwitchingSpikeOscillatorModel

Standalone class (not inheriting from BaseModel or BaseSwitchingPointProcessModel). Largest parameter set.

#### Key design issue: no scientific parameterization for A

Unlike the oscillator hierarchy classes, `SwitchingSpikeOscillatorModel` does NOT expose `freqs`, `auto_regressive_coef`, `coupling_strength`, or `phase_difference` as public parameters. It owns `continuous_transition_matrix` directly as a `(2*n_osc, 2*n_osc, n_states)` array, initialized internally from hardcoded defaults (`_initialize_continuous_transition_matrix` in `switching_point_process.py`).

**Decision:** SGD for this model optimizes the raw matrices directly, matching the model's existing public API. This means:

- `continuous_transition_matrix` — UNCONSTRAINED (per state). No oscillator structure is enforced during SGD. This is a tradeoff: it gives the optimizer full freedom but may produce non-oscillatory solutions. If structure preservation is needed, a future extension could add a `use_scientific_parameterization` flag that introduces `freqs`/`damping` as learnable parameters and reconstructs A from them (similar to the oscillator hierarchy).
- `process_cov` — PSD_MATRIX (per state).
- `discrete_transition_matrix` (Z) — STOCHASTIC_ROW.
- `baseline_`, `spike_weights_` — UNCONSTRAINED (spike GLM params, optionally per state).
- `init_mean`, `init_cov` — per state.

The `update_*` flags (`update_continuous_transition_matrix`, `update_process_cov`, etc.) should map to the `trainable` field in the param spec: when a flag is False, the parameter is included but frozen via `stop_gradient`.

#### Step 9.1: Add mixin, implement four methods

Respect existing `update_*` flags when building the param spec.

#### Step 9.2: Tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_point_process.py -v -k "sgd"
```

---

## Implementation Schedule

| Phase | Tasks | Models | Risk | Duration |
|-------|-------|--------|------|----------|
| 1 | 0-1 | Infrastructure + choice model refactor | Low | 1 week |
| 2 | 2-4 | SmithLearning, PointProcess, PlaceField | Low-Med | 1-2 weeks |
| 3 | 5-6 | BaseModel + oscillator subclasses | Medium | 1-2 weeks |
| 4 | 7-9 | Switching point-process (gated) | Med-High | 1-2 weeks |

## Stop Conditions

- **Gradient stability gate (Tasks 3, 7):** If Laplace-EKF gradients are NaN or > 1e6 on synthetic data, defer that model family and document the blocker. Consider generalized-EM (differentiable M-step penalties) as fallback.
- **Mixin regression (Task 1):** If any existing SGD test fails after refactoring to the mixin, revise the mixin design before proceeding to new models.
- **SGD quality (all):** If SGD consistently converges to optima > 10% worse than EM for a model class, investigate before proceeding.
- **Import cycles:** If the mixin creates circular imports, restructure to a standalone function approach.

## Completion Definition

A task is complete when:
1. `fit_sgd()` works on the model class.
2. SGD improves LL from initial parameters.
3. All parameter constraints are satisfied after optimization.
4. Model-specific post-optimization state is populated (`is_fitted`, `smoothed_values`, etc.).
5. Existing tests still pass (strict regression).
6. Ruff passes.

## Deferred

- Custom VJP for Kalman smoother or Laplace-EKF (only if gradient stability gate fails)
- Minibatched multi-session training
- Neural observation/transition components
- Automatic backend selection (SGD vs EM based on model structure)
