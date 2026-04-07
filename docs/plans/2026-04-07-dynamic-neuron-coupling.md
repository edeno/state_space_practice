# Dynamic Neuron-to-Neuron Spike Coupling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Build a non-switching dynamic spike-history model that tracks neuron-to-neuron coupling weights over time using the existing point-process state-space machinery.

**Architecture:** Reuse the Laplace-EKF smoother in `point_process_kalman.py`, but reinterpret the latent state as a concatenated vector of time-varying spike-history GLM coefficients. Each neuron's log intensity depends on lagged spikes from all source neurons through a block-structured design matrix. Start without switching states, without latent common-drive factors, and without area penalties. Those become follow-on extensions after the base model is stable.

**Tech Stack:** JAX, `point_process_kalman.py`, `PointProcessModel` patterns, new feature builders in `src/state_space_practice/spike_history_features.py`, new model wrapper in `src/state_space_practice/dynamic_spike_coupling.py`.

**Fitting paths:** The model supports two fitting approaches:

1. **Iterated Laplace smoothing** (`fit()`): E-step runs point-process smoother, M-step updates Q only. This is the default.
2. **SGD via mixin** (`fit_sgd()`): Inherits from `SGDFittableMixin` (from `docs/plans/2026-04-06-sgd-fitting-all-models.md`). Implements `_build_param_spec` (Q as PSD_MATRIX, optionally A as UNCONSTRAINED), `_sgd_loss_fn` (negative marginal LL from point-process filter), `_store_sgd_params`, `_finalize_sgd`. This requires the SGD plan's Task 3 (PointProcessModel gradient stability gate) to pass first.

---

**Prerequisite Gates:**

- The existing point-process smoother in `src/state_space_practice/point_process_kalman.py` must remain the computational backbone. Do not introduce a second filtering stack.
- MVP is non-switching only.
- MVP is small-network only: target synthetic tests with 2-5 neurons and a modest number of lag basis functions.
- Shared-input confounds are acknowledged but not solved in MVP; the first model estimates functional coupling, not causal synaptic connectivity.
- The gradient stability gate for `PointProcessModel` (SGD plan Task 3) must pass before implementing `fit_sgd()`. If Laplace-EKF gradients are unstable, the SGD path is deferred but the iterated-smoothing `fit()` path remains available.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_spike_history_features.py src/state_space_practice/tests/test_dynamic_spike_coupling.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_point_process_kalman.py -v`
- Lint after each completed task: `conda run -n state_space_practice ruff check src/state_space_practice`
- The first end-to-end model must recover directionality on synthetic data with known sparse coupling.

**Feasibility Status:** MEDIUM (direct reuse of point-process machinery, but state dimension grows quickly)

**Why this is a separate family:** Current oscillator spike models infer latent population coupling with neurons as conditionally independent readouts of a latent state. This plan instead makes the coupling weights themselves the latent state, which is a different model class even though the observation likelihood and smoother machinery are reused.

**MVP Scope Lock:**

- One model class: `DynamicSpikeCouplingModel`
- Non-switching only
- Poisson log-linear spike-history GLM only
- Time-varying coupling weights follow random-walk dynamics
- Optional self-history terms allowed, but no latent common-drive state in MVP
- MVP supports ≤ 3 neurons (state dim ≤ 36 with 4 basis functions). Model constructor must validate and raise if `n_neurons > 3`. For N neurons with L basis functions, state dim is N² × L; the Laplace-EKF covariance is (N²L)² entries with O((N²L)³) per-timestep inversion.

**Defer Until Post-MVP:**

- Switching connectivity regimes
- Area-to-area penalties
- Shared latent common-drive factors
- Hawkes-process style self-excitation with exact branching-process semantics
- Large-scale sparse optimization for tens to hundreds of neurons

---

## Model Formulation

For neuron `i` at time `t`, define spike-history basis features from all source neurons `j`:

$$
\log \lambda_i(t) = b_i(t) + \sum_j \sum_{\ell=1}^{L} w_{ij\ell}(t) \, h_\ell * y_j(t)
$$

State dynamics:

$$
w_t = w_{t-1} + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, Q)
$$

**Design decision: time-varying baselines.** The baseline firing rates `b_i(t)` are included in the latent state vector alongside the coupling weights. This adds N dimensions to the state (minor compared to N²L for coupling weights) and prevents confounding rate changes with coupling changes. The full latent state is `x_t = [b_1(t), ..., b_N(t), w_{111}(t), ..., w_{NNL}(t)]`. Static baselines would force the model to absorb non-stationarity in firing rates into the coupling weights.

Flatten all weights into one latent vector `x_t`. Build a block design matrix `Z_t` so that:

$$
\log \lambda(t) = Z_t x_t
$$

This matches the multi-neuron point-process API already used by `stochastic_point_process_smoother` in `src/state_space_practice/point_process_kalman.py`.

---

## Task 1: Spike-History Basis and Feature Builder

**Files:**
- Create: `src/state_space_practice/spike_history_features.py`
- Test: `src/state_space_practice/tests/test_spike_history_features.py`

### Step 1: Write failing tests for basis generation and feature layout

```python
import numpy as np

from state_space_practice.spike_history_features import (
    make_exponential_history_basis,
    build_spike_history_design,
)


def test_history_basis_shape():
    basis = make_exponential_history_basis(n_lags=20, n_basis=4)
    assert basis.shape == (20, 4)


def test_design_matrix_shape_for_three_neurons():
    spikes = np.zeros((100, 3), dtype=int)
    basis = make_exponential_history_basis(n_lags=10, n_basis=2)
    design = build_spike_history_design(spikes, basis)
    assert design.shape == (100, 3, 3 + 3 * 3 * 2)  # baselines + coupling


def test_design_matrix_block_sparsity():
    """Each target neuron's row should be zero outside its own source block."""
    spikes = np.zeros((100, 3), dtype=int)
    basis = make_exponential_history_basis(n_lags=10, n_basis=2)
    design = build_spike_history_design(spikes, basis)
    n_sources, n_basis = 3, 2
    for target in range(3):
        block_start = target * n_sources * n_basis
        block_end = (target + 1) * n_sources * n_basis
        # Zero outside own block
        assert np.all(design[:, target, :block_start] == 0)
        assert np.all(design[:, target, block_end:] == 0)
```

### Step 2: Run tests to verify failure

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_spike_history_features.py -v`

Expected: FAIL because the module does not exist.

### Step 3: Implement minimal basis and design helpers

Implementation requirements:

- `make_exponential_history_basis(n_lags, n_basis)` returns lag basis functions shaped `(n_lags, n_basis)`
- `build_spike_history_design(spikes, basis, include_self_history=True)` returns `(n_time, n_neurons, n_state_dims)`
- each target neuron row should activate only its own block of coefficients inside the flattened state vector

Example state packing for `n_neurons=3`, `n_basis=2`:

```python
n_state_dims = n_targets + n_targets * n_sources * n_basis  # baselines + coupling
# First n_targets entries are baselines b_i(t)
# Remaining entries are coupling weights, packed as:
coupling_index = n_targets + ((target * n_sources) + source) * n_basis + basis_index
```

### Step 4: Re-run tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_spike_history_features.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/spike_history_features.py src/state_space_practice/tests/test_spike_history_features.py
git commit -m "feat: add spike history feature builders"
```

---

## Task 2: Dynamic Coupling Observation Model Wrapper

**Files:**
- Create: `src/state_space_practice/dynamic_spike_coupling.py`
- Test: `src/state_space_practice/tests/test_dynamic_spike_coupling.py`

### Step 1: Write failing tests for model initialization and fit

Add tests covering:

```python
def test_model_initializes_state_dimensions_from_spikes():
    ...


def test_model_fits_small_synthetic_dataset():
    ...


def test_model_returns_coupling_trajectory_with_expected_shape():
    ...
```

### Step 2: Run tests to verify failure

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_dynamic_spike_coupling.py -v`

Expected: FAIL because the model does not exist.

### Step 3: Implement `DynamicSpikeCouplingModel`

Model API sketch:

```python
class DynamicSpikeCouplingModel:
    def __init__(self, dt: float, n_lags: int = 20, n_basis: int = 4, include_self_history: bool = True):
        ...

    def fit(self, spikes: ArrayLike, max_iter: int = 50, tolerance: float = 1e-4) -> list[float]:
        ...

    def get_coupling_trajectory(self) -> Array:
        ...
```

Implementation guidance:

- build spike-history features from observed spikes
- initialize a wrapped `PointProcessModel`-like parameter set with random-walk dynamics on the flattened coupling vector
- use `stochastic_point_process_smoother` and `kalman_maximization_step` patterns from `point_process_kalman.py`
- store the smoothed latent state as the time-varying coupling trajectory

**Fitting procedure:** The model uses iterated Laplace smoothing with Q updates — not standard EM. The E-step runs `stochastic_point_process_smoother` to get smoothed coupling trajectories. The M-step updates only Q (process noise on coupling weights) using standard Kalman M-step formulas from the smoother sufficient statistics. There is no closed-form M-step for the spike-history GLM observation model — the Laplace approximation in the E-step handles observation parameters implicitly. Convergence is monitored via marginal log-likelihood but is not guaranteed in the EM sense. Document this in the class docstring.

### Step 4: Re-run focused tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_dynamic_spike_coupling.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/dynamic_spike_coupling.py src/state_space_practice/tests/test_dynamic_spike_coupling.py
git commit -m "feat: add dynamic neuron coupling model"
```

---

## Task 3: Synthetic Recovery Tests for Directionality

**Files:**
- Modify: `src/state_space_practice/tests/test_dynamic_spike_coupling.py`
- Optional helper in: `src/state_space_practice/dynamic_spike_coupling.py`

### Step 1: Write failing synthetic recovery test

Construct a synthetic 2-neuron system where neuron 0 drives neuron 1, but not vice versa.

Test requirements:

- simulated coupling trajectory has known sign and target
- fitted model recovers larger absolute coupling for `0 -> 1` than for `1 -> 0`
- recovered trajectory is smoother than the raw spike train and remains finite

### Step 2: Run test to verify failure

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_dynamic_spike_coupling.py -v -k directionality`

Expected: FAIL until the model exposes coupling trajectory accessors and synthetic helper functions.

### Step 3: Add trajectory accessors and summary helpers

Expose helpers such as:

```python
def get_mean_coupling_matrix(self) -> Array:
    ...


def get_coupling_trajectory(self) -> Array:
    ...
```

Where the returned trajectory is shaped `(n_time, n_targets, n_sources, n_basis)` or a documented flattened equivalent.

### Step 4: Re-run focused tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_dynamic_spike_coupling.py -v -k directionality`

Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/dynamic_spike_coupling.py src/state_space_practice/tests/test_dynamic_spike_coupling.py
git commit -m "feat: validate directional recovery in dynamic neuron coupling model"
```

---

## Task 4: Stability and Complexity Guards

**Files:**
- Modify: `src/state_space_practice/dynamic_spike_coupling.py`
- Modify: `src/state_space_practice/tests/test_dynamic_spike_coupling.py`

### Step 1: Write failing tests for guardrails

Cover:

- model raises if `n_state_dims` becomes unreasonably large for MVP
- model rejects negative spikes or malformed arrays
- optional exclusion of self-history removes diagonal blocks from the state vector

### Step 2: Implement guardrails

Add:

- input validation for spike array shape and values
- a conservative maximum state-size threshold for the MVP implementation
- explicit handling of `include_self_history=False`

### Step 3: Run targeted tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_dynamic_spike_coupling.py -v`

Expected: PASS

### Step 4: Run neighbor regression tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_point_process_kalman.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/dynamic_spike_coupling.py src/state_space_practice/tests/test_dynamic_spike_coupling.py
git commit -m "feat: add guardrails for dynamic neuron coupling model"
```

---

## Success Criteria

- A small-network non-switching dynamic neuron-coupling model fits end to end using the existing point-process smoother.
- Synthetic tests recover directional asymmetry in a known 2-neuron system.
- The design matrix layout is explicit and test-covered.
- The model exposes interpretable coupling summaries over time.

## Open Questions to Resolve During Implementation

- Should self-history be mandatory in MVP for numerical stability?
- Is a diagonal `Q` enough, or do we need block-diagonal process covariance over target neurons?

## Related Plans

- `docs/plans/2026-04-06-sgd-fitting-all-models.md`
- `docs/plans/2026-04-07-regularized-oscillator-connectivity.md`