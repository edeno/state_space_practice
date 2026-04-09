# Regularized Oscillator Connectivity Implementation Plan — COMPLETE

> **Status:** DONE as of 2026-04-08. All 4 tasks complete, 24 tests.

**Goal:** Add structured SGD penalties for oscillator coupling parameters so `DirectedInfluenceModel` and later `CorrelatedNoiseModel` can express sparse edge structure and area-to-area pathway priors.

## Implementation Summary

- `src/state_space_practice/oscillator_regularization.py` — penalty utilities, config, area summary
- `DirectedInfluenceModel.fit_sgd()` accepts `connectivity_penalty` kwarg
- Three penalty families: edge L1, area group L2, state-shared group L2
- `get_area_coupling_summary()` for reporting block-level norms
- 21 regularization tests + 3 DIM regularized SGD tests
- Integration tests verify edge penalty shrinks coupling and area penalty reduces cross-area coupling

**Architecture:** Keep the existing EM path unchanged. Add a small regularization layer that operates on the constrained scientific parameterization used by oscillator SGD (`coupling_strength`, `phase_difference`, optionally `process_variance`) and augments `_sgd_loss_fn` with additive penalties. The first implementation targets `DirectedInfluenceModel`, where coupling is the main scientific quantity; `CorrelatedNoiseModel` support comes later and remains explicitly gated.

**Tech Stack:** JAX, optax, `SGDFittableMixin`, `oscillator_models.py`, `oscillator_utils.py`, new regularization helpers in `src/state_space_practice/oscillator_regularization.py`.

---

**Prerequisite Gates:**

- The SGD infrastructure plan in `docs/plans/2026-04-06-sgd-fitting-all-models.md` must be implemented for `DirectedInfluenceModel` before starting this work.
- The oscillator SGD path must already reconstruct `A` from scientific parameters rather than optimizing raw matrices.
- This plan must not change EM behavior, existing projections, or the semantics of `fit()`.
- Start with non-switching penalties inside a single fitted model instance. Do not combine this plan with multi-region block dynamics or mixture priors.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_regularization.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v`
- Lint after each completed task: `conda run -n state_space_practice ruff check src/state_space_practice`
- If any penalty causes synthetic recovery to collapse relative to the unpenalized SGD baseline, stop and tune the parameterization before proceeding.

**Feasibility Status:** HIGH for `DirectedInfluenceModel`; MEDIUM for `CorrelatedNoiseModel`

**Why this is a separate plan:** The base SGD rollout solves optimizer availability and constrained parameterization. This plan adds new scientific objectives. Keeping it separate makes it easier to tell whether any regression comes from the optimizer itself or from the new penalties.

**MVP Scope Lock:**

- Implement penalties only for SGD, not EM.
- Start with `DirectedInfluenceModel` only.
- Support three penalty families in the first release:
  - edge-wise sparsity on `coupling_strength`
  - area-to-area group penalty on blocks of `coupling_strength`
  - state-shared group penalty that ties the same pathway across discrete states
- Require explicit oscillator-to-area labels from the caller.

**Defer Until Post-MVP:**

- Multi-region block transition models from `docs/plans/2026-04-03-cross-region-coupling.md`
- Regularized EM or generalized-EM updates
- Penalties on `phase_difference`
- Bayesian priors or variational inference for coupling pathways
- Automatic model selection over area group structure

---

## Model Formulation

For a fitted `DirectedInfluenceModel`, let `C[s, i, j]` denote the constrained coupling strength from oscillator `j` to oscillator `i` in discrete state `s`. Let `g(i)` map oscillator `i` to a brain area label.

Base SGD objective:

$$
\mathcal{L}_{\text{sgd}} = -\log p(y \mid \theta)
$$

Regularized objective:

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{sgd}}
+ \lambda_{\text{edge}} \sum_{s,i,j} |C[s,i,j]|
+ \lambda_{\text{area}} \sum_{s,a,b} \lVert C_{ab}^{(s)} \rVert_F
+ \lambda_{\text{shared}} \sum_{a,b} \sqrt{\sum_s \lVert C_{ab}^{(s)} \rVert_F^2}
$$

where `C_{ab}^{(s)}` is the area-pair block for area `a -> b` in state `s`.

**Smooth approximations:** All penalties use smooth approximations to avoid gradient singularities at zero:
- Edge L1: `sum |C[s,i,j]|` is implemented as `sum sqrt(C[s,i,j]² + ε)` with `ε=1e-8`
- Area group L2: `||C_ab||_F` is implemented as `sqrt(sum C_ab² + ε)`
- State-shared group L2: outer sqrt uses the same `sqrt(· + ε)` pattern

Exact sparsity (zero entries) is not achievable via gradient descent. To obtain hard sparsity, apply post-hoc thresholding after optimization: set entries with `|C[s,i,j]| < threshold` to zero and optionally re-fit with those entries frozen.

Interpretation:

- `edge_l1`: sparse individual oscillator-to-oscillator edges
- `area_group_l2`: sparse area-to-area pathways
- `state_shared_group_l2`: pathway selection shared across switching states

---

## Task 1: Penalty Config and Utilities

**Files:**
- Create: `src/state_space_practice/oscillator_regularization.py`
- Test: `src/state_space_practice/tests/test_oscillator_regularization.py`

### Step 1: Write failing tests for penalty primitives

```python
import jax.numpy as jnp

from state_space_practice.oscillator_regularization import (
    OscillatorPenaltyConfig,
    edge_l1_penalty,
    area_group_penalty,
    state_shared_area_penalty,
)


def test_edge_l1_penalty_near_zero_when_no_coupling():
    coupling = jnp.zeros((2, 3, 3))
    assert float(edge_l1_penalty(coupling)) < 1e-3  # smooth approx, not exact zero


def test_area_group_penalty_groups_by_area_labels():
    coupling = jnp.array([
        [[0.0, 1.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 2.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    area_labels = jnp.array([0, 0, 1])
    value = area_group_penalty(coupling, area_labels)
    assert value > 0.0


def test_state_shared_penalty_invariant_to_state_permutation():
    """Shared penalty should not depend on which state is which."""
    coupling = jnp.array([
        [[0.0, 1.0], [0.5, 0.0]],
        [[0.0, 2.0], [0.5, 0.0]],
        [[0.0, 0.5], [1.0, 0.0]],
    ])
    area_labels = jnp.array([0, 1])
    original = state_shared_area_penalty(coupling, area_labels)
    permuted = state_shared_area_penalty(coupling[[2, 0, 1]], area_labels)
    assert jnp.allclose(original, permuted)


def test_state_shared_penalty_reduces_to_area_penalty_for_single_state():
    """When only one state has nonzero coupling, shared = single-state area penalty."""
    coupling = jnp.zeros((3, 2, 2))
    coupling = coupling.at[0].set(jnp.array([[0.0, 1.0], [0.5, 0.0]]))
    area_labels = jnp.array([0, 1])
    shared = state_shared_area_penalty(coupling, area_labels)
    single = area_group_penalty(coupling[:1], area_labels)
    assert jnp.allclose(shared, single, atol=1e-6)
```

### Step 2: Run tests to verify they fail

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_regularization.py -v`

Expected: FAIL because the module does not exist.

### Step 3: Implement minimal penalty utilities

```python
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class OscillatorPenaltyConfig:
    edge_l1: float = 0.0
    area_group_l2: float = 0.0
    state_shared_group_l2: float = 0.0
    area_labels: Array | None = None
    within_area_scale: float = 1.0
    cross_area_scale: float = 1.0
```

Implement helpers that:

- accept constrained `coupling_strength`
- group edges by area labels
- return scalar penalties with no side effects
- treat `area_labels=None` as “no area-group penalties requested”

### Step 4: Re-run tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_regularization.py -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/oscillator_regularization.py src/state_space_practice/tests/test_oscillator_regularization.py
git commit -m "feat: add oscillator connectivity penalty utilities"
```

---

## Task 2: DirectedInfluenceModel SGD Regularization Hook

**Files:**
- Modify: `src/state_space_practice/oscillator_models.py`
- Test: `src/state_space_practice/tests/test_oscillator_models.py`
- Test: `src/state_space_practice/tests/test_oscillator_regularization.py`

### Step 1: Write failing tests for regularized SGD behavior

Add tests to `test_oscillator_models.py` covering:

```python
def test_directed_influence_sgd_accepts_penalty_config():
    ...


def test_directed_influence_edge_penalty_shrinks_coupling_norm():
    ...


def test_directed_influence_area_penalty_shrinks_block_norms():
    ...
```

Synthetic setup should compare two SGD fits with identical seeds:

- baseline: no penalties
- regularized: positive `edge_l1` or `area_group_l2`

Assert:

- both fits remain finite
- regularized fit has smaller total coupling norm or smaller area-block norm
- log-likelihood degradation stays within an explicit tolerance chosen in the test

### Step 2: Run focused tests to verify failure

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_models.py -v -k "regularization or directed_influence"`

Expected: FAIL because `fit_sgd` does not accept penalty config yet.

### Step 3: Add penalty config to `DirectedInfluenceModel`

Implementation requirements:

- add `connectivity_penalty_config: OscillatorPenaltyConfig | None = None` to the model constructor or `fit_sgd` API
- keep EM path unchanged
- inside `_sgd_loss_fn`, compute the base negative marginal log-likelihood first, then add penalties on constrained `coupling_strength`
- default to zero penalty when config is `None`

Minimal pattern:

```python
base_loss = -marginal_log_likelihood
penalty = connectivity_penalty(params["coupling_strength"], self.connectivity_penalty_config)
return base_loss + penalty
```

### Step 4: Re-run focused tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_models.py -v -k "regularization or directed_influence"`

Expected: PASS

### Step 5: Run neighbor regression tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_oscillator_utils.py -v`

Expected: PASS

### Step 5.5: Add hyperparameter range calibration test

Include a synthetic calibration test that documents reasonable penalty ranges:

```python
def test_edge_penalty_calibration_on_sparse_synthetic():
    """Document reasonable lambda_edge range for sparse ground truth.

    Generates a 2-oscillator, 2-area DIM with one active cross-area edge
    and sweeps lambda_edge to find the range that recovers sparsity.
    Results are not asserted tightly — this test documents the landscape.
    """
    # Ground truth: oscillator 0 (area A) drives oscillator 1 (area B)
    # coupling_strength[0,1] = 0.3, all others = 0.0
    # Sweep lambda_edge in [0.001, 0.01, 0.1, 1.0]
    # Assert: lambda_edge=0.1 shrinks false edges below 0.05
    #         lambda_edge=0.001 does not suppress false edges
    ...
```

### Step 6: Commit

```bash
git add src/state_space_practice/oscillator_models.py src/state_space_practice/tests/test_oscillator_models.py
git commit -m "feat: add regularized sgd for directed oscillator coupling"
```

---

## Task 3: Area-to-Area Reporting Helpers

**Files:**
- Modify: `src/state_space_practice/oscillator_models.py`
- Test: `src/state_space_practice/tests/test_oscillator_regularization.py`

### Step 1: Write failing tests for pathway summaries

Add tests for a helper such as:

```python
summary = model.get_area_coupling_summary(area_labels)

assert "block_norms" in summary
assert summary["block_norms"].shape == (n_states, n_areas, n_areas)
```

### Step 2: Implement summary helper

Expose a method on `DirectedInfluenceModel` that returns:

- per-state block Frobenius norms
- total within-area norm
- total cross-area norm
- optionally normalized block strengths

This is analysis/reporting code only. It must not mutate model parameters.

### Step 3: Run tests

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_regularization.py -v`

Expected: PASS

### Step 4: Commit

```bash
git add src/state_space_practice/oscillator_models.py src/state_space_practice/tests/test_oscillator_regularization.py
git commit -m "feat: add area-level oscillator coupling summaries"
```

---

## Deferred: CorrelatedNoiseModel Extension

Penalizing noise coupling (Q) has different scientific meaning than penalizing directed influence (A). CNM regularization is deferred to a separate plan with its own scientific justification. The penalty utilities in `oscillator_regularization.py` are designed to be reusable, but the integration with CNM's `_sgd_loss_fn` requires separate analysis of what "sparse noise coupling" means scientifically.

---

## Task 4: End-to-End Integration Test

**Files:**
- Modify: `src/state_space_practice/tests/test_oscillator_regularization.py`

### Step 1: Write integration test

```python
def test_regularized_sgd_recovers_sparse_area_structure():
    """Full stack: generate sparse DIM data → fit with SGD + area penalty → verify sparsity."""
    # Generate: 4 oscillators, 2 areas (osc 0,1 in area A, osc 2,3 in area B)
    # Ground truth: area A → B pathway active, B → A pathway inactive
    # Fit: DirectedInfluenceModel with fit_sgd + area_group_l2 penalty
    # Verify: model.get_area_coupling_summary() shows A→B block > B→A block
    ...
```

### Step 2: Run integration test

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_oscillator_regularization.py -v -k integration`

Expected: PASS

### Step 3: Commit

```bash
git add src/state_space_practice/tests/test_oscillator_regularization.py
git commit -m "feat: add end-to-end integration test for regularized oscillator SGD"
```

---

## Success Criteria

- `DirectedInfluenceModel.fit_sgd` supports sparse edge and area-to-area penalties without changing EM behavior.
- Synthetic tests recover smaller, cleaner coupling structure under known sparse ground truth.
- Penalty utilities stay separate from filtering code and operate on constrained scientific parameters.
- Area-level summaries can be reported directly from fitted models.
- End-to-end integration test passes: synthetic sparse DIM data → regularized SGD → correct sparsity pattern recovered.
- Smooth penalty approximations are documented and post-hoc thresholding is available for exact sparsity.

## Open Questions to Resolve During Implementation

- Should diagonal self-coupling entries be excluded from penalties by default?
- Should within-area and cross-area pathway penalties use separate hyperparameters or only separate scaling factors?
- Is `phase_difference` best left unpenalized in MVP to keep interpretation simple?

## Related Plans

- `docs/plans/2026-04-06-sgd-fitting-all-models.md`
- `docs/plans/2026-04-03-cross-region-coupling.md`
