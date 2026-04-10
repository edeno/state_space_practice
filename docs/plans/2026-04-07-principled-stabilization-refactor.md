# Principled Stabilization Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Replace ad hoc, model-local stabilization with a principled, shared stability framework that enforces hard constraints by construction, treats regularization as part of the algorithm, and exposes emergency guards as explicit telemetry rather than silent behavior.

**Architecture:** Introduce a shared stability policy layer for hard constraints, soft regularization, and emergency guards. Move PSD, simplex, and finite-value handling into shared helpers; move oscillator and coupled-transition stability from optimize-then-clip to stable-by-construction parameterizations or explicit projected updates; unify approximate-EM trust-region updates across switching and non-switching point-process models; and add invariant-focused tests that exercise both normal and pathological paths.

**Tech Stack:** JAX, existing numerical helpers in `src/state_space_practice/kalman.py`, oscillator transition utilities in `src/state_space_practice/oscillator_utils.py`, oscillator models in `src/state_space_practice/oscillator_models.py`, point-process models in `src/state_space_practice/point_process_models.py`, switching point-process models in `src/state_space_practice/switching_point_process.py`, pytest.

---

## Why This Refactor Exists

The current codebase mixes three different concerns:

1. **Hard constraints** that must always hold, such as PSD covariances, stable transition matrices, valid probabilities, and finite updates.
2. **Soft modeling preferences** such as smooth dynamics, conservative covariance updates, and preference for specific oscillator block structure.
3. **Emergency guards** such as log floors, eigenvalue clipping, and fallback matrices used only to keep execution finite.

Those concerns currently live in multiple files with inconsistent policies. The refactor should make the algorithmic intent explicit:

- Hard constraints are enforced by parameterization or explicit projection as part of the update rule.
- Soft preferences enter as trust-region or proximal penalties, not hidden cleanup.
- Emergency guards are isolated, counted, and tested.

## Success Criteria

The refactor is successful when all of the following are true:

1. No model class silently depends on post-hoc clipping to restore stability after an unconstrained update.
2. Shared hard-constraint helpers define one canonical PSD policy, one canonical simplex policy, and one canonical finite fallback policy.
3. Oscillator and coupled transition updates are stable by construction or use an explicitly documented projected update.
4. Switching and non-switching point-process models use the same trust-region semantics for approximate-EM covariance updates.
5. Tests verify invariants directly: PSD, spectral radius bounds, probability normalization, finite outputs, and fallback-path behavior.
6. Any remaining emergency guard emits telemetry so repeated activation is observable.

## Scope Lock

Implement now:

- Shared stability policy API and telemetry.
- Covariance and probability stabilization centralization.
- Stable-by-construction transition updates for oscillator-family models.
- Unified trust-region update semantics for approximate-EM models.
- Invariant and failure-path regression tests.

Defer until after this plan:

- A full manifold or Riemannian optimizer for transition parameters.
- End-to-end replacement of all legacy models in `models.py` and experimental notebooks.
- Large-scale performance work unrelated to stabilization semantics.

## Verification Gates

Run all commands in the conda environment from `CLAUDE.md`.

Targeted unit tests:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_stability.py \
  src/state_space_practice/tests/test_oscillator_utils.py \
  src/state_space_practice/tests/test_oscillator_models.py \
  src/state_space_practice/tests/test_point_process_models.py \
  src/state_space_practice/tests/test_switching_point_process.py -v
```

Neighbor regression tests:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py \
  src/state_space_practice/tests/test_point_process_kalman.py -v
```

Lint gate:

```bash
conda run -n state_space_practice ruff check src/
```

Optional type gate if enabled in the repo:

```bash
conda run -n state_space_practice mypy src/state_space_practice/
```

Stop conditions:

- If a proposed stable-by-construction parameterization cannot represent currently supported oscillator families, stop and implement the projected-update path first rather than widening scope.
- If invariant tests pass only because emergency guards fire repeatedly, stop and instrument the underlying update instead of accepting the behavior.
- If switching and non-switching point-process models need materially different trust-region semantics, stop and document that divergence explicitly rather than pretending the API is shared.

---

## Task 1: Introduce a Shared Stability Policy Layer

**Files:**
- Create: `src/state_space_practice/stability.py`
- Create: `src/state_space_practice/tests/test_stability.py`
- Modify: `src/state_space_practice/__init__.py`

**Step 1: Write the failing tests for shared policies**

Add `test_stability.py` with direct invariant tests for PSD projection, simplex normalization, finite fallback handling, and telemetry counters.

```python
import jax.numpy as jnp

from state_space_practice.stability import (
    StabilityTelemetry,
    project_psd_with_policy,
    stabilize_probability_vector,
)


def test_project_psd_with_policy_returns_psd_matrix():
    cov = jnp.array([[1.0, 2.0], [2.0, -1.0]])
    telemetry = StabilityTelemetry()

    projected = project_psd_with_policy(cov, min_eigenvalue=1e-6, telemetry=telemetry)

    eigvals = jnp.linalg.eigvalsh(projected)
    assert jnp.min(eigvals) >= 0.0
    assert telemetry.psd_projections == 1


def test_stabilize_probability_vector_returns_simplex_point():
    probs = jnp.array([0.0, -1e-5, 2.0])
    telemetry = StabilityTelemetry()

    stabilized = stabilize_probability_vector(probs, floor=1e-10, telemetry=telemetry)

    assert jnp.all(stabilized >= 0.0)
    assert jnp.isclose(jnp.sum(stabilized), 1.0)
    assert telemetry.simplex_repairs == 1
```

**Step 2: Run the new tests to verify they fail**

Run:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_stability.py -v
```

Expected: FAIL with import errors.

**Step 3: Implement shared stability policy objects**

Create `stability.py` with a small API surface. Keep it focused.

```python
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class StabilityTelemetry:
    psd_projections: int = 0
    simplex_repairs: int = 0
    finite_fallbacks: int = 0
    spectral_radius_repairs: int = 0


@dataclass(frozen=True)
class CovariancePolicy:
    min_eigenvalue: float = 1e-8


@dataclass(frozen=True)
class ProbabilityPolicy:
    floor: float = 1e-10
```

Add only the helpers this refactor needs immediately:

- `project_psd_with_policy`
- `stabilize_probability_vector`
- `finite_fallback_matrix`
- `counted_spectral_radius_projection`

Do not move model-specific heuristics into this module.

**Step 4: Export the public policy types**

Add the new policy objects and helpers to `src/state_space_practice/__init__.py` only if they are already part of the repo’s public import style. Otherwise keep them internal.

**Step 5: Re-run the targeted tests**

Run:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_stability.py -v
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/state_space_practice/stability.py src/state_space_practice/tests/test_stability.py src/state_space_practice/__init__.py
git commit -m "refactor: add shared stability policy layer"
```

---

## Task 2: Centralize Hard Constraints for Covariances and Probabilities

**Files:**
- Modify: `src/state_space_practice/kalman.py`
- Modify: `src/state_space_practice/switching_kalman.py`
- Modify: `src/state_space_practice/point_process_kalman.py`
- Modify: `src/state_space_practice/switching_point_process.py`
- Modify: `src/state_space_practice/tests/test_kalman.py`
- Modify: `src/state_space_practice/tests/test_switching_kalman.py`
- Modify: `src/state_space_practice/tests/test_point_process_kalman.py`

**Step 1: Write failing invariant tests before moving code**

Add tests asserting that shared covariance outputs are PSD within tolerance and that probability vectors are simplex-valid after any shared stabilization path.

Example PSD assertion:

```python
eigvals = jnp.linalg.eigvalsh(cov_est)
assert jnp.all(jnp.isfinite(cov_est))
assert jnp.min(eigvals) >= -1e-8
```

Example probability assertion:

```python
assert jnp.all(prob >= 0.0)
assert jnp.isclose(jnp.sum(prob), 1.0)
```

**Step 2: Run the focused tests and confirm the new cases fail or are unimplemented**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py \
  src/state_space_practice/tests/test_point_process_kalman.py -v
```

**Step 3: Replace duplicate PSD logic with the shared policy**

In `kalman.py` and callers, swap local eigendecomposition clipping or bare symmetrization with `project_psd_with_policy` from the new module.

Suggested shape:

```python
from state_space_practice.stability import CovariancePolicy, project_psd_with_policy


DEFAULT_COVARIANCE_POLICY = CovariancePolicy(min_eigenvalue=1e-8)


def stabilize_covariance(cov: Array, policy: CovariancePolicy = DEFAULT_COVARIANCE_POLICY) -> Array:
    return project_psd_with_policy(cov, min_eigenvalue=policy.min_eigenvalue)
```

**Step 4: Replace duplicate probability floors with the shared policy**

In `switching_kalman.py` and any switching-model helper, replace model-local probability normalization code with `stabilize_probability_vector`, unless the function has genuinely different semantics.

**Step 5: Keep only one canonical hard-constraint policy per constraint type**

Do not leave parallel implementations of:

- PSD covariance repair
- simplex repair
- finite matrix fallback

If a caller needs a different threshold, pass a different policy object rather than writing a new helper.

**Step 6: Re-run the shared invariant tests**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_stability.py \
  src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py \
  src/state_space_practice/tests/test_point_process_kalman.py -v
```

**Step 7: Commit**

```bash
git add src/state_space_practice/stability.py src/state_space_practice/kalman.py src/state_space_practice/switching_kalman.py src/state_space_practice/point_process_kalman.py src/state_space_practice/switching_point_process.py src/state_space_practice/tests/test_kalman.py src/state_space_practice/tests/test_switching_kalman.py src/state_space_practice/tests/test_point_process_kalman.py
git commit -m "refactor: centralize hard constraint stabilization"
```

---

## Task 3: Replace Post-hoc Spectral Shrinkage With Stable-by-Construction Oscillator Updates

**Files:**
- Modify: `src/state_space_practice/oscillator_utils.py`
- Modify: `src/state_space_practice/oscillator_models.py`
- Modify: `src/state_space_practice/point_process_models.py`
- Modify: `src/state_space_practice/tests/test_oscillator_utils.py`
- Modify: `src/state_space_practice/tests/test_oscillator_models.py`
- Modify: `src/state_space_practice/tests/test_point_process_models.py`

**Step 1: Write failing invariant tests for stable-by-construction parameterization**

Add tests that construct raw unconstrained oscillator parameters, map them into transition matrices, and assert the resulting matrices satisfy the target spectral radius bound without any extra rescue step.

```python
def test_parameterized_transition_is_stable_by_construction():
    raw_damping = jnp.array([100.0, -100.0])
    raw_frequency = jnp.array([0.3, 1.2])

    A = build_stable_oscillator_transition(raw_damping, raw_frequency, max_radius=0.99)

    radius = jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    assert radius <= 0.99 + 1e-6
```

Add a separate pathological-input test for the fallback path in `oscillator_utils.py`.

```python
def test_project_to_closest_rotation_nan_input_uses_finite_fallback():
    mat = jnp.array([[jnp.nan, 0.0], [0.0, 1.0]])
    projected = _project_to_closest_rotation(mat)
    assert jnp.all(jnp.isfinite(projected))
```

**Step 2: Run the targeted tests and verify failure**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_oscillator_utils.py \
  src/state_space_practice/tests/test_oscillator_models.py \
  src/state_space_practice/tests/test_point_process_models.py -v
```

**Step 3: Add stable parameter maps in `oscillator_utils.py`**

Implement helpers that produce valid oscillator blocks from unconstrained raw parameters.

```python
def bounded_radius(raw_radius: Array, max_radius: float = 0.99) -> Array:
    return max_radius * jax.nn.sigmoid(raw_radius)


def build_stable_oscillator_block(raw_radius: Array, raw_phase: Array) -> Array:
    radius = bounded_radius(raw_radius)
    phase = raw_phase
    c = jnp.cos(phase)
    s = jnp.sin(phase)
    rotation = jnp.array([[c, -s], [s, c]])
    return radius * rotation
```

For coupled systems, if an exact stable-by-construction parameterization is too large for one pass, implement an explicit projected update function named as such, for example `project_transition_update_to_feasible_set`, and call that the algorithm.

**Step 4: Update oscillator models to optimize raw parameters, not repaired matrices**

In `oscillator_models.py` and `point_process_models.py`, move the update flow from:

- estimate `A`
- optionally project block structure
- always shrink spectral radius

to:

- estimate or optimize raw oscillator parameters
- build `A` through the stable parameter map
- keep block-structure projection only if it remains part of a clearly named projected-update path

The implementation should not rely on a final unconditional `A_j = A_j * scale` as the normal path.

**Step 5: Keep emergency fallback only as a guard**

Retain the damped-identity fallback in `oscillator_utils.py`, but route it through the shared finite fallback helper and attach telemetry. The fallback should be rare and observable.

**Step 6: Re-run the oscillator tests**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_oscillator_utils.py \
  src/state_space_practice/tests/test_oscillator_models.py \
  src/state_space_practice/tests/test_point_process_models.py -v
```

**Step 7: Commit**

```bash
git add src/state_space_practice/oscillator_utils.py src/state_space_practice/oscillator_models.py src/state_space_practice/point_process_models.py src/state_space_practice/tests/test_oscillator_utils.py src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_point_process_models.py
git commit -m "refactor: make oscillator transition updates stable by construction"
```

---

## Task 4: Unify Approximate-EM Trust-Region Updates Across Point-process Models

**Files:**
- Modify: `src/state_space_practice/point_process_models.py`
- Modify: `src/state_space_practice/switching_point_process.py`
- Modify: `src/state_space_practice/tests/test_point_process_models.py`
- Modify: `src/state_space_practice/tests/test_switching_point_process.py`

**Step 1: Write failing tests for shared trust-region semantics**

Add tests asserting that switching and non-switching models apply the same update semantics to `Q` and `init_cov` under a common config:

- blend new estimate with previous estimate
- clip eigenvalues through the same policy object
- preserve PSD

Suggested shape:

```python
def test_regularized_init_cov_update_is_psd_and_bounded():
    old_cov = jnp.eye(2)
    new_cov = 1000.0 * jnp.eye(2)

    updated = regularize_covariance_update(old_cov, new_cov, weight=0.3, max_eigenvalue=10.0)

    eigvals = jnp.linalg.eigvalsh(updated)
    assert jnp.min(eigvals) >= 0.0
    assert jnp.max(eigvals) <= 10.0 + 1e-6
```

**Step 2: Run the tests and verify failure or missing API**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_point_process_models.py \
  src/state_space_practice/tests/test_switching_point_process.py -v
```

**Step 3: Introduce a shared approximate-EM covariance update helper**

Implement a helper, likely in `stability.py` or `kalman.py`, with explicit semantics.

```python
def regularize_covariance_update(
    previous: Array,
    proposed: Array,
    trust_region_weight: float,
    min_eigenvalue: float,
    max_eigenvalue: float | None,
) -> Array:
    blended = trust_region_weight * proposed + (1.0 - trust_region_weight) * previous
    return project_psd_with_policy(blended, min_eigenvalue=min_eigenvalue, max_eigenvalue=max_eigenvalue)
```

Use that helper in both point-process model families. Do not maintain a switching-only trust-region semantics if the underlying approximation issue is the same.

**Step 4: Unify config naming and validation**

Keep one shared configuration vocabulary for approximate-EM regularization. A reasonable target is:

- `trust_region_weight`
- `q_min_eigenvalue`
- `q_max_eigenvalue`
- `init_cov_max_eigenvalue`

Validate all bound relationships explicitly.

**Step 5: Remove hard-coded, model-local init covariance caps where possible**

Replace direct clips like `[1e-4, 2.0]` in `point_process_models.py` with the shared helper and a named policy. If the non-switching model genuinely needs a tighter cap, encode that difference in a model-specific policy constant with a docstring.

**Step 6: Re-run the trust-region tests**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_point_process_models.py \
  src/state_space_practice/tests/test_switching_point_process.py -v
```

**Step 7: Commit**

```bash
git add src/state_space_practice/point_process_models.py src/state_space_practice/switching_point_process.py src/state_space_practice/tests/test_point_process_models.py src/state_space_practice/tests/test_switching_point_process.py src/state_space_practice/stability.py
git commit -m "refactor: unify approximate-em regularization semantics"
```

---

## Task 5: Add Telemetry and Make Emergency Guards Observable

**Files:**
- Modify: `src/state_space_practice/stability.py`
- Modify: `src/state_space_practice/oscillator_utils.py`
- Modify: `src/state_space_practice/oscillator_models.py`
- Modify: `src/state_space_practice/point_process_models.py`
- Modify: `src/state_space_practice/switching_point_process.py`
- Modify: `src/state_space_practice/tests/test_stability.py`

**Step 1: Write failing tests for telemetry increments**

Add tests that call fallback-heavy paths and assert telemetry counters increment.

```python
def test_spectral_radius_repair_increments_counter():
    telemetry = StabilityTelemetry()
    A = 2.0 * jnp.eye(2)

    repaired = counted_spectral_radius_projection(A, max_radius=0.99, telemetry=telemetry)

    assert telemetry.spectral_radius_repairs == 1
    assert jnp.max(jnp.abs(jnp.linalg.eigvals(repaired))) <= 0.99 + 1e-6
```

**Step 2: Run the new telemetry tests and verify failure**

Run:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_stability.py -v
```

**Step 3: Thread telemetry through emergency paths**

Add optional telemetry arguments to emergency helpers only. Do not thread telemetry through every inner-loop call unless you can do so without noisy overhead.

Minimum coverage:

- finite fallback in `oscillator_utils.py`
- PSD projection in shared covariance repair
- simplex repair in switching probability stabilization
- any remaining spectral radius rescue path kept for compatibility

**Step 4: Add warnings or diagnostics at model boundaries**

At the model `fit()` level, summarize whether repairs occurred, for example via logger output or a returned diagnostics object. The goal is to make repeated emergency repair observable during debugging.

**Step 5: Re-run telemetry tests**

Run:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_stability.py -v
```

**Step 6: Commit**

```bash
git add src/state_space_practice/stability.py src/state_space_practice/oscillator_utils.py src/state_space_practice/oscillator_models.py src/state_space_practice/point_process_models.py src/state_space_practice/switching_point_process.py src/state_space_practice/tests/test_stability.py
git commit -m "refactor: add stabilization telemetry and diagnostics"
```

---

## Task 6: Replace Outcome-only Tests With Invariant and Failure-path Tests

**Files:**
- Modify: `src/state_space_practice/tests/test_oscillator_utils.py`
- Modify: `src/state_space_practice/tests/test_oscillator_models.py`
- Modify: `src/state_space_practice/tests/test_point_process_models.py`
- Modify: `src/state_space_practice/tests/test_switching_point_process.py`
- Modify: `src/state_space_practice/tests/test_switching_kalman.py`

**Step 1: Add explicit invariant assertions to existing recovery tests**

Where tests currently assert only output shape or approximate recovery, also assert:

- PSD covariance slices
- bounded spectral radius
- probability vectors on the simplex
- finite posterior means and covariances

Example invariant block:

```python
assert jnp.all(jnp.isfinite(A_est))
assert jnp.max(jnp.abs(jnp.linalg.eigvals(A_est))) <= 0.99 + 1e-6
assert jnp.all(jnp.isfinite(Q_est))
assert jnp.min(jnp.linalg.eigvalsh(Q_est)) >= -1e-8
```

**Step 2: Add failure-path tests, not just nominal-path tests**

Add cases for:

- NaN inputs to `_project_to_closest_rotation`
- huge or indefinite proposed covariances in approximate-EM updates
- low-occupancy switching states
- exact-zero probabilities in switching diagnostics

**Step 3: Add one integration-style regression for repeated repair activation**

Create a short-run model fit that would previously trigger unstable updates and assert:

- outputs remain finite
- repair counts are bounded
- the model does not spend every iteration in fallback mode

If the integration path is too expensive for unit tests, keep it short and deterministic.

**Step 4: Run the full targeted stability suite**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_stability.py \
  src/state_space_practice/tests/test_oscillator_utils.py \
  src/state_space_practice/tests/test_oscillator_models.py \
  src/state_space_practice/tests/test_point_process_models.py \
  src/state_space_practice/tests/test_switching_point_process.py \
  src/state_space_practice/tests/test_switching_kalman.py -v
```

**Step 5: Commit**

```bash
git add src/state_space_practice/tests/test_stability.py src/state_space_practice/tests/test_oscillator_utils.py src/state_space_practice/tests/test_oscillator_models.py src/state_space_practice/tests/test_point_process_models.py src/state_space_practice/tests/test_switching_point_process.py src/state_space_practice/tests/test_switching_kalman.py
git commit -m "test: add invariant and failure-path stabilization coverage"
```

---

## Task 7: Clean Up Docs and Developer Guidance So The New Rules Stick

**Files:**
- Modify: `CLAUDE.md`
- Modify: `src/state_space_practice/kalman.py`
- Modify: `src/state_space_practice/stability.py`
- Modify: `src/state_space_practice/oscillator_models.py`
- Modify: `src/state_space_practice/point_process_models.py`
- Modify: `src/state_space_practice/switching_point_process.py`

**Step 1: Document the taxonomy in `CLAUDE.md`**

Add a short section that says:

- hard constraints must be enforced by construction or explicit projection
- soft regularization belongs in the objective or update rule
- emergency guards must be instrumented and tested

**Step 2: Update docstrings at the main APIs**

Make sure the public or quasi-public functions explain which parts are hard constraints and which parts are soft regularization.

For example, `QRegularizationConfig` should say it defines a proximal approximate-EM update, not a cleanup pass.

**Step 3: Remove misleading comments that imply emergency guards are ordinary behavior**

Audit comments such as “fall back to original matrix” or “only accept if it improves Q” where the new algorithmic interpretation is different.

**Step 4: Run lint and the full neighbor regression suite**

Run:

```bash
conda run -n state_space_practice ruff check src/
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py \
  src/state_space_practice/tests/test_point_process_kalman.py \
  src/state_space_practice/tests/test_oscillator_utils.py \
  src/state_space_practice/tests/test_oscillator_models.py \
  src/state_space_practice/tests/test_point_process_models.py \
  src/state_space_practice/tests/test_switching_point_process.py -v
```

**Step 5: Commit**

```bash
git add CLAUDE.md src/state_space_practice/kalman.py src/state_space_practice/stability.py src/state_space_practice/oscillator_models.py src/state_space_practice/point_process_models.py src/state_space_practice/switching_point_process.py
git commit -m "docs: codify principled stabilization rules"
```

---

## Final Verification

Run the full set before declaring the refactor complete:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_stability.py \
  src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py \
  src/state_space_practice/tests/test_point_process_kalman.py \
  src/state_space_practice/tests/test_oscillator_utils.py \
  src/state_space_practice/tests/test_oscillator_models.py \
  src/state_space_practice/tests/test_point_process_models.py \
  src/state_space_practice/tests/test_switching_point_process.py -v

conda run -n state_space_practice ruff check src/
```

If enabled:

```bash
conda run -n state_space_practice mypy src/state_space_practice/
```

Expected final state:

- One shared stability-policy layer exists and is reused.
- Oscillator-family models do not depend on normal-path post-hoc spectral shrinkage.
- Switching and non-switching point-process models share approximate-EM regularization semantics.
- Pathological inputs produce finite, instrumented fallback behavior.
- Tests assert invariants directly and cover failure paths.
