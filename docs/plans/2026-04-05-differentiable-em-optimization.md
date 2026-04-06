# Differentiable Parameter Optimization for Existing State-Space Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Add a gradient-based alternative to closed-form EM for models that already exist in this repository, starting with linear-Gaussian systems and only then extending to point-process models.

**Architecture:** Build a small optimization layer around the current JAX Kalman and point-process code rather than rewriting the inference stack. Phase 1 optimizes a differentiable negative log-likelihood for linear-Gaussian models using JAX-compatible parameter containers and stable parameterizations for $A$, $Q$, and $R$. Phase 2 adds a generalized-EM style objective for point-process models with differentiable penalties, reusing the existing Laplace-EKF machinery and deferring custom VJPs until they are proven necessary.

**Tech Stack:** JAX, Optax, existing modules in `src/state_space_practice/kalman.py`, `src/state_space_practice/point_process_kalman.py`, `src/state_space_practice/switching_kalman.py`, pytest.

---

## Why This Revision

The original plan jumped directly to differentiating through the full smoother with `jax.custom_vjp`. That is not a good first milestone in this repository because:

1. The linear-Gaussian stack already exposes the pieces needed for an optimizer-based baseline.
2. The point-process stack uses PSD projection, line search, and Laplace approximations that make gradients more delicate than the original plan suggests.
3. `optax` is present in `environment.yml` but not in `pyproject.toml`, so dependency hygiene needs to be cleaned up before the feature is considered part of the package.

This revised plan starts with the tractable case, establishes stable parameterizations and tests, and postpones custom adjoints until profiling shows they are the bottleneck.

## Non-Goals for V1

- No `jax.custom_vjp` in the first implementation.
- No minibatched multi-session training API.
- No neural components or amortized inference.
- No claim that optimizer-based training will replace EM everywhere.

## Success Criteria

1. A linear-Gaussian model can be fit with Adam from the same initial parameters as EM.
2. Learned parameters remain stable and valid, with PSD covariance matrices throughout training.
3. Penalized optimization supports at least one differentiable regularizer that is awkward in closed-form EM.
4. Tests demonstrate parity with existing EM on a small synthetic benchmark.

### Task 1: Add the Optimization Scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `src/state_space_practice/optimization.py`
- Modify: `src/state_space_practice/__init__.py`
- Test: `src/state_space_practice/tests/test_optimization.py`

**Step 1: Add the missing package dependency**

Update `pyproject.toml` so the project dependency list includes `optax`, matching `environment.yml`.

**Step 2: Create optimizer parameter containers**

Add a small module with parameter containers and transforms such as:

```python
from typing import NamedTuple

import jax.numpy as jnp


class LinearGaussianParams(NamedTuple):
	transition_matrix: jnp.ndarray
	process_noise_tril: jnp.ndarray
	measurement_matrix: jnp.ndarray
	measurement_noise_tril: jnp.ndarray


def tril_to_covariance(cholesky_factor: jnp.ndarray) -> jnp.ndarray:
	cov = cholesky_factor @ cholesky_factor.T
	return 0.5 * (cov + cov.T)
```

Use a pytree-safe structure such as `NamedTuple` or nested dictionaries so the parameters work cleanly with `jax.jit`, `jax.value_and_grad`, and Optax. Use Cholesky-like parameterizations for $Q$ and $R$ so the optimizer never proposes invalid covariance matrices.

**Step 3: Write a minimal smoke test**

Add a test that instantiates the parameter container and verifies the covariance transform is symmetric positive semidefinite.

**Step 4: Run the test**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_optimization.py -v`

Expected: PASS for the covariance parameterization test.

### Task 2: Implement Linear-Gaussian Gradient-Based Fitting

**Files:**
- Modify: `src/state_space_practice/optimization.py`
- Reuse: `src/state_space_practice/kalman.py`
- Test: `src/state_space_practice/tests/test_optimization.py`

**Step 1: Add the differentiable loss**

Implement a loss that sums the one-step marginal log-likelihood terms produced by the Kalman filter:

```python
def linear_gaussian_neg_log_likelihood(params, observations, init_mean, init_cov):
	filtered_mean, filtered_cov, log_likelihood = kalman_filter(
		init_mean=init_mean,
		init_cov=init_cov,
		obs=observations,
		transition_matrix=params.transition_matrix,
		process_cov=tril_to_covariance(params.process_noise_tril),
		measurement_matrix=params.measurement_matrix,
		measurement_cov=tril_to_covariance(params.measurement_noise_tril),
	)
	del filtered_mean, filtered_cov
	return -jnp.sum(log_likelihood)
```

Use this exact likelihood instead of an ELBO for the first milestone.

**Step 2: Add an Optax train step**

Implement `make_linear_gaussian_train_step(optimizer)` that returns a JIT-compatible update function computing the loss and gradients with `jax.value_and_grad`.

**Step 3: Add a short fitting loop**

Implement `fit_linear_gaussian_with_gradient_descent(...)` with the following behavior:

- fixed number of iterations
- optional gradient clipping
- optional $L_2$ penalty on `transition_matrix`
- history tracking for loss values

**Step 4: Write failing tests first**

Add tests that verify:

- the loss is finite for valid inputs
- a few optimization steps reduce the loss on synthetic data
- optimized $Q$ and $R$ remain PSD

**Step 5: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_optimization.py -v`

Expected: PASS for linear-Gaussian loss and optimization tests.

### Task 3: Benchmark Against Existing EM

**Files:**
- Modify: `src/state_space_practice/optimization.py`
- Modify: `src/state_space_practice/tests/test_optimization.py`
- Reference: `src/state_space_practice/kalman.py`

**Step 1: Build an explicit reference EM loop**

Do not assume a ready-made top-level EM fitting function already exists. Implement a small helper such as `fit_linear_gaussian_with_em(...)` in `optimization.py` that explicitly alternates:

- `kalman_smoother(...)`
- `kalman_maximization_step(...)`

for a fixed number of iterations.

This helper is primarily for parity testing and should stay minimal.

**Step 2: Build a shared synthetic dataset fixture**

Use a deterministic synthetic linear state-space dataset so both EM and gradient descent start from the same initialization.

**Step 3: Add the parity test**

Write a test with loose but meaningful assertions:

- both methods improve likelihood versus initialization
- both methods recover parameters within a reasonable tolerance
- optimizer history improves overall, not necessarily every step

Avoid asserting near-identical parameter recovery unless the initialization and identifiability are tightly controlled.

**Step 4: Keep runtime bounded**

Cap the number of EM iterations and optimizer steps so this test remains unit-test sized rather than turning into a benchmark suite.

**Step 5: Run the targeted test**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_optimization.py::test_gradient_fit_matches_em_directionally -v`

Expected: PASS with both methods improving over initialization.

### Task 4: Add Differentiable Regularization

**Files:**
- Modify: `src/state_space_practice/optimization.py`
- Test: `src/state_space_practice/tests/test_optimization.py`

**Step 1: Add penalty hooks**

Support optional penalties in the loss function:

- `transition_l2_weight`
- `transition_l1_weight`
- `measurement_l2_weight`

Start with penalties on dense parameter matrices only. Do not add structured sparsity machinery yet.

**Step 2: Write tests for penalty behavior**

Add a test showing a nonzero $L_1$ or $L_2$ penalty shrinks the transition matrix norm relative to an unpenalized fit.

**Step 3: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_optimization.py -v`

Expected: PASS with penalty-specific assertions.

### Task 5: Extend Carefully to Point-Process Models

**Files:**
- Modify: `src/state_space_practice/optimization.py`
- Reference: `src/state_space_practice/point_process_kalman.py`
- Reference: `src/state_space_practice/switching_kalman.py`
- Test: `src/state_space_practice/tests/test_optimization.py`

**Step 1: Define the narrow objective**

Do not start with "differentiate through the full smoother." Start with a generalized-EM objective that reuses the current point-process filter or smoother outputs and adds differentiable penalties to the parameter update step.

**Step 2: Add a point-process loss wrapper**

Implement a loss that is explicit about approximation:

- either negative filtered Laplace log-likelihood
- or expected complete-data objective using fixed posterior summaries

Name it accordingly so the API does not imply exact ELBO optimization.

**Step 3: Add a stability gate**

Before exposing a public optimizer loop, add tests that verify gradients are finite on small synthetic point-process problems.

**Step 4: Stop if gradients are unstable**

If gradients become NaN or highly sensitive to line-search internals, document the blocker and open a follow-up plan for custom VJP or implicit differentiation. That is the point where deeper autodiff work becomes justified.

## Verification Checklist

- [ ] `optax` is declared in `pyproject.toml`
- [ ] Linear-Gaussian gradient fit reduces loss on synthetic data
- [ ] Covariance parameterization keeps matrices PSD
- [ ] Penalized loss changes parameter shrinkage as expected
- [ ] Directional parity with EM is demonstrated on a small benchmark
- [ ] Point-process extension is only considered successful if gradients are finite and reproducible

## Known Implementation Risks

- JAX container choice matters; use pytrees from the start rather than retrofitting them later.
- Linear-Gaussian EM parity requires an explicit reference loop, not just reuse of isolated filter and M-step functions.
- Point-process gradient stability may fail because of Laplace approximations, line search, and PSD projections.

## Deferred Research Extensions

- Custom VJP for Kalman smoother scans
- Custom VJP or implicit differentiation for Laplace-EKF updates
- Minibatched multi-session training
- Hybrid neural observation or transition models

## Expected Outcome

This plan yields a practical optimizer-based fitting path for the models already present in the repository. If successful, it will support regularized training where closed-form EM is awkward, without committing the project to a fragile end-to-end differentiable inference stack too early.
