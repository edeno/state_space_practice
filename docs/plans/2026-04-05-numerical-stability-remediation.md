# Numerical Stability Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the highest-risk correctness and numerical stability problems in the Kalman, switching Kalman, and point-process code paths without changing the intended model semantics.

**Architecture:** Treat this as a targeted stabilization pass, not a redesign. First, lock in the current failure modes with focused regression tests. Then centralize covariance stabilization, make the point-process exponential path overflow-safe, fix the public firing-rate convention, repair diagnostic log handling for exact-zero probabilities, and finally bring the legacy point-process implementation up to the same numerical standard or explicitly deprecate it.

**Tech Stack:** JAX, existing Kalman helpers in `src/state_space_practice/kalman.py`, switching EM utilities in `src/state_space_practice/switching_kalman.py`, point-process inference in `src/state_space_practice/point_process_kalman.py`, legacy point-process code in `src/state_space_practice/models.py`, pytest.

---

## Scope and Success Criteria

This plan fixes the issues identified in the static review:

1. EM M-step covariance outputs can become indefinite and are sometimes reused without PSD enforcement.
2. The main Laplace point-process update uses raw exponentials that can overflow.
3. `PointProcessModel.get_rate_estimate()` applies the wrong `dt` convention.
4. ELBO / entropy diagnostics use a floored log that distorts exact-zero probability behavior.
5. The older point-process implementation in `models.py` is materially less stable than the newer implementation.

Success means all of the following are true:

- M-step covariance outputs are PSD by construction in the shared helpers.
- High-log-rate point-process updates remain finite.
- Public rate reporting matches the model definition: `exp(log_rate)` is the rate in Hz.
- Diagnostic terms handle zero probabilities by masking contributions, not by pretending zero has a small finite log.
- The legacy point-process path is either numerically aligned with the main implementation or explicitly fenced off.

## Verification Gates

Run these in the conda environment from `CLAUDE.md`.

Targeted tests:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py \
  src/state_space_practice/tests/test_point_process_kalman.py \
  src/state_space_practice/tests/test_models.py -v
```

Neighbor regression tests:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_oscillator_models.py \
  src/state_space_practice/tests/test_switching_point_process.py -v
```

Lint gate:

```bash
conda run -n state_space_practice ruff check src/
```

Stop conditions:

- If PSD enforcement changes enough outputs to break existing recovery tests, stop and compare whether the tests were implicitly depending on indefinite covariances.
- If overflow-safe clipping changes model semantics beyond numerical saturation cases, stop and tighten the clipping policy instead of broadening it.
- If `models.py` is still used by downstream code and cannot be safely aligned in one pass, stop and deprecate it explicitly rather than leaving two inconsistent implementations.

## Task Order

Execute in this order:

1. Add regression tests for each reviewed bug.
2. Centralize covariance stabilization and apply it in all EM helpers.
3. Harden the point-process exponential path.
4. Fix the public rate API and its docs.
5. Repair zero-probability diagnostic handling.
6. Align or deprecate the legacy `models.py` point-process path.
7. Run full verification and update inline docs where conventions changed.

---

### Task 1: Lock In the Reviewed Failures With Regression Tests

**Files:**
- Modify: `src/state_space_practice/tests/test_kalman.py`
- Modify: `src/state_space_practice/tests/test_switching_kalman.py`
- Modify: `src/state_space_practice/tests/test_point_process_kalman.py`
- Modify: `src/state_space_practice/tests/test_models.py`

**Step 1: Add PSD regression tests for non-switching EM helpers**

Add tests in `test_kalman.py` that call `kalman_maximization_step()` on smoothed statistics engineered to be near-singular or slightly inconsistent, then assert:

```python
eigvals_q = jnp.linalg.eigvalsh(Q_est)
eigvals_r = jnp.linalg.eigvalsh(R_est)

assert jnp.all(jnp.isfinite(Q_est))
assert jnp.all(jnp.isfinite(R_est))
assert jnp.min(eigvals_q) >= -1e-8
assert jnp.min(eigvals_r) >= -1e-8
```

**Step 2: Add PSD regression tests for switching EM helpers**

Add tests in `test_switching_kalman.py` that call `switching_kalman_maximization_step()` with low-occupancy discrete-state weights and assert each state slice of `Q` and `R` is symmetric, finite, and PSD within tolerance.

**Step 3: Add overflow regression tests for the main Laplace update**

Add tests in `test_point_process_kalman.py` that use a log-intensity function returning very large positive values, for example `80.0`, `100.0`, and `200.0`, then assert the update returns finite posterior mean, covariance, and log-likelihood.

Suggested assertion shape:

```python
posterior_mean, posterior_cov, log_lik = _point_process_laplace_update(...)

assert jnp.all(jnp.isfinite(posterior_mean))
assert jnp.all(jnp.isfinite(posterior_cov))
assert jnp.isfinite(log_lik)
```

**Step 4: Add a rate-convention regression test**

Add a test in `test_point_process_kalman.py` that fits or stubs a `PointProcessModel`, evaluates a known `log_rate`, and asserts the returned rate is `exp(log_rate)` rather than `exp(log_rate) / dt`.

**Step 5: Add diagnostic zero-probability regression tests**

Add tests in `test_switching_kalman.py` for `compute_expected_complete_log_likelihood()`, `compute_posterior_entropy()`, or `compute_elbo()` where some discrete probabilities are exactly zero. Assert the outputs remain finite and do not depend on the particular log-floor constant.

**Step 6: Add a legacy path regression test**

Add a test in `test_models.py` that drives the old `models.stochastic_point_process_filter()` with large positive log-rates and checks that the posterior covariance remains finite and symmetric.

**Step 7: Run targeted tests and confirm they fail for the right reasons**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py \
  src/state_space_practice/tests/test_point_process_kalman.py \
  src/state_space_practice/tests/test_models.py -v
```

Expected before implementation:

- PSD tests fail on negative eigenvalues or non-finite solves.
- Overflow tests fail with `inf` / `nan` outputs.
- Rate-convention test fails because the returned value is off by a factor of `1 / dt`.

---

### Task 2: Centralize Covariance Stabilization and Use It in Every EM M-Step

**Files:**
- Modify: `src/state_space_practice/kalman.py`
- Modify: `src/state_space_practice/switching_kalman.py`
- Modify: `src/state_space_practice/point_process_kalman.py`
- Modify: `src/state_space_practice/oscillator_models.py`

**Step 1: Add one shared covariance-stabilization helper in `kalman.py`**

Implement a helper with semantics like:

```python
def stabilize_covariance(
    cov: jax.Array,
    min_eigenvalue: float = 1e-8,
) -> jax.Array:
    cov = symmetrize(cov)
    eigvals, eigvecs = jnp.linalg.eigh(cov)
    eigvals = jnp.maximum(eigvals, min_eigenvalue)
    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T
```

Do not keep multiple independent PSD-fix policies unless there is a model-specific reason.

**Step 2: Use the helper in `kalman_maximization_step()`**

Replace the bare `symmetrize()` post-processing for `measurement_cov` and `process_cov` in `kalman.py` with the shared stabilizer.

**Step 3: Use the helper in `point_process_kalman.kalman_maximization_step()`**

Replace the bare `symmetrize()` process covariance post-processing there as well.

**Step 4: Use the helper in switching covariance solves**

Update `cov_solve_per_discrete_state` in `switching_kalman.py` so each returned covariance slice is stabilized, not just symmetrized.

Suggested structure:

```python
cov_solve_per_discrete_state = jax.vmap(
    lambda x, y, z, n: stabilize_covariance((x - y @ z.T) / n),
    in_axes=(-1, -1, -1, -1),
    out_axes=-1,
)
```

**Step 5: Remove caller-side raw reuse where appropriate**

In `oscillator_models.py`, stop depending on raw `Q` and `R` being safe by accident. After the shared helper is in place, callers can still assign directly, but add a lightweight assertion or comment that those matrices are already PSD-stabilized by the M-step helper.

**Step 6: Run the PSD-focused tests**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py -v
```

Expected: the new PSD tests pass, and no old tests fail because of asymmetric or indefinite covariance outputs.

---

### Task 3: Make the Point-Process Exponential Path Overflow-Safe

**Files:**
- Modify: `src/state_space_practice/point_process_kalman.py`
- Modify: `src/state_space_practice/switching_point_process.py` if helper reuse is beneficial
- Modify: `src/state_space_practice/tests/test_point_process_kalman.py`

**Step 1: Add a helper that converts log-rate to expected count safely**

Implement a helper in `point_process_kalman.py` that works on the count scale:

```python
def safe_expected_count(log_rate: Array, dt: float, max_log_count: float = 20.0) -> Array:
    log_count = log_rate + jnp.log(dt)
    return jnp.exp(jnp.clip(log_count, -20.0, max_log_count))
```

Use a single policy everywhere in this module so the posterior objective, Newton step, and reported mode likelihood all agree.

**Step 2: Replace raw exponentials inside `_point_process_laplace_update()`**

Use the new helper in these locations:

- `_neg_log_posterior()`
- `_newton_step_at()`
- the single-step `max_newton_iter == 1` branch
- the mode likelihood calculation after the posterior mean is computed

**Step 3: Keep gradients and Hessians consistent with the clipped count path**

Do not mix clipped counts in one branch and unclipped counts in another. If clipping is active, the same effective count should be used consistently in gradient, Hessian, and likelihood terms. If needed, treat the clip as a saturation policy rather than an exact derivative of the unconstrained model.

**Step 4: Revisit Laplace log-determinant handling**

Replace the current pattern that silently returns `0.0` on non-positive `slogdet` sign with a safer PSD-stabilized path. Preferred options:

1. stabilize the covariance first, then call `slogdet`, or
2. compute the log-determinant from clipped eigenvalues.

Avoid silently hiding indefinite matrices.

**Step 5: Reuse the same safe count helper in GLM-related code only if it simplifies consistency**

`switching_point_process.py` already clips `eta` in several places. Reuse a shared helper only if it reduces duplicated policy without widening scope.

**Step 6: Run point-process targeted tests**

Run:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_point_process_kalman.py -v
```

Expected: overflow regression tests pass and existing point-process tests remain finite.

---

### Task 4: Fix the Public Firing-Rate Convention

**Files:**
- Modify: `src/state_space_practice/point_process_kalman.py`
- Modify: `src/state_space_practice/tests/test_point_process_kalman.py`

**Step 1: Fix the implementation of `PointProcessModel.get_rate_estimate()`**

Change the method so it returns firing rate in Hz:

```python
rate = jnp.exp(log_rate)
```

Do not divide by `self.dt`; that quantity is expected count per bin, not rate.

**Step 2: Fix the docstring and any nearby comments**

Update the method docs so they match the model definition used everywhere else:

- `exp(log_rate)` = rate in Hz
- `exp(log_rate) * dt` = expected count in a bin

**Step 3: Run the rate-specific tests**

Run:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_point_process_kalman.py -k rate -v
```

Expected: the regression test passes and no other public API rate tests break.

---

### Task 5: Repair Exact-Zero Probability Handling in ELBO and Entropy Diagnostics

**Files:**
- Modify: `src/state_space_practice/switching_kalman.py`
- Modify: `src/state_space_practice/tests/test_switching_kalman.py`

**Step 1: Separate “safe log for filtering” from “masked contribution for diagnostics”**

Keep filtering and HMM-forward accumulation logic isolated from ELBO / entropy logic. For diagnostics, stop using a single floored `_safe_log()` as a universal tool.

Introduce a contribution helper with semantics like:

```python
def masked_xlogy(weights: jax.Array, probs: jax.Array) -> jax.Array:
    return jnp.where(weights > 0, weights * jnp.log(probs), 0.0)
```

Use a probability floor only where a true logarithm of an accumulated scalar is unavoidable, such as `log(predictive_likelihood_term_sum)` in the filter.

**Step 2: Guard the filtering path against complete underflow**

Before taking `jnp.log(predictive_likelihood_term_sum)` in the filter, clamp the scalar to a tiny positive minimum.

Suggested pattern:

```python
predictive_likelihood_term_sum = jnp.maximum(predictive_likelihood_term_sum, 1e-300)
marginal_log_likelihood += ll_max + jnp.log(predictive_likelihood_term_sum)
```

Choose a floor compatible with the repo’s active dtype policy.

**Step 3: Replace floored-log usage in ELBO and entropy terms**

Update these sections to use masked contributions instead of `_safe_log()` when the corresponding probability weight is zero:

- initial discrete-state term
- transition term
- entropy terms over state marginals and conditionals

**Step 4: Keep `_safe_log()` only if it still has one clear responsibility**

If it remains useful for a specific path, narrow its scope and rename or document it accordingly. Otherwise remove it.

**Step 5: Run the switching diagnostic tests**

Run:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_switching_kalman.py -k "elbo or entropy or likelihood" -v
```

Expected: exact-zero probability tests pass and existing ELBO tests remain finite.

---

### Task 6: Align or Fence Off the Legacy Point-Process Filter in `models.py`

**Files:**
- Modify: `src/state_space_practice/models.py`
- Modify: `src/state_space_practice/tests/test_models.py`
- Optionally modify: `README.md` or module docstrings if deprecating

**Step 1: Decide the narrowest safe path**

Preferred order:

1. If `models.py` is still part of active usage, align its numerics with the new helper patterns.
2. If it is legacy and redundant, mark it deprecated and direct users to `point_process_kalman.py`.

Do not leave it as a silently weaker implementation.

**Step 2: If keeping it, replace `pinv`-based covariance updates**

Replace:

```python
jnp.linalg.pinv(one_step_variance)
jnp.linalg.pinv(inverse_posterior_covariance)
```

with the same PSD-aware solve strategy used in `point_process_kalman.py`, plus symmetry enforcement and safe expected-count handling.

**Step 3: If keeping it, reuse shared helpers instead of copying math again**

If practical, refactor it to call the generalized helper in `point_process_kalman.py` rather than maintaining two separate Newton update implementations.

**Step 4: If deprecating instead, add a clear warning and test the warning path**

Document which replacement API to use and keep only the minimal compatibility layer.

**Step 5: Run legacy tests**

Run:

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_models.py -v
```

Expected: either the numerics are finite and symmetric, or the module clearly signals deprecation with tested behavior.

---

### Task 7: Final Verification and Documentation Cleanup

**Files:**
- Modify: `src/state_space_practice/point_process_kalman.py`
- Modify: `src/state_space_practice/kalman.py`
- Modify: `src/state_space_practice/switching_kalman.py`
- Modify: any touched test files

**Step 1: Review docstrings for conventions that changed**

Specifically confirm:

- rate vs expected count language is consistent
- covariance outputs are documented as PSD-stabilized when relevant
- overflow behavior is described as numerical saturation, not model semantics

**Step 2: Run the full targeted test bundle**

Run:

```bash
conda run -n state_space_practice pytest \
  src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py \
  src/state_space_practice/tests/test_point_process_kalman.py \
  src/state_space_practice/tests/test_models.py \
  src/state_space_practice/tests/test_oscillator_models.py \
  src/state_space_practice/tests/test_switching_point_process.py -v
```

**Step 3: Run lint**

Run:

```bash
conda run -n state_space_practice ruff check src/
```

**Step 4: Record follow-up items explicitly instead of sneaking them into this pass**

Possible follow-ups to defer unless a test forces them now:

- make covariance floors dtype-aware and model-scale-aware
- unify switching-point-process and non-switching point-process clipping helpers
- benchmark the effect of clipping on EM convergence behavior

**Step 5: Commit in focused chunks**

Suggested commit sequence:

```bash
git add src/state_space_practice/tests/test_kalman.py \
  src/state_space_practice/tests/test_switching_kalman.py \
  src/state_space_practice/tests/test_point_process_kalman.py \
  src/state_space_practice/tests/test_models.py
git commit -m "test: add numerical stability regressions"

git add src/state_space_practice/kalman.py \
  src/state_space_practice/switching_kalman.py \
  src/state_space_practice/point_process_kalman.py \
  src/state_space_practice/oscillator_models.py
git commit -m "fix: stabilize covariance updates and point-process numerics"

git add src/state_space_practice/models.py
git commit -m "fix: align legacy point-process numerics"
```

---

## Notes for the Implementer

- Fix the shared helpers before patching downstream model classes.
- Do not weaken existing recovery tests by broadening tolerances unless the old behavior was numerically invalid.
- Prefer one stabilization policy per mathematical object class:
  - one covariance PSD policy
  - one expected-count saturation policy
  - one exact-zero masking policy for diagnostics
- If a fix is only needed in diagnostics, keep it out of the inference path.
