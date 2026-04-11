# Square-Root Laplace-EKF Filter Investigation

> **Status:** DEFERRED. This plan captures the design and open questions for a future implementation. Do not execute without an explicit green-light from the user. See "When to un-defer" below.
>
> **Context:** This document is the output of a 2026-04-11 investigation into f32 numerical stability of `stochastic_point_process_filter`. Two failure modes ("Problem A" at T=1501 and "Problem B" at T=28,373) were traced to accumulated f32 roundoff in the information-form covariance propagation. The user decided to defer a square-root rewrite and proceed with (a) observability warnings, (b) f64 documentation, and (c) block-diagonal filter specialization. This memo documents what a square-root filter would look like if the deferred work ever becomes necessary.

---

## Goal

Make `stochastic_point_process_filter` and `stochastic_point_process_smoother` numerically stable in float32 across the full range of realistic CA1 place-cell configurations (T up to ~100k bins, per-neuron basis dimension up to 64, condition numbers up to ~1e6). Preserve bit-for-bit agreement with the current f64 information-form filter within `atol=1e-5` on equivalence tests.

The current filter is information-form: it stores the posterior covariance `P` directly, propagates via `A @ P @ A^T + Q`, and inverts via `psd_solve`. In f32 this accumulates roundoff at roughly `sqrt(T·d)·ε·max_eig` per bin, which crosses the `min_eig` boundary after ~250-2000 bins depending on conditioning. The symptom is negative eigenvalues in the predicted covariance, followed by a Cholesky failure and silent NaN.

The proposed fix is a square-root filter that stores `S` where `P = S S^T`. This gives two structural guarantees:

1. `S S^T` is PSD for any `S`. Negative eigenvalues are representable only by a `NaN` or `inf` in `S`, not by a "valid but nonsense" value. The filter cannot silently lose PSD.
2. `cond(S) = sqrt(cond(P))`. A `P` with `cond(P) = 1e8` corresponds to an `S` with `cond(S) = 1e4`, well within the range where f32 Cholesky and QR are numerically stable.

## Scope and success criteria

**In scope:**

- New square-root forward filter `_stochastic_point_process_filter_sqrt` alongside the existing information-form filter.
- New square-root RTS smoother `_stochastic_point_process_smoother_sqrt`.
- Opt-in via `filter_mode: Literal["dense", "block_diagonal", "sqrt"] = "dense"` on the public entry points. No auto-detection — square-root is an explicit mode because its tradeoffs differ from the dense path.
- Regression tests proving numerical equivalence with the f64 dense filter on a set of reference configurations.
- Benchmarks establishing the f32 sqrt-filter speed penalty vs the f64 dense filter at matched precision levels.

**Out of scope:**

- Integrating square-root with the block-diagonal filter. The two are orthogonal but composing them is a second implementation. Defer indefinitely.
- Square-root EM M-step. The M-step operates on smoothed covariances; the sqrt smoother would need to expose the dense covariance or a block-form depending on downstream consumers.
- Changing the public API of `filtered_cov` / `smoother_cov` beyond what's already planned for the block-diagonal filter.
- Making square-root the default. Too invasive for a future-proofing concern.

**Success criteria:**

1. `_stochastic_point_process_filter_sqrt` produces `filtered_mean` and `marginal_log_likelihood` matching the f64 dense filter to `atol=1e-5` on a set of equivalence tests covering single-neuron, multi-neuron, and long-sequence configurations.
2. `_stochastic_point_process_smoother_sqrt` similarly matches.
3. A regression test demonstrates the current f32 failure mode ("Problem A" reproducer: T=1500, d=36, cond=5e3, min_eig=2e-4) produces NaN in `filter_mode="dense"` but finite output in `filter_mode="sqrt"`.
4. Benchmark results are published alongside the implementation, including per-step microseconds for d in {16, 36, 64} and T in {5k, 30k}, both f32 and f64, across the three filter modes.

## When to un-defer

Execute this plan only when one or more of the following become true:

1. A user reports an f32 failure on a configuration that the block-diagonal filter cannot resolve (e.g. single very-ill-conditioned neuron with T > 50k, or a multi-neuron problem where `update_transition_matrix=True` breaks block-diagonality).
2. Memory pressure on the f64 code path becomes a blocker (the block-diagonal filter already mitigates this for the multi-neuron case; the remaining risk is single-neuron at very large `n_basis`).
3. A real use case appears where f32 speed is worth its own implementation tier — typically GPU workloads where memory bandwidth dominates compute.
4. The user proactively decides square-root is worth building as a learning exercise or for general infrastructure hardening.

---

## Algorithm

### Notation

- `x_t ∈ R^d`: latent state at time t (spline weights)
- `P_t ∈ R^(d×d)`: posterior covariance, PSD
- `S_t ∈ R^(d×d)`: Cholesky-like factor with `P_t = S_t S_t^T`. Lower triangular by convention.
- `A ∈ R^(d×d)`: transition matrix (default: I)
- `Q ∈ R^(d×d)`: process noise covariance, PSD
- `L_Q ∈ R^(d×d)`: Cholesky factor of Q, `Q = L_Q L_Q^T`
- `Z_t ∈ R^(n_obs × d)`: Jacobian of log-intensity at prior mode (Fisher-scoring linearization)
- `y_t ∈ R^n_obs`: spike counts
- `λ_t ∈ R^n_obs`: expected counts per bin (`exp(log_rate) * dt`), clipped by `max_log_count`
- `W_t ∈ R^(n_obs×n_obs)`: diagonal observation "noise" precision, `W_t = diag(λ_t)` (Fisher info per observation)

### Forward filter: one step

**Predict (square-root form).** Compute `S_pred` such that `S_pred S_pred^T = A S_prev S_prev^T A^T + Q`.

The standard trick: stack `[A S_prev, L_Q]` into a `(d, 2d)` matrix and take its QR decomposition. The resulting `R` is the upper-triangular factor whose product `R R^T` equals the sum `A P_prev A^T + Q`.

```python
# Both A @ S_prev and L_Q are (d, d); stack row-wise to get (d, 2d).
M_pred = jnp.concatenate([A @ S_prev, L_Q], axis=1)  # (d, 2d)
# QR on (d, 2d): Q_factor is (d, d), R is (d, 2d). We only need R's first d cols.
_, R_pred = jnp.linalg.qr(M_pred.T, mode="reduced")  # QR on (2d, d) → R (d, d)
S_pred = R_pred.T  # lower triangular, (d, d)
```

Why it works: if `M @ M^T = A S_prev S_prev^T A^T + L_Q L_Q^T`, then `M = Q_factor @ R` gives `M @ M^T = R^T @ R`, so `S_pred = R^T` is a valid square root. PSD is structural because we never form the sum explicitly — we only factor a matrix whose Gram is the target.

**Laplace update (Fisher-scoring step in square-root form).** The information-form update is:

```
post_prec = prior_prec + Z^T W Z
post_cov = inv(post_prec)
```

The square-root equivalent uses the **array form** of the update (Kailath, Sayed & Hassibi 2000, §12):

```
[S_pred^T                     0      ]       [S_pred_new^T   K_new^T]
[W^{1/2} Z S_pred^T    W^{1/2}]  QR→  [   0         U^T    ]
```

where QR is a unitary rotation that triangularizes the left block. The result is:

- `S_pred_new^T` is the upper-triangular factor of the posterior (before the cov update).
- The gain `K = S_prev Z^T W^{1/2} U^{-T} U^{-1}` emerges from the rotation.

In practice, implementation is more direct using the **potter form** of the square-root update. For each observation (or per-bin observation vector), apply a rank-k update to `S_pred` via a sequence of Givens rotations.

**Simpler alternative for our use case**: because our observation "noise" `W = diag(λ·dt)` is diagonal, we can update `S_pred` one observation at a time using the **rank-one square-root update**:

```python
# Rank-one update: new P = P - u u^T where u captures the observation gain.
# Maps to a rank-one update on S via QR on [S; u^T].

for j in range(n_obs):
    z_j = Z[j]                              # (d,)
    w_j_sqrt = jnp.sqrt(λ[j])               # scalar
    # Stack prior factor and observation row, then QR to get new factor.
    M = jnp.concatenate([S_pred.T, (w_j_sqrt * z_j)[None, :]], axis=0)  # (d+1, d)
    _, R = jnp.linalg.qr(M)                 # (d, d) upper triangular
    S_pred = R.T                            # new posterior factor
```

This is a sequence of `n_obs` rank-one updates. For single-neuron (`n_obs=1`), one QR per step. For multi-neuron (`n_obs=N`), N QRs per step.

**Mean update.** The mean update is unchanged from the information form:

```python
# Fisher scoring: post_mean = pred_mean + K @ innovation
# where K = S_post @ S_post^T @ Z^T  (since post_cov @ Z^T = K)
innovation = y - λ
K = (S_post @ S_post.T) @ Z.T              # (d, n_obs); uses post, not pred
post_mean = pred_mean + K @ innovation
```

Note: the gain `K` materializes `S_post @ S_post^T` internally, which is `P_post`. This is PSD by construction (it's the outer product of `S_post` with itself), so no stability issue here — the savings come from never letting `P_post` accumulate roundoff across steps, not from never forming it.

**Log-likelihood (Laplace normalization in square-root form).** The Laplace correction is:

```
log p(y_t | y_{1:t-1}) ≈ log p(y_t | x*_t)
                         + log p(x*_t | y_{1:t-1})
                         + 0.5 * log |P_post|
                         - 0.5 * log |P_prior|    ← wait, this is already in the formula
```

In the current code ([point_process_kalman.py:388-396](src/state_space_practice/point_process_kalman.py#L388-L396)):

```python
logdet_prior = _logdet_psd(one_step_cov, diagonal_boost)
logdet_post = _logdet_psd(posterior_cov, diagonal_boost)
log_likelihood = log_lik_poisson - 0.5*quad - 0.5*logdet_prior + 0.5*logdet_post
```

In the square-root form, `logdet(P) = 2 * sum(log(diag(S)))`, exactly and without any Cholesky because we have `S` already:

```python
logdet_prior = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(S_pred))))
logdet_post = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(S_post))))
```

This is a **side benefit**: no Cholesky factorization inside `_logdet_psd` at all. The logdet computation becomes a single reduction over the diagonal of a triangular matrix. This is the cheapest part of the entire update.

### Backward smoother (RTS)

The RTS smoother backward pass is trickier in square-root form because it involves `inv(P_{t+1|t})` and a cross-covariance. Two standard approaches:

**Option 1: Modified Bryson-Frazier smoother.** Store the information form backwards — propagate `J_t = P_{t|t} A^T inv(P_{t+1|t})` implicitly via a square-root update on the forward pass. Reference: Bierman (1977) §8.

**Option 2: Square-root RTS (Kailath §10.5).** Explicitly compute the cross-gain `J_t = P_{t|t} A^T S_{pred_{t+1}}^{-T} S_{pred_{t+1}}^{-1}` using triangular solves against `S_{pred}`, then propagate smoothed means and covariance factors:

```python
# Backward step:
# J_t = (P_{t|t} A^T) @ inv(P_{t+1|t})
#     = (S_{t|t} S_{t|t}^T A^T) @ (S_pred_{t+1} S_pred_{t+1}^T)^{-1}
#     = S_{t|t} S_{t|t}^T A^T S_pred_{t+1}^{-T} S_pred_{t+1}^{-1}

# Solve against S_pred (lower triangular)
temp = jax.scipy.linalg.solve_triangular(S_pred_next, A @ S_filt.T, lower=True)
J = S_filt @ jax.scipy.linalg.solve_triangular(
    S_pred_next.T, temp, lower=False
).T

# Smoothed mean
m_smooth = m_filt + J @ (m_smooth_next - A @ m_filt)

# Smoothed covariance factor via QR
# P_{t|T} = P_{t|t} - J (P_{t+1|t} - P_{t+1|T}) J^T
# Expressed as S_{t|T} via stacking and QR:
M = jnp.concatenate([
    S_filt.T,               # (d, d)  — positive part P_{t|t}
    (J @ S_pred_next).T,    # (d, d)  — to be subtracted
    (J @ S_smooth_next).T,  # (d, d)  — positive part P_{t+1|T}
], axis=0)  # (3d, d)
# The "subtract" part is the issue — standard QR doesn't handle sign changes.
# The hyperbolic rotation trick or a explicit "downdate" step is required.
```

**Subtlety**: the smoothed covariance factor has a subtraction that standard QR cannot express. The canonical answer is to use **hyperbolic Householder rotations** for the downdate step, which requires a custom implementation (JAX does not provide `jax.lax.hyperbolic_qr` — this is the most time-consuming part of the project).

Alternative: use an unstable but simpler approach — explicitly compute `P_{t|T}` in the dense form for the smoother (accepting the roundoff risk only in the backward pass, which is shorter than the forward pass and has less accumulation because the backward pass runs over the already-smoothed forward posteriors). This would be a **partial square-root smoother**: forward pass in sqrt form, backward pass in dense form. Less elegant but dramatically simpler to implement. The backward pass's roundoff accumulation is bounded by `sqrt(T·d)·ε·max_eig` — exactly the same formula as the forward pass, but only if the backward pass operates in the same precision regime.

**Recommendation for implementation**: start with the partial square-root smoother. The forward pass is where the empirical failures occur (Problem A and Problem B both NaN in the forward pass). The backward pass is shorter and runs on already-smoothed Gaussians with more uniform conditioning. Implement the full square-root smoother only if the partial one shows measurable roundoff problems in practice.

---

## Open questions (must be answered before implementation)

### Q1: QR-predict speed penalty in f32

**The core question**: is a square-root filter in f32 actually faster than the current dense filter in f64 at matched correctness, or do the extra QR factorizations eat the precision-halving speedup?

**What's known**:
- Current f64 dense filter: ~279 μs/step at d=36, T=10k (post cholesky-logdet optimization; see [point_process_kalman.py:79](src/state_space_practice/point_process_kalman.py#L79)).
- Theoretical f32 speedup from halved precision: ~2x on CPU (memory bandwidth bound), up to ~4x on GPU (if the matrix sizes are right for Tensor Cores).
- Per-step QR on (d, 2d) matrices: roughly 3-5x the cost of a single `psd_solve` on (d, d) by flop count, because QR is `~4d³/3` flops vs Cholesky solve `~d³/3`.

**What's unknown**:
- XLA's fusion behavior on the QR-predict → rank-one update → QR-update pipeline. In the best case, XLA fuses the whole scan body and the measured cost is much less than the flop count suggests. In the worst case, each QR is its own kernel launch, and the GPU path becomes launch-latency-bound.
- Whether JAX's `jnp.linalg.qr` differentiates through efficiently. The Fisher scoring loop is inside `jax.grad` for `fit_sgd`, so differentiable QR is mandatory. JAX does provide a gradient rule for QR, but it's not as well-optimized as the Cholesky path.
- The practical condition number of `S` on real CA1 data after a warm-start. If `cond(S) < 1e3` in practice, f32 is safe. If `cond(S) ≈ 1e4-5`, f32 is borderline. The investigation should establish this on a real dataset.

**What would settle it**: a microbenchmark implementing just the predict step in isolation, both as (f64 dense cholesky solve) and (f32 square-root QR), running the same 10k-step scan and comparing wall time. This is a ~1-day investigation, not a full implementation.

**Rough prediction** (without running the benchmark): f32 square-root is **roughly break-even to 2x slower than f64 dense** on CPU for d=36. At d=64, the QR penalty compounds and it's probably 2-3x slower. On GPU, the answer depends entirely on XLA fusion quality — could be anywhere from 3x faster to 2x slower.

**If this turns out poorly**, the square-root filter is still valuable as a correctness fallback for f32 users, but not as a speed optimization. The marketing becomes "numerical robustness mode" rather than "fast mode."

### Q2: How much precision do we need, really?

Current observation: f32 with a well-warmed-up prior and T<1000 bins works fine. The failures happen at T>250 with poorly-conditioned cov, or T>28k regardless of starting condition number.

Alternative to square-root: **periodic PSD re-projection every K bins**. Every K time steps, force `S = chol(max(eigvalsh(P), epsilon))`. This costs one eigendecomposition per K bins — cheap if K ~ 100. Gives a bounded-roundoff filter without the square-root rewrite. Downsides: not differentiable-through-cleanly (`eigvalsh` gradient is pathological near degenerate eigenvalues), and the projection introduces a small discontinuity that can confuse the Laplace normalization's logdet calculation.

This should be **benchmarked against the square-root filter** as a cheaper alternative. If periodic re-projection gets us to T=30k in f32 with acceptable accuracy, the square-root rewrite is not needed.

### Q3: Partial vs full square-root smoother

As discussed in the smoother section: the partial approach (sqrt forward, dense backward) is dramatically simpler to implement and probably sufficient for the failure modes we've seen. The full sqrt approach requires hyperbolic rotations which are not in JAX.

Decision can be deferred to implementation time. Start with partial, escalate to full only if the backward pass shows its own numerical issues.

### Q4: Integration with block-diagonal filter

The block-diagonal filter (Plan B from 2026-04-11-psd-robustness-and-block-filter.md) operates at per-neuron `nb=36` scale. Each per-neuron block is small enough that f32 viability extends from T~250 in the dense-state filter to T~2000-5000 in the block filter, without any square-root rewrite.

**The block-diagonal filter partially obviates the need for square-root.** The remaining f32 risk is:
- Single-neuron at very high `n_basis` (unlikely in practice)
- Multi-neuron at very long T (>10k) even at per-block scale
- Any configuration with `update_transition_matrix=True` breaking block-diagonality

If the block filter is implemented first and proves sufficient for all realistic cases, the square-root filter can be deferred indefinitely.

### Q5: Where does the sqrt filter live in the API?

Options:
- `stochastic_point_process_filter(..., filter_mode="sqrt")` — explicit mode on the existing entry point
- `stochastic_point_process_filter_sqrt(...)` — separate function
- `PlaceFieldModel(..., numerical_mode="sqrt")` — class-level knob

**Recommendation**: `filter_mode` on the existing entry point. Same dispatch pattern as the block-diagonal filter. Auto-fallback is NOT appropriate for square-root (its output is mathematically equivalent but with different numerical behavior — user should opt in explicitly).

---

## Implementation outline

**Estimated total effort**: 3-5 days of focused implementation if the open questions have clean answers. Double that if Q1 (QR speed) or Q3 (partial vs full smoother) require iteration.

### Phase 1: Microbenchmark and precision audit (1 day)

Before writing any filter code, answer Q1 and Q2.

- **Microbenchmark 1 (predict step in isolation)**: write a standalone script that runs 10k predict-only steps in f64 dense form and f32 square-root form, measures wall time, and reports the ratio. Run on CPU and GPU if available.
- **Microbenchmark 2 (periodic re-projection)**: add a third variant — f32 dense with `eigvalsh`-based re-projection every K=100 bins. Measure both speed and accuracy vs the f64 reference on a reference trajectory.
- **Precision audit**: on a real CA1 dataset, run the f32 filter to NaN and log `cond(P)`, `cond(S)`, and `min_eig(P)` just before the failure. This establishes the working condition-number regime for the sqrt filter.

**Decision gate**: if the microbenchmark shows f32 sqrt is ≤ 1.5x the cost of f64 dense at matched correctness, proceed to implementation. If it shows 3x+ slowdown and periodic re-projection handles the real-data cases, use re-projection instead and abandon the sqrt plan.

### Phase 2: Forward filter in square-root form (1-2 days)

- Implement `_stochastic_point_process_filter_sqrt` alongside the existing `stochastic_point_process_filter`.
- Internally uses the `S` factor throughout; materializes `P = S @ S.T` only in the returned `filtered_cov` at the end (single allocation per filter call, not per step).
- Use the rank-one QR update per observation for the measurement step (simpler than batched triangular solves for small `n_obs`).
- Compute `logdet` directly from `diag(S)` — free vs the existing `_logdet_psd` path.
- Keep the marginal-log-likelihood math identical to the information form.

### Phase 3: Smoother (partial square-root) (1 day)

- Implement `_stochastic_point_process_smoother_sqrt` that runs the sqrt forward filter, then a **dense backward pass** on the already-computed filtered Gaussians.
- Proof of numerical equivalence: the backward pass operates on forward posteriors that were never allowed to lose PSD, so its input is always well-conditioned. The backward pass's own roundoff accumulation is much smaller than the forward pass's in practice.

### Phase 4: Dispatch and API (half day)

- Add `filter_mode: Literal["dense", "sqrt"]` to `stochastic_point_process_filter` and `stochastic_point_process_smoother`. (Block-diagonal mode is orthogonal and gets its own parameter or auto-detection.)
- Same parameter on `PlaceFieldModel.fit`, `fit_sgd`, with pass-through.
- Default: `"dense"` — no silent behavior change.

### Phase 5: Tests (1 day)

- **Equivalence tests**: `test_sqrt_filter_matches_dense_f64` across single-neuron, multi-neuron, short/long T. Assert `filtered_mean`, `marginal_log_likelihood`, and `filtered_cov` agree with `atol=1e-5` in f64 and `atol=1e-3` in f32 (reflecting the precision floor).
- **Regression tests**: "Problem A" and "Problem B" reproducers. Assert they NaN in `filter_mode="dense"` with f32, and produce finite output in `filter_mode="sqrt"` with f32.
- **Gradient tests**: `jax.grad` through both paths must agree in f64. Critical for `fit_sgd` correctness.
- **Smoother tests**: equivalence on same problem sizes.
- **Benchmark run**: per-step microseconds for d in {16, 36, 64}, T in {5k, 30k}, dtype in {f32, f64}, mode in {dense, sqrt}. Store results in this plan document.

### Phase 6: Documentation (half day)

- Update `CLAUDE.md` numerical precision section with when to use `filter_mode="sqrt"`.
- Docstring notes on `PlaceFieldModel` class docstring and the filter entry points.
- Example in a notebook showing the f32 failure-then-recovery with the sqrt mode.

---

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| QR-predict is 3x+ slower than f64 dense, killing the speed story | Medium | High | Microbenchmark in Phase 1 before committing to implementation |
| `jax.grad` through QR has accuracy or speed issues | Medium | High | Test early in Phase 2; fall back to Laplace-form Jacobian if needed |
| Full sqrt smoother needs hyperbolic rotations | High | Medium | Use partial sqrt smoother (sqrt fwd + dense bwd) |
| Integration with block-diagonal filter is non-trivial | High | Low | Explicitly out of scope; document as future work |
| Current f32 failures are 99% solved by block-diagonal filter, making sqrt filter unnecessary | High | Low | Explicitly defer this plan; pick it up only when block-diagonal is proven insufficient |

---

## References

- Bierman, G.J. (1977). *Factorization Methods for Discrete Sequential Estimation*. Academic Press. §5-8 (forward filter), §10 (smoother).
- Kailath, T., Sayed, A.H., & Hassibi, B. (2000). *Linear Estimation*. Prentice Hall. §12 (array form), §14 (square-root filters and smoothers).
- Morf, M. & Kailath, T. (1975). "Square-root algorithms for least-squares estimation." *IEEE Transactions on Automatic Control* 20(4), 487-497.
- Grewal, M.S. & Andrews, A.P. (2014). *Kalman Filtering: Theory and Practice Using MATLAB*, 4th ed. Chapter 7 (square-root filters).
- Särkkä, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press. §5 (square-root form).

---

## Decision log

- **2026-04-11**: Document created, investigation deferred. The user chose to proceed with observability warnings (Plan A), block-diagonal filter specialization (Plan B), and f64 documentation. The square-root filter is preserved as a concrete future option with open questions logged above. Revisit when block-diagonal + f64 proves insufficient for a realistic use case.
