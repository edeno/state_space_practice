# Gaussian Sum Filter / Smoother for Replay Decoding

> **Status:** PLANNED as of 2026-04-17. Follow-on to the position-decoder work.
> **Revision:** 2026-04-17, post-review. V1 scope tightened: prune-only (no merging), no built-in EM-GMM initializer, explicit smoother formula, SPD placeholders for inactive slots.

**Goal:** Add a Gaussian Sum Filter / Smoother (GSF/GSS) as a continuous-state alternative to the Laplace-EKF `position_decoder` for replay-like decoding, where the posterior is multi-modal and per-bin expected counts are sparse. Grid-based decoding is handled elsewhere in the lab's toolchain and is ruled out here on memory grounds.

## Motivation

The current [`position_decoder_filter`](../../src/state_space_practice/position_decoder.py) is a Laplace-EKF with a unimodal Gaussian posterior. Three structural failures for replay:

1. Random-walk prior with head-speed-scale `q_pos` actively opposes fast decoded sweeps.
2. Gaussian posterior can't represent "position is at arm A **or** arm B" — it picks one mode and reports a wide variance, destroying the replay-content signal.
3. Laplace-to-Poisson approximation breaks down when `λ·dt ≲ 1` (replay regime, `dt = 1–5 ms`). Empirically this shows up as p90 decoding-error blowouts at fine dt.

A GSF keeps continuous state (no grid memory cost) but represents the posterior as `Σ wₖ N(mₖ, Pₖ)`. Multi-modality is native. Per-component updates reuse the existing Laplace-EKF machinery. Memory is `K · n_state²` per step with `K` on the order of 20–50.

## Prior art reviewed

Two Eden-Kramer-Lab MATLAB packages inform this design:

- **GMM_PointProcess** ([github](https://github.com/Eden-Kramer-Lab/GMM_PointProcess)) — closed-form spike update requires Gaussian-parameterized place fields. `ay_gmm_approx_fast` multiplies prior components by field-Gaussians; `ay_gmm_drop_alpha_optimized` and `ay_gmm_merge_alpha_optimized` manage component count via ISD optimization.
- **Multi-Dimensional-Decoder** ([github](https://github.com/Eden-Kramer-Lab/Multi-Dimensional-Decoder/tree/master/SourceCode)) — newer (2018), uses Laplace per component (`ay_gmm_update`), compatible with non-Gaussian rate maps. `ay_gmm_select` does greedy likelihood-based component control; `ay_gmm_merge` supports both merging and EM-based new-component addition.

We adopt **Multi-Dimensional-Decoder's Laplace-per-component pattern** because our library uses KDE rate maps — the closed-form path requires fitting a Gaussian mixture to each KDE map as preprocessing (out of scope).

## V1 scope decisions

**V1 is prune-only, no merging.** The RTS smoother requires stable component identity across time. Forward-pass merging re-assigns slot-to-lineage mappings, which invalidates the standard per-component RTS recursion (a merged component at `t+1` was two separate components at `t`, so there's no single `m_k(t|t)`, `P_k(t|t)` to recurse from). Handling this correctly requires either (a) storing explicit parent-child lineage and walking it during smoothing, or (b) implementing a two-filter smoother (Helmick-Blair-Shea) that decouples forward and backward component management. Both add significant complexity. V1 avoids the problem by pruning only — pruned components stay in their slot with weight zero and frozen moments, so slot-to-lineage identity is preserved.

**No built-in EM-GMM initializer.** V1 supports only `init_mode="track_uniform"` and user-supplied `init_means`, `init_covariances`, `init_log_weights`. Users who want a grid-posterior warm-start compute the GMM fit themselves (e.g., via `sklearn.mixture.GaussianMixture`) and pass the result in. An in-library EM-GMM fitter is a non-trivial subproject and doesn't belong on the critical path for v1.

**No splitting.** Same scope decision as the prior draft; unchanged.

**Filter-only run mode is supported.** Because smoothing is meaningful and correct without merging, v1 ships both a filter and a smoother. Scope is: GSF filter, GSF smoother (both with prune-only identity-preserving component management).

## Design

### State representation

```python
@dataclass
class GSFState:
    means: Array               # (K_max, n_state)
    covariances: Array         # (K_max, n_state, n_state) — always SPD; inactive slots use placeholder
    log_weights: Array         # (K_max,) — log mixing weights
    active: Array              # (K_max,) bool — mask for pruned components
```

Fixed maximum `K_max` at JIT compile time. Typical `K_max = 30–50`. Pruned components set `active=False`, `log_weight=-inf`, and their `means`/`covariances` are **frozen to their last valid pre-prune state** (not zeroed). Zeroing is numerically unsafe — singular covariances produce inversions and log-determinants that explode even when masked.

**Placeholder initialization for never-activated slots.** If `K_init < K_max`, the extra `K_max - K_init` slots at t=0 get `active=False`, `log_weight=-inf`, `mean = arena_center`, `covariance = sigma_track² · I` (a valid SPD placeholder). These slots remain valid SPD Gaussians throughout the trajectory but contribute zero weight to any posterior summary.

### Mask-before-logsumexp discipline

All weight reductions use `jnp.where(active, log_weight, -jnp.inf)` **before** `logsumexp`. Adding `-inf + NaN = NaN` is real — any inactive slot whose covariance went singular (should not happen with placeholders but is possible if Laplace-EKF fails) could poison the reduction if masking happens after the logsumexp. Enforced by construction: every `logsumexp` call in the filter reads `log_weights_safe = jnp.where(active, log_weights, -jnp.inf)`.

### Forward filter: one step

For each time step `t`:

1. **Predict each active component** via linear dynamics `(A, Q)` — `vmap` of Kalman predict over the `K_max` slots. Inactive slots are still predicted (cheap, keeps shape-stable) but their weights remain `-inf`.

2. **Penalty update** (if `track_penalty` supplied) — apply the rank-1 precision update per component. Same math as the current [single-Gaussian filter](../../src/state_space_practice/position_decoder.py#L1043). `vmap` over components.

3. **Observation update via Laplace-EKF per component.** `vmap` [`_point_process_laplace_update`](../../src/state_space_practice/point_process_kalman.py#L615) over components. Each returns `(post_mean, post_cov, log_marginal_ll)`.

   **Per-component failure handling.** If `post_cov` is not finite or not PSD (detectable via `jnp.linalg.cholesky` returning NaN), freeze that component at its predicted (pre-Laplace) state and set its log-likelihood contribution to a large negative constant (`-1e6`) so it gets aggressively pruned on the next weight update. Expressed inside the vmapped body via `jnp.where(cov_is_valid, post_cov, pred_cov)` etc.

4. **Reweight components.** The log-marginal-likelihood from the Laplace update is the correct per-component log-evidence `log p(y_t | x_t ~ N(m_k(t|t-1), P_k(t|t-1)))` (confirmed: `_point_process_laplace_update` with `include_laplace_normalization=True` returns this quantity, not the plug-in log-intensity at the mode).

   ```text
   log_w_k(t) ← log_w_k(t-1) + log_marginal_ll_k(t)
   log_w_k(t) ← log_w_k(t) − logsumexp_{k'}(log_w_{k'}(t))   (normalize)
   ```

5. **Prune.** Mark `active = False` for components whose normalized weight falls below `w_drop = 1e-4`. Freeze their `means` and `covariances` to their pre-prune values (valid SPD). Set `log_weight = -jnp.inf`. Count of active components is non-increasing across time in v1.

### Backward smoother: explicit formula

Forward pass outputs, cached per time step:

- `means_f`, `covariances_f`, `log_weights_f` — filter posteriors `(T, K_max, ...)`.
- `active_f` — `(T, K_max)` bool.
- `means_pred`, `covariances_pred` — **one-step predictive** means and covariances BEFORE the penalty / observation update, `(T, K_max, ...)`. These are what the smoother needs for the gain computation (per the Option-A analysis: standard RTS uses pre-track-prior predictives).

Backward recursion per component `k`, walking t from T-1 down to 0:

**Gain:**

```text
J_k(t) = P_k(t|t) Aᵀ (A P_k(t|t) Aᵀ + Q)⁻¹
        = P_k(t|t) Aᵀ P_k_pred(t+1)⁻¹
```

where `P_k_pred(t+1) = A P_k(t|t) Aᵀ + Q` is the cached one-step predictive covariance from the forward pass.

**Smoothed moments (per-component RTS):**

```text
m_k(t|T) = m_k(t|t) + J_k(t) · (m_k(t+1|T) − A m_k(t|t))
P_k(t|T) = P_k(t|t) + J_k(t) · (P_k(t+1|T) − P_k_pred(t+1)) · J_k(t)ᵀ
```

Valid because component identity `k` is stable across time (no merging in v1).

**Smoothed weights via Sorenson-Alspach (1971) / Helmick-Blair-Shea (1995) recursion.** The backward information factor for component `k` at time `t` is:

```text
log β_k(t) = logsumexp_{k'} [
    log w_{k'}(t+1|T)
    + log N(A m_k(t|t); m_{k'}(t+1|T), A P_k(t|t) Aᵀ + Q + P_{k'}(t+1|T))
]
```

The mean being evaluated is the **forward one-step predicted mean for component `k`** (`A m_k(t|t)`), evaluated against the **smoothed mean of component `k'`** (`m_{k'}(t+1|T)`). The combined covariance is the forward one-step predictive `A P_k(t|t) Aᵀ + Q` plus the smoothed covariance of `k'`.

Smoothed log-weight:

```text
log w_k(t|T) = log w_k(t|t) + log β_k(t)
log w_k(t|T) ← log w_k(t|T) − logsumexp_{k''}(log w_{k''}(t|T))   (normalize)
```

Masking: all sums use `jnp.where(active_f(t+1), ..., -jnp.inf)` to exclude inactive-at-time-t+1 slots from the backward reduction. Since pruning only removes components, `active` at time `t` is a superset of `active` at time `t+1` in v1 — so any slot active at time `t` has at least one active slot at `t+1` (itself) to combine with.

**This is the Sorenson-Alspach formula for linear-Gaussian transitions with Gaussian-mixture posteriors.** It's exact for LG dynamics. The approximation error in our setting comes from the Laplace-to-Poisson approximation in the forward pass, not from the smoother formula itself.

### Component management (prune-only in v1)

After each forward step, prune components with normalized weight below `w_drop = 1e-4`. Implementation is `vmap`-stable:

```python
log_w_normalized = log_weights - logsumexp(log_weights_safe)
prune = log_w_normalized < jnp.log(w_drop)
active_new = active & ~prune
log_weights_new = jnp.where(prune, -jnp.inf, log_weights)
# means and covariances are not modified on prune — they stay frozen at last-valid state
```

No merging, no splitting. If K_max proves inadequate empirically, raise it (cost is `K · n_state²` per step memory and `K`-way vmap compute).

### Initialization

Two modes in v1:

- **`track_uniform`** (default). Spread `K_init` components across **on-track grid bins** (using `rate_maps.occupancy_mask`), not the full arena bounding box. This reduces early-mortality of components initialized off-track with zero information. Identity covariance scaled to a few sigma_track; uniform log-weights. Extra `K_max - K_init` slots are inactive placeholders (see "Placeholder initialization" above).

- **User-supplied.** Caller passes `init_means: (K_init, n_state)`, `init_covariances: (K_init, n_state, n_state)`, `init_log_weights: (K_init,)`. This is the path for grid-posterior warm-start: user fits a GMM to the grid posterior using their preferred library (e.g., `sklearn.mixture.GaussianMixture`), then passes the parameters in. If `K_init < K_max`, remaining slots are placeholders.

Not in v1: `"from_grid_posterior"` mode with in-library EM fitting. Implementing a robust weighted-EM fitter over a 2D grid posterior is a subproject in its own right (iteration limits, convergence criteria, covariance regularization, degenerate-component handling) and isn't on the v1 critical path. Users who want it can call out to `sklearn` or equivalent.

### API

```python
def position_decoder_gsf_filter(
    spikes: ArrayLike,
    rate_maps: PlaceFieldRateMaps,
    dt: float,
    q_pos: Optional[float] = None,
    q_vel: float = 10.0,
    include_velocity: bool = True,
    # Initialization — exactly one of these two paths
    init_mode: Literal["track_uniform"] = "track_uniform",
    init_means: Optional[ArrayLike] = None,       # (K_init, n_state)
    init_covariances: Optional[ArrayLike] = None, # (K_init, n_state, n_state)
    init_log_weights: Optional[ArrayLike] = None, # (K_init,)
    K_init: int = 30,
    K_max: int = 50,
    # Component management
    weight_drop_threshold: float = 1e-4,
    # Observation / penalty (same as single-Gaussian decoder)
    track_penalty: Optional[Array] = None,
    sigma_track: float = 5.0,
    max_newton_iter: int = 3,
) -> GSFResult:
    """Gaussian Sum Filter with point-process Laplace-EKF per component."""
    ...

def position_decoder_gsf_smoother(...) -> GSFResult:
    ...
```

### GSFResult — API shape

```python
@dataclass
class GSFResult:
    # Full per-time-step mixture
    means: Array          # (T, K_max, n_state)
    covariances: Array    # (T, K_max, n_state, n_state)
    log_weights: Array    # (T, K_max) — logsumexp-normalized; -inf for inactive
    active: Array         # (T, K_max) bool — which slots carry weight at each t
    # Convenience: moment-matched single-Gaussian summary
    position_mean: Array          # (T, 2) — mixture mean, position block only
    position_cov: Array           # (T, 2, 2) — mixture covariance, position block
    # Forward-only marginal LL (smoother does not change it)
    marginal_log_likelihood: float
```

**Masked-fixed-size, not ragged.** Downstream code filters by `active[t]` to get the live components at time `t`. This keeps JAX arrays regular and avoids a host-side conversion to ragged form. Document in the docstring that `position_mean` / `position_cov` is a lossy summary (it collapses a potentially multi-modal posterior to a single Gaussian) — replay analysis should consume the full mixture, not the collapsed summary.

## JAX considerations

The whole filter + smoother runs inside `jax.jit` with `jax.lax.scan` for the time loop and `jax.vmap` for the per-component inner loop. This constrains the implementation in several specific ways; getting these wrong produces recompilation storms, silent NaN propagation, or unscannable control flow.

**Static compile-time shapes.** `K_max`, `K_init`, `n_state`, `n_neurons`, `include_velocity`, and `warmup_steps` are all JIT-static. Callers who change any of these pay a full recompile. `K_max` in particular is part of the trace key — running GSF with `K_max=20` and then `K_max=50` in the same session caches two compiled variants. Document this in the entry-point docstring so users expecting to sweep `K_max` for calibration know what they're paying for.

**`jax.lax.scan` for the time loop, `vmap` over components inside the scan body.** The scan carry is `GSFState`; each step runs `vmap(per_component_step, in_axes=0)` over the `K_max` axis. Inner work is `vmap`-ed, outer time loop is scanned. Don't nest another scan over components — XLA fuses vmap over reasonable K_max sizes much better than it fuses a scan-inside-scan.

**No Python control flow inside scan or vmap.** Pruning condition, warmup gating, per-component Laplace failure detection — all expressed via `jnp.where` over fixed-shape arrays, never `if`/`else` on traced values. Concretely:

- Pruning: `active_new = active & (log_w_normalized >= jnp.log(w_drop)) & (step_idx >= warmup_steps)`.
- Warmup: carry a `step_idx` scalar in the scan; compare against the static `warmup_steps`.
- Laplace failure: detect non-PSD post-covariance via `jnp.linalg.cholesky(post_cov)` returning NaN; fall back to `jnp.where(chol_is_finite, post_cov, pred_cov)`.

**Reverse scan for the smoother.** Backward pass is `jax.lax.scan(smoother_step, init_carry, forward_outputs, reverse=True)`. `init_carry` is the terminal forward posterior (`m_k(T|T)`, `P_k(T|T)`, `log_w_k(T|T)`). `forward_outputs` is the per-time-step stack of `(means_f, covariances_f, log_weights_f, active_f, means_pred, covariances_pred)`. The scan body receives one time slice at a time and updates the smoothed state.

**Mask-before-logsumexp discipline.** Every `logsumexp` call on `log_weights` or on the Sorenson-Alspach backward sum reads `log_weights_safe = jnp.where(active, log_weights, -jnp.inf)` *before* the reduction. Adding `-inf + NaN = NaN`, so any inactive slot that somehow carries a NaN (from an earlier failed Laplace step) would poison the reduction if masking happened after. Enforce by encapsulating this pattern in a helper (e.g., `masked_logsumexp(log_weights, active)`) and use it everywhere.

**Reuse existing PSD helpers from `kalman.py`.** The backward gain `J_k(t) = P_k(t|t) Aᵀ P_k_pred(t+1)⁻¹` is a PSD solve — use `psd_solve` from [`kalman.py`](../../src/state_space_practice/kalman.py) (the same helper the single-Gaussian smoother uses), not a direct `jnp.linalg.inv`. Symmetrize after each covariance update via the existing `symmetrize`. These reuse avoids two common JAX gotchas: XLA tolerates minor asymmetry in `inv` but not in `cholesky`, and Cholesky-based solves are ~2× faster than general solves for the same result.

**Gaussian log-pdf.** For the Sorenson-Alspach term, compute `log N(x; μ, Σ)` via a Cholesky-based closed form, not `jax.scipy.stats.multivariate_normal.logpdf` — the latter is slower and has worse gradient behavior when `Σ` is near-singular. Write a small helper `_gaussian_logpdf_cho(x, mean, cov)` that Choleskys `cov` once and reuses for both the quadratic form and the log-determinant.

**No PRNG.** The GSF is fully deterministic given its inputs. No need to thread a `jax.random.PRNGKey`. Initialization for `track_uniform` uses deterministic grid indexing, not sampling.

**Donation for scan carries (optional).** If compile time becomes painful, consider `jax.jit(filter_fn, donate_argnums=...)` on the carry arrays. Marginal win; don't optimize prematurely.

## Non-goals

- **Gaussian place fields.** Closed-form spike update is out of scope. Laplace-EKF per component.
- **Merging.** v2 feature. Requires parent-child lineage or two-filter smoother.
- **Splitting.** v2 feature. Requires particle-based new-component seeding or equivalent.
- **Built-in EM-GMM initializer.** User passes pre-fit parameters instead.
- **Joint parameter learning (Q, A, baselines).** This is a decoder, not a fitter.
- **Replacing the Laplace-EKF decoder.** Existing decoder stays for real-motion tracking.

## Risks (revised post-review)

1. **JIT trace size.** At `K_max=50`, `max_newton_iter=3`, `n_neurons=100`, `n_grid_points=2500`, the compile-time graph is large. Expect 20–30s initial compile. **Fallback:** reduce `K_max` to 20 or switch to bilinear-interpolated rate evaluation inside the GSF path (sacrificing a little accuracy for compile time). Specify this fallback at benchmark time, don't leave open.

2. **Weight numerics at long T.** Products of `λ·dt ≲ 1` over hundreds of time steps in log-space with repeated renormalization are fine in principle but need a regression test at `T=1000` to confirm no silent underflow.

3. **Component mortality during initialization.** At `K_init=30` with `track_uniform` across the on-track grid and a replay event concentrated on one arm, many components lose most of their weight in the first 2-3 steps. With `w_drop=1e-4` they may be pruned before they can contribute any useful prior. **Mitigation:** accept a "warm-up" period in v1 where `w_drop` effectively kicks in only after 5 time steps (implemented as: ignore prune condition for the first N steps). Set `warmup_steps=5` as a constructor default.

4. **Prune-only may duplicate components.** Without merging, nearly-identical components waste slots. If this shows up in validation (manifests as `K_active` staying at `K_max` with visually redundant components), we'll add merging in v2. For short replay events, empirically unlikely to be a blocker.

5. **Track geometry coverage by K_init.** `track_uniform` with `K_init=30` may under-cover bifurcation points on complex tracks (W-mazes, Y-mazes). For the validation dataset specifically, we need to eyeball the initialization vs arena shape and confirm at least ~3 components per arm junction. If insufficient, bump `K_init` to 50 (K_max to 100). This is a tunable, not a structural issue.

6. **Laplace-EKF failure modes per component.** Individual components whose means wander into low-rate regions can produce non-PSD post-covariances. The per-component failure handling described above (freeze at predicted state, large negative log-likelihood to force pruning) is the right primitive; regression test it with an adversarially-placed component.

7. **Smoother weight formula correctness.** The Sorenson-Alspach recursion above is correct for LG dynamics. Regression test against a linear-Gaussian toy model with known exact posterior.

## Validation plan

1. **Regression against single-Gaussian decoder.** `position_decoder_gsf_filter(K_init=1, ...)` output must match `position_decoder_filter(...)` to floating-point precision on a fixed-seed synthetic fixture. This is a gating regression test from step 1 of implementation, not an afterthought.

2. **Multi-modal synthetic replay.** Hand-crafted 2D track with two arms, simulated spikes ambiguous between them. GSF with `K_init=10`. Expected: posterior at ambiguous time step has non-trivial weight on at least two well-separated components. Compare to single-Gaussian decoder, which should produce a wide unimodal posterior. Metric: number of active components at ambiguity points, weight distribution across arms.

3. **Linear-Gaussian toy for smoother correctness.** A purely LG SSM with known analytical posterior. GSF smoother output vs. exact posterior. Validates the Sorenson-Alspach weight formula implementation.

4. **Real CA1 replay events vs. grid decoder.** Run both decoders on detected SWR events from `j1620210710_02_r1`. Compare: (a) marginal posterior agreement per time bin, (b) memory footprint (expect dramatic reduction), (c) correlation between GSF-derived and grid-derived replay content scores.

Success criterion: GSF posterior summary ≈ grid posterior summary on real replay events, at a fraction of the memory.

## Effort estimate (revised)

Based on the combined review feedback:

- Core forward filter + scan integration + SPD placeholder handling: ~3 days
- Pruning + per-component Laplace failure handling: ~2 days
- Backward smoother with Sorenson-Alspach weights + correctness test against LG toy: ~4 days (was 2; reviewer was right — this is the subtlest code)
- Regression test (K_init=1 matches single-Gaussian decoder): ~1 day
- Synthetic multi-modal test: ~2 days
- Real-data calibration against grid decoder: **1–2 weeks** (was 3 days; multi-parameter tuning is larger than originally estimated)
- Documentation + demo notebook: ~1 day

**Total: ~3–4 weeks** of focused work, assuming no major surprises during calibration. Splitting (if needed) adds another 2–3 weeks for EM-based or particle-based component seeding.

## Dependencies

None new. `jax`, `numpy`. No TFP, no new runtime deps.

## Relationship to other library pieces

- **Reuses** [`_point_process_laplace_update`](../../src/state_space_practice/point_process_kalman.py#L615) via `vmap`.
- **Reuses** [`PlaceFieldRateMaps`](../../src/state_space_practice/position_decoder.py#L138) unchanged.
- **Reuses** [`_build_track_penalty`](../../src/state_space_practice/position_decoder.py#L767) unchanged.
- **Does not modify** the single-Gaussian decoder. GSF is additive.
- **Reuses** [`_kalman_smoother_update`](../../src/state_space_practice/kalman.py#L345) — per-component in the backward pass (with `one_step_cov_override` if we ever ship the EKS addition; not required for GSF since smoother identity is stable in v1).

## Order of execution

1. `GSFState` dataclass, SPD-placeholder initialization, K_init=1 regression test scaffolding (before any real logic). Gate: K_init=1 path returns identical output to `position_decoder_filter` on a fixed seed.
2. Forward per-step: predict + penalty + Laplace, vmapped, with SPD placeholders and per-component failure handling. No pruning yet. Gate: K_init=5 on synthetic data produces sensible-looking mixture.
3. Pruning with warmup. Gate: inactive slots don't contribute to posterior summary, don't cause NaN in logsumexp.
4. Backward smoother (per-component RTS moments). Gate: smoother at K_init=1 matches `position_decoder_smoother` on fixed seed.
5. Sorenson-Alspach smoother-weight formula. Gate: linear-Gaussian toy test produces the exact analytical smoothed weights.
6. End-to-end multi-modal synthetic test.
7. Real CA1 replay validation and parameter calibration.
