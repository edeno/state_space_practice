# Shared contracts — streaming filters

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Contracts that more than one phase depends on. Each appears once; phases link
here by anchor. "Do not weaken" means a later phase may extend but must not
change the stated semantics.

- [FilterState](#filterstate)
- [Streaming filter surface](#streaming-filter-surface)
- [Parity policy](#parity-policy)
- [Optional cross-plan keyword forwarding](#optional-cross-plan-keyword-forwarding)
- [Prediction hand-off for fixed-lag smoothing](#prediction-hand-off-for-fixed-lag-smoothing)

## FilterState

Defined in `src/state_space_practice/streaming.py` (Phase 1). The one public
state type for every streaming filter in this plan.

```python
class FilterState(NamedTuple):
    """Posterior of the latent state after the most recent observation.

    Attributes
    ----------
    mean : Array, shape (n_state,)
        Filtered mean ``m_{t|t}``.
    cov : Array, shape (n_state, n_state)
        Filtered covariance ``P_{t|t}`` (symmetric PSD by construction: Joseph
        form for the Gaussian filter, inverse of a PSD precision for the
        Laplace filters).
    t : Array, int32 scalar
        Number of observations absorbed since the last reset. ``t == 0`` is the
        prior; after the first ``step`` it is 1.
    log_likelihood : Array, float scalar
        Cumulative ``sum_{s<=t} log p(y_s | y_{1:s-1})`` since the last reset,
        accumulated in the same order the batch filters use.
    """

    mean: Array
    cov: Array
    t: Array
    log_likelihood: Array
```

Semantics and invariants (do not weaken):

- It is a pytree (NamedTuple), so it can be a `jax.jit` input/output and a
  `lax.scan` carry. All array fields share the filter's common float dtype
  except `t`, which is `int32` (2^31 bins is ~24.8 days at 1 kHz; documented).
- Exactly these four fields. Diagnostics that are not part of the posterior
  (innovations, one-step predictions, line-search / inflation counters) live on
  the filter object, not on the state — see
  [Prediction hand-off](#prediction-hand-off-for-fixed-lag-smoothing).
- `reset(state)` with a caller-supplied `FilterState` is allowed (restart from
  a stored posterior); `state.t` is then taken at face value.
- `smoothed(lag)` (Phase 2+) returns a `FilterState` whose `mean` / `cov` are
  the fixed-lag smoothed moments of bin `t - lag`, whose `t` is `state.t - lag`,
  and whose `log_likelihood` is the *filter's* cumulative log-likelihood
  through the current bin (smoothing does not change the marginal likelihood,
  matching `kalman_smoother` returning the filter's value).

## Streaming filter surface

Every streaming class (`StreamingKalmanFilter`, `StreamingPointProcessFilter`,
`StreamingPositionDecoder`) exposes this surface with these semantics.

| Member | Semantics (do not weaken) |
| --- | --- |
| `__init__(...)` | Host-side (eager): validates shapes and covariances with `ValueError`, promotes parameters to one common float dtype, warns `StateSpaceWarning` if that dtype is `float32` (a streaming run has unbounded T, so the batch filters' "f32 + long T" risk always applies). Never call inside `jax.jit`. Accepts `lag: int = 0` (Phase 2+) and `robust_weight=None`. |
| `.state` | The current `FilterState`. |
| `.step(obs, ..., mask=None) -> FilterState` | Absorbs one bin. Exactly one dispatch of a `jax.jit`-compiled module-level function per call; compiled once per (shapes, dtype, `mask is None`, static configuration). Does no value checks on `obs` (no host sync); a non-finite observation poisons the state, so drop channels with `mask` instead. Raises `ValueError` on a shape mismatch (which would otherwise silently retrace). |
| `.run(...)` | Python loop over `.step` **from the current state** (call `reset()` first for a fresh run). Returns what the corresponding batch filter returns for the bins passed in — Gaussian and point-process: `(filtered_mean (T, n_state), filtered_cov (T, n_state, n_state), log_likelihood)` where `log_likelihood` is the sum over the bins of this call (`state.log_likelihood` after minus before; from a fresh state this is bit-identical to the batch value because `x - 0.0 == x`); decoder: a `DecoderResult`. Exists for parity testing and short offline runs; long offline runs should use the batch filters. |
| `.reset(state=None)` | Restores the construction-time prior (or the given state) and empties the fixed-lag buffer and diagnostic counters. |
| `.n_steps` | Python `int` mirror of `int(state.t)`, maintained without a device sync. |
| `.smoothed(lag=None) -> FilterState` | Phase 2+. `lag` defaults to the construction-time `lag`; `0 < lag <= self.lag` else `ValueError`; `lag == 0` returns `.state`; fewer than `lag + 1` steps since the last reset raises `ValueError` (warm-up). |
| `.lag` | The buffer length fixed at construction. |

Point-process classes additionally expose `.n_line_search_failures` (int:
bins since reset in which the Fisher-scoring backtracking was exhausted,
mirroring the batch filters' `n_failed_bins` carry) and the decoder
`.n_inflation_capped_bins` (mirroring `n_capped_bins`). `.run()` on these
classes emits the same end-of-run warnings the batch entry points emit from
those counters (`_warn_line_search_failures`); `.step()` never warns.

## Parity policy

"Bit-identical" in this plan means `np.testing.assert_array_equal` on the
stacked filtered means and covariances and `float(a) == float(b)` on the
summed log-likelihood, comparing `.run()` from a fresh object against the
batch filter on the same arrays. It is achievable because each streaming step
executes the *same jaxpr* as the batch scan body (verified at planning time on
CPU, JAX 0.10.2, float64, for the Gaussian, point-process and bilinear-rate-map
decoder paths).

| Path | Required agreement | Reason |
| --- | --- | --- |
| `StreamingKalmanFilter.run` vs `kalman_filter` | bit-identical | same ops as `kalman._kalman_filter_update` |
| `StreamingPointProcessFilter.run` vs `stochastic_point_process_filter` (dense path) | bit-identical | `glm_laplace_update(poisson_family(dt))` is the batch update |
| `StreamingPositionDecoder.run` vs `position_decoder_filter`, bilinear rate maps | bit-identical | shared `_decoder_step` |
| `StreamingPositionDecoder.run` vs `position_decoder_filter`, KDE rate maps | `assert_allclose(rtol=1e-11, atol=1e-11 * scale)` | the KDE kernel sum over the grid is a large reduction whose summation order XLA chooses differently inside vs outside the scan (observed max abs difference 2.8e-14 on means, 5.7e-14 on the log-likelihood) |
| Fixed-lag window covering the whole sequence vs batch smoother | `rtol=1e-12` | different scans over the same arithmetic (observed 1e-16) |

If a future JAX upgrade breaks a bit-identical row, first confirm the
difference is round-off-sized (max relative difference ≤ 1e-13) and that the
jaxprs still match; only then relax that row to `rtol=1e-12` with a CHANGELOG
note. A larger difference is a logic change and must be fixed, not tolerated.

## Optional cross-plan keyword forwarding

Two parallel plans add keyword arguments to the update functions this plan
builds on: `docs/plans/wolf-robust-updates/` adds `robust_weight=` and
`docs/plans/masks-and-multi-sequence/` adds `obs_mask=` to
`kalman.kalman_measurement_update` and `point_process_kalman.glm_laplace_update`
(and, for the decoder path, to `point_process_kalman._point_process_laplace_update`).
Their semantics are owned there; this plan only forwards. What those plans fix
(read at planning time from their `shared-contracts.md` / `designs.md`; re-read
before executing Phase 1):

- `obs_mask` — keyword name fixed by cross-plan agreement; boolean array,
  `True` = observed; **non-boolean dtypes are rejected with `ValueError`**
  (0/1 integers included); masked entries of the observation may hold NaN;
  `obs_mask=None` leaves the pre-existing code path with no added operations
  (bit-identical off state). On the single-step updates the per-bin shape is
  the observation shape.
- `robust_weight` — a hashable `RobustWeight` object (e.g. `imq_weight(c=...)`),
  passed as a **static** `jax.jit` argument; `None` reproduces today's output
  bit-for-bit; when not `None` the update returns **one extra trailing
  element** `RobustOutput(objective, weight)`, i.e. `kalman_measurement_update`
  returns 4 values and `glm_laplace_update` one more than today.

Rules (do not weaken):

- Support is detected once, host-side, with
  `name in inspect.signature(update_fn).parameters` (works through `jax.jit`;
  verified). Detection lives in one helper, `streaming._accepts_kwarg`.
- `robust_weight=None` (constructor) and `mask=None` (per step) mean **the
  keyword is not passed at all**, so the default path stays bit-identical to
  the batch filters regardless of how those plans implement the defaults.
- A non-`None` value when the underlying update lacks the keyword raises
  `NotImplementedError` naming the plan directory that adds it. No local
  re-implementation of masking or robust weights in `streaming.py`.
- `mask` is forwarded as `obs_mask=mask`, a boolean array of the observation
  shape (`(n_obs,)` for the Gaussian filter, `(n_neurons,)` for the
  point-process filters); it is a traced argument of a second compiled variant
  of the step (`mask is None` selects the variant), so passing a mask on some
  bins and not others costs one extra compilation, not a retrace per bin. The
  step checks `mask.dtype == bool` and the shape host-side and raises
  `ValueError` otherwise (**no** `astype(bool)` cast, matching the owning
  plan's rejection of 0/1 masks); these checks run before the support check.
- `robust_weight` is forwarded verbatim (closed over by the per-instance
  update callable, which is itself a static argument of the compiled step, so
  the object's hashability is never exercised by our jit). The wrapper
  returns exactly the batch arity — `result[:3]` for the Gaussian update,
  `result[:4]` for `glm_laplace_update(..., return_line_search_failures=True)`
  — so a trailing `RobustOutput` is dropped; the streaming filters do not
  surface the robust objective or per-bin weights (see overview Open
  Question 1 for the trigger to add them).
- Revisit trigger: when both plans have merged, delete the
  `NotImplementedError` branches and their skip-guarded tests, and turn the
  `skipif` semantics tests into unconditional tests.

## Prediction hand-off for fixed-lag smoothing

Each compiled step returns, besides the new `FilterState`, the one-step
prediction the update conditioned on: `(pred_mean, pred_cov)` of shape
`(n_state,)`, `(n_state, n_state)`. This is the quantity
`kalman.rts_backward_scan_with_predictions` documents as `predicted_mean[t]` /
`predicted_cov[t]` (kalman.py:869-877), so the fixed-lag smoother (Phase 2)
can run that one backward pass for all three filters:

- Gaussian and point-process filters: `A m_{t-1|t-1}` and
  `symmetrize(A P_{t-1|t-1} A^T + Q)`.
- Position decoder: the *dynamics* prediction including the adaptive-inflation
  factor and **excluding** the track-penalty Woodbury downdate, exactly what
  `position_decoder._run_filter_scan` stores as `dynamics_mean` /
  `dynamics_cov` (position_decoder.py:1008-1016, 1091-1094, 1129-1131) and what
  `position_decoder_smoother` consumes (position_decoder.py:1599-1606).

Phase 1 ships the return value (unused, `buf` argument accepted and passed
through as `None`); Phase 2 consumes it. Phases 3a/3b must return it with the
semantics above — the decoder must not hand the penalised prediction to the
smoother (the penalty is a pseudo-observation already in the filtered moments).

## Oscillator summary edge cases

The phase and amplitude summary containers follow [design D3](designs.md#d3-phase-and-amplitude-posteriors-of-a-2-d-oscillator-block). At numerically zero mean, phase has `phase_defined=False`, phase 0 by convention, `phase_sd=inf` and `half_width=pi` (the whole circle). The amplitude fallback has `used_delta_method=False`, a finite conservative SD bound and an interval obtained from the Gaussian confidence ellipse. Consumers must use the flags; coincident wrapped phase endpoints do not represent a zero-width interval when `half_width=pi`. This leaves the four-field `FilterState` contract unchanged.
