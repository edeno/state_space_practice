# Designs — streaming filters

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

Per-component designs with complete code for the non-obvious parts. Phases
reference these by anchor and do not repeat them.

- [D1 Module layout and compiled-step pattern](#d1-module-layout-and-compiled-step-pattern)
- [D2 Gaussian step and StreamingKalmanFilter](#d2-gaussian-step-and-streamingkalmanfilter)
- [D3 Phase and amplitude posteriors of a 2-D oscillator block](#d3-phase-and-amplitude-posteriors-of-a-2-d-oscillator-block)
- [D4 Fixed-lag smoother ring buffer](#d4-fixed-lag-smoother-ring-buffer)
- [D5 Point-process step and StreamingPointProcessFilter](#d5-point-process-step-and-streamingpointprocessfilter)
- [D6 Position decoder step extraction and StreamingPositionDecoder](#d6-position-decoder-step-extraction-and-streamingpositiondecoder)
- [D7 Latency benchmark script](#d7-latency-benchmark-script)

## D1 Module layout and compiled-step pattern

One new module, `src/state_space_practice/streaming.py`, holding everything
public in this plan (`FilterState`, the three streaming classes, the phase /
amplitude posteriors) plus the private step functions and the lag buffer. The
position-decoder step body is *not* duplicated here: Phase 3b factors it out
of `position_decoder.py` and `streaming.py` imports it (see
[D6](#d6-position-decoder-step-extraction-and-streamingpositiondecoder)).
`position_decoder.py` must never import `streaming.py` (no import cycle).

### Compiled-step pattern

Each filter's per-bin work is a **module-level** function decorated with
`jax.jit` and taking the model parameters as ordinary (traced) arguments, so
the compilation cache is shared by all instances with the same shapes and
static configuration (tests construct many instances; a real-time user
constructs one). Callables that differ between instances (the measurement
update wrapper, the log-intensity function, its Jacobian) are `static_argnames`
and are created **once per instance** and stored on it, because `jax.jit`
hashes static arguments by identity: a fresh closure per call would recompile
on every step.

```python
@functools.partial(jax.jit, static_argnames=("update",))
def _kalman_streaming_step(state, buf, y, mask, A, Q, H, R, *, update): ...
```

`buf` (the fixed-lag ring buffer, [D4](#d4-fixed-lag-smoother-ring-buffer)) and
`mask` are pytree arguments that may be `None`; `None` is an empty pytree, so
"no buffer" / "no mask" compile to their own specialisations without any
Python flag.

### Base class

```python
class _StreamingFilter:
    """State bookkeeping shared by the streaming filters (not public)."""

    _transition_matrix: Array  # set by subclasses; used by smoothed()

    def __init__(self, init_mean: Array, init_cov: Array, *, lag: int = 0) -> None:
        self.lag = validate_int(lag, "lag", nonnegative=True)
        self._init_state = FilterState(
            mean=init_mean,
            cov=init_cov,
            t=jnp.zeros((), dtype=jnp.int32),
            log_likelihood=jnp.zeros((), dtype=init_mean.dtype),
        )
        self.reset()

    def reset(self, state: FilterState | None = None) -> None:
        self.state = self._init_state if state is None else state
        self.n_steps = int(self.state.t)
        self._n_since_reset = 0
        n_state = self._init_state.mean.shape[0]
        self._buffer = (
            None
            if self.lag == 0
            else _empty_lag_buffer(self.lag, n_state, self._init_state.mean.dtype)
        )
        self._reset_counters()  # no-op here; point-process subclasses zero their aux counters

    def _reset_counters(self) -> None:
        return None

    def _advance(self, new_state: FilterState) -> FilterState:
        self.state = new_state
        self.n_steps += 1
        self._n_since_reset += 1
        return new_state

    # smoothed() is added in Phase 2; see D4.
```

`validate_int` is `state_space_practice.utils.validate_int(value, name, *,
positive=False, nonnegative=False) -> int` (utils.py:1113).

### Construction-time validation and dtype

Mirrors what the batch entry points do once per call, because a streaming
object is the "call":

- Shapes: as `kalman._validate_kalman_public_inputs` (kalman.py:301-345) minus
  the `obs` checks; a mismatch raises `ValueError` with the same wording.
- Values: `utils.validate_covariance` (utils.py:1308) on `init_cov`
  (positive definite), `measurement_cov` (positive definite) and
  `process_cov` (`require_positive_definite=False`); finiteness of the
  transition / measurement matrices and `init_mean`.
- Dtype: `dtype = jnp.result_type(float, *arrays)` over the parameters and the
  initial moments, then `astype(dtype)` on all of them, exactly as
  `kalman._kalman_filter_impl` (kalman.py:551-567) and
  `point_process_kalman._common_float_dtype` (point_process_kalman.py:512-520)
  do. If `dtype == jnp.float32`, `warnings.warn(..., StateSpaceWarning,
  stacklevel=2)` unconditionally (unbounded T).
- Per-step inputs are converted with `jnp.asarray(obs, dtype=dtype)` and their
  shape checked against the construction-time shape; a mismatch raises
  `ValueError` instead of silently retracing.

## D2 Gaussian step and StreamingKalmanFilter

The step computes the prediction inline and calls the public
`kalman_measurement_update` (kalman.py:426-475). This is the body of
`kalman._kalman_filter_update` (kalman.py:517-526) unrolled by one call so the
prediction is available for the lag buffer; the arithmetic is identical and
was verified bit-identical to `kalman_filter` for
`(n_state, n_obs) in {(2, 1), (4, 3), (8, 16)}`, 300 bins. (Calling
`_kalman_filter_update` directly is also bit-identical but hides the
prediction; `rts_backward_scan` could recompute it for the Gaussian case, but
the decoder cannot, and one smoother implementation is wanted.)

```python
def _plain_kalman_update(pred_mean, pred_cov, y, H, R, mask):
    """Default measurement update: no mask, no robust weight (parity path)."""
    return kalman_measurement_update(pred_mean, pred_cov, y, H, R)


def _make_kalman_update(robust_weight, *, use_mask):
    """Build the per-instance update wrapper that forwards optional keywords."""
    if robust_weight is not None and not _accepts_kwarg(
        kalman_measurement_update, "robust_weight"
    ):
        raise NotImplementedError(
            "robust_weight= needs kalman_measurement_update(robust_weight=...), "
            "added by docs/plans/wolf-robust-updates/."
        )
    if use_mask and not _accepts_kwarg(kalman_measurement_update, "obs_mask"):
        raise NotImplementedError(
            "mask= needs kalman_measurement_update(obs_mask=...), added by "
            "docs/plans/masks-and-multi-sequence/."
        )
    extra = {} if robust_weight is None else {"robust_weight": robust_weight}

    def update(pred_mean, pred_cov, y, H, R, mask):
        if use_mask:
            result = kalman_measurement_update(
                pred_mean, pred_cov, y, H, R, obs_mask=mask, **extra
            )
        else:
            result = kalman_measurement_update(pred_mean, pred_cov, y, H, R, **extra)
        # A robust update appends RobustOutput; keep the batch arity (contract).
        return result[:3]

    return update


@functools.partial(jax.jit, static_argnames=("update",))
def _kalman_streaming_step(state, buf, y, mask, A, Q, H, R, *, update):
    """One predict + measurement update; mirrors ``kalman._kalman_filter_update``."""
    pred_mean = A @ state.mean
    pred_cov = symmetrize(A @ state.cov @ A.T + Q)
    mean, cov, ll_t = update(pred_mean, pred_cov, y, H, R, mask)
    new_state = FilterState(mean, cov, state.t + 1, state.log_likelihood + ll_t)
    if buf is not None:  # Phase 2
        buf = _push_lag_buffer(buf, new_state, pred_mean, pred_cov)
    return new_state, buf
```

```python
class StreamingKalmanFilter(_StreamingFilter):
    def __init__(
        self,
        transition_matrix: ArrayLike,
        process_cov: ArrayLike,
        measurement_matrix: ArrayLike,
        measurement_cov: ArrayLike,
        init_mean: ArrayLike,
        init_cov: ArrayLike,
        *,
        robust_weight=None,
        lag: int = 0,
    ) -> None:
        # D1 validation + dtype promotion -> A, Q, H, R, m0, P0, self.n_state, self.n_obs, self._dtype
        ...
        self._params = (A, Q, H, R)
        self._transition_matrix = A
        self._robust_weight = robust_weight
        self._update = (
            _plain_kalman_update
            if robust_weight is None
            else _make_kalman_update(robust_weight, use_mask=False)
        )
        self._masked_update = None  # built on the first masked step
        super().__init__(m0, P0, lag=lag)

    def step(self, y: ArrayLike, mask: ArrayLike | None = None) -> FilterState:
        y = jnp.asarray(y, dtype=self._dtype)
        if y.shape != (self.n_obs,):
            raise ValueError(f"y must have shape ({self.n_obs},), got {y.shape}.")
        if mask is None:
            update, mask_arr = self._update, None
        else:
            if self._masked_update is None:
                self._masked_update = _make_kalman_update(self._robust_weight, use_mask=True)
            update = self._masked_update
            mask_arr = jnp.asarray(mask)
            if mask_arr.dtype != jnp.bool_:
                raise ValueError("mask must be boolean (True = channel observed); cast explicitly.")
            if mask_arr.shape != (self.n_obs,):
                raise ValueError(...)
        new_state, self._buffer = _kalman_streaming_step(
            self.state, self._buffer, y, mask_arr, *self._params, update=update
        )
        return self._advance(new_state)

    def run(self, ys: ArrayLike) -> tuple[Array, Array, Array]:
        ys = jnp.asarray(ys, dtype=self._dtype)
        if ys.ndim != 2 or ys.shape[1] != self.n_obs:
            raise ValueError(...)
        ll_before = self.state.log_likelihood
        means, covs = [], []
        for y in ys:
            state = self.step(y)
            means.append(state.mean)
            covs.append(state.cov)
        return jnp.stack(means), jnp.stack(covs), self.state.log_likelihood - ll_before
```

## D3 Phase and amplitude posteriors of a 2-D oscillator block

Conventions (from the code, not chosen here): an oscillator block occupies
state rows `2k, 2k+1` = `(re, im)` — `coupling_model.logit` reads
`re = state[0::2]`, `im = state[1::2]` (coupling_model.py:300-301) and
`oscillator_models.get_phase_difference` takes `arctan2(coef[1], coef[0])`
(oscillator_models.py:1377-1378); `oscillator_utils.get_block_slice(k, k)`
(oscillator_utils.py:55-73) returns the `(rows, cols)` slices of that block.
So the phase of block `k` is `atan2(mean[2k+1], mean[2k])`.

Delta method on `x = (a, b) ~ N(m, S)` with `S` the block's 2×2 covariance:

- Phase `φ = atan2(b, a)`, gradient `∇φ = (−b, a) / (a² + b²)`, so
  `Var φ ≈ (b² S₀₀ − 2ab S₀₁ + a² S₁₁) / (a² + b²)²`.
- Amplitude `r = sqrt(a² + b²)`, gradient `∇r = (a, b) / r`, so
  `Var r ≈ (a² S₀₀ + 2ab S₀₁ + b² S₁₁) / (a² + b²)`.

Credible interval at level `1 − α`: `φ ± z sd` with
`z = Φ⁻¹(1 − α/2)`, half-width clipped at `π` (the interval never exceeds the
circle), endpoints wrapped to `(−π, π]` with `atan2(sin, cos)` — the same range
`circular_stats.wrap_to_pi` returns. At numerically zero amplitude the delta
method is undefined. Return a whole-circle phase interval explicitly, with
`phase_defined=False`, conventional phase 0 and `phase_sd=inf`. Keep its
`half_width=pi` so coincident wrapped endpoints cannot be mistaken for a
zero-width interval. For amplitude, return the norm of the mean (0 at the
origin), a conservative uncertainty bound `sqrt(trace(S))`, and the interval
`[0, ||m|| + sqrt(chi2.ppf(level, 2) * lambda_max(S))]`. This follows by
enclosing the Gaussian confidence ellipse in a ball and applying the triangle
inequality. Mark `used_delta_method=False`: the reported SD is an upper bound
in this branch, not a radial standard-deviation estimate. Accuracy away from
the numerical origin: the delta method is a high-SNR approximation;
against 400k-sample Monte Carlo (anisotropic `S = [[1, .3], [.3, .6]]`) the
relative error of the phase sd was 0.1 % at `r / sd_r = 30`, 1.3 % at 10,
6 % at 5, 20 % at 3, 34 % at 1, and the angle of the mean drifts from the
posterior circular mean below `r / sd_r ≈ 5` (0.700 vs 0.706 at 5, vs 0.748
at 1). Document "reliable for amplitude / amplitude_sd ≳ 10" and pin the trend
in a test.

```python
class PhasePosterior(NamedTuple):
    phase: Array        # atan2(im, re) of the block mean, in (-pi, pi]
    phase_sd: Array     # delta-method standard deviation, radians
    lower: Array        # wrapped phase - min(z * sd, pi)
    upper: Array        # wrapped phase + min(z * sd, pi)
    half_width: Array   # pi explicitly denotes the whole circle
    phase_defined: Array  # False at numerically zero mean; phase=0 is then conventional


class AmplitudePosterior(NamedTuple):
    amplitude: Array
    amplitude_sd: Array
    lower: Array        # max(amplitude - z * sd, 0)
    upper: Array
    used_delta_method: Array  # False => amplitude_sd is a conservative bound


def _block_rows(block: int | slice) -> slice:
    if isinstance(block, slice):
        return block
    rows, _ = get_block_slice(block, block)
    return rows


def _wrap_angle(angle: Array) -> Array:
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def phase_posterior(state: FilterState, block: int | slice, level: float = 0.95) -> PhasePosterior:
    rows = _block_rows(block)
    m = state.mean[rows]
    S = state.cov[rows, rows]
    a, b = m[0], m[1]
    r2 = a * a + b * b
    defined = r2 > jnp.finfo(m.dtype).eps * jnp.maximum(jnp.trace(S), 0.0)
    safe_r2 = jnp.where(defined, r2, 1.0)
    phase = jnp.where(defined, jnp.arctan2(b, a), 0.0)
    var = (b * b * S[0, 0] - 2.0 * a * b * S[0, 1] + a * a * S[1, 1]) / safe_r2**2
    sd = jnp.where(defined, jnp.sqrt(jnp.maximum(var, 0.0)), jnp.inf)
    z = jax.scipy.stats.norm.ppf(0.5 + 0.5 * level)
    half = jnp.minimum(z * sd, jnp.pi)
    return PhasePosterior(phase, sd, _wrap_angle(phase - half), _wrap_angle(phase + half), half, defined)


def amplitude_posterior(state: FilterState, block: int | slice, level: float = 0.95) -> AmplitudePosterior:
    rows = _block_rows(block)
    m = state.mean[rows]
    S = state.cov[rows, rows]
    a, b = m[0], m[1]
    r2 = a * a + b * b
    amplitude = jnp.sqrt(r2)
    trace = jnp.maximum(jnp.trace(S), 0.0)
    use_delta = r2 > jnp.finfo(m.dtype).eps * trace
    safe_r2 = jnp.where(use_delta, r2, 1.0)
    var = (a * a * S[0, 0] + 2.0 * a * b * S[0, 1] + b * b * S[1, 1]) / safe_r2
    sd = jnp.sqrt(jnp.maximum(jnp.where(use_delta, var, trace), 0.0))
    z = jax.scipy.stats.norm.ppf(0.5 + 0.5 * level)
    max_eigenvalue = jnp.maximum(jnp.linalg.eigvalsh(S)[-1], 0.0)
    # chi-square quantile with 2 degrees of freedom, available without a PPF call.
    radius = jnp.sqrt(-2.0 * jnp.log1p(-level) * max_eigenvalue)
    lower = jnp.where(use_delta, jnp.maximum(amplitude - z * sd, 0.0), 0.0)
    upper = jnp.where(use_delta, amplitude + z * sd, amplitude + radius)
    return AmplitudePosterior(amplitude, sd, lower, upper, use_delta)
```

Both functions are pure `jnp` (jit- and vmap-friendly) and accept any
`FilterState` (filtered or `smoothed(...)`). Validate `level` in `(0, 1)`
host-side with `ValueError`; `block` as an `int` must be `< n_state // 2`
(`ValueError`, checked from `state.mean.shape` which is concrete outside jit).

## D4 Fixed-lag smoother ring buffer

Fixed-lag smoothing at lag `L` (Anderson & Moore, *Optimal Filtering*, 1979,
ch. 7) is the RTS backward pass over the window `[t−L, t]` started from the
filtered moments at `t`. Rather than a dedicated fixed-lag recursion (which
augments the state and is O(L²) per step) the design keeps the last `L + 1`
filtered moments and one-step predictions in a ring buffer and runs
`kalman.rts_backward_scan_with_predictions` (kalman.py:845-912) over the
window on demand: O(L) per `smoothed()` call, zero extra work per `step()`
beyond the buffer write. That backward pass is used for all three filters
because it consumes the stored predictions (the decoder's are inflated; see
[the contract](shared-contracts.md#prediction-hand-off-for-fixed-lag-smoothing));
with unmodified predictions it reduces exactly to `rts_backward_scan`
(kalman.py:866-867).

```python
class _LagBuffer(NamedTuple):
    """Ring buffer of the last ``lag + 1`` filtered moments and predictions."""

    filtered_mean: Array   # (lag + 1, n_state)
    filtered_cov: Array    # (lag + 1, n_state, n_state)
    predicted_mean: Array  # (lag + 1, n_state)   prediction of x_t used at step t
    predicted_cov: Array   # (lag + 1, n_state, n_state)
    head: Array            # int32 scalar: slot holding the most recent entry


def _empty_lag_buffer(lag: int, n_state: int, dtype) -> _LagBuffer:
    zeros_1 = jnp.zeros((lag + 1, n_state), dtype=dtype)
    zeros_2 = jnp.zeros((lag + 1, n_state, n_state), dtype=dtype)
    # head = lag so the first push lands in slot 0.
    return _LagBuffer(zeros_1, zeros_2, zeros_1, zeros_2, jnp.asarray(lag, dtype=jnp.int32))


def _push_lag_buffer(buf: _LagBuffer, state: FilterState, pred_mean: Array, pred_cov: Array) -> _LagBuffer:
    head = (buf.head + 1) % buf.filtered_mean.shape[0]
    return _LagBuffer(
        buf.filtered_mean.at[head].set(state.mean),
        buf.filtered_cov.at[head].set(state.cov),
        buf.predicted_mean.at[head].set(pred_mean),
        buf.predicted_cov.at[head].set(pred_cov),
        head,
    )


@functools.partial(jax.jit, static_argnames=("lag",))
def _smooth_lag_window(buf: _LagBuffer, transition_matrix: Array, *, lag: int) -> tuple[Array, Array]:
    """Smoothed moments of the bin ``lag`` steps before the most recent one."""
    size = buf.filtered_mean.shape[0]
    order = (buf.head - lag + jnp.arange(lag + 1)) % size  # chronological, oldest first
    smoothed_mean, smoothed_cov, _ = rts_backward_scan_with_predictions(
        buf.filtered_mean[order],
        buf.filtered_cov[order],
        buf.predicted_mean[order],
        buf.predicted_cov[order],
        transition_matrix,
    )
    return smoothed_mean[0], smoothed_cov[0]
```

`_StreamingFilter.smoothed`:

```python
    def smoothed(self, lag: int | None = None) -> FilterState:
        lag = self.lag if lag is None else validate_int(lag, "lag", nonnegative=True)
        if lag > self.lag:
            raise ValueError(f"lag={lag} exceeds the buffer length lag={self.lag} fixed at construction.")
        if lag == 0:
            return self.state
        if self._n_since_reset < lag + 1:
            raise ValueError(
                f"smoothed(lag={lag}) needs {lag + 1} steps since the last reset; "
                f"only {self._n_since_reset} have been absorbed."
            )
        mean, cov = _smooth_lag_window(self._buffer, self._transition_matrix, lag=lag)
        return FilterState(mean, cov, self.state.t - lag, self.state.log_likelihood)
```

One compilation per distinct `lag` value (static). The window indices are
traced (`order` depends on `head`), so wraparound costs nothing extra. The
alignment matches `rts_backward_scan_with_predictions`: `predicted_*[0]` of
the window is unused and `predicted_*[i]` is the prediction the filter used at
window bin `i` (kalman.py:873-877).

Approximation behaviour to pin (planning-time measurement, 8 Hz oscillator
block at 1 kHz, damping 0.99, `Q = I`, `H = [1, 0]`, `R = 4`, 600 bins, gap
measured over bins 100–534 against `kalman_smoother`):

| L | RMS mean gap | mean trace(P) gap |
| --- | --- | --- |
| 1 | 2.14 | 7.09 |
| 2 | 2.03 | 6.37 |
| 4 | 1.84 | 5.20 |
| 8 | 1.50 | 3.45 |
| 16 | 0.99 | 1.50 |
| 32 | 0.43 | 0.28 |
| 64 | 0.079 | 0.0096 |

Filtered-vs-smoothed RMS was 2.19, so the L = 1 gap is not vacuous. The
covariance gap was strictly decreasing in L for *every* bin (it is a
deterministic, PSD-ordered quantity); the mean gap is only monotone in RMS over
bins (its per-bin sign is random), so the test asserts the RMS trend.

## D5 Point-process step and StreamingPointProcessFilter

The step mirrors the scan body of
`point_process_kalman._stochastic_point_process_filter_impl`
(point_process_kalman.py:1800-1846): symmetrise the previous covariance, predict,
build per-bin closures over the design row, call the Laplace update with a
precomputed Jacobian. It uses the public, family-generic `glm_laplace_update`
(point_process_kalman.py:1333-1464), which with `poisson_family(dt, max_log_count)`
(point_process_kalman.py:1242-1265) *is* the batch update
(`test_glm_laplace.TestPoissonParity`); the streaming loop was verified
bit-identical to `stochastic_point_process_filter` for `max_newton_iter in
{1, 3}` on a 2-latent / 4-neuron / 200-bin problem. The Jacobian is built once
per instance exactly as the batch does (point_process_kalman.py:1794-1798):

```python
        def eta(design_t, x):
            return jnp.atleast_1d(log_intensity_func(design_t, x))

        self._eta = eta
        self._grad_eta = jax.jacfwd(eta, argnums=1)
```

```python
def _make_glm_update(family, *, max_newton_iter, include_laplace_normalization, robust_weight, use_mask):
    extra = {}
    if robust_weight is not None:
        if not _accepts_kwarg(glm_laplace_update, "robust_weight"):
            raise NotImplementedError("... docs/plans/wolf-robust-updates/ ...")
        extra["robust_weight"] = robust_weight
    if use_mask and not _accepts_kwarg(glm_laplace_update, "obs_mask"):
        raise NotImplementedError("... docs/plans/masks-and-multi-sequence/ ...")

    def update(pred_mean, pred_cov, counts, eta_t, grad_eta_t, mask):
        kwargs = dict(extra)
        if use_mask:
            kwargs["obs_mask"] = mask
        result = glm_laplace_update(
            pred_mean,
            pred_cov,
            counts,
            eta_t,
            family,
            grad_eta_func=grad_eta_t,
            include_laplace_normalization=include_laplace_normalization,
            max_newton_iter=max_newton_iter,
            return_line_search_failures=True,
            **kwargs,
        )
        # (mean, cov, ll, n_failed); a robust update appends RobustOutput (contract).
        return result[:4]

    return update


@functools.partial(jax.jit, static_argnames=("eta", "grad_eta", "update"))
def _point_process_streaming_step(state, buf, n_failed_bins, counts, design_t, mask, A, Q, *, eta, grad_eta, update):
    """One predict + Laplace update; mirrors the batch point-process scan body."""
    pred_mean = A @ state.mean
    pred_cov = symmetrize(A @ symmetrize(state.cov) @ A.T + Q)

    def eta_t(x):
        return eta(design_t, x)

    def grad_eta_t(x):
        return grad_eta(design_t, x)

    mean, cov, ll_t, n_failed = update(pred_mean, pred_cov, counts, eta_t, grad_eta_t, mask)
    new_state = FilterState(mean, cov, state.t + 1, state.log_likelihood + ll_t)
    n_failed_bins = n_failed_bins + (n_failed > 0).astype(jnp.int32)
    if buf is not None:
        buf = _push_lag_buffer(buf, new_state, pred_mean, pred_cov)
    return new_state, buf, n_failed_bins
```

`n_failed_bins` is an `int32` scalar carried in and out of the compiled step
(mirrors the batch carry at point_process_kalman.py:1805, 1836) so the
diagnostic costs no extra dispatch; the instance keeps it as
`self._n_failed_bins` and exposes `int(...)` through `.n_line_search_failures`.

```python
class StreamingPointProcessFilter(_StreamingFilter):
    def __init__(
        self,
        transition_matrix: ArrayLike,
        process_cov: ArrayLike,
        init_mean: ArrayLike,
        init_cov: ArrayLike,
        log_intensity_func: Callable[[ArrayLike, ArrayLike], Array] = log_conditional_intensity,
        *,
        dt: float | None = None,
        family: GLMFamily | None = None,
        max_newton_iter: int = 3,
        include_laplace_normalization: bool = True,
        max_log_count: float = 20.0,
        robust_weight=None,
        lag: int = 0,
    ) -> None:
        # D1 validation; dtype = _common_float_dtype(A, Q, m0, P0) then astype
        # (the design rows and counts arrive per step and are left as given,
        # as the batch filter leaves them).
        if family is None:
            if dt is None:
                raise ValueError("Pass dt (Poisson family with log link) or an explicit family.")
            family = poisson_family(dt, max_log_count)   # validates dt > 0
        self.family = family
        self.max_newton_iter = validate_int(max_newton_iter, "max_newton_iter", nonnegative=True)
        ...
        self._update = _make_glm_update(family, max_newton_iter=..., include_laplace_normalization=..., robust_weight=robust_weight, use_mask=False)
        self._masked_update = None
        self._obs_shape: tuple[int, ...] | None = None   # fixed by the first step
        super().__init__(m0, P0, lag=lag)

    def _reset_counters(self) -> None:
        self._n_failed_bins = jnp.zeros((), dtype=jnp.int32)

    @property
    def n_line_search_failures(self) -> int:
        return int(self._n_failed_bins)

    def step(self, counts: ArrayLike, design_t: ArrayLike, mask: ArrayLike | None = None) -> FilterState:
        counts = jnp.asarray(counts)
        if counts.ndim != 1:
            raise ValueError(...)
        if self._obs_shape is None:
            self._obs_shape = counts.shape
        elif counts.shape != self._obs_shape:
            raise ValueError(...)   # would otherwise retrace silently
        design_t = jnp.asarray(design_t)
        ... select update / mask variant as in D2 ...
        new_state, self._buffer, self._n_failed_bins = _point_process_streaming_step(
            self.state, self._buffer, self._n_failed_bins, counts, design_t, mask_arr,
            self._transition_matrix, self._process_cov,
            eta=self._eta, grad_eta=self._grad_eta, update=update,
        )
        return self._advance(new_state)

    def run(self, design_matrix: ArrayLike, spikes: ArrayLike) -> tuple[Array, Array, Array]:
        """Mirror of ``stochastic_point_process_filter``'s dense path."""
        spikes = jnp.asarray(spikes)
        validate_count_array(spikes, "spikes", allow_empty=False)
        if spikes.ndim == 1:
            spikes = spikes[:, None]          # single-neuron promotion, as the batch does
        design_matrix = jnp.asarray(design_matrix)
        n_failed_before = self._n_failed_bins
        ll_before = self.state.log_likelihood
        means, covs = [], []
        for design_t, counts in zip(design_matrix, spikes):
            state = self.step(counts, design_t)
            means.append(state.mean)
            covs.append(state.cov)
        _warn_line_search_failures(
            self._n_failed_bins - n_failed_before, int(spikes.shape[0]),
            self.max_newton_iter, "StreamingPointProcessFilter.run",
        )
        return jnp.stack(means), jnp.stack(covs), self.state.log_likelihood - ll_before
```

`_warn_line_search_failures` (point_process_kalman.py:882-924) is jit-safe and
eager-safe; it logs when more than 10 % of the bins failed, as the batch does.

## D6 Position decoder step extraction and StreamingPositionDecoder

### Why extract

`position_decoder._run_filter_scan` (position_decoder.py:969-1178) builds the
log-intensity / Jacobian / track-penalty closures (1022-1082) and defines the
scan step inline (1084-1153): predict (1087-1094), Woodbury track-penalty
downdate (1096-1107), adaptive inflation (1109-1131), Laplace update
(1133-1144). A streaming decoder that re-implemented this body would be a
second copy of decoding logic that has already drifted once (adaptive
inflation was added to the filter and then had to be threaded into the
smoother). Phase 3b therefore moves the body into two module-level functions
that both the batch scan and the streaming step call. The refactor is
behaviour-preserving and must be verified bit-identical on a captured
baseline before the streaming class is added.

### New functions in `position_decoder.py`

```python
def _build_decoder_functions(
    jax_log_rate_maps, jax_x_edges, jax_y_edges, track_penalty, kde_args, *,
    sigma_track, grid_dx, grid_dy, n_neurons, include_velocity, use_kde,
):
    """Closures for one decoding problem.

    Returns ``(log_intensity_func, grad_log_intensity_func, penalty_value_fn,
    penalty_grad_fn)``; the bodies are the current lines 1022-1082 of
    ``_run_filter_scan`` moved verbatim (``_vel_pad``, ``jax_penalty_map``,
    ``_x_min`` ... ``_penalty_inv_sigma2`` become locals here).
    """


def _decoder_step(
    mean_prev, cov_prev, spike_t, *, A, Q, dt, fns, infl_args, inflate, n_state,
    max_newton_iter, robust_weight=None, obs_mask=None,
):
    """Predict, track-penalty downdate, adaptive inflation, Laplace update for one bin.

    Body: current lines 1087-1144 verbatim, with ``log_intensity_func,
    grad_log_intensity_func, penalty_value_fn, penalty_grad_fn = fns`` and the
    ``_infl_gain, _infl_max, _infl_eps, _infl_min_ft = infl_args`` /
    ``_infl_d = jnp.array(2.0)`` unpacking (current lines 1023-1024) at the top.
    Returns ``(post_mean, post_cov, ll, n_failed, capped, dynamics_mean,
    dynamics_cov)`` where ``capped`` is the bool from line 1126 when
    ``inflate`` and ``jnp.zeros((), dtype=bool)`` otherwise. ``robust_weight``
    / ``obs_mask`` are appended to the ``_point_process_laplace_update`` call
    only when not None (a trace-time Python branch), so the batch call site,
    which passes neither, emits the identical call.
    """
```

`_point_process_laplace_update` must remain a **module-global lookup** inside
`_decoder_step` (not bound as a default argument or captured at import in a
partial): `tests/test_position_decoder.py:2024-2053` monkeypatches
`position_decoder._point_process_laplace_update` to count traces.

`_run_filter_scan` keeps its name, its `jax.jit` decoration and static
argument list (the same test calls `_run_filter_scan.clear_cache()`) and
becomes:

```python
    fns = _build_decoder_functions(
        jax_log_rate_maps, jax_x_edges, jax_y_edges, track_penalty, kde_args,
        sigma_track=sigma_track, grid_dx=grid_dx, grid_dy=grid_dy,
        n_neurons=n_neurons, include_velocity=include_velocity, use_kde=use_kde,
    )

    def _step(carry, spike_t):
        mean_prev, cov_prev, total_ll, n_failed_bins, n_capped_bins = carry
        post_mean, post_cov, ll, n_failed, capped, dynamics_mean, dynamics_cov = _decoder_step(
            mean_prev, cov_prev, spike_t, A=A, Q=Q, dt=dt, fns=fns, infl_args=infl_args,
            inflate=inflate, n_state=n_state, max_newton_iter=max_newton_iter,
        )
        total_ll = total_ll + ll
        n_failed_bins = n_failed_bins + (n_failed > 0).astype(jnp.int32)
        n_capped_bins = n_capped_bins + capped.astype(jnp.int32)
        return (post_mean, post_cov, total_ll, n_failed_bins, n_capped_bins), (
            post_mean, post_cov, dynamics_mean, dynamics_cov,
        )
    # scan call and the trailing _warn_line_search_failures / return unchanged (1155-1178)
```

(Adding an `int32` zero to `n_capped_bins` on the no-inflation path is
bit-identical to not adding.)

From `_position_decoder_filter_with_predictions` (position_decoder.py:1260-1536)
extract, keeping the existing bodies and error messages:

```python
def _resolve_decoder_init(rate_maps, include_velocity, init_position, init_cov, sigma_track, n_state) -> tuple[Array, Array]:
    """Lines 1309-1370 (defaults: arena centre / tight-vs-loose prior) and 1430-1439 (shape + validate_covariance)."""

def _resolve_track_penalty(rate_maps, track_penalty, sigma_track) -> Array:
    """Lines 1380-1391."""

def _kde_args(rate_maps) -> tuple:
    """Lines 1393-1416 (real KDE statistics or zero-shaped sentinels)."""

def _inflation_args(adaptive_inflation) -> tuple[bool, tuple]:
    """Lines 1418-1428; returns (inflate, infl_args)."""
```

and call them from `_position_decoder_filter_with_predictions` in place of the
inlined code. Everything after the scan (cap-fraction warning 1472-1484,
escape warning 1486-1529, `DecoderResult`) is untouched.

### Streaming step and class (in `streaming.py`)

```python
@functools.partial(jax.jit, static_argnames=("fns", "inflate", "n_state", "max_newton_iter"))
def _decoder_streaming_step(state, buf, counters, counts, mask, robust_weight, A, Q, dt, infl_args, *, fns, inflate, n_state, max_newton_iter):
    post_mean, post_cov, ll, n_failed, capped, dyn_mean, dyn_cov = _decoder_step(
        state.mean, state.cov, counts, A=A, Q=Q, dt=dt, fns=fns, infl_args=infl_args,
        inflate=inflate, n_state=n_state, max_newton_iter=max_newton_iter,
        robust_weight=robust_weight, obs_mask=mask,
    )
    new_state = FilterState(post_mean, post_cov, state.t + 1, state.log_likelihood + ll)
    n_failed_bins, n_capped_bins = counters
    counters = (
        n_failed_bins + (n_failed > 0).astype(jnp.int32),
        n_capped_bins + capped.astype(jnp.int32),
    )
    if buf is not None:
        buf = _push_lag_buffer(buf, new_state, dyn_mean, dyn_cov)   # inflated dynamics prediction, not the penalised one
    return new_state, buf, counters
```

`fns` is the 4-tuple of closures from `_build_decoder_functions`, built once
per instance (static, hashed by identity). `dt` is traced (as in the batch);
`sigma_track` and the grid spacing are Python floats baked into the closures
(traced scalars in the batch) — bit-identical on the bilinear path, verified.
`mask` / `robust_weight` are `None` unless supplied; the host-side check that
`_point_process_laplace_update` accepts `obs_mask` / `robust_weight` happens
in the class before the first use (same `NotImplementedError` wording as D2).

```python
class StreamingPositionDecoder(_StreamingFilter):
    def __init__(
        self, rate_maps: PlaceFieldRateMaps, dt: float, *, q_pos: float | None = None,
        q_vel: float = 10.0, include_velocity: bool = True, init_position=None,
        init_cov=None, track_penalty=None, sigma_track: float = 5.0,
        max_newton_iter: int = 3, adaptive_inflation: AdaptiveInflationConfig | None = None,
        robust_weight=None, lag: int = 0,
    ) -> None:
        # Mirrors _position_decoder_filter_with_predictions lines 1297-1439 via the helpers:
        if q_pos is None:
            q_pos = rate_maps.suggested_q_pos if rate_maps.suggested_q_pos is not None else 100.0
        q_pos = validate_scalar(q_pos, "q_pos", nonnegative=True)
        q_vel = validate_scalar(q_vel, "q_vel", nonnegative=True)
        A, Q = build_position_dynamics(dt, q_pos, q_vel, include_velocity)
        n_state = A.shape[0]
        init_position, init_cov = _resolve_decoder_init(rate_maps, include_velocity, init_position, init_cov, sigma_track, n_state)
        track_penalty = _resolve_track_penalty(rate_maps, track_penalty, sigma_track)
        self._inflate, self._infl_args = _inflation_args(adaptive_inflation)
        self._fns = _build_decoder_functions(
            rate_maps._jax_log_rate_maps, rate_maps._jax_x_edges, rate_maps._jax_y_edges,
            track_penalty, _kde_args(rate_maps), sigma_track=sigma_track,
            grid_dx=rate_maps._dx, grid_dy=rate_maps._dy, n_neurons=rate_maps.n_neurons,
            include_velocity=include_velocity, use_kde=rate_maps._use_analytical,
        )
        self.rate_maps, self.dt, self.n_neurons, self.n_state = rate_maps, dt, rate_maps.n_neurons, n_state
        self.max_newton_iter = validate_int(max_newton_iter, "max_newton_iter", nonnegative=True)
        self._transition_matrix, self._process_cov = A, Q
        self._robust_weight = robust_weight   # checked against _point_process_laplace_update's signature here
        super().__init__(init_position, init_cov, lag=lag)

    @classmethod
    def from_decoder(cls, decoder: PositionDecoder, *, init_position=None, init_cov=None, lag: int = 0) -> "StreamingPositionDecoder":
        if decoder.rate_maps is None:
            raise NotFittedError("PositionDecoder has no rate maps; call fit() or fit_from_model() first.")
        return cls(decoder.rate_maps, decoder.dt, q_pos=decoder.q_pos, q_vel=decoder.q_vel,
                   include_velocity=decoder.include_velocity, init_position=init_position,
                   init_cov=init_cov, max_newton_iter=decoder.max_newton_iter,
                   adaptive_inflation=decoder.adaptive_inflation, lag=lag)

    def _reset_counters(self) -> None:
        self._counters = (jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32))

    n_line_search_failures / n_inflation_capped_bins: int(...) of the two counters
    position_xy property: self.state.mean[:2]

    def step(self, counts, mask=None) -> FilterState:
        counts = jnp.asarray(counts); shape must be (n_neurons,) else ValueError
        new_state, self._buffer, self._counters = _decoder_streaming_step(
            self.state, self._buffer, self._counters, counts, mask_arr, self._robust_weight,
            self._transition_matrix, self._process_cov, self.dt, self._infl_args,
            fns=self._fns, inflate=self._inflate, n_state=self.n_state, max_newton_iter=self.max_newton_iter,
        )
        return self._advance(new_state)

    def run(self, spikes) -> DecoderResult:
        spikes = jnp.asarray(spikes); 1-D -> (T, 1); validate_count_array(spikes, "spikes", allow_empty=False); column count == n_neurons
        loop step; _warn_line_search_failures(n_failed_this_run, T, max_newton_iter, "StreamingPositionDecoder.run")
        return DecoderResult(position_mean=stack(means), position_cov=stack(covs), marginal_log_likelihood=float(ll_after - ll_before))
```

`.run()` deliberately does **not** re-emit the batch entry point's
inflation-cap-fraction and left-the-arena warnings (position_decoder.py:1472-1529):
those are offline diagnostics over a whole recording; the counters are
exposed instead.

## D7 Latency benchmark script

`scripts/benchmark_streaming_latency.py` (dev-only; `scripts/**` is formatted
but not linted by ruff). It measures, it does not assert:

```python
def time_steps(step, inputs, *, n_warmup: int = 5, n_timed: int = 500) -> tuple[float, float]:
    """Median and 95th-percentile wall-clock per call in microseconds, after warm-up."""
    for i in range(n_warmup):
        jax.block_until_ready(step(inputs[i % len(inputs)]))
    times = []
    for i in range(n_timed):
        t0 = time.perf_counter()
        jax.block_until_ready(step(inputs[i % len(inputs)]))
        times.append(time.perf_counter() - t0)
    return 1e6 * float(np.median(times)), 1e6 * float(np.percentile(times, 95))
```

Sections (one Markdown table each, plus the dispatch floor of a jitted scalar
add): Gaussian `n_state ∈ {2, 4, 8} × n_obs ∈ {1, 16, 100}` on random stable
`A` (spectral radius 0.9), `Q = 0.3 I + 0.1`, random `H`, `R = 0.5 I`
(Phase 1); point-process, same grid over `n_neurons` with the default linear
log-intensity, `max_newton_iter = 3` (Phase 3a); decoder, `n_neurons ∈ {1, 16,
100}` for `include_velocity ∈ {False, True}` on the KDE path (`n_grid = 50`)
and the bilinear path (Phase 3b). Expected results are in
[overview.md — Metrics](overview.md#metrics); the executor pastes the measured
table into each PR description and flags any row more than 2× slower than
expected.
