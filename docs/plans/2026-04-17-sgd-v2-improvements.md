# SGD v2 Improvements — log_det_jacobian + lax.scan Inner Loop

> **Status:** PLANNED as of 2026-04-17. Follow-on to [2026-04-06-sgd-fitting-all-models.md](2026-04-06-sgd-fitting-all-models.md).

**Goal:** Two targeted upgrades to [sgd_fitting.py](../../src/state_space_practice/sgd_fitting.py) / [parameter_transforms.py](../../src/state_space_practice/parameter_transforms.py) — the two specific things we'd steal from dynamax after reviewing their implementation:

1. **`log_det_jacobian` on `ParameterTransform`** — unblocks MAP / HMC / black-box VI without switching to TFP.
2. **`lax.scan` inner loop** — speed matters on long time series, and a Python-for loop dispatching 200 JIT calls is measurable overhead at our trajectory lengths.

Both are additive; no API break for existing callers.

---

## Motivation

### log_det_jacobian

Our [ParameterTransform](../../src/state_space_practice/parameter_transforms.py#L23) has a forward map and an inverse map but no Jacobian log-det. This is fine for pure MLE (the log-det is a constant with respect to gradient-based MLE, so it drops out), but it blocks:

- **MAP estimation** with informative priors — the Jacobian correction is needed to express the prior in unconstrained space.
- **HMC / NUTS** — needs correct volume in the unconstrained space.
- **Black-box VI** — same reason.

dynamax inherits this for free from TFP bijectors via `forward_log_det_jacobian` / `inverse_log_det_jacobian`. We can get the same property for the 5 transforms we actually use with ~30 lines of hand-rolled math, without taking a TFP dependency.

### lax.scan inner loop

Our current SGD loop is a Python `for` over num_steps, dispatching a jitted `train_step` per iteration ([sgd_fitting.py:132-156](../../src/state_space_practice/sgd_fitting.py#L132-L156)). For our typical workload — one long trajectory (`n_timesteps` = 10k–100k time bins), 200 SGD steps — the per-step Python overhead is small as a fraction but non-zero, and it accumulates:

- Each step: one jit trace-cache lookup, one XLA dispatch, one host↔device sync to read `loss` for the finite check.
- The host↔device sync is the expensive part. `jnp.isfinite(loss)` forces the scalar back to host before the next step can launch, blocking pipelining.

dynamax's `run_gradient_descent` uses `lax.scan` for exactly this reason — the whole SGD loop compiles into a single XLA graph with no host sync per step. But we can't naively adopt it because we have NaN recovery and convergence-check logic that both require host-side control flow.

Strategy: **chunked scan**. Scan `chunk_size` (e.g., 20) steps inside one jit, emit the per-step LLs and final params/opt_state, do the finite + convergence checks on host between chunks. Keeps NaN recovery + convergence check working; eliminates 95% of host syncs.

---

## Task 1: `log_det_jacobian` on `ParameterTransform`

### 1.1 Extend the dataclass

```python
@dataclass(frozen=True)
class ParameterTransform:
    to_unconstrained: Callable[[Array], Array]
    to_constrained: Callable[[Array], Array]
    log_det_jacobian: Callable[[Array], Array] | None = None  # NEW
    trainable: bool = True
```

The log-det convention: **given unconstrained input `u`, return `log|det(dθ/du)|`** where `θ = to_constrained(u)`. This is the "forward log-det" — what you add to a log-density of `θ` to convert it to a log-density of `u`.

Leave `None` as the default so existing transforms that don't need it keep working unchanged. Callers that need it check `if transform.log_det_jacobian is None: raise NotImplementedError(...)`.

### 1.2 Fill in log-dets for each transform

Scalar transforms (each element contributes independently — sum over elements):

| Transform | `to_constrained(u)` | `log_det_jacobian(u)` |
|---|---|---|
| `POSITIVE` (softplus) | `log(1 + e^u)` | `sum(-softplus(-u))` = `sum(log_sigmoid(u))` |
| `UNIT_INTERVAL` (sigmoid) | `1/(1+e^-u)` | `sum(-softplus(u) - softplus(-u))` = `sum(log_sigmoid(u) + log_sigmoid(-u))` |
| `UNCONSTRAINED` | `u` | `0` |
| `positive_capped(max)` | `min(softplus(u), max)` | piecewise: same as `POSITIVE` below the cap, `-inf` at saturation. Document that HMC on a capped parameter is ill-posed and raise if called at saturation. |

Matrix transforms (use the closed-form Jacobians from the standard bijector catalog):

- **`PSD_MATRIX`** (unconstrained reals → Cholesky-with-log-diag → PSD): the forward map is `L @ L.T` where `L` is reconstructed by placing the flat vector into the lower-triangle and exponentiating the diagonal. Log-det of the flat→PSD Jacobian is `n * log(2) + sum((n - i) * L_diag[i])` + `sum(L_diag)` (standard result for the `FillScaleTriL ∘ CholeskyOuterProduct` composition in TFP). Reference: TFP [`FillScaleTriL`](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/FillScaleTriL) log-det formula.
- **`STOCHASTIC_ROW`** (drop-last-column logits → softmax): Jacobian of the softmax-centered bijector is well-known — `sum(log(p_i))` over the kept components minus appropriate correction. Reference: TFP [`SoftmaxCentered`](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/SoftmaxCentered).

### 1.3 Compose via pytree sum

Add a helper that sums log-dets across a pytree of transforms:

```python
def sum_log_det_jacobian(unc_params: dict, spec: dict) -> Array:
    """Sum log|det J| across all trainable parameters for MAP/HMC."""
    total = jnp.array(0.0)
    for k, v in unc_params.items():
        if not spec[k].trainable:
            continue
        if spec[k].log_det_jacobian is None:
            raise NotImplementedError(
                f"Transform for {k!r} does not provide log_det_jacobian. "
                "Required for MAP/HMC/VI."
            )
        total = total + spec[k].log_det_jacobian(v)
    return total
```

Frozen parameters skip the sum (their unconstrained values are not sampled).

### 1.4 Tests ([tests/test_parameter_transforms.py](../../src/state_space_practice/tests/test_parameter_transforms.py))

For each transform with a log-det:

- **Numerical-Jacobian check**: use `jax.jacfwd(to_constrained)` at a random `u`, compute `jnp.linalg.slogdet` (for matrix transforms) or `jnp.sum(jnp.log(jnp.abs(jac)))` (for scalar transforms), assert equal to `log_det_jacobian(u)` within `1e-6`.
- **Composition**: verify `sum_log_det_jacobian` sums correctly across a synthetic param dict with 2–3 different transforms.
- **Frozen exclusion**: verify frozen params are skipped.
- **Shape**: `log_det_jacobian` returns a scalar (0-d Array) for any valid input.

### 1.5 Docstring update

Update the module docstring in [parameter_transforms.py](../../src/state_space_practice/parameter_transforms.py) to note that log-det is available for MAP/HMC. No usage change for MLE callers.

---

## Task 2: Chunked `lax.scan` inner loop in `fit_sgd`

### 2.1 Refactor the training step into a scan body

```python
def _scan_body(carry, _):
    unc_p, opt_st = carry
    loss, grads = jax.value_and_grad(_loss_inner)(unc_p)
    updates, new_opt_st = optimizer.update(grads, opt_st, unc_p)
    new_unc_p = optax.apply_updates(unc_p, updates)
    return (new_unc_p, new_opt_st), loss  # emit per-step loss

@jax.jit
def scan_chunk(unc_p, opt_st, n_steps):
    (unc_p, opt_st), losses = jax.lax.scan(
        _scan_body, (unc_p, opt_st), xs=None, length=n_steps,
    )
    return unc_p, opt_st, losses
```

`n_steps` as a concrete int (compiled per distinct chunk size — but we use one fixed chunk size, so only one compile).

### 2.2 Outer Python driver handles safety and convergence

```python
CHUNK_SIZE = 20  # tunable; see 2.4

log_likelihoods: list[float] = []
last_valid = unc_params
stall_count = 0

steps_remaining = num_steps
while steps_remaining > 0:
    n = min(CHUNK_SIZE, steps_remaining)
    candidate_unc, candidate_opt, chunk_losses = scan_chunk(
        unc_params, opt_state, n
    )
    chunk_losses_host = np.asarray(chunk_losses)  # single sync per chunk

    # NaN detection: find first non-finite loss in the chunk
    finite_mask = np.isfinite(chunk_losses_host)
    if not finite_mask.all():
        first_bad = int(np.argmin(finite_mask))
        # Accept up to (but not including) first_bad
        if first_bad == 0:
            # First step of this chunk already bad — roll back entirely.
            logger.warning("SGD step %d: NaN loss — restoring last valid params",
                           num_steps - steps_remaining)
            unc_params = last_valid
        else:
            # Replay first_bad steps to recover the good intermediate state.
            # Cheaper than saving every-step state: re-run a shorter scan.
            unc_params, opt_state, good_losses = scan_chunk(
                unc_params, opt_state, first_bad
            )
            log_likelihoods.extend(
                (-np.asarray(good_losses) * n_timesteps).tolist()
            )
            last_valid = unc_params
        break

    # All steps in this chunk finite — commit.
    last_valid = unc_params   # snapshot BEFORE swapping, same invariant as today
    unc_params = candidate_unc
    opt_state = candidate_opt
    log_likelihoods.extend((-chunk_losses_host * n_timesteps).tolist())
    steps_remaining -= n

    # Convergence check (same logic as today, vectorized over the chunk)
    if convergence_tol is not None and len(log_likelihoods) >= 2:
        lls = np.asarray(log_likelihoods[-CHUNK_SIZE - 1:])
        avg = (np.abs(lls[:-1]) + np.abs(lls[1:])) / 2
        rel = np.abs(np.diff(lls)) / np.maximum(avg, 1e-10)
        stalled_run = np.concatenate([[stall_count], (rel < convergence_tol).cumsum()])
        if stalled_run.max() >= 5:
            break
        stall_count = int(stalled_run[-1])  # carry forward for next chunk
```

The convergence logic above needs careful verification — the "5 consecutive stalled steps" check doesn't decompose trivially across chunks. Simpler correct version: at the end of each chunk, re-evaluate `stall_count` by walking `log_likelihoods[-6:]` on the host. Exact match to current behavior.

### 2.3 Invariants preserved

- NaN recovery rolls back to a set of params we *observed* to produce finite loss (the "snapshot before swap" property from [sgd_fitting.py:146-152](../../src/state_space_practice/sgd_fitting.py#L146-L152) — carry it through unchanged).
- Verbose logging now happens per-chunk instead of per-step. Explicitly note this in the docstring — current `verbose and step % 10 == 0` becomes `verbose and chunk_end % chunk_size == 0`. Keep the same cadence by setting `CHUNK_SIZE = 10` or make chunk size the verbose cadence.
- `convergence_tol` behavior is bit-identical (verified by the reconstruction above).
- `log_likelihood_history_` is populated step-by-step as before.

### 2.4 Picking `CHUNK_SIZE`

Tradeoff: larger chunks = fewer host syncs = faster; but coarser NaN granularity and worse early-stop responsiveness.

- `CHUNK_SIZE = 1` reproduces current behavior (host sync per step).
- `CHUNK_SIZE = num_steps` reproduces `run_gradient_descent` (one scan, no NaN recovery).
- Sweet spot likely 10–50 based on dynamax precedent and host-sync latency.

Add as a keyword arg `chunk_size: int = 20` with docstring guidance:

- Raise if filter is prone to NaN (e.g., `PlaceFieldModel` at `n_basis >= 7`).
- Lower if `num_steps` is small (e.g., <50) — fewer chunks, less benefit.

### 2.5 Benchmarks before merging

Before changing the default, benchmark on three representative fits:

1. **[place_field_model](../../src/state_space_practice/place_field_model.py)** on real CA1 data, `n_basis=36, n_neurons=8, n_time=50_000, num_steps=200` — current baseline established in [2026-03-24-place-field-model-baseline-benchmark](../../docs/plans/) (wherever it lives, see commit 24f5308).
2. **[oscillator_models](../../src/state_space_practice/oscillator_models.py)** on a 4-oscillator coupled network, `n_time=10_000, num_steps=500`.
3. **[switching_point_process](../../src/state_space_practice/switching_point_process.py)** 2-state, `n_time=20_000, num_steps=100`.

Record wall-clock for `chunk_size ∈ {1, 10, 20, 50, 100}`. Ship the fastest chunk size that also passes all existing tests. If chunk_size=1 is within 5% of the best, don't change the default — the host-sync optimization isn't material at our scale and the added complexity isn't worth it.

### 2.6 Tests

- **Bit-identical outputs vs current implementation** for a fixed-seed fit at `chunk_size=1`: regression guard that the scan body is equivalent to the Python-loop body.
- **Matching-within-tolerance outputs** at `chunk_size > 1`: final params should match to `atol=1e-10` (no algorithmic difference, just graph compilation order).
- **NaN recovery correctness**: inject a non-finite loss at step k, assert the recovered params equal `last_valid` from step k-1.
- **Convergence-tol match**: fixed-seed fit with tight tolerance, assert the stop step number matches the Python-loop version.

---

## Non-goals

- **Not** adopting TFP bijectors wholesale. Adding log-det to our five transforms is 30 lines; switching to TFP is a heavy dependency (either full TF or TFP-substrates-for-JAX).
- **Not** adding minibatching over sessions. Our core workload is single long trajectories; if we ever need multi-session fitting, that's a separate plan with different data-shape conventions.
- **Not** adopting dynamax's `NamedTuple` param trees. Cosmetic; doesn't change perf or correctness.

## Risks

- **lax.scan compilation time**: a 200-step scan compiles into a much larger XLA graph than a single step. First-fit compile time may jump from 5s to 30s+. Mitigate with chunk_size — a 20-step chunk compiles once and is reused `num_steps/20` times. Benchmark compile time alongside wall-clock.
- **Scan body JIT tracing of `self`**: the existing comment at [sgd_fitting.py:102-109](../../src/state_space_practice/sgd_fitting.py#L102-L109) warns that `self` must not mutate inside `_sgd_loss_fn`. Same invariant applies under scan — reaffirm in docstring.
- **Chunked convergence-check semantics**: the "5 consecutive stalled steps spanning a chunk boundary" case is subtle. The reconstruction at 2.2 handles it, but verify with a targeted test where tolerance stalls for exactly 4 steps in chunk N and 1 step in chunk N+1.

## Dependencies

- **None for log_det_jacobian** — pure JAX math.
- **None for lax.scan refactor** — already using `jax.lax.scan` elsewhere in the codebase.

## Order of execution

1. Task 1 (`log_det_jacobian`) — self-contained, no risk to existing fits. Merge first.
2. Task 2 benchmarks — measure before changing defaults. If benefit is <5%, document and close without merging.
3. Task 2 implementation + tests — only if benchmarks justify.
