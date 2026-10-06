# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

Used unchanged (the streaming steps call these; any change to them changes
the streaming filters identically):

- `src/state_space_practice/kalman.py:426-475` — `kalman_measurement_update`: the Gaussian measurement update the streaming step calls after an inline prediction.
- `src/state_space_practice/kalman.py:478-526` — `_kalman_filter_update` (public alias `kalman_filter_update`, kalman.py:1631-1634): its prediction (517-526) is mirrored, not called, so the prediction is available to the fixed-lag buffer. Untouched.
- `src/state_space_practice/kalman.py:529-610` — `_kalman_filter_impl`: the scan body (570-594) the Gaussian parity test compares against; dtype promotion (551-567) mirrored at construction. Untouched.
- `src/state_space_practice/kalman.py:845-912` — `rts_backward_scan_with_predictions`: the fixed-lag smoother's backward pass for all three filters. `rts_backward_scan` (778-842) is not used (it recomputes the prediction, which is wrong for the inflated decoder).
- `src/state_space_practice/point_process_kalman.py:1333-1464` — `glm_laplace_update`, `:1242-1265` `poisson_family`, `:1292-1297` `BERNOULLI_LOGIT_FAMILY`: the point-process step. `:1800-1846` scan body mirrored; `:882-924` `_warn_line_search_failures` reused by `.run()`. Untouched.
- `src/state_space_practice/utils.py:1308-1405` `validate_covariance`, `:1113` `validate_int`, `:1016` `validate_count_array`, `:1161` `validate_scalar`, `:26` `symmetrize`: construction-time validation. Untouched.
- `src/state_space_practice/oscillator_utils.py:55-73` `get_block_slice`: oscillator block addressing for the phase posterior. Untouched.
- `src/state_space_practice/circular_stats.py:45-72`, `:75-102`, `:105-121`, `:410-424`: NumPy references the phase-posterior tests compare against. Untouched.

Modified:

- `src/state_space_practice/position_decoder.py:958-1178` — `_run_filter_scan`: closures (1022-1082) and step body (1084-1153) move to module-level `_build_decoder_functions` / `_decoder_step` and are called from the scan; name, `jax.jit`, static args and outputs preserved; behaviour bitwise-preserved (Phase 3b).
- `src/state_space_practice/position_decoder.py:1260-1536` — `_position_decoder_filter_with_predictions`: init resolution (1309-1370, 1430-1439), track penalty (1380-1391), KDE args (1393-1416) and inflation args (1418-1428) move to helpers it calls; host-side warnings (1472-1529) untouched (Phase 3b).
- `src/state_space_practice/__init__.py:35-53`, `:55-76`, `:78-100` — lazy exports for the new public names (each phase adds its own).
- `pyproject.toml:134-159` — `[tool.mypy] files` gains `streaming.py` (Phase 1).
- `CHANGELOG.md:8` (`### Added`), `README.md:60-68` (Package layout + a new streaming subsection) — per phase.

New:

- `src/state_space_practice/streaming.py` — the whole public surface of this plan.
- `src/state_space_practice/tests/test_streaming.py`; new test classes in `tests/test_likelihood_identities.py` (Phases 1, 3a) and `tests/test_position_decoder.py` (Phase 3b).
- `scripts/benchmark_streaming_latency.py`.

Left alone on purpose: `position_decoder_filter` (1181-1257), `position_decoder_smoother` (1539-1612), `PositionDecoder` (1615-1841), `stochastic_point_process_filter`'s block-diagonal path (1648-1703), the switching filters, `models.py`'s deprecated filter.

## Scope and dependency policy

### Goals

- A per-bin ("streaming") API for the Gaussian, point-process and
  position-decoder filters: one `jax.jit`-compiled step on an explicit
  `FilterState`, compiled once, no per-step host checks or allocations beyond
  the returned state.
- Exact agreement with the batch filters: a streaming run's filtered moments
  and summed log-likelihood equal the batch filter's (bit-identical where the
  same jaxpr runs; see the [parity policy](shared-contracts.md#parity-policy)).
- Fixed-lag smoothing with a bounded ring buffer, using the library's
  prediction-aware RTS pass so the decoder's inflated predictions are honoured.
- Phase and amplitude posteriors of a 2-D oscillator block (real-time theta
  phase) with delta-method circular credible intervals.
- A measured per-step latency table and honest guidance on what per-bin
  budgets are reachable on CPU.

### Non-Goals

- Acquisition-system integration (Trodes, network sockets), threading, timers,
  or any event loop. The API is a plain Python object; the caller owns the loop.
- Online parameter adaptation (updating `A`, `Q`, `R`, rate maps, or place
  fields while streaming); switching (SLDS) streaming filters; the
  block-diagonal point-process path; time-varying `R_t` per bin.
- Sub-0.1 ms per-bin latency guarantees. The plan measures and documents; it
  does not add batching, fusion or a C++ loop.
- Any masking or robust-weight *semantics* — those belong to the parallel
  plans below and are only forwarded.
- Float32 support beyond a warning: the batch filters' float64 requirement
  applies with more force to an unbounded stream.

### Dependency policy

No new runtime or test dependencies. Two parallel plans are **optional**
dependencies, forwarded through keyword detection and otherwise reported with
`NotImplementedError` (rules in
[shared-contracts — Optional cross-plan keyword forwarding](shared-contracts.md#optional-cross-plan-keyword-forwarding)):

- `docs/plans/wolf-robust-updates/` — adds `robust_weight=` to
  `kalman_measurement_update` and `glm_laplace_update`; `StreamingKalmanFilter`,
  `StreamingPointProcessFilter` and `StreamingPositionDecoder` take
  `robust_weight=None` and pass it through.
- `docs/plans/masks-and-multi-sequence/` — adds `obs_mask=`; every `.step()`
  takes `mask=None` and passes it through as `obs_mask`.

Neither plan's content is restated here. If either lands first, the
`skipif`-guarded semantics tests in this plan start running automatically.

## Metrics

Correctness:

- Gaussian and point-process parity: bit-identical means, covariances and
  summed LL vs `kalman_filter` / `stochastic_point_process_filter` (dense path)
  on 300- and 25-bin problems across four shape pairs.
- Decoder parity: bit-identical on the bilinear rate-map path; KDE path within
  `rtol=1e-11` (observed 2.8e-14 on means, 5.7e-14 on LL).
- Likelihood identities: streaming summed LL equals the batch LL exactly;
  chain rule by `reset` to a filtered state holds to `rtol=1e-10`.
- Fixed-lag: full-window `smoothed()` equals the batch smoother to `rtol=1e-12`
  (observed 1e-16); the RMS gap to the batch smoother and the per-bin
  covariance gap decrease monotonically over `L ∈ {1, 2, 4, 8, 16, 32}`
  (expected RMS mean gap 2.14 → 0.43, trace gap 7.09 → 0.28 on the D4 problem).
- Phase posterior: delta-method sd within 3 % of Monte Carlo at
  amplitude/sd = 30 (observed 0.1 %), 95 % interval coverage in `[0.94, 0.96]`;
  error monotone in falling SNR (observed 0.001, 0.013, 0.061, 0.197 at 30, 10,
  5, 3).

Latency (expected results; planning-time measurement on the planner's laptop,
Apple M5 Max, macOS, CPU-only JAX 0.10.2, float64, median / p95 of 200 timed
steps after warm-up, `block_until_ready` per step; the executor re-measures
with `scripts/benchmark_streaming_latency.py` and flags rows > 2× slower):

| Filter | n_state | n_obs or n_neurons | median µs | p95 µs |
| --- | --- | --- | --- | --- |
| jit dispatch floor (scalar add) | — | — | 2.4 | — |
| StreamingKalmanFilter | 2 | 1 | 7.6 | 8.2 |
| | 2 | 16 | 22.3 | 31.1 |
| | 2 | 100 | 77.0 | 98.0 |
| | 4 | 1 | 8.6 | 10.8 |
| | 4 | 16 | 24.0 | 38.9 |
| | 4 | 100 | 73.4 | 89.9 |
| | 8 | 1 | 20.4 | 28.2 |
| | 8 | 16 | 23.9 | 32.0 |
| | 8 | 100 | 80.2 | 97.4 |
| StreamingPointProcessFilter (linear log-rate, 3 Fisher iterations) | 2 | 1 | 50.4 | 80.0 |
| | 2 | 16 | 69.0 | 122.4 |
| | 2 | 100 | 83.8 | 137.2 |
| | 4 | 1 | 66.4 | 108.2 |
| | 4 | 16 | 78.4 | 142.1 |
| | 4 | 100 | 92.5 | 137.1 |
| | 8 | 1 | 68.2 | 99.8 |
| | 8 | 16 | 82.8 | 152.9 |
| | 8 | 100 | 96.5 | 287.8 |
| StreamingPositionDecoder (KDE path, 50×50 grid, inflation on, 3 iterations) | 2 | 1 | 470 | 513 |
| | 2 | 16 | 593 | 663 |
| | 2 | 100 | 1350 | 1505 |
| | 4 | 1 | 453 | 499 |
| | 4 | 16 | 608 | 904 |
| | 4 | 100 | 1350 | 1519 |

Reading: per-step cost is dominated by Python-side `jit` dispatch and
tiny-kernel launch overhead — the floor is a few µs on this machine (tens of
µs on slower hosts and on any machine when the process is not otherwise idle),
the smallest Gaussian step ≈ 8 µs, the smallest Laplace step ≈ 50 µs. A
per-bin budget below ~0.1 ms is therefore reachable only by the Gaussian
filter at small sizes and never by the point-process filters; it cannot be
reached by micro-optimising the step. Callers with such budgets should batch
several bins per call (`.run()` on a chunk, or a `lax.scan` over a chunk in
their own code). The decoder's KDE path is 0.5–1.4 ms per bin (the kernel sum
over 2500 grid points, three times per bin plus its Jacobian); it fits 4 ms
bins, not 1 ms bins — the bilinear rate-map path is the alternative and the
benchmark reports both. The README documents this.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A JAX/XLA upgrade changes fusion so a standalone step is no longer bit-identical to the scan body. | Parity is pinned by tests; the [parity policy](shared-contracts.md#parity-policy) says exactly when a row may be relaxed (round-off-sized, jaxprs equal) and requires a CHANGELOG note. |
| The decoder KDE path already differs at 1e-14 (reduction order), which could mask a real logic drift under a loose tolerance. | Tolerance is tight (`rtol=1e-11`) and the bilinear path stays bit-identical, so any drift in the shared `_decoder_step` is caught there. |
| Refactoring `_run_filter_scan` / `_position_decoder_filter_with_predictions` changes batch decoder behaviour. | Executor captures a baseline `.npz` on three configurations *before* the refactor and asserts bitwise equality after each extraction task; the existing decoder suite (including the monkeypatch / `clear_cache` test) must stay green; `_point_process_laplace_update` stays a module-global lookup. |
| Ring-buffer index errors that only show after wraparound. | The intermediate-lag tests compare against a restarted batch smoother at bins beyond two wraparounds of the ring. |
| Per-bin Python dispatch dominates latency; users expect sub-0.1 ms. | Measured table above, README sentence, benchmark script; explicit non-goal. |
| Mask semantics diverge from `masks-and-multi-sequence` if implemented locally. | Forwarding only; `NotImplementedError` until the keyword exists; semantics tests are `skipif`-guarded on the signature. |
| Delta-method phase interval is wrong at low amplitude (the true posterior is not von Mises-like). | Documented reliability threshold (amplitude / sd ≳ 10), half-width capped at π, and an approximation-trend test pinning how the error grows as SNR falls. |
| A per-instance closure as a `jax.jit` static argument recompiles per instance (tests build many). | Default paths use module-level callables (shared cache); only non-default configurations (custom intensity, robust weight, mask) compile per instance. Compile-once tests count traces. |
| Silent recompilation when a caller passes a differently shaped or typed observation. | `step` checks the shape against the construction-time shape and raises `ValueError`; inputs are cast to the filter dtype. |
| `FilterState.t` overflows int32 on a very long stream. | 2^31 bins ≈ 24.8 days at 1 kHz; documented; `reset()` restarts the count. |

## Rollout Strategy

Purely additive public API in a new module; nothing changes for existing
callers. The only edit to existing behaviour-bearing code is the Phase 3b
extraction inside `position_decoder.py`, which is verified bitwise on a
captured baseline and by the existing decoder tests, and is described in the
CHANGELOG as an internal refactor. Each phase is one PR and leaves the fast
suite green; Phase 1 already ships something useful (the Gaussian streaming
filter and theta phase), so later phases can slip without stranding users.
No feature flags, no deprecations.

## Open Questions

1. **Mask and robust-weight availability at execution time.** Decided:
   forward-only with `NotImplementedError` fallbacks and `skipif`-guarded
   tests; delete the fallbacks when both parallel plans have merged (trigger
   recorded in the contract). Keyword names (`obs_mask`, `robust_weight`),
   the boolean-only mask rule and the robust update's extra `RobustOutput`
   return element were confirmed against those plans' contracts at planning
   time; the wrappers slice the batch arity. Still open: surfacing the robust
   objective / per-bin weight from a streaming filter. Trigger: once
   wolf-robust-updates has merged, carry `RobustOutput.weight` through the
   compiled step as an aux value (like the point-process failure counter) and
   expose `.last_robust_weight`; until then the streaming filters with
   `robust_weight` set give robust moments and the marginal log-likelihood
   only.
2. **`.run()` as a Python loop rather than a `lax.scan`.** Decided: Python
   loop — it exercises the exact per-bin path the parity tests are about and
   keeps one implementation; offline users have the batch filters. Revisit
   only if someone needs a fast chunked streaming call, in which case add
   `run_chunk` built on `lax.scan` over the same step function.
3. **Should `FilterState` carry the one-step prediction / innovation?** Deferred:
   the contract fixes four fields; predictions flow through the compiled step's
   second return value into the lag buffer. If real-time innovation monitoring
   is requested, add a `predicted()` accessor on the filter object that reads
   the buffer (requires `lag ≥ 1`), not a new state field.
4. **KDE-path parity tolerance.** Decided: `rtol=1e-11` with the reduction-order
   explanation; if a future JAX makes it bit-identical, tighten the test.
5. **Citation for the real-time phase method.** Resolved (verified 2026-09-28
   against the eLife record): Wodeyar A, Schatza M, Widge AS, Eden UT, Kramer MA
   (2021), "A state space modeling approach to real-time phase estimation",
   eLife 10:e68803, https://doi.org/10.7554/eLife.68803. Cite exactly this.
6. **Per-instance compile cost for custom log-intensity functions.** Accepted:
   one compile per instance (hundreds of ms) is negligible for a real-time
   session; tests keep problems small.

## Estimated Effort

- Phase 1: `streaming.py` ≈ 350 LOC (state, base, Gaussian step, class, phase
  posteriors), tests ≈ 400 LOC, benchmark ≈ 120 LOC, docs ≈ 40 lines,
  `__init__.py` / `pyproject.toml` a few lines each.
- Phase 2: `streaming.py` +≈ 110 LOC, tests ≈ 220 LOC, docs ≈ 15 lines.
- Phase 3a: `streaming.py` +≈ 200 LOC, tests ≈ 280 LOC, benchmark +≈ 40 LOC,
  docs ≈ 20 lines.
- Phase 3b: `position_decoder.py` ≈ +150 / −90 LOC (pure movement plus six
  signatures), `streaming.py` +≈ 220 LOC, tests ≈ 300 LOC, benchmark +≈ 50
  LOC, docs ≈ 25 lines.
