# Phase 3a — `StreamingPointProcessFilter`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d5-point-process-step-and-streamingpointprocessfilter)

A per-bin Laplace-EKF filter for spike counts (Eden, Frank, Barbieri, Solo &
Brown 2004, *Neural Computation* 16:971-998 — the point-process adaptive
filter whose Gaussian-approximation update `glm_laplace_update` implements),
bit-identical to `stochastic_point_process_filter`'s dense path, with the
fixed-lag smoother from Phase 2.

**Inputs to read first:**

- `src/state_space_practice/point_process_kalman.py:1333-1464` — `glm_laplace_update`, the public single-step update the streaming step calls (`return_line_search_failures=True` returns the 4-tuple).
- `src/state_space_practice/point_process_kalman.py:1215-1239` (`GLMFamily`), `:1242-1265` (`poisson_family(dt, max_log_count)`), `:1292-1297` (`BERNOULLI_LOGIT_FAMILY`), `:523-547` (`log_conditional_intensity`, the default `(design_t, x) -> log rate`).
- `src/state_space_practice/point_process_kalman.py:1737-1866` — `_stochastic_point_process_filter_impl`: dtype promotion 1780-1791, Jacobian precompute 1794-1798, scan body 1800-1846 (the covariance symmetrisation at 1809, the `n_failed_bins` carry at 1805/1836).
- `src/state_space_practice/point_process_kalman.py:1467-1734` — `stochastic_point_process_filter`: validation 1635-1646, single-neuron promotion 1707-1710, dense path 1714-1734 (what `.run()` mirrors); `:882-924` `_warn_line_search_failures`; `:96-145` `_validate_public_inputs`; `:512-520` `_common_float_dtype`.
- `src/state_space_practice/point_process_kalman.py:2392-2411`, `:2571-2603` — `stochastic_point_process_smoother` (signature, returns `(smoother_mean, smoother_cov, cross_cov, ll)`), the batch reference for the fixed-lag full-window test.
- `src/state_space_practice/tests/test_glm_laplace.py:36-68` — the Poisson parity precedent.
- `src/state_space_practice/tests/test_likelihood_identities.py:579-660` — `_affine_log_rate`, `_pp_problem`, `_laplace_terms`; `:734-760` chain rule by restart for the point-process filter.
- `src/state_space_practice/streaming.py` (Phases 1-2) — `_StreamingFilter`, `_accepts_kwarg`, `_push_lag_buffer`, `smoothed`.

**Contracts referenced:**

- [FilterState](shared-contracts.md#filterstate), [Streaming filter surface](shared-contracts.md#streaming-filter-surface) — including `.n_line_search_failures` and the `.run()` warning.
- [Parity policy](shared-contracts.md#parity-policy) — point-process row: bit-identical (dense path).
- [Optional cross-plan keyword forwarding](shared-contracts.md#optional-cross-plan-keyword-forwarding) — `robust_weight` / `mask` → `glm_laplace_update`.
- [Prediction hand-off](shared-contracts.md#prediction-hand-off-for-fixed-lag-smoothing) — push `(A m, symmetrize(A P A^T + Q))`.

**Designs referenced:** [D5](designs.md#d5-point-process-step-and-streamingpointprocessfilter), [D1](designs.md#d1-module-layout-and-compiled-step-pattern), [D7](designs.md#d7-latency-benchmark-script).

## Tasks

- **Add `_make_glm_update`, `_point_process_streaming_step` and
  `StreamingPointProcessFilter`** to `streaming.py` per D5. Constructor
  validation per D1 (`validate_covariance` on `init_cov` / `process_cov`,
  finite `A` / `init_mean`, shapes), dtype via `_common_float_dtype` over
  `(A, Q, init_mean, init_cov)`; `family=None` requires `dt` and builds
  `poisson_family(dt, max_log_count)`. `_reset_counters` zeroes
  `_n_failed_bins`. `.step(counts, design_t, mask=None)` fixes the count shape
  on first use; `.run(design_matrix, spikes)` mirrors the batch dense path
  (count validation, 1-D promotion, end-of-run line-search warning).
  Docstrings: NumPy style with shapes; state that the block-diagonal path
  (`block_n_neurons` / `block_size`) is not offered and why (per-neuron
  factorisation is an offline memory optimisation; the dense step is what a
  real-time loop needs); note that a custom `log_intensity_func` is traced
  once per instance and must be a stable callable.
- **Export** `StreamingPointProcessFilter` lazily (`_LAZY_API`, `__all__`,
  `TYPE_CHECKING`), keep `uv run mypy` green (`streaming.py` is already in the
  files list from Phase 1; `GLMFamily` and `glm_laplace_update` are typed in
  `point_process_kalman.py`, which mypy follows silently).
- **Tests** in `tests/test_streaming.py` (new `TestStreamingPointProcessFilter`
  class) and a `TestStreamingPointProcessIdentities` class in
  `tests/test_likelihood_identities.py` reusing `_pp_problem` /
  `_affine_log_rate`.
- **Benchmark**: add the point-process section to
  `scripts/benchmark_streaming_latency.py` (D7) and paste the measured table
  into the PR description against [overview.md — Metrics](overview.md#metrics).
- **User-facing docs:** `CHANGELOG.md` `### Added` (extend the streaming
  entry: `StreamingPointProcessFilter`, bit-identical to the dense
  `stochastic_point_process_filter`, `family=` for Bernoulli, `lag=`);
  `README.md` streaming subsection: a three-line spike-count example
  (`StreamingPointProcessFilter(A, Q, m0, P0, dt=0.001)`, `step(counts,
  design_t)`), and the measured per-step latency (≈50–100 µs) with the
  batching advice.

## Deliberately not in this phase

- `StreamingPositionDecoder` and the `position_decoder.py` extraction —
  Phase 3b (the decoder's step is not `glm_laplace_update`; it carries the
  track penalty and inflation).
- The block-diagonal filter path and `BlockDiagonalCovariance` outputs.
- Bernoulli-specific conveniences beyond passing `family=BERNOULLI_LOGIT_FAMILY`.
- Any masking / robust-weight semantics of our own (forwarding only).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_run_matches_stochastic_point_process_filter_bitwise[max_newton_iter ∈ {1, 3}, normalize ∈ {True, False}]` | Default `log_conditional_intensity`, `_pp_problem`-style data with a `(T, n_neurons, n_latent)` design (no baseline column): `.run(design, y)` means/covs `assert_array_equal` to `stochastic_point_process_filter(...)`; `float(ll) == float(batch_ll)`. Guard: `y.sum() > 20`. |
| `test_single_neuron_spikes_are_promoted_like_batch` | 1-D `spikes (T,)` with a `(T, n_latent)` design: `.run` equals the batch bitwise (batch promotes at point_process_kalman.py:1707-1710). |
| `test_custom_log_intensity_matches_batch_bitwise` | `_affine_log_rate` with the `_pp_problem(0)` design: bitwise equality with `stochastic_point_process_filter(..., _affine_log_rate)`. |
| `test_bernoulli_family_step_matches_glm_laplace_update` | One `step` with `family=BERNOULLI_LOGIT_FAMILY` equals a direct `glm_laplace_update(A m0, symmetrize(A P0 A^T + Q), y, eta, BERNOULLI_LOGIT_FAMILY, grad_eta_func=...)` bitwise; `n_line_search_failures` equals the direct call's failure flag. |
| `TestStreamingPointProcessIdentities::test_summed_ll_equals_batch[max_newton_iter ∈ {1, 3}]` | Streaming cumulative LL `==` batch LL exactly on `_pp_problem(0)`; and equals `sum(_laplace_terms(...))` to `rtol=1e-10` (reusing test_likelihood_identities.py:615-652's loop on the streaming outputs). |
| `TestStreamingPointProcessIdentities::test_chain_rule_by_reset[split]` | Mirror of `:734-760` with `reset(FilterState(f_mean[split-1], f_cov[split-1], split, 0.0))`: head + tail LL `== f_ll` to `rtol=1e-10`, tail means `rtol=1e-12`. |
| `test_requires_dt_or_family` | No `dt`, no `family` → `ValueError(match="dt")`; `dt=-0.1` → `ValueError` (from `poisson_family`'s `validate_scalar`). |
| `test_counts_shape_fixed_by_first_step` | Second `step` with a different count length raises `ValueError`; the state is unchanged. |
| `test_step_compiles_once` | Monkeypatch `streaming.glm_laplace_update` with a counter, `_point_process_streaming_step.clear_cache()`, 30 steps → one trace; a second instance with the same shapes and the default intensity adds none. Guard: ≥ 1 trace. |
| `test_line_search_failure_counter_and_run_warning` (wiring test) | Monkeypatch `streaming.glm_laplace_update` to return the real result with the failure flag forced to `jnp.int32(1)`, clear the step cache: after 10 steps `n_line_search_failures == 10`; `.run(design, y)` (T = 20, `max_newton_iter=3`) logs the `_log_line_search_failures` warning (`caplog`, logger `state_space_practice.point_process_kalman`, `match="rejected all"`); `reset()` zeroes the counter. With `max_newton_iter=1` and the same patch, `.run` logs nothing. |
| `test_mask_and_robust_weight_forwarding` | `skipif`-paired pattern from Phase 1 against `glm_laplace_update`'s signature: with support, a mask dropping neurons `{0, 2}` matches a filter on the reduced design / counts to `rtol=1e-11`; without, `NotImplementedError` naming the owning plan. |
| `test_fixed_lag_full_window_matches_point_process_smoother` | `lag = T − 1` on `_pp_problem(1)`: `smoothed(T − 1)` equals `stochastic_point_process_smoother(...)[0][0]`, `[1][0]` to `rtol=1e-10`. Guard: differs from the filtered mean at bin 0. |
| `test_streaming_posterior_beats_prior` (behavioural) | On `_pp_problem(2, n_time=200)`: RMSE of streamed `state.mean` (collected per step) to the true latent `< 0.7 ×` RMSE of the prior mean `m0` broadcast; median posterior variance `< 0.5 × trace(P0)/n_latent`. |

Slow marking: none of these fit; each compiles one small step. The parity
test parametrised 4 ways compiles 4 variants (~1 s each) — keep the problem
at `n_time = 25` (the `_pp_problem` default).

## Fixtures

- `pp_cases` (module scope): `_pp_problem(seed)` for seeds `{0, 1}` and a
  design without the baseline column for the default-intensity parity case,
  plus the batch filter outputs.
- The true latent trajectory for the behavioural test: extend the local
  problem builder to return `x` (the `_pp_problem` helper does not; simulate
  in `test_streaming.py` with the same recipe and seed 2).
- Bernoulli data: `default_rng(3)` 0/1 indicators from a logistic model with
  the same design shape.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
