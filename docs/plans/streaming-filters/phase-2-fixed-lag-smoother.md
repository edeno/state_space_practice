# Phase 2 — Fixed-lag smoothing for the streaming Gaussian filter

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d4-fixed-lag-smoother-ring-buffer)

Adds the ring buffer of the last `lag + 1` filtered moments and one-step
predictions, and `.smoothed(lag)`, which re-runs the RTS backward pass over
that window. The fixed-lag estimate at `t − L` converges to the batch
smoother's as `L` grows; this phase pins that trend.

**Inputs to read first:**

- `src/state_space_practice/kalman.py:845-912` — `rts_backward_scan_with_predictions`, the backward pass run over the window; note the alignment of `predicted_mean[1:]` / `predicted_cov[1:]` (lines 873-877, 900-905).
- `src/state_space_practice/kalman.py:778-842` — `rts_backward_scan` (recomputes the prediction; *not* used, see D4 for why) and `:714-775` `_kalman_smoother_update` (the recursion being reproduced).
- `src/state_space_practice/kalman.py:944-1044` — `kalman_smoother`, the batch reference for the full-window test.
- `src/state_space_practice/streaming.py` (Phase 1) — `_StreamingFilter`, `_kalman_streaming_step`; the `buf` argument threaded through in Phase 1.
- `src/state_space_practice/tests/test_position_decoder.py:1922-1989` — `TestPredictionAwareSmoother`: how a hand-rolled backward pass on stored predictions is checked against the library smoother (pattern to mirror).
- `src/state_space_practice/tests/test_likelihood_identities.py:216-220` — smoother equals filter at the last step (the `lag == 0` case).
- `src/state_space_practice/tests/test_approximation_trends.py:63-65` — `_assert_decreasing`.

**Contracts referenced:**

- [Streaming filter surface](shared-contracts.md#streaming-filter-surface) — `.smoothed(lag)` and `.lag` semantics (warm-up, bounds, `lag == 0`).
- [FilterState](shared-contracts.md#filterstate) — fields of the returned smoothed state (`t = state.t − lag`, filter LL unchanged).
- [Prediction hand-off](shared-contracts.md#prediction-hand-off-for-fixed-lag-smoothing) — what goes into `predicted_*`.
- [Parity policy](shared-contracts.md#parity-policy) — full-window row: `rtol=1e-12`.

**Designs referenced:** [D4](designs.md#d4-fixed-lag-smoother-ring-buffer).

## Tasks

- **Add the ring buffer** to `streaming.py`: `_LagBuffer`, `_empty_lag_buffer`,
  `_push_lag_buffer`, `_smooth_lag_window` exactly as in D4. In
  `_StreamingFilter.reset`, allocate the buffer when `self.lag > 0`; remove
  the Phase 1 `NotImplementedError` for `lag > 0`.
- **Push from the step**: add the `if buf is not None: buf = _push_lag_buffer(...)`
  branch to `_kalman_streaming_step` (D2 final form). `StreamingKalmanFilter`
  already passes `self._buffer` in and stores the returned buffer.
- **Add `_StreamingFilter.smoothed`** (D4) and a `_transition_matrix`
  attribute set by `StreamingKalmanFilter.__init__` (`A` after dtype
  promotion).
- **Tests** in `tests/test_streaming.py` (validation slice below).
- **User-facing docs:**
  - `CHANGELOG.md` `### Added`: extend the streaming entry with `lag=` /
    `.smoothed(lag)`, the O(L)-per-call cost, and the approximation statement
    ("the lag-L estimate equals the batch smoother only as L grows; the gap
    decays geometrically with the smoother gain — see the test for numbers").
  - `README.md` streaming subsection: one sentence plus a two-line snippet
    (`StreamingKalmanFilter(..., lag=32)`; `f.smoothed()` after warm-up) and
    the warm-up rule.
  - `StreamingKalmanFilter` and `smoothed` docstrings: the buffer semantics,
    the warm-up `ValueError`, and that `smoothed()` returns the estimate for
    bin `t − lag` with `t` set accordingly.

## Deliberately not in this phase

- Smoothing for the point-process filter and the decoder — they reuse this
  machinery unchanged in Phases 3a/3b (they only have to push their
  predictions; the decoder pushes the inflated dynamics prediction).
- Returning the whole smoothed window (`smoothed_window()`) or lag-one
  cross-covariances — not needed for real-time use; add when a caller
  appears.
- A dedicated O(1)-per-step fixed-lag recursion (augmented-state form) — the
  window re-run is O(L) per call and L ≤ 64 in the target applications; revisit
  if a benchmark shows `smoothed()` dominating a per-bin budget.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_smoothed_lag_zero_is_filtered_state` | `smoothed(0) is f.state` and `smoothed()` with `lag=0` at construction returns the filtered state bitwise. |
| `test_smoothed_full_window_matches_batch_smoother` | `lag = T − 1` (T = 40 bins), after `run(obs)`: `smoothed(T − 1)` mean/cov equal `kalman_smoother(...)[0][0]`, `[1][0]` to `rtol=1e-12` (`atol=1e-12·scale`); `.t == 1`; `.log_likelihood == state.log_likelihood`. Guard: differs from `f.state.mean` (smoothing changed the estimate). |
| `test_smoothed_intermediate_lags_match_restarted_batch_smoother[lag ∈ {1, 3, 10}]` | Filter built with `lag=10`; after `t` steps (0-based bins `0..t−1` absorbed) for `t ∈ {12, 25, 37, 40}` — past two wraparounds of the 11-slot ring: `smoothed(lag)` is the estimate for bin `s = t − 1 − lag` given bins `0..t−1`, so it must equal the first element of `kalman_smoother(f_mean[s−1], f_cov[s−1], obs[s:t], A, Q, H, R)` (restart from the filtered moments one bin before the window; for `s == 0` restart from `init_mean`, `init_cov`) — `rtol=1e-12`. Also `smoothed(lag).t == t − lag`. |
| `test_fixed_lag_gap_shrinks_with_lag` (`@pytest.mark.slow`, ~6 window-size compiles) | Oscillator problem (D4 table setup, 600 bins), `lag=32` filter; for `L ∈ {1, 2, 4, 8, 16, 32}` collect `smoothed(L)` at every `t ≥ 133` and compare with `kalman_smoother` at bin `t − L`: RMS mean gap over bins strictly decreasing in `L` and `gap(1) > 0.5 · RMS(filtered − smoothed)`, `gap(32) < 0.25 · gap(1)`; `trace(P_{s given s+L}) − trace(P_{s given T})` strictly decreasing in `L` for every checked bin and `> 0` at `L = 1` (expected values in the D4 table). |
| `test_smoothed_before_warm_up_raises` | `lag=5`, 3 steps: `smoothed()` raises `ValueError(match="needs 6 steps")`; after 3 more steps it returns. |
| `test_smoothed_lag_above_buffer_raises` | `lag=5`: `smoothed(6)` raises `ValueError(match="exceeds")`; `smoothed(5)` works after warm-up. Constructor `lag=-1` / `lag=2.0` raise `ValueError` (via `validate_int`). |
| `test_reset_clears_buffer` | After warm-up, `smoothed()` works; `reset()` then `smoothed()` raises the warm-up `ValueError`; `reset(state)` with a caller state also empties the buffer (`_n_since_reset == 0`). |
| `test_step_compiles_once_with_lag` | As Phase 1's compile-once test but with `lag=8`: one trace of the step; `smoothed(8)` then `smoothed(3)` add exactly two `_smooth_lag_window` compilations (count with a monkeypatched `streaming.rts_backward_scan_with_predictions`); repeated `smoothed(8)` adds none. |
| `test_run_parity_unaffected_by_lag` | `StreamingKalmanFilter(..., lag=16).run(obs)` is bitwise identical to `kalman_filter` (the buffer write does not touch the filtered moments). |

## Fixtures

- Reuse Phase 1's `lgssm_cases` and `oscillator_problem` fixtures
  (`test_streaming.py`, module scope). The trend test extends
  `oscillator_problem` to 600 bins with `R = 4`; if the Phase 1 fixture is
  2000 bins, slice it rather than simulate again (the gap statistics only need
  ~400 evaluation bins).
- Batch references: `kalman_filter` / `kalman_smoother` on the same arrays,
  computed once per fixture.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
