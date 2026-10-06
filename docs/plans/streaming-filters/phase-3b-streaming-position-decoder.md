# Phase 3b — Decoder step extraction and `StreamingPositionDecoder`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d6-position-decoder-step-extraction-and-streamingpositiondecoder)

Real-time position decoding from population spikes (the decoding lineage of
Denovellis, Gillespie, Coulter, Sosa, Chung, Eden & Frank 2021, eLife
10:e64505; the application class is closed-loop SWR disruption, Jadhav,
Kemere, German & Frank 2012, *Science* 336:1454-1458). Two parts, in this
order: a behaviour-preserving extraction of the decoder's per-bin step out of
`position_decoder._run_filter_scan`, then `StreamingPositionDecoder` built on
it with the fixed-lag smoother.

**Inputs to read first:**

- `src/state_space_practice/position_decoder.py:958-1178` — `_run_filter_scan`: static args 958-968, closures 1022-1082, step body 1084-1153 (prediction 1087-1094, Woodbury track penalty 1096-1107, inflation 1109-1131, Laplace update 1133-1144, carry bookkeeping 1146-1153), scan + warning 1155-1178. This is what gets extracted.
- `src/state_space_practice/position_decoder.py:1260-1536` — `_position_decoder_filter_with_predictions`: `q_pos` default 1297-1304, dynamics 1306, init resolution 1309-1370, rate-map arrays 1372-1378, track penalty 1380-1391, KDE args 1393-1416, inflation args 1418-1428, `init_cov` validation 1430-1439, scan call 1441-1470, host-side warnings 1472-1529.
- `src/state_space_practice/position_decoder.py:1181-1257` (`position_decoder_filter`), `:1539-1612` (`position_decoder_smoother`, uses the stored predictions at 1599-1606), `:1615-1841` (`PositionDecoder`; `decode` at 1774-1841 raises `RuntimeError` when unfitted, 1798-1803).
- `src/state_space_practice/position_decoder.py:58` (`_DECODER_DIAGONAL_BOOST`), `:77-133` (`AdaptiveInflationConfig`), `:136-195` (`build_position_dynamics`), `:239-359` (`PlaceFieldRateMaps.__init__`: the `_jax_*`, `_dx`, `_dy`, `_use_analytical`, `n_neurons`, `suggested_q_pos` attributes the streaming class reads), `:870-914` (`DecoderResult`).
- `src/state_space_practice/kalman.py:845-912` — `rts_backward_scan_with_predictions` and its docstring on modified predictions (853-867).
- `src/state_space_practice/tests/test_position_decoder.py:531-574` (`decoding_data`, bilinear rate maps), `:618-663` (single-bin parity with `_point_process_laplace_update`), `:1882-1919` (`inflation_problem`, KDE + inflation + zero penalty), `:1922-2018` (`TestPredictionAwareSmoother`, including the active-penalty strip at 2001-2004), `:2021-2053` (`TestTracedDecoderHyperparameters`: monkeypatches `position_decoder._point_process_laplace_update` and calls `_run_filter_scan.clear_cache()` — constraints on the refactor).
- `src/state_space_practice/exceptions.py` — `NotFittedError`.
- `src/state_space_practice/streaming.py` (Phases 1-3a).

**Contracts referenced:**

- [FilterState](shared-contracts.md#filterstate), [Streaming filter surface](shared-contracts.md#streaming-filter-surface) — including `.n_line_search_failures`, `.n_inflation_capped_bins`, `.run() -> DecoderResult`.
- [Parity policy](shared-contracts.md#parity-policy) — decoder rows: bit-identical on the bilinear path, `rtol=1e-11` on the KDE path.
- [Optional cross-plan keyword forwarding](shared-contracts.md#optional-cross-plan-keyword-forwarding) — forwarded through `_decoder_step` to `_point_process_laplace_update`.
- [Prediction hand-off](shared-contracts.md#prediction-hand-off-for-fixed-lag-smoothing) — push the inflated *dynamics* prediction, never the penalised one.

**Designs referenced:** [D6](designs.md#d6-position-decoder-step-extraction-and-streamingpositiondecoder), [D4](designs.md#d4-fixed-lag-smoother-ring-buffer), [D7](designs.md#d7-latency-benchmark-script).

## Tasks

- **Capture a baseline before touching `position_decoder.py`** (executor-only,
  not committed): run `_position_decoder_filter_with_predictions` on (a) the
  `inflation_problem` configuration (test_position_decoder.py:1887-1912: KDE
  path, inflation `gain=1, max_alpha=5`, zero track penalty), (b) the same
  rate maps with the off-track strip of lines 2001-2004 and no inflation
  (active penalty), (c) the `decoding_data` configuration (bilinear rate maps,
  default init, `include_velocity=True`). Save `position_mean`,
  `position_cov`, `marginal_log_likelihood`, `predicted_mean`,
  `predicted_cov` to an `.npz` in the scratch directory. Also record
  `uv run pytest src/state_space_practice/tests/test_position_decoder.py -q`
  timing.
- **Extract the step body and closures in `position_decoder.py`** exactly as
  D6 specifies: new module-level `_build_decoder_functions` and
  `_decoder_step` (bodies moved verbatim, `_point_process_laplace_update`
  looked up as a module global at call time), and `_run_filter_scan` rewired
  to call them. Keep `_run_filter_scan`'s name, `jax.jit` decoration and
  static argument names unchanged. Then re-run the three baseline
  configurations and assert **bitwise** equality with the saved arrays
  (`np.testing.assert_array_equal`); run `test_position_decoder.py` — all
  green, including `TestTracedDecoderHyperparameters`.
- **Extract the setup helpers** `_resolve_decoder_init`,
  `_resolve_track_penalty`, `_kde_args`, `_inflation_args` from
  `_position_decoder_filter_with_predictions` (D6) and call them from it;
  error messages unchanged. Re-run the baseline comparison (bitwise) and the
  decoder tests.
- **Add `_decoder_streaming_step` and `StreamingPositionDecoder`** to
  `streaming.py` per D6, including `from_decoder` (raises `NotFittedError`
  when `decoder.rate_maps is None`), `position_xy`, the two counters,
  `.run() -> DecoderResult`, and the host-side `NotImplementedError` checks
  for `mask` / `robust_weight` against `_point_process_laplace_update`'s
  signature. Push the inflated dynamics prediction into the lag buffer.
  Docstrings: NumPy style; state the KDE-path parity tolerance and its
  reason, that `.run()` does not re-emit the batch entry point's
  whole-recording warnings, and the measured per-step cost of the KDE path
  (see Metrics) so users pick `PlaceFieldRateMaps(rate_maps=...)`
  (bilinear) when the bin budget is tight.
- **Export** `StreamingPositionDecoder` lazily; `uv run mypy` stays green
  (`position_decoder.py` is not in the mypy file list; `follow_imports =
  silent` means only `streaming.py`'s own annotations are checked — annotate
  the imported helpers' results explicitly where mypy infers `Any`).
- **Tests**: a new `TestStreamingPositionDecoder` class in
  `tests/test_position_decoder.py` (it needs the module-scoped
  `inflation_problem` fixture and the decoder test helpers, which cannot be
  imported into another module as fixtures).
- **Benchmark**: add the decoder section (KDE and bilinear paths,
  `include_velocity ∈ {False, True}`, `n_neurons ∈ {1, 16, 100}`) to
  `scripts/benchmark_streaming_latency.py`; paste the table into the PR
  description against [overview.md — Metrics](overview.md#metrics).
- **User-facing docs:** `CHANGELOG.md` — under `### Added` extend the streaming
  entry with `StreamingPositionDecoder` / `from_decoder` (parity statement
  with both tolerances, `lag=` reuse, counters); under `### Changed` (or the
  internal-refactor subsection the CHANGELOG uses) one line that the decoder's
  per-bin step is now `position_decoder._decoder_step`, shared by the batch
  filter and the streaming decoder, behaviour unchanged (bitwise). `README.md`
  streaming subsection: a decoder example (`decoder.fit(...)`;
  `StreamingPositionDecoder.from_decoder(decoder, lag=25)`; per-bin
  `step(counts)`; `smoothed().mean[:2]`) and the measured KDE-path cost with
  the bilinear alternative.

## Deliberately not in this phase

- Changing decoder behaviour in any way (defaults, warnings, penalty, inflation
  formula). The extraction is bitwise-preserving by construction and by the
  baseline check; anything the executor notices while in there (e.g. the
  `RuntimeError` at position_decoder.py:1799 that CLAUDE.md would want to be a
  `NotFittedError`) is reported in the PR, not fixed.
- Re-emitting the batch entry point's inflation-cap-fraction and
  left-the-arena warnings from `.run()` (counters are exposed instead).
- Graph / linearised-track decoding (`graph_place_field.py`) — a different
  observation model.
- Streaming clusterless (mark-based) decoding — no batch counterpart exists in
  this library yet.

## Validation slice

| Test | Asserts |
| --- | --- |
| (executor check, not committed) baseline bitwise after each refactor task | `assert_array_equal` on all five saved arrays for configurations (a)–(c); LL `==`. |
| existing `tests/test_position_decoder.py` (whole file) | Green after the extraction, in particular `test_single_bin_filter_matches_core_laplace_update`, `TestPredictionAwareSmoother`, `TestTracedDecoderHyperparameters::test_sigma_track_sweep_compiles_once`. |
| `test_streaming_run_matches_filter_bitwise_bilinear[inflate ∈ {None, AdaptiveInflationConfig(gain=1.0, max_alpha=5.0)}]` | `PlaceFieldRateMaps(rate_maps=...)` (bilinear) with an occupancy-mask strip so the penalty is active, `include_velocity=False`, fixed `init_position` / `init_cov`: `StreamingPositionDecoder(...).run(spikes)` equals `position_decoder_filter(...)` in `position_mean`, `position_cov` (`assert_array_equal`) and `marginal_log_likelihood` (`==`); the pushed predictions (read back from `_buffer` with `lag = T − 1`) equal `_position_decoder_filter_with_predictions`' `predicted_mean` / `predicted_cov` bitwise. Guard: the penalty value at some predicted mean is `> 0` (penalty exercised; evaluate `_build_track_penalty`'s map at the predicted `xy`) and, in the inflated case, `max over t of pred_cov[t] / (A f_cov[t−1] A^T + Q) > 1.05` (inflation exercised). |
| `test_streaming_run_matches_filter_kde` | `inflation_problem["kwargs"]` (KDE path): `assert_allclose(rtol=1e-11, atol=1e-11 * max\|value\|)` on means, covs and the pushed predictions; LL `abs diff < 1e-10`. Do not assert that the outputs differ (a future XLA may make them bitwise equal); guard instead that inflation was active (`max over t of pred_cov[t] / (A f_cov[t−1] A^T + Q) > 1.05`, as test_position_decoder.py:1964 guards). |
| `test_default_init_matches_batch` | No `init_position` / `init_cov`, `include_velocity=True`, `decoding_data` rate maps: the first `step` equals `position_decoder_filter(spikes[:1], ...)` bitwise (the arena-centre / loose-prior defaults are the same helper). |
| `test_from_decoder_matches_decode_filter` | `PositionDecoder(dt=...)` fitted on `inflation_problem`-style data (`fit` → slow-marked automatically by conftest): `StreamingPositionDecoder.from_decoder(decoder, init_position=p0).run(spikes)` equals `decoder.decode(spikes, method="filter", init_position=p0)` to the KDE tolerance. `from_decoder` on an unfitted decoder raises `NotFittedError`. |
| `test_fixed_lag_full_window_matches_position_decoder_smoother` | `inflation_problem` kwargs, `lag = T − 1`: `smoothed(T − 1)` equals `position_decoder_smoother(**kwargs).position_mean[0]` / `.position_cov[0]` to `rtol=1e-10`. Guard: the naive `rts_backward_scan` on the filtered moments differs by `> 1e-3` (as at test_position_decoder.py:1986-1989), proving the inflated predictions were used. |
| `test_fixed_lag_intermediate_lag_matches_restarted_smoother` | `lag=10`, at `t ∈ {11, 27, 40}`: `smoothed(10)` vs `position_decoder_smoother(spikes[s:t], init_position=f_mean[s−1], init_cov=f_cov[s−1], track_penalty=zeros, adaptive_inflation=cfg, ...)`'s first element, `rtol=1e-9` (window start `s = t − 11`). |
| `test_smoothed_is_closer_to_truth_than_filtered` (behavioural) | `decoding_data`-style simulation (500 bins, 2 → use 6 fields as in test_position_decoder.py:816-870 for coverage), `lag=25`: over bins where both exist, median `\|smoothed_xy − true\|` `<` median `\|filtered_xy − true\|` (filtered taken at the same bin from the recorded states). Guard: both medians finite and `< 30 cm`. |
| `test_counts_shape_mismatch_raises` | `step(jnp.zeros(n_neurons + 1))` → `ValueError`; state unchanged. `run` with the wrong column count → `ValueError(match="neurons")`. |
| `test_step_compiles_once` | Monkeypatch `position_decoder._point_process_laplace_update` (as test_position_decoder.py:2026-2032), `_decoder_streaming_step.clear_cache()`, 30 steps → one trace. |
| `test_counters_and_run_warning` (wiring) | Monkeypatch `_point_process_laplace_update` to force the failure flag to 1: `n_line_search_failures == n_steps` and `.run` logs the line-search warning (`caplog`); with `AdaptiveInflationConfig(gain=1e4, max_alpha=1.5)` on `inflation_problem` data, `n_inflation_capped_bins > 0` after `run` and equals the batch's capped count (expose it from `_run_filter_scan`'s return, index 5). |

Slow marking: `test_from_decoder_matches_decode_filter` calls `.fit(` and is
auto-marked slow by `conftest.py`; the others compile one step each and run in
a few seconds total. Mark `test_smoothed_is_closer_to_truth_than_filtered`
slow if it exceeds ~3 s (500 Python-loop steps plus 500 `smoothed()` calls).

## Fixtures

- Existing module-scoped `inflation_problem` (test_position_decoder.py:1882-1919)
  for the KDE-path parity, smoother and counter tests.
- Existing `TestPositionDecoderFilter.decoding_data` recipe: lift it to a
  module-scoped `bilinear_decoding_problem` fixture (rate maps, spikes, true
  position, `dt`) with an added occupancy-mask strip variant for the
  active-penalty parity case; keep the class-level fixture delegating to it so
  the existing tests are untouched.
- Baseline `.npz` (scratch directory, executor-only) for the extraction check.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
