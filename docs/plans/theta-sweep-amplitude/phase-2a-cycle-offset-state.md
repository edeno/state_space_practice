# Phase 2a — Per-theta-cycle random offset state with a reset at cycle boundaries

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#time-varying-dynamics)

Adds the cycle-to-cycle "sampling" component: a scalar offset `epsilon` shared by all bins of one theta cycle, drawn afresh (`N(0, sigma_c^2)`) at every cycle start, so `r_t = p_t + b_t . x + d_t epsilon_t`. The reset needs a transition matrix and process covariance that change per bin, which `stochastic_point_process_filter` does not support (one `A`, one `Q`), so this phase introduces a forward scan over per-bin `(A_t, Q_t)` built from `_point_process_laplace_update` and a backward pass through `parallel_kalman_smoother`, re-routes the constant-dynamics path of phase 1 through the same scan (one engine), and pins that path to the library smoother with an equivalence test. `sigma_c^2` is a user-set hyperparameter here; phase 2b fits it.

**Inputs to read first:**

- `src/state_space_practice/theta_sweep.py` (phase 1) — `theta_sweep_smoother`, `build_sweep_design`, `ThetaSweepModel._fit_bouts` / `_assemble_result`, `ThetaSweepResult`.
- `src/state_space_practice/point_process_kalman.py:1745-1866` — `_stochastic_point_process_filter_impl`: the scan body to reproduce with stacked dynamics (dtype promotion `:1780-1791`, `jacfwd` `:1794-1798`, the `_step` `:1800-1846`, the carry `:1848-1864`).
- `src/state_space_practice/point_process_kalman.py:927-940` and `:1210-1212` — `_point_process_laplace_update` signature and `return_line_search_failures`; `:882-924` `_warn_line_search_failures` (logging callback, jit-safe).
- `src/state_space_practice/position_decoder.py:958-1178` — `_run_filter_scan`: an existing jitted scan that calls `_point_process_laplace_update` directly with static `max_newton_iter` and traced `dt` (`:958-967`, `:1133-1144`, `:1165-1170`).
- `src/state_space_practice/kalman.py:1072-1216` — `parallel_kalman_smoother`; time-varying stacks `(T-1, D, D)` with `transition_matrix[t]` mapping `t -> t+1` (`:1090-1096`, `:1130-1150`), finite-input checks when concrete (`:1152-1163`); `tests/test_kalman.py:1885` tests it against a sequential time-varying reference.
- `src/state_space_practice/point_process_kalman.py:2605-2641` — the sequential backward pass the library smoother uses (what the equivalence test compares against, via `stochastic_point_process_smoother`).
- `src/state_space_practice/utils.py:910-917` `contains_tracer`, `:1187-1305` `_validate_filter_numerics`, `:26` `symmetrize`.
- `src/state_space_practice/tests/test_calibration_point_process.py` — the `TestThetaSweepSmootherCalibration` class added in phase 1 (extend it).

**Contracts referenced:**

- [Design-matrix layout](shared-contracts.md#design-matrix-layout) — the offset column `d_t` is appended at index `1 + K`; coefficient block stays `[:K]`.
- [theta_sweep_smoother](shared-contracts.md#theta_sweep_smoother) — adds `cycle_starts` and `cycle_offset_variance`; outputs unchanged in meaning (`D = K + 1` with the offset).
- [ThetaSweepResult](shared-contracts.md#thetasweepresult) — fills `cycle_index`, `cycle_offset_mean`, `cycle_offset_var`; the `NaN`-outside-bouts invariant extends to them (`cycle_index = -1`).
- [ThetaSweepModel](shared-contracts.md#thetasweepmodel) — adds `cycle_offset` and `cycle_offset_variance`; reuses phase 1's `min_cycle_duration`.

**Designs referenced:** [cycle boundaries](designs.md#cycle-boundaries), [time-varying dynamics](designs.md#time-varying-dynamics).

## Tasks

- **Cycle boundaries.** Reuse phase 1's `theta_cycle_index` and `cycle_starts_from_index` from `theta_sweep.py` as in [designs](designs.md#cycle-boundaries) (NumPy; `ValueError` on non-finite or non-advancing phase). `ThetaSweepModel` computes `min_cycle_bins = max(1, round(min_cycle_duration / dt))`. In `_fit_bouts`, when `cycle_offset` is on, compute per bout `cycle_index_b = theta_cycle_index(cycle_phase[start:end], min_cycle_bins)` and `cycle_starts_b = cycle_starts_from_index(cycle_index_b)`; the result's `cycle_index` is `cycle_index_b + offset_b` with `offset_b` the running total of cycles in earlier bouts, `-1` outside bouts.
- **Time-varying engine.** Add `sweep_transition_stacks` and `_sweep_filter_scan` exactly as in [designs](designs.md#time-varying-dynamics). Re-implement the body of `theta_sweep_smoother`: build `init_mean` (zeros, `D = K + 1` when `cycle_starts` is given) and `init_cov = blockdiag(init_std^2 I_K, cycle_offset_variance)`; when inputs are concrete, validate as before plus `validate_scalar(cycle_offset_variance, positive=True)`, `cycle_starts` boolean of length `n_time` with `cycle_starts[0] == True`, and `_validate_filter_numerics(init_cov, n_time, filter_name="theta_sweep_smoother")`; build the stacks; run `_sweep_filter_scan(..., dt=dt, max_log_count=float(np.log(max_rate_hz * dt)), log_intensity=rate_maps.log_intensity, max_newton_iter=max_newton_iter)`; finish with `parallel_kalman_smoother(filtered_mean, filtered_cov, transition_stack[1:], process_cov_stack[1:])`. **Remove** the direct `stochastic_point_process_smoother` call and its import from `theta_sweep.py` (the equivalence test below imports the library smoother in the test module). Passing `cycle_starts` without `cycle_offset_variance` (or vice versa) raises `ValueError`.
- **Model and result.** `ThetaSweepModel.__init__` gains `cycle_offset: bool = False`, `cycle_offset_variance: float = 25.0` (validated: variance positive); `min_cycle_duration` already ships in phase 1. With `cycle_offset=True`, `_fit_bouts` appends the direction column to the design (`np.column_stack([design, direction[start:end]])`), passes `cycle_starts=` and `cycle_offset_variance=` to `theta_sweep_smoother`, and `_assemble_result` fills `cycle_index`, `cycle_offset_mean = mean[:, -1]`, `cycle_offset_var = cov[:, -1, -1]`, computes the amplitude / offset / correlation / origin summaries on the coefficient block `mean[:, :K]`, `cov[:, :K, :K]` only, and `represented_position` from the full design row and full state. `phase_shuffle_scores` supplies randomized observation phase and the original `cycle_phase`; offset resets always use the original phase. Never run cycle detection on the surrogate, whose independent rotations introduce jumps.
- **Docs.** `ThetaSweepModel` docstring: the offset model, its reset semantics, the interpretation of `cycle_offset_variance` as cycle-to-cycle sampling variance (Ujfalussy & Orbán 2022; see [overview](overview.md#scientific-background-and-literature)), that it is fixed in this phase. `CHANGELOG.md` `### Added`: the offset state and the engine change (documented as internal: "the constant-dynamics path now runs through the module's own time-varying scan; pinned to `stochastic_point_process_smoother` by test"). Docstrings of `theta_sweep_smoother` updated for the new arguments and the `D = K + 1` shapes.

## Deliberately not in this phase

- Fitting `sigma_c^2` or `q` (EM) and the `sigma_c^2 = 0` vs free test — phase 2b. In this phase users compare `result.log_likelihood` across a grid of `cycle_offset_variance` values by hand (documented in the class docstring as a profile).
- Any change to `point_process_kalman.py`, `kalman.py` or `position_decoder.py`; `parallel_kalman_smoother` is used as-is.
- Changes to the simulator (`cycle_offset_std` already exists and generates offsets per detected-cycle boundary).
- A per-cycle *amplitude* (random effect on `a` rather than an additive offset) — a different model; note it as a follow-up in the CHANGELOG only if the real-data smoke motivates it.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_theta_sweep.py::test_theta_cycle_index_counts_wraps` | Phase `wrap_to_pi(2 pi 8 t)` over 2 s at 4 ms gives 16 cycle increments at exactly the bins where the phase wraps from `+pi` to `-pi`; the same phase shifted into `[0, 2 pi)` gives the same index array; a constant phase raises `ValueError`. |
| `::test_surrogate_preserves_offset_resets` | For cycle-randomized observation phase, the transition/process-covariance stacks and offset reset bins equal those from the original phase exactly; only observation design columns change. |
| `::test_theta_cycle_index_merges_jittered_wraps` | Phase that crosses `pi` back and forth over 3 bins (jitter) yields one cycle start with `min_cycle_bins=5` and more than one with `min_cycle_bins=1`. |
| `::test_cycle_starts_include_segment_start` | `cycle_starts_from_index([0, 0, 1, 1, 2])` is `[T, F, T, F, T]`. |
| `::test_transition_stacks_reset_offset_only_at_cycle_starts` | `sweep_transition_stacks`: `A[t, K, K] == 0` and `Q[t, K, K] == sigma_c^2` where `cycle_starts[t]`, `A[t, K, K] == 1` and `Q[t, K, K] == 0` elsewhere; coefficient block `A = I`, `Q = q dt I` everywhere; without `cycle_starts` shapes are `(T, K, K)`. |
| `::test_constant_dynamics_path_matches_library_smoother` | `theta_sweep_smoother` (no offset) on `small_design` / `small_maps`, `K = 2`, and on a fixed-offset `K = 1` design, equals `stochastic_point_process_smoother` with the same arguments: means, covariances, cross-covariances and log-likelihood to `rtol 1e-7, atol 1e-9`. Guard: the smoothed means differ from the filtered means by `> 1e-3` somewhere. |
| `::test_offset_posterior_is_constant_within_a_cycle` | With `cycle_starts` from a 3-cycle phase on `small_design`, `cycle_offset_mean` and `cycle_offset_var` are constant (to `1e-9`) within each cycle and change across cycle starts. |
| `::test_offset_variance_never_exceeds_reset_variance` | `cycle_offset_var <= cycle_offset_variance (1 + 1e-9)` at every bin (observations only shrink the reset prior). |
| `::test_offset_arguments_must_be_paired` | `cycle_starts` without `cycle_offset_variance`, `cycle_offset_variance <= 0`, or `cycle_starts[0] == False` raise `ValueError`. |
| `::test_result_offset_fields_are_nan_outside_bouts` | After a fit with `cycle_offset=True` (inside a slow test's shared fixture): `cycle_index == -1`, `cycle_offset_mean` and `cycle_offset_var` are `NaN` exactly where `running_mask` is `False`; `cycle_index` is non-decreasing over running bins and increments by 1 at each new cycle. |
| `::test_offset_model_recovers_cycle_offsets` (slow) | Session with `cycle_offset_std = 6`, `a = 15`: Pearson `r > 0.6` between per-cycle `cycle_offset_mean` (one value per detected cycle, post-burn-in) and `true_cycle_offset`; amplitude RMSE `< 3 cm`; `>= 85%` of per-cycle 95% intervals (`mean +- 1.96 sqrt(var)`) cover the true offset. |
| `::test_offset_model_reduces_to_base_model_as_variance_vanishes` (slow) | `cycle_offset_variance = 1e-6` on the `constant_session` fixture: `coefficient_mean` matches the `cycle_offset=False` fit to `atol 1e-2 cm` and `log_likelihood` to `1 nat`. |
| `::test_offset_model_null_is_not_detected` (slow) | `a = 0`, `cycle_offset_std = 0`, `cycle_offset_variance = 25`: the origin lies in the coefficient credible region in `>= 90%` of bins and the per-cycle offset intervals cover 0 in `>= 90%` of cycles. |
| `tests/test_calibration_point_process.py::TestThetaSweepSmootherCalibration::test_offset_model_at_true_parameters` (slow) | Replicates as in phase 1 plus a per-cycle offset drawn with `sigma_c = 5`; `z` over `(c, s, epsilon)`: `mean` within 4 SE of 0; `E[z^2]` pinned `+-4 SE` and within `(0.95, 1.2)`; coverage `> 0.87`. |
| `uv run mypy`, ruff check / format | Clean. |

Mark slow / integration tests explicitly (e.g., `pytest.mark.slow`).

## Fixtures

- `small_maps`, `small_design` (phase 1); a `three_cycle_design` built with `build_sweep_design` on a phase covering three cycles (`T = 90`, 4 ms bins at 8 Hz is ~31 bins per cycle) plus the appended direction column, and its `cycle_starts`.
- `offset_session` (module scope): `session(seed=0, amplitude=15.0, cycle_offset_std=6.0)` fitted once with `ThetaSweepModel(..., cycle_offset=True, cycle_offset_variance=36.0)`; the `constant_session` fixture from phase 1 is reused for the vanishing-variance test.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): `theta_sweep.py` no longer imports or calls `stochastic_point_process_smoother`; the only engine is `_sweep_filter_scan` + `parallel_kalman_smoother`.
- User-facing documentation listed as tasks is updated, not deferred.
