# Phase 3 — `n_iterations` / `parallel` on `position_decoder_smoother` and `PositionDecoder`, with the track penalty as a Gauss–Newton pseudo-observation

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#position-decoder-sites)

**Inputs to read first:**

- [src/state_space_practice/position_decoder.py:958-1178](../../../src/state_space_practice/position_decoder.py) — `_run_filter_scan`: intensity closures (`:1028-1060`), penalty value/gradient (`:1068-1082`), the step (`:1084-1153`) with the rank-1 penalty downdate (`:1096-1107`), adaptive inflation (`:1109-1131`) and the Laplace update (`:1133-1144`, `diagonal_boost=_DECODER_DIAGONAL_BOOST`, `:58`).
- [src/state_space_practice/position_decoder.py:1260-1537](../../../src/state_space_practice/position_decoder.py) — `_position_decoder_filter_with_predictions`: dynamics and `init_position` / `init_cov` handling (`:1306-1371`), KDE vs bilinear and inflation arguments (`:1373-1428`), the scan call (`:1441-1470`), the inflation and divergence warnings (`:1472-1535`).
- [src/state_space_practice/position_decoder.py:1539-1612](../../../src/state_space_practice/position_decoder.py) — `position_decoder_smoother` (RTS with stored predictions at `:1599-1606`).
- [src/state_space_practice/position_decoder.py:870-915, 1665-1707, 1774-1841](../../../src/state_space_practice/position_decoder.py) — `DecoderResult`, `PositionDecoder.__init__`, `decode`.
- [src/state_space_practice/position_decoder.py:136-197](../../../src/state_space_practice/position_decoder.py) — `build_position_dynamics` (`A`, diagonal PD `Q`).
- Phase-1/2 code: `_LinearisationProblem` (`extra_sites` hook), `_iterated_laplace_smoother_core`, `_warn_iterated_smoother_diagnostics`, `IteratedSmootherDiagnostics`.
- [src/state_space_practice/tests/test_position_decoder.py:1262, 1882-1921](../../../src/state_space_practice/tests/test_position_decoder.py) — `realistic_decoding` and `inflation_problem` fixtures.
- [src/state_space_practice/tests/test_likelihood_identities.py:1141-1183](../../../src/state_space_practice/tests/test_likelihood_identities.py) — `TestPositionDecoderIdentities` (how the decoder's LL is checked; the identical-last-step property).

**Contracts referenced:**

- [Public keywords and return layout](shared-contracts.md#public-keywords) — same four keywords on `position_decoder_smoother`; `PositionDecoder(n_iterations=1, parallel=False)`; `decode(method="filter")` with `n_iterations > 1` raises `ValueError`.
- [Pseudo-observation site contract](shared-contracts.md#pseudo-observation-site-contract) — additional sites add to `(G_t, g_t)` and to the merit's log prior.
- [Gauss–Newton core interface](shared-contracts.md#gauss-newton-core-interface) — `extra_sites` is filled here.
- [Warning contract](shared-contracts.md#warning-contract).

**Designs referenced:** [designs.md#position-decoder-sites](designs.md#position-decoder-sites), [#iteration-semantics](designs.md#iteration-semantics), [#parallel-in-time-pass](designs.md#parallel-in-time-pass).

## Tasks

- **Hoist the decoder's intensity and penalty closures** out of `_run_filter_scan` into a module-level factory `_decoder_intensity_functions(jax_log_rate_maps, jax_x_edges, jax_y_edges, track_penalty, kde_args, *, dt, sigma_track, grid_dx, grid_dy, n_neurons, n_state, include_velocity, use_kde) -> (log_intensity_func, grad_log_intensity_func, penalty_value_fn, penalty_grad_fn)` returning exactly the closures currently defined at `:1028-1082`. `_run_filter_scan` calls the factory (behaviour unchanged; `TestPositionDecoderIdentities` and the golden regression guard it).

- **Penalty sites and the decoder problem.** Add `_track_penalty_sites` as in [designs.md#position-decoder-sites](designs.md#position-decoder-sites) and `_decoder_linearisation_problem(spikes, A, Q, init_position, init_cov, intensity_fns, dt) -> _LinearisationProblem` with `design_matrix = jnp.zeros((n_time, 0))` (unused; the intensity ignores `Z_t`), `log_intensity = lambda _Z, x: log_intensity_func(x)`, `grad_log_intensity = lambda _Z, x: grad_log_intensity_func(x)`, `family = poisson_family(dt)` (default `max_log_count=20.0` as the filter uses), `include_laplace_normalization=True`, `extra_sites = partial(_track_penalty_sites, penalty_value_fn=..., penalty_grad_fn=..., n_state=n_state)`. Pass `diagonal_boost=_DECODER_DIAGONAL_BOOST` through to `_linearised_measurement_update` (add a `diagonal_boost: float = 0.0` field to `_LinearisationProblem` with default `0.0` so phase-1 callers are unchanged).

- **Wire `position_decoder_smoother`.** Keyword-only `n_iterations=1, parallel=False, convergence_tol=1e-6, return_diagnostics=False`. `n_iterations == 1 and not parallel`: existing body (`:1572-1612`) unchanged. Otherwise: `parallel=False` → the existing filter + prediction-aware RTS is pass 1 (`initial_path = smoother_mean`, `n_gauss_newton_passes = n_iterations - 1`); `parallel=True` → `initial_path = broadcast(init_position)` (after the same `init_position` / `init_cov` resolution as `_position_decoder_filter_with_predictions`, `:1306-1371` — factor that block into `_resolve_decoder_initial_state(...)` so both paths share it), `n_gauss_newton_passes = n_iterations`. Run the core with the decoder problem (un-inflated `A, Q`), warn via `_warn_iterated_smoother_diagnostics(name="position_decoder_smoother")`, return `DecoderResult(position_mean=path, position_cov=final.smoother_cov, marginal_log_likelihood=float(evidence))` (+ diagnostics when requested). Run the divergence check of `:1486-1535` on the returned path too (same warning text, `stacklevel` adjusted).

- **`PositionDecoder`.** `__init__` (`:1665-1707`) gains `n_iterations: int = 1`, `parallel: bool = False` (validated like `max_newton_iter`), attribute `smoother_diagnostics_ = None`. `decode` (`:1774-1841`): `method="filter"` with `n_iterations > 1 or parallel` raises `ValueError("n_iterations > 1 / parallel=True require method='smoother'")`; `method="smoother"` passes the options with `return_diagnostics=True`, stores `self.smoother_diagnostics_`, returns the `DecoderResult`.

- **Public documentation (ships with this phase).** `position_decoder_smoother` and `PositionDecoder` docstrings (parameters; that adaptive inflation only shapes the initialisation; that `DecoderResult.marginal_log_likelihood` is the Laplace evidence at the MAP path including the penalty's log-prior term when `n_iterations > 1`; the `parallel=True` nominal start at `init_position`). `DecoderResult.marginal_log_likelihood` attribute docstring (`:881-886`) amended accordingly. `CHANGELOG.md` `### Added` entry. `README.md` subsection: one sentence that `PositionDecoder` accepts the same options.

## Deliberately not in this phase

- Changing `_run_filter_scan`'s numerics, the inflation heuristic or the penalty downdate — untouched apart from the closure hoist.
- A dedicated `DecoderResult` field for diagnostics (would change the object's constructor); diagnostics live on the decoder instance and in the optional second return value.
- Time-chunked parallel scans (phase-2 follow-up) — the decoder's `d ≤ 4` fits the memory formula up to `T ~ 1e6` on a 24 GB GPU.
- Any change to `PlaceFieldRateMaps` or the KDE evaluation.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_extra_quadratic_site_evidence[sequential,parallel]` | Two-bin linear-Gaussian fixture with nonzero quadratic penalty: compare evidence to an independent dense Gaussian integral at an arbitrary path and the converged path (rtol `1e-9`), including the site constant. Guard that a filtered mean differs from its linearisation point. Compare both modes; sequential/parallel agreement alone cannot catch a shared missing term. |
| `test_decoder_default_call_is_unchanged` (`realistic_decoding` fixture, `inflation_problem` kwargs) | `position_decoder_smoother(**kw)` and `(**kw, n_iterations=1)` return `assert_array_equal`-identical `position_mean`, `position_cov`, equal `marginal_log_likelihood`; `position_decoder_filter` unchanged (`TestPositionDecoderIdentities` passes). |
| `test_penalty_sites_reproduce_filter_downdate` | At `x̂ = m⁻`, the information-form update with `_track_penalty_sites` (no spikes) equals the filter's Woodbury downdate at `:1096-1107` to rtol `1e-10` on a point with `pen > 0`; guard `pen > 1e-3`. |
| `test_decoder_converged_path_is_stationary_point` (bilinear maps, zero penalty inside the arena, `n_iterations=12`) | `jax.grad` of the decoder objective (Poisson log-lik + Markov prior) at the returned path has `‖∇‖∞ < 1e-8 · ‖∇‖∞ at the one-pass path`; guard one-pass gradient `> 1e-2`; `smoother_diagnostics_.converged`. |
| `test_decoder_log_posterior_is_monotone` | `diagnostics.log_posterior` non-decreasing; first pass increases it by > 1e-3 (guard). |
| `test_decoder_inflation_only_affects_initialisation` (`inflation_problem` kwargs, `n_iterations=10`) — **slow** | Runs with `adaptive_inflation=cfg` and `adaptive_inflation=None` converge to paths within rtol `1e-6` and evidence within rtol `1e-8`; guard: at `n_iterations=1` they differ by > 1e-3. |
| `test_decoder_iterations_reduce_decoding_error` (`realistic_decoding`, `n_iterations=8`) — **slow** | RMSE to the true trajectory of the iterated path ≤ one-pass RMSE, and the two differ by > 0.1 cm (guard). Record both numbers in the docstring; if no fixture regime shows an improvement, report that instead of loosening the assertion. |
| `test_decoder_penalty_keeps_map_path_on_track` (occupancy-masked rate maps with a track) | Fraction of `n_iterations=8` path points off the track ≤ that of the one-pass path; converged. |
| `test_decoder_parallel_matches_sequential_at_convergence` (`n_iterations=15` both, `include_velocity` on/off) — **slow** | `position_mean` rtol `1e-7`, `position_cov` rtol `1e-7`, evidence rtol `1e-8`; both `converged`. |
| `test_decode_filter_with_iterations_raises` | `PositionDecoder(dt, n_iterations=3).decode(spikes, method="filter")` raises `ValueError`; `method="smoother"` works and sets `smoother_diagnostics_`. |
| `test_decoder_unconverged_warns` (`n_iterations=2`, `init_position` far from the trajectory, tight `init_cov`) | `pytest.warns(StateSpaceWarning, match="did not converge")`. |
| Existing gates: `tests/test_position_decoder.py` (all), `TestPositionDecoderIdentities`, golden regression | Pass unchanged. |

## Fixtures

- `realistic_decoding` (`tests/test_position_decoder.py:1262`) and `inflation_problem` (`:1882-1921`) — reused as-is.
- New module-scoped `track_decoding` fixture: a linear track (occupancy mask from an `x`-only trajectory, `_build_track_penalty`) with 8 Gaussian fields, `dt = 4 ms`, `T = 400`, seed 11, spikes from the true path — used by the on-track and stationary-point tests.
- Bilinear (non-KDE) rate maps for the stationary-point test: `PlaceFieldRateMaps(rates, edges, edges)` as in `TestPositionDecoderIdentities` (`tests/test_likelihood_identities.py:1147-1155`), so the objective is cheap to differentiate independently.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Additionally for this phase: the closure hoist is behaviour-preserving (`TestPositionDecoderIdentities` and the golden regression unchanged); the decoder's `n_iterations=1` path is the pre-existing body.
