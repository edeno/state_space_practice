# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

**Graph prerequisite update (2026-10-08):** The drift-scale execution added
`laplace_smoothing.py`, a reusable single-observation Poisson/log-linear joint
Laplace core with information sites, Armijo damping, final Hessian/lag moments,
normalized evidence, implicit derivatives and a reduced q=0 branch. It reuses
shared RTS and agrees with `TemporalRateGP` and an independent dense oracle.
This implements the machinery needed by the graph branch, not all phases here:
the planned general GLM interfaces, integration into unrelated point-process
models/decoder, parallel scans and TemporalRateGP consolidation remain open.
The graph task requires adaptive convergence and masks, and therefore uses
adaptive stopped primal iteration with an exact implicit derivative. Reconcile
these interfaces and current master before implementing this plan's remaining
phases; its older fixed-pass/derivative assumptions are not completed contracts.

## Current codebase integration points

Paths are relative to `src/state_space_practice/`. Line numbers were verified against
`master` at `4b74f03`.

- `point_process_kalman.py:2392-2410` — `stochastic_point_process_smoother` signature: gains keyword-only `n_iterations`, `convergence_tol`, `return_diagnostics` (phase 1) and `parallel` (phase 2). The positional/keyword parameters that exist today are untouched. `:2495-2518` (validation), `:2526-2569` (block dispatch) and `:2571-2602` (dense filter + backward pass) remain the `n_iterations=1, parallel=False` path and are executed unchanged in that case.
- `point_process_kalman.py:927-1212` — `_point_process_laplace_update`: the per-bin information-form Laplace update. Its structure (`psd_cholesky` of the predicted covariance at `:1090-1091`, `prior_precision + J' diag(w) J` at `:1138-1139`, the Laplace normaliser at `:1192-1208`) is the template for the new linearised measurement update; the function itself is untouched.
- `point_process_kalman.py:1215-1265` — `GLMFamily` / `poisson_family`: the family abstraction the new pseudo-observation sites are built on (`fisher_weight`, `mean`, `loglik_*`). Untouched; the sequential and parallel Gauss–Newton passes take a `GLMFamily`, which is also the hook the negative-binomial plan needs.
- `point_process_kalman.py:727-730` — `_ARMIJO_C`, `_LINE_SEARCH_MAX_BACKTRACKS`: the Armijo convention reused by the trajectory-level line search (constant `1e-4`, halving steps).
- `point_process_kalman.py:1745-1866` — `_stochastic_point_process_filter_impl`: iteration-0 forward pass (dense). Untouched; its scan (`:1855-1864`) shows the carry/prediction convention (`x_0 ~ N(m_0, P_0)`, predict before the first update) the linear passes must match.
- `point_process_kalman.py:1873-1979`, `:2037-2122`, `:2269-2389` — block-diagonal forward core, smoother core and block smoother wrapper: iteration 0 on the block path. Untouched; the Gauss–Newton passes on the block path `vmap` the single-neuron core over the per-neuron problems in the same `(n_neurons, n_time, block_size)` layout and reuse `_concatenate_neuron_means` / `_package_block_covs` (`:2137-2149`).
- `point_process_kalman.py:2605-2641` — `_stochastic_point_process_smoother_backward`: sequential RTS pass (duplicate of `kalman.rts_backward_scan`). Untouched; the sequential linear passes call `kalman.rts_backward_scan` directly.
- `point_process_kalman.py:2993-3010`, `:3065-3105`, `:3314-3336`, `:3348-3371` — `PointProcessModel.__init__`, `_e_step`, `_sgd_loss_fn`, `_finalize_sgd`: gain `n_iterations` (phase 1) and `parallel` (phase 2) pass-through; `_sgd_loss_fn` switches from the filter's LL to the iterated smoother's evidence only when `n_iterations > 1 or parallel`.
- `place_field_model.py:361-374`, `:983-1024`, `:1335-1348`, `:1574-1609`, `:1623-1658` — `PlaceFieldModel.__init__`, `_e_step`, `_capture_state`, `_sgd_loss_fn`, `_finalize_sgd`: same pass-through; `score` (`:1886`, filter call at `:1961-1978`) is untouched (see Open Questions 3).
- `position_decoder.py:958-1178` — `_run_filter_scan`: iteration-0 filter for the decoder (track-penalty rank-1 downdate at `:1096-1107`, adaptive inflation at `:1109-1131`, Laplace update at `:1133-1144`). Untouched. `position_decoder_smoother` (`:1539-1612`) gains `n_iterations` / `parallel` / `convergence_tol` / `return_diagnostics`; its RTS call at `:1599-1606` is the `n_iterations=1, parallel=False` path. `PositionDecoder.__init__` (`:1665-1707`) and `decode` (`:1774-1841`) pass the options through.
- `kalman.py:1072-1216` — `parallel_kalman_smoother`: the associative-scan RTS pass reused as the backward half of every parallel pass. Untouched. A new `parallel_kalman_filter` (information-form elements, time-varying `H_t` / `R_t`) is added next to it (phase 2); `kalman_filter` / `_validate_kalman_public_inputs` (`:270-423`, constant `H` at `:315`) are not extended — the sequential linear passes use the information-form update instead (see designs.md, *Alternatives considered*).
- `kalman.py:778-842` — `rts_backward_scan`: sequential backward pass reused by every sequential linear pass.
- `kalman.py` (new helpers) — `markov_prior_residuals` / `markov_prior_quadratic_forms`: the prior quadratic form of the line-search merit, generalising `temporal_rate_gp._prior_whitened_residuals` (`temporal_rate_gp.py:89-115`) to a non-zero first mean.
- `temporal_rate_gp.py:236-395` — `_infer_log_rate_traced` (`_newton_target` `:272-286`, `_newton_step` `:288-349`, evidence `:357-386`): the existing 1-D iterated Laplace / Gauss–Newton smoother. Phase 1 mirrors its algorithm and phase 4 replaces this private loop by a call to the shared core; `poisson_log_rate_site` (`:190-233`), `LaplaceRateResult` (`:118-146`) and `_warn_laplace_diagnostics` (`:149-187`) stay public/behaviour-compatible.
- `tests/test_oracle_point_process.py`, `tests/test_approximation_trends.py`, `tests/test_likelihood_identities.py`, `tests/test_oracle_gp.py`, `tests/recovery_helpers.py` — existing verification infrastructure extended, not replaced (see the phase validation slices).

## Scope and dependency policy

### Goals

- A Gauss–Newton (iterated Laplace) smoother for the point-process state-space model that (a) re-linearises the GLM likelihood at the smoothed path, (b) is a monotone ascent on the joint log posterior through Armijo damping, (c) returns the Laplace covariances, lag-one cross-covariances and Laplace evidence at the converged MAP path, and (d) reports convergence and line-search fallbacks the way `temporal_rate_gp` does.
- Bit-for-bit backwards compatibility: every existing call (and `n_iterations=1, parallel=False` explicitly) runs today's code and returns today's arrays.
- The same core on the dense and block-diagonal paths and, for the decoder, with its track penalty as an extra Gauss–Newton pseudo-observation.
- A parallel-in-time variant (`parallel=True`) in which every pass, including the initialisation, is an associative scan; the two variants converge to the same MAP path (unique for the log-concave Poisson–linear model).
- Differentiable evidence for `fit_sgd` (unrolled fixed-length iteration).
- One implementation of the iterated Laplace smoother in the library: `temporal_rate_gp` runs on the shared core after phase 4.

### Non-Goals

- No change to the one-pass Laplace-EKF *filter* (`stochastic_point_process_filter`) and its per-bin Fisher iterations (`max_newton_iter`); the iterated smoother composes with them (they define iteration 0).
- No time-varying `measurement_matrix` on the sequential `kalman_filter` / `kalman_smoother`: the sequential linearised passes use the information-form update inside `point_process_kalman.py` (no `R = W^{-1}` is ever formed), so that extension is not needed here.
- No implicit (fixed-point) differentiation, no early exit with `lax.while_loop`, no exposure of the nominal trajectory / warm start on the public API, no time-chunked associative scan — each is listed under *Deliberately not in this plan* in the phase that would own it, with a trigger.
- No new observation families (negative binomial, zero-inflated) and no observation masks — see the dependency policy.
- No changes to the EM M-steps, `run_em`, or the SGD mixin.

### Dependency policy

- No new Python dependencies. Benchmarks use `time.perf_counter`, `resource.getrusage` and `jax.Device.memory_stats()` only.
- [docs/plans/masks-and-multi-sequence/](../masks-and-multi-sequence/) adds `obs_mask=` to the same entry points. This plan's pseudo-observation sites are information-form pairs `(G_t, g_t)`, so a masked bin is `G_t = 0, g_t = 0` and a zero log-likelihood term; the sequential update (`I + Q·0 = I`) and the parallel elements need no special case. That plan owns the mask keyword and its propagation into iteration 0; this plan only guarantees the composition rule stated in [shared-contracts.md#pseudo-observation-site-contract](shared-contracts.md#pseudo-observation-site-contract).
- [docs/plans/glm-families-nb-zig/](../glm-families-nb-zig/) adds `GLMFamily` instances. The Gauss–Newton core takes a `GLMFamily`; for a non-canonical family the pass is Fisher scoring (expected Hessian), still a monotone ascent under the line search. That plan owns the families; this plan owns the family-generic core.
- The deliverable names other plans will use are exactly `n_iterations: int = 1` and `parallel: bool = False` on `stochastic_point_process_smoother`, `PlaceFieldModel`, `PositionDecoder` and `PointProcessModel`.

## Metrics

- **Bit-identical default**: `np.testing.assert_array_equal` on every output of `stochastic_point_process_smoother` / `position_decoder_smoother` between the unchanged call and `n_iterations=1, parallel=False`; the recorded EM golden values (`tests/test_em_golden_regression.py::test_em_matches_recorded_values`) pass unchanged.
- **Code computes the approximation it claims**: one Gauss–Newton pass equals the dense block-tridiagonal Newton step (rtol 1e-8); the converged path is a stationary point of the joint log posterior (`max|∇Ψ| < 1e-8 · scale`); the returned covariances and cross-covariances reconstruct the inverse joint Hessian (rtol 1e-8); the returned evidence equals the dense Laplace evidence (rtol 1e-8).
- **Sibling agreement**: on a 1-D Matérn-3/2 LGCP the iterated smoother's mode, variance and evidence match `infer_log_rate` to rtol 1e-8 (both converged).
- **Accuracy gain**: in the near-Gaussian regime of `tests/test_oracle_point_process.py` the standardised smoothed-mean error against the grid-quadrature posterior decreases monotonically over `n_iterations ∈ {1, 2, 4, 8}` and the converged evidence/variance errors are no larger than the one-pass ones; observed values are recorded in the test docstring like the existing trend tests.
- **Parallel correctness**: `parallel=True` and `parallel=False` agree at convergence to rtol 1e-7 (means, covariances) and 1e-8 (evidence); the parallel information filter matches `kalman_filter` to rtol 1e-10 on random LGSSMs.
- **Parallel performance**: measured wall-clock per pass on the baseline grid (T ∈ {1e4, 1e5, 1e6}, n_state ∈ {2, 8}, 50 neurons) before and after, CPU always, GPU when present; outputs on the grid agree to rtol 1e-6 at convergence. No speedup claim is made without the table.
- **Diagnostics**: unconverged runs emit exactly one `StateSpaceWarning` naming the final relative update and `n_iterations`; converged runs emit none (pytest treats warnings as errors).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The evidence reported for `n_iterations ≥ 2` (joint Laplace at the MAP path) is a different approximation of `log p(y)` than the one-pass sequential Laplace-EKF LL; EM convergence checks and user comparisons across `n_iterations` could be misread. | Documented in the smoother docstring and CHANGELOG; the oracle test compares *both* against the exact quadrature `log p(y)`; models store `smoother_diagnostics_` so the user can see which quantity was computed. The definition is fixed once (shared-contracts.md#public-keywords). |
| A non-converged fixed-length iteration returns an unconverged mode, variance and evidence silently. | Final pass measures the un-taken Gauss–Newton step (`max_abs_update`); `converged` is reported and a `StateSpaceWarning` is raised host-side when concrete, mirroring `temporal_rate_gp._warn_laplace_diagnostics`. |
| Undamped Gauss–Newton overshoots when the prior mean is far from the data (documented for the 1-D case in `temporal_rate_gp.py:293-298`). | Armijo backtracking on the exact joint log posterior at every Gauss–Newton pass (12 halvings, `c = 1e-4`, relative slack `1e-10`); rejected passes keep the iterate (monotone by construction) and are counted. |
| Zero-count bins with tiny expected count give infinite `R_t = W_t^{-1}`. | Information form throughout: only `W_t` and `y_t − μ_t` are used; the evidence is written so the `log w` and `r²/w` terms cancel algebraically (designs.md#laplace-evidence). |
| Dense vs block-diagonal paths differ when the per-neuron line searches damp differently (same caveat as today's block filter, `point_process_kalman.py:2191-2198`). | Parity test guards that all step sizes are 1 on both paths; the divergence when they are not is documented, not hidden. |
| Parallel pass memory: the filter elements are `(T, d, d)` arrays with ~3× scan intermediates; at `d = 36` (PlaceFieldModel) and `T = 1e6` this is ~100 GB per neuron. | Block path runs `lax.map` over neurons (parallel over time, sequential over neurons); the memory model is documented with a formula and the baseline grid is `n_state ∈ {2, 8}`. Time-chunked scans are a listed follow-up with a trigger. |
| Reverse-mode memory grows linearly with `n_iterations` (unrolled). | Documented; `n_iterations` defaults to 1 so SGD users opt in; implicit differentiation is a listed follow-up with a trigger. |
| Singular `Q` makes the merit's prior quadratic form ill-defined. | `psd_cholesky`'s scale-relative shift (`utils.py:74-112`) gives constrained directions a huge but finite precision; the Gauss–Newton direction respects the constraint so the whitened innovation is round-off; documented in designs.md#numerical-notes. |
| Refactoring `temporal_rate_gp` onto the shared core changes its numerics. | Phase 4 keeps `min_weight` semantics through a custom `GLMFamily`, is gated by the dense-GP oracle (`tests/test_oracle_gp.py::test_laplace_rate_gp_equals_dense_gp_laplace`, rtol 1e-8) and the finite-difference gradient test, and records the one intended behaviour change (rejected step keeps the iterate). |

## Rollout Strategy

All additions are keyword-only with defaults that select today's code path: `n_iterations=1`, `parallel=False`, `convergence_tol=1e-6`, `return_diagnostics=False`. No deprecations, no removals of public behaviour, no changes to positional signatures or to the tuple layout of existing returns (diagnostics are appended only when requested). The existing sequential path is not removed; it is iteration 0 of the new algorithm. Phases ship in order 1 → 2 → 3 → 4; each leaves the suite green. Phase 4 is the only phase that removes code (the private Newton loop in `temporal_rate_gp.py`) and it is behaviour-preserving up to the recorded fallback change.

## Open Questions

1. **Early exit.** Fixed-length `lax.scan` (differentiable, compile-once, matches `temporal_rate_gp`) vs `lax.while_loop` on the convergence test (saves passes, not reverse-differentiable). Current answer: fixed length; convergence is *reported*. Revisit when a profiled EM run spends > 30 % of its E-step time in passes after convergence.
2. **Warm-starting the nominal trajectory across EM iterations** (previous E-step's smoothed means as `initial_path`). Current answer: internal `initial_path` only (used by the parallel initialisation and by `temporal_rate_gp`); exposing it changes EM's path dependence and deserves its own decision. Trigger: an EM profile showing ≥ 3 Gauss–Newton passes per E-step after the first few iterations.
3. **`PlaceFieldModel.score` and `PointProcessModel`-style held-out scoring** keep the one-pass filter LL even when the model was fitted with `n_iterations > 1`. Current answer: leave as is (a causal predictive score is a legitimate held-out quantity); document the mismatch in the `score` docstring in phase 1.
4. **Diagnostics for the `n_iterations=1, parallel=False` path** when `return_diagnostics=True`: current answer is a stub with `n_gauss_newton_passes = 0`, `log_posterior` of length 1 (the joint log posterior of the one-pass path, cheap to evaluate), `max_abs_update = nan`, `converged = False` ("not assessed").
5. **Parallel initialisation** uses the constant nominal trajectory `x̂_t = init_mean` (the prior mean path for `A = I`, the default of both models). Alternative `x̂_t = A^t m_0` needs an eigendecomposition or an O(T d²) scan; deferred until a stable-`A` user reports slow convergence from the constant start.

## Estimated Effort

- Phase 1: ~500 LOC in `point_process_kalman.py` (sites, linearised update, sequential pass, line search, core, diagnostics, wiring), ~40 LOC in `kalman.py` (prior quadratic-form helpers), ~60 LOC of model pass-through, ~700 LOC of tests, ~120 lines of docstrings/CHANGELOG/README.
- Phase 2: ~250 LOC in `kalman.py` (`parallel_kalman_filter` elements/operator/wrapper), ~120 LOC in `point_process_kalman.py` (parallel branch of the linear pass, `lax.map` block variant), ~150 LOC benchmark script, ~350 LOC tests, docs.
- Phase 3: ~250 LOC in `position_decoder.py` (penalty sites, decoder problem, wiring), ~300 LOC tests, docs.
- Phase 4: −180 / +90 LOC in `temporal_rate_gp.py`, ~30 LOC of test message updates.
