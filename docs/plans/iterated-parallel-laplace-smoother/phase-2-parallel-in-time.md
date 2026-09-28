# Phase 2 — `parallel=True`: every pass through associative scans (parallel information filter + parallel RTS), with baseline capture and comparison

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#parallel-in-time-pass)

**Inputs to read first:**

- [src/state_space_practice/kalman.py:1047-1216](../../../src/state_space_practice/kalman.py) — `_SmootherElement` and `parallel_kalman_smoother`: element construction (`:1168-1178`), terminal element and concatenation (`:1180-1192`), the vmapped associative operator (`:1196-1205`), cross-covariances (`:1210-1214`). The new filter follows the same conventions and this smoother is reused verbatim.
- [src/state_space_practice/kalman.py:270-423](../../../src/state_space_practice/kalman.py) — `_validate_kalman_public_inputs` (constant `H` at `:315`, time-varying `R` at `:330-345`): the validation the public `parallel_kalman_filter` wrapper extends to a time-varying `H`.
- [src/state_space_practice/kalman.py:426-475, 613-711](../../../src/state_space_practice/kalman.py) — `kalman_measurement_update`, `kalman_filter`: the sequential references the parallel filter is tested against.
- Phase-1 code: `_linearised_smoother_pass`, `_LinearisationProblem`, `_iterated_laplace_smoother_core`, the block-path wiring, and `stochastic_point_process_smoother`'s new keyword block.
- [src/state_space_practice/utils.py:114-183](../../../src/state_space_practice/utils.py) — `psd_cholesky` / `psd_logdet` accept batches (needed for the vectorised evidence).
- [docs/plans/2026-04-11-square-root-filter-investigation.md](../2026-04-11-square-root-filter-investigation.md) — numerics context only: the f32 failure modes of the information-form recursion; the parallel pass inherits the float64 requirement.
- `scripts/` (exists; `README.md:72` formats it) — home of the benchmark script.

**Contracts referenced:**

- [Public keywords and return layout](shared-contracts.md#public-keywords) — `parallel`; the nominal-trajectory initialisation; `return_filtered` semantics under `parallel=True`.
- [Linear-pass interface](shared-contracts.md#linear-pass-interface) — the `parallel=True` branch must match the sequential branch to rtol `1e-9` on the same `(problem, path)`.
- [`IteratedSmootherDiagnostics`](shared-contracts.md#iterated-smoother-diagnostics) — `log_posterior` has `n_iterations + 1` entries under `parallel=True`.

**Designs referenced:** [designs.md#iteration-semantics](designs.md#iteration-semantics), [#parallel-in-time-pass](designs.md#parallel-in-time-pass), [#numerical-notes](designs.md#numerical-notes) (memory formula).

## Tasks

- **Baseline capture (before any code change).** Add `scripts/benchmarks/iterated_smoother_benchmark.py` (argparse: `--mode {sequential,parallel}`, `--n-time`, `--n-state`, `--n-neurons 50`, `--n-iterations 6`, `--seed 0`, `--out <json>`, `--save-outputs <npz>`). It simulates an AR(1) latent (`A = 0.98 I`, `Q = 0.01 I`, `m_0 = 0`, `P_0 = I`) with 50 affine Poisson neurons (`baseline log 20 Hz`, weights `N(0, 0.5²)`, `dt = 2 ms`) via `_affine_log_rate`-style design, runs `stochastic_point_process_smoother(..., n_iterations=6, return_diagnostics=True)` twice (first call compiles), and records per run: wall time of the second call (`time.perf_counter`, `jax.block_until_ready`), peak RSS (`resource.getrusage(RUSAGE_SELF).ru_maxrss`), device peak bytes when `jax.devices()[0].memory_stats()` is available (GPU), `converged`, `n_iterations`, device kind, `jax.__version__`, hostname. Run the grid `T ∈ {1e4, 1e5, 1e6} × n_state ∈ {2, 8}` with `--mode sequential` on CPU (and GPU if `jax.devices()` shows one), save JSON to `docs/plans/iterated-parallel-laplace-smoother/benchmarks/baseline-<device>-<host>.json` and outputs (`smoother_mean`, `marginal_log_likelihood`, `smoother_cov` diagonal) to `.npz` next to it. Smoke-test `T = 1e4` first and extrapolate (measured here: 0.165 s / 0.213 s per one-pass smoother at `T = 1e4`, `d = 2 / 8`, so the 6-pass sequential run at `T = 1e6` is ~2–3 min on CPU; abort the `1e6` point if the `1e5` point exceeds 60 s). Commit the JSON/NPZ files (they are small) before touching source.

- **Information-form parallel filter in `kalman.py`.** Add `_FilterElement`, `_combine_filter_elements`, `_parallel_information_filter(init_mean, init_cov, transition_matrix, process_cov, info_matrices, info_vectors)` exactly as in [designs.md#parallel-in-time-pass](designs.md#parallel-in-time-pass), after `parallel_kalman_smoother`. Validate shapes only (`info_matrices` `(n_time, d, d)`, `info_vectors` `(n_time, d)`), no host-side value checks (it is called from jitted code).

- **Public `parallel_kalman_filter` wrapper in `kalman.py`.** Signature
  `parallel_kalman_filter(init_mean, init_cov, obs, transition_matrix, process_cov, measurement_matrix, measurement_cov, validate_inputs=True) -> (filtered_mean, filtered_cov, marginal_log_likelihood)`;
  `measurement_matrix` `(n_obs, d)` or `(n_time, n_obs, d)`, `measurement_cov` `(n_obs, n_obs)` or `(n_time, n_obs, n_obs)`. Extend `_validate_kalman_public_inputs` with an `allow_time_varying_measurement_matrix: bool = False` keyword (default preserves current behaviour for `kalman_filter` / `kalman_smoother`) that accepts the 3-D `H` shape and checks finiteness. `G_t = H_tᵀ R_t⁻¹ H_t`, `g_t = H_tᵀ R_t⁻¹ y_t` via `psd_solve`; the LL vectorised as in the design. NumPy docstring with the Särkkä & García-Fernández 2021 reference and the memory formula. Export nothing new at package level (keep `__init__.py` untouched); mention in CHANGELOG.

- **`parallel=True` branch of `_linearised_smoother_pass`** in `point_process_kalman.py`: vmapped sites over time (`jacobians`, `weights`, `residuals`, `mu`, `G`, `g`), `_parallel_information_filter`, `parallel_kalman_smoother`, vectorised evidence and plug-in log-likelihood as in [designs.md#parallel-in-time-pass](designs.md#parallel-in-time-pass). Remove the phase-1 `NotImplementedError`. `extra_sites` (phase 3) is applied per time step inside the same vmap; pass the combined information matrices/vectors and likelihood-plus-extra-site value to `_site_log_value` for evidence. Preserve the resolved caller family and its predictor-sized score in this path too, including NB/ZIG; supplied families use dense dispatch.

- **Wire `parallel` into `stochastic_point_process_smoother`.** Keyword-only `parallel: bool = False` (after `n_iterations`; validate `bool`). `parallel=True` path: skip the Laplace-EKF smoother; `initial_path = jnp.broadcast_to(init_mean, (n_time, d))`; `n_gauss_newton_passes = n_iterations`; the core with `parallel=True`; `filtered_*` are `final.filtered_mean/filtered_cov` (documented as pseudo-model filtered moments). Block path: `jax.lax.map` over neurons of the single-neuron core (not `vmap`) when `parallel=True`; nominal path per neuron = `init_means_per_neuron[j]` broadcast. The dense jitted impl gets `parallel` as a static argument.

- **Model pass-through.** `parallel: bool = False` on `PointProcessModel.__init__` and `PlaceFieldModel.__init__`, forwarded wherever `n_iterations` is (E-step, SGD loss, finalize). In `_sgd_loss_fn` the iterated-smoother branch is taken when `self.n_iterations > 1 or self.parallel`.

- **Comparison (after the code change).** Run the benchmark script with `--mode parallel` on the same grid and device(s) (`n_iterations` chosen so both runs report `converged`; increase to 10 if 6 does not converge from the nominal path and record it). Save `parallel-<device>-<host>.json` / `.npz`. Add a small `scripts/benchmarks/compare_iterated_smoother_outputs.py` that loads a baseline and a comparison `.npz` and asserts `smoother_mean` rtol `1e-6` (atol `1e-8`), `marginal_log_likelihood` rtol `1e-8`, covariance diagonal rtol `1e-6`, and prints a table `T | n_state | sequential s | parallel s | speedup | peak mem seq | peak mem par`. Paste the table into this file under a new "Results" heading with device, jax version and date; on CPU a speedup ≤ 1 is expected and must be reported as measured. If no GPU is present, the GPU column is `n/a` and the phase's speed claim in the CHANGELOG is limited to "O(log T) span; CPU timings in docs".

- **Public documentation (ships with this phase).**
  - `stochastic_point_process_smoother` docstring: `parallel` parameter; the nominal-trajectory initialisation; that `parallel=True` and `parallel=False` agree at convergence but not at small `n_iterations`; the `return_filtered` caveat; the memory formula and the block-path `lax.map` behaviour; reference Yaghoobi et al. 2021 and Särkkä & García-Fernández 2021.
  - `PointProcessModel` / `PlaceFieldModel` docstrings: `parallel`.
  - `CHANGELOG.md` `### Added`: `parallel=` on the smoother and models; `kalman.parallel_kalman_filter` (time-varying `H_t`, `R_t`); benchmark location.
  - `README.md` "Iterated Laplace smoothing" subsection: one sentence on `parallel=True` for GPU and the memory rule of thumb.

## Deliberately not in this phase

- Time-chunked (blocked) associative scans to fit `d = 36`, `T ≳ 2e5` on one GPU. Trigger: a `PlaceFieldModel` user with `parallel=True` hitting an OOM at their `T`; design note: carry the composed prefix element across chunks (the operator is associative), same for the reverse smoother scan.
- Parallelising the Laplace-EKF initialisation (`parallel=False` pass 1): not possible without changing what pass 1 is; the nominal-trajectory start is the parallel initialisation by design.
- `PositionDecoder` — phase 3 (it adds `parallel` there once the penalty sites exist).
- Time-varying `H_t` on the *sequential* `kalman_filter` / `kalman_smoother`; only the validation helper learns the 3-D shape, gated by a keyword that the sequential entry points do not set.
- `n_iterations=1, parallel=False` semantics: untouched; the bit-identical test from phase 1 must still pass.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_parallel_noncanonical_families[nb,zig]` — **slow** | The public parallel smoother preserves the supplied family, reaches the same stationary path/objective as sequential iteration, and matches an independent dense MAP solve; ZIG uses N=2 observations and 2N predictors. Compare each family against its own likelihood, never a Poisson surrogate. |
| `test_parallel_kalman_filter_matches_sequential_filter` (`tests/test_kalman.py`, random LGSSM from `kalman_model_params` in `conftest.py:369`, constant `H`, `R`) | `filtered_mean`, `filtered_cov`, `marginal_log_likelihood` match `kalman_filter` to rtol `1e-10`; T ∈ {1, 2, 7, 64} (odd lengths and `T=1` exercise the scan edges). |
| `test_parallel_kalman_filter_time_varying_measurement_matches_step_loop` | With `H_t`, `R_t` varying over time, matches a Python loop of prediction + `kalman_measurement_update` to rtol `1e-10`. |
| `test_parallel_kalman_filter_rejects_bad_shapes` | `ValueError` for `H` of shape `(n_time - 1, n_obs, d)` and for non-PD `R_t`; `kalman_filter` still rejects a 3-D `H` (default keyword). |
| `test_information_filter_zero_information_is_pure_prediction` | With `G = g = 0` the filtered moments equal the prior propagation `m_t = A^t m_0`, `P_t = A P_{t-1} Aᵀ + Q` (rtol `1e-12`) — the masked-bin composition rule. |
| `test_parallel_pass_matches_sequential_pass` (`_linearised_smoother_pass` both branches on the phase-1 `iterated_problem` at its one-pass path) | All `_LinearPassOutput` fields agree to rtol `1e-9` (evidence rtol `1e-10`). |
| `test_parallel_and_sequential_smoothers_converge_to_same_map` (`n_iterations=15` both) | Means rtol `1e-7`, covariances and cross-covariances rtol `1e-7`, evidence rtol `1e-8`; both `converged`; guard: at `n_iterations=1` the two differ by > 1e-3 (the parallel start is the nominal trajectory). |
| `test_parallel_block_path_matches_dense` (block problem, `parallel=True`, `n_iterations=8`) | Means rtol `1e-9`, block covariances (`to_dense`) rtol `1e-9`; guard `all(step_sizes == 1)`. |
| `test_parallel_filtered_output_is_pseudo_model_filter` | With `parallel=True, return_filtered=True`, `filtered_mean` equals `_parallel_information_filter` applied to the sites at the returned path (rtol `1e-10`) — pins the documented semantics. |
| `test_parallel_evidence_gradient_matches_sequential_gradient` (`n_iterations=12`, both converged) — **slow** | `jax.grad` of the evidence w.r.t. `init_mean` and `diag(Q)` agree to rtol `1e-5` between `parallel=True/False`. |
| `test_parallel_unconverged_warns` | `parallel=True, n_iterations=1` on the hard fixture warns `StateSpaceWarning` (`did not converge`); `n_iterations=30` does not. |
| Model threading: `test_place_field_model_parallel_fit_matches_sequential_fit_ll` — **slow (auto-marked)** | `PlaceFieldModel(n_iterations=10, parallel=True).fit` and `parallel=False` on the same small problem give final LLs within rtol `1e-6` and both `smoother_diagnostics_.converged`. |
| Baseline / comparison scripts | `compare_iterated_smoother_outputs.py` passes on the saved `.npz` pairs for every grid point run; table pasted into this file. |
| Existing gates: phase-1 slice, `tests/test_kalman.py` (all), golden regression | Pass unchanged. |

## Fixtures

- `kalman_model_params` (`tests/conftest.py:369`) and `random_stable_matrix` / `random_spd_matrix` (`tests/oracles.py:826-835`) for LGSSMs.
- Phase-1 fixtures (`iterated_problem`, block-parity problem, hard problem).
- Benchmark data are generated by the script (fixed seed 0); JSON/NPZ committed under `docs/plans/iterated-parallel-laplace-smoother/benchmarks/`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Additionally for this phase: the baseline JSON predates the source changes in git history; the results table reports measured numbers (including any slowdown on CPU) and the device; no speedup is claimed in CHANGELOG beyond what the table shows.
