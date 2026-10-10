# Phase 2a — `parallel` for the joint Laplace core and `GraphPlaceFieldModel`: associative-scan linear pass, backend default, neuron chunking

[← back to PLAN.md](PLAN.md) · [overview](overview.md#prototype-evidence) · [designs](designs.md#parallel-pass-for-the-joint-laplace-core)

Independent of phase 1: it changes only the linear pass inside the existing joint core.

**Inputs to read first:**

- [src/state_space_practice/laplace_smoothing.py:64-123](../../../src/state_space_practice/laplace_smoothing.py) — `_linear_sites`, the sequential pass being given a parallel twin; its `update` (`:76-92`) holds the site log-normalizer (`:88-91`) to lift out.
- [src/state_space_practice/laplace_smoothing.py:182-432](../../../src/state_space_practice/laplace_smoothing.py) — `poisson_laplace_smoother`: static args (`:182`), dynamic branch (`:276`) with the prior-path scan (`:291-296`) and the two differentiated linear passes (`:326-333`), static branch (`:348`) with the loading scan (`:351-359`), the `static_noise` dispatch (`:427-432`). `_iterate_mode` (`:126`) and the implicit derivative are not touched.
- [src/state_space_practice/kalman.py:1192-1386](../../../src/state_space_practice/kalman.py) — `_SmootherElement`, `parallel_kalman_smoother` (`:1217`) and `_parallel_kalman_smoother_impl` (`:1315`): conventions the new filter follows; the smoother is reused verbatim. Its cross-covariance convention `Cov(x_t, x_{t+1})` equals `rts_backward_scan`'s (`:914`).
- [src/state_space_practice/graph_place_field.py:1454-1716](../../../src/state_space_practice/graph_place_field.py) — `_vmap_by_noise_structure` (`:1454`), `_graph_neuron_log_evidence` (`:1484`, smoother call `:1510`), `_GRAPH_LOSS_STATIC_ARGS` (`:1545`), `_graph_forward_loss` (`:1556`), `_graph_profile_losses` (`:1593`), `_graph_posterior` (`:1660`, smoother call `:1708`).
- [src/state_space_practice/graph_place_field.py:1793-1813](../../../src/state_space_practice/graph_place_field.py) — `GraphPlaceFieldModel.__init__` (`inference_method`, `max_smoother_iter` at `:1812-1813`); the call sites `_e_step` (`:2010`), `_forward_loss` (`:2397`), `_profile_losses` (`:2426`).
- [src/state_space_practice/tests/test_laplace_smoothing.py](../../../src/state_space_practice/tests/test_laplace_smoothing.py) — `joint_problem` fixture (`:17`), dense-oracle test (`:30`), gradient test (`:95`), mixed vmap test (`:139`), branch-hint test (`:283`); [tests/graph_math_reference.py:307](../../../src/state_space_practice/tests/graph_math_reference.py) — `joint_poisson_laplace`, the independent dense oracle.

**Contracts referenced:**

- [Parallel information filter interface](shared-contracts.md#parallel-information-filter-interface) — defined here; phase 2b reuses it unchanged, so do not specialise it to rank-one sites.
- [Joint-core `parallel` keyword](shared-contracts.md#joint-core-parallel-keyword) — `None` resolves by backend; both values converge to the same mode.

**Designs referenced:** [designs.md#parallel-pass-for-the-joint-laplace-core](designs.md#parallel-pass-for-the-joint-laplace-core), [#parallel-in-time-pass](designs.md#parallel-in-time-pass) (filter elements and operator), [#numerical-notes](designs.md#numerical-notes).

## Tasks

- **Baseline capture (before any source change).** Add `scripts/benchmarks/joint_laplace_parallel_benchmark.py` (argparse: `--parallel {false,true}`, `--n-time`, `--n-neurons`, `--rank 20`, `--seed 0`, `--out <json>`, `--save-outputs <npz>`, `--fit`). It simulates the graph setting deterministically: a 1-D track `Environment.from_samples(np.linspace(0, 250, 1001)[:, None], bin_size=5.0)`, a trajectory sweeping the bins, drifting Gaussian place fields (0.5 Hz floor, 15 Hz peak, 15 cm width, centres drifting linearly), `dt = 0.05`, Poisson counts. Per case it builds `GraphPlaceFieldModel(env, dt, rank=rank)`, calls `_prepare_sgd_data`, and times the batched joint-smoother evidence and `jax.value_and_grad` w.r.t. a shared drift scale: compile+first call separately, then 5 steady runs (3 for `T ≥ 12000`) with `jax.block_until_ready` on the **whole output pytree**, reporting min and median. It records device peak bytes from `jax.devices()[0].memory_stats()["peak_bytes_in_use"]` (GPU), so **each case runs in its own process**. With `--fit` it also runs `fit()` (`profile_points=9, profile_rounds=2`) and records wall time and `log_likelihood_`. It saves the evidences, smoothed means and, with `--fit`, the fitted parameters to the `.npz`. Smoke-test `T=3000, N=1` first; from the prototype (overview.md#prototype-evidence) the sequential A100 case at `T=70930, N=1` takes ~85 s per value+grad call, so the full sequential grid is ~30 min. Run the grid `T ∈ {3000, 12000, 70930} × N ∈ {1, 8, 32}` with `--parallel false` on CPU and on GPU when present. Skip `N=32` at `T=70930`; sequential value+grad needs ~3.6 GiB per neuron there. Commit `docs/plans/iterated-parallel-laplace-smoother/benchmarks/joint-baseline-<device>-<host>.json` / `.npz` before touching source.

- **`_parallel_information_filter` in `kalman.py`.** Add `_FilterElement`, `_combine_filter_elements` and `_parallel_information_filter` after `_parallel_kalman_smoother_impl` (before `sum_of_outer_products`, `kalman.py:1388`), with the code in [designs.md#parallel-in-time-pass](designs.md#parallel-in-time-pass) and the first-observed-state prior convention of the [interface contract](shared-contracts.md#parallel-information-filter-interface). Shape checks only; no host-side value checks. NumPy docstring with shapes, the convention and the Särkkä & García-Fernández 2021 reference.

- **Parallel linear pass in `laplace_smoothing.py`.** Lift the site log-normalizer out of `_linear_sites.update` into module-level `_site_log_normalizer`, and have `update` call it so the sequential pass is unchanged. Add `_linear_sites_parallel` and `_transition_powers` as in [designs.md#parallel-pass-for-the-joint-laplace-core](designs.md#parallel-pass-for-the-joint-laplace-core). Add `resolve_parallel(parallel: bool | None) -> bool`.

- **`parallel` on `poisson_laplace_smoother`.** Add the keyword-only static argument `parallel: bool | None = None` (append `"parallel"` to `static_argnames`, `:182`), resolved once with `resolve_parallel`. When resolved `True`:
  - the dynamic branch's two `_linear_sites` calls use `_linear_sites_parallel`: the one inside `newton_target` (`:287`, which serves both the stopped iteration and the exact Newton map) and the final Gaussian pass (`:331`);
  - the prior path (`:291-296`) and the static loadings (`:351-359`) come from `_transition_powers`.
  - When `False`, the code is byte-for-byte today's path.

  Update the docstring: Parameters entry, the CPU/GPU trade-off with measured numbers, the memory scaling, and the reference.

- **Model wiring in `graph_place_field.py`.**
  - `GraphPlaceFieldModel.__init__` gains `parallel: bool | None = None` and `neuron_batch_size: int | None = None`. Validate `parallel` as `None` or `bool`, and `neuron_batch_size` as `None` or `validate_int(..., positive=True)`. Store both as given, plus a `_parallel` property returning `resolve_parallel(self.parallel)`.
  - Add `"parallel"` and `"neuron_batch_size"` to `_GRAPH_LOSS_STATIC_ARGS` (`:1545`) and to `_graph_posterior`'s `static_argnames`. Thread them through `_graph_neuron_log_evidence` → `poisson_laplace_smoother(..., parallel=...)` (`:1510`) and `_graph_posterior` → `:1708`.
  - The model methods `_forward_loss`, `_profile_losses` and `_e_step` pass `self._parallel` and `self.neuron_batch_size`.
  - `_vmap_by_noise_structure` gains `batch_size: int | None`. When set, each noise group runs `jax.lax.map(lambda args: fn(flag, *args), group_args, batch_size=batch_size)` instead of `jax.vmap`, as in [designs.md](designs.md#parallel-pass-for-the-joint-laplace-core).
  - The sequential `inference_method` ignores `parallel`. The filter has no joint linear pass, so it does not need resolving there.

- **Rematerialisation experiment.** On a GPU, measure the `jax.checkpoint` variant against plain parallel at `T=70930`, `N ∈ {1, 4}` (peak bytes, value+grad min time). Apply the keep/drop rule in [designs.md](designs.md#parallel-pass-for-the-joint-laplace-core) and record both numbers in this file under "Results". If kept, it is unconditional under `parallel=True`; there is no new keyword.

- **Comparison (after the code change).** Re-run the benchmark grid with `--parallel true` (and `--fit` at `T=3000, N=8`) on the same devices. Add `scripts/benchmarks/compare_joint_laplace_outputs.py`. It loads a baseline/comparison `.npz` pair and asserts:
  - evidence rtol `1e-10`;
  - smoothed means rtol `1e-8` (atol `1e-10`);
  - with `--fit`, final log-likelihood abs `1e-6` and drift scales rtol `1e-5`.

  It prints `T | N | device | seq value+grad | par value+grad | speedup | seq peak GiB | par peak GiB`. Paste the table under a new "Results" heading in this file with device, `jax.__version__` and date. CPU slowdowns are reported as measured. If the lab's A100 harness is available (overview.md#prototype-evidence), also re-run its real-data fit with `parallel=True` and record the wall time next to the sequential 3,422 s.

- **Public documentation (ships with this phase).**
  - `poisson_laplace_smoother` docstring: as above.
  - `GraphPlaceFieldModel` docstring: `parallel` (backend default, CPU slower) and `neuron_batch_size`. Give the memory rule of thumb: parallel value+grad peaks at ~11.6 GiB per neuron at `T≈71k`, rank 20, scaling with `T · rank²`.
  - `docs/graph-place-fields.md`: a short "GPU" subsection covering when to set `parallel` and `neuron_batch_size`, and that a GPU is only faster with `parallel=True`.
  - `CHANGELOG.md`: extend the existing "Drifting graph-GP place fields" Added entry (the model is new on this branch).

## Deliberately not in this phase

- **`parallel` on `stochastic_point_process_smoother`, the point-process models, and the public `kalman.parallel_kalman_filter` wrapper:** phase 2b, which reuses this phase's filter.
- **Time-chunked (blocked) associative scans for a single neuron that exceeds device memory.** Trigger: one neuron OOMs at the user's `T`. From the measured 11.6 GiB per neuron at `T≈71k`, that's about `T ≳ 4.5e5` at rank 20 on an 80 GB card.
- **Choosing `neuron_batch_size` automatically from device memory.** Trigger: users repeatedly hitting OOM with the default `None`.
- **Batching the drift-profile refinement across neurons, or loosening its `xatol`:** a separate decision. Trigger: after this phase, profiles still exceed 50 % of a GPU `fit()`.
- **Float32 parallel passes:** the float64 requirement stands (designs.md#numerical-notes).
- **Removing the sequential pass:** it remains the CPU path, so no code is retired.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_parallel_information_filter_matches_sequential_recursion` (`tests/test_kalman.py`) | Against a Python loop of the information update with the first site conditioned directly on the prior: filtered means and covariances rtol `1e-10`. Covers full-rank and rank-one `G_t`, zero-information rows, non-identity `A`, and `T ∈ {1, 2, 7, 64}` (scan edges). |
| `test_parallel_information_filter_zero_information_is_prediction` | `G = g = 0` gives `m_t = A^t m⁻` and `P_t = A P_{t-1} Aᵀ + Q` (rtol `1e-12`). |
| `test_parallel_linear_pass_matches_sequential` (`tests/test_laplace_smoothing.py`) | Every `LinearSiteResult` field of `_linear_sites_parallel` equals `_linear_sites` to rtol `1e-10`, with masked rows and non-identity `A`. Guard: some rows have `weight == 0`. |
| `test_transition_powers_match_sequential_scan` | `_transition_powers(A, T)` equals the sequential product scan, rtol `1e-12`, `T ∈ {1, 2, 9}`, non-identity `A`. |
| `test_joint_mode_curvature_lag_and_evidence_match_dense_reference` (existing, parametrised over `parallel ∈ {False, True}`) | Unchanged tolerances against `joint_poisson_laplace` for `q ∈ {0.0, 0.03, 1e-8}`. |
| `test_parallel_smoother_matches_sequential_with_gradients` — **slow** | `joint_problem`, `q ∈ {0.0, 0.03}`: mode, covariances, cross-covariances rtol `1e-9`; evidence rtol `1e-10`; equal `n_iterations`, `n_rejected_steps`, `converged`. `jax.grad` of the evidence w.r.t. `q` and a `mean0` shift agrees to rtol `1e-8`. |
| `test_parallel_default_resolves_by_backend` | `resolve_parallel(None)` is `True` with `jax.default_backend` monkeypatched to `"gpu"` and `False` for `"cpu"`; explicit `True`/`False` are returned unchanged. |
| `test_parallel_and_hint_compose` | `static_noise=True` and `False` with `parallel=True` each equal the automatic branch with `parallel=True` (rtol `1e-12`). |
| `test_neuron_chunking_matches_unchunked` (`tests/test_graph_place_field_learning.py`) | For `N=5` neurons with mixed static/drifting, `neuron_batch_size=2` (uneven last chunk) gives the same `_graph_forward_loss` value and drift gradient (rtol `1e-12`) and the same `_graph_posterior` outputs as `None`. |
| `test_graph_parallel_fit_matches_sequential` — **slow (auto-marked)** | `GraphPlaceFieldModel(..., parallel=True)` and `parallel=False` on the scalar fixture: `fit()` log-likelihood abs `1e-6`, drift rtol `1e-5`, both `converged_`. |
| `test_model_rejects_bad_parallel_settings` | `parallel="yes"` and `neuron_batch_size ∈ {0, -1, 1.5, True}` raise `ValueError`. |
| Benchmark comparison scripts | `compare_joint_laplace_outputs.py` passes for every grid point run; the table is pasted under "Results". |
| Existing gates | `tests/test_laplace_smoothing.py`, all graph test files, `tests/test_kalman.py`, `tests/test_fit_contract.py`, `uv run mypy` pass unchanged. On CPU the default resolves to the sequential path. |

## Fixtures

- `joint_problem` (`tests/test_laplace_smoothing.py:17`) and the dense oracle `joint_poisson_laplace` (`tests/graph_math_reference.py:307`).
- `kalman_model_params` (`tests/conftest.py`) and `random_stable_matrix` / `random_spd_matrix` (`tests/oracles.py`) for the filter tests.
- `learning_env` / `scalar_data` (`tests/test_graph_place_field_learning.py`) for the model tests.
- Benchmark data are simulated by the script (seed 0); JSON/NPZ committed under `docs/plans/iterated-parallel-laplace-smoother/benchmarks/`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Additionally for this phase:
  - The baseline JSON predates the source changes in git history.
  - With `parallel` resolved to `False`, `poisson_laplace_smoother` runs exactly today's code.
  - The results table reports measured numbers, including CPU slowdowns, with device and JAX version.
  - The remat decision cites its measurements.

## Results

Measured 2026-10-09 on `breeze` (NVIDIA A100 80GB PCIe; CPU rows on the same
host with `OPENBLAS_NUM_THREADS=32`), JAX 0.10.2, x64, simulated data from
`joint_laplace_parallel_benchmark.py` (seed 0, rank 20, `dt = 0.05`). The
sequential baseline ran from a worktree pinned at `22bcb2a` (benchmark script
only, before any source change); the parallel rows ran from `9dd8bc9`, the
chunked rows from `9de8d86`. Other users' jobs shared the host for part of the
run, so the timings carry some extra noise. JSON records are in
[benchmarks/joint-laplace-2a/](benchmarks/joint-laplace-2a/). The NPZ outputs
stay on `breeze` under `/cumulus/edeno/ssp-2a-bench/`.

`compare_joint_laplace_outputs.py` passes for every pair: evidence rtol `1e-10`,
smoothed means rtol `1e-8`, equal convergence, and for the fit a log-likelihood
within `1e-6` and drift rtol `1e-3`. Timings are value+grad min.

| T | N | device | sequential s | parallel s | speedup | seq peak GiB | par peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3,000 | 1 | A100 | 3.577 | 0.073 | 49.3× | 0.27 | 0.76 |
| 3,000 | 8 | A100 | 4.647 | 0.269 | 17.3× | 1.42 | 5.68 |
| 3,000 | 32 | A100 | 4.939 | 0.922 | 5.4× | 5.44 | 22.83 |
| 12,000 | 1 | A100 | 14.349 | 0.153 | 94.1× | 0.74 | 2.92 |
| 12,000 | 8 | A100 | 20.733 | 1.018 | 20.4× | 5.46 | 22.98 |
| 12,000 | 32 | A100 | 21.695 | OOM (71.7 GiB) / 7.108 chunked by 8 | 3.1× chunked | 21.78 | 26.10 chunked |
| 70,930 | 1 | A100 | 84.189 | 0.653 | 128.9× | 4.29 | 16.60 |
| 70,930 | 8 | A100 | 134.339 | OOM (114 GiB) / 11.000 chunked by 2 | 12.2× chunked | 32.25 | 38.19 chunked |
| 3,000 | 1 | CPU | 1.474 | 4.158 | 0.4× | — | — |
| 3,000 | 8 | CPU | 7.863 | 22.239 | 0.4× | — | — |
| 12,000 | 1 | CPU | 7.962 | 13.111 | 0.6× | — | — |
| 12,000 | 8 | CPU | 39.716 | 90.674 | 0.4× | — | — |

**End-to-end `fit()`** (T=3000, N=8, `profile_points=9, profile_rounds=2`, A100):
2,065 s sequential → 165 s parallel (12.5×). Log-likelihood −7559.009301239684 vs
−7559.009301239671, both converged.

**Rematerialisation** (T=70,930, value+grad, A100):

| N | variant | peak | time |
| --- | --- | --- | --- |
| 1 | plain | 16.13 GiB | 0.653 s |
| 1 | `jax.checkpoint` on the linear pass | 9.70 GiB | 0.771 s |
| 4 | plain | OOM (61.4 GiB) | — |
| 4 | `jax.checkpoint` on the linear pass | 39.71 GiB | 3.60 s |

The memory drop is 1.66× at +18 % time, which fails the keep rule (≥ 2× at
≤ 30 %), so whole-pass rematerialisation was dropped. Neuron chunking bounds
memory instead.

**Chunking needed checkpointing.** `lax.map` alone did not bound memory under
differentiation: a chunked value+grad (T=12,000, N=32, chunks of 8) still tried
to allocate 101.7 GiB. Reverse mode through the map keeps every chunk's
intermediates. Each chunk is now `jax.checkpoint`-ed, which produced the
chunked rows above.
