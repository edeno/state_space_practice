# Phase 4 — `temporal_rate_gp` runs on the shared Gauss–Newton core; its private Newton loop is removed

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#laplace-evidence)

**Inputs to read first:**

- [src/state_space_practice/temporal_rate_gp.py:76-115](../../../src/state_space_practice/temporal_rate_gp.py) — constants (`_DEFAULT_MIN_WEIGHT`, `_N_BACKTRACK`, `_MERIT_RTOL`) and `_prior_whitened_residuals`.
- [src/state_space_practice/temporal_rate_gp.py:118-233](../../../src/state_space_practice/temporal_rate_gp.py) — `LaplaceRateResult`, `_warn_laplace_diagnostics`, `poisson_log_rate_site` (public; stays).
- [src/state_space_practice/temporal_rate_gp.py:236-395](../../../src/state_space_practice/temporal_rate_gp.py) — `_infer_log_rate_traced`: `_newton_target` (`:272-286`), `_newton_step` (`:288-349`), the scan (`:351-355`), the evidence (`:357-386`). This is the code being replaced.
- [src/state_space_practice/temporal_rate_gp.py:398-602, 858-920](../../../src/state_space_practice/temporal_rate_gp.py) — `infer_log_rate`, `infer_log_rate_batch` (vmapped core), `TemporalRateGP._sgd_loss_fn` / `_finalize_sgd` (the traced hyperparameters).
- [src/state_space_practice/gp_ssm.py:44-181](../../../src/state_space_practice/gp_ssm.py) — `matern32_continuous`, `matern32_discretize` (stationary prior: `A Pinf Aᵀ + Q = Pinf`).
- Phase-1 code: `_LinearisationProblem`, `_iterated_laplace_smoother_core`, `IteratedSmootherDiagnostics`, `GLMFamily`.
- [src/state_space_practice/tests/test_oracle_gp.py:528-563](../../../src/state_space_practice/tests/test_oracle_gp.py) — the dense-GP Laplace gate (rtol `1e-8`); [:565-600](../../../src/state_space_practice/tests/test_oracle_gp.py) convergence-from-far-below and jitted-core equivalence tests; [:722](../../../src/state_space_practice/tests/test_oracle_gp.py) SGD stationary-point test.
- [src/state_space_practice/tests/test_temporal_rate_gp.py:189-300](../../../src/state_space_practice/tests/test_temporal_rate_gp.py) — convergence and warning tests, including the monkeypatched fallback tests (`:249-283`).
- [src/state_space_practice/tests/test_gradients.py:280](../../../src/state_space_practice/tests/test_gradients.py) — `test_temporal_rate_gp_loss`.
- `pyproject.toml:134-158` — `temporal_rate_gp.py` is in the mypy file list; it must stay type-clean.

**Contracts referenced:**

- [Gauss–Newton core interface](shared-contracts.md#gauss-newton-core-interface) — called un-jitted from a traced function; the family closes over the traced offset.
- [`IteratedSmootherDiagnostics`](shared-contracts.md#iterated-smoother-diagnostics) — mapped onto `LaplaceRateResult`.
- [Warning contract](shared-contracts.md#warning-contract) — `_warn_laplace_diagnostics` keeps its messages except for the recorded change below.

**Designs referenced:** [designs.md#pseudo-observation-sites](designs.md#pseudo-observation-sites), [#line-search](designs.md#line-search) (rejected-step policy), [#laplace-evidence](designs.md#laplace-evidence) (identity with the site form), [#numerical-notes](designs.md#numerical-notes).

## Tasks

- **A `GLMFamily` that reproduces `poisson_log_rate_site` semantics.** In `temporal_rate_gp.py` add `_log_rate_family(offset, min_weight) -> GLMFamily` with `mean = lambda eta: jnp.exp(eta + offset)` (no clipping, as today), `fisher_weight = lambda eta, mu: jnp.maximum(mu, min_weight)` (the floor of `poisson_log_rate_site`, `:230`), `loglik_plugin = lambda y, eta, mu: jnp.sum(y * (eta + offset) - mu)` (the merit at `:319-325`), `loglik_normalized = lambda y, eta, mu: jnp.sum(y * (eta + offset) - mu - gammaln(y + 1.0))` (`:376-378`). `offset = mean + log(dt)` may be a tracer; the family is a closure passed to the un-jitted core, never a static jit argument.

- **Rewrite `_infer_log_rate_traced` on the core.** Build `A, Q` (`matern32_discretize`), `Pinf` (`matern32_continuous`), `_LinearisationProblem(init_mean=zeros(2), init_cov=Pinf, transition_matrix=A, process_cov=Q, design_matrix=broadcast([1, 0], (n_time, 1, 2)), counts=counts[:, None], log_intensity=lambda Z, x: Z @ x, grad_log_intensity=lambda Z, _x: Z, family=_log_rate_family(offset, min_weight), include_laplace_normalization=True, extra_sites=None)`; call `_iterated_laplace_smoother_core(problem, initial_path=zeros((n_time, 2)), n_gauss_newton_passes=n_iter, convergence_tol=<unused for the result>, parallel=False)`. Map the outputs: `log_rate_mean = mean + path[:, 0]`, `log_rate_var = final.smoother_cov[:, 0, 0]`, `log_marginal_likelihood = final.evidence`, `max_abs_update = diagnostics.max_abs_update`, `n_nonfinite_merit`, `n_unaccepted_steps` from the diagnostics. Note the first-state prior: the core uses `x_1 ~ N(A·0, A Pinf Aᵀ + Q) = N(0, Pinf)` (stationarity), matching today's `stationary_cov` whitening (`:264`, `:112`).

- **Remove the replaced code.** Delete `_newton_target`, `_newton_step`, the local `backtrack_steps`, `_N_BACKTRACK`, `_MERIT_RTOL`, `_prior_whitened_residuals` and the `kalman_smoother` import (`:61`) if unused. Keep `poisson_log_rate_site` (public, tested at `tests/test_temporal_rate_gp.py:113`) and note in its docstring that inference now uses the information-form sites of `point_process_kalman` (same weight floor). Update the module docstring (`:16-27`) to point at the shared core.

- **Recorded behaviour changes** (document in CHANGELOG `### Changed — behavior`):
  1. When no Armijo step length passes, the iterate is kept (`α = 0`) instead of taking the smallest trial step. `LaplaceRateResult.n_unaccepted_steps` docstring (`:135-138`) and the warning text in `_warn_laplace_diagnostics` (`:178-187`, "smallest trial step taken") are updated to "path kept". Tests `test_unaccepted_step_fallback_is_counted_and_warns` (`tests/test_temporal_rate_gp.py:269-283`) and `test_nonfinite_merit_fallback_is_counted_and_warns` (`:249-267`) monkeypatch the merit; retarget the monkeypatch at the core's path log-likelihood and update the message match.
  2. `LaplaceRateResult.max_abs_update` is the infinity norm of the *un-taken* Gauss–Newton step at the returned mode (the fixed-point residual) rather than of the last taken step. The convergence test in `_warn_laplace_diagnostics` (`:160-175`) keeps its `sqrt(eps)(1 + max|mean|)` threshold; existing tests asserting `< 1e-10` after convergence and `> 1e-3` after `n_iter=1` (`:189-229`) still hold (the un-taken step is at most the last taken step near the mode, and large when unconverged) — verify, do not loosen.

- **Type cleanliness and docs.** `uv run mypy` must pass with `temporal_rate_gp.py` unchanged in the file list (the core's NamedTuples are typed; add a `# type: ignore[...]` only with a comment naming the reason). `CHANGELOG.md`: `### Changed` entry (consolidation + the two behaviour notes) and `### Removed` entry for the private helpers. `infer_log_rate` docstring (`:407-455`): the algorithm sentence now cites the shared core.

## Deliberately not in this phase

- Exposing `parallel=True` on `infer_log_rate` / `TemporalRateGP`. The core supports it after phase 2; the batch path `vmap`s over neurons and would need the `lax.map` treatment. Trigger: a user with `n_time > 1e5` single-train inference on GPU.
- Removing or deprecating `poisson_log_rate_site` (public, tested; unused internally after this phase). Trigger: the next public-API review.
- Any change to `TemporalRateGP`'s hyperparameter parameterisation or to `gp_ssm`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_oracle_gp.py::test_laplace_rate_gp_equals_dense_gp_laplace` (existing, Hypothesis sweep) — **slow** | Mode, Laplace variance and evidence match the dense GP-Laplace to rtol `1e-8` with the refactored core; the `max_abs_update < 1e-10` guard still holds. |
| `tests/test_oracle_gp.py::test_laplace_converges_when_prior_mean_is_far_below_data`, `::test_jitted_core_equals_public_infer_log_rate`, `::test_laplace_posterior_is_exact_gp_regression_on_its_gaussian_sites`, `::test_fit_sgd_reaches_stationary_point_of_dense_evidence` (existing) | Pass unchanged (the last is slow). |
| `tests/test_temporal_rate_gp.py` (existing suite) | Passes with the two monkeypatched fallback tests retargeted and the "path kept" message; `test_default_iterations_converge_when_baseline_is_far_below_data` (`:189-201`) still reaches `max_abs_update < 1e-10`. |
| `tests/test_gradients.py::test_temporal_rate_gp_loss` (existing, shared/per-neuron) — **slow** | Autodiff of the evidence w.r.t. the hyperparameters matches central differences under the module's tolerance (exercises tracer-carrying closures through the core). |
| `test_infer_log_rate_batch_equals_per_train_calls` (new) | `infer_log_rate_batch` on 3 trains equals three `infer_log_rate` calls exactly (the vmapped core). |
| `test_iterated_smoother_matches_temporal_rate_gp` (phase 1) | Still passes — now both sides run the same core, so tighten to rtol `1e-12` and keep it as the regression that the two entry points stay wired to one implementation. |
| `grep -rn -e _newton_step -e _newton_target -e _prior_whitened_residuals src/` | No hits (removed code is gone). |
| `uv run mypy` | Clean. |

## Fixtures

- `small_counts` (`tests/test_temporal_rate_gp.py`) and the Hypothesis strategies of `tests/test_oracle_gp.py` — unchanged.
- No new data; the new batch-equivalence test reuses `small_counts` stacked three times with different means.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Additionally for this phase: the dense-GP oracle sweep was run under `HYPOTHESIS_PROFILE=ci`; the CHANGELOG records both behaviour changes; `temporal_rate_gp.py` remains in the mypy file list and passes.
