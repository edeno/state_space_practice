# Phase 1 — Sequential iterated Laplace smoother (`n_iterations`) on the dense and block paths, threaded through `PointProcessModel` and `PlaceFieldModel`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#gauss-newton-step-as-a-linear-smoother-pass)

**Inputs to read first:**

- [src/state_space_practice/point_process_kalman.py:927-1212](../../../src/state_space_practice/point_process_kalman.py) — `_point_process_laplace_update`: the information-form update (`:1090-1091`, `:1138-1139`) and Laplace normaliser (`:1192-1208`) the new `_linearised_measurement_update` mirrors.
- [src/state_space_practice/point_process_kalman.py:1215-1265](../../../src/state_space_practice/point_process_kalman.py) — `GLMFamily`, `poisson_family`: the family the sites are built from.
- [src/state_space_practice/point_process_kalman.py:2392-2602](../../../src/state_space_practice/point_process_kalman.py) — `stochastic_point_process_smoother`: the entry point; `:2526-2569` block dispatch, `:2571-2602` dense path.
- [src/state_space_practice/point_process_kalman.py:2037-2122](../../../src/state_space_practice/point_process_kalman.py) and [:2269-2389](../../../src/state_space_practice/point_process_kalman.py) — the block smoother core and wrapper (per-neuron layout, `_concatenate_neuron_means` / `_package_block_covs` at `:2137-2149`).
- [src/state_space_practice/point_process_kalman.py:736-856](../../../src/state_space_practice/point_process_kalman.py) — `_fisher_scoring_line_search`: the "keep the iterate when no step qualifies" policy and the `_ARMIJO_C` constant (`:727`).
- [src/state_space_practice/kalman.py:778-842](../../../src/state_space_practice/kalman.py) — `rts_backward_scan` (the sequential backward pass to reuse); [:1251-1278](../../../src/state_space_practice/kalman.py) `InitialStatePrior` (the `x_0` prediction convention).
- [src/state_space_practice/temporal_rate_gp.py:236-395](../../../src/state_space_practice/temporal_rate_gp.py) — the 1-D iterated Laplace smoother this generalises: `_newton_step` (`:288-349`, line search on the segment), the evidence (`:357-386`), `_warn_laplace_diagnostics` (`:149-187`).
- [src/state_space_practice/multinomial_choice.py:45-56](../../../src/state_space_practice/multinomial_choice.py) and [:225-250](../../../src/state_space_practice/multinomial_choice.py) — the other Armijo implementation (slack rationale at `:226-233`).
- [src/state_space_practice/point_process_kalman.py:2993-3010, 3065-3105, 3314-3336, 3348-3371](../../../src/state_space_practice/point_process_kalman.py) — `PointProcessModel` constructor, `_e_step`, `_sgd_loss_fn`, `_finalize_sgd`.
- [src/state_space_practice/place_field_model.py:361-374, 983-1024, 1335-1348, 1574-1609, 1623-1658](../../../src/state_space_practice/place_field_model.py) — `PlaceFieldModel` constructor, `_e_step`, `_capture_state`, `_sgd_loss_fn`, `_finalize_sgd`; `score` at `:1886` (filter call `:1961-1978`).
- [src/state_space_practice/tests/test_oracle_point_process.py](../../../src/state_space_practice/tests/test_oracle_point_process.py) — `_affine_log_rate` (`:76-78`), `_simulate_problem` (`:91-121`), `_run_laplace` (`:124-146`), `_grid_posterior` (`:191-240`), `_standardized_errors` (`:243-253`), `_NEAR_GAUSSIAN` / `_SEEDS` / `_METRICS` (`:431-440`), `_regime_errors` (`:448-458`), `TestLaplaceVsExactPosterior` (`:472-536`).
- [src/state_space_practice/tests/test_approximation_trends.py:63-99](../../../src/state_space_practice/tests/test_approximation_trends.py) — `_assert_decreasing` and the pinned point-process trend test.
- [src/state_space_practice/tests/test_oracle_gp.py:167-231, 294-311, 528-563](../../../src/state_space_practice/tests/test_oracle_gp.py) — `dense_laplace_lgcp`, `_full_covariance_from_smoother`, the dense-GP Laplace agreement test (rtol `1e-8` convention).
- [src/state_space_practice/tests/test_likelihood_identities.py:583-612](../../../src/state_space_practice/tests/test_likelihood_identities.py) — `_pp_problem`, `_laplace_terms`.
- [src/state_space_practice/tests/test_point_process_kalman.py:3630-3700](../../../src/state_space_practice/tests/test_point_process_kalman.py) — `TestBlockDiagonalSmootherEquivalence._make_problem` (block/dense parity problem construction).
- [src/state_space_practice/tests/conftest.py:47-149](../../../src/state_space_practice/tests/conftest.py) — automatic `slow` marking of tests that call `.fit(` / `.fit_sgd(`.

**Contracts referenced:**

- [Public keywords and return layout](shared-contracts.md#public-keywords) — `n_iterations`, `convergence_tol`, `return_diagnostics`; evidence and `return_filtered` semantics. Do not weaken the bit-identical guarantee.
- [`IteratedSmootherDiagnostics`](shared-contracts.md#iterated-smoother-diagnostics) — defined in this phase.
- [Pseudo-observation site contract](shared-contracts.md#pseudo-observation-site-contract) — information form only; masks compose as `G_t = g_t = 0`.
- [Gauss–Newton core interface](shared-contracts.md#gauss-newton-core-interface) and [Linear-pass interface](shared-contracts.md#linear-pass-interface) — defined in this phase with `parallel=False`; `parallel=True` must raise `NotImplementedError` inside the private core until phase 2 (the public keyword is not added in this phase).
- [Warning contract](shared-contracts.md#warning-contract).

**Designs referenced:** [designs.md#pseudo-observation-sites](designs.md#pseudo-observation-sites), [#gauss-newton-step-as-a-linear-smoother-pass](designs.md#gauss-newton-step-as-a-linear-smoother-pass), [#line-search](designs.md#line-search), [#laplace-evidence](designs.md#laplace-evidence), [#convergence-and-diagnostics](designs.md#convergence-and-diagnostics), [#block-diagonal-path](designs.md#block-diagonal-path), [#gradients](designs.md#gradients), [#numerical-notes](designs.md#numerical-notes).

## Tasks

- **Prior quadratic-form helpers in `kalman.py`.** Add `markov_prior_residuals` and `markov_prior_quadratic_form` exactly as in [designs.md#line-search](designs.md#line-search) (public names, NumPy docstrings with shapes), next to `rts_backward_scan`. Unit-test against a dense `(T d) × (T d)` prior precision built in NumPy (`x_1 ~ N(A m_0, A P_0 Aᵀ + Q)`, innovations `N(0, Q)`): `markov_prior_quadratic_form(res, res, ...)` equals `(X − μ)ᵀ Σ⁻¹ (X − μ)` to rtol `1e-10`. Leave `temporal_rate_gp._prior_whitened_residuals` in place (phase 4 removes it).

- **Sites, linearised update and sequential linear pass in `point_process_kalman.py`.** New section "Iterated Gauss–Newton (Laplace) smoother" after `_stochastic_point_process_smoother_backward` (`:2605-2641`): `_pseudo_observation_sites`, `_information_site`, `_site_log_value`, `_linearised_measurement_update`, `_linearised_forward_scan`, `_LinearPassOutput`, `_LinearisationProblem` and `_linearised_smoother_pass(problem, path, *, parallel)` (sequential branch = forward scan + `kalman.rts_backward_scan`; `parallel=True` raises `NotImplementedError` here). Code is in [designs.md#gauss-newton-step-as-a-linear-smoother-pass](designs.md#gauss-newton-step-as-a-linear-smoother-pass). `grad_log_intensity` for the default linear intensity is the analytic `Z_t` (as at `:1926-1933`); otherwise `jax.jacfwd(log_intensity, argnums=1)` hoisted outside the scan (as at `:1798`). Use `_common_float_dtype` (`:512-520`) for the carry dtype like the existing scan.

- **Line search and core.** `_MERIT_RTOL = 1e-10`, `_N_BACKTRACK = 12` module constants (reuse `_ARMIJO_C`); `_MarkovPrior` NamedTuple (`first_mean`, `first_cho`, `process_cho`); `_gauss_newton_line_search` and `_iterated_laplace_smoother_core` as in [designs.md#line-search](designs.md#line-search) and [#convergence-and-diagnostics](designs.md#convergence-and-diagnostics); `IteratedSmootherDiagnostics` as in the contract; `_warn_iterated_smoother_diagnostics(diagnostics, name, convergence_tol, n_iterations, stacklevel)` following `temporal_rate_gp._warn_laplace_diagnostics` (`:149-187`) and the [warning contract](shared-contracts.md#warning-contract). The path log-likelihood `psi_log_lik(path)` is `Σ_t family.loglik_plugin(y_t, η_t, μ_t)` with `η_t = log_intensity(Z_t, x_t)` vmapped over time (plus `Σ_t log_prior_extra` when `extra_sites` is set — keep the hook, phase 3 fills it).

- **Wire the dense path of `stochastic_point_process_smoother`.** Add the keyword-only parameters `n_iterations: int = 1`, `convergence_tol: float = 1e-6`, `return_diagnostics: bool = False` after `return_block_covariances` (`:2409`). Validate (`validate_int(..., positive=True)`, `validate_scalar(..., positive=True)`). If `n_iterations == 1`: run the existing code unchanged (`:2571-2602`); if `return_diagnostics`, append the stub described in overview.md Open Question 4. Otherwise: run the existing dense smoother with `return_filtered=True` (iteration 0), build `_LinearisationProblem` with `family=family if family is not None else poisson_family(dt, max_log_count)` (preserve the supplied family in the initial pass, all iterated passes and the merit/evidence; keep the GLM plan's dense dispatch and family-specific observation validation) and `counts=spike_indicator.astype(float)` (promote 1-D spikes to `(T, 1)` as `:1708-1710` does), call the core with `initial_path=smoother_mean`, `n_gauss_newton_passes=n_iterations - 1`, `parallel=False`, warn via `_warn_iterated_smoother_diagnostics`, and return `(path, final.smoother_cov, final.smoother_cross_cov, evidence[, filtered_mean_0, filtered_cov_0][, diagnostics])` where `evidence = final.evidence if include_laplace_normalization else final.plugin_log_likelihood`. Put the iterated part in a jitted `_iterated_smoother_dense_impl` with the resolved `family`, `log_conditional_intensity`, `n_gauss_newton_passes`, `include_laplace_normalization` static (same pattern as `_stochastic_point_process_filter_impl`, `:1737-1758`).

- **Wire the block path.** In `_stochastic_point_process_smoother_block_diagonal` (`:2269-2389`) accept the same three keywords; after `_block_diagonal_smoother_core` (`:2344-2363`) and when `n_iterations > 1`, build per-neuron `_LinearisationProblem`s (`design_matrix=Z_base[:, None, :]`, `counts=spikes[:, j][:, None]`, `A_blocks[j]`, `Q_blocks[j]`, `init_means_per_neuron[j]`, `init_covs_per_neuron[j]`, linear intensity, analytic gradient) and `jax.vmap` the core over neurons with `initial_path=smoother_means_per_neuron` (`(n_neurons, n_time, nb)`). Aggregate diagnostics per the contract. Package outputs with `_concatenate_neuron_means` / `_package_block_covs`; `filtered_*` stay the iteration-0 per-neuron filter output. Put it in a jitted `_block_diagonal_iterated_core` with static `n_gauss_newton_passes`, `include_laplace_normalization` (like `_block_diagonal_smoother_core`, `:2033-2036`).

- **`PointProcessModel` pass-through.** `__init__` (`:2993-3006`) gains `n_iterations: int = 1` (validated, stored) and `self.smoother_diagnostics_: IteratedSmootherDiagnostics | None = None`. `_e_step` (`:3089-3103`) passes `n_iterations=self.n_iterations, return_diagnostics=True` and stores the diagnostics; `_snapshot_state` (`:3188-3199`) / `_clear_posteriors` (`:3205-3213`) include `smoother_diagnostics_`. `_sgd_loss_fn` (`:3314-3336`): when `self.n_iterations > 1`, call `stochastic_point_process_smoother(..., validate_inputs=False, n_iterations=self.n_iterations)` and return `-marginal_ll` (unrolled evidence, [designs.md#gradients](designs.md#gradients)); otherwise unchanged. `_finalize_sgd` (`:3348-3371`) passes `n_iterations` and stores diagnostics.

- **`PlaceFieldModel` pass-through.** Same in `__init__` (`:361-374`, validate like `max_newton_iter`), `_e_step` (`:1002-1023`), `_capture_state` (`:1335-1348`) and `_clear_posteriors` (`:1354-1362`), `_sgd_loss_fn` (`:1586-1608`, with `block_n_neurons`/`block_size`/`return_block_covariances=True` unchanged) and `_finalize_sgd` (`:1631-1650`). `score` (`:1886`) is unchanged; add one docstring sentence: it is the one-pass filter's predictive log-likelihood even when the model was fitted with `n_iterations > 1`.

- **Public documentation (ships with this phase).**
  - `stochastic_point_process_smoother` docstring: new parameters; an "Iterated Laplace smoothing" paragraph stating what pass 1 is, that `n_iterations=1` is today's smoother, the evidence semantics, the `return_filtered` semantics, and the warning behaviour; references Bell 1994, García-Fernández et al. 2017, Paninski et al. 2010 (from [designs.md#references](designs.md#references)).
  - `PointProcessModel` and `PlaceFieldModel` class docstrings: `n_iterations` parameter and `smoother_diagnostics_` attribute.
  - `CHANGELOG.md` under `## [Unreleased]` → `### Added` (`CHANGELOG.md:6-8`): one entry for the smoother keywords and `IteratedSmootherDiagnostics`, one for the model constructor argument, one for `kalman.markov_prior_residuals` / `markov_prior_quadratic_form`; note explicitly that the returned `marginal_log_likelihood` is the joint Laplace evidence when `n_iterations > 1`.
  - `README.md`: a short "Iterated Laplace smoothing" subsection after "Package layout" (`README.md:60-66`) with the two-line usage `PlaceFieldModel(dt=..., n_iterations=5)` and the float64 reminder.

- **Tests** (new module `tests/test_iterated_laplace_smoother.py` plus additions listed in the validation slice). Add an `n_iterations` keyword to `_run_laplace` in `tests/test_oracle_point_process.py:124-146` (forwarded to the smoother) and a NumPy helper `_joint_log_posterior_and_hessian(problem, path)` returning `Ψ(X)`, `∇Ψ(X)` and the dense block-tridiagonal `H = −∇²Ψ` for the affine model (reusing `_log_evidence_terms`, `:261-267`).

## Deliberately not in this phase

- `parallel=True` and `kalman.parallel_kalman_filter` — phase 2 (the private core raises `NotImplementedError` for `parallel=True`; the public keyword does not exist yet).
- `PositionDecoder` / `position_decoder_smoother` — phase 3 (different filter; needs the penalty sites; `extra_sites` hook is left `None` here).
- Replacing `temporal_rate_gp`'s Newton loop — phase 4 (this phase only proves agreement with it).
- Implicit (fixed-point) differentiation of the evidence. Trigger: a `fit_sgd` run with `n_iterations ≥ 4` whose reverse-mode memory exceeds the device, or a measured > 2× slowdown of the SGD step versus `n_iterations=1` on the phase-2 baseline grid.
- Early exit via `lax.while_loop`; exposing `initial_path` (EM warm start) — overview.md Open Questions 1–2.
- Any change to `stochastic_point_process_filter`, `max_newton_iter`, the M-steps or `run_em`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_iterated_noncanonical_family_score[nb,zig]` — **slow** | Use the public smoother with a supplied family and `n_iterations > 1`. At convergence, compare the path to an independent dense MAP solve and the exact joint-objective gradient; for NB include `y=5, r=1, eta=x`, prior `N(0,1)` whose mode is about `0.826418`. For ZIG use at least two observations and four predictors. Assert every pass and line-search objective uses the supplied family. |
| `test_default_call_equals_explicit_single_iteration` (dense, block, `return_filtered` on/off) | Every output of the unchanged call and of `n_iterations=1` is `assert_array_equal`-identical; `return_diagnostics=True` at `n_iterations=1` returns the stub (`n_gauss_newton_passes == 0`, `isnan(max_abs_update)`, `not converged`). |
| `test_markov_prior_quadratic_form_matches_dense_prior_precision` (`tests/test_kalman.py`) | `markov_prior_quadratic_form` equals the dense `(X−μ)ᵀΣ⁻¹(X−μ)` to rtol `1e-10` on random `(A, Q, m_0, P_0, X)`; guard `q > 1`. |
| `test_one_gauss_newton_pass_equals_dense_newton_step` (oracle, `_pp_problem`-sized affine model, `n_iterations=2`, `convergence_tol=inf`) | The path after one pass equals `X̂ + H⁻¹∇Ψ(X̂)` from the NumPy block-tridiagonal Hessian to rtol `1e-8`, with `X̂` the one-pass path; guard `‖H⁻¹∇Ψ‖∞ > 1e-3`; `diagnostics.step_sizes == [1.0]`. |
| `test_joint_log_posterior_is_monotone_and_converges` (`n_iterations=10`) | `diagnostics.log_posterior` non-decreasing (`diff >= -1e-10·(1+abs(Ψ))`), strictly increasing at the first pass by > 1e-3 nats (guard), `converged` is `True`, `max_abs_update < 1e-8·(1+‖X‖∞)`, no warning raised. |
| `test_converged_path_is_stationary_point` | NumPy `‖∇Ψ(X̂)‖∞ < 1e-8 · ‖∇Ψ(X^{(1)})‖∞`; guard `‖∇Ψ(X^{(1)})‖∞ > 1e-2`. |
| `test_laplace_covariance_is_inverse_joint_hessian` | Full covariance rebuilt from `smoother_cov` + `smoother_cross_cov` with `_full_covariance_from_smoother` equals `inv(H)` at the converged path to rtol `1e-8`; guard: differs from the one-pass covariance by > 1 % somewhere. |
| `test_evidence_equals_dense_laplace_evidence` | `marginal_log_likelihood` equals `Ψ_full(X̂) + ½ logdet(2π H⁻¹)` (with the prior normalisers restored and `gammaln`) to rtol `1e-8`; `include_laplace_normalization=False` returns the plug-in Poisson log-pmf at `X̂` to rtol `1e-10`. |
| `test_iterated_smoother_matches_temporal_rate_gp` (Matérn-3/2, `matern32_discretize`, `Z_t = [1, 0]`, `log_conditional_intensity = lambda Z, x: Z @ x + mean`, `n_iterations=40` vs `infer_log_rate(n_iter=60)`) | Mode (`smoother_mean[:, 0]`), variance (`smoother_cov[:, 0, 0]`) and evidence agree to rtol `1e-8`; guard both `converged` / `max_abs_update < 1e-10`. |
| `test_zero_count_low_rate_bins_are_finite_and_uninformative` | A neuron with baseline `log(1e-12)` Hz and all-zero counts: all outputs finite, and equal to the run with that neuron removed to atol `1e-6` (its site weight is `exp(-20)`). |
| `test_block_path_matches_dense_when_no_backtracking` (problem from `TestBlockDiagonalSmootherEquivalence._make_problem`, `n_iterations=5`) | Means rtol `1e-9`, covariances (via `to_dense`) rtol `1e-9`, evidence rtol `1e-10`; guard `all(step_sizes == 1)` on both paths and `n_iterations` changed the path by > 1e-4. |
| `test_return_filtered_gives_initialisation_filter` (`n_iterations=4`) | `filtered_mean` / `filtered_cov` are `assert_array_equal` to the `n_iterations=1` filtered outputs. |
| `test_unconverged_iteration_warns` (`n_iterations=2`, prior mean far below the data) | `pytest.warns(StateSpaceWarning, match="did not converge")`; guard `not diagnostics.converged`; the same problem with `n_iterations=30` raises no warning. |
| `test_rejected_step_keeps_path_and_warns` (monkeypatch `psi_log_lik` to `-inf` for every trial, as `tests/test_temporal_rate_gp.py:269-283` does) | `step_sizes == 0`, path unchanged, `n_unaccepted_steps == 1`, `pytest.warns(StateSpaceWarning, match="Armijo")`. |
| `test_iterated_gap_to_quadrature_shrinks_with_iterations` (`tests/test_approximation_trends.py`, `_NEAR_GAUSSIAN`, seeds `_SEEDS`, `n_iterations ∈ {1, 2, 4, 8}`, mean over seeds) — **slow** | `_assert_decreasing` on the standardised `smooth_mean` error; at `n_iterations=8` the `smooth_var` and `log_lik` errors are ≤ their `n_iterations=1` values; nonzero-gap guard (`smooth_mean` error at `n=1` > 1e-3); observed values recorded in the docstring. |
| `test_evidence_gradient_matches_finite_difference` (`n_iterations=6`, `jax.grad` w.r.t. `diag(Q)` and `init_mean`) — **slow** | Central differences (`h = 1e-5`) agree with autodiff to rtol `1e-5`; guard `‖grad‖∞ > 1e-2`. |
| `test_point_process_model_fit_with_iterations` (`PointProcessModel(n_iterations=4).fit`) — **slow (auto-marked)** | `smoother_diagnostics_.converged`, `assert_ll_improves(log_likelihoods)` (`recovery_helpers.py:68`), snapshot/restore round-trips the diagnostics on a forced rollback (`assert_em_rolls_back_on_ll_decrease`, `conftest.py:680`). |
| `test_place_field_model_e_step_matches_direct_smoother_call` (block path, `n_iterations=3`) | `_e_step` outputs equal `stochastic_point_process_smoother(..., n_iterations=3, return_block_covariances=True)` outputs exactly; `smoother_diagnostics_` set. |
| `test_sgd_loss_uses_iterated_evidence` (`PointProcessModel`, `PlaceFieldModel`; `n_iterations=3`) | `_sgd_loss_fn(params, ...)` equals `-marginal_log_likelihood` of the direct smoother call; with `n_iterations=1` equals the filter's `-marginal_ll` exactly. |
| `test_gradients.py::test_point_process_model_loss` / `::test_place_field_model_loss` (existing) with a parametrised `n_iterations=3` variant — **slow** | Autodiff matches central differences under the module's existing tolerance. |
| Existing gates unchanged: `test_em_golden_regression.py::test_em_matches_recorded_values`, `test_oracle_point_process.py` (all), `test_likelihood_identities.py::TestPointProcessIdentities`, `test_point_process_kalman.py::TestBlockDiagonalSmootherEquivalence` | Pass without modification (defaults select today's path). |

Mark slow tests explicitly (`@pytest.mark.slow`); `conftest.py` auto-marks the `.fit(` /
`.fit_sgd(` ones.

## Fixtures

- Reuse `_simulate_problem` / `_pp_problem` / `_informative_problem` for affine-model
  problems (`tests/test_oracle_point_process.py:91-121, 539-559`,
  `tests/test_likelihood_identities.py:583`).
- New module-scoped fixture `iterated_problem` in `tests/test_iterated_laplace_smoother.py`:
  `_simulate_problem(n_latent=2, n_neurons=4, n_time=12, rate_hz=200.0, weight_scale=0.7, seed=5)`
  plus its one-pass and `n_iterations=10` smoother outputs (computed once).
- Matérn-3/2 counts for the `temporal_rate_gp` cross-check: `matern32_gram` sample as in
  `tests/test_oracle_gp.py:543-548` with `n_time=40, dt=0.05, variance=0.9, lengthscale=0.3, mean=log(25)`.
- Block-parity problem: `TestBlockDiagonalSmootherEquivalence._make_problem(3, 2, T=60)`
  (`tests/test_point_process_kalman.py:3642-3688`), refactored into a module-level helper if
  the new test module needs it.
- The hard (unconverged) problem: `_simulate_problem(1, 2, 6, rate_hz=3000.0, weight_scale=0.7, seed=1)`
  with `init_mean = -5` (prior far below the data).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Additionally for this phase: `git diff` of `stochastic_point_process_smoother`'s pre-existing body is limited to the keyword additions and the `if n_iterations == 1` guard (the bit-identical guarantee is structural, not just tested); `ruff check` / `ruff format` clean; `uv run mypy` still passes (the touched modules are not in its file list, but `temporal_rate_gp.py` is untouched).
