# Phase 2b — EM for the process and cycle-offset variances, and the cycle-to-cycle sampling test

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#hyperparameter-m-steps)

Replaces hand-chosen hyperparameters with exact EM: the coefficient process variance `q` (isotropic random walk) and, with the offset state, the cycle-offset variance `sigma_c^2`, through the shared `run_em` loop. Adds `cycle_variance_test`, the likelihood-ratio comparison of `sigma_c^2 = 0` (the phase 1 model) against `sigma_c^2` free, with the boundary-corrected asymptotic p-value — the Ujfalussy & Orbán "sampling" question as a model comparison.

**Inputs to read first:**

- `src/state_space_practice/theta_sweep.py` (phases 1–2a) — `_fit_bouts` (per-bout posteriors), `theta_sweep_smoother`, `ThetaSweepResult`.
- `src/state_space_practice/em_driver.py:35-56` (`EMResult`) and `:58` onward (`run_em`: hooks, `on_first_nonfinite="clear"`, `snapshot` must return values later steps cannot mutate).
- `src/state_space_practice/point_process_kalman.py:3188-3232` — `PointProcessModel.fit`: snapshot / restore / clear hooks and the `run_em` call to mirror.
- `src/state_space_practice/place_field_model.py:1026-1138` — `PlaceFieldModel._m_step`: exact EM with the `x_0 -> x_1` transition (`:1076-1098`) and the isotropic constraint (`:1120-1127`).
- `src/state_space_practice/kalman.py:1251-1278` (`InitialStatePrior`), `:1320-1361` (`smooth_initial_state_with_cross_cov`), `:1399-1445` (`process_cov_residual_form`, divides by `n_time - 1` transitions of its input).
- `src/state_space_practice/tests/test_oracle_point_process.py:607-681` — the Q-function / finite-difference machinery (`_expected_log_normal`, `_q_function`, `_fd_gradient`) whose pattern the M-step exactness test follows; `:711-773` an example of asserting a vanishing gradient at the update.
- `src/state_space_practice/tests/recovery_helpers.py:68-92` — `assert_ll_improves`, `assert_ll_monotonic`.
- `src/state_space_practice/tests/test_em_driver.py` — how `run_em` behaviour is tested (rollback warnings are logged, not raised).

**Contracts referenced:**

- [ThetaSweepModel](shared-contracts.md#thetasweepmodel) — `fit(..., fit_hyperparameters, max_iter, tolerance)`; `phase_shuffle_scores` never runs EM.
- [ThetaSweepResult](shared-contracts.md#thetasweepresult) — fills `process_variance`, `cycle_offset_variance`, `em_log_likelihoods`.
- [Design-matrix layout](shared-contracts.md#design-matrix-layout) — coefficient block `[:K]`, offset `[-1]`.

**Designs referenced:** [hyperparameter M-steps](designs.md#hyperparameter-m-steps), [cycle-variance test](designs.md#cycle-variance-test).

## Tasks

- **M-steps.** Add `coefficient_process_variance_update` and `cycle_offset_variance_update` to `theta_sweep.py` exactly as in [designs](designs.md#hyperparameter-m-steps), with docstrings stating the expected complete-data log-likelihood they maximise and that the `x_0 -> x_1` transition is included (block-diagonality argument for the offset model).
- **EM in `fit`.** Add `fit_hyperparameters: bool = False`, `max_iter: int = 50`, `tolerance: float = 1e-4` to `ThetaSweepModel.fit`. With `fit_hyperparameters=True`, run the `run_em` loop from [designs](designs.md#hyperparameter-m-steps): the E-step is `_fit_bouts` at the current `(process_variance, cycle_offset_variance)`, the M-step installs the two updates (`sigma_c^2` floored at `1e-8`), snapshots hold the scalars and the per-bout posteriors, `on_first_nonfinite="clear"`. After EM, `_assemble_result` runs on the accepted posteriors and the result carries `process_variance`, `cycle_offset_variance` (`None` when `cycle_offset=False`) and `em_log_likelihoods`. The model's `process_variance` / `cycle_offset_variance` attributes hold the fitted values afterwards (so `phase_shuffle_scores` uses them without refitting). With `fit_hyperparameters=False` behaviour is unchanged and the new result fields stay `None`.
- **Likelihood-ratio test.** Add `CycleVarianceTest` and `cycle_variance_test` as in [designs](designs.md#cycle-variance-test) (asymptotic boundary-mixture p-value; `ValueError` on mismatched inputs). Export both in `__all__` of the module (not the package).
- **Smoke script.** Extend `scripts/theta_sweep_smoke.py` with `--fit-hyperparameters`, `--cycle-offset` and `--bootstrap B`: fit the null (no offset, EM on `q`) and the alternative (offset, EM on both), print `cycle_variance_test` and, with `B > 0`, the plug-in parametric bootstrap p-value described in [designs](designs.md#cycle-variance-test) (spikes re-drawn from the null fit's smoothed represented position on the running bins; both models refit per replicate; `p_boot = mean(LR* >= LR)`).
- **Docs.** `CHANGELOG.md` `### Added`: EM for the two variances and `cycle_variance_test` (with the Self & Liang 1987 boundary mixture named). `ThetaSweepModel.fit` docstring: what is fitted, that the initial prior is not, and that EM log-likelihood decreases roll back (behaviour inherited from `run_em`). `cycle_variance_test` docstring: null/alternative definitions, the mixture distribution, the caveat that the Laplace log-likelihood is approximate and the bootstrap in the script.

## Deliberately not in this phase

- Updating the initial prior `(0, init_std^2 I)` by EM — deliberately fixed (shared across bouts, weakly informative); revisit if the smoke script shows the first seconds of bouts dominated by the prior at the fitted `q`.
- SGD / gradient fitting of hyperparameters (`SGDFittableMixin`) — two scalars with closed-form updates do not need it.
- Library-level parametric bootstrap — script only ([overview open question 6](overview.md#open-questions)).
- Per-direction or per-bout hyperparameters.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_theta_sweep.py::test_process_variance_update_is_stationary_point` | For random smoothed moments of a 2-bout problem (`_random_smoother_moments`-style, `K = 2`) and a random prior, the expected complete-data log-likelihood `Q(q) = sum_t E[log N(u_t; u_{t-1}, q dt I)]` written out in the test (with the smoothed `x_0` prepended via `smooth_initial_state_with_cross_cov`) has a central-difference derivative `< 1e-6 * \|dQ/dq at q_old\|` at `coefficient_process_variance_update`'s output, `\|dQ/dq at q_old\| > 1e-2` (guard), and `Q(q_new) >= Q(q_old)`. |
| `::test_cycle_offset_variance_update_is_stationary_point` | Same for `Q(sigma_c^2) = sum_{starts} E[log N(epsilon_t; 0, sigma_c^2)]` and `cycle_offset_variance_update`; bins that are not cycle starts do not enter (changing their moments leaves the update unchanged). |
| `::test_cycle_variance_test_uses_boundary_mixture` | `statistic <= 0 -> p = 1`; `LL_alt - LL_null = 1.92` gives `statistic = 3.84`, `p ~ 0.025` (`0.5 * chi2.sf(3.84, 1)`); a null result carrying `cycle_offset_mean`, an alternative without `cycle_offset_variance`, or differing `running_mask`s raise `ValueError`. |
| `::test_em_log_likelihood_is_monotone_and_recovers_process_variance` (slow) | 30 s session, `a` drifting `5 -> 25`, true `q = 1.0 cm^2/s`, start at `q = 0.05`: `assert_ll_monotonic(em_log_likelihoods, tol=1e-3)`; `assert_ll_improves`; fitted `q` within a factor 2 of 1.0 (weakly identified); the amplitude RMSE after EM is `<=` the RMSE at the starting `q` + 0.5 cm. |
| `::test_em_recovers_cycle_offset_variance_and_test_detects_it` (slow) | Session with `cycle_offset_std = 6`: fitted `sqrt(cycle_offset_variance)` in `[4, 8]`; `cycle_variance_test(null_fit, alt_fit).p_value < 0.01`; `statistic > 20`. |
| `::test_cycle_variance_test_size_under_null[seed]` (slow, seeds 0-5) | `cycle_offset_std = 0`: fitted `cycle_offset_variance < 1 cm^2` in every seed; `p_value > 0.05` in at least 5 of 6 seeds (collected into one assertion over the parametrised results via a module-level list, or a single test looping over seeds). |
| `::test_shuffle_scores_use_fitted_hyperparameters_without_refitting` (slow) | After `fit(fit_hyperparameters=True)`, `phase_shuffle_scores(n_shuffles=2)` leaves `model.process_variance` and `model.cycle_offset_variance` unchanged and `result_` identical. |
| `::test_fit_without_hyperparameter_fitting_is_unchanged` | `fit(fit_hyperparameters=False)` on `constant_session` returns `process_variance is None`, `em_log_likelihoods is None`, and `coefficient_mean` identical (bit-for-bit) to the phase 1 fixture's result. |
| `tests/test_em_driver.py` (existing) | Unchanged. |
| `uv run mypy`, ruff check / format (including `scripts/`) | Clean. |

Mark slow / integration tests explicitly (e.g., `pytest.mark.slow`). Tests that call `.fit(` are auto-marked.

## Fixtures

- `constant_session`, `offset_session` from earlier phases; a `drift_session` (`session(seed=3, amplitude=np.linspace(5, 25, n_time), n_time=7500)`) and `null_offset_sessions` (seeds 0-5, `amplitude=15.0`, `cycle_offset_std=0.0`, `n_time=7500`), each fitted with `max_iter=20` — module scope so every EM fit runs once.
- Random smoothed moments for the M-step tests: two bouts of `T = 5` and `T = 7` bins, `D = 3`, built as in `test_oracle_point_process.py:689-697` with the offset block appended, and random boolean `cycle_starts` with the first bin `True`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind) — none in this phase.
- User-facing documentation listed as tasks is updated, not deferred.
