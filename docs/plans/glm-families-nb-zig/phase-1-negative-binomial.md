# Phase 1 — Negative-binomial observation family for over-dispersed spike counts

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#2-negative-binomial-family)

Ships `negative_binomial_family(dt, dispersion, ...)` next to `poisson_family`,
the `score` hook it needs on `GLMFamily`, a `family=` keyword on
`stochastic_point_process_filter` / `stochastic_point_process_smoother`, and
`PointProcessModel(family="negative_binomial", dispersion=..., update_dispersion=...)`
with the dispersion learnable by SGD. Every default path stays byte-identical.

**Inputs to read first:**

- [src/state_space_practice/point_process_kalman.py:1215-1239](../../../src/state_space_practice/point_process_kalman.py#L1215) — `GLMFamily`; the field to append is `score`.
- [point_process_kalman.py:1242-1265](../../../src/state_space_practice/point_process_kalman.py#L1242) — `poisson_family`: the closure style, `_safe_expected_count` clip and the hoisting note to copy.
- [point_process_kalman.py:1333-1464](../../../src/state_space_practice/point_process_kalman.py#L1333) — `glm_laplace_update`; the two `innovation = observations - mu` lines (`:1407`, `:1422`) become the score hook.
- [point_process_kalman.py:589-610](../../../src/state_space_practice/point_process_kalman.py#L589) — `_safe_expected_count` (reused as the NB mean).
- [point_process_kalman.py:1467-1484, 1657-1662, 1714-1726](../../../src/state_space_practice/point_process_kalman.py#L1467) — filter signature, `use_block_dispatch`, the impl call.
- [point_process_kalman.py:1737-1744, 1800-1846](../../../src/state_space_practice/point_process_kalman.py#L1737) — impl `static_argnames` and the scan body `_step` with the legacy update call at `:1820-1833`.
- [point_process_kalman.py:2392-2410, 2526-2531, 2571-2588](../../../src/state_space_practice/point_process_kalman.py#L2392) — smoother signature, block dispatch, inner filter call.
- [point_process_kalman.py:2928-3474](../../../src/state_space_practice/point_process_kalman.py#L2928) — `PointProcessModel`: `__init__` `:2993-3006`, `_e_step` `:3065-3105`, `fit` `:3150-3230` (count validation `:3180`), `fit_sgd` `:3234-3288` (`:3266`), `_build_param_spec` `:3294-3312`, `_sgd_loss_fn` `:3314-3336`, `_store_sgd_params` `:3338-3346`, `_finalize_sgd` `:3348-3371`.
- [src/state_space_practice/parameter_transforms.py:96-99](../../../src/state_space_practice/parameter_transforms.py#L96) — `POSITIVE` (softplus) transform for the learnable dispersion.
- [src/state_space_practice/sgd_fitting.py:354-373, 506-560](../../../src/state_space_practice/sgd_fitting.py#L354) — mixin protocol; how `_build_param_spec` / `_sgd_loss_fn` are consumed.
- [src/state_space_practice/coupling_ekf.py:87-106](../../../src/state_space_practice/coupling_ekf.py#L87) — the existing non-Poisson caller of `glm_laplace_update` (constant Jacobian pattern); must be unaffected.
- [src/state_space_practice/hamiltonian_core.py:179-191](../../../src/state_space_practice/hamiltonian_core.py#L179) — the other `poisson_family` caller; unaffected.
- [src/state_space_practice/simulate_data.py](../../../src/state_space_practice/simulate_data.py) — NumPy simulators; `rng.poisson` at `:51`, `:72`, `:195` show the sampling style. mypy-gated (`pyproject.toml:157`).
- [src/state_space_practice/tests/test_glm_laplace.py](../../../src/state_space_practice/tests/test_glm_laplace.py) — fixtures `:24-33`, parity `:36-68`, `TestFamilyConsistency` `:196-214` (canonical-only; do not add NB), recomputing reference `:229-287`, zero-iteration tests `:389-436`.
- [src/state_space_practice/tests/test_oracle_point_process.py](../../../src/state_space_practice/tests/test_oracle_point_process.py) — `_Problem` `:81-89`, `_simulate_problem` `:91-121`, `_run_laplace` `:124-146`, `_grid_posterior` `:191-240`, `_standardized_errors` `:243-253`, `_log_evidence_terms` `:261-267`, reference filter `:270-316`, regimes `:362-363` / `:431-438`, `TestLaplaceVsExactPosterior` `:472-536`.
- [src/state_space_practice/tests/test_calibration_point_process.py:42-78, 86-168](../../../src/state_space_practice/tests/test_calibration_point_process.py#L42) — z-statistics, `_simulate_replicates`, `_pp_smoother_z`, the calibrated Poisson tests.
- [src/state_space_practice/tests/test_sbc_ranks.py:60-81, 190-228](../../../src/state_space_practice/tests/test_sbc_ranks.py#L60) — rank helpers and the point-process rank test.
- [src/state_space_practice/tests/recovery_helpers.py:68-75, 204-228](../../../src/state_space_practice/tests/recovery_helpers.py#L68) — `assert_ll_improves`, `simulate_poisson_spikes` (pattern for the NB sampler).
- [src/state_space_practice/tests/test_point_process_kalman.py:889-912, 2731-2745, 4279-4330](../../../src/state_space_practice/tests/test_point_process_kalman.py#L889) — `PointProcessModel` fixture, SGD fixture, recovery fixture patterns.
- [src/state_space_practice/tests/test_em_golden_regression.py:239-248](../../../src/state_space_practice/tests/test_em_golden_regression.py#L239) — the `point_process_glm` golden case that must not move.
- [src/state_space_practice/tests/conftest.py:47-149](../../../src/state_space_practice/tests/conftest.py#L47) — automatic `slow` marking (`.fit(`/`.fit_sgd(` callers); tests that fit through helpers need `@pytest.mark.slow` explicitly.

**Contracts referenced:**

- [C1 — `GLMFamily` extended contract](shared-contracts.md#c1-glmfamily-extended-contract) — add/reuse `score` and `loglik_per_obs`; do not weaken invariant 4.
- [C2 — `family=` on the filter and smoother](shared-contracts.md#c2-family-on-the-filter-and-smoother) — items 1-3, 5, 6 (item 4 is Phase 2).
- [C3 — `PointProcessModel` family kinds](shared-contracts.md#c3-pointprocessmodel-family-kinds) — the `"poisson"`, `"negative_binomial"` and custom rows.

**Designs referenced:** [designs §1](designs.md#1-why-the-update-needs-a-score-hook), [§2](designs.md#2-negative-binomial-family), [§4](designs.md#4-threading-family-through-the-filter-and-smoother), [§5](designs.md#5-pointprocessmodel-family-plumbing), [§6](designs.md#6-simulators), [§7](designs.md#7-extending-the-grid-quadrature-oracle-and-the-calibration-harness).

## Tasks

- **T1 — `score` hook.** Append/reuse `score: Callable[[Array, Array, Array], Array] | None = None` and `loglik_per_obs` from the shared contract on `GLMFamily` (`point_process_kalman.py:1236-1239`) and extend its docstring (`:1223-1233`: "for a canonical link `score` is `None` and the update uses `y - mu`; non-canonical families supply `d loglik / d eta`; `fisher_weight` is always the *expected* information"). Supply per-observation normalized likelihoods for Poisson, Bernoulli and NB, reusing the field if masks/WoLF already added it. Keep scalar callback signatures and unmasked bodies unchanged. In `glm_laplace_update` add the `_score` closure from [designs §1](designs.md#1-why-the-update-needs-a-score-hook) after `prior_precision` (`:1394`) and replace both `innovation = observations - mu` lines (`:1407`, `:1422`). Update the `observations`/`eta_func` parameter docs (`:1360-1365`) to say `eta` has length `n_eta >= n_obs`. Run `uv run pytest src/state_space_practice/tests/test_glm_laplace.py src/state_space_practice/tests/test_coupling_ekf.py src/state_space_practice/tests/test_hamiltonian_core.py` — all unchanged.

- **T2 — `_log_gamma_ratio` and `negative_binomial_family`.** Add both after `poisson_family` (`:1265`), exactly as in [designs §2.2-2.3](designs.md#22-stable-lgammay--r---lgammar) including the docstring's Fisher-scoring / `r -> inf` statement. Tests in `tests/test_glm_laplace.py`: a new `TestLogGammaRatio` (value and `jax.grad` w.r.t. `r` against the exact integer sums, on both sides of the `1e3` switch and at `r = 1e10`) and `TestNegativeBinomialFamilyMath` (score vs `jax.grad`, weight vs expected negative Hessian computed as an exact truncated sum over `y` with `scipy.stats.nbinom.pmf`, weight vs `E[score^2]`, PSD, and a guard that `d mean / d eta != w` so the test actually exercises the hook).

- **T3 — Thread `family` through the filter and smoother.** Implement [designs §4](designs.md#4-threading-family-through-the-filter-and-smoother): keyword on both public functions (documented in their parameter lists, with the C2 caveats: block path Poisson-only, `max_log_count` warning, hoist the family), `and family is None` in both `use_block_dispatch` expressions, `family` forwarded to `_stochastic_point_process_filter_impl` (new static arg) and, in the smoother, to its inner filter call; the `if family is None: ... else: glm_laplace_update(...)` branch in `_step`; import `StateSpaceWarning`. Update the module docstring (`:1-29`) and the filter docstring (`:1492-1494`, `:1586-1592`) so the observation model reads "`y ~ family` (Poisson by default; see `negative_binomial_family`)".

- **T4 — `PointProcessModel` family kinds.** Implement [designs §5](designs.md#5-pointprocessmodel-family-plumbing) for `"poisson"`, `"negative_binomial"` and custom `GLMFamily` (C3): `family`, `dispersion`, `update_dispersion` kwargs with validation; `_resolve_family`, `_build_family`, `_check_observation_shape`; `family=self._family` in `_e_step` and `_finalize_sgd`; the `dispersion` entry (`POSITIVE`) in `_build_param_spec`; the rebuilt family in `_sgd_loss_fn`; the store hook in `_store_sgd_params`. Extend the class docstring (`:2928-2991`): model line becomes `y ~ family(eta = log_intensity_func(Z_k, x_k))`, new parameter entries, and a note that `dispersion` is fixed under `fit` and learned only by `fit_sgd(update_dispersion=True)`.

- **T5 — Simulators.** Add `simulate_negative_binomial_counts` to `simulate_data.py` and `simulate_negative_binomial_spikes` to `tests/recovery_helpers.py` ([designs §6](designs.md#6-simulators)). `uv run mypy` must stay clean (`simulate_data.py` is gated).

- **T6 — Oracle extension.** Refactor `tests/test_oracle_point_process.py` per [designs §7](designs.md#7-extending-the-grid-quadrature-oracle-and-the-calibration-harness) (`_Observation` record, Poisson and NB instances; Poisson defaults keep every existing test identical), then add: `TestLaplaceFilterComputesItsApproximation` parametrised with the NB observation (`r = 3`, both regimes, N1 to roundoff and N-converged to `1e-7`) and `TestNegativeBinomialLaplaceVsExactPosterior` (T = 6, 1-2 latents, seeds 0-2, an over-dispersed moderate-rate regime `rate_hz = 40, r = 2` and the near-Gaussian regime with `r = 5`). Measure the standardised errors, record them in the class docstring like `:473-499`, and pin with ~10x headroom on N3 / ~2x on N1 **and** the non-vacuous lower bounds (`> 1e-3`).

- **T7 — Calibration, misspecification and ranks (slow).** In `tests/test_calibration_point_process.py`: a `TestNegativeBinomialSmootherCalibration` class with (i) NB truth (`r = 1.5`, rates 30-80 Hz, `dt = 0.02`, 2000 replicates) smoothed with the NB family → `_assert_calibrated`; (ii) the **same data** smoothed with `family=None` (Poisson) → assert `E[z^2] - 1 > 4 SE` and `coverage < 0.9 - 4 SE`, then pin the observed values ±4 SE with the reason in the docstring (the Poisson likelihood claims `Var = mu` when the truth is `mu (1 + mu / r)`). In `tests/test_sbc_ranks.py`: NB truth + NB family rank test with the existing power guard (`0.75 * var` rejected).

- **T8 — Dispersion recovery (slow).** In `tests/test_point_process_kalman.py` (or a new `test_negative_binomial_model.py`): simulate 5 neurons, T = 2000, 2-D AR(1) latent, rates 10-40 Hz at `dt = 0.02`, true `r = 2` (via `simulate_negative_binomial_spikes`); fit `PointProcessModel(family="negative_binomial", dispersion=20.0, update_dispersion=True, update_transition_matrix=False, update_init_state=False)` with `fit_sgd(num_steps=300)`; assert `|log r_hat - log 2| < 0.35` and `< 0.5 |log 20 - log 2|`, final LL above the initial, and that with `update_dispersion=False` the stored `dispersion` is unchanged after `fit_sgd`.

- **T9 — Parity and plumbing tests (fast).** `tests/test_glm_laplace.py`: `glm_laplace_update` with `negative_binomial_family(_DT, 1e10)` vs `_point_process_laplace_update` on the `TestPoissonParity` inputs, `atol = rtol = 1e-8` on mean, cov and LL (and a guard that `r = 5` differs by `> 1e-3`). `tests/test_point_process_kalman.py`: the same at filter level (T = 50, 3 neurons, `family=negative_binomial_family(dt, 1e10)` vs `family=None`, 1e-8); `family` + `block_n_neurons` equals `force_dense=True` output; `max_log_count=5.0` with a family warns `StateSpaceWarning`; `PointProcessModel` kwarg validation (`dispersion` without NB, NB without `dispersion`, non-positive `dispersion`, wrong per-neuron length at `fit`, `update_dispersion` with Poisson) each raise `ValueError`; a custom `GLMFamily` instance (`BERNOULLI_LOGIT_FAMILY` on 0/1 data) runs through `PointProcessModel._e_step`.

- **T10 — Docs.** `CHANGELOG.md` "Unreleased / Added": `negative_binomial_family`, the `family=` keyword on the two filters, `PointProcessModel(family=, dispersion=, update_dispersion=)`, `simulate_negative_binomial_counts`, and the `GLMFamily.score` / `loglik_per_obs` fields. `README.md` package-layout paragraph (`:60-67`): one sentence that `point_process_kalman` exposes Poisson, Bernoulli and negative-binomial families for the Laplace-EKF filters. All new public docstrings are NumPy style with shapes.

## Deliberately not in this phase

- Zero-inflated gamma, `GLMFamily.validate_observations`, family-aware observation validation, `gamma_shape`/`gamma_loc` kwargs — Phase 2 ([phase-2](phase-2-zero-inflated-gamma.md)); NB observations are counts, so `validate_count_array` is right here.
- An EM M-step for `dispersion` — [overview Open Question 1](overview.md#open-questions); SGD is the learning path.
- Negative binomial on the block-diagonal path / `PlaceFieldModel` / `PositionDecoder` / `SwitchingSpikeOscillatorModel` — [overview non-goals](overview.md#non-goals).
- Routing `family=None` through `glm_laplace_update(poisson_family)` — C2 item 6; [overview Open Question 2](overview.md#open-questions).
- Extending `TestFamilyConsistency.test_fisher_weight_is_mean_derivative` to NB — it is a canonical-link identity (C1 item 2).

## Validation slice

| Test | Asserts |
| --- | --- |
| `TestFamilyObservationTerms::test_sum_and_masked_score[poisson,bernoulli,nb]` | `loglik_per_obs.sum()` equals the normalized scalar likelihood; the gradient of selected terms matches the masked family score. Optional callback defaults preserve four-positional-argument custom families and unmasked parity. |
| `test_glm_laplace.py::TestPoissonParity` (existing) | still passes unchanged after T1 (bit-identical program for `score=None`) |
| `test_em_golden_regression.py` `point_process_glm`, `common_oscillator_pp`, `directed_influence_pp` (existing, slow) | recorded LL histories / parameters unchanged — proves the default paths did not move |
| `test_coupling_ekf.py`, `test_hamiltonian_core.py` (existing) | unchanged (Bernoulli / Poisson callers of `glm_laplace_update`) |
| `TestLogGammaRatio::test_matches_exact_integer_sum` | `_log_gamma_ratio(y, r)` vs `fsum(log(r + i))` for `y ∈ {0, 1, 3, 17}`, `r ∈ {0.3, 2, 999, 1001, 1e6, 1e10}`: `rtol 1e-11, atol 1e-10` |
| `TestLogGammaRatio::test_gradient_in_r_is_stable` | `jax.grad` w.r.t. `r` vs `sum 1/(r + i)`, `rtol 1e-6`, including `r = 1e10` (naive digamma difference would be off by orders of magnitude) |
| `TestNegativeBinomialFamilyMath::test_score_matches_autodiff` | `J' score(y, eta, mu)` == `jax.grad` of `loglik_normalized` at random `x`, `atol 1e-10` |
| `…::test_weight_is_expected_negative_hessian` | `w` == `sum_y pmf(y) (-d^2 loglik/d eta^2)` (truncated where tail mass `< 1e-12`) and == `sum_y pmf(y) score^2`, `rtol 1e-8` |
| `…::test_weight_is_not_mean_derivative` | guard: `max(abs(w - d mean/d eta)) > 1e-3` at `r = 2` (the hook is exercised) |
| `…::test_fisher_is_psd` | `w >= 0`, `eigvalsh(J' diag(w) J).min() >= -1e-10` at extreme `eta` |
| `TestNegativeBinomialPoissonParity::test_update_matches_legacy_at_large_r` | `glm_laplace_update(NB, r=1e10)` vs `_point_process_laplace_update`: mean/cov/LL within `1e-8`; guard `r = 5` differs `> 1e-3` |
| `test_point_process_kalman.py::…::test_filter_family_nb_large_r_matches_default` | filter with `family=NB(1e10)` vs `family=None`, T = 50: means/covs/LL within `1e-8` |
| `…::test_family_disables_block_dispatch` | `family` + block ints == `force_dense=True` result exactly (`assert_array_equal`) |
| `…::test_family_with_max_log_count_warns` | `pytest.warns(StateSpaceWarning)` |
| `…::TestPointProcessModelFamilyValidation` (5 cases) | each invalid kwarg combination raises `ValueError` with the documented message; custom `GLMFamily` accepted |
| `TestLaplaceFilterComputesItsApproximation[nb]` (oracle) | library NB filter vs independent NumPy Fisher recursion: N1 `rtol 1e-10`, converged `1e-7`; guard that updates are nonlinear |
| `TestNegativeBinomialLaplaceVsExactPosterior` (oracle, ~10 s) | standardised errors vs grid posterior below the pins recorded in the docstring; `> 1e-3` (non-vacuous); N3 ≤ N1 on means |
| `TestNegativeBinomialSmootherCalibration::test_nb_family_is_calibrated` — **slow** | NB truth, NB family: `_assert_calibrated` (mean, `E[z^2]`, 90% coverage within ±4 SE; `SE(E[z^2]) < 0.025`) |
| `…::test_poisson_family_under_covers_on_nb_truth` — **slow** | same data, Poisson: `E[z^2] - 1 > 4 SE`, `coverage < 0.9 - 4 SE`; observed values pinned ±4 SE |
| `test_sbc_ranks.py::test_nb_smoother_ranks_are_uniform` — **slow** | chi² `< CRITICAL`; power guard `0.75 * var` rejected |
| `TestNegativeBinomialDispersionRecovery` — **slow** | `abs(log r_hat - log 2) < 0.35` and `< 0.5 * abs(log 20 - log 2)`; LL improves; `update_dispersion=False` leaves `dispersion` untouched |
| `test_simulate_data.py::TestSimulateNegativeBinomialCounts` | `N = 2e5` draws: mean within 1% of `mu`, variance within 2% of `mu + mu^2/r`; `r = 1e12` gives `var/mean` within 1% of 1; same seed → identical draws |
| `test_recovery_helpers`-style check of `simulate_negative_binomial_spikes` | index of dispersion `var/mean ≈ 1 + mu/r` on a constant-rate latent, within 3% |

Mark T7, T8 and the SBC test `@pytest.mark.slow` explicitly (T8 is auto-marked
via `.fit_sgd(`, but the helper-driven calibration tests are not). Fast suite
must stay under one minute.

## Fixtures

- `test_glm_laplace.py`: reuse the module constants `_MEAN, _COV, _C, _DT, _eta`
  (`:24-33`); add a module-level `_NB_R = 2.0` and `_NB_FAMILY =
  negative_binomial_family(_DT, _NB_R)` built once.
- Oracle: `_negative_binomial_observation(dt, r)` instances built in the
  parametrisation; grids as in `_grids` (`:443-445`).
- Calibration / ranks: NumPy `default_rng(seed)` replicates via the
  `sample` callable `lambda rng, m: rng.negative_binomial(r, r / (r + m))`.
- Recovery: a class-scoped fixture that simulates once (2-D AR(1), 5 neurons,
  T = 2000, `simulate_negative_binomial_spikes` with `PRNGKey(7)`) and fits once.
- No real-data slice: no over-dispersed real spike file is checked in; the
  misspecification test on simulated NB truth is the realistic-case stand-in.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind). (None are flagged in this phase; confirm no parallel Poisson path was introduced beyond the documented `family=None` legacy branch.)
- User-facing documentation listed as tasks is updated, not deferred.
