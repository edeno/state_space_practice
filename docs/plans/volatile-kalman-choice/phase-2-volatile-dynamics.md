# Phase 2 — `CovariateChoiceModel(dynamics="volatile")`: volatility as the process noise of the Laplace-EKF choice filter

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#hybrid-filter)

Add a volatile variant of the covariate-driven choice filter in which the
VKF volatility state supplies the per-trial process variance `Q_t = v_{t-1} I`
and is updated from the change in the filtered value after the softmax
update. Exposed as an additive `dynamics="volatile"` option on
`CovariateChoiceModel`; `dynamics="fixed"` (the default) is untouched and
must stay bit-for-bit identical. Includes the identifiability gate for the
volatility/inverse-temperature trade-off.

**Inputs to read first:**

- [designs.md §6-8](designs.md#hybrid-filter) — the substitution table, `_volatility_prediction_error`, the volatile filter scan, the model hooks, `signed_reward_covariates`, and the identifiability gate.
- `src/state_space_practice/volatile_kalman.py` (phase 1): `_volatility_update` (reused) and `simulate_volatile_bandit_data` (test data).
- `src/state_space_practice/covariate_choice.py:1-24` (module docstring; reference [2] there has the wrong year/volume, see tasks), `:54-87` (`covariate_predict`, unchanged), `:210-320` (`covariate_choice_filter`: validation and coercion to copy), `:323-384` (`_covariate_choice_filter_jit`; the scan at `:342-370` is the template — `Q` at `:339`, predict `:347-348`, update `:354-361`), `:387-426` (`_rts_smoother_pass_with_predictions`; `:409-412` are the gain / cross-covariance lines the volatility update reuses), `:429-487` (`covariate_choice_smoother`; `:464-487` is factored into a helper), `:522-616` (class docstring, `__init__`), `:618-626` (`__repr__`), `:630-648` (`_filter_kwargs`, `_run_filter`, `_run_smoother`), `:764-843` (`fit`, `fit_sgd`), `:847-904` (SGD hooks), `:938-968` (`n_free_params`, `summary`).
- `src/state_space_practice/multinomial_choice.py:116-283` (`_softmax_update_core`, reused unchanged; Newton constants at `:45`, `:47`, `:62` untouched), `:772-814` (`_populate_uncertainty`, overridden in the subclass), `:1009-1018` (`process_noise` in `_build_param_spec` — reused as `v0`), `:1053-1086` (`_m_step_process_noise`; its per-trial summand is what the volatility update evaluates online).
- `src/state_space_practice/tests/test_covariate_choice.py:275-320` (parity-test shape), `:363-379` and `:424-442` (smoother invariants to mirror), `:1341-1412` (`TestSGDFitting` conventions).
- `src/state_space_practice/tests/test_gradients.py:196-216` (the shape of the new gradient test).
- `docs/plans/identifiability-diagnostics/` — the `.identifiability_report()` this phase's gate depends on (see overview Dependency policy). Read its plan before writing the gate test; if it has landed, use the report, not the interim Hessian.

**Contracts referenced:** none.

**Designs referenced:** [designs.md#hybrid-filter](designs.md#hybrid-filter), [#hybrid-model-hooks](designs.md#hybrid-model-hooks), [#identifiability-gate](designs.md#identifiability-gate).

## Tasks

- **Baseline capture (before any edit).** Run `covariate_choice_filter` and `covariate_choice_smoother` (with and without covariates / obs covariates, `decay=0.9`) on a fixed simulated dataset (`simulate_rl_choice_data(n_trials=200, seed=3)`) and `CovariateChoiceModel(n_options=3, n_covariates=2).fit_sgd(..., num_steps=30)` (record `log_likelihood_history_`, `input_gain_`, `process_noise`, `inverse_temperature`); save to `<scratchpad>/covariate_choice_baseline.npz` (`*.npz` is gitignored). After the edits, re-run identically and `assert_array_equal` every array (bit-for-bit). This is the only check that the default path was not touched.
- **Volatile filter core in `covariate_choice.py`**: `VolatileChoiceFilterResult`, `_volatility_prediction_error`, `_volatile_covariate_choice_filter_jit` exactly as in [designs.md §6](designs.md#hybrid-filter); public `volatile_covariate_choice_filter(...)` with the same validation/coercion as `covariate_choice_filter` plus `0 <= volatility_learning_rate < 1` and `initial_volatility > 0`. `_covariate_choice_filter_jit` is not modified.
- **Smoother**: factor `covariate_choice_smoother`'s tail (`:464-487`) into `_smooth_filter_result(filt, A) -> ChoiceSmootherResult` and add `volatile_covariate_choice_smoother(...)` = volatile filter + `_smooth_filter_result`. The backward pass consumes the stored `predicted_covariances`, so no smoother change is needed for the time-varying `Q_t`; document that in the new function's docstring.
- **`signed_reward_covariates(choices, rewards, n_options)`** in `covariate_choice.py` ([designs.md §7](designs.md#hybrid-model-hooks)), with the covariate-indexing convention of the module docstring (`covariates[t]` drives `x_{t-1} -> x_t`; row 0 zero).
- **Model hooks** ([designs.md §7](designs.md#hybrid-model-hooks)): additive `dynamics`, `init_volatility_learning_rate`, `learn_volatility_learning_rate` on `__init__` (`ValueError` for unknown `dynamics` or `λ ∉ [0, 1)`); `process_noise` documented as `v0` in volatile mode; dispatch in `_filter_kwargs` / `_run_filter` / `_run_smoother` / `_sgd_loss_fn`; `fit()` raises `ValueError` in volatile mode before binding anything; `_build_param_spec`, `_store_sgd_params`, `n_free_params`, `__repr__`, `summary`; `_populate_uncertainty` override adding `volatility_`, `volatility_prediction_error_` and `learning_rate_ = 1 − diag(P_t)/diag(P_pred,t)`. Update the class docstring (`:522-564`) with the volatile dynamics and the "use for claims only after the identifiability report passes" note.
- **Module docstring**: document `dynamics="volatile"` in `covariate_choice.py:1-24` and, while there, correct reference [2] to Piray & Daw (2020), *PLoS Computational Biology* 16(7), e1007963 (the current text says 2021, 17(4)).
- **Tests — `tests/test_covariate_choice_volatile.py`** (see validation slice), plus `test_gradients.py::test_covariate_choice_volatile_loss` modelled on `:196-216` with `dynamics="volatile"`, `n_covariates=3`, `learn_decay=True`.
- **Identifiability gate** ([designs.md §8](designs.md#identifiability-gate)): slow test on `simulate_volatile_bandit_data(n_trials=600, seed=1)` with `signed_reward_covariates`, fit by `fit_sgd(num_steps=200)`. If `docs/plans/identifiability-diagnostics/` has landed, call `model.identifiability_report()` and assert no flagged near-null direction loads on both `{volatility_learning_rate, process_noise}` and `inverse_temperature`. Otherwise implement the interim finite-difference Hessian check from designs.md §8 as a test helper and open a follow-up with the trigger "replace with `.identifiability_report()` when identifiability-diagnostics merges". Record the observed spectrum in the docstring.
- **User-facing docs**: CHANGELOG `### Added` bullet — "**`CovariateChoiceModel(dynamics=\"volatile\")`**: the process variance of the Laplace-EKF choice filter is a tracked volatility (Piray & Daw 2020) updated online from the change in the filtered values, so the animal's effective learning rate is trial-varying; `fit_sgd` only; `volatile_covariate_choice_filter` / `_smoother`, `signed_reward_covariates`. Defaults unchanged." Plus a `### Fixed` note for the reference correction.

## Deliberately not in this phase

- **Changing `_covariate_choice_filter_jit`, `covariate_predict`, `_softmax_update_core` or the Newton constants**: the fixed path must stay bit-for-bit; the volatile scan is a sibling function.
- **EM for the volatile dynamics**: no closed-form M-step for `λ`; `fit()` raises. Revisit only if a generalised EM (gradient M-step for `λ`) is wanted — not planned.
- **Adding the hybrid to the model-comparison harness**: phase 3 compares `{VKF, fixed-q, switching}` as specified; adding `"volatile_hybrid"` as a fourth candidate is a follow-up with trigger "after phase 3 merges and the identifiability gate passes".
- **A `SwitchingChoiceModel` volatile variant** (per-state volatility): out of scope.
- **Calibration / SBC of the hybrid smoother**: the hybrid has no implemented generative counterpart (overview Non-Goals).

## Validation slice

| Test | Asserts |
| --- | --- |
| baseline comparison (task, not a permanent test) | Every array of the pre-edit `.npz` equals the post-edit run, `assert_array_equal`. |
| `test_default_dynamics_returns_fixed_result_type` | `CovariateChoiceModel(n_options=3)._run_filter(choices)` is a `ChoiceFilterResult` (not the volatile type); `repr` has no `lambda`. |
| `test_zero_learning_rate_matches_fixed_filter` | `volatile_covariate_choice_filter(λ=0, v0=q)` equals `covariate_choice_filter(process_noise=q)` on all five shared fields (`atol=1e-12`, exact expected) with covariates, obs covariates and `decay=0.9`; `volatility` constant; guard: `λ=0.5` differs. |
| `test_scalar_gaussian_volatility_error_matches_reference` | `_volatility_prediction_error` fed the scalar Kalman quantities (`filt_cov_prev=w`, `pred_cov=w+v`, `post_cov=(1−k)(w+v)`, `post_mean−pred_mean=kδ`, `A=1`) equals `vkf.m`'s `delta_v` for 20 random `(w, v, σ², δ)` draws, `atol=1e-12`. |
| `test_volatility_positive_and_finite` | Random choices, `λ=0.5`, `v0=0.01`, `K=4`: `volatility > 0`, all outputs finite; guard: `volatility` changes by > 10% somewhere. |
| `test_smoother_invariants_volatile` | `volatile_covariate_choice_smoother`: `smoothed[-1] == filtered[-1]`; `tr(smoothed cov) ≤ tr(filtered cov) + 1e-6` (mirrors `test_covariate_choice.py:363-379`, `:424-442`). |
| `test_signed_reward_covariates_encoding` | Row 0 zero; `u[t, c_{t-1}] == 2 r_{t-1} − 1`; all other entries zero; shape `(T, K)`. |
| `test_fit_raises_for_volatile_dynamics` | `fit()` raises `ValueError` matching "fit_sgd" and binds nothing (`_covariates is None`). |
| `test_repr_summary_and_free_params` | `repr` shows `lambda=`; `n_free_params` is the fixed count + 1 (and unchanged when `learn_volatility_learning_rate=False`); `summary()` lists `volatility_learning_rate`. |
| `test_sgd_populates_volatility_outputs` (slow) | After `fit_sgd`: `volatility_` `(T,)` positive, `learning_rate_` `(T, K−1)` in `[0, 1]`, `volatility_prediction_error_` finite; `model.process_noise` (v0) > 0, `0 < volatility_learning_rate < 1`. |
| `test_volatile_nests_fixed_and_wins_on_volatile_data` (slow) | On `simulate_volatile_bandit_data(seed=0)` with `signed_reward_covariates`: `LL_volatile ≥ LL_fixed − 0.5` (nesting sanity) and `LL_volatile − LL_fixed > 5` nats; on stationary `simulate_rl_choice_data(seed=7)` data `LL_volatile ≥ LL_fixed − 0.5`. Both fits `num_steps=150`; record observed gaps. |
| `test_fitted_volatility_rises_after_contingency_changes` (slow) | `volatility_` mean over the 15 trials after each change point > the 15 before, for ≥ 2 of 3 change points; fitted `λ̂ > 0.02` (guard). |
| `test_gradients.py::test_covariate_choice_volatile_loss` (slow) | Finite-difference agreement for `λ`, `v0`, `β`, `decay`, `B`. |
| `test_identifiability_gate_volatility_vs_inverse_temperature` (slow) | Per designs.md §8: no near-null direction mixing `{λ, v0}` with `β`; observed spectrum recorded. |

Mark slow / integration tests explicitly (e.g., `pytest.mark.slow`).

## Fixtures

- `simulate_volatile_bandit_data(seed=0)` (phase 1) with `signed_reward_covariates` as a `scope="module"` fixture; `simulate_rl_choice_data(n_trials=300, seed=7)` for the stationary control.
- Random `(w, v, σ², δ)` draws for the scalar-equivalence test from `np.random.default_rng(0)`, positive via `abs` + `0.01`.
- The baseline `.npz` lives in the scratchpad directory only (never committed).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind). (The only removal is the inlined smoother tail replaced by `_smooth_filter_result`; confirm no duplicate remains.)
- User-facing documentation listed as tasks is updated, not deferred.
- The baseline `.npz` comparison was actually run and reported in the PR description (bit-for-bit for `dynamics="fixed"`).
