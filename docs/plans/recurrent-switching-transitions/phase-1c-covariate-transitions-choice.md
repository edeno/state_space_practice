# Phase 1c — Covariate-dependent transitions in the switching choice family

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#j-sibling-integration-notes)

Mirrors phase 1a into `switching_choice_filter`, the control-aware
`switching_choice_smoother`, `SwitchingChoiceModel` (EM + SGD, including its
uncertainty summaries) and `simulate_switching_choice_data`. The choice model
already threads a covariate `B u_t` into the *dynamics*
(`switching_choice.py:331-343`); this phase adds covariates that drive the
*strategy switching* itself (e.g. reward history or block cues), keeping the
two covariate channels independent (`covariates=` vs `transition_covariates=`).

**Inputs to read first:**

- [shared-contracts.md](shared-contracts.md) — [sibling-mirroring checklist](shared-contracts.md#sibling-mirroring-checklist) (choice column), [model attribute contract](shared-contracts.md#model-attribute-contract) (trailing-underscore names), [time alignment](shared-contracts.md#time-alignment-contract).
- Phase 1a's diffs of `switching_kalman.py` and `oscillator_models.py` (the helpers to mirror).
- `src/state_space_practice/switching_choice.py:1-22` (module docstring, already cites Linderman et al. 2017), `:45-53` (imports from `switching_kalman`), `:193-204` (result tuple), `:206-249` (public filter; positional forwarding `:234-249`), `:252-271` (jitted signature), `:331-343` (both-or-neither precedent), `:369-374` (prior sanitising), `:432-490` (`_step`; discrete update `:464-469`), `:492-513` (scan; inputs `:499`), `:544-593` (smoother signature and docstring; the promise at `:571-572`), `:595-633` (`_backward_step`; inner call `:601-608`; `xs` `:626-631`), `:687-793` (model; `:742-761` constructor checks; `:763-778` shared params and covariate slots), `:799-903` (`_populate_uncertainty`; predicted prior `:813-822`), `:912-931` (`_run_filter`), `:933-1011` (`fit`; binding `:965-971`; loop `:977-992`), `:1013-1034` (`_run_smoother`), `:1036-1118` (`_m_step`; transition `:1115-1118`), `:1122-1173` (`fit_sgd`; binding `:1159-1165`), `:1179-1203` (spec), `:1205-1227` (loss), `:1229-1238` (store), `:1240-1248` (finalize), `:1251-1359` (simulator; `_state_step` `:1325-1330`).
- `src/state_space_practice/tests/test_switching_choice.py:228-287` (filter tests incl. structural-zero and gradient tests), `:385` (`TestSwitchingChoiceSmootherControlInput`), `:829-851` (`fitted_switching_choice_model` fixture), `:989` (`TestSwitchingChoiceRecovery`), `:1121-1217` (`TestSwitchingChoiceMStepExactness`; transition stationarity `:1209-1217`).

**Contracts referenced:**

- All of [shared-contracts.md](shared-contracts.md); `n_cont_states` for this family is `n_options - 1`.

**Designs referenced:** [designs.md J](designs.md#j-sibling-integration-notes) (choice notes), [C](designs.md#c-gpb1-smoother-integration) (per-step smoother call), [E](designs.md#e-logit-m-step) (install helper), [I](designs.md#i-simulators).

## Tasks

- **Filter.** `_switching_choice_filter_jit` and `switching_choice_filter` gain
  `transition_covariates: ArrayLike | None = None, transition_weights: ArrayLike | None = None`
  (keyword-only; the wrapper passes them by keyword, `:234-249`). In the core:
  `resolve_scan_transitions(discrete_transition_matrix, transition_covariates,
  transition_weights, n_trials)` after the defaults block (`:301-329`, so the
  default matrix is the baseline when the caller passed `None`); `_step`
  unpacks `(choice_t, u_t, z_t, transition_input)` (`:434`), computes
  `transition_t = transition_at_step(transition_input, prev_mean)` and passes it
  at `:466-468`; `scan_inputs = (choices[1:], cov_arr[1:], obs_cov_arr[1:], per_step)`
  (`:499`). The wrapper calls `validate_transition_inputs(...)` after
  `validate_choice_indices` (`:233`) with `n_cont_states=n_options - 1` and the
  *resolved* baseline (the `0.9 I + 0.1 / S` default when `None`). Docstring
  parameters (`:274-291`).
- **Smoother.** `switching_choice_smoother(..., *, transition_covariates=None,
  transition_weights=None)`: resolve `stack = transition_matrix_stack(discrete_transition_matrix,
  ..., n_time=filtered_values.shape[0])` once; add `None if stack is None else stack[1:]`
  to the scan `xs` (`:626-631`); `_backward_step` unpacks it and passes
  `discrete_state_transition_matrix=transition_next` (or the fixed matrix when
  `None`) in the inner `switching_kalman_smoother` call (`:601-608`). Update the
  docstring's equivalence statement (`:571-572`): with `control_input == 0` and
  the same transition keywords it reproduces `switching_kalman_smoother(...,
  transition_covariates=..., transition_weights=...)` exactly.
- **`SwitchingChoiceModel`.** Keyword-only `transition_regularization: float = 1e-5`
  on `__init__` (`:711-720`), validated `>= 0` next to `:742-761`; attributes
  `self.transition_weights_ = None`, `self._transition_covariates = None` next to
  `:776-778`; helpers `_transition_kwargs`, `_install_discrete_transition`,
  `_bind_transition_covariates` (attribute names per the contract;
  `n_cont_states=n_options - 1`; validate against `self.discrete_transition_matrix_`);
  `_run_filter` (`:912-931`) and `_run_smoother` (`:1026-1034`) forward
  `**self._transition_kwargs()`; `_m_step` replaces `:1115-1118` with
  `self._install_discrete_transition(trans_counts / row_sums)` (keep the `eps`
  guard for the count-based value; the logit path uses `smoother_result[3]` as
  `xi`); `fit` (`:933-940`) and `fit_sgd` (`:1122-1131`) gain keyword-only
  `transition_covariates=None` and call `_bind_transition_covariates(...,
  self._n_trials)` next to the existing binding (`:968-971`, `:1162-1165`);
  `_build_param_spec` (`:1179-1203`) adds `"transition_weights"` (`UNCONSTRAINED`)
  when covariates are bound; `_sgd_loss_fn` (`:1205-1227`) forwards the keywords
  to `_switching_choice_filter_jit` and adds
  `(self._n_trials - 1) * self.transition_regularization * jnp.sum(gamma ** 2)`;
  `_store_sgd_params` (`:1229-1238`) stores `transition_weights_`;
  `_populate_uncertainty` (`:813-822`) computes `predicted_disc[1:]` as
  `jnp.einsum("ti,tij->tj", disc_probs[:-1], stack[1:])` when
  `transition_matrix_stack(self.discrete_transition_matrix_, **kwargs)` is not
  `None` (falls back to the existing matmul otherwise). Class docstring
  (`:693-709`) gains the new constructor parameter and a sentence distinguishing
  `covariates` (dynamics) from `transition_covariates` (switching).
- **Simulator.** `simulate_switching_choice_data(..., transition_covariates=None,
  transition_weights=None)`: `_state_step` (`:1325-1330`) takes `(key_t, u_t)` and
  draws from `centered_log_softmax(eta[prev] + u_t @ Gamma[:, prev, :])` on the
  covariate path; the fixed path keeps `jax.random.choice(key_t, S, p=transition_matrix[prev_state])`
  verbatim. Docstring gains the arguments and the alignment rule.
- **Tests** — validation slice below, in `tests/test_switching_choice.py`.
- **User-facing docs.** `CHANGELOG.md` `### Added`: extend the phase-1 entry with
  the choice functions, model and simulator. Docstrings as listed.

## Deliberately not in this phase

- `transition_state_weights_` — phase 2b (the choice filter's in-scan rule already
  receives `prev_mean`).
- Learning `inverse_temperatures` / `decays` by EM (out of scope, unchanged;
  `:941-945`).
- Changing the x₀ convention of the choice filter (`:376-382`) or how
  `covariates` / `input_gain` enter the dynamics.
- `ContingencyBeliefModel` (already covariate-driven; only its helpers moved in 1a).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_switching_choice.py::TestSwitchingChoiceFilter::test_fixed_path_bit_identical_with_none_transition_options` | 100 trials, K=3, S=2: every field of `SwitchingChoiceFilterResult` is `assert_array_equal` with and without the keywords as `None`; same for the 9 smoother outputs. |
| `test_switching_choice.py::TestSwitchingChoiceFilter::test_zero_transition_weights_reproduce_fixed_matrix` | `Gamma = 0` + random covariates: all fields match the fixed path to 1e-12. |
| `test_switching_choice.py::TestSwitchingChoiceFilter::test_one_hot_transition_covariate_switches_at_its_own_trial` | identical per-state parameters (`inverse_temperatures`, `process_noises`, `decays` shared; posterior = prior chain), one-hot covariate at trial `t* = 5`, logits `-12`: `discrete_state_probs[t*, 1] > 0.999`, `[t*-1, 1] < 0.05`; same on the smoother's discrete marginals. |
| `test_switching_choice.py::TestSwitchingChoiceFilter::test_transition_covariates_require_weights` | one of the two given → `ValueError`; `init_discrete_prob=[1, 0]` + `discrete_transition_matrix=eye` + covariates → `ValueError` about structural zeros (the `:260-278` structural-zero behaviour is preserved on the fixed path). |
| `test_switching_choice.py::TestSwitchingChoiceFilter::test_marginal_ll_gradient_finite_wrt_transition_weights` | `jax.grad` through `_switching_choice_filter_jit` w.r.t. `Gamma` finite and nonzero (extends `:280-287`). |
| `test_switching_choice.py::TestSwitchingChoiceSmootherControlInput::test_smoother_with_transition_covariates_matches_library_smoother` | `control_input = 0`: `switching_choice_smoother(..., transition_covariates=u, transition_weights=Gamma)` equals `switching_kalman_smoother(filtered..., Z, transition_covariates=u, transition_weights=Gamma)` to 1e-12 (all 9 outputs); guard `ptp(stack[1:]) > 0.1`. |
| `test_switching_choice.py::TestSwitchingChoiceModel::test_run_filter_and_smoother_forward_transition_kwargs` | with `_transition_covariates` bound and nonzero `transition_weights_`, `_run_filter` / `_run_smoother` equal direct calls with the same keywords (1e-12); unbound → equal the fixed-path calls. |
| `test_switching_choice.py::TestSwitchingChoiceMStepExactness::test_transition_logits_are_stationary_when_covariates_bound` | extends `:1209-1217`: with covariates bound, the installed `(eta, Gamma)` is a stationary point of `sum_t sum_ij joint[t,i,j] log T_t[i,j]` (central-difference gradient `< 1e-6` relative) and the objective is ≥ its value at the pre-M-step parameters; guard: `max(abs(Gamma)) > 1e-3`. Without covariates the existing `N / Z` Lagrange check still passes (fixed path unchanged). |
| `test_switching_choice.py::TestSwitchingChoiceUncertainty::test_predicted_discrete_prior_uses_transition_stack` | after `fit(..., transition_covariates=u)`: `predicted_option_variances_` equals a recomputation with `predicted_disc[t] = disc_probs[t-1] @ stack[t]` (1e-10) and differs from the fixed-matrix recomputation by > 1e-6 (guard). |
| `test_switching_choice.py::TestSwitchingChoiceRecovery::test_em_recovers_transition_covariate_sign` (slow) | `simulate_switching_choice_data(n_trials=1500, transition_covariates=u, transition_weights=Gamma)` with `Gamma = -1.5` on both rows, 3 seeds: `fit(choices, transition_covariates=u, max_iter=30)` gives both `transition_weights_` entries negative after `find_permutation` alignment (magnitude in `[0.5, 4.0]`; the choice observation is weak, so the band is wider than the Gaussian family's). |
| `test_switching_choice.py::TestSwitchingChoiceRecovery::test_sgd_learns_transition_covariate_sign` (slow) | `fit_sgd(choices, transition_covariates=u, num_steps=80)`: weights negative, LL improves; spec contains `"transition_weights"` only when bound. |
| `test_switching_choice.py::TestSimulateSwitchingChoiceData::test_fixed_path_unchanged_by_none_transition_options` | identical `SimulatedSwitchingChoiceData` fields with and without the `None` keywords for the same seed. |
| `test_switching_choice.py::TestSimulateSwitchingChoiceData::test_covariate_gated_switching_follows_covariate` | 20000 trials: state-1 occupancy during `u > 1` exceeds that during `u < -1` by > 0.2; < 0.05 with `Gamma = 0`. |

## Fixtures

- Reuse `fitted_switching_choice_model` (`tests/test_switching_choice.py:829-851`)
  and the `TestSwitchingChoiceMStepExactness.em_inputs` class fixture
  (`:1138-1158`), parametrised with `transition_covariates` bound
  (`model._bind_transition_covariates(u, n_trials)`, `model.transition_weights_ = Gamma`).
- `tests/recovery_helpers.py::simulate_speed_covariate` (added in 1a/1b) for the
  covariate; rename its docstring wording to "smooth standardized covariate" if
  the speed framing reads oddly for a bandit.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the `_m_step` transition update (`switching_choice.py:1115-1118`) and `_populate_uncertainty`'s fixed-matrix matmul (`:819`) are replaced, not duplicated.
- User-facing documentation listed as tasks is updated, not deferred.
- Diff the choice helpers against phase 1a's `BaseModel` helpers; identical behaviour apart from attribute names.
