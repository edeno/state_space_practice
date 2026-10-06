# Phase 1a — Covariate-dependent transitions: shared primitive and the Gaussian switching family

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#a-transition-stack-resolution-and-in-scan-step)

Ships `state_space_practice.discrete_transitions` and threads
`transition_covariates=` / `transition_weights=` through
`switching_kalman_filter`, the GPB1 smoother, both Viterbi decoders, the logit
M-step, `BaseModel` (COM / CNM / DIM, EM and SGD) and the Gaussian simulator,
with the discrete-path-enumeration oracle extended to time-varying transition
matrices. After this PR a Gaussian switching model can have speed-, reward- or
theta-power-gated switching; the point-process and choice families follow in
phases 1b and 1c.

**Inputs to read first:**

- [shared-contracts.md](shared-contracts.md) in full — every shape, name and alignment rule below is defined there.
- `src/state_space_practice/contingency_belief.py:84-130` (softmax helpers to move), `:225-282` (`dirichlet_neg_log_likelihood` and its `jax.grad` / `jax.hessian` aliases), `:803-846` (`_optimize_transition_rows`), `:1278-1320` (`_m_step`, the only caller) — what is lifted and how it is used today; `:1296-1302` documents the row `t + 1` alignment.
- `src/state_space_practice/switching_kalman.py:49-57` (pair-update vmap), `:454-597` (`_update_discrete_state_probabilities`, unchanged consumer of the per-step matrix), `:794-1076` (filter), `:1079-1230` (Viterbi), `:1317-1383` and `:1386-1683` (GPB1 smoother), `:2063-2110` (occupancy gate helpers), `:2275-2281` (count-based discrete update, stays), `:2708-2711` (ELBO discrete term).
- `src/state_space_practice/utils.py:1769-1824` (`hmm_viterbi`), `:1827-1847` (`zero_preserving_log`).
- `src/state_space_practice/oscillator_models.py:288-299`, `:353-456`, `:404-438`, `:503-514`, `:794-800`, `:886-1030`, `:1082-1220`, `:1420-1462`, `:1745-1811`, `:2023-2084`, `:2169-2328` — the Gaussian model surface.
- `src/state_space_practice/simulate/simulate_switching_kalman.py:98-130` (`simulate`).
- `src/state_space_practice/sgd_fitting.py:66-106` (array attributes are fingerprinted by digest, so binding covariates as a model attribute is cache-safe), `:354-417` (mixin protocol and `_sgd_param_attrs`).
- `src/state_space_practice/parameter_transforms.py:225-252` (`STOCHASTIC_ROW` is the same centered parametrisation).
- `src/state_space_practice/tests/oracles.py:473-651`, `tests/test_oracle_switching_kalman.py:60-219` and `:443-525`, `tests/test_invariances.py:248-345`, `tests/test_switching_kalman.py:356-433` and `:6994-7057`, `tests/test_contingency_belief.py:1157-1197`, `tests/test_utils.py:703-712`.
- `CHANGELOG.md` (`## [Unreleased]` / `### Added` at lines 6-8), `README.md:60-67` ("Package layout"), `pyproject.toml` `[tool.mypy] files`.

**Contracts referenced:**

- [Parametrisation and layout](shared-contracts.md#parametrisation-and-layout-contract), [time alignment](shared-contracts.md#time-alignment-contract), [keywords](shared-contracts.md#keyword-contract), [None-path invariant](shared-contracts.md#none-path-invariant) — implement exactly; do not weaken.
- [`discrete_transitions` module API](shared-contracts.md#discrete_transitions-module-api) — this phase creates it (without the phase-2 arguments).
- [M-step contract](shared-contracts.md#m-step-contract), [validation contract](shared-contracts.md#validation-contract), [model attribute contract](shared-contracts.md#model-attribute-contract) (Gaussian column).
- [Sibling-mirroring checklist](shared-contracts.md#sibling-mirroring-checklist) — this phase fills the Gaussian column; the other columns are phases 1b/1c.

**Designs referenced:** [designs.md A](designs.md#a-transition-stack-resolution-and-in-scan-step), [B](designs.md#b-gaussian-filter-integration), [C](designs.md#c-gpb1-smoother-integration), [D](designs.md#d-viterbi-integration), [E](designs.md#e-logit-m-step), [F](designs.md#f-oracle-extension), [H](designs.md#h-relabelling-transform), [I](designs.md#i-simulators).

## Tasks

- **Create `src/state_space_practice/discrete_transitions.py`** with the module
  docstring and the functions of [designs.md A](designs.md#a-transition-stack-resolution-and-in-scan-step)
  (without `state_dependent_logits`, and with `transition_matrix_stack` /
  `resolve_scan_transitions` / `validate_transition_inputs` taking only the
  phase-1 arguments) and [designs.md E](designs.md#e-logit-m-step)
  (`dirichlet_neg_log_likelihood`, `optimize_transition_rows`,
  `maximize_transition_coefficients` without the `transition_state_weights` /
  `state_cond_smoother_means` arguments). Move `centered_log_softmax`,
  `centered_softmax`, `centered_softmax_inverse` (`contingency_belief.py:84-130`)
  and `dirichlet_neg_log_likelihood` (`:225-278`) verbatim. NumPy-style
  docstrings with shapes on every public function. Add the module to
  `[tool.mypy] files` in `pyproject.toml` and make it type-clean
  (`uv run mypy`).
- **Re-point `contingency_belief.py`.** Replace the moved definitions with
  `from state_space_practice.discrete_transitions import (centered_log_softmax,
  centered_softmax, centered_softmax_inverse, dirichlet_neg_log_likelihood,
  optimize_transition_rows)`; keep `_dirichlet_gradient` / `_dirichlet_hessian`
  (`:281-282`) defined from the imported function; replace the body of
  `_optimize_transition_rows` (`:803-846`) with the two-line broadcast wrapper of
  [designs.md E](designs.md#e-logit-m-step) (drop its `@jax.jit`; the lifted
  optimizer is the jitted one). `_m_step` (`:1309-1315`) is otherwise unchanged.
  Update `tests/test_contingency_belief.py:1157-1197`
  (`test_m_step_optimizer_compiles_once_across_em_iterations`) to monkeypatch
  `state_space_practice.discrete_transitions.dirichlet_neg_log_likelihood` and to
  call `discrete_transitions.optimize_transition_rows.clear_cache()`; the
  assertion (one trace across EM iterations) is unchanged. Every other
  contingency test must pass untouched.
- **Gaussian filter** ([designs.md B](designs.md#b-gaussian-filter-integration)).
  Rename the jitted body to `_switching_kalman_filter_jit` (keyword-only
  `transition_covariates=None, transition_weights=None`), add the public
  validating wrapper `switching_kalman_filter` with the same positional
  signature plus the keywords, change `_step` to unpack `(obs_t, transition_input)`
  and call `transition_at_step(transition_input, prev_state_cond_filter_mean)`
  before `_update_discrete_state_probabilities` (`switching_kalman.py:947-957`),
  and scan over `(obs[1:], per_step)` (`:1032-1042`). Extend the docstring's
  Parameters with the two keywords (text in designs B) and a Notes line on the
  `(n_time, S, S)` stack's memory footprint. Nothing else in `:990-1076` changes.
- **GPB1 smoother** ([designs.md C](designs.md#c-gpb1-smoother-integration)).
  Keyword-only `transition_covariates=None, transition_weights=None` on
  `switching_kalman_smoother`; resolve `stack` once with
  `transition_matrix_stack(..., n_time=filter_mean.shape[0])`; add the fourth
  `xs` leaf (`None if stack is None else stack[1:]`) at `:1639-1648`; use it in
  place of the closed-over matrix at `:1538-1542`. Docstring: the keywords and
  the alignment ("`stack[t + 1]` is the `S_t -> S_{t+1}` step"). `switching_kalman_smoother_gpb2` is not touched.
- **Viterbi** ([designs.md D](designs.md#d-viterbi-integration)).
  `switching_kalman_viterbi`: keywords, `validate_transition_inputs` next to the
  prior check (`:1104-1117`), forward scan over `(obs[1:], per_step)`,
  per-step `log_trans` in `_viterbi_backward` (`:1200-1207`).
  `utils.hmm_viterbi`: accept a 3-D `transition_matrix` (`:1797-1805`); docstring
  states entry `t` is the step into time `t` and entry 0 is unused.
- **ELBO diagnostics.** `compute_expected_complete_log_likelihood` and
  `compute_elbo` (`switching_kalman.py:2621-2639`, `:2997-3016`): docstring line
  that `discrete_transition_matrix` may be the `(n_time - 1, S, S)` slice
  `stack[1:]` (the term at `:2708-2711` broadcasts); no code change.
- **`BaseModel` (oscillator_models.py).** Keyword-only
  `transition_regularization: float = 1e-5` on `__init__` (`:440-456`; reject
  negative values); attributes `self.transition_weights = None`,
  `self._transition_covariates = None`; the three helpers of
  [designs.md E](designs.md#e-logit-m-step) (`_transition_kwargs`,
  `_install_discrete_transition`, `_bind_transition_covariates` with
  `n_cont_states=self.n_cont_states`); `_e_step` forwards
  `**self._transition_kwargs()` to `switching_kalman_filter` (`:913-923`) and to
  the GPB1 `switching_kalman_smoother` call (`:963-966`, not the GPB2 call);
  `_m_step` (`:1027-1028`) and `DirectedInfluenceModel._m_step_reparameterized`
  (`:2082-2083`) call `self._install_discrete_transition(Z)`; `fit` (`:1082-1142`)
  and `fit_sgd` (`:1146-1194`) gain keyword-only `transition_covariates=None`
  and call `_bind_transition_covariates(transition_covariates, observations.shape[0])`
  before initialisation (`_bind` validates against `self.discrete_transition_matrix`,
  so call it after `_initialize_parameters` when `skip_init` is False);
  `_sgd_param_attrs` (`:404-413`) gains `"transition_weights": "transition_weights"`;
  `_EM_SNAPSHOT_KEYS` (`:415-438`) and the warm-init restore tuple (`:794-800`)
  gain `"transition_weights"`; `OscillatorParameterBase._validate_parameter_shapes`
  (`:288-299`) checks `transition_weights.shape == (n_features, k, k - 1)` when
  the attribute is not `None`.
- **COM / CNM / DIM SGD.** In `_build_param_spec` (`:1420-1442`, `:1745-1775`,
  `:2169-2197`): when `self.update_discrete_transition_matrix and
  self._transition_covariates is not None`, add `params["transition_weights"]`
  with `spec = UNCONSTRAINED`. In `_sgd_loss_fn` (`:1444-1462`, `:1777-1811`,
  `:2253-2313`): pass `transition_covariates=self._transition_covariates,
  transition_weights=params.get("transition_weights", self.transition_weights)`
  to `switching_kalman_filter` when covariates are bound, and add the penalty
  `(n_time - 1) * self.transition_regularization * jnp.sum(gamma ** 2)` (n_time
  from `observations.shape[0]`). The base `_store_sgd_params` (`:1205-1217`)
  already stores mapped keys; nothing else to add. Docstrings of `fit` /
  `fit_sgd` describe the new keyword and that `transition_weights` is
  zero-initialised on first binding.
- **Gaussian simulator** ([designs.md I](designs.md#i-simulators)).
  `simulate(..., *, transition_covariates=None, transition_weights=None)`; the
  fixed path keeps `Z[s[t - 1], :]` verbatim so existing seeds reproduce; the
  docstring gains the new arguments and the alignment rule; `mypy` stays clean
  (this module is in the checked list).
- **Oracle** ([designs.md F](designs.md#f-oracle-extension)).
  `switching_lgssm_exact_posterior` accepts a `(n_time, K, K)` stack
  (`tests/oracles.py:503`, `:515`, `:525`; docstring `:29` and `:490`).
- **Tests** — the validation slice below, in `tests/test_discrete_transitions.py`
  (new: primitive + M-step), `tests/test_switching_kalman.py` (filter / smoother /
  Viterbi identities and simulator), `tests/test_oracle_switching_kalman.py`
  (exactness and M-step stationarity), `tests/test_invariances.py`
  (relabelling), `tests/test_utils.py` (`hmm_viterbi`),
  `tests/test_oscillator_models.py` (model-level EM / SGD, slow).
- **User-facing docs.** `CHANGELOG.md` `### Added`: one entry for
  "covariate-dependent discrete transitions" naming the keywords, the new
  module, the moved helpers (still importable from `contingency_belief`), the
  `transition_regularization` option and the simulator argument, plus a
  `### Changed — behavior` note that `switching_kalman_filter` is now a thin
  wrapper around a jitted core (`jax.jit(switching_kalman_filter)` still works).
  `README.md` "Package layout" (`:60-67`): one sentence pointing at
  `state_space_practice.discrete_transitions` for covariate-dependent switching.
  Public docstrings as listed in the tasks above.

## Deliberately not in this phase

- The point-process (`switching_point_process.py`, `point_process_models.py`)
  and choice (`switching_choice.py`) families — phases 1b and 1c. Do not
  "while I'm here" thread the keyword into `switching_point_process_filter`;
  1b owns its validators and the mirror tests.
- `transition_state_weights` / `state_dependent_logits` / the recurrent model — phase 2a.
- Any change to `switching_kalman_smoother_gpb2`, `_switching_kalman_m_step_inner`
  or `switching_kalman_maximization_step`'s outputs.
- A `simulate/scenarios.py` scenario for speed-gated switching (the test builds
  its own covariate; add a scenario only when a downstream plan asks for it).
- Re-basing `ContingencyBeliefModel` on the primitive beyond the import swap.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_discrete_transitions.py::test_transition_matrix_stack_rows_are_stochastic_and_entry0_is_baseline` | random `Z`, `u`, `Gamma` (S=3, F=2, T=7): rows of every entry sum to 1 within 1e-12, all entries > 0, `stack[0] == Z` to 1e-12, `stack[1:]` varies over time (`ptp > 0.05`, guard). |
| `test_discrete_transitions.py::test_transition_matrix_stack_agrees_with_contingency_io_helper` | `stack[t]` equals `compute_input_output_transition_matrix(eta, jnp.moveaxis(Gamma, 0, -1), u[t])` to 1e-12 (the two layouts describe the same model). |
| `test_discrete_transitions.py::test_resolve_scan_transitions_fixed_path_returns_none_and_identity_rule` | with both options `None`: `per_step is None` and `transition_at_step(None, anything) is Z` (identity of object). |
| `test_discrete_transitions.py::test_validate_transition_inputs_rejects_bad_shapes_and_structural_zero` | parametrised: one-of-two given; wrong covariate rows; wrong weight shape; non-finite; baseline with an exact zero → `ValueError` with the documented message fragments; skipped under `jax.jit` (no raise inside a traced function). |
| `test_discrete_transitions.py::test_logit_m_step_without_covariates_equals_count_based_map` | intercept-only, `transition_prior = get_transition_prior(1.0, 2.0, S)`: result equals `(xi.sum(0) + alpha - 1)` row-normalised to 1e-5 (guard: `alpha - 1 != 0` changes the answer by > 1e-3 relative to the ML estimate). |
| `test_discrete_transitions.py::test_low_occupancy_row_keeps_previous_logits_and_warns` | `xi` with row 1's total mass `1e-3`: row 1 of the returned matrix equals the input baseline row, row 0 changes by > 1e-3; a WARNING record from `state_space_practice.switching_kalman` containing `maximize_transition_coefficients` is logged (`caplog`). |
| `test_oracle_switching_kalman.py::test_logit_m_step_is_stationary_for_exact_statistics` | exact `xi` from `_oracle` on a K=2, T=6 model simulated with `Gamma != 0`; `l2 = 0`, no prior; finite-difference gradient (`_fd_grad`, `:462-480`) of `sum_t sum_ij xi log T_t(eta, Gamma)` at the returned `(eta, Gamma)` has max-abs < 1e-6 × scale; objective at the update ≥ objective at the start (zeros); guard: `max(abs(Gamma_new)) > 0.1`. |
| `test_switching_kalman.py::test_switching_filter_fixed_path_bit_identical_with_none_transition_options` | `simple_skf_model`: all 7 filter outputs, all 9 GPB1 outputs and the Viterbi path are `np.testing.assert_array_equal` with and without the keywords passed as `None`. Also `tests/test_em_golden_regression.py` passes unchanged. |
| `test_switching_kalman.py::test_zero_transition_weights_reproduce_fixed_matrix` | `Gamma = 0`, random covariates: filter / GPB1 / Viterbi outputs match the fixed path to `rtol=atol=1e-12`. |
| `test_switching_kalman.py::test_one_hot_covariate_switches_at_its_own_time_step` | identical per-state parameters (posterior = prior chain), `Z = 0.99 I + ...`, covariate one-hot at `t* = 5` with `Gamma` row logits `-12` toward the reference state: `filter_prob[t*, 1] > 0.999`, `filter_prob[t* - 1, 1] < 0.05`, `filter_prob[t* + 1, 1] > 0.98` (the chain then sticks); the same three checks on the GPB1 smoothed probabilities. |
| `test_switching_kalman.py::test_filter_wrapper_rejects_structural_zero_baseline_with_covariates` | `Z = I` with covariates → `ValueError` mentioning `centered_softmax_inverse`; the same call with no keywords still runs (structural zeros remain supported on the fixed path; `tests/test_switching_kalman.py:7253` `test_filter_preserves_structural_zero_in_initial_prob` keeps passing). |
| `test_switching_kalman.py::test_filter_marginal_ll_gradient_finite_wrt_transition_weights` | `jax.grad` of the marginal LL w.r.t. `Gamma` is finite and nonzero (SGD path). |
| `test_switching_kalman.py::test_viterbi_time_varying_transitions_matches_brute_force_map_path` | identical-states K=2, T=6 with a strongly varying stack: Viterbi path equals `argmax` over the 64 paths of the oracle's `path_log_prior + path_log_likelihood`; guard: the MAP path under the fixed `Z` differs. |
| `test_utils.py::test_hmm_viterbi_time_varying_stack_matches_brute_force` | K=2, T=5, random log-likelihoods and random stack: equals brute force over 32 paths; 2-D input still gives `[0, 1, 0]` on the `:703-712` case. |
| `test_oracle_switching_kalman.py::test_identical_states_exact_with_time_varying_transitions` (slow) | `_random_switching_model(identical=True)` + random `u`, `Gamma`: filter (`_assert_filter_exact`), GPB1 and GPB2 (`_assert_smoother_exact`) at `RTOL = 1e-8` against the oracle fed `stack`; guard `ptp(stack[1:]) > 0.1` and `ptp(smoothed prob) > 0.05`. |
| `test_oracle_switching_kalman.py::test_two_steps_filter_and_gpb2_exact_with_time_varying_transitions` (slow) | distinct states, T=2: filter and GPB2 exact at `RTOL`; GPB1 gap > 1e-6 (the `:330-344` pattern). |
| `test_oracle_switching_kalman.py::test_elbo_transition_term_accepts_stack` | `compute_expected_complete_log_likelihood(..., discrete_transition_matrix=stack[1:])` minus the same with `Z` equals `sum(xi * (log stack[1:] - log Z))` to 1e-10. |
| `test_invariances.py::TestSwitchingKalmanInvariances::test_discrete_state_relabelling_with_transition_covariates` (slow) | `_switching(3, K=3)` + covariates; permuted parameters from `permute_transition_params` ([designs.md H](designs.md#h-relabelling-transform)); the `:338-345` assertions hold; guard `ptp(prob) > 0.1`. |
| `test_switching_kalman.py::test_simulate_fixed_path_unchanged_by_none_transition_options` | `simulate(...)` with and without the `None` keywords returns identical `(y, s, x)` for the same seed. |
| `test_switching_kalman.py::test_simulate_speed_gated_switching_follows_covariate` | T=20000, smooth standardized "speed" `u`, `Gamma = -1.5` on both rows (speed pushes toward state 1): fraction of time in state 1 when `u > 1` exceeds that when `u < -1` by > 0.2; with `Gamma = 0` the two fractions differ by < 0.05. |
| `test_oscillator_models.py::test_com_em_recovers_speed_gated_transition_weights` (slow) | `simulate_com_scenario`-style COM parameters, T=3000, `Gamma = -1.5` on both rows, 3 seeds: `CommonOscillatorModel(...).fit(obs, transition_covariates=u)` gives both learned weights negative and within `[0.75, 3.0]` in magnitude (labels resolved by `utils.find_permutation` against the true states); LL non-decreasing beyond `tol`; `model.transition_weights.shape == (1, 2, 1)`. |
| `test_oscillator_models.py::test_com_fit_sgd_moves_transition_weights_toward_truth` (slow) | same data, `fit_sgd(obs, transition_covariates=u, num_steps=60)` from zero weights: both weights negative after fitting and LL improves (`assert_ll_improves`); `transition_weights` appears in `_build_param_spec()` only when covariates are bound. |
| `test_oscillator_models.py::test_fit_rejects_mismatched_transition_weights` | `model.transition_weights` set to shape `(2, 2, 1)` then `fit(obs, transition_covariates=u)` with `u.shape[1] == 1` → `ValueError`. |
| `test_oscillator_models.py::test_em_snapshot_restores_transition_weights_on_rollback` (slow) | via `conftest.assert_em_rolls_back_on_ll_decrease` pattern: after a forced rollback `transition_weights` equals the pre-M-step value. |
| `test_contingency_belief.py` (all existing) | pass unchanged except the monkeypatch target of `:1157-1197`. |

Slow tests are marked (`@pytest.mark.slow` or auto-marked by `conftest.py` for `.fit(` / `.fit_sgd(`). The whole fast suite stays under a minute; the new fast tests use `T <= 8`, `S <= 3`.

## Fixtures

- `tests/test_discrete_transitions.py`: module fixture `small_transition_problem`
  (seed 0): `S = 3`, `F = 2`, `T = 7`, `Z = 0.7 I + 0.3 Dirichlet`, `u ~ N(0, 1)`,
  `Gamma ~ N(0, 0.5)`; and `exact_xi` built from a random `(T-1, S, S)` positive
  array normalised to sum 1 per `t` (an M-step needs only a valid joint).
- `tests/test_oracle_switching_kalman.py`: reuse `_random_switching_model` and
  add `_with_covariates(model, rng, n_features)` returning `u`, `Gamma` and the
  `stack` from `transition_matrix_stack`; simulated data regenerated from the
  stack (the `:107-125` loop with `stack[t]` in place of `Z`).
- `tests/test_switching_kalman.py`: reuse `simple_skf_model` (`:356-433`); the
  speed covariate is `standardize(cumsum of AR(1) noise)` generated in the test.
- `tests/test_oscillator_models.py`: reuse `common_oscillator_params` (fixture defined at `:33`, before the next fixture at `:52`)
  and simulate with the extended `simulate`; labels aligned with
  `utils.find_permutation`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the softmax helpers and `dirichlet_neg_log_likelihood` exist once, in `discrete_transitions.py`; `contingency_belief._optimize_transition_rows` is a wrapper, not a second implementation.
- User-facing documentation listed as tasks is updated, not deferred.
- The None-path bit-for-bit test and `test_em_golden_regression.py` pass without any golden regeneration.
- `uv run ruff check src/`, `uv run ruff format --check src/`, `uv run mypy` are clean with the new module in the mypy list.
