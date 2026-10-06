# Phase 1 — Static-maps HMM: `MultiMapPlaceFieldModel` with EM and SGD fitting

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md) · [contracts](shared-contracts.md)

Ships the complete static-maps model: exact log-space HMM helpers in `utils`, the model
class (k-means initialisation, EM through `run_em`, SGD through `SGDFittableMixin`,
occupancy-ordered labels, Viterbi, per-map rate maps, switch probability, `score`, BIC),
behaviour-alignment helpers, a two-map simulator, the tests, user docs and a real-data
smoke script. After this PR a user can fit K maps to a session and pick K.

**Inputs to read first:**

- `CLAUDE.md` (repo root) — conventions: time axis leads / discrete-state axis trails, float64, `StateSpaceWarning` / `NotFittedError`, behavioural tests, slow marking, NumPy docstrings with shapes.
- `src/state_space_practice/utils.py:1762-1824` — `hmm_viterbi` (the neighbour the new HMM helpers sit next to; note the dynamax attribution block at `:1762-1766` covers Viterbi only), `:1827-1847` `zero_preserving_log` (structural zeros → `-inf`), `:1722-1759` `make_discrete_transition_matrix`, `:1895-1921` `find_permutation`.
- `src/state_space_practice/place_field_model.py:652-821` — `_fit_stationary_glm`: the penalised Poisson GLM the K = 1 model must reproduce (`:742-767` intercept-matching start, `:769-800` Newton loop, `:777-779` `_safe_expected_count` with `self._max_log_count`); `:550-565` `_check_fitted`; `:852-862` `_max_log_count`; `:920-981` `_warn_if_rate_saturated`; `:1196-1416` `fit` (`:1335-1362` snapshot/restore/clear, `:1392-1403` `run_em` call); `:1420-1534` `fit_sgd` delegating to `super().fit_sgd`; `:1547-1621` SGD hooks; `:1886-1979` `score`; `:2005-2049` `n_free_params` / `bic`.
- `src/state_space_practice/switching_point_process.py:1035-1045` `_ridged_newton_direction`, `:1048-1090` `_descent_step`, `:1093-1105` `_warn_newton_fallbacks` — reused Newton primitives; `:3245-3270` the low-occupancy gate in `_m_step_spikes`; `:3455-3468` penalties added to the SGD loss so EM and SGD share one objective; `:3966-4006` snapshot/restore/`run_em` wiring of a switching model.
- `src/state_space_practice/switching_kalman.py:2063-2110` `minimum_state_occupancy`, `warn_low_occupancy_states`; `:2275-2287` discrete-transition MAP and initial-probability M-step.
- `src/state_space_practice/contingency_belief.py:200-222` `get_transition_prior`.
- `src/state_space_practice/point_process_kalman.py:589-610` `_safe_expected_count`.
- `src/state_space_practice/em_driver.py:58-77` `run_em` signature (read the docstring `:78-175` for the rollback policies); `:34-55` `EMResult`.
- `src/state_space_practice/sgd_fitting.py:354-373` mixin protocol, `:384-397` `_prepare_sgd_data`, `:399-417` default `_store_sgd_params` via `_sgd_param_attrs`, `:506-535` `fit_sgd` (kwargs popped at `:536-539`), `:549-550` "no learnable parameters", `:719-721` what the mixin stores before `_finalize_sgd`.
- `src/state_space_practice/parameter_transforms.py:157` `UNCONSTRAINED`, `:225-252` `STOCHASTIC_ROW` (strict positivity check at `:232`; zero-column case at `:243-244`).
- `src/state_space_practice/simulate_data.py:77-206` `simulate_2d_moving_place_field` (`:138-166` the trajectory block to lift; `:164` its only RNG draw; `:195` the Poisson draw).
- `src/state_space_practice/graph_place_field.py:58-92` `GraphBasis`, `:328-376` `graph_design_matrix` (returns `Z, valid`) — for the docstring recipe and the graph-basis test.
- `src/state_space_practice/__init__.py:35-53, 55-76, 78-100` — lazy export registry, `__all__`, type-checking imports.
- `pyproject.toml:134-159` — `[tool.mypy] files`; `CHANGELOG.md:6-8`; `README.md:60-67`.
- Tests to pattern-match: `tests/test_utils.py:10-40` (import block), `:692-712` (`TestDiscreteStateUtilities`, structural-zero Viterbi test); `tests/test_invariances.py:82-88` (`_close`), `:1130-1205` (`TestContingencyBeliefInvariances`, state relabelling with a guard); `tests/test_place_field_model.py:84-95` (`sim_data` fixture), `:304` (`NotFittedError` match string), `:2260-2312` (recovery test shape); `tests/test_oracle_switching_point_process.py:125-126, 238-261` (path enumeration → posteriors); `tests/recovery_helpers.py:28-60, 68-86`; `tests/test_sbc_ranks.py:60-71`; `tests/test_simulate_data.py:181` (`TestSimulate2DMovingPlaceField`); `tests/conftest.py:62-68, 143` (automatic `slow` marking; registry for fitting hidden in helpers).
- `notebooks/real_ca1_data_exploration.py:36-58, 67, 129, 286-293, 305, 327` — the shape of a real-data script: x64 first, gitignored `data/` loader, `head_speed`, 250 Hz binning with `preprocessing.bin_spike_times`, speed interpolation with `interpolate_to_new_times`.

**Contracts referenced:**

- [HMM forward-backward](shared-contracts.md#hmm-forward-backward) — this phase implements it; the 3-D stack branch must already be exact (phase 1b relies on it). Do not weaken.
- [Model attributes](shared-contracts.md#model-attributes) — this phase defines them; every name/shape is asserted by tests.

**Designs referenced:** [A](designs.md#a-exact-hmm-forwardbackward-in-log-space), [B](designs.md#b-per-map-poisson-log-likelihoods), [C](designs.md#c-weighted-penalised-poisson-glm-m-step), [D](designs.md#d-k-means-initialisation-on-windowed-population-rate-maps), [E](designs.md#e-transition-and-initial-state-m-step-occupancy-ordering), [F](designs.md#f-model-class-wiring), [G](designs.md#g-two-map-session-simulator), [H](designs.md#h-exact-path-enumeration-oracle-for-tests), [I](designs.md#i-behavioural-alignment-helpers), [J](designs.md#j-tensor-product-spline-difference-penalty).

## Tasks

- **Capture the simulator baseline before touching `simulate_data.py`.** Run
  `simulate_2d_moving_place_field(total_time=30.0, dt=0.020, arena_size=80.0, peak_rate=25.0, background_rate=1.0, n_interior_knots=3, rng=np.random.default_rng(42))`
  (the `sim_data` fixture arguments, `tests/test_place_field_model.py:84-95`) and the
  function's defaults with `rng=None`, and `np.savez` `position`, `spikes`, `true_center`
  to the scratchpad. After the lift below, assert byte-equality (`np.array_equal`) for both
  calls. Then pin three values in `tests/test_simulate_data.py::TestSimulate2DMovingPlaceField`
  (`position[0]`, `position[-1]`, `int(spikes.sum())` for the fixture call) as a
  permanent guard.

- **Lift the lawnmower trajectory and add the two-map simulator** in
  `src/state_space_practice/simulate_data.py` per [designs G](designs.md#g-two-map-session-simulator):
  `_lawnmower_trajectory(n_time, dt, arena_size, speed, rng)` replaces `:138-166` inside
  `simulate_2d_moving_place_field` (same RNG draw, same order → bit-identical);
  `gaussian_place_field_rate(...)`; `simulate_multi_map_session(...)` with the exact
  signature and return dict in the design. `simulate_data.py` is in the mypy `files` list —
  full annotations, `-> dict` for the return with a documented key list.

- **Add the exact HMM helpers to `utils.py`** per [designs A](designs.md#a-exact-hmm-forwardbackward-in-log-space):
  `HMMPosterior`, `_log_transition_stack`, `_hmm_forward`, `hmm_filter`,
  `hmm_forward_backward`, inserted after `hmm_viterbi` (after `utils.py:1824`). Add the
  `NamedTuple` / `logsumexp` imports if missing. Docstrings document the
  `(K, K)`-or-`(n_time - 1, K, K)` transition input and the index convention. Do not modify
  `hmm_viterbi`.

- **Create `src/state_space_practice/multi_map_place_field.py`.** Module docstring:
  the model equation, the EM (exact E-step, penalised MAP M-steps), the basis-agnostic
  design-matrix contract with both recipes (spline: `build_2d_spline_basis` /
  `evaluate_basis`; graph: `graph_design_matrix` + dropping `~valid` rows), and
  references (Low et al. 2021; Sheintuch et al. 2020; Sanders, Wilson & Gershman 2020;
  Rabiner 1989; Schwarz 1978; Eilers & Marx 1996 — say what each supports, as in
  [overview](overview.md)). Contents, in this order:
  1. `per_map_log_likelihoods` ([B](designs.md#b-per-map-poisson-log-likelihoods)).
  2. `_weighted_poisson_glm_newton`, `update_map_weights`, `_intercept_matching_weights`
     ([C](designs.md#c-weighted-penalised-poisson-glm-m-step)).
  3. `kmeans_map_initialization` ([D](designs.md#d-k-means-initialisation-on-windowed-population-rate-maps)).
  4. `_transition_matrix_map_estimate` ([E](designs.md#e-transition-and-initial-state-m-step-occupancy-ordering)).
  5. `tensor_spline_difference_penalty` ([J](designs.md#j-tensor-product-spline-difference-penalty)).
  6. `switch_probability`, `lagged_correlation`, `circular_shift_null`, `covariate_by_map`
     ([I](designs.md#i-behavioural-alignment-helpers)).
  7. `MultiMapPlaceFieldModel` ([F](designs.md#f-model-class-wiring)): constructor with
     validation (`validate_int`, `validate_scalar`; `penalty` resolved at fit;
     `init_transition_diag` in `(0, 1)`; `init_window_bins >= 1`), `_validate_data`,
     `_initial_responsibilities`, `_initialize_parameters`, `_e_step`, `_m_step`,
     `_update_weights` (occupancy gate), `_snapshot` / `_restore`, `_log_prior`, `fit`,
     `_finalize_fit` (occupancy ordering + `switch_probability_`), the SGD hooks
     (`fit_sgd`, `_prepare_sgd_data`, `_n_timesteps`, `_build_param_spec`, `_sgd_loss_fn`,
     `_sgd_param_attrs`, `_finalize_sgd`), `from_parameters`, `predict_map_posterior`,
     `score`, `predict_rate_map`, `viterbi_path`, `n_free_params`, `bic`, `aic`,
     `__repr__` (n_maps, dt, fitted flag, n_neurons/n_basis when fitted), and a saturation
     warning after `fit` / `fit_sgd` mirroring `place_field_model.py:920-981` (fraction of
     `(bin, neuron)` log-counts at the ceiling under the Viterbi map > 1e-3 →
     `StateSpaceWarning`).
  8. `fit_over_n_maps`, `select_n_maps` ([F](designs.md#f-model-class-wiring), end).

  Class docstring states the objective semantics of `log_likelihood_history_` vs
  `log_likelihood_` ([contract](shared-contracts.md#model-attributes)), the label
  convention (map 0 = most occupied), the float64 requirement (same recipe as
  `place_field_model.py:212-225`), and an `Examples` block fitting K = 2 on
  `simulate_multi_map_session` output. All `fit`-time errors are `ValueError`; unfitted
  access is `NotFittedError` with the message "Call model.fit(design_matrix, spikes) before
  X()". Every jitted helper takes `max_log_count` / `max_iter` / `batch_size` as static
  arguments (hashable Python scalars) so EM iterations reuse one compiled program.

- **Register the module.** `__init__.py`: add `"MultiMapPlaceFieldModel": "multi_map_place_field"`
  to `_LAZY_API` (`:35-53`), the name to `__all__` (`:55-76`), and the `TYPE_CHECKING`
  import (`:78-100`). `pyproject.toml`: add
  `"src/state_space_practice/multi_map_place_field.py"` to `[tool.mypy] files` after the
  `models.py` line (`:147`); run `uv run mypy` and fix until clean (the module imports
  private helpers from non-type-clean modules — `follow_imports = "silent"` handles that;
  annotate every function).

- **Tests — `tests/test_utils.py`** (fast; add to `TestDiscreteStateUtilities` at `:692`,
  import the new names in the block at `:10-40`): the exact-oracle test of
  `hmm_forward_backward` and `hmm_filter` against [designs H](designs.md#h-exact-path-enumeration-oracle-for-tests)
  for a `(K, K)` matrix and a `(n_time - 1, K, K)` stack; pairwise marginalisation
  identities; structural-zero preservation; prior-chain recovery under constant
  likelihoods; Viterbi path equals the enumeration argmax. Put `enumerate_hmm_posterior`
  in `tests/oracles.py` (next to `_logsumexp`, `:437-442`) so the model tests reuse it.

- **Tests — `tests/test_multi_map_place_field.py`** (new). Fixtures: `two_map_session`
  (module scope; `simulate_multi_map_session(n_neurons=8, total_time=300.0, dt=0.02,
  dwell_time=20.0, map_separation=1.0, rng=default_rng(0))`), `fitted_two_map`
  (class scope; `MultiMapPlaceFieldModel(n_maps=2, dt).fit(Z, y, max_iter=50)`),
  `tiny_problem` (T = 6, K = 2, N = 2, B = 3, random weights) for exact checks. Tests listed
  in the validation slice below. Tests calling `.fit(` are auto-marked slow
  (`conftest.py:143`); tests whose fitting hides in a helper (the K-grid test) get explicit
  `@pytest.mark.slow`.

- **Tests — `tests/test_invariances.py`**: new `TestMultiMapPlaceFieldInvariances` after
  `TestContingencyBeliefInvariances` (`:1130-1205`), using `from_parameters(...).predict_map_posterior`:
  map relabelling (LL `1e-12`; posteriors permuted `1e-10`; guard: permuting only the
  weights changes the LL by more than `1e-3`), neuron relabelling, invertible basis change
  (`Z @ M`, `M^{-1} @ weights` per map → identical LL and posteriors).

- **Tests — `tests/test_simulate_data.py`**: `TestSimulateMultiMapSession` (shapes; states
  are a valid Markov sample: empirical stay probability within 0.02 of `1 - dt/dwell_time`;
  `map_separation=0` gives identical centres across maps; `transition_gain > 0` with a
  standardised covariate raises the empirical switch rate at high covariate values) and the
  three pinned values from the baseline task.

- **User-facing docs.** `CHANGELOG.md` `### Added` (`:8`): one bullet for
  `MultiMapPlaceFieldModel` (what it fits, EM + SGD, K selection, behaviour helpers), one
  for `utils.hmm_filter` / `hmm_forward_backward`, one for
  `simulate_data.simulate_multi_map_session`. `README.md` Package layout (`:60-67`): add
  `MultiMapPlaceFieldModel` to the lazily loaded entry points list. Public docstrings are
  written as part of the module task above, not afterwards.

- **Real-data smoke script** `notebooks/multi_map_ca1_smoke.py` (jupytext percent format,
  like `notebooks/real_ca1_data_exploration.py`; formatted but not linted, `pyproject.toml`
  `[tool.ruff.lint] exclude`). Steps: enable x64; load the J16 session through the
  gitignored `data.load_bandit_data.load_neural_recording_from_files` (`:48, :67`); bin
  spikes at `dt = 0.02` with `preprocessing.bin_spike_times`; `select_units` as in `:305`
  region; interpolate `head_speed` with `interpolate_to_new_times` (`:327`); build
  `build_2d_spline_basis(position, n_interior_knots=6)`; `fit_over_n_maps(K ∈ {1, 2, 3})`
  with a 70/30 train/held-out split; print the BIC / held-out table and `select_n_maps`;
  for the selected K: `lagged_correlation(switch_probability_, speed[1:], max_lag=50)`,
  `circular_shift_null(..., n_shuffles=200, min_shift=500)`, `covariate_by_map`; save
  `multi_map_ca1_switch_vs_speed.png` (switch probability and speed over 5 minutes; rate
  maps per map for the six highest-rate units). Not a test; no data is checked in. Record
  runtime and peak memory in the PR description.

## Deliberately not in this phase

- Covariate-dependent transitions (`transition_covariates=`): [phase 1b](phase-1b-speed-gated-transitions.md), gated on `docs/plans/recurrent-switching-transitions/`.
- Per-map weight drift: not in this plan — [overview → Deliberately not in this plan](overview.md#scope-and-dependency-policy), [designs K](designs.md#k-slow-per-map-drift--sketch-not-scheduled).
- `obs_mask=` / multi-session fitting: consumed later from `docs/plans/masks-and-multi-sequence/`; callers drop invalid graph-basis rows themselves for now (documented in the module docstring).
- `identifiability_report()`: `docs/plans/identifiability-diagnostics/`.
- Any change to `PlaceFieldModel`, `update_spike_glm_params`, `contingency_belief`, or `hmm_viterbi` (1b adapts to the recurrent plan's Viterbi stack convention).
- Plotting methods on the model; the smoke script plots inline.
- Effective-degrees-of-freedom BIC.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_utils.py::TestDiscreteStateUtilities::test_hmm_forward_backward_matches_path_enumeration` | `T=6, K=3`, random `pi`, `Z`, log-likelihoods: `smoothed_prob`, `pairwise_prob`, `filtered_prob`, `log_likelihood` match [designs H](designs.md#h-exact-path-enumeration-oracle-for-tests) at `rtol=1e-10`; same with a random `(T-1, K, K)` stack |
| `...::test_hmm_filter_matches_forward_backward_filtered` | `hmm_filter` returns the same `filtered_prob` and `log_likelihood` as `hmm_forward_backward` (bit-identical) |
| `...::test_hmm_pairwise_marginalises_to_smoothed` | `pairwise[t].sum(1) == smoothed[t]`, `pairwise[t].sum(0) == smoothed[t+1]`, all rows sum to 1 (`atol=1e-12`) |
| `...::test_hmm_forward_backward_preserves_structural_zeros` | `Z[0, 1] = 0` → `pairwise[:, 0, 1] == 0` exactly and no NaN; mirrors `:703-712` |
| `...::test_hmm_forward_backward_constant_likelihood_recovers_prior_chain` | constant log-likelihood rows → `smoothed[t] == pi @ Z**t` (`1e-12`); guard: non-constant rows differ |
| `...::test_hmm_viterbi_equals_enumeration_argmax` | `hmm_viterbi` path equals the oracle's max-weight path on the tiny problem (data drawn so no ties: max minus runner-up > 1e-6) |
| `test_multi_map_place_field.py::test_per_map_log_likelihoods_match_scipy` | equals `scipy.stats.poisson.logpmf(y, exp(Z w_k) dt).sum(1)` per map (`rtol=1e-10`); `batch_size=7` on `T=20` gives identical output (batching does not change values) |
| `::test_weighted_glm_newton_matches_scipy_minimize` | weights within `1e-5` of `scipy.optimize.minimize` (L-BFGS-B, `gtol=1e-12`) on the same weighted penalised objective; bins with zero responsibility have no influence (arbitrary counts there leave the solution unchanged to `1e-12`) |
| `::test_intercept_matching_start_reproduces_weighted_mean_rate` | `mean_t exp(Z w0) ≈ weighted mean count / dt` within 5 % for a basis that spans constants |
| `::test_kmeans_initialization_separates_planted_maps` | two planted maps in alternating 2 s blocks → window labels match blocks after `find_permutation` (accuracy 1.0); responsibilities shape `(T, K)`, rows sum to 1, min entry `soft_floor / K`; `n_windows < n_maps` raises `ValueError` |
| `::test_tensor_spline_difference_penalty_annihilates_linear_ramp` | quadratic form is `0` (`atol=1e-12`) for coefficients linear in both grid indices, `> 0` for a Gaussian bump; matrix symmetric PSD; shape `(n**2, n**2)` |
| `::test_switch_probability_and_lagged_correlation` | `switch_probability` of a pairwise stack with known diagonal mass; `lagged_correlation(x, roll(x, 3), 5)` peaks at lag 3 with `corr > 0.99`; `circular_shift_null` mean within `0.05` of 0 on independent noise |
| `::test_covariate_by_map_weighted_moments` | analytic weighted mean/std on a 3-bin example (`1e-12`) |
| `::test_transition_matrix_map_estimate_with_prior_and_empty_row` | pseudo-counts shift the estimate toward the diagonal; a zero-mass row keeps the previous row; rows sum to 1 |
| `::test_validation_errors_and_not_fitted` | wrong spike shape, negative counts, `n_maps=0`, non-symmetric `penalty`, `initial_responsibilities` rows not summing to 1, refit with a different `n_basis` → `ValueError`; `predict_rate_map`, `score`, `viterbi_path`, `bic` before fit → `NotFittedError` matching `"Call model.fit"`; `from_parameters(...).viterbi_path()` → `NotFittedError` |
| `::test_low_occupancy_map_keeps_weights_and_warns` | call `_update_weights` with responsibilities giving map 1 occupancy `< n_basis + 1` → map 1 weights unchanged, map 0 weights changed, `caplog` has the occupancy warning |
| `::test_smoothed_map_posterior_is_calibrated_at_true_parameters` | 20 simulated sessions (60 s, 6 neurons), `from_parameters(true W, Z, pi).predict_map_posterior`: per decile of `smoothed[:, 1]` with ≥ 200 samples, `\|freq(s_t = 1) - mean posterior\| < 0.05`; power guard: the renormalised cube of the posterior fails the same check |
| `::test_k1_reduces_to_stationary_glm` (slow) | static single neuron from `simulate_2d_moving_place_field(drift_speed=0.0)`, `n_maps=1`, `penalty=1.0`, `max_newton_iter=15`, `max_iter=3`: `weights_[0, :, 0]` within `rtol=1e-6` of `PlaceFieldModel._fit_stationary_glm(Z, y, prior_precision=1.0)` (set `n_basis_per_neuron` first) and within `1e-4` of scipy L-BFGS-B; `log_likelihood_` equals `poisson.logpmf` sum (`1e-12`); `smoothed_map_prob_` all ones; `switch_probability_` all zeros; `viterbi_path()` all zeros; `n_free_params == n_basis` |
| `::test_two_map_recovery` (slow, `fitted_two_map`) | Viterbi accuracy after `find_permutation` ≥ 0.95; per-map rate map on a 20×20 grid (`predict_rate_map(evaluate_basis(grid, basis_info))`) vs `gaussian_place_field_rate(grid, centers[:, :, k])` correlation ≥ 0.9 for every neuron with in-arena centre; `transition_matrix_` diagonal within 0.02 of truth; `smoothed_map_prob_.sum(0)` non-increasing (label convention) |
| `::test_segmentation_degrades_gracefully_as_maps_converge` (slow) | `map_separation ∈ (1.0, 0.5, 0.2)`, same seed: accuracies non-increasing (tolerance 0.02); guards `acc[0] ≥ 0.95` and `acc[-1] ≤ acc[0] - 0.1` |
| `::test_em_objective_monotone_and_ll_improves` (slow) | `assert_ll_monotonic(log_likelihood_history_, tol=1e-8 * \|LL\|)`; `converged_` is a bool; `log_likelihood_` of a 30-iteration fit `>` that of a 1-iteration fit from the same `initial_responsibilities`; `len(history) <= max_iter + 1` |
| `::test_bic_and_held_out_select_true_n_maps` (slow, explicit mark) | `fit_over_n_maps(K ∈ {1,2,3})` on a 70/30 split: `select_n_maps(rows, "bic") == 2` and `select_n_maps(rows, "held_out") == 2`; guard `LL(2) - LL(1) > 2 * n_neurons * n_basis * log(n_train)`; `n_free_params` for K = 2 equals `2 N B + 2 + 1` |
| `::test_labels_ordered_by_occupancy_and_init_permutation_invariant` (slow) | fit from `gamma0` and from `gamma0[:, ::-1]` → `weights_`, `transition_matrix_`, `smoothed_map_prob_` identical to `1e-8`; `smoothed_map_prob_.sum(0)` descending |
| `::test_fit_sgd_reaches_em_objective` (slow) | small problem (4 neurons, 60 s): `fit_sgd(num_steps=300)` final `log_likelihood_history_[-1]` within `1e-3` relative of EM's; SGD history improves first→last (`assert_ll_improves`); warm SGD from the EM solution does not decrease the objective by more than `1e-6` relative |
| `::test_score_matches_training_log_likelihood_and_prefers_true_k` (slow) | `score(Z_train, y_train) == log_likelihood_` (`1e-10`); K = 2 model's `score(Z_test, y_test)` > K = 1 model's by at least `n_neurons * log(n_test)` |
| `::test_saturation_warning_fires_on_flooded_bin` | inject a `1000`-count bin with `max_firing_rate_hz=100` → `StateSpaceWarning` mentioning the ceiling (mirrors `place_field_model.py:920-981`) |
| `::test_graph_basis_design_matrix_is_accepted` (skip if `neurospatial` missing) | `graph_design_matrix` output with `~valid` rows dropped fits without error; `predict_rate_map(basis.eigvecs)` shape `(n_bins, N, K)`; a `np.diag(basis.eigvals)` penalty is accepted |
| `test_invariances.py::TestMultiMapPlaceFieldInvariances::test_map_relabelling` | LL unchanged (`1e-12`); `smoothed[:, perm]`, `pairwise[:, perm][:, :, perm]`, `filtered[:, perm]` (`1e-10`); guard: permuting weights only changes LL by `> 1e-3` |
| `...::test_neuron_relabelling` | permuting neurons and `weights_[perm]` leaves LL and posteriors unchanged (`1e-12`) |
| `...::test_basis_change` | `Z @ M`, `solve(M, weights)` per map: LL and posteriors unchanged (`1e-10`) |
| `test_simulate_data.py::TestSimulateMultiMapSession::*` | shapes; empirical stay probability within 0.02 of `1 - dt/dwell_time`; `map_separation=0` → identical centres; positive `transition_gain` raises switch rate in the top covariate quartile vs the bottom; `states` in `range(n_maps)` |
| `test_simulate_data.py::TestSimulate2DMovingPlaceField::test_seeded_values_unchanged` | pinned `position[0]`, `position[-1]`, `spikes.sum()` from the baseline task (exact) |
| `test_package.py` (existing, parametrised) | `MultiMapPlaceFieldModel` resolves to `state_space_practice.multi_map_place_field` and is in `__all__` — no new test needed, confirm it runs |

Mark slow / integration tests explicitly where the fit is inside a helper
(`@pytest.mark.slow`); the others are marked automatically by `conftest.py:143`.

## Fixtures

- Simulated only; no real data is checked in. `simulate_multi_map_session` (new, [designs G](designs.md#g-two-map-session-simulator)) with fixed `np.random.default_rng` seeds; `simulate_2d_moving_place_field(drift_speed=0.0)` for the K = 1 parity case.
- `tiny_problem`: hand-built `T=6, K=2, N=2, B=3` arrays for exact-oracle checks (no simulator).
- `enumerate_hmm_posterior` in `tests/oracles.py` ([designs H](designs.md#h-exact-path-enumeration-oracle-for-tests)).
- Graph-basis test uses `neurospatial`'s `make_w_maze` as `tests/test_graph_place_field.py:14-15, 36-40` does; `pytest.importorskip("neurospatial")`.
- Real-data smoke: J16 session via the gitignored `data/` loader, run manually (`PYTHONPATH=. uv run --no-sync python notebooks/multi_map_ca1_smoke.py`).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind). (None are removed here; confirm `simulate_2d_moving_place_field` is byte-identical via the baseline task.)
- User-facing documentation listed as tasks is updated, not deferred.
- Additionally: `uv run mypy` is clean with the new module in `[tool.mypy] files`; `uv run ruff check src/` and `ruff format --check` pass; the fast suite (`-m "not slow"`) still finishes in under a minute; the recovery fit's runtime is recorded in the PR.
