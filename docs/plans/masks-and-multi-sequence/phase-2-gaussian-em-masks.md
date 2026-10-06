# Phase 2 — Masked Gaussian EM end-to-end: M-steps, switching filter, oscillator models

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d2--masked-gaussian-m-step-imputation-form)

Ships `obs_mask=` on `kalman_maximization_step`, `switching_kalman_filter`,
`switching_kalman_maximization_step`, and on `fit` / `fit_sgd` of the oscillator
`BaseModel` family (`CommonOscillatorModel`, `CorrelatedNoiseModel`,
`DirectedInfluenceModel`). The oscillator models are implemented on the
switching filter and M-step (`oscillator_models.py:913`, `:1000`), so those two
functions are the only route to masked oscillator fits; they are in scope for
that reason and no other (the switching *point-process* models stay out).

**Inputs to read first:**

- `src/state_space_practice/kalman.py:1364-1396` (`measurement_cov_residual_form`),
  `:1448-1551` (`kalman_maximization_step`), `:1554-1628`
  (`_kalman_maximization_step`; observation block `:1571-1581`, transition block
  `:1583-1619`) — the reference M-step that gets the imputation branch.
- `src/state_space_practice/switching_kalman.py:686-790` (`_first_timestep_kalman_update`,
  vmapped update `:747-758`), `:793-1076` (`switching_kalman_filter`; pair update
  `:937-945`, first step `:1000-1007`, scan `:1032-1042`) — where the mask enters
  the switching E-step.
- `src/state_space_practice/switching_kalman.py:2042-2060` (`weighted_sum_of_outer_products`),
  `:2150-2300` (`_switching_kalman_m_step_inner`; observation block `:2206-2251`,
  `gamma2` `:2222-2229`), `:2303-2597` (`switching_kalman_maximization_step`;
  occupancy gate `:2514-2539`) — the M-step to refactor into the two helpers of
  [D2](designs.md#d2--masked-gaussian-m-step-imputation-form).
- `src/state_space_practice/oscillator_models.py:606-646` (`_warm_initialize_states`),
  `:648-671` (`_prepare_windows`), `:673-730` (`_apply_warm_init`), `:732-803`
  (`_seed_state_parameters_from_windows`; `cls._e_step` `:780`, `cls._m_step`
  `:792`), `:886-970` (`_e_step`), `:972-1030` (`_m_step`), `:1032-1064`
  (`_m_step_fixed_and_previous_params`), `:1082-1142` (`fit`), `:1146-1194`
  (`fit_sgd`; `_sgd_n_time` `:1186`), `:1219-1220` (`_finalize_sgd`).
- Subclass overrides that must forward the mask: `CommonOscillatorModel.fit`
  `:1382-1416`, `_sgd_loss_fn` `:1444-1462`; `CorrelatedNoiseModel.fit`
  `:1707-1741`, `_sgd_loss_fn` `:1777-1811`; `DirectedInfluenceModel._m_step`
  `:2008-2021`, `_m_step_reparameterized` `:2023-2129` (M-step call `:2054-2069`),
  `fit` `:2131-2165`, `fit_sgd` `:2199-2251`, `_sgd_loss_fn` `:2253-2313`.
- `src/state_space_practice/sgd_fitting.py:384-397` (`_prepare_sgd_data`), `:536-544`
  (settings popped, data kwargs forwarded), `:592-593` (`_split_leaves`: a `None`
  leaf is static, a bool array is dynamic) — why `obs_mask` can flow as a data
  kwarg.
- `src/state_space_practice/tests/oracles.py:294-353`
  (`lgssm_expected_complete_log_likelihood`), `:473-646`
  (`switching_lgssm_exact_posterior`; per-path call `:528-537`).
- `src/state_space_practice/tests/test_oracle_kalman.py:316-473` (Q-function,
  `_fd_gradient`, `TestKalmanMStepMaximisesExactQ`).
- `src/state_space_practice/tests/test_oracle_switching_kalman.py:73` (`_random_switching_model`),
  `:129` (`_run_library`), `:200` (`_assert_filter_exact`), `:483`
  (`_m_step_from_exact_stats`), `:528` (`TestSwitchingMStepMaximisesExactQ`).
- `src/state_space_practice/tests/test_oscillator_models.py:608` (`TestEMAlgorithm`),
  `:921` (`TestSingleDiscreteState`), `:1314` (`TestOscillatorModelInputValidation`),
  `:1941` (`TestCommonOscillatorSGDFitting`); `tests/conftest.py:680-730`
  (`assert_em_rolls_back_on_ll_decrease`).
- `src/state_space_practice/tests/test_em_golden_regression.py:263-292`
  (`_fit_common_oscillator`, `_fit_directed_influence`), `:416-429` (tolerances).

**Contracts referenced:**

- [C1](shared-contracts.md#c1--obs_mask-argument), [C2](shared-contracts.md#c2--masked-bin-semantics-e-step-and-log-likelihood)
  (item 5 for the switching filter), [C3](shared-contracts.md#c3--masked-bin-semantics-m-step-statistics)
  (items 1, 2, 4 — implemented here; do not weaken item 1's "bins with no
  observed entry are excluded"), [C5](shared-contracts.md#c5--backwards-compatibility-off-means-the-old-code-runs),
  [C8](shared-contracts.md#c8--sgdfittablemixin_n_timesteps-under-masks-and-sequences)
  (masked bins only; sequences come in phase 4).

**Designs referenced:** [D2](designs.md#d2--masked-gaussian-m-step-imputation-form),
[D6 (augmented-Q oracle, per-path masks)](designs.md#d6--oracle-extensions-for-masks-and-sequences).

## Tasks

- **`kalman.py`: `masked_observation_moments`, `_masked_observation_statistics`,
  `_masked_measurement_scatter`** exactly as in
  [D2](designs.md#d2--masked-gaussian-m-step-imputation-form), and the static
  branch in `_kalman_maximization_step` (`kalman.py:1571-1581`): mask `None` →
  the existing four statements; otherwise the imputed statistics with
  `n_bins = sum(any(mask, 1))` as the `R` normaliser. `kalman_maximization_step`
  (`:1448-1551`) gains keyword-only `obs_mask=None`, canonicalised with
  `validate_observation_mask`, and passes it to the jitted inner function.
  Docstring: a *Notes* paragraph stating the complete-data convention of
  [C3](shared-contracts.md#c3--masked-bin-semantics-m-step-statistics) and citing
  Shumway & Stoffer, *Time Series Analysis and Its Applications*, §6.4
  (missing-data modifications of the EM algorithm) for the imputation form.
  `measurement_cov_residual_form` (`:1364-1396`) is left as the unmasked helper.

- **`switching_kalman.py`: mask in the filter.** `switching_kalman_filter`
  (`:793-1076`) gains keyword-only `obs_mask: jax.Array | None = None` (the
  function is `@jax.jit`; canonicalise with `validate_observation_mask` inside —
  static checks only — and zero-fill `obs` with `jnp.where`). Pass `obs_mask[0]`
  to `_first_timestep_kalman_update` (new trailing parameter, forwarded to the
  vmapped `kalman_measurement_update` with `in_axes` `None`, `:747-758`), and
  `obs_mask[1:]` as a second scan input (`:1032-1042`) forwarded to
  `_kalman_filter_update_per_discrete_state_pair` (`:937-945`) as the eighth
  argument. `None` → existing scan over `obs[1:]` only.

- **`switching_kalman.py`: mask in the M-step.** Refactor
  `_switching_kalman_m_step_inner` (`:2150-2300`) into the two helpers
  `_switching_observation_statistics` / `_switching_measurement_scatter` of
  [D2](designs.md#d2--masked-gaussian-m-step-imputation-form) whose unmasked
  branches are the current einsum statements verbatim; add static
  `use_obs_mask` and traced `obs_mask`, `previous_measurement_matrix`,
  `previous_measurement_cov` parameters (the E-step's `H`, `R`, needed for
  imputation; when `use_obs_mask` is False pass zeros and never read them).
  `gamma2` keeps the unmasked `smoother_discrete_state_prob`. Public
  `switching_kalman_maximization_step` (`:2303-2597`) gains keyword-only
  `obs_mask=None`; when given, it requires `previous_params["measurement_matrix"]`
  and `["measurement_cov"]` (raise `ValueError` otherwise — the oscillator
  models always pass them, `oscillator_models.py:1058-1063`), and the occupancy
  gate (`:2514-2539`) uses the mask-weighted observation occupancy for `H`/`R`
  and the unmasked transition occupancy for `A`/`Q`.

- **Oscillator `BaseModel`: thread the mask.** `_e_step(observations, obs_mask=None)`
  (`oscillator_models.py:886-970`) passes it to `switching_kalman_filter`;
  `_m_step(observations, obs_mask=None)` (`:972-1030`) passes it to the M-step;
  `fit(..., *, obs_mask=None)` (`:1082-1142`) canonicalises once
  (`validate_observation_mask` with `n_obs = self.n_sources`), forwards to warm
  init, and closes it into the `run_em` callables (`:1130-1140`);
  `fit_sgd(..., *, obs_mask=None)` (`:1146-1194`) sets
  `self._sgd_n_time = n_observed_bins` (count of bins with any observed channel,
  computed here inline; phase 4 moves the helper to `sequences.py`) and passes
  `obs_mask=obs_mask` through `super().fit_sgd(observations, obs_mask=obs_mask, ...)`
  as a data kwarg; `_finalize_sgd(observations, obs_mask=None)` (`:1219`).
  Forward in every subclass override listed under *Inputs to read first*
  (`fit` of COM/CNM/DIM, DIM `fit_sgd`, the three `_sgd_loss_fn(params, observations, obs_mask=None)`,
  DIM `_m_step` / `_m_step_reparameterized(observations, obs_mask=None)` →
  `switching_kalman_maximization_step(..., obs_mask=obs_mask)`).
  `_seed_state_parameters_from_windows` (`:732-803`) calls
  `cls._e_step(self, observations, obs_mask)` / `cls._m_step(self, observations, obs_mask)`.

- **Warm initialisation under masks** (`_warm_initialize_states`, `:606-646`):
  build the window features from an imputed copy of the observations — masked
  entries replaced by their channel's mean over observed entries — and drop
  windows with no observed bin from `features` / the GMM (and from the
  `per_state_means` averages, `:707-714`); `obs_var` (`:719`) over observed
  entries only. This is a heuristic initialiser, not an estimator; say so in a
  comment. `_seed_state_parameters_from_windows` then runs the real masked
  E/M-step, so the seeded parameters are consistent with the model.

- **Tests** (validation slice): extend `oracles.py` with
  `lgssm_dense_posterior_with_observations` and
  `lgssm_expected_complete_log_likelihood_augmented`
  ([D6](designs.md#d6--oracle-extensions-for-masks-and-sequences)), pass
  `obs_mask` through `switching_lgssm_exact_posterior`; new tests in
  `test_oracle_kalman.py`, `test_kalman.py::TestKalmanMStepMathCorrectness`
  (`:1438`), `test_oracle_switching_kalman.py`, `test_switching_kalman.py`,
  `test_oscillator_models.py`.

- **User-facing docs:** CHANGELOG `### Added` ("masked EM for the linear-Gaussian
  and switching models; `obs_mask` on `CommonOscillatorModel` /
  `CorrelatedNoiseModel` / `DirectedInfluenceModel` `fit` and `fit_sgd`");
  docstrings of `kalman_maximization_step`, `switching_kalman_filter`,
  `switching_kalman_maximization_step`, `BaseModel.fit` / `fit_sgd` (the
  *Parameters* entry plus the sentence "bins with no observed channel are
  predict-only steps and contribute nothing to the observation statistics").
  Extend the README "Missing observations" subsection (phase 1) with one line on
  the oscillator models.

- **Confirm nothing moved:** `test_em_golden_regression.py` cases
  `common_oscillator` and `directed_influence` (`:263-292`) reproduce at their
  `_EXACT` tolerances (`:422-423`); `TestSwitchingMStepMaximisesExactQ`
  (`test_oracle_switching_kalman.py:528`) unchanged.

## Deliberately not in this phase

- Point-process models (phase 3) and any leading sequence axis (phases 4-5).
- The switching point-process models (`SwitchingSpikeOscillatorModel`,
  `point_process_models.py`, `switching_point_process.py`): they call
  `switching_kalman_maximization_step` with dummy observations and
  `estimate_measurement_params=False` (`point_process_models.py:1351`,
  `switching_point_process.py:3073`) and are untouched. Revisit trigger:
  `docs/plans/multi-map-place-fields/` phase 2 needing masked spike-oscillator
  fits.
- `switching_kalman_viterbi` (`switching_kalman.py:1079`): no mask (not on any
  model's fit path). Revisit if a consumer appears.
- Per-neuron mask handling in warm-init GMM beyond mean imputation.
- A faster masked M-step for wide observations (per-bin `n_obs x n_obs` solve;
  see overview risks). Revisit trigger: a user fit with `n_obs > 64` and
  per-channel masks where the M-step exceeds the E-step's wall time.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_oracle_kalman.py::TestMaskedKalmanMStep::test_gradient_of_augmented_q_vanishes` (Hypothesis `max_examples=6`; n=2, m=3, T=6, random mask with ≥1 partial and ≥1 fully masked bin) | The masked `kalman_maximization_step` output zeroes the finite-difference gradient of `Q_aug` (all six parameters, `atol=1e-6`); a random θ has gradient `> 1e-2` (guard). |
| `test_oracle_kalman.py::TestMaskedKalmanMStep::test_masked_em_is_monotone_in_exact_observed_ll` (`slow`) | Three masked EM iterations: the exact observed-data LL from `lgssm_dense_posterior(..., obs_mask)` is strictly increasing. |
| `test_kalman.py::TestKalmanMStepMathCorrectness::test_masked_m_step_no_mask_is_bit_identical` | `kalman_maximization_step(..., obs_mask=None)` == old call (`assert_array_equal`, all six outputs); all-`True` mask → `rtol=1e-12`. |
| `test_kalman.py::TestKalmanMStepMathCorrectness::test_masked_m_step_diagonal_r_closed_form` | With diagonal `R` the returned `R` diagonal equals `(1/N) Σ_t e_t [m_ti (r² + hPh) + (1 − m_ti) R_old,ii]` computed by hand from the smoother moments (`rtol=1e-10`); fully masked bins are excluded (N counts observed bins). Guard: `N < T`. |
| `test_kalman.py::TestKalmanMStepMathCorrectness::test_masked_m_step_dynamics_unchanged_by_mask` | `A`, `Q`, `init_mean`, `init_cov` from the masked M-step equal those of the unmasked M-step run on the *same smoother moments* (`assert_array_equal`): masks touch only `H`/`R`. |
| `test_oracle_switching_kalman.py::test_masked_switching_filter_matches_path_enumeration` (K=2, n=2, m=2, T=5, one fully masked bin, two partially masked) | Filtered means/covs/discrete probs and LL match the masked exact posterior via `_assert_filter_exact`; at the fully masked bin the discrete posterior equals `prev @ Z` (`rtol=1e-12`). |
| `test_oracle_switching_kalman.py::TestSwitchingMStepMaximisesExactQ::test_masked_h_r_reduce_to_kalman_m_step_at_k_1` | K=1: masked switching M-step `H`, `R` equal `kalman_maximization_step(obs_mask=...)` `H`, `R` fed the same smoother moments (`rtol=1e-10`). |
| `test_switching_kalman.py::TestSwitchingKalmanFilterProperties::test_mask_none_is_bit_identical` | `switching_kalman_filter(...)` vs `(..., obs_mask=None)` and all-`True`: `assert_array_equal` on all seven outputs. |
| `test_switching_kalman.py::TestMStepProperties::test_masked_m_step_requires_previous_measurement_params` | `obs_mask` without `previous_params` measurement entries raises `ValueError`. |
| `test_oscillator_models.py::TestEMAlgorithm::test_fit_with_all_true_mask_matches_unmasked` (`slow`) | `CommonOscillatorModel.fit(obs, obs_mask=ones)` LL history equals `fit(obs)` at `rtol=1e-10` (the golden tolerance), same iteration count. |
| `test_oscillator_models.py::TestEMAlgorithm::test_fit_with_dropped_channel_and_gap` (`slow`, one COM and one DIM case) | Channel 1 masked for 30% of bins plus a 20-bin fully masked gap: LL history finite and non-decreasing until stop; smoothed covariance trace inside the gap exceeds the trace at its edges (guard the gap is informative); `decode()` returns `(n_time,)`. |
| `test_oscillator_models.py::TestEMAlgorithm::test_masked_fit_ignores_nan_in_masked_entries` (`slow`) | NaN at masked entries → same LL history as zeros there (`assert_array_equal`). |
| `test_oscillator_models.py::TestOscillatorModelInputValidation::test_obs_mask_shape_errors` | Wrong mask shapes / dtype raise `ValueError` before any E-step (inside `pytest.raises`, stays fast). |
| `test_oscillator_models.py::TestCommonOscillatorSGDFitting::test_fit_sgd_mask_normalises_by_observed_bins` (`slow`) | `fit_sgd(obs, obs_mask=m, num_steps=3)` runs; `model._n_timesteps == int(m.any(1).sum())`; `log_likelihood_history_` finite. |
| `test_em_golden_regression.py` (existing, `slow`) | `common_oscillator`, `directed_influence` unchanged at `_EXACT`. |

## Fixtures

- Reuse `_simulate_lgssm` / `random_mask` (phase 1) and `_random_switching_model`
  (`test_oracle_switching_kalman.py:73`); add `obs_mask` to `_run_library`
  (`:129`).
- Oscillator tests use the module-scoped observation fixtures already in
  `test_oscillator_models.py:32-100` plus a deterministic mask built from
  `np.random.default_rng(0)`.
- No real data.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind) — the refactor of `_switching_kalman_m_step_inner` must leave one implementation of the observation statistics, with the unmasked branch verbatim.
- User-facing documentation listed as tasks is updated, not deferred.
