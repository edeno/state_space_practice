# Phase 3 — Observation masks in the point-process filters, smoothers and models

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d3--masked-laplace-ekf-update)

Ships `obs_mask=` on `stochastic_point_process_filter`,
`stochastic_point_process_smoother`, `glm_laplace_update` (dense and
block-diagonal paths) and on `fit` / `fit_sgd` / `score` of `PointProcessModel`
and `PlaceFieldModel`. A masked `(bin, neuron)` contributes nothing to the
Fisher score, information or log-likelihood; a fully masked bin is predict-only.
The dynamics M-steps are unchanged (every bin is a latent state); the
place-field warm-start GLM gets mask weights.

**Inputs to read first:**

- `src/state_space_practice/point_process_kalman.py:96-145` (`_validate_public_inputs`;
  count validation `:140`), `:736-856` (`_fisher_scoring_line_search`), `:927-1212`
  (`_point_process_laplace_update`; `_neg_log_posterior` `:1093-1103`,
  `_fisher_step_at` `:1105-1143`, single step `:1145-1162`, LL `:1183-1190`,
  Laplace normaliser `:1192-1208`), `:1215-1265` (`GLMFamily`, `poisson_family`),
  `:1284-1297` (Bernoulli family), `:1333-1464` (`glm_laplace_update`).
- `src/state_space_practice/point_process_kalman.py:1467-1734`
  (`stochastic_point_process_filter`; validation `:1635-1646`, block dispatch
  `:1657-1703`, single-neuron promotion `:1708-1710`, impl call `:1714-1726`),
  `:1737-1866` (`_stochastic_point_process_filter_impl`; `_step` `:1800-1846`, scan
  `:1849-1864`), `:1869-1979` (`_block_diagonal_forward_core`; `_step_one_neuron`
  `:1915-1955`, vmap `:1977-1979`), `:2033-2122` (`_block_diagonal_smoother_core`),
  `:2152-2266` and `:2269-2389` (block filter / smoother wrappers), `:2392-2602`
  (`stochastic_point_process_smoother`; validation `:2507-2518`, dispatch
  `:2526-2569`, filter call `:2571-2588`), `:2605-2641` (backward pass — no change).
- `src/state_space_practice/point_process_kalman.py:2928-3371` (`PointProcessModel`;
  `_e_step` `:3065-3105`, `_m_step` `:3107-3148`, `fit` `:3150-3230`, `fit_sgd`
  `:3234-3288` with `_sgd_n_time` `:3269`, `_sgd_loss_fn` `:3314-3336`,
  `_finalize_sgd` `:3348-3371`).
- `src/state_space_practice/place_field_model.py:652-821` (`_fit_stationary_glm`;
  Newton step `:772-783`, initial guess `:788-790`, Laplace covariance `:794-800`),
  `:823-850` (`_warm_start_parameters`), `:920-981` (`_warn_if_rate_saturated`),
  `:983-1024` (`_e_step`), `:1026-1137` (`_m_step` — dynamics only, unchanged),
  `:1196-1416` (`fit`; validation `:1264-1278`, warm start `:1293-1298`, `run_em`
  `:1392-1403`), `:1420-1534` (`fit_sgd`; `_sgd_n_time` `:1487`), `:1574-1609`
  (`_sgd_loss_fn`), `:1623-1658` (`_finalize_sgd`), `:1886-1979` (`score`).
- `src/state_space_practice/utils.py:1016-1041` (`validate_count_array`).
- `src/state_space_practice/tests/test_likelihood_identities.py:583-612`
  (`_pp_problem`, `_laplace_terms`), `:617-652`
  (`test_ll_is_sum_of_laplace_evidence_terms`), `:672-707`
  (`test_block_diagonal_path_ll_identity`).
- `src/state_space_practice/tests/test_oracle_point_process.py:270`
  (`_laplace_filter_reference` — independent NumPy Laplace recursion), `:366`
  (`TestLaplaceFilterComputesItsApproximation`), `:931`
  (`TestStationaryGLMFitIsStationary`).
- `src/state_space_practice/tests/test_point_process_kalman.py:183`
  (`TestStochasticPointProcessFilter`), `:889` (`TestPointProcessModel`), `:3208`
  (`TestBlockDiagonalFilterEquivalence`), `:3630` (`TestBlockDiagonalSmootherEquivalence`).
- `src/state_space_practice/tests/test_place_field_model.py:145` (`TestPlaceFieldModelFit`),
  `:558` (`TestPlaceFieldModelScore`), `:923` (`TestMultiNeuron`), `:1246`
  (`TestWarmStart`), `:1735` (`TestBlockDiagonalDispatch`).
- `src/state_space_practice/tests/test_glm_laplace.py:36` (`TestPoissonParity`), `:108`
  (`TestBernoulliUpdate`).
- `src/state_space_practice/tests/test_em_golden_regression.py:239-260`
  (`_fit_point_process`, `_fit_place_field`), `:418-421` (tolerances).

**Contracts referenced:**

- [C1](shared-contracts.md#c1--obs_mask-argument) — for spikes `n_obs = n_neurons`;
  a 1-D `spike_indicator` pairs with a `(n_time,)` mask. NaN allowed in masked
  entries; family-specific observation validation on unmasked entries only (counts for Poisson/NB, continuous support for ZIG).
- [C2](shared-contracts.md#c2--masked-bin-semantics-e-step-and-log-likelihood)
  items 2-4, [C3](shared-contracts.md#c3--masked-bin-semantics-m-step-statistics)
  items 2-3, [C4](shared-contracts.md#c4--validate_observation_mask),
  [C5](shared-contracts.md#c5--backwards-compatibility-off-means-the-old-code-runs),
  [C8](shared-contracts.md#c8--sgdfittablemixin_n_timesteps-under-masks-and-sequences).

**Designs referenced:** [D3](designs.md#d3--masked-laplace-ekf-update).

## Tasks

- **Mask weights in the two Laplace updates** as in
  [D3](designs.md#d3--masked-laplace-ekf-update): add `obs_mask_t` to the
  legacy update and `obs_mask` to `glm_laplace_update`. Zero-fill observations
  before family evaluation. Reuse/add `GLMFamily.loglik_per_obs`, supplying
  Poisson/Bernoulli and consuming the NB/ZIG implementations when available;
  scalar likelihood signatures stay unchanged. Use the family score, tile the
  N-element observation mask over predictor groups for score/information, and
  reduce the N normalized per-observation likelihood terms for masked line
  search and evidence. Unmasked calls keep their existing expressions. Custom
  families without this callback raise `ValueError` for masked calls. Update
  overloads/docstrings and use family-specific validation on sanitized data.

- **Dense filter path.** `_stochastic_point_process_filter_impl` (`:1737-1866`)
  gains keyword-only `obs_mask=None`; when given, the scan carries
  `(design_matrix, spike_indicator, obs_mask)` and `_step` passes `obs_mask_t`
  to the Laplace update; `None` → the existing scan. `stochastic_point_process_filter`
  (`:1467-1734`) gains keyword-only `obs_mask=None`: canonicalise with
  `validate_observation_mask` (`n_obs = spike_indicator.shape[1]` after the
  single-neuron promotion `:1708-1710`, so a `(n_time,)` mask promotes to
  `(n_time, 1)` alongside the spikes), pass to `_validate_public_inputs` (new
  parameter; `_observation_validator(family)(jnp.where(mask, spikes, 0), ...)`, `:140`) and
  to the impl / block path.

- **Block-diagonal path.** `_block_diagonal_forward_core` (`:1869-1979`) and
  `_block_diagonal_smoother_core` (`:2033-2122`) gain `obs_mask` (positional,
  after `spike_indicator`; `None` allowed) with `in_axes` `1` in the neuron
  `vmap` (`:1977-1979`; `None` when absent) and a per-bin `obs_mask_t = jnp.atleast_1d(m_t)`
  in `_step_one_neuron` (`:1915-1955`). `_run_forward_block_diagonal`,
  `_stochastic_point_process_filter_block_diagonal`,
  `_stochastic_point_process_smoother_block_diagonal` forward it.
  `stochastic_point_process_smoother` (`:2392-2602`) gains keyword-only
  `obs_mask=None`, validates once and forwards to both paths.

- **`PointProcessModel`** (`:2928-3371`): `fit(..., *, obs_mask=None)`,
  `fit_sgd(..., *, obs_mask=None)`, `_e_step(design, spikes, obs_mask=None)`,
  `_sgd_loss_fn(params, design, spikes, obs_mask=None)`,
  `_finalize_sgd(design, spikes, obs_mask=None)`; `fit` uses the selected family's validator on
  unmasked entries (`:3180`); `fit_sgd` sets `_sgd_n_time` to the number of bins
  with any observed neuron (`:3269`) and forwards `obs_mask=obs_mask` as a data
  kwarg. `_m_step` (`:3107-3148`) unchanged — state in its docstring that masked
  bins remain latent states and enter the dynamics statistics.

- **`PlaceFieldModel`**: `fit` / `fit_sgd` / `score` gain keyword-only
  `obs_mask=None` (validated against `(n_time, n_neurons)` before the
  single-neuron squeeze `:1285-1286`, `:1491-1492`, `:1919-1947`, and squeezed
  alongside); `_e_step(design, spikes, obs_mask=None)` (`:983-1024`) forwards;
  `_sgd_loss_fn` / `_finalize_sgd` forward; `_fit_stationary_glm(Z_base, spikes, obs_mask=None, ...)`
  (`:652-821`) uses per-neuron float weights `w` in the Newton step
  (`grad = Z^T (w (mu - y)) + prior w`, `hess = Z^T diag(w mu) Z + prior`,
  `:780-781`, `:798`) and in the intercept-matching guess
  (`mean_count = (Σ w y + 1) / Σ w`, `:788`); `_warm_start_parameters` forwards
  `obs_mask[window]`; `_warn_if_rate_saturated` (`:920-981`) counts only
  observed `(bin, neuron)` entries. `_m_step` (`:1026-1137`) unchanged.

- **Tests** (validation slice): extend `_laplace_filter_reference`
  (`test_oracle_point_process.py:270`) with an `obs_mask` that multiplies the
  per-neuron score, Fisher weight and log-pmf terms; extend `_laplace_terms`
  callers in `test_likelihood_identities.py` to skip masked neurons.

- **User-facing docs:** CHANGELOG `### Added` ("`obs_mask` on the point-process
  filters/smoothers, `PointProcessModel` and `PlaceFieldModel` `fit` / `fit_sgd`
  / `score`; masked neurons contribute nothing; fully masked bins are
  predict-only"); *Parameters* entries in the six public docstrings; extend the
  README "Missing observations" subsection with a `PlaceFieldModel` line
  (tracking loss → `obs_mask` of shape `(n_time,)`).

- **Confirm nothing moved:** golden cases `point_process_glm` and the three
  `place_field_*` cases (`test_em_golden_regression.py:387-394`) at `_EXACT`;
  `TestPoissonParity`, `TestBlockDiagonalFilterEquivalence`,
  `TestBlockDiagonalSmootherEquivalence` unchanged.

## Deliberately not in this phase

- Leading sequence axis for `PointProcessModel` / `PlaceFieldModel`: phase 5.
- `PositionDecoder`, `SwitchingSpikeOscillatorModel`, the switching
  point-process oscillator models (`point_process_models.py`),
  `coupling_ekf.py` / `hamiltonian_core.py` callers of `glm_laplace_update`
  (`coupling_ekf.py:96`, `hamiltonian_core.py:179` — they never pass a mask and
  are untouched), `models.stochastic_point_process_filter` (deprecated,
  `models.py:54-131`). Revisit trigger: `docs/plans/multi-map-place-fields/`
  phase 2.
- Deriving a mask from `bin_spike_times` gaps (`place_field_model.py:1140`):
  users construct masks.
- `steepest_descent_point_process_filter` (`point_process_kalman.py:2851`).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_noncanonical_masks_match_observation_removal[nb,zig]` | Once the GLM families land, remove one of at least two observations explicitly and compare against masking it, including NaN placeholders, matching reduced dispersion/shape/location arrays and predictor rows (ZIG removes both corresponding predictor rows). Check posterior moments, evidence, latent gradients and continuous-data validation. Fully masked bins return the prediction with zero evidence; unmasked invalid observations still raise. |
| `test_custom_family_mask_requires_observation_terms` | A four-field custom family works without a mask and raises a clear `ValueError` when masked without `loglik_per_obs`; it does not receive an unexpected `weights` keyword. |
| `test_oracle_point_process.py::TestLaplaceFilterComputesItsApproximation::test_masked_filter_matches_independent_recursion` (parametrised `max_newton_iter` 1 and 3) | Masked filter means/covs/LL match the masked NumPy Laplace recursion to roundoff (same tolerances as the unmasked test at `:374`). Guard: mask has ≥1 fully masked bin and ≥1 bin with a subset of neurons masked. |
| `test_likelihood_identities.py::TestPointProcessIdentities::test_masked_ll_is_sum_of_laplace_evidence_terms` | LL == Σ_t Laplace terms over *observed* neurons only, 0 at fully masked bins (`rtol=1e-10`); the per-bin normaliser terms (quad, logdets) are unchanged by the mask. |
| `test_likelihood_identities.py::TestPointProcessIdentities::test_masked_block_diagonal_path_ll_identity` (`slow`, compile-dominated) | Same identity on the block path with a per-neuron mask. Guard: off-block covariance exactly zero. |
| `test_point_process_kalman.py::TestStochasticPointProcessFilter::test_mask_none_is_bit_identical` | `stochastic_point_process_filter(...)` vs `(..., obs_mask=None)` and all-`True`: `assert_array_equal` on means, covs, LL (dense and block paths). |
| `test_point_process_kalman.py::TestStochasticPointProcessFilter::test_fully_masked_bin_is_predict_only` | Mean equals `A m` (`assert_array_equal`), cov equals `symmetrize(A P A^T + Q)` at `rtol=1e-10` (Cholesky round trip), LL contribution 0 to 1e-12 (chain-rule restart around the bin). |
| `test_point_process_kalman.py::TestStochasticPointProcessFilter::test_masked_neuron_has_no_influence` | Masking neuron `j` at bin `t` gives the same posterior as deleting neuron `j`'s row from the design and spikes at that bin (`rtol=1e-12`, dense path, `max_newton_iter=1` and 3). |
| `test_point_process_kalman.py::TestStochasticPointProcessFilter::test_nan_in_masked_spikes_is_ignored_and_unmasked_nan_raises` | NaN at masked entries → `assert_array_equal` with zeros there; NaN or negative at an unmasked entry raises `ValueError` from count validation. |
| `test_point_process_kalman.py::TestBlockDiagonalFilterEquivalence::test_masked_block_equals_masked_dense` (`max_newton_iter=1`) | Block and dense paths agree to roundoff with a per-neuron mask; masking neuron `j` changes only block `j` of the block covariances (other blocks `assert_array_equal` to the unmasked run). |
| `test_glm_laplace.py::TestPoissonParity::test_masked_glm_update_matches_masked_legacy_update` | `glm_laplace_update(..., obs_mask=m)` equals `_point_process_laplace_update(..., obs_mask_t=m)` bit-for-bit (as the unmasked parity test at `:36`). |
| `test_glm_laplace.py::TestBernoulliUpdate::test_masked_bernoulli_ll_drops_masked_terms` | Bernoulli LL with mask == LL on the unmasked subset (`rtol=1e-12`); fully masked → posterior = prior (`rtol=1e-10`), LL 0. |
| `test_point_process_kalman.py::TestPointProcessModel::test_fit_with_mask_matches_unmasked_when_all_true` (`slow`) | LL history identical at `rtol=1e-10`; `fit_sgd(obs_mask=m)` sets `_n_timesteps` to the observed-bin count. |
| `test_place_field_model.py::TestWarmStart::test_stationary_glm_mask_weights_are_exact` | Masked `_fit_stationary_glm` MAP equals the MAP of the GLM fit on the unmasked rows only (`rtol=1e-8`) and its covariance is the inverse masked Hessian (pattern of `TestStationaryGLMFitIsStationary`, `test_oracle_point_process.py:931`). |
| `test_place_field_model.py::TestPlaceFieldModelFit::test_fit_with_tracking_gap` (`slow`, block path, 2 neurons) | A 40-bin fully masked gap: LL finite/non-decreasing until stop; smoothed variance of every neuron's weights peaks inside the gap (guard); `predict_rate_map` still works. |
| `test_place_field_model.py::TestMultiNeuron::test_neuron_masked_everywhere_keeps_prior` (`slow`) | A neuron masked at every bin: its smoothed mean equals its warm-start/init mean at every bin (`rtol=1e-10`) and its variance grows monotonically along time (random-walk prior). |
| `test_place_field_model.py::TestPlaceFieldModelScore::test_score_with_mask` | `score(position, spikes, obs_mask=m)` equals the masked filter LL; all-`True` equals the unmasked score. |
| `test_em_golden_regression.py` (existing, `slow`) | `point_process_glm`, `place_field_*` unchanged at `_EXACT`. |

## Fixtures

- `_pp_problem` (`test_likelihood_identities.py:583-600`) and the module-scoped
  problem fixtures of `test_point_process_kalman.py:140`, `:1299`, `:1689`, plus
  a deterministic `random_neuron_mask(rng, n_time, n_neurons, p_missing, fully_masked_bins)`
  helper in `test_point_process_kalman.py` (shared by the place-field tests via
  import).
- Place-field data from `test_em_golden_regression._place_field_data`-style
  simulation (`:150-167`) or the existing fixture at `test_place_field_model.py:84`.
- No real data: the independent Laplace recursion and the per-neuron deletion
  equivalence are the references.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind) — none flagged; confirm unmasked scalar family likelihood calls are the old expressions and only masked calls require `loglik_per_obs`.
- User-facing documentation listed as tasks is updated, not deferred.
