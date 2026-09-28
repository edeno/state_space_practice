# Phase 5 — Multi-sequence fitting for `PointProcessModel` and `PlaceFieldModel`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d7--block-diagonal-covariances-with-a-sequence-axis)

Ships a leading `n_sequences` axis on `PointProcessModel.fit` / `fit_sgd` and
`PlaceFieldModel.fit` / `fit_sgd` / `score` (position, spikes, design matrices
and masks batched; padding carried by the mask), the batched
`dynamics_only_m_step`, and a `BlockDiagonalCovariance` that holds a sequence
axis so the place-field block path never materialises dense covariances. This
is the deliverable `docs/plans/multi-map-place-fields/` (phase 2) consumes.

**Inputs to read first:**

- `src/state_space_practice/point_process_kalman.py:216-368` (`BlockDiagonalCovariance`;
  `__init__` ndim check `:252-259`, `sum` `:320-330`, `at_time` `:345-347`,
  `__getitem__` `:349-358`, `to_dense` `:360-362`), `:2125-2149` (block
  assembly helpers), `:2644-2740` (`dynamics_only_m_step`), `:2743-2817`
  (`_dynamics_only_m_step`; sums `:2766-2767`, prior branch `:2776-2785`, `A`
  `:2791-2797`, `Q` `:2801-2810`).
- `src/state_space_practice/point_process_kalman.py:2928-3371` (`PointProcessModel`:
  `_e_step` `:3065-3105`, `_m_step` `:3107-3148`, `fit` `:3150-3230`, `fit_sgd`
  `:3234-3288`, `_sgd_loss_fn` `:3314-3336`, `_finalize_sgd` `:3348-3371`).
- `src/state_space_practice/place_field_model.py:361-426` (`__init__`; `_n_time`
  `:426`), `:567-592` (`_build_spline_basis_matrix`), `:594-626`
  (`_expand_to_block_diagonal`, `_filter_design_matrix`), `:652-821`
  (`_fit_stationary_glm`), `:872-918` (`_detect_block_structure`), `:920-981`
  (`_warn_if_rate_saturated`), `:983-1024` (`_e_step`), `:1026-1137` (`_m_step`;
  `x_0` `:1076-1087`, sums `:1094-1098`, `A` `:1100-1110`, `Q` `:1112-1127`, init
  `:1129-1137`), `:1196-1416` (`fit`), `:1420-1534` (`fit_sgd`), `:1574-1609`
  (`_sgd_loss_fn`), `:1623-1658` (`_finalize_sgd`), `:1886-1979` (`score`), and
  the posterior accessors `_neuron_smoother_cov` `:1660`,
  `_posterior_rate_map_for_basis` `:1674`, `predict_rate_map` `:1730`,
  `predict_center` `:1798`, `get_state_confidence_interval` `:1981`,
  `drift_summary` `:2137`, `plot_rate_maps` `:2205`, `plot_drift` `:2278`.
- `src/state_space_practice/sequences.py` (phase 4) and the batched statistics of
  [D4](designs.md#d4--multi-sequence-sufficient-statistics-two-pass).
- `src/state_space_practice/tests/test_oracle_point_process.py:711`
  (`TestDynamicsMStepMaximisesQ`), `:844` (`TestPlaceFieldMStepMaximisesQ`, incl.
  `test_block_container_matches_dense` `:908`).
- `src/state_space_practice/tests/test_point_process_kalman.py:889`
  (`TestPointProcessModel`), `:4180` (`TestBlockDiagonalCovarianceContainer`).
- `src/state_space_practice/tests/test_place_field_model.py:145`
  (`TestPlaceFieldModelFit`), `:409` (`TestPlaceFieldModelPredict`), `:558`
  (`TestPlaceFieldModelScore`), `:923` (`TestMultiNeuron`), `:1735`
  (`TestBlockDiagonalDispatch`), `:2320` (`TestPlaceFieldMStep`).
- `src/state_space_practice/tests/test_em_golden_regression.py:150-167`
  (`_place_field_data`), `:239-260` (`_fit_point_process`, `_fit_place_field`).

**Contracts referenced:**

- [C6](shared-contracts.md#c6--multi-sequence-inputs-padding-and-sequence-lengths),
  [C7](shared-contracts.md#c7--state_space_practicesequences-helper-api),
  [C8](shared-contracts.md#c8--sgdfittablemixin_n_timesteps-under-masks-and-sequences),
  [C9](shared-contracts.md#c9--where-batching-lives-not-in-run_em),
  [C5](shared-contracts.md#c5--backwards-compatibility-off-means-the-old-code-runs),
  [C3 item 2](shared-contracts.md#c3--masked-bin-semantics-m-step-statistics)
  (masked bins stay in the dynamics statistics; only padding is excluded).

**Designs referenced:** [D4 (dynamics-only)](designs.md#d4--multi-sequence-sufficient-statistics-two-pass),
[D5](designs.md#d5--batching-strategy-and-measured-memory),
[D7](designs.md#d7--block-diagonal-covariances-with-a-sequence-axis).

## Tasks

- **`BlockDiagonalCovariance` with a sequence axis** per
  [D7](designs.md#d7--block-diagonal-covariances-with-a-sequence-axis): accept
  5-D blocks; `n_sequences`, `shape`, `weighted_time_sum(weights)`,
  `sequence(s)`, `sequence_first()` (dense `(n_sequences, n_state, n_state)` first-bin
  covariances), `at_time(t, sequence=None)`, tuple indexing, `neuron_blocks(...,
  sequence=None)`, `diagonal(sequence=None)`; `sum(axis=0)` on a batched
  container raises pointing at `weighted_time_sum`; `to_dense` on a batched
  container returns `(n_sequences, n_time, n_state, n_state)`. Container tests
  extend `TestBlockDiagonalCovarianceContainer` (`test_point_process_kalman.py:4180`).

- **Batched `dynamics_only_m_step`.** `_dynamics_only_m_step`
  (`point_process_kalman.py:2743-2817`) dispatches on `smoother_mean.ndim == 3`
  to `_dynamics_only_m_step_batched(smoother_mean, smoother_cov, smoother_cross_cov, fixed_transition_matrix, initial_state_prior, transition_weight)`:
  per-sequence `smooth_initial_state_with_cross_cov` (vmapped), the
  transition-weighted sums and two-pass `A` / `Q` of
  [D4](designs.md#d4--multi-sequence-sufficient-statistics-two-pass), initial
  state averaged across sequences. `dynamics_only_m_step` gains keyword-only
  `transition_weight: ArrayLike | None = None` (required when batched with
  padding; all-ones default). The 2-D path is the old code.

- **`PointProcessModel`** (`:2928-3371`): `fit` / `fit_sgd` accept
  `design_matrix (n_sequences, n_time_max, ...)`, `spike_indicator (n_sequences, n_time_max[, n_neurons])`
  and a batched `obs_mask` (canonicalised with `sequences.canonicalize_sequences`;
  `len_s >= 1`); `_e_step` maps `stochastic_point_process_smoother` over
  sequences with `sequences.map_over_sequences` (dense path only — this model has
  no block dispatch), stores stacked posteriors, returns the summed LL; `_m_step`
  calls `dynamics_only_m_step(..., transition_weight=transition_weights(obs_mask))`;
  `_sgd_loss_fn` sums mapped filter LLs; `_sgd_n_time = n_observed_bins`;
  `sequence_lengths_` attribute; `get_rate_estimate` / `get_confidence_interval`
  (`:3373`, `:3446`) gain `sequence: int | None = None` (required when fitted on
  sequences, else must be `None`).

- **`PlaceFieldModel`: inputs and design matrices.** `fit` / `fit_sgd` / `score`
  accept `position (n_sequences, n_time_max, 2)` and `spikes (n_sequences, n_time_max[, n_neurons])`
  (or Python lists, padded with `sequences.pad_sequences`; positions padded by
  repeating each sequence's last observed position so the spline basis sees
  in-range finite values) plus a batched `obs_mask`. `_build_spline_basis_matrix`
  (`:567-592`) fits knots on the observed positions pooled over sequences
  (`position[mask_any]`) and evaluates the basis on the flattened
  `(n_sequences * n_time_max, 2)` array, reshaping to `(n_sequences, n_time_max, nb)`;
  `_filter_design_matrix` / `_expand_to_block_diagonal` (`:594-626`) map over
  the leading axis; `_n_time` (`:426`) becomes the number of observed bins and
  `_total_spikes` sums observed entries.

- **`PlaceFieldModel`: E-step, warm start, M-step.** `_e_step` (`:983-1024`)
  maps the block (or dense) smoother over sequences; block outputs are wrapped
  in one 5-D `BlockDiagonalCovariance`. `_fit_stationary_glm` (`:652-821`)
  flattens `(n_sequences * n_time_max, nb)` with the phase-3 mask weights (padding
  rows have weight 0), so multi-sequence warm start is the single-sequence
  masked code. `_m_step` (`:1026-1137`) batched branch exactly as the code in
  [D7](designs.md#d7--block-diagonal-covariances-with-a-sequence-axis)
  (transition-weighted sums via `weighted_time_sum`, per-sequence `x_0`,
  averaged initial state), then the existing diagonal/isotropic constraint and
  floors (`:1120-1137`). `_detect_block_structure` (`:872-918`) is unchanged
  (parameters carry no sequence axis). `_warn_if_rate_saturated` (`:920-981`)
  flattens `(s, t)` and counts observed entries only.

- **`PlaceFieldModel`: posterior accessors.** Add a private
  `_sequence_posterior(sequence: int | None) -> tuple[Array, BlockDiagonalCovariance | Array, ...]`
  that returns the single-sequence view (`smoother_mean[s, :len_s]`,
  `smoother_cov.sequence(s)` truncated, filtered likewise) and raises
  `ValueError` when `sequence is None` after a multi-sequence fit (or non-`None`
  after a single-sequence fit). Route `_neuron_smoother_cov` (`:1660`),
  `_posterior_rate_map_for_basis` (`:1674`), `predict_rate_map` (`:1730`),
  `predict_center` (`:1798`), `get_state_confidence_interval` (`:1981`),
  `drift_summary` (`:2137`), `plot_rate_maps` (`:2205`), `plot_drift` (`:2278`)
  through it with a new keyword-only `sequence: int | None = None`; read each
  accessor's time indexing before editing (they were not opened for this plan
  beyond their signatures).

- **Memory smoke test (task).** Block path at `S=10, T=2000, n_neurons=20, nb=16`:
  `jax.jit(mapped_e_step).lower(...).compile().memory_analysis()` for `"vmap"`
  and `"map"`; expected output ≈ `3 S n_neurons T nb² x 8 B` = 2.5 GB
  ([D5](designs.md#d5--batching-strategy-and-measured-memory)). If the
  executor's machine cannot hold that, halve `T` and extrapolate linearly (the
  storage is linear in `S x T`). Record temp/output and wall time in the
  CHANGELOG entry; the total is the same as a single `S x T`-bin session today,
  so no chunking is added unless temporaries dominate (rule in D5).

- **Tests** (validation slice) — M-step exactness against summed per-sequence
  Q (`TestPlaceFieldMStepMaximisesQ` pattern), padding inertness, LL additivity,
  accessors.

- **User-facing docs:** CHANGELOG `### Added` ("multi-sequence `PointProcessModel`
  / `PlaceFieldModel` fits; `sequence=` on the place-field accessors;
  `BlockDiagonalCovariance` sequence axis; measured memory ..."); README
  "Multiple sequences" subsection gains a `PlaceFieldModel` example (list of
  trial positions/spikes → `pad_sequences` → `fit` → `predict_rate_map(sequence=0)`);
  docstrings of every public method touched.

- **Confirm nothing moved:** golden cases `point_process_glm`, `place_field_*`
  at `_EXACT`; `TestDynamicsMStepMaximisesQ`, `TestPlaceFieldMStepMaximisesQ`,
  `TestBlockDiagonalCovarianceContainer` unchanged.

## Deliberately not in this phase

- Multi-map / per-sequence place fields (`docs/plans/multi-map-place-fields/`
  phase 2 builds on this phase; shared parameters only here).
- `PositionDecoder` multi-sequence decoding; `SwitchingSpikeOscillatorModel`
  and the switching point-process oscillator models (trigger: multi-map plan
  phase 2).
- `bin_spike_times` producing padded arrays directly; `sequences.pad_sequences`
  is the tool.
- Per-sequence `init_mean` / `init_cov` (a shared initial prior is the model;
  per-trial initial states are a cross-session-drift concern).
- Chunked / streaming E-steps for posteriors that exceed device memory.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_batched_previous_covariance_sum_matches_dense_reference` | For unequal sequence lengths, compare `sc.weighted_time_sum(concat([tw, zero_column])) + sum(P0)` with an explicit per-sequence sum of previous-state covariance blocks. Guard that the last stored covariance contributes zero and the first contributes the next transition weight. Include T=1 at the container/statistics level and compare resulting dynamics sufficient statistics with independent dense accumulation. |
| `test_point_process_kalman.py::TestBlockDiagonalCovarianceContainer::test_sequence_axis_reductions` | 5-D container: `weighted_time_sum(w)` equals the dense weighted sum (`rtol=1e-12`); `sequence(s)` equals a 4-D container of `blocks[s]`; `sum(axis=0)` raises on 5-D; 4-D `weighted_time_sum(None)` `assert_array_equal` to `sum(axis=0)`; `to_dense` shape `(S, T, n, n)`. |
| `test_oracle_point_process.py::TestDynamicsMStepMaximisesQ::test_batched_update_maximises_summed_q` | Two sequences (T=5 and 3, padded): batched `dynamics_only_m_step` zeroes the finite-difference gradient of `Q_1 + Q_2` (existing `_q_function` `:631` per sequence, `atol=1e-6`); shared init equals the closed-form average / average-plus-spread (`rtol=1e-10`). |
| `test_oracle_point_process.py::TestPlaceFieldMStepMaximisesQ::test_batched_m_step_maximises_summed_q_with_block_container` | Two-sequence `PlaceFieldModel._m_step` on synthetic block posteriors zeroes the gradient of the summed Q under the diagonal-Q constraint (pattern of `:856`); block container result equals the dense-array result (`rtol=1e-12`, pattern of `:908`). |
| `test_point_process_kalman.py::TestPointProcessModel::test_multi_sequence_e_step_ll_is_sum` | `_e_step` on 3 padded sequences == Σ single-sequence LLs (`rtol=1e-12`); posteriors on real bins equal the single runs (`rtol=1e-12`). |
| `test_point_process_kalman.py::TestPointProcessModel::test_padding_is_inert_in_m_step` | Appending padding to a sequence leaves `dynamics_only_m_step` outputs unchanged (`rtol=1e-12`); `(1, T, ...)` input equals the 2-D M-step at `rtol=1e-12`; 2-D call `assert_array_equal` to the pre-change outputs (golden). |
| `test_point_process_kalman.py::TestPointProcessModel::test_multi_sequence_fit_and_accessors` (`slow`) | `fit` on 3 sequences: LL history finite, `sequence_lengths_` correct, `get_rate_estimate(sequence=1)` shape `(len_1, ...)`, `get_rate_estimate()` raises `ValueError` after a multi-sequence fit. |
| `test_place_field_model.py::TestPlaceFieldModelFit::test_multi_trial_fit_matches_concatenated_when_trials_are_contiguous` (`slow`, 2 neurons, block path) | Two trials that are halves of one continuous session: the multi-sequence E-step LL differs from the concatenated LL only by the missing `x_{T1} -> x_{T1+1}` coupling (guard: not equal), and the multi-sequence fit's process variances stay within 2x of the concatenated fit's (behavioural sanity, not exactness). |
| `test_place_field_model.py::TestPlaceFieldModelFit::test_multi_trial_fit_recovers_fields` (`slow`) | 6 simulated trials of 250 bins each vs one trial: the multi-trial `predict_rate_map(sequence=0)` correlation with the true field exceeds the single-trial fit's (guard `> 0.05` gap). |
| `test_place_field_model.py::TestPlaceFieldModelFit::test_padding_and_masks_compose` (`slow`) | A trial with an interior tracking gap plus padding: LL equals the sum of per-trial masked LLs (`rtol=1e-12`); `_n_time` counts observed bins. |
| `test_place_field_model.py::TestWarmStart::test_multi_sequence_warm_start_equals_pooled_glm` | `_fit_stationary_glm` on padded sequences equals the GLM fit on the concatenated observed rows (`rtol=1e-8`). |
| `test_place_field_model.py::TestPlaceFieldModelPredict::test_sequence_keyword_on_accessors` (`slow`) | After a 2-sequence fit each accessor with `sequence=1` uses only sequence 1's real bins (compare with a single-sequence fit on that trial's posteriors: `predict_rate_map`, `drift_summary` agree at `rtol=1e-10` when fed identical posteriors); `sequence=None` raises; single-sequence fit with `sequence=0` raises. |
| `test_place_field_model.py::TestPlaceFieldModelScore::test_multi_sequence_score` | `score` on padded sequences == Σ per-sequence masked scores (`rtol=1e-12`). |
| `test_place_field_model.py::TestPlaceFieldSGDFitting::test_multi_sequence_fit_sgd` (`slow`) | `fit_sgd` runs; `_n_timesteps == n_observed_bins`; block dispatch retained when `update_transition_matrix=False`. |
| `test_em_golden_regression.py` (existing, `slow`) | `point_process_glm`, `place_field_*` unchanged at `_EXACT`. |

## Fixtures

- Trials from a `simulate_place_field_trials(n_trials, lengths, n_neurons, seed)`
  helper modelled on `_place_field_data` (`test_em_golden_regression.py:150-167`),
  padded with `sequences.pad_sequences`; reuse `_random_smoother_moments`
  (`test_oracle_point_process.py:689`) per sequence for the M-step oracles.
- Container tests build blocks with `np.random.default_rng`.
- No real data; the real-data smoke run belongs to the consuming plan
  (`docs/plans/multi-map-place-fields/`).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind) — the 4-D container behaviour and the 2-D M-step paths are the pre-existing code; `weighted_time_sum(None)` must not duplicate `sum(axis=0)`'s body (one delegates to the other).
- User-facing documentation listed as tasks is updated, not deferred.
