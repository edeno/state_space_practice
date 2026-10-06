# Phase 4 — Multi-sequence fitting: shared machinery and the oscillator models

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d4--multi-sequence-sufficient-statistics-two-pass)

Ships the `state_space_practice.sequences` helper module, batched (leading
`n_sequences` axis) sufficient statistics for `kalman_maximization_step`,
`switching_kalman_maximization_step`, `compute_transition_sufficient_stats` and
`compute_process_covariance_sufficient_stats`, and multi-sequence `fit` /
`fit_sgd` for the oscillator `BaseModel` family: several trials / epochs /
sessions with shared parameters, padded to a common length with the mask
carrying the padding. `run_em` and `SGDFittableMixin` are unchanged
([C9](shared-contracts.md#c9--where-batching-lives-not-in-run_em)).

**Inputs to read first:**

- `src/state_space_practice/em_driver.py:12-14`, `:58-77`, `:222-223` — the driver's
  callable contract; confirm nothing here needs to know about sequences.
- `src/state_space_practice/sgd_fitting.py:304-351` (`_build_sgd_step`; `/ n_timesteps`
  `:327`), `:384-397` (`_prepare_sgd_data`), `:506-723` (`fit_sgd`; `n_timesteps`
  `:560-564`, data leaves `:592-593`, LL rescaling `:658`, `:711`).
- `src/state_space_practice/kalman.py:1219-1248` (`sum_of_outer_products`),
  `:1251-1278` (`InitialStatePrior`), `:1320-1361`
  (`smooth_initial_state_with_cross_cov`), `:1399-1445` (`process_cov_residual_form`),
  `:1554-1628` (`_kalman_maximization_step`, transition block `:1583-1619`).
- `src/state_space_practice/switching_kalman.py:2150-2300`
  (`_switching_kalman_m_step_inner`, after phase 2's refactor), `:2303-2597`
  (`switching_kalman_maximization_step`; `n_time < 2` check `:2426-2430`,
  transition stats `:2440-2448`, occupancy gate `:2514-2539`), `:3087-3184`
  (`compute_transition_sufficient_stats`), `:3187-3241`
  (`compute_process_covariance_sufficient_stats`).
- `src/state_space_practice/oscillator_models.py:188-220` (`_sgd_n_time` `:203`),
  `:234-286` (`decode`, `predict_proba`, `_discrete_state_posterior`), `:310-312`
  (`_n_timesteps`), `:415-438` (`_EM_SNAPSHOT_KEYS`), `:606-803` (warm init and
  seeding), `:886-970` (`_e_step`), `:972-1030` (`_m_step`), `:1082-1142` (`fit`),
  `:1146-1194` (`fit_sgd`), the subclass overrides listed in phase 2, and
  `:1640-1687` (CNM constrained Q), `:2023-2129` (DIM reparameterised M-step;
  `compute_transition_sufficient_stats` call `:2089-2097`).
- `src/state_space_practice/tests/oracles.py:689-752`
  (`switching_path_sufficient_statistics`), `:755-818` (`switching_q_from_statistics`).
- `src/state_space_practice/tests/test_oracle_switching_kalman.py:483`
  (`_m_step_from_exact_stats`), `:528` (`TestSwitchingMStepMaximisesExactQ`).
- `src/state_space_practice/tests/test_oracle_kalman.py:390-473`
  (`TestKalmanMStepMaximisesExactQ`).
- `src/state_space_practice/tests/test_sgd_fitting.py:105` (`_ToyModel`), `:147`
  (`TestSGDFittableMixin`), `:604` (`TestPrepareSGDDataHook`).
- `pyproject.toml:134-159` (`[tool.mypy] files` — add `sequences.py`).

**Contracts referenced:**

- [C6 — multi-sequence inputs, padding, lengths](shared-contracts.md#c6--multi-sequence-inputs-padding-and-sequence-lengths)
  — implemented here; do not weaken "padding = trailing all-False bins" or
  "`len_s >= 1`".
- [C7 — `sequences` helper API](shared-contracts.md#c7--state_space_practicesequences-helper-api)
  — implemented here.
- [C8 — `_n_timesteps`](shared-contracts.md#c8--sgdfittablemixin_n_timesteps-under-masks-and-sequences),
  [C9 — batching lives in the models](shared-contracts.md#c9--where-batching-lives-not-in-run_em),
  [C5](shared-contracts.md#c5--backwards-compatibility-off-means-the-old-code-runs)
  (2-D input → old code path).

**Designs referenced:** [D4](designs.md#d4--multi-sequence-sufficient-statistics-two-pass),
[D5](designs.md#d5--batching-strategy-and-measured-memory),
[D6 (sequences)](designs.md#d6--oracle-extensions-for-masks-and-sequences).

## Tasks

- **Create `src/state_space_practice/sequences.py`** with the functions of
  [C7](shared-contracts.md#c7--state_space_practicesequences-helper-api)
  (NumPy-style docstrings with shapes; `jnp` throughout; `pad_sequences` is
  host-side and may use NumPy for the ragged input). `sequence_lengths` is
  `jnp.where(any_obs.any(1), 1 + argmax over reversed time, 0)` — write it with
  `jnp.flip` + `jnp.argmax` so it is jit-safe. Add the module to
  `[tool.mypy] files`. Unit tests in a new `tests/test_sequences.py` (fast).

- **Batched Gaussian M-step.** Refactor `process_cov_residual_form`
  (`kalman.py:1399-1445`) into `_process_scatter(...)` (the expression at
  `:1437-1444`, unnormalised) plus the division, and give
  `_kalman_maximization_step` a batched branch (`smoother_mean.ndim == 3`)
  implementing the two-pass statistics of
  [D4](designs.md#d4--multi-sequence-sufficient-statistics-two-pass) with
  `transition_weights(obs_mask)` (all-ones when `obs_mask is None`) and the
  cross-sequence initial-state average. `kalman_maximization_step` accepts
  `(n_sequences, n_time, ...)` arrays and a batched `obs_mask`; the 2-D path is
  the old code.

- **Batched switching M-step.** `compute_transition_sufficient_stats` and
  `compute_process_covariance_sufficient_stats` (`switching_kalman.py:3087-3241`)
  gain `transition_weight=None` and accept a leading sequence axis (detected by
  `ndim`; `jax.vmap` + `jax.tree.map(lambda x: x.sum(0), ...)`); the unbatched,
  unweighted branch is untouched. Add `_switching_kalman_m_step_batched(...)`
  per [D4](designs.md#d4--multi-sequence-sufficient-statistics-two-pass) and
  dispatch from `switching_kalman_maximization_step` on
  `state_cond_smoother_means.ndim == 4`; the `n_time < 2` check (`:2426-2430`)
  becomes "at least one sequence has ≥2 real bins"; the occupancy gate
  (`:2514-2539`) uses summed occupancies. Initial state and `pi_0` averaged
  across sequences as specified.

- **Oscillator `BaseModel`: batched E-step.** `_e_step(observations, obs_mask)`
  (`oscillator_models.py:886-970`): when `observations.ndim == 3`, run
  `sequences.map_over_sequences` over a per-sequence function that calls
  `switching_kalman_filter` and the configured smoother (GPB1 `:963-966` or GPB2
  `:946-951`), store every posterior with the leading sequence axis, return
  `jnp.sum(ll)`. `_m_step(observations, obs_mask)` passes the batched
  posteriors to the batched M-step; `_pool_measurement_covariance` (`:1066-1080`)
  sums `smoother_discrete_state_prob` over `(s, t)` with the bin weights. Add
  `sequence_lengths_` (set in `fit` / `fit_sgd` from the mask; `None` for a 2-D
  fit) and document `decode()` / `predict_proba()` shapes `(n_sequences, n_time_max[, K])`
  with padding rows to be sliced by `sequence_lengths_` (`:234-286`).
  `_EM_SNAPSHOT_KEYS` (`:415-438`) needs no change (posteriors are arrays).

- **`fit` / `fit_sgd` / warm init.** `fit(observations, ..., obs_mask=None)`
  (`:1082-1142`) calls `sequences.canonicalize_sequences` (2-D → unchanged
  path; 3-D → validated, `len_s >= 1`), then proceeds; `fit_sgd` (`:1146-1194`)
  sets `_sgd_n_time = n_observed_bins(obs_mask, n_time)`; the three
  `_sgd_loss_fn`s call `switching_kalman_filter` through
  `map_over_sequences` when `observations.ndim == 3` and return `-jnp.sum(ll)`
  (DIM's connectivity penalty `:2299-2311` unchanged). Warm init
  (`_warm_initialize_states` `:606-646`, `_prepare_windows` `:648-671`): build
  windows per sequence over its real bins (`t < len_s`), stack the window
  features across sequences; `_seed_state_parameters_from_windows` (`:732-803`)
  builds `probs` of shape `(n_sequences, n_time_max, K)` (rows beyond `len_s`
  uniform — they carry zero weight) and `joint` `(n_sequences, n_time_max - 1, K, K)`,
  then calls the batched E/M-steps. CNM `_m_step_constrained_process_covariance`
  (`:1640-1687`) and DIM `_m_step_reparameterized` (`:2089-2097`) pass
  `transition_weight=transition_weights(obs_mask)` to the statistics functions.

- **Memory smoke test (task, not a unit test).** With
  `jax.jit(vmapped_e_step).lower(...).compile().memory_analysis()` at
  `S=20, T=2000, n_oscillators=2 (n=4), n_sources=4, K=2`, record temp/output bytes
  for `_SEQUENCE_MAP="vmap"` and `"map"`, and wall time of the second call. The
  session baseline in [D5](designs.md#d5--batching-strategy-and-measured-memory)
  is 54 MB temp / 52 MB output for `vmap` (filter + GPB1). Keep `vmap` unless
  temp > 4x output; record the numbers in the CHANGELOG entry ("measured ...").
  Extrapolate to the executor's largest intended fit (`S x T x n² x K²` for the
  pair-conditional covariances) before running it.

- **Tests** (validation slice): sequences module unit tests; batched M-step
  exactness against summed path-enumeration statistics
  ([D6](designs.md#d6--oracle-extensions-for-masks-and-sequences)); padding
  inertness; additivity; oscillator fits.

- **User-facing docs:** CHANGELOG `### Added` ("multi-sequence fitting for the
  oscillator models: pass `(n_sequences, n_time_max, n_sources)` observations
  with an `obs_mask` carrying the padding, or `sequences.pad_sequences(list)`;
  `sequence_lengths_`; `decode()` / `predict_proba()` shapes"); README
  subsection "Multiple sequences" (pad → fit → slice with `sequence_lengths_`,
  five lines); docstrings of `fit` / `fit_sgd` / `decode` / `predict_proba`,
  `kalman_maximization_step`, `switching_kalman_maximization_step`,
  `compute_transition_sufficient_stats`, `compute_process_covariance_sufficient_stats`,
  and the `sequences` module docstring. Cite Shumway & Stoffer (§6 EM for
  state-space models) for pooled sufficient statistics across independent
  series in the `sequences` module docstring only if the executor confirms the
  section number in the edition at hand; otherwise cite the book without a
  section.

- **Confirm nothing moved:** golden cases `common_oscillator`,
  `directed_influence` at `_EXACT`; `TestSwitchingMStepMaximisesExactQ`,
  `TestKalmanMStepMaximisesExactQ` unchanged.

## Deliberately not in this phase

- `PointProcessModel` / `PlaceFieldModel` multi-sequence (phase 5) — they need
  the block-diagonal container work of
  [D7](designs.md#d7--block-diagonal-covariances-with-a-sequence-axis).
- The switching point-process models and `SwitchingSpikeOscillatorModel`
  (revisit trigger: `docs/plans/multi-map-place-fields/` phase 2; their
  `_m_step_reparameterized` calls the switching M-step with dummy observations
  and would need the batched transition statistics only).
- Per-sequence parameters (e.g. per-session `R`): shared parameters only; a
  per-sequence `R` is the cross-session-drift plan's concern.
- A public `lax.map` knob: `_SEQUENCE_MAP` is module-private; flipping it is a
  code change with a measured justification
  ([D5](designs.md#d5--batching-strategy-and-measured-memory)).
- Streaming / chunked E-steps for sequences that do not fit in memory.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_sequences.py::test_pad_sequences_roundtrip` | Ragged lists → padded array + mask; `sequence_lengths` recovers the lengths; padding entries are zero and masked; per-sequence masks are respected. |
| `test_sequences.py::test_sequence_lengths_trailing_masked_bins_are_padding` | A trailing fully masked stretch shortens `len_s`; an interior fully masked stretch does not; a sequence with no observed bin has length 0 and `canonicalize_sequences` raises `ValueError` naming it. |
| `test_sequences.py::test_weights_are_consistent` | `transition_weights[s, t] == bin_weights[s, t+1]`; `observed_bin_weights <= bin_weights` elementwise; `n_observed_bins` equals the sum. |
| `test_oracle_kalman.py::TestKalmanMStepMaximisesExactQ::test_batched_m_step_maximises_summed_q` | Two sequences of different lengths (padded): batched `kalman_maximization_step` zeroes the finite-difference gradient of `Q_1 + Q_2` (per-sequence `lgssm_expected_complete_log_likelihood`, `atol=1e-6`); the shared `init_mean`/`init_cov` equal the sequence average / averaged covariance-plus-spread (closed form, `rtol=1e-10`). |
| `test_oracle_switching_kalman.py::TestSwitchingMStepMaximisesExactQ::test_batched_m_step_maximises_summed_path_q` | Two sequences (K=2, T=4 and 3): batched switching M-step zeroes the gradient of `switching_q_from_statistics` on the *summed* `switching_path_sufficient_statistics` (`atol=1e-6`). |
| `test_switching_kalman.py::TestMStepProperties::test_padding_is_inert` | M-step on `(1, T, ...)` input equals the 2-D M-step at `rtol=1e-12`; appending 5 padding bins (mask False) to a sequence leaves every M-step output unchanged at `rtol=1e-12`. Guard: the padded E-step's posteriors at padding bins equal their predictions. |
| `test_switching_kalman.py::TestMStepProperties::test_batched_transition_stats_equal_sum_of_per_sequence_stats` | `compute_transition_sufficient_stats` / `compute_process_covariance_sufficient_stats` on batched input == sum of per-sequence calls (`rtol=1e-12`); unbatched call bit-identical to before (`assert_array_equal`). |
| `test_oscillator_models.py::TestEMAlgorithm::test_multi_sequence_e_step_ll_is_sum_of_sequences` | `_e_step` on 3 padded sequences returns Σ of the three single-sequence LLs (`rtol=1e-12`); per-sequence posteriors equal the single-sequence runs on their real bins (`rtol=1e-12`). |
| `test_oscillator_models.py::TestEMAlgorithm::test_two_d_input_is_bit_identical` | `fit(obs)` before/after the change: covered by the golden test; additionally `_e_step(obs)` on 2-D input `assert_array_equal` to `_e_step(obs[None])[…, 0]`-free path — i.e. 2-D runs the old code (assert via `jax.make_jaxpr` equality of the E-step is overkill; use the golden test plus `assert_array_equal` of `fit(obs)` vs the recorded LL history). |
| `test_oscillator_models.py::TestEMAlgorithm::test_multi_trial_fit_recovers_shared_parameters` (`slow`, COM, K=1 via `TestSingleDiscreteState` style) | 10 simulated trials of 300 bins from one model: the multi-sequence fit's `measurement_matrix` is closer (Frobenius) to the truth than a fit on a single trial (guard: single-trial error `> 1.5x` the multi-trial error). |
| `test_oscillator_models.py::TestEMAlgorithm::test_multi_sequence_decode_shapes_and_lengths` (`slow`) | After a 3-sequence fit `decode().shape == (3, T_max)`, `sequence_lengths_` equals the true lengths; rows beyond them are not read by any public accessor without slicing (documented). |
| `test_oscillator_models.py::TestCommonOscillatorSGDFitting::test_multi_sequence_fit_sgd` (`slow`) | `fit_sgd` on padded sequences runs; `_n_timesteps == Σ len_s` (no interior masks); loss equals `-(Σ LL)/n_observed_bins` at step 0 (`rtol=1e-10`) — check by evaluating `_sgd_loss_fn` directly. |
| `test_oscillator_models.py::TestOscillatorEMRollback` (existing helper, `slow`) | `assert_em_rolls_back_on_ll_decrease` also passes with a 3-D `fit_args[0]`. |
| `test_em_golden_regression.py` (existing, `slow`) | `common_oscillator`, `directed_influence` unchanged at `_EXACT`. |

## Fixtures

- `test_sequences.py`: small NumPy ragged lists.
- Oscillator multi-trial data: extend the module-scoped simulators in
  `test_oscillator_models.py:32-100` with a `simulate_trials(model, n_trials, lengths, key)`
  helper that returns a list; pad with `sequences.pad_sequences`.
- Oracle tests reuse `_random_switching_model` / `_simulate_lgssm` per sequence.
- No real data in tests; the memory smoke test is a task with recorded numbers.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind) — `process_cov_residual_form` must delegate to `_process_scatter` (one implementation of the residual), and the single-sequence M-step branches must be the pre-existing statements.
- User-facing documentation listed as tasks is updated, not deferred.
