# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

Verified against the working tree at `4b74f03` (master). Nothing in the library
handles missing observations or a leading sequence axis today (`grep` for
`mask` / `n_sequences` in `kalman.py`, `em_driver.py`, `sgd_fitting.py` finds
only an unrelated `is_array_mask` name in `sgd_fitting.py:145`).

- `src/state_space_practice/kalman.py:426-475` — `kalman_measurement_update`: the
  single point where `H`, `R`, `y` meet (innovation covariance `:456-458`,
  `_gain_solve` `:461`, Joseph form `:464-466`, `logpdf` `:471-473`). Gains the
  static-shape mask transform. `_kalman_filter_update` `:478-526`,
  `_kalman_filter_impl` `:529-610` (scan `:570-608`), `kalman_filter` `:613-711`,
  `_kalman_smoother_impl` `:915-941`, `kalman_smoother` `:944-1044`,
  `_validate_kalman_public_inputs` `:270-423`: gain / thread `obs_mask`.
- `src/state_space_practice/kalman.py:714-912` — RTS backward passes: untouched
  (observation-agnostic). `:1072-1216` `parallel_kalman_smoother`: untouched
  except a docstring sentence (consumes filtered moments only). `:104-232`
  `woodbury_kalman_gain` / `standard_kalman_gain`: untouched (no filter callers).
- `src/state_space_practice/kalman.py:1364-1445` — residual-form covariance
  estimators: `process_cov_residual_form` split into scatter + division
  (bit-identical); `:1448-1628` `kalman_maximization_step` /
  `_kalman_maximization_step`: masked observation statistics (phase 2) and a
  batched two-pass branch (phase 4); the unmasked 2-D path is the existing code.
- `src/state_space_practice/switching_kalman.py:49-57`, `:686-790`, `:793-1076` —
  the pair-vmapped update, first-step update and `switching_kalman_filter`:
  mask threaded (phase 2). `:1387`, `:1687` GPB1 / GPB2 smoothers: untouched.
  `:2150-2300` `_switching_kalman_m_step_inner` and `:2303-2597`
  `switching_kalman_maximization_step`: observation block refactored into two
  helpers (unmasked branch verbatim), masked branch (phase 2), batched dispatch
  (phase 4). `:3087-3241` `compute_transition_sufficient_stats` /
  `compute_process_covariance_sufficient_stats`: `transition_weight` and a
  leading sequence axis (phase 4). `:1079` `switching_kalman_viterbi`: untouched.
- `src/state_space_practice/oscillator_models.py` — `BaseModel._e_step` `:886-970`
  calls `switching_kalman_filter` (`:913`) and the GPB smoothers; `_m_step`
  `:972-1030` calls `switching_kalman_maximization_step` (`:1000`); `fit`
  `:1082-1142` runs `run_em` (`:1130`); `fit_sgd` `:1146-1194` sets `_sgd_n_time`
  (`:1186`); warm init `:606-803`. Subclass overrides: COM `fit` `:1382-1416` /
  `_sgd_loss_fn` `:1444-1462`; CNM `_m_step_constrained_process_covariance`
  `:1640-1687`, `fit` `:1707-1741`, `_sgd_loss_fn` `:1777-1811`; DIM `_m_step`
  `:2008-2021`, `_m_step_reparameterized` `:2023-2129`, `fit` `:2131-2165`,
  `fit_sgd` `:2199-2251`, `_sgd_loss_fn` `:2253-2313`. All gain `obs_mask`
  (phase 2) and a sequence axis (phase 4).
- `src/state_space_practice/point_process_kalman.py:927-1212` and `:1333-1464` —
  the two Laplace updates: per-neuron mask weights (phase 3). `:1467-1866`
  dense filter, `:1869-2389` block-diagonal path, `:2392-2602` smoother: thread
  the mask (phase 3). `:216-368` `BlockDiagonalCovariance`: sequence axis
  (phase 5). `:2644-2817` `dynamics_only_m_step`: batched branch (phase 5).
  `:2928-3371` `PointProcessModel`: masks (phase 3), sequences (phase 5).
- `src/state_space_practice/place_field_model.py:652-821` `_fit_stationary_glm`,
  `:983-1024` `_e_step`, `:1026-1137` `_m_step`, `:1196-1416` `fit`, `:1420-1534`
  `fit_sgd`, `:1886-1979` `score`: masks (phase 3), sequences (phase 5); the
  posterior accessors `:1660-2278` gain `sequence=` (phase 5).
- `src/state_space_practice/em_driver.py:58-356` — `run_em`: **untouched**
  (see shared contract C9).
- `src/state_space_practice/sgd_fitting.py:354-723` — `SGDFittableMixin`:
  **untouched**; adopters redefine what `_n_timesteps` counts (C8).
- `src/state_space_practice/utils.py` — gains `validate_observation_mask` (C4).
- New module `src/state_space_practice/sequences.py` (C7, phase 4).
- Tests reused: `tests/oracles.py:198-272` (dense conditioning — the perfect mask
  oracle: masked entries are dropped rows), `:473-646` (path enumeration),
  `test_oracle_kalman.py`, `test_oracle_switching_kalman.py`,
  `test_oracle_point_process.py`, `test_likelihood_identities.py`,
  `test_em_driver.py` (unchanged — `run_em` does not change),
  `test_em_golden_regression.py` (must not move in any phase),
  `tests/conftest.py:47-149` (auto-`slow` marking of anything calling `.fit(` /
  `run_em(`).

## Scope and dependency policy

### Goals

- **Observation masks** (`obs_mask=`, `True` = observed) on the Gaussian Kalman
  filter/smoother, the switching Kalman filter, the point-process Laplace-EKF
  filters/smoothers, and on `fit` / `fit_sgd` / `score` of the oscillator
  models, `PointProcessModel` and `PlaceFieldModel`: dropped LFP channels,
  tracking loss and excluded epochs become exact conditioning on observed
  entries; fully masked bins are predict-only; log-likelihoods are the marginal
  over observed entries.
- **Masked EM**: M-steps that remain exact EM under masks (imputation of missing
  Gaussian entries for `H`/`R`; unchanged dynamics statistics; mask-weighted
  place-field warm start).
- **Multi-sequence fitting** with shared parameters for the same model
  families: a leading `n_sequences` axis on padded observations, padding
  carried by the mask, `vmap`ped E-steps, summed sufficient statistics,
  sequence-averaged initial-state updates, summed log-likelihood.
- **Bit-for-bit backwards compatibility** when the options are off, verified by
  the golden regression suite and explicit `assert_array_equal` tests.

### Non-Goals

- The switching *point-process* models (`SwitchingSpikeOscillatorModel`,
  `switching_point_process.py`, `point_process_models.py`) and
  `switching_kalman_viterbi`. The Gaussian switching filter and M-step *are*
  changed, but only because the oscillator `BaseModel` is implemented on them
  (`oscillator_models.py:913`, `:1000`); nothing else in `switching_kalman.py`
  moves. Revisit trigger: `docs/plans/multi-map-place-fields/` phase 2 needing
  masked or multi-session spike-oscillator fits.
- `PositionDecoder`, `coupling_*`, `hamiltonian_*`, `temporal_rate_gp`,
  `smith_learning_algorithm`, the choice / belief models, and the deprecated
  `models.stochastic_point_process_filter`.
- Per-sequence parameters (per-session `R`, per-trial initial states): the
  deferred cross-session-drift plan's territory.
- Inferring masks from NaNs; masks are explicit (NaN is merely tolerated in
  masked entries).
- Chunked / streaming E-steps for posteriors exceeding device memory; a
  sequential `lax.map` mode exists as a module-private switch with a measured
  trigger (designs D5), nothing more.
- Changing `run_em` or `SGDFittableMixin`.

### Dependency policy

No new runtime dependencies. Consumers of this work:
`docs/plans/multi-map-place-fields/` (phase 2 of that plan consumes phase 5
here — multi-sequence `PlaceFieldModel`) and the deferred cross-session-drift
plan (multi-sequence fitting generally). This plan depends on no other plan.
Names other plans rely on are fixed in
[shared-contracts C1 and C6-C7](shared-contracts.md); `parallel_kalman_smoother`
deliberately gets no `obs_mask` (it never sees observations) — consumers pass
the mask to `kalman_filter`.

## Metrics

- Masked filters/smoothers agree with the dense Gaussian oracle at the suite's
  existing `RTOL = 1e-8` (`test_oracle_kalman.py:41`), including fully masked
  bins; masked Laplace-EKF agrees with the independent NumPy recursion to
  roundoff (as `test_oracle_point_process.py:366`).
- Log-likelihood identities hold at `rtol=1e-10`: LL = Σ one-step predictive
  densities over observed entries, 0 for fully masked / padding bins; LL over
  sequences = Σ per-sequence LLs at `rtol=1e-12`.
- Masked and batched M-steps zero the finite-difference gradient of the exact
  auxiliary function (`atol=1e-6`, the suite's existing bar) and EM is monotone
  in the exact observed-data LL.
- Off-path bit identity: `assert_array_equal` tests pass and
  `test_em_golden_regression.py` is unchanged after every phase.
- Multi-sequence memory: `vmap` temporaries ≤ 4x stored outputs at the measured
  sizes (designs D5; 54 MB / 52 MB for the switching E-step at S=20, T=2000).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The oscillator models are switching models in implementation, so "no switching changes" could not be honoured literally. | Scope narrowed to exactly the two switching functions on the oscillator fit path; unmasked branches are the verbatim existing statements; golden cases `common_oscillator` / `directed_influence` pin them. |
| Masked-bin M-step convention (impute missing entries of partially observed bins; drop fully unobserved bins) is a valid EM but not the only one; a reviewer may expect the observed-only Q. | Convention fixed in C3 with the augmented-Q oracle (designs D6) proving stationarity, plus a monotonicity test on the exact observed-data LL. |
| Padding vs interior missing-data bins are distinguished only by position (trailing all-False = padding). A user who masks the true tail of a session gets it treated as padding. | Harmless for the observed-data likelihood and its fixed point (those latents marginalise out); documented in C6 and the `fit` docstrings; tested by the padding-inertness tests. |
| Per-bin `n_obs x n_obs` solve in the masked Gaussian M-step is O(T n_obs³) — heavy for wide LFP arrays with per-channel masks. | Only taken when a mask is present; noted as a revisit trigger (n_obs > 64, M-step slower than E-step). Whole-bin masks reduce to weights and could skip the solve later. |
| `vmap` over sequences multiplies stored posteriors by `n_sequences` and can add XLA temporaries. | Measured (designs D5): temporaries ≈ outputs; storage equals a single session of the same total length. Smoke-test tasks in phases 4-5 re-measure at the executor's sizes; `lax.map` switch with a numeric trigger. |
| Bit-identity claim is easy to break by "cleaning up" the unmasked branch while adding the masked one. | C5 makes the rule explicit; `assert_array_equal` tests and the golden suite gate every phase; review checklist item. |
| `jax.vmap` maps keyword arguments over axis 0, so a keyword-only mask on the inner jitted updates would silently be mapped. | Inner updates take `obs_mask` as a trailing positional parameter (designs D1); public entry points are keyword-only. |
| NaNs in masked entries propagate through `obs * mask`. | C1 mandates `jnp.where`; explicit NaN tests in every phase. |
| Multi-sequence `PlaceFieldModel` accessors index time and were not read line-by-line for this plan. | Phase 5 routes them through one `_sequence_posterior` helper and instructs the executor to read each before editing. |

## Rollout Strategy

Additive, keyword-only options; no flags, no deprecation windows, no signature
breaks. Each phase ships as one PR that is independently useful:

1. Gaussian filter/smoother masks (library primitives).
2. Masked Gaussian EM end-to-end (oscillator models).
3. Point-process masks (filters, `PointProcessModel`, `PlaceFieldModel`).
4. Multi-sequence machinery + oscillator models.
5. Multi-sequence `PointProcessModel` / `PlaceFieldModel`.

Users who pass neither `obs_mask` nor a leading sequence axis run the
pre-existing code paths (contract C5). Phases 1→2, 3→5 and 4→5 are ordered
dependencies; 3 is independent of 1-2 except for sharing
`validate_observation_mask` (phase 1) — run phase 3 after phase 1.

## Open Questions

1. **Faster masked Gaussian M-step for whole-bin masks.** When every mask row is
   all-`True` or all-`False`, the imputation reduces to weights and the per-bin
   solve is wasted. Deferred — see the risk above; implement only against a
   measured slow case.
2. **`decode()` / `predict_proba()` return layout for sequences.** Current best
   answer: stacked `(n_sequences, n_time_max[, K])` plus `sequence_lengths_`
   (C6). A list-of-arrays alternative would break `jnp` consumers; revisit if
   the multi-map plan needs ragged outputs.
3. **`lax.map` vs `vmap` on GPU.** Measured on CPU only (designs D5).
   Phase 4's smoke test records GPU numbers if one is available; the switch
   stays private until a measured case needs it.
4. **Warm-init imputation for masked oscillator observations** (channel-mean
   fill for the GMM features) is a heuristic; if masked fits initialise poorly,
   the fallback is to seed from the masked E/M-step only (skip the GMM).
   Deferred until observed.
5. **Citation precision.** Särkkä (2013) is cited for the missing-measurement =
   prediction-only step; Shumway & Stoffer for the missing-data EM and pooled
   statistics. The executor should confirm section numbers in the edition at
   hand before writing them into docstrings (the plan gives §6.4 for the
   missing-data modification; verify).

## Estimated Effort

Diff sizing (source + tests, excluding docs):

- Phase 1: ~250 src / ~350 tests (`utils.py`, `kalman.py`, `oracles.py`).
- Phase 2: ~550 src / ~550 tests (`kalman.py` M-step, `switching_kalman.py`,
  `oscillator_models.py`, oracle extensions).
- Phase 3: ~400 src / ~450 tests (`point_process_kalman.py`, `place_field_model.py`).
- Phase 4: ~650 src / ~550 tests (`sequences.py`, batched statistics,
  `oscillator_models.py`).
- Phase 5: ~500 src / ~450 tests (`BlockDiagonalCovariance`, `dynamics_only_m_step`,
  `PointProcessModel`, `PlaceFieldModel`).

Total ≈ 2 350 source lines, ≈ 2 350 test lines.
