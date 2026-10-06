# Phase 1 — Observation masks in the Gaussian Kalman filter and smoothers

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d1--masked-gaussian-measurement-update-static-shape-trick)

Ships `obs_mask=` on `kalman_measurement_update`, `kalman_filter` and
`kalman_smoother`: missing channels / bins are handled as exact conditioning on
the observed entries, fully masked bins are predict-only, and the returned
log-likelihood is the marginal over observed entries. `parallel_kalman_smoother`
needs no change (it consumes filtered moments). No M-step, no model, no
point-process code in this phase.

**Inputs to read first:**

- `src/state_space_practice/kalman.py:426-475` (`kalman_measurement_update`) — the
  one place `H`, `R`, `y` meet: innovation covariance `:456-458`, gain solve `:461`,
  Joseph form `:464-466`, log-likelihood `:471-473`. The mask transform wraps this.
- `src/state_space_practice/kalman.py:478-526` (`_kalman_filter_update`) and
  `:529-610` (`_kalman_filter_impl`; scan body `:570-594`, scan inputs `:596-608`) —
  where the per-step mask has to be threaded into the scan.
- `src/state_space_practice/kalman.py:613-711` (`kalman_filter`), `:915-941`
  (`_kalman_smoother_impl`), `:944-1044` (`kalman_smoother`) — the public signatures
  that gain the keyword-only argument.
- `src/state_space_practice/kalman.py:270-423` (`_validate_kalman_public_inputs`) —
  shape checks `:301-345`, the `obs` finiteness check `:370-378` that must exempt
  masked entries.
- `src/state_space_practice/kalman.py:1072-1216` (`parallel_kalman_smoother`) — confirm
  it never touches `H`/`R`/`y`; only its docstring changes.
- `src/state_space_practice/switching_kalman.py:49-57` and `:747-751` — the two
  `jax.vmap`s over `_kalman_filter_update` / `kalman_measurement_update` whose
  `in_axes` tuples must grow by one `None` entry (this phase keeps them
  mask-less but must not break them).
- `src/state_space_practice/utils.py:910-916` (`contains_tracer`), `:1187-1305`
  (`_validate_filter_numerics`) — validation idioms to mirror.
- `src/state_space_practice/tests/oracles.py:64-77` (`_condition`), `:129-195`
  (`lgssm_joint_prior`), `:198-272` (`lgssm_dense_posterior`) — the dense oracle to
  extend with masked-row dropping.
- `src/state_space_practice/tests/test_oracle_kalman.py:41-42` (tolerances), `:45-81`
  (`_simulate_lgssm`), `:113-152` (`_assert_matches_oracle`), `:286-298`
  (`test_parallel_smoother_matches_oracle`) — test scaffolding to reuse.
- `src/state_space_practice/tests/test_likelihood_identities.py:142-214`
  (`_lgssm`, `_kalman_args`, `lgssm` fixture, `test_ll_is_sum_of_predictive_log_densities`)
  and `:287-306` (`test_chain_rule_by_restart`).
- `src/state_space_practice/tests/test_kalman.py:1017` (`TestKalmanFilterInputHandling`),
  `:1757` (`TestParallelKalmanSmoother`), `:2151` (`TestKalmanInputValidation`) —
  where validation / handling tests for the new argument belong.
- `CHANGELOG.md:6-8` (`## [Unreleased]` / `### Added`), `README.md:60-68`
  (`## Package layout`) — documentation targets.

**Contracts referenced:**

- [C1 — `obs_mask` argument](shared-contracts.md#c1--obs_mask-argument) — shapes,
  dtype, NaN-in-masked-entries rule; do not weaken.
- [C2 — masked-bin E-step semantics](shared-contracts.md#c2--masked-bin-semantics-e-step-and-log-likelihood) —
  items 1, 3, 4 are this phase's acceptance criteria.
- [C4 — `validate_observation_mask`](shared-contracts.md#c4--validate_observation_mask) —
  implemented here, in `utils.py`.
- [C5 — "off" means the old code runs](shared-contracts.md#c5--backwards-compatibility-off-means-the-old-code-runs).

**Designs referenced:** [D1](designs.md#d1--masked-gaussian-measurement-update-static-shape-trick),
[D6 (dense posterior with masked entries)](designs.md#d6--oracle-extensions-for-masks-and-sequences).

## Tasks

- **Add `validate_observation_mask` to `utils.py`** exactly as in
  [C4](shared-contracts.md#c4--validate_observation_mask), after
  `validate_count_array` (`utils.py:1016-1041`). Unit-test it directly
  (`test_utils.py`): 1-D broadcast, 2-D pass-through, wrong shape, non-bool
  dtype (`jnp.int32` zeros/ones and `float`), and that it traces under `jax.jit`
  with a traced mask (shape/dtype checks are static).

- **Add `_mask_gaussian_measurement` and the mask parameter to
  `kalman_measurement_update` / `_kalman_filter_update`** as written in
  [D1](designs.md#d1--masked-gaussian-measurement-update-static-shape-trick).
  `obs_mask` is the trailing positional-or-keyword parameter (default `None`)
  on both jitted functions; the `None` branch adds no operations. Update the two
  `jax.vmap` definitions in `switching_kalman.py:49-57` (`in_axes` gain a
  trailing `None` on both levels) and `:747-751` (`in_axes=(-1, -1, None, -1, -1, None)`)
  so the switching filter keeps calling them with `obs_mask=None` implicitly —
  it passes no mask in this phase. Update the docstrings (`kalman.py:434-454`,
  `:488-515`) with the new parameter and the LL semantics.

- **Thread the mask through `_kalman_filter_impl`, `kalman_filter`,
  `_kalman_smoother_impl`, `kalman_smoother`.** Keyword-only `obs_mask=None`
  on the public functions after `validate_inputs`; the jitted impls take it as a
  trailing keyword-only parameter (a `None` pytree is static). In
  `_kalman_filter_impl` zero-fill `obs` with `jnp.where(obs_mask, obs, 0.0)` and
  add the mask to the scan inputs under the static branch shown in D1. When
  `validate_inputs=False`, canonicalise the mask with
  `validate_observation_mask` anyway (it is static and cheap) so the impl always
  sees `(n_time, n_obs)` bool or `None`.

- **Extend `_validate_kalman_public_inputs`** (`kalman.py:270-423`) with
  `obs_mask: ArrayLike | None` → canonicalise via `validate_observation_mask`
  (`n_time = obs.shape[0]`, `n_obs = obs.shape[1]`) and return it as an eighth
  element; the `obs` finiteness check (`:370-378`) becomes
  `jnp.all(jnp.isfinite(obs) | ~mask)` when a mask is given (concrete path
  only, as today). Every other check is unchanged.

- **Document `parallel_kalman_smoother`** (`kalman.py:1078-1111`): one sentence
  in *Parameters* that it consumes the filter's output unchanged, so a masked
  run passes `obs_mask` to `kalman_filter` only; no signature change (see
  [C1](shared-contracts.md#c1--obs_mask-argument), last paragraph).

- **Extend the dense oracle** (`tests/oracles.py`) with `obs_mask` on
  `lgssm_dense_posterior` via `_condition_subset` as in
  [D6](designs.md#d6--oracle-extensions-for-masks-and-sequences); keep the
  `obs_mask=None` behaviour identical (`keep` all-`True` selects every
  row — `np.ix_` with all-True is a copy of the same matrix). Add an oracle
  self-check: with a mask that hides channel 1 at every bin, the masked oracle
  equals the unmasked oracle of the model with `H`/`R` reduced to channel 0.

- **Tests** — see the validation slice. New masked tests go to
  `test_oracle_kalman.py` (oracle agreement), `test_likelihood_identities.py`
  (LL identity), `test_kalman.py::TestKalmanInputValidation` /
  `TestKalmanFilterInputHandling` (validation, NaN handling, bit-identity),
  `test_kalman.py::TestParallelKalmanSmoother` (masked filtered moments in).
  Reuse `_simulate_lgssm` and add a `random_mask(rng, n_time, n_obs, p_missing, *, fully_masked_bins=())`
  helper in `test_oracle_kalman.py` that guarantees at least one partially
  masked bin and at least one fully masked bin (guards: assert both occur).

- **User-facing docs:** CHANGELOG `### Added` entry
  ("`obs_mask` on `kalman_filter` / `kalman_smoother` / `kalman_measurement_update`:
  missing channels or bins ... exact marginal log-likelihood over observed
  entries; fully masked bins are predict-only"); a short README subsection
  "Missing observations" under `## Package layout` (`README.md:60`) with a
  three-line example (`mask = ~np.isnan(y); kalman_smoother(..., obs_mask=mask)`
  — note NaN is allowed in masked entries). Public docstrings of the three
  functions carry the *Parameters* entry and a *Notes* sentence on the LL
  convention (cite Särkkä 2013, *Bayesian Filtering and Smoothing*, for the
  missing-measurement = prediction-only step; the module already cites it,
  `kalman.py:12-13`).

- **Confirm nothing moved:** run
  `uv run pytest src/state_space_practice/tests/test_em_golden_regression.py`
  and the full `test_kalman.py`, `test_switching_kalman.py`,
  `test_oracle_switching_kalman.py`; every value must be unchanged (the mask-less
  path is the old code).

## Deliberately not in this phase

- `kalman_maximization_step(obs_mask=)` and the switching filter / M-step: phase 2
  (the M-step needs the imputation design, and the switching filter is only
  useful once the oscillator models can consume masks end-to-end).
- Point-process filters: phase 3.
- Any leading sequence axis: phases 4-5.
- `woodbury_kalman_gain` / `standard_kalman_gain` (`kalman.py:104-232`): no
  callers in the filters; leave untouched.
- Deriving masks from NaNs automatically: never (explicit masks only,
  [C1](shared-contracts.md#c1--obs_mask-argument)); NaNs in masked entries are
  merely tolerated.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_utils.py::test_validate_observation_mask_shapes_and_dtype` | 1-D `(T,)` → `(T, m)` broadcast; 2-D unchanged; `(T+1,)`, `(T, m+1)`, int and float dtypes raise `ValueError` naming `obs_mask`; works on a traced mask under `jax.jit`. |
| `test_oracle_kalman.py::test_masked_recursions_match_oracle` (Hypothesis, `max_examples=10`, fixed shape n=2, m=3, T=6, p_missing ∈ {0.3, 0.6}) | Filter/smoother means, covariances, cross-covariances and both LLs match `lgssm_dense_posterior(..., obs_mask=mask)` at `RTOL=1e-8` (reuse `_assert_matches_oracle` with a mask parameter). Guard: mask has ≥1 partially masked and ≥1 fully masked bin. |
| `test_oracle_kalman.py::test_masked_recursions_match_oracle_degenerate` (parametrised: all channels masked at t=0; only the last bin observed; `n_obs > n_latent` with R ~ 1e-6) | Same agreement; the "only last bin observed" case's smoothed `x_1` equals the prior-propagated moments conditioned once (the oracle says so) and its LL equals a single Gaussian logpdf. |
| `test_oracle_kalman.py::test_oracle_mask_drops_channel_exactly` (oracle self-check) | Oracle with channel 1 masked everywhere == oracle of the 1-channel model (`H[:1]`, `R[:1, :1]`) to 1e-12. |
| `test_oracle_kalman.py::test_parallel_smoother_matches_masked_oracle` | `parallel_kalman_smoother(*kalman_filter(..., obs_mask=m)[:2], A, Q)` matches the masked oracle's smoothed moments and cross-covariances at `RTOL`. |
| `test_likelihood_identities.py::TestKalmanIdentities::test_masked_ll_is_sum_of_observed_predictive_densities` | Filter LL == Σ_t `gaussian_logpdf(y_o, H_o m_pred, H_o P_pred H_o^T + R_oo)` with 0 for fully masked bins (`rtol=1e-10`); smoother LL == filter LL exactly. Guard: ≥1 fully masked bin, terms vary (`ptp > 0.5`). |
| `test_likelihood_identities.py::TestKalmanIdentities::test_masked_chain_rule_by_restart` | Splitting a masked sequence at a fully masked bin and restarting from the filtered moments reproduces the total LL (`rtol=1e-10`) and the tail posteriors (`1e-12`). |
| `test_kalman.py::TestKalmanFilterInputHandling::test_no_mask_is_bit_identical_to_omitting_it` | `kalman_filter(*args)` vs `kalman_filter(*args, obs_mask=None)` and `kalman_smoother` likewise: `assert_array_equal` on every output. |
| `test_kalman.py::TestKalmanFilterInputHandling::test_all_true_mask_equals_no_mask` | All-`True` mask → `assert_array_equal` on means, covariances, LL (exact arithmetic, [C5](shared-contracts.md#c5--backwards-compatibility-off-means-the-old-code-runs)). |
| `test_kalman.py::TestKalmanFilterInputHandling::test_fully_masked_bin_is_predict_only` | At a fully masked bin the filtered mean equals `A m_{t-1}` and cov equals `symmetrize(A P A^T + Q)` (`assert_array_equal` on mean, `rtol=1e-14` on cov); its LL contribution (LL with vs without that bin's neighbours restarted) is 0 to 1e-12. |
| `test_kalman.py::TestKalmanFilterInputHandling::test_nan_in_masked_entries_is_ignored` | `obs` with NaN at masked entries gives `assert_array_equal` results to `obs` with zeros there; NaN at an *unmasked* entry raises `ValueError` ("finite"). |
| `test_kalman.py::TestKalmanFilterInputHandling::test_mask_is_traceable_and_differentiable` | `jax.grad` of the masked LL w.r.t. `H` is finite; `jax.jit(kalman_filter, static_argnames="validate_inputs")` with a traced mask runs. |
| `test_kalman.py::TestKalmanInputValidation::test_obs_mask_shape_and_dtype_errors` | `(T+1,)`, `(T, m+1)` and int masks raise `ValueError` at the public entry points, with `validate_inputs=True` and `False`. |
| `test_switching_kalman.py::TestSwitchingKalmanFilterProperties` (existing) + `test_em_golden_regression.py` (existing, `slow`) | Unchanged results — the vmapped update signatures grew but the switching path passes no mask. |

Mark nothing new as slow except tests that run EM (none here); the oracle tests
follow `test_oracle_kalman.py`'s existing fast/slow split (`:200-229`).

## Fixtures

- `random_mask(rng, n_time, n_obs, p_missing, *, fully_masked_bins=())` helper in
  `test_oracle_kalman.py` (NumPy RNG, deterministic seeds via Hypothesis
  `derandomize=True` as the file already does).
- Reuse `_simulate_lgssm` (`test_oracle_kalman.py:45-81`) and the `lgssm`
  fixture pattern (`test_likelihood_identities.py:182-197`) with a module-scoped
  `masked_lgssm` fixture (model, mask, filter/smoother outputs).
- No real data: the dense oracle is the ground truth.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind) — none are flagged here; confirm no dead branch was introduced.
- User-facing documentation listed as tasks is updated, not deferred.
