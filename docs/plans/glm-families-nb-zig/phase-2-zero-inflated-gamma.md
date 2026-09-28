# Phase 2 — Zero-inflated-gamma family for deconvolved calcium traces

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#3-zero-inflated-gamma-family)

Ships `zero_inflated_gamma_family(shape, loc=0.0)` (Wei et al. 2020: point mass
at zero with probability `1 - q`, gamma of fixed per-neuron shape `k` and
latent-dependent scale `theta` otherwise; `logit q` and `log theta` are two
stacked linear predictors per neuron), `zero_inflated_gamma_mean`, the
`validate_observations` hook that lets continuous traces through the count
validators, `PointProcessModel(family="zero_inflated_gamma", gamma_shape=,
gamma_loc=)`, a calcium-like simulator, and the ZIG oracle / recovery /
calibration tests. Requires Phase 1 (the `score` hook and `family=` threading).

**Inputs to read first:**

- [docs/plans/glm-families-nb-zig/phase-1-negative-binomial.md](phase-1-negative-binomial.md) — what Phase 1 added (`GLMFamily.score`, `family=` keyword, `PointProcessModel._resolve_family` / `_build_family`); read the merged code, not just the plan.
- [src/state_space_practice/point_process_kalman.py:1215-1239](../../../src/state_space_practice/point_process_kalman.py#L1215) — `GLMFamily` (now with `score`); append `validate_observations`.
- [point_process_kalman.py:1268-1297](../../../src/state_space_practice/point_process_kalman.py#L1268) — `_BERNOULLI_ETA_CLIP` and the clip-asymmetry comment the ZIG family reuses for `logit q`.
- [point_process_kalman.py:96-145](../../../src/state_space_practice/point_process_kalman.py#L96) — `_validate_public_inputs` (`validate_count_array` at `:140`) → family-aware.
- [point_process_kalman.py:3150-3186, 3234-3279](../../../src/state_space_practice/point_process_kalman.py#L3150) — `PointProcessModel.fit` / `fit_sgd` count validation (`:3180`, `:3266`) → family-aware.
- [point_process_kalman.py:3373-3444](../../../src/state_space_practice/point_process_kalman.py#L3373) — `get_rate_estimate` returns `exp(eta)` (`:3444`); guard for ZIG.
- [src/state_space_practice/utils.py:1016-1041, 1067-1087](../../../src/state_space_practice/utils.py#L1016) — `validate_count_array`, `validate_nonnegative_array` (argument orders differ: `(counts, name)` vs `(name, value)`; the hook uses `(obs, name)`).
- [src/state_space_practice/simulate_data.py](../../../src/state_space_practice/simulate_data.py) — add the ZIG simulator here (mypy-gated).
- [src/state_space_practice/tests/test_glm_laplace.py:71-193](../../../src/state_space_practice/tests/test_glm_laplace.py#L71) — the Bernoulli math/update test shapes to mirror for ZIG.
- [src/state_space_practice/tests/test_oracle_point_process.py](../../../src/state_space_practice/tests/test_oracle_point_process.py) — the `_Observation` record from Phase 1 (design §7); `_grids` `:443-445`.
- [src/state_space_practice/tests/test_calibration_point_process.py:42-78](../../../src/state_space_practice/tests/test_calibration_point_process.py#L42) — `_calibration_stats`, `_assert_calibrated`.
- [src/state_space_practice/tests/recovery_helpers.py:94-111](../../../src/state_space_practice/tests/recovery_helpers.py#L94) — `assert_smoother_beats_prior`.
- Wei et al. 2020 — the model statement transcribed in [designs §3.1](designs.md#31-the-model-wei-et-al-2020); no need to fetch the paper.

**Contracts referenced:**

- [C1 — `GLMFamily` extended contract](shared-contracts.md#c1-glmfamily-extended-contract) — add `validate_observations`; invariants 1-6 (especially 4: Poisson/Bernoulli/NB programs unchanged).
- [C2 — `family=` on the filter and smoother](shared-contracts.md#c2-family-on-the-filter-and-smoother) — item 4 (family-aware validation) is implemented here.
- [C3 — `PointProcessModel` family kinds](shared-contracts.md#c3-pointprocessmodel-family-kinds) — the `"zero_inflated_gamma"` row.
- [C4 — Stacked-predictor layout](shared-contracts.md#c4-stacked-predictor-layout) — `eta = [logit q ; log theta]`, design rows in the same order.

**Designs referenced:** [designs §3](designs.md#3-zero-inflated-gamma-family) (model, alternatives, derivation, code, validation, non-goals), [§5](designs.md#5-pointprocessmodel-family-plumbing) (Phase-2 paragraph), [§6](designs.md#6-simulators), [§7](designs.md#7-extending-the-grid-quadrature-oracle-and-the-calibration-harness) (`_zig_observation`).

## Tasks

- **T1 — `validate_observations` hook and family-aware validation.** Append `validate_observations: Callable[[ArrayLike, str], None] | None = None` to `GLMFamily`; add `_observation_validator(family)` and `_validate_zig_observations` ([designs §3.5](designs.md#35-observation-validation)); give `_validate_public_inputs` a keyword-only `family=None` and use the validator at `:140`; pass `family=family` from both public entry points; in `PointProcessModel.fit` / `fit_sgd` replace the two `validate_count_array(spike_indicator, ...)` calls by `_observation_validator(self._family)(spike_indicator, "spike_indicator")`. Existing count families keep `validate_count_array` (default `None`).

- **T2 — `zero_inflated_gamma_family` and `zero_inflated_gamma_mean`.** Add `_ZIG_LOG_SCALE_CLIP`, the family factory and the mean helper after `negative_binomial_family`, exactly as in [designs §3.4](designs.md#34-code) (docstring includes the Wei et al. citation, the stacked layout, "Fisher scoring with expected information `diag(q(1-q)) ⊕ diag(qk)`, exactly block-diagonal", and that `dt` plays no role). Tests in `tests/test_glm_laplace.py`, `TestZeroInflatedGammaFamilyMath` / `TestZeroInflatedGammaUpdate`, using a 3-latent, 4-neuron affine `eta_func` returning `(8,)`.

- **T3 — `PointProcessModel(family="zero_inflated_gamma")`.** Per [designs §5](designs.md#5-pointprocessmodel-family-plumbing) (Phase-2 paragraph) and C3: `gamma_shape` (required for this kind, `> 0`, scalar or `(n_neurons,)`), `gamma_loc=0.0` (`>= 0`), both rejected for other kinds; `_check_observation_shape` also checks a per-neuron `gamma_shape`/`gamma_loc` length; `_build_family` returns the ZIG family; `get_rate_estimate` raises `ValueError` for this kind with the message in designs §5. Docstring: the model line, the two new parameters, the design-row layout (C4) and "`log_intensity_func` returns `2 * n_neurons` predictors for this family".

- **T4 — Simulator.** `simulate_data.simulate_zero_inflated_gamma_traces` ([designs §6](designs.md#6-simulators)); mypy clean. Tests in `tests/test_simulate_data.py`: zero fraction, conditional mean of positives, overall mean vs `q (k theta + loc)`, all positive values `> loc` when `loc > 0`, determinism.

- **T5 — Oracle extension.** Add `_zig_observation(k, loc)` to `tests/test_oracle_point_process.py` (designs §7); `_simulate_problem` builds the `2 * n_neurons`-row design (intercepts `a_n ~ N(-0.3, 0.4)`, `c_n ~ N(0, 0.3)`, weights `N(0, weight_scale)`). Parametrise `TestLaplaceFilterComputesItsApproximation` with ZIG (`k = 2`, `loc = 0`; N1 roundoff, converged `1e-7`) and add `TestZeroInflatedGammaLaplaceVsExactPosterior` (T = 6, 1-2 latents, seeds 0-2, `weight_scale = 0.7`, grid ±3): record the standardised errors in the docstring and pin with ~10x headroom on N3 / ~2x on N1 plus the `> 1e-3` non-vacuous bounds.

- **T6 — Calcium-like recovery and calibration (slow).** New `tests/test_zero_inflated_gamma_model.py`: (i) `simulate_zero_inflated_gamma_traces(n_time=1000, n_neurons=8, shape=2.0)`, `PointProcessModel(n_state_dims=2, family="zero_inflated_gamma", gamma_shape=..., transition_matrix=0.95 I, process_cov=0.05 I, log_intensity_func=_affine_log_rate)` E-step at the true parameters — `assert_smoother_beats_prior`, smoothed-latent RMSE `< 0.5 ×` the prior-mean RMSE, Pearson correlation with the true latent `> 0.9` on each dimension, and `zero_inflated_gamma_mean` of the smoothed predictor correlating `> 0.8` with the true mean trace; (ii) pooled z-calibration over 400 replicates × T = 30 × 2 latents (fresh coefficients per replicate) → `_assert_calibrated`, or, if measurably miscalibrated, pin ±4 SE with the reason; (iii) `fit_sgd(num_steps=100)` with `update_process_cov=True` only improves the LL and keeps `process_cov` PSD.

- **T7 — Regression guard.** Run and confirm unchanged: `tests/test_glm_laplace.py` (Poisson parity, Bernoulli, NB), `tests/test_em_golden_regression.py` (all cases), `tests/test_coupling_ekf.py`, `tests/test_hamiltonian_core.py`, the Phase-1 NB parity tests. Add one fast test that `GLMFamily(mean, w, lp, ln)` (4 positional fields) still constructs with `score is None`, `loglik_per_obs is None` and `validate_observations is None`.

- **T8 — Validation tests (fast).** Continuous non-negative traces: accepted by the filter with the ZIG family, rejected (`ValueError` from `validate_count_array`) with `family=None` and with the NB family; negative values rejected for ZIG; with `loc = 0.05`, a value `0.02` rejected with the "must exceed the gamma location" message; `PointProcessModel` kwarg validation (`gamma_shape` with Poisson, ZIG without `gamma_shape`, non-positive shape, negative loc, wrong per-neuron length at `fit`); `get_rate_estimate` raises for ZIG.

- **T9 — Docs.** `CHANGELOG.md` "Unreleased / Added": `zero_inflated_gamma_family`, `zero_inflated_gamma_mean`, `GLMFamily.validate_observations`, `PointProcessModel(family="zero_inflated_gamma", gamma_shape=, gamma_loc=)`, `simulate_zero_inflated_gamma_traces`, and a line that the point-process filters now accept continuous observations through a family's validator. `README.md` description sentence (`:3-6`, the "State-space models in JAX for neural data: ..." list) gains "zero-inflated-gamma observations for deconvolved calcium-imaging traces"; package-layout sentence from Phase 1 extended with "zero-inflated gamma". `CLAUDE.md` "Project Overview" bullet list is developer-facing repo documentation: add the same bullet.

## Deliberately not in this phase

- Latent-dependent gamma shape `k(x)` — breaks the diagonal expected information ([designs §3.6](designs.md#36-non-goals-with-a-trigger)); revisit with design (a) only on demand.
- Learning `gamma_shape` / `gamma_loc` by SGD — [overview Open Question 4](overview.md#open-questions).
- ZIG in `PlaceFieldModel`, `PositionDecoder`, the switching models, or the block-diagonal path — [overview non-goals](overview.md#non-goals).
- A generic `(n_obs, k, k)` block-weight update (design alternative (a)) — not needed while every shipped multi-predictor family has zero expected cross-information.
- Any change to the NB family or its tests beyond running them (regression guard T7).

## Validation slice

| Test | Asserts |
| --- | --- |
| `TestZeroInflatedGammaFamilyMath::test_per_observation_terms_and_mask_layout` | Two observations/four predictors with a nonzero location and continuous positive values: per-observation terms sum to normalized likelihood, and selecting observation 1 selects predictor rows 1 and 3. Its likelihood gradient equals the correspondingly masked score. |
| `TestZeroInflatedGammaFamilyMath::test_score_matches_autodiff` | `J' score` == `jax.grad` of `loglik_normalized` at 5 random `x` and `y` with zeros and positives, `atol 1e-10` |
| `…::test_expected_information_matches_expected_hessian` | analytic `diag(w)` (2N×2N) == `(1 - q) * negH(y=0) + q * ∫ negH(y) Gamma(y - loc; k, θ) dy` with `negH = -jax.hessian(loglik)` (scipy `quad` per neuron, or the affine identity `E[(y-loc)/θ] = k`), **including** off-diagonal blocks ≈ 0, `atol 1e-8` |
| `…::test_information_identity` | `w` == `E_y[score^2]` (same mixture expectation), `rtol 1e-8` |
| `…::test_fisher_is_psd` | `w >= 0`; `eigvalsh(J' diag(w) J).min() >= -1e-10` at extreme `eta` (`±40` logits, `±25` log-scales: clips active) |
| `…::test_plugin_gradient_equals_score` | `jax.grad(loglik_plugin)` w.r.t. `eta` == `score`, `atol 1e-10` (line search consistent with the Fisher step) |
| `…::test_zero_and_positive_observations_are_finite` | value and gradient finite at `y = 0` and `y = loc + 1e-9` (double-where guard) |
| `TestZeroInflatedGammaUpdate::test_update_increases_log_posterior` | one Fisher step from a weak prior increases the ZIG log-posterior; posterior cov finite and PSD |
| `…::test_laplace_cov_is_inverse_expected_information_at_mode` | `post_cov` == `inv(prior_prec + J' diag(w(mode)) J)` at the converged mode (`max_newton_iter=25`), `rtol 1e-6`; and **differs** from `inv(-observed Hessian)` by `> 1e-3` relative (guard that Fisher ≠ Newton here) |
| `…::test_zero_iterations_return_prior` | as `TestZeroNewtonIterations` (`test_glm_laplace.py:389-436`) for the ZIG family |
| `TestLaplaceFilterComputesItsApproximation[zig]` (oracle) | library vs independent NumPy Fisher recursion with the ZIG score/weight: N1 `rtol 1e-10`, converged `1e-7` |
| `TestZeroInflatedGammaLaplaceVsExactPosterior` (oracle) | standardised errors vs the continuous-y grid posterior below the recorded pins; `> 1e-3`; N3 ≤ N1 on means |
| `test_zero_inflated_gamma_model.py::test_latent_recovery_at_true_parameters` — **slow** | smoother beats prior; RMSE `< 0.5 ×` prior RMSE; per-dimension correlation `> 0.9`; mean-trace correlation `> 0.8` |
| `…::test_smoother_is_calibrated` — **slow** | pooled `z` over 400 replicates: `_assert_calibrated` (or pinned ±4 SE with documented reason) |
| `…::test_sgd_improves_ll_and_keeps_psd` — **slow** | final LL > initial; `eigvalsh(process_cov) > 0` |
| `TestObservationValidation` (fast, 6 cases) | continuous data accepted for ZIG / rejected for Poisson and NB; negatives rejected; `0 < y <= loc` rejected with the documented message; `PointProcessModel` kwarg errors; `get_rate_estimate` raises for ZIG |
| `test_glm_laplace.py::test_four_field_family_constructs` | `GLMFamily(m, w, lp, ln).score is None and .validate_observations is None` |
| `test_simulate_data.py::TestSimulateZeroInflatedGammaTraces` | zero fraction within 2% of `1 - mean(q)`; mean of positives within 3% of `k θ + loc` (per neuron, N ≥ 5e4 bins); overall mean within 3% of `q (k θ + loc)`; all positives `> loc`; determinism |
| Regression (existing, T7) | `test_glm_laplace.py`, `test_em_golden_regression.py`, `test_coupling_ekf.py`, `test_hamiltonian_core.py`, Phase-1 NB parity: unchanged |

`test_zero_inflated_gamma_model.py` tests that call `_e_step` directly are not
auto-marked slow — mark the class `@pytest.mark.slow`.

## Fixtures

- `test_glm_laplace.py`: module-level `_ZIG_K = jnp.array([1.5, 2.0, 3.0, 2.5])`,
  `_ZIG_FAMILY = zero_inflated_gamma_family(_ZIG_K)`, an affine `_eta8(x)`
  returning `(8,)` from `_C`-style `(8, 3)` weights and an `(8,)` intercept, and
  a fixed `y = [0, 0.7, 0, 2.3]`.
- Oracle: `_zig_observation(2.0, 0.0)`; the same `_grids`.
- Recovery / calibration: `simulate_zero_inflated_gamma_traces` with
  `default_rng(seed)`; class-scoped fixture simulates and runs the E-step once.
- No real-data slice: no deconvolved calcium file is checked in. The simulated
  dataset mirrors Wei et al.'s regime (zero fractions 0.3-0.8, shape ≈ 2);
  document in the test module that a real two-photon slice should be added when
  one becomes available ([overview Open Question 5](overview.md#open-questions)).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind). (None flagged; confirm `validate_count_array` is still the default validator and was not duplicated.)
- User-facing documentation listed as tasks is updated, not deferred.
