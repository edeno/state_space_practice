# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

- `src/state_space_practice/point_process_kalman.py:1215-1239` — `GLMFamily`: optional fields appended/reused (`score` and `loglik_per_obs`, Phase 1; `validate_observations`, Phase 2). Existing four-field constructors untouched.
- `point_process_kalman.py:1242-1265` — `poisson_family`: gains `loglik_per_obs` while keeping existing scalar callbacks unchanged; new factories `negative_binomial_family` (Phase 1) and `zero_inflated_gamma_family` / `zero_inflated_gamma_mean` (Phase 2) are added after it.
- `point_process_kalman.py:1273-1297` — Bernoulli family: gains `loglik_per_obs` while keeping existing scalar callbacks unchanged (its `_BERNOULLI_ETA_CLIP` is reused by ZIG).
- `point_process_kalman.py:1333-1464` — `glm_laplace_update`: the two `innovation = observations - mu` lines (`:1407`, `:1422`) route through the family score; everything else preserved.
- `point_process_kalman.py:927-1212` — `_point_process_laplace_update` (legacy Poisson update): untouched; remains the `family=None` path and the block-path update, and the update used by `position_decoder.py:1133` and `switching_point_process.py:595`.
- `point_process_kalman.py:1467-1734` / `:1737-1866` / `:2392-2602` — public filter, jitted impl, public smoother: new `family` keyword, block dispatch requires `family is None`, `_step` branches on `family`. Block-diagonal cores (`:1873-2389`) untouched.
- `point_process_kalman.py:96-145` — `_validate_public_inputs`: family-aware observation validation (Phase 2).
- `point_process_kalman.py:2928-3474` — `PointProcessModel`: new kwargs, `_family` attribute, SGD dispersion, validation and `get_rate_estimate` guard; `_m_step` (`:3107-3148`) and the EM rollback in `fit` (`:3188-3229`) untouched.
- `src/state_space_practice/simulate_data.py` — two new simulators; existing functions untouched.
- `src/state_space_practice/coupling_ekf.py:87-106`, `hamiltonian_core.py:179-191` — existing `glm_laplace_update` callers: untouched, must remain bit-identical.
- `src/state_space_practice/place_field_model.py`, `position_decoder.py`, `switching_point_process.py`, `__init__.py` — untouched (no new lazy exports: the families live in `point_process_kalman`, as `poisson_family` does).
- Tests extended: `tests/test_glm_laplace.py`, `tests/test_oracle_point_process.py` (observation-model record), `tests/test_calibration_point_process.py`, `tests/test_sbc_ranks.py`, `tests/recovery_helpers.py`, `tests/test_simulate_data.py`, `tests/test_point_process_kalman.py`; new `tests/test_zero_inflated_gamma_model.py`. `tests/test_em_golden_regression.py` is a guard, not edited.

## Scope and dependency policy

### Goals

- Over-dispersed spike counts: a negative-binomial family with log link whose Laplace-EKF update is Fisher scoring with the exact expected information `mu / (1 + mu / r)`, coinciding with Poisson as `r -> inf`; dispersion per neuron, fixed under EM, learnable under SGD.
- Deconvolved calcium traces: a zero-inflated-gamma family (Wei et al. 2020) with latent-dependent non-zero probability and gamma scale and fixed per-neuron shape, opening `PointProcessModel` to two-photon data.
- One shared update: both families run through `glm_laplace_update` and the existing filter/smoother, with analytic, PSD-by-construction expected information (no observed Hessians).
- Verification at the level the repo already holds Poisson to: independent-recursion agreement, grid-quadrature oracle pins, pooled z-calibration and SBC ranks, parameter recovery, and a demonstration that the misspecified Poisson family under-covers on over-dispersed data.

### Non-Goals

- Changing any default numerics: `poisson_family`, `BERNOULLI_LOGIT_FAMILY`, the `family=None` filter path and `PointProcessModel()` defaults stay byte-identical (additive-only policy).
- Non-Poisson families on the block-diagonal path, in `PlaceFieldModel`, `PositionDecoder`, `SwitchingSpikeOscillatorModel`, the oscillator point-process models or the Hamiltonian models.
- An EM M-step for the NB dispersion (Open Question 1) or SGD learning of the ZIG shape/location (Open Question 4).
- Latent-dependent gamma shape, multi-component slabs, or a general `(n_obs, k, k)` block-information update.
- Retiring `_point_process_laplace_update` in favour of `glm_laplace_update(poisson_family)` (Open Question 2).
- Real-data notebooks or examples beyond docstrings/README (no over-dispersed or calcium dataset is checked in).

### Dependency policy

No new runtime dependencies: everything uses `jax.scipy.special` / `jax.nn` (verified present in the locked JAX 0.10.2: `gammaln`, `digamma`, `nbinom.logpmf`, `gamma.logpdf`). Tests use `scipy.stats.nbinom` / `scipy.special.gammaln`, already imported by `tests/test_oracle_point_process.py:61`. `numpy.random.Generator.negative_binomial` and `.gamma` provide the NumPy samplers; JAX has no NB sampler, so the JAX helper uses a gamma-Poisson mixture.

This plan depends on none of the parallel plans. Names other plans may import from this one, exactly as delivered: `state_space_practice.point_process_kalman.negative_binomial_family(dt, dispersion, max_log_count=20.0)`, `zero_inflated_gamma_family(shape, loc=0.0)`, `zero_inflated_gamma_mean(eta, shape, loc=0.0)`, the `family=` keyword on `stochastic_point_process_filter` / `stochastic_point_process_smoother`, and `PointProcessModel(family=...)` with the kwargs in [shared-contracts C3](shared-contracts.md#c3-pointprocessmodel-family-kinds).

## Metrics

- **Additivity:** `test_em_golden_regression.py` records unchanged after both phases; `TestPoissonParity` unchanged; `test_coupling_ekf.py` / `test_hamiltonian_core.py` unchanged.
- **NB ↔ Poisson parity:** update-level and filter-level agreement within `1e-8` (mean, covariance, marginal LL) at `r = 1e10`, with `_log_gamma_ratio` accurate to `1e-10` absolute for `r` up to `1e10`.
- **Analytic correctness:** scores match `jax.grad` to `1e-10`; expected information matches the exact expectation of the negative Hessian to `1e-8` (NB: truncated pmf sum; ZIG: mixture of the zero atom and a gamma quadrature, including zero off-diagonal blocks).
- **Oracle:** library agrees with the independent NumPy Fisher recursion (N1 `1e-10`, converged `1e-7`); standardised errors against the exact grid posterior are pinned with ≥ 2x headroom and a `> 1e-3` non-vacuous floor, for NB and ZIG.
- **Calibration:** NB truth + NB family within ±4 cluster-SE on `E[z]`, `E[z^2]`, 90% coverage; NB truth + Poisson family shows `E[z^2] > 1` and coverage `< 0.9` beyond 4 SE; NB SBC ranks pass the χ² test with the 0.75-variance power guard; ZIG calibrated or pinned with a documented reason.
- **Recovery:** NB dispersion recovered within `|Δ log r| < 0.35` from a 10x-off start; ZIG latent correlation `> 0.9` at true parameters.
- **Suite budget:** fast suite (`-m "not slow"`) still under one minute; `uv run ruff check`, `ruff format --check` and `uv run mypy` clean.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The `score` hook or family branch perturbs the Poisson/Bernoulli traced program (rounding-level drift in golden values). | Python-level `is None` branches only; golden regression and parity tests run in both phases (Phase-1 T9, Phase-2 T7). |
| Large-`r` cancellation in `lgamma(y + r) - lgamma(r)` breaks the LL parity target and makes the SGD gradient in `r` noisy on Poisson-like data. | `_log_gamma_ratio` Stirling branch above `r = 1e3`, tested against exact integer sums for value and gradient up to `r = 1e10`. |
| Static `family` argument: each factory call is a new jit cache key → recompilation every EM iteration or SGD step. | `PointProcessModel` builds the family once and rebuilds only when its parameters change; under SGD the family is rebuilt inside the loss but traced once per compiled step (verified mechanics: a family closing over a traced dispersion works as a static arg under `jit(value_and_grad)` with correct gradients). Documented on the filters. |
| `max_log_count` silently ignored when a family is passed. | `StateSpaceWarning` when both are given; families own their ceilings. |
| ZIG log/division at `y = 0` or `y = loc` produce NaN gradients through masked branches. | Double-`where` (`excess = where(y > 0, y - loc, 1.0)`), validation rejects `0 < y <= loc`, finiteness test at `y = 0` and `y = loc + 1e-9`. |
| Continuous ZIG data rejected by `validate_count_array` at three entry points, or count data accidentally accepted for ZIG. | Family-owned `validate_observations` with `validate_count_array` as the default; validation tests for both directions. |
| Line search objective inconsistent with the score for a new family (Armijo test uses `loglik_plugin`). | Contract C1 item 3; `test_plugin_gradient_equals_score` for ZIG; the NB plugin's derivative is derived to equal the score in designs §2.1. |
| Oracle refactor changes the existing Poisson pins. | Poisson `_Observation` reproduces the current formulas; existing tests stay parametrised on it and must pass unchanged before NB/ZIG instances are added. |
| Pinned approximation-error values are guesses. | The plan never states numbers: the executor measures, records them in the test docstring (repo convention), and pins with the stated headroom plus non-vacuous floors. |

## Rollout Strategy

Additive, opt-in, two PRs. Users who do not pass `family=` (functions) or `family=`/`dispersion=`/`gamma_shape=` (model) see identical behaviour and identical numbers. No deprecation is introduced; the legacy Poisson update stays the default indefinitely under this plan (its retirement is a separate decision, Open Question 2). Phase 2 depends on Phase 1 being merged (it uses the `score` hook and the `family=` plumbing); Phase 1 is useful on its own.

## Open Questions

1. **NB dispersion under EM.** Deferred — `fit` keeps `dispersion` fixed; `fit_sgd(update_dispersion=True)` is the shipped answer. If an M-step is wanted later, the concrete route is a per-neuron 1-D Newton on the expected complete-data log-likelihood with the Gaussian expectation over each scalar `eta_tn` taken by Gauss-Hermite quadrature ([designs §2.4](designs.md#24-why-dispersion-is-not-updated-by-em)).
2. **Retire `_point_process_laplace_update`?** Deferred. Trigger: when `position_decoder.py:1133` and `switching_point_process.py:595` migrate to `glm_laplace_update(poisson_family(dt))`, the `family=None` branch in the filter can become `poisson_family(dt, max_log_count)` and the block cores can take a family. Out of scope here because it would change the golden numbers at round-off and touches three other modules.
3. **Block-diagonal path for non-Poisson families** (`PlaceFieldModel` with NB place fields). Deferred; depends on 2.
4. **Learning the ZIG shape/location by SGD.** Deferred. `gamma_shape` could follow `dispersion` (`POSITIVE` transform, family rebuilt in the loss); `loc` is a data property (deconvolution minimum spike size) and should stay fixed. Trigger: a dataset where a fixed per-neuron shape fits poorly.
5. **Real-data slices.** No over-dispersed spike file or deconvolved calcium trace is checked in; both phases validate on simulated truth. Trigger: add a small real slice (and a smoke test) when a dataset is available in `docs/` or a fixture directory.
6. **ZIG mean prediction on the model.** `get_rate_estimate` is count-specific; the plan ships `zero_inflated_gamma_mean` as a function and a clear `ValueError` on the model. A `predict_observation_mean` method generic over families is deferred until a second non-count family needs it.

## Estimated Effort

- Phase 1: ~+300 LOC source (`point_process_kalman.py` ~+220: family, `_log_gamma_ratio`, filter/smoother/impl plumbing, model kwargs; `simulate_data.py` ~+25; `recovery_helpers.py` ~+20) and ~+550 LOC tests (glm_laplace ~+180, oracle refactor + NB ~+150, calibration/SBC ~+100, model/filter/recovery ~+120).
- Phase 2: ~+250 LOC source (`point_process_kalman.py` ~+200: family, mean helper, validator hook, model kwargs/guard; `simulate_data.py` ~+50) and ~+450 LOC tests (glm_laplace ~+170, oracle ZIG ~+80, new model test module ~+150, validation/simulator ~+50).
