# Phase 1 — WoLF on the linear-Gaussian path: `utils.imq_weight`, `kalman.py`, weighted M-step

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#g1-update)

**Inputs to read first:**

- `CLAUDE.md` (repo root) — conventions: `uv run --no-sync`, float64, `StateSpaceWarning`, behavioural tests, slow marking, NumPy docstrings with shapes.
- [src/state_space_practice/kalman.py:426-475](../../../src/state_space_practice/kalman.py) — `kalman_measurement_update`: the body that becomes the `None` branch; the LL-on-unboosted-`obs_cov` comment (468-473) that must stay true for the unweighted LL.
- [src/state_space_practice/kalman.py:478-610](../../../src/state_space_practice/kalman.py) — `_kalman_filter_update` and `_kalman_filter_impl`: the scan whose step must emit the per-step weight and objective when robust.
- [src/state_space_practice/kalman.py:613-711, 915-1044](../../../src/state_space_practice/kalman.py) — `kalman_filter`, `_kalman_smoother_impl`, `kalman_smoother`: keyword pass-through points.
- [src/state_space_practice/kalman.py:1364-1396, 1448-1628](../../../src/state_space_practice/kalman.py) — `measurement_cov_residual_form`, `kalman_maximization_step`, `_kalman_maximization_step`: the observation block to weight (H at 1577, R at 1578-1581) and the `initial_state_prior is None` pattern (1585) for a `None` argument inside jit.
- [src/state_space_practice/utils.py:26-50, 114-182, 1161, 1662-1719](../../../src/state_space_practice/utils.py) — `symmetrize`, `psd_cholesky`/`psd_logdet`, `validate_scalar`, `check_converged` (the new section goes after it).
- [src/state_space_practice/em_driver.py:58-77, 222-299](../../../src/state_space_practice/em_driver.py) — confirm nothing here changes.
- [`src/state_space_practice/__init__.py:35-100`](../../../src/state_space_practice/__init__.py) — lazy public API to extend (`_LAZY_API` 35-53, `__all__` 55-76, `TYPE_CHECKING` imports 78-100).
- [src/state_space_practice/tests/oracles.py:45-52, 129-272](../../../src/state_space_practice/tests/oracles.py) — `lgssm_dense_posterior` accepts a per-time `R` stack (`_as_sequence`), which is how the robust filter is checked against dense conditioning.
- [src/state_space_practice/tests/test_oracle_kalman.py:45-152, 316-454](../../../src/state_space_practice/tests/test_oracle_kalman.py) — `_simulate_lgssm`, `_args`, `_assert_matches_oracle`, and the finite-difference Q-function pattern to copy for the weighted M-step test.
- [src/state_space_practice/tests/test_kalman.py:76-127, 2283-2345, 2704-2792](../../../src/state_space_practice/tests/test_kalman.py) — fixtures reused for bit-identity (`test_kalman_measurement_update_1d`, `simple_1d_model`, `_time_varying_r_model`) and the traceability tests to mirror.
- [src/state_space_practice/tests/conftest.py:12, 143-149, 369-443, 637-677](../../../src/state_space_practice/tests/conftest.py) — x64, automatic slow marking, `kalman_model_params`, `simple_1d_model`.
- [src/state_space_practice/tests/test_em_driver.py:11-47](../../../src/state_space_practice/tests/test_em_driver.py) — `ScriptedModel`: pattern for a tiny test-only model that drives `run_em`.
- `CHANGELOG.md:6-8` (`## [Unreleased]` / `### Added`; entries end at line 82 before `### Testing`) and `README.md:60-67` (`## Package layout`).

**Contracts referenced:**

- [`RobustWeight`](shared-contracts.md#robustweight), [`IMQWeight` / `imq_weight`](shared-contracts.md#imq-weight), [`RobustOutput`](shared-contracts.md#robustoutput) — this phase defines them; do not weaken the hashability or the `[0, 1]` range.
- [Return-arity rule](shared-contracts.md#return-arity) — rows `kalman_measurement_update`, `kalman_filter`, `kalman_smoother`, `kalman_maximization_step`.
- [The two objectives](shared-contracts.md#objectives) and [Invariants](shared-contracts.md#invariants) 1–5.

**Designs referenced:** [G1](designs.md#g1-update), [G2](designs.md#g2-evidence), [G3](designs.md#g3-bias), [G4](designs.md#g4-em-sgd).

## Tasks

- **`utils.py` — protocol, weight, output container, standardised residual.** Add a section "Robust (generalised-Bayes) observation weights" after `check_converged` containing `RobustWeight`, `IMQWeight`, `imq_weight`, `RobustOutput` exactly as in [shared-contracts.md](shared-contracts.md#robustweight) (with `import dataclasses`, `from typing import Protocol, runtime_checkable`), plus `standardized_residual(residual, measurement_cov)` from [G1](designs.md#g1-update) (it lives in `utils` so `switching_kalman.py` can import it without touching `kalman.py` internals). `imq_weight` validates with `validate_scalar(c, "c", positive=True)` and `validate_scalar(core, "core", nonnegative=True)`. Docstrings: paper reference (Eq. 18), the χ²_n heuristic for `c`, the `core` semantics, the hashability requirement, and the dtype rule.

- **`kalman.py` — `weighted_kalman_measurement_update` and the robust branch of `kalman_measurement_update`.** Add `import functools`; import `RobustOutput`, `RobustWeight`, `standardized_residual` from `utils` (extend the block at lines 28-37). Add the public `weighted_kalman_measurement_update` from [G1](designs.md#g1-update) (docstring: Prop. 3.1, the `S̃` form, finiteness at `w = 0`, the two returned scalars per [G2](designs.md#g2-evidence)). Re-decorate `kalman_measurement_update` with `functools.partial(jax.jit, static_argnames=("robust_weight",))`, add keyword-only `robust_weight: RobustWeight | None = None`, keep the existing body as the `None` branch verbatim, and add the robust branch that computes the weight under `stop_gradient` and returns `(mean, cov, ll, RobustOutput(objective, weight))`. Update the docstring (parameter, returns, arity rule).

- **`kalman.py` — thread through the filter and smoother.** `_kalman_filter_update` and `_kalman_filter_impl` get the same static keyword. In `_kalman_filter_impl`, keep today's `_step` for `None`; for the robust case the carry adds `objective` and the per-step outputs add `weight`, and the function returns `(filtered_mean, filtered_cov, marginal_log_likelihood, RobustOutput(objective, weights))`. `kalman_filter` and `kalman_smoother` / `_kalman_smoother_impl` pass the keyword through (the smoother appends the same `RobustOutput` after `marginal_log_likelihood`; `rts_backward_scan` is untouched). Add `@overload` stubs on `kalman_filter` and `kalman_smoother` (`robust_weight: None` → current tuple; `robust_weight: RobustWeight` → tuple + `RobustOutput`). Docstrings: describe both objectives ([contracts](shared-contracts.md#objectives)) and that a time-varying `R_t` standardises with `R_t`.

- **`kalman.py` — weighted observation M-step.** `measurement_cov_residual_form(..., *, weights=None, previous_measurement_cov=None)` and `kalman_maximization_step(..., robust_weights=None, previous_params=None)` / `_kalman_maximization_step` as in [G3](designs.md#g3-bias) (`robust_weights` shape `(n_time,)`, the weights returned by the robust filter; H and R weighted, transition block untouched). Validate `robust_weights.shape == (n_time,)` in the public wrapper when concrete; require previous H/R whenever weights are supplied, validate their shapes, and preserve them exactly when total squared weight is zero. Keep solve operands and divisors finite under `jit` and `vmap`. Docstring notes: the exact fixed-weight EM property, the clean-data bias table's gist (10–17 % for plain IMQ at `c = 3–5`, ≤ 2.5 % with `core = √χ²_d(0.99)`), and the recommendation.

- **Package export.** Add `"imq_weight": "utils"` to `_LAZY_API`, `"imq_weight"` to `__all__`, and the `TYPE_CHECKING` import in `__init__.py`.

- **Tests — new module `tests/test_robust_kalman.py`.** Implement the validation slice below. Reuse `_simulate_lgssm` / `_args` / `_oracle` from `test_oracle_kalman.py` (import them), `lgssm_dense_posterior`, `simple_1d_model`, `_time_varying_r_model`, `kalman_model_params`. The EM-recovery test drives `run_em` with a small test-only class (E-step = `kalman_smoother(robust_weight=...)` storing weights and returning `robust.objective`; M-step = `kalman_maximization_step(..., robust_weights=..., previous_params=the_E_step_snapshot, initial_state_prior=...)`; snapshot/restore of the parameter dict), on a 1-D or 2-D LGSSM with 5 % of bins replaced by `N(0, (20σ)²)` noise, T = 2000; compare with the same loop and `robust_weight=None`.

- **User-facing docs.** CHANGELOG `### Added` bullet: `utils.imq_weight` / `RobustWeight` / `RobustOutput`, `robust_weight=` on `kalman_measurement_update` / `kalman_filter` / `kalman_smoother`, `robust_weights=` on `kalman_maximization_step` and `weights=` on `measurement_cov_residual_form`, `weighted_kalman_measurement_update`; one sentence on the return convention and one on the EM bias/`core` recommendation, with the paper citation. README: one paragraph after "Package layout" ("Outlier-robust updates") with a three-line example (`from state_space_practice import kalman_filter, imq_weight`, call with `robust_weight=imq_weight(c=3.0)`, read `robust.weights`).

## Deliberately not in this phase

- `switching_kalman_filter`, the switching M-step and the oscillator models — phase 2 (they need the shared-weight design S1 and touch a different module).
- Anything under `point_process_kalman.py` — phase 3.
- Any change to `run_em` (none is needed; see [G4](designs.md#g4-em-sgd)).
- A consistency correction for the R bias (rejected in [G3](designs.md#g3-bias)).
- `parallel_kalman_smoother` (consumes filtered moments; works unchanged with robust outputs — assert this in one test rather than modifying it).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_zero_weight_mstep_preserves_observation_parameters` | All-zero weights return the previous H/R bit-for-bit, with finite outputs under eager, `jit` and `vmap`; dynamics match their unweighted updates for the same moments. Test the residual covariance helper directly with previous R. A missing fallback parameter raises `ValueError`; all-positive weights still solve the weighted stationary equations. |
| `test_none_is_bit_identical_measurement_update` | `kalman_measurement_update(...)` (default) on the `test_kalman_measurement_update_1d` inputs and on 5 `kalman_model_params` draws equals, with `assert_array_equal` on mean, cov and LL, a reference implementation kept in the test module: a verbatim copy of today's body (`kalman.py:455-475`, ~15 lines, jitted the same way). The `None` branch *is* that body, so this guards against accidental drift (e.g. a refactor that routes `None` through the weighted code with `w = 1`). |
| `test_none_is_bit_identical_filter_and_smoother` | `kalman_filter` / `kalman_smoother` with `robust_weight=None` on `simple_1d_model` and `_time_varying_r_model` equal the reference implementation bit-for-bit; existing `test_kalman.py`, `test_oracle_kalman.py`, `test_likelihood_identities.py::TestKalmanIdentities`, `test_invariances.py::TestKalmanInvariances` keep passing untouched. |
| `test_imq_weight_properties` | `IMQWeight(2.0) == IMQWeight(2.0)` with equal hashes; `imq_weight(c=1e150)(z) == 1.0`; `0 < w <= 1` on Hypothesis-drawn `z`; `w == 1` inside the core and `< 1` outside; `imq_weight(c=-1)` / `core=-1` raise `ValueError`; a plain lambda passed as `robust_weight` still runs (no hard type check) but `IMQWeight` is an instance of `RobustWeight`. |
| `test_c_to_infinity_matches_standard` | `imq_weight(c=1e150)` on `_time_varying_r_model`: means/covs/LL within `rtol=1e-12` of `None`; `robust.objective` within `rtol=1e-12` of `ll`; `robust.weights == 1`. |
| `test_weighted_update_equals_R_over_w2` | For `w ∈ {1, 0.7, 0.3, 0.05}`, `weighted_kalman_measurement_update(..., w)` mean/cov equal `kalman_measurement_update(..., R / w²)` at `rtol=1e-10`; at `w = 0` posterior equals prior exactly and `objective == 0` within `1e-12`. |
| `test_objective_matches_quadrature` | 1-D problem: `objective` from `weighted_kalman_measurement_update` equals `log scipy.integrate.quad(N(x; m, P) N(y; Hx, R)**w²)` at `rtol=1e-8` for `w ∈ {1, 0.5, 0.1, 1e-3}`. |
| `test_robust_filter_matches_dense_oracle_given_weights` | Run the robust filter (`c = 2`, several outliers injected) to get `robust.weights`; run `lgssm_dense_posterior` with `measurement_cov = R / w_t²` stacked over time; filtered/smoothed means and covariances agree at `RTOL = 1e-8` (reuse `_assert_matches_oracle`'s tolerancing); `robust.objective == oracle.log_likelihood + Σ_t ½(1 − w_t²)(d log 2π + log det R)` at `rtol=1e-8`. |
| `test_bounded_posterior_influence` | Contaminate `y_T` by `ε e₁`, `ε ∈ {1e2, 1e4, 1e6}`; KL between the contaminated and clean last-step posteriors (closed-form Gaussian KL): robust (`c = 3`) values at `1e4` and `1e6` agree within `1e-3` relative and are below `KL(N(m_pred, P_pred) ‖ N(m_0, P_0)) + 1e-6`; standard KF `KL(1e6) / KL(1e4) > 1e3`. Guard: the robust weight at `ε = 1e2` is `< 0.1` (the interesting regime was reached). |
| `test_contamination_state_rmse` | 2-state model, T = 1000, 5 % bins replaced by `N(0, (20σ)²)`; robust (`c = 3`) filtered-mean RMSE vs truth `< 0.5 ×` standard and `< 1.5 ×` the standard filter's RMSE on the clean series. |
| `test_covariances_psd_and_finite_under_contamination` | Same data: all robust filtered/smoothed covariances symmetric with `eigvalsh > 0`, all outputs finite, `weights ∈ (0, 1]`. |
| `test_unit_equivariance_robust` | Extend `TestKalmanInvariances.test_observation_units`' construction: scaling `y`, `H`-row units and `R` by `c` leaves robust means/weights invariant (weights are unit-free by construction). |
| `test_weighted_mstep_is_stationary_point` | Random smoothed moments and fixed weights; the weighted expected observation objective (written out in NumPy like `_q_function`) has vanishing finite-difference gradient at the returned `(H, R)` and increases from a random perturbation; with `robust_weights=None` output equals today's M-step bit-for-bit. |
| `test_em_recovers_R_under_contamination` (**slow**) | The test-only `run_em` loop, `imq_weight(c=3, core=√χ²_d(0.99))`, 5 % `20σ` contamination, T = 2000: final `R̂` within 15 % of truth; the `None` loop's `R̂ > 3 ×` truth; the robust objective history is non-decreasing up to `decrease_tol` (guard: at least 3 accepted iterations). |
| `test_run_em_history_is_the_robust_objective` | The test-only EM model with `robust_weight` set: `EMResult.log_likelihoods` equals the sequence of `RobustOutput.objective` values the E-steps returned (not the unweighted LLs, which are also recorded by the test model and differ by `> 1` nat on contaminated data — guard). |
| `test_gradients_flow` | `jax.grad` of `robust.objective` w.r.t. `R`, `A`, `init_cov` is finite and non-zero on `_time_varying_r_model`; `jax.jit` of `kalman_filter` with `robust_weight` compiles once for two equal `IMQWeight` instances (count traces via a wrapped weight). |
| `test_parallel_smoother_consumes_robust_filter_output` | `parallel_kalman_smoother(robust filtered means/covs)` equals `kalman_smoother(robust_weight=...)` smoothed moments at `rtol=1e-10`. |

Mark `test_em_recovers_R_under_contamination` with `@pytest.mark.slow` (it calls `run_em`, which `conftest.py` also auto-marks). Everything else must run in the fast suite (`T ≤ 1000`, jit warm-up included).

## Fixtures

- Reuse `simple_1d_model` (session fixture), `kalman_model_params` (Hypothesis), `_time_varying_r_model` and `_simulate_lgssm` (module helpers — import from their test modules).
- New module-scoped fixture `contaminated_lgssm` in `test_robust_kalman.py`: a seeded 2-state / 1-obs model (`A = [[1, 0.1], [0, 0.95]]`, `Q = 0.01 I`, `H = [1, 0]`, `R = 0.2`), T = 1000, truth states, clean observations, and a contaminated copy with 5 % of bins replaced by `N(0, (20 √R)²)` draws; returns arrays plus the outlier mask (used to assert `weights[mask] < 0.5` and `weights[~mask] > 0.9` on median).
- No real-data slice: the phase is exact-math; real LFP arrives with the oscillator models in phase 2.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
