# Phase 3 — WoLF on the Laplace-EKF point-process path

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#p1-deviance)

Depends on phase 1 (`RobustWeight`, `RobustOutput`, `imq_weight`); independent of phase 2.

**Inputs to read first:**

- [src/state_space_practice/point_process_kalman.py:44-62](../../../src/state_space_practice/point_process_kalman.py) — imports (add `RobustOutput`, `RobustWeight` from `utils`).
- [src/state_space_practice/point_process_kalman.py:927-1212](../../../src/state_space_practice/point_process_kalman.py) — `_point_process_laplace_update`: prior factor (1090-1091), `_neg_log_posterior` (1093-1103), `_fisher_step_at` (1105-1143), single step (1145-1162), line search (1163-1176), LL (1183-1190), normalisation (1192-1208), return (1210-1212).
- [src/state_space_practice/point_process_kalman.py:1215-1297, 1300-1331, 1333-1464](../../../src/state_space_practice/point_process_kalman.py) — `GLMFamily`, `poisson_family`, the Bernoulli family, the `@overload` pattern, `glm_laplace_update` (same structure as the legacy update).
- [src/state_space_practice/point_process_kalman.py:1467-1734, 1737-1866](../../../src/state_space_practice/point_process_kalman.py) — `stochastic_point_process_filter` (block dispatch 1657-1703, impl call 1714-1726) and `_stochastic_point_process_filter_impl` (`static_argnames` 1739-1743, scan step 1800-1846).
- [src/state_space_practice/point_process_kalman.py:1869-1979, 1982-2030, 2033-2122, 2152-2266, 2269-2392, 2392-2602](../../../src/state_space_practice/point_process_kalman.py) — block forward core (per-neuron step 1915-1955), its wrapper, the block smoother core (forward call 2064-2076), the block filter (forward 2239-2248, return 2266), the block smoother, `stochastic_point_process_smoother` (block dispatch 2526-2569, filter call 2571-2588, return 2599-2602).
- [src/state_space_practice/point_process_kalman.py:2993-3063, 3065-3105, 3150-3230, 3234-3288, 3314-3336, 3348-3371](../../../src/state_space_practice/point_process_kalman.py) — `PointProcessModel`.
- [src/state_space_practice/place_field_model.py:361-426, 983-1024, 1364-1365, 1392-1403, 1574-1609, 1623-1658, 1886-1979, 2025-2063](../../../src/state_space_practice/place_field_model.py) — constructor, `_e_step`, the `run_em` closure and call, `_sgd_loss_fn`, `_finalize_sgd`, `score`, `bic`/`aic`.
- [src/state_space_practice/position_decoder.py:35, 870-914, 958-1027, 1084-1178, 1181-1257, 1260-1273, 1441-1470, 1531-1536, 1539-1612, 1665-1709, 1774-1841](../../../src/state_space_practice/position_decoder.py) — the Laplace-update import, `DecoderResult`, `_run_filter_scan` (statics 958-968, `_step` 1084-1153 with the penalty downdate 1100-1107, inflation 1109-1131, Laplace call 1133-1144), its call site, `DecoderResult` construction, the filter/smoother wrappers, `PositionDecoder.__init__` (validation pattern 1691-1706) and `decode`.
- [src/state_space_practice/switching_point_process.py:580-600](../../../src/state_space_practice/switching_point_process.py) and `hamiltonian_core.py:179`, `coupling_ekf.py:96` — other callers of the two Laplace updates; they must keep working unchanged (default keyword).
- [src/state_space_practice/tests/test_glm_laplace.py:24-68](../../../src/state_space_practice/tests/test_glm_laplace.py) — `TestPoissonParity` (the sibling-parity test to extend with `robust_weight`).
- [src/state_space_practice/tests/test_point_process_kalman.py:140-180, 1299-1363, 3208-3276, 4279, 5304](../../../src/state_space_practice/tests/test_point_process_kalman.py) — `point_process_test_data`, `multi_neuron_test_data` + `multi_neuron_log_intensity`, `TestBlockDiagonalFilterEquivalence._make_problem`, `TestPointProcessModelRecovery`, `TestScaleEquivariance`.
- [src/state_space_practice/tests/test_likelihood_identities.py:579-653](../../../src/state_space_practice/tests/test_likelihood_identities.py) — `_pp_problem`, `_laplace_terms`: the per-step Laplace evidence identity to extend to the tempered form.
- [src/state_space_practice/tests/test_place_field_model.py:84-96, 145, 1735](../../../src/state_space_practice/tests/test_place_field_model.py) and [tests/test_position_decoder.py:531, 1222-1316, 1459](../../../src/state_space_practice/tests/test_position_decoder.py) — fixtures (`sim_data`, `realistic_decoding`) and the classes whose setup to reuse.
- [src/state_space_practice/tests/test_em_golden_regression.py:239-262, 791-793](../../../src/state_space_practice/tests/test_em_golden_regression.py) — point-process / place-field golden cases (bit-identical `None` path).

**Contracts referenced:**

- [Return-arity rule](shared-contracts.md#return-arity) — rows for the two Laplace updates, the filter, the smoother, `DecoderResult.robust`.
- [Weight granularity](shared-contracts.md#weight-granularity) — one weight per (time step, neuron) from the signed deviance residual; recommended `imq_weight(c, core=3.0)`.
- [Invariants](shared-contracts.md#invariants) 1–6 (the two Laplace updates and the dense/block cores are siblings).

**Designs referenced:** [P1](designs.md#p1-deviance), [P2](designs.md#p2-filters), [G4](designs.md#g4-em-sgd).

## Tasks

- **`GLMFamily` additions.** Add `unit_deviance` and reuse/add the trailing optional `loglik_per_obs` field shared with the masks and GLM plans (preserve existing optional fields/order) ([P1](designs.md#p1-deviance)); provide them in `poisson_family` (with `xlogy`) and `BERNOULLI_LOGIT_FAMILY` (with the existing logit clipping). Add a private module helper `_poisson_unit_deviance(y, mu)` used by both the Poisson family and the legacy update, and `_deviance_weights(y, mu, unit_deviance, robust_weight)` (stop-gradient, per-observation vmap of the weight on a length-1 vector). Docstring for `GLMFamily`: what the two fields are for and that `robust_weight` requires them.

- **Tempered Laplace updates (siblings, changed together).** `_point_process_laplace_update` and `glm_laplace_update` gain keyword-only `robust_weight: RobustWeight | None = None`. `None` → existing bodies unchanged. Robust: compute `w`, `w2` at the prior mean before any iteration; temper the line-search objective, the score and the Fisher information by `w2` ([P1](designs.md#p1-deviance)); after the mode, compute `per_obs`, the tempered `objective` (using the robust precision factor already in hand) and the unweighted `log_likelihood` (one extra `psd_cholesky` / `psd_logdet` of `P⁻¹ + Jᵀ diag(v*) J` when normalised); append `RobustOutput(objective, weights=w)` after `n_line_search_failures` if requested. Extend the `@overload` stubs of `glm_laplace_update` (four combinations) and raise `ValueError` if the family lacks the two fields. Docstrings: paper App. D.1 lineage, deviance standardisation and the Pearson numbers in one sentence, the `core` recommendation, both returned objectives.

- **Dense and block-diagonal filters and smoothers.** `_stochastic_point_process_filter_impl` and `_block_diagonal_forward_core` / `_block_diagonal_smoother_core`: add `robust_weight` to `static_argnames`, thread it to the update, carry the objective and emit weights ([P2](designs.md#p2-filters)); block cores transpose per-neuron weights to `(n_time, n_neurons)` and sum objectives. `_run_forward_block_diagonal`, `_stochastic_point_process_filter_block_diagonal`, `_stochastic_point_process_smoother_block_diagonal`, `stochastic_point_process_filter`, `stochastic_point_process_smoother`: pass-through and append `RobustOutput` per the arity rule; `@overload` stubs on the two public functions. Docstrings updated (parameter, returns, that the block path yields the same per-neuron weights as the dense path).

- **`PointProcessModel` opt-in.** Constructor `robust_weight: RobustWeight | None = None` (stored; forwarded in `_e_step`, `_sgd_loss_fn`, `_finalize_sgd`); `_e_step` returns `robust.objective` and stores `self.filter_robust_weights` (`(n_time, n_neurons)`) when set, and the `fit` snapshot/restore/clear closures include it; `_sgd_loss_fn` returns `-robust.objective`; `_finalize_sgd` sets `log_likelihood_` to the objective when robust. Docstrings on `fit` / `fit_sgd` / class.

- **`PlaceFieldModel` opt-in.** Constructor keyword (validated like `max_newton_iter`); forwarded in `_e_step`, `_sgd_loss_fn`, `_finalize_sgd`; `_e_step` returns the objective and stores `filter_robust_weights`; `fit`'s `_capture_state` / `_restore_state` / `_clear_posteriors` include it; `score()` keeps calling the filter **without** `robust_weight` (documented: an unweighted held-out predictive score); `bic()` / `aic()` docstrings: computed from the generalised objective when robust, comparable only across fits with the same `robust_weight`.

- **`PositionDecoder` opt-in.** `_run_filter_scan` gains static `robust_weight`; in `_step` compute per-neuron weights from the prior-predictive rate after the penalty downdate and before inflation, feed `w2` into the inflation statistic, pass `robust_weight` to the Laplace update, carry the objective and emit weights; return two extra outputs. `_position_decoder_filter_with_predictions` / `position_decoder_filter` / `position_decoder_smoother` gain `robust_weight=None`; `DecoderResult.__init__` gains `robust: RobustOutput | None = None` (attribute; `__repr__` unchanged); `PositionDecoder.__init__(..., robust_weight=None)` stores it and `decode` forwards it. Docstrings: `marginal_log_likelihood` remains unweighted; `result.robust.weights` is the per-bin/per-neuron diagnostic.

- **Tests — new module `tests/test_robust_point_process.py`** implementing the validation slice; extend `TestPoissonParity` with `robust_weight` parity between the two Laplace updates; extend `TestBlockDiagonalFilterEquivalence` with a robust case.

- **User-facing docs.** CHANGELOG `### Added` bullets: `robust_weight=` on `glm_laplace_update` / the point-process filter and smoother / `position_decoder_filter` / `position_decoder_smoother`, constructor keyword on `PointProcessModel` / `PlaceFieldModel` / `PositionDecoder`, `GLMFamily.unit_deviance` / `loglik_per_obs`, `DecoderResult.robust`; one sentence on deviance standardisation and the `core=3` recommendation. README: extend the "Outlier-robust updates" paragraph with the spike-train example (`PlaceFieldModel(dt, robust_weight=imq_weight(c=4.0, core=3.0))`).

## Deliberately not in this phase

- The switching point-process family (`switching_point_process.py:595` caller and `point_process_models.py`) — overview non-goal; the legacy update's new keyword is simply not passed there.
- `hamiltonian_core.py:179` / `coupling_ekf.py:96` — untouched callers.
- A joint per-bin weight across neurons for spikes (paper's scalar `W`) — rejected for this phase in [contracts](shared-contracts.md#weight-granularity); revisit if simultaneous multi-unit artifacts dominate a dataset.
- A robust `PlaceFieldModel.score` (open question 3).
- Any change to `dynamics_only_m_step` (the point-process M-steps estimate no observation-noise parameter, so no weighting applies).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_laplace_updates_none_bit_identical` | Both Laplace updates with default `robust_weight` equal reference copies of the pre-change bodies bit-for-bit on the `test_glm_laplace` fixed setup and `multi_neuron_test_data`; `test_glm_laplace.py`, `test_point_process_kalman.py`, `test_oracle_point_process.py`, `test_likelihood_identities.py::TestPointProcessIdentities`, the point-process / place-field golden cases and `test_position_decoder.py` pass untouched. |
| `test_robust_poisson_parity` (extends `TestPoissonParity`) | `glm_laplace_update(poisson_family(dt), robust_weight=imq_weight(4.0, core=3.0))` equals `_point_process_laplace_update(..., robust_weight=...)` at `1e-10` on mean, cov, LL, objective and weights for `max_newton_iter ∈ {1, 3}` and both normalisation settings. |
| `test_deviance_weights_values` | Weights on `(y, μ) ∈ {(1, 0.05), (2, 0.05), (8, 0.05), (8, 2.0)}` with `imq_weight(4.0, core=3.0)` equal `(1.0, 0.947, 0.470, 0.965)` (√ of the `w²` column `(1.00, 0.896, 0.221, 0.931)` in [P1](designs.md#p1-deviance)) at `2e-3`; `y = 0` gives `w = 1`; weights are `stop_gradient`ed (`jax.grad` of `Σ w` w.r.t. the prior mean is `0`). |
| `test_c_to_infinity_matches_standard_glm` | `imq_weight(c=1e150)`: both updates within `1e-12` of `None`; `objective == ll`. |
| `test_tempered_update_is_weighted_map` | The robust posterior mean is the maximiser of `Σ_n w_n² log p(y_n; x) − ½ (x − m)ᵀ P⁻¹ (x − m)` (gradient at the mode `< 1e-8` after `max_newton_iter=20` on a well-conditioned problem); the robust precision equals `P⁻¹ + Jᵀ diag(w² v*) J` at `1e-10`. |
| `test_objective_is_tempered_laplace_evidence` | Extend `_laplace_terms` from `test_likelihood_identities.py`: the filter's `robust.objective` equals the sum of per-step tempered Laplace terms recomputed from the filter's own weights/moments at `1e-8`; `log_likelihood` equals the untempered terms with the unweighted curvature at the robust mode. |
| `test_robust_glm_gaussian_limit_matches_wolf` | Bernoulli-free check: with a *Gaussian-like* large-count Poisson regime (μ ≈ 200) the tempered Fisher update's mean/precision agree with `weighted_kalman_measurement_update` using `R = diag(μ)` and the same weight at `rtol=1e-2` (documents the App. D.1 lineage; guard: weights `< 0.9` at the perturbed bin). |
| `test_dense_block_equivalence_robust` (extends `TestBlockDiagonalFilterEquivalence`) | With `robust_weight=imq_weight(4.0, core=3.0)` and `max_newton_iter=1`: block and dense filtered means/covs, `robust.weights` and `robust.objective` agree at `1e-10`; with `max_newton_iter=3` within the class's existing looser tolerance. |
| `test_filter_smoother_arity_and_shapes` | Robust filter returns 4-tuple with `weights.shape == (n_time, n_neurons)`; smoother with `return_filtered=True` returns 7 elements with `RobustOutput` last; `None` arity unchanged. |
| `test_clean_data_rate_bias_is_negligible` | Low-rate multi-neuron simulation (μ ≈ 0.05–0.2 per bin, T = 2000): posterior mean rate per neuron with `imq_weight(4.0, core=3.0)` within 2 % of the `None` estimate; with `imq_weight(4.0)` (no core) the bias exceeds 5 % (documents why `core` is recommended — guard that the design concern is real). |
| `test_burst_injection_latent_rmse` | `multi_neuron_test_data`-style simulation with 2 % of bins given an 8-spike burst on one neuron: robust smoothed latent RMSE vs `true_states` `< 0.6 ×` standard; weights at burst bins `< 0.5`, elsewhere median `> 0.95`. |
| `test_bounded_influence_counts` | Increase one bin's count for one neuron through `{10, 100, 1000}`: the robust posterior-mean shift is non-increasing beyond `100` and bounded by the shift at `10`; the standard shift grows (guard: `> 2×` from 10 to 1000). |
| `test_covariances_psd_and_finite` | Robust dense and block outputs under bursts: symmetric, `eigvalsh > 0`, finite; `weights ∈ (0, 1]`. |
| `test_scale_equivariance_robust` (extends `TestScaleEquivariance`) | Rescaling the latent by `c` (covariances by `c²`, design by `1/c`) rescales robust means exactly and leaves weights unchanged. |
| `test_point_process_model_fit_robust` (**slow**) | `PointProcessModel(robust_weight=imq_weight(4.0, core=3.0)).fit` on burst-injected data: recovers `true_params` in `point_process_test_data` within the tolerance `TestPointProcessModelRecovery` uses, where the `None` fit misses by `> 2×` that tolerance (guard); history is the objective and `filter_robust_weights` is populated. |
| `test_place_field_model_fit_robust` (**slow**) | `PlaceFieldModel(dt, robust_weight=...)` on `sim_data` with injected bursts: block path used (`_block_n_neurons` set), rate-map RMSE vs truth `< 0.7 ×` the `None` fit, `score()` returns an unweighted LL (equal to a direct filter call without `robust_weight`), `bic()` uses the objective. |
| `test_position_decoder_robust` | `realistic_decoding` fixture with 2 % burst bins: robust decode error (median) `≤` standard; `result.robust.weights.shape == (n_time, n_neurons)`; `result.marginal_log_likelihood` equals the unweighted sum; with `adaptive_inflation` on, `n_capped_bins` is not larger than without bursts (inflation does not fire on ignored spikes). |
| `test_fit_sgd_runs_with_robust_weight` (**slow**) | `PointProcessModel.fit_sgd(robust)` 30 steps: finite objective history, finite parameters; `PlaceFieldModel.fit_sgd(robust)` likewise. |

## Fixtures

- Reuse `point_process_test_data`, `multi_neuron_test_data`, `multi_neuron_log_intensity`, `TestBlockDiagonalFilterEquivalence._make_problem`, `sim_data` (place field), `realistic_decoding` (decoder), `_pp_problem` / `_laplace_terms` (likelihood identities).
- New module-scoped fixture `burst_injected(multi_neuron_test_data)`: copies the spikes, sets 2 % of bins (seeded) for a random neuron to 8 counts, returns spikes, mask and the original `true_states`.
- New module-scoped fixture `low_rate_population`: 6 neurons, `dt = 0.02`, rates 2–10 Hz (μ ≈ 0.04–0.2), T = 2000, with a fixed latent trajectory — used by the clean-data bias test.
- Real data: none checked in; the PR description reports a smoke run of `PlaceFieldModel` with and without `robust_weight` on one session from `notebooks/`, with the fraction of `(bin, neuron)` weights below 0.5.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
