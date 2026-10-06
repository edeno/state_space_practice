# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

All paths are relative to `src/state_space_practice/`. Line numbers were verified against the working tree at commit `4b74f03`.

### Phase 1 (Gaussian path)

- `kalman.py:28-37` — imports from `utils`: gains `RobustOutput`, `RobustWeight` and the standardised-residual helper; `import functools` is added (not currently imported).
- `kalman.py:426-475` — `kalman_measurement_update` (`@jax.jit` at line 426, body 455-475): gains keyword-only `robust_weight=None` as a **static** jit argument; the `None` branch is the existing code unchanged; the comment at 468-473 (LL evaluated on the unboosted `obs_cov`) is preserved for the unweighted LL.
- `kalman.py:478-526` — `_kalman_filter_update` and `kalman.py:529-610` — `_kalman_filter_impl` (scan step 570-594, scan 604-608): thread `robust_weight`; when active the scan additionally emits the per-step weight and objective.
- `kalman.py:613-711` — `kalman_filter` (validation 683-701, impl call 703-711) and `kalman.py:944-1044` — `kalman_smoother` (impl call 1036-1044) / `kalman.py:915-941` — `_kalman_smoother_impl`: keyword pass-through; `rts_backward_scan` (778-842) is **untouched** (RTS consumes filtered moments).
- `kalman.py:1364-1396` — `measurement_cov_residual_form`: gains `weights=None`.
- `kalman.py:1448-1551` — `kalman_maximization_step` and `kalman.py:1554-1628` — `_kalman_maximization_step` (H at 1577, R at 1578-1581, transition block 1585-1619): gains `robust_weights=None`; H and R use the weighted forms, A / Q / initial state unchanged. The jitted inner function already branches on a `None` argument (`initial_state_prior`, line 1585), which is the pattern to copy.
- `kalman.py:1072-1216` — `parallel_kalman_smoother`, `kalman.py:104-204` — `woodbury_kalman_gain`, `kalman.py:846-912` — `rts_backward_scan_with_predictions`: untouched.
- `utils.py` — new section (after `check_converged`, `utils.py:1662-1719`) holding the protocol, `IMQWeight`, `imq_weight`, `RobustOutput` and `standardized_residual`; `utils.py:1161` `validate_scalar` is reused for `c`.
- `em_driver.py:58-77, 222-299` — `run_em`: **no change**. The convergence quantity is whatever the model's `e_step` closure returns (`current_ll = float(e_step())` at line 223), so switching to the robust objective is a change in the models' closures, not in the driver.
- `__init__.py:35-53` (`_LAZY_API`), `55-76` (`__all__`), `78-100` (`TYPE_CHECKING` imports): `imq_weight` is added as a lazily-loaded public name.

### Phase 2 (switching filter and oscillator models)

- `switching_kalman.py:23-28` — imports from `kalman` (gains `weighted_kalman_measurement_update`, `standardized_residual`).
- `switching_kalman.py:49-57` — `_kalman_filter_update_per_discrete_state_pair` (double vmap over `_kalman_filter_update`): a weighted sibling is added next to it; the existing object is untouched (still used by the Viterbi decoder at `switching_kalman.py:1144-1156`, which stays non-robust).
- `switching_kalman.py:686-790` — `_first_timestep_kalman_update` (vmap of `kalman_measurement_update` at 747-758): gains `robust_weight`.
- `switching_kalman.py:793-1076` — `switching_kalman_filter` (`@jax.jit` at 793; `_step` 871-988 with the per-pair update at 933-945, the discrete update at 947-957, the LL accumulation at 962 and the collapse at 967-973; first step 992-1007; scan 1022-1042; return 1068-1076): gains static keyword-only `robust_weight=None` and an eighth output when active.
- `switching_kalman.py:454-597` — `_update_discrete_state_probabilities`: **untouched**; called a second time on the unweighted per-pair LLs when robust (only its `log_predictive` output is used).
- `switching_kalman.py:1386-1404` — `switching_kalman_smoother`, `switching_kalman.py:1686` — `switching_kalman_smoother_gpb2`: untouched (they consume filter outputs).
- `switching_kalman.py:2150-2300` — `_switching_kalman_m_step_inner` (`n_time` 2206, `gamma`/`delta` 2210-2220, `gamma2` 2222-2229, H 2232-2235, R 2238-2251, Q 2254-2269) and `switching_kalman.py:2303-2597` — `switching_kalman_maximization_step` (inner call 2488-2512, occupancy gate 2514-2539): gain `robust_weights=None`; the observation block (H, R) is weighted, the transition block and its occupancy gate are not. An independent zero-effective-weight observation gate preserves previous H/R.
- `oscillator_models.py:440-533` — `BaseModel.__init__` (flags stored 516-523): gains `robust_weight: RobustWeight | None = None`; the three subclass constructors forward `**kwargs` to it (`1258-1260`, `1518-1524`, `1906-1912`) so no subclass signature changes.
- `oscillator_models.py:415-438` — `_EM_SNAPSHOT_KEYS`; `535-557` snapshot/restore; `559-576` `_clear_smoother_state`: a `filter_robust_weights` attribute joins the snapshot.
- `oscillator_models.py:886-970` — `BaseModel._e_step` (filter call 905-923, return 970): returns the robust objective and stores the weights when active.
- `oscillator_models.py:972-1030` — `BaseModel._m_step` (M-step call 991-1012, R pooling 1021-1022) and `oscillator_models.py:2023-2129` — `DirectedInfluenceModel._m_step_reparameterized` (M-step call 2045-2069): pass `robust_weights=`.
- `oscillator_models.py:1082-1142` — `BaseModel.fit` (run_em 1130-1140): docstring only; the `e_step` closure at 1131 already returns whatever `_e_step` returns.
- `oscillator_models.py:1444-1462`, `1777-1811`, `2253-2313` — the three `_sgd_loss_fn`s read `result[6]`; they switch to the robust objective when active.

### Phase 3 (point-process path)

- `point_process_kalman.py:1215-1239` — `GLMFamily`: optional trailing `unit_deviance` plus the shared `loglik_per_obs` field (reuse it when masks/GLM has landed); `1242-1265` `poisson_family` and `1276-1297` the Bernoulli family provide them.
- `point_process_kalman.py:927-1212` — `_point_process_laplace_update` (prior factor 1090-1091, `_neg_log_posterior` 1093-1103, `_fisher_step_at` 1105-1143, single step 1145-1162, line search 1163-1176, LL 1183-1190, normalisation 1192-1208, return 1210-1212) and `point_process_kalman.py:1333-1464` — `glm_laplace_update` (same structure at 1393-1464; overloads 1300-1331): both gain keyword-only `robust_weight=None`. They are siblings and must change together (see the `fix-sibling-implementations` memory).
- `point_process_kalman.py:1467-1734` — `stochastic_point_process_filter` (block dispatch 1657-1703, impl call 1714-1726), `1737-1866` — `_stochastic_point_process_filter_impl` (`static_argnames` 1739-1743, Laplace call 1820-1833), `1869-1979` — `_block_diagonal_forward_core` (`static_argnames` 1869-1872, per-neuron Laplace call 1935-1946), `1982-2030` — `_run_forward_block_diagonal`, `2033-2122` — `_block_diagonal_smoother_core`, `2152-2266` — `_stochastic_point_process_filter_block_diagonal`, `2269` — `_stochastic_point_process_smoother_block_diagonal`, `2392-2602` — `stochastic_point_process_smoother` (block dispatch 2526-2569, filter call 2571-2588, return 2599-2602): thread `robust_weight`.
- `point_process_kalman.py:2928-3474` — `PointProcessModel` (`__init__` 2993-3063, `_e_step` 3065-3105, `fit` 3150-3230 with `run_em` at 3219-3229, `fit_sgd` 3234-3288, `_sgd_loss_fn` 3314-3336, `_finalize_sgd` 3348-3371): constructor opt-in.
- `place_field_model.py:361-426` — `PlaceFieldModel.__init__`; `983-1024` `_e_step`; `1364-1365` the `run_em` closure; `1574-1609` `_sgd_loss_fn`; `1623-1658` `_finalize_sgd`; `1886-1979` `score` (filter call 1961-1978); `2025-2063` `bic` / `aic`: constructor opt-in; `score` stays unweighted.
- `position_decoder.py:958-968` — `_run_filter_scan` static args; `1109-1131` the inflation block; `1133-1144` the Laplace call; `1441-1470` the `_run_filter_scan` call site; `1531-1536` `DecoderResult` construction; `870-914` `DecoderResult`; `1181-1257` `position_decoder_filter`; `1260-1273` `_position_decoder_filter_with_predictions` signature; `1539-1612` `position_decoder_smoother`; `1665-1709` `PositionDecoder.__init__`; `1774-1841` `decode`: opt-in threading.
- `hamiltonian_core.py:179` and `coupling_ekf.py:96` call `glm_laplace_update` positionally without the new keyword: untouched by construction.
- `switching_point_process.py:595` calls `_point_process_laplace_update` for the switching point-process family: untouched in this plan (see Non-Goals).

## Scope and dependency policy

### Goals

- A single `RobustWeight` protocol and `utils.imq_weight(c, core=0.0)` factory usable as a jit-static argument by every filter in the library.
- WoLF measurement updates exactly as in Duran-Martin et al. (2024), Proposition 3.1 / Algorithm 1, on the Gaussian path, with the Mahalanobis-standardised IMQ weight of their Eq. (18); the update is written in a form that stays finite at weight zero.
- The unweighted one-step predictive log-likelihood is still returned; a generalised-Bayes objective (`RobustOutput.objective`) and the per-step weights (`RobustOutput.weights`) are returned in addition.
- EM through `run_em` monitors the generalised-Bayes objective when a robust weight is active; the M-step for the observation block (H, R) uses the weighted forms; A, Q and the initial state are unchanged.
- The oscillator models (LFP artifacts) and the point-process models (bursts, sorting errors) expose the option through their constructors; `fit`, `fit_sgd`, `decode` and the posteriors all respect it.
- `robust_weight=None` is bit-identical to today at every public entry point (asserted with `assert_array_equal`); the EM golden-value regression (`tests/test_em_golden_regression.py:791-793`) keeps passing.

### Non-Goals

- The switching point-process models (`switching_point_process.py`, `point_process_models.py`, including `DirectedInfluencePointProcessModel`). Their Laplace update at `switching_point_process.py:595` would accept the new keyword mechanically, but the model-level opt-in threads through `SwitchingPointProcessBase` and its GPB2 pair-conditional paths. **Revisit trigger:** when `DirectedInfluenceModel` ships with `robust_weight` (end of phase 2), the `dim-pp-mirrors-gaussian-dim` memory requires a follow-up issue for DIM-PP; open it in the phase 2 PR description rather than widening this plan.
- Robust Viterbi decoding (`switching_kalman_viterbi`) — decoding under tempered likelihoods is a separate question.
- Per-channel (dimension-specific, paper Appendix D.2) weights on the Gaussian path — LFP artifacts are treated as common-mode with one weight per time step. Point-process weights are per neuron by design (see [shared-contracts.md](shared-contracts.md#weight-granularity)).
- Learning `c` (or `core`) by gradient descent or EM; it is a fixed hyperparameter, as in the paper (tuned there by Bayesian optimisation, Section 4).
- Any change to `run_em`, `parallel_kalman_smoother`, `woodbury_kalman_gain`, or the Student-t / variational-Bayes alternatives (see Literature).
- A robust *held-out* score (`PlaceFieldModel.score` and `DecoderResult.marginal_log_likelihood` stay unweighted predictive log-likelihoods).

### Dependency policy

No new runtime dependency. `scipy` (already a dependency) is used only in tests (quadrature, χ² quantiles).

Cross-plan links (other plans written in parallel; do not restate their content):

- `docs/plans/streaming-filters/` consumes `robust_weight=` on `kalman_measurement_update` and passes it through its streaming step. Phase 1 must land first; the names it relies on are exactly `robust_weight=`, `RobustWeight`, `utils.imq_weight(...)` and the return convention in [shared-contracts.md](shared-contracts.md#return-arity).

## Metrics

| Property | Check | Tolerance |
| --- | --- | --- |
| Backwards compatibility | `robust_weight=None` outputs vs pre-change outputs on existing fixtures (`simple_1d_model`, `_time_varying_r_model`, `point_process_test_data`, `multi_neuron_test_data`, oscillator fixtures) | `assert_array_equal` (bit-identical) |
| Limit `c → ∞` | `imq_weight(c=1e150)` vs `None` | `rtol=1e-12` on means, covariances, LL, and `objective == ll` |
| Exactness given the weights | robust filter vs `tests/oracles.lgssm_dense_posterior` run with `R_t = R / w_t²` | `rtol=1e-8` (the oracle suite's `RTOL`) |
| Objective correctness | `weighted_kalman_measurement_update` objective vs 1-D `scipy.integrate.quad` of `∫ N(x; m, P) N(y; Hx, R)^{w²} dx` for `w ∈ {1, 0.5, 0.1, 1e-3}` | `rtol=1e-8` |
| Bounded influence (paper Thm 3.2) | KL posterior-influence at contamination `ε ∈ {1e2, 1e4, 1e6}` | robust: `PIF(1e6)` and `PIF(1e4)` within `1e-3` relative and below the "observation ignored" limit + `1e-6`; standard: `PIF(1e6) / PIF(1e4) > 1e3` |
| Contamination (5 % of bins, `20σ` Gaussian outliers) | state RMSE vs truth | robust `< 0.5 ×` standard and `< 1.5 ×` the clean-data KF RMSE |
| EM recovery under contamination | `R̂` after EM with `imq_weight(c=3, core=√χ²_d(0.99))` vs truth; standard EM | robust within 15 % of truth; standard inflated `> 3×` |
| Point-process rate bias on clean data (design invariant) | posterior rate at a low-rate neuron with `imq_weight(c=4, core=3)` vs `None` | within 2 % |
| Point-process burst rejection | injected 8-spike bursts at 2 % of bins | robust latent RMSE `< 0.6 ×` standard |
| Gradients | `jax.grad` of the robust objective w.r.t. R, A, init_cov | finite and non-zero; `fit_sgd` runs (slow) |

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Return arity of public functions changes when the option is on (an appended `RobustOutput`) could surprise callers | Only when `robust_weight` is not `None`; documented in every docstring and typed with `@overload` on the plain-Python wrappers; matches existing precedents (`return_line_search_failures`, `return_filtered`, `return_block_covariances`). |
| The weighted-residual R M-step is biased low on clean data with the plain IMQ (EM fixed point `R̂/R ≈ 0.83–0.93` for `c = 3–5`) | Quantified in [designs.md](designs.md#g3-bias); `imq_weight(..., core=...)` makes the bias ≤ 2.5 %; the recommended EM setting is documented; no consistency-correction machinery (it would break the fixed-weight EM monotonicity). |
| Pearson-standardised weights make every spike in a low-rate bin an outlier (−53 % rate bias at `μ = 0.05`) | The point-process path standardises with the signed unit-deviance residual and the plan recommends `core ≈ 3`; numbers in [designs.md](designs.md#p1-deviance). |
| Gradient descent could drive R → 0 so that every weight → 0 and the objective → 0 (a spurious plateau) | Weights are computed under `jax.lax.stop_gradient` in every filter ([shared-contracts.md](shared-contracts.md#invariants)); the parameter gradient is that of the fixed-weight objective, mirroring the EM M-step. |
| Per-pair weights in the switching filter would *favour* poorly fitting discrete states (their evidence is tempered more) | One shared weight per time step from the mixture prior predictive ([designs.md](designs.md#s1-shared-weight)). |
| Recompilation on every call if the weight object hashes by identity | `IMQWeight` is a frozen dataclass (value hash/eq); a plain lambda retraces (verified); documented on the protocol. |
| `w → 0` underflow (`R / w²` infinite) | The update uses `S̃ = w² H P Hᵀ + R`, finite at `w = 0` exactly ([designs.md](designs.md#g1-update)). |
| EM objective can decrease because weights are re-evaluated between iterations | Documented as generalised EM; `run_em`'s existing rollback-and-stop policy is the safety net; open question 1 tracks whether `decrease_tol` needs relaxing. |

## Rollout Strategy

All at once per phase, behind the additive keyword. Every signature gains `robust_weight: RobustWeight | None = None` (keyword-only, jit-static); models gain the same constructor keyword. No deprecation, no flags, no parallel code paths: the `None` branch *is* the existing code, and the robust branch is the only new code. Users who do not opt in see identical numerics (asserted). Phase order: 1 → {2, 3} (2 and 3 independent).

## Open Questions

1. **`decrease_tol` under robust EM.** Weights are recomputed each E-step, so the generalised objective is not guaranteed monotone even though the fixed-weight M-step is exact. Current best answer: keep the models' existing `tol`/`decrease_tol`; `run_em` rolls back and stops on a decrease, which is safe. Trigger to revisit: the phase 2 contamination EM test stops by rollback before recovering R.
2. **Per-channel Gaussian weights (paper D.2).** Current best answer: deferred; common-mode artifacts dominate the target data. Trigger: a recording where artifacts hit a subset of channels.
3. **Robust held-out score.** `score()` and `DecoderResult.marginal_log_likelihood` stay unweighted. Trigger: model selection under contaminated held-out data.
4. **DIM-PP parity.** Deferred to a follow-up opened from the phase 2 PR (see Non-Goals).
5. **Default `c`.** No default; the docstring gives the χ²_d heuristic (`c² ≈` the 0.99 quantile of χ²_d puts `w = 1/√2` at that quantile) as a starting point and cites the paper's Bayesian-optimisation tuning. Trigger: user feedback after first real-data use.

## Estimated Effort

- Phase 1: `kalman.py` ≈ +180 LOC, `utils.py` ≈ +110, `__init__.py` +3, tests ≈ +380, CHANGELOG/README ≈ +40.
- Phase 2: `switching_kalman.py` ≈ +230, `oscillator_models.py` ≈ +80, tests ≈ +320, CHANGELOG ≈ +15.
- Phase 3: `point_process_kalman.py` ≈ +280, `position_decoder.py` ≈ +70, `place_field_model.py` ≈ +40, tests ≈ +420, CHANGELOG/README ≈ +25.

## Literature

Each entry states the claim it supports in this plan.

- Duran-Martin, G., Altamirano, M., Shestopaloff, A. Y., Sánchez-Betancourt, L., Knoblauch, J., Jones, M., Briol, F.-X., Murphy, K. (2024). *Outlier-robust Kalman filtering through generalised Bayes.* ICML 2024, PMLR 235; arXiv:2405.05646. Read in full for this plan. Supports: the weighted loss `ℓ_t(θ) = −W²(y_t, ŷ_t) log q(y_t | θ)` (Eq. 15); the closed-form update with `R_t⁻¹` replaced by `W² R_t⁻¹` (Prop. 3.1, Algorithm 1); the IMQ weight with Euclidean norm (Eq. 17) and the Mahalanobis-standardised IMQ (Eq. 18, "WoLF-MD") and thresholded variant (Eq. 19); the bounded posterior-influence theorem and its two conditions `sup W < ∞`, `sup W² ‖y‖ < ∞` (Thm 3.2, Lemma C.2); the exponential-family sketch that weights the moment-matched Gaussian by `W²` and leaves the weight choice for non-Gaussian families open (App. D.1, Eq. 60); dimension-specific weights (App. D.2); hyperparameter `c` tuned by Bayesian optimisation (Section 4).
- Knoblauch, J., Jewson, J., Damoulas, T. (2022). *An optimization-centric view on Bayes' rule: reviewing and generalizing variational inference.* JMLR 23(132). Supports: the generalised-Bayes framing (a loss in place of the negative log-likelihood defines a valid belief update).
- Bissiri, P. G., Holmes, C. C., Walker, S. G. (2016). *A general framework for updating belief distributions.* JRSS-B 78(5), 1103–1130. Supports: the generalised posterior `∝ prior × exp(−loss)`.
- West, M. (1981). *Robust sequential approximate Bayesian estimation.* JRSS-B 43(2), 157–166. Supports (via the paper's App. A): the WoLF-IMQ posterior-mean update coincides with the Cauchy-likelihood update of West, i.e. IMQ weighting is an established heavy-tail approximation.
- Ting, J.-A., Theodorou, E., Schaal, S. (2007). *Learning an outlier-robust Kalman filter.* ECML 2007. Alternative considered: a variational Gamma-scale weight per observation (paper App. A.4) — iterative per step, not closed form.
- Agamennoni, G., Nieto, J. I., Nebot, E. M. (2012). *Approximate inference in state-space models with heavy-tailed noise.* IEEE Trans. Signal Processing 60(10). Alternative considered: inverse-Wishart / Student-t variational robust KF (the paper's KF-IW baseline) — several inner iterations per step and 2–5× slower than the KF in the paper's Table 2.
- Huang, Y., Zhang, Y., Li, N., Chambers, J. (2016). *A robust Gaussian approximate filter for nonlinear systems with heavy tailed measurement noises.* IEEE ICASSP 2016. Alternative considered: variational Bayes with a Student-t measurement model (paper App. A.5). The brief cited "Huang et al. 2017, IEEE TAC"; that specific reference could not be verified from the paper's bibliography, so this ICASSP paper (which is in it) is cited instead.
- Wang, H., Li, H., Fang, J., Wang, H. (2018). *Robust Gaussian Kalman filter with outlier detection.* IEEE Signal Processing Letters 25(8). Alternative considered: Bernoulli detect-and-reject VB filter (paper's KF-B baseline).
- Ollivier, Y. (2018). *Online natural gradient as a Kalman filter.* Electronic Journal of Statistics 12(2). Supports: the moment-matched exponential-family EKF that the paper's App. D.1 weights; our Fisher-scoring update is that construction.
- McCullagh, P., Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed., Chapman & Hall. Supports: unit deviance and deviance residuals for Poisson/binomial families, and their better normal calibration than Pearson residuals at small means.
- Eden, U. T., Frank, L. M., Barbieri, R., Solo, V., Brown, E. N. (2004). *Dynamic analysis of neural encoding by point process adaptive filtering.* Neural Computation 16, 971–998. Supports: the Laplace-EKF point-process filter that phase 3 robustifies (already cited in `point_process_kalman.py`).
