# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Scientific background and literature

Hippocampal place cells fire at progressively earlier theta phases as the animal crosses a field (phase precession; Skaggs, McNaughton, Wilson & Barnes 1996, *Hippocampus* 6:149–172). At the population level this means the position represented by the ensemble sweeps from behind the animal to ahead of it within each theta cycle — theta sequences (Foster & Wilson 2007, *Hippocampus* 17:1093–1099). Each citation below supports one claim this plan relies on:

- **Skaggs et al. 1996** — phase precession implies within-cycle sweeps of the represented position; the sweep is phase-locked to theta, which is why the model parameterises it by theta phase.
- **Foster & Wilson 2007** — theta sequences are a population phenomenon during running; supports restricting estimation to running bouts and modelling the population jointly.
- **Wikenheiser & Redish 2015** (*Nature Neuroscience* 18:289–294) — the look-ahead distance of theta sequences scales with the distance to the goal; supports a *time-varying* amplitude `a_t` (a random walk) rather than one session constant.
- **Kay et al. 2020** (*Cell* 180:552–567) — represented position can alternate between options on sub-second (theta-cycle) timescales; supports allowing cycle-to-cycle variation (phase 2a's per-cycle offset) rather than assuming every cycle sweeps identically.
- **Joshi et al. 2023** (*Nature* 617:125–131; the user is a co-author) — theta sweeps are synchronised to stepping; supports the need for a per-bin (not per-session) estimate that can be related to concurrent behaviour.
- **Ujfalussy & Orbán 2022** (*eLife* 11:e79294) — cycle-to-cycle variability of theta sequences is consistent with *sampling* of trajectories; the quantity of interest is the variance of a per-cycle random offset, which is exactly phase 2b's `sigma_c^2 = 0` vs free comparison.
- **Denovellis et al. 2021** (*eLife* 10:e64505) — state-space decoding of represented position from CA1 spikes at real-world speeds; the decoding lineage this library's `position_decoder` and this estimator follow.
- **Eden, Frank, Barbieri, Solo & Brown 2004** (*Neural Computation* 16:971–998) — the point-process adaptive filter (Laplace-EKF) this plan builds on; already cited throughout `point_process_kalman.py`.
- **Self & Liang 1987** (*JASA* 82:605–610) — the likelihood-ratio statistic for a variance parameter tested at its boundary (`sigma_c^2 = 0`) is asymptotically a 50:50 mixture of a point mass at 0 and `chi^2_1`; used by phase 2b's test.

**Novelty claim, stated honestly:** we are not aware of a published state-space estimator of the theta-sweep amplitude that returns per-bin uncertainty, and we have *not* verified that claim with a literature search. Prior work estimates sweep extent from decoded position within cycles (e.g. Foster & Wilson 2007; Wikenheiser & Redish 2015) without a latent-variable model of the amplitude. Do not state novelty in user-facing docs beyond "this library's approach".

**Model.** Per bin `t` the inputs are linear position `p_t` (cm), theta phase `phi_t` (radians, from the LFP; this library does not extract phase), travel direction `d_t = sign(velocity_t)` and spike counts `y_t` over `n_neurons` neurons; each neuron has a fixed 1-D rate map `lambda_n(r)` fit beforehand on running periods. The represented position is `r_t = p_t + d_t * f(phi_t)` with `f(phi) = sum_{h=1..H} [c_h cos(h phi) + s_h sin(h phi)]` (`H = 1` by default, so `f(phi) = a cos(phi - psi)` with amplitude `a = sqrt(c_1^2 + s_1^2)` and phase offset `psi = atan2(s_1, c_1)`), and `y_{n,t} ~ Poisson(lambda_n(r_t) dt)`. The latent state is the coefficient vector, following a random walk with small process noise; the observation is the nonlinearity. Positive `a cos(phi_t - psi)` means the representation is *ahead* in the direction of travel. The derivation and Jacobian are in [designs.md](designs.md#observation-model).

**Data requirement.** The estimator needs an LFP theta phase and a linearised position on a single track segment. The J16 session used elsewhere in this repository has no LFP (`docs/real_ca1_theta_analysis_plan.md:13`), so **synthetic validation is the primary gate** for every phase and the real-data smoke script (phase 1) is a script, not a test, that runs only when a dataset with LFP phase is available.

## Current codebase integration points

Everything below is *read or imported*; no existing source file is modified except the additive registrations listed, and the private helpers imported from `point_process_kalman.py` are used as-is (the same precedent as `position_decoder.py:34-38`).

- `src/state_space_practice/point_process_kalman.py:1467-1484` — `stochastic_point_process_filter` signature; `:1551-1554` the `log_conditional_intensity(Z_k, x_k) -> (n_neurons,)` contract; `:1530-1540` the design matrix may have any shape `(n_time, ...)` the callable expects; `:1657-1662` a non-default callable always takes the dense path (block dispatch requires the default linear intensity). Phase 1 calls this through the smoother; untouched.
- `src/state_space_practice/point_process_kalman.py:2392-2410` — `stochastic_point_process_smoother` signature; `:2599-2602` returns `(mean, cov, cross_cov, log_likelihood[, filtered_mean, filtered_cov])`. Phase 1's engine; untouched.
- `src/state_space_practice/point_process_kalman.py:1737-1744` — the inner filter is jitted with `log_conditional_intensity` as a *static* argument (hashed by identity; `functools.partial` objects do not compare equal), so the model must build its log-intensity callable once and reuse it. `:1794-1798` the Jacobian is `jax.jacfwd` of that callable with respect to `x` — autodiff through the interpolation is what the filter already does.
- `src/state_space_practice/point_process_kalman.py:927-940` — `_point_process_laplace_update` (Fisher-scoring update; `:955-976` explains why the expected Hessian is used for nonlinear intensities such as rate maps). Phase 2a's forward scan calls it directly, exactly as `position_decoder.py:1133-1144` does; untouched.
- `src/state_space_practice/point_process_kalman.py:589-612` — `_safe_expected_count(log_rate, dt, min_log_count=-20, max_log_count=20)`; imported for the null log-likelihood so null and sweep models share the same rate ceiling; untouched.
- `src/state_space_practice/point_process_kalman.py:882-924` — `_warn_line_search_failures` (a `jax.debug.callback` that *logs*; it never raises a warning, so it is safe under `filterwarnings = error`); phase 2a's scan calls it as `position_decoder.py:1165-1170` does; untouched.
- `src/state_space_practice/position_decoder.py:802-845` — `_bilinear_log_rate`: the 2-D clamped-index interpolation whose 1-D analogue phase 1 writes (same clamping so the Jacobian is zero off the grid); `:848-867` its `jax.jacfwd` Jacobian; `:969-1178` `_run_filter_scan`, the template for phase 2a's time-varying forward scan (minus track penalty and inflation); `:1596-1602` the smoother's use of stored predictions. All untouched.
- `src/state_space_practice/position_decoder.py:198-237` and `:393-624` — `PlaceFieldRateMaps` and `from_spike_position_data` (2-D only; occupancy-normalised, Gaussian-smoothed). Phase 1's `LinearRateMaps` is the 1-D counterpart; `PlaceFieldRateMaps` is untouched.
- `src/state_space_practice/place_field_model.py:79-157` — `build_2d_spline_basis` is 2-D (tensor product) only; there is no 1-D basis helper in the library, which is why phase 1 uses linear interpolation of gridded log-rate maps rather than a spline basis. Untouched.
- `src/state_space_practice/kalman.py:1072-1216` — `parallel_kalman_smoother` accepts `(T-1, D, D)` transition and process-covariance stacks (`:1090-1096`; `transition_matrix[t]` maps `t -> t+1`) and is tested against a sequential time-varying reference in `tests/test_kalman.py:1885` (`test_time_varying_transition_matches_sequential_reference`). Phase 2a's backward pass; untouched.
- `src/state_space_practice/kalman.py:1251-1278`, `:1320-1361`, `:1399-1445` — `InitialStatePrior`, `smooth_initial_state_with_cross_cov`, `process_cov_residual_form`: the exact-EM building blocks phase 2b's M-step reuses, following `place_field_model.py:1076-1127`. Untouched.
- `src/state_space_practice/em_driver.py:58` — `run_em(e_step, m_step, snapshot, restore, *, max_iter, tol, ...)`; phase 2b's EM loop, configured as `point_process_kalman.py:3188-3232` configures it. Untouched.
- `src/state_space_practice/preprocessing.py:248` — `identify_behavioral_bouts(speed, speed_threshold, min_duration, above_threshold=True) -> list[(start, end_exclusive)]`; phase 1 uses it on `|velocity|`. `:334` `interpolate_to_new_times` and `:56` `bin_spike_times` are used by the smoke script. Untouched.
- `src/state_space_practice/circular_stats.py:410` — `wrap_to_pi`; `:388` `angular_distance`; `:45` `circular_mean` (NumPy; used at the model boundary and in tests). Untouched.
- `src/state_space_practice/utils.py:1016` `validate_count_array`, `:1161` `validate_scalar`, `:1187` `_validate_filter_numerics`, `:26` `symmetrize` — input validation and symmetrisation; untouched.
- `src/state_space_practice/__init__.py:35-51` (`_LAZY_API`), `:55-75` (`__all__`), `:78-101` (`TYPE_CHECKING` imports) — **additive registration** of `ThetaSweepModel` (phase 1); `tests/test_package.py:35-45` enforces that the three stay in sync.
- `pyproject.toml:134-159` — `[tool.mypy] files`: **additive** entries for the new module and its simulation module (phase 1).
- `README.md:60-67` ("Package layout") and `CHANGELOG.md:6-8` (`## [Unreleased]` / `### Added`), `:84` (`### Testing`) — **additive** user-facing doc entries (every phase).
- `src/state_space_practice/tests/test_oracle_point_process.py:81-88` (`_Problem`), `:124-146` (`_run_laplace`, hard-codes `_affine_log_rate` at `:134`), `:191-240` (`_grid_posterior`, hard-codes the affine log-rate at `:210-211`) — phase 1 **generalises** these two helpers to accept a problem-specific log-rate (default: the existing affine form, so existing tests are unchanged) and adds a scalar-amplitude oracle.
- `src/state_space_practice/tests/test_calibration_point_process.py:42-82` (`_calibration_stats`, `_describe`, `_assert_calibrated`), `:190-229` (`_decoder_z`, the replicate-loop pattern), `:280-308` (pinned mild overconfidence of a piecewise-linear log-rate) — phase 1 and 2a **add** a `TestThetaSweepSmootherCalibration` class next to these.
- `src/state_space_practice/tests/test_sbc_ranks.py:54-85` (`N_DRAWS`, `CRITICAL`, `_ranks`, `_chi_square`, `_select`, `_describe`) — phase 1 **adds** a rank test.
- `src/state_space_practice/tests/recovery_helpers.py:68` `assert_ll_improves`, `:77` `assert_ll_monotonic`, `:395` `standardized_errors` — reused by phase 2b's EM tests.
- `src/state_space_practice/tests/conftest.py:47-149` — tests calling `.fit(` / `run_em(` are auto-marked slow; fast tests in this plan therefore use the functional API (`theta_sweep_smoother`, `build_sweep_design`, ...) and never call `.fit(`.

## Scope and dependency policy

### Goals

- A public `ThetaSweepModel` (lazy top-level export) that estimates per-bin sweep coefficients, amplitude and phase offset with credible intervals on running bouts of a single linear segment, plus a log-likelihood score against the no-sweep null and a phase-shuffle reference for it.
- Fixed-offset scalar-amplitude mode (`phase_offset=` given) so the latent is 1-D — needed for the grid-quadrature oracle and useful when the population's sweep phase is known.
- A 1-D `LinearRateMaps` container with an occupancy-normalised builder that accepts a time-bin mask (the hook for phase-restricted rate maps that de-bias the circularity described under Risks).
- A simulator of a linear-track session with theta phase and a known (constant / step / drifting) amplitude, generating spikes from exactly the likelihood the model evaluates.
- Verification that reuses the repository's oracle, calibration and SBC infrastructure: Laplace vs exact grid posterior for the scalar amplitude; z-calibration and rank uniformity of the coefficient posterior at true parameters; recovery RMSE and interval coverage; null and phase-shuffle controls.
- (Phase 2a) A per-theta-cycle random offset state, reset at cycle boundaries detected from the phase, so cycle-to-cycle sampling variance is a model parameter.
- (Phase 2b) Exact EM for the process variance and the cycle-offset variance, and a likelihood-ratio test of `sigma_c^2 = 0` vs free.

### Non-Goals

- **Multi-arm tracks.** V1 handles one linear segment: the represented position `p_t + b_t . x` is interpolated on a single 1-D grid, so an arm discontinuity in linearised position is meaningless. Trigger to revisit: a dataset whose linearised position jumps between arms — then the interpolation must move to graph distance (see `graph_place_field.py`), which is a separate design.
- **Extracting theta phase from LFP.** `phi_t` is an input array. The smoke script documents how to obtain it (band-pass 6–12 Hz, Hilbert angle) but the library does not implement it.
- **Direction-specific sweep coefficients.** One coefficient vector serves both travel directions (the basis is multiplied by `d_t`). Trigger: the real-data smoke shows systematically different amplitudes when bouts of each direction are fit separately; then add a per-direction coefficient block.
- **Predict-only handling of non-running bins.** Bouts are processed independently from the same prior (see Rollout). Trigger: `docs/plans/masks-and-multi-sequence/` lands; then replace the bout loop with one masked pass.
- **Automated alternation between rate-map fitting and sweep fitting.** Phase 1 ships the phase-restricted mask and documents one manual iteration; automating it is a follow-up once the real-data smoke shows how many iterations it takes to stabilise.
- **2-D rate maps or the KDE evaluation path of `PlaceFieldRateMaps`.** Fixed gridded 1-D maps only.
- **SGD fitting (`SGDFittableMixin`).** Hyperparameters are learned by closed-form EM (phase 2b); there are only two scalars.

### Dependency policy

No new third-party dependencies: NumPy, SciPy (`scipy.ndimage.gaussian_filter1d`, `scipy.stats.chi2`, `scipy.stats.rice` in tests) and JAX are already required.

Ten other plans are being written in parallel; only the following interact with this one, and this plan does not restate their content:

- `docs/plans/identifiability-diagnostics/` — its `.identifiability_report()` is a **gate** for this estimator: the report must show the amplitude direction identified separately from the phase-offset direction and from a global rate-map-scale nuisance. Until it lands, phase 1 ships the module's own fallback diagnostics (posterior `corr(c_1, s_1)` and a rate-map-broadening sensitivity test, see [phase 1](phase-1-sweep-amplitude.md)); the smoke script calls the report when the attribute exists.
- `docs/plans/iterated-parallel-laplace-smoother/` — optional. If it adds `n_iterations=` to `stochastic_point_process_smoother`, `ThetaSweepModel` should pass it through; the rate-map observation is nonlinear and the calibration tests (which pin mild overconfidence) are the place to measure the benefit. No task here depends on it.
- `docs/plans/wolf-robust-updates/` — optional robust weight for outlier bins (spikes far from any field). Trigger: the smoke script reports more than 10% line-search failures or large innovations; until then the `rate_floor_hz` parameter bounds the penalty of an out-of-field spike.
- `docs/plans/masks-and-multi-sequence/` — absent at planning time; see Non-Goals.

## Metrics

All on the simulator in [designs.md](designs.md#simulator) unless noted; seeds fixed.

- **Recovery (phase 1):** constant `a = 15 cm`, `psi = 0.5`, 40 neurons, 60 s at `dt = 4 ms`, three seeds: RMSE of the posterior-median amplitude over running bins after a 2 s burn-in per bout `< 2 cm`; circular error of the phase offset `< 0.3 rad`. Step change `8 -> 20 cm`: last 5 s of each half within `3 cm` of its truth. Linear drift `5 -> 25 cm`: Pearson `r > 0.9`, RMSE `< 3 cm`.
- **Calibration (phase 1, 2a):** pooled z-scores of the smoothed coefficients at true parameters over `>= 300` replicates: mean within `4 SE` of 0; `E[z^2]` pinned at the observed value `+-4 SE` and within `(0.95, 1.2)`; 90% coverage `> 0.87` (the same mildness bounds `test_calibration_point_process.py:280-308` uses for the decoder's piecewise-linear log-rate). Rank histogram (SBC, 20 bins, `alpha = 1e-3`) not rejected, and rejected for a posterior 25% too narrow.
- **Oracle (phase 1):** scalar-amplitude Laplace smoother vs exact grid posterior, `T = 6`, near-Gaussian regime: mean error `< 0.4` posterior sd, variance ratio error `< 0.25`, log-likelihood error `< 0.3` nats, and a non-vacuous gap (`> 1e-3`).
- **Null (phase 1):** `a = 0`: the origin lies in the 95% credible region of `(c_1, s_1)` in `>= 90%` of running bins; `score - median(shuffle scores) < 5` nats per 60 s.
- **Shuffle control (phase 1):** independent rotations per original theta cycle reduce the median surrogate score on the default coherent-sweep fixture by at least 5 nats. A constant phase rotation preserves the free-offset fit and is tested as a symmetry. Surrogate comparisons depend on process variance; no universal amplitude-collapse threshold or exact permutation-test claim. Validate no-sweep calibration separately.
- **Identifiability fallback (phase 1):** broadening the rate maps by an extra 4 cm moves the posterior-median amplitude by `< 1` posterior sd in `>= 90%` of bins; `|corr(c_1, s_1)| < 0.5` in `>= 95%` of bins.
- **Phase 2a:** constant-dynamics path re-routed through the new scan matches `stochastic_point_process_smoother` to `rtol 1e-7` (means, covariances, log-likelihood); smoothed cycle-offset means correlate with the simulated per-cycle offsets at `r > 0.6` (`sigma_c = 6 cm`); the offset posterior is constant within a cycle.
- **Phase 2b:** EM log-likelihood non-decreasing (`assert_ll_monotonic`, tol `1e-3`); `sigma_c` recovered within `[0.67, 1.33] x` truth at `sigma_c = 6 cm`; the LR test rejects at `p < 0.01` there and does not reject at `alpha = 0.05` in `>= 5/6` seeds at `sigma_c = 0`; `q` recovered within a factor 2 (weakly identified by design).
- **Runtime:** fast-suite additions `< 5 s` in total; slow additions `< 5 min` in total on a laptop CPU.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Per-bin Fisher information is rank-1 (`b_t b_t'`): the `(c_1, s_1)` split is identified only by pooling bins across phases within the random walk's memory. Too large a process variance elongates the posterior along the direction orthogonal to `b_t` and the amplitude is over-estimated. | Small default process variance (`0.5 cm^2/s`), a documented log-likelihood profile for choosing it (phase 1) and exact EM for it (phase 2b); `corr(c_1, s_1)` reported per bin; the identifiability gate. |
| Piecewise-linear log-rate (linear interpolation): the Jacobian is constant within a grid cell and jumps at cell edges, so the Laplace curvature misses the kinks — the decoder shows ~9% overconfidence for the same reason (`test_calibration_point_process.py:280-308`). | 1 cm grid and Gaussian smoothing of the maps (`smoothing_sigma >= 2` grid steps) in `LinearRateMaps.from_spike_position_data`; calibration tests pin the overconfidence and bound it; the iterated smoother plan is the principled fix. |
| The population log-likelihood in `r` is not log-concave (Gaussian fields), so with few active neurons the posterior over the amplitude can be multimodal; Laplace picks one mode. | Informative initial prior (`init_std = 20 cm`), small process noise, many neurons; the oracle test uses densely tiled fields; line-search failures above 10% are logged by the filter. |
| Rate-map circularity: maps fit at `p_t` while sweeps are present are broadened by the sweep, which flattens the log-rate gradient and biases the amplitude toward the prior (0). | `bin_mask` in the rate-map builder plus `phase_restricted_bins` (bins where `cos(phi - psi) ~ 0`); one documented refit iteration; the broadening sensitivity test. |
| The amplitude `a >= 0`, so its credible interval can never contain 0: "the CI covers 0 under the null" is not a meaningful check. | Report `origin_in_credible_region` (Mahalanobis test on the coefficient vector) and the score against the null; amplitude-interval coverage is asserted only where the truth is `>= 3` posterior sd from 0. |
| Bouts have different lengths, and the jitted filter recompiles per distinct `n_time` (as the position decoder does). Sessions with 50–100 bouts pay 50–100 compiles. | Documented; `min_bout_duration` drops short bouts; the masks plan would allow padding to a few bucket lengths. |
| Noisy or mis-referenced LFP phase. | The offset `psi` absorbs the phase reference; phase noise attenuates the estimate (documented). |
| The Laplace marginal log-likelihood is approximate, so the LR test's asymptotic null distribution is approximate too. | Phase 2b's test uses the boundary mixture (Self & Liang 1987) and the smoke script provides a plug-in parametric bootstrap for a data-driven reference. |
| An out-of-field spike is penalised by the rate floor, which can dominate the update. | `rate_floor_hz` is a documented parameter (default 0.01 Hz); the Wolf robust-update plan is the follow-up. |

## Rollout Strategy

All at once per phase, additive: a new module plus additive registrations. Nothing changes for users who do not import `theta_sweep`. No deprecation period is needed because no public surface is modified.

Phase 1 processes each running bout independently from the same prior (`x_0 ~ N(0, init_std^2 I)`), because the library filters have no predict-only mask and `docs/plans/masks-and-multi-sequence/` did not exist at planning time. Consequence: the amplitude estimate restarts at every bout and the first ~2 s of each bout are prior-dominated; results carry `NaN` outside bouts. This is a documented behaviour of V1, not a bug.

Phase 2a introduces a forward scan with per-bin `(A_t, Q_t)` (required for the cycle-boundary reset) and re-routes the constant-dynamics path through it, deleting the direct `stochastic_point_process_smoother` call in `theta_sweep_smoother`, so the module has one engine. The equivalence test against the library smoother is kept permanently as the regression pin. Phase 2b changes no engine code.

## Open Questions

1. **Default process variance.** Best answer: `0.5 cm^2/s` (per-cycle sd `~0.25 cm`, so the posterior pools ~5 s of data), chosen by the information-per-cycle estimate in [designs.md](designs.md#observation-model); phase 2b's EM makes the default matter less. Revisit after the real-data smoke.
2. **Carrying the posterior across bouts** (with inflated covariance) instead of restarting from the prior. Deferred — see Non-Goals / `docs/plans/masks-and-multi-sequence/`.
3. **Direction-specific coefficients.** Deferred with a trigger (Non-Goals).
4. **Backward pass for the time-varying dynamics: `parallel_kalman_smoother` or a sequential stacked scan?** Best answer: `parallel_kalman_smoother` (public, tested with time-varying stacks). Revisit if the phase 2a equivalence test cannot meet `rtol 1e-7` on CPU.
5. **Amplitude summary for `H > 1`.** Best answer: report the first-harmonic amplitude and offset as *the* amplitude and offset (higher harmonics describe the shape of the sweep); the full coefficient posterior is always returned.
6. **Plug-in parametric bootstrap for the LR test** (simulate from the smoothed coefficient path under the null). Kept in the smoke script, not the library, because the plug-in reference is approximate; promote if it proves necessary on real data.

## Estimated Effort

- Phase 1: `theta_sweep.py` ~900 LOC (container, design, log-intensity, summaries, null/score, bouts, shuffle, model class), `simulate/simulate_theta_sweep.py` ~200 LOC, `tests/test_theta_sweep.py` ~600 LOC, additions to three existing test files ~250 LOC, smoke script ~150 LOC, docs ~40 lines.
- Phase 2a: ~300 LOC library (cycle index, stacks, forward scan, smoother routing, result fields), ~300 LOC tests.
- Phase 2b: ~250 LOC library (M-steps, EM hooks, LR test), ~300 LOC tests, ~30 lines script/docs.
