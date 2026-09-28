# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

A population-level switching model of spatial maps. At each time bin a shared discrete
state `s_t ∈ {1..K}` selects which of `K` maps generates spikes at the animal's *actual*
position: `y_{n,t} ~ Poisson(exp(Z(p_t) · w_{n, s_t}) dt)`. Scientific targets: spontaneous,
reversible, speed-correlated map switching in medial entorhinal cortex (Low et al. 2021,
*Neuron* — k-means on population vectors identified the map states, and remapping
covaried with running speed) and coexisting CA1 maps of one environment (Sheintuch et al.
2020, *Current Biology*); the theoretical framing of remapping as inference over a hidden
context state is Sanders, Wilson & Gershman 2020, *eLife*. The HMM machinery is textbook
(Rabiner 1989: forward–backward and Viterbi); the model-selection criterion is Schwarz 1978
(BIC); the smoothness penalty is the P-spline difference penalty of Eilers & Marx 1996.

This is distinct from the deferred represented-state plan
(`docs/plans/2026-04-05-ca1-represented-state-switching.md`): there the represented
*position* switches away from the tracked position; here the *map* switches while the
position is the animal's.

## Current codebase integration points

Nothing existing changes behaviour. New module `src/state_space_practice/multi_map_place_field.py`;
new helpers appended to `utils.py` and `simulate_data.py`; registrations in `__init__.py`,
`pyproject.toml`, `CHANGELOG.md`, `README.md`.

- `src/state_space_practice/utils.py:1769-1824` — `hmm_viterbi`: reused unchanged for the
  Viterbi path (phase 1). New `HMMPosterior`, `hmm_filter`, `hmm_forward_backward` are
  inserted after it ([designs A](designs.md#a-exact-hmm-forwardbackward-in-log-space)).
  Phase 1b consumes the recurrent plan's `(T, K, K)` branch of `hmm_viterbi` through an edge-stack adapter; the 2-D path stays
  bit-identical (existing test `tests/test_utils.py:703-712`).
- `src/state_space_practice/utils.py:1722-1759` (`make_discrete_transition_matrix`),
  `:1827-1847` (`zero_preserving_log`), `:1895-1921` (`find_permutation`), `:1857-1892`
  (`compute_state_overlap`), `:1016-1041` (`validate_count_array`), `:1044`
  (`validate_finite_array`), `:1113-1158` (`validate_int`), `:1161-1184`
  (`validate_scalar`), `:1308-1405` (`validate_covariance`), `:1408-1457`
  (`validate_transition_matrix`), `:1460-1499` (`validate_probability_vector`): consumed,
  untouched.
- `src/state_space_practice/switching_point_process.py:1035-1045`
  (`_ridged_newton_direction`), `:1048-1090` (`_descent_step`), `:1093-1105`
  (`_warn_newton_fallbacks`): reused by the weighted GLM M-step
  ([designs C](designs.md#c-weighted-penalised-poisson-glm-m-step)); untouched.
  `update_spike_glm_params` (`:1813-1968`) and `update_spike_glm_params_mixture`
  (`:1971-2031`) are *not* reused — see designs C for why (baseline term, scalar L2 only).
- `src/state_space_practice/switching_kalman.py:2063-2071` (`minimum_state_occupancy`),
  `:2074-2110` (`warn_low_occupancy_states`): reused for the low-occupancy map gate;
  `:2275-2287` (discrete-transition MAP and initial-probability update) is the pattern the
  transition M-step mirrors ([designs E](designs.md#e-transition-and-initial-state-m-step-occupancy-ordering)).
- `src/state_space_practice/contingency_belief.py:200-222` (`get_transition_prior`):
  reused for the Dirichlet pseudo-counts. Phase 1b additionally reuses
  `compute_input_output_transition_matrix` (`:173-197`), `centered_softmax_inverse`
  (`:115-130`) and `_optimize_transition_rows` (`:804-846`).
- `src/state_space_practice/point_process_kalman.py:589-610` (`_safe_expected_count`):
  the one clipped expected-count function used by both E- and M-step. `BlockDiagonalStructure`
  (`:148-200`) / `BlockDiagonalCovariance` (`:216-368`) are only discussed in the drift
  sketch ([designs K](designs.md#k-slow-per-map-drift--sketch-not-scheduled)); untouched.
- `src/state_space_practice/place_field_model.py:79-157` (`build_2d_spline_basis`),
  `:160-190` (`evaluate_basis`): the spline basis the model accepts (via the caller);
  `:652-821` (`_fit_stationary_glm`) is the K = 1 parity reference; `:550-565`
  (`_check_fitted` / `NotFittedError`), `:852-862` (`_max_log_count`), `:1886-1979`
  (`score`), `:2005-2049` (`n_free_params`, `bic`) are the API conventions mirrored. Untouched.
- `src/state_space_practice/graph_place_field.py:58-92` (`GraphBasis`), `:328-376`
  (`graph_design_matrix`): the graph basis the model accepts. `graph_design_matrix` returns a
  `valid` mask whose invalid rows the caller must drop until `obs_mask=` lands
  (`docs/plans/masks-and-multi-sequence/`). Untouched.
- `src/state_space_practice/em_driver.py:58-77` (`run_em`): the EM loop, with default
  rollback policy (exact EM). `src/state_space_practice/sgd_fitting.py:354-373`
  (`SGDFittableMixin` protocol), `:384-397` (`_prepare_sgd_data`), `:506-535` (`fit_sgd`),
  `:719-721` (history/convergence storage): the SGD path.
- `src/state_space_practice/parameter_transforms.py:157` (`UNCONSTRAINED`), `:225-252`
  (`STOCHASTIC_ROW`): SGD parameter transforms.
- `src/state_space_practice/simulate_data.py:138-166` — the lawnmower trajectory block of
  `simulate_2d_moving_place_field` is lifted into a private helper and reused by the new
  `simulate_multi_map_session`; the function's output must stay bit-identical (baseline
  capture task in phase 1).
- `src/state_space_practice/__init__.py:35-53` (`_LAZY_API`), `:55-76` (`__all__`),
  `:78-100` (`TYPE_CHECKING` imports): register `MultiMapPlaceFieldModel`.
  `tests/test_package.py:36-46` then covers the export automatically.
- `pyproject.toml:134-159` (`[tool.mypy] files`): add the new module (alphabetically after
  `models.py` at `:147`). `CHANGELOG.md:8` (`### Added` under Unreleased) and
  `README.md:60-67` (Package layout) get entries.
- Tests reused: `tests/oracles.py:437-442` (`_logsumexp`), `tests/test_oracle_switching_point_process.py:125-126,238-261`
  (path-enumeration pattern), `tests/recovery_helpers.py:28-60,68-86`
  (`state_segmentation_accuracy`, `assert_ll_improves`, `assert_ll_monotonic`),
  `tests/test_invariances.py:82-88` (`_close`) and `:1130-1205` (state-relabelling
  pattern), `tests/test_sbc_ranks.py:60-71` (rank/chi-square helpers as the model for a
  discrete calibration check), `tests/conftest.py:143` (automatic `slow` marking of tests
  that call `.fit(`; `:68` registry for fitting hidden in helpers).

## Scope and dependency policy

### Goals

- `MultiMapPlaceFieldModel(n_maps, dt, ...)`: K static maps, first-order Markov map state,
  exact EM (log-space forward–backward + per-(neuron, map) weighted penalised Poisson-GLM
  refits with a ridge or smoothness penalty), fitted through `run_em`; the same penalised
  objective through `SGDFittableMixin.fit_sgd`.
- Basis-agnostic: the model consumes a per-time design matrix `(n_time, n_basis)` from
  `build_2d_spline_basis` / `evaluate_basis` or `graph_design_matrix`.
- Initialisation: k-means on windowed population rate-map vectors with a fixed seed;
  labels ordered by occupancy after the fit; `initial_responsibilities=` override.
- Outputs: smoothed and filtered map posteriors, pairwise posteriors, Viterbi path,
  per-map rate maps, per-transition map-switch probability, behaviour-alignment helpers
  (lagged correlation with a covariate, circular-shift null, posterior-weighted covariate
  mean per map).
- Model selection over K: `bic()` and held-out `score()`, plus `fit_over_n_maps` /
  `select_n_maps` with a parsimony tolerance.
- K = 1 reduces exactly to a penalised Poisson-GLM place-field fit (parity with
  `PlaceFieldModel._fit_stationary_glm`).
- Phase 1b (gated): speed-gated transitions through the recurrent-transitions interface, so
  Low et al.'s speed correlation is a fitted parameter rather than a post-hoc correlation.

### Non-Goals

- No within-map drift of the weights (the phase-2 idea). See "Deliberately not in this
  plan" below and [designs K](designs.md#k-slow-per-map-drift--sketch-not-scheduled).
- No represented-position switching (`docs/plans/2026-04-05-ca1-represented-state-switching.md`).
- No semi-Markov / explicit dwell-time distributions; no hierarchical (session-level) map
  priors; no multi-session fitting or observation masks in this plan (consumed from
  `docs/plans/masks-and-multi-sequence/` later, additively: `fit(..., obs_mask=)`).
- No change to `PlaceFieldModel`, `switching_point_process.py` solvers, or
  `contingency_belief.py`; helpers are imported, not modified.
- No new plotting helpers beyond what the smoke script does inline.
- No automatic K selection inside `fit`.

### Dependency policy

No new runtime dependencies: `scikit-learn` (k-means) is already required
(`pyproject.toml` `dependencies`, used lazily at `oscillator_models.py:687`), and the
graph basis stays behind the existing `spatial` extra. Sibling plans this one touches
(link, don't restate):

- `docs/plans/recurrent-switching-transitions/` — phase 1b consumes its phase-1
  `transition_covariates=` / `transition_weights=` interface. Phase 1 designs the
  transition input as "a `(K, K)` matrix or a `(n_time - 1, K, K)` stack"
  ([contract](shared-contracts.md#hmm-forward-backward)) so plugging it in is additive.
  **Gate:** phase 1b does not start until that plan's phase 1 has merged.
- `docs/plans/masks-and-multi-sequence/` — `obs_mask=` and multi-session fitting. Not
  consumed in phase 1 (callers drop invalid graph-basis rows themselves); it is the
  prerequisite for the drift follow-on.
- `docs/plans/identifiability-diagnostics/` — `.identifiability_report()` is the intended
  gate on K-map fits (near-identical maps, low-occupancy maps). Phase 1 exposes what it
  needs (per-map occupancy, per-map rate maps, pairwise posteriors) and does not implement
  the report.

### Deliberately not in this plan

- **Slow per-map drift (random-walk weights per map with a shared discrete state).** Not
  designable yet: the GPB collapse correlates a neuron's weights across maps
  (block-diagonal only conditional on the path), and the exact-vs-approximate choice has no
  evidence; the state-conditional storage is 10²–10³× the already-heavy `PlaceFieldModel`
  covariance store ([designs K](designs.md#k-slow-per-map-drift--sketch-not-scheduled) has
  the numbers). **Trigger to plan it:** (1) `docs/plans/masks-and-multi-sequence/` has
  merged (segment-wise processing), (2) phase-1 fits on real data show within-map drift
  (rate-map correlation between the first and second half of a map's occupancy below ~0.8
  for a majority of neurons), and (3) a `T <= 5` path-enumeration study (extending
  `tests/test_oracle_switching_point_process.py`) has picked between exact GPB1, block-GPB,
  and frozen-inactive-map inference.
- Effective-degrees-of-freedom BIC (`tr(H (H + P)^{-1})`) — revisit if nominal-count BIC
  under-selects K on real data relative to held-out likelihood.

## Metrics

- **Exactness:** `hmm_forward_backward` and `hmm_filter` match path enumeration to
  `rtol 1e-10` (smoothed, pairwise, filtered, log-likelihood), for constant and stacked
  transitions; pairwise marginalisation identities hold to `1e-12`.
- **K = 1 parity:** weights within `rtol 1e-6` of `PlaceFieldModel._fit_stationary_glm`
  (same penalty), within `1e-4` of an independent `scipy.optimize.minimize` solution of the
  same objective; log-likelihood equals `sum(poisson.logpmf)` to `1e-12`.
- **EM:** `log_likelihood_history_` non-decreasing (tolerance `1e-8` relative);
  final unpenalised `log_likelihood_` above the one-iteration fit.
- **Recovery (simulated two-map session, full remap):** Viterbi accuracy ≥ 0.95 after
  optimal relabelling; per-map rate-map correlation with truth ≥ 0.9; segmentation
  accuracy non-increasing as `map_separation` shrinks (1.0 → 0.5 → 0.2), with the last at
  least 0.1 below the first.
- **Model selection:** BIC minimum and held-out-likelihood maximum both at the true K = 2
  over `{1, 2, 3}`, with the guard `LL(K=2) − LL(K=1) > 2 × (per-map parameter count) × log(n_time)`.
- **Invariances:** map relabelling, neuron relabelling and an invertible basis change
  leave the log-likelihood unchanged to `1e-12` and permute/transform posteriors to
  `1e-10`; two fits from column-permuted initial responsibilities end identical to `1e-8`
  after occupancy ordering.
- **Calibration:** at true parameters, the smoothed map posterior is calibrated on
  simulated sessions (reliability error < 0.05 per decile with ≥ 200 samples) and a
  sharpened posterior fails the same check (power guard).
- **SGD:** on a small problem `fit_sgd` reaches the EM objective within `1e-3` relative.
- **Runtime:** fast suite additions ≤ 10 s total; every fitting test marked slow; the
  simulated two-map recovery fit (8 neurons, 15 000 bins, 49 basis functions) completes in
  under 60 s on CPU (measure and record in the PR).
- **Real data (script, not test):** on the J16 session the switch probability–speed
  correlation is reported with a circular-shift null and a BIC table over K.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Label switching across fits / seeds makes maps incomparable | Deterministic occupancy ordering after every fit; `find_permutation` in tests; relabelling-invariance test pins the ordering is applied to every fitted quantity. |
| A map collapses to near-zero occupancy during EM (degenerate M-step) | Occupancy gate keeps that map's previous weights and logs a warning (`warn_low_occupancy_states`); transition row keeps its previous value when its expected mass is zero. |
| Poor k-means initialisation on real data (windows too short, sparse spiking) | `init_window_bins` and `seed` are parameters; `initial_responsibilities=` lets the user seed from anything (e.g. lap labels); k-means quality test on planted maps. |
| EM and SGD optimise different objectives, so `fit` and `fit_sgd` disagree | One `_log_prior` term shared by the EM objective and the SGD loss (pattern of `switching_point_process._sgd_loss_fn`). |
| Penalty-clipped expected counts (`_safe_expected_count`) make Newton stall for saturated neurons | Same clip in E- and M-step keeps the objective consistent; line-searched steps never increase it; a saturation warning mirrors `PlaceFieldModel._warn_if_rate_saturated` (`place_field_model.py:920-981`). |
| Materialising `(n_time, n_neurons, n_maps)` expected counts blows memory | `lax.map(..., batch_size=)` over bins; only `(n_time, n_maps)` is stored. |
| BIC with a penalised likelihood mis-counts parameters | Nominal count documented as conservative; `select_n_maps` parsimony tolerance; held-out `score()` as the second criterion. |
| Behaviour-correlation false positives from autocorrelated switch probability | Circular-shift null preserves autocorrelation; 1b tests the relationship as a parameter. |
| Lifting the trajectory helper changes `simulate_2d_moving_place_field` | Baseline capture before, byte-equality after, and a pinned test on seeded values. |

## Rollout Strategy

All at once per phase, purely additive: a new module, new `utils` functions, a new
simulator, a lazy export. No existing public behaviour changes, so no deprecation window.
Phase 1b is gated on the sibling plan and lands as its own PR; `fit(...,
transition_covariates=None)` defaults preserve phase-1 behaviour exactly.

## Open Questions

1. **What `fit` returns / what `log_likelihood_history_` means.** Best answer (decided):
   the *penalised* objective EM maximises (marginal LL + log prior), because that is the
   quantity `run_em`'s monotonicity and rollback checks are valid for and it is what the
   SGD mixin already stores under that name (`sgd_fitting.py:719`); `log_likelihood_` holds
   the unpenalised marginal LL. Documented in the class docstring and the
   [model contract](shared-contracts.md#model-attributes).
2. **BIC parameter count with a penalty.** Best answer: nominal count (documented as
   conservative). Effective-df variant deferred — see "Deliberately not in this plan".
3. **Initial map distribution for `score()` on a held-out segment.** Best answer: the
   fitted `init_map_prob_` (mirrors `PlaceFieldModel.score`). Alternative: the stationary
   distribution of `transition_matrix_`. Revisit if held-out scores are visibly sensitive on
   the real-data smoke script.
4. **k-means features.** Best answer: ridge least-squares projection of counts onto the
   basis per window, z-scored (basis-agnostic). Low et al. clustered position-binned rate
   maps; if the projection initialises poorly on real data, add an occupancy-normalised
   binned-rate-map feature as an option.
5. **Newton steps per M-step.** Best answer: `max_newton_iter=5` (generalised EM keeps the
   objective monotone; the K = 1 parity test converges by running more EM iterations).
   Tune from the recorded runtime of the recovery fit.
6. **Drift follow-on.** Deferred — see "Deliberately not in this plan" and
   [designs K](designs.md#k-slow-per-map-drift--sketch-not-scheduled).
7. **Phase 1b parametrisation.** Assumed to be the centered-softmax (last map as reference)
   of `contingency_belief.compute_input_output_transition_matrix`; if the merged
   recurrent-transitions interface differs, phase 1b adopts that plan's parametrisation and
   only the M-step reshaping changes.

## Estimated Effort

- `multi_map_place_field.py`: ~700 LOC (class ~350, GLM M-step ~120, k-means ~60,
  helpers ~120, model selection ~50).
- `utils.py` additions: ~130 LOC. `simulate_data.py` additions: ~140 LOC (helper lift +
  simulator + Gaussian rate helper).
- Tests: ~900 LOC across `test_multi_map_place_field.py` (new), `test_utils.py`,
  `test_invariances.py`, `test_simulate_data.py`.
- Docs/registration: ~40 LOC (`__init__`, `pyproject`, CHANGELOG, README). Smoke script
  ~150 LOC.
- Phase 1b: ~200 LOC source, ~250 LOC tests.
