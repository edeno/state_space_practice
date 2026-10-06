# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

All paths are under `src/state_space_practice/`. "Touched" means a keyword-only
option or a new branch is added; the existing behaviour with the option at
`None` is preserved (see [Rollout Strategy](#rollout-strategy)).

**Shared primitive (new module, phase 1a).**

- `discrete_transitions.py` (new) — home of the transition-logit machinery
  described in [shared-contracts.md](shared-contracts.md#discrete_transitions-module-api).
  The three centered-softmax helpers move here verbatim from
  `contingency_belief.py:84-130` (`centered_log_softmax`, `centered_softmax`,
  `centered_softmax_inverse`); `contingency_belief.py` keeps importing them so
  `from state_space_practice.contingency_belief import centered_softmax`
  (used by `tests/test_likelihood_identities.py:43`, `tests/test_invariances.py:36`,
  `tests/test_oracle_choice.py:31`, `tests/test_contingency_belief.py:12-21`)
  keeps working. `contingency_belief.py:225-278`
  (`dirichlet_neg_log_likelihood`) and `contingency_belief.py:803-846`
  (`_optimize_transition_rows`, per-row BFGS, jitted) move too, the optimizer
  generalised to a per-row design matrix; `contingency_belief._m_step`
  (`contingency_belief.py:1278-1320`) calls the lifted optimizer.
  `compute_transition_matrix_from_design` (`contingency_belief.py:149-170`) and
  `compute_input_output_transition_matrix` (`contingency_belief.py:173-197`)
  stay where they are, untouched (the latter uses a feature-axis-last layout the
  new contract does not adopt; see Open Question 4).

**Gaussian switching family (phase 1a, 2a).**

- `switching_kalman.py:794-812` — `switching_kalman_filter` signature: gains
  keyword-only `transition_covariates=`, `transition_weights=` (2a:
  `transition_state_weights=`). Its scan body `_step` (`switching_kalman.py:871-988`)
  takes the per-step transition matrix from the scan input instead of the
  closed-over `discrete_transition_matrix` at `switching_kalman.py:947-957`; the scan
  call at `switching_kalman.py:1032-1042` gains a second `xs` leaf. Everything else
  in the body, `_first_timestep_kalman_update` (`:686-790`), the support threading
  (`:1021`) and the outputs (`:1068-1076`) are untouched.
- `switching_kalman.py:454-597` — `_update_discrete_state_probabilities`: untouched.
  It already takes the transition matrix as an argument, so a per-step matrix
  flows through unchanged (the support update at `:535-537` becomes all-True under
  a softmax matrix, which is correct: softmax rows have no structural zeros).
- `switching_kalman.py:1079-1230` — `switching_kalman_viterbi`: the forward scan
  (`:1140-1194`) mirrors the filter change; the backward pass reads `log_trans`
  once at `:1200` and uses it at `:1204`; it becomes a per-step slice of the
  log-stack.
- `switching_kalman.py:1386-1683` — `switching_kalman_smoother` (GPB1): gains the
  same keywords; the backward `_step` (`:1436-1618`) receives the transition
  matrix for the `S_t -> S_{t+1}` step from the scan input instead of the
  closed-over argument at `:1538-1542`; the scan call at `:1639-1648` gains an
  `xs` leaf. `_update_smoother_discrete_probabilities` (`:1317-1383`) is untouched.
- `switching_kalman.py:1686-2039` — `switching_kalman_smoother_gpb2`: untouched.
  It takes no discrete transition matrix (it works from the pair-filter
  probabilities).
- `switching_kalman.py:2154-2300` — `_switching_kalman_m_step_inner`: untouched.
  Its count-based discrete update (`:2275-2281`) is still returned by
  `switching_kalman_maximization_step` (`:2303-2597`); callers on the covariate
  path ignore that output and install the logit M-step's result instead.
- `switching_kalman.py:2063-2110` — `minimum_state_occupancy`,
  `warn_low_occupancy_states`: reused by the logit M-step's identifiability gate
  (imported inside the function to avoid an import cycle; precedent
  `switching_kalman.py:3322-3324`).
- `switching_kalman.py:2621-2711` — `compute_expected_complete_log_likelihood`:
  the discrete term at `:2708-2711` already broadcasts, so a `(n_time - 1, S, S)`
  stack can be passed as `discrete_transition_matrix`; docstring + test only.
- `switching_kalman.py:3087-3184` — `compute_transition_sufficient_stats` and
  `:3244-3333` (`compute_transition_q_function`, `compute_transition_q_from_params`):
  untouched. They are the *continuous* transition-matrix (`A`) statistics, not
  the discrete M-step (the request suggested checking whether they were
  structured for the discrete update; they are not).
- `utils.py:1769-1824` — `hmm_viterbi`: accepts a `(n_time, K, K)` transition
  stack; `:1798` and `:1802` index it per step.
- `utils.py:1408-1457` — `validate_transition_matrix`: untouched; the new
  validator for covariates/weights lives in the new module.
- `oscillator_models.py:353-456` — `BaseModel`: new attributes and a
  keyword-only `transition_regularization` constructor option; `_e_step`
  (`:886-970`, filter call `:913-923`, GPB1 call `:963-966`) forwards the new
  keywords; `_m_step` (`:972-1030`, install at `:1027-1028`) and
  `DirectedInfluenceModel._m_step_reparameterized` (`:2023-2084`, install at
  `:2082-2083`) route through one install helper; `fit` (`:1082-1142`) and
  `fit_sgd` (`:1146-1194`) gain `transition_covariates=`; `_sgd_param_attrs`
  (`:404-413`), `_EM_SNAPSHOT_KEYS` (`:415-438`), the warm-init restore list
  (`:794-800`) and `OscillatorParameterBase._validate_parameter_shapes`
  (`:288-299`) learn the new attribute. `_build_param_spec` / `_sgd_loss_fn` of
  COM (`:1420-1462`), CNM (`:1745-1811`) and DIM (`:2169-2313`) add the weights.
  COM/CNM/DIM constructors forward `**kwargs` to `BaseModel`
  (`oscillator_models.py:1245-1259` for COM), so the new constructor option
  needs no per-subclass plumbing.
- `simulate/simulate_switching_kalman.py:98-130` — `simulate`: the discrete
  sampling loop at `:118-121` draws from a per-step matrix when covariates are
  given. `simulate_distinguishable_states` (`:228-`, sampling at `:277-278`) is
  untouched.

**Point-process switching family (phase 1b, 2b).**

- `switching_point_process.py:2042-2056` / `:2415-2429` — jitted core and public
  wrapper of `switching_point_process_filter`: new keywords; the scan body
  (`:2194-2321`, discrete update `:2277-2292`, scan `:2355-2376`) mirrors the
  Gaussian filter. `_validate_switching_point_process_filter_shapes`
  (`:286-370`) and `_validate_discrete_state_transitions` (`:372-428`, host-side,
  tracer-skipped at `:396-397`) validate the new inputs.
- `switching_point_process.py:2473-2637` — `SwitchingPointProcessBase`: attributes
  and constructor option; `_e_step` (`:2889-3006`, filter call `:2934-2947`, GPB1
  call `:2989-2992`), `_m_step_dynamics` (`:3012-3135`, install at `:3124-3125`),
  `_shared_sgd_param_spec` (`:3379-3409`), `_sgd_loss_fn` (`:3411-3470`, filter call
  `:3439-3453`, penalties `:3455-3468`), `_store_sgd_params` (`:3472-3485`),
  `_validate_parameter_shapes` (`:2786-2836`), `fit_sgd` (`:3313-3359`).
- `switching_point_process.py:3610-3719` — `SwitchingSpikeOscillatorModel.__init__`;
  `fit` (`:3841-4007`, snapshot/restore `:3966-3984`).
- `point_process_models.py:173-229` — `BaseSwitchingPointProcessModel.__init__`;
  `_snapshot_em_state` attrs (`:474-497`); `fit` (`:516-568`) / `_fit_single`
  (`:570-636`); `DirectedInfluencePointProcessModel._m_step_reparameterized`
  (`:1332-1419`, install at `:1368-1369`).
- `simulate/simulate_switching_spikes.py:15-30` — `simulate_switching_spike_oscillator`:
  the discrete draw in `_step` (`:153-155`) becomes covariate-dependent; the
  scan at `:170-175` gains an `xs` input.

**Choice switching family (phase 1c, 2b).**

- `switching_choice.py:206-249` / `:252-271` — public and jitted
  `switching_choice_filter`: new keywords; scan body `_step` (`:432-490`,
  discrete update `:464-469`), scan inputs (`:499`). The dynamics-covariate
  plumbing (`:331-343`, `B u_t`) is the precedent for both-or-neither
  validation and is untouched.
- `switching_choice.py:544-659` — `switching_choice_smoother`: gains the
  keywords; its per-step call into `switching_kalman_smoother` (`:601-608`)
  passes the per-step matrix; scan `xs` (`:626-631`).
- `switching_choice.py:687-1248` — `SwitchingChoiceModel`: attributes (`:763-778`),
  `_populate_uncertainty` (`:813-822`, the predicted discrete prior at `:819`
  multiplies by the fixed matrix and must use the stack), `_run_filter`
  (`:912-931`), `fit` (`:933-1011`, covariate binding `:965-971`), `_run_smoother`
  (`:1013-1034`), `_m_step` (`:1036-1118`, install `:1115-1118`), `fit_sgd`
  (`:1122-1173`), `_build_param_spec` (`:1179-1203`), `_sgd_loss_fn`
  (`:1205-1227`), `_store_sgd_params` (`:1229-1238`).
- `switching_choice.py:1260-1359` — `simulate_switching_choice_data`
  (`_state_step` at `:1325-1330`).

**Verification infrastructure reused.**

- `tests/oracles.py:473-651` — `switching_lgssm_exact_posterior`: accepts a
  `(n_time, K, K)` stack (`:503`, `:515`, `:525`). Every other oracle quantity
  (smoothed joint `(n_time - 1, K, K)`, pair moments) is already per time step.
- `tests/test_oracle_switching_kalman.py` (`_random_switching_model` `:73-126`,
  `_run_library` `:129-157`, `_assert_*_exact` `:173-219`, `_fd_grad` `:462-480`),
  `tests/test_oracle_switching_point_process.py` (`laplace_path_oracle` `:171-188`,
  `model_posteriors` `:285-312`), `tests/test_approximation_trends.py`
  (`_assert_decreasing` `:63-65`, `_gpb_errors` `:116-131`),
  `tests/test_invariances.py` (`_switching` `:248-262`, `_run_switching` `:265-282`,
  relabelling test `:332-345`), `tests/test_switching_kalman.py`
  (`simple_skf_model` `:356-433`, Viterbi tests `:6994-7057`),
  `tests/test_switching_choice.py` (`:228-278`, `:1121-1217`),
  `tests/test_contingency_belief.py` (`:1078-1197`), `tests/test_utils.py:703-712`.

## Scope and dependency policy

### Goals

- Covariate-dependent discrete transitions (phase 1) and continuous-state-dependent
  transitions (phase 2) for all three switching families, through one primitive,
  with identical keyword names, shapes and time alignment everywhere
  ([shared-contracts.md](shared-contracts.md)).
- EM (logit M-step with the existing Dirichlet-style pseudo-counts and an L2
  penalty) and SGD (weights added to the parameter spec) for the new parameters,
  in every model class that already learns `discrete_transition_matrix`.
- Exactness where exactness is available: the discrete-path-enumeration oracle
  extended to time-varying transitions (phase 1) and a 1-D grid oracle for the
  recurrent model (phase 2), with approximation-trend tests for the phase-2
  plug-in.
- Simulators for all three families that generate covariate- and
  state-driven switching, so recovery tests and downstream plans have data.
- The fixed-matrix path is unchanged: bit-identical outputs when the new options
  are `None`.

### Non-Goals

- No exact expectation of the softmax under the Gaussian `x_{t-1}` (probit /
  unscented / Monte-Carlo corrections). Phase 2 ships the collapsed-mean plug-in
  only; the correction is a recorded follow-up (Open Question 2).
- No change to `switching_kalman_smoother_gpb2` (it does not use the discrete
  transition matrix) and no new discrete update inside
  `_switching_kalman_m_step_inner` (its count-based estimate is simply not
  installed on the covariate path).
- No structural zeros under covariate-driven transitions: a softmax row is
  strictly positive. A baseline `discrete_transition_matrix` with exact zeros
  combined with the new options is rejected (see
  [validation contract](shared-contracts.md#validation-contract)), not silently
  lifted.
- `hamiltonian_switching.py` (its own filter, SGD-only; the discrete matrix is
  initialised at `:462-467` and exposed to SGD at `:572-576`) is out of scope.
  Revisit trigger: when that model adopts `_update_discrete_state_probabilities`.
- `contingency_belief.py`'s public functions keep their signatures and layouts;
  this plan only moves shared helpers behind re-exports.
- No spline / basis expansion of covariates: `transition_covariates` is a design
  matrix the caller builds (as `ContingencyBeliefModel._build_design_matrix`
  does for the intercept, `contingency_belief.py:1119-1130`).
- No `predict` / forecasting API that takes future covariates.

### Dependency policy

No new third-party dependency. `jax.scipy.optimize.minimize` (BFGS) is already
used by `contingency_belief._optimize_transition_rows`.

Cross-plan links (do not restate their content):

- `docs/plans/multi-map-place-fields/` lists speed-gated transitions as an
  optional dependency on phase 1 of this plan. It consumes the phase-1 keyword
  names and shapes fixed in [shared-contracts.md](shared-contracts.md); nothing
  here depends on it.
- `docs/plans/2026-04-05-value-gated-sequence-expression.md` (deferred) is a
  downstream consumer of phase 2 (`transition_state_weights=`).
- `docs/plans/identifiability-diagnostics/` — when its identifiability report
  gate lands, rows of `transition_weights` / `transition_state_weights` that hit
  the low-occupancy gate ([M-step contract](shared-contracts.md#m-step-contract))
  should be surfaced through that report. Until then the logged
  `warn_low_occupancy_states` warning is the only signal. This plan does not
  depend on it.

## Metrics

- **Bit-for-bit on the None path**: every filter / smoother / Viterbi output of
  the three families is `array_equal` with and without the new keywords passed
  as `None` (fast tests), and `tests/test_em_golden_regression.py` passes
  unchanged (do not regenerate goldens).
- **Reduction identity**: with `transition_weights` all zero and any covariates,
  outputs match the fixed-matrix path to `1e-12` (the centered-softmax
  round-trip error is ~1e-16; measured `1.1e-16` on a 3x3).
- **Exactness**: filter, GPB1 and GPB2 smoothers match the path-enumeration
  oracle with time-varying transitions at `RTOL = 1e-8` in the exact regimes
  (identical per-state parameters; `T = 2` for GPB2) while the transition stack
  genuinely varies (`ptp` over time `> 0.1`).
- **M-step**: the logit update is a stationary point of the exact expected
  complete-data transition term (finite-difference gradient `< 1e-6`) and does
  not decrease it.
- **Recovery (slow)**: simulated speed-gated switching recovers the sign of every
  entry of `transition_weights` and its magnitude within a factor of two over 3
  seeds, in each family.
- **Phase-2 approximation trend**: the plug-in / exact gap decreases strictly
  over five covariance scales (primitive level via Gauss-Hermite quadrature;
  filter level via the grid oracle) and is `> 1e-2` at the loosest scale.
- **Invariances**: the discrete-state relabelling test still passes with
  covariate- and state-dependent transitions (reference-column re-centering as
  in [designs.md](designs.md#h-relabelling-transform)).
- **Tooling**: `uv run ruff check src/`, `uv run ruff format --check src/`,
  `uv run mypy` (with `discrete_transitions.py` added to the checked files) clean.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Silent divergence between the three families (the memory rule `fix-sibling-implementations` / `dim-pp-mirrors-gaussian-dim`: DIM-PP has repeatedly lagged the Gaussian DIM). | One primitive does the work; each phase file carries the [sibling-mirroring checklist](shared-contracts.md#sibling-mirroring-checklist) as tasks, and the phase-1b/1c/2b validation slices re-run the phase-1a identity tests on their family. The install sites (`oscillator_models.py:1027-1028`, `:2082-2083`; `switching_point_process.py:3124-3125`; `point_process_models.py:1368-1369`; `switching_choice.py:1115-1118`) all route through one helper per base class. |
| Time-alignment off-by-one (covariate row `t` vs `t+1` driving `S_{t-1} -> S_t`); `tests/test_contingency_belief.py:1113-1143` documents exactly this bug class. | Single alignment rule in [shared-contracts.md](shared-contracts.md#time-alignment-contract); the oracle uses the same rule; a test with a one-hot covariate at a single time step asserts the switch happens at that step and not its neighbour. |
| Softmax cannot represent structural zeros; lifting a zero to `1e-10` would silently change a caller's model. | Filters raise `ValueError` (host-side, tracer-skipped) when the baseline matrix has an exact zero and any new option is given. |
| Unidentified `transition_weights` rows when a state is rarely visited or a covariate never varies (perfect separation inflates logits). | Per-row occupancy gate (`minimum_state_occupancy(n_regressors)`, keep previous coefficients, `warn_low_occupancy_states`), L2 penalty `transition_regularization` (default `1e-5`, as `ContingencyBeliefModel`), Dirichlet pseudo-counts via the existing `transition_prior`. |
| The phase-2 plug-in `softmax(a + W m)` is a biased estimate of `E[softmax(a + W x)]`; a bug in the plug-in would be indistinguishable from the approximation without an exact reference. | 1-D grid oracle for the recurrent model ([designs.md](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle)) plus a self-check that it reproduces the path-enumeration oracle when `W = 0`; monotone-trend tests with a nonzero-gap guard. |
| The smoother under phase 2 must use exactly the transition matrices the filter used; recomputing them differently would make the E-step inconsistent. | Both the filter (in-scan) and the smoother (from `filter_mean`) call the same helper on the same collapsed means; a fixed-point test expresses the plug-in as phase-1 covariates (`vec(m_{t-1})`) and asserts `1e-12` agreement ([designs.md](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle)). |
| Per-row BFGS in the M-step is `S` optimisations of `(1 + F [+ n_cont]) (S - 1)` coefficients each EM iteration. | Same cost class as `ContingencyBeliefModel` today; jitted once per shape (`tests/test_contingency_belief.py:1157-1197` guards compile-once). If it dominates, cap `maxiter` and warm-start from the previous coefficients (already the case). |
| Memory of the `(n_time, S, S)` stack for long recordings (e.g. 3e6 bins x 9 = 216 MB float64). | Accepted (stated as fine in the request). The stack is a scan input, not a carry; document the footprint in the filter docstrings. |

## Rollout Strategy

Additive, keyword-only options on existing functions and classes; no signature
breaks, no deprecation phases. With every new option at `None`:

- the filters, smoothers and Viterbi decoders trace exactly as today (the scan
  gets a `None` leaf in `xs`, the fixed matrix stays a closed-over constant; see
  [designs.md](designs.md#a-transition-stack-resolution-and-in-scan-step));
- the model classes install the count-based `discrete_transition_matrix` from
  `switching_kalman_maximization_step` as before;
- the SGD parameter specs are unchanged;
- the simulators draw from the fixed matrix with the same RNG stream.

Each phase ships independently: 1a is usable on its own (Gaussian family +
primitive), 1b and 1c add the sibling families, 2a and 2b add the recurrent
option. Phase 2 builds on phase 1's stack machinery but adds no new keyword to
phase 1's API.

## Open Questions

1. **Default `transition_regularization`.** Current best answer: `1e-5` on the
   non-intercept logit coefficients, matching `ContingencyBeliefModel`
   (`contingency_belief.py:911`), exposed as a keyword-only constructor option on
   the three base classes and as the equivalent penalty in the SGD losses
   ([M-step contract](shared-contracts.md#m-step-contract)). Revisit if the
   recovery tests need stronger shrinkage for `transition_state_weights`.
2. **Exact expectation of the transition under the Gaussian `x_{t-1}`.**
   Deferred. Trigger: a downstream consumer (the value-gated plan) reports that
   the plug-in bias changes a conclusion, or the phase-2 trend test's loosest
   scale gap exceeds `0.1` on realistic parameters. Candidates, in order of
   cost: unscented / Gauss-Hermite expectation of the row softmax inside the
   scan (deterministic, `O(n_cont)` sigma points); the probit-style closed form
   for `S = 2` (`E[sigmoid(a + w x)]` with the `sqrt(1 + pi lambda^2 sigma^2 / 8)`
   approximation); Monte-Carlo. The primitive's in-scan hook
   (`transition_at_step`) is the single place to swap.
3. **Should the covariate path allow a baseline with structural zeros by keeping
   the zero pattern as a mask?** Current best answer: no (rejected with
   `ValueError`); a masked softmax changes the reference-column convention and
   no consumer needs it. Trigger: a model that needs both forbidden transitions
   and covariate modulation.
4. **Layout of `compute_input_output_transition_matrix`
   (`contingency_belief.py:173-197`, weights `(S, S-1, d_h)`).** It differs from
   the contract's feature-axis-first layout `(n_features, S, S-1)`, which is
   what `ContingencyBeliefModel` stores (`transition_coefficients_`
   `(n_coef, S, S-1)`, `contingency_belief.py:976-987`) and what CLAUDE.md's
   "discrete axis trails" convention requires. Current best answer: leave the
   contingency filter's internal layout alone (it moves axes at
   `contingency_belief.py:1147` and `:1507`); do not add a third layout. Trigger
   for unifying: the contingency filter is rewritten onto the shared primitive.
5. **Should `ContingencyBeliefModel` itself be re-based on the shared primitive?**
   Deferred; it is an HMM without a continuous state and already has the
   feature. Trigger: a bug fix in the lifted M-step that the contingency model
   also needs (it will already receive it, since it calls the lifted optimizer).

## Estimated Effort

Source (excluding tests): phase 1a ~+750 LOC (new module ~300, `switching_kalman`
~150, `utils` ~20, `oscillator_models` ~150, simulator ~40, `contingency_belief`
import swap ~30, docs); phase 1b ~+250; phase 1c ~+200; phase 2a ~+250; phase 2b
~+150. Tests: 1a ~600, 1b ~300, 1c ~250, 2a ~500 (grid oracle ~150), 2b ~200.
