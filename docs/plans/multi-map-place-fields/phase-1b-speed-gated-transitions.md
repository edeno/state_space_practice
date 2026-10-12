# Phase 1b — Learning- and event-dependent map transitions

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md) · [contracts](shared-contracts.md)

**Gate:** do not start until `docs/plans/recurrent-switching-transitions/` phase 1 (the
`transition_covariates=` / `transition_weights=` interface) and this plan's
[phase 1](phase-1-static-maps-hmm.md) have both merged. Consume that interface; do not
restate or re-implement it. If its transition parametrisation differs from the
centered-softmax convention assumed below (`contingency_belief.compute_input_output_transition_matrix`,
last state as reference), adopt its parametrisation — only the M-step reshaping in this
file changes ([overview open question 7](overview.md#open-questions)).

Low et al. (2021) report that map switching covaries with running speed. Speed remains
the minimal recovery fixture. This phase also tests learning history and environmental
events using the same interface. It makes the switch hazard a
*function of the covariate*: `Z_t[i, :] = softmax(logits_i + Gamma_i h_t)`, so the
covariate effect is a fitted parameter with a likelihood, and `hmm_forward_backward` runs on
the resulting `(n_time - 1, K, K)` stack — the branch phase 1 already made exact.

This phase is optional for basic recurrence, and required for claims about fitted
learning/event effects on within-visit transitions. Transition covariates do not
account for direct effects on firing; phase 3 still needs behavioral emission and
spatial-support controls. Follow the [sampling and barrier design](sampling-and-barriers.md).
In particular, [phase 1a](phase-1a-directional-fields.md) models position-dependent
directional firing within each map. Adding direction to a switch hazard cannot
substitute for that emission model or phase 3's evolving-directionality alternative.
If phase 2 has already merged, accept its leading
sequence axis, apply covariates only to real within-sequence edges, and exclude
padding/boundaries from transition fitting and behavior-alignment summaries. Test
that batched covariate sufficient statistics equal explicit per-sequence sums.

**Inputs to read first:**

- `docs/plans/recurrent-switching-transitions/` — the merged interface (argument names, shapes, parametrisation, any shared M-step helper). Authoritative over this file where they differ.
- `src/state_space_practice/multi_map_place_field.py` (phase 1) — `fit`, `_e_step`, `_m_step`, `_build_param_spec`, `_sgd_loss_fn`, `_finalize_fit`, `viterbi_path`.
- `src/state_space_practice/utils.py` — `hmm_forward_backward` / `hmm_filter` stack input ([contract](shared-contracts.md#hmm-forward-backward)); `hmm_viterbi` `:1769-1824` (the backward step at `:1801-1805` indexes a constant `log_transition_matrix`; the prerequisite's `(T, K, K)` stack branch indexes `[t + 1]`).
- `src/state_space_practice/contingency_belief.py:84-98` `centered_log_softmax`, `:115-130` `centered_softmax_inverse`, `:173-197` `compute_input_output_transition_matrix`, `:226-278` `dirichlet_neg_log_likelihood` (per-sample averaged loss with `alpha - 1` pseudo-counts), `:804-846` `_optimize_transition_rows` (vmapped BFGS over from-states), `:1296-1320` how `_m_step` reshapes coefficients `(n_coef, S, S-1) ↔ (S, n_coef (S-1))` and pairs `xi[t]` with design row `t + 1` (`:1298-1302`).
- `src/state_space_practice/parameter_transforms.py:157` `UNCONSTRAINED`.
- `tests/test_utils.py:703-712` — the existing `hmm_viterbi` structural-zero test that must keep passing bit-identically.

**Contracts referenced:**

- [HMM forward-backward](shared-contracts.md#hmm-forward-backward) — consumed with a stack; no change.
- [Model attributes](shared-contracts.md#model-attributes) — extended with `transition_logits_` and `transition_weights_`; `transition_matrix_` becomes the baseline (zero-covariate) matrix. Every phase-1 row keeps its meaning.
- [Sampling and barriers](sampling-and-barriers.md) — coverage, event timing, uncertainty and interpretation.

**Designs referenced:** [designs A](designs.md#a-exact-hmm-forwardbackward-in-log-space) (stack semantics), [designs I](designs.md#i-behavioural-alignment-helpers) (the post-hoc comparison this replaces).

## Tasks

- **Define a small event/history covariate set and availability times.** Include speed
  as a nuisance covariate, barrier presence, first-encounter events, lagged reward/error,
  cumulative experience and elapsed time as candidate columns; select a predeclared
  subset with enough independent events to estimate it. Save units, encoding, timestamp,
  availability cutoff and training-only scaling for every column. Destination-bin
  indexing does not make an outcome observed at the end of that bin available at its
  start: lag it for prediction, or label the fit contemporaneous association. A barrier
  inserted between visits has no within-sequence edge; analyse initial map occupancy
  in phase 3 rather than inventing an overnight transition.

- **Link learning without circularity.** Estimate learning/uncertainty from behavior
  alone first, using training-fit parameters and past behavior at prediction time.
  Evaluate covariate effects across behavioral posterior draws or a documented
  uncertainty sensitivity analysis; a plug-in learning curve is not known truth.
  Test whether filtered neural state adds prediction of later behavior beyond its
  history on held-out visits. Fully joint learning/neural inference is deferred to
  its sibling plan and cannot serve as independent evidence for this relationship.

- **Consume the existing `hmm_viterbi` stack branch.** Recurrent transitions
  already supplies the public `(T, K, K)` convention: entry `t` is the
  transition into bin `t`, row 0 is ignored, and the backward step indexes
  `[t + 1]`. Do not add a second `(T-1, K, K)` public convention. Multi-map's
  `hmm_filter` / `hmm_forward_backward` use edge stacks `(T-1, K, K)`; adapt
  that stack only at the `viterbi_path()` boundary:

  ```python
  viterbi_stack = jnp.concatenate([jnp.eye(K)[None], edge_stack], axis=0)
  path = hmm_viterbi(initial_probs, viterbi_stack, log_likelihoods)
  ```

  The placeholder row is ignored, including `T=1`. Static transitions continue
  to use the existing 2-D call. Validate the public stack's leading length
  against the likelihood time axis so a missing placeholder fails clearly.

- **Model parameters and `fit(..., transition_covariates=None)`.** New attributes
  `transition_logits_` `(K, K-1)` and `transition_weights_` `(K, K-1, d_h)`, initialised
  from `transition_matrix_` via `centered_softmax_inverse` and zeros. Store the covariates
  `(n_time, d_h)` (validated finite; caller standardises them) for the fit. Build the stack
  with `jax.vmap(compute_input_output_transition_matrix, in_axes=(None, None, 0))(logits,
  weights, covariates[1:])` → `(n_time - 1, K, K)` (row `t` uses `h_{t+1}`, the covariate at
  the bin entered, matching `contingency_belief.py:1298-1302`). `_e_step` passes the stack
  to `hmm_forward_backward`; `transition_matrix_` is refreshed as
  `centered_softmax(transition_logits_)` after every M-step so phase-1 consumers keep
  working. With `transition_covariates=None` nothing changes: the phase-1 code path runs
  (test pins the LL equality).

- **Transition M-step with covariates.** Prefer the shared M-step helper the
  recurrent-transitions plan ships. Fallback (documented): design
  `D = [1, h_t]` for `t = 1..n_time-1`, shape `(n_time - 1, 1 + d_h)`; coefficients
  `(1 + d_h, K, K-1)` with row 0 = logits, rows 1.. = weights; `x0_all =
  coefficients.transpose(1, 0, 2).reshape(K, -1)`; `response_all = xi.transpose(1, 0, 2)`;
  `alpha = get_transition_prior(concentration, stickiness, K)`; call
  `_optimize_transition_rows(x0_all, response_all, alpha, D, l2)` and reshape back exactly
  as `contingency_belief.py:1304-1320`. `l2` = new constructor argument
  `transition_regularization: float = 1e-5` (only the covariate weights are penalised, as
  in `dirichlet_neg_log_likelihood` `:276-277`). `_log_prior` gains the matching
  `-l2 * sum(weights**2)` term so EM's reported objective and the SGD loss agree; the
  Dirichlet term is evaluated on the *baseline* matrix (document this — with covariates the
  pseudo-count prior is a regulariser, not a conjugate prior, as `:239-244` notes).

- **SGD path.** When covariates are present, `_build_param_spec` exposes
  `transition_logits` and `transition_weights` as `UNCONSTRAINED` instead of
  `transition_matrix` as `STOCHASTIC_ROW`; `_sgd_loss_fn` builds the stack and calls
  `hmm_filter`. `_sgd_param_attrs` maps the two new keys; `_store_sgd_params` refreshes
  `transition_matrix_`.

- **Interpretation helpers.** `switch_hazard(covariate_values) -> (n_values, K)`:
  `1 - diag(Z(h))` per map for a grid of covariate values, so a user can plot hazard vs
  speed, experience or an event indicator while holding other columns at declared
  supported values. Avoid unsupported covariate combinations. `viterbi_path()` pads
  the stored edge stack as above before calling the shared helper.

- **Docs.** CHANGELOG `### Added`: "`MultiMapPlaceFieldModel.fit(...,
  transition_covariates=)`: learning- and event-dependent map transitions".
  Class docstring: the transition model, the standardisation advice, the `xi[t] ↔ h_{t+1}`
  pairing, and a two-line example. Extend `notebooks/multi_map_ca1_smoke.py` with a
  speed-gated fit and a printed `switch_hazard` table at speed quantiles.
  Add a synthetic barrier/encounter example; the controlled real-data learning/barrier
  benchmark lives in phase 3, not in the single-session smoke script. Retain a
  constant-transition comparison and test held-out gain beyond speed/position/coverage.

## Deliberately not in this phase

- Any change to the recurrent-transitions interface itself, or a second parametrisation of it here.
- Covariate-dependent *maps* (weights depending on speed) — a different model.
- Nonlinear (spline) covariate effects on the hazard; callers can pass basis-expanded covariates.
- Drift and first implementation of masks/multi-session support (phase 2 owns the
  latter; if already merged, maintain its semantics in this extension).
- Changing geometry/bases: phase 3's prototypes. Unknown map count and sequential
  discovery: [phase 4](phase-4-unknown-map-repertoire.md), initially with constant
  transitions; joint integration with these covariate logits needs a later design.
- A latent joint learning model or covariate-dependent initial map distribution in
  the production API; phase 3 may use a small visit-level analysis reference.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_utils.py::TestDiscreteStateUtilities::test_hmm_viterbi_stack_matches_enumeration` | with a random edge stack `(T-1, K, K)` padded to public shape `(T, K, K)`, the path equals the oracle argmax ([designs H](designs.md#h-exact-path-enumeration-oracle-for-tests)); a stack of identical matrices gives the same path as the 2-D call (bit-identical); changing ignored row 0 has no effect, including T=1, and an unpadded stack raises `ValueError` |
| `test_utils.py::...::test_hmm_viterbi_preserves_structural_zero_transitions` (existing, `:703`) | unchanged and passing |
| `test_multi_map_place_field.py::test_zero_transition_weights_match_static_fit` | `from_parameters` with covariates and `transition_weights_ = 0`: `predict_map_posterior` LL and posteriors equal the static model's (`1e-12`) |
| `::test_covariate_pairing_uses_destination_bin` | a covariate that is huge at exactly one bin `t*` (forcing a switch into `t*`) moves `switch_probability_` mass to index `t* - 1`, not `t*` |
| `::test_speed_gated_fit_recovers_positive_gain` (slow) | `simulate_multi_map_session(transition_covariate=c, transition_gain=1.0)` with `c` standardised AR(1): fitted `transition_weights_[0, 0, 0] < -0.3` and `transition_weights_[1, 0, 0] > 0.3` (for K = 2, centered softmax: row 0's logit is `log P(0→0)/P(0→1)`, which falls with the hazard; row 1's logit `log P(1→0)/P(1→1)` rises); `switch_hazard` is increasing in `c` for both maps; Viterbi accuracy ≥ 0.9 |
| `::test_speed_gated_fit_with_no_effect_recovers_near_zero_gain` (slow) | `transition_gain=0.0`: `\|transition_weights_\| < 0.2`; held-out LL of the covariate model within `2 * d_h * K * (K-1)` nats of the static model (no spurious gain) |
| `::test_covariate_fit_objective_monotone` (slow) | `assert_ll_monotonic(log_likelihood_history_, tol=1e-6 * \|LL\|)` — the BFGS row optimiser is approximate; the tolerance is looser than phase 1's and the reason is stated in the test docstring |
| `::test_fit_sgd_with_covariates_matches_em` (slow) | small problem: SGD final objective within `1e-3` relative of EM's; `transition_weights_` signs agree |
| `::test_covariate_validation` | covariates with wrong `n_time`, non-finite values, or 1-D shape → `ValueError`; `score(..., transition_covariates=)` requires covariates when the model was fitted with them |
| Event/history timing | Distinct insertion and encounter times recover the planted encounter effect; modifying future rewards/choices does not change earlier predictive covariates or filtered estimates. |
| Coverage-only negative control | Stationary fields with changed route sampling and event-correlated speed are included in the phase-3 false-switch calibration; a transition effect must add predictive value beyond those controls. |
| Event generalisation | Fit on earlier manipulations and score held-out repetitions/animals; fixed event effects alone cannot pass a neural switching claim without held-out-neuron evidence on shared support. |

Mark slow / integration tests explicitly where fitting hides in helpers; the rest are
auto-marked (`conftest.py:143`).

## Fixtures

- `simulate_multi_map_session(..., transition_covariate=c, transition_gain=g)` from phase 1
  ([designs G](designs.md#g-two-map-session-simulator)) with a standardised AR(1)
  covariate `c` (`phi = 0.99`, unit variance, seeded) — speed-like autocorrelation without
  needing a speed-varying trajectory.
- The phase-1 `tiny_problem` and `enumerate_hmm_posterior` oracle for the stack tests.
- Synthetic insertion/encounter/reward histories with known information cutoffs,
  repeated events and stationary-coverage controls from [sampling and barriers](sampling-and-barriers.md).
  Event nulls preserve the experiment's block structure; do not circularly shift across
  nonstationary geometry blocks as if they were exchangeable.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind). (None; confirm the 2-D `hmm_viterbi` path is unchanged and the `transition_covariates=None` path reproduces phase-1 numbers exactly.)
- User-facing documentation listed as tasks is updated, not deferred.
- Additionally: the interface consumed is the merged recurrent-transitions one (names, shapes, parametrisation), not a local re-implementation.
