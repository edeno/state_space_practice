# Phase 2 — WoLF in the switching Kalman filter and the LFP oscillator models

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#s1-shared-weight)

Depends on phase 1 (`weighted_kalman_measurement_update`, `standardized_residual`, `RobustOutput`, `imq_weight`).

**Inputs to read first:**

- [src/state_space_practice/switching_kalman.py:23-57, 140-142](../../../src/state_space_practice/switching_kalman.py) — imports from `kalman`; `_kalman_filter_update_per_discrete_state_pair` (the double vmap to mirror) and the collapse helper.
- [src/state_space_practice/switching_kalman.py:454-683](../../../src/state_space_practice/switching_kalman.py) — `_update_discrete_state_probabilities` and `_first_timestep_discrete_update`: called twice when robust (once with per-pair objectives, once with unweighted LLs); understand the support masking so the second call is safe.
- [src/state_space_practice/switching_kalman.py:686-790](../../../src/state_space_practice/switching_kalman.py) — `_first_timestep_kalman_update` (vmap at 747-758).
- [src/state_space_practice/switching_kalman.py:793-1076](../../../src/state_space_practice/switching_kalman.py) — `switching_kalman_filter`: `_step` (871-988; per-pair update 933-945, discrete update 947-957, LL 962, collapse 967-973), first step (992-1007), scan (1022-1042), return (1068-1076).
- [src/state_space_practice/switching_kalman.py:1144-1156](../../../src/state_space_practice/switching_kalman.py) — Viterbi's use of the per-pair update: stays on the unweighted helper.
- [src/state_space_practice/switching_kalman.py:2042-2060, 2150-2300, 2303-2597](../../../src/state_space_practice/switching_kalman.py) — `weighted_sum_of_outer_products`, `_switching_kalman_m_step_inner` (observation block 2206-2251, `gamma2` 2222-2229, transition block 2253-2269), `switching_kalman_maximization_step` (inner call 2488-2512, occupancy gate 2514-2539).
- [src/state_space_practice/oscillator_models.py:322-337, 415-438, 440-533, 535-576, 578-604](../../../src/state_space_practice/oscillator_models.py) — shared-R stacking, `_EM_SNAPSHOT_KEYS`, `BaseModel.__init__`, snapshot/restore/clear, `__repr__`.
- [src/state_space_practice/oscillator_models.py:886-1080, 1082-1142, 1219-1220](../../../src/state_space_practice/oscillator_models.py) — `_e_step`, `_m_step`, `_m_step_fixed_and_previous_params`, `_pool_measurement_covariance`, `fit`, `_finalize_sgd`.
- [src/state_space_practice/oscillator_models.py:1245-1260, 1444-1462, 1518-1524, 1777-1811, 1906-1912, 2008-2129, 2253-2313](../../../src/state_space_practice/oscillator_models.py) — the three constructors' `super().__init__(..., **kwargs)` calls, the three `_sgd_loss_fn`s (`result[6]`), DIM's `_m_step` / `_m_step_reparameterized`.
- [src/state_space_practice/tests/recovery_helpers.py:244-368, 493-552](../../../src/state_space_practice/tests/recovery_helpers.py) — `oscillator_model_at_truth`, `simulate_from_oscillator_model` (the LFP simulator to contaminate), `observation_objective`, `central_difference_gradient`.
- [src/state_space_practice/tests/test_oscillator_models.py:32-116, 2592, 3072, 3238-3300, 3402](../../../src/state_space_practice/tests/test_oscillator_models.py) — fixtures, `TestOscillatorEMRollback`, `test_dim_em_started_at_truth_does_not_roll_back`, the M-step stationary-point machinery to copy for the weighted observation block.
- [src/state_space_practice/tests/test_em_golden_regression.py:263-295, 791-793](../../../src/state_space_practice/tests/test_em_golden_regression.py) — oscillator golden cases that must keep passing (bit-identical `None` path).
- [src/state_space_practice/tests/oracles.py:198-272, 473-490](../../../src/state_space_practice/tests/oracles.py) — `lgssm_dense_posterior` (per-time `R` stack, `prior_on_first_state=True` for the `x_1` convention) and `switching_lgssm_exact_posterior`, whose `measurement_cov` is per *state* `(m, m, K)`, not per time. The oracle check of the robust switching filter is therefore done for `n_discrete_states = 1` through `lgssm_dense_posterior(prior_on_first_state=True, measurement_cov=R / w_t²)`.

**Contracts referenced:**

- [Return-arity rule](shared-contracts.md#return-arity) — row `switching_kalman_filter`; `switching_kalman_maximization_step(robust_weights=)` never changes its return.
- [Weight granularity](shared-contracts.md#weight-granularity) — one weight per time step, shared across pairs.
- [Invariants](shared-contracts.md#invariants) 1–6 (the three `_sgd_loss_fn`s are siblings).

**Designs referenced:** [S1](designs.md#s1-shared-weight), [S2](designs.md#s2-switching-mstep), [G4](designs.md#g4-em-sgd).

## Tasks

- **`switching_kalman.py` — shared-weight robust filter.** Import `standardized_residual`, `RobustOutput`, `RobustWeight` (from `utils`) and `weighted_kalman_measurement_update` (from `kalman`). Add `_weighted_kalman_filter_update` and `_weighted_kalman_filter_update_per_discrete_state_pair` next to lines 49-57 as in [S1](designs.md#s1-shared-weight). Give `_first_timestep_kalman_update` a keyword `robust_weight=None` (None → existing body; else compute the shared first-step weight, vmap the weighted update, call `_first_timestep_discrete_update` twice, return the extra `(objective, weight)`). Re-decorate `switching_kalman_filter` with `functools.partial(jax.jit, static_argnames=("robust_weight",))`, add keyword-only `robust_weight: RobustWeight | None = None`; keep the current `_step`/scan for `None`; add a robust `_step` that computes the shared weight from the mixture prior predictive (S1 code), runs the weighted per-pair update, calls the discrete update twice, accumulates both `marginal_log_likelihood` (unweighted) and `objective`, emits `weight`, and collapses exactly as today. Return the 7-tuple plus `RobustOutput(objective, weights)` (`weights` shape `(n_time,)`, first-step weight prepended). Docstring: both objectives, the shared-weight rationale in two sentences, the arity rule.

- **`switching_kalman.py` — weighted observation M-step.** `switching_kalman_maximization_step(..., robust_weights=None)` validates `robust_weights.shape == (n_time,)` when concrete, forms `obs_weights` per [S2](designs.md#s2-switching-mstep) and passes it plus a static `use_robust_weights` flag to `_switching_kalman_m_step_inner` (extend its `static_argnames`). Inside, weight `weighted_cov_sum`, `gamma`, `delta`, the R scatter and its divisor by `obs_weights` when the flag is set, keeping an unweighted `gamma` for `gamma2` and the transition block; add a per-state observation gate on effective weight, preserving previous H/R at zero weight independently of the unweighted dynamics occupancy gate. Require and forward previous H/R from `previous_params`; shared observation parameters use pooled effective weight. The unset branch is the current code. Docstring: the fixed-weight exactness, the bias note from [G3](designs.md#g3-bias) (one sentence + pointer to `kalman_maximization_step`).

- **`oscillator_models.py` — model opt-in.** `BaseModel.__init__` gains `robust_weight: RobustWeight | None = None` (stored as `self.robust_weight`; shown in `__repr__`); the subclasses need no signature change (they forward `**kwargs`). Add `"filter_robust_weights"` to `_EM_SNAPSHOT_KEYS`; initialise `self.filter_robust_weights: jax.Array | None = None` in `__init__`; set it to `None` in `_clear_smoother_state`. `_e_step`: pass `robust_weight=self.robust_weight`; when set, unpack the eighth output, store `self.filter_robust_weights = robust.weights`, return `robust.objective`; otherwise unchanged. `_m_step` and `DirectedInfluenceModel._m_step_reparameterized`: pass `robust_weights=self.filter_robust_weights` to `switching_kalman_maximization_step` (it is `None` when not robust). Add `BaseModel._negative_fit_objective(result)` returning `-(result[7].objective if self.robust_weight is not None else result[6])` and use it in the three `_sgd_loss_fn`s (also pass `robust_weight=self.robust_weight` to their `switching_kalman_filter` calls). `fit` / `fit_sgd` docstrings: the returned history is the generalised-Bayes objective when `robust_weight` is set; `filter_robust_weights` is the per-bin artifact diagnostic. `decode()` / `predict_proba()` need no change.

- **Tests — new module `tests/test_robust_switching_kalman.py` and `tests/test_robust_oscillator_models.py`.** Implement the validation slice. Contamination for the LFP tests: take `simulate_from_oscillator_model(oscillator_model_at_truth(kind, ...), n_time, rng)` and replace 5 % of bins of `y` with `N(0, (20 √R)²)` draws (or add a chewing-like burst: 3 consecutive bins × 15 √R); keep the mask.

- **User-facing docs.** CHANGELOG `### Added` bullet: `robust_weight=` on `switching_kalman_filter`, `robust_weights=` on `switching_kalman_maximization_step`, constructor `robust_weight=` on the three oscillator models with the shared-weight semantics and the `filter_robust_weights` diagnostic. Follow-up issue text for DIM-PP parity (overview, non-goals) goes in the PR description, not the code.

## Deliberately not in this phase

- Robust Viterbi decoding (`switching_kalman_viterbi`) — keep it on the unweighted per-pair helper (overview non-goal).
- `DirectedInfluencePointProcessModel` and the other switching point-process models — non-goal with a revisit trigger (overview).
- Any change to `switching_kalman_smoother` / `_gpb2`, `compute_elbo`, or the ELBO tests: they consume filter outputs / posteriors and are unaffected.
- Relaxing `decrease_tol` in `BaseModel.fit` (open question 1) — only if the contamination EM test forces it, and then as its own commit with the evidence.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_switching_filter_none_is_bit_identical` | `switching_kalman_filter(robust_weight=None)` on the `switching_kalman_model_params` draws and on `oscillator_model_at_truth("COM", 2, True)` data equals a reference copy of the pre-change scan bit-for-bit (all 7 outputs); `test_switching_kalman.py`, `test_oracle_switching_kalman.py`, `test_likelihood_identities.py::TestSwitchingKalmanIdentities` and the oscillator golden cases in `test_em_golden_regression.py` pass untouched. |
| `test_single_state_matches_robust_kalman_filter` | With `n_discrete_states = 1`, the robust switching filter's means/covs/weights/objective equal `kalman_filter(robust_weight=...)` run with the `x_1` convention (first step measurement-only: compare from `t = 2` using the switching filter's first posterior as the Kalman filter's `init`) at `rtol=1e-10`; and equal `lgssm_dense_posterior(prior_on_first_state=True, measurement_cov=R / w_t²)` at `1e-8`. |
| `test_shared_weight_is_mixture_prior_predictive` | For `K = 2` with distinct `H_j`, the emitted weight equals `IMQWeight(c)(standardized_residual(y_t − Σ_ij π_ij H_j A_j m^i, R̄))` recomputed in NumPy from the previous step's filtered quantities at `rtol=1e-10`. |
| `test_outlier_does_not_flip_discrete_state` | Two well-separated regimes; inject one `20σ` outlier mid-regime: robust filtered `P(S_t)` at the outlier bin stays within `0.05` of the previous bin's; the standard filter's changes by `> 0.3` (guard). |
| `test_c_to_infinity_matches_standard_switching` | `imq_weight(c=1e150)`: 7 outputs within `1e-12` of `None`; `objective == ll`. |
| `test_robust_switching_gradients` | `jax.grad` of the objective w.r.t. the shared `measurement_cov` and `continuous_transition_matrix` finite and non-zero; two equal `IMQWeight` instances compile once. |
| `test_switching_mstep_weighted_block_is_stationary` | Fixed weights and GPB1 posteriors from `oscillator_model_at_truth("COM", 2, True)`: the weighted per-state observation objective (extend `recovery_helpers.observation_objective` with weights in the test) has vanishing `central_difference_gradient` at the returned `(H_j, R_j)`; transition outputs equal the unweighted call exactly; `robust_weights=None` output equals today's bit-for-bit. |
| `test_mstep_separates_observation_and_dynamics_gates` | A state with occupancy above the dynamics gate but exactly zero effective observation weight retains previous H/R bit-for-bit and still updates dynamics. Test both per-state and shared H/R, all-zero weights, a zero-weight occupied state alongside an observed state, and small strictly positive weights (which use the weighted maximizer). No NaNs under `jit`/`vmap`. |
| `test_oscillator_fit_none_unchanged` | `CommonOscillatorModel(...).fit(obs)` with default `robust_weight` matches a pre-change run's log-likelihood list exactly (bit-identical; the golden regression already pins this — this test exists for the fast suite with `max_iter=3`). |
| `test_oscillator_e_step_returns_objective_and_stores_weights` | With `robust_weight=imq_weight(3.0)`, `_e_step` returns `RobustOutput.objective` (equal to a direct `switching_kalman_filter` call) and `filter_robust_weights` has shape `(n_time,)`, `∈ (0, 1]`, and is `< 0.5` on the contaminated bins / `> 0.9` median elsewhere. |
| `test_lfp_artifact_state_rmse` (kinds COM, CNM, DIM; **slow** only if `fit` is used — use `_e_step` at true parameters, which is fast) | 5 % `20σ` contamination of `simulate_from_oscillator_model` output: robust smoothed latent RMSE vs truth `< 0.5 ×` standard and `< 1.5 ×` clean-data standard RMSE. |
| `test_em_recovers_measurement_variance_under_contamination` (**slow**) | `CorrelatedNoiseModel` at truth (`update_measurement_cov=True`, `n_discrete_states=1`, `skip_init=True`), `imq_weight(c=3, core=√χ²_d(0.99))`, T = 3000: fitted shared `measurement_cov` diagonal within 15 % of truth; standard EM's `> 3 ×`; robust objective history non-decreasing up to `decrease_tol` with ≥ 3 accepted iterations (guard); rollback warnings absent. |
| `test_fit_sgd_runs_with_robust_weight` (**slow**) | `DirectedInfluenceModel(robust_weight=imq_weight(3.0)).fit_sgd(contaminated, num_steps=30)` returns finite non-decreasing-on-average objective history and finite parameters; `filter_robust_weights` populated by `_finalize_sgd`. |
| `test_em_rollback_still_works_with_robust_weight` | `assert_em_rolls_back_on_ll_decrease` (conftest) on a robust model: the driver rolls back on a scripted objective decrease. |
| `test_repr_and_snapshot_include_robust_weight` | `repr(model)` mentions `robust_weight`; `_snapshot_em_state()` contains `filter_robust_weights` after an E-step; `_restore_em_state` restores it. |

## Fixtures

- Reuse `oscillator_model_at_truth` / `simulate_from_oscillator_model` (recovery_helpers), `switching_kalman_model_params` (conftest), the module-scoped parameter fixtures in `test_oscillator_models.py`.
- New module-scoped fixture `contaminated_oscillator_data(kind)` (parametrised helper in `test_robust_oscillator_models.py`): model at truth, `T = 1500`, `(x, y, s)` simulated with `np.random.default_rng(0)`, plus a contaminated `y` and mask.
- Real data: none checked in; the target is the notebooks' LFP sessions, which are out of the test suite. Add a `notebooks/` smoke run to the PR description (not a test): fit `DirectedInfluenceModel` with and without `robust_weight` on one session and report the fraction of bins with `w < 0.5`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
