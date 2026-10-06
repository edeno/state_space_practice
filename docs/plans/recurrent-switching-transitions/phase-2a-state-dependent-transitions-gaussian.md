# Phase 2a — State-dependent (recurrent) transitions: primitive and the Gaussian switching family

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle)

Adds `transition_state_weights=` — logits linear in the continuous latent state
`x_{t-1}`, the recurrent SLDS of Linderman et al. (2017) — to the primitive, the
Gaussian filter / GPB1 smoother / Viterbi, the logit M-step, `BaseModel` and the
Gaussian simulator. Under the GPB collapse the transition probability is an
expectation of a softmax over the Gaussian `x_{t-1} | S_{t-1} = i`; this phase
ships the documented plug-in at the collapsed mean, with approximation-trend
tests against deterministic quadrature and a 1-D grid oracle, and keeps the
smoother on the forward pass's transition matrices. The phase-1 machinery
carries the realised `(n_time, S, S)` stack, so no new keyword is added to
phase 1's API.

**Inputs to read first:**

- [shared-contracts.md](shared-contracts.md): phase-2 rows of the [keyword contract](shared-contracts.md#keyword-contract) and [model attribute contract](shared-contracts.md#model-attribute-contract); the `W` shape `(n_cont_states, S, S - 1)` and the contraction `einsum("di,dij->ij", m_prev, W)`.
- [designs.md G](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle) in full (approximation, IO-HMM equivalence, quadrature test, grid oracle), [A](designs.md#a-transition-stack-resolution-and-in-scan-step) (`state_dependent_logits`, the state branch of `resolve_scan_transitions`, the `state_cond_means` argument of `transition_matrix_stack`), [E](designs.md#e-logit-m-step) (the `transition_state_weights` / `state_cond_smoother_means` branch), [H](designs.md#h-relabelling-transform), [I](designs.md#i-simulators).
- Phase 1a's diff: `discrete_transitions.py`, `switching_kalman.py` (`_switching_kalman_filter_jit`, `switching_kalman_smoother`, `switching_kalman_viterbi`), `oscillator_models.py` helpers.
- `src/state_space_practice/switching_kalman.py:49-57` (the pair update is vmapped over the previous state `i` on axis `-1` of `prev_state_cond_filter_mean`, i.e. column `i` is `m^i_{t-1|t-1}` — the quantity the plug-in uses), `kalman.py:479-526` (`_kalman_filter_update` predicts from `mean_prev`, confirming the same collapsed mean drives the continuous prediction).
- `src/state_space_practice/tests/test_approximation_trends.py:1-131` (structure, `_assert_decreasing`, `_gpb_errors`), `tests/test_oracle_switching_kalman.py:73-161`, `tests/test_oracle_point_process.py` (`_grid_posterior`, grid-choice pattern used at `tests/test_approximation_trends.py:88-91`), `tests/test_switching_kalman.py:3941` (`run_em` test helper: a hand-rolled EM loop over `switching_kalman_filter` / smoother / `switching_kalman_maximization_step` to extend with the logit M-step), `tests/test_invariances.py:332-345`.
- Literature: Linderman, Johnson, Miller, Adams, Blei & Paninski (2017), AISTATS, "Bayesian learning and inference in recurrent switching linear dynamical systems" — the rSLDS model form `logit_t = W x_{t-1} + r` (phase-2 parametrisation); Zoltowski, Pillow & Linderman (2020), ICML, "A general recurrent state space framework for modeling neural dynamics during decision-making" — Laplace-EM for rSLDS with the discrete variational factor's transition potentials held fixed from the forward pass (the structural choice the smoother makes here); Murphy (1998), "Switching Kalman filters" — the GPB collapse whose collapsed mean is the plug-in point; Glaser, Whiteway, Cunningham, Paninski & Linderman (2020), NeurIPS, "Recurrent switching dynamical systems models for multiple interacting neural populations" — recurrent transitions shared across observation models (why the primitive is family-agnostic).

**Contracts referenced:**

- [Parametrisation and layout](shared-contracts.md#parametrisation-and-layout-contract) (`W` and its contraction), [keywords](shared-contracts.md#keyword-contract) (`transition_state_weights` may be given alone; model-level `recurrent_transitions: bool = False`), [None-path invariant](shared-contracts.md#none-path-invariant), [M-step contract](shared-contracts.md#m-step-contract) (design row `[1, u_{t+1}, m^i_{t|T}]`), [validation contract](shared-contracts.md#validation-contract).

**Designs referenced:** [designs.md G](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle) (primary), [A](designs.md#a-transition-stack-resolution-and-in-scan-step), [D](designs.md#d-viterbi-integration), [E](designs.md#e-logit-m-step), [H](designs.md#h-relabelling-transform), [I](designs.md#i-simulators).

## Tasks

- **Primitive.** Add `state_dependent_logits`; the `transition_state_weights`
  and `state_cond_means` arguments of `transition_matrix_stack` (entry `t` uses
  `state_cond_means[t - 1]`, entry 0 the baseline); the state branch of
  `resolve_scan_transitions` (returns `logits[1:]` and a rule that applies
  `centered_softmax(state_dependent_logits(base_t, W, prev_mean))`);
  `validate_transition_inputs`'s `transition_state_weights` shape / finiteness
  checks; the `transition_state_weights` / `state_cond_smoother_means` branch of
  `maximize_transition_coefficients` (per-row design rows
  `[1, u_{t+1}, m^i_{t|T}]`, `n_coef = 1 + n_features + n_cont_states`). Module
  docstring: the rSLDS form, the plug-in statement and its `O(||W||^2 tr P)`
  gap, and that the correction is a documented follow-up.
- **Gaussian filter / smoother / Viterbi.** Keyword-only
  `transition_state_weights=None` on `switching_kalman_filter`,
  `_switching_kalman_filter_jit`, `switching_kalman_smoother` and
  `switching_kalman_viterbi`; the wrapper validates it; the filter and Viterbi
  forward scans pass it to `resolve_scan_transitions` (no other body change);
  the smoother passes `transition_state_weights=..., state_cond_means=filter_mean`
  to `transition_matrix_stack`; the Viterbi forward scan additionally stacks
  `state_cond_mean` so the backward pass can build the stack from
  `jnp.concatenate([first_state_cond_mean[None], means_rest])`
  ([designs.md D](designs.md#d-viterbi-integration)). Docstrings: the keyword,
  the plug-in approximation (one paragraph, naming the collapsed mean
  `E[x_{t-1} | S_{t-1} = i, y_{1:t-1}]`), and for the smoother the sentence
  "uses the transition matrices the filter used, rebuilt from `filter_mean`".
- **`BaseModel`.** Keyword-only `recurrent_transitions: bool = False` on
  `__init__` (`oscillator_models.py:440-456`); when True,
  `_initialize_parameters` (`:868-884`) zero-initialises
  `self.transition_state_weights = jnp.zeros((n_cont_states, S, S - 1))`
  (else `None`); `_transition_kwargs` adds `transition_state_weights` when not
  `None`; `_install_discrete_transition` passes
  `transition_state_weights=self.transition_state_weights,
  state_cond_smoother_means=self.smoother_state_cond_mean` and stores the third
  output; `_sgd_param_attrs` (`:404-413`), `_EM_SNAPSHOT_KEYS` (`:415-438`), the
  warm-init restore tuple (`:794-800`) and `_validate_parameter_shapes`
  (`:288-299`) learn `transition_state_weights`; COM / CNM / DIM
  `_build_param_spec` add `"transition_state_weights"` (`UNCONSTRAINED`) when
  `update_discrete_transition_matrix and self.transition_state_weights is not None`;
  `_sgd_loss_fn` forwards `transition_state_weights=params.get(...)` and adds
  `(n_time - 1) * self.transition_regularization * jnp.sum(W ** 2)`.
  `recurrent_transitions=True` together with `update_discrete_transition_matrix=False`
  is allowed (a fixed, user-supplied `W` is legitimate): the flag only decides
  whether `W` is allocated; the update flag decides whether EM / SGD learn it.
- **Gaussian simulator.** `simulate(..., transition_state_weights=None)` per
  [designs.md I](designs.md#i-simulators) (`x[t - 1] @ W[:, s_prev, :]` added to
  the row logits); the fixed path is unchanged.
- **Grid oracle.** New `tests/test_oracle_recurrent_switching.py` with
  `grid_recurrent_posterior` ([designs.md G](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle)),
  its self-check against `switching_lgssm_exact_posterior` when `W = 0`, and the
  filter-level trend test; the primitive-level quadrature trend test and the
  IO-HMM fixed-point test go to `tests/test_approximation_trends.py` and
  `tests/test_switching_kalman.py` respectively.
- **Tests** — validation slice below.
- **User-facing docs.** `CHANGELOG.md` `### Added`: "recurrent (state-dependent)
  discrete transitions" naming `transition_state_weights`, `recurrent_transitions`,
  the plug-in approximation and the follow-up; add a line under
  `### Known approximation limits (documented and pinned by tests)` (CHANGELOG
  `:462`) with the trend test's observed gaps. `README.md` package-layout
  sentence extended with "and recurrent (rSLDS-style) switching". Docstrings as
  listed.

## Deliberately not in this phase

- The point-process and choice families — phase 2b.
- Any exact / probit / unscented expectation of the row softmax — recorded
  follow-up (overview Open Question 2); the swap point is the state branch of
  `resolve_scan_transitions`.
- Refitting the smoother's transition matrices from smoothed means (a
  different, iterative approximation); the smoother uses the forward pass's
  matrices by design.
- A `simulate/scenarios.py` rSLDS scenario (the test builds its own 1-D model).
- Changing `switching_kalman_smoother_gpb2` or the M-step inner.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_discrete_transitions.py::test_state_dependent_logits_matches_row_wise_contraction` | for random `W`, `m`, `base`: `state_dependent_logits(base, W, m)[i]` equals `base[i] + m[:, i] @ W[:, i, :]` for every row `i` (1e-12). |
| `test_discrete_transitions.py::test_transition_matrix_stack_uses_previous_time_means` | `state_cond_means` one-hot in time at `t0`: only `stack[t0 + 1]` differs from the covariate-only stack (guard: it differs by > 1e-3); `stack[0]` is the baseline. |
| `test_discrete_transitions.py::test_validate_rejects_bad_state_weight_shapes` | `W` of shape `(n_cont + 1, S, S - 1)` or with a NaN → `ValueError`; `W` alone (no covariates) is accepted. |
| `test_switching_kalman.py::test_switching_filter_fixed_path_bit_identical_with_none_transition_options` | extended: also passes `transition_state_weights=None` explicitly; outputs `assert_array_equal`. |
| `test_switching_kalman.py::test_zero_state_weights_reproduce_fixed_matrix` | `W = 0` (with and without covariates): filter / GPB1 / Viterbi equal the phase-1 or fixed path to 1e-12. |
| `test_switching_kalman.py::test_state_dependent_filter_equals_io_hmm_with_collapsed_mean_covariates` | the fixed-point construction of [designs.md G](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle): all 7 outputs of the `transition_state_weights=W` call equal the `transition_covariates=vec(m_{t-1}), transition_weights=Gamma(W)` call to 1e-12; guard: the two differ from the fixed path by > 1e-3 in discrete probability. |
| `test_switching_kalman.py::test_smoother_uses_filter_transition_matrices_under_state_weights` | `switching_kalman_smoother(..., transition_state_weights=W)` equals `switching_kalman_smoother(..., transition_covariates=vec(m_{t-1}), transition_weights=Gamma(W))` to 1e-12; and the last smoothed discrete marginal equals the last filtered one (`assert_array_equal`). |
| `test_switching_kalman.py::test_viterbi_state_dependent_matches_brute_force_on_realised_stack` | identical-states K=2, T=6: Viterbi path equals brute-force MAP over paths with the realised stack (built from the filter's `state_cond_filter_mean`); guard: differs from the fixed-`Z` MAP path. |
| `test_switching_kalman.py::test_filter_marginal_ll_gradient_finite_wrt_state_weights` | `jax.grad` w.r.t. `W` finite and nonzero. |
| `test_oracle_switching_kalman.py::test_logit_m_step_with_state_weights_is_stationary_for_exact_statistics` | exact `xi` and exact state-conditional smoothed means from `_oracle`; `l2 = 0`: FD gradient of `sum_t sum_ij xi log softmax(eta + m_smoothed[t, :, i] @ W[:, i, :])` at the returned `(eta, W)` `< 1e-6` × scale; objective non-decreasing from zeros; guard `max(abs(W_new)) > 0.1`. |
| `test_approximation_trends.py::test_recurrent_transition_plugin_gap_shrinks_with_latent_variance` | Gauss-Hermite (200 nodes) row expectation vs plug-in, `var in (1, .3, .1, .03, .01)`, `W_row = (2, -1.5)`: gaps strictly decreasing (`_assert_decreasing`), `gaps[0] > 1e-2`, `gaps[0] > 20 * gaps[-1]`. |
| `test_oracle_recurrent_switching.py::test_grid_oracle_reproduces_path_enumeration_without_state_dependence` | `transition_fn = lambda x: Z`, K=2, T=6, `G = 1201`: filtered / smoothed / joint probabilities and `log_lik` match `switching_lgssm_exact_posterior` to 1e-6. |
| `test_oracle_recurrent_switching.py::test_filter_gap_to_grid_oracle_shrinks_as_latent_covariance_shrinks` (slow) | scale `P0`, `Q` by `c in (1, .3, .1, .03, .01)` (fixed `R`, `W`, data): RMS filtered-probability gap and `abs(log_lik)` gap vs the grid oracle strictly decreasing; loosest gap `> 1e-2`; GPB1 smoothed-probability gap also decreasing. Observed values recorded in the test docstring. |
| `test_invariances.py::TestSwitchingKalmanInvariances::test_discrete_state_relabelling_with_state_dependent_transitions` (slow) | `_switching(3, K=3)` + `W`; permuted parameters via `permute_transition_params`; the `:338-345` assertions; guard `ptp(prob) > 0.1`. |
| `test_switching_kalman.py::test_em_recovers_recurrent_transition_weight_sign` (slow) | 1-D latent, K=2 (`A = (0.95, 0.5)`, `Q = (0.05, 0.05)`, `H = 1`, `R = 0.01`), `W = (-2, -2)` (large `x` pushes toward state 1), T=3000, 3 seeds; hand-rolled EM (extend the `:3941` helper with `maximize_transition_coefficients`) from `W = 0`: both learned `W` entries negative with magnitude in `[1.0, 4.0]` after `find_permutation`; LL non-decreasing beyond `tol`. |
| `test_oscillator_models.py::test_com_recurrent_transitions_flag_wires_em_and_sgd` (slow) | `CommonOscillatorModel(..., recurrent_transitions=True)`: `transition_state_weights.shape == (n_cont_states, S, S - 1)` after init; `fit` changes it (norm `> 1e-6`) and snapshots it; `_build_param_spec()` contains `"transition_state_weights"`; `fit_sgd(num_steps=20)` runs with finite LL. `recurrent_transitions=False` leaves it `None` and the spec unchanged. |
| `test_switching_kalman.py::test_simulate_state_gated_switching_follows_latent` | `simulate` with `W = (-2, -2)`, T=20000: state-1 occupancy when `x_{t-1} > 1` exceeds that when `x_{t-1} < -1` by > 0.2; `< 0.05` with `W = 0`; the fixed path unchanged (`assert_array_equal` against the `None` call). |

## Fixtures

- `tests/test_oracle_recurrent_switching.py`: module fixture `recurrent_1d_model`
  (seed 0): `K = 2`, `T = 6`, `A = (0.9, 0.5)`, `Q = (0.3, 0.3)`, `H = 1`,
  `R = 0.2`, `m0 = (0.5, -0.5)`, `P0 = 1`, `Z = [[0.8, 0.2], [0.3, 0.7]]`,
  `W = (1.5, -1.0)`; data simulated with the extended `simulate`; grid `G = 1201`
  over `[-8, 8]` (rebuilt from `±8 sd` of the path posteriors as the
  scale changes). `transition_fn(x) = centered_softmax(eta + x * W)` in NumPy.
- Reuse `_random_switching_model`, `_oracle`, `_run_library`, `simple_skf_model`,
  `_switching` / `_run_switching`, `permute_transition_params` (phase 1a),
  `common_oscillator_params`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): there is one in-scan transition rule (`resolve_scan_transitions`), used by the filter and the Viterbi forward pass; the smoother has no second implementation of the logits.
- User-facing documentation listed as tasks is updated, not deferred; the approximation and its follow-up are stated in the filter docstring and the CHANGELOG, not only in this plan.
- The IO-HMM fixed-point test and the grid-oracle self-check both pass (the two independent checks on the plug-in wiring).
