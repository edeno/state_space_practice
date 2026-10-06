# Phase 2b — State-dependent transitions in the point-process and choice families

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#j-sibling-integration-notes)

Mirrors phase 2a's `transition_state_weights=` into `switching_point_process_filter`,
`SwitchingPointProcessBase` and its models, `switching_choice_filter` /
`switching_choice_smoother` / `SwitchingChoiceModel`, and the spike and choice
simulators. The in-scan rules of both filters already receive the previous
collapsed mean since phases 1b/1c, so the filter work is a keyword and an
argument; the substantive work is the model wiring, the choice simulator's
joint scan, and the per-family fixed-point tests.

**Inputs to read first:**

- [shared-contracts.md](shared-contracts.md): phase-2 rows of the [keyword](shared-contracts.md#keyword-contract) and [model attribute](shared-contracts.md#model-attribute-contract) contracts; the [sibling-mirroring checklist](shared-contracts.md#sibling-mirroring-checklist).
- Phase 2a's diff (`discrete_transitions.py`, `switching_kalman.py`, `oscillator_models.py`) — the reference behaviour.
- Phases 1b and 1c's diffs (the point-process and choice helpers to extend).
- `src/state_space_practice/switching_point_process.py:2042-2056`, `:2194-2200`, `:2284-2289`, `:2415-2429`, `:2453-2470`, `:2514-2536`, `:2700-2733` (`_initialize_parameters`), `:2786-2836`, `:3379-3409`, `:3411-3470`, `:3472-3485`, `:3610-3719`, `:3966-3984`; `point_process_models.py:173-229`, `:474-497`, `:1332-1374`.
- `src/state_space_practice/switching_choice.py:206-271`, `:432-490`, `:544-633`, `:711-793`, `:799-822`, `:912-931`, `:1013-1034`, `:1036-1118`, `:1179-1238`, `:1260-1359` (simulator; separate state and value scans at `:1325-1341`).
- `src/state_space_practice/simulate/simulate_switching_spikes.py:142-175`.
- [designs.md G](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle) (IO-HMM fixed-point construction, reused per family), [I](designs.md#i-simulators) (joint scan for the choice simulator), [J](designs.md#j-sibling-integration-notes).

**Contracts referenced:**

- All of [shared-contracts.md](shared-contracts.md). `n_cont_states` is `n_latent` (point process) and `n_options - 1` (choice).

**Designs referenced:** [designs.md G](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle), [I](designs.md#i-simulators), [J](designs.md#j-sibling-integration-notes).

## Tasks

- **Point-process filter.** Keyword-only `transition_state_weights=None` on the
  jitted core and the wrapper; passed to `resolve_scan_transitions`; static
  shape check in `_validate_switching_point_process_filter_shapes`; value check
  in the wrapper via `validate_transition_inputs`. Docstring paragraph on the
  plug-in (mirror the Gaussian wording).
- **Point-process models.** `SwitchingPointProcessBase.__init__` gains
  keyword-only `recurrent_transitions: bool = False`; `_initialize_parameters`
  (`:2700-2733`) zero-initialises `transition_state_weights` `(n_latent, S, S - 1)`
  when True (else `None`); `_transition_kwargs` / `_install_discrete_transition`
  extended exactly as `BaseModel`'s (pass `state_cond_smoother_means=self.smoother_state_cond_mean`);
  `_validate_parameter_shapes`, `_shared_sgd_param_spec`, `_sgd_loss_fn` (forward
  + penalty `(n_time - 1) * l2 * sum(W ** 2)`), `_store_sgd_params`; the SSOM
  `_snapshot_params` / `_restore_params` (`:3966-3984`) and
  `BaseSwitchingPointProcessModel._snapshot_em_state` (`:474-497`) include
  `transition_state_weights`; `SwitchingSpikeOscillatorModel.__init__` and
  `BaseSwitchingPointProcessModel.__init__` forward the flag (docstrings).
- **Choice filter / smoother.** Keyword-only `transition_state_weights=None` on
  `switching_choice_filter`, `_switching_choice_filter_jit` (to
  `resolve_scan_transitions`; the wrapper validates with `n_cont_states = n_options - 1`)
  and `switching_choice_smoother` (`transition_matrix_stack(..., state_cond_means=filtered_values)`).
- **`SwitchingChoiceModel`.** Keyword-only `recurrent_transitions: bool = False`;
  `transition_state_weights_` zero-initialised `(n_options - 1, S, S - 1)` when
  True; helpers, spec / loss / store, `_populate_uncertainty` (the stack now
  needs `state_cond_means=result.filtered_values`, [designs.md J](designs.md#j-sibling-integration-notes)).
- **Spike simulator.** `transition_state_weights=None` per [designs.md I](designs.md#i-simulators)
  (`x_prev @ W[:, s_prev, :]` in the scan; fixed path unchanged).
- **Choice simulator.** Merge the state and value scans (`switching_choice.py:1325-1341`)
  into one scan over `(state_keys[1:], value_keys[1:], u[1:])` whose step draws
  `s_t` from logits built with `x_{t-1}` (carry) and then `x_t`; consume the
  pre-split keys so the fixed path reproduces today's draws bit-for-bit
  (baseline capture below). Docstring gains the argument.
- **Tests** — validation slice below.
- **User-facing docs.** `CHANGELOG.md` `### Added`: extend the phase-2a entry
  with the two families and simulators; `### Known approximation limits`: note
  the plug-in applies to all three families. Docstrings as listed.

## Deliberately not in this phase

- Any exact-expectation correction (overview Open Question 2).
- New behaviour in the Gaussian family (phase 2a is complete and is the
  reference).
- Changing the choice filter's x₀ convention or its dynamics covariates.
- Learning `W` for `hamiltonian_switching.py`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_switching_point_process.py::test_point_process_filter_fixed_path_bit_identical_with_none_transition_options` | extended with `transition_state_weights=None`; `assert_array_equal`. |
| `test_switching_point_process.py::test_point_process_state_dependent_filter_equals_io_hmm_with_collapsed_mean_covariates` | the [designs.md G](designs.md#g-phase-2-collapse-approximation-and-exact-grid-oracle) fixed-point construction on the spike filter (`_params_1d` model, `max_newton_iter in (1, 3)`): all 7 outputs to 1e-12; guard: differs from the fixed path by > 1e-3. |
| `test_switching_point_process.py::test_point_process_smoother_uses_filter_transition_matrices_under_state_weights` | GPB1 smoother with `transition_state_weights=W` equals the IO-HMM-covariate call to 1e-12; last smoothed marginal equals last filtered. |
| `test_switching_point_process.py::test_point_process_marginal_ll_gradient_finite_wrt_state_weights` | `jax.grad` w.r.t. `W` finite and nonzero. |
| `test_switching_point_process.py::test_spike_oscillator_recurrent_transitions_flag_wires_em_and_sgd` (slow) | `SwitchingSpikeOscillatorModel(..., recurrent_transitions=True)`: shape `(n_latent, S, S - 1)`; `fit` changes it and snapshots it; spec contains the key; `fit_sgd(num_steps=20)` finite; `False` leaves it `None`. |
| `test_switching_point_process.py::test_spike_em_recovers_recurrent_transition_weight_sign` (slow) | `simulate_switching_spike_oscillator` with `W` pushing toward state 1 for large first-coordinate `x`, 1 oscillator, 12 neurons, T=2500, 3 seeds: learned `W[0, :, 0]` entries have the true sign after `find_permutation` (magnitude band `[0.5, 4.0]`). |
| `test_point_process_models.py::test_dim_pp_reparameterized_m_step_updates_state_weights` | DIM-PP `recurrent_transitions=True`, `use_reparameterized_mstep=True`: `_m_step_dynamics` changes `transition_state_weights` (norm > 1e-6) — the DIM-PP install site is on the logit path (`dim-pp-mirrors-gaussian-dim`). |
| `test_switching_choice.py::TestSwitchingChoiceFilter::test_fixed_path_bit_identical_with_none_transition_options` | extended with `transition_state_weights=None`. |
| `test_switching_choice.py::TestSwitchingChoiceFilter::test_state_dependent_filter_equals_io_hmm_with_collapsed_mean_covariates` | fixed-point construction on the choice filter (`u_t = vec(filtered_values[t-1])`, `n_cont = n_options - 1`): all fields to 1e-12; guard as above. |
| `test_switching_choice.py::TestSwitchingChoiceSmootherControlInput::test_smoother_with_state_weights_matches_library_smoother` | `control_input = 0`: `switching_choice_smoother(..., transition_state_weights=W)` equals `switching_kalman_smoother(..., transition_state_weights=W)` to 1e-12. |
| `test_switching_choice.py::TestSwitchingChoiceUncertainty::test_predicted_discrete_prior_uses_state_dependent_stack` | after `fit` with `recurrent_transitions=True`: `predicted_option_variances_` matches a recomputation with `stack = transition_matrix_stack(Z, W=..., state_cond_means=filtered_values)` (1e-10) and differs from the fixed-matrix recomputation (guard). |
| `test_switching_choice.py::TestSwitchingChoiceRecovery::test_em_recovers_recurrent_transition_weight_sign` (slow) | `simulate_switching_choice_data(n_trials=1500, transition_state_weights=W)` with `W` pushing to the explore state when the first option's value is high, 3 seeds: learned sign correct after alignment. |
| `test_switching_choice.py::TestSimulateSwitchingChoiceData::test_joint_scan_reproduces_captured_fixed_path_draws` | `simulate_switching_choice_data(n_trials=20, seed=42)` equals the pre-change captured `choices`, `true_values`, `true_states`, `true_probs` (checked-in constants, `assert_array_equal` for integers, 1e-12 for floats). |
| `test_switching_choice.py::TestSimulateSwitchingChoiceData::test_state_gated_switching_follows_latent` | 20000 trials: state-1 occupancy when `x_{t-1, 0} > 1` exceeds that when `< -1` by > 0.2; `< 0.05` with `W = 0`. |
| `test_simulate_switching_spikes.py::test_state_gated_switching_follows_latent` | same statistic for the spike simulator; fixed path `assert_array_equal` against the `None` call. |

## Fixtures

- **Baseline capture (before touching the choice simulator):** run
  `simulate_switching_choice_data(n_trials=20, seed=42)` on the pre-change
  code and paste the four arrays into
  `tests/test_switching_choice.py` as module constants (`_CAPTURED_SIM_SEED42`),
  with a comment stating the commit they were captured at. This is the
  regression guard for the joint-scan refactor (the verification idiom for a
  behaviour-preserving refactor: capture, change, compare).
- Reuse `_params_1d`, `_spike_oscillator_model`, `switching_spike_params`,
  `fitted_switching_choice_model`, `TestSwitchingChoiceMStepExactness.em_inputs`,
  `com_pp_params` / `synthetic_spikes`, and phase 2a's `permute_transition_params`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the choice simulator has one scan, not a state scan plus a value scan kept alongside a joint one.
- User-facing documentation listed as tasks is updated, not deferred.
- Diff the point-process and choice helpers against phase 2a's `BaseModel` helpers; identical behaviour apart from attribute names (`dim-pp-mirrors-gaussian-dim`, `fix-sibling-implementations`).
