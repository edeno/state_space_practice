# Phase 1b — Covariate-dependent transitions in the switching point-process family

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#j-sibling-integration-notes)

Mirrors phase 1a into `switching_point_process_filter`, `SwitchingPointProcessBase`
(EM + SGD), `SwitchingSpikeOscillatorModel`, `BaseSwitchingPointProcessModel`
(COM-PP / CNM-PP / DIM-PP) and `simulate_switching_spike_oscillator`, using the
primitive shipped in phase 1a. The smoother needs no work here: the point-process
E-step already calls the Gaussian `switching_kalman_smoother`, which accepts the
keywords since phase 1a. After this PR spike-observed switching models can have
speed-gated switching (the `docs/plans/multi-map-place-fields/` consumer).

**Inputs to read first:**

- [shared-contracts.md](shared-contracts.md) — especially the [sibling-mirroring checklist](shared-contracts.md#sibling-mirroring-checklist) (point-process column) and the [model attribute contract](shared-contracts.md#model-attribute-contract).
- Phase 1a's diff of `switching_kalman.py` (`_step` / `resolve_scan_transitions` usage) and of `oscillator_models.py` (`_transition_kwargs`, `_install_discrete_transition`, `_bind_transition_covariates`): the point-process versions must be behaviourally identical (memory rule `dim-pp-mirrors-gaussian-dim`).
- `src/state_space_practice/switching_point_process.py:286-370` (static shape validator), `:372-428` (host-side value validator; `:396-397` tracer skip), `:2042-2056` and `:2194-2321` (jitted core and its `_step`; discrete update `:2277-2292`; scan `:2355-2376`), `:2415-2470` (public wrapper), `:2473-2637` (base constructor; `:2588-2597` diag / `transition_prior`; `:2619-2636` placeholders), `:2741-2749`, `:2786-2836` (`_validate_parameter_shapes`), `:2889-3006` (`_e_step`; filter call `:2934-2947`; GPB1 call `:2989-2992`), `:3012-3135` (`_m_step_dynamics`; install `:3124-3125`), `:3313-3359` (`fit_sgd`), `:3379-3409` (`_shared_sgd_param_spec`), `:3411-3470` (`_sgd_loss_fn`; filter call `:3439-3453`; penalties `:3455-3468`), `:3472-3485` (`_store_sgd_params`), `:3610-3719` (`SwitchingSpikeOscillatorModel.__init__`), `:3841-4007` (`fit`; snapshot `:3966-3984`).
- `src/state_space_practice/point_process_models.py:173-229` (`BaseSwitchingPointProcessModel.__init__`; `transition_prior` `:218-229`), `:464-510` (`_snapshot_em_state` attrs `:474-497`), `:516-636` (`fit` / `_fit_single`), `:835-836` (COM-PP spec), `:1090-1157` (CNM-PP spec / loss / store), `:1322-1419` (DIM-PP `_m_step_dynamics` / `_m_step_reparameterized`; install `:1368-1369`), `:1423-1521` (DIM-PP SGD).
- `src/state_space_practice/simulate/simulate_switching_spikes.py:15-30` (signature), `:121` (`s_0`), `:142-166` (`_step`; categorical draw `:153-155`), `:170-175` (scan).
- `src/state_space_practice/tests/test_oracle_switching_point_process.py:77-100` (`_params_1d`), `:102-123` (`_simulate_spikes`), `:135-168` (`_laplace_path_log_weights`), `:171-188` (`laplace_path_oracle`), `:238-262` (`path_posterior`), `:285-312` (`model_posteriors`), `:353-376` and `:401-430` (identical-states / two-steps tests), `:531-561` (`_spike_oscillator_model`); `tests/test_simulate_switching_spikes.py:22-77`; `tests/test_point_process_models.py:28-119` (fixtures), `:1340` (`TestSwitchingPPSGDFitting`); `tests/test_switching_point_process.py:6233` (`TestSwitchingSpikeOscillatorModelFit`), `:9676` (SGD tests).

**Contracts referenced:**

- All of [shared-contracts.md](shared-contracts.md); do not weaken. `n_cont_states` for this family is `n_latent = 2 * n_oscillators`.

**Designs referenced:** [designs.md J](designs.md#j-sibling-integration-notes) (point-process notes), [A](designs.md#a-transition-stack-resolution-and-in-scan-step), [E](designs.md#e-logit-m-step) (install helper), [I](designs.md#i-simulators).

## Tasks

- **Jitted core and wrapper.** `_switching_point_process_filter_jit` and
  `switching_point_process_filter` gain keyword-only
  `transition_covariates=None, transition_weights=None`. In the core: call
  `resolve_scan_transitions(discrete_transition_matrix, transition_covariates,
  transition_weights, spikes.shape[0])` after the shape validation (`:2183-2192`),
  unpack `(y_t, transition_input)` in `_step` (`:2194-2200`), compute
  `transition_t = transition_at_step(transition_input, prev_state_cond_filter_mean)`
  and pass it to `_update_discrete_state_probabilities` (`:2284-2289`), scan over
  `(spikes[1:], per_step)` (`:2365-2376`). Static shape checks of the new inputs
  go into `_validate_switching_point_process_filter_shapes` (`:286-370`, next to
  `:319-324`); the wrapper calls `validate_transition_inputs(...)` next to
  `_validate_discrete_state_transitions` (`:2453-2455`) with
  `n_cont_states=init_state_cond_mean.shape[0]`. Docstrings (`:2081-2083` area and
  the wrapper's) gain the two parameters with the alignment rule.
- **`SwitchingPointProcessBase`.** Keyword-only `transition_regularization: float = 1e-5`
  in `__init__` (`:2514-2536`; the constructor is already keyword-only after
  `dt`), stored and validated (`>= 0`); attributes `self.transition_weights = None`,
  `self._transition_covariates = None` next to the placeholders (`:2619-2636`);
  the three helpers (`_transition_kwargs`, `_install_discrete_transition`,
  `_bind_transition_covariates` with `n_cont_states=self.n_latent`) copied from
  `BaseModel` with the point-process attribute names; `_e_step` forwards
  `**self._transition_kwargs()` to `switching_point_process_filter` (`:2934-2947`)
  and to the GPB1 `switching_kalman_smoother` call (`:2989-2992`), not to GPB2;
  `_m_step_dynamics` replaces `:3124-3125` with
  `self._install_discrete_transition(new_discrete_transition)`;
  `_validate_parameter_shapes` (`:2786-2836`) checks `transition_weights` when
  not `None`; `_shared_sgd_param_spec` (`:3395-3397`) adds
  `params["transition_weights"]` (`UNCONSTRAINED`) when
  `update_discrete_transition_matrix and self._transition_covariates is not None`;
  `_sgd_loss_fn` (`:3439-3453`) forwards `transition_covariates=self._transition_covariates,
  transition_weights=params.get("transition_weights", self.transition_weights)`
  when bound and adds `(spikes.shape[0] - 1) * self.transition_regularization * jnp.sum(gamma ** 2)`
  next to the existing penalties (`:3455-3468`); `_store_sgd_params` (`:3472-3485`)
  stores `"transition_weights"`; `fit_sgd` (`:3313-3359`) gains keyword-only
  `transition_covariates=None` and calls `_bind_transition_covariates` after
  `_initialize_parameters` / before `super().fit_sgd`.
- **`SwitchingSpikeOscillatorModel`.** `__init__` (`:3610-3719`) forwards the
  new `transition_regularization` keyword; `fit` (`:3841-3849`) gains keyword-only
  `transition_covariates=None`, binds it after `_initialize_parameters`
  (`:3960-3961`) and before `run_em`; `_snapshot_params` / `_restore_params`
  (`:3966-3984`) include `transition_weights` (copy when not `None`).
- **`BaseSwitchingPointProcessModel` (point_process_models.py).** `__init__`
  (`:173-216`) forwards `transition_regularization` (add the keyword with default
  `1e-5`; its docstring `:118-162` gains the entry); `fit` (`:516-524`) gains
  keyword-only `transition_covariates=None`, binds it after the shape checks and
  before `_fit_multi_restart` / `_fit_single` (both read the attribute; the
  restart loop re-initialises parameters, so `_bind` must run inside
  `_fit_single` after `_initialize_parameters` when `skip_init` is False — pass
  the covariates through to `_fit_single`); `_snapshot_em_state` attrs
  (`:474-497`) gain `"transition_weights"`; `DirectedInfluencePointProcessModel._m_step_reparameterized`
  replaces `:1368-1369` with `self._install_discrete_transition(new_discrete_transition)`.
  COM-PP (`:835-836`) and CNM-PP (`:1090-1101`) inherit the spec change through
  `_shared_sgd_param_spec`; DIM-PP (`:1423-1432`) likewise. If the point-process
  warm initialisation (`_warm_initialize_states`, `point_process_models.py`,
  after `:286`) restores `discrete_transition_matrix` after a seeded M-step the
  way `oscillator_models.py:794-800` does, add `transition_weights` to that
  restore list too.
- **Spike simulator** ([designs.md I](designs.md#i-simulators)).
  `simulate_switching_spike_oscillator(..., transition_covariates=None,
  transition_weights=None)`: Python-level branch so the fixed path draws
  `categorical(key, log Z[s_prev])` exactly as today (identical random stream);
  the covariate path scans over `transition_covariates[1:]` and draws from
  `centered_log_softmax(eta[s_prev] + u_t @ Gamma[:, s_prev, :])`. The
  docstring (`:28-96`) gains the arguments; `mypy` clean (module is in the
  checked list).
- **Point-process oracle.** Extend `_laplace_path_log_weights`
  (`tests/test_oracle_switching_point_process.py:135-168`) and
  `laplace_path_oracle` (`:171-188`) to accept a `(n_time, K, K)` stack in place
  of `Z` the same way `tests/oracles.py` does in phase 1a (per-step `log_Z[t, s[t-1], s[t]]`).
- **Tests** — the validation slice below, in `tests/test_switching_point_process.py`
  (filter identities, model EM / SGD), `tests/test_oracle_switching_point_process.py`
  (exactness), `tests/test_point_process_models.py` (structured models),
  `tests/test_simulate_switching_spikes.py` (simulator).
- **User-facing docs.** `CHANGELOG.md` `### Added`: extend the phase-1a entry
  with the point-process functions and models and the simulator argument.
  Docstrings as listed above (`switching_point_process_filter`, both `fit` /
  `fit_sgd`, `SwitchingSpikeOscillatorModel` class docstring parameter list
  `:3506-3558`, `BaseSwitchingPointProcessModel` docstring, simulator).

## Deliberately not in this phase

- The choice family — phase 1c.
- `transition_state_weights` — phase 2b (the point-process filter's in-scan
  rule already receives `prev_state_cond_filter_mean`, so 2b is a keyword and a
  `resolve_scan_transitions` argument, not a body change).
- Changing the smoother selection or GPB2 (`:2959-2977`): GPB2 takes no
  transition matrix.
- Touching `_validate_discrete_state_transitions`'s existing row-sum semantics:
  it keeps validating the *baseline* matrix.
- Any change to `hamiltonian_switching.py`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_switching_point_process.py::test_point_process_filter_fixed_path_bit_identical_with_none_transition_options` | `_params_1d()`-style 1-D, K=2 problem, `max_newton_iter in (1, 3)`: all 7 filter outputs are `assert_array_equal` with and without the keywords passed as `None`. |
| `test_switching_point_process.py::test_point_process_filter_zero_transition_weights_reproduce_fixed_matrix` | `Gamma = 0`, random covariates: 7 outputs match the fixed path to `rtol=atol=1e-12`. |
| `test_switching_point_process.py::test_point_process_one_hot_covariate_switches_at_its_own_time_step` | `_params_1d(identical=True)` (posterior = prior chain), one-hot covariate at `t* = 5`, logits `-12` toward the reference state: `filter_prob[t*, 1] > 0.999`, `filter_prob[t*-1, 1] < 0.05`; identical checks on the GPB1 smoothed probabilities from `switching_kalman_smoother(..., transition_covariates=u, transition_weights=Gamma)`. |
| `test_switching_point_process.py::test_point_process_filter_rejects_structural_zero_baseline_with_covariates` | `Z = I` + covariates → `ValueError` (wrapper); shapes wrong → `ValueError` from the static validator at trace time (`pytest.raises` around the jitted core call). |
| `test_switching_point_process.py::test_point_process_marginal_ll_gradient_finite_wrt_transition_weights` | `jax.grad` w.r.t. `Gamma` through `switching_point_process_filter` is finite and nonzero (the wrapper's validation is skipped under trace, `:396-397` pattern). |
| `test_oracle_switching_point_process.py::test_identical_states_posterior_equals_time_varying_prior_chain` | identical states + covariate-driven stack: filtered and GPB1/GPB2 smoothed probabilities equal the prior chain computed with the stack (`prior_chain_posteriors` `:264-273` generalised) to 1e-9; guard `ptp(stack[1:]) > 0.1`. |
| `test_oracle_switching_point_process.py::test_two_steps_filter_and_gpb2_match_per_path_laplace_with_time_varying_transitions` | T=2, distinct states, `max_newton_iter=1`: `log_lik`, `filtered`, `gpb2`, `gpb2_joint` match `laplace_path_oracle(params, spikes, stack=...)` to 1e-10 (the `:401-430` pattern). |
| `test_switching_point_process.py::test_spike_oscillator_e_step_forwards_transition_covariates` | `SwitchingSpikeOscillatorModel` with bound covariates and nonzero `transition_weights`: `_e_step` posteriors equal a direct `switching_point_process_filter` + `switching_kalman_smoother` call with the same keywords (1e-12); with covariates unbound, equal the fixed-path call. |
| `test_switching_point_process.py::test_spike_oscillator_m_step_dynamics_installs_logit_update_when_covariates_bound` | after one `_e_step`, `_m_step_dynamics` with covariates bound sets `discrete_transition_matrix` and `transition_weights` to `maximize_transition_coefficients(...)`'s outputs (1e-12) and *not* to the count-based matrix (guard: the two matrices differ by > 1e-3); with no covariates, the count-based matrix is installed as before. |
| `test_switching_point_process.py::test_spike_oscillator_fit_recovers_speed_gated_transition_weights` (slow) | `simulate_switching_spike_oscillator` with `Gamma = -1.5` on both rows, T=2000, 10 neurons, 3 seeds: `fit(spikes, transition_covariates=u, max_iter=30)` gives both weights negative and within `[0.75, 3.0]` after `find_permutation` alignment; `transition_weights.shape == (1, 2, 1)`. |
| `test_switching_point_process.py::test_spike_oscillator_fit_sgd_learns_transition_weight_sign` (slow) | same data, `fit_sgd(spikes, transition_covariates=u, num_steps=60)`: weights negative, LL improves; `"transition_weights"` in the spec only when bound. |
| `test_switching_point_process.py::test_spike_oscillator_rollback_restores_transition_weights` (slow) | `conftest.assert_em_rolls_back_on_ll_decrease(model, (spikes,), caplog, fit_kwargs={"transition_covariates": u})`: after rollback `transition_weights` equals the accepted iterate's value. |
| `test_point_process_models.py::test_dim_pp_reparameterized_m_step_installs_logit_update_when_covariates_bound` | DIM-PP with `use_reparameterized_mstep=True` and bound covariates: `_m_step_dynamics` updates `transition_weights` (changes by > 1e-6 from zero on informative `xi`) — the DIM-PP install site is not left on the count-based path (`dim-pp-mirrors-gaussian-dim`). |
| `test_point_process_models.py::test_com_pp_fit_with_transition_covariates_smoke` (slow) | COM-PP `fit(spikes, transition_covariates=u, max_iter=5, n_restarts=2)` runs, LL finite, `transition_weights` finite and snapshotted (`_snapshot_em_state()` contains the key). |
| `test_simulate_switching_spikes.py::test_simulate_fixed_path_unchanged_by_none_transition_options` | identical `(spikes, states, discrete)` with and without the `None` keywords for the same key. |
| `test_simulate_switching_spikes.py::test_simulate_speed_gated_switching_follows_covariate` | T=20000: state-1 occupancy during `u > 1` exceeds that during `u < -1` by > 0.2; < 0.05 with `Gamma = 0`. |

## Fixtures

- Reuse `tests/test_oracle_switching_point_process.py::_params_1d` (`:77-100`),
  `_simulate_spikes` (`:102-123`) and `_spike_oscillator_model` (`:531-561`); add
  `_covariate_stack(params, rng, n_time)` returning `u`, `Gamma`, `stack`.
- Reuse `tests/test_simulate_switching_spikes.py::switching_spike_params` (`:22`).
- The speed covariate generator from phase 1a (`standardize(cumsum(AR(1)))`)
  moved to `tests/recovery_helpers.py` as `simulate_speed_covariate(n_time, seed)`
  so 1a, 1b and 1c share it (1a may create it there directly if it lands first).
- `tests/test_point_process_models.py::com_pp_params` (`:28-48`) and
  `synthetic_spikes` (`:103-119`) for the structured-model smoke test.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): every install site in the point-process family (`switching_point_process.py:3124-3125`, `point_process_models.py:1368-1369`) routes through `_install_discrete_transition`.
- User-facing documentation listed as tasks is updated, not deferred.
- Diff the point-process helpers against phase 1a's `BaseModel` helpers line by line; they must be behaviourally identical (memory rule `dim-pp-mirrors-gaussian-dim`).
