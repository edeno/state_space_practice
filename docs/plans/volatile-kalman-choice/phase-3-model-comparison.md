# Phase 3 — Model comparison: VKF vs fixed process noise vs switching regimes

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#comparison-harness)

Ship a harness that fits the three competing explanations of learning-rate
changes — `VolatileKalmanChoiceModel`, fixed-`q` `CovariateChoiceModel` and
`SwitchingChoiceModel` — to the same `(choices, rewards)` sequence and scores
them by BIC and held-out log-likelihood; simulate from each generative agent
and show the generating model is recovered (confusion matrix, slow test);
and a script that runs the same comparison on real bandit data through the
gitignored loaders.

**Inputs to read first:**

- [designs.md §9-10](designs.md#comparison-harness) — candidates, scoring, `count_free_parameters`, `sequence_log_likelihood`, the three generative agents.
- `src/state_space_practice/volatile_kalman.py` (phase 1): `VolatileKalmanChoiceModel.predictive_log_likelihood`, `simulate_volatile_bandit_data`, `volatile_kalman_step`.
- `src/state_space_practice/covariate_choice.py`: `signed_reward_covariates` (phase 2), `covariate_choice_filter` (`:210-264` signature), `CovariateChoiceModel.fit_sgd` (`:805-843`), `n_free_params` (`:938-951`).
- `src/state_space_practice/switching_choice.py:206-249` (`switching_choice_filter` signature and its both-or-neither rule for `covariates` / `input_gain`, `:331-343`), `:687-794` (`SwitchingChoiceModel.__init__` attributes: `process_noises_`, `inverse_temperatures_`, `decays_`, `discrete_transition_matrix_`, `init_mean_`, `init_cov_`, `input_gain_`), `:1122-1173` (`fit_sgd`), `:1179-1203` (`_build_param_spec`, the source for the parameter count).
- `src/state_space_practice/parameter_transforms.py:288-306` (`transform_to_unconstrained`, used to count free coordinates).
- `src/state_space_practice/tests/test_switching_choice.py:718-748` (`TestModelComparison`: the existing switching-vs-fixed comparison this phase generalises).
- `scripts/position_decoding_demo.py:1-45` — header conventions for a real-data script (x64 before imports, `PROJECT_ROOT`, `from data.load_bandit_data import ...`, run with `PYTHONPATH=.`); `README.md:36-39` (the gitignored `data/` note).
- `pyproject.toml:117` (scripts are formatted but not linted), `:134-159` (`[tool.mypy] files`).

**Contracts referenced:** none.

**Designs referenced:** [designs.md#comparison-harness](designs.md#comparison-harness), [#comparison-simulators](designs.md#comparison-simulators).

## Tasks

- **Create `src/state_space_practice/choice_model_comparison.py`** with `CandidateName`, `CANDIDATES`, `FittedCandidate`, `count_free_parameters`, `sequence_log_likelihood`, `fit_candidate`, `compare_candidates` as in [designs.md §9](designs.md#comparison-harness). Candidate construction uses only public constructors / `fit_sgd`; scoring uses only public filter functions and fitted public attributes. Held-out LL is `sequence_log_likelihood(full) − sequence_log_likelihood(prefix)` with the model fitted on the prefix; document why (causal filters; held-out trials keep their true history).
- **Add the generative agents** `SimulatedBanditData` / `simulate_bandit_agent` ([designs.md §10](designs.md#comparison-simulators)) to the same module: one `lax.scan`, `jax.random`, block-wise reward probabilities shared with `simulate_volatile_bandit_data`; the `"volatile"` branch delegates to it.
- **Add a `confusion_matrix(generators, n_seeds, n_trials, ...) -> np.ndarray` helper** returning, for BIC and for held-out LL, the `(n_generators, n_candidates)` count of wins; used by the slow test and the script.
- **Tests — `tests/test_choice_model_comparison.py`** (see validation slice).
- **Script `scripts/bandit_choice_model_comparison.py`**: `--simulate {volatile,fixed_q,switching}` runs the harness on simulated data (so the script is exercisable without the loader); without it, loads real trials via `data.load_bandit_data.load_bandit_trials(data_dir, session) -> {"choices", "rewards", "epoch_boundaries"}` (see overview Open Questions: this accessor is unverified — fail with a clear message naming the expected keys if the loader lacks it). Prints a table (candidate, `n_free_params`, training LL, BIC, held-out LL, fitted `λ`, `v0`) and saves a figure (VKF `predicted_option_values_` and `volatility_` with epoch boundaries as vertical lines; `learning_rate_` of the chosen option) under `output/` (gitignored). Header mirrors `scripts/position_decoding_demo.py:1-45`.
- **Export and type-check**: add `"src/state_space_practice/choice_model_comparison.py"` to `[tool.mypy] files`. No lazy top-level export (the harness is a submodule utility, like `behavioral_uncertainty`).
- **User-facing docs**: CHANGELOG `### Added` bullet — "**`choice_model_comparison`**: fit `VolatileKalmanChoiceModel`, fixed-process-noise `CovariateChoiceModel` and `SwitchingChoiceModel` to one bandit sequence and score them by BIC and held-out log-likelihood; `simulate_bandit_agent` for the three generative agents; `scripts/bandit_choice_model_comparison.py` runs it on simulated or real bandit data." Add the script to the README sentence about `scripts/` (`README.md:36-39`) only if that sentence lists scripts; otherwise leave README unchanged.

## Deliberately not in this phase

- **Adding `CovariateChoiceModel(dynamics="volatile")` as a candidate**: follow-up once phase 2 has merged *and* its identifiability gate passes (trigger recorded in phase 2).
- **Multi-session fitting** of real data: one session per run; pooling sessions needs `docs/plans/masks-and-multi-sequence/` (overview Dependency policy).
- **Hyperparameter search over `num_steps` / optimizers**: fixed defaults; the confusion test pins observed margins rather than tuning.
- **Statistical model-comparison refinements** (cross-validated folds, WAIC, bootstrap CIs): the prefix hold-out is what this phase ships.
- **Writing or changing the gitignored loader**: the script consumes a documented accessor; the loader itself is outside the repository.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_count_free_parameters_matches_n_free_params` | For `VolatileKalmanChoiceModel` and `CovariateChoiceModel(n_covariates=3, learn_decay=True)` the unconstrained-coordinate count equals `n_free_params`; for `SwitchingChoiceModel(S=2, K=3, n_covariates=3)` it equals `2+2+2+2+2+6 = 16` (per-state `q`, `β`, `decay`, `Z` with `S(S−1)`, `init_mean` `(K−1)`, `B` `(K−1)·K`). |
| `test_sequence_log_likelihood_is_causal_for_every_candidate` | For each candidate at its initial parameters: `heldout(n_train=T) == 0`, `heldout(n_train=0) == LL(full)`, and `LL(prefix n)` equals the `n`-trial prefix of a second run's partial sum computed by rerunning on `choices[:n]` (identity check at `1e-8`). |
| `test_simulated_agents_have_block_structure` | For each agent: `rewards ∈ {0, 1}`; within each block, the empirical reward rate of every option pulled ≥ 50 times is within 0.1 of `reward_probs`; `true_probs` rows sum to 1; `change_points == arange(block_length, T, block_length)`; same seed → identical data. |
| `test_fixed_q_agent_learns_and_switching_agent_alternates` | Fixed-`q` agent picks the block's best option on > 50% of each block's last 50 trials; switching agent's `true_probs` entropy has a bimodal spread (max−min per-trial entropy > 0.5 nats), guard that the two regimes are visible. |
| `test_compare_candidates_smoke` (slow) | `T=60`, `num_steps=3`: three `FittedCandidate`s with finite `bic` and `heldout_log_likelihood`; `n_free_params` per designs.md table. |
| `test_generating_model_is_recovered` (slow, parametrised over the three generators) | 3 seeds, `T=400`, `train_fraction=0.75`, `num_steps=150`: the generating model wins held-out LL in ≥ 2 of 3 seeds **and** BIC in ≥ 2 of 3 seeds; record the observed win counts and median margins in the docstring after the first run. If a row fails only on BIC, report the margin and revisit the candidate defaults (designs.md §9) before loosening the test. |
| script smoke (manual, in PR description) | `uv run --no-sync python scripts/bandit_choice_model_comparison.py --simulate volatile` prints the table and writes the figure; runtime and the winning candidate recorded. |

Long-running note (smoke first): time one `compare_candidates` call on `T=400`
before running the confusion test; expected order of magnitude is tens of
seconds after compilation (switching candidate dominates), so 9 datasets ≈ a
few minutes. If a single call exceeds 60 s, reduce `n_seeds` to 2 and record
why.

Mark slow / integration tests explicitly (e.g., `pytest.mark.slow`).

## Fixtures

- `simulate_bandit_agent(agent, seed)` for the three agents as parametrised module-scope fixtures (`T=400`, `K=3`, `block_length=100`).
- Real data: none checked in; the script reads `data/` (gitignored) and is exercised with `--simulate` in CI-free manual runs.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind). (None flagged; `test_switching_choice.py::TestModelComparison` stays — it tests a different claim on choice-only data.)
- User-facing documentation listed as tasks is updated, not deferred.
- The confusion-matrix test's recorded margins are real observed numbers, not placeholders.
