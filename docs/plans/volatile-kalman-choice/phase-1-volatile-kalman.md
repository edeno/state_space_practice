# Phase 1 — `volatile_kalman.py`: VKF recursions and `VolatileKalmanChoiceModel`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#vkf-reference)

Ship the volatile Kalman filter (Gaussian and binary variants) as pure
`lax.scan` functions that reproduce the reference implementation, a
reward-learning choice model fitted by SGD through the recursion, a
block-change bandit simulator with ground truth, and their tests and docs.
Nothing in existing modules changes except the package export list.

**Inputs to read first:**

- [designs.md §1-5](designs.md#vkf-reference) — the transcribed recursions, the JAX core, the NumPy oracle, the model and the simulator. Read all of it before writing code.
- `src/state_space_practice/multinomial_choice.py:567-617` (class docstring and hook list), `:772-814` (`_populate_uncertainty`: the attribute names to mirror), `:1005-1049` (SGD protocol), `:1172-1215` (`n_free_params`, `bic`, `compare_to_null`), `:1221-1238` (`summary`) — the conventions `VolatileKalmanChoiceModel` follows.
- `src/state_space_practice/covariate_choice.py:210-320` (`covariate_choice_filter`: host-side validation, coercion, then a jitted core), `:735-762` (`_bind_covariates`: validate everything before assigning), `:1089-1187` (`SimulatedRLChoiceData` / `simulate_rl_choice_data`: simulator conventions).
- `src/state_space_practice/switching_choice.py:1318-1352` — `lax.scan` + `jax.random` simulation pattern reused by the simulator.
- `src/state_space_practice/sgd_fitting.py:354-397` (mixin protocol, `_prepare_sgd_data` hook at `:384`), `:506-535` (`fit_sgd` settings).
- `src/state_space_practice/parameter_transforms.py:96` (`POSITIVE`), `:152` (`UNIT_INTERVAL`).
- `src/state_space_practice/behavioral_uncertainty.py:46-68` (`categorical_entropy`), `:85-105` (`compute_surprise`).
- `src/state_space_practice/utils.py:910-917` (`contains_tracer`), `:985-1014` (`validate_choice_indices`), `:1044` (`validate_finite_array`), `:1090` (`validate_unit_interval_array`).
- `src/state_space_practice/kalman.py:613-622` (`kalman_filter` signature; its scan at `:570-592` predicts before every update) — the independent oracle for `λ = 0`.
- `src/state_space_practice/__init__.py:35-53` (`_LAZY_API`), `:55-76` (`__all__`), `:78-100` (`TYPE_CHECKING` imports); `src/state_space_practice/tests/test_package.py:41-44` parametrises over `_LAZY_API`, so the export is tested automatically.
- `src/state_space_practice/tests/test_gradients.py:47-69` (`capture_sgd_problem`), `:99-177` (`check_sgd_loss`), `:196-216` (`test_covariate_choice_loss`, the shape to copy).
- `src/state_space_practice/tests/oracles.py` — home of the NumPy reference (add at the end of the file).
- `src/state_space_practice/tests/conftest.py:47-149` — tests calling `.fit_sgd(` are auto-marked slow; anything else slow needs `@pytest.mark.slow`.
- `pyproject.toml:134-159` (`[tool.mypy] files`), `CHANGELOG.md:6-8` (`## [Unreleased]` / `### Added`), `README.md:60-67` (Package layout).

**Contracts referenced:** none (no `shared-contracts.md`; the cross-phase surface is the public API of this module, documented in designs.md §2 and §4).

**Designs referenced:** [designs.md#vkf-reference](designs.md#vkf-reference), [#vkf-jax-core](designs.md#vkf-jax-core), [#vkf-oracle](designs.md#vkf-oracle), [#vkf-choice-model](designs.md#vkf-choice-model), [#vkf-simulator](designs.md#vkf-simulator).

## Tasks

- **Create `src/state_space_practice/volatile_kalman.py` with the VKF core** exactly as in [designs.md §2](designs.md#vkf-jax-core): `VolatileKalmanState`, `VolatileKalmanResult`, `_volatility_update`, `volatile_kalman_step`, `_volatile_kalman_scan` (jitted, `observation` and `shared_volatility` static), and the public `volatile_kalman_filter` / `binary_volatile_kalman_filter` wrappers. Defaults (`outcome_mask=None`, `shared_volatility=False`, `init_mean=0`, `init_variance=noise`) must reproduce `vkf.m` / `vkf_bin.m`; `λ = 0` is accepted (the reference rejects it; we need it for the Kalman special case and phase 2). Validation is host-side and skipped for traced inputs (`contains_tracer`), as in `covariate_choice_filter`. Module docstring cites Piray & Daw 2020 with the correct year/volume (*PLoS Comput Biol* 16(7): e1007963) and Daw et al. 2006 for the masked (unchosen-option) treatment.
- **Add the NumPy reference to `tests/oracles.py`**: `reference_volatile_kalman(outcomes, lam, v0, noise, binary=False)` from [designs.md §3](designs.md#vkf-oracle) — a loop transcription of the MATLAB, independent of the JAX code.
- **Add `VolatileKalmanChoiceModel`** to `volatile_kalman.py` as in [designs.md §4](designs.md#vkf-choice-model): constructor validation (`ValueError`), `_prepare_sgd_data` that validates before assigning, `fit_sgd(choices, rewards, ...)` with a documented signature, the `SGDFittableMixin` protocol (`_build_param_spec`, `_sgd_loss_fn`, `_store_sgd_params`, `_finalize_sgd`, `_n_timesteps`), trial-aligned attributes (`predicted_option_values_`, `filtered_option_values_`, `predicted_option_variances_`, `filtered_option_variances_`, `volatility_`, `learning_rate_`, `volatility_prediction_error_`, `predicted_choice_entropy_`, `surprise_`), `predictive_log_likelihood(choices, rewards)`, `choice_probabilities()`, `n_free_params`, `bic()`, `compare_to_null()`, `summary()`, `__repr__`, `is_fitted`, and `NotFittedError` from `_check_fitted`. Decisions to encode in the class docstring: full-`K` values (no reference option, values are anchored by the reward scale), softmax on the VKF prediction `m`, `observation="binary"` and `shared_volatility=True` defaults with the one-sentence rationale from designs.md §4.
- **Add the simulator** `SimulatedVolatileChoiceData` / `simulate_volatile_bandit_data` ([designs.md §5](designs.md#vkf-simulator)) to `volatile_kalman.py`, built on `volatile_kalman_step` so the simulator and the filter share one recursion; document that `block_reward_probs` rows are blocks and `change_points` are the trial indices where they change.
- **Export and type-check**: add `"VolatileKalmanChoiceModel": "volatile_kalman"` to `_LAZY_API` (`__init__.py:35-53`), the name to `__all__` (`:55-76`) and the `TYPE_CHECKING` import (`:78-100`); add `"src/state_space_practice/volatile_kalman.py"` to `[tool.mypy] files` (`pyproject.toml:134-159`) and make `uv run mypy` clean for it (`Literal` types for the options, `Array` returns, `ArrayLike` inputs).
- **Tests — `tests/test_volatile_kalman.py`** (see validation slice). Fixtures: a module-level random Gaussian outcome matrix `(200, 2)` and a binary one, the hand trace of designs.md §3, and one `simulate_volatile_bandit_data(seed=0)` dataset (`scope="module"`). Every fit-based test is auto-marked slow by conftest; mark the recovery loop and the change-point test explicitly as well.
- **Tests — `tests/test_gradients.py`**: add `test_volatile_kalman_choice_loss(monkeypatch)` after `test_covariate_choice_loss` (`:196-216`), using `capture_sgd_problem(monkeypatch, model, choices, rewards)` and `check_sgd_loss` on a 40-trial simulated dataset with all four parameters learnable (default `rtol`).
- **User-facing docs**: CHANGELOG `### Added` bullet (`CHANGELOG.md:8`) — "**`VolatileKalmanChoiceModel` and `volatile_kalman`**: volatile Kalman filter (Piray & Daw 2020; Gaussian and binary variants, transcribed from the reference code, with an observation mask and optional shared volatility) and a reward-learning bandit model with a trial-varying learning rate, fitted by `fit_sgd`; `simulate_volatile_bandit_data` for block-wise reward-contingency changes." Add `VolatileKalmanChoiceModel` to the entry-point list in README "Package layout" (`README.md:60-67`). Public docstrings are the NumPy-style ones specified in designs.md (shapes on every array).

## Deliberately not in this phase

- **Volatility as the process noise of the Laplace-EKF choice filter** (`CovariateChoiceModel(dynamics="volatile")`): phase 2. This phase's model learns values from rewards; the hybrid infers them from choices.
- **Model comparison against `CovariateChoiceModel` / `SwitchingChoiceModel`** and the real-data script: phase 3. Do not add comparison helpers here beyond `predictive_log_likelihood`, which phase 3 consumes.
- **A simulator of the article's generative volatility process** (multiplicative Beta diffusion, S1 Appendix): not planned (overview Open Questions); the block-change bandit is the motivating scenario.
- **EM fitting** of the VKF model: none exists (no latent posterior over parameters); `fit_sgd` only.
- **SBC / calibration tests**: not applicable — the model has no posterior over a latent inferred from choices (see overview Non-Goals).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_gaussian_filter_matches_reference_transcription` | `volatile_kalman_filter` on the `(200, 2)` Gaussian fixture equals `reference_volatile_kalman(...)` on all seven signals (`predictions`, `volatility`, `learning_rate`, `prediction_error`, `volatility_prediction_error`, `posterior_means`, `posterior_variances`), `atol=1e-10`; guard: `volatility` is not constant (λ actually acts). |
| `test_binary_filter_matches_reference_transcription` | Same for `binary_volatile_kalman_filter` vs `reference_volatile_kalman(..., binary=True)` on 0/1 outcomes; also `learning_rate == sqrt(predicted_variances)` exactly. |
| `test_hand_computed_trace` | Gaussian, `λ=0.5, v0=1, σ²=1`, outcomes `[1, 0, 2]`: trials 1-2 equal the exact fractions of designs.md §3 (`k = [2/3, 17/26]`, posterior `m = [2/3, 3/13]`, post-update `v = [11/9, 2303/2028]`) at `atol=1e-12`; trial 3 matches the tabulated decimals at `atol=1e-4`. |
| `test_zero_learning_rate_is_kalman_filter` | `λ=0`: `posterior_means` / `posterior_variances` equal `kalman.kalman_filter` (A=1, Q=v0, H=1, R=σ², prior N(0, σ²)) to `1e-10`; `predictions[1:] == posterior_means[:-1]`; `volatility` constant `v0`; guard: `λ=0.3` differs from the KF by more than `1e-3`. |
| `test_unobserved_cue_is_predict_only` | With a mask that hides cue 1 on trials 5-20: its `posterior_means` unchanged, `posterior_variances[t] == posterior_variances[t-1] + volatility[t]` exactly, `volatility_prediction_error == 0` exactly, per-cue volatility unchanged; cue 0 identical to the unmasked run. |
| `test_all_ones_mask_is_bit_identical` | `outcome_mask=ones` vs `None`: `np.testing.assert_array_equal` on every field (both observation models). |
| `test_shared_volatility_single_cue_equals_per_cue` | `C=1`: `shared_volatility=True` and `False` agree on every field to `1e-12` (shapes differ only by the trailing axis). |
| `test_shared_volatility_uses_mean_observed_error` | `C=3`, full mask: shared `v_{t+1} - v_t == λ * mean_k δv_k` (per-cue run's `volatility_prediction_error` recomputed at the shared state) to `1e-12`; with a one-hot mask it equals `λ * δv_chosen`. |
| `test_volatility_stays_positive` | Outcomes with jumps of ±50 and `λ=0.9`: `volatility > 0` at every trial for both variants; guard: `volatility` spans at least two orders of magnitude. |
| `test_volatility_rises_after_step_change` | Gaussian outcomes with a mean step at trial 100 (`σ²=1`, `λ=0.3`): mean `volatility` over trials 101-115 > over trials 85-99 (guard: with `λ=0` both are equal). |
| `test_binary_rejects_outcomes_outside_unit_interval`, `test_rejects_invalid_hyperparameters` | `ValueError` for `λ ∉ [0, 1)`, `v0 ≤ 0`, noise `≤ 0`, non-finite outcomes, mask shape mismatch, binary outcome `1.5`. |
| `test_model_rejects_bad_inputs_without_binding` | `fit_sgd` raises `ValueError` for out-of-range choices, length mismatch, rewards outside [0, 1] (binary), fewer than 2 trials; afterwards `model._choices is None` and `model.is_fitted` is False. |
| `test_not_fitted_raises` | `bic()`, `summary()`, `choice_probabilities()` raise `NotFittedError` before `fit_sgd`. |
| `test_n_free_params_and_bic` | Flags → count (4 by default, 2 with two frozen); `bic() == -2 LL + k log T` to `1e-10`; `compare_to_null()["null_ll"] == T log(1/K)`. |
| `test_simulator_consistent_with_filter` | `volatile_kalman_filter(rewards broadcast, mask=one_hot(choices), same params)` reproduces `true_values`, `true_volatility`, `true_learning_rate` to `1e-10`; `rewards ∈ {0, 1}`; `change_points == [100, 200, 300]`; `reward_probs` constant within blocks and different across them. |
| `test_simulator_seed_reproducible_and_agent_learns` | Same seed → identical data; agent chooses the block's best option on > 50% of the last 50 trials of each block (guard on a non-trivial simulation). |
| `test_predictive_log_likelihood_is_causal` (slow, fits) | After `fit_sgd`: `predictive_log_likelihood(choices, rewards) == log_likelihood_` to `1e-10`, and `predictive_log_likelihood(full) - predictive_log_likelihood(prefix)` equals the direct sum of the held-out trials' `log softmax` terms computed from `predicted_option_values_` at `1e-10`. |
| `test_fit_sgd_improves_ll_and_populates_outputs` (slow) | `log_likelihood_history_[-1] > [0]`; all `*_option_*_` attributes `(T, K)`, `volatility_` `(T,)`, `surprise_[t] == -log choice_probabilities()[t, c_t]`; `predicted_option_variances_` of an option grows monotonically over any run of ≥ 5 trials in which it is not chosen and drops on the trial it is chosen (`filtered < predicted`). |
| `test_fit_recovers_parameters_over_seeds` (slow) | 5 seeds, `T=600`, `λ_true ∈ {0.05, 0.2, 0.5, 0.2, 0.05}`, `β_true = 2`: for every seed `LL(fit) ≥ LL(true params) − 1.0`; `β̂` within 30% of truth in ≥ 4/5 seeds; Spearman correlation between `λ_true` and `λ̂` > 0.7. Record the observed values in the docstring; also record whether `v0`/`ω` recover (ridge expected, no assertion). |
| `test_fitted_volatility_spikes_after_contingency_changes` (slow) | Fit on `simulate_volatile_bandit_data(seed=0)`: `volatility_` mean over the 15 trials after each change point exceeds the mean over the 15 before it for ≥ 2 of 3 change points; fitted `λ̂ > 0.02` (guard). |
| `test_gradients.py::test_volatile_kalman_choice_loss` (slow) | `check_sgd_loss` finite-difference agreement and transform round trips for all four parameters. |
| `test_package.py` (existing, automatic) | `VolatileKalmanChoiceModel` resolves lazily to `state_space_practice.volatile_kalman`. |

Mark slow / integration tests explicitly (e.g., `pytest.mark.slow`).

## Fixtures

- Random Gaussian outcomes `(200, 2)` from `np.random.default_rng(0)` (mean 0.5, sd 1, with a +3 mean shift after trial 100 so volatility moves) and binary outcomes `(200, 2)` with per-cue Bernoulli probabilities `0.8 / 0.3` switching at trial 100 — module-level constants in `test_volatile_kalman.py`.
- `simulate_volatile_bandit_data(seed=0)` (defaults: `T=400`, `K=3`, `block_length=100`) as a `scope="module"` fixture; the recovery test simulates its own five datasets inline.
- No real-data fixture: the model is validated on simulated data here; real bandit data enters in phase 3's script.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind). (None are flagged: this phase is purely additive.)
- User-facing documentation listed as tasks is updated, not deferred.
- The default calls of `volatile_kalman_filter` / `binary_volatile_kalman_filter` are the reference recursions line for line (compare against designs.md §1, not against memory).
