# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

New code lives in `volatile_kalman.py` (phase 1) and
`choice_model_comparison.py` (phase 3); phase 2 adds to `covariate_choice.py`
without touching its existing compiled paths.

- `src/state_space_practice/covariate_choice.py:54-87` — `covariate_predict`: untouched (the hybrid inlines the same predict with a per-trial `Q_t`, as the existing scan already inlines it).
- `src/state_space_practice/covariate_choice.py:323-384` — `_covariate_choice_filter_jit`: **untouched**; the volatile filter is a sibling function modelled on its scan (`:342-370`: carry `:343`, predict `:347-348`, constant `Q` at `:339`, softmax update `:354-361`). Keeping a separate compiled function is what makes `dynamics="fixed"` bit-for-bit unchanged.
- `src/state_space_practice/covariate_choice.py:387-426` — `_rts_smoother_pass_with_predictions`: reused unchanged by the volatile smoother (it consumes stored predictions, so a time-varying `Q_t` needs no smoother change); its gain / cross-covariance lines `:409-412` are the formula the volatility update reuses.
- `src/state_space_practice/covariate_choice.py:429-487` — `covariate_choice_smoother`: lines `:464-487` factored into `_smooth_filter_result(filt, A)` (behaviour-preserving) so the volatile smoother shares it.
- `src/state_space_practice/covariate_choice.py:522-616` (`CovariateChoiceModel` docstring, `__init__`), `:618-626` (`__repr__`), `:630-648` (`_filter_kwargs` / `_run_filter` / `_run_smoother`), `:764-843` (`fit`, `fit_sgd`), `:847-904` (SGD hooks), `:938-968` (`n_free_params`, `summary`) — additive `dynamics="volatile"` keyword and dispatch; defaults unchanged.
- `src/state_space_practice/covariate_choice.py:1-24` — module docstring: documents the new dynamics; reference [2] (`:20-21`) currently cites Piray & Daw as 2021, 17(4) — corrected to 2020, 16(7) in phase 2 while editing this docstring.
- `src/state_space_practice/covariate_choice.py:1089-1187` — `SimulatedRLChoiceData` / `simulate_rl_choice_data`: untouched; used as the stationary control dataset.
- `src/state_space_practice/multinomial_choice.py:116-283` — `_softmax_update_core`: reused unchanged by the hybrid; Newton line-search constants `_NEWTON_STEP_SIZES` (`:45`), `_ARMIJO_C` (`:47`), `NEWTON_GAP_TOL` (`:62`) untouched.
- `src/state_space_practice/multinomial_choice.py:772-814` — `_populate_uncertainty`: untouched; `CovariateChoiceModel` overrides it (phase 2) to add `volatility_`; `VolatileKalmanChoiceModel` mirrors its attribute names.
- `src/state_space_practice/multinomial_choice.py:1009-1049` (SGD protocol), `:1053-1086` (`_m_step_process_noise`: its per-trial summand is what the hybrid volatility evaluates online), `:1172-1215` (`n_free_params`, `bic`, `compare_to_null`) — patterns mirrored, not modified.
- `src/state_space_practice/switching_choice.py:206-249` (`switching_choice_filter`), `:361-363` (per-state `Q_j = q_j I`), `:687-794` (`SwitchingChoiceModel` attributes), `:1122-1173` (`fit_sgd`), `:1179-1203` (`_build_param_spec`) — read-only consumers in the comparison harness.
- `src/state_space_practice/sgd_fitting.py:354-397` (`SGDFittableMixin` protocol; `_prepare_sgd_data` hook `:384`), `:506-535` (`fit_sgd`) — the VKF model implements the protocol; no change.
- `src/state_space_practice/parameter_transforms.py:96` (`POSITIVE`), `:152` (`UNIT_INTERVAL`), `:288-306` (`transform_to_unconstrained`) — used, not modified.
- `src/state_space_practice/behavioral_uncertainty.py:46-68` (`categorical_entropy`), `:85-105` (`compute_surprise`) — used for the VKF model's uncertainty summaries; `append_reference_option` / `option_variances_from_covariances` (`:12`, `:27`) are not needed (full-`K` values, per-option variances are already scalars).
- `src/state_space_practice/kalman.py:613-622` — `kalman_filter`: the independent oracle for the `λ = 0` special case (its scan `:570-592` predicts before each update, matching the VKF's `w0 + v0` prior at the first trial).
- `src/state_space_practice/utils.py:26` (`symmetrize`), `:185` (`psd_solve`), `:910` (`contains_tracer`), `:985` (`validate_choice_indices`), `:1044` (`validate_finite_array`), `:1090` (`validate_unit_interval_array`) — used.
- `src/state_space_practice/__init__.py:35-53` (`_LAZY_API`), `:55-76` (`__all__`), `:78-100` (`TYPE_CHECKING`) — one new entry each (phase 1); `tests/test_package.py:41-44` covers it automatically.
- `pyproject.toml:134-159` — `[tool.mypy] files`: add the two new modules; `covariate_choice.py` is not in the list and stays out.
- `CHANGELOG.md:6-8` (`## [Unreleased]` / `### Added`), `README.md:60-67` (Package layout), `:36-39` (gitignored `data/` loaders) — documentation touch points.
- Tests reused as infrastructure: `tests/test_gradients.py:47-69`, `:99-177` (`capture_sgd_problem`, `check_sgd_loss`); `tests/oracles.py` (new NumPy reference lives here); `tests/recovery_helpers.py:68-74` (`assert_ll_improves`); `tests/test_covariate_choice.py:275-320` (parity-test shape), `:456-504` (data helper); `tests/test_switching_choice.py:718-748` (existing switching-vs-fixed comparison); `tests/conftest.py:47-149` (auto slow marking). `tests/test_oracle_choice.py`, `test_calibration_behavioural.py`, `test_sbc_ranks.py` are **not** extended: see Non-Goals.

## Scope and dependency policy

### Goals

- A faithful, differentiable JAX implementation of the VKF (Gaussian and binary variants) whose default call reproduces the authors' reference code, with two additive options a bandit needs: an observation mask (unchosen options are not observed) and a volatility shared across options.
- `VolatileKalmanChoiceModel`: option values learned from rewards by the VKF, choices `~ softmax(β · values)`, `λ, v0, ω (or σ²), β` fitted by SGD through the recursion, trial-aligned values / variances / volatility / learning rate and the repo's uncertainty summaries, `bic()`, a simulator with ground truth, recovery over seeds.
- `CovariateChoiceModel(dynamics="volatile")`: the VKF volatility as the process variance of the existing Laplace-EKF choice filter, updated online from the filtered value change; `λ = 0` reproduces the fixed-`q` filter exactly; gated by the identifiability report before scientific use.
- A model-comparison harness showing that BIC / held-out log-likelihood recover the generating model among `{VKF, fixed-q CovariateChoiceModel, SwitchingChoiceModel}` on simulated data, and a script that runs the same comparison on real bandit data.
- Backwards compatibility: new module plus additive keywords; every default reproduces today's behaviour bit-for-bit.

### Non-Goals

- Changing `_covariate_choice_filter_jit`, `_softmax_update_core`, the Newton line-search constants, `ChoiceFilterResult` / `ChoiceSmootherResult`, or any EM M-step.
- EM fitting for either volatile model (no closed-form M-step for `λ`; `fit_sgd` only; `CovariateChoiceModel.fit` raises in volatile mode).
- A simulator of the article's generative volatility process (multiplicative Beta diffusion): not read in detail, not implemented.
- Calibration / SBC tests for the new models: the pure VKF model has no posterior over a latent inferred from choices (its "posterior variance" is the agent's belief, not an inferential posterior), and the hybrid has no implemented generative counterpart, so `test_calibration_behavioural.py` / `test_sbc_ranks.py` do not apply. Quadrature oracles (`test_oracle_choice.py`) do not apply either: the VKF is itself an approximate filter whose reference is the authors' code.
- Learning ω online (Piray & Daw 2021's joint stochasticity/volatility estimator) — follow-up.
- Per-state volatility inside `SwitchingChoiceModel`; multi-session pooling; cross-validated or bootstrap model comparison.
- Per-trial log-likelihood outputs from the existing filters (held-out LL is computed as a prefix difference instead).

### Dependency policy

No new third-party dependencies. Cross-plan dependencies (directories under
`docs/plans/`, written in parallel; link, do not restate):

- `docs/plans/identifiability-diagnostics/` — provides `.identifiability_report()`. This plan uses it as a **gate on the hybrid model** (phase 2): no near-null direction may mix `{volatility_learning_rate, process_noise}` with `inverse_temperature` on simulated data before the hybrid is used for claims. If it has not landed when phase 2 executes, phase 2 ships an interim finite-difference Hessian check (designs.md §8) with the explicit trigger to replace it. The same report is the tool for quantifying the pure model's `(v0, ω)` ridge (Open Question 3).
- `docs/plans/masks-and-multi-sequence/` — needed only if real-data comparisons are to pool several sessions; this plan fits one session at a time and does not depend on it.

Literature (each cited for one claim):

- Piray, P. & Daw, N.D. (2020). A simple model for learning in volatile environments. *PLoS Comput Biol* 16(7): e1007963 — the VKF recursions (Eqs 9-13 Gaussian, 14-19 binary), the `λ = 0` Kalman special case and the inference-only noise parameter ω; reference code `github.com/payampiray/VKF`.
- Piray, P. & Daw, N.D. (2021). A model for learning based on the joint estimation of stochasticity and volatility. *Nature Communications* 12: 6587 — extension that also tracks the outcome noise online; the follow-up if a fitted constant ω proves inadequate.
- Behrens, T.E.J., Woolrich, M.W., Walton, M.E. & Rushworth, M.F.S. (2007). Learning the value of information in an uncertain world. *Nature Neuroscience* 10(9): 1214-1221 — learning rates track environmental volatility; the behavioural motivation for a trial-varying learning rate.
- Mathys, C., Daunizeau, J., Friston, K.J. & Stephan, K.E. (2011). A Bayesian foundation for individual learning under uncertainty. *Frontiers in Human Neuroscience* 5: 39 — the hierarchical Gaussian filter, the alternative hierarchical volatility model considered; VKF chosen for its closed-form updates and reference code.
- Nassar, M.R., Wilson, R.C., Heasly, B. & Gold, J.I. (2010). An approximately Bayesian delta-rule model explains the dynamics of belief updating in a changing environment. *Journal of Neuroscience* 30(37): 12366-12378 — change-point-driven learning rates, the alternative considered; the repo's `SwitchingChoiceModel` is the discrete-regime competitor in the comparison.
- Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J. (2006). Cortical substrates for exploratory decisions in humans. *Nature* 441: 876-879 — Kalman-filter bandit with softmax choice in which unchosen arms' uncertainty grows; the semantics of the observation mask.
- Comrie, A.E., et al. (Frank, L.M.) (2024). Hippocampal representations of alternative possibilities are flexibly generated to meet cognitive demands. *bioRxiv* 2024.09.23.613567 — the Frank-lab spatial bandit whose reward probabilities change across epochs (the block-change scenario the simulators mimic). Preprint; check for a journal version before citing in a manuscript.

## Metrics

- Reference parity: JAX VKF equals the NumPy transcription of `vkf.m` / `vkf_bin.m` on all seven signals at `1e-10`; the `λ = 0` case equals `kalman.kalman_filter` at `1e-10`; the hand-computed trace matches exactly for trials 1-2.
- Backwards compatibility: pre/post baseline of `covariate_choice_filter` / `_smoother` / `fit_sgd` outputs identical under `assert_array_equal`; `volatile_covariate_choice_filter(λ=0, v0=q)` equals the fixed filter at `1e-12`.
- Recovery (phase 1): `LL(fit) ≥ LL(true) − 1` nat on every seed; `β̂` within 30% in ≥ 4/5 seeds; Spearman(`λ_true`, `λ̂`) > 0.7.
- Behaviour: fitted volatility rises after ≥ 2 of 3 contingency changes (both models).
- Model comparison: generating model wins held-out LL and BIC in ≥ 2 of 3 seeds per generator.
- Hygiene: fast suite (`-m "not slow"`) still under a minute (every fit test slow-marked); `uv run mypy` clean with the two new modules listed; `ruff check` / `ruff format --check` clean.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Phase 2 silently changes the fixed-`q` path (refactor of the smoother tail, dispatch bugs) | Separate compiled volatile function; baseline `.npz` captured before edits and compared bit-for-bit after; `λ = 0` parity test. |
| Our "reference" is a NumPy transcription, not a MATLAB run | Two transcription-independent checks: the exact hand trace and the Kalman-filter oracle at `λ = 0`; the transcription itself is verbatim from the reference files quoted in designs.md §1. |
| Binary VKF's mean update uses `sqrt(w+v)`, easy to "correct" to the gain from memory | Quoted reference code and article Eqs 14-16 in designs.md; the parity test on `learning_rate == sqrt(predicted_variances)`. |
| Volatility / inverse-temperature trade-off makes the hybrid's parameters unidentifiable | Identifiability gate (report from the sibling plan, interim Hessian otherwise) is a documented precondition for scientific use; the pure model's `(v0, ω)` ridge is reported by the recovery test. |
| Volatile vs fixed-`q` agents confusable at `T = 400` | Distinct generator regimes (`λ = 0.3` with block changes vs `λ`-free latent), both BIC and held-out LL reported, observed margins pinned in the test; if margins are thin the plan says so rather than loosening thresholds. |
| SGD instabilities: `λ → 1` inflates `v`, `β → ∞` saturates the softmax | `UNIT_INTERVAL` keeps `λ < 1` (and `v_t = (1−λk)v + λ(≥0)` stays positive); `POSITIVE` for `β`; the model has no Laplace step, so large `β` cannot produce NaN, only vanishing gradients (documented). |
| Slow confusion-matrix test blows the nightly budget | Smoke-time one `compare_candidates` call first; `n_seeds` knob; compilation is shared across seeds (same shapes). |
| Real-data script depends on an unverified loader accessor | `--simulate` keeps the script exercisable; the loader contract is one dict with named keys and the script fails loud if absent (Open Question 1). |

## Rollout Strategy

Purely additive, three independent PRs (phase 3 needs phase 1; phase 2 needs
phase 1 only for `_volatility_update` and the simulator; phases 2 and 3 are
independent of each other). No feature flags: `dynamics="fixed"` is the
default and untouched; the new module has no effect on importing users (lazy
export). No deprecations. Backwards compatibility was required by the
request and is verified per phase (baseline comparison, parity tests). The
hybrid's *scientific* use is gated by the identifiability report; the code
ships regardless, with the gate documented in the class docstring and
CHANGELOG.

## Open Questions

1. **Real-data trial accessor.** `data/` is gitignored and absent from this checkout; the known loader (`data.load_bandit_data.load_neural_recording_from_files`, used by `scripts/position_decoding_demo.py:23-35`) returns position / spikes / track graph, not trials. Best answer: the script expects `load_bandit_trials(data_dir, session) -> {"choices", "rewards", "epoch_boundaries"}` and fails with that contract in the message; whoever owns `data/` adds it. Deferred to phase 3 execution.
2. **Choice input for the binary VKF.** Decided: softmax on the logit-scale prediction `m`. Revisit (add a `choice_input="probability"` keyword using `sigmoid(m)`) if real-data fits prefer it; the model comparison would show it.
3. **`(v0, ω)` ridge in the pure VKF model.** Both enter the gain through `w + v`; expected to recover poorly. Best answer: report it from the recovery test; if the identifiability report flags it on real data, fix ω (`learn_observation_noise=False`) as the documented recipe.
4. **Shared vs per-option volatility default.** Decided: shared (environment-level contingency changes). Breaks if the real task changes one patch's probability at a time; then `shared_volatility=False` is the right model and the default should be revisited.
5. **Generative volatility simulator (Beta diffusion, S1 Appendix).** Deferred; would enable tracking-quality tests of the pure filter against a true `v_t`. Trigger: a claim about the *filter's* tracking (rather than the choice models) is needed.
6. **Per-trial log-likelihoods from the existing filters** would simplify held-out scoring but change `ChoiceFilterResult`; deferred — the prefix difference is exact for causal filters.
7. **Hybrid as a fourth comparison candidate.** Trigger: phase 2 merged and its identifiability gate passing.

Unrelated issues noticed (not fixed by this plan except where noted):
`docs/plans/2026-04-04-multinomial-choice-model.md:53` and
`docs/plans/2026-04-05-rl-state-space-covariates.md:72` repeat the wrong
Piray & Daw year/volume; only the `covariate_choice.py` module docstring is
corrected (phase 2, same file being edited).

## Estimated Effort

- Phase 1: ~600-700 LOC in `volatile_kalman.py` (core ~200, model ~300, simulator ~80, docstrings), ~450 LOC tests, ~40 LOC exports/docs.
- Phase 2: ~250-300 LOC added to `covariate_choice.py` (filter core, smoother refactor, hooks), ~350 LOC tests (incl. the gate).
- Phase 3: ~300-350 LOC `choice_model_comparison.py`, ~250 LOC tests, ~150 LOC script.
- Total ≈ 2.3k LOC; no dependency changes.
