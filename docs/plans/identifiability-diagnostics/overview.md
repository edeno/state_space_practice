# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

Everything below was read in the planning session; line numbers are for the
`master` tip at planning time (`4b74f03`).

- `src/state_space_practice/sgd_fitting.py:506-535` — `SGDFittableMixin.fit_sgd`
  signature and docstring: **gains** a stored copy of the prepared data (see below) and
  a sibling method `identifiability_report`. Optimisation loop untouched.
- `src/state_space_practice/sgd_fitting.py:544` — `args, kwargs =
  self._prepare_sgd_data(*args, **kwargs)`: the one line after which phase 2 inserts
  `self._sgd_data_ = (args, kwargs)`. This is *the* prepared form `_sgd_loss_fn`
  consumes (every public `fit_sgd` override — `place_field_model.py:1420-1534`,
  `oscillator_models.py:1146-1190`, `switching_point_process.py:3313-3359`,
  `multinomial_choice.py:954-1001`, `covariate_choice.py:805-843`, the choice / Smith /
  GP / point-process models — validates, converts and binds auxiliary state *before*
  calling `super().fit_sgd`, so nothing else on the model reproduces this form).
- `src/state_space_practice/sgd_fitting.py:546-559` — `_check_sgd_initialized`,
  `_build_param_spec`, the frozen/trainable split and
  `transform_to_unconstrained(include_non_trainable=False)`: the report mirrors this
  split (frozen entries are held fixed, trainable entries become coordinates) but never
  enters unconstrained space.
- `src/state_space_practice/sgd_fitting.py:580-586, 703-721` — `_loss_inner` divides by
  `n_timesteps` for the optimiser; the final iterate is written back with
  `_store_sgd_params` and **neither the unconstrained iterate nor the data is kept**.
  The report therefore re-reads parameters from `_build_param_spec()` (post-store) and
  needs the data (phase 2's real work). It uses the *unnormalised* loss so the Hessian is
  the observed information and `H^{-1}` a covariance.
- `src/state_space_practice/parameter_transforms.py:30-36` (`ParameterTransform`),
  `:96-99` (`POSITIVE`), `:152-160` (`UNIT_INTERVAL`, `UNCONSTRAINED`), `:163-222`
  (`PSD_MATRIX`: Cholesky-softplus, 1e-9 jitter at `:180-183`), `:225-252`
  (`STOCHASTIC_ROW`: drop last column), `:255-261` (`frozen`): untouched. The report
  identifies the structured transforms by `transform.to_constrained is
  PSD_MATRIX.to_constrained` (the identity check `tests/test_gradients.py:133` already
  relies on).
- `src/state_space_practice/utils.py:751-786` — `differentiable_spectral_radius` is a
  `jax.custom_jvp` whose rule calls `jax.pure_callback`; **`jax.hessian` through it
  raises** `ValueError: Pure callbacks do not support JVP` (verified in the planning
  session). It is inside `DirectedInfluenceModel._sgd_loss_fn`
  (`oscillator_models.py:2253-2263` via `compute_directed_influence_stability_scale`)
  and `DirectedInfluencePointProcessModel._sgd_loss_fn` (`point_process_models.py:1477-1494`
  via `construct_stable_directed_influence_transition_stack`). Untouched; the report
  falls back to finite differences of the exact gradient for these losses.
- `src/state_space_practice/exceptions.py:15-27` — `StateSpaceWarning`,
  `NotFittedError`: reused, untouched.
- `src/state_space_practice/__init__.py:33-52` (`_LAZY_API`), `:54-75` (`__all__`),
  `:77-101` (`TYPE_CHECKING` imports): phase 2 adds four names.
- `pyproject.toml:134-159` (`[tool.mypy] files`): phase 1 adds the new module.
- `CHANGELOG.md:8-83` (`[Unreleased] / ### Added`, last entry ends at `:82`; `### Testing`
  starts at `:84`): both phases append entries. `README.md:60-67` (Package layout):
  phase 2 adds the entry point and a short usage section.
- Existing verification infrastructure reused by the tests:
  `tests/test_gradients.py:47-69` (`capture_sgd_problem`: the exact `(args, kwargs)` a
  model's `_sgd_loss_fn` receives, without optimising) and `:185-448` (one small problem
  per SGD model); `tests/conftest.py:12` (x64), `:62-143` (automatic `slow` marking of
  anything calling `.fit(`/`.fit_sgd(`), `:788-826` (`coupling_params_small`,
  `simulated_coupling_small`); `coupling_model.py:262-282, 341-366`
  (`build_transition`, `interleave_coupling`), `simulate_coupling.py:21-23`,
  `point_process_kalman.py:1292-1299, 1333-1345` (`BERNOULLI_LOGIT_FAMILY`,
  `glm_laplace_update`), `kalman.py:427-433` (`kalman_measurement_update`), `:613-621`
  (`kalman_filter`).
- Not touched: `coupling_ekf.py`, `coupling_model.py`, `coupling_pg.py` (they expose no
  differentiable joint loss — `tests/test_gradients.py:17-19`; the acceptance tests build
  the joint losses from their building blocks), every model's `_build_param_spec` /
  `_sgd_loss_fn`, `em_driver.py`.

## Scope and dependency policy

### Goals

- A single report answering "which parameters (or combinations) does this fit not
  determine?" for any `SGDFittableMixin` model and any user loss, from the observed
  information in scientific (constrained) coordinates, with parameter *names*.
- Detect, with tests that would fail if the detector were wrong: exact continuous
  symmetries (latent scale vs loading; oscillator phase vs free measurement matrix),
  coordinates with zero curvature at the evaluation point (the spike-only
  `x = β = 0` trap of `docs/plans/2026-06-21-spike-field-coupling-findings.md:10-27`),
  saddle points, and unit-independence of the verdict.
- Pass the LFP-anchored coupling model: no flags, and Wald standard errors that cover the
  truth at close to nominal rate.
- Work on every one of the 15 SGD losses exercised by `tests/test_gradients.py`,
  including the two whose loss contains a host callback.

### Non-Goals

- No change to how any model is fitted, parameterised or initialised; no signature
  change anywhere (backwards-compatibility policy: additive keyword-only options and new
  functions whose defaults reproduce current behaviour).
- No profile likelihood (re-optimising the other parameters along a direction). The
  report exposes a *slice*, named as such; see Open Question 4.
- No expected (Gauss–Newton) Fisher information; the observed Hessian plus the gradient
  suffices for every failure mode targeted here (verified in the planning session on the
  spike-only coupling loss, see [designs.md](designs.md#why-the-observed-hessian-suffices-at-initialisation)).
- No log-parameter ("sloppiness in log coordinates") variant; see Open Question 3.
- No new differentiable coupling estimator in the library: the spike-only and
  LFP-anchored joint losses live in the test module. Promoting them is another plan's
  job.
- No fix to `differentiable_spectral_radius`'s missing second derivative.

### Dependency policy

No new runtime dependency (`jax`, `numpy`; `scipy` is already a dependency and is used
only by tests, for L-BFGS polishing). Downstream consumers of the names fixed here
(`identifiability.py`, `identifiability_report(...) -> IdentifiabilityReport`,
`SGDFittableMixin.identifiability_report()`):

- `docs/plans/volatile-kalman-choice/` — calls the mixin method as a gate.
- `docs/plans/theta-sweep-amplitude/` — same.

This plan depends on no other plan.

## Decided defaults and what the codebase forced us to change

The brief's defaults are recorded here with the evidence that confirmed or amended them.

| Brief | Decision | Evidence |
| --- | --- | --- |
| Hessian in constrained coordinates via `loss_theta(theta) = loss_u(transform_to_unconstrained(theta))` | **Amended.** `_sgd_loss_fn` already takes constrained parameters, so no composition is needed. For `PSD_MATRIX` the constrained value has `n²` entries for `n(n+1)/2` degrees of freedom; the round trip `to_constrained ∘ to_unconstrained` is a *projection* (and adds a 1e-9 jitter, `parameter_transforms.py:180-183`), so a Hessian in the `n²` coordinates has `n(n−1)/2` spurious null directions. Likewise `STOCHASTIC_ROW` has one redundant column per row. The report uses **minimal charts**: `vech` (lower triangle) for PSD, drop-last-column for row-stochastic, identity otherwise ([shared-contracts.md](shared-contracts.md#scientific-coordinates)). | `parameter_transforms.py:163-252` |
| `jax.hessian`, exact | **Amended: `hessian_method="auto"`** tries `jax.hessian` and falls back to Richardson-extrapolated central differences of `jax.grad` when the loss contains a `pure_callback` (DIM, DIM-PP). The method used is recorded in the report. | `utils.py:751-786`; verified failure in session |
| Frozen params excluded | Confirmed. `spec[k].trainable is False` → held fixed, passed to the loss. | `sgd_fitting.py:552-559` |
| `ravel_pytree` for flattening, pytree paths as names | Confirmed; `jax.tree_util.tree_flatten_with_path` yields the same leaf order as `ravel_pytree` (both sort dict keys). Names: `key[i,j]`, nested `key.sub.leaf[i]`. | verified in session (jax 0.10.2) |
| Near-null = eigenvalue < tol × max | Use full-row magnitudes for numerically flat annotations, then the active correlation spectrum for near-null directions. Count negative eigenvalues on a positive congruence of the full Hessian before excluding any rows; use `max(abs(eigenvalues))` as the tolerance scale. Zero diagonals with nonzero cross-curvature remain in the saddle check. | [designs.md](designs.md#spectrum-analysis) |
| Gradient norm at the fit | **Strengthened to a stationarity verdict.** Hessian null directions coincide with flat directions of the objective *only at a critical point*: along a curved orbit θ(c) of an exact symmetry, `genᵀ H gen = −g·θ''(c)`. Measured: the spike-only coupling loss is scale-invariant to 1e-14 and `g·gen = 1e-10`, yet at a non-stationary point `‖H gen‖/‖H‖ = 3e-3` with `cos(v_min, gen) = 0.82`. The report therefore carries `scaled_gradient_max = max_i |g_i|/d_i` (using the spectrum's row-scale fallback when `H_ii = 0`) (dimensionless: distance to the optimum in coordinate-SE units) and `is_stationary`; acceptance tests polish to a stationary point before asserting on null directions. | planning smoke, [designs.md](designs.md#the-hessian-is-only-meaningful-at-a-stationary-point) |
| Slice along the worst direction | Confirmed, named `worst_direction_slice`; a true profile is Open Question 4. | — |
| `StateSpaceWarning` when near-null directions exist | Confirmed (`warn=True` default; also fires for zero-curvature coordinates). | `exceptions.py:15-21` |
| Information at initialisation | Confirmed as a *recipe*, not a new API: models initialise inside `fit_sgd`, so `fit_sgd(..., num_steps=0)` followed by `identifiability_report()` reports the initial point (at the cost of the 1e-9 PSD round trip and one `_finalize_sgd`). | `sgd_fitting.py:610, 703-721` |
| Mixin keeps final unconstrained params? | **No — and it keeps no data either.** Phase 2 stores the prepared `(args, kwargs)` after `_prepare_sgd_data`; `identifiability_report()` with no data uses it, with data runs `_prepare_sgd_data` on what it is given. Parameters come from `_build_param_spec()` (the stored, possibly re-stabilised values — which is what the user will report on). | `sgd_fitting.py:544, 703-721` |
| Penalised objectives | **Caveat recorded.** `_sgd_loss_fn` may include priors/penalties (`switching_point_process.py:3455-3468`: `spike_weight_l2=100` by default; DIM connectivity penalty; Hamiltonian `l2_reg`). They add curvature and can mask a data degeneracy — the report describes *the objective the model optimises*. Model-level tests that assert exact symmetries set the penalties to zero. | `switching_point_process.py:3411-3470` |

## Metrics

- `fisher_information` on the i.i.d. Gaussian `(μ, σ²)` model matches the analytic
  observed information to `rtol=1e-8` at an arbitrary point, and the analytic expected
  information (`n/σ²`, `n/(2σ⁴)`) at the MLE.
- Exact symmetries are found: at a polished stationary point the linear-Gaussian SSM with
  free `(H, Q, P0, m0)` and the spike-only coupling loss each report exactly one
  near-null direction with `|λ|/λ_max < 1e-8` (autodiff) and `|cos(v_min, generator)| >
  0.999`; the `CommonOscillatorModel` reports `n_oscillators` of them.
- The zero-curvature trap is found: spike-only coupling at β = 0 → `process_noise_var`
  flagged, `gradient_norm < 1e-8`, saddle; spike-only latent oscillator with zero
  loadings → every dynamics coordinate flagged.
- Nothing spurious: LFP-anchored coupling at its optimum has no flags and condition
  number < 100; 95% Wald intervals cover the truth with frequency ≥ 0.85 and mean `z² ∈
  [0.6, 1.6]` over ≥ 24 seeds.
- Unit independence: rescaling coordinates by 1e±3 leaves the normalised spectrum
  unchanged to 1e-10 and the near-null count unchanged, while the raw condition number
  changes by > 1e6.
- FD and autodiff Hessians agree to `rtol=1e-6` where both exist; every model in
  `tests/test_gradients.py` produces a finite report; DIM and DIM-PP report
  `hessian_method_used == "finite_difference"`.
- Fast-suite cost of the new tests < 15 s; `uv run mypy` clean with the new module in
  `[tool.mypy] files`; `ruff check`/`ruff format` clean.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A user reads near-null directions at a non-stationary point (e.g. after a few Adam steps) and draws the wrong conclusion. | `is_stationary` / `scaled_gradient_max` in the report and its `__str__`; the docstring states the caveat; tests assert the confound is real (the same loss at a non-stationary point does *not* show the null) so the flag cannot be dropped silently. |
| Correlation-normalisation hides an isolated flat coordinate. | Annotate small full rows before forming the active spectrum; test both exact-zero rows and tiny isolated residual curvature. Negative-curvature detection always retains every row. |
| Raw spectra, flat-row thresholds and zero-diagonal fallback scales depend on parameter units. | Document these limits. Correlation-spectrum invariance applies when diagonals are nonzero and the active set is unchanged; test collinear columns rescaled by 1e±3. Do not claim exact unit independence for threshold decisions or the fallback. |
| FD Hessian accuracy makes exact nulls look like `1e-7` instead of `1e-12`. | Richardson-extrapolated central differences with relative step `1e-3·max(|θ|, 1e-2)`; tests compare FD vs autodiff to `rtol=1e-6` and use `near_null_tol=1e-6` (default) which still admits them. |
| `jax.hessian` cost `O(P)` forward-over-reverse passes and `O(P²)` memory for large `P` (e.g. `PlaceFieldModel` with a full `init_cov`: hundreds of `vech` coordinates). | Documented in the docstring; the all-models test uses the tiny problems of `test_gradients.py`. Not a correctness risk. |
| Finite-difference steps or slices leave the PSD cone / positive orthant and the loss returns NaN. | Steps are relative (`1e-3·max(|θ|,1e-2)`) so positive parameters stay positive; slice values that are NaN are reported as NaN (never masked); a non-finite Hessian raises `ValueError` naming the offending coordinates. |
| Penalties in `_sgd_loss_fn` mask a data degeneracy. | Documented caveat; model-level symmetry tests zero the penalties; the spike-oscillator test uses `spike_weight_l2=0.0`. |
| Storing the prepared data on the model grows pickles / deep copies. | Judged acceptable: several models already keep the data (`point_process_kalman.py` stores `_sgd_design_matrix`/`_sgd_spike_indicator`; `temporal_rate_gp.py` stores `_counts`) and all keep `(T, n, n)` smoother covariances larger than the data. See Open Question 1. |

(Operational risks like upstream drift / docs ambiguity belong in PR review, not here.)

## Rollout Strategy

All at once, purely additive. No feature flag. For users who never call the new API the
only observable change is one extra attribute (`_sgd_data_`) on models after
`fit_sgd`. No deprecation, no parallel paths.

## Open Questions

1. **Keep the prepared data on the model (`_sgd_data_`)?** Best answer: yes (phase 2).
   It is the only model-agnostic way to obtain the exact form `_sgd_loss_fn` consumes
   (every public `fit_sgd` override transforms the data before `super()`), several models
   already store their data, and an explicit-data call path exists for EM-fitted models.
   Revisit if pickled-model size becomes a complaint (trigger: a user report), in which
   case make the attribute opt-out via a class variable.
2. **Default tolerances** (`near_null_tol=1e-6`, `zero_curvature_tol=1e-10`,
   `stationary_tol=1e-2`). Best answer: keep; they are three or more decades away from
   the measured numerical residuals on both sides and are keyword options. A softer
   "poorly identified" band (e.g. `1e-6..1e-3`, Brun et al.'s collinearity index > 20) is
   visible in `condition_number` and the sorted spectrum but not flagged.
3. **Sloppiness in log-parameters** (Transtrum et al. 2015 convention). Deferred; the raw
   spectrum is in the parameters' own units. Trigger: a consumer plan needs decade
   counts comparable across models. Implementation would be a `scales` option
   (`H̃ = S H S`).
4. **True profile likelihood** along the worst direction (Raue et al. 2009). Deferred;
   the slice is honestly named `worst_direction_slice`, and the test suite pins the
   difference (a linear slice along a *curved* flat valley rises at fourth order, a
   profile is flat). Trigger: a gate plan needs to separate practical
   non-identifiability from a curved-but-flat valley. Implementation: re-optimise the
   remaining coordinates at each slice point with L-BFGS in unconstrained coordinates
   (the helper the tests use, [designs.md](designs.md#test-helper-polishing-to-a-stationary-point)).
5. **Should the `"auto"` fallback to finite differences warn?** Decided no: it would fire
   on every DIM call; it is logged at INFO and recorded in `hessian_method_used`.

## Estimated Effort

- Phase 1: `identifiability.py` ≈ 450 LOC (module + report + `__str__`);
  `tests/test_identifiability.py` ≈ 550 LOC (analytic unit tests, symmetry tests,
  coupling acceptance tests with the two hand-built losses and the polishing helper);
  pyproject + CHANGELOG ≈ 10 lines.
- Phase 2: `sgd_fitting.py` ≈ +45 LOC; `identifiability.py` ≈ +60 LOC (model entry
  point); `__init__.py` ≈ +12; `tests/test_identifiability_models.py` ≈ 350 LOC;
  README/CHANGELOG ≈ 40 lines.
