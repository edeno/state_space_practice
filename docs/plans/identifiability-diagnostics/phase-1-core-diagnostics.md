# Phase 1 — `identifiability.py`: observed-information report for any differentiable loss

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#module-header-and-coordinate-charts)

Ships the module, the functional API (`scientific_coordinates`, `fisher_information`,
`identifiability_report_from_loss`, `IdentifiabilityReport`), its unit tests against
analytic information matrices and exact symmetries, and the acceptance tests on the
spike-only vs LFP-anchored coupling losses. No model is touched; the mixin method and
the top-level exports are phase 2.

**Inputs to read first:**

- `src/state_space_practice/parameter_transforms.py:30-36, 96-99, 152-160, 163-222,
  225-252, 255-261` — `ParameterTransform`, the transforms and `frozen`. The chart rules
  key on `transform.to_constrained is PSD_MATRIX.to_constrained` /
  `STOCHASTIC_ROW.to_constrained`; `:163-193` shows why the constrained PSD value is not a
  minimal coordinate system (full symmetric `n×n`, 1e-9 jitter on the round trip).
- `src/state_space_practice/sgd_fitting.py:1-17, 354-373, 549-559` — the mixin protocol
  the loss contract mirrors (`_sgd_loss_fn(params, *data)` over constrained params; frozen
  entries via `spec[k].trainable`); `:294-301` for why the data must be closed over, not
  traced.
- `src/state_space_practice/utils.py:751-786` — `differentiable_spectral_radius`: the
  `custom_jvp` on a `pure_callback` whose second derivative does not exist; the reason
  for `hessian_method="auto"`. `:910-916` `contains_tracer`; `:26` `symmetrize`.
- `src/state_space_practice/exceptions.py:15-27` — `StateSpaceWarning`,
  `NotFittedError`.
- `src/state_space_practice/kalman.py:613-621` (`kalman_filter` signature; pass
  `validate_inputs=False` inside losses) and `:427-433` (`kalman_measurement_update`).
- `src/state_space_practice/point_process_kalman.py:1292-1299`
  (`BERNOULLI_LOGIT_FAMILY`), `:1333-1345, 1372-1376` (`glm_laplace_update` signature and
  return).
- `src/state_space_practice/coupling_model.py:34-75, 262-282, 341-366`
  (`CouplingModelParams`, `build_transition`, `interleave_coupling`);
  `src/state_space_practice/simulate_coupling.py:21-23` (`simulate_coupling(params,
  n_time, seed)` → `.spikes (T, S)`, `.lfp (T, 2J)`).
- `docs/plans/2026-06-21-spike-field-coupling-findings.md:10-38` — the failure the
  acceptance tests reproduce and the LFP-anchored construction that passes.
- `src/state_space_practice/tests/conftest.py:12, 62-143` — x64 is on for tests;
  anything calling `.fit(`/`.fit_sgd(` is auto-marked slow (nothing in this phase does).
- `src/state_space_practice/tests/test_gradients.py:1-20, 99-177` — style of an
  existing loss-level check (float64, finite differences, non-vacuity guards).
- `pyproject.toml:128-160` — `[tool.mypy] files` (add the module), `[tool.ruff]`
  (`B905`: `zip(..., strict=True)` is enforced for new modules).
- `CHANGELOG.md:6-8, 78-84` — `[Unreleased] / ### Added`; the last entry ends at `:82`.

**Contracts referenced:**

- [Public API](shared-contracts.md#public-api) — implement exactly these signatures
  (`identifiability_report(model, ...)` is phase 2; do not add it here).
- [Scientific coordinates](shared-contracts.md#scientific-coordinates) — chart rules,
  ordering, naming, zero-size and error cases; do not weaken.
- [`IdentifiabilityReport`](shared-contracts.md#identifiabilityreport) — fields and
  semantics; NumPy arrays / Python scalars only.
- [Report options](shared-contracts.md#report-options) — defaults.

**Designs referenced:** [designs.md](designs.md) §1–6 (module code), §8–9 (the two
caveats the tests pin), §10–12 (test losses and helpers).

## Tasks

- **Create `src/state_space_practice/identifiability.py`** with the module docstring,
  chart helpers and `scientific_coordinates` ([designs §1](designs.md#module-header-and-coordinate-charts)),
  the Hessian methods ([§2](designs.md#hessian-methods)), `_analyse_spectrum` and
  `NullDirection` ([§3](designs.md#spectrum-analysis)), the slice ([§4](designs.md#worst-direction-slice)),
  `IdentifiabilityReport` with `is_identifiable`, `is_local_minimum` and `__str__`
  ([§5](designs.md#report-text)), and `fisher_information` /
  `identifiability_report_from_loss` ([§6](designs.md#entry-points)). NumPy-style
  docstrings with shapes on every public name; the module docstring carries the two
  caveats (stationarity, penalties). Define `__all__ = ["IdentifiabilityReport",
  "NullDirection", "ScientificCoordinates", "fisher_information",
  "identifiability_report_from_loss", "scientific_coordinates"]`.
- **Type-clean the module and register it**: add
  `"src/state_space_practice/identifiability.py"` to `[tool.mypy] files`
  (`pyproject.toml:134-159`, alphabetical position after `hamiltonian_switching.py`).
  `uv run mypy` must pass; the `Protocol` for the model entry point is phase 2, so no
  `sgd_fitting` import here.
- **CHANGELOG**: append to `### Added` under `[Unreleased]` (after `CHANGELOG.md:82`,
  before `### Testing` at `:84`) an entry: **`state_space_practice.identifiability`**:
  `fisher_information` (observed information — the loss Hessian — in minimal scientific
  coordinates: lower triangle of PSD matrices, free columns of row-stochastic matrices,
  frozen parameters held fixed) and `identifiability_report_from_loss`
  (`IdentifiabilityReport`: named near-null directions, zero-curvature parameters,
  stationarity check, raw and correlation-normalised spectra, condition number, Wald
  SEs, a slice along the worst direction; `StateSpaceWarning` when not identifiable;
  `hessian_method="auto"` falls back to finite differences of the gradient for losses
  containing `differentiable_spectral_radius`). Mention that the mixin method follows
  in the next entry only when phase 2 lands (phase 2 edits this entry).
- **Write `src/state_space_practice/tests/test_identifiability.py`** with the module
  header used by every test module (`# ruff: noqa: E402`, `jax.config.update("jax_enable_x64",
  True)` before other imports), the helpers of [designs §10–12](designs.md#test-losses-spike-only-and-lfp-anchored-coupling)
  (`coupling_params`, `coupling_marginal_nll`, `coupling_theta`, `scale_latent`,
  `polish_to_stationary`, `symmetry_generator`, `cosine_with_raw_null_space`), the
  fixtures below and the validation slice below. Group into classes
  `TestScientificCoordinates`, `TestFisherInformation`, `TestReportFlags`,
  `TestReparameterisation`, `TestHessianMethods`, `TestCouplingDegeneracy` (the last
  one carries `pytestmark = pytest.mark.slow` at class level except where noted).
  Analytic/toy losses used by the fast tests:
  - `gaussian_nll(params)` for i.i.d. `x ~ N(mu, sigma2)`: `0.5 * n * log(2π sigma2) +
    sum((x − mu)²) / (2 sigma2)`; analytic observed information `[[n/σ², Σ(x−μ)/σ⁴],
    [·, −n/(2σ⁴) + Σ(x−μ)²/σ⁶]]`.
  - `collinear_regression_nll(params)`: Gaussian regression with design columns
    `X[:, 0] == X[:, 1]` and known unit noise, `0.5 * sum((y − X @ beta)²)`; exact null
    direction `(1, −1)/√2` at *every* point (straight orbit), `H = XᵀX` exactly.
  - `flat_coordinate_loss(params)`: `(theta[0] − 1)² + 0 * theta[1]`.
  - `lgssm_nll(params)`: `-kalman_filter(m0, P0, obs, A, Q, H, R, validate_inputs=False)[2]`
    with 1-D latent and observation, `obs` simulated from `(a=0.9, q=0.3, h=1.0, r=0.5,
    m0=0, p0=1)` with `T=60` and a fixed key; spec `{"h": UNCONSTRAINED, "q": POSITIVE,
    "p0": POSITIVE, "m0": UNCONSTRAINED}` (`a`, `r` fixed); symmetry
    `scale_lgssm(theta, c) = {h: h/c, q: c²q, p0: c²p0, m0: c·m0}`.
  - `psd_logdet_loss(params) = -logdet(params["Q"])` with `Q = I₂`, spec `{"Q":
    PSD_MATRIX}`: in `vech` coordinates `(Q00, Q10, Q11)` the Hessian at `I` is exactly
    `diag(1, 2, 1)` (`log(1 − q²)'' = −2` for the shared off-diagonal coordinate).
  - `spectral_radius_loss(params) = (differentiable_spectral_radius(params["A"]) − 0.5)²
    + sum(params["A"]²)` on a `2×2` `A` — the only way to exercise the `"auto"` fallback
    without a model.

## Deliberately not in this phase

- `identifiability_report(model, ...)`, `SGDFittableMixin.identifiability_report`, the
  `_sgd_data_` attribute, top-level exports, README — phase 2 (they need
  `sgd_fitting.py`, and the model-level tests need `capture_sgd_problem`).
- A profile likelihood (`profile=True`) — [overview Open Question 4](overview.md#open-questions).
- Log-parameter / `scales` option — [overview Open Question 3](overview.md#open-questions).
- Promoting `coupling_marginal_nll` into the library — another plan's scope
  ([overview Non-Goals](overview.md#non-goals)); it stays a test helper.
- Any change to `differentiable_spectral_radius`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `TestScientificCoordinates::test_names_follow_ravel_order_and_charts` | params `{"b": (2,), "a": (2,2) PSD, "z": (2,3) STOCHASTIC_ROW, "s": scalar, "f": frozen(POSITIVE)}` → `names == ("a[0,0]","a[1,0]","a[1,1]","b[0]","b[1]","s","z[0,0]","z[0,1]","z[1,0]","z[1,1]")`; `values` equals the concatenation in that order; `unravel(values)` reproduces every input (including the frozen `f`) to 1e-15; `unravel` of a perturbed vector re-symmetrises `a` and makes `z` rows sum to 1 |
| `TestScientificCoordinates::test_nested_pytree_leaf_names` | `{"mlp": {"layer_0": {"w": (2,1)}}}` → `("mlp.layer_0.w[0,0]", "mlp.layer_0.w[1,0]")` |
| `TestScientificCoordinates::test_zero_size_leaf_and_errors` | a `(1,1)` STOCHASTIC_ROW contributes no coordinate; all-frozen → `ValueError("No trainable parameters")`; PSD key holding a dict → `ValueError`; unknown transform key → `ValueError` |
| `TestFisherInformation::test_gaussian_matches_analytic_observed_information` | at `(mu, sigma2) = (0.3, 1.7)` with `n = 50` fixed-seed data, `fisher_information` matches the analytic matrix to `rtol=1e-8`; at the MLE `(x̄, s²)` the off-diagonal is < 1e-8 and the diagonal matches `(n/σ², n/(2σ⁴))` to 1e-8; `wald_standard_errors` at the MLE equal `sqrt(σ²/n)`, `sqrt(2σ⁴/n)` to 1e-8 |
| `TestFisherInformation::test_psd_chart_hessian_is_exact` | `psd_logdet_loss` at `Q = I₂` → `hessian == diag(1, 2, 1)` to 1e-10 with names `("Q[0,0]","Q[1,0]","Q[1,1]")`; guard: the same loss evaluated on the *full* `2×2` coordinates (identity chart, `transforms=None`) has a spurious zero eigenvalue (antisymmetric direction) — the reason for the chart |
| `TestReportFlags::test_collinear_regression_has_one_exact_null_direction` | `collinear_regression_nll` at an arbitrary β: `len(near_null_directions) == 1`, kind `"collinear"`, coefficients `{"beta[0]": ±0.707, "beta[1]": ∓0.707}` to 1e-6; `|cos(eigenvectors[:,0], (1,−1)/√2)| > 1 − 1e-10`; `wald_standard_errors is None` (singular `H`); `pytest.warns(StateSpaceWarning)` whose message contains `beta[0]`; `warn=False` emits nothing |
| `TestReportFlags::test_flat_coordinate_detected_before_normalisation` | `flat_coordinate_loss`: `zero_curvature_parameters == ("theta[1]",)`, `normalized_eigenvalues.shape == (1,)`, `active_parameter_names == ("theta[0]",)`, one near-null direction of kind `"zero_curvature"`, `is_identifiable is False`; also test `H = diag(2, 1e-13)`: the tiny row is annotated even though naive diagonal normalisation would yield `[1, 1]` |
| `TestReportFlags::test_identified_quadratic_has_no_flags_and_correct_ses` | `loss = 0.5 θᵀ M θ` with `M = [[4, 1], [1, 3]]` at `θ = 0`: `is_identifiable`, `is_stationary`, `n_negative_eigenvalues == 0`, `wald_standard_errors == sqrt(diag(M⁻¹))` to 1e-12, `condition_number` equals that of `D^{-1/2} M D^{-1/2}` to 1e-12; no warning (pytest errors on any) |
| `TestReportFlags::test_saddle_is_reported_not_stationary_minimum` | `loss = θ₀² − θ₁²` at `0`: `is_stationary`, `n_negative_eigenvalues == 1`, `is_local_minimum is False`, `wald_standard_errors is None` |
| `TestReportFlags::test_slice_flat_on_straight_null_orbit_and_curved_elsewhere` | collinear regression: `max − min` of `worst_direction_slice[1]` < 1e-8·(1+`loss`), `worst_direction` ∝ `(1, −1)`; identified quadratic: slice values are ≥ `loss` with equality only at `t = 0`, symmetric to 1e-12; `n_slice_points=0` → `worst_direction_slice is None` and the loss is evaluated exactly once more than with `n_slice_points=9` minus 9 (count calls with a wrapper) |
| `TestReportFlags::test_str_names_the_verdict` | `str(report)` for the collinear case contains `"NOT IDENTIFIABLE"`, `"beta[0]"`, `"stationary"`, `"slice"`; for the identified quadratic contains `"identifiable: no near-null direction"` and every parameter name with `+/-` |
| `TestReportFlags::test_off_diagonal_saddle` | `loss = theta[0] * theta[1]` at zero: eigenvalues `[-1, 1]`, no flat rows, exactly one negative eigenvalue, stationary but not a local minimum, finite stationarity metric, and negative curvature along `worst_direction`. |
| `TestReportFlags::test_flat_annotations_do_not_hide_negative_curvature` | `H = diag(1, -1e-13)`: second row is annotated as numerically flat, but the full congruence still reports one negative eigenvalue and rejects a local minimum. Also test a negative-definite Hessian so tolerance uses spectral radius rather than the largest signed eigenvalue. |
| `TestReparameterisation::test_normalised_spectrum_is_unit_free` | collinear regression with design columns rescaled by `1e-3` and `1e3` (so `beta` scales inversely): `normalized_eigenvalues` identical to the unscaled case to 1e-10 and `len(near_null_directions)` unchanged; guard: the largest raw eigenvalue changes by a factor > 1e5 (raw condition numbers are infinite on this singular fixture) |
| `TestReparameterisation::test_null_count_invariant_under_smooth_reparameterisation` | Gaussian regression + variance with collinear columns at its exact MLE (`β` any ridge point, `σ̂² = RSS/n`): `len(near_null_directions) == 1` both when the variance coordinate is `sigma2` (`POSITIVE`) and when it is `log_sigma2` (`UNCONSTRAINED`, loss exponentiates); both `is_stationary` |
| `TestHessianMethods::test_finite_difference_matches_autodiff` | `lgssm_nll` at the truth: `hessian_method="finite_difference"` vs `"autodiff"` agree to `rtol=1e-6, atol=1e-8·max|H|`; `hessian_method_used` is reported correctly for each; an invalid method string raises `ValueError` |
| `TestHessianMethods::test_auto_falls_back_only_for_callback_losses` | `spectral_radius_loss`: `"auto"` → `hessian_method_used == "finite_difference"` and a finite `hessian`; `"autodiff"` raises `ValueError` mentioning `callback`; `"auto"` on `lgssm_nll` → `"autodiff"`; a loss that raises an unrelated `ValueError` propagates unchanged under `"auto"` |
| `TestHessianMethods::test_non_finite_point_raises` | `lgssm_nll` with `q = -1` passed through `transforms=None` (so no domain is enforced) → `ValueError` mentioning `finite` |
| `TestReportFlags::test_lgssm_scale_degeneracy_at_polished_optimum` (slow) | polish `lgssm_nll` from the truth: `is_stationary`; exactly one near-null direction; `|cos(eigenvectors[:,0], symmetry_generator(scale_lgssm, θ̂))| > 0.999`; `eigenvalues[0]/eigenvalues[-1] < 1e-8`; with `h` frozen (`frozen(UNCONSTRAINED)`) and re-polished: `is_identifiable` and `condition_number < 1e4` |
| `TestReportFlags::test_hessian_null_requires_stationarity` (slow) | `lgssm_nll` at the truth (not the optimum, guard `scaled_gradient_max > 0.1`): `is_stationary is False`; the smallest raw eigenvector has `|cos| < 0.95` with the generator — the confound of [designs §8](designs.md#the-hessian-is-only-meaningful-at-a-stationary-point) is real, so the flag stays |
| `TestCouplingDegeneracy::test_spike_only_has_no_information_at_zero_coupling` (fast: no optimisation, `T=1000`) | `coupling_marginal_nll(base, spikes, lfp=None)` at `coupling_theta(zero_coupling=True)` with `warn=False`: `gradient_norm < 1e-8`, `is_stationary`, `"process_noise_var[0]" in zero_curvature_parameters`, `n_negative_eigenvalues > 0` (a saddle: β = 0 maximises the NLL along β for coupled data), `is_identifiable is False`, `is_local_minimum is False`; guard: the same loss at the truth has `"process_noise_var[0]"` *not* flat |
| `TestCouplingDegeneracy::test_spike_only_scale_degeneracy_at_bad_fit` (slow) | polish from `scale_latent(truth, 0.5)`: `is_stationary`; exactly one near-null direction, `normalized_eigenvalues[0] < 1e-6 · normalized_eigenvalues[-1]`; `cosine_with_raw_null_space(report, symmetry_generator(scale_latent, θ̂), 1) > 0.999`; the fit is *bad*: `‖β̂ − β_true‖ / ‖β_true‖ > 0.2` while the invariant `β̂ · sqrt(q̂)` matches `β_true · sqrt(q_true)` to 5%; `str(report)` names both `beta_real` and `process_noise_var` in the direction; `wald_standard_errors` for `process_noise_var[0]` exceeds its value by > 10× (SE inflation along the ridge) |
| `TestCouplingDegeneracy::test_lfp_anchored_is_identified` (slow) | polish `coupling_marginal_nll(base, spikes, lfp)` from the truth: `is_stationary`, `is_identifiable`, `n_negative_eigenvalues == 0`, `condition_number < 100`, `wald_standard_errors` all finite and positive; no warning; guard: same data, `lfp=None`, is flagged |
| `TestCouplingDegeneracy::test_lfp_anchored_wald_intervals_cover_truth` (slow) | 24 seeds, `T=3000`, base rate 0.2, `lfp_noise_var=0.25`: per seed polish from the truth and form `z = (θ̂ − θ_true)/SE` for the 7 coordinates; assert coverage of `|z| < 1.96` ≥ 0.85 over the 168 intervals and `mean(z²) ∈ [0.6, 1.6]`; every report `is_identifiable`; guard: every SE finite and positive |

Mark slow / integration tests explicitly: everything in `TestCouplingDegeneracy` except
the zero-coupling test, and the two polished LG-SSM tests, carry `@pytest.mark.slow`
(nothing here calls `.fit`, so the conftest auto-marker does not apply).

## Fixtures

- Fixed-seed synthetic data built inside the test module (module-scope fixtures):
  `gaussian_sample` (`n=50`, `rng(0)`), `collinear_design` (`n=40`, two identical columns
  plus `y`), `lgssm_data` (`T=60`, jax key 0, parameters above), `coupling_sim`
  (`simulate_coupling(coupling_params(), n_time=1000, seed=0)`) and, for the coverage
  test, a `seed`-parametrised generator using `simulate_coupling(coupling_params(base_rate=0.2),
  n_time=3000, seed=seed)`. Nothing is checked in; nothing from `conftest.py` is needed
  beyond x64.
- `coupling_params_small` / `simulated_coupling_small` (`conftest.py:788-826`) are *not*
  used: they have two bands and 5000 bins; the acceptance tests need one band (a single
  scale generator) and shorter sequences.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Additionally for this phase: every guard assertion in the slice (the "guard:" clauses)
  is present — they are what makes the flag tests non-vacuous; `uv run mypy` is clean
  with the module registered; the report contains no JAX arrays (`pickle.dumps(report)`
  round-trips in a test or in review).
