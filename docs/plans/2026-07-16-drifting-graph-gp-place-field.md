# Drifting Graph-GP Place Field Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover each neuron's time-varying, graph-respecting place-field trajectory `eta_{c,t} = Phi w_{c,t}` from position-aligned Poisson spikes. The drift scale `q_c` controls temporal flexibility and may be supplied or tuned; recovering its generating value is not an acceptance target.

**Architecture:** Reuse the existing point-process Laplace-EKF primitives (`point_process_kalman.py`), Kalman EM helpers (`kalman.py`), and the SGD mixin (`sgd_fitting.py`). The novelty is (1) a geometry-aware basis `Φ` (the smoothest Laplacian eigenvectors) already built by the Stage-0 substrate in `graph_place_field.py`, and (2) a shared *diagonal-in-the-eigenbasis* spectral shape `S = (κ²I + diag(λ))^(−α)` that ties the prior `P₀ = τ²S`, the initial covariance, and the per-neuron drift `Q_c = q_c·S` together, so "smoothness" reduces to a handful of tunable scalars (`τ²`, `κ²`, per-neuron `q_c`). `τ²` and `κ²` are learnable; `q_c` is fixed by default and its learning path is experimental (Amendment 1). Neurons are conditionally independent given `Φ`, so the E-step is a `jax.vmap` of the single-neuron Laplace-EKF (dimension `rank`) over neurons using the local masked full-grid filter described in Amendment 2.

**Tech Stack:** Python 3.10–3.12, JAX (x64), `neurospatial` (spatial substrate), `optax` (SGD), `scipy` (eigensolve, independent-optimizer parity check), `pytest` + `hypothesis`. Managed by `uv`.

---

## Implementation status & amendments (post-implementation reconciliation)

**Status:** Stages 0–2 (Tasks 1–8) are **implemented** on `feat/graph-place-field`; integrated with `master` at `3feccfc` on 2026-10-05. The committed code in `graph_place_field.py` / `test_graph_place_field.py` is authoritative; several task sections **below describe the original design and were superseded during implementation** — the amendments here take precedence over the task text and code blocks that follow.

1. **Latent field tracking is the primary acceptance target; `q_c` estimation is not.** The implementation must recover the time-varying log-rate field over time and every graph bin and substantially outperform both causal filtering and a single static field. `predict_log_rate_trajectory` exposes this primary posterior output without averaging away the drift. The synthetic tracking contract removes both the common rate baseline and each bin's temporal mean before scoring, so a static spatial pattern cannot pass. It uses the constructor's default `q_c`, deliberately 10x below the generating value, and validates three independent sessions after cold-start burn-in. `q_c` is fixed by default because the approximate Laplace-EKF marginal likelihood does not reliably identify its generating value: profiled with all else fixed, the marginal LL is monotone in `q_c` rather than peaking at the generating value, and the closed-form Gaussian M-step is a self-confirming fixed point (returns ~its initialization; the same is true of the sibling `PlaceFieldModel`). Enabling `update_drift_scale=True` (EM) or learning `q_c` via `fit_sgd` is an explicit, experimental opt-in and is **not** validated to recover the true drift scale. There is therefore no generating-`q_c` recovery test; instead, `test_default_model_tracks_drifting_field_over_time` validates full latent-trajectory recovery under timescale misspecification and `test_drift_scale_mstep_matches_closed_form` validates the optional M-step algebra. Future automatic timescale selection can use validation data or a redesigned likelihood without changing the trajectory-recovery goal.

2. **Architecture: a masked full-grid filter, not the library smoother.** The E-step, `fit`, `fit_sgd`, and `score` use a new local `_masked_graph_point_process_filter` (`glm_laplace_update` + `poisson_family`) plus `rts_backward_scan`, **not** `stochastic_point_process_filter/_smoother`. It runs each neuron on the **full** time grid: out-of-bounds trajectory rows advance the latent random walk but **skip their observation update** (rather than being dropped, as the original design had it). Row 0 is conditioned **directly on `(init_mean, P₀)`** (no predict-before-first-update; see the `point_process_kalman` docstring note documenting the library filter's differing convention). `_design_and_spikes` returns `(Z, spikes, valid)`.

3. **`update_kappa2` flag added** (default `True`): `κ²` is learnable by `fit_sgd` only (it reshapes `S` nonlinearly and has no EM M-step). `fit_sgd` raises if all `update_*` flags are `False`.

4. **Static-limit parity is APPROXIMATE, not exact.** The per-bin aggregated GLM = per-time static-limit *objective* is exact algebra (unchanged). But the sequential Laplace-EKF smoother reproduces the batch static-GLM **MAP** only approximately (about 1.8% coefficient relative-L2 error; field log-rate to ~6%). The parity test is non-circular (zero fixed prior mean, not warm-started to the static MAP) and asserts coefficient/field error, not correlation.

5. **Offline comparisons are DEFERRED, not complete.** The MRF / `non_local_detector` parity and the full spline-vs-graph (`PlaceFieldModel`) cross-arm head-to-head are **not implemented** in this branch; they remain offline-notebook follow-ups. The in-suite geometry test is the graph-vs-Euclidean leakage check (`test_graph_basis_does_not_smear_across_wmaze_arms`).

6. **CNM working-tree WIP excluded.** Unrelated `CorrelatedNoisePointProcessModel` changes that were in the working tree are stashed, not part of this branch.

7. **Current fitting contracts (2026-10-05).** The integrated model uses the shared `run_em` driver, typed `FittedAttribute` posteriors, and `_record_fit_result` / `_clear_fit_state` for fit metadata. `fit_sgd` is a forwarding wrapper; data validation and initialization happen in `_prepare_sgd_data` after optimizer settings are validated. `_finalize_sgd` returns the marginal log-likelihood, and `_sgd_param` keeps repeated fits from invalidating the compiled-step cache. Invalid SGD settings/data preserve an earlier fit; a non-finite fit clears posteriors and shared results. Unfitted predictions raise `NotFittedError`; unusable likelihoods raise `NonFiniteLikelihoodError`. The historical task snippets below predate these contracts. The `spatial` extra now resolves the git commit pinned in `uv.lock`, rather than an editable sibling checkout. Both topology-collision and same-size edge-reweighting fingerprint regression tests are retained.

8. **Evidence search, diagnostics, and compilation (2026-10-05 review fixes).** Evidence can have multiple local maxima. `select_tau2_by_evidence` now samples 65 logarithmically spaced amplitudes, refines each candidate peak/boundary basin, and compares all evaluated values including the exact interval endpoints. This supersedes Task 2's single bounded-search sketch; the finite grid is a practical search, not a guarantee for arbitrarily narrow peaks. The static Newton solve and masked filter use stable module-level JIT functions with changing parameters passed as inputs. Repeated E-steps reuse compilation, including after parameter changes. The filter reports exhausted line searches through the existing host-side logging policy from eager, jit, and gradient paths. Failure fractions use observed rows; masked rows and fully unobserved grids do not cause spurious warnings. Regression coverage includes multimodal evidence, boundary optima, one-/two-neuron EM/scoring/SGD diagnostics, masking, and compilation reuse.

9. **Independent evidence audit (2026-10-06).** The independent blocked-holdout experiment below demonstrates retrospective tracking beyond eigenbasis-generated truth. It also exposes limits: a simple time-varying occupancy map outperforms the tested graph configurations on smooth/sparse drift, static controls favor a static estimator, and nominal uncertainty intervals under-cover changing truth. Implementation completion is not evidence of superiority or calibrated biological drift inference. Fitted branching-track comparisons, real recordings, and causal forecasting remain open.


---

## Integration validation (2026-10-05)

Validated with Python 3.11, x64 enabled, and the `neurospatial` git commit pinned
in `uv.lock` (`uv sync --locked --extra test --extra coupling --extra spatial`).
The editable sibling checkout is not used for this validation.

- Graph and fitting-contract slice: **97 passed**, including every slow graph
  test, three-seed drift tracking, fit metadata, failed-fit cleanup, and repeated
  SGD step reuse, evidence-peak comparisons, masked line-search diagnostics, and
  E-step compilation reuse. Command: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py src/state_space_practice/tests/test_fit_contract.py -k "graph or Graph" -q`.
- Package fast suite: **2430 passed, 1 skipped**. Command:
  `LOKY_MAX_CPU_COUNT=4 uv run --no-sync pytest -m "not slow" -q`. The worker
  limit prevents joblib's physical-core detection warning in the restricted
  macOS execution environment.
- Ruff lint and formatting checks pass for `src/`, `notebooks/`, and `scripts/`.
- Whole-package mypy passes for all **47 source files**. The graph module also
  passes with an untyped spatial dependency, matching CI without the spatial extra.

The field-recovery acceptance thresholds were preserved. External estimator
comparisons and Stage 3 remain deferred as described above and below.

---

## Evidence audit and independent validation (2026-10-06)

Implementation completion establishes the retained contracts; it does not establish
general usefulness on real recordings. The original evidence has distinct scopes:

| Evidence | What it establishes | Limit of the claim |
| --- | --- | --- |
| Newton MAP versus independent SciPy optimizer | Correct optimization of the static penalized objective | Both solve the same assumed objective; this is numerical verification, not a generalization benchmark |
| Static eigenbasis-field recovery | Recovery under dense observations | Truth uses the fitted basis, with uniform 300-second exposure per bin; correlation alone does not validate absolute rates or uncertainty |
| Independent W-maze simulator, six cells | Static recovery beyond eigenbasis-generated truth | One seed, visited bins only, median correlation threshold of 0.6; this is modest static evidence |
| Three-seed full-field drift tracking | Temporal spatial structure is recovered after removing baselines, even with a misspecified drift scale | Eight-mode truth uses the model's spectral random walk, random spatial sampling and a 30-Hz baseline; the test fixes several parameters and is not a test of the complete default fitting pipeline |
| Cross-arm delta reconstruction | The truncated graph basis respects topology on the selected bin pair | No spike fitting and no actual spline/MRF comparison; it does not demonstrate a predictive advantage on branching tracks |
| Held-out aligned counts beat shifted counts | `score` consumes the supplied observations | No model comparison or hyperparameter-selection evaluation; training likelihood improvement also does not establish generalization |

The next evidence should remove the shared generative parameterization, hide actual
observations, compare stronger alternatives, include a no-drift control, and measure
absolute error and uncertainty. The reproducible experiment in
[`notebooks/graph_place_field_validation.py`](../../notebooks/graph_place_field_validation.py)
implements this for a **linear track**:

- Analytic Gaussian firing fields (12-cm width), never generated from eigenvectors
  or spectral increments. Animal movement is continuous reflected laps with variable
  speed and repeated pauses. Conditions are stable, smoothly moving, abrupt remapping,
  and sparse firing (3-Hz peak increment, rather than 12 Hz).
- Each session lasts 400 seconds at 100-ms resolution. Whole one-second blocks are
  assigned 60/20/20 to training, validation and test. Withheld observations are masked
  and counts are zeroed during fitting; their time rows still propagate dynamics.
  The first 40 seconds are excluded from evaluation. Regression tests poison masked
  counts and confirm both EM and SGD fields remain unchanged while changing observed
  counts changes the fit.
- Graph rank/drift, the static graph penalty, and the widths of a simple time-varying
  occupancy-map baseline are selected with **validation counts only**. The graph grid
  has ranks 8/16 and drift scales 1e-10/1e-3/1e-2/1e-1, with fixed amplitude 100 and
  kappa2 1. Static amplitudes are 1/100. Occupancy-map time widths are 10/30 seconds
  and spatial widths 5/10 cm. A separate model uses constructor defaults, full rank,
  and the normal 100-iteration EM limit, with convergence recorded, so tuned
  performance is not confused with default behavior.
- Final test counts are used after all selections. Pointwise predictive log scores
  integrate the graph/static Gaussian log-rate uncertainty using 256-node quadrature;
  the occupancy map uses a plug-in Poisson density. These are **retrospective missing-
  observation predictions**, with neighboring training data on both sides, not causal
  forecasts or a joint held-out marginal likelihood. Full-field log-rate RMSE and
  pointwise 95% log-rate coverage are measured against analytic truth. Continuous-
  position and bin-center coverage are recorded separately to expose discretization.
- Pilot seed 0 was used for a smoke run. The final experiment uses ten fresh seeds
  (10–19) per condition, with fixed grids and no threshold adjustments after viewing
  their test results. An initial ten-iteration default-setting run hit its budget;
  a representative-seed diagnostic prompted extending that run to the API's normal
  100-iteration limit for all sessions, without changing candidate grids. This is
  an exploratory audit, not a preregistered confirmatory study. Paired bootstrap
  intervals resample sessions, not correlated time bins, and are descriptive
  summaries of this synthetic experiment.

Run from the repository root after installing the pinned spatial/test extras:

```bash
MPLCONFIGDIR=/private/tmp/ssp-matplotlib LOKY_MAX_CPU_COUNT=4 \
  uv run --no-sync python notebooks/graph_place_field_validation.py \
  --output-dir docs/validation/graph-place-field
```

This experiment still leaves the fitted cross-arm graph-versus-spline/MRF comparison,
longer gaps, broader movement/field families, causal forecasting and real-data
validation open. The existing real-data notebook uses splines, and its local `data/`
directory is absent from this checkout; it supplies no evidence for this graph model.

### Results and interpretation

The forty-session run completed with the pinned spatial dependency. Full session
records and validation grids are in
[`results.json`](../validation/graph-place-field/results.json). The command also
generates representative seed-10 field reconstructions at
`docs/validation/graph-place-field/fields.png`; PNG outputs are ignored by Git.
Gains below are mean test predictive bits per spike; positive values favor the graph
model. Each condition has ten independent sessions.

| Condition | Tuned graph vs static | Tuned graph vs windowed map | Default graph vs static | Tuned graph nominal 95% coverage |
| --- | ---: | ---: | ---: | ---: |
| Stable | -0.024 | -0.008 | -0.040 | 94.2% |
| Smooth drift | +0.370 | -0.095 | +0.154 | 72.1% |
| Abrupt remap | +0.501 | -0.010 | +0.440 | 89.9% |
| Sparse drift | +0.157 | -0.176 | -0.066 | 71.4% |

- The graph model recovers changing fields outside its exact generative assumptions,
  with predictive improvements over a single static map in the changing-field cases.
  Tuned graph models beat static on 9/10 smooth, 10/10 remapping and 9/10 sparse
  sessions. This establishes useful retrospective tracking for these scenarios.
- A stronger comparison changes the conclusion: the simple windowed occupancy map
  scores better than the tuned graph model in 9/10 smooth and 9/10 sparse sessions.
  Paired descriptive bootstrap intervals for the graph-minus-windowed gain are
  [-0.208, -0.027] and [-0.257, -0.092] bits/spike, respectively. Remapping has no
  clear advantage either way ([-0.027, +0.003], five wins each). Full-field RMSE
  corroborates this: graph/windowed errors are 0.661/0.519 nats for smooth drift,
  0.833/0.646 for sparse drift and 0.527/0.521 for remapping. This experiment does
  **not establish superiority over a simple time-varying estimator**.
- Defaults are weak in the sparse case: mean gain over static is -0.066 bits/spike,
  and static wins 9/10 sessions. Selected settings consistently favor rank 8 and
  q=1e-2 for moving fields; defaults use all 41 modes and q=1e-3. This points toward
  validation-based regularization/timescale selection rather than relying on the
  approximate training likelihood to learn q. All forty default-setting runs
  converged within the normal 100-iteration budget (mean accepted E-step counts
  12.2–19.5 across conditions). The initial budget check is retained separately in
  [`default_convergence_checks.json`](../validation/graph-place-field/default_convergence_checks.json);
  convergence did not resolve the baseline or coverage gaps. The tuned/default
  comparison also differs in amplitude and initialization updates, so it does not
  isolate which setting causes the performance difference.
- The no-drift control favors static: it beats tuned graph in 9/10 sessions and
  defaults in all ten. Tuned estimates have mean temporal spatial fluctuation RMS
  of 0.235 log-rate nats despite constant truth. A varying posterior mean alone is
  insufficient evidence for biological drift; model selection against a static
  null and explicit false-positive checks are needed.
- Nominal 95% log-rate intervals substantially under-cover changing truth. Defaults
  are worse (48.1% smooth, 74.4% remapping, 58.2% sparse). Bin-center coverage is
  nearly identical, so nearest-bin discretization does not explain the gap. These
  conditional Laplace intervals are not validated for scientific uncertainty claims
  under model misspecification or parameter selection.

Priority follow-ups are (1) validation-based selection including a static null and
spatial regularization, (2) calibrated uncertainty and static-control drift detection,
(3) fitted comparisons on branching tracks where graph topology should matter, and
(4) blocked real-recording validation and separate causal forecasting experiments.
Keep this independent benchmark fixed when changing inference. Any further tuning
should use validation data and a fresh final evaluation, rather than adapting the
acceptance thresholds to these forty sessions.

Verification: graph/fitting-contract tests now **99 passed**, including the new
EM/SGD leakage checks. The predictive scorer matches independent adaptive integration
across 48 count/rate/variance combinations (maximum absolute log-score error 1.3e-8),
and its zero-variance limit matches the Poisson density. These checks validate the
evaluation machinery; they do not remove the empirical limitations above.

## Global Constraints

Every task's requirements implicitly include this section. Values copied verbatim from the spec and repo `CLAUDE.md`.

- **x64 is mandatory.** The Laplace-EKF covariance propagation NaNs in float32 on long sequences. Tests inherit x64 from `tests/conftest.py` (`jax.config.update("jax_enable_x64", True)` at line 8). Any standalone script must set x64 **before** importing `jax.numpy` or this package.
- **`neurospatial` dependency.** It is declared in the `spatial` optional extra and pinned to a git commit in `pyproject.toml` / `uv.lock`. Set up the environment with `uv sync --locked --extra test --extra coupling --extra spatial`. Do **not** replace the pinned dependency with an editable sibling checkout when validating this plan.
- **Tests importing `neurospatial` must `pytest.importorskip("neurospatial")`** at module top (see `tests/test_graph_place_field.py:13`), so the fast suite still collects when the `spatial` extra is absent.
- **`α` (smoothness exponent) is a fixed hyperparameter, default `1.0`.** It is never an EM scalar step. `κ²` is fit by SGD or an outer optimizer only, never inside the EM M-step (it reshapes `S` nonlinearly).
- **Per-neuron `vmap`.** Use the graph model's local masked full-grid filter with independent per-neuron `q_c` values. Preserve its row-zero prior and missing-observation conventions (Amendment 2).
- **`PlaceFieldModel` is not modified.** All new code lives in `graph_place_field.py` and its test file.
- **Behavioral assertions only** (recovery, ordering, LL improvement), per repo testing guidance — not shape/type checks on deterministic constructors. Every test must be able to fail.
- **Mark any test that runs EM, SGD, or a full filter/smoother pipeline `@pytest.mark.slow`.** The fast suite (`-m "not slow"`) must finish in under a minute.
- **Run all tooling through `uv run`** so it uses the locked `.venv` (add `--no-sync` to skip re-resolution): e.g. `uv run --no-sync pytest ...`, `uv run --no-sync ruff format src/`.
- **Ruff formatting of whole edited files is acceptable here** even when it churns pre-existing lines.

---

## Pre-flight (not a task)

Stage 0 (the substrate seam) is **already complete** in `graph_place_field.py`: `GraphBasis`, `build_graph_laplacian`, `build_graph_basis`, `spectral_shape`, `graph_design_matrix`, `bin_occupancy`, `bin_spike_counts`, `validate_graph_laplacian`, plus 30 contract tests in `test_graph_place_field.py`. This plan builds Stages 1–2 on top of it and leaves Stage 3 deferred.

Before starting, note the working tree currently has **unrelated** uncommitted changes (`point_process_models.py`, `test_point_process_models.py` — oscillator/CNM work; and in-flight substrate polish in `graph_place_field.py`/`test_graph_place_field.py` adding the `inverse_distance` convention). Per commit discipline, commit or stash those on their own branch **before** starting Task 1, so the graph-model commits stay self-contained. Verify with:

```bash
uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -q
```

Expected: all Stage-0 contract tests pass (they need the `spatial` extra; run `uv sync --extra test --extra spatial` first if `neurospatial` is missing).

---

## File Structure

- **`src/state_space_practice/graph_place_field.py`** (modify — append only; do not touch the Stage-0 functions above): the Stage-1 static estimator (`spectral_precision`, `parity_penalty`, `fit_static_graph_glm`, `static_log_evidence`, `select_tau2_by_evidence`) and the Stage-2 `GraphPlaceFieldModel` class. One module keeps the substrate and the model that consumes it together, matching the spec ("New standalone model class ... in a new module `graph_place_field.py`").
- **`src/state_space_practice/tests/test_graph_place_field.py`** (modify — append): Stage-1 and Stage-2 tests alongside the existing Stage-0 contract tests.
- No change to `__init__.py` (the package exposes modules by fully-qualified import, e.g. `from state_space_practice.graph_place_field import GraphPlaceFieldModel`; there is no top-level re-export convention — `PlaceFieldModel` is imported the same way).
- No change to `pyproject.toml` (the `spatial` extra already exists).

---

# Stage 1 — Static graph-GP field

The static (drift-free) estimator: a penalized Poisson GLM in the eigenbasis with occupancy as a log-offset. It is the `Q → 0` limit of the drifting model, the MRF-parity target, and the warm-start for Stage 2. Because grouping the per-time Poisson likelihood by bin gives `Σ_i [count_i·(Φ_i w) − occ_i·exp(Φ_i w)] = Σ_i [count_i·(Φ_i w) − exp(Φ_i w + log occ_i)]`, the **per-bin aggregated GLM with a `log(occupancy)` offset is exactly the per-time static limit** — this is the bridge the parity test rests on.

## Task 1: Spectral penalty builders + per-bin penalized Poisson solver

**Files:**
- Modify: `src/state_space_practice/graph_place_field.py` (append after the Stage-0 functions; extend `__all__`)
- Test: `src/state_space_practice/tests/test_graph_place_field.py` (append)

**Interfaces:**
- Consumes: `spectral_shape(eigvals, kappa2, alpha)` and `GraphBasis` (already in this module); `psd_solve`, `symmetrize` from `state_space_practice.utils`.
- Produces:
  - `spectral_precision(eigvals: NDArray, tau2: float, kappa2: float, alpha: float = 1.0) -> NDArray` — diagonal prior precision `(κ²+λ)^α / τ²`, shape `(rank,)`.
  - `parity_penalty(eigvals: NDArray, n_components: int) -> NDArray` — pure `diag(λ)` penalty with the first `n_components` (null) modes set to `0.0`, shape `(rank,)`.
  - `fit_static_graph_glm(counts: ArrayLike, occupancy: ArrayLike, eigvecs: ArrayLike, penalty_diag: ArrayLike, *, max_iter: int = 25) -> tuple[Array, Array]` — returns `(weights, cov)` where `weights` is `(rank,)` (single neuron) or `(n_neurons, rank)`, and `cov` is the Laplace posterior covariance `(rank, rank)` or `(n_neurons, rank, rank)`.

- [x] **Step 1: Write the failing tests**

Append to `test_graph_place_field.py`:

```python
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from state_space_practice.graph_place_field import (  # noqa: E402
    fit_static_graph_glm,
    parity_penalty,
    spectral_precision,
)


def test_spectral_precision_matches_inverse_prior():
    eigvals = np.array([0.0, 0.5, 2.0])
    tau2, kappa2, alpha = 0.7, 0.25, 1.0
    prec = spectral_precision(eigvals, tau2=tau2, kappa2=kappa2, alpha=alpha)
    # precision is the reciprocal of the prior variance tau2 * S
    from state_space_practice.graph_place_field import spectral_shape

    S = spectral_shape(eigvals, kappa2, alpha)
    np.testing.assert_allclose(prec, 1.0 / (tau2 * S))
    assert np.all(np.isfinite(prec))  # null mode finite (kappa2 > 0)


def test_parity_penalty_leaves_null_modes_unpenalized():
    eigvals = np.array([0.0, 0.0, 0.8, 3.0])  # two null modes
    pen = parity_penalty(eigvals, n_components=2)
    assert np.all(pen[:2] == 0.0)  # per-component intercepts, unpenalized
    np.testing.assert_allclose(pen[2:], eigvals[2:])  # positive modes penalized by lambda


def test_static_glm_recovers_smooth_field(small_grid_env):
    # Simulate per-bin Poisson counts from a KNOWN smooth field in the eigenbasis,
    # then check the penalized GLM recovers the field (log-rate) it was generated from.
    # NOTE: small_grid_env is only ~25 bins; rank and exposure are calibrated so a
    # correct solver clears corr > 0.95 with margin (empirically mean 0.99 / min 0.97
    # across seeds) while a broken gradient/Hessian still fails. Offset-bug sensitivity
    # is covered separately by the independent-optimizer parity test (Task 3), which
    # uses the real log(occupancy) offset; with uniform occupancy a correlation test
    # cannot see offset bugs.
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=6)
    rng = np.random.default_rng(0)
    # A smooth field: energy only in the low modes.
    w_true = np.zeros(6)
    w_true[:4] = rng.standard_normal(4)
    eta_true = basis.eigvecs @ w_true  # (n_bins,) log-rate
    occ = np.full(small_grid_env.n_bins, 300.0)  # uniform exposure (s) per bin
    counts = rng.poisson(np.exp(eta_true) * occ)

    # Weak ridge on the rough modes; recovery should be accurate where data is dense.
    prec = spectral_precision(basis.eigvals, tau2=100.0, kappa2=1e-2, alpha=1.0)
    w_hat, cov = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    eta_hat = basis.eigvecs @ np.asarray(w_hat)
    # Correlation between recovered and true log-rate is high.
    corr = np.corrcoef(eta_hat, eta_true)[0, 1]
    assert corr > 0.95
    # Laplace covariance is PSD.
    assert np.all(np.linalg.eigvalsh(np.asarray(cov)) > -1e-8)


def test_static_glm_ignores_unvisited_bins(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=8)
    occ = np.zeros(small_grid_env.n_bins)
    occ[: small_grid_env.n_bins // 2] = 3.0  # only half the bins visited
    counts = np.zeros(small_grid_env.n_bins)
    counts[: small_grid_env.n_bins // 2] = 1.0
    prec = spectral_precision(basis.eigvals, tau2=10.0, kappa2=1.0, alpha=1.0)
    w_hat, _ = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    # Unvisited bins must not blow up the fit (finite, no NaN).
    assert np.all(np.isfinite(np.asarray(w_hat)))


def test_static_glm_multineuron_matches_per_neuron(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=8)
    rng = np.random.default_rng(1)
    occ = np.full(small_grid_env.n_bins, 4.0)
    counts = rng.poisson(0.3, size=(small_grid_env.n_bins, 3)).astype(float)
    prec = spectral_precision(basis.eigvals, tau2=10.0, kappa2=1.0, alpha=1.0)
    w_multi, _ = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    assert w_multi.shape == (3, 8)
    for j in range(3):
        w_j, _ = fit_static_graph_glm(counts[:, j], occ, basis.eigvecs, prec)
        np.testing.assert_allclose(np.asarray(w_multi[j]), np.asarray(w_j), atol=1e-8)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "spectral_precision or parity_penalty or static_glm" -q`
Expected: FAIL with `ImportError: cannot import name 'fit_static_graph_glm'`.

- [x] **Step 3: Implement the penalty builders and solver**

Append to `graph_place_field.py` (add `import jax`, `import jax.numpy as jnp`, `from jax import Array`, `from jax.typing import ArrayLike`, and `from state_space_practice.kalman import psd_solve, symmetrize` to the imports; add the three new names to `__all__`):

```python
def spectral_precision(
    eigvals: NDArray[np.float64],
    tau2: float,
    kappa2: float,
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Diagonal prior precision ``P0^{-1} = diag((kappa2 + lambda)^alpha / tau2)``.

    The reciprocal of the prior variance ``tau2 * S`` with ``S`` the spectral shape.
    Finite at the null modes because ``kappa2 > 0``.
    """
    if not tau2 > 0:
        raise ValueError(f"tau2 (prior amplitude) must be positive, got {tau2}.")
    shape = spectral_shape(eigvals, kappa2, alpha)  # validates kappa2, alpha > 0
    return 1.0 / (tau2 * shape)


def parity_penalty(
    eigvals: NDArray[np.float64], n_components: int
) -> NDArray[np.float64]:
    """Pure ``diag(lambda)`` penalty with the null modes left unpenalized.

    The first ``n_components`` entries (the per-component null modes, ordered first by
    :func:`build_graph_basis`) are set to zero so they act as unpenalized per-component
    intercepts. This is the MRF-parity configuration: the penalty is the eigenvalues
    themselves and the REML counterpart counts only positive eigenvalues.
    """
    penalty = np.array(eigvals, dtype=float)
    if n_components < 0 or n_components > penalty.shape[0]:
        raise ValueError(
            f"n_components={n_components} out of range for {penalty.shape[0]} modes."
        )
    penalty[:n_components] = 0.0
    return penalty


def fit_static_graph_glm(
    counts: ArrayLike,
    occupancy: ArrayLike,
    eigvecs: ArrayLike,
    penalty_diag: ArrayLike,
    *,
    max_iter: int = 25,
) -> tuple[Array, Array]:
    """Penalized per-bin Poisson GLM in the eigenbasis (Newton / Fisher scoring).

    Fits ``count_i ~ Poisson(exp(Phi_i w) * occ_i)`` with a diagonal quadratic penalty
    ``0.5 * w^T diag(penalty_diag) w``. Grouping the per-time Poisson likelihood by bin
    with a ``log(occupancy)`` offset makes this the exact ``Q -> 0`` static limit of the
    drifting model, and the MRF-parity target.

    Parameters
    ----------
    counts : ArrayLike, shape (n_bins,) or (n_bins, n_neurons)
        Per-active-bin spike counts (from :func:`bin_spike_counts`).
    occupancy : ArrayLike, shape (n_bins,)
        Per-active-bin exposure in seconds (from :func:`bin_occupancy`).
    eigvecs : ArrayLike, shape (n_bins, rank)
        The basis ``Phi`` (``GraphBasis.eigvecs``).
    penalty_diag : ArrayLike, shape (rank,)
        Diagonal penalty ``P0^{-1}`` (from :func:`spectral_precision` or
        :func:`parity_penalty`).
    max_iter : int, optional
        Newton iterations (the objective is convex; ~15 suffice), by default 25.

    Returns
    -------
    weights : Array, shape (rank,) or (n_neurons, rank)
        MAP coefficients.
    cov : Array, shape (rank, rank) or (n_neurons, rank, rank)
        Laplace posterior covariance (inverse Fisher + penalty) at the MAP.
    """
    Phi = jnp.asarray(eigvecs)
    occ = jnp.asarray(occupancy)
    counts_arr = jnp.asarray(counts)
    single = counts_arr.ndim == 1
    counts_2d = counts_arr[:, None] if single else counts_arr
    rank = Phi.shape[1]
    Lam = jnp.diag(jnp.asarray(penalty_diag))
    eye = jnp.eye(rank)
    visited = occ > 0
    # log-offset; unvisited bins contribute nothing (mu forced to 0 there).
    log_occ = jnp.where(visited, jnp.log(jnp.where(visited, occ, 1.0)), 0.0)

    def _fit_one(y: Array) -> tuple[Array, Array]:
        y = y.astype(Phi.dtype)

        def _step(w: Array, _: None) -> tuple[Array, None]:
            mu = jnp.where(visited, jnp.exp(Phi @ w + log_occ), 0.0)
            grad = Phi.T @ (mu - y) + Lam @ w
            hess = Phi.T @ (mu[:, None] * Phi) + Lam
            return w - psd_solve(hess, grad), None

        w, _ = jax.lax.scan(_step, jnp.zeros(rank), None, length=max_iter)
        mu = jnp.where(visited, jnp.exp(Phi @ w + log_occ), 0.0)
        cov = psd_solve(Phi.T @ (mu[:, None] * Phi) + Lam, eye)
        return w, symmetrize(cov)

    weights, cov = jax.vmap(_fit_one, in_axes=1)(counts_2d)
    if single:
        return weights[0], cov[0]
    return weights, cov
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "spectral_precision or parity_penalty or static_glm" -q`
Expected: PASS (5 tests).

- [x] **Step 5: Format and commit**

```bash
uv run --no-sync ruff format src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git add src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git commit -m "feat(graph-pp): static penalized-Poisson estimator in the eigenbasis"
```

---

## Task 2: Amplitude (τ²) selection by Laplace evidence

**Files:**
- Modify: `src/state_space_practice/graph_place_field.py` (append; extend `__all__`)
- Test: `src/state_space_practice/tests/test_graph_place_field.py` (append)

**Interfaces:**
- Consumes: `fit_static_graph_glm`, `spectral_precision`, `spectral_shape`, `GraphBasis`.
- Produces:
  - `static_log_evidence(counts, occupancy, eigvecs, eigvals, *, tau2, kappa2, alpha=1.0) -> float` — the Laplace log-evidence (summed over neurons), constant `log(count!)` terms dropped.
  - `select_tau2_by_evidence(counts, occupancy, basis, *, kappa2, alpha=1.0, bounds=(1e-4, 1e4)) -> float` — selects `τ²` by comparing a log-space grid, refined candidate peaks, and the interval endpoints (Amendment 8).

- [x] **Step 1: Write the failing tests**

```python
from state_space_practice.graph_place_field import (  # noqa: E402
    select_tau2_by_evidence,
    static_log_evidence,
)


def _simulate_smooth_counts(basis, rng, tau2_gen, occ_val=5.0):
    """Per-bin counts from w ~ N(0, tau2_gen * S), rate = exp(Phi w)."""
    from state_space_practice.graph_place_field import spectral_shape

    S = spectral_shape(basis.eigvals, kappa2=1e-2, alpha=1.0)
    rank = basis.eigvecs.shape[1]
    w = rng.standard_normal(rank) * np.sqrt(tau2_gen * S)
    eta = basis.eigvecs @ w
    n_bins = basis.eigvecs.shape[0]
    occ = np.full(n_bins, occ_val)
    counts = rng.poisson(np.exp(eta) * occ).astype(float)
    return counts, occ


def test_evidence_is_maximized_near_selected_tau2(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=12)
    rng = np.random.default_rng(3)
    counts, occ = _simulate_smooth_counts(basis, rng, tau2_gen=1.0)
    tau2_hat = select_tau2_by_evidence(counts, occ, basis, kappa2=1e-2, alpha=1.0)
    ev_at = static_log_evidence(
        counts, occ, basis.eigvecs, basis.eigvals, tau2=tau2_hat, kappa2=1e-2
    )
    # The selected tau2 beats both a 10x-too-large and a 10x-too-small amplitude.
    ev_hi = static_log_evidence(
        counts, occ, basis.eigvecs, basis.eigvals, tau2=10 * tau2_hat, kappa2=1e-2
    )
    ev_lo = static_log_evidence(
        counts, occ, basis.eigvecs, basis.eigvals, tau2=0.1 * tau2_hat, kappa2=1e-2
    )
    assert ev_at >= ev_hi
    assert ev_at >= ev_lo
    assert 1e-4 < tau2_hat < 1e4  # inside the search bounds (not railed)
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "evidence" -q`
Expected: FAIL with `ImportError: cannot import name 'select_tau2_by_evidence'`.

- [x] **Step 3: Implement the evidence and selection**

Append to `graph_place_field.py` (add `import scipy.optimize` to imports; add both names to `__all__`):

```python
def static_log_evidence(
    counts: ArrayLike,
    occupancy: ArrayLike,
    eigvecs: ArrayLike,
    eigvals: ArrayLike,
    *,
    tau2: float,
    kappa2: float,
    alpha: float = 1.0,
) -> float:
    """Laplace log-evidence of the static graph GLM at amplitude ``tau2``.

    ``log Z(tau2) ~ [y.eta - mu] - 0.5 w^T Lam w + 0.5 logdet(Lam) - 0.5 logdet(H)``
    evaluated at the MAP ``w``, with ``Lam = P0^{-1}`` the spectral precision and
    ``H = Phi^T diag(mu) Phi + Lam``. Constant ``log(count!)`` terms are dropped (they
    do not depend on ``tau2``). Summed over neurons for multi-neuron ``counts``.
    """
    Phi = jnp.asarray(eigvecs)
    occ = jnp.asarray(occupancy)
    counts_arr = jnp.asarray(counts)
    single = counts_arr.ndim == 1
    counts_2d = counts_arr[:, None] if single else counts_arr
    prec = jnp.asarray(spectral_precision(np.asarray(eigvals), tau2, kappa2, alpha))
    Lam = jnp.diag(prec)
    weights, _ = fit_static_graph_glm(counts_arr, occ, Phi, prec)
    weights_2d = weights[None, :] if single else weights
    visited = occ > 0
    log_occ = jnp.where(visited, jnp.log(jnp.where(visited, occ, 1.0)), 0.0)
    # logdet(Lam) is constant across neurons; sign of prec is positive.
    logdet_lam = jnp.sum(jnp.log(prec))

    def _one(w: Array, y: Array) -> Array:
        eta = Phi @ w + log_occ
        mu = jnp.where(visited, jnp.exp(eta), 0.0)
        data_term = jnp.sum(jnp.where(visited, y * eta - mu, 0.0))
        hess = Phi.T @ (mu[:, None] * Phi) + Lam
        _, logdet_h = jnp.linalg.slogdet(hess)
        return data_term - 0.5 * (w @ (prec * w)) + 0.5 * logdet_lam - 0.5 * logdet_h

    ev = jax.vmap(_one, in_axes=(0, 1))(weights_2d, counts_2d.astype(Phi.dtype))
    return float(jnp.sum(ev))


def select_tau2_by_evidence(
    counts: ArrayLike,
    occupancy: ArrayLike,
    basis: GraphBasis,
    *,
    kappa2: float,
    alpha: float = 1.0,
    bounds: tuple[float, float] = (1e-4, 1e4),
) -> float:
    """Return the ``tau2`` maximizing :func:`static_log_evidence`.

    Historical single-basin search; superseded by the multi-peak grid search in
    Amendment 8. Evidence is not necessarily unimodal.
    """
    lo, hi = np.log(bounds[0]), np.log(bounds[1])

    def _neg_ev(log_tau2: float) -> float:
        return -static_log_evidence(
            counts,
            occupancy,
            basis.eigvecs,
            basis.eigvals,
            tau2=float(np.exp(log_tau2)),
            kappa2=kappa2,
            alpha=alpha,
        )

    result = scipy.optimize.minimize_scalar(
        _neg_ev, bounds=(lo, hi), method="bounded"
    )
    return float(np.exp(result.x))
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "evidence" -q`
Expected: PASS.

- [x] **Step 5: Format and commit**

```bash
uv run --no-sync ruff format src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git add src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git commit -m "feat(graph-pp): Laplace-evidence amplitude selection for the static field"
```

---

## Task 3: Static-field validation — independent-optimizer parity + W-maze recovery (slow)

**Files:**
- Test: `src/state_space_practice/tests/test_graph_place_field.py` (append)

No new source. This task validates Task 1/2 against (a) an independent optimizer on the identical convex objective (catches solver/offset/penalty bugs), and (b) ground-truth place cells simulated by `neurospatial` on the primary W-maze environment.

**Interfaces:**
- Consumes: `fit_static_graph_glm`, `spectral_precision`, `select_tau2_by_evidence`, `bin_spike_counts`, `bin_occupancy`, `build_graph_basis`; `neurospatial.simulation.simulate_session`; `state_space_practice.preprocessing.bin_spike_times`.

- [x] **Step 1: Write the failing tests**

```python
import pytest  # (already imported at top; shown for clarity)


@pytest.mark.slow
def test_newton_map_matches_independent_optimizer(small_grid_env):
    """Our Newton MAP equals a scipy L-BFGS optimum on the identical penalized NLL."""
    import scipy.optimize

    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=10)
    rng = np.random.default_rng(5)
    occ = np.full(small_grid_env.n_bins, 4.0)
    w_true = np.zeros(10)
    w_true[:4] = rng.standard_normal(4)
    counts = rng.poisson(np.exp(basis.eigvecs @ w_true) * occ).astype(float)
    prec = spectral_precision(basis.eigvals, tau2=50.0, kappa2=1e-2, alpha=1.0)

    Phi = np.asarray(basis.eigvecs)
    log_occ = np.log(occ)

    def nll(w):
        eta = Phi @ w + log_occ
        return float(np.sum(np.exp(eta) - counts * eta) + 0.5 * np.sum(prec * w**2))

    def grad(w):
        eta = Phi @ w + log_occ
        mu = np.exp(eta)
        return Phi.T @ (mu - counts) + prec * w

    ref = scipy.optimize.minimize(nll, np.zeros(10), jac=grad, method="L-BFGS-B")
    w_hat, _ = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    np.testing.assert_allclose(np.asarray(w_hat), ref.x, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_static_field_recovers_wmaze_place_cells(w_maze_env):
    """Recovered rate map correlates with the true place fields on the W-maze."""
    from neurospatial.simulation import simulate_session

    from state_space_practice.graph_place_field import build_graph_basis
    from state_space_practice.preprocessing import bin_spike_times

    session = simulate_session(
        w_maze_env, duration=600.0, n_cells=6, seed=7, show_progress=False
    )
    dt = float(np.median(np.diff(session.times)))
    # Bin each cell's spike times onto the position time grid.
    counts_tc = bin_spike_times(session.spike_trains, session.times).astype(float)

    basis = build_graph_basis(w_maze_env, sigma=15.0)  # bandwidth-truncated rank
    counts_bin = bin_spike_counts(
        w_maze_env, counts_tc, session.times, session.positions, basis
    )
    occ = bin_occupancy(w_maze_env, session.times, session.positions, dt)
    tau2 = select_tau2_by_evidence(counts_bin, occ, basis, kappa2=1e-2)
    prec = spectral_precision(basis.eigvals, tau2=tau2, kappa2=1e-2)
    w_hat, _ = fit_static_graph_glm(counts_bin, occ, basis.eigvecs, prec)

    bin_centers = np.asarray(w_maze_env.bin_centers)
    visited = occ > 0
    corrs = []
    for j, model in enumerate(session.models):
        rate_true = np.asarray(model.firing_rate(bin_centers))
        rate_hat = np.exp(np.asarray(basis.eigvecs @ w_hat[j]))
        corrs.append(np.corrcoef(rate_hat[visited], rate_true[visited])[0, 1])
    # The graph-GP field tracks the true place fields on the maze.
    assert np.median(corrs) > 0.6
```

- [x] **Step 2: Run to verify they fail**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "independent_optimizer or wmaze_place_cells" -q`
Expected: both tests present and FAIL only on the assertions if thresholds are wrong; if a helper name is off they error. (If `neurospatial` is missing they SKIP — install `--extra spatial`.)

Note on thresholds: the correlation floor (`0.6`) is deliberately conservative for a truncated basis + 600 s of OU coverage on a branching maze. If it fails on the real run, first confirm coverage (`(occ > 0).mean()`) and raise `duration`/lower `sigma` before weakening the assertion — a genuinely low correlation is a real bug, not a threshold to relax.

- [x] **Step 3: (implementation already exists)** — this task is validation only. If a test reveals a bug in Task 1/2, fix it there with a regression test, then return here.

- [x] **Step 4: Run to verify they pass**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "independent_optimizer or wmaze_place_cells" -q`
Expected: PASS (2 tests).

- [x] **Step 5: Commit**

```bash
git add src/state_space_practice/tests/test_graph_place_field.py
git commit -m "test(graph-pp): static-field parity vs independent optimizer + W-maze recovery"
```

> **MRF / `non_local_detector` comparison** is intentionally out of the test suite (no dependency on `non_local_detector`). Add it as an offline notebook that feeds *our* basis to both estimators, per the spec. Not part of this plan's automated tests.

---

# Stage 2 — Drifting field

`GraphPlaceFieldModel`: the coefficients drift as a random walk with per-neuron `Q_c = q_c·S`, inferred by the per-neuron `vmap` Laplace-EKF. EM/`fit_sgd` learn `τ²` (and `κ²` via SGD, `init_mean`); **`q_c` is a fixed hyperparameter by default** (learning it is an experimental opt-in — see Amendment 1: the marginal LL does not identify `q_c`).

## Task 4: `GraphPlaceFieldModel.__init__` and prior/`Q` construction

**Files:**
- Modify: `src/state_space_practice/graph_place_field.py` (append the class; add `GraphPlaceFieldModel` to `__all__`)
- Test: `src/state_space_practice/tests/test_graph_place_field.py` (append)

**Interfaces:**
- Consumes: `build_graph_basis`, `spectral_shape`, `GraphBasis`; `SGDFittableMixin` from `state_space_practice.sgd_fitting`.
- Produces: `GraphPlaceFieldModel(env, dt, *, rank=None, sigma=None, kappa2=1.0, alpha=1.0, tau2=1.0, init_drift_scale=1e-3, interpolation="nearest", laplacian_convention="distance", update_drift_scale=False, update_amplitude=True, update_init_mean=True, max_firing_rate_hz=500.0, max_newton_iter=1)`. After construction: `self.basis` (`GraphBasis`), `self.rank` (int), and `self.prior_cov()` / `self.drift_cov(q_c)` helpers returning `(rank, rank)` diagonal PSD matrices. Later tasks rely on: `self._spectral_shape_current()`, `self.rank`, `self.kappa2`, `self.alpha`, `self.tau2`, `self.dt`, `self.basis.eigvecs`, `self.transition_matrix` (`= I(rank)`), `self._log_intensity_func`, `self.max_newton_iter`, `self._max_log_count`. **The spectral shape `S` is never cached on the instance** — `kappa2` is fittable, so a snapshot would go stale; always derive `S` from `_spectral_shape_current()`.

- [x] **Step 1: Write the failing tests**

```python
from state_space_practice.graph_place_field import GraphPlaceFieldModel  # noqa: E402


def test_model_builds_diagonal_psd_prior_and_drift(small_grid_env):
    model = GraphPlaceFieldModel(
        small_grid_env, dt=0.02, rank=10, kappa2=0.5, alpha=1.0, tau2=2.0
    )
    assert model.rank == 10
    P0 = np.asarray(model.prior_cov())
    # P0 = tau2 * S, diagonal, strictly PSD (kappa2 > 0 keeps null modes finite).
    assert np.allclose(P0, np.diag(np.diag(P0)))
    assert np.all(np.linalg.eigvalsh(P0) > 0)
    from state_space_practice.graph_place_field import spectral_shape

    S = spectral_shape(model.basis.eigvals, 0.5, 1.0)
    np.testing.assert_allclose(np.diag(P0), 2.0 * S)
    Q = np.asarray(model.drift_cov(0.01))
    np.testing.assert_allclose(np.diag(Q), 0.01 * S)
    assert np.all(np.linalg.eigvalsh(Q) > 0)


def test_model_rejects_bad_hyperparameters(small_grid_env):
    with pytest.raises(ValueError, match="dt"):
        GraphPlaceFieldModel(small_grid_env, dt=0.0)
    with pytest.raises(ValueError, match="kappa2"):
        GraphPlaceFieldModel(small_grid_env, dt=0.02, kappa2=0.0)
    with pytest.raises(ValueError, match="tau2"):
        GraphPlaceFieldModel(small_grid_env, dt=0.02, tau2=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        GraphPlaceFieldModel(small_grid_env, dt=0.02, alpha=0.0)
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "builds_diagonal_psd or bad_hyperparameters" -q`
Expected: FAIL with `ImportError: cannot import name 'GraphPlaceFieldModel'`.

- [x] **Step 3: Implement `__init__` and the covariance helpers**

Append to `graph_place_field.py` (add `import logging`, `from state_space_practice.sgd_fitting import SGDFittableMixin`, `from state_space_practice.point_process_kalman import log_conditional_intensity`; add `GraphPlaceFieldModel` to `__all__`; add `logger = logging.getLogger(__name__)` near the top of the module if not present):

```python
class GraphPlaceFieldModel(SGDFittableMixin):
    """Drifting place-field model over a graph-Laplacian eigenbasis.

    Latent state per neuron ``c`` is ``w_{c,t} in R^rank``, the coefficients on the
    smoothest ``rank`` eigenvectors ``Phi`` of the environment graph Laplacian. The
    log-rate map is ``eta_{c,t} = Phi w_{c,t}``; at the animal's position ``z_t = Phi[bin(t)]``
    the point-process log-intensity is ``z_t^T w_{c,t}`` and spikes are Poisson with rate
    ``exp(z_t^T w_{c,t}) * dt``. The coefficients drift as a random walk
    ``w_{c,t} = w_{c,t-1} + eps_t``, ``eps_t ~ N(0, q_c * S)`` with the spectral shape
    ``S = (kappa2 I + diag(lambda))^(-alpha)`` shared by the prior ``P0 = tau2 * S``.

    x64 is required (see the module and repo CLAUDE.md notes).
    """

    def __init__(
        self,
        env: "Environment",
        dt: float,
        *,
        rank: Optional[int] = None,
        sigma: Optional[float] = None,
        kappa2: float = 1.0,
        alpha: float = 1.0,
        tau2: float = 1.0,
        init_drift_scale: float = 1e-3,
        interpolation: str = "nearest",
        laplacian_convention: LaplacianConvention = "distance",
        update_drift_scale: bool = False,
        update_amplitude: bool = True,
        update_init_mean: bool = True,
        max_firing_rate_hz: float = 500.0,
        max_newton_iter: int = 1,
    ) -> None:
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if not kappa2 > 0:
            raise ValueError(f"kappa2 must be positive, got {kappa2}.")
        if not tau2 > 0:
            raise ValueError(f"tau2 must be positive, got {tau2}.")
        if not alpha > 0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        if not init_drift_scale >= 0:
            raise ValueError(f"init_drift_scale must be >= 0, got {init_drift_scale}.")
        if max_firing_rate_hz <= 0:
            raise ValueError(
                f"max_firing_rate_hz must be positive, got {max_firing_rate_hz}."
            )

        self.env = env
        self.dt = dt
        self.basis = build_graph_basis(
            env, rank=rank, sigma=sigma, laplacian_convention=laplacian_convention
        )
        self.rank = int(self.basis.eigvecs.shape[1])
        self.kappa2 = kappa2
        self.alpha = alpha
        self.tau2 = tau2
        self.init_drift_scale = init_drift_scale
        self.interpolation = interpolation
        self.max_firing_rate_hz = max_firing_rate_hz
        self.max_newton_iter = max_newton_iter
        self.update_drift_scale = update_drift_scale
        self.update_amplitude = update_amplitude
        self.update_init_mean = update_init_mean
        self._log_intensity_func = log_conditional_intensity

        # The spectral shape S is deliberately NOT cached on the instance: kappa2 is
        # fittable (SGD), so a snapshot taken here would go stale. Always derive it
        # from the current kappa2 via _spectral_shape_current().
        self.transition_matrix = jnp.eye(self.rank)

        # Populated during fit.
        self.n_neurons: int = 1
        self.drift_scale: Optional[Array] = None  # (n_neurons,)
        self.init_mean: Optional[Array] = None  # (n_neurons, rank)
        self.smoother_mean: Optional[Array] = None  # (n_neurons, n_time, rank)
        self.smoother_cov: Optional[Array] = None
        self.smoother_cross_cov: Optional[Array] = None
        self.filtered_mean: Optional[Array] = None
        self.filtered_cov: Optional[Array] = None
        self.log_likelihoods: list[float] = []
        self._n_time: int = 0

    @property
    def _max_log_count(self) -> float:
        return float(np.log(self.max_firing_rate_hz * self.dt))

    def _spectral_shape_current(self) -> Array:
        """S at the current kappa2 (recomputed so SGD updates to kappa2 take effect)."""
        return jnp.asarray(spectral_shape(self.basis.eigvals, self.kappa2, self.alpha))

    def prior_cov(self) -> Array:
        """Prior / initial covariance ``P0 = tau2 * diag(S)``, shape (rank, rank)."""
        return jnp.diag(self.tau2 * self._spectral_shape_current())

    def drift_cov(self, q_c: float) -> Array:
        """Per-neuron drift covariance ``Q_c = q_c * diag(S)``, shape (rank, rank)."""
        return jnp.diag(q_c * self._spectral_shape_current())
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "builds_diagonal_psd or bad_hyperparameters" -q`
Expected: PASS.

- [x] **Step 5: Format and commit**

```bash
uv run --no-sync ruff format src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git add src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git commit -m "feat(graph-pp): GraphPlaceFieldModel constructor and spectral prior/Q"
```

---

## Task 5: E-step — per-neuron `vmap` Laplace-EKF smoother

**Files:**
- Modify: `src/state_space_practice/graph_place_field.py` (add methods to `GraphPlaceFieldModel`)
- Test: `src/state_space_practice/tests/test_graph_place_field.py` (append)

**Interfaces:**
- Consumes: `stochastic_point_process_smoother`, `stochastic_point_process_filter` from `state_space_practice.point_process_kalman`; `graph_design_matrix`.
- Produces:
  - `GraphPlaceFieldModel._design_and_spikes(times, trajectory, spikes) -> tuple[Array, Array]` — returns `(Z, spikes)` restricted to in-bounds rows: `Z` is `(n_valid, rank)`, `spikes` is `(n_neurons, n_valid)`.
  - `GraphPlaceFieldModel._e_step(Z, spikes) -> float` — runs the per-neuron `vmap` smoother, stores `smoother_mean/cov/cross_cov`, `filtered_mean/cov`, returns the **total** marginal log-likelihood (summed over neurons).

The per-neuron smoother is `jax.vmap`ped over `init_mean` (`0`), `spikes` (`0`), `process_cov` (`0`) with `design_matrix`/`transition_matrix`/`init_cov` shared (`None`). This was validated to give per-neuron independence and finite total LL.

- [x] **Step 1: Write the failing tests**

```python
def _toy_trajectory(env, n_time, seed=0):
    """A trajectory that visits real bin centers (so all rows are in-bounds)."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(env.bin_centers)
    idx = rng.integers(0, centers.shape[0], size=n_time)
    times = np.arange(n_time, dtype=float) * 0.02
    return times, centers[idx]


@pytest.mark.slow
def test_estep_returns_finite_total_ll_and_stores_posteriors(small_grid_env):
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=8)
    n_time = 150
    times, traj = _toy_trajectory(small_grid_env, n_time, seed=1)
    rng = np.random.default_rng(2)
    spikes = rng.poisson(0.05, size=(n_time, 3)).astype(float)
    model.n_neurons = 3
    model.drift_scale = jnp.full(3, 1e-3)
    model.init_mean = jnp.zeros((3, model.rank))
    Z, spk = model._design_and_spikes(times, traj, spikes)
    ll = model._e_step(Z, spk)
    assert np.isfinite(ll)
    assert model.smoother_mean.shape == (3, n_time, model.rank)


@pytest.mark.slow
def test_estep_neurons_are_independent(small_grid_env):
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=8)
    n_time = 120
    times, traj = _toy_trajectory(small_grid_env, n_time, seed=3)
    rng = np.random.default_rng(4)
    spikes = rng.poisson(0.05, size=(n_time, 2)).astype(float)
    model.n_neurons = 2
    model.drift_scale = jnp.full(2, 1e-3)
    model.init_mean = jnp.zeros((2, model.rank))
    Z, spk = model._design_and_spikes(times, traj, spikes)
    model._e_step(Z, spk)
    sm0 = np.asarray(model.smoother_mean[0]).copy()
    # Perturb neuron 1's spikes only; neuron 0's smoothed trajectory must not change.
    spikes2 = spikes.copy()
    spikes2[:, 1] += 1.0
    Z2, spk2 = model._design_and_spikes(times, traj, spikes2)
    model._e_step(Z2, spk2)
    np.testing.assert_allclose(sm0, np.asarray(model.smoother_mean[0]), atol=1e-9)
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "estep" -q`
Expected: FAIL with `AttributeError: 'GraphPlaceFieldModel' object has no attribute '_design_and_spikes'`.

- [x] **Step 3: Implement `_design_and_spikes` and `_e_step`**

Add these methods to `GraphPlaceFieldModel` (add only `stochastic_point_process_smoother` to the `point_process_kalman` import — import `_validate_filter_numerics` and `stochastic_point_process_filter` in the tasks that first use them, Tasks 6 and 7, so every commit stays `ruff check` clean):

```python
    def _design_and_spikes(
        self,
        times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
        spikes: ArrayLike,
    ) -> tuple[Array, Array]:
        """Build the in-bounds design matrix Z and aligned per-neuron spikes.

        Returns ``Z`` of shape ``(n_valid, rank)`` and ``spikes`` of shape
        ``(n_neurons, n_valid)`` (neuron axis first, ready for ``vmap``). Out-of-bounds
        samples are dropped consistently from both.
        """
        Z_full, valid = graph_design_matrix(
            self.env, self.basis, times, trajectory, interpolation=self.interpolation
        )
        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]
        Z = jnp.asarray(Z_full)[valid]
        spikes_valid = spikes_arr[valid]  # (n_valid, n_neurons)
        return Z, spikes_valid.T  # (n_valid, rank), (n_neurons, n_valid)

    def _e_step(self, Z: Array, spikes: Array) -> float:
        """Per-neuron vmap Laplace-EKF smoother; returns total marginal LL."""
        assert self.init_mean is not None
        assert self.drift_scale is not None
        S = self._spectral_shape_current()
        P0 = jnp.diag(self.tau2 * S)
        A = self.transition_matrix

        def _one(m0: Array, spk: Array, q_c: Array):
            Q = jnp.diag(q_c * S)
            return stochastic_point_process_smoother(
                init_mean_params=m0,
                init_covariance_params=P0,
                design_matrix=Z,
                spike_indicator=spk,
                dt=self.dt,
                transition_matrix=A,
                process_cov=Q,
                log_conditional_intensity=self._log_intensity_func,
                return_filtered=True,
                max_log_count=self._max_log_count,
                validate_inputs=False,
                max_newton_iter=self.max_newton_iter,
            )

        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_ll,
            self.filtered_mean,
            self.filtered_cov,
        ) = jax.vmap(_one, in_axes=(0, 0, 0))(
            self.init_mean, spikes, self.drift_scale
        )
        return float(jnp.sum(marginal_ll))
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "estep" -q`
Expected: PASS (2 tests).

- [x] **Step 5: Format and commit**

```bash
uv run --no-sync ruff format src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git add src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git commit -m "feat(graph-pp): per-neuron vmap Laplace-EKF E-step"
```

---

## Task 6: `fit` — latent trajectory inference with GEM rollback (slow)

**Files:**
- Modify: `src/state_space_practice/graph_place_field.py` (add `fit`, `_m_step`, warm-start helper)
- Test: `src/state_space_practice/tests/test_graph_place_field.py` (append)

**Interfaces:**
- Consumes: `fit_static_graph_glm`, `bin_spike_counts`, `bin_occupancy`, `spectral_precision` (warm-start); `sum_of_outer_products` from `state_space_practice.kalman`; `check_converged` from `state_space_practice.utils`.
- Produces:
  - `GraphPlaceFieldModel.fit(times, trajectory, spikes, *, max_iter=100, tolerance=1e-4, warm_start=True, verbose=True) -> list[float]`.
  - `GraphPlaceFieldModel._m_step() -> None` — closed-form scalar updates: `q_c* = mean_j(diag(Q_inc_c)_j / S_j)`, `τ²* = mean over neurons and modes of diag(V0_c)_j / S_j`, and `init_mean_c = m0_c` when `update_init_mean`.

**Optional hyperparameter-update derivation (recorded so the implementer can verify):** for `A = I` and `Q_c = q_c·S`, the complete-data increment term is `-0.5 Σ_t [(1/q_c) tr(S^{-1} E[Δw Δw^T]) + rank·log q_c]`, giving `q_c* = tr(S^{-1} M_c) / (rank·(T-1))` where `M_c = Σ_t E[Δw Δw^T]`. Since `S` is diagonal this is `mean_j(diag(M_c/(T-1))_j / S_j)`. The per-neuron increment covariance `M_c/(T-1)` is computed with the exact `gamma/beta` algebra proven in `PlaceFieldModel._m_step`. `τ²` scales the init prior `P0 = τ²S`; its update is `mean_{c,j}(diag(V0_c)_j / S_j)` (the mean term drops when `init_mean_c := m0_c`). This algebra is ancillary to the primary trajectory-inference acceptance criterion.

- [x] **Step 1: Write the revised failing tests**

```python
def _simulate_drifting_spikes(env, basis, dt, n_time, q_c, tau2, kappa2, seed):
    """Poisson spikes from a coefficient random walk w_t = w_{t-1} + N(0, q_c*S)."""
    from state_space_practice.graph_place_field import spectral_shape

    rng = np.random.default_rng(seed)
    centers = np.asarray(env.bin_centers)
    idx = rng.integers(0, centers.shape[0], size=n_time)
    times = np.arange(n_time, dtype=float) * dt
    traj = centers[idx]
    S = spectral_shape(basis.eigvals, kappa2, 1.0)
    rank = basis.eigvecs.shape[1]
    w = rng.standard_normal(rank) * np.sqrt(tau2 * S)  # w_0 ~ N(0, tau2 S)
    Z = basis.eigvecs[idx]  # (n_time, rank), design at visited bins
    spikes = np.zeros(n_time)
    for t in range(n_time):
        if t > 0:
            w = w + rng.standard_normal(rank) * np.sqrt(q_c * S)
        rate = np.exp(Z[t] @ w)
        spikes[t] = rng.poisson(rate * dt)
    return times, traj, spikes


@pytest.mark.slow
def test_fit_em_is_monotone_under_rollback(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=8)
    times, traj, spikes = _simulate_drifting_spikes(
        small_grid_env, basis, dt=0.02, n_time=800,
        q_c=1e-3, tau2=1.0, kappa2=1e-2, seed=0,
    )
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=8, kappa2=1e-2)
    lls = model.fit(times, traj, spikes, max_iter=30, verbose=False)
    diffs = np.diff(lls)
    # GEM rollback guarantees the accepted LL sequence never decreases.
    assert np.all(diffs >= -1e-6)
    assert len(lls) >= 2  # guard: EM actually iterated
```

> **Withdrawn acceptance criterion:** the original
> `test_fit_recovers_drift_scale_on_fixed_S` was not implemented. As documented in
> Amendment 1, the approximate inference objective does not support an honest
> generating-`q_c` recovery claim. The revised Task-6 acceptance suite instead includes
> `test_drift_scale_mstep_matches_closed_form` (algebraic correctness) and
> `test_static_limit_approximates_static_estimator` (non-circular, approximate static-limit
> agreement at about 1.8% coefficient relative-L2 error), alongside the monotonic-rollback test
> above.

- [x] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "fit_em or drift_scale_mstep or static_limit" -q`
Expected before implementation: FAIL because `fit` / `_m_step` are not yet implemented.

- [x] **Step 3: Implement the warm-start, `_m_step`, and `fit`**

Add to `GraphPlaceFieldModel` (add `sum_of_outer_products` to the existing `from state_space_practice.kalman import psd_solve, symmetrize` line; add `from state_space_practice.utils import check_converged, validate_count_array`; add `_validate_filter_numerics` to the existing `from state_space_practice.point_process_kalman import ...` line — this task is its first use):

```python
    def _warm_start(self, times, trajectory, spikes) -> None:
        """Set per-neuron init_mean from the static GLM MAP on aggregated bins."""
        counts = bin_spike_counts(self.env, spikes, times, trajectory, self.basis)
        occ = bin_occupancy(self.env, times, trajectory, self.dt)
        prec = spectral_precision(self.basis.eigvals, self.tau2, self.kappa2, self.alpha)
        w0, _ = fit_static_graph_glm(counts, occ, self.basis.eigvecs, prec)
        w0 = jnp.atleast_2d(w0)  # (n_neurons, rank)
        self.init_mean = w0

    def _m_step(self) -> None:
        """Closed-form scalar updates of per-neuron q_c and shared tau2."""
        assert self.smoother_mean is not None
        assert self.smoother_cov is not None
        assert self.smoother_cross_cov is not None
        S = self._spectral_shape_current()

        def _stats(sm, sc, scc):
            # sm (T, rank), sc (T, rank, rank), scc (T-1, rank, rank)
            n_time = sm.shape[0]
            gamma = jnp.sum(sc, axis=0) + sum_of_outer_products(sm, sm)
            gamma1 = gamma - jnp.outer(sm[-1], sm[-1]) - sc[-1]
            gamma2 = gamma - jnp.outer(sm[0], sm[0]) - sc[0]
            beta = (scc.sum(axis=0) + sum_of_outer_products(sm[:-1], sm[1:])).T
            q_inc = (gamma2 - beta.T - beta + gamma1) / (n_time - 1)  # E[dw dw^T]
            return jnp.diag(q_inc), jnp.diag(sc[0])  # (rank,), (rank,) diagonals

        diag_q_inc, diag_v0 = jax.vmap(_stats)(
            self.smoother_mean, self.smoother_cov, self.smoother_cross_cov
        )
        if self.update_drift_scale:
            # q_c* = mean_j( diag(E[dw dw^T])_j / S_j )
            self.drift_scale = jnp.maximum(
                jnp.mean(diag_q_inc / S[None, :], axis=1), 1e-12
            )
        if self.update_amplitude:
            # tau2* = mean over neurons and modes of diag(V0)_j / S_j
            self.tau2 = float(jnp.maximum(jnp.mean(diag_v0 / S[None, :]), 1e-12))
        if self.update_init_mean:
            self.init_mean = self.smoother_mean[:, 0, :]

    def fit(
        self,
        times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
        spikes: ArrayLike,
        *,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        warm_start: bool = True,
        verbose: bool = True,
    ) -> list[float]:
        """Fit by EM (GEM with rollback). Returns the accepted marginal-LL history."""
        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]
        validate_count_array(spikes_arr, "spikes", allow_empty=False)
        self.n_neurons = int(spikes_arr.shape[1])
        Z, spk = self._design_and_spikes(times, trajectory, spikes_arr)
        self._n_time = int(Z.shape[0])

        if warm_start:
            self._warm_start(times, trajectory, spikes_arr)
        elif self.init_mean is None:
            self.init_mean = jnp.zeros((self.n_neurons, self.rank))
        self.drift_scale = jnp.full(self.n_neurons, self.init_drift_scale)

        _validate_filter_numerics(self.prior_cov(), n_time=self._n_time)

        def _log(msg: str) -> None:
            if verbose:
                print(msg)

        self.log_likelihoods = []
        last_state: Optional[dict] = None

        def _capture() -> dict:
            return {
                k: getattr(self, k)
                for k in (
                    "smoother_mean", "smoother_cov", "smoother_cross_cov",
                    "filtered_mean", "filtered_cov", "drift_scale",
                    "init_mean", "tau2",
                )
            }

        def _restore(state: Optional[dict]) -> None:
            if state is not None:
                for k, v in state.items():
                    setattr(self, k, v)

        for iteration in range(max_iter):
            ll = self._e_step(Z, spk)
            self.log_likelihoods.append(ll)
            _log(f"  EM iter {iteration + 1}/{max_iter}: LL = {ll:.2f}")
            if not np.isfinite(ll):
                _log("  WARNING: non-finite LL; stopping.")
                self.log_likelihoods.pop()
                _restore(last_state)
                break
            if iteration > 0:
                converged, increasing = check_converged(
                    ll, self.log_likelihoods[-2], tolerance
                )
                if not increasing:
                    _restore(last_state)
                    bad = self.log_likelihoods.pop()
                    _log(
                        f"  WARNING: LL decreased {self.log_likelihoods[-1]:.2f} -> "
                        f"{bad:.2f}; rolling back and stopping."
                    )
                    break
                if converged:
                    _log(f"  Converged after {iteration + 1} iterations.")
                    break
            last_state = _capture()
            self._m_step()

        return self.log_likelihoods
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "fit_em or drift_scale_mstep or static_limit" -q`
Expected: PASS (3 tests: rollback monotonicity, closed-form M-step algebra, and approximate static-limit agreement).

- [x] **Step 5: Format and commit**

```bash
uv run --no-sync ruff format src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git add src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git commit -m "feat(graph-pp): EM fit with scalar q_c/tau2 M-step and GEM rollback"
```

---

## Task 7: `fit_sgd` — optimize enabled graph hyperparameters and improve marginal LL (slow)

**Files:**
- Modify: `src/state_space_practice/graph_place_field.py` (add the `SGDFittableMixin` protocol methods + `fit_sgd`)
- Test: `src/state_space_practice/tests/test_graph_place_field.py` (append)

**Interfaces:**
- Consumes: `SGDFittableMixin.fit_sgd` (base); `POSITIVE` from `state_space_practice.parameter_transforms`.
- Produces `GraphPlaceFieldModel.fit_sgd(times, trajectory, spikes, *, optimizer=None, num_steps=200, verbose=False, convergence_tol=None, warm_start=True) -> list[float]` plus the required protocol hooks: `_n_timesteps` (property), `_check_sgd_initialized`, `_build_param_spec`, `_sgd_loss_fn`, `_store_sgd_params`, `_finalize_sgd`. Enabled positive parameters (`τ²`, `κ²`, and optionally per-neuron `q_c`) use `POSITIVE`; `init_mean` is optionally unconstrained. `q_c` optimization is experimental and disabled by default, and no recovery guarantee is claimed. Because `Q_c = q_c·diag(S)` is diagonal, it is PSD for any `q_c, κ² > 0`, so the eigendecomp→Cholesky gradient-NaN issue that affects dense-`Q` SGD models does not arise here.

- [x] **Step 1: Write the revised failing tests**

> **Withdrawn acceptance criteria:** `test_fit_sgd_recovers_drift_scale` and
> `test_em_and_sgd_agree_on_marginal_ll` were not implemented. The former would make the
> unsupported recovery claim rejected in Amendment 1; the latter is not meaningful when
> EM and SGD intentionally optimize different enabled parameter sets. The revised Task-7
> acceptance suite is `test_fit_sgd_runs_and_default_keeps_drift_fixed`,
> `test_fit_sgd_improves_marginal_ll`, and
> `test_update_kappa2_false_freezes_kappa2_under_sgd`.

- [x] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "fit_sgd_runs or fit_sgd_improves or update_kappa2_false" -q`
Expected before implementation: FAIL with `AttributeError` (no `fit_sgd` / protocol hooks).

- [x] **Step 3: Implement the SGD protocol and `fit_sgd`**

Add to `GraphPlaceFieldModel` (add `stochastic_point_process_filter` to the existing `from state_space_practice.point_process_kalman import ...` line — this task is its first use):

```python
    # --- SGDFittableMixin protocol ---

    @property
    def _n_timesteps(self) -> int:
        return self._n_time

    def _check_sgd_initialized(self) -> None:
        if self.init_mean is None or self.drift_scale is None:
            raise RuntimeError(
                "Model not initialized. Call fit_sgd(times, trajectory, spikes), "
                "not super().fit_sgd() directly."
            )

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import POSITIVE, UNCONSTRAINED

        params: dict = {}
        spec: dict = {}
        if self.update_drift_scale:
            params["drift_scale"] = self.drift_scale
            spec["drift_scale"] = POSITIVE
        if self.update_amplitude:
            params["tau2"] = jnp.asarray(self.tau2)
            spec["tau2"] = POSITIVE
        params["kappa2"] = jnp.asarray(self.kappa2)
        spec["kappa2"] = POSITIVE
        if self.update_init_mean:
            params["init_mean"] = self.init_mean
            spec["init_mean"] = UNCONSTRAINED
        return params, spec

    def _sgd_loss_fn(self, params: dict, Z: Array, spikes: Array) -> Array:
        kappa2 = params.get("kappa2", self.kappa2)
        tau2 = params.get("tau2", self.tau2)
        drift = params.get("drift_scale", self.drift_scale)
        m_init = params.get("init_mean", self.init_mean)
        S = jnp.asarray((kappa2 + jnp.asarray(self.basis.eigvals)) ** (-self.alpha))
        P0 = jnp.diag(tau2 * S)
        A = self.transition_matrix

        def _one(m0, spk, q_c):
            _, _, ll = stochastic_point_process_filter(
                init_mean_params=m0,
                init_covariance_params=P0,
                design_matrix=Z,
                spike_indicator=spk,
                dt=self.dt,
                transition_matrix=A,
                process_cov=jnp.diag(q_c * S),
                log_conditional_intensity=self._log_intensity_func,
                max_log_count=self._max_log_count,
                validate_inputs=False,
                max_newton_iter=self.max_newton_iter,
            )
            return ll

        lls = jax.vmap(_one, in_axes=(0, 0, 0))(m_init, spikes, drift)
        return -jnp.sum(lls)

    def _store_sgd_params(self, params: dict) -> None:
        if "drift_scale" in params:
            self.drift_scale = params["drift_scale"]
        if "tau2" in params:
            self.tau2 = float(params["tau2"])
        if "kappa2" in params:
            self.kappa2 = float(params["kappa2"])
        if "init_mean" in params:
            self.init_mean = params["init_mean"]

    def _finalize_sgd(self, Z: Array, spikes: Array) -> None:
        ll = self._e_step(Z, spikes)
        self.log_likelihoods = [ll]

    def fit_sgd(  # type: ignore[override]
        self,
        times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
        spikes: ArrayLike,
        *,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        warm_start: bool = True,
    ) -> list[float]:
        """Fit by minimizing the negative marginal LL via gradient descent."""
        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]
        validate_count_array(spikes_arr, "spikes", allow_empty=False)
        self.n_neurons = int(spikes_arr.shape[1])
        Z, spk = self._design_and_spikes(times, trajectory, spikes_arr)
        self._n_time = int(Z.shape[0])

        if warm_start:
            self._warm_start(times, trajectory, spikes_arr)
        elif self.init_mean is None:
            self.init_mean = jnp.zeros((self.n_neurons, self.rank))
        self.drift_scale = jnp.full(self.n_neurons, self.init_drift_scale)

        _validate_filter_numerics(self.prior_cov(), n_time=self._n_time)

        return super().fit_sgd(
            Z,
            spk,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "fit_sgd_runs or fit_sgd_improves or update_kappa2_false" -q`
Expected: PASS (3 tests, slow: default fixed drift, marginal-LL improvement, and frozen `κ²`).

- [x] **Step 5: Format and commit**

```bash
uv run --no-sync ruff format src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git add src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git commit -m "feat(graph-pp): SGD fit over enabled graph hyperparameters"
```

---

## Task 8: Latent field trajectory recovery, rate-map prediction, and graph geometry (slow)

**Files:**
- Modify: `src/state_space_practice/graph_place_field.py` (add `predict_log_rate_trajectory`, `predict_rate_map`, `score`)
- Test: `src/state_space_practice/tests/test_graph_place_field.py` (append)

**Interfaces:**
- Produces:
  - `GraphPlaceFieldModel.predict_log_rate_trajectory(neuron_idx=0, time_slice=None) -> NDArray` — posterior-mean log-rate field over selected times and all active graph bins. Shape `(n_selected_times, n_bins)`. This is the primary drift-tracking output.
  - `GraphPlaceFieldModel.predict_rate_map(neuron_idx=0, time_slice=None) -> NDArray` — per-active-bin firing rate (Hz), the log-normal posterior mean `mean_t exp(Φ m_{c,t} + 0.5 Φ V_{c,t} Φ^T)` over the window. Shape `(n_bins,)`.
  - `GraphPlaceFieldModel.score(times, trajectory, spikes) -> float` — held-out total marginal LL (filter only, fitted parameters).

- [x] **Step 1: Write the failing tests**

```python
@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_model_tracks_drifting_field_over_time(small_grid_env, seed):
    """Recover time-varying spatial structure over every graph bin."""
    basis = build_graph_basis(small_grid_env, rank=8)
    times, traj, spikes, _, w_true = _simulate_field_drift_spikes(
        small_grid_env, basis, dt=0.02, n_time=3000, q_c=1e-2, seed=seed
    )
    # Use the default q_c=1e-3 (10x misspecified), cold initialization, and one
    # inference pass so filtered_mean has no future-data leakage.
    model = GraphPlaceFieldModel(
        small_grid_env, dt=0.02, rank=8, update_drift_scale=False,
        update_amplitude=False, update_init_mean=False,
    )
    model.fit(times, traj, spikes, warm_start=False, max_iter=1, verbose=False)

    burn_in = 500
    phi = np.asarray(basis.eigvecs)
    field_true = w_true[burn_in:] @ phi.T
    field_sm = model.predict_log_rate_trajectory(time_slice=slice(burn_in, None))

    def isolate_spatial_drift(field):
        spatial = field - field.mean(axis=1, keepdims=True)
        return spatial - spatial.mean(axis=0, keepdims=True)

    drift_true = isolate_spatial_drift(field_true)
    drift_sm = isolate_spatial_drift(field_sm)
    truth_rms = np.std(drift_true)
    drift_nrmse = np.sqrt(np.mean((drift_sm - drift_true) ** 2)) / truth_rms
    drift_corr = np.corrcoef(drift_sm.ravel(), drift_true.ravel())[0, 1]
    amplitude_ratio = np.linalg.norm(drift_sm) / np.linalg.norm(drift_true)

    assert truth_rms > 0.2
    assert drift_nrmse < 0.75
    assert drift_corr > 0.7
    assert 0.4 < amplitude_ratio < 1.2


@pytest.mark.slow
def test_graph_basis_does_not_smear_across_wmaze_arms(w_maze_env):
    """A field localized to one W-maze arm stays on that arm in the graph basis."""
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(w_maze_env, sigma=12.0)
    centers = np.asarray(w_maze_env.bin_centers)
    # Pick a bin, define a true field as a tight bump around it (geodesic sense via graph).
    b = int(np.argmax(centers[:, 1]))  # a bin at one extremity
    labels = basis.component_labels  # single component here; use graph distance proxy
    # Reconstruct a delta at b through the truncated basis: leakage measured as mass
    # landing on bins far in Euclidean space but that the arm structure separates.
    delta = np.zeros(w_maze_env.n_bins)
    delta[b] = 1.0
    recon = basis.eigvecs @ (basis.eigvecs.T @ delta)
    far = np.linalg.norm(centers - centers[b], axis=1) > 0.5 * np.ptp(centers[:, 0])
    # The truncated graph basis keeps most reconstructed mass near b (no cross-arm smear).
    assert np.sum(np.abs(recon[far])) < 0.35 * np.sum(np.abs(recon))
```

Note: the head-to-head *against splines* (showing `PlaceFieldModel`'s tensor-product splines leak across arms where the graph basis does not) is a stronger but heavier comparison. Keep the in-suite test to the graph-basis locality assertion above; put the full spline-vs-graph side-by-side (fitting both models on the same W-maze session and comparing cross-arm rate leakage) in an offline notebook, since it fits two full models and is inherently slow.

- [x] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "default_model_tracks or smear_across" -q`
Expected: `default_model_tracks` FAILs with `AttributeError` before the trajectory output/inference path exists; `smear_across` may already pass (uses only the substrate) — that is fine, it is a guardrail on the basis.

- [x] **Step 3: Implement `predict_log_rate_trajectory`, `predict_rate_map`, and `score`**

Add to `GraphPlaceFieldModel`:

```python
    def _check_fitted(self, method: str) -> None:
        if self.smoother_mean is None:
            raise RuntimeError(f"Model not fitted; call fit(...) before {method}().")

    def predict_log_rate_trajectory(
        self, neuron_idx: int = 0, time_slice: Optional[slice] = None
    ) -> NDArray[np.float64]:
        """Posterior-mean log-rate field, shape (selected times, active bins)."""
        self._check_fitted("predict_log_rate_trajectory")
        if time_slice is None:
            time_slice = slice(None)
        means = np.asarray(self.smoother_mean[neuron_idx][time_slice])
        return means @ np.asarray(self.basis.eigvecs).T

    def predict_rate_map(
        self, neuron_idx: int = 0, time_slice: Optional[slice] = None
    ) -> NDArray[np.float64]:
        """Per-active-bin firing rate (Hz), the log-normal posterior mean over time.

        ``rate_i = mean_t exp(Phi_i m_{c,t} + 0.5 Phi_i V_{c,t} Phi_i^T)`` — the exact
        posterior expected rate for the linear log-intensity model and Gaussian posterior.
        """
        self._check_fitted("predict_rate_map")
        if time_slice is None:
            time_slice = slice(None)
        Phi = np.asarray(self.basis.eigvecs)  # (n_bins, rank)
        means = np.asarray(self.smoother_mean[neuron_idx][time_slice])  # (T, rank)
        covs = np.asarray(self.smoother_cov[neuron_idx][time_slice])  # (T, rank, rank)
        log_rate = means @ Phi.T  # (T, n_bins)
        var = np.einsum("br,trs,bs->tb", Phi, covs, Phi)  # (T, n_bins)
        rate = np.exp(log_rate + 0.5 * np.maximum(var, 0.0))
        return rate.mean(axis=0)

    def score(
        self,
        times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
        spikes: ArrayLike,
    ) -> float:
        """Held-out total marginal log-likelihood (filter only, fitted parameters)."""
        self._check_fitted("score")
        assert self.init_mean is not None and self.drift_scale is not None
        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]
        Z, spk = self._design_and_spikes(times, trajectory, spikes_arr)
        S = self._spectral_shape_current()
        P0 = jnp.diag(self.tau2 * S)

        def _one(m0, s, q_c):
            _, _, ll = stochastic_point_process_filter(
                init_mean_params=m0,
                init_covariance_params=P0,
                design_matrix=Z,
                spike_indicator=s,
                dt=self.dt,
                transition_matrix=self.transition_matrix,
                process_cov=jnp.diag(q_c * S),
                log_conditional_intensity=self._log_intensity_func,
                max_log_count=self._max_log_count,
                validate_inputs=False,
                max_newton_iter=self.max_newton_iter,
            )
            return ll

        lls = jax.vmap(_one, in_axes=(0, 0, 0))(self.init_mean, spk, self.drift_scale)
        return float(jnp.sum(lls))
```

- [x] **Step 4: Run to verify it passes**

Run: `uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -k "default_model_tracks or smear_across" -q`
Expected: PASS (the three-seed full-field tracking contract and graph-geometry guard).

- [x] **Step 5: Full-suite regression + commit**

```bash
uv run --no-sync pytest src/state_space_practice/tests/test_graph_place_field.py -q
uv run --no-sync ruff format src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git add src/state_space_practice/graph_place_field.py src/state_space_practice/tests/test_graph_place_field.py
git commit -m "feat(graph-pp): expose and validate latent field drift trajectory"
```

Also run the fast suite to confirm nothing regressed and it stays under a minute:

```bash
uv run --no-sync pytest src/state_space_practice/tests/ -m "not slow" -q
```

---

# Stage 3 — Deferred (not in this plan)

Richer dynamics are explicitly out of scope here and feed existing deferred plans. Do **not** implement them as part of this plan; they are recorded so the reviewer knows where the seams are:

- **Per-mode Matérn-SDE drift** via `gp_ssm.py` (`matern32_continuous`, `matern32_discretize`) — replace the scalar `q_c·S` random walk with a continuous-time Matérn process per mode. Feeds `docs/plans/2026-04-03-joint-learning-drift.md`.
- **Covariate / session-driven drift gains `B`** — feeds `docs/plans/2026-04-04-covariate-driven-drift.md` and `docs/plans/2026-04-03-cross-session-drift.md`.
- **Localized-basis alternative** (`neurospatial.ops.basis` `geodesic_rbf_basis` / `heat_kernel_wavelet_basis`) with a dense `Φ^T L Φ` penalty — the model's `(Φ, penalty)` interface already admits it; a dense-`Q` variant would need the eigendecomp→Cholesky gradient-NaN mitigation noted in the spec.

These are preferred via SGD (`fit_sgd`) because they add parameters without clean EM M-steps.

---

## Self-Review

**1. Spec coverage.** Every spec section maps to a task:
- Stage 0 substrate → pre-existing (pre-flight note).
- Model (latent, spatial field, observation, dynamics, spectral prior `P0 = τ²S`, process noise `Q_c = q_c·S`) → Task 4 (`__init__`, `prior_cov`, `drift_cov`).
- Null modes / intercepts (finite ridge in default mode; unpenalized in parity mode) → Task 1 (`spectral_precision` finite at null; `parity_penalty` zeros null modes).
- Basis choice (Laplacian eigenbasis, component-local, bandwidth truncation) → Stage 0 (`build_graph_basis`, used in Task 4).
- Fitting `.fit()` with fixed-`S` scalar M-steps + GEM rollback → Task 6. `.fit_sgd()` over the enabled `τ²`/`κ²`/`init_mean` parameters, with `q_c` available only as an experimental opt-in → Task 7.
- Per-neuron `vmap` filter (not block path) → Task 5.
- Stage 1 static estimator + MRF-parity-configuration + W-maze recovery → Tasks 1–3.
- Stage 2 full latent-field trajectory recovery under a deliberately misspecified default `q_c`, approximate static-limit agreement, SGD marginal-LL improvement, causal/static comparator ordering, and W-maze geometry → Tasks 6–8. Generating-`q_c` recovery and EM↔SGD parity were withdrawn (Amendment 1).
- Module & dependency (new module, `neurospatial` extra, x64, PSD validation) → Global Constraints + Task 4 (`_validate_filter_numerics`).
- Validation (behavioral assertions, `neurospatial.simulation` spatial ground truth, temporally centered full-graph trajectory recovery across three sessions, closed-form optional `q_c` M-step algebra, frozen-`κ²` behavior, and SGD marginal-LL improvement) → Tasks 3, 6, 7, 8.
- Risks (rank truncation, disconnected graphs, f32 NaN, simulator circularity, `neurospatial` API drift) → covered by Stage-0 substrate + Global Constraints; simulator-circularity mitigation (abrupt-remap / non-graph-drift stress cases) is noted as an offline extension.

**2. Placeholder scan.** No "TBD"/"add validation"/"handle edge cases" placeholders; retained acceptance steps name concrete behavioral tests, and withdrawn criteria are explicitly documented.

**3. Type consistency.** Names are consistent across tasks: `spectral_precision`/`parity_penalty`/`fit_static_graph_glm` (Task 1) are reused verbatim in Tasks 2, 3, 6, 8; `GraphPlaceFieldModel` attributes (`rank`, `drift_scale`, `init_mean`, `smoother_mean`, `tau2`, `kappa2`, `transition_matrix`, `_log_intensity_func`, `_max_log_count`) and the `_spectral_shape_current()` accessor (the spectral shape `S` is never cached — `kappa2` is fittable) are defined in Task 4 and consumed by Tasks 5–8; `_design_and_spikes`/`_e_step` (Task 5) are consumed by `fit` (Task 6), `_finalize_sgd`/`score` (Tasks 7–8). `drift_scale` (not `q_c`) is the canonical per-neuron array name throughout. The SGD protocol hooks match `SGDFittableMixin`'s current surface (`_prepare_sgd_data`, `_build_param_spec`, `_sgd_loss_fn`, `_store_sgd_params`, `_finalize_sgd`, `_n_timesteps`). The unused `_check_sgd_initialized` hook was removed during integration.

**Known soft spots (flag for the executor, not blockers):**
- Retained statistical thresholds (static W-maze recovery, approximate static-limit error, posterior rate-map recovery, and drift-tracking ordering) may need recalibration against dependency or simulator changes. If one fails, first check coverage/`n_time`/seed and confirm it is not a genuine bug before adjusting it — never add a generating-`q_c` recovery threshold under the current model.
- `test_graph_basis_does_not_smear_across_wmaze_arms` selects a Euclidean-near but graph-far pair using graph shortest-path distance. The full spline-vs-graph head-to-head remains a deferred offline notebook (Amendment 5).

---

## Execution Handoff

**Plan complete and saved to `docs/plans/2026-07-16-drifting-graph-gp-place-field.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
