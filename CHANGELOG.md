# Changelog

All notable changes to `state_space_practice` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Public API and version at package level**: `state_space_practice.__version__`
  and lazily loaded (PEP 562) entry points — `kalman_filter`, `kalman_smoother`,
  `switching_kalman_filter`, `switching_kalman_smoother`, `run_em`,
  `PointProcessModel`, `PlaceFieldModel`, `PositionDecoder`,
  `CommonOscillatorModel`, `CorrelatedNoiseModel`, `DirectedInfluenceModel`,
  `SwitchingSpikeOscillatorModel`, `MultinomialChoiceModel`,
  `CovariateChoiceModel`, `SwitchingChoiceModel`, `ContingencyBeliefModel`,
  `SmithLearningModel`. `import state_space_practice` imports only JAX.
- **`state_space_practice.exceptions`**: `StateSpaceWarning` (a `UserWarning`
  subclass; all library `UserWarning`s now use it) and `NotFittedError` (a
  `RuntimeError` subclass).
- **Float64 import warning**: importing the package with `jax_enable_x64` off
  emits a `StateSpaceWarning` with the enable-x64 recipe.
- **`kalman.kalman_filter_update` / `kalman.kalman_smoother_update`**: public
  names for the single-step Kalman updates.
- **Optional extras** `plot` (matplotlib) and `notebooks` (matplotlib, pandas).
- **Tooling**: GitHub Actions CI (ruff, ruff format, mypy on the type-clean
  modules, fast tests on Python 3.10-3.12, nightly full suite),
  `.pre-commit-config.yaml` (ruff, ruff-format, nbstripout), `[tool.ruff]`
  config, and `HYPOTHESIS_PROFILE` selection in the test conftest. mypy gates
  the 24 modules listed in `[tool.mypy] files`; `mypy src/state_space_practice`
  still reports 142 errors in the other 21 modules (was 150 before the
  `fit_sgd` override fix).
- Tests that call `.fit(` / `.fit_sgd(` / `run_em(` (outside `pytest.raises`),
  directly or through a fixture, are marked `slow` automatically at collection.
- **`em_driver.run_em`**: the shared EM loop (E-step, convergence, rollback,
  M-step) used by the oscillator models, `PointProcessModel`, `PlaceFieldModel`,
  the switching point-process models and `SmithLearningModel`. Invalid option
  combinations raise `ValueError` up front.
- **Public helpers in `utils`**: `contains_tracer`, `clip_eigenvalues`,
  `psd_logdet`, `validate_int`, `validate_finite_array`,
  `validate_nonnegative_array`, `validate_unit_interval_array` and
  `zero_preserving_log`.
- **`return_block_covariances=False`** on `stochastic_point_process_filter` /
  `stochastic_point_process_smoother`: on the block-diagonal path, return a
  `BlockDiagonalCovariance` (per-neuron blocks) instead of dense
  `(n_time, n_state, n_state)` arrays.
- **`SwitchingSpikeOscillatorModel(max_spectral_radius=0.999)`** replaces a
  hard-coded bound (same default). It is enforced when the transition matrix is
  projected (`update_continuous_transition_matrix=True`).
- **Exact initial-state EM update**: `kalman.InitialStatePrior`,
  `kalman.smooth_initial_state` and `initial_state_prior=` on
  `kalman_maximization_step` / `point_process_kalman.dynamics_only_m_step`
  return the smoothed `x_0` (one RTS step behind the smoother output) instead
  of the `x_1` moments.
- **`kalman.measurement_cov_residual_form` / `process_cov_residual_form`**
  (centred M-step covariance estimates, PSD by construction) and
  **`kalman.rts_backward_scan_with_predictions`** (RTS pass for filters whose
  one-step prediction was modified, e.g. by covariance inflation).
- **Scale-relative PSD floors in `utils`**: `relative_psd_floor`,
  `clip_eigenvalues_relative`, `project_psd_relative`,
  `floor_variances_relative` and `warn_if_floored` (a jit-safe logged warning).
- **`utils.differentiable_spectral_radius`** (exact largest eigenvalue
  magnitude, usable under `jit` / `grad` / `vmap`), and
  `stabilize_transition_matrix(block_size=, warn=)` which clamps each
  strongly connected oscillator block separately and logs the radius and scale.
  The docstring explains how to pick `max_spectral_radius`
  (`>= 1 - pi * min_freq_gap / fs`).
- **`switching_kalman_maximization_step(fixed_measurement_matrix=,
  fixed_continuous_transition_matrix=, previous_params=,
  estimate_measurement_params=)`**: R and Q are estimated at the H / A that
  are actually installed (residual forms), and a discrete state with fewer
  than `n_cont + 1` expected bins keeps its previous per-state parameters
  (logged). `optimize_dim_transition_params(max_spectral_radius=)` replaces a
  hard-coded 0.99.
- **`ContingencyBeliefModel(seed=None)`**; `None` keeps the historical
  initialization.

### Changed — behavior (may affect existing callers)

- **Runtime dependencies trimmed** to what the package imports: numpy, scipy,
  jax, optax, patsy, networkx, scikit-learn (with minimum versions).
  matplotlib moved to the `plot` extra; pandas to `notebooks`; jaxlib (pulled
  in by jax), xarray, tqdm, track_linearization and seaborn are no longer
  dependencies. Install `state_space_practice[plot]` for the `plot_*` helpers.
- **The wheel no longer ships the test suite** (`state_space_practice.tests`).
- **`verbose=True` progress** in `fit_sgd` (all models), `PlaceFieldModel.fit`
  and `SmithLearningModel.fit` is logged at INFO on the module logger instead
  of printed to stdout (enable with `logging.basicConfig(level=logging.INFO)`).
  `SmithLearningModel`'s per-iteration record is DEBUG unless `verbose=True`.
- **Not-fitted errors are `NotFittedError`** (still a `RuntimeError`) in
  `PlaceFieldModel`, `TemporalRateGP`, the oscillator models, the switching
  point-process models' `decode` / `predict_proba` and the choice models.
  `PlaceFieldModel.n_free_params` on an unfitted model raises it instead of
  an `AssertionError`.
- **`SGDFittableMixin.fit_sgd` is declared `(*args, **kwargs)`**; the optimizer
  settings (`optimizer`, `num_steps`, `verbose`, `convergence_tol`) are still
  keyword-only with the same defaults.
- **pytest no longer ignores all `DeprecationWarning`s**; they are errors like
  every other warning.
- **`PlaceFieldModel.smoother_cov` / `smoother_cross_cov` / `filtered_cov`**
  hold a `BlockDiagonalCovariance` when a multi-neuron model runs on the
  block-diagonal path (dense arrays otherwise). Integer/slice indexing,
  `.sum(axis=0)`, `.diagonal()` and `np.asarray` / `jnp.asarray` work; other
  `jnp.*` functions and `jax.jit` raise, so densify with `jnp.asarray(cov)`
  first.
- **Non-finite log-likelihood during EM** (`SmithLearningModel`,
  `PlaceFieldModel`): a non-finite E-step after the first is no longer appended
  to the history. The model rolls back to the last accepted parameters and
  posteriors, so `log_likelihood_`, `n_iter_` and `bic()` describe that state. A
  non-finite *first* E-step now clears the posteriors (the model reports as
  unfitted) and logs a warning, instead of leaving NaN posteriors installed.
- **`SwitchingSpikeOscillatorModel.fit`** converging on exactly the last allowed
  iteration no longer warns "Reached maximum iterations" or appends a duplicate
  final log-likelihood.
- **Switching Kalman discrete-state update**: a NaN or negative
  previous-state probability or transition-matrix entry now yields a NaN
  posterior and log-likelihood (fail loud). It was previously treated as a
  structural zero or floored.
- **Switching choice filter, first trial**: an initial prior with no positive
  mass gives a NaN log-likelihood (previously a finite value); a tiny prior such
  as `1e-12` is used exactly (previously floored to `1e-10`); NaN, inf or
  negative entries are treated as structural zeros.
- **`project_correlated_noise_process_covariance`** computes the PSD shrink
  factor in closed form. On ill-conditioned inputs the result can differ from
  the old bisection by up to ~1e-2 relative (the bisection over-shrank); in
  float32 the safety margin now scales with the dtype.
- **`decode` / `predict_proba` on single-regime Hamiltonian models**
  (`HamiltonianLFPModel`, `HamiltonianSpikeModel`, `JointHamiltonianModel`)
  raise `NotImplementedError` pointing to `SwitchingHamiltonianJointModel`.
  They always failed, previously with a `RuntimeError` asking to fit first.
- **`preprocessing.binned_to_spike_times`** raises on negative counts or a
  `time_bins` length that does not match the number of bins (previously
  accepted silently).
- **EM initial-state updates use the smoothed `x_0`** (`PointProcessModel`,
  `PlaceFieldModel`, and the Kalman M-step when `initial_state_prior` is
  passed). Init-only EM is now monotone; fitted `init_mean` / `init_cov` and
  later log-likelihoods change.
- **Covariance floors are relative to the matrix scale** (`1e-8 * lambda_max`,
  `1e-10 * trace / n` in the switching M-step) instead of absolute constants,
  so M-steps at scale `1e-3` or below are no longer pinned. A warning is logged
  when a floor binds. The point-process `init_cov` clip bounds are multiples of
  the fit's initial latent scale.
- **Laplace-EKF marginal log-likelihood** takes both log-determinants from the
  Cholesky factors the update used (one jitter policy). Values move for small
  or ill-conditioned posterior covariances; the switching point-process models
  share this path.
- **Fisher-scoring line search** (`max_newton_iter > 1`) requires an Armijo
  decrease and logs a warning when more than 10% of bins exhaust the
  backtracking. The spike-GLM Newton steps use one relative ridge and a
  Cholesky solve; a non-descent direction falls back to the gradient with one
  warning instead of a silent zero step.
- **Directed-influence stability scale** is the exact spectral radius (was a
  block-row-norm bound that over-damped coupled models by up to ~20%). Stable
  parameters are a fixed point of construct / rebuild and EM started at the
  truth no longer rolls back. `compute_directed_influence_stability_scale` now
  requires `phase_difference=`.
- **`CorrelatedNoiseModel` / `CorrelatedNoisePointProcessModel` default to
  `use_reparameterized_mstep=True`**; the DIM reparameterized M-step updates Q
  at the current A (ECM style).
- **Switching-model ELBO and posterior entropies** use Cholesky
  log-determinants (NaN on an indefinite covariance instead of a finite value).
- **`PlaceFieldRateMaps.from_spike_position_data`** smooths isotropically in
  cm (`sigma` is divided by each axis's bin width), so rate maps on non-square
  arenas change.
- **`position_decoder_smoother`** uses the filter's inflation-aware
  predictions in the backward pass; smoothed output changes only with
  `adaptive_inflation`. The decoder's `dt`, `sigma_track` and grid spacing are
  traced rather than static, so parameter sweeps no longer recompile.
- **`PointProcessModel.fit`** clears the posteriors on a non-finite first
  E-step (aligned with the other EM models); its `run_em` call no longer
  passes the no-op `require_increase_to_converge`.
- **`fit_sgd` reuses its compiled step** across calls on the same model with
  same-shaped data (the default optimizer is a shared module-level instance),
  and the contingency / covariate-choice M-step optimizers compile once per
  shape. Log-likelihood histories can differ from before by up to ~1e-13.
- **`SwitchingSpikeOscillatorModel`**'s default transition diagonal follows the
  default float dtype (was float32).

### Deprecated

- **`models.stochastic_point_process_filter`** (observed-Hessian SSPPF) will be
  removed in **0.2.0**; use `point_process_kalman.stochastic_point_process_filter`.
  The warning and docstring now state the removal version.

### Removed

- `environment.yml` / `environment_gpu.yml` (superseded by `pyproject.toml` +
  `uv.lock`) and `black` from the `test` extra.
- `scripts/_test_gaussian_boundary.py` (assertion-free scipy exploration).

- **`switching_kalman_smoother(last_filter_conditional_cont_mean=...)`**: the
  argument was never read. Drop it from calls.
- **`fit` on the Hamiltonian models** (`HamiltonianLFPModel`,
  `HamiltonianSpikeModel`, `JointHamiltonianModel`,
  `SwitchingHamiltonianJointModel`): it only raised `NotImplementedError`. Use
  `fit_sgd`.
- **`project_correlated_noise_process_covariance(max_shrink_iter=...)`**: no
  longer needed by the closed-form shrink.
- **`simulate.simulate_switching_kalman.simulate_challenging_states`**: unused.

### Fixed

- **Fresh clones install**: the `neurospatial` source no longer points at a
  sibling `../neurospatial` checkout; uv resolves it from a git commit that
  declares version 0.8.0 (neurospatial 0.8.0 is not on PyPI yet, so pip users
  of the `spatial` extra must install it from GitHub first; see README).
- `import state_space_practice.kalman` no longer imports `scipy.optimize`,
  `patsy` or `sklearn` (deferred to the functions that use them); its import
  time roughly halves.
- `warnings.warn` calls in the library pass `stacklevel=2`.

- **`PlaceFieldModel` block-diagonal path uses each neuron's own A and Q
  blocks.** Previously it required identical blocks, so `fit_sgd` trained only
  neuron 0's process noise (other neurons kept their initial value), and EM
  smoothed every neuron with neuron 0's Q while the per-neuron values differed
  by less than the equal-blocks tolerance. `fit_sgd` with
  `update_transition_matrix=True` now runs on the dense path, where
  cross-neuron transition entries receive gradients.
- A rejected `fit` / `fit_sgd` of `MultinomialChoiceModel` or
  `CovariateChoiceModel` (invalid choices, too few trials or bad covariates) no
  longer changes the previously fitted model.
- `update_spike_glm_params` and the mixture Newton update accept float32
  parameters when x64 is enabled (the scan carry dtype no longer changes).
- LaTeX docstrings in `smith_learning_algorithm` are raw strings (`\frac` and
  `\text` were being rendered as control characters).
- **`kalman_filter`, `kalman_smoother`, `stochastic_point_process_filter`
  and `stochastic_point_process_smoother` work under `jax.jit` / `jax.grad`
  with the default `validate_inputs=True`** (host-side checks are skipped when
  an input is a tracer). A float32 `init_mean` / `init_cov` with float64
  parameters no longer fails with "scan carry types differ".
- **`ContingencyBeliefModel` EM** regressed each transition posterior on the
  previous trial's covariate row, so transition-covariate effects were not
  learned.
- **`CovariateChoiceModel` decay M-step** omitted the lag-one smoother
  cross-covariance, biasing the learned decay toward 0.
- **`SwitchingChoiceModel`** between-state variance is computed in centred
  form (no catastrophic cancellation for large option values).
- **`SwitchingHamiltonianJointModel`** per-state MLPs reused the parent
  model's PRNG keys; `JointHamiltonianModel` now advances its key after its
  own draws (which are unchanged).
- 0-d or 2-D `choices` (and non-2-D covariates) passed to the choice models'
  `fit` / `fit_sgd` raise `ValueError` (was `IndexError`); the choices are
  validated once.
- `models.stochastic_point_process_filter` is jitted and no longer retraces on
  every call. The position decoder's inflation statistic uses a Cholesky solve.
- `SmithLearningModel.find_first_significant_trial` is vectorized (same
  result, including NaN entries).

### Changed — default behavior (may affect existing callers)

- **`QRegularizationConfig.min_eigenvalue` default `0.01` → `None`**
  (`switching_point_process.py`). The process-noise (Q) eigenvalue floor is now
  **off by default**, so EM can learn genuinely small process noise instead of
  being pinned at `0.01`. PSD safety is still guaranteed by the `1e-8` floor in
  `_project_parameters`. Callers that relied on the old floor should pass
  `min_eigenvalue=0.01` explicitly to reproduce prior fits.

- **`get_confidence_interval` default `alpha` `0.01` → `0.05`** in both
  `point_process_kalman.py` (the free function and the `PlaceFieldModel` method)
  and `models.py` (the legacy free function, now aligned). The default interval
  is now **95%** (was 99%). Pass `alpha=0.01` explicitly for the previous 99%
  interval.
