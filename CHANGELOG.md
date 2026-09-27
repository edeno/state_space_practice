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
