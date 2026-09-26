# Changelog

All notable changes to `state_space_practice` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Removed

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
