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
- **Tooling**: GitHub Actions CI (ruff, ruff format, mypy on the entire
  package, fast tests on Python 3.10-3.12, nightly full suite),
  `.pre-commit-config.yaml` (ruff, ruff-format, nbstripout), `[tool.ruff]`
  config, and `HYPOTHESIS_PROFILE` selection in the test conftest. mypy checks
  all package modules except tests with the `--strict` checks enabled
  (every function annotated, generic types parameterized, no implicit
  re-exports); calls into untyped JAX APIs are exempt. SGD parameter dicts are
  typed `sgd_fitting.SGDParams` / `SGDParamSpec`. Jitted functions are declared with
  `utils.typed_jit`, a `jax.jit` that keeps the wrapped signature visible to
  mypy, so arguments and results at jitted calls are type-checked.
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
- **`kalman.smooth_initial_state_with_cross_cov`**: smoothed ``x_0`` moments
  plus ``Cov(x_0, x_1 | y) = J_0 P_{1|T}``.
- **`oscillator_utils.optimize_dim_transition_params_joint_until_stationary`**:
  restarts the DIM joint BFGS solve until it stops moving.
- **`smith_learning_algorithm.smith_laplace_log_likelihood`**: the per-step
  Laplace evidence of the Smith model (what `log_likelihood_`, `bic()` and
  `fit_sgd` now use).
- **`switching_choice.switching_choice_smoother`**: GPB1 backward pass that
  accounts for the covariate input ``B u_t``. It returns a
  `switching_kalman.SwitchingSmootherResult` (a NamedTuple, so positional
  access still works).
- **`em_driver.snapshot_attributes` / `restore_attributes` /
  `clear_attributes`** (and the `AttributeSnapshot` they pass): build
  `run_em`'s snapshot, restore and clear hooks from a tuple of attribute
  names; a snapshot carries its own keys, and `deepcopy_keys` outside them
  raise. Every `run_em` caller builds its snapshot / restore hooks from them.
- **`utils.as_2d_count_matrix`**: coerce `(n_time,)` or `(n_time, n_neurons)`
  counts to 2-D and validate them.
- **Stacked oscillator constructors in `oscillator_utils`**:
  `construct_common_oscillator_transition_matrix_stack`,
  `construct_correlated_noise_process_covariance_stack` and
  `extract_correlated_noise_params_from_covariance_stack`.
- **`seed=` on `simulate_eden_brown_2004_jump` /
  `simulate_eden_brown_2004_linear`** for a reproducible spike realization
  without building a generator. The default (`rng=None, seed=None`) is
  unchanged and nondeterministic; passing both `rng` and `seed` raises
  `ValueError`.

### Testing

- **Verification suite**: exact oracles and calibration tests that check the
  inference core against independent reference computations rather than
  against itself — `tests/oracles.py` (dense Gaussian conditioning and
  discrete path enumeration, NumPy only), `test_oracle_kalman.py`,
  `test_oracle_switching_kalman.py`, `test_oracle_point_process.py` (grid
  quadrature for the Laplace filter, asymptotic rates),
  `test_oracle_switching_point_process.py`, `test_oracle_choice.py`
  (quadrature / enumeration for the choice, Smith and contingency models),
  simulation-based calibration (`test_calibration_*.py`), finite-difference
  M-step stationarity checks for every EM M-step, gradient checks for all 15
  SGD losses (`test_gradients.py`), particle-filter references for the
  Hamiltonian EKF, and multi-seed / two-length recovery statistics
  (`test_recovery_statistics.py`). Where an approximation is inherent (GPB1/2
  collapse, Kim's smoother, the Laplace-EKF, the softmax Gaussian posterior)
  the observed gap is pinned with headroom and documented as approximation,
  not bug.
- **Property-test suites**: `test_invariances.py` (exact invariances under
  latent coordinate changes, unit rescaling, channel / neuron / option /
  state relabelling, phase shifts, time reversal and arena translation for
  every non-GP model), `test_likelihood_identities.py` (each filter's LL
  equals the sum of its own one-step predictive densities; smoother equals
  filter at ``T``; cross-covariance and Schur-complement identities; discrete
  marginal sums; ELBO identity; chain rule by restart), `test_sbc_ranks.py`
  (rank-histogram simulation-based calibration with a power guard) and
  `test_approximation_trends.py` (each documented approximation gap shrinks
  monotonically in the direction theory predicts).
- **Coupling estimators verified against ground truth**
  (`test_oracle_coupling.py`, `test_calibration_coupling.py`): stage-1
  smoother vs dense Gaussian conditioning (1e-10), the Laplace estimator vs a
  closed-form Newton MAP (2e-9), the Pólya-Gamma sampler vs the exact
  quadrature posterior, a Geweke successive-conditional test of the Gibbs
  kernel, PG-draw moments vs closed forms, calibration, and multi-seed
  recovery. `coupling_pg.pg_gibbs_sweep` (the Gibbs kernel, bit-identical
  output) and `coupling_validation.batch_means_mcse` are public; crosscheck
  records carry `pg_mean_mcse_max` and `ekf_pg_mean_max_z`.
- **GP and dynamics oracles** (`test_oracle_gp.py`, `test_oracle_dynamics.py`):
  the state-space Matérn-3/2 GP matches dense kernel regression (filtered and
  smoothed moments, full posterior covariance, marginal likelihood) to 1e-14;
  the GP-Laplace mode and evidence match a dense reference and its gap to
  quadrature is pinned and shrinks as O(1/count); the leapfrog integrator has
  order 2 against `solve_ivp`, is time-reversible to 1e-13, conserves energy
  in a bounded band and preserves phase-space volume to 1e-14 (an RK4
  comparator fails each). Hypothesis property tests for preprocessing
  (binning round trips, count conservation), circular statistics vs
  `scipy.stats`, and the behavioural-uncertainty closed forms.
- Tests that call `.fit(` / `.fit_sgd(` / `run_em(` (outside `pytest.raises`),
  directly or through a fixture, are marked `slow` automatically at collection.
- The test suite treats `DeprecationWarning`s as errors like every other
  warning. The persistent XLA compilation cache is opt-in
  (`SSP_JAX_CACHE_DIR=<dir>`); a shared default directory was corrupted by
  concurrent runs. The EM golden-value tests use an absolute floor scaled to
  each parameter array, so cross-platform round-off passes and an
  algorithm-sized change fails. The Smith property tests caught every
  exception, including their own assertion failures; they can fail now.

### Changed — behavior (may affect existing callers)

- **Smith trial-to-trial comparisons are exact**
  (`compute_trial_comparison_matrix`, `compare_two_trials` and the
  `SmithLearningModel` methods `get_trial_comparison_matrix`,
  `compare_trials`, `find_first_significant_improvement`,
  `plot_trial_comparison_matrix`): ``P(x_i > x_j | y_{1:T}) =
  Phi((m_i - m_j) / sd(x_i - x_j))`` under the joint smoothed posterior,
  replacing the 10k-sample Monte Carlo estimate. The matrix built an
  ``(n_samples, T, T)`` comparison array: peak RSS 1.0 GB at T = 100 and
  14 GB at T = 400, now ~10 MB above baseline at both (and 0.1 s instead of
  0.9-4.9 s for the first call). Values move by Monte Carlo noise only (at
  most 0.017 vs the old estimate at T = 30-50, |z| <= 3.6 over ~2000
  entries) and no longer depend on `key`. `key` and `n_samples` are still
  accepted but unused; `compare_probability` gives the same result as the
  latent comparison (the sigmoid link is strictly increasing).
- **Smith learning-curve summaries and the criterion trial are exact**
  (`calculate_probability_confidence_limits`,
  `calculate_latent_state_percentiles` and the `SmithLearningModel` methods
  `get_learning_curve`, `get_latent_state_percentiles`,
  `find_criterion_trial`, `plot_learning_curve`, `plot_summary`, `summary`):
  percentiles are the Gaussian quantiles ``m_k + s_k z_q`` (through the
  sigmoid for the probability curve; quantiles commute with the increasing
  link) and ``P(p_k > p_chance) = Phi(m_k / s_k)``, replacing 10k-sample
  Monte Carlo estimates. `find_criterion_trial` thresholded the noisy
  estimate, so on 7 of 20 simulated 60-trial sessions the criterion trial
  changed (by one trial) with the PRNG key; it is now deterministic. Values
  move by Monte Carlo noise only (at most 0.017 in probability, 0.036 in
  the latent state over 10 keys on a fitted session); percentiles 0 and 100
  are now the exact limits. `key` and `n_samples` are still accepted but
  unused.
- **`covariate_choice_smoother` uses `kalman.rts_backward_scan_with_predictions`**
  (jitted; about 10x faster per call -- the previous un-jitted RTS pass
  re-traced on every EM iteration). Its gain solve is the shared, retrying one, so smoothed values
  and fitted parameters can move at round-off level (~1e-9 relative after 200
  EM iterations).
- **The choice models skip the duplicate final E-step after convergence**
  (`MultinomialChoiceModel`, `CovariateChoiceModel`, `SwitchingChoiceModel`).
- **`PlaceFieldModel` posterior rate maps are computed by a jitted JAX
  kernel** in bounded-memory, fixed-size time chunks (`predict_rate_map`,
  `predict_center`, `drift_summary`, and
  `PlaceFieldRateMaps.from_place_field_model`). NumPy's einsum fell back to a
  naive contraction path at realistic grid sizes: one 50x50-grid map over
  2000 time bins with 49 basis functions took ~25 s and now takes ~0.2 s.
  Results agree to ~2e-15 relative.
- **The point-process Fisher-scoring line search stops at the first accepted
  step size** (a `lax.while_loop`) instead of always evaluating all 10 trial
  step sizes, so an accepted full step costs one objective evaluation instead
  of ten. Results and reverse-mode gradients are bit-identical; on typical
  data the dense `stochastic_point_process_filter` and the switching
  point-process filter run about 40% faster and the block-diagonal path
  about 20% faster.
- **Eager `utils.debug_print_if` calls no longer compile a `jit(cond)` each
  time.**
- **The DIM / DIM-PP standard-EM projection helpers are jitted**:
  `oscillator_utils.extract_dim_params_from_matrix(_stack)`,
  `construct_stable_directed_influence_transition_stack` and
  `compute_directed_influence_stability_scale` compile once per shape
  (`n_oscillators` static; `sampling_freq` / `max_spectral_radius` traced) and
  the rebuild evaluates the host spectral-radius callback once instead of
  twice. Per call they drop from 3-9 ms eager to 0.04-0.17 ms, cutting the
  M-step projection from ~33 ms to ~15 ms per EM iteration independent of T.
  XLA fusion moves outputs at round-off (~1e-15 relative per call; ~1e-9
  relative in DIM-PP parameters after 12 EM iterations).
- **The CNM / CNM-PP parameter sync is vectorized and jitted**:
  `oscillator_utils.extract_correlated_noise_params_from_covariance(_stack)`
  reduces all oscillator blocks in one pass instead of an O(n_osc²) Python
  loop of `.at[].set` updates (3 / 22 / 91 ms -> 0.01 / 0.02 / 0.06 ms per
  call at n_osc = 2 / 6 / 12; CNM standard-EM iterations at n_osc = 12 go from
  ~180-250 ms to ~40-50 ms). Variance and phase are bit-identical; coupling
  strength moves by at most 1 ulp, which the CNM-PP Laplace-EKF carries to
  ~1e-9 relative in fitted parameters after 8 EM iterations.
- **Area connectivity penalties no longer build `(n_osc)⁴` masks**:
  `oscillator_regularization.area_group_penalty`,
  `state_shared_area_penalty` and `get_area_coupling_summary` sum squared
  coupling per area pair as `M @ c**2 @ M.T` with a one-hot area assignment
  `M`. At n_osc = 100 the jitted value-and-gradient scratch memory drops from
  1.6 GB to 0.24 MB (81 ms -> ~2 ms); values and gradients match the previous
  implementation to <= 7e-16 relative.
- **`TemporalRateGP.fit_sgd` / `infer_log_rate` gradients differentiate only
  the final Newton step** (the implicit derivative at the converged mode)
  instead of all `n_iter` iterations, so gradient memory no longer grows with
  `n_iter`: at the default 25 iterations the compiled `value_and_grad`
  temporaries drop from 184 MB to 13 MB at 10^4 bins and from 1.8 GB to
  129 MB at 10^5 bins, and a gradient evaluation is ~3x faster. Forward
  results are bit-identical; gradients match the unrolled ones to ~1e-14
  relative once the iteration has converged, so SGD trajectories move at
  round-off level.
- **`temporal_rate_gp.infer_log_rate` / `infer_log_rate_batch` are jitted**
  (input validation stays eager): every call used to re-compile its Newton
  scan, so a repeat call with same-shaped inputs, including new
  hyperparameter values, now compiles nothing (~3.5x faster per call at 2000
  bins: 0.42 -> 0.12 s single, 0.52 -> 0.17 s for 3 trains). The fused
  program moves results at round-off level (log-rates ~1e-15, evidence
  ~1e-15 relative), which also reaches `TemporalRateGP`'s fitted posterior.
- **Reading a fitted attribute before fitting raises `NotFittedError`**
  instead of returning `None` (or a bare `AttributeError`). Fitted outputs of
  every model (smoothed/filtered states, posteriors, log likelihoods,
  histories, ...) are declared with the new `fitted_state.FittedAttribute`,
  so type checkers see `Array` rather than `Array | None`. The error is also an
  `AttributeError`, so `hasattr(model, "smoother_mean")` is `False` before
  fitting; `fitted_state.is_set(model, name)` tests without raising. Values
  that can legitimately be `None` after fitting (e.g. covariate weights of a
  model without covariates, GPB2-only smoother outputs under GPB1, `n_iter_`
  after an SGD-only fit) keep `None`. `PointProcessModel` and `PositionDecoder`
  raise `NotFittedError` (a `RuntimeError`) where they raised `RuntimeError`.
- **Subclasses that could not stand in for their parent no longer inherit
  from it**: `CovariateChoiceModel` takes covariates positionally after
  `choices` in `fit` / `fit_sgd`, so it now derives from the shared
  `_MultinomialChoiceBase` instead of `MultinomialChoiceModel`; and
  `SwitchingHamiltonianJointModel`, whose `filter` / `smooth` also return
  discrete-state probabilities, derives from the shared `_JointHamiltonianBase`
  instead of `JointHamiltonianModel`. Behavior and signatures are unchanged;
  only `isinstance` / `issubclass` against the former parent change.
- **`DirectedInfluenceModel.fit_sgd`**: `connectivity_penalty` now follows
  `skip_init` (it was inserted before it, so a positional `skip_init` bound to
  the penalty).
- **Long result tuples are NamedTuples**: `switching_kalman_filter` and
  `switching_point_process_filter` return `SwitchingFilterResult`,
  `switching_kalman_smoother` / `switching_kalman_smoother_gpb2` return
  `SwitchingSmootherResult` / `SwitchingSmootherGPB2Result`,
  `switching_kalman_maximization_step` returns `SwitchingMStepResult`,
  `smith_learning_filter` returns `SmithFilterResult`, `matern32_continuous`
  returns `Matern32SDE`, and the Eden & Brown 2004 simulators return
  `EdenBrownJumpSimulation` / `EdenBrownLinearSimulation`. Positional
  unpacking and indexing are unchanged; fields can now be read by name. The
  types are JAX pytree nodes distinct from `tuple`, so code that mixes them
  with plain tuples in one `lax.cond` / `tree_map` must use the same type.
- **Public array inputs are typed `ArrayLike`**: every public function and
  method that takes a JAX array (except PRNG `key` arguments) is annotated
  `jax.typing.ArrayLike` and converts with `jnp.asarray`, so numpy inputs
  type-check in user code as they already ran. Return types stay `Array`.
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
  point-process models' `decode` / `predict_proba`, the choice models and
  `SmithLearningModel`.
  `PlaceFieldModel.n_free_params` on an unfitted model raises it instead of
  an `AssertionError`.
- **`SGDFittableMixin.fit_sgd` is declared `(*args, **kwargs)`**; the optimizer
  settings (`optimizer`, `num_steps`, `verbose`, `convergence_tol`) are still
  keyword-only with the same defaults.
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
- **M-step covariance floors act on the correlation scale**: the eigenvalues
  of ``D^{-1/2} C D^{-1/2}`` (``D`` the diagonal of ``C``) are floored at
  `1e-8` of the largest (`1e-10` in the switching M-step) and the result is
  rescaled, instead of flooring ``C`` at an absolute constant. Floors no
  longer pin M-steps at scale `1e-3` or below, nor inflate a small-variance
  coordinate next to a large one. A warning is logged when a floor binds; a
  materially negative eigenvalue (an inconsistent estimate, not rank
  deficiency) gets its own warning naming the minimum eigenvalue. The
  point-process `init_cov` clip bounds are multiples of the latent scale
  recorded when the fit initialises (kept across `fit(skip_init=True)`
  refits).
- **Laplace-EKF marginal log-likelihood** takes both log-determinants from the
  Cholesky factors the update used (one jitter policy). Values move for small
  or ill-conditioned posterior covariances; the switching point-process models
  share this path.
- **Fisher-scoring line search** (`max_newton_iter > 1`) requires an Armijo
  decrease and logs a warning when more than 10% of bins exhaust the
  backtracking — from every Laplace-EKF caller (point-process filter and
  smoother, switching point-process filter, position decoder, coupling EKF),
  including inside `jax.jit` / `jax.grad`. `glm_laplace_update` and
  `switching_point_process.point_process_kalman_update` take
  `return_line_search_failures=`. The spike-GLM Newton steps use one relative ridge and a
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
  E-step (aligned with the other EM models).
- **`fit_sgd` reuses its compiled step** across calls on the same model with
  same-shaped data (the default optimizer is a shared module-level instance),
  and the contingency / covariate-choice M-step optimizers compile once per
  shape. Log-likelihood histories can differ from before by up to ~1e-13.
- **`SwitchingSpikeOscillatorModel`**'s default transition diagonal follows the
  default float dtype (was float32).
- **Scale-relative linear algebra throughout.** `utils.psd_solve` /
  `psd_cholesky` shift the diagonal by a per-entry relative amount
  (``1e-12 |A_ii|`` in float64, machine epsilon in float32) instead of an
  absolute ``1e-9``; `diagonal_boost` now defaults to ``0.0``. The Kalman and
  RTS gain solves use their own relative shift. The Kalman, point-process and
  switching filters are now invariant to the units of the state: with the
  old absolute shift, models whose covariances were below ~1e-6 (e.g. volts
  instead of millivolts) returned means off by more than one posterior
  standard deviation, and at scale 1e-10 the point-process filter was wrong
  by a factor of 40. Place-field process-noise estimates (~1e-6) were biased
  by ~0.8%. The position decoder keeps an explicit ``1e-9 cm^2`` floor.
- **EM is exact for the initial transition.** `kalman_maximization_step`,
  `point_process_kalman.dynamics_only_m_step` (with `initial_state_prior`)
  and `PlaceFieldModel` count the ``x_0 -> x_1`` transition in the ``A`` /
  ``Q`` statistics (``T`` transitions, ``Q`` divided by ``T``); the
  finite-difference gradient of the exact expected complete-data
  log-likelihood is zero at the returned parameters.
- **Laplace-mode Newton steps are line-searched** (multinomial, covariate and
  switching choice; Smith `fit_sgd` and filter). The undamped 3-step Newton
  oscillated when the prior mean sat on the saturated side of the softmax
  (filter means off by up to 21 units; a 400-trial log-evidence at
  ``beta=4`` of −1836 against the exact −118). The default is now 10
  line-searched iterations; well-behaved updates are unchanged.
  `MultinomialChoiceModel`'s inverse-temperature M-step keeps the best of the
  refined, best-grid and current values, so it never decreases the marginal LL.
- **`SmithLearningModel` reports the Laplace evidence** instead of the
  plug-in ``sum_k log Binom(y_k | sigmoid(mu + m_{k|k-1}))``, which ignored
  the predictive variance and biased `fit_sgd` towards too little process
  noise. `log_likelihood_`, `bic()`, `compare_to_null` and the SGD loss
  change accordingly.
- **Laplace-mode Newton convergence is checked** (choice models, Smith): each
  filter call warns (`StateSpaceWarning`) when an update ends more than
  `multinomial_choice.NEWTON_GAP_TOL` (1e-6) nats below its mode by the
  Newton-decrement estimate. The line search takes the Armijo-acceptable step
  with the highest log posterior (the full step within round-off), which
  stops the zig-zag that left ~1 in 2000 updates unconverged.
- **More fitting fallbacks are reported**: `temporal_rate_gp.infer_log_rate`
  / `infer_log_rate_batch` warn on a non-finite merit, an unaccepted step or
  a non-converged mode (`LaplaceRateResult` gains `n_nonfinite_merit` and
  `n_unaccepted_steps`); the position decoder warns when more than 10% of
  bins hit `max_alpha`; the switching M-step warns about near-empty states
  even without `previous_params`; a rejected DIM projected M-step and a
  skipped warm-init seeding are logged at WARNING (were INFO / DEBUG).
- **Warm init of COM / CNM / DIM** seeds each discrete state's parameters
  with one M-step on its GMM window cluster; it previously only set the
  first-step probabilities and initial state, which has no effect at a
  symmetric start (first E-step accuracy on the DIM fixture 0.51 → 0.84).
- **`SwitchingChoiceModel`** smooths with the covariate input; with
  covariates every smoothed mean was previously compared against a
  prediction missing ``B u_{t+1}``.
- **DIM / DIM-PP M-steps**: the joint BFGS solve restarts on a line-search
  failure (it previously stopped at gradient norms of 67–173), and the
  projected (standard) path keeps the previous dynamics when the Frobenius
  projection would lower the M-step objective (generalized EM).
- **`max_newton_iter` defaults to 3** (was 1) in `stochastic_point_process_filter`
  / `_smoother`, `PointProcessModel`, `PlaceFieldModel`, the point-process
  oscillator models and `SwitchingSpikeOscillatorModel`. One Fisher-scoring
  step biased the switching spike-oscillator's low-noise process variance
  about 2x upward even at ``T = 3200`` and could diverge from a broad prior;
  three steps remove most of the bias. Block and dense place-field paths now
  agree to round-off only when no line search backtracks.
- **Default `diagonal_boost` of `switching_point_process.point_process_kalman_update`**
  is ``0.0`` (relative shift), like the other Laplace updates.
- **`circular_stats.circular_std` delegates to `scipy.stats.circstd`** (with
  the existing ``R >= 1e-10`` floor). The other circular functions stay
  hand-written where SciPy has no equivalent or different conventions
  (documented per function). `rayleigh_test` applies its small-sample
  correction at every ``n`` (it stopped at 50, so p jumped 14% at ``z = 6``);
  `circular_correlation` is documented as Jammalamadaka–SenGupta.
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

### Deprecated

- **`models.stochastic_point_process_filter`** (observed-Hessian SSPPF) will be
  removed in **0.2.0**; use `point_process_kalman.stochastic_point_process_filter`.
  The warning and docstring now state the removal version.
- **`kalman_maximization_step(initial_state_prior=None)`** and
  **`point_process_kalman.dynamics_only_m_step(initial_state_prior=None)`**
  (the ``x_1``-prior update, which is not an EM step) warn and will be
  removed in **0.2.0**; pass an `InitialStatePrior`.

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
- **`point_process_kalman.smoothed_initial_transition_moments`**: an alias of
  `kalman.smooth_initial_state_with_cross_cov`; call that instead.

### Fixed

- **Hamiltonian `fit_sgd` gradient memory no longer scales with the MLP
  size**: the EKF predict steps inside the Hamiltonian filter and smoother
  scans (`run_ekf_filter` / `run_ekf_smoother` and the switching
  predict-and-collapse step) run under `jax.checkpoint`, so reverse mode
  recomputes the leapfrog Jacobian instead of storing the MLP's
  second-derivative residuals at every time step. For `hidden_dims=[32, 32]`,
  peak memory of a 20,000-step `SwitchingHamiltonianJointModel.fit_sgd`
  (3 states) drops from ~10 GB to ~1.1 GB, and `HamiltonianLFPModel.fit_sgd`
  from ~1.7 GB to ~0.7 GB (5.6 GB to 0.8 GB at 100,000 steps), with no
  slowdown. Filter and smoother outputs are bit-identical; gradients agree
  to round-off.
- **`SmithLearningModel.fit_sgd` compiled its SGD step twice per call**: the
  `init_learning_state` parameter was a weakly typed scalar
  (`jnp.array(python_float)`) that the first optimizer update made strong,
  forcing a retrace. The parameters are now built with an explicit dtype,
  so the step is traced once (first `fit_sgd` on 100 trials 2.4-2.5 s ->
  1.5-1.6 s); fitted values are bit-identical.
- **Smith filter log posterior no longer saturates to `-inf`**: the
  per-trial objective used `binom.logpmf(p=sigmoid(mu + x))`, which is
  `-inf` with NaN gradient and Hessian once `sigmoid` rounds to 1 on a trial
  with an error (`mu + x >~ 37`) or to 0 on a trial with a success. It now
  uses the `log_sigmoid` form shared with `smith_laplace_log_likelihood`
  (mathematically identical). In the normal range values change by at most
  1.5e-11 relative (the old form's rounding of `1 - sigmoid` near
  `|mu + x| = 16`; the new form is within 4e-16 of an exact reference);
  filter outputs and EM fits move by <= 1.4e-14.
- **Importing `simulate.simulate_switching_kalman` no longer reseeds the
  global NumPy RNG** (it ran `np.random.seed(0)` at import), and
  `simdata_settings` / `simulate_model` no longer draw the initial state from
  the global RNG, so consecutive `simulate_model()` calls return the same
  data. New `init_seed` (initial state, default 0 -- the value the old
  import-time seeding gave on a first call) and `simulate_model(noise_seed=14)`
  (the existing noise seed) arguments; `None` makes either part
  nondeterministic.
- **A failed fit no longer leaves stale or non-finite results that look
  fitted**: when a fit raises `NonFiniteLikelihoodError` (its starting
  parameters give a non-finite log-likelihood), it first clears every fit
  output -- posteriors, cached inference results, uncertainty summaries and
  `log_likelihood_` / `log_likelihood_history_` / `converged_` / `n_iter_`
  (declared per model as `_fit_output_attrs`, cleared by
  `SGDFittableMixin._clear_fit_state`). Previously the choice and belief
  models kept the earlier fit's results next to the newly bound data, and the
  switching point-process models kept non-finite posteriors. `run_em`'s
  `"raise"` policy now accepts a `clear_state` hook. `fit_sgd` raises
  `NonFiniteLikelihoodError` (after clearing) when the log-likelihood at the
  fitted parameters is non-finite, instead of returning with
  `log_likelihood_ = NaN`; a multi-restart switching point-process fit whose
  restarts all fail raises it too (was a plain `ValueError`). The choice and
  belief models' `fit` rejects `max_iter < 1` before binding data, `fit_sgd`
  validates `convergence_tol` (finite, >= 0) before touching the model, and
  the oscillator models reject non-2-D observations with `ValueError` (was
  `IndexError`).
- **The choice and belief models' EM no longer runs on after a non-finite
  log-likelihood**: `MultinomialChoiceModel`, `CovariateChoiceModel`,
  `SwitchingChoiceModel` and `ContingencyBeliefModel` run their own EM loops,
  which recorded a NaN log-likelihood and kept iterating, so an M-step that
  produced a NaN parameter left the model with NaN parameters, posteriors and
  `log_likelihood_`, flagged only by "did not converge". Like
  `em_driver.run_em`, a non-finite first E-step now raises
  `NonFiniteLikelihoodError`, and a later one (including the final E-step
  after `max_iter`) restores the last accepted parameters and their
  posteriors, warns and stops. Their convergence criteria (relative or
  absolute `tolerance`) are unchanged, and finite fits are bit-identical.
  After a rollback stop the models no longer also log the misleading "did not
  converge in max_iter EM iterations" warning; `MultinomialChoiceModel` /
  `CovariateChoiceModel` now warn when their monotonicity guard rolls back a
  decreasing step.
- **Repeat `fit_sgd` calls reuse the compiled SGD step**: losses read trained
  parameters as `params.get(key, self.<attr>)`, so the compiled-step cache
  (which fingerprints model attributes read while tracing) missed after every
  fit rewrote them, and each repeat `fit_sgd` re-traced and recompiled --
  about 0.4-2 s per call at test sizes for `PointProcessModel`,
  `PlaceFieldModel.fit_sgd(warm_start=False)`, the Gaussian and point-process
  oscillator models and `SwitchingSpikeOscillatorModel` (now ~0.04-0.35 s).
  Losses read parameters through the new `SGDFittableMixin._sgd_param` /
  `_sgd_per_state_param`, which touch the attribute only when the parameter is
  not optimized. `DirectedInfluenceModel` / `DirectedInfluencePointProcessModel`
  pass `freqs` and `damping_coef` as frozen SGD parameters (warm-start seeding
  and EM re-derive them from the transition matrix), which moves their SGD
  results at round-off level (~1e-15 relative). `SwitchingChoiceModel` also
  recompiled on every call, because `parameter_transforms.positive_capped`
  built a new transform each time, so equal parameter specs compared unequal;
  it now returns one transform per cap. Results of every other model are
  bit-identical.
- **`TemporalRateGP.fit_sgd` compiles its SGD step once per fit**: the initial
  hyperparameters were weak-typed (built from Python floats) and the first
  optimizer update returned them strong-typed, so the second step re-traced
  and re-compiled the step (~1.5 s extra per fit at 2000 bins). They are now
  built with an explicit float dtype; results are bit-identical.
- **A rejected `fit_sgd` call no longer changes the model**: model-specific
  setup (data binding, initialization, warm start, recorded lengths) ran in
  each model's `fit_sgd` override before the shared settings checks, so e.g.
  `fit_sgd(data, num_steps=-1)` raised only after re-initializing the model --
  on a fitted oscillator model or `PlaceFieldModel` that discarded the fit.
  That setup now runs in the `_prepare_sgd_data` hook, after `num_steps` and
  the optimizer are validated, and data is validated before anything is
  stored. The oscillator models' `fit_sgd` also rejects a wrong observation
  column count with `ValueError` (it re-initialized, then failed with
  `TypeError`/`IndexError`). Removed the private `TemporalRateGP._counts` and
  the `RuntimeError` guards against calling `SGDFittableMixin.fit_sgd`
  directly on `PlaceFieldModel` / `TemporalRateGP`, which the hook makes
  unnecessary, and the `SGDFittableMixin._check_sgd_initialized` hook, which
  no model overrides any more (initialization belongs in `_prepare_sgd_data`).
- **Fit results no longer go stale when switching between `fit` and
  `fit_sgd`**: every EM `fit` and `fit_sgd` now records `log_likelihood_`,
  `log_likelihood_history_`, `converged_` and `n_iter_` together (through
  `SGDFittableMixin._record_fit_result`). Before, e.g. `PointProcessModel.fit`
  after `fit_sgd` kept the SGD log-likelihood, history and convergence flag;
  `PointProcessModel.fit`, `SwitchingSpikeOscillatorModel.fit`,
  `PlaceFieldModel.fit` and the oscillator models' `fit` did not set
  `log_likelihood_` at all; and `n_iter_` survived a later `fit_sgd`.
  `n_iter_` is now available on every `SGDFittableMixin` model: the number of
  log-likelihoods an EM fit recorded (EM's final synchronising E-step can
  make it `max_iter + 1`), `None` until an EM fit and after `fit_sgd`. Multi-restart switching point-process fits report the best
  restart's convergence flag. After `fit_sgd`, `log_likelihood_` is the
  marginal log-likelihood from the final inference at the fitted parameters
  (each model's `_finalize_sgd` now returns it), not the training objective:
  with a connectivity penalty the two differ, and `log_likelihood_history_`
  records the objective. A zero-step `fit_sgd` also sets `log_likelihood_`.
- **Switching point-process `fit_sgd` validates spike shape like `fit`**: a
  1-D array or a wrong neuron count raised a misleading
  "spike_params.baseline neuron dimension must match ..." error, after
  initializing the model; it now raises the same "must be 2D" / "must match
  n_neurons" errors as `fit`, before touching the model.
- **Multi-restart switching point-process fits no longer hide real errors**:
  `fit(n_restarts>1)` treated every `ValueError` (e.g. from initialization or
  the M-step) as a non-finite-likelihood restart and finally reported "All N
  restarts failed with non-finite log-likelihood". It now skips only restarts
  whose first E-step is non-finite; `run_em` raises the new
  `NonFiniteLikelihoodError` (a `ValueError` subclass, exported at package
  level) for that case, and other errors propagate.
- **Plotting into axes inside a matplotlib SubFigure**: the plot methods of
  `MultinomialChoiceModel`, `CovariateChoiceModel`, `PlaceFieldModel` and
  `PositionDecoder` that accept `ax` called `tight_layout` on `ax.figure`, which
  is the SubFigure (no `tight_layout`) and raised `AttributeError`; they now
  use the root Figure.
- **Fresh clones install**: the `neurospatial` source no longer points at a
  sibling `../neurospatial` checkout; uv resolves it from a git commit that
  declares version 0.8.0 (neurospatial 0.8.0 is not on PyPI yet, so pip users
  of the `spatial` extra must install it from GitHub first; see README).
- `import state_space_practice.kalman` no longer imports `scipy.optimize`,
  `patsy` or `sklearn` (deferred to the functions that use them).
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
  with the default `validate_inputs=True`**. Concrete inputs, including
  constants closed over by a jitted function, are validated on the host;
  when an input is traced the host checks are skipped and a non-positive-
  definite `init_cov` is reported by an in-graph `StateSpaceWarning` at run
  time. A float32 `init_mean` / `init_cov` with float64
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
- **Kalman / RTS gain solves** used an absolute ``1e-9`` Cholesky shift;
  small-unit models were wrong (see *Scale-relative linear algebra*).
- **`SwitchingSpikeOscillatorModel` M-step** installed the ``Q`` that is
  optimal for the unconstrained ``A*`` and then projected ``A``; ``Q`` is now
  re-estimated at the installed ``A``.
- **`PlaceFieldModel._m_step`** with ``update_transition_matrix=False``
  computed ``Q`` at a hard-coded identity while smoothing ``x_0`` with the
  model's ``A``; it now uses the model's ``A`` for both.
- **`SmithLearningModel.maximization_step`** returned ``P_0 = P_{1|T}`` for
  the initial-state variance; the maximiser is ``P_0 = P_{1|T} - sigma^2``
  (or the KKT solution at the ``P_0 >= 0`` floor).
- **`simulate_coupling` LFP noise** used a PRNG key that collides with the
  process-noise key under JAX's partitionable threefry
  (``fold_in(k, 1) == split(k, 3)[1]``), so the simulated LFP noise was a
  scaled copy of the process noise. Latent and spikes per seed are unchanged;
  the LFP differs. Experiment write-ups produced with the old simulator
  (`experiments/coupling_ekf_vs_pg/conclusion.md`) need re-running.
- **Softmax Laplace evidence** (multinomial, covariate and switching choice)
  took its log-determinants through an absolute ``1e-9`` shift while the
  update used scale-relative shifts, so the evidence was not invariant to the
  units of the latent state (0.29 nats per trial at prior variance 1e-10).
  Both log-determinants now come from the update's own Cholesky factors.
- **float32 Kalman / RTS gain solves** could return NaN (e.g. rank-deficient
  dynamics with zero process noise): the ``1e-14`` relative shift rounded
  away. It is now at least machine epsilon of the dtype, or ``sqrt(eps)``
  when the smaller shift gives a non-finite Cholesky factor. The successful
  factor is reused for the solve and its implicit derivatives, keeping
  gradients finite without an extra factorization on the successful path.
- **`stabilize_transition_matrix`** truncated integer inputs (``[[2, 1],
  [0, 2]]`` became zeros); integers are promoted first. A `block_size` that
  does not divide the dimension is logged before the uniform-scale fallback.
- **`softmax_observation_update`** raised a scan-carry `TypeError` for integer
  or float32 (with x64 on) priors; `multinomial_choice_filter` and
  `smith_learning_filter` likewise for integer / float32 initial states.
  Inputs are promoted to one float dtype (float32 stays float32).
- **`MultinomialChoiceModel`**'s one-point-grid inverse-temperature M-step
  skipped the comparison with the current value.
- **`optimize_dim_transition_params`** never applied its per-oscillator
  spectral clamp (the sigmoid-parameterised coupling is never exactly 0); the
  uncoupled case is now decided from the caller's coupling.
- **Point-process `init_cov` clip ceiling** was re-anchored on each
  `fit(skip_init=True)`, so it doubled on every warm restart.
- The spike-GLM gradient-fallback warning is a `StateSpaceWarning` (was a
  plain `UserWarning`).
- **Point-process Fisher line search** froze at a converged mode (strict
  decrease test), so reverse-mode gradients through the scan carried a 2%
  error; a round-off slack accepts the negligible full step.
- **`temporal_rate_gp.infer_log_rate`** Newton steps had no step control: with
  a baseline far below the data (the default ``mean=0`` on a 50 Hz train) the
  default 25 iterations returned an unconverged mode and an evidence off by
  tens of nats up to ~1e5. Steps are now damped by an Armijo line search on
  the exact log-posterior; converged problems are unchanged to 1e-13.
- **`behavioral_uncertainty.categorical_entropy`** added 2.3e-9 nats per zero
  entry (clipping); it is exact now with finite gradients.
  `bernoulli_mixture_mean_variance` returns ``mean (1 - mean)`` and can no
  longer be slightly negative.

### Known approximation limits (documented and pinned by tests)

- A single Fisher-scoring step (``max_newton_iter=1``) biases
  `SwitchingSpikeOscillatorModel`'s low-noise process variance about 2x
  upward even at ``T = 3200`` and can diverge from a broad prior in
  `PointProcessModel`; the default of 3 steps removes most of the bias.
- The softmax Gaussian posterior of the choice models is over-confident at
  high inverse temperature (90% intervals cover ~69% at ``beta = 5``; the
  exact posterior on the same data is calibrated), and the Smith smoother at
  ``sigma^2 = 1`` covers ~84%. Both are calibrated in the ordinary regime.
- GPB1's discrete smoother (Kim's recursion) can be off from the exact
  path-enumeration posterior by up to 0.25 in probability on 5-step
  problems; GPB2 by 0.05. The Laplace-EKF's marginal log-likelihood is 3–5
  nats too high for the Smith model at ``beta = 12``.
- The position decoder's smoothed covariance is ~9% too small (bilinear
  log-rate surrogate), and its filtered variance can collapse on the
  realistic-trajectory fixture (kept finite by the ``1e-9 cm^2`` floor).
- `utils.debug_print_if` fires on every element under `jax.vmap` (both
  `lax.cond` branches run), so vmapped switching filters emit spurious
  warnings; values are unaffected.
- The coupling estimators fit a static Bernoulli-logit regression on the
  plug-in smoothed LFP; ignoring the smoother's uncertainty costs calibration
  (90% coverage 0.86 at the default ``lfp_noise_var``, 0.69 at 4.0) but not
  location. The PG chain mixes slowly under near-separation with a diffuse
  prior (500 draws cover 0.87; 4000 sweeps 0.91).
- The Smith model with zero process noise matches the static binomial MAP
  only to ~0.1 posterior sd: each sequential Laplace step carries its
  approximation forward (the gap shrinks with ``T``).
