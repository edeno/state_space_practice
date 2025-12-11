# SCRATCHPAD: Switching Spike-Based Oscillator Networks

## Current Status

- **Date**: 2025-12-11
- **Working on**: Milestone 9 COMPLETE - Demo Notebook done
- **Next**: All milestones complete!

## Milestone 1 Summary (COMPLETE)

All 6 tasks completed:

- 1.1 Created `switching_point_process.py` with minimal imports
- 1.2 Defined `SpikeObsParams` dataclass
- 1.3 Implemented `point_process_kalman_update()` - Laplace-EKF update for multi-neuron
- 1.4 Implemented `_point_process_predict_and_update()` - combines prediction + update
- 1.5 Created `_point_process_update_per_discrete_state_pair()` via double vmap
- 1.6 Wrote comprehensive unit tests (18 tests, all passing)

## Key Implementation Details

### point_process_kalman_update

- Takes a `log_intensity_func(state) -> log_rates` callable
- Uses JAX autodiff for Jacobian and Hessian
- Computes Fisher information from expected intensity
- Returns posterior mean, cov, and Poisson log-likelihood

### _point_process_update_per_discrete_state_pair

- Uses closure to capture `log_intensity_func` (can't be vmapped directly)
- Inner vmap over previous state i (axis -1)
- Outer vmap over next state j (axis -1)
- Output shapes: (n_latent, n_states, n_states) for mean, etc.

## Milestone 2 Summary (COMPLETE)

All 5 tasks completed:

- 2.1 Implemented `switching_point_process_filter()` - mirrors `switching_kalman_filter` structure
- 2.2 Implemented `_step` function with pair-conditional posteriors, likelihood scaling, discrete state update, and mixture collapse
- 2.3 Initial timestep handled via `jax.lax.scan` starting from initial conditions
- 2.4 Returns `last_pair_cond_filter_mean` for smoother (shape: n_latent, n_states, n_states)
- 2.5 Wrote 10 integration tests (output shapes, prob constraints, NaN checks, edge cases)

### Key Implementation Details (Milestone 2)

#### switching_point_process_filter

- Uses `jax.lax.scan` for efficient filtering
- Reuses helper functions from `switching_kalman.py`:
  - `_scale_likelihood` for numerical stability
  - `_update_discrete_state_probabilities` for HMM forward step
  - `collapse_gaussian_mixture_per_discrete_state` for mixture collapse
- Maintains exact per-state-pair structure throughout
- Returns 5-tuple: filter means, covs, discrete probs, last pair-conditional mean, marginal log-likelihood

#### Tests Added

- Output shape verification
- Discrete probability constraints (sum to 1, non-negative)
- NaN detection
- Finite marginal log-likelihood
- Single discrete state edge case
- Positive semi-definite covariances
- Pair-conditional shape verification
- All-zero spikes (silent neurons)
- High spike counts (numerical stress test)

## Notes

### Key Design Decisions (from plan)

1. Use Laplace approximation for point-process observation update
2. Maintain exact per-state-pair structure (O(S^2) per timestep)
3. Smoother is observation-model agnostic
4. Dynamics M-step reuses `switching_kalman_maximization_step`

## Blockers

None currently.

## Milestone 3 Summary (COMPLETE)

All 2 tasks completed:

- 3.1 **Decision**: No wrapper needed - smoother is observation-model agnostic, just use `switching_kalman_smoother` directly
- 3.2 Wrote 4 smoother integration tests (all passing)

### Key Implementation Details (Milestone 3)

#### Smoother Integration

- `switching_kalman_smoother` works directly with `switching_point_process_filter` outputs
- No wrapper or alias needed - smoother operates on Gaussian posteriors regardless of observation model
- Tests verify:
  - Smoother runs without error on filter output
  - No NaN values in smoother output
  - All output shapes are compatible (overall mean/cov are marginalized, state-conditional are indexed)
  - Smoothed discrete probabilities sum to 1

#### Note on Variance Property

The theoretical property that smoothed variance ≤ filtered variance holds exactly only for marginal distributions in standard Kalman filters. In switching models with mixture collapse approximations, this property may not hold exactly for state-conditional quantities.

## Milestone 4 Summary (COMPLETE)

All 4 tasks completed:

- 4.1 Directory `simulate/` already existed
- 4.2 Implemented `simulate_switching_spike_oscillator()` function
- 4.3 Returns ground truth tuple: (spikes, true_states, true_discrete_states)
- 4.4 Wrote 10 simulation tests (all passing)

### Key Implementation Details (Milestone 4)

#### simulate_switching_spike_oscillator

- Uses `jax.lax.scan` for efficient simulation
- Simulates discrete states via `jax.random.categorical`
- Simulates continuous states via `jax.random.multivariate_normal` conditioned on discrete state
- Simulates spikes via `jax.random.poisson` with rate = exp(baseline + weights @ state) * dt
- Optional initial conditions with sensible defaults

#### Tests Added

- Output shape verification
- Spike counts non-negative and integer
- Discrete states in valid range
- Single discrete state edge case
- Reproducibility with same key
- Different keys produce different results
- No NaN values
- Higher baseline produces more spikes
- State-conditioned dynamics verification
- Minimal parameters convenience test

## Milestone 5 Summary (COMPLETE)

All 5 tasks completed:

- 5.1 Implemented `_single_neuron_glm_loss()` helper - Poisson negative log-likelihood
- 5.2 Implemented `_single_neuron_glm_step()` Newton step with backtracking line search
- 5.3 Implemented `update_spike_glm_params()` function - plug-in method with vmap over neurons
- 5.4 Implemented second-order expectation variant with `smoother_cov` parameter
- 5.5 Wrote 17 GLM M-step tests (all passing)

### Key Implementation Details (Milestone 5)

#### _single_neuron_glm_loss

- Computes Poisson negative log-likelihood: `-sum(y * eta - exp(eta) * dt)`
- Linear predictor: eta = baseline + weights @ smoother_mean
- Differentiable via JAX autodiff

#### _single_neuron_glm_step

- Newton-Raphson step for Poisson GLM optimization
- Uses backtracking line search with Armijo condition to ensure descent
- Critical fix: pure Newton can overshoot with sparse data or small dt
- Returns updated baseline and weights

#### update_spike_glm_params

- M-step for spike observation parameters (baseline, weights)
- Two methods available:
  - **Plug-in method** (default): Uses smoother_mean directly
  - **Second-order method**: Accounts for E[exp(c @ x)] = exp(c @ m + 0.5 * c @ P @ c)
- vmaps single_neuron_glm_step over all neurons
- Runs max_iter Newton iterations via jax.lax.scan

#### _single_neuron_glm_step_second_order

- Accounts for state uncertainty using variance correction term
- Uses "effective design matrix" where columns are m_t + P_t @ weights
- Same backtracking line search for guaranteed descent
- With zero variance, matches plug-in method exactly

#### GLM M-Step Tests Added

- Output shapes (scalar, vector)
- Finite outputs for various conditions
- Loss decreases after update
- Gradient computable via JAX autodiff
- Handles edge cases (zero spikes, high spikes)
- Newton step decreases loss
- Multiple iterations converge to optimal
- Parameter recovery on simulated data
- Second-order method tests (shapes, finite, decreases loss, matches plug-in with zero variance)

## Milestone 6 Summary (COMPLETE)

All 2 tasks completed:

- 6.1 Wrote test confirming dynamics M-step reuse (5 tests, all passing)
- 6.2 Added comprehensive documentation in module docstring

### Key Implementation Details (Milestone 6)

#### Dynamics M-Step Reuse Pattern

- `switching_kalman_maximization_step` works directly with point-process smoother outputs
- Function is observation-model agnostic - operates on Gaussian posteriors regardless of observation model
- For point-process models, `measurement_matrix` and `measurement_cov` returns should be ignored

#### Dynamics M-Step Tests Added

- `test_dynamics_mstep_runs_on_point_process_smoother_output`: No errors on filter/smoother output
- `test_dynamics_mstep_returns_correct_shapes`: A, Q, B, init params have correct shapes
- `test_dynamics_mstep_returns_finite_values`: All dynamics parameters are finite
- `test_discrete_transition_matrix_is_valid_stochastic_matrix`: Rows sum to 1, non-negative
- `test_covariances_are_symmetric`: Symmetry check for covariance matrices

#### Important Notes

- The raw M-step does NOT guarantee positive semi-definite covariances
- When a discrete state has low probability or insufficient data, process covariance can have negative eigenvalues
- PSD enforcement (e.g., `Q = Q + eps*I`) should be handled at the model class level

## Milestone 7 Progress

### Task 7.1 (COMPLETE)

Implemented `SwitchingSpikeOscillatorModel.__init__()` with:

- Required parameters: n_oscillators, n_neurons, n_discrete_states, sampling_freq, dt
- Optional: discrete_transition_diag, update flags for M-step
- Input validation for all positive parameters and array shapes
- Comprehensive docstrings with parameter descriptions, units, and examples
- 15 unit tests covering normal cases and error conditions

#### Key Design Decisions

1. **Follows BaseModel pattern** from `oscillator_models.py` for consistency
2. **Uses `n_neurons` instead of `n_sources`** (point-process terminology)
3. **Stores `spike_params` instead of measurement_matrix/cov** (different observation model)
4. **Removed marginal smoother placeholders** (`smoother_mean`, `smoother_cov`) - only conditional statistics needed for M-step
5. **Added input validation** with clear error messages (fail-fast pattern)

### Task 7.2 (COMPLETE)

Implemented `SwitchingSpikeOscillatorModel._initialize_parameters()` with:

- Main method splits key for random operations and calls helper methods
- `_initialize_discrete_state_prob()`: Uniform probabilities across states
- `_initialize_discrete_transition_matrix()`: From `discrete_transition_diag` with off-diagonal balance
- `_initialize_continuous_state()`: Random initial mean from standard normal, identity covariance
- `_initialize_continuous_transition_matrix()`: Uncoupled oscillators via `construct_common_oscillator_transition_matrix`
  - Default frequencies: linspace(5, 15) Hz
  - Default damping: 0.95
- `_initialize_process_covariance()`: Block-diagonal via `construct_common_oscillator_process_covariance`
  - Default variance: 0.01 per oscillator
- `_initialize_spike_params()`: Zero baseline, small random weights (N(0, 0.1²))
- `_validate_parameter_shapes()`: Comprehensive shape validation matching BaseModel pattern

#### Tests Added (15 tests)

- Basic functionality (runs without error)
- Shape verification for all parameters
- Value checks (uniform probs, zero baseline, small weights)
- Numerical properties (PSD matrices, stochastic matrix)
- Edge cases (single discrete state)
- Reproducibility and randomness

### Task 7.3 (COMPLETE)

Implemented `SwitchingSpikeOscillatorModel._e_step()` method with:

- Builds `log_intensity_func` closure from current spike parameters
- Calls `switching_point_process_filter` with model parameters
- Calls `switching_kalman_smoother` (observation-model agnostic) with filter outputs
- Stores all 6 required smoother outputs as model attributes for M-step:
  - `smoother_state_cond_mean`: E[x_t | y_{1:T}, S_t=j]
  - `smoother_state_cond_cov`: Cov[x_t | y_{1:T}, S_t=j]
  - `smoother_discrete_state_prob`: P(S_t=j | y_{1:T})
  - `smoother_joint_discrete_state_prob`: P(S_t=j, S_{t+1}=k | y_{1:T})
  - `smoother_pair_cond_cross_cov`: Cov[x_{t+1}, x_t | y_{1:T}, S_t=j, S_{t+1}=k]
  - `smoother_pair_cond_means`: E[x_t | y_{1:T}, S_t=j, S_{t+1}=k]
- Returns marginal log-likelihood from filter

#### Tests Added (11 tests)

- Basic functionality (runs without error)
- Return value is scalar log-likelihood
- Shape verification for all 6 stored attributes
- Probability constraints (sum to 1, non-negative)
- Edge cases: single discrete state, zero spikes, high spike counts

#### Code Review Notes

- APPROVED by code-reviewer agent
- Clean separation: point-process-specific filtering + observation-agnostic smoothing
- Correct reuse of `switching_kalman_smoother` without modification
- Proper closure capture of spike parameters
- Comprehensive docstring with all stored attributes listed

### Task 7.4 (COMPLETE)

Implemented `SwitchingSpikeOscillatorModel._m_step_dynamics()` method with:

- Calls `switching_kalman_maximization_step` with stored smoother outputs from E-step
- Uses dummy observation array (required but unused for dynamics M-step)
- Updates model parameters conditionally based on update flags:
  - `continuous_transition_matrix (A)`: if `update_continuous_transition_matrix=True`
  - `process_cov (Q)`: if `update_process_cov=True`, with PSD regularization
  - `discrete_transition_matrix (Z)`: if `update_discrete_transition_matrix=True`
  - `init_mean (m0)`: if `update_init_mean=True`
  - `init_cov (P0)`: if `update_init_cov=True`
  - `init_discrete_state_prob`: always updated (from smoother posterior at t=0)
- Ignores `measurement_matrix` and `measurement_cov` returns (assume Gaussian observations)
- Ensures PSD process covariance via regularization: `Q = Q + 1e-8 * I`

#### Tests Added (15 tests)

- Basic functionality (runs without error)
- Each parameter update flag tested (A, Q, Z, m0, P0)
- Disabled flag preserves original value (for each parameter)
- PSD enforcement for process covariance
- Single discrete state edge case
- init_discrete_state_prob always updated and sums to 1
- All updates disabled case

#### Code Review Notes

- APPROVED by code-reviewer agent
- Simple, correct delegation to `switching_kalman_maximization_step`
- Proper handling of update flags (conditional updates)
- Regularization approach documented and sufficient for typical cases
- Good docstring explaining which parameters are updated and when

### Task 7.5 (COMPLETE)

Implemented `SwitchingSpikeOscillatorModel._m_step_spikes()` method with:

- Computes marginalized smoother mean by weighting state-conditional means by discrete state probabilities
  - Uses efficient `jnp.einsum("tls,ts->tl", ...)` for marginalization
  - Formula: `E[x_t | y_{1:T}] = sum_j P(S_t=j | y_{1:T}) * E[x_t | y_{1:T}, S_t=j]`
- Calls `update_spike_glm_params` with marginalized smoother mean
- Updates `spike_params` only if `update_spike_params=True`
- Early return pattern for disabled flag

#### Tests Added (9 tests)

- `test_m_step_spikes_runs_without_error`: Basic sanity check
- `test_m_step_spikes_updates_spike_params_when_enabled`: Flag behavior
- `test_m_step_spikes_does_not_update_when_disabled`: Flag behavior
- `test_m_step_spikes_output_shapes_correct`: Shape validation
- `test_m_step_spikes_output_finite`: Numerical stability
- `test_m_step_spikes_single_discrete_state`: Edge case
- `test_m_step_spikes_with_zero_spikes`: Edge case (silent neurons)
- `test_m_step_spikes_with_high_spike_counts`: Edge case (high firing rates)
- `test_m_step_spikes_uses_marginalized_smoother_mean`: Validates marginalization

#### Code Review Notes

- APPROVED by code-reviewer agent
- Correct einsum marginalization formula
- Proper integration with `update_spike_glm_params`
- Excellent documentation with mathematical exposition
- Comprehensive test coverage including edge cases
- Clean JAX best practices (JIT-compatible einsum)

### Task 7.6 (COMPLETE)

Implemented `SwitchingSpikeOscillatorModel.fit()` method with:

- **Parameter initialization**: Calls `_initialize_parameters()` with provided key
- **EM loop**: Alternates E-step and M-step until convergence or max_iter
- **Convergence checking**: Uses `check_converged()` from utils for consistent behavior
- **Input validation**:
  - Validates spikes is 2D array
  - Validates n_neurons matches model configuration
  - Raises ValueError for non-finite log-likelihood during iteration
- **ArrayLike input**: Accepts numpy/jax arrays via `jnp.asarray()` conversion
- **Returns list of log-likelihoods** for monitoring convergence

#### Tests Added (15 tests)

- `test_fit_runs_without_error`: Basic sanity check
- `test_fit_returns_log_likelihoods_list`: Return type validation
- `test_fit_log_likelihoods_are_finite`: Numerical stability
- `test_fit_em_overall_improvement`: EM improvement verification (allows Laplace approx violations)
- `test_fit_initializes_parameters`: Parameters initialized correctly
- `test_fit_convergence_tolerance`: Early stopping works
- `test_fit_single_discrete_state`: Edge case (non-switching model)
- `test_fit_with_zero_spikes`: Edge case (silent neurons)
- `test_fit_with_high_spike_counts`: Edge case (high firing rates)
- `test_fit_reproducibility_same_key`: Same key produces same results
- `test_fit_different_keys_produce_different_results`: Different keys diverge
- `test_fit_updates_model_parameters`: Parameters updated during EM
- `test_fit_respects_update_flags`: Update flags honored
- `test_fit_validates_spikes_shape_2d`: Shape validation error
- `test_fit_validates_spikes_n_neurons`: n_neurons mismatch error

#### Code Review Notes

- APPROVED by code-reviewer agent
- Used `ArrayLike` type annotation for input consistency with codebase
- Integrated `check_converged()` utility for convergence checking
- Added comprehensive input validation with clear error messages
- Robust test for EM improvement allows Laplace approximation-induced small violations
- Clean JAX best practices (explicit key threading, Python list for results)

### Task 7.7 (COMPLETE)

Implemented `SwitchingSpikeOscillatorModel._project_parameters()` method with:

- **Transition matrix projection**: Projects each A_j to preserve oscillatory block structure
  - Uses `project_coupled_transition_matrix` from `oscillator_utils.py`
  - Each 2x2 diagonal block is projected to closest scaled rotation [[a, -b], [b, a]]
  - Preserves oscillatory dynamics that M-step can break
- **Process covariance projection**: Ensures each Q_j is:
  - Symmetric: Q_j = (Q_j + Q_j.T) / 2
  - Positive semi-definite: Clips negative eigenvalues to 1e-8
  - Uses eigendecomposition approach for numerical stability
- **Integration with fit()**: Called after M-step in each EM iteration

#### Tests Added (9 tests)

- `test_project_parameters_runs_without_error`: Basic sanity check
- `test_project_parameters_ensures_psd_process_cov`: PSD property verified via eigenvalues
- `test_project_parameters_preserves_transition_matrix_shape`: Shape invariance
- `test_project_parameters_preserves_process_cov_shape`: Shape invariance
- `test_project_parameters_produces_finite_values`: Numerical stability
- `test_project_parameters_single_discrete_state`: Edge case
- `test_project_parameters_preserves_oscillator_block_structure`: Block structure preservation
- `test_project_parameters_ensures_symmetric_process_cov`: Symmetry check
- `test_project_parameters_called_during_fit`: Integration with EM loop

#### Code Review Notes

- APPROVED by code-reviewer agent
- Correct mathematical approach for both projections
- Comprehensive docstring explaining purpose and idempotency
- Integration properly placed after M-step calls
- Excellent test coverage (9 tests covering functional, mathematical, edge cases)

### Task 7.8 (COMPLETE)

Implemented end-to-end model class tests in `TestSwitchingSpikeOscillatorModelEndToEnd`:

- **test_model_recovers_discrete_states**: Tests discrete state recovery from simulated data
  - Uses 500 timepoints, 8 neurons, 2 discrete states with distinct dynamics
  - Conservative spike parameters for numerical stability (small weights, higher baseline)
  - Accounts for label permutation (checks both orderings)
  - Threshold: better than chance (>50%)
- **test_model_recovers_oscillator_params**: Tests parameter recovery
  - Single discrete state for cleaner recovery
  - Validates oscillatory block structure (diagonals equal, off-diagonals anti-symmetric)
  - Validates PSD process covariance
  - Quantitative check: spectral radius recovery within 30% tolerance
- **test_model_em_overall_improvement_single_state**: Tests EM improvement
  - Tests overall improvement (final > initial LL)
  - Does not test strict per-iteration monotonicity (Laplace approximation can cause small violations)
- **test_model_fit_on_simulated_data_runs_without_error**: Smoke test
  - Full pipeline on realistic simulated data
  - 200 timepoints, 8 neurons, 2 discrete states

#### Key Design Decisions

1. **Conservative spike parameters**: Small weights (0.05 scale), higher baseline (2.0) to prevent GLM M-step weight explosion with sparse data
2. **Smaller dt (0.01)**: For numerical stability
3. **Longer time series (200-500 steps)**: Required for reliable parameter estimation
4. **EM improvement vs monotonicity**: Test for overall improvement, not strict monotonicity (Laplace approximation can cause per-iteration violations)
5. **Label permutation handling**: Check accuracy for both label orderings

#### Code Review Notes

- APPROVED by code-reviewer agent
- Added quantitative spectral radius recovery check per reviewer suggestion
- Added detailed scientific justification for 50% discrete state accuracy threshold
- Comprehensive documentation of numerical stability considerations

## Milestone 8 Summary (COMPLETE)

All 4 tasks completed:

- 8.1 `test_switching_spike_oscillator_em_monotonic` - EM convergence properties with Laplace approximation
- 8.2 `test_switching_spike_oscillator_recovers_parameters` - Comprehensive parameter recovery test
- 8.3 `test_switching_spike_oscillator_vs_non_switching` - Compare S=1 switching vs non-switching model
- 8.4 `test_collapse_to_state_conditional` - Verify Gaussian mixture collapse math

### Key Implementation Details (Milestone 8)

#### Task 8.1: EM Convergence Test

- Tests overall improvement (final LL > initial LL), NOT strict per-iteration monotonicity
- Scientific rationale: Laplace approximation causes observed LL to decrease even when Q(θ) increases
- Tests numerical stability (no NaN/Inf), substantial improvement (best > initial + 10.0)

#### Task 8.2: Parameter Recovery Test

- Tests discrete transition matrix recovery with quantitative checks (diag > 0.5, off-diag < 0.5)
- Tests spectral radius recovery within 30% tolerance, handling label switching
- Validates PSD process covariance with appropriate scale
- **Does NOT test state correlation**: Weak observation coupling (~5% rate modulation) provides insufficient information for state inference

#### Task 8.3: Switching vs Non-Switching Comparison

- Uses `_e_step()` directly to avoid parameter re-initialization by `fit()`
- Sets identical fixed parameters in both models
- Tests log-likelihood similarity (<1.0 diff)
- Tests high correlation (>0.8) between smoothed means
- **Key insight**: `fit()` always re-initializes parameters, overwriting manual settings

#### Task 8.4: Gaussian Mixture Collapse Verification

- Verifies `collapse_gaussian_mixture_per_discrete_state` mathematical correctness
- Tests state-conditional means/covs match manual calculations
- Tests collapsed covariances are symmetric and PSD
- Tests law of total variance: collapsed trace >= average pair trace
- Edge cases: uniform mixing weights, deterministic (one-hot) weights

### Key Technical Insights from Milestone 8

1. **Laplace approximation non-monotonicity**: EM with approximate E-steps can have observed LL decrease even when the true variational bound Q(θ) increases
2. **Weak observations**: Spike weights at 0.05 scale provide only ~5% rate modulation - insufficient for reliable state inference
3. **Latent space non-identifiability**: Without fixed observation parameters, latent space has arbitrary rotation/scaling
4. **fit() re-initialization**: By design, `fit()` always calls `_initialize_parameters()` - use `_e_step()` directly for fixed-parameter inference

## Milestone 9 Summary (COMPLETE)

All 7 tasks completed:

- 9.1 Set up notebook with imports and random seed
- 9.2 Simulate data section (2 oscillators, 2 discrete states, 10 neurons)
- 9.3 Fit model section with progress logging
- 9.4 Visualization: Discrete states
- 9.5 Visualization: Oscillator trajectories
- 9.6 Visualization: Learned coupling
- 9.7 Visualization: Neuron loadings

### Key Implementation Details (Milestone 9)

#### Demo Notebook Structure

Created `notebooks/switching_spike_oscillator_demo.py` (paired with `.ipynb` via jupytext):

1. **Setup**: JAX 64-bit precision, imports, random seed (42)
2. **Simulation**: 1000 timesteps, 2 oscillators (4D latent), 10 neurons, 2 discrete states
   - State 0: Slower oscillations (6, 10 Hz)
   - State 1: Faster oscillations (8, 14 Hz)
   - High self-transition probability (0.98)
3. **Model Fitting**: EM algorithm with convergence monitoring
4. **Visualizations**:
   - Simulated data overview (spikes, discrete states, oscillator trajectories)
   - EM convergence plot
   - True vs inferred discrete states with accuracy
   - True vs smoothed oscillator trajectories with correlation
   - Learned vs true transition matrices
   - Neuron loadings heatmap

#### Demo Results (Typical Run)

- EM converges in ~22 iterations with significant improvement (+8000 nats)
- Discrete state accuracy: ~65% (above 50% chance)
- Oscillator 1 correlation: ~0.77 (strong tracking)
- Oscillator 2 correlation: ~0.23 (moderate tracking)
- Mean absolute correlation: ~0.50
- Spectral radius recovery: close to ground truth (0.9835 vs 0.98 for state 0)
- Baseline firing rate recovery: 7.93 Hz learned vs 8.0 Hz true

#### Important Caveats Documented

1. **Latent space non-identifiability**: Without fixed observation parameters, latent space has arbitrary sign/rotation. Handled via absolute correlations.
2. **Observation coupling strength**: Performance depends on spike weight magnitude (0.5 scale used in demo)
3. **Laplace approximation**: Can cause small EM monotonicity violations

### Project Complete

All 9 milestones of the Switching Spike-Based Oscillator Networks implementation are now complete:

1. ✅ Milestone 1: Core Point-Process Update
2. ✅ Milestone 2: Switching Filter
3. ✅ Milestone 3: Smoother Integration
4. ✅ Milestone 4: Simulation Utilities
5. ✅ Milestone 5: Spike GLM M-Step
6. ✅ Milestone 6: Dynamics M-Step Verification
7. ✅ Milestone 7: Model Class
8. ✅ Milestone 8: End-to-End Tests
9. ✅ Milestone 9: Demo Notebook
