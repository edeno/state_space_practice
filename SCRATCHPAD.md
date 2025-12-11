# SCRATCHPAD: Switching Spike-Based Oscillator Networks

## Current Status

- **Date**: 2025-12-11
- **Working on**: Milestone 7 - Tasks 7.1, 7.2, and 7.3 COMPLETE
- **Current Task**: Task 7.4 - Implement `_m_step_dynamics()` method

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

### Next: Task 7.4

Implement `_m_step_dynamics()` method
