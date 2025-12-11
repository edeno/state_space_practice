# TASKS: Switching Spike-Based Oscillator Networks

Implementation tasks for the spike-based switching oscillator SLDS model.
See [docs/switching_spike_oscillator_plan.md](docs/switching_spike_oscillator_plan.md) for full design details.

---

## Milestone 1: Core Point-Process Update (Foundation)

**Goal**: Create the fundamental building block - a single-step point-process Laplace-EKF update that can be vmapped over state pairs.

**Success criteria**: Unit tests pass; single-neuron case matches existing `stochastic_point_process_filter` behavior.

**File**: `src/state_space_practice/switching_point_process.py`

### Tasks

- [x] **1.1** Create new file `src/state_space_practice/switching_point_process.py` with imports
  - Import from `switching_kalman.py`: `collapse_gaussian_mixture_per_discrete_state`, `_update_discrete_state_probabilities`, `_scale_likelihood`, `switching_kalman_smoother`, `switching_kalman_maximization_step`
  - Import from `kalman.py`: `symmetrize`, `psd_solve`
  - Import JAX, jax.numpy, typing

- [x] **1.2** Define `SpikeObsParams` dataclass
  - Fields: `baseline: Array` (n_neurons,), `weights: Array` (n_neurons, n_latent)
  - Add docstring with shape annotations

- [x] **1.3** Implement `point_process_kalman_update()` function
  - Signature: `(one_step_mean, one_step_cov, y_t, dt, log_intensity_func) -> (posterior_mean, posterior_cov, log_likelihood)`
  - Port gradient/Hessian computation from `stochastic_point_process_filter` lines 101-157
  - Handle vector spikes (n_neurons,) not scalar
  - Compute log-likelihood as sum of Poisson logpmfs at posterior mode
  - Use `symmetrize()` on posterior covariance

- [x] **1.4** Implement `_point_process_predict_and_update()` helper
  - Combines one-step prediction (A @ m, A @ P @ A.T + Q) with Laplace update
  - Single (i, j) state pair version before vmapping

- [x] **1.5** Create `_point_process_update_per_discrete_state_pair` via double vmap
  - Inner vmap over previous state i (axis -1 of mean/cov)
  - Outer vmap over next state j (axis -1 of A, Q)
  - Verify output shapes: mean (n_latent, n_states, n_states), cov (n_latent, n_latent, n_states, n_states), ll (n_states, n_states)

- [x] **1.6** Write unit tests for `point_process_kalman_update`
  - Test file: `src/state_space_practice/tests/test_switching_point_process.py`
  - `test_point_process_update_single_neuron`: Compare to existing filter single step
  - `test_point_process_update_multiple_neurons`: Verify shapes with N neurons
  - `test_point_process_update_zero_spikes`: Posterior should equal prior prediction
  - `test_point_process_update_positive_definite`: Posterior cov is PSD

---

## Milestone 2: Switching Filter

**Goal**: Full switching point-process filter with exact per-state-pair structure.

**Success criteria**: With S=1 discrete state, matches non-switching `stochastic_point_process_filter`. Shapes correct for all outputs.

**Depends on**: Milestone 1

**File**: `src/state_space_practice/switching_point_process.py`

### Tasks

- [x] **2.1** Implement `switching_point_process_filter()` function
  - Mirror structure of `switching_kalman_filter` (switching_kalman.py:229-414)
  - Use `_point_process_update_per_discrete_state_pair` instead of Gaussian version
  - Reuse `_scale_likelihood`, `_update_discrete_state_probabilities`, `collapse_gaussian_mixture_per_discrete_state`

- [x] **2.2** Implement the `_step` function inside filter
  - Compute pair-conditional posteriors for all (i,j) pairs
  - Scale likelihoods for numerical stability
  - Update discrete state probabilities (HMM forward step)
  - Accumulate marginal log-likelihood
  - Collapse to state-conditional via mixture

- [x] **2.3** Handle initial timestep correctly
  - No previous state to condition on; use `init_discrete_state_prob` directly
  - May need special first-step logic or padded initialization

- [x] **2.4** Return `last_pair_cond_filter_mean` for smoother
  - Shape: (n_latent, n_discrete_states, n_discrete_states)
  - Needed by `switching_kalman_smoother`

- [x] **2.5** Write filter integration tests
  - `test_switching_point_process_filter_single_state`: S=1 matches non-switching filter
  - `test_switching_point_process_filter_output_shapes`: Verify all output shapes
  - `test_switching_point_process_filter_discrete_probs_sum_to_one`: Probabilities valid
  - `test_pair_conditional_shapes`: Pair-conditional outputs have correct shapes

---

## Milestone 3: Smoother Integration

**Goal**: Verify smoother works with point-process filter outputs.

**Success criteria**: Smoother runs without error; smoothed means are "smoother" than filtered.

**Depends on**: Milestone 2

**File**: `src/state_space_practice/switching_point_process.py`

### Tasks

- [x] **3.1** ~~Create `switching_point_process_smoother()` wrapper/alias~~
  - **DECISION**: No wrapper needed - smoother is observation-model agnostic
  - Just use `switching_kalman_smoother` directly with filter outputs
  - Verified input shapes match what smoother expects

- [x] **3.2** Write smoother integration tests
  - `test_smoother_runs_on_filter_output`: No errors on filter output
  - `test_smoother_output_no_nans`: Smoother doesn't produce NaN values
  - `test_smoother_shapes_compatible`: Verify all output shapes compatible
  - `test_smoother_discrete_probs_sum_to_one`: Probability constraints

---

## Milestone 4: Simulation Utilities

**Goal**: Generate synthetic data for testing parameter recovery.

**Success criteria**: Can simulate spikes from known parameters; ground truth recoverable by inspection.

**Depends on**: None (can be done in parallel with Milestones 1-3)

**File**: `src/state_space_practice/simulate/simulate_switching_spikes.py`

### Tasks

- [x] **4.1** Create `src/state_space_practice/simulate/` directory if needed
  - Directory already exists with `simulate_switching_kalman.py`

- [x] **4.2** Implement `simulate_switching_spike_oscillator()` function
  - Inputs: n_time, transition_matrices, process_covs, discrete_transition_matrix, spike_weights, spike_baseline, dt, key
  - Simulate discrete state sequence via categorical sampling
  - Simulate continuous state via Gaussian dynamics conditioned on discrete state
  - Simulate spikes via Poisson with rate = exp(baseline + weights @ state) * dt

- [x] **4.3** Return ground truth for validation
  - Return tuple: (spikes, true_states, true_discrete_states)
  - All as JAX arrays with documented shapes

- [x] **4.4** Write simulation tests
  - `test_simulate_output_shapes`: Verify shapes
  - `test_simulate_spike_rates_positive`: Rates are positive
  - `test_simulate_discrete_states_valid`: States in valid range
  - Plus 7 additional tests for reproducibility, edge cases, dynamics verification

---

## Milestone 5: Spike GLM M-Step

**Goal**: Update spike observation parameters (C, b) given smoothed latent states.

**Success criteria**: GLM converges; loss decreases; parameters have correct shapes.

**Depends on**: Milestone 3

**File**: `src/state_space_practice/switching_point_process.py`

### Tasks

- [x] **5.1** Implement `_single_neuron_glm_loss()` helper
  - Poisson negative log-likelihood: -sum(y *eta - exp(eta)* dt) where eta = b + c @ m
  - Works with smoother_mean as design matrix

- [x] **5.2** Implement `_single_neuron_glm_step()` Newton step
  - Compute gradient and Hessian of loss
  - Newton update with backtracking line search for guaranteed descent
  - Return updated params

- [x] **5.3** Implement `update_spike_glm_params()` function
  - Plug-in method: use smoother_mean directly as design matrix
  - vmap single_neuron_glm over neurons
  - Run max_iter Newton iterations
  - Return updated SpikeObsParams

- [x] **5.4** Implement second-order expectation variant
  - Account for E[exp(c @ x)] = exp(c @ m + 0.5 * c @ P @ c)
  - More accurate but requires iterative solve
  - Added `smoother_cov` and `use_second_order` parameters to `update_spike_glm_params()`

- [x] **5.5** Write GLM M-step tests
  - `test_glm_mstep_decreases_loss`: Loss decreases after update
  - `test_glm_mstep_output_shapes`: Params have correct shapes
  - `test_glm_mstep_recovers_true_params`: Recovery on simulated data
  - Additional tests for second-order variant

---

## Milestone 6: Dynamics M-Step Verification

**Goal**: Confirm `switching_kalman_maximization_step` works for point-process model.

**Success criteria**: Can call dynamics M-step on smoother outputs; returned A, Q, B have correct shapes.

**Depends on**: Milestone 3

**File**: `src/state_space_practice/switching_point_process.py`

### Tasks

- [x] **6.1** Write test confirming dynamics M-step reuse
  - Call `switching_kalman_maximization_step` with smoother outputs from point-process filter
  - Verify returned A, Q, B, init params have correct shapes
  - Document that measurement_matrix, measurement_cov returns should be ignored

- [x] **6.2** Add docstring/comment in code documenting this reuse pattern

---

## Milestone 7: Model Class

**Goal**: Unified `SwitchingSpikeOscillatorModel` class with fit() method.

**Success criteria**: EM converges; log-likelihood monotonically increases (up to Laplace approximation); API matches existing oscillator models.

**Depends on**: Milestones 3, 5, 6

**File**: `src/state_space_practice/switching_point_process.py`

### Tasks

- [x] **7.1** Implement `SwitchingSpikeOscillatorModel.__init__()`
  - Parameters: n_oscillators, n_neurons, n_discrete_states, sampling_freq, dt
  - Optional: initial freqs, damping, coupling strengths, phase differences
  - Store update flags (update_A, update_Q, update_B, update_spike_params, etc.)

- [x] **7.2** Implement `_initialize_parameters()` method
  - Initialize A via `construct_common_oscillator_transition_matrix` (uncoupled oscillators)
  - Initialize Q via `construct_common_oscillator_process_covariance`
  - Initialize spike params (baseline=0, weights=small random)
  - Initialize discrete transition matrix
  - Initialize continuous initial state
  - Added shape validation via `_validate_parameter_shapes()`

- [x] **7.3** Implement `_e_step()` method
  - Call `switching_point_process_filter`
  - Call `switching_kalman_smoother` (observation-model agnostic)
  - Store smoother outputs as attributes for M-step
  - Return marginal log-likelihood

- [x] **7.4** Implement `_m_step_dynamics()` method
  - Call `switching_kalman_maximization_step`
  - Update A, Q, discrete transitions, initial state based on update flags
  - Ensure Q is PSD via regularization (eps=1e-8)
  - Ignore measurement_matrix/measurement_cov returns (Gaussian-only)

- [x] **7.5** Implement `_m_step_spikes()` method
  - Call `update_spike_glm_params`
  - Update spike_params

- [x] **7.6** Implement `fit()` method
  - Initialize parameters
  - EM loop: E-step, check convergence, M-step
  - Input validation (shape, dtype)
  - Return list of log-likelihoods

- [x] **7.7** Add `_project_parameters()` method
  - Apply `project_coupled_transition_matrix` if using DIM structure
  - Ensure Q is PSD

- [x] **7.8** Write model class tests
  - `test_model_fit_runs`: No errors on simulated data
  - `test_model_em_monotonic`: Log-likelihood increases each iteration
  - `test_model_recovers_discrete_states`: Inferred states match true states
  - `test_model_recovers_oscillator_params`: Reasonable parameter recovery

---

## Milestone 8: End-to-End Tests

**Goal**: Comprehensive validation of full pipeline.

**Success criteria**: All tests pass; parameter recovery on simulated data.

**Depends on**: Milestones 4, 7

**File**: `src/state_space_practice/tests/test_switching_point_process.py`

### Tasks

- [x] **8.1** `test_switching_spike_oscillator_em_monotonic`
  - Fit model on simulated data
  - Assert log-likelihood[t+1] >= log-likelihood[t] - epsilon

- [x] **8.2** `test_switching_spike_oscillator_recovers_parameters`
  - Simulate with known parameters
  - Fit model
  - Assert recovered params close to true params (within tolerance)

- [x] **8.3** `test_switching_spike_oscillator_vs_non_switching`
  - S=1 discrete state
  - Compare to `PointProcessModel` from point_process_kalman.py
  - Smoothed means should be similar

- [x] **8.4** `test_collapse_to_state_conditional`
  - Verify collapse math is correct

---

## Milestone 9: Demo Notebook

**Goal**: Working example demonstrating the model on realistic-ish data. Use jupyter-notebook skill.

**Success criteria**: Notebook runs end-to-end; visualizations are clear.

**Depends on**: Milestone 7

**File**: `notebooks/switching_spike_oscillator_demo.py`

### Tasks

- [x] **9.1** Set up notebook with imports and random seed

- [x] **9.2** Simulate data section
  - 2 oscillators, 2 discrete states, 10 neurons
  - Different coupling patterns per state
  - ~1000 timesteps

- [x] **9.3** Fit model section
  - Initialize model
  - Call fit() with progress logging
  - Plot log-likelihood convergence

- [x] **9.4** Visualization: Discrete states
  - Plot true vs inferred P(S_t = j)
  - Highlight state transitions

- [x] **9.5** Visualization: Oscillator trajectories
  - Plot true states vs smoothed means
  - Show confidence intervals

- [x] **9.6** Visualization: Learned coupling
  - Show coupling matrices per discrete state
  - Compare to ground truth

- [x] **9.7** Visualization: Neuron loadings
  - Show which neurons load onto which oscillators
  - Heatmap of C matrix

---

## Quick Reference: File Locations

| Component | File |
|-----------|------|
| Core implementation | `src/state_space_practice/switching_point_process.py` |
| Simulation utilities | `src/state_space_practice/simulate/simulate_switching_spikes.py` |
| Tests | `src/state_space_practice/tests/test_switching_point_process.py` |
| Demo notebook | `notebooks/switching_spike_oscillator_demo.py` |
| Design document | `docs/switching_spike_oscillator_plan.md` |

---

## Dependencies Between Milestones

```
Milestone 1 (Core Update)
    │
    ├──► Milestone 2 (Filter)
    │        │
    │        └──► Milestone 3 (Smoother)
    │                 │
    │                 ├──► Milestone 5 (GLM M-step)
    │                 │
    │                 └──► Milestone 6 (Dynamics M-step)
    │                          │
    │                          └──► Milestone 7 (Model Class) ──► Milestone 8 (E2E Tests)
    │                                       │
    │                                       └──► Milestone 9 (Demo)
    │
    └── Milestone 4 (Simulation) ──────────────────────────────────┘
        (can be done in parallel)
```
