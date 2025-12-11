# SCRATCHPAD: Switching Spike-Based Oscillator Networks

## Current Status

- **Date**: 2025-12-11
- **Working on**: Milestone 2 COMPLETE - Starting Milestone 3 (Smoother Integration)
- **Current Task**: 3.1 - Create `switching_point_process_smoother()` wrapper/alias

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

## Next Steps

- Implement `switching_point_process_smoother()` - thin wrapper around `switching_kalman_smoother`
- Verify smoother runs correctly with filter outputs
- Write smoother integration tests
