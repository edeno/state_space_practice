# Implementation Plan: Switching Spike-Based Oscillator Networks

This document outlines the implementation plan for a **Switching Linear Dynamical System (SLDS)** that combines:

- Oscillator network dynamics with switching discrete states (DIM/CNM style)
- Point-process observations (spikes) using the Eden/Brown Laplace-EKF approach

## 1. Overview

### 1.1 Model Structure

**Latent Dynamics** (per discrete state $s_t$):
$$
x_t = A_{s_t} x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q_{s_t})
$$

where:

- $x_t \in \mathbb{R}^{2K}$ is a stack of $K$ 2D oscillators
- $s_t \in \{1, \ldots, S\}$ indexes different coupling patterns
- $A_{s_t}$ is constructed via `construct_directed_influence_transition_matrix`
- $Q_{s_t}$ is the oscillator process noise covariance

**Spike Observation Model** (for $N$ neurons):
$$
\log \lambda_{n,t} = b_n + c_n^\top x_t
$$
$$
y_{n,t} \sim \text{Poisson}(\lambda_{n,t} \Delta t)
$$

where:

- $C \in \mathbb{R}^{N \times 2K}$ maps oscillator states to log-rates
- $b \in \mathbb{R}^N$ are baseline log-rates

### 1.2 Key Design Decisions

1. **Reuse existing infrastructure**: Maximize reuse of `switching_kalman.py` and `point_process_kalman.py`
2. **Laplace approximation for spikes**: Replace Gaussian Kalman update with point-process Laplace update
3. **Exact per-state-pair structure**: Track $p(x_t | y_{1:t}, S_{t-1}=i, S_t=j)$ for all state pairs, mirroring the Gaussian switching filter exactly. This ensures correct expected sufficient statistics and EM monotonicity (up to Laplace approximation).
4. **Smoother reuse**: RTS smoother logic is observation-model agnostic once filter gives Gaussian posteriors
5. **M-step modularity**: Dynamics M-step reuses `switching_kalman_maximization_step`; add separate GLM M-step for spike params

### 1.3 Approximations

The only approximation in this implementation is the **Laplace approximation** for the point-process observation update. We do NOT make the "per-current-state only" approximation that would drop the pair-conditional structure. This means:

- EM monotonicity is guaranteed up to numerical precision of the Laplace approximation
- Expected sufficient statistics for the M-step are computed correctly
- The smoother receives exactly the quantities it expects (`last_pair_cond_filter_mean`)

### 1.4 Performance Considerations

The exact per-state-pair structure has computational cost $O(S^2)$ per timestep (where $S$ = number of discrete states), since we compute:

$$
p(x_t | y_{1:t}, S_{t-1}=i, S_t=j) \quad \forall (i, j) \in \{1, \ldots, S\}^2
$$

Each update involves a Laplace approximation (Newton-Raphson iteration). Mitigations:

1. **Single Newton step**: The existing `stochastic_point_process_filter` uses a single Newton step per timestep. This is often sufficient and avoids nested iteration.

2. **Warm starting**: Use the collapsed state-conditional mean from the previous timestep as initialization for Newton updates.

3. **JAX compilation**: The entire filter (including all $S^2$ updates) compiles to a single fused kernel via `jax.lax.scan` + `jax.vmap`.

4. **Moderate $S$**: For typical neuroscience applications, $S \in \{2, 3, 4\}$ discrete states, so $S^2 \leq 16$ is manageable.

5. **Batched neurons**: The multi-neuron gradient/Hessian computation vectorizes over neurons, not requiring per-neuron iteration.

---

## 2. Implementation Tasks

### Phase 1: Core Point-Process Update Function

**File**: `src/state_space_practice/switching_point_process.py` (new file)

#### Task 1.1: Extract Single-Step Point-Process Update

Factor out the single-timestep update logic from `stochastic_point_process_filter` into a reusable function.

```python
def point_process_kalman_update(
    one_step_mean: Array,          # shape (n_latent,)
    one_step_cov: Array,           # shape (n_latent, n_latent)
    y_t: Array,                    # shape (n_neurons,) - spike counts
    dt: float,
    log_intensity_func: Callable,  # (state) -> (n_neurons,) log-rates
) -> tuple[Array, Array, float]:
    """Single point-process Laplace-EKF update.

    Parameters
    ----------
    one_step_mean : Array, shape (n_latent,)
        Predicted mean from dynamics: A @ m_{t-1}
    one_step_cov : Array, shape (n_latent, n_latent)
        Predicted covariance: A @ P_{t-1} @ A.T + Q
    y_t : Array, shape (n_neurons,)
        Spike counts at time t for all neurons
    dt : float
        Time bin width
    log_intensity_func : Callable
        Function mapping state to log-intensities for all neurons

    Returns
    -------
    posterior_mean : Array, shape (n_latent,)
    posterior_cov : Array, shape (n_latent, n_latent)
    log_likelihood : float
        log p(y_t | y_{1:t-1}) approximated from Laplace update
    """
```

**Implementation notes**:

- Port gradient/Hessian computation from existing `stochastic_point_process_filter`
- Handle vector spikes (multiple neurons) rather than scalar
- Return approximate **predictive log-likelihood** $\log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)$ for discrete state probability updates

**Log-likelihood computation** (critical for discrete state updates):

The Laplace approximation gives us a Gaussian approximation to the posterior:
$$
p(x_t | y_t, x_{t|t-1}^{ij}) \approx \mathcal{N}(\hat{x}_t^{ij}, \hat{P}_t^{ij})
$$

The predictive log-likelihood needed for HMM updates is:
$$
\log p(y_t | y_{1:t-1}, S_{t-1}=i, S_t=j)
$$

For the Laplace approximation, this can be computed as:

```python
# Laplace approximation to log p(y_t | prior)
# = log p(y_t | x_mode) + log p(x_mode | prior) - log q(x_mode)
# where q is the Laplace Gaussian approximation

# In practice, sum Poisson log-likelihoods at the posterior mode:
log_likelihood = jnp.sum(
    jax.scipy.stats.poisson.logpmf(y_t, conditional_intensity * dt)
)
# Plus a correction term from the Laplace approximation (often omitted for simplicity)
```

This matches what `stochastic_point_process_filter` already computes (line 150-152).

#### Task 1.2: Create Vmapped Update for State Pairs

Mirror the exact structure of `_kalman_filter_update_per_discrete_state_pair` from `switching_kalman.py`:

```python
# First, create a prediction + update function that takes:
# - Previous state-conditional mean/cov (from state i)
# - Dynamics for next state j (A_j, Q_j)
# - Spikes y_t
# And returns pair-conditional posterior p(x_t | y_{1:t}, S_{t-1}=i, S_t=j)

def _point_process_predict_and_update(
    prev_state_cond_mean: Array,   # shape (n_latent,) - m_{t-1|t-1}^i
    prev_state_cond_cov: Array,    # shape (n_latent, n_latent) - P_{t-1|t-1}^i
    y_t: Array,                    # shape (n_neurons,)
    continuous_transition_matrix: Array,  # shape (n_latent, n_latent) - A_j
    process_cov: Array,            # shape (n_latent, n_latent) - Q_j
    dt: float,
    log_intensity_func: Callable,
) -> tuple[Array, Array, float]:
    """Predict with dynamics j, then update with spikes.

    Returns pair-conditional posterior p(x_t | y_{1:t}, S_{t-1}=i, S_t=j).
    """
    # One-step prediction using dynamics for state j
    one_step_mean = continuous_transition_matrix @ prev_state_cond_mean
    one_step_cov = (
        continuous_transition_matrix @ prev_state_cond_cov @ continuous_transition_matrix.T
        + process_cov
    )

    # Point-process Laplace update
    return point_process_kalman_update(
        one_step_mean, one_step_cov, y_t, dt, log_intensity_func
    )

# Double vmap over (i, j) state pairs - mirrors switching_kalman.py:22-30
_point_process_update_per_discrete_state_pair = jax.vmap(
    jax.vmap(
        _point_process_predict_and_update,
        in_axes=(-1, -1, None, None, None, None, None),  # vmap over prev state i
        out_axes=-1,
    ),
    in_axes=(None, None, None, -1, -1, None, None),  # vmap over next state j
    out_axes=-1,
)
# Output shapes: (n_latent, n_discrete_states, n_discrete_states) for mean
#                (n_latent, n_latent, n_discrete_states, n_discrete_states) for cov
#                (n_discrete_states, n_discrete_states) for log_likelihood
```

This exactly mirrors `_kalman_filter_update_per_discrete_state_pair` but replaces the Gaussian observation update with the point-process Laplace update.

---

### Phase 2: Switching Point-Process Filter

**File**: `src/state_space_practice/switching_point_process.py`

#### Task 2.1: Implement `switching_point_process_filter`

```python
def switching_point_process_filter(
    init_state_cond_mean: Array,         # (n_latent, n_discrete_states)
    init_state_cond_cov: Array,          # (n_latent, n_latent, n_discrete_states)
    init_discrete_state_prob: Array,     # (n_discrete_states,)
    spikes: Array,                       # (n_time, n_neurons)
    discrete_transition_matrix: Array,   # (n_discrete_states, n_discrete_states)
    continuous_transition_matrix: Array, # (n_latent, n_latent, n_discrete_states)
    process_cov: Array,                  # (n_latent, n_latent, n_discrete_states)
    spike_params: SpikeObsParams,        # baseline, weights
    dt: float,
    log_intensity_func: Callable | None = None,
) -> tuple[
    Array,  # state_cond_filter_mean: (n_time, n_latent, n_discrete_states)
    Array,  # state_cond_filter_cov: (n_time, n_latent, n_latent, n_discrete_states)
    Array,  # filter_discrete_state_prob: (n_time, n_discrete_states)
    Array,  # last_pair_cond_filter_mean: (n_latent, n_discrete_states, n_discrete_states)
    float,  # marginal_log_likelihood
]:
```

**Implementation approach**:

1. **Mirror structure exactly** from `switching_kalman_filter` (`switching_kalman.py:229-414`)
2. Replace `_kalman_filter_update_per_discrete_state_pair` with `_point_process_update_per_discrete_state_pair`
3. Reuse discrete state probability update logic unchanged:
   - `_scale_likelihood` (line 209-226)
   - `_update_discrete_state_probabilities` (line 158-206)
4. Reuse Gaussian mixture collapse: `collapse_gaussian_mixture_per_discrete_state`

**Filter step structure** (matches Gaussian case exactly):

```python
def _step(carry, y_t):
    prev_state_cond_mean, prev_state_cond_cov, prev_discrete_prob, marginal_ll = carry

    # 1. Compute pair-conditional posteriors p(x_t | y_{1:t}, S_{t-1}=i, S_t=j)
    #    for ALL (i, j) pairs
    pair_cond_mean, pair_cond_cov, pair_cond_log_likelihood = (
        _point_process_update_per_discrete_state_pair(
            prev_state_cond_mean,   # shape (n_latent, n_discrete_states) - one per prev state i
            prev_state_cond_cov,    # shape (n_latent, n_latent, n_discrete_states)
            y_t,                    # shape (n_neurons,)
            continuous_transition_matrix,  # shape (n_latent, n_latent, n_discrete_states)
            process_cov,            # shape (n_latent, n_latent, n_discrete_states)
            dt,
            log_intensity_func,
        )
    )
    # pair_cond_mean: (n_latent, n_discrete_states, n_discrete_states) - [i, j]
    # pair_cond_cov: (n_latent, n_latent, n_discrete_states, n_discrete_states)
    # pair_cond_log_likelihood: (n_discrete_states, n_discrete_states) - L(i, j)

    # 2. Scale likelihood for numerical stability
    pair_cond_likelihood_scaled, ll_max = _scale_likelihood(pair_cond_log_likelihood)

    # 3. Update discrete state probabilities (HMM step)
    filter_discrete_prob, filter_backward_cond_prob, pred_ll_sum = (
        _update_discrete_state_probabilities(
            pair_cond_likelihood_scaled,
            discrete_transition_matrix,
            prev_discrete_prob,
        )
    )

    # 4. Accumulate marginal log-likelihood
    marginal_ll += ll_max + jnp.log(pred_ll_sum)

    # 5. Collapse pair-conditional to state-conditional
    #    p(x_t | y_{1:t}, S_t=j) by marginalizing over S_{t-1}=i
    state_cond_mean, state_cond_cov = collapse_gaussian_mixture_per_discrete_state(
        pair_cond_mean,           # (n_latent, n_discrete_states, n_discrete_states)
        pair_cond_cov,            # (n_latent, n_latent, n_discrete_states, n_discrete_states)
        filter_backward_cond_prob,  # P(S_{t-1}=i | S_t=j, y_{1:t})
    )

    return (state_cond_mean, state_cond_cov, filter_discrete_prob, marginal_ll), (
        state_cond_mean, state_cond_cov, filter_discrete_prob, pair_cond_mean
    )
```

**Key point**: The filter maintains pair-conditional structure internally and outputs `pair_cond_mean` at the last timestep for the smoother.

---

### Phase 3: Switching Point-Process Smoother

**File**: `src/state_space_practice/switching_point_process.py`

#### Task 3.1: Implement `switching_point_process_smoother`

The smoother is **observation-model agnostic** once we have Gaussian posteriors from the filter. We can directly reuse most of `switching_kalman_smoother`.

```python
def switching_point_process_smoother(
    filter_mean: Array,
    filter_cov: Array,
    filter_discrete_state_prob: Array,
    last_filter_conditional_cont_mean: Array,
    process_cov: Array,
    continuous_transition_matrix: Array,
    discrete_state_transition_matrix: Array,
) -> tuple[...]:  # Same return signature as switching_kalman_smoother
    """Switching point-process smoother.

    This is a thin wrapper that calls switching_kalman_smoother,
    since the RTS smoother equations depend only on filtered posteriors
    and dynamics, not on the observation model.
    """
    return switching_kalman_smoother(
        filter_mean=filter_mean,
        filter_cov=filter_cov,
        filter_discrete_state_prob=filter_discrete_state_prob,
        last_filter_conditional_cont_mean=last_filter_conditional_cont_mean,
        process_cov=process_cov,
        continuous_transition_matrix=continuous_transition_matrix,
        discrete_state_transition_matrix=discrete_state_transition_matrix,
    )
```

**Note**: This may literally just be an alias/re-export, documenting that the smoother works for point-process observations.

---

### Phase 4: Spike GLM M-Step

**File**: `src/state_space_practice/switching_point_process.py`

#### Task 4.1: Define `SpikeObsParams` dataclass

```python
from dataclasses import dataclass

@dataclass
class SpikeObsParams:
    """Spike observation model parameters.

    Attributes
    ----------
    baseline : Array, shape (n_neurons,)
        Baseline log-rate b_n for each neuron
    weights : Array, shape (n_neurons, n_latent)
        Linear weights C mapping oscillator state to log-rates
    """
    baseline: Array
    weights: Array
```

#### Task 4.2: Implement `update_spike_glm_params`

```python
def update_spike_glm_params(
    spikes: Array,           # (n_time, n_neurons)
    smoother_mean: Array,    # (n_time, n_latent)
    smoother_cov: Array,     # (n_time, n_latent, n_latent)
    current_params: SpikeObsParams,
    dt: float,
    use_second_order: bool = False,
    max_iter: int = 10,
) -> SpikeObsParams:
    """M-step for spike observation parameters.

    Maximizes E_q[log p(y | x; C, b)] with respect to C and b.

    Two approaches:
    1. Plug-in (default): Replace x_t with E[x_t] = smoother_mean
       - Simple Poisson GLM per neuron
    2. Second-order: Account for state uncertainty
       - E[exp(c^T x)] = exp(c^T m + 0.5 * c^T P c)
       - Still convex, but requires iterative optimization

    Parameters
    ----------
    spikes : Array, shape (n_time, n_neurons)
    smoother_mean : Array, shape (n_time, n_latent)
    smoother_cov : Array, shape (n_time, n_latent, n_latent)
    current_params : SpikeObsParams
        Current parameter estimates (for warm-starting)
    dt : float
    use_second_order : bool
        If True, account for state uncertainty in expectation
    max_iter : int
        Max Newton iterations for GLM fitting

    Returns
    -------
    SpikeObsParams
        Updated baseline and weights
    """
```

**Implementation approach for plug-in method**:

```python
# For each neuron n:
#   Maximize sum_t [ y_{n,t} * (b_n + c_n @ m_t) - exp(b_n + c_n @ m_t) * dt ]
# This is standard Poisson GLM with design matrix = smoother_mean

# Use JAX autodiff + Newton-Raphson or optax optimizer
def single_neuron_loss(params, y_n, design, dt):
    b, c = params[0], params[1:]
    log_rate = b + design @ c
    return -jnp.sum(y_n * log_rate - jnp.exp(log_rate) * dt)

# vmap over neurons
```

---

### Phase 5: Dynamics M-Step Integration

**File**: `src/state_space_practice/switching_point_process.py`

#### Task 5.1: Verify dynamics M-step reuse

The dynamics M-step in `switching_kalman_maximization_step` is observation-model agnostic. It operates on:

- `state_cond_smoother_means`
- `state_cond_smoother_covs`
- `smoother_discrete_state_prob`
- `smoother_joint_discrete_state_prob`
- `pair_cond_smoother_cross_cov`
- `pair_cond_smoother_means`

These are all outputs of the smoother, not dependent on observation model.

**Action**: Document that `switching_kalman_maximization_step` can be called directly for dynamics parameters, ignoring returned `measurement_matrix` and `measurement_cov`.

---

### Phase 6: Model Class

**File**: `src/state_space_practice/switching_point_process.py`

#### Task 6.1: Implement `SwitchingSpikeOscillatorModel` class

```python
class SwitchingSpikeOscillatorModel:
    """Switching oscillator network with spike-based observations.

    Combines:
    - Oscillator network dynamics (DIM/CNM style transition matrices)
    - Switching discrete states for different coupling patterns
    - Point-process (spike) observations via Laplace-EKF

    Parameters
    ----------
    n_oscillators : int
        Number of latent oscillators (state dim = 2 * n_oscillators)
    n_neurons : int
        Number of observed neurons
    n_discrete_states : int
        Number of discrete network states
    sampling_freq : float
        Sampling frequency in Hz
    dt : float
        Time bin width in seconds

    Attributes (after fit)
    ----------------------
    smoother_mean : Array, shape (n_time, n_latent)
    smoother_cov : Array, shape (n_time, n_latent, n_latent)
    smoother_discrete_prob : Array, shape (n_time, n_discrete_states)
    spike_params : SpikeObsParams
    continuous_transition_matrix : Array
    process_cov : Array
    """

    def __init__(self, ...):
        ...

    def _initialize_parameters(self, key: PRNGKey) -> None:
        """Initialize all parameters."""
        ...

    def _e_step(self, spikes: Array) -> float:
        """E-step: filter + smoother."""
        ...

    def _m_step_dynamics(self) -> None:
        """M-step for A, Q, B, initial state."""
        ...

    def _m_step_spikes(self, spikes: Array) -> None:
        """M-step for C, b (spike GLM params)."""
        ...

    def fit(
        self,
        spikes: Array,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> list[float]:
        """Fit model via EM."""
        ...
```

#### Task 6.2: Support DIM/CNM oscillator structures

The model should accept:

- Initial oscillator frequencies, damping coefficients
- Initial coupling strengths and phase differences per discrete state
- Option to use `project_coupled_transition_matrix` after M-step
- Option to use reparameterized M-step (as in `DirectedInfluenceModel`)

---

### Phase 7: Tests

**File**: `src/state_space_practice/tests/test_switching_point_process.py` (new file)

#### Task 7.1: Unit tests for `point_process_kalman_update`

```python
def test_point_process_update_single_neuron():
    """Test update with single neuron matches non-switching filter."""
    ...

def test_point_process_update_multiple_neurons():
    """Test update with multiple neurons."""
    ...

def test_point_process_update_zero_spikes():
    """Test update when no spikes observed."""
    ...
```

#### Task 7.2: Integration tests for filter

```python
def test_switching_point_process_filter_single_state():
    """With 1 discrete state, should match non-switching filter."""
    ...

def test_switching_point_process_filter_state_transitions():
    """Test that discrete state probabilities update correctly."""
    ...
```

#### Task 7.3: Integration tests for smoother

```python
def test_switching_point_process_smoother_consistency():
    """Smoother means should be smoother than filtered means."""
    ...
```

#### Task 7.4: End-to-end EM tests

```python
def test_switching_spike_oscillator_em_monotonic():
    """Test that EM log-likelihood increases monotonically.

    With exact per-state-pair structure, EM should be monotonic
    up to numerical precision of the Laplace approximation.
    """
    ...

def test_switching_spike_oscillator_recovers_parameters():
    """Test parameter recovery on simulated data."""
    ...

def test_switching_spike_oscillator_vs_non_switching():
    """With 1 discrete state, should match non-switching point-process filter."""
    ...
```

#### Task 7.5: Pair-conditional structure tests

```python
def test_pair_conditional_shapes():
    """Verify pair-conditional outputs have correct shapes."""
    ...

def test_collapse_to_state_conditional():
    """Verify collapse from pair-conditional to state-conditional is correct."""
    ...

def test_last_pair_cond_mean_for_smoother():
    """Verify last_pair_cond_filter_mean has correct shape for smoother."""
    ...
```

---

### Phase 8: Simulation Utilities

**File**: `src/state_space_practice/simulate/simulate_switching_spikes.py` (new file)

#### Task 8.1: Implement spike data simulation

```python
def simulate_switching_spike_oscillator(
    n_time: int,
    n_neurons: int,
    n_oscillators: int,
    n_discrete_states: int,
    transition_matrices: Array,      # (n_latent, n_latent, n_discrete_states)
    process_covs: Array,             # (n_latent, n_latent, n_discrete_states)
    discrete_transition_matrix: Array,
    spike_weights: Array,            # (n_neurons, n_latent)
    spike_baseline: Array,           # (n_neurons,)
    dt: float,
    key: PRNGKey,
) -> tuple[Array, Array, Array]:
    """Simulate spikes from switching oscillator network.

    Returns
    -------
    spikes : Array, shape (n_time, n_neurons)
    true_states : Array, shape (n_time, n_latent)
    true_discrete_states : Array, shape (n_time,)
    """
```

---

### Phase 9: Notebook/Example

**File**: `notebooks/switching_spike_oscillator_demo.py`

#### Task 9.1: Create demonstration notebook

1. Simulate data from switching spike-oscillator model
2. Fit model with EM
3. Visualize:
   - True vs inferred discrete states
   - True vs smoothed oscillator trajectories
   - Learned coupling patterns per state
   - Neuron loadings on oscillatory modes

---

## 3. File Structure Summary

```
src/state_space_practice/
├── switching_point_process.py       # NEW: Core implementation
│   ├── SpikeObsParams
│   ├── point_process_kalman_update
│   ├── switching_point_process_filter
│   ├── switching_point_process_smoother (wraps switching_kalman_smoother)
│   ├── update_spike_glm_params
│   └── SwitchingSpikeOscillatorModel
├── simulate/
│   └── simulate_switching_spikes.py  # NEW: Simulation utilities
└── tests/
    └── test_switching_point_process.py  # NEW: Tests

notebooks/
└── switching_spike_oscillator_demo.py  # NEW: Example
```

---

## 4. Dependencies on Existing Code

### From `switching_kalman.py`

- `collapse_gaussian_mixture` (line 34-64)
- `collapse_gaussian_mixture_per_discrete_state` (line 109)
- `_update_discrete_state_probabilities` (line 158-206)
- `_scale_likelihood` (line 209-226)
- `switching_kalman_smoother` (line 486-714)
- `switching_kalman_maximization_step` (line 749-953)

### From `point_process_kalman.py`

- Gradient/Hessian computation patterns (line 101-157)
- `log_conditional_intensity` (line 20-35)
- Newton-Raphson update structure

### From `oscillator_utils.py`

- `construct_directed_influence_transition_matrix` (line 302-387)
- `construct_common_oscillator_process_covariance` (line 201-223)
- `project_coupled_transition_matrix` (line 573-632)

### From `oscillator_models.py`

- `BaseModel` class structure for EM (line 71-553)
- `DirectedInfluenceModel` reparameterized M-step pattern (line 1087-1198)

---

## 5. Implementation Order

1. **Phase 1**: Core update function - establishes foundation
2. **Phase 4**: `SpikeObsParams` dataclass - needed early
3. **Phase 2**: Filter - depends on Phase 1
4. **Phase 3**: Smoother - simple wrapper, quick win
5. **Phase 4 (continued)**: Spike GLM M-step
6. **Phase 5**: Verify dynamics M-step reuse
7. **Phase 8**: Simulation - enables testing
8. **Phase 7**: Tests - validate implementation
9. **Phase 6**: Model class - integrate everything
10. **Phase 9**: Demo notebook

---

## 6. Potential Extensions

1. **Non-linear intensity functions**: Support arbitrary `log_intensity_func` beyond linear
2. **History dependence**: Add spike history features (refractory effects)
3. **Multiple spike types**: Different neurons with different coupling to oscillators
4. **LFP + Spikes jointly**: Multi-modal observations
5. **Online learning**: Streaming EM for real-time applications

---

## 7. Neuroscientific Applications

With this implementation, researchers can:

1. **Infer dynamic coupling from spikes**: Discover oscillatory coupling patterns that switch with behavior
2. **Compare LFP vs spike-inferred networks**: Validate that spike-based and LFP-based coupling agree
3. **Cell-type specific oscillator loading**: Which neurons load onto which oscillatory modes?
4. **Replay detection**: Do discrete state transitions align with replay events?
5. **Sleep stage classification**: Different coupling patterns in different sleep stages
