# Implementation Plan: Real CA1 Theta State Analysis

## Overview

Apply the `SwitchingSpikeOscillatorModel` to real hippocampal CA1 recordings to:

1. Segment theta-on vs theta-off states from spikes alone
2. Recover theta phase and amplitude from spike activity
3. Validate against behavioral correlates (running vs immobility)
4. Demonstrate downstream applications (phase-locking analysis)

**Note**: No LFP data is available in the current dataset. Validation will rely on
behavioral correlates (speed/running) rather than LFP theta power/phase.

## Data Source

**J16 Bandit Session** (`data/j1620210710_02_r1_*`):

- 203 CA1 units, 870k total spikes
- ~24 minutes at 500 Hz position sampling
- Position, speed, linear track position available
- Plus maze with 3 reward patches

---

## Phase 1: Data Preprocessing Notebook

**File**: `notebooks/real_ca1_data_exploration.py`

### 1.1 Load and Explore Data

```python
# Load data using existing utility
from data.load_bandit_data import load_neural_recording_from_files

data = load_neural_recording_from_files('data/', 'j1620210710_02_r1')
position_info = data['position_info']
spike_times = data['spike_times']  # List of 203 arrays
```

### 1.2 Basic Data Quality Checks

- [ ] Plot position trajectory on track
- [ ] Compute speed distribution
- [ ] Plot spike raster (subset of neurons)
- [ ] Compute firing rate distributions per unit
- [ ] Identify periods of running vs immobility (speed threshold ~5 cm/s)

### 1.3 Bin Spikes into Time Series

```python
def bin_spike_times(
    spike_times: list[np.ndarray],
    time_bins: np.ndarray,
) -> np.ndarray:
    """Bin spike times into count matrix.

    Parameters
    ----------
    spike_times : list of arrays
        Spike times for each unit (in seconds)
    time_bins : array
        Bin edges (left edges, uniform spacing assumed)

    Returns
    -------
    spikes : array, shape (n_time, n_neurons)
        Spike counts per bin
    """
```

### 1.4 Select Analysis Parameters

- **Sampling frequency**: 250 Hz (4 ms bins) - better temporal resolution for theta phase
- **Time selection**: Choose continuous running epochs (>5 cm/s for >2s)
- **Unit selection**:
  - Include all units with at least some spikes (place cells have low mean rates ~0.1-0.5 Hz)
  - Exclude units with >50 Hz mean rate (likely MUA)
  - Target: 20-100+ CA1 units

### 1.5 Create Behavioral Epoch Labels

```python
# From speed, create:
# - running_mask: speed > 5 cm/s
# - immobility_mask: speed < 2 cm/s
# - transition_mask: in between

# Identify continuous bouts:
# - running_bouts: list of (start_idx, end_idx) for running epochs
# - immobility_bouts: list of (start_idx, end_idx) for rest epochs
```

### 1.6 Save Preprocessed Data

```python
# Save to pickle for use in subsequent notebooks:
# - binned_spikes: (n_time, n_neurons) at 50 Hz
# - time_axis: timestamps
# - speed: interpolated to 50 Hz
# - behavioral_labels: 0=immobility, 1=running, 2=transition
# - selected_unit_ids: indices of units kept
```

---

## Phase 2: Spike-Only Theta Segmentation

**File**: `notebooks/real_ca1_theta_segmentation.py`

### 2.1 Model Setup

```python
from state_space_practice.switching_point_process import SwitchingSpikeOscillatorModel

# Model configuration
model = SwitchingSpikeOscillatorModel(
    n_oscillators=1,        # Single theta oscillator
    n_neurons=n_selected,   # Number of selected units
    n_discrete_states=2,    # theta-off (0), theta-on (1)
    sampling_freq=250.0,    # 250 Hz
    dt=0.004,               # 4 ms bins
    spike_weight_l2=0.05,   # Moderate regularization
)
```

### 2.2 Fit Model

```python
# Use a subset of data first for faster iteration
# ~5 minutes at 250 Hz = 75000 timesteps

log_likelihoods = model.fit(
    spikes=binned_spikes[:75000],
    max_iter=30,
    tol=1e-4,
    key=jax.random.PRNGKey(42),
)
```

### 2.3 Extract Inferred States

```python
# Discrete state probabilities
theta_on_prob = model.smoother_discrete_state_prob[:, 1]  # After handling label switching

# Marginalized continuous state
smoother_mean = jnp.einsum(
    'tls,ts->tl',
    model.smoother_state_cond_mean,
    model.smoother_discrete_state_prob,
)

# Compute amplitude and phase
inferred_amplitude = jnp.linalg.norm(smoother_mean, axis=1)
inferred_phase = jnp.arctan2(smoother_mean[:, 1], smoother_mean[:, 0])
```

### 2.4 Visualize Results

- [ ] Plot P(theta-on) vs time, overlay speed
- [ ] Plot inferred amplitude vs time
- [ ] Spike raster colored by inferred discrete state
- [ ] Zoomed view: oscillator trajectory during inferred theta-on

### 2.5 Compare to Behavioral Labels

```python
# Classification metrics
from sklearn.metrics import roc_auc_score, classification_report

# Compare inferred theta-on to running
inferred_theta_on = theta_on_prob > 0.5
behavioral_running = behavioral_labels == 1

# Confusion matrix, ROC curve
# Note: Perfect agreement NOT expected - theta can occur during some immobility
```

---

## Phase 3: Learned Parameters Analysis

**File**: Same as Phase 2 or separate `notebooks/real_ca1_parameter_analysis.py`

### 3.1 Transition Matrix Analysis

```python
# Extract learned dynamics
A_theta_off = model.continuous_transition_matrix[:, :, 0]
A_theta_on = model.continuous_transition_matrix[:, :, 1]

# Compute spectral properties
def analyze_oscillator_dynamics(A: np.ndarray) -> dict:
    """Extract frequency and damping from 2x2 rotation block."""
    eigenvalues = np.linalg.eigvals(A)
    # Complex eigenvalue: r * exp(i*omega*dt)
    # r = damping, omega = frequency
    r = np.abs(eigenvalues[0])
    omega = np.angle(eigenvalues[0])
    freq_hz = omega * sampling_freq / (2 * np.pi)
    return {'damping': r, 'frequency_hz': freq_hz}
```

### 3.2 Process Noise Analysis

```python
# Compare process variance between states
Q_theta_off = model.process_cov[:, :, 0]
Q_theta_on = model.process_cov[:, :, 1]

# In theta demo: theta-on has larger variance (sustained oscillations)
# Check if real data shows similar pattern
```

### 3.3 Spike Weight Analysis

```python
# Each neuron's coupling to the oscillator
weights = model.spike_params.weights  # (n_neurons, 2)
baselines = model.spike_params.baseline  # (n_neurons,)

# Compute preferred phase for each neuron
preferred_phases = np.arctan2(weights[:, 1], weights[:, 0])
modulation_strength = np.linalg.norm(weights, axis=1)

# Plot: polar histogram of preferred phases
# Plot: modulation strength distribution
```

---

## Phase 4: Phase-Locking Analysis

**File**: `notebooks/real_ca1_phase_locking.py`

### 4.1 Spike-Phase Histograms Using Inferred Phase

```python
def compute_phase_histogram(
    spike_times: np.ndarray,
    inferred_phase: np.ndarray,
    time_axis: np.ndarray,
    theta_on_mask: np.ndarray,
    n_bins: int = 36,
) -> np.ndarray:
    """Compute spike-phase histogram during theta-on epochs."""
    # Interpolate inferred_phase to spike times
    # Only include spikes during theta_on_mask
    # Bin into phase histogram
```

### 4.2 Rayleigh Test for Phase Locking

```python
from scipy.stats import circmean, circstd

def rayleigh_test(phases: np.ndarray) -> tuple:
    """Test for non-uniformity of circular distribution."""
    n = len(phases)
    R = np.abs(np.mean(np.exp(1j * phases)))
    z = n * R**2
    p_value = np.exp(-z)  # Rayleigh test
    return R, p_value
```

### 4.3 Validate Phase-Locking with GLM

```python
# Build place-cell GLM with phase regressor
# log(lambda) = f(position) + g(phase)
#
# Use inferred phase from model as regressor
# Compare log-likelihood with/without phase term
# Quantify deviance explained by phase
```

---

## Phase 5: Extended Models (Future Work)

### 5.1 Three-State Model: Theta-Off / Theta-On / Ambiguous

```python
model_3state = SwitchingSpikeOscillatorModel(
    n_oscillators=1,
    n_neurons=n_selected,
    n_discrete_states=3,  # theta-off, theta-on, transition/ambiguous
    sampling_freq=250.0,
    dt=0.004,
)
```

### 5.2 Two-Oscillator Model: Global Theta + Assembly Mode

```python
model_2osc = SwitchingSpikeOscillatorModel(
    n_oscillators=2,  # theta + assembly
    n_neurons=n_selected,
    n_discrete_states=2,
    sampling_freq=250.0,
    dt=0.004,
)
# Analyze: does oscillator 2 capture trial-type or location differences?
```

### 5.3 Theta vs SWR Detection (Requires Rest Data)

```python
# For SWR detection, consider:
# - Different time resolution (maybe 100 Hz for fast transients)
# - Non-oscillatory latent dimension for ripple "envelope"
# - Train on mixed run+rest data
```

---

## Implementation Order

### Week 1: Data Exploration & Preprocessing

1. **Day 1-2**: Phase 1.1-1.3 (load, explore, bin spikes)
2. **Day 3-4**: Phase 1.4-1.6 (select units, create labels, save)

### Week 2: Core Model Fitting

1. **Day 1-2**: Phase 2.1-2.3 (fit model, extract states)
2. **Day 3-4**: Phase 2.4-2.5 (visualize, compare to behavior)

### Week 3: Analysis & Validation

1. **Day 1-2**: Phase 3 (parameter analysis)
2. **Day 3-4**: Phase 4 (phase-locking analysis)

### Week 4: Extensions

1. Phase 5 (extended models)

---

## Key Metrics for Success

### Minimum Viable Demo

- [ ] Model fits without numerical issues
- [ ] P(theta-on) correlates with running epochs (r > 0.5 or AUC > 0.7)
- [ ] Inferred frequency is in theta range (6-10 Hz)
- [ ] Amplitude is higher during running vs immobility

### Strong Demo

- [ ] Clear separation of theta-on/off states in discrete probability
- [ ] Phase-locked neurons show consistent preferred phases
- [ ] Learned dynamics show expected pattern (low damping for theta-on)
- [ ] Visual spike rasters show rhythmic firing during theta-on

### Publication-Ready Demo

- [ ] Behavioral validation shows strong running/theta correlation
- [ ] Spike-inferred phase predicts spiking (phase regressor improves GLM)
- [ ] Novel insights: neuron-level phase preferences from spikes alone

---

## Utility Functions Needed

### `bin_spike_times(spike_times, time_bins) -> spikes`

Bin list of spike time arrays into count matrix.

### `select_units(spike_times, min_rate, max_rate) -> selected_indices`

Filter units by firing rate criteria.

### `identify_behavioral_bouts(speed, min_duration, speed_threshold) -> bout_list`

Find continuous running/immobility epochs.

### `circular_correlation(phase1, phase2) -> r`

Compute circular correlation coefficient.

### `rayleigh_test(phases) -> (R, p_value)`

Test for non-uniform circular distribution.

---

## File Structure

```text
notebooks/
├── real_ca1_data_exploration.py      # Phase 1
├── real_ca1_data_exploration.ipynb
├── real_ca1_theta_segmentation.py    # Phase 2 & 3
├── real_ca1_theta_segmentation.ipynb
├── real_ca1_phase_locking.py         # Phase 4
└── real_ca1_phase_locking.ipynb

src/state_space_practice/
├── preprocessing.py                   # New: spike binning utilities
└── circular_stats.py                  # New: circular statistics
```

---

## Notes

### Label Switching

The model identifies discrete states but doesn't know which is "theta-on". Handle by:

1. Checking which state has higher amplitude oscillations
2. Checking which state correlates with running
3. Applying consistent relabeling post-hoc

### Oscillator Sign Ambiguity

The latent oscillator x can be flipped to -x without changing the model. Handle by:

1. Checking correlation with expected phase direction
2. Applying sign correction for visualization

### Computational Considerations

- Start with shorter time windows (~5 min) for rapid iteration
- Full dataset (~24 min at 250 Hz = 360k timesteps) will take longer
- Consider subsampling units if fitting is slow
