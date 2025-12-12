# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Switching Spike-Based Oscillator Network Demo
#
# This notebook demonstrates the `SwitchingSpikeOscillatorModel` - a switching linear
# dynamical system (SLDS) that combines:
#
# - **Oscillator network dynamics**: Multiple 2D oscillators with state-dependent coupling
# - **Discrete state switching**: Different coupling patterns in each discrete regime
# - **Point-process (spike) observations**: Poisson spike observations via Laplace-EKF
#
# ## Model Structure
#
# **Latent Dynamics** (per discrete state $s_t$):
# $$x_t = A_{s_t} x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q_{s_t})$$
#
# where:
# - $x_t \in \mathbb{R}^{2K}$ is a stack of $K$ 2D oscillators
# - $s_t \in \{1, \ldots, S\}$ indexes different coupling patterns
# - $A_{s_t}$ is the state-dependent transition matrix
#
# **Spike Observation Model** (for $N$ neurons):
# $$\log \lambda_{n,t} = b_n + c_n^\top x_t$$
# $$y_{n,t} \sim \text{Poisson}(\lambda_{n,t} \Delta t)$$

# %% [markdown]
# ## Setup

# %%
# Enable 64-bit precision for numerical stability
import jax

jax.config.update("jax_enable_x64", True)

# %%
# Imports
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
)
from state_space_practice.simulate.simulate_switching_spikes import (
    simulate_switching_spike_oscillator,
)
from state_space_practice.switching_point_process import SwitchingSpikeOscillatorModel

# Set random seed for reproducibility
SEED = 42
key = jax.random.PRNGKey(SEED)

print(f"JAX version: {jax.__version__}")
print(f"Using device: {jax.devices()[0]}")
print(f"Random seed: {SEED}")

# %% [markdown]
# ## 1. Simulate Data
#
# We'll create a switching oscillator network with:
# - **2 oscillators** (4D latent state: 2 dimensions per oscillator)
# - **2 discrete states** (different stability: stable vs. variable)
# - **40 neurons** (dense population providing rich observations)
# - **~4000 timesteps** at 50 Hz (80 seconds of data)
#
# The two discrete states differ in both **damping** and **process noise**:
# - **State 0**: High damping (0.995), low noise (0.01) - tight, sustained oscillations
# - **State 1**: Lower damping (0.85), higher noise (0.1) - more variable, decaying oscillations
#
# This creates clearly distinguishable regimes that the model can learn.

# %%
# Simulation parameters
n_time = 4000  # Number of time steps (longer for better estimation)
n_oscillators = 2  # Number of oscillators
n_neurons = 40  # More neurons -> more information about latent states
n_discrete_states = 2  # Number of discrete regimes
n_latent = 2 * n_oscillators  # Latent dimension (4)

sampling_freq = 50.0  # Hz
dt = 1.0 / sampling_freq  # 20 ms bins

print(f"Simulation setup:")
print(f"  Time steps: {n_time}")
print(f"  Duration: {n_time * dt:.1f} seconds")
print(f"  Oscillators: {n_oscillators}")
print(f"  Latent dimension: {n_latent}")
print(f"  Neurons: {n_neurons}")
print(f"  Discrete states: {n_discrete_states}")

# %%
# Define oscillator parameters
# States differ in BOTH damping and process noise for clear identifiability
# State 0: Stable oscillations (high damping, low noise) - tight trajectories
# State 1: Variable oscillations (lower damping, higher noise) - messy trajectories

freqs_state0 = jnp.array([8.0, 12.0])  # Hz (theta/alpha)
freqs_state1 = jnp.array([8.0, 12.0])  # Same frequencies

# Strongly different damping - creates clearly different amplitude dynamics
damping_state0 = jnp.array([0.995, 0.995])  # Very stable - sustained oscillations
damping_state1 = jnp.array([0.85, 0.85])  # More lossy - quicker decay

# Build transition matrices for each discrete state
A_state0 = construct_common_oscillator_transition_matrix(
    freqs_state0, damping_state0, sampling_freq
)
A_state1 = construct_common_oscillator_transition_matrix(
    freqs_state1, damping_state1, sampling_freq
)

# Stack into (n_latent, n_latent, n_discrete_states) array
transition_matrices = jnp.stack([A_state0, A_state1], axis=-1)

print(f"Transition matrices shape: {transition_matrices.shape}")
print(f"State 0: freqs={freqs_state0} Hz, damping={damping_state0} (very stable)")
print(f"State 1: freqs={freqs_state1} Hz, damping={damping_state1} (more lossy)")

# %%
# Process noise covariances - amplifies the state difference
# State 0: Very low noise - tight, stable oscillations
# State 1: Higher noise - more variable oscillations (10x difference)
variance_state0 = jnp.array([0.01, 0.01])  # Very low noise
variance_state1 = jnp.array([0.1, 0.1])  # Higher noise (10x difference)

Q_state0 = construct_common_oscillator_process_covariance(variance_state0)
Q_state1 = construct_common_oscillator_process_covariance(variance_state1)

# Stack into (n_latent, n_latent, n_discrete_states) array
process_covs = jnp.stack([Q_state0, Q_state1], axis=-1)

print(f"Process covariance shape: {process_covs.shape}")
print(f"State 0 variance: {variance_state0} (very low - tight oscillations)")
print(f"State 1 variance: {variance_state1} (higher - 10x difference)")

# %%
# Discrete state transition matrix
# Very high probability of staying in the same state -> long, clean blocks
stay_prob = 0.99
transition_prob = 1.0 - stay_prob

discrete_transition_matrix = jnp.array(
    [
        [stay_prob, transition_prob],
        [transition_prob, stay_prob],
    ]
)

expected_dwell = 1.0 / transition_prob
print(f"Discrete transition matrix:")
print(discrete_transition_matrix)
print(f"Expected dwell time: {expected_dwell:.0f} steps ({expected_dwell * dt:.1f} seconds)")

# %%
# Spike observation model parameters
# Each neuron has different coupling to the oscillators
key, key_weights = jax.random.split(key)

# Baseline log-firing rates (~8 Hz when latent is 0)
spike_baseline = jnp.ones(n_neurons) * jnp.log(8.0)  # ~8 Hz baseline

# Weights: how each neuron couples to oscillators
# Moderate weights (0.5 scale) + many neurons = highly informative observations
spike_weights = jax.random.normal(key_weights, (n_neurons, n_latent)) * 0.5

print(f"Spike baseline shape: {spike_baseline.shape}")
print(f"Spike weights shape: {spike_weights.shape}")
print(f"Baseline firing rate: {jnp.exp(spike_baseline[0]):.1f} Hz")
print(f"Weight magnitude (mean |w|): {jnp.abs(spike_weights).mean():.2f}")

# %%
# Run simulation
key, key_sim = jax.random.split(key)

spikes, true_states, true_discrete_states = simulate_switching_spike_oscillator(
    n_time=n_time,
    transition_matrices=transition_matrices,
    process_covs=process_covs,
    discrete_transition_matrix=discrete_transition_matrix,
    spike_weights=spike_weights,
    spike_baseline=spike_baseline,
    dt=dt,
    key=key_sim,
)

# Summary statistics
n_transitions = int(jnp.sum(jnp.abs(jnp.diff(true_discrete_states))))
total_spikes = int(jnp.sum(spikes))
mean_rate = total_spikes / (n_time * dt * n_neurons)

print(f"\nSimulation results:")
print(f"  Spikes shape: {spikes.shape}")
print(f"  True states shape: {true_states.shape}")
print(f"  True discrete states shape: {true_discrete_states.shape}")
print(f"  Number of state transitions: {n_transitions}")
print(f"  Total spike count: {total_spikes}")
print(f"  Mean firing rate: {mean_rate:.1f} Hz")
print(f"  Time in state 0: {jnp.mean(true_discrete_states == 0) * 100:.1f}%")
print(f"  Time in state 1: {jnp.mean(true_discrete_states == 1) * 100:.1f}%")

# %% [markdown]
# ## 2. Visualize Simulated Data

# %%
# Plot simulated data overview
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
time = np.arange(n_time) * dt

# Panel 1: Spike raster
ax = axes[0]
for n in range(n_neurons):
    spike_times = time[np.array(spikes[:, n]) > 0]
    ax.eventplot([spike_times], lineoffsets=n, linelengths=0.8, colors="black", linewidths=0.5)
ax.set_ylabel("Neuron")
ax.set_ylim(-0.5, n_neurons - 0.5)
ax.set_title("Simulated Data: Switching Spike-Based Oscillator Network")

# Panel 2: True discrete state
ax = axes[1]
ax.fill_between(time, true_discrete_states, alpha=0.7, step="mid", color="C1")
ax.set_ylabel("Discrete State")
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])

# Panel 3: True oscillator 1 trajectory (amplitude and phase as 2D)
ax = axes[2]
ax.plot(time, true_states[:, 0], "C0-", alpha=0.8, label="Osc 1, dim 1")
ax.plot(time, true_states[:, 1], "C0--", alpha=0.5, label="Osc 1, dim 2")
ax.set_ylabel("Oscillator 1")
ax.legend(loc="upper right", fontsize=8)
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

# Panel 4: True oscillator 2 trajectory
ax = axes[3]
ax.plot(time, true_states[:, 2], "C2-", alpha=0.8, label="Osc 2, dim 1")
ax.plot(time, true_states[:, 3], "C2--", alpha=0.5, label="Osc 2, dim 2")
ax.set_ylabel("Oscillator 2")
ax.set_xlabel("Time (s)")
ax.legend(loc="upper right", fontsize=8)
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("simulated_data_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Fit Model
#
# Now we fit the `SwitchingSpikeOscillatorModel` to the simulated spike data using the EM algorithm.

# %%
# Create model
model = SwitchingSpikeOscillatorModel(
    n_oscillators=n_oscillators,
    n_neurons=n_neurons,
    n_discrete_states=n_discrete_states,
    sampling_freq=sampling_freq,
    dt=dt,
)

print(f"Model: {model}")

# %%
# Fit model with EM
print("\nFitting model with EM algorithm...")
print("-" * 50)

key, key_fit = jax.random.split(key)
log_likelihoods = model.fit(
    spikes=spikes,
    max_iter=30,
    tol=1e-4,
    key=key_fit,
)

print(f"\nEM completed after {len(log_likelihoods)} iterations")
print(f"  Initial log-likelihood: {log_likelihoods[0]:.1f}")
print(f"  Final log-likelihood: {log_likelihoods[-1]:.1f}")
print(f"  Improvement: {log_likelihoods[-1] - log_likelihoods[0]:.1f}")

# %%
# Plot log-likelihood convergence
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, "b.-", markersize=8)
ax.set_xlabel("EM Iteration")
ax.set_ylabel("Marginal Log-Likelihood")
ax.set_title("EM Convergence")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("em_convergence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Visualization: Discrete States
#
# Compare the true discrete states with the inferred state probabilities.

# %%
# Get smoothed discrete state probabilities
smoother_discrete_prob = model.smoother_discrete_state_prob

# Compute accuracy (handling label switching)
inferred_states = jnp.argmax(smoother_discrete_prob, axis=1)
accuracy_direct = jnp.mean(inferred_states == true_discrete_states)
accuracy_flipped = jnp.mean(inferred_states == (1 - true_discrete_states))

# Handle label switching
if accuracy_flipped > accuracy_direct:
    print("Note: Labels were flipped")
    inferred_states = 1 - inferred_states
    smoother_discrete_prob = smoother_discrete_prob[:, ::-1]
    accuracy = float(accuracy_flipped)
else:
    accuracy = float(accuracy_direct)

print(f"Discrete state classification accuracy: {accuracy:.1%}")

# %%
# Plot true vs inferred discrete states
fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

# Panel 1: True discrete state
ax = axes[0]
ax.fill_between(time, true_discrete_states, alpha=0.7, step="mid", color="C0", label="True")
ax.set_ylabel("True State")
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.legend(loc="upper right")
ax.set_title(f"Discrete State Inference (Accuracy: {accuracy:.1%})")

# Panel 2: Inferred state probability
ax = axes[1]
ax.fill_between(
    time,
    smoother_discrete_prob[:, 1],
    alpha=0.7,
    color="C1",
    label="P(State 1 | data)",
)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("P(State 1)")
ax.set_ylim(-0.1, 1.1)
ax.legend(loc="upper right")

# Panel 3: Comparison
ax = axes[2]
ax.fill_between(
    time, true_discrete_states, alpha=0.4, step="mid", color="C0", label="True state"
)
ax.plot(time, smoother_discrete_prob[:, 1], "C1-", alpha=0.8, label="P(State 1)")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("State")
ax.set_xlabel("Time (s)")
ax.set_ylim(-0.1, 1.1)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("discrete_state_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Visualization: Oscillator Trajectories
#
# Compare the true latent oscillator states with the smoothed estimates.

# %%
# Compute marginalized smoother mean (weighted average over discrete states)
smoother_mean = jnp.einsum(
    "tls,ts->tl", model.smoother_state_cond_mean, smoother_discrete_prob
)

# Compute correlation with true states
# Note: Due to latent space non-identifiability, the sign of each dimension is arbitrary.
# We use absolute value of correlation to account for sign flips.
correlations = []
for dim in range(n_latent):
    corr = jnp.corrcoef(true_states[:, dim], smoother_mean[:, dim])[0, 1]
    correlations.append(float(jnp.abs(corr)))  # Absolute value for sign ambiguity

print("Latent state correlations (true vs inferred, absolute value):")
for dim in range(n_latent):
    osc = dim // 2 + 1
    subdim = dim % 2 + 1
    print(f"  Oscillator {osc}, dim {subdim}: {correlations[dim]:.3f}")
print(f"  Mean correlation: {np.mean(correlations):.3f}")

# %%
# Plot oscillator trajectories
# Sign-correct the inferred states for visualization (match sign of correlation)
smoother_mean_signed = smoother_mean.copy()
for dim in range(n_latent):
    corr_raw = jnp.corrcoef(true_states[:, dim], smoother_mean[:, dim])[0, 1]
    if corr_raw < 0:
        smoother_mean_signed = smoother_mean_signed.at[:, dim].set(-smoother_mean[:, dim])

fig, axes = plt.subplots(n_oscillators, 1, figsize=(14, 3 * n_oscillators), sharex=True)

colors = ["C0", "C2"]
for osc in range(n_oscillators):
    ax = axes[osc]
    dim1, dim2 = 2 * osc, 2 * osc + 1

    # True states
    ax.plot(time, true_states[:, dim1], f"{colors[osc]}-", alpha=0.4, linewidth=1, label="True (dim 1)")
    ax.plot(time, true_states[:, dim2], f"{colors[osc]}--", alpha=0.3, linewidth=1, label="True (dim 2)")

    # Inferred states (sign-corrected for visualization)
    ax.plot(time, smoother_mean_signed[:, dim1], "k-", alpha=0.8, linewidth=1.5, label="Inferred (dim 1)")
    ax.plot(time, smoother_mean_signed[:, dim2], "k--", alpha=0.6, linewidth=1.5, label="Inferred (dim 2)")

    ax.set_ylabel(f"Oscillator {osc + 1}")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    corr_avg = (correlations[dim1] + correlations[dim2]) / 2
    ax.set_title(f"Oscillator {osc + 1} (mean |correlation|: {corr_avg:.3f})")

axes[-1].set_xlabel("Time (s)")
fig.suptitle("True vs Inferred Oscillator Trajectories", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("oscillator_trajectories.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Visualization: Learned Coupling Matrices
#
# Examine the learned transition matrices (coupling patterns) for each discrete state.

# %%
# Get learned transition matrices
learned_A = model.continuous_transition_matrix

print("Learned transition matrix shapes:", learned_A.shape)

# %%
# Plot learned vs true transition matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# True transition matrices
vmin, vmax = -1, 1

ax = axes[0, 0]
im = ax.imshow(transition_matrices[:, :, 0], cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("True A (State 0)")
ax.set_xlabel("From dimension")
ax.set_ylabel("To dimension")
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
im = ax.imshow(transition_matrices[:, :, 1], cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("True A (State 1)")
ax.set_xlabel("From dimension")
ax.set_ylabel("To dimension")
plt.colorbar(im, ax=ax)

# Learned transition matrices
ax = axes[1, 0]
im = ax.imshow(learned_A[:, :, 0], cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("Learned A (State 0)")
ax.set_xlabel("From dimension")
ax.set_ylabel("To dimension")
plt.colorbar(im, ax=ax)

ax = axes[1, 1]
im = ax.imshow(learned_A[:, :, 1], cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("Learned A (State 1)")
ax.set_xlabel("From dimension")
ax.set_ylabel("To dimension")
plt.colorbar(im, ax=ax)

plt.suptitle("Transition Matrices: True vs Learned", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("transition_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Compare spectral properties (eigenvalues)
print("Spectral radius comparison:")
for state in range(n_discrete_states):
    true_eigs = jnp.linalg.eigvals(transition_matrices[:, :, state])
    learned_eigs = jnp.linalg.eigvals(learned_A[:, :, state])

    true_spectral_radius = float(jnp.max(jnp.abs(true_eigs)))
    learned_spectral_radius = float(jnp.max(jnp.abs(learned_eigs)))

    print(f"  State {state}:")
    print(f"    True spectral radius: {true_spectral_radius:.4f}")
    print(f"    Learned spectral radius: {learned_spectral_radius:.4f}")

# %% [markdown]
# ## 7. Visualization: Neuron Loadings
#
# Examine how each neuron couples to the oscillatory modes through the spike weights.

# %%
# Get learned spike parameters
learned_baseline = model.spike_params.baseline
learned_weights = model.spike_params.weights

print(f"Learned spike parameters:")
print(f"  Baseline shape: {learned_baseline.shape}")
print(f"  Weights shape: {learned_weights.shape}")

# %%
# Plot neuron loadings heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# True weights
ax = axes[0]
im = ax.imshow(spike_weights, cmap="RdBu_r", aspect="auto")
ax.set_title("True Spike Weights (C)")
ax.set_xlabel("Latent Dimension")
ax.set_ylabel("Neuron")
ax.set_xticks(range(n_latent))
ax.set_xticklabels([f"Osc{i//2+1}.d{i%2+1}" for i in range(n_latent)])
plt.colorbar(im, ax=ax)

# Learned weights
ax = axes[1]
im = ax.imshow(learned_weights, cmap="RdBu_r", aspect="auto")
ax.set_title("Learned Spike Weights (C)")
ax.set_xlabel("Latent Dimension")
ax.set_ylabel("Neuron")
ax.set_xticks(range(n_latent))
ax.set_xticklabels([f"Osc{i//2+1}.d{i%2+1}" for i in range(n_latent)])
plt.colorbar(im, ax=ax)

plt.suptitle("Neuron Loadings on Oscillators", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("neuron_loadings.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Compare baseline firing rates
print("Baseline firing rate comparison:")
print(f"  True baseline (Hz): {float(jnp.exp(spike_baseline[0])):.2f}")
print(f"  Learned baseline mean (Hz): {float(jnp.mean(jnp.exp(learned_baseline))):.2f}")
print(f"  Learned baseline std (Hz): {float(jnp.std(jnp.exp(learned_baseline))):.2f}")

# Weight correlation
# Note: Due to latent space non-identifiability, direct comparison may not be meaningful
# Instead, look at whether weight structure is preserved (relative loadings)
print("\nWeight statistics:")
print(f"  True weight range: [{float(spike_weights.min()):.3f}, {float(spike_weights.max()):.3f}]")
print(f"  Learned weight range: [{float(learned_weights.min()):.3f}, {float(learned_weights.max()):.3f}]")

# %% [markdown]
# ## 8. Summary
#
# This demo demonstrated the `SwitchingSpikeOscillatorModel` for inferring switching oscillator
# dynamics from spike observations. The two discrete states are clearly distinguishable through
# their different stability characteristics (damping) and noise levels.
#
# The model reliably:
#
# 1. **EM convergence**: The log-likelihood improves dramatically across iterations, confirming
#    that the EM algorithm is working correctly.
#
# 2. **Discrete state inference**: The model identifies discrete state transitions with high
#    accuracy, clearly distinguishing "stable" (tight oscillations) vs "variable" (messy oscillations).
#
# 3. **Oscillator tracking**: Latent oscillator trajectories are recovered with good correlations.
#
# 4. **Parameter recovery**: The model learns transition matrices that capture the key difference
#    between states (different spectral radii corresponding to different damping).
#
# **Key design choices**:
# - **Different damping**: State 0 (0.995) vs State 1 (0.85) - creates clearly different dynamics
# - **Different noise**: 10x variance difference (0.01 vs 0.1) amplifies the state separation
# - **Dense observations**: 40 neurons provide rich information about latent dynamics
# - **Long dwell times**: 99% stay probability creates clean blocks for classification
#
# **Technical notes**:
# - Latent space non-identifiability (sign/rotation ambiguity) is handled via absolute correlations
# - The Laplace approximation can cause small EM monotonicity violations
# - This demo uses parameters designed for clear demonstration; real data may be more challenging

# %%
# Final summary statistics
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  Data: {n_time} timesteps, {n_neurons} neurons, {n_oscillators} oscillators")
print(f"  EM iterations: {len(log_likelihoods)}")
print(f"  Final log-likelihood: {log_likelihoods[-1]:.1f}")
print(f"  Discrete state accuracy: {accuracy:.1%}")
print(f"  Mean latent correlation: {np.mean(correlations):.3f}")
print("=" * 60)
