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
# # Theta-On vs Theta-Off Hippocampal Demo
#
# This notebook demonstrates the `SwitchingSpikeOscillatorModel` in a physiologically
# motivated scenario: **theta rhythms in hippocampus**.
#
# ## Concept: Hippocampal Theta States
#
# A rat alternates between:
#
# - **State 0 = Theta-off (immobility)**
#   - Little or no theta oscillation
#   - Neurons fire at low baseline rates, weakly modulated
#
# - **State 1 = Theta-on (running)**
#   - Strong 8 Hz theta oscillation in CA1
#   - Neurons are strongly **phase-locked** to theta with different preferred phases
#
# ## Model Structure
#
# We represent theta as a **single 2D oscillator**:
#
# $$
# x_t = \begin{bmatrix} x_t^{(1)} \\ x_t^{(2)} \end{bmatrix} \in \mathbb{R}^2
# $$
#
# where the **phase** is $\theta_t = \operatorname{atan2}(x_t^{(2)}, x_t^{(1)})$
# and the **amplitude** is $r_t = \sqrt{(x_t^{(1)})^2 + (x_t^{(2)})^2}$.
#
# **Discrete state effects**:
# - In **theta-on** ($s_t = 1$): High damping (~0.99), moderate noise → large amplitude
# - In **theta-off** ($s_t = 0$): Low damping (~0.7), tiny noise → amplitude near zero
#
# **Spike observation model** for neuron $n$:
# $$
# \log \lambda_{n,t} = b_n + c_n^\top x_t
# $$
#
# with phase-locked weights:
# $$
# c_n = \alpha \begin{bmatrix} \cos \phi_n \\ \sin \phi_n \end{bmatrix}
# $$
#
# so each neuron has a different **preferred phase** $\phi_n$.
#
# ## Why This Demo is Compelling
#
# 1. **Physiological story**: Matches real hippocampal theta vs non-theta regimes
# 2. **Observable impact**: Theta-on shows rhythmic spiking; theta-off shows sparse noise
# 3. **Model-specific strengths**: Explicitly encodes phase, amplitude, and brain state

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
SEED = 123
key = jax.random.PRNGKey(SEED)

print(f"JAX version: {jax.__version__}")
print(f"Using device: {jax.devices()[0]}")
print(f"Random seed: {SEED}")

# %% [markdown]
# ## 1. Simulation Parameters
#
# Single oscillator (theta), two discrete states, many neurons, long recording:
# - **5000 timesteps** at 50 Hz (100 seconds of data)
# - **40 neurons** with different preferred theta phases
# - **2 discrete states**: theta-off vs theta-on

# %%
# Simulation parameters
n_time = 5000  # 100 s at 50 Hz
n_oscillators = 1  # Single theta oscillator
n_latent = 2 * n_oscillators
n_neurons = 40
n_discrete_states = 2

sampling_freq = 50.0
dt = 1.0 / sampling_freq  # 20 ms bins

print("Simulation setup:")
print(f"  Time steps: {n_time}")
print(f"  Duration: {n_time * dt:.1f} s")
print(f"  Latent dim: {n_latent}")
print(f"  Neurons: {n_neurons}")
print(f"  Discrete states: {n_discrete_states}")

# %% [markdown]
# ## 2. Theta Dynamics: Theta-off vs Theta-on
#
# The key difference between states:
# - **Theta-off (State 0)**: Strongly damped (0.7), tiny noise → amplitude collapses to ~0
# - **Theta-on (State 1)**: Weakly damped (0.99), moderate noise → sustained oscillations

# %%
# Theta frequency ~8 Hz
theta_freq = jnp.array([8.0])  # Hz

# State 0: theta-off (immobility-like): strongly damped, tiny noise
damping_off = jnp.array([0.7])
var_off = jnp.array([0.001])

# State 1: theta-on (running-like): weakly damped, larger noise
damping_on = jnp.array([0.99])
var_on = jnp.array([0.05])

# Build transition matrices
A_off = construct_common_oscillator_transition_matrix(
    theta_freq, damping_off, sampling_freq
)
A_on = construct_common_oscillator_transition_matrix(
    theta_freq, damping_on, sampling_freq
)

# Build process covariances
Q_off = construct_common_oscillator_process_covariance(var_off)
Q_on = construct_common_oscillator_process_covariance(var_on)

# Stack into (n_latent, n_latent, n_discrete_states) arrays
transition_matrices = jnp.stack([A_off, A_on], axis=-1)
process_covs = jnp.stack([Q_off, Q_on], axis=-1)

print(f"Transition matrices shape: {transition_matrices.shape}")
print(f"Process covariances shape: {process_covs.shape}")
print(f"\nState 0 (theta-off): damping={float(damping_off[0])}, var={float(var_off[0])}")
print(f"State 1 (theta-on): damping={float(damping_on[0])}, var={float(var_on[0])}")

# %%
# Verify spectral properties
print("\nSpectral radius (determines stability):")
for state, name in enumerate(["theta-off", "theta-on"]):
    A = transition_matrices[:, :, state]
    spectral_radius = float(jnp.max(jnp.abs(jnp.linalg.eigvals(A))))
    print(f"  State {state} ({name}): {spectral_radius:.3f}")

# %% [markdown]
# ## 3. Discrete State Transitions
#
# High dwell time (~5 seconds per state) creates long, clean theta/non-theta bouts.

# %%
# High dwell time ~5 s per state
stay_prob = 0.99
transition_prob = 1.0 - stay_prob

discrete_transition_matrix = jnp.array(
    [
        [stay_prob, transition_prob],
        [transition_prob, stay_prob],
    ]
)

expected_dwell = 1.0 / transition_prob
print("Discrete transition matrix:")
print(discrete_transition_matrix)
print(f"Expected dwell time: {expected_dwell:.0f} steps ({expected_dwell * dt:.1f} s)")

# %% [markdown]
# ## 4. Phase-Locked Spike GLM
#
# Each neuron has a different preferred theta phase $\phi_n$ uniformly distributed
# around the circle. The spike weights are:
# $$
# c_n = \alpha \begin{bmatrix} \cos \phi_n \\ \sin \phi_n \end{bmatrix}
# $$
#
# This means:
# - In **theta-on**: Neurons fire rhythmically at their preferred phase
# - In **theta-off**: $x_t \approx 0$, so $\lambda_{n,t} \approx e^{b_n}$ (baseline firing)

# %%
# Baseline firing rate ~3 Hz
baseline_rate_hz = 3.0
spike_baseline = jnp.ones(n_neurons) * jnp.log(baseline_rate_hz)

# Preferred phases evenly spaced in [0, 2*pi)
neuron_phases = jnp.linspace(0.0, 2.0 * jnp.pi, n_neurons, endpoint=False)

# Modulation strength
mod_strength = 1.0  # Strong theta modulation

# Each neuron: c_n = alpha * [cos(phi_n), sin(phi_n)]
spike_weights = jnp.stack(
    [
        mod_strength * jnp.cos(neuron_phases),
        mod_strength * jnp.sin(neuron_phases),
    ],
    axis=-1,
)  # (n_neurons, 2)

print(f"Spike baseline shape: {spike_baseline.shape}")
print(f"Spike weights shape: {spike_weights.shape}")
print(f"Baseline firing rate: {baseline_rate_hz:.1f} Hz")
print(f"Modulation strength: {mod_strength}")

# %%
# Visualize neuron phase preferences
fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={"projection": "polar"})
ax.scatter(neuron_phases, jnp.ones(n_neurons), c=np.arange(n_neurons), cmap="hsv", s=50)
ax.set_title("Neuron Preferred Phases")
ax.set_rticks([])
plt.tight_layout()
plt.savefig("neuron_phase_preferences.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Simulate Spikes

# %%
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
total_spikes = int(jnp.sum(spikes))
mean_rate = total_spikes / (n_time * dt * n_neurons)
n_transitions = int(jnp.sum(jnp.abs(jnp.diff(true_discrete_states))))

print("\nSimulation results:")
print(f"  Spikes shape: {spikes.shape}")
print(f"  True states shape: {true_states.shape}")
print(f"  True discrete states shape: {true_discrete_states.shape}")
print(f"  Number of state transitions: {n_transitions}")
print(f"  Total spike count: {total_spikes}")
print(f"  Mean firing rate: {mean_rate:.2f} Hz")
print(f"  Time in state 0 (theta-off): {jnp.mean(true_discrete_states == 0) * 100:.1f}%")
print(f"  Time in state 1 (theta-on): {jnp.mean(true_discrete_states == 1) * 100:.1f}%")

# %% [markdown]
# ## 6. Visualize Simulated Data
#
# This visualization shows the key contrast between theta-on and theta-off:
# - **Theta-on**: Large oscillator amplitude, rhythmic spiking
# - **Theta-off**: Near-zero amplitude, sparse random spiking

# %%
# Compute amplitude and phase from 2D oscillator state
true_amp = np.linalg.norm(np.array(true_states), axis=1)
true_phase = np.arctan2(np.array(true_states[:, 1]), np.array(true_states[:, 0]))

time = np.arange(n_time) * dt

# %%
# Overview plot
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Panel 1: Discrete state
ax = axes[0]
ax.fill_between(
    time,
    true_discrete_states,
    alpha=0.7,
    step="mid",
    color="C1",
    label="Theta-on (state 1)",
)
ax.set_ylabel("Discrete State")
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Theta-off", "Theta-on"])
ax.set_title("Simulated Theta/Hippocampus Data: Theta-On vs Theta-Off")
ax.legend(loc="upper right")

# Panel 2: Theta amplitude
ax = axes[1]
ax.plot(time, true_amp, "C0-", alpha=0.8, linewidth=0.8)
ax.set_ylabel("Theta Amplitude")
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

# Panel 3: Theta phase (only meaningful when amplitude is large)
ax = axes[2]
# Mask phase when amplitude is small
phase_masked = np.where(true_amp > 0.1, true_phase, np.nan)
ax.scatter(time, phase_masked, c="C2", s=0.5, alpha=0.5)
ax.set_ylabel("Theta Phase")
ax.set_ylim(-np.pi, np.pi)
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

# Panel 4: Spike raster
ax = axes[3]
for n in range(n_neurons):
    spike_times = time[np.array(spikes[:, n]) > 0]
    ax.eventplot(
        [spike_times], lineoffsets=n, linelengths=0.8, colors="black", linewidths=0.5
    )
ax.set_ylabel("Neuron")
ax.set_xlabel("Time (s)")
ax.set_ylim(-0.5, n_neurons - 0.5)

plt.tight_layout()
plt.savefig("theta_simulated_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Zoomed view to show phase locking during theta-on
# Find a theta-on period
theta_on_times = np.where(np.array(true_discrete_states) == 1)[0]
if len(theta_on_times) > 0:
    # Find a continuous theta-on bout
    diffs = np.diff(theta_on_times)
    bout_starts = np.concatenate([[0], np.where(diffs > 1)[0] + 1])
    bout_ends = np.concatenate([np.where(diffs > 1)[0], [len(theta_on_times) - 1]])
    bout_lengths = bout_ends - bout_starts

    # Find longest bout
    longest_bout_idx = np.argmax(bout_lengths)
    bout_start_t = theta_on_times[bout_starts[longest_bout_idx]]
    bout_end_t = theta_on_times[bout_ends[longest_bout_idx]]

    # Zoom to first 2 seconds of this bout
    zoom_start = bout_start_t
    zoom_end = min(bout_start_t + int(2.0 / dt), bout_end_t)

    zoom_time = time[zoom_start:zoom_end]
    zoom_states = true_states[zoom_start:zoom_end]
    zoom_spikes = spikes[zoom_start:zoom_end]
    zoom_phase = true_phase[zoom_start:zoom_end]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Oscillator trajectory (both dimensions show sinusoidal pattern)
    ax = axes[0]
    ax.plot(zoom_time, zoom_states[:, 0], "C0-", alpha=0.8, label="x1 (cos-like)")
    ax.plot(zoom_time, zoom_states[:, 1], "C1-", alpha=0.8, label="x2 (sin-like)")
    ax.set_ylabel("Oscillator State")
    ax.legend(loc="upper right")
    ax.set_title("Zoomed: Theta-On Period (Phase-Locked Spiking)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    # Theta phase
    ax = axes[1]
    ax.plot(zoom_time, zoom_phase, "C2-", alpha=0.8)
    ax.set_ylabel("Theta Phase")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

    # Spike raster (subset of neurons to see phase locking)
    ax = axes[2]
    n_show = min(20, n_neurons)
    for n in range(n_show):
        spike_times_zoom = zoom_time[np.array(zoom_spikes[:, n]) > 0]
        ax.eventplot(
            [spike_times_zoom],
            lineoffsets=n,
            linelengths=0.8,
            colors=plt.cm.hsv(neuron_phases[n] / (2 * np.pi)),
            linewidths=1.0,
        )
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-0.5, n_show - 0.5)

    plt.tight_layout()
    plt.savefig("theta_zoomed_phase_locking.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 7. Fit Model
#
# Now we fit the `SwitchingSpikeOscillatorModel` to infer:
# 1. **Discrete states**: When is theta on vs off?
# 2. **Latent trajectory**: What is the theta phase and amplitude over time?

# %%
# Create model
# Use a loose L2 regularization on spike weights (prior std ~ 4.5 on |c|)
# This allows up to ~100x modulation without penalty, preventing only pathological values
model = SwitchingSpikeOscillatorModel(
    n_oscillators=n_oscillators,
    n_neurons=n_neurons,
    n_discrete_states=n_discrete_states,
    sampling_freq=sampling_freq,
    dt=dt,
    spike_weight_l2=0.05,
)

print(f"Model: {model}")

# %%
# Fit model with EM
print("\nFitting model with EM algorithm...")
print("-" * 60)

key, key_fit = jax.random.split(key)
log_likelihoods = model.fit(
    spikes=spikes,
    max_iter=30,
    tol=1e-4,
    key=key_fit,
)

print(f"\nEM completed after {len(log_likelihoods)} iterations")
print(f"  Initial log-likelihood: {float(log_likelihoods[0]):.1f}")
print(f"  Final log-likelihood: {float(log_likelihoods[-1]):.1f}")
print(f"  Improvement: {float(log_likelihoods[-1] - log_likelihoods[0]):.1f}")

# %%
# Plot log-likelihood convergence
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, "b.-", markersize=8)
ax.set_xlabel("EM Iteration")
ax.set_ylabel("Marginal Log-Likelihood")
ax.set_title("EM Convergence")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("theta_em_convergence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Evaluate Discrete State Inference
#
# The model should identify theta-on vs theta-off periods with high accuracy.

# %%
# Get smoothed discrete state probabilities
smoother_discrete_prob = model.smoother_discrete_state_prob
inferred_states = jnp.argmax(smoother_discrete_prob, axis=1)

# Handle label switching (discrete states are only identifiable up to permutation)
acc_direct = jnp.mean(inferred_states == true_discrete_states)
acc_flipped = jnp.mean(inferred_states == (1 - true_discrete_states))

# Track whether we need to permute state indices for parameter comparison
state_perm = np.array([0, 1])
if acc_flipped > acc_direct:
    print("Note: Labels flipped; relabeling inferred states.")
    inferred_states = 1 - inferred_states
    smoother_discrete_prob = smoother_discrete_prob[:, ::-1]
    state_perm = np.array([1, 0])  # Flip state order for parameter alignment
    accuracy = float(acc_flipped)
else:
    accuracy = float(acc_direct)

print(f"Discrete state accuracy (theta-on vs off): {accuracy:.1%}")

# %%
# Plot discrete state inference
fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

# Panel 1: True discrete state
ax = axes[0]
ax.fill_between(
    time, true_discrete_states, alpha=0.7, step="mid", color="C0", label="True"
)
ax.set_ylabel("True State")
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Theta-off", "Theta-on"])
ax.legend(loc="upper right")
ax.set_title(f"Discrete State Inference: Theta-On vs Theta-Off (Accuracy: {accuracy:.1%})")

# Panel 2: Inferred state probability
ax = axes[1]
ax.fill_between(
    time,
    smoother_discrete_prob[:, 1],
    alpha=0.7,
    color="C1",
    label="P(Theta-on | spikes)",
)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("P(Theta-on)")
ax.set_ylim(-0.1, 1.1)
ax.legend(loc="upper right")

# Panel 3: Comparison overlay
ax = axes[2]
ax.fill_between(
    time, true_discrete_states, alpha=0.4, step="mid", color="C0", label="True state"
)
ax.plot(time, smoother_discrete_prob[:, 1], "C1-", alpha=0.8, label="P(Theta-on)")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("State")
ax.set_xlabel("Time (s)")
ax.set_ylim(-0.1, 1.1)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("theta_discrete_state_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Evaluate Latent Theta Trajectory
#
# Compare the true vs inferred theta amplitude and phase.

# %%
# Compute marginalized smoother mean (weighted average over discrete states)
smoother_mean = jnp.einsum(
    "tls,ts->tl", model.smoother_state_cond_mean, smoother_discrete_prob
)

# Handle oscillator sign ambiguity (x -> -x is a symmetry of the latent representation)
# Check if we need to flip the sign based on correlation with true states
raw_correlations = []
for dim in range(n_latent):
    corr = float(jnp.corrcoef(true_states[:, dim], smoother_mean[:, dim])[0, 1])
    raw_correlations.append(corr)

# If mean correlation is negative, flip the oscillator sign
mean_raw_corr = np.mean(raw_correlations)
sign_flip = False
if mean_raw_corr < 0:
    print("Note: Oscillator sign flipped (x -> -x symmetry); correcting for comparison.")
    smoother_mean = -smoother_mean
    sign_flip = True

# Recompute correlations after potential sign flip
correlations = []
for dim in range(n_latent):
    corr = float(jnp.corrcoef(true_states[:, dim], smoother_mean[:, dim])[0, 1])
    correlations.append(corr)

print("Latent theta correlations:")
for dim in range(n_latent):
    print(f"  Dimension {dim + 1}: {correlations[dim]:.3f}")
print(f"  Mean correlation: {np.mean(correlations):.3f}")

# %%
# Compute inferred amplitude
inf_amp = np.linalg.norm(np.array(smoother_mean), axis=1)

# Amplitude correlation
amp_corr = float(jnp.corrcoef(true_amp, inf_amp)[0, 1])
print(f"\nAmplitude correlation: {amp_corr:.3f}")

# %%
# Plot true vs inferred amplitude and state
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Discrete states
ax = axes[0]
ax.fill_between(
    time,
    true_discrete_states,
    step="mid",
    alpha=0.6,
    color="C0",
    label="True (theta-on=1)",
)
ax.plot(
    time, smoother_discrete_prob[:, 1], "k-", alpha=0.8, label="P(theta-on)"
)
ax.set_ylabel("State / P(theta-on)")
ax.set_ylim(-0.1, 1.1)
ax.legend(loc="upper right")
ax.set_title("Theta State and Amplitude Recovery")

# Theta amplitude
ax = axes[1]
ax.plot(time, true_amp, "C0-", alpha=0.7, linewidth=0.8, label="True amplitude")
ax.plot(time, inf_amp, "k-", alpha=0.7, linewidth=0.8, label="Inferred amplitude")
ax.set_ylabel("Theta Amplitude")
ax.legend(loc="upper right")

# Spike raster (subset)
ax = axes[2]
n_show = 10
for n in range(n_show):
    spike_times_plot = time[np.array(spikes[:, n]) > 0]
    ax.eventplot(
        [spike_times_plot],
        lineoffsets=n,
        linelengths=0.8,
        linewidths=0.5,
        colors="black",
    )
ax.set_ylabel("Neuron")
ax.set_xlabel("Time (s)")
ax.set_title("Spike Raster (first 10 neurons)")
ax.set_ylim(-0.5, n_show - 0.5)

plt.tight_layout()
plt.savefig("theta_amplitude_recovery.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Visualize Phase Recovery
#
# In theta-on periods, the model should recover the theta phase trajectory.

# %%
# Compute inferred phase (sign flip already applied to smoother_mean if needed)
inf_phase = np.arctan2(np.array(smoother_mean[:, 1]), np.array(smoother_mean[:, 0]))

# Phase alignment (only during theta-on periods with sufficient amplitude)
theta_on_mask = (true_discrete_states == 1) & (true_amp > 0.2)
if np.sum(theta_on_mask) > 0:
    # Use circular correlation (via cos/sin)
    phase_diff = inf_phase[theta_on_mask] - true_phase[theta_on_mask]
    phase_alignment = float(np.mean(np.cos(phase_diff)))
    print(f"Phase alignment during theta-on (mean cos(error)): {phase_alignment:.3f}")
    print(f"  (1.0 = perfect alignment, 0.0 = random)")
    if sign_flip:
        print("  (Note: This is after correcting for the x -> -x sign flip)")

# %%
# Plot phase comparison during a theta-on period
if len(theta_on_times) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Use the same zoomed window as before
    zoom_inf_phase = inf_phase[zoom_start:zoom_end]
    zoom_true_phase = true_phase[zoom_start:zoom_end]

    # Phase trajectories
    ax = axes[0]
    ax.plot(zoom_time, zoom_true_phase, "C0-", alpha=0.8, label="True phase")
    ax.plot(zoom_time, zoom_inf_phase, "k--", alpha=0.8, label="Inferred phase")
    ax.set_ylabel("Theta Phase")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.legend(loc="upper right")
    ax.set_title("Phase Recovery During Theta-On Period")

    # Phase error
    ax = axes[1]
    phase_error = np.angle(np.exp(1j * (zoom_inf_phase - zoom_true_phase)))
    ax.plot(zoom_time, phase_error, "C2-", alpha=0.8)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Phase Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

    plt.tight_layout()
    plt.savefig("theta_phase_recovery.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 11. Learned Parameters
#
# Examine how well the model recovered the true parameters.

# %%
# Get learned parameters and align with true state labels
learned_A_raw = model.continuous_transition_matrix
learned_Q_raw = model.process_cov

# Permute learned parameters to match true state labels
learned_A = learned_A_raw[:, :, state_perm]
learned_Q = learned_Q_raw[:, :, state_perm]

if not np.array_equal(state_perm, [0, 1]):
    print("Note: Learned parameters permuted to align with true state labels.\n")

print("Learned vs True Spectral Radius:")
for state, name in enumerate(["theta-off", "theta-on"]):
    true_sr = float(jnp.max(jnp.abs(jnp.linalg.eigvals(transition_matrices[:, :, state]))))
    learned_sr = float(jnp.max(jnp.abs(jnp.linalg.eigvals(learned_A[:, :, state]))))
    print(f"  State {state} ({name}):")
    print(f"    True: {true_sr:.4f}")
    print(f"    Learned: {learned_sr:.4f}")

print("\nLearned vs True Process Variance:")
for state, name in enumerate(["theta-off", "theta-on"]):
    true_var = float(process_covs[0, 0, state])
    learned_var = float(learned_Q[0, 0, state])
    print(f"  State {state} ({name}):")
    print(f"    True: {true_var:.4f}")
    print(f"    Learned: {learned_var:.4f}")

# %%
# Plot learned transition matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

vmin, vmax = -1, 1

ax = axes[0, 0]
im = ax.imshow(transition_matrices[:, :, 0], cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("True A (Theta-off)")
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
im = ax.imshow(transition_matrices[:, :, 1], cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("True A (Theta-on)")
plt.colorbar(im, ax=ax)

ax = axes[1, 0]
im = ax.imshow(learned_A[:, :, 0], cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("Learned A (Theta-off)")
plt.colorbar(im, ax=ax)

ax = axes[1, 1]
im = ax.imshow(learned_A[:, :, 1], cmap="RdBu_r", vmin=vmin, vmax=vmax)
ax.set_title("Learned A (Theta-on)")
plt.colorbar(im, ax=ax)

plt.suptitle("Transition Matrices: True vs Learned", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("theta_transition_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 12. Summary
#
# This demo showed that the `SwitchingSpikeOscillatorModel` can:
#
# 1. **Identify brain states**: Accurately detect when theta is on vs off from spikes alone
# 2. **Recover oscillator trajectory**: Track theta amplitude and phase over time
# 3. **Learn state-specific dynamics**: Discover the different damping/noise in each state
#
# **Why this matters for hippocampal analysis**:
# - The model quantifies and segments theta vs non-theta epochs automatically
# - It provides continuous estimates of theta phase for spike phase analysis
# - The discrete state probability gives confidence about regime transitions
# - The learned parameters reveal dynamical differences between states

# %%
# Final summary
print("=" * 60)
print("Summary: Theta-On vs Theta-Off Demo")
print("=" * 60)
print(f"  Data: {n_time} timesteps ({n_time * dt:.0f} s), {n_neurons} neurons")
print(f"  EM iterations: {len(log_likelihoods)}")
print(f"  Final log-likelihood: {float(log_likelihoods[-1]):.1f}")
print(f"  Discrete state accuracy: {accuracy:.1%}")
print(f"  Latent correlation (mean): {np.mean(correlations):.3f}")
print(f"  Amplitude correlation: {amp_corr:.3f}")
if np.sum(theta_on_mask) > 0:
    print(f"  Phase alignment (theta-on): {phase_alignment:.3f}")
print("=" * 60)

# %% [markdown]
# ## 13. Effect of L2 Regularization on Amplitude Estimation
#
# The model tends to underestimate amplitude. Let's compare different L2 regularization
# strengths to see if we can improve amplitude recovery.
#
# Lower regularization allows larger weights, which may help recover larger amplitudes.

# %%
# Compare different L2 regularization strengths
l2_values = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

print("Comparing L2 regularization strengths on amplitude estimation...")
print("-" * 70)

results = []
for l2_val in l2_values:
    # Create and fit model with this L2 value
    model_test = SwitchingSpikeOscillatorModel(
        n_oscillators=n_oscillators,
        n_neurons=n_neurons,
        n_discrete_states=n_discrete_states,
        sampling_freq=sampling_freq,
        dt=dt,
        spike_weight_l2=l2_val,
    )

    key, key_test = jax.random.split(key)
    _ = model_test.fit(spikes=spikes, max_iter=30, tol=1e-4, key=key_test)

    # Get smoother results and handle label switching
    test_discrete_prob = model_test.smoother_discrete_state_prob
    test_inferred = jnp.argmax(test_discrete_prob, axis=1)

    acc_d = float(jnp.mean(test_inferred == true_discrete_states))
    acc_f = float(jnp.mean(test_inferred == (1 - true_discrete_states)))

    if acc_f > acc_d:
        test_discrete_prob = test_discrete_prob[:, ::-1]
        test_accuracy = acc_f
    else:
        test_accuracy = acc_d

    # Compute marginalized smoother mean
    test_smoother_mean = jnp.einsum(
        "tls,ts->tl", model_test.smoother_state_cond_mean, test_discrete_prob
    )

    # Handle sign flip
    test_corrs = [
        float(jnp.corrcoef(true_states[:, d], test_smoother_mean[:, d])[0, 1])
        for d in range(n_latent)
    ]
    test_corr = np.mean(test_corrs)
    if test_corr < 0:
        test_smoother_mean = -test_smoother_mean

    # Compute amplitude metrics
    test_inf_amp = np.linalg.norm(np.array(test_smoother_mean), axis=1)
    test_amp_corr = float(jnp.corrcoef(true_amp, test_inf_amp)[0, 1])

    # Amplitude ratio (inferred / true) - want this close to 1.0
    # Only compute during theta-on when amplitude is meaningful
    theta_on_idx = np.array(true_discrete_states) == 1
    if np.sum(theta_on_idx) > 0:
        mean_true_amp = float(np.mean(true_amp[theta_on_idx]))
        mean_inf_amp = float(np.mean(test_inf_amp[theta_on_idx]))
        amp_ratio = mean_inf_amp / mean_true_amp
    else:
        amp_ratio = np.nan

    # Weight magnitude
    weight_norm = float(jnp.mean(jnp.linalg.norm(model_test.spike_params.weights, axis=1)))

    results.append({
        'l2': l2_val,
        'accuracy': test_accuracy,
        'amp_corr': test_amp_corr,
        'amp_ratio': amp_ratio,
        'weight_norm': weight_norm,
    })

    print(f"L2={l2_val:.2f}: Acc={test_accuracy:.1%}, "
          f"AmpCorr={test_amp_corr:.3f}, AmpRatio={amp_ratio:.3f}, "
          f"|weights|={weight_norm:.3f}")

print("-" * 70)

# %%
# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

l2_vals = [r['l2'] for r in results]

# Accuracy vs L2
ax = axes[0, 0]
ax.plot(l2_vals, [r['accuracy'] for r in results], 'bo-', markersize=8)
ax.set_xlabel("L2 Regularization")
ax.set_ylabel("Discrete State Accuracy")
ax.set_xscale('symlog', linthresh=0.01)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(0.8, 1.02)
ax.set_title("State Classification Accuracy")

# Amplitude correlation vs L2
ax = axes[0, 1]
ax.plot(l2_vals, [r['amp_corr'] for r in results], 'go-', markersize=8)
ax.set_xlabel("L2 Regularization")
ax.set_ylabel("Amplitude Correlation")
ax.set_xscale('symlog', linthresh=0.01)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(0.8, 1.02)
ax.set_title("Amplitude Tracking (correlation)")

# Amplitude ratio vs L2
ax = axes[1, 0]
ax.plot(l2_vals, [r['amp_ratio'] for r in results], 'ro-', markersize=8)
ax.set_xlabel("L2 Regularization")
ax.set_ylabel("Amplitude Ratio (inferred / true)")
ax.set_xscale('symlog', linthresh=0.01)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect recovery')
ax.set_ylim(0, 1.5)
ax.set_title("Amplitude Scale Recovery")
ax.legend()

# Weight magnitude vs L2
ax = axes[1, 1]
ax.plot(l2_vals, [r['weight_norm'] for r in results], 'mo-', markersize=8)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='True |c|=1.0')
ax.set_xlabel("L2 Regularization")
ax.set_ylabel("Mean Weight Magnitude")
ax.set_xscale('symlog', linthresh=0.01)
ax.set_title("Learned Weight Magnitude")
ax.legend()

plt.suptitle("Effect of L2 Regularization on Model Performance", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("theta_l2_regularization_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Summary of L2 comparison
print("\nL2 Regularization Comparison Summary:")
print("=" * 70)
print(f"{'L2':<8} {'Accuracy':<10} {'Amp Corr':<10} {'Amp Ratio':<12} {'|weights|':<10}")
print("-" * 70)
for r in results:
    print(f"{r['l2']:<8.2f} {r['accuracy']:<10.1%} {r['amp_corr']:<10.3f} "
          f"{r['amp_ratio']:<12.3f} {r['weight_norm']:<10.3f}")
print("=" * 70)
print("\nNote: Amp Ratio < 1 means underestimating amplitude")
print("      True weight magnitude |c| = 1.0")
