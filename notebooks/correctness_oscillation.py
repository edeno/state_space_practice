"""Correctness verification with oscillatory dynamics.

Demonstrates the switching point-process model on data with real oscillations:
- State 0: 8 Hz oscillation with heavy damping (amplitude decays fast)
- State 1: 8 Hz oscillation with light damping (sustained oscillation)
- Neurons are phase-locked to the oscillation via spike coupling weights
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from state_space_practice.switching_kalman import switching_kalman_smoother
from state_space_practice.switching_point_process import (
    QRegularizationConfig,
    SpikeObsParams,
    SwitchingSpikeOscillatorModel,
    switching_point_process_filter,
)

if Path.cwd().name == "notebooks":
    project_root = Path.cwd().parent
else:
    project_root = Path.cwd()
output_dir = project_root / "output" / "correctness"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Simulate oscillatory switching data
#
# Two states with the same frequency (8 Hz) but different damping:
# - State 0: damping = 0.85 (amplitude halves every ~4 cycles)
# - State 1: damping = 0.99 (nearly undamped, sustained oscillation)

# %%
n_time = 30000  # 5 minutes at 100 Hz — many oscillation cycles
n_neurons = 10
n_latent = 2  # one 2D oscillator
dt = 0.01
sampling_freq = 1.0 / dt

# Oscillator transition matrices: rotation with damping
# A = r * [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
freq_hz = 8.0
theta = 2 * np.pi * freq_hz * dt  # rotation angle per step

damping_0 = 0.85  # heavily damped
damping_1 = 0.99  # nearly undamped

def make_rotation_matrix(damping, theta):
    return damping * jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta),  jnp.cos(theta)],
    ])

A0 = make_rotation_matrix(damping_0, theta)
A1 = make_rotation_matrix(damping_1, theta)
A_true = jnp.stack([A0, A1], axis=-1)

Q_true = jnp.stack([jnp.eye(n_latent) * 0.05] * 2, axis=-1)

# Transition matrix: long blocks (~200 steps = 2 sec per block)
Z_true = jnp.array([[0.995, 0.005], [0.005, 0.995]])

# Spike weights: neurons have different preferred phases
key = jax.random.PRNGKey(42)
k_w, k_s = jax.random.split(key)
# Create weights as cos/sin at evenly spaced phases
preferred_phases = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
W_true = jnp.stack([
    jnp.cos(preferred_phases) * 0.5,
    jnp.sin(preferred_phases) * 0.5,
], axis=-1)  # (n_neurons, 2)

b_true = jnp.ones(n_neurons) * 2.5  # ~12 Hz baseline

print(f"Oscillation: {freq_hz} Hz, {1/(freq_hz*dt):.0f} steps/cycle")
print(f"Damping: state 0 = {damping_0}, state 1 = {damping_1}")
print(f"A0 eigenvalues: {np.array(jnp.linalg.eigvals(A0))}")
print(f"A1 eigenvalues: {np.array(jnp.linalg.eigvals(A1))}")
print(f"Total time: {n_time * dt:.0f} seconds = {n_time * dt * freq_hz:.0f} cycles")

# %%
# Simulate manually to have full control
key_disc, key_state, key_spike = jax.random.split(k_s, 3)

# Discrete states
disc_states = np.zeros(n_time, dtype=int)
disc_states[0] = 1  # start in oscillatory state
for t in range(1, n_time):
    key_disc, k = jax.random.split(key_disc)
    p_switch = float(Z_true[disc_states[t - 1], 1 - disc_states[t - 1]])
    disc_states[t] = disc_states[t - 1]
    if float(jax.random.uniform(k)) < p_switch:
        disc_states[t] = 1 - disc_states[t - 1]

# Continuous states
states = np.zeros((n_time, n_latent))
states[0] = [1.0, 0.0]  # start with unit amplitude
for t in range(1, n_time):
    key_state, k = jax.random.split(key_state)
    A_t = np.array(A_true[:, :, disc_states[t]])
    Q_t = np.array(Q_true[:, :, disc_states[t]])
    noise = np.array(jax.random.multivariate_normal(k, jnp.zeros(n_latent), jnp.array(Q_t)))
    states[t] = A_t @ states[t - 1] + noise

# Spikes
spikes = np.zeros((n_time, n_neurons))
for t in range(n_time):
    key_spike, k = jax.random.split(key_spike)
    log_rate = np.array(b_true) + np.array(W_true) @ states[t]
    rate = np.exp(log_rate) * dt
    spikes[t] = np.array(jax.random.poisson(k, jnp.array(rate)))

spikes = jnp.array(spikes)
disc_states_jnp = jnp.array(disc_states)

# Compute phase and amplitude
amplitude = np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2)
phase = np.arctan2(states[:, 1], states[:, 0])

n_transitions = np.sum(np.diff(disc_states) != 0)
print(f"\nSimulated:")
print(f"  Total spikes: {int(jnp.sum(spikes))}")
print(f"  Mean rate: {float(jnp.mean(spikes) / dt):.1f} Hz/neuron")
print(f"  State occupancy: {[f'{float((disc_states==j).mean()):.2f}' for j in range(2)]}")
print(f"  State transitions: {n_transitions}")
print(f"  Mean block length: {n_time / (n_transitions + 1):.0f} steps")

# %% [markdown]
# # Plot 1: Simulated data overview

# %%
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
time_sec = np.arange(n_time) * dt

# Show first 60 seconds (6000 steps)
t_show = slice(0, 6000)

# Panel 1: True discrete state
ax = axes[0]
ax.fill_between(time_sec[t_show], 0, disc_states[t_show], alpha=0.3, color="C1",
                label="State 1 (sustained)")
ax.fill_between(time_sec[t_show], 0, 1 - disc_states[t_show], alpha=0.3, color="C0",
                label="State 0 (damped)")
ax.set_ylabel("State")
ax.set_title("True discrete state")
ax.legend(loc="upper right")
ax.set_ylim(-0.05, 1.05)

# Panel 2: Oscillator trajectory
ax = axes[1]
ax.plot(time_sec[t_show], states[t_show, 0], alpha=0.8, label="x[0] (cos)")
ax.plot(time_sec[t_show], states[t_show, 1], alpha=0.8, label="x[1] (sin)")
ax.set_ylabel("Latent state")
ax.set_title("Oscillator trajectory (damped in state 0, sustained in state 1)")
ax.legend(loc="upper right")

# Panel 3: Amplitude
ax = axes[2]
ax.plot(time_sec[t_show], amplitude[t_show], color="C2", alpha=0.8)
ax.set_ylabel("Amplitude")
ax.set_title("Oscillation amplitude")

# Panel 4: Spike raster (subset of neurons)
ax = axes[3]
for n in range(min(6, n_neurons)):
    spike_times = np.where(np.array(spikes[t_show, n]) > 0)[0] * dt
    ax.scatter(spike_times, np.ones_like(spike_times) * n, s=1, c=f"C{n}", alpha=0.5)
ax.set_ylabel("Neuron")
ax.set_title("Spike raster (subset)")
ax.set_yticks(range(min(6, n_neurons)))

# Panel 5: Total spike count
ax = axes[4]
total_spikes = np.array(jnp.sum(spikes, axis=1))
smoothed = np.convolve(total_spikes, np.ones(25) / 25, mode="same")
ax.plot(time_sec[t_show], smoothed[t_show], color="gray", alpha=0.8)
ax.set_ylabel("Spike count\n(smoothed)")
ax.set_xlabel("Time (seconds)")
ax.set_title("Total spike count (25-step moving average)")

plt.tight_layout()
plt.savefig(output_dir / "oscillation_1_simulated_data.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Fit the model with warm start near true parameters

# %%
model = SwitchingSpikeOscillatorModel(
    n_oscillators=1,
    n_neurons=n_neurons,
    n_discrete_states=2,
    sampling_freq=sampling_freq,
    dt=dt,
    q_regularization=QRegularizationConfig(enabled=False),
    separate_spike_params=False,
)
model._initialize_parameters(jax.random.PRNGKey(0))

# Warm start near true params (5% perturbation)
k_perturb = jax.random.PRNGKey(99)
k1, k2, k3, k4 = jax.random.split(k_perturb, 4)
model.continuous_transition_matrix = A_true + jax.random.normal(k1, A_true.shape) * 0.02
model.process_cov = Q_true + jnp.abs(jax.random.normal(k2, Q_true.shape)) * 0.005
model.discrete_transition_matrix = Z_true
model.spike_params = SpikeObsParams(
    baseline=b_true + jax.random.normal(k3, b_true.shape) * 0.1,
    weights=W_true + jax.random.normal(k4, W_true.shape) * 0.05,
)

print("Fitting model...")
lls = model.fit(spikes, max_iter=30, skip_init=True)
print(f"Done. LL improved: {lls[-1] - lls[0]:.1f}")

# %% [markdown]
# # Plot 2: EM convergence and state recovery

# %%
prob = np.array(model.smoother_discrete_state_prob)
corr = [np.corrcoef(disc_states.astype(float), prob[:, j])[0, 1] for j in range(2)]
best_j = np.argmax(np.abs(corr))
sustained_label = best_j if corr[best_j] > 0 else 1 - best_j
best_corr = max(np.abs(corr))

# Recovered spectral properties
fitted_eigs = [np.array(jnp.linalg.eigvals(model.continuous_transition_matrix[:, :, j])) for j in range(2)]
fitted_sr = [np.max(np.abs(e)) for e in fitted_eigs]
fitted_freq = [np.abs(np.angle(e[0])) / (2 * np.pi * dt) for e in fitted_eigs]
fitted_damping = fitted_sr  # spectral radius = damping for rotation matrix

# Smoother mean
smoother_mean = np.array(jnp.einsum(
    "tls,ts->tl",
    model.smoother_state_cond_mean,
    model.smoother_discrete_state_prob,
))
smoother_amp = np.sqrt(smoother_mean[:, 0] ** 2 + smoother_mean[:, 1] ** 2)
smoother_phase = np.arctan2(smoother_mean[:, 1], smoother_mean[:, 0])

fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=False)

# Panel 1: EM convergence
ax = axes[0]
ax.plot(lls, "o-", markersize=3)
ax.set_ylabel("Log-likelihood")
ax.set_xlabel("EM iteration")
ax.set_title(f"EM convergence (improvement = {lls[-1] - lls[0]:.1f})")

# Panels 2-5 share x axis (time)
for ax in axes[1:]:
    ax.set_xlim(0, time_sec[999])

# Panel 2: Discrete state recovery
ax = axes[1]
ax.fill_between(time_sec[t_show], 0, disc_states[t_show], alpha=0.2, color="C1",
                label="True sustained state")
ax.plot(time_sec[t_show], prob[t_show, sustained_label], color="C0", alpha=0.8,
        label=f"P(sustained | y)")
ax.set_ylabel("P(sustained)")
ax.set_title(f"Discrete state recovery (|corr| = {best_corr:.3f})")
ax.legend(loc="upper right")
ax.set_ylim(-0.05, 1.05)

# Panel 3: True vs smoothed oscillator (x[0])
ax = axes[2]
ax.plot(time_sec[t_show], states[t_show, 0], alpha=0.4, label="True x[0]")
ax.plot(time_sec[t_show], smoother_mean[t_show, 0], alpha=0.8, label="Smoothed x[0]")
ax.set_ylabel("x[0]")
ax.set_title("Oscillator tracking (cos component)")
ax.legend(loc="upper right")

# Panel 4: Amplitude comparison
ax = axes[3]
ax.plot(time_sec[t_show], amplitude[t_show], alpha=0.4, label="True amplitude")
ax.plot(time_sec[t_show], smoother_amp[t_show], alpha=0.8, label="Smoothed amplitude")
ax.set_ylabel("Amplitude")
ax.set_title("Amplitude tracking")
ax.legend(loc="upper right")

# Panel 5: Parameter recovery summary
ax = axes[4]
Z_fit = model.discrete_transition_matrix

# Sort by damping to match states
order = np.argsort(fitted_damping)
text = (
    f"Oscillator frequency:\n"
    f"  State 0: true = {freq_hz:.1f} Hz, fitted = {fitted_freq[order[0]]:.1f} Hz\n"
    f"  State 1: true = {freq_hz:.1f} Hz, fitted = {fitted_freq[order[1]]:.1f} Hz\n\n"
    f"Damping (spectral radius):\n"
    f"  State 0: true = {damping_0:.3f}, fitted = {fitted_sr[order[0]]:.3f}\n"
    f"  State 1: true = {damping_1:.3f}, fitted = {fitted_sr[order[1]]:.3f}\n\n"
    f"Transition matrix diagonal:\n"
    f"  true = [{float(Z_true[0,0]):.3f}, {float(Z_true[1,1]):.3f}], "
    f"fitted = [{float(Z_fit[0,0]):.3f}, {float(Z_fit[1,1]):.3f}]\n\n"
    f"Spike params:\n"
    f"  Baseline error: {float(jnp.max(jnp.abs(model.spike_params.baseline - b_true))):.3f} "
    f"(true = {float(b_true[0]):.1f})\n"
    f"  Weights MAE: {float(jnp.mean(jnp.abs(model.spike_params.weights - W_true))):.3f}"
)
ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=11,
        verticalalignment="center", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_axis_off()
ax.set_title("Recovered parameters")

plt.tight_layout()
plt.savefig(output_dir / "oscillation_2_recovery.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Plot 3: Phase-locking of neurons to the oscillation

# %%
# Recovered spike weights define each neuron's preferred phase
fitted_weights = np.array(model.spike_params.weights)
fitted_preferred_phase = np.arctan2(fitted_weights[:, 1], fitted_weights[:, 0])
fitted_modulation = np.sqrt(fitted_weights[:, 0] ** 2 + fitted_weights[:, 1] ** 2)
true_preferred_phase = np.array(preferred_phases)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel 1: True vs fitted preferred phase
ax = axes[0]
ax.scatter(true_preferred_phase, fitted_preferred_phase, s=50, alpha=0.8)
# Handle wraparound: add diagonal lines
for offset in [-2 * np.pi, 0, 2 * np.pi]:
    ax.plot([-np.pi, np.pi], [-np.pi + offset, np.pi + offset],
            "k--", alpha=0.3)
ax.set_xlabel("True preferred phase (rad)")
ax.set_ylabel("Fitted preferred phase (rad)")
ax.set_title("Phase preference recovery")
ax.set_xlim(-0.5, 2 * np.pi + 0.5)
ax.set_ylim(-np.pi - 0.5, np.pi + 0.5)

# Panel 2: Polar plot of neuron phases
ax = axes[1]
ax = fig.add_subplot(132, projection="polar")
ax.scatter(true_preferred_phase, np.ones(n_neurons), c="C0", s=60, alpha=0.6,
           label="True", zorder=5)
ax.scatter(fitted_preferred_phase, np.ones(n_neurons) * 0.7, c="C1", s=60, alpha=0.6,
           label="Fitted", zorder=5)
ax.set_title("Neuron preferred phases", pad=15)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

# Panel 3: Weight magnitude
ax = axes[2]
true_modulation = np.sqrt(np.array(W_true[:, 0]) ** 2 + np.array(W_true[:, 1]) ** 2)
ax.scatter(true_modulation, fitted_modulation, s=50, alpha=0.8)
max_val = max(true_modulation.max(), fitted_modulation.max()) * 1.1
ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
ax.set_xlabel("True modulation strength")
ax.set_ylabel("Fitted modulation strength")
ax.set_title("Modulation strength recovery")

plt.tight_layout()
plt.savefig(output_dir / "oscillation_3_phase_locking.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Plot 4: Zoomed view showing oscillation cycles

# %%
# Find a stretch where we transition from damped to sustained
transition_idx = None
for t in range(100, n_time - 200):
    if disc_states[t - 1] == 0 and disc_states[t] == 1:
        transition_idx = t
        break

if transition_idx is None:
    transition_idx = 500  # fallback

zoom = slice(transition_idx - 50, transition_idx + 150)
time_zoom = np.arange(zoom.start, zoom.stop) * dt

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Panel 1: State
ax = axes[0]
ax.fill_between(time_zoom, 0, disc_states[zoom], alpha=0.3, color="C1")
ax.plot(time_zoom, prob[zoom, sustained_label], color="C0", alpha=0.8)
ax.axvline(transition_idx * dt, color="red", linestyle="--", alpha=0.5, label="Transition")
ax.set_ylabel("P(sustained)")
ax.set_title("Zoomed: state transition from damped to sustained")
ax.legend()

# Panel 2: True oscillator
ax = axes[1]
ax.plot(time_zoom, states[zoom, 0], alpha=0.6, label="True x[0]")
ax.plot(time_zoom, smoother_mean[zoom, 0], alpha=0.8, label="Smoothed x[0]")
ax.axvline(transition_idx * dt, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("x[0]")
ax.set_title("Oscillator: damped → sustained")
ax.legend()

# Panel 3: Amplitude
ax = axes[2]
ax.plot(time_zoom, amplitude[zoom], alpha=0.5, label="True")
ax.plot(time_zoom, smoother_amp[zoom], alpha=0.8, label="Smoothed")
ax.axvline(transition_idx * dt, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("Amplitude")
ax.set_title("Amplitude increases after transition to sustained state")
ax.legend()

# Panel 4: Spike raster
ax = axes[3]
for n in range(min(6, n_neurons)):
    spike_times_zoom = np.where(np.array(spikes[zoom, n]) > 0)[0]
    if len(spike_times_zoom) > 0:
        ax.scatter(
            time_zoom[spike_times_zoom],
            np.ones_like(spike_times_zoom) * n,
            s=3, c=f"C{n}", alpha=0.6,
        )
ax.axvline(transition_idx * dt, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("Neuron")
ax.set_xlabel("Time (seconds)")
ax.set_title("Spikes become more rhythmic in sustained state")
ax.set_yticks(range(min(6, n_neurons)))

plt.tight_layout()
plt.savefig(output_dir / "oscillation_4_zoomed.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Data: {n_time} steps ({n_time*dt:.0f} sec), {n_neurons} neurons, {freq_hz} Hz oscillation")
print(f"EM: {len(lls)} iterations, LL improvement = {lls[-1]-lls[0]:.1f}")
print(f"State recovery |corr|: {best_corr:.3f}")
print(f"Frequency recovery: {fitted_freq[order[0]]:.1f} / {fitted_freq[order[1]]:.1f} Hz (true = {freq_hz:.1f})")
print(f"Damping recovery: {fitted_sr[order[0]]:.3f} / {fitted_sr[order[1]]:.3f} (true = {damping_0}/{damping_1})")
print(f"Latent state tracking corr: {np.corrcoef(states[:,0], smoother_mean[:,0])[0,1]:.3f}")
