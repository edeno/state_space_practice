"""Focused theta segmentation on full CA1 dataset.

Key insight from data analysis:
- Theta ACF at 8 Hz lag = 0.21 during running (clearly present)
- Rates are very sparse (median 1.2 Hz running, 0.4 Hz immobile)
- Need strong damping contrast + adequate Q for oscillator to matter
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle

jax.config.update("jax_enable_x64", True)

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
)
from state_space_practice.switching_point_process import (
    QRegularizationConfig,
    SpikeObsParams,
    SwitchingSpikeOscillatorModel,
)

if Path.cwd().name == "notebooks":
    project_root = Path.cwd().parent
else:
    project_root = Path.cwd()
output_dir = project_root / "output" / "real_ca1"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
with open("data/ca1_preprocessed_250Hz.pkl", "rb") as f:
    data = pickle.load(f)

spikes = jnp.array(data["binned_spikes"])
speed = data["speed"]
labels = data["behavioral_labels"]
time_bins = data["time_bins"]
sampling_freq = data["sampling_freq"]
dt = data["dt"]
n_neurons = data["n_neurons"]
n_time = data["n_time"]

print(f"Data: {n_time} steps ({n_time*dt:.0f}s), {n_neurons} neurons, {sampling_freq} Hz")

# %% [markdown]
# # Model setup: strong theta contrast
#
# State 0 (theta-off): heavily damped, oscillation dies within ~2 cycles
# State 1 (theta-on): nearly undamped, sustained theta oscillation
#
# Q large enough that the oscillator has meaningful amplitude.

# %%
theta_freq = jnp.array([8.0])

# Strong damping contrast
A_off = construct_common_oscillator_transition_matrix(
    theta_freq, jnp.array([0.95]), sampling_freq  # damped
)
A_on = construct_common_oscillator_transition_matrix(
    theta_freq, jnp.array([0.99]), sampling_freq  # sustained
)

# Process noise — keep small to avoid large covariance growth
Q_off = construct_common_oscillator_process_covariance(jnp.array([0.01]))
Q_on = construct_common_oscillator_process_covariance(jnp.array([0.02]))

print(f"A_off spectral radius: {float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_off)))):.3f}")
print(f"A_on spectral radius:  {float(jnp.max(jnp.abs(jnp.linalg.eigvals(A_on)))):.3f}")

# Empirical transition matrix from behavior
valid = labels[labels != 2]
n_im = np.sum(valid[:-1] == 0)
n_run = np.sum(valid[:-1] == 1)
p_im_stay = (np.sum((valid[:-1] == 0) & (valid[1:] == 0)) + 1) / (n_im + 2)
p_run_stay = (np.sum((valid[:-1] == 1) & (valid[1:] == 1)) + 1) / (n_run + 2)
running_frac = float((labels == 1).mean())

print(f"Empirical: P(stay|immobile)={p_im_stay:.4f}, P(stay|run)={p_run_stay:.4f}")

# Data-adaptive baseline
mean_counts = np.array(jnp.mean(spikes, axis=0))
empirical_baseline = np.log(mean_counts / dt + 1e-10)

# %%
model = SwitchingSpikeOscillatorModel(
    n_oscillators=1,
    n_neurons=n_neurons,
    n_discrete_states=2,
    sampling_freq=sampling_freq,
    dt=dt,
    q_regularization=QRegularizationConfig(),
    separate_spike_params=True,
    spike_weight_l2=0.05,
    update_continuous_transition_matrix=False,
    update_process_cov=False,
    update_init_mean=False,
    update_init_cov=False,
    # Fix Z to empirical behavioral transitions — theta is a sustained state
    update_discrete_transition_matrix=False,
)

key = jax.random.PRNGKey(42)
model._initialize_parameters(key)
model.continuous_transition_matrix = jnp.stack([A_off, A_on], axis=-1)
model.process_cov = jnp.stack([Q_off, Q_on], axis=-1)
model.init_cov = jnp.stack([jnp.eye(2) * 0.5, jnp.eye(2) * 0.5], axis=-1)

model.discrete_transition_matrix = jnp.array([
    [p_im_stay, 1 - p_im_stay],
    [1 - p_run_stay, p_run_stay],
])
model.init_discrete_state_prob = jnp.array([1 - running_frac, running_frac])

# Initialize spike params from data
model.spike_params = SpikeObsParams(
    baseline=jnp.stack([
        jnp.array(empirical_baseline),
        jnp.array(empirical_baseline),
    ], axis=-1),
    weights=jax.random.normal(key, (n_neurons, 2, 2)) * 0.01,
)

print(f"\nFitting on full dataset ({n_time} steps, {n_neurons} neurons)...")
lls = model.fit(spikes, max_iter=20, skip_init=True)
print(f"Done. {len(lls)} iterations.")
print(f"LL: {lls[0]:.0f} -> {lls[-1]:.0f} (delta={lls[-1]-lls[0]:.0f})")

# %%
prob = np.array(model.smoother_discrete_state_prob)
running_mask = (labels == 1).astype(float)

# Identify theta-on state
corr = [np.corrcoef(running_mask, prob[:, j])[0, 1] for j in range(2)]
theta_on = np.argmax(corr)
theta_off = 1 - theta_on

clear_mask = (labels == 0) | (labels == 1)
auc = roc_auc_score(running_mask[clear_mask], prob[clear_mask, theta_on])

# Smoother
smoother_mean = np.array(jnp.einsum(
    "tls,ts->tl", model.smoother_state_cond_mean, model.smoother_discrete_state_prob,
))
amplitude = np.sqrt(smoother_mean[:, 0] ** 2 + smoother_mean[:, 1] ** 2)
phase = np.arctan2(smoother_mean[:, 1], smoother_mean[:, 0])

print(f"\nTheta-on = state {theta_on}")
print(f"AUC vs running: {auc:.3f}")
print(f"Corr with running: {corr[theta_on]:.3f}")
print(f"Theta-on occupancy: {prob[:, theta_on].mean():.3f}")
print(f"Z:\n{np.array(model.discrete_transition_matrix)}")

# Spectral properties
for j in range(2):
    eigs = np.linalg.eigvals(np.array(model.continuous_transition_matrix[:, :, j]))
    sr = np.max(np.abs(eigs))
    freq = np.abs(np.angle(eigs[0])) * sampling_freq / (2 * np.pi)
    name = "theta-off" if j == theta_off else "theta-on"
    print(f"  State {j} ({name}): sr={sr:.4f}, freq={freq:.1f} Hz")

# %%
# === Plot 1: Full overview (first 2 minutes) ===
t_show = slice(0, int(120 * sampling_freq))
time_show = time_bins[t_show] - time_bins[0]

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

ax = axes[0]
ax.plot(time_show, speed[t_show], color="gray", alpha=0.5, linewidth=0.5)
ax.axhline(5.0, color="red", linestyle="--", alpha=0.3, label="5 cm/s")
ax.set_ylabel("Speed (cm/s)")
ax.set_title(f"CA1 theta segmentation — full dataset ({n_time*dt:.0f}s, AUC={auc:.3f})")
ax.legend(fontsize=8)

ax = axes[1]
ax.fill_between(time_show, 0, running_mask[t_show], alpha=0.15, color="C0", label="Running")
ax.plot(time_show, prob[t_show, theta_on], color="C1", alpha=0.8, linewidth=0.5,
        label="P(theta-on)")
ax.set_ylabel("P(theta-on)")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(time_show, smoother_mean[t_show, 0], color="C1", alpha=0.8, linewidth=0.3)
ax.set_ylabel("Oscillator x[0]")
ax.set_title("Inferred theta oscillation")

ax = axes[3]
ax.plot(time_show, amplitude[t_show], color="C2", alpha=0.8, linewidth=0.5)
ax.set_ylabel("Amplitude")
ax.set_title("Theta amplitude")

ax = axes[4]
# Spike raster (10 highest-rate neurons)
top_neurons = np.argsort(mean_counts)[-10:]
for i, n in enumerate(top_neurons):
    spike_times = np.where(np.array(spikes[t_show, n]) > 0)[0] * dt
    ax.scatter(spike_times, np.ones_like(spike_times) * i, s=0.3, c="k", alpha=0.3)
ax.set_ylabel("Neuron")
ax.set_xlabel("Time (seconds)")
ax.set_yticks(range(10))

plt.tight_layout()
plt.savefig(output_dir / "theta_focused_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# === Plot 2: Zoomed view showing theta onset relative to movement ===
# Find a transition from immobility to running
transition_t = None
for t in range(1000, n_time - 2000):
    if labels[t - 1] == 0 and labels[t] == 1 and speed[t] > 5:
        if (labels[max(0, t - 500):t] == 0).mean() > 0.5 and (labels[t:min(n_time, t + 500)] == 1).mean() > 0.5:
            transition_t = t
            break

if transition_t is None:
    # Fallback: find any speed crossing
    for t in range(1000, n_time - 2000):
        if speed[t - 1] < 3 and speed[t] > 7:
            transition_t = t
            break

if transition_t is None:
    transition_t = 5000  # fallback

zoom = slice(transition_t - int(3 * sampling_freq), transition_t + int(5 * sampling_freq))
time_zoom = (np.arange(zoom.start, zoom.stop) - transition_t) * dt  # centered on transition

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

ax = axes[0]
ax.plot(time_zoom, speed[zoom], color="gray", alpha=0.8)
ax.axhline(5.0, color="red", linestyle="--", alpha=0.3)
ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="Movement onset")
ax.set_ylabel("Speed (cm/s)")
ax.set_title("Zoomed: theta onset relative to movement")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(time_zoom, prob[zoom, theta_on], color="C1", alpha=0.8)
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_ylabel("P(theta-on)")
ax.set_ylim(-0.05, 1.05)

ax = axes[2]
ax.plot(time_zoom, smoother_mean[zoom, 0], color="C1", alpha=0.8)
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_ylabel("Oscillator")
ax.set_title("Theta oscillation around movement onset")

ax = axes[3]
ax.plot(time_zoom, amplitude[zoom], color="C2", alpha=0.8)
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_ylabel("Amplitude")
ax.set_xlabel("Time relative to movement onset (seconds)")

plt.tight_layout()
plt.savefig(output_dir / "theta_focused_onset.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# === Plot 3: Movement-triggered average of theta probability ===
# Find transitions using smoothed speed
speed_smooth = np.convolve(speed, np.ones(50) / 50, mode="same")
transitions = []
for t in range(750, n_time - 750):
    if speed_smooth[t - 1] < 4.0 and speed_smooth[t] >= 4.0:
        if len(transitions) == 0 or t - transitions[-1] > 750:
            transitions.append(t)

print(f"Found {len(transitions)} speed transitions")

# Average theta probability around transitions
window = int(3 * sampling_freq)  # 3 seconds each side
aligned_prob = []
aligned_speed = []
for t in transitions:
    if t - window >= 0 and t + window <= n_time:
        p = prob[t - window:t + window, theta_on]
        s = speed[t - window:t + window]
        if len(p) == 2 * window and len(s) == 2 * window:
            aligned_prob.append(p)
            aligned_speed.append(s)

if len(aligned_prob) == 0:
    print("WARNING: No valid transitions found for triggered average")
    aligned_prob = np.zeros((1, 2 * window))
    aligned_speed = np.zeros((1, 2 * window))

aligned_prob = np.array(aligned_prob)
aligned_speed = np.array(aligned_speed)
time_aligned = (np.arange(-window, window)) * dt

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax = axes[0]
ax.plot(time_aligned, aligned_speed.mean(axis=0), color="gray")
ax.fill_between(time_aligned,
                aligned_speed.mean(axis=0) - aligned_speed.std(axis=0),
                aligned_speed.mean(axis=0) + aligned_speed.std(axis=0),
                alpha=0.2, color="gray")
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.axhline(5.0, color="red", linestyle="--", alpha=0.3)
ax.set_ylabel("Speed (cm/s)")
ax.set_title(f"Movement-triggered average (n={len(aligned_prob)} transitions)")

ax = axes[1]
ax.plot(time_aligned, aligned_prob.mean(axis=0), color="C1")
ax.fill_between(time_aligned,
                aligned_prob.mean(axis=0) - aligned_prob.std(axis=0) / np.sqrt(len(aligned_prob)),
                aligned_prob.mean(axis=0) + aligned_prob.std(axis=0) / np.sqrt(len(aligned_prob)),
                alpha=0.3, color="C1")
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_ylabel("P(theta-on)")
ax.set_xlabel("Time relative to movement onset (seconds)")

plt.tight_layout()
plt.savefig(output_dir / "theta_focused_triggered_avg.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# === Summary stats ===
# Check timing: does theta onset precede movement?
# For each transition, find when P(theta) first exceeds 0.5
onset_lags = []
for i, t in enumerate(transitions):
    if t - window >= 0 and t + window < n_time:
        p = prob[t - window:t + window, theta_on]
        # Find first crossing above 0.5 relative to transition
        above = np.where(p > 0.5)[0]
        if len(above) > 0:
            first_above = above[0] - window  # relative to transition
            onset_lags.append(first_above * dt)

onset_lags = np.array(onset_lags)
print(f"\nTheta onset timing (relative to movement):")
print(f"  Median: {np.median(onset_lags)*1000:.0f} ms")
print(f"  Mean: {np.mean(onset_lags)*1000:.0f} ms")
print(f"  Fraction before movement: {(onset_lags < 0).mean():.1%}")

# Save results
with open(output_dir / "theta_focused_results.pkl", "wb") as f:
    pickle.dump({
        "model": model, "lls": lls, "prob": prob,
        "theta_on": theta_on, "auc": auc,
        "smoother_mean": smoother_mean, "amplitude": amplitude, "phase": phase,
        "onset_lags": onset_lags,
    }, f)
print(f"\nResults saved to {output_dir / 'theta_focused_results.pkl'}")
