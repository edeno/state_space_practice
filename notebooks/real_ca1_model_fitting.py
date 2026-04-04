"""Fit switching point-process model to full CA1 dataset.

Progressive approach:
1. S=2, fixed A/Q, separate spike params — replicate existing result on full data
2. S=3, fixed A/Q, separate spike params — add SWR detection
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle

jax.config.update("jax_enable_x64", True)

from sklearn.metrics import roc_auc_score

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
# Load preprocessed data
with open("data/ca1_preprocessed_250Hz.pkl", "rb") as f:
    data = pickle.load(f)

spikes = jnp.array(data["binned_spikes"])
speed = data["speed"]
labels = data["behavioral_labels"]  # 0=immobility, 1=running
time_bins = data["time_bins"]
sampling_freq = data["sampling_freq"]
dt = data["dt"]
n_neurons = data["n_neurons"]
n_time = data["n_time"]

print(f"Full dataset: {n_time} steps ({n_time*dt:.0f} sec), {n_neurons} neurons")
print(f"Sampling: {sampling_freq} Hz, dt={dt} sec")
print(f"Running fraction: {(labels==1).mean():.2f}")
print(f"Mean firing rate: {float(jnp.mean(spikes)/dt):.1f} Hz/neuron")

# %% [markdown]
# # Step 1: 2-state model on full data
#
# Fix A and Q to theta-specific values (like the existing notebook).
# Only learn spike params and discrete transitions.
# This avoids the numerical instability from learning dynamics.

# %%
theta_freq = jnp.array([8.0])

# Two states with different damping
A_off = construct_common_oscillator_transition_matrix(
    theta_freq, jnp.array([0.90]), sampling_freq
)
A_on = construct_common_oscillator_transition_matrix(
    theta_freq, jnp.array([0.995]), sampling_freq
)
Q_off = construct_common_oscillator_process_covariance(jnp.array([0.01]))
Q_on = construct_common_oscillator_process_covariance(jnp.array([0.03]))

print("2-state model setup:")
print(f"  State 0 (theta-off): damping=0.90, Q_var=0.01")
print(f"  State 1 (theta-on):  damping=0.995, Q_var=0.03")

model2 = SwitchingSpikeOscillatorModel(
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
)

key = jax.random.PRNGKey(42)
model2._initialize_parameters(key)
model2.continuous_transition_matrix = jnp.stack([A_off, A_on], axis=-1)
model2.process_cov = jnp.stack([Q_off, Q_on], axis=-1)
model2.init_cov = jnp.stack([jnp.eye(2) * 0.5, jnp.eye(2) * 0.5], axis=-1)

# Empirical transition matrix from behavioral labels
running_frac = float((labels == 1).mean())
valid = labels[labels != 2]
n_im = np.sum(valid[:-1] == 0)
n_run = np.sum(valid[:-1] == 1)
p_im_stay = (np.sum((valid[:-1] == 0) & (valid[1:] == 0)) + 1) / (n_im + 2)
p_run_stay = (np.sum((valid[:-1] == 1) & (valid[1:] == 1)) + 1) / (n_run + 2)
model2.discrete_transition_matrix = jnp.array([
    [p_im_stay, 1 - p_im_stay],
    [1 - p_run_stay, p_run_stay],
])
model2.init_discrete_state_prob = jnp.array([1 - running_frac, running_frac])

# Data-adaptive spike param initialization
# Set baseline to empirical log-rate for each neuron (same for both states)
mean_counts = np.array(jnp.mean(spikes, axis=0))
empirical_baseline = np.log(mean_counts / dt + 1e-10)
model2.spike_params = SpikeObsParams(
    baseline=jnp.stack([jnp.array(empirical_baseline), jnp.array(empirical_baseline)], axis=-1),
    weights=jnp.zeros((n_neurons, 2, 2)) + jax.random.normal(key, (n_neurons, 2, 2)) * 0.01,
)

print(f"Empirical baseline range: [{empirical_baseline.min():.1f}, {empirical_baseline.max():.1f}]")
print(f"Transition: P(stay|immobile)={float(p_im_stay):.4f}, P(stay|run)={float(p_run_stay):.4f}")

print(f"\nFitting on full dataset ({n_time} steps)...")
lls2 = model2.fit(spikes, max_iter=20, skip_init=True)
print(f"Done. {len(lls2)} iterations.")
print(f"LL: {lls2[0]:.0f} -> {lls2[-1]:.0f}")

# %%
# Evaluate
prob2 = np.array(model2.smoother_discrete_state_prob)
running_mask = (labels == 1).astype(float)

corr2 = [np.corrcoef(running_mask, prob2[:, j])[0, 1] for j in range(2)]
theta_on_state = np.argmax(corr2)

clear_mask = (labels == 0) | (labels == 1)
auc2 = roc_auc_score(running_mask[clear_mask], prob2[clear_mask, theta_on_state])

smoother_mean2 = np.array(jnp.einsum(
    "tls,ts->tl", model2.smoother_state_cond_mean, model2.smoother_discrete_state_prob,
))
amplitude2 = np.sqrt(smoother_mean2[:, 0] ** 2 + smoother_mean2[:, 1] ** 2)

print(f"Theta-on = state {theta_on_state}")
print(f"AUC vs running: {auc2:.3f}")
print(f"Corr with running: {corr2[theta_on_state]:.3f}")
print(f"Z: {np.array(model2.discrete_transition_matrix)}")

# Save 2-state results
with open(output_dir / "model2_results.pkl", "wb") as f:
    pickle.dump({
        "model": model2, "lls": lls2, "prob": prob2,
        "theta_on_state": theta_on_state, "auc": auc2,
        "smoother_mean": smoother_mean2, "amplitude": amplitude2,
    }, f)

# %%
# Plot first 2 minutes
t_show = slice(0, int(120 * sampling_freq))
time_show = time_bins[t_show] - time_bins[0]

fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

ax = axes[0]
ax.plot(time_show, speed[t_show], color="gray", alpha=0.5, linewidth=0.5)
ax.axhline(5.0, color="red", linestyle="--", alpha=0.3, label="Run threshold")
ax.set_ylabel("Speed (cm/s)")
ax.set_title(f"2-state model on full CA1 data (AUC={auc2:.3f})")
ax.legend()

ax = axes[1]
ax.plot(time_show, prob2[t_show, theta_on_state], color="C1", alpha=0.8, linewidth=0.5)
ax.fill_between(time_show, 0, running_mask[t_show], alpha=0.15, color="C0", label="Running")
ax.set_ylabel("P(theta-on)")
ax.legend()

ax = axes[2]
ax.plot(time_show, smoother_mean2[t_show, 0], color="C1", alpha=0.8, linewidth=0.5)
ax.set_ylabel("Oscillator x[0]")
ax.set_title("Inferred theta oscillation")

ax = axes[3]
ax.plot(time_show, amplitude2[t_show], color="C2", alpha=0.8, linewidth=0.5)
ax.set_ylabel("Amplitude")
ax.set_xlabel("Time (seconds)")
ax.set_title("Inferred theta amplitude")

plt.tight_layout()
plt.savefig(output_dir / "real_ca1_2state.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Step 2: 3-state model — add SWR detection
#
# - State 0: theta-off (immobility, low rate, damped)
# - State 1: theta-on (running, elevated rate, sustained theta)
# - State 2: SWR (brief high-rate bursts during immobility, heavily damped)

# %%
print()
print("=" * 50)
print("3-state model: theta-off / theta-on / SWR")
print("=" * 50)

A_swr = construct_common_oscillator_transition_matrix(
    theta_freq, jnp.array([0.80]), sampling_freq  # moderately damped
)

model3 = SwitchingSpikeOscillatorModel(
    n_oscillators=1,
    n_neurons=n_neurons,
    n_discrete_states=3,
    sampling_freq=sampling_freq,
    dt=dt,
    q_regularization=QRegularizationConfig(),
    separate_spike_params=False,  # shared params for stability
    spike_weight_l2=0.05,
    update_continuous_transition_matrix=False,
    update_process_cov=False,
    update_init_mean=False,
    update_init_cov=False,
)

key3 = jax.random.PRNGKey(42)
model3._initialize_parameters(key3)
model3.continuous_transition_matrix = jnp.stack([A_off, A_on, A_swr], axis=-1)
model3.process_cov = jnp.stack([Q_off, Q_on, Q_off], axis=-1)
model3.init_cov = jnp.stack([jnp.eye(2) * 0.5] * 3, axis=-1)

# Transition: theta-off and theta-on are stable, SWR is brief
model3.discrete_transition_matrix = jnp.array([
    [p_im_stay - 0.001, 1 - p_im_stay, 0.001],  # theta-off: mostly stay, rare SWR
    [1 - p_run_stay, p_run_stay - 0.001, 0.001],  # theta-on: mostly stay, rare SWR
    [0.15,  0.05,  0.80],   # SWR: brief (~5 steps = 20ms), mostly back to off
])
model3.init_discrete_state_prob = jnp.array([1 - running_frac - 0.02, running_frac, 0.02])

# Data-adaptive spike params: shared across states
model3.spike_params = SpikeObsParams(
    baseline=jnp.array(empirical_baseline),
    weights=jnp.zeros((n_neurons, 2)) + jax.random.normal(key3, (n_neurons, 2)) * 0.01,
)

print(f"Fitting 3-state model on full dataset...")
lls3 = model3.fit(spikes, max_iter=20, skip_init=True)
print(f"Done. {len(lls3)} iterations.")
print(f"LL: {lls3[0]:.0f} -> {lls3[-1]:.0f}")

# %%
prob3 = np.array(model3.smoother_discrete_state_prob)

# Identify states by their properties
mean_rates = []
for j in range(3):
    # Weight of this state at each timestep
    w = prob3[:, j]
    # Weighted mean spike rate
    weighted_rate = float(np.sum(w[:, None] * np.array(spikes), axis=0).sum() /
                          (np.sum(w) * n_neurons * dt))
    mean_rates.append(weighted_rate)

# SWR state should have highest rate
swr_state = np.argmax(mean_rates)
# Theta-on should correlate with running
remaining = [j for j in range(3) if j != swr_state]
corr_remaining = [np.corrcoef(running_mask, prob3[:, j])[0, 1] for j in remaining]
theta_on_3 = remaining[np.argmax(corr_remaining)]
theta_off_3 = [j for j in range(3) if j not in [swr_state, theta_on_3]][0]

print(f"State mapping: theta-off={theta_off_3}, theta-on={theta_on_3}, SWR={swr_state}")
print(f"Mean rates: off={mean_rates[theta_off_3]:.1f}, on={mean_rates[theta_on_3]:.1f}, SWR={mean_rates[swr_state]:.1f} Hz")
print(f"State occupancy: off={prob3[:, theta_off_3].mean():.3f}, on={prob3[:, theta_on_3].mean():.3f}, SWR={prob3[:, swr_state].mean():.3f}")

auc3 = roc_auc_score(running_mask[clear_mask], prob3[clear_mask, theta_on_3])
print(f"AUC (theta-on vs running): {auc3:.3f}")
print(f"Z: {np.array(model3.discrete_transition_matrix)}")

# %%
smoother_mean3 = np.array(jnp.einsum(
    "tls,ts->tl", model3.smoother_state_cond_mean, model3.smoother_discrete_state_prob,
))
amplitude3 = np.sqrt(smoother_mean3[:, 0] ** 2 + smoother_mean3[:, 1] ** 2)

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

ax = axes[0]
ax.plot(time_show, speed[t_show], color="gray", alpha=0.5, linewidth=0.5)
ax.axhline(5.0, color="red", linestyle="--", alpha=0.3)
ax.set_ylabel("Speed (cm/s)")
ax.set_title(f"3-state model: theta-off / theta-on / SWR  (AUC={auc3:.3f})")

ax = axes[1]
ax.plot(time_show, prob3[t_show, theta_on_3], color="C1", alpha=0.8, linewidth=0.5, label="P(theta-on)")
ax.plot(time_show, prob3[t_show, swr_state], color="C3", alpha=0.8, linewidth=0.5, label="P(SWR)")
ax.fill_between(time_show, 0, running_mask[t_show], alpha=0.1, color="C0", label="Running")
ax.set_ylabel("Probability")
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(time_show, prob3[t_show, theta_off_3], color="C0", alpha=0.8, linewidth=0.5, label="P(theta-off)")
ax.set_ylabel("P(theta-off)")
ax.legend(fontsize=8)

ax = axes[3]
ax.plot(time_show, smoother_mean3[t_show, 0], color="C1", alpha=0.8, linewidth=0.5)
ax.set_ylabel("Oscillator x[0]")
ax.set_title("Inferred theta oscillation")

ax = axes[4]
ax.plot(time_show, amplitude3[t_show], color="C2", alpha=0.8, linewidth=0.5)
ax.set_ylabel("Amplitude")
ax.set_xlabel("Time (seconds)")

plt.tight_layout()
plt.savefig(output_dir / "real_ca1_3state.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
print()
print("=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"2-state AUC: {auc2:.3f}")
print(f"3-state AUC: {auc3:.3f}")
print(f"2-state final LL: {lls2[-1]:.0f}")
print(f"3-state final LL: {lls3[-1]:.0f}")
