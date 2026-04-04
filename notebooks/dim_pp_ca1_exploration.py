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
# # DIM-PP on Real CA1 Data
#
# Fit a Directed Influence Model with point-process observations to
# hippocampal CA1 spike data from a rat on a plus maze bandit task.
#
# The goal: discover whether directed coupling between theta oscillators
# in CA1 changes across behavioral states (running vs immobility).
#
# ## Approach
# 1. Start simple: 1 oscillator (theta), 2 states — does the model
#    find running vs immobility from spikes alone?
# 2. Then: 2 oscillators (theta), 2 states — does coupling differ?

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle

jax.config.update("jax_enable_x64", True)

# %%
# Load preprocessed data
with open("data/ca1_preprocessed_250Hz.pkl", "rb") as f:
    data = pickle.load(f)

spikes_all = data["binned_spikes"]
speed = data["speed"]
labels = data["behavioral_labels"]  # 0=immobile, 1=running, 2=transition
sf = data["sampling_freq"]
dt = data["dt"]
time_bins = data["time_bins"]

print(f"Data: {spikes_all.shape[0]} steps, {spikes_all.shape[1]} neurons, {sf} Hz")
print(f"Duration: {spikes_all.shape[0] * dt:.0f}s")

# %% [markdown]
# ## Select neurons and subset data
#
# Use a subset of the data for initial exploration — full 354K steps
# is expensive for EM. Start with 5 minutes (~75K steps at 250Hz).
# Select higher-firing neurons for better signal.

# %%
# Select neurons with firing rate > 2 Hz
rates = spikes_all.sum(axis=0) / (spikes_all.shape[0] * dt)
good_neurons = rates > 2.0
n_good = good_neurons.sum()
print(f"Neurons > 2 Hz: {n_good} out of {spikes_all.shape[1]}")

spikes_selected = spikes_all[:, good_neurons]

# Use first 5 minutes
n_steps_5min = int(5 * 60 * sf)
spikes = jnp.array(spikes_selected[:n_steps_5min])
speed_sub = speed[:n_steps_5min]
labels_sub = labels[:n_steps_5min]

print(f"Working data: {spikes.shape[0]} steps, {spikes.shape[1]} neurons")
print(f"Total spikes: {int(jnp.sum(spikes))}")
print(f"Running fraction: {(labels_sub == 1).mean():.3f}")

# %% [markdown]
# ## Step 1: Single oscillator DIM-PP
#
# Start with 1 oscillator at theta (8 Hz), 2 discrete states.
# This tests whether the model finds meaningful state segmentation
# from the spike data, and whether it correlates with behavior.

# %%
from state_space_practice.point_process_models import DirectedInfluencePointProcessModel

n_osc = 1
n_disc = 2
n_neurons = spikes.shape[1]
theta_freq = jnp.array([8.0])
damping = jnp.array([0.95])
process_var = jnp.array([0.1])

# With 1 oscillator, there's no coupling to discover (coupling is
# between oscillators). So DIM with 1 oscillator is equivalent to
# COM — the only thing that can switch is the observation model.
# Let's use it anyway to see what the model finds.
model_1osc = DirectedInfluencePointProcessModel(
    n_oscillators=n_osc,
    n_neurons=n_neurons,
    n_discrete_states=n_disc,
    sampling_freq=sf,
    dt=dt,
    freqs=theta_freq,
    auto_regressive_coef=damping,
    process_variance=process_var,
    phase_difference=jnp.zeros((n_osc, n_osc, n_disc)),
    coupling_strength=jnp.zeros((n_osc, n_osc, n_disc)),
)

print(f"Model: {model_1osc}")
print(f"Fitting on {spikes.shape[0]} steps, {spikes.shape[1]} neurons...")

# %%
import time as time_mod

t0 = time_mod.time()
lls_1osc = model_1osc.fit(
    spikes,
    max_iter=30,
    key=jax.random.PRNGKey(42),
)
elapsed = time_mod.time() - t0
print(f"Fit time: {elapsed:.1f}s, iterations: {len(lls_1osc)}")

# %%
# Plot log-likelihood trajectory
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(lls_1osc, ".-")
ax.set_xlabel("EM iteration")
ax.set_ylabel("Log-likelihood")
ax.set_title("1-oscillator DIM-PP: EM convergence")
plt.tight_layout()
plt.show()

# %%
# Compare inferred states with behavioral labels
smoother_prob = np.array(model_1osc.smoother_discrete_state_prob)
inferred = np.argmax(smoother_prob, axis=1)

# Check correlation with running
from sklearn.metrics import balanced_accuracy_score

# Try both label assignments
valid = labels_sub != 2  # exclude transitions
for perm_label, perm_name in [("direct", {0: 0, 1: 1}), ("swapped", {0: 1, 1: 0})]:
    remapped = np.array([perm_name[s] for s in inferred[valid]])
    bacc = balanced_accuracy_score(labels_sub[valid], remapped)
    print(f"  {perm_label}: balanced accuracy = {bacc:.3f}")

# %%
# Plot state probabilities vs speed
fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
t = np.arange(len(speed_sub)) * dt

axes[0].plot(t, speed_sub, "k", alpha=0.5, lw=0.5)
axes[0].set_ylabel("Speed (cm/s)")
axes[0].set_title("1-oscillator DIM-PP: State segmentation vs behavior")

axes[1].plot(t, smoother_prob[:, 0], label="State 0", alpha=0.7)
axes[1].plot(t, smoother_prob[:, 1], label="State 1", alpha=0.7)
axes[1].set_ylabel("P(state)")
axes[1].legend()

axes[2].fill_between(t, 0, 1, where=labels_sub == 1, alpha=0.3, color="green", label="Running")
axes[2].fill_between(t, 0, 1, where=labels_sub == 0, alpha=0.3, color="red", label="Immobile")
axes[2].set_ylabel("Behavior")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2: Two oscillator DIM-PP
#
# Now use 2 theta oscillators. The DIM model can discover directed
# coupling between them, and whether that coupling changes across
# discrete states. This is the real test of the model.

# %%
n_osc_2 = 2
theta_freqs = jnp.array([8.0, 8.0])  # Two theta oscillators
damping_2 = jnp.array([0.95, 0.95])
process_var_2 = jnp.array([0.1, 0.1])

model_2osc = DirectedInfluencePointProcessModel(
    n_oscillators=n_osc_2,
    n_neurons=n_neurons,
    n_discrete_states=n_disc,
    sampling_freq=sf,
    dt=dt,
    freqs=theta_freqs,
    auto_regressive_coef=damping_2,
    process_variance=process_var_2,
    phase_difference=jnp.zeros((n_osc_2, n_osc_2, n_disc)),
    coupling_strength=jnp.zeros((n_osc_2, n_osc_2, n_disc)),
)

print(f"Model: {model_2osc}")
print(f"Fitting on {spikes.shape[0]} steps, {spikes.shape[1]} neurons...")

# %%
t0 = time_mod.time()
lls_2osc = model_2osc.fit(
    spikes,
    max_iter=30,
    key=jax.random.PRNGKey(42),
)
elapsed = time_mod.time() - t0
print(f"Fit time: {elapsed:.1f}s, iterations: {len(lls_2osc)}")

# %%
# Plot convergence
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(lls_2osc, ".-")
ax.set_xlabel("EM iteration")
ax.set_ylabel("Log-likelihood")
ax.set_title("2-oscillator DIM-PP: EM convergence")
plt.tight_layout()
plt.show()

# %%
# Check state segmentation vs behavior
smoother_prob_2 = np.array(model_2osc.smoother_discrete_state_prob)
inferred_2 = np.argmax(smoother_prob_2, axis=1)

valid = labels_sub != 2
for perm_label, perm_name in [("direct", {0: 0, 1: 1}), ("swapped", {0: 1, 1: 0})]:
    remapped = np.array([perm_name[s] for s in inferred_2[valid]])
    bacc = balanced_accuracy_score(labels_sub[valid], remapped)
    print(f"  {perm_label}: balanced accuracy = {bacc:.3f}")

# %%
# Plot state probabilities vs speed
fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

axes[0].plot(t, speed_sub, "k", alpha=0.5, lw=0.5)
axes[0].set_ylabel("Speed (cm/s)")
axes[0].set_title("2-oscillator DIM-PP: State segmentation vs behavior")

axes[1].plot(t, smoother_prob_2[:, 0], label="State 0", alpha=0.7)
axes[1].plot(t, smoother_prob_2[:, 1], label="State 1", alpha=0.7)
axes[1].set_ylabel("P(state)")
axes[1].legend()

axes[2].fill_between(t, 0, 1, where=labels_sub == 1, alpha=0.3, color="green", label="Running")
axes[2].fill_between(t, 0, 1, where=labels_sub == 0, alpha=0.3, color="red", label="Immobile")
axes[2].set_ylabel("Behavior")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

plt.tight_layout()
plt.show()

# %%
# Examine the recovered coupling parameters
from state_space_practice.oscillator_utils import extract_dim_params_from_matrix

for j in range(n_disc):
    A_j = model_2osc.continuous_transition_matrix[:, :, j]
    params = extract_dim_params_from_matrix(A_j, sf, n_osc_2)
    print(f"\nState {j}:")
    print(f"  Frequencies: {np.array(params['freq'])} Hz")
    print(f"  Damping: {np.array(params['damping'])}")
    print(f"  Coupling strength:")
    print(f"    osc1 -> osc2: {float(params['coupling_strength'][1, 0]):.4f}")
    print(f"    osc2 -> osc1: {float(params['coupling_strength'][0, 1]):.4f}")
    print(f"  Phase difference:")
    print(f"    osc1 -> osc2: {float(params['phase_diff'][1, 0]):.4f} rad")
    print(f"    osc2 -> osc1: {float(params['phase_diff'][0, 1]):.4f} rad")

# %%
# Compare LL: 1 osc vs 2 osc
print(f"\nModel comparison:")
print(f"  1 oscillator: final LL = {lls_1osc[-1]:.0f}")
print(f"  2 oscillators: final LL = {lls_2osc[-1]:.0f}")
print(f"  Improvement: {lls_2osc[-1] - lls_1osc[-1]:.0f}")
