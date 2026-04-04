"""Build up to a 3-state CA1 model: theta-off, theta-on, SWR.

Progressive complexity:
1. S=2: theta-off vs theta-on (damping only)
2. S=2: theta-off vs theta-on (damping + rate difference)
3. S=3: theta-off, theta-on, SWR (full model)

All simulations use realistic CA1-like parameters:
- Theta at ~8 Hz
- 20-50 neurons with heterogeneous firing rates
- Place-cell-like low baseline rates (1-5 Hz)
- Theta modulation via phase-locked weights
- SWR as brief high-synchrony bursts
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

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
output_dir = project_root / "output" / "correctness"
output_dir.mkdir(parents=True, exist_ok=True)

# === Shared simulation parameters ===
sampling_freq = 100.0  # Hz
dt = 1.0 / sampling_freq
freq_theta = 8.0
n_neurons = 20
n_latent = 2  # one 2D oscillator

# Neuron preferred phases (evenly spaced around the cycle)
preferred_phases = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
# Add some jitter to make it realistic
key_phase = jax.random.PRNGKey(0)
preferred_phases += np.array(jax.random.normal(key_phase, (n_neurons,))) * 0.3

# Theta modulation strength varies across neurons (some strongly modulated, some weakly)
key_mod = jax.random.PRNGKey(1)
modulation_strength = np.abs(np.array(jax.random.normal(key_mod, (n_neurons,)))) * 0.3 + 0.1


def simulate_switching_spikes(
    n_time, A_per_state, Q_per_state, Z, b_per_state, W_per_state,
    init_state_probs, seed=42,
):
    """Simulate spikes from a switching model with per-state spike params.

    Parameters
    ----------
    A_per_state : list of arrays, each (n_latent, n_latent)
    Q_per_state : list of arrays, each (n_latent, n_latent)
    Z : array (n_states, n_states)
    b_per_state : list of arrays, each (n_neurons,)
    W_per_state : list of arrays, each (n_neurons, n_latent)
    init_state_probs : array (n_states,)

    Returns
    -------
    spikes, states, disc_states, time_sec
    """
    n_states = len(A_per_state)
    key = jax.random.PRNGKey(seed)
    k_disc, k_state, k_spike = jax.random.split(key, 3)

    # Discrete states
    disc = np.zeros(n_time, dtype=int)
    disc[0] = int(jax.random.choice(jax.random.PRNGKey(seed), n_states, p=jnp.array(init_state_probs)))
    for t in range(1, n_time):
        k_disc, k = jax.random.split(k_disc)
        p = np.array(Z[disc[t - 1]])
        disc[t] = int(jax.random.choice(k, n_states, p=jnp.array(p)))

    # Continuous states
    states = np.zeros((n_time, n_latent))
    states[0] = [0.5, 0.0]
    for t in range(1, n_time):
        k_state, k = jax.random.split(k_state)
        A_t = A_per_state[disc[t]]
        Q_t = Q_per_state[disc[t]]
        noise = np.array(jax.random.multivariate_normal(k, jnp.zeros(n_latent), jnp.array(Q_t)))
        states[t] = A_t @ states[t - 1] + noise

    # Spikes
    spikes = np.zeros((n_time, n_neurons))
    for t in range(n_time):
        k_spike, k = jax.random.split(k_spike)
        b = b_per_state[disc[t]]
        W = W_per_state[disc[t]]
        log_rate = np.array(b) + np.array(W) @ states[t]
        rate = np.exp(log_rate) * dt
        spikes[t] = np.array(jax.random.poisson(k, jnp.array(rate)))

    time_sec = np.arange(n_time) * dt
    return jnp.array(spikes), states, disc, time_sec


def plot_summary(time_sec, disc, states, spikes, prob, smoother_mean,
                 state_names, state_colors, title, t_show_sec=60):
    """Standard 4-panel summary plot."""
    n_show = int(t_show_sec / dt)
    t_show = slice(0, min(n_show, len(time_sec)))
    n_states = prob.shape[1]

    amplitude = np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2)
    if smoother_mean is not None:
        sm_amp = np.sqrt(smoother_mean[:, 0] ** 2 + smoother_mean[:, 1] ** 2)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Panel 1: Discrete states
    ax = axes[0]
    for j in range(n_states):
        mask = (disc[t_show] == j).astype(float)
        ax.fill_between(time_sec[t_show], j - 0.4, j + 0.4,
                        where=mask > 0, alpha=0.3, color=state_colors[j],
                        label=f"True: {state_names[j]}")
    for j in range(n_states):
        ax.plot(time_sec[t_show], prob[t_show, j] * (n_states - 1),
                color=state_colors[j], alpha=0.8, linewidth=0.8)
    ax.set_ylabel("State")
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(state_names)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: Oscillator
    ax = axes[1]
    ax.plot(time_sec[t_show], states[t_show, 0], alpha=0.3, color="C0", label="True x[0]")
    if smoother_mean is not None:
        ax.plot(time_sec[t_show], smoother_mean[t_show, 0], alpha=0.8, color="C1",
                label="Smoothed x[0]")
    ax.set_ylabel("Oscillator")
    ax.set_title("Theta oscillator (cos component)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3: Amplitude
    ax = axes[2]
    ax.plot(time_sec[t_show], amplitude[t_show], alpha=0.3, color="C0", label="True")
    if smoother_mean is not None:
        ax.plot(time_sec[t_show], sm_amp[t_show], alpha=0.8, color="C1", label="Smoothed")
    ax.set_ylabel("Amplitude")
    ax.set_title("Oscillation amplitude")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 4: Spike raster
    ax = axes[3]
    n_show_neurons = min(10, n_neurons)
    for n in range(n_show_neurons):
        spike_times = np.where(np.array(spikes[t_show, n]) > 0)[0] * dt
        ax.scatter(spike_times, np.ones_like(spike_times) * n, s=0.5, c="k", alpha=0.3)
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Spike raster")
    ax.set_yticks(range(n_show_neurons))

    plt.tight_layout()
    return fig


# %% [markdown]
# # Step 1: S=2, shared spike params — theta-off vs theta-on
#
# Simplest case: two states differ only in oscillator damping.
# Both states have the same firing rates and phase preferences.
# The model must segment using dynamics alone.

# %%
print("=" * 70)
print("STEP 1: S=2, shared spike params (theta-off vs theta-on)")
print("=" * 70)

n_time_1 = 30000  # 5 minutes

# Dynamics
A_off = np.array(construct_common_oscillator_transition_matrix(
    jnp.array([freq_theta]), jnp.array([0.85]), sampling_freq))
A_on = np.array(construct_common_oscillator_transition_matrix(
    jnp.array([freq_theta]), jnp.array([0.995]), sampling_freq))
Q_shared = np.array(construct_common_oscillator_process_covariance(jnp.array([0.05])))

# Same spike params for both states
W_theta = np.stack([
    np.cos(preferred_phases) * modulation_strength,
    np.sin(preferred_phases) * modulation_strength,
], axis=-1)
b_low = np.log(np.random.RandomState(0).uniform(1.0, 5.0, n_neurons))  # 1-5 Hz baseline

Z_1 = np.array([[0.998, 0.002], [0.002, 0.998]])

spikes_1, states_1, disc_1, time_1 = simulate_switching_spikes(
    n_time_1,
    A_per_state=[A_off, A_on],
    Q_per_state=[Q_shared, Q_shared],
    Z=Z_1,
    b_per_state=[b_low, b_low],      # same baseline
    W_per_state=[W_theta, W_theta],   # same weights
    init_state_probs=[0.5, 0.5],
    seed=42,
)

print(f"Spikes: {int(jnp.sum(spikes_1))}, transitions: {np.sum(np.diff(disc_1) != 0)}")
print(f"State occupancy: {[f'{(disc_1==j).mean():.2f}' for j in range(2)]}")

# Fit
model_1 = SwitchingSpikeOscillatorModel(
    n_oscillators=1, n_neurons=n_neurons, n_discrete_states=2,
    sampling_freq=sampling_freq, dt=dt,
    q_regularization=QRegularizationConfig(),
    separate_spike_params=False,
)
model_1._initialize_parameters(jax.random.PRNGKey(0))

# Warm start
k_p = jax.random.PRNGKey(77)
k1, k2, k3, k4 = jax.random.split(k_p, 4)
A_true_1 = jnp.stack([jnp.array(A_off), jnp.array(A_on)], axis=-1)
Q_true_1 = jnp.stack([jnp.array(Q_shared)] * 2, axis=-1)
model_1.continuous_transition_matrix = A_true_1 + jax.random.normal(k1, A_true_1.shape) * 0.005
model_1.process_cov = Q_true_1 + jnp.abs(jax.random.normal(k2, Q_true_1.shape)) * 0.001
model_1.discrete_transition_matrix = jnp.array(Z_1)
model_1.spike_params = SpikeObsParams(
    baseline=jnp.array(b_low) + jax.random.normal(k3, (n_neurons,)) * 0.05,
    weights=jnp.array(W_theta) + jax.random.normal(k4, (n_neurons, n_latent)) * 0.02,
)

lls_1 = model_1.fit(spikes_1, max_iter=20, skip_init=True)

prob_1 = np.array(model_1.smoother_discrete_state_prob)
corr_1 = max(abs(np.corrcoef(disc_1.astype(float), prob_1[:, j])[0, 1]) for j in range(2))
sm_1 = np.array(jnp.einsum("tls,ts->tl", model_1.smoother_state_cond_mean,
                             model_1.smoother_discrete_state_prob))

print(f"LL improved: {lls_1[-1] > lls_1[0]} ({lls_1[-1] - lls_1[0]:.1f})")
print(f"State recovery |corr|: {corr_1:.3f}")
print(f"Latent tracking corr: {np.corrcoef(states_1[:,0], sm_1[:,0])[0,1]:.3f}")

fig_1 = plot_summary(time_1, disc_1, states_1, spikes_1, prob_1, sm_1,
                     ["Theta-off", "Theta-on"], ["C0", "C1"],
                     f"Step 1: Shared spike params, damping only (|corr|={corr_1:.3f})")
fig_1.savefig(output_dir / "ca1_step1_shared_params.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Step 2: S=2, separate spike params — theta-off (low rate) vs theta-on (higher rate + modulation)
#
# Now the two states also differ in firing rate: theta-on has elevated baseline.
# This is realistic: CA1 neurons fire more during theta.
# The model uses separate_spike_params=True.

# %%
print()
print("=" * 70)
print("STEP 2: S=2, separate spike params (rate + damping difference)")
print("=" * 70)

n_time_2 = 30000

# Theta-off: low baseline, weak modulation
b_off = np.log(np.random.RandomState(0).uniform(0.5, 2.0, n_neurons))  # 0.5-2 Hz
W_off = W_theta * 0.3  # weak modulation during non-theta

# Theta-on: elevated baseline, strong modulation
b_on = np.log(np.random.RandomState(1).uniform(3.0, 10.0, n_neurons))  # 3-10 Hz
W_on = W_theta  # full modulation

spikes_2, states_2, disc_2, time_2 = simulate_switching_spikes(
    n_time_2,
    A_per_state=[A_off, A_on],
    Q_per_state=[Q_shared, Q_shared],
    Z=Z_1,
    b_per_state=[b_off, b_on],
    W_per_state=[W_off, W_on],
    init_state_probs=[0.5, 0.5],
    seed=42,
)

print(f"Spikes: {int(jnp.sum(spikes_2))}, transitions: {np.sum(np.diff(disc_2) != 0)}")
print(f"Rate theta-off: {float(jnp.mean(spikes_2[disc_2==0])/dt):.1f} Hz/neuron")
print(f"Rate theta-on: {float(jnp.mean(spikes_2[disc_2==1])/dt):.1f} Hz/neuron")

# Fit with separate spike params
model_2 = SwitchingSpikeOscillatorModel(
    n_oscillators=1, n_neurons=n_neurons, n_discrete_states=2,
    sampling_freq=sampling_freq, dt=dt,
    q_regularization=QRegularizationConfig(),
    separate_spike_params=True,
)
model_2._initialize_parameters(jax.random.PRNGKey(0))

k_p2 = jax.random.PRNGKey(88)
k1, k2, k3, k4 = jax.random.split(k_p2, 4)
model_2.continuous_transition_matrix = A_true_1 + jax.random.normal(k1, A_true_1.shape) * 0.005
model_2.process_cov = Q_true_1 + jnp.abs(jax.random.normal(k2, Q_true_1.shape)) * 0.001
model_2.discrete_transition_matrix = jnp.array(Z_1)

b_per_state_init = jnp.stack([
    jnp.array(b_off) + jax.random.normal(jax.random.PRNGKey(10), (n_neurons,)) * 0.1,
    jnp.array(b_on) + jax.random.normal(jax.random.PRNGKey(11), (n_neurons,)) * 0.1,
], axis=-1)
W_per_state_init = jnp.stack([
    jnp.array(W_off) + jax.random.normal(jax.random.PRNGKey(12), (n_neurons, n_latent)) * 0.02,
    jnp.array(W_on) + jax.random.normal(jax.random.PRNGKey(13), (n_neurons, n_latent)) * 0.02,
], axis=-1)
model_2.spike_params = SpikeObsParams(baseline=b_per_state_init, weights=W_per_state_init)

lls_2 = model_2.fit(spikes_2, max_iter=20, skip_init=True)

prob_2 = np.array(model_2.smoother_discrete_state_prob)
corr_2 = max(abs(np.corrcoef(disc_2.astype(float), prob_2[:, j])[0, 1]) for j in range(2))
sm_2 = np.array(jnp.einsum("tls,ts->tl", model_2.smoother_state_cond_mean,
                             model_2.smoother_discrete_state_prob))

print(f"LL improved: {lls_2[-1] > lls_2[0]} ({lls_2[-1] - lls_2[0]:.1f})")
print(f"State recovery |corr|: {corr_2:.3f}")
print(f"Latent tracking corr: {np.corrcoef(states_2[:,0], sm_2[:,0])[0,1]:.3f}")

fig_2 = plot_summary(time_2, disc_2, states_2, spikes_2, prob_2, sm_2,
                     ["Theta-off", "Theta-on"], ["C0", "C1"],
                     f"Step 2: Separate spike params (|corr|={corr_2:.3f})")
fig_2.savefig(output_dir / "ca1_step2_separate_params.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Step 3: S=3 — theta-off, theta-on, SWR
#
# The full CA1 model:
# - State 0 (theta-off): low rate, damped oscillator, quiet immobility
# - State 1 (theta-on): elevated rate, theta-modulated, sustained oscillator
# - State 2 (SWR): very high synchronous firing, brief bursts, damped oscillator
#
# SWR characteristics:
# - Very high firing rates (50-100 Hz for active cells)
# - Brief (50-100 ms = 5-10 time steps at 100 Hz)
# - Synchronous across many neurons
# - No theta oscillation

# %%
print()
print("=" * 70)
print("STEP 3: S=3, theta-off / theta-on / SWR")
print("=" * 70)

n_time_3 = 30000

# SWR dynamics: heavily damped (no oscillation)
A_swr = np.array(construct_common_oscillator_transition_matrix(
    jnp.array([freq_theta]), jnp.array([0.5]), sampling_freq))  # very damped

# SWR spike params: very high rates, no phase modulation
b_swr = np.log(np.random.RandomState(2).uniform(30.0, 80.0, n_neurons))  # 30-80 Hz!
W_swr = W_theta * 0.05  # minimal modulation

# Transition matrix: SWR states are brief
# From theta-off: mostly stay, small chance of theta-on or SWR
# From theta-on: mostly stay, small chance of theta-off
# From SWR: quickly leave (brief events), go to theta-off
Z_3 = np.array([
    [0.996, 0.002, 0.002],  # theta-off: stable, rare transitions
    [0.002, 0.996, 0.002],  # theta-on: stable
    [0.10,  0.02,  0.88],   # SWR: short-lived (mean ~8 steps = 80ms), mostly back to off
])

spikes_3, states_3, disc_3, time_3 = simulate_switching_spikes(
    n_time_3,
    A_per_state=[A_off, A_on, A_swr],
    Q_per_state=[Q_shared, Q_shared, Q_shared],
    Z=Z_3,
    b_per_state=[b_off, b_on, b_swr],
    W_per_state=[W_off, W_on, W_swr],
    init_state_probs=[0.5, 0.4, 0.1],
    seed=42,
)

print(f"Spikes: {int(jnp.sum(spikes_3))}")
print(f"State occupancy: {[f'{(disc_3==j).mean():.3f}' for j in range(3)]}")
print(f"Rate theta-off: {float(jnp.mean(spikes_3[disc_3==0])/dt):.1f} Hz/neuron")
print(f"Rate theta-on: {float(jnp.mean(spikes_3[disc_3==1])/dt):.1f} Hz/neuron")
print(f"Rate SWR: {float(jnp.mean(spikes_3[disc_3==2])/dt):.1f} Hz/neuron")
print(f"SWR events: {np.sum(np.diff((disc_3==2).astype(int)) == 1)}")

# %%
# Fit 3-state model
model_3 = SwitchingSpikeOscillatorModel(
    n_oscillators=1, n_neurons=n_neurons, n_discrete_states=3,
    sampling_freq=sampling_freq, dt=dt,
    q_regularization=QRegularizationConfig(),
    separate_spike_params=True,
)
model_3._initialize_parameters(jax.random.PRNGKey(0))

# Warm start from true params
k_p3 = jax.random.PRNGKey(99)
k1, k2 = jax.random.split(k_p3)
A_true_3 = jnp.stack([jnp.array(A_off), jnp.array(A_on), jnp.array(A_swr)], axis=-1)
Q_true_3 = jnp.stack([jnp.array(Q_shared)] * 3, axis=-1)
model_3.continuous_transition_matrix = A_true_3 + jax.random.normal(k1, A_true_3.shape) * 0.005
model_3.process_cov = Q_true_3 + jnp.abs(jax.random.normal(k2, Q_true_3.shape)) * 0.001
model_3.discrete_transition_matrix = jnp.array(Z_3)

# Per-state spike params (3 states)
b_init_3 = jnp.stack([
    jnp.array(b_off) + jax.random.normal(jax.random.PRNGKey(20), (n_neurons,)) * 0.1,
    jnp.array(b_on) + jax.random.normal(jax.random.PRNGKey(21), (n_neurons,)) * 0.1,
    jnp.array(b_swr) + jax.random.normal(jax.random.PRNGKey(22), (n_neurons,)) * 0.1,
], axis=-1)
W_init_3 = jnp.stack([
    jnp.array(W_off) + jax.random.normal(jax.random.PRNGKey(23), (n_neurons, n_latent)) * 0.02,
    jnp.array(W_on) + jax.random.normal(jax.random.PRNGKey(24), (n_neurons, n_latent)) * 0.02,
    jnp.array(W_swr) + jax.random.normal(jax.random.PRNGKey(25), (n_neurons, n_latent)) * 0.02,
], axis=-1)
model_3.spike_params = SpikeObsParams(baseline=b_init_3, weights=W_init_3)

print("Fitting 3-state model...")
lls_3 = model_3.fit(spikes_3, max_iter=20, skip_init=True)

prob_3 = np.array(model_3.smoother_discrete_state_prob)
sm_3 = np.array(jnp.einsum("tls,ts->tl", model_3.smoother_state_cond_mean,
                             model_3.smoother_discrete_state_prob))

# State recovery: find best permutation
from itertools import permutations
best_corr_3 = 0
best_perm = None
for perm in permutations(range(3)):
    # Map fitted states to true states via permutation
    mapped_prob = np.zeros_like(prob_3)
    for j_fit, j_true in enumerate(perm):
        mapped_prob[:, j_true] = prob_3[:, j_fit]
    # Compute correlation for each true state
    corrs = []
    for j in range(3):
        c = np.corrcoef((disc_3 == j).astype(float), mapped_prob[:, j])[0, 1]
        corrs.append(c)
    mean_corr = np.mean(corrs)
    if mean_corr > best_corr_3:
        best_corr_3 = mean_corr
        best_perm = perm
        best_corrs = corrs

print(f"LL improved: {lls_3[-1] > lls_3[0]} ({lls_3[-1] - lls_3[0]:.1f})")
print(f"Best permutation: {best_perm}")
print(f"Per-state recovery: theta-off={best_corrs[0]:.3f}, theta-on={best_corrs[1]:.3f}, SWR={best_corrs[2]:.3f}")
print(f"Mean state recovery: {best_corr_3:.3f}")
print(f"Latent tracking corr: {np.corrcoef(states_3[:,0], sm_3[:,0])[0,1]:.3f}")

# Reorder prob to match true states
prob_3_ordered = np.zeros_like(prob_3)
for j_fit, j_true in enumerate(best_perm):
    prob_3_ordered[:, j_true] = prob_3[:, j_fit]

fig_3 = plot_summary(time_3, disc_3, states_3, spikes_3, prob_3_ordered, sm_3,
                     ["Theta-off", "Theta-on", "SWR"], ["C0", "C1", "C3"],
                     f"Step 3: 3-state model (mean |corr|={best_corr_3:.3f})")
fig_3.savefig(output_dir / "ca1_step3_three_states.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Summary comparison

# %%
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Step 1 (S=2, shared params, damping only):     |corr| = {corr_1:.3f}")
print(f"Step 2 (S=2, separate params, rate + damping):  |corr| = {corr_2:.3f}")
print(f"Step 3 (S=3, separate params, +SWR):            mean   = {best_corr_3:.3f}")
print(f"  Per-state: theta-off={best_corrs[0]:.3f}, theta-on={best_corrs[1]:.3f}, SWR={best_corrs[2]:.3f}")
