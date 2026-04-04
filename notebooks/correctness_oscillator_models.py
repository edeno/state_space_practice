"""Correctness verification using proper oscillator model structure.

Demonstrates the switching point-process model with:
- COM: Common Oscillator Model (independent oscillators, block-diagonal A)
- DIM: Directed Influence Model (coupled oscillators, off-diagonal blocks in A)
- Switching between two coupling patterns

Uses the library's oscillator construction utilities directly.
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
    construct_correlated_noise_process_covariance,
    construct_directed_influence_transition_matrix,
)
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
# # 1. COM: Two independent oscillators, switching damping
#
# - Oscillator 1: theta rhythm at 8 Hz
# - Oscillator 2: gamma rhythm at 40 Hz
# - State 0: both heavily damped (no rhythmicity)
# - State 1: both lightly damped (strong rhythmicity)
#
# This is the simplest oscillator structure. The 4D latent state is
# [x1_cos, x1_sin, x2_cos, x2_sin].

# %%
sampling_freq = 100.0  # Hz
dt = 1.0 / sampling_freq
n_time = 30000  # 5 minutes at 100 Hz
n_neurons = 8
n_oscillators = 1
n_latent = 2 * n_oscillators

freqs = jnp.array([8.0])  # theta only

# State 0: heavily damped (non-rhythmic)
damping_0 = jnp.array([0.85])
A0_com = construct_common_oscillator_transition_matrix(freqs, damping_0, sampling_freq)

# State 1: lightly damped (rhythmic)
damping_1 = jnp.array([0.995])
A1_com = construct_common_oscillator_transition_matrix(freqs, damping_1, sampling_freq)

A_com = jnp.stack([A0_com, A1_com], axis=-1)

Q_var = jnp.array([0.05])
Q_com = jnp.stack([
    construct_common_oscillator_process_covariance(Q_var),
    construct_common_oscillator_process_covariance(Q_var),
], axis=-1)

Z_com = jnp.array([[0.998, 0.002], [0.002, 0.998]])

print("COM Model:")
print(f"  Oscillator: {float(freqs[0])} Hz (theta)")
print(f"  A0 shape: {A0_com.shape}")
print(f"  A0 eigenvalues: {np.abs(np.linalg.eigvals(np.array(A0_com)))}")
print(f"  A1 eigenvalues: {np.abs(np.linalg.eigvals(np.array(A1_com)))}")
print(f"  Latent dim: {n_latent}")

# %%
# Spike weights: neurons at different preferred phases
key = jax.random.PRNGKey(42)

phases = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
W_com = jnp.stack([jnp.cos(phases) * 0.5, jnp.sin(phases) * 0.5], axis=-1)

b_com = jnp.ones(n_neurons) * 2.5  # ~12 Hz baseline

# Simulate
k_disc, k_state, k_spike = jax.random.split(key, 3)

disc_states = np.zeros(n_time, dtype=int)
disc_states[0] = 0
for t in range(1, n_time):
    k_disc, k = jax.random.split(k_disc)
    if float(jax.random.uniform(k)) < float(Z_com[disc_states[t - 1], 1 - disc_states[t - 1]]):
        disc_states[t] = 1 - disc_states[t - 1]
    else:
        disc_states[t] = disc_states[t - 1]

states = np.zeros((n_time, n_latent))
states[0] = [1.0, 0.0]
for t in range(1, n_time):
    k_state, k = jax.random.split(k_state)
    A_t = np.array(A_com[:, :, disc_states[t]])
    Q_t = np.array(Q_com[:, :, disc_states[t]])
    noise = np.array(jax.random.multivariate_normal(k, jnp.zeros(n_latent), jnp.array(Q_t)))
    states[t] = A_t @ states[t - 1] + noise

spikes_com = np.zeros((n_time, n_neurons))
for t in range(n_time):
    k_spike, k = jax.random.split(k_spike)
    log_rate = np.array(b_com) + np.array(W_com) @ states[t]
    rate = np.exp(log_rate) * dt
    spikes_com[t] = np.array(jax.random.poisson(k, jnp.array(rate)))

spikes_com = jnp.array(spikes_com)

# Compute oscillator amplitude
theta_amp = np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2)

print(f"\nSimulated COM data:")
print(f"  Total spikes: {int(jnp.sum(spikes_com))}")
print(f"  State transitions: {np.sum(np.diff(disc_states) != 0)}")

# %%
# Fit model with warm start
model_com = SwitchingSpikeOscillatorModel(
    n_oscillators=n_oscillators,
    n_neurons=n_neurons,
    n_discrete_states=2,
    sampling_freq=sampling_freq,
    dt=dt,
    q_regularization=QRegularizationConfig(),
    separate_spike_params=False,
)
model_com._initialize_parameters(jax.random.PRNGKey(0))

# Warm start
k_p = jax.random.PRNGKey(77)
k1, k2, k3, k4 = jax.random.split(k_p, 4)
model_com.continuous_transition_matrix = A_com + jax.random.normal(k1, A_com.shape) * 0.005
model_com.process_cov = Q_com + jnp.abs(jax.random.normal(k2, Q_com.shape)) * 0.001
model_com.discrete_transition_matrix = Z_com
model_com.spike_params = SpikeObsParams(
    baseline=b_com + jax.random.normal(k3, b_com.shape) * 0.05,
    weights=W_com + jax.random.normal(k4, W_com.shape) * 0.02,
)

print("Fitting COM model...")
lls_com = model_com.fit(spikes_com, max_iter=20, skip_init=True)
print(f"Done. LL improvement: {lls_com[-1] - lls_com[0]:.1f}")

# %%
# Extract results
prob_com = np.array(model_com.smoother_discrete_state_prob)
corr_com = [np.corrcoef(disc_states.astype(float), prob_com[:, j])[0, 1] for j in range(2)]
best_j_com = np.argmax(np.abs(corr_com))
rhythmic_label = best_j_com if corr_com[best_j_com] > 0 else 1 - best_j_com

smoother_mean_com = np.array(jnp.einsum(
    "tls,ts->tl", model_com.smoother_state_cond_mean, model_com.smoother_discrete_state_prob,
))
smoother_theta_amp = np.sqrt(smoother_mean_com[:, 0] ** 2 + smoother_mean_com[:, 1] ** 2)

# Recovered frequencies
fitted_eigs = [np.linalg.eigvals(np.array(model_com.continuous_transition_matrix[:, :, j])) for j in range(2)]

def extract_oscillator_freq(eigs, n_osc, sampling_freq):
    """Extract frequencies from eigenvalues of block-diagonal rotation matrix."""
    freqs_out = []
    for k in range(n_osc):
        # Each oscillator contributes a conjugate pair
        # Sort by angle to pair them
        angles = np.angle(eigs)
        positive = angles > 0
        pos_angles = sorted(angles[positive])
        if len(pos_angles) >= k + 1:
            freqs_out.append(pos_angles[k] * sampling_freq / (2 * np.pi))
        else:
            freqs_out.append(np.nan)
    return freqs_out

freqs_state0 = extract_oscillator_freq(fitted_eigs[0], n_oscillators, sampling_freq)
freqs_state1 = extract_oscillator_freq(fitted_eigs[1], n_oscillators, sampling_freq)

# %%
# Plot
t_show = slice(0, 6000)  # first 60 seconds
time_sec = np.arange(n_time) * dt

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel 1: True discrete state
ax = axes[0]
ax.fill_between(time_sec[t_show], 0, disc_states[t_show], alpha=0.3, color="C1", label="State 1 (rhythmic)")
ax.plot(time_sec[t_show], prob_com[t_show, rhythmic_label], color="C0", alpha=0.8, label="P(rhythmic | y)")
ax.set_ylabel("State")
ax.set_title(f"COM: Discrete state recovery (|corr| = {max(np.abs(corr_com)):.3f})")
ax.legend(loc="upper right")
ax.set_ylim(-0.05, 1.05)

# Panel 2: Theta oscillator
ax = axes[1]
ax.plot(time_sec[t_show], states[t_show, 0], alpha=0.3, label="True theta x")
ax.plot(time_sec[t_show], smoother_mean_com[t_show, 0], alpha=0.8, label="Smoothed theta x")
ax.set_ylabel("Theta (8 Hz)")
ax.set_title("Theta oscillator tracking")
ax.legend(loc="upper right")

# Panel 3: Theta amplitude
ax = axes[2]
ax.plot(time_sec[t_show], theta_amp[t_show], alpha=0.3, label="True")
ax.plot(time_sec[t_show], smoother_theta_amp[t_show], alpha=0.8, label="Smoothed")
ax.set_ylabel("Theta amplitude")
ax.set_title("Theta amplitude envelope")
ax.legend(loc="upper right")

# Panel 4: Parameter summary
ax = axes[3]
text = (
    f"COM Model: theta oscillator at {float(freqs[0]):.0f} Hz\n"
    f"State 0: damped (r={float(damping_0[0]):.2f}) | State 1: rhythmic (r={float(damping_1[0]):.3f})\n\n"
    f"Recovered frequencies (Hz):\n"
    f"  State 0: {freqs_state0[0]:.1f}  (true: {float(freqs[0]):.0f})\n"
    f"  State 1: {freqs_state1[0]:.1f}  (true: {float(freqs[0]):.0f})\n\n"
    f"Spectral radii (damping):\n"
    f"  State 0: {sorted(np.abs(fitted_eigs[0]))[-1]:.3f}  (true: {float(damping_0[0]):.3f})\n"
    f"  State 1: {sorted(np.abs(fitted_eigs[1]))[-1]:.3f}  (true: {float(damping_1[0]):.3f})\n\n"
    f"State recovery |corr|: {max(np.abs(corr_com)):.3f}\n"
    f"Theta tracking corr: {np.corrcoef(states[:,0], smoother_mean_com[:,0])[0,1]:.3f}"
)
ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=10,
        verticalalignment="center", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_axis_off()

plt.tight_layout()
plt.savefig(output_dir / "oscillator_1_COM.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # 2. CNM: Correlated Noise Model — switching noise correlation
#
# Two oscillators at 8 Hz and 12 Hz with the SAME independent dynamics (COM-style A).
# The coupling is through the process noise covariance Q:
# - State 0: independent noise (Q block-diagonal, like COM)
# - State 1: correlated noise (Q has off-diagonal blocks)
#
# This represents shared unobserved inputs driving both oscillators.

# %%
n_time_cnm = 30000  # 5 minutes
n_neurons_cnm = 10
n_osc_cnm = 2
n_latent_cnm = 2 * n_osc_cnm
freqs_cnm = jnp.array([8.0, 12.0])
damping_cnm = jnp.array([0.99, 0.99])
sampling_freq_cnm = 100.0
dt_cnm = 1.0 / sampling_freq_cnm

# Same A for both states (COM-style, no direct coupling)
A_cnm_single = construct_common_oscillator_transition_matrix(
    freqs_cnm, damping_cnm, sampling_freq_cnm,
)
A_cnm = jnp.stack([A_cnm_single, A_cnm_single], axis=-1)

# State 0: independent noise (block-diagonal Q)
Q0_cnm = construct_common_oscillator_process_covariance(jnp.array([0.03, 0.03]))

# State 1: correlated noise (off-diagonal blocks in Q)
Q1_cnm = construct_correlated_noise_process_covariance(
    variance=jnp.array([0.03, 0.03]),
    phase_difference=jnp.array([[0.0, 0.0], [jnp.pi / 3, 0.0]]),  # 60 degree phase lag
    coupling_strength=jnp.array([[0.0, 0.0], [0.02, 0.0]]),  # osc1 noise drives osc2
)

Q_cnm = jnp.stack([Q0_cnm, Q1_cnm], axis=-1)
Z_cnm = jnp.array([[0.998, 0.002], [0.002, 0.998]])

print("CNM Model:")
print(f"  A is SAME for both states (independent oscillators, COM-style)")
print(f"  Q0 off-diagonal block: {np.array(Q0_cnm[2:4, 0:2])}")
print(f"  Q1 off-diagonal block:\n    {np.array(Q1_cnm[2:4, 0:2])}")
print(f"  Q0 is block-diag: {np.allclose(np.array(Q0_cnm[2:4, 0:2]), 0)}")
print(f"  Q1 has off-diag: {not np.allclose(np.array(Q1_cnm[2:4, 0:2]), 0)}")

# %%
# Spike weights and simulation
key_cnm = jax.random.PRNGKey(33)
k_w_cnm, k_s_cnm = jax.random.split(key_cnm)
W_cnm = jax.random.normal(k_w_cnm, (n_neurons_cnm, n_latent_cnm)) * 0.3
b_cnm = jnp.ones(n_neurons_cnm) * 2.0

k_disc_c, k_state_c, k_spike_c = jax.random.split(k_s_cnm, 3)

disc_cnm = np.zeros(n_time_cnm, dtype=int)
for t in range(1, n_time_cnm):
    k_disc_c, k = jax.random.split(k_disc_c)
    if float(jax.random.uniform(k)) < float(Z_cnm[disc_cnm[t - 1], 1 - disc_cnm[t - 1]]):
        disc_cnm[t] = 1 - disc_cnm[t - 1]
    else:
        disc_cnm[t] = disc_cnm[t - 1]

states_cnm = np.zeros((n_time_cnm, n_latent_cnm))
states_cnm[0] = [1.0, 0.0, 0.5, 0.0]
for t in range(1, n_time_cnm):
    k_state_c, k = jax.random.split(k_state_c)
    A_t = np.array(A_cnm[:, :, disc_cnm[t]])
    Q_t = np.array(Q_cnm[:, :, disc_cnm[t]])
    noise = np.array(jax.random.multivariate_normal(k, jnp.zeros(n_latent_cnm), jnp.array(Q_t)))
    states_cnm[t] = A_t @ states_cnm[t - 1] + noise

spikes_cnm = np.zeros((n_time_cnm, n_neurons_cnm))
for t in range(n_time_cnm):
    k_spike_c, k = jax.random.split(k_spike_c)
    log_rate = np.array(b_cnm) + np.array(W_cnm) @ states_cnm[t]
    rate = np.exp(log_rate) * dt_cnm
    spikes_cnm[t] = np.array(jax.random.poisson(k, jnp.array(rate)))

spikes_cnm = jnp.array(spikes_cnm)
print(f"\nSimulated CNM data: {int(jnp.sum(spikes_cnm))} spikes, "
      f"{np.sum(np.diff(disc_cnm) != 0)} transitions")

# %%
# Fit
model_cnm = SwitchingSpikeOscillatorModel(
    n_oscillators=n_osc_cnm,
    n_neurons=n_neurons_cnm,
    n_discrete_states=2,
    sampling_freq=sampling_freq_cnm,
    dt=dt_cnm,
    q_regularization=QRegularizationConfig(),
    separate_spike_params=False,
)
model_cnm._initialize_parameters(jax.random.PRNGKey(0))

k_p_cnm = jax.random.PRNGKey(44)
k1, k2, k3, k4 = jax.random.split(k_p_cnm, 4)
model_cnm.continuous_transition_matrix = A_cnm + jax.random.normal(k1, A_cnm.shape) * 0.003
model_cnm.process_cov = Q_cnm + jnp.abs(jax.random.normal(k2, Q_cnm.shape)) * 0.001
model_cnm.discrete_transition_matrix = Z_cnm
model_cnm.spike_params = SpikeObsParams(
    baseline=b_cnm + jax.random.normal(k3, b_cnm.shape) * 0.05,
    weights=W_cnm + jax.random.normal(k4, W_cnm.shape) * 0.02,
)

print("Fitting CNM model...")
lls_cnm = model_cnm.fit(spikes_cnm, max_iter=20, skip_init=True)
print(f"Done. LL improvement: {lls_cnm[-1] - lls_cnm[0]:.1f}")

# %%
prob_cnm = np.array(model_cnm.smoother_discrete_state_prob)
corr_cnm = [np.corrcoef(disc_cnm.astype(float), prob_cnm[:, j])[0, 1] for j in range(2)]
best_corr_cnm = max(np.abs(corr_cnm))
corr_label_cnm = np.argmax(np.abs(corr_cnm))
if corr_cnm[corr_label_cnm] < 0:
    corr_label_cnm = 1 - corr_label_cnm

smoother_mean_cnm = np.array(jnp.einsum(
    "tls,ts->tl", model_cnm.smoother_state_cond_mean, model_cnm.smoother_discrete_state_prob,
))

# Check Q recovery: off-diagonal blocks
fitted_Q = [np.array(model_cnm.process_cov[:, :, j]) for j in range(2)]
Q_offdiag = [np.linalg.norm(fitted_Q[j][2:4, 0:2]) for j in range(2)]
Q_order = np.argsort(Q_offdiag)

time_sec_cnm = np.arange(n_time_cnm) * dt_cnm
t_show_cnm = slice(0, 6000)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

ax = axes[0]
ax.fill_between(time_sec_cnm[t_show_cnm], 0, disc_cnm[t_show_cnm], alpha=0.3, color="C1",
                label="State 1 (correlated noise)")
ax.plot(time_sec_cnm[t_show_cnm], prob_cnm[t_show_cnm, corr_label_cnm], color="C0", alpha=0.8,
        label="P(correlated | y)")
ax.set_ylabel("State")
ax.set_title(f"CNM: Discrete state recovery (|corr| = {best_corr_cnm:.3f})")
ax.legend(loc="upper right")
ax.set_ylim(-0.05, 1.05)

ax = axes[1]
ax.plot(time_sec_cnm[t_show_cnm], states_cnm[t_show_cnm, 0], alpha=0.3, label="True osc1 x")
ax.plot(time_sec_cnm[t_show_cnm], smoother_mean_cnm[t_show_cnm, 0], alpha=0.8, label="Smoothed osc1 x")
ax.set_ylabel("Osc1 (8 Hz)")
ax.set_title("Oscillator 1 tracking")
ax.legend(loc="upper right")

ax = axes[2]
ax.plot(time_sec_cnm[t_show_cnm], states_cnm[t_show_cnm, 2], alpha=0.3, label="True osc2 x")
ax.plot(time_sec_cnm[t_show_cnm], smoother_mean_cnm[t_show_cnm, 2], alpha=0.8, label="Smoothed osc2 x")
ax.set_ylabel("Osc2 (12 Hz)")
ax.set_title("Oscillator 2 tracking")
ax.legend(loc="upper right")

ax = axes[3]
true_Q_offdiag_0 = np.linalg.norm(np.array(Q0_cnm[2:4, 0:2]))
true_Q_offdiag_1 = np.linalg.norm(np.array(Q1_cnm[2:4, 0:2]))
text = (
    f"CNM Model: 2 oscillators ({float(freqs_cnm[0]):.0f} Hz + {float(freqs_cnm[1]):.0f} Hz)\n"
    f"A is IDENTICAL for both states (COM-style, no direct coupling)\n"
    f"States differ ONLY in noise correlation (Q off-diagonal blocks)\n\n"
    f"Q off-diagonal block norm (osc2-osc1 noise coupling):\n"
    f"  State w/ less coupling: {Q_offdiag[Q_order[0]]:.4f}  (true: {true_Q_offdiag_0:.4f})\n"
    f"  State w/ more coupling: {Q_offdiag[Q_order[1]]:.4f}  (true: {true_Q_offdiag_1:.4f})\n\n"
    f"State recovery |corr|: {best_corr_cnm:.3f}\n"
    f"Osc1 tracking corr: {np.corrcoef(states_cnm[:,0], smoother_mean_cnm[:,0])[0,1]:.3f}\n"
    f"Osc2 tracking corr: {np.corrcoef(states_cnm[:,2], smoother_mean_cnm[:,2])[0,1]:.3f}"
)
ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=10,
        verticalalignment="center", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_axis_off()

plt.tight_layout()
plt.savefig(output_dir / "oscillator_2_CNM.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # 3. DIM: Coupled oscillators with switching coupling
#
# Two oscillators at 8 Hz and 12 Hz. The coupling pattern switches:
# - State 0: no coupling (independent, like COM)
# - State 1: oscillator 1 drives oscillator 2 (directed influence)

# %%
n_time_dim = 30000  # 5 minutes at 100 Hz
n_neurons_dim = 10
n_osc_dim = 2
n_latent_dim = 2 * n_osc_dim
freqs_dim = jnp.array([8.0, 12.0])
damping_dim = jnp.array([0.99, 0.99])
sampling_freq_dim = 100.0
dt_dim = 1.0 / sampling_freq_dim

# State 0: NO coupling (COM-like)
coupling_0 = jnp.zeros((n_osc_dim, n_osc_dim))
phase_diff_0 = jnp.zeros((n_osc_dim, n_osc_dim))
A0_dim = construct_directed_influence_transition_matrix(
    freqs_dim, damping_dim, coupling_0, phase_diff_0, sampling_freq_dim,
)

# State 1: oscillator 1 → oscillator 2 coupling
coupling_1 = jnp.array([[0.0, 0.0], [0.15, 0.0]])  # osc1 drives osc2
phase_diff_1 = jnp.array([[0.0, 0.0], [jnp.pi / 4, 0.0]])  # 45 degree lag
A1_dim = construct_directed_influence_transition_matrix(
    freqs_dim, damping_dim, coupling_1, phase_diff_1, sampling_freq_dim,
)

A_dim = jnp.stack([A0_dim, A1_dim], axis=-1)
Q_dim = jnp.stack([
    construct_common_oscillator_process_covariance(jnp.array([0.02, 0.02])),
] * 2, axis=-1)
Z_dim = jnp.array([[0.998, 0.002], [0.002, 0.998]])

print("DIM Model:")
print(f"  State 0 (uncoupled): A0 has off-diagonal blocks = 0")
print(f"  State 1 (coupled): osc1→osc2 with strength=0.15, phase_diff=pi/4")
print(f"  A0 spectral radius: {float(jnp.max(jnp.abs(jnp.linalg.eigvals(A0_dim)))):.3f}")
print(f"  A1 spectral radius: {float(jnp.max(jnp.abs(jnp.linalg.eigvals(A1_dim)))):.3f}")

# Off-diagonal check
print(f"\n  A0 off-diagonal block (osc2←osc1):\n    {np.array(A0_dim[2:4, 0:2])}")
print(f"  A1 off-diagonal block (osc2←osc1):\n    {np.array(A1_dim[2:4, 0:2])}")

# %%
# Spike weights: all neurons coupled to both oscillators
key_dim = jax.random.PRNGKey(55)
k_w_dim, k_s_dim = jax.random.split(key_dim)
W_dim = jax.random.normal(k_w_dim, (n_neurons_dim, n_latent_dim)) * 0.3
b_dim = jnp.ones(n_neurons_dim) * 2.0

# Simulate
k_disc_d, k_state_d, k_spike_d = jax.random.split(k_s_dim, 3)

disc_dim = np.zeros(n_time_dim, dtype=int)
for t in range(1, n_time_dim):
    k_disc_d, k = jax.random.split(k_disc_d)
    if float(jax.random.uniform(k)) < float(Z_dim[disc_dim[t - 1], 1 - disc_dim[t - 1]]):
        disc_dim[t] = 1 - disc_dim[t - 1]
    else:
        disc_dim[t] = disc_dim[t - 1]

states_dim = np.zeros((n_time_dim, n_latent_dim))
states_dim[0] = [1.0, 0.0, 0.5, 0.0]
for t in range(1, n_time_dim):
    k_state_d, k = jax.random.split(k_state_d)
    A_t = np.array(A_dim[:, :, disc_dim[t]])
    Q_t = np.array(Q_dim[:, :, disc_dim[t]])
    noise = np.array(jax.random.multivariate_normal(k, jnp.zeros(n_latent_dim), jnp.array(Q_t)))
    states_dim[t] = A_t @ states_dim[t - 1] + noise

spikes_dim = np.zeros((n_time_dim, n_neurons_dim))
for t in range(n_time_dim):
    k_spike_d, k = jax.random.split(k_spike_d)
    log_rate = np.array(b_dim) + np.array(W_dim) @ states_dim[t]
    rate = np.exp(log_rate) * dt_dim
    spikes_dim[t] = np.array(jax.random.poisson(k, jnp.array(rate)))

spikes_dim = jnp.array(spikes_dim)
print(f"\nSimulated DIM data: {int(jnp.sum(spikes_dim))} spikes, {np.sum(np.diff(disc_dim) != 0)} transitions")

# %%
# Fit
model_dim = SwitchingSpikeOscillatorModel(
    n_oscillators=n_osc_dim,
    n_neurons=n_neurons_dim,
    n_discrete_states=2,
    sampling_freq=sampling_freq_dim,
    dt=dt_dim,
    q_regularization=QRegularizationConfig(),
    separate_spike_params=False,
)
model_dim._initialize_parameters(jax.random.PRNGKey(0))

k_p2 = jax.random.PRNGKey(88)
k1, k2, k3, k4 = jax.random.split(k_p2, 4)
model_dim.continuous_transition_matrix = A_dim + jax.random.normal(k1, A_dim.shape) * 0.003
model_dim.process_cov = Q_dim + jnp.abs(jax.random.normal(k2, Q_dim.shape)) * 0.001
model_dim.discrete_transition_matrix = Z_dim
model_dim.spike_params = SpikeObsParams(
    baseline=b_dim + jax.random.normal(k3, b_dim.shape) * 0.05,
    weights=W_dim + jax.random.normal(k4, W_dim.shape) * 0.02,
)

print("Fitting DIM model...")
lls_dim = model_dim.fit(spikes_dim, max_iter=20, skip_init=True)
print(f"Done. LL improvement: {lls_dim[-1] - lls_dim[0]:.1f}")

# %%
prob_dim = np.array(model_dim.smoother_discrete_state_prob)
corr_dim = [np.corrcoef(disc_dim.astype(float), prob_dim[:, j])[0, 1] for j in range(2)]
best_corr_dim = max(np.abs(corr_dim))
coupled_label = np.argmax(np.abs(corr_dim))
if corr_dim[coupled_label] < 0:
    coupled_label = 1 - coupled_label

smoother_mean_dim = np.array(jnp.einsum(
    "tls,ts->tl", model_dim.smoother_state_cond_mean, model_dim.smoother_discrete_state_prob,
))

# Compare off-diagonal coupling blocks
fitted_A = [np.array(model_dim.continuous_transition_matrix[:, :, j]) for j in range(2)]
coupling_block_0 = fitted_A[0][2:4, 0:2]  # osc2←osc1 in state 0
coupling_block_1 = fitted_A[1][2:4, 0:2]  # osc2←osc1 in state 1

# After projection, the blocks should be rotation-like
# Coupling strength = Frobenius norm of off-diagonal block
coupling_strength = [np.linalg.norm(coupling_block_0), np.linalg.norm(coupling_block_1)]

# Sort to handle label swap
order_dim = np.argsort(coupling_strength)

# %%
time_sec_dim = np.arange(n_time_dim) * dt_dim
t_show_dim = slice(0, 6000)  # first 60 seconds

fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

ax = axes[0]
ax.fill_between(time_sec_dim[t_show_dim], 0, disc_dim[t_show_dim], alpha=0.3, color="C1", label="State 1 (coupled)")
ax.plot(time_sec_dim[t_show_dim], prob_dim[t_show_dim, coupled_label], color="C0", alpha=0.8, label="P(coupled | y)")
ax.set_ylabel("State")
ax.set_title(f"DIM: Discrete state recovery (|corr| = {best_corr_dim:.3f})")
ax.legend(loc="upper right")
ax.set_ylim(-0.05, 1.05)

ax = axes[1]
ax.plot(time_sec_dim[t_show_dim], states_dim[t_show_dim, 0], alpha=0.3, label="True osc1 x")
ax.plot(time_sec_dim[t_show_dim], smoother_mean_dim[t_show_dim, 0], alpha=0.8, label="Smoothed osc1 x")
ax.set_ylabel("Osc1 (8 Hz)")
ax.set_title("Oscillator 1 (driver)")
ax.legend(loc="upper right")

ax = axes[2]
ax.plot(time_sec_dim[t_show_dim], states_dim[t_show_dim, 2], alpha=0.3, label="True osc2 x")
ax.plot(time_sec_dim[t_show_dim], smoother_mean_dim[t_show_dim, 2], alpha=0.8, label="Smoothed osc2 x")
ax.set_ylabel("Osc2 (12 Hz)")
ax.set_title("Oscillator 2 (driven, when coupled)")
ax.legend(loc="upper right")

# Panel 4: Coupling matrix visualization
ax = axes[3]
true_coupling_0 = np.array(A0_dim[2:4, 0:2])
true_coupling_1 = np.array(A1_dim[2:4, 0:2])

im_data = np.concatenate([
    true_coupling_0, np.ones((2, 1)) * np.nan, coupling_block_0,
    np.ones((2, 1)) * np.nan,
    true_coupling_1, np.ones((2, 1)) * np.nan, coupling_block_1,
], axis=1)
# Just show as text
text = (
    f"Off-diagonal coupling block (osc2 ← osc1):\n\n"
    f"State 0 (uncoupled):\n"
    f"  True:    {true_coupling_0[0]}\n"
    f"           {true_coupling_0[1]}\n"
    f"  Fitted:  {coupling_block_0[0]}\n"
    f"           {coupling_block_0[1]}\n"
    f"  Coupling strength: true={np.linalg.norm(true_coupling_0):.4f}, "
    f"fitted={coupling_strength[0]:.4f}\n\n"
    f"State 1 (coupled):\n"
    f"  True:    {true_coupling_1[0]}\n"
    f"           {true_coupling_1[1]}\n"
    f"  Fitted:  {coupling_block_1[0]}\n"
    f"           {coupling_block_1[1]}\n"
    f"  Coupling strength: true={np.linalg.norm(true_coupling_1):.4f}, "
    f"fitted={coupling_strength[1]:.4f}"
)
ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=9,
        verticalalignment="center", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_axis_off()
ax.set_title("Coupling block recovery")

# Panel 5: Summary
ax = axes[4]
osc1_corr = np.corrcoef(states_dim[:, 0], smoother_mean_dim[:, 0])[0, 1]
osc2_corr = np.corrcoef(states_dim[:, 2], smoother_mean_dim[:, 2])[0, 1]

eigs_0 = np.linalg.eigvals(fitted_A[0])
eigs_1 = np.linalg.eigvals(fitted_A[1])
freq_0 = sorted(np.abs(np.angle(eigs_0[eigs_0.imag > 0])) * sampling_freq_dim / (2 * np.pi))
freq_1 = sorted(np.abs(np.angle(eigs_1[eigs_1.imag > 0])) * sampling_freq_dim / (2 * np.pi))

text2 = (
    f"DIM Model: 2 oscillators ({float(freqs_dim[0]):.0f} Hz + {float(freqs_dim[1]):.0f} Hz)\n"
    f"State 0: uncoupled | State 1: osc1→osc2 coupled\n\n"
    f"Recovered frequencies:\n"
    f"  State 0: {[f'{f:.1f}' for f in freq_0]} Hz  (true: [8, 12])\n"
    f"  State 1: {[f'{f:.1f}' for f in freq_1]} Hz  (true: [8, 12])\n\n"
    f"Coupling strength (osc2←osc1):\n"
    f"  State 0: {coupling_strength[0]:.4f}  (true: 0.0000)\n"
    f"  State 1: {coupling_strength[1]:.4f}  (true: {np.linalg.norm(true_coupling_1):.4f})\n\n"
    f"State recovery |corr|: {best_corr_dim:.3f}\n"
    f"Osc1 tracking corr: {osc1_corr:.3f}\n"
    f"Osc2 tracking corr: {osc2_corr:.3f}"
)
ax.text(0.05, 0.5, text2, transform=ax.transAxes, fontsize=10,
        verticalalignment="center", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_axis_off()

plt.tight_layout()
plt.savefig(output_dir / "oscillator_3_DIM.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nCOM Model (theta, switching damping):")
print(f"  State recovery |corr|: {max(np.abs(corr_com)):.3f}")
print(f"  Theta tracking corr: {np.corrcoef(states[:,0], smoother_mean_com[:,0])[0,1]:.3f}")
print(f"  Freq recovery: {[f'{f:.1f}' for f in freqs_state0]} / {[f'{f:.1f}' for f in freqs_state1]} Hz")

print(f"\nCNM Model (8 Hz + 12 Hz, switching noise correlation):")
print(f"  State recovery |corr|: {best_corr_cnm:.3f}")
print(f"  Osc1 tracking corr: {np.corrcoef(states_cnm[:,0], smoother_mean_cnm[:,0])[0,1]:.3f}")
print(f"  Osc2 tracking corr: {np.corrcoef(states_cnm[:,2], smoother_mean_cnm[:,2])[0,1]:.3f}")
print(f"  Q off-diag recovery: indep={Q_offdiag[Q_order[0]]:.4f}, corr={Q_offdiag[Q_order[1]]:.4f}")

print(f"\nDIM Model (8 Hz + 12 Hz, switching coupling):")
print(f"  State recovery |corr|: {best_corr_dim:.3f}")
print(f"  Osc1 tracking corr: {osc1_corr:.3f}")
print(f"  Osc2 tracking corr: {osc2_corr:.3f}")
print(f"  Coupling recovery: uncoupled={coupling_strength[order_dim[0]]:.4f}, coupled={coupling_strength[order_dim[1]]:.4f}")
