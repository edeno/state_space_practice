"""Correctness verification plots for the switching point-process model.

Demonstrates that the implementation produces correct results by comparing
against known analytical values in controlled scenarios.
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Output directory
if Path.cwd().name == "notebooks":
    project_root = Path.cwd().parent
else:
    project_root = Path.cwd()
output_dir = project_root / "output" / "correctness"
output_dir.mkdir(parents=True, exist_ok=True)

jax.config.update("jax_enable_x64", True)

from state_space_practice.switching_kalman import switching_kalman_smoother
from state_space_practice.switching_point_process import (
    QRegularizationConfig,
    SpikeObsParams,
    SwitchingSpikeOscillatorModel,
    switching_point_process_filter,
)

# %% [markdown]
# # 1. Known Log-Likelihood: Filter LL Matches HMM Forward Algorithm
#
# With Q≈0 and zero weights, the model reduces to a pure HMM with Poisson
# emissions. We compute the true marginal log-likelihood via an independent
# HMM forward algorithm and verify the switching filter produces the same value.

# %%
n_time, n_neurons, n_latent, dt = 500, 4, 2, 0.01

# Two very distinct emission rates
baseline_0 = jnp.zeros(n_neurons)        # exp(0) = 1 Hz
baseline_1 = jnp.ones(n_neurons) * 3.4   # exp(3.4) ≈ 30 Hz
weights = jnp.zeros((n_neurons, n_latent))

spike_params = SpikeObsParams(
    baseline=jnp.stack([baseline_0, baseline_1], axis=-1),
    weights=jnp.stack([weights, weights], axis=-1),
)

# Known discrete state sequence: 125-step blocks
true_disc = np.concatenate([
    np.zeros(125, dtype=int), np.ones(125, dtype=int),
    np.zeros(125, dtype=int), np.ones(125, dtype=int),
])

rates_0 = jnp.exp(baseline_0) * dt
rates_1 = jnp.exp(baseline_1) * dt
true_rates = jnp.where(jnp.array(true_disc)[:, None] == 0, rates_0, rates_1)

key = jax.random.PRNGKey(42)
spikes = jax.random.poisson(key, true_rates).astype(float)

Z = jnp.array([[0.97, 0.03], [0.03, 0.97]])

# Independent HMM forward algorithm
log_alpha = np.zeros((n_time, 2))
for j in range(2):
    r = rates_0 if j == 0 else rates_1
    log_alpha[0, j] = np.log(0.5) + float(
        jnp.sum(jax.scipy.stats.poisson.logpmf(spikes[0], r))
    )
for t in range(1, n_time):
    for j in range(2):
        r = rates_0 if j == 0 else rates_1
        obs_ll = float(jnp.sum(jax.scipy.stats.poisson.logpmf(spikes[t], r)))
        log_trans = [
            log_alpha[t - 1, i] + np.log(float(Z[i, j])) for i in range(2)
        ]
        log_alpha[t, j] = obs_ll + np.logaddexp(log_trans[0], log_trans[1])

true_marginal_ll = float(np.logaddexp(log_alpha[-1, 0], log_alpha[-1, 1]))

# Cumulative LL from HMM forward
hmm_cumulative_ll = np.zeros(n_time)
for t in range(n_time):
    hmm_cumulative_ll[t] = float(np.logaddexp(log_alpha[t, 0], log_alpha[t, 1]))

# HMM forward posterior: P(S_t=j | y_{1:t})
hmm_filter_prob = np.zeros((n_time, 2))
for t in range(n_time):
    log_norm = np.logaddexp(log_alpha[t, 0], log_alpha[t, 1])
    hmm_filter_prob[t, 0] = np.exp(log_alpha[t, 0] - log_norm)
    hmm_filter_prob[t, 1] = np.exp(log_alpha[t, 1] - log_norm)

# Run switching point-process filter
def log_intensity_func(state, params):
    return params.baseline + params.weights @ state

init_mean = jnp.zeros((n_latent, 2))
init_cov = jnp.stack([jnp.eye(n_latent) * 1e-10] * 2, axis=-1)
A = jnp.stack([jnp.eye(n_latent)] * 2, axis=-1)
Q = jnp.stack([jnp.eye(n_latent) * 1e-10] * 2, axis=-1)

fm, fc, fp, lpm, filter_ll = switching_point_process_filter(
    init_mean, init_cov, jnp.array([0.5, 0.5]), spikes,
    Z, A, Q, dt, log_intensity_func, spike_params,
    include_laplace_normalization=False,
)

# Smoother
(_, _, sdsp, _, _, scsm, _, _, _) = switching_kalman_smoother(
    filter_mean=fm, filter_cov=fc, filter_discrete_state_prob=fp,
    last_filter_conditional_cont_mean=lpm, process_cov=Q,
    continuous_transition_matrix=A, discrete_state_transition_matrix=Z,
)

filter_prob = np.array(fp)
smoother_prob = np.array(sdsp)

# %%
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Panel 1: Spike raster
ax = axes[0]
for n in range(n_neurons):
    spike_times = np.where(np.array(spikes[:, n]) > 0)[0]
    ax.scatter(spike_times, np.ones_like(spike_times) * n, s=1, c="k", alpha=0.5)
# Shade true states
for t in range(n_time - 1):
    if true_disc[t] == 1:
        ax.axvspan(t, t + 1, alpha=0.1, color="C1")
ax.set_ylabel("Neuron")
ax.set_title("Spike raster (orange = high-rate state)")
ax.set_yticks(range(n_neurons))

# Panel 2: Filter vs HMM forward discrete state probabilities
ax = axes[1]
ax.plot(filter_prob[:, 1], label="Switching filter P(S=1)", alpha=0.8)
ax.plot(hmm_filter_prob[:, 1], "--", label="HMM forward P(S=1)", alpha=0.8)
ax.fill_between(range(n_time), 0, true_disc, alpha=0.15, color="C1", label="True state")
ax.set_ylabel("P(S=1 | y)")
ax.set_title("Filter discrete state probabilities: switching filter vs HMM forward")
ax.legend(loc="upper right")
ax.set_ylim(-0.05, 1.05)

# Panel 3: Smoother discrete state probabilities
ax = axes[2]
ax.plot(smoother_prob[:, 1], label="Smoother P(S=1)", color="C2", alpha=0.8)
ax.plot(filter_prob[:, 1], "--", label="Filter P(S=1)", color="C0", alpha=0.5)
ax.fill_between(range(n_time), 0, true_disc, alpha=0.15, color="C1", label="True state")
ax.set_ylabel("P(S=1 | y)")
ax.set_title("Smoother improves on filter (uses future observations)")
ax.legend(loc="upper right")
ax.set_ylim(-0.05, 1.05)

# Panel 4: Filter vs HMM forward probability difference
ax = axes[3]
diff = filter_prob[:, 1] - hmm_filter_prob[:, 1]
ax.plot(diff, color="C3", alpha=0.8)
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_ylabel("Difference")
ax.set_xlabel("Time step")
ax.set_title(
    f"Filter - HMM forward difference (max |diff| = {np.max(np.abs(diff)):.2e}, "
    f"LL diff = {abs(float(filter_ll) - true_marginal_ll):.2e})"
)

plt.tight_layout()
plt.savefig(output_dir / "correctness_1_hmm_forward.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Filter LL:       {float(filter_ll):.4f}")
print(f"HMM forward LL:  {true_marginal_ll:.4f}")
print(f"Difference:      {abs(float(filter_ll) - true_marginal_ll):.6f}")

# %% [markdown]
# # 2. Known State: Filter LL Matches Poisson Log-PMF
#
# With a known fixed latent state (Q≈0), the filter should compute the
# exact same log-likelihood as directly evaluating the Poisson log-pmf.

# %%
true_state = jnp.array([0.5, -0.3])

baseline_single = jnp.array([1.0, 2.0, 1.5, 2.5])
weights_single = jnp.array([
    [0.5, 0.3], [-0.2, 0.4], [0.1, -0.5], [0.3, 0.2],
])
spike_params_single = SpikeObsParams(baseline=baseline_single, weights=weights_single)

true_log_rates = baseline_single + weights_single @ true_state
true_rates_single = jnp.exp(true_log_rates) * dt

n_time_single = 1000
spikes_single = jax.random.poisson(
    jax.random.PRNGKey(42), true_rates_single[None, :],
    shape=(n_time_single, n_neurons),
).astype(float)

# True LL
true_ll_single = float(jnp.sum(
    jax.scipy.stats.poisson.logpmf(spikes_single, true_rates_single[None, :])
))

# Run filter at true state
init_mean_s = true_state[:, None]
init_cov_s = jnp.eye(n_latent)[..., None] * 1e-10
A_s = jnp.eye(n_latent)[..., None]
Q_s = jnp.eye(n_latent)[..., None] * 1e-10

fm_s, fc_s, _, _, filter_ll_s = switching_point_process_filter(
    init_mean_s, init_cov_s, jnp.array([1.0]), spikes_single,
    jnp.array([[1.0]]), A_s, Q_s, dt, log_intensity_func, spike_params_single,
    include_laplace_normalization=False,
)

# Per-timestep LL comparison
per_step_true_ll = np.array(jnp.sum(
    jax.scipy.stats.poisson.logpmf(spikes_single, true_rates_single[None, :]),
    axis=1,
))
cumulative_true_ll = np.cumsum(per_step_true_ll)

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Panel 1: Filter state vs true state
ax = axes[0]
fm_np = np.array(fm_s[:, :, 0])
ax.plot(fm_np[:, 0], label=f"Filter x[0] (true={float(true_state[0]):.2f})", alpha=0.8)
ax.plot(fm_np[:, 1], label=f"Filter x[1] (true={float(true_state[1]):.2f})", alpha=0.8)
ax.axhline(float(true_state[0]), color="C0", linestyle="--", alpha=0.5)
ax.axhline(float(true_state[1]), color="C1", linestyle="--", alpha=0.5)
ax.set_ylabel("Latent state")
ax.set_title("Filter state stays at true value (Q ≈ 0)")
ax.legend()

# Panel 2: Spike rates
ax = axes[1]
for n in range(n_neurons):
    empirical_rate = np.convolve(
        np.array(spikes_single[:, n]), np.ones(50) / 50, mode="same"
    ) / dt
    ax.plot(empirical_rate, alpha=0.4, label=f"Neuron {n}" if n < 2 else None)
    ax.axhline(float(jnp.exp(true_log_rates[n])), color=f"C{n}", linestyle="--", alpha=0.5)
ax.set_ylabel("Rate (Hz)")
ax.set_title("Observed spike rates vs true rates (dashed)")
ax.legend()

# Panel 3: LL comparison
ax = axes[2]
ax.text(
    0.5, 0.5,
    f"Filter LL = {float(filter_ll_s):.4f}\n"
    f"True LL   = {true_ll_single:.4f}\n"
    f"Difference = {abs(float(filter_ll_s) - true_ll_single):.6f}\n\n"
    f"Filter mean error = {float(jnp.max(jnp.abs(fm_s[:, :, 0] - true_state))):.2e}",
    transform=ax.transAxes, fontsize=14, verticalalignment="center",
    horizontalalignment="center",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
)
ax.set_axis_off()
ax.set_title("Log-likelihood verification")

plt.tight_layout()
plt.savefig(output_dir / "correctness_2_known_state.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # 3. Warm-Started Parameter Recovery
#
# Simulate from a switching model with well-separated dynamics, initialize
# near the truth, and verify EM refines the parameters.

# %%
from state_space_practice.simulate.simulate_switching_spikes import (
    simulate_switching_spike_oscillator,
)

n_time_r, n_neurons_r, n_latent_r, dt_r = 2000, 10, 2, 0.01
key_r = jax.random.PRNGKey(3)
k_sim, k_fit = jax.random.split(key_r)

A0_true = jnp.eye(n_latent_r) * 0.5
A1_true = jnp.eye(n_latent_r) * 0.98
A_true = jnp.stack([A0_true, A1_true], axis=-1)
Q_true = jnp.stack([jnp.eye(n_latent_r) * 0.1] * 2, axis=-1)
Z_true = jnp.array([[0.98, 0.02], [0.02, 0.98]])

k_w, k_s = jax.random.split(k_sim)
W_true = jax.random.normal(k_w, (n_neurons_r, n_latent_r)) * 0.8
b_true = jnp.ones(n_neurons_r) * 3.0

spikes_r, true_states_r, true_disc_r = simulate_switching_spike_oscillator(
    n_time=n_time_r, transition_matrices=A_true, process_covs=Q_true,
    discrete_transition_matrix=Z_true, spike_weights=W_true,
    spike_baseline=b_true, dt=dt_r, key=k_s,
)

# Fit with warm start
model_r = SwitchingSpikeOscillatorModel(
    n_oscillators=1, n_neurons=n_neurons_r, n_discrete_states=2,
    sampling_freq=100.0, dt=dt_r,
    q_regularization=QRegularizationConfig(enabled=False),
    separate_spike_params=False,
)
model_r._initialize_parameters(jax.random.PRNGKey(0))
model_r.continuous_transition_matrix = A_true + jax.random.normal(k_fit, A_true.shape) * 0.05
model_r.process_cov = Q_true + jnp.abs(jax.random.normal(k_fit, Q_true.shape)) * 0.01
model_r.discrete_transition_matrix = Z_true
model_r.spike_params = SpikeObsParams(
    baseline=b_true + jax.random.normal(jax.random.PRNGKey(1), b_true.shape) * 0.1,
    weights=W_true + jax.random.normal(jax.random.PRNGKey(2), W_true.shape) * 0.1,
)

lls_r = model_r.fit(spikes_r, max_iter=30, skip_init=True)

prob_r = np.array(model_r.smoother_discrete_state_prob)
corr_r = [np.corrcoef(np.array(true_disc_r, float), prob_r[:, j])[0, 1] for j in range(2)]
best_j = np.argmax(np.abs(corr_r))
state1_label = best_j if corr_r[best_j] > 0 else 1 - best_j

sr_fitted = [
    float(jnp.max(jnp.abs(jnp.linalg.eigvals(
        model_r.continuous_transition_matrix[:, :, j]
    ))))
    for j in range(2)
]

# %%
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Panel 1: LL convergence
ax = axes[0]
ax.plot(lls_r, "o-", markersize=3)
ax.set_ylabel("Log-likelihood")
ax.set_xlabel("EM iteration")
ax.set_title(f"EM convergence (improvement = {lls_r[-1] - lls_r[0]:.1f})")

# Panel 2: True vs inferred discrete states
ax = axes[1]
ax.fill_between(range(n_time_r), 0, np.array(true_disc_r), alpha=0.2, color="C1",
                label="True state 1")
ax.plot(prob_r[:, state1_label], alpha=0.7, label=f"P(S={state1_label} | y)")
ax.set_ylabel("P(high-persistence state)")
ax.set_title(f"Discrete state recovery (|corr| = {max(np.abs(corr_r)):.3f})")
ax.legend()
ax.set_ylim(-0.05, 1.05)

# Panel 3: True vs smoothed latent state (first dimension)
ax = axes[2]
smoother_mean_r = np.array(jnp.einsum(
    "tls,ts->tl",
    model_r.smoother_state_cond_mean,
    model_r.smoother_discrete_state_prob,
))
ax.plot(np.array(true_states_r[:, 0]), alpha=0.4, label="True x[0]")
ax.plot(smoother_mean_r[:, 0], alpha=0.7, label="Smoothed x[0]")
ax.set_ylabel("Latent state")
ax.set_title("Latent state tracking")
ax.legend()

# Panel 4: Parameter recovery summary
ax = axes[3]
Z_fit = model_r.discrete_transition_matrix
text = (
    f"Spectral radii:  true = [0.50, 0.98],  fitted = [{min(sr_fitted):.3f}, {max(sr_fitted):.3f}]\n"
    f"Z diagonal:      true = [0.98, 0.98],  fitted = [{float(Z_fit[0,0]):.3f}, {float(Z_fit[1,1]):.3f}]\n"
    f"Baseline error:  {float(jnp.max(jnp.abs(model_r.spike_params.baseline - b_true))):.3f} "
    f"(true baseline = {float(b_true[0]):.1f})\n"
    f"Weights MAE:     {float(jnp.mean(jnp.abs(model_r.spike_params.weights - W_true))):.3f}"
)
ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=12,
        verticalalignment="center", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_axis_off()
ax.set_title("Parameter recovery (warm-started near truth)")

plt.tight_layout()
plt.savefig(output_dir / "correctness_3_parameter_recovery.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # 4. Discrete State Segmentation from Firing Rates
#
# States with very different firing rates (1 Hz vs 30 Hz) should be
# perfectly segmented by the model with per-state spike parameters.

# %%
n_time_d = 500
n_neurons_d = 5

true_disc_d = np.concatenate([
    np.zeros(125, dtype=int), np.ones(125, dtype=int),
    np.zeros(125, dtype=int), np.ones(125, dtype=int),
])

low_rate = 1.0 * 0.01   # 1 Hz
high_rate = 30.0 * 0.01  # 30 Hz
rates_d = np.where(
    true_disc_d[:, None] == 0,
    low_rate * np.ones((n_time_d, n_neurons_d)),
    high_rate * np.ones((n_time_d, n_neurons_d)),
)

spikes_d = jax.random.poisson(
    jax.random.PRNGKey(7), jnp.array(rates_d)
).astype(float)

model_d = SwitchingSpikeOscillatorModel(
    n_oscillators=1, n_neurons=n_neurons_d, n_discrete_states=2,
    sampling_freq=100.0, dt=0.01,
    q_regularization=QRegularizationConfig(),
    separate_spike_params=True,
    update_continuous_transition_matrix=False,
    update_process_cov=False,
    update_init_mean=False,
    update_init_cov=False,
)

# Try seeds
for seed in range(20):
    try:
        lls_d = model_d.fit(spikes_d, max_iter=30, key=jax.random.PRNGKey(seed))
        break
    except ValueError:
        continue

prob_d = np.array(model_d.smoother_discrete_state_prob)
corr_d = [np.corrcoef(true_disc_d.astype(float), prob_d[:, j])[0, 1] for j in range(2)]
best_j_d = np.argmax(np.abs(corr_d))
high_state = best_j_d if corr_d[best_j_d] > 0 else 1 - best_j_d

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Panel 1: Spike raster
ax = axes[0]
total_spikes_per_bin = np.array(jnp.sum(spikes_d, axis=1))
ax.bar(range(n_time_d), total_spikes_per_bin, width=1.0, color="gray", alpha=0.6)
for t in range(n_time_d - 1):
    if true_disc_d[t] == 1:
        ax.axvspan(t, t + 1, alpha=0.1, color="C1")
ax.set_ylabel("Total spike count")
ax.set_title("Total spikes per bin (orange = high-rate state)")

# Panel 2: Inferred states
ax = axes[1]
ax.fill_between(range(n_time_d), 0, true_disc_d, alpha=0.2, color="C1",
                label="True high-rate state")
ax.plot(prob_d[:, high_state], color="C0", alpha=0.8,
        label=f"P(high-rate | y)")
ax.set_ylabel("Probability")
ax.set_title(f"State recovery (|corr| = {max(np.abs(corr_d)):.3f})")
ax.legend()
ax.set_ylim(-0.05, 1.05)

# Panel 3: LL convergence
ax = axes[2]
ax.plot(lls_d, "o-", markersize=3)
ax.set_ylabel("Log-likelihood")
ax.set_xlabel("Time step")
ax.set_title(f"EM convergence ({len(lls_d)} iterations)")

plt.tight_layout()
plt.savefig(output_dir / "correctness_4_state_segmentation.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"State recovery |corr|: {max(np.abs(corr_d)):.3f}")
b_fitted = np.array(model_d.spike_params.baseline)
print(f"Fitted baselines: state 0 mean = {b_fitted[:, 1-high_state].mean():.2f} "
      f"(true ≈ {np.log(1.0):.2f}), "
      f"state 1 mean = {b_fitted[:, high_state].mean():.2f} "
      f"(true ≈ {np.log(30.0):.2f})")
