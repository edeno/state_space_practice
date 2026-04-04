"""Systematic evaluation of switching spike oscillator on CA1 data.

Compares:
1. GPB1 vs GPB2 smoother
2. Conservative (0.95/0.99) vs strong (0.80/0.999) damping contrast
3. Fixed vs learned dynamics (A, Q)
"""
# %%
from pathlib import Path
import time

import jax
import jax.numpy as jnp
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

# %%
with open("data/ca1_preprocessed_250Hz.pkl", "rb") as f:
    data = pickle.load(f)

spikes = jnp.array(data["binned_spikes"])
speed = data["speed"]
labels = data["behavioral_labels"]
sampling_freq = data["sampling_freq"]
dt = data["dt"]
n_neurons = data["n_neurons"]
n_time = data["n_time"]

print(f"Data: {n_time} steps ({n_time*dt:.0f}s), {n_neurons} neurons, {sampling_freq} Hz")

# Empirical transition matrix
valid = labels[labels != 2]
n_im = np.sum(valid[:-1] == 0)
n_run = np.sum(valid[:-1] == 1)
p_im_stay = (np.sum((valid[:-1] == 0) & (valid[1:] == 0)) + 1) / (n_im + 2)
p_run_stay = (np.sum((valid[:-1] == 1) & (valid[1:] == 1)) + 1) / (n_run + 2)
running_frac = float((labels == 1).mean())

Z_empirical = jnp.array([
    [p_im_stay, 1 - p_im_stay],
    [1 - p_run_stay, p_run_stay],
])
init_prob = jnp.array([1 - running_frac, running_frac])

# Data-adaptive baseline
mean_counts = np.array(jnp.mean(spikes, axis=0))
empirical_baseline = np.log(mean_counts / dt + 1e-10)

print(f"Empirical: P(stay|immobile)={p_im_stay:.4f}, P(stay|run)={p_run_stay:.4f}")

# %%
theta_freq = jnp.array([8.0])


def build_model(damping_off, damping_on, q_off, q_on,
                smoother_type="gpb1", update_dynamics=False,
                max_em_iter=20):
    """Build and fit a model with given parameters."""
    A_off = construct_common_oscillator_transition_matrix(
        theta_freq, jnp.array([damping_off]), sampling_freq
    )
    A_on = construct_common_oscillator_transition_matrix(
        theta_freq, jnp.array([damping_on]), sampling_freq
    )
    Q_off = construct_common_oscillator_process_covariance(jnp.array([q_off]))
    Q_on = construct_common_oscillator_process_covariance(jnp.array([q_on]))

    model = SwitchingSpikeOscillatorModel(
        n_oscillators=1,
        n_neurons=n_neurons,
        n_discrete_states=2,
        sampling_freq=sampling_freq,
        dt=dt,
        q_regularization=QRegularizationConfig(),
        separate_spike_params=True,
        spike_weight_l2=0.05,
        update_continuous_transition_matrix=update_dynamics,
        update_process_cov=update_dynamics,
        update_init_mean=False,
        update_init_cov=False,
        update_discrete_transition_matrix=False,
        smoother_type=smoother_type,
    )

    key = jax.random.PRNGKey(42)
    model._initialize_parameters(key)
    model.continuous_transition_matrix = jnp.stack([A_off, A_on], axis=-1)
    model.process_cov = jnp.stack([Q_off, Q_on], axis=-1)
    model.init_cov = jnp.stack([jnp.eye(2) * 0.5, jnp.eye(2) * 0.5], axis=-1)
    model.discrete_transition_matrix = Z_empirical
    model.init_discrete_state_prob = init_prob

    model.spike_params = SpikeObsParams(
        baseline=jnp.stack([
            jnp.array(empirical_baseline),
            jnp.array(empirical_baseline),
        ], axis=-1),
        weights=jax.random.normal(key, (n_neurons, 2, 2)) * 0.01,
    )

    t0 = time.time()
    lls = model.fit(spikes, max_iter=max_em_iter, skip_init=True)
    elapsed = time.time() - t0

    return model, lls, elapsed


def evaluate_model(model, label=""):
    """Evaluate model against behavioral labels."""
    prob = np.array(model.smoother_discrete_state_prob)
    running_mask = (labels == 1).astype(float)
    clear_mask = (labels == 0) | (labels == 1)

    corr = [np.corrcoef(running_mask, prob[:, j])[0, 1] for j in range(2)]
    theta_on = np.argmax(corr)

    auc = roc_auc_score(running_mask[clear_mask], prob[clear_mask, theta_on])

    # Spectral properties
    specs = []
    for j in range(2):
        eigs = np.linalg.eigvals(np.array(model.continuous_transition_matrix[:, :, j]))
        sr = np.max(np.abs(eigs))
        freq = np.abs(np.angle(eigs[0])) * sampling_freq / (2 * np.pi)
        specs.append((sr, freq))

    return {
        "label": label,
        "auc": auc,
        "theta_on": int(theta_on),
        "occupancy": float(prob[:, theta_on].mean()),
        "corr": float(corr[theta_on]),
        "specs": specs,
        "lls": list(model.fit_log_likelihoods) if hasattr(model, "fit_log_likelihoods") else [],
    }


# %% [markdown]
# ## 1. Does the model still work with merged code?

# %%
print("=" * 70)
print("1. BASELINE: Conservative damping (0.95/0.99), GPB1, fixed dynamics")
print("=" * 70)
model_base, lls_base, t_base = build_model(
    damping_off=0.95, damping_on=0.99, q_off=0.01, q_on=0.02,
    smoother_type="gpb1", update_dynamics=False,
)
r_base = evaluate_model(model_base, "conservative/gpb1/fixed")
print(f"  AUC: {r_base['auc']:.3f}")
print(f"  LL: {lls_base[0]:.0f} -> {lls_base[-1]:.0f}")
print(f"  Time: {t_base:.1f}s")
print(f"  Theta-on=state {r_base['theta_on']}, occupancy={r_base['occupancy']:.3f}")
for j, (sr, freq) in enumerate(r_base['specs']):
    print(f"  State {j}: sr={sr:.4f}, freq={freq:.1f} Hz")

# %% [markdown]
# ## 2. Strong damping contrast

# %%
print("\n" + "=" * 70)
print("2. STRONG DAMPING: (0.80/0.999), GPB1, fixed dynamics")
print("=" * 70)
model_strong, lls_strong, t_strong = build_model(
    damping_off=0.80, damping_on=0.999, q_off=0.01, q_on=0.02,
    smoother_type="gpb1", update_dynamics=False,
)
r_strong = evaluate_model(model_strong, "strong/gpb1/fixed")
print(f"  AUC: {r_strong['auc']:.3f}")
print(f"  LL: {lls_strong[0]:.0f} -> {lls_strong[-1]:.0f}")
print(f"  Time: {t_strong:.1f}s")
print(f"  Theta-on=state {r_strong['theta_on']}, occupancy={r_strong['occupancy']:.3f}")
for j, (sr, freq) in enumerate(r_strong['specs']):
    print(f"  State {j}: sr={sr:.4f}, freq={freq:.1f} Hz")

# %% [markdown]
# ## 3. GPB2 vs GPB1

# %%
print("\n" + "=" * 70)
print("3. GPB2: Strong damping (0.80/0.999), GPB2, fixed dynamics")
print("=" * 70)
model_gpb2, lls_gpb2, t_gpb2 = build_model(
    damping_off=0.80, damping_on=0.999, q_off=0.01, q_on=0.02,
    smoother_type="gpb2", update_dynamics=False,
)
r_gpb2 = evaluate_model(model_gpb2, "strong/gpb2/fixed")
print(f"  AUC: {r_gpb2['auc']:.3f}")
print(f"  LL: {lls_gpb2[0]:.0f} -> {lls_gpb2[-1]:.0f}")
print(f"  Time: {t_gpb2:.1f}s")
print(f"  Theta-on=state {r_gpb2['theta_on']}, occupancy={r_gpb2['occupancy']:.3f}")
for j, (sr, freq) in enumerate(r_gpb2['specs']):
    print(f"  State {j}: sr={sr:.4f}, freq={freq:.1f} Hz")

# %% [markdown]
# ## 4. Can we learn dynamics?

# %%
print("\n" + "=" * 70)
print("4. LEARNED DYNAMICS: Strong damping (0.80/0.999), GPB1, learn A+Q")
print("=" * 70)
model_learn, lls_learn, t_learn = build_model(
    damping_off=0.80, damping_on=0.999, q_off=0.01, q_on=0.02,
    smoother_type="gpb1", update_dynamics=True,
)
r_learn = evaluate_model(model_learn, "strong/gpb1/learned")
print(f"  AUC: {r_learn['auc']:.3f}")
print(f"  LL: {lls_learn[0]:.0f} -> {lls_learn[-1]:.0f}")
print(f"  Time: {t_learn:.1f}s")
print(f"  Theta-on=state {r_learn['theta_on']}, occupancy={r_learn['occupancy']:.3f}")
for j, (sr, freq) in enumerate(r_learn['specs']):
    print(f"  State {j}: sr={sr:.4f}, freq={freq:.1f} Hz")

# %% [markdown]
# ## Summary

# %%
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Config':<40} {'AUC':>6} {'Time':>7} {'LL_final':>12}")
print("-" * 70)
for r, lls, t in [
    (r_base, lls_base, t_base),
    (r_strong, lls_strong, t_strong),
    (r_gpb2, lls_gpb2, t_gpb2),
    (r_learn, lls_learn, t_learn),
]:
    print(f"{r['label']:<40} {r['auc']:>6.3f} {t:>6.1f}s {lls[-1]:>12.0f}")
