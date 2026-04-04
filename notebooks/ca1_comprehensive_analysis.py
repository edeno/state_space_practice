"""Comprehensive analysis of switching spike oscillator on CA1 theta data.

Evaluates whether the model captures genuine theta oscillatory structure
beyond simple rate-based state segmentation.
"""
# %%
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time

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

with open("data/ca1_preprocessed_250Hz.pkl", "rb") as f:
    data = pickle.load(f)

spikes = jnp.array(data["binned_spikes"])
labels = data["behavioral_labels"]
speed = data["speed"]
sf, dt, nn = data["sampling_freq"], data["dt"], data["n_neurons"]
n_time = data["n_time"]
time_bins = data["time_bins"]
theta_freq = jnp.array([8.0])

spikes_np = np.array(spikes)
rates = spikes_np.sum(0) / (n_time * dt)

# Empirical transitions and baselines
valid = labels[labels != 2]
p_im_stay = (np.sum((valid[:-1] == 0) & (valid[1:] == 0)) + 1) / (np.sum(valid[:-1] == 0) + 2)
p_run_stay = (np.sum((valid[:-1] == 1) & (valid[1:] == 1)) + 1) / (np.sum(valid[:-1] == 1) + 2)
running_frac = float((labels == 1).mean())
baseline_imm = np.log(spikes_np[labels == 0].sum(0) / (np.sum(labels == 0) * dt) + 1e-10)
baseline_run = np.log(spikes_np[labels == 1].sum(0) / (np.sum(labels == 1) * dt) + 1e-10)

print(f"Data: {n_time} steps ({n_time*dt:.0f}s), {nn} neurons, {sf} Hz")
print(f"Running fraction: {running_frac:.3f}")
print(f"Firing rates: median={np.median(rates):.1f} Hz, mean={rates.mean():.1f} Hz")

# %% Fit model
A_off = construct_common_oscillator_transition_matrix(theta_freq, jnp.array([0.95]), sf)
A_on = construct_common_oscillator_transition_matrix(theta_freq, jnp.array([0.99]), sf)
Q_off = construct_common_oscillator_process_covariance(jnp.array([0.01]))
Q_on = construct_common_oscillator_process_covariance(jnp.array([0.02]))

model = SwitchingSpikeOscillatorModel(
    n_oscillators=1, n_neurons=nn, n_discrete_states=2,
    sampling_freq=sf, dt=dt, q_regularization=QRegularizationConfig(),
    separate_spike_params=True, spike_weight_l2=100.0,
    update_continuous_transition_matrix=False, update_process_cov=False,
    update_init_mean=False, update_init_cov=False,
    update_discrete_transition_matrix=False,
)
key = jax.random.PRNGKey(42)
model._initialize_parameters(key)
model.continuous_transition_matrix = jnp.stack([A_off, A_on], axis=-1)
model.process_cov = jnp.stack([Q_off, Q_on], axis=-1)
model.init_cov = jnp.stack([jnp.eye(2) * 0.5] * 2, axis=-1)
model.discrete_transition_matrix = jnp.array([
    [p_im_stay, 1 - p_im_stay], [1 - p_run_stay, p_run_stay]
])
model.init_discrete_state_prob = jnp.array([1 - running_frac, running_frac])
model.spike_params = SpikeObsParams(
    baseline=jnp.stack([jnp.array(baseline_imm), jnp.array(baseline_run)], axis=-1),
    weights=jax.random.normal(key, (nn, 2, 2)) * 0.01,
)

t0 = time.time()
lls = model.fit(spikes, max_iter=50, tol=1e-10, skip_init=True)
elapsed = time.time() - t0
print(f"\nFit: {elapsed:.0f}s, {len(lls)} iterations")
print(f"LL: {lls[0]:.0f} -> {lls[-1]:.0f}")

# %% Extract results
prob = np.array(model.smoother_discrete_state_prob)
running_mask = (labels == 1).astype(float)
clear_mask = (labels == 0) | (labels == 1)
corr = [np.corrcoef(running_mask, prob[:, j])[0, 1] for j in range(2)]
theta_on = np.argmax(corr)
theta_off = 1 - theta_on

auc = roc_auc_score(running_mask[clear_mask], prob[clear_mask, theta_on])
pred = (prob[:, theta_on] > 0.5).astype(float)
bacc = balanced_accuracy_score(running_mask[clear_mask], pred[clear_mask])

sm = np.array(model.smoother_state_cond_mean)  # (T, L, S)
sp = np.array(model.smoother_discrete_state_prob)
overall_mean = np.einsum("tls,ts->tl", sm, sp)
amplitude = np.sqrt(overall_mean[:, 0] ** 2 + overall_mean[:, 1] ** 2)
phase = np.arctan2(overall_mean[:, 1], overall_mean[:, 0])

w = np.array(model.spike_params.weights)  # (nn, 2, S)
b = np.array(model.spike_params.baseline)  # (nn, S)
w_norm_on = np.sqrt(np.sum(w[:, :, theta_on] ** 2, axis=1))
w_norm_off = np.sqrt(np.sum(w[:, :, theta_off] ** 2, axis=1))
w_angle_on = np.degrees(np.arctan2(w[:, 1, theta_on], w[:, 0, theta_on]))

print(f"\n{'='*70}")
print(f"1. SEGMENTATION QUALITY")
print(f"{'='*70}")
print(f"Theta-on = state {theta_on}")
print(f"AUC vs running: {auc:.3f}")
print(f"Balanced accuracy: {bacc:.3f}")
print(f"Theta-on occupancy: {prob[:, theta_on].mean():.3f}")

print(f"\n{'='*70}")
print(f"2. WEIGHT ANALYSIS — IS THE OSCILLATOR BEING USED?")
print(f"{'='*70}")
print(f"Theta-on state: {np.sum(w_norm_on > 0.1)}/{nn} neurons with |w|>0.1, mean={w_norm_on.mean():.3f}, max={w_norm_on.max():.3f}")
print(f"Theta-off state: {np.sum(w_norm_off > 0.1)}/{nn} neurons with |w|>0.1, mean={w_norm_off.mean():.3f}, max={w_norm_off.max():.3f}")
print(f"\nWeight norms vs firing rate:")
for rate_thresh in [0.5, 2, 5, 10]:
    mask = rates > rate_thresh
    if mask.sum() > 0:
        print(f"  Neurons >{rate_thresh} Hz ({mask.sum()}): |w|_on mean={w_norm_on[mask].mean():.3f}, |w|_off mean={w_norm_off[mask].mean():.3f}")

print(f"\n{'='*70}")
print(f"3. PREFERRED THETA PHASE — DO NEURONS HAVE DIVERSE PHASES?")
print(f"{'='*70}")
# Neurons with significant coupling
sig_mask = w_norm_on > 0.1
sig_angles = w_angle_on[sig_mask]
print(f"Phase distribution of {sig_mask.sum()} significantly coupled neurons:")
for lo, hi, name in [(-180, -135, "trough-to-rise"), (-135, -90, "rising"),
                      (-90, -45, "rise-to-peak"), (-45, 0, "peak"),
                      (0, 45, "peak-to-fall"), (45, 90, "falling"),
                      (90, 135, "fall-to-trough"), (135, 180, "trough")]:
    count = np.sum((sig_angles >= lo) & (sig_angles < hi))
    print(f"  {name:>15} [{lo:>4},{hi:>4}): {count} neurons")

# Circular statistics
sig_angles_rad = np.radians(sig_angles)
mean_vec = np.abs(np.mean(np.exp(1j * sig_angles_rad)))
mean_angle = np.degrees(np.angle(np.mean(np.exp(1j * sig_angles_rad))))
print(f"\nMean resultant length (0=uniform, 1=concentrated): {mean_vec:.3f}")
print(f"Mean preferred phase: {mean_angle:.0f} deg")
print(f"(If uniform ~0.1, this means neurons fire at diverse theta phases)")

print(f"\n{'='*70}")
print(f"4. OSCILLATION AMPLITUDE — DOES IT TRACK BEHAVIOR?")
print(f"{'='*70}")
amp_run = amplitude[labels == 1]
amp_imm = amplitude[labels == 0]
print(f"Amplitude during running: {amp_run.mean():.4f} ± {amp_run.std():.4f}")
print(f"Amplitude during immobile: {amp_imm.mean():.4f} ± {amp_imm.std():.4f}")
print(f"Ratio (run/imm): {amp_run.mean() / (amp_imm.mean() + 1e-10):.2f}")
print(f"Speed-amplitude correlation: {np.corrcoef(speed, amplitude)[0,1]:.3f}")
print(f"Speed-P(theta_on) correlation: {np.corrcoef(speed, prob[:,theta_on])[0,1]:.3f}")

# Per-state amplitude
for j in range(2):
    amp_j = np.sqrt(sm[:, 0, j] ** 2 + sm[:, 1, j] ** 2)
    name = "theta-on" if j == theta_on else "theta-off"
    print(f"State {j} ({name}) oscillator amplitude: mean={amp_j.mean():.4f}, max={amp_j.max():.4f}")

print(f"\n{'='*70}")
print(f"5. SANITY CHECKS")
print(f"{'='*70}")

# Check: do baselines match expected rates?
for j in range(2):
    name = "theta-on" if j == theta_on else "theta-off"
    pred_rate = np.exp(b[:, j]) / dt
    actual_label = 1 if j == theta_on else 0
    actual_rate = spikes_np[labels == actual_label].sum(0) / (np.sum(labels == actual_label) * dt)
    rate_corr = np.corrcoef(pred_rate, actual_rate)[0, 1]
    print(f"State {j} ({name}): baseline-predicted vs actual rate corr = {rate_corr:.3f}")

# Check: is the oscillator frequency correct?
# ACF of oscillator during theta-on periods
theta_on_mask = prob[:, theta_on] > 0.5
x_theta = overall_mean[theta_on_mask, 0][:10000]
x_theta = x_theta - x_theta.mean()
if len(x_theta) > 1000 and x_theta.std() > 1e-10:
    acf = np.correlate(x_theta, x_theta, mode="full")
    mid = len(acf) // 2
    acf_norm = acf / (acf[mid] + 1e-10)
    # Find first peak after lag 0
    theta_lag = int(sf / 8)  # expected lag for 8 Hz
    # Check a range around expected theta
    lags_to_check = range(max(1, theta_lag - 5), theta_lag + 5)
    acf_at_theta = max(acf_norm[mid + l] for l in lags_to_check)
    print(f"Oscillator ACF at theta frequency: {acf_at_theta:.3f}")
    # Find actual peak frequency
    from scipy import signal
    freqs, psd = signal.welch(x_theta, fs=sf, nperseg=min(2048, len(x_theta)))
    peak_freq = freqs[np.argmax(psd[1:]) + 1]  # skip DC
    print(f"Oscillator peak frequency: {peak_freq:.1f} Hz (expected: 8.0 Hz)")

# Check: do theta-on neurons differ from theta-off neurons?
w_diff = w_norm_on - w_norm_off
print(f"\nWeight difference (on - off): mean={w_diff.mean():.3f}, "
      f"neurons with stronger on-coupling: {np.sum(w_diff > 0.05)}")

# Check: is the result non-trivial vs a rate-only HMM?
# The baselines alone give AUC=0.917 (from our no-update run)
print(f"\nBaseline-only AUC (no oscillator): ~0.917")
print(f"Full model AUC: {auc:.3f}")
print(f"AUC improvement from oscillator: {auc - 0.917:.3f}")

print(f"\n{'='*70}")
print(f"6. TOP THETA-COUPLED NEURONS")
print(f"{'='*70}")
top20 = np.argsort(w_norm_on)[::-1][:20]
print(f"{'Neuron':>6} {'Rate':>6} {'|w|_on':>7} {'|w|_off':>7} {'Phase':>6} {'Rate_run':>8} {'Rate_imm':>8}")
print("-" * 60)
rates_run = spikes_np[labels == 1].sum(0) / (np.sum(labels == 1) * dt)
rates_imm = spikes_np[labels == 0].sum(0) / (np.sum(labels == 0) * dt)
for i in top20:
    print(f"{i:>6} {rates[i]:>6.1f} {w_norm_on[i]:>7.3f} {w_norm_off[i]:>7.3f} "
          f"{w_angle_on[i]:>5.0f}° {rates_run[i]:>8.1f} {rates_imm[i]:>8.1f}")

print(f"\n{'='*70}")
print(f"7. POPULATION THETA COHERENCE")
print(f"{'='*70}")
# For each neuron with significant coupling, compute the predicted modulation
# at the inferred theta phase. If the oscillator is real, neurons should
# show consistent phase relationships.
sig_neurons = np.where(w_norm_on > 0.1)[0]
if len(sig_neurons) > 5:
    # Compute pairwise phase differences
    angles_rad = np.radians(w_angle_on[sig_neurons])
    phase_diffs = angles_rad[:, None] - angles_rad[None, :]
    # Circular variance of phase differences (should be structured, not random)
    print(f"Number of significantly theta-coupled neurons: {len(sig_neurons)}")

    # Check if phase preferences cluster
    from scipy.stats import circmean, circstd
    circ_std = circstd(angles_rad)
    print(f"Circular std of preferred phases: {np.degrees(circ_std):.0f}° (uniform=~103°)")

    # Split by putative cell type (interneurons fire >10 Hz)
    fast = sig_neurons[rates[sig_neurons] > 10]
    slow = sig_neurons[rates[sig_neurons] <= 10]
    if len(fast) > 0:
        fast_std = circstd(np.radians(w_angle_on[fast]))
        print(f"Fast-spiking (>{10}Hz, n={len(fast)}): phase std={np.degrees(fast_std):.0f}°, "
              f"mean phase={np.degrees(circmean(np.radians(w_angle_on[fast]))):.0f}°")
    if len(slow) > 2:
        slow_std = circstd(np.radians(w_angle_on[slow]))
        print(f"Regular-spiking (≤{10}Hz, n={len(slow)}): phase std={np.degrees(slow_std):.0f}°, "
              f"mean phase={np.degrees(circmean(np.radians(w_angle_on[slow]))):.0f}°")
