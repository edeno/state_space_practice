# %% [markdown]
# # 2D Place Field Tracking — Real CA1 Data (J16 Plus Maze)
#
# Apply the point-process state-space model to track place field drift
# in real hippocampal recordings. Single neuron at a time.

# %%
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrix

sys.path.insert(0, "..")
from data.load_bandit_data import load_neural_recording_from_files
from state_space_practice.point_process_kalman import (
    kalman_maximization_step,
    log_conditional_intensity,
    stochastic_point_process_smoother,
)

# %%
# --- Load real data ---
raw = load_neural_recording_from_files("../data/", "j1620210710_02_r1")
pos_df = raw["position_info"]
spike_times_all = raw["spike_times"]

# %%
# --- Preprocessing ---
dt = 0.004  # 4ms bins (250 Hz), matching simulation
t_start = pos_df.index[0]
t_end = pos_df.index[-1]
time_bins = np.arange(t_start, t_end, dt)
n_time = len(time_bins)

# Interpolate 2D position to time bins
pos_x = np.interp(time_bins, pos_df.index, pos_df["head_position_x"].values)
pos_y = np.interp(time_bins, pos_df.index, pos_df["head_position_y"].values)
speed = np.interp(time_bins, pos_df.index, pos_df["head_speed"].values)
position = np.column_stack([pos_x, pos_y])

print(f"n_time: {n_time} ({(t_end - t_start):.0f}s)")
print(f"X range: [{pos_x.min():.1f}, {pos_x.max():.1f}]")
print(f"Y range: [{pos_y.min():.1f}, {pos_y.max():.1f}]")

# %%
# --- Select a good place cell ---
# Find units with moderate rates that are likely place cells
duration = t_end - t_start
rates = np.array([len(s) / duration for s in spike_times_all])

# Use unit 93: compact place field, 5 Hz mean, 35 Hz peak, 5625 running spikes
best_unit = 93
print(f"Selected unit {best_unit} (rate: {rates[best_unit]:.1f} Hz)")

# %%
# --- Bin spikes for selected unit ---
spike_counts = np.zeros(n_time)
st = spike_times_all[best_unit]
st = st[(st >= t_start) & (st < t_end)]
bin_indices = np.searchsorted(time_bins, st) - 1
bin_indices = bin_indices[(bin_indices >= 0) & (bin_indices < n_time)]
np.add.at(spike_counts, bin_indices, 1)
print(f"Total spikes: {int(spike_counts.sum())}")
print(f"Bins with spikes: {(spike_counts > 0).sum()} ({100*(spike_counts > 0).mean():.1f}%)")

# %%
# --- Only use running epochs (speed > 5 cm/s) ---
running_mask = speed > 5.0
print(f"Running fraction: {running_mask.mean():.1%}")

# Use only running times
time_running = time_bins[running_mask]
pos_running = position[running_mask]
spikes_running = spike_counts[running_mask]
n_running = running_mask.sum()
print(f"Running bins: {n_running} ({n_running * dt:.0f}s)")
print(f"Spikes during running: {int(spikes_running.sum())}")

# %%
# --- Build 2D spline basis with data-driven knots ---
# Place knots at quantiles of the running position so they concentrate
# along the plus maze arms where the animal actually goes.
n_interior_knots = 5
x_knots = np.quantile(pos_running[:, 0], np.linspace(0.05, 0.95, n_interior_knots))
y_knots = np.quantile(pos_running[:, 1], np.linspace(0.05, 0.95, n_interior_knots))
x_lo, x_hi = float(pos_running[:, 0].min()), float(pos_running[:, 0].max())
y_lo, y_hi = float(pos_running[:, 1].min()), float(pos_running[:, 1].max())

print(f"X knots: {np.round(x_knots, 1)}")
print(f"Y knots: {np.round(y_knots, 1)}")

spline_formula = (
    "te(bs(x, knots=x_knots, lower_bound=x_lo, upper_bound=x_hi), "
    "   bs(y, knots=y_knots, lower_bound=y_lo, upper_bound=y_hi)) - 1"
)
spline_env = {
    "x_knots": x_knots, "y_knots": y_knots,
    "x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi,
}

design_matrix = np.asarray(
    dmatrix(spline_formula, {"x": pos_running[:, 0], "y": pos_running[:, 1], **spline_env})
)
n_basis = design_matrix.shape[1]
print(f"Design matrix: ({n_running}, {n_basis})")

# %%
# --- Fit model ---
design_matrix_jax = jnp.asarray(design_matrix)
spikes_jax = jnp.asarray(spikes_running)

init_mean = jnp.zeros(n_basis)
init_cov = jnp.eye(n_basis) * 1.0
A = jnp.eye(n_basis)
Q = jnp.eye(n_basis) * 1e-5

n_em_iter = 5

for em_iter in range(n_em_iter):
    smoother_mean, smoother_cov, smoother_cross_cov, marginal_ll = (
        stochastic_point_process_smoother(
            init_mean,
            init_cov,
            design_matrix_jax,
            spikes_jax,
            dt,
            A,
            Q,
            log_conditional_intensity,
        )
    )

    ll_val = float(marginal_ll)
    print(f"EM iter {em_iter}: marginal LL = {ll_val:.1f}")

    _, Q_new, init_mean_new, init_cov_new = kalman_maximization_step(
        smoother_mean, smoother_cov, smoother_cross_cov
    )
    q_diag = jnp.maximum(jnp.diag(Q_new), 1e-10)
    Q = jnp.diag(q_diag)
    init_mean = init_mean_new
    init_cov = init_cov_new

# %%
# --- Reconstruct place fields over time ---
# Grid must stay within the spline bounds (running position range)
n_grid = 80
x_grid = np.linspace(x_lo, x_hi, n_grid)
y_grid = np.linspace(y_lo, y_hi, n_grid)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_positions = np.column_stack([xx.ravel(), yy.ravel()])

Z_grid = np.asarray(
    dmatrix(spline_formula, {"x": grid_positions[:, 0], "y": grid_positions[:, 1], **spline_env})
)

# %%
# --- Plot: place fields in temporal thirds ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

thirds = [
    ("Early", slice(0, n_running // 3)),
    ("Middle", slice(n_running // 3, 2 * n_running // 3)),
    ("Late", slice(2 * n_running // 3, n_running)),
]

vmax = 0
rate_grids = []
ci_grids = []
for label, sl in thirds:
    weights = np.array(smoother_mean[sl].mean(axis=0))
    log_rate_grid = Z_grid @ weights
    rate_grid = np.exp(log_rate_grid).reshape(n_grid, n_grid)
    rate_grids.append(rate_grid)
    vmax = max(vmax, np.percentile(rate_grid, 99))

    # Uncertainty: average covariance over the block
    cov_block = np.array(smoother_cov[sl].mean(axis=0))
    var_log_rate = np.sum(Z_grid @ cov_block * Z_grid, axis=1)
    ci_width = 1.96 * np.sqrt(np.maximum(var_log_rate, 0))
    ci_grids.append(ci_width.reshape(n_grid, n_grid))

# Top row: estimated rate maps
for i, (label, sl) in enumerate(thirds):
    ax = axes[0, i]
    im = ax.pcolormesh(x_grid, y_grid, rate_grids[i], cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(f"Estimated Rate — {label}")
    ax.set_xlabel("x (cm)")
    if i == 0:
        ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Rate (Hz)")

# Bottom row: uncertainty (CI width on log-rate)
ci_vmax = max(c.max() for c in ci_grids)
for i, (label, sl) in enumerate(thirds):
    ax = axes[1, i]
    im = ax.pcolormesh(x_grid, y_grid, ci_grids[i], cmap="viridis", vmin=0, vmax=ci_vmax)
    ax.set_title(f"95% CI Width (log-rate) — {label}")
    ax.set_xlabel("x (cm)")
    if i == 0:
        ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="CI width")

plt.suptitle(f"Unit {best_unit} — Place Field Tracking (Real CA1 Data)", fontsize=14)
plt.tight_layout()
plt.savefig("../output/place_field_tracking_real.png", dpi=150)
plt.show()

# %%
# --- Plot: place field center drift ---
fig, ax = plt.subplots(figsize=(6, 6))

n_blocks = 20
block_size = n_running // n_blocks
centers = []
for i in range(n_blocks):
    w = np.array(smoother_mean[i * block_size : (i + 1) * block_size].mean(axis=0))
    log_r = Z_grid @ w
    peak_idx = np.argmax(log_r)
    centers.append(grid_positions[peak_idx])
centers = np.array(centers)

# Color by time
colors = plt.cm.coolwarm(np.linspace(0, 1, n_blocks))
for i in range(n_blocks - 1):
    ax.plot(
        centers[i : i + 2, 0],
        centers[i : i + 2, 1],
        "-o",
        color=colors[i],
        markersize=5,
    )
ax.plot(centers[0, 0], centers[0, 1], "ko", markersize=10, label="Start")
ax.plot(centers[-1, 0], centers[-1, 1], "k^", markersize=10, label="End")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title(f"Unit {best_unit} — Place Field Center Over Time")
ax.legend()
ax.set_aspect("equal")
total_drift = np.linalg.norm(centers[-1] - centers[0])
print(f"Total center drift: {total_drift:.1f} cm")

plt.tight_layout()
plt.savefig("../output/place_field_center_drift_real.png", dpi=150)
plt.show()
print("Done!")
