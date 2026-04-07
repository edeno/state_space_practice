"""Test gaussian_filter boundary mode effect on rate estimation."""
import numpy as np
from scipy.ndimage import gaussian_filter

# 2D test: narrow track arm with uniform rate
grid_occ = np.zeros((50, 50))
grid_occ[22:28, 10:40] = 10.0  # track arm, 6 bins wide, 30 bins long
grid_spk = np.zeros((50, 50))
grid_spk[22:28, 10:40] = 5.0  # uniform spike count

sigma_bins = 1.2  # sigma=5cm / dx=4cm

# The true rate everywhere on the track is 5/10 = 0.5
print("True rate on track = 5/10 = 0.5")
print()

for mode_name, mode_kw in [
    ("reflect (scipy default)", {}),
    ("constant (zero-pad)", {"mode": "constant", "cval": 0}),
]:
    occ_s = gaussian_filter(grid_occ, sigma_bins, **mode_kw)
    spk_s = gaussian_filter(grid_spk, sigma_bins, **mode_kw)
    rate = np.where(occ_s > 0.02, spk_s / occ_s, 0)

    print(f"mode={mode_name}")
    print("  Rate profile across track at x=25 (track is y=22-27):")
    for y in range(19, 32):
        marker = " <-- track" if 22 <= y <= 27 else ""
        print(f"    y={y}: rate={rate[y, 25]:.4f}{marker}")

    # Also check along the track near the array boundary (x=0..12)
    print("  Rate profile along track at y=25 near array edge (x=0..15):")
    for x in range(0, 16):
        marker = " <-- track" if 10 <= x <= 39 else ""
        print(f"    x={x}: rate={rate[25, x]:.4f}{marker}")
    print()

# Now test on actual data
print("=== Actual data ===")
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "data")
from load_bandit_data import load_neural_recording_from_files
from state_space_practice.preprocessing import (
    bin_spike_times, select_units, interpolate_to_new_times,
)

data = load_neural_recording_from_files("data", "j1620210710_02_r1")
pos_info = data["position_info"]
spike_times = data["spike_times"]
dt = 0.004
pos_times = pos_info.index.values
t_start, t_end = pos_times[0], pos_times[-1]
time_bins = np.arange(t_start, t_end, dt)
position_xy = np.column_stack([
    interpolate_to_new_times(pos_info["head_position_x"].values, pos_times, time_bins),
    interpolate_to_new_times(pos_info["head_position_y"].values, pos_times, time_bins),
])
selected = select_units(spike_times, min_rate=0.1, start_time=t_start, end_time=t_end)
spikes = bin_spike_times([spike_times[i] for i in selected], time_bins)

n_grid = 50
sigma = 5.0
x_min, x_max = position_xy[:, 0].min(), position_xy[:, 0].max()
y_min, y_max = position_xy[:, 1].min(), position_xy[:, 1].max()
x_bin_edges = np.linspace(x_min, x_max, n_grid + 1)
y_bin_edges = np.linspace(y_min, y_max, n_grid + 1)
dx = x_bin_edges[1] - x_bin_edges[0]
sigma_bins_actual = sigma / dx

occ, _, _ = np.histogram2d(
    position_xy[:, 0], position_xy[:, 1], bins=[x_bin_edges, y_bin_edges]
)
occ_time = occ * dt

# Pick one neuron for illustration
n_idx = 10
spike_map, _, _ = np.histogram2d(
    position_xy[:, 0], position_xy[:, 1],
    bins=[x_bin_edges, y_bin_edges],
    weights=spikes[:, n_idx],
)

for mode_name, mode_kw in [
    ("reflect (default)", {}),
    ("constant (zero-pad)", {"mode": "constant", "cval": 0}),
]:
    occ_s = gaussian_filter(occ_time.T, sigma_bins_actual, **mode_kw)
    spk_s = gaussian_filter(spike_map.T, sigma_bins_actual, **mode_kw)
    rate = np.where(occ_s > 0.02, spk_s / occ_s, 0)

    # Profile along top row (y index 0) and bottom row (y index -1)
    # These are the array boundaries where reflect vs constant matters
    print(f"\nmode={mode_name}, neuron {n_idx}:")
    print(f"  Rate at array corners and edges:")
    print(f"    [0,0]={rate[0,0]:.4f}  [0,-1]={rate[0,-1]:.4f}")
    print(f"    [-1,0]={rate[-1,0]:.4f}  [-1,-1]={rate[-1,-1]:.4f}")

    # Compare rates at the grid boundary vs 2 bins inward
    # Top edge (low y)
    for label, y_idx in [("y=0 (edge)", 0), ("y=1", 1), ("y=2", 2), ("y=5", 5)]:
        nonzero = np.count_nonzero(rate[y_idx])
        max_r = rate[y_idx].max()
        print(f"    {label}: max_rate={max_r:.4f}, nonzero_bins={nonzero}")
