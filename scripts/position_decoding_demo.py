"""Position decoding demo: decode hippocampal position and generate a movie.

Loads the J16 bandit session, fits place field rate maps from the full session,
decodes position on a single running bout using Laplace-EKF + RTS smoother,
and saves an MP4 movie showing true vs decoded position on the 2D track.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

jax.config.update("jax_enable_x64", True)

# Add project root to path so imports work when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from load_bandit_data import load_neural_recording_from_files
from state_space_practice.position_decoder import PositionDecoder
from state_space_practice.preprocessing import (
    bin_spike_times,
    identify_behavioral_bouts,
    interpolate_to_new_times,
    select_units,
)


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading data...")
    data = load_neural_recording_from_files(
        PROJECT_ROOT / "data", "j1620210710_02_r1"
    )
    position_info = data["position_info"]
    spike_times = data["spike_times"]
    track_graph = data["track_graph"]

    # ── 2. Prepare time bins ──────────────────────────────────────────────
    dt = 0.004  # 4 ms bins (250 Hz)
    pos_times = position_info.index.values
    t_start, t_end = pos_times[0], pos_times[-1]
    time_bins = np.arange(t_start, t_end, dt)
    n_time = len(time_bins)

    # ── 3. Select active neurons ──────────────────────────────────────────
    print("Selecting active neurons...")
    selected = select_units(spike_times, min_rate=0.5, start_time=t_start, end_time=t_end)
    spike_times_sel = [spike_times[i] for i in selected]
    n_neurons = len(spike_times_sel)
    print(f"  Selected {n_neurons} neurons with rate > 0.5 Hz")

    # ── 4. Bin spikes and interpolate position ────────────────────────────
    print("Binning spikes...")
    spikes = bin_spike_times(spike_times_sel, time_bins)

    position_xy = np.column_stack([
        interpolate_to_new_times(
            position_info["head_position_x"].values, pos_times, time_bins
        ),
        interpolate_to_new_times(
            position_info["head_position_y"].values, pos_times, time_bins
        ),
    ])
    speed = interpolate_to_new_times(
        position_info["head_speed"].values, pos_times, time_bins
    )

    # ── 5. Set process noise ─────────────────────────────────────────────
    # For a random walk decoder at high sampling rates (250 Hz), the
    # process noise must be large enough that the sparse spike observations
    # can steer the position estimate.  Empirically calibrated to q_pos=50
    # via grid search on decoding error.
    q_pos = 50.0

    # ── 6. Find a good running bout ───────────────────────────────────────
    print("Finding running bouts...")
    min_dur_samples = int(2.0 / dt)
    max_dur_samples = int(6.0 / dt)
    bouts = identify_behavioral_bouts(
        speed, speed_threshold=10.0, min_duration=min_dur_samples
    )

    # Pick the bout with the largest spatial displacement
    best_bout = None
    best_dist = 0
    for s, e in bouts:
        dur = e - s
        if dur > max_dur_samples:
            continue
        displacement = np.sqrt(
            (position_xy[e - 1, 0] - position_xy[s, 0]) ** 2
            + (position_xy[e - 1, 1] - position_xy[s, 1]) ** 2
        )
        if displacement > best_dist:
            best_dist = displacement
            best_bout = (s, e)

    if best_bout is None:
        raise RuntimeError("No suitable running bout found")

    bout_start, bout_end = best_bout
    bout_dur = (bout_end - bout_start) * dt
    print(
        f"  Selected bout: samples {bout_start}-{bout_end}, "
        f"duration={bout_dur:.2f}s, displacement={best_dist:.1f}cm"
    )

    # ── 6. Fit decoder on full session ────────────────────────────────────
    print("Fitting place field rate maps on full session...")
    decoder = PositionDecoder(
        dt=dt,
        q_pos=q_pos,
        include_velocity=False,
        n_grid=50,
        smoothing_sigma=5.0,
    )
    decoder.fit(position_xy, spikes)

    # ── 7. Decode the running bout ────────────────────────────────────────
    bout_spikes = spikes[bout_start:bout_end]
    bout_position = position_xy[bout_start:bout_end]
    init_pos = bout_position[0]

    print("Decoding position (smoother)...")
    result = decoder.decode(
        bout_spikes,
        method="smoother",
        init_position=init_pos,
    )
    decoded_xy = np.array(result.position_xy)
    decoded_cov = np.array(result.position_cov_xy)

    error = np.sqrt(np.sum((decoded_xy - bout_position) ** 2, axis=1))
    print(f"  Median error: {np.median(error):.1f} cm")

    # ── 8. Build the movie ────────────────────────────────────────────────
    print("Generating movie...")

    # Track outline from graph edges
    track_edges = []
    for u, v in track_graph.edges():
        p1 = track_graph.nodes[u]["pos"]
        p2 = track_graph.nodes[v]["pos"]
        track_edges.append((p1, p2))

    # Subsample all positions for background scatter
    all_pos_sub = position_xy[::20]  # every 20th sample

    # Animation frame stride: target ~30 fps at real-time playback
    # dt=0.004s per sample, so 1s = 250 samples. At 30fps, show every ~8 samples
    stride = 8
    frame_indices = np.arange(0, len(bout_position), stride)
    n_frames = len(frame_indices)
    fps = 30
    trail_len = 15  # number of past frames to show as trail

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.set_facecolor("white")

    # Draw track outline
    for (x1, y1), (x2, y2) in track_edges:
        ax.plot([x1, x2], [y1, y2], color="#999999", linewidth=3, zorder=1)

    # Background: all visited positions
    ax.scatter(
        all_pos_sub[:, 0], all_pos_sub[:, 1],
        s=0.3, c="#DDDDDD", alpha=0.5, zorder=0, rasterized=True,
    )

    # Set axis limits with padding
    x_all = position_xy[:, 0]
    y_all = position_xy[:, 1]
    pad = 15
    ax.set_xlim(x_all.min() - pad, x_all.max() + pad)
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)", fontsize=12)
    ax.set_ylabel("y (cm)", fontsize=12)
    ax.set_title("Hippocampal Position Decoding", fontsize=14, fontweight="bold")

    # Initialize artists
    (true_trail,) = ax.plot([], [], color="#2ca02c", linewidth=2, alpha=0.5, zorder=3)
    (true_dot,) = ax.plot([], [], "o", color="#2ca02c", markersize=10, zorder=4, label="True position")
    (decoded_trail,) = ax.plot([], [], color="#d62728", linewidth=2, alpha=0.5, zorder=3)
    (decoded_dot,) = ax.plot([], [], "o", color="#d62728", markersize=10, zorder=4, label="Decoded position")
    conf_ellipse = Ellipse(
        (0, 0), 0, 0, angle=0,
        facecolor="#d62728", alpha=0.15, edgecolor="#d62728", linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(conf_ellipse)
    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    error_text = ax.text(
        0.02, 0.92, "", transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()

    def _confidence_ellipse_params(cov_2x2, n_std=2.0):
        """Compute ellipse width, height, angle from 2x2 covariance."""
        eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)
        eigenvalues = np.maximum(eigenvalues, 0)  # numerical safety
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        width = 2 * n_std * np.sqrt(eigenvalues[1])
        height = 2 * n_std * np.sqrt(eigenvalues[0])
        return width, height, angle

    def update(frame_num):
        idx = frame_indices[frame_num]

        # Trail indices
        trail_start = max(0, frame_num - trail_len)
        trail_idx = frame_indices[trail_start : frame_num + 1]

        # True position
        true_trail.set_data(bout_position[trail_idx, 0], bout_position[trail_idx, 1])
        true_dot.set_data([bout_position[idx, 0]], [bout_position[idx, 1]])

        # Decoded position
        decoded_trail.set_data(decoded_xy[trail_idx, 0], decoded_xy[trail_idx, 1])
        decoded_dot.set_data([decoded_xy[idx, 0]], [decoded_xy[idx, 1]])

        # Confidence ellipse (95% = 2 std)
        w, h, angle = _confidence_ellipse_params(decoded_cov[idx], n_std=2.0)
        conf_ellipse.set_center((decoded_xy[idx, 0], decoded_xy[idx, 1]))
        conf_ellipse.width = w
        conf_ellipse.height = h
        conf_ellipse.angle = angle

        # Time and error text
        t = idx * dt
        err = error[idx]
        time_text.set_text(f"t = {t:.2f} s")
        error_text.set_text(f"error = {err:.1f} cm")

        return true_trail, true_dot, decoded_trail, decoded_dot, conf_ellipse, time_text, error_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=True,
    )

    out_path = PROJECT_ROOT / "output" / "position_decoding_demo.mp4"
    anim.save(
        str(out_path),
        writer=animation.FFMpegWriter(fps=fps, bitrate=2000),
        dpi=150,
    )
    plt.close(fig)
    print(f"Saved movie to {out_path}")
    print(f"  {n_frames} frames, {n_frames / fps:.1f}s at {fps} fps")


if __name__ == "__main__":
    main()
