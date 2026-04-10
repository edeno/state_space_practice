"""Phase 1 baseline + Phase 2 ablation ladder for the PositionDecoder
debugging plan (docs/plans/2026-04-10-position-decoder-tracking-fix.md).

Runs the broken trajectory_data fixture and the working
circular_trajectory_2d fixture side by side, plus the ablation ladder
that interpolates between them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_space_practice.position_decoder import (  # noqa: E402
    PlaceFieldRateMaps,
    PositionDecoder,
    position_decoder_filter,
    position_decoder_smoother,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def fixture_trajectory_data():
    """Exact copy of TestPositionDecoder.trajectory_data."""
    rng = np.random.default_rng(42)
    n_time, dt = 1000, 0.004
    t = np.arange(n_time) * dt
    true_x = 50 + 20 * np.cos(2 * np.pi * t / 2.0)
    true_y = 50 + 20 * np.sin(2 * np.pi * t / 2.0)
    position = np.column_stack([true_x, true_y])
    n_neurons = 5
    centers = rng.uniform(20, 80, (n_neurons, 2))
    spikes = np.zeros((n_time, n_neurons))
    for n in range(n_neurons):
        dist_sq = np.sum((position - centers[n]) ** 2, axis=1)
        rate = 25 * np.exp(-dist_sq / (2 * 15**2)) + 0.5
        spikes[:, n] = rng.poisson(rate * dt)
    return position, spikes, dt, centers


def fixture_circular_2d():
    """Exact copy of TestPositionDecoderIntegration.circular_trajectory_2d."""
    rng = np.random.default_rng(123)
    n_time, dt = 1000, 0.004
    t = np.arange(n_time) * dt
    true_x = 50 + 25 * np.cos(2 * np.pi * t / 2.0)
    true_y = 50 + 25 * np.sin(2 * np.pi * t / 2.0)
    position = np.column_stack([true_x, true_y])

    n_grid = 40
    x_edges = np.linspace(0, 100, n_grid)
    y_edges = np.linspace(0, 100, n_grid)
    xx, yy = np.meshgrid(x_edges, y_edges)
    field_centers = [
        (30, 30), (70, 30),
        (30, 70), (70, 70),
        (50, 25), (50, 75),
    ]
    rate_maps_list = [
        40 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 12**2)) + 0.5
        for cx, cy in field_centers
    ]
    rate_maps_arr = np.stack(rate_maps_list)
    rm = PlaceFieldRateMaps(
        rate_maps=rate_maps_arr, x_edges=x_edges, y_edges=y_edges,
    )

    n_neurons = len(field_centers)
    spikes = np.zeros((n_time, n_neurons))
    for ti in range(n_time):
        log_r = rm.log_rate(jnp.array(position[ti]))
        rates = np.exp(np.array(log_r))
        spikes[ti] = rng.poisson(rates * dt)

    return position, spikes, dt, rm, field_centers


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def decode_metrics(result, position, warmup=100):
    decoded = np.array(result.position_mean[:, :2])
    err = np.linalg.norm(decoded - position, axis=1)[warmup:]
    xs = decoded[warmup:, 0]
    ys = decoded[warmup:, 1]
    tx = position[warmup:, 0]
    ty = position[warmup:, 1]
    with np.errstate(invalid="ignore"):
        corr_x = (
            np.corrcoef(xs, tx)[0, 1] if xs.std() > 1e-9 else float("nan")
        )
        corr_y = (
            np.corrcoef(ys, ty)[0, 1] if ys.std() > 1e-9 else float("nan")
        )
    return dict(
        median_err=float(np.median(err)),
        mean_err=float(np.mean(err)),
        p90_err=float(np.percentile(err, 90)),
        corr_x=float(corr_x),
        corr_y=float(corr_y),
        decoded_std_x=float(xs.std()),
        decoded_std_y=float(ys.std()),
        marginal_ll=float(result.marginal_log_likelihood),
    )


def run(label, decode_fn, position, warmup=100):
    result = decode_fn()
    m = decode_metrics(result, position, warmup=warmup)
    print(
        f"  {label:<45} "
        f"median={m['median_err']:6.2f}  "
        f"corr=(x:{m['corr_x']:+.3f},y:{m['corr_y']:+.3f})  "
        f"std=({m['decoded_std_x']:5.2f},{m['decoded_std_y']:5.2f})  "
        f"ll={m['marginal_ll']:9.2f}"
    )
    return m


# ---------------------------------------------------------------------------
# Phase 1.1 & 1.2
# ---------------------------------------------------------------------------


def phase1():
    print("=" * 92)
    print("PHASE 1.1 — Baseline on both fixtures")
    print("=" * 92)

    # ----- trajectory_data -----
    pos_td, spikes_td, dt_td, centers_td = fixture_trajectory_data()
    print(f"\ntrajectory_data: n_time={len(pos_td)}, n_neurons={spikes_td.shape[1]}")
    print(f"  total spikes: {int(spikes_td.sum())}")
    print(f"  neuron centers: {centers_td.round(1).tolist()}")
    print(f"  true position[0] = ({pos_td[0,0]:.2f}, {pos_td[0,1]:.2f})")

    dec_td = PositionDecoder(dt=dt_td)
    dec_td.fit(pos_td, spikes_td)
    print(f"  _use_analytical={dec_td.rate_maps._use_analytical}")

    print("\n  default init (rate map center):")
    run("smoother",
        lambda: dec_td.decode(spikes=spikes_td, method="smoother"),
        pos_td)
    run("filter",
        lambda: dec_td.decode(spikes=spikes_td, method="filter"),
        pos_td)

    print("\n  init at animal's true first position:")
    init_td = jnp.concatenate([jnp.asarray(pos_td[0]), jnp.zeros(2)])
    run("smoother, init@true",
        lambda: dec_td.decode(spikes=spikes_td, method="smoother",
                              init_position=init_td),
        pos_td)
    run("filter,   init@true",
        lambda: dec_td.decode(spikes=spikes_td, method="filter",
                              init_position=init_td),
        pos_td)

    # ----- circular_trajectory_2d -----
    pos_c2, spikes_c2, dt_c2, rm_c2, centers_c2 = fixture_circular_2d()
    print(f"\ncircular_trajectory_2d: n_time={len(pos_c2)}, n_neurons={spikes_c2.shape[1]}")
    print(f"  total spikes: {int(spikes_c2.sum())}")
    print(f"  neuron centers: {centers_c2}")
    print(f"  true position[0] = ({pos_c2[0,0]:.2f}, {pos_c2[0,1]:.2f})")
    print(f"  _use_analytical={rm_c2._use_analytical}")

    print("\n  default init:")
    run("filter",
        lambda: position_decoder_filter(
            spikes=jnp.asarray(spikes_c2), rate_maps=rm_c2, dt=dt_c2,
        ),
        pos_c2)
    run("smoother",
        lambda: position_decoder_smoother(
            spikes=jnp.asarray(spikes_c2), rate_maps=rm_c2, dt=dt_c2,
        ),
        pos_c2)

    print("\n  init at animal's true first position:")
    init_c2 = jnp.concatenate([jnp.asarray(pos_c2[0]), jnp.zeros(2)])
    run("filter,   init@true",
        lambda: position_decoder_filter(
            spikes=jnp.asarray(spikes_c2), rate_maps=rm_c2, dt=dt_c2,
            init_position=init_c2,
        ),
        pos_c2)
    run("smoother, init@true",
        lambda: position_decoder_smoother(
            spikes=jnp.asarray(spikes_c2), rate_maps=rm_c2, dt=dt_c2,
            init_position=init_c2,
        ),
        pos_c2)


# ---------------------------------------------------------------------------
# Phase 2 — Ablation ladder (KDE-path target)
# ---------------------------------------------------------------------------
#
# Walk from `circular_trajectory_2d` (works) toward `trajectory_data`
# (broken), changing one variable at a time. KDE path is the production
# target, so rungs that build rate maps from data always exercise
# `_use_analytical=True`.


def build_rate_maps_from_function(
    centers, peak_rate, sigma_rf, x_edges, y_edges,
):
    """Analytic rate maps via direct PlaceFieldRateMaps(...)."""
    xx, yy = np.meshgrid(x_edges, y_edges)
    rate_maps_list = [
        peak_rate * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_rf**2)) + 0.5
        for cx, cy in centers
    ]
    return PlaceFieldRateMaps(
        rate_maps=np.stack(rate_maps_list),
        x_edges=np.asarray(x_edges),
        y_edges=np.asarray(y_edges),
    )


def simulate_spikes(position, centers, dt, peak_rate, sigma_rf, rng):
    n_time = len(position)
    n_neurons = len(centers)
    spikes = np.zeros((n_time, n_neurons))
    for n, (cx, cy) in enumerate(centers):
        dist_sq = (position[:, 0] - cx) ** 2 + (position[:, 1] - cy) ** 2
        rate = peak_rate * np.exp(-dist_sq / (2 * sigma_rf**2)) + 0.5
        spikes[:, n] = rng.poisson(rate * dt)
    return spikes


def make_trajectory(radius: float, n_time: int = 1000, dt: float = 0.004):
    t = np.arange(n_time) * dt
    return np.column_stack([
        50 + radius * np.cos(2 * np.pi * t / 2.0),
        50 + radius * np.sin(2 * np.pi * t / 2.0),
    ])


# Hand-placed 6 neurons tiling the box (matches circular_trajectory_2d)
HAND_CENTERS = [
    (30, 30), (70, 30),
    (30, 70), (70, 70),
    (50, 25), (50, 75),
]


def run_rung(label, position, spikes, rate_maps, dt, init_at_true=True):
    """Standardized decoding run for a ladder rung."""
    init = None
    if init_at_true:
        init = jnp.concatenate([jnp.asarray(position[0]), jnp.zeros(2)])
    result = position_decoder_filter(
        spikes=jnp.asarray(spikes),
        rate_maps=rate_maps,
        dt=dt,
        init_position=init,
    )
    m = decode_metrics(result, position)
    print(
        f"  {label:<50} "
        f"use_kde={rate_maps._use_analytical}  "
        f"median={m['median_err']:6.2f}  "
        f"corr=(x:{m['corr_x']:+.3f},y:{m['corr_y']:+.3f})  "
        f"std=({m['decoded_std_x']:5.2f},{m['decoded_std_y']:5.2f})"
    )
    return m


def phase2():
    print("\n" + "=" * 92)
    print("PHASE 2 — Ablation ladder from circular_trajectory_2d to trajectory_data")
    print("=" * 92)
    print("All rungs use filter (to match the test path) and init@true.\n")

    rng = np.random.default_rng(42)
    dt = 0.004
    n_time = 1000

    # --- Rung A: exact circular_trajectory_2d (known working baseline) ---
    pos_a, spikes_a, _, rm_a, _ = fixture_circular_2d()
    print("[A] exact circular_trajectory_2d (hand-placed analytic maps):")
    run_rung("A: 6 hand, analytic maps, r=25, 40 Hz, sig=12", pos_a, spikes_a, rm_a, dt)

    # --- Rung B: 5 random neurons (seed 42), otherwise Rung A ---
    pos_b = make_trajectory(radius=25, n_time=n_time, dt=dt)
    rng_b = np.random.default_rng(42)
    centers_b = rng_b.uniform(20, 80, (5, 2))
    # Generate spikes from the analytic rate function (no from_data pipeline)
    spikes_b = simulate_spikes(pos_b, centers_b, dt, peak_rate=40, sigma_rf=12, rng=rng_b)
    x_edges_b = np.linspace(0, 100, 40)
    y_edges_b = np.linspace(0, 100, 40)
    rm_b = build_rate_maps_from_function(
        centers_b, peak_rate=40, sigma_rf=12, x_edges=x_edges_b, y_edges=y_edges_b,
    )
    print("[B] 5 random neurons, analytic maps, r=25, 40 Hz, sig=12:")
    run_rung("B: 5 rand, analytic maps, r=25, 40 Hz, sig=12", pos_b, spikes_b, rm_b, dt)

    # --- Rung C: drop rate to 25 Hz, sigma to 15 (matches trajectory_data) ---
    pos_c = pos_b
    centers_c = centers_b
    rng_c = np.random.default_rng(42)
    spikes_c = simulate_spikes(pos_c, centers_c, dt, peak_rate=25, sigma_rf=15, rng=rng_c)
    rm_c = build_rate_maps_from_function(
        centers_c, peak_rate=25, sigma_rf=15, x_edges=x_edges_b, y_edges=y_edges_b,
    )
    print("[C] 5 rand, analytic maps, r=25, 25 Hz, sig=15:")
    run_rung("C: 5 rand, analytic maps, r=25, 25 Hz, sig=15", pos_c, spikes_c, rm_c, dt)

    # --- Rung D: trajectory radius 20 ---
    pos_d = make_trajectory(radius=20, n_time=n_time, dt=dt)
    centers_d = centers_b
    rng_d = np.random.default_rng(42)
    spikes_d = simulate_spikes(pos_d, centers_d, dt, peak_rate=25, sigma_rf=15, rng=rng_d)
    rm_d = build_rate_maps_from_function(
        centers_d, peak_rate=25, sigma_rf=15, x_edges=x_edges_b, y_edges=y_edges_b,
    )
    print("[D] 5 rand, analytic maps, r=20, 25 Hz, sig=15:")
    run_rung("D: 5 rand, analytic maps, r=20, 25 Hz, sig=15", pos_d, spikes_d, rm_d, dt)

    # --- Rung E: now build rate maps via from_spike_position_data (KDE) ---
    # Everything else matches Rung D. Use the trajectory_data fixture's seed
    # for the neuron centers so the spikes match.
    rng_e = np.random.default_rng(42)
    centers_e = rng_e.uniform(20, 80, (5, 2))  # same as trajectory_data centers
    pos_e = make_trajectory(radius=20, n_time=n_time, dt=dt)
    # Use trajectory_data's Poisson sampling (same rng state as the fixture)
    spikes_e = np.zeros((n_time, 5))
    for n in range(5):
        dist_sq = np.sum((pos_e - centers_e[n]) ** 2, axis=1)
        rate = 25 * np.exp(-dist_sq / (2 * 15**2)) + 0.5
        spikes_e[:, n] = rng_e.poisson(rate * dt)
    # Rate maps from data — this is where the KDE path kicks in
    rm_e = PlaceFieldRateMaps.from_spike_position_data(
        pos_e, spikes_e, dt=dt, n_grid=50, sigma=5.0,
    )
    print("[E] same as D, but rate maps from_spike_position_data (n_grid=50, sigma=5):")
    run_rung("E: from_data n_grid=50 sigma=5", pos_e, spikes_e, rm_e, dt)

    # --- Rung F: try KDE path with sigma=12 (match the analytic field width) ---
    rm_f = PlaceFieldRateMaps.from_spike_position_data(
        pos_e, spikes_e, dt=dt, n_grid=50, sigma=12.0,
    )
    print("[F.4] same as E but sigma=12 (matches generative sigma_rf):")
    run_rung("F.4: from_data n_grid=50 sigma=12", pos_e, spikes_e, rm_f, dt)

    # --- Rung F.2: E with occupancy_mask cleared (no track penalty) ---
    rm_f2 = PlaceFieldRateMaps.from_spike_position_data(
        pos_e, spikes_e, dt=dt, n_grid=50, sigma=5.0,
    )
    rm_f2.occupancy_mask = None
    print("[F.2] E with occupancy_mask=None (no track penalty):")
    run_rung("F.2: E no track penalty", pos_e, spikes_e, rm_f2, dt)

    # --- Rung F.4 + init_cov smaller ---
    rm_f6 = PlaceFieldRateMaps.from_spike_position_data(
        pos_e, spikes_e, dt=dt, n_grid=50, sigma=12.0,
    )
    init_f6 = jnp.concatenate([jnp.asarray(pos_e[0]), jnp.zeros(2)])
    print("[F.6] F.4 with init_cov = 1·I (instead of 100·I):")
    result_f6 = position_decoder_filter(
        spikes=jnp.asarray(spikes_e),
        rate_maps=rm_f6,
        dt=dt,
        init_position=init_f6,
        init_cov=jnp.eye(4) * 1.0,
    )
    m = decode_metrics(result_f6, pos_e)
    print(
        f"  {'F.6: F.4 + init_cov=1*I':<50} "
        f"use_kde={rm_f6._use_analytical}  "
        f"median={m['median_err']:6.2f}  "
        f"corr=(x:{m['corr_x']:+.3f},y:{m['corr_y']:+.3f})  "
        f"std=({m['decoded_std_x']:5.2f},{m['decoded_std_y']:5.2f})"
    )


if __name__ == "__main__":
    phase1()
    phase2()
