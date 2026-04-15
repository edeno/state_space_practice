import numpy as np
from numpy.typing import ArrayLike

from state_space_practice.place_field_model import build_2d_spline_basis


def receptive_field_model(position: ArrayLike, params: np.ndarray) -> np.ndarray:
    if params.ndim == 1:
        params = params[None]
    log_max_rate, place_field_center, scale = params.T
    result: np.ndarray = np.exp(log_max_rate - (position - place_field_center) ** 2 / (2 * scale**2))
    return result


def _eden_brown_2004_base(
    dt: float = 0.020,
    total_time: float = 8000.0,
    speed: float = 125.0,
    track_length: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Shared setup for Eden & Brown 2004 simulations."""
    n_total_steps = int(total_time / dt)
    time = np.arange(0, total_time, dt)

    run1 = np.arange(0, track_length, speed * dt)
    run2 = np.arange(track_length, 0, -speed * dt)
    run = np.concatenate((run1, run2))

    position = np.concatenate([run] * int(np.ceil(n_total_steps / run.shape[0])))
    position = position[:n_total_steps]

    true_params1 = np.array([np.log(10.0), 250.0, np.sqrt(12.0)])
    true_params2 = np.array([np.log(30.0), 150.0, np.sqrt(20.0)])

    return time, position, n_total_steps, true_params1, true_params2


def simulate_eden_brown_2004_jump(
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    dt = 0.020
    time, position, _, true_params1, true_params2 = _eden_brown_2004_base(dt=dt)

    true_rate1 = receptive_field_model(position[: position.shape[0] // 2], true_params1)
    true_rate2 = receptive_field_model(position[position.shape[0] // 2 :], true_params2)
    true_rate = np.concatenate((true_rate1, true_rate2))
    spike_indicator = rng.poisson(true_rate * dt)

    return time, position, spike_indicator, dt, true_params1, true_params2


def simulate_eden_brown_2004_linear(
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()
    dt = 0.020
    time, position, n_total_steps, true_params1, true_params2 = _eden_brown_2004_base(dt=dt)

    # Interpolate between true_params1 and true_params2
    true_params = np.linspace(true_params1, true_params2, n_total_steps)
    log_max_rate, place_field_center, scale = true_params.T
    true_rate = np.exp(
        log_max_rate - (position - place_field_center) ** 2 / (2 * scale**2)
    )
    spike_indicator = rng.poisson(true_rate * dt)

    return time, position, spike_indicator, dt, true_params


def simulate_2d_moving_place_field(
    total_time: float = 600.0,
    dt: float = 0.004,
    arena_size: float = 100.0,
    speed: float = 20.0,
    peak_rate: float = 30.0,
    background_rate: float = 1.0,
    place_field_sigma: float = 12.0,
    drift_speed: float = 0.02,
    n_interior_knots: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate a neuron with a 2D place field that drifts over time.

    The animal runs a lawnmower trajectory in a square arena. The neuron's
    place field center moves linearly from one location to another over
    the session. The design matrix uses the same ``build_2d_spline_basis``
    as ``PlaceFieldModel``, so the basis is directly comparable.

    Parameters
    ----------
    total_time : float
        Duration in seconds.
    dt : float
        Time bin width in seconds.
    arena_size : float
        Side length of square arena in cm.
    speed : float
        Animal speed in cm/s.
    peak_rate : float
        Peak in-field firing rate in Hz (typical CA1: 10-40 Hz).
    background_rate : float
        Out-of-field baseline firing rate in Hz (typical CA1: 0.5-2 Hz).
    place_field_sigma : float
        Place field width (std dev) in cm (typical CA1 open field: 10-15 cm).
    drift_speed : float
        Place field center drift in cm/s.
    n_interior_knots : int
        Number of interior B-spline knots per spatial dimension.
        Total basis functions = (n_interior_knots + 3)^2.
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    dict with keys:
        time : (n_time,) array
        position : (n_time, 2) array — animal x, y position
        spikes : (n_time,) int array — spike counts per bin
        true_rate : (n_time,) array — true firing rate in Hz
        true_center : (n_time, 2) array — true place field center over time
        design_matrix : (n_time, n_basis) array — 2D spline basis evaluated at position
        basis_info : dict — spline basis specification (same format as PlaceFieldModel)
        dt : float
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_time = int(total_time / dt)
    time = np.arange(n_time) * dt

    # --- Simulate 2D trajectory with uniform coverage ---
    # Lawnmower pattern: sweep x at each y level, alternating direction.
    # Fully vectorized — no loops.
    step_size = speed * dt
    steps_per_row = int(np.ceil(arena_size / step_size))
    row_spacing = 5.0  # cm between rows
    n_rows = int(np.ceil(arena_size / row_spacing))
    y_levels = np.linspace(0, arena_size, n_rows)

    # Build x positions: even rows go 0→arena, odd rows go arena→0
    x_forward = np.linspace(0, arena_size, steps_per_row)
    x_reverse = x_forward[::-1]
    # Stack all rows: (n_rows, steps_per_row)
    row_indices = np.arange(n_rows)
    all_x = np.where(
        (row_indices % 2 == 0)[:, None],
        x_forward[None, :],
        x_reverse[None, :],
    ).ravel()
    all_y = np.repeat(y_levels, steps_per_row)

    # Tile to fill session, add jitter
    n_sweep = len(all_x)
    n_tiles = int(np.ceil(n_time / n_sweep))
    tiled_x = np.tile(all_x, n_tiles)[:n_time]
    tiled_y = np.tile(all_y, n_tiles)[:n_time]
    noise = rng.normal(0, step_size * 0.5, (n_time, 2))
    position = np.column_stack([tiled_x, tiled_y]) + noise
    position = np.clip(position, 0, arena_size)

    # --- Drifting place field center ---
    total_drift = drift_speed * total_time
    start_center = np.array([arena_size * 0.35, arena_size * 0.35])
    drift_direction = np.array([1.0, 0.5])
    drift_direction = drift_direction / np.linalg.norm(drift_direction)
    end_center = start_center + drift_direction * total_drift

    # Clip to stay within arena (with margin)
    margin = place_field_sigma
    end_center = np.clip(end_center, margin, arena_size - margin)

    true_center = np.linspace(start_center, end_center, n_time)

    # --- True firing rate from 2D Gaussian place field + background ---
    dist_sq = np.sum((position - true_center) ** 2, axis=1)
    true_rate = (
        background_rate
        + (peak_rate - background_rate) * np.exp(-dist_sq / (2 * place_field_sigma**2))
    )

    # --- Build design matrix from 2D spline basis ---
    # Uses the same build_2d_spline_basis as PlaceFieldModel, so the basis
    # is directly comparable between simulation and model fitting.
    design_matrix, basis_info = build_2d_spline_basis(
        position, n_interior_knots=n_interior_knots
    )

    # --- Generate spikes ---
    spikes = rng.poisson(true_rate * dt)

    return {
        "time": time,
        "position": position,
        "spikes": spikes,
        "true_rate": true_rate,
        "true_center": true_center,
        "design_matrix": design_matrix,
        "basis_info": basis_info,
        "dt": dt,
    }
