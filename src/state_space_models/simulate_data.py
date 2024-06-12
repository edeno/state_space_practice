import numpy as np


def receptive_field_model(position: np.ndarray, params: np.ndarray) -> np.ndarray:
    if params.ndim == 1:
        params = params[None]
    log_max_rate, place_field_center, scale = params.T
    return np.exp(log_max_rate - (position - place_field_center) ** 2 / (2 * scale**2))



def simulate_eden_brown_2004_jump() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    dt = 0.020  # seconds
    total_time = 8000.0  # seconds
    n_total_steps = int(total_time / dt)

    time = np.arange(0, total_time, dt)

    speed = 125.0  # cm/s
    track_length = 300.0  # cm

    run1 = np.arange(0, track_length, speed * dt)
    run2 = np.arange(track_length, 0, -speed * dt)
    run = np.concatenate((run1, run2))

    position = np.concatenate([run] * int(np.ceil(n_total_steps / run.shape[0])))
    position = position[:n_total_steps]

    true_params1 = np.array([np.log(10.0), 250.0, np.sqrt(12.0)])
    true_params2 = np.array([np.log(30.0), 150.0, np.sqrt(20.0)])
    true_rate1 = receptive_field_model(position[: position.shape[0] // 2], true_params1)
    true_rate2 = receptive_field_model(position[position.shape[0] // 2 :], true_params2)
    true_rate = np.concatenate((true_rate1, true_rate2))
    spike_indicator = np.random.poisson(true_rate * dt)

    return time, position, spike_indicator, dt, true_params1, true_params2


def simulate_eden_brown_2004_linear():
    dt = 0.020  # seconds
    total_time = 8000.0  # seconds
    n_total_steps = int(total_time / dt)

    time = np.arange(0, total_time, dt)

    speed = 125.0  # cm/s
    track_length = 300.0  # cm

    run1 = np.arange(0, track_length, speed * dt)
    run2 = np.arange(track_length, 0, -speed * dt)
    run = np.concatenate((run1, run2))

    position = np.concatenate([run] * int(np.ceil(n_total_steps / run.shape[0])))
    position = position[:n_total_steps]

    true_params1 = np.array([np.log(10.0), 250.0, np.sqrt(12.0)])
    true_params2 = np.array([np.log(30.0), 150.0, np.sqrt(20.0)])

    # Interpolate between true_params1 and true_params2
    true_params = np.linspace(true_params1, true_params2, n_total_steps)
    log_max_rate, place_field_center, scale = true_params.T
    true_rate = np.exp(
        log_max_rate - (position - place_field_center) ** 2 / (2 * scale**2)
    )
    spike_indicator = np.random.poisson(true_rate * dt)

    return time, position, spike_indicator, dt, true_params
