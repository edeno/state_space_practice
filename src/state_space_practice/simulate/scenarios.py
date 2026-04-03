"""Generative simulation scenarios for validating switching oscillator models.

Each scenario simulates data from a specific model's generative process with
parameters chosen to make discrete states obviously distinguishable. These are
used by test_scenario_recovery.py to verify that models can recover the
ground truth structure.

All scenarios use:
- 2 discrete states
- 2 oscillators (theta=8Hz, beta=25Hz)
- sampling_freq=100Hz
- Z diagonal=0.98 (long dwell ~50 steps)
- n_time=3000
"""

import jax
import jax.numpy as jnp
import numpy as np

from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
    construct_correlated_noise_measurement_matrix,
    construct_correlated_noise_process_covariance,
    construct_directed_influence_measurement_matrix,
    construct_directed_influence_transition_matrix,
)
from state_space_practice.simulate.simulate_switching_kalman import simulate
from state_space_practice.simulate.simulate_switching_spikes import (
    simulate_switching_spike_oscillator,
)

# Common parameters
FREQS = jnp.array([8.0, 25.0])  # theta, beta
DAMPING = jnp.array([0.95, 0.95])
SAMPLING_FREQ = 100.0
N_OSCILLATORS = 2
N_LATENT = 2 * N_OSCILLATORS  # 4
N_DISCRETE_STATES = 2
N_TIME = 3000
Z = np.array([[0.98, 0.02], [0.02, 0.98]])


# ============================================================================
# Gaussian observation scenarios
# ============================================================================


def simulate_com_scenario(n_time: int = N_TIME, seed: int = 42) -> dict:
    """COM scenario: measurement matrix H switches between states.

    State 0: sources observe theta oscillations (8 Hz).
    State 1: sources observe beta oscillations (25 Hz).

    The frequency content of the observations switches dramatically,
    making the states trivially distinguishable.
    """
    n_sources = 3

    # A: constant, uncoupled oscillators
    A_single = np.array(construct_common_oscillator_transition_matrix(
        freqs=FREQS, auto_regressive_coef=DAMPING, sampling_freq=SAMPLING_FREQ
    ))
    A = np.stack([A_single, A_single], axis=2)

    # Q: constant, block-diagonal
    Q_single = np.array(construct_common_oscillator_process_covariance(
        variance=jnp.array([0.1, 0.1])
    ))
    Q = np.stack([Q_single, Q_single], axis=2)

    # H: state 0 sees theta, state 1 sees beta
    H = np.zeros((n_sources, N_LATENT, N_DISCRETE_STATES))
    # State 0: sources 0,1 observe theta (dims 0-1)
    H[0, 0, 0] = 0.5
    H[1, 1, 0] = 0.5
    # State 1: sources 0,1 observe beta (dims 2-3)
    H[0, 2, 1] = 0.5
    H[1, 3, 1] = 0.5

    # R: low observation noise
    R_single = np.eye(n_sources) * 0.05
    R = np.stack([R_single, R_single], axis=2)

    # Initial conditions
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal(N_LATENT)

    # Simulate
    obs, true_states, true_continuous = simulate(A, H, Q, R, Z, X0, 0, n_time)

    return {
        "obs": obs,
        "true_states": true_states,
        "true_continuous": true_continuous,
        "params": {
            "A": A, "Q": Q, "H": H, "R": R, "Z": Z,
            "n_oscillators": N_OSCILLATORS,
            "n_sources": n_sources,
            "n_discrete_states": N_DISCRETE_STATES,
            "sampling_freq": SAMPLING_FREQ,
            "freqs": np.array(FREQS),
            "damping": np.array(DAMPING),
            "process_variance": np.array([0.1, 0.1]),
            "measurement_variance": 0.05,
        },
    }


def simulate_cnm_scenario(n_time: int = N_TIME, seed: int = 42) -> dict:
    """CNM scenario: process noise covariance Q switches between states.

    State 0: independent noise (diagonal Q).
    State 1: strongly correlated noise (large off-diagonal coupling in Q).

    The covariance structure of the latent dynamics changes, which propagates
    to observable correlations between sources.
    """
    n_sources = N_OSCILLATORS  # CNM requires n_sources == n_oscillators

    # A: constant, uncoupled oscillators
    A_single = np.array(construct_common_oscillator_transition_matrix(
        freqs=FREQS, auto_regressive_coef=DAMPING, sampling_freq=SAMPLING_FREQ
    ))
    A = np.stack([A_single, A_single], axis=2)

    # Q: state 0 = low noise, state 1 = high noise
    # Using very different variance levels makes states obviously distinguishable
    # through the amplitude of the observed signal
    variance_0 = jnp.array([0.01, 0.01])
    variance_1 = jnp.array([0.5, 0.5])
    Q0 = np.array(construct_correlated_noise_process_covariance(
        variance=variance_0,
        phase_difference=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
        coupling_strength=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
    ))
    Q1 = np.array(construct_correlated_noise_process_covariance(
        variance=variance_1,
        phase_difference=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
        coupling_strength=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
    ))
    Q = np.stack([Q0, Q1], axis=2)

    # H: constant
    H_single = np.array(construct_correlated_noise_measurement_matrix(n_sources))
    H = np.stack([H_single, H_single], axis=2)

    # R: low observation noise
    R_single = np.eye(n_sources) * 0.05
    R = np.stack([R_single, R_single], axis=2)

    # Initial conditions
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal(N_LATENT)

    # Simulate
    obs, true_states, true_continuous = simulate(A, H, Q, R, Z, X0, 0, n_time)

    return {
        "obs": obs,
        "true_states": true_states,
        "true_continuous": true_continuous,
        "params": {
            "A": A, "Q": Q, "H": H, "R": R, "Z": Z,
            "n_oscillators": N_OSCILLATORS,
            "n_sources": n_sources,
            "n_discrete_states": N_DISCRETE_STATES,
            "sampling_freq": SAMPLING_FREQ,
            "freqs": np.array(FREQS),
            "damping": np.array(DAMPING),
            "process_variance": np.stack([np.array(variance_0), np.array(variance_1)], axis=1),
            "measurement_variance": 0.05,
            "phase_difference": np.zeros((N_OSCILLATORS, N_OSCILLATORS, N_DISCRETE_STATES)),
            "coupling_strength": np.zeros((N_OSCILLATORS, N_OSCILLATORS, N_DISCRETE_STATES)),
        },
    }


def simulate_dim_scenario(n_time: int = N_TIME, seed: int = 42) -> dict:
    """DIM scenario: transition matrix A switches between states.

    State 0: oscillator 1 drives oscillator 2 (osc1 -> osc2).
    State 1: oscillator 2 drives oscillator 1 (osc2 -> osc1).

    The lead-lag relationship between oscillators reverses completely.
    """
    n_sources = N_OSCILLATORS  # DIM requires n_sources == n_oscillators

    # A: state-dependent coupling direction
    coupling_0 = jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)).at[1, 0].set(0.3)
    coupling_1 = jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)).at[0, 1].set(0.3)
    phase_diffs = jnp.zeros((N_OSCILLATORS, N_OSCILLATORS))

    A0 = np.array(construct_directed_influence_transition_matrix(
        freqs=FREQS, damping_coeffs=DAMPING,
        coupling_strengths=coupling_0, phase_diffs=phase_diffs,
        sampling_freq=SAMPLING_FREQ,
    ))
    A1 = np.array(construct_directed_influence_transition_matrix(
        freqs=FREQS, damping_coeffs=DAMPING,
        coupling_strengths=coupling_1, phase_diffs=phase_diffs,
        sampling_freq=SAMPLING_FREQ,
    ))
    A = np.stack([A0, A1], axis=2)

    # Q: constant, block-diagonal
    Q_single = np.array(construct_common_oscillator_process_covariance(
        variance=jnp.array([0.1, 0.1])
    ))
    Q = np.stack([Q_single, Q_single], axis=2)

    # H: constant
    H_single = np.array(construct_directed_influence_measurement_matrix(n_sources))
    H = np.stack([H_single, H_single], axis=2)

    # R: low observation noise
    R_single = np.eye(n_sources) * 0.05
    R = np.stack([R_single, R_single], axis=2)

    # Initial conditions
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal(N_LATENT)

    # Simulate
    obs, true_states, true_continuous = simulate(A, H, Q, R, Z, X0, 0, n_time)

    return {
        "obs": obs,
        "true_states": true_states,
        "true_continuous": true_continuous,
        "params": {
            "A": A, "Q": Q, "H": H, "R": R, "Z": Z,
            "n_oscillators": N_OSCILLATORS,
            "n_sources": n_sources,
            "n_discrete_states": N_DISCRETE_STATES,
            "sampling_freq": SAMPLING_FREQ,
            "freqs": np.array(FREQS),
            "damping": np.array(DAMPING),
            "process_variance": np.array([0.1, 0.1]),
            "measurement_variance": 0.05,
            "phase_difference": np.stack([np.array(phase_diffs)] * N_DISCRETE_STATES, axis=2),
            "coupling_strength": np.stack([np.array(coupling_0), np.array(coupling_1)], axis=2),
        },
    }


# ============================================================================
# Point-process observation scenarios
# ============================================================================


def simulate_com_pp_scenario(n_time: int = N_TIME, seed: int = 42) -> dict:
    """COM-PP scenario: spike observation params switch between states.

    State 0: neurons 0-2 are modulated by theta, neurons 3-5 are silent.
    State 1: neurons 0-2 are silent, neurons 3-5 are modulated by beta.

    Different neuron subsets are active in each state, making the states
    trivially distinguishable from population firing patterns.
    """
    n_neurons = 6
    dt = 0.01

    # A: constant, uncoupled oscillators
    A_single = construct_common_oscillator_transition_matrix(
        freqs=FREQS, auto_regressive_coef=DAMPING, sampling_freq=SAMPLING_FREQ
    )
    A = jnp.stack([A_single, A_single], axis=2)

    # Q: constant
    Q_single = construct_common_oscillator_process_covariance(
        variance=jnp.array([0.1, 0.1])
    )
    Q = jnp.stack([Q_single, Q_single], axis=2)

    # Per-state spike params
    # State 0: neurons 0-2 fire actively (~10 Hz), neurons 3-5 are silent
    # State 1: neurons 0-2 are silent, neurons 3-5 fire actively (~10 Hz)
    # baseline=2.3 with dt=0.01 gives exp(2.3)*0.01 ~ 0.1 spikes/bin = 10 Hz
    baseline = jnp.full((n_neurons, N_DISCRETE_STATES), 2.3)
    # Make silent neurons very low rate
    baseline = baseline.at[3:, 0].set(-10.0)  # neurons 3-5 silent in state 0
    baseline = baseline.at[:3, 1].set(-10.0)  # neurons 0-2 silent in state 1

    weights = jnp.zeros((n_neurons, N_LATENT, N_DISCRETE_STATES))
    # State 0: neurons 0-2 see theta (moderate weights)
    weights = weights.at[0, 0, 0].set(0.5)
    weights = weights.at[1, 1, 0].set(0.5)
    weights = weights.at[2, 0, 0].set(0.3)
    # State 1: neurons 3-5 see beta (moderate weights)
    weights = weights.at[3, 2, 1].set(0.5)
    weights = weights.at[4, 3, 1].set(0.5)
    weights = weights.at[5, 2, 1].set(0.3)

    key = jax.random.PRNGKey(seed)
    spikes, true_continuous, true_discrete = simulate_switching_spike_oscillator(
        n_time=n_time,
        transition_matrices=A,
        process_covs=Q,
        discrete_transition_matrix=jnp.array(Z),
        spike_weights=weights,
        spike_baseline=baseline,
        dt=dt,
        key=key,
    )

    return {
        "spikes": spikes,
        "true_states": np.array(true_discrete),
        "true_continuous": np.array(true_continuous),
        "params": {
            "A": A, "Q": Q, "Z": Z,
            "n_oscillators": N_OSCILLATORS,
            "n_neurons": n_neurons,
            "n_discrete_states": N_DISCRETE_STATES,
            "sampling_freq": SAMPLING_FREQ,
            "dt": dt,
            "freqs": FREQS,
            "damping": DAMPING,
            "process_variance": jnp.array([0.1, 0.1]),
            "spike_baseline": baseline,
            "spike_weights": weights,
        },
    }


def simulate_cnm_pp_scenario(n_time: int = N_TIME, seed: int = 42) -> dict:
    """CNM-PP scenario: process noise Q switches, observed through spikes.

    State 0: independent noise (diagonal Q).
    State 1: correlated noise (off-diagonal coupling in Q).

    With shared spike params, the change in latent covariance structure
    is visible through correlated spiking patterns.
    """
    n_neurons = 24
    dt = 0.01

    # A: constant
    A_single = construct_common_oscillator_transition_matrix(
        freqs=FREQS, auto_regressive_coef=DAMPING, sampling_freq=SAMPLING_FREQ
    )
    A = jnp.stack([A_single, A_single], axis=2)

    # Q: state 0 low noise, state 1 moderate noise
    variance_0 = jnp.array([0.01, 0.01])
    variance_1 = jnp.array([0.1, 0.1])
    Q0 = construct_correlated_noise_process_covariance(
        variance=variance_0,
        phase_difference=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
        coupling_strength=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
    )
    Q1 = construct_correlated_noise_process_covariance(
        variance=variance_1,
        phase_difference=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
        coupling_strength=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
    )
    Q = jnp.stack([Q0, Q1], axis=2)

    # Per-state spike params — 24 neurons (6 per latent dim)
    # baseline=0 with dt=0.01 gives ~1 Hz per neuron when active
    # State 0: first half fires, second half silent
    # State 1: first half silent, second half fires
    key = jax.random.PRNGKey(seed)
    _, k2 = jax.random.split(key)
    half = n_neurons // 2
    baseline = jnp.concatenate([
        jnp.array([[0.0, -8.0]] * half),    # first half: active in state 0
        jnp.array([[-8.0, 0.0]] * half),    # second half: active in state 1
    ])
    weights = jnp.zeros((n_neurons, N_LATENT, N_DISCRETE_STATES))
    for s in range(N_DISCRETE_STATES):
        for i in range(n_neurons):
            weights = weights.at[i, i % N_LATENT, s].set(0.1)

    spikes, true_continuous, true_discrete = simulate_switching_spike_oscillator(
        n_time=n_time,
        transition_matrices=A,
        process_covs=Q,
        discrete_transition_matrix=jnp.array(Z),
        spike_weights=weights,
        spike_baseline=baseline,
        dt=dt,
        key=k2,
    )

    return {
        "spikes": spikes,
        "true_states": np.array(true_discrete),
        "true_continuous": np.array(true_continuous),
        "params": {
            "A": A, "Q": Q, "Z": Z,
            "n_oscillators": N_OSCILLATORS,
            "n_neurons": n_neurons,
            "n_discrete_states": N_DISCRETE_STATES,
            "sampling_freq": SAMPLING_FREQ,
            "dt": dt,
            "freqs": FREQS,
            "damping": DAMPING,
            "process_variance": jnp.stack([variance_0, variance_1], axis=1),
            "phase_difference": jnp.zeros((N_OSCILLATORS, N_OSCILLATORS, N_DISCRETE_STATES)),
            "coupling_strength": jnp.zeros((N_OSCILLATORS, N_OSCILLATORS, N_DISCRETE_STATES)),
            "spike_baseline": baseline,
            "spike_weights": weights,
        },
    }


def simulate_dim_pp_scenario(n_time: int = N_TIME, seed: int = 42) -> dict:
    """DIM-PP scenario: transition matrix A switches, observed through spikes.

    State 0: oscillator 1 drives oscillator 2 (osc1 -> osc2).
    State 1: oscillator 2 drives oscillator 1 (osc2 -> osc1).

    The reversal of coupling direction changes the latent dynamics,
    which is detectable through the spike patterns.
    """
    n_neurons = 24
    dt = 0.01

    # A: coupling direction reverses
    coupling_0 = jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)).at[1, 0].set(0.3)
    coupling_1 = jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)).at[0, 1].set(0.3)
    phase_diffs = jnp.zeros((N_OSCILLATORS, N_OSCILLATORS))

    A0 = construct_directed_influence_transition_matrix(
        freqs=FREQS, damping_coeffs=DAMPING,
        coupling_strengths=coupling_0, phase_diffs=phase_diffs,
        sampling_freq=SAMPLING_FREQ,
    )
    A1 = construct_directed_influence_transition_matrix(
        freqs=FREQS, damping_coeffs=DAMPING,
        coupling_strengths=coupling_1, phase_diffs=phase_diffs,
        sampling_freq=SAMPLING_FREQ,
    )
    A = jnp.stack([A0, A1], axis=2)

    # Q: constant
    Q_single = construct_common_oscillator_process_covariance(
        variance=jnp.array([0.1, 0.1])
    )
    Q = jnp.stack([Q_single, Q_single], axis=2)

    # Per-state spike params — 24 neurons, state-dependent baselines
    key = jax.random.PRNGKey(seed)
    _, k2 = jax.random.split(key)
    half = n_neurons // 2
    baseline = jnp.concatenate([
        jnp.array([[0.0, -8.0]] * half),    # first half: active in state 0
        jnp.array([[-8.0, 0.0]] * half),    # second half: active in state 1
    ])
    weights = jnp.zeros((n_neurons, N_LATENT, N_DISCRETE_STATES))
    for s in range(N_DISCRETE_STATES):
        for i in range(n_neurons):
            weights = weights.at[i, i % N_LATENT, s].set(0.1)

    spikes, true_continuous, true_discrete = simulate_switching_spike_oscillator(
        n_time=n_time,
        transition_matrices=A,
        process_covs=Q,
        discrete_transition_matrix=jnp.array(Z),
        spike_weights=weights,
        spike_baseline=baseline,
        dt=dt,
        key=k2,
    )

    return {
        "spikes": spikes,
        "true_states": np.array(true_discrete),
        "true_continuous": np.array(true_continuous),
        "params": {
            "A": A, "Q": Q, "Z": Z,
            "n_oscillators": N_OSCILLATORS,
            "n_neurons": n_neurons,
            "n_discrete_states": N_DISCRETE_STATES,
            "sampling_freq": SAMPLING_FREQ,
            "dt": dt,
            "freqs": FREQS,
            "damping": DAMPING,
            "process_variance": jnp.array([0.1, 0.1]),
            "phase_difference": jnp.stack([phase_diffs, phase_diffs], axis=2),
            "coupling_strength": jnp.stack([coupling_0, coupling_1], axis=2),
            "spike_baseline": baseline,
            "spike_weights": weights,
        },
    }
