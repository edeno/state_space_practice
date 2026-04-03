"""Generative simulation scenarios for validating switching oscillator models.

Each scenario simulates data from a specific model's generative process with
parameters chosen to make discrete states distinguishable. These are used by
test_scenario_recovery.py to verify that models can recover the ground truth.

All scenarios use:
- 2 discrete states
- 2 oscillators (theta=8Hz, beta=25Hz)
- sampling_freq=100Hz
- Z diagonal=0.98 (long dwell ~50 steps)

Gaussian scenarios use n_time=3000 (sufficient for continuous observations).
Point-process scenarios use n_time=10000 (more data needed for sparse spikes).

Design principles:
- Each scenario tests the SPECIFIC mechanism of its model type
- States are distinguishable through the model's switching parameter, not
  through trivial observation differences (e.g., silent/active neurons)
- E-step with true parameters achieves >=0.70 accuracy for all scenarios
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
N_TIME_GAUSSIAN = 3000
N_TIME_PP = 10000
Z = np.array([[0.98, 0.02], [0.02, 0.98]])


# ============================================================================
# Gaussian observation scenarios
# ============================================================================


def simulate_com_scenario(n_time: int = N_TIME_GAUSSIAN, seed: int = 42) -> dict:
    """COM scenario: measurement matrix H switches between states.

    State 0: sources observe theta oscillations (8 Hz).
    State 1: sources observe beta oscillations (25 Hz).

    The frequency content of the observations switches dramatically.
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
    H[0, 0, 0] = 0.5
    H[1, 1, 0] = 0.5
    H[0, 2, 1] = 0.5
    H[1, 3, 1] = 0.5

    # R: low observation noise
    R_single = np.eye(n_sources) * 0.05
    R = np.stack([R_single, R_single], axis=2)

    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal(N_LATENT)
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


def simulate_cnm_scenario(n_time: int = N_TIME_GAUSSIAN, seed: int = 42) -> dict:
    """CNM scenario: process noise covariance Q switches between states.

    State 0: independent noise (diagonal Q).
    State 1: correlated noise between oscillators (off-diagonal Q blocks).

    Tests the actual correlation structure mechanism of CNM, not just
    variance contrast.
    """
    n_sources = N_OSCILLATORS

    # A: constant, uncoupled oscillators
    A_single = np.array(construct_common_oscillator_transition_matrix(
        freqs=FREQS, auto_regressive_coef=DAMPING, sampling_freq=SAMPLING_FREQ
    ))
    A = np.stack([A_single, A_single], axis=2)

    # Q: state 0 = independent, state 1 = correlated
    # Higher variance (0.3) allows stronger coupling while staying PSD.
    # Q1 eigvals [0.1, 0.1, 0.5, 0.5] vs Q0 [0.3, 0.3, 0.3, 0.3] —
    # same total variance but different structure.
    variance = jnp.array([0.3, 0.3])
    coupling_strength_corr = jnp.zeros(
        (N_OSCILLATORS, N_OSCILLATORS)
    ).at[0, 1].set(0.2).at[1, 0].set(0.2)
    Q0 = np.array(construct_correlated_noise_process_covariance(
        variance=variance,
        phase_difference=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
        coupling_strength=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
    ))
    Q1 = np.array(construct_correlated_noise_process_covariance(
        variance=variance,
        phase_difference=jnp.zeros((N_OSCILLATORS, N_OSCILLATORS)),
        coupling_strength=coupling_strength_corr,
    ))
    Q = np.stack([Q0, Q1], axis=2)

    # H: constant
    H_single = np.array(construct_correlated_noise_measurement_matrix(n_sources))
    H = np.stack([H_single, H_single], axis=2)

    # R: low observation noise
    R_single = np.eye(n_sources) * 0.05
    R = np.stack([R_single, R_single], axis=2)

    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal(N_LATENT)
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
            "process_variance": np.stack(
                [np.array(variance)] * N_DISCRETE_STATES, axis=1
            ),
            "measurement_variance": 0.05,
            "phase_difference": np.zeros(
                (N_OSCILLATORS, N_OSCILLATORS, N_DISCRETE_STATES)
            ),
            "coupling_strength": np.stack(
                [
                    np.zeros((N_OSCILLATORS, N_OSCILLATORS)),
                    np.array(coupling_strength_corr),
                ],
                axis=2,
            ),
        },
    }


def simulate_dim_scenario(n_time: int = N_TIME_GAUSSIAN, seed: int = 42) -> dict:
    """DIM scenario: transition matrix A switches between states.

    State 0: oscillator 1 drives oscillator 2 (osc1 -> osc2).
    State 1: oscillator 2 drives oscillator 1 (osc2 -> osc1).

    The directed coupling between oscillators reverses completely.
    """
    n_sources = N_OSCILLATORS

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

    Q_single = np.array(construct_common_oscillator_process_covariance(
        variance=jnp.array([0.1, 0.1])
    ))
    Q = np.stack([Q_single, Q_single], axis=2)

    H_single = np.array(construct_directed_influence_measurement_matrix(n_sources))
    H = np.stack([H_single, H_single], axis=2)

    R_single = np.eye(n_sources) * 0.05
    R = np.stack([R_single, R_single], axis=2)

    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal(N_LATENT)
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
            "phase_difference": np.stack(
                [np.array(phase_diffs)] * N_DISCRETE_STATES, axis=2
            ),
            "coupling_strength": np.stack(
                [np.array(coupling_0), np.array(coupling_1)], axis=2
            ),
        },
    }


# ============================================================================
# Point-process observation scenarios
# ============================================================================


def simulate_com_pp_scenario(n_time: int = N_TIME_PP, seed: int = 42) -> dict:
    """COM-PP scenario: spike observation params switch between states.

    All neurons are active in both states with similar overall firing rates.
    State 0: neurons couple to the theta oscillator (8 Hz modulation).
    State 1: neurons couple to the beta oscillator (25 Hz modulation).

    The model must learn which oscillator drives the neurons in each state.
    E-step accuracy with true parameters: ~0.81 (30 neurons, 10K steps).
    """
    n_neurons = 30
    dt = 0.01

    A_single = construct_common_oscillator_transition_matrix(
        freqs=FREQS, auto_regressive_coef=DAMPING, sampling_freq=SAMPLING_FREQ
    )
    A = jnp.stack([A_single, A_single], axis=2)

    Q_single = construct_common_oscillator_process_covariance(
        variance=jnp.array([0.1, 0.1])
    )
    Q = jnp.stack([Q_single, Q_single], axis=2)

    # Per-state weights: all neurons active at ~4.5 Hz in both states
    # State 0: couple to theta (latent dims 0-1)
    # State 1: couple to beta (latent dims 2-3)
    baseline = jnp.full((n_neurons, N_DISCRETE_STATES), 1.5)
    weights = jnp.zeros((n_neurons, N_LATENT, N_DISCRETE_STATES))
    for i in range(n_neurons):
        weights = weights.at[i, i % 2, 0].set(0.5)       # theta dims
        weights = weights.at[i, 2 + i % 2, 1].set(0.5)   # beta dims

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


def simulate_cnm_pp_scenario(n_time: int = N_TIME_PP, seed: int = 42) -> dict:
    """CNM-PP scenario: process noise Q switches, observed through spikes.

    State 0: low process noise (Q variance = 0.05).
    State 1: high process noise (Q variance = 0.3).

    Uses variance contrast rather than correlation structure because the
    Laplace-EKF approximation for point-process observations is insensitive
    to off-diagonal Q entries (correlation structure is not identifiable
    from Poisson observations with per-step Laplace updates). The Gaussian
    CNM scenario tests correlation structure recovery.

    Shared spike params — state differences are detected purely through
    the dynamics, not observation model differences.
    E-step accuracy with true parameters: ~0.71 (20 neurons, 10K steps).
    """
    n_neurons = 20
    dt = 0.01

    A_single = construct_common_oscillator_transition_matrix(
        freqs=FREQS, auto_regressive_coef=DAMPING, sampling_freq=SAMPLING_FREQ
    )
    A = jnp.stack([A_single, A_single], axis=2)

    # Q: variance contrast — low vs high noise
    Q0 = construct_common_oscillator_process_covariance(
        variance=jnp.array([0.05, 0.05])
    )
    Q1 = construct_common_oscillator_process_covariance(
        variance=jnp.array([0.3, 0.3])
    )
    Q = jnp.stack([Q0, Q1], axis=2)

    # Shared spike params: one neuron per latent dim, ~7.4 Hz
    baseline = jnp.full(n_neurons, 2.0)
    weights = jnp.zeros((n_neurons, N_LATENT))
    for i in range(n_neurons):
        weights = weights.at[i, i % N_LATENT].set(0.3)

    key = jax.random.PRNGKey(seed)
    _, k2 = jax.random.split(key)
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
            "process_variance": jnp.stack(
                [jnp.array([0.05, 0.05]), jnp.array([0.3, 0.3])], axis=1
            ),
            "phase_difference": jnp.zeros(
                (N_OSCILLATORS, N_OSCILLATORS, N_DISCRETE_STATES)
            ),
            "coupling_strength": jnp.zeros(
                (N_OSCILLATORS, N_OSCILLATORS, N_DISCRETE_STATES)
            ),
            "spike_baseline": baseline,
            "spike_weights": weights,
        },
    }


def simulate_dim_pp_scenario(n_time: int = N_TIME_PP, seed: int = 42) -> dict:
    """DIM-PP scenario: transition matrix A switches, observed through spikes.

    State 0: oscillator 1 drives oscillator 2 (osc1 -> osc2).
    State 1: oscillator 2 drives oscillator 1 (osc2 -> osc1).

    Shared spike params — state differences are detected purely through
    the coupling direction change in the dynamics.
    E-step accuracy with true parameters: ~0.80 (20 neurons, 10K steps).
    """
    n_neurons = 20
    dt = 0.01

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

    Q_single = construct_common_oscillator_process_covariance(
        variance=jnp.array([0.1, 0.1])
    )
    Q = jnp.stack([Q_single, Q_single], axis=2)

    # Shared spike params: ~7.4 Hz, one neuron per latent dim
    baseline = jnp.full(n_neurons, 2.0)
    weights = jnp.zeros((n_neurons, N_LATENT))
    for i in range(n_neurons):
        weights = weights.at[i, i % N_LATENT].set(0.3)

    key = jax.random.PRNGKey(seed)
    _, k2 = jax.random.split(key)
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
            "phase_difference": jnp.stack(
                [phase_diffs, phase_diffs], axis=2
            ),
            "coupling_strength": jnp.stack(
                [coupling_0, coupling_1], axis=2
            ),
            "spike_baseline": baseline,
            "spike_weights": weights,
        },
    }
