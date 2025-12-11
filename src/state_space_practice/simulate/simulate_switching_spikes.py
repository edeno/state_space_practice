"""Simulation utilities for switching spike-based oscillator networks.

This module provides functions to generate synthetic data from the switching
point-process oscillator model, useful for testing parameter recovery and
validating inference algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from jax.random import PRNGKey


def simulate_switching_spike_oscillator(
    n_time: int,
    transition_matrices: Array,
    process_covs: Array,
    discrete_transition_matrix: Array,
    spike_weights: Array,
    spike_baseline: Array,
    dt: float,
    key: PRNGKey,
    init_mean: Array | None = None,
    init_cov: Array | None = None,
    init_discrete_prob: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Simulate spikes from a switching oscillator network model.

    Generates synthetic data from a switching linear dynamical system (SLDS)
    with point-process (spike) observations. The model has:

    - Latent continuous dynamics: x_t = A_{s_t} @ x_{t-1} + w_t
    - Discrete state transitions: s_t ~ Categorical(Z[s_{t-1}, :])
    - Spike observations: y_{n,t} ~ Poisson(exp(b_n + c_n @ x_t) * dt)

    Parameters
    ----------
    n_time : int
        Number of time steps to simulate.
    transition_matrices : Array, shape (n_latent, n_latent, n_discrete_states)
        State transition matrices A_s for each discrete state s.
    process_covs : Array, shape (n_latent, n_latent, n_discrete_states)
        Process noise covariances Q_s for each discrete state s.
    discrete_transition_matrix : Array, shape (n_discrete_states, n_discrete_states)
        Discrete state transition probabilities Z[i,j] = P(s_t=j | s_{t-1}=i).
        Rows should sum to 1.
    spike_weights : Array, shape (n_neurons, n_latent)
        Linear weights C mapping latent state to log firing rates.
    spike_baseline : Array, shape (n_neurons,)
        Baseline log firing rates b for each neuron.
    dt : float
        Time bin width in seconds.
    key : PRNGKey
        JAX random key for reproducibility.
    init_mean : Array, shape (n_latent,), optional
        Initial continuous state mean. Defaults to zeros.
    init_cov : Array, shape (n_latent, n_latent), optional
        Initial continuous state covariance. Defaults to identity.
    init_discrete_prob : Array, shape (n_discrete_states,), optional
        Initial discrete state probabilities. Defaults to uniform.

    Returns
    -------
    spikes : Array, shape (n_time, n_neurons)
        Simulated spike counts at each time step.
    true_states : Array, shape (n_time, n_latent)
        True latent continuous states.
    true_discrete_states : Array, shape (n_time,)
        True discrete state sequence (integer indices).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax
    >>> n_time, n_neurons, n_latent, n_discrete_states = 100, 5, 4, 2
    >>> key = jax.random.PRNGKey(0)
    >>> A = jnp.stack([jnp.eye(n_latent) * 0.99] * n_discrete_states, axis=-1)
    >>> Q = jnp.stack([jnp.eye(n_latent) * 0.01] * n_discrete_states, axis=-1)
    >>> Z = jnp.array([[0.95, 0.05], [0.05, 0.95]])
    >>> C = jax.random.normal(key, (n_neurons, n_latent)) * 0.1
    >>> b = jnp.zeros(n_neurons)
    >>> spikes, states, discrete = simulate_switching_spike_oscillator(
    ...     n_time, A, Q, Z, C, b, dt=0.02, key=key
    ... )
    """
    n_latent = transition_matrices.shape[0]
    n_discrete_states = transition_matrices.shape[-1]

    # Set defaults for initial conditions
    if init_mean is None:
        init_mean = jnp.zeros(n_latent)
    if init_cov is None:
        init_cov = jnp.eye(n_latent)
    if init_discrete_prob is None:
        init_discrete_prob = jnp.ones(n_discrete_states) / n_discrete_states

    # Split keys for different random operations
    key, key_init_state, key_init_discrete = jax.random.split(key, 3)

    # Sample initial continuous state
    x_0 = jax.random.multivariate_normal(key_init_state, init_mean, init_cov)

    # Sample initial discrete state
    s_0 = jax.random.categorical(key_init_discrete, jnp.log(init_discrete_prob))

    def _step(carry: tuple[Array, Array, PRNGKey], _: None) -> tuple[tuple[Array, Array, PRNGKey], tuple[Array, Array, Array]]:
        """Single simulation step."""
        x_prev, s_prev, key = carry

        # Split key for this step
        key, key_discrete, key_continuous, key_spikes = jax.random.split(key, 4)

        # Sample next discrete state: s_t ~ Categorical(Z[s_{t-1}, :])
        s_t = jax.random.categorical(
            key_discrete, jnp.log(discrete_transition_matrix[s_prev])
        )

        # Get dynamics for current discrete state
        A_s = transition_matrices[:, :, s_t]
        Q_s = process_covs[:, :, s_t]

        # Sample continuous state: x_t = A_s @ x_{t-1} + w_t, w_t ~ N(0, Q_s)
        x_mean = A_s @ x_prev
        x_t = jax.random.multivariate_normal(key_continuous, x_mean, Q_s)

        # Compute firing rates: lambda_n = exp(b_n + c_n @ x_t)
        log_rates = spike_baseline + spike_weights @ x_t
        rates = jnp.exp(log_rates) * dt

        # Sample spikes: y_n ~ Poisson(lambda_n * dt)
        y_t = jax.random.poisson(key_spikes, rates).astype(jnp.float64)

        return (x_t, s_t, key), (y_t, x_t, s_t)

    # Run simulation using scan
    _, (spikes, true_states, true_discrete_states) = jax.lax.scan(
        _step,
        (x_0, s_0, key),
        None,
        length=n_time,
    )

    return spikes, true_states, true_discrete_states
