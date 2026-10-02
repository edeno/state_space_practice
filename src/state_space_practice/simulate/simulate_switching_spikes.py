"""Simulation utilities for switching spike-based oscillator networks.

This module provides functions to generate synthetic data from the switching
point-process oscillator model, useful for testing parameter recovery and
validating inference algorithms.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def simulate_switching_spike_oscillator(
    n_time: int,
    transition_matrices: ArrayLike,
    process_covs: ArrayLike,
    discrete_transition_matrix: ArrayLike,
    spike_weights: ArrayLike,
    spike_baseline: ArrayLike,
    dt: float,
    key: Array,
    init_mean: ArrayLike | None = None,
    init_cov: ArrayLike | None = None,
    init_discrete_prob: ArrayLike | None = None,
) -> tuple[Array, Array, Array]:
    """Simulate spikes from a switching oscillator network model.

    Generates synthetic data from a switching linear dynamical system (SLDS)
    with point-process (spike) observations. The model has:

    - Initial state: x_1 ~ N(init_mean, init_cov),
      s_1 ~ Categorical(init_discrete_prob)
    - Latent continuous dynamics: x_t = A_{s_t} @ x_{t-1} + w_t
      for t > 1
    - Discrete state transitions: s_t ~ Categorical(Z[s_{t-1}, :])
      for t > 1
    - Spike observations: y_{n,t} ~ Poisson(exp(b_n + c_n @ x_t) * dt)

    Returned index 0 is the first observation time. No transition is applied
    before emitting it, matching the x_1 convention used by the switching
    point-process filter.

    Parameters
    ----------
    n_time : int
        Number of time steps to simulate.
    transition_matrices : ArrayLike, shape (n_latent, n_latent, n_discrete_states)
        State transition matrices A_s for each discrete state s.
    process_covs : ArrayLike, shape (n_latent, n_latent, n_discrete_states)
        Process noise covariances Q_s for each discrete state s.
    discrete_transition_matrix : ArrayLike, shape (n_discrete_states, n_discrete_states)
        Discrete state transition probabilities Z[i,j] = P(s_t=j | s_{t-1}=i).
        Rows should sum to 1.
    spike_weights : ArrayLike, shape (n_neurons, n_latent) or (n_neurons, n_latent, n_discrete_states)
        Linear weights C mapping latent state to log firing rates.
        If 3D, per-state weights are indexed by the current discrete state.
    spike_baseline : ArrayLike, shape (n_neurons,) or (n_neurons, n_discrete_states)
        Baseline log firing rates b for each neuron.
        If 2D, per-state baselines are indexed by the current discrete state.
    dt : float
        Time bin width in seconds.
    key : Array
        JAX random key for reproducibility.
    init_mean : ArrayLike, shape (n_latent,), optional
        Initial continuous state mean. Defaults to zeros.
    init_cov : ArrayLike, shape (n_latent, n_latent), optional
        Initial continuous state covariance. Defaults to identity.
    init_discrete_prob : ArrayLike, shape (n_discrete_states,), optional
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
    if n_time <= 0:
        raise ValueError(f"n_time must be positive, got {n_time}.")

    transition_matrices = jnp.asarray(transition_matrices)
    process_covs = jnp.asarray(process_covs)
    discrete_transition_matrix = jnp.asarray(discrete_transition_matrix)
    spike_weights = jnp.asarray(spike_weights)
    spike_baseline = jnp.asarray(spike_baseline)

    n_latent = transition_matrices.shape[0]
    n_discrete_states = transition_matrices.shape[-1]

    # Determine if spike params are per-state
    per_state_spikes = spike_weights.ndim == 3

    # Set defaults for initial conditions
    if init_mean is None:
        init_mean = jnp.zeros(n_latent)
    if init_cov is None:
        init_cov = jnp.eye(n_latent)
    if init_discrete_prob is None:
        init_discrete_prob = jnp.ones(n_discrete_states) / n_discrete_states

    # Split keys for different random operations
    key, key_init_state, key_init_discrete, key_init_spikes = jax.random.split(key, 4)

    # Sample initial continuous state
    x_0 = jax.random.multivariate_normal(key_init_state, init_mean, init_cov)

    # Sample initial discrete state
    s_0 = jax.random.categorical(key_init_discrete, jnp.log(init_discrete_prob))

    def _sample_spikes(x_t: Array, s_t: Array, key_spikes: Array) -> Array:
        """Sample spikes at the current state/time."""
        # Get spike params for current discrete state (or shared)
        if per_state_spikes:
            b_s = spike_baseline[:, s_t]
            c_s = spike_weights[:, :, s_t]
        else:
            b_s = spike_baseline
            c_s = spike_weights

        # Compute firing rates: lambda_n = exp(b_n + c_n @ x_t)
        log_rates = b_s + c_s @ x_t
        rates = jnp.exp(log_rates) * dt

        # Sample spikes: y_n ~ Poisson(lambda_n * dt)
        return jax.random.poisson(key_spikes, rates).astype(jnp.float64)

    y_0 = _sample_spikes(x_0, s_0, key_init_spikes)

    def _step(
        carry: tuple[Array, Array, Array],
        _: None,
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array]]:
        """Single transition and observation step."""
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

        y_t = _sample_spikes(x_t, s_t, key_spikes)

        return (x_t, s_t, key), (y_t, x_t, s_t)

    # Run transitions for t=2,...,T. The initial sample is prepended below.
    _, (spikes_rest, true_states_rest, true_discrete_states_rest) = jax.lax.scan(
        _step,
        (x_0, s_0, key),
        None,
        length=n_time - 1,
    )

    spikes = jnp.concatenate([y_0[None], spikes_rest], axis=0)
    true_states = jnp.concatenate([x_0[None], true_states_rest], axis=0)
    true_discrete_states = jnp.concatenate(
        [jnp.asarray(s_0)[None], true_discrete_states_rest], axis=0
    )

    return spikes, true_states, true_discrete_states
