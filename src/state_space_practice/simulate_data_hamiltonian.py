"""Synthetic Data Simulation for Hamiltonian Oscillators.
"""

from typing import Dict, Any, NamedTuple
import jax
import jax.numpy as jnp
from jax import Array

from state_space_practice.hamiltonian_models import (
    init_structured_hamiltonian_params, 
    symplectic_transition
)

class HamiltonianSimResult(NamedTuple):
    latent_states: Array
    spikes: Array
    design_matrix: Array
    true_params: Dict[str, Any]

def simulate_hamiltonian_spike_data(
    seed: int,
    n_time: int,
    n_oscillators: int,
    n_neurons: int,
    dt: float = 0.01,
    process_noise_std: float = 0.01,
) -> HamiltonianSimResult:
    """Simulate ground-truth Hamiltonian oscillator spikes."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # 1. Initialize Dynamics
    true_h_params = init_structured_hamiltonian_params(n_oscillators, k1)
    # Increase stiffness for interesting dynamics
    true_h_params["log_stiffness"] = true_h_params["log_stiffness"] + 1.0
    
    # 2. Initialize Readout
    # C: (n_neurons, 2 * n_oscillators)
    C_true = jax.random.normal(k2, (n_neurons, 2 * n_oscillators)) * 1.0
    d_true = jnp.full((n_neurons,), -1.0) # Baseline sparse rate
    
    # 3. Simulate Latents
    def step_fn(carry, k):
        x_prev = carry
        # Deterministic step
        x_next_det = symplectic_transition(x_prev, true_h_params, dt)
        # Add process noise
        noise = jax.random.normal(k, x_prev.shape) * process_noise_std
        x_next = x_next_det + noise
        return x_next, x_next

    x0 = jax.random.normal(k3, (2 * n_oscillators,)) * 0.5
    latent_keys = jax.random.split(k4, n_time)
    _, x_traj = jax.lax.scan(step_fn, x0, latent_keys)
    
    # 4. Generate Spikes
    log_lambda = jnp.dot(x_traj, C_true.T) + d_true
    rates = jnp.exp(log_lambda) * dt
    
    k5 = jax.random.fold_in(key, 4)
    spikes = jax.random.poisson(k5, rates)
    
    # Design matrix: (n_time, n_neurons, n_latent)
    # For a simple linear log-readout, it's just repeating the latents for each neuron
    design_matrix = jnp.broadcast_to(x_traj[:, None, :], (n_time, n_neurons, 2 * n_oscillators))
    
    return HamiltonianSimResult(
        latent_states=x_traj,
        spikes=spikes,
        design_matrix=design_matrix,
        true_params={"hamiltonian": true_h_params, "C": C_true, "d": d_true}
    )
