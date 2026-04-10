"""Verification of Hamiltonian LFP Model on Synthetic Data.

This script:
1. Simulates a nonlinear oscillator (Anharmonic Pendulum).
2. Generates synthetic LFP data.
3. Fits a HamiltonianLFPModel using EKF-based SGD.
4. Visualizes the recovered trajectory and learned energy landscape.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

from state_space_practice.nonlinear_dynamics import leapfrog_step, apply_mlp
from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel

def ground_truth_H(state: jnp.ndarray) -> jnp.ndarray:
    """Anharmonic Pendulum: H(q, p) = 0.5*p^2 + (1 - cos(q))"""
    q, p = state[0], state[1]
    # Potential: 1 - cos(q) (Nonlinear)
    # Kinetic: 0.5 * p^2
    return 0.5 * p**2 + (1.0 - jnp.cos(q))

def simulate_data(n_time=1000, dt=0.05, noise_std=0.01):
    """Simulate ground-truth nonlinear trajectory."""
    def step_fn(carry, _):
        x_prev = carry
        # Use leapfrog with ground truth H
        def h_apply(params, x): return ground_truth_H(x)
        x_next = leapfrog_step(x_prev, {}, h_apply, dt)
        # Add a tiny bit of process noise
        x_next = x_next + jax.random.normal(jax.random.PRNGKey(0), (2,)) * 1e-4
        return x_next, x_next

    x0 = jnp.array([1.5, 0.0]) # Start with high amplitude
    _, x_true = jax.lax.scan(step_fn, x0, jnp.arange(n_time))
    
    # Generate LFP (linear projection + noise)
    C_true = jnp.array([[1.0, 0.0]]) # Observe position only
    lfp = jnp.dot(x_true, C_true.T) + jax.random.normal(jax.random.PRNGKey(1), (n_time, 1)) * noise_std
    
    return x_true, lfp

def main():
    n_time = 2000
    dt = 0.05
    sampling_freq = 1.0 / dt
    
    print(f"Simulating {n_time} steps of nonlinear oscillator...")
    x_true, lfp = simulate_data(n_time=n_time, dt=dt)
    
    # 2. Initialize Model
    model = HamiltonianLFPModel(
        n_oscillators=1,
        n_sources=1,
        sampling_freq=sampling_freq,
        hidden_dims=[32, 32],
        seed=42
    )
    model.obs_noise_std = 0.05 # Slightly higher than true for robustness
    model._sgd_n_time = n_time
    
    print("Fitting HamiltonianLFPModel (EKF-based SGD)...")
    # Pre-train for 100 steps with deterministic rollout (faster)
    history_det = model.fit_sgd(lfp, key=jax.random.PRNGKey(2), num_steps=200, use_filter=False, verbose=True)
    
    # Fine-tune with EKF for 100 steps
    print("Fine-tuning with EKF marginal likelihood...")
    history_ekf = model.fit_sgd(lfp, key=jax.random.PRNGKey(3), num_steps=100, use_filter=True, verbose=True)
    
    # 3. Filter the data to get latent estimates
    params, _ = model._build_param_spec()
    means, covs, _ = model.filter(lfp, params)
    
    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time series of LFP
    axes[0, 0].plot(lfp[:200], 'k.', alpha=0.3, label="Observed LFP")
    axes[0, 0].plot(jnp.dot(means[:200], model.C.T) + model.d, 'r', label="Filtered Mean")
    axes[0, 0].set_title("LFP Recovery (First 200 steps)")
    axes[0, 0].legend()
    
    # Latent Phase Space
    axes[0, 1].plot(x_true[:, 0], x_true[:, 1], 'k--', alpha=0.5, label="True Latent")
    axes[0, 1].plot(means[:, 0], means[:, 1], 'r', label="Recovered Latent")
    axes[0, 1].set_title("Latent Phase Space")
    axes[0, 1].set_xlabel("q (Position)")
    axes[0, 1].set_ylabel("p (Momentum)")
    axes[0, 1].legend()
    
    # Energy Landscape Comparison
    q_grid = jnp.linspace(-3, 3, 50)
    p_grid = jnp.linspace(-3, 3, 50)
    Q, P = jnp.meshgrid(q_grid, p_grid)
    states = jnp.stack([Q.flatten(), P.flatten()], axis=1)
    
    # True H
    h_true = jax.vmap(ground_truth_H)(states).reshape(Q.shape)
    axes[1, 0].contourf(Q, P, h_true, levels=20)
    axes[1, 0].set_title("Ground Truth H(q, p)")
    
    # Learned H
    h_learned = jax.vmap(lambda s: apply_mlp(model.mlp_params, s))(states).reshape(Q.shape)
    axes[1, 1].contourf(Q, P, h_learned, levels=20)
    axes[1, 1].set_title("Learned Hamiltonian (MLP)")
    
    plt.tight_layout()
    plt.savefig("output/hamiltonian_lfp_verification.png")
    print("Verification complete. Results saved to output/hamiltonian_lfp_verification.png")

if __name__ == "__main__":
    main()
