"""Verification of Joint Hamiltonian Model (LFP + Spikes).

Scenario:
- 1 High-noise LFP channel (SNR = 0.5)
- 5 Very sparse neurons (avg 2 Hz)
- Combined, they should recover the Hamiltonian better than either alone.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from state_space_practice.nonlinear_dynamics import leapfrog_step, apply_mlp
from state_space_practice.hamiltonian_joint import JointHamiltonianModel

def ground_truth_H(state):
    return 0.5 * state[1]**2 + (1.0 - jnp.cos(state[0]))

def simulate_joint_data(n_time=2000, dt=0.01, n_lfp=1, n_spikes=5):
    # Latent
    def step_fn(carry, _):
        def h_apply(p, x): return ground_truth_H(x)
        x_next = leapfrog_step(carry, {}, h_apply, dt)
        return x_next, x_next
    _, x_true = jax.lax.scan(step_fn, jnp.array([1.5, 0.0]), jnp.arange(n_time))
    
    # LFP (Noisy)
    lfp = x_true[:, :n_lfp] + jax.random.normal(jax.random.PRNGKey(0), (n_time, n_lfp)) * 0.5
    
    # Spikes (Sparse)
    angles = jnp.linspace(0, 2*jnp.pi, n_spikes, endpoint=False)
    C_s = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1) * 1.5
    rates = jnp.exp(jnp.dot(x_true, C_s.T) - 2.0) * dt # Sparse: -2.0 baseline
    spikes = jax.random.poisson(jax.random.PRNGKey(1), rates)
    
    return x_true, lfp, spikes

def main():
    n_time = 2000
    dt = 0.01
    x_true, lfp, spikes = simulate_joint_data(n_time=n_time, dt=dt)
    
    model = JointHamiltonianModel(1, 1, 5, 1/dt)
    model._sgd_n_time = n_time
    
    print("Fitting JointHamiltonianModel (LFP + Spikes)...")
    # Pre-train deterministic
    model.fit_sgd(lfp, spikes, key=jax.random.PRNGKey(2), num_steps=200, use_filter=False, verbose=True)
    # Fine-tune Filtered
    model.fit_sgd(lfp, spikes, key=jax.random.PRNGKey(3), num_steps=100, use_filter=True, verbose=True)
    
    # Evaluate
    params, _ = model._build_param_spec()
    full_params = {**model.mlp_params, "omega": model.omega}
    means, _, _ = model.filter(lfp, spikes, params)
    
    # Grid for H
    q_grid = jnp.linspace(-3, 3, 30); p_grid = jnp.linspace(-3, 3, 30)
    Q, P = jnp.meshgrid(q_grid, p_grid); states = jnp.stack([Q.flatten(), P.flatten()], axis=1)
    h_true = jax.vmap(ground_truth_H)(states)
    h_learned = jax.vmap(lambda s: apply_mlp(full_params, s))(states)
    h_corr, _ = pearsonr(np.array(h_true), np.array(h_learned))
    
    lat_corr, _ = pearsonr(np.array(x_true[:, 0]), np.array(means[:, 0]))
    
    print(f"\n--- JOINT METRICS ---")
    print(f"Hamiltonian Correlation: {h_corr:.4f}")
    print(f"Latent Recovery Corr:    {lat_corr:.4f}")
    print(f"Learned Omega:           {model.omega:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x_true[:, 0], x_true[:, 1], 'k--', alpha=0.3, label="True")
    axes[0].plot(means[:, 0], means[:, 1], 'r', label="Recovered")
    axes[0].set_title("Joint Latent Recovery")
    
    axes[1].contourf(Q, P, h_learned.reshape(Q.shape), levels=20)
    axes[1].set_title(f"Learned Joint Hamiltonian (Corr={h_corr:.3f})")
    
    plt.savefig("output/hamiltonian_joint_verification.png")
    print("Results saved to output/hamiltonian_joint_verification.png")

if __name__ == "__main__":
    main()
