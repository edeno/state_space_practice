"""Rigorous Verification of Hamiltonian Spike Model on Synthetic Data.

Metrics:
1. Hamiltonian Correlation: Correlation between ground truth H and learned H on a grid.
2. Latent Correlation (Phase Correlation): R^2 of recovered latents vs ground truth.
3. Spike Deviance: Predictive performance of the learned model.
4. Energy Conservation Error: Variance of H(x) along a predicted latent orbit.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from scipy.stats import pearsonr

from state_space_practice.nonlinear_dynamics import leapfrog_step, apply_mlp
from state_space_practice.hamiltonian_spikes import HamiltonianSpikeModel

def ground_truth_H(state: jnp.ndarray) -> jnp.ndarray:
    """Anharmonic Pendulum: H(q, p) = 0.5*p^2 + (1 - cos(q))"""
    q, p = state[0], state[1]
    return 0.5 * p**2 + (1.0 - jnp.cos(q))

def simulate_spike_data(n_time=2000, dt=0.01, n_neurons=20):
    """Simulate ground-truth nonlinear trajectory and spikes."""
    def step_fn(carry, _):
        x_prev = carry
        def h_apply(params, x): return ground_truth_H(x)
        x_next = leapfrog_step(x_prev, {}, h_apply, dt)
        return x_next, x_next

    x0 = jnp.array([2.0, 0.0])
    _, x_true = jax.lax.scan(step_fn, x0, jnp.arange(n_time))
    
    # Generate Spikes
    angles = jnp.linspace(0, 2*jnp.pi, n_neurons, endpoint=False)
    C_true = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1) * 2.0
    d_true = jnp.full((n_neurons,), -1.0)
    
    log_lambda = jnp.dot(x_true, C_true.T) + d_true
    rates = jnp.exp(log_lambda) * dt
    spikes = jax.random.poisson(jax.random.PRNGKey(1), rates)
    
    return x_true, spikes, C_true, d_true

def main():
    n_time = 2000
    dt = 0.01
    n_neurons = 20
    sampling_freq = 1.0 / dt
    
    x_true, spikes, C_gt, d_gt = simulate_spike_data(n_time=n_time, dt=dt, n_neurons=n_neurons)
    
    # Train/Test Split
    n_train = int(n_time * 0.8)
    spikes_train = spikes[:n_train]
    spikes_test = spikes[n_train:]
    x_test_true = x_true[n_train:]
    
    # Initialize Model
    model = HamiltonianSpikeModel(
        n_oscillators=1,
        n_sources=n_neurons,
        sampling_freq=sampling_freq,
        hidden_dims=[32, 32],
        seed=42
    )
    model._sgd_n_time = n_train
    
    print("Fitting HamiltonianSpikeModel (Deterministic -> Laplace-EKF)...")
    # 1. Faster deterministic pre-training
    model.fit_sgd(spikes_train, key=jax.random.PRNGKey(2), num_steps=200, verbose=True, l2_reg=1e-3, use_filter=False)
    
    # 2. Refined Laplace-EKF fine-tuning
    print("Fine-tuning with Laplace-EKF...")
    model.fit_sgd(spikes_train, key=jax.random.PRNGKey(3), num_steps=100, verbose=True, l2_reg=1e-3, use_filter=True)
    
    # 1. EVALUATE METRICS
    params, _ = model._build_param_spec()
    
    # Update mlp_params with learned omega for evaluation
    full_params = {**model.mlp_params, "omega": model.omega}
    
    # A. Hamiltonian Correlation (Grid-based)
    q_grid = jnp.linspace(-3, 3, 30)
    p_grid = jnp.linspace(-3, 3, 30)
    Q, P = jnp.meshgrid(q_grid, p_grid)
    states = jnp.stack([Q.flatten(), P.flatten()], axis=1)
    
    h_true = jax.vmap(ground_truth_H)(states)
    h_learned = jax.vmap(lambda s: apply_mlp(full_params, s))(states)
    
    # Normalize learned H to be zero at origin for fair correlation
    h_learned = h_learned - jnp.min(h_learned)
    h_corr, _ = pearsonr(np.array(h_true), np.array(h_learned))
    
    # B. Latent Trajectory Correlation (Test Set)
    # Rollout from test starting point
    def scan_fn(x_prev, _):
        x_next = model.transition_func(x_prev, full_params)
        return x_next, x_next
    _, x_recovered = jax.lax.scan(scan_fn, x_true[n_train], jnp.arange(n_time - n_train))
    
    lat_corr_q, _ = pearsonr(np.array(x_test_true[:, 0]), np.array(x_recovered[:, 0]))
    lat_corr_p, _ = pearsonr(np.array(x_test_true[:, 1]), np.array(x_recovered[:, 1]))
    
    # C. Predictive Log-Likelihood (Test Set)
    log_lambda_test = jnp.dot(x_recovered, model.C.T) + model.d
    rates_test = jnp.exp(log_lambda_test) * dt
    test_ll = jnp.sum(spikes_test * jnp.log(rates_test + 1e-10) - rates_test)
    
    # D. Energy Conservation Error
    learned_energies = jax.vmap(lambda s: apply_mlp(full_params, s))(x_recovered)
    energy_cv = jnp.std(learned_energies) / jnp.abs(jnp.mean(learned_energies))

    print("\n--- QUANTITATIVE METRICS ---")
    print(f"Learned Omega:                 {model.omega:.4f}")
    print(f"Hamiltonian Grid Correlation: {h_corr:.4f}")
    print(f"Latent Recovery Corr (q):      {lat_corr_q:.4f}")
    print(f"Latent Recovery Corr (p):      {lat_corr_p:.4f}")
    print(f"Test Log-Likelihood:           {test_ll:.2f}")
    print(f"Energy Conservation Error:     {energy_cv:.4e}")
    print("----------------------------\n")

    # Final Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 1].plot(x_test_true[:, 0], x_test_true[:, 1], 'k--', alpha=0.5, label="True Test")
    axes[0, 1].plot(x_recovered[:, 0], x_recovered[:, 1], 'r', label="Learned Rollout")
    axes[0, 1].set_title("Test Trajectory Recovery")
    axes[0, 1].legend()
    
    axes[1, 0].contourf(Q, P, h_true.reshape(Q.shape), levels=20)
    axes[1, 0].set_title("Ground Truth H")
    
    axes[1, 1].contourf(Q, P, h_learned.reshape(Q.shape), levels=20)
    axes[1, 1].set_title(f"Learned H (Corr={h_corr:.3f})")
    
    plt.tight_layout()
    plt.savefig("output/hamiltonian_spike_rigorous_verification.png")
    print("Verification complete. Results saved to output/hamiltonian_spike_rigorous_verification.png")

if __name__ == "__main__":
    main()
