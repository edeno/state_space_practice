"""Master Verification Suite for all Hamiltonian Models.

Standardized rigorous evaluation of:
1. HamiltonianLFPModel
2. HamiltonianSpikeModel
3. JointHamiltonianModel
4. SwitchingHamiltonianJointModel
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Dict, Any

from state_space_practice.nonlinear_dynamics import leapfrog_step, apply_mlp
from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel
from state_space_practice.hamiltonian_spikes import HamiltonianSpikeModel
from state_space_practice.hamiltonian_joint import JointHamiltonianModel
from state_space_practice.hamiltonian_switching import SwitchingHamiltonianJointModel

# --- Ground Truths ---

def h_pendulum(state):
    return 0.5 * state[1]**2 + (1.0 - jnp.cos(state[0]))

def h_stiff(state):
    # Faster, more quadratic
    return 0.5 * state[1]**2 + 2.0 * state[0]**2

def simulate_trajectory(h_func, n_time, dt, x0=jnp.array([1.5, 0.0])):
    def step_fn(carry, _):
        def h_apply(p, x): return h_func(x)
        x_next = leapfrog_step(carry, {}, h_apply, dt)
        return x_next, x_next
    _, x_traj = jax.lax.scan(step_fn, x0, jnp.arange(n_time))
    return x_traj

# --- Evaluation Core ---

def compute_metrics(model, x_true, learned_h_params, grid_size=30):
    # 1. Hamiltonian Grid Correlation
    q_grid = jnp.linspace(-3, 3, grid_size)
    p_grid = jnp.linspace(-3, 3, grid_size)
    Q, P = jnp.meshgrid(q_grid, p_grid)
    states = jnp.stack([Q.flatten(), P.flatten()], axis=1)
    
    # We assume state 0 if switching
    if "Z" in learned_h_params:
        # Extract first state params
        h_p = {**jax.tree_util.tree_map(lambda x: x[0], learned_h_params["mlp"]), "omega": learned_h_params["omega"][0]}
    else:
        h_p = {**learned_h_params["mlp"], "omega": learned_h_params["omega"]}
        
    h_gt = jax.vmap(h_pendulum)(states)
    h_learned = jax.vmap(lambda s: apply_mlp(h_p, s))(states)
    h_corr, _ = pearsonr(np.array(h_gt), np.array(h_learned))
    
    return {"H-Corr": h_corr}

# --- Standardized Test Loop ---

def run_verification():
    n_time = 1500
    dt = 0.02
    sampling_freq = 1.0 / dt
    results = []

    print("Generating synthetic data (Anharmonic Pendulum)...")
    x_true = simulate_trajectory(h_pendulum, n_time, dt)
    
    # Generate Observations
    lfp = x_true[:, :1] + jax.random.normal(jax.random.PRNGKey(0), (n_time, 1)) * 0.1
    angles = jnp.linspace(0, 2*jnp.pi, 10, endpoint=False)
    C_s = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1) * 1.5
    rates = jnp.exp(jnp.dot(x_true, C_s.T) - 1.0) * dt
    spikes = jax.random.poisson(jax.random.PRNGKey(1), rates)

    # 1. VERIFY LFP MODEL
    print("\n[1/4] Verifying LFP-only Model...")
    m_lfp = HamiltonianLFPModel(1, 1, sampling_freq)
    m_lfp.fit_sgd(lfp, key=jax.random.PRNGKey(2), num_steps=200, verbose=False)
    p_lfp, _ = m_lfp._build_param_spec()
    res_lfp = compute_metrics(m_lfp, x_true, p_lfp)
    res_lfp["Model"] = "LFP-only"
    results.append(res_lfp)

    # 2. VERIFY SPIKE MODEL
    print("[2/4] Verifying Spike-only Model...")
    m_spk = HamiltonianSpikeModel(1, 10, sampling_freq)
    m_spk.fit_sgd(spikes, key=jax.random.PRNGKey(3), num_steps=200, verbose=False)
    p_spk, _ = m_spk._build_param_spec()
    res_spk = compute_metrics(m_spk, x_true, p_spk)
    res_spk["Model"] = "Spike-only"
    results.append(res_spk)

    # 3. VERIFY JOINT MODEL
    print("[3/4] Verifying Joint Model...")
    m_jnt = JointHamiltonianModel(1, 1, 10, sampling_freq)
    m_jnt.fit_sgd(lfp, spikes, key=jax.random.PRNGKey(4), num_steps=200, verbose=False)
    p_jnt, _ = m_jnt._build_param_spec()
    res_jnt = compute_metrics(m_jnt, x_true, p_jnt)
    res_jnt["Model"] = "Joint (LFP+Spk)"
    results.append(res_jnt)

    # 4. VERIFY SWITCHING MODEL
    print("[4/4] Verifying Switching Model...")
    m_swi = SwitchingHamiltonianJointModel(1, 2, 1, 10, sampling_freq)
    m_swi.fit_sgd(lfp, spikes, key=jax.random.PRNGKey(5), num_steps=200, verbose=False)
    p_swi, _ = m_swi._build_param_spec()
    res_swi = compute_metrics(m_swi, x_true, p_swi)
    res_swi["Model"] = "Switching (K=2)"
    results.append(res_swi)

    # Summary
    df = pd.DataFrame(results)
    print("\n" + "="*30)
    print("   CONSOLIDATED RESULTS")
    print("="*30)
    print(df.to_string(index=False))
    print("="*30)

if __name__ == "__main__":
    run_verification()
