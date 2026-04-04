"""Deep diagnostic: investigate non-monotone EM and degenerate Q issues."""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from itertools import permutations

from state_space_practice.simulate.scenarios import (
    simulate_com_pp_scenario,
    simulate_cnm_pp_scenario,
    simulate_dim_pp_scenario,
)
from state_space_practice.point_process_models import (
    CommonOscillatorPointProcessModel,
    CorrelatedNoisePointProcessModel,
    DirectedInfluencePointProcessModel,
)
from state_space_practice.switching_point_process import SpikeObsParams
from state_space_practice.oscillator_utils import (
    construct_directed_influence_transition_matrix,
    project_coupled_transition_matrix,
)


def state_accuracy(true_states, probs):
    inferred = np.array(jnp.argmax(jnp.array(probs), axis=1))
    true = np.array(true_states)
    n_states = probs.shape[1]
    best = 0.0
    for perm in permutations(range(n_states)):
        remapped = np.array([perm[s] for s in inferred])
        acc = float(np.mean(remapped == true))
        best = max(best, acc)
    return best


# ============================================================================
# Test 1: Is DIM non-monotone EM caused by the projection step?
# ============================================================================
print("=" * 60)
print("TEST 1: DIM-PP NON-MONOTONE EM — PROJECTION INVESTIGATION")
print("=" * 60)

data_dim = simulate_dim_pp_scenario()
p = data_dim["params"]

# Run DIM-PP without reparameterized mstep (uses projection)
model_dim_proj = DirectedInfluencePointProcessModel(
    n_oscillators=p["n_oscillators"],
    n_neurons=p["n_neurons"],
    n_discrete_states=p["n_discrete_states"],
    sampling_freq=p["sampling_freq"],
    dt=p["dt"],
    freqs=jnp.array(p["freqs"]),
    auto_regressive_coef=jnp.array(p["damping"]),
    process_variance=jnp.array(p["process_variance"]),
    phase_difference=jnp.array(p["phase_difference"]),
    coupling_strength=jnp.array(p["coupling_strength"]),
    use_reparameterized_mstep=False,  # standard projection
)
lls_proj = model_dim_proj.fit(jnp.array(data_dim["spikes"]), max_iter=15, key=jax.random.PRNGKey(0))
print(f"Projection M-step LLs: {[f'{ll:.1f}' for ll in lls_proj]}")
print(f"Monotone? {all(lls_proj[i+1] >= lls_proj[i] for i in range(len(lls_proj)-1))}")
n_decreases = sum(1 for i in range(len(lls_proj)-1) if lls_proj[i+1] < lls_proj[i])
print(f"Number of LL decreases: {n_decreases}")

# Run DIM-PP with reparameterized mstep (known to produce NaN — investigating)
try:
    model_dim_reparam = DirectedInfluencePointProcessModel(
        n_oscillators=p["n_oscillators"],
        n_neurons=p["n_neurons"],
        n_discrete_states=p["n_discrete_states"],
        sampling_freq=p["sampling_freq"],
        dt=p["dt"],
        freqs=jnp.array(p["freqs"]),
        auto_regressive_coef=jnp.array(p["damping"]),
        process_variance=jnp.array(p["process_variance"]),
        phase_difference=jnp.array(p["phase_difference"]),
        coupling_strength=jnp.array(p["coupling_strength"]),
        use_reparameterized_mstep=True,  # reparameterized
    )
    lls_reparam = model_dim_reparam.fit(jnp.array(data_dim["spikes"]), max_iter=15, key=jax.random.PRNGKey(0))
    print(f"\nReparam M-step LLs: {[f'{ll:.1f}' for ll in lls_reparam]}")
    print(f"Monotone? {all(lls_reparam[i+1] >= lls_reparam[i] for i in range(len(lls_reparam)-1))}")
    n_decreases_r = sum(1 for i in range(len(lls_reparam)-1) if lls_reparam[i+1] < lls_reparam[i])
    print(f"Number of LL decreases: {n_decreases_r}")
except ValueError as e:
    print(f"\nReparam M-step FAILED: {e}")
    print("→ BUG: Reparameterized M-step produces NaN at iteration 1")


# ============================================================================
# Test 2: What does the projection do to the M-step A?
# ============================================================================
print("\n" + "=" * 60)
print("TEST 2: WHAT DOES PROJECT_COUPLED_TRANSITION_MATRIX DO?")
print("=" * 60)

# Quantify the distortion from projection
true_A0 = jnp.array(p["A"])[:, :, 0]
true_A1 = jnp.array(p["A"])[:, :, 1]
proj_A0 = project_coupled_transition_matrix(true_A0)
proj_A1 = project_coupled_transition_matrix(true_A1)
print(f"True A0 vs projected A0 max diff: {float(jnp.max(jnp.abs(true_A0 - proj_A0))):.6f}")
print(f"True A1 vs projected A1 max diff: {float(jnp.max(jnp.abs(true_A1 - proj_A1))):.6f}")

# Check spectral radii
print(f"True A0 spectral radius: {float(jnp.max(jnp.abs(jnp.linalg.eigvals(true_A0)))):.4f}")
print(f"True A1 spectral radius: {float(jnp.max(jnp.abs(jnp.linalg.eigvals(true_A1)))):.4f}")
print(f"Proj A0 spectral radius: {float(jnp.max(jnp.abs(jnp.linalg.eigvals(proj_A0)))):.4f}")
print(f"Proj A1 spectral radius: {float(jnp.max(jnp.abs(jnp.linalg.eigvals(proj_A1)))):.4f}")


# ============================================================================
# Test 3: CNM-PP collapse — does Q degenerate?
# ============================================================================
print("\n" + "=" * 60)
print("TEST 3: CNM-PP Q DEGENERATE COLLAPSE")
print("=" * 60)

data_cnm = simulate_cnm_pp_scenario()
p_cnm = data_cnm["params"]

model_cnm = CorrelatedNoisePointProcessModel(
    n_oscillators=p_cnm["n_oscillators"],
    n_neurons=p_cnm["n_neurons"],
    n_discrete_states=p_cnm["n_discrete_states"],
    sampling_freq=p_cnm["sampling_freq"],
    dt=p_cnm["dt"],
    freqs=jnp.array(p_cnm["freqs"]),
    auto_regressive_coef=jnp.array(p_cnm["damping"]),
    process_variance=jnp.array(p_cnm["process_variance"]),
    phase_difference=jnp.array(p_cnm["phase_difference"]),
    coupling_strength=jnp.array(p_cnm["coupling_strength"]),
)
lls_cnm = model_cnm.fit(jnp.array(data_cnm["spikes"]), max_iter=20, key=jax.random.PRNGKey(0))
print(f"LLs: {[f'{ll:.1f}' for ll in lls_cnm]}")
acc = state_accuracy(data_cnm["true_states"], np.array(model_cnm.smoother_discrete_state_prob))
print(f"Final accuracy: {acc:.3f}")

# Check Q eigenvalues after convergence
print("\nFinal Q state 0 eigenvalues:", np.round(np.linalg.eigvalsh(np.array(model_cnm.process_cov[:, :, 0])), 4))
print("Final Q state 1 eigenvalues:", np.round(np.linalg.eigvalsh(np.array(model_cnm.process_cov[:, :, 1])), 4))
print("True Q state 0 eigenvalues:", np.round(np.linalg.eigvalsh(np.array(p_cnm["Q"])[:, :, 0]), 4))
print("True Q state 1 eigenvalues:", np.round(np.linalg.eigvalsh(np.array(p_cnm["Q"])[:, :, 1]), 4))

# Check discrete state probabilities — are they collapsed?
disc_probs = np.array(model_cnm.smoother_discrete_state_prob)
print(f"\nMean discrete state prob [s0,s1]: {disc_probs.mean(axis=0)}")
print(f"Min/max of state 0 prob: {disc_probs[:,0].min():.3f}, {disc_probs[:,0].max():.3f}")


# ============================================================================
# Test 4: COM-PP — does it converge to good params when starting from true spike params?
# ============================================================================
print("\n" + "=" * 60)
print("TEST 4: COM-PP WITH TRUE SPIKE PARAMS (only EM dynamics)")
print("=" * 60)

data_com = simulate_com_pp_scenario()
p_com = data_com["params"]

model_com_true_spikes = CommonOscillatorPointProcessModel(
    n_oscillators=p_com["n_oscillators"],
    n_neurons=p_com["n_neurons"],
    n_discrete_states=p_com["n_discrete_states"],
    sampling_freq=p_com["sampling_freq"],
    dt=p_com["dt"],
    freqs=jnp.array(p_com["freqs"]),
    auto_regressive_coef=jnp.array(p_com["damping"]),
    process_variance=jnp.array(p_com["process_variance"]),
    update_spike_params=False,  # Fix spike params!
)
# Initialize with true spike params
model_com_true_spikes._initialize_parameters(jax.random.PRNGKey(0))
model_com_true_spikes.spike_params = SpikeObsParams(
    baseline=jnp.array(p_com["spike_baseline"]),
    weights=jnp.array(p_com["spike_weights"]),
)
lls_true_spikes = model_com_true_spikes.fit(
    jnp.array(data_com["spikes"]), max_iter=10, key=jax.random.PRNGKey(0),
    skip_init=True
)
print(f"LLs (true spikes, random Z init): {[f'{ll:.1f}' for ll in lls_true_spikes]}")
acc = state_accuracy(data_com["true_states"], np.array(model_com_true_spikes.smoother_discrete_state_prob))
print(f"Final accuracy: {acc:.3f}")


# ============================================================================
# Test 5: CNM-PP with true spike params
# ============================================================================
print("\n" + "=" * 60)
print("TEST 5: CNM-PP WITH TRUE SPIKE PARAMS")
print("=" * 60)

model_cnm_true_spikes = CorrelatedNoisePointProcessModel(
    n_oscillators=p_cnm["n_oscillators"],
    n_neurons=p_cnm["n_neurons"],
    n_discrete_states=p_cnm["n_discrete_states"],
    sampling_freq=p_cnm["sampling_freq"],
    dt=p_cnm["dt"],
    freqs=jnp.array(p_cnm["freqs"]),
    auto_regressive_coef=jnp.array(p_cnm["damping"]),
    process_variance=jnp.array(p_cnm["process_variance"]),
    phase_difference=jnp.array(p_cnm["phase_difference"]),
    coupling_strength=jnp.array(p_cnm["coupling_strength"]),
    update_spike_params=False,  # Fix spike params!
)
model_cnm_true_spikes._initialize_parameters(jax.random.PRNGKey(0))
model_cnm_true_spikes.spike_params = SpikeObsParams(
    baseline=jnp.array(p_cnm["spike_baseline"]),
    weights=jnp.array(p_cnm["spike_weights"]),
)
lls_cnm_true = model_cnm_true_spikes.fit(
    jnp.array(data_cnm["spikes"]), max_iter=20, key=jax.random.PRNGKey(0),
    skip_init=True
)
print(f"LLs (true spikes): {[f'{ll:.1f}' for ll in lls_cnm_true]}")
acc = state_accuracy(data_cnm["true_states"], np.array(model_cnm_true_spikes.smoother_discrete_state_prob))
print(f"Final accuracy: {acc:.3f}")
Q0_final = model_cnm_true_spikes.process_cov[:, :, 0]
Q1_final = model_cnm_true_spikes.process_cov[:, :, 1]
print("Final Q0 eigenvalues:", np.round(np.linalg.eigvalsh(np.array(Q0_final)), 4))
print("Final Q1 eigenvalues:", np.round(np.linalg.eigvalsh(np.array(Q1_final)), 4))


# ============================================================================
# Test 6: DIM-PP with true spike params
# ============================================================================
print("\n" + "=" * 60)
print("TEST 6: DIM-PP WITH TRUE SPIKE PARAMS")
print("=" * 60)

model_dim_true_spikes = DirectedInfluencePointProcessModel(
    n_oscillators=p["n_oscillators"],
    n_neurons=p["n_neurons"],
    n_discrete_states=p["n_discrete_states"],
    sampling_freq=p["sampling_freq"],
    dt=p["dt"],
    freqs=jnp.array(p["freqs"]),
    auto_regressive_coef=jnp.array(p["damping"]),
    process_variance=jnp.array(p["process_variance"]),
    phase_difference=jnp.array(p["phase_difference"]),
    coupling_strength=jnp.array(p["coupling_strength"]),
    update_spike_params=False,  # Fix spike params!
)
model_dim_true_spikes._initialize_parameters(jax.random.PRNGKey(0))
model_dim_true_spikes.spike_params = SpikeObsParams(
    baseline=jnp.array(p["spike_baseline"]),
    weights=jnp.array(p["spike_weights"]),
)
lls_dim_true = model_dim_true_spikes.fit(
    jnp.array(data_dim["spikes"]), max_iter=20, key=jax.random.PRNGKey(0),
    skip_init=True
)
print(f"LLs (true spikes, standard projection): {[f'{ll:.1f}' for ll in lls_dim_true]}")
print(f"Monotone? {all(lls_dim_true[i+1] >= lls_dim_true[i] for i in range(len(lls_dim_true)-1))}")
acc = state_accuracy(data_dim["true_states"], np.array(model_dim_true_spikes.smoother_discrete_state_prob))
print(f"Final accuracy: {acc:.3f}")
