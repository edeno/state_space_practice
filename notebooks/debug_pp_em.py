"""Diagnostic script to investigate PP EM convergence issues."""
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
from state_space_practice.switching_point_process import (
    switching_point_process_filter,
    SpikeObsParams,
)
from state_space_practice.switching_kalman import switching_kalman_smoother


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


def run_e_step_with_true_params(scenario_name, data):
    """Run E-step with true params, report accuracy."""
    p = data["params"]
    spikes = jnp.array(data["spikes"])
    A = jnp.asarray(p["A"])
    Q = jnp.asarray(p["Q"])
    Z = jnp.array(p["Z"])
    spike_baseline = jnp.asarray(p["spike_baseline"])
    spike_weights = jnp.asarray(p["spike_weights"])
    spike_params = SpikeObsParams(baseline=spike_baseline, weights=spike_weights)

    def log_intensity_func(state, params):
        return params.baseline + params.weights @ state

    n_latent = A.shape[0]
    n_discrete = A.shape[2]
    init_mean = jnp.zeros((n_latent, n_discrete))
    init_cov = jnp.stack([jnp.eye(n_latent)] * n_discrete, axis=2)
    init_prob = jnp.ones(n_discrete) / n_discrete

    filter_out = switching_point_process_filter(
        init_state_cond_mean=init_mean,
        init_state_cond_cov=init_cov,
        init_discrete_state_prob=init_prob,
        spikes=spikes,
        discrete_transition_matrix=Z,
        continuous_transition_matrix=A,
        process_cov=Q,
        dt=p["dt"],
        log_intensity_func=log_intensity_func,
        spike_params=spike_params,
    )
    state_filter_mean, state_filter_cov, filter_probs, last_pair_mean, last_pair_cov, ll = filter_out

    smoother_out = switching_kalman_smoother(
        filter_mean=state_filter_mean,
        filter_cov=state_filter_cov,
        filter_discrete_state_prob=filter_probs,
        last_filter_conditional_cont_mean=last_pair_mean,
        process_cov=Q,
        continuous_transition_matrix=A,
        discrete_state_transition_matrix=Z,
    )
    smoother_prob = smoother_out[2]

    acc = state_accuracy(data["true_states"], np.array(smoother_prob))
    print(f"\n{scenario_name} E-step (true params): accuracy={acc:.3f}, LL={float(ll):.1f}")
    return acc


def run_em_with_verbose(scenario_name, model, spikes, true_states, n_iter=30):
    """Run EM, print LL per iter and final accuracy."""
    print(f"\n{scenario_name} EM (random init):")
    lls = model.fit(jnp.array(spikes), max_iter=n_iter, key=jax.random.PRNGKey(0))
    print(f"  LLs: {[f'{ll:.1f}' for ll in lls]}")
    if hasattr(model, "smoother_discrete_state_prob"):
        acc = state_accuracy(true_states, np.array(model.smoother_discrete_state_prob))
        print(f"  Final accuracy: {acc:.3f}")
    return lls


# ===== COM-PP =====
print("=" * 60)
print("COM-PP SCENARIO")
print("=" * 60)
data_com = simulate_com_pp_scenario()
p = data_com["params"]
run_e_step_with_true_params("COM-PP", data_com)

model_com = CommonOscillatorPointProcessModel(
    n_oscillators=p["n_oscillators"],
    n_neurons=p["n_neurons"],
    n_discrete_states=p["n_discrete_states"],
    sampling_freq=p["sampling_freq"],
    dt=p["dt"],
    freqs=jnp.array(p["freqs"]),
    auto_regressive_coef=jnp.array(p["damping"]),
    process_variance=jnp.array(p["process_variance"]),
)
run_em_with_verbose("COM-PP", model_com, data_com["spikes"], data_com["true_states"])

# ===== CNM-PP =====
print("\n" + "=" * 60)
print("CNM-PP SCENARIO")
print("=" * 60)
data_cnm = simulate_cnm_pp_scenario()
p = data_cnm["params"]
run_e_step_with_true_params("CNM-PP", data_cnm)

model_cnm = CorrelatedNoisePointProcessModel(
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
)
run_em_with_verbose("CNM-PP", model_cnm, data_cnm["spikes"], data_cnm["true_states"])

# ===== DIM-PP =====
print("\n" + "=" * 60)
print("DIM-PP SCENARIO")
print("=" * 60)
data_dim = simulate_dim_pp_scenario()
p = data_dim["params"]
run_e_step_with_true_params("DIM-PP", data_dim)

model_dim = DirectedInfluencePointProcessModel(
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
)
run_em_with_verbose("DIM-PP", model_dim, data_dim["spikes"], data_dim["true_states"])
