"""Check specific numerical stability and M-step correctness issues."""
import jax
jax.config.update("jax_enable_x64", True)
import warnings
warnings.filterwarnings("ignore")

import jax.numpy as jnp
import numpy as np

from state_space_practice.simulate.scenarios import simulate_dim_pp_scenario, simulate_cnm_pp_scenario
from state_space_practice.point_process_models import DirectedInfluencePointProcessModel
from state_space_practice.switching_point_process import SpikeObsParams
from state_space_practice.oscillator_utils import construct_directed_influence_transition_matrix
from state_space_practice.switching_kalman import (
    compute_transition_sufficient_stats,
    compute_transition_q_function,
    optimize_dim_transition_params,
)
from state_space_practice.oscillator_utils import extract_dim_params_from_matrix

# ============================================================================
# Test 7: Check reparameterized M-step - does BFGS go out of bounds?
# ============================================================================
print("=" * 60)
print("TEST 7: REPARAMETERIZED M-STEP BOUNDS CHECK")
print("=" * 60)

data_dim = simulate_dim_pp_scenario()
p = data_dim["params"]
spikes = jnp.array(data_dim["spikes"])
A = jnp.asarray(p["A"])
Q = jnp.asarray(p["Q"])
Z = jnp.array(p["Z"])
spike_baseline = jnp.asarray(p["spike_baseline"])  # shared
spike_weights = jnp.asarray(p["spike_weights"])    # shared

# Build model and run one E-step then check what M-step produces
model = DirectedInfluencePointProcessModel(
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
    use_reparameterized_mstep=True,
)
model._initialize_parameters(jax.random.PRNGKey(0))
model.spike_params = SpikeObsParams(baseline=spike_baseline, weights=spike_weights)

# Run E-step only to get smoother statistics
_ = model._e_step(spikes)

# Extract sufficient stats for M-step
gamma1, beta = compute_transition_sufficient_stats(
    state_cond_smoother_means=model.smoother_state_cond_mean,
    state_cond_smoother_covs=model.smoother_state_cond_cov,
    smoother_joint_discrete_state_prob=model.smoother_joint_discrete_state_prob,
    pair_cond_smoother_cross_cov=model.smoother_pair_cond_cross_cov,
    pair_cond_smoother_means=model.smoother_pair_cond_means,
)

# Extract initial params from true A
init_params_0 = extract_dim_params_from_matrix(
    model.continuous_transition_matrix[:, :, 0], p["sampling_freq"], p["n_oscillators"]
)
init_params_1 = extract_dim_params_from_matrix(
    model.continuous_transition_matrix[:, :, 1], p["sampling_freq"], p["n_oscillators"]
)

print(f"Initial params state 0: damping={np.array(init_params_0['damping'])}, freqs={np.array(init_params_0['freq'])}")
print(f"Initial params state 1: damping={np.array(init_params_1['damping'])}, freqs={np.array(init_params_1['freq'])}")
print(f"Initial coupling 0:\n{np.array(init_params_0['coupling_strength'])}")
print(f"Initial coupling 1:\n{np.array(init_params_1['coupling_strength'])}")

# Run optimization for state 0
try:
    opt_0 = optimize_dim_transition_params(
        gamma1=gamma1[:, :, 0],
        beta=beta[:, :, 0],
        init_params=init_params_0,
        sampling_freq=p["sampling_freq"],
    )
    print(f"\nOptimized state 0: damping={np.array(opt_0['damping'])}, freqs={np.array(opt_0['freq'])}")
    print(f"Optimized coupling 0:\n{np.array(opt_0['coupling_strength'])}")
    A0_opt = construct_directed_influence_transition_matrix(
        freqs=opt_0["freq"],
        damping_coeffs=opt_0["damping"],
        coupling_strengths=opt_0["coupling_strength"],
        phase_diffs=opt_0["phase_diff"],
        sampling_freq=p["sampling_freq"],
    )
    sr_0 = float(jnp.max(jnp.abs(jnp.linalg.eigvals(A0_opt))))
    print(f"Spectral radius state 0: {sr_0:.4f}")
    if sr_0 > 1.0:
        print("⚠️  UNSTABLE A0! Spectral radius > 1.0")
except Exception as e:
    print(f"State 0 optimization failed: {e}")

try:
    opt_1 = optimize_dim_transition_params(
        gamma1=gamma1[:, :, 1],
        beta=beta[:, :, 1],
        init_params=init_params_1,
        sampling_freq=p["sampling_freq"],
    )
    print(f"\nOptimized state 1: damping={np.array(opt_1['damping'])}, freqs={np.array(opt_1['freq'])}")
    print(f"Optimized coupling 1:\n{np.array(opt_1['coupling_strength'])}")
    A1_opt = construct_directed_influence_transition_matrix(
        freqs=opt_1["freq"],
        damping_coeffs=opt_1["damping"],
        coupling_strengths=opt_1["coupling_strength"],
        phase_diffs=opt_1["phase_diff"],
        sampling_freq=p["sampling_freq"],
    )
    sr_1 = float(jnp.max(jnp.abs(jnp.linalg.eigvals(A1_opt))))
    print(f"Spectral radius state 1: {sr_1:.4f}")
    if sr_1 > 1.0:
        print("⚠️  UNSTABLE A1! Spectral radius > 1.0")
except Exception as e:
    print(f"State 1 optimization failed: {e}")


# ============================================================================
# Test 8: Check spike GLM M-step convergence (does fixed alpha=0.5 work?)
# ============================================================================
print("\n" + "=" * 60)
print("TEST 8: SPIKE GLM M-STEP CONVERGENCE CHECK")
print("=" * 60)

from state_space_practice.switching_point_process import (
    _single_neuron_glm_step_second_order,
    _neg_Q_single_neuron,
)

# Simulate a simple neuron with known params
np.random.seed(42)
n_time = 1000
n_latent = 4
dt = 0.01
true_b = jnp.array(1.5)
true_w = jnp.array([0.5, 0.3, -0.2, 0.1])

# Generate latent state and spikes
rng = np.random.default_rng(42)
x = rng.standard_normal((n_time, n_latent)).astype(np.float64) * 0.5
x = jnp.array(x)
log_rates = true_b + x @ true_w
spikes = jnp.array(rng.poisson(jnp.exp(log_rates) * dt))

cov = jnp.stack([jnp.eye(n_latent) * 0.1] * n_time)  # small uncertainty

# Start from wrong init
b_init = jnp.array(0.0)
w_init = jnp.zeros(n_latent)

# Compute initial loss
time_weights = jnp.ones(n_time)
init_loss = _neg_Q_single_neuron(
    jnp.concatenate([jnp.atleast_1d(b_init), w_init]),
    spikes.astype(jnp.float64),
    x, cov, dt, 0.01, time_weights,
)
print(f"Initial Q-function loss: {float(init_loss):.4f}")

# Test iterations
b, w = b_init, w_init
for i in range(10):
    b_new, w_new = _single_neuron_glm_step_second_order(
        b, w, spikes.astype(jnp.float64), x, cov, dt,
        time_weights, weight_l2=0.01,
    )
    new_loss = _neg_Q_single_neuron(
        jnp.concatenate([jnp.atleast_1d(b_new), w_new]),
        spikes.astype(jnp.float64),
        x, cov, dt, 0.01, time_weights,
    )
    direction = "↓" if float(new_loss) < float(init_loss if i == 0 else prev_loss) else "↑⚠️"
    print(f"  Iter {i+1}: loss={float(new_loss):.4f} {direction}, b={float(b_new):.3f}, w_max={float(jnp.max(jnp.abs(w_new))):.3f}")
    if i > 0:
        prev_loss = new_loss
    else:
        prev_loss = init_loss
    b, w = b_new, w_new

print(f"\nTrue b={float(true_b):.3f}, True w={np.array(true_w)}")
print(f"Final b={float(b):.3f}, Final w={np.array(w)}")


# ============================================================================
# Test 9: Check Laplace normalization consistency
# ============================================================================
print("\n" + "=" * 60)
print("TEST 9: LAPLACE LL GRADIENT — IS IT ACTUALLY THE CORRECT SURROGATE?")
print("=" * 60)

from state_space_practice.switching_point_process import switching_point_process_filter

data_cnm = simulate_cnm_pp_scenario()
p_cnm = data_cnm["params"]
spikes_cnm = jnp.array(data_cnm["spikes"])
A_cnm = jnp.asarray(p_cnm["A"])
Q_cnm = jnp.asarray(p_cnm["Q"])
Z_cnm = jnp.array(p_cnm["Z"])
spike_params_cnm = SpikeObsParams(
    baseline=jnp.asarray(p_cnm["spike_baseline"]),
    weights=jnp.asarray(p_cnm["spike_weights"]),
)

def log_intensity_func(state, params):
    return params.baseline + params.weights @ state

n_latent = A_cnm.shape[0]
n_discrete = A_cnm.shape[2]

# Test LL sensitivity to Q perturbation
def compute_ll(Q_perturb):
    Q_test = Q_cnm + Q_perturb
    init_mean = jnp.zeros((n_latent, n_discrete))
    init_cov = jnp.stack([jnp.eye(n_latent)] * n_discrete, axis=2)
    init_prob = jnp.ones(n_discrete) / n_discrete
    
    out = switching_point_process_filter(
        init_state_cond_mean=init_mean,
        init_state_cond_cov=init_cov,
        init_discrete_state_prob=init_prob,
        spikes=spikes_cnm[:500],  # Use subset for speed
        discrete_transition_matrix=Z_cnm,
        continuous_transition_matrix=A_cnm,
        process_cov=Q_test,
        dt=p_cnm["dt"],
        log_intensity_func=log_intensity_func,
        spike_params=spike_params_cnm,
    )
    return out[-1]  # marginal_log_likelihood

# Check LL at true Q
ll_true = compute_ll(jnp.zeros_like(Q_cnm))
print(f"LL at true Q: {float(ll_true):.2f}")

# Check LL at Q0 increased (state 0 variance), Q1 decreased
delta_Q = jnp.zeros_like(Q_cnm)
# Increase Q0 slightly, decrease Q1 slightly
perturb = jnp.array([[[0.01, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 0.01, 0], [0, 0, 0, 0.01]]])
# Wrong direction: Q0 up (true is low), Q1 down (true is high)
wrong_perturb = jnp.zeros_like(Q_cnm)
wrong_perturb = wrong_perturb.at[:, :, 0].set(jnp.eye(n_latent) * 0.1)  # Q0 += 0.1*I
wrong_perturb = wrong_perturb.at[:, :, 1].set(-jnp.eye(n_latent) * 0.05)  # Q1 -= 0.05*I
ll_wrong = compute_ll(wrong_perturb)
print(f"LL at Q perturbed wrong direction: {float(ll_wrong):.2f}")

# Correct direction: Q0 down from truth, Q1 up from truth
right_perturb = jnp.zeros_like(Q_cnm)
right_perturb = right_perturb.at[:, :, 0].set(-jnp.eye(n_latent) * 0.01)  # Q0 -= 0.01
right_perturb = right_perturb.at[:, :, 1].set(jnp.eye(n_latent) * 0.05)  # Q1 += 0.05
try:
    ll_right = compute_ll(right_perturb)
    print(f"LL at Q perturbed toward more contrast: {float(ll_right):.2f}")
    print(f"More contrast improves LL: {float(ll_right) > float(ll_true)}")
except Exception as e:
    print(f"Right perturb failed: {e}")
