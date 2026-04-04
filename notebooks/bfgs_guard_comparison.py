"""Compare BFGS with and without the loss-improvement guard on CA1 data."""
# %%
import time
import jax
import jax.numpy as jnp
import numpy as np
import pickle

jax.config.update("jax_enable_x64", True)

from sklearn.metrics import roc_auc_score

from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
)
from state_space_practice.switching_point_process import (
    QRegularizationConfig,
    SpikeObsParams,
    SwitchingSpikeOscillatorModel,
)
import state_space_practice.switching_point_process as spp
import optimistix as optx

# %%
with open("data/ca1_preprocessed_250Hz.pkl", "rb") as f:
    data = pickle.load(f)

spikes = jnp.array(data["binned_spikes"])
labels = data["behavioral_labels"]
sf = data["sampling_freq"]
dt = data["dt"]
nn = data["n_neurons"]
theta_freq = jnp.array([8.0])

valid = labels[labels != 2]
n_im = np.sum(valid[:-1] == 0)
n_run = np.sum(valid[:-1] == 1)
p_im_stay = (np.sum((valid[:-1] == 0) & (valid[1:] == 0)) + 1) / (n_im + 2)
p_run_stay = (np.sum((valid[:-1] == 1) & (valid[1:] == 1)) + 1) / (n_run + 2)
running_frac = float((labels == 1).mean())
Z_emp = jnp.array([[p_im_stay, 1 - p_im_stay], [1 - p_run_stay, p_run_stay]])
init_prob = jnp.array([1 - running_frac, running_frac])
mean_counts = np.array(jnp.mean(spikes, axis=0))
empirical_baseline = np.log(mean_counts / dt + 1e-10)

print(f"Data: {data['n_time']} steps, {nn} neurons")

# %%
# Save original function
_original_glm_step = spp._single_neuron_glm_step_second_order


def _no_guard_glm_step(baseline, weights, y_n, smoother_mean, smoother_cov,
                        dt, time_weights=None, weight_l2=0.0,
                        baseline_prior=None, baseline_prior_l2=0.0,
                        max_steps=50):
    """BFGS GLM step without loss-improvement guard — accept if finite."""
    params = jnp.concatenate([jnp.atleast_1d(baseline), weights])

    def loss_fn(p, args):
        return spp._neg_Q_single_neuron(
            p, y_n, smoother_mean, smoother_cov, dt, weight_l2,
            time_weights, baseline_prior, baseline_prior_l2,
        )

    solver = optx.BFGS(rtol=1e-5, atol=1e-5)
    result = optx.minimise(
        loss_fn, solver, params, args=None, max_steps=max_steps, throw=False,
    )
    # Accept if finite only — no loss comparison
    new_loss = loss_fn(result.value, None)
    final_params = jnp.where(jnp.isfinite(new_loss), result.value, params)
    return final_params[0], final_params[1:]


def make_model(damping_off, damping_on):
    A_off = construct_common_oscillator_transition_matrix(theta_freq, jnp.array([damping_off]), sf)
    A_on = construct_common_oscillator_transition_matrix(theta_freq, jnp.array([damping_on]), sf)
    Q_off = construct_common_oscillator_process_covariance(jnp.array([0.01]))
    Q_on = construct_common_oscillator_process_covariance(jnp.array([0.02]))

    model = SwitchingSpikeOscillatorModel(
        n_oscillators=1, n_neurons=nn, n_discrete_states=2,
        sampling_freq=sf, dt=dt,
        q_regularization=QRegularizationConfig(),
        separate_spike_params=True, spike_weight_l2=0.05,
        update_continuous_transition_matrix=False, update_process_cov=False,
        update_init_mean=False, update_init_cov=False,
        update_discrete_transition_matrix=False,
    )
    key = jax.random.PRNGKey(42)
    model._initialize_parameters(key)
    model.continuous_transition_matrix = jnp.stack([A_off, A_on], axis=-1)
    model.process_cov = jnp.stack([Q_off, Q_on], axis=-1)
    model.init_cov = jnp.stack([jnp.eye(2) * 0.5, jnp.eye(2) * 0.5], axis=-1)
    model.discrete_transition_matrix = Z_emp
    model.init_discrete_state_prob = init_prob
    model.spike_params = SpikeObsParams(
        baseline=jnp.stack([jnp.array(empirical_baseline), jnp.array(empirical_baseline)], axis=-1),
        weights=jax.random.normal(key, (nn, 2, 2)) * 0.01,
    )
    return model


def run_em(damping_off, damping_on, use_guard, max_iter=10, label=""):
    if not use_guard:
        spp._single_neuron_glm_step_second_order = _no_guard_glm_step
    else:
        spp._single_neuron_glm_step_second_order = _original_glm_step

    try:
        model = make_model(damping_off, damping_on)
        t0 = time.time()
        lls = model.fit(spikes, max_iter=max_iter, tol=1e-10, skip_init=True)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  FAILED: {e}")
        spp._single_neuron_glm_step_second_order = _original_glm_step
        return None, None, None
    finally:
        spp._single_neuron_glm_step_second_order = _original_glm_step

    prob = np.array(model.smoother_discrete_state_prob)
    running_mask = (labels == 1).astype(float)
    clear_mask = (labels == 0) | (labels == 1)
    corr = [np.corrcoef(running_mask, prob[:, j])[0, 1] for j in range(2)]
    theta_on = np.argmax(corr)
    auc = roc_auc_score(running_mask[clear_mask], prob[clear_mask, theta_on])

    print(f"\n{label}:")
    print(f"  AUC: {auc:.3f}, LL: {lls[0]:.0f} -> {lls[-1]:.0f}, Time: {elapsed:.1f}s")
    for i, ll in enumerate(lls):
        print(f"    {i}: {ll:.0f}")
    return auc, lls, elapsed


# %%
print("=" * 70)
print("CONSERVATIVE DAMPING (0.95/0.99), 10 EM iterations")
print("=" * 70)
auc_cg, lls_cg, t_cg = run_em(0.95, 0.99, use_guard=True, label="WITH guard")
auc_cn, lls_cn, t_cn = run_em(0.95, 0.99, use_guard=False, label="WITHOUT guard")

print("\n" + "=" * 70)
print("STRONG DAMPING (0.80/0.999), 10 EM iterations")
print("=" * 70)
auc_sg, lls_sg, t_sg = run_em(0.80, 0.999, use_guard=True, label="WITH guard")
auc_sn, lls_sn, t_sn = run_em(0.80, 0.999, use_guard=False, label="WITHOUT guard")

# %%
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Config':<45} {'AUC':>6} {'LL_final':>12} {'Time':>7}")
print("-" * 75)
for label, auc, lls, t in [
    ("conservative WITH guard", auc_cg, lls_cg, t_cg),
    ("conservative WITHOUT guard", auc_cn, lls_cn, t_cn),
    ("strong WITH guard", auc_sg, lls_sg, t_sg),
    ("strong WITHOUT guard", auc_sn, lls_sn, t_sn),
]:
    if auc is not None:
        print(f"{label:<45} {auc:>6.3f} {lls[-1]:>12.0f} {t:>6.1f}s")
    else:
        print(f"{label:<45} {'FAILED':>6}")
