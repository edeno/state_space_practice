"""Example: Switching Poisson neuron with state-dependent dynamics.

This script demonstrates inference with true parameters (no EM) for a switching
point-process model where a neuron's firing rate is modulated by a continuous
latent state that has different dynamics in each discrete regime:

- State 0: Very stable dynamics (high A, low Q) - latent stays near 0
- State 1: Noisy dynamics (low A, high Q) - latent fluctuates widely

The firing rate model is:
    log(lambda) = baseline + weight * x

How spikes inform discrete state inference:
-------------------------------------------
With the SAME observation model for both states, spikes provide INDIRECT evidence
about the discrete state through the latent:

1. The filter maintains separate predictions for each state's latent distribution
2. A spike updates the latent estimate toward values consistent with high firing
3. The state probability shifts toward whichever state's prediction was more
   consistent with the spike-implied latent value

State 0 (stable): Tight latent prediction → spikes that deviate from prediction
                  are "surprising" and shift probability toward state 1

State 1 (noisy): Broad latent prediction → can explain a wider range of spikes,
                 so extreme spikes are less surprising

This is different from having state-dependent baselines, where each spike would
directly provide evidence for one state based on the instantaneous rate.

Numerical Stability Notes
-------------------------
This example includes several techniques for numerical stability:
1. Enable JAX float64 for better precision
2. Clip log-intensities to prevent overflow in exp()
3. Use log-sum-exp tricks for probability computations
"""

import jax

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from state_space_practice.switching_point_process import (
    SpikeObsParams,
    switching_point_process_filter,
)

# Constants for numerical stability
LOG_RATE_MIN = -10.0  # Minimum log-rate (corresponds to ~0.00005 Hz)
LOG_RATE_MAX = 8.0  # Maximum log-rate (corresponds to ~3000 Hz)
PROB_MIN = 1e-10  # Minimum probability to avoid log(0)
# Very different dynamics - strong contrast for clear state separation
STATE0_A = 0.99
STATE0_Q = 0.001
STATE1_A = 0.5
STATE1_Q = 1.0
DEFAULT_BASELINE = 2.0
DEFAULT_WEIGHT = 1.0


def safe_log_intensity(x: jax.Array, params: SpikeObsParams) -> jax.Array:
    """Numerically stable log-intensity function.

    Clips the output to prevent overflow when computing exp(log_intensity).

    Parameters
    ----------
    x : Array, shape (n_latent,)
        Current latent state.
    params : SpikeObsParams
        Spike observation parameters (baseline and weights).

    Returns
    -------
    Array, shape (n_neurons,)
        Log firing rate for each neuron.
    """
    log_rate = params.baseline + params.weights @ x
    # Clip to prevent numerical overflow/underflow
    return jnp.clip(log_rate, LOG_RATE_MIN, LOG_RATE_MAX)


def _standardize_spike_params(
    baseline: jax.Array, weights: jax.Array, n_neurons: int
) -> tuple[jax.Array, jax.Array]:
    baseline = jnp.asarray(baseline)
    if baseline.ndim == 0:
        baseline = jnp.full((n_neurons,), baseline)
    if baseline.shape != (n_neurons,):
        raise ValueError(
            f"baseline must have shape ({n_neurons},), got {baseline.shape}"
        )

    weights = jnp.asarray(weights)
    if weights.ndim == 0:
        weights = jnp.full((n_neurons, 1), weights)
    elif weights.ndim == 1:
        weights = weights[:, None]
    if weights.shape != (n_neurons, 1):
        raise ValueError(
            f"weights must have shape ({n_neurons}, 1), got {weights.shape}"
        )

    return baseline, weights


def simulate_switching_dynamics_neuron(
    n_time: int,
    dt: float,
    transition_prob: float,
    key: jax.Array,
    n_neurons: int = 1,
    baseline: jax.Array | None = None,
    weights: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Simulate a neuron with state-dependent latent dynamics.

    The two states have different VARIANCES in their latent dynamics:
    - State 0: Low variance (stable), latent stays near 0 → moderate rate (~10 Hz)
    - State 1: High variance (noisy), latent fluctuates widely → variable rate

    When the latent goes high (due to state 1's variance), spike rate increases.
    When the latent goes low, spike rate decreases.
    This makes high/low spike rates indirect evidence for the discrete state.

    Parameters
    ----------
    n_time : int
        Number of time bins.
    dt : float
        Time bin width in seconds.
    transition_prob : float
        Probability of switching states per time step.
    key : jax.Array
        JAX random key.

    Returns
    -------
    spikes : jax.Array, shape (n_time, n_neurons)
        Spike counts.
    true_states : jax.Array, shape (n_time,)
        True discrete state sequence (0 or 1).
    true_latent : jax.Array, shape (n_time,)
        True continuous latent state.
    baseline : jax.Array, shape (n_neurons,)
        Baseline log-rate for each neuron.
    weights : jax.Array, shape (n_neurons, 1)
        Weights for each neuron.
    """
    # Discrete transition matrix
    stay_prob = 1.0 - transition_prob
    Z = jnp.array([[stay_prob, transition_prob], [transition_prob, stay_prob]])

    # Dynamics: x_t = A * x_{t-1} + sqrt(Q) * noise
    # State 0: Very stable (high A, low Q) - latent stays near 0
    # State 1: Very noisy (low A, high Q) - latent fluctuates widely
    A_0, Q_0 = STATE0_A, STATE0_Q  # Very stable, stationary var ≈ 0.5
    A_1, Q_1 = STATE1_A, STATE1_Q  # Noisy, stationary var ≈ 1.4

    # Observation model: log(rate) = baseline + weight * x
    # Defaults are chosen to make state differences visually obvious.
    if baseline is None:
        baseline = DEFAULT_BASELINE
    if weights is None:
        weights = DEFAULT_WEIGHT
    baseline, weights = _standardize_spike_params(baseline, weights, n_neurons)

    def step(carry, _):
        x_prev, s_prev, key = carry
        key, k1, k2, k3 = jax.random.split(key, 4)

        # Sample next discrete state
        s_t = jax.random.categorical(k1, jnp.log(Z[s_prev]))

        # Dynamics depend on discrete state
        A = jnp.where(s_t == 0, A_0, A_1)
        Q = jnp.where(s_t == 0, Q_0, Q_1)

        # AR(1) dynamics
        x_t = A * x_prev + jnp.sqrt(Q) * jax.random.normal(k2)

        # Compute firing rate
        log_rate = baseline + weights[:, 0] * x_t
        rate = jnp.exp(log_rate) * dt

        # Sample spikes
        spike = jax.random.poisson(k3, rate).astype(jnp.float64)

        return (x_t, s_t, key), (spike, s_t, x_t)

    # Initialize near 0
    key, subkey = jax.random.split(key)
    x_0 = jax.random.normal(subkey) * 0.1
    key, subkey = jax.random.split(key)
    s_0 = jax.random.categorical(subkey, jnp.log(jnp.array([0.5, 0.5])))

    _, (spikes, true_states, true_latent) = jax.lax.scan(
        step, (x_0, s_0, key), None, length=n_time
    )

    return spikes, true_states, true_latent, baseline, weights


class SwitchingDynamicsModel:
    """Switching dynamics model with point-process observations.

    The model has different continuous dynamics in each discrete state,
    with firing rate depending on the continuous latent state.

    Parameters
    ----------
    n_discrete_states : int
        Number of discrete states.
    dt : float
        Time bin width in seconds.
    """

    def __init__(
        self,
        n_discrete_states: int = 2,
        dt: float = 0.02,
        max_newton_iter: int = 1,
        line_search_beta: float = 0.5,
    ):
        self.n_discrete_states = n_discrete_states
        self.dt = dt
        self.n_latent = 1
        self.max_newton_iter = max_newton_iter
        self.line_search_beta = line_search_beta

        # Parameters
        self.discrete_transition_matrix: jax.Array | None = None
        self.init_discrete_prob: jax.Array | None = None
        self.init_mean: jax.Array | None = None
        self.init_cov: jax.Array | None = None
        self.continuous_transition_matrix: jax.Array | None = None
        self.process_cov: jax.Array | None = None
        self.spike_baseline: jax.Array | None = None
        self.spike_weight: jax.Array | None = None

    def _make_spike_params(self) -> SpikeObsParams:
        """Create spike observation parameters."""
        return SpikeObsParams(baseline=self.spike_baseline, weights=self.spike_weight)

    def infer_states(self, spikes: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Run inference.

        Note: Currently returns filter (not smoother) results because the
        smoother has numerical issues with point-process observations that
        need to be investigated.
        """
        spikes = jnp.asarray(spikes)
        if spikes.ndim == 1:
            spikes = spikes[:, None]
        spike_params = self._make_spike_params()

        filter_results = switching_point_process_filter(
            init_state_cond_mean=self.init_mean,
            init_state_cond_cov=self.init_cov,
            init_discrete_state_prob=self.init_discrete_prob,
            spikes=spikes,
            discrete_transition_matrix=self.discrete_transition_matrix,
            continuous_transition_matrix=self.continuous_transition_matrix,
            process_cov=self.process_cov,
            dt=self.dt,
            log_intensity_func=safe_log_intensity,
            spike_params=spike_params,
            max_newton_iter=self.max_newton_iter,
            line_search_beta=self.line_search_beta,
        )

        filter_mean, filter_cov, filter_discrete_prob, _, _ = filter_results

        # Compute marginal filter mean across discrete states
        # filter_mean: (n_time, n_latent, n_discrete_states)
        # filter_discrete_prob: (n_time, n_discrete_states)
        marginal_filter_mean = jnp.einsum(
            "tls,ts->tl", filter_mean, filter_discrete_prob
        )

        return filter_discrete_prob, marginal_filter_mean


def _stationary_variance(a: float, q: float) -> jax.Array:
    return q / jnp.maximum(1.0 - a**2, 1e-6)


def _infer_n_neurons_from_params(baseline: jax.Array, weights: jax.Array) -> int:
    weights = jnp.asarray(weights)
    if weights.ndim == 0:
        return 1
    if weights.ndim == 1:
        return int(weights.shape[0])
    return int(weights.shape[0])


def build_true_param_model(
    dt: float,
    transition_prob: float,
    baseline: jax.Array,
    weights: jax.Array,
    max_newton_iter: int = 1,
    line_search_beta: float = 0.5,
) -> SwitchingDynamicsModel:
    model = SwitchingDynamicsModel(
        n_discrete_states=2,
        dt=dt,
        max_newton_iter=max_newton_iter,
        line_search_beta=line_search_beta,
    )

    stay_prob = 1.0 - transition_prob
    model.discrete_transition_matrix = jnp.array(
        [[stay_prob, transition_prob], [transition_prob, stay_prob]]
    )
    model.init_discrete_prob = jnp.array([0.5, 0.5])

    model.init_mean = jnp.zeros((model.n_latent, model.n_discrete_states))
    var0 = _stationary_variance(STATE0_A, STATE0_Q)
    var1 = _stationary_variance(STATE1_A, STATE1_Q)
    model.init_cov = jnp.stack(
        [
            jnp.eye(model.n_latent) * var0,
            jnp.eye(model.n_latent) * var1,
        ],
        axis=-1,
    )

    model.continuous_transition_matrix = jnp.stack(
        [
            jnp.eye(model.n_latent) * STATE0_A,
            jnp.eye(model.n_latent) * STATE1_A,
        ],
        axis=-1,
    )
    model.process_cov = jnp.stack(
        [
            jnp.eye(model.n_latent) * STATE0_Q,
            jnp.eye(model.n_latent) * STATE1_Q,
        ],
        axis=-1,
    )

    n_neurons = _infer_n_neurons_from_params(baseline, weights)
    baseline, weights = _standardize_spike_params(baseline, weights, n_neurons)
    model.spike_baseline = baseline
    model.spike_weight = weights

    return model


def summarize_inference(
    true_states: jax.Array,
    true_latent: jax.Array,
    smoother_discrete_prob: jax.Array,
    marginal_smoother_mean: jax.Array,
) -> tuple[float, float, float, jax.Array, jax.Array, bool]:
    inferred_states = jnp.argmax(smoother_discrete_prob, axis=1)
    accuracy_direct = jnp.mean(inferred_states == true_states)
    accuracy_flipped = jnp.mean(inferred_states == (1 - true_states))
    flipped = accuracy_flipped > accuracy_direct

    if flipped:
        inferred_states = 1 - inferred_states
        smoother_discrete_prob = smoother_discrete_prob[:, ::-1]

    accuracy = float(jnp.maximum(accuracy_direct, accuracy_flipped))
    latent_corr = float(jnp.corrcoef(true_latent, marginal_smoother_mean[:, 0])[0, 1])
    posterior_entropy = -jnp.mean(
        jnp.sum(smoother_discrete_prob * jnp.log(smoother_discrete_prob + PROB_MIN), axis=1)
    )

    return (
        accuracy,
        latent_corr,
        float(posterior_entropy),
        smoother_discrete_prob,
        inferred_states,
        flipped,
    )


def main():
    # Simulation parameters
    n_time = 50000  # 1000 seconds at 50 Hz (~17 minutes)
    dt = 0.02
    transition_prob = 0.0005  # Switch every ~2000 time steps on average

    print("=" * 60)
    print("Switching Dynamics Example")
    print("=" * 60)
    print(f"\nSimulation parameters:")
    print(f"  Time bins: {n_time}")
    print(f"  Time bin width: {dt * 1000:.1f} ms")
    print(f"  Total duration: {n_time * dt:.1f} s")
    print(f"  Transition probability: {transition_prob:.3%} per bin")
    mean_dwell_bins = int(round(1.0 / transition_prob))
    print(f"  Mean dwell time: {mean_dwell_bins * dt:.1f} s")
    print(f"\nDynamics:")
    print(f"  State 0: Very stable (A={STATE0_A}, Q={STATE0_Q})")
    print(f"  State 1: Noisy (A={STATE1_A}, Q={STATE1_Q})")

    # Simulate data
    key = jax.random.PRNGKey(42)
    key, sim_key = jax.random.split(key)
    spikes, true_states, true_latent, baseline, weights = simulate_switching_dynamics_neuron(
        n_time=n_time,
        dt=dt,
        transition_prob=transition_prob,
        key=sim_key,
    )

    n_transitions = jnp.sum(jnp.abs(jnp.diff(true_states)))
    mean_rate = jnp.sum(spikes) / (n_time * dt)
    print(f"\nSimulated data:")
    print(f"  Total spikes: {int(jnp.sum(spikes))}")
    print(f"  Number of state transitions: {int(n_transitions)}")
    print(f"  Mean firing rate: {float(mean_rate):.2f} Hz")
    print(
        f"  Latent state range: [{float(true_latent.min()):.2f}, {float(true_latent.max()):.2f}]"
    )

    # True-parameter inference (single neuron)
    print("\n" + "-" * 60)
    print("True-parameter inference (single neuron)")
    print("-" * 60)
    true_model = build_true_param_model(
        dt, transition_prob, baseline, weights, max_newton_iter=10
    )
    smoother_prob_true, marginal_mean_true = true_model.infer_states(spikes)
    (
        accuracy_true,
        latent_corr_true,
        posterior_entropy_true,
        smoother_prob_true,
        _,
        flipped_true,
    ) = summarize_inference(
        true_states, true_latent, smoother_prob_true, marginal_mean_true
    )
    if flipped_true:
        print("  (Note: Labels were flipped)")
    print(f"  State classification accuracy: {accuracy_true:.1%}")
    print(f"  Latent state correlation: {latent_corr_true:.3f}")
    print(f"  Posterior state entropy: {posterior_entropy_true:.3f}")

    # True-parameter inference (multi-neuron sanity check)
    print("\n" + "-" * 60)
    print("True-parameter inference (multi-neuron sanity check)")
    print("-" * 60)
    n_neurons_check = 10
    key, weight_key = jax.random.split(key)
    # Use smaller weights to avoid numerical issues
    weights_multi = 0.5 + 0.5 * jax.random.uniform(weight_key, (n_neurons_check, 1))
    baseline_multi = jnp.full((n_neurons_check,), 2.0)
    key, sim_key = jax.random.split(key)
    (
        spikes_multi,
        true_states_multi,
        true_latent_multi,
        baseline_multi,
        weights_multi,
    ) = simulate_switching_dynamics_neuron(
        n_time=n_time,
        dt=dt,
        transition_prob=transition_prob,
        key=sim_key,
        n_neurons=n_neurons_check,
        baseline=baseline_multi,
        weights=weights_multi,
    )
    mean_rate_multi = jnp.sum(spikes_multi, axis=0) / (n_time * dt)
    print(f"  Neurons: {n_neurons_check}")
    print(
        f"  Mean firing rate per neuron: {float(jnp.mean(mean_rate_multi)):.2f} Hz"
    )
    print(
        f"  Rate range: [{float(mean_rate_multi.min()):.2f}, {float(mean_rate_multi.max()):.2f}] Hz"
    )
    multi_model = build_true_param_model(
        dt, transition_prob, baseline_multi, weights_multi, max_newton_iter=10
    )
    smoother_prob_multi, marginal_mean_multi = multi_model.infer_states(spikes_multi)
    (
        accuracy_multi,
        latent_corr_multi,
        posterior_entropy_multi,
        _,
        _,
        flipped_multi,
    ) = summarize_inference(
        true_states_multi, true_latent_multi, smoother_prob_multi, marginal_mean_multi
    )
    if flipped_multi:
        print("  (Note: Labels were flipped)")
    print(f"  State classification accuracy: {accuracy_multi:.1%}")
    print(f"  Latent state correlation: {latent_corr_multi:.3f}")
    print(f"  Posterior state entropy: {posterior_entropy_multi:.3f}")

    # Plot results - compare single and multi-neuron
    plot_start = 0
    plot_end = min(10000, n_time)

    fig, axes = plt.subplots(4, 2, figsize=(18, 12), sharex=True)
    time = jnp.arange(plot_start, plot_end) * dt

    # Left column: Single neuron
    # Panel 1: Spike raster
    ax = axes[0, 0]
    spike_times = time[spikes[plot_start:plot_end, 0] > 0]
    ax.eventplot([spike_times], colors="black", linewidths=0.5)
    ax.set_ylabel("Spikes")
    ax.set_title(f"Single Neuron (Accuracy: {accuracy_true:.1%})")

    # Panel 2: True latent state
    ax = axes[1, 0]
    ax.plot(time, true_latent[plot_start:plot_end], "k-", alpha=0.7, label="True")
    ax.plot(time, marginal_mean_true[plot_start:plot_end, 0], "b-", alpha=0.7, label="Inferred")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Latent State")
    ax.legend(loc="upper right")

    # Panel 3: True discrete state
    ax = axes[2, 0]
    ax.fill_between(time, true_states[plot_start:plot_end], alpha=0.5, step="mid", color="C0", label="True state")
    ax.set_ylabel("True State")
    ax.set_ylim(-0.1, 1.1)

    # Panel 4: Inferred state probability
    ax = axes[3, 0]
    ax.fill_between(
        time,
        smoother_prob_true[plot_start:plot_end, 1],
        alpha=0.7,
        color="C1",
        label="P(Noisy state | data)",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("P(State 1)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (s)")

    # Right column: Multi-neuron
    # Panel 1: Spike raster
    ax = axes[0, 1]
    n_show = min(10, spikes_multi.shape[1])
    for i in range(n_show):
        spike_times = time[spikes_multi[plot_start:plot_end, i] > 0]
        ax.eventplot([spike_times], lineoffsets=i, colors="black", linewidths=0.5)
    ax.set_ylabel("Neuron")
    ax.set_title(f"Multi-neuron ({n_neurons_check} neurons, Accuracy: {accuracy_multi:.1%})")

    # Panel 2: True latent state
    ax = axes[1, 1]
    ax.plot(time, true_latent_multi[plot_start:plot_end], "k-", alpha=0.7, label="True")
    ax.plot(time, marginal_mean_multi[plot_start:plot_end, 0], "b-", alpha=0.7, label="Inferred")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Latent State")
    ax.legend(loc="upper right")

    # Panel 3: True discrete state
    ax = axes[2, 1]
    ax.fill_between(time, true_states_multi[plot_start:plot_end], alpha=0.5, step="mid", color="C0", label="True state")
    ax.set_ylabel("True State")
    ax.set_ylim(-0.1, 1.1)

    # Panel 4: Inferred state probability
    ax = axes[3, 1]
    ax.fill_between(
        time,
        smoother_prob_multi[plot_start:plot_end, 1],
        alpha=0.7,
        color="C1",
        label="P(Noisy state | data)",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("P(State 1)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (s)")

    plt.tight_layout()

    output_path = "switching_dynamics_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    plt.show()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  True transition prob: {transition_prob:.3%}")
    print(f"  Single neuron accuracy: {accuracy_true:.1%}")
    print(f"  Multi-neuron ({n_neurons_check}) accuracy: {accuracy_multi:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
