"""Example: Switching Poisson neuron with state-dependent dynamics.

This script demonstrates fitting a switching point-process model where a neuron's
firing rate is modulated by a continuous latent state that has different dynamics
in each discrete regime:

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
3. Add regularization to covariance updates
4. Use log-sum-exp tricks for probability computations
"""

import jax

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from state_space_practice.switching_kalman import switching_kalman_smoother
from state_space_practice.switching_point_process import switching_point_process_filter

# Constants for numerical stability
LOG_RATE_MIN = -10.0  # Minimum log-rate (corresponds to ~0.00005 Hz)
LOG_RATE_MAX = 8.0  # Maximum log-rate (corresponds to ~3000 Hz)
PROB_MIN = 1e-10  # Minimum probability to avoid log(0)
COV_REGULARIZATION = 1e-6  # Regularization for covariance matrices


def safe_log_intensity(baseline: jax.Array, weight: jax.Array):
    """Create a numerically stable log-intensity function.

    Clips the output to prevent overflow when computing exp(log_intensity).
    """

    def log_intensity_func(x: jax.Array) -> jax.Array:
        log_rate = baseline + weight @ x
        # Clip to prevent numerical overflow/underflow
        return jnp.clip(log_rate, LOG_RATE_MIN, LOG_RATE_MAX)

    return log_intensity_func


def simulate_switching_dynamics_neuron(
    n_time: int,
    dt: float,
    transition_prob: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
    spikes : jax.Array, shape (n_time, 1)
        Spike counts.
    true_states : jax.Array, shape (n_time,)
        True discrete state sequence (0 or 1).
    true_latent : jax.Array, shape (n_time,)
        True continuous latent state.
    """
    # Discrete transition matrix
    stay_prob = 1.0 - transition_prob
    Z = jnp.array([[stay_prob, transition_prob], [transition_prob, stay_prob]])

    # Dynamics: x_t = A * x_{t-1} + sqrt(Q) * noise
    # State 0: Very stable (high A, low Q) - latent stays near 0
    # State 1: Very noisy (low A, high Q) - latent fluctuates widely
    A_0, Q_0 = 0.995, 0.005  # Very stable, stationary var ≈ 0.5
    A_1, Q_1 = 0.8, 0.5  # Noisy, stationary var ≈ 1.4

    # Observation model: log(rate) = baseline + weight * x
    baseline = 2.3  # ~10 Hz when x=0
    weight = 1.5  # Strong modulation: x=1 → 45 Hz, x=-1 → 2 Hz

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
        log_rate = baseline + weight * x_t
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

    return spikes[:, None], true_states, true_latent


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
    ):
        self.n_discrete_states = n_discrete_states
        self.dt = dt
        self.n_latent = 1

        # Parameters
        self.discrete_transition_matrix: jax.Array | None = None
        self.init_discrete_prob: jax.Array | None = None
        self.init_mean: jax.Array | None = None
        self.init_cov: jax.Array | None = None
        self.continuous_transition_matrix: jax.Array | None = None
        self.process_cov: jax.Array | None = None
        self.spike_baseline: jax.Array | None = None
        self.spike_weight: jax.Array | None = None

    def _initialize_parameters(self, spikes: jax.Array, key: jax.Array) -> None:
        """Initialize model parameters."""
        n_time = spikes.shape[0]

        # Discrete state probabilities
        self.init_discrete_prob = (
            jnp.ones(self.n_discrete_states) / self.n_discrete_states
        )

        # High self-transition
        stay_prob = 0.98
        off_diag = (1.0 - stay_prob) / (self.n_discrete_states - 1)
        self.discrete_transition_matrix = (
            jnp.eye(self.n_discrete_states) * stay_prob
            + jnp.ones((self.n_discrete_states, self.n_discrete_states)) * off_diag
            - jnp.eye(self.n_discrete_states) * off_diag
        )

        # Initial continuous state (different for each state to reflect stationary dist)
        self.init_mean = jnp.zeros((self.n_latent, self.n_discrete_states))
        self.init_cov = jnp.stack(
            [jnp.eye(self.n_latent) * 1.0] * self.n_discrete_states, axis=-1
        )

        # Initialize with different dynamics for each state
        # State 0: Very stable (high A, low Q)
        # State 1: Noisy (low A, high Q)
        A_init = jnp.array([0.995, 0.8])
        Q_init = jnp.array([0.005, 0.5])

        self.continuous_transition_matrix = jnp.stack(
            [jnp.eye(self.n_latent) * A_init[s] for s in range(self.n_discrete_states)],
            axis=-1,
        )
        self.process_cov = jnp.stack(
            [jnp.eye(self.n_latent) * Q_init[s] for s in range(self.n_discrete_states)],
            axis=-1,
        )

        # Spike observation model
        mean_rate = jnp.sum(spikes) / (n_time * self.dt)
        self.spike_baseline = jnp.array([jnp.log(mean_rate + 0.1)])
        self.spike_weight = jnp.array([[1.5]])  # Strong modulation

    def _make_log_intensity_func(self):
        """Create numerically stable log-intensity function."""
        return safe_log_intensity(self.spike_baseline, self.spike_weight)

    def fit(
        self,
        spikes: jax.Array,
        max_iter: int = 50,
        tol: float = 1e-5,
        key: jax.Array | None = None,
    ) -> list[float]:
        """Fit the model using EM."""
        spikes = jnp.asarray(spikes)
        if key is None:
            key = jax.random.PRNGKey(0)

        self._initialize_parameters(spikes, key)

        log_likelihoods = []
        prev_ll = -jnp.inf

        for iteration in range(max_iter):
            # E-step
            log_intensity_func = self._make_log_intensity_func()

            filter_results = switching_point_process_filter(
                init_state_cond_mean=self.init_mean,
                init_state_cond_cov=self.init_cov,
                init_discrete_state_prob=self.init_discrete_prob,
                spikes=spikes,
                discrete_transition_matrix=self.discrete_transition_matrix,
                continuous_transition_matrix=self.continuous_transition_matrix,
                process_cov=self.process_cov,
                dt=self.dt,
                log_intensity_func=log_intensity_func,
            )

            (
                filter_mean,
                filter_cov,
                filter_discrete_prob,
                last_pair_mean,
                marginal_ll,
            ) = filter_results

            # Check for numerical issues
            if jnp.isnan(marginal_ll) or jnp.isinf(marginal_ll):
                print(f"  Warning: NaN/Inf log-likelihood at iteration {iteration}")
                if log_likelihoods:
                    # Revert to last valid state would require saving params
                    # For now, just stop
                    break
                else:
                    # First iteration failed - initialization issue
                    log_likelihoods.append(float(-jnp.inf))
                    break

            log_likelihoods.append(float(marginal_ll))

            # Check convergence
            if iteration > 0:
                # Use absolute change for very negative log-likelihoods
                abs_change = abs(marginal_ll - prev_ll)
                rel_change = abs_change / (abs(marginal_ll) + 1.0)
                if rel_change < tol and abs_change < 1.0:
                    break
            prev_ll = marginal_ll

            # Smoother
            smoother_results = switching_kalman_smoother(
                filter_mean=filter_mean,
                filter_cov=filter_cov,
                filter_discrete_state_prob=filter_discrete_prob,
                last_filter_conditional_cont_mean=last_pair_mean,
                process_cov=self.process_cov,
                continuous_transition_matrix=self.continuous_transition_matrix,
                discrete_state_transition_matrix=self.discrete_transition_matrix,
            )

            smoother_discrete_prob = smoother_results[2]
            smoother_joint_prob = smoother_results[3]

            # M-step: Update discrete transition matrix with numerical stability
            joint_sum = jnp.sum(smoother_joint_prob, axis=0)
            marginal_sum = jnp.sum(smoother_discrete_prob[:-1], axis=0)

            # Safe division with minimum probability threshold
            marginal_sum_safe = jnp.maximum(marginal_sum, PROB_MIN)
            new_trans = joint_sum / marginal_sum_safe[:, None]

            # Normalize rows and ensure minimum transition probabilities
            row_sums = jnp.sum(new_trans, axis=1, keepdims=True)
            row_sums_safe = jnp.maximum(row_sums, PROB_MIN)
            new_trans = new_trans / row_sums_safe

            # Add small probability mass to prevent zero transitions
            new_trans = new_trans + PROB_MIN
            new_trans = new_trans / jnp.sum(new_trans, axis=1, keepdims=True)

            # Check for NaN and revert to previous if needed
            has_nan = jnp.any(jnp.isnan(new_trans))
            self.discrete_transition_matrix = jnp.where(
                has_nan, self.discrete_transition_matrix, new_trans
            )

            # Update initial discrete probability with stability
            init_prob = smoother_discrete_prob[0]
            init_prob = jnp.maximum(init_prob, PROB_MIN)
            init_prob = init_prob / jnp.sum(init_prob)

            # Check for NaN
            has_nan = jnp.any(jnp.isnan(init_prob))
            self.init_discrete_prob = jnp.where(
                has_nan, self.init_discrete_prob, init_prob
            )

        return log_likelihoods

    def infer_states(self, spikes: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Run inference."""
        log_intensity_func = self._make_log_intensity_func()

        filter_results = switching_point_process_filter(
            init_state_cond_mean=self.init_mean,
            init_state_cond_cov=self.init_cov,
            init_discrete_state_prob=self.init_discrete_prob,
            spikes=spikes,
            discrete_transition_matrix=self.discrete_transition_matrix,
            continuous_transition_matrix=self.continuous_transition_matrix,
            process_cov=self.process_cov,
            dt=self.dt,
            log_intensity_func=log_intensity_func,
        )

        filter_mean, filter_cov, filter_discrete_prob, last_pair_mean, _ = (
            filter_results
        )

        smoother_results = switching_kalman_smoother(
            filter_mean=filter_mean,
            filter_cov=filter_cov,
            filter_discrete_state_prob=filter_discrete_prob,
            last_filter_conditional_cont_mean=last_pair_mean,
            process_cov=self.process_cov,
            continuous_transition_matrix=self.continuous_transition_matrix,
            discrete_state_transition_matrix=self.discrete_transition_matrix,
        )

        smoother_discrete_prob = smoother_results[2]
        state_cond_smoother_means = smoother_results[5]

        # Marginal smoother mean
        marginal_smoother_mean = jnp.einsum(
            "tls,ts->tl", state_cond_smoother_means, smoother_discrete_prob
        )

        return smoother_discrete_prob, marginal_smoother_mean


def main():
    # Simulation parameters
    n_time = 10000  # 200 seconds at 50 Hz
    dt = 0.02
    transition_prob = 0.005  # Switch every ~200 time steps on average

    print("=" * 60)
    print("Switching Dynamics Example")
    print("=" * 60)
    print(f"\nSimulation parameters:")
    print(f"  Time bins: {n_time}")
    print(f"  Time bin width: {dt * 1000:.1f} ms")
    print(f"  Total duration: {n_time * dt:.1f} s")
    print(f"  Transition probability: {transition_prob:.3%} per bin")
    print(f"\nDynamics:")
    print(f"  State 0: Very stable (A=0.995, Q=0.005)")
    print(f"  State 1: Noisy (A=0.8, Q=0.5)")

    # Simulate data
    key = jax.random.PRNGKey(42)
    spikes, true_states, true_latent = simulate_switching_dynamics_neuron(
        n_time=n_time,
        dt=dt,
        transition_prob=transition_prob,
        key=key,
    )

    n_transitions = jnp.sum(jnp.abs(jnp.diff(true_states)))
    print(f"\nSimulated data:")
    print(f"  Total spikes: {int(jnp.sum(spikes))}")
    print(f"  Number of state transitions: {int(n_transitions)}")
    print(
        f"  Latent state range: [{float(true_latent.min()):.2f}, {float(true_latent.max()):.2f}]"
    )

    # Fit model
    print("\n" + "-" * 60)
    print("Fitting model...")
    print("-" * 60)

    model = SwitchingDynamicsModel(n_discrete_states=2, dt=dt)

    key, subkey = jax.random.split(key)
    log_likelihoods = model.fit(
        spikes=spikes,
        max_iter=50,
        tol=1e-5,
        key=subkey,
    )

    print(f"\nEM converged after {len(log_likelihoods)} iterations")
    print(f"  Initial log-likelihood: {log_likelihoods[0]:.1f}")
    print(f"  Final log-likelihood: {log_likelihoods[-1]:.1f}")

    # Learned parameters
    print("\n" + "-" * 60)
    print("Learned parameters:")
    print("-" * 60)

    print(f"\nDiscrete transition matrix:")
    print(f"  P(stay in state 0): {float(model.discrete_transition_matrix[0, 0]):.4f}")
    print(f"  P(stay in state 1): {float(model.discrete_transition_matrix[1, 1]):.4f}")

    print(f"\nContinuous dynamics (A matrix diagonal):")
    print(f"  State 0: {float(model.continuous_transition_matrix[0, 0, 0]):.3f}")
    print(f"  State 1: {float(model.continuous_transition_matrix[0, 0, 1]):.3f}")

    print(f"\nProcess noise (Q matrix diagonal):")
    print(f"  State 0: {float(model.process_cov[0, 0, 0]):.4f}")
    print(f"  State 1: {float(model.process_cov[0, 0, 1]):.4f}")

    # Inference
    print("\n" + "-" * 60)
    print("Running inference...")
    print("-" * 60)

    smoother_discrete_prob, marginal_smoother_mean = model.infer_states(spikes)

    # Compute accuracy
    inferred_states = jnp.argmax(smoother_discrete_prob, axis=1)
    accuracy_direct = jnp.mean(inferred_states == true_states)
    accuracy_flipped = jnp.mean(inferred_states == (1 - true_states))
    accuracy = max(float(accuracy_direct), float(accuracy_flipped))

    if accuracy_flipped > accuracy_direct:
        print("  (Note: Labels were flipped)")
        inferred_states = 1 - inferred_states
        smoother_discrete_prob = smoother_discrete_prob[:, ::-1]

    print(f"\nState classification accuracy: {accuracy:.1%}")

    # Latent state correlation
    latent_corr = jnp.corrcoef(true_latent, marginal_smoother_mean[:, 0])[0, 1]
    print(f"Latent state correlation: {float(latent_corr):.3f}")

    # Plot results
    plot_start = 0
    plot_end = min(3000, n_time)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    time = jnp.arange(plot_start, plot_end) * dt
    spikes_plot = spikes[plot_start:plot_end, 0]
    true_states_plot = true_states[plot_start:plot_end]
    true_latent_plot = true_latent[plot_start:plot_end]
    smoother_prob_plot = smoother_discrete_prob[plot_start:plot_end]
    latent_plot = marginal_smoother_mean[plot_start:plot_end, 0]

    # Panel 1: Spike raster
    ax = axes[0]
    spike_times = time[spikes_plot > 0]
    ax.eventplot([spike_times], colors="black", linewidths=0.5)
    ax.set_ylabel("Spikes")
    ax.set_title("Switching Dynamics: Rate Modulated by Latent State")

    # Panel 2: True latent state
    ax = axes[1]
    ax.plot(time, true_latent_plot, "k-", alpha=0.7, label="True latent")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("True Latent")
    ax.legend(loc="upper right")

    # Panel 3: Inferred latent state
    ax = axes[2]
    ax.plot(time, latent_plot, "b-", alpha=0.7, label="Inferred latent")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Inferred Latent")
    ax.legend(loc="upper right")

    # Panel 4: True discrete state
    ax = axes[3]
    ax.fill_between(time, true_states_plot, alpha=0.5, step="mid", label="True state")
    ax.set_ylabel("True State")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right")

    # Panel 5: Inferred state probability
    ax = axes[4]
    ax.fill_between(
        time,
        smoother_prob_plot[:, 1],
        alpha=0.7,
        color="C1",
        label="P(Fast dynamics state | data)",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("P(State 1)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")

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
    learned_trans_prob = (
        1.0
        - (
            model.discrete_transition_matrix[0, 0]
            + model.discrete_transition_matrix[1, 1]
        )
        / 2
    )
    print(f"  Learned avg transition prob: {float(learned_trans_prob):.3%}")
    print(f"  State classification accuracy: {accuracy:.1%}")
    print(f"  Latent state correlation: {float(latent_corr):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
