"""Smoke check for ContingencyBeliefModel.

Simulates block-structured bandit data, fits with both EM and SGD,
and prints diagnostics: posterior state occupancy, learned reward
templates, and transition summary.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from state_space_practice.contingency_belief import ContingencyBeliefModel


def simulate_block_bandit(n_trials=200, n_options=3, seed=42):
    """Block-structured bandit: state 0 for first half, state 1 for second."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    reward_probs = jnp.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8],
    ])
    half = n_trials // 2
    true_states = jnp.concatenate([
        jnp.zeros(half), jnp.ones(n_trials - half)
    ]).astype(jnp.int32)

    best_options = jnp.array([0, 2])
    choices = best_options[true_states]
    noise_mask = jax.random.bernoulli(k1, 0.2, (n_trials,))
    random_choices = jax.random.randint(k2, (n_trials,), 0, n_options)
    choices = jnp.where(noise_mask, random_choices, choices)

    reward_p = reward_probs[true_states, choices]
    rewards = jax.random.bernoulli(
        jax.random.PRNGKey(seed + 1), reward_p
    ).astype(jnp.int32)

    return choices, rewards, true_states, reward_probs


def main():
    print("=" * 60)
    print("ContingencyBeliefModel Smoke Check")
    print("=" * 60)

    choices, rewards, true_states, true_rp = simulate_block_bandit(
        n_trials=200, n_options=3,
    )
    print(f"\nData: {len(choices)} trials, 3 options")
    print(f"True states: 0 for first half, 1 for second half")
    print(f"True reward probs:\n  State 0: {true_rp[0]}")
    print(f"  State 1: {true_rp[1]}")

    # --- EM Fit ---
    print("\n--- EM Fit ---")
    model_em = ContingencyBeliefModel(n_states=2, n_options=3)
    em_lls = model_em.fit(choices, rewards, max_iter=30)
    print(f"EM iterations: {len(em_lls)}")
    print(f"Final LL: {em_lls[-1]:.2f}")
    print(f"Learned reward probs:\n  State 0: {model_em.reward_probs_[0]}")
    print(f"  State 1: {model_em.reward_probs_[1]}")
    trans = jax.nn.softmax(model_em.transition_logits_, axis=1)
    print(f"Transition matrix:\n{trans}")

    # --- SGD Fit ---
    print("\n--- SGD Fit ---")
    model_sgd = ContingencyBeliefModel(n_states=2, n_options=3)
    sgd_lls = model_sgd.fit_sgd(choices, rewards, num_steps=200)
    print(f"SGD steps: {len(sgd_lls)}")
    print(f"Final LL: {sgd_lls[-1]:.2f}")
    print(f"Learned reward probs:\n  State 0: {model_sgd.reward_probs_[0]}")
    print(f"  State 1: {model_sgd.reward_probs_[1]}")
    print(f"Inverse temperature: {model_sgd.inverse_temperature_:.3f}")
    print(f"State values:\n  State 0: {model_sgd.state_values_[0]}")
    print(f"  State 1: {model_sgd.state_values_[1]}")

    # --- Posterior State Occupancy ---
    posterior = model_sgd.smoothed_state_posterior_
    print("\n--- Posterior State Occupancy (SGD) ---")
    print(f"First 10 trials:  state 0 avg = {posterior[:10, 0].mean():.3f}")
    print(f"Last 10 trials:   state 0 avg = {posterior[-10:, 0].mean():.3f}")

    first_half_state = jnp.argmax(posterior[:50].mean(axis=0))
    second_half_state = jnp.argmax(posterior[150:].mean(axis=0))
    if first_half_state != second_half_state:
        print("Block structure RECOVERED by SGD")
    else:
        print("Block structure NOT recovered (states not distinguished)")

    # --- Transition Covariates ---
    print("\n--- SGD with Transition Covariates ---")
    n_trials = len(choices)
    covariates = jnp.zeros((n_trials, 1))
    covariates = covariates.at[n_trials // 2, 0].set(1.0)

    model_cov = ContingencyBeliefModel(
        n_states=2, n_options=3, n_transition_covariates=1,
    )
    cov_lls = model_cov.fit_sgd(
        choices, rewards,
        transition_covariates=covariates,
        num_steps=200,
    )
    print(f"Final LL with covariates: {cov_lls[-1]:.2f}")
    print(f"Transition weights: {model_cov.transition_weights_}")

    print("\n" + "=" * 60)
    print("Smoke check complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
