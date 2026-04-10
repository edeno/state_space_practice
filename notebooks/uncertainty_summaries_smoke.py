"""Smoke check: uncertainty summaries across all behavioral models.

Demonstrates that every model produces finite, correctly-shaped uncertainty
outputs on synthetic data.  Run with:

    conda run -n state_space_practice python notebooks/uncertainty_summaries_smoke.py
"""

import jax.numpy as jnp
import numpy as np

from state_space_practice.contingency_belief import ContingencyBeliefModel
from state_space_practice.covariate_choice import (
    CovariateChoiceModel,
    simulate_rl_choice_data,
)
from state_space_practice.multinomial_choice import (
    MultinomialChoiceModel,
    simulate_choice_data,
)
from state_space_practice.switching_choice import (
    SwitchingChoiceModel,
    simulate_switching_choice_data,
)

# ---------------------------------------------------------------------------
# 1. MultinomialChoiceModel — value variance over trials
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. MultinomialChoiceModel")
print("=" * 60)

sim = simulate_choice_data(n_trials=80, n_options=3, seed=0)
m1 = MultinomialChoiceModel(n_options=3)
m1.fit(sim.choices, max_iter=5)

assert m1.predicted_option_values_.shape == (80, 3), "predicted values shape"
assert m1.filtered_option_values_.shape == (80, 3), "filtered values shape"
assert m1.smoothed_option_values_.shape == (80, 3), "smoothed values shape"
assert m1.predicted_option_variances_.shape == (80, 3), "predicted var shape"
assert m1.filtered_option_variances_.shape == (80, 3), "filtered var shape"
assert m1.smoothed_option_variances_.shape == (80, 3), "smoothed var shape"
assert m1.predicted_choice_entropy_.shape == (80,), "entropy shape"
assert m1.surprise_.shape == (80,), "surprise shape"
assert jnp.all(jnp.isfinite(m1.predicted_option_variances_))
assert jnp.all(jnp.isfinite(m1.surprise_))
print(f"  Predicted var range: [{float(m1.predicted_option_variances_.min()):.4f}, "
      f"{float(m1.predicted_option_variances_.max()):.4f}]")
print(f"  Entropy range:       [{float(m1.predicted_choice_entropy_.min()):.4f}, "
      f"{float(m1.predicted_choice_entropy_.max()):.4f}]")
print(f"  Surprise range:      [{float(m1.surprise_.min()):.4f}, "
      f"{float(m1.surprise_.max()):.4f}]")
print("  PASS")

# ---------------------------------------------------------------------------
# 2. CovariateChoiceModel — value variance over trials
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. CovariateChoiceModel")
print("=" * 60)

sim2 = simulate_rl_choice_data(n_trials=80, n_options=3, seed=0)
m2 = CovariateChoiceModel(n_options=3, n_covariates=sim2.covariates.shape[1])
m2.fit(sim2.choices, covariates=sim2.covariates, max_iter=5)

assert m2.predicted_option_values_.shape == (80, 3)
assert m2.filtered_option_values_.shape == (80, 3)
assert m2.smoothed_option_values_.shape == (80, 3)
assert m2.filtered_option_variances_.shape == (80, 3)
assert jnp.all(jnp.isfinite(m2.predicted_option_variances_))
assert jnp.all(jnp.isfinite(m2.surprise_))
print(f"  Predicted var range: [{float(m2.predicted_option_variances_.min()):.4f}, "
      f"{float(m2.predicted_option_variances_.max()):.4f}]")
print(f"  Surprise range:      [{float(m2.surprise_.min()):.4f}, "
      f"{float(m2.surprise_.max()):.4f}]")
print("  PASS")

# ---------------------------------------------------------------------------
# 3. ContingencyBeliefModel — surprise spikes at contingency changes,
#    belief entropy, change-point probability
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. ContingencyBeliefModel")
print("=" * 60)

rng = np.random.default_rng(42)
n_trials = 120
# Block 1 (0-59): option 0 rewarded 80%, Block 2 (60-119): option 1 rewarded 80%
choices = np.array([0] * 60 + [1] * 60)
rewards = np.zeros(n_trials)
for t in range(60):
    rewards[t] = rng.binomial(1, 0.8)
for t in range(60, n_trials):
    rewards[t] = rng.binomial(1, 0.8)

m3 = ContingencyBeliefModel(n_states=2, n_options=2)
m3.fit(choices, rewards, max_iter=10)

assert m3.belief_entropy_ is not None and m3.belief_entropy_.shape == (n_trials,)
assert m3.surprise_ is not None and m3.surprise_.shape == (n_trials,)
assert m3.change_point_probability_ is not None
assert m3.change_point_probability_.shape == (n_trials,)
assert m3.predicted_reward_mean_ is not None
assert m3.predicted_reward_variance_ is not None
assert jnp.all(jnp.isfinite(m3.belief_entropy_))
assert jnp.all(jnp.isfinite(m3.surprise_))

# Surprise should spike near the block boundary
early_surprise = float(jnp.mean(m3.surprise_[10:50]))
boundary_surprise = float(jnp.mean(m3.surprise_[55:75]))
print(f"  Belief entropy range: [{float(m3.belief_entropy_.min()):.4f}, "
      f"{float(m3.belief_entropy_.max()):.4f}]")
print(f"  Surprise (early):     {early_surprise:.4f}")
print(f"  Surprise (boundary):  {boundary_surprise:.4f}")
print(f"  Change-point prob max: {float(m3.change_point_probability_.max()):.4f}")
print("  PASS")

# ---------------------------------------------------------------------------
# 4. SwitchingChoiceModel — predicted choice entropy, exploit vs explore
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. SwitchingChoiceModel")
print("=" * 60)

sim4 = simulate_switching_choice_data(
    n_trials=80, n_options=3, n_discrete_states=2, seed=0,
)
m4 = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
m4.fit_sgd(sim4.choices, num_steps=20)

assert m4.predicted_choice_entropy_.shape == (80,)
assert m4.surprise_.shape == (80,)
assert m4.predicted_option_variances_.shape == (80, 3)
assert m4.per_state_predicted_variances_.shape == (80, 3, 2)
assert jnp.all(jnp.isfinite(m4.predicted_choice_entropy_))
assert jnp.all(jnp.isfinite(m4.surprise_))
print(f"  Entropy range:       [{float(m4.predicted_choice_entropy_.min()):.4f}, "
      f"{float(m4.predicted_choice_entropy_.max()):.4f}]")
print(f"  Surprise range:      [{float(m4.surprise_.min()):.4f}, "
      f"{float(m4.surprise_.max()):.4f}]")
print(f"  Per-state var shape: {m4.per_state_predicted_variances_.shape}")
print("  PASS")

# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("All smoke checks passed.")
print("=" * 60)
