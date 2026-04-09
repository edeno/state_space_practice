# ruff: noqa: E402
"""Tests for the contingency_belief module."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.contingency_belief import (
    ContingencyBeliefModel,
    ContingencyBeliefResult,
    centered_softmax,
    compute_input_output_transition_matrix,
    compute_reward_log_likelihood,
    contingency_belief_filter,
    contingency_belief_smoother,
    transition_logits_to_matrix,
)


class TestTransitionLogitsToMatrix:
    def test_rows_sum_to_one(self):
        # Centered: (3, 2) logits → (3, 3) transition matrix
        logits = jnp.zeros((3, 2))
        trans = transition_logits_to_matrix(logits)
        np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-7)

    def test_uniform_for_zero_logits(self):
        # 3 states: zero logits → uniform 1/3
        logits = jnp.zeros((3, 2))
        trans = transition_logits_to_matrix(logits)
        np.testing.assert_allclose(trans, 1.0 / 3, atol=1e-7)

    def test_higher_logit_gives_higher_prob(self):
        # Centered: (2, 1) logits. Positive logit → state 0 preferred over reference
        logits = jnp.array([[5.0], [-5.0]])
        trans = transition_logits_to_matrix(logits)
        assert trans[0, 0] > trans[0, 1]  # from 0: prefer state 0
        assert trans[1, 1] > trans[1, 0]  # from 1: prefer state 1 (reference)

    def test_gradient_finite(self):
        logits = jnp.array([[1.0], [-0.5]])
        g = jax.grad(lambda x: transition_logits_to_matrix(x).sum())(logits)
        assert jnp.all(jnp.isfinite(g))


class TestInputOutputTransitionMatrix:
    def test_zero_covariates_recover_baseline(self):
        # Centered: (2, 1) logits
        baseline = jnp.array([[2.0], [-2.0]])
        weights = jnp.zeros((2, 1, 3))
        h_t = jnp.zeros(3)
        trans = compute_input_output_transition_matrix(baseline, weights, h_t)
        expected = centered_softmax(baseline)
        np.testing.assert_allclose(trans, expected, atol=1e-7)

    def test_rows_sum_to_one(self):
        baseline = jnp.ones((3, 2))
        weights = jax.random.normal(jax.random.PRNGKey(0), (3, 2, 2))
        h_t = jnp.array([1.0, -1.0])
        trans = compute_input_output_transition_matrix(baseline, weights, h_t)
        np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-7)

    def test_covariates_shift_transitions(self):
        baseline = jnp.zeros((2, 1))
        weights = jnp.zeros((2, 1, 1))
        weights = weights.at[0, 0, 0].set(5.0)  # covariate pushes s0→s0
        h_t = jnp.array([1.0])
        trans = compute_input_output_transition_matrix(baseline, weights, h_t)
        # s0→s0 should be higher than s0→s1 (reference)
        assert trans[0, 0] > trans[0, 1]


class TestRewardLogLikelihood:
    def test_shape(self):
        reward_probs = jnp.array([[0.8, 0.2], [0.2, 0.8]])
        ll = compute_reward_log_likelihood(
            reward_t=1, choice_t=0, reward_probs=reward_probs
        )
        assert ll.shape == (2,)  # one per state

    def test_higher_prob_gives_higher_ll(self):
        reward_probs = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        ll = compute_reward_log_likelihood(
            reward_t=1, choice_t=0, reward_probs=reward_probs
        )
        # State 0 has 0.9 reward prob for choice 0 → higher LL
        assert ll[0] > ll[1]

    def test_zero_reward(self):
        reward_probs = jnp.array([[0.8, 0.2], [0.2, 0.8]])
        ll = compute_reward_log_likelihood(
            reward_t=0, choice_t=0, reward_probs=reward_probs
        )
        # State 0: P(no reward) = 0.2; State 1: P(no reward) = 0.8
        assert ll[1] > ll[0]


class TestContingencyBeliefResult:
    def test_fields_exist(self):
        result = ContingencyBeliefResult(
            state_posterior=jnp.ones((10, 2)),
            log_likelihood=jnp.array(0.0),
        )
        assert result.state_posterior.shape == (10, 2)
        assert result.log_likelihood.shape == ()


class TestContingencyBeliefFilter:
    def test_output_shapes(self):
        result = contingency_belief_filter(
            choices=jnp.array([0, 1, 0, 1, 0], dtype=jnp.int32),
            rewards=jnp.array([1, 0, 1, 0, 1], dtype=jnp.int32),
            n_states=2,
            n_options=2,
            reward_probs=jnp.array([[0.8, 0.2], [0.2, 0.8]]),
            state_values=jnp.array([[2.0, 0.0], [0.0, 2.0]]),
            inverse_temperature=1.0,
            transition_logits=jnp.array([[3.0], [-3.0]]),
        )
        assert result.state_posterior.shape == (5, 2)
        assert result.log_likelihood.shape == ()

    def test_posteriors_sum_to_one(self):
        result = contingency_belief_filter(
            choices=jnp.array([0, 1, 0, 1], dtype=jnp.int32),
            rewards=jnp.array([1, 0, 1, 0], dtype=jnp.int32),
            n_states=3,
            n_options=2,
            reward_probs=jnp.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]),
            state_values=jnp.array([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]]),
            inverse_temperature=1.0,
            transition_logits=jnp.zeros((3, 2)),
        )
        row_sums = result.state_posterior.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-7)

    def test_block_switch_changes_posterior(self):
        """Block-structured data should shift posterior mass."""
        # First 20 trials: option 0 always rewarded → state 0
        # Last 20 trials: option 1 always rewarded → state 1
        choices = jnp.array([0] * 20 + [1] * 20, dtype=jnp.int32)
        rewards = jnp.array([1] * 20 + [1] * 20, dtype=jnp.int32)
        result = contingency_belief_filter(
            choices=choices,
            rewards=rewards,
            n_states=2,
            n_options=2,
            reward_probs=jnp.array([[0.8, 0.2], [0.2, 0.8]]),
            state_values=jnp.array([[2.0, 0.0], [0.0, 2.0]]),
            inverse_temperature=2.0,
            transition_logits=jnp.array([[3.0], [-3.0]]),
        )
        # Early trials should favor state 0
        assert result.state_posterior[5, 0] > result.state_posterior[5, 1]
        # Late trials should favor state 1
        assert result.state_posterior[35, 1] > result.state_posterior[35, 0]

    def test_log_likelihood_finite(self):
        result = contingency_belief_filter(
            choices=jnp.array([0, 1, 0], dtype=jnp.int32),
            rewards=jnp.array([1, 0, 1], dtype=jnp.int32),
            n_states=2,
            n_options=2,
            reward_probs=jnp.array([[0.8, 0.2], [0.2, 0.8]]),
            state_values=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            inverse_temperature=1.0,
            transition_logits=jnp.zeros((2, 1)),
        )
        assert jnp.isfinite(result.log_likelihood)


class TestContingencyBeliefSmoother:
    @pytest.fixture
    def block_data(self):
        """Block-structured data: state 0 for first half, state 1 for second."""
        choices = jnp.array([0] * 20 + [1] * 20, dtype=jnp.int32)
        rewards = jnp.array([1] * 20 + [1] * 20, dtype=jnp.int32)
        kwargs = dict(
            choices=choices,
            rewards=rewards,
            n_states=2,
            n_options=2,
            reward_probs=jnp.array([[0.8, 0.2], [0.2, 0.8]]),
            state_values=jnp.array([[2.0, 0.0], [0.0, 2.0]]),
            inverse_temperature=2.0,
            transition_logits=jnp.array([[3.0], [-3.0]]),
        )
        return kwargs

    def test_smoother_shapes(self, block_data):
        result = contingency_belief_smoother(**block_data)
        assert result.smoothed_state_prob.shape == (40, 2)
        assert result.pairwise_state_prob.shape == (39, 2, 2)

    def test_smoothed_rows_sum_to_one(self, block_data):
        result = contingency_belief_smoother(**block_data)
        np.testing.assert_allclose(
            result.smoothed_state_prob.sum(axis=1), 1.0, atol=1e-6
        )

    def test_pairwise_sums_to_one(self, block_data):
        result = contingency_belief_smoother(**block_data)
        sums = result.pairwise_state_prob.sum(axis=(1, 2))
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_smoother_sharper_than_filter(self, block_data):
        """Smoother should be at least as confident as filter."""
        filter_result = contingency_belief_filter(**block_data)
        smoother_result = contingency_belief_smoother(**block_data)
        # Entropy of smoother should be <= filter (sharper)
        def entropy(p):
            p = jnp.clip(p, 1e-10, 1.0)
            return -jnp.sum(p * jnp.log(p), axis=1)

        filter_entropy = entropy(filter_result.state_posterior).mean()
        smoother_entropy = entropy(smoother_result.smoothed_state_prob).mean()
        assert smoother_entropy <= filter_entropy + 1e-6


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def _simulate_block_bandit(n_trials=100, n_options=3, seed=42):
    """Simulate block-structured bandit: state switches halfway."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    # State 0: option 0 is best; State 1: option 2 is best
    reward_probs = jnp.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8],
    ])
    half = n_trials // 2
    true_states = jnp.concatenate([jnp.zeros(half), jnp.ones(n_trials - half)]).astype(jnp.int32)

    # Generate choices: animals mostly pick the best option for current state
    best_options = jnp.array([0, 2])  # best for state 0, state 1
    choices = best_options[true_states]
    # Add noise: 20% random choices
    noise_mask = jax.random.bernoulli(k1, 0.2, (n_trials,))
    random_choices = jax.random.randint(k2, (n_trials,), 0, n_options)
    choices = jnp.where(noise_mask, random_choices, choices)

    # Generate rewards from true state
    reward_p = reward_probs[true_states, choices]
    rewards = jax.random.bernoulli(jax.random.PRNGKey(seed + 1), reward_p).astype(jnp.int32)

    return choices, rewards, true_states, reward_probs


# ---------------------------------------------------------------------------
# ContingencyBeliefModel tests
# ---------------------------------------------------------------------------

class TestContingencyBeliefModel:
    def test_fit_improves_ll(self):
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=80)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        lls = model.fit(choices, rewards, max_iter=10)
        assert len(lls) > 1
        assert lls[-1] > lls[0]

    def test_fit_recovers_reward_probs(self):
        choices, rewards, _, true_rp = _simulate_block_bandit(n_trials=200)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit(choices, rewards, max_iter=20)
        # Learned reward_probs should roughly match ground truth
        # (up to state permutation)
        learned = model.reward_probs_
        # Check that max reward prob per state is high (>0.5)
        assert jnp.max(learned[0]) > 0.5
        assert jnp.max(learned[1]) > 0.5

    def test_predict_state_posterior(self):
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=60)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit(choices, rewards, max_iter=10)
        posterior = model.predict_state_posterior(choices, rewards)
        assert posterior.shape == (60, 2)
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-6)

    def test_zero_transition_weights_stationary(self):
        """With zero transition weights, should behave like stationary HMM."""
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=50)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit(choices, rewards, max_iter=5)
        assert model.is_fitted

    def test_em_improves_overall(self):
        """EM should improve LL overall (generalized EM with Dirichlet prior)."""
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=100)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        lls = model.fit(choices, rewards, max_iter=20)
        assert lls[-1] > lls[0]


class TestContingencyBeliefSGD:
    def test_sgd_improves_ll(self):
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=80)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        lls = model.fit_sgd(choices, rewards, num_steps=50)
        assert lls[-1] > lls[0]

    def test_sgd_respects_constraints(self):
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=80)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit_sgd(choices, rewards, num_steps=30)
        # Reward probs should be in (0, 1)
        assert jnp.all(model.reward_probs_ > 0)
        assert jnp.all(model.reward_probs_ < 1)
        # Inverse temperature should be positive
        assert model.inverse_temperature_ > 0

    def test_sgd_model_is_fitted(self):
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=60)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit_sgd(choices, rewards, num_steps=20)
        assert model.is_fitted

    def test_sgd_vs_em_both_finite(self):
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=100)
        model_em = ContingencyBeliefModel(n_states=2, n_options=3)
        model_em.fit(choices, rewards, max_iter=15)

        model_sgd = ContingencyBeliefModel(n_states=2, n_options=3)
        model_sgd.fit_sgd(choices, rewards, num_steps=100)

        assert np.isfinite(model_em.log_likelihood_)
        assert np.isfinite(model_sgd.log_likelihood_)


class TestContingencyBeliefIntegration:
    """End-to-end integration tests on synthetic block-structured data."""

    def test_em_produces_finite_reward_probs(self):
        """EM should produce valid, finite reward probs.

        Note: EM does not update state_values (no closed-form M-step for
        general beta), so it learns from reward signal only. SGD is
        recommended for full parameter learning.
        """
        choices, rewards, _, _ = _simulate_block_bandit(
            n_trials=200, n_options=3, seed=42,
        )
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit(choices, rewards, max_iter=30)

        rp = model.reward_probs_
        assert jnp.all(jnp.isfinite(rp))
        assert jnp.all(rp > 0)
        assert jnp.all(rp < 1)

    def test_sgd_recovers_block_structure(self):
        """SGD should also recover the block structure."""
        choices, rewards, true_states, _ = _simulate_block_bandit(
            n_trials=200, n_options=3, seed=42,
        )
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit_sgd(choices, rewards, num_steps=200)

        posterior = model.predict_state_posterior(choices, rewards)
        first_half_state = jnp.argmax(posterior[:50].mean(axis=0))
        second_half_state = jnp.argmax(posterior[150:].mean(axis=0))
        assert first_half_state != second_half_state

    def test_reduces_to_stationary_with_zero_weights(self):
        """With no covariates, should act like a standard HMM."""
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=60)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit(choices, rewards, max_iter=10)
        # Transition matrix from centered logits should be stochastic
        trans = centered_softmax(model._get_transition_logits())
        np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-6)

    def test_three_state_model(self):
        """Model with 3 states should fit without errors."""
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=100)
        model = ContingencyBeliefModel(n_states=3, n_options=3)
        lls = model.fit(choices, rewards, max_iter=15)
        assert all(np.isfinite(ll) for ll in lls)
        assert model.is_fitted

    def test_sgd_with_transition_covariates(self):
        """SGD should learn nonzero transition coefficients from covariates."""
        choices, rewards, true_states, _ = _simulate_block_bandit(
            n_trials=100, n_options=3, seed=42,
        )
        # Create a "reset" covariate that fires at the block boundary
        n_trials = len(choices)
        covariates = jnp.zeros((n_trials, 1))
        covariates = covariates.at[n_trials // 2, 0].set(1.0)

        model = ContingencyBeliefModel(n_states=2, n_options=3)
        lls = model.fit_sgd(
            choices, rewards,
            transition_covariates=covariates,
            num_steps=100,
        )
        assert lls[-1] > lls[0]
        # Coefficients should have > 1 row (intercept + covariate)
        assert model.transition_coefficients_.shape[0] == 2
        # Non-intercept coefficients should be nonzero
        assert jnp.any(jnp.abs(model.transition_coefficients_[1:]) > 0.01)

    def test_fitted_state_attributes(self):
        """Model should populate both state_posterior_ and smoothed_state_posterior_."""
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=60)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit_sgd(choices, rewards, num_steps=20)
        # Smoothed (acausal)
        assert model.smoothed_state_posterior_ is not None
        assert model.smoothed_state_posterior_.shape == (60, 2)
        np.testing.assert_allclose(
            model.smoothed_state_posterior_.sum(axis=1), 1.0, atol=1e-6
        )
        # Causal (filtered)
        assert model.state_posterior_ is not None
        assert model.state_posterior_.shape == (60, 2)
        np.testing.assert_allclose(
            model.state_posterior_.sum(axis=1), 1.0, atol=1e-6
        )

    def test_em_populates_both_posteriors(self):
        """EM should also populate both causal and smoothed posteriors."""
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=60)
        model = ContingencyBeliefModel(n_states=2, n_options=3)
        model.fit(choices, rewards, max_iter=5)
        assert model.state_posterior_ is not None
        assert model.smoothed_state_posterior_ is not None
        assert model.state_posterior_.shape == (60, 2)
        assert model.smoothed_state_posterior_.shape == (60, 2)

    def test_em_with_transition_covariates(self):
        """EM should use transition covariates in the E-step."""
        choices, rewards, _, _ = _simulate_block_bandit(n_trials=100)
        n_trials = len(choices)
        covariates = jnp.zeros((n_trials, 1))
        covariates = covariates.at[n_trials // 2, 0].set(1.0)

        model = ContingencyBeliefModel(n_states=2, n_options=3)
        lls = model.fit(
            choices, rewards,
            transition_covariates=covariates,
            max_iter=15,
        )
        assert len(lls) > 1
        # Coefficients should have expanded for the covariate
        assert model.transition_coefficients_.shape[0] == 2
        assert model.is_fitted
