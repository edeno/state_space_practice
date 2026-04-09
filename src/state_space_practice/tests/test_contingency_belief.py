# ruff: noqa: E402
"""Tests for the contingency_belief module."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.contingency_belief import (
    ContingencyBeliefResult,
    compute_input_output_transition_matrix,
    compute_reward_log_likelihood,
    contingency_belief_filter,
    contingency_belief_smoother,
    transition_logits_to_matrix,
)


class TestTransitionLogitsToMatrix:
    def test_rows_sum_to_one(self):
        logits = jnp.zeros((3, 3))
        trans = transition_logits_to_matrix(logits)
        np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-7)

    def test_uniform_for_zero_logits(self):
        logits = jnp.zeros((2, 2))
        trans = transition_logits_to_matrix(logits)
        np.testing.assert_allclose(trans, 0.5, atol=1e-7)

    def test_higher_logit_gives_higher_prob(self):
        logits = jnp.array([[5.0, 0.0], [0.0, 5.0]])
        trans = transition_logits_to_matrix(logits)
        assert trans[0, 0] > trans[0, 1]
        assert trans[1, 1] > trans[1, 0]

    def test_gradient_finite(self):
        logits = jnp.array([[1.0, -1.0], [-0.5, 0.5]])
        g = jax.grad(lambda l: transition_logits_to_matrix(l).sum())(logits)
        assert jnp.all(jnp.isfinite(g))


class TestInputOutputTransitionMatrix:
    def test_zero_covariates_recover_baseline(self):
        baseline = jnp.array([[2.0, 0.0], [0.0, 2.0]])
        weights = jnp.zeros((2, 2, 3))
        h_t = jnp.zeros(3)
        trans = compute_input_output_transition_matrix(baseline, weights, h_t)
        expected = jax.nn.softmax(baseline, axis=1)
        np.testing.assert_allclose(trans, expected, atol=1e-7)

    def test_rows_sum_to_one(self):
        baseline = jnp.ones((3, 3))
        weights = jax.random.normal(jax.random.PRNGKey(0), (3, 3, 2))
        h_t = jnp.array([1.0, -1.0])
        trans = compute_input_output_transition_matrix(baseline, weights, h_t)
        np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-7)

    def test_covariates_shift_transitions(self):
        baseline = jnp.zeros((2, 2))
        weights = jnp.zeros((2, 2, 1))
        weights = weights.at[0, 1, 0].set(5.0)  # covariate pushes s0→s1
        h_t = jnp.array([1.0])
        trans = compute_input_output_transition_matrix(baseline, weights, h_t)
        # s0→s1 should be higher than s0→s0
        assert trans[0, 1] > trans[0, 0]


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
            transition_logits=jnp.array([[3.0, 0.0], [0.0, 3.0]]),
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
            transition_logits=jnp.zeros((3, 3)),
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
            transition_logits=jnp.array([[2.0, -1.0], [-1.0, 2.0]]),
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
            transition_logits=jnp.zeros((2, 2)),
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
            transition_logits=jnp.array([[2.0, -1.0], [-1.0, 2.0]]),
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
