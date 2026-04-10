# ruff: noqa: E402
"""Tests for behavioral uncertainty helpers."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from state_space_practice.behavioral_uncertainty import (
    append_reference_option,
    belief_entropy,
    bernoulli_mixture_mean_variance,
    categorical_entropy,
    compute_surprise,
    option_variances_from_covariances,
    pairwise_change_point_probability,
)


class TestAppendReferenceOption:
    def test_prepends_zero_column(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        out = append_reference_option(x)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[:, 0], 0.0)
        np.testing.assert_allclose(out[:, 1:], x)

    def test_1d(self):
        x = jnp.array([1.0, 2.0])
        out = append_reference_option(x)
        assert out.shape == (3,)
        assert float(out[0]) == 0.0


class TestOptionVariances:
    def test_adds_reference_zero(self):
        cov = jnp.array([
            [[0.2, 0.0], [0.0, 0.5]],
            [[0.1, 0.0], [0.0, 0.3]],
        ])
        out = option_variances_from_covariances(cov)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[:, 0], 0.0)
        np.testing.assert_allclose(out[:, 1:], jnp.array([[0.2, 0.5], [0.1, 0.3]]))

    def test_single_matrix(self):
        cov = jnp.eye(2) * 0.3
        out = option_variances_from_covariances(cov)
        assert out.shape == (3,)
        np.testing.assert_allclose(out, [0.0, 0.3, 0.3])


class TestCategoricalEntropy:
    def test_uniform(self):
        probs = jnp.array([[0.25, 0.25, 0.25, 0.25]])
        ent = categorical_entropy(probs)
        np.testing.assert_allclose(ent, np.log(4.0), atol=1e-6)

    def test_deterministic_is_zero(self):
        probs = jnp.array([[1.0, 0.0, 0.0]])
        ent = categorical_entropy(probs)
        np.testing.assert_allclose(ent, 0.0, atol=1e-6)

    def test_batch(self):
        probs = jnp.array([[0.5, 0.5], [1.0, 0.0]])
        ent = categorical_entropy(probs)
        assert ent.shape == (2,)
        assert float(ent[0]) > float(ent[1])


class TestBeliefEntropy:
    def test_zero_for_certain_state(self):
        probs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        ent = belief_entropy(probs)
        np.testing.assert_allclose(ent, 0.0, atol=1e-8)


class TestComputeSurprise:
    def test_high_for_unlikely_choice(self):
        probs = jnp.array([[0.9, 0.05, 0.05]])
        choices = jnp.array([2])
        surp = compute_surprise(probs, choices)
        assert float(surp[0]) > 1.0  # -log(0.05) ≈ 3.0

    def test_low_for_likely_choice(self):
        probs = jnp.array([[0.9, 0.05, 0.05]])
        choices = jnp.array([0])
        surp = compute_surprise(probs, choices)
        assert float(surp[0]) < 0.2  # -log(0.9) ≈ 0.1

    def test_shape(self):
        probs = jnp.ones((50, 3)) / 3
        choices = jnp.zeros(50, dtype=jnp.int32)
        surp = compute_surprise(probs, choices)
        assert surp.shape == (50,)


class TestPairwiseChangePointProbability:
    def test_zero_when_no_switches(self):
        # Pairwise joint with all mass on the diagonal → no switches
        n_states = 2
        T_minus_1 = 5
        pairwise = jnp.zeros((T_minus_1, n_states, n_states))
        pairwise = pairwise.at[:, 0, 0].set(1.0)  # always stay in state 0
        cp = pairwise_change_point_probability(pairwise)
        assert cp.shape == (T_minus_1 + 1,)
        np.testing.assert_allclose(cp, 0.0, atol=1e-8)

    def test_one_when_all_switch(self):
        # Pairwise joint with all mass off-diagonal → always switching
        n_states = 2
        T_minus_1 = 4
        pairwise = jnp.zeros((T_minus_1, n_states, n_states))
        pairwise = pairwise.at[:, 0, 1].set(0.5)
        pairwise = pairwise.at[:, 1, 0].set(0.5)
        cp = pairwise_change_point_probability(pairwise)
        # First entry is 0, rest are 1
        np.testing.assert_allclose(cp[0], 0.0)
        np.testing.assert_allclose(cp[1:], 1.0, atol=1e-8)

    def test_spike_at_block_boundary(self):
        """Real end-to-end: contingency smoother + pairwise cp on block data."""
        from state_space_practice.contingency_belief import (
            contingency_belief_smoother,
        )

        # Strong block structure: reward on option 0 first, then option 1
        choices = jnp.array([0] * 20 + [1] * 20, dtype=jnp.int32)
        rewards = jnp.array([1] * 20 + [1] * 20, dtype=jnp.int32)
        result = contingency_belief_smoother(
            choices=choices,
            rewards=rewards,
            n_states=2,
            n_options=2,
            reward_probs=jnp.array([[0.9, 0.1], [0.1, 0.9]]),
            state_values=jnp.array([[2.0, 0.0], [0.0, 2.0]]),
            inverse_temperature=2.0,
            transition_logits=jnp.array([[3.0], [-3.0]]),  # sticky
        )
        cp = pairwise_change_point_probability(result.pairwise_state_prob)
        # Total switch mass should concentrate near the block boundary (t=20)
        boundary_region = cp[15:25].sum()
        early_region = cp[1:15].sum()
        late_region = cp[25:].sum()
        assert float(boundary_region) > float(early_region)
        assert float(boundary_region) > float(late_region)

    def test_trial_zero_is_zero(self):
        pairwise = jnp.array([[[0.3, 0.2], [0.2, 0.3]]])  # (1, 2, 2)
        cp = pairwise_change_point_probability(pairwise)
        assert cp.shape == (2,)
        np.testing.assert_allclose(cp[0], 0.0)


class TestBernoulliMixture:
    def test_shapes(self):
        state_probs = jnp.array([[0.7, 0.3], [0.1, 0.9]])
        reward_probs = jnp.array([[0.8, 0.2, 0.1], [0.2, 0.4, 0.9]])
        mean, var = bernoulli_mixture_mean_variance(state_probs, reward_probs)
        assert mean.shape == (2, 3)
        assert var.shape == (2, 3)

    def test_variance_nonnegative(self):
        state_probs = jnp.array([[0.7, 0.3], [0.1, 0.9]])
        reward_probs = jnp.array([[0.8, 0.2, 0.1], [0.2, 0.4, 0.9]])
        _, var = bernoulli_mixture_mean_variance(state_probs, reward_probs)
        assert jnp.all(var >= -1e-10)

    def test_certain_state_gives_bernoulli_variance(self):
        state_probs = jnp.array([[1.0, 0.0]])
        reward_probs = jnp.array([[0.8, 0.2], [0.3, 0.7]])
        mean, var = bernoulli_mixture_mean_variance(state_probs, reward_probs)
        np.testing.assert_allclose(mean, [[0.8, 0.2]], atol=1e-6)
        # Bernoulli variance: p(1-p)
        np.testing.assert_allclose(var, [[0.8 * 0.2, 0.2 * 0.8]], atol=1e-6)
