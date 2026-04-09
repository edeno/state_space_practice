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
    change_point_probability,
    compute_surprise,
    option_variances_from_covariances,
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


class TestChangePointProbability:
    def test_zero_for_certain_state(self):
        probs = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        cp = change_point_probability(probs)
        np.testing.assert_allclose(cp, 0.0, atol=1e-8)

    def test_high_for_uncertain_state(self):
        probs = jnp.array([[0.5, 0.5]])
        cp = change_point_probability(probs)
        np.testing.assert_allclose(cp, 0.5, atol=1e-8)


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
