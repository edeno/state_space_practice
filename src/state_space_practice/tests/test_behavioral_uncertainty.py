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
        cov = jnp.array(
            [
                [[0.2, 0.0], [0.0, 0.5]],
                [[0.1, 0.0], [0.0, 0.3]],
            ]
        )
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


# ---------------------------------------------------------------------------
# Closed forms vs scipy.stats / analytic values (Hypothesis)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import arrays  # noqa: E402
from scipy import stats  # noqa: E402


@st.composite
def _simplex_rows(draw, n_rows: int = 6, ks=(2, 3, 8), allow_zeros: bool = True):
    """Rows on the probability simplex, optionally with exact zeros.

    Shapes come from a small fixed set: every new shape re-traces the eager
    JAX ops (~0.3 s), which would dominate these tests.
    """
    k = draw(st.sampled_from(ks))
    raw = draw(arrays(np.float64, (n_rows, k), elements=st.floats(0.0, 1.0)))
    if allow_zeros:
        zero_mask = draw(arrays(np.bool_, (n_rows, k)))
        raw = np.where(zero_mask, 0.0, raw)
    raw[:, 0] += 1e-3  # keep every row normalisable
    return raw / raw.sum(axis=1, keepdims=True)


class TestEntropyOracle:
    @settings(deadline=None, max_examples=60)
    @given(probs=_simplex_rows())
    def test_matches_scipy_entropy(self, probs):
        ours = np.asarray(categorical_entropy(jnp.asarray(probs)))
        np.testing.assert_allclose(ours, stats.entropy(probs, axis=-1), atol=1e-14)
        np.testing.assert_allclose(
            np.asarray(belief_entropy(jnp.asarray(probs))), ours, atol=0
        )
        assert np.all(ours <= np.log(probs.shape[-1]) + 1e-12)

    @settings(deadline=None, max_examples=30)
    @given(k=st.integers(1, 64))
    def test_uniform_on_support_is_log_support_size(self, k):
        """Padding with impossible options leaves the entropy exactly log k.

        (The previous clip-at-1e-10 version added 2.3e-9 nats per zero.)
        """
        probs = np.concatenate([np.full(k, 1.0 / k), np.zeros(64 - k)])
        ent = float(categorical_entropy(jnp.asarray(probs)))
        assert abs(ent - np.log(k)) < 1e-14
        if k == 1:
            assert ent == 0.0

    def test_gradient_is_finite_at_zero_probabilities(self):
        grad = jax.grad(lambda p: categorical_entropy(p))(jnp.array([0.7, 0.3, 0.0]))
        assert np.all(np.isfinite(np.asarray(grad)))
        np.testing.assert_allclose(
            np.asarray(grad[:2]), -(np.log([0.7, 0.3]) + 1.0), rtol=1e-12
        )


class TestSurpriseOracle:
    @settings(deadline=None, max_examples=25)
    @given(probs=_simplex_rows(allow_zeros=False), seed=st.integers(0, 99))
    def test_is_negative_log_pmf(self, probs, seed):
        choices = np.random.default_rng(seed).integers(0, probs.shape[1], len(probs))
        expected = -np.array(
            [
                stats.rv_discrete(values=(np.arange(p.size), p)).logpmf(c)
                for p, c in zip(probs, choices)
            ]
        )
        ours = np.asarray(compute_surprise(jnp.asarray(probs), jnp.asarray(choices)))
        floor = -np.log(1e-10)
        np.testing.assert_allclose(ours, np.minimum(expected, floor), rtol=1e-12)

    def test_impossible_choice_is_capped(self):
        surp = compute_surprise(jnp.array([[1.0, 0.0]]), jnp.array([1]))
        assert float(surp[0]) == pytest.approx(-np.log(1e-10))


def _legacy_mixture_variance(state_probs, reward_probs):
    mean = state_probs @ reward_probs
    bernoulli_var = state_probs @ (reward_probs * (1 - reward_probs))
    mixture_var = state_probs @ (reward_probs**2) - mean**2
    return bernoulli_var + mixture_var


class TestBernoulliMixtureOracle:
    @settings(deadline=None, max_examples=60)
    @given(
        state_probs=_simplex_rows(n_rows=5, ks=(2, 5)),
        data=st.data(),
    )
    def test_is_the_bernoulli_marginal(self, state_probs, data):
        """mean = P(r = 1); variance = mean (1 - mean) = scipy bernoulli.var."""
        n_states = state_probs.shape[1]
        reward_probs = data.draw(
            arrays(np.float64, (n_states, 3), elements=st.floats(0.0, 1.0))
        )
        mean, var = bernoulli_mixture_mean_variance(
            jnp.asarray(state_probs), jnp.asarray(reward_probs)
        )
        mean, var = np.asarray(mean), np.asarray(var)
        # brute force: P(r = 1) = sum_s P(s) rho_s
        brute_mean = np.einsum("ts,sk->tk", state_probs, reward_probs)
        np.testing.assert_allclose(mean, brute_mean, atol=1e-15)
        np.testing.assert_allclose(
            var, stats.bernoulli(np.clip(brute_mean, 0.0, 1.0)).var(), atol=1e-15
        )
        # identical (to round-off) to the previous two-term implementation
        np.testing.assert_allclose(
            var, _legacy_mixture_variance(state_probs, reward_probs), atol=1e-14
        )
        assert np.all(var >= 0.0)

    def test_mixture_adds_between_state_variance(self):
        """Two certain-but-opposite states: Bernoulli part 0, total 1/4."""
        mean, var = bernoulli_mixture_mean_variance(
            jnp.array([[0.5, 0.5]]), jnp.array([[1.0], [0.0]])
        )
        assert float(mean[0, 0]) == 0.5
        assert float(var[0, 0]) == 0.25


class TestChangePointOracle:
    @settings(deadline=None, max_examples=40)
    @given(
        marginals=_simplex_rows(n_rows=7, ks=(2, 4), allow_zeros=False),
        data=st.data(),
    )
    def test_equals_markov_chain_switch_probability(self, marginals, data):
        """For P(s_t = i, s_{t+1} = j) = pi_t(i) Z_ij, P(switch) = 1 - sum pi Z_ii."""
        n_states = marginals.shape[1]
        raw = data.draw(
            arrays(np.float64, (n_states, n_states), elements=st.floats(0.01, 1.0))
        )
        transition = raw / raw.sum(axis=1, keepdims=True)
        pairwise = marginals[:, :, None] * transition[None, :, :]
        cp = np.asarray(pairwise_change_point_probability(jnp.asarray(pairwise)))
        expected = 1.0 - marginals @ np.diag(transition)
        assert cp[0] == 0.0
        np.testing.assert_allclose(cp[1:], expected, atol=1e-14)
        # brute force over all (i, j) pairs with i != j
        brute = [
            sum(
                pairwise[t, i, j]
                for i in range(n_states)
                for j in range(n_states)
                if i != j
            )
            for t in range(len(pairwise))
        ]
        np.testing.assert_allclose(cp[1:], brute, atol=1e-14)
