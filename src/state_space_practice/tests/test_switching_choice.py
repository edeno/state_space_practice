# ruff: noqa: E402
"""Tests for the switching_choice module."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.switching_choice import (
    _softmax_predict_and_update,
    _softmax_update_per_state_pair,
    switching_choice_filter,
)


class TestSoftmaxPredictAndUpdate:
    """Tests for the single-pair predict + softmax update."""

    def test_output_shapes(self):
        k_free = 2  # K=3 options
        mean = jnp.zeros(k_free)
        cov = jnp.eye(k_free)
        A = 0.95 * jnp.eye(k_free)
        Q = 0.01 * jnp.eye(k_free)
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        post_mean, post_cov, ll = _softmax_predict_and_update(
            mean, cov, jnp.int32(0), A, Q, 3, 1.0, B, u, obs_offset,
        )
        assert post_mean.shape == (k_free,)
        assert post_cov.shape == (k_free, k_free)
        assert ll.shape == ()

    def test_log_likelihood_finite(self):
        k_free = 2
        mean = jnp.array([0.5, -0.3])
        cov = jnp.eye(k_free) * 0.5
        A = 0.9 * jnp.eye(k_free)
        Q = 0.01 * jnp.eye(k_free)
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        _, _, ll = _softmax_predict_and_update(
            mean, cov, jnp.int32(1), A, Q, 3, 2.0, B, u, obs_offset,
        )
        assert jnp.isfinite(ll)

    def test_covariance_psd(self):
        k_free = 2
        mean = jnp.zeros(k_free)
        cov = jnp.eye(k_free)
        A = 0.95 * jnp.eye(k_free)
        Q = 0.01 * jnp.eye(k_free)
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        _, post_cov, _ = _softmax_predict_and_update(
            mean, cov, jnp.int32(0), A, Q, 3, 1.0, B, u, obs_offset,
        )
        eigvals = jnp.linalg.eigvalsh(post_cov)
        assert jnp.all(eigvals > -1e-10)


class TestSoftmaxUpdatePerStatePair:
    """Tests for the double-vmapped per-state-pair update."""

    def test_output_shapes(self):
        S = 2
        k_free = 2
        mean = jnp.zeros((k_free, S))
        cov = jnp.stack([jnp.eye(k_free)] * S, axis=-1)
        A = jnp.stack([0.95 * jnp.eye(k_free)] * S, axis=-1)
        Q = jnp.stack([0.01 * jnp.eye(k_free)] * S, axis=-1)
        betas = jnp.array([1.0, 3.0])
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        pair_mean, pair_cov, pair_ll = _softmax_update_per_state_pair(
            mean, cov, jnp.int32(0), A, Q, 3, betas, B, u, obs_offset,
        )
        assert pair_mean.shape == (k_free, S, S)
        assert pair_cov.shape == (k_free, k_free, S, S)
        assert pair_ll.shape == (S, S)

    def test_single_state_matches_softmax_update(self):
        """S=1 should match _softmax_predict_and_update directly."""
        k_free = 2
        mean = jnp.zeros((k_free, 1))
        cov = jnp.eye(k_free)[:, :, None]
        A = (0.95 * jnp.eye(k_free))[:, :, None]
        Q = (0.01 * jnp.eye(k_free))[:, :, None]
        betas = jnp.array([2.0])
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        pair_mean, pair_cov, pair_ll = _softmax_update_per_state_pair(
            mean, cov, jnp.int32(1), A, Q, 3, betas, B, u, obs_offset,
        )

        ref_mean, ref_cov, ref_ll = _softmax_predict_and_update(
            mean[:, 0], cov[:, :, 0], jnp.int32(1),
            A[:, :, 0], Q[:, :, 0], 3, 2.0, B, u, obs_offset,
        )

        np.testing.assert_allclose(pair_mean[:, 0, 0], ref_mean, atol=1e-10)
        np.testing.assert_allclose(pair_cov[:, :, 0, 0], ref_cov, atol=1e-10)
        np.testing.assert_allclose(float(pair_ll[0, 0]), float(ref_ll), atol=1e-10)

    def test_different_betas_give_different_posteriors(self):
        S = 2
        k_free = 2
        mean = jnp.zeros((k_free, S))
        cov = jnp.stack([jnp.eye(k_free)] * S, axis=-1)
        A = jnp.stack([0.95 * jnp.eye(k_free)] * S, axis=-1)
        Q = jnp.stack([0.01 * jnp.eye(k_free)] * S, axis=-1)
        betas = jnp.array([0.5, 5.0])  # very different
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        pair_mean, _, _ = _softmax_update_per_state_pair(
            mean, cov, jnp.int32(0), A, Q, 3, betas, B, u, obs_offset,
        )
        # Posteriors should differ across next-state axis
        assert not jnp.allclose(pair_mean[:, 0, 0], pair_mean[:, 0, 1])

    def test_all_log_likelihoods_finite(self):
        S = 2
        k_free = 2
        mean = jnp.zeros((k_free, S))
        cov = jnp.stack([jnp.eye(k_free)] * S, axis=-1)
        A = jnp.stack([0.95 * jnp.eye(k_free)] * S, axis=-1)
        Q = jnp.stack([0.01 * jnp.eye(k_free)] * S, axis=-1)
        betas = jnp.array([1.0, 3.0])
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        _, _, pair_ll = _softmax_update_per_state_pair(
            mean, cov, jnp.int32(0), A, Q, 3, betas, B, u, obs_offset,
        )
        assert jnp.all(jnp.isfinite(pair_ll))


class TestSwitchingChoiceFilter:
    """Tests for the switching choice filter."""

    def test_output_shapes(self):
        n_trials, K, S = 50, 3, 2
        choices = jax.random.randint(jax.random.PRNGKey(0), (n_trials,), 0, K)
        result = switching_choice_filter(
            choices, n_options=K, n_discrete_states=S,
        )
        assert result.filtered_values.shape == (n_trials, K - 1, S)
        assert result.filtered_covs.shape == (n_trials, K - 1, K - 1, S)
        assert result.discrete_state_probs.shape == (n_trials, S)
        assert result.marginal_log_likelihood.shape == ()

    def test_discrete_probs_sum_to_one(self):
        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        result = switching_choice_filter(choices, n_options=3, n_discrete_states=2)
        row_sums = result.discrete_state_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_discrete_probs_nonnegative(self):
        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        result = switching_choice_filter(choices, n_options=3, n_discrete_states=2)
        assert jnp.all(result.discrete_state_probs >= -1e-10)

    def test_marginal_ll_finite(self):
        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        result = switching_choice_filter(choices, n_options=3, n_discrete_states=2)
        assert jnp.isfinite(result.marginal_log_likelihood)

    def test_single_state_matches_covariate_filter(self):
        """S=1 must produce identical filtered values to CovariateChoiceModel.

        Uses nonzero init_mean, nonzero covariates, and decay != 1 to
        exercise the full prediction path including A @ x at trial 0.
        """
        from state_space_practice.covariate_choice import _covariate_choice_filter_jit

        choices = jax.random.randint(jax.random.PRNGKey(42), (100,), 0, 4)
        k_free = 3
        q = 0.01
        beta = 2.0
        decay = 0.6  # nontrivial decay

        init_mean = jnp.array([0.5, -0.3, 0.1])
        init_cov = jnp.eye(k_free) * 0.5
        covariates = jax.random.normal(jax.random.PRNGKey(1), (100, 2)) * 0.1
        input_gain = jax.random.normal(jax.random.PRNGKey(2), (k_free, 2)) * 0.1

        # Switching filter with S=1
        result_sw = switching_choice_filter(
            choices, n_options=4, n_discrete_states=1,
            process_noises=jnp.array([q]),
            inverse_temperatures=jnp.array([beta]),
            decays=jnp.array([decay]),
            init_mean=init_mean,
            init_cov=init_cov,
            covariates=covariates,
            input_gain=input_gain,
        )

        # Non-switching CovariateChoiceModel filter
        result_cov = _covariate_choice_filter_jit(
            choices, 4,
            covariates,
            input_gain,
            jnp.zeros((100, 1)),  # obs_covariates
            jnp.zeros((4, 1)),  # obs_weights
            q, beta, decay,
            init_mean,
            init_cov,
        )

        # Filtered values must match exactly (same Newton steps, same prediction)
        sw_values = result_sw.filtered_values[:, :, 0]  # (T, K-1)
        cov_values = result_cov.filtered_values  # (T, K-1)
        np.testing.assert_allclose(sw_values, cov_values, atol=1e-6)

    def test_two_state_switching_detected(self):
        """First half exploit (deterministic), second half explore (random)."""
        key = jax.random.PRNGKey(0)
        # Exploit phase: always choose 0
        exploit = jnp.zeros(50, dtype=jnp.int32)
        # Explore phase: random
        explore = jax.random.randint(key, (50,), 0, 3)
        choices = jnp.concatenate([exploit, explore])

        result = switching_choice_filter(
            choices, n_options=3, n_discrete_states=2,
            inverse_temperatures=jnp.array([5.0, 0.5]),  # high vs low beta
            process_noises=jnp.array([0.001, 0.05]),
        )
        # The filter should detect some difference in state probs
        # between the two halves
        first_half = result.discrete_state_probs[:25].mean(axis=0)
        second_half = result.discrete_state_probs[75:].mean(axis=0)
        assert not jnp.allclose(first_half, second_half, atol=0.05)


class TestSwitchingChoiceModel:
    """Tests for the SwitchingChoiceModel class."""

    def test_fit_returns_log_likelihoods(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        lls = model.fit(choices, max_iter=5)
        assert len(lls) > 0
        assert all(np.isfinite(ll) for ll in lls)

    def test_is_fitted(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (50,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        assert not model.is_fitted
        model.fit(choices, max_iter=3)
        assert model.is_fitted

    def test_discrete_state_posterior_shape(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (80,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit(choices, max_iter=3)
        assert model.smoothed_discrete_probs_.shape == (80, 2)
        np.testing.assert_allclose(
            model.smoothed_discrete_probs_.sum(axis=1), 1.0, atol=1e-5
        )

    def test_em_does_not_update_betas(self):
        """EM does not update inverse_temperatures (no closed-form M-step).

        Per-state betas are learned via SGD only. EM updates Q and Z.
        """
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        # Start with equal betas
        model = SwitchingChoiceModel(
            n_options=3, n_discrete_states=2,
            init_inverse_temperatures=jnp.array([2.0, 2.0]),
        )
        model.fit(choices, max_iter=5)
        # Betas should be unchanged (EM doesn't update them)
        np.testing.assert_allclose(model.inverse_temperatures_, [2.0, 2.0])

    def test_sgd_learns_different_betas(self):
        """SGD should learn different betas for exploit/explore data."""
        from state_space_practice.switching_choice import (
            SwitchingChoiceModel,
            simulate_switching_choice_data,
        )

        sim = simulate_switching_choice_data(
            n_trials=200, n_options=3,
            inverse_temperatures=jnp.array([5.0, 0.5]),
            process_noises=jnp.array([0.001, 0.05]),
            seed=42,
        )
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(sim.choices, num_steps=100)
        # Per-state betas should differ after SGD
        assert abs(float(model.inverse_temperatures_[0] - model.inverse_temperatures_[1])) > 0.1

    def test_sgd_improves_ll(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        lls = model.fit_sgd(choices, num_steps=30)
        assert lls[-1] > lls[0]

    def test_sgd_model_is_fitted(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (60,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(choices, num_steps=15)
        assert model.is_fitted


class TestSimulateSwitchingChoiceData:
    """Tests for the simulation helper."""

    def test_output_shapes(self):
        from state_space_practice.switching_choice import simulate_switching_choice_data

        sim = simulate_switching_choice_data(n_trials=100, n_options=3, seed=42)
        assert sim.choices.shape == (100,)
        assert sim.true_values.shape == (100, 2)  # K-1
        assert sim.true_states.shape == (100,)
        assert sim.true_probs.shape == (100, 3)

    def test_choices_valid(self):
        from state_space_practice.switching_choice import simulate_switching_choice_data

        sim = simulate_switching_choice_data(n_trials=200, n_options=4, seed=0)
        assert jnp.all(sim.choices >= 0)
        assert jnp.all(sim.choices < 4)

    def test_states_valid(self):
        from state_space_practice.switching_choice import simulate_switching_choice_data

        sim = simulate_switching_choice_data(
            n_trials=200, n_options=3, n_discrete_states=3, seed=0,
            process_noises=jnp.array([0.001, 0.01, 0.05]),
            inverse_temperatures=jnp.array([5.0, 2.0, 0.5]),
        )
        assert jnp.all(sim.true_states >= 0)
        assert jnp.all(sim.true_states < 3)

    def test_seed_reproducibility(self):
        from state_space_practice.switching_choice import simulate_switching_choice_data

        s1 = simulate_switching_choice_data(n_trials=50, n_options=3, seed=42)
        s2 = simulate_switching_choice_data(n_trials=50, n_options=3, seed=42)
        np.testing.assert_array_equal(s1.choices, s2.choices)


class TestModelComparison:
    """Tests for switching vs non-switching model comparison."""

    def test_switching_beats_nonswitching_on_switching_data(self):
        from state_space_practice.covariate_choice import CovariateChoiceModel
        from state_space_practice.switching_choice import (
            SwitchingChoiceModel,
            simulate_switching_choice_data,
        )

        sim = simulate_switching_choice_data(
            n_trials=200, n_options=3, n_discrete_states=2,
            inverse_temperatures=jnp.array([5.0, 0.5]),
            process_noises=jnp.array([0.001, 0.05]),
            seed=42,
        )

        # Switching model
        model_sw = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model_sw.fit_sgd(sim.choices, num_steps=100)

        # Non-switching model
        model_ns = CovariateChoiceModel(n_options=3)
        model_ns.fit_sgd(sim.choices, num_steps=100)

        # Switching model should have better LL on switching data
        assert model_sw.log_likelihood_ > model_ns.log_likelihood_


class TestSwitchingChoiceUncertainty:
    """Tests for switching choice model uncertainty summaries."""

    def test_uncertainty_populated_after_sgd(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (80,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(choices, num_steps=15)
        assert model.predicted_option_variances_ is not None
        assert model.predicted_option_variances_.shape == (80, 3)
        assert model.surprise_ is not None
        assert model.surprise_.shape == (80,)
        assert model.predicted_choice_entropy_ is not None
        assert model.predicted_choice_entropy_.shape == (80,)

    def test_uncertainty_populated_after_em(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (60,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit(choices, max_iter=3)
        assert model.predicted_option_variances_ is not None
        assert model.surprise_ is not None

    def test_surprise_is_positive(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (50,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(choices, num_steps=10)
        assert jnp.all(model.surprise_ >= 0)

    def test_per_state_predicted_variances_shape(self):
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (50,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(choices, num_steps=10)
        assert model.per_state_predicted_variances_ is not None
        assert model.per_state_predicted_variances_.shape == (50, 3, 2)  # (T, K, S)

    def test_predicted_uses_prior_not_posterior(self):
        """Predicted variances should be larger than filtered (prior > posterior)."""
        from state_space_practice.switching_choice import SwitchingChoiceModel

        choices = jax.random.randint(jax.random.PRNGKey(0), (50,), 0, 3)
        model = SwitchingChoiceModel(n_options=3, n_discrete_states=2)
        model.fit_sgd(choices, num_steps=10)
        # Predicted (prior) variance should generally be >= filtered (posterior)
        # because the observation update reduces uncertainty
        pred_var = model.predicted_option_variances_[:, 1:].mean()  # exclude ref
        # filtered_covs are posterior — extract variance
        filt_diag = jnp.diagonal(
            model._filter_result.filtered_covs, axis1=1, axis2=2
        )  # (T, K-1, S)
        filt_var = jnp.einsum(
            "tks,ts->tk", filt_diag, model._filter_result.discrete_state_probs
        ).mean()
        # Prior should be at least as large as posterior on average
        assert float(pred_var) >= float(filt_var) - 1e-6
