"""Tests for multinomial choice learning model."""

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.multinomial_choice import (
    ChoiceFilterResult,
    ChoiceSmootherResult,
    MultinomialChoiceModel,
    multinomial_choice_filter,
    multinomial_choice_smoother,
    simulate_choice_data,
    softmax_observation_update,
)


class TestSoftmaxObservationUpdate:
    def test_output_shapes(self):
        """4 options -> 3 free params."""
        prior_mean = jnp.zeros(3)
        prior_cov = jnp.eye(3)
        post_mean, post_cov, ll = softmax_observation_update(
            prior_mean, prior_cov, choice=2, n_options=4,
        )
        assert post_mean.shape == (3,)
        assert post_cov.shape == (3, 3)
        assert ll.shape == ()

    def test_chosen_option_value_increases(self):
        """Choose option 1 (free param index 0) -> value should increase."""
        prior_mean = jnp.zeros(2)
        prior_cov = jnp.eye(2)
        post_mean, _, _ = softmax_observation_update(
            prior_mean, prior_cov, choice=1, n_options=3,
        )
        assert post_mean[0] > prior_mean[0]

    def test_unchosen_option_value_decreases(self):
        """Choose option 1 -> free param for option 2 should decrease."""
        prior_mean = jnp.zeros(2)
        prior_cov = jnp.eye(2)
        post_mean, _, _ = softmax_observation_update(
            prior_mean, prior_cov, choice=1, n_options=3,
        )
        assert post_mean[1] < prior_mean[1] or np.isclose(post_mean[1], prior_mean[1], atol=1e-6)

    def test_reference_option_choice_decreases_all(self):
        """Choose option 0 (reference) -> all free values should decrease."""
        prior_mean = jnp.ones(2) * 0.5  # Start slightly positive
        prior_cov = jnp.eye(2)
        post_mean, _, _ = softmax_observation_update(
            prior_mean, prior_cov, choice=0, n_options=3,
        )
        assert jnp.all(post_mean < prior_mean)

    def test_high_temperature_weak_update(self):
        """Low beta should produce smaller update than high beta."""
        prior_mean = jnp.zeros(2)
        prior_cov = jnp.eye(2)
        post_low, _, _ = softmax_observation_update(
            prior_mean, prior_cov, choice=1, n_options=3,
            inverse_temperature=0.1,
        )
        post_high, _, _ = softmax_observation_update(
            prior_mean, prior_cov, choice=1, n_options=3,
            inverse_temperature=5.0,
        )
        update_low = jnp.linalg.norm(post_low - prior_mean)
        update_high = jnp.linalg.norm(post_high - prior_mean)
        assert update_high > update_low

    def test_posterior_covariance_shrinks(self):
        """Posterior covariance should be smaller than prior."""
        prior_mean = jnp.zeros(3)
        prior_cov = jnp.eye(3)
        _, post_cov, _ = softmax_observation_update(
            prior_mean, prior_cov, choice=1, n_options=4,
        )
        assert jnp.trace(post_cov) < jnp.trace(prior_cov)

    def test_log_likelihood_is_finite(self):
        """Log-likelihood should be finite and negative."""
        prior_mean = jnp.zeros(2)
        prior_cov = jnp.eye(2)
        _, _, ll = softmax_observation_update(
            prior_mean, prior_cov, choice=1, n_options=3,
        )
        assert jnp.isfinite(ll)
        assert ll < 0

    def test_posterior_covariance_is_psd(self):
        """All eigenvalues should be >= 0."""
        prior_mean = jnp.zeros(3)
        prior_cov = jnp.eye(3)
        _, post_cov, _ = softmax_observation_update(
            prior_mean, prior_cov, choice=2, n_options=4,
        )
        eigvals = jnp.linalg.eigvalsh(post_cov)
        assert jnp.all(eigvals >= -1e-8)

    def test_invalid_choice_raises(self):
        """choice >= n_options or choice < 0 should raise."""
        prior_mean = jnp.zeros(2)
        prior_cov = jnp.eye(2)
        with pytest.raises(ValueError):
            softmax_observation_update(prior_mean, prior_cov, choice=3, n_options=3)
        with pytest.raises(ValueError):
            softmax_observation_update(prior_mean, prior_cov, choice=-1, n_options=3)


class TestMultinomialChoiceFilter:
    def test_output_shapes(self):
        """100 trials, 4 options -> filtered_values shape (100, 3)."""
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 4, size=100)
        result = multinomial_choice_filter(choices, n_options=4)
        assert isinstance(result, ChoiceFilterResult)
        assert result.filtered_values.shape == (100, 3)
        assert result.filtered_covariances.shape == (100, 3, 3)
        assert result.predicted_values.shape == (100, 3)
        assert result.predicted_covariances.shape == (100, 3, 3)

    def test_preferred_option_has_highest_value(self):
        """Option 1 chosen 80% -> its value should be highest."""
        rng = np.random.default_rng(42)
        n_trials = 200
        choices = np.where(rng.random(n_trials) < 0.8, 1, 2)
        result = multinomial_choice_filter(
            choices, n_options=3, inverse_temperature=1.0,
        )
        final_values = result.filtered_values[-1]
        # Option 1 = free param index 0, option 2 = free param index 1
        assert final_values[0] > final_values[1]

    def test_switching_preference_tracked(self):
        """First 100 trials: option 1, next 100: option 2."""
        choices = np.concatenate([
            np.ones(100, dtype=int),   # option 1
            np.full(100, 2, dtype=int),  # option 2
        ])
        result = multinomial_choice_filter(
            choices, n_options=3, process_noise=0.05,
        )
        # At trial 50: value[0] > value[1]
        assert result.filtered_values[49, 0] > result.filtered_values[49, 1]
        # At trial 180: value[1] > value[0]
        assert result.filtered_values[179, 1] > result.filtered_values[179, 0]

    def test_marginal_ll_is_finite(self):
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 3, size=50)
        result = multinomial_choice_filter(choices, n_options=3)
        assert jnp.isfinite(result.marginal_log_likelihood)
        assert result.marginal_log_likelihood < 0


class TestMultinomialChoiceSmoother:
    def test_output_shapes(self):
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 4, size=100)
        result = multinomial_choice_smoother(choices, n_options=4)
        assert isinstance(result, ChoiceSmootherResult)
        assert result.smoothed_values.shape == (100, 3)
        assert result.smoothed_covariances.shape == (100, 3, 3)
        assert result.smoother_cross_cov.shape == (99, 3, 3)

    def test_smoother_reduces_variance(self):
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 3, size=100)
        filt = multinomial_choice_filter(choices, n_options=3)
        smooth = multinomial_choice_smoother(choices, n_options=3)
        filt_var = np.mean([np.trace(c) for c in np.array(filt.filtered_covariances)])
        smooth_var = np.mean([np.trace(c) for c in np.array(smooth.smoothed_covariances)])
        assert smooth_var <= filt_var * 1.01

    def test_last_trial_matches_filter(self):
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 3, size=50)
        filt = multinomial_choice_filter(choices, n_options=3)
        smooth = multinomial_choice_smoother(choices, n_options=3)
        np.testing.assert_allclose(
            smooth.smoothed_values[-1],
            filt.filtered_values[-1],
            atol=1e-5,
        )

    def test_smoother_cross_cov_shape(self):
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 4, size=80)
        result = multinomial_choice_smoother(choices, n_options=4)
        assert result.smoother_cross_cov.shape == (79, 3, 3)


class TestMultinomialChoiceModel:
    def test_init_and_repr(self):
        model = MultinomialChoiceModel(n_options=4)
        r = repr(model)
        assert "n_options=4" in r
        assert "fitted=False" in r

    def test_fit_returns_log_likelihoods(self):
        rng = np.random.default_rng(42)
        choices = np.where(rng.random(300) < 0.7, 1, rng.integers(0, 4, size=300))
        model = MultinomialChoiceModel(n_options=4)
        lls = model.fit(choices, max_iter=5)
        assert isinstance(lls, list)
        assert len(lls) <= 5
        assert all(np.isfinite(ll) for ll in lls)

    def test_is_fitted(self):
        model = MultinomialChoiceModel(n_options=3)
        assert not model.is_fitted
        rng = np.random.default_rng(42)
        model.fit(rng.integers(0, 3, size=50), max_iter=2)
        assert model.is_fitted

    def test_fit_learns_from_deterministic_choices(self):
        """Always choose option 1 -> high inverse_temperature."""
        choices = np.ones(200, dtype=int)
        model = MultinomialChoiceModel(n_options=3)
        model.fit(choices, max_iter=10)
        assert model.inverse_temperature > 2.0

    def test_fit_learns_process_noise(self):
        """Switching preferences should yield higher Q than stable."""
        stable = np.ones(200, dtype=int)
        switching = np.concatenate([np.ones(100, dtype=int), np.full(100, 2, dtype=int)])

        model_stable = MultinomialChoiceModel(n_options=3)
        model_stable.fit(stable, max_iter=10)

        model_switch = MultinomialChoiceModel(n_options=3)
        model_switch.fit(switching, max_iter=10)

        assert model_switch.process_noise > model_stable.process_noise

    def test_fit_verbose(self, caplog):
        import logging
        rng = np.random.default_rng(42)
        model = MultinomialChoiceModel(n_options=3)
        with caplog.at_level(logging.INFO, logger="state_space_practice.multinomial_choice"):
            model.fit(rng.integers(0, 3, size=50), max_iter=2, verbose=True)
        assert "EM iter" in caplog.text

    def test_choice_probabilities(self):
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 4, size=100)
        model = MultinomialChoiceModel(n_options=4)
        model.fit(choices, max_iter=3)
        probs = model.choice_probabilities()
        assert probs.shape == (100, 4)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_bic_is_finite(self):
        rng = np.random.default_rng(42)
        model = MultinomialChoiceModel(n_options=3)
        model.fit(rng.integers(0, 3, size=100), max_iter=3)
        assert np.isfinite(model.bic())

    def test_summary_returns_string(self):
        rng = np.random.default_rng(42)
        model = MultinomialChoiceModel(n_options=3)
        model.fit(rng.integers(0, 3, size=100), max_iter=3)
        s = model.summary()
        assert "MultinomialChoiceModel" in s
        assert "inverse_temperature" in s
        assert "learning_detected" in s

    def test_compare_to_null(self):
        """With biased choices, model should beat null."""
        choices = np.where(np.random.default_rng(42).random(200) < 0.8, 1, 0)
        model = MultinomialChoiceModel(n_options=2)
        model.fit(choices, max_iter=10)
        comparison = model.compare_to_null()
        assert comparison["learning_detected"]
        assert comparison["delta_bic"] > 0

    def test_em_log_likelihood_non_decreasing(self):
        rng = np.random.default_rng(42)
        choices = np.where(rng.random(200) < 0.7, 1, rng.integers(0, 3, size=200))
        model = MultinomialChoiceModel(n_options=3)
        lls = model.fit(choices, max_iter=15)
        # Allow small decreases due to grid discretization
        for i in range(1, len(lls)):
            assert lls[i] >= lls[i - 1] - 0.5, (
                f"LL decreased by {lls[i-1] - lls[i]:.4f} at iter {i}"
            )

    def test_invalid_choice_raises(self):
        model = MultinomialChoiceModel(n_options=3)
        with pytest.raises(ValueError):
            model.fit(np.array([0, 1, 3]))  # 3 is out of range

    def test_single_trial_raises(self):
        model = MultinomialChoiceModel(n_options=3)
        with pytest.raises(ValueError, match="at least 2 trials"):
            model.fit(np.array([1]))

    def test_two_options_consistent_with_smith(self):
        """K=2 multinomial should agree with SmithLearningModel."""
        from state_space_practice.smith_learning_algorithm import SmithLearningModel

        rng = np.random.default_rng(42)
        n_trials = 200
        correct = rng.binomial(1, 0.7, size=n_trials)

        # Smith model with fixed params
        sigma_eps = 0.1  # corresponds to process_noise = sigma_eps^2
        smith = SmithLearningModel(
            sigma_epsilon=sigma_eps,
            prob_correct_by_chance=0.5,
        )
        smith.fit(correct, max_iter=1)  # 1 iter with fixed params

        # Multinomial with same fixed params: choices = correct (option 1 = correct)
        multi = MultinomialChoiceModel(
            n_options=2,
            init_process_noise=sigma_eps**2,
            init_inverse_temperature=1.0,
            learn_process_noise=False,
            learn_inverse_temperature=False,
        )
        multi.fit(correct, max_iter=1)

        # Compare smoothed values
        smith_vals = np.array(smith.smoothed_learning_state_mode).flatten()
        multi_vals = np.array(multi._smoother_result.smoothed_values).flatten()

        corr = np.corrcoef(smith_vals, multi_vals)[0, 1]
        assert corr > 0.9, f"K=2 correlation with Smith: {corr:.3f}"


class TestSimulateChoiceData:
    def test_output_shapes(self):
        data = simulate_choice_data(n_trials=100, n_options=4, seed=42)
        assert data.choices.shape == (100,)
        assert data.true_values.shape == (100, 3)
        assert data.true_probs.shape == (100, 4)

    def test_choices_are_valid(self):
        data = simulate_choice_data(n_trials=200, n_options=4, seed=42)
        assert jnp.all(data.choices >= 0)
        assert jnp.all(data.choices < 4)

    def test_biased_simulation(self):
        """With high beta, the most-chosen option in last trials should
        correspond to the highest-valued option."""
        data = simulate_choice_data(
            n_trials=500, n_options=3,
            process_noise=0.1, inverse_temperature=5.0, seed=42,
        )
        # Find the option with highest final value (including reference at 0)
        final_free_values = np.array(data.true_values[-1])
        full_final = np.concatenate([[0.0], final_free_values])
        best_option = int(np.argmax(full_final))
        # In the last 100 trials, best option should be chosen most
        last_choices = np.array(data.choices[-100:])
        counts = np.bincount(last_choices, minlength=3)
        assert counts[best_option] == counts.max()

    def test_seed_reproducibility(self):
        d1 = simulate_choice_data(seed=42)
        d2 = simulate_choice_data(seed=42)
        np.testing.assert_array_equal(d1.choices, d2.choices)
        np.testing.assert_array_equal(d1.true_values, d2.true_values)


class TestMultinomialChoiceIntegration:
    """End-to-end: simulate data, fit model, check recovery."""

    def test_recovers_simulated_values(self):
        """Smoothed values should track true values on simulated data."""
        data = simulate_choice_data(
            n_trials=300, n_options=4,
            process_noise=0.05, inverse_temperature=2.0, seed=42,
        )
        model = MultinomialChoiceModel(n_options=4)
        model.fit(np.array(data.choices), max_iter=15)

        smoothed = np.array(model.smoothed_values)
        true_vals = np.array(data.true_values)

        # Check correlation per option (after warmup)
        warmup = 50
        for k in range(3):
            corr = np.corrcoef(smoothed[warmup:, k], true_vals[warmup:, k])[0, 1]
            assert corr > 0.5, (
                f"Option {k+1} correlation {corr:.2f} < 0.5"
            )

    def test_compare_to_null_detects_learning(self):
        """Model should beat null on simulated biased data."""
        data = simulate_choice_data(
            n_trials=200, n_options=3,
            process_noise=0.05, inverse_temperature=3.0, seed=42,
        )
        model = MultinomialChoiceModel(n_options=3)
        model.fit(np.array(data.choices), max_iter=10)
        comparison = model.compare_to_null()
        assert comparison["learning_detected"]

    def test_uniform_random_no_learning(self):
        """Uniform random choices should not detect learning."""
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 4, size=200)
        model = MultinomialChoiceModel(n_options=4)
        model.fit(choices, max_iter=10)
        comparison = model.compare_to_null()
        # With uniform data, delta_bic should be small or negative
        assert comparison["delta_bic"] < 10.0

    def test_all_outputs_finite(self):
        """No NaN/Inf in any output."""
        data = simulate_choice_data(n_trials=200, n_options=3, seed=42)
        model = MultinomialChoiceModel(n_options=3)
        model.fit(np.array(data.choices), max_iter=5)

        assert np.isfinite(model.log_likelihood_)
        assert np.isfinite(model.bic())
        assert np.all(np.isfinite(np.array(model.smoothed_values)))
        assert np.all(np.isfinite(np.array(model.choice_probabilities())))


class TestMultinomialChoiceModelPlotting:
    @pytest.fixture
    def fitted_model(self):
        import matplotlib
        matplotlib.use("Agg")

        rng = np.random.default_rng(42)
        choices = np.where(rng.random(100) < 0.7, 1, rng.integers(0, 3, size=100))
        model = MultinomialChoiceModel(n_options=3)
        model.fit(choices, max_iter=3)
        return model, choices

    def test_plot_values_returns_fig(self, fitted_model):
        model, choices = fitted_model
        fig, axes = model.plot_values(observed_choices=choices)
        assert fig is not None
        assert len(axes) == 2

    def test_plot_convergence_returns_fig(self, fitted_model):
        model, _ = fitted_model
        fig, ax = model.plot_convergence()
        assert fig is not None

    def test_plot_summary_returns_fig(self, fitted_model):
        model, choices = fitted_model
        fig, axes = model.plot_summary(observed_choices=choices)
        assert fig is not None
        assert len(axes) == 3

    def test_plot_requires_fit(self):
        model = MultinomialChoiceModel(n_options=3)
        with pytest.raises(RuntimeError):
            model.plot_values()


class TestMultinomialSGDFitting:
    """Tests for SGD fitting on MultinomialChoiceModel."""

    def test_sgd_improves_ll(self):
        rng = np.random.default_rng(42)
        choices = np.where(rng.random(200) < 0.7, 1, rng.integers(0, 3, size=200))
        model = MultinomialChoiceModel(n_options=3)
        lls = model.fit_sgd(choices, num_steps=50)
        assert lls[-1] > lls[0]

    def test_sgd_respects_constraints(self):
        rng = np.random.default_rng(42)
        model = MultinomialChoiceModel(n_options=3)
        model.fit_sgd(rng.integers(0, 3, size=100), num_steps=50)
        assert model.process_noise > 0
        assert model.inverse_temperature > 0

    def test_sgd_model_is_fitted(self):
        rng = np.random.default_rng(42)
        model = MultinomialChoiceModel(n_options=3)
        model.fit_sgd(rng.integers(0, 3, size=100), num_steps=20)
        assert model.is_fitted
        assert model.smoothed_values.shape[0] == 100

    def test_sgd_matches_em_approximately(self):
        """SGD and EM should converge to similar LL on same data."""
        rng = np.random.default_rng(42)
        choices = np.where(rng.random(200) < 0.6, 1, rng.integers(0, 3, size=200))

        m_em = MultinomialChoiceModel(n_options=3)
        m_em.fit(choices, max_iter=20)

        m_sgd = MultinomialChoiceModel(n_options=3)
        m_sgd.fit_sgd(choices, num_steps=200)

        # LLs should be within 10% of each other
        assert abs(m_em.log_likelihood_ - m_sgd.log_likelihood_) / abs(m_em.log_likelihood_) < 0.1


class TestMultinomialUncertaintySummaries:
    """Tests for uncertainty summary attributes."""

    def test_uncertainty_populated_after_em(self):
        choices = simulate_choice_data(n_trials=50, n_options=3, seed=0).choices
        model = MultinomialChoiceModel(n_options=3)
        model.fit(choices, max_iter=3)
        assert model.predicted_option_variances_ is not None
        assert model.predicted_option_variances_.shape == (50, 3)
        assert model.smoothed_option_variances_ is not None
        assert model.smoothed_option_variances_.shape == (50, 3)
        assert model.predicted_choice_entropy_ is not None
        assert model.predicted_choice_entropy_.shape == (50,)
        assert model.surprise_ is not None
        assert model.surprise_.shape == (50,)

    def test_uncertainty_populated_after_sgd(self):
        choices = simulate_choice_data(n_trials=50, n_options=3, seed=0).choices
        model = MultinomialChoiceModel(n_options=3)
        model.fit_sgd(choices, num_steps=10)
        assert model.predicted_option_variances_ is not None
        assert model.surprise_ is not None

    def test_reference_option_variance_is_zero(self):
        choices = simulate_choice_data(n_trials=50, n_options=3, seed=0).choices
        model = MultinomialChoiceModel(n_options=3)
        model.fit(choices, max_iter=3)
        # Reference option (index 0) should have zero variance
        np.testing.assert_allclose(model.predicted_option_variances_[:, 0], 0.0)
        np.testing.assert_allclose(model.smoothed_option_variances_[:, 0], 0.0)

    def test_option_values_populated(self):
        choices = simulate_choice_data(n_trials=50, n_options=3, seed=0).choices
        model = MultinomialChoiceModel(n_options=3)
        model.fit(choices, max_iter=3)
        assert model.predicted_option_values_.shape == (50, 3)
        assert model.filtered_option_values_.shape == (50, 3)
        assert model.smoothed_option_values_.shape == (50, 3)
        assert model.filtered_option_variances_.shape == (50, 3)
        # Reference option value should be zero
        np.testing.assert_allclose(model.predicted_option_values_[:, 0], 0.0)
        np.testing.assert_allclose(model.filtered_option_values_[:, 0], 0.0)
        np.testing.assert_allclose(model.smoothed_option_values_[:, 0], 0.0)
        # Filtered variance reference option is zero
        np.testing.assert_allclose(model.filtered_option_variances_[:, 0], 0.0)

    def test_option_values_populated_after_sgd(self):
        choices = simulate_choice_data(n_trials=50, n_options=3, seed=0).choices
        model = MultinomialChoiceModel(n_options=3)
        model.fit_sgd(choices, num_steps=10)
        assert model.predicted_option_values_.shape == (50, 3)
        assert model.filtered_option_values_.shape == (50, 3)
        assert model.smoothed_option_values_.shape == (50, 3)
        assert model.filtered_option_variances_.shape == (50, 3)

    def test_surprise_is_positive(self):
        choices = simulate_choice_data(n_trials=50, n_options=3, seed=0).choices
        model = MultinomialChoiceModel(n_options=3)
        model.fit(choices, max_iter=3)
        assert jnp.all(model.surprise_ >= 0)
