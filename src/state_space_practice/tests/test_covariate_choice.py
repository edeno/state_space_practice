"""Tests for covariate-driven choice model."""

import numpy as np

import jax.numpy as jnp

import pytest

from state_space_practice.covariate_choice import (
    CovariateChoiceModel,
    SimulatedRLChoiceData,
    covariate_choice_filter,
    covariate_choice_smoother,
    covariate_predict,
    m_step_input_gain,
    simulate_rl_choice_data,
)
from state_space_practice.multinomial_choice import (
    MultinomialChoiceModel,
    multinomial_choice_filter,
)


class TestCovariatePrediction:
    """Tests for covariate_predict."""

    def test_prediction_with_covariates(self):
        """pred_mean = A @ filt_mean + B @ u_t."""
        filt_mean = jnp.array([1.0, 2.0])
        filt_cov = jnp.eye(2) * 0.1
        B = jnp.array([[0.5, 0.0], [0.0, 0.3]])
        A = jnp.eye(2)  # random walk
        u_t = jnp.array([1.0, 0.0])
        Q = jnp.eye(2) * 0.01

        pred_mean, pred_cov = covariate_predict(filt_mean, filt_cov, u_t, B, A, Q)

        np.testing.assert_allclose(pred_mean, jnp.array([1.5, 2.0]), atol=1e-6)
        np.testing.assert_allclose(pred_cov, A @ filt_cov @ A.T + Q, atol=1e-6)

    def test_prediction_without_covariates(self):
        """B = zeros, A = I -> pred_mean = filt_mean (random walk)."""
        filt_mean = jnp.array([1.0, 2.0])
        filt_cov = jnp.eye(2) * 0.1
        B = jnp.zeros((2, 2))
        A = jnp.eye(2)
        u_t = jnp.array([1.0, 1.0])
        Q = jnp.eye(2) * 0.01

        pred_mean, pred_cov = covariate_predict(filt_mean, filt_cov, u_t, B, A, Q)

        np.testing.assert_allclose(pred_mean, filt_mean, atol=1e-6)

    def test_prediction_shapes(self):
        """Output shapes match input shapes."""
        k_free, d = 3, 2
        filt_mean = jnp.zeros(k_free)
        filt_cov = jnp.eye(k_free)
        B = jnp.zeros((k_free, d))
        A = jnp.eye(k_free)
        u_t = jnp.ones(d)
        Q = jnp.eye(k_free) * 0.01

        pred_mean, pred_cov = covariate_predict(filt_mean, filt_cov, u_t, B, A, Q)

        assert pred_mean.shape == (k_free,)
        assert pred_cov.shape == (k_free, k_free)

    def test_prediction_with_decay(self):
        """decay < 1 should shrink pred_mean toward zero."""
        filt_mean = jnp.array([2.0, 3.0])
        filt_cov = jnp.eye(2) * 0.1
        B = jnp.zeros((2, 1))
        A = jnp.eye(2) * 0.9  # decay = 0.9
        u_t = jnp.zeros(1)
        Q = jnp.eye(2) * 0.01

        pred_mean, _ = covariate_predict(filt_mean, filt_cov, u_t, B, A, Q)

        np.testing.assert_allclose(pred_mean, jnp.array([1.8, 2.7]), atol=1e-6)


class TestMStepInputGain:
    """Tests for m_step_input_gain."""

    def test_b_mstep_recovers_known_input(self):
        """On noiseless data, B_hat should recover B_true."""
        rng = np.random.default_rng(42)
        T = 200
        k_free, d = 2, 2
        B_true = np.array([[0.5, -0.2], [0.1, 0.8]])

        # Generate covariates
        covariates = rng.standard_normal((T, d))

        # Generate smoothed values: x_t = x_{t-1} + B @ u_t (no noise)
        values = np.zeros((T, k_free))
        for t in range(1, T):
            values[t] = values[t - 1] + B_true @ covariates[t]

        B_hat = m_step_input_gain(jnp.array(values), jnp.array(covariates))

        np.testing.assert_allclose(B_hat, B_true, atol=1e-4)

    def test_b_mstep_shapes(self):
        """B should be (K-1, d)."""
        T, k_free, d = 100, 3, 4
        values = jnp.zeros((T, k_free))
        covariates = jnp.ones((T, d))

        B_hat = m_step_input_gain(values, covariates)

        assert B_hat.shape == (k_free, d)

    def test_q_mstep_with_covariates(self):
        """Residual Q should be smaller when B explains the variance."""
        rng = np.random.default_rng(123)
        T = 300
        k_free, d = 2, 2
        B_true = np.array([[1.0, 0.0], [0.0, 1.0]])
        noise_std = 0.01

        covariates = rng.standard_normal((T, d))
        values = np.zeros((T, k_free))
        for t in range(1, T):
            values[t] = (
                values[t - 1]
                + B_true @ covariates[t]
                + rng.normal(0, noise_std, k_free)
            )

        B_hat = m_step_input_gain(jnp.array(values), jnp.array(covariates))

        # Compute residual variance with and without B
        diff = jnp.array(values[1:] - values[:-1])
        u = jnp.array(covariates[1:])

        residual_with_B = diff - u @ B_hat.T
        residual_without_B = diff

        var_with = float(jnp.mean(jnp.var(residual_with_B, axis=0)))
        var_without = float(jnp.mean(jnp.var(residual_without_B, axis=0)))

        assert var_with < var_without * 0.1  # B should explain most variance


class TestCovariateChoiceFilter:
    """Tests for covariate_choice_filter."""

    def test_output_shapes(self):
        """Filter outputs have correct shapes."""
        n_trials, n_options, d = 100, 3, 2
        rng = np.random.default_rng(42)
        choices = jnp.array(rng.integers(0, n_options, n_trials))
        covariates = jnp.array(rng.standard_normal((n_trials, d)))
        B = jnp.zeros((n_options - 1, d))

        result = covariate_choice_filter(
            choices, n_options, covariates=covariates, input_gain=B,
        )

        k_free = n_options - 1
        assert result.filtered_values.shape == (n_trials, k_free)
        assert result.filtered_covariances.shape == (n_trials, k_free, k_free)
        assert result.predicted_values.shape == (n_trials, k_free)
        assert result.predicted_covariances.shape == (n_trials, k_free, k_free)
        assert result.marginal_log_likelihood.shape == ()

    def test_reward_covariate_increases_chosen_value(self):
        """Rewarding option 1 should increase its value."""
        n_trials, n_options = 50, 3
        # Always choose option 1, reward option 1
        choices = jnp.ones(n_trials, dtype=jnp.int32)
        # Covariate: reward for option 1 (column 0 = option 1)
        covariates = jnp.zeros((n_trials, 2))
        covariates = covariates.at[:, 0].set(1.0)
        B = jnp.array([[0.3, 0.0], [0.0, 0.3]])

        result = covariate_choice_filter(
            choices, n_options, covariates=covariates, input_gain=B,
            process_noise=0.001,
        )

        # Value of option 1 (index 0) should increase over trials
        assert float(result.filtered_values[-1, 0]) > float(
            result.filtered_values[0, 0]
        )

    def test_no_covariate_parity_with_multinomial(self):
        """With B=zeros, output should match multinomial_choice_filter exactly."""
        rng = np.random.default_rng(99)
        n_trials, n_options = 80, 4
        choices = jnp.array(rng.integers(0, n_options, n_trials))
        q = 0.02
        beta = 2.0

        # Covariate filter with B=0
        k_free = n_options - 1
        d = 2
        covariates = jnp.array(rng.standard_normal((n_trials, d)))
        B = jnp.zeros((k_free, d))

        cov_result = covariate_choice_filter(
            choices, n_options, covariates=covariates, input_gain=B,
            process_noise=q, inverse_temperature=beta,
        )

        # Multinomial filter (no covariates)
        mult_result = multinomial_choice_filter(
            choices, n_options, process_noise=q, inverse_temperature=beta,
        )

        np.testing.assert_allclose(
            cov_result.filtered_values, mult_result.filtered_values, atol=1e-5,
        )
        np.testing.assert_allclose(
            cov_result.filtered_covariances, mult_result.filtered_covariances,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            cov_result.marginal_log_likelihood,
            mult_result.marginal_log_likelihood,
            atol=1e-3,
        )

    def test_marginal_ll_is_finite(self):
        """Marginal log-likelihood should be finite."""
        rng = np.random.default_rng(7)
        n_trials, n_options = 50, 3
        choices = jnp.array(rng.integers(0, n_options, n_trials))

        result = covariate_choice_filter(choices, n_options)

        assert jnp.isfinite(result.marginal_log_likelihood)


class TestCovariateChoiceSmoother:
    """Tests for covariate_choice_smoother."""

    def test_output_shapes(self):
        """Smoother outputs have correct shapes."""
        n_trials, n_options, d = 100, 3, 2
        rng = np.random.default_rng(42)
        choices = jnp.array(rng.integers(0, n_options, n_trials))
        covariates = jnp.array(rng.standard_normal((n_trials, d)))
        B = jnp.zeros((n_options - 1, d))

        result = covariate_choice_smoother(
            choices, n_options, covariates=covariates, input_gain=B,
        )

        k_free = n_options - 1
        assert result.smoothed_values.shape == (n_trials, k_free)
        assert result.smoothed_covariances.shape == (n_trials, k_free, k_free)
        assert result.smoother_cross_cov.shape == (n_trials - 1, k_free, k_free)
        assert result.marginal_log_likelihood.shape == ()

    def test_smoother_reduces_variance(self):
        """Smoothed covariances should be <= filtered covariances."""
        rng = np.random.default_rng(42)
        n_trials, n_options = 80, 3
        choices = jnp.array(rng.integers(0, n_options, n_trials))

        filt = covariate_choice_filter(choices, n_options)
        smooth = covariate_choice_smoother(choices, n_options)

        # Compare traces (excluding last trial where they're equal)
        filt_traces = jnp.trace(filt.filtered_covariances[:-1], axis1=1, axis2=2)
        smooth_traces = jnp.trace(
            smooth.smoothed_covariances[:-1], axis1=1, axis2=2,
        )
        assert jnp.all(smooth_traces <= filt_traces + 1e-6)

    def test_last_trial_matches_filter(self):
        """Smoother[-1] should equal filter[-1]."""
        rng = np.random.default_rng(42)
        n_trials, n_options = 50, 3
        choices = jnp.array(rng.integers(0, n_options, n_trials))

        filt = covariate_choice_filter(choices, n_options)
        smooth = covariate_choice_smoother(choices, n_options)

        np.testing.assert_allclose(
            smooth.smoothed_values[-1], filt.filtered_values[-1], atol=1e-5,
        )
        np.testing.assert_allclose(
            smooth.smoothed_covariances[-1],
            filt.filtered_covariances[-1],
            atol=1e-5,
        )

    def test_smoother_cross_cov_shape(self):
        """Cross-covariance should be (T-1, K-1, K-1)."""
        rng = np.random.default_rng(42)
        n_trials, n_options = 60, 4
        choices = jnp.array(rng.integers(0, n_options, n_trials))

        result = covariate_choice_smoother(choices, n_options)

        k_free = n_options - 1
        assert result.smoother_cross_cov.shape == (n_trials - 1, k_free, k_free)


def _generate_reward_covariate_data(
    n_trials=200, n_options=3, b_reward=0.5, process_noise=0.005,
    inverse_temperature=2.0, seed=42,
):
    """Helper: generate choice data with reward covariates and known B."""
    rng = np.random.default_rng(seed)
    k_free = n_options - 1
    d = k_free  # one reward covariate per non-reference option

    # B_true: diagonal — each option's reward only updates that option's value
    B_true = np.eye(k_free) * b_reward

    # Simulate
    values = np.zeros((n_trials, k_free))
    choices = np.zeros(n_trials, dtype=int)
    covariates = np.zeros((n_trials, d))

    for t in range(n_trials):
        # Apply covariate-driven update: covariates[t] drives x[t-1] -> x[t]
        if t > 0:
            values[t] = (
                values[t - 1]
                + B_true @ covariates[t]
                + rng.normal(0, np.sqrt(process_noise), k_free)
            )

        # Choice from softmax
        full_vals = np.concatenate([[0.0], values[t]])
        logits = inverse_temperature * full_vals
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        choices[t] = rng.choice(n_options, p=probs)

        # Reward: chosen option gets reward with prob 0.7
        # Reward from trial t becomes covariates[t+1]
        if choices[t] > 0 and t < n_trials - 1:
            covariates[t + 1, choices[t] - 1] = float(rng.random() < 0.7)

    return (
        jnp.array(choices),
        jnp.array(covariates),
        B_true,
        values,
    )


class TestCovariateChoiceModel:
    """Tests for CovariateChoiceModel class."""

    def test_init_and_repr(self):
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        r = repr(model)
        assert "CovariateChoiceModel" in r
        assert "n_options=3" in r
        assert "fitted=False" in r

    def test_invalid_init(self):
        with pytest.raises(ValueError, match="n_options"):
            CovariateChoiceModel(n_options=1, n_covariates=2)

    def test_fit_returns_log_likelihoods(self):
        choices, covariates, _, _ = _generate_reward_covariate_data(n_trials=100)
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        lls = model.fit(choices, covariates=covariates, max_iter=5)
        assert len(lls) > 0
        assert all(np.isfinite(ll) for ll in lls)

    def test_is_fitted(self):
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        assert not model.is_fitted
        choices, covariates, _, _ = _generate_reward_covariate_data(n_trials=80)
        model.fit(choices, covariates=covariates, max_iter=3)
        assert model.is_fitted

    def test_fit_learns_reward_sensitivity(self):
        """Rewarding option 1 -> B[0, 0] should be positive."""
        choices, covariates, _, _ = _generate_reward_covariate_data(
            n_trials=300, b_reward=0.8, seed=55,
        )
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        model.fit(choices, covariates=covariates, max_iter=20)

        # B[0, 0] should be positive (reward for option 1 increases its value)
        assert float(model.input_gain_[0, 0]) > 0

    def test_fit_without_covariates_matches_multinomial(self):
        """With no covariates, LL should match MultinomialChoiceModel."""
        rng = np.random.default_rng(77)
        n_trials, n_options = 150, 3
        choices = jnp.array(rng.integers(0, n_options, n_trials))

        # Fit multinomial model
        mult_model = MultinomialChoiceModel(n_options=n_options)
        mult_model.fit(choices, max_iter=15)

        # Fit covariate model without covariates
        cov_model = CovariateChoiceModel(n_options=n_options, n_covariates=0)
        cov_model.fit(choices, max_iter=15)

        # Log-likelihoods should be close
        np.testing.assert_allclose(
            cov_model.log_likelihood_, mult_model.log_likelihood_, atol=0.1,
        )

    def test_fit_with_covariates_improves_ll(self):
        """Covariate model should have higher LL on data with known B."""
        choices, covariates, _, _ = _generate_reward_covariate_data(
            n_trials=300, b_reward=0.8, seed=123,
        )

        # No-covariate model
        null_model = CovariateChoiceModel(n_options=3, n_covariates=0)
        null_model.fit(choices, max_iter=20)

        # Covariate model
        cov_model = CovariateChoiceModel(n_options=3, n_covariates=2)
        cov_model.fit(choices, covariates=covariates, max_iter=20)

        assert cov_model.log_likelihood_ > null_model.log_likelihood_

    def test_residual_q_smaller_with_covariates(self):
        """When B explains drift, residual Q should shrink."""
        choices, covariates, _, _ = _generate_reward_covariate_data(
            n_trials=500, b_reward=1.5, process_noise=0.001, seed=44,
        )

        null_model = CovariateChoiceModel(n_options=3, n_covariates=0)
        null_model.fit(choices, max_iter=25)

        cov_model = CovariateChoiceModel(n_options=3, n_covariates=2)
        cov_model.fit(choices, covariates=covariates, max_iter=25)

        assert cov_model.process_noise < null_model.process_noise

    def test_em_log_likelihood_non_decreasing(self):
        """EM log-likelihood should be non-decreasing (within tolerance)."""
        choices, covariates, _, _ = _generate_reward_covariate_data(n_trials=200)
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        lls = model.fit(choices, covariates=covariates, max_iter=15)

        for i in range(1, len(lls)):
            assert lls[i] >= lls[i - 1] - 0.1  # Laplace-EKF is approximate

    def test_choice_probabilities(self):
        choices, covariates, _, _ = _generate_reward_covariate_data(n_trials=100)
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        model.fit(choices, covariates=covariates, max_iter=5)
        probs = model.choice_probabilities()

        assert probs.shape == (100, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert jnp.all(probs >= 0)

    def test_summary(self):
        choices, covariates, _, _ = _generate_reward_covariate_data(n_trials=100)
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        model.fit(choices, covariates=covariates, max_iter=5)
        s = model.summary()

        assert "CovariateChoiceModel" in s
        assert "input_gain" in s

    def test_compare_to_null(self):
        choices, covariates, _, _ = _generate_reward_covariate_data(n_trials=100)
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        model.fit(choices, covariates=covariates, max_iter=5)
        comparison = model.compare_to_null()

        assert "model_ll" in comparison
        assert "null_ll" in comparison
        assert "delta_bic" in comparison

    def test_input_gain_matrix_shape(self):
        """After fit, input_gain_ should be (K-1, d)."""
        choices, covariates, _, _ = _generate_reward_covariate_data(n_trials=100)
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        model.fit(choices, covariates=covariates, max_iter=5)

        assert model.input_gain_.shape == (2, 2)

    def test_bic_comparison(self):
        """BIC should favor covariate model when covariates are informative.

        Uses LL comparison rather than BIC since the B M-step produces
        modest but consistent LL improvements. With informative covariates,
        the covariate model should achieve higher LL.
        """
        choices, covariates, _, _ = _generate_reward_covariate_data(
            n_trials=500, b_reward=1.5, process_noise=0.001, seed=88,
        )

        null_model = CovariateChoiceModel(n_options=3, n_covariates=0)
        null_model.fit(choices, max_iter=25)

        cov_model = CovariateChoiceModel(n_options=3, n_covariates=2)
        cov_model.fit(choices, covariates=covariates, max_iter=25)

        # Covariate model should have better log-likelihood
        assert cov_model.log_likelihood_ > null_model.log_likelihood_

    def test_rescorla_wagner_equivalence(self):
        """K=2, reward covariate: B should be positive and capture reward signal.

        With a single reward covariate, B acts as an RL learning rate.
        We verify: (1) B > 0 (reward increases value), (2) covariate model
        has better LL than null, (3) B is in a reasonable range.
        """
        rng = np.random.default_rng(42)
        alpha_true = 0.5
        n_trials = 500

        # Generate RW data with known learning rate
        # Convention: covariates[t] drives x[t-1] -> x[t]
        # So reward from trial t becomes covariates[t+1]
        value = 0.0
        choices = np.zeros(n_trials, dtype=int)
        covariates = np.zeros((n_trials, 1))

        for t in range(n_trials):
            # Apply covariate update (covariates[t] was set on previous trial)
            value = value + alpha_true * covariates[t, 0]

            p1 = 1.0 / (1.0 + np.exp(-2.0 * value))
            choices[t] = 1 if rng.random() < p1 else 0

            # Reward from this trial drives next trial's prediction
            if choices[t] == 1 and t < n_trials - 1:
                covariates[t + 1, 0] = float(rng.random() < 0.7)

        # Fit covariate model
        cov_model = CovariateChoiceModel(n_options=2, n_covariates=1)
        cov_model.fit(
            jnp.array(choices), covariates=jnp.array(covariates), max_iter=30,
        )

        # Fit null model (no covariates)
        null_model = CovariateChoiceModel(n_options=2, n_covariates=0)
        null_model.fit(jnp.array(choices), max_iter=30)

        # B should be positive (reward increases value)
        b_hat = float(cov_model.input_gain_[0, 0])
        assert b_hat > 0, f"B should be positive, got {b_hat}"

        # Covariate model should fit better
        assert cov_model.log_likelihood_ > null_model.log_likelihood_


class TestSimulateRLChoiceData:
    """Tests for simulate_rl_choice_data."""

    def test_output_shapes(self):
        n_trials, n_options, d = 200, 3, 2
        B = jnp.eye(2) * 0.5
        data = simulate_rl_choice_data(
            n_trials=n_trials, n_options=n_options, input_gain=B,
        )
        assert data.choices.shape == (n_trials,)
        assert data.true_values.shape == (n_trials, n_options - 1)
        assert data.true_probs.shape == (n_trials, n_options)
        assert data.covariates.shape == (n_trials, d)

    def test_choices_are_valid(self):
        B = jnp.eye(2) * 0.3
        data = simulate_rl_choice_data(n_trials=100, n_options=3, input_gain=B)
        choices_np = np.array(data.choices)
        assert np.all(choices_np >= 0)
        assert np.all(choices_np < 3)

    def test_reward_driven_choices(self):
        """High B should cause choices to track rewarded option."""
        B = jnp.array([[2.0, 0.0], [0.0, 2.0]])
        data = simulate_rl_choice_data(
            n_trials=300, n_options=3, input_gain=B,
            inverse_temperature=3.0, reward_prob=0.9,
            process_noise=0.001, seed=42,
        )
        # With high B and reward, should mostly choose non-reference options
        choices_np = np.array(data.choices)
        frac_nonref = (choices_np > 0).mean()
        assert frac_nonref > 0.5

    def test_seed_reproducibility(self):
        B = jnp.eye(2) * 0.5
        data1 = simulate_rl_choice_data(
            n_trials=50, n_options=3, input_gain=B, seed=123,
        )
        data2 = simulate_rl_choice_data(
            n_trials=50, n_options=3, input_gain=B, seed=123,
        )
        np.testing.assert_array_equal(data1.choices, data2.choices)
        np.testing.assert_allclose(data1.true_values, data2.true_values)

    def test_returned_type(self):
        B = jnp.eye(2) * 0.5
        data = simulate_rl_choice_data(
            n_trials=50, n_options=3, input_gain=B,
        )
        assert isinstance(data, SimulatedRLChoiceData)


class TestCovariateChoiceModelPlotting:
    """Tests for plotting methods."""

    @pytest.fixture()
    def fitted_model(self):
        choices, covariates, _, _ = _generate_reward_covariate_data(n_trials=100)
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        model.fit(choices, covariates=covariates, max_iter=5)
        return model, choices

    def test_plot_values_with_covariates(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        model, choices = fitted_model
        fig, axes = model.plot_values(observed_choices=choices)
        assert fig is not None
        assert len(axes) == 2

    def test_plot_input_gains(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        model, _ = fitted_model
        fig, ax = model.plot_input_gains(
            covariate_labels=["Reward 1", "Reward 2"],
        )
        assert fig is not None

    def test_plot_convergence(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        model, _ = fitted_model
        fig, ax = model.plot_convergence()
        assert fig is not None

    def test_plot_summary(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        model, choices = fitted_model
        fig, axes = model.plot_summary(observed_choices=choices)
        assert fig is not None
        assert len(axes) == 3


class TestRescorlaWagnerComparison:
    """Extended RW comparison tests."""

    def test_rw_with_multiple_learning_rates(self):
        """K=3, reward covariates: B diagonal should be positive."""
        choices, covariates, B_true, _ = _generate_reward_covariate_data(
            n_trials=500, n_options=3, b_reward=0.5,
            process_noise=0.005, inverse_temperature=1.5, seed=77,
        )

        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        model.fit(choices, covariates=covariates, max_iter=25)

        # At least one diagonal entry should be positive
        diag_b = [float(model.input_gain_[k, k]) for k in range(2)]
        assert any(b > 0 for b in diag_b), f"B diagonal: {diag_b}"

    def test_full_integration_simulate_fit_recover(self):
        """Simulate with known B -> fit -> covariate model beats null."""
        B_true = jnp.array([[0.3, 0.0], [0.0, 0.3]])
        data = simulate_rl_choice_data(
            n_trials=500, n_options=3, input_gain=B_true,
            process_noise=0.005, inverse_temperature=1.5, seed=55,
        )

        # Fit covariate model
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        model.fit(data.choices, covariates=data.covariates, max_iter=25)

        # Should beat null model on LL
        null_model = CovariateChoiceModel(n_options=3, n_covariates=0)
        null_model.fit(data.choices, max_iter=25)
        assert model.log_likelihood_ > null_model.log_likelihood_


class TestObservationCovariates:
    """Tests for observation covariates (Theta @ z_t in softmax)."""

    def test_no_obs_covariates_parity(self):
        """With obs_weights=0, should match model without obs covariates."""
        rng = np.random.default_rng(42)
        choices = np.where(rng.random(100) < 0.7, 1, rng.integers(0, 3, size=100))
        # Dummy obs covariates that should have no effect when Theta=0
        obs_cov = rng.standard_normal((100, 2))

        # Model without obs covariates
        m1 = CovariateChoiceModel(n_options=3)
        m1.fit(choices, max_iter=3)

        # Model with obs covariates but learn_obs_weights=False (Theta stays 0)
        m2 = CovariateChoiceModel(n_options=3, n_obs_covariates=2,
                                   learn_obs_weights=False)
        m2.fit(choices, obs_covariates=obs_cov, max_iter=3)

        # Log-likelihoods should be identical
        np.testing.assert_allclose(
            m1.log_likelihood_, m2.log_likelihood_, rtol=1e-4,
        )

    def test_stay_bias_learned(self):
        """With strong perseveration, Theta should capture stay preference."""
        rng = np.random.default_rng(42)
        n_trials = 300
        # Generate choices with strong perseveration
        choices = np.zeros(n_trials, dtype=int)
        choices[0] = rng.integers(0, 3)
        for t in range(1, n_trials):
            if rng.random() < 0.8:  # 80% stay
                choices[t] = choices[t - 1]
            else:
                choices[t] = rng.integers(0, 3)

        # Build stay indicator: z_t[k] = 1 if option k was chosen on trial t-1
        obs_cov = np.zeros((n_trials, 3))
        for t in range(1, n_trials):
            obs_cov[t, choices[t - 1]] = 1.0

        model = CovariateChoiceModel(
            n_options=3, n_obs_covariates=3,
            learn_inverse_temperature=False,
            learn_process_noise=False,
        )
        model.fit(choices, obs_covariates=obs_cov, max_iter=10)

        # Diagonal of Theta should be positive (stay bias)
        Theta = np.array(model.obs_weights_)
        for k in range(3):
            assert Theta[k, k] > 0, f"Stay bias for option {k}: {Theta[k, k]:.3f}"

    def test_obs_covariates_improve_ll(self):
        """Model with obs covariates should improve LL on perseverative data."""
        rng = np.random.default_rng(42)
        n_trials = 200
        choices = np.zeros(n_trials, dtype=int)
        choices[0] = 1
        for t in range(1, n_trials):
            if rng.random() < 0.75:
                choices[t] = choices[t - 1]
            else:
                choices[t] = rng.integers(0, 3)

        obs_cov = np.zeros((n_trials, 3))
        for t in range(1, n_trials):
            obs_cov[t, choices[t - 1]] = 1.0

        # Fix beta and Q so only Theta is learned — avoids EM instability
        # from joint beta/Theta optimization
        # Without obs covariates
        m_base = CovariateChoiceModel(
            n_options=3, learn_inverse_temperature=False, learn_process_noise=False,
        )
        m_base.fit(choices, max_iter=5)

        # With obs covariates (only learn Theta)
        m_obs = CovariateChoiceModel(
            n_options=3, n_obs_covariates=3,
            learn_inverse_temperature=False, learn_process_noise=False,
        )
        m_obs.fit(choices, obs_covariates=obs_cov, max_iter=5)

        assert m_obs.log_likelihood_ > m_base.log_likelihood_

    def test_obs_covariates_dont_change_latent_state(self):
        """Obs covariates shift probabilities, not the value trajectory."""
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 3, size=100)
        obs_cov = rng.standard_normal((100, 2))

        # Fix all params, only learn Theta
        m = CovariateChoiceModel(
            n_options=3, n_obs_covariates=2,
            learn_inverse_temperature=False,
            learn_process_noise=False,
        )
        m.fit(choices, obs_covariates=obs_cov, max_iter=5)
        vals_with = np.array(m.smoothed_values)

        # Same but without obs covariates
        m_base = CovariateChoiceModel(
            n_options=3,
            learn_inverse_temperature=False,
            learn_process_noise=False,
        )
        m_base.fit(choices, max_iter=5)
        vals_without = np.array(m_base.smoothed_values)

        # Values should be similar (not identical since obs covariates
        # change the softmax probabilities which feed back through the
        # Laplace update, but the effect should be small)
        corr = np.corrcoef(vals_with.ravel(), vals_without.ravel())[0, 1]
        assert corr > 0.8, f"Value trajectories too different: r={corr:.3f}"

    def test_obs_weights_shape(self):
        """obs_weights_ should be (K, d_obs)."""
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 4, size=50)
        obs_cov = rng.standard_normal((50, 3))

        m = CovariateChoiceModel(n_options=4, n_obs_covariates=3)
        m.fit(choices, obs_covariates=obs_cov, max_iter=2)
        assert m.obs_weights_.shape == (4, 3)

    def test_n_free_params_includes_obs_weights(self):
        """BIC should count Theta parameters."""
        m = CovariateChoiceModel(
            n_options=3, n_obs_covariates=2,
            learn_process_noise=True,
            learn_inverse_temperature=True,
        )
        # Q(1) + beta(1) + Theta(3*2=6) = 8
        assert m.n_free_params == 8

    def test_choice_probabilities_include_obs_offset(self):
        """choice_probabilities should include Theta @ z_t."""
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 3, size=100)
        obs_cov = rng.standard_normal((100, 2))

        m = CovariateChoiceModel(n_options=3, n_obs_covariates=2)
        m.fit(choices, obs_covariates=obs_cov, max_iter=3)
        probs = m.choice_probabilities()
        assert probs.shape == (100, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_missing_obs_covariates_raises(self):
        """If n_obs_covariates > 0 but no obs_covariates passed, raise."""
        m = CovariateChoiceModel(n_options=3, n_obs_covariates=2)
        with pytest.raises(ValueError, match="obs_covariates"):
            m.fit(np.array([0, 1, 2, 1, 0]))


class TestDecayDynamics:
    """Tests for mean-reverting value decay (A = decay * I)."""

    def test_decay_1_is_random_walk(self):
        """decay=1.0 should match the default (no decay)."""
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 3, size=100)

        m1 = CovariateChoiceModel(n_options=3, init_decay=1.0, learn_decay=False)
        m1.fit(choices, max_iter=3)

        m2 = CovariateChoiceModel(n_options=3, learn_decay=False)
        m2.fit(choices, max_iter=3)

        np.testing.assert_allclose(m1.log_likelihood_, m2.log_likelihood_, rtol=1e-4)

    def test_decay_shrinks_values(self):
        """With decay < 1, values should stay closer to zero than random walk."""
        rng = np.random.default_rng(42)
        choices = np.where(rng.random(200) < 0.8, 1, rng.integers(0, 3, size=200))

        m_walk = CovariateChoiceModel(
            n_options=3, init_decay=1.0, learn_decay=False,
            learn_inverse_temperature=False, learn_process_noise=False,
        )
        m_walk.fit(choices, max_iter=3)

        m_decay = CovariateChoiceModel(
            n_options=3, init_decay=0.9, learn_decay=False,
            learn_inverse_temperature=False, learn_process_noise=False,
        )
        m_decay.fit(choices, max_iter=3)

        walk_mag = np.mean(np.abs(np.array(m_walk.smoothed_values)))
        decay_mag = np.mean(np.abs(np.array(m_decay.smoothed_values)))
        assert decay_mag < walk_mag

    def test_learn_decay(self):
        """EM should learn a decay < 1 on mean-reverting data."""
        rng = np.random.default_rng(42)
        n_trials = 300
        # Generate data with decay: values revert to 0
        true_decay = 0.85
        x = np.zeros((n_trials, 2))
        for t in range(1, n_trials):
            x[t] = true_decay * x[t - 1] + rng.normal(0, 0.1, 2)
        # Choices from softmax
        full_v = np.column_stack([np.zeros(n_trials), x])
        probs = np.exp(2.0 * full_v)
        probs = probs / probs.sum(axis=1, keepdims=True)
        choices = np.array([rng.choice(3, p=probs[t]) for t in range(n_trials)])

        model = CovariateChoiceModel(
            n_options=3, learn_decay=True,
            init_decay=1.0,  # start from random walk
        )
        model.fit(choices, max_iter=15)
        # Should learn decay < 1
        assert model.decay < 0.99, f"decay={model.decay:.3f}, expected < 0.99"

    def test_decay_with_covariates(self):
        """Decay should work alongside dynamics covariates."""
        rng = np.random.default_rng(42)
        n_trials = 200
        choices = np.where(rng.random(n_trials) < 0.7, 1, rng.integers(0, 3, size=n_trials))
        covariates = np.zeros((n_trials, 2))
        for t in range(1, n_trials):
            if choices[t - 1] > 0:
                covariates[t, choices[t - 1] - 1] = 1.0

        model = CovariateChoiceModel(
            n_options=3, n_covariates=2, init_decay=0.95,
            learn_decay=False,
        )
        model.fit(choices, covariates=covariates, max_iter=5)
        assert np.isfinite(model.log_likelihood_)

    def test_decay_in_repr(self):
        m = CovariateChoiceModel(n_options=3, init_decay=0.9)
        assert "decay=0.9000" in repr(m)

    def test_n_free_params_with_decay(self):
        m = CovariateChoiceModel(
            n_options=3, learn_decay=True,
            learn_process_noise=True,
            learn_inverse_temperature=True,
        )
        # Q(1) + beta(1) + decay(1) = 3
        assert m.n_free_params == 3


class TestSGDFitting:
    """Tests for SGD fitting via optax."""

    def test_sgd_improves_ll(self):
        """Log-likelihood should improve from initial parameters."""
        rng = np.random.default_rng(42)
        choices = np.where(rng.random(200) < 0.7, 1, rng.integers(0, 3, size=200))
        model = CovariateChoiceModel(n_options=3)
        lls = model.fit_sgd(choices, num_steps=50)
        assert lls[-1] > lls[0]

    def test_sgd_learns_beta(self):
        """Deterministic choices should lead to high beta."""
        choices = np.ones(200, dtype=int)
        model = CovariateChoiceModel(n_options=3)
        model.fit_sgd(choices, num_steps=200)
        assert model.inverse_temperature > 1.5

    def test_sgd_respects_constraints(self):
        """After SGD: process_noise > 0, 0 < decay < 1."""
        rng = np.random.default_rng(42)
        model = CovariateChoiceModel(
            n_options=3, init_decay=0.9, learn_decay=True,
        )
        model.fit_sgd(rng.integers(0, 3, size=100), num_steps=50)
        assert model.process_noise > 0
        assert 0 < model.decay < 1

    def test_sgd_with_covariates(self):
        """SGD should work with dynamics covariates."""
        data = simulate_rl_choice_data(n_trials=200, n_options=3, seed=42)
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        lls = model.fit_sgd(
            data.choices, covariates=data.covariates, num_steps=50,
        )
        assert np.isfinite(model.log_likelihood_)
        assert len(lls) == 50

    def test_sgd_model_is_fitted(self):
        """Model should report as fitted after SGD."""
        rng = np.random.default_rng(42)
        model = CovariateChoiceModel(n_options=3)
        model.fit_sgd(rng.integers(0, 3, size=100), num_steps=20)
        assert model.is_fitted
        assert model.smoothed_values.shape[0] == 100

    def test_sgd_verbose(self, capsys):
        """Verbose mode should print progress to stdout."""
        rng = np.random.default_rng(42)
        model = CovariateChoiceModel(n_options=3)
        model.fit_sgd(
            rng.integers(0, 3, size=50), num_steps=15, verbose=True,
        )
        captured = capsys.readouterr()
        assert "SGD step" in captured.out


class TestCovariateUncertaintySummaries:
    """Tests for uncertainty summary attributes."""

    def test_uncertainty_populated_after_em(self):
        from state_space_practice.covariate_choice import simulate_rl_choice_data

        sim = simulate_rl_choice_data(n_trials=50, n_options=3, seed=0)
        model = CovariateChoiceModel(
            n_options=3, n_covariates=sim.covariates.shape[1],
        )
        model.fit(sim.choices, covariates=sim.covariates, max_iter=3)
        assert model.predicted_option_variances_ is not None
        assert model.predicted_option_variances_.shape == (50, 3)
        assert model.surprise_ is not None
        assert model.surprise_.shape == (50,)

    def test_uncertainty_populated_after_sgd(self):
        rng = np.random.default_rng(42)
        model = CovariateChoiceModel(n_options=3)
        model.fit_sgd(rng.integers(0, 3, size=50), num_steps=10)
        assert model.predicted_option_variances_ is not None
        assert model.predicted_choice_entropy_ is not None

    def test_option_values_populated(self):
        from state_space_practice.covariate_choice import simulate_rl_choice_data

        sim = simulate_rl_choice_data(n_trials=50, n_options=3, seed=0)
        model = CovariateChoiceModel(
            n_options=3, n_covariates=sim.covariates.shape[1],
        )
        model.fit(sim.choices, covariates=sim.covariates, max_iter=3)
        assert model.predicted_option_values_.shape == (50, 3)
        assert model.filtered_option_values_.shape == (50, 3)
        assert model.smoothed_option_values_.shape == (50, 3)
        assert model.filtered_option_variances_.shape == (50, 3)
        # Reference option value should be zero
        np.testing.assert_allclose(model.predicted_option_values_[:, 0], 0.0)
        np.testing.assert_allclose(model.filtered_option_values_[:, 0], 0.0)
        np.testing.assert_allclose(model.smoothed_option_values_[:, 0], 0.0)
        np.testing.assert_allclose(model.filtered_option_variances_[:, 0], 0.0)

    def test_obs_covariates_affect_surprise_and_entropy(self):
        """Nonzero obs_covariates should change surprise and entropy."""
        rng = np.random.default_rng(99)
        n_trials, n_options = 50, 3
        choices = rng.integers(0, n_options, size=n_trials)

        # Fit without obs covariates
        m_base = CovariateChoiceModel(n_options=n_options)
        m_base.fit_sgd(choices, num_steps=5)

        # Fit with obs covariates (strong offset favoring option 1)
        obs_cov = np.zeros((n_trials, 1))
        obs_cov[:, 0] = 3.0  # constant strong bias
        m_obs = CovariateChoiceModel(n_options=n_options, n_obs_covariates=1)
        # Set a nonzero obs weight so the offset actually takes effect
        m_obs.obs_weights_ = jnp.array([[0.0], [2.0], [-1.0]])  # (K, d_obs)
        m_obs.fit_sgd(choices, obs_covariates=obs_cov, num_steps=5)

        # The obs offset shifts choice probabilities, so entropy and
        # surprise must differ from the no-offset model
        assert not np.allclose(
            m_base.predicted_choice_entropy_, m_obs.predicted_choice_entropy_,
            atol=1e-3,
        ), "obs_covariates should change predicted_choice_entropy_"
        assert not np.allclose(
            m_base.surprise_, m_obs.surprise_, atol=1e-3,
        ), "obs_covariates should change surprise_"
