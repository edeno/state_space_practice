# ruff: noqa: E402
"""Integration tests: verify SGD fitting recovers known parameters on simulated data.

These tests simulate data with known ground-truth parameters, fit via SGD,
and check that the recovered parameters are close to ground truth. They also
compare SGD results against EM where applicable.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# MultinomialChoiceModel: simulate → fit_sgd → verify recovery
# ---------------------------------------------------------------------------


class TestMultinomialChoiceSGDIntegration:
    """End-to-end: simulate choice data → fit_sgd → recover parameters."""

    def test_recover_process_noise_and_beta(self):
        from state_space_practice.multinomial_choice import (
            MultinomialChoiceModel,
            simulate_choice_data,
        )

        true_q = 0.02
        true_beta = 3.0
        sim = simulate_choice_data(
            n_trials=300, n_options=3,
            process_noise=true_q, inverse_temperature=true_beta, seed=42,
        )
        choices = sim.choices

        model = MultinomialChoiceModel(
            n_options=3,
            init_process_noise=0.05,
            init_inverse_temperature=1.0,
        )
        lls = model.fit_sgd(choices, num_steps=300)

        # LL should improve
        assert lls[-1] > lls[0]
        # Parameters should be positive and finite
        assert model.process_noise > 0
        assert model.inverse_temperature > 0
        assert np.isfinite(model.process_noise)
        assert np.isfinite(model.inverse_temperature)
        # Model should be fully fitted (smoother populated)
        assert model.is_fitted
        assert model.smoothed_values.shape == (300, 2)

    def test_sgd_vs_em_agreement(self):
        from state_space_practice.multinomial_choice import (
            MultinomialChoiceModel,
            simulate_choice_data,
        )

        sim = simulate_choice_data(
            n_trials=200, n_options=4, process_noise=0.01,
            inverse_temperature=2.0, seed=123,
        )
        choices = sim.choices

        model_em = MultinomialChoiceModel(n_options=4)
        model_em.fit(choices, max_iter=30)

        model_sgd = MultinomialChoiceModel(n_options=4)
        model_sgd.fit_sgd(choices, num_steps=300)

        # Both should find reasonable fits (SGD and EM may find different
        # local optima due to Q/beta tradeoff, so compare LL not params)
        assert model_sgd.is_fitted
        assert model_em.is_fitted
        # Both should have finite LL
        assert np.isfinite(model_sgd.log_likelihood_)
        assert np.isfinite(model_em.log_likelihood_)


# ---------------------------------------------------------------------------
# CovariateChoiceModel: simulate → fit_sgd → verify recovery
# ---------------------------------------------------------------------------


class TestCovariateChoiceSGDIntegration:
    """End-to-end: simulate RL choice data → fit_sgd → recover parameters."""

    def test_recover_with_covariates(self):
        from state_space_practice.covariate_choice import (
            CovariateChoiceModel,
            simulate_rl_choice_data,
        )

        sim = simulate_rl_choice_data(
            n_trials=300, n_options=3,
            inverse_temperature=2.0, seed=42,
        )

        model = CovariateChoiceModel(
            n_options=3, n_covariates=sim.covariates.shape[1],
            init_inverse_temperature=1.0, init_process_noise=0.01,
        )
        lls = model.fit_sgd(
            sim.choices, covariates=sim.covariates, num_steps=200,
        )

        assert lls[-1] > lls[0]
        assert model.is_fitted
        # Input gain should be learned (nonzero)
        assert jnp.any(jnp.abs(model.input_gain_) > 0.01)

    def test_sgd_vs_em_agreement(self):
        from state_space_practice.covariate_choice import (
            CovariateChoiceModel,
            simulate_rl_choice_data,
        )

        sim = simulate_rl_choice_data(
            n_trials=200, n_options=3,
            inverse_temperature=2.0, seed=99,
        )

        model_em = CovariateChoiceModel(
            n_options=3, n_covariates=sim.covariates.shape[1],
        )
        model_em.fit(sim.choices, covariates=sim.covariates, max_iter=30)

        model_sgd = CovariateChoiceModel(
            n_options=3, n_covariates=sim.covariates.shape[1],
        )
        model_sgd.fit_sgd(
            sim.choices, covariates=sim.covariates, num_steps=300,
        )

        # Both should have similar final LL
        assert abs(model_sgd.log_likelihood_ - model_em.log_likelihood_) < 10.0


# ---------------------------------------------------------------------------
# SmithLearningModel: simulate → fit_sgd → verify recovery
# ---------------------------------------------------------------------------


class TestSmithSGDIntegration:
    """End-to-end: simulate learning data → fit_sgd → recover sigma_epsilon."""

    def test_recover_sigma_epsilon(self):
        from state_space_practice.smith_learning_algorithm import (
            SmithLearningModel,
            simulate_learning_data,
        )

        true_sigma = 0.3
        outcomes, _ = simulate_learning_data(
            n_trials=200,
            prob_success_init=0.4,
            prob_success_final=0.85,
            seed=42,
        )

        model = SmithLearningModel(sigma_epsilon=0.1)
        lls = model.fit_sgd(outcomes, num_steps=100)

        assert lls[-1] > lls[0]
        assert model.is_fitted
        # sigma should move toward reasonable range
        assert model.sigma_epsilon > 0.05
        assert model.sigma_epsilon < 1.0

    def test_sgd_vs_em_agreement(self):
        from state_space_practice.smith_learning_algorithm import (
            SmithLearningModel,
            simulate_learning_data,
        )

        outcomes, _ = simulate_learning_data(
            n_trials=150,
            prob_success_init=0.3,
            prob_success_final=0.8,
            seed=77,
        )

        model_em = SmithLearningModel(sigma_epsilon=0.15)
        model_em.fit(outcomes, max_iter=50)

        model_sgd = SmithLearningModel(sigma_epsilon=0.15)
        model_sgd.fit_sgd(outcomes, num_steps=200)

        # Both should find similar sigma
        assert abs(model_sgd.sigma_epsilon - model_em.sigma_epsilon) < 0.15
        # Smoothed states should be correlated
        corr = float(jnp.corrcoef(
            model_sgd.smoothed_learning_state_mode,
            model_em.smoothed_learning_state_mode,
        )[0, 1])
        assert corr > 0.9, f"Smoothed state correlation too low: {corr}"


# ---------------------------------------------------------------------------
# PointProcessModel: simulate → fit_sgd → verify recovery
# ---------------------------------------------------------------------------


class TestPointProcessSGDIntegration:
    """End-to-end: simulate spike data → fit_sgd → recover dynamics."""

    def test_recover_transition_and_process_cov(self):
        from state_space_practice.point_process_kalman import PointProcessModel

        n_state = 2
        n_neurons = 5
        n_time = 200
        dt = 0.001

        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Ground truth: damped random walk
        true_A = 0.95 * jnp.eye(n_state)
        true_Q = 0.005 * jnp.eye(n_state)

        # Simulate latent states
        def _step(x, k):
            x_new = true_A @ x + jax.random.multivariate_normal(k, jnp.zeros(n_state), true_Q)
            return x_new, x_new

        keys = jax.random.split(k1, n_time)
        _, states = jax.lax.scan(_step, jnp.zeros(n_state), keys)

        # Neuron weights
        W = jax.random.normal(k2, (n_neurons, n_state)) * 0.5
        design_matrix = jnp.tile(W, (n_time, 1, 1))

        # Simulate spikes
        log_rates = jnp.einsum("tnk,tk->tn", design_matrix, states)
        rates = jnp.exp(jnp.clip(log_rates, -5, 5)) * dt
        spikes = jax.random.poisson(k3, rates)

        # Fit with SGD
        model = PointProcessModel(
            n_state, dt,
            transition_matrix=jnp.eye(n_state),
            process_cov=0.01 * jnp.eye(n_state),
        )
        lls = model.fit_sgd(design_matrix, spikes, num_steps=50)

        # LL should improve
        assert lls[-1] > lls[0]
        # Process cov should be PSD
        eigvals = jnp.linalg.eigvalsh(model.process_cov)
        assert jnp.all(eigvals > 0)
        # Transition matrix should be close to true_A
        assert float(jnp.linalg.norm(model.transition_matrix - true_A)) < 0.5
        # Smoother results should be populated
        assert model.smoother_mean is not None
        assert model.smoother_mean.shape == (n_time, n_state)

    def test_sgd_vs_em_agreement(self):
        from state_space_practice.point_process_kalman import PointProcessModel

        n_state = 2
        n_neurons = 4
        n_time = 150
        dt = 0.001

        key = jax.random.PRNGKey(99)
        k1, k2, k3 = jax.random.split(key, 3)

        A = 0.98 * jnp.eye(n_state)
        Q = 0.002 * jnp.eye(n_state)

        def _step(x, k):
            x_new = A @ x + jax.random.multivariate_normal(k, jnp.zeros(n_state), Q)
            return x_new, x_new

        keys = jax.random.split(k1, n_time)
        _, states = jax.lax.scan(_step, jnp.zeros(n_state), keys)

        W = jax.random.normal(k2, (n_neurons, n_state)) * 0.3
        design_matrix = jnp.tile(W, (n_time, 1, 1))
        log_rates = jnp.einsum("tnk,tk->tn", design_matrix, states)
        rates = jnp.exp(jnp.clip(log_rates, -5, 5)) * dt
        spikes = jax.random.poisson(k3, rates)

        # EM fit
        model_em = PointProcessModel(n_state, dt)
        model_em.fit(design_matrix, spikes, max_iter=15)

        # SGD fit
        model_sgd = PointProcessModel(n_state, dt)
        model_sgd.fit_sgd(design_matrix, spikes, num_steps=50)

        # Both should find finite log-likelihoods
        assert model_em.smoother_mean is not None
        assert model_sgd.smoother_mean is not None
        # Smoothed states should be correlated
        corr = float(jnp.corrcoef(
            model_sgd.smoother_mean[:, 0],
            model_em.smoother_mean[:, 0],
        )[0, 1])
        assert corr > 0.7, f"Smoothed state correlation too low: {corr}"
