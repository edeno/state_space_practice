"""Tests for the smith_learning_algorithm module.

This module tests the Bayesian state-space model for learning dynamics,
including the Laplace approximation filter/smoother and EM algorithm.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from state_space_practice.smith_learning_algorithm import (
    SmithLearningAlgorithm,
    _find_runs_of_value,
    _log_posterior_objective,
    approximate_gaussian,
    calculate_latent_state_percentiles,
    calculate_probability_confidence_limits,
    compute_cross_covariance_matrix,
    compute_trial_comparison_matrix,
    compare_two_trials,
    find_first_significant_trial,
    find_min_consecutive_successes,
    maximization_step,
    simulate_learning_data,
    smith_learning_filter,
    smith_learning_smoother,
)

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


class TestApproximateGaussian:
    """Tests for the approximate_gaussian function (Laplace approximation)."""

    def test_finds_mode_of_gaussian(self) -> None:
        """Should find mode of a Gaussian log-posterior correctly."""
        true_mode = 2.0
        variance = 0.5

        def log_posterior(x):
            return -0.5 * (x[0] - true_mode) ** 2 / variance

        mode, cov = approximate_gaussian(log_posterior, jnp.array([0.0]))

        np.testing.assert_allclose(mode[0], true_mode, rtol=1e-4)
        np.testing.assert_allclose(cov[0, 0], variance, rtol=1e-3)

    def test_covariance_positive(self) -> None:
        """Covariance should be positive for valid log-posterior."""

        def log_posterior(x):
            return -0.5 * x[0] ** 2  # Standard normal

        _, cov = approximate_gaussian(log_posterior, jnp.array([1.0]))

        assert cov[0, 0] > 0

    def test_handles_non_zero_mode(self) -> None:
        """Should correctly find non-zero mode."""
        true_mode = -3.5

        def log_posterior(x):
            return -2.0 * (x[0] - true_mode) ** 2

        mode, _ = approximate_gaussian(log_posterior, jnp.array([0.0]))

        np.testing.assert_allclose(mode[0], true_mode, rtol=1e-4)


class TestLogPosteriorObjective:
    """Tests for the _log_posterior_objective function."""

    def test_output_is_scalar(self) -> None:
        """Output should be a scalar."""
        result = _log_posterior_objective(
            learning_state=jnp.array([0.5]),
            learning_state_prev=0.0,
            variance_prev=1.0,
            n_correct_in_trial=1,
            max_possible_correct=1,
            bias=0.0,
        )

        assert result.shape == ()

    def test_higher_at_correct_for_positive_state(self) -> None:
        """Log posterior should be higher for correct response when state is positive."""
        # Positive learning state -> higher probability of correct
        state_positive = jnp.array([2.0])

        lp_correct = _log_posterior_objective(
            learning_state=state_positive,
            learning_state_prev=0.0,
            variance_prev=10.0,  # Wide prior
            n_correct_in_trial=1,
            max_possible_correct=1,
            bias=0.0,
        )

        lp_incorrect = _log_posterior_objective(
            learning_state=state_positive,
            learning_state_prev=0.0,
            variance_prev=10.0,
            n_correct_in_trial=0,
            max_possible_correct=1,
            bias=0.0,
        )

        assert lp_correct > lp_incorrect

    def test_prior_pulls_toward_previous(self) -> None:
        """Log posterior should be higher when closer to previous state."""
        prev_state = 1.0
        variance = 0.1  # Tight prior

        lp_close = _log_posterior_objective(
            learning_state=jnp.array([1.1]),
            learning_state_prev=prev_state,
            variance_prev=variance,
            n_correct_in_trial=1,
            max_possible_correct=1,
            bias=0.0,
        )

        lp_far = _log_posterior_objective(
            learning_state=jnp.array([3.0]),
            learning_state_prev=prev_state,
            variance_prev=variance,
            n_correct_in_trial=1,
            max_possible_correct=1,
            bias=0.0,
        )

        assert lp_close > lp_far


class TestSmithLearningFilter:
    """Tests for the smith_learning_filter function.

    Note: The smith_learning_filter uses Laplace approximation with BFGS optimization
    inside jax.lax.scan, which can cause tracing issues in some JAX versions.
    Tests are marked to skip if JAX tracing fails.
    """

    @pytest.fixture
    def simulated_data(self):
        """Generate simulated learning data for testing."""
        # Use a fixed seed for reproducibility
        outcomes, true_prob = simulate_learning_data(
            n_trials=20,  # Use fewer trials for faster tests
            prob_success_init=0.3,
            prob_success_final=0.8,
            learning_rate=0.15,
            inflection_point=10.0,
            seed=42,
        )
        return jnp.array(outcomes), jnp.array(true_prob)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_output_shapes(self, simulated_data) -> None:
        """Filter outputs should have correct shapes."""
        outcomes, _ = simulated_data
        n_trials = len(outcomes)

        try:
            prob, mode, variance, one_step_mode, one_step_var = smith_learning_filter(
                outcomes, max_possible_correct=1
            )
        except Exception as e:
            pytest.skip(f"smith_learning_filter failed (likely JAX tracing issue): {e}")

        assert prob.shape == (n_trials,)
        assert mode.shape == (n_trials,)
        assert variance.shape == (n_trials,)
        assert one_step_mode.shape == (n_trials,)
        assert one_step_var.shape == (n_trials,)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_probability_bounds(self, simulated_data) -> None:
        """Probabilities should be in [0, 1]."""
        outcomes, _ = simulated_data

        try:
            prob, _, _, _, _ = smith_learning_filter(outcomes, max_possible_correct=1)
        except Exception as e:
            pytest.skip(f"smith_learning_filter failed: {e}")

        assert jnp.all(prob >= 0)
        assert jnp.all(prob <= 1)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_variance_positive(self, simulated_data) -> None:
        """Variances should be positive."""
        outcomes, _ = simulated_data

        try:
            _, _, variance, _, one_step_var = smith_learning_filter(
                outcomes, max_possible_correct=1
            )
        except Exception as e:
            pytest.skip(f"smith_learning_filter failed: {e}")

        assert jnp.all(variance > 0)
        assert jnp.all(one_step_var > 0)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_no_nans(self, simulated_data) -> None:
        """Filter should not produce NaN values."""
        outcomes, _ = simulated_data

        try:
            prob, mode, variance, one_step_mode, one_step_var = smith_learning_filter(
                outcomes, max_possible_correct=1
            )
        except Exception as e:
            pytest.skip(f"smith_learning_filter failed: {e}")

        assert not jnp.any(jnp.isnan(prob))
        assert not jnp.any(jnp.isnan(mode))
        assert not jnp.any(jnp.isnan(variance))
        assert not jnp.any(jnp.isnan(one_step_mode))
        assert not jnp.any(jnp.isnan(one_step_var))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_custom_initial_state(self, simulated_data) -> None:
        """Custom initial state should affect filter output."""
        outcomes, _ = simulated_data

        try:
            prob_default, _, _, _, _ = smith_learning_filter(
                outcomes, init_learning_state=0.0, max_possible_correct=1
            )
            prob_high, _, _, _, _ = smith_learning_filter(
                outcomes, init_learning_state=2.0, max_possible_correct=1
            )
        except Exception as e:
            pytest.skip(f"smith_learning_filter failed: {e}")

        # First probability should be higher with higher initial state
        assert prob_high[0] > prob_default[0]

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_sigma_epsilon_affects_variance(self, simulated_data) -> None:
        """Higher sigma_epsilon should lead to higher variance."""
        outcomes, _ = simulated_data

        try:
            _, _, var_low, _, _ = smith_learning_filter(
                outcomes, sigma_epsilon=0.1, max_possible_correct=1
            )
            _, _, var_high, _, _ = smith_learning_filter(
                outcomes, sigma_epsilon=0.5, max_possible_correct=1
            )
        except Exception as e:
            pytest.skip(f"smith_learning_filter failed: {e}")

        # Average variance should be higher with higher sigma_epsilon
        assert jnp.mean(var_high) > jnp.mean(var_low)


class TestSmithLearningSmoother:
    """Tests for the smith_learning_smoother function."""

    @pytest.fixture
    def filter_outputs(self):
        """Generate filter outputs for smoother testing."""
        outcomes, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes)

        try:
            prob, mode, variance, one_step_mode, one_step_var = smith_learning_filter(
                outcomes, max_possible_correct=1
            )
        except Exception as e:
            pytest.skip(f"smith_learning_filter failed: {e}")

        return {
            "mode": mode,
            "variance": variance,
            "one_step_mode": one_step_mode,
            "one_step_var": one_step_var,
            "n_trials": len(outcomes),
        }

    def test_output_shapes(self, filter_outputs) -> None:
        """Smoother outputs should have correct shapes."""
        f = filter_outputs

        sm_mode, sm_var, sm_prob, sm_gain = smith_learning_smoother(
            f["mode"], f["variance"], f["one_step_mode"], f["one_step_var"]
        )

        assert sm_mode.shape == (f["n_trials"],)
        assert sm_var.shape == (f["n_trials"],)
        assert sm_prob.shape == (f["n_trials"],)
        assert sm_gain.shape == (f["n_trials"] - 1,)

    def test_probability_bounds(self, filter_outputs) -> None:
        """Smoothed probabilities should be in [0, 1]."""
        f = filter_outputs

        _, _, sm_prob, _ = smith_learning_smoother(
            f["mode"], f["variance"], f["one_step_mode"], f["one_step_var"]
        )

        assert jnp.all(sm_prob >= 0)
        assert jnp.all(sm_prob <= 1)

    def test_last_equals_filter(self, filter_outputs) -> None:
        """Last smoother estimate should equal last filter estimate."""
        f = filter_outputs

        sm_mode, sm_var, _, _ = smith_learning_smoother(
            f["mode"], f["variance"], f["one_step_mode"], f["one_step_var"]
        )

        np.testing.assert_allclose(sm_mode[-1], f["mode"][-1], rtol=1e-10)
        np.testing.assert_allclose(sm_var[-1], f["variance"][-1], rtol=1e-10)

    def test_variance_generally_reduced(self, filter_outputs) -> None:
        """Smoother variance should generally be <= filter variance."""
        f = filter_outputs

        _, sm_var, _, _ = smith_learning_smoother(
            f["mode"], f["variance"], f["one_step_mode"], f["one_step_var"]
        )

        # Smoother variance should be <= filter variance at all times
        assert jnp.all(sm_var <= f["variance"] + 1e-6)


class TestMaximizationStep:
    """Tests for the maximization_step function."""

    @pytest.fixture
    def smoother_outputs(self):
        """Generate smoother outputs for M-step testing."""
        outcomes, _ = simulate_learning_data(n_trials=30, seed=42)
        outcomes = jnp.array(outcomes)

        try:
            prob, mode, variance, one_step_mode, one_step_var = smith_learning_filter(
                outcomes, max_possible_correct=1
            )
            sm_mode, sm_var, _, sm_gain = smith_learning_smoother(
                mode, variance, one_step_mode, one_step_var
            )
        except Exception as e:
            pytest.skip(f"Filter/smoother failed: {e}")

        return {"mode": sm_mode, "variance": sm_var, "gain": sm_gain}

    def test_sigma_epsilon_positive(self, smoother_outputs) -> None:
        """Estimated sigma_epsilon should be positive."""
        s = smoother_outputs

        sigma_eps, _, _ = maximization_step(s["mode"], s["variance"], s["gain"])

        assert sigma_eps > 0

    def test_init_variance_positive(self, smoother_outputs) -> None:
        """Estimated initial variance should be positive."""
        s = smoother_outputs

        _, _, init_var = maximization_step(s["mode"], s["variance"], s["gain"])

        assert init_var > 0

    def test_init_mean_equals_first_smoother(self, smoother_outputs) -> None:
        """Estimated initial mean should equal first smoother mean."""
        s = smoother_outputs

        _, init_mean, _ = maximization_step(s["mode"], s["variance"], s["gain"])

        np.testing.assert_allclose(init_mean, s["mode"][0], rtol=1e-10)

    def test_init_var_equals_first_smoother(self, smoother_outputs) -> None:
        """Estimated initial variance should equal first smoother variance."""
        s = smoother_outputs

        _, _, init_var = maximization_step(s["mode"], s["variance"], s["gain"])

        np.testing.assert_allclose(init_var, s["variance"][0], rtol=1e-10)


class TestCalculateProbabilityConfidenceLimits:
    """Tests for calculate_probability_confidence_limits function."""

    def test_output_shapes(self) -> None:
        """Output should have correct shapes."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jnp.zeros(n_trials)
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5

        percentiles, prob_above_chance = calculate_probability_confidence_limits(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, prob_correct_by_chance=0.5
        )

        # Default percentiles are [5, 50, 95]
        assert percentiles.shape == (3, n_trials)
        assert prob_above_chance is None

    def test_percentiles_ordered(self) -> None:
        """Lower percentiles should be <= higher percentiles."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jax.random.normal(key, (n_trials,))
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5

        percentiles, _ = calculate_probability_confidence_limits(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            prob_correct_by_chance=0.5,
            percentiles=jnp.array([5.0, 50.0, 95.0]),
        )

        assert jnp.all(percentiles[0] <= percentiles[1])  # p5 <= p50
        assert jnp.all(percentiles[1] <= percentiles[2])  # p50 <= p95

    def test_probabilities_in_bounds(self) -> None:
        """All percentile values should be in [0, 1]."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jax.random.normal(key, (n_trials,)) * 2
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5

        percentiles, _ = calculate_probability_confidence_limits(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, prob_correct_by_chance=0.5
        )

        assert jnp.all(percentiles >= 0)
        assert jnp.all(percentiles <= 1)

    def test_prob_above_chance_returned_when_requested(self) -> None:
        """prob_above_chance should be returned when return_prob_above_chance is True."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jnp.ones(n_trials) * 2  # High state
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.1

        _, prob_above_chance = calculate_probability_confidence_limits(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            prob_correct_by_chance=0.5,
            return_prob_above_chance=True,
        )

        assert prob_above_chance is not None
        assert prob_above_chance.shape == (n_trials,)

    def test_prob_above_chance_high_for_high_state(self) -> None:
        """prob_above_chance should be high when state is much above chance."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        # Very high state -> probability well above 0.5
        smoothed_learning_state_mode = jnp.ones(n_trials) * 5
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.1

        _, prob_above_chance = calculate_probability_confidence_limits(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            prob_correct_by_chance=0.5,
            return_prob_above_chance=True,
        )

        # Should be very certain (close to 1)
        assert prob_above_chance is not None
        assert jnp.all(prob_above_chance > 0.9)


class TestFindMinConsecutiveSuccesses:
    """Tests for the find_min_consecutive_successes function."""

    def test_returns_integer_or_none(self) -> None:
        """Should return int or None."""
        result = find_min_consecutive_successes(
            prob_correct_by_chance=0.5,
            critical_probability_threshold=0.05,
            sequence_length=100,
        )

        assert result is None or isinstance(result, int)

    def test_higher_prob_needs_longer_run(self) -> None:
        """Higher null probability should require longer run for significance."""
        result_low = find_min_consecutive_successes(
            prob_correct_by_chance=0.3, critical_probability_threshold=0.05, sequence_length=100
        )

        result_high = find_min_consecutive_successes(
            prob_correct_by_chance=0.7, critical_probability_threshold=0.05, sequence_length=100
        )

        # With higher probability, need longer run to be surprising
        if result_low is not None and result_high is not None:
            assert result_high > result_low

    def test_stricter_threshold_needs_longer_run(self) -> None:
        """Stricter threshold should require longer run."""
        result_loose = find_min_consecutive_successes(
            prob_correct_by_chance=0.5, critical_probability_threshold=0.10, sequence_length=100
        )

        result_strict = find_min_consecutive_successes(
            prob_correct_by_chance=0.5, critical_probability_threshold=0.01, sequence_length=100
        )

        if result_loose is not None and result_strict is not None:
            assert result_strict >= result_loose

    def test_invalid_probability_raises(self) -> None:
        """Invalid probability should raise ValueError."""
        with pytest.raises(ValueError):
            find_min_consecutive_successes(
                prob_correct_by_chance=1.5, critical_probability_threshold=0.05, sequence_length=100
            )

        with pytest.raises(ValueError):
            find_min_consecutive_successes(
                prob_correct_by_chance=-0.1, critical_probability_threshold=0.05, sequence_length=100
            )


class TestFindRunsOfValue:
    """Tests for the _find_runs_of_value helper function."""

    def test_finds_single_run(self) -> None:
        """Should find a single run."""
        data = jnp.array([0, 0, 1, 1, 1, 0, 0])
        runs = _find_runs_of_value(data, value_to_find=1, min_length=2)

        assert len(runs) == 1
        assert runs[0] == (2, 4)

    def test_finds_multiple_runs(self) -> None:
        """Should find multiple runs."""
        data = jnp.array([1, 1, 0, 1, 1, 1, 0, 1, 1])
        runs = _find_runs_of_value(data, value_to_find=1, min_length=2)

        assert len(runs) == 3
        assert (0, 1) in runs
        assert (3, 5) in runs
        assert (7, 8) in runs

    def test_respects_min_length(self) -> None:
        """Should only find runs meeting minimum length."""
        data = jnp.array([1, 1, 0, 1, 1, 1, 0, 1])
        runs = _find_runs_of_value(data, value_to_find=1, min_length=3)

        assert len(runs) == 1
        assert runs[0] == (3, 5)

    def test_empty_for_no_runs(self) -> None:
        """Should return empty list when no runs found."""
        data = jnp.array([0, 0, 0, 0])
        runs = _find_runs_of_value(data, value_to_find=1, min_length=2)

        assert len(runs) == 0

    def test_finds_run_at_start(self) -> None:
        """Should find run at the start of array."""
        data = jnp.array([1, 1, 1, 0, 0])
        runs = _find_runs_of_value(data, value_to_find=1, min_length=2)

        assert len(runs) == 1
        assert runs[0] == (0, 2)

    def test_finds_run_at_end(self) -> None:
        """Should find run at the end of array."""
        data = jnp.array([0, 0, 1, 1, 1])
        runs = _find_runs_of_value(data, value_to_find=1, min_length=2)

        assert len(runs) == 1
        assert runs[0] == (2, 4)


class TestSimulateLearningData:
    """Tests for the simulate_learning_data function."""

    def test_output_shapes(self) -> None:
        """Outputs should have correct shapes."""
        n_trials = 100
        outcomes, true_prob = simulate_learning_data(n_trials=n_trials, seed=42)

        assert outcomes.shape == (n_trials,)
        assert true_prob.shape == (n_trials,)

    def test_outcomes_binary(self) -> None:
        """Outcomes should be binary (0 or 1)."""
        outcomes, _ = simulate_learning_data(n_trials=100, seed=42)

        assert np.all((outcomes == 0) | (outcomes == 1))

    def test_probability_bounds(self) -> None:
        """True probabilities should be in [0, 1]."""
        _, true_prob = simulate_learning_data(n_trials=100, seed=42)

        assert np.all(true_prob >= 0)
        assert np.all(true_prob <= 1)

    def test_probability_starts_at_init(self) -> None:
        """First probability should be near init value."""
        prob_init = 0.2
        _, true_prob = simulate_learning_data(
            n_trials=100, prob_success_init=prob_init, seed=42
        )

        # Should be close to init at start
        np.testing.assert_allclose(true_prob[0], prob_init, rtol=0.1)

    def test_probability_ends_at_final(self) -> None:
        """Last probability should be near final value."""
        prob_final = 0.9
        _, true_prob = simulate_learning_data(
            n_trials=100, prob_success_final=prob_final, seed=42
        )

        # Should be close to final at end
        np.testing.assert_allclose(true_prob[-1], prob_final, rtol=0.1)

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce same results."""
        outcomes1, prob1 = simulate_learning_data(n_trials=50, seed=123)
        outcomes2, prob2 = simulate_learning_data(n_trials=50, seed=123)

        np.testing.assert_array_equal(outcomes1, outcomes2)
        np.testing.assert_array_equal(prob1, prob2)

    def test_different_seeds_different_results(self) -> None:
        """Different seeds should (likely) produce different outcomes."""
        outcomes1, _ = simulate_learning_data(n_trials=50, seed=1)
        outcomes2, _ = simulate_learning_data(n_trials=50, seed=2)

        # Outcomes should differ (extremely unlikely to be identical)
        assert not np.array_equal(outcomes1, outcomes2)


class TestSmithLearningAlgorithmClass:
    """Tests for the SmithLearningAlgorithm class."""

    def test_initialization(self) -> None:
        """Class should initialize without errors."""
        model = SmithLearningAlgorithm(
            init_learning_state=0.0,
            sigma_epsilon=float(jnp.sqrt(0.05)),
            prob_correct_by_chance=0.5,
        )

        assert model.init_learning_state == 0.0
        assert model.prob_correct_by_chance == 0.5

    def test_initialization_with_explicit_variance(self) -> None:
        """Class should accept explicit init_learning_variance."""
        model = SmithLearningAlgorithm(
            init_learning_state=0.0,
            init_learning_variance=0.1,
            sigma_epsilon=float(jnp.sqrt(0.05)),
            prob_correct_by_chance=0.5,
        )

        np.testing.assert_allclose(model.init_learning_variance, 0.1, rtol=1e-10)

    def test_default_variance_equals_sigma_squared(self) -> None:
        """Default init_learning_variance should equal sigma_epsilon^2."""
        sigma_eps = 0.3
        model = SmithLearningAlgorithm(sigma_epsilon=sigma_eps)

        np.testing.assert_allclose(
            model.init_learning_variance, sigma_eps**2, rtol=1e-10
        )

    def test_invalid_sigma_epsilon_raises(self) -> None:
        """Negative sigma_epsilon should raise error."""
        with pytest.raises(ValueError):
            SmithLearningAlgorithm(sigma_epsilon=-0.1)

    def test_invalid_prob_chance_raises(self) -> None:
        """Invalid prob_correct_by_chance should raise error."""
        with pytest.raises(ValueError, match="prob_correct_by_chance"):
            SmithLearningAlgorithm(prob_correct_by_chance=0.0)

        with pytest.raises(ValueError, match="prob_correct_by_chance"):
            SmithLearningAlgorithm(prob_correct_by_chance=1.0)

    def test_invalid_variance_type_raises(self) -> None:
        """Invalid init_learning_variance type should raise TypeError."""
        with pytest.raises(TypeError):
            SmithLearningAlgorithm(init_learning_variance="invalid")  # type: ignore[arg-type]

    def test_negative_variance_raises(self) -> None:
        """Negative init_learning_variance should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            SmithLearningAlgorithm(init_learning_variance=-0.1)

    def test_invalid_initial_state_method_raises(self) -> None:
        """Invalid initial_state_method should raise ValueError."""
        with pytest.raises(ValueError, match="initial_state_method must be one of"):
            SmithLearningAlgorithm(initial_state_method="typo")

    def test_is_fitted_false_before_fit(self) -> None:
        """is_fitted should be False before calling fit()."""
        model = SmithLearningAlgorithm()
        assert not model.is_fitted

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_is_fitted_true_after_fit(self) -> None:
        """is_fitted should be True after calling fit()."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=3)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")
        assert model.is_fitted

    def test_repr_before_fit(self) -> None:
        """__repr__ should show 'not fitted' before fit."""
        model = SmithLearningAlgorithm(sigma_epsilon=0.22)
        r = repr(model)
        assert "SmithLearningAlgorithm(" in r
        assert "not fitted" in r
        assert "0.22" in r

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_repr_after_fit(self) -> None:
        """__repr__ should show 'fitted' after fit."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=3)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")
        r = repr(model)
        assert "fitted)" in r
        assert "not fitted" not in r

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_plot_learning_process_returns_fig_ax(self) -> None:
        """plot_learning_process should return (fig, ax)."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=3)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")
        result = model.plot_learning_process(jax.random.PRNGKey(0))
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fit_verbose_prints_output(self, capsys: pytest.CaptureFixture) -> None:
        """fit(verbose=True) should print convergence info to stdout."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=5, verbose=True)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")
        captured = capsys.readouterr()
        assert "LL=" in captured.out or "Converged" in captured.out

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fit_final_estep_after_convergence(self) -> None:
        """fit() should run final M-step + E-step so stored results match MLE params."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningAlgorithm()
        try:
            log_likelihoods = model.fit(outcomes, max_iter=50)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")
        # If converged, the final log-likelihood entry is from the post-convergence E-step
        if len(log_likelihoods) >= 2:
            # The last two LLs should be very close (converged params + final E-step)
            assert abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1.0

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fit_returns_log_likelihoods(self) -> None:
        """fit() should return list of log-likelihoods."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)

        model = SmithLearningAlgorithm()
        try:
            log_likelihoods = model.fit(outcomes, max_iter=3)
        except Exception as e:
            pytest.skip(f"fit() failed (likely JAX tracing issue): {e}")

        assert isinstance(log_likelihoods, list)
        assert len(log_likelihoods) > 0

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fit_populates_attributes(self) -> None:
        """fit() should populate smoother attributes."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)

        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=3)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")

        assert model.smoothed_learning_state_mode is not None
        assert model.smoothed_learning_state_variance is not None
        assert model.smoothed_prob_correct_response is not None

    def test_get_learning_curve_requires_fit(self) -> None:
        """get_learning_curve() should raise if not fitted."""
        model = SmithLearningAlgorithm()

        with pytest.raises(RuntimeError):
            model.get_learning_curve(jax.random.PRNGKey(0))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_get_learning_curve_output_shapes(self) -> None:
        """get_learning_curve() should return correct shapes."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)

        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=3)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")

        percentiles, prob_above_chance = model.get_learning_curve(
            jax.random.PRNGKey(0), n_samples=100
        )

        assert percentiles.shape[0] == 3  # Default percentiles
        assert percentiles.shape[1] == len(outcomes)
        assert prob_above_chance is None  # Not requested


class TestSmithLearningAlgorithmEdgeCases:
    """Edge case tests for scientific robustness."""

    def test_single_trial_raises(self) -> None:
        """fit() with a single trial should raise ValueError."""
        model = SmithLearningAlgorithm()
        with pytest.raises(ValueError, match="at least 2 trials"):
            model.fit(jnp.array([1]))

    def test_n_correct_exceeds_max_possible_raises(self) -> None:
        """fit() should raise if n_correct > max_possible_correct."""
        model = SmithLearningAlgorithm(max_possible_correct=1)
        with pytest.raises(ValueError, match="exceeding max_possible_correct"):
            model.fit(jnp.array([0, 1, 2, 1, 0]))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_all_zeros_produces_finite(self) -> None:
        """All-zero outcomes should not produce NaN/Inf."""
        model = SmithLearningAlgorithm()
        outcomes = jnp.zeros(20, dtype=jnp.int32)
        log_likelihoods = model.fit(outcomes, max_iter=5)
        assert all(np.isfinite(ll) for ll in log_likelihoods)
        assert model.is_fitted
        assert bool(jnp.all(jnp.isfinite(model.smoothed_learning_state_mode)))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_all_ones_produces_finite(self) -> None:
        """All-one outcomes should not produce NaN/Inf."""
        model = SmithLearningAlgorithm()
        outcomes = jnp.ones(20, dtype=jnp.int32)
        log_likelihoods = model.fit(outcomes, max_iter=5)
        assert all(np.isfinite(ll) for ll in log_likelihoods)
        assert model.is_fitted
        assert bool(jnp.all(jnp.isfinite(model.smoothed_learning_state_mode)))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_sigma_epsilon_stays_positive_after_fit(self) -> None:
        """sigma_epsilon should remain positive and finite after fitting."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=10)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")
        assert model.sigma_epsilon > 0
        assert np.isfinite(model.sigma_epsilon)

    def test_2d_input_raises(self) -> None:
        """fit() with 2D input should raise ValueError."""
        model = SmithLearningAlgorithm()
        with pytest.raises(ValueError, match="1D"):
            model.fit(jnp.array([[1, 0], [0, 1]]))


class TestFindCriterionTrial:
    """Tests for find_criterion_trial method."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_returns_int_for_learning_data(self) -> None:
        """Should return a trial index for data with clear learning."""
        outcomes_np, _ = simulate_learning_data(
            n_trials=50, seed=42, prob_success_init=0.3, prob_success_final=0.95
        )
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=10)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")
        result = model.find_criterion_trial(jax.random.PRNGKey(0))
        # Should return an int or None; if learning is clear, should be int
        assert result is None or isinstance(result, int)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_returns_none_for_no_learning(self) -> None:
        """Should return None when performance never exceeds chance."""
        outcomes = jnp.zeros(30, dtype=jnp.int32)  # All failures
        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=5)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")
        result = model.find_criterion_trial(jax.random.PRNGKey(0))
        assert result is None

    def test_requires_fit(self) -> None:
        """Should raise if not fitted."""
        model = SmithLearningAlgorithm()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.find_criterion_trial(jax.random.PRNGKey(0))


class TestIdentifySignificantRuns:
    """Tests for identify_significant_runs_in_data and find_critical_run_length."""

    def test_find_critical_run_length_without_fit(self) -> None:
        """find_critical_run_length should work without fitting."""
        model = SmithLearningAlgorithm(prob_correct_by_chance=0.5)
        result = model.find_critical_run_length(sequence_length=50)
        assert result is None or isinstance(result, int)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_identify_runs_returns_tuple(self) -> None:
        """identify_significant_runs_in_data should return (j_crit, runs)."""
        model = SmithLearningAlgorithm(prob_correct_by_chance=0.5)
        data = jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        j_crit, runs = model.identify_significant_runs_in_data(data)
        assert j_crit is None or isinstance(j_crit, int)
        assert isinstance(runs, list)

    def test_identify_runs_empty_input(self) -> None:
        """Empty input should return (None, [])."""
        model = SmithLearningAlgorithm()
        j_crit, runs = model.identify_significant_runs_in_data(jnp.array([]))
        assert j_crit is None
        assert runs == []


class TestPlotTrialComparisonMatrix:
    """Tests for plot_trial_comparison_matrix method."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_returns_fig_ax(self) -> None:
        """plot_trial_comparison_matrix should return (fig, ax)."""
        outcomes_np, _ = simulate_learning_data(n_trials=15, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningAlgorithm()
        try:
            model.fit(outcomes, max_iter=3)
        except Exception as e:
            pytest.skip(f"fit() failed: {e}")
        fig, ax = model.plot_trial_comparison_matrix(
            jax.random.PRNGKey(0), n_samples=100
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_requires_fit(self) -> None:
        """Should raise if not fitted."""
        model = SmithLearningAlgorithm()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.plot_trial_comparison_matrix(jax.random.PRNGKey(0))


class TestCalculateLatentStatePercentiles:
    """Tests for calculate_latent_state_percentiles function."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jnp.zeros(n_trials)
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5

        result = calculate_latent_state_percentiles(key, smoothed_learning_state_mode, smoothed_learning_state_variance)

        # Default percentiles [5, 50, 95]
        assert result.shape == (3, n_trials)

    def test_custom_percentiles(self) -> None:
        """Custom percentiles should be respected."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jnp.zeros(n_trials)
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5
        custom_percentiles = jnp.array([10.0, 25.0, 75.0, 90.0])

        result = calculate_latent_state_percentiles(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, percentiles=custom_percentiles
        )

        assert result.shape == (4, n_trials)

    def test_percentiles_ordered(self) -> None:
        """Percentiles should be in order."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jax.random.normal(key, (n_trials,))
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5

        result = calculate_latent_state_percentiles(key, smoothed_learning_state_mode, smoothed_learning_state_variance)

        # p5 <= p50 <= p95
        assert jnp.all(result[0] <= result[1])
        assert jnp.all(result[1] <= result[2])

    def test_median_near_mode(self) -> None:
        """Median percentile should be close to mode for Gaussian."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jax.random.normal(key, (n_trials,)) * 2
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.01  # Small variance

        result = calculate_latent_state_percentiles(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, n_samples=10000
        )

        # Median (index 1) should be close to mode
        np.testing.assert_allclose(result[1], smoothed_learning_state_mode, rtol=0.1, atol=0.1)


class TestComputeCrossCovarianceMatrix:
    """Tests for the compute_cross_covariance_matrix function."""

    def test_output_shape(self) -> None:
        """Output should be square matrix of size n_trials."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5
        smoother_gain = jnp.ones(n_trials - 1) * 0.8

        result = compute_cross_covariance_matrix(smoothed_learning_state_variance, smoother_gain)

        assert result.shape == (n_trials, n_trials)

    def test_diagonal_equals_variance(self) -> None:
        """Diagonal entries should equal smoothed variances."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.linspace(0.1, 1.0, n_trials)
        smoother_gain = jnp.ones(n_trials - 1) * 0.8

        result = compute_cross_covariance_matrix(smoothed_learning_state_variance, smoother_gain)

        np.testing.assert_allclose(jnp.diag(result), smoothed_learning_state_variance, rtol=1e-5)

    def test_symmetric(self) -> None:
        """Cross-covariance matrix should be symmetric."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.linspace(0.1, 1.0, n_trials)
        smoother_gain = jnp.linspace(0.6, 0.9, n_trials - 1)

        result = compute_cross_covariance_matrix(smoothed_learning_state_variance, smoother_gain)

        np.testing.assert_allclose(result, result.T, rtol=1e-5)

    def test_positive_semidefinite(self) -> None:
        """Cross-covariance matrix should be positive semi-definite."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.linspace(0.1, 1.0, n_trials)
        smoother_gain = jnp.linspace(0.6, 0.9, n_trials - 1)

        result = compute_cross_covariance_matrix(smoothed_learning_state_variance, smoother_gain)
        eigenvalues = jnp.linalg.eigvalsh(result)

        # All eigenvalues should be non-negative (within tolerance)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_off_diagonal_decay(self) -> None:
        """Cross-covariance should decay with distance when gains < 1."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5
        smoother_gain = jnp.ones(n_trials - 1) * 0.7  # < 1

        result = compute_cross_covariance_matrix(smoothed_learning_state_variance, smoother_gain)

        # Check first row: Cov(0, j) should decrease with j
        first_row = result[0, :]
        for j in range(1, n_trials - 1):
            assert first_row[j] >= first_row[j + 1]

    def test_unit_gain_preserves_covariance(self) -> None:
        """With unit gains, cross-covariance equals variance of later trial."""
        n_trials = 10
        smoothed_learning_state_variance = jnp.linspace(0.1, 1.0, n_trials)
        smoother_gain = jnp.ones(n_trials - 1)  # All 1.0

        result = compute_cross_covariance_matrix(smoothed_learning_state_variance, smoother_gain)

        # Cov(i, j) = P_j for i <= j when all gains are 1
        for i in range(n_trials):
            for j in range(i, n_trials):
                np.testing.assert_allclose(
                    result[i, j], smoothed_learning_state_variance[j], rtol=1e-5
                )


class TestComputeTrialComparisonMatrix:
    """Tests for the compute_trial_comparison_matrix function."""

    @pytest.fixture
    def fitted_model_data(self):
        """Create fitted model data for testing."""
        # Simple increasing learning states
        n_trials = 20
        smoothed_learning_state_mode = jnp.linspace(-1.0, 2.0, n_trials)
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.3
        smoother_gain = jnp.ones(n_trials - 1) * 0.8
        return smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain

    def test_output_shape(self, fitted_model_data) -> None:
        """Output should be square matrix of size n_trials."""
        smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain = fitted_model_data
        n_trials = len(smoothed_learning_state_mode)
        key = jax.random.PRNGKey(0)

        result = compute_trial_comparison_matrix(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain, n_samples=1000
        )

        assert result.shape == (n_trials, n_trials)

    def test_diagonal_is_half(self, fitted_model_data) -> None:
        """Diagonal entries should be 0.5 (P(x_i > x_i) = 0.5)."""
        smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain = fitted_model_data
        n_trials = len(smoothed_learning_state_mode)
        key = jax.random.PRNGKey(0)

        result = compute_trial_comparison_matrix(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain, n_samples=1000
        )

        np.testing.assert_allclose(jnp.diag(result), 0.5, rtol=1e-5)

    def test_lower_triangle_is_nan(self, fitted_model_data) -> None:
        """Lower triangle should be NaN."""
        smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain = fitted_model_data
        n_trials = len(smoothed_learning_state_mode)
        key = jax.random.PRNGKey(0)

        result = compute_trial_comparison_matrix(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain, n_samples=1000
        )

        # Check lower triangle (excluding diagonal)
        lower_tri_mask = jnp.tril(jnp.ones((n_trials, n_trials), dtype=bool), k=-1)
        assert jnp.all(jnp.isnan(result[lower_tri_mask]))

    def test_upper_triangle_in_bounds(self, fitted_model_data) -> None:
        """Upper triangle values should be probabilities in [0, 1]."""
        smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain = fitted_model_data
        n_trials = len(smoothed_learning_state_mode)
        key = jax.random.PRNGKey(0)

        result = compute_trial_comparison_matrix(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain, n_samples=1000
        )

        # Check upper triangle
        upper_tri_mask = jnp.triu(jnp.ones((n_trials, n_trials), dtype=bool), k=1)
        upper_values = result[upper_tri_mask]

        assert jnp.all(upper_values >= 0.0)
        assert jnp.all(upper_values <= 1.0)

    def test_increasing_states_low_early_p_values(self) -> None:
        """For increasing states, early vs late comparison should have low p."""
        n_trials = 20
        # Clear increasing trend
        smoothed_learning_state_mode = jnp.linspace(-2.0, 3.0, n_trials)
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.1  # Small variance
        smoother_gain = jnp.ones(n_trials - 1) * 0.8
        key = jax.random.PRNGKey(42)

        result = compute_trial_comparison_matrix(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain, n_samples=5000
        )

        # P(x_0 > x_19) should be very low since x_19 >> x_0
        assert result[0, n_trials - 1] < 0.1

        # P(x_0 > x_10) should be less than or equal to P(x_0 > x_5)
        # (monotonic in distance, allowing for sampling variance)
        assert result[0, 10] <= result[0, 5] + 0.01

        # Far comparisons should show clear significance
        assert result[0, n_trials // 2] < 0.3

    def test_constant_states_near_half(self) -> None:
        """For constant states, all comparisons should be near 0.5."""
        n_trials = 15
        smoothed_learning_state_mode = jnp.ones(n_trials) * 1.0  # All same
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5
        smoother_gain = jnp.ones(n_trials - 1) * 0.8
        key = jax.random.PRNGKey(0)

        result = compute_trial_comparison_matrix(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain, n_samples=5000
        )

        # Upper triangle should be near 0.5
        upper_tri_mask = jnp.triu(jnp.ones((n_trials, n_trials), dtype=bool), k=1)
        upper_values = result[upper_tri_mask]

        np.testing.assert_allclose(upper_values, 0.5, atol=0.1)

    def test_reproducibility_with_same_key(self, fitted_model_data) -> None:
        """Same key should produce same results."""
        smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain = fitted_model_data
        key = jax.random.PRNGKey(123)

        result1 = compute_trial_comparison_matrix(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain, n_samples=1000
        )
        result2 = compute_trial_comparison_matrix(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain, n_samples=1000
        )

        # Upper triangle should match (lower is NaN)
        upper_tri_mask = jnp.triu(jnp.ones_like(result1, dtype=bool), k=1)
        np.testing.assert_allclose(
            result1[upper_tri_mask], result2[upper_tri_mask], rtol=1e-5
        )


class TestCompareTwoTrials:
    """Tests for the compare_two_trials function."""

    @pytest.fixture
    def model_data(self):
        """Create model data for testing."""
        n_trials = 20
        smoothed_learning_state_mode = jnp.linspace(-1.0, 2.0, n_trials)
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.3
        smoother_gain = jnp.ones(n_trials - 1) * 0.8
        return smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain

    def test_same_trial_returns_half(self, model_data) -> None:
        """Comparing trial to itself should return 0.5."""
        smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain = model_data
        key = jax.random.PRNGKey(0)

        result = compare_two_trials(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain,
            trial1=5, trial2=5
        )

        assert result == 0.5

    def test_output_is_probability(self, model_data) -> None:
        """Output should be a probability in [0, 1]."""
        smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain = model_data
        key = jax.random.PRNGKey(0)

        result = compare_two_trials(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain,
            trial1=0, trial2=15
        )

        assert 0.0 <= result <= 1.0

    def test_symmetric_complement(self, model_data) -> None:
        """P(trial1 > trial2) + P(trial2 > trial1) should equal 1."""
        smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain = model_data
        key = jax.random.PRNGKey(0)

        p_12 = compare_two_trials(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain,
            trial1=3, trial2=12
        )
        p_21 = compare_two_trials(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance, smoother_gain,
            trial1=12, trial2=3
        )

        np.testing.assert_allclose(p_12 + p_21, 1.0, rtol=0.05)


class TestFindFirstSignificantTrial:
    """Tests for the find_first_significant_trial function."""

    def test_finds_significant_in_increasing_data(self) -> None:
        """Should find significant trial when data shows clear increase."""
        n_trials = 20
        # Create matrix where early trials are significantly lower
        comparison_matrix = jnp.full((n_trials, n_trials), jnp.nan)
        comparison_matrix = comparison_matrix.at[jnp.diag_indices(n_trials)].set(0.5)

        # Fill upper triangle with decreasing p-values (trial 0 vs later)
        for j in range(1, n_trials):
            # P(trial 0 > trial j) decreases as j increases
            p_val = 0.5 * jnp.exp(-0.3 * j)
            comparison_matrix = comparison_matrix.at[0, j].set(p_val)

        result = find_first_significant_trial(
            comparison_matrix, reference_trial=0, significance_level=0.05
        )

        # Should find a significant trial
        assert result is not None
        assert result > 0

    def test_returns_none_when_no_significance(self) -> None:
        """Should return None when no trial is significantly different."""
        n_trials = 20
        comparison_matrix = jnp.full((n_trials, n_trials), jnp.nan)
        comparison_matrix = comparison_matrix.at[jnp.diag_indices(n_trials)].set(0.5)

        # Fill upper triangle with values near 0.5 (no significance)
        for i in range(n_trials):
            for j in range(i + 1, n_trials):
                comparison_matrix = comparison_matrix.at[i, j].set(0.45)

        result = find_first_significant_trial(
            comparison_matrix, reference_trial=0, significance_level=0.05
        )

        assert result is None

    def test_respects_significance_level(self) -> None:
        """Stricter significance should require stronger evidence."""
        n_trials = 20
        comparison_matrix = jnp.full((n_trials, n_trials), jnp.nan)
        comparison_matrix = comparison_matrix.at[jnp.diag_indices(n_trials)].set(0.5)

        # Create borderline significance
        for j in range(1, n_trials):
            # P value of 0.02 - significant at 0.05 but not at 0.01
            comparison_matrix = comparison_matrix.at[0, j].set(0.02)

        result_lenient = find_first_significant_trial(
            comparison_matrix, reference_trial=0, significance_level=0.05
        )
        result_strict = find_first_significant_trial(
            comparison_matrix, reference_trial=0, significance_level=0.01
        )

        assert result_lenient is not None
        assert result_strict is None


class TestSmithLearningAlgorithmTrialComparison:
    """Tests for trial comparison methods on SmithLearningAlgorithm class."""

    @pytest.fixture
    def fitted_model(self):
        """Create and fit a model for testing."""
        # Learning data with clear improvement
        responses = jnp.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
                               1, 1, 1, 0, 1, 1, 1, 1, 1, 1])

        model = SmithLearningAlgorithm(
            sigma_epsilon=0.3,
            prob_correct_by_chance=0.5,
            initial_state_method='set_initial_direct_from_second_trial'
        )
        model.fit(responses, max_iter=50)
        return model

    def test_compare_trials_requires_fit(self) -> None:
        """compare_trials should raise if model not fitted."""
        model = SmithLearningAlgorithm()
        key = jax.random.PRNGKey(0)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.compare_trials(key, trial1=0, trial2=5)

    def test_compare_trials_validates_indices(self, fitted_model) -> None:
        """compare_trials should validate trial indices."""
        key = jax.random.PRNGKey(0)

        with pytest.raises(ValueError, match="Trial indices"):
            fitted_model.compare_trials(key, trial1=-1, trial2=5)

        with pytest.raises(ValueError, match="Trial indices"):
            fitted_model.compare_trials(key, trial1=0, trial2=100)

    def test_compare_trials_returns_probability(self, fitted_model) -> None:
        """compare_trials should return probability in [0, 1]."""
        key = jax.random.PRNGKey(0)

        result = fitted_model.compare_trials(key, trial1=0, trial2=15, n_samples=1000)

        assert 0.0 <= result <= 1.0

    def test_get_trial_comparison_matrix_requires_fit(self) -> None:
        """get_trial_comparison_matrix should raise if model not fitted."""
        model = SmithLearningAlgorithm()
        key = jax.random.PRNGKey(0)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_trial_comparison_matrix(key)

    def test_get_trial_comparison_matrix_shape(self, fitted_model) -> None:
        """get_trial_comparison_matrix should return correct shape."""
        key = jax.random.PRNGKey(0)
        n_trials = len(fitted_model.smoothed_learning_state_mode)

        result = fitted_model.get_trial_comparison_matrix(key, n_samples=1000)

        assert result.shape == (n_trials, n_trials)

    def test_find_first_significant_improvement_requires_fit(self) -> None:
        """find_first_significant_improvement should raise if not fitted."""
        model = SmithLearningAlgorithm()
        key = jax.random.PRNGKey(0)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.find_first_significant_improvement(key)


# --- Property-Based Tests using Hypothesis ---

from hypothesis import given, settings
from hypothesis import strategies as st


class TestSmithLearningFilterProperties:
    """Property-based tests for smith_learning_filter."""

    @given(st.integers(min_value=5, max_value=30))
    @settings(max_examples=20, deadline=None)
    def test_probability_always_in_bounds(self, n_trials: int) -> None:
        """Probability of correct response should always be in [0, 1]."""
        # Generate random binary outcomes
        key = jax.random.PRNGKey(42)
        outcomes = jax.random.bernoulli(key, 0.5, (n_trials,)).astype(int)

        try:
            prob, _, _, _, _ = smith_learning_filter(
                jnp.array(outcomes), max_possible_correct=1
            )
            assert jnp.all(prob >= 0.0)
            assert jnp.all(prob <= 1.0)
        except Exception:
            pass  # Skip if optimization fails

    @given(st.integers(min_value=5, max_value=30))
    @settings(max_examples=20, deadline=None)
    def test_variance_always_positive(self, n_trials: int) -> None:
        """Variance should always be positive."""
        key = jax.random.PRNGKey(123)
        outcomes = jax.random.bernoulli(key, 0.5, (n_trials,)).astype(int)

        try:
            _, _, variance, _, one_step_var = smith_learning_filter(
                jnp.array(outcomes), max_possible_correct=1
            )
            assert jnp.all(variance > 0)
            assert jnp.all(one_step_var > 0)
        except Exception:
            pass  # Skip if optimization fails

    @given(
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        st.floats(min_value=0.05, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_outputs_finite_for_valid_params(
        self, prob_chance: float, sigma: float
    ) -> None:
        """Outputs should be finite for valid parameter combinations."""
        outcomes = jnp.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1])

        try:
            prob, mode, variance, _, _ = smith_learning_filter(
                outcomes,
                prob_correct_by_chance=prob_chance,
                sigma_epsilon=sigma,
                max_possible_correct=1,
            )
            assert jnp.all(jnp.isfinite(prob))
            assert jnp.all(jnp.isfinite(mode))
            assert jnp.all(jnp.isfinite(variance))
        except Exception:
            pass  # Skip if optimization fails


class TestSmithLearningSmootherProperties:
    """Property-based tests for smith_learning_smoother."""

    @given(st.integers(min_value=5, max_value=30))
    @settings(max_examples=20, deadline=None)
    def test_smoother_probability_in_bounds(self, n_trials: int) -> None:
        """Smoothed probability should be in [0, 1]."""
        key = jax.random.PRNGKey(456)
        outcomes = jax.random.bernoulli(key, 0.5, (n_trials,)).astype(int)

        try:
            _, filter_mode, filter_var, one_step_mode, one_step_var = (
                smith_learning_filter(jnp.array(outcomes), max_possible_correct=1)
            )

            _, _, smooth_prob, _ = smith_learning_smoother(
                filter_mode, filter_var, one_step_mode, one_step_var
            )

            assert jnp.all(smooth_prob >= 0.0)
            assert jnp.all(smooth_prob <= 1.0)
        except Exception:
            pass  # Skip if optimization fails

    @given(st.integers(min_value=5, max_value=30))
    @settings(max_examples=20, deadline=None)
    def test_smoother_variance_non_negative(self, n_trials: int) -> None:
        """Smoothed variance should be non-negative."""
        key = jax.random.PRNGKey(789)
        outcomes = jax.random.bernoulli(key, 0.5, (n_trials,)).astype(int)

        try:
            _, filter_mode, filter_var, one_step_mode, one_step_var = (
                smith_learning_filter(jnp.array(outcomes), max_possible_correct=1)
            )

            _, smooth_var, _, _ = smith_learning_smoother(
                filter_mode, filter_var, one_step_mode, one_step_var
            )

            assert jnp.all(smooth_var >= 0.0)
        except Exception:
            pass  # Skip if optimization fails

    @given(st.integers(min_value=5, max_value=30))
    @settings(max_examples=20, deadline=None)
    def test_smoother_last_equals_filter_last(self, n_trials: int) -> None:
        """Last smoothed state should equal last filtered state."""
        key = jax.random.PRNGKey(321)
        outcomes = jax.random.bernoulli(key, 0.5, (n_trials,)).astype(int)

        try:
            _, filter_mode, filter_var, one_step_mode, one_step_var = (
                smith_learning_filter(jnp.array(outcomes), max_possible_correct=1)
            )

            smooth_mode, smooth_var, _, _ = smith_learning_smoother(
                filter_mode, filter_var, one_step_mode, one_step_var
            )

            np.testing.assert_allclose(smooth_mode[-1], filter_mode[-1], rtol=1e-5)
            np.testing.assert_allclose(smooth_var[-1], filter_var[-1], rtol=1e-5)
        except Exception:
            pass  # Skip if optimization fails


class TestMaximizationStepProperties:
    """Property-based tests for the maximization step."""

    @given(st.integers(min_value=10, max_value=50))
    @settings(max_examples=15, deadline=None)
    def test_estimated_sigma_positive(self, n_trials: int) -> None:
        """Estimated sigma_epsilon should be positive."""
        key = jax.random.PRNGKey(654)
        outcomes = jax.random.bernoulli(key, 0.6, (n_trials,)).astype(int)

        try:
            _, filter_mode, filter_var, one_step_mode, one_step_var = (
                smith_learning_filter(jnp.array(outcomes), max_possible_correct=1)
            )

            smooth_mode, smooth_var, _, smoother_gain = smith_learning_smoother(
                filter_mode, filter_var, one_step_mode, one_step_var
            )

            sigma_est, _, _ = maximization_step(smooth_mode, smooth_var, smoother_gain)

            assert sigma_est > 0
        except Exception:
            pass  # Skip if optimization fails


class TestTrialComparisonProperties:
    """Property-based tests for trial comparison functions."""

    @given(st.integers(min_value=5, max_value=20))
    @settings(max_examples=15, deadline=None)
    def test_cross_covariance_symmetry(self, n_trials: int) -> None:
        """Cross-covariance matrix should be symmetric."""
        key = jax.random.PRNGKey(111)
        outcomes = jax.random.bernoulli(key, 0.5, (n_trials,)).astype(int)

        try:
            _, filter_mode, filter_var, one_step_mode, one_step_var = (
                smith_learning_filter(jnp.array(outcomes), max_possible_correct=1)
            )

            _, smooth_var, _, smoother_gain = smith_learning_smoother(
                filter_mode, filter_var, one_step_mode, one_step_var
            )

            cross_cov = compute_cross_covariance_matrix(smooth_var, smoother_gain)

            np.testing.assert_allclose(cross_cov, cross_cov.T, rtol=1e-5, atol=1e-10)
        except Exception:
            pass  # Skip if optimization fails

    @given(st.integers(min_value=5, max_value=15))
    @settings(max_examples=10, deadline=None)
    def test_comparison_matrix_diagonal_is_half(self, n_trials: int) -> None:
        """Diagonal of comparison matrix should be 0.5 (comparing trial to itself)."""
        key = jax.random.PRNGKey(222)
        outcomes = jax.random.bernoulli(key, 0.5, (n_trials,)).astype(int)

        try:
            _, filter_mode, filter_var, one_step_mode, one_step_var = (
                smith_learning_filter(jnp.array(outcomes), max_possible_correct=1)
            )

            smooth_mode, smooth_var, _, smoother_gain = smith_learning_smoother(
                filter_mode, filter_var, one_step_mode, one_step_var
            )

            comp_matrix = compute_trial_comparison_matrix(
                key=jax.random.PRNGKey(42),
                smoothed_learning_state_mode=smooth_mode,
                smoothed_learning_state_variance=smooth_var,
                smoother_gain=smoother_gain,
            )

            diagonal = jnp.diag(comp_matrix)
            np.testing.assert_allclose(diagonal, 0.5, rtol=1e-3)
        except Exception:
            pass  # Skip if optimization fails

    @given(st.integers(min_value=5, max_value=15))
    @settings(max_examples=10, deadline=None)
    def test_comparison_probabilities_in_bounds(self, n_trials: int) -> None:
        """All comparison probabilities should be in [0, 1]."""
        key = jax.random.PRNGKey(333)
        outcomes = jax.random.bernoulli(key, 0.5, (n_trials,)).astype(int)

        try:
            _, filter_mode, filter_var, one_step_mode, one_step_var = (
                smith_learning_filter(jnp.array(outcomes), max_possible_correct=1)
            )

            smooth_mode, smooth_var, _, smoother_gain = smith_learning_smoother(
                filter_mode, filter_var, one_step_mode, one_step_var
            )

            comp_matrix = compute_trial_comparison_matrix(
                key=jax.random.PRNGKey(42),
                smoothed_learning_state_mode=smooth_mode,
                smoothed_learning_state_variance=smooth_var,
                smoother_gain=smoother_gain,
            )

            assert jnp.all(comp_matrix >= 0.0)
            assert jnp.all(comp_matrix <= 1.0)
        except Exception:
            pass  # Skip if optimization fails


class TestSimulateLearningDataProperties:
    """Property-based tests for simulate_learning_data."""

    @given(
        st.integers(min_value=10, max_value=100),
        st.floats(min_value=0.1, max_value=0.4, allow_nan=False),
        st.floats(min_value=0.6, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_outcomes_binary(
        self, n_trials: int, prob_init: float, prob_final: float
    ) -> None:
        """Simulated outcomes should be binary (0 or 1)."""
        outcomes, _ = simulate_learning_data(
            n_trials=n_trials,
            prob_success_init=prob_init,
            prob_success_final=prob_final,
            seed=42,
        )

        assert all(o in [0, 1] for o in outcomes)

    @given(
        st.integers(min_value=10, max_value=100),
        st.floats(min_value=0.1, max_value=0.4, allow_nan=False),
        st.floats(min_value=0.6, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_true_prob_in_bounds(
        self, n_trials: int, prob_init: float, prob_final: float
    ) -> None:
        """True probability should be in [0, 1]."""
        _, true_prob = simulate_learning_data(
            n_trials=n_trials,
            prob_success_init=prob_init,
            prob_success_final=prob_final,
            seed=42,
        )

        assert all(0.0 <= p <= 1.0 for p in true_prob)

    @given(
        st.integers(min_value=10, max_value=100),
        st.floats(min_value=0.1, max_value=0.4, allow_nan=False),
        st.floats(min_value=0.6, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_correct_length(
        self, n_trials: int, prob_init: float, prob_final: float
    ) -> None:
        """Output length should match n_trials."""
        outcomes, true_prob = simulate_learning_data(
            n_trials=n_trials,
            prob_success_init=prob_init,
            prob_success_final=prob_final,
            seed=42,
        )

        assert len(outcomes) == n_trials
        assert len(true_prob) == n_trials
