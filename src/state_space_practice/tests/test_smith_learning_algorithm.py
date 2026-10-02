# ruff: noqa: E402
"""Tests for the smith_learning_algorithm module.

This module tests the Bayesian state-space model for learning dynamics,
including the Laplace approximation filter/smoother and EM algorithm.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.exceptions import NotFittedError, StateSpaceWarning
from state_space_practice.fitted_state import is_set
from state_space_practice.smith_learning_algorithm import (
    DEFAULT_SIGMA_EPSILON,
    SmithLearningModel,
    _find_runs_of_value,
    _log_posterior_objective,
    approximate_gaussian,
    calculate_latent_state_percentiles,
    calculate_probability_confidence_limits,
    compare_two_trials,
    compute_cross_covariance_matrix,
    compute_trial_comparison_matrix,
    find_first_significant_trial,
    find_min_consecutive_successes,
    maximization_step,
    simulate_learning_data,
    smith_learning_filter,
    smith_learning_smoother,
)
from state_space_practice.tests.model_state import (
    assert_model_state_unchanged,
    snapshot_model_state,
)
from state_space_practice.tests.recovery_helpers import assert_ll_monotonic

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

        prob, mode, variance, one_step_mode, one_step_var = smith_learning_filter(
            outcomes, max_possible_correct=1
        )

        assert prob.shape == (n_trials,)
        assert mode.shape == (n_trials,)
        assert variance.shape == (n_trials,)
        assert one_step_mode.shape == (n_trials,)
        assert one_step_var.shape == (n_trials,)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_probability_bounds(self, simulated_data) -> None:
        """Probabilities should be in [0, 1]."""
        outcomes, _ = simulated_data

        prob, _, _, _, _ = smith_learning_filter(outcomes, max_possible_correct=1)

        assert jnp.all(prob >= 0)
        assert jnp.all(prob <= 1)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_variance_positive(self, simulated_data) -> None:
        """Variances should be positive."""
        outcomes, _ = simulated_data

        _, _, variance, _, one_step_var = smith_learning_filter(
            outcomes, max_possible_correct=1
        )

        assert jnp.all(variance > 0)
        assert jnp.all(one_step_var > 0)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_no_nans(self, simulated_data) -> None:
        """Filter should not produce NaN values."""
        outcomes, _ = simulated_data

        prob, mode, variance, one_step_mode, one_step_var = smith_learning_filter(
            outcomes, max_possible_correct=1
        )

        assert not jnp.any(jnp.isnan(prob))
        assert not jnp.any(jnp.isnan(mode))
        assert not jnp.any(jnp.isnan(variance))
        assert not jnp.any(jnp.isnan(one_step_mode))
        assert not jnp.any(jnp.isnan(one_step_var))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_custom_initial_state(self, simulated_data) -> None:
        """Custom initial state should affect filter output."""
        outcomes, _ = simulated_data

        prob_default, _, _, _, _ = smith_learning_filter(
            outcomes, init_learning_state=0.0, max_possible_correct=1
        )
        prob_high, _, _, _, _ = smith_learning_filter(
            outcomes, init_learning_state=2.0, max_possible_correct=1
        )

        # First probability should be higher with higher initial state
        assert prob_high[0] > prob_default[0]

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_sigma_epsilon_affects_variance(self, simulated_data) -> None:
        """Higher sigma_epsilon should lead to higher variance."""
        outcomes, _ = simulated_data

        _, _, var_low, _, _ = smith_learning_filter(
            outcomes, sigma_epsilon=0.1, max_possible_correct=1
        )
        _, _, var_high, _, _ = smith_learning_filter(
            outcomes, sigma_epsilon=0.5, max_possible_correct=1
        )

        # Average variance should be higher with higher sigma_epsilon
        assert jnp.mean(var_high) > jnp.mean(var_low)

    @pytest.mark.parametrize("differentiable", [False, True])
    def test_integer_initial_state_matches_float(
        self, simulated_data, differentiable
    ) -> None:
        """An integer initial state is promoted to float, not a TypeError."""
        outcomes, _ = simulated_data
        out_int = smith_learning_filter(
            outcomes,
            init_learning_state=1,
            max_possible_correct=1,
            differentiable=differentiable,
        )
        out_float = smith_learning_filter(
            outcomes,
            init_learning_state=1.0,
            max_possible_correct=1,
            differentiable=differentiable,
        )
        for a, b in zip(out_int, out_float):
            assert a.dtype == jnp.float64
            np.testing.assert_allclose(a, b, rtol=1e-12)

    @pytest.mark.parametrize("differentiable", [False, True])
    def test_stalled_newton_warns(self, simulated_data, differentiable) -> None:
        """A prior deep in saturation opposite the data (N=1000, y=0, prior
        N(30, 1e4)): every Armijo step overshoots, the step is 0 and the mode
        never moves. The filter must say so; ordinary data stays silent."""
        with pytest.warns(StateSpaceWarning, match="smith_learning_filter"):
            _, mode, _, _, _ = smith_learning_filter(
                jnp.array([0]),
                init_learning_state=30.0,
                init_learning_variance=1e4 - DEFAULT_SIGMA_EPSILON**2,
                max_possible_correct=1000,
                differentiable=differentiable,
            )
            jax.effects_barrier()
        np.testing.assert_allclose(mode[0], 30.0)  # guard: really stalled
        outcomes, _ = simulated_data
        with warnings.catch_warnings():
            warnings.simplefilter("error", StateSpaceWarning)
            smith_learning_filter(
                outcomes, max_possible_correct=1, differentiable=differentiable
            )
            jax.effects_barrier()

    def test_float32_inputs_stay_float32(self, simulated_data) -> None:
        outcomes, _ = simulated_data
        kwargs = dict(init_learning_variance=0.3, max_possible_correct=1)
        ref = smith_learning_filter(
            outcomes, init_learning_state=0.5, sigma_epsilon=0.2, **kwargs
        )
        out = smith_learning_filter(
            outcomes,
            init_learning_state=np.float32(0.5),
            sigma_epsilon=np.float32(0.2),
            **kwargs,
        )
        for a, b in zip(out, ref):
            assert a.dtype == jnp.float32
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


class TestSmithLearningSmoother:
    """Tests for the smith_learning_smoother function."""

    @pytest.fixture
    def filter_outputs(self):
        """Generate filter outputs for smoother testing."""
        outcomes, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes)

        prob, mode, variance, one_step_mode, one_step_var = smith_learning_filter(
            outcomes, max_possible_correct=1
        )

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

        prob, mode, variance, one_step_mode, one_step_var = smith_learning_filter(
            outcomes, max_possible_correct=1
        )
        sm_mode, sm_var, _, sm_gain = smith_learning_smoother(
            mode, variance, one_step_mode, one_step_var
        )

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

    def test_init_var_is_first_smoother_variance_minus_process_noise(
        self, smoother_outputs
    ) -> None:
        """x_1 ~ N(x_0, P_0 + sigma^2): the optimum is P_0 = P_{1|T} - sigma^2."""
        s = smoother_outputs

        sigma, _, init_var = maximization_step(s["mode"], s["variance"], s["gain"])

        np.testing.assert_allclose(
            init_var, max(float(s["variance"][0] - sigma**2), 1e-8), rtol=1e-10
        )

    def test_transition_only_estimate_for_fixed_initial_variance(
        self, smoother_outputs
    ) -> None:
        """Heuristic initial-state methods keep sigma^2 = S / (T - 1)."""
        s = smoother_outputs
        m, P, G = (np.asarray(s[k]) for k in ("mode", "variance", "gain"))
        S = np.sum((m[1:] - m[:-1]) ** 2 + P[1:] + P[:-1] - 2 * P[1:] * G)
        sigma, _, init_var = maximization_step(
            s["mode"], s["variance"], s["gain"], estimate_initial_variance=False
        )
        np.testing.assert_allclose(sigma**2, S / (len(m) - 1), rtol=1e-10)
        np.testing.assert_allclose(init_var, P[0], rtol=1e-10)


class TestCalculateProbabilityConfidenceLimits:
    """Tests for calculate_probability_confidence_limits function."""

    def test_output_shapes(self) -> None:
        """Output should have correct shapes."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jnp.zeros(n_trials)
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5

        percentiles, prob_above_chance = calculate_probability_confidence_limits(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            prob_correct_by_chance=0.5,
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
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            prob_correct_by_chance=0.5,
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
            prob_correct_by_chance=0.3,
            critical_probability_threshold=0.05,
            sequence_length=100,
        )

        result_high = find_min_consecutive_successes(
            prob_correct_by_chance=0.7,
            critical_probability_threshold=0.05,
            sequence_length=100,
        )

        # With higher probability, need longer run to be surprising
        if result_low is not None and result_high is not None:
            assert result_high > result_low

    def test_stricter_threshold_needs_longer_run(self) -> None:
        """Stricter threshold should require longer run."""
        result_loose = find_min_consecutive_successes(
            prob_correct_by_chance=0.5,
            critical_probability_threshold=0.10,
            sequence_length=100,
        )

        result_strict = find_min_consecutive_successes(
            prob_correct_by_chance=0.5,
            critical_probability_threshold=0.01,
            sequence_length=100,
        )

        if result_loose is not None and result_strict is not None:
            assert result_strict >= result_loose

    def test_invalid_probability_raises(self) -> None:
        """Invalid probability should raise ValueError."""
        with pytest.raises(ValueError):
            find_min_consecutive_successes(
                prob_correct_by_chance=1.5,
                critical_probability_threshold=0.05,
                sequence_length=100,
            )

        with pytest.raises(ValueError):
            find_min_consecutive_successes(
                prob_correct_by_chance=-0.1,
                critical_probability_threshold=0.05,
                sequence_length=100,
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


class TestSmithLearningModelClass:
    """Tests for the SmithLearningModel class."""

    def test_initialization(self) -> None:
        """Class should initialize without errors."""
        model = SmithLearningModel(
            init_learning_state=0.0,
            sigma_epsilon=float(jnp.sqrt(0.05)),
            prob_correct_by_chance=0.5,
        )

        assert model.init_learning_state == 0.0
        assert model.prob_correct_by_chance == 0.5

    def test_initialization_with_explicit_variance(self) -> None:
        """Class should accept explicit init_learning_variance."""
        model = SmithLearningModel(
            init_learning_state=0.0,
            init_learning_variance=0.1,
            sigma_epsilon=float(jnp.sqrt(0.05)),
            prob_correct_by_chance=0.5,
        )

        np.testing.assert_allclose(model.init_learning_variance, 0.1, rtol=1e-10)

    def test_default_variance_equals_sigma_squared(self) -> None:
        """Default init_learning_variance should equal sigma_epsilon^2."""
        sigma_eps = 0.3
        model = SmithLearningModel(sigma_epsilon=sigma_eps)

        np.testing.assert_allclose(
            model.init_learning_variance, sigma_eps**2, rtol=1e-10
        )

    def test_invalid_sigma_epsilon_raises(self) -> None:
        """Negative sigma_epsilon should raise error."""
        with pytest.raises(ValueError):
            SmithLearningModel(sigma_epsilon=-0.1)

    def test_invalid_prob_chance_raises(self) -> None:
        """Invalid prob_correct_by_chance should raise error."""
        with pytest.raises(ValueError, match="prob_correct_by_chance"):
            SmithLearningModel(prob_correct_by_chance=0.0)

        with pytest.raises(ValueError, match="prob_correct_by_chance"):
            SmithLearningModel(prob_correct_by_chance=1.0)

    def test_invalid_variance_type_raises(self) -> None:
        """Invalid init_learning_variance type should raise TypeError."""
        with pytest.raises(TypeError):
            SmithLearningModel(init_learning_variance="invalid")  # type: ignore[arg-type]

    def test_negative_variance_raises(self) -> None:
        """Negative init_learning_variance should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            SmithLearningModel(init_learning_variance=-0.1)

    def test_invalid_initial_state_method_raises(self) -> None:
        """Invalid initial_state_method should raise ValueError."""
        with pytest.raises(ValueError, match="initial_state_method must be one of"):
            SmithLearningModel(initial_state_method="typo")

    def test_is_fitted_false_before_fit(self) -> None:
        """is_fitted should be False before calling fit()."""
        model = SmithLearningModel()
        assert not model.is_fitted

    @pytest.mark.parametrize(
        "attr", ["smoothed_learning_state_mode", "log_likelihood_"]
    )
    def test_fitted_attribute_unset_before_fit(self, attr: str) -> None:
        """Fitted outputs raise NotFittedError (and look absent) before fit()."""
        model = SmithLearningModel()
        with pytest.raises(NotFittedError, match=attr):
            getattr(model, attr)
        assert not hasattr(model, attr)

    def test_n_iter_is_none_after_sgd_fit(self) -> None:
        """n_iter_ counts EM iterations only: None (not an error) on a model
        fitted by fit_sgd, an int after fit()."""
        outcomes = jnp.asarray(simulate_learning_data(n_trials=20, seed=0)[0])
        model = SmithLearningModel()
        assert model.n_iter_ is None
        model.fit_sgd(outcomes, num_steps=3)
        assert model.is_fitted
        assert model.n_iter_ is None
        assert "EM iterations:          None" in model.summary()

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_is_fitted_true_after_fit(self) -> None:
        """is_fitted should be True after calling fit()."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=3)
        assert model.is_fitted

    def test_repr_before_fit(self) -> None:
        """__repr__ should show 'not fitted' before fit."""
        model = SmithLearningModel(sigma_epsilon=0.22)
        r = repr(model)
        assert "SmithLearningModel(" in r
        assert "not fitted" in r
        assert "0.22" in r

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_repr_after_fit(self) -> None:
        """__repr__ should show 'fitted' after fit."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=3)
        r = repr(model)
        assert "fitted)" in r
        assert "not fitted" not in r

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_plot_learning_curve_returns_fig_ax(self) -> None:
        """plot_learning_curve should return (fig, ax)."""
        import matplotlib.pyplot as plt

        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=3)
        result = model.plot_learning_curve(jax.random.PRNGKey(0))
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fit_verbose_logs_progress(self, caplog: pytest.LogCaptureFixture) -> None:
        """fit(verbose=True) logs per-iteration progress at INFO; quiet does not."""
        import logging

        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        smith_logger = "state_space_practice.smith_learning_algorithm"
        with caplog.at_level(logging.INFO, logger=smith_logger):
            SmithLearningModel().fit(outcomes, max_iter=5, verbose=False)
        assert "Log-Likelihood" not in caplog.text
        with caplog.at_level(logging.INFO, logger=smith_logger):
            SmithLearningModel().fit(outcomes, max_iter=5, verbose=True)
        assert "Iteration 1/5" in caplog.text

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fit_final_estep_after_convergence(self) -> None:
        """fit() should run final M-step + E-step so stored results match MLE params."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        log_likelihoods = model.fit(outcomes, max_iter=50)
        # If converged, the final log-likelihood entry is from the post-convergence E-step
        if len(log_likelihoods) >= 2:
            # The last two LLs should be very close (converged params + final E-step)
            assert abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1.0

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fit_returns_log_likelihoods(self) -> None:
        """fit() should return list of log-likelihoods."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)

        model = SmithLearningModel()
        log_likelihoods = model.fit(outcomes, max_iter=3)

        assert isinstance(log_likelihoods, list)
        assert len(log_likelihoods) > 0

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_fit_populates_attributes(self) -> None:
        """fit() should populate smoother attributes."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)

        model = SmithLearningModel()
        model.fit(outcomes, max_iter=3)

        assert model.smoothed_learning_state_mode is not None
        assert model.smoothed_learning_state_variance is not None
        assert model.smoothed_prob_correct_response is not None

    def test_get_learning_curve_requires_fit(self) -> None:
        """get_learning_curve() should raise if not fitted."""
        model = SmithLearningModel()

        with pytest.raises(NotFittedError):
            model.get_learning_curve(jax.random.PRNGKey(0))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_get_learning_curve_output_shapes(self) -> None:
        """get_learning_curve() should return correct shapes."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)

        model = SmithLearningModel()
        model.fit(outcomes, max_iter=3)

        percentiles, prob_above_chance = model.get_learning_curve(
            jax.random.PRNGKey(0), n_samples=100
        )

        assert percentiles.shape[0] == 3  # Default percentiles
        assert percentiles.shape[1] == len(outcomes)
        assert prob_above_chance is None  # Not requested


class TestSmithLearningModelEdgeCases:
    """Edge case tests for scientific robustness."""

    def test_single_trial_raises(self) -> None:
        """fit() with a single trial should raise ValueError."""
        model = SmithLearningModel()
        with pytest.raises(ValueError, match="at least 2 trials"):
            model.fit(jnp.array([1]))

    def test_n_correct_exceeds_max_possible_raises(self) -> None:
        """fit() should raise if n_correct > max_possible_correct."""
        model = SmithLearningModel(max_possible_correct=1)
        with pytest.raises(ValueError, match="exceeding max_possible_correct"):
            model.fit(jnp.array([0, 1, 2, 1, 0]))

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            (jnp.array([0.0, -1.0, 1.0, 0.0]), "non-negative"),
            (jnp.array([0.0, 0.5, 1.0, 0.0]), "integer-valued"),
            (jnp.array([0.0, jnp.nan, 1.0, 0.0]), "finite"),
        ],
    )
    def test_invalid_n_correct_counts_raise(self, data, match) -> None:
        model = SmithLearningModel(max_possible_correct=1)
        with pytest.raises(ValueError, match=match):
            model.fit(data, max_iter=2)
        with pytest.raises(ValueError, match=match):
            model.fit_sgd(data, num_steps=1)
        with pytest.raises(ValueError, match=match):
            smith_learning_filter(data, max_possible_correct=1)

    def test_max_iter_final_estep_matches_returned_ll(self) -> None:
        rng = np.random.default_rng(7)
        outcomes = jnp.asarray(rng.integers(0, 2, size=30).astype(float))
        model = SmithLearningModel(max_possible_correct=1)
        lls = model.fit(outcomes, max_iter=1)
        fresh_ll = model._e_step(outcomes)
        np.testing.assert_allclose(fresh_ll, lls[-1], atol=1e-6)
        np.testing.assert_allclose(model.log_likelihood_, lls[-1], atol=1e-6)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_all_zeros_produces_finite(self) -> None:
        """All-zero outcomes should not produce NaN/Inf."""
        model = SmithLearningModel()
        outcomes = jnp.zeros(20, dtype=jnp.int32)
        log_likelihoods = model.fit(outcomes, max_iter=5)
        assert all(np.isfinite(ll) for ll in log_likelihoods)
        assert model.is_fitted
        assert bool(jnp.all(jnp.isfinite(model.smoothed_learning_state_mode)))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_all_ones_produces_finite(self) -> None:
        """All-one outcomes should not produce NaN/Inf."""
        model = SmithLearningModel()
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
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=10)
        assert model.sigma_epsilon > 0
        assert np.isfinite(model.sigma_epsilon)

    def test_2d_input_raises(self) -> None:
        """fit() with 2D input should raise ValueError."""
        model = SmithLearningModel()
        with pytest.raises(ValueError, match="1D"):
            model.fit(jnp.array([[1, 0], [0, 1]]))


class TestSummaryAndScoring:
    """Tests for summary(), bic(), and compare_to_null()."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_log_likelihood_stored_after_fit(self) -> None:
        """log_likelihood_ should be populated after fit."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        model = SmithLearningModel()
        model.fit(jnp.array(outcomes_np), max_iter=5)
        assert model.log_likelihood_ is not None
        assert np.isfinite(model.log_likelihood_)
        assert model.n_iter_ is not None
        assert model.n_iter_ > 0

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_bic_is_finite(self) -> None:
        """bic() should return a finite value after fit."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        model = SmithLearningModel()
        model.fit(jnp.array(outcomes_np), max_iter=5)
        bic = model.bic()
        assert np.isfinite(bic)

    def test_bic_requires_fit(self) -> None:
        """bic() should raise if not fitted."""
        model = SmithLearningModel()
        with pytest.raises(NotFittedError, match="not been fitted"):
            model.bic()

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_compare_to_null_returns_dict(self) -> None:
        """compare_to_null() should return a dict with expected keys."""
        outcomes_np, _ = simulate_learning_data(n_trials=30, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=10)
        result = model.compare_to_null(outcomes)
        assert isinstance(result, dict)
        expected_keys = {
            "model_ll",
            "null_ll",
            "model_bic",
            "null_bic",
            "delta_bic",
            "learning_detected",
        }
        assert set(result.keys()) == expected_keys
        assert np.isfinite(result["model_ll"])
        assert np.isfinite(result["null_ll"])
        assert np.isfinite(result["delta_bic"])

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_compare_to_null_detects_learning(self) -> None:
        """Learning model should beat null on clear learning data."""
        outcomes_np, _ = simulate_learning_data(
            n_trials=50,
            seed=42,
            prob_success_init=0.2,
            prob_success_final=0.9,
        )
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=30)
        result = model.compare_to_null(outcomes)
        assert result["model_ll"] > result["null_ll"], (
            "Learning model should have higher LL than null"
        )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_summary_returns_string(self) -> None:
        """summary() should return a multi-line string."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=5)
        s = model.summary()
        assert isinstance(s, str)
        assert "sigma_epsilon" in s
        assert "Log-likelihood" in s
        assert "BIC" in s

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_summary_with_key_and_data(self) -> None:
        """summary() with key and data should include null comparison."""
        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=5)
        s = model.summary(
            key=jax.random.PRNGKey(0),
            n_correct_responses=outcomes,
        )
        assert "Delta BIC" in s
        assert "Criterion trial" in s

    def test_summary_requires_fit(self) -> None:
        """summary() should raise if not fitted."""
        model = SmithLearningModel()
        with pytest.raises(NotFittedError, match="not been fitted"):
            model.summary()


class TestFindCriterionTrial:
    """Tests for find_criterion_trial method."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_returns_int_for_learning_data(self) -> None:
        """Should return a trial index for data with clear learning."""
        outcomes_np, _ = simulate_learning_data(
            n_trials=50, seed=42, prob_success_init=0.3, prob_success_final=0.95
        )
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=10)
        result = model.find_criterion_trial(jax.random.PRNGKey(0))
        # Should return an int or None; if learning is clear, should be int
        assert result is None or isinstance(result, int)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_returns_none_for_no_learning(self) -> None:
        """Should return None when performance never exceeds chance."""
        outcomes = jnp.zeros(30, dtype=jnp.int32)  # All failures
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=5)
        result = model.find_criterion_trial(jax.random.PRNGKey(0))
        assert result is None

    def test_requires_fit(self) -> None:
        """Should raise if not fitted."""
        model = SmithLearningModel()
        with pytest.raises(NotFittedError, match="not been fitted"):
            model.find_criterion_trial(jax.random.PRNGKey(0))


class TestIdentifySignificantRuns:
    """Tests for find_significant_runs and find_critical_run_length."""

    def test_find_critical_run_length_without_fit(self) -> None:
        """find_critical_run_length should work without fitting."""
        model = SmithLearningModel(prob_correct_by_chance=0.5)
        result = model.find_critical_run_length(sequence_length=50)
        assert result is None or isinstance(result, int)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_identify_runs_returns_tuple(self) -> None:
        """find_significant_runs should return (j_crit, runs)."""
        model = SmithLearningModel(prob_correct_by_chance=0.5)
        data = jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        j_crit, runs = model.find_significant_runs(data)
        assert j_crit is None or isinstance(j_crit, int)
        assert isinstance(runs, list)

    def test_identify_runs_empty_input(self) -> None:
        """Empty input should return (None, [])."""
        model = SmithLearningModel()
        j_crit, runs = model.find_significant_runs(jnp.array([]))
        assert j_crit is None
        assert runs == []


class TestPlotTrialComparisonMatrix:
    """Tests for plot_trial_comparison_matrix method."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_returns_fig_ax(self) -> None:
        """plot_trial_comparison_matrix should return (fig, ax)."""
        import matplotlib.pyplot as plt

        outcomes_np, _ = simulate_learning_data(n_trials=15, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=3)
        fig, ax = model.plot_trial_comparison_matrix(
            jax.random.PRNGKey(0), n_samples=100
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_requires_fit(self) -> None:
        """Should raise if not fitted."""
        model = SmithLearningModel()
        with pytest.raises(NotFittedError, match="not been fitted"):
            model.plot_trial_comparison_matrix(jax.random.PRNGKey(0))


class TestPlotConvergence:
    """Tests for plot_convergence method."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_returns_fig_ax(self) -> None:
        """plot_convergence should return (fig, ax)."""
        import matplotlib.pyplot as plt

        outcomes_np, _ = simulate_learning_data(n_trials=15, seed=42)
        model = SmithLearningModel()
        model.fit(jnp.array(outcomes_np), max_iter=3)
        fig, ax = model.plot_convergence()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_requires_fit(self) -> None:
        """Should raise if not fitted."""
        model = SmithLearningModel()
        with pytest.raises(NotFittedError, match="not been fitted"):
            model.plot_convergence()


class TestPlotSummary:
    """Tests for plot_summary method."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_returns_fig_axes(self) -> None:
        """plot_summary should return (fig, axes) with 3 panels."""
        import matplotlib.pyplot as plt

        outcomes_np, _ = simulate_learning_data(n_trials=20, seed=42)
        outcomes = jnp.array(outcomes_np)
        model = SmithLearningModel()
        model.fit(outcomes, max_iter=5)
        fig, axes = model.plot_summary(
            jax.random.PRNGKey(0),
            observed_n_correct=outcomes,
            n_samples=200,
        )
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3
        plt.close(fig)

    def test_requires_fit(self) -> None:
        """Should raise if not fitted."""
        model = SmithLearningModel()
        with pytest.raises(NotFittedError, match="not been fitted"):
            model.plot_summary(jax.random.PRNGKey(0))


class TestCalculateLatentStatePercentiles:
    """Tests for calculate_latent_state_percentiles function."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jnp.zeros(n_trials)
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5

        result = calculate_latent_state_percentiles(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance
        )

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
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            percentiles=custom_percentiles,
        )

        assert result.shape == (4, n_trials)

    def test_percentiles_ordered(self) -> None:
        """Percentiles should be in order."""
        n_trials = 50
        key = jax.random.PRNGKey(0)

        smoothed_learning_state_mode = jax.random.normal(key, (n_trials,))
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5

        result = calculate_latent_state_percentiles(
            key, smoothed_learning_state_mode, smoothed_learning_state_variance
        )

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
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            n_samples=10000,
        )

        # Median (index 1) should be close to mode
        np.testing.assert_allclose(
            result[1], smoothed_learning_state_mode, rtol=0.1, atol=0.1
        )


class TestComputeCrossCovarianceMatrix:
    """Tests for the compute_cross_covariance_matrix function."""

    def test_output_shape(self) -> None:
        """Output should be square matrix of size n_trials."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5
        smoother_gain = jnp.ones(n_trials - 1) * 0.8

        result = compute_cross_covariance_matrix(
            smoothed_learning_state_variance, smoother_gain
        )

        assert result.shape == (n_trials, n_trials)

    def test_diagonal_equals_variance(self) -> None:
        """Diagonal entries should equal smoothed variances."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.linspace(0.1, 1.0, n_trials)
        smoother_gain = jnp.ones(n_trials - 1) * 0.8

        result = compute_cross_covariance_matrix(
            smoothed_learning_state_variance, smoother_gain
        )

        np.testing.assert_allclose(
            jnp.diag(result), smoothed_learning_state_variance, rtol=1e-5
        )

    def test_symmetric(self) -> None:
        """Cross-covariance matrix should be symmetric."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.linspace(0.1, 1.0, n_trials)
        smoother_gain = jnp.linspace(0.6, 0.9, n_trials - 1)

        result = compute_cross_covariance_matrix(
            smoothed_learning_state_variance, smoother_gain
        )

        np.testing.assert_allclose(result, result.T, rtol=1e-5)

    def test_positive_semidefinite(self) -> None:
        """Cross-covariance matrix should be positive semi-definite."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.linspace(0.1, 1.0, n_trials)
        smoother_gain = jnp.linspace(0.6, 0.9, n_trials - 1)

        result = compute_cross_covariance_matrix(
            smoothed_learning_state_variance, smoother_gain
        )
        eigenvalues = jnp.linalg.eigvalsh(result)

        # All eigenvalues should be non-negative (within tolerance)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_off_diagonal_decay(self) -> None:
        """Cross-covariance should decay with distance when gains < 1."""
        n_trials = 20
        smoothed_learning_state_variance = jnp.ones(n_trials) * 0.5
        smoother_gain = jnp.ones(n_trials - 1) * 0.7  # < 1

        result = compute_cross_covariance_matrix(
            smoothed_learning_state_variance, smoother_gain
        )

        # Check first row: Cov(0, j) should decrease with j
        first_row = result[0, :]
        for j in range(1, n_trials - 1):
            assert first_row[j] >= first_row[j + 1]

    def test_unit_gain_preserves_covariance(self) -> None:
        """With unit gains, cross-covariance equals variance of later trial."""
        n_trials = 10
        smoothed_learning_state_variance = jnp.linspace(0.1, 1.0, n_trials)
        smoother_gain = jnp.ones(n_trials - 1)  # All 1.0

        result = compute_cross_covariance_matrix(
            smoothed_learning_state_variance, smoother_gain
        )

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
        return (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        )

    def test_output_shape(self, fitted_model_data) -> None:
        """Output should be square matrix of size n_trials."""
        (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        ) = fitted_model_data
        n_trials = len(smoothed_learning_state_mode)
        key = jax.random.PRNGKey(0)

        result = compute_trial_comparison_matrix(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            n_samples=1000,
        )

        assert result.shape == (n_trials, n_trials)

    def test_diagonal_is_half(self, fitted_model_data) -> None:
        """Diagonal entries should be 0.5 (P(x_i > x_i) = 0.5)."""
        (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        ) = fitted_model_data
        key = jax.random.PRNGKey(0)

        result = compute_trial_comparison_matrix(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            n_samples=1000,
        )

        np.testing.assert_allclose(jnp.diag(result), 0.5, rtol=1e-5)

    def test_lower_triangle_is_nan(self, fitted_model_data) -> None:
        """Lower triangle should be NaN."""
        (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        ) = fitted_model_data
        n_trials = len(smoothed_learning_state_mode)
        key = jax.random.PRNGKey(0)

        result = compute_trial_comparison_matrix(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            n_samples=1000,
        )

        # Check lower triangle (excluding diagonal)
        lower_tri_mask = jnp.tril(jnp.ones((n_trials, n_trials), dtype=bool), k=-1)
        assert jnp.all(jnp.isnan(result[lower_tri_mask]))

    def test_upper_triangle_in_bounds(self, fitted_model_data) -> None:
        """Upper triangle values should be probabilities in [0, 1]."""
        (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        ) = fitted_model_data
        n_trials = len(smoothed_learning_state_mode)
        key = jax.random.PRNGKey(0)

        result = compute_trial_comparison_matrix(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            n_samples=1000,
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
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            n_samples=5000,
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
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            n_samples=5000,
        )

        # Upper triangle should be near 0.5
        upper_tri_mask = jnp.triu(jnp.ones((n_trials, n_trials), dtype=bool), k=1)
        upper_values = result[upper_tri_mask]

        np.testing.assert_allclose(upper_values, 0.5, atol=0.1)

    def test_reproducibility_with_same_key(self, fitted_model_data) -> None:
        """Same key should produce same results."""
        (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        ) = fitted_model_data
        key = jax.random.PRNGKey(123)

        result1 = compute_trial_comparison_matrix(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            n_samples=1000,
        )
        result2 = compute_trial_comparison_matrix(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            n_samples=1000,
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
        return (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        )

    def test_same_trial_returns_half(self, model_data) -> None:
        """Comparing trial to itself should return 0.5."""
        (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        ) = model_data
        key = jax.random.PRNGKey(0)

        result = compare_two_trials(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            trial1=5,
            trial2=5,
        )

        assert result == 0.5

    def test_output_is_probability(self, model_data) -> None:
        """Output should be a probability in [0, 1]."""
        (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        ) = model_data
        key = jax.random.PRNGKey(0)

        result = compare_two_trials(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            trial1=0,
            trial2=15,
        )

        assert 0.0 <= result <= 1.0

    def test_symmetric_complement(self, model_data) -> None:
        """P(trial1 > trial2) + P(trial2 > trial1) should equal 1."""
        (
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
        ) = model_data
        key = jax.random.PRNGKey(0)

        p_12 = compare_two_trials(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            trial1=3,
            trial2=12,
        )
        p_21 = compare_two_trials(
            key,
            smoothed_learning_state_mode,
            smoothed_learning_state_variance,
            smoother_gain,
            trial1=12,
            trial2=3,
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

    @staticmethod
    def _reference_loop(matrix, reference_trial, significance_level):
        """The original per-trial loop, kept as the behavioural reference."""
        for j in range(reference_trial + 1, matrix.shape[0]):
            if matrix[reference_trial, j] < significance_level / 2:
                return j
        return None

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_per_trial_loop(self, seed) -> None:
        """The vectorized search returns exactly what the per-trial loop did,
        including NaN entries (never significant) and every reference trial."""
        rng = np.random.default_rng(seed)
        n_trials = 30
        matrix = rng.uniform(0.0, 0.2, size=(n_trials, n_trials))
        matrix[rng.random((n_trials, n_trials)) < 0.2] = np.nan
        found_some = False
        for reference_trial in [-1, 0, 1, 7, n_trials - 2, n_trials - 1, n_trials]:
            for level in (0.01, 0.05, 0.2):
                expected = self._reference_loop(matrix, reference_trial, level)
                found_some |= expected is not None
                result = find_first_significant_trial(
                    jnp.asarray(matrix), reference_trial, level
                )
                assert result == expected
                assert result is None or type(result) is int
        assert found_some  # guard: the comparison covered positive cases


class TestSmithLearningModelTrialComparison:
    """Tests for trial comparison methods on SmithLearningModel class."""

    @pytest.fixture
    def fitted_model(self):
        """Create and fit a model for testing."""
        # Learning data with clear improvement
        responses = jnp.array(
            [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
        )

        model = SmithLearningModel(
            sigma_epsilon=0.3,
            prob_correct_by_chance=0.5,
            initial_state_method="set_initial_direct_from_second_trial",
        )
        model.fit(responses, max_iter=50)
        return model

    def test_compare_trials_requires_fit(self) -> None:
        """compare_trials should raise if model not fitted."""
        model = SmithLearningModel()
        key = jax.random.PRNGKey(0)

        with pytest.raises(NotFittedError, match="not been fitted"):
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
        model = SmithLearningModel()
        key = jax.random.PRNGKey(0)

        with pytest.raises(NotFittedError, match="not been fitted"):
            model.get_trial_comparison_matrix(key)

    def test_get_trial_comparison_matrix_shape(self, fitted_model) -> None:
        """get_trial_comparison_matrix should return correct shape."""
        key = jax.random.PRNGKey(0)
        n_trials = len(fitted_model.smoothed_learning_state_mode)

        result = fitted_model.get_trial_comparison_matrix(key, n_samples=1000)

        assert result.shape == (n_trials, n_trials)

    def test_find_first_significant_improvement_requires_fit(self) -> None:
        """find_first_significant_improvement should raise if not fitted."""
        model = SmithLearningModel()
        key = jax.random.PRNGKey(0)

        with pytest.raises(NotFittedError, match="not been fitted"):
            model.find_first_significant_improvement(key)


# --- Property-Based Tests using Hypothesis ---

from hypothesis import example, given, settings
from hypothesis import strategies as st


def _binary_outcomes(lengths: tuple[int, ...]) -> st.SearchStrategy[jax.Array]:
    """Arbitrary 0/1 outcome sequences whose length is one of ``lengths``.

    The filter and smoother are jitted, so every distinct length compiles
    them again; a couple of fixed lengths keeps each example cheap while the
    drawn outcomes (including all-0 / all-1 runs) explore the input space.
    """
    return st.sampled_from(lengths).flatmap(
        lambda n: st.lists(st.integers(0, 1), min_size=n, max_size=n).map(jnp.asarray)
    )


def _filter_and_smooth(outcomes: jax.Array):
    """Run the filter then the smoother on ``outcomes``.

    Returns
    -------
    filtered : tuple
        ``smith_learning_filter`` outputs (prob, mode, variance,
        one_step_mode, one_step_variance), each of shape (n_trials,).
    smoothed : tuple
        ``smith_learning_smoother`` outputs (mode, variance, prob, gain).
    """
    filtered = smith_learning_filter(outcomes, max_possible_correct=1)
    _, filter_mode, filter_var, one_step_mode, one_step_var = filtered
    smoothed = smith_learning_smoother(
        filter_mode, filter_var, one_step_mode, one_step_var
    )
    return filtered, smoothed


class TestSmithLearningFilterProperties:
    """Property-based tests for smith_learning_filter."""

    @given(_binary_outcomes((8, 25)))
    def test_probability_always_in_bounds(self, outcomes: jax.Array) -> None:
        """Probability of correct response should always be in [0, 1]."""
        prob, _, _, _, _ = smith_learning_filter(outcomes, max_possible_correct=1)
        assert jnp.all(prob >= 0.0)
        assert jnp.all(prob <= 1.0)

    @given(_binary_outcomes((8, 25)))
    def test_variance_always_positive(self, outcomes: jax.Array) -> None:
        """Variance should always be positive."""
        _, _, variance, _, one_step_var = smith_learning_filter(
            outcomes, max_possible_correct=1
        )
        assert jnp.all(variance > 0)
        assert jnp.all(one_step_var > 0)

    @given(
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        st.floats(min_value=0.05, max_value=0.5, allow_nan=False),
    )
    def test_outputs_finite_for_valid_params(
        self, prob_chance: float, sigma: float
    ) -> None:
        """Outputs should be finite for valid parameter combinations."""
        outcomes = jnp.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1])
        prob, mode, variance, _, _ = smith_learning_filter(
            outcomes,
            prob_correct_by_chance=prob_chance,
            sigma_epsilon=sigma,
            max_possible_correct=1,
        )
        assert jnp.all(jnp.isfinite(prob))
        assert jnp.all(jnp.isfinite(mode))
        assert jnp.all(jnp.isfinite(variance))


class TestSmithLearningSmootherProperties:
    """Property-based tests for smith_learning_smoother."""

    @given(_binary_outcomes((8, 25)))
    def test_smoother_probability_in_bounds(self, outcomes: jax.Array) -> None:
        """Smoothed probability should be in [0, 1]."""
        _, (_, _, smooth_prob, _) = _filter_and_smooth(outcomes)
        assert jnp.all(smooth_prob >= 0.0)
        assert jnp.all(smooth_prob <= 1.0)

    @given(_binary_outcomes((8, 25)))
    def test_smoother_variance_non_negative(self, outcomes: jax.Array) -> None:
        """Smoothed variance should be non-negative."""
        _, (_, smooth_var, _, _) = _filter_and_smooth(outcomes)
        assert jnp.all(smooth_var >= 0.0)

    @given(_binary_outcomes((8, 25)))
    def test_smoother_last_equals_filter_last(self, outcomes: jax.Array) -> None:
        """Last smoothed state should equal last filtered state."""
        filtered, (smooth_mode, smooth_var, _, _) = _filter_and_smooth(outcomes)
        _, filter_mode, filter_var, _, _ = filtered
        np.testing.assert_allclose(smooth_mode[-1], filter_mode[-1], rtol=1e-5)
        np.testing.assert_allclose(smooth_var[-1], filter_var[-1], rtol=1e-5)


class TestMaximizationStepProperties:
    """Property-based tests for the maximization step."""

    @given(_binary_outcomes((12, 40)))
    def test_estimated_sigma_positive(self, outcomes: jax.Array) -> None:
        """Estimated sigma_epsilon should be positive."""
        _, (smooth_mode, smooth_var, _, smoother_gain) = _filter_and_smooth(outcomes)
        sigma_est, _, _ = maximization_step(smooth_mode, smooth_var, smoother_gain)
        assert sigma_est > 0


class TestTrialComparisonProperties:
    """Property-based tests for trial comparison functions."""

    @given(_binary_outcomes((6, 15)))
    def test_cross_covariance_symmetry(self, outcomes: jax.Array) -> None:
        """Cross-covariance matrix should be symmetric."""
        _, (_, smooth_var, _, smoother_gain) = _filter_and_smooth(outcomes)
        cross_cov = compute_cross_covariance_matrix(smooth_var, smoother_gain)
        np.testing.assert_allclose(cross_cov, cross_cov.T, rtol=1e-5, atol=1e-10)

    @given(_binary_outcomes((6, 15)))
    def test_comparison_matrix_diagonal_is_half(self, outcomes: jax.Array) -> None:
        """Diagonal of comparison matrix should be 0.5 (comparing trial to itself)."""
        _, (smooth_mode, smooth_var, _, smoother_gain) = _filter_and_smooth(outcomes)
        comp_matrix = compute_trial_comparison_matrix(
            key=jax.random.PRNGKey(42),
            smoothed_learning_state_mode=smooth_mode,
            smoothed_learning_state_variance=smooth_var,
            smoother_gain=smoother_gain,
        )
        np.testing.assert_allclose(jnp.diag(comp_matrix), 0.5, rtol=1e-3)

    @given(_binary_outcomes((6, 15)))
    def test_comparison_probabilities_in_bounds(self, outcomes: jax.Array) -> None:
        """Comparison probabilities (upper triangle) are in [0, 1]; the
        lower triangle is NaN as documented."""
        _, (smooth_mode, smooth_var, _, smoother_gain) = _filter_and_smooth(outcomes)
        comp_matrix = compute_trial_comparison_matrix(
            key=jax.random.PRNGKey(42),
            smoothed_learning_state_mode=smooth_mode,
            smoothed_learning_state_variance=smooth_var,
            smoother_gain=smoother_gain,
        )
        upper = comp_matrix[jnp.triu_indices(len(outcomes))]
        assert jnp.all(upper >= 0.0)
        assert jnp.all(upper <= 1.0)
        assert jnp.all(jnp.isnan(comp_matrix[jnp.tril_indices(len(outcomes), k=-1)]))


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


class TestSmithSGDFitting:
    """Tests for SmithLearningModel.fit_sgd()."""

    @pytest.fixture
    def simple_outcomes(self):
        """Generate simple learning data."""
        from state_space_practice.smith_learning_algorithm import (
            simulate_learning_data,
        )

        outcomes, _ = simulate_learning_data(
            n_trials=100, prob_success_init=0.3, prob_success_final=0.9, seed=42
        )
        return outcomes

    def test_sgd_improves_ll(self, simple_outcomes):
        model = SmithLearningModel(sigma_epsilon=0.1)
        initial_ll = model._e_step(jnp.asarray(simple_outcomes))
        model2 = SmithLearningModel(sigma_epsilon=0.1)
        lls = model2.fit_sgd(simple_outcomes, num_steps=50)
        assert lls[-1] > initial_ll

    def test_sgd_respects_constraints(self, simple_outcomes):
        model = SmithLearningModel(sigma_epsilon=0.1)
        model.fit_sgd(simple_outcomes, num_steps=50)
        assert model.sigma_epsilon > 0

    def test_sgd_model_is_fitted(self, simple_outcomes):
        model = SmithLearningModel(sigma_epsilon=0.1)
        model.fit_sgd(simple_outcomes, num_steps=50)
        assert model.is_fitted
        assert model.log_likelihood_ is not None
        assert model.smoothed_learning_state_mode is not None

    def test_sgd_matches_em_approximately(self, simple_outcomes):
        model_em = SmithLearningModel(sigma_epsilon=0.1)
        model_em.fit(simple_outcomes, max_iter=50)

        model_sgd = SmithLearningModel(sigma_epsilon=0.1)
        model_sgd.fit_sgd(simple_outcomes, num_steps=200)

        # Both should find reasonable sigma_epsilon
        assert abs(model_sgd.sigma_epsilon - model_em.sigma_epsilon) < 0.3

    def test_sgd_respects_initial_state_method(self, simple_outcomes):
        """When initial_state_method='set_initial_to_zero', init params are frozen."""
        model = SmithLearningModel(
            sigma_epsilon=0.1, initial_state_method="set_initial_to_zero"
        )
        model.fit_sgd(simple_outcomes, num_steps=50)
        # Init state should still be 0
        assert model.init_learning_state == 0.0
        assert model.is_fitted

    @pytest.mark.parametrize("prefit", [False, True], ids=["fresh", "fitted"])
    def test_rejected_settings_leave_model_unchanged(self, simple_outcomes, prefit):
        """fit_sgd rejects bad settings before it touches the model."""
        model = SmithLearningModel(sigma_epsilon=0.1)
        if prefit:
            model.fit_sgd(simple_outcomes, num_steps=2)
        # A different length than the prefit, so a stashed length would show.
        before = snapshot_model_state(model)
        with pytest.raises(ValueError, match="num_steps"):
            model.fit_sgd(simple_outcomes[:50], num_steps=-1)
        assert_model_state_unchanged(model, before)

    def test_repeat_fit_sgd_reuses_compiled_step(
        self, simple_outcomes, sgd_step_builds
    ):
        """Refitting rewrites the trained parameters; the loss must not read
        them, or every repeat call misses the step cache and recompiles."""
        builds = sgd_step_builds
        model = SmithLearningModel(sigma_epsilon=0.1)
        for _ in range(3):
            model.fit_sgd(simple_outcomes, num_steps=2)
        assert len(builds) == 1


# ============================================================================
# Integration: learning curve recovery on simulated data
# ============================================================================


@pytest.mark.slow
class TestSmithLearningModelRecovery:
    """Fit SmithLearningModel on simulated sigmoid learning data and verify
    the recovered learning curve tracks the true probability trajectory."""

    @pytest.fixture(scope="class")
    @classmethod
    def fitted(cls):
        outcomes, true_prob = simulate_learning_data(
            n_trials=200,
            prob_success_init=0.125,
            prob_success_final=0.6,
            seed=42,
        )
        model = SmithLearningModel(sigma_epsilon=0.2)
        lls = model.fit(outcomes, max_iter=30)
        return model, true_prob, lls

    def test_ll_monotonic(self, fitted):
        _, _, lls = fitted
        assert_ll_monotonic(lls, tol=1e-3, label="SmithLearningModel")

    def test_learning_curve_correlation(self, fitted):
        model, true_prob, _ = fitted
        smoothed_prob = np.array(model.smoothed_prob_correct_response)
        corr = float(np.corrcoef(smoothed_prob, true_prob)[0, 1])
        assert corr > 0.7, (
            f"Smoothed-vs-true learning curve correlation {corr:.3f} < 0.7"
        )

    def test_early_probability_near_chance(self, fitted):
        model, _, _ = fitted
        early_prob = float(model.smoothed_prob_correct_response[0])
        assert early_prob < 0.3, (
            f"Early smoothed probability {early_prob:.3f} >= 0.3 (true is ~0.125)"
        )

    def test_late_probability_above_chance(self, fitted):
        model, _, _ = fitted
        late_prob = float(model.smoothed_prob_correct_response[-1])
        assert late_prob > 0.4, (
            f"Late smoothed probability {late_prob:.3f} <= 0.4 (true is ~0.6)"
        )

    def test_smoother_reduces_variance(self, fitted):
        model, _, _ = fitted
        filter_var = np.mean(np.array(model.filtered_learning_state_variance))
        smoother_var = np.mean(np.array(model.smoothed_learning_state_variance))
        assert smoother_var <= filter_var * 1.01, (
            f"Smoother variance ({smoother_var:.6f}) not less than "
            f"filter variance ({filter_var:.6f})"
        )


class TestSmithEMRollback:
    """EM loop must roll back state on LL decrease."""

    def test_smith_em_rolls_back(self, caplog) -> None:
        from state_space_practice.tests.conftest import (
            assert_em_rolls_back_on_ll_decrease,
        )

        rng = np.random.default_rng(0)
        n_correct = jnp.asarray(rng.integers(0, 2, size=50).astype(float))
        model = SmithLearningModel(
            init_learning_state=0.0,
            sigma_epsilon=float(jnp.sqrt(0.05)),
            prob_correct_by_chance=0.5,
        )
        assert_em_rolls_back_on_ll_decrease(
            model,
            (n_correct,),
            caplog,
        )

    def test_smith_em_rollback_restores_parameters(self, caplog) -> None:
        rng = np.random.default_rng(1)
        n_correct = jnp.asarray(rng.integers(0, 2, size=50).astype(float))
        model = SmithLearningModel(max_possible_correct=1)
        initial_params = (
            model.sigma_epsilon,
            model.init_learning_state,
            model.init_learning_variance,
        )
        real_e_step = model._e_step
        ll_iter = iter([0.0, -1e6])

        def fake_e_step(*args, **kwargs):
            real_e_step(*args, **kwargs)
            return next(ll_iter)

        model._e_step = fake_e_step
        with caplog.at_level("WARNING"):
            lls = model.fit(n_correct, max_iter=3)

        assert lls == [0.0]
        assert any("rolling back" in r.message.lower() for r in caplog.records)
        np.testing.assert_allclose(
            [
                model.sigma_epsilon,
                model.init_learning_state,
                model.init_learning_variance,
            ],
            initial_params,
            atol=1e-10,
        )

    @pytest.mark.slow
    def test_nonfinite_first_e_step_leaves_model_unfitted(self, caplog) -> None:
        rng = np.random.default_rng(2)
        n_correct = jnp.asarray(rng.integers(0, 2, size=30).astype(float))
        model = SmithLearningModel(max_possible_correct=1)
        # A NaN parameter makes the very first E-step non-finite.
        model.sigma_epsilon = float("nan")

        with caplog.at_level("WARNING"):
            lls = model.fit(n_correct, max_iter=5)

        assert any("non-finite" in r.message.lower() for r in caplog.records)
        assert lls == []
        assert not is_set(model, "log_likelihood_") and model.n_iter_ == 0
        # The NaN posteriors are cleared rather than left looking fitted.
        assert not model.is_fitted
        assert not is_set(model, "smoothed_prob_correct_response")
        assert not is_set(model, "filtered_prob_correct_response")

    @pytest.mark.slow
    def test_nonfinite_later_e_step_rolls_back_to_last_accepted(self, caplog) -> None:
        rng = np.random.default_rng(3)
        n_correct = jnp.asarray(rng.integers(0, 2, size=30).astype(float))
        model = SmithLearningModel(max_possible_correct=1)
        real_e_step = model._e_step
        calls = []

        def e_step_nan_on_third_call(*args, **kwargs):
            ll = real_e_step(*args, **kwargs)
            calls.append(
                {
                    "ll": ll,
                    "sigma_epsilon": model.sigma_epsilon,
                    "init_learning_variance": model.init_learning_variance,
                    "smoothed_mode": np.asarray(model.smoothed_learning_state_mode),
                }
            )
            if len(calls) == 3:
                model.smoothed_learning_state_mode = jnp.full_like(
                    model.smoothed_learning_state_mode, jnp.nan
                )
                return float("nan")
            return ll

        model._e_step = e_step_nan_on_third_call
        with caplog.at_level("WARNING"):
            lls = model.fit(n_correct, max_iter=10, tolerance=1e-12)

        assert len(calls) == 3  # the NaN E-step was reached
        accepted = calls[1]
        # The M-step between E-steps 2 and 3 changed the parameters, so
        # restoring them is observable.
        param_keys = ("sigma_epsilon", "init_learning_variance")
        assert [accepted[k] for k in param_keys] != [calls[2][k] for k in param_keys]
        assert lls == [calls[0]["ll"], accepted["ll"]]
        assert np.all(np.isfinite(lls))
        assert model.log_likelihood_ == accepted["ll"]
        assert model.n_iter_ == 2
        assert model.sigma_epsilon == accepted["sigma_epsilon"]
        assert model.init_learning_variance == accepted["init_learning_variance"]
        np.testing.assert_array_equal(
            model.smoothed_learning_state_mode, accepted["smoothed_mode"]
        )
        assert any("rolling back" in r.message.lower() for r in caplog.records)


class TestSmithMStepExactness:
    """The M-step maximises the expected complete-data log-likelihood.

    The objective is written out here independently of the implementation:
    with ``x_1 ~ N(x_0, P_0 + s2)`` and ``x_{k+1} ~ N(x_k, s2)``,

        Q(s2, x_0, P_0) = -1/2 log(P_0 + s2) - ((m_1 - x_0)^2 + P_1)/(2 (P_0 + s2))
                          - (T-1)/2 log s2 - S / (2 s2),

    S = sum_k E[(x_{k+1} - x_k)^2 | y]. The returned parameters must be a
    stationary point (central finite differences) and must not decrease Q.
    """

    @staticmethod
    def _objective(stats, s2, x0, p0):
        m, P, G = stats
        n = m.shape[0]
        S = np.sum((m[1:] - m[:-1]) ** 2 + P[1:] + P[:-1] - 2 * P[1:] * G)
        v = p0 + s2
        return (
            -0.5 * np.log(v)
            - ((m[0] - x0) ** 2 + P[0]) / (2 * v)
            - 0.5 * (n - 1) * np.log(s2)
            - S / (2 * s2)
        )

    @pytest.mark.slow
    @given(
        seed=st.integers(0, 10_000),
        sigma=st.floats(0.1, 0.8),
        init_var=st.floats(0.05, 2.0),
    )
    @example(seed=0, sigma=0.75, init_var=0.25)  # P_0 >= 0 binds
    @settings(max_examples=8, deadline=None)
    def test_m_step_is_stationary_and_ascends(self, seed, sigma, init_var):
        rng = np.random.default_rng(seed)
        n_trials = 25
        x = np.cumsum(rng.normal(0, 0.4, n_trials)) + 1.0
        y = (rng.random(n_trials) < 1 / (1 + np.exp(-x))).astype(int)
        model = SmithLearningModel(
            max_possible_correct=1,
            sigma_epsilon=sigma,
            init_learning_variance=init_var,
            init_learning_state=0.2,
        )
        model._e_step(jnp.asarray(y))
        stats = tuple(
            np.asarray(a, dtype=float)
            for a in (
                model.smoothed_learning_state_mode,
                model.smoothed_learning_state_variance,
                model.smoother_gain,
            )
        )
        old = (sigma**2, 0.2, init_var)
        model._m_step(jnp.asarray(y))
        new = (
            model.sigma_epsilon**2,
            model.init_learning_state,
            model.init_learning_variance,
        )
        q_old = self._objective(stats, *old)
        q_new = self._objective(stats, *new)
        assert q_new >= q_old - 1e-10, (q_old, q_new)

        # Stationarity in s2 and x_0 (always interior) ...
        eps = 1e-6
        for i in (0, 1):
            up = list(new)
            dn = list(new)
            up[i] += eps
            dn[i] -= eps
            grad = (self._objective(stats, *up) - self._objective(stats, *dn)) / (
                2 * eps
            )
            assert abs(grad) < 1e-5 * max(1.0, abs(q_new)), (i, grad, new)
        # ... and in P_0 unless the optimum P_{1|T} - s2 is clipped at the
        # floor, where Q must be non-increasing in P_0 (KKT).
        up = list(new)
        up[2] += eps
        dn = list(new)
        dn[2] = max(new[2] - eps, 0.0)
        grad_p0 = (self._objective(stats, *up) - self._objective(stats, *dn)) / (
            up[2] - dn[2]
        )
        if new[2] > 1e-6:
            assert abs(grad_p0) < 1e-4, (grad_p0, new)
        else:
            assert grad_p0 <= 1e-6, (grad_p0, new)

    @pytest.mark.slow
    def test_old_init_variance_update_was_not_the_maximiser(self):
        """Guard for the regression: P_0 = P_{1|T} leaves ascent on the table."""
        rng = np.random.default_rng(1)
        x = np.cumsum(rng.normal(0, 0.3, 40))
        y = (rng.random(40) < 1 / (1 + np.exp(-x))).astype(int)
        model = SmithLearningModel(
            max_possible_correct=1, init_learning_variance=0.5, sigma_epsilon=0.4
        )
        model._e_step(jnp.asarray(y))
        stats = tuple(
            np.asarray(a, dtype=float)
            for a in (
                model.smoothed_learning_state_mode,
                model.smoothed_learning_state_variance,
                model.smoother_gain,
            )
        )
        model._m_step(jnp.asarray(y))
        s2, x0, p0 = (
            model.sigma_epsilon**2,
            model.init_learning_state,
            model.init_learning_variance,
        )
        old_rule = self._objective(stats, s2, x0, float(stats[1][0]))
        new_rule = self._objective(stats, s2, x0, p0)
        assert p0 > 1e-6  # interior optimum on this data
        assert new_rule > old_rule + 0.01, (old_rule, new_rule)


@pytest.mark.slow
def test_sigma_recovery_evidence_vs_em_as_statistics():
    """Process-noise recovery over 3 seeds (300 Bernoulli trials, sigma^2=0.05).

    Reference: the exact maximum-likelihood sigma^2 on a grid (quadrature).
    The model's Laplace evidence (the fit_sgd objective) peaks at the same
    grid value on every seed. EM's fixed point is biased upwards by the
    Laplace E-step (observed EM / exact = 1.6, 1.6, 2.1): pinned, so a change
    in either direction is noticed.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_oracle_choice import _exact_smith

    grid = np.linspace(-10, 10, 401)
    s2_grid = np.array([0.01, 0.02, 0.035, 0.05, 0.07, 0.1, 0.15, 0.2])
    ratios, report = [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.normal(0, np.sqrt(0.05), 300))
        y = (rng.random(300) < 1 / (1 + np.exp(-x))).astype(int)
        exact = [_exact_smith(y, 1, v, 0.0, v, 0.0, grid=grid)[2] for v in s2_grid]
        evidence = [
            SmithLearningModel(
                max_possible_correct=1,
                sigma_epsilon=float(np.sqrt(v)),
                init_learning_variance=float(v),
            )._e_step(jnp.asarray(y))
            for v in s2_grid
        ]
        s2_exact = s2_grid[np.argmax(exact)]
        model = SmithLearningModel(max_possible_correct=1, sigma_epsilon=0.5)
        model.fit(jnp.asarray(y), max_iter=60)
        ratios.append(model.sigma_epsilon**2 / s2_exact)
        report.append((s2_exact, s2_grid[np.argmax(evidence)], model.sigma_epsilon**2))
        assert s2_grid[np.argmax(evidence)] == s2_exact, report
    msg = f"(exact MLE, evidence argmax, EM) per seed: {report}"
    assert all(1.2 < r < 2.8 for r in ratios), msg


class TestDifferentiableNewtonDoesNotOscillate:
    """Regression: the Laplace mode solvers were not reliable.

    The differentiable (fit_sgd) Newton path took full steps: for N=10, y=0
    and a prior N(3, 4) the logistic is saturated at the prior mean, the full
    step jumps to -10.5 (saturated the other way) and the iteration
    oscillated between 3.0 and -10.5, ending at 2.98 instead of the mode
    -1.956. The default path used jax.scipy BFGS, which stopped early on a
    line-search failure (mode 2.708 vs the exact 2.647 on trial 3 of the
    sequence below). Both filter paths now use the line-searched Newton.
    """

    def test_newton_matches_bfgs_and_exact_mode(self):
        from functools import partial

        from scipy.optimize import brentq

        from state_space_practice.smith_learning_algorithm import (
            _approximate_gaussian_newton,
            _log_posterior_objective,
            approximate_gaussian,
        )

        m, v, y, n = 3.0, 4.0, 0, 10
        f = partial(
            _log_posterior_objective,
            learning_state_prev=m,
            variance_prev=v,
            n_correct_in_trial=y,
            max_possible_correct=n,
            bias=0.0,
        )
        mode = brentq(lambda x: y - n / (1 + np.exp(-x)) - (x - m) / v, -30, 30)
        newton_mode, newton_var, _ = _approximate_gaussian_newton(f, jnp.array([m]))
        bfgs_mode, bfgs_var = approximate_gaussian(f, jnp.array([m]))
        assert abs(float(newton_mode[0]) - mode) < 1e-8
        assert abs(float(bfgs_mode[0]) - mode) < 1e-4
        np.testing.assert_allclose(newton_var, bfgs_var, rtol=1e-3)

    def test_newton_does_not_zigzag_across_the_mode(self):
        """A broad prior on the wrong saturated side: y=0, prior N(5.163, 11.51).

        Taking the largest Armijo-acceptable step zigzags across the mode
        (5.16, -5.58, 4.68, -5.21, ...) and ten iterations end 4 nats short
        of it; the best acceptable step converges.
        """
        from functools import partial

        from scipy.optimize import brentq

        from state_space_practice.multinomial_choice import NEWTON_GAP_TOL
        from state_space_practice.smith_learning_algorithm import (
            _approximate_gaussian_newton,
            _log_posterior_objective,
        )

        m, v, y, n = 5.163, 11.51, 0, 1
        f = partial(
            _log_posterior_objective,
            learning_state_prev=m,
            variance_prev=v,
            n_correct_in_trial=y,
            max_possible_correct=n,
            bias=0.0,
        )
        mode = brentq(lambda x: y - n / (1 + np.exp(-x)) - (x - m) / v, -30, 30)
        newton_mode, _, gap = _approximate_gaussian_newton(f, jnp.array([m]))
        assert abs(float(newton_mode[0]) - mode) < 1e-8
        assert float(gap) < NEWTON_GAP_TOL

    def test_filter_modes_are_exact_one_step_modes(self):
        from scipy.optimize import brentq

        rng = np.random.default_rng(0)
        y = np.concatenate([np.full(8, 10), np.zeros(4), np.full(4, 10)]).astype(int)
        y = np.clip(y + rng.integers(-1, 1, y.size), 0, 10)
        kwargs = dict(
            init_learning_variance=4.0,
            sigma_epsilon=1.5,
            max_possible_correct=10,
        )
        for differentiable in (False, True):
            _, mode, _, pred_mode, pred_var = (
                np.asarray(a)
                for a in smith_learning_filter(
                    jnp.asarray(y), differentiable=differentiable, **kwargs
                )
            )
            # Each filtered mode solves y - N sigmoid(x) = (x - m_pred) / P_pred.
            exact = [
                brentq(
                    lambda x, yt=y[t], m=pred_mode[t], v=pred_var[t]: (
                        yt - 10 / (1 + np.exp(-x)) - (x - m) / v
                    ),
                    -40,
                    40,
                )
                for t in range(y.size)
            ]
            np.testing.assert_allclose(mode, exact, atol=1e-8)
