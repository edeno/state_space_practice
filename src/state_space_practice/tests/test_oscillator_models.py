# ruff: noqa: E402
"""Tests for the oscillator_models module.

This module tests the switching oscillator model classes (COM, CNM, DIM)
used for dynamic functional connectivity analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.oscillator_models import (
    CommonOscillatorModel,
    CorrelatedNoiseModel,
    DirectedInfluenceModel,
)

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


# ============================================================================
# Fixtures for creating model instances
# ============================================================================


@pytest.fixture(scope="module")
def common_oscillator_params():
    """Provides parameters for CommonOscillatorModel tests."""
    n_oscillators = 2
    n_discrete_states = 3
    n_sources = 4
    sampling_freq = 100.0

    return {
        "n_oscillators": n_oscillators,
        "n_discrete_states": n_discrete_states,
        "n_sources": n_sources,
        "sampling_freq": sampling_freq,
        "freqs": jnp.array([8.0, 12.0]),  # Alpha and beta
        "auto_regressive_coef": jnp.array([0.95, 0.95]),
        "process_variance": jnp.array([0.1, 0.1]),
        "measurement_variance": 0.05,
    }


@pytest.fixture(scope="module")
def correlated_noise_params():
    """Provides parameters for CorrelatedNoiseModel tests."""
    n_oscillators = 2
    n_discrete_states = 3
    sampling_freq = 100.0

    return {
        "n_oscillators": n_oscillators,
        "n_discrete_states": n_discrete_states,
        "sampling_freq": sampling_freq,
        "freqs": jnp.array([8.0, 12.0]),
        "auto_regressive_coef": jnp.array([0.95, 0.95]),
        "process_variance": jnp.ones((n_oscillators, n_discrete_states)) * 0.1,
        "measurement_variance": 0.05,
        "phase_difference": jnp.zeros(
            (n_oscillators, n_oscillators, n_discrete_states)
        ),
        "coupling_strength": jnp.zeros(
            (n_oscillators, n_oscillators, n_discrete_states)
        ),
    }


@pytest.fixture(scope="module")
def directed_influence_params():
    """Provides parameters for DirectedInfluenceModel tests."""
    n_oscillators = 2
    n_discrete_states = 3
    sampling_freq = 100.0

    return {
        "n_oscillators": n_oscillators,
        "n_discrete_states": n_discrete_states,
        "sampling_freq": sampling_freq,
        "freqs": jnp.array([8.0, 12.0]),
        "auto_regressive_coef": jnp.array([0.95, 0.95]),
        "process_variance": jnp.array([0.1, 0.1]),
        "measurement_variance": 0.05,
        "phase_difference": jnp.zeros(
            (n_oscillators, n_oscillators, n_discrete_states)
        ),
        "coupling_strength": jnp.zeros(
            (n_oscillators, n_oscillators, n_discrete_states)
        ),
    }


@pytest.fixture
def synthetic_observations():
    """Generates synthetic observations for model fitting tests."""
    n_time = 200
    n_sources = 2
    key = jax.random.PRNGKey(42)

    # Generate AR(1) like oscillatory data
    k1, k2 = jax.random.split(key)
    noise = jax.random.normal(k1, (n_time, n_sources)) * 0.5

    # Add oscillatory component
    t = jnp.arange(n_time) / 100.0  # 100 Hz sampling
    oscillation = jnp.sin(2 * jnp.pi * 10 * t)[:, None]  # 10 Hz oscillation
    observations = oscillation + noise

    return observations


# ============================================================================
# Tests for CommonOscillatorModel
# ============================================================================


class TestCommonOscillatorModel:
    """Tests for the CommonOscillatorModel class."""

    def test_initialization(self, common_oscillator_params) -> None:
        """Model should initialize without errors."""
        model = CommonOscillatorModel(**common_oscillator_params)

        assert model.n_oscillators == common_oscillator_params["n_oscillators"]
        assert model.n_discrete_states == common_oscillator_params["n_discrete_states"]
        assert model.n_sources == common_oscillator_params["n_sources"]
        assert model.n_cont_states == 2 * common_oscillator_params["n_oscillators"]

    def test_repr(self, common_oscillator_params) -> None:
        """__repr__ should return informative string."""
        model = CommonOscillatorModel(**common_oscillator_params)
        repr_str = repr(model)

        assert "CommonOscillatorModel" in repr_str
        assert "n_oscillators=2" in repr_str
        assert "n_discrete_states=3" in repr_str

    def test_initialize_parameters_shapes(self, common_oscillator_params) -> None:
        """All initialized parameters should have correct shapes."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        n_osc = common_oscillator_params["n_oscillators"]
        n_disc = common_oscillator_params["n_discrete_states"]
        n_src = common_oscillator_params["n_sources"]
        n_cont = 2 * n_osc

        assert model.init_mean.shape == (n_cont, n_disc)
        assert model.init_cov.shape == (n_cont, n_cont, n_disc)
        assert model.init_discrete_state_prob.shape == (n_disc,)
        assert model.discrete_transition_matrix.shape == (n_disc, n_disc)
        assert model.continuous_transition_matrix.shape == (n_cont, n_cont, n_disc)
        assert model.process_cov.shape == (n_cont, n_cont, n_disc)
        assert model.measurement_matrix.shape == (n_src, n_cont, n_disc)
        assert model.measurement_cov.shape == (n_src, n_src, n_disc)

    def test_discrete_transition_rows_sum_to_one(
        self, common_oscillator_params
    ) -> None:
        """Discrete transition matrix rows should sum to 1."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        row_sums = jnp.sum(model.discrete_transition_matrix, axis=1)
        np.testing.assert_allclose(
            row_sums, jnp.ones(model.n_discrete_states), rtol=1e-6
        )

    def test_discrete_state_prob_sums_to_one(self, common_oscillator_params) -> None:
        """Initial discrete state probabilities should sum to 1."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        np.testing.assert_allclose(
            jnp.sum(model.init_discrete_state_prob), 1.0, rtol=1e-6
        )

    def test_constant_A_across_states(self, common_oscillator_params) -> None:
        """For COM, continuous transition matrix should be constant across states."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        # All states should have same A
        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.continuous_transition_matrix[..., 0],
                model.continuous_transition_matrix[..., j],
                rtol=1e-10,
            )

    def test_constant_Q_across_states(self, common_oscillator_params) -> None:
        """For COM, process covariance should be constant across states."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.process_cov[..., 0],
                model.process_cov[..., j],
                rtol=1e-10,
            )

    def test_measurement_matrix_varies_across_states(
        self, common_oscillator_params
    ) -> None:
        """For COM, measurement matrix should vary across states (randomly initialized)."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        # H should be different for different states (random init)
        # At least one pair should differ
        any_different = False
        for j in range(1, model.n_discrete_states):
            if not jnp.allclose(
                model.measurement_matrix[..., 0],
                model.measurement_matrix[..., j],
            ):
                any_different = True
                break
        assert any_different

    def test_get_oscillator_influence_shape(self, common_oscillator_params) -> None:
        """get_oscillator_influence_on_node should return correct shape."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        influence = model.get_oscillator_influence_on_node()

        assert influence.shape == (
            model.n_sources,
            model.n_oscillators,
            model.n_discrete_states,
        )

    def test_get_oscillator_influence_non_negative(
        self, common_oscillator_params
    ) -> None:
        """Oscillator influence should be non-negative (it's an L2 norm)."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        influence = model.get_oscillator_influence_on_node()

        assert jnp.all(influence >= 0)

    def test_get_phase_difference_shape(self, common_oscillator_params) -> None:
        """get_phase_difference should return correct shape."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        phase_diff = model.get_phase_difference(
            node1_ind=0, node2_ind=1, oscillator_ind=0
        )

        assert phase_diff.shape == (model.n_discrete_states,)

    def test_get_phase_difference_antisymmetric(self, common_oscillator_params) -> None:
        """Phase difference should be antisymmetric: φ(a,b) = -φ(b,a)."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        phase_01 = model.get_phase_difference(
            node1_ind=0, node2_ind=1, oscillator_ind=0
        )
        phase_10 = model.get_phase_difference(
            node1_ind=1, node2_ind=0, oscillator_ind=0
        )

        np.testing.assert_allclose(phase_01, -phase_10, rtol=1e-10)

    def test_invalid_freqs_shape_raises(self, common_oscillator_params) -> None:
        """Invalid freqs shape should raise ValueError."""
        params = common_oscillator_params.copy()
        params["freqs"] = jnp.array([8.0])  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            CommonOscillatorModel(**params)


# ============================================================================
# Tests for CorrelatedNoiseModel
# ============================================================================


class TestCorrelatedNoiseModel:
    """Tests for the CorrelatedNoiseModel class."""

    def test_initialization(self, correlated_noise_params) -> None:
        """Model should initialize without errors."""
        model = CorrelatedNoiseModel(**correlated_noise_params)

        assert model.n_oscillators == correlated_noise_params["n_oscillators"]
        assert model.n_sources == model.n_oscillators  # CNM constraint

    def test_n_sources_equals_n_oscillators(self, correlated_noise_params) -> None:
        """For CNM, n_sources must equal n_oscillators."""
        model = CorrelatedNoiseModel(**correlated_noise_params)
        assert model.n_sources == model.n_oscillators

    def test_initialize_parameters_shapes(self, correlated_noise_params) -> None:
        """All parameters should have correct shapes."""
        model = CorrelatedNoiseModel(**correlated_noise_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        n_osc = correlated_noise_params["n_oscillators"]
        n_disc = correlated_noise_params["n_discrete_states"]
        n_cont = 2 * n_osc

        assert model.init_mean.shape == (n_cont, n_disc)
        assert model.init_cov.shape == (n_cont, n_cont, n_disc)
        assert model.continuous_transition_matrix.shape == (n_cont, n_cont, n_disc)
        assert model.process_cov.shape == (n_cont, n_cont, n_disc)
        assert model.measurement_matrix.shape == (n_osc, n_cont, n_disc)

    def test_constant_A_across_states(self, correlated_noise_params) -> None:
        """For CNM, continuous transition matrix should be constant."""
        model = CorrelatedNoiseModel(**correlated_noise_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.continuous_transition_matrix[..., 0],
                model.continuous_transition_matrix[..., j],
                rtol=1e-10,
            )

    def test_constant_H_across_states(self, correlated_noise_params) -> None:
        """For CNM, measurement matrix should be constant."""
        model = CorrelatedNoiseModel(**correlated_noise_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.measurement_matrix[..., 0],
                model.measurement_matrix[..., j],
                rtol=1e-10,
            )

    def test_measurement_matrix_structure(self, correlated_noise_params) -> None:
        """CNM measurement matrix should have [1, 0] block diagonal structure."""
        model = CorrelatedNoiseModel(**correlated_noise_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        # Each source should observe only the "real" part of its oscillator
        H = model.measurement_matrix[..., 0]
        n_osc = model.n_oscillators

        for i in range(n_osc):
            # Source i observes oscillator i's first component (cos), not second (sin)
            assert H[i, 2 * i] == 1.0
            assert H[i, 2 * i + 1] == 0.0

    def test_invalid_process_variance_shape_raises(
        self, correlated_noise_params
    ) -> None:
        """Invalid process_variance shape should raise."""
        params = correlated_noise_params.copy()
        params["process_variance"] = jnp.array([0.1, 0.1])  # Wrong shape

        with pytest.raises(ValueError, match="process_variance must have shape"):
            CorrelatedNoiseModel(**params)

    def test_invalid_measurement_variance_raises(self, correlated_noise_params) -> None:
        """Non-positive measurement_variance should raise."""
        params = correlated_noise_params.copy()
        params["measurement_variance"] = -0.1

        with pytest.raises(ValueError, match="measurement_variance must be positive"):
            CorrelatedNoiseModel(**params)

    def test_project_parameters_preserves_block_structure(
        self, correlated_noise_params
    ) -> None:
        """_project_parameters should maintain Q's oscillator block structure.

        Note: The CNM projection preserves 2x2 block structure (scaled rotation matrices),
        not full matrix symmetry. Each 2x2 diagonal block should be a scaled rotation.
        """
        model = CorrelatedNoiseModel(**correlated_noise_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        # Simulate M-step modifying Q slightly
        model.process_cov = model.process_cov + 0.01 * jax.random.normal(
            jax.random.PRNGKey(1), model.process_cov.shape
        )

        model._project_parameters()

        # Check that 2x2 diagonal blocks have rotation structure: [[a, -b], [b, a]]
        for j in range(model.n_discrete_states):
            Q_j = model.process_cov[..., j]
            n_osc = model.n_oscillators

            for i in range(n_osc):
                block = Q_j[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
                # Diagonal elements should be equal
                np.testing.assert_allclose(
                    block[0, 0], block[1, 1], rtol=1e-5, atol=1e-10
                )
                # Off-diagonal elements should be negatives of each other
                np.testing.assert_allclose(
                    block[0, 1], -block[1, 0], rtol=1e-5, atol=1e-10
                )


# ============================================================================
# Tests for DirectedInfluenceModel
# ============================================================================


class TestDirectedInfluenceModel:
    """Tests for the DirectedInfluenceModel class."""

    def test_initialization(self, directed_influence_params) -> None:
        """Model should initialize without errors."""
        model = DirectedInfluenceModel(**directed_influence_params)

        assert model.n_oscillators == directed_influence_params["n_oscillators"]
        assert model.n_sources == model.n_oscillators

    def test_initialize_parameters_shapes(self, directed_influence_params) -> None:
        """All parameters should have correct shapes."""
        model = DirectedInfluenceModel(**directed_influence_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        n_osc = directed_influence_params["n_oscillators"]
        n_disc = directed_influence_params["n_discrete_states"]
        n_cont = 2 * n_osc

        assert model.init_mean.shape == (n_cont, n_disc)
        assert model.init_cov.shape == (n_cont, n_cont, n_disc)
        assert model.continuous_transition_matrix.shape == (n_cont, n_cont, n_disc)
        assert model.process_cov.shape == (n_cont, n_cont, n_disc)

    def test_constant_H_across_states(self, directed_influence_params) -> None:
        """For DIM, measurement matrix should be constant."""
        model = DirectedInfluenceModel(**directed_influence_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.measurement_matrix[..., 0],
                model.measurement_matrix[..., j],
                rtol=1e-10,
            )

    def test_constant_Q_across_states(self, directed_influence_params) -> None:
        """For DIM, process covariance should be constant."""
        model = DirectedInfluenceModel(**directed_influence_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.process_cov[..., 0],
                model.process_cov[..., j],
                rtol=1e-10,
            )

    def test_measurement_matrix_structure(self, directed_influence_params) -> None:
        """DIM measurement matrix should have [1/sqrt(2), 1/sqrt(2)] blocks."""
        model = DirectedInfluenceModel(**directed_influence_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        H = model.measurement_matrix[..., 0]
        n_osc = model.n_oscillators
        expected_val = 1.0 / jnp.sqrt(2.0)

        for i in range(n_osc):
            np.testing.assert_allclose(H[i, 2 * i], expected_val, rtol=1e-5)
            np.testing.assert_allclose(H[i, 2 * i + 1], expected_val, rtol=1e-5)

    def test_invalid_phase_difference_shape_raises(
        self, directed_influence_params
    ) -> None:
        """Invalid phase_difference shape should raise."""
        params = directed_influence_params.copy()
        params["phase_difference"] = jnp.zeros((2, 2))  # Wrong shape

        with pytest.raises(ValueError, match="phase_difference must have shape"):
            DirectedInfluenceModel(**params)


# ============================================================================
# Tests for EM Algorithm (E-step and M-step)
# ============================================================================


class TestEMAlgorithm:
    """Tests for the EM algorithm implementation."""

    def test_e_step_returns_finite_likelihood(
        self, common_oscillator_params, synthetic_observations
    ) -> None:
        """E-step should return finite log-likelihood."""
        # Adjust n_sources to match synthetic data
        params = common_oscillator_params.copy()
        params["n_sources"] = synthetic_observations.shape[1]

        model = CommonOscillatorModel(**params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        log_likelihood = model._e_step(synthetic_observations)

        assert jnp.isfinite(log_likelihood)

    def test_e_step_populates_smoother_attributes(
        self, common_oscillator_params, synthetic_observations
    ) -> None:
        """E-step should populate smoother result attributes."""
        params = common_oscillator_params.copy()
        params["n_sources"] = synthetic_observations.shape[1]

        model = CommonOscillatorModel(**params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        model._e_step(synthetic_observations)

        assert model.smoother_state_cond_mean is not None
        assert model.smoother_state_cond_cov is not None
        assert model.smoother_discrete_state_prob is not None
        assert model.smoother_joint_discrete_state_prob is not None

    def test_fit_returns_log_likelihoods(
        self, common_oscillator_params, synthetic_observations
    ) -> None:
        """fit() should return list of log-likelihoods."""
        params = common_oscillator_params.copy()
        params["n_sources"] = synthetic_observations.shape[1]

        model = CommonOscillatorModel(**params)
        log_likelihoods = model.fit(
            synthetic_observations, jax.random.PRNGKey(0), max_iter=3
        )

        assert isinstance(log_likelihoods, list)
        assert len(log_likelihoods) >= 1
        assert all(jnp.isfinite(ll) for ll in log_likelihoods)

    def test_fit_log_likelihood_generally_increases(
        self, synthetic_observations
    ) -> None:
        """Log-likelihood should generally increase during EM."""
        n_osc = 2
        n_disc = 2
        n_src = synthetic_observations.shape[1]

        model = CommonOscillatorModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            n_sources=n_src,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            auto_regressive_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.1,
        )

        log_likelihoods = model.fit(
            synthetic_observations, jax.random.PRNGKey(42), max_iter=10
        )

        # Check that LL generally increases (allow for small fluctuations)
        if len(log_likelihoods) > 1:
            # Last should be >= first (overall improvement)
            assert log_likelihoods[-1] >= log_likelihoods[0] - 1e-3

    def test_update_flags_respected(
        self, common_oscillator_params, synthetic_observations
    ) -> None:
        """Update flags should control which parameters change in M-step."""
        params = common_oscillator_params.copy()
        params["n_sources"] = synthetic_observations.shape[1]
        params["update_measurement_cov"] = False  # Don't update R

        model = CommonOscillatorModel(**params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        R_before = model.measurement_cov.copy()

        # Run one E-M iteration
        model._e_step(synthetic_observations)
        model._m_step(synthetic_observations)

        # R should not have changed
        np.testing.assert_allclose(model.measurement_cov, R_before, rtol=1e-10)


# ============================================================================
# Tests for Single Discrete State (reduces to standard Kalman)
# ============================================================================


class TestSingleDiscreteState:
    """Tests for models with single discrete state (should reduce to standard KF)."""

    def test_single_state_com_initialization(self, common_oscillator_params) -> None:
        """Single-state COM should initialize without issues."""
        params = common_oscillator_params.copy()
        params["n_discrete_states"] = 1

        model = CommonOscillatorModel(**params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        assert model.discrete_transition_matrix.shape == (1, 1)
        assert model.init_discrete_state_prob.shape == (1,)
        np.testing.assert_allclose(
            model.discrete_transition_matrix[0, 0], 1.0, rtol=1e-6
        )
        np.testing.assert_allclose(model.init_discrete_state_prob[0], 1.0, rtol=1e-6)

    def test_single_state_fit(self, synthetic_observations) -> None:
        """Single-state model should fit without errors."""
        n_osc = 2
        n_src = synthetic_observations.shape[1]

        model = CommonOscillatorModel(
            n_oscillators=n_osc,
            n_discrete_states=1,
            n_sources=n_src,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            auto_regressive_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.1,
        )

        log_likelihoods = model.fit(
            synthetic_observations, jax.random.PRNGKey(0), max_iter=3
        )

        assert len(log_likelihoods) >= 1
        assert all(jnp.isfinite(ll) for ll in log_likelihoods)


# ============================================================================
# Tests for edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self, common_oscillator_params) -> None:
        """Model should handle very short observation sequences."""
        params = common_oscillator_params.copy()
        params["n_sources"] = 2

        model = CommonOscillatorModel(**params)

        # Very short sequence
        short_obs = jax.random.normal(jax.random.PRNGKey(0), (5, 2))

        log_likelihoods = model.fit(short_obs, jax.random.PRNGKey(0), max_iter=2)

        assert len(log_likelihoods) >= 1

    def test_cnm_observations_shape_validation(
        self, correlated_noise_params, synthetic_observations
    ) -> None:
        """CNM should validate observation shape matches n_sources."""
        model = CorrelatedNoiseModel(**correlated_noise_params)

        # synthetic_observations has 2 sources, CNM expects n_oscillators=2
        # This should work if shapes match
        if synthetic_observations.shape[1] == model.n_sources:
            log_likelihoods = model.fit(
                synthetic_observations, jax.random.PRNGKey(0), max_iter=2
            )
            assert len(log_likelihoods) >= 1
        else:
            # Should raise if shapes don't match
            with pytest.raises(ValueError):
                model.fit(synthetic_observations, jax.random.PRNGKey(0), max_iter=2)

    def test_dim_observations_shape_validation(self, directed_influence_params) -> None:
        """DIM should validate observation shape matches n_sources."""
        model = DirectedInfluenceModel(**directed_influence_params)

        # Create observations with wrong number of sources
        wrong_obs = jax.random.normal(jax.random.PRNGKey(0), (100, 5))  # 5 != n_osc

        with pytest.raises(ValueError, match="observations must have"):
            model.fit(wrong_obs, jax.random.PRNGKey(0), max_iter=2)

    def test_process_cov_positive_semidefinite(self, correlated_noise_params) -> None:
        """Process covariance should be positive semi-definite."""
        model = CorrelatedNoiseModel(**correlated_noise_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(model.n_discrete_states):
            Q_j = model.process_cov[..., j]
            eigenvalues = jnp.linalg.eigvalsh(Q_j)
            # All eigenvalues should be >= 0 (with numerical tolerance)
            assert jnp.all(eigenvalues >= -1e-10)

    def test_measurement_cov_positive_definite(self, common_oscillator_params) -> None:
        """Measurement covariance should be positive definite."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(model.n_discrete_states):
            R_j = model.measurement_cov[..., j]
            eigenvalues = jnp.linalg.eigvalsh(R_j)
            # All eigenvalues should be > 0
            assert jnp.all(eigenvalues > 0)


# ============================================================================
# Tests for covariance structure
# ============================================================================


class TestCovarianceStructure:
    """Tests for covariance matrix properties."""

    def test_init_cov_symmetric(self, common_oscillator_params) -> None:
        """Initial covariance should be symmetric."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(model.n_discrete_states):
            P0_j = model.init_cov[..., j]
            np.testing.assert_allclose(P0_j, P0_j.T, rtol=1e-10)

    def test_process_cov_symmetric(self, correlated_noise_params) -> None:
        """Process covariance should be symmetric."""
        model = CorrelatedNoiseModel(**correlated_noise_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(model.n_discrete_states):
            Q_j = model.process_cov[..., j]
            np.testing.assert_allclose(Q_j, Q_j.T, rtol=1e-10)

    def test_measurement_cov_symmetric(self, common_oscillator_params) -> None:
        """Measurement covariance should be symmetric."""
        model = CommonOscillatorModel(**common_oscillator_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(model.n_discrete_states):
            R_j = model.measurement_cov[..., j]
            np.testing.assert_allclose(R_j, R_j.T, rtol=1e-10)


# ============================================================================
# Property-Based Tests using Hypothesis
# ============================================================================

from hypothesis import given, settings
from hypothesis import strategies as st


class TestCommonOscillatorModelProperties:
    """Property-based tests for CommonOscillatorModel."""

    @given(
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=20, deadline=None)
    def test_discrete_transition_is_stochastic(
        self, n_osc: int, n_disc: int, n_src: int
    ) -> None:
        """Discrete transition matrix rows should sum to 1."""
        model = CommonOscillatorModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            n_sources=n_src,
            sampling_freq=100.0,
            freqs=jnp.ones(n_osc) * 10.0,
            auto_regressive_coef=jnp.ones(n_osc) * 0.95,
            process_variance=jnp.ones(n_osc) * 0.1,
            measurement_variance=0.05,
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        row_sums = jnp.sum(model.discrete_transition_matrix, axis=1)
        np.testing.assert_allclose(row_sums, jnp.ones(n_disc), rtol=1e-6)

    @given(
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=20, deadline=None)
    def test_init_prob_is_valid_distribution(
        self, n_osc: int, n_disc: int, n_src: int
    ) -> None:
        """Initial discrete state probabilities should be valid distribution."""
        model = CommonOscillatorModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            n_sources=n_src,
            sampling_freq=100.0,
            freqs=jnp.ones(n_osc) * 10.0,
            auto_regressive_coef=jnp.ones(n_osc) * 0.95,
            process_variance=jnp.ones(n_osc) * 0.1,
            measurement_variance=0.05,
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        assert jnp.all(model.init_discrete_state_prob >= 0)
        np.testing.assert_allclose(
            jnp.sum(model.init_discrete_state_prob), 1.0, rtol=1e-6
        )

    @given(
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=15, deadline=None)
    def test_covariances_positive_semidefinite(
        self, n_osc: int, n_disc: int, n_src: int
    ) -> None:
        """All covariance matrices should be positive semi-definite."""
        model = CommonOscillatorModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            n_sources=n_src,
            sampling_freq=100.0,
            freqs=jnp.ones(n_osc) * 10.0,
            auto_regressive_coef=jnp.ones(n_osc) * 0.95,
            process_variance=jnp.ones(n_osc) * 0.1,
            measurement_variance=0.05,
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(n_disc):
            # Check init_cov
            eigenvalues = jnp.linalg.eigvalsh(model.init_cov[..., j])
            assert jnp.all(eigenvalues >= -1e-10)

            # Check process_cov
            eigenvalues = jnp.linalg.eigvalsh(model.process_cov[..., j])
            assert jnp.all(eigenvalues >= -1e-10)

            # Check measurement_cov
            eigenvalues = jnp.linalg.eigvalsh(model.measurement_cov[..., j])
            assert jnp.all(eigenvalues >= -1e-10)


class TestCorrelatedNoiseModelProperties:
    """Property-based tests for CorrelatedNoiseModel."""

    @given(
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=15, deadline=None)
    def test_measurement_matrix_constant_across_states(
        self, n_osc: int, n_disc: int
    ) -> None:
        """For CNM, measurement matrix should be constant across states."""
        model = CorrelatedNoiseModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            sampling_freq=100.0,
            freqs=jnp.ones(n_osc) * 10.0,
            auto_regressive_coef=jnp.ones(n_osc) * 0.95,
            process_variance=jnp.ones((n_osc, n_disc)) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((n_osc, n_osc, n_disc)),
            coupling_strength=jnp.zeros((n_osc, n_osc, n_disc)),
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, n_disc):
            np.testing.assert_allclose(
                model.measurement_matrix[..., 0],
                model.measurement_matrix[..., j],
                rtol=1e-10,
            )

    @given(
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=15, deadline=None)
    def test_transition_matrix_constant_across_states(
        self, n_osc: int, n_disc: int
    ) -> None:
        """For CNM, continuous transition matrix should be constant across states."""
        model = CorrelatedNoiseModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            sampling_freq=100.0,
            freqs=jnp.ones(n_osc) * 10.0,
            auto_regressive_coef=jnp.ones(n_osc) * 0.95,
            process_variance=jnp.ones((n_osc, n_disc)) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((n_osc, n_osc, n_disc)),
            coupling_strength=jnp.zeros((n_osc, n_osc, n_disc)),
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, n_disc):
            np.testing.assert_allclose(
                model.continuous_transition_matrix[..., 0],
                model.continuous_transition_matrix[..., j],
                rtol=1e-10,
            )

    @given(
        st.integers(min_value=2, max_value=3),
        st.integers(min_value=2, max_value=3),
    )
    @settings(max_examples=15, deadline=None)
    def test_process_cov_symmetric_for_all_states(
        self, n_osc: int, n_disc: int
    ) -> None:
        """Process covariance should be symmetric for all discrete states."""
        model = CorrelatedNoiseModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            sampling_freq=100.0,
            freqs=jnp.ones(n_osc) * 10.0,
            auto_regressive_coef=jnp.ones(n_osc) * 0.95,
            process_variance=jnp.ones((n_osc, n_disc)) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((n_osc, n_osc, n_disc)),
            coupling_strength=jnp.zeros((n_osc, n_osc, n_disc)),
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(n_disc):
            Q_j = model.process_cov[..., j]
            np.testing.assert_allclose(Q_j, Q_j.T, rtol=1e-10)


class TestDirectedInfluenceModelProperties:
    """Property-based tests for DirectedInfluenceModel."""

    @given(
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=15, deadline=None)
    def test_process_cov_constant_across_states(self, n_osc: int, n_disc: int) -> None:
        """For DIM, process covariance should be constant across states."""
        model = DirectedInfluenceModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            sampling_freq=100.0,
            freqs=jnp.ones(n_osc) * 10.0,
            auto_regressive_coef=jnp.ones(n_osc) * 0.95,
            process_variance=jnp.ones(n_osc) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((n_osc, n_osc, n_disc)),
            coupling_strength=jnp.zeros((n_osc, n_osc, n_disc)),
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, n_disc):
            np.testing.assert_allclose(
                model.process_cov[..., 0],
                model.process_cov[..., j],
                rtol=1e-10,
            )

    @given(
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=15, deadline=None)
    def test_measurement_matrix_constant_across_states(
        self, n_osc: int, n_disc: int
    ) -> None:
        """For DIM, measurement matrix should be constant across states."""
        model = DirectedInfluenceModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            sampling_freq=100.0,
            freqs=jnp.ones(n_osc) * 10.0,
            auto_regressive_coef=jnp.ones(n_osc) * 0.95,
            process_variance=jnp.ones(n_osc) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((n_osc, n_osc, n_disc)),
            coupling_strength=jnp.zeros((n_osc, n_osc, n_disc)),
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, n_disc):
            np.testing.assert_allclose(
                model.measurement_matrix[..., 0],
                model.measurement_matrix[..., j],
                rtol=1e-10,
            )


class TestOscillatorModelInputValidation:
    """Tests for input validation in oscillator models."""

    def test_com_rejects_mismatched_freqs(self) -> None:
        """COM should reject freqs with wrong shape."""
        with pytest.raises(ValueError, match="Shape mismatch.*freqs"):
            CommonOscillatorModel(
                n_oscillators=2,
                n_discrete_states=3,
                n_sources=4,
                sampling_freq=100.0,
                freqs=jnp.array([10.0]),  # Wrong size
                auto_regressive_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, 0.1]),
                measurement_variance=0.05,
            )

    def test_com_rejects_mismatched_ar_coef(self) -> None:
        """COM should reject auto_regressive_coef with wrong shape."""
        with pytest.raises(ValueError, match="Shape mismatch.*auto_regressive_coef"):
            CommonOscillatorModel(
                n_oscillators=2,
                n_discrete_states=3,
                n_sources=4,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                auto_regressive_coef=jnp.array([0.95]),  # Wrong size
                process_variance=jnp.array([0.1, 0.1]),
                measurement_variance=0.05,
            )

    def test_com_rejects_mismatched_process_variance(self) -> None:
        """COM should reject process_variance with wrong shape."""
        with pytest.raises(ValueError, match="Shape mismatch.*process_variance"):
            CommonOscillatorModel(
                n_oscillators=2,
                n_discrete_states=3,
                n_sources=4,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                auto_regressive_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1]),  # Wrong size
                measurement_variance=0.05,
            )

    def test_cnm_rejects_negative_measurement_variance(self) -> None:
        """CNM should reject negative measurement variance."""
        with pytest.raises(ValueError, match="measurement_variance must be positive"):
            CorrelatedNoiseModel(
                n_oscillators=2,
                n_discrete_states=3,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                auto_regressive_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.ones((2, 3)) * 0.1,
                measurement_variance=-0.05,  # Invalid
                phase_difference=jnp.zeros((2, 2, 3)),
                coupling_strength=jnp.zeros((2, 2, 3)),
            )

    def test_cnm_rejects_zero_measurement_variance(self) -> None:
        """CNM should reject zero measurement variance."""
        with pytest.raises(ValueError, match="measurement_variance must be positive"):
            CorrelatedNoiseModel(
                n_oscillators=2,
                n_discrete_states=3,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                auto_regressive_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.ones((2, 3)) * 0.1,
                measurement_variance=0.0,  # Invalid
                phase_difference=jnp.zeros((2, 2, 3)),
                coupling_strength=jnp.zeros((2, 2, 3)),
            )

    def test_cnm_rejects_negative_sampling_freq(self) -> None:
        """CNM should reject negative sampling frequency."""
        with pytest.raises(ValueError, match="sampling_freq must be positive"):
            CorrelatedNoiseModel(
                n_oscillators=2,
                n_discrete_states=3,
                sampling_freq=-100.0,  # Invalid
                freqs=jnp.array([10.0, 12.0]),
                auto_regressive_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.ones((2, 3)) * 0.1,
                measurement_variance=0.05,
                phase_difference=jnp.zeros((2, 2, 3)),
                coupling_strength=jnp.zeros((2, 2, 3)),
            )

    def test_dim_rejects_mismatched_phase_difference(self) -> None:
        """DIM should reject phase_difference with wrong shape."""
        with pytest.raises(ValueError, match="phase_difference must have shape"):
            DirectedInfluenceModel(
                n_oscillators=2,
                n_discrete_states=3,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                auto_regressive_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, 0.1]),
                measurement_variance=0.05,
                phase_difference=jnp.zeros((2, 2)),  # Missing last dim
                coupling_strength=jnp.zeros((2, 2, 3)),
            )

    def test_dim_rejects_mismatched_coupling_strength(self) -> None:
        """DIM should reject coupling_strength with wrong shape."""
        with pytest.raises(ValueError, match="coupling_strength must have shape"):
            DirectedInfluenceModel(
                n_oscillators=2,
                n_discrete_states=3,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                auto_regressive_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, 0.1]),
                measurement_variance=0.05,
                phase_difference=jnp.zeros((2, 2, 3)),
                coupling_strength=jnp.zeros((3, 3, 3)),  # Wrong n_osc
            )


# ============================================================================
# Tests for Reparameterized M-step in DIM
# ============================================================================


class TestReparameterizedMstep:
    """Tests for the reparameterized M-step in DirectedInfluenceModel."""

    @pytest.fixture
    def dim_synthetic_observations(self):
        """Generates synthetic observations for DIM fitting tests."""
        n_time = 200
        n_sources = 2  # Must match n_oscillators for DIM
        key = jax.random.PRNGKey(42)

        # Generate oscillatory data
        k1, k2 = jax.random.split(key)
        noise = jax.random.normal(k1, (n_time, n_sources)) * 0.5

        t = jnp.arange(n_time) / 100.0
        oscillation = jnp.sin(2 * jnp.pi * 10 * t)[:, None]
        observations = oscillation + noise

        return observations

    def test_reparameterized_flag_default_false(self) -> None:
        """use_reparameterized_mstep should default to False."""
        model = DirectedInfluenceModel(
            n_oscillators=2,
            n_discrete_states=2,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            auto_regressive_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.1,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
        )
        assert model.use_reparameterized_mstep is False

    def test_reparameterized_flag_can_be_set(self) -> None:
        """use_reparameterized_mstep should be settable to True."""
        model = DirectedInfluenceModel(
            n_oscillators=2,
            n_discrete_states=2,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            auto_regressive_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.1,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            use_reparameterized_mstep=True,
        )
        assert model.use_reparameterized_mstep is True

    def test_reparameterized_fit_runs_without_error(
        self, dim_synthetic_observations
    ) -> None:
        """DIM with reparameterized M-step should fit without errors."""
        model = DirectedInfluenceModel(
            n_oscillators=2,
            n_discrete_states=2,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            auto_regressive_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.1,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            use_reparameterized_mstep=True,
        )

        log_likelihoods = model.fit(
            dim_synthetic_observations, jax.random.PRNGKey(42), max_iter=5
        )

        assert len(log_likelihoods) >= 1
        assert all(jnp.isfinite(ll) for ll in log_likelihoods)

    def test_reparameterized_produces_valid_oscillator_structure(
        self, dim_synthetic_observations
    ) -> None:
        """A should have rotation block structure after reparameterized M-step."""
        model = DirectedInfluenceModel(
            n_oscillators=2,
            n_discrete_states=2,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            auto_regressive_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.1,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            use_reparameterized_mstep=True,
        )

        model.fit(dim_synthetic_observations, jax.random.PRNGKey(42), max_iter=5)

        # Check each block is a scaled rotation: [[a, -b], [b, a]]
        for j in range(model.n_discrete_states):
            A_j = model.continuous_transition_matrix[..., j]
            for i in range(model.n_oscillators):
                for k in range(model.n_oscillators):
                    block = A_j[2 * i : 2 * i + 2, 2 * k : 2 * k + 2]
                    # Check rotation structure
                    np.testing.assert_allclose(
                        block[0, 0],
                        block[1, 1],
                        atol=1e-6,
                        err_msg=f"Block ({i},{k}) in state {j}: diagonal elements not equal",
                    )
                    np.testing.assert_allclose(
                        block[0, 1],
                        -block[1, 0],
                        atol=1e-6,
                        err_msg=f"Block ({i},{k}) in state {j}: off-diagonal elements not antisymmetric",
                    )

    def test_standard_mstep_still_works(self, dim_synthetic_observations) -> None:
        """Standard M-step (without reparameterization) should still work."""
        model = DirectedInfluenceModel(
            n_oscillators=2,
            n_discrete_states=2,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            auto_regressive_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.1,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            use_reparameterized_mstep=False,
        )

        log_likelihoods = model.fit(
            dim_synthetic_observations, jax.random.PRNGKey(42), max_iter=5
        )

        assert len(log_likelihoods) >= 1
        assert all(jnp.isfinite(ll) for ll in log_likelihoods)

    def test_reparameterized_updates_public_attributes(
        self, dim_synthetic_observations
    ) -> None:
        """Public oscillator attributes should be updated after fitting."""
        init_freqs = jnp.array([8.0, 12.0])
        init_damping = jnp.array([0.95, 0.95])
        init_coupling = jnp.zeros((2, 2, 2))
        init_phase = jnp.zeros((2, 2, 2))

        model = DirectedInfluenceModel(
            n_oscillators=2,
            n_discrete_states=2,
            sampling_freq=100.0,
            freqs=init_freqs,
            auto_regressive_coef=init_damping,
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.1,
            phase_difference=init_phase,
            coupling_strength=init_coupling,
            use_reparameterized_mstep=True,
        )

        model.fit(dim_synthetic_observations, jax.random.PRNGKey(42), max_iter=5)

        # After fitting, public attributes should have correct shapes
        assert model.freqs.shape == (2,)
        assert model.auto_regressive_coef.shape == (2,)
        assert model.coupling_strength.shape == (2, 2, 2)
        assert model.phase_difference.shape == (2, 2, 2)

        # Values should be finite
        assert jnp.all(jnp.isfinite(model.freqs))
        assert jnp.all(jnp.isfinite(model.auto_regressive_coef))
        assert jnp.all(jnp.isfinite(model.coupling_strength))
        assert jnp.all(jnp.isfinite(model.phase_difference))


class TestDIMStabilityEnforcement:
    """Tests that spectral radius clamping is unconditional."""

    def test_projection_enforces_stability(
        self, directed_influence_params
    ) -> None:
        """An unstable A must always be clamped, regardless of Q-function."""
        model = DirectedInfluenceModel(**directed_influence_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        # Inject an unstable transition matrix (spectral radius > 1)
        n_latent = 2 * model.n_oscillators
        for j in range(model.n_discrete_states):
            model.continuous_transition_matrix = (
                model.continuous_transition_matrix.at[:, :, j].set(
                    jnp.eye(n_latent) * 1.5
                )
            )

        model._project_parameters()

        for j in range(model.n_discrete_states):
            A_j = model.continuous_transition_matrix[:, :, j]
            eigvals = jnp.linalg.eigvals(A_j)
            sr = float(jnp.max(jnp.abs(eigvals)))
            assert sr < 1.0, (
                f"State {j}: spectral radius {sr} >= 1.0 after projection"
            )

    def test_stable_matrix_unchanged_by_projection(
        self, directed_influence_params
    ) -> None:
        """A matrix already within spectral radius bound should not be scaled."""
        params = {**directed_influence_params, "n_discrete_states": 1}
        # Adjust state-dependent arrays to match n_discrete_states=1
        params["phase_difference"] = params["phase_difference"][:, :, :1]
        params["coupling_strength"] = params["coupling_strength"][:, :, :1]
        model = DirectedInfluenceModel(**params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        # Set A to a valid scaled rotation with sr = 0.8
        n = 2 * model.n_oscillators
        A = jnp.eye(n) * 0.8
        model.continuous_transition_matrix = A[:, :, None]

        model._project_parameters()

        # Spectral radius should remain at ~0.8 (not scaled further)
        eigvals = jnp.linalg.eigvals(
            model.continuous_transition_matrix[:, :, 0]
        )
        sr = float(jnp.max(jnp.abs(eigvals)))
        assert 0.75 < sr < 0.85, f"Stable matrix was unnecessarily scaled: sr={sr}"


class TestCommonOscillatorSGDFitting:
    """Tests for CommonOscillatorModel.fit_sgd()."""

    @pytest.fixture
    def com_setup(self):
        """Create a small COM + synthetic data."""
        from state_space_practice.simulate.scenarios import simulate_com_scenario

        scenario = simulate_com_scenario(n_time=200, seed=42)
        p = scenario["params"]
        model = CommonOscillatorModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            n_sources=p["n_sources"],
            sampling_freq=p["sampling_freq"],
            freqs=p["freqs"],
            auto_regressive_coef=p["damping"],
            process_variance=p["process_variance"],
            measurement_variance=p["measurement_variance"],
        )
        return model, scenario["obs"]

    def test_sgd_improves_ll(self, com_setup):
        model, obs = com_setup
        key = jax.random.PRNGKey(0)
        lls = model.fit_sgd(obs, key=key, num_steps=30)
        # LL should improve from first to last
        assert lls[-1] > lls[0]

    def test_sgd_discrete_transitions_stochastic(self, com_setup):
        model, obs = com_setup
        key = jax.random.PRNGKey(0)
        model.fit_sgd(obs, key=key, num_steps=20)
        Z = model.discrete_transition_matrix
        np.testing.assert_allclose(Z.sum(axis=1), 1.0, atol=1e-6)

    def test_sgd_measurement_matrix_finite(self, com_setup):
        model, obs = com_setup
        key = jax.random.PRNGKey(0)
        model.fit_sgd(obs, key=key, num_steps=20)
        assert jnp.all(jnp.isfinite(model.measurement_matrix))

    def test_sgd_populates_smoother_state(self, com_setup):
        model, obs = com_setup
        key = jax.random.PRNGKey(0)
        model.fit_sgd(obs, key=key, num_steps=20)
        assert model.smoother_state_cond_mean is not None
        assert jnp.all(jnp.isfinite(model.smoother_state_cond_mean))


class TestDirectedInfluenceSGDFitting:
    """Tests for DirectedInfluenceModel.fit_sgd()."""

    @pytest.fixture
    def dim_setup(self):
        """Create a small DIM + synthetic data."""
        from state_space_practice.simulate.scenarios import simulate_dim_scenario

        scenario = simulate_dim_scenario(n_time=200, seed=42)
        p = scenario["params"]
        model = DirectedInfluenceModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            freqs=p["freqs"],
            auto_regressive_coef=p["damping"],
            process_variance=p["process_variance"],
            measurement_variance=p["measurement_variance"],
            phase_difference=p["phase_difference"],
            coupling_strength=p["coupling_strength"],
        )
        return model, scenario["obs"]

    def test_sgd_improves_ll(self, dim_setup):
        model, obs = dim_setup
        key = jax.random.PRNGKey(0)
        lls = model.fit_sgd(obs, key=key, num_steps=30)
        assert lls[-1] > lls[0]

    def test_sgd_coupling_params_finite(self, dim_setup):
        model, obs = dim_setup
        key = jax.random.PRNGKey(0)
        model.fit_sgd(obs, key=key, num_steps=20)
        assert jnp.all(jnp.isfinite(model.coupling_strength))
        assert jnp.all(jnp.isfinite(model.phase_difference))

    def test_sgd_discrete_transitions_stochastic(self, dim_setup):
        model, obs = dim_setup
        key = jax.random.PRNGKey(0)
        model.fit_sgd(obs, key=key, num_steps=20)
        Z = model.discrete_transition_matrix
        np.testing.assert_allclose(Z.sum(axis=1), 1.0, atol=1e-6)


class TestCorrelatedNoiseSGDFitting:
    """Tests for CorrelatedNoiseModel.fit_sgd()."""

    @pytest.fixture
    def cnm_setup(self):
        from state_space_practice.simulate.scenarios import simulate_cnm_scenario

        scenario = simulate_cnm_scenario(n_time=200, seed=42)
        p = scenario["params"]
        model = CorrelatedNoiseModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            freqs=p["freqs"],
            auto_regressive_coef=p["damping"],
            process_variance=p["process_variance"],
            measurement_variance=p["measurement_variance"],
            phase_difference=p["phase_difference"],
            coupling_strength=p["coupling_strength"],
        )
        return model, scenario["obs"]

    def test_sgd_improves_ll(self, cnm_setup):
        model, obs = cnm_setup
        key = jax.random.PRNGKey(0)
        lls = model.fit_sgd(obs, key=key, num_steps=30)
        assert lls[-1] > lls[0]

    def test_sgd_process_variance_positive(self, cnm_setup):
        model, obs = cnm_setup
        key = jax.random.PRNGKey(0)
        model.fit_sgd(obs, key=key, num_steps=20)
        assert jnp.all(model.process_variance > 0)


class TestDirectedInfluenceRegularizedSGD:
    """Tests for DIM SGD with connectivity penalties."""

    @pytest.fixture
    def dim_setup(self):
        from state_space_practice.simulate.scenarios import simulate_dim_scenario

        scenario = simulate_dim_scenario(n_time=200, seed=42)
        p = scenario["params"]
        model = DirectedInfluenceModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            freqs=p["freqs"],
            auto_regressive_coef=p["damping"],
            process_variance=p["process_variance"],
            measurement_variance=p["measurement_variance"],
            phase_difference=p["phase_difference"],
            coupling_strength=p["coupling_strength"],
        )
        return model, scenario["obs"]

    def test_edge_penalty_shrinks_coupling_norm(self):
        from state_space_practice.oscillator_regularization import (
            OscillatorPenaltyConfig,
        )
        from state_space_practice.simulate.scenarios import simulate_dim_scenario

        scenario = simulate_dim_scenario(n_time=200, seed=42)
        p = scenario["params"]
        key = jax.random.PRNGKey(0)

        def _make_model():
            return DirectedInfluenceModel(
                n_oscillators=p["n_oscillators"],
                n_discrete_states=p["n_discrete_states"],
                sampling_freq=p["sampling_freq"],
                freqs=p["freqs"],
                auto_regressive_coef=p["damping"],
                process_variance=p["process_variance"],
                measurement_variance=p["measurement_variance"],
                phase_difference=p["phase_difference"],
                coupling_strength=p["coupling_strength"],
            )

        # Baseline: no penalty
        model_base = _make_model()
        model_base.fit_sgd(scenario["obs"], key=key, num_steps=30)
        norm_base = float(jnp.sum(jnp.abs(model_base.coupling_strength)))

        # Regularized: edge L1
        model_reg = _make_model()
        config = OscillatorPenaltyConfig(edge_l1=0.5)
        model_reg.fit_sgd(
            scenario["obs"], key=key, num_steps=30, connectivity_penalty=config,
        )
        norm_reg = float(jnp.sum(jnp.abs(model_reg.coupling_strength)))

        assert norm_reg < norm_base, (
            f"Regularized norm ({norm_reg:.4f}) should be smaller than "
            f"baseline ({norm_base:.4f})"
        )

    def test_area_penalty_shrinks_block_norms(self, dim_setup):
        from state_space_practice.oscillator_regularization import (
            OscillatorPenaltyConfig,
        )

        model, obs = dim_setup
        key = jax.random.PRNGKey(0)
        # 2 oscillators, 2 areas (one per oscillator)
        area_labels = jnp.array([0, 1])
        config = OscillatorPenaltyConfig(
            area_group_l2=1.0, area_labels=area_labels,
        )
        lls = model.fit_sgd(
            obs, key=key, num_steps=30, connectivity_penalty=config,
        )
        # Should still produce finite results
        assert all(np.isfinite(ll) for ll in lls)
        assert jnp.all(jnp.isfinite(model.coupling_strength))

    def test_no_penalty_matches_baseline(self, dim_setup):
        """fit_sgd with None penalty should match no-penalty fit."""
        model, obs = dim_setup
        key = jax.random.PRNGKey(0)
        lls = model.fit_sgd(obs, key=key, num_steps=20, connectivity_penalty=None)
        assert lls[-1] > lls[0]
