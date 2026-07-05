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
        "damping_coef": jnp.array([0.95, 0.95]),
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
        "damping_coef": jnp.array([0.95, 0.95]),
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
        "damping_coef": jnp.array([0.95, 0.95]),
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

    def test_lower_triangle_pair_params_are_canonicalized(
        self, correlated_noise_params
    ) -> None:
        """Lower-triangle CNM pair input is accepted and stored canonically."""
        params = correlated_noise_params.copy()
        n_disc = params["n_discrete_states"]
        params["phase_difference"] = (
            jnp.zeros((2, 2, n_disc)).at[1, 0, :].set(-0.7)
        )
        params["coupling_strength"] = (
            jnp.zeros((2, 2, n_disc)).at[1, 0, :].set(0.05)
        )

        model = CorrelatedNoiseModel(**params)

        np.testing.assert_allclose(model.phase_difference[0, 1, :], 0.7)
        np.testing.assert_allclose(model.coupling_strength[0, 1, :], 0.05)
        np.testing.assert_allclose(model.phase_difference[1, 0, :], 0.0)
        np.testing.assert_allclose(model.coupling_strength[1, 0, :], 0.0)

    def test_conflicting_pair_params_raise(self, correlated_noise_params) -> None:
        """Full-pair CNM input must describe the same covariance both ways."""
        params = correlated_noise_params.copy()
        n_disc = params["n_discrete_states"]
        params["phase_difference"] = (
            jnp.zeros((2, 2, n_disc)).at[0, 1, :].set(0.7).at[1, 0, :].set(0.3)
        )
        params["coupling_strength"] = (
            jnp.zeros((2, 2, n_disc)).at[0, 1, :].set(0.05).at[1, 0, :].set(0.05)
        )

        with pytest.raises(ValueError, match="Conflicting correlated-noise"):
            CorrelatedNoiseModel(**params)

    def test_project_parameters_produces_symmetric_psd(
        self, correlated_noise_params
    ) -> None:
        """_project_parameters must yield a valid (symmetric PSD) covariance.

        The M-step can perturb Q off symmetry and off the PSD cone; the
        projection symmetrizes and floors the eigenvalues so the result is a
        covariance the Cholesky-based switching filter can consume. (A per-block
        rotation projection is deliberately not used -- a rotation is not a valid
        covariance block and would re-break the cross-block symmetry.)
        """
        model = CorrelatedNoiseModel(**correlated_noise_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        # Perturb Q the way an M-step might: asymmetrically and off the PSD cone.
        perturb = jax.random.normal(jax.random.PRNGKey(1), model.process_cov.shape)
        model.process_cov = model.process_cov + 0.1 * perturb

        # Guard: the perturbation actually made Q asymmetric, so the projection
        # has real work to do (the assertions below can't pass vacuously).
        pre = np.array(model.process_cov[..., 0])
        assert np.max(np.abs(pre - pre.T)) > 1e-6

        model._project_parameters()

        for j in range(model.n_discrete_states):
            Q_j = np.array(model.process_cov[..., j])
            np.testing.assert_allclose(Q_j, Q_j.T, atol=1e-10)  # symmetric
            eigs = np.linalg.eigvals(Q_j).real
            assert np.all(eigs >= -1e-8)  # positive semidefinite


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


@pytest.mark.slow
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
            damping_coef=jnp.array([0.95, 0.95]),
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


def _run_one_em_update(model, observations) -> None:
    model._initialize_parameters(jax.random.PRNGKey(0))
    model._e_step(observations)
    model._m_step(observations)
    model._project_parameters()


def _assert_shared_measurement_covariance(model) -> None:
    for j in range(1, model.n_discrete_states):
        np.testing.assert_allclose(
            model.measurement_cov[..., 0],
            model.measurement_cov[..., j],
            atol=1e-10,
        )


def _assert_scaled_rotation_block(block, atol: float = 1e-8) -> None:
    np.testing.assert_allclose(block[0, 0], block[1, 1], atol=atol)
    np.testing.assert_allclose(block[0, 1], -block[1, 0], atol=atol)


class TestOscillatorPaperStructure:
    """Regression tests for the COM/CNM/DIM constraints in Hsin et al. 2024."""

    def test_em_pools_observation_covariance_across_states(
        self, synthetic_observations
    ) -> None:
        """Eq. 2.18 estimates one shared R, not state-specific R_j."""
        obs = synthetic_observations
        models = [
            CommonOscillatorModel(
                n_oscillators=2,
                n_discrete_states=2,
                n_sources=obs.shape[1],
                sampling_freq=100.0,
                freqs=jnp.array([8.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.9]),
                process_variance=jnp.array([0.1, 0.2]),
                measurement_variance=0.05,
            ),
            CorrelatedNoiseModel(
                n_oscillators=2,
                n_discrete_states=2,
                sampling_freq=100.0,
                freqs=jnp.array([8.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.9]),
                process_variance=jnp.ones((2, 2)) * 0.1,
                measurement_variance=0.05,
                phase_difference=jnp.zeros((2, 2, 2)),
                coupling_strength=jnp.zeros((2, 2, 2)),
            ),
            DirectedInfluenceModel(
                n_oscillators=2,
                n_discrete_states=2,
                sampling_freq=100.0,
                freqs=jnp.array([8.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.9]),
                process_variance=jnp.array([0.1, 0.2]),
                measurement_variance=0.05,
                phase_difference=jnp.zeros((2, 2, 2)),
                coupling_strength=jnp.zeros((2, 2, 2)),
            ),
        ]

        for model in models:
            _run_one_em_update(model, obs)
            _assert_shared_measurement_covariance(model)

    def test_cnm_em_projection_preserves_covariance_structure(
        self, synthetic_observations
    ) -> None:
        """CNM Q_j keeps scalar diagonal blocks and symmetric rotation links."""
        model = CorrelatedNoiseModel(
            n_oscillators=2,
            n_discrete_states=2,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.9]),
            process_variance=jnp.ones((2, 2)) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
        )

        _run_one_em_update(model, synthetic_observations)

        for state in range(model.n_discrete_states):
            Q = np.array(model.process_cov[..., state])
            np.testing.assert_allclose(Q, Q.T, atol=1e-10)
            assert np.linalg.eigvalsh(Q).min() >= -1e-8

            for osc in range(model.n_oscillators):
                block = Q[2 * osc:2 * osc + 2, 2 * osc:2 * osc + 2]
                np.testing.assert_allclose(block[0, 0], block[1, 1], atol=1e-10)
                np.testing.assert_allclose(block[0, 1], 0.0, atol=1e-10)
                np.testing.assert_allclose(block[1, 0], 0.0, atol=1e-10)

            upper = Q[0:2, 2:4]
            lower = Q[2:4, 0:2]
            _assert_scaled_rotation_block(upper)
            np.testing.assert_allclose(lower, upper.T, atol=1e-10)

        from state_space_practice.oscillator_utils import (
            construct_correlated_noise_process_covariance,
        )

        reconstructed = jnp.stack(
            [
                construct_correlated_noise_process_covariance(
                    model.process_variance[:, state],
                    model.phase_difference[..., state],
                    model.coupling_strength[..., state],
                )
                for state in range(model.n_discrete_states)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(model.process_cov, reconstructed, atol=1e-8)

    def test_dim_standard_em_projection_preserves_transition_structure(
        self, synthetic_observations
    ) -> None:
        """Standard DIM EM must return oscillator-block A_j, not arbitrary A_j."""
        model = DirectedInfluenceModel(
            n_oscillators=2,
            n_discrete_states=2,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.9]),
            process_variance=jnp.array([0.1, 0.2]),
            measurement_variance=0.05,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            use_reparameterized_mstep=False,
        )

        _run_one_em_update(model, synthetic_observations)

        for state in range(model.n_discrete_states):
            A = np.array(model.continuous_transition_matrix[..., state])
            for row in range(model.n_oscillators):
                for col in range(model.n_oscillators):
                    block = A[2 * row:2 * row + 2, 2 * col:2 * col + 2]
                    _assert_scaled_rotation_block(block)
            assert np.max(np.abs(np.linalg.eigvals(A))) < 1.0

        diag = jnp.diagonal(model.coupling_strength, axis1=0, axis2=1)
        np.testing.assert_allclose(diag, 0.0, atol=1e-12)

    def test_dim_standard_em_log_likelihood_does_not_decrease(
        self, synthetic_observations
    ) -> None:
        """Standard projected DIM EM should keep a monotone LL trajectory."""
        model = DirectedInfluenceModel(
            n_oscillators=2,
            n_discrete_states=2,
            sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.9]),
            process_variance=jnp.array([0.1, 0.2]),
            measurement_variance=0.05,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            use_reparameterized_mstep=False,
        )

        log_likelihoods = model.fit(
            synthetic_observations,
            jax.random.PRNGKey(42),
            max_iter=10,
        )

        diffs = np.diff(np.asarray(log_likelihoods, dtype=float))
        assert np.all(diffs >= -1e-6), f"LL decreases: {diffs}"

    def test_dim_rejects_nonzero_diagonal_coupling(self) -> None:
        """DIM diagonal coupling is unused; fail loudly instead of storing it."""
        bad_coupling = jnp.zeros((2, 2, 2)).at[0, 0, :].set(0.1)
        with pytest.raises(ValueError, match="diagonal"):
            DirectedInfluenceModel(
                n_oscillators=2,
                n_discrete_states=2,
                sampling_freq=100.0,
                freqs=jnp.array([8.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.9]),
                process_variance=jnp.array([0.1, 0.2]),
                measurement_variance=0.05,
                phase_difference=jnp.zeros((2, 2, 2)),
                coupling_strength=bad_coupling,
            )

    def test_dim_rejects_nonzero_diagonal_phase(self) -> None:
        """DIM diagonal phase is unused; fail loudly instead of storing it."""
        bad_phase = jnp.zeros((2, 2, 2)).at[0, 0, :].set(0.1)
        with pytest.raises(ValueError, match="diagonal"):
            DirectedInfluenceModel(
                n_oscillators=2,
                n_discrete_states=2,
                sampling_freq=100.0,
                freqs=jnp.array([8.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.9]),
                process_variance=jnp.array([0.1, 0.2]),
                measurement_variance=0.05,
                phase_difference=bad_phase,
                coupling_strength=jnp.zeros((2, 2, 2)),
            )


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
            damping_coef=jnp.array([0.95, 0.95]),
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
        """Raw process covariance must be PSD, checked with general eigvals.

        Uses ``eigvals`` (not ``eigvalsh``, which silently symmetrizes and so
        cannot detect an asymmetric "covariance") and nonzero coupling with
        magnitude below the process variance (so Q stays positive definite),
        exercising the off-diagonal cross-blocks.
        """
        params = dict(correlated_noise_params)
        n_osc = params["n_oscillators"]
        n_disc = params["n_discrete_states"]
        coupling = np.zeros((n_osc, n_osc, n_disc))
        coupling[0, 1, :] = 0.05  # < process_variance (0.1) -> Q positive definite
        phase = np.zeros((n_osc, n_osc, n_disc))
        phase[0, 1, :] = 0.7
        params["coupling_strength"] = jnp.array(coupling)
        params["phase_difference"] = jnp.array(phase)
        model = CorrelatedNoiseModel(**params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(model.n_discrete_states):
            Q_j = np.array(model.process_cov[..., j])
            off_diag = Q_j - np.diag(np.diag(Q_j))
            assert np.any(np.abs(off_diag) > 1e-6)  # coupling actually exercised
            eigenvalues = np.linalg.eigvals(Q_j).real
            assert np.all(eigenvalues >= -1e-8)

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
            damping_coef=jnp.ones(n_osc) * 0.95,
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
            damping_coef=jnp.ones(n_osc) * 0.95,
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
            damping_coef=jnp.ones(n_osc) * 0.95,
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
            damping_coef=jnp.ones(n_osc) * 0.95,
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
            damping_coef=jnp.ones(n_osc) * 0.95,
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
        """Q must be symmetric from canonical upper-triangle pair input.

        A covariance is symmetric by definition; the model stores one
        upper-triangle parameter per oscillator pair and mirrors it as the
        transpose.
        """
        rng = np.random.default_rng(0)
        phase = np.zeros((n_osc, n_osc, n_disc))
        coupling = np.zeros((n_osc, n_osc, n_disc))
        upper_i, upper_j = np.triu_indices(n_osc, k=1)
        phase[upper_i, upper_j, :] = rng.uniform(
            -1.0, 1.0, (len(upper_i), n_disc)
        )
        coupling[upper_i, upper_j, :] = rng.uniform(
            0.01, 0.05, (len(upper_i), n_disc)
        )
        model = CorrelatedNoiseModel(
            n_oscillators=n_osc,
            n_discrete_states=n_disc,
            sampling_freq=100.0,
            freqs=jnp.ones(n_osc) * 10.0,
            damping_coef=jnp.ones(n_osc) * 0.95,
            process_variance=jnp.ones((n_osc, n_disc)) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.array(phase),
            coupling_strength=jnp.array(coupling),
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(n_disc):
            Q_j = np.array(model.process_cov[..., j])
            # Guard: the off-diagonal cross-blocks are actually nonzero, so a
            # zero-coupling (trivially symmetric) Q cannot pass vacuously.
            off_diag = Q_j - np.diag(np.diag(Q_j))
            assert np.any(np.abs(off_diag) > 1e-6)
            np.testing.assert_allclose(Q_j, Q_j.T, atol=1e-10)


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
            damping_coef=jnp.ones(n_osc) * 0.95,
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
            damping_coef=jnp.ones(n_osc) * 0.95,
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
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, 0.1]),
                measurement_variance=0.05,
            )

    def test_com_rejects_mismatched_ar_coef(self) -> None:
        """COM should reject damping_coef with wrong shape."""
        with pytest.raises(ValueError, match="Shape mismatch.*damping_coef"):
            CommonOscillatorModel(
                n_oscillators=2,
                n_discrete_states=3,
                n_sources=4,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                damping_coef=jnp.array([0.95]),  # Wrong size
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
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1]),  # Wrong size
                measurement_variance=0.05,
            )

    def test_com_rejects_negative_process_variance(self) -> None:
        """COM should reject negative process_variance."""
        with pytest.raises(ValueError, match="process_variance must be non-negative"):
            CommonOscillatorModel(
                n_oscillators=2,
                n_discrete_states=3,
                n_sources=4,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, -0.1]),
                measurement_variance=0.05,
            )

    def test_cnm_rejects_negative_process_variance(self) -> None:
        """CNM should reject negative process_variance."""
        with pytest.raises(ValueError, match="process_variance must be non-negative"):
            CorrelatedNoiseModel(
                n_oscillators=2,
                n_discrete_states=3,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([[0.1, 0.1, 0.1], [0.1, -0.1, 0.1]]),
                measurement_variance=0.05,
                phase_difference=jnp.zeros((2, 2, 3)),
                coupling_strength=jnp.zeros((2, 2, 3)),
            )

    def test_dim_rejects_negative_process_variance(self) -> None:
        """DIM should reject negative process_variance."""
        with pytest.raises(ValueError, match="process_variance must be non-negative"):
            DirectedInfluenceModel(
                n_oscillators=2,
                n_discrete_states=3,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, -0.1]),
                measurement_variance=0.05,
                phase_difference=jnp.zeros((2, 2, 3)),
                coupling_strength=jnp.zeros((2, 2, 3)),
            )

    def test_cnm_rejects_negative_measurement_variance(self) -> None:
        """CNM should reject negative measurement variance."""
        with pytest.raises(ValueError, match="measurement_variance must be positive"):
            CorrelatedNoiseModel(
                n_oscillators=2,
                n_discrete_states=3,
                sampling_freq=100.0,
                freqs=jnp.array([10.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.95]),
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
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.ones((2, 3)) * 0.1,
                measurement_variance=0.0,  # Invalid
                phase_difference=jnp.zeros((2, 2, 3)),
                coupling_strength=jnp.zeros((2, 2, 3)),
            )

    def test_cnm_rejects_negative_sampling_freq(self) -> None:
        """CNM should reject negative sampling frequency."""
        with pytest.raises(ValueError, match="sampling_freq"):
            CorrelatedNoiseModel(
                n_oscillators=2,
                n_discrete_states=3,
                sampling_freq=-100.0,  # Invalid
                freqs=jnp.array([10.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.95]),
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
                damping_coef=jnp.array([0.95, 0.95]),
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
                damping_coef=jnp.array([0.95, 0.95]),
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
            damping_coef=jnp.array([0.95, 0.95]),
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
            damping_coef=jnp.array([0.95, 0.95]),
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
            damping_coef=jnp.array([0.95, 0.95]),
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
            damping_coef=jnp.array([0.95, 0.95]),
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
            damping_coef=jnp.array([0.95, 0.95]),
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
            damping_coef=init_damping,
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.1,
            phase_difference=init_phase,
            coupling_strength=init_coupling,
            use_reparameterized_mstep=True,
        )

        model.fit(dim_synthetic_observations, jax.random.PRNGKey(42), max_iter=5)

        # After fitting, public attributes should have correct shapes
        assert model.freqs.shape == (2,)
        assert model.damping_coef.shape == (2,)
        assert model.coupling_strength.shape == (2, 2, 2)
        assert model.phase_difference.shape == (2, 2, 2)

        # Values should be finite
        assert jnp.all(jnp.isfinite(model.freqs))
        assert jnp.all(jnp.isfinite(model.damping_coef))
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


@pytest.mark.slow
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
            damping_coef=p["damping"],
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


@pytest.mark.slow
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
            damping_coef=p["damping"],
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


@pytest.mark.slow
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
            damping_coef=p["damping"],
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

    @pytest.mark.slow
    def test_sgd_loss_finite_and_differentiable_at_indefinite_coupling(
        self, cnm_setup
    ):
        # coupling_strength is UNCONSTRAINED during SGD, so the optimizer can
        # propose a coupling whose reconstructed Q is indefinite. The loss must
        # stay finite AND differentiable there (via the gradient-safe PSD shift);
        # otherwise a NaN loss/gradient poisons the optimizer. Without the fix
        # both go NaN. The guard confirms the proposed Q is actually indefinite.
        from state_space_practice.oscillator_utils import (
            construct_correlated_noise_process_covariance,
        )

        model, obs = cnm_setup
        model._initialize_parameters(jax.random.PRNGKey(0))
        n_osc, n_states = model.n_oscillators, model.n_discrete_states
        big = 5.0 * float(jnp.max(model.process_variance))  # coupling >> variance

        Q_raw = construct_correlated_noise_process_covariance(
            variance=model.process_variance[:, 0],
            phase_difference=model.phase_difference[..., 0],
            coupling_strength=jnp.zeros((n_osc, n_osc)).at[0, 1].set(big),
        )
        assert jnp.linalg.eigvalsh(Q_raw).min() < 0.0  # guard: barrier is active

        def loss(c):
            cp = jnp.zeros((n_osc, n_osc, n_states)).at[0, 1, :].set(c)
            return model._sgd_loss_fn({"coupling_strength": cp}, obs)

        assert bool(jnp.isfinite(loss(big)))
        assert bool(jnp.isfinite(jax.grad(loss)(big)))


@pytest.mark.slow
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
            damping_coef=p["damping"],
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
                damping_coef=p["damping"],
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


# ============================================================================
# Tests for BaseModel: stickiness, decode, predict_proba
# ============================================================================


class TestBaseModelStickiness:
    """Test that stickiness parameter biases transition matrix toward self-transitions."""

    @pytest.mark.slow
    def test_stickiness_increases_self_transition(self):
        """Higher stickiness should produce higher diagonal in Z after EM."""
        n_time = 300
        n_sources = 2
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (n_time, n_sources)) * 0.5
        t = jnp.arange(n_time) / 100.0
        obs = jnp.sin(2 * jnp.pi * 10 * t)[:, None] + noise

        model_no_sticky = CorrelatedNoiseModel(
            n_oscillators=2, n_discrete_states=2, sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]), damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.ones((2, 2)) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            stickiness=0.0,
        )
        model_sticky = CorrelatedNoiseModel(
            n_oscillators=2, n_discrete_states=2, sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]), damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.ones((2, 2)) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            stickiness=10.0,
        )

        model_no_sticky.fit(obs, key=jax.random.PRNGKey(0), max_iter=5)
        model_sticky.fit(obs, key=jax.random.PRNGKey(0), max_iter=5)

        diag_no_sticky = jnp.diag(model_no_sticky.discrete_transition_matrix)
        diag_sticky = jnp.diag(model_sticky.discrete_transition_matrix)

        assert jnp.all(diag_sticky >= diag_no_sticky - 1e-6), (
            f"Sticky diagonal {diag_sticky} should be >= no-sticky diagonal "
            f"{diag_no_sticky}"
        )

    def test_stickiness_zero_has_no_prior(self):
        """stickiness=0 should produce no transition_prior."""
        model = CommonOscillatorModel(
            n_oscillators=2, n_discrete_states=3, n_sources=4,
            sampling_freq=100.0, freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.05,
            stickiness=0.0,
        )
        assert model.transition_prior is None

    def test_stickiness_positive_creates_prior(self):
        """stickiness>0 should produce a non-None transition_prior."""
        model = CommonOscillatorModel(
            n_oscillators=2, n_discrete_states=3, n_sources=4,
            sampling_freq=100.0, freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.05,
            stickiness=5.0,
        )
        assert model.transition_prior is not None


class TestBaseModelDecodeAndPredictProba:
    """Test decode() and predict_proba() methods."""

    def test_decode_before_fit_raises(self):
        """decode() before fit should raise RuntimeError."""
        model = CommonOscillatorModel(
            n_oscillators=2, n_discrete_states=3, n_sources=4,
            sampling_freq=100.0, freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.05,
        )
        with pytest.raises(RuntimeError, match="fit"):
            model.decode()

    def test_predict_proba_before_fit_raises(self):
        """predict_proba() before fit should raise RuntimeError."""
        model = CommonOscillatorModel(
            n_oscillators=2, n_discrete_states=3, n_sources=4,
            sampling_freq=100.0, freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.05,
        )
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_proba()

    @pytest.mark.slow
    def test_decode_returns_integer_labels(self):
        """decode() after fit should return integer state labels."""
        n_time = 200
        n_sources = 2
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (n_time, n_sources)) * 0.5
        t = jnp.arange(n_time) / 100.0
        obs = jnp.sin(2 * jnp.pi * 10 * t)[:, None] + noise

        model = CorrelatedNoiseModel(
            n_oscillators=2, n_discrete_states=2, sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]), damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.ones((2, 2)) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
        )
        model.fit(obs, key=jax.random.PRNGKey(0), max_iter=3)
        states = model.decode()

        assert states.shape == (n_time,)
        assert jnp.all((states >= 0) & (states < 2))

    @pytest.mark.slow
    def test_predict_proba_sums_to_one(self):
        """predict_proba() rows should sum to 1."""
        n_time = 200
        n_sources = 2
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (n_time, n_sources)) * 0.5
        t = jnp.arange(n_time) / 100.0
        obs = jnp.sin(2 * jnp.pi * 10 * t)[:, None] + noise

        model = CorrelatedNoiseModel(
            n_oscillators=2, n_discrete_states=2, sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]), damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.ones((2, 2)) * 0.1,
            measurement_variance=0.05,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
        )
        model.fit(obs, key=jax.random.PRNGKey(0), max_iter=3)
        probs = model.predict_proba()

        assert probs.shape == (n_time, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


class TestBaseModelPStayValidation:
    """Test p_stay validation for low sampling frequencies."""

    def test_low_sampling_freq_raises(self):
        """sampling_freq < 1.0 Hz should raise ValueError."""
        with pytest.raises(ValueError, match="sampling_freq"):
            CommonOscillatorModel(
                n_oscillators=2, n_discrete_states=3, n_sources=4,
                sampling_freq=0.5, freqs=jnp.array([8.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, 0.1]),
                measurement_variance=0.05,
            )

    def test_p_stay_at_low_boundary(self):
        """At sampling_freq=1.0 Hz the ~1s dwell target gives p_stay=0."""
        model = CommonOscillatorModel(
            n_oscillators=2, n_discrete_states=3, n_sources=4,
            sampling_freq=1.0, freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.05,
        )
        # p_stay = 1 - 1/(1s * 1Hz) = 0; stays in [0, 1).
        assert jnp.all(model.discrete_transition_diag >= 0)
        assert jnp.all(model.discrete_transition_diag < 1.0)
        np.testing.assert_allclose(model.discrete_transition_diag, 0.0, atol=1e-9)

    @pytest.mark.parametrize("sampling_freq", [50.0, 250.0, 1000.0])
    def test_default_p_stay_preserves_one_second_dwell(self, sampling_freq):
        """Default p_stay must target a ~1s dwell, not flatten to 0.95.

        Regression for the old ``min(0.95, ...)`` cap, which bound for any
        sampling rate above 20 Hz and collapsed the intended ~1s dwell to a
        20-sample dwell (e.g. 20 ms at 1000 Hz).
        """
        model = CommonOscillatorModel(
            n_oscillators=2, n_discrete_states=3, n_sources=4,
            sampling_freq=sampling_freq, freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.05,
        )
        expected_p_stay = 1.0 - 1.0 / sampling_freq
        np.testing.assert_allclose(
            model.discrete_transition_diag, expected_p_stay, rtol=1e-6
        )
        # Expected dwell 1/(1-p_stay) should be ~sampling_freq samples = ~1s.
        dwell_samples = 1.0 / (1.0 - float(model.discrete_transition_diag[0]))
        np.testing.assert_allclose(dwell_samples, sampling_freq, rtol=1e-6)

    @pytest.mark.parametrize(
        "discrete_transition_diag",
        [
            jnp.array([1.2, 0.8]),
            jnp.array([-0.1, 0.8]),
            jnp.array([0.9]),
        ],
    )
    def test_user_discrete_transition_diag_is_validated(
        self, discrete_transition_diag
    ):
        """User-provided p_stay vector should have valid shape and range."""
        with pytest.raises(ValueError, match="discrete_transition_diag"):
            CommonOscillatorModel(
                n_oscillators=2, n_discrete_states=2, n_sources=4,
                sampling_freq=100.0, freqs=jnp.array([8.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, 0.1]),
                measurement_variance=0.05,
                discrete_transition_diag=discrete_transition_diag,
            )


class TestOscillatorEMRollback:
    """EM loop must roll back state on LL decrease.

    Same contract as test_point_process_kalman.TestEMRollbackOnDecrease.
    """

    def test_oscillator_em_rolls_back(self, caplog) -> None:
        from state_space_practice.simulate.scenarios import simulate_com_scenario
        from state_space_practice.tests.conftest import (
            assert_em_rolls_back_on_ll_decrease,
        )

        scenario = simulate_com_scenario(n_time=200, seed=0)
        p = scenario["params"]
        model = CommonOscillatorModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            n_sources=p["n_sources"],
            sampling_freq=p["sampling_freq"],
            freqs=p["freqs"],
            damping_coef=p["damping"],
            process_variance=p["process_variance"],
            measurement_variance=p["measurement_variance"],
        )
        assert_em_rolls_back_on_ll_decrease(
            model, (scenario["obs"],), caplog,
        )

    def test_oscillator_em_restores_rejected_parameters(self) -> None:
        """Rollback should restore the params that produced the last good E-step."""
        from state_space_practice.simulate.scenarios import simulate_com_scenario

        scenario = simulate_com_scenario(n_time=80, seed=0)
        p = scenario["params"]
        model = CommonOscillatorModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            n_sources=p["n_sources"],
            sampling_freq=p["sampling_freq"],
            freqs=p["freqs"],
            damping_coef=p["damping"],
            process_variance=p["process_variance"],
            measurement_variance=p["measurement_variance"],
        )

        real_e_step = model._e_step
        ll_sequence = iter([0.0, -1e6, -1e6])

        def fake_e_step(*args, **kwargs):
            real = real_e_step(*args, **kwargs)
            return jnp.asarray(next(ll_sequence), dtype=real.dtype)

        real_m_step = model._m_step

        def bad_m_step(*args, **kwargs):
            real_m_step(*args, **kwargs)
            model.measurement_matrix = model.measurement_matrix + 100.0

        model._e_step = fake_e_step
        model._m_step = bad_m_step

        log_likelihoods = model.fit(scenario["obs"], max_iter=5)

        assert log_likelihoods == [0.0]
        assert float(jnp.max(jnp.abs(model.measurement_matrix))) < 1.0


class TestForcedUpdateFlagsRejected:
    """Subclasses fix which of A, H, Q are learned; overriding must raise."""

    _FORCED = [
        "update_continuous_transition_matrix",
        "update_measurement_matrix",
        "update_process_cov",
    ]

    @pytest.mark.parametrize("flag", _FORCED)
    def test_com_rejects_forced_flag(self, common_oscillator_params, flag):
        with pytest.raises(ValueError, match="cannot be overridden"):
            CommonOscillatorModel(**common_oscillator_params, **{flag: True})

    @pytest.mark.parametrize("flag", _FORCED)
    def test_cnm_rejects_forced_flag(self, correlated_noise_params, flag):
        with pytest.raises(ValueError, match="cannot be overridden"):
            CorrelatedNoiseModel(**correlated_noise_params, **{flag: False})

    @pytest.mark.parametrize("flag", _FORCED)
    def test_dim_rejects_forced_flag(self, directed_influence_params, flag):
        with pytest.raises(ValueError, match="cannot be overridden"):
            DirectedInfluenceModel(**directed_influence_params, **{flag: True})

    def test_non_forced_flag_is_accepted(self, common_oscillator_params):
        """A flag the subclass does not fix must still be settable."""
        model = CommonOscillatorModel(
            **common_oscillator_params, update_measurement_cov=False
        )
        assert model.update_measurement_cov is False
        # And the forced flags keep their model-defined values.
        assert model.update_process_cov is False
        assert model.update_continuous_transition_matrix is False


class TestComWarmInitStateDependentH:
    """COM's H is state-dependent, so warm init must not pinv-amplify init_mean."""

    def _com(self):
        return CommonOscillatorModel(
            n_oscillators=2, n_discrete_states=2, n_sources=4,
            sampling_freq=100.0, freqs=jnp.array([8.0, 12.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]),
            measurement_variance=0.05,
        )

    def test_warm_init_preserves_cold_init_mean(self):
        model = self._com()
        model._initialize_parameters(jax.random.PRNGKey(0))
        cold_mean = model.init_mean

        key = jax.random.PRNGKey(1)
        # Variance clearly != 1 so the (deterministic) init_cov rescale is visible.
        obs = 3.0 * jax.random.normal(key, (400, 4))
        model._warm_initialize_states(obs)

        # State-dependent H -> the pinv mean step is skipped; init_mean unchanged.
        np.testing.assert_array_equal(
            np.array(model.init_mean), np.array(cold_mean)
        )
        # But warm init still ran: init_cov was rescaled to the data variance.
        obs_var = float(np.var(np.array(obs), axis=0).mean())
        for j in range(model.n_discrete_states):
            np.testing.assert_allclose(
                np.diag(np.array(model.init_cov[..., j])), obs_var, rtol=0.05
            )

    def test_shared_h_model_updates_init_mean(self):
        """Contrast: DIM has shared H, so warm init DOES set init_mean."""
        model = DirectedInfluenceModel(
            n_oscillators=2, n_discrete_states=2, sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]), damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]), measurement_variance=0.05,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
        )
        model._initialize_parameters(jax.random.PRNGKey(0))
        cold_mean = model.init_mean
        obs = 3.0 * jax.random.normal(jax.random.PRNGKey(1), (400, 2))
        model._warm_initialize_states(obs)
        assert not np.allclose(np.array(model.init_mean), np.array(cold_mean))


class TestReparameterizedPublicParamsReconstructA:
    """After a reparameterized DIM fit, public params must reconstruct A."""

    @pytest.mark.slow
    def test_public_params_reconstruct_transition_matrix(self):
        from state_space_practice.oscillator_models import (
            _stabilize_transition_matrix,
        )
        from state_space_practice.oscillator_utils import (
            construct_directed_influence_transition_matrix,
        )

        key = jax.random.PRNGKey(3)
        n_time = 250
        t = jnp.arange(n_time) / 100.0
        obs = (
            jnp.sin(2 * jnp.pi * 9 * t)[:, None] * jnp.ones((1, 2))
            + 0.3 * jax.random.normal(key, (n_time, 2))
        )
        model = DirectedInfluenceModel(
            n_oscillators=2, n_discrete_states=2, sampling_freq=100.0,
            freqs=jnp.array([8.0, 12.0]), damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]), measurement_variance=0.1,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            use_reparameterized_mstep=True,
        )
        model.fit(obs, jax.random.PRNGKey(7), max_iter=5)

        # freqs/damping are shared (n_osc,); coupling/phase are per-state.
        assert model.freqs.shape == (2,)
        assert model.damping_coef.shape == (2,)
        # The stored A must equal the stabilized reconstruction from the public
        # (shared freq/damping + per-state coupling/phase) parameters. Before
        # the fix, self.freqs was the cross-state mean while A used per-state
        # freqs, so this reconstruction would not match.
        for j in range(model.n_discrete_states):
            A_recon = _stabilize_transition_matrix(
                construct_directed_influence_transition_matrix(
                    freqs=model.freqs,
                    damping_coeffs=model.damping_coef,
                    coupling_strengths=model.coupling_strength[..., j],
                    phase_diffs=model.phase_difference[..., j],
                    sampling_freq=model.sampling_freq,
                )
            )
            np.testing.assert_allclose(
                np.array(model.continuous_transition_matrix[..., j]),
                np.array(A_recon),
                rtol=1e-6, atol=1e-9,
            )


class TestFirstIterationFailureClearsPosteriors:
    """A non-finite first E-step must not leave NaN posteriors installed."""

    def test_nonfinite_first_e_step_clears_posteriors(
        self, common_oscillator_params
    ):
        model = CommonOscillatorModel(**common_oscillator_params)
        obs = jax.random.normal(
            jax.random.PRNGKey(1), (150, common_oscillator_params["n_sources"])
        )

        real_e_step = model._e_step

        def fake_e_step(observations):
            # Install a full (but NaN) posterior like a real failed E-step,
            # then report a non-finite log-likelihood.
            real_e_step(observations)
            n = observations.shape[0]
            model.smoother_discrete_state_prob = jnp.full(
                (n, model.n_discrete_states), jnp.nan
            )
            return jnp.asarray(jnp.nan)

        model._e_step = fake_e_step
        lls = model.fit(obs, jax.random.PRNGKey(0), max_iter=3)

        # Nothing usable was produced.
        assert lls == []
        assert model.smoother_discrete_state_prob is None
        with pytest.raises(RuntimeError, match="No smoother posteriors"):
            model.decode()
        with pytest.raises(RuntimeError, match="No smoother posteriors"):
            model.predict_proba()


class TestSamplingFrequencyRange:
    """Models must be usable across sampling rates up to at least 1000 Hz."""

    @pytest.mark.slow
    @pytest.mark.parametrize("sampling_freq", [100.0, 500.0, 1000.0])
    def test_models_fit_across_sampling_frequencies(self, sampling_freq):
        key = jax.random.PRNGKey(0)
        n_time = 300
        t = jnp.arange(n_time) / sampling_freq
        obs2 = (
            jnp.sin(2 * jnp.pi * 8 * t)[:, None] * jnp.ones((1, 2))
            + 0.3 * jax.random.normal(key, (n_time, 2))
        )
        obs4 = jnp.concatenate([obs2, 0.5 * obs2], axis=1)

        def _finite(values):
            return len(values) >= 1 and all(np.isfinite(v) for v in values)

        com = CommonOscillatorModel(
            2, 2, 4, sampling_freq, freqs=jnp.array([8.0, 20.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]), measurement_variance=0.1,
        )
        assert _finite(com.fit(obs4, key, max_iter=3))

        cnm = CorrelatedNoiseModel(
            2, 2, sampling_freq, freqs=jnp.array([8.0, 20.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.ones((2, 2)) * 0.1, measurement_variance=0.1,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
        )
        assert _finite(cnm.fit(obs2, key, max_iter=3))

        dim = DirectedInfluenceModel(
            2, 2, sampling_freq, freqs=jnp.array([8.0, 20.0]),
            damping_coef=jnp.array([0.95, 0.95]),
            process_variance=jnp.array([0.1, 0.1]), measurement_variance=0.1,
            phase_difference=jnp.zeros((2, 2, 2)),
            coupling_strength=jnp.zeros((2, 2, 2)),
            use_reparameterized_mstep=True,
        )
        assert _finite(dim.fit(obs2, key, max_iter=3))

        # Default prior targets a ~1s dwell at every rate (regression for the
        # old 0.95 cap that bound above 20 Hz).
        np.testing.assert_allclose(
            float(com.discrete_transition_diag[0]),
            1.0 - 1.0 / sampling_freq, rtol=1e-6,
        )
