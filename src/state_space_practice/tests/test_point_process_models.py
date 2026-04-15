"""Tests for the point_process_models module.

This module tests the switching point-process oscillator model classes
(COM-PP, CNM-PP, DIM-PP) for spike-based dynamic functional connectivity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.point_process_models import (
    CommonOscillatorPointProcessModel,
    CorrelatedNoisePointProcessModel,
    DirectedInfluencePointProcessModel,
)

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def com_pp_params():
    """Parameters for CommonOscillatorPointProcessModel tests."""
    n_oscillators = 2
    n_neurons = 5
    n_discrete_states = 3
    sampling_freq = 100.0
    dt = 0.01

    return {
        "n_oscillators": n_oscillators,
        "n_neurons": n_neurons,
        "n_discrete_states": n_discrete_states,
        "sampling_freq": sampling_freq,
        "dt": dt,
        "freqs": jnp.array([8.0, 12.0]),
        "damping_coef": jnp.array([0.95, 0.95]),
        "process_variance": jnp.array([0.1, 0.1]),
    }


@pytest.fixture(scope="module")
def cnm_pp_params():
    """Parameters for CorrelatedNoisePointProcessModel tests."""
    n_oscillators = 2
    n_neurons = 5
    n_discrete_states = 3
    sampling_freq = 100.0
    dt = 0.01

    return {
        "n_oscillators": n_oscillators,
        "n_neurons": n_neurons,
        "n_discrete_states": n_discrete_states,
        "sampling_freq": sampling_freq,
        "dt": dt,
        "freqs": jnp.array([8.0, 12.0]),
        "damping_coef": jnp.array([0.95, 0.95]),
        "process_variance": jnp.ones((n_oscillators, n_discrete_states)) * 0.1,
        "phase_difference": jnp.zeros(
            (n_oscillators, n_oscillators, n_discrete_states)
        ),
        "coupling_strength": jnp.zeros(
            (n_oscillators, n_oscillators, n_discrete_states)
        ),
    }


@pytest.fixture(scope="module")
def dim_pp_params():
    """Parameters for DirectedInfluencePointProcessModel tests."""
    n_oscillators = 2
    n_neurons = 5
    n_discrete_states = 3
    sampling_freq = 100.0
    dt = 0.01

    return {
        "n_oscillators": n_oscillators,
        "n_neurons": n_neurons,
        "n_discrete_states": n_discrete_states,
        "sampling_freq": sampling_freq,
        "dt": dt,
        "freqs": jnp.array([8.0, 12.0]),
        "damping_coef": jnp.array([0.95, 0.95]),
        "process_variance": jnp.array([0.1, 0.1]),
        "phase_difference": jnp.zeros(
            (n_oscillators, n_oscillators, n_discrete_states)
        ),
        "coupling_strength": jnp.zeros(
            (n_oscillators, n_oscillators, n_discrete_states)
        ),
    }


@pytest.fixture(scope="module")
def synthetic_spikes():
    """Generate synthetic spike data for model fitting tests."""
    n_time = 200
    n_neurons = 5
    key = jax.random.PRNGKey(42)

    # Poisson spikes with low rate (~1 Hz at dt=0.01 -> ~0.01 expected per bin)
    spikes = jax.random.poisson(key, 0.01, (n_time, n_neurons))
    return spikes


# ============================================================================
# Tests for CommonOscillatorPointProcessModel
# ============================================================================


class TestCommonOscillatorPointProcessModel:
    """Tests for the COM-PP model class."""

    def test_initialization(self, com_pp_params) -> None:
        """Model should initialize without errors."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)

        assert model.n_oscillators == com_pp_params["n_oscillators"]
        assert model.n_neurons == com_pp_params["n_neurons"]
        assert model.n_discrete_states == com_pp_params["n_discrete_states"]
        assert model.n_latent == 2 * com_pp_params["n_oscillators"]

    def test_repr(self, com_pp_params) -> None:
        """__repr__ should include class name and key parameters."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        repr_str = repr(model)

        assert "CommonOscillatorPointProcessModel" in repr_str
        assert "n_oscillators=2" in repr_str

    def test_update_flags(self, com_pp_params) -> None:
        """COM-PP should not update A or Q."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)

        assert model.update_continuous_transition_matrix is False
        assert model.update_process_cov is False
        assert model.update_spike_params is True
        assert model.separate_spike_params is True

    def test_parameter_shapes(self, com_pp_params) -> None:
        """All parameters should have correct shapes after initialization."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        n_osc = com_pp_params["n_oscillators"]
        n_disc = com_pp_params["n_discrete_states"]
        n_neurons = com_pp_params["n_neurons"]
        n_lat = 2 * n_osc

        assert model.init_mean.shape == (n_lat, n_disc)
        assert model.init_cov.shape == (n_lat, n_lat, n_disc)
        assert model.init_discrete_state_prob.shape == (n_disc,)
        assert model.discrete_transition_matrix.shape == (n_disc, n_disc)
        assert model.continuous_transition_matrix.shape == (n_lat, n_lat, n_disc)
        assert model.process_cov.shape == (n_lat, n_lat, n_disc)
        assert model.spike_params.baseline.shape == (n_neurons, n_disc)
        assert model.spike_params.weights.shape == (n_neurons, n_lat, n_disc)

    def test_discrete_transition_rows_sum_to_one(self, com_pp_params) -> None:
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        row_sums = jnp.sum(model.discrete_transition_matrix, axis=1)
        np.testing.assert_allclose(
            row_sums, jnp.ones(model.n_discrete_states), rtol=1e-6
        )

    def test_discrete_state_prob_sums_to_one(self, com_pp_params) -> None:
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        np.testing.assert_allclose(
            jnp.sum(model.init_discrete_state_prob), 1.0, rtol=1e-6
        )

    def test_constant_A_across_states(self, com_pp_params) -> None:
        """A should be identical across all discrete states."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.continuous_transition_matrix[..., 0],
                model.continuous_transition_matrix[..., j],
                rtol=1e-10,
            )

    def test_constant_Q_across_states(self, com_pp_params) -> None:
        """Q should be identical across all discrete states."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.process_cov[..., 0],
                model.process_cov[..., j],
                rtol=1e-10,
            )

    @pytest.mark.slow
    def test_e_step_runs(self, com_pp_params, synthetic_spikes) -> None:
        """E-step should produce finite log-likelihood."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        marginal_ll = model._e_step(synthetic_spikes)
        assert jnp.isfinite(marginal_ll)

    @pytest.mark.slow
    def test_fit_runs(self, com_pp_params, synthetic_spikes) -> None:
        """fit() should complete without error for a few iterations."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        log_likelihoods = model.fit(
            synthetic_spikes, max_iter=3, key=jax.random.PRNGKey(42)
        )

        assert len(log_likelihoods) == 3
        assert all(np.isfinite(ll) for ll in log_likelihoods)


# ============================================================================
# Tests for CorrelatedNoisePointProcessModel
# ============================================================================


class TestCorrelatedNoisePointProcessModel:
    """Tests for the CNM-PP model class."""

    def test_initialization(self, cnm_pp_params) -> None:
        model = CorrelatedNoisePointProcessModel(**cnm_pp_params)

        assert model.n_oscillators == cnm_pp_params["n_oscillators"]
        assert model.n_neurons == cnm_pp_params["n_neurons"]
        assert model.n_discrete_states == cnm_pp_params["n_discrete_states"]

    def test_update_flags(self, cnm_pp_params) -> None:
        """CNM-PP should update Q but not A."""
        model = CorrelatedNoisePointProcessModel(**cnm_pp_params)

        assert model.update_continuous_transition_matrix is False
        assert model.update_process_cov is True

    def test_parameter_shapes(self, cnm_pp_params) -> None:
        model = CorrelatedNoisePointProcessModel(**cnm_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        n_osc = cnm_pp_params["n_oscillators"]
        n_disc = cnm_pp_params["n_discrete_states"]
        n_neurons = cnm_pp_params["n_neurons"]
        n_lat = 2 * n_osc

        assert model.continuous_transition_matrix.shape == (n_lat, n_lat, n_disc)
        assert model.process_cov.shape == (n_lat, n_lat, n_disc)
        assert model.spike_params.baseline.shape == (n_neurons, n_disc)
        assert model.spike_params.weights.shape == (n_neurons, n_lat, n_disc)

    def test_constant_A_across_states(self, cnm_pp_params) -> None:
        """A should be identical across all discrete states."""
        model = CorrelatedNoisePointProcessModel(**cnm_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.continuous_transition_matrix[..., 0],
                model.continuous_transition_matrix[..., j],
                rtol=1e-10,
            )

    def test_process_cov_is_psd(self, cnm_pp_params) -> None:
        """Q should be PSD for each discrete state."""
        model = CorrelatedNoisePointProcessModel(**cnm_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(model.n_discrete_states):
            eigvals = jnp.linalg.eigvalsh(model.process_cov[..., j])
            assert jnp.all(eigvals >= -1e-10)

    @pytest.mark.slow
    def test_e_step_runs(self, cnm_pp_params, synthetic_spikes) -> None:
        model = CorrelatedNoisePointProcessModel(**cnm_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        marginal_ll = model._e_step(synthetic_spikes)
        assert jnp.isfinite(marginal_ll)

    @pytest.mark.slow
    def test_fit_runs(self, cnm_pp_params, synthetic_spikes) -> None:
        model = CorrelatedNoisePointProcessModel(**cnm_pp_params)
        log_likelihoods = model.fit(
            synthetic_spikes, max_iter=3, key=jax.random.PRNGKey(42)
        )

        assert len(log_likelihoods) == 3
        assert all(np.isfinite(ll) for ll in log_likelihoods)

    def test_projection_preserves_psd(self, cnm_pp_params, synthetic_spikes) -> None:
        """After projection, Q should remain PSD."""
        model = CorrelatedNoisePointProcessModel(**cnm_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        # Run one E-step + M-step + projection
        model._e_step(synthetic_spikes)
        model._m_step_dynamics()
        model._project_parameters()

        for j in range(model.n_discrete_states):
            eigvals = jnp.linalg.eigvalsh(model.process_cov[..., j])
            assert jnp.all(eigvals >= -1e-10)


# ============================================================================
# Tests for DirectedInfluencePointProcessModel
# ============================================================================


class TestDirectedInfluencePointProcessModel:
    """Tests for the DIM-PP model class."""

    def test_initialization(self, dim_pp_params) -> None:
        model = DirectedInfluencePointProcessModel(**dim_pp_params)

        assert model.n_oscillators == dim_pp_params["n_oscillators"]
        assert model.n_neurons == dim_pp_params["n_neurons"]
        assert model.n_discrete_states == dim_pp_params["n_discrete_states"]

    def test_update_flags(self, dim_pp_params) -> None:
        """DIM-PP should update A but not Q."""
        model = DirectedInfluencePointProcessModel(**dim_pp_params)

        assert model.update_continuous_transition_matrix is True
        assert model.update_process_cov is False

    def test_parameter_shapes(self, dim_pp_params) -> None:
        model = DirectedInfluencePointProcessModel(**dim_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        n_osc = dim_pp_params["n_oscillators"]
        n_disc = dim_pp_params["n_discrete_states"]
        n_neurons = dim_pp_params["n_neurons"]
        n_lat = 2 * n_osc

        assert model.continuous_transition_matrix.shape == (n_lat, n_lat, n_disc)
        assert model.process_cov.shape == (n_lat, n_lat, n_disc)
        assert model.spike_params.baseline.shape == (n_neurons, n_disc)
        assert model.spike_params.weights.shape == (n_neurons, n_lat, n_disc)

    def test_constant_Q_across_states(self, dim_pp_params) -> None:
        """Q should be identical across all discrete states."""
        model = DirectedInfluencePointProcessModel(**dim_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        for j in range(1, model.n_discrete_states):
            np.testing.assert_allclose(
                model.process_cov[..., 0],
                model.process_cov[..., j],
                rtol=1e-10,
            )

    @pytest.mark.slow
    def test_e_step_runs(self, dim_pp_params, synthetic_spikes) -> None:
        model = DirectedInfluencePointProcessModel(**dim_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        marginal_ll = model._e_step(synthetic_spikes)
        assert jnp.isfinite(marginal_ll)

    @pytest.mark.slow
    def test_fit_runs(self, dim_pp_params, synthetic_spikes) -> None:
        model = DirectedInfluencePointProcessModel(**dim_pp_params)
        log_likelihoods = model.fit(
            synthetic_spikes, max_iter=3, key=jax.random.PRNGKey(42)
        )

        assert len(log_likelihoods) == 3
        assert all(np.isfinite(ll) for ll in log_likelihoods)

    @pytest.mark.slow
    def test_reparameterized_mstep(self, dim_pp_params, synthetic_spikes) -> None:
        """DIM-PP with reparameterized M-step should run without error."""
        model = DirectedInfluencePointProcessModel(
            **dim_pp_params, use_reparameterized_mstep=True
        )
        log_likelihoods = model.fit(
            synthetic_spikes, max_iter=3, key=jax.random.PRNGKey(42)
        )

        assert len(log_likelihoods) == 3
        assert all(np.isfinite(ll) for ll in log_likelihoods)

    def test_projection_preserves_oscillatory_structure(
        self, dim_pp_params, synthetic_spikes
    ) -> None:
        """After projection, A should still have rotation-like 2x2 blocks."""
        model = DirectedInfluencePointProcessModel(**dim_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))

        model._e_step(synthetic_spikes)
        model._m_step_dynamics()
        model._project_parameters()

        # Check each 2x2 diagonal block is approximately a scaled rotation
        # A scaled rotation has equal singular values
        n_osc = model.n_oscillators
        for j in range(model.n_discrete_states):
            A_j = model.continuous_transition_matrix[..., j]
            for k in range(n_osc):
                block = A_j[2 * k : 2 * (k + 1), 2 * k : 2 * (k + 1)]
                s = jnp.linalg.svd(block, compute_uv=False)
                # Singular values should be approximately equal (scaled rotation)
                np.testing.assert_allclose(s[0], s[1], rtol=0.1)


# ============================================================================
# Tests for input validation
# ============================================================================


class TestInputValidation:
    """Tests for constructor input validation across all model classes."""

    def test_invalid_n_oscillators(self) -> None:
        with pytest.raises(ValueError, match="n_oscillators must be positive"):
            CommonOscillatorPointProcessModel(
                n_oscillators=0,
                n_neurons=5,
                n_discrete_states=2,
                sampling_freq=100.0,
                dt=0.01,
                freqs=jnp.array([8.0]),
                damping_coef=jnp.array([0.95]),
                process_variance=jnp.array([0.1]),
            )

    def test_invalid_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            CommonOscillatorPointProcessModel(
                n_oscillators=2,
                n_neurons=5,
                n_discrete_states=2,
                sampling_freq=100.0,
                dt=-0.01,
                freqs=jnp.array([8.0, 12.0]),
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, 0.1]),
            )

    def test_invalid_freqs_shape(self) -> None:
        with pytest.raises(ValueError, match="freqs shape"):
            CommonOscillatorPointProcessModel(
                n_oscillators=2,
                n_neurons=5,
                n_discrete_states=2,
                sampling_freq=100.0,
                dt=0.01,
                freqs=jnp.array([8.0]),  # Wrong shape
                damping_coef=jnp.array([0.95, 0.95]),
                process_variance=jnp.array([0.1, 0.1]),
            )

    def test_invalid_spikes_shape(self, com_pp_params) -> None:
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        wrong_spikes = jnp.zeros((100, 3))  # Wrong n_neurons

        with pytest.raises(ValueError, match="n_neurons"):
            model.fit(wrong_spikes, max_iter=1, key=jax.random.PRNGKey(0))

    def test_invalid_spikes_ndim(self, com_pp_params) -> None:
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        wrong_spikes = jnp.zeros((100,))  # 1D

        with pytest.raises(ValueError, match="2D"):
            model.fit(wrong_spikes, max_iter=1, key=jax.random.PRNGKey(0))


@pytest.mark.slow
class TestSwitchingPPSGDFitting:
    """Tests for switching point-process SGD fitting."""

    @pytest.fixture
    def com_pp_setup(self):
        from state_space_practice.simulate.scenarios import simulate_com_pp_scenario

        scenario = simulate_com_pp_scenario(n_time=100, seed=42)
        p = scenario["params"]
        model = CommonOscillatorPointProcessModel(
            n_oscillators=p["n_oscillators"],
            n_neurons=p["n_neurons"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            dt=p["dt"],
            freqs=p["freqs"],
            damping_coef=p["damping"],
            process_variance=p["process_variance"],
        )
        return model, scenario["spikes"]

    @pytest.fixture
    def dim_pp_setup(self):
        from state_space_practice.simulate.scenarios import simulate_dim_pp_scenario

        scenario = simulate_dim_pp_scenario(n_time=100, seed=42)
        p = scenario["params"]
        model = DirectedInfluencePointProcessModel(
            n_oscillators=p["n_oscillators"],
            n_neurons=p["n_neurons"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            dt=p["dt"],
            freqs=p["freqs"],
            damping_coef=p["damping"],
            process_variance=p["process_variance"],
            phase_difference=p["phase_difference"],
            coupling_strength=p["coupling_strength"],
        )
        return model, scenario["spikes"]

    def test_com_pp_sgd_improves_ll(self, com_pp_setup):
        model, spikes = com_pp_setup
        key = jax.random.PRNGKey(0)
        lls = model.fit_sgd(spikes, key=key, num_steps=20)
        assert lls[-1] > lls[0]

    def test_com_pp_sgd_populates_smoother(self, com_pp_setup):
        model, spikes = com_pp_setup
        key = jax.random.PRNGKey(0)
        model.fit_sgd(spikes, key=key, num_steps=10)
        assert model.smoother_state_cond_mean is not None

    def test_dim_pp_sgd_improves_ll(self, dim_pp_setup):
        model, spikes = dim_pp_setup
        key = jax.random.PRNGKey(0)
        lls = model.fit_sgd(spikes, key=key, num_steps=20)
        assert lls[-1] > lls[0]

    def test_dim_pp_sgd_coupling_finite(self, dim_pp_setup):
        model, spikes = dim_pp_setup
        key = jax.random.PRNGKey(0)
        model.fit_sgd(spikes, key=key, num_steps=10)
        assert jnp.all(jnp.isfinite(model.coupling_strength))
        assert jnp.all(jnp.isfinite(model.phase_difference))

    def test_dim_pp_sgd_with_edge_penalty(self, dim_pp_setup):
        from state_space_practice.oscillator_regularization import (
            OscillatorPenaltyConfig,
        )

        model, spikes = dim_pp_setup
        key = jax.random.PRNGKey(0)
        config = OscillatorPenaltyConfig(edge_l1=0.5)
        lls = model.fit_sgd(
            spikes, key=key, num_steps=15, connectivity_penalty=config,
        )
        assert len(lls) > 1
        assert jnp.all(jnp.isfinite(model.coupling_strength))

    def test_dim_pp_penalty_shrinks_coupling(self):
        """Edge penalty should shrink coupling norm vs unpenalized."""
        from state_space_practice.oscillator_regularization import (
            OscillatorPenaltyConfig,
        )
        from state_space_practice.simulate.scenarios import (
            simulate_dim_pp_scenario,
        )

        scenario = simulate_dim_pp_scenario(n_time=100, seed=42)
        p = scenario["params"]
        key = jax.random.PRNGKey(0)

        def _make():
            return DirectedInfluencePointProcessModel(
                n_oscillators=p["n_oscillators"],
                n_neurons=p["n_neurons"],
                n_discrete_states=p["n_discrete_states"],
                sampling_freq=p["sampling_freq"],
                dt=p["dt"],
                freqs=p["freqs"],
                damping_coef=p["damping"],
                process_variance=p["process_variance"],
                phase_difference=p["phase_difference"],
                coupling_strength=p["coupling_strength"],
            )

        model_base = _make()
        model_base.fit_sgd(scenario["spikes"], key=key, num_steps=20)
        norm_base = float(jnp.sum(jnp.abs(model_base.coupling_strength)))

        model_reg = _make()
        config = OscillatorPenaltyConfig(edge_l1=1.0)
        model_reg.fit_sgd(
            scenario["spikes"], key=key, num_steps=20,
            connectivity_penalty=config,
        )
        norm_reg = float(jnp.sum(jnp.abs(model_reg.coupling_strength)))

        assert norm_reg < norm_base

    def test_spike_weight_l2_shrinks_weights(self):
        """Higher spike_weight_l2 should produce smaller spike weights."""
        from state_space_practice.simulate.scenarios import (
            simulate_dim_pp_scenario,
        )

        scenario = simulate_dim_pp_scenario(n_time=100, seed=42)
        p = scenario["params"]
        key = jax.random.PRNGKey(0)

        def _make(l2):
            return DirectedInfluencePointProcessModel(
                n_oscillators=p["n_oscillators"],
                n_neurons=p["n_neurons"],
                n_discrete_states=p["n_discrete_states"],
                sampling_freq=p["sampling_freq"],
                dt=p["dt"],
                freqs=p["freqs"],
                damping_coef=p["damping"],
                process_variance=p["process_variance"],
                phase_difference=p["phase_difference"],
                coupling_strength=p["coupling_strength"],
                spike_weight_l2=l2,
            )

        model_low = _make(0.01)
        model_low.fit_sgd(scenario["spikes"], key=key, num_steps=20)
        norm_low = float(jnp.sum(model_low.spike_params.weights**2))

        model_high = _make(10.0)
        model_high.fit_sgd(scenario["spikes"], key=key, num_steps=20)
        norm_high = float(jnp.sum(model_high.spike_params.weights**2))

        assert norm_high < norm_low, (
            f"Higher L2 ({norm_high:.4f}) should give smaller weight norm "
            f"than lower L2 ({norm_low:.4f})"
        )
