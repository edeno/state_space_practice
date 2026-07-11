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

        assert len(log_likelihoods) == 4
        assert all(np.isfinite(ll) for ll in log_likelihoods)

    def test_fit_appends_synced_final_ll(self, com_pp_params, synthetic_spikes) -> None:
        """Final LL should match the final parameters after max_iter exhaustion."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        log_likelihoods = model.fit(
            synthetic_spikes, max_iter=1, key=jax.random.PRNGKey(42)
        )
        fresh_ll = float(model._e_step(synthetic_spikes))

        assert len(log_likelihoods) == 2
        assert log_likelihoods[-1] == pytest.approx(fresh_ll)

    def test_fit_rolls_back_bad_final_m_step(
        self, com_pp_params, monkeypatch, caplog
    ) -> None:
        """A final E-step LL decrease should restore the accepted EM state."""
        params = dict(com_pp_params)
        params["n_oscillators"] = 1
        params["n_neurons"] = 2
        params["n_discrete_states"] = 2
        params["freqs"] = jnp.array([8.0])
        params["damping_coef"] = jnp.array([0.95])
        params["process_variance"] = jnp.array([0.1])
        model = CommonOscillatorPointProcessModel(**params)
        model._initialize_parameters(jax.random.PRNGKey(0))
        spikes = jnp.zeros((5, 2))
        original_A = model.continuous_transition_matrix.copy()
        good_probs = jnp.ones((spikes.shape[0], model.n_discrete_states)) / 2.0
        bad_probs = jnp.tile(jnp.array([[0.9, 0.1]]), (spikes.shape[0], 1))
        e_step_values = iter([(100.0, good_probs), (50.0, bad_probs)])

        def fake_e_step(_spikes):
            ll, probs = next(e_step_values)
            model.smoother_discrete_state_prob = probs
            return jnp.asarray(ll)

        def bad_m_step_dynamics():
            model.continuous_transition_matrix = (
                model.continuous_transition_matrix + 5.0
            )

        monkeypatch.setattr(model, "_e_step", fake_e_step)
        monkeypatch.setattr(model, "_m_step_dynamics", bad_m_step_dynamics)
        monkeypatch.setattr(model, "_m_step_spikes", lambda _spikes: None)
        monkeypatch.setattr(model, "_project_parameters", lambda: None)

        with caplog.at_level("WARNING"):
            log_likelihoods = model.fit(spikes, max_iter=1, skip_init=True)

        assert log_likelihoods == [100.0]
        assert any(
            "final e-step decreased ll" in record.message.lower()
            for record in caplog.records
        )
        np.testing.assert_allclose(model.continuous_transition_matrix, original_A)
        np.testing.assert_allclose(model.smoother_discrete_state_prob, good_probs)

    def test_fit_rolls_back_nonfinite_final_m_step(
        self, com_pp_params, monkeypatch, caplog
    ) -> None:
        """A non-finite final E-step should not leave bad state or LL history."""
        params = dict(com_pp_params)
        params["n_oscillators"] = 1
        params["n_neurons"] = 2
        params["n_discrete_states"] = 2
        params["freqs"] = jnp.array([8.0])
        params["damping_coef"] = jnp.array([0.95])
        params["process_variance"] = jnp.array([0.1])
        model = CommonOscillatorPointProcessModel(**params)
        model._initialize_parameters(jax.random.PRNGKey(0))
        spikes = jnp.zeros((5, 2))
        original_A = model.continuous_transition_matrix.copy()
        good_probs = jnp.ones((spikes.shape[0], model.n_discrete_states)) / 2.0
        bad_probs = jnp.tile(jnp.array([[0.9, 0.1]]), (spikes.shape[0], 1))
        e_step_values = iter([(100.0, good_probs), (jnp.nan, bad_probs)])

        def fake_e_step(_spikes):
            ll, probs = next(e_step_values)
            model.smoother_discrete_state_prob = probs
            return jnp.asarray(ll)

        def bad_m_step_dynamics():
            model.continuous_transition_matrix = (
                model.continuous_transition_matrix + 5.0
            )

        monkeypatch.setattr(model, "_e_step", fake_e_step)
        monkeypatch.setattr(model, "_m_step_dynamics", bad_m_step_dynamics)
        monkeypatch.setattr(model, "_m_step_spikes", lambda _spikes: None)
        monkeypatch.setattr(model, "_project_parameters", lambda: None)

        with caplog.at_level("WARNING"):
            log_likelihoods = model.fit(spikes, max_iter=1, skip_init=True)

        assert log_likelihoods == [100.0]
        assert any(
            "non-finite log-likelihood" in record.message.lower()
            for record in caplog.records
        )
        np.testing.assert_allclose(model.continuous_transition_matrix, original_A)
        np.testing.assert_allclose(model.smoother_discrete_state_prob, good_probs)

    def test_fit_rolls_back_nonfinite_mid_loop(
        self, com_pp_params, monkeypatch, caplog
    ) -> None:
        """A non-finite E-step *mid-EM* (e.g. GPB smoother divergence poisoning a
        later E-step) must roll back to the last accepted state and stop, not
        raise -- the smoother signals divergence via a non-finite marginal LL."""
        params = dict(com_pp_params)
        params["n_oscillators"] = 1
        params["n_neurons"] = 2
        params["n_discrete_states"] = 2
        params["freqs"] = jnp.array([8.0])
        params["damping_coef"] = jnp.array([0.95])
        params["process_variance"] = jnp.array([0.1])
        model = CommonOscillatorPointProcessModel(**params)
        model._initialize_parameters(jax.random.PRNGKey(0))
        spikes = jnp.zeros((5, 2))
        original_A = model.continuous_transition_matrix.copy()
        good_probs = jnp.ones((spikes.shape[0], model.n_discrete_states)) / 2.0
        bad_probs = jnp.tile(jnp.array([[0.9, 0.1]]), (spikes.shape[0], 1))
        # iter 0 finite (accepted); iter 1 non-finite -> exercises the mid-loop
        # branch (max_iter=2), not the final-sync path.
        e_step_values = iter([(100.0, good_probs), (jnp.nan, bad_probs)])

        def fake_e_step(_spikes):
            ll, probs = next(e_step_values)
            model.smoother_discrete_state_prob = probs
            return jnp.asarray(ll)

        def bad_m_step_dynamics():
            model.continuous_transition_matrix = (
                model.continuous_transition_matrix + 5.0
            )

        monkeypatch.setattr(model, "_e_step", fake_e_step)
        monkeypatch.setattr(model, "_m_step_dynamics", bad_m_step_dynamics)
        monkeypatch.setattr(model, "_m_step_spikes", lambda _spikes: None)
        monkeypatch.setattr(model, "_project_parameters", lambda: None)

        with caplog.at_level("WARNING"):
            log_likelihoods = model.fit(spikes, max_iter=2, skip_init=True)

        assert log_likelihoods == [100.0]  # the NaN was dropped, not raised
        assert model.converged_ is False
        assert any(
            "non-finite log-likelihood" in record.message.lower()
            and "rolling back" in record.message.lower()
            for record in caplog.records
        )
        np.testing.assert_allclose(model.continuous_transition_matrix, original_A)
        np.testing.assert_allclose(model.smoother_discrete_state_prob, good_probs)

    @pytest.mark.parametrize(
        ("bad_value", "match"),
        [(-1.0, "non-negative"), (0.5, "integer-valued"), (jnp.nan, "finite")],
    )
    def test_fit_rejects_invalid_spike_counts(
        self, com_pp_params, synthetic_spikes, bad_value, match
    ) -> None:
        """Switching PP fit validates spike counts, matching PointProcessModel."""
        model = CommonOscillatorPointProcessModel(**com_pp_params)
        bad_spikes = synthetic_spikes.astype(float).at[0, 0].set(bad_value)

        with pytest.raises(ValueError, match=match):
            model.fit(bad_spikes, max_iter=1, key=jax.random.PRNGKey(0))


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

    def test_lower_triangle_pair_params_are_canonicalized(self, cnm_pp_params) -> None:
        params = dict(cnm_pp_params)
        n_disc = params["n_discrete_states"]
        params["phase_difference"] = jnp.zeros((2, 2, n_disc)).at[1, 0, :].set(-0.7)
        params["coupling_strength"] = jnp.zeros((2, 2, n_disc)).at[1, 0, :].set(0.05)

        model = CorrelatedNoisePointProcessModel(**params)

        np.testing.assert_allclose(model.phase_difference[0, 1, :], 0.7)
        np.testing.assert_allclose(model.coupling_strength[0, 1, :], 0.05)
        np.testing.assert_allclose(model.phase_difference[1, 0, :], 0.0)
        np.testing.assert_allclose(model.coupling_strength[1, 0, :], 0.0)

    def test_conflicting_pair_params_raise(self, cnm_pp_params) -> None:
        params = dict(cnm_pp_params)
        n_disc = params["n_discrete_states"]
        params["phase_difference"] = (
            jnp.zeros((2, 2, n_disc)).at[0, 1, :].set(0.7).at[1, 0, :].set(0.3)
        )
        params["coupling_strength"] = (
            jnp.zeros((2, 2, n_disc)).at[0, 1, :].set(0.05).at[1, 0, :].set(0.05)
        )

        with pytest.raises(ValueError, match="Conflicting correlated-noise"):
            CorrelatedNoisePointProcessModel(**params)

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

        assert len(log_likelihoods) == 4
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

    @pytest.mark.slow
    def test_sgd_loss_finite_and_differentiable_at_indefinite_coupling(
        self, cnm_pp_params, synthetic_spikes
    ) -> None:
        # coupling_strength is UNCONSTRAINED during SGD, so the optimizer can
        # propose a coupling whose reconstructed Q is indefinite. The loss must
        # stay finite AND differentiable there (via the gradient-safe PSD shift);
        # without the fix both go NaN and poison the optimizer. The guard confirms
        # the proposed Q is actually indefinite so the barrier path is exercised.
        from state_space_practice.oscillator_utils import (
            construct_correlated_noise_process_covariance,
        )

        model = CorrelatedNoisePointProcessModel(**cnm_pp_params)
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
            return model._sgd_loss_fn({"coupling_strength": cp}, synthetic_spikes)

        assert bool(jnp.isfinite(loss(big)))
        assert bool(jnp.isfinite(jax.grad(loss)(big)))


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

        assert len(log_likelihoods) == 4
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

        assert len(log_likelihoods) == 4
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

    def test_projection_is_hard_structural_constraint(self) -> None:
        """DIM-PP projection should not keep an out-of-family unconstrained A."""
        model = DirectedInfluencePointProcessModel(
            n_oscillators=1,
            n_neurons=1,
            n_discrete_states=1,
            sampling_freq=100.0,
            dt=0.01,
            freqs=jnp.array([8.0]),
            damping_coef=jnp.array([0.95]),
            process_variance=jnp.array([0.1]),
            phase_difference=jnp.zeros((1, 1, 1)),
            coupling_strength=jnp.zeros((1, 1, 1)),
        )
        model._initialize_parameters(jax.random.PRNGKey(0))
        unconstrained_A = jnp.array([[0.8, 0.2], [0.1, 0.7]])
        model.continuous_transition_matrix = unconstrained_A[:, :, None]

        model._project_parameters()

        projected = model.continuous_transition_matrix[:, :, 0]
        assert not jnp.allclose(projected, unconstrained_A)
        np.testing.assert_allclose(projected[0, 0], projected[1, 1], atol=1e-8)
        np.testing.assert_allclose(projected[0, 1], -projected[1, 0], atol=1e-8)

    def test_reparameterized_mstep_clips_init_state_updates(
        self, dim_pp_params, monkeypatch
    ) -> None:
        """DIM-PP reparameterized M-step should share init-state regularization."""
        import state_space_practice.point_process_models as pp_models

        params = dict(dim_pp_params)
        params["n_oscillators"] = 1
        params["n_neurons"] = 1
        params["n_discrete_states"] = 1
        params["freqs"] = jnp.array([8.0])
        params["damping_coef"] = jnp.array([0.95])
        params["process_variance"] = jnp.array([0.1])
        params["phase_difference"] = jnp.zeros((1, 1, 1))
        params["coupling_strength"] = jnp.zeros((1, 1, 1))
        model = DirectedInfluencePointProcessModel(
            **params, use_reparameterized_mstep=True
        )
        model._initialize_parameters(jax.random.PRNGKey(0))

        n_time, n_latent, n_states = 2, model.n_latent, model.n_discrete_states
        model.smoother_state_cond_mean = jnp.zeros((n_time, n_latent, n_states))
        model.smoother_state_cond_cov = jnp.tile(
            jnp.eye(n_latent)[None, :, :, None],
            (n_time, 1, 1, n_states),
        )
        model.smoother_discrete_state_prob = jnp.ones((n_time, n_states))
        model.smoother_joint_discrete_state_prob = jnp.ones((n_time - 1, 1, 1))
        model.smoother_pair_cond_cross_cov = jnp.zeros(
            (n_time - 1, n_latent, n_latent, 1, 1)
        )
        model.smoother_pair_cond_means = None
        model.smoother_pair_cond_covs = None
        model.smoother_next_pair_cond_means = None

        def fake_m_step(**_kwargs):
            return (
                model.continuous_transition_matrix,
                jnp.zeros((1, n_latent, 1)),
                model.process_cov,
                jnp.eye(1)[:, :, None],
                jnp.ones((n_latent, 1)) * 100.0,
                jnp.eye(n_latent)[:, :, None] * 100.0,
                jnp.ones((1, 1)),
                jnp.ones((1,)),
            )

        def fake_optimize(**_kwargs):
            # The joint optimizer returns shared freq/damping and per-state
            # (n_osc, n_osc, n_states) coupling/phase.
            return {
                "freq": jnp.array([8.0]),
                "damping": jnp.array([0.95]),
                "coupling_strength": jnp.zeros((1, 1, 1)),
                "phase_diff": jnp.zeros((1, 1, 1)),
            }

        monkeypatch.setattr(
            pp_models, "switching_kalman_maximization_step", fake_m_step
        )
        monkeypatch.setattr(
            pp_models, "optimize_dim_transition_params_joint", fake_optimize
        )

        model._m_step_reparameterized()

        assert jnp.max(jnp.abs(model.init_mean)) <= 10.0
        eigvals = jnp.linalg.eigvalsh(model.init_cov[:, :, 0])
        assert jnp.min(eigvals) >= 1e-4 - 1e-10
        assert jnp.max(eigvals) <= 2.0 + 1e-10

    def test_reparameterized_mstep_produces_stable_reconstructable_A(
        self, dim_pp_params
    ) -> None:
        """The joint reparameterized M-step must leave a stable A that
        reconstructs from the intrinsic public params via the shared scale
        (no damping drift; A applies the global stability scale)."""
        from state_space_practice.oscillator_utils import (
            compute_directed_influence_stability_scale,
            construct_directed_influence_transition_matrix,
        )

        params = dict(dim_pp_params)
        params["coupling_strength"] = (
            jnp.zeros_like(params["coupling_strength"])
            .at[0, 1, :]
            .set(0.4)
            .at[1, 0, :]
            .set(0.4)
        )
        model = DirectedInfluencePointProcessModel(
            **params, use_reparameterized_mstep=True
        )
        model._initialize_parameters(jax.random.PRNGKey(0))
        spikes = jax.random.poisson(
            jax.random.PRNGKey(1), 0.3, (150, model.n_neurons)
        ).astype(float)
        model.fit(spikes, max_iter=5, key=jax.random.PRNGKey(0))

        assert isinstance(model._current_osc_params, dict)  # joint (not per-state)
        assert bool(jnp.all((model.damping_coef >= 0) & (model.damping_coef <= 1)))
        scale = compute_directed_influence_stability_scale(
            model.freqs,
            model.damping_coef,
            model.coupling_strength,
            model.sampling_freq,
        )
        for j in range(model.n_discrete_states):
            A = model.continuous_transition_matrix[:, :, j]
            assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(A)))) <= 0.99 + 1e-6
            recon = construct_directed_influence_transition_matrix(
                freqs=model.freqs,
                damping_coeffs=model.damping_coef * scale,
                coupling_strengths=model.coupling_strength[:, :, j] * scale,
                phase_diffs=model.phase_difference[:, :, j],
                sampling_freq=model.sampling_freq,
            )
            np.testing.assert_allclose(A, recon, atol=1e-9)

    def test_rejects_invalid_stability_bounds(self, dim_pp_params) -> None:
        """Stability bounds outside ``(0, 1)`` should raise at construction."""
        with pytest.raises(ValueError, match="max_spectral_radius must lie in"):
            DirectedInfluencePointProcessModel(**dim_pp_params, max_spectral_radius=1.5)
        with pytest.raises(ValueError, match="max_damping must lie in"):
            DirectedInfluencePointProcessModel(**dim_pp_params, max_damping=0.0)

    @pytest.mark.slow
    def test_custom_max_spectral_radius_bounds_reparameterized_A(
        self, dim_pp_params
    ) -> None:
        """A custom ``max_spectral_radius`` must bind the reparameterized M-step.

        The fitted transition matrix must (a) stay within the tightened radius
        and (b) reconstruct from the intrinsic public params via the *same*
        custom scale -- reconstruction with the default 0.99 scale would not
        match, so this proves the bound is threaded end-to-end.
        """
        from state_space_practice.oscillator_utils import (
            compute_directed_influence_stability_scale,
            construct_directed_influence_transition_matrix,
        )

        max_spectral_radius = 0.7
        params = dict(dim_pp_params)
        params["coupling_strength"] = (
            jnp.zeros_like(params["coupling_strength"])
            .at[0, 1, :]
            .set(0.4)
            .at[1, 0, :]
            .set(0.4)
        )
        model = DirectedInfluencePointProcessModel(
            **params,
            use_reparameterized_mstep=True,
            max_spectral_radius=max_spectral_radius,
        )
        model._initialize_parameters(jax.random.PRNGKey(0))
        spikes = jax.random.poisson(
            jax.random.PRNGKey(1), 0.3, (150, model.n_neurons)
        ).astype(float)
        model.fit(spikes, max_iter=5, key=jax.random.PRNGKey(0))

        scale = compute_directed_influence_stability_scale(
            model.freqs,
            model.damping_coef,
            model.coupling_strength,
            model.sampling_freq,
            max_spectral_radius=max_spectral_radius,
        )
        # Guard: the tightened bound must actually bind (scale < 1), else the
        # radius checks below would pass trivially for an unscaled matrix.
        assert float(scale) < 1.0
        for j in range(model.n_discrete_states):
            A = model.continuous_transition_matrix[:, :, j]
            assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(A)))) <= (
                max_spectral_radius + 1e-6
            )
            recon = construct_directed_influence_transition_matrix(
                freqs=model.freqs,
                damping_coeffs=model.damping_coef * scale,
                coupling_strengths=model.coupling_strength[:, :, j] * scale,
                phase_diffs=model.phase_difference[:, :, j],
                sampling_freq=model.sampling_freq,
            )
            np.testing.assert_allclose(A, recon, atol=1e-9)

    def test_initialization_respects_max_spectral_radius(self, dim_pp_params) -> None:
        """The initial transition matrices must honor ``max_spectral_radius``
        before the first E-step (matching the Gaussian DIM)."""
        from state_space_practice.oscillator_utils import (
            construct_directed_influence_transition_matrix,
        )

        params = dict(dim_pp_params)
        params["coupling_strength"] = (
            jnp.zeros_like(params["coupling_strength"])
            .at[0, 1, :]
            .set(0.4)
            .at[1, 0, :]
            .set(0.4)
        )
        model = DirectedInfluencePointProcessModel(**params, max_spectral_radius=0.7)
        model._initialize_parameters(jax.random.PRNGKey(0))

        # Guard: raw (unscaled) construction would exceed the bound, so the
        # scale must actually be doing work here.
        raw = construct_directed_influence_transition_matrix(
            freqs=model.freqs,
            damping_coeffs=model.damping_coef,
            coupling_strengths=model.coupling_strength[:, :, 0],
            phase_diffs=model.phase_difference[:, :, 0],
            sampling_freq=model.sampling_freq,
        )
        assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(raw)))) > 0.7
        for j in range(model.n_discrete_states):
            A = model.continuous_transition_matrix[:, :, j]
            assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(A)))) <= 0.7 + 1e-6

    def test_store_sgd_params_respects_max_spectral_radius(self, dim_pp_params) -> None:
        """Storing SGD parameters must rebuild A through the stability scale so
        stored matrices honor the bound and stay reconstructable."""
        from state_space_practice.oscillator_utils import (
            compute_directed_influence_stability_scale,
            construct_directed_influence_transition_matrix,
        )

        model = DirectedInfluencePointProcessModel(
            **dim_pp_params, max_spectral_radius=0.7
        )
        model._initialize_parameters(jax.random.PRNGKey(0))
        n_osc, n_disc = model.n_oscillators, model.n_discrete_states
        strong = (
            jnp.zeros((n_osc, n_osc, n_disc)).at[0, 1, :].set(0.4).at[1, 0, :].set(0.4)
        )
        model._store_sgd_params(
            {
                "coupling_strength": strong,
                "phase_difference": jnp.zeros((n_osc, n_osc, n_disc)),
            }
        )

        scale = compute_directed_influence_stability_scale(
            model.freqs,
            model.damping_coef,
            model.coupling_strength,
            model.sampling_freq,
            max_spectral_radius=0.7,
        )
        assert float(scale) < 1.0  # guard: the bound binds for this coupling
        for j in range(n_disc):
            A = model.continuous_transition_matrix[:, :, j]
            assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(A)))) <= 0.7 + 1e-6
            recon = construct_directed_influence_transition_matrix(
                freqs=model.freqs,
                damping_coeffs=model.damping_coef * scale,
                coupling_strengths=model.coupling_strength[:, :, j] * scale,
                phase_diffs=model.phase_difference[:, :, j],
                sampling_freq=model.sampling_freq,
            )
            np.testing.assert_allclose(A, recon, atol=1e-9)

    @pytest.mark.slow
    def test_default_em_leaves_reconstructable_transition_matrix(
        self, dim_pp_params
    ) -> None:
        """The default (non-reparameterized) EM must sync all four scientific
        parameters so the fitted A reconstructs from the public params."""
        from state_space_practice.oscillator_utils import (
            compute_directed_influence_stability_scale,
            construct_directed_influence_transition_matrix,
        )

        params = dict(dim_pp_params)
        params["coupling_strength"] = (
            jnp.zeros_like(params["coupling_strength"])
            .at[0, 1, :]
            .set(0.4)
            .at[1, 0, :]
            .set(0.4)
        )
        model = DirectedInfluencePointProcessModel(
            **params, use_reparameterized_mstep=False
        )
        model._initialize_parameters(jax.random.PRNGKey(0))
        spikes = jax.random.poisson(
            jax.random.PRNGKey(1), 0.3, (150, model.n_neurons)
        ).astype(float)
        model.fit(spikes, max_iter=5, key=jax.random.PRNGKey(0))

        scale = compute_directed_influence_stability_scale(
            model.freqs,
            model.damping_coef,
            model.coupling_strength,
            model.sampling_freq,
            max_spectral_radius=model.max_spectral_radius,
        )
        for j in range(model.n_discrete_states):
            A = model.continuous_transition_matrix[:, :, j]
            assert float(jnp.max(jnp.abs(jnp.linalg.eigvals(A)))) <= (
                model.max_spectral_radius + 1e-6
            )
            recon = construct_directed_influence_transition_matrix(
                freqs=model.freqs,
                damping_coeffs=model.damping_coef * scale,
                coupling_strengths=model.coupling_strength[:, :, j] * scale,
                phase_diffs=model.phase_difference[:, :, j],
                sampling_freq=model.sampling_freq,
            )
            np.testing.assert_allclose(A, recon, atol=1e-8)

    def test_rebuild_refreshes_joint_optimizer_cache(self, dim_pp_params) -> None:
        """Changing public dynamics via a rebuild (e.g. an SGD store) must
        refresh the joint-optimizer warm-start cache, so a later
        reparameterized M-step does not warm-start from -- or, on BFGS
        fallback, restore -- stale pre-change dynamics."""
        n_osc = dim_pp_params["n_oscillators"]
        n_disc = dim_pp_params["n_discrete_states"]

        # A fresh init leaves the cache empty (no spurious warm start).
        fresh = DirectedInfluencePointProcessModel(**dim_pp_params)
        fresh._initialize_parameters(jax.random.PRNGKey(0))
        assert fresh._current_osc_params is None

        model = DirectedInfluencePointProcessModel(**dim_pp_params)
        model._initialize_parameters(jax.random.PRNGKey(0))
        # Simulate a prior joint M-step having cached (now stale) zero coupling.
        model._current_osc_params = {
            "freq": model.freqs,
            "damping": model.damping_coef,
            "coupling_strength": jnp.zeros((n_osc, n_osc, n_disc)),
            "phase_diff": jnp.zeros((n_osc, n_osc, n_disc)),
        }
        strong = (
            jnp.zeros((n_osc, n_osc, n_disc)).at[0, 1, :].set(0.4).at[1, 0, :].set(0.4)
        )
        model._store_sgd_params(
            {
                "coupling_strength": strong,
                "phase_difference": jnp.zeros((n_osc, n_osc, n_disc)),
            }
        )

        # Guard: the public coupling genuinely changed away from the cached zeros.
        assert float(jnp.max(jnp.abs(model.coupling_strength))) > 0.0
        # The cache now tracks the updated public params, not the stale zeros.
        assert model._current_osc_params is not None
        np.testing.assert_allclose(
            model._current_osc_params["coupling_strength"], model.coupling_strength
        )
        np.testing.assert_allclose(
            model._current_osc_params["damping"], model.damping_coef
        )


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

    @pytest.mark.parametrize("field", ["phase_difference", "coupling_strength"])
    def test_dim_rejects_nonzero_diagonal_pair_params(
        self, dim_pp_params, field
    ) -> None:
        params = dict(dim_pp_params)
        params[field] = params[field].at[0, 0, :].set(0.2)

        with pytest.raises(ValueError, match="diagonal"):
            DirectedInfluencePointProcessModel(**params)

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
            spikes,
            key=key,
            num_steps=15,
            connectivity_penalty=config,
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
            scenario["spikes"],
            key=key,
            num_steps=20,
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


class TestPointProcessValidation:
    """Construction-time validators wired into _validate_parameter_shapes."""

    def test_rejects_low_sampling_freq_via_p_stay(self, com_pp_params):
        # sampling_freq < 1/expected_dwell -> computed default p_stay < 0.
        params = dict(com_pp_params)
        params["sampling_freq"] = 0.5
        with pytest.raises(ValueError, match="p_stay"):
            CommonOscillatorPointProcessModel(**params)

    def test_rejects_negative_process_variance(self, com_pp_params):
        # Negative variance -> indefinite diagonal Q. process_cov is built and
        # validated in _initialize_parameters (fit-time setup), before the EM
        # loop, so it is caught there rather than at __init__.
        params = dict(com_pp_params)
        params["process_variance"] = jnp.array([-0.1, 0.1])
        model = CommonOscillatorPointProcessModel(**params)
        with pytest.raises(ValueError, match="process_cov"):
            model._initialize_parameters(jax.random.PRNGKey(0))

    def test_cnm_rejects_indefinite_process_cov(self, cnm_pp_params):
        # Coupling magnitude above the process variance makes the (symmetric)
        # correlated-noise Q indefinite; caught in _initialize_parameters before
        # it reaches the Cholesky-based filter on EM iteration 0.
        params = dict(cnm_pp_params)
        n_osc = params["n_oscillators"]
        n_disc = params["n_discrete_states"]
        coupling = np.zeros((n_osc, n_osc, n_disc))
        coupling[0, 1, :] = 0.5  # > process_variance (0.1)
        params["coupling_strength"] = jnp.array(coupling)
        model = CorrelatedNoisePointProcessModel(**params)
        with pytest.raises(ValueError, match="process_cov"):
            model._initialize_parameters(jax.random.PRNGKey(0))
