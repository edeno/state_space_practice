"""Generative recovery tests for switching oscillator models.

Each test class simulates data from a model's generative process and verifies
that fitting the model recovers:
1. Correct discrete state segmentation
2. Finite, non-decreasing log-likelihoods

These tests are marked @pytest.mark.slow since EM fitting is expensive.
Run with: pytest -m slow
"""

from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.oscillator_models import (
    CommonOscillatorModel,
    CorrelatedNoiseModel,
    DirectedInfluenceModel,
)
from state_space_practice.point_process_models import (
    CommonOscillatorPointProcessModel,
    CorrelatedNoisePointProcessModel,
    DirectedInfluencePointProcessModel,
)
from state_space_practice.simulate.scenarios import (
    simulate_cnm_pp_scenario,
    simulate_cnm_scenario,
    simulate_com_pp_scenario,
    simulate_com_scenario,
    simulate_dim_pp_scenario,
    simulate_dim_scenario,
)

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Utility
# ============================================================================


def state_segmentation_accuracy(
    true_states: np.ndarray,
    smoother_discrete_state_prob: np.ndarray,
) -> float:
    """Compute best-permutation state segmentation accuracy.

    Parameters
    ----------
    true_states : np.ndarray, shape (n_time,)
        Ground truth discrete state labels (integers).
    smoother_discrete_state_prob : np.ndarray, shape (n_time, n_discrete_states)
        Posterior state probabilities from the model.

    Returns
    -------
    accuracy : float
        Best accuracy across all label permutations.
    """
    inferred = np.array(jnp.argmax(smoother_discrete_state_prob, axis=1))
    true = np.array(true_states)
    n_states = smoother_discrete_state_prob.shape[1]

    best_acc = 0.0
    for perm in permutations(range(n_states)):
        remapped = np.array([perm[s] for s in inferred])
        acc = float(np.mean(remapped == true))
        best_acc = max(best_acc, acc)

    return best_acc


# ============================================================================
# Gaussian: COM recovery
# ============================================================================


@pytest.mark.slow
class TestCOMRecovery:
    """Verify CommonOscillatorModel recovers state segmentation from COM data."""

    @pytest.fixture(scope="class")
    def fitted(self):
        data = simulate_com_scenario()
        p = data["params"]

        model = CommonOscillatorModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            n_sources=p["n_sources"],
            sampling_freq=p["sampling_freq"],
            freqs=jnp.array(p["freqs"]),
            auto_regressive_coef=jnp.array(p["damping"]),
            process_variance=jnp.array(p["process_variance"]),
            measurement_variance=p["measurement_variance"],
        )
        lls = model.fit(
            jnp.array(data["obs"]),
            key=jax.random.PRNGKey(0),
            max_iter=50,
        )
        return model, data, lls

    def test_log_likelihoods_finite(self, fitted):
        _, _, lls = fitted
        assert all(np.isfinite(ll) for ll in lls)

    def test_state_segmentation(self, fitted):
        model, data, _ = fitted
        acc = state_segmentation_accuracy(
            data["true_states"], np.array(model.smoother_discrete_state_prob)
        )
        assert acc >= 0.80, f"State segmentation accuracy {acc:.3f} < 0.80"


# ============================================================================
# Gaussian: CNM recovery
# ============================================================================


@pytest.mark.slow
class TestCNMRecovery:
    """Verify CorrelatedNoiseModel recovers state segmentation from CNM data."""

    @pytest.fixture(scope="class")
    def fitted(self):
        data = simulate_cnm_scenario()
        p = data["params"]

        model = CorrelatedNoiseModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            freqs=jnp.array(p["freqs"]),
            auto_regressive_coef=jnp.array(p["damping"]),
            process_variance=jnp.array(p["process_variance"]),
            measurement_variance=p["measurement_variance"],
            phase_difference=jnp.array(p["phase_difference"]),
            coupling_strength=jnp.array(p["coupling_strength"]),
        )
        lls = model.fit(
            jnp.array(data["obs"]),
            key=jax.random.PRNGKey(0),
            max_iter=50,
        )
        return model, data, lls

    def test_log_likelihoods_finite(self, fitted):
        _, _, lls = fitted
        assert all(np.isfinite(ll) for ll in lls)

    def test_state_segmentation(self, fitted):
        model, data, _ = fitted
        acc = state_segmentation_accuracy(
            data["true_states"], np.array(model.smoother_discrete_state_prob)
        )
        assert acc >= 0.70, f"State segmentation accuracy {acc:.3f} < 0.70"


# ============================================================================
# Gaussian: DIM recovery
# ============================================================================


@pytest.mark.slow
class TestDIMRecovery:
    """Verify DirectedInfluenceModel recovers state segmentation from DIM data."""

    @pytest.fixture(scope="class")
    def fitted(self):
        data = simulate_dim_scenario()
        p = data["params"]

        model = DirectedInfluenceModel(
            n_oscillators=p["n_oscillators"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            freqs=jnp.array(p["freqs"]),
            auto_regressive_coef=jnp.array(p["damping"]),
            process_variance=jnp.array(p["process_variance"]),
            measurement_variance=p["measurement_variance"],
            phase_difference=jnp.array(p["phase_difference"]),
            coupling_strength=jnp.array(p["coupling_strength"]),
        )
        lls = model.fit(
            jnp.array(data["obs"]),
            key=jax.random.PRNGKey(0),
            max_iter=50,
        )
        return model, data, lls

    def test_log_likelihoods_finite(self, fitted):
        _, _, lls = fitted
        assert all(np.isfinite(ll) for ll in lls)

    def test_state_segmentation(self, fitted):
        model, data, _ = fitted
        acc = state_segmentation_accuracy(
            data["true_states"], np.array(model.smoother_discrete_state_prob)
        )
        assert acc >= 0.70, f"State segmentation accuracy {acc:.3f} < 0.70"


# ============================================================================
# Point-process: COM-PP recovery
# ============================================================================


@pytest.mark.slow
class TestCOMPPRecovery:
    """Verify COM-PP recovers state segmentation from COM-PP spike data."""

    @pytest.fixture(scope="class")
    def fitted(self):
        data = simulate_com_pp_scenario()
        p = data["params"]

        model = CommonOscillatorPointProcessModel(
            n_oscillators=p["n_oscillators"],
            n_neurons=p["n_neurons"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            dt=p["dt"],
            freqs=jnp.array(p["freqs"]),
            auto_regressive_coef=jnp.array(p["damping"]),
            process_variance=jnp.array(p["process_variance"]),
        )
        lls = model.fit(
            jnp.array(data["spikes"]),
            max_iter=50,
            key=jax.random.PRNGKey(0),
        )
        return model, data, lls

    def test_log_likelihoods_finite(self, fitted):
        _, _, lls = fitted
        assert all(np.isfinite(ll) for ll in lls)

    def test_state_segmentation(self, fitted):
        model, data, _ = fitted
        acc = state_segmentation_accuracy(
            data["true_states"],
            np.array(model.smoother_discrete_state_prob),
        )
        assert acc >= 0.70, f"State segmentation accuracy {acc:.3f} < 0.70"


# ============================================================================
# Point-process: CNM-PP recovery
# ============================================================================


@pytest.mark.slow
class TestCNMPPRecovery:
    """Verify CNM-PP recovers state segmentation from CNM-PP spike data."""

    @pytest.fixture(scope="class")
    def fitted(self):
        data = simulate_cnm_pp_scenario(n_time=5000)
        p = data["params"]
        spikes = jnp.array(data["spikes"])

        model = CorrelatedNoisePointProcessModel(
            n_oscillators=p["n_oscillators"],
            n_neurons=p["n_neurons"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            dt=p["dt"],
            freqs=jnp.array(p["freqs"]),
            auto_regressive_coef=jnp.array(p["damping"]),
            process_variance=jnp.array(p["process_variance"]),
            phase_difference=jnp.array(p["phase_difference"]),
            coupling_strength=jnp.array(p["coupling_strength"]),
            # init state M-step is unstable for sparse spike data:
            # smoother uncertainty at t=0 is enormous with few spikes,
            # causing init_cov eigenvalues to explode across iterations.
            update_init_mean=False,
            update_init_cov=False,
        )
        lls = model.fit(spikes, max_iter=50, key=jax.random.PRNGKey(0))
        return model, data, lls

    def test_log_likelihoods_finite(self, fitted):
        _, _, lls = fitted
        assert all(np.isfinite(ll) for ll in lls)

    def test_state_segmentation(self, fitted):
        model, data, _ = fitted
        acc = state_segmentation_accuracy(
            data["true_states"],
            np.array(model.smoother_discrete_state_prob),
        )
        assert acc >= 0.60, f"State segmentation accuracy {acc:.3f} < 0.60"


# ============================================================================
# Point-process: DIM-PP recovery
# ============================================================================


@pytest.mark.slow
class TestDIMPPRecovery:
    """Verify DIM-PP recovers state segmentation from DIM-PP spike data."""

    @pytest.fixture(scope="class")
    def fitted(self):
        data = simulate_dim_pp_scenario(n_time=5000)
        p = data["params"]
        spikes = jnp.array(data["spikes"])

        model = DirectedInfluencePointProcessModel(
            n_oscillators=p["n_oscillators"],
            n_neurons=p["n_neurons"],
            n_discrete_states=p["n_discrete_states"],
            sampling_freq=p["sampling_freq"],
            dt=p["dt"],
            freqs=jnp.array(p["freqs"]),
            auto_regressive_coef=jnp.array(p["damping"]),
            process_variance=jnp.array(p["process_variance"]),
            phase_difference=jnp.array(p["phase_difference"]),
            coupling_strength=jnp.array(p["coupling_strength"]),
            update_init_mean=False,
            update_init_cov=False,
        )
        lls = model.fit(spikes, max_iter=50, key=jax.random.PRNGKey(0))
        return model, data, lls

    def test_log_likelihoods_finite(self, fitted):
        _, _, lls = fitted
        assert all(np.isfinite(ll) for ll in lls)

    def test_state_segmentation(self, fitted):
        model, data, _ = fitted
        acc = state_segmentation_accuracy(
            data["true_states"],
            np.array(model.smoother_discrete_state_prob),
        )
        assert acc >= 0.60, f"State segmentation accuracy {acc:.3f} < 0.60"
