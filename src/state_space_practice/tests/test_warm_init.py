# ruff: noqa: E402
"""Tests for warm initialization of oscillator models.

Warm init uses windowed cross-covariance features + GMM clustering to
break symmetry before the first E-step. These tests verify that warm
init produces better first-iteration accuracy than cold (random) init.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.oscillator_models import DirectedInfluenceModel
from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_directed_influence_measurement_matrix,
    construct_directed_influence_transition_matrix,
)
from state_space_practice.simulate.simulate_switching_kalman import simulate
from state_space_practice.tests.recovery_helpers import state_segmentation_accuracy


@pytest.fixture(scope="module")
def dim_simulation():
    """Simulate a 2-oscillator, 2-state DIM dataset at 100 Hz."""
    n_osc, n_disc = 2, 2
    sampling_freq = 100.0
    n_time = 3000

    freqs = jnp.array([8.0, 25.0])
    damping = jnp.array([0.95, 0.95])
    process_var = jnp.array([0.1, 0.1])
    measurement_var = 0.5

    coupling_0 = jnp.array([[0.0, 0.0], [0.3, 0.0]])
    phase_0 = jnp.array([[0.0, 0.0], [0.5, 0.0]])
    coupling_1 = jnp.array([[0.0, 0.3], [0.0, 0.0]])
    phase_1 = jnp.array([[0.0, 0.5], [0.0, 0.0]])

    A_list = []
    for coup, ph in [(coupling_0, phase_0), (coupling_1, phase_1)]:
        A_list.append(
            np.array(
                construct_directed_influence_transition_matrix(
                    freqs=freqs,
                    damping_coeffs=damping,
                    coupling_strengths=coup,
                    phase_diffs=ph,
                    sampling_freq=sampling_freq,
                )
            )
        )
    A = np.stack(A_list, axis=-1)
    Q = np.array(
        jnp.stack(
            [construct_common_oscillator_process_covariance(variance=process_var)]
            * n_disc,
            axis=-1,
        )
    )
    H = np.array(
        jnp.stack(
            [construct_directed_influence_measurement_matrix(n_osc)] * n_disc, axis=-1
        )
    )
    R = np.array(
        jnp.stack([jnp.eye(n_osc) * measurement_var] * n_disc, axis=-1)
    )

    p_stay = 1.0 - 2.0 / sampling_freq
    Z = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])

    y, s, x = simulate(A, H, Q, R, Z, np.zeros(2 * n_osc), 0, n_time)
    return {
        "obs": y,
        "true_states": s,
        "n_osc": n_osc,
        "n_disc": n_disc,
        "sampling_freq": sampling_freq,
        "freqs": freqs,
        "damping": damping,
        "process_var": process_var,
        "measurement_var": measurement_var,
    }


def _build_model(data):
    """Create a fresh DirectedInfluenceModel from simulation parameters."""
    return DirectedInfluenceModel(
        n_oscillators=data["n_osc"],
        n_discrete_states=data["n_disc"],
        sampling_freq=data["sampling_freq"],
        freqs=data["freqs"],
        damping_coef=data["damping"],
        process_variance=data["process_var"],
        measurement_variance=data["measurement_var"],
        phase_difference=jnp.zeros(
            (data["n_osc"], data["n_osc"], data["n_disc"])
        ),
        coupling_strength=jnp.zeros(
            (data["n_osc"], data["n_osc"], data["n_disc"])
        ),
    )


@pytest.mark.slow
class TestWarmInitConvergence:
    """Warm init should give better first-iteration state segmentation."""

    def test_warm_init_first_iteration_accuracy(self, dim_simulation):
        """Warm init should achieve higher accuracy after 1 E-step than cold init."""
        data = dim_simulation
        obs = jnp.array(data["obs"])
        key = jax.random.PRNGKey(0)

        # Cold init: random parameters only
        model_cold = _build_model(data)
        model_cold._initialize_parameters(key)
        model_cold._e_step(obs)
        acc_cold = state_segmentation_accuracy(
            data["true_states"], np.array(model_cold.smoother_discrete_state_prob)
        )

        # Warm init: random parameters + warm state initialization
        model_warm = _build_model(data)
        model_warm._initialize_parameters(key)
        model_warm._warm_initialize_states(obs)
        model_warm._e_step(obs)
        acc_warm = state_segmentation_accuracy(
            data["true_states"], np.array(model_warm.smoother_discrete_state_prob)
        )

        assert acc_warm > acc_cold, (
            f"Warm init accuracy ({acc_warm:.3f}) should exceed "
            f"cold init accuracy ({acc_cold:.3f}) after first E-step"
        )

    def test_warm_init_converges_to_higher_accuracy(self, dim_simulation):
        """Warm init should converge to a strictly higher, stable accuracy.

        This deliberately does NOT measure "iterations to first cross a fixed
        threshold". Early-EM segmentation accuracy is non-monotonic: cold
        (random) init can spike above a threshold on a transient that
        immediately collapses (observed here: 0.52 -> 0.76 -> 0.53 -> 0.90),
        crossing the line an iteration *before* warm init's stable monotonic
        climb even though warm converges to a strictly higher accuracy (~0.96
        vs ~0.90). A "fewer iterations to cross" assertion therefore rewards
        cold init's lucky instability and fails despite warm init working. The
        robust, meaningful benefit is the accuracy warm init converges to.
        """
        data = dim_simulation
        obs = jnp.array(data["obs"])
        key = jax.random.PRNGKey(0)
        target_acc = 0.70
        n_iter = 15

        def _converged_accuracy(use_warm):
            model = _build_model(data)
            model._initialize_parameters(key)
            if use_warm:
                model._warm_initialize_states(obs)

            acc = 0.0
            for i in range(n_iter):
                model._e_step(obs)
                acc = state_segmentation_accuracy(
                    data["true_states"],
                    np.array(model.smoother_discrete_state_prob),
                )
                if i < n_iter - 1:
                    model._m_step(obs)
                    model._project_parameters()
            return acc

        acc_warm = _converged_accuracy(use_warm=True)
        acc_cold = _converged_accuracy(use_warm=False)

        # Guard: warm init must actually converge (not a vacuous comparison).
        assert acc_warm >= target_acc, (
            f"Warm init should converge above {target_acc:.0%} accuracy, "
            f"got {acc_warm:.3f}"
        )
        # The real benefit: warm init reaches a strictly higher stable accuracy
        # than cold. The observed margin is ~0.06; 0.02 leaves headroom while
        # still failing if warm init regresses to a no-op (== cold).
        assert acc_warm > acc_cold + 0.02, (
            f"Warm init should converge to a strictly higher accuracy than "
            f"cold init (warm={acc_warm:.3f}, cold={acc_cold:.3f})"
        )
