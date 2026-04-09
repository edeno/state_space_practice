# ruff: noqa: E402
"""Tests for oscillator connectivity regularization penalties."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.oscillator_regularization import (
    OscillatorPenaltyConfig,
    area_group_penalty,
    edge_l1_penalty,
    get_area_coupling_summary,
    state_shared_area_penalty,
    total_connectivity_penalty,
)


class TestEdgeL1Penalty:
    def test_near_zero_when_no_coupling(self):
        coupling = jnp.zeros((2, 3, 3))
        # Smooth L1 gives sqrt(eps) per element, so not exactly zero
        assert float(edge_l1_penalty(coupling)) < 0.01

    def test_positive_for_nonzero_coupling(self):
        coupling = jnp.ones((2, 3, 3)) * 0.5
        assert float(edge_l1_penalty(coupling)) > 0

    def test_excludes_diagonal_by_default(self):
        # Only diagonal entries — penalty should be near zero
        coupling = jnp.zeros((2, 3, 3))
        for j in range(2):
            coupling = coupling.at[j].set(jnp.diag(jnp.ones(3)))
        assert float(edge_l1_penalty(coupling, exclude_diagonal=True)) < 0.01

    def test_includes_diagonal_when_requested(self):
        coupling = jnp.zeros((2, 3, 3))
        for j in range(2):
            coupling = coupling.at[j].set(jnp.diag(jnp.ones(3)))
        assert float(edge_l1_penalty(coupling, exclude_diagonal=False)) > 0.1

    def test_gradient_finite(self):
        coupling = jnp.ones((2, 3, 3)) * 0.5
        g = jax.grad(lambda c: edge_l1_penalty(c))(coupling)
        assert jnp.all(jnp.isfinite(g))


class TestAreaGroupPenalty:
    def test_groups_by_area_labels(self):
        coupling = jnp.array([
            [[0.0, 1.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        area_labels = jnp.array([0, 0, 1])
        value = area_group_penalty(coupling, area_labels)
        assert value > 0.0

    def test_zero_coupling_near_zero_penalty(self):
        coupling = jnp.zeros((2, 3, 3))
        area_labels = jnp.array([0, 0, 1])
        assert float(area_group_penalty(coupling, area_labels)) < 1e-3

    def test_gradient_finite(self):
        coupling = jnp.ones((2, 3, 3)) * 0.3
        area_labels = jnp.array([0, 0, 1])
        g = jax.grad(lambda c: area_group_penalty(c, area_labels))(coupling)
        assert jnp.all(jnp.isfinite(g))


class TestStateSharedAreaPenalty:
    def test_invariant_to_state_permutation(self):
        coupling = jnp.array([
            [[0.0, 1.0], [0.5, 0.0]],
            [[0.0, 2.0], [0.5, 0.0]],
            [[0.0, 0.5], [1.0, 0.0]],
        ])
        area_labels = jnp.array([0, 1])
        original = state_shared_area_penalty(coupling, area_labels)
        permuted = state_shared_area_penalty(coupling[jnp.array([2, 0, 1])], area_labels)
        np.testing.assert_allclose(float(original), float(permuted), atol=1e-10)

    def test_reduces_to_area_penalty_for_single_state(self):
        coupling = jnp.zeros((3, 2, 2))
        coupling = coupling.at[0].set(jnp.array([[0.0, 1.0], [0.5, 0.0]]))
        area_labels = jnp.array([0, 1])
        shared = state_shared_area_penalty(coupling, area_labels)
        single = area_group_penalty(coupling[:1], area_labels)
        np.testing.assert_allclose(float(shared), float(single), atol=1e-6)

    def test_gradient_finite(self):
        coupling = jnp.ones((2, 3, 3)) * 0.3
        area_labels = jnp.array([0, 0, 1])
        g = jax.grad(lambda c: state_shared_area_penalty(c, area_labels))(coupling)
        assert jnp.all(jnp.isfinite(g))


class TestOscillatorPenaltyConfig:
    def test_default_zero_penalties(self):
        config = OscillatorPenaltyConfig()
        assert config.edge_l1 == 0.0
        assert config.area_group_l2 == 0.0
        assert config.state_shared_group_l2 == 0.0

    def test_total_penalty_zero_with_default_config(self):
        config = OscillatorPenaltyConfig()
        coupling = jnp.ones((2, 3, 3))
        penalty = total_connectivity_penalty(coupling, config)
        assert float(penalty) == 0.0

    def test_total_penalty_with_edge_l1(self):
        config = OscillatorPenaltyConfig(edge_l1=1.0)
        coupling = jnp.ones((2, 3, 3)) * 0.5
        penalty = total_connectivity_penalty(coupling, config)
        assert float(penalty) > 0.0

    def test_total_penalty_with_area_labels(self):
        config = OscillatorPenaltyConfig(
            area_group_l2=1.0,
            area_labels=jnp.array([0, 0, 1]),
        )
        coupling = jnp.ones((2, 3, 3)) * 0.5
        penalty = total_connectivity_penalty(coupling, config)
        assert float(penalty) > 0.0


class TestAreaCouplingSummary:
    def test_block_norms_shape(self):
        coupling = jnp.ones((2, 3, 3)) * 0.5
        area_labels = jnp.array([0, 0, 1])
        summary = get_area_coupling_summary(coupling, area_labels)
        assert summary["block_norms"].shape == (2, 2, 2)

    def test_within_and_cross_area(self):
        # Cross-area coupling only
        coupling = jnp.zeros((1, 3, 3))
        coupling = coupling.at[0, 0, 2].set(1.0)  # osc 0 (area 0) → osc 2 (area 1)
        area_labels = jnp.array([0, 0, 1])
        summary = get_area_coupling_summary(coupling, area_labels)
        assert float(summary["cross_area_norm"][0]) > 0
        # Within-area should be ~0 (only diagonal excluded)
        assert float(summary["within_area_norm"][0]) < 0.01

    def test_summary_finite(self):
        coupling = jnp.ones((2, 4, 4)) * 0.3
        area_labels = jnp.array([0, 0, 1, 1])
        summary = get_area_coupling_summary(coupling, area_labels)
        assert jnp.all(jnp.isfinite(summary["block_norms"]))
        assert jnp.all(jnp.isfinite(summary["within_area_norm"]))
        assert jnp.all(jnp.isfinite(summary["cross_area_norm"]))


class TestRegularizedSGDIntegration:
    """End-to-end: sparse DIM data → fit with penalty → verify sparsity."""

    def test_edge_penalty_reduces_false_edges(self):
        """Edge L1 penalty should shrink false coupling entries."""
        from state_space_practice.oscillator_models import DirectedInfluenceModel
        from state_space_practice.simulate.scenarios import simulate_dim_scenario

        scenario = simulate_dim_scenario(n_time=300, seed=42)
        p = scenario["params"]

        # Baseline: no penalty
        model_base = DirectedInfluenceModel(
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
        key = jax.random.PRNGKey(0)
        model_base.fit_sgd(scenario["obs"], key=key, num_steps=40)

        # Regularized: edge L1
        model_reg = DirectedInfluenceModel(
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
        config = OscillatorPenaltyConfig(edge_l1=0.5)
        model_reg.fit_sgd(
            scenario["obs"], key=key, num_steps=40,
            connectivity_penalty=config,
        )

        # Regularized should have smaller total coupling
        norm_base = float(jnp.sum(jnp.abs(model_base.coupling_strength)))
        norm_reg = float(jnp.sum(jnp.abs(model_reg.coupling_strength)))
        assert norm_reg < norm_base

    def test_area_penalty_with_summary(self):
        """Area group penalty + summary helper work end-to-end."""
        from state_space_practice.oscillator_models import DirectedInfluenceModel
        from state_space_practice.simulate.scenarios import simulate_dim_scenario

        scenario = simulate_dim_scenario(n_time=200, seed=99)
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
        area_labels = jnp.array([0, 1])  # 2 oscillators, 2 areas
        config = OscillatorPenaltyConfig(
            area_group_l2=1.0, area_labels=area_labels,
        )
        key = jax.random.PRNGKey(0)
        lls = model.fit_sgd(
            scenario["obs"], key=key, num_steps=30,
            connectivity_penalty=config,
        )

        # Get summary
        coupling_t = jnp.moveaxis(model.coupling_strength, -1, 0)
        summary = get_area_coupling_summary(coupling_t, area_labels)

        assert summary["block_norms"].shape == (2, 2, 2)
        assert jnp.all(jnp.isfinite(summary["block_norms"]))
        assert len(lls) > 1

    def test_area_penalty_reduces_cross_area_coupling(self):
        """Area group penalty should reduce cross-area coupling more than within."""
        from state_space_practice.oscillator_models import DirectedInfluenceModel
        from state_space_practice.simulate.scenarios import simulate_dim_scenario

        scenario = simulate_dim_scenario(n_time=300, seed=42)
        p = scenario["params"]
        area_labels = jnp.array([0, 1])  # each oscillator in its own area

        # Baseline: no penalty
        model_base = DirectedInfluenceModel(
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
        key = jax.random.PRNGKey(0)
        model_base.fit_sgd(scenario["obs"], key=key, num_steps=40)
        c_base = jnp.moveaxis(model_base.coupling_strength, -1, 0)
        summary_base = get_area_coupling_summary(c_base, area_labels)

        # Regularized: area group penalty on cross-area only
        model_reg = DirectedInfluenceModel(
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
        config = OscillatorPenaltyConfig(
            area_group_l2=2.0, area_labels=area_labels,
        )
        model_reg.fit_sgd(
            scenario["obs"], key=key, num_steps=40,
            connectivity_penalty=config,
        )
        c_reg = jnp.moveaxis(model_reg.coupling_strength, -1, 0)
        summary_reg = get_area_coupling_summary(c_reg, area_labels)

        # Cross-area coupling should be reduced by the penalty
        cross_base = float(jnp.sum(summary_base["cross_area_norm"]))
        cross_reg = float(jnp.sum(summary_reg["cross_area_norm"]))
        assert cross_reg < cross_base, (
            f"Regularized cross-area ({cross_reg:.4f}) should be < "
            f"baseline ({cross_base:.4f})"
        )
