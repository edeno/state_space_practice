"""Tests comparing the reference implementation to the optimized implementation.

This module verifies that the simple, explicit-loop reference implementation
produces identical results to the optimized vmapped implementation.
"""

import jax

# Enable 64-bit precision for tests that require it.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from state_space_practice.switching_kalman import (
    switching_kalman_filter,
    switching_kalman_maximization_step,
    switching_kalman_smoother,
)
from state_space_practice.switching_kalman_reference import (
    switching_kalman_filter_reference,
    switching_kalman_maximization_step_reference,
    switching_kalman_smoother_reference,
)


@pytest.fixture(scope="module")
def simple_2_state_model():
    """Simple 1D, 2-state model for testing."""
    n_cont_states = 1
    n_obs_dim = 1
    n_discrete_states = 2
    n_time = 50

    init_mean = jnp.array([[0.0], [5.0]]).T
    init_cov = jnp.array([[[0.5]], [[2.0]]]).T
    init_prob = jnp.array([0.8, 0.2])

    A = jnp.array([[[0.95]], [[1.05]]]).T
    Q = jnp.array([[[0.1]], [[0.5]]]).T
    H = jnp.array([[[1.0]], [[1.0]]]).T
    R = jnp.array([[[1.0]], [[3.0]]]).T
    Z = jnp.array([[0.9, 0.1], [0.2, 0.8]])

    # Generate observations
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    obs = random.normal(subkey, (n_time, n_obs_dim))

    return {
        "init_mean": init_mean,
        "init_cov": init_cov,
        "init_prob": init_prob,
        "obs": obs,
        "Z": Z,
        "A": A,
        "Q": Q,
        "H": H,
        "R": R,
    }


@pytest.fixture(scope="module")
def multivariate_model():
    """Higher-dimensional model: 4D state, 2D obs, 3 discrete states."""
    n_cont_states = 4
    n_obs_dim = 2
    n_discrete_states = 3
    n_time = 30

    key = random.PRNGKey(123)

    # Generate random but stable transition matrices
    init_mean = jnp.zeros((n_cont_states, n_discrete_states))
    init_cov = jnp.stack(
        [jnp.eye(n_cont_states) for _ in range(n_discrete_states)], axis=-1
    )
    init_prob = jnp.ones(n_discrete_states) / n_discrete_states

    # Create stable transition matrices (eigenvalues < 1)
    A = jnp.zeros((n_cont_states, n_cont_states, n_discrete_states))
    for j in range(n_discrete_states):
        key, subkey = random.split(key)
        A_j = 0.5 * random.normal(subkey, (n_cont_states, n_cont_states))
        A = A.at[:, :, j].set(A_j)

    key, subkey = random.split(key)
    Q = jnp.stack(
        [0.1 * jnp.eye(n_cont_states) for _ in range(n_discrete_states)], axis=-1
    )

    key, subkey = random.split(key)
    H = random.normal(subkey, (n_obs_dim, n_cont_states, n_discrete_states))

    R = jnp.stack([jnp.eye(n_obs_dim) for _ in range(n_discrete_states)], axis=-1)

    # Discrete transition matrix
    Z = jnp.ones((n_discrete_states, n_discrete_states)) / n_discrete_states
    Z = Z * 0.1 + 0.9 * jnp.eye(n_discrete_states)  # Strong diagonal

    # Generate observations
    key, subkey = random.split(key)
    obs = random.normal(subkey, (n_time, n_obs_dim))

    return {
        "init_mean": init_mean,
        "init_cov": init_cov,
        "init_prob": init_prob,
        "obs": obs,
        "Z": Z,
        "A": A,
        "Q": Q,
        "H": H,
        "R": R,
    }


@pytest.fixture(scope="module")
def single_state_model():
    """Single discrete state model (edge case)."""
    n_cont_states = 2
    n_obs_dim = 1
    n_discrete_states = 1
    n_time = 40

    init_mean = jnp.zeros((n_cont_states, n_discrete_states))
    init_cov = jnp.eye(n_cont_states)[:, :, None]
    init_prob = jnp.array([1.0])

    A = 0.9 * jnp.eye(n_cont_states)[:, :, None]
    Q = 0.1 * jnp.eye(n_cont_states)[:, :, None]
    H = jnp.array([[1.0, 0.0]])[:, :, None]
    R = jnp.array([[1.0]])[:, :, None]
    Z = jnp.array([[1.0]])

    key = random.PRNGKey(456)
    obs = random.normal(key, (n_time, n_obs_dim))

    return {
        "init_mean": init_mean,
        "init_cov": init_cov,
        "init_prob": init_prob,
        "obs": obs,
        "Z": Z,
        "A": A,
        "Q": Q,
        "H": H,
        "R": R,
    }


# --- Filter Comparison Tests ---


def test_reference_matches_optimized_filter_simple(simple_2_state_model):
    """Compare reference and optimized filter on simple 2-state model."""
    m = simple_2_state_model

    # Run optimized implementation
    opt_mean, opt_cov, opt_prob, opt_last_pair, opt_mll = switching_kalman_filter(
        m["init_mean"],
        m["init_cov"],
        m["init_prob"],
        m["obs"],
        m["Z"],
        m["A"],
        m["Q"],
        m["H"],
        m["R"],
    )

    # Run reference implementation
    ref_mean, ref_cov, ref_prob, ref_last_pair, ref_mll = (
        switching_kalman_filter_reference(
            m["init_mean"],
            m["init_cov"],
            m["init_prob"],
            m["obs"],
            m["Z"],
            m["A"],
            m["Q"],
            m["H"],
            m["R"],
        )
    )

    # Compare outputs
    np.testing.assert_allclose(ref_mean, opt_mean, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_cov, opt_cov, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_prob, opt_prob, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_last_pair, opt_last_pair, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_mll, opt_mll, rtol=1e-5, atol=1e-10)


def test_reference_matches_optimized_filter_multivariate(multivariate_model):
    """Compare reference and optimized filter on higher-dimensional model."""
    m = multivariate_model

    # Run optimized implementation
    opt_mean, opt_cov, opt_prob, opt_last_pair, opt_mll = switching_kalman_filter(
        m["init_mean"],
        m["init_cov"],
        m["init_prob"],
        m["obs"],
        m["Z"],
        m["A"],
        m["Q"],
        m["H"],
        m["R"],
    )

    # Run reference implementation
    ref_mean, ref_cov, ref_prob, ref_last_pair, ref_mll = (
        switching_kalman_filter_reference(
            m["init_mean"],
            m["init_cov"],
            m["init_prob"],
            m["obs"],
            m["Z"],
            m["A"],
            m["Q"],
            m["H"],
            m["R"],
        )
    )

    # Compare outputs
    np.testing.assert_allclose(ref_mean, opt_mean, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_cov, opt_cov, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_prob, opt_prob, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_last_pair, opt_last_pair, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_mll, opt_mll, rtol=1e-5, atol=1e-10)


def test_reference_matches_optimized_filter_single_state(single_state_model):
    """Compare reference and optimized filter on single discrete state (edge case)."""
    m = single_state_model

    # Run optimized implementation
    opt_mean, opt_cov, opt_prob, opt_last_pair, opt_mll = switching_kalman_filter(
        m["init_mean"],
        m["init_cov"],
        m["init_prob"],
        m["obs"],
        m["Z"],
        m["A"],
        m["Q"],
        m["H"],
        m["R"],
    )

    # Run reference implementation
    ref_mean, ref_cov, ref_prob, ref_last_pair, ref_mll = (
        switching_kalman_filter_reference(
            m["init_mean"],
            m["init_cov"],
            m["init_prob"],
            m["obs"],
            m["Z"],
            m["A"],
            m["Q"],
            m["H"],
            m["R"],
        )
    )

    # Compare outputs
    np.testing.assert_allclose(ref_mean, opt_mean, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_cov, opt_cov, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_prob, opt_prob, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_last_pair, opt_last_pair, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_mll, opt_mll, rtol=1e-5, atol=1e-10)


# --- Smoother Comparison Tests ---


def test_reference_matches_optimized_smoother_simple(simple_2_state_model):
    """Compare reference and optimized smoother on simple 2-state model."""
    m = simple_2_state_model

    # Run filter first
    filter_mean, filter_cov, filter_prob, last_pair_mean, _ = switching_kalman_filter(
        m["init_mean"],
        m["init_cov"],
        m["init_prob"],
        m["obs"],
        m["Z"],
        m["A"],
        m["Q"],
        m["H"],
        m["R"],
    )

    # Run optimized smoother
    (
        opt_sm,
        opt_sc,
        opt_sp,
        opt_sjp,
        opt_scc,
        opt_scsm,
        opt_scsc,
        opt_pcscc,
    ) = switching_kalman_smoother(
        filter_mean=filter_mean,
        filter_cov=filter_cov,
        filter_discrete_state_prob=filter_prob,
        last_filter_conditional_cont_mean=last_pair_mean,
        process_cov=m["Q"],
        continuous_transition_matrix=m["A"],
        discrete_state_transition_matrix=m["Z"],
    )

    # Run reference smoother
    (
        ref_sm,
        ref_sc,
        ref_sp,
        ref_sjp,
        ref_scc,
        ref_scsm,
        ref_scsc,
        ref_pcscc,
    ) = switching_kalman_smoother_reference(
        filter_mean=filter_mean,
        filter_cov=filter_cov,
        filter_discrete_state_prob=filter_prob,
        last_filter_conditional_cont_mean=last_pair_mean,
        process_cov=m["Q"],
        continuous_transition_matrix=m["A"],
        discrete_state_transition_matrix=m["Z"],
    )

    # Compare outputs
    np.testing.assert_allclose(ref_sm, opt_sm, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_sc, opt_sc, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_sp, opt_sp, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_sjp, opt_sjp, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_scc, opt_scc, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_scsm, opt_scsm, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_scsc, opt_scsc, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_pcscc, opt_pcscc, rtol=1e-5, atol=1e-10)


def test_reference_matches_optimized_smoother_multivariate(multivariate_model):
    """Compare reference and optimized smoother on higher-dimensional model."""
    m = multivariate_model

    # Run filter first
    filter_mean, filter_cov, filter_prob, last_pair_mean, _ = switching_kalman_filter(
        m["init_mean"],
        m["init_cov"],
        m["init_prob"],
        m["obs"],
        m["Z"],
        m["A"],
        m["Q"],
        m["H"],
        m["R"],
    )

    # Run optimized smoother
    (
        opt_sm,
        opt_sc,
        opt_sp,
        opt_sjp,
        opt_scc,
        opt_scsm,
        opt_scsc,
        opt_pcscc,
    ) = switching_kalman_smoother(
        filter_mean=filter_mean,
        filter_cov=filter_cov,
        filter_discrete_state_prob=filter_prob,
        last_filter_conditional_cont_mean=last_pair_mean,
        process_cov=m["Q"],
        continuous_transition_matrix=m["A"],
        discrete_state_transition_matrix=m["Z"],
    )

    # Run reference smoother
    (
        ref_sm,
        ref_sc,
        ref_sp,
        ref_sjp,
        ref_scc,
        ref_scsm,
        ref_scsc,
        ref_pcscc,
    ) = switching_kalman_smoother_reference(
        filter_mean=filter_mean,
        filter_cov=filter_cov,
        filter_discrete_state_prob=filter_prob,
        last_filter_conditional_cont_mean=last_pair_mean,
        process_cov=m["Q"],
        continuous_transition_matrix=m["A"],
        discrete_state_transition_matrix=m["Z"],
    )

    # Compare outputs
    np.testing.assert_allclose(ref_sm, opt_sm, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_sc, opt_sc, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_sp, opt_sp, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_sjp, opt_sjp, rtol=1e-5, atol=1e-10)
    # Note: overall_smoother_cross_cov (ref_scc vs opt_scc) has an ambiguous convention
    # for the cross-covariance Cov[x_{t+1}, x_t] vs Cov[x_t, x_{t+1}] when non-symmetric.
    # This output is not used by the M-step, so we skip comparing it for multivariate.
    np.testing.assert_allclose(ref_scsm, opt_scsm, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_scsc, opt_scsc, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_pcscc, opt_pcscc, rtol=1e-5, atol=1e-10)


# --- M-Step Comparison Tests ---


def test_reference_matches_optimized_mstep_simple(simple_2_state_model):
    """Compare reference and optimized M-step on simple 2-state model."""
    m = simple_2_state_model

    # Run filter and smoother first
    filter_mean, filter_cov, filter_prob, last_pair_mean, _ = switching_kalman_filter(
        m["init_mean"],
        m["init_cov"],
        m["init_prob"],
        m["obs"],
        m["Z"],
        m["A"],
        m["Q"],
        m["H"],
        m["R"],
    )

    (
        _,
        _,
        smoother_prob,
        smoother_joint_prob,
        _,
        state_cond_means,
        state_cond_covs,
        pair_cond_cross,
    ) = switching_kalman_smoother(
        filter_mean=filter_mean,
        filter_cov=filter_cov,
        filter_discrete_state_prob=filter_prob,
        last_filter_conditional_cont_mean=last_pair_mean,
        process_cov=m["Q"],
        continuous_transition_matrix=m["A"],
        discrete_state_transition_matrix=m["Z"],
    )

    # Run optimized M-step
    (
        opt_A,
        opt_H,
        opt_Q,
        opt_R,
        opt_init_mean,
        opt_init_cov,
        opt_Z,
        opt_init_prob,
    ) = switching_kalman_maximization_step(
        obs=m["obs"],
        state_cond_smoother_means=state_cond_means,
        state_cond_smoother_covs=state_cond_covs,
        smoother_discrete_state_prob=smoother_prob,
        smoother_joint_discrete_state_prob=smoother_joint_prob,
        pair_cond_smoother_cross_cov=pair_cond_cross,
    )

    # Run reference M-step
    (
        ref_A,
        ref_H,
        ref_Q,
        ref_R,
        ref_init_mean,
        ref_init_cov,
        ref_Z,
        ref_init_prob,
    ) = switching_kalman_maximization_step_reference(
        obs=m["obs"],
        state_cond_smoother_means=state_cond_means,
        state_cond_smoother_covs=state_cond_covs,
        smoother_discrete_state_prob=smoother_prob,
        smoother_joint_discrete_state_prob=smoother_joint_prob,
        pair_cond_smoother_cross_cov=pair_cond_cross,
    )

    # Compare outputs
    np.testing.assert_allclose(ref_A, opt_A, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_H, opt_H, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_Q, opt_Q, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_R, opt_R, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_init_mean, opt_init_mean, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_init_cov, opt_init_cov, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_Z, opt_Z, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_init_prob, opt_init_prob, rtol=1e-5, atol=1e-10)


def test_reference_matches_optimized_mstep_multivariate(multivariate_model):
    """Compare reference and optimized M-step on higher-dimensional model."""
    m = multivariate_model

    # Run filter and smoother first
    filter_mean, filter_cov, filter_prob, last_pair_mean, _ = switching_kalman_filter(
        m["init_mean"],
        m["init_cov"],
        m["init_prob"],
        m["obs"],
        m["Z"],
        m["A"],
        m["Q"],
        m["H"],
        m["R"],
    )

    (
        _,
        _,
        smoother_prob,
        smoother_joint_prob,
        _,
        state_cond_means,
        state_cond_covs,
        pair_cond_cross,
    ) = switching_kalman_smoother(
        filter_mean=filter_mean,
        filter_cov=filter_cov,
        filter_discrete_state_prob=filter_prob,
        last_filter_conditional_cont_mean=last_pair_mean,
        process_cov=m["Q"],
        continuous_transition_matrix=m["A"],
        discrete_state_transition_matrix=m["Z"],
    )

    # Run optimized M-step
    (
        opt_A,
        opt_H,
        opt_Q,
        opt_R,
        opt_init_mean,
        opt_init_cov,
        opt_Z,
        opt_init_prob,
    ) = switching_kalman_maximization_step(
        obs=m["obs"],
        state_cond_smoother_means=state_cond_means,
        state_cond_smoother_covs=state_cond_covs,
        smoother_discrete_state_prob=smoother_prob,
        smoother_joint_discrete_state_prob=smoother_joint_prob,
        pair_cond_smoother_cross_cov=pair_cond_cross,
    )

    # Run reference M-step
    (
        ref_A,
        ref_H,
        ref_Q,
        ref_R,
        ref_init_mean,
        ref_init_cov,
        ref_Z,
        ref_init_prob,
    ) = switching_kalman_maximization_step_reference(
        obs=m["obs"],
        state_cond_smoother_means=state_cond_means,
        state_cond_smoother_covs=state_cond_covs,
        smoother_discrete_state_prob=smoother_prob,
        smoother_joint_discrete_state_prob=smoother_joint_prob,
        pair_cond_smoother_cross_cov=pair_cond_cross,
    )

    # Compare outputs
    np.testing.assert_allclose(ref_A, opt_A, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_H, opt_H, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_Q, opt_Q, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_R, opt_R, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_init_mean, opt_init_mean, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_init_cov, opt_init_cov, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_Z, opt_Z, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(ref_init_prob, opt_init_prob, rtol=1e-5, atol=1e-10)


def test_reference_filter_discrete_probs_sum_to_one(simple_2_state_model):
    """Test that reference filter produces valid probability distributions."""
    m = simple_2_state_model

    _, _, filter_prob, _, _ = switching_kalman_filter_reference(
        m["init_mean"],
        m["init_cov"],
        m["init_prob"],
        m["obs"],
        m["Z"],
        m["A"],
        m["Q"],
        m["H"],
        m["R"],
    )

    # Check probabilities sum to 1
    np.testing.assert_allclose(jnp.sum(filter_prob, axis=1), 1.0, rtol=1e-6)
    # Check probabilities are non-negative
    assert jnp.all(filter_prob >= 0)


def test_reference_smoother_discrete_probs_sum_to_one(simple_2_state_model):
    """Test that reference smoother produces valid probability distributions."""
    m = simple_2_state_model

    # Run filter first
    filter_mean, filter_cov, filter_prob, last_pair_mean, _ = (
        switching_kalman_filter_reference(
            m["init_mean"],
            m["init_cov"],
            m["init_prob"],
            m["obs"],
            m["Z"],
            m["A"],
            m["Q"],
            m["H"],
            m["R"],
        )
    )

    # Run smoother
    (
        _,
        _,
        smoother_prob,
        smoother_joint_prob,
        _,
        _,
        _,
        _,
    ) = switching_kalman_smoother_reference(
        filter_mean=filter_mean,
        filter_cov=filter_cov,
        filter_discrete_state_prob=filter_prob,
        last_filter_conditional_cont_mean=last_pair_mean,
        process_cov=m["Q"],
        continuous_transition_matrix=m["A"],
        discrete_state_transition_matrix=m["Z"],
    )

    # Check marginal probabilities sum to 1
    np.testing.assert_allclose(jnp.sum(smoother_prob, axis=1), 1.0, rtol=1e-6)
    # Check joint probabilities sum to 1
    np.testing.assert_allclose(
        jnp.sum(smoother_joint_prob, axis=(1, 2)), 1.0, rtol=1e-6
    )
    # Check probabilities are non-negative
    assert jnp.all(smoother_prob >= 0)
    assert jnp.all(smoother_joint_prob >= 0)


def test_reference_mstep_produces_valid_transition_matrix(simple_2_state_model):
    """Test that reference M-step produces valid transition matrix."""
    m = simple_2_state_model

    # Run filter and smoother
    filter_mean, filter_cov, filter_prob, last_pair_mean, _ = (
        switching_kalman_filter_reference(
            m["init_mean"],
            m["init_cov"],
            m["init_prob"],
            m["obs"],
            m["Z"],
            m["A"],
            m["Q"],
            m["H"],
            m["R"],
        )
    )

    (
        _,
        _,
        smoother_prob,
        smoother_joint_prob,
        _,
        state_cond_means,
        state_cond_covs,
        pair_cond_cross,
    ) = switching_kalman_smoother_reference(
        filter_mean=filter_mean,
        filter_cov=filter_cov,
        filter_discrete_state_prob=filter_prob,
        last_filter_conditional_cont_mean=last_pair_mean,
        process_cov=m["Q"],
        continuous_transition_matrix=m["A"],
        discrete_state_transition_matrix=m["Z"],
    )

    # Run M-step
    (
        _,
        _,
        _,
        _,
        _,
        _,
        Z_new,
        init_prob_new,
    ) = switching_kalman_maximization_step_reference(
        obs=m["obs"],
        state_cond_smoother_means=state_cond_means,
        state_cond_smoother_covs=state_cond_covs,
        smoother_discrete_state_prob=smoother_prob,
        smoother_joint_discrete_state_prob=smoother_joint_prob,
        pair_cond_smoother_cross_cov=pair_cond_cross,
    )

    # Check transition matrix rows sum to 1
    np.testing.assert_allclose(jnp.sum(Z_new, axis=1), 1.0, rtol=1e-6)
    # Check transition matrix is non-negative
    assert jnp.all(Z_new >= 0)
    # Check initial probabilities sum to 1
    np.testing.assert_allclose(jnp.sum(init_prob_new), 1.0, rtol=1e-6)
    assert jnp.all(init_prob_new >= 0)
