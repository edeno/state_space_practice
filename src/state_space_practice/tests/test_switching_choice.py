# ruff: noqa: E402
"""Tests for the switching_choice module."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.switching_choice import (
    _softmax_predict_and_update,
    _softmax_update_per_state_pair,
)


class TestSoftmaxPredictAndUpdate:
    """Tests for the single-pair predict + softmax update."""

    def test_output_shapes(self):
        k_free = 2  # K=3 options
        mean = jnp.zeros(k_free)
        cov = jnp.eye(k_free)
        A = 0.95 * jnp.eye(k_free)
        Q = 0.01 * jnp.eye(k_free)
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        post_mean, post_cov, ll = _softmax_predict_and_update(
            mean, cov, jnp.int32(0), A, Q, 3, 1.0, B, u, obs_offset,
        )
        assert post_mean.shape == (k_free,)
        assert post_cov.shape == (k_free, k_free)
        assert ll.shape == ()

    def test_log_likelihood_finite(self):
        k_free = 2
        mean = jnp.array([0.5, -0.3])
        cov = jnp.eye(k_free) * 0.5
        A = 0.9 * jnp.eye(k_free)
        Q = 0.01 * jnp.eye(k_free)
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        _, _, ll = _softmax_predict_and_update(
            mean, cov, jnp.int32(1), A, Q, 3, 2.0, B, u, obs_offset,
        )
        assert jnp.isfinite(ll)

    def test_covariance_psd(self):
        k_free = 2
        mean = jnp.zeros(k_free)
        cov = jnp.eye(k_free)
        A = 0.95 * jnp.eye(k_free)
        Q = 0.01 * jnp.eye(k_free)
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        _, post_cov, _ = _softmax_predict_and_update(
            mean, cov, jnp.int32(0), A, Q, 3, 1.0, B, u, obs_offset,
        )
        eigvals = jnp.linalg.eigvalsh(post_cov)
        assert jnp.all(eigvals > -1e-10)


class TestSoftmaxUpdatePerStatePair:
    """Tests for the double-vmapped per-state-pair update."""

    def test_output_shapes(self):
        S = 2
        k_free = 2
        mean = jnp.zeros((k_free, S))
        cov = jnp.stack([jnp.eye(k_free)] * S, axis=-1)
        A = jnp.stack([0.95 * jnp.eye(k_free)] * S, axis=-1)
        Q = jnp.stack([0.01 * jnp.eye(k_free)] * S, axis=-1)
        betas = jnp.array([1.0, 3.0])
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        pair_mean, pair_cov, pair_ll = _softmax_update_per_state_pair(
            mean, cov, jnp.int32(0), A, Q, 3, betas, B, u, obs_offset,
        )
        assert pair_mean.shape == (k_free, S, S)
        assert pair_cov.shape == (k_free, k_free, S, S)
        assert pair_ll.shape == (S, S)

    def test_single_state_matches_softmax_update(self):
        """S=1 should match _softmax_predict_and_update directly."""
        k_free = 2
        mean = jnp.zeros((k_free, 1))
        cov = jnp.eye(k_free)[:, :, None]
        A = (0.95 * jnp.eye(k_free))[:, :, None]
        Q = (0.01 * jnp.eye(k_free))[:, :, None]
        betas = jnp.array([2.0])
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        pair_mean, pair_cov, pair_ll = _softmax_update_per_state_pair(
            mean, cov, jnp.int32(1), A, Q, 3, betas, B, u, obs_offset,
        )

        ref_mean, ref_cov, ref_ll = _softmax_predict_and_update(
            mean[:, 0], cov[:, :, 0], jnp.int32(1),
            A[:, :, 0], Q[:, :, 0], 3, 2.0, B, u, obs_offset,
        )

        np.testing.assert_allclose(pair_mean[:, 0, 0], ref_mean, atol=1e-10)
        np.testing.assert_allclose(pair_cov[:, :, 0, 0], ref_cov, atol=1e-10)
        np.testing.assert_allclose(float(pair_ll[0, 0]), float(ref_ll), atol=1e-10)

    def test_different_betas_give_different_posteriors(self):
        S = 2
        k_free = 2
        mean = jnp.zeros((k_free, S))
        cov = jnp.stack([jnp.eye(k_free)] * S, axis=-1)
        A = jnp.stack([0.95 * jnp.eye(k_free)] * S, axis=-1)
        Q = jnp.stack([0.01 * jnp.eye(k_free)] * S, axis=-1)
        betas = jnp.array([0.5, 5.0])  # very different
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        pair_mean, _, _ = _softmax_update_per_state_pair(
            mean, cov, jnp.int32(0), A, Q, 3, betas, B, u, obs_offset,
        )
        # Posteriors should differ across next-state axis
        assert not jnp.allclose(pair_mean[:, 0, 0], pair_mean[:, 0, 1])

    def test_all_log_likelihoods_finite(self):
        S = 2
        k_free = 2
        mean = jnp.zeros((k_free, S))
        cov = jnp.stack([jnp.eye(k_free)] * S, axis=-1)
        A = jnp.stack([0.95 * jnp.eye(k_free)] * S, axis=-1)
        Q = jnp.stack([0.01 * jnp.eye(k_free)] * S, axis=-1)
        betas = jnp.array([1.0, 3.0])
        B = jnp.zeros((k_free, 1))
        u = jnp.zeros(1)
        obs_offset = jnp.zeros(3)

        _, _, pair_ll = _softmax_update_per_state_pair(
            mean, cov, jnp.int32(0), A, Q, 3, betas, B, u, obs_offset,
        )
        assert jnp.all(jnp.isfinite(pair_ll))
