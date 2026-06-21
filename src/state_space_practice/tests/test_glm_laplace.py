"""Tests for the family-generic GLM Laplace update.

Two things are pinned here:
1. Parity: ``glm_laplace_update`` with ``poisson_family`` reproduces the legacy
   ``_point_process_laplace_update`` bit-for-bit, so generalizing the update did
   not change the Poisson path (used by PlaceFieldModel / PositionDecoder).
2. The Bernoulli-logit family's score and Fisher information match autodiff of
   its log-likelihood, and the update improves the log-posterior.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.point_process_kalman import (
    BERNOULLI_LOGIT_FAMILY,
    _point_process_laplace_update,
    glm_laplace_update,
    poisson_family,
)

# Fixed, deterministic setup (no RNG): 3 latent dims, 4 observations.
_MEAN = jnp.array([0.2, -0.1, 0.3])
_L = jnp.array([[0.7, 0.0, 0.0], [0.1, 0.6, 0.0], [-0.05, 0.1, 0.5]])
_COV = _L @ _L.T  # PSD by construction
_C = jnp.array([[0.5, -0.2, 0.1], [0.3, 0.4, -0.1], [-0.2, 0.1, 0.5], [0.1, 0.2, 0.3]])
_DT = 0.1


def _eta(x):
    return _C @ x


class TestPoissonParity:
    """glm_laplace_update(poisson_family) == legacy _point_process_laplace_update."""

    @pytest.mark.parametrize("max_newton_iter", [1, 3])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_matches_legacy(self, max_newton_iter, normalize):
        spikes = jnp.array([0.0, 1.0, 2.0, 0.0])
        legacy = _point_process_laplace_update(
            _MEAN,
            _COV,
            spikes,
            _DT,
            _eta,
            include_laplace_normalization=normalize,
            max_newton_iter=max_newton_iter,
        )
        new = glm_laplace_update(
            _MEAN,
            _COV,
            spikes,
            _eta,
            poisson_family(_DT),
            include_laplace_normalization=normalize,
            max_newton_iter=max_newton_iter,
        )
        for legacy_arr, new_arr, name in zip(legacy, new, ("mean", "cov", "ll")):
            np.testing.assert_allclose(
                np.asarray(new_arr),
                np.asarray(legacy_arr),
                atol=1e-10,
                rtol=1e-10,
                err_msg=f"Poisson parity mismatch in {name}",
            )


class TestBernoulliFamilyMath:
    def _bernoulli_loglik(self, x, y):
        eta = _eta(x)
        return jnp.sum(y * eta - jax.nn.softplus(eta))

    def test_score_matches_autodiff(self):
        """Family score J' (y - mu) equals grad of the Bernoulli log-likelihood."""
        x = jnp.array([0.4, -0.3, 0.2])
        y = jnp.array([1.0, 0.0, 1.0, 0.0])
        analytic = _C.T @ (y - jax.nn.sigmoid(_eta(x)))
        autodiff = jax.grad(self._bernoulli_loglik)(x, y)
        np.testing.assert_allclose(
            np.asarray(analytic), np.asarray(autodiff), atol=1e-10
        )

    def test_fisher_matches_negative_hessian(self):
        """For the canonical logit link, J' diag(mu(1-mu)) J == -Hessian(loglik)."""
        x = jnp.array([0.4, -0.3, 0.2])
        y = jnp.array([1.0, 0.0, 1.0, 0.0])
        mu = jax.nn.sigmoid(_eta(x))
        weight = BERNOULLI_LOGIT_FAMILY.fisher_weight(_eta(x), mu)
        fisher = _C.T @ (weight[:, None] * _C)
        neg_hessian = -jax.hessian(self._bernoulli_loglik)(x, y)
        np.testing.assert_allclose(
            np.asarray(fisher), np.asarray(neg_hessian), atol=1e-10
        )

    def test_fisher_is_psd(self):
        """The Fisher weight is nonnegative, so J' diag(w) J is PSD for any eta."""
        for x in (jnp.array([3.0, -4.0, 2.0]), jnp.array([-8.0, 0.0, 9.0])):
            mu = BERNOULLI_LOGIT_FAMILY.mean(_eta(x))
            weight = BERNOULLI_LOGIT_FAMILY.fisher_weight(_eta(x), mu)
            assert float(weight.min()) >= 0.0
            fisher = _C.T @ (weight[:, None] * _C)
            assert float(jnp.linalg.eigvalsh(fisher).min()) >= -1e-10


class TestBernoulliUpdate:
    def test_update_increases_log_posterior(self):
        """A Fisher step from the prior mean increases the (concave) log-posterior."""
        mean = jnp.zeros(3)
        cov = 100.0 * jnp.eye(3)  # weak prior, so the likelihood dominates
        y = jnp.array([1.0, 1.0, 0.0, 0.0])
        prior_precision = jnp.linalg.inv(cov)

        def log_posterior(x):
            eta = _eta(x)
            log_lik = jnp.sum(y * eta - jax.nn.softplus(eta))
            delta = x - mean
            return log_lik - 0.5 * delta @ (prior_precision @ delta)

        post_mean, post_cov, _ = glm_laplace_update(
            mean, cov, y, _eta, BERNOULLI_LOGIT_FAMILY
        )
        assert float(log_posterior(post_mean)) > float(log_posterior(mean))  # improved
        # posterior covariance is finite and PSD
        assert np.all(np.isfinite(np.asarray(post_cov)))
        assert float(jnp.linalg.eigvalsh(post_cov).min()) >= -1e-10

    def test_posterior_sharper_than_prior(self):
        """Observing data reduces posterior uncertainty (smaller covariance trace)."""
        mean = jnp.zeros(3)
        cov = 10.0 * jnp.eye(3)
        y = jnp.array([1.0, 0.0, 1.0, 1.0])
        _, post_cov, _ = glm_laplace_update(mean, cov, y, _eta, BERNOULLI_LOGIT_FAMILY)
        assert float(jnp.trace(post_cov)) < float(jnp.trace(cov))

    def test_line_search_converges_and_is_monotone(self):
        """Iterating Fisher steps is non-decreasing in log-posterior and converges.

        Exercises the iterative line-search branch with the Bernoulli family (the
        config fit_coupling_ekf uses); the fast suite otherwise only hits it for
        Poisson via the parity test.
        """
        mean = jnp.zeros(3)
        cov = 25.0 * jnp.eye(3)
        y = jnp.array([1.0, 0.0, 1.0, 0.0])
        prior_precision = jnp.linalg.inv(cov)

        def log_posterior(b):
            eta = _eta(b)
            log_lik = jnp.sum(y * eta - jax.nn.softplus(eta))
            delta = b - mean
            return log_lik - 0.5 * delta @ (prior_precision @ delta)

        results = {}
        for n_iter in (1, 3, 10):
            m, _, _ = glm_laplace_update(
                mean, cov, y, _eta, BERNOULLI_LOGIT_FAMILY, max_newton_iter=n_iter
            )
            results[n_iter] = m
        lp = {k: float(log_posterior(v)) for k, v in results.items()}
        assert lp[1] > float(log_posterior(mean))  # guard: the update did something
        assert lp[1] <= lp[3] + 1e-9 <= lp[10] + 1e-9  # non-decreasing in iterations
        # converged by iter 3
        np.testing.assert_allclose(
            np.asarray(results[3]), np.asarray(results[10]), atol=1e-4
        )

    def test_laplace_cov_matches_inverse_hessian(self):
        """The Laplace posterior cov equals inv(-Hessian of the log-posterior) at the
        mode (variance calibration, since the Wald test divides by these variances)."""
        mean = jnp.zeros(3)
        cov = 25.0 * jnp.eye(3)
        y = jnp.array([1.0, 0.0, 1.0, 0.0])
        prior_precision = jnp.linalg.inv(cov)

        def neg_log_posterior(b):
            eta = _eta(b)
            log_lik = jnp.sum(y * eta - jax.nn.softplus(eta))
            delta = b - mean
            return -(log_lik - 0.5 * delta @ (prior_precision @ delta))

        post_mean, post_cov, _ = glm_laplace_update(
            mean, cov, y, _eta, BERNOULLI_LOGIT_FAMILY, max_newton_iter=25
        )
        hessian = jax.hessian(neg_log_posterior)(post_mean)
        np.testing.assert_allclose(
            np.asarray(post_cov),
            np.asarray(jnp.linalg.inv(hessian)),
            rtol=1e-5,
            atol=1e-7,
        )


class TestFamilyConsistency:
    @pytest.mark.parametrize(
        "family",
        [poisson_family(0.1), BERNOULLI_LOGIT_FAMILY],
        ids=["poisson", "bernoulli"],
    )
    def test_fisher_weight_is_mean_derivative(self, family):
        """For a canonical link, fisher_weight(eta) == d mean / d eta elementwise.

        An executable form of the GLMFamily consistency invariant — guards any
        future family against a mismatched mean/fisher_weight pair.
        """
        eta = jnp.array([-1.5, -0.3, 0.0, 0.8, 2.0])
        mu = family.mean(eta)
        dmean_deta = jnp.diag(jax.jacfwd(family.mean)(eta))  # mean is elementwise
        weight = family.fisher_weight(eta, mu)
        np.testing.assert_allclose(
            np.asarray(weight), np.asarray(dmean_deta), atol=1e-10
        )
