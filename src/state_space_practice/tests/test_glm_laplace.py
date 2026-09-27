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
from state_space_practice.utils import psd_cholesky, psd_solve, symmetrize

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


class TestCarriedLineSearch:
    """The iterative branch carries each accepted point's Fisher step and loss.

    ``_fisher_scoring_line_search`` computes the Fisher direction, posterior
    precision and negative log-posterior once per accepted point and reuses
    them as the next iteration's direction and loss reference (instead of
    recomputing them at the top of every iteration). Pin the result against
    a plain Python reference that recomputes everything at every iterate --
    the two must agree to roundoff, for both families, on problems where the
    backtracking actually shortens steps.
    """

    @staticmethod
    def _reference(
        one_step_mean,
        one_step_cov,
        y,
        eta_func,
        family,
        max_newton_iter,
        line_search_beta=0.5,
        diagonal_boost=0.0,
    ):
        """Recomputing Fisher scoring with the same 10-step Armijo backtracking."""
        identity = jnp.eye(one_step_mean.shape[0])
        prior_cho = psd_cholesky(one_step_cov, diagonal_boost=diagonal_boost)
        prior_precision = jax.scipy.linalg.cho_solve(prior_cho, identity)

        def neg_log_posterior(x):
            eta = eta_func(x)
            mu = family.mean(eta)
            delta = x - one_step_mean
            log_prior = -0.5 * delta @ (prior_precision @ delta)
            return -(family.loglik_plugin(y, eta, mu) + log_prior)

        def fisher_step(x):
            eta = eta_func(x)
            mu = family.mean(eta)
            jacobian = jax.jacfwd(eta_func)(x)
            gradient = jacobian.T @ (y - mu) - prior_precision @ (x - one_step_mean)
            weight = family.fisher_weight(eta, mu)
            post_prec = symmetrize(
                prior_precision + jacobian.T @ (weight[:, None] * jacobian)
            )
            delta = psd_solve(post_prec, gradient, diagonal_boost=diagonal_boost)
            return delta, post_prec, gradient

        x = one_step_mean
        n_shortened = n_rejected = 0
        for _ in range(max_newton_iter):
            delta, _, gradient = fisher_step(x)
            slope = float(gradient @ delta)
            loss = neg_log_posterior(x)
            alpha, improved = 1.0, False
            for _ in range(10):
                # Strict decrease plus the Armijo sufficient-decrease test.
                new_loss = neg_log_posterior(x + alpha * delta)
                improved = bool(
                    (new_loss < loss) & (new_loss <= loss - 1e-4 * alpha * slope)
                )
                if not improved:
                    alpha *= line_search_beta
            if improved:
                n_shortened += alpha < 1.0
                x = x + alpha * delta
            else:
                n_rejected += 1
        _, post_prec, _ = fisher_step(x)
        post_cho = psd_cholesky(post_prec, diagonal_boost=diagonal_boost)
        post_cov = symmetrize(jax.scipy.linalg.cho_solve(post_cho, identity))
        return x, post_cov, n_shortened, n_rejected

    @pytest.mark.parametrize("max_newton_iter", [2, 4])
    def test_poisson_matches_recomputing_reference(self, max_newton_iter):
        # Weak prior + large counts: the full Fisher step overshoots, so the
        # backtracking shortens it and the carried loss / direction matter.
        cov = 100.0 * jnp.eye(3)
        spikes = jnp.array([40.0, 0.0, 55.0, 3.0])
        ref_mean, ref_cov, n_shortened, _ = self._reference(
            _MEAN, cov, spikes, _eta, poisson_family(_DT), max_newton_iter
        )
        assert n_shortened > 0  # guard: the line search actually backtracked
        post_mean, post_cov, _ = _point_process_laplace_update(
            _MEAN, cov, spikes, _DT, _eta, max_newton_iter=max_newton_iter
        )
        np.testing.assert_allclose(
            np.asarray(post_mean), np.asarray(ref_mean), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(post_cov), np.asarray(ref_cov), rtol=1e-11, atol=1e-12
        )

    @pytest.mark.parametrize("max_newton_iter", [2, 4])
    def test_bernoulli_matches_recomputing_reference(self, max_newton_iter):
        cov = 400.0 * jnp.eye(3)
        y = jnp.array([1.0, 1.0, 0.0, 1.0])
        ref_mean, ref_cov, _, _ = self._reference(
            _MEAN, cov, y, _eta, BERNOULLI_LOGIT_FAMILY, max_newton_iter
        )
        assert float(jnp.max(jnp.abs(ref_mean - _MEAN))) > 1.0  # guard: moved
        post_mean, post_cov, _ = glm_laplace_update(
            _MEAN, cov, y, _eta, BERNOULLI_LOGIT_FAMILY, max_newton_iter=max_newton_iter
        )
        np.testing.assert_allclose(
            np.asarray(post_mean), np.asarray(ref_mean), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(post_cov), np.asarray(ref_cov), rtol=1e-11, atol=1e-12
        )

    # One neuron whose expected count saturates the ``max_log_count`` clip:
    # beyond the clip the Poisson term is flat in x while the Fisher step
    # still points further up (y exceeds the clipped count), so no step size
    # decreases the objective and every backtracking search is rejected.
    _C1 = jnp.array([[0.5, 0.3, 0.2]])

    @classmethod
    def _eta1(cls, x):
        return cls._C1 @ x

    def _check_rejected_steps(self, one_step_mean, max_log_count, expect_accepted):
        cov = 100.0 * jnp.eye(3)
        spikes = jnp.array([5.0])
        max_newton_iter = 4
        ref_mean, ref_cov, _, n_rejected = self._reference(
            one_step_mean,
            cov,
            spikes,
            self._eta1,
            poisson_family(_DT, max_log_count=max_log_count),
            max_newton_iter,
        )
        # guard: the rejected-step branch is exercised
        assert n_rejected == max_newton_iter - expect_accepted
        post_mean, post_cov, _ = _point_process_laplace_update(
            one_step_mean,
            cov,
            spikes,
            _DT,
            self._eta1,
            max_newton_iter=max_newton_iter,
            max_log_count=max_log_count,
        )
        np.testing.assert_allclose(
            np.asarray(post_mean), np.asarray(ref_mean), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(post_cov), np.asarray(ref_cov), rtol=1e-11, atol=1e-12
        )
        return post_mean

    def test_all_steps_rejected_keeps_the_start_point(self):
        """Starting inside the saturated region, every iteration is rejected:
        the point never moves (and so neither does the carried loss), and the
        precision is the one at the start point."""
        start = jnp.array([6.0, 0.0, 0.0])  # log count 0.70 > max_log_count 0
        post_mean = self._check_rejected_steps(
            start, max_log_count=0.0, expect_accepted=0
        )
        np.testing.assert_array_equal(np.asarray(post_mean), np.asarray(start))

    def test_rejections_after_an_accepted_step_keep_the_accepted_point(self):
        """The first (shortened) step is accepted and lands in the saturated
        region; the remaining iterations are rejected and must keep the
        accepted point, with the loss and direction carried from it."""
        start = jnp.array([-2.0, 0.0, 0.0])  # log count -3.3 < max_log_count -2
        post_mean = self._check_rejected_steps(
            start, max_log_count=-2.0, expect_accepted=1
        )
        assert float(jnp.max(jnp.abs(post_mean - start))) > 1.0  # guard: moved


class TestZeroNewtonIterations:
    """``max_newton_iter=0`` performs no measurement update: the prior is returned.

    ``PositionDecoder`` accepts ``max_newton_iter=0``. With no Fisher step the
    posterior must be the prior itself -- mean *and* covariance -- and the
    Laplace log-evidence collapses to the log-likelihood at the prior mean
    (the ``log|P_post| - log|P_prior|`` normalization term cancels). A
    covariance shrunk by the Fisher information at an unmoved mean would be
    an inconsistent posterior.
    """

    _Y_POISSON = jnp.array([0.0, 1.0, 2.0, 0.0])
    _Y_BERNOULLI = jnp.array([1.0, 0.0, 1.0, 1.0])

    def _assert_prior_returned(self, post_mean, post_cov, ll, expected_ll):
        np.testing.assert_array_equal(np.asarray(post_mean), np.asarray(_MEAN))
        # The covariance is the prior precision inverted back, each inversion
        # regularized by the scale-relative 1e-12 * |A_ii| Cholesky shift,
        # so it matches to ~1e-11 (the former absolute 1e-9 shift left ~1e-9).
        np.testing.assert_allclose(
            np.asarray(post_cov), np.asarray(_COV), rtol=1e-10, atol=1e-12
        )
        # Same regularization residue in 0.5 * (log|P_post| - log|P_prior|).
        np.testing.assert_allclose(
            float(ll), float(expected_ll), rtol=1e-10, atol=1e-10
        )

    def test_point_process_update_returns_prior(self):
        post_mean, post_cov, ll = _point_process_laplace_update(
            _MEAN, _COV, self._Y_POISSON, _DT, _eta, max_newton_iter=0
        )
        expected_ll = jnp.sum(
            jax.scipy.stats.poisson.logpmf(self._Y_POISSON, jnp.exp(_eta(_MEAN)) * _DT)
        )
        self._assert_prior_returned(post_mean, post_cov, ll, expected_ll)

    @pytest.mark.parametrize(
        "family, y",
        [(poisson_family(_DT), _Y_POISSON), (BERNOULLI_LOGIT_FAMILY, _Y_BERNOULLI)],
        ids=["poisson", "bernoulli"],
    )
    def test_glm_update_returns_prior(self, family, y):
        post_mean, post_cov, ll = glm_laplace_update(
            _MEAN, _COV, y, _eta, family, max_newton_iter=0
        )
        eta = _eta(_MEAN)
        expected_ll = family.loglik_normalized(y, eta, family.mean(eta))
        self._assert_prior_returned(post_mean, post_cov, ll, expected_ll)

    def test_one_iteration_does_update(self):
        """Guard: the same data does move the posterior with one Fisher step,
        so the zero-iteration assertions above are not vacuous."""
        post_mean, post_cov, _ = _point_process_laplace_update(
            _MEAN, _COV, self._Y_POISSON, _DT, _eta, max_newton_iter=1
        )
        assert float(jnp.max(jnp.abs(post_mean - _MEAN))) > 1e-3
        assert float(jnp.trace(post_cov)) < float(jnp.trace(_COV)) - 1e-3
