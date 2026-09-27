# ruff: noqa: E402
"""Exact oracles for the behavioural choice / learning filters.

Every model here approximates a non-Gaussian posterior (Laplace-EKF, GPB1
collapse) or claims to be exact (discrete HMM recursions). The oracles below
compute the *exact* answer independently:

- continuous latents of dimension 1-2 by quadrature on a dense grid
  (trapezoid rule; the integrands are smooth and Gaussian-tailed, so the
  quadrature error is far below every tolerance used here);
- discrete latents by brute-force enumeration of every state path.

For the approximate filters we pin the approximation error from above (with
headroom) *and* from below (the gap must be nonzero, so a test that silently
compares the approximation with itself fails). For the exact recursions we
require agreement to ~1e-10.
"""

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaln, log_softmax, logsumexp

from state_space_practice.contingency_belief import (
    centered_softmax,
    contingency_belief_filter,
    contingency_belief_smoother,
)
from state_space_practice.multinomial_choice import (
    multinomial_choice_filter,
    multinomial_choice_smoother,
)
from state_space_practice.smith_learning_algorithm import (
    smith_learning_filter,
    smith_learning_smoother,
)
from state_space_practice.switching_choice import (
    SwitchingChoiceModel,
    switching_choice_filter,
)

# ---------------------------------------------------------------------------
# Grid oracles (numpy, float64)
# ---------------------------------------------------------------------------


def _gauss_kernel(grid: np.ndarray, var: float) -> np.ndarray:
    """K[i, j] = N(grid[i]; grid[j], var) (density, not normalised on grid)."""
    diff = grid[:, None] - grid[None, :]
    return np.exp(-0.5 * diff**2 / var) / np.sqrt(2 * np.pi * var)


def _grid_forward_backward_1d(grid, prior_x1, log_liks, transition_var):
    """Exact filter / smoother / marginal LL for a 1-D random-walk SSM.

    Parameters
    ----------
    grid : (G,) evenly spaced
    prior_x1 : (G,) density of x_1 before the first observation
    log_liks : (T, G) log p(y_t | x_t = grid)
    transition_var : float, variance of x_t - x_{t-1}

    Returns
    -------
    filt : (T, G) normalised filtered densities
    smooth : (T, G) normalised smoothed densities
    log_marginal : float
    """
    h = grid[1] - grid[0]
    K = _gauss_kernel(grid, transition_var) * h
    n_time = log_liks.shape[0]
    filt = np.zeros((n_time, grid.size))
    lik = np.exp(log_liks - log_liks.max(axis=1, keepdims=True))
    log_marginal = 0.0
    prior = prior_x1
    for t in range(n_time):
        unnorm = prior * lik[t]
        z = unnorm.sum() * h
        log_marginal += np.log(z) + log_liks[t].max()
        filt[t] = unnorm / z
        prior = K @ filt[t]
    beta = np.ones(grid.size)
    smooth = np.zeros_like(filt)
    smooth[-1] = filt[-1]
    for t in range(n_time - 2, -1, -1):
        beta = K.T @ (lik[t + 1] * beta)
        beta = beta / beta.max()
        s = filt[t] * beta
        smooth[t] = s / (s.sum() * h)
    return filt, smooth, log_marginal


def _moments_1d(grid, dens):
    h = grid[1] - grid[0]
    mean = (dens * grid).sum(axis=-1) * h
    var = (dens * grid**2).sum(axis=-1) * h - mean**2
    return mean, var


def _grid_forward_backward_2d(grid, prior_x1, log_liks, transition_var):
    """2-D analogue with isotropic (separable) transition noise.

    ``prior_x1`` and each ``log_liks[t]`` have shape (G, G), axis 0 = x[0].
    """
    h = grid[1] - grid[0]
    K = _gauss_kernel(grid, transition_var) * h

    def propagate(d, K_):
        return K_ @ d @ K_.T

    n_time = log_liks.shape[0]
    lik = np.exp(log_liks - log_liks.max(axis=(1, 2), keepdims=True))
    filt = np.zeros((n_time,) + grid.shape * 2)
    log_marginal = 0.0
    prior = prior_x1
    for t in range(n_time):
        unnorm = prior * lik[t]
        z = unnorm.sum() * h**2
        log_marginal += np.log(z) + log_liks[t].max()
        filt[t] = unnorm / z
        prior = propagate(filt[t], K)
    smooth = np.zeros_like(filt)
    smooth[-1] = filt[-1]
    beta = np.ones_like(prior)
    for t in range(n_time - 2, -1, -1):
        beta = propagate(lik[t + 1] * beta, K.T)
        beta = beta / beta.max()
        s = filt[t] * beta
        smooth[t] = s / (s.sum() * h**2)
    return filt, smooth, log_marginal


def _moments_2d(grid, dens):
    """Means (T, 2) and covariances (T, 2, 2) of (T, G, G) densities."""
    h = grid[1] - grid[0]
    X0, X1 = np.meshgrid(grid, grid, indexing="ij")
    w = dens * h**2
    m0 = (w * X0).sum(axis=(1, 2))
    m1 = (w * X1).sum(axis=(1, 2))
    c00 = (w * X0**2).sum(axis=(1, 2)) - m0**2
    c11 = (w * X1**2).sum(axis=(1, 2)) - m1**2
    c01 = (w * X0 * X1).sum(axis=(1, 2)) - m0 * m1
    means = np.stack([m0, m1], axis=1)
    covs = np.stack([np.stack([c00, c01], -1), np.stack([c01, c11], -1)], -2)
    return means, covs


def _softmax_choice_log_liks_1d(grid, choices, beta):
    """log p(c | x) for K=2, logits [0, beta x]."""
    logits = np.stack([np.zeros_like(grid), beta * grid], axis=-1)  # (G, 2)
    lp = log_softmax(logits, axis=-1)
    return np.stack([lp[:, c] for c in choices])


def _softmax_choice_log_liks_2d(grid, choices, beta):
    """log p(c | x) for K=3, logits [0, beta x0, beta x1] on a (G, G) grid."""
    X0, X1 = np.meshgrid(grid, grid, indexing="ij")
    logits = np.stack([np.zeros_like(X0), beta * X0, beta * X1], axis=-1)
    lp = log_softmax(logits, axis=-1)
    return np.stack([lp[..., c] for c in choices])


def _exact_multinomial_1d(choices, beta, q, grid):
    # The model's x_0 ~ N(0, 1) is propagated once before the first choice.
    prior = np.exp(-0.5 * grid**2 / (1 + q)) / np.sqrt(2 * np.pi * (1 + q))
    lls = _softmax_choice_log_liks_1d(grid, choices, beta)
    filt, smooth, lm = _grid_forward_backward_1d(grid, prior, lls, q)
    fm, fv = _moments_1d(grid, filt)
    sm, sv = _moments_1d(grid, smooth)
    return fm, fv, sm, sv, lm


def _exact_multinomial_2d(choices, beta, q, grid):
    X0, X1 = np.meshgrid(grid, grid, indexing="ij")
    v = 1 + q
    prior = np.exp(-0.5 * (X0**2 + X1**2) / v) / (2 * np.pi * v)
    lls = _softmax_choice_log_liks_2d(grid, choices, beta)
    filt, smooth, lm = _grid_forward_backward_2d(grid, prior, lls, q)
    fm, fc = _moments_2d(grid, filt)
    sm, sc = _moments_2d(grid, smooth)
    return fm, fc, sm, sc, lm


_GRID_1D = np.linspace(-10.0, 10.0, 2001)
_GRID_2D = np.linspace(-7.0, 7.0, 281)


# ---------------------------------------------------------------------------
# Multinomial choice (Laplace-EKF + RTS)
# ---------------------------------------------------------------------------


class TestMultinomialChoiceQuadratureOracle:
    """Laplace-EKF filter / RTS smoother vs the exact posterior."""

    SEQUENCES_K2 = ([1, 1, 0, 1, 1, 1], [0, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 0])
    SEQUENCES_K3 = ([2, 2, 1, 0, 2, 2], [1, 1, 2, 1, 1, 0])

    @pytest.mark.parametrize("choices", SEQUENCES_K2)
    def test_two_options_filter_smoother_and_marginal(self, choices):
        beta, q = 2.0, 0.1
        fm, fv, sm, sv, lm = _exact_multinomial_1d(choices, beta, q, _GRID_1D)
        filt = multinomial_choice_filter(
            jnp.array(choices), 2, process_noise=q, inverse_temperature=beta
        )
        smooth = multinomial_choice_smoother(
            jnp.array(choices), 2, process_noise=q, inverse_temperature=beta
        )
        f_mean = np.asarray(filt.filtered_values[:, 0])
        f_var = np.asarray(filt.filtered_covariances[:, 0, 0])
        s_mean = np.asarray(smooth.smoothed_values[:, 0])
        s_var = np.asarray(smooth.smoothed_covariances[:, 0, 0])

        err_fm = np.max(np.abs(f_mean - fm))
        err_fv = np.max(np.abs(f_var - fv) / fv)
        err_sm = np.max(np.abs(s_mean - sm))
        err_sv = np.max(np.abs(s_var - sv) / sv)
        err_ll = abs(float(filt.marginal_log_likelihood) - lm)
        msg = (
            f"choices={choices}: filter mean {err_fm:.3g}, filter var (rel) "
            f"{err_fv:.3g}, smoother mean {err_sm:.3g}, smoother var (rel) "
            f"{err_sv:.3g}, log-marginal {err_ll:.3g}"
        )
        # Pins with headroom over the observed errors (max over the three
        # sequences: filter mean 0.28, filter var 0.23 rel, smoother mean
        # 0.19, smoother var 0.23 rel, log-marginal 0.12). The error is the
        # Laplace mode-vs-mean gap of a skewed one-step posterior, not an
        # unconverged Newton solve (see test_update_is_exact_laplace_step).
        assert err_fm < 0.4, msg
        assert err_fv < 0.35, msg
        assert err_sm < 0.3, msg
        assert err_sv < 0.35, msg
        assert err_ll < 0.2, msg
        # The approximation is not exact at beta=2: the gap is measurable.
        assert max(err_fm, err_fv, err_ll) > 1e-4, msg

        # The RTS smoother estimates the exact smoothed mean better than the
        # filtered mean does (interior trials only; the last trial is shared).
        dist_smooth = np.linalg.norm(s_mean[:-1] - sm[:-1])
        dist_filter = np.linalg.norm(f_mean[:-1] - sm[:-1])
        assert dist_smooth < 0.8 * dist_filter, msg

    def test_error_shrinks_with_inverse_temperature(self):
        """At small beta the softmax likelihood is nearly Gaussian in x."""
        choices = self.SEQUENCES_K2[0]
        errors = {}
        for beta in (0.2, 2.0):
            fm, _, sm, _, lm = _exact_multinomial_1d(choices, beta, 0.1, _GRID_1D)
            filt = multinomial_choice_filter(
                jnp.array(choices), 2, process_noise=0.1, inverse_temperature=beta
            )
            errors[beta] = (
                np.max(np.abs(np.asarray(filt.filtered_values[:, 0]) - fm)),
                abs(float(filt.marginal_log_likelihood) - lm),
            )
        assert errors[0.2][0] < 0.1 * errors[2.0][0], errors
        assert errors[0.2][1] < 0.1 * errors[2.0][1], errors
        assert errors[0.2][0] < 1e-3, errors

    @pytest.mark.parametrize("choices", SEQUENCES_K3)
    def test_three_options_filter_smoother_and_marginal(self, choices):
        beta, q = 2.0, 0.1
        fm, fc, sm, sc, lm = _exact_multinomial_2d(choices, beta, q, _GRID_2D)
        filt = multinomial_choice_filter(
            jnp.array(choices), 3, process_noise=q, inverse_temperature=beta
        )
        smooth = multinomial_choice_smoother(
            jnp.array(choices), 3, process_noise=q, inverse_temperature=beta
        )
        f_mean = np.asarray(filt.filtered_values)
        s_mean = np.asarray(smooth.smoothed_values)
        f_cov = np.asarray(filt.filtered_covariances)
        s_cov = np.asarray(smooth.smoothed_covariances)
        err_fm = np.max(np.abs(f_mean - fm))
        err_sm = np.max(np.abs(s_mean - sm))
        err_fc = np.max(np.abs(f_cov - fc)) / np.max(np.abs(fc))
        err_sc = np.max(np.abs(s_cov - sc)) / np.max(np.abs(sc))
        err_ll = abs(float(filt.marginal_log_likelihood) - lm)
        msg = (
            f"choices={choices}: filter mean {err_fm:.3g}, smoother mean "
            f"{err_sm:.3g}, filter cov {err_fc:.3g}, smoother cov {err_sc:.3g}, "
            f"log-marginal {err_ll:.3g}"
        )
        # Observed (both sequences): filter mean 0.21, smoother mean 0.15,
        # covariances 0.19 relative, log-marginal 0.09.
        assert err_fm < 0.35, msg
        assert err_sm < 0.3, msg
        assert err_fc < 0.3, msg
        assert err_sc < 0.3, msg
        assert err_ll < 0.2, msg
        assert max(err_fm, err_fc, err_ll) > 1e-3, msg
        dist_smooth = np.linalg.norm(s_mean[:-1] - sm[:-1])
        dist_filter = np.linalg.norm(f_mean[:-1] - sm[:-1])
        assert dist_smooth < 0.6 * dist_filter, msg

    @pytest.mark.parametrize(
        "prior_mean, prior_var, choice, beta",
        [(0.0, 1.1, 1, 2.0), (1.5, 0.5, 0, 2.0), (2.5, 0.4, 0, 4.0)],
    )
    def test_update_is_exact_laplace_step(self, prior_mean, prior_var, choice, beta):
        """One update = mode and inverse curvature of the exact 1-step posterior.

        The mode solves ``(x - m)/v = beta (e_c - p(x))``; we solve it by
        bisection and compare with the model's (3 unrolled Newton steps).
        """
        from scipy.optimize import brentq

        from state_space_practice.multinomial_choice import _softmax_update_core

        def score(x):
            p1 = 1.0 / (1.0 + np.exp(-beta * x))
            return beta * (float(choice == 1) - p1) - (x - prior_mean) / prior_var

        mode = brentq(score, -20, 20, xtol=1e-14)
        p1 = 1.0 / (1.0 + np.exp(-beta * mode))
        post_var = 1.0 / (1.0 / prior_var + beta**2 * p1 * (1 - p1))
        g = _GRID_1D
        h = g[1] - g[0]
        log_post = (
            -0.5 * (g - prior_mean) ** 2 / prior_var
            - 0.5 * np.log(2 * np.pi * prior_var)
            + log_softmax(np.stack([0 * g, beta * g], -1), -1)[:, choice]
        )
        log_z = logsumexp(log_post) + np.log(h)
        laplace_log_z = (
            -0.5 * (mode - prior_mean) ** 2 / prior_var
            - 0.5 * np.log(prior_var)
            + log_softmax(np.array([0.0, beta * mode]))[choice]
            + 0.5 * np.log(post_var)
        )
        # 10 steps: converged up to psd_solve's 1e-9 diagonal boost.
        for n_steps, tol in ((3, 2e-3), (10, 1e-7)):
            x, P, ll, _ = _softmax_update_core(
                jnp.array([prior_mean]),
                jnp.array([[prior_var]]),
                jnp.int32(choice),
                2,
                beta,
                max_newton_steps=n_steps,
            )
            assert abs(float(x[0]) - mode) < tol, (n_steps, float(x[0]), mode)
            assert abs(float(P[0, 0]) - post_var) < 10 * tol
            assert abs(float(ll) - laplace_log_z) < 10 * tol
        # The Laplace evidence is an approximation of the exact one-step
        # evidence: close, but measurably different for a skewed posterior.
        assert 1e-3 < abs(laplace_log_z - log_z) < 0.1, (laplace_log_z, log_z)


class TestLaplaceNewtonDoesNotOscillate:
    """Regression: the softmax Laplace update used undamped Newton steps.

    When the prior mean sits on the saturated side of the softmax opposite the
    observed choice, the likelihood curvature there is ~0 and a full Newton
    step overshoots to the other saturated side; Newton then oscillates
    between the two (prior N(1.512, 0.974), beta=4, choice 0: 1.51, -2.23,
    1.50, ... around the mode -0.09). The 3-step filter landed on -2.23 and on
    a 400-trial sequence at beta=4 reported a log-evidence of -1836 against
    the exact -118. The line-searched update converges to the mode.
    """

    def test_update_does_not_zigzag_across_the_mode(self):
        """Taking the largest Armijo-acceptable step (rather than the best
        one) let the iterates zigzag across the mode with slowly shrinking
        amplitude: for this prior (trial 41 of a simulated 200-trial run,
        beta=3, choice 1) the first components went -0.61, 1.89, -0.33,
        1.77, ... and ten iterations ended 0.02 from the mode (0.1 nats of
        filter evidence). Ten iterations must converge."""
        from state_space_practice.multinomial_choice import _softmax_update_core

        def mode(n_steps):
            return _softmax_update_core(
                jnp.array([-0.614, 1.363]),
                jnp.array([[0.931, 0.076], [0.076, 0.488]]),
                jnp.int32(1),
                3,
                3.0,
                max_newton_steps=n_steps,
            )[0]

        np.testing.assert_allclose(mode(10), mode(40), rtol=0, atol=1e-10)

    def test_single_update_reaches_the_mode(self):
        from scipy.optimize import brentq

        from state_space_practice.multinomial_choice import _softmax_update_core

        m, v, beta = 1.512, 0.974, 4.0

        def score(x):
            return beta * (0.0 - 1.0 / (1.0 + np.exp(-beta * x))) - (x - m) / v

        mode = brentq(score, -20, 20, xtol=1e-14)
        for n_steps, tol in ((3, 2e-3), (10, 1e-7)):
            x, _, _, _ = _softmax_update_core(
                jnp.array([m]),
                jnp.array([[v]]),
                jnp.int32(0),
                2,
                beta,
                max_newton_steps=n_steps,
            )
            assert abs(float(x[0]) - mode) < tol, (n_steps, float(x[0]), mode)
        assert abs(mode - m) > 1.0  # guard: the prior is far from the mode

    @pytest.mark.parametrize(
        "choices, q",
        [
            ([1] * 12 + [0, 1, 0, 0, 1], 0.3),
            ([1] * 8 + [0] * 2 + [1] * 8 + [0], 0.5),
            ([0] * 10 + [1] + [0] * 5 + [1], 0.4),
        ],
    )
    def test_filter_evidence_stays_near_exact_after_saturated_runs(self, choices, q):
        """Old code: log-evidence -70.7 / -93.7 / -67.0, filter-mean errors
        5.6 / 21.2 / 13.2. Now within the Laplace error (observed log-evidence
        errors 0.7 / 1.6 / 0.8, mean errors ~1.2) of the exact -8.9 / -10.5 /
        -8.8."""
        beta = 4.0
        fm, _, _, _, lm = _exact_multinomial_1d(choices, beta, q, _GRID_1D)
        filt = multinomial_choice_filter(
            jnp.array(choices), 2, process_noise=q, inverse_temperature=beta
        )
        err_ll = abs(float(filt.marginal_log_likelihood) - lm)
        err_m = np.max(np.abs(np.asarray(filt.filtered_values[:, 0]) - fm))
        assert err_ll < 2.5, (err_ll, float(filt.marginal_log_likelihood), lm)
        assert err_m < 2.0, err_m


@pytest.mark.slow
def test_laplace_evidence_is_biased_upwards_at_large_inverse_temperature():
    """Documented limitation of the Laplace evidence (not a code bug).

    At a saturating softmax the Laplace step evaluates p(c | x*) at the mode,
    where it is ~1 whenever the prior mean already favours the choice, and so
    ignores the prior mass on the wrong side of the decision boundary. On
    100 simulated trials (beta=2) the exact evidence falls by 4-12 nats from
    beta=2 to beta=12; the Laplace evidence falls much less (or rises), i.e.
    the beta M-step / SGD objective favours too-large inverse temperatures.
    Observed Laplace - exact at beta=12: 3.0, 5.2, 5.5, 3.6; at beta=2:
    -0.19, 0.00, 0.13, -0.46.
    """
    from state_space_practice.multinomial_choice import simulate_choice_data

    grid = np.linspace(-15, 15, 751)
    for seed in range(4):
        sim = simulate_choice_data(
            n_trials=100,
            n_options=2,
            process_noise=0.05,
            inverse_temperature=2.0,
            seed=seed,
        )
        c = np.asarray(sim.choices)
        gaps = {}
        for beta in (2.0, 12.0):
            exact = _exact_multinomial_1d(c, beta, 0.05, grid)[4]
            laplace = float(
                multinomial_choice_filter(
                    c, 2, process_noise=0.05, inverse_temperature=beta
                ).marginal_log_likelihood
            )
            gaps[beta] = laplace - exact
        assert abs(gaps[2.0]) < 0.8, (seed, gaps)
        assert 2.0 < gaps[12.0] < 9.0, (seed, gaps)


# ---------------------------------------------------------------------------
# Smith learning model (1-D Laplace filter + RTS)
# ---------------------------------------------------------------------------


def _binomial_log_liks(grid, y, n, mu):
    eta = mu + grid
    log_p = -np.logaddexp(0.0, -eta)
    log_q = -np.logaddexp(0.0, eta)
    coef = gammaln(n + 1) - gammaln(np.asarray(y) + 1) - gammaln(n - np.asarray(y) + 1)
    return coef[:, None] + np.outer(y, log_p) + np.outer(n - np.asarray(y), log_q)


def _exact_smith(y, n, sigma2, init_state, init_var, mu, grid=_GRID_1D):
    v1 = init_var + sigma2
    prior = np.exp(-0.5 * (grid - init_state) ** 2 / v1) / np.sqrt(2 * np.pi * v1)
    lls = _binomial_log_liks(grid, y, n, mu)
    filt, smooth, lm = _grid_forward_backward_1d(grid, prior, lls, sigma2)
    return _moments_1d(grid, filt), _moments_1d(grid, smooth), lm


def _smith_filter_smoother(y, n, sigma2, pc):
    out = smith_learning_filter(
        jnp.array(y),
        init_learning_state=0.0,
        init_learning_variance=sigma2,
        sigma_epsilon=float(np.sqrt(sigma2)),
        prob_correct_by_chance=pc,
        max_possible_correct=n,
    )
    sm_mode, sm_var, _, _ = smith_learning_smoother(
        out[1], out[2], out[3], out[4], prob_correct_by_chance=pc
    )
    return [np.asarray(o) for o in out[1:]] + [np.asarray(sm_mode), np.asarray(sm_var)]


class TestSmithQuadratureOracle:
    """Laplace filter / RTS smoother / evidence vs the exact posterior."""

    CASES = (
        # (y, max_possible_correct, sigma_epsilon^2, prob_correct_by_chance)
        ([0, 0, 1, 0, 1, 1], 1, 0.3, 0.5),
        ([1, 1, 1, 0, 1, 1], 1, 0.5, 0.5),
        ([1, 2, 2, 4, 5, 5], 5, 0.3, 0.25),
    )

    @pytest.mark.parametrize("y, n, sigma2, pc", CASES)
    def test_filter_smoother_and_evidence(self, y, n, sigma2, pc):
        from state_space_practice.smith_learning_algorithm import (
            smith_laplace_log_likelihood,
        )

        mu = np.log(pc / (1 - pc))
        (fm, fv), (sm, sv), lm = _exact_smith(np.array(y), n, sigma2, 0.0, sigma2, mu)
        mode, var, osm, osv, s_mode, s_var = _smith_filter_smoother(y, n, sigma2, pc)
        err_fm = np.max(np.abs(mode - fm))
        err_fv = np.max(np.abs(var - fv) / fv)
        err_sm = np.max(np.abs(s_mode - sm))
        err_sv = np.max(np.abs(s_var - sv) / sv)
        laplace_ll = float(
            jnp.sum(
                smith_laplace_log_likelihood(
                    jnp.array(y), jnp.full(len(y), n), mode, var, osm, osv, mu
                )
            )
        )
        err_ll = abs(laplace_ll - lm)
        msg = (
            f"y={y}: filter mode-vs-mean {err_fm:.3g}, filter var (rel) "
            f"{err_fv:.3g}, smoother {err_sm:.3g}, smoother var (rel) "
            f"{err_sv:.3g}, log-evidence {err_ll:.3g}"
        )
        # Observed maxima over the cases: 0.18, 0.12, 0.18, 0.14, 0.04.
        assert err_fm < 0.3, msg
        assert err_fv < 0.2, msg
        assert err_sm < 0.3, msg
        assert err_sv < 0.25, msg
        assert err_ll < 0.1, msg
        assert max(err_fm, err_fv) > 1e-3, msg
        dist_smooth = np.linalg.norm(s_mode[:-1] - sm[:-1])
        dist_filter = np.linalg.norm(mode[:-1] - sm[:-1])
        assert dist_smooth < 0.5 * dist_filter, msg

    def test_model_log_likelihood_is_laplace_evidence(self):
        """Regression: log_likelihood_ was the plug-in log p(y_k | m_{k|k-1}).

        The plug-in ignores the predictive variance. On a binomial sequence
        (N=5) it is ~0.95 nats from the exact evidence, the Laplace evidence
        0.02.
        """
        from state_space_practice.smith_learning_algorithm import SmithLearningModel

        y, n, sigma2, pc = self.CASES[2]
        mu = np.log(pc / (1 - pc))
        _, _, exact_ll = _exact_smith(np.array(y), n, sigma2, 0.0, sigma2, mu)
        model = SmithLearningModel(
            init_learning_variance=sigma2,
            sigma_epsilon=float(np.sqrt(sigma2)),
            prob_correct_by_chance=pc,
            max_possible_correct=n,
        )
        model_ll = model._e_step(jnp.array(y))
        _, _, _, osm, _, _, _ = [None] + _smith_filter_smoother(y, n, sigma2, pc)
        p = 1.0 / (1.0 + np.exp(-(mu + osm)))
        yy = np.array(y)
        plug_in = float(
            np.sum(
                gammaln(n + 1)
                - gammaln(yy + 1)
                - gammaln(n - yy + 1)
                + yy * np.log(p)
                + (n - yy) * np.log1p(-p)
            )
        )
        assert abs(model_ll - exact_ll) < 0.1, (model_ll, exact_ll)
        assert abs(plug_in - exact_ll) > 0.5, (plug_in, exact_ll)

    def test_evidence_argmax_over_sigma_matches_exact(self):
        """The model's evidence picks the same sigma as the exact evidence.

        150 Bernoulli trials simulated at sigma^2=0.05: the exact and Laplace
        evidence both peak at 0.05 on the grid, the old plug-in peaked at 0.02.
        """
        rng = np.random.default_rng(0)
        n_trials, sigma2_true = 150, 0.05
        x = np.cumsum(rng.normal(0, np.sqrt(sigma2_true), n_trials))
        y = (rng.random(n_trials) < 1 / (1 + np.exp(-x))).astype(int)
        grid_s2 = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
        coarse = np.linspace(-12, 12, 1201)
        exact, model_ll = [], []
        from state_space_practice.smith_learning_algorithm import SmithLearningModel

        for s2 in grid_s2:
            exact.append(_exact_smith(y, 1, s2, 0.0, s2, 0.0, grid=coarse)[2])
            m = SmithLearningModel(
                init_learning_variance=s2,
                sigma_epsilon=float(np.sqrt(s2)),
                max_possible_correct=1,
            )
            model_ll.append(m._e_step(jnp.array(y)))
        exact, model_ll = np.array(exact), np.array(model_ll)
        msg = f"exact {exact.round(2)}, model {model_ll.round(2)}"
        assert grid_s2[np.argmax(exact)] == 0.05, msg
        assert grid_s2[np.argmax(model_ll)] == 0.05, msg
        assert np.max(np.abs(model_ll - exact)) < 0.5, msg


# ---------------------------------------------------------------------------
# Contingency belief (discrete input-output HMM): exact by enumeration
# ---------------------------------------------------------------------------


def _enumerate_contingency(
    choices,
    rewards,
    reward_probs,
    state_values,
    beta,
    trans_mats,
    init_prob,
    obs_offsets,
):
    """Brute-force posterior over every state path.

    ``trans_mats[t]`` is the (S, S) matrix of the transition into trial t
    (row 0 unused); ``obs_offsets[t]`` is (S, K).
    """
    n_time = len(choices)
    S = init_prob.size
    log_emit = np.zeros((n_time, S))
    for t in range(n_time):
        c, r = choices[t], rewards[t]
        lp = log_softmax(beta * state_values + obs_offsets[t], axis=1)[:, c]
        p = reward_probs[:, c]
        log_emit[t] = lp + (np.log(p) if r == 1 else np.log1p(-p))

    def path_log_joint(path, upto):
        lj = np.log(init_prob[path[0]]) + log_emit[0, path[0]]
        for t in range(1, upto):
            lj += np.log(trans_mats[t][path[t - 1], path[t]]) + log_emit[t, path[t]]
        return lj

    filt = np.zeros((n_time, S))
    for t in range(n_time):
        paths = list(itertools.product(range(S), repeat=t + 1))
        lj = np.array([path_log_joint(p, t + 1) for p in paths])
        w = np.exp(lj - logsumexp(lj))
        for p, wi in zip(paths, w):
            filt[t, p[-1]] += wi
    paths = list(itertools.product(range(S), repeat=n_time))
    lj = np.array([path_log_joint(p, n_time) for p in paths])
    log_marginal = logsumexp(lj)
    w = np.exp(lj - log_marginal)
    smooth = np.zeros((n_time, S))
    pair = np.zeros((n_time - 1, S, S))
    for p, wi in zip(paths, w):
        for t in range(n_time):
            smooth[t, p[t]] += wi
        for t in range(n_time - 1):
            pair[t, p[t], p[t + 1]] += wi
    return filt, smooth, pair, log_marginal


class TestContingencyBeliefExactEnumeration:
    """The contingency-belief filter / smoother is an exact HMM recursion."""

    @pytest.mark.parametrize("per_state_obs", [False, True])
    def test_filter_smoother_pairwise_match_enumeration(self, per_state_obs):
        rng = np.random.default_rng(3)
        S, K, T, d_h, d_obs = 3, 3, 5, 2, 2
        choices = rng.integers(0, K, T)
        rewards = rng.integers(0, 2, T)
        reward_probs = rng.uniform(0.1, 0.9, (S, K))
        state_values = rng.normal(size=(S, K))
        beta = 1.7
        logits = rng.normal(size=(S, S - 1))
        weights = rng.normal(size=(S, S - 1, d_h))
        h = rng.normal(size=(T, d_h))
        z = rng.normal(size=(T, d_obs))
        init_prob = np.array([0.5, 0.3, 0.2])
        if per_state_obs:
            obs_w = rng.normal(size=(S, K, d_obs))
            offsets = np.einsum("skd,td->tsk", obs_w, z)
        else:
            obs_w = rng.normal(size=(K, d_obs))
            offsets = np.broadcast_to((z @ obs_w.T)[:, None, :], (T, S, K))
        trans = [None] + [
            np.asarray(
                centered_softmax(
                    jnp.asarray(logits + np.einsum("ijk,k->ij", weights, h[t]))
                )
            )
            for t in range(1, T)
        ]
        ef, es, ep, elm = _enumerate_contingency(
            choices,
            rewards,
            reward_probs,
            state_values,
            beta,
            trans,
            init_prob,
            offsets,
        )
        kwargs = dict(
            choices=choices,
            rewards=rewards,
            n_states=S,
            n_options=K,
            reward_probs=jnp.asarray(reward_probs),
            state_values=jnp.asarray(state_values),
            inverse_temperature=beta,
            transition_logits=jnp.asarray(logits),
            transition_covariates=jnp.asarray(h),
            transition_weights=jnp.asarray(weights),
            init_state_prob=jnp.asarray(init_prob),
            obs_design_matrix=jnp.asarray(z),
            obs_weights=jnp.asarray(obs_w),
        )
        filt = contingency_belief_filter(**kwargs)
        smooth = contingency_belief_smoother(**kwargs)
        np.testing.assert_allclose(filt.state_posterior, ef, atol=1e-10)
        np.testing.assert_allclose(smooth.smoothed_state_prob, es, atol=1e-10)
        np.testing.assert_allclose(smooth.pairwise_state_prob, ep, atol=1e-10)
        np.testing.assert_allclose(float(filt.log_likelihood), elm, atol=1e-10)
        np.testing.assert_allclose(float(smooth.log_likelihood), elm, atol=1e-10)
        # Guard: the smoother differs from the filter (future data matter).
        assert np.max(np.abs(es - ef)) > 0.05


# ---------------------------------------------------------------------------
# Switching choice: discrete part
# ---------------------------------------------------------------------------


def _hmm_enumerate(log_emit, init_prob, trans):
    """Exact filter / smoother / joint / evidence for a plain HMM by enumeration."""
    n_time, S = log_emit.shape

    def lj(path):
        v = np.log(init_prob[path[0]]) + log_emit[0, path[0]]
        for t in range(1, len(path)):
            v += np.log(trans[path[t - 1], path[t]]) + log_emit[t, path[t]]
        return v

    filt = np.zeros((n_time, S))
    for t in range(n_time):
        paths = list(itertools.product(range(S), repeat=t + 1))
        w = np.array([lj(p) for p in paths])
        w = np.exp(w - logsumexp(w))
        for p, wi in zip(paths, w):
            filt[t, p[-1]] += wi
    paths = list(itertools.product(range(S), repeat=n_time))
    w = np.array([lj(p) for p in paths])
    lm = logsumexp(w)
    w = np.exp(w - lm)
    smooth = np.zeros((n_time, S))
    joint = np.zeros((n_time - 1, S, S))
    for p, wi in zip(paths, w):
        for t in range(n_time):
            smooth[t, p[t]] += wi
        for t in range(n_time - 1):
            joint[t, p[t], p[t + 1]] += wi
    return filt, smooth, joint, lm


class TestSwitchingChoiceDiscreteOracle:
    """Discrete-state posterior of the GPB1 switching choice filter."""

    Z = np.array([[0.85, 0.15], [0.25, 0.75]])

    def test_exact_in_frozen_latent_regime(self):
        """With a (numerically) frozen latent the model is a plain HMM.

        init_cov ~ 0 and process noise ~ 0 pin x_t = init_mean, so each state
        s emits choice c with the fixed probability softmax(beta_s [0, x])[c];
        the Laplace updates and the GPB1 collapse are then exact up to O(cov).
        (init_cov is 1e-6, not smaller: below ~1e-8 the absolute 1e-9
        diagonal boost of ``psd_solve`` in the prior precision no longer
        matches the unboosted prior log-determinant, which offsets every
        state's evidence equally -- the discrete posterior stays exact but
        the marginal log-likelihood drifts: 1.7 nats at init_cov=1e-10,
        5e-3 at 1e-6.)
        """
        init_mean = np.array([0.8, -0.5])
        betas = np.array([0.5, 3.0])
        choices = np.array([1, 1, 0, 2, 1])
        model = SwitchingChoiceModel(
            n_options=3,
            n_discrete_states=2,
            init_inverse_temperatures=betas,
            init_process_noises=[1e-12, 1e-12],
        )
        model.init_mean_ = jnp.asarray(init_mean)
        model.init_cov_ = 1e-6 * jnp.eye(2)
        model.discrete_transition_matrix_ = jnp.asarray(self.Z)
        filt = model._run_filter(jnp.asarray(choices))
        smooth = model._run_smoother(filt)

        logits = np.concatenate([[0.0], init_mean])
        log_emit = np.stack(
            [log_softmax(b * logits)[choices] for b in betas], axis=1
        )  # (T, S)
        ef, es, ej, elm = _hmm_enumerate(log_emit, np.array([0.5, 0.5]), self.Z)
        np.testing.assert_allclose(filt.discrete_state_probs, ef, atol=1e-5)
        # The boost mismatch is 1e-3 per dimension and trial here (5e-3 total).
        np.testing.assert_allclose(float(filt.marginal_log_likelihood), elm, atol=1e-2)
        np.testing.assert_allclose(smooth[2], es, atol=1e-5)
        np.testing.assert_allclose(smooth[3], ej, atol=1e-5)
        assert np.max(np.abs(es - ef)) > 0.01  # guard: smoothing matters

    def test_identical_states_reduce_to_markov_prior(self):
        """Identical per-state parameters: the data carry no state information.

        The filtered / smoothed discrete posteriors must equal the Markov
        chain's prior marginals, and the evidence the non-switching filter's.
        """
        choices = np.array([2, 2, 1, 0, 2, 1])
        filt = switching_choice_filter(
            choices,
            3,
            n_discrete_states=2,
            process_noises=jnp.array([0.1, 0.1]),
            inverse_temperatures=jnp.array([2.0, 2.0]),
            discrete_transition_matrix=jnp.asarray(self.Z),
            init_discrete_prob=jnp.array([0.9, 0.1]),
        )
        prior = [np.array([0.9, 0.1])]
        for _ in range(len(choices) - 1):
            prior.append(prior[-1] @ self.Z)
        np.testing.assert_allclose(
            filt.discrete_state_probs, np.stack(prior), atol=1e-12
        )
        ref = multinomial_choice_filter(
            choices, 3, process_noise=0.1, inverse_temperature=2.0
        )
        np.testing.assert_allclose(
            float(filt.marginal_log_likelihood),
            float(ref.marginal_log_likelihood),
            atol=1e-10,
        )

    def test_general_regime_against_path_quadrature(self):
        """K=2, S=2, T=4: exact p(s_t | c) by path enumeration x quadrature.

        Each discrete path is a 1-D nonlinear SSM integrated on a grid. The
        GPB1 + Laplace filter's discrete posterior and evidence are pinned.
        """
        betas, qs = np.array([0.5, 3.0]), np.array([0.02, 0.3])
        choices = np.array([1, 1, 0, 1])
        T = len(choices)
        grid = np.linspace(-10, 10, 1001)
        h = grid[1] - grid[0]
        init_prob = np.array([0.5, 0.5])

        def path_log_evidence(path):
            """log p(c_{1:t} | path_{1:t}) for each prefix length t."""
            v1 = 1.0 + qs[path[0]]
            dens = np.exp(-0.5 * grid**2 / v1) / np.sqrt(2 * np.pi * v1)
            out = []
            total = 0.0
            for t, s in enumerate(path):
                if t > 0:
                    dens = (_gauss_kernel(grid, qs[s]) * h) @ dens
                lik = np.exp(
                    _softmax_choice_log_liks_1d(grid, [choices[t]], betas[s])[0]
                )
                z = (dens * lik).sum() * h
                total += np.log(z)
                dens = dens * lik / z
                out.append(total)
            return out

        paths = list(itertools.product(range(2), repeat=T))
        evid = {p: path_log_evidence(p) for p in paths}
        exact_filt = np.zeros((T, 2))
        for t in range(T):
            prefixes = {p[: t + 1] for p in paths}
            lj = {}
            for pre in prefixes:
                lp = np.log(init_prob[pre[0]]) + sum(
                    np.log(self.Z[pre[i - 1], pre[i]]) for i in range(1, t + 1)
                )
                lj[pre] = lp + evid[pre + (0,) * (T - t - 1)][t]
            vals = np.array(list(lj.values()))
            norm = logsumexp(vals)
            for pre, v in lj.items():
                exact_filt[t, pre[-1]] += np.exp(v - norm)
        full = np.array(
            [
                np.log(init_prob[p[0]])
                + sum(np.log(self.Z[p[i - 1], p[i]]) for i in range(1, T))
                + evid[p][-1]
                for p in paths
            ]
        )
        exact_lm = logsumexp(full)

        filt = switching_choice_filter(
            choices,
            2,
            n_discrete_states=2,
            process_noises=jnp.asarray(qs),
            inverse_temperatures=jnp.asarray(betas),
            discrete_transition_matrix=jnp.asarray(self.Z),
            init_discrete_prob=jnp.asarray(init_prob),
        )
        err_p = np.max(np.abs(np.asarray(filt.discrete_state_probs) - exact_filt))
        err_ll = abs(float(filt.marginal_log_likelihood) - exact_lm)
        msg = f"discrete-prob error {err_p:.3g}, log-evidence error {err_ll:.3g}"
        # Observed: discrete-prob error 0.025, log-evidence error 0.017.
        assert err_p < 0.06, msg
        assert err_ll < 0.05, msg
        assert max(err_p, err_ll) > 1e-3, msg
