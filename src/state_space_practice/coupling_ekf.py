"""Laplace-EKF spike-field coupling estimator.

Estimates the complex coupling ``beta`` in two stages. Because the LFP observes
the latent, the two stages decouple — there is no bilinear ``beta * x``
degeneracy in *this estimator* (a spikes-only joint fit would be degenerate):

1. **Smooth the latent from the LFP** (linear, exact). The field observes the
   latent, ``lfp_k = x_k + N(0, lfp_noise_var I)``, so a linear-Gaussian Kalman
   smoother recovers ``x`` without using the spikes.
2. **Regress coupling from spikes** (Laplace). For each neuron, logistic-regress
   its spikes on the smoothed latent via :func:`glm_laplace_update` with the
   Bernoulli family. The returned coupling posterior mean/variance is the Laplace
   (Fisher-scoring) approximation; its bias is quantified by the Polya-Gamma
   cross-check (:mod:`coupling_pg` / :mod:`coupling_crosscheck`).

Stage 2 is exactly a MAP logistic regression with a known intercept
(``params.baseline``) and a static coupling: the returned mean is the penalised
MLE and the covariance the inverse observed information at it (the canonical
link makes Fisher scoring and Newton coincide). There are no coupling dynamics.

The posterior is conditional on the smoothed latent *mean*: the smoother's
uncertainty is ignored (errors-in-variables). Against simulated truth
(``|beta| = 1.8``, T = 3000, 60 seeds) the posterior mean of the magnitude stays
unbiased (within ~1%) but the intervals are too narrow as the field gets noisier:
90% coverage ~0.89 / 0.86 / 0.69 at ``lfp_noise_var`` = 0.01 / 0.25 (default) /
4.0. The Polya-Gamma arm shares this plug-in.

Requires float64 (the test suite enables ``jax_enable_x64``).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.coupling_model import (
    CouplingModelParams,
    deinterleave_coupling,
    smooth_latent_from_lfp,
    validate_coupling_observations,
    validate_coupling_params,
)
from state_space_practice.coupling_validation import CouplingPosterior
from state_space_practice.point_process_kalman import (
    BERNOULLI_LOGIT_FAMILY,
    _warn_line_search_failures,
    glm_laplace_update,
)
from state_space_practice.utils import typed_jit


@functools.partial(typed_jit, static_argnames=("max_newton_iter",))
def _regress_coupling_all_neurons(
    spikes: Array,
    smoothed_latent: Array,
    baseline: Array,
    prior_mean: Array,
    prior_cov: Array,
    *,
    max_newton_iter: int,
) -> tuple[Array, Array]:
    """Bernoulli Laplace regression of every neuron's spikes on the latent.

    The neurons share the prior and the constant design ``smoothed_latent``
    and differ only in their spike column and baseline, so the whole
    population is one ``jax.vmap`` over neurons under one ``jax.jit``: the
    Newton / line-search scan inside :func:`glm_laplace_update` is traced
    and compiled once for the whole population. A warning is logged when more
    than 10% of the regressions exhaust their line search.

    Parameters
    ----------
    spikes : Array, shape (T, S)
    smoothed_latent : Array, shape (T, 2J)
    baseline : Array, shape (S,)
    prior_mean : Array, shape (2J,)
    prior_cov : Array, shape (2J, 2J)
    max_newton_iter : int

    Returns
    -------
    beta_mean : Array, shape (S, 2J)
    beta_cov : Array, shape (S, 2J, 2J)
    """

    def constant_jacobian(_beta: Array) -> Array:
        # eta(beta) = baseline + smoothed_latent @ beta is linear in beta, so
        # its Jacobian is the constant design (passed to skip jacfwd).
        return smoothed_latent

    def _fit_one(spikes_n: Array, baseline_n: Array) -> tuple[Array, Array, Array]:
        def eta(beta: Array) -> Array:
            return baseline_n + smoothed_latent @ beta

        beta_mean, beta_cov, _, n_failed = glm_laplace_update(
            prior_mean,
            prior_cov,
            spikes_n,
            eta,
            BERNOULLI_LOGIT_FAMILY,
            grad_eta_func=constant_jacobian,
            max_newton_iter=max_newton_iter,
            return_line_search_failures=True,
        )
        return beta_mean, beta_cov, n_failed

    beta_mean, beta_cov, n_failed = jax.vmap(_fit_one, in_axes=(1, 0))(spikes, baseline)
    _warn_line_search_failures(
        jnp.sum(n_failed > 0),
        spikes.shape[1],
        max_newton_iter,
        "fit_coupling_ekf",
        unit="neuron regressions",
    )
    return beta_mean, beta_cov


def fit_coupling_ekf(
    spikes: ArrayLike,
    lfp: ArrayLike,
    params: CouplingModelParams,
    sigma_beta: float = 5.0,
    max_newton_iter: int = 10,
) -> CouplingPosterior:
    """Estimate spike-field coupling by LFP smoothing + Bernoulli Laplace regression.

    Parameters
    ----------
    spikes : ArrayLike, shape (T, S)
        0/1 spike indicators.
    lfp : ArrayLike, shape (T, 2J)
        Field observation of the latent.
    params : CouplingModelParams
        Model parameters (oscillator dynamics, baseline, ``lfp_noise_var``). The
        coupling fields are not used by the fit; everything else is.
    sigma_beta : float, default 5.0
        Standard deviation of the zero-mean Gaussian prior on the coupling
        (weakly informative).
    max_newton_iter : int, default 10
        Fisher-scoring iterations for each per-neuron logistic regression.

    Returns
    -------
    CouplingPosterior
        Gaussian coupling posterior (``samples=None``), consumed by
        :mod:`coupling_validation`.
    """
    validate_coupling_params(params)
    n_latent = 2 * int(np.shape(params.osc_frequencies)[0])
    n_neurons = int(np.shape(params.beta_real)[0])
    spikes_np, lfp_np = validate_coupling_observations(
        spikes, lfp, n_neurons=n_neurons, n_latent=n_latent
    )
    sigma_beta_arr = np.asarray(sigma_beta)
    if (
        sigma_beta_arr.shape != ()
        or not np.issubdtype(sigma_beta_arr.dtype, np.number)
        or np.issubdtype(sigma_beta_arr.dtype, np.complexfloating)
    ):
        raise ValueError(f"sigma_beta must be finite and positive, got {sigma_beta}.")
    sigma_beta_float = float(sigma_beta_arr)
    if not np.isfinite(sigma_beta_float) or sigma_beta_float <= 0.0:
        raise ValueError(f"sigma_beta must be finite and positive, got {sigma_beta}.")
    max_iter_arr = np.asarray(max_newton_iter)
    if (
        max_iter_arr.shape != ()
        or not np.issubdtype(max_iter_arr.dtype, np.integer)
        or int(max_iter_arr) <= 0
    ):
        raise ValueError(
            f"max_newton_iter must be a positive integer, got {max_newton_iter}."
        )
    spikes = jnp.asarray(spikes_np)
    lfp = jnp.asarray(lfp_np)

    # Stage 1: Kalman-smooth the latent from the LFP (shared with the PG estimator).
    smoothed_latent = smooth_latent_from_lfp(lfp, params)

    # Stage 2: Bernoulli logistic regression of every neuron's spikes on the
    # smoothed x, vmapped over neurons under one jit (shared prior, constant
    # Jacobian).
    prior_mean = jnp.zeros(n_latent)
    prior_cov = sigma_beta_float**2 * jnp.eye(n_latent)
    beta_mean_rows, beta_covs = _regress_coupling_all_neurons(
        spikes,
        smoothed_latent,
        jnp.asarray(params.baseline),
        prior_mean,
        prior_cov,
        max_newton_iter=int(max_iter_arr),
    )  # (S, 2J), (S, 2J, 2J)
    real_indices = jnp.arange(0, n_latent, 2)
    imag_indices = real_indices + 1
    beta_var_rows = jnp.diagonal(beta_covs, axis1=1, axis2=2)  # (S, 2J)
    beta_real_imag_cov = beta_covs[:, real_indices, imag_indices]  # (S, J)

    # A non-finite posterior means the smoother or a logistic regression blew up
    # (e.g. ill-conditioned init, separable spikes). Fail loudly rather than
    # return silent NaNs that propagate into detection downstream.
    if not (
        jnp.all(jnp.isfinite(beta_mean_rows))
        and jnp.all(jnp.isfinite(beta_var_rows))
        and jnp.all(jnp.isfinite(beta_real_imag_cov))
    ):
        raise FloatingPointError(
            "coupling posterior contains non-finite values; check the LFP/spikes "
            "and that x64 is enabled"
        )

    beta_real_mean, beta_imag_mean = deinterleave_coupling(beta_mean_rows)
    beta_real_var, beta_imag_var = deinterleave_coupling(beta_var_rows)

    return CouplingPosterior(
        beta_real_mean=np.asarray(beta_real_mean),
        beta_imag_mean=np.asarray(beta_imag_mean),
        beta_real_var=np.asarray(beta_real_var),
        beta_imag_var=np.asarray(beta_imag_var),
        beta_real_imag_cov=np.asarray(beta_real_imag_cov),
        samples=None,
    )
