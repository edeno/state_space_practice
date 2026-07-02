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

Requires float64 (the test suite enables ``jax_enable_x64``).
"""

import jax.numpy as jnp
import numpy as np

from state_space_practice.coupling_model import (
    CouplingModelParams,
    deinterleave_coupling,
    smooth_latent_from_lfp,
    validate_coupling_params,
)
from state_space_practice.coupling_validation import CouplingPosterior
from state_space_practice.point_process_kalman import (
    BERNOULLI_LOGIT_FAMILY,
    glm_laplace_update,
)


def fit_coupling_ekf(
    spikes,
    lfp,
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
    spikes = jnp.asarray(spikes)
    lfp = jnp.asarray(lfp)
    n_latent = 2 * int(np.shape(params.osc_frequencies)[0])
    n_neurons = int(np.shape(params.beta_real)[0])

    # Cross-check shapes so a mismatch raises rather than silently broadcasting or
    # dropping neurons (e.g. extra spike columns going unfit).
    if spikes.ndim != 2 or spikes.shape[1] != n_neurons:
        raise ValueError(
            f"spikes must have shape (T, {n_neurons}); got {tuple(spikes.shape)}"
        )
    if lfp.shape != (spikes.shape[0], n_latent):
        raise ValueError(
            f"lfp must have shape ({spikes.shape[0]}, {n_latent}); got {tuple(lfp.shape)}"
        )

    # Stage 1: Kalman-smooth the latent from the LFP (shared with the PG estimator).
    smoothed_latent = smooth_latent_from_lfp(lfp, params)

    # Stage 2: per-neuron Bernoulli logistic regression of spikes on smoothed x.
    # eta(beta) = baseline + smoothed_latent @ beta is linear in beta, so its
    # Jacobian is the constant design smoothed_latent (passed to skip jacfwd).
    prior_mean = jnp.zeros(n_latent)
    prior_cov = sigma_beta**2 * jnp.eye(n_latent)

    def constant_jacobian(_beta):
        return smoothed_latent

    means = []
    variances = []
    for neuron in range(n_neurons):
        baseline_n = params.baseline[neuron]

        def eta(beta, baseline_n=baseline_n):
            return baseline_n + smoothed_latent @ beta

        beta_mean, beta_cov, _ = glm_laplace_update(
            prior_mean,
            prior_cov,
            spikes[:, neuron],
            eta,
            BERNOULLI_LOGIT_FAMILY,
            grad_eta_func=constant_jacobian,
            max_newton_iter=max_newton_iter,
        )
        means.append(beta_mean)
        variances.append(jnp.diag(beta_cov))

    beta_mean_rows = jnp.stack(means)  # (S, 2J)
    beta_var_rows = jnp.stack(variances)  # (S, 2J)

    # A non-finite posterior means the smoother or a logistic regression blew up
    # (e.g. ill-conditioned init, separable spikes). Fail loudly rather than
    # return silent NaNs that propagate into detection downstream.
    if not (
        jnp.all(jnp.isfinite(beta_mean_rows)) and jnp.all(jnp.isfinite(beta_var_rows))
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
        samples=None,
    )
