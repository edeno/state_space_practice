"""Polya-Gamma Gibbs spike-field coupling estimator.

The "exact" arm of the EKF-vs-Polya-Gamma cross-check. Like
:func:`coupling_ekf.fit_coupling_ekf`, it conditions on the LFP-smoothed latent
(stage 1, shared via :func:`coupling_model.smooth_latent_from_lfp`), so the two
estimators differ *only* in how they infer the coupling posterior given that
latent: Laplace/Fisher-scoring (EKF) vs. exact Polya-Gamma Gibbs (here).

For each neuron, the coupling is a Bayesian Bernoulli-logit regression of its
spikes on the smoothed latent. Polya-Gamma augmentation (Polson, Scott & Windle,
2013) introduces a latent ``omega_k > 0`` per bin that renders the logistic
likelihood conditionally Gaussian, giving a two-block Gibbs sampler:

- ``omega_k | beta ~ PG(1, eta_k)`` with ``eta_k = baseline + x_k . beta``;
- ``beta | omega`` is Gaussian with precision ``Sigma_beta^-1 + X' diag(omega) X``
  and mean solving ``(prec) mu = X' (kappa - omega * baseline)``, ``kappa = y - 1/2``.

Unlike the EKF, this returns posterior *samples*, so credible intervals are
sample (percentile) intervals rather than Gaussian. Both estimators share the
stage-1 plug-in latent, so PG is "exact" only in stage 2 — it removes the EKF's
Laplace approximation of the coupling posterior, not the smoother plug-in. No
convergence/mixing diagnostic is computed; the caller owns choosing ``n_iter`` and
``burn_in`` large enough for the chain to mix. Requires float64 (tests enable
``jax_enable_x64``). Needs the ``coupling`` optional dependency (``polyagamma``).
"""

import numpy as np
import operator
from polyagamma import random_polyagamma
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from state_space_practice.coupling_model import (
    CouplingModelParams,
    smooth_latent_from_lfp,
    validate_coupling_observations,
    validate_coupling_params,
)
from state_space_practice.coupling_validation import CouplingPosterior


def fit_coupling_pg(
    spikes,
    lfp,
    params: CouplingModelParams,
    n_iter: int = 400,
    burn_in: int = 200,
    sigma_beta: float = 5.0,
    seed: int = 0,
) -> CouplingPosterior:
    """Estimate spike-field coupling by LFP smoothing + per-neuron Polya-Gamma Gibbs.

    Parameters
    ----------
    spikes : ArrayLike, shape (T, S)
        0/1 spike indicators.
    lfp : ArrayLike, shape (T, 2J)
        Field observation of the latent.
    params : CouplingModelParams
        Model parameters (oscillator dynamics, baseline, ``lfp_noise_var``). The
        coupling fields are not used by the fit.
    n_iter, burn_in : int
        Total Gibbs sweeps and burn-in; ``n_iter - burn_in`` samples are kept.
    sigma_beta : float, default 5.0
        Std of the zero-mean Gaussian prior on the coupling.
    seed : int, default 0
        Seed for the Gibbs sampler.

    Returns
    -------
    CouplingPosterior
        ``samples`` is filled, shape ``(n_iter - burn_in, S, J)`` complex; the
        mean/variance fields are the sample mean/variance.
    """
    validate_coupling_params(params)
    try:
        n_iter = operator.index(n_iter)
        burn_in = operator.index(burn_in)
    except TypeError as exc:
        raise ValueError(
            f"n_iter and burn_in must be integers; got n_iter={n_iter}, "
            f"burn_in={burn_in}."
        ) from exc
    if not 0 <= burn_in < n_iter:
        raise ValueError(
            f"need 0 <= burn_in < n_iter; got burn_in={burn_in}, n_iter={n_iter}"
        )
    n_latent = 2 * int(np.shape(params.osc_frequencies)[0])
    n_neurons = int(np.shape(params.beta_real)[0])
    spikes, lfp = validate_coupling_observations(
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

    # Stage 1: LFP-smoothed latent (shared design with the EKF estimator).
    design = np.asarray(smooth_latent_from_lfp(lfp, params))  # (T, n_latent)
    baseline = np.asarray(params.baseline, dtype=float)
    prior_precision = np.eye(n_latent) / sigma_beta_float**2
    rng = np.random.default_rng(seed)
    n_kept = n_iter - burn_in
    samples = np.empty((n_kept, n_neurons, n_latent))  # interleaved design rows

    for neuron in range(n_neurons):
        y = spikes[:, neuron]
        kappa = y - 0.5
        offset = baseline[neuron]
        beta = np.zeros(n_latent)
        for sweep in range(n_iter):
            eta = offset + design @ beta
            omega = random_polyagamma(h=1.0, z=eta, random_state=rng)
            # beta | omega ~ N(mu, prec^-1), prec = prior + X' diag(omega) X
            precision = prior_precision + design.T @ (omega[:, None] * design)
            precision = 0.5 * (precision + precision.T) + 1e-10 * np.eye(n_latent)
            chol = cho_factor(precision, lower=True)
            mean = cho_solve(chol, design.T @ (kappa - omega * offset))
            # sample: beta = mean + L^{-T} z  (cov = (L L^T)^{-1})
            standard_normal = rng.standard_normal(n_latent)
            beta = mean + solve_triangular(
                chol[0], standard_normal, lower=True, trans="T"
            )
            if sweep >= burn_in:
                samples[sweep - burn_in, neuron] = beta

    # Reduce interleaved rows -> complex (n_kept, S, J).
    real = samples[..., 0::2]
    imag = samples[..., 1::2]
    if not np.all(np.isfinite(samples)):
        raise FloatingPointError(
            "Polya-Gamma coupling samples contain non-finite values; check the "
            "LFP/spikes and that x64 is enabled"
        )
    real_mean = real.mean(axis=0)
    imag_mean = imag.mean(axis=0)
    real_imag_cov = ((real - real_mean) * (imag - imag_mean)).mean(axis=0)
    return CouplingPosterior(
        beta_real_mean=real_mean,
        beta_imag_mean=imag_mean,
        beta_real_var=real.var(axis=0),
        beta_imag_var=imag.var(axis=0),
        beta_real_imag_cov=real_imag_cov,
        samples=real + 1j * imag,
    )
