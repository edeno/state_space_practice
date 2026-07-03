"""Head-to-head cross-check: Laplace-EKF vs. exact Polya-Gamma coupling posteriors.

Both estimators condition on the same LFP-smoothed latent and differ only in how
they infer the coupling posterior (Laplace/Fisher-scoring vs. exact Polya-Gamma
Gibbs). This module sweeps the coupling strength on identical simulated data and
scores both posteriors, to answer: *does the Laplace approximation bias or
mis-calibrate the coupling posterior, and where?*

Per cell it reports, for each method:

- ``coverage95`` — fraction of 95% credible intervals (over all real/imag
  components) that contain the true coefficient. ~0.95 if calibrated.
- ``abs_bias`` — mean absolute posterior-mean error over coupled components.
- ``ci_width`` — mean 95% interval width (posterior sharpness).
- ``detection_auc`` — ROC AUC of the Wald p-values vs the coupling mask.
- ``phase_mae`` — circular mean abs error of the recovered preferred phase.

plus ``ekf_pg_mean_maxdiff``: the max abs difference between the EKF and PG
posterior means (how much the two methods disagree), and shared latent plug-in
diagnostics: ``latent_correlation``, ``latent_rmse``, and
``latent_variance_ratio`` between the LFP-smoothed design and simulated truth.

Requires float64 and the ``coupling`` extra (``polyagamma``).
"""

import operator

import numpy as np

from state_space_practice.coupling_ekf import fit_coupling_ekf
from state_space_practice.coupling_model import (
    CouplingModelParams,
    smooth_latent_from_lfp,
)
from state_space_practice.coupling_pg import fit_coupling_pg
from state_space_practice.coupling_validation import (
    CouplingPosterior,
    phase_recovery_mae,
    roc_auc,
    summarize_posterior,
    wald_test,
)
from state_space_practice.simulate_coupling import simulate_coupling


def scale_coupling(params: CouplingModelParams, scale: float) -> CouplingModelParams:
    """Return ``params`` with the coupling magnitudes multiplied by ``scale``."""
    scale_arr = np.asarray(scale)
    if (
        scale_arr.shape != ()
        or not np.issubdtype(scale_arr.dtype, np.number)
        or np.issubdtype(scale_arr.dtype, np.complexfloating)
    ):
        raise ValueError(f"scale must be a finite non-negative scalar, got {scale}.")
    scale_float = float(scale_arr)
    if not np.isfinite(scale_float) or scale_float < 0.0:
        raise ValueError(f"scale must be a finite non-negative scalar, got {scale}.")
    return params._replace(
        beta_real=params.beta_real * scale_float,
        beta_imag=params.beta_imag * scale_float,
    )


def _score(post: CouplingPosterior, sim) -> dict:
    """Score one posterior against simulated ground truth."""
    mask = np.asarray(sim.coupling_mask)
    true_real = np.asarray(sim.beta_real_true)
    true_imag = np.asarray(sim.beta_imag_true)

    summary = summarize_posterior(post, cred_mass=0.95)
    covered_real = (summary["beta_real_ci_lower"] <= true_real) & (
        true_real <= summary["beta_real_ci_upper"]
    )
    covered_imag = (summary["beta_imag_ci_lower"] <= true_imag) & (
        true_imag <= summary["beta_imag_ci_upper"]
    )
    coverage = float(
        np.mean(np.concatenate([covered_real.ravel(), covered_imag.ravel()]))
    )

    if mask.any():
        abs_bias = float(
            np.mean(
                np.abs(
                    np.concatenate(
                        [
                            (np.asarray(post.beta_real_mean) - true_real)[mask],
                            (np.asarray(post.beta_imag_mean) - true_imag)[mask],
                        ]
                    )
                )
            )
        )
    else:
        abs_bias = float("nan")
    width_real = summary["beta_real_ci_upper"] - summary["beta_real_ci_lower"]
    width_imag = summary["beta_imag_ci_upper"] - summary["beta_imag_ci_lower"]
    ci_width = float(np.mean(np.concatenate([width_real.ravel(), width_imag.ravel()])))

    _, pval = wald_test(post)
    auc = roc_auc(pval, mask)
    mae = phase_recovery_mae(post, true_real, true_imag, mask)
    return {
        "coverage95": coverage,
        "abs_bias": abs_bias,
        "ci_width": ci_width,
        "detection_auc": auc,
        "phase_mae": mae,
    }


def _latent_plugin_diagnostics(sim) -> dict:
    """Compare the shared plug-in smoother mean to the simulator's true latent."""
    if not all(hasattr(sim, name) for name in ("latent_true", "lfp", "params")):
        return {}

    truth = np.asarray(sim.latent_true, dtype=float)
    smoothed = np.asarray(smooth_latent_from_lfp(sim.lfp, sim.params), dtype=float)
    if truth.shape != smoothed.shape:
        raise ValueError(
            "smoothed latent and simulated truth must have matching shapes; "
            f"got {smoothed.shape} and {truth.shape}"
        )

    truth_flat = truth.ravel()
    smoothed_flat = smoothed.ravel()
    truth_std = float(np.std(truth_flat))
    smoothed_std = float(np.std(smoothed_flat))
    if truth_std > 0.0 and smoothed_std > 0.0:
        latent_correlation = float(np.corrcoef(truth_flat, smoothed_flat)[0, 1])
    else:
        latent_correlation = float("nan")

    truth_var = float(np.var(truth_flat))
    smoothed_var = float(np.var(smoothed_flat))
    variance_ratio = smoothed_var / truth_var if truth_var > 0.0 else float("nan")
    return {
        "latent_correlation": latent_correlation,
        "latent_rmse": float(np.sqrt(np.mean((smoothed - truth) ** 2))),
        "latent_variance_ratio": float(variance_ratio),
    }


def run_crosscheck(
    base_params: CouplingModelParams,
    scales,
    n_time: int,
    n_replicates: int,
    seed: int = 0,
    pg_n_iter: int = 400,
    pg_burn_in: int = 200,
) -> list[dict]:
    """Sweep coupling strength and compare EKF vs PG posteriors on identical data.

    Parameters
    ----------
    base_params : CouplingModelParams
        Reference model; its coupling is scaled by each entry of ``scales``.
    scales : sequence of float
        Coupling-magnitude multipliers (the swept axis).
    n_time : int
        Bins per simulated dataset.
    n_replicates : int
        Independent simulations per scale (distinct seeds).
    seed : int
        Base seed; replicate ``r`` of scale index ``i`` uses ``seed + 1000*i + r``.
    pg_n_iter, pg_burn_in : int
        Gibbs sweeps / burn-in for the PG estimator.

    Returns
    -------
    list of dict
        One record per (scale, replicate) with keys ``scale``, ``coupling_mag``,
        ``replicate``, ``ekf`` (dict from :func:`_score`), ``pg`` (dict), and
        ``ekf_pg_mean_maxdiff``. Simulated runs also include
        ``latent_correlation``, ``latent_rmse``, and ``latent_variance_ratio`` for
        the shared plug-in latent design.
    """
    try:
        n_time = operator.index(n_time)
        n_replicates = operator.index(n_replicates)
    except TypeError as exc:
        raise ValueError("n_time and n_replicates must be positive integers.") from exc
    if n_time <= 0:
        raise ValueError(f"n_time must be positive, got {n_time}.")
    if n_replicates <= 0:
        raise ValueError(f"n_replicates must be positive, got {n_replicates}.")

    scales = list(scales)
    if not scales:
        raise ValueError("scales must contain at least one coupling multiplier.")
    scales_arr = np.asarray(scales, dtype=float)
    if scales_arr.ndim != 1 or not np.all(np.isfinite(scales_arr)):
        raise ValueError("scales must be a one-dimensional sequence of finite values.")
    if np.any(scales_arr < 0.0):
        raise ValueError("scales must be non-negative coupling multipliers.")

    records = []
    base_mag = float(
        np.max(
            np.hypot(
                np.asarray(base_params.beta_real), np.asarray(base_params.beta_imag)
            )
        )
    )
    for scale_index, scale in enumerate(scales_arr):
        params = scale_coupling(base_params, scale)
        for replicate in range(n_replicates):
            cell_seed = seed + 1000 * scale_index + replicate
            sim = simulate_coupling(params, n_time=n_time, seed=cell_seed)
            latent_diagnostics = _latent_plugin_diagnostics(sim)
            ekf = fit_coupling_ekf(sim.spikes, sim.lfp, params)
            pg = fit_coupling_pg(
                sim.spikes,
                sim.lfp,
                params,
                n_iter=pg_n_iter,
                burn_in=pg_burn_in,
                seed=cell_seed,
            )
            mean_maxdiff = float(
                np.max(
                    np.abs(
                        np.concatenate(
                            [
                                (
                                    np.asarray(ekf.beta_real_mean)
                                    - np.asarray(pg.beta_real_mean)
                                ).ravel(),
                                (
                                    np.asarray(ekf.beta_imag_mean)
                                    - np.asarray(pg.beta_imag_mean)
                                ).ravel(),
                            ]
                        )
                    )
                )
            )
            records.append(
                {
                    "scale": float(scale),
                    "coupling_mag": float(scale) * base_mag,
                    "replicate": replicate,
                    "ekf": _score(ekf, sim),
                    "pg": _score(pg, sim),
                    "ekf_pg_mean_maxdiff": mean_maxdiff,
                    **latent_diagnostics,
                }
            )
    return records


def aggregate(records: list[dict]) -> dict:
    """Average each metric across replicates, grouped by coupling magnitude.

    Returns a dict keyed by rounded ``coupling_mag`` -> {method -> {metric -> mean},
    ``ekf_pg_mean_maxdiff`` -> mean, and mean latent diagnostics when present}.
    """
    if not records:
        raise ValueError("records must contain at least one crosscheck result.")

    by_mag: dict = {}
    for record in records:
        by_mag.setdefault(round(record["coupling_mag"], 4), []).append(record)
    out = {}
    for mag, cell in by_mag.items():
        summary = {
            "n": len(cell),
            "ekf_pg_mean_maxdiff": float(
                np.mean([c["ekf_pg_mean_maxdiff"] for c in cell])
            ),
        }
        for metric in (
            "latent_correlation",
            "latent_rmse",
            "latent_variance_ratio",
        ):
            if metric in cell[0]:
                summary[metric] = _mean_ignore_nan([c[metric] for c in cell])
        for method in ("ekf", "pg"):
            summary[method] = {
                metric: _mean_ignore_nan([c[method][metric] for c in cell])
                for metric in cell[0][method]
            }
        out[mag] = summary
    return out


def _mean_ignore_nan(values) -> float:
    """Mean of finite values, or NaN when every value is undefined."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))
