"""Recovery-validation metrics for spike-field coupling estimates.

Scores an inferred coupling posterior against simulated ground truth: detection
(Wald chi-squared test, ROC/AUC, F1), magnitude and phase recovery, and credible
intervals. The metrics consume a single :class:`CouplingPosterior`, so any
inference method (Laplace-EKF, Polya-Gamma Gibbs) that produces one can be scored
the same way.

Shapes use ``S`` = number of neurons, ``J`` = number of latent oscillators
(bands). Coupling for neuron ``s`` and band ``j`` is the complex number
``beta_real[s, j] + 1j * beta_imag[s, j]``; its magnitude is coupling strength and
its angle is the preferred phase.

This module is pure post-hoc analysis on summary arrays (NumPy/SciPy/scikit-learn),
not part of the JAX inference path.
"""

import logging
from typing import NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from scipy import stats
from sklearn.metrics import roc_auc_score

from state_space_practice.circular_stats import angular_distance

logger = logging.getLogger(__name__)

# Variance below this is floored in Wald denominators. Zero mean with zero
# variance remains null, while a nonzero collapsed estimate remains significant
# instead of being silently converted to p=1.
_MIN_VARIANCE = 1e-12


class CouplingPosterior(NamedTuple):
    """Posterior summary of spike-field coupling, shared across inference methods.

    Parameters
    ----------
    beta_real_mean, beta_imag_mean : ndarray, shape (S, J)
        Posterior means of the in-phase and quadrature coupling.
    beta_real_var, beta_imag_var : ndarray, shape (S, J)
        Posterior variances of the in-phase and quadrature coupling. Required by
        the Wald test and the Gaussian credible interval; must be filled by every
        inference method.
    samples : ndarray or None, shape (n_samples, S, J), complex
        Posterior samples of ``beta_real + 1j * beta_imag``. ``None`` for a
        Gaussian (Laplace-EKF) posterior; populated by the Polya-Gamma sampler so
        percentile credible intervals are available.
    """

    beta_real_mean: npt.NDArray[np.floating]
    beta_imag_mean: npt.NDArray[np.floating]
    beta_real_var: npt.NDArray[np.floating]
    beta_imag_var: npt.NDArray[np.floating]
    samples: Optional[npt.NDArray[np.complexfloating]] = None


def wald_test(
    posterior: CouplingPosterior,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Wald chi-squared(2) test for nonzero complex coupling, per (neuron, band).

    Treats ``(beta_real, beta_imag)`` as a 2-vector and tests it against zero:

        W = beta_real_mean**2 / beta_real_var + beta_imag_mean**2 / beta_imag_var

    which is chi-squared(2) under the null of no coupling. Variances below
    ``_MIN_VARIANCE`` are floored per component rather than dropped; this avoids
    division by zero without turning strong collapsed estimates into nulls.

    Parameters
    ----------
    posterior : CouplingPosterior
        Uses ``beta_real_mean``, ``beta_imag_mean``, ``beta_real_var``,
        ``beta_imag_var``, each shape (S, J).

    Returns
    -------
    W : ndarray, shape (S, J)
        Wald statistic.
    pval : ndarray, shape (S, J)
        Upper-tail chi-squared(2) p-value.
    """
    mr = np.asarray(posterior.beta_real_mean, dtype=float)
    mi = np.asarray(posterior.beta_imag_mean, dtype=float)
    vr = np.asarray(posterior.beta_real_var, dtype=float)
    vi = np.asarray(posterior.beta_imag_var, dtype=float)

    if not (mr.shape == mi.shape == vr.shape == vi.shape):
        raise ValueError("posterior mean and variance arrays must have matching shapes")
    for name, arr in (
        ("beta_real_mean", mr),
        ("beta_imag_mean", mi),
        ("beta_real_var", vr),
        ("beta_imag_var", vi),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(vr < 0.0) or np.any(vi < 0.0):
        raise ValueError("posterior variances must be non-negative")

    safe_vr = np.maximum(vr, _MIN_VARIANCE)
    safe_vi = np.maximum(vi, _MIN_VARIANCE)
    W = mr**2 / safe_vr + mi**2 / safe_vi
    pval = stats.chi2.sf(W, df=2)
    return W, pval


def summarize_posterior(posterior: CouplingPosterior, cred_mass: float = 0.95) -> dict:
    """Posterior magnitude, phase, and per-component credible intervals.

    When ``posterior.samples`` is present the intervals are sample percentiles;
    otherwise they are Gaussian intervals ``mean +/- z * sqrt(var)`` per component.

    Parameters
    ----------
    posterior : CouplingPosterior
    cred_mass : float, default 0.95
        Central credible mass, e.g. 0.95 for a 95% interval.

    Returns
    -------
    dict
        Keys ``magnitude``, ``phase`` and the interval bounds
        ``beta_real_ci_lower/upper`` and ``beta_imag_ci_lower/upper``, each (S, J).
    """
    mr = np.asarray(posterior.beta_real_mean, dtype=float)
    mi = np.asarray(posterior.beta_imag_mean, dtype=float)

    magnitude = np.hypot(mr, mi)
    phase = np.arctan2(mi, mr)

    if posterior.samples is not None:
        samples = np.asarray(posterior.samples)
        lo_pct = 100.0 * (1.0 - cred_mass) / 2.0
        hi_pct = 100.0 - lo_pct
        real_lo, real_hi = np.percentile(samples.real, [lo_pct, hi_pct], axis=0)
        imag_lo, imag_hi = np.percentile(samples.imag, [lo_pct, hi_pct], axis=0)
    else:
        z = stats.norm.ppf(0.5 + cred_mass / 2.0)
        sd_r = np.sqrt(np.asarray(posterior.beta_real_var, dtype=float))
        sd_i = np.sqrt(np.asarray(posterior.beta_imag_var, dtype=float))
        real_lo, real_hi = mr - z * sd_r, mr + z * sd_r
        imag_lo, imag_hi = mi - z * sd_i, mi + z * sd_i

    return {
        "magnitude": magnitude,
        "phase": phase,
        "beta_real_ci_lower": real_lo,
        "beta_real_ci_upper": real_hi,
        "beta_imag_ci_lower": imag_lo,
        "beta_imag_ci_upper": imag_hi,
    }


def detection_metrics(
    pval: npt.NDArray[np.floating],
    coupling_mask: npt.NDArray[np.bool_],
    alpha: float = 0.05,
) -> dict:
    """Detection confusion counts and rates vs a ground-truth coupling mask.

    A (neuron, band) entry is "detected" when ``pval < alpha``. Element-wise
    counts are reported alongside a per-band view in which a band counts as
    detected if any neuron is significant for it
    (``(pval < alpha).any(axis=0)``) and as truly coupled if any neuron couples
    to it (``coupling_mask.any(axis=0)``). The per-band "any neuron" rule applies
    no multiple-comparison correction, so band-level false positives grow with
    the number of neurons; treat the per-band view as a screen, not a controlled
    test.

    Parameters
    ----------
    pval : ndarray, shape (S, J)
    coupling_mask : ndarray of bool, shape (S, J)
        Ground-truth coupling (``True`` where coupled).
    alpha : float, default 0.05

    Returns
    -------
    dict
        Element-wise ``tp/fp/fn/tn``, ``sensitivity``, ``specificity``,
        ``precision``, ``f1``, and the per-band counterparts prefixed ``band_``.
    """
    pval = np.asarray(pval, dtype=float)
    mask = np.asarray(coupling_mask, dtype=bool)
    out = _confusion(pval < alpha, mask)
    band_detected = (pval < alpha).any(axis=0)
    band_true = mask.any(axis=0)
    out.update(
        {f"band_{k}": v for k, v in _confusion(band_detected, band_true).items()}
    )
    return out


def _confusion(predicted: npt.NDArray[np.bool_], truth: npt.NDArray[np.bool_]) -> dict:
    """Confusion counts and rates from boolean prediction/truth arrays."""
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))

    def _ratio(num: int, den: int) -> float:
        return num / den if den > 0 else float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "precision": _ratio(tp, tp + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def roc_auc(
    pval: npt.NDArray[np.floating],
    coupling_mask: npt.NDArray[np.bool_],
) -> float:
    """ROC AUC for coupling detection, scoring by ``-log10(pval)``.

    Returns ``nan`` (with a logged reason) when the ground-truth labels are a
    single class, where AUC is undefined. If every entry shares the same score
    (e.g. a degenerate posterior collapsing all p-values), ``roc_auc_score``
    returns 0.5 (no skill) rather than raising.

    Parameters
    ----------
    pval : ndarray, shape (S, J)
    coupling_mask : ndarray of bool, shape (S, J)

    Returns
    -------
    float
        ROC AUC, or ``nan`` if labels are single-class.
    """
    labels = np.asarray(coupling_mask, dtype=int).ravel()
    if labels.min() == labels.max():
        logger.info("roc_auc undefined: coupling_mask is single-class; returning nan")
        return float("nan")
    score = -np.log10(np.asarray(pval, dtype=float).ravel() + 1e-300)
    return float(roc_auc_score(labels, score))


def phase_recovery_mae(
    posterior: CouplingPosterior,
    beta_real_true: npt.NDArray[np.floating],
    beta_imag_true: npt.NDArray[np.floating],
    coupling_mask: npt.NDArray[np.bool_],
) -> float:
    """Mean circular distance between recovered and true preferred phase.

    Averaged over coupled entries only (``coupling_mask == True``). Returns
    ``nan`` if no entry is coupled.

    Parameters
    ----------
    posterior : CouplingPosterior
    beta_real_true, beta_imag_true : ndarray, shape (S, J)
    coupling_mask : ndarray of bool, shape (S, J)

    Returns
    -------
    float
        Mean angular distance in radians, range [0, pi].
    """
    mask = np.asarray(coupling_mask, dtype=bool)
    if not mask.any():
        return float("nan")
    recovered_phase = np.arctan2(
        np.asarray(posterior.beta_imag_mean, dtype=float),
        np.asarray(posterior.beta_real_mean, dtype=float),
    )
    true_phase = np.arctan2(
        np.asarray(beta_imag_true, dtype=float),
        np.asarray(beta_real_true, dtype=float),
    )
    distances = angular_distance(recovered_phase[mask], true_phase[mask])
    return float(np.mean(distances))


def magnitude_recovery(
    posterior: CouplingPosterior,
    beta_real_true: npt.NDArray[np.floating],
    beta_imag_true: npt.NDArray[np.floating],
    coupling_mask: npt.NDArray[np.bool_],
) -> dict:
    """Correlation between recovered and true coupling magnitude over coupled entries.

    Parameters
    ----------
    posterior : CouplingPosterior
    beta_real_true, beta_imag_true : ndarray, shape (S, J)
    coupling_mask : ndarray of bool, shape (S, J)

    Returns
    -------
    dict
        ``n`` (number of coupled entries scored), ``pearson_r`` and
        ``spearman_r``. Correlations are ``nan`` when fewer than 3 entries are
        coupled (correlation is not meaningful).
    """
    mask = np.asarray(coupling_mask, dtype=bool)
    recovered_mag = np.hypot(
        np.asarray(posterior.beta_real_mean, dtype=float),
        np.asarray(posterior.beta_imag_mean, dtype=float),
    )[mask]
    true_mag = np.hypot(
        np.asarray(beta_real_true, dtype=float),
        np.asarray(beta_imag_true, dtype=float),
    )[mask]

    n = int(mask.sum())
    # Correlation is undefined for fewer than 3 points or a constant input
    # (zero variance). Return NaN directly rather than let SciPy emit a
    # ``ConstantInputWarning`` — the test suite escalates warnings to errors.
    if n < 3 or np.std(recovered_mag) == 0.0 or np.std(true_mag) == 0.0:
        return {"n": n, "pearson_r": float("nan"), "spearman_r": float("nan")}
    return {
        "n": n,
        "pearson_r": float(stats.pearsonr(recovered_mag, true_mag).statistic),
        "spearman_r": float(stats.spearmanr(recovered_mag, true_mag).statistic),
    }
