"""Circular statistics utilities for phase analysis.

This module provides functions for circular statistics, useful for analyzing
phase relationships in oscillatory neural data.

Note: These utilities use NumPy/SciPy for data analysis and are not
JIT-compatible. For JAX-compatible operations, see the core model modules.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def circular_mean(phases: NDArray[np.floating]) -> float:
    """Compute the circular mean of angles.

    Parameters
    ----------
    phases : array
        Phase values in radians.

    Returns
    -------
    mean_phase : float
        Circular mean in radians, range [-pi, pi].

    Examples
    --------
    >>> phases = np.array([0, np.pi/4, -np.pi/4])
    >>> circular_mean(phases)  # Should be close to 0
    """
    return float(np.angle(np.mean(np.exp(1j * phases))))


def circular_std(phases: NDArray[np.floating]) -> float:
    """Compute the circular standard deviation of angles.

    Uses the formula: std = sqrt(-2 * log(R)) where R is the mean resultant
    length.

    Parameters
    ----------
    phases : array
        Phase values in radians.

    Returns
    -------
    std : float
        Circular standard deviation in radians.
    """
    R = mean_resultant_length(phases)
    # Clamp R to valid range [0, 1] to avoid numerical issues
    # (floating-point errors could cause R slightly > 1)
    R = np.clip(R, 1e-10, 1.0)
    return float(np.sqrt(-2 * np.log(R)))


def mean_resultant_length(phases: NDArray[np.floating]) -> float:
    """Compute the mean resultant length (R) of circular data.

    The mean resultant length is a measure of concentration of circular data.
    R = 1 means all phases are identical; R = 0 means uniform distribution.

    Parameters
    ----------
    phases : array
        Phase values in radians.

    Returns
    -------
    R : float
        Mean resultant length, range [0, 1].
    """
    return float(np.abs(np.mean(np.exp(1j * phases))))


def rayleigh_test(phases: NDArray[np.floating]) -> tuple[float, float]:
    """Rayleigh test for non-uniformity of circular distribution.

    Tests the null hypothesis that the phases are uniformly distributed
    on the circle.

    Parameters
    ----------
    phases : array
        Phase values in radians.

    Returns
    -------
    R : float
        Mean resultant length (test statistic).
    p_value : float
        P-value for the test. Small p-values indicate non-uniform distribution
        (i.e., significant phase locking).

    Notes
    -----
    The Rayleigh test is appropriate for unimodal alternatives to uniformity.
    For the approximation used here, the p-value is:
        p = exp(-n * R^2)
    which is accurate for large n.

    For small samples (n < 50), uses the corrected p-value from Mardia & Jupp (2000).

    References
    ----------
    Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics. Wiley.

    Examples
    --------
    >>> # Random phases (uniform distribution)
    >>> phases = np.random.uniform(-np.pi, np.pi, 1000)
    >>> R, p = rayleigh_test(phases)
    >>> p > 0.05  # Should be True (not significant)

    >>> # Phase-locked data (non-uniform)
    >>> phases = np.random.vonmises(0, 2, 1000)
    >>> R, p = rayleigh_test(phases)
    >>> p < 0.05  # Should be True (significant phase locking)
    """
    n = len(phases)
    R = mean_resultant_length(phases)
    z = n * R**2

    # Rayleigh test p-value (asymptotic approximation)
    p_value = float(np.exp(-z))

    # More accurate approximation for small samples (Mardia & Jupp, 2000)
    if n < 50:
        p_value = float(np.exp(-z) * (1 + (2 * z - z**2) / (4 * n) -
                        (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)))
        p_value = max(0, min(1, p_value))  # Clip to [0, 1]

    return R, p_value


def circular_correlation(
    phases1: NDArray[np.floating],
    phases2: NDArray[np.floating],
) -> float:
    """Compute circular-circular correlation coefficient.

    Uses the Fisher-Lee correlation coefficient for two circular variables.

    Parameters
    ----------
    phases1 : array
        First set of phase values in radians.
    phases2 : array
        Second set of phase values in radians.

    Returns
    -------
    r : float
        Circular correlation coefficient, range [-1, 1].

    Notes
    -----
    The Fisher-Lee correlation is defined as:
        r = sum(sin(a_i - a_bar) * sin(b_i - b_bar)) /
            sqrt(sum(sin(a_i - a_bar)^2) * sum(sin(b_i - b_bar)^2))

    References
    ----------
    Fisher, N.I. and Lee, A.J. (1983). A correlation coefficient for circular data.
    Biometrika, 70(2), 327-332.
    """
    if len(phases1) != len(phases2):
        raise ValueError("phases1 and phases2 must have the same length")

    # Compute circular means
    mean1 = circular_mean(phases1)
    mean2 = circular_mean(phases2)

    # Compute centered phases
    sin_centered1 = np.sin(phases1 - mean1)
    sin_centered2 = np.sin(phases2 - mean2)

    # Fisher-Lee correlation
    numerator = np.sum(sin_centered1 * sin_centered2)
    denominator = np.sqrt(np.sum(sin_centered1**2) * np.sum(sin_centered2**2))

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


def compute_phase_histogram(
    spike_times: NDArray[np.floating],
    inferred_phase: NDArray[np.floating],
    time_axis: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
    n_bins: int = 36,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute spike-phase histogram.

    Parameters
    ----------
    spike_times : array
        Spike times in seconds.
    inferred_phase : array, shape (n_time,)
        Phase values at each time bin in radians (should be in [-pi, pi]).
    time_axis : array, shape (n_time,)
        Time axis corresponding to inferred_phase (in seconds, same units
        as spike_times).
    mask : array, shape (n_time,), optional
        Boolean mask for which time points to include. If None, all points used.
    n_bins : int, default=36
        Number of phase bins (e.g., 36 = 10 degree bins).

    Returns
    -------
    histogram : array, shape (n_bins,)
        Spike counts in each phase bin.
    bin_centers : array, shape (n_bins,)
        Center of each phase bin in radians.

    Examples
    --------
    >>> spike_times = np.array([0.1, 0.5, 1.2, 1.8])
    >>> time_axis = np.arange(0, 2, 0.01)
    >>> phase = np.sin(2 * np.pi * 8 * time_axis)  # Dummy phase
    >>> hist, bins = compute_phase_histogram(spike_times, phase, time_axis)
    """
    from scipy.interpolate import interp1d

    # Interpolate phase to spike times
    # Use nearest interpolation to avoid phase wrapping issues
    phase_interp = interp1d(
        time_axis, inferred_phase, kind="nearest", bounds_error=False, fill_value=np.nan
    )
    spike_phases = phase_interp(spike_times)

    # Apply mask if provided
    if mask is not None:
        # Determine which spikes fall in masked regions
        mask_interp = interp1d(
            time_axis,
            mask.astype(float),
            kind="nearest",
            bounds_error=False,
            fill_value=0,
        )
        spike_in_mask = mask_interp(spike_times) > 0.5
        spike_phases = spike_phases[spike_in_mask]

    # Remove NaN phases (spikes outside time range)
    spike_phases = spike_phases[~np.isnan(spike_phases)]

    # Create phase bins
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histogram
    histogram, _ = np.histogram(spike_phases, bins=bin_edges)

    return histogram.astype(float), bin_centers


def compute_preferred_phase(
    spike_times: NDArray[np.floating],
    inferred_phase: NDArray[np.floating],
    time_axis: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> tuple[float, float, float]:
    """Compute preferred firing phase for a neuron.

    Parameters
    ----------
    spike_times : array
        Spike times in seconds.
    inferred_phase : array, shape (n_time,)
        Phase values at each time bin in radians (should be in [-pi, pi]).
    time_axis : array, shape (n_time,)
        Time axis corresponding to inferred_phase (in seconds, same units
        as spike_times).
    mask : array, shape (n_time,), optional
        Boolean mask for which time points to include.

    Returns
    -------
    preferred_phase : float
        Circular mean of spike phases (radians).
    phase_locking_strength : float
        Mean resultant length (R), measure of phase locking strength.
    p_value : float
        P-value from Rayleigh test for non-uniformity.
    """
    from scipy.interpolate import interp1d

    # Interpolate phase to spike times
    phase_interp = interp1d(
        time_axis, inferred_phase, kind="nearest", bounds_error=False, fill_value=np.nan
    )
    spike_phases = phase_interp(spike_times)

    # Apply mask if provided
    if mask is not None:
        mask_interp = interp1d(
            time_axis,
            mask.astype(float),
            kind="nearest",
            bounds_error=False,
            fill_value=0,
        )
        spike_in_mask = mask_interp(spike_times) > 0.5
        spike_phases = spike_phases[spike_in_mask]

    # Remove NaN phases
    spike_phases = spike_phases[~np.isnan(spike_phases)]

    if len(spike_phases) < 3:
        return np.nan, np.nan, np.nan

    preferred_phase = circular_mean(spike_phases)
    R, p_value = rayleigh_test(spike_phases)

    return preferred_phase, R, p_value


def angular_distance(
    phase1: float | NDArray[np.floating],
    phase2: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Compute angular distance between phases.

    Parameters
    ----------
    phase1 : float or array
        First phase value(s) in radians.
    phase2 : float or array
        Second phase value(s) in radians.

    Returns
    -------
    distance : float or array
        Angular distance in radians, range [0, pi].
    """
    diff = np.angle(np.exp(1j * (phase1 - phase2)))
    return np.abs(diff)


def wrap_to_pi(phases: NDArray[np.floating]) -> NDArray[np.floating]:
    """Wrap phases to [-pi, pi].

    Parameters
    ----------
    phases : array
        Phase values in radians.

    Returns
    -------
    wrapped : array
        Phase values wrapped to [-pi, pi].
    """
    result: npt.NDArray[np.floating] = np.angle(np.exp(1j * phases))
    return result
