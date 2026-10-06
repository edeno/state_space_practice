"""Circular statistics utilities for phase analysis.

This module provides functions for circular statistics, useful for analyzing
phase relationships in oscillatory neural data.

Note: These utilities use NumPy/SciPy for data analysis and are not
JIT-compatible. For JAX-compatible operations, see the core model modules.

Relation to :mod:`scipy.stats`
------------------------------
Where SciPy computes the same quantity with the same conventions the function
delegates to it; the remaining functions are hand-written because SciPy has no
equivalent or its conventions differ (each docstring says which):

========================  ====================================================
``circular_std``          ``scipy.stats.circstd(phases)`` (radians), plus the
                          documented floor ``R >= 1e-10``.
``circular_mean``         hand-written: ``scipy.stats.circmean(..., low=-pi,
                          high=pi)`` returns ``[-pi, pi)`` (not ``(-pi, pi]``)
                          and loses relative precision near 0.
``mean_resultant_length`` hand-written: SciPy exposes ``R`` only as
                          ``1 - circvar`` (cancellation) or via
                          ``directional_stats`` on Cartesian vectors.
``rayleigh_test``         hand-written: no SciPy Rayleigh test
                          (``scipy.stats.rayleigh`` is the distribution).
``circular_correlation``  hand-written: no SciPy circular correlation.
``angular_distance``,     hand-written NumPy one-liners (no SciPy equivalent).
``wrap_to_pi``
========================  ====================================================
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from scipy import stats

#: Floor on the mean resultant length in :func:`circular_std`, so (numerically)
#: uniform phases give a large finite spread instead of ``inf``.
_MIN_RESULTANT_LENGTH = 1e-10
_MAX_CIRCULAR_STD = float(np.sqrt(-2.0 * np.log(_MIN_RESULTANT_LENGTH)))


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

    Notes
    -----
    Not delegated to ``scipy.stats.circmean(phases, high=pi, low=-pi)``: that
    agrees on the circle to round-off, but wraps with
    ``(x + pi) % (2 pi) - pi`` and so returns ``[-pi, pi)`` (a mean at ``+pi``
    becomes ``-pi``) and carries ``ulp(pi)``-sized absolute error (a mean of
    ``1e-20`` becomes ``0.0``). ``np.angle`` keeps ``(-pi, pi]`` and full
    relative precision.
    """
    return float(np.angle(np.mean(np.exp(1j * phases))))


def circular_std(phases: NDArray[np.floating]) -> float:
    """Compute the circular standard deviation of angles.

    ``std = sqrt(-2 log R)`` with ``R`` the mean resultant length (Mardia &
    Jupp, 2000), computed by ``scipy.stats.circstd`` (which also
    clips ``R <= 1`` against round-off).

    Parameters
    ----------
    phases : array
        Phase values in radians.

    Returns
    -------
    std : float
        Circular standard deviation in radians.

    Notes
    -----
    Delegates to ``scipy.stats.circstd(phases)`` (default ``low=0``,
    ``high=2 pi``, ``normalize=False``: result in radians), except that
    ``R`` is floored at ``1e-10``: the result is capped at
    ``sqrt(-2 log 1e-10) ~ 6.79`` rather than growing to ``inf`` for
    (numerically) uniform phases.
    """
    with np.errstate(divide="ignore"):  # R == 0 -> inf, then capped below
        std = stats.circstd(phases)
    return float(np.minimum(std, _MAX_CIRCULAR_STD))


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
    With ``z = n R^2`` the p-value is the second-order expansion (Greenwood &
    Durand, 1955; Zar, 1999)::

        p = exp(-z) * (1 + (2z - z^2) / (4n)
                         - (24z - 132z^2 + 76z^3 - 9z^4) / (288 n^2))

    clipped to ``[0, 1]``. It tends to the leading-order ``exp(-z)`` as
    ``n -> inf`` and is applied at every ``n``: switching to ``exp(-z)``
    above a cutoff would make the p-value jump there (by 14% at ``z = 6``
    for a cutoff at ``n = 50``). SciPy has no Rayleigh test
    (``scipy.stats.rayleigh`` is the Rayleigh distribution).

    References
    ----------
    Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics. Wiley.
    Greenwood, J. A. & Durand, D. (1955). The distribution of length and
    components of the sum of n random unit vectors. Ann. Math. Statist. 26.
    Zar, J. H. (1999). Biostatistical Analysis, 4th ed. Prentice Hall.

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

    p_value = float(
        np.exp(-z)
        * (
            1
            + (2 * z - z**2) / (4 * n)
            - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)
        )
    )
    p_value = max(0.0, min(1.0, p_value))  # Clip to [0, 1]

    return R, p_value


def circular_correlation(
    phases1: NDArray[np.floating],
    phases2: NDArray[np.floating],
) -> float:
    """Compute circular-circular correlation coefficient.

    Uses the Jammalamadaka-SenGupta circular correlation coefficient.

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
    The coefficient is defined as::

        r = sum(sin(a_i - a_bar) * sin(b_i - b_bar)) /
            sqrt(sum(sin(a_i - a_bar)^2) * sum(sin(b_i - b_bar)^2))

    with ``a_bar``, ``b_bar`` the circular means (Jammalamadaka & Sarma, 1988;
    Jammalamadaka & SenGupta, 2001; the ``circcorrcoef`` of
    astropy / pycircstat). It is *not* the Fisher-Lee (1983) coefficient, which
    is built from pairwise differences ``sin(a_i - a_j) sin(b_i - b_j)``.
    SciPy has no circular correlation.

    References
    ----------
    Jammalamadaka, S. R. & Sarma, Y. R. (1988). A correlation coefficient for
    angular variables. Statistical Theory and Data Analysis II, 349-364.
    Jammalamadaka, S. R. & SenGupta, A. (2001). Topics in Circular Statistics.
    World Scientific.
    """
    if len(phases1) != len(phases2):
        raise ValueError("phases1 and phases2 must have the same length")

    # Compute circular means
    mean1 = circular_mean(phases1)
    mean2 = circular_mean(phases2)

    # Compute centered phases
    sin_centered1 = np.sin(phases1 - mean1)
    sin_centered2 = np.sin(phases2 - mean2)

    # Jammalamadaka-SenGupta correlation
    numerator = np.sum(sin_centered1 * sin_centered2)
    denominator = np.sqrt(np.sum(sin_centered1**2) * np.sum(sin_centered2**2))

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


def _phases_at_spike_times(
    spike_times: NDArray[np.floating],
    inferred_phase: NDArray[np.floating],
    time_axis: NDArray[np.floating],
    mask: NDArray[np.bool_] | None,
) -> NDArray[np.floating]:
    """Phase at each spike time, restricted to masked, in-range spikes.

    Parameters
    ----------
    spike_times : array, shape (n_spikes,)
        Spike times in seconds.
    inferred_phase : array, shape (n_time,)
        Phase values at each time bin in radians.
    time_axis : array, shape (n_time,)
        Time axis corresponding to ``inferred_phase``.
    mask : array, shape (n_time,), or None
        Boolean mask of time points to include; None includes all.

    Returns
    -------
    spike_phases : array, shape (n_kept_spikes,)
        Phases of the spikes inside the mask and inside ``time_axis``.
    """
    from scipy.interpolate import interp1d

    # Nearest-neighbor interpolation avoids phase-wrapping artifacts.
    phase_interp = interp1d(
        time_axis, inferred_phase, kind="nearest", bounds_error=False, fill_value=np.nan
    )
    spike_phases: NDArray[np.floating] = phase_interp(spike_times)

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
    in_range: NDArray[np.floating] = spike_phases[~np.isnan(spike_phases)]
    return in_range


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
    spike_phases = _phases_at_spike_times(spike_times, inferred_phase, time_axis, mask)

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
    spike_phases = _phases_at_spike_times(spike_times, inferred_phase, time_axis, mask)

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
    distance: NDArray[np.floating] = np.abs(wrap_to_pi(np.asarray(phase1 - phase2)))
    return distance


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
