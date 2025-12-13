"""Preprocessing utilities for neural spike data.

This module provides functions for preparing neural data for state-space
model analysis, including spike binning, unit selection, and behavioral
epoch identification.

Note: These utilities use NumPy/SciPy for data preprocessing and are not
JIT-compatible. For JAX-compatible operations, see the core model modules.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike, NDArray


def bin_spike_times(
    spike_times: list[NDArray[np.floating]],
    time_bins: ArrayLike,
) -> NDArray[np.int_]:
    """Bin spike times into count matrix.

    Parameters
    ----------
    spike_times : list of arrays
        Spike times for each unit (in seconds). Each array contains the
        timestamps when that unit fired.
    time_bins : array-like
        Bin edges (left edges, uniform spacing assumed). The last bin
        extends to time_bins[-1] + bin_width where bin_width is inferred
        from the spacing.

    Returns
    -------
    spikes : array, shape (n_time, n_neurons)
        Spike counts per bin for each neuron. Note: Unlike latent state
        arrays (n_latent, n_time), spike observations use (n_time, n_neurons)
        following observation matrix conventions.

    Examples
    --------
    >>> spike_times = [np.array([0.1, 0.3, 0.5]), np.array([0.2, 0.4])]
    >>> time_bins = np.arange(0, 1.0, 0.2)  # 0, 0.2, 0.4, 0.6, 0.8
    >>> counts = bin_spike_times(spike_times, time_bins)
    >>> counts.shape
    (5, 2)
    """
    time_bins = np.asarray(time_bins)
    n_neurons = len(spike_times)
    n_time = len(time_bins)

    # Infer bin width from spacing
    bin_width = time_bins[1] - time_bins[0]

    # Create bin edges for histogram (need right edge too)
    bin_edges = np.append(time_bins, time_bins[-1] + bin_width)

    # Bin each neuron's spikes
    spikes = np.zeros((n_time, n_neurons), dtype=np.int_)
    for n, st in enumerate(spike_times):
        counts, _ = np.histogram(st, bins=bin_edges)
        spikes[:, n] = counts

    return spikes


def compute_firing_rates(
    spike_times: list[NDArray[np.floating]],
    start_time: float | None = None,
    end_time: float | None = None,
) -> NDArray[np.floating]:
    """Compute mean firing rate for each unit.

    Parameters
    ----------
    spike_times : list of arrays
        Spike times for each unit (in seconds).
    start_time : float | None, optional
        Start of analysis window. If None, uses minimum spike time across units.
    end_time : float | None, optional
        End of analysis window. If None, uses maximum spike time across units.

    Returns
    -------
    firing_rates : array, shape (n_neurons,)
        Mean firing rate (Hz) for each unit.
    """
    n_neurons = len(spike_times)

    # Handle edge case: no spikes in any unit
    non_empty_spikes = [st for st in spike_times if len(st) > 0]
    if len(non_empty_spikes) == 0:
        return np.zeros(n_neurons)

    # Determine time window
    all_spikes = np.concatenate(non_empty_spikes)
    if start_time is None:
        start_time = float(np.min(all_spikes))
    if end_time is None:
        end_time = float(np.max(all_spikes))

    duration = end_time - start_time
    if duration <= 0:
        return np.zeros(n_neurons)

    # Compute rates
    firing_rates = np.zeros(n_neurons)
    for n, st in enumerate(spike_times):
        # Count spikes in window
        n_spikes = np.sum((st >= start_time) & (st <= end_time))
        firing_rates[n] = n_spikes / duration

    return firing_rates


def select_units(
    spike_times: list[NDArray[np.floating]],
    min_rate: float = 0.0,
    max_rate: float = np.inf,
    start_time: float | None = None,
    end_time: float | None = None,
) -> NDArray[np.int_]:
    """Filter units by firing rate criteria.

    Parameters
    ----------
    spike_times : list of arrays
        Spike times for each unit (in seconds).
    min_rate : float, default=0.0
        Minimum firing rate (Hz) to include.
    max_rate : float, default=np.inf
        Maximum firing rate (Hz) to include.
    start_time : float | None, optional
        Start of analysis window for computing rates.
    end_time : float | None, optional
        End of analysis window for computing rates.

    Returns
    -------
    selected_indices : array
        Indices of units that meet the firing rate criteria.

    Examples
    --------
    >>> spike_times = [np.random.uniform(0, 10, 50) for _ in range(10)]
    >>> # Select units with firing rate between 1 and 20 Hz
    >>> selected = select_units(spike_times, min_rate=1.0, max_rate=20.0)
    """
    firing_rates = compute_firing_rates(spike_times, start_time, end_time)
    mask = (firing_rates >= min_rate) & (firing_rates <= max_rate)
    return np.where(mask)[0]


def identify_behavioral_bouts(
    speed: NDArray[np.floating],
    speed_threshold: float,
    min_duration: int,
    above_threshold: bool = True,
) -> list[tuple[int, int]]:
    """Find continuous epochs above or below speed threshold.

    Parameters
    ----------
    speed : array, shape (n_time,)
        Speed values at each time point.
    speed_threshold : float
        Speed threshold in same units as speed array.
    min_duration : int
        Minimum number of samples for a valid bout.
    above_threshold : bool, default=True
        If True, find epochs where speed > threshold (running).
        If False, find epochs where speed < threshold (immobility).

    Returns
    -------
    bout_list : list of (start_idx, end_idx)
        List of tuples with start and end indices for each bout.
        The end index is exclusive (Python convention).

    Examples
    --------
    >>> speed = np.array([1, 2, 10, 15, 12, 3, 2, 8, 9, 10, 1])
    >>> # Find running bouts (speed > 5) lasting at least 2 samples
    >>> bouts = identify_behavioral_bouts(speed, speed_threshold=5.0, min_duration=2)
    """
    if above_threshold:
        mask = speed > speed_threshold
    else:
        mask = speed < speed_threshold

    # Find transitions
    padded_mask = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded_mask.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Filter by duration
    bout_list = []
    for start, end in zip(starts, ends):
        if end - start >= min_duration:
            bout_list.append((int(start), int(end)))

    return bout_list


def create_behavioral_labels(
    speed: NDArray[np.floating],
    running_threshold: float = 5.0,
    immobility_threshold: float = 2.0,
) -> NDArray[np.int_]:
    """Create behavioral state labels from speed.

    Parameters
    ----------
    speed : array, shape (n_time,)
        Speed values at each time point (typically cm/s).
    running_threshold : float, default=5.0
        Speed above this is labeled as running.
    immobility_threshold : float, default=2.0
        Speed below this is labeled as immobility.

    Returns
    -------
    labels : array, shape (n_time,)
        Behavioral labels: 0=immobility, 1=running, 2=transition.

    Examples
    --------
    >>> speed = np.array([1, 3, 8, 10, 4, 1])
    >>> labels = create_behavioral_labels(speed)
    >>> # labels: [0, 2, 1, 1, 2, 0]
    """
    labels = np.full(len(speed), 2, dtype=np.int_)  # Default: transition
    labels[speed > running_threshold] = 1  # Running
    labels[speed < immobility_threshold] = 0  # Immobility
    return labels


def interpolate_to_new_times(
    values: NDArray[np.floating],
    original_times: NDArray[np.floating],
    new_times: NDArray[np.floating],
    kind: str = "linear",
) -> NDArray[np.floating]:
    """Interpolate values to new time points.

    Parameters
    ----------
    values : array, shape (n_original,) or (n_original, n_features)
        Values at original time points.
    original_times : array, shape (n_original,)
        Original time points.
    new_times : array, shape (n_new,)
        New time points to interpolate to.
    kind : str, default="linear"
        Interpolation method. Options: "linear", "nearest", "cubic".

    Returns
    -------
    interpolated : array, shape (n_new,) or (n_new, n_features)
        Values interpolated to new time points.
    """
    from scipy.interpolate import interp1d

    if values.ndim == 1:
        interpolator = interp1d(
            original_times, values, kind=kind, bounds_error=False, fill_value="extrapolate"
        )
        result: npt.NDArray[np.floating] = interpolator(new_times)
        return result
    else:
        # Handle multi-dimensional case
        n_features = values.shape[1]
        result_arr: npt.NDArray[np.floating] = np.zeros((len(new_times), n_features))
        for i in range(n_features):
            interpolator = interp1d(
                original_times,
                values[:, i],
                kind=kind,
                bounds_error=False,
                fill_value="extrapolate",
            )
            result_arr[:, i] = interpolator(new_times)
        return result_arr


def get_spike_times_subset(
    spike_times: list[NDArray[np.floating]],
    unit_indices: ArrayLike,
) -> list[NDArray[np.floating]]:
    """Extract spike times for a subset of units.

    Parameters
    ----------
    spike_times : list of arrays
        Spike times for all units.
    unit_indices : array-like
        Indices of units to extract.

    Returns
    -------
    subset : list of arrays
        Spike times for selected units only.
    """
    unit_indices = np.asarray(unit_indices)
    return [spike_times[i] for i in unit_indices]


def clip_spike_times_to_window(
    spike_times: list[NDArray[np.floating]],
    start_time: float,
    end_time: float,
) -> list[NDArray[np.floating]]:
    """Clip spike times to a time window.

    Parameters
    ----------
    spike_times : list of arrays
        Spike times for each unit (in seconds).
    start_time : float
        Start of time window.
    end_time : float
        End of time window.

    Returns
    -------
    clipped : list of arrays
        Spike times within the specified window.
    """
    clipped = []
    for st in spike_times:
        mask = (st >= start_time) & (st <= end_time)
        clipped.append(st[mask])
    return clipped


def binned_to_spike_times(
    binned_spikes: NDArray[np.int_],
    time_bins: ArrayLike,
    neuron_idx: int | None = None,
) -> NDArray[np.floating] | list[NDArray[np.floating]]:
    """Convert binned spike counts back to spike times.

    Parameters
    ----------
    binned_spikes : array, shape (n_time, n_neurons)
        Spike counts per bin.
    time_bins : array-like, shape (n_time,)
        Bin times (left edges).
    neuron_idx : int | None, optional
        If provided, return spike times for this neuron only.
        If None, return list of spike times for all neurons.

    Returns
    -------
    spike_times : array or list of arrays
        Spike times in same units as time_bins.
        If neuron_idx is provided, returns a single array.
        Otherwise, returns a list of arrays (one per neuron).

    Notes
    -----
    This function places all spikes at the left edge of their bin.
    For finer temporal resolution, use a smaller bin size.
    """
    time_bins = np.asarray(time_bins)

    def extract_spike_times_single(spike_counts: NDArray[np.int_]) -> NDArray[np.floating]:
        """Extract spike times for a single neuron."""
        spike_times_list = []
        for t_idx in np.where(spike_counts > 0)[0]:
            # Add spike time for each spike in the bin
            for _ in range(spike_counts[t_idx]):
                spike_times_list.append(time_bins[t_idx])
        return np.array(spike_times_list)

    if neuron_idx is not None:
        return extract_spike_times_single(binned_spikes[:, neuron_idx])
    else:
        n_neurons = binned_spikes.shape[1]
        return [extract_spike_times_single(binned_spikes[:, n]) for n in range(n_neurons)]
