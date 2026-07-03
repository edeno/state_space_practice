"""Bernoulli-logistic spike-field coupling state-space model.

The latent state is ``J`` independent oscillators, each a 2-D (real, imag)
rotation+decay block built from :mod:`oscillator_utils`, so the state vector is
ordered ``[re_0, im_0, re_1, im_1, ...]`` (band ``j`` occupies indices ``2j`` and
``2j + 1``). Spikes for neuron ``s`` at bin ``k`` are Bernoulli with logit

    eta_{s,k} = baseline_s + sum_j (beta_real[s, j] * re_j + beta_imag[s, j] * im_j)

so the complex coupling ``beta_real[s, j] + 1j * beta_imag[s, j]`` has magnitude
equal to coupling strength and angle equal to the preferred latent phase.

Shapes use ``S`` = neurons, ``J`` = oscillators (bands), ``T`` = time bins.
This module defines the model parameters, the simulator output container, and the
two deterministic maps (transition, logit) shared by the simulator and the
estimators. Requires float64 (the test suite enables ``jax_enable_x64``).
"""

from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import kalman_smoother
from state_space_practice.oscillator_utils import (
    construct_common_oscillator_process_covariance,
    construct_common_oscillator_transition_matrix,
)


class CouplingModelParams(NamedTuple):
    """Parameters of the Bernoulli-logistic coupling model.

    Parameters
    ----------
    osc_frequencies : Array, shape (J,)
        Oscillation frequency of each band in Hz.
    osc_decay : Array, shape (J,)
        Per-step amplitude decay (block radius) of each band, strictly in (0, 1).
        A value of 1 is an undamped oscillator with no stationary distribution
        (infinite variance), so it is rejected by :func:`validate_coupling_params`.
    process_noise_var : Array, shape (J,)
        Process-noise variance injected per band per step.
    beta_real, beta_imag : Array, shape (S, J)
        In-phase and quadrature coupling of each neuron to each band.
    baseline : Array, shape (S,)
        Logit intercept per neuron; sets the base rate via ``sigmoid(baseline)``.
    dt : float
        Seconds per time bin (sampling frequency is ``1 / dt``).
    history_kernel : Array or None, shape (S, L)
        Optional causal spike-history (refractory) kernel. Not yet implemented;
        must be ``None``. Kept in the contract so adding it later does not change
        the type. Placed last (after ``dt``) because NamedTuple requires
        defaulted fields to follow non-defaulted ones.
    lfp_noise_var : float
        Isotropic variance of the LFP observation noise. The latent is also
        observed as a "field": ``lfp_k = x_k + N(0, lfp_noise_var * I)`` (the
        measurement matrix is the identity). The LFP is what makes coupling
        inference well-posed — it pins the latent ``x`` so spikes only need to
        identify the coupling. (Observing the full state is an idealization; a
        lower-dimensional real-valued LFP projection is a future refinement.)
    """

    osc_frequencies: Array
    osc_decay: Array
    process_noise_var: Array
    beta_real: Array
    beta_imag: Array
    baseline: Array
    dt: float
    history_kernel: Optional[Array] = None
    lfp_noise_var: float = 0.25


class SimulatedCoupling(NamedTuple):
    """Simulator output and ground truth for scoring an estimator.

    Parameters
    ----------
    spikes : Array, shape (T, S)
        0/1 Bernoulli draws.
    lfp : Array, shape (T, 2J)
        Noisy field observation of the latent, ``x_k + N(0, lfp_noise_var * I)``.
    latent_true : Array, shape (T, 2J)
        True latent trajectory, ordered ``[re_0, im_0, ...]``.
    beta_real_true, beta_imag_true : Array, shape (S, J)
        True coupling coefficients.
    coupling_mask : Array of bool, shape (S, J)
        ``True`` exactly where a band is genuinely coupled to a neuron. Columns
        that are all-``False`` are false-positive control bands (they drive the
        latent but never enter any logit).
    baseline_true : Array, shape (S,)
        True logit intercepts.
    params : CouplingModelParams
        The parameters used to generate the data.
    seed : int
        PRNG seed used.
    """

    spikes: Array
    lfp: Array
    latent_true: Array
    beta_real_true: Array
    beta_imag_true: Array
    coupling_mask: Array
    baseline_true: Array
    params: CouplingModelParams
    seed: int


def validate_coupling_params(params: CouplingModelParams) -> None:
    """Validate the cross-field shape and range invariants; raise on violation.

    Enforces J consistency (``osc_*`` and the second axis of ``beta_*``), S
    consistency (first axis of ``beta_*`` and ``baseline``), and the ranges
    required for a well-defined stationary model: ``0 < osc_decay < 1`` (a value
    of 1 gives an undamped oscillator with no stationary distribution, which would
    silently produce ``inf``/``NaN``), ``process_noise_var >= 0`` (a negative
    variance would silently ``NaN`` through ``sqrt``), ``dt > 0``, and finite
    coupling/baseline. These are static-shape / concrete-value checks meant for
    the simulator and estimator entry points, not for inside a ``jit``/``vmap``
    hot path.

    Raises
    ------
    ValueError
        If any invariant is violated, with a message naming the offending field.
    """
    freq_shape = tuple(np.shape(params.osc_frequencies))
    beta_real_shape = tuple(np.shape(params.beta_real))
    if len(freq_shape) != 1 or freq_shape[0] == 0:
        raise ValueError(
            f"osc_frequencies must have shape (J,), got {freq_shape}."
        )
    if len(beta_real_shape) != 2 or 0 in beta_real_shape:
        raise ValueError(
            f"beta_real must have shape (S, J), got {beta_real_shape}."
        )
    n_bands = int(freq_shape[0])
    n_neurons = int(beta_real_shape[0])
    expected_shapes = {
        "osc_frequencies": ((n_bands,), params.osc_frequencies),
        "osc_decay": ((n_bands,), params.osc_decay),
        "process_noise_var": ((n_bands,), params.process_noise_var),
        "beta_real": ((n_neurons, n_bands), params.beta_real),
        "beta_imag": ((n_neurons, n_bands), params.beta_imag),
        "baseline": ((n_neurons,), params.baseline),
    }
    for name, (shape, arr) in expected_shapes.items():
        if tuple(np.shape(arr)) != shape:
            raise ValueError(
                f"{name} has shape {tuple(np.shape(arr))}, expected {shape} "
                f"(S={n_neurons}, J={n_bands})"
            )
        arr_np = np.asarray(arr)
        if (
            not np.issubdtype(arr_np.dtype, np.number)
            or np.issubdtype(arr_np.dtype, np.complexfloating)
        ):
            raise ValueError(f"{name} must be real-valued numeric data.")
        if not np.all(np.isfinite(arr_np)):
            raise ValueError(f"{name} contains non-finite values")

    decay = np.asarray(params.osc_decay)
    if not np.all((decay > 0.0) & (decay < 1.0)):
        raise ValueError(
            "osc_decay must be strictly within (0, 1); a value of 1 has no "
            "stationary distribution"
        )
    if np.any(np.asarray(params.process_noise_var) < 0.0):
        raise ValueError("process_noise_var must be nonnegative")
    for name, value in (
        ("dt", params.dt),
        ("lfp_noise_var", params.lfp_noise_var),
    ):
        value_arr = np.asarray(value)
        if (
            value_arr.shape != ()
            or not np.issubdtype(value_arr.dtype, np.number)
            or np.issubdtype(value_arr.dtype, np.complexfloating)
        ):
            raise ValueError(f"{name} must be a real-valued numeric scalar, got {value}.")
        value_float = float(value_arr)
        if not np.isfinite(value_float) or value_float <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}.")


def validate_coupling_observations(
    spikes: ArrayLike,
    lfp: ArrayLike,
    *,
    n_neurons: int,
    n_latent: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate spike/LFP arrays shared by coupling estimators."""
    spikes_np = np.asarray(spikes)
    if spikes_np.ndim != 2 or spikes_np.shape[1] != n_neurons:
        raise ValueError(
            f"spikes must have shape (T, {n_neurons}); got {tuple(spikes_np.shape)}"
        )
    if spikes_np.shape[0] == 0:
        raise ValueError("spikes and lfp must contain at least one time bin.")
    if not (
        (
            np.issubdtype(spikes_np.dtype, np.number)
            and not np.issubdtype(spikes_np.dtype, np.complexfloating)
        )
        or np.issubdtype(spikes_np.dtype, np.bool_)
    ):
        raise ValueError("spikes must be finite 0/1 indicators")
    spikes_float = spikes_np.astype(float)
    if not np.all(np.isfinite(spikes_float)) or not np.all(
        (spikes_float == 0.0) | (spikes_float == 1.0)
    ):
        raise ValueError("spikes must be finite 0/1 indicators")

    lfp_np = np.asarray(lfp)
    if lfp_np.shape != (spikes_np.shape[0], n_latent):
        raise ValueError(
            f"lfp must have shape ({spikes_np.shape[0]}, {n_latent}); "
            f"got {tuple(lfp_np.shape)}"
        )
    if not np.issubdtype(lfp_np.dtype, np.number) or np.issubdtype(
        lfp_np.dtype, np.complexfloating
    ):
        raise ValueError("lfp must be finite")
    lfp_float = lfp_np.astype(float)
    if not np.all(np.isfinite(lfp_float)):
        raise ValueError("lfp must be finite")
    return spikes_float, lfp_float


def build_transition(params: CouplingModelParams) -> tuple[Array, Array]:
    """Build the latent transition matrix and process covariance.

    Returns
    -------
    transition_matrix : Array, shape (2J, 2J)
        Block-diagonal oscillator transition (rotation + decay per band).
    process_covariance : Array, shape (2J, 2J)
        Diagonal process covariance ``diag(repeat(process_noise_var, 2))``.
    """
    transition_matrix = construct_common_oscillator_transition_matrix(
        jnp.asarray(params.osc_frequencies),
        jnp.asarray(params.osc_decay),
        1.0 / params.dt,
    )
    process_covariance = construct_common_oscillator_process_covariance(
        jnp.asarray(params.process_noise_var)
    )
    return transition_matrix, process_covariance


def logit(state: ArrayLike, params: CouplingModelParams) -> Array:
    """Per-neuron logit (linear predictor) of the spike probability at one bin.

    Parameters
    ----------
    state : ArrayLike, shape (2J,)
        Latent state ``[re_0, im_0, ...]`` at a single time bin.
    params : CouplingModelParams

    Returns
    -------
    eta : Array, shape (S,)
        ``baseline + beta_real @ re + beta_imag @ im``.
    """
    state = jnp.asarray(state)
    re = state[0::2]  # (J,)
    im = state[1::2]  # (J,)
    return params.baseline + params.beta_real @ re + params.beta_imag @ im


def smooth_latent_from_lfp(lfp: ArrayLike, params: CouplingModelParams) -> Array:
    """Kalman-smooth the latent from the LFP (``H = I``, ``R = lfp_noise_var I``).

    Stage 1 of both coupling estimators: the field observes the latent, so this is
    a linear-Gaussian RTS smoother (no spikes, no bilinear degeneracy). The initial
    covariance is the oscillator stationary covariance. Returns only the smoothed
    mean — using it as a fixed design for the coupling regression is the plug-in
    approximation (it ignores the smoother's posterior uncertainty in ``x``).

    Parameters
    ----------
    lfp : ArrayLike, shape (T, 2J)
    params : CouplingModelParams

    Returns
    -------
    smoothed_latent : Array, shape (T, 2J)
    """
    transition_matrix, process_cov = build_transition(params)
    n_latent = transition_matrix.shape[0]
    stationary_var = jnp.asarray(params.process_noise_var) / (
        1.0 - jnp.asarray(params.osc_decay) ** 2
    )
    init_cov = jnp.diag(jnp.repeat(stationary_var, 2))
    smoothed_latent, *_ = kalman_smoother(
        jnp.zeros(n_latent),
        init_cov,
        jnp.asarray(lfp),
        transition_matrix,
        process_cov,
        jnp.eye(n_latent),
        params.lfp_noise_var * jnp.eye(n_latent),
    )
    return smoothed_latent


def interleave_coupling(beta_real: ArrayLike, beta_imag: ArrayLike) -> Array:
    """Pack ``(S, J)`` real/imag coupling blocks into interleaved design rows.

    The latent state is ordered ``[re_0, im_0, re_1, im_1, ...]``, so the matching
    coupling "design row" for neuron ``s`` is
    ``[beta_real[s,0], beta_imag[s,0], beta_real[s,1], beta_imag[s,1], ...]``.
    Then ``design_row @ state`` equals the coupling term of the logit (the
    per-band sum ``sum_j beta_real[s,j] re_j + beta_imag[s,j] im_j``).

    This is the single source of truth for the real/imag <-> interleaved
    conversion, shared by the simulator, the EKF augmented state, and the PG
    design, so the four sites cannot drift apart.

    Parameters
    ----------
    beta_real, beta_imag : ArrayLike, shape (S, J)

    Returns
    -------
    design : Array, shape (S, 2J)
        Interleaved coupling, columns ``[bR_0, bI_0, bR_1, bI_1, ...]``.
    """
    beta_real = jnp.asarray(beta_real)
    beta_imag = jnp.asarray(beta_imag)
    n_neurons, n_bands = beta_real.shape
    return jnp.stack([beta_real, beta_imag], axis=-1).reshape(n_neurons, 2 * n_bands)


def deinterleave_coupling(design: ArrayLike) -> tuple[Array, Array]:
    """Inverse of :func:`interleave_coupling`: split interleaved rows into blocks.

    Parameters
    ----------
    design : ArrayLike, shape (S, 2J)
        Interleaved coupling, columns ``[bR_0, bI_0, bR_1, bI_1, ...]``.

    Returns
    -------
    beta_real, beta_imag : Array, shape (S, J)
    """
    design = jnp.asarray(design)
    n_neurons, n_interleaved = design.shape
    blocks = design.reshape(n_neurons, n_interleaved // 2, 2)
    return blocks[..., 0], blocks[..., 1]
