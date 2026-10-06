"""Structured regularization penalties for oscillator coupling parameters.

Provides sparsity and group-sparsity penalties on coupling_strength arrays
from DirectedInfluenceModel (and later CorrelatedNoiseModel). These penalties
are added to _sgd_loss_fn during SGD fitting — the EM path is unaffected.

Penalty families:
- edge_l1: smooth L1 on individual oscillator-to-oscillator coupling strengths
- area_group_l2: group L2 (Frobenius) on area-to-area pathway blocks
- state_shared_group_l2: group L2 that ties pathway selection across states

All penalties use zero-centered smooth approximations
``sqrt(x² + ε) - sqrt(ε)`` to avoid gradient singularities at zero without
shifting the objective when all penalized couplings are zero. Post-hoc
thresholding can be applied after optimization for exact sparsity.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.utils import (
    contains_tracer,
    validate_finite_array,
    validate_int,
    validate_scalar,
)


def _validate_eps(eps: float) -> Array:
    """Validate smoothing epsilon and return it as a scalar array."""
    eps_arr = jnp.asarray(eps)
    if eps_arr.shape != ():
        raise ValueError(f"eps must be a scalar, got shape {eps_arr.shape}.")
    if not contains_tracer(eps_arr):
        eps_float = float(eps_arr)
        if not np.isfinite(eps_float) or eps_float <= 0.0:
            raise ValueError(f"eps must be positive and finite, got {eps}.")
    return eps_arr


def _as_coupling(coupling: ArrayLike) -> Array:
    """Convert and validate a coupling array with convention (state, osc, osc)."""
    arr = jnp.asarray(coupling)
    if arr.ndim != 3:
        raise ValueError(
            f"coupling must have shape (n_states, n_osc, n_osc), got {arr.shape}."
        )
    if arr.shape[-2] != arr.shape[-1]:
        raise ValueError(
            f"coupling must have square oscillator axes, got {arr.shape[-2:]}."
        )
    if not contains_tracer(arr):
        validate_finite_array("coupling", arr)
    return arr


def _validate_area_labels(area_labels: ArrayLike, n_osc: int | None = None) -> Array:
    """Validate contiguous integer area labels and return a JAX array."""
    labels_np = np.asarray(area_labels)
    if labels_np.ndim != 1:
        raise ValueError(
            f"area_labels must be a 1D integer array, got shape {labels_np.shape}."
        )
    if labels_np.size == 0:
        raise ValueError("area_labels must contain at least one label.")
    if n_osc is not None and labels_np.shape[0] != n_osc:
        raise ValueError(
            "area_labels length must match the number of oscillators; "
            f"got {labels_np.shape[0]} labels for n_osc={n_osc}."
        )
    if not np.issubdtype(labels_np.dtype, np.integer):
        raise ValueError("area_labels must contain integer labels.")
    if not np.all(np.isfinite(labels_np)):
        raise ValueError("area_labels must contain only finite labels.")
    if np.any(labels_np < 0):
        raise ValueError("area_labels must be non-negative integers.")

    unique_labels = np.unique(labels_np)
    expected = np.arange(int(unique_labels[-1]) + 1)
    if not np.array_equal(unique_labels, expected):
        raise ValueError(
            "area_labels must be contiguous integers starting at 0; "
            f"got labels {unique_labels.tolist()}."
        )
    return jnp.asarray(labels_np, dtype=jnp.int32)


def _area_labels_for_penalty(area_labels: ArrayLike, n_osc: int) -> Array:
    """Validate labels when possible; keep penalty helpers jit-compatible."""
    labels = jnp.asarray(area_labels)
    if labels.ndim != 1:
        raise ValueError(
            f"area_labels must be a 1D integer array, got shape {labels.shape}."
        )
    if labels.shape[0] != n_osc:
        raise ValueError(
            "area_labels length must match the number of oscillators; "
            f"got {labels.shape[0]} labels for n_osc={n_osc}."
        )
    if contains_tracer(labels):
        return labels
    return _validate_area_labels(labels, n_osc=n_osc)


@dataclass(frozen=True)
class OscillatorPenaltyConfig:
    """Configuration for oscillator connectivity penalties.

    Parameters
    ----------
    edge_l1 : float
        Weight for smooth L1 penalty on individual coupling strengths.
    area_group_l2 : float
        Weight for state-specific group L2 penalty on area-to-area pathway blocks.
    state_shared_group_l2 : float
        Weight for group L2 penalty shared across discrete states.
    area_labels : Array or None
        Integer array mapping oscillator index to brain area. Labels must
        be contiguous integers starting at 0 (e.g., [0, 0, 1, 1, 2]).
        Required when area_group_l2 > 0 or state_shared_group_l2 > 0.
    exclude_diagonal : bool
        If True (default), exclude self-coupling (diagonal) from penalties.
        Self-coupling represents damping, not inter-oscillator influence.
    scale_with_length : bool
        If False (default), the penalty is length-invariant: it is
        multiplied by n_timesteps internally so that the effective
        gradient contribution is lambda (not lambda / T). This means
        the same lambda produces the same regularization strength
        regardless of recording length.
        If True, the raw penalty is added to the loss without scaling,
        so longer recordings dilute the penalty (lambda_eff = lambda / T).
    """

    edge_l1: float = 0.0
    area_group_l2: float = 0.0
    state_shared_group_l2: float = 0.0
    area_labels: Array | None = None
    exclude_diagonal: bool = True
    scale_with_length: bool = False

    def __post_init__(self) -> None:
        for name in ("edge_l1", "area_group_l2", "state_shared_group_l2"):
            object.__setattr__(
                self,
                name,
                validate_scalar(getattr(self, name), name, nonnegative=True),
            )

        if self.area_labels is not None:
            object.__setattr__(
                self,
                "area_labels",
                _validate_area_labels(self.area_labels),
            )
        elif self.area_group_l2 > 0.0 or self.state_shared_group_l2 > 0.0:
            raise ValueError(
                "area_labels is required when area_group_l2 or "
                "state_shared_group_l2 > 0."
            )


def _mask_diagonal(coupling: Array, exclude: bool) -> Array:
    """Zero out diagonal entries if exclude=True."""
    if not exclude:
        return coupling
    n_osc = coupling.shape[-1]
    mask = 1.0 - jnp.eye(n_osc)
    return coupling * mask


def edge_l1_penalty(
    coupling: ArrayLike,
    eps: float = 1e-8,
    exclude_diagonal: bool = True,
) -> Array:
    """Zero-centered smooth L1 penalty on individual coupling strengths.

    Parameters
    ----------
    coupling : ArrayLike, shape (n_states, n_osc, n_osc)
        Coupling strength array.
    eps : float
        Smoothing constant to avoid gradient singularity at zero.
    exclude_diagonal : bool
        If True, exclude self-coupling (diagonal entries).

    Returns
    -------
    Array, shape ()
        Scalar penalty value.
    """
    c = _mask_diagonal(_as_coupling(coupling), exclude_diagonal)
    eps_arr = _validate_eps(eps).astype(c.dtype)
    return jnp.sum(jnp.sqrt(c**2 + eps_arr) - jnp.sqrt(eps_arr))


def _area_block_sum_sq(coupling: Array, area_labels: Array, n_areas: int) -> Array:
    """Sum of squared coupling within every area-pair block, per state.

    With the one-hot assignment ``M[a, i] = (area_labels[i] == a)`` this is
    ``M @ coupling**2 @ M.T`` -- ``O(n_areas * n_osc)`` extra memory, not the
    ``(n_areas, n_areas, n_osc, n_osc)`` pair masks of a direct einsum.

    Parameters
    ----------
    coupling : Array, shape (n_states, n_osc, n_osc)
    area_labels : Array, shape (n_osc,)
    n_areas : int
        Static number of area rows; labels ``>= n_areas`` are dropped.

    Returns
    -------
    Array, shape (n_states, n_areas, n_areas)
        ``out[s, a, b] = sum over i in area a, j in area b of coupling[s, i, j]**2``.
    """
    assignment = (area_labels[None, :] == jnp.arange(n_areas)[:, None]).astype(
        coupling.dtype
    )  # (n_areas, n_osc)
    return assignment @ coupling**2 @ assignment.T


def area_group_penalty(
    coupling: ArrayLike,
    area_labels: ArrayLike,
    eps: float = 1e-8,
    exclude_diagonal: bool = True,
) -> Array:
    """Group L2 penalty on area-to-area pathway blocks.

    Sums zero-centered smooth Frobenius norms of each state-specific
    area-pair block.

    Parameters
    ----------
    coupling : ArrayLike, shape (n_states, n_osc, n_osc)
    area_labels : ArrayLike, shape (n_osc,)
    eps : float
    exclude_diagonal : bool

    Returns
    -------
    Array, shape ()
    """
    c = _mask_diagonal(_as_coupling(coupling), exclude_diagonal)
    labels = _area_labels_for_penalty(area_labels, n_osc=c.shape[-1])
    eps_arr = _validate_eps(eps).astype(c.dtype)
    # (n_states, n_areas, n_areas): sum c^2 within each area block per state.
    # Use n_osc as a static upper bound on n_areas so this scalar penalty
    # remains jit-compatible when area_labels is a dynamic JAX argument. Valid
    # labels are still checked host-side when values are available; unused
    # area rows contribute exactly zero because the smooth norm is
    # zero-centered.
    block_sq = _area_block_sum_sq(c, labels, n_areas=c.shape[-1])
    return jnp.sum(jnp.sqrt(block_sq + eps_arr) - jnp.sqrt(eps_arr))


def state_shared_area_penalty(
    coupling: ArrayLike,
    area_labels: ArrayLike,
    eps: float = 1e-8,
    exclude_diagonal: bool = True,
) -> Array:
    """Group L2 penalty shared across discrete states.

    For each area pair (a, b), computes a zero-centered smooth version of
    sqrt(sum_s ||C_ab^(s)||_F^2), encouraging the same pathways to be active
    or inactive across states.

    Parameters
    ----------
    coupling : ArrayLike, shape (n_states, n_osc, n_osc)
    area_labels : ArrayLike, shape (n_osc,)
    eps : float
    exclude_diagonal : bool

    Returns
    -------
    Array, shape ()
    """
    c = _mask_diagonal(_as_coupling(coupling), exclude_diagonal)
    labels = _area_labels_for_penalty(area_labels, n_osc=c.shape[-1])
    eps_arr = _validate_eps(eps).astype(c.dtype)
    # Sum c^2 within each area block per state: (n_states, n_areas, n_areas),
    # with n_osc as the static upper bound on n_areas (see area_group_penalty).
    block_sq = _area_block_sum_sq(c, labels, n_areas=c.shape[-1])
    # Sum across states, then sqrt for group penalty
    return jnp.sum(jnp.sqrt(jnp.sum(block_sq, axis=0) + eps_arr) - jnp.sqrt(eps_arr))


def get_area_coupling_summary(
    coupling: ArrayLike,
    area_labels: ArrayLike,
    exclude_diagonal: bool = True,
) -> dict[str, Array]:
    """Compute area-level coupling summary from oscillator coupling strengths.

    Parameters
    ----------
    coupling : ArrayLike, shape (n_states, n_osc, n_osc)
        Coupling strength array (states as first axis).
    area_labels : ArrayLike, shape (n_osc,)
        Integer area label per oscillator.
    exclude_diagonal : bool
        Exclude self-coupling from block norms.

    Returns
    -------
    dict with keys:
        block_norms : Array, shape (n_states, n_areas, n_areas)
            Frobenius norm of each area-pair coupling block per state.
        within_area_norm : Array, shape (n_states,)
            Sum of within-area block Frobenius norms per state.
        cross_area_norm : Array, shape (n_states,)
            Sum of cross-area block Frobenius norms per state.
    """
    c = _mask_diagonal(_as_coupling(coupling), exclude_diagonal)
    labels = _validate_area_labels(area_labels, n_osc=c.shape[-1])
    n_areas = int(np.asarray(labels).max()) + 1

    # (n_states, n_areas, n_areas)
    block_norms = jnp.sqrt(_area_block_sum_sq(c, labels, n_areas))

    within_mask = jnp.eye(n_areas, dtype=bool)
    within_area_norm = jnp.sum(block_norms * within_mask[None, :, :], axis=(-2, -1))
    cross_area_norm = jnp.sum(block_norms * ~within_mask[None, :, :], axis=(-2, -1))

    return {
        "block_norms": block_norms,
        "within_area_norm": within_area_norm,
        "cross_area_norm": cross_area_norm,
    }


def total_connectivity_penalty(
    coupling: ArrayLike,
    config: OscillatorPenaltyConfig,
    n_timesteps: int = 1,
) -> Array:
    """Compute the total regularization penalty from a config.

    Parameters
    ----------
    coupling : ArrayLike, shape (n_states, n_osc, n_osc)
        Coupling strength array. For DirectedInfluenceModel, this is
        ``model.coupling_strength`` transposed to (n_states, n_osc, n_osc).
    config : OscillatorPenaltyConfig
        Penalty configuration with weights and area labels.
    n_timesteps : int
        Number of timesteps in the dataset. Used to make the penalty
        length-invariant when ``config.scale_with_length=False``.

    Returns
    -------
    Array, shape ()
        Total scalar penalty. When ``scale_with_length=False``, this is
        multiplied by ``n_timesteps`` so that after the mixin divides
        the total loss by T, the effective penalty is exactly lambda.
    """
    coupling = _as_coupling(coupling)
    n_timesteps = validate_int(n_timesteps, "n_timesteps", positive=True)

    penalty = jnp.array(0.0, dtype=coupling.dtype)

    if config.edge_l1 > 0:
        penalty = penalty + config.edge_l1 * edge_l1_penalty(
            coupling, exclude_diagonal=config.exclude_diagonal
        )

    if config.area_group_l2 > 0:
        if config.area_labels is None:
            raise ValueError("area_labels required when area_group_l2 > 0")
        penalty = penalty + config.area_group_l2 * area_group_penalty(
            coupling,
            config.area_labels,
            exclude_diagonal=config.exclude_diagonal,
        )

    if config.state_shared_group_l2 > 0:
        if config.area_labels is None:
            raise ValueError("area_labels required when state_shared_group_l2 > 0")
        penalty = penalty + config.state_shared_group_l2 * state_shared_area_penalty(
            coupling,
            config.area_labels,
            exclude_diagonal=config.exclude_diagonal,
        )

    # When scale_with_length=False (default), multiply by T so that
    # after the mixin divides total loss by T, the effective penalty
    # is exactly lambda (length-invariant).
    if not config.scale_with_length:
        penalty = penalty * n_timesteps

    return penalty
