"""Structured regularization penalties for oscillator coupling parameters.

Provides sparsity and group-sparsity penalties on coupling_strength arrays
from DirectedInfluenceModel (and later CorrelatedNoiseModel). These penalties
are added to _sgd_loss_fn during SGD fitting — the EM path is unaffected.

Penalty families:
- edge_l1: smooth L1 on individual oscillator-to-oscillator coupling strengths
- area_group_l2: group L2 (Frobenius) on area-to-area pathway blocks
- state_shared_group_l2: group L2 that ties pathway selection across states

All penalties use smooth approximations (sqrt(x² + ε)) to avoid gradient
singularities at zero. Post-hoc thresholding can be applied after optimization
for exact sparsity.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass(frozen=True)
class OscillatorPenaltyConfig:
    """Configuration for oscillator connectivity penalties.

    Parameters
    ----------
    edge_l1 : float
        Weight for smooth L1 penalty on individual coupling strengths.
    area_group_l2 : float
        Weight for group L2 penalty on area-to-area pathway blocks.
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
    area_labels: Optional[Array] = None
    exclude_diagonal: bool = True
    scale_with_length: bool = False


def _mask_diagonal(coupling: Array, exclude: bool) -> Array:
    """Zero out diagonal entries if exclude=True."""
    if not exclude:
        return coupling
    n_osc = coupling.shape[-1]
    mask = 1.0 - jnp.eye(n_osc)
    return coupling * mask


def edge_l1_penalty(
    coupling: Array,
    eps: float = 1e-8,
    exclude_diagonal: bool = True,
) -> Array:
    """Smooth L1 penalty on individual coupling strengths.

    Parameters
    ----------
    coupling : Array, shape (n_states, n_osc, n_osc)
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
    c = _mask_diagonal(coupling, exclude_diagonal)
    return jnp.sum(jnp.sqrt(c**2 + eps))


def _build_area_pair_masks(area_labels: Array, n_areas: int) -> Array:
    """Build boolean masks for all area pairs.

    Returns
    -------
    Array, shape (n_areas, n_areas, n_osc, n_osc)
        masks[a, b, i, j] is True iff area_labels[i]==a and area_labels[j]==b.
    """
    areas = jnp.arange(n_areas)
    mask_rows = (area_labels[None, :] == areas[:, None])  # (n_areas, n_osc)
    # (n_areas, 1, n_osc, 1) * (1, n_areas, 1, n_osc) -> (n_areas, n_areas, n_osc, n_osc)
    return mask_rows[:, None, :, None] * mask_rows[None, :, None, :]


def area_group_penalty(
    coupling: Array,
    area_labels: Array,
    eps: float = 1e-8,
    exclude_diagonal: bool = True,
) -> Array:
    """Group L2 penalty on area-to-area pathway blocks.

    Sums the Frobenius norm of each area-pair block across all states.

    Parameters
    ----------
    coupling : Array, shape (n_states, n_osc, n_osc)
    area_labels : Array, shape (n_osc,)
    eps : float
    exclude_diagonal : bool

    Returns
    -------
    Array, shape ()
    """
    c = _mask_diagonal(coupling, exclude_diagonal)
    n_areas = int(np.asarray(area_labels).max()) + 1
    masks = _build_area_pair_masks(area_labels, n_areas)
    # (n_states, n_areas, n_areas): sum c^2 within each area block per state
    block_sq = jnp.einsum("sij,abij->sab", c ** 2, masks.astype(c.dtype))
    return jnp.sum(jnp.sqrt(block_sq + eps))


def state_shared_area_penalty(
    coupling: Array,
    area_labels: Array,
    eps: float = 1e-8,
    exclude_diagonal: bool = True,
) -> Array:
    """Group L2 penalty shared across discrete states.

    For each area pair (a, b), computes sqrt(sum_s ||C_ab^(s)||_F^2),
    encouraging the same pathways to be active or inactive across states.

    Parameters
    ----------
    coupling : Array, shape (n_states, n_osc, n_osc)
    area_labels : Array, shape (n_osc,)
    eps : float
    exclude_diagonal : bool

    Returns
    -------
    Array, shape ()
    """
    c = _mask_diagonal(coupling, exclude_diagonal)
    n_areas = int(np.asarray(area_labels).max()) + 1
    masks = _build_area_pair_masks(area_labels, n_areas)
    # Sum c^2 within each area block per state: (n_states, n_areas, n_areas)
    block_sq = jnp.einsum("sij,abij->sab", c ** 2, masks.astype(c.dtype))
    # Sum across states, then sqrt for group penalty
    return jnp.sum(jnp.sqrt(jnp.sum(block_sq, axis=0) + eps))


def get_area_coupling_summary(
    coupling: Array,
    area_labels: Array,
    exclude_diagonal: bool = True,
) -> dict:
    """Compute area-level coupling summary from oscillator coupling strengths.

    Parameters
    ----------
    coupling : Array, shape (n_states, n_osc, n_osc)
        Coupling strength array (states as first axis).
    area_labels : Array, shape (n_osc,)
        Integer area label per oscillator.
    exclude_diagonal : bool
        Exclude self-coupling from block norms.

    Returns
    -------
    dict with keys:
        block_norms : Array, shape (n_states, n_areas, n_areas)
            Frobenius norm of each area-pair coupling block per state.
        within_area_norm : Array, shape (n_states,)
            Total within-area coupling norm per state.
        cross_area_norm : Array, shape (n_states,)
            Total cross-area coupling norm per state.
    """
    n_states = coupling.shape[0]
    area_labels_np = np.asarray(area_labels)
    n_areas = int(area_labels_np.max()) + 1

    c = _mask_diagonal(coupling, exclude_diagonal)
    masks = _build_area_pair_masks(area_labels, n_areas)
    # (n_states, n_areas, n_areas)
    block_sq = jnp.einsum("sij,abij->sab", c ** 2, masks.astype(c.dtype))
    block_norms = jnp.sqrt(block_sq)

    within_mask = jnp.eye(n_areas, dtype=bool)
    within_area_norm = jnp.sum(block_norms * within_mask[None, :, :], axis=(-2, -1))
    cross_area_norm = jnp.sum(block_norms * ~within_mask[None, :, :], axis=(-2, -1))

    return {
        "block_norms": block_norms,
        "within_area_norm": within_area_norm,
        "cross_area_norm": cross_area_norm,
    }


def total_connectivity_penalty(
    coupling: Array,
    config: OscillatorPenaltyConfig,
    n_timesteps: int = 1,
) -> Array:
    """Compute the total regularization penalty from a config.

    Parameters
    ----------
    coupling : Array, shape (n_states, n_osc, n_osc)
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
    penalty = jnp.array(0.0)

    if config.edge_l1 > 0:
        penalty = penalty + config.edge_l1 * edge_l1_penalty(
            coupling, exclude_diagonal=config.exclude_diagonal
        )

    if config.area_group_l2 > 0:
        if config.area_labels is None:
            raise ValueError(
                "area_labels required when area_group_l2 > 0"
            )
        penalty = penalty + config.area_group_l2 * area_group_penalty(
            coupling, config.area_labels,
            exclude_diagonal=config.exclude_diagonal,
        )

    if config.state_shared_group_l2 > 0:
        if config.area_labels is None:
            raise ValueError(
                "area_labels required when state_shared_group_l2 > 0"
            )
        penalty = penalty + config.state_shared_group_l2 * state_shared_area_penalty(
            coupling, config.area_labels,
            exclude_diagonal=config.exclude_diagonal,
        )

    # When scale_with_length=False (default), multiply by T so that
    # after the mixin divides total loss by T, the effective penalty
    # is exactly lambda (length-invariant).
    if not config.scale_with_length:
        penalty = penalty * n_timesteps

    return penalty
