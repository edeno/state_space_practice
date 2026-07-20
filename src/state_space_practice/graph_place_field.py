"""Graph-Laplacian spatial substrate for a drifting place-field model.

This module is the *substrate seam* between ``neurospatial`` (which owns the spatial
environment — bins, connectivity, geodesic geometry) and the state-space inference in
this package. It turns a fitted ``neurospatial.Environment`` into a compact, geometry
-aware spatial basis (the smoothest eigenvectors of the environment's graph Laplacian)
plus the helpers a Poisson place-field model needs: a per-time design matrix, a per-bin
Poisson exposure (occupancy), and per-bin spike counts.

All per-bin arrays are over the environment's **active** bins (``env.n_bins`` = rows of
``env.bin_centers``); ``env.bin_sequence`` returns indices in this same ``0..n_bins-1``
space (or ``-1`` for out-of-bounds samples). There is no separate interior/full-grid
array here — scattering to a dense plotting grid is a ``neurospatial`` concern.

Laplacian choice
----------------
The convention is explicit rather than hidden inside the basis builder:

``"distance"`` (default)
    The public ``Environment.get_differential_operator()`` convention,
    ``L = D @ D.T``. Its edge weights are ``distance`` because ``D`` uses
    ``sqrt(distance)`` per edge. This preserves the original substrate behavior.
``"inverse_distance"``
    A conductance graph whose edge weights are ``1 / distance``. This is useful
    for sensitivity checks because nearby bins are coupled more strongly.

Both constructions are validated as finite symmetric graph Laplacians and the
selected convention is stored in ``GraphBasis``. ``neurospatial``'s own diffusion
smoothing uses a different finite-volume operator that is currently exposed only
through a private API. It is deliberately not copied here under a misleading
public name.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse as sp
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

from state_space_practice.kalman import psd_solve, symmetrize
from state_space_practice.point_process_kalman import log_conditional_intensity
from state_space_practice.sgd_fitting import SGDFittableMixin

if TYPE_CHECKING:
    from neurospatial import Environment

logger = logging.getLogger(__name__)

__all__ = [
    "GraphBasis",
    "LaplacianConvention",
    "SUPPORTED_LAPLACIAN_CONVENTIONS",
    "build_graph_laplacian",
    "build_graph_basis",
    "spectral_shape",
    "graph_design_matrix",
    "bin_occupancy",
    "bin_spike_counts",
    "validate_graph_laplacian",
    "spectral_precision",
    "parity_penalty",
    "fit_static_graph_glm",
    "static_log_evidence",
    "select_tau2_by_evidence",
    "GraphPlaceFieldModel",
]

# Attribute used to cache the full eigensystem on an Environment instance (keyed by the
# Laplacian's identity so a re-fit environment does not reuse a stale basis).
_CACHE_ATTR = "_graph_place_field_basis_cache"

LaplacianConvention = Literal["distance", "inverse_distance"]
SUPPORTED_LAPLACIAN_CONVENTIONS: tuple[LaplacianConvention, ...] = (
    "distance",
    "inverse_distance",
)


class GraphBasis(NamedTuple):
    """Truncated graph-Laplacian eigenbasis over an environment's active bins.

    Attributes
    ----------
    eigvecs : NDArray, shape (n_bins, rank)
        Smoothest eigenvectors of the selected graph Laplacian as columns,
        ordered by ascending eigenvalue. Row ``i`` is active bin ``i`` (same
        space as ``env.bin_sequence``). On a disconnected graph the eigenvectors
        are component-local (zero outside their connected component).
    eigvals : NDArray, shape (rank,)
        Corresponding eigenvalues (ascending, clipped to be non-negative). The first
        ``n_components`` entries are the null modes (0 up to eigensolver round-off;
        ``_full_eigensystem`` clips only negative round-off, so a tiny positive
        round-off value may remain).
    component_labels : NDArray, shape (n_bins,)
        Connected-component id (``0..n_components-1``) for each active bin.
    bin_sizes : NDArray, shape (n_bins,)
        Per-bin volume (``env.bin_sizes``) for integration/density. **Not** exposure.
    n_components : int
        Number of connected components (= number of retained null modes).
    laplacian_convention : {"distance", "inverse_distance"}
        Edge-weight convention used to construct the Laplacian and eigenbasis.
    env_key : tuple
        Identity fingerprint ``(convention, n_bins, nnz, |L|-sum)`` of the
        environment's Laplacian this basis was built from. Consumers refuse a
        basis built for a different environment or convention.
    """

    eigvecs: NDArray[np.float64]
    eigvals: NDArray[np.float64]
    component_labels: NDArray[np.int_]
    bin_sizes: NDArray[np.float64]
    n_components: int
    laplacian_convention: LaplacianConvention
    env_key: tuple[str, int, int, float]


def _validate_laplacian_convention(convention: str) -> LaplacianConvention:
    """Return a supported convention or raise at the public boundary."""
    if convention not in SUPPORTED_LAPLACIAN_CONVENTIONS:
        supported = ", ".join(repr(value) for value in SUPPORTED_LAPLACIAN_CONVENTIONS)
        raise ValueError(
            f"laplacian convention must be one of {supported}; got {convention!r}. "
            "The finite-volume operator is not exposed by neurospatial's public API."
        )
    return cast(LaplacianConvention, convention)


def validate_graph_laplacian(
    laplacian: sp.spmatrix | NDArray[np.float64],
    *,
    n_bins: Optional[int] = None,
    atol: float = 1e-10,
) -> None:
    """Validate the structural invariants of a weighted graph Laplacian.

    A finite symmetric matrix with zero row sums, non-negative diagonal, and
    non-positive off-diagonal entries is a symmetric diagonally dominant graph
    Laplacian and therefore positive semidefinite. These structural checks avoid
    performing another dense eigendecomposition solely for validation.
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError(f"atol must be finite and non-negative, got {atol}.")
    matrix = sp.csr_matrix(laplacian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"graph Laplacian must be square; got shape {matrix.shape}.")
    if matrix.shape[0] == 0:
        raise ValueError("graph Laplacian must contain at least one bin.")
    if n_bins is not None and matrix.shape != (n_bins, n_bins):
        raise ValueError(
            f"graph Laplacian must have shape ({n_bins}, {n_bins}); got {matrix.shape}."
        )
    if matrix.data.size and not np.all(np.isfinite(matrix.data)):
        raise ValueError("graph Laplacian must contain only finite values.")

    max_abs = float(np.max(np.abs(matrix.data))) if matrix.data.size else 0.0
    threshold = atol * max(1.0, max_abs)

    skew = matrix - matrix.T
    skew_max = float(np.max(np.abs(skew.data))) if skew.data.size else 0.0
    if skew_max > threshold:
        raise ValueError(
            f"graph Laplacian must be symmetric; maximum asymmetry is {skew_max:.3g}."
        )

    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    row_sum_max = float(np.max(np.abs(row_sums)))
    if row_sum_max > threshold:
        raise ValueError(
            "graph Laplacian row sums must be zero; "
            f"maximum absolute row sum is {row_sum_max:.3g}."
        )

    diagonal = matrix.diagonal()
    if np.any(diagonal < -threshold):
        raise ValueError("graph Laplacian diagonal entries must be non-negative.")
    coo = matrix.tocoo()
    off_diagonal = coo.data[coo.row != coo.col]
    if off_diagonal.size and np.any(off_diagonal > threshold):
        raise ValueError("graph Laplacian off-diagonal entries must be non-positive.")


def _inverse_distance_laplacian(env: "Environment") -> sp.csr_matrix:
    """Construct a conductance Laplacian with edge weight ``1 / distance``."""
    graph = env.connectivity.copy()
    if graph.is_directed():
        raise ValueError("environment connectivity must be an undirected graph.")
    weight_name = "_state_space_practice_inverse_distance"
    for source, target, edge_data in graph.edges(data=True):
        if "distance" not in edge_data:
            raise ValueError(
                f"edge ({source}, {target}) is missing the required 'distance' "
                "attribute for inverse-distance weighting."
            )
        distance = float(edge_data["distance"])
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError(
                f"edge ({source}, {target}) distance must be positive and finite; "
                f"got {distance}."
            )
        edge_data[weight_name] = 1.0 / distance
    return sp.csr_matrix(
        nx.laplacian_matrix(
            graph,
            nodelist=range(env.n_bins),
            weight=weight_name,
        ),
        dtype=float,
    )


def build_graph_laplacian(
    env: "Environment",
    convention: LaplacianConvention = "distance",
) -> sp.csr_matrix:
    """Build and validate the environment graph Laplacian.

    ``"distance"`` uses the public ``D @ D.T`` operator and preserves the
    original substrate behavior. ``"inverse_distance"`` treats reciprocal edge
    distance as graph conductance.
    """
    convention = _validate_laplacian_convention(convention)
    if convention == "distance":
        differential_operator = env.get_differential_operator()
        laplacian = sp.csr_matrix(
            differential_operator @ differential_operator.T, dtype=float
        )
    else:
        laplacian = _inverse_distance_laplacian(env)
    validate_graph_laplacian(laplacian, n_bins=env.n_bins)
    return laplacian


def _distance_weighted_laplacian(env: "Environment") -> sp.csr_matrix:
    """Compatibility helper for the default ``L = D @ D.T`` convention."""
    return build_graph_laplacian(env, convention="distance")


def _laplacian_key(
    laplacian: sp.spmatrix, convention: LaplacianConvention
) -> tuple[str, int, int, float]:
    """Cheap identity fingerprint including the selected convention.

    ``abs`` because a graph Laplacian's signed entries sum to ~0; the absolute-value
    sum distinguishes different edge weightings at the same sparsity pattern.
    """
    return (
        convention,
        int(laplacian.shape[0]),
        int(laplacian.nnz),
        float(np.round(np.abs(laplacian.data).sum(), 6)),
    )


def _env_key(
    env: "Environment", convention: LaplacianConvention
) -> tuple[str, int, int, float]:
    """Fingerprint of ``env`` under the requested Laplacian convention."""
    return _laplacian_key(build_graph_laplacian(env, convention), convention)


def _check_basis_matches_env(env: "Environment", basis: GraphBasis) -> None:
    """Refuse a basis built for a different (or since-refit) environment.

    The basis rows and the ``bin_sequence`` ids the consumers index by must belong to
    the same environment; otherwise the design rows / counts are silently wrong (or,
    if ``env`` has more bins than the basis, a bare ``IndexError``). Cheap bin-count
    check first, then the Laplacian fingerprint (catches a same-size re-fit).
    """
    if basis.eigvecs.shape[0] != env.n_bins:
        raise ValueError(
            f"basis has {basis.eigvecs.shape[0]} bins but env has {env.n_bins} "
            "active bins; the basis was built for a different environment. "
            "Rebuild it with build_graph_basis(env)."
        )
    env_key = _env_key(env, basis.laplacian_convention)
    if basis.env_key != env_key:
        raise ValueError(
            "basis.env_key does not match this environment's Laplacian "
            f"({basis.env_key} != {env_key}); the basis was built for a different "
            "(or since-refit) environment. Rebuild it with build_graph_basis(env)."
        )


def _resolve_rank(
    eigvals: NDArray[np.float64],
    n_components: int,
    rank: Optional[int],
    sigma: Optional[float],
    tol: float,
) -> int:
    """Resolve the number of modes to keep.

    An explicit ``rank`` that is too small **raises** (never a silent clamp); only the
    bandwidth-driven auto-rank is floored at ``n_components`` so every null mode is
    retained.
    """
    n_available = int(eigvals.shape[0])
    if rank is not None:
        rank = int(rank)
        if rank < n_components:
            raise ValueError(
                f"rank={rank} < n_components={n_components} would drop a null mode; "
                f"raise rank to at least {n_components} (or use rank=None)."
            )
        return min(rank, n_available)
    if sigma is None:
        return n_available
    if not sigma > 0:
        raise ValueError(f"sigma (smoothing bandwidth) must be positive, got {sigma}.")
    if not 0.0 < tol < 1.0:
        raise ValueError(f"tol must lie in (0, 1), got {tol}.")
    # Keep every mode whose heat-kernel weight exp(-(sigma**2/2) * lambda) >= tol.
    lambda_cut = -np.log(tol) / (sigma**2 / 2.0)
    keep = int(np.searchsorted(eigvals, lambda_cut, side="right"))
    return max(keep, n_components)


def _full_eigensystem(
    laplacian: sp.spmatrix, labels: NDArray[np.int_], n_components: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Component-local dense eigendecomposition of the graph Laplacian.

    Each connected component is decomposed on its own so every eigenvector is localized
    to a single component (zero elsewhere); the null space is then exactly one constant
    mode per component. Returns all ``n_bins`` modes sorted by ascending eigenvalue.
    """
    n_bins = laplacian.shape[0]
    laplacian = laplacian.tocsr()
    val_parts: list[NDArray[np.float64]] = []
    vec_parts: list[NDArray[np.float64]] = []
    for component in range(n_components):
        idx = np.flatnonzero(labels == component)
        block = laplacian[idx][:, idx].toarray()
        block = 0.5 * (block + block.T)  # symmetrize away round-off
        w, v = scipy.linalg.eigh(block)
        w = np.clip(w, 0.0, None)
        padded = np.zeros((n_bins, v.shape[1]))
        padded[idx] = v
        val_parts.append(w)
        vec_parts.append(padded)
    eigvals = np.concatenate(val_parts)
    eigvecs = np.concatenate(vec_parts, axis=1)
    order = np.argsort(eigvals, kind="stable")
    return eigvals[order], eigvecs[:, order]


def build_graph_basis(
    env: "Environment",
    rank: Optional[int] = None,
    *,
    sigma: Optional[float] = None,
    tol: float = 1e-6,
    laplacian_convention: LaplacianConvention = "distance",
) -> GraphBasis:
    """Build (and cache) the truncated graph-Laplacian eigenbasis for ``env``.

    The default uses the public distance-weighted Laplacian ``L = D @ D.T`` from
    ``env.get_differential_operator()``; ``laplacian_convention`` can instead
    request inverse-distance conductance (see the module docstring). The basis is
    built locally because ``neurospatial`` does not expose its diffusion
    eigenbasis publicly.

    Parameters
    ----------
    env : neurospatial.Environment
        A fitted environment.
    rank : int or None, optional
        Number of smoothest modes to keep. ``None`` (default) keeps all modes unless
        ``sigma`` is given. An explicit ``rank`` below the number of connected
        components raises. ``rank`` is capped at ``env.n_bins``.
    sigma : float or None, optional
        Smoothing bandwidth (coordinate units). When ``rank is None`` and ``sigma`` is
        set, the rank is chosen so every mode with heat-kernel weight
        ``exp(-(sigma**2/2) * lambda) >= tol`` is kept (floored at ``n_components``).
    tol : float, optional
        Heat-kernel weight cutoff for bandwidth-driven truncation, by default 1e-6.
    laplacian_convention : {"distance", "inverse_distance"}, optional
        Explicit edge-weight convention. The default ``"distance"`` preserves
        the original ``Environment.get_differential_operator()`` behavior.

    Returns
    -------
    GraphBasis
        The truncated eigenbasis; arrays are read-only and cached on ``env``.
    """
    laplacian_convention = _validate_laplacian_convention(laplacian_convention)
    laplacian = build_graph_laplacian(env, laplacian_convention)
    n_components, labels = connected_components(laplacian, directed=False)

    # Cache the full sorted eigensystem keyed by the Laplacian's identity (shape + nnz +
    # data checksum); a cached full basis serves any smaller rank by slicing.
    cache = getattr(env, _CACHE_ATTR, None)
    key = _laplacian_key(laplacian, laplacian_convention)
    if cache is None or cache.get("key") != key:
        eigvals, eigvecs = _full_eigensystem(laplacian, labels, int(n_components))
        cache = {"key": key, "eigvals": eigvals, "eigvecs": eigvecs, "labels": labels}
        try:
            setattr(env, _CACHE_ATTR, cache)
        except AttributeError:  # environment forbids attribute assignment; skip caching
            pass

    eigvals_full = cache["eigvals"]
    keep = _resolve_rank(eigvals_full, int(n_components), rank, sigma, tol)

    # np.array (a copy), not np.asarray: freezing below must not make the
    # environment's own borrowed bin_sizes array read-only.
    bin_sizes = np.array(env.bin_sizes, dtype=float)
    basis = GraphBasis(
        eigvecs=cache["eigvecs"][:, :keep],
        eigvals=eigvals_full[:keep],
        component_labels=np.asarray(cache["labels"]),
        bin_sizes=bin_sizes,
        n_components=int(n_components),
        laplacian_convention=laplacian_convention,
        env_key=key,
    )
    for arr in (basis.eigvecs, basis.eigvals, basis.component_labels, basis.bin_sizes):
        arr.setflags(write=False)
    return basis


def spectral_shape(
    eigvals: NDArray[np.float64], kappa2: float, alpha: float = 1.0
) -> NDArray[np.float64]:
    """Spectral-Matérn diagonal shape ``S = (kappa2 + lambda) ** (-alpha)``.

    ``kappa2 > 0`` keeps every entry finite, including the null modes (``lambda = 0`` ->
    ``kappa2 ** (-alpha)``), so the singular ``1 / lambda`` is never formed. This is the
    diagonal (in the eigenbasis) shape shared by the prior ``P0 = tau2 * S`` and the
    per-neuron drift ``Q_c = q_c * S``.

    Parameters
    ----------
    eigvals : NDArray, shape (rank,)
        Laplacian eigenvalues (non-negative).
    kappa2 : float
        Inverse-lengthscale squared; must be positive.
    alpha : float, optional
        Smoothness exponent, by default 1.0; must be positive.

    Returns
    -------
    NDArray, shape (rank,)
        The strictly positive spectral shape.
    """
    if not kappa2 > 0:
        raise ValueError(
            f"kappa2 (inverse-lengthscale^2) must be positive, got {kappa2}."
        )
    if not alpha > 0:
        raise ValueError(f"alpha (smoothness) must be positive, got {alpha}.")
    return (kappa2 + np.asarray(eigvals, dtype=float)) ** (-alpha)


def _bin_ids(
    env: "Environment",
    times: NDArray[np.float64],
    trajectory: NDArray[np.float64],
) -> NDArray[np.int_]:
    """Per-sample active-bin ids on the ``times`` grid (``-1`` out-of-bounds).

    ``dedup=False`` is required: the ``bin_sequence`` default (``dedup=True``) collapses
    consecutive repeats, returning fewer rows than ``n_time`` and misaligning spikes /
    design matrix / time.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory[:, None]
    # neurospatial types bin_sequence with a ``Self: EnvironmentProtocol`` bound
    # that mypy does not recognize the concrete Environment as satisfying; the
    # call is valid at runtime.
    ids = env.bin_sequence(  # type: ignore[misc]
        np.asarray(times, dtype=float), trajectory, dedup=False, outside_value=-1
    )
    return np.asarray(ids, dtype=np.int_)


def graph_design_matrix(
    env: "Environment",
    basis: GraphBasis,
    times: NDArray[np.float64],
    trajectory: NDArray[np.float64],
    *,
    interpolation: str = "nearest",
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Per-time design matrix ``Z`` and a ``valid`` row mask.

    ``Z[t] = Phi[bin_id(t)]`` — the basis evaluated at the animal's bin at time ``t``.
    Bin ids come from ``env.bin_sequence(..., dedup=False)`` and index ``basis.eigvecs``
    directly (both live in the ``0..n_bins-1`` active-bin space). Out-of-bounds samples
    (bin id ``-1``) get ``valid=False`` and a zero design row; callers drop/mask those
    rows consistently across spikes, ``Z`` and times.

    Parameters
    ----------
    env : neurospatial.Environment
    basis : GraphBasis
    times : NDArray, shape (n_time,)
    trajectory : NDArray, shape (n_time,) or (n_time, n_dims)
    interpolation : {"nearest"}, optional
        Only nearest-bin lookup is implemented; ``"linear"`` is accepted by the
        signature but not yet supported.

    Returns
    -------
    Z : NDArray, shape (n_time, rank)
    valid : NDArray[bool], shape (n_time,)
        ``True`` where the sample fell inside the environment.
    """
    if interpolation not in ("nearest", "linear"):
        raise ValueError(
            f"interpolation must be 'nearest' or 'linear', got {interpolation!r}."
        )
    if interpolation == "linear":
        raise NotImplementedError(
            "linear interpolation is not implemented; only nearest-bin lookup is "
            "supported."
        )

    _check_basis_matches_env(env, basis)
    bin_ids = _bin_ids(env, times, trajectory)
    valid = bin_ids >= 0
    rank = basis.eigvecs.shape[1]
    Z = np.zeros((bin_ids.shape[0], rank), dtype=float)
    Z[valid] = basis.eigvecs[bin_ids[valid]]
    return Z, valid


def bin_occupancy(
    env: "Environment",
    times: NDArray[np.float64],
    trajectory: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Poisson exposure (seconds spent in each active bin) from ``dt`` x visit counts.

    Computed from the **same** ``bin_sequence(dedup=False)`` binning as
    :func:`bin_spike_counts`, so occupancy and counts are aligned by construction
    (``occupancy_i == 0`` implies ``count_i == 0``). Not weighted by ``bin_sizes`` —
    that is for density/integration, not exposure. (``neurospatial.Environment.occupancy``
    uses an interval/gap-aware allocation that can leave a spiked bin with zero exposure;
    this per-sample version avoids that ill-posed offset.)

    Parameters
    ----------
    env : neurospatial.Environment
    times : NDArray, shape (n_time,)
    trajectory : NDArray, shape (n_time,) or (n_time, n_dims)
    dt : float
        Sampling interval in seconds; must be positive.

    Returns
    -------
    NDArray, shape (n_bins,)
        Seconds of exposure in each active bin.
    """
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    bin_ids = _bin_ids(env, times, trajectory)
    valid = bin_ids >= 0
    counts = np.bincount(bin_ids[valid], minlength=env.n_bins).astype(float)
    return counts * float(dt)


def bin_spike_counts(
    env: "Environment",
    spikes: NDArray[np.float64],
    times: NDArray[np.float64],
    trajectory: NDArray[np.float64],
    basis: GraphBasis,
) -> NDArray[np.float64]:
    """Aggregate per-time spike counts into per-active-bin counts.

    ``spikes`` is per-time counts of shape ``(n_time,)`` or ``(n_time, n_neurons)`` (bin
    raw spike-time arrays onto the ``times`` grid first). Uses the same
    ``bin_sequence(dedup=False)`` ids as :func:`graph_design_matrix` /
    :func:`bin_occupancy`; out-of-bounds samples are excluded.

    Parameters
    ----------
    env : neurospatial.Environment
    spikes : NDArray, shape (n_time,) or (n_time, n_neurons)
        Per-time spike counts.
    times : NDArray, shape (n_time,)
    trajectory : NDArray, shape (n_time,) or (n_time, n_dims)
    basis : GraphBasis
        Built from ``env``; used to assert the basis and ``env`` share one active-bin
        space, so the returned ``(n_bins, ...)`` counts align with the basis rows
        indexed in :func:`graph_design_matrix`.

    Returns
    -------
    NDArray, shape (n_bins, n_neurons)
        Per-active-bin summed spike counts.
    """
    _check_basis_matches_env(env, basis)
    spikes = np.asarray(spikes, dtype=float)
    if spikes.ndim == 1:
        spikes = spikes[:, None]
    bin_ids = _bin_ids(env, times, trajectory)
    if spikes.shape[0] != bin_ids.shape[0]:
        raise ValueError(
            f"spikes has {spikes.shape[0]} time rows but trajectory/times has "
            f"{bin_ids.shape[0]}."
        )
    valid = bin_ids >= 0
    n_neurons = spikes.shape[1]
    out = np.zeros((env.n_bins, n_neurons), dtype=float)
    np.add.at(out, bin_ids[valid], spikes[valid])
    return out


def laplacian_matches_distance_weight(env: "Environment") -> bool:
    """True if ``get_differential_operator`` gives the distance-weighted Laplacian.

    A cheap invariant check (used by the contract test): ``D @ D.T`` must equal
    ``nx.laplacian_matrix(env.connectivity, weight="distance")``.
    """
    L = build_graph_laplacian(env, convention="distance").toarray()
    L_nx = nx.laplacian_matrix(
        env.connectivity, nodelist=range(env.n_bins), weight="distance"
    ).toarray()
    return bool(np.allclose(L, L_nx, atol=1e-9))


def spectral_precision(
    eigvals: NDArray[np.float64],
    tau2: float,
    kappa2: float,
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Diagonal prior precision ``P0^{-1} = diag((kappa2 + lambda)^alpha / tau2)``.

    The reciprocal of the prior variance ``tau2 * S`` with ``S`` the spectral shape.
    Finite at the null modes because ``kappa2 > 0``.
    """
    if not tau2 > 0:
        raise ValueError(f"tau2 (prior amplitude) must be positive, got {tau2}.")
    shape = spectral_shape(eigvals, kappa2, alpha)  # validates kappa2, alpha > 0
    return 1.0 / (tau2 * shape)


def parity_penalty(
    eigvals: NDArray[np.float64], n_components: int
) -> NDArray[np.float64]:
    """Pure ``diag(lambda)`` penalty with the null modes left unpenalized.

    The first ``n_components`` entries (the per-component null modes, ordered first by
    :func:`build_graph_basis`) are set to zero so they act as unpenalized per-component
    intercepts. This is the MRF-parity configuration: the penalty is the eigenvalues
    themselves and the REML counterpart counts only positive eigenvalues.
    """
    penalty = np.array(eigvals, dtype=float)
    if n_components < 0 or n_components > penalty.shape[0]:
        raise ValueError(
            f"n_components={n_components} out of range for {penalty.shape[0]} modes."
        )
    penalty[:n_components] = 0.0
    return penalty


def fit_static_graph_glm(
    counts: ArrayLike,
    occupancy: ArrayLike,
    eigvecs: ArrayLike,
    penalty_diag: ArrayLike,
    *,
    max_iter: int = 25,
) -> tuple[Array, Array]:
    """Penalized per-bin Poisson GLM in the eigenbasis (Newton / Fisher scoring).

    Fits ``count_i ~ Poisson(exp(Phi_i w) * occ_i)`` with a diagonal quadratic penalty
    ``0.5 * w^T diag(penalty_diag) w``. Grouping the per-time Poisson likelihood by bin
    with a ``log(occupancy)`` offset makes this the exact ``Q -> 0`` static limit of the
    drifting model, and the MRF-parity target.

    Parameters
    ----------
    counts : ArrayLike, shape (n_bins,) or (n_bins, n_neurons)
        Per-active-bin spike counts (from :func:`bin_spike_counts`).
    occupancy : ArrayLike, shape (n_bins,)
        Per-active-bin exposure in seconds (from :func:`bin_occupancy`).
    eigvecs : ArrayLike, shape (n_bins, rank)
        The basis ``Phi`` (``GraphBasis.eigvecs``).
    penalty_diag : ArrayLike, shape (rank,)
        Diagonal penalty ``P0^{-1}`` (from :func:`spectral_precision` or
        :func:`parity_penalty`).
    max_iter : int, optional
        Newton iterations (the objective is convex; ~15 suffice), by default 25.

    Returns
    -------
    weights : Array, shape (rank,) or (n_neurons, rank)
        MAP coefficients.
    cov : Array, shape (rank, rank) or (n_neurons, rank, rank)
        Laplace posterior covariance (inverse Fisher + penalty) at the MAP.
    """
    Phi = jnp.asarray(eigvecs)
    occ = jnp.asarray(occupancy)
    counts_arr = jnp.asarray(counts)
    single = counts_arr.ndim == 1
    counts_2d = counts_arr[:, None] if single else counts_arr
    rank = Phi.shape[1]
    Lam = jnp.diag(jnp.asarray(penalty_diag))
    eye = jnp.eye(rank)
    visited = occ > 0
    # log-offset; unvisited bins contribute nothing (mu forced to 0 there).
    log_occ = jnp.where(visited, jnp.log(jnp.where(visited, occ, 1.0)), 0.0)

    def _fit_one(y: Array) -> tuple[Array, Array]:
        y = y.astype(Phi.dtype)

        def _step(w: Array, _: None) -> tuple[Array, None]:
            mu = jnp.where(visited, jnp.exp(Phi @ w + log_occ), 0.0)
            grad = Phi.T @ (mu - y) + Lam @ w
            hess = Phi.T @ (mu[:, None] * Phi) + Lam
            return w - psd_solve(hess, grad), None

        w, _ = jax.lax.scan(_step, jnp.zeros(rank), None, length=max_iter)
        mu = jnp.where(visited, jnp.exp(Phi @ w + log_occ), 0.0)
        cov = psd_solve(Phi.T @ (mu[:, None] * Phi) + Lam, eye)
        return w, symmetrize(cov)

    weights, cov = jax.vmap(_fit_one, in_axes=1)(counts_2d)
    if single:
        return weights[0], cov[0]
    return weights, cov


def static_log_evidence(
    counts: ArrayLike,
    occupancy: ArrayLike,
    eigvecs: ArrayLike,
    eigvals: ArrayLike,
    *,
    tau2: float,
    kappa2: float,
    alpha: float = 1.0,
) -> float:
    """Laplace log-evidence of the static graph GLM at amplitude ``tau2``.

    ``log Z(tau2) ~ [y.eta - mu] - 0.5 w^T Lam w + 0.5 logdet(Lam) - 0.5 logdet(H)``
    evaluated at the MAP ``w``, with ``Lam = P0^{-1}`` the spectral precision and
    ``H = Phi^T diag(mu) Phi + Lam``. Constant ``log(count!)`` terms are dropped (they
    do not depend on ``tau2``). Summed over neurons for multi-neuron ``counts``.
    """
    Phi = jnp.asarray(eigvecs)
    occ = jnp.asarray(occupancy)
    counts_arr = jnp.asarray(counts)
    single = counts_arr.ndim == 1
    counts_2d = counts_arr[:, None] if single else counts_arr
    prec = jnp.asarray(spectral_precision(np.asarray(eigvals), tau2, kappa2, alpha))
    Lam = jnp.diag(prec)
    weights, _ = fit_static_graph_glm(counts_arr, occ, Phi, prec)
    weights_2d = weights[None, :] if single else weights
    visited = occ > 0
    log_occ = jnp.where(visited, jnp.log(jnp.where(visited, occ, 1.0)), 0.0)
    # logdet(Lam) is constant across neurons; sign of prec is positive.
    logdet_lam = jnp.sum(jnp.log(prec))

    def _one(w: Array, y: Array) -> Array:
        eta = Phi @ w + log_occ
        mu = jnp.where(visited, jnp.exp(eta), 0.0)
        data_term = jnp.sum(jnp.where(visited, y * eta - mu, 0.0))
        hess = Phi.T @ (mu[:, None] * Phi) + Lam
        _, logdet_h = jnp.linalg.slogdet(hess)
        return data_term - 0.5 * (w @ (prec * w)) + 0.5 * logdet_lam - 0.5 * logdet_h

    ev = jax.vmap(_one, in_axes=(0, 1))(weights_2d, counts_2d.astype(Phi.dtype))
    return float(jnp.sum(ev))


def select_tau2_by_evidence(
    counts: ArrayLike,
    occupancy: ArrayLike,
    basis: GraphBasis,
    *,
    kappa2: float,
    alpha: float = 1.0,
    bounds: tuple[float, float] = (1e-4, 1e4),
) -> float:
    """Return the ``tau2`` maximizing :func:`static_log_evidence`.

    Optimizes over ``log tau2`` with a bounded scalar optimizer (the evidence is smooth
    and unimodal in ``log tau2`` for a fixed ``kappa2``).
    """
    lo, hi = np.log(bounds[0]), np.log(bounds[1])

    def _neg_ev(log_tau2: float) -> float:
        return -static_log_evidence(
            counts,
            occupancy,
            basis.eigvecs,
            basis.eigvals,
            tau2=float(np.exp(log_tau2)),
            kappa2=kappa2,
            alpha=alpha,
        )

    result = scipy.optimize.minimize_scalar(_neg_ev, bounds=(lo, hi), method="bounded")
    return float(np.exp(result.x))


class GraphPlaceFieldModel(SGDFittableMixin):
    """Drifting place-field model over a graph-Laplacian eigenbasis.

    Latent state per neuron ``c`` is ``w_{c,t} in R^rank``, the coefficients on the
    smoothest ``rank`` eigenvectors ``Phi`` of the environment graph Laplacian. The
    log-rate map is ``eta_{c,t} = Phi w_{c,t}``; at the animal's position ``z_t = Phi[bin(t)]``
    the point-process log-intensity is ``z_t^T w_{c,t}`` and spikes are Poisson with rate
    ``exp(z_t^T w_{c,t}) * dt``. The coefficients drift as a random walk
    ``w_{c,t} = w_{c,t-1} + eps_t``, ``eps_t ~ N(0, q_c * S)`` with the spectral shape
    ``S = (kappa2 I + diag(lambda))^(-alpha)`` shared by the prior ``P0 = tau2 * S``.

    x64 is required (see the module and repo CLAUDE.md notes).
    """

    def __init__(
        self,
        env: "Environment",
        dt: float,
        *,
        rank: Optional[int] = None,
        sigma: Optional[float] = None,
        kappa2: float = 1.0,
        alpha: float = 1.0,
        tau2: float = 1.0,
        init_drift_scale: float = 1e-3,
        interpolation: str = "nearest",
        laplacian_convention: LaplacianConvention = "distance",
        update_drift_scale: bool = True,
        update_amplitude: bool = True,
        update_init_mean: bool = True,
        max_firing_rate_hz: float = 500.0,
        max_newton_iter: int = 1,
    ) -> None:
        if not dt > 0:
            raise ValueError(f"dt must be positive, got {dt}.")
        if not kappa2 > 0:
            raise ValueError(f"kappa2 must be positive, got {kappa2}.")
        if not tau2 > 0:
            raise ValueError(f"tau2 must be positive, got {tau2}.")
        if not alpha > 0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        if not init_drift_scale >= 0:
            raise ValueError(f"init_drift_scale must be >= 0, got {init_drift_scale}.")
        if max_firing_rate_hz <= 0:
            raise ValueError(
                f"max_firing_rate_hz must be positive, got {max_firing_rate_hz}."
            )

        self.env = env
        self.dt = dt
        self.basis = build_graph_basis(
            env, rank=rank, sigma=sigma, laplacian_convention=laplacian_convention
        )
        self.rank = int(self.basis.eigvecs.shape[1])
        self.kappa2 = kappa2
        self.alpha = alpha
        self.tau2 = tau2
        self.init_drift_scale = init_drift_scale
        self.interpolation = interpolation
        self.max_firing_rate_hz = max_firing_rate_hz
        self.max_newton_iter = max_newton_iter
        self.update_drift_scale = update_drift_scale
        self.update_amplitude = update_amplitude
        self.update_init_mean = update_init_mean
        self._log_intensity_func = log_conditional_intensity

        # Spectral shape S (diagonal, in the eigenbasis). Fixed unless kappa2 changes.
        self.spectral_S = jnp.asarray(spectral_shape(self.basis.eigvals, kappa2, alpha))
        self.transition_matrix = jnp.eye(self.rank)

        # Populated during fit.
        self.n_neurons: int = 1
        self.drift_scale: Optional[Array] = None  # (n_neurons,)
        self.init_mean: Optional[Array] = None  # (n_neurons, rank)
        self.smoother_mean: Optional[Array] = None  # (n_neurons, n_time, rank)
        self.smoother_cov: Optional[Array] = None
        self.smoother_cross_cov: Optional[Array] = None
        self.filtered_mean: Optional[Array] = None
        self.filtered_cov: Optional[Array] = None
        self.log_likelihoods: list[float] = []
        self._n_time: int = 0

    @property
    def _max_log_count(self) -> float:
        return float(np.log(self.max_firing_rate_hz * self.dt))

    def _spectral_shape_current(self) -> Array:
        """S at the current kappa2 (recomputed so SGD updates to kappa2 take effect)."""
        return jnp.asarray(spectral_shape(self.basis.eigvals, self.kappa2, self.alpha))

    def prior_cov(self) -> Array:
        """Prior / initial covariance ``P0 = tau2 * diag(S)``, shape (rank, rank)."""
        return jnp.diag(self.tau2 * self._spectral_shape_current())

    def drift_cov(self, q_c: float) -> Array:
        """Per-neuron drift covariance ``Q_c = q_c * diag(S)``, shape (rank, rank)."""
        return jnp.diag(q_c * self._spectral_shape_current())
