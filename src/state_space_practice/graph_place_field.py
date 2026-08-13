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

from state_space_practice.kalman import (
    psd_solve,
    rts_backward_scan,
    sum_of_outer_products,
    symmetrize,
)
from state_space_practice.point_process_kalman import (
    _validate_filter_numerics,
    glm_laplace_update,
    log_conditional_intensity,
    poisson_family,
)
from state_space_practice.sgd_fitting import SGDFittableMixin
from state_space_practice.utils import check_converged, validate_count_array

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

    Parameters
    ----------
    eigvals : NDArray, shape (rank,)
        Laplacian eigenvalues (non-negative).
    tau2 : float
        Prior amplitude; must be positive.
    kappa2 : float
        Inverse-lengthscale squared; must be positive.
    alpha : float, optional
        Smoothness exponent, by default 1.0; must be positive.

    Returns
    -------
    NDArray, shape (rank,)
        Diagonal prior precision.
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

    Parameters
    ----------
    eigvals : NDArray, shape (rank,)
        Laplacian eigenvalues (non-negative, ascending).
    n_components : int
        Number of leading null modes to zero out.

    Returns
    -------
    NDArray, shape (rank,)
        ``eigvals`` with the first ``n_components`` entries set to zero.
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
    tol: float = 1e-8,
) -> tuple[Array, Array]:
    """Penalized per-bin Poisson GLM in the eigenbasis (damped Newton).

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
        Maximum Newton iterations, by default 25.
    tol : float, optional
        Relative gradient convergence tolerance, by default ``1e-8``.

    Returns
    -------
    weights : Array, shape (rank,) or (n_neurons, rank)
        MAP coefficients.
    cov : Array, shape (rank, rank) or (n_neurons, rank, rank)
        Laplace posterior covariance (inverse Fisher + penalty) at the MAP.
    """
    counts_np = np.asarray(counts)
    validate_count_array(counts_np, "counts", allow_empty=False)
    if counts_np.ndim not in (1, 2):
        raise ValueError(
            "counts must have shape (n_bins,) or (n_bins, n_neurons); "
            f"got shape {counts_np.shape}."
        )

    Phi_np = np.asarray(eigvecs, dtype=float)
    if Phi_np.ndim != 2 or min(Phi_np.shape) == 0:
        raise ValueError(
            "eigvecs must be a non-empty array with shape (n_bins, rank); "
            f"got shape {Phi_np.shape}."
        )
    if not np.all(np.isfinite(Phi_np)):
        raise ValueError("eigvecs must contain only finite values.")

    occ_np = np.asarray(occupancy, dtype=float)
    if occ_np.ndim != 1 or occ_np.shape[0] != Phi_np.shape[0]:
        raise ValueError(
            "occupancy must have shape (n_bins,) matching eigvecs; "
            f"got {occ_np.shape} and {Phi_np.shape}."
        )
    if not np.all(np.isfinite(occ_np)) or np.any(occ_np < 0):
        raise ValueError("occupancy must contain only finite, non-negative values.")
    if counts_np.shape[0] != Phi_np.shape[0]:
        raise ValueError(
            "counts and eigvecs must have the same n_bins; "
            f"got {counts_np.shape[0]} and {Phi_np.shape[0]}."
        )

    penalty_np = np.asarray(penalty_diag, dtype=float)
    rank = Phi_np.shape[1]
    if penalty_np.shape != (rank,):
        raise ValueError(
            f"penalty_diag must have shape ({rank},); got {penalty_np.shape}."
        )
    if not np.all(np.isfinite(penalty_np)) or np.any(penalty_np < 0):
        raise ValueError("penalty_diag must contain only finite, non-negative values.")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise ValueError(f"max_iter must be a positive integer; got {max_iter!r}.")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer; got {max_iter!r}.")
    tol_arr = np.asarray(tol)
    if tol_arr.shape != () or not np.isfinite(tol_arr) or float(tol_arr) <= 0:
        raise ValueError(f"tol must be a finite positive scalar; got {tol!r}.")

    counts_2d_np = counts_np[:, None] if counts_np.ndim == 1 else counts_np
    if np.any((occ_np == 0) & np.any(counts_2d_np > 0, axis=1)):
        raise ValueError("counts must be zero in bins with zero occupancy.")

    dtype = jnp.result_type(Phi_np, occ_np, penalty_np, jnp.float32)
    Phi = jnp.asarray(Phi_np, dtype=dtype)
    occ = jnp.asarray(occ_np, dtype=dtype)
    counts_arr = jnp.asarray(counts_np, dtype=dtype)
    single = counts_arr.ndim == 1
    counts_2d = counts_arr[:, None] if single else counts_arr
    penalty = jnp.asarray(penalty_np, dtype=dtype)
    Lam = jnp.diag(penalty)
    eye = jnp.eye(rank, dtype=dtype)
    visited = occ > 0
    # log-offset; unvisited bins contribute nothing (mu forced to 0 there).
    log_occ = jnp.where(visited, jnp.log(jnp.where(visited, occ, 1.0)), 0.0)
    tolerance = jnp.asarray(float(tol_arr), dtype=dtype)

    def _loss(w: Array, y: Array) -> Array:
        eta = Phi @ w + log_occ
        mu = jnp.where(visited, jnp.exp(eta), 0.0)
        poisson_nll = jnp.sum(jnp.where(visited, mu - y * eta, 0.0))
        return poisson_nll + 0.5 * (w @ (penalty * w))

    def _derivatives(w: Array, y: Array) -> tuple[Array, Array]:
        mu = jnp.where(visited, jnp.exp(Phi @ w + log_occ), 0.0)
        grad = Phi.T @ (mu - y) + penalty * w
        hess = Phi.T @ (mu[:, None] * Phi) + Lam
        return grad, hess

    def _fit_one(y: Array) -> tuple[Array, Array, Array]:
        gradient_scale = 1.0 + jnp.sum(y)

        def _step(
            carry: tuple[Array, Array], _: None
        ) -> tuple[tuple[Array, Array], None]:
            w, converged = carry
            current_loss = _loss(w, y)
            grad, hess = _derivatives(w, y)
            delta = psd_solve(hess, grad)
            directional_derivative = grad @ delta
            is_descent = (
                (~converged)
                & jnp.isfinite(directional_derivative)
                & (directional_derivative > 0.0)
            )

            def _backtrack(
                line_carry: tuple[Array, Array, Array], _: None
            ) -> tuple[tuple[Array, Array, Array], None]:
                step_size, accepted, accepted_w = line_carry
                trial_w = w - step_size * delta
                trial_loss = _loss(trial_w, y)
                sufficient_decrease = (
                    (~accepted)
                    & is_descent
                    & jnp.isfinite(trial_loss)
                    & (
                        trial_loss
                        <= current_loss - 1e-4 * step_size * directional_derivative
                    )
                )
                accepted_w = jnp.where(sufficient_decrease, trial_w, accepted_w)
                accepted = accepted | sufficient_decrease
                step_size = jnp.where(accepted, step_size, 0.5 * step_size)
                return (step_size, accepted, accepted_w), None

            (_step_size, accepted, candidate_w), _backtrack_history = jax.lax.scan(
                _backtrack,
                (
                    jnp.asarray(1.0, dtype=dtype),
                    jnp.asarray(False),
                    w,
                ),
                None,
                length=60,
            )
            new_w = jnp.where(accepted & (~converged), candidate_w, w)
            new_grad, _new_hessian = _derivatives(new_w, y)
            new_converged = converged | (
                jnp.max(jnp.abs(new_grad)) <= tolerance * gradient_scale
            )
            return (new_w, new_converged), None

        initial_w = jnp.zeros(rank, dtype=dtype)
        initial_grad, _ = _derivatives(initial_w, y)
        initially_converged = (
            jnp.max(jnp.abs(initial_grad)) <= tolerance * gradient_scale
        )
        (w, converged), _ = jax.lax.scan(
            _step,
            (initial_w, initially_converged),
            None,
            length=max_iter,
        )
        mu = jnp.where(visited, jnp.exp(Phi @ w + log_occ), 0.0)
        cov = psd_solve(Phi.T @ (mu[:, None] * Phi) + Lam, eye)
        return w, symmetrize(cov), converged

    weights, cov, converged = jax.vmap(_fit_one, in_axes=1)(counts_2d)
    n_unconverged = int(jnp.sum(~converged))
    if n_unconverged:
        # A non-converged solve returns a MAP far from the optimum and a Laplace
        # covariance evaluated off the stationary point; surface it rather than let
        # it silently poison tau2 selection or the EM warm-start.
        logger.warning(
            "fit_static_graph_glm: %d/%d neuron(s) did not reach the gradient "
            "tolerance %.1e within max_iter=%d; raise max_iter or check the "
            "conditioning of penalty_diag / counts.",
            n_unconverged,
            counts_2d.shape[1],
            float(tol_arr),
            max_iter,
        )
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

    Parameters
    ----------
    counts : ArrayLike, shape (n_bins,) or (n_bins, n_neurons)
        Per-active-bin spike counts (from :func:`bin_spike_counts`).
    occupancy : ArrayLike, shape (n_bins,)
        Per-active-bin exposure in seconds (from :func:`bin_occupancy`).
    eigvecs : ArrayLike, shape (n_bins, rank)
        The basis ``Phi`` (``GraphBasis.eigvecs``).
    eigvals : ArrayLike, shape (rank,)
        Laplacian eigenvalues (``GraphBasis.eigvals``).
    tau2 : float
        Prior amplitude; must be positive.
    kappa2 : float
        Inverse-lengthscale squared; must be positive.
    alpha : float, optional
        Smoothness exponent, by default 1.0.

    Returns
    -------
    float
        Laplace log-evidence, summed over neurons for multi-neuron ``counts``.
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
        return cast(
            Array,
            data_term - 0.5 * (w @ (prec * w)) + 0.5 * logdet_lam - 0.5 * logdet_h,
        )

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

    Parameters
    ----------
    counts : ArrayLike, shape (n_bins,) or (n_bins, n_neurons)
        Per-active-bin spike counts (from :func:`bin_spike_counts`).
    occupancy : ArrayLike, shape (n_bins,)
        Per-active-bin exposure in seconds (from :func:`bin_occupancy`).
    basis : GraphBasis
        Supplies ``eigvecs`` and ``eigvals`` for the evidence evaluation.
    kappa2 : float
        Inverse-lengthscale squared; must be positive.
    alpha : float, optional
        Smoothness exponent, by default 1.0.
    bounds : tuple[float, float], optional
        Search interval for ``tau2``, by default ``(1e-4, 1e4)``; both endpoints
        must be finite and positive, with the lower strictly below the upper.

    Returns
    -------
    float
        The evidence-maximizing ``tau2``.
    """
    bounds_arr = np.asarray(bounds, dtype=float)
    if (
        bounds_arr.shape != (2,)
        or not np.all(np.isfinite(bounds_arr))
        or np.any(bounds_arr <= 0)
        or bounds_arr[0] >= bounds_arr[1]
    ):
        raise ValueError(
            "bounds must be two finite positive values with lower < upper; "
            f"got {bounds!r}."
        )
    lo, hi = np.log(bounds_arr)

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
    if not result.success or not np.isfinite(result.x) or not np.isfinite(result.fun):
        raise RuntimeError(f"tau2 evidence optimization failed: {result.message}")
    return float(np.exp(result.x))


def _masked_graph_point_process_filter(
    init_mean: Array,
    init_cov: Array,
    design_matrix: Array,
    spikes: Array,
    valid: Array,
    transition_matrix: Array,
    process_cov: Array,
    *,
    dt: float,
    max_log_count: float,
    max_newton_iter: int,
) -> tuple[Array, Array, Array]:
    """Filter one neuron on the full time grid with masked observations.

    Unlike :func:`stochastic_point_process_filter`, the first observation is
    conditioned directly on ``(init_mean, init_cov)``.  Subsequent rows receive one
    dynamics transition each.  A row whose position is invalid skips only the
    observation update; its dynamics propagation remains in the state trajectory.

    Parameters
    ----------
    init_mean : Array, shape (rank,)
        Prior mean conditioning the first observation.
    init_cov : Array, shape (rank, rank)
        Prior covariance conditioning the first observation.
    design_matrix : Array, shape (n_time, rank)
        Per-time design rows ``Z`` (basis evaluated at the animal's position).
    spikes : Array, shape (n_time,)
        Per-time spike counts for this neuron.
    valid : Array, shape (n_time,)
        Boolean mask; ``False`` rows skip the observation update but still
        receive a dynamics transition.
    transition_matrix : Array, shape (rank, rank)
        State transition matrix ``A``.
    process_cov : Array, shape (rank, rank)
        Process noise covariance ``Q``.
    dt : float
        Sampling interval in seconds, passed to the Poisson observation family.
    max_log_count : float
        Clamp on the log conditional intensity, passed to the Poisson
        observation family.
    max_newton_iter : int
        Maximum Newton iterations for each per-time Laplace update.

    Returns
    -------
    filtered_mean : Array, shape (n_time, rank)
        Filtered state means.
    filtered_cov : Array, shape (n_time, rank, rank)
        Filtered state covariances.
    marginal_ll : Array, scalar
        Total marginal log-likelihood summed over time bins.
    """
    family = poisson_family(dt, max_log_count=max_log_count)

    def _observation_update(
        prior_mean: Array,
        prior_cov: Array,
        design_row: Array,
        spike_count: Array,
        is_valid: Array,
    ) -> tuple[Array, Array, Array]:
        def _observed(_: None) -> tuple[Array, Array, Array]:
            def _eta(state: Array) -> Array:
                return jnp.atleast_1d(log_conditional_intensity(design_row, state))

            def _grad_eta(_state: Array) -> Array:
                return design_row[None, :]

            return glm_laplace_update(
                prior_mean,
                prior_cov,
                jnp.atleast_1d(spike_count),
                _eta,
                family,
                grad_eta_func=_grad_eta,
                max_newton_iter=max_newton_iter,
            )

        def _missing(_: None) -> tuple[Array, Array, Array]:
            return prior_mean, prior_cov, jnp.zeros((), dtype=prior_mean.dtype)

        return cast(
            tuple[Array, Array, Array],
            jax.lax.cond(is_valid, _observed, _missing, operand=None),
        )

    first_mean, first_cov, first_ll = _observation_update(
        init_mean,
        init_cov,
        design_matrix[0],
        spikes[0],
        valid[0],
    )

    def _step(
        carry: tuple[Array, Array, Array],
        args: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array]]:
        previous_mean, previous_cov, marginal_ll = carry
        design_row, spike_count, is_valid = args
        prior_mean = transition_matrix @ previous_mean
        prior_cov = symmetrize(
            transition_matrix @ previous_cov @ transition_matrix.T + process_cov
        )
        posterior_mean, posterior_cov, log_likelihood = _observation_update(
            prior_mean,
            prior_cov,
            design_row,
            spike_count,
            is_valid,
        )
        marginal_ll = marginal_ll + log_likelihood
        return (posterior_mean, posterior_cov, marginal_ll), (
            posterior_mean,
            posterior_cov,
        )

    (_, _, marginal_ll), (remaining_mean, remaining_cov) = jax.lax.scan(
        _step,
        (first_mean, first_cov, first_ll),
        (design_matrix[1:], spikes[1:], valid[1:]),
    )
    filtered_mean = jnp.concatenate((first_mean[None, :], remaining_mean), axis=0)
    filtered_cov = jnp.concatenate((first_cov[None, :, :], remaining_cov), axis=0)
    return filtered_mean, filtered_cov, marginal_ll


class GraphPlaceFieldModel(SGDFittableMixin):
    """Drifting place-field model over a graph-Laplacian eigenbasis.

    Latent state per neuron ``c`` is ``w_{c,t} in R^rank``, the coefficients on the
    smoothest ``rank`` eigenvectors ``Phi`` of the environment graph Laplacian. The
    log-rate map is ``eta_{c,t} = Phi w_{c,t}``; at the animal's position ``z_t = Phi[bin(t)]``
    the point-process log-intensity is ``z_t^T w_{c,t}`` and spikes are Poisson with rate
    ``exp(z_t^T w_{c,t}) * dt``. The coefficients drift as a random walk
    ``w_{c,t} = w_{c,t-1} + eps_t``, ``eps_t ~ N(0, q_c * S)`` with the spectral shape
    ``S = (kappa2 I + diag(lambda))^(-alpha)`` shared by the prior ``P0 = tau2 * S``.

    Drift-scale learning is disabled by default because the approximate Laplace-EKF
    marginal likelihood does not reliably identify ``q_c`` from spike observations.
    Setting ``update_drift_scale=True`` enables the closed-form update as an explicit
    experimental opt-in.

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
        update_drift_scale: bool = False,
        update_amplitude: bool = True,
        update_init_mean: bool = True,
        update_kappa2: bool = True,
        max_firing_rate_hz: float = 500.0,
        max_newton_iter: int = 1,
    ) -> None:
        """Build the graph basis and initialize model hyperparameters.

        Parameters
        ----------
        env : neurospatial.Environment
            A fitted environment; supplies the graph substrate for
            :func:`build_graph_basis`.
        dt : float
            Sampling interval in seconds; must be positive.
        rank : int or None, optional
            Number of smoothest graph-Laplacian modes to keep, forwarded to
            :func:`build_graph_basis`. ``None`` (default) keeps all modes unless
            ``sigma`` is given.
        sigma : float or None, optional
            Smoothing bandwidth, forwarded to :func:`build_graph_basis`.
        kappa2 : float, optional
            Inverse-lengthscale squared in the spectral shape
            ``S = (kappa2 + lambda) ** (-alpha)``, by default 1.0; must be
            positive.
        alpha : float, optional
            Smoothness exponent, by default 1.0; must be positive.
        tau2 : float, optional
            Initial prior/initial-state amplitude ``P0 = tau2 * S``, by default
            1.0; must be positive.
        init_drift_scale : float, optional
            Initial per-neuron drift scale ``q_c`` used to seed ``drift_scale``
            before fitting, by default 1e-3; must be non-negative.
        interpolation : {"nearest"}, optional
            Design-matrix lookup mode forwarded to :func:`graph_design_matrix`,
            by default ``"nearest"``. ``"linear"`` is accepted by the signature
            but not implemented.
        laplacian_convention : {"distance", "inverse_distance"}, optional
            Edge-weight convention forwarded to :func:`build_graph_basis`, by
            default ``"distance"``.
        update_drift_scale : bool, optional
            Whether ``fit``'s M-step updates the per-neuron drift scale ``q_c``,
            by default **False**. The approximate Laplace-EKF marginal
            likelihood does not reliably identify ``q_c`` from spike
            observations, so learning it is an explicit opt-in.
        update_amplitude : bool, optional
            Whether ``fit``'s M-step updates ``tau2``, by default True.
        update_init_mean : bool, optional
            Whether ``fit``'s M-step updates the per-neuron initial state mean,
            by default True.
        update_kappa2 : bool, optional
            Whether ``fit_sgd`` optimizes ``kappa2``, by default True.
            ``kappa2`` reshapes the spectral shape nonlinearly, so it has no
            closed-form EM M-step and is learnable by ``fit_sgd`` only.
        max_firing_rate_hz : float, optional
            Maximum firing rate (Hz) used to clamp the log conditional
            intensity for numerical stability, by default 500.0; must be
            positive.
        max_newton_iter : int, optional
            Maximum Newton iterations per time-step Laplace update in the
            filter, by default 1.
        """
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
        if not max_firing_rate_hz > 0:  # also rejects NaN, matching the checks above
            raise ValueError(
                f"max_firing_rate_hz must be positive, got {max_firing_rate_hz}."
            )
        if isinstance(max_newton_iter, bool) or not isinstance(
            max_newton_iter, (int, np.integer)
        ):
            raise ValueError(
                f"max_newton_iter must be a positive integer, got {max_newton_iter!r}."
            )
        if max_newton_iter < 1:
            # 0 would make the Laplace update a no-op (observation ignored) silently.
            raise ValueError(f"max_newton_iter must be >= 1, got {max_newton_iter}.")

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
        # The approximate Laplace-EKF marginal likelihood does not reliably
        # identify q_c from spike observations. Keep it fixed by default; the
        # closed-form update remains available as an explicit experimental opt-in.
        self.update_drift_scale = update_drift_scale
        self.update_amplitude = update_amplitude
        self.update_init_mean = update_init_mean
        # kappa2 is learnable by fit_sgd only (it reshapes S nonlinearly, so it has no
        # closed-form EM M-step); default on to preserve prior behavior.
        self.update_kappa2 = update_kappa2
        self._log_intensity_func = log_conditional_intensity

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
        # Set True only after a fit/fit_sgd that finished with finite, accepted
        # posteriors; the sole gate for _check_fitted. A failed fit that leaves NaN
        # posteriors behind must not read as fitted.
        self._is_fitted: bool = False
        self.log_likelihoods: list[float] = []
        self._n_time: int = 0

    @property
    def _max_log_count(self) -> float:
        return float(np.log(self.max_firing_rate_hz * self.dt))

    def _spectral_shape_current(self) -> Array:
        """Spectral shape ``S`` at the current ``kappa2``.

        Recomputed on every call rather than cached: ``kappa2`` is fittable by
        ``fit_sgd``, so a construction-time snapshot would silently go stale. All
        consumers of ``S`` (prior, drift, filter, M-step) must call this.
        """
        return jnp.asarray(spectral_shape(self.basis.eigvals, self.kappa2, self.alpha))

    def prior_cov(self) -> Array:
        """Prior / initial covariance ``P0 = tau2 * diag(S)``, shape (rank, rank)."""
        return jnp.diag(self.tau2 * self._spectral_shape_current())

    def drift_cov(self, q_c: float) -> Array:
        """Per-neuron drift covariance ``Q_c = q_c * diag(S)``, shape (rank, rank)."""
        return jnp.diag(q_c * self._spectral_shape_current())

    def _design_and_spikes(
        self,
        times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
        spikes: ArrayLike,
    ) -> tuple[Array, Array, Array]:
        """Build the full-grid design, aligned spikes, and observation mask.

        Returns ``Z`` of shape ``(n_time, rank)``, ``spikes`` of shape
        ``(n_neurons, n_time)`` (neuron axis first, ready for ``vmap``), and a
        ``valid`` mask. Out-of-bounds rows retain their place in the time grid so the
        latent random walk still advances; only their observation update is skipped.
        """
        Z_full, valid = graph_design_matrix(
            self.env, self.basis, times, trajectory, interpolation=self.interpolation
        )
        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]
        if spikes_arr.ndim != 2:
            raise ValueError(
                "spikes must be 1D (n_time,) or 2D (n_time, n_neurons); "
                f"got shape {spikes_arr.shape}."
            )
        if spikes_arr.shape[0] != Z_full.shape[0]:
            raise ValueError(
                "spikes and trajectory/times must have the same number of rows; "
                f"got {spikes_arr.shape[0]} and {Z_full.shape[0]}."
            )
        return jnp.asarray(Z_full), spikes_arr.T, jnp.asarray(valid)

    def _e_step(self, Z: Array, spikes: Array, valid: Optional[Array] = None) -> float:
        """Per-neuron vmap Laplace-EKF smoother; returns total marginal LL."""
        assert self.init_mean is not None
        assert self.drift_scale is not None
        if valid is None:
            valid = jnp.ones((Z.shape[0],), dtype=bool)
        S = self._spectral_shape_current()
        P0 = jnp.diag(self.tau2 * S)
        A = self.transition_matrix

        def _one(m0: Array, spk: Array, q_c: Array):
            Q = jnp.diag(q_c * S)
            filtered_mean, filtered_cov, marginal_ll = (
                _masked_graph_point_process_filter(
                    m0,
                    P0,
                    Z,
                    spk,
                    valid,
                    A,
                    Q,
                    dt=self.dt,
                    max_log_count=self._max_log_count,
                    max_newton_iter=self.max_newton_iter,
                )
            )
            smoother_mean, smoother_cov, smoother_cross_cov = rts_backward_scan(
                filtered_mean,
                filtered_cov,
                A,
                Q,
            )
            return (
                smoother_mean,
                smoother_cov,
                smoother_cross_cov,
                marginal_ll,
                filtered_mean,
                filtered_cov,
            )

        (
            self.smoother_mean,
            self.smoother_cov,
            self.smoother_cross_cov,
            marginal_ll,
            self.filtered_mean,
            self.filtered_cov,
        ) = jax.vmap(_one, in_axes=(0, 0, 0))(self.init_mean, spikes, self.drift_scale)
        return float(jnp.sum(marginal_ll))

    def _warm_start(self, times, trajectory, spikes) -> None:
        """Set per-neuron init_mean from the static GLM MAP on aggregated bins."""
        counts = bin_spike_counts(self.env, spikes, times, trajectory, self.basis)
        occ = bin_occupancy(self.env, times, trajectory, self.dt)
        prec = spectral_precision(
            self.basis.eigvals, self.tau2, self.kappa2, self.alpha
        )
        w0, _ = fit_static_graph_glm(counts, occ, self.basis.eigvecs, prec)
        w0 = jnp.atleast_2d(w0)  # (n_neurons, rank)
        self.init_mean = w0

    def _m_step(self) -> None:
        """Closed-form scalar updates of per-neuron q_c and shared tau2."""
        assert self.smoother_mean is not None
        assert self.smoother_cov is not None
        assert self.smoother_cross_cov is not None
        assert self.init_mean is not None
        S = self._spectral_shape_current()

        if self.update_drift_scale:
            n_time = int(self.smoother_mean.shape[1])
            if n_time < 2:
                raise ValueError(
                    "at least two time rows are required to update drift_scale."
                )

            def _increment_stats(sm, sc, scc):
                # sm (T, rank), sc (T, rank, rank), scc (T-1, rank, rank)
                gamma = jnp.sum(sc, axis=0) + sum_of_outer_products(sm, sm)
                gamma1 = gamma - jnp.outer(sm[-1], sm[-1]) - sc[-1]
                gamma2 = gamma - jnp.outer(sm[0], sm[0]) - sc[0]
                beta = (scc.sum(axis=0) + sum_of_outer_products(sm[:-1], sm[1:])).T
                q_inc = (gamma2 - beta.T - beta + gamma1) / (n_time - 1)
                return jnp.diag(q_inc)

            diag_q_inc = jax.vmap(_increment_stats)(
                self.smoother_mean,
                self.smoother_cov,
                self.smoother_cross_cov,
            )
            # q_c* = mean_j( diag(E[dw dw^T])_j / S_j )
            self.drift_scale = jnp.maximum(
                jnp.mean(diag_q_inc / S[None, :], axis=1), 1e-12
            )
        if self.update_amplitude:
            diag_initial_second_moment = jnp.diagonal(
                self.smoother_cov[:, 0], axis1=-2, axis2=-1
            )
            if not self.update_init_mean:
                mean_residual = self.smoother_mean[:, 0, :] - self.init_mean
                diag_initial_second_moment = (
                    diag_initial_second_moment + mean_residual**2
                )
            # tau2* = mean over neurons/modes of E[(w0-m0)^2] / S.
            self.tau2 = float(
                jnp.maximum(
                    jnp.mean(diag_initial_second_moment / S[None, :]),
                    1e-12,
                )
            )
        if self.update_init_mean:
            self.init_mean = self.smoother_mean[:, 0, :]

    def fit(
        self,
        times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
        spikes: ArrayLike,
        *,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        warm_start: bool = True,
        verbose: bool = True,
    ) -> list[float]:
        """Fit by EM (GEM with rollback). Returns the accepted marginal-LL history.

        Parameters
        ----------
        times : NDArray, shape (n_time,)
            Sample times in seconds.
        trajectory : NDArray, shape (n_time,) or (n_time, n_dims)
            Animal position at each time sample.
        spikes : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Per-time spike counts.
        max_iter : int, optional
            Maximum EM iterations, by default 100.
        tolerance : float, optional
            Relative marginal-LL convergence tolerance, by default 1e-4.
        warm_start : bool, optional
            Initialize ``init_mean`` from the static GLM MAP on aggregated bins
            (and reset ``drift_scale`` to ``init_drift_scale``), by default True.
        verbose : bool, optional
            Print per-iteration marginal LL, by default True.

        Returns
        -------
        list[float]
            Accepted marginal log-likelihood at each EM iteration.
        """
        # Reset before any validation or attribute mutation: a re-fit that fails (bad
        # args, no in-bounds rows, non-finite E-step) must not leave the model reading
        # as fitted with stale posteriors for a previous neuron count.
        self._is_fitted = False
        self._clear_posteriors()
        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
            max_iter, (int, np.integer)
        ):
            raise ValueError(f"max_iter must be a positive integer; got {max_iter!r}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be a positive integer; got {max_iter!r}.")
        tolerance_arr = np.asarray(tolerance)
        if (
            tolerance_arr.shape != ()
            or not np.isfinite(tolerance_arr)
            or float(tolerance_arr) <= 0
        ):
            raise ValueError(
                f"tolerance must be a finite positive scalar; got {tolerance!r}."
            )

        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]
        validate_count_array(spikes_arr, "spikes", allow_empty=False)
        if spikes_arr.ndim != 2:
            raise ValueError(
                "spikes must be 1D (n_time,) or 2D (n_time, n_neurons); "
                f"got shape {spikes_arr.shape}."
            )
        self.n_neurons = int(spikes_arr.shape[1])
        Z, spk, valid = self._design_and_spikes(times, trajectory, spikes_arr)
        self._n_time = int(Z.shape[0])
        n_valid = int(jnp.sum(valid))
        if n_valid == 0:
            raise ValueError(
                "trajectory contains no in-bounds observations for this environment."
            )
        if self.update_drift_scale and self._n_time < 2:
            raise ValueError(
                "at least two time rows are required to update "
                "drift_scale; set update_drift_scale=False to keep it fixed."
            )

        if warm_start:
            self._warm_start(times, trajectory, spikes_arr)
        elif self.init_mean is None or self.init_mean.shape != (
            self.n_neurons,
            self.rank,
        ):
            self.init_mean = jnp.zeros((self.n_neurons, self.rank))
        # Reset drift_scale only on a warm start or a neuron-count change, matching
        # fit_sgd: a repeated fit(warm_start=False) preserves a learned drift_scale.
        drift_shape_matches = (
            self.drift_scale is not None and self.drift_scale.shape == (self.n_neurons,)
        )
        if warm_start or not drift_shape_matches:
            self.drift_scale = jnp.full(self.n_neurons, self.init_drift_scale)

        _validate_filter_numerics(self.prior_cov(), n_time=self._n_time)

        def _log(msg: str) -> None:
            if verbose:
                print(msg)

        self.log_likelihoods = []
        last_state: Optional[dict] = None
        converged = False

        def _capture() -> dict:
            return {
                k: getattr(self, k)
                for k in (
                    "smoother_mean",
                    "smoother_cov",
                    "smoother_cross_cov",
                    "filtered_mean",
                    "filtered_cov",
                    "drift_scale",
                    "init_mean",
                    "tau2",
                )
            }

        def _restore(state: Optional[dict]) -> None:
            if state is not None:
                for k, v in state.items():
                    setattr(self, k, v)

        for iteration in range(max_iter):
            ll = self._e_step(Z, spk, valid)
            self.log_likelihoods.append(ll)
            _log(f"  EM iter {iteration + 1}/{max_iter}: LL = {ll:.2f}")
            if not np.isfinite(ll):
                self.log_likelihoods.pop()
                if last_state is None:
                    # First E-step is non-finite: there is no accepted state to fall
                    # back to. Clear the NaN posteriors and fail loud (the model is
                    # left not-fitted via self._is_fitted, so predict_rate_map/score
                    # will refuse rather than return NaN).
                    self._clear_posteriors()
                    raise RuntimeError(
                        "GraphPlaceFieldModel.fit: the first E-step produced a "
                        "non-finite marginal log-likelihood; the fit cannot proceed. "
                        "Check init_mean/tau2/kappa2, trajectory coverage, and that "
                        "x64 is enabled."
                    )
                _restore(last_state)
                msg = "non-finite marginal LL; rolling back to the last accepted state."
                logger.warning("GraphPlaceFieldModel.fit: %s", msg)
                _log(f"  WARNING: {msg}")
                break
            if iteration > 0:
                converged, increasing = check_converged(
                    ll, self.log_likelihoods[-2], tolerance
                )
                if not increasing:
                    _restore(last_state)
                    bad = self.log_likelihoods.pop()
                    msg = (
                        f"LL decreased {self.log_likelihoods[-1]:.2f} -> {bad:.2f}; "
                        "rolling back and stopping."
                    )
                    logger.warning("GraphPlaceFieldModel.fit: %s", msg)
                    _log(f"  WARNING: {msg}")
                    break
                if converged:
                    _log(f"  Converged after {iteration + 1} iterations.")
                    break
            last_state = _capture()
            self._m_step()
        else:
            # The loop ends immediately after an M-step. Evaluate that candidate so
            # returned parameters, posteriors, and LL all describe the same state.
            final_ll = self._e_step(Z, spk, valid)
            final_is_finite = np.isfinite(final_ll)
            final_converged, final_increasing = check_converged(
                final_ll, self.log_likelihoods[-1], tolerance
            )
            if final_is_finite and final_increasing:
                self.log_likelihoods.append(final_ll)
                converged = bool(final_converged)
            else:
                _restore(last_state)
                msg = "final E-step rejected the last M-step; rolling back."
                logger.warning("GraphPlaceFieldModel.fit: %s", msg)
                _log(f"  WARNING: {msg}")

        self._finalize_convergence(bool(converged), max_iter)
        # Reached only via a normal loop exit (converge/rollback/max-iter), all of
        # which leave finite, accepted posteriors; the iteration-0 failure raises above.
        self._is_fitted = True
        return self.log_likelihoods

    # --- SGDFittableMixin protocol ---

    @property
    def _n_timesteps(self) -> int:
        return self._n_time

    def _check_sgd_initialized(self) -> None:
        if self.init_mean is None or self.drift_scale is None or self._n_time <= 0:
            raise RuntimeError(
                "Model not initialized. Call fit_sgd(times, trajectory, spikes), "
                "not SGDFittableMixin.fit_sgd() directly."
            )

    def _build_param_spec(self) -> tuple[dict, dict]:
        from state_space_practice.parameter_transforms import POSITIVE, UNCONSTRAINED

        assert self.init_mean is not None
        assert self.drift_scale is not None
        params: dict = {}
        spec: dict = {}
        if self.update_kappa2:
            params["kappa2"] = jnp.asarray(self.kappa2)
            spec["kappa2"] = POSITIVE
        if self.update_drift_scale:
            params["drift_scale"] = self.drift_scale
            spec["drift_scale"] = POSITIVE
        if self.update_amplitude:
            params["tau2"] = jnp.asarray(self.tau2)
            spec["tau2"] = POSITIVE
        if self.update_init_mean:
            params["init_mean"] = self.init_mean
            spec["init_mean"] = UNCONSTRAINED
        if not params:
            raise ValueError(
                "fit_sgd has nothing to optimize: all of update_kappa2, "
                "update_drift_scale, update_amplitude, and update_init_mean are "
                "False. Enable at least one, or use fit() for EM."
            )
        return params, spec

    def _sgd_loss_fn(
        self, params: dict, Z: Array, spikes: Array, valid: Array
    ) -> Array:
        assert self.init_mean is not None
        assert self.drift_scale is not None
        kappa2 = params.get("kappa2", self.kappa2)
        tau2 = params.get("tau2", self.tau2)
        drift_scale = params.get("drift_scale", self.drift_scale)
        init_mean = params.get("init_mean", self.init_mean)
        eigvals = jnp.asarray(self.basis.eigvals)
        S = (kappa2 + eigvals) ** (-self.alpha)
        P0 = jnp.diag(tau2 * S)

        def _one(m0: Array, spk: Array, q_c: Array) -> Array:
            _, _, marginal_ll = _masked_graph_point_process_filter(
                m0,
                P0,
                Z,
                spk,
                valid,
                self.transition_matrix,
                jnp.diag(q_c * S),
                dt=self.dt,
                max_log_count=self._max_log_count,
                max_newton_iter=self.max_newton_iter,
            )
            return marginal_ll

        marginal_ll = jax.vmap(_one, in_axes=(0, 0, 0))(init_mean, spikes, drift_scale)
        return -jnp.sum(marginal_ll)

    def _store_sgd_params(self, params: dict) -> None:
        if "drift_scale" in params:
            self.drift_scale = params["drift_scale"]
        if "tau2" in params:
            self.tau2 = float(params["tau2"])
        if "kappa2" in params:
            self.kappa2 = float(params["kappa2"])
        if "init_mean" in params:
            self.init_mean = params["init_mean"]

    def _finalize_sgd(self, Z: Array, spikes: Array, valid: Array) -> None:
        final_ll = self._e_step(Z, spikes, valid)
        self.log_likelihoods = [final_ll]
        if np.isfinite(final_ll):
            self._is_fitted = True
        else:
            self._clear_posteriors()
            raise RuntimeError(
                "GraphPlaceFieldModel.fit_sgd: the final E-step produced a non-finite "
                "marginal log-likelihood; the model was not fitted."
            )

    def fit_sgd(  # type: ignore[override]
        self,
        times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
        spikes: ArrayLike,
        *,
        optimizer: Optional[object] = None,
        num_steps: int = 200,
        verbose: bool = False,
        convergence_tol: Optional[float] = None,
        warm_start: bool = True,
    ) -> list[float]:
        """Fit graph hyperparameters by minimizing the negative marginal LL.

        Parameters
        ----------
        times : NDArray, shape (n_time,)
            Sample times in seconds.
        trajectory : NDArray, shape (n_time,) or (n_time, n_dims)
            Animal position at each time sample.
        spikes : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Per-time spike counts.
        optimizer : optax optimizer or None, optional
            Gradient transformation to use; ``None`` (default) uses
            ``adam(1e-2)`` with gradient clipping.
        num_steps : int, optional
            Number of optimization steps, by default 200.
        verbose : bool, optional
            Log progress every 10 steps, by default False.
        convergence_tol : float or None, optional
            If set, stop early once the relative change in marginal LL falls
            below this tolerance for 5 consecutive steps; ``None`` (default)
            disables early stopping.
        warm_start : bool, optional
            Initialize ``init_mean`` from the static GLM MAP on aggregated bins
            (and reset ``drift_scale``), by default True.

        Returns
        -------
        list[float]
            Marginal log-likelihood at each evaluated optimization step that
            produced a finite loss.
        """
        # Reset before any validation or attribute mutation (see fit()): a failed
        # re-fit must not leave the model reading as fitted with stale posteriors.
        self._is_fitted = False
        self._clear_posteriors()
        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]
        validate_count_array(spikes_arr, "spikes", allow_empty=False)
        if spikes_arr.ndim != 2:
            raise ValueError(
                "spikes must be 1D (n_time,) or 2D (n_time, n_neurons); "
                f"got shape {spikes_arr.shape}."
            )
        self.n_neurons = int(spikes_arr.shape[1])
        Z, spk, valid = self._design_and_spikes(times, trajectory, spikes_arr)
        self._n_time = int(Z.shape[0])
        n_valid = int(jnp.sum(valid))
        if n_valid == 0:
            raise ValueError(
                "trajectory contains no in-bounds observations for this environment."
            )
        if self.update_drift_scale and self._n_time < 2:
            raise ValueError(
                "at least two time rows are required to update "
                "drift_scale; set update_drift_scale=False to keep it fixed."
            )

        state_shape_matches = self.init_mean is not None and self.init_mean.shape == (
            self.n_neurons,
            self.rank,
        )
        if warm_start:
            self._warm_start(times, trajectory, spikes_arr)
        elif not state_shape_matches:
            self.init_mean = jnp.zeros((self.n_neurons, self.rank))

        drift_shape_matches = (
            self.drift_scale is not None and self.drift_scale.shape == (self.n_neurons,)
        )
        if warm_start or not drift_shape_matches:
            initial_drift = self.init_drift_scale
            if self.update_drift_scale:
                initial_drift = max(initial_drift, 1e-12)
            self.drift_scale = jnp.full(self.n_neurons, initial_drift)

        _validate_filter_numerics(self.prior_cov(), n_time=self._n_time)

        # _is_fitted stays False (set at entry) until _finalize_sgd confirms a
        # finite final log-likelihood.
        return super().fit_sgd(
            Z,
            spk,
            valid,
            optimizer=optimizer,
            num_steps=num_steps,
            verbose=verbose,
            convergence_tol=convergence_tol,
        )

    # --- prediction / scoring ---

    def _clear_posteriors(self) -> None:
        """Drop any (possibly NaN) posterior arrays left by a failed fit."""
        self.smoother_mean = None
        self.smoother_cov = None
        self.smoother_cross_cov = None
        self.filtered_mean = None
        self.filtered_cov = None

    def _check_fitted(self, method: str) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"Model not fitted (or the last fit failed); call fit(...)/fit_sgd(...) "
                f"successfully before {method}()."
            )

    def predict_rate_map(
        self, neuron_idx: int = 0, time_slice: Optional[slice] = None
    ) -> NDArray[np.float64]:
        """Per-active-bin firing rate (Hz), the log-normal posterior mean over time.

        For the linear log-intensity model and Gaussian posterior
        ``w_{c,t} | y ~ N(m_{c,t}, V_{c,t})``,

            E[rate_i | y] = mean_t exp(Phi_i m_{c,t} + 0.5 Phi_i V_{c,t} Phi_i^T),

        the exact posterior expected firing rate at active bin ``i``, averaged over the
        requested time window. This uses ``mean_t E[exp(...)]`` (not ``exp(mean_t ...)``),
        which matters for a drifting field because the two differ.

        Parameters
        ----------
        neuron_idx : int, optional
            Which neuron's map to return, by default 0.
        time_slice : slice or None, optional
            Time window to average over; the whole session when ``None``.

        Returns
        -------
        NDArray, shape (n_bins,)
            Estimated firing rate (Hz) at each active bin.
        """
        self._check_fitted("predict_rate_map")
        assert self.smoother_mean is not None
        assert self.smoother_cov is not None
        if neuron_idx < 0 or neuron_idx >= self.n_neurons:
            raise ValueError(
                f"neuron_idx={neuron_idx} out of range for n_neurons={self.n_neurons}."
            )
        if time_slice is None:
            time_slice = slice(None)
        Phi = np.asarray(self.basis.eigvecs)  # (n_bins, rank)
        means = np.asarray(self.smoother_mean[neuron_idx][time_slice])  # (T, rank)
        covs = np.asarray(self.smoother_cov[neuron_idx][time_slice])  # (T, rank, rank)
        if means.shape[0] == 0:
            raise ValueError("time_slice selects no time bins.")
        log_rate = means @ Phi.T  # (T, n_bins)
        var = np.einsum("br,trs,bs->tb", Phi, covs, Phi)  # (T, n_bins)
        rate = np.exp(log_rate + 0.5 * np.maximum(var, 0.0))
        return cast(NDArray[np.float64], rate.mean(axis=0))

    def score(
        self,
        times: NDArray[np.float64],
        trajectory: NDArray[np.float64],
        spikes: ArrayLike,
    ) -> float:
        """Held-out total marginal log-likelihood (forward filter, fitted parameters).

        Runs the same masked forward filter as the E-step (no smoothing, no parameter
        updates) with the current parameters, summed over neurons. On the training data
        this matches the fit's final marginal log-likelihood.

        Parameters
        ----------
        times : NDArray, shape (n_time,)
            Sample times in seconds.
        trajectory : NDArray, shape (n_time,) or (n_time, n_dims)
            Animal position at each time sample.
        spikes : ArrayLike, shape (n_time,) or (n_time, n_neurons)
            Per-time spike counts; the neuron axis must match the fitted
            ``n_neurons``.

        Returns
        -------
        float
            Total marginal log-likelihood, summed over neurons.
        """
        self._check_fitted("score")
        assert self.init_mean is not None
        assert self.drift_scale is not None
        spikes_arr = jnp.asarray(spikes)
        if spikes_arr.ndim == 1:
            spikes_arr = spikes_arr[:, None]
        validate_count_array(spikes_arr, "spikes", allow_empty=False)
        if spikes_arr.ndim != 2:
            raise ValueError(
                "spikes must be 1D (n_time,) or 2D (n_time, n_neurons); "
                f"got shape {spikes_arr.shape}."
            )
        if spikes_arr.shape[1] != self.n_neurons:
            raise ValueError(
                f"spikes has {spikes_arr.shape[1]} neurons but the model was fitted "
                f"with n_neurons={self.n_neurons}."
            )
        Z, spk, valid = self._design_and_spikes(times, trajectory, spikes_arr)
        if int(jnp.sum(valid)) == 0:
            raise ValueError(
                "trajectory contains no in-bounds observations for this environment; "
                "the marginal log-likelihood would depend on no data."
            )
        S = self._spectral_shape_current()
        P0 = jnp.diag(self.tau2 * S)

        def _one(m0: Array, spike_row: Array, q_c: Array) -> Array:
            _, _, marginal_ll = _masked_graph_point_process_filter(
                m0,
                P0,
                Z,
                spike_row,
                valid,
                self.transition_matrix,
                jnp.diag(q_c * S),
                dt=self.dt,
                max_log_count=self._max_log_count,
                max_newton_iter=self.max_newton_iter,
            )
            return marginal_ll

        lls = jax.vmap(_one, in_axes=(0, 0, 0))(self.init_mean, spk, self.drift_scale)
        return float(jnp.sum(lls))
