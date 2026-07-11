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
The basis is built from the **public** ``Environment.get_differential_operator()``,
whose documented identity is ``L = D @ D.T`` — a Laplacian **weighted by edge
``"distance"``** (``D`` uses ``sqrt(distance)`` per edge). Note ``neurospatial``'s *own*
diffusion smoothing (``Environment.smooth``/``diffuse``) uses a different, **finite
-volume** Laplacian (exposed only privately via ``Environment._diffusion_geometry``),
whose spectrum differs materially. We deliberately use the public, well-defined
distance-weighted operator; if a future goal is to match ``neurospatial``'s diffusion
convention exactly, switch to the finite-volume operator (and ideally ask upstream to
expose it publicly). Parity checks against external estimators (e.g. the ``non_local
_detector`` MRF) feed *our* basis to both sides, so they are robust to this choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

if TYPE_CHECKING:
    from neurospatial import Environment

__all__ = [
    "GraphBasis",
    "build_graph_basis",
    "spectral_shape",
    "graph_design_matrix",
    "bin_occupancy",
    "bin_spike_counts",
]

# Attribute used to cache the full eigensystem on an Environment instance (keyed by the
# Laplacian's identity so a re-fit environment does not reuse a stale basis).
_CACHE_ATTR = "_graph_place_field_basis_cache"


class GraphBasis(NamedTuple):
    """Truncated graph-Laplacian eigenbasis over an environment's active bins.

    Attributes
    ----------
    eigvecs : NDArray, shape (n_bins, rank)
        Smoothest eigenvectors of ``L = D @ D.T`` as columns, ordered by ascending
        eigenvalue. Row ``i`` is active bin ``i`` (same space as ``env.bin_sequence``).
        On a disconnected graph the eigenvectors are component-local (zero outside
        their connected component).
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
    env_key : tuple
        Identity fingerprint ``(n_bins, nnz, |L|-sum)`` of the environment's Laplacian
        this basis was built from. Consumers (:func:`graph_design_matrix`,
        :func:`bin_spike_counts`) check it to refuse a basis built for a different
        (or since-refit) environment, which would otherwise return silently wrong
        design rows / counts.
    """

    eigvecs: NDArray[np.float64]
    eigvals: NDArray[np.float64]
    component_labels: NDArray[np.int_]
    bin_sizes: NDArray[np.float64]
    n_components: int
    env_key: tuple[int, int, float]


def _distance_weighted_laplacian(env: "Environment") -> sp.csr_matrix:
    """Return ``L = D @ D.T`` (distance-weighted graph Laplacian) as CSR."""
    D = env.get_differential_operator()
    return (D @ D.T).tocsr()


def _laplacian_key(laplacian: sp.spmatrix) -> tuple[int, int, float]:
    """Cheap identity fingerprint of a Laplacian: ``(n_bins, nnz, |data|-sum)``.

    ``abs`` because a graph Laplacian's signed entries sum to ~0; the absolute-value
    sum distinguishes different edge weightings at the same sparsity pattern.
    """
    return (
        int(laplacian.shape[0]),
        int(laplacian.nnz),
        float(np.round(np.abs(laplacian.data).sum(), 6)),
    )


def _env_key(env: "Environment") -> tuple[int, int, float]:
    """Fingerprint of ``env``'s current distance-weighted Laplacian."""
    return _laplacian_key(_distance_weighted_laplacian(env))


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
    env_key = _env_key(env)
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
) -> GraphBasis:
    """Build (and cache) the truncated graph-Laplacian eigenbasis for ``env``.

    Uses the public distance-weighted Laplacian ``L = D @ D.T`` from
    ``env.get_differential_operator()`` (see the module docstring on the Laplacian
    choice). ``neurospatial`` does not expose its diffusion eigenbasis publicly
    (``Environment._diffusion_eigenbasis`` is private), so the basis is built here.

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

    Returns
    -------
    GraphBasis
        The truncated eigenbasis; arrays are read-only and cached on ``env``.
    """
    laplacian = _distance_weighted_laplacian(env)
    n_components, labels = connected_components(laplacian, directed=False)

    # Cache the full sorted eigensystem keyed by the Laplacian's identity (shape + nnz +
    # data checksum); a cached full basis serves any smaller rank by slicing.
    cache = getattr(env, _CACHE_ATTR, None)
    key = _laplacian_key(laplacian)
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
    L = _distance_weighted_laplacian(env).toarray()
    L_nx = nx.laplacian_matrix(
        env.connectivity, nodelist=range(env.n_bins), weight="distance"
    ).toarray()
    return bool(np.allclose(L, L_nx, atol=1e-9))
