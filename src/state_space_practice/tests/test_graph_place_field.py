"""Phase-0 contract tests for the graph-Laplacian spatial substrate.

These exercise the seam between ``neurospatial`` and this package: the eigenbasis
build, the spectral shape, and the trajectory -> (design matrix, occupancy, spike
counts) helpers. They are the tripwires for the interface bugs (row misalignment,
wrong Laplacian weighting, off-by-component null modes) called out in the plan.
"""

import networkx as nx
import numpy as np
import pytest

neurospatial = pytest.importorskip("neurospatial")
from neurospatial import Environment  # noqa: E402
from neurospatial.simulation.mazes.w_maze import make_w_maze  # noqa: E402

from state_space_practice.graph_place_field import (  # noqa: E402
    GraphBasis,
    bin_occupancy,
    bin_spike_counts,
    build_graph_basis,
    graph_design_matrix,
    laplacian_matches_distance_weight,
    spectral_shape,
)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def small_grid_env():
    """A small connected grid environment (deterministic)."""
    pos = np.random.default_rng(42).uniform(0, 20, (400, 2))
    return Environment.from_samples(pos, bin_size=5.0)


@pytest.fixture(scope="module")
def w_maze_env():
    """The W-maze 2D environment (a connected branching track)."""
    return make_w_maze().env_2d


@pytest.fixture(scope="module")
def two_component_env():
    """An explicitly disconnected environment (two far-apart clusters)."""
    clusters = np.r_[
        np.random.default_rng(2).uniform(0, 10, (200, 2)),
        np.random.default_rng(3).uniform(50, 60, (200, 2)),
    ]
    return Environment.from_samples(clusters, bin_size=2.5)


# --------------------------------------------------------------------------- Laplacian
def test_laplacian_is_distance_weighted(small_grid_env):
    # L = D @ D.T equals the distance-weighted nx Laplacian, not the unweighted one.
    assert laplacian_matches_distance_weight(small_grid_env)
    L_unw = nx.laplacian_matrix(
        small_grid_env.connectivity,
        nodelist=range(small_grid_env.n_bins),
        weight=None,
    ).toarray()
    D = small_grid_env.get_differential_operator()
    L = (D @ D.T).toarray()
    assert not np.allclose(L, L_unw)  # guard: distance-weighting actually matters


def test_wmaze_is_connected(w_maze_env):
    basis = build_graph_basis(w_maze_env)
    assert basis.n_components == 1
    assert np.count_nonzero(basis.eigvals < 1e-8) == 1  # exactly one null mode


# --------------------------------------------------------------------------- eigenbasis
def test_full_rank_reconstructs_any_field(small_grid_env):
    basis = build_graph_basis(small_grid_env)  # full rank
    Phi = basis.eigvecs
    assert Phi.shape[1] == small_grid_env.n_bins
    f = np.random.default_rng(0).standard_normal(small_grid_env.n_bins)
    recon = Phi @ (Phi.T @ f)
    assert np.linalg.norm(recon - f) / np.linalg.norm(f) < 1e-8


def test_truncation_keeps_smooth_loses_rough(small_grid_env):
    full = build_graph_basis(small_grid_env)
    n = small_grid_env.n_bins
    low = build_graph_basis(small_grid_env, rank=max(3, n // 5))
    smooth = full.eigvecs[:, 1]  # a smoothest non-constant mode
    rough = full.eigvecs[:, -1]  # a highest-frequency mode
    smooth_err = np.linalg.norm(low.eigvecs @ (low.eigvecs.T @ smooth) - smooth)
    rough_err = np.linalg.norm(low.eigvecs @ (low.eigvecs.T @ rough) - rough)
    assert smooth_err < 1e-8  # low-rank reproduces the smooth field
    assert rough_err > 0.5  # guard: it genuinely drops the rough field


def test_null_modes_retained_disconnected(two_component_env):
    basis = build_graph_basis(two_component_env)
    assert basis.n_components == 2
    assert np.count_nonzero(basis.eigvals < 1e-8) == 2
    with pytest.raises(ValueError):
        build_graph_basis(two_component_env, rank=1)  # < n_components must raise


def test_component_local_modes(two_component_env):
    basis = build_graph_basis(two_component_env)
    labels = basis.component_labels
    # every eigenvector is supported on exactly one connected component
    for k in range(basis.eigvecs.shape[1]):
        support = np.abs(basis.eigvecs[:, k]) > 1e-9
        comps = np.unique(labels[support])
        assert comps.size == 1


# --------------------------------------------------------------------------- alignment
def test_contract_alignment(small_grid_env):
    basis = build_graph_basis(small_grid_env)
    n = small_grid_env.n_bins
    assert basis.eigvecs.shape[0] == n
    assert basis.component_labels.shape[0] == n
    # a trajectory that sits exactly at bin-center b returns Phi[b]
    b = n // 2
    center = np.asarray(small_grid_env.bin_centers)[b]
    times = np.array([0.0, 0.02])
    traj = np.vstack([center, center])
    Z, valid = graph_design_matrix(small_grid_env, basis, times, traj)
    assert valid.all()
    np.testing.assert_allclose(Z[0], basis.eigvecs[b])


def test_design_matrix_dedup_false_and_out_of_bounds(small_grid_env):
    basis = build_graph_basis(small_grid_env)
    center = np.asarray(small_grid_env.bin_centers)[0]
    # three identical in-bounds samples then one far out-of-bounds
    traj = np.vstack([center, center, center, [1e6, 1e6]])
    times = np.arange(4, dtype=float) * 0.02
    Z, valid = graph_design_matrix(small_grid_env, basis, times, traj)
    assert Z.shape[0] == 4  # dedup=False keeps every row (no collapse of repeats)
    assert valid.tolist() == [True, True, True, False]
    assert np.allclose(Z[3], 0.0)  # out-of-bounds row is zeroed, not Phi[-1]


def test_bin_spikes_and_occupancy_aligned(small_grid_env):
    basis = build_graph_basis(small_grid_env)
    centers = np.asarray(small_grid_env.bin_centers)
    b = 3
    times = np.arange(5, dtype=float) * 0.1
    traj = np.tile(centers[b], (5, 1))
    spikes = np.array([2.0, 0.0, 1.0, 0.0, 3.0])  # 6 spikes, all in bin b
    counts = bin_spike_counts(small_grid_env, spikes, times, traj, basis)
    occ = bin_occupancy(small_grid_env, times, traj, dt=0.1)
    assert counts.shape == (small_grid_env.n_bins, 1)
    assert counts[b, 0] == 6.0
    assert counts.sum() == 6.0
    assert np.isclose(occ[b], 5 * 0.1)  # 5 samples * dt in bin b
    # alignment invariant: no positive-count / zero-occupancy bin
    assert not np.any((counts.sum(axis=1) > 0) & (occ == 0))


# --------------------------------------------------------------------------- caching
def test_basis_cache_rank_safe(small_grid_env):
    b10 = build_graph_basis(small_grid_env, rank=10)
    b20 = build_graph_basis(small_grid_env, rank=20)
    assert b10.eigvecs.shape[1] == 10
    assert b20.eigvecs.shape[1] == 20
    # rank-10 basis is exactly the leading slice of the rank-20 basis (same cached system)
    np.testing.assert_allclose(b10.eigvecs, b20.eigvecs[:, :10])
    np.testing.assert_allclose(b10.eigvals, b20.eigvals[:10])


def test_bandwidth_rank_floored_and_monotone(small_grid_env):
    basis = build_graph_basis(small_grid_env, sigma=3.0)
    assert basis.eigvecs.shape[1] >= basis.n_components
    # a larger bandwidth keeps no more modes than a smaller one
    wide = build_graph_basis(small_grid_env, sigma=8.0)
    assert wide.eigvecs.shape[1] <= basis.eigvecs.shape[1]


# --------------------------------------------------------------------------- spectral
def test_spectral_shape_finite_at_null():
    eigvals = np.array([0.0, 0.5, 2.0])
    S = spectral_shape(eigvals, kappa2=0.25, alpha=1.0)
    assert np.all(np.isfinite(S))
    assert np.isclose(S[0], 0.25 ** (-1.0))  # null mode -> kappa2 ** (-alpha)
    assert np.all(np.diff(S) < 0)  # decreasing in lambda
    with pytest.raises(ValueError):
        spectral_shape(eigvals, kappa2=0.0)


def test_read_only_outputs(small_grid_env):
    basis = build_graph_basis(small_grid_env, rank=8)
    assert isinstance(basis, GraphBasis)
    with pytest.raises(ValueError):
        basis.eigvecs[0, 0] = 1.0  # arrays are read-only


def test_basis_cache_invalidates_on_laplacian_change(monkeypatch):
    """A changed Laplacian (e.g. a re-fit env) must rebuild the basis, not reuse
    the stale cached eigensystem. Uses a fresh env so the module-scoped fixtures
    are not polluted."""
    pos = np.random.default_rng(7).uniform(0, 20, (400, 2))
    env = Environment.from_samples(pos, bin_size=5.0)
    basis0 = build_graph_basis(env)

    d_original = env.get_differential_operator()
    # Scaling D scales L = D @ D.T by 4, so every eigenvalue changes.
    monkeypatch.setattr(env, "get_differential_operator", lambda: d_original * 2.0)
    basis1 = build_graph_basis(env)

    assert basis0.env_key != basis1.env_key
    assert not np.allclose(basis0.eigvals, basis1.eigvals)


def test_consumers_reject_mismatched_env(small_grid_env, two_component_env):
    """A basis built for one env must not be silently used with another."""
    basis = build_graph_basis(small_grid_env)
    centers = np.asarray(two_component_env.bin_centers)
    times = np.array([0.0, 0.1])
    traj = np.vstack([centers[0], centers[0]])
    with pytest.raises(ValueError, match="different"):
        graph_design_matrix(two_component_env, basis, times, traj)
    with pytest.raises(ValueError, match="different"):
        bin_spike_counts(two_component_env, np.ones((2, 1)), times, traj, basis)


def test_bin_spike_counts_rejects_time_row_mismatch(small_grid_env):
    """Misaligned spikes vs trajectory rows must raise, not silently misbin."""
    basis = build_graph_basis(small_grid_env)
    centers = np.asarray(small_grid_env.bin_centers)
    times = np.arange(5, dtype=float) * 0.1
    traj = np.tile(centers[0], (5, 1))
    spikes = np.ones((4, 1))  # 4 rows != 5 time rows
    with pytest.raises(ValueError, match="time rows"):
        bin_spike_counts(small_grid_env, spikes, times, traj, basis)


def test_graph_design_matrix_interpolation_validation(small_grid_env):
    basis = build_graph_basis(small_grid_env)
    times = np.array([0.0])
    traj = np.asarray(small_grid_env.bin_centers)[:1]
    with pytest.raises(NotImplementedError, match="linear"):
        graph_design_matrix(small_grid_env, basis, times, traj, interpolation="linear")
    with pytest.raises(ValueError, match="interpolation"):
        graph_design_matrix(small_grid_env, basis, times, traj, interpolation="cubic")


def test_bin_occupancy_rejects_nonpositive_dt(small_grid_env):
    times = np.array([0.0, 0.1])
    traj = np.asarray(small_grid_env.bin_centers)[:2]
    with pytest.raises(ValueError, match="dt"):
        bin_occupancy(small_grid_env, times, traj, dt=0.0)


def test_spectral_shape_rejects_nonpositive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        spectral_shape(np.array([0.0, 1.0]), kappa2=0.25, alpha=0.0)


def test_build_graph_basis_rejects_nonpositive_sigma(small_grid_env):
    with pytest.raises(ValueError, match="sigma"):
        build_graph_basis(small_grid_env, sigma=0.0)
