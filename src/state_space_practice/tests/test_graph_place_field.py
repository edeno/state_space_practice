"""Tests for the graph-Laplacian place-field model and its spatial substrate.

These exercise the seam between ``neurospatial`` and this package (eigenbasis build,
spectral shape, and the trajectory -> (design matrix, occupancy, spike counts)
helpers) plus the static and drifting model on top of it. The substrate tests are
tripwires for the interface bugs that silently corrupt place fields: row
misalignment, wrong Laplacian weighting, and off-by-component null modes.
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
    build_graph_laplacian,
    graph_design_matrix,
    laplacian_matches_distance_weight,
    spectral_shape,
    validate_graph_laplacian,
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


def test_default_laplacian_convention_is_explicit_and_backward_compatible(
    small_grid_env,
):
    L_default = build_graph_laplacian(small_grid_env)
    D = small_grid_env.get_differential_operator()
    L_public = (D @ D.T).toarray()

    np.testing.assert_allclose(L_default.toarray(), L_public, atol=1e-12)
    basis = build_graph_basis(small_grid_env, rank=8)
    assert basis.laplacian_convention == "distance"


def test_inverse_distance_laplacian_uses_conductance_weights(small_grid_env):
    L_distance = build_graph_laplacian(small_grid_env, convention="distance")
    L_inverse = build_graph_laplacian(small_grid_env, convention="inverse_distance")

    graph = small_grid_env.connectivity.copy()
    for _source, _target, edge_data in graph.edges(data=True):
        edge_data["test_conductance"] = 1.0 / float(edge_data["distance"])
    expected = nx.laplacian_matrix(
        graph,
        nodelist=range(small_grid_env.n_bins),
        weight="test_conductance",
    ).toarray()

    np.testing.assert_allclose(L_inverse.toarray(), expected, atol=1e-12)
    assert not np.allclose(L_inverse.toarray(), L_distance.toarray())
    basis = build_graph_basis(
        small_grid_env,
        rank=8,
        laplacian_convention="inverse_distance",
    )
    assert basis.laplacian_convention == "inverse_distance"


@pytest.mark.parametrize("convention", ["distance", "inverse_distance"])
def test_constructed_laplacians_pass_structural_validation(small_grid_env, convention):
    laplacian = build_graph_laplacian(small_grid_env, convention=convention)
    validate_graph_laplacian(laplacian, n_bins=small_grid_env.n_bins)


def test_laplacian_convention_validation_rejects_unknown_value(small_grid_env):
    with pytest.raises(ValueError, match="laplacian convention"):
        build_graph_laplacian(small_grid_env, convention="finite_volume")


def test_graph_laplacian_validation_rejects_non_laplacian_matrix():
    nonsymmetric = np.array([[1.0, -1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        validate_graph_laplacian(nonsymmetric, n_bins=2)

    nonzero_rows = np.array([[2.0, -1.0], [-1.0, 2.0]])
    with pytest.raises(ValueError, match="row sums"):
        validate_graph_laplacian(nonzero_rows, n_bins=2)


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


# --------------------------------------------------------------------------- static GLM
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from state_space_practice.graph_place_field import (  # noqa: E402
    fit_static_graph_glm,
    parity_penalty,
    spectral_precision,
)


def test_spectral_precision_matches_inverse_prior():
    eigvals = np.array([0.0, 0.5, 2.0])
    tau2, kappa2, alpha = 0.7, 0.25, 1.0
    prec = spectral_precision(eigvals, tau2=tau2, kappa2=kappa2, alpha=alpha)
    # precision is the reciprocal of the prior variance tau2 * S
    from state_space_practice.graph_place_field import spectral_shape

    S = spectral_shape(eigvals, kappa2, alpha)
    np.testing.assert_allclose(prec, 1.0 / (tau2 * S))
    assert np.all(np.isfinite(prec))  # null mode finite (kappa2 > 0)


def test_parity_penalty_leaves_null_modes_unpenalized():
    eigvals = np.array([0.0, 0.0, 0.8, 3.0])  # two null modes
    pen = parity_penalty(eigvals, n_components=2)
    assert np.all(pen[:2] == 0.0)  # per-component intercepts, unpenalized
    np.testing.assert_allclose(
        pen[2:], eigvals[2:]
    )  # positive modes penalized by lambda


def test_static_glm_recovers_smooth_field(small_grid_env):
    # Simulate per-bin Poisson counts from a KNOWN smooth field in the eigenbasis,
    # then check the penalized GLM recovers the field (log-rate) it was generated from.
    # NOTE: small_grid_env is only ~25 bins; rank and exposure are calibrated so a
    # correct solver clears corr > 0.95 with margin (empirically mean 0.99 / min 0.97
    # across seeds) while a broken gradient/Hessian still fails. Offset-bug sensitivity
    # is covered separately by test_newton_map_matches_independent_optimizer, which
    # uses the real log(occupancy) offset; with uniform occupancy a correlation test
    # cannot see offset bugs.
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=6)
    rng = np.random.default_rng(0)
    # A smooth field: energy only in the low modes.
    w_true = np.zeros(6)
    w_true[:4] = rng.standard_normal(4)
    eta_true = basis.eigvecs @ w_true  # (n_bins,) log-rate
    occ = np.full(small_grid_env.n_bins, 300.0)  # uniform exposure (s) per bin
    counts = rng.poisson(np.exp(eta_true) * occ)

    # Weak ridge on the rough modes; recovery should be accurate where data is dense.
    prec = spectral_precision(basis.eigvals, tau2=100.0, kappa2=1e-2, alpha=1.0)
    w_hat, cov = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    eta_hat = basis.eigvecs @ np.asarray(w_hat)
    # Correlation between recovered and true log-rate is high.
    corr = np.corrcoef(eta_hat, eta_true)[0, 1]
    assert corr > 0.95
    # Laplace covariance is PSD.
    assert np.all(np.linalg.eigvalsh(np.asarray(cov)) > -1e-8)


def test_static_glm_ignores_unvisited_bins(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=8)
    occ = np.zeros(small_grid_env.n_bins)
    occ[: small_grid_env.n_bins // 2] = 3.0  # only half the bins visited
    counts = np.zeros(small_grid_env.n_bins)
    counts[: small_grid_env.n_bins // 2] = 1.0
    prec = spectral_precision(basis.eigvals, tau2=10.0, kappa2=1.0, alpha=1.0)
    w_hat, _ = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    # Unvisited bins must not blow up the fit (finite, no NaN).
    assert np.all(np.isfinite(np.asarray(w_hat)))


def test_static_glm_multineuron_matches_per_neuron(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=8)
    rng = np.random.default_rng(1)
    occ = np.full(small_grid_env.n_bins, 4.0)
    counts = rng.poisson(0.3, size=(small_grid_env.n_bins, 3)).astype(float)
    prec = spectral_precision(basis.eigvals, tau2=10.0, kappa2=1.0, alpha=1.0)
    w_multi, _ = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    assert w_multi.shape == (3, 8)
    for j in range(3):
        w_j, _ = fit_static_graph_glm(counts[:, j], occ, basis.eigvecs, prec)
        np.testing.assert_allclose(np.asarray(w_multi[j]), np.asarray(w_j), atol=1e-8)


def test_static_glm_damps_high_count_newton_step():
    """A high-count intercept must not take the divergent full Newton steps."""
    from state_space_practice.graph_place_field import static_log_evidence

    counts = np.array([100.0])
    occupancy = np.array([1.0])
    eigvecs = np.array([[1.0]])
    penalty = np.array([0.1])

    weights, cov = fit_static_graph_glm(counts, occupancy, eigvecs, penalty)

    # The MAP solves exp(w) + 0.1 w = 100. A full first Newton step is about 90,
    # so this regression specifically requires damping/backtracking.
    np.testing.assert_allclose(np.asarray(weights), [4.600559011835084], atol=1e-8)
    assert np.all(np.isfinite(np.asarray(cov)))

    evidence = static_log_evidence(
        counts,
        occupancy,
        eigvecs,
        np.array([0.0]),
        tau2=10.0,
        kappa2=1.0,
    )
    assert np.isfinite(evidence)


# --------------------------------------------------------------------------- evidence
from state_space_practice.graph_place_field import (  # noqa: E402
    select_tau2_by_evidence,
    static_log_evidence,
)


def _simulate_smooth_counts(basis, rng, tau2_gen, occ_val=5.0):
    """Per-bin counts from w ~ N(0, tau2_gen * S), rate = exp(Phi w)."""
    from state_space_practice.graph_place_field import spectral_shape

    S = spectral_shape(basis.eigvals, kappa2=1e-2, alpha=1.0)
    rank = basis.eigvecs.shape[1]
    w = rng.standard_normal(rank) * np.sqrt(tau2_gen * S)
    eta = basis.eigvecs @ w
    n_bins = basis.eigvecs.shape[0]
    occ = np.full(n_bins, occ_val)
    counts = rng.poisson(np.exp(eta) * occ).astype(float)
    return counts, occ


def test_evidence_is_maximized_near_selected_tau2(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=12)
    rng = np.random.default_rng(3)
    counts, occ = _simulate_smooth_counts(basis, rng, tau2_gen=1.0)
    tau2_hat = select_tau2_by_evidence(counts, occ, basis, kappa2=1e-2, alpha=1.0)
    ev_at = static_log_evidence(
        counts, occ, basis.eigvecs, basis.eigvals, tau2=tau2_hat, kappa2=1e-2
    )
    # The selected tau2 beats both a 10x-too-large and a 10x-too-small amplitude.
    ev_hi = static_log_evidence(
        counts, occ, basis.eigvecs, basis.eigvals, tau2=10 * tau2_hat, kappa2=1e-2
    )
    ev_lo = static_log_evidence(
        counts, occ, basis.eigvecs, basis.eigvals, tau2=0.1 * tau2_hat, kappa2=1e-2
    )
    assert ev_at >= ev_hi
    assert ev_at >= ev_lo
    assert 1e-4 < tau2_hat < 1e4  # inside the search bounds (not railed)


# ------------------------------------------------- static parity + W-maze recovery
@pytest.mark.slow
def test_newton_map_matches_independent_optimizer(small_grid_env):
    """Our Newton MAP equals a scipy L-BFGS optimum on the identical penalized NLL."""
    import scipy.optimize

    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=10)
    rng = np.random.default_rng(5)
    occ = np.full(small_grid_env.n_bins, 4.0)
    w_true = np.zeros(10)
    w_true[:4] = rng.standard_normal(4)
    counts = rng.poisson(np.exp(basis.eigvecs @ w_true) * occ).astype(float)
    prec = spectral_precision(basis.eigvals, tau2=50.0, kappa2=1e-2, alpha=1.0)

    Phi = np.asarray(basis.eigvecs)
    log_occ = np.log(occ)

    def nll(w):
        eta = Phi @ w + log_occ
        return float(np.sum(np.exp(eta) - counts * eta) + 0.5 * np.sum(prec * w**2))

    def grad(w):
        eta = Phi @ w + log_occ
        mu = np.exp(eta)
        return Phi.T @ (mu - counts) + prec * w

    ref = scipy.optimize.minimize(nll, np.zeros(10), jac=grad, method="L-BFGS-B")
    w_hat, _ = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    np.testing.assert_allclose(np.asarray(w_hat), ref.x, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_static_field_recovers_wmaze_place_cells(w_maze_env):
    """Recovered rate map correlates with the true place fields on the W-maze."""
    from neurospatial.simulation import simulate_session

    from state_space_practice.graph_place_field import build_graph_basis
    from state_space_practice.preprocessing import bin_spike_times

    session = simulate_session(
        w_maze_env, duration=600.0, n_cells=6, seed=7, show_progress=False
    )
    dt = float(np.median(np.diff(session.times)))
    # Bin each cell's spike times onto the position time grid.
    counts_tc = bin_spike_times(session.spike_trains, session.times).astype(float)

    basis = build_graph_basis(w_maze_env, sigma=15.0)  # bandwidth-truncated rank
    counts_bin = bin_spike_counts(
        w_maze_env, counts_tc, session.times, session.positions, basis
    )
    occ = bin_occupancy(w_maze_env, session.times, session.positions, dt)
    tau2 = select_tau2_by_evidence(counts_bin, occ, basis, kappa2=1e-2)
    prec = spectral_precision(basis.eigvals, tau2=tau2, kappa2=1e-2)
    w_hat, _ = fit_static_graph_glm(counts_bin, occ, basis.eigvecs, prec)

    bin_centers = np.asarray(w_maze_env.bin_centers)
    visited = occ > 0
    corrs = []
    for j, model in enumerate(session.models):
        rate_true = np.asarray(model.firing_rate(bin_centers))
        rate_hat = np.exp(np.asarray(basis.eigvecs @ w_hat[j]))
        corrs.append(np.corrcoef(rate_hat[visited], rate_true[visited])[0, 1])
    # The graph-GP field tracks the true place fields on the maze.
    assert np.median(corrs) > 0.6


# --------------------------------------------------------------- model construction
from state_space_practice.graph_place_field import GraphPlaceFieldModel  # noqa: E402


def test_model_builds_diagonal_psd_prior_and_drift(small_grid_env):
    model = GraphPlaceFieldModel(
        small_grid_env, dt=0.02, rank=10, kappa2=0.5, alpha=1.0, tau2=2.0
    )
    assert model.rank == 10
    P0 = np.asarray(model.prior_cov())
    # P0 = tau2 * S, diagonal, strictly PSD (kappa2 > 0 keeps null modes finite).
    assert np.allclose(P0, np.diag(np.diag(P0)))
    assert np.all(np.linalg.eigvalsh(P0) > 0)
    from state_space_practice.graph_place_field import spectral_shape

    S = spectral_shape(model.basis.eigvals, 0.5, 1.0)
    np.testing.assert_allclose(np.diag(P0), 2.0 * S)
    Q = np.asarray(model.drift_cov(0.01))
    np.testing.assert_allclose(np.diag(Q), 0.01 * S)
    assert np.all(np.linalg.eigvalsh(Q) > 0)


def test_model_rejects_bad_hyperparameters(small_grid_env):
    with pytest.raises(ValueError, match="dt"):
        GraphPlaceFieldModel(small_grid_env, dt=0.0)
    with pytest.raises(ValueError, match="kappa2"):
        GraphPlaceFieldModel(small_grid_env, dt=0.02, kappa2=0.0)
    with pytest.raises(ValueError, match="tau2"):
        GraphPlaceFieldModel(small_grid_env, dt=0.02, tau2=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        GraphPlaceFieldModel(small_grid_env, dt=0.02, alpha=0.0)


# -------------------------------------------------------- E-step (masked filter)
def _toy_trajectory(env, n_time, seed=0):
    """A trajectory that visits real bin centers (so all rows are in-bounds)."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(env.bin_centers)
    idx = rng.integers(0, centers.shape[0], size=n_time)
    times = np.arange(n_time, dtype=float) * 0.02
    return times, centers[idx]


@pytest.mark.slow
def test_estep_returns_finite_total_ll_and_stores_posteriors(small_grid_env):
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=8)
    n_time = 150
    times, traj = _toy_trajectory(small_grid_env, n_time, seed=1)
    rng = np.random.default_rng(2)
    spikes = rng.poisson(0.05, size=(n_time, 3)).astype(float)
    model.n_neurons = 3
    model.drift_scale = jnp.full(3, 1e-3)
    model.init_mean = jnp.zeros((3, model.rank))
    Z, spk, valid = model._design_and_spikes(times, traj, spikes)
    ll = model._e_step(Z, spk, valid)
    assert np.isfinite(ll)
    assert model.smoother_mean.shape == (3, n_time, model.rank)


@pytest.mark.slow
def test_estep_neurons_are_independent(small_grid_env):
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=8)
    n_time = 120
    times, traj = _toy_trajectory(small_grid_env, n_time, seed=3)
    rng = np.random.default_rng(4)
    spikes = rng.poisson(0.05, size=(n_time, 2)).astype(float)
    model.n_neurons = 2
    model.drift_scale = jnp.full(2, 1e-3)
    model.init_mean = jnp.zeros((2, model.rank))
    Z, spk, valid = model._design_and_spikes(times, traj, spikes)
    model._e_step(Z, spk, valid)
    sm0 = np.asarray(model.smoother_mean[0]).copy()
    # Perturb neuron 1's spikes only; neuron 0's smoothed trajectory must not change.
    spikes2 = spikes.copy()
    spikes2[:, 1] += 1.0
    Z2, spk2, valid2 = model._design_and_spikes(times, traj, spikes2)
    model._e_step(Z2, spk2, valid2)
    np.testing.assert_allclose(sm0, np.asarray(model.smoother_mean[0]), atol=1e-9)


def test_graph_filter_uses_p0_at_row_zero_and_propagates_masked_rows():
    from state_space_practice.graph_place_field import (
        _masked_graph_point_process_filter,
    )

    _, filtered_cov, _ = _masked_graph_point_process_filter(
        init_mean=jnp.zeros(1),
        init_cov=jnp.array([[2.0]]),
        design_matrix=jnp.zeros((3, 1)),
        spikes=jnp.zeros(3),
        valid=jnp.array([True, False, True]),
        transition_matrix=jnp.eye(1),
        process_cov=jnp.array([[3.0]]),
        dt=0.02,
        max_log_count=20.0,
        max_newton_iter=1,
    )

    # Zero design rows carry no state information. Row zero therefore keeps P0
    # exactly, while the masked middle row still advances the random walk.
    np.testing.assert_allclose(
        np.asarray(filtered_cov[:, 0, 0]), [2.0, 5.0, 8.0], atol=1e-7
    )


def test_tau_mstep_includes_fixed_prior_mean_residual(small_grid_env):
    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=1,
        kappa2=1.0,
        update_drift_scale=False,
        update_amplitude=True,
        update_init_mean=False,
    )
    model.init_mean = jnp.zeros((1, 1))
    model.drift_scale = jnp.array([1e-3])
    model.smoother_mean = jnp.array([[[10.0]]])
    model.smoother_cov = jnp.array([[[[1.0]]]])
    model.smoother_cross_cov = jnp.zeros((1, 0, 1, 1))

    shape = float(model._spectral_shape_current()[0])
    model._m_step()

    assert model.tau2 == pytest.approx((1.0 + 10.0**2) / shape)
    np.testing.assert_array_equal(np.asarray(model.init_mean), [[0.0]])


# --------------------------------------------------------------------------- fit (EM)
def _simulate_drifting_spikes(env, basis, dt, n_time, q_c, tau2, kappa2, seed):
    """Poisson spikes from a coefficient random walk w_t = w_{t-1} + N(0, q_c*S)."""
    from state_space_practice.graph_place_field import spectral_shape

    rng = np.random.default_rng(seed)
    centers = np.asarray(env.bin_centers)
    idx = rng.integers(0, centers.shape[0], size=n_time)
    times = np.arange(n_time, dtype=float) * dt
    traj = centers[idx]
    S = spectral_shape(basis.eigvals, kappa2, 1.0)
    rank = basis.eigvecs.shape[1]
    w = rng.standard_normal(rank) * np.sqrt(tau2 * S)  # w_0 ~ N(0, tau2 S)
    Z = basis.eigvecs[idx]  # (n_time, rank), design at visited bins
    spikes = np.zeros(n_time)
    for t in range(n_time):
        if t > 0:
            w = w + rng.standard_normal(rank) * np.sqrt(q_c * S)
        rate = np.exp(Z[t] @ w)
        spikes[t] = rng.poisson(rate * dt)
    return times, traj, spikes


@pytest.mark.slow
def test_fit_em_is_monotone_under_rollback(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=8)
    times, traj, spikes = _simulate_drifting_spikes(
        small_grid_env,
        basis,
        dt=0.02,
        n_time=800,
        q_c=1e-3,
        tau2=1.0,
        kappa2=1e-2,
        seed=0,
    )
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=8, kappa2=1e-2)
    lls = model.fit(times, traj, spikes, max_iter=30, verbose=False)
    diffs = np.diff(lls)
    # GEM rollback guarantees the accepted LL sequence never decreases.
    assert np.all(diffs >= -1e-6)
    assert len(lls) >= 2  # guard: EM actually iterated


@pytest.mark.slow
def test_terminal_mstep_returns_matching_ll_and_posteriors(small_grid_env):
    times, traj = _toy_trajectory(small_grid_env, 60, seed=11)
    spikes = np.random.default_rng(12).poisson(0.05, size=60).astype(float)
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=4)

    lls = model.fit(times, traj, spikes, max_iter=1, verbose=False)
    stored_mean = np.asarray(model.smoother_mean).copy()
    Z, spk, valid = model._design_and_spikes(times, traj, spikes)
    fresh_ll = model._e_step(Z, spk, valid)

    assert lls[-1] == pytest.approx(fresh_ll, abs=1e-8)
    np.testing.assert_allclose(stored_mean, np.asarray(model.smoother_mean), atol=1e-8)


@pytest.mark.slow
def test_missing_positions_keep_timeline_and_validate_observation_count(
    small_grid_env,
):
    center = np.asarray(small_grid_env.bin_centers[0])
    outside = np.full_like(center, 1e9)
    times = np.arange(3, dtype=float) * 0.02
    one_valid_trajectory = np.stack((center, outside, outside))
    spikes = np.zeros(3)

    frozen = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=3,
        update_amplitude=False,
        update_init_mean=False,
    )
    frozen.fit(
        times,
        one_valid_trajectory,
        spikes,
        max_iter=1,
        warm_start=False,
        verbose=False,
    )
    assert frozen.smoother_mean.shape[1] == 3
    assert np.all(np.isfinite(np.asarray(frozen.drift_scale)))

    updating = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=3,
        update_drift_scale=True,
    )
    updating.fit(times, one_valid_trajectory, spikes, max_iter=1, verbose=False)
    assert np.all(np.isfinite(np.asarray(updating.drift_scale)))

    with pytest.raises(ValueError, match="at least two time rows"):
        updating.fit(
            times[:1],
            one_valid_trajectory[:1],
            spikes[:1],
            max_iter=1,
            verbose=False,
        )

    with pytest.raises(ValueError, match="no in-bounds"):
        frozen.fit(
            times,
            np.stack((outside, outside, outside)),
            spikes,
            max_iter=1,
            verbose=False,
        )


@pytest.mark.slow
def test_cold_start_fit_resets_init_mean_when_neuron_count_changes(small_grid_env):
    times, traj = _toy_trajectory(small_grid_env, 10, seed=31)
    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=3,
        update_amplitude=False,
        update_init_mean=False,
    )
    model.fit(
        times,
        traj,
        np.zeros(10),
        max_iter=1,
        warm_start=False,
        verbose=False,
    )
    model.fit(
        times,
        traj,
        np.zeros((10, 2)),
        max_iter=1,
        warm_start=False,
        verbose=False,
    )

    assert model.init_mean.shape == (2, model.rank)
    assert model.smoother_mean.shape[0] == 2


@pytest.mark.slow
def test_fit_sgd_runs_and_default_keeps_drift_fixed(small_grid_env):
    times, traj = _toy_trajectory(small_grid_env, 30, seed=21)
    spikes = np.random.default_rng(22).poisson(0.05, size=30).astype(float)
    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=3,
        init_drift_scale=2e-3,
    )

    lls = model.fit_sgd(
        times,
        traj,
        spikes,
        num_steps=1,
        warm_start=False,
        verbose=False,
    )

    assert lls and np.all(np.isfinite(lls))
    np.testing.assert_allclose(np.asarray(model.drift_scale), [2e-3], atol=0.0)
    assert model.smoother_mean is not None


# NOTE: there is deliberately no q_c-recovery test (EM/SGD recovering the simulated
# drift scale). The Laplace-EKF marginal log-likelihood used here is monotonically
# increasing in q_c rather than peaking at the generating value, so q_c is not
# identified by this objective and EM returns approximately its initialization -- a
# recovery test would assert a false claim. This is why update_drift_scale defaults
# to False. The M-step algebra itself is correct (verified below in
# test_drift_scale_mstep_matches_closed_form, and by feeding the exact generative
# w_{c,t} trajectory into the M-step, which recovers q_c to ~1%). Drift *tracking*
# (not learning) is validated by test_smoother_beats_filter_beats_static_on_drifting_data.


@pytest.mark.slow
def test_static_limit_matches_static_estimator(small_grid_env):
    """With q_c pinned to ~0, the smoothed field equals the static GLM MAP."""
    from state_space_practice.graph_place_field import (
        build_graph_basis,
        fit_static_graph_glm,
        spectral_precision,
    )

    basis = build_graph_basis(small_grid_env, rank=8)
    times, traj, spikes = _simulate_drifting_spikes(
        small_grid_env,
        basis,
        dt=0.02,
        n_time=1500,
        q_c=0.0,
        tau2=1.0,
        kappa2=1e-2,
        seed=2,  # no drift
    )
    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=8,
        kappa2=1e-2,
        tau2=1.0,
        init_drift_scale=1e-12,
        update_drift_scale=False,
        update_amplitude=False,
    )
    model.fit(times, traj, spikes, warm_start=True, max_iter=5, verbose=False)
    # Static estimator on the same aggregated data with the matching penalty.
    counts = bin_spike_counts(small_grid_env, spikes, times, traj, basis)
    occ = bin_occupancy(small_grid_env, times, traj, dt=0.02)
    prec = spectral_precision(basis.eigvals, tau2=1.0, kappa2=1e-2)
    w_static, _ = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    # Time-averaged smoothed coefficients ~ the static MAP (drift-free limit).
    w_dyn = np.asarray(model.smoother_mean[0]).mean(axis=0)
    corr = np.corrcoef(
        np.asarray(basis.eigvecs @ w_dyn), np.asarray(basis.eigvecs @ w_static[0])
    )[0, 1]
    assert corr > 0.95


# ------------------------------------------- prediction, scoring & drift/geometry
def _simulate_field_drift_spikes(env, basis, dt, n_time, q_c, seed, rate_hz=30.0):
    """Poisson spikes from a field that drifts in the non-null modes only.

    The null (constant/baseline) mode is pinned so the overall firing rate is stable
    and only the spatial pattern drifts -- the identifiable, physiological regime (a
    freely drifting baseline is both unrealistic and makes q_c unidentifiable). Returns
    per-time truth so tests can score tracking. Uses the model's spectral shape S for
    the drift so the simulation matches the model's Q_c = q_c * S on the field modes.
    """
    rng = np.random.default_rng(seed)
    S = spectral_shape(basis.eigvals, kappa2=1.0, alpha=1.0)
    n_components = basis.n_components
    s_drift = np.asarray(S).copy()
    s_drift[:n_components] = 0.0  # baseline does not drift
    rank = basis.eigvecs.shape[1]
    centers = np.asarray(env.bin_centers)
    idx = rng.integers(0, centers.shape[0], size=n_time)
    times = np.arange(n_time, dtype=float) * dt
    traj = centers[idx]
    w = rng.standard_normal(rank) * np.sqrt(np.asarray(S))
    w[0] = np.log(rate_hz) / float(basis.eigvecs[0, 0])  # pin the baseline log-rate
    step_sd = np.sqrt(q_c * s_drift)
    Z = np.asarray(basis.eigvecs)[idx]
    spikes = np.zeros(n_time)
    eta_true = np.zeros(n_time)
    w_traj = np.zeros((n_time, rank))
    for t in range(n_time):
        if t > 0:
            w = w + rng.standard_normal(rank) * step_sd
        w_traj[t] = w
        eta_true[t] = Z[t] @ w
        spikes[t] = rng.poisson(np.exp(eta_true[t]) * dt)
    return times, traj, spikes, eta_true, w_traj


def test_predict_rate_map_requires_fit_and_valid_neuron(small_grid_env):
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=6)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_rate_map()


@pytest.mark.slow
def test_predict_rate_map_recovers_static_field(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=8)
    # q_c = 0 -> a static field; predict_rate_map should recover its rate map.
    times, traj, spikes, _eta, w_traj = _simulate_field_drift_spikes(
        small_grid_env, basis, dt=0.02, n_time=2000, q_c=0.0, seed=3
    )
    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=8,
        kappa2=1.0,
        tau2=1.0,
        update_drift_scale=False,
        update_amplitude=False,
    )
    model.fit(times, traj, spikes, max_iter=8, verbose=False)
    rate_hat = model.predict_rate_map(neuron_idx=0)
    rate_true = np.exp(np.asarray(basis.eigvecs @ w_traj[0]))  # static (Hz)
    occ = bin_occupancy(small_grid_env, times, traj, dt=0.02)
    visited = occ > 0
    corr = np.corrcoef(rate_hat[visited], rate_true[visited])[0, 1]
    assert corr > 0.9
    # Range check on the fitted model.
    with pytest.raises(ValueError, match="neuron_idx"):
        model.predict_rate_map(neuron_idx=1)


@pytest.mark.slow
def test_score_matches_fit_marginal_ll(small_grid_env):
    times, traj = _toy_trajectory(small_grid_env, 120, seed=41)
    spikes = np.random.default_rng(42).poisson(0.05, size=120).astype(float)
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=6)
    lls = model.fit(times, traj, spikes, max_iter=3, verbose=False)
    held = model.score(times, traj, spikes)
    # score() reruns the same masked forward filter with the fitted parameters, so on
    # the training data it reproduces the fit's final marginal log-likelihood.
    assert held == pytest.approx(lls[-1], rel=1e-6, abs=1e-4)
    # Wrong neuron count is rejected.
    with pytest.raises(ValueError, match="neurons"):
        model.score(times, traj, np.zeros((120, 2)))


@pytest.mark.slow
def test_smoother_beats_filter_beats_static_on_drifting_data(small_grid_env):
    from state_space_practice.graph_place_field import build_graph_basis

    basis = build_graph_basis(small_grid_env, rank=8)
    dt, kappa2, q_true = 0.02, 1.0, 1e-2
    times, traj, spikes, eta_true, _w = _simulate_field_drift_spikes(
        small_grid_env, basis, dt=dt, n_time=3000, q_c=q_true, seed=0
    )
    # Per-time design at the visited bins (same binning the model uses internally).
    Zt = np.asarray(basis.eigvecs)[_bin_ids_for_test(small_grid_env, times, traj)]
    # Fit with q fixed at the true drift scale: validate tracking, not learning.
    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=dt,
        rank=8,
        kappa2=kappa2,
        tau2=1.0,
        init_drift_scale=q_true,
        update_drift_scale=False,
        update_amplitude=False,
    )
    model.fit(times, traj, spikes, max_iter=10, verbose=False)

    eta_sm = np.einsum("tr,tr->t", Zt, np.asarray(model.smoother_mean[0]))
    eta_fi = np.einsum("tr,tr->t", Zt, np.asarray(model.filtered_mean[0]))
    counts = bin_spike_counts(small_grid_env, spikes[:, None], times, traj, basis)
    occ = bin_occupancy(small_grid_env, times, traj, dt)
    prec = spectral_precision(basis.eigvals, tau2=1.0, kappa2=kappa2)
    w_static, _ = fit_static_graph_glm(counts, occ, basis.eigvecs, prec)
    eta_st = Zt @ np.asarray(w_static[0])

    rmse_sm = np.sqrt(np.mean((eta_sm - eta_true) ** 2))
    rmse_fi = np.sqrt(np.mean((eta_fi - eta_true) ** 2))
    rmse_st = np.sqrt(np.mean((eta_st - eta_true) ** 2))
    assert rmse_sm < rmse_fi  # smoothing (future data) beats causal filtering
    assert rmse_fi < rmse_st  # tracking drift beats a single static field


@pytest.mark.slow
def test_graph_basis_does_not_smear_across_wmaze_arms(w_maze_env):
    """A field smooth in the graph basis stays on its connected arm: it leaks far less
    to a Euclidean-near bin on a different arm (across a wall) than to a graph-adjacent
    bin. This is the geometry-aware property tensor-product splines lack."""
    from scipy.sparse.csgraph import shortest_path

    centers = np.asarray(w_maze_env.bin_centers)
    n = w_maze_env.n_bins
    laplacian = build_graph_laplacian(w_maze_env, convention="distance").toarray()
    adjacency = (laplacian < -1e-12).astype(float)  # off-diagonal negatives are edges
    hops = shortest_path(adjacency, method="D", directed=False)
    eucl = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # Euclidean-nearest pair that is far apart on the graph (opposite arms).
    far = np.isfinite(hops) & (hops > 8)
    cand = np.where(far, eucl, np.inf)
    np.fill_diagonal(cand, np.inf)
    i, j = np.unravel_index(np.argmin(cand), cand.shape)
    assert hops[i, j] > 8 and eucl[i, j] < 20  # guard: pair is genuinely near + far
    # a graph-adjacent reference bin to i (as Euclidean-far as possible within 3 hops)
    near = (hops[i] > 0) & (hops[i] <= 3)
    k = int(np.flatnonzero(near)[np.argmax(eucl[i][near])])

    basis = build_graph_basis(w_maze_env, rank=15)
    Phi = np.asarray(basis.eigvecs)
    bump = np.zeros(n)
    bump[i] = 1.0
    recon = Phi @ (Phi.T @ bump)  # low-rank graph reconstruction of a delta at bin i
    leak_far = abs(recon[j]) / abs(recon[i])  # to the graph-far / Euclidean-near bin
    leak_near = abs(recon[k]) / abs(recon[i])  # to the graph-adjacent bin
    assert leak_far < 0.1  # almost no mass crosses to the other arm
    assert leak_far < 0.3 * leak_near  # and far less than to a graph-adjacent bin


def _bin_ids_for_test(env, times, trajectory):
    """Active-bin ids on the time grid (mirrors the module's internal binning)."""
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim == 1:
        traj = traj[:, None]
    ids = env.bin_sequence(
        np.asarray(times, dtype=float), traj, dedup=False, outside_value=-1
    )
    return np.asarray(ids, dtype=int)


# ------------------------------------------------- multi-neuron, scoring & config gaps
@pytest.mark.slow
def test_multineuron_fit_recovers_distinct_per_neuron_fields(small_grid_env):
    """The vmap warm-start + M-step must keep neurons separate: each neuron recovers
    ITS own field, not a mix. A neuron-axis transposition would fail the cross guard."""
    basis = build_graph_basis(small_grid_env, rank=8)
    centers = np.asarray(small_grid_env.bin_centers)
    n_time = 2500
    rng = np.random.default_rng(7)
    idx = rng.integers(0, centers.shape[0], size=n_time)
    times = np.arange(n_time, dtype=float) * 0.02
    traj = centers[idx]
    Z = np.asarray(basis.eigvecs)[idx]

    def make_field(seed):
        r = np.random.default_rng(seed)
        w = np.zeros(8)
        w[1:5] = r.standard_normal(4)
        w[0] = np.log(25.0) / float(basis.eigvecs[0, 0])
        return w

    w0, w1 = make_field(1), make_field(2)
    spikes = np.stack(
        [rng.poisson(np.exp(Z @ w0) * 0.02), rng.poisson(np.exp(Z @ w1) * 0.02)],
        axis=1,
    ).astype(float)

    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=8,
        kappa2=1.0,
        tau2=1.0,
        update_drift_scale=False,
        update_amplitude=False,
    )
    model.fit(times, traj, spikes, max_iter=8, verbose=False)
    assert model.smoother_mean.shape[0] == 2

    occ = bin_occupancy(small_grid_env, times, traj, 0.02)
    visited = occ > 0
    true0 = np.exp(np.asarray(basis.eigvecs @ w0))
    true1 = np.exp(np.asarray(basis.eigvecs @ w1))
    r00 = np.corrcoef(model.predict_rate_map(0)[visited], true0[visited])[0, 1]
    r11 = np.corrcoef(model.predict_rate_map(1)[visited], true1[visited])[0, 1]
    r01 = np.corrcoef(model.predict_rate_map(0)[visited], true1[visited])[0, 1]
    assert r00 > 0.9 and r11 > 0.9  # each neuron recovers its own field
    assert r00 > r01 + 0.2  # and not the other's (guard against neuron-axis mixup)


@pytest.mark.slow
def test_score_uses_heldout_spikes_and_positions(small_grid_env):
    """score() must actually run the filter on its arguments: a genuinely held-out,
    correctly aligned spike train scores higher than a shifted (mis-aligned) one."""
    basis = build_graph_basis(small_grid_env, rank=8)
    times, traj, spikes, _eta, _w = _simulate_field_drift_spikes(
        small_grid_env, basis, dt=0.02, n_time=2400, q_c=0.0, seed=5
    )
    half = 1200
    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=8,
        kappa2=1.0,
        tau2=1.0,
        update_drift_scale=False,
        update_amplitude=False,
    )
    model.fit(times[:half], traj[:half], spikes[:half], max_iter=8, verbose=False)
    ll_aligned = model.score(times[half:], traj[half:], spikes[half:])
    # Rolling the held-out spikes by a large shift breaks the position<->rate
    # alignment; the fitted field must assign it a lower likelihood.
    ll_shifted = model.score(times[half:], traj[half:], np.roll(spikes[half:], 600))
    assert np.isfinite(ll_aligned)
    assert ll_aligned > ll_shifted


def test_drift_scale_mstep_matches_closed_form(small_grid_env):
    """The closed-form drift_scale M-step q* = mean_j(diag(E[dw dw^T])_j / S_j),
    pinned to a hand-computed value on a 1-mode, 2-step problem (mirrors the tau2 test).
    """
    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=1,
        kappa2=2.0,
        update_drift_scale=True,
        update_amplitude=False,
        update_init_mean=False,
    )
    model.init_mean = jnp.zeros((1, 1))
    # a=Var_0=1, b=Var_1=3, d=mean_1=2 (mean_0=0), c=cross-cov=0.5.
    model.smoother_mean = jnp.array([[[0.0], [2.0]]])  # (n_neurons=1, T=2, rank=1)
    model.smoother_cov = jnp.array([[[[1.0]], [[3.0]]]])  # (1, 2, 1, 1)
    model.smoother_cross_cov = jnp.array([[[[0.5]]]])  # (1, T-1=1, 1, 1)
    model.drift_scale = jnp.array([1e-3])
    shape = float(model._spectral_shape_current()[0])
    model._m_step()
    # E[dw dw^T] = a + b + d^2 - 2c = 1 + 3 + 4 - 1 = 7 ; q* = 7 / S.
    assert float(model.drift_scale[0]) == pytest.approx(7.0 / shape)


@pytest.mark.slow
def test_fit_sgd_improves_marginal_ll(small_grid_env):
    """fit_sgd must actually optimize: the marginal LL at the end is >= the start.
    Catches a sign error in the loss or a broken parameter-transform wiring."""
    times, traj = _toy_trajectory(small_grid_env, 250, seed=9)
    spikes = np.random.default_rng(10).poisson(0.1, size=250).astype(float)
    model = GraphPlaceFieldModel(small_grid_env, dt=0.02, rank=6)
    lls = model.fit_sgd(
        times, traj, spikes, num_steps=60, warm_start=True, verbose=False
    )
    assert np.all(np.isfinite(lls))
    assert (
        lls[-1] >= lls[0]
    )  # marginal LL improved (or held); returned LLs, higher=better


# ------------------------------------------------- regression guards for review fixes
def test_constructor_rejects_nan_max_firing_rate(small_grid_env):
    with pytest.raises(ValueError, match="max_firing_rate_hz"):
        GraphPlaceFieldModel(small_grid_env, dt=0.02, max_firing_rate_hz=float("nan"))


def test_fit_sgd_all_frozen_raises(small_grid_env):
    times, traj = _toy_trajectory(small_grid_env, 20, seed=1)
    spikes = np.zeros(20)
    model = GraphPlaceFieldModel(
        small_grid_env,
        dt=0.02,
        rank=3,
        update_kappa2=False,
        update_drift_scale=False,
        update_amplitude=False,
        update_init_mean=False,
    )
    with pytest.raises(ValueError, match="nothing to optimize"):
        model.fit_sgd(times, traj, spikes, num_steps=1, warm_start=False, verbose=False)


def test_update_kappa2_false_freezes_kappa2_under_sgd(small_grid_env):
    times, traj = _toy_trajectory(small_grid_env, 40, seed=2)
    spikes = np.random.default_rng(3).poisson(0.1, size=40).astype(float)
    model = GraphPlaceFieldModel(
        small_grid_env, dt=0.02, rank=4, kappa2=1.5, update_kappa2=False
    )
    model.fit_sgd(times, traj, spikes, num_steps=5, warm_start=True, verbose=False)
    assert model.kappa2 == 1.5  # frozen exactly


@pytest.mark.slow
def test_fit_preserves_learned_drift_scale_on_repeat(small_grid_env):
    times, traj = _toy_trajectory(small_grid_env, 60, seed=4)
    spikes = np.random.default_rng(5).poisson(0.1, size=60).astype(float)
    model = GraphPlaceFieldModel(
        small_grid_env, dt=0.02, rank=4, update_drift_scale=True, init_drift_scale=2e-3
    )
    model.fit(times, traj, spikes, max_iter=3, verbose=False)
    learned = np.asarray(model.drift_scale).copy()
    assert not np.allclose(learned, 2e-3)  # guard: EM actually moved it
    # A second fit(warm_start=False) must NOT reset drift_scale to init_drift_scale.
    model.fit(times, traj, spikes, max_iter=1, warm_start=False, verbose=False)
    assert not np.allclose(np.asarray(model.drift_scale), 2e-3)
