# Successor-Representation Spatial Basis Implementation Plan

**Status:** Not started.

**Integration note (2026-10-05):** The drifting graph-GP model is now implemented in this module. The shared `GraphBasis` prefix has seven fields, including `laplacian_convention`; the SR fingerprint uses `build_graph_laplacian(env)` and `_laplacian_key(laplacian, "distance")`. Task 4 should reuse the existing `fit_static_graph_glm`. Historical source line references and the standalone static-solver sketch below must be reconciled with these interfaces before execution.

**Goal:** Add a successor-representation (predictive-map) spatial basis to `graph_place_field` as a drop-in alternative to the graph-Laplacian eigenbasis, together with the held-out spike log-likelihood comparison at matched rank that decides whether to adopt it.

**Architecture:** Estimate the animal's bin-to-bin transition matrix `T` from the trajectory (adjacency-restricted counts, pseudo-count smoothing on graph edges, row-normalised; NumPy/SciPy like the graph substrate), form the successor representation `M = (I - gamma T)^-1` and use its leading left singular vectors — computed per connected component as right singular vectors of `I - gamma T`, so no inverse is ever formed — as a real orthonormal basis. Package it as `SuccessorBasis`, a `NamedTuple` whose leading seven fields are those of `GraphBasis`, so `graph_design_matrix`, `bin_spike_counts` and `spectral_shape` consume it unchanged and the per-time design-matrix contract `(Z (n_time, rank), valid (n_time,))` is untouched. Decide adoption with a static ridge-penalised Poisson fit on per-bin sufficient statistics and contiguous-block cross-validated held-out log-likelihood, sweeping `gamma`, against the Laplacian basis at the same rank and against a *symmetrised-SR* control arm (same movement graph, direction removed). The Laplacian path (`build_graph_basis`, `GraphBasis`) is not modified.

**Tech stack:** NumPy, SciPy (`linalg.svd`, `linalg.solve`, `linalg.subspace_angles`, `special.gammaln`, `sparse.csgraph.connected_components`), networkx (adjacency of `env.connectivity`), `neurospatial.Environment` (the `spatial` extra). JAX supplies the existing static estimator reused by Task 4 and the downstream Laplace-EKF smoke test on the SR design matrix.

**Out of scope:** modifying `PlaceFieldModel` (its basis is the patsy spline built inside `_build_spline_basis_matrix`; a graph-basis model is the separate drifting-graph-GP plan), a *drifting* SR model, learning the SR online (Fang et al. 2023; Bono et al. 2023), direction-split state spaces, sparse/iterative SVD, GPU.

## Design finding from the planning prototype — read before the Tasks

A NumPy prototype of exactly this design (ring of 48 bins from `Environment.from_samples` on a circle, 24 cells, 600 s at 50 Hz, 5-fold contiguous cross-validation, isotropic ridge `1e-3`, `gamma` selected on the grid `0.3 … 0.98`, three seeds) gave, in **nats per held-out spike** at rank 10:

| Scenario | directed SR − Laplacian | symmetrised SR − Laplacian | directed − symmetrised |
| --- | --- | --- | --- |
| Fields = SR columns of the directional policy (backward-skewed), directional trajectory | +0.0125 … +0.0147 | +0.0118 … +0.0143 | +0.0000 … +0.0008 |
| Gaussian bumps, symmetric (run-and-tumble) trajectory | +0.0038 … +0.0053 | +0.0039 … +0.0054 | −0.0001 … 0.0000 |
| Gaussian bumps, directional trajectory | +0.0035 … +0.0046 | +0.0049 … +0.0055 | −0.0013 … −0.0007 |
| Fields = SR columns of the symmetric policy, symmetric trajectory | +0.0185 … +0.0216 | +0.0193 … +0.0221 | −0.0009 … −0.0004 |

Rank 15 was the same picture; the directed SR beat the Laplacian on 5/5 folds in every run. Three consequences shape the plan:

1. **"SR beats Laplacian at matched rank" holds robustly — but for movement-graph adaptation, not direction.** An empirical walk lives on the bins the animal actually traverses (with their visitation weights); the distance-weighted Laplacian of the `from_samples` band graph spends modes on the band's transverse structure and irregular degrees. The symmetrised SR gets the same gain.
2. **The requested "tie with the Laplacian on symmetric fields with a symmetric policy" is false for empirical walks** (the SR wins by +0.003 … +0.005 there). The tie is exact only when `T` is a function of the Laplacian — the lazy symmetric walk `T = I - eps L` — which is the exact-property test (principal angles `< 1e-14` observed).
3. **Direction is invisible to an SVD basis on a loop.** On an ideal loop a directional `T` is circulant, so `M Mᵀ` is circulant and its eigenvectors are the Fourier modes whatever the direction — identical to the symmetrised basis at every rank. On the empirical band ring the two subspaces do differ (principal angles up to 0.7 rad at rank 10) yet their held-out LLs tie within 0.001 nats/spike **even on backward-skewed fields**. Direction-dependent skew (Mehta, Quirk & Wilson 2000) lives in the *coefficients*, not in the subspace; the SR still spans skewed fields well because its columns `M(:, j)` are the skewed fields (Stachenfeld, Botvinick & Gershman 2017), but so does the symmetrised SR at the same rank.

Therefore the validation is restated to what is actually testable: a positive control (directed SR beats the Laplacian by a margin, every fold), a tie control (directed vs symmetrised SR tie on Gaussian fields with a symmetric policy — an estimator-noise control), the exact property, and the backward-skew property asserted on `successor_representation` columns. The real-data script reports all three arms and adds a coefficient-level skew diagnostic, because the directional hypothesis must be tested on fitted fields, not by basis choice. See Open Questions 1 for the direction-sensitive alternative this rules in.

## Decisions (recorded defaults; what breaks them)

- **D1 — argument order.** `estimate_transition_matrix(env, times, trajectory, *, ...)`, matching `bin_occupancy` / `graph_design_matrix` (`graph_place_field.py:336-343`, `:387-392`). `times` is required, not optional, because `Environment.bin_sequence` requires monotone `times` (neurospatial `environment/trajectory.py:494`). Deviates from the requested `(env, positions, times?)`.
- **D2 — self-transitions do not count.** The chain is over consecutive *distinct* bins (`dedup=True` semantics), implemented by collapsing the module's own `_bin_ids` (`dedup=False`, `graph_place_field.py:313-333`) so a single bin-lookup path is shared with `graph_design_matrix` / `bin_occupancy` / `bin_spike_counts`. Justification: (i) `gamma` then has a sampling-rate-invariant meaning (horizon in bin hops), so one default grid serves 50 Hz and 250 Hz data; (ii) dwell time is already the Poisson exposure (`bin_occupancy`) and would otherwise enter the model twice; (iii) the unvisited-row fallback and the exact-property reference walk are diagonal-free, so estimator and references share one form. Upstream `Environment.transitions` (`trajectory.py:787`, `_empirical_transitions` at `:965`) always counts self-transitions from `dedup=False` ids and has no gap handling, smoothing or symmetrised variant — that is why it is not called. *Breaks if* the discrete-time SR with dwell (Stachenfeld's `gamma` per time step) is wanted; see Open Question 2.
- **D3 — non-adjacent moves are dropped** (tracking jumps, or bins smaller than one sample's displacement); when they exceed 5% of moves a `StateSpaceWarning` (`exceptions.py:14`) is emitted naming the likely cause. Out-of-bounds samples (`-1`) and time gaps `> max_gap` break the chain (no pair straddles them). Zero usable moves raises `ValueError`.
- **D4 — smoothing.** `counts + smoothing_alpha * A` with `A` the boolean adjacency of `env.connectivity` (pseudo-count on every graph edge, default `0.5`), then row-normalise. For `alpha > 0` an unvisited row is therefore automatically uniform over its neighbours; for `alpha = 0` that fallback is explicit. A degree-0 (isolated) bin is absorbing (`T_ii = 1`). Non-adjacent pairs stay exactly zero.
- **D5 — `symmetrize=True`** adds `counts + countsᵀ` before smoothing: the direction-free control arm required by the finding above.
- **D6 — basis = leading `rank` left singular vectors of `M`,** computed per connected component from the SVD of `B = I - gamma T_c`: `M_c = B^-1 = V Σ^-1 Uᵀ`, so the left singular vectors of `M_c` are the right singular vectors of `B` with singular values `1/σ_B`. No inverse and no solve for the basis. `successor_representation` (returns `M` itself, for fields and diagnostics) uses `scipy.linalg.solve(B, I)`. Why singular vectors rather than eigenvectors: for non-symmetric `T` the eigenvectors of `M` are complex and non-orthogonal; the SVD gives a real orthonormal basis whose rank-`k` span is the least-squares-optimal subspace for the columns `M(:, j)` — Stachenfeld's predicted place fields.
- **D7 — the `eigvals` field holds Laplacian-equivalent penalties** `λ̃_j = (σ_1/σ_j − 1)(1 − gamma)/(gamma·eps)` with `eps = 1 / max_i L_ii` and `σ_1` the global largest singular value; non-negative, ascending, zero for the leading mode. Under the lazy symmetric walk `T = I − eps L`, `σ_j = 1/((1−gamma) + gamma·eps·λ_j)`, so `λ̃_j = λ_j` exactly (prototype: `1e-16`). This keeps `spectral_shape(basis.eigvals, …)` (`graph_place_field.py:280-310`) meaningful for an SR basis and gives the exact-property test a second, scalar identity to assert.
- **D8 — `rank >= n_components` is enforced by reusing `_resolve_rank`** (`graph_place_field.py:150-181`; raises, never clamps); `gamma` must satisfy `0 < gamma < 1` (`gamma = 0` makes `M = I` and the SVD order arbitrary).
- **D9 — `SuccessorBasis` is a new `NamedTuple`; `GraphBasis` is untouched.** A module alias `SpatialBasis = GraphBasis | SuccessorBasis` widens the *annotations* of `_check_basis_matches_env`, `graph_design_matrix` and `bin_spike_counts` (runtime behaviour unchanged). `kind = "successor"` is a field of `SuccessorBasis`; consumers distinguish the two with `isinstance`. Rejected: adding `kind` with a default to `GraphBasis` (changes the class; the hard rule keeps it bit-identical).
- **D10 — no caching on `env` for SR bases** (`T` and `gamma` change per call). Dense per-component SVD is `O(n³)`: ~1 s at 1000 bins, ~10 s at 2500 bins. `select_gamma_by_held_out_ll` performs `n_folds × n_gammas` builds (35 by default); a sparse `svds` path is a later optimisation, not part of this plan.
- **D11 — the comparison model is a static, ridge-penalised Poisson GLM on per-bin sufficient statistics** (`counts (n_bins, n_neurons)` from `bin_spike_counts`, `occupancy (n_bins,)` from `bin_occupancy`), fitted by precision-space Newton with step halving. It is deterministic, has no EM/drift hyperparameters, and equals the per-time static Poisson fit up to constants (the per-bin counts and exposures are sufficient statistics). The ridge is isotropic (`1e-3`) and identical for both bases, so the comparison is about the *subspace*, not the prior. Held-out LL is the per-bin Poisson log-likelihood including `−log n_i!` (a constant across bases; kept so values are interpretable). *Breaks if* the prior is part of the question — then compare with the spectral penalty on `eigvals` (Open Question 4).
- **D12 — cross-validation = `n_folds` (default 5) contiguous equal blocks of rows.** Random splits leak through field drift. `T` is re-estimated from the training rows of each fold (conservative: the held-out block's behaviour never shapes the basis), with `max_gap = 2 · median(diff(times))` so the removed block does not create a spurious move.
- **D13 — `gamma` grid default `(0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)`;** the result carries `best_gamma_at_grid_edge` instead of warning (warnings are errors in tests, and the optimum sits at `0.98` routinely for smooth fields). The `gamma → 1` limit is well defined — the SVD of `I − T` — and Laplacian-like for symmetric walks, so a high optimum is not a defect.
- **D14 — the validation script lives in `scripts/`** in the plain `main()` style of `scripts/position_decoding_demo.py`, not as a jupytext notebook.

## Dependencies

- **Cross-plan.** `docs/plans/multi-map-place-fields/` consumes any spatial basis through the per-time design matrix. This plan keeps `graph_design_matrix`'s contract identical (`Z[t] = basis.eigvecs[bin_id(t)]`, zero row + `valid=False` out of bounds; `graph_place_field.py:344-384`); only the `basis` annotation widens to `SpatialBasis`. The [drifting graph-GP plan](../2026-07-16-drifting-graph-gp-place-field.md) has implemented `fit_static_graph_glm` with per-coefficient penalties and zero penalties allowed. Task 4 should adapt/reuse that estimator under the proposed API rather than add another static Poisson solver; its original solver sketch below predates this integration.
- **neurospatial.** Importable in this checkout's `.venv` as an *editable* install of `/Users/edeno/Documents/GitHub/neurospatial` at commit `2522021fe2e7`, while `uv.lock` / `pyproject.toml:167-171` pin the git commit `e81dec62…` (both report version 0.8.0). Every neurospatial API this plan uses is already used by the shipped module or its tests (`bin_sequence`, `connectivity`, `get_differential_operator`, `n_bins`, `bin_sizes`, `bin_centers`, `from_samples`, `make_w_maze`); tests use `env.connectivity.neighbors` (networkx) rather than `env.neighbors` to avoid a new upstream surface. The executor should confirm the install source with `uv run --no-sync python -c "import neurospatial, inspect; print(inspect.getsourcefile(neurospatial))"` and expect the ring fixture's bin count to be a property of the installed version (never hard-code `48`).
- **Real data.** `data/` (gitignored) is absent in this checkout; the script is written against the loader signature used by `scripts/position_decoding_demo.py:22-36` and cannot be run here.

## Inputs to read first

- [src/state_space_practice/graph_place_field.py:1-27](../../../src/state_space_practice/graph_place_field.py) — module docstring: active-bin convention and the distance-weighted Laplacian choice; extend it (Task 1).
- [graph_place_field.py:59-93](../../../src/state_space_practice/graph_place_field.py) — `GraphBasis` fields; `SuccessorBasis` mirrors the first seven in order.
- [graph_place_field.py:96-124](../../../src/state_space_practice/graph_place_field.py) — `build_graph_laplacian`, `_laplacian_key`, `_env_key`: reused for `env_key`, components and `eps`.
- [graph_place_field.py:127-147](../../../src/state_space_practice/graph_place_field.py) — `_check_basis_matches_env`: the environment check reads `eigvecs.shape[0]`, `laplacian_convention` and `env_key`; error strings at `:136-140` and `:143-147` name `build_graph_basis` (Task 6 generalises them).
- [graph_place_field.py:150-181](../../../src/state_space_practice/graph_place_field.py) — `_resolve_rank` (reuse with `sigma=None`).
- [graph_place_field.py:184-210](../../../src/state_space_practice/graph_place_field.py) — `_full_eigensystem`: the per-component decompose-pad-sort pattern to mirror.
- [graph_place_field.py:213-277](../../../src/state_space_practice/graph_place_field.py) — `build_graph_basis` (untouched): read-only arrays, `np.array(env.bin_sizes)` copy at `:266`.
- [graph_place_field.py:313-333](../../../src/state_space_practice/graph_place_field.py) — `_bin_ids` (`dedup=False`, `outside_value=-1` at `:330-332`); the transition estimator collapses these ids.
- [graph_place_field.py:336-384](../../../src/state_space_practice/graph_place_field.py) — `graph_design_matrix`: contract to preserve; annotation at `:338`, check at `:378`, row lookup `:381-383`.
- [graph_place_field.py:387-468](../../../src/state_space_practice/graph_place_field.py) — `bin_occupancy`, `bin_spike_counts` (annotation `:428`, check `:454`): the per-bin sufficient statistics the comparison consumes.
- [src/state_space_practice/tests/test_graph_place_field.py:30-50](../../../src/state_space_practice/tests/test_graph_place_field.py) — fixtures `small_grid_env`, `w_maze_env`, `two_component_env`; `:54-64` how tests obtain `L = D @ D.T`; `:103-110` component-locality pattern; `:114-126` design-row alignment pattern; `:245-254` mismatched-env pattern.
- [src/state_space_practice/exceptions.py:14](../../../src/state_space_practice/exceptions.py) — `StateSpaceWarning`.
- [src/state_space_practice/point_process_kalman.py:1467-1484](../../../src/state_space_practice/point_process_kalman.py) — `stochastic_point_process_filter` signature (returns `(filtered_mean, filtered_cov, marginal_ll)`); `:523` `log_conditional_intensity`. Used by the downstream smoke test only.
- [src/state_space_practice/place_field_model.py:1886-1980](../../../src/state_space_practice/place_field_model.py) — `score`: held-out LL for the spline model is the filter's marginal LL on new data with `evaluate_basis(position, basis_info)` (`:1957-1959`); it is bound to the patsy spline (`_build_spline_basis_matrix` `:567-592`), which is why this plan evaluates held-out LL with its own static fit rather than through `PlaceFieldModel`. `:652-659` `_fit_stationary_glm` is the private per-time Newton GLM whose *shape* the per-bin fit mirrors.
- [src/state_space_practice/tests/conftest.py:12](../../../src/state_space_practice/tests/conftest.py) — x64 on for tests; `:143-149` slow auto-marking keys on `.fit(`/`run_em(` calls, which the new tests do not make — mark slow tests explicitly.
- [pyproject.toml:61](../../../pyproject.toml) — `spatial` extra; `:117` ruff excludes `scripts/**` from lint (still formatted); `:134-141` mypy `files` list (insert the module after `exceptions.py`); `:167-171` neurospatial source pin.
- [CHANGELOG.md:6-8](../../../CHANGELOG.md) — `## [Unreleased]` / `### Added` where the entry goes. [README.md:27](../../../README.md) — the `spatial` extra row; `:37-39` the `data/` loader note the script relies on.
- [scripts/position_decoding_demo.py:1-27, 30-73](../../../scripts/position_decoding_demo.py) — script conventions: x64 before imports, `PROJECT_ROOT`, `from data.load_bandit_data import load_neural_recording_from_files`, `select_units` / `bin_spike_times` / `interpolate_to_new_times` usage. [src/state_space_practice/preprocessing.py:56](../../../src/state_space_practice/preprocessing.py) `bin_spike_times` returns `(n_time, n_neurons)`; `:210` `select_units`; `:334` `interpolate_to_new_times`.
- neurospatial (installed source): `environment/trajectory.py:494` `bin_sequence(times, positions, *, dedup=True, outside_value=-1) -> int32 (n,)`; `:787` `transitions` / `:965` `_empirical_transitions` (semantics rejected in D2); `environment/core.py:1233` `get_differential_operator`; `environment/factories.py:255` `from_samples(positions, bin_size, *, bin_count_threshold=0, connect_diagonal_neighbors=True, …)`. `env.connectivity` is an undirected `nx.Graph` on nodes `0..n_bins-1` with edge attribute `distance` and no self-loops (verified by instantiation).

## Tasks

All library code goes in `src/state_space_practice/graph_place_field.py`; tests in `src/state_space_practice/tests/test_graph_place_field.py`. Suggested commit boundaries: (1) Tasks 1–3 + their tests, (2) Tasks 4–5 + tests, (3) Task 6 + tests, (4) Tasks 7–9.

### Task 1 — `SuccessorBasis`, `SpatialBasis`, module docstring, exports

Add after `GraphBasis` (`graph_place_field.py:93`):

```python
class SuccessorBasis(NamedTuple):
    """Truncated successor-representation (predictive-map) basis over active bins.

    The leading seven fields are those of :class:`GraphBasis`, in the same order, so
    every consumer of a ``GraphBasis`` (:func:`graph_design_matrix`,
    :func:`bin_spike_counts`, :func:`spectral_shape`) accepts it unchanged.

    Attributes
    ----------
    eigvecs : NDArray, shape (n_bins, rank)
        Leading left singular vectors of the successor representation
        ``M = (I - gamma T)^-1`` as columns, orthonormal, ordered by descending
        singular value. Row ``i`` is active bin ``i``. Component-local on a
        disconnected graph (zero outside the column's connected component).
    eigvals : NDArray, shape (rank,)
        Laplacian-equivalent penalties ``(sigma_1 / sigma_j - 1) (1 - gamma) /
        (gamma eps)`` with ``eps = 1 / max_i L_ii``: non-negative, ascending, ``0``
        for the leading mode. When ``T = I - eps L`` (the lazy symmetric walk) these
        equal the Laplacian eigenvalues exactly, so :func:`spectral_shape` keeps its
        meaning for a successor basis.
    component_labels : NDArray, shape (n_bins,)
        Connected-component id per active bin (from the environment's graph).
    bin_sizes : NDArray, shape (n_bins,)
        Per-bin volume (``env.bin_sizes``). **Not** exposure.
    n_components : int
        Number of connected components (``rank >= n_components`` is enforced).
    laplacian_convention : LaplacianConvention
        ``"distance"`` for the environment fingerprint; the SR basis itself is
        defined by the empirical transition matrix.
    env_key : tuple
        The environment's Laplacian fingerprint (as in :class:`GraphBasis`), so the
        consumers' environment check applies unchanged.
    kind : str
        ``"successor"``.
    gamma : float
        Discount used to build ``M``.
    singular_values : NDArray, shape (rank,)
        Singular values ``sigma_j`` of ``M`` (descending).
    """

    eigvecs: NDArray[np.float64]
    eigvals: NDArray[np.float64]
    component_labels: NDArray[np.int_]
    bin_sizes: NDArray[np.float64]
    n_components: int
    laplacian_convention: LaplacianConvention
    env_key: tuple[str, int, int, str]
    kind: str
    gamma: float
    singular_values: NDArray[np.float64]


SpatialBasis = GraphBasis | SuccessorBasis
```

- Extend `__all__` (`:44-51`) with `"SuccessorBasis"`, `"SpatialBasis"`, `"estimate_transition_matrix"`, `"successor_representation"`, `"build_successor_basis"`, `"fit_penalized_poisson_field"`, `"poisson_field_log_likelihood"`, `"GammaSelection"`, `"select_gamma_by_held_out_ll"`.
- Module docstring (`:1-27`): retitle to a spatial-basis substrate offering two bases; add a short "Successor-representation basis" paragraph stating D2 (movement chain, no self-transitions), D6 (singular vectors of `M`, why not eigenvectors) and the finding that the SVD basis cannot express travel direction (direction is a coefficient property), citing Stachenfeld, Botvinick & Gershman (2017, *Nat. Neurosci.*, place fields as SR columns), Dayan (1993, *Neural Comput.*, the SR), Mehta, Quirk & Wilson (2000, *Neuron*, backward field asymmetry) and Mahadevan & Maggioni (2007, *JMLR*, the Laplacian-eigenbasis lineage). Keep the existing "Laplacian choice" section intact.
- New imports: `import warnings`, `from scipy.special import gammaln`, `from state_space_practice.exceptions import StateSpaceWarning`.

### Task 2 — `estimate_transition_matrix`

```python
def _adjacency(env: Environment) -> NDArray[np.bool_]:
    """Dense boolean adjacency of ``env.connectivity`` in active-bin order."""
    adjacency = nx.to_scipy_sparse_array(env.connectivity, nodelist=range(env.n_bins))
    return np.asarray(adjacency.toarray() > 0)


def estimate_transition_matrix(
    env: Environment,
    times: NDArray[np.float64],
    trajectory: NDArray[np.float64],
    *,
    smoothing_alpha: float = 0.5,
    symmetrize: bool = False,
    max_gap: float | None = None,
) -> NDArray[np.float64]:
    """Row-stochastic bin-to-bin transition matrix of the animal's movement.

    Counts moves between consecutive *distinct* active bins (dwell samples within
    a bin add nothing, so the diagonal is zero and ``gamma`` in
    :func:`build_successor_basis` is a horizon in bin hops, independent of the
    sampling rate). Only graph-adjacent moves count; non-adjacent moves (tracking
    jumps, or bins smaller than one sample's displacement) are dropped and a
    :class:`StateSpaceWarning` is emitted when they exceed 5% of moves.
    Out-of-bounds samples and time gaps longer than ``max_gap`` break the chain.
    Every graph edge receives ``smoothing_alpha`` pseudo-counts, so unvisited-but-
    adjacent pairs get small mass, non-adjacent pairs stay exactly zero, and a bin
    that was never left is uniform over its neighbours (an isolated bin is absorbing).

    Parameters
    ----------
    env : neurospatial.Environment
    times : NDArray, shape (n_time,)
        Monotone sample times (seconds).
    trajectory : NDArray, shape (n_time,) or (n_time, n_dims)
    smoothing_alpha : float, optional
        Pseudo-count added to every adjacent pair before row normalisation, by
        default 0.5. Must be non-negative.
    symmetrize : bool, optional
        If True, use ``counts + counts.T`` (the direction-free control), by default
        False.
    max_gap : float or None, optional
        Consecutive samples further apart than this (seconds) are not a move. None
        (default) never breaks the chain on time alone.

    Returns
    -------
    NDArray, shape (n_bins, n_bins)
        ``T[i, j] = P(next distinct bin = j | current bin = i)``; rows sum to 1.
    """
    if smoothing_alpha < 0:
        raise ValueError(f"smoothing_alpha must be non-negative, got {smoothing_alpha}.")
    if max_gap is not None and not max_gap > 0:
        raise ValueError(f"max_gap must be positive, got {max_gap}.")
    times = np.asarray(times, dtype=float)
    ids = _bin_ids(env, times, trajectory)
    if max_gap is not None:
        # A sentinel between samples separated by a gap breaks the chain there.
        gap_after = np.diff(times) > max_gap
        ids = np.insert(ids, np.flatnonzero(gap_after) + 1, -1)
    change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    runs = ids[np.r_[0, change]]  # one entry per visited bin, in order
    source, target = runs[:-1], runs[1:]
    keep = (source >= 0) & (target >= 0)
    source, target = source[keep], target[keep]
    if source.size == 0:
        raise ValueError(
            "no bin-to-bin moves in the trajectory (all samples in one bin, out of "
            "bounds, or separated by gaps); cannot estimate a transition matrix."
        )
    adjacency = _adjacency(env)
    adjacent = adjacency[source, target]
    fraction_jumps = 1.0 - float(adjacent.mean())
    if fraction_jumps > 0.05:
        warnings.warn(
            f"{fraction_jumps:.1%} of bin-to-bin moves were between non-adjacent bins "
            "and were dropped; check for tracking jumps, or bins smaller than the "
            "displacement between consecutive samples.",
            StateSpaceWarning,
            stacklevel=2,
        )
    n_bins = env.n_bins
    counts = np.zeros((n_bins, n_bins))
    np.add.at(counts, (source[adjacent], target[adjacent]), 1.0)
    if symmetrize:
        counts = counts + counts.T
    smoothed = counts + smoothing_alpha * adjacency
    row_sums = smoothed.sum(axis=1)
    transition = np.zeros_like(smoothed)
    visited = row_sums > 0
    transition[visited] = smoothed[visited] / row_sums[visited, None]
    degree = adjacency.sum(axis=1)
    fallback = ~visited & (degree > 0)
    transition[fallback] = adjacency[fallback] / degree[fallback, None]
    isolated = np.flatnonzero(~visited & (degree == 0))
    transition[isolated, isolated] = 1.0
    return transition
```

### Task 3 — `successor_representation` and `build_successor_basis`

```python
def _validate_gamma(gamma: float) -> float:
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie strictly in (0, 1), got {gamma}.")
    return float(gamma)


def _validate_transition(
    transition: NDArray[np.float64], n_bins: int, labels: NDArray[np.int_]
) -> NDArray[np.float64]:
    """A finite, row-stochastic (n_bins, n_bins) matrix with no mass between components."""
    T = np.asarray(transition, dtype=float)
    if T.shape != (n_bins, n_bins):
        raise ValueError(f"transition must have shape {(n_bins, n_bins)}, got {T.shape}.")
    if not np.all(np.isfinite(T)):
        raise ValueError("transition contains non-finite entries.")
    if T.min() < -1e-12:
        raise ValueError("transition has negative entries.")
    if not np.allclose(T.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("transition rows must sum to 1 (row-stochastic).")
    across = labels[:, None] != labels[None, :]
    if across.any() and np.abs(T[across]).max() > 1e-12:
        raise ValueError(
            "transition carries mass between disconnected components of the "
            "environment graph; the successor basis would not be component-local."
        )
    return np.clip(T, 0.0, None)


def successor_representation(
    transition: NDArray[np.float64], gamma: float
) -> NDArray[np.float64]:
    """Successor representation ``M = (I - gamma T)^-1`` (Dayan 1993).

    ``M[s, j]`` is the expected discounted number of future visits to bin ``j``
    from bin ``s``; column ``j`` is the place field Stachenfeld et al. (2017) predict
    for a cell anchored at ``j``. Computed by an LU solve against the identity.

    Parameters
    ----------
    transition : NDArray, shape (n_bins, n_bins)
        Row-stochastic transition matrix.
    gamma : float
        Discount in ``(0, 1)``.

    Returns
    -------
    NDArray, shape (n_bins, n_bins)
    """
    gamma = _validate_gamma(gamma)
    T = np.asarray(transition, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"transition must be square, got shape {T.shape}.")
    n = T.shape[0]
    return np.asarray(scipy.linalg.solve(np.eye(n) - gamma * T, np.eye(n)))


def build_successor_basis(
    env: Environment,
    transition: NDArray[np.float64],
    *,
    gamma: float,
    rank: int | None = None,
) -> SuccessorBasis:
    """Leading singular subspace of the successor representation as a spatial basis.

    For each connected component ``c`` of the environment graph, the left singular
    vectors of ``M_c = (I - gamma T_c)^-1`` are the right singular vectors of
    ``B_c = I - gamma T_c`` (``M_c = B_c^-1 = V S^-1 U^T``) with singular values
    ``1 / s``; no inverse is formed. Modes are sorted by descending singular value
    across components and the leading ``rank`` kept. Singular vectors (not
    eigenvectors) because ``T`` is not symmetric in general: eigenvectors of ``M``
    are then complex and non-orthogonal, whereas the rank-``k`` singular subspace
    is real, orthonormal and the best least-squares approximation to the columns
    ``M[:, j]`` (the predicted fields). Note that on a translation-invariant graph
    (a loop) a directional ``T`` is circulant, so this subspace is identical to the
    symmetrised walk's at every rank: travel direction shows up in the fitted
    coefficients, not in the basis.

    Parameters
    ----------
    env : neurospatial.Environment
        The fitted environment ``transition`` was estimated on.
    transition : NDArray, shape (n_bins, n_bins)
        Row-stochastic, e.g. from :func:`estimate_transition_matrix`. Must carry no
        mass between disconnected components.
    gamma : float
        Discount in ``(0, 1)``; the horizon is ``1 / (1 - gamma)`` bin hops when the
        transition matrix is the movement chain of :func:`estimate_transition_matrix`.
    rank : int or None, optional
        Number of modes to keep (``None`` keeps all). Below ``n_components`` raises.

    Returns
    -------
    SuccessorBasis
        Read-only arrays; ``env_key`` is the environment's Laplacian fingerprint, so
        :func:`graph_design_matrix` accepts the basis exactly like a ``GraphBasis``.
    """
    gamma = _validate_gamma(gamma)
    laplacian = build_graph_laplacian(env)
    n_components, labels = connected_components(laplacian, directed=False)
    labels = np.asarray(labels)
    n_bins = int(laplacian.shape[0])
    T = _validate_transition(transition, n_bins, labels)
    eps = 1.0 / float(laplacian.diagonal().max())

    value_parts: list[NDArray[np.float64]] = []
    vector_parts: list[NDArray[np.float64]] = []
    for component in range(int(n_components)):
        idx = np.flatnonzero(labels == component)
        B = np.eye(idx.size) - gamma * T[np.ix_(idx, idx)]
        _, s, vt = np.linalg.svd(B)
        padded = np.zeros((n_bins, idx.size))
        padded[idx] = vt.T
        value_parts.append(1.0 / s)
        vector_parts.append(padded)
    singular_values = np.concatenate(value_parts)
    vectors = np.concatenate(vector_parts, axis=1)
    order = np.argsort(-singular_values, kind="stable")
    singular_values, vectors = singular_values[order], vectors[:, order]
    eigvals = (singular_values[0] / singular_values - 1.0) * (1.0 - gamma) / (gamma * eps)

    keep = _resolve_rank(eigvals, int(n_components), rank, None, 1e-6)
    basis = SuccessorBasis(
        eigvecs=vectors[:, :keep].copy(),
        eigvals=eigvals[:keep].copy(),
        component_labels=labels,
        bin_sizes=np.array(env.bin_sizes, dtype=float),
        n_components=int(n_components),
        laplacian_convention="distance",
        env_key=_laplacian_key(laplacian, "distance"),
        kind="successor",
        gamma=gamma,
        singular_values=singular_values[:keep].copy(),
    )
    for arr in (
        basis.eigvecs,
        basis.eigvals,
        basis.component_labels,
        basis.bin_sizes,
        basis.singular_values,
    ):
        arr.setflags(write=False)
    return basis
```

### Task 4 — `fit_penalized_poisson_field` and `poisson_field_log_likelihood`

```python
def _validate_field_inputs(
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    basis_matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    occupancy = np.asarray(occupancy, dtype=float)
    Phi = np.asarray(basis_matrix, dtype=float)
    n_bins = Phi.shape[0]
    if counts.shape[0] != n_bins or occupancy.shape != (n_bins,):
        raise ValueError(
            f"counts ({counts.shape}), occupancy ({occupancy.shape}) and basis rows "
            f"({n_bins}) must share the active-bin axis."
        )
    if np.any(occupancy < 0) or np.any(counts < 0):
        raise ValueError("counts and occupancy must be non-negative.")
    if np.any((counts > 0).any(axis=1) & (occupancy == 0)):
        raise ValueError(
            "a bin with spikes has zero occupancy; counts and occupancy must come from "
            "the same binning (bin_spike_counts / bin_occupancy on the same rows)."
        )
    return counts, occupancy, Phi


def fit_penalized_poisson_field(
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    basis_matrix: NDArray[np.float64],
    *,
    penalty: float | NDArray[np.float64],
    max_iter: int = 50,
    tol: float = 1e-9,
) -> NDArray[np.float64]:
    """Static penalised-Poisson place fields in a spatial basis, one fit per neuron.

    Maximises ``sum_i [n_i eta_i - e_i exp(eta_i)] - 0.5 * sum_j pen_j w_j^2`` with
    ``eta = Phi @ w`` (log-rate in spikes per second), ``n`` per-bin counts and ``e``
    per-bin exposure (seconds), by Newton's method in precision space with step
    halving. ``penalty`` is a diagonal precision: a float (isotropic ridge) or a
    ``(rank,)`` vector; zeros are allowed (unpenalised modes) as long as the visited
    bins support them.

    Parameters
    ----------
    counts : NDArray, shape (n_bins, n_neurons) or (n_bins,)
        Per-bin spike counts (:func:`bin_spike_counts`).
    occupancy : NDArray, shape (n_bins,)
        Per-bin exposure in seconds (:func:`bin_occupancy`).
    basis_matrix : NDArray, shape (n_bins, rank)
        ``basis.eigvecs`` of a :class:`GraphBasis` or :class:`SuccessorBasis`.
    penalty : float or NDArray, shape (rank,)
        Diagonal precision (non-negative).
    max_iter : int, optional
        Newton iterations per neuron, by default 50.
    tol : float, optional
        Relative objective-change tolerance, by default 1e-9.

    Returns
    -------
    NDArray, shape (rank, n_neurons)
        Fitted coefficients; ``exp(basis_matrix @ coeffs)`` is the rate map (Hz).
    """
    counts, occupancy, Phi = _validate_field_inputs(counts, occupancy, basis_matrix)
    rank = Phi.shape[1]
    pen = np.broadcast_to(np.asarray(penalty, dtype=float), (rank,)).copy()
    if np.any(pen < 0):
        raise ValueError("penalty must be non-negative.")
    coeffs = np.zeros((rank, counts.shape[1]))
    for neuron in range(counts.shape[1]):
        y = counts[:, neuron]
        w = np.zeros(rank)

        def objective(w: NDArray[np.float64]) -> float:
            with np.errstate(over="ignore", invalid="ignore"):
                eta = Phi @ w
                value = np.sum(y * eta - occupancy * np.exp(eta)) - 0.5 * np.sum(pen * w * w)
            return float(value) if np.isfinite(value) else -np.inf

        f = objective(w)
        for _ in range(max_iter):
            mu = occupancy * np.exp(Phi @ w)
            gradient = Phi.T @ (y - mu) - pen * w
            hessian = Phi.T @ (Phi * mu[:, None]) + np.diag(pen)
            try:
                # np.linalg.solve: SciPy's solve(assume_a="pos") emits LinAlgWarning on
                # ill-conditioning, which the test suite turns into an error.
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError as err:
                raise ValueError(
                    "singular Newton system: an unpenalised basis direction is not "
                    "supported by the visited bins; add a small penalty."
                ) from err
            t = 1.0
            f_new = objective(w + t * step)
            while f_new < f and t > 1e-8:
                t *= 0.5
                f_new = objective(w + t * step)
            if f_new < f:
                break  # no ascent along the Newton direction: converged
            w, previous, f = w + t * step, f, f_new
            if abs(f - previous) <= tol * (1.0 + abs(f)):
                break
        coeffs[:, neuron] = w
    return coeffs


def poisson_field_log_likelihood(
    counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    basis_matrix: NDArray[np.float64],
    coeffs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-neuron Poisson log-likelihood of per-bin counts under fitted fields.

    ``sum_i [n_i log(e_i lambda_i) - e_i lambda_i - log n_i!]`` with
    ``lambda = exp(Phi @ coeffs)``. Bins with zero exposure and zero count
    contribute 0; a positive count in a zero-exposure bin raises. Use it on
    *held-out* counts/occupancy (same environment and basis) to compare bases.

    Parameters
    ----------
    counts : NDArray, shape (n_bins, n_neurons) or (n_bins,)
    occupancy : NDArray, shape (n_bins,)
    basis_matrix : NDArray, shape (n_bins, rank)
    coeffs : NDArray, shape (rank, n_neurons)

    Returns
    -------
    NDArray, shape (n_neurons,)
    """
    counts, occupancy, Phi = _validate_field_inputs(counts, occupancy, basis_matrix)
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    if coeffs.shape != (Phi.shape[1], counts.shape[1]):
        raise ValueError(
            f"coeffs must have shape {(Phi.shape[1], counts.shape[1])}, got {coeffs.shape}."
        )
    expected = occupancy[:, None] * np.exp(Phi @ coeffs)
    log_term = np.zeros_like(expected)
    spiking = counts > 0
    log_term[spiking] = counts[spiking] * np.log(expected[spiking])
    return np.asarray(np.sum(log_term - expected - gammaln(counts + 1.0), axis=0))
```

### Task 5 — `GammaSelection` and `select_gamma_by_held_out_ll`

```python
DEFAULT_GAMMAS: tuple[float, ...] = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)


class GammaSelection(NamedTuple):
    """Cross-validated held-out log-likelihoods of a successor-basis gamma sweep.

    Attributes
    ----------
    gammas : NDArray, shape (n_gammas,)
    held_out_ll : NDArray, shape (n_gammas, n_folds)
        Summed over neurons, per held-out block, successor basis at each gamma.
    laplacian_ll : NDArray, shape (n_folds,)
        The graph-Laplacian basis at the same rank on the same folds (baseline).
    n_held_out_spikes : NDArray, shape (n_folds,)
        For reporting margins in nats per held-out spike.
    best_gamma : float
        ``argmax`` of ``held_out_ll.sum(axis=1)``.
    best_gamma_at_grid_edge : bool
        True when the best gamma is the first or last grid value (extend the grid).
    """

    gammas: NDArray[np.float64]
    held_out_ll: NDArray[np.float64]
    laplacian_ll: NDArray[np.float64]
    n_held_out_spikes: NDArray[np.float64]
    best_gamma: float
    best_gamma_at_grid_edge: bool


def _per_bin_statistics(
    env: Environment,
    times: NDArray[np.float64],
    trajectory: NDArray[np.float64],
    spikes: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-bin (counts, occupancy) from one bin lookup; out-of-bounds rows excluded."""
    ids = _bin_ids(env, times, trajectory)
    valid = ids >= 0
    occupancy = np.bincount(ids[valid], minlength=env.n_bins).astype(float) * dt
    counts = np.zeros((env.n_bins, spikes.shape[1]))
    np.add.at(counts, ids[valid], spikes[valid])
    return counts, occupancy


def select_gamma_by_held_out_ll(
    env: Environment,
    times: NDArray[np.float64],
    trajectory: NDArray[np.float64],
    spikes: NDArray[np.float64],
    *,
    dt: float,
    rank: int,
    gammas: Sequence[float] = DEFAULT_GAMMAS,
    n_folds: int = 5,
    smoothing_alpha: float = 0.5,
    symmetrize: bool = False,
    ridge: float = 1e-3,
) -> GammaSelection:
    """Sweep gamma by contiguous-block cross-validated held-out spike log-likelihood.

    Rows are split into ``n_folds`` contiguous equal blocks (random splits leak
    through field drift). For each block: the transition matrix is estimated from
    the *training* rows only (chain broken at the removed block), the successor basis
    at each gamma and the Laplacian basis at the same ``rank`` are fitted with
    :func:`fit_penalized_poisson_field` (isotropic ``ridge``, identical for every
    basis so the comparison is about the subspace), and the held-out block's per-bin
    Poisson log-likelihood is recorded.

    Parameters
    ----------
    env : neurospatial.Environment
    times : NDArray, shape (n_time,)
    trajectory : NDArray, shape (n_time,) or (n_time, n_dims)
    spikes : NDArray, shape (n_time, n_neurons)
        Per-time spike counts.
    dt : float
        Sampling interval (seconds).
    rank : int
        Basis size for both the successor and the Laplacian basis.
    gammas : sequence of float, optional
        Discounts in ``(0, 1)`` to try.
    n_folds : int, optional
        Contiguous blocks, by default 5 (at least 2).
    smoothing_alpha, symmetrize : see :func:`estimate_transition_matrix`.
    ridge : float, optional
        Isotropic penalty for every basis, by default 1e-3.

    Returns
    -------
    GammaSelection
    """
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}.")
    gammas_arr = np.asarray([_validate_gamma(g) for g in gammas], dtype=float)
    times = np.asarray(times, dtype=float)
    trajectory = np.asarray(trajectory, dtype=float)
    spikes = np.asarray(spikes, dtype=float)
    if spikes.ndim == 1:
        spikes = spikes[:, None]
    if spikes.shape[0] != times.shape[0] or trajectory.shape[0] != times.shape[0]:
        raise ValueError("times, trajectory and spikes must have the same number of rows.")
    n_time = times.shape[0]
    edges = np.linspace(0, n_time, n_folds + 1).astype(int)
    max_gap = 2.0 * float(np.median(np.diff(times)))
    laplacian = build_graph_basis(env, rank=rank)

    held_out = np.empty((gammas_arr.size, n_folds))
    laplacian_ll = np.empty(n_folds)
    n_spikes = np.empty(n_folds)
    for fold in range(n_folds):
        test = np.zeros(n_time, dtype=bool)
        test[edges[fold] : edges[fold + 1]] = True
        train = ~test
        counts_tr, occ_tr = _per_bin_statistics(env, times[train], trajectory[train], spikes[train], dt)
        counts_te, occ_te = _per_bin_statistics(env, times[test], trajectory[test], spikes[test], dt)
        n_spikes[fold] = counts_te.sum()
        coeffs = fit_penalized_poisson_field(counts_tr, occ_tr, laplacian.eigvecs, penalty=ridge)
        laplacian_ll[fold] = poisson_field_log_likelihood(
            counts_te, occ_te, laplacian.eigvecs, coeffs
        ).sum()
        transition = estimate_transition_matrix(
            env,
            times[train],
            trajectory[train],
            smoothing_alpha=smoothing_alpha,
            symmetrize=symmetrize,
            max_gap=max_gap,
        )
        for g_index, gamma in enumerate(gammas_arr):
            basis = build_successor_basis(env, transition, gamma=float(gamma), rank=rank)
            coeffs = fit_penalized_poisson_field(counts_tr, occ_tr, basis.eigvecs, penalty=ridge)
            held_out[g_index, fold] = poisson_field_log_likelihood(
                counts_te, occ_te, basis.eigvecs, coeffs
            ).sum()
    best = int(np.argmax(held_out.sum(axis=1)))
    return GammaSelection(
        gammas=gammas_arr,
        held_out_ll=held_out,
        laplacian_ll=laplacian_ll,
        n_held_out_spikes=n_spikes,
        best_gamma=float(gammas_arr[best]),
        best_gamma_at_grid_edge=best in (0, gammas_arr.size - 1),
    )
```

(`Sequence` comes from `collections.abc`.) Long lines above are for readability of the plan; let `ruff format` wrap them.

### Task 6 — widen the consumers' annotations (runtime unchanged)

- `_check_basis_matches_env(env, basis: SpatialBasis)` (`graph_place_field.py:127`), `graph_design_matrix(..., basis: SpatialBasis, ...)` (`:338`) and `bin_spike_counts(..., basis: SpatialBasis)` (`:428`). Update the two docstring `basis : GraphBasis` lines (`:355`, `:444-447`) to `GraphBasis or SuccessorBasis`.
- Generalise the two error strings at `:136-140` and `:143-147` from "Rebuild it with build_graph_basis(env)." to "Rebuild it from this environment (build_graph_basis or build_successor_basis)." — the existing tests match only `"different"` (`test_graph_place_field.py:251, 253`).
- No behavioural change to `build_graph_basis`; `test_basis_cache_rank_safe` and `test_read_only_outputs` must stay green untouched.

### Task 7 — tests (`tests/test_graph_place_field.py`)

Add a module-scoped `ring_env` fixture and helpers next to the existing fixtures (`:29-49`):

```python
RING_RADIUS = 30.0
RING_DT = 0.02


@pytest.fixture(scope="module")
def ring_env():
    """A closed loop of bins (annular band from points on a circle; one component)."""
    theta = np.linspace(0.0, 2.0 * np.pi, 4000, endpoint=False)
    xy = np.c_[RING_RADIUS * np.cos(theta), RING_RADIUS * np.sin(theta)]
    return Environment.from_samples(xy, bin_size=5.0)


def _ring_trajectory(duration, *, directional, rng, speed=20.0):
    """Counter-clockwise run (directional) or constant-speed run-and-tumble (symmetric)."""
    n_time = int(duration / RING_DT)
    if directional:
        dtheta = (speed * RING_DT + RING_DT * rng.standard_normal(n_time)) / RING_RADIUS
    else:
        sign = np.cumprod(np.where(rng.random(n_time) < 0.02, -1.0, 1.0))
        dtheta = sign * speed * RING_DT / RING_RADIUS
    theta = np.cumsum(dtheta)
    times = np.arange(n_time) * RING_DT
    xy = np.c_[RING_RADIUS * np.cos(theta), RING_RADIUS * np.sin(theta)]
    return times, xy + 0.3 * rng.standard_normal((n_time, 2))


def _signed_angular_offset(env, reference_bin):
    """Angle of every bin centre relative to ``reference_bin`` in (-pi, pi]; > 0 is ahead."""
    centers = np.asarray(env.bin_centers)
    angle = np.arctan2(centers[:, 1], centers[:, 0])
    return (angle - angle[reference_bin] + np.pi) % (2.0 * np.pi) - np.pi


def _poisson_spikes(env, times, xy, rate_map_hz, rng):
    ids = np.asarray(env.bin_sequence(times, xy, dedup=False, outside_value=-1))
    rates = np.where(ids[:, None] >= 0, rate_map_hz[np.maximum(ids, 0)], 0.0)
    return rng.poisson(rates * RING_DT)


def _successor_rate_maps(transition, gamma, goal_bins, peak_hz=20.0, floor_hz=0.5):
    """Stachenfeld-style fields: column ``j`` of M, rescaled to [floor, peak] Hz."""
    columns = successor_representation(transition, gamma)[:, goal_bins]
    return floor_hz + (peak_hz - floor_hz) * columns / columns.max(axis=0, keepdims=True)


def _lazy_symmetric_walk(env):
    """T = I - eps L with eps = 1 / max L_ii: symmetric, row-stochastic, a function of L."""
    D = env.get_differential_operator()
    L = (D @ D.T).toarray()
    return np.eye(env.n_bins) - L / L.diagonal().max()


def _trajectory_through_bins(env, bin_path, samples_per_bin=3):
    centers = np.asarray(env.bin_centers)
    xy = np.repeat(centers[list(bin_path)], samples_per_bin, axis=0)
    return np.arange(xy.shape[0]) * RING_DT, xy
```

The exact-property test (fast; `small_grid_env` has 25 bins, `w_maze_env` a few hundred):

```python
@pytest.mark.parametrize("env_name", ["ring_env", "small_grid_env", "w_maze_env"])
def test_successor_basis_matches_laplacian_for_lazy_symmetric_walk(request, env_name):
    env = request.getfixturevalue(env_name)
    T = _lazy_symmetric_walk(env)
    laplacian = build_graph_basis(env)
    gaps = np.diff(laplacian.eigvals)
    ranks = [k for k in (3, 5, 8) if gaps[k - 1] > 1e-6]  # subspace well defined
    assert len(ranks) >= 2
    for k in ranks:
        sr = build_successor_basis(env, T, gamma=0.7, rank=k)
        angles = scipy.linalg.subspace_angles(sr.eigvecs, laplacian.eigvecs[:, :k])
        assert angles.max() < 1e-8
        np.testing.assert_allclose(sr.eigvals, laplacian.eigvals[:k], atol=1e-8)
    # guard: the test discriminates -- the uniform-over-neighbours walk is not a
    # function of the distance-weighted Laplacian on an irregular graph
    adjacency = nx.to_numpy_array(env.connectivity, nodelist=range(env.n_bins)) > 0
    uniform = adjacency / adjacency.sum(axis=1, keepdims=True)
    sr_uniform = build_successor_basis(env, uniform, gamma=0.7, rank=ranks[-1])
    assert scipy.linalg.subspace_angles(
        sr_uniform.eigvecs, laplacian.eigvecs[:, : ranks[-1]]
    ).max() > 1e-3
```

The skew test on `successor_representation` (fast):

```python
def test_successor_columns_skew_against_travel_direction(ring_env):
    rng = np.random.default_rng(1)
    times, xy = _ring_trajectory(1200.0, directional=True, rng=rng)
    M = successor_representation(estimate_transition_matrix(ring_env, times, xy), 0.8)
    centroid = np.array(
        [np.sum(M[:, j] / M[:, j].sum() * _signed_angular_offset(ring_env, j)) for j in range(ring_env.n_bins)]
    )
    assert np.all(centroid < -0.1)  # mass behind the goal bin (observed mean -0.37 rad)
    times, xy = _ring_trajectory(1200.0, directional=False, rng=rng)
    M_sym = successor_representation(estimate_transition_matrix(ring_env, times, xy), 0.8)
    centroid_sym = np.array(
        [np.sum(M_sym[:, j] / M_sym[:, j].sum() * _signed_angular_offset(ring_env, j)) for j in range(ring_env.n_bins)]
    )
    assert abs(centroid_sym.mean()) < 0.05  # observed 0.003
```

The positive control and the tie control (slow):

```python
@pytest.mark.slow
def test_successor_basis_beats_laplacian_on_successor_shaped_fields(ring_env):
    rng = np.random.default_rng(0)
    policy = estimate_transition_matrix(
        ring_env, *_ring_trajectory(1200.0, directional=True, rng=rng)
    )
    goal_bins = np.linspace(0, ring_env.n_bins - 1, 24, dtype=int)
    rate_map = _successor_rate_maps(policy, 0.8, goal_bins)
    times, xy = _ring_trajectory(600.0, directional=True, rng=rng)
    spikes = _poisson_spikes(ring_env, times, xy, rate_map, rng)
    result = select_gamma_by_held_out_ll(ring_env, times, xy, spikes, dt=RING_DT, rank=10)
    best = np.argmax(result.held_out_ll.sum(axis=1))
    margin = (result.held_out_ll[best].sum() - result.laplacian_ll.sum()) / result.n_held_out_spikes.sum()
    assert margin > 0.005  # observed +0.0125 .. +0.0147 nats per held-out spike
    assert np.all(result.held_out_ll[best] > result.laplacian_ll)  # every fold
    assert not result.best_gamma_at_grid_edge  # observed best gamma 0.8-0.9
    # guard: a too-short grid reports the edge
    short = select_gamma_by_held_out_ll(ring_env, times, xy, spikes, dt=RING_DT, rank=10, gammas=(0.3, 0.5))
    assert short.best_gamma == 0.5 and short.best_gamma_at_grid_edge


@pytest.mark.slow
def test_directed_and_symmetrised_successor_bases_tie_on_symmetric_fields(ring_env):
    rng = np.random.default_rng(2)
    goal_bins = np.linspace(0, ring_env.n_bins - 1, 24, dtype=int)
    offsets = np.stack([_signed_angular_offset(ring_env, j) for j in goal_bins], axis=1)
    rate_map = 0.5 + 19.5 * np.exp(-0.5 * (RING_RADIUS * offsets / 12.0) ** 2)
    times, xy = _ring_trajectory(600.0, directional=False, rng=rng)
    spikes = _poisson_spikes(ring_env, times, xy, rate_map, rng)
    directed = select_gamma_by_held_out_ll(ring_env, times, xy, spikes, dt=RING_DT, rank=10)
    symmetrised = select_gamma_by_held_out_ll(ring_env, times, xy, spikes, dt=RING_DT, rank=10, symmetrize=True)
    n_spikes = directed.n_held_out_spikes.sum()
    gap = (directed.held_out_ll.sum(axis=1).max() - symmetrised.held_out_ll.sum(axis=1).max()) / n_spikes
    assert abs(gap) < 0.002  # observed |gap| <= 1e-4
    # guard: the comparison has power here -- both successor arms beat the Laplacian
    # (movement-graph adaptation; observed +0.0038 .. +0.0054)
    assert directed.held_out_ll.sum(axis=1).max() > directed.laplacian_ll.sum()
    assert symmetrised.held_out_ll.sum(axis=1).max() > symmetrised.laplacian_ll.sum()
```

The downstream-contract smoke test (slow; imports `jax.numpy as jnp`, `log_conditional_intensity` and `stochastic_point_process_filter` from `point_process_kalman`):

```python
@pytest.mark.slow
def test_successor_design_matrix_drives_the_laplace_ekf_filter(ring_env):
    rng = np.random.default_rng(3)
    policy = estimate_transition_matrix(ring_env, *_ring_trajectory(1200.0, directional=True, rng=rng))
    rate_map = _successor_rate_maps(policy, 0.8, np.array([0, ring_env.n_bins // 2]))
    times, xy = _ring_trajectory(120.0, directional=True, rng=rng)
    spikes = _poisson_spikes(ring_env, times, xy, rate_map, rng)
    basis = build_successor_basis(ring_env, policy, gamma=0.9, rank=10)
    Z, valid = graph_design_matrix(ring_env, basis, times, xy)
    Z_lap, valid_lap = graph_design_matrix(ring_env, build_graph_basis(ring_env, rank=10), times, xy)
    assert np.array_equal(valid, valid_lap)  # same bin lookup, same masking

    def marginal_ll(design):
        _, _, ll = stochastic_point_process_filter(
            init_mean_params=jnp.zeros(10),
            init_covariance_params=0.1 * jnp.eye(10),
            design_matrix=jnp.asarray(design[valid]),
            spike_indicator=jnp.asarray(spikes[valid, 0]),
            dt=RING_DT,
            transition_matrix=jnp.eye(10),
            process_cov=1e-6 * jnp.eye(10),
            log_conditional_intensity=log_conditional_intensity,
        )
        return float(ll)

    ll = marginal_ll(Z)
    assert np.isfinite(ll)
    # a time-shuffled design (same rows, wrong times) must fit far worse
    assert ll - marginal_ll(Z[rng.permutation(Z.shape[0])]) > 10.0
```

(The prototype's per-neuron filter LL differences between SR and Laplacian were −26 … +16 nats, too noisy to assert a direction there; the shuffled-design control is the assertion that can fail.)

The remaining tests in the validation slice are short, single-purpose and follow the patterns at `test_graph_place_field.py:103-126` and `:245-292`; their asserted behaviour is fully specified in the table.

### Task 8 — `scripts/successor_basis_comparison.py` (real data)

Plain script in the style of `scripts/position_decoding_demo.py:1-73` (x64 before imports, `PROJECT_ROOT`, the gitignored loader, `select_units` / `bin_spike_times` / `interpolate_to_new_times`). Module docstring states the decision rule verbatim:

```python
"""Compare spatial bases for static place fields on a real session by held-out spike LL.

Three bases at matched rank, scored by contiguous-block cross-validated Poisson
log-likelihood of a static ridge-penalised place-field fit (graph_place_field):

* the graph-Laplacian eigenbasis (current default, ``build_graph_basis``);
* the successor-representation basis of the animal's movement
  (``build_successor_basis``, discount gamma selected on the same folds);
* the successor basis of the *symmetrised* movement (direction removed): a control
  that separates movement-graph adaptation from direction dependence.

Decision rule -- adopt the successor basis as the default spatial basis:
    ADOPT if, at every rank in RANKS, the directed successor basis at its selected
    gamma beats the Laplacian on at least 4 of the 5 folds and by >= MARGIN nats per
    held-out spike pooled over folds; otherwise KEEP the Laplacian.
Interpretation: (directed - symmetrised) >= MARGIN indicates direction-dependent
structure in the basis (Stachenfeld et al. 2017; Mehta et al. 2000);
|directed - symmetrised| < MARGIN means the gain is movement-graph adaptation, and
direction must be read off the fitted fields -- the skew diagnostic printed last
(negative = field mass behind the peak relative to travel, Mehta et al. 2000).

Loads the J16 bandit session through the gitignored ``data/`` loaders; run from the
repository root: ``PYTHONPATH=. uv run --no-sync python scripts/successor_basis_comparison.py``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from neurospatial import Environment

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data/ holds local, gitignored loaders; run from the repo root with PYTHONPATH=.
from data.load_bandit_data import load_neural_recording_from_files
from state_space_practice.graph_place_field import (
    bin_occupancy,
    bin_spike_counts,
    build_successor_basis,
    estimate_transition_matrix,
    fit_penalized_poisson_field,
    select_gamma_by_held_out_ll,
)
from state_space_practice.preprocessing import (
    bin_spike_times,
    interpolate_to_new_times,
    select_units,
)

DT = 0.004  # s; 250 Hz like the other scripts
BIN_SIZE = 5.0  # cm
SPEED_THRESHOLD = 4.0  # cm/s; fields are estimated from running samples
RANKS = (10, 20, 40)
N_FOLDS = 5
MARGIN = 0.002  # nats per held-out spike (prototype: real effects ~0.01, noise ~0.001)
MIN_PEAK_HZ = 3.0  # cells entering the skew diagnostic


def field_skew_along_flow(env, transition, rate_maps):
    """Signed distance (cm) of each field's mass from its peak along local travel.

    ``flow_i = sum_j T_ij (c_j - c_i)`` is the mean displacement out of bin ``i``.
    For a field ``r`` with peak bin ``p``: ``sum_s r(s) <c_s - c_p, flow_p / |flow_p|>
    / sum_s r(s)``. Negative = mass behind the peak (Mehta et al. 2000). NaN where the
    local flow is undirected (|flow_p| ~ 0, e.g. traversed both ways).
    """
    centers = np.asarray(env.bin_centers)
    flow = transition @ centers - centers
    skews = []
    for rate in rate_maps.T:
        peak = int(np.argmax(rate))
        norm = np.linalg.norm(flow[peak])
        if norm < 1e-9:
            skews.append(np.nan)
            continue
        along = (centers - centers[peak]) @ (flow[peak] / norm)
        skews.append(float(np.sum(rate * along) / np.sum(rate)))
    return np.asarray(skews)


def main():
    data = load_neural_recording_from_files(PROJECT_ROOT / "data", "j1620210710_02_r1")
    position_info = data["position_info"]
    spike_times = data["spike_times"]
    pos_times = position_info.index.values
    t_start, t_end = pos_times[0], pos_times[-1]
    time_bins = np.arange(t_start, t_end, DT)

    selected = select_units(spike_times, min_rate=0.5, start_time=t_start, end_time=t_end)
    spikes = bin_spike_times([spike_times[i] for i in selected], time_bins)
    xy = np.column_stack(
        [
            interpolate_to_new_times(position_info[c].values, pos_times, time_bins)
            for c in ("head_position_x", "head_position_y")
        ]
    )
    speed = interpolate_to_new_times(position_info["head_speed"].values, pos_times, time_bins)
    running = speed > SPEED_THRESHOLD
    times_r, xy_r, spikes_r = time_bins[running], xy[running], spikes[running]
    env = Environment.from_samples(xy_r, bin_size=BIN_SIZE, bin_count_threshold=5)
    print(
        f"{len(selected)} neurons, {running.sum()} running samples "
        f"({running.mean():.0%} of session), {env.n_bins} active bins"
    )

    print("rank  Laplacian-LL  dir-Lap  sym-Lap  dir-sym  gamma(dir)  gamma(sym)  folds dir>Lap")
    adopt = True
    directed = None
    for rank in RANKS:
        directed = select_gamma_by_held_out_ll(env, times_r, xy_r, spikes_r, dt=DT, rank=rank, n_folds=N_FOLDS)
        symmetrised = select_gamma_by_held_out_ll(
            env, times_r, xy_r, spikes_r, dt=DT, rank=rank, n_folds=N_FOLDS, symmetrize=True
        )
        n_spikes = directed.n_held_out_spikes.sum()
        laplacian = directed.laplacian_ll.sum()
        dir_total = directed.held_out_ll.sum(axis=1)
        sym_total = symmetrised.held_out_ll.sum(axis=1)
        d, s = int(np.argmax(dir_total)), int(np.argmax(sym_total))
        folds_won = int(np.sum(directed.held_out_ll[d] > directed.laplacian_ll))
        m_dir = (dir_total[d] - laplacian) / n_spikes
        m_sym = (sym_total[s] - laplacian) / n_spikes
        m_dir_sym = (dir_total[d] - sym_total[s]) / n_spikes
        edge = " (grid edge)" if directed.best_gamma_at_grid_edge else ""
        print(
            f"{rank:4d}  {laplacian:12.1f}  {m_dir:+.4f}  {m_sym:+.4f}  {m_dir_sym:+.4f}  "
            f"{directed.best_gamma:.2f}{edge}  {symmetrised.best_gamma:.2f}  {folds_won}/{N_FOLDS}"
        )
        adopt &= folds_won >= 4 and m_dir >= MARGIN
    print("(margins in nats per held-out spike)")
    print("DECISION:", "ADOPT the successor basis" if adopt else "KEEP the Laplacian basis")

    # Coefficient-level direction diagnostic at the largest rank (fields, not basis).
    transition = estimate_transition_matrix(env, times_r, xy_r)
    basis = build_successor_basis(env, transition, gamma=directed.best_gamma, rank=RANKS[-1])
    counts = bin_spike_counts(env, spikes_r, times_r, xy_r, basis)
    occupancy = bin_occupancy(env, times_r, xy_r, DT)
    coeffs = fit_penalized_poisson_field(counts, occupancy, basis.eigvecs, penalty=1e-3)
    rate_maps = np.exp(basis.eigvecs @ coeffs)
    skews = field_skew_along_flow(env, transition, rate_maps[:, rate_maps.max(axis=0) >= MIN_PEAK_HZ])
    skews = skews[np.isfinite(skews)]
    print(
        f"field skew along travel: median {np.median(skews):+.1f} cm over {skews.size} cells; "
        f"fraction behind the peak = {np.mean(skews < 0):.2f} (Mehta et al. 2000 predicts < 0)"
    )


if __name__ == "__main__":
    main()
```

`ruff format` applies to `scripts/` (lint does not, `pyproject.toml:117`). The script cannot be run in this checkout (no `data/`); the executor runs it where the loaders exist and pastes the printed table into the PR description.

### Task 9 — user-facing documentation and tooling

- `CHANGELOG.md` under `## [Unreleased]` / `### Added` (`:6-8`): one bullet — "**Successor-representation spatial basis** in `graph_place_field`: `estimate_transition_matrix` (adjacency-restricted, smoothed bin-to-bin movement chain), `successor_representation`, `build_successor_basis` (`SuccessorBasis`, a drop-in for `GraphBasis` in `graph_design_matrix` / `bin_spike_counts` / `spectral_shape`), `fit_penalized_poisson_field` / `poisson_field_log_likelihood` (static per-bin Poisson fit and held-out log-likelihood in any spatial basis) and `select_gamma_by_held_out_ll` (contiguous-block cross-validated gamma sweep with the Laplacian basis at matched rank as baseline). `scripts/successor_basis_comparison.py` runs the three-way comparison on a real session and states the adoption rule."
- `README.md:27`: change the row's "Used by" cell to `` `graph_place_field` (Laplacian and successor-representation bases) ``.
- `pyproject.toml:134-141`: add `"src/state_space_practice/graph_place_field.py",` after the `exceptions.py` line (the module is mypy-clean today — verified with `uv run --no-sync mypy src/state_space_practice/graph_place_field.py`; keep it so).
- Public docstrings are written in Tasks 1–5 (NumPy style, shapes documented); no separate docs pass.
- Final checks the executor runs: `uv run ruff check src/ && uv run ruff format src/ scripts/`, `uv run mypy`, `uv run pytest src/state_space_practice/tests/test_graph_place_field.py -v` (then `-m "not slow"` for the fast subset).

## Deliberately not in this plan

- **An option on `PlaceFieldModel` to take a graph/SR basis.** Its basis is the patsy spline built in `_build_spline_basis_matrix` (`place_field_model.py:567-592`) and consumed by `score` / `predict_rate_map` through `basis_info`; threading an external basis through warm-start, block dispatch and prediction is the drifting-graph-GP plan's `GraphPlaceFieldModel`, not an additive flag. The seam this plan preserves is the per-time design matrix from `graph_design_matrix`.
- **A direction-sensitive basis** (e.g. a dictionary of SR columns at anchor bins). Ruled in by the design finding but a different construction — see Open Question 1.
- **A dwell-including (per-sample) transition chain** — Open Question 2.
- **Sparse/iterative SVD (`svds`) and caching of SR bases** — only if a real environment exceeds ~2500 bins.
- **Learning the SR online or from spikes** (Fang et al. 2023; Bono et al. 2023) — this plan estimates it from behaviour.
- **Spectral (`eigvals`-based) penalties in the comparison** — Open Question 4.

## Validation slice

All in `src/state_space_practice/tests/test_graph_place_field.py`; `slow` marked explicitly (the conftest auto-marker keys on `.fit(`/`run_em(` calls, which these tests do not make). Margins are calibrated from the planning prototype (three seeds); the fixed seeds make the slow tests deterministic. The library code of Tasks 2–5 was executed as written during planning against every assertion below (all passed: positive-control margin `+0.0156` nats/spike with per-fold gains of 53–132 nats, tie gap `−1e-5`, exact property to `1e-8` on all three environments with the uniform-walk guard at 0.10–0.83 rad, coefficient recovery error 0.8%; the whole set ran in ~1 s, so the `slow` marks are for the pipeline convention, not runtime).

| Test | Asserts |
| --- | --- |
| `test_transition_matrix_counts_bin_changes_not_dwell` (`small_grid_env`) | Path `[a, b, a]` with 5 samples per bin, `smoothing_alpha=0`: `T[a, b] == 1.0`, `T[b, a] == 1.0`, `trace(T) == 0`; every unvisited row equals `A[i] / deg(i)`; rows sum to 1. |
| `test_transition_matrix_smoothing_only_on_edges` | Same path, `smoothing_alpha=0.5`: `T[a, b] == 1.5 / (1 + 0.5 deg(a))`, other neighbours of `a` get `0.5 / (1 + 0.5 deg(a))`, non-adjacent entries are exactly 0. |
| `test_transition_matrix_breaks_at_out_of_bounds_and_gaps` | Path `[a, OOB, b]` gives no `a -> b` mass with `alpha=0` (row `a` is the uniform fallback); path `[a, b, c]` with a 10 s gap between `a` and `b` and `max_gap=1.0` counts only `b -> c`. |
| `test_transition_matrix_warns_on_nonadjacent_jumps` | A path alternating between two non-adjacent bins raises `StateSpaceWarning` (via `pytest.warns`) and the returned `T` has zero mass on that pair. |
| `test_transition_matrix_rejects_degenerate_input` | All samples in one bin raises `ValueError` ("no bin-to-bin moves"); `smoothing_alpha=-1` and `max_gap=0` raise `ValueError`. |
| `test_transition_matrix_reflects_travel_direction` (`ring_env`) | Counter-clockwise trajectory, `alpha=0`: mean forward mass `sum_j T[i, j]·[offset > 0]` over bins `> 0.75` (observed `0.85`; on the annular band some neighbours are transverse, neither ahead nor behind); with `symmetrize=True` it lies within `0.1` of `0.5`. |
| `test_successor_basis_matches_laplacian_for_lazy_symmetric_walk` (parametrised over `ring_env`, `small_grid_env`, `w_maze_env`) | For `T = I - eps L`, at ranks with a spectral gap: principal angles `< 1e-8` and `eigvals == laplacian.eigvals[:k]` (`atol 1e-8`); guard: the uniform-over-neighbours walk gives max angle `> 1e-3`. |
| `test_successor_basis_is_orthonormal_component_local_and_read_only` (`two_component_env`) | With the uniform walk: `eigvecsᵀ eigvecs == I` (`atol 1e-10`); each column supported on one component; `n_components == 2`; singular values descending; `eigvals` ascending with `eigvals[0] == 0`; writing to `eigvecs` raises `ValueError`; `rank=1` raises `ValueError` (drops a null mode); `kind == "successor"`, `gamma` stored. |
| `test_successor_basis_rejects_bad_transition_or_gamma` | Wrong shape, rows not summing to 1, negative entries, `1e-6` mass between the two components of `two_component_env`, and `gamma in (0.0, 1.0, -0.1)` each raise `ValueError`. |
| `test_successor_representation_is_neumann_series` | `successor_representation(T, 0.5)` matches `sum_{n<=60} 0.5^n T^n` to `atol 1e-9` on `small_grid_env`'s uniform walk; `M @ 1 == 1 / (1 - gamma)`. |
| `test_successor_columns_skew_against_travel_direction` (`ring_env`) | All column centroids of `M` under the directional policy have angular offset `< -0.1` rad (behind the goal bin); under the run-and-tumble policy the mean centroid offset has magnitude `< 0.05`. |
| `test_successor_basis_feeds_graph_design_matrix` (`small_grid_env`, `two_component_env`) | `graph_design_matrix(env, sr_basis, times, centre_of_bin_b)` returns `Z[0] == sr.eigvecs[b]` and `valid.all()`; the same basis with the other environment raises `ValueError` matching `"different"`; `bin_spike_counts` accepts the SR basis and matches the Laplacian-basis call exactly. |
| `test_fit_penalized_poisson_field_recovers_field_in_span` (`small_grid_env`) | Counts drawn from a known log-rate in the span of a rank-6 Laplacian basis with 200 s exposure per bin: relative coefficient error `< 5%` at `penalty=1e-6`; `poisson_field_log_likelihood` at the fit `>=` at the truth `>` at zero coefficients; a zero-entry penalty vector is accepted and a negative one raises. |
| `test_poisson_field_log_likelihood_exposure_edge_cases` | Appending zero-count/zero-exposure bins leaves the LL unchanged; a positive count in a zero-exposure bin raises `ValueError` from both functions; `counts` of shape `(n_bins,)` returns shape `(1,)`. |
| `test_select_gamma_validates_inputs` | `n_folds=1`, `dt=0`, a gamma of `1.0`, `rank` below `n_components` (on `two_component_env`), and mismatched row counts each raise `ValueError` before any fitting. |
| `test_successor_basis_beats_laplacian_on_successor_shaped_fields` (**slow**, `ring_env`) | Fields = SR columns (`gamma=0.8`) of the directional policy; rank 10, default grid: pooled margin over the Laplacian `> 0.005` nats/held-out spike (observed `+0.0125 … +0.0147`), positive on every fold, `best_gamma_at_grid_edge is False`; a `(0.3, 0.5)` grid reports `best_gamma == 0.5` and the edge flag. |
| `test_directed_and_symmetrised_successor_bases_tie_on_symmetric_fields` (**slow**, `ring_env`) | Gaussian bumps (12 cm) + run-and-tumble policy, rank 10: `abs(directed − symmetrised) < 0.002` nats/spike (observed at most `1e-4`); guard: both arms beat the Laplacian (observed `+0.0038 … +0.0054`). |
| `test_successor_design_matrix_drives_the_laplace_ekf_filter` (**slow**, `ring_env`) | `valid` masks from SR and Laplacian design matrices are identical; the single-neuron Laplace-EKF marginal LL on the SR design matrix is finite and exceeds the time-shuffled design's by `> 10` nats. |
| Existing tests `:53-258` | Unchanged and green — in particular `test_basis_cache_rank_safe`, `test_read_only_outputs`, `test_consumers_reject_mismatched_env` (message still contains `"different"`). |

## Fixtures

- Reuse the module-scoped `small_grid_env` (25 bins), `w_maze_env` and `two_component_env` at `test_graph_place_field.py:30-50`.
- New module-scoped `ring_env` (Task 7): deterministic (`linspace`, no RNG) annular band from `Environment.from_samples` on a circle of radius 30 with 5 cm bins; one component, degrees 2–4 in the installed neurospatial (48 bins there — never hard-coded).
- Trajectories and spikes are synthesised in-test by the helpers in Task 7 with explicit `np.random.default_rng` seeds; the 1200 s "policy" trajectory defines the generating `T`, a separate 600 s trajectory generates the fitted data. No checked-in data.
- Real data only in `scripts/successor_basis_comparison.py` via the gitignored `data/` loaders (absent here).

## Open questions

1. **Direction-sensitive basis.** The design finding shows the SVD basis cannot test the Stachenfeld/Mehta directional prediction. The natural direction-sensitive construction is a dictionary of SR columns at anchor bins, `M[:, anchors]` (each atom is a backward-skewed field), or its symmetrised twin for control. Current answer: out of scope; trigger = the real-data script prints `|directed − symmetrised| < MARGIN` **and** a clearly negative skew diagnostic, i.e. direction is in the coefficients and a localised skewed basis could lower the rank needed. Would be its own plan.
2. **Dwell-including chain.** `self_transitions=True` (per-sample chain, `gamma` per sample, horizon `dt / (1 − gamma)` seconds) reproduces Stachenfeld's discrete-time SR with dwell. Current answer: not needed for the basis comparison (dwell is already the exposure); trigger = wanting the SR to encode dwell-time structure (e.g. reward-site lingering) as a spatial prior.
3. **Behaviour-only leakage.** `T` is re-estimated from training rows per fold (D12). Estimating once from the whole trajectory would be cheaper and arguably legitimate (no spikes involved); current answer: keep the conservative choice, revisit only if the script's runtime on a full session is a problem (35 SVDs of `n_bins²`).
4. **Prior vs subspace.** The comparison uses an isotropic ridge (D11), so it measures subspace quality. Comparing with the spectral penalty `spectral_shape(basis.eigvals, kappa2)` on both bases would also test the Laplacian-equivalent `eigvals` of D7 as a prior; current answer: deferred to the drifting-graph-GP plan's phase 1, which is where the spectral prior is fitted.
5. **neurospatial install source.** The `.venv` carries an editable sibling checkout at a different commit than the lock (see Dependencies). Current answer: the APIs used are stable across both; the executor verifies the source once and reports the ring fixture's `n_bins` in the PR description.

## Estimated effort

Roughly +430 LOC in `graph_place_field.py` (types, transition estimator, SR basis, static fit, CV sweep), +420 LOC of tests, ~170 LOC script, ~15 lines across `CHANGELOG.md` / `README.md` / `pyproject.toml`. One PR; four commits as suggested at the top of Tasks.

## Review

Before opening the PR, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task is implemented as specified; nothing in "Deliberately not in this plan" was added — in particular no change to `build_graph_basis` / `GraphBasis` and no `PlaceFieldModel` option.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.) Check specifically that the two slow comparison tests keep their guards (per-fold positivity; both SR arms beating the Laplacian in the tie test) and that the exact-property test keeps its discriminating uniform-walk guard.
- Docstrings, test names, and module names don't reference this plan.
- The module docstring and `build_successor_basis` docstring state that travel direction is a coefficient property, not a basis property, so users do not read an SR-vs-Laplacian win as evidence for direction-dependent fields.
- `uv run ruff check src/`, `uv run ruff format --check src/ scripts/`, `uv run mypy` (with `graph_place_field.py` now in the `files` list) are clean.
