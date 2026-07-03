# Phase 2 — Fix `CorrelatedNoiseModel`'s invalid process covariance

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

`construct_correlated_noise_process_covariance` assembles off-diagonal 2×2 blocks from
*independent* directed `phase_difference`/`coupling_strength` entries, so block `(i,j)`
and block `(j,i)` are unrelated and the assembled `Q` is generically **non-symmetric and
non-PSD** — invalid as a covariance, yet fed straight into the switching Kalman filter's
`slogdet(Q)`/`psd_solve(Q,·)`. Make `Q` symmetric-PSD by construction (tie blocks; see
[overview Open Question 1](overview.md#open-questions)) and PSD-project during fitting.
The existing tests cannot catch this — fix them too.

**Inputs to read first:**

- `src/state_space_practice/oscillator_utils.py:253-293`
  (`construct_correlated_noise_process_covariance`) — the function to rewrite. Current
  logic: `vmap(vmap(_compute_coupling_transition_block))` over
  `(phase_difference, coupling_strength)` → `(n,n,2,2)`, overwrite diagonal blocks with
  `variance·I`, reshape to `(2n,2n)`.
- `src/state_space_practice/oscillator_utils.py:149-170`
  (`_compute_coupling_transition_block`) — reused unchanged: returns
  `coupling·R(phase)` (a scaled rotation), zero when coupling≈0.
- `src/state_space_practice/oscillator_models.py:1255-1282`
  (`_initialize_process_covariance`, `_project_parameters`).
- `src/state_space_practice/tests/test_oscillator_models.py:48-69` (fixture, zero
  coupling), `:689-698` (PSD test via `eigvalsh`), `:877-896` (symmetry test, zero
  coupling hardcoded).
- `src/state_space_practice/models.py` — locate the existing `stabilize_covariance`
  helper (symmetrize + eigenvalue floor) to reuse for the PSD projection; if unsuitable,
  use the inline `_project_cov_to_psd` below.

## Tasks

- **Rewrite `construct_correlated_noise_process_covariance` to build a symmetric `Q`.**
  Keep the signature. After computing `all_blocks` `(n,n,2,2)`, enforce that the `(j,i)`
  block equals the transpose of the `(i,j)` block using the upper triangle (`i<j`) as the
  source of truth, then set the diagonal blocks and reshape. Concretely:

  ```python
  n_oscillators = variance.shape[0]
  coupling_row = jax.vmap(_compute_coupling_transition_block, in_axes=(0, 0))
  all_blocks = jax.vmap(coupling_row, in_axes=(0, 0))(
      phase_difference, coupling_strength
  )  # (n, n, 2, 2)

  # Enforce covariance symmetry: block (j, i) must equal block (i, j)^T.
  # Use the strict upper triangle (i < j) as the source; mirror to the lower.
  # lower_source[i, j] = all_blocks[j, i]^T  (transpose of the mirrored block)
  lower_source = jnp.swapaxes(jnp.swapaxes(all_blocks, 0, 1), -1, -2)
  upper_mask = (jnp.arange(n_oscillators)[:, None]
                < jnp.arange(n_oscillators)[None, :])  # (n, n), True where i < j
  sym_blocks = jnp.where(upper_mask[..., None, None], all_blocks, lower_source)

  # Diagonal blocks are variance * I (overwrites the symmetric placeholder).
  diag_blocks = variance[:, None, None] * IDENTITY_2x2[None, :, :]
  diag_idx = jnp.arange(n_oscillators)
  sym_blocks = sym_blocks.at[diag_idx, diag_idx].set(diag_blocks)

  return sym_blocks.swapaxes(1, 2).reshape(2 * n_oscillators, 2 * n_oscillators)
  ```

  Update the docstring: state that only the strict upper triangle of
  `phase_difference`/`coupling_strength` is used and that the lower triangle is its
  transpose (covariance symmetry). This function is used by both
  `_initialize_process_covariance` and the SGD reconstruction path, so this single change
  covers construction and fitting.

- **PSD-project the assembled `Q` in `_project_parameters`** (`oscillator_models.py:1270`).
  Symmetry alone does not guarantee PSD for large coupling. `_project_parameters`'s job is
  to map parameters back into the valid space during fitting, so after the existing
  block-rotation projection, project each `process_cov[..., j]` to the nearest symmetric
  PSD matrix. Reuse `stabilize_covariance` if it floors eigenvalues; otherwise:

  ```python
  def _project_cov_to_psd(cov: jax.Array, min_eigenvalue: float = 1e-8) -> jax.Array:
      cov = 0.5 * (cov + cov.T)
      eigvals, eigvecs = jnp.linalg.eigh(cov)
      eigvals = jnp.maximum(eigvals, min_eigenvalue)
      return (eigvecs * eigvals) @ eigvecs.T
  ```

  Do **not** apply `project_matrix_blockwise` (rotation projection) as the PSD guard — a
  rotation matrix is not a valid covariance block.

- **Add a public-entry validation hook** so a user who *constructs* `CorrelatedNoiseModel`
  with a coupling that yields an indefinite `Q` gets a loud error rather than silent
  fitting on projected values. This is the CNM-specific instance of the shared validator
  built in [phase 3](phase-3-invariant-validation.md); if phase 3 lands first, call that
  validator from `CorrelatedNoiseModel.__init__` / `_initialize_process_covariance`. If
  this phase lands first, add a local `eigvalsh`-based symmetric-PSD check that raises
  `ValueError` naming the offending discrete state, and phase 3 will consolidate it.

- **Fix the tests to catch asymmetry** (`test_oscillator_models.py`):
  - Add a fixture (or parametrize the existing symmetry/PSD tests) with **nonzero,
    asymmetric** `phase_difference` and `coupling_strength` (e.g. `phase[0,1]=0.7`,
    `coupling[0,1]=0.4`, `phase[1,0]=0.3`, `coupling[1,0]=0.5`).
  - `test_process_cov_symmetric_for_all_states` (`:877`): use the nonzero fixture and keep
    `assert_allclose(Q_j, Q_j.T, atol=1e-10)`. This must FAIL on the pre-fix constructor
    and PASS after (verify by running against the old code first).
  - `test_process_cov_positive_semidefinite` (`:689`): switch the PSD check from
    `jnp.linalg.eigvalsh` to `jnp.linalg.eigvals(Q_j).real` (general eigenvalues, which do
    **not** implicitly symmetrize) and assert `>= -1e-8`. Add a guard asserting coupling is
    actually nonzero so the test can't pass vacuously.

## Deliberately not in this phase

- Do not touch the DIM directed *transition*-matrix construction
  (`construct_directed_influence_transition_matrix`) — asymmetry there is correct (it is a
  transition matrix, not a covariance).
- Do not build the general shared covariance validator here beyond the CNM entry hook —
  that is [phase 3](phase-3-invariant-validation.md).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_process_cov_symmetric_for_all_states` (nonzero coupling) | `max|Q_j − Q_jᵀ| < 1e-10`; fails on pre-fix constructor |
| `test_process_cov_positive_semidefinite` (nonzero coupling, `eigvals`) | `min Re(eigvals(Q_j)) ≥ -1e-8`; guard: coupling ≠ 0 |
| new `test_correlated_noise_rejects_indefinite_Q` | `ValueError` when a chosen coupling makes symmetric `Q` indefinite |
| `test_scenario_recovery.py::...CNM...` (slow) | Segmentation-accuracy threshold still met with corrected `Q` (regression check) — mark `slow` |
| `TestCorrelatedNoiseModel` (existing) | Still pass (fit runs, shapes, LL improves) |

## Fixtures

Nonzero-coupling `CorrelatedNoiseModel` parameters synthesized in the test module (small,
`n_oscillators=2`, `n_discrete_states=2-3`). No real data. The scenario-recovery
regression uses the existing `test_scenario_recovery.py` fixture.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The symmetry test was demonstrated to FAIL on the pre-fix constructor (paste evidence in
  the PR) — otherwise it does not actually guard the bug.
- PSD checks use `eigvals`, not `eigvalsh`, everywhere symmetry is not independently
  asserted.
- The rewrite changed only symmetry/validity, not the diagonal-variance or block-rotation
  semantics for the zero-coupling case (existing zero-coupling tests still pass unchanged).
- `_project_parameters` uses a covariance PSD projection, not a rotation projection, for
  the `Q` guard.
- Docstring reflects the "upper triangle is the source of truth" convention.
