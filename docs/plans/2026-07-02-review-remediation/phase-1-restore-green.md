# Phase 1 — Restore a green fast suite (test-only)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

The fast suite has one deterministic failure. Its assertion is brittle, not a real
regression: the quantity it bounds is a single-bin, denominator-dominated artifact that
is *non-monotonic* in the per-step cap it claims to test. Rewrite the assertion to the
mechanism's real invariant. Also seed the suite's one unseeded RNG. **No `src/` changes.**

**Inputs to read first:**

- `src/state_space_practice/tests/test_position_decoder.py:1127-1156` — the failing test
  and its relaxed-3-times threshold comment.
- `src/state_space_practice/position_decoder.py:944-964` — the inflation mechanism under
  test: `alpha_t = clip(1 + gain·(s_t − 1), 1, max_alpha)` multiplies the one-step
  predicted covariance. The invariant is per-step multiplicative, capped at `max_alpha`.
- `src/state_space_practice/tests/test_position_decoder.py:1062-1088` — the
  `rate_maps_and_data` fixture (seeded `default_rng(42)`), for how to obtain per-step
  covariances.

## Tasks

- Rewrite `test_capped_inflation_bounds_covariance_growth`
  (`test_position_decoder.py:1127`) to assert the **real per-step-cap invariant** instead
  of a fixed bound on the accumulated trace ratio. Concretely, compare the *one-step
  predicted* covariances that inflation actually multiplies (not the post-measurement
  covariances, whose mixing is what makes the current ratio meaningless): run the filter
  once with `AdaptiveInflationConfig(gain=100.0, max_alpha=1.01)` and once with inflation
  disabled, and assert, per time bin, that the capped predicted-covariance trace does not
  exceed `max_alpha` times the uncapped predicted-covariance trace (within tolerance).
  If the public result does not expose the pre-update predicted covariance, assert the
  weaker but still mechanism-true monotone-ordering property: a *tighter* `max_alpha`
  never yields a larger per-bin covariance trace than a *looser* one at the same bin.
- Delete the misleading threshold-history comment (`test_position_decoder.py:1148-1155`)
  and replace it with a one-line statement of the invariant now asserted.
- Add a **guard assertion** that the cap actually engaged for at least one bin (e.g. some
  bin's inflation factor equals `max_alpha` within tolerance), so the test cannot pass
  vacuously on a run where inflation never triggered.
- Seed the unseeded RNG: `test_position_decoder.py:184-185` uses bare
  `np.random.randn(100, 2)` / `np.random.poisson(0.1, (90, 3))`. Replace with draws from a
  `np.random.default_rng(seed)` instance (matching the rest of the suite).

## Deliberately not in this phase

- **Do not modify `position_decoder.py`.** The implementation is correct; the test was
  wrong. (`AdaptiveInflationConfig`'s missing *upper* bound is a separate defect handled in
  [phase 3](phase-3-invariant-validation.md).)
- Do not touch other shape-only or tautological tests — that cleanup is a separate concern
  and not needed to restore green.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_capped_inflation_bounds_covariance_growth` (rewritten) | Per-bin capped predicted-cov trace ≤ `max_alpha` × uncapped (or the monotone-ordering fallback); **and** at least one bin hit the cap |
| full fast suite | `pytest -m "not slow"` → 0 failed, 1324 passed |

## Fixtures

Reuse the existing seeded `rate_maps_and_data` fixture
(`test_position_decoder.py:1062`). No new data.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- The rewritten test asserts a mechanism-true invariant, not a re-tuned magic number, and
  includes the cap-engaged guard assertion (can genuinely fail).
- `position_decoder.py` is untouched (`git diff --stat` shows tests only).
- Full fast suite is green; no test was marked `slow` to hide a failure.
- Test names/docstrings don't reference this plan.
