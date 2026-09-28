# Phase 1 — `streaming.py`: `FilterState`, `StreamingKalmanFilter`, oscillator phase posterior, latency benchmark

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d1-module-layout-and-compiled-step-pattern)

Ships a per-bin Gaussian filter whose `.run()` is bit-identical to
`kalman_filter`, the phase / amplitude posterior of a 2-D oscillator block
(real-time theta phase, Wodeyar, Schatza, Widge, Eden & Kramer 2021, eLife
10:e68803 — a state-space model with a rotating 2-D block whose phase is the
angle of the filtered state), the latency benchmark script, and the user-facing
docs for all of it.

**Inputs to read first:**

- `src/state_space_practice/kalman.py:426-475` — `kalman_measurement_update`, the public single-step measurement update the streaming step calls.
- `src/state_space_practice/kalman.py:478-526` — `_kalman_filter_update`; lines 517-526 are the prediction + update the streaming step reproduces (the public alias `kalman_filter_update` is assigned at kalman.py:1631-1634).
- `src/state_space_practice/kalman.py:529-610` — `_kalman_filter_impl`: dtype promotion at 551-567 (mirror it), scan body at 570-594 (what parity is measured against, including the LL accumulation order at 589).
- `src/state_space_practice/kalman.py:270-423` — `_validate_kalman_public_inputs`: shape/value checks and error wording to mirror at construction.
- `src/state_space_practice/utils.py:1308-1405` (`validate_covariance`), `:1113` (`validate_int`), `:26` (`symmetrize`).
- `src/state_space_practice/oscillator_utils.py:55-73` (`get_block_slice`), `:199-250` (`construct_common_oscillator_transition_matrix`), `:253-274` (`construct_common_oscillator_process_covariance`), `:434-462` (a `[1, 0]`-per-block measurement matrix, the "LFP observes the real part" convention).
- `src/state_space_practice/coupling_model.py:285-302` and `src/state_space_practice/oscillator_models.py:1373-1378` — the `(re, im)` block convention and `arctan2(im, re)` phase.
- `src/state_space_practice/circular_stats.py:75-102` (`circular_std`), `:45-72` (`circular_mean`), `:410-424` (`wrap_to_pi`) — the NumPy references the phase tests compare against.
- `src/state_space_practice/__init__.py:35-53` (`_LAZY_API`), `:55-76` (`__all__`), `:78-100` (`TYPE_CHECKING` imports) — the lazy-export mechanism; `tests/test_package.py:34-45` checks every lazy name resolves to its defining module.
- `src/state_space_practice/tests/test_likelihood_identities.py:142-197` (`_lgssm`, `_kalman_args`, the module-scoped `lgssm` fixture) and `:287-305` (chain rule by restart) — reuse for the streaming LL identities.
- `src/state_space_practice/tests/test_oracle_kalman.py:45-94` (`_simulate_lgssm`, `_args`) — random stable problems for the parity tests.
- `src/state_space_practice/tests/test_kalman.py:2746-2760` — float32-promotion precedent; `tests/test_position_decoder.py:2021-2053` — the monkeypatch-and-count pattern for "compiles once".
- `pyproject.toml:128-160` — `[tool.mypy] files`; `CHANGELOG.md:6-8`; `README.md:60-68`.

**Contracts referenced:**

- [FilterState](shared-contracts.md#filterstate) — defined here; do not add fields.
- [Streaming filter surface](shared-contracts.md#streaming-filter-surface) — `step` / `run` / `reset` / `n_steps` semantics; `smoothed` is Phase 2.
- [Parity policy](shared-contracts.md#parity-policy) — Gaussian row: bit-identical.
- [Optional cross-plan keyword forwarding](shared-contracts.md#optional-cross-plan-keyword-forwarding) — `robust_weight` and `mask`.
- [Prediction hand-off](shared-contracts.md#prediction-hand-off-for-fixed-lag-smoothing) — the step accepts and returns `buf=None`.

**Designs referenced:** [D1](designs.md#d1-module-layout-and-compiled-step-pattern), [D2](designs.md#d2-gaussian-step-and-streamingkalmanfilter), [D3](designs.md#d3-phase-and-amplitude-posteriors-of-a-2-d-oscillator-block), [D7](designs.md#d7-latency-benchmark-script).

## Tasks

- **Create `src/state_space_practice/streaming.py`** with the module docstring
  (what "streaming" means here: one `jax.jit`-compiled step per bin on an
  explicit state; the x64 recipe from `__init__.py:6-13`; references: Särkkä
  2013 *Bayesian Filtering and Smoothing* for the recursions, Wodeyar et al.
  2021 for real-time phase from an oscillator state-space model),
  `FilterState`, `_accepts_kwarg`, `_StreamingFilter` (D1; without
  `smoothed`, and with `lag` accepted, validated and stored so Phase 2 does not
  change the constructor signature — `lag > 0` raises `NotImplementedError`
  in this phase), `_plain_kalman_update`, `_make_kalman_update`,
  `_kalman_streaming_step` (D2, without the `buf is not None` branch; the `buf`
  argument is accepted and returned unchanged) and `StreamingKalmanFilter`
  (D2). Construction validation and dtype promotion per D1; the float32
  `StateSpaceWarning` text should say why (unbounded T) and give the x64
  recipe. Public docstrings are NumPy-style with shapes and an example that
  streams a few bins and reads `state.mean`.
- **Add the oscillator posteriors** `PhasePosterior`, `AmplitudePosterior`,
  `phase_posterior`, `amplitude_posterior` (D3) to `streaming.py`, with the
  delta-method formulas, explicit numerical-origin fallbacks/flags (`phase_defined`, `half_width`, `used_delta_method`), and the "reliable for amplitude / amplitude_sd ≳ 10"
  note in the docstrings.
- **Export** `StreamingKalmanFilter`, `FilterState`, `phase_posterior`,
  `amplitude_posterior` lazily: add them to `_LAZY_API` (→ `"streaming"`),
  `__all__` and the `TYPE_CHECKING` block of `__init__.py`.
  `tests/test_package.py::TestPublicAPI` picks them up automatically.
- **Type-check the module**: add `"src/state_space_practice/streaming.py"` to
  `[tool.mypy] files` (pyproject.toml:134-159, alphabetical position) and make
  `uv run mypy` pass (`FilterState` fields typed `Array`; the update wrappers
  typed with `Callable[..., tuple[Array, Array, Array]]`).
- **Tests** — new `src/state_space_practice/tests/test_streaming.py` and a new
  `TestStreamingKalmanIdentities` class in `tests/test_likelihood_identities.py`
  (uses the existing `lgssm` fixture). Validation slice below; all tests
  import with x64 enabled through `conftest.py`.
- **Latency benchmark script** `scripts/benchmark_streaming_latency.py` (D7,
  Gaussian section and the dispatch floor). Run it once; paste the table into
  the PR description next to the expected-results table in
  [overview.md — Metrics](overview.md#metrics).
- **User-facing docs (this phase, not deferred):**
  - `CHANGELOG.md` under `## [Unreleased]` → `### Added` (line 8): an entry
    for `streaming.StreamingKalmanFilter` / `FilterState` /
    `phase_posterior` / `amplitude_posterior`, stating the bit-identical
    parity with `kalman_filter`, the per-step contract (compiled once, no
    value checks per step), and that `lag=`, masks and robust weights are
    accepted-but-not-yet-implemented where applicable.
  - `README.md`: add the four names to the entry-point list in "Package
    layout" (line 60-67) and a new subsection **"Streaming / real-time
    filtering"** after it with the theta-phase example (an 8 Hz block at
    1 kHz built with `construct_common_oscillator_transition_matrix` /
    `construct_common_oscillator_process_covariance`, `H = [[1, 0]]`, one
    `step` per LFP sample, `phase_posterior(state, block=0)`), and the
    measured per-step latency sentence from the Metrics section (dispatch
    floor, "batch bins if you need < 0.1 ms per bin").

## Deliberately not in this phase

- Fixed-lag smoothing (`lag > 0`, `.smoothed()`, the ring buffer) — Phase 2.
  The constructor already takes `lag` so its signature does not change later.
- `StreamingPointProcessFilter` (Phase 3a) and `StreamingPositionDecoder`
  (Phase 3b), and the point-process / decoder sections of the benchmark.
- Any local implementation of channel masks or robust weights — forwarding
  only, per the contract.
- Time-varying `R_t` (the batch filter accepts a `(n_time, n_obs, n_obs)`
  stack, kalman.py:640-645): out of scope for a per-bin API; a user with a
  per-bin `R_t` constructs one filter per noise level or waits for a
  follow-up. Document in the class docstring.
- Online parameter updates (adaptive `A`, `Q`, `R`), threading, acquisition
  integration — non-goals in overview.md.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_run_matches_kalman_filter_bitwise[(n_state, n_obs) ∈ {(1,1),(2,1),(4,3),(8,16)}]` | From a fresh `StreamingKalmanFilter` on `_simulate_lgssm(rng, n_state, n_obs, 300)`, `.run(obs)` means/covs equal `kalman_filter(...)` outputs under `np.testing.assert_array_equal`; `float(ll) == float(batch_ll)`. Guard: the batch filtered means differ from the prior (`np.ptp > 0`). |
| `test_step_state_matches_batch_prefix` | After `k ∈ {1, 7, 50}` steps, `state.mean` / `state.cov` equal `kalman_filter(obs[:k])[0][-1]` / `[1][-1]` bitwise; `int(state.t) == k`; `f.n_steps == k`. |
| `TestStreamingKalmanIdentities::test_summed_ll_equals_batch` (in `test_likelihood_identities.py`) | Streaming `state.log_likelihood` after all bins of the `lgssm` fixture `== out["f_ll"]` exactly; and `float(run_ll) == out["f_ll"]`. |
| `TestStreamingKalmanIdentities::test_chain_rule_by_reset[split ∈ {1, 5, 11}]` | Run `obs[:split]` (ll_head), `reset(FilterState(f_mean[split-1], f_cov[split-1], split, 0.0))`, run `obs[split:]` (ll_tail): `ll_head + ll_tail == f_ll` to `rtol=1e-10`; tail means equal `out["f_mean"][split:]` to `rtol=1e-12`. |
| `test_reset_restores_prior_and_rerun_is_identical` | After 20 steps `reset()` gives `state == init` (bitwise), `t == 0`, `log_likelihood == 0`, `n_steps == 0`; a second `run(obs)` is bitwise identical to the first. Guard: the first run moved the state. |
| `test_run_continues_from_current_state` | `run(obs[:10])` then `run(obs[10:])`: concatenated outputs equal a single `run(obs)` from reset bitwise; the second call's returned LL equals total − head. |
| `test_float32_inputs_promote_and_warn` | `float32` init with `float64` params: `pytest.warns` is *not* raised (dtype promotes to float64, mirroring test_kalman.py:2746-2760) and `state.mean.dtype == float64`, results equal the float64 filter to `rtol=1e-12`. All-float32 parameters: `pytest.warns(StateSpaceWarning, match="float32")`. |
| `test_constructor_rejects_invalid_inputs[case]` | `ValueError` for: non-PD `init_cov` (`match="positive definite"`), non-PD `R`, non-symmetric `Q`, `H` with the wrong column count, non-finite `A`. |
| `test_step_rejects_wrong_observation_shape` | `step(jnp.zeros(n_obs + 1))` raises `ValueError` and leaves `state`, `n_steps` unchanged. |
| `test_step_compiles_once` | Monkeypatch `streaming.kalman_measurement_update` with a counting wrapper (pattern of test_position_decoder.py:2024-2053), `streaming._kalman_streaming_step.clear_cache()`, 30 steps mixing `np.ndarray` and `jax.Array` inputs → exactly one trace. Guard: at least one trace happened. A second instance with the same shapes adds no trace. |
| `test_mask_matches_reduced_problem` (`skipif(not _SUPPORTS_MASK)`) | With `mask` dropping channels `{1, 3}` of a 5-channel problem: `step(y, mask)` equals, to `rtol=1e-12`, a step of a filter built on `H[keep]`, `R[keep][:, keep]` with `y[keep]` (means, covs and the LL increment). Guard: the masked step differs from the unmasked step. |
| `test_mask_raises_until_supported` (`skipif(_SUPPORTS_MASK)`) | `step(y, mask=...)` raises `NotImplementedError` mentioning `masks-and-multi-sequence`; `step(y)` afterwards still works. |
| `test_mask_must_be_boolean_of_observation_shape` (unconditional) | `step(y, mask=jnp.array([1, 0, 1]))` (integer dtype) raises `ValueError(match="boolean")` and a boolean mask of the wrong length raises `ValueError`, both before any support check; the state is unchanged. |
| `test_robust_weight_forwarded_or_rejected` | `skipif`-paired as above: with support, constructing with `robust_weight=<value the wolf plan documents as "no robustification">` reproduces the default path to `rtol=1e-12`; without, `NotImplementedError` mentioning `wolf-robust-updates`. |
| `test_phase_posterior_matches_monte_carlo_at_high_snr` | Block mean `30·(cos 0.7, sin 0.7)`, block cov `[[1, .3], [.3, .6]]`, 400k `default_rng(0)` samples: `phase_sd` vs `circular_std(atan2 samples)` `rtol=0.03`; `phase` vs `circular_mean` within 0.01 rad; fraction of samples inside `[lower, upper]` (angular distance to `phase` ≤ half-width) in `[0.94, 0.96]`. |
| `test_phase_posterior_error_grows_as_snr_falls` | Relative error of `phase_sd` vs Monte Carlo over amplitude/sd ∈ `{30, 10, 5, 3}` is strictly increasing (`_assert_decreasing` on the negated sequence, test_approximation_trends.py:63-65 pattern) and exceeds 0.1 at 3 (observed 0.001, 0.013, 0.061, 0.197). |
| `test_amplitude_posterior_matches_monte_carlo` | Same high-SNR setup: `amplitude_sd` vs sample std of `hypot` `rtol=0.03`; `amplitude` vs sample mean `rtol=0.01`. |
| `test_oscillator_summaries_at_zero_mean` | Initial/reset state with mean `(0,0)` and isotropic, correlated anisotropic, and zero covariance: no NaNs under eager, `jit` or `vmap`; phase is flagged undefined with SD `inf` and half-width `pi`; amplitude uses the flagged finite conservative bound. For `S = sigma² I`, the upper interval endpoint equals the Rayleigh `level` quantile; for anisotropic S, independent Monte Carlo coverage is at least `level` within sampling error. Repeat after a fully masked update and a zero-valued observation that leaves the mean zero. |
| `test_phase_posterior_selects_block` | `n_state = 4`, block 0 at phase 0.5, block 1 at phase −2.0: `phase_posterior(state, 1) == phase_posterior(state, slice(2, 4))` (all fields bitwise) and `.phase == atan2(mean[3], mean[2])`; differs from block 0. `block=2` raises `ValueError`. |
| `test_phase_interval_wraps_and_caps` | Mean at phase `π − 0.05` with sd 0.1: `upper < lower` numerically (wrapped past π) and both in `(−π, π]`; with block cov `1e6·I`: `upper − lower` spans the circle (half-width == π). |
| `test_streaming_theta_phase_tracks_truth` | Simulate an 8 Hz block at 1 kHz (damping 0.99, `Q = I`, `R = 4`, 2000 bins, seed 0); stream; over bins ≥ 200 the mean resultant length of `phase − true_phase` (`circular_stats.mean_resultant_length`) `> 0.9`, median `phase_sd < 0.3`, and the 95 % interval covers the true phase on a fraction in `[0.85, 1.0]` of bins. |

Slow marking: none of these run EM or a fit; the theta test and Monte Carlo
tests take well under a second each. If the four-SNR Monte Carlo test exceeds
~2 s on CI, mark it `@pytest.mark.slow`.

## Fixtures

- `lgssm_cases` (module scope, `test_streaming.py`): dict of
  `(n_state, n_obs) → _simulate_lgssm(np.random.default_rng(seed), ...)` for the
  four shape pairs, 300 bins, plus the `kalman_filter` reference outputs.
- `oscillator_problem` (module scope): the 8 Hz / 1 kHz block model
  (`A`, `Q`, `H`, `R`, `init_mean = 0`, `init_cov = 50 I`), simulated latent
  states and observations for 2000 bins (seed 0), true phase
  `atan2(x[:, 1], x[:, 0])`.
- Monte Carlo tests build their own deterministic samples inline
  (`np.random.default_rng(0)`); a small helper `_block_state(phase, amplitude,
  block_cov, n_state=2)` returns a `FilterState` with the block placed at rows
  `2k, 2k+1`.
- The existing `lgssm` fixture in `test_likelihood_identities.py` (module
  scope) for the identity tests.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
