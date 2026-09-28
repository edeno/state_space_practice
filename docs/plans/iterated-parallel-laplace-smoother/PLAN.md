# Iterated and Parallel Laplace Smoother Implementation Plan

**Status:** Not started.

Adds an *iterated* Laplace smoother for point-process (Poisson GLM) observations to
`stochastic_point_process_smoother` and the models built on it (`PointProcessModel`,
`PlaceFieldModel`, `PositionDecoder`): after today's one-pass Laplace-EKF filter + RTS
smoother, the Poisson likelihood is re-linearised at the *smoothed* trajectory and a
linear-Gaussian smoother is re-run with Armijo damping until the joint MAP path converges
(Gauss–Newton = Newton for the canonical log link). Covariances and the marginal
log-likelihood become the Laplace approximation at the joint mode, which shrinks the
approximation gap the oracle tests pin today. A second option runs every pass through
associative scans (a parallel information-form Kalman filter plus the existing parallel RTS
smoother) so each pass has O(log T) span on GPU. Everything is opt-in through keyword-only
arguments (`n_iterations`, `parallel`, ...); the defaults reproduce today's outputs
bit-for-bit. A final phase retires the private Newton loop in `temporal_rate_gp.py` in
favour of the shared core.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points (file:line), goals/non-goals, dependency policy toward the sibling plans, metrics, risks, rollout, open questions, effort.
- [shared-contracts.md](shared-contracts.md) — the public keywords and return layout, `IteratedSmootherDiagnostics`, the pseudo-observation site contract, the Gauss–Newton core and linear-pass interfaces, the warning contract.
- [designs.md](designs.md) — full derivation (pseudo-observations in information form, Gauss–Newton step as a linear smoother pass, line search, Laplace evidence, parallel filter elements, decoder penalty sites, gradients) with complete code for the non-obvious parts.
- Phases (each ships as a separable PR):
  - [phase-1-sequential-iterated-smoother.md](phase-1-sequential-iterated-smoother.md) — `n_iterations` on `stochastic_point_process_smoother` (dense and block-diagonal paths), threaded through `PointProcessModel` and `PlaceFieldModel`; oracle, monotone-ascent, temporal-rate-GP agreement, block-parity and bit-identical tests; docs.
  - [phase-2-parallel-in-time.md](phase-2-parallel-in-time.md) — `parallel=True`: information-form parallel Kalman filter (`kalman.parallel_kalman_filter`) + `parallel_kalman_smoother` for every pass; baseline capture before, output comparison and speedup table after.
  - [phase-3-position-decoder.md](phase-3-position-decoder.md) — `n_iterations` / `parallel` on `position_decoder_smoother` and `PositionDecoder`, with the track penalty as a Gauss–Newton pseudo-observation and adaptive inflation confined to the initialisation.
  - [phase-4-consolidate-temporal-rate-gp.md](phase-4-consolidate-temporal-rate-gp.md) — `temporal_rate_gp._infer_log_rate_traced` runs on the shared core; its private Newton loop is removed; dense-GP oracle agreement preserved.
