# Streaming Filters Implementation Plan

**Status:** Not started.

Adds a real-time, per-bin interface to the library's filters: a
`StreamingKalmanFilter` (e.g. theta phase from an oscillator state-space model,
one LFP sample at a time, with a delta-method circular credible interval), a
`StreamingPointProcessFilter` for spike counts, and a `StreamingPositionDecoder`
that wraps the place-field decoder for real-time position / replay decoding.
Each holds an explicit `FilterState`, runs one pre-compiled `jax.jit` step per
bin, reproduces the batch filters bit-for-bit, offers fixed-lag smoothing over a
bounded ring buffer, and ships with a measured per-step latency table.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals / non-goals, optional cross-plan dependencies, expected latency table, risks, open questions
- [shared-contracts.md](shared-contracts.md) — `FilterState`, the streaming-filter surface, the parity policy, keyword forwarding to the parallel plans, the prediction hand-off for fixed-lag smoothing
- [designs.md](designs.md) — module layout and compiled-step pattern, Gaussian / point-process / decoder steps, phase posterior math, ring buffer, decoder extraction, benchmark script
- Phases (each ships as a separable PR):
  - [phase-1-streaming-kalman.md](phase-1-streaming-kalman.md) — `streaming.py` with `FilterState` + `StreamingKalmanFilter` (bit-identical to `kalman_filter`), oscillator phase / amplitude posteriors, latency benchmark, docs
  - [phase-2-fixed-lag-smoother.md](phase-2-fixed-lag-smoother.md) — ring buffer + `.smoothed(lag)` for the Gaussian filter, approximation-trend test against the batch smoother
  - [phase-3a-streaming-point-process.md](phase-3a-streaming-point-process.md) — `StreamingPointProcessFilter` on `glm_laplace_update`, bit-identical to the dense `stochastic_point_process_filter`, fixed-lag reuse
  - [phase-3b-streaming-position-decoder.md](phase-3b-streaming-position-decoder.md) — extract the decoder's per-bin step from `_run_filter_scan` (bitwise-preserving), then `StreamingPositionDecoder` with parity against `position_decoder_filter` and fixed-lag smoothing
