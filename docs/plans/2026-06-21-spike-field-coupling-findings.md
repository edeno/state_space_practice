# Spike-Field Coupling Cross-Check: Findings and Impact on Existing Plans

**Date:** 2026-06-21. **Status:** complete; code on `master` (Phases 1–5 of the
EKF-vs-Polya-Gamma coupling cross-check). Experiment write-up:
`experiments/coupling_ekf_vs_pg/conclusion.md`.

This note records the reusable conclusions from that project and how they change
several plans in `docs/plans/`. Other plans should link here rather than restate.

## The identifiability principle (the load-bearing lesson)

**Inferring a latent oscillation *and* its coupling to spikes from spikes alone is
degenerate.** The spike logit is bilinear in the unknowns, `η = β · x`; at the
symmetric point `x = β = 0` its Jacobian is zero, so a deterministic joint
estimator gets no first-order information and cannot bootstrap. Verified
empirically: a single-pass augmented-state EKF recovers garbage (diverges with >1
Newton step), and naive EM (`smooth x | β`, then `regress β | x`) diverges — when
`β` is wrong the smoothed `x` is noise, and the next `β` fits noise.

**Rule for planning:** if a coupling/loading parameter multiplies an **unobserved
latent**, spike-only joint inference is ill-posed. You need either

- an **observed field** that pins the latent (an LFP: `lfp = x + noise`), or
- **observed regressors** (e.g. lagged spike history), in which case the problem
  is a standard GLM and is fine.

This cleanly separates plans that are safe from plans that are not (see table).

## What works: LFP-conditioned two-stage estimation

With an LFP observation the problem decouples and is well-posed:

1. **Smooth the latent from the LFP** — linear-Gaussian Kalman/RTS smoother
   (`coupling_model.smooth_latent_from_lfp`), no spikes, no degeneracy.
2. **Regress coupling from spikes** given the smoothed latent — a Bernoulli-logit
   GLM (`coupling_ekf.fit_coupling_ekf`, or the Gibbs sampler `coupling_pg`).

Recovers coupling to rmse ~0.03 (detection F1 = 1.0, controls nulled).

## Validated: the cheap Laplace ≈ exact Polya-Gamma

Across coupling strengths (0.1–2.0) the Laplace-EKF and exact Polya-Gamma coupling
posteriors agree to ~0.02 in the mean with comparable, near-nominal 95% coverage
(PG intervals ≤2–3% narrower). **For Bernoulli-logit coupling, the Laplace/Fisher
approximation is adequate — do not reach for Polya-Gamma/sampling complexity.**
(Scope: validates stage 2; both share the smoothed-latent plug-in.)

## Reusable infrastructure now in the codebase

- `point_process_kalman.py`: `glm_laplace_update` + `GLMFamily` — a family-generic
  Laplace measurement update (`POISSON_FAMILY` is bit-for-bit with the legacy
  `_point_process_laplace_update`; `BERNOULLI_LOGIT_FAMILY` added), PSD by
  construction.
- `coupling_model.py`: Bernoulli-logit coupling model, simulator (with an LFP
  channel), `interleave_coupling`/`deinterleave_coupling`, `smooth_latent_from_lfp`.
- `coupling_ekf.py` (Laplace), `coupling_pg.py` (Polya-Gamma Gibbs),
  `coupling_validation.py` (Wald χ²(2) / ROC / coverage harness),
  `coupling_crosscheck.py` (the sweep).

Plans that assumed they must build a spike GLM update or a coupling estimator can
reuse these instead.

## Impact on existing plans

| Plan | Impact |
| --- | --- |
| `2026-04-03-cross-region-coupling.md` | **Blocked as specified** — infers per-region latents + cross-region loadings from spikes alone (degenerate). Needs per-region LFP + two-stage. |
| `switching_spike_oscillator_plan.md` | **Blocked as specified** — switching spike-only joint oscillation+loading inference is degenerate; EM monotonicity ≠ identifiability. Make the LFP pass a pipeline stage. |
| `2026-04-08-hamiltonian-oscillator-state-space-model.md` | **Corroborated risk** — its "zero-gradient startup" Known Risk #1 is this degeneracy. Harden acceptance (ground-truth recovery, not just "LL improves"); keep the Gaussian/LFP-observation wrapper as the fallback. |
| `2026-04-07-dynamic-neuron-coupling.md` | **Safe + de-risked** — regressors are observed lagged spikes, not a latent, so no degeneracy; reuse `glm_laplace_update`/`GLMFamily(POISSON)`. |
| `2026-04-05-differentiable-em-optimization.md` | **De-risked** — its point-process EM extension can use the GLM core; Laplace-vs-PG validation answers "is the approximation good enough." Flag the bilinear degeneracy for any joint-latent extension. |
| `2026-04-05-numerical-stability-remediation.md` | **Update target** — name `glm_laplace_update`/`poisson_family` as the consolidation point; record the follow-up to fold the legacy `_point_process_laplace_update` into it. |

## Tracked follow-ups

- Fold the legacy `_point_process_laplace_update` into `glm_laplace_update` (one core).
- Lower-dimensional real-valued LFP projection (current LFP idealizes as observing
  the full latent state).
- In-code PG convergence diagnostic (ESS/R-hat) in `coupling_pg`.
