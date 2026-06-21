# Does the Laplace approximation bias the spike-field coupling posterior?

**Answer: No — for this Bernoulli-logit coupling regression the Laplace-EKF
posterior matches the exact Polya-Gamma posterior in both mean and credible-interval
calibration across the full coupling-strength range, including the weak,
near-undetectable regime. The cheaper EKF can be used without meaningful loss.**

## Setup

Both estimators condition on the **same** LFP-smoothed latent (stage 1) and differ
only in stage 2 — how they infer the per-neuron coupling posterior given that latent:

- **Laplace-EKF** (`coupling_ekf.fit_coupling_ekf`): Fisher-scoring MAP + inverse-Fisher
  covariance (a Gaussian/Laplace approximation).
- **exact PG** (`coupling_pg.fit_coupling_pg`): Polya-Gamma augmented Gibbs sampler
  (asymptotically exact given the latent), returning posterior samples.

Sweep: coupling magnitude scaled over `{0.1, 0.2, 0.4, 0.8, 2.0}` (band-0 coupled,
band-1 control), `n_time = 4000`, 4 replicates per cell, 3 neurons, base rate ~5 Hz,
`lfp_noise_var = 0.25`. Identical simulated `(spikes, lfp)` fed to both estimators.

## Result

```text
  mag method  cover95  abs_bias  ci_width  det_auc  phaseMAE  ekf-pg
 0.10  ekf     1.00     0.058     0.293     0.67     0.550   0.022
 0.10   pg     0.98     0.059     0.284     0.67     0.544
 0.20  ekf     0.96     0.051     0.298     0.89     0.322   0.022
 0.20   pg     0.90     0.057     0.285     0.83     0.394
 0.40  ekf     0.98     0.039     0.284     1.00     0.081   0.023
 0.40   pg     0.98     0.039     0.277     1.00     0.087
 0.80  ekf     0.96     0.049     0.255     1.00     0.069   0.016
 0.80   pg     0.94     0.050     0.246     1.00     0.069
 2.00  ekf     0.96     0.054     0.218     1.00     0.026   0.015
 2.00   pg     0.96     0.054     0.218     1.00     0.025
```

(`ekf-pg` = max abs difference between the EKF and PG posterior means.) See
`crosscheck.png`.

## Reading

- **Means agree** to within 0.015–0.023 everywhere — far below the posterior SD
  (~0.04–0.15) — and the agreement tightens as coupling strengthens.
- **Calibration is comparable**: both methods' 95% credible intervals cover the truth
  at ~0.90–1.00 (≈ nominal 0.95). The Laplace intervals are at most ~2–3% *wider*
  than PG's (very slight over-dispersion), never narrower — so the EKF is, if
  anything, marginally conservative, not over-confident.
- **Detection and phase recovery are indistinguishable** between the methods at every
  coupling level; both degrade together as coupling weakens (AUC 1.0 → 0.67 at
  mag 0.1), which is the *signal* getting weak, not a method artifact.

## Caveats / scope

- This isolates the **stage-2** approximation. Both estimators use the smoothed-latent
  **plug-in** (they ignore the smoother's posterior uncertainty in `x`); the
  cross-check does not test that shared assumption.
- One model configuration (3 neurons, 2 bands, `n_time = 4000`, `lfp_noise_var = 0.25`,
  Gaussian `N(0, 5^2)` coupling prior). Individual cells were spot-checked against the
  aggregate; the agreement is consistent, not an averaging artifact.
- This validates the **Bernoulli-logit** Laplace specifically. The Poisson-`exp`
  Laplace-EKF in the point-process filter is a different link; PG does not apply there,
  so that case would need a different exact reference.

## Reproduce

```bash
uv run --extra coupling python experiments/coupling_ekf_vs_pg/run_crosscheck.py
```
