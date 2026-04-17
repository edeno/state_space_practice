# Hamiltonian Model Architecture

## Status

The `hamiltonian_*` family of modules is **intentionally standalone** — it
does not integrate with the linear-Gaussian `BaseModel` EM machinery in
[models.py](../src/state_space_practice/models.py) or the switching
abstractions in [point_process_models.py](../src/state_space_practice/point_process_models.py).

Modules:

- [hamiltonian_spikes.py](../src/state_space_practice/hamiltonian_spikes.py) — point-process observation
- [hamiltonian_lfp.py](../src/state_space_practice/hamiltonian_lfp.py) — Gaussian (LFP) observation
- [hamiltonian_joint.py](../src/state_space_practice/hamiltonian_joint.py) — joint LFP + spike observation
- [hamiltonian_switching.py](../src/state_space_practice/hamiltonian_switching.py) — switching variant with per-state Hamiltonians
- [nonlinear_dynamics.py](../src/state_space_practice/nonlinear_dynamics.py) — shared leapfrog / EKF primitives

## Why standalone

1. **No closed-form M-step.** The Hamiltonian parameters (masses, frequencies,
   coupling weights, MLP residual weights) enter the transition density
   nonlinearly. There is no closed-form EM update analogous to the
   linear-Gaussian A/Q updates. Any EM wrapper would delegate to SGD anyway,
   so direct SGD via `SGDFittableMixin` is both simpler and more honest.

2. **Different prediction semantics.** `BaseModel` assumes `x_t = A x_{t-1} + w`.
   The Hamiltonian models use a symplectic leapfrog step `x_t = f_θ(x_{t-1})`
   plus Gaussian process noise, with covariance propagated via the local
   Jacobian `A_t = ∂f_θ/∂x`. Shoehorning this into the linear interface would
   require either lying about `A` (storing a stale Jacobian) or leaking
   per-step linearization details into the base class.

3. **Shared primitives, not shared inheritance.** Where reuse makes sense,
   the Hamiltonian modules import directly:
   - `SGDFittableMixin` for the optimizer loop and parameter-transform plumbing
   - `point_process_kalman.{_logdet_psd, psd_solve, symmetrize}` and
     `kalman.joseph_form_update` for numerics
   - `nonlinear_dynamics.{leapfrog_step, ekf_predict_step_with_jacobian,
     ekf_smooth_step}` for the nonlinear prediction/smoothing primitives

## Consequences

- Each Hamiltonian model implements its own `filter` / `smooth` /
  `_sgd_loss_fn` / `_finalize_sgd`. Recent fixes
  ([2eeb62d](../#), [32b740b](../#)) added `_finalize_sgd` overrides because
  the switching variant returns a 4-tuple from `filter` while the
  single-regime variants return a 3-tuple — that divergence is expected, not
  a defect.
- Fitting is **SGD only**. Calling `.fit(...)` (EM) raises `NotImplementedError`
  by design; use `.fit_sgd(...)`.
- If a future Hamiltonian variant needs to participate in the linear-Gaussian
  EM pipeline (e.g., to share M-step code with a non-Hamiltonian peer), the
  right move is to extract shared helpers — not to force Hamiltonian models
  under `BaseModel`.

## See also

- Original plan: [docs/plans/2026-04-08-hamiltonian-oscillator-state-space-model.md](plans/2026-04-08-hamiltonian-oscillator-state-space-model.md)
- Review fixes: [docs/plans/2026-04-08-hamiltonian-review-fixes.md](plans/2026-04-08-hamiltonian-review-fixes.md)
