# Hamiltonian Model Architecture

## Status

The `hamiltonian_*` family of modules is **fit by SGD only** and does not
integrate with the linear-Gaussian EM machinery: it does not inherit the EM
layer `BaseModel` in
[oscillator_models.py](../src/state_space_practice/oscillator_models.py) or
the switching abstractions in
[point_process_models.py](../src/state_space_practice/point_process_models.py).
It does share the parameter containers below that layer: every model derives
from `HamiltonianModelBase(OscillatorParameterBase, SGDFittableMixin)`.

Modules:

- [hamiltonian_core.py](../src/state_space_practice/hamiltonian_core.py) — per-step EKF / Laplace-EKF helpers, the jitted single-regime filter/smoother cores, and `HamiltonianModelBase`
- [hamiltonian_spikes.py](../src/state_space_practice/hamiltonian_spikes.py) — point-process observation
- [hamiltonian_lfp.py](../src/state_space_practice/hamiltonian_lfp.py) — Gaussian (LFP) observation
- [hamiltonian_joint.py](../src/state_space_practice/hamiltonian_joint.py) — joint LFP + spike observation
- [hamiltonian_switching.py](../src/state_space_practice/hamiltonian_switching.py) — switching variant with per-state Hamiltonians
- [nonlinear_dynamics.py](../src/state_space_practice/nonlinear_dynamics.py) — shared leapfrog / EKF primitives

## Why no EM layer

1. **No closed-form M-step.** The Hamiltonian parameters (masses, frequencies,
   coupling weights, MLP residual weights) enter the transition density
   nonlinearly. There is no closed-form EM update analogous to the
   linear-Gaussian A/Q updates. Any EM wrapper would delegate to SGD anyway,
   so direct SGD via `SGDFittableMixin` is both simpler and more honest.

2. **Different prediction semantics.** The EM layer assumes `x_t = A x_{t-1} + w`.
   The Hamiltonian models use a symplectic leapfrog step `x_t = f_θ(x_{t-1})`
   plus Gaussian process noise, with covariance propagated via the local
   Jacobian `A_t = ∂f_θ/∂x`. Shoehorning this into the linear interface would
   require either lying about `A` (storing a stale Jacobian) or leaking
   per-step linearization details into the base class.

3. **Shared parameter base and primitives, not the EM layer.** The family
   inherits:
   - `OscillatorParameterBase` for the parameter containers, dimensions,
     `decode` / `predict_proba`, and the SGD storage helpers
   - `SGDFittableMixin` for the optimizer loop and parameter-transform plumbing

   and imports directly:
   - `utils.{psd_cholesky, psd_logdet, psd_solve, symmetrize}` and
     `kalman.joseph_form_update` for numerics
   - `point_process_kalman.{glm_laplace_update, poisson_family}` for the
     Laplace point-process update
   - `nonlinear_dynamics.{leapfrog_step, ekf_predict_step_with_jacobian,
     ekf_smooth_step}` for the nonlinear prediction/smoothing primitives
   - `switching_kalman.collapse_gaussian_mixture` for the switching collapse

## Division of work

- `HamiltonianModelBase` implements the SGD hooks shared by the family:
  `_prepare_sgd_data` (validation, sequence length), `_store_sgd_params`
  (mapping-driven via `_sgd_param_attrs`, plus the single-regime `init_mean`
  slot and `Q`), `_finalize_sgd` (filter + smoother into the fitted
  attributes), and `_filter_jit` / `_smooth_jit`, which dispatch to the
  module-level jitted cores.
- Each concrete model implements its observation head: the validating
  `filter` / `smooth` wrappers, a thin public `fit_sgd` with its own named
  data arguments, `_validate_fit_data`, `_build_param_spec`, `_sgd_loss_fn`,
  and a `_store_sgd_params` override that resyncs its observation containers.
- The switching model additionally overrides `_filter_jit` / `_smooth_jit`
  (its own jitted Kim filter and smoother) and is the only model that
  overrides `_finalize_sgd`: its filter returns
  `(means, covs, discrete_probs, marginal_lls)` and its smoother
  `(means, covs, discrete_probs)`, versus 3- and 2-tuples for the
  single-regime models.

## Consequences

- Fitting is **SGD only**. The Hamiltonian models do not inherit the EM
  layer, so there is no `.fit(...)` method (calling it is an `AttributeError`);
  use `.fit_sgd(...)`. The attributes the EM layer's constructor set are
  absent too: the `update_*` flags, `discrete_transition_diag`,
  `transition_prior` and `smoother_type`.
- `decode()` / `predict_proba()` are only defined for the switching model; the
  single-regime models raise `NotImplementedError` because they have no
  discrete-state posterior.
- If a future Hamiltonian variant needs to participate in the linear-Gaussian
  EM pipeline (e.g., to share M-step code with a non-Hamiltonian peer), the
  right move is to extract shared helpers — not to force Hamiltonian models
  under the EM `BaseModel`.

## See also

- Original plan: [docs/plans/2026-04-08-hamiltonian-oscillator-state-space-model.md](plans/2026-04-08-hamiltonian-oscillator-state-space-model.md)
- Review fixes: [docs/plans/2026-04-08-hamiltonian-review-fixes.md](plans/2026-04-08-hamiltonian-review-fixes.md)
