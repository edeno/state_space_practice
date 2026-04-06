# Structured Nonlinear Oscillator Dynamics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Add a small, interpretable nonlinear oscillator transition model that is compatible with the repository's existing Kalman and Laplace-EKF machinery.

**Architecture:** Start with a discrete-time nonlinear transition in low dimension, with an analytic or autodiff Jacobian available at each prediction step. Use this to support mildly nonlinear oscillatory dynamics without introducing a full neural ODE stack. Keep the first implementation in a separate nonlinear module rather than the existing switching-model hierarchy. If the discrete-time model improves synthetic recovery and held-out spike likelihood, a later plan can consider neural parameterizations or ODE solvers.

**Tech Stack:** JAX, existing oscillator and point-process modules, pytest. No new neural-network dependency in V1.

---

## Why This Revision

The original plan combined several incompatible or premature ideas:

1. A Hamiltonian model conserves energy, while an attracting limit cycle generally requires dissipation.
2. A Neural ODE plus `diffrax` plus point-process Laplace-EKF integration is too large a jump from the current linear transition stack.
3. The first practical need is not a PINN. It is a transition model that can express amplitude-dependent frequency and waveform distortion while remaining linearizable.

This revision replaces the "PINN first" approach with a structured nonlinear oscillator plan that fits the current repository and can be tested incrementally.

## Non-Goals for V1

- No `diffrax`, Equinox, or Flax.
- No Hamiltonian claim.
- No arbitrary MLP transition model.
- No switching-model integration until a single-state nonlinear model is stable.

## Candidate Model Family

Use a Cartesian normal-form oscillator rather than radius-phase coordinates, for example:

$$
x_{t+1} = x_t + dt\left[(\alpha - \beta r_t^2)x_t - \omega_t y_t\right] + \epsilon_x
$$

$$
y_{t+1} = y_t + dt\left[\omega_t x_t + (\alpha - \beta r_t^2)y_t\right] + \epsilon_y
$$

with

$$
r_t^2 = x_t^2 + y_t^2, \quad \omega_t = \omega_0 + \gamma r_t^2.
$$

This is a practical first model because it supports amplitude-dependent frequency and self-limiting amplitude while avoiding the Jacobian singularity introduced by `arctan2` at the origin.

## Success Criteria

1. The nonlinear transition can be evaluated and linearized at every time step.
2. The prediction step remains numerically stable inside the current filter code.
3. On synthetic nonlinear oscillators, the new model outperforms the linear oscillator on state recovery or likelihood.
4. The first point-process integration works for a single-state model before any switching extension is attempted.

### Task 1: Add a Nonlinear Dynamics Utility Module

**Files:**
- Create: `src/state_space_practice/nonlinear_dynamics.py`
- Modify: `src/state_space_practice/__init__.py`
- Test: `src/state_space_practice/tests/test_nonlinear_dynamics.py`

**Step 1: Add parameter and step-function definitions**

Create a small utility module with explicit parameter containers and pure functions such as:

```python
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class NonlinearOscillatorParams:
   growth_rate: float
   saturation: float
   base_frequency: float
   amplitude_frequency_coupling: float


def nonlinear_oscillator_step(
   state: jnp.ndarray,
   params: NonlinearOscillatorParams,
   dt: float,
) -> jnp.ndarray:
   x, y = state
   radius_sq = x * x + y * y
   omega = params.base_frequency + params.amplitude_frequency_coupling * radius_sq
   dx = (params.growth_rate - params.saturation * radius_sq) * x - omega * y
   dy = omega * x + (params.growth_rate - params.saturation * radius_sq) * y
   return state + dt * jnp.array([dx, dy])
```

**Step 2: Add Jacobian helpers**

Implement `linearize_nonlinear_oscillator_step(...)` using `jax.jacfwd` and write a test checking the Jacobian shape and finiteness, including small-amplitude states near the origin.

**Step 3: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_nonlinear_dynamics.py -v`

Expected: PASS for step-function and Jacobian tests.

### Task 2: Add Synthetic Nonlinear Oscillator Data

**Files:**
- Modify: `src/state_space_practice/simulate_data.py`
- Test: `src/state_space_practice/tests/test_nonlinear_dynamics.py`

**Step 1: Add a simulator**

Implement a small helper that simulates latent trajectories from the nonlinear oscillator with Gaussian process noise.

**Step 2: Add observable outputs**

Generate either Gaussian observations or point-process covariates so the model can be evaluated within the existing inference stack.

**Step 3: Write tests**

Add tests verifying that:

- trajectories are finite
- the system approaches a bounded orbit for stable parameter settings
- the simulator is deterministic under a fixed seed

**Step 4: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_nonlinear_dynamics.py -v`

Expected: PASS for simulation tests.

### Task 3: Add an EKF-Compatible Prediction Interface

**Files:**
- Modify: `src/state_space_practice/point_process_kalman.py`
- Possibly create: `src/state_space_practice/nonlinear_filter.py`
- Test: `src/state_space_practice/tests/test_nonlinear_dynamics.py`

**Step 1: Isolate the prediction contract**

Introduce a small prediction API that accepts:

- a transition function `f(x)`
- a linearization function returning the local Jacobian
- process covariance `Q`

Do not refactor the full codebase yet. Keep the interface additive and local.

**Step 2: Implement a single-step nonlinear prediction helper**

The helper should compute:

- `predicted_mean = f(mean_prev)`
- `predicted_cov = A_t @ cov_prev @ A_t.T + Q`

where `A_t` is the Jacobian of the step function at the current expansion point.

**Step 3: Write failing tests first**

Add tests checking that the nonlinear prediction reduces to the linear prediction in a nearly linear parameter regime.

**Step 4: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_nonlinear_dynamics.py -v`

Expected: PASS for nonlinear prediction tests.

### Task 4: Add a Minimal Point-Process Model Wrapper

**Files:**
- Create: `src/state_space_practice/nonlinear_point_process.py`
- Possibly modify: `src/state_space_practice/__init__.py`
- Test: `src/state_space_practice/tests/test_nonlinear_dynamics.py`

**Step 1: Add a narrow wrapper class**

Create a small model class for a single nonlinear oscillator observed with the existing point-process update. Do not place the first implementation in `point_process_models.py`, which is currently organized around switching-model abstractions. Keep the nonlinear experiment isolated in its own module.

**Step 2: Reuse existing observation machinery**

Keep the observation model unchanged. Only replace the linear prediction step.

**Step 3: Add tests**

Verify that fitting or filtering runs end-to-end on a small synthetic problem and returns finite means, covariances, and likelihood values.

**Step 4: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_nonlinear_dynamics.py -v`

Expected: PASS for end-to-end filtering tests.

### Task 5: Compare Against the Existing Linear Oscillator

**Files:**
- Modify: `src/state_space_practice/tests/test_nonlinear_dynamics.py`
- Optional analysis script: `notebooks/correctness_oscillator_models.py`

**Step 1: Build a synthetic benchmark**

Simulate a nonlinear oscillator with amplitude-dependent frequency or mild waveform distortion.

**Step 2: Evaluate both models**

Compare:

- state reconstruction error
- one-step prediction error
- held-out log-likelihood where applicable

**Step 3: Use a pragmatic pass criterion**

The nonlinear model only needs to outperform the linear baseline on the nonlinear synthetic benchmark. It does not need to dominate on every dataset.

## Verification Checklist

- [ ] Step function and Jacobian are finite and deterministic
- [ ] Synthetic nonlinear trajectories remain bounded under stable settings
- [ ] Nonlinear prediction is integrated without destabilizing covariance updates
- [ ] End-to-end point-process filtering works for a single-state nonlinear model
- [ ] Nonlinear model outperforms the linear baseline on at least one controlled synthetic benchmark

## Known Implementation Risks

- A radius-phase parameterization creates derivative pathologies near the origin; stay in Cartesian coordinates for V1.
- The first nonlinear wrapper should not be threaded through the switching hierarchy until the single-state path is stable.
- Success depends more on numerical stability of the Jacobian-based prediction than on representational flexibility.

## Deferred Research Extensions

- Switching nonlinear oscillators
- Learned neural parameterizations of the vector field
- ODE solvers and continuous-time latent dynamics
- More ambitious physics constraints once the target behavior is specified clearly

## Expected Outcome

This plan delivers a practical nonlinear dynamics extension that can represent richer oscillations than the current linear models, while staying compatible with the repository's existing inference architecture.
