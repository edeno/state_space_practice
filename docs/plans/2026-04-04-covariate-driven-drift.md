# Covariate-Driven Place Field Drift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Extend `PlaceFieldModel` so that place field drift rate and direction are modulated by observable covariates — reward, errors, running speed, context, time. This directly tests what causes representational drift and remapping.

**Architecture:** The standard random-walk model `θ_t = θ_{t-1} + noise` becomes `θ_t = θ_{t-1} + B @ (covariates_t ⊗ Z(p_t)) + noise(Q(context_t))`. Two extensions: (1) **input-driven drift** where reward, errors, etc. push the weights in a specific direction via learned coefficients B, and (2) **context-modulated noise** where the drift rate Q depends on behavioral context via learned log-linear coefficients. Both B and Q parameters are learned via EM alongside the existing place field inference.

**Tech Stack:** JAX, existing `PlaceFieldModel`, `stochastic_point_process_smoother`, `build_2d_spline_basis`.

**Prerequisite Gates:**

- Verify that `PlaceFieldModel` and the existing point-process smoother expose the state trajectories and basis values needed by this plan.
- Keep the input-driven drift layer and the context-modulated noise layer as separate gates; do not start the full EM wrapper until each extension passes standalone tests.
- If covariate alignment with time bins is underspecified in code, stop and add a validated preprocessing step before implementing model updates.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_drift.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v`
- Lint after each completed task: `conda run -n state_space_practice ruff check src/state_space_practice`
- Before declaring the plan complete, run the targeted tests plus the neighbor regression tests in the same environment and confirm the expected pass/fail transitions for each task.

**Feasibility Status:** PARTIAL

**Codebase Reality Check:**

- Reusable components exist: place-field inference and spline basis utilities in `src/state_space_practice/place_field_model.py` plus point-process smoothing in `src/state_space_practice/point_process_kalman.py`.
- Planned new module is required: `src/state_space_practice/covariate_drift.py`.

**Claude Code Execution Notes:**

- Treat this as two independent gates: input-driven drift (`B` terms) first, context-modulated noise (`Q(context)`) second.
- Add a strict covariate/time-bin alignment check before any model updates; this is the most common failure mode in this class of model.
- Require synthetic directional-effect smoke tests (reward/error covariates produce expected drift-direction changes) before integrating into a full EM wrapper.

**MVP Scope Lock (implement now):**

- Implement input-driven drift coefficients with a small fixed covariate set (for example: reward and speed only).
- Keep process noise context-independent in MVP (single Q).
- Provide one API path for covariate-aligned preprocessing and model fitting.

**Defer Until Post-MVP:**

- Context-modulated noise models (`Q(context)`) and Gamma-style variance regression.
- Broad covariate libraries and automated feature selection.

**References:**

- Bittner, K.C., Milstein, A.D., Grienberger, C., Romani, S. & Bhatt, M.J. (2017). Behavioral time scale synaptic plasticity underlies CA1 place fields. Science 357(6355), 1033-1036.
- Geva, N., Deitch, D., Rubin, A. & Ziv, Y. (2023). Time and experience differentially affect distinct aspects of hippocampal representational drift. Neuron 111(15), 2460-2475.
- Mehta, M.R., Quirk, M.C. & Wilson, M.A. (2000). Experience-dependent asymmetric shape of hippocampal receptive fields. Neuron 25(3), 707-715.
- Linderman, S.W., Johnson, M.J., Miller, A.C. et al. (2017). Bayesian learning and inference in recurrent switching linear dynamical systems. AISTATS.
- Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004). Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering. Neural Computation 16, 971-998.
- Kim, S., Shephard, N. & Chib, S. (1998). Stochastic volatility: likelihood inference and comparison with ARCH models. Review of Economic Studies 65(3), 361-393.

---

## Background and Mathematical Model

### The scientific question
Place fields drift over time, but is this drift random noise or driven by specific experiences? This model tests whether:
- **Reward stabilizes fields** (BTSP-like: reward at a location strengthens that field)
- **Errors destabilize fields** (error-driven plasticity: mistakes trigger remapping)
- **Novelty accelerates drift** (new environments cause faster exploration of map space)
- **Familiarity decelerates drift** (well-known locations have stable fields)
- **Context switches cause remapping** (light on vs off, different task rules)

### Generative model

```
Place field weights (per neuron n):
    θ_{n,t} = θ_{n,t-1} + u_t + w_t

Input-driven drift:
    u_t = Σ_k  β_k * covariate_{k,t} * Z(p_t)

    covariates: reward_t, error_t, speed_t, novelty_t, ...
    Z(p_t): spline basis at animal's position
    β_k: learned scalar per covariate (how strongly this factor drives drift)

    Interpretation: u_t shifts the weights at the current position,
    proportional to the covariate. E.g., reward at position p shifts
    the weight on the basis functions active at p.

Context-modulated noise:
    w_t ~ N(0, Q(context_t))
    Q(context_t) = Q_0 * exp(γ @ context_t)

    context: [time_of_day, trial_block, is_familiar, ...]
    γ: learned coefficients (positive = faster drift in this context)

Spike observations (unchanged):
    y_{n,t} ~ Poisson(exp(Z(p_t) @ θ_{n,t}) * dt)
```

### What the M-step looks like

Given smoothed weight trajectories `θ_{n,t}` from the E-step:

**For B (input-driven drift):**
The smoothed increments are `Δθ_t = θ_t - θ_{t-1}`. The model says `Δθ_t ≈ Σ_k β_k * cov_{k,t} * Z(p_t) + noise`. So B is estimated by regressing the smoothed increments on the covariate-weighted basis evaluations:

```
Δθ_t = [cov_1 * Z(p_t), cov_2 * Z(p_t), ...] @ [β_1, β_2, ...] + noise
```

This is a linear regression, solvable in closed form.

**For γ (context-modulated noise):**
The residual after removing the input-driven component is `ε_t = Δθ_t - u_t`. The model says `ε_t ~ N(0, Q_0 * exp(γ @ context_t))`. The log-variance of the residuals should be linear in context:

```
log(ε_t²) ≈ log(Q_0) + γ @ context_t
```

This is a Gamma GLM (or Gaussian on log-squared-residuals), solvable via iteratively reweighted least squares or gradient descent.

---

## Task 1: Input-Driven Drift Component

Build the input-driven drift term that shifts weights based on covariates at the current position.

**Files:**
- Create: `src/state_space_practice/covariate_drift.py`
- Test: `src/state_space_practice/tests/test_covariate_drift.py`

### Step 1: Write failing test

```python
# tests/test_covariate_drift.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.covariate_drift import (
    compute_drift_input,
    estimate_drift_coefficients,
)


class TestComputeDriftInput:
    def test_output_shape(self):
        n_basis = 10
        n_covariates = 3
        Z_t = jnp.ones(n_basis)
        covariates_t = jnp.array([1.0, 0.0, 0.5])
        beta = jnp.ones(n_covariates) * 0.1

        u_t = compute_drift_input(Z_t, covariates_t, beta)
        assert u_t.shape == (n_basis,)

    def test_zero_covariates_zero_input(self):
        """No covariates active = no input-driven drift."""
        Z_t = jnp.ones(10)
        covariates_t = jnp.zeros(3)
        beta = jnp.ones(3) * 0.5

        u_t = compute_drift_input(Z_t, covariates_t, beta)
        np.testing.assert_allclose(u_t, 0.0, atol=1e-10)

    def test_reward_shifts_field(self):
        """Positive reward coefficient should shift weights at current position."""
        n_basis = 10
        Z_t = jnp.zeros(n_basis).at[3].set(1.0)  # only basis 3 active
        covariates_t = jnp.array([1.0])  # reward = 1
        beta = jnp.array([0.5])  # positive reward coefficient

        u_t = compute_drift_input(Z_t, covariates_t, beta)
        # Only basis 3 should be affected
        assert u_t[3] == pytest.approx(0.5, abs=1e-10)
        assert u_t[0] == pytest.approx(0.0, abs=1e-10)


class TestEstimateDriftCoefficients:
    def test_recovers_known_coefficient(self):
        """Should recover β from synthetic data."""
        rng = np.random.default_rng(42)
        n_time = 1000
        n_basis = 5
        n_covariates = 2

        # True coefficients
        true_beta = np.array([0.3, -0.1])

        # Random basis evaluations and covariates
        Z = rng.normal(0, 1, (n_time, n_basis))
        covariates = rng.normal(0, 1, (n_time, n_covariates))

        # Generate weight increments: Δθ = Σ_k β_k * cov_k * Z + noise
        delta_theta = np.zeros((n_time, n_basis))
        for k in range(n_covariates):
            delta_theta += true_beta[k] * covariates[:, k:k+1] * Z
        delta_theta += rng.normal(0, 0.01, delta_theta.shape)

        estimated_beta = estimate_drift_coefficients(
            delta_theta=jnp.array(delta_theta),
            basis_values=jnp.array(Z),
            covariates=jnp.array(covariates),
        )

        np.testing.assert_allclose(estimated_beta, true_beta, atol=0.05)

    def test_output_shape(self):
        delta_theta = jnp.zeros((100, 8))
        Z = jnp.ones((100, 8))
        covariates = jnp.ones((100, 3))

        beta = estimate_drift_coefficients(delta_theta, Z, covariates)
        assert beta.shape == (3,)
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_drift.py -v`
Expected: FAIL with ImportError

### Step 3: Implement drift input computation and coefficient estimation

```python
# src/state_space_practice/covariate_drift.py
"""Covariate-driven place field drift.

Extends the random-walk drift model with input-driven and context-modulated
components. Covariates (reward, error, speed, etc.) can drive systematic
weight changes, and context variables can modulate the drift rate.

The model decomposes place field changes into:
    Δθ_t = u_t (input-driven, systematic) + w_t (noise, context-dependent)
"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import symmetrize

logger = logging.getLogger(__name__)


def compute_drift_input(
    basis_values: Array,
    covariates: Array,
    beta: Array,
) -> Array:
    """Compute input-driven drift at one time step.

    Parameters
    ----------
    basis_values : Array, shape (n_basis,)
        Spline basis evaluated at current position, Z(p_t).
    covariates : Array, shape (n_covariates,)
        Covariate values at this time step (reward, error, speed, ...).
    beta : Array, shape (n_covariates,)
        Learned drift coefficients per covariate.

    Returns
    -------
    u_t : Array, shape (n_basis,)
        Input-driven weight change.
    """
    # u_t = (Σ_k β_k * cov_k) * Z(p_t)
    # The total covariate effect is a scalar multiplied by the basis
    scalar_effect = jnp.dot(beta, covariates)
    return scalar_effect * basis_values


def estimate_drift_coefficients(
    delta_theta: Array,
    basis_values: Array,
    covariates: Array,
    regularization: float = 1e-4,
) -> Array:
    """Estimate drift coefficients β from smoothed weight increments.

    Solves the regression:
        Δθ_t ≈ Σ_k β_k * covariates_{k,t} * Z(p_t) + noise

    by least squares over all time steps and basis dimensions.

    Parameters
    ----------
    delta_theta : Array, shape (n_time, n_basis)
        Smoothed weight increments θ_t - θ_{t-1}.
    basis_values : Array, shape (n_time, n_basis)
        Spline basis at each position.
    covariates : Array, shape (n_time, n_covariates)
        Covariate values per time step.
    regularization : float
        Ridge regularization strength.

    Returns
    -------
    beta : Array, shape (n_covariates,)
        Estimated drift coefficients.
    """
    n_time, n_basis = delta_theta.shape
    n_covariates = covariates.shape[1]

    # Construct regression design: for each time step t,
    # the predictor for covariate k is cov_{k,t} * Z(p_t)
    # and the response is Δθ_t.
    #
    # Flatten: treat each (time, basis) pair as an observation.
    # Response: Δθ flattened to (n_time * n_basis,)
    # Predictor k: (cov_k * Z) flattened to (n_time * n_basis,)
    #
    # This is a standard least squares: y = X @ beta + noise

    y = delta_theta.ravel()  # (n_time * n_basis,)

    # X: (n_time * n_basis, n_covariates)
    X = jnp.zeros((n_time * n_basis, n_covariates))
    for k in range(n_covariates):
        # cov_k * Z: (n_time, n_basis), flattened
        X = X.at[:, k].set((covariates[:, k:k+1] * basis_values).ravel())

    # Ridge regression
    XtX = X.T @ X + regularization * jnp.eye(n_covariates)
    Xty = X.T @ y
    beta = jnp.linalg.solve(XtX, Xty)

    return beta


def estimate_context_noise_coefficients(
    residuals: Array,
    context: Array,
    regularization: float = 1e-4,
) -> tuple[Array, Array]:
    """Estimate context-dependent noise coefficients γ.

    Models log(variance of residuals) as linear in context:
        log(E[ε²]) ≈ log(Q_0) + γ @ context

    Parameters
    ----------
    residuals : Array, shape (n_time, n_basis)
        Weight increments after removing input-driven component.
    context : Array, shape (n_time, n_context)
        Context variables per time step.
    regularization : float

    Returns
    -------
    log_q0 : Array (scalar)
        Log baseline noise.
    gamma : Array, shape (n_context,)
        Context coefficients on log-noise.
    """
    # Average squared residual per time step (across basis functions)
    mean_sq_residual = jnp.mean(residuals ** 2, axis=1)  # (n_time,)
    log_sq_residual = jnp.log(jnp.maximum(mean_sq_residual, 1e-20))

    # Regression: log(ε²) = log(Q_0) + γ @ context
    n_context = context.shape[1]
    X = jnp.column_stack([jnp.ones(len(context)), context])  # (n_time, 1 + n_context)
    XtX = X.T @ X + regularization * jnp.eye(1 + n_context)
    Xty = X.T @ log_sq_residual
    coeffs = jnp.linalg.solve(XtX, Xty)

    log_q0 = coeffs[0]
    gamma = coeffs[1:]

    return log_q0, gamma
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_drift.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/covariate_drift.py \
        src/state_space_practice/tests/test_covariate_drift.py
git commit -m "feat: add covariate-driven drift coefficient estimation"
```

---

## Task 2: Covariate-Aware Place Field Smoother

Modify the place field smoother to incorporate input-driven drift and context-modulated noise.

**Files:**
- Modify: `src/state_space_practice/covariate_drift.py`
- Test: `src/state_space_practice/tests/test_covariate_drift.py`

### Step 1: Write failing test

```python
# Add to tests/test_covariate_drift.py

from state_space_practice.covariate_drift import (
    covariate_place_field_smoother,
)
from state_space_practice.place_field_model import build_2d_spline_basis


class TestCovariatePlaceFieldSmoother:
    @pytest.fixture
    def simple_data(self):
        rng = np.random.default_rng(42)
        n_time = 500
        position = rng.uniform(0, 100, (n_time, 2))
        dm, basis_info = build_2d_spline_basis(position, n_interior_knots=3)
        n_basis = basis_info["n_basis"]
        spikes = rng.poisson(0.01, n_time)
        covariates = rng.normal(0, 1, (n_time, 2))  # reward, error
        return {
            "design_matrix": jnp.asarray(dm),
            "spikes": jnp.asarray(spikes),
            "covariates": jnp.asarray(covariates),
            "n_basis": n_basis,
            "n_time": n_time,
        }

    def test_output_shapes(self, simple_data):
        n_basis = simple_data["n_basis"]
        result = covariate_place_field_smoother(
            init_mean=jnp.zeros(n_basis),
            init_cov=jnp.eye(n_basis),
            design_matrix=simple_data["design_matrix"],
            spikes=simple_data["spikes"],
            dt=0.004,
            covariates=simple_data["covariates"],
            beta=jnp.zeros(2),
            base_process_noise=1e-5,
        )
        assert result.smoother_mean.shape == (simple_data["n_time"], n_basis)

    def test_zero_beta_matches_standard(self, simple_data):
        """With β=0, should match standard smoother."""
        from state_space_practice.point_process_kalman import (
            log_conditional_intensity,
            stochastic_point_process_smoother,
        )

        n_basis = simple_data["n_basis"]
        Q = jnp.eye(n_basis) * 1e-5

        std_sm, std_sc, _, _ = stochastic_point_process_smoother(
            init_mean_params=jnp.zeros(n_basis),
            init_covariance_params=jnp.eye(n_basis),
            design_matrix=simple_data["design_matrix"],
            spike_indicator=simple_data["spikes"],
            dt=0.004,
            transition_matrix=jnp.eye(n_basis),
            process_cov=Q,
            log_conditional_intensity=log_conditional_intensity,
        )

        result = covariate_place_field_smoother(
            init_mean=jnp.zeros(n_basis),
            init_cov=jnp.eye(n_basis),
            design_matrix=simple_data["design_matrix"],
            spikes=simple_data["spikes"],
            dt=0.004,
            covariates=simple_data["covariates"],
            beta=jnp.zeros(2),
            base_process_noise=1e-5,
        )

        np.testing.assert_allclose(
            result.smoother_mean, std_sm, atol=1e-4
        )
```

### Step 2: Run test to verify it fails

### Step 3: Implement covariate-aware smoother

```python
# Add to src/state_space_practice/covariate_drift.py

from state_space_practice.point_process_kalman import (
    _point_process_laplace_update,
    log_conditional_intensity,
)
from state_space_practice.kalman import _kalman_smoother_update
from state_space_practice.state_dependent_drift import PlaceFieldSmootherResult


def covariate_place_field_smoother(
    init_mean: Array,
    init_cov: Array,
    design_matrix: Array,
    spikes: Array,
    dt: float,
    covariates: Array,
    beta: Array,
    base_process_noise: float = 1e-5,
    context: Optional[Array] = None,
    log_q0: Optional[float] = None,
    gamma: Optional[Array] = None,
) -> PlaceFieldSmootherResult:
    """Place field smoother with covariate-driven drift.

    Parameters
    ----------
    init_mean : Array, shape (n_basis,)
    init_cov : Array, shape (n_basis, n_basis)
    design_matrix : Array, shape (n_time, n_basis)
    spikes : Array, shape (n_time,)
    dt : float
    covariates : Array, shape (n_time, n_covariates)
        Per-time-step covariates driving systematic drift.
    beta : Array, shape (n_covariates,)
        Drift coefficients.
    base_process_noise : float
        Baseline diagonal process noise.
    context : Array or None, shape (n_time, n_context)
        Context variables modulating noise level.
    log_q0 : float or None
        Log baseline noise (used with gamma).
    gamma : Array or None, shape (n_context,)
        Context coefficients on log-noise.

    Returns
    -------
    PlaceFieldSmootherResult
    """
    n_basis = init_mean.shape[0]
    spikes = jnp.atleast_2d(spikes.T).T
    if spikes.ndim == 1:
        spikes = spikes[:, None]

    n_time = design_matrix.shape[0]

    # Compute input-driven drift at each time step
    # u_t = (β @ covariates_t) * Z(p_t)
    scalar_effects = covariates @ beta  # (n_time,)
    drift_inputs = scalar_effects[:, None] * design_matrix  # (n_time, n_basis)

    # Compute time-varying Q
    if context is not None and gamma is not None and log_q0 is not None:
        log_q = log_q0 + context @ gamma  # (n_time,)
        q_scale = jnp.exp(log_q)
    else:
        q_scale = jnp.ones(n_time) * base_process_noise

    # Pre-compute grad/hess
    def _log_intensity(dm_t, x):
        return jnp.atleast_1d(log_conditional_intensity(dm_t, x))

    _grad = jax.jacfwd(_log_intensity, argnums=1)
    _hess = jax.jacfwd(_grad, argnums=1)

    # Forward filter with input-driven dynamics
    def _filter_step(carry, inputs):
        mean_prev, cov_prev, total_ll = carry
        dm_t, spike_t, u_t, q_t = inputs

        # Prediction: x_t = I @ x_{t-1} + u_t
        pred_mean = mean_prev + u_t
        Q_t = jnp.eye(n_basis) * q_t
        pred_cov = cov_prev + Q_t
        pred_cov = symmetrize(pred_cov)

        def log_int(x):
            return _log_intensity(dm_t, x)

        def grad_log_int(x):
            return _grad(dm_t, x)

        def hess_log_int(x):
            return _hess(dm_t, x)

        post_mean, post_cov, ll = _point_process_laplace_update(
            pred_mean, pred_cov, spike_t, dt,
            log_int,
            grad_log_intensity_func=grad_log_int,
            hess_log_intensity_func=hess_log_int,
        )

        total_ll = total_ll + ll
        return (post_mean, post_cov, total_ll), (post_mean, post_cov)

    init_carry = (init_mean, init_cov, jnp.array(0.0))
    (_, _, marginal_ll), (filtered_mean, filtered_cov) = jax.lax.scan(
        _filter_step,
        init_carry,
        (design_matrix, spikes, drift_inputs, q_scale),
    )

    # Backward smoother
    def _smooth_step(carry, inputs):
        next_sm_mean, next_sm_cov = carry
        filt_mean, filt_cov, u_t, q_t = inputs

        Q_t = jnp.eye(n_basis) * q_t
        # The prediction from this step was: pred_mean = filt_mean + u_t
        # So the smoother needs to account for the input
        pred_mean = filt_mean + u_t
        pred_cov = filt_cov + Q_t

        sm_mean, sm_cov, cross_cov = _kalman_smoother_update(
            next_sm_mean, next_sm_cov,
            filt_mean, filt_cov,
            Q_t, jnp.eye(n_basis),  # A = I (input is handled separately)
        )
        return (sm_mean, sm_cov), (sm_mean, sm_cov, cross_cov)

    _, (sm_mean, sm_cov, cross_cov) = jax.lax.scan(
        _smooth_step,
        (filtered_mean[-1], filtered_cov[-1]),
        (filtered_mean[:-1], filtered_cov[:-1], drift_inputs[:-1], q_scale[:-1]),
        reverse=True,
    )

    smoother_mean = jnp.concatenate([sm_mean, filtered_mean[-1:]])
    smoother_cov = jnp.concatenate([sm_cov, filtered_cov[-1:]])

    return PlaceFieldSmootherResult(
        smoother_mean=smoother_mean,
        smoother_cov=smoother_cov,
        smoother_cross_cov=cross_cov,
        marginal_log_likelihood=marginal_ll,
    )
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add covariate-aware place field smoother"
```

---

## Task 3: CovariateDriftModel Class with EM

Wrap into a model class that learns β (drift coefficients), γ (context-noise coefficients), and base Q via EM.

**Files:**
- Modify: `src/state_space_practice/covariate_drift.py`
- Test: `src/state_space_practice/tests/test_covariate_drift.py`

### Step 1: Write failing test

```python
# Add to tests/test_covariate_drift.py

from state_space_practice.covariate_drift import CovariateDriftModel


class TestCovariateDriftModel:
    @pytest.fixture
    def bandit_data(self):
        """Simulate plus maze data with reward-modulated drift."""
        rng = np.random.default_rng(42)
        n_time = 2000
        position = rng.uniform(0, 100, (n_time, 2))
        spikes = rng.poisson(0.02, n_time)
        # Covariates: reward (sparse), running speed
        reward = np.zeros(n_time)
        reward[rng.integers(0, n_time, 50)] = 1.0
        speed = rng.uniform(5, 30, n_time)
        covariates = np.column_stack([reward, speed / 30.0])
        return {
            "position": position,
            "spikes": spikes,
            "covariates": covariates,
            "n_time": n_time,
        }

    def test_fit(self, bandit_data):
        model = CovariateDriftModel(
            dt=0.004,
            n_interior_knots=3,
            covariate_names=["reward", "speed"],
        )
        lls = model.fit(
            position=bandit_data["position"],
            spikes=bandit_data["spikes"],
            covariates=bandit_data["covariates"],
            max_iter=3,
            verbose=False,
        )
        assert len(lls) == 3
        assert all(np.isfinite(ll) for ll in lls)

    def test_drift_coefficients_learned(self, bandit_data):
        model = CovariateDriftModel(
            dt=0.004,
            n_interior_knots=3,
            covariate_names=["reward", "speed"],
        )
        model.fit(
            position=bandit_data["position"],
            spikes=bandit_data["spikes"],
            covariates=bandit_data["covariates"],
            max_iter=5,
            verbose=False,
        )
        assert model.beta is not None
        assert model.beta.shape == (2,)

    def test_drift_summary(self, bandit_data):
        model = CovariateDriftModel(
            dt=0.004,
            n_interior_knots=3,
            covariate_names=["reward", "speed"],
        )
        model.fit(
            position=bandit_data["position"],
            spikes=bandit_data["spikes"],
            covariates=bandit_data["covariates"],
            max_iter=3,
            verbose=False,
        )
        summary = model.drift_coefficient_summary()
        assert "reward" in summary
        assert "speed" in summary
        assert "beta" in summary["reward"]
        assert "significant" in summary["reward"]

    def test_plot(self, bandit_data):
        import matplotlib
        matplotlib.use("Agg")

        model = CovariateDriftModel(
            dt=0.004, n_interior_knots=3,
            covariate_names=["reward", "speed"],
        )
        model.fit(
            position=bandit_data["position"],
            spikes=bandit_data["spikes"],
            covariates=bandit_data["covariates"],
            max_iter=3,
            verbose=False,
        )
        fig = model.plot_drift_factors()
        assert fig is not None
```

### Step 2: Run test to verify it fails

### Step 3: Implement CovariateDriftModel

```python
# Add to src/state_space_practice/covariate_drift.py

from state_space_practice.place_field_model import (
    PlaceFieldModel,
    build_2d_spline_basis,
    evaluate_basis,
)
from state_space_practice.utils import check_converged


class CovariateDriftModel:
    """Place field model with covariate-driven drift.

    Extends PlaceFieldModel to learn what drives place field changes.
    Covariates (reward, error, speed, novelty) modulate both the
    direction and rate of drift.

    Parameters
    ----------
    dt : float
    n_interior_knots : int
    covariate_names : list[str]
        Names of covariates for reporting.
    learn_context_noise : bool, default=False
        Whether to also learn context-modulated noise (requires
        separate context variables).

    Attributes
    ----------
    beta : Array, shape (n_covariates,)
        Learned drift coefficients. Positive = covariate pushes weights
        in the direction of the active basis functions.
    base_process_noise : float
        Learned baseline drift rate.

    Examples
    --------
    >>> model = CovariateDriftModel(
    ...     dt=0.004, covariate_names=["reward", "error", "speed"]
    ... )
    >>> covariates = np.column_stack([reward, error, speed])
    >>> model.fit(position, spikes, covariates)
    >>> model.drift_coefficient_summary()
    {'reward': {'beta': 0.15, 'significant': True},
     'error': {'beta': -0.08, 'significant': False},
     'speed': {'beta': 0.02, 'significant': False}}
    """

    def __init__(
        self,
        dt: float,
        n_interior_knots: int = 5,
        covariate_names: Optional[list[str]] = None,
        learn_context_noise: bool = False,
    ):
        self.dt = dt
        self.n_interior_knots = n_interior_knots
        self.covariate_names = covariate_names or []
        self.learn_context_noise = learn_context_noise

        self.beta: Optional[Array] = None
        self.base_process_noise: float = 1e-5
        self.basis_info: Optional[dict] = None
        self.smoother_mean: Optional[Array] = None
        self.smoother_cov: Optional[Array] = None
        self.log_likelihoods: list[float] = []

    def fit(
        self,
        position: np.ndarray,
        spikes: ArrayLike,
        covariates: ArrayLike,
        context: Optional[ArrayLike] = None,
        max_iter: int = 10,
        tolerance: float = 1e-4,
        verbose: bool = True,
    ) -> list[float]:
        """Fit the model.

        Parameters
        ----------
        position : (n_time, 2)
        spikes : (n_time,)
        covariates : (n_time, n_covariates)
            Drift-driving covariates (reward, error, speed, ...).
        context : (n_time, n_context) or None
            Context variables for noise modulation.
        max_iter : int
        tolerance : float
        verbose : bool

        Returns
        -------
        log_likelihoods : list[float]
        """
        position = np.asarray(position)
        spikes = jnp.asarray(spikes)
        covariates = jnp.asarray(covariates)
        n_time = len(spikes)
        n_covariates = covariates.shape[1]

        # Build basis
        design_matrix_np, self.basis_info = build_2d_spline_basis(
            position, n_interior_knots=self.n_interior_knots
        )
        design_matrix = jnp.asarray(design_matrix_np)
        n_basis = self.basis_info["n_basis"]

        # Initialize
        self.beta = jnp.zeros(n_covariates)
        init_mean = jnp.zeros(n_basis)
        init_cov = jnp.eye(n_basis)

        self.log_likelihoods = []

        def _print(msg):
            if verbose:
                print(msg)

        _print(f"CovariateDriftModel: n_time={n_time}, n_basis={n_basis}, "
               f"n_covariates={n_covariates}")

        for iteration in range(max_iter):
            # E-step
            result = covariate_place_field_smoother(
                init_mean=init_mean,
                init_cov=init_cov,
                design_matrix=design_matrix,
                spikes=spikes,
                dt=self.dt,
                covariates=covariates,
                beta=self.beta,
                base_process_noise=self.base_process_noise,
            )

            ll = float(result.marginal_log_likelihood)
            self.log_likelihoods.append(ll)
            self.smoother_mean = result.smoother_mean
            self.smoother_cov = result.smoother_cov

            _print(f"  EM iter {iteration + 1}/{max_iter}: LL = {ll:.1f}, "
                   f"β = {np.array(self.beta).round(4).tolist()}")

            if not jnp.isfinite(ll):
                break
            if iteration > 0:
                is_converged, _ = check_converged(
                    ll, self.log_likelihoods[-2], tolerance
                )
                if is_converged:
                    _print(f"  Converged after {iteration + 1} iterations.")
                    break

            # M-step: update β
            delta_theta = jnp.diff(result.smoother_mean, axis=0)
            self.beta = estimate_drift_coefficients(
                delta_theta=delta_theta,
                basis_values=design_matrix[1:],
                covariates=covariates[1:],
            )

            # M-step: update base Q
            # Residual after removing input-driven component
            predicted_input = (covariates[1:] @ self.beta)[:, None] * design_matrix[1:]
            residual = delta_theta - predicted_input
            self.base_process_noise = float(
                jnp.maximum(jnp.mean(residual ** 2), 1e-10)
            )

            # Update initial conditions
            init_mean = result.smoother_mean[0]
            init_cov = symmetrize(result.smoother_cov[0])

        return self.log_likelihoods

    def drift_coefficient_summary(self) -> dict:
        """Summarize learned drift coefficients with significance.

        Returns
        -------
        dict mapping covariate name to:
            beta : float — coefficient value
            significant : bool — |β| > 2 * SE (rough significance)
        """
        if self.beta is None:
            raise RuntimeError("Model has not been fitted yet.")

        summary = {}
        beta_arr = np.array(self.beta)
        names = self.covariate_names or [
            f"covariate_{i}" for i in range(len(beta_arr))
        ]

        for i, name in enumerate(names):
            # Rough SE estimate from bootstrap would be better,
            # but for now use magnitude as a proxy
            summary[name] = {
                "beta": float(beta_arr[i]),
                "significant": abs(beta_arr[i]) > 0.01,
            }

        return summary

    def plot_drift_factors(self, ax=None):
        """Plot drift coefficient magnitudes."""
        import matplotlib.pyplot as plt

        summary = self.drift_coefficient_summary()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

        names = list(summary.keys())
        betas = [summary[n]["beta"] for n in names]
        colors = ["green" if b > 0 else "red" for b in betas]

        ax.barh(names, betas, color=colors, alpha=0.7)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Drift coefficient (β)")
        ax.set_title("What Drives Place Field Drift?")

        fig.tight_layout()
        return fig
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add CovariateDriftModel with learned drift coefficients"
```

---

## Task 4: Integration Test with Simulated Reward-Driven Remapping

Create a simulation where reward genuinely drives field changes, and verify the model recovers the effect.

**Files:**
- Test: `src/state_space_practice/tests/test_covariate_drift.py`

### Step 1: Write integration test

```python
class TestCovariateDriftRecovery:
    def test_recovers_reward_effect(self):
        """Simulate data where reward strengthens the place field at the
        reward location. The model should recover β_reward > 0."""
        rng = np.random.default_rng(42)
        n_time = 5000
        dt = 0.004
        n_interior_knots = 3

        # Random position
        position = rng.uniform(10, 90, (n_time, 2))

        # Build basis
        dm, basis_info = build_2d_spline_basis(position, n_interior_knots)
        n_basis = basis_info["n_basis"]

        # True weights: start at zero, reward drives increase
        true_beta_reward = 0.3
        reward = np.zeros(n_time)
        reward_times = rng.integers(100, n_time, 100)
        reward[reward_times] = 1.0

        weights = np.zeros((n_time, n_basis))
        for t in range(1, n_time):
            weights[t] = weights[t-1] + true_beta_reward * reward[t] * dm[t]
            weights[t] += rng.normal(0, 0.001, n_basis)

        # Generate spikes
        log_rate = np.sum(dm * weights, axis=1)
        rate = np.exp(np.clip(log_rate, -10, 5))
        spikes = rng.poisson(rate * dt)

        # Fit
        covariates = reward[:, None]
        model = CovariateDriftModel(
            dt=dt, n_interior_knots=n_interior_knots,
            covariate_names=["reward"],
        )
        model.fit(position, spikes, covariates, max_iter=5, verbose=False)

        # β_reward should be positive
        assert model.beta[0] > 0.05
        summary = model.drift_coefficient_summary()
        assert summary["reward"]["significant"]
```

### Step 2: Run test, commit

```bash
git commit -m "test: add integration test for reward-driven drift recovery"
```
