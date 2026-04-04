# Adaptive Position Decoder with Drifting Tuning Curves Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Build a decoder that simultaneously tracks the animal's position AND the evolution of each neuron's place field, so that decoding remains accurate across hours or days without retraining. When position tracking is available, the model calibrates; when tracking drops out, decoding continues using the self-maintained tuning curves.

**Architecture:** A factored state-space model with two interleaved components: (1) a position decoder (4-dim state: x, y, vx, vy) and (2) per-neuron place field trackers (n_basis-dim weights per neuron). At each time step, the algorithm alternates: decode position given current weight estimates, then update weights given decoded position. When external position is available (e.g., from video tracking), it enters as an additional observation on the position state, anchoring the estimates. The factored structure keeps computation tractable — the position filter is 4-dim regardless of neuron count, and weight updates are independent across neurons.

**Tech Stack:** JAX, existing `_point_process_laplace_update`, `build_2d_spline_basis`/`evaluate_basis` from `place_field_model.py`, `build_position_dynamics` from `position_decoder.py` (Task 1 of position decoding plan).

**Prerequisite Gates:**

- This plan depends on checked-in position-decoder functionality, including `build_position_dynamics` and any decoder-side helper APIs introduced by the position-decoding plan.
- Verify that `place_field_model.py` and any required basis-evaluation utilities exist and match the assumptions in this document before implementation.
- If the position-decoding plan is only partially implemented, stop and complete the missing decoder prerequisites before starting this plan.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_adaptive_decoder.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py src/state_space_practice/tests/test_place_field_model.py -v`
- Lint after each completed task: `conda run -n state_space_practice ruff check src/state_space_practice`
- Before declaring the plan complete, run the targeted tests plus the neighbor regression tests in the same environment and confirm the expected pass/fail transitions for each task.

**Feasibility Status:** PARTIAL (depends on position-decoder completion)

**Codebase Reality Check:**

- Reusable pieces already exist: point-process Laplace updates and place-field basis machinery in `src/state_space_practice/point_process_kalman.py` and `src/state_space_practice/place_field_model.py`.
- Planned new module is required: `src/state_space_practice/adaptive_decoder.py`.
- This plan assumes production-ready decoder APIs from the position-decoding plan (`build_position_dynamics` and decoder-side helpers).

**Claude Code Execution Notes:**

- Hard stop before Task 1: run `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py -v` and proceed only if fully passing.
- Implement in two isolated gates: first position update from spikes, then per-neuron weight update; only then wire the alternating full loop.
- Add a synthetic drift smoke check after integration: verify decode error remains bounded when place fields drift over time.

**MVP Scope Lock (implement now):**

- Implement semi-supervised mode only (position observations available for part of the session).
- Keep drift covariance simple and fixed/diagonal in MVP.
- Support a single alternating schedule without optional heuristics.

**Defer Until Post-MVP:**

- Fully unsupervised long-horizon adaptation as the primary mode.
- Rich process-noise parameterizations and schedule tuning.
- Aggressive online optimization variants.

**References:**

- Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004). Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering. Neural Computation 16, 971-998.
- Orsborn, A.L., Dangi, S., Moorman, H.G. & Bhatt, J.M.C. (2012). Closed-loop decoder adaptation shapes neural plasticity for skillful neuroprosthetic control. Neuron 82(6), 1380-1393.
- Li, Z., O'Doherty, J.E., Hanson, T.L. et al. (2011). Unscented Kalman filter for brain-machine interfaces. PLoS ONE 6(5), e19307.
- Geva, N., Deitch, D., Rubin, A. & Ziv, Y. (2023). Time and experience differentially affect distinct aspects of hippocampal representational drift. Neuron 111(15), 2460-2475.
- Wan, E.A. & Nelson, A.T. (2001). Dual extended Kalman filter methods. In Kalman Filtering and Neural Networks, Wiley.
- Ziv, Y., Burns, L.D., Cocker, E.D. et al. (2013). Long-term dynamics of CA1 hippocampal place codes. Nature Neuroscience 16, 264-266.

---

## Background and Mathematical Model

### Why this is needed
Standard decoders assume place fields are static. Over hours, place fields drift 10-30 cm. A decoder trained at t=0 becomes increasingly inaccurate. Existing solutions require periodic recalibration with known position. This model self-calibrates: when tracking is available, it learns; when tracking drops out, it maintains its own tuning curve estimates and keeps decoding.

### Generative model

```
Position dynamics:
    p_t = A_pos @ p_{t-1} + w_t^{pos},  w_t^{pos} ~ N(0, Q_pos)
    p_t = [x_t, y_t, vx_t, vy_t]

Weight dynamics (per neuron n):
    θ_{n,t} = θ_{n,t-1} + w_t^{drift},  w_t^{drift} ~ N(0, Q_drift)

Spike observation (per neuron n):
    y_{n,t} ~ Poisson(exp(Z(p_t) @ θ_{n,t}) * dt)

    where Z(p_t) is the spline basis evaluated at position p_t

Optional position observation (when tracking available):
    p_t^{obs} ~ N(H @ p_t, R_pos)    # H = [I_2, 0] extracts (x, y)
```

### Factored inference

The full state `[p_t, θ_{1,t}, ..., θ_{N,t}]` is high-dimensional but has exploitable structure:

1. **Position update:** Given current weight estimates `{θ_{n,t-1}}`, the spike likelihood depends on position through `Z(p_t) @ θ_n`. This is a nonlinear function of the 4-dim position state — handled by the Laplace-EKF.

2. **Weight update:** Given decoded position `p_t`, the spline basis `Z(p_t)` is known, and the spike likelihood for each neuron is `log(λ_n) = Z(p_t) @ θ_n` — linear in `θ_n`. Each neuron's weight update is an independent Kalman-like step.

3. **Position observation:** When tracking data is available, it enters as a linear Gaussian observation on `p_t`, fusing with the spike-based position estimate.

The alternating updates are a single-pass algorithm (not iterative EM) — at each time step, you do one position update and one weight update. This makes it suitable for online/streaming use.

### Calibration modes

- **Supervised:** Position tracking available. Position posterior is tight. Weight updates are well-informed.
- **Unsupervised:** No tracking. Position estimated from spikes only. Weight updates use the uncertain decoded position (uncertainty propagates correctly through the Kalman update).
- **Semi-supervised:** Tracking intermittent. Model seamlessly transitions between modes based on whether `p_t^{obs}` is provided.

---

## Task 1: Position Update from Spikes with Fixed Weights

Build the position-update step: given current weight estimates for all neurons, update the position posterior using observed spikes. This is the core decoding step.

**Files:**
- Create: `src/state_space_practice/adaptive_decoder.py`
- Test: `src/state_space_practice/tests/test_adaptive_decoder.py`

### Step 1: Write failing test

```python
# tests/test_adaptive_decoder.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.adaptive_decoder import (
    position_update_from_spikes,
)


class TestPositionUpdateFromSpikes:
    @pytest.fixture
    def setup(self):
        from state_space_practice.place_field_model import build_2d_spline_basis

        n_neurons = 3
        n_interior_knots = 3
        # Build a basis (we need basis_info for evaluation)
        dummy_pos = np.random.default_rng(0).uniform(0, 100, (100, 2))
        _, basis_info = build_2d_spline_basis(dummy_pos, n_interior_knots)
        n_basis = basis_info["n_basis"]

        # Random weights per neuron
        rng = np.random.default_rng(42)
        weights = jnp.array(rng.normal(0, 0.5, (n_neurons, n_basis)))

        return {
            "basis_info": basis_info,
            "weights": weights,
            "n_neurons": n_neurons,
            "n_basis": n_basis,
        }

    def test_output_shapes(self, setup):
        prior_mean = jnp.array([50.0, 50.0, 0.0, 0.0])
        prior_cov = jnp.eye(4) * 100.0
        spikes = jnp.array([0, 1, 0])

        post_mean, post_cov, ll = position_update_from_spikes(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            spikes=spikes,
            neuron_weights=setup["weights"],
            basis_info=setup["basis_info"],
            dt=0.004,
        )

        assert post_mean.shape == (4,)
        assert post_cov.shape == (4, 4)
        assert jnp.isfinite(ll)

    def test_spike_pulls_toward_field(self, setup):
        """A spike from neuron 0 should pull the position estimate
        toward neuron 0's place field center."""
        from state_space_practice.place_field_model import evaluate_basis

        # Find where neuron 0 has highest rate
        grid_x = np.linspace(0, 100, 20)
        grid_y = np.linspace(0, 100, 20)
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        Z_grid = evaluate_basis(grid, setup["basis_info"])
        rates_0 = np.exp(Z_grid @ np.array(setup["weights"][0]))
        peak_idx = np.argmax(rates_0)
        peak_pos = grid[peak_idx]

        # Start far from peak
        start = jnp.array([10.0, 10.0, 0.0, 0.0])
        prior_cov = jnp.eye(4) * 500.0

        # One spike from neuron 0, no spikes from others
        spikes = jnp.array([1, 0, 0])

        post_mean, _, _ = position_update_from_spikes(
            prior_mean=start,
            prior_cov=prior_cov,
            spikes=spikes,
            neuron_weights=setup["weights"],
            basis_info=setup["basis_info"],
            dt=0.004,
        )

        # Posterior should be closer to the peak than the prior
        dist_before = np.linalg.norm(np.array(start[:2]) - peak_pos)
        dist_after = np.linalg.norm(np.array(post_mean[:2]) - peak_pos)
        assert dist_after < dist_before
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_adaptive_decoder.py::TestPositionUpdateFromSpikes -v`
Expected: FAIL with ImportError

### Step 3: Implement position update

```python
# src/state_space_practice/adaptive_decoder.py
"""Adaptive position decoder with drifting tuning curves.

Simultaneously decodes position and tracks place field drift, enabling
continuous decoding across hours or days without retraining. When position
tracking is available, the model calibrates its tuning curve estimates;
when tracking drops out, decoding continues using self-maintained fields.

References
----------
[1] Brown, E.N., Frank, L.M., Tang, D., Quirk, M.C. & Wilson, M.A. (1998).
    A statistical paradigm for neural spike train decoding applied to
    position prediction from ensemble firing patterns of rat hippocampal
    place cells. J Neuroscience 18(18), 7411-7425.
[2] Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004).
    Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering.
    Neural Computation 16, 971-998.
"""

import logging
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import psd_solve, symmetrize
from state_space_practice.place_field_model import evaluate_basis

logger = logging.getLogger(__name__)


def position_update_from_spikes(
    prior_mean: Array,
    prior_cov: Array,
    spikes: Array,
    neuron_weights: Array,
    basis_info: dict,
    dt: float,
    diagonal_boost: float = 1e-9,
) -> tuple[Array, Array, Array]:
    """Update position posterior given spikes and current weight estimates.

    Uses a Laplace (Gaussian) approximation to the Poisson posterior.
    The log-intensity for neuron n is Z(position) @ weights_n, which is
    nonlinear in position (through the spline basis Z).

    Parameters
    ----------
    prior_mean : Array, shape (n_state,)
        Prior position state [x, y, vx, vy] or [x, y].
    prior_cov : Array, shape (n_state, n_state)
        Prior covariance.
    spikes : Array, shape (n_neurons,)
        Spike counts at this time step.
    neuron_weights : Array, shape (n_neurons, n_basis)
        Current weight estimates for each neuron.
    basis_info : dict
        Spline basis specification.
    dt : float
        Time bin width.
    diagonal_boost : float
        Numerical stability term.

    Returns
    -------
    posterior_mean : Array, shape (n_state,)
    posterior_cov : Array, shape (n_state, n_state)
    log_likelihood : Array (scalar)
    """
    n_state = prior_mean.shape[0]
    n_neurons = neuron_weights.shape[0]
    n_pos_dims = 2  # x, y

    # Evaluate basis and its Jacobian at the prior mean position
    pos_2d = prior_mean[:n_pos_dims]

    # Basis at current position: Z(p), shape (n_basis,)
    Z_at_pos = jnp.array(
        evaluate_basis(np.array(pos_2d)[None], basis_info)
    )[0]

    # Log-rates: (n_neurons,)
    log_rates = neuron_weights @ Z_at_pos
    cond_intensity = jnp.exp(log_rates) * dt

    # Jacobian of Z w.r.t. position via finite differences
    eps = 0.5  # cm
    Z_xp = jnp.array(
        evaluate_basis(np.array(pos_2d + jnp.array([eps, 0]))[None], basis_info)
    )[0]
    Z_xm = jnp.array(
        evaluate_basis(np.array(pos_2d + jnp.array([-eps, 0]))[None], basis_info)
    )[0]
    Z_yp = jnp.array(
        evaluate_basis(np.array(pos_2d + jnp.array([0, eps]))[None], basis_info)
    )[0]
    Z_ym = jnp.array(
        evaluate_basis(np.array(pos_2d + jnp.array([0, -eps]))[None], basis_info)
    )[0]

    dZ_dx = (Z_xp - Z_xm) / (2 * eps)  # (n_basis,)
    dZ_dy = (Z_yp - Z_ym) / (2 * eps)  # (n_basis,)

    # Jacobian of log_rate w.r.t. position: d(log_rate_n)/d(x,y)
    # = weights_n @ dZ/d(x,y)
    jac_pos = jnp.column_stack([
        neuron_weights @ dZ_dx,  # (n_neurons,)
        neuron_weights @ dZ_dy,  # (n_neurons,)
    ])  # (n_neurons, 2)

    # Full Jacobian w.r.t. state (pad with zeros for velocity dims)
    if n_state > n_pos_dims:
        jac_full = jnp.concatenate(
            [jac_pos, jnp.zeros((n_neurons, n_state - n_pos_dims))],
            axis=1,
        )  # (n_neurons, n_state)
    else:
        jac_full = jac_pos

    # Innovation
    innovation = spikes - cond_intensity  # (n_neurons,)

    # Gradient of log-posterior w.r.t. state
    likelihood_gradient = jac_full.T @ innovation  # (n_state,)

    # Fisher information (Gauss-Newton Hessian approximation)
    fisher_info = jac_full.T @ (cond_intensity[:, None] * jac_full)  # (n_state, n_state)

    # Prior precision
    identity = jnp.eye(n_state)
    prior_precision = psd_solve(prior_cov, identity, diagonal_boost=diagonal_boost)

    # Posterior precision
    post_precision = prior_precision + fisher_info
    post_precision = symmetrize(post_precision)
    eigvals, eigvecs = jnp.linalg.eigh(post_precision)
    eigvals_safe = jnp.maximum(eigvals, diagonal_boost)
    post_precision = eigvecs @ jnp.diag(eigvals_safe) @ eigvecs.T

    # Newton step
    posterior_mean = prior_mean + psd_solve(
        post_precision, likelihood_gradient, diagonal_boost=diagonal_boost
    )

    # Posterior covariance
    posterior_cov = psd_solve(post_precision, identity, diagonal_boost=diagonal_boost)
    posterior_cov = symmetrize(posterior_cov)

    # Log-likelihood
    log_likelihood = jnp.sum(
        jax.scipy.stats.poisson.logpmf(spikes, jnp.maximum(cond_intensity, 1e-30))
    )

    return posterior_mean, posterior_cov, log_likelihood
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_adaptive_decoder.py::TestPositionUpdateFromSpikes -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/adaptive_decoder.py \
        src/state_space_practice/tests/test_adaptive_decoder.py
git commit -m "feat: add position update from spikes for adaptive decoder"
```

---

## Task 2: Weight Update Given Decoded Position

Build the weight-update step: given decoded position, update each neuron's weight posterior using observed spikes. This is the tuning curve tracking step.

**Files:**
- Modify: `src/state_space_practice/adaptive_decoder.py`
- Test: `src/state_space_practice/tests/test_adaptive_decoder.py`

### Step 1: Write failing test

```python
# Add to tests/test_adaptive_decoder.py

from state_space_practice.adaptive_decoder import weight_update_from_spikes


class TestWeightUpdateFromSpikes:
    @pytest.fixture
    def setup(self):
        from state_space_practice.place_field_model import build_2d_spline_basis

        rng = np.random.default_rng(42)
        dummy_pos = rng.uniform(0, 100, (100, 2))
        _, basis_info = build_2d_spline_basis(dummy_pos, n_interior_knots=3)
        n_basis = basis_info["n_basis"]
        return {"basis_info": basis_info, "n_basis": n_basis}

    def test_output_shapes(self, setup):
        n_basis = setup["n_basis"]
        prior_mean = jnp.zeros(n_basis)
        prior_cov = jnp.eye(n_basis) * 0.1

        post_mean, post_cov = weight_update_from_spikes(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            spike_count=1,
            position=jnp.array([50.0, 50.0]),
            basis_info=setup["basis_info"],
            dt=0.004,
        )

        assert post_mean.shape == (n_basis,)
        assert post_cov.shape == (n_basis, n_basis)

    def test_spike_increases_rate_at_position(self, setup):
        """A spike at position (50, 50) should increase the predicted
        rate at that position."""
        n_basis = setup["n_basis"]
        pos = jnp.array([50.0, 50.0])
        Z_at_pos = jnp.array(
            evaluate_basis(np.array(pos)[None], setup["basis_info"])
        )[0]

        prior_mean = jnp.zeros(n_basis)
        prior_cov = jnp.eye(n_basis) * 0.1

        rate_before = float(jnp.exp(Z_at_pos @ prior_mean))

        post_mean, _ = weight_update_from_spikes(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            spike_count=1,
            position=pos,
            basis_info=setup["basis_info"],
            dt=0.004,
        )

        rate_after = float(jnp.exp(Z_at_pos @ post_mean))
        assert rate_after > rate_before

    def test_no_spike_decreases_rate(self, setup):
        """No spike at a high-rate position should decrease the rate there."""
        n_basis = setup["n_basis"]
        pos = jnp.array([50.0, 50.0])
        Z_at_pos = jnp.array(
            evaluate_basis(np.array(pos)[None], setup["basis_info"])
        )[0]

        # Start with high rate everywhere
        prior_mean = jnp.ones(n_basis) * 2.0
        prior_cov = jnp.eye(n_basis) * 0.1

        rate_before = float(jnp.exp(Z_at_pos @ prior_mean))

        post_mean, _ = weight_update_from_spikes(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            spike_count=0,
            position=pos,
            basis_info=setup["basis_info"],
            dt=0.004,
        )

        rate_after = float(jnp.exp(Z_at_pos @ post_mean))
        assert rate_after < rate_before
```

### Step 2: Run test to verify it fails

### Step 3: Implement weight update

```python
# Add to src/state_space_practice/adaptive_decoder.py

def weight_update_from_spikes(
    prior_mean: Array,
    prior_cov: Array,
    spike_count: int,
    position: Array,
    basis_info: dict,
    dt: float,
    diagonal_boost: float = 1e-9,
) -> tuple[Array, Array]:
    """Update a single neuron's weight posterior given decoded position and spike.

    The observation model is log(λ) = Z(position) @ weights, which is
    LINEAR in weights (given position). So this is an exact Kalman update
    with Poisson observation, using the Laplace approximation.

    Parameters
    ----------
    prior_mean : Array, shape (n_basis,)
        Prior weight mean.
    prior_cov : Array, shape (n_basis, n_basis)
        Prior weight covariance.
    spike_count : int
        Spike count for this neuron at this time step.
    position : Array, shape (2,) or (4,)
        Decoded position [x, y] (or [x, y, vx, vy]).
    basis_info : dict
        Spline basis specification.
    dt : float
        Time bin width.
    diagonal_boost : float
        Numerical stability term.

    Returns
    -------
    posterior_mean : Array, shape (n_basis,)
    posterior_cov : Array, shape (n_basis, n_basis)
    """
    n_basis = prior_mean.shape[0]

    # Evaluate basis at the decoded position
    pos_2d = position[:2]
    Z = jnp.array(
        evaluate_basis(np.array(pos_2d)[None], basis_info)
    )[0]  # (n_basis,)

    # Log-rate and conditional intensity
    log_rate = Z @ prior_mean
    cond_intensity = jnp.exp(log_rate) * dt

    # Innovation
    innovation = spike_count - cond_intensity

    # Jacobian of log-rate w.r.t. weights is just Z (linear model)
    # Gradient of log-likelihood: (spike - λ*dt) * Z
    gradient = innovation * Z  # (n_basis,)

    # Fisher information: λ*dt * Z ⊗ Z
    fisher = cond_intensity * jnp.outer(Z, Z)  # (n_basis, n_basis)

    # Prior precision
    identity = jnp.eye(n_basis)
    prior_precision = psd_solve(prior_cov, identity, diagonal_boost=diagonal_boost)

    # Posterior precision and covariance
    post_precision = prior_precision + fisher
    post_precision = symmetrize(post_precision)

    # Newton step
    posterior_mean = prior_mean + psd_solve(
        post_precision, gradient, diagonal_boost=diagonal_boost
    )

    posterior_cov = psd_solve(post_precision, identity, diagonal_boost=diagonal_boost)
    posterior_cov = symmetrize(posterior_cov)

    return posterior_mean, posterior_cov
```

### Step 4: Run test, commit

```bash
git commit -m "feat: add weight update step for adaptive decoder"
```

---

## Task 3: Position Observation Update (Tracking Fusion)

Build the update step for when external position tracking is available — fuses tracking data with the spike-based position estimate.

**Files:**
- Modify: `src/state_space_practice/adaptive_decoder.py`
- Test: `src/state_space_practice/tests/test_adaptive_decoder.py`

### Step 1: Write failing test

```python
# Add to tests/test_adaptive_decoder.py

from state_space_practice.adaptive_decoder import position_observation_update


class TestPositionObservationUpdate:
    def test_tracking_reduces_uncertainty(self):
        prior_mean = jnp.array([50.0, 50.0, 0.0, 0.0])
        prior_cov = jnp.eye(4) * 100.0  # large uncertainty

        observed_position = jnp.array([55.0, 45.0])
        tracking_noise = 2.0  # cm

        post_mean, post_cov = position_observation_update(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            observed_position=observed_position,
            tracking_noise_std=tracking_noise,
        )

        # Posterior should be closer to observation
        assert jnp.linalg.norm(post_mean[:2] - observed_position) < \
               jnp.linalg.norm(prior_mean[:2] - observed_position)

        # Uncertainty should decrease
        assert jnp.trace(post_cov) < jnp.trace(prior_cov)

    def test_nan_position_is_noop(self):
        """NaN observed position means no tracking — should return prior."""
        prior_mean = jnp.array([50.0, 50.0, 0.0, 0.0])
        prior_cov = jnp.eye(4) * 100.0

        post_mean, post_cov = position_observation_update(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            observed_position=jnp.array([jnp.nan, jnp.nan]),
            tracking_noise_std=2.0,
        )

        np.testing.assert_allclose(post_mean, prior_mean)
        np.testing.assert_allclose(post_cov, prior_cov)
```

### Step 2: Run test to verify it fails

### Step 3: Implement position observation update

```python
# Add to src/state_space_practice/adaptive_decoder.py

def position_observation_update(
    prior_mean: Array,
    prior_cov: Array,
    observed_position: Array,
    tracking_noise_std: float = 2.0,
) -> tuple[Array, Array]:
    """Update position state with external tracking observation.

    Standard Kalman update with linear measurement model:
    y = H @ state + noise, where H extracts (x, y) from state.

    If observed_position contains NaN, the observation is skipped
    (no tracking available at this time step).

    Parameters
    ----------
    prior_mean : Array, shape (n_state,)
    prior_cov : Array, shape (n_state, n_state)
    observed_position : Array, shape (2,)
        Observed (x, y) from tracking. NaN = no observation.
    tracking_noise_std : float
        Standard deviation of tracking noise (cm).

    Returns
    -------
    posterior_mean : Array, shape (n_state,)
    posterior_cov : Array, shape (n_state, n_state)
    """
    has_observation = jnp.all(jnp.isfinite(observed_position))

    n_state = prior_mean.shape[0]
    H = jnp.zeros((2, n_state))
    H = H.at[0, 0].set(1.0)
    H = H.at[1, 1].set(1.0)

    R = jnp.eye(2) * tracking_noise_std ** 2

    # Kalman update
    innovation = observed_position - H @ prior_mean
    S = H @ prior_cov @ H.T + R
    K = prior_cov @ H.T @ jnp.linalg.inv(S)

    post_mean = prior_mean + K @ innovation
    post_cov = prior_cov - K @ S @ K.T
    post_cov = symmetrize(post_cov)

    # If no observation, return prior unchanged
    posterior_mean = jnp.where(has_observation, post_mean, prior_mean)
    posterior_cov = jnp.where(has_observation, post_cov, prior_cov)

    return posterior_mean, posterior_cov
```

### Step 4: Run test, commit

```bash
git commit -m "feat: add position tracking fusion for adaptive decoder"
```

---

## Task 4: AdaptiveDecoder Model Class

Assemble all steps into a model class that runs the full adaptive decoding loop.

**Files:**
- Modify: `src/state_space_practice/adaptive_decoder.py`
- Test: `src/state_space_practice/tests/test_adaptive_decoder.py`

### Step 1: Write failing test

```python
# Add to tests/test_adaptive_decoder.py

from state_space_practice.adaptive_decoder import AdaptiveDecoder


class TestAdaptiveDecoder:
    @pytest.fixture
    def simulation(self):
        """Simulate trajectory + spikes from 5 neurons with drifting fields."""
        rng = np.random.default_rng(42)
        n_time = 2000
        dt = 0.004
        n_neurons = 5

        # Circular trajectory
        t = np.arange(n_time) * dt
        true_x = 50 + 20 * np.cos(2 * np.pi * t / 2.0)
        true_y = 50 + 20 * np.sin(2 * np.pi * t / 2.0)
        position = np.column_stack([true_x, true_y])

        # Place field centers (fixed for this simulation)
        centers = np.array([
            [30, 40], [70, 60], [50, 30], [40, 70], [60, 50],
        ], dtype=float)

        # Generate spikes
        spikes = np.zeros((n_time, n_neurons))
        for n in range(n_neurons):
            dist_sq = np.sum((position - centers[n]) ** 2, axis=1)
            rate = 25 * np.exp(-dist_sq / (2 * 15**2)) + 0.5
            spikes[:, n] = rng.poisson(rate * dt)

        return {
            "position": position,
            "spikes": spikes,
            "dt": dt,
            "n_time": n_time,
            "n_neurons": n_neurons,
        }

    def test_supervised_decoding(self, simulation):
        """With position tracking, decoded position should track well."""
        decoder = AdaptiveDecoder(
            dt=simulation["dt"],
            n_interior_knots=3,
        )

        result = decoder.run(
            spikes=simulation["spikes"],
            observed_position=simulation["position"],  # supervised
        )

        assert result["decoded_position"].shape == (simulation["n_time"], 2)
        # Error should be small with tracking
        error = np.median(np.linalg.norm(
            result["decoded_position"][200:] - simulation["position"][200:],
            axis=1,
        ))
        assert error < 20.0

    def test_unsupervised_decoding(self, simulation):
        """Train on first half (supervised), decode second half (unsupervised)."""
        decoder = AdaptiveDecoder(
            dt=simulation["dt"],
            n_interior_knots=3,
        )

        n_time = simulation["n_time"]
        half = n_time // 2

        # First half: supervised (tracking available)
        # Second half: unsupervised (NaN position)
        observed_position = np.full((n_time, 2), np.nan)
        observed_position[:half] = simulation["position"][:half]

        result = decoder.run(
            spikes=simulation["spikes"],
            observed_position=observed_position,
        )

        # Unsupervised portion should still decode reasonably
        decoded = result["decoded_position"][half + 200:]
        true = simulation["position"][half + 200:]
        error = np.median(np.linalg.norm(decoded - true, axis=1))
        assert error < 40.0  # looser criterion without tracking

    def test_weight_evolution_tracked(self, simulation):
        """Neuron weights should be available after running."""
        decoder = AdaptiveDecoder(
            dt=simulation["dt"],
            n_interior_knots=3,
        )
        result = decoder.run(
            spikes=simulation["spikes"],
            observed_position=simulation["position"],
        )

        assert "neuron_weights" in result
        assert result["neuron_weights"].shape == (
            simulation["n_time"],
            simulation["n_neurons"],
            decoder.n_basis,
        )

    def test_plot_decoding(self, simulation):
        import matplotlib
        matplotlib.use("Agg")

        decoder = AdaptiveDecoder(dt=simulation["dt"], n_interior_knots=3)
        result = decoder.run(
            spikes=simulation["spikes"],
            observed_position=simulation["position"],
        )
        fig = decoder.plot_decoding(
            result, true_position=simulation["position"],
        )
        assert fig is not None
```

### Step 2: Run test to verify it fails

### Step 3: Implement AdaptiveDecoder

```python
# Add to src/state_space_practice/adaptive_decoder.py

from state_space_practice.place_field_model import build_2d_spline_basis
from state_space_practice.position_decoder import build_position_dynamics


class AdaptiveDecoder:
    """Adaptive position decoder with self-calibrating tuning curves.

    Simultaneously decodes position and tracks place field drift.
    When position tracking is available, the model calibrates; when
    tracking drops out, decoding continues using self-maintained fields.

    Parameters
    ----------
    dt : float
        Time bin width in seconds.
    n_interior_knots : int, default=3
        Spline knots per spatial dimension.
    q_pos : float, default=1.0
        Position process noise (cm^2/s).
    q_vel : float, default=10.0
        Velocity process noise (cm^2/s^3).
    q_drift : float, default=1e-5
        Weight drift process noise (per basis function per time step).
    tracking_noise_std : float, default=2.0
        Position tracking noise (cm).

    Examples
    --------
    >>> decoder = AdaptiveDecoder(dt=0.004)
    >>> # Supervised: tracking available
    >>> result = decoder.run(spikes, observed_position=position)
    >>> # Semi-supervised: tracking drops out
    >>> position_with_gaps = position.copy()
    >>> position_with_gaps[1000:] = np.nan  # tracking lost
    >>> result = decoder.run(spikes, observed_position=position_with_gaps)
    """

    def __init__(
        self,
        dt: float,
        n_interior_knots: int = 3,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        q_drift: float = 1e-5,
        tracking_noise_std: float = 2.0,
    ):
        self.dt = dt
        self.n_interior_knots = n_interior_knots
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_drift = q_drift
        self.tracking_noise_std = tracking_noise_std

        self.basis_info: Optional[dict] = None
        self.n_basis: Optional[int] = None

    def run(
        self,
        spikes: np.ndarray,
        observed_position: Optional[np.ndarray] = None,
        arena_range_x: Optional[tuple[float, float]] = None,
        arena_range_y: Optional[tuple[float, float]] = None,
        verbose: bool = True,
    ) -> dict:
        """Run the adaptive decoder.

        Parameters
        ----------
        spikes : np.ndarray, shape (n_time, n_neurons)
            Spike counts per time bin.
        observed_position : np.ndarray or None, shape (n_time, 2)
            Position tracking data. NaN entries = no tracking at that time.
            If None, fully unsupervised (requires arena_range).
        arena_range_x : tuple or None
            (x_min, x_max) for building spline basis. Required if no position.
        arena_range_y : tuple or None
            (y_min, y_max) for building spline basis.
        verbose : bool

        Returns
        -------
        dict with keys:
            decoded_position : (n_time, 2) — decoded x, y
            decoded_cov : (n_time, 2, 2) — position uncertainty
            neuron_weights : (n_time, n_neurons, n_basis) — evolving weights
            log_likelihood : float — total spike log-likelihood
        """
        spikes = np.asarray(spikes)
        if spikes.ndim == 1:
            spikes = spikes[:, None]
        n_time, n_neurons = spikes.shape

        # Build spline basis from position data (or arena range)
        if observed_position is not None:
            valid_mask = np.all(np.isfinite(observed_position), axis=1)
            valid_pos = observed_position[valid_mask]
            if len(valid_pos) < 10:
                raise ValueError("Need at least 10 valid position observations")
        else:
            if arena_range_x is None or arena_range_y is None:
                raise ValueError(
                    "Must provide arena_range_x/y when no position tracking"
                )
            # Create dummy positions spanning the arena
            grid_x = np.linspace(*arena_range_x, 20)
            grid_y = np.linspace(*arena_range_y, 20)
            xx, yy = np.meshgrid(grid_x, grid_y)
            valid_pos = np.column_stack([xx.ravel(), yy.ravel()])

        _, self.basis_info = build_2d_spline_basis(
            valid_pos, n_interior_knots=self.n_interior_knots
        )
        self.n_basis = self.basis_info["n_basis"]

        # Position dynamics
        A_pos, Q_pos = build_position_dynamics(
            self.dt, self.q_pos, self.q_vel, include_velocity=True
        )

        # Initialize position state
        if observed_position is not None and np.any(np.isfinite(observed_position[0])):
            init_pos = jnp.array([
                observed_position[0, 0], observed_position[0, 1], 0.0, 0.0
            ])
        else:
            cx = float(valid_pos[:, 0].mean())
            cy = float(valid_pos[:, 1].mean())
            init_pos = jnp.array([cx, cy, 0.0, 0.0])
        pos_cov = jnp.eye(4) * 100.0

        # Initialize per-neuron weights
        weight_means = [jnp.zeros(self.n_basis) for _ in range(n_neurons)]
        weight_covs = [jnp.eye(self.n_basis) * 1.0 for _ in range(n_neurons)]
        Q_drift = jnp.eye(self.n_basis) * self.q_drift

        # Storage
        decoded_positions = np.zeros((n_time, 2))
        decoded_covs = np.zeros((n_time, 2, 2))
        all_weights = np.zeros((n_time, n_neurons, self.n_basis))
        total_ll = 0.0

        pos_mean = init_pos

        for t in range(n_time):
            # 1. Position prediction
            pos_mean = A_pos @ pos_mean
            pos_cov = A_pos @ pos_cov @ A_pos.T + Q_pos
            pos_cov = symmetrize(pos_cov)

            # 2. Position observation update (if tracking available)
            if observed_position is not None:
                obs_pos = jnp.array(observed_position[t])
                pos_mean, pos_cov = position_observation_update(
                    pos_mean, pos_cov, obs_pos, self.tracking_noise_std,
                )

            # 3. Position update from spikes
            current_weights = jnp.stack([wm for wm in weight_means])
            spike_t = jnp.array(spikes[t])

            pos_mean, pos_cov, ll = position_update_from_spikes(
                prior_mean=pos_mean,
                prior_cov=pos_cov,
                spikes=spike_t,
                neuron_weights=current_weights,
                basis_info=self.basis_info,
                dt=self.dt,
            )
            total_ll += float(ll)

            # 4. Weight prediction (random walk)
            for n in range(n_neurons):
                weight_covs[n] = weight_covs[n] + Q_drift

            # 5. Weight update from spikes (using decoded position)
            for n in range(n_neurons):
                weight_means[n], weight_covs[n] = weight_update_from_spikes(
                    prior_mean=weight_means[n],
                    prior_cov=weight_covs[n],
                    spike_count=int(spikes[t, n]),
                    position=pos_mean,
                    basis_info=self.basis_info,
                    dt=self.dt,
                )

            # Store
            decoded_positions[t] = np.array(pos_mean[:2])
            decoded_covs[t] = np.array(pos_cov[:2, :2])
            for n in range(n_neurons):
                all_weights[t, n] = np.array(weight_means[n])

            if verbose and (t + 1) % 10000 == 0:
                print(f"  t={t+1}/{n_time}, LL={total_ll:.1f}")

        return {
            "decoded_position": decoded_positions,
            "decoded_cov": decoded_covs,
            "neuron_weights": all_weights,
            "log_likelihood": total_ll,
        }

    def plot_decoding(
        self,
        result: dict,
        true_position: Optional[np.ndarray] = None,
        ax=None,
    ):
        """Plot decoded trajectory and error."""
        import matplotlib.pyplot as plt

        decoded = result["decoded_position"]

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        if true_position is not None:
            axes[0].plot(
                true_position[:, 0], true_position[:, 1],
                "k-", alpha=0.3, label="True",
            )
        axes[0].plot(
            decoded[:, 0], decoded[:, 1],
            "r-", alpha=0.7, label="Decoded",
        )
        axes[0].set_xlabel("x (cm)")
        axes[0].set_ylabel("y (cm)")
        axes[0].set_title("Decoded Trajectory")
        axes[0].legend()
        axes[0].set_aspect("equal")

        if true_position is not None:
            error = np.linalg.norm(decoded - true_position, axis=1)
            t = np.arange(len(error)) * self.dt
            axes[1].plot(t, error, "k-", alpha=0.7)
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Position error (cm)")
            axes[1].set_title(f"Median error: {np.median(error[200:]):.1f} cm")

        fig.tight_layout()
        return fig
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add AdaptiveDecoder with supervised/unsupervised/semi-supervised modes"
```

---

## Task 5: Drift Detection and Rate Map Reconstruction

Add methods to detect remapping events and reconstruct evolving rate maps from the tracked weights.

**Files:**
- Modify: `src/state_space_practice/adaptive_decoder.py`
- Test: extend tests

### Step 1: Add methods

```python
    # Add to AdaptiveDecoder class

    def detect_remapping_events(
        self,
        result: dict,
        threshold_cm: float = 10.0,
        n_blocks: int = 50,
    ) -> dict:
        """Detect sudden place field remapping events.

        Looks for blocks where the place field center moves faster
        than expected under smooth drift.

        Parameters
        ----------
        result : dict from run()
        threshold_cm : float
            Minimum center displacement per block to flag as remapping.
        n_blocks : int

        Returns
        -------
        dict with keys:
            remapping_blocks : list[int] — block indices with remapping
            center_displacements : (n_blocks-1,) — displacement per block
            block_centers : (n_blocks, n_neurons, 2) — field centers per block
        """
        ...

    def reconstruct_rate_maps(
        self,
        result: dict,
        n_grid: int = 50,
        time_indices: Optional[list[int]] = None,
    ) -> dict:
        """Reconstruct rate maps at specific time points from tracked weights.

        Parameters
        ----------
        result : dict from run()
        n_grid : int
        time_indices : list[int] or None
            Time indices at which to reconstruct. None = evenly spaced.

        Returns
        -------
        dict with keys:
            rate_maps : (n_timepoints, n_neurons, n_grid, n_grid)
            x_edges, y_edges : grid coordinates
            time_indices : which time points were used
        """
        ...

    def plot_field_evolution(
        self,
        result: dict,
        neuron_idx: int = 0,
        n_snapshots: int = 5,
        n_grid: int = 30,
    ):
        """Plot a neuron's place field at multiple time points."""
        ...
```

### Step 2: Implement, test, commit

```bash
git commit -m "feat: add remapping detection and rate map reconstruction to AdaptiveDecoder"
```
