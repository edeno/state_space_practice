# Spatial-Value Population Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a model that jointly infers spatial tuning and value representations from a population of neurons that mix both signals, discovering which neurons encode space, value, or both, and tracking how each component drifts over time.

**Architecture:** The latent state has two parts: (1) value features `v_t ∈ R^K` representing the animal's valuation of K options, inferred from choices and spikes, and (2) per-neuron spatial weights `w_{n,t}` on a 2D spline basis, inferred from spikes and known position. Each neuron's firing rate is a GLM combining both: `log(λ_n) = baseline_n + α_n^{space} @ Z(p_t) + α_n^{value} @ v_t`. The model alternates between inferring values (given spatial weights) and updating spatial weights (given values), with choices providing an additional observation on the value state.

**Tech Stack:** JAX, existing `PlaceFieldModel` infrastructure (spline basis, Laplace-EKF), `multinomial_choice` (softmax update), `build_2d_spline_basis`/`evaluate_basis`.

**References:**

- Gauthier, J.L. & Tank, D.W. (2018). A dedicated population for reward coding in the hippocampus. Neuron 99(1), 179-193.
- Aronov, D., Nevers, R. & Tank, D.W. (2017). Mapping of a non-spatial dimension by the hippocampal-entorhinal circuit. Nature 543(7647), 719-722.
- Stachenfeld, K.L., Botvinick, M.M. & Gershman, S.J. (2017). The hippocampus as a predictive map. Nature Neuroscience 20(11), 1643-1653.
- Rigotti, M. et al. (2013). The importance of mixed selectivity in complex cognitive tasks. Nature 497(7451), 585-590.
- Daw, N.D., O'Doherty, J.P., Dayan, P., Seymour, B. & Dolan, R.J. (2006). Cortical substrates for exploratory decisions in humans. Nature 441, 876-879.
- Keeley, S.L., Aoi, M.C. et al. (2020). Identifying signal and noise structure in neural population activity with Gaussian process factor models. NeurIPS.
- Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004). Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering. Neural Computation 16, 971-998.

---

## Background and Mathematical Model

### The scientific question
Hippocampal and prefrontal neurons encode mixtures of spatial position, reward value, task rules, and other variables. Current models analyze each signal in isolation — place fields ignore value coding, value models ignore spatial tuning. This model asks: for each neuron, how much does it encode space vs value? Do these components drift independently? Does value-driven remapping of place cells happen through the spatial component, the value component, or both?

### Generative model

```
Value state (shared across population):
    v_t = v_{t-1} + noise_v,  noise_v ~ N(0, Q_value)
    v_t ∈ R^K   (K = number of choice options / reward sources)

Choice observation:
    c_t ~ Categorical(softmax(β * v_t))   [on trials with choices]

Spatial weights (per neuron n, drifting):
    w_{n,t} = w_{n,t-1} + noise_w,  noise_w ~ N(0, Q_spatial)
    w_{n,t} ∈ R^{n_basis}

Spike observation (per neuron n):
    log(λ_{n,t}) = b_n + w_{n,t} @ Z(p_t) + α_{n}^{value} @ v_t

    Z(p_t): spline basis at position (known from tracking)
    b_n: baseline (learned)
    α_n^{value} ∈ R^K: value loading per neuron (learned)
```

### Neuron types emerge from learned parameters

After fitting, each neuron has:
- `||w_n||`: spatial tuning strength
- `||α_n^{value}||`: value tuning strength
- The ratio classifies: pure place cell, pure value cell, or mixed

### Factored inference

The full state `[v_t, w_{1,t}, ..., w_{N,t}]` is high-dimensional but factored:

1. **Value update:** Given current spatial weights `{w_n}`, each neuron's spatial contribution to firing rate is `w_n @ Z(p_t)` (known). Subtract this from the spike log-likelihood to get the residual that depends on `v_t`. Update `v_t` using these residual spike likelihoods + the choice likelihood.

2. **Spatial weight update:** Given current value state `v_t`, each neuron's value contribution is `α_n @ v_t` (known scalar per neuron). This acts as a time-varying baseline offset. Update each `w_{n,t}` independently using the standard place field smoother with this offset.

3. **Value loading update (M-step):** Given smoothed `v_t` and smoothed `w_{n,t}`, fit `α_n^{value}` per neuron via Poisson regression of spikes on `v_t` (after accounting for spatial component).

---

## Task 1: Combined Spatial-Value Log-Intensity Function

Build the observation model that combines spatial weights and value loadings.

**Files:**
- Create: `src/state_space_practice/spatial_value_model.py`
- Test: `src/state_space_practice/tests/test_spatial_value_model.py`

### Step 1: Write failing test

```python
# tests/test_spatial_value_model.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.spatial_value_model import (
    compute_log_rate_spatial_value,
    NeuronParams,
)


class TestComputeLogRate:
    def test_output_shape(self):
        n_neurons = 5
        n_basis = 10
        n_values = 3

        params = NeuronParams(
            baseline=jnp.zeros(n_neurons),
            spatial_weights=jnp.zeros((n_neurons, n_basis)),
            value_loadings=jnp.zeros((n_neurons, n_values)),
        )
        Z_t = jnp.ones(n_basis)
        v_t = jnp.ones(n_values)

        log_rate = compute_log_rate_spatial_value(params, Z_t, v_t)
        assert log_rate.shape == (n_neurons,)

    def test_spatial_only(self):
        """With zero value loadings, rate depends only on spatial weights."""
        n_neurons = 2
        n_basis = 5
        n_values = 3

        spatial_w = jnp.zeros((n_neurons, n_basis))
        spatial_w = spatial_w.at[0, 2].set(2.0)  # neuron 0 has a field

        params = NeuronParams(
            baseline=jnp.zeros(n_neurons),
            spatial_weights=spatial_w,
            value_loadings=jnp.zeros((n_neurons, n_values)),
        )
        Z_t = jnp.zeros(n_basis).at[2].set(1.0)  # basis 2 active
        v_t = jnp.array([10.0, 5.0, 3.0])  # high values, but zero loadings

        log_rate = compute_log_rate_spatial_value(params, Z_t, v_t)
        assert log_rate[0] == pytest.approx(2.0, abs=1e-6)
        assert log_rate[1] == pytest.approx(0.0, abs=1e-6)

    def test_value_only(self):
        """With zero spatial weights, rate depends only on value loadings."""
        params = NeuronParams(
            baseline=jnp.zeros(2),
            spatial_weights=jnp.zeros((2, 5)),
            value_loadings=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        Z_t = jnp.ones(5)  # doesn't matter
        v_t = jnp.array([3.0, -1.0])

        log_rate = compute_log_rate_spatial_value(params, Z_t, v_t)
        # Neuron 0 loads on value 0 (=3), neuron 1 loads on value 1 (=-1)
        assert log_rate[0] == pytest.approx(3.0, abs=1e-6)
        assert log_rate[1] == pytest.approx(-1.0, abs=1e-6)

    def test_mixed(self):
        """Combined spatial + value contributions."""
        params = NeuronParams(
            baseline=jnp.array([1.0]),
            spatial_weights=jnp.array([[0.5, 0.0]]),
            value_loadings=jnp.array([[0.0, 2.0]]),
        )
        Z_t = jnp.array([1.0, 0.0])
        v_t = jnp.array([0.0, 0.3])

        log_rate = compute_log_rate_spatial_value(params, Z_t, v_t)
        # 1.0 (baseline) + 0.5*1.0 (spatial) + 2.0*0.3 (value) = 2.1
        assert log_rate[0] == pytest.approx(2.1, abs=1e-6)
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_spatial_value_model.py::TestComputeLogRate -v`
Expected: FAIL with ImportError

### Step 3: Implement log-rate function and NeuronParams

```python
# src/state_space_practice/spatial_value_model.py
"""Spatial-value population model.

Jointly infers spatial tuning and value representations from a population
of neurons that mix both signals. Discovers which neurons encode space,
value, or both, and tracks how each component drifts.

The observation model for neuron n at time t:
    log(λ_n) = baseline_n + w_n @ Z(p_t) + α_n @ v_t

where w_n are spatial weights (on 2D spline basis), α_n are value loadings,
Z(p_t) is the spline basis at position, and v_t is the latent value state.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import psd_solve, symmetrize
from state_space_practice.place_field_model import (
    build_2d_spline_basis,
    evaluate_basis,
)
from state_space_practice.utils import check_converged

logger = logging.getLogger(__name__)


@dataclass
class NeuronParams:
    """Per-neuron parameters for the spatial-value model.

    Attributes
    ----------
    baseline : Array, shape (n_neurons,)
        Baseline log firing rate.
    spatial_weights : Array, shape (n_neurons, n_basis)
        Spatial tuning weights on the spline basis.
        For drifting weights, this is the current (time-averaged) estimate.
    value_loadings : Array, shape (n_neurons, n_values)
        Loading of each neuron onto the latent value dimensions.
    """

    baseline: Array
    spatial_weights: Array
    value_loadings: Array

    @property
    def n_neurons(self) -> int:
        return self.baseline.shape[0]

    @property
    def n_basis(self) -> int:
        return self.spatial_weights.shape[1]

    @property
    def n_values(self) -> int:
        return self.value_loadings.shape[1]


def compute_log_rate_spatial_value(
    params: NeuronParams,
    basis_at_position: Array,
    value_state: Array,
) -> Array:
    """Compute log firing rate for all neurons.

    Parameters
    ----------
    params : NeuronParams
    basis_at_position : Array, shape (n_basis,)
        Spline basis Z(p_t) evaluated at current position.
    value_state : Array, shape (n_values,)
        Current latent value state.

    Returns
    -------
    log_rate : Array, shape (n_neurons,)
    """
    spatial_contribution = params.spatial_weights @ basis_at_position
    value_contribution = params.value_loadings @ value_state
    return params.baseline + spatial_contribution + value_contribution
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_spatial_value_model.py::TestComputeLogRate -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/spatial_value_model.py \
        src/state_space_practice/tests/test_spatial_value_model.py
git commit -m "feat: add spatial-value log-rate function and NeuronParams"
```

---

## Task 2: Value State Update from Spikes and Choices

Build the value-state update: given current spatial weights, update `v_t` using the residual spike signal and the choice observation.

**Files:**
- Modify: `src/state_space_practice/spatial_value_model.py`
- Test: `src/state_space_practice/tests/test_spatial_value_model.py`

### Step 1: Write failing test

```python
# Add to tests/test_spatial_value_model.py

from state_space_practice.spatial_value_model import (
    value_update_from_spikes_and_choice,
)


class TestValueUpdate:
    def test_output_shapes(self):
        n_values = 3
        n_neurons = 5
        prior_mean = jnp.zeros(n_values)
        prior_cov = jnp.eye(n_values)

        params = NeuronParams(
            baseline=jnp.zeros(n_neurons),
            spatial_weights=jnp.zeros((n_neurons, 10)),
            value_loadings=jnp.ones((n_neurons, n_values)) * 0.1,
        )
        spikes = jnp.array([0, 1, 0, 0, 2])
        spatial_log_rate = jnp.zeros(n_neurons)  # spatial contribution pre-computed

        post_mean, post_cov, ll = value_update_from_spikes_and_choice(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            spikes=spikes,
            spatial_log_rate=spatial_log_rate,
            value_loadings=params.value_loadings,
            dt=0.004,
            choice=1,
            inverse_temperature=1.0,
        )

        assert post_mean.shape == (n_values,)
        assert post_cov.shape == (n_values, n_values)
        assert jnp.isfinite(ll)

    def test_choice_pulls_value(self):
        """Choosing option 0 should increase v_0 relative to others."""
        n_values = 3
        prior_mean = jnp.zeros(n_values)
        prior_cov = jnp.eye(n_values)

        # No spikes (isolate choice effect)
        spikes = jnp.zeros(5)
        spatial_log_rate = jnp.zeros(5)
        value_loadings = jnp.zeros((5, n_values))

        post_mean, _, _ = value_update_from_spikes_and_choice(
            prior_mean, prior_cov, spikes, spatial_log_rate,
            value_loadings, dt=0.004,
            choice=0, inverse_temperature=2.0,
        )

        assert post_mean[0] > post_mean[1]
        assert post_mean[0] > post_mean[2]

    def test_spike_from_value_cell_updates_value(self):
        """A spike from a neuron that loads on value 1 should increase v_1."""
        n_values = 2
        prior_mean = jnp.zeros(n_values)
        prior_cov = jnp.eye(n_values) * 0.5

        # One neuron that loads strongly on value 1
        value_loadings = jnp.array([[0.0, 2.0]])
        spikes = jnp.array([1])
        spatial_log_rate = jnp.array([0.0])  # no spatial contribution

        post_mean, _, _ = value_update_from_spikes_and_choice(
            prior_mean, prior_cov, spikes, spatial_log_rate,
            value_loadings, dt=0.004,
            choice=None,  # no choice this time step
            inverse_temperature=1.0,
        )

        # Value 1 should increase (spike consistent with high v_1)
        assert post_mean[1] > post_mean[0]

    def test_no_choice_is_handled(self):
        """When choice is None, only spike update should happen."""
        n_values = 2
        prior_mean = jnp.zeros(n_values)
        prior_cov = jnp.eye(n_values)
        spikes = jnp.zeros(3)
        spatial_log_rate = jnp.zeros(3)
        value_loadings = jnp.zeros((3, n_values))

        post_mean, post_cov, _ = value_update_from_spikes_and_choice(
            prior_mean, prior_cov, spikes, spatial_log_rate,
            value_loadings, dt=0.004,
            choice=None,
            inverse_temperature=1.0,
        )

        # With no spikes and no choice, posterior ≈ prior
        np.testing.assert_allclose(post_mean, prior_mean, atol=0.01)
```

### Step 2: Run test to verify it fails

### Step 3: Implement value update

```python
# Add to src/state_space_practice/spatial_value_model.py

from state_space_practice.multinomial_choice import softmax_observation_update


def value_update_from_spikes_and_choice(
    prior_mean: Array,
    prior_cov: Array,
    spikes: Array,
    spatial_log_rate: Array,
    value_loadings: Array,
    dt: float,
    choice: Optional[int] = None,
    inverse_temperature: float = 1.0,
    diagonal_boost: float = 1e-9,
) -> tuple[Array, Array, Array]:
    """Update value state from spike and choice observations.

    Given the spatial contribution to each neuron's rate (pre-computed
    from spatial weights and position), update the value state using:
    1. Poisson spike likelihood (through value loadings)
    2. Categorical choice likelihood (through softmax)

    Parameters
    ----------
    prior_mean : Array, shape (n_values,)
        Prior value state.
    prior_cov : Array, shape (n_values, n_values)
        Prior value covariance.
    spikes : Array, shape (n_neurons,)
        Spike counts.
    spatial_log_rate : Array, shape (n_neurons,)
        Pre-computed spatial contribution: baseline + w_n @ Z(p_t).
    value_loadings : Array, shape (n_neurons, n_values)
        Per-neuron value loading matrix α.
    dt : float
    choice : int or None
        Observed choice (None if no choice at this time step).
    inverse_temperature : float
    diagonal_boost : float

    Returns
    -------
    posterior_mean : Array, shape (n_values,)
    posterior_cov : Array, shape (n_values, n_values)
    log_likelihood : Array (scalar)
    """
    n_values = prior_mean.shape[0]
    n_neurons = spikes.shape[0]
    total_ll = jnp.array(0.0)

    post_mean = prior_mean
    post_cov = prior_cov

    # 1. Spike update: Laplace approximation for Poisson observations
    # log(λ_n) = spatial_log_rate_n + α_n @ v
    # This is LINEAR in v, so the Jacobian is just α (the value loadings)
    log_rates = spatial_log_rate + value_loadings @ post_mean
    cond_intensity = jnp.exp(log_rates) * dt
    innovation = spikes - cond_intensity  # (n_neurons,)

    # Gradient of log-likelihood w.r.t. v: Σ_n (y_n - λ_n*dt) * α_n
    gradient = value_loadings.T @ innovation  # (n_values,)

    # Fisher information: Σ_n λ_n*dt * α_n ⊗ α_n
    fisher = value_loadings.T @ (cond_intensity[:, None] * value_loadings)

    # Prior precision
    identity = jnp.eye(n_values)
    prior_precision = psd_solve(post_cov, identity, diagonal_boost=diagonal_boost)

    # Posterior from spikes
    spike_precision = prior_precision + fisher
    spike_precision = symmetrize(spike_precision)
    eigvals, eigvecs = jnp.linalg.eigh(spike_precision)
    eigvals_safe = jnp.maximum(eigvals, diagonal_boost)
    spike_precision = eigvecs @ jnp.diag(eigvals_safe) @ eigvecs.T

    post_mean = post_mean + psd_solve(
        spike_precision, gradient, diagonal_boost=diagonal_boost
    )
    post_cov = psd_solve(spike_precision, identity, diagonal_boost=diagonal_boost)
    post_cov = symmetrize(post_cov)

    # Spike log-likelihood
    total_ll = total_ll + jnp.sum(
        jax.scipy.stats.poisson.logpmf(spikes, jnp.maximum(cond_intensity, 1e-30))
    )

    # 2. Choice update (if a choice was made)
    if choice is not None:
        post_mean, post_cov, choice_ll = softmax_observation_update(
            post_mean, post_cov, choice, inverse_temperature, diagonal_boost,
        )
        total_ll = total_ll + choice_ll

    return post_mean, post_cov, total_ll
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add value state update from spikes and choices"
```

---

## Task 3: Spatial Weight Update Given Value State

Build the per-neuron spatial weight update: given the current value state, update each neuron's spatial weights with the value contribution treated as a known offset.

**Files:**
- Modify: `src/state_space_practice/spatial_value_model.py`
- Test: `src/state_space_practice/tests/test_spatial_value_model.py`

### Step 1: Write failing test

```python
# Add to tests/test_spatial_value_model.py

from state_space_practice.spatial_value_model import (
    spatial_weight_update,
)


class TestSpatialWeightUpdate:
    def test_output_shapes(self):
        n_basis = 10
        prior_mean = jnp.zeros(n_basis)
        prior_cov = jnp.eye(n_basis) * 0.1

        post_mean, post_cov = spatial_weight_update(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            spike_count=1,
            basis_at_position=jnp.ones(n_basis) * 0.1,
            value_offset=0.5,  # α_n @ v_t + baseline_n
            dt=0.004,
        )

        assert post_mean.shape == (n_basis,)
        assert post_cov.shape == (n_basis, n_basis)

    def test_spike_with_value_offset(self):
        """A spike should update spatial weights even when the value
        component explains some of the firing."""
        n_basis = 5
        Z_t = jnp.zeros(n_basis).at[2].set(1.0)
        prior_mean = jnp.zeros(n_basis)
        prior_cov = jnp.eye(n_basis) * 0.5

        # Value offset = 1.0 (value component predicts some firing)
        post_mean, _ = spatial_weight_update(
            prior_mean, prior_cov,
            spike_count=1, basis_at_position=Z_t,
            value_offset=1.0, dt=0.004,
        )

        # The spatial weight should still update, but less than
        # if value_offset were 0 (because the value already explains
        # some of the spike)
        post_mean_no_value, _ = spatial_weight_update(
            prior_mean, prior_cov,
            spike_count=1, basis_at_position=Z_t,
            value_offset=0.0, dt=0.004,
        )

        # Both should increase w[2], but the one without value offset
        # should increase more (more of the spike is "unexplained")
        assert post_mean[2] > 0
        assert post_mean_no_value[2] > post_mean[2]
```

### Step 2: Run test to verify it fails

### Step 3: Implement spatial weight update

```python
# Add to src/state_space_practice/spatial_value_model.py

def spatial_weight_update(
    prior_mean: Array,
    prior_cov: Array,
    spike_count: int,
    basis_at_position: Array,
    value_offset: float,
    dt: float,
    diagonal_boost: float = 1e-9,
) -> tuple[Array, Array]:
    """Update one neuron's spatial weights given the value contribution.

    The observation model is:
        log(λ) = value_offset + w @ Z(p_t)

    which is LINEAR in w (given position and value). The value_offset
    absorbs the baseline and value contribution (baseline + α @ v_t).

    Parameters
    ----------
    prior_mean : Array, shape (n_basis,)
    prior_cov : Array, shape (n_basis, n_basis)
    spike_count : int
    basis_at_position : Array, shape (n_basis,)
        Spline basis Z(p_t).
    value_offset : float
        Pre-computed value contribution: baseline_n + α_n @ v_t.
    dt : float
    diagonal_boost : float

    Returns
    -------
    posterior_mean : Array, shape (n_basis,)
    posterior_cov : Array, shape (n_basis, n_basis)
    """
    n_basis = prior_mean.shape[0]
    Z = basis_at_position

    # Log-rate and conditional intensity
    log_rate = value_offset + Z @ prior_mean
    cond_intensity = jnp.exp(log_rate) * dt

    # Innovation
    innovation = spike_count - cond_intensity

    # Gradient: (y - λdt) * Z
    gradient = innovation * Z

    # Fisher: λdt * Z ⊗ Z
    fisher = cond_intensity * jnp.outer(Z, Z)

    # Prior precision
    identity = jnp.eye(n_basis)
    prior_precision = psd_solve(prior_cov, identity, diagonal_boost=diagonal_boost)

    # Posterior
    post_precision = prior_precision + fisher
    post_precision = symmetrize(post_precision)

    posterior_mean = prior_mean + psd_solve(
        post_precision, gradient, diagonal_boost=diagonal_boost
    )
    posterior_cov = psd_solve(post_precision, identity, diagonal_boost=diagonal_boost)
    posterior_cov = symmetrize(posterior_cov)

    return posterior_mean, posterior_cov
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add spatial weight update with value offset"
```

---

## Task 4: Value Loading Estimation (M-step)

Estimate each neuron's value loading `α_n` via Poisson regression of spikes on the smoothed value state, after accounting for the spatial component.

**Files:**
- Modify: `src/state_space_practice/spatial_value_model.py`
- Test: `src/state_space_practice/tests/test_spatial_value_model.py`

### Step 1: Write failing test

```python
# Add to tests/test_spatial_value_model.py

from state_space_practice.spatial_value_model import (
    estimate_value_loadings,
)


class TestEstimateValueLoadings:
    def test_recovers_known_loadings(self):
        """Given known spatial weights and value trajectory,
        recover the true value loadings."""
        rng = np.random.default_rng(42)
        n_time = 1000
        n_neurons = 3
        n_basis = 5
        n_values = 2

        # True parameters
        true_alpha = np.array([
            [1.0, 0.0],   # neuron 0: loads on value 0
            [0.0, 1.5],   # neuron 1: loads on value 1
            [0.5, 0.5],   # neuron 2: mixed
        ])
        baselines = np.array([-2.0, -2.0, -2.0])
        spatial_w = rng.normal(0, 0.1, (n_neurons, n_basis))

        # Simulated data
        Z_t = rng.normal(0, 0.5, (n_time, n_basis))
        v_t = np.cumsum(rng.normal(0, 0.1, (n_time, n_values)), axis=0)

        # Generate spikes
        spikes = np.zeros((n_time, n_neurons))
        for n in range(n_neurons):
            log_rate = baselines[n] + Z_t @ spatial_w[n] + v_t @ true_alpha[n]
            rate = np.exp(np.clip(log_rate, -10, 5))
            spikes[:, n] = rng.poisson(rate * 0.004)

        # Estimate loadings
        spatial_log_rates = np.array([
            baselines[n] + Z_t @ spatial_w[n] for n in range(n_neurons)
        ]).T  # (n_time, n_neurons)

        estimated_alpha = estimate_value_loadings(
            spikes=jnp.array(spikes),
            smoothed_values=jnp.array(v_t),
            spatial_log_rates=jnp.array(spatial_log_rates),
            dt=0.004,
        )

        assert estimated_alpha.shape == (n_neurons, n_values)
        # Should roughly match true loadings
        np.testing.assert_allclose(estimated_alpha, true_alpha, atol=0.3)

    def test_output_shape(self):
        n_time, n_neurons, n_values = 100, 4, 3
        estimated = estimate_value_loadings(
            spikes=jnp.zeros((n_time, n_neurons)),
            smoothed_values=jnp.zeros((n_time, n_values)),
            spatial_log_rates=jnp.zeros((n_time, n_neurons)),
            dt=0.004,
        )
        assert estimated.shape == (n_neurons, n_values)
```

### Step 2: Run test to verify it fails

### Step 3: Implement value loading estimation

```python
# Add to src/state_space_practice/spatial_value_model.py

def estimate_value_loadings(
    spikes: Array,
    smoothed_values: Array,
    spatial_log_rates: Array,
    dt: float,
    regularization: float = 0.01,
    max_iter: int = 20,
) -> Array:
    """Estimate per-neuron value loadings via Poisson regression.

    For each neuron, fits:
        log(λ_n) = spatial_log_rate_n + α_n @ v_t

    via iteratively reweighted least squares (Newton's method for
    Poisson GLM), given the spatial component as an offset.

    Parameters
    ----------
    spikes : Array, shape (n_time, n_neurons)
    smoothed_values : Array, shape (n_time, n_values)
    spatial_log_rates : Array, shape (n_time, n_neurons)
        Pre-computed spatial contribution (baseline + w_n @ Z_t).
    dt : float
    regularization : float
        L2 regularization on loadings.
    max_iter : int
        Newton iterations.

    Returns
    -------
    value_loadings : Array, shape (n_neurons, n_values)
    """
    n_time, n_neurons = spikes.shape
    n_values = smoothed_values.shape[1]

    loadings = jnp.zeros((n_neurons, n_values))

    for n in range(n_neurons):
        alpha_n = jnp.zeros(n_values)
        offset = spatial_log_rates[:, n]

        for _ in range(max_iter):
            # Current rate
            log_rate = offset + smoothed_values @ alpha_n
            rate = jnp.exp(jnp.clip(log_rate, -10, 10)) * dt

            # Gradient: Σ_t (y_t - λ_t*dt) * v_t
            residual = spikes[:, n] - rate
            grad = smoothed_values.T @ residual - regularization * alpha_n

            # Hessian: -Σ_t λ_t*dt * v_t ⊗ v_t - regularization * I
            fisher = smoothed_values.T @ (rate[:, None] * smoothed_values)
            hessian = fisher + regularization * jnp.eye(n_values)

            # Newton step
            delta = jnp.linalg.solve(hessian, grad)
            alpha_n = alpha_n + delta

            if jnp.max(jnp.abs(delta)) < 1e-6:
                break

        loadings = loadings.at[n].set(alpha_n)

    return loadings
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add value loading estimation via Poisson regression"
```

---

## Task 5: SpatialValueModel Class

Assemble into a model class with the alternating E-step and full EM.

**Files:**
- Modify: `src/state_space_practice/spatial_value_model.py`
- Test: `src/state_space_practice/tests/test_spatial_value_model.py`

### Step 1: Write failing test

```python
# Add to tests/test_spatial_value_model.py

from state_space_practice.spatial_value_model import SpatialValueModel


class TestSpatialValueModel:
    @pytest.fixture
    def simulated_data(self):
        rng = np.random.default_rng(42)
        n_time = 1000
        n_neurons = 5
        n_values = 3

        position = rng.uniform(0, 100, (n_time, 2))
        spikes = rng.poisson(0.05, (n_time, n_neurons))
        choices = rng.integers(0, n_values, n_time)
        # Only some time steps have choices
        has_choice = rng.random(n_time) < 0.1
        choices_sparse = np.where(has_choice, choices, -1)

        return {
            "position": position,
            "spikes": spikes,
            "choices": choices_sparse,
            "n_time": n_time,
            "n_neurons": n_neurons,
            "n_values": n_values,
        }

    def test_fit(self, simulated_data):
        model = SpatialValueModel(
            dt=0.004,
            n_values=simulated_data["n_values"],
            n_interior_knots=3,
        )
        lls = model.fit(
            position=simulated_data["position"],
            spikes=simulated_data["spikes"],
            choices=simulated_data["choices"],
            max_iter=3,
            verbose=False,
        )
        assert len(lls) == 3

    def test_neuron_classification(self, simulated_data):
        model = SpatialValueModel(
            dt=0.004,
            n_values=simulated_data["n_values"],
            n_interior_knots=3,
        )
        model.fit(
            position=simulated_data["position"],
            spikes=simulated_data["spikes"],
            choices=simulated_data["choices"],
            max_iter=3,
            verbose=False,
        )

        classification = model.classify_neurons()
        assert "spatial_strength" in classification
        assert "value_strength" in classification
        assert "neuron_type" in classification
        assert len(classification["neuron_type"]) == simulated_data["n_neurons"]
        assert all(t in ["place", "value", "mixed", "untuned"]
                   for t in classification["neuron_type"])

    def test_plot(self, simulated_data):
        import matplotlib
        matplotlib.use("Agg")

        model = SpatialValueModel(
            dt=0.004, n_values=simulated_data["n_values"],
            n_interior_knots=3,
        )
        model.fit(
            position=simulated_data["position"],
            spikes=simulated_data["spikes"],
            choices=simulated_data["choices"],
            max_iter=2, verbose=False,
        )
        fig = model.plot_neuron_types()
        assert fig is not None
```

### Step 2: Run test to verify it fails

### Step 3: Implement SpatialValueModel

The class should have:

```python
class SpatialValueModel:
    """Joint spatial-value population model.

    Discovers which neurons encode space, value, or both.

    Parameters
    ----------
    dt : float
    n_values : int
        Number of value dimensions (= number of choice options).
    n_interior_knots : int
    inverse_temperature : float
    q_value : float
        Value state process noise.
    q_spatial : float
        Spatial weight process noise.

    Examples
    --------
    >>> model = SpatialValueModel(dt=0.004, n_values=4)
    >>> model.fit(position, spikes, choices)
    >>> model.classify_neurons()
    >>> model.plot_neuron_types()
    """

    def fit(self, position, spikes, choices, max_iter, verbose):
        """Alternating E-step:
        1. Update value state (from spikes + choices, given spatial weights)
        2. Update spatial weights per neuron (from spikes, given value state)
        M-step:
        3. Update value loadings α_n
        4. Update baselines
        5. Update Q_value, Q_spatial
        """
        ...

    def classify_neurons(self) -> dict:
        """Classify neurons by spatial vs value tuning strength.
        Returns spatial_strength, value_strength, neuron_type per neuron."""
        ...

    def plot_neuron_types(self, ax=None):
        """Scatter plot of spatial vs value strength, colored by type."""
        ...

    def plot_value_trajectory(self, ax=None):
        """Plot inferred value state over time."""
        ...

    def plot_rate_maps(self, neuron_idx, n_grid=50, ax=None):
        """Plot spatial rate map for a single neuron (at mean value)."""
        ...
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add SpatialValueModel with neuron classification"
```

---

## Task 6: Integration Test with Known Neuron Types

Simulate data with known place cells, value cells, and mixed cells, and verify the model classifies them correctly.

**Files:**
- Test: `src/state_space_practice/tests/test_spatial_value_model.py`

### Step 1: Write integration test

```python
class TestSpatialValueRecovery:
    def test_classifies_neuron_types(self):
        """Simulate 3 place cells, 3 value cells, 3 mixed cells.
        Model should classify them correctly."""
        rng = np.random.default_rng(42)
        n_time = 3000
        n_values = 2
        dt = 0.004

        position = rng.uniform(0, 100, (n_time, 2))
        dm, basis_info = build_2d_spline_basis(position, n_interior_knots=3)
        n_basis = basis_info["n_basis"]

        # True values (random walk)
        true_values = np.cumsum(rng.normal(0, 0.05, (n_time, n_values)), axis=0)

        # 9 neurons: 3 place, 3 value, 3 mixed
        n_neurons = 9
        true_spatial = np.zeros((n_neurons, n_basis))
        true_value_load = np.zeros((n_neurons, n_values))

        # Place cells: strong spatial, no value
        true_spatial[:3] = rng.normal(0, 1.0, (3, n_basis))
        # Value cells: no spatial, strong value
        true_value_load[3:6] = rng.normal(0, 1.5, (3, n_values))
        # Mixed: both
        true_spatial[6:] = rng.normal(0, 0.7, (3, n_basis))
        true_value_load[6:] = rng.normal(0, 1.0, (3, n_values))

        baselines = np.full(n_neurons, -3.0)

        # Generate spikes
        spikes = np.zeros((n_time, n_neurons))
        for n in range(n_neurons):
            log_rate = (
                baselines[n] + dm @ true_spatial[n] + true_values @ true_value_load[n]
            )
            rate = np.exp(np.clip(log_rate, -10, 5))
            spikes[:, n] = rng.poisson(rate * dt)

        # Choices from values
        choices = np.full(n_time, -1)
        choice_trials = np.where(rng.random(n_time) < 0.05)[0]
        for t in choice_trials:
            probs = np.exp(true_values[t]) / np.exp(true_values[t]).sum()
            choices[t] = rng.choice(n_values, p=probs)

        # Fit
        model = SpatialValueModel(dt=dt, n_values=n_values, n_interior_knots=3)
        model.fit(position, jnp.array(spikes), choices, max_iter=5, verbose=False)

        classification = model.classify_neurons()
        types = classification["neuron_type"]

        # Place cells (0-2) should be classified as "place" or "mixed"
        assert all(t in ["place", "mixed"] for t in types[:3])
        # Value cells (3-5) should be classified as "value" or "mixed"
        assert all(t in ["value", "mixed"] for t in types[3:6])
```

### Step 2: Run tests, commit

```bash
git commit -m "test: add integration test for neuron type classification recovery"
```
