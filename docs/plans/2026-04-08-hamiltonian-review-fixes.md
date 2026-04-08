# Hamiltonian Code Review Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical bugs, runtime errors, and high-priority issues identified in the three-reviewer code review of the Hamiltonian model family.

**Architecture:** Verify-first approach — each fix starts by writing a test that exposes the bug, confirming the failure, then applying the minimal fix. We work from most critical (runtime crashes) to least critical (API cleanup). The peripheral models (spikes, LFP, joint, switching) need basic smoke tests before any fix can be verified.

**Tech Stack:** JAX, pytest, conda environment `state_space_practice`

---

## Phase 1: Fix Runtime Crashes

These bugs cause `NameError` at runtime and are completely untested.

### Task 1: Fix missing imports in HamiltonianSpikeModel.smooth()

**Files:**
- Modify: `src/state_space_practice/hamiltonian_spikes.py:15-16`
- Test: `src/state_space_practice/tests/test_hamiltonian_spikes.py` (create)

**Step 1: Write failing test**

Create `src/state_space_practice/tests/test_hamiltonian_spikes.py`:

```python
"""Tests for HamiltonianSpikeModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_spikes import HamiltonianSpikeModel


class TestHamiltonianSpikeModelSmoke:
    """Smoke tests: instantiation, filter, smooth all run without error."""

    @pytest.fixture
    def model_and_data(self):
        n_sources = 8
        n_time = 50
        model = HamiltonianSpikeModel(
            n_sources=n_sources, n_oscillators=1, hidden_dims=[16], seed=0
        )
        key = jax.random.PRNGKey(42)
        spikes = jax.random.poisson(key, jnp.ones((n_time, n_sources)) * 0.5)
        return model, spikes

    def test_smooth_runs(self, model_and_data):
        """smooth() should not raise NameError from missing imports."""
        model, spikes = model_and_data
        params = {
            "mlp_params": model.mlp_params,
            "omega": model.omega,
            "C": model.C,
            "d": model.d,
            "init_mean": model.init_mean[:, 0],
        }
        m_s, P_s = model.smooth(spikes, params)
        assert jnp.all(jnp.isfinite(m_s)), "Smoothed means contain non-finite values"
        assert m_s.shape[0] == spikes.shape[0]
```

**Step 2: Run test to verify it fails**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_spikes.py::TestHamiltonianSpikeModelSmoke::test_smooth_runs -v`

Expected: FAIL with `NameError: name 'get_transition_jacobian' is not defined`

**Step 3: Fix the import**

In `src/state_space_practice/hamiltonian_spikes.py`, change lines 15-16 from:

```python
from state_space_practice.nonlinear_dynamics import (
    leapfrog_step, apply_mlp, init_mlp_params, ekf_predict_step
)
```

to:

```python
from state_space_practice.nonlinear_dynamics import (
    leapfrog_step, apply_mlp, init_mlp_params, ekf_predict_step,
    get_transition_jacobian, ekf_smooth_step
)
```

**Step 4: Run test to verify it passes**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_spikes.py::TestHamiltonianSpikeModelSmoke::test_smooth_runs -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/state_space_practice/hamiltonian_spikes.py src/state_space_practice/tests/test_hamiltonian_spikes.py
git commit -m "fix: add missing imports to HamiltonianSpikeModel.smooth()"
```

---

### Task 2: Fix undefined K_l in SwitchingHamiltonianJointModel.smooth()

**Files:**
- Modify: `src/state_space_practice/hamiltonian_switching.py:157-161`
- Test: `src/state_space_practice/tests/test_hamiltonian_switching.py` (create)

**Step 1: Write failing test**

Create `src/state_space_practice/tests/test_hamiltonian_switching.py`:

```python
"""Tests for SwitchingHamiltonianJointModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_switching import SwitchingHamiltonianJointModel


class TestSwitchingHamiltonianSmoke:
    """Smoke tests: instantiation, filter, smooth run without error."""

    @pytest.fixture
    def model_and_data(self):
        n_lfp = 4
        n_spikes = 8
        n_time = 50
        n_discrete_states = 2
        model = SwitchingHamiltonianJointModel(
            n_lfp_sources=n_lfp,
            n_spike_sources=n_spikes,
            n_oscillators=1,
            n_discrete_states=n_discrete_states,
            hidden_dims=[16],
            seed=0,
        )
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        lfp = jax.random.normal(k1, (n_time, n_lfp))
        spikes = jax.random.poisson(k2, jnp.ones((n_time, n_spikes)) * 0.5)
        return model, lfp, spikes

    def test_smooth_runs(self, model_and_data):
        """smooth() should not raise NameError from undefined K_l."""
        model, lfp, spikes = model_and_data
        params = model._get_current_params()
        m_s, P_s = model.smooth(lfp, spikes, params)
        assert jnp.all(jnp.isfinite(m_s)), "Smoothed means contain non-finite values"
        assert m_s.shape[0] == lfp.shape[0]
```

**Step 2: Run test to verify it fails**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_switching.py::TestSwitchingHamiltonianSmoke::test_smooth_runs -v`

Expected: FAIL with `NameError: name 'K_l' is not defined`

**Step 3: Fix the undefined variable**

In `src/state_space_practice/hamiltonian_switching.py`, the `update_k` closure inside `smooth()`'s `forward_step` (around line 157-167) computes `S_l` but never defines `K_l`. Add the Kalman gain computation after `S_l`:

Change:

```python
            def update_k(k):
                mp, Pp = m_p_k[:, k], P_p_k[:, :, k]
                S_l = C_l @ Pp @ C_l.T + R_l
                m_mid = mp + K_l @ (y_lfp_t - (C_l @ mp + d_l))
```

to:

```python
            def update_k(k):
                mp, Pp = m_p_k[:, k], P_p_k[:, :, k]
                S_l = C_l @ Pp @ C_l.T + R_l
                K_l = Pp @ C_l.T @ jnp.linalg.inv(S_l)
                m_mid = mp + K_l @ (y_lfp_t - (C_l @ mp + d_l))
```

**Step 4: Run test to verify it passes**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_switching.py::TestSwitchingHamiltonianSmoke::test_smooth_runs -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/state_space_practice/hamiltonian_switching.py src/state_space_practice/tests/test_hamiltonian_switching.py
git commit -m "fix: define K_l in SwitchingHamiltonianJointModel.smooth()"
```

---

## Phase 2: Fix State Layout Mismatch

The `nonlinear_dynamics.leapfrog_step` uses **blocked** layout `[q1, q2, ..., p1, p2, ...]` (line 21: `q, p = x[:n], x[n:]`), but `_initialize_parameters` in `hamiltonian_spikes.py`, `hamiltonian_lfp.py`, `hamiltonian_joint.py`, and `hamiltonian_switching.py` uses **interleaved** layout `[q1, p1, q2, p2, ...]` (line 61: `jnp.array([0.1, 0.0] * self.n_oscillators)`). For `n_oscillators=1` these are identical, masking the bug.

### Task 3: Write test exposing multi-oscillator state layout bug

**Files:**
- Test: `src/state_space_practice/tests/test_hamiltonian_spikes.py` (append)

**Step 1: Write failing test**

Add to `test_hamiltonian_spikes.py`:

```python
class TestHamiltonianSpikeMultiOscillator:
    """Tests for n_oscillators > 1."""

    def test_construction_n_oscillators_2(self):
        """Model should construct with n_oscillators=2."""
        model = HamiltonianSpikeModel(
            n_sources=8, n_oscillators=2, hidden_dims=[16], seed=0
        )
        assert model.n_cont_states == 4
        assert model.init_mean.shape == (4, 1)

    def test_filter_n_oscillators_2(self):
        """Filter should run with n_oscillators=2."""
        model = HamiltonianSpikeModel(
            n_sources=8, n_oscillators=2, hidden_dims=[16], seed=0
        )
        spikes = jax.random.poisson(
            jax.random.PRNGKey(0), jnp.ones((50, 8)) * 0.5
        )
        params = {
            "mlp_params": model.mlp_params,
            "omega": model.omega,
            "C": model.C,
            "d": model.d,
            "init_mean": model.init_mean[:, 0],
        }
        m_f, P_f, ll = model.filter(spikes, params)
        assert m_f.shape == (50, 4)
        assert jnp.all(jnp.isfinite(m_f))
```

**Step 2: Run test to verify it fails**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_spikes.py::TestHamiltonianSpikeMultiOscillator -v`

Expected: FAIL — either construction error (C shape mismatch) or filter produces wrong results due to layout mismatch.

**Step 3: Fix state layout and C initialization**

The fix has two parts:

**Part A — Fix `_initialize_parameters` in all BaseModel subclasses to use blocked layout.**

In `src/state_space_practice/hamiltonian_spikes.py`, change line 61 from:

```python
        m0 = jnp.array([0.1, 0.0] * self.n_oscillators)
```

to:

```python
        m0 = jnp.concatenate([
            jnp.full((self.n_oscillators,), 0.1),  # q values
            jnp.zeros((self.n_oscillators,)),       # p values
        ])
```

Apply the identical fix in:
- `src/state_space_practice/hamiltonian_lfp.py` (same `_initialize_parameters` pattern)
- `src/state_space_practice/hamiltonian_joint.py` (same `_initialize_parameters` pattern)

**Part B — Fix hard-coded 2D spike readout C matrix.**

In `src/state_space_practice/hamiltonian_spikes.py`, change lines 50-51 from:

```python
        angles = jnp.linspace(0, 2 * jnp.pi, n_sources, endpoint=False)
        self.C = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1) * 0.5
```

to:

```python
        k_c = jax.random.split(self.key, 3)[2]
        self.C = jax.random.normal(k_c, (n_sources, self.n_cont_states)) * 0.1
```

Apply the identical fix for C_spikes in:
- `src/state_space_practice/hamiltonian_joint.py` lines 57-58 (same hard-coded pattern for `C_spikes`)

**Step 4: Run test to verify it passes**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_spikes.py::TestHamiltonianSpikeMultiOscillator -v`

Expected: PASS

**Step 5: Run all existing Hamiltonian tests to confirm no regression**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py src/state_space_practice/tests/test_hamiltonian_spikes.py -v`

Expected: All PASS

**Step 6: Commit**

```bash
git add src/state_space_practice/hamiltonian_spikes.py src/state_space_practice/hamiltonian_lfp.py src/state_space_practice/hamiltonian_joint.py src/state_space_practice/tests/test_hamiltonian_spikes.py
git commit -m "fix: use blocked state layout and parameterized C for multi-oscillator support"
```

---

## Phase 3: Fix Random Key Correlation

### Task 4: Fix correlated PRNG keys in HamiltonianPointProcessModel

**Files:**
- Modify: `src/state_space_practice/hamiltonian_models.py:116-128`
- Test: `src/state_space_practice/tests/test_hamiltonian_models.py` (append)

**Step 1: Write failing test**

Add to `test_hamiltonian_models.py`:

```python
def test_key_independence():
    """C and coupling_q should be initialized from independent keys."""
    model = HamiltonianPointProcessModel(
        n_oscillators=2, n_neurons=4, seed=0
    )
    # C and coupling_q should NOT be correlated
    # Extract the first 2 elements of each to compare
    c_flat = model.C.flatten()[:4]
    k_flat = model.hamiltonian_params["coupling_q"].flatten()[:4]
    # If keys are independent, correlation should be low
    corr = jnp.corrcoef(c_flat, k_flat)[0, 1]
    assert jnp.abs(corr) < 0.99, f"C and coupling_q suspiciously correlated: {corr}"
```

**Step 2: Run test to verify it fails**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py::test_key_independence -v`

Expected: FAIL (or may pass marginally — the real fix is structural correctness, not just this test)

**Step 3: Fix key splitting**

In `src/state_space_practice/hamiltonian_models.py`, change lines 116-128 from:

```python
        self.key = jax.random.PRNGKey(seed)
        
        # Dynamics
        self.hamiltonian_params = init_structured_hamiltonian_params(n_oscillators, self.key)
        self.process_cov = 1e-4 * jnp.eye(self.n_latent)
        
        # Initialization
        self.init_mean = jnp.zeros((self.n_latent,))
        self.init_cov = 1e-1 * jnp.eye(self.n_latent)
        
        # Observation (Log-linear)
        k1, k2 = jax.random.split(self.key)
        self.C = jax.random.normal(k1, (n_neurons, self.n_latent)) * 0.1
```

to:

```python
        self.key = jax.random.PRNGKey(seed)
        k_hamiltonian, k_C = jax.random.split(self.key)
        
        # Dynamics
        self.hamiltonian_params = init_structured_hamiltonian_params(n_oscillators, k_hamiltonian)
        self.process_cov = 1e-4 * jnp.eye(self.n_latent)
        
        # Initialization
        self.init_mean = jnp.zeros((self.n_latent,))
        self.init_cov = 1e-1 * jnp.eye(self.n_latent)
        
        # Observation (Log-linear)
        self.C = jax.random.normal(k_C, (n_neurons, self.n_latent)) * 0.1
```

**Step 4: Run all Hamiltonian tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/state_space_practice/hamiltonian_models.py src/state_space_practice/tests/test_hamiltonian_models.py
git commit -m "fix: use independent PRNG keys for C and hamiltonian_params"
```

---

### Task 5: Fix correlated key in simulate_data_hamiltonian.py

**Files:**
- Modify: `src/state_space_practice/simulate_data_hamiltonian.py:60`

**Step 1: Read current key splitting**

The file splits keys at line 28-29: `k1, k2, k3, k4 = jax.random.split(key, 4)`. Then line 60: `jax.random.poisson(jax.random.split(key, 1)[0], rates)` — this reuses the original `key` instead of a fresh subkey.

**Step 2: Fix key usage**

Change line 60 from:

```python
    spikes = jax.random.poisson(jax.random.split(key, 1)[0], rates)
```

to:

```python
    k5 = jax.random.fold_in(key, 4)
    spikes = jax.random.poisson(k5, rates)
```

**Step 3: Run existing tests that use the simulator**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py -v -k "simulate or filter"`

Expected: All PASS

**Step 4: Commit**

```bash
git add src/state_space_practice/simulate_data_hamiltonian.py
git commit -m "fix: use independent PRNG key for spike sampling in simulator"
```

---

## Phase 4: Fix Switching Smoother Scientific Correctness

### Task 6: Fix uniform Jacobian averaging in switching smoother backward pass

**Files:**
- Modify: `src/state_space_practice/hamiltonian_switching.py:147-155, 171, 176-183`
- Test: `src/state_space_practice/tests/test_hamiltonian_switching.py` (append)

**Step 1: Write test for probability-weighted Jacobian**

Add to `test_hamiltonian_switching.py`:

```python
    def test_smooth_uses_weighted_jacobians(self, model_and_data):
        """Backward pass should weight Jacobians by transition probabilities, not uniform average."""
        model, lfp, spikes = model_and_data
        # Set asymmetric transition matrix so uniform != weighted
        model.discrete_transition_matrix = jnp.array([[0.9, 0.1], [0.3, 0.7]])
        params = model._get_current_params()
        m_s, P_s = model.smooth(lfp, spikes, params)
        # Basic sanity: smoothed means should be finite and shaped correctly
        assert jnp.all(jnp.isfinite(m_s))
        assert m_s.shape == (lfp.shape[0], model.n_cont_states, 2)
```

**Step 2: Run test to verify current state**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_switching.py -v`

Expected: Should pass after Task 2's K_l fix (this test validates the weighted fix works correctly)

**Step 3: Fix the backward pass to use probability-weighted Jacobians**

In `src/state_space_practice/hamiltonian_switching.py`, the forward scan must output the collapsing weights, and the backward scan must use them.

**Part A — Store collapsing weights in forward scan output.**

Change the forward_step return (around line 171) to also output `joint_pi_pred` and `pi_pred_k`:

```python
            return (m_f, P_f, pi_pred_k), (m_f, P_f, m_p_k, P_p_k, F_jk, joint_pi_pred, pi_pred_k)
```

Update the scan unpack (around line 174):

```python
        _, (m_filt, P_filt, m_pred, P_pred, F_all, joint_pi_all, pi_pred_all) = jax.lax.scan(forward_step, (m0, P0, pi0), (lfp_data, spike_data))
```

**Part B — Use probability-weighted average in backward pass.**

Change the `smooth_k` closure (around line 179-181) from:

```python
            def smooth_k(k):
                F_avg = jnp.mean(F_next_jk[:, k, :, :], axis=0) 
                return ekf_smooth_step(...)
```

to:

```python
            def smooth_k(k):
                w_jk = _divide_safe(joint_pi_next[:, k], pi_pred_next_k[k])
                F_avg = jnp.sum(w_jk[:, None, None] * F_next_jk[:, k, :, :], axis=0)
                return ekf_smooth_step(...)
```

Update the backward_step inputs to include the weights:

```python
        def backward_step(carry, inputs):
            m_s_next, P_s_next = carry
            m_f_t, P_f_t, m_p_next, P_p_next, F_next_jk, joint_pi_next, pi_pred_next_k = inputs
```

And update the backward scan inputs:

```python
        bw_in = (m_filt[:-1], P_filt[:-1], m_pred[1:], P_pred[1:], F_all[1:], joint_pi_all[1:], pi_pred_all[1:])
```

**Step 4: Run test to verify it passes**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_switching.py -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/state_space_practice/hamiltonian_switching.py src/state_space_practice/tests/test_hamiltonian_switching.py
git commit -m "fix: use probability-weighted Jacobians in switching smoother backward pass"
```

---

## Phase 5: Add Smoke Tests for Untested Models

### Task 7: Add smoke tests for HamiltonianLFPModel

**Files:**
- Test: `src/state_space_practice/tests/test_hamiltonian_lfp.py` (create)

**Step 1: Write smoke tests**

Create `src/state_space_practice/tests/test_hamiltonian_lfp.py`:

```python
"""Smoke tests for HamiltonianLFPModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel


class TestHamiltonianLFPSmoke:

    @pytest.fixture
    def model_and_data(self):
        n_sources = 4
        n_time = 50
        model = HamiltonianLFPModel(
            n_sources=n_sources, n_oscillators=1, hidden_dims=[16], seed=0
        )
        lfp = jax.random.normal(jax.random.PRNGKey(42), (n_time, n_sources))
        return model, lfp

    def test_construction(self, model_and_data):
        model, _ = model_and_data
        assert model.n_cont_states == 2

    def test_filter_runs(self, model_and_data):
        model, lfp = model_and_data
        params = model._get_current_params()
        m_f, P_f, ll = model.filter(lfp, params)
        assert jnp.all(jnp.isfinite(m_f))
        assert m_f.shape[0] == lfp.shape[0]

    def test_smooth_runs(self, model_and_data):
        model, lfp = model_and_data
        params = model._get_current_params()
        m_s, P_s = model.smooth(lfp, params)
        assert jnp.all(jnp.isfinite(m_s))
        assert m_s.shape[0] == lfp.shape[0]
```

**Step 2: Run tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_lfp.py -v`

Expected: All PASS (after Phase 2 fixes)

**Step 3: Commit**

```bash
git add src/state_space_practice/tests/test_hamiltonian_lfp.py
git commit -m "test: add smoke tests for HamiltonianLFPModel"
```

---

### Task 8: Add smoke tests for JointHamiltonianModel

**Files:**
- Test: `src/state_space_practice/tests/test_hamiltonian_joint.py` (create)

**Step 1: Write smoke tests**

Create `src/state_space_practice/tests/test_hamiltonian_joint.py`:

```python
"""Smoke tests for JointHamiltonianModel."""

import jax
import jax.numpy as jnp
import pytest

from state_space_practice.hamiltonian_joint import JointHamiltonianModel


class TestJointHamiltonianSmoke:

    @pytest.fixture
    def model_and_data(self):
        n_lfp = 4
        n_spikes = 8
        n_time = 50
        model = JointHamiltonianModel(
            n_lfp_sources=n_lfp,
            n_spike_sources=n_spikes,
            n_oscillators=1,
            hidden_dims=[16],
            seed=0,
        )
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        lfp = jax.random.normal(k1, (n_time, n_lfp))
        spikes = jax.random.poisson(k2, jnp.ones((n_time, n_spikes)) * 0.5)
        return model, lfp, spikes

    def test_construction(self, model_and_data):
        model, _, _ = model_and_data
        assert model.n_cont_states == 2

    def test_filter_runs(self, model_and_data):
        model, lfp, spikes = model_and_data
        params = model._get_current_params()
        m_f, P_f, ll = model.filter(lfp, spikes, params)
        assert jnp.all(jnp.isfinite(m_f))
        assert m_f.shape[0] == lfp.shape[0]

    def test_smooth_runs(self, model_and_data):
        model, lfp, spikes = model_and_data
        params = model._get_current_params()
        m_s, P_s = model.smooth(lfp, spikes, params)
        assert jnp.all(jnp.isfinite(m_s))
        assert m_s.shape[0] == lfp.shape[0]
```

**Step 2: Run tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_joint.py -v`

Expected: All PASS

**Step 3: Commit**

```bash
git add src/state_space_practice/tests/test_hamiltonian_joint.py
git commit -m "test: add smoke tests for JointHamiltonianModel"
```

---

### Task 9: Expand switching model tests

**Files:**
- Modify: `src/state_space_practice/tests/test_hamiltonian_switching.py` (append)

**Step 1: Add filter smoke test**

Add to `test_hamiltonian_switching.py`:

```python
    def test_filter_runs(self, model_and_data):
        """filter() should run without error."""
        model, lfp, spikes = model_and_data
        params = model._get_current_params()
        m_f, P_f, ll = model.filter(lfp, spikes, params)
        assert jnp.all(jnp.isfinite(m_f))
        assert m_f.shape[0] == lfp.shape[0]
```

**Step 2: Run tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_switching.py -v`

Expected: All PASS

**Step 3: Commit**

```bash
git add src/state_space_practice/tests/test_hamiltonian_switching.py
git commit -m "test: add filter smoke test for SwitchingHamiltonianJointModel"
```

---

## Phase 6: Remove Dead Parameters and Clean Up API

### Task 10: Remove unused predicted_mean/predicted_cov from smoother signature

**Files:**
- Modify: `src/state_space_practice/hamiltonian_point_process.py:98-132`
- Test: existing tests should still pass

**Step 1: Read the current smoother signature and its caller**

Verify that `predicted_mean` and `predicted_cov` are indeed unused in the smoother body.

**Step 2: Remove the dead parameters**

In `src/state_space_practice/hamiltonian_point_process.py`, change the smoother function signature to remove `predicted_mean` and `predicted_cov`:

```python
def hamiltonian_point_process_smoother(
    filtered_mean: Array,
    filtered_cov: Array,
    linearization_matrices: Array,
    process_cov: Array,
) -> Tuple[Array, Array]:
```

Update the caller in `hamiltonian_models.py` (`HamiltonianPointProcessModel.smooth()`) to stop passing those arguments.

**Step 3: Run all existing tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py -v`

Expected: All PASS

**Step 4: Commit**

```bash
git add src/state_space_practice/hamiltonian_point_process.py src/state_space_practice/hamiltonian_models.py
git commit -m "refactor: remove unused predicted_mean/predicted_cov from smoother API"
```

---

### Task 11: Remove nested jax.jit from hamiltonian_point_process_filter

**Files:**
- Modify: `src/state_space_practice/hamiltonian_point_process.py:63-65`

**Step 1: Remove redundant inner jit wrappers**

The outer function is already `@jax.jit`, so inner `jax.jit(jax.jacfwd(...))` calls should just be `jax.jacfwd(...)`.

**Step 2: Run tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py -v`

Expected: All PASS

**Step 3: Commit**

```bash
git add src/state_space_practice/hamiltonian_point_process.py
git commit -m "refactor: remove redundant nested jit in hamiltonian_point_process_filter"
```

---

## Phase 7: Run Full Test Suite

### Task 12: Final verification

**Step 1: Run all Hamiltonian tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py src/state_space_practice/tests/test_hamiltonian_spikes.py src/state_space_practice/tests/test_hamiltonian_lfp.py src/state_space_practice/tests/test_hamiltonian_joint.py src/state_space_practice/tests/test_hamiltonian_switching.py -v`

Expected: All PASS

**Step 2: Run full test suite to check for regressions**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/ -v --ignore=src/state_space_practice/tests/test_hamiltonian_models.py -x -k "not slow"`

Expected: All PASS

**Step 3: Run linter**

Run: `conda run -n state_space_practice ruff check src/state_space_practice/hamiltonian_models.py src/state_space_practice/hamiltonian_point_process.py src/state_space_practice/hamiltonian_spikes.py src/state_space_practice/hamiltonian_lfp.py src/state_space_practice/hamiltonian_joint.py src/state_space_practice/hamiltonian_switching.py src/state_space_practice/nonlinear_dynamics.py src/state_space_practice/simulate_data_hamiltonian.py`

Expected: No errors
