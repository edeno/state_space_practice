# Hamiltonian Oscillator State-Space Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a SymODEN-style latent oscillator model for neural data: symplectic Hamiltonian latent dynamics, stochastic state uncertainty, and existing Gaussian or point-process observation likelihoods.

**Architecture:** Start with a single discrete-state Hamiltonian point-process model outside the existing switching `BaseModel` hierarchy. Reuse the current Laplace-EKF observation update from `point_process_kalman.py` and RTS smoother math from `kalman.py`, but replace the linear prediction step with a symplectic nonlinear transition and local Jacobian linearization. Keep the first implementation structured and low-dimensional: oscillator states are canonical `[q, p]` pairs, the Hamiltonian is interpretable, and process noise remains Gaussian.

**Tech Stack:** JAX, existing `point_process_kalman.py`, `kalman.py`, `simulate_data.py`, pytest. No new deep-learning dependency in V1.

---

## Why This Plan

The current prototype in `src/state_space_practice/hamiltonian_models.py` is a deterministic rollout model with a Poisson readout. That is not yet a usable state-space model for this repository because it does not perform latent-state inference and it does not fit the abstract `BaseModel` contract.

The correct first milestone is smaller and better aligned with the paper and the repo:

1. Hamiltonian latent dynamics over oscillator phase-space.
2. Symplectic integration for state prediction.
3. Gaussian process noise for uncertainty.
4. Existing point-process Laplace update for spike observations.
5. End-to-end filtering, smoothing, and likelihood-based fitting on synthetic oscillator data.

## Non-Goals for V1

- No switching Hamiltonian model yet.
- No control inputs yet.
- No autoencoder or image-space latent embedding.
- No arbitrary black-box MLP Hamiltonian as the only dynamic prior.
- No attempt to force the current `BaseModel` EM hierarchy to support nonlinear dynamics in the first pass.

## V1 Model Definition

For `n_oscillators`, the latent state is:

$$
x_t = [q_{1,t}, p_{1,t}, q_{2,t}, p_{2,t}, \dots, q_{K,t}, p_{K,t}]^\top
$$

Use a structured Hamiltonian:

$$
H(q, p) = T(p; m) + V(q; \omega, \alpha) + C(q; \theta_{\mathrm{coupling}}) + R_\phi(q, p)
$$

with:

- `T(p; m)` diagonal kinetic energy with learnable positive masses.
- `V(q; ω, α)` oscillator-local quadratic or weakly nonlinear potential.
- `C(q; θ_coupling)` pairwise coupling energy between oscillators.
- `R_φ(q, p)` optional small residual MLP that is initialized near zero and added only after the structured model is working.

Discrete-time prediction uses leapfrog:

$$
x_t^- = f_\theta(x_{t-1})
$$

and covariance propagation uses the local Jacobian:

$$
P_t^- = A_t P_{t-1} A_t^\top + Q, \qquad A_t = \frac{\partial f_\theta}{\partial x}(x_{t-1})
$$

Spike observations use the existing point-process update machinery.

## Success Criteria

1. A Hamiltonian oscillator transition can be evaluated and linearized for any finite latent state.
2. A nonlinear point-process filter and smoother run end-to-end on synthetic spike data.
3. The model returns finite filtered means, smoothed means, covariances, and log-likelihoods.
4. On synthetic Hamiltonian oscillator data, fitting improves marginal log-likelihood over initialization.
5. The nonlinear Hamiltonian model outperforms a linear random-walk or linear oscillator baseline on at least one nonlinear synthetic benchmark.

### Task 1: Replace the Prototype With a Viable Module Boundary

**Files:**
- Modify: `src/state_space_practice/hamiltonian_models.py`
- Modify: `src/state_space_practice/__init__.py`
- Modify: `src/state_space_practice/tests/test_hamiltonian_models.py`

**Step 1: Write failing contract tests**

Replace the current smoke tests with tests that assert the real V1 contract:

```python
def test_hamiltonian_model_is_standalone_sgd_model():
    model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=2,
        sampling_freq=100.0,
        dt=0.01,
    )
    assert model.n_latent == 2
    assert model.process_cov.shape == (2, 2)


def test_fit_sgd_sets_internal_sequence_length():
    model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=2,
        sampling_freq=100.0,
        dt=0.01,
    )
    spikes = jnp.zeros((5, 2))
    design_matrix = jnp.zeros((5, 2, 2))
    model.fit_sgd(design_matrix, spikes, num_steps=1)
    assert model.log_likelihood_history_
```

**Step 2: Run the test to verify the current code fails**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py -v`

Expected: FAIL because the current class still uses the wrong base class and wrong SGD lifecycle.

**Step 3: Rewrite the module boundary**

Make `hamiltonian_models.py` stop inheriting from `BaseModel`. Replace the current `HamiltonianOscillatorModel` prototype with a narrow standalone model class and pure dynamics helpers.

Use this skeleton:

```python
class HamiltonianPointProcessModel(SGDFittableMixin):
    def __init__(self, n_oscillators: int, n_neurons: int, sampling_freq: float, dt: float, ...):
        self.n_oscillators = n_oscillators
        self.n_neurons = n_neurons
        self.n_latent = 2 * n_oscillators
        self.sampling_freq = sampling_freq
        self.dt = dt
        self.process_cov = 1e-4 * jnp.eye(self.n_latent)
        self.init_mean = jnp.zeros((self.n_latent,))
        self.init_cov = jnp.eye(self.n_latent)
        ...

    def fit_sgd(self, design_matrix, spike_indicator, ...):
        self._sgd_n_time = spike_indicator.shape[0]
        return super().fit_sgd(design_matrix, spike_indicator, ...)

    @property
    def _n_timesteps(self) -> int:
        return self._sgd_n_time
```

**Step 4: Fix the SGD contract**

- `_build_param_spec()` must use existing transforms from `parameter_transforms.py`.
- `_sgd_loss_fn()` must return total negative log-likelihood, not divide by time again.
- `_check_sgd_initialized()` must raise if `fit_sgd()` was not called correctly.

**Step 5: Run the tests to verify the module imports and the lifecycle works**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py -v`

Expected: PASS for import, initialization, and basic SGD lifecycle tests.

**Step 6: Commit**

```bash
git add src/state_space_practice/hamiltonian_models.py src/state_space_practice/__init__.py src/state_space_practice/tests/test_hamiltonian_models.py
git commit -m "refactor: replace hamiltonian prototype with standalone sgd model"
```

### Task 2: Add Structured Hamiltonian Dynamics Utilities

**Files:**
- Modify: `src/state_space_practice/hamiltonian_models.py`
- Test: `src/state_space_practice/tests/test_hamiltonian_models.py`

**Step 1: Write the failing dynamics tests**

Add tests for the structured Hamiltonian pieces:

```python
def test_leapfrog_step_returns_finite_state():
    params = init_structured_hamiltonian_params(n_oscillators=2, key=jax.random.PRNGKey(0))
    x = jnp.array([1.0, 0.0, 0.5, 0.0])
    x_next = symplectic_transition(x, params, dt=0.01)
    assert x_next.shape == x.shape
    assert jnp.all(jnp.isfinite(x_next))


def test_transition_jacobian_is_finite():
    params = init_structured_hamiltonian_params(n_oscillators=1, key=jax.random.PRNGKey(0))
    x = jnp.array([0.7, -0.2])
    jac = transition_jacobian(x, params, dt=0.01)
    assert jac.shape == (2, 2)
    assert jnp.all(jnp.isfinite(jac))
```

**Step 2: Implement structured Hamiltonian parameters**

Add a parameter dictionary or dataclass with:

- `log_mass`: shape `(n_oscillators,)`
- `log_frequency`: shape `(n_oscillators,)`
- `log_stiffness`: shape `(n_oscillators,)`
- `coupling_q`: shape `(n_oscillators, n_oscillators)` with zero diagonal
- optional `residual_mlp`

Implement pure functions:

```python
def kinetic_energy(p, params):
    mass = jnp.exp(params["log_mass"])
    return 0.5 * jnp.sum((p ** 2) / jnp.repeat(mass, 1))


def potential_energy(q, params):
    stiffness = jnp.exp(params["log_stiffness"])
    return 0.5 * jnp.sum(stiffness * q ** 2)


def coupling_energy(q, params):
    K = params["coupling_q"]
    return 0.5 * q @ K @ q


def hamiltonian(state, params):
    q = state[0::2]
    p = state[1::2]
    return kinetic_energy(p, params) + potential_energy(q, params) + coupling_energy(q, params)
```

Use canonical ordering consistently. If you choose `[q1, q2, ..., p1, p2, ...]` instead, update all helpers and tests to match. Do not mix conventions.

**Step 3: Implement symplectic transition and Jacobian**

Expose:

- `symplectic_transition(state, params, dt)`
- `transition_jacobian(state, params, dt)` via `jax.jacfwd`

Keep the residual MLP disabled by default in V1. The first working version should rely on structured energies only.

**Step 4: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_models.py -k "transition or jacobian" -v`

Expected: PASS for finite-state and finite-Jacobian tests.

**Step 5: Commit**

```bash
git add src/state_space_practice/hamiltonian_models.py src/state_space_practice/tests/test_hamiltonian_models.py
git commit -m "feat: add structured hamiltonian oscillator dynamics"
```

### Task 3: Add a Nonlinear Point-Process Filter Using the Existing Laplace Update

**Files:**
- Create: `src/state_space_practice/hamiltonian_point_process.py`
- Modify: `src/state_space_practice/__init__.py`
- Test: `src/state_space_practice/tests/test_hamiltonian_point_process.py`

**Step 1: Write the failing filter test**

Create a small end-to-end test:

```python
def test_hamiltonian_point_process_filter_returns_finite_outputs():
    design_matrix = jnp.zeros((20, 3, 2))
    spikes = jnp.zeros((20, 3), dtype=int)
    model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=3,
        sampling_freq=100.0,
        dt=0.01,
    )
    result = model.filter(design_matrix, spikes)
    assert result.filtered_mean.shape == (20, 2)
    assert jnp.all(jnp.isfinite(result.filtered_mean))
    assert jnp.all(jnp.isfinite(result.filtered_cov))
```

**Step 2: Reuse the existing observation update**

Do not reimplement the point-process observation math. In `hamiltonian_point_process.py`, import and reuse:

- `_point_process_laplace_update` from `point_process_kalman.py`
- `_kalman_smoother_update` and `symmetrize` from `kalman.py`

Implement a nonlinear prediction helper:

```python
def nonlinear_predict(mean_prev, cov_prev, params, process_cov, dt):
    mean_pred = symplectic_transition(mean_prev, params, dt)
    A_t = transition_jacobian(mean_prev, params, dt)
    cov_pred = symmetrize(A_t @ cov_prev @ A_t.T + process_cov)
    return mean_pred, cov_pred, A_t
```

**Step 3: Add filter and smoother result containers**

Mirror the shape conventions from `PointProcessModel`:

- `filtered_mean`: `(n_time, n_latent)`
- `filtered_cov`: `(n_time, n_latent, n_latent)`
- `smoother_mean`: `(n_time, n_latent)`
- `smoother_cov`: `(n_time, n_latent, n_latent)`

Keep the first version single-state only. No discrete-state arrays.

**Step 4: Implement `filter()` and `smooth()`**

Use `jax.lax.scan` in the filter. Save the linearization matrices `A_t` or predicted covariances needed for smoothing.

For the first smoother implementation, use `_kalman_smoother_update` with the locally linearized `A_t` at each step.

**Step 5: Run the filter test**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_point_process.py -v`

Expected: PASS for finite filtered and smoothed outputs.

**Step 6: Commit**

```bash
git add src/state_space_practice/hamiltonian_point_process.py src/state_space_practice/__init__.py src/state_space_practice/tests/test_hamiltonian_point_process.py
git commit -m "feat: add nonlinear hamiltonian point-process filter"
```

### Task 4: Add Synthetic Hamiltonian Oscillator Simulation

**Files:**
- Modify: `src/state_space_practice/simulate_data.py`
- Test: `src/state_space_practice/tests/test_hamiltonian_point_process.py`
- Optional analysis: `notebooks/correctness_oscillator_models.py`

**Step 1: Write the failing simulator test**

```python
def test_simulate_hamiltonian_spikes_is_deterministic_under_seed():
    out1 = simulate_hamiltonian_spike_data(seed=0, n_time=50, n_oscillators=1, n_neurons=3)
    out2 = simulate_hamiltonian_spike_data(seed=0, n_time=50, n_oscillators=1, n_neurons=3)
    np.testing.assert_allclose(out1.latent_states, out2.latent_states)
    np.testing.assert_array_equal(out1.spikes, out2.spikes)
```

**Step 2: Implement the simulator**

Add a helper returning a small dataclass with:

- `latent_states`
- `spikes`
- `design_matrix`
- `true_params`

Simulation should:

1. Roll out latent states with `symplectic_transition(...) + Gaussian process noise`.
2. Form log-rates with a fixed neuron-weight matrix.
3. Sample spikes with `jax.random.poisson`.

Use the same design-matrix shape as `PointProcessModel`: `(n_time, n_neurons, n_latent)`.

**Step 3: Add boundedness and shape tests**

Test that:

- latent states are finite
- spike array has shape `(n_time, n_neurons)`
- design matrix has shape `(n_time, n_neurons, n_latent)`
- moderate stable parameters do not produce immediate blow-up

**Step 4: Run the tests**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_point_process.py -k simulate -v`

Expected: PASS for deterministic and finite simulation tests.

**Step 5: Commit**

```bash
git add src/state_space_practice/simulate_data.py src/state_space_practice/tests/test_hamiltonian_point_process.py
git commit -m "feat: add synthetic hamiltonian oscillator spike simulation"
```

### Task 5: Add End-to-End SGD Fitting and Verify Likelihood Improvement

**Files:**
- Modify: `src/state_space_practice/hamiltonian_point_process.py`
- Modify: `src/state_space_practice/tests/test_hamiltonian_point_process.py`

**Step 1: Write the failing SGD integration test**

Use the same style as `test_sgd_integration.py`:

```python
def test_hamiltonian_point_process_fit_sgd_improves_ll():
    data = simulate_hamiltonian_spike_data(
        seed=0,
        n_time=100,
        n_oscillators=1,
        n_neurons=4,
    )
    model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=4,
        sampling_freq=100.0,
        dt=0.01,
    )
    ll_history = model.fit_sgd(data.design_matrix, data.spikes, num_steps=25)
    assert ll_history[-1] > ll_history[0]
    assert model.smoother_mean is not None
```

**Step 2: Implement the SGD loss correctly**

The loss must call the nonlinear filter and return total negative marginal log-likelihood:

```python
def _sgd_loss_fn(self, params, design_matrix, spike_indicator):
    _, _, marginal_ll = hamiltonian_point_process_filter(
        init_mean=params.get("init_mean", self.init_mean),
        init_cov=params.get("init_cov", self.init_cov),
        process_cov=params.get("process_cov", self.process_cov),
        hamiltonian_params=params.get("hamiltonian", self.hamiltonian_params),
        design_matrix=design_matrix,
        spike_indicator=spike_indicator,
        dt=self.dt,
        log_intensity_func=self.log_intensity_func,
    )
    return -marginal_ll
```

Do not divide by sequence length inside `_sgd_loss_fn()`.

**Step 3: Store smoother outputs in `_finalize_sgd()`**

After optimization, run the nonlinear smoother so the fitted model behaves like the other SGD models in the repo.

**Step 4: Run the integration test**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_point_process.py -k fit_sgd -v`

Expected: PASS with increasing log-likelihood.

**Step 5: Commit**

```bash
git add src/state_space_practice/hamiltonian_point_process.py src/state_space_practice/tests/test_hamiltonian_point_process.py
git commit -m "feat: add sgd fitting for hamiltonian point-process model"
```

### Task 6: Benchmark Against the Existing Linear Point-Process Model

**Files:**
- Modify: `src/state_space_practice/tests/test_hamiltonian_point_process.py`
- Optional analysis: `notebooks/correctness_oscillator_models.py`

**Step 1: Write the benchmark test**

```python
def test_hamiltonian_model_beats_linear_baseline_on_nonlinear_synthetic_data():
    data = simulate_hamiltonian_spike_data(
        seed=0,
        n_time=150,
        n_oscillators=1,
        n_neurons=4,
    )
    linear_model = PointProcessModel(n_state_dims=2, dt=0.01)
    nonlinear_model = HamiltonianPointProcessModel(
        n_oscillators=1,
        n_neurons=4,
        sampling_freq=100.0,
        dt=0.01,
    )
    linear_ll = linear_model.fit_sgd(data.design_matrix, data.spikes, num_steps=25)[-1]
    nonlinear_ll = nonlinear_model.fit_sgd(data.design_matrix, data.spikes, num_steps=25)[-1]
    assert nonlinear_ll > linear_ll
```

**Step 2: Keep the benchmark pragmatic**

The nonlinear model only needs to beat the linear baseline on nonlinear synthetic data generated from the Hamiltonian simulator. It does not need to dominate on every synthetic regime.

**Step 3: Run the benchmark**

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_hamiltonian_point_process.py -k baseline -v`

Expected: PASS on at least one controlled nonlinear synthetic regime.

**Step 4: Commit**

```bash
git add src/state_space_practice/tests/test_hamiltonian_point_process.py
git commit -m "test: benchmark hamiltonian oscillator model against linear baseline"
```

## Verification Checklist

- [ ] `hamiltonian_models.py` no longer depends on the switching `BaseModel` contract.
- [ ] Symplectic transition and Jacobian are finite on representative oscillator states.
- [ ] Nonlinear point-process filter returns finite means, covariances, and log-likelihoods.
- [ ] Synthetic Hamiltonian spike simulation is deterministic under a fixed seed.
- [ ] `fit_sgd()` improves marginal log-likelihood on synthetic Hamiltonian data.
- [ ] The nonlinear Hamiltonian model beats the linear point-process baseline on at least one nonlinear synthetic benchmark.

## Known Risks and Mitigations

1. **Zero-gradient startup through the Hamiltonian parameters.**
   Mitigation: initialize observation weights away from exact zero or warm-start them from a small Gaussian.

2. **Energy conservation can be too rigid for neural data.**
   Mitigation: keep Gaussian process noise in the state dynamics and allow weak dissipative or residual terms later if needed.

3. **A fully generic neural Hamiltonian is hard to identify.**
   Mitigation: keep V1 structured and interpretable. Add a residual MLP only after the structured model works.

4. **Switching integration will multiply the complexity.**
   Mitigation: do not touch `point_process_models.py` switching abstractions until the single-state model is stable.

## Deferred Extensions

- Regime-specific Hamiltonians for switching point-process models.
- Control inputs in the spirit of SymODEN with control.
- Gaussian observation wrapper for LFP or continuous neural signals.
- Residual Hamiltonian network on top of the structured oscillator energy.
- Joint training of observation parameters and latent dynamics on real CA1 data.

## Expected Outcome

This plan produces a first real Hamiltonian oscillator state-space model for the repository: symplectic nonlinear latent dynamics, uncertainty propagation, existing spike-observation inference, and a synthetic benchmark showing where the approach improves on the linear baseline.