# Computational and Numerical Improvements — Task Breakdown

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Tasks are independent — implement in any order. Each task has its own verification gate. Existing tests must continue to pass after every task (strict regression).

**Goal:** Implement the computational improvements described in `docs/plans/2026-04-05-computational-improvements.md`.

**Design doc:** `docs/plans/2026-04-05-computational-improvements.md`

**Key files:**

- Modify: `src/state_space_practice/kalman.py`
- Modify: `src/state_space_practice/point_process_kalman.py`
- Modify: `src/state_space_practice/multinomial_choice.py`
- Modify: `src/state_space_practice/covariate_choice.py`
- Create: `src/state_space_practice/parameter_transforms.py`
- Create: `src/state_space_practice/tests/test_parameter_transforms.py`
- Optionally create: `src/state_space_practice/info_kalman.py`

**Prerequisite Gates:**

- All existing tests pass before starting any task.

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/ -q
```

**Verification Gates (per task):**

- Targeted tests for the new functionality pass.
- ALL existing tests still pass (strict regression).
- `conda run -n state_space_practice ruff check src/state_space_practice`

---

## Task 1: Parallel Kalman Smoother via Associative Scan

### Step 1.1: Write failing tests

```python
# Add to tests/test_kalman.py

class TestParallelKalmanSmoother:
    def test_matches_sequential_smoother(self):
        # Generate synthetic LG-SSM data (T=500, D=4)
        # Run sequential kalman_smoother
        # Run parallel_kalman_smoother
        # Assert: means match (atol=1e-5), covs match, cross_covs match

    def test_output_shapes(self):
        # T=100, D=3 → smoothed_mean (T,3), smoothed_cov (T,3,3), cross_cov (T-1,3,3)

    def test_time_varying_transition(self):
        # transition_matrix shape (T-1, D, D), process_cov shape (T-1, D, D)
        # Should handle per-timestep parameters

    def test_single_timestep(self):
        # T=1 → smoother output equals filter output

    def test_two_timesteps(self):
        # T=2 → verify manually against closed-form RTS update
```

### Step 1.2: Implement parallel smoother

Add to `src/state_space_practice/kalman.py`:

```python
class _SmootherElement(NamedTuple):
    """Associative scan element for RTS smoother."""
    E: Array   # smoother gain (D, D)
    g: Array   # bias (D,)
    L: Array   # residual covariance (D, D, D)

def _smoother_operator(elem1: _SmootherElement, elem2: _SmootherElement) -> _SmootherElement:
    """Associative binary operator for composing smoother elements."""
    E = elem1.E @ elem2.E
    g = elem1.E @ elem2.g + elem1.g
    L = elem1.E @ elem2.L @ elem1.E.T + elem1.L
    return _SmootherElement(E=E, g=g, L=L)

def parallel_kalman_smoother(
    filtered_means: Array,
    filtered_covariances: Array,
    transition_matrix: Array,
    process_cov: Array,
) -> tuple[Array, Array, Array]:
    """RTS smoother via jax.lax.associative_scan(reverse=True).

    Algebraically equivalent to the sequential kalman_smoother.
    O(log T) parallel depth on GPU/TPU.
    """
    # 1. Compute per-timestep smoother elements (E_t, g_t, L_t)
    #    from filtered means/covs and dynamics
    # 2. Run jax.lax.associative_scan(_smoother_operator, elements, reverse=True)
    # 3. Extract smoothed means/covs from scan output
    # 4. Compute cross-covariances from smoother gains
```

**Reference:** Särkkä & García-Fernández (2021), Section IV.

### Step 1.3: Run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_kalman.py -v -k "parallel"
conda run -n state_space_practice pytest src/state_space_practice/tests/test_kalman.py -v  # regression
```

### Step 1.4: Commit

```bash
git commit -m "Add parallel RTS smoother via associative scan"
```

---

## Task 2: Woodbury-Optimized Observation Update

### Step 2.1: Write failing tests

```python
# Add to tests/test_kalman.py

class TestWoodburyKalmanGain:
    def test_matches_standard_gain(self):
        # D_state=4, D_obs=50, diagonal R
        # Woodbury gain == standard gain (atol=1e-6)

    def test_many_neurons(self):
        # D_state=4, D_obs=200
        # Correct Kalman gain and innovation covariance

    def test_log_likelihood_matches(self):
        # Innovation log-likelihood from Woodbury matches standard

    def test_auto_select(self):
        # _compute_kalman_gain selects Woodbury when D_obs > 2*D_state
        # and standard otherwise

    def test_full_covariance_R(self):
        # When R is full matrix (not diagonal), falls back to standard
```

### Step 2.2: Implement

Add to `src/state_space_practice/kalman.py`:

```python
def woodbury_kalman_gain(
    prior_cov: Array,
    emission_matrix: Array,
    emission_cov_diag: Array,
) -> tuple[Array, Array]:
    """Kalman gain via Woodbury identity for diagonal observation noise.

    O(D_state^3) instead of O(D_obs^3).
    """
    D = prior_cov.shape[0]
    I = jnp.eye(D)
    L = emission_matrix @ jnp.linalg.cholesky(prior_cov)  # (D_obs, D_state)
    X = L / emission_cov_diag[:, None]                      # (D_obs, D_state)
    # Woodbury: S_inv = R^{-1} - X (I + X'X)^{-1} X'
    S_inv = jnp.diag(1.0 / emission_cov_diag) - X @ psd_solve(I + L.T @ X, X.T)
    K = prior_cov @ emission_matrix.T @ S_inv
    S = jnp.diag(emission_cov_diag) + emission_matrix @ prior_cov @ emission_matrix.T
    return K, S
```

### Step 2.3: Wire into point_process_kalman.py

In `_point_process_laplace_update`, when computing the posterior precision:
- If `n_neurons > 2 * n_state`, use Woodbury
- Otherwise, use standard

This requires refactoring the Fisher information computation to optionally use the Woodbury form.

### Step 2.4: Run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_kalman.py -v -k "woodbury"
conda run -n state_space_practice pytest src/state_space_practice/tests/ -q  # full regression
```

### Step 2.5: Commit

```bash
git commit -m "Add Woodbury-optimized Kalman gain for diagonal observation noise"
```

---

## Task 3: Joseph Form Covariance Update

### Step 3.1: Write failing tests

```python
# Add to tests/test_kalman.py

class TestJosephFormUpdate:
    def test_matches_standard_well_conditioned(self):
        # Standard and Joseph form agree on well-conditioned problem

    def test_maintains_psd_ill_conditioned(self):
        # Create near-singular prior covariance
        # Standard form may lose PSD; Joseph form should not
        # Check: all eigenvalues >= 0

    def test_symmetric_output(self):
        # Joseph form output is exactly symmetric (no symmetrize needed)
```

### Step 3.2: Implement

Add to `src/state_space_practice/kalman.py`:

```python
def _joseph_form_update(
    prior_cov: Array,
    kalman_gain: Array,
    emission_matrix: Array,
    emission_cov: Array,
) -> Array:
    """Joseph form covariance update: always PSD by construction.

    P_post = (I - K H) P (I - K H)' + K R K'
    """
    D = prior_cov.shape[0]
    I_KH = jnp.eye(D) - kalman_gain @ emission_matrix
    return I_KH @ prior_cov @ I_KH.T + kalman_gain @ emission_cov @ kalman_gain.T
```

Add `use_joseph_form: bool = False` parameter to `_kalman_update` in `kalman.py`. When True, use `_joseph_form_update` instead of `P - K @ S @ K.T`.

### Step 3.3: Run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_kalman.py -v -k "joseph"
conda run -n state_space_practice pytest src/state_space_practice/tests/ -q  # regression
```

### Step 3.4: Commit

```bash
git commit -m "Add Joseph form covariance update for PSD stability"
```

---

## Task 4: Parameter Constraint System

### Step 4.1: Write failing tests

```python
# tests/test_parameter_transforms.py

from state_space_practice.parameter_transforms import (
    POSITIVE, UNIT_INTERVAL, UNCONSTRAINED, PSD_MATRIX,
    ParameterTransform, transform_to_unconstrained, transform_to_constrained,
)

class TestParameterTransforms:
    def test_positive_roundtrip(self):
        x = jnp.array([0.01, 1.0, 100.0])
        unc = POSITIVE.to_unconstrained(x)
        recovered = POSITIVE.to_constrained(unc)
        np.testing.assert_allclose(recovered, x, rtol=1e-5)

    def test_positive_always_positive(self):
        unc = jnp.array([-10.0, 0.0, 10.0])
        x = POSITIVE.to_constrained(unc)
        assert jnp.all(x > 0)

    def test_unit_interval_roundtrip(self):
        x = jnp.array([0.01, 0.5, 0.99])
        recovered = UNIT_INTERVAL.to_constrained(UNIT_INTERVAL.to_unconstrained(x))
        np.testing.assert_allclose(recovered, x, rtol=1e-5)

    def test_unit_interval_bounds(self):
        unc = jnp.linspace(-10, 10, 100)
        x = UNIT_INTERVAL.to_constrained(unc)
        assert jnp.all(x > 0) and jnp.all(x < 1)

    def test_psd_roundtrip(self):
        P = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        unc = PSD_MATRIX.to_unconstrained(P)
        recovered = PSD_MATRIX.to_constrained(unc)
        np.testing.assert_allclose(recovered, P, atol=1e-5)

    def test_psd_always_psd(self):
        unc = jnp.array([0.5, -0.3, 0.1])  # arbitrary reals
        P = PSD_MATRIX.to_constrained(unc)
        eigvals = jnp.linalg.eigvalsh(P)
        assert jnp.all(eigvals > 0)

    def test_gradient_through_positive(self):
        grad_fn = jax.grad(lambda unc: POSITIVE.to_constrained(unc).sum())
        g = grad_fn(jnp.array([0.0, 1.0, -1.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_dict_transform_roundtrip(self):
        spec = {"q": POSITIVE, "beta": POSITIVE, "decay": UNIT_INTERVAL}
        params = {"q": jnp.array(0.01), "beta": jnp.array(2.0), "decay": jnp.array(0.95)}
        unc = transform_to_unconstrained(params, spec)
        recovered = transform_to_constrained(unc, spec)
        for k in params:
            np.testing.assert_allclose(recovered[k], params[k], rtol=1e-5)
```

### Step 4.2: Implement

Create `src/state_space_practice/parameter_transforms.py`:

```python
"""Lightweight parameter constraint transforms for SGD optimization.

Maps between constrained parameter spaces (positive reals, unit interval,
PSD matrices) and unconstrained reals. No TFP dependency — pure JAX.

Usage:
    unc_params = transform_to_unconstrained(params, param_spec)
    params = transform_to_constrained(unc_params, param_spec)
"""

class ParameterTransform(NamedTuple):
    to_unconstrained: Callable[[Array], Array]
    to_constrained: Callable[[Array], Array]

POSITIVE = ParameterTransform(to_unconstrained=jnp.log, to_constrained=jnp.exp)
UNIT_INTERVAL = ParameterTransform(
    to_unconstrained=lambda x: jnp.log(x / (1 - x)),
    to_constrained=jax.nn.sigmoid,
)
UNCONSTRAINED = ParameterTransform(
    to_unconstrained=lambda x: x,
    to_constrained=lambda x: x,
)

def _psd_to_real(P):
    L = jnp.linalg.cholesky(P)
    diag = jnp.log(jnp.diag(L))
    # Return lower triangle with log-diagonal
    return L.at[jnp.diag_indices_from(L)].set(diag)[jnp.tril_indices_from(L)]

def _real_to_psd(flat):
    n = int((-1 + jnp.sqrt(1 + 8 * len(flat))) / 2)
    L = jnp.zeros((n, n)).at[jnp.tril_indices(n)].set(flat)
    L = L.at[jnp.diag_indices(n)].set(jnp.exp(jnp.diag(L)))
    return L @ L.T

PSD_MATRIX = ParameterTransform(to_unconstrained=_psd_to_real, to_constrained=_real_to_psd)

def transform_to_unconstrained(params: dict, spec: dict) -> dict:
    return {k: spec[k].to_unconstrained(v) for k, v in params.items()}

def transform_to_constrained(unc_params: dict, spec: dict) -> dict:
    return {k: spec[k].to_constrained(v) for k, v in unc_params.items()}
```

### Step 4.3: Run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_parameter_transforms.py -v
conda run -n state_space_practice ruff check src/state_space_practice/parameter_transforms.py
```

### Step 4.4: Commit

```bash
git commit -m "Add lightweight parameter constraint transforms for SGD"
```

---

## Task 5: SGD Fitting via Optax

**Depends on:** Task 4 (parameter transforms).

### Step 5.1: Write failing tests

```python
# Add to tests/test_covariate_choice.py

class TestSGDFitting:
    def test_sgd_improves_ll(self):
        # Fit with SGD, check LL improves from initial
        rng = np.random.default_rng(42)
        choices = np.where(rng.random(200) < 0.7, 1, rng.integers(0, 3, size=200))
        model = CovariateChoiceModel(n_options=3)
        lls = model.fit_sgd(choices, num_steps=50)
        assert lls[-1] > lls[0]

    def test_sgd_learns_beta(self):
        # Deterministic choices → high beta without grid search
        choices = np.ones(200, dtype=int)
        model = CovariateChoiceModel(n_options=3)
        model.fit_sgd(choices, num_steps=100)
        assert model.inverse_temperature > 2.0

    def test_sgd_respects_constraints(self):
        # After SGD: process_noise > 0, 0 < decay < 1
        rng = np.random.default_rng(42)
        model = CovariateChoiceModel(n_options=3, init_decay=0.9, learn_decay=True)
        model.fit_sgd(rng.integers(0, 3, size=100), num_steps=50)
        assert model.process_noise > 0
        assert 0 < model.decay < 1

    def test_sgd_matches_em_approximately(self):
        # On same data, SGD and EM should converge to similar LL (within 10%)
        data = simulate_rl_choice_data(n_trials=200, n_options=3, seed=42)
        m_em = CovariateChoiceModel(n_options=3, n_covariates=2)
        m_em.fit(data.choices, covariates=data.covariates, max_iter=20)
        m_sgd = CovariateChoiceModel(n_options=3, n_covariates=2)
        m_sgd.fit_sgd(data.choices, covariates=data.covariates, num_steps=200)
        # LLs should be in the same ballpark
        assert abs(m_em.log_likelihood_ - m_sgd.log_likelihood_) / abs(m_em.log_likelihood_) < 0.1

    def test_sgd_with_covariates(self):
        data = simulate_rl_choice_data(n_trials=200, n_options=3, seed=42)
        model = CovariateChoiceModel(n_options=3, n_covariates=2)
        lls = model.fit_sgd(data.choices, covariates=data.covariates, num_steps=50)
        assert np.isfinite(model.log_likelihood_)

    def test_sgd_verbose(self, caplog):
        import logging
        rng = np.random.default_rng(42)
        model = CovariateChoiceModel(n_options=3)
        with caplog.at_level(logging.INFO):
            model.fit_sgd(rng.integers(0, 3, size=50), num_steps=5, verbose=True)
        assert "SGD step" in caplog.text
```

### Step 5.2: Implement

Add to `CovariateChoiceModel` in `covariate_choice.py`:

```python
def fit_sgd(
    self,
    choices: ArrayLike,
    covariates: Optional[ArrayLike] = None,
    obs_covariates: Optional[ArrayLike] = None,
    optimizer: Optional[Any] = None,
    num_steps: int = 200,
    verbose: bool = False,
) -> list[float]:
    """Fit by minimizing negative marginal LL via gradient descent.

    Uses optax for optimization. Parameters are transformed to
    unconstrained space, optimized, and transformed back.
    """
    import optax
    from state_space_practice.parameter_transforms import (
        POSITIVE, UNIT_INTERVAL, UNCONSTRAINED,
        transform_to_unconstrained, transform_to_constrained,
    )

    # ... validate inputs same as fit() ...

    # Build parameter spec based on what's learnable
    param_spec = {}
    params = {}
    if self.learn_process_noise:
        param_spec["process_noise"] = POSITIVE
        params["process_noise"] = jnp.array(self.process_noise)
    if self.learn_inverse_temperature:
        param_spec["inverse_temperature"] = POSITIVE
        params["inverse_temperature"] = jnp.array(self.inverse_temperature)
    if self.learn_decay:
        param_spec["decay"] = UNIT_INTERVAL
        params["decay"] = jnp.array(self.decay)
    if self.n_covariates > 0:
        param_spec["input_gain"] = UNCONSTRAINED
        params["input_gain"] = self.input_gain_
    if self.n_obs_covariates > 0 and self.learn_obs_weights:
        param_spec["obs_weights"] = UNCONSTRAINED
        params["obs_weights"] = self.obs_weights_

    unc_params = transform_to_unconstrained(params, param_spec)

    @jax.value_and_grad
    def loss_fn(unc_params):
        p = transform_to_constrained(unc_params, param_spec)
        result = covariate_choice_filter(
            choices_arr, self.n_options,
            covariates=covariates_arr,
            input_gain=p.get("input_gain", self.input_gain_),
            obs_covariates=obs_cov_arr,
            obs_weights=p.get("obs_weights", self.obs_weights_),
            process_noise=p.get("process_noise", self.process_noise),
            inverse_temperature=p.get("inverse_temperature", self.inverse_temperature),
            decay=p.get("decay", self.decay),
        )
        return -result.marginal_log_likelihood

    if optimizer is None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(1e-2),
        )
    opt_state = optimizer.init(unc_params)

    log_likelihoods = []
    for step in range(num_steps):
        loss, grads = loss_fn(unc_params)
        updates, opt_state = optimizer.update(grads, opt_state, unc_params)
        unc_params = optax.apply_updates(unc_params, updates)
        log_likelihoods.append(-float(loss))

        if verbose and (step % 10 == 0 or step == num_steps - 1):
            logger.info("SGD step %d: LL=%.2f", step, -float(loss))

    # Store learned parameters
    final_params = transform_to_constrained(unc_params, param_spec)
    # ... update self.process_noise, self.inverse_temperature, etc. from final_params ...

    # Final smoother pass
    self._smoother_result = covariate_choice_smoother(...)
    self.log_likelihood_ = float(self._smoother_result.marginal_log_likelihood)
    self.log_likelihood_history_ = log_likelihoods
    return log_likelihoods
```

Also add `fit_sgd` to `MultinomialChoiceModel` with the same pattern (fewer parameters).

### Step 5.3: Run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_covariate_choice.py -v -k "sgd"
conda run -n state_space_practice pytest src/state_space_practice/tests/ -q  # full regression
```

### Step 5.4: Commit

```bash
git commit -m "Add SGD fitting via optax as alternative to EM"
```

---

## Task 6: Information Form Kalman Filter (Optional)

**Priority:** Low. Implement only if switching model stability requires it after Tasks 1-3.

### Step 6.1: Write failing tests

```python
# tests/test_info_kalman.py

class TestInfoKalmanFilter:
    def test_matches_moment_form(self):
        # Same synthetic data, info form == moment form (atol=1e-5)

    def test_large_initial_uncertainty(self):
        # P_0 = 1e6 * I — moment form may lose precision, info form stable

    def test_high_snr_observation(self):
        # R very small (high SNR) — info form update is well-conditioned

    def test_info_smoother_matches_moment(self):
        # Full filter+smoother in info form == moment form
```

### Step 6.2: Implement

Create `src/state_space_practice/info_kalman.py`:

```python
def info_kalman_filter(
    init_eta, init_precision,
    obs, transition_matrix, process_precision,
    measurement_matrix, measurement_precision,
):
    """Kalman filter in information form (precision parameterization).

    Update (additive, no inverse):
        Λ_post = Λ_pred + H' R^{-1} H
        η_post = η_pred + H' R^{-1} y

    Prediction (requires one inverse):
        P_pred = F Λ^{-1} F' + Q
        Λ_pred = P_pred^{-1}
        η_pred = Λ_pred F Λ^{-1} η
    """
```

### Step 6.3: Run tests

```bash
conda run -n state_space_practice pytest src/state_space_practice/tests/test_info_kalman.py -v
```

### Step 6.4: Commit

```bash
git commit -m "Add information form Kalman filter for numerical stability"
```
