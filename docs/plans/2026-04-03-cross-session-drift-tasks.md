# Cross-Session Representational Drift — Task Breakdown

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Implement the cross-session drift model described in `docs/plans/2026-04-03-cross-session-drift.md`.

**Design doc:** `docs/plans/2026-04-03-cross-session-drift.md`

**Key files:**

- Create: `src/state_space_practice/cross_session_drift.py`
- Create: `src/state_space_practice/tests/test_cross_session_drift.py`
- Reference: `src/state_space_practice/place_field_model.py` (`PlaceFieldModel`, `evaluate_basis`)
- Reference: `src/state_space_practice/kalman.py` (`symmetrize`)

**Prerequisite Gates:**

- Verify `PlaceFieldModel` exposes `smoother_mean`, `smoother_cov`, `basis_info`, `dt`, `make_grid()` after fitting.
- Verify `PlaceFieldModel.fit()` accepts `knots_x` and `knots_y` (needed for shared basis across sessions).
- If `PlaceFieldModel` does not expose these, update it first.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_kalman.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

**Critical design decisions:**

- **Shared spline basis across sessions:** All sessions must use the same knots and bounds so that weight vectors are comparable. The first session's knots are used for all subsequent sessions (passed via `knots_x`/`knots_y` to `PlaceFieldModel.fit()`).
- **Session-level observations have known noise:** The observation noise at each session is the within-session posterior uncertainty (`smoother_cov.mean(axis=0)`), not a learned parameter. This is the key hierarchical structure — the within-session model provides calibrated uncertainty to the between-session model.
- **Time-gap scaling:** The between-session process noise is `Q_between * Δt_k`, where `Δt_k` is the time gap in hours between sessions k-1 and k. This assumes drift rate is constant per unit time.
- **A = I by default:** The between-session transition is a random walk. Learning A tests whether drift is mean-reverting (|eigenvalues| < 1) or directional, but this is deferred to post-MVP.
- **The session-level Kalman smoother is NOT the same as the within-session smoother.** The within-session smoother operates at the dt timescale (150k steps). The between-session smoother operates at the session timescale (5-20 steps). The observations are the session-averaged weight vectors with known covariance.

**Known limitations:**

- If the within-session `PlaceFieldModel` is poorly calibrated (overconfident or underconfident posterior), the between-session smoother will give wrong results. This is inherent to the two-stage approach.
- With very few sessions (< 5), the drift rate `Q_between` is poorly estimated. The model still works but the uncertainty is large.
- The model assumes drift is the same for all neurons (shared `Q_between`). If drift rates vary per neuron, this is a population-level average.

**Identifiability notes:**

- The between-session model is a standard linear-Gaussian state-space model (Kalman filter/smoother). It is identifiable as long as:
  - At least 2 sessions exist (otherwise Q_between is undefined)
  - The within-session uncertainty is not infinite (otherwise the observations are uninformative)
  - The drift rate is not exactly zero (otherwise the model reduces to a fixed mean)
- With the population extension (low-rank shared drift), the shared factor is identifiable only up to rotation. The loadings matrix L is not uniquely identifiable — only `L @ L.T` is. This is standard for factor models and does not affect the scientific conclusions (you care about the explained variance, not the exact factors).

**MVP Scope Lock:**

- Implement: `SessionSummary`, `extract_session_summary`, `cross_session_smoother`, `CrossSessionDriftModel`.
- Require: synthetic 4-session drift recovery test (known drift rate).
- Require: drift rate increases with longer time gaps.
- Require: more data (smaller observation cov) → tighter smoothed estimates.

**Defer:**

- Population low-rank extension (`population_cross_session_smoother`)
- Learning A (testing mean-reversion)
- Joint within+between session fitting

---

## Task 1: Session Summary Extraction

Implement `SessionSummary` dataclass and `extract_session_summary` function.

### Tests to write:

```python
class TestExtractSessionSummary:
    def test_from_fitted_model(self):
        # Fit PlaceFieldModel on short simulation
        # Extract summary: check mean_weights shape, weight_covariance shape
        # Covariance should be PSD

    def test_from_time_slice(self):
        # Extract summaries from first half and second half
        # With drifting field, they should differ

    def test_covariance_is_psd(self):
        # All eigenvalues >= -1e-10

    def test_unfitted_model_raises(self):
        # RuntimeError if model not fitted

    def test_basis_info_preserved(self):
        # summary.basis_info matches model.basis_info
```

### Implementation:

```python
@dataclass
class SessionSummary:
    mean_weights: Array          # (n_basis,)
    weight_covariance: Array     # (n_basis, n_basis)
    session_time: float          # seconds
    n_spikes: int
    basis_info: dict

def extract_session_summary(model, time_slice=None) -> SessionSummary:
    # Average smoother_mean and smoother_cov over the time slice
    # Symmetrize the covariance
```

### Verification checkpoint:

- [ ] All tests pass
- [ ] Covariance is PSD for multiple random seeds
- [ ] Summary from full session vs time slice gives different means
- [ ] `ruff check` passes

### Commit:

```bash
git commit -m "Add session summary extraction for cross-session drift"
```

---

## Task 2: Cross-Session Kalman Smoother

Implement `CrossSessionResult` and `cross_session_smoother`.

### Tests to write:

```python
class TestCrossSessionSmoother:
    def test_output_shapes(self):
        # 5 sessions, 10 basis functions
        # smoothed_means: (5, 10), smoothed_covariances: (5, 10, 10)
        # drift_rate: (10,)

    def test_more_data_reduces_uncertainty(self):
        # Session with small observation cov → tighter smoothed estimate
        # Session with large observation cov → looser

    def test_drift_scales_with_time_gap(self):
        # Longer gap → more prediction uncertainty
        # Compare short gaps vs long gaps

    def test_single_session_returns_observation(self):
        # With 1 session, smoothed = observation (no drift to estimate)
        # Should handle gracefully (Q_between undefined, return input)

    def test_two_sessions_minimal(self):
        # Minimum viable: 2 sessions, 1 gap
        # Smoothed means should be between the two observations

    def test_drift_rate_is_positive(self):
        # All elements of drift_rate should be > 0

    def test_em_convergence(self):
        # Log-likelihood should be finite and improve across EM iterations
```

### Critical: Drift recovery test

```python
def test_recovers_known_drift_rate(self):
    """Simulate sessions with known drift and verify recovery."""
    rng = np.random.default_rng(42)
    n_sessions = 8
    n_basis = 5
    true_drift_rate = 0.1  # per hour

    # Simulate: weight vector drifts with known rate
    time_gaps = [12.0, 24.0, 12.0, 48.0, 12.0, 24.0, 12.0]  # hours
    true_weights = [np.zeros(n_basis)]
    for gap in time_gaps:
        prev = true_weights[-1]
        drift = rng.normal(0, np.sqrt(true_drift_rate * gap), n_basis)
        true_weights.append(prev + drift)

    # Observe with small noise
    obs_cov = np.eye(n_basis) * 0.01
    session_means = [jnp.array(w + rng.normal(0, 0.1, n_basis)) for w in true_weights]
    session_covs = [jnp.array(obs_cov) for _ in range(n_sessions)]

    result = cross_session_smoother(
        session_means=session_means,
        session_covariances=session_covs,
        time_gaps=time_gaps,
    )

    # Recovered drift rate should be in the right ballpark
    assert 0.01 < float(result.drift_rate.mean()) < 1.0
    # Smoothed means should be close to true weights
    for k in range(n_sessions):
        error = np.linalg.norm(
            np.array(result.smoothed_means[k]) - true_weights[k]
        )
        assert error < 2.0  # reasonable for noisy recovery
```

### Implementation notes:

The cross-session smoother is a standard Kalman filter/smoother but implemented manually (not using `kalman_filter` from `kalman.py`) because:
- The observation model is `y_k = θ_k + noise` with `H = I` (identity measurement)
- The observation noise `R_k` varies per session (heteroscedastic)
- The process noise `Q_k = Q_between * Δt_k` varies per gap
- With only 5-20 sessions, there's no need for `lax.scan` — a Python loop is clearer

The M-step for `Q_between`:
```
Q_hat = (1 / Σ_k Δt_k) * Σ_{k=1}^{K-1} (1/Δt_k) * [
    E[(θ_k - θ_{k-1})(θ_k - θ_{k-1})' | data]
]
```
where the expectation uses smoother means and covariances. Take the diagonal and clamp to positive.

### Verification checkpoint:

- [ ] All tests pass
- [ ] Drift recovery test: estimated rate within 10x of true rate
- [ ] Time-gap scaling: longer gaps → larger prediction uncertainty
- [ ] With homogeneous gaps: drift rate is consistent
- [ ] `ruff check` passes

### Commit:

```bash
git commit -m "Add cross-session Kalman smoother with time-scaled drift"
```

---

## Task 3: CrossSessionDriftModel Class

Wrap the two-level pipeline into a user-facing model class.

### Tests to write:

```python
class TestCrossSessionDriftModel:
    @pytest.fixture
    def multi_session_data(self):
        # 4 short simulations with different drift seeds

    def test_fit_runs(self, multi_session_data):
        # model.fit(sessions, time_gaps) completes without error

    def test_shared_knots_across_sessions(self, multi_session_data):
        # All session models should use the same knots

    def test_drift_summary(self, multi_session_data):
        # Returns dict with total_drift, cumulative_drift, drift_rate_mean, centers

    def test_predict_future(self, multi_session_data):
        # Predict 24 hours ahead: returns mean and cov with correct shapes
        # Predicted cov should be larger than last session's cov

    def test_plot_drift(self, multi_session_data):
        # Smoke test: returns figure

    def test_plot_rate_maps(self, multi_session_data):
        # Smoke test: returns figure with n_sessions panels

    def test_invalid_time_gaps_length(self, multi_session_data):
        # time_gaps length != n_sessions - 1 should raise ValueError

    def test_repr(self):
        # Shows fitted status and drift rate
```

### Critical: Shared basis test

```python
def test_all_sessions_use_same_basis(self, multi_session_data):
    """All within-session models must use identical knots."""
    model = CrossSessionDriftModel(dt=0.004, n_interior_knots=3)
    model.fit(sessions=multi_session_data, time_gaps=[12.0, 24.0, 12.0])

    knots_x_0 = model.session_models[0].basis_info["knots_x"]
    for k in range(1, len(model.session_models)):
        np.testing.assert_array_equal(
            model.session_models[k].basis_info["knots_x"],
            knots_x_0,
        )
```

### Methods to implement:

- `__init__(dt, n_interior_knots, within_session_max_iter)`
- `__repr__`
- `fit(sessions, time_gaps, verbose)` — Level 1 (per-session PlaceFieldModel) then Level 2 (cross-session smoother)
- `drift_summary(n_grid)` — total/cumulative drift, drift rate, per-session centers
- `predict_future(hours_ahead)` — extrapolate from last session
- `plot_drift(ax)` — center trajectory across sessions
- `plot_rate_maps(n_grid, ax)` — side-by-side smoothed rate maps per session

### Verification checkpoint:

- [ ] All tests pass
- [ ] Shared knots test passes
- [ ] End-to-end: 4 simulated sessions → fit → drift_summary → plot
- [ ] `predict_future` covariance grows with hours_ahead
- [ ] `ruff check` passes
- [ ] Neighbor regression tests pass

### Commit:

```bash
git commit -m "Add CrossSessionDriftModel with prediction and plotting"
```
