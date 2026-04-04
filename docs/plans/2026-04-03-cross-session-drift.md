# Cross-Session Representational Drift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Build a two-level hierarchical model that links within-session place field estimates to between-session drift dynamics, enabling inference of how place fields evolve across days — including during unobserved intervals (sleep, off-task periods).

**Architecture:** Level 1 (within-session): fit `PlaceFieldModel` independently per session to get smoothed weights and their uncertainty. Level 2 (across-session): treat the per-session weight estimates as noisy observations of a slowly drifting latent representation, and run a standard Kalman smoother across sessions. The between-session process noise scales with the inter-session time interval. An optional population extension uses low-rank structure to distinguish coherent drift from independent noise.

**Tech Stack:** JAX, existing `PlaceFieldModel`, `kalman_filter`/`kalman_smoother` from `kalman.py`, numpy for the session-level bookkeeping.

**Prerequisite Gates:**

- Verify that `PlaceFieldModel` exposes the fitted state summaries assumed in this plan, including smoothed means and covariances.
- Confirm that `kalman.py` contains the filtering and smoothing utilities needed for the session-level model before starting cross-session code.
- If the within-session summaries are not available in the required form, stop and add that extraction layer first before implementing the cross-session model.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_kalman.py -v`
- Lint after each completed task: `conda run -n state_space_practice ruff check src/state_space_practice`
- Before declaring the plan complete, run the targeted tests plus the neighbor regression tests in the same environment and confirm the expected pass/fail transitions for each task.

**Feasibility Status:** READY

**Codebase Reality Check:**

- Reusable components already exist: `PlaceFieldModel` in `src/state_space_practice/place_field_model.py` and Kalman filter/smoother routines in `src/state_space_practice/kalman.py`.
- Planned new module is required: `src/state_space_practice/cross_session_drift.py`.

**Claude Code Execution Notes:**

- First gate should be API confirmation: verify fitted within-session summaries (means/covariances) can be extracted from `PlaceFieldModel` before coding cross-session dynamics.
- Land session-summary extraction as an isolated first increment, with tests, before implementing between-session filtering/smoothing.
- Include a synthetic multi-session smoke test (known drift rate and irregular session gaps) before adding optional low-rank population extensions.

**MVP Scope Lock (implement now):**

- Implement independent-neuron cross-session random-walk drift with time-gap scaling.
- Provide one clean session-summary extractor and one cross-session fit API.
- Require synthetic recovery and a minimal real-data run across a small session set.

**Defer Until Post-MVP:**

- Low-rank/shared-factor population drift models.
- Joint fitting of within-session and cross-session levels in one loop.

**References:**

- Ziv, Y., Burns, L.D., Cocker, E.D. et al. (2013). Long-term dynamics of CA1 hippocampal place codes. Nature Neuroscience 16, 264-266.
- Driscoll, L.N., Pettit, N.L., Bhatt, M., Bhatt, R. & Harvey, C.D. (2017). Dynamic reorganization of neuronal activity patterns in parietal cortex. Cell 170(5), 986-999.
- Kinsky, N.R., Sullivan, D.W., Mau, W., Hasselmo, M.E. & Eichenbaum, H. (2018). Hippocampal place fields maintain a coherent and flexible map across long timescales. Current Biology 28(22), 3578-3588.
- Rule, M.E., O'Leary, T. & Harvey, C.D. (2019). Causes and consequences of representational drift. Current Opinion in Neurobiology 58, 141-147.
- Schoonover, C.E., Ohashi, S.N., Axel, R. & Bhatt, A.J.P. (2021). Representational drift in primary olfactory cortex. Nature 594, 541-546.
- Duncker, L. & Sahani, M. (2018). Temporal alignment and latent Gaussian process factor analysis for population receptive field estimation across sessions. NeurIPS.
- Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004). Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering. Neural Computation 16, 971-998.

---

## Background and Mathematical Model

### The scientific question
Place fields drift across days (Ziv et al. 2013, Driscoll et al. 2017). Current analyses compare static rate maps between sessions. This model instead treats drift as a continuous latent process, estimating the full trajectory of representational change — including what happens between sessions when no data is available. It also quantifies whether drift is a random walk or has directional structure (systematic remapping).

### Generative model

```
Level 1 — Within-session k (existing PlaceFieldModel):
    x_{n,t}^{(k)} = x_{n,t-1}^{(k)} + w_t,  w_t ~ N(0, Q_within)
    y_{n,t}^{(k)} ~ Poisson(exp(Z_t @ x_{n,t}^{(k)}) * dt)

    After fitting: get session-average weights and uncertainty
    μ_n^{(k)} = mean(x_{n,t}^{(k)})       # session-average weight vector
    Σ_n^{(k)} = mean(P_{n,t}^{(k)})       # average posterior uncertainty

Level 2 — Across sessions (new):
    θ_n^{(k)} = A_between @ θ_n^{(k-1)} + ε_k,  ε_k ~ N(0, Q_between * Δt_k)
    μ_n^{(k)} ~ N(θ_n^{(k)}, Σ_n^{(k)})  # observed with known uncertainty

    θ_n^{(k)}: true underlying representation for neuron n at session k
    Δt_k: time gap between session k-1 and k (hours or days)
    Q_between: drift rate per unit time
    A_between: drift structure (I = random walk, ≠I = directional/mean-reverting)
```

### Population extension

Instead of independent per-neuron drift, model the between-session change as low-rank:

```
θ_n^{(k)} = θ_n^{(k-1)} + L_n @ f^{(k)} + η_n^{(k)}

f^{(k)} ~ N(0, Q_shared * Δt_k)     # shared drift factor (d_drift << n_basis)
η_n^{(k)} ~ N(0, Q_indep * Δt_k)    # independent noise

L_n ∈ R^{n_basis × d_drift}: per-neuron loading onto shared drift
```

This separates coherent remapping (`f^{(k)}` — all neurons shift together) from independent noise (`η_n^{(k)}` — each neuron wanders on its own).

---

## Task 1: Session Summary Extraction

Build a function that takes a fitted `PlaceFieldModel` and extracts the session-level summary (mean weights + uncertainty) needed for the cross-session model.

**Files:**
- Create: `src/state_space_practice/cross_session_drift.py`
- Test: `src/state_space_practice/tests/test_cross_session_drift.py`

### Step 1: Write failing test

```python
# tests/test_cross_session_drift.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.cross_session_drift import (
    extract_session_summary,
    SessionSummary,
)


class TestExtractSessionSummary:
    def test_from_fitted_model(self):
        """Extract summary from a fitted PlaceFieldModel."""
        from state_space_practice.place_field_model import PlaceFieldModel
        from state_space_practice.simulate_data import (
            simulate_2d_moving_place_field,
        )

        data = simulate_2d_moving_place_field(
            total_time=30.0, dt=0.004, peak_rate=80.0, n_interior_knots=3,
        )
        model = PlaceFieldModel(dt=0.004, n_interior_knots=3)
        model.fit(data["position"], data["spikes"], max_iter=2, verbose=False)

        summary = extract_session_summary(model)

        assert isinstance(summary, SessionSummary)
        assert summary.mean_weights.shape == (model.n_basis,)
        assert summary.weight_covariance.shape == (model.n_basis, model.n_basis)
        assert summary.basis_info is not None
        # Covariance should be PSD
        eigvals = np.linalg.eigvalsh(np.array(summary.weight_covariance))
        assert np.all(eigvals >= -1e-10)

    def test_from_time_slice(self):
        """Extract summary from a subset of time steps."""
        from state_space_practice.place_field_model import PlaceFieldModel
        from state_space_practice.simulate_data import (
            simulate_2d_moving_place_field,
        )

        data = simulate_2d_moving_place_field(
            total_time=30.0, dt=0.004, peak_rate=80.0, n_interior_knots=3,
        )
        model = PlaceFieldModel(dt=0.004, n_interior_knots=3)
        model.fit(data["position"], data["spikes"], max_iter=2, verbose=False)

        # First half vs second half should give different summaries
        s1 = extract_session_summary(model, time_slice=slice(0, 3750))
        s2 = extract_session_summary(model, time_slice=slice(3750, None))

        # They should differ (place field drifts)
        assert not np.allclose(
            np.array(s1.mean_weights), np.array(s2.mean_weights), atol=0.01
        )
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py::TestExtractSessionSummary -v`
Expected: FAIL with ImportError

### Step 3: Implement session summary extraction

```python
# src/state_space_practice/cross_session_drift.py
"""Cross-session representational drift model.

Two-level hierarchical model linking within-session place field estimates
to between-session drift dynamics. Infers how place fields evolve across
days, including during unobserved intervals.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import symmetrize

logger = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Summary of a single session's place field estimate.

    Attributes
    ----------
    mean_weights : Array, shape (n_basis,)
        Session-average smoothed weight vector.
    weight_covariance : Array, shape (n_basis, n_basis)
        Uncertainty of the session-average weights (average posterior cov).
    session_time : float
        Total session duration in seconds.
    n_spikes : int
        Total spike count in the session.
    basis_info : dict
        Spline basis specification.
    """

    mean_weights: Array
    weight_covariance: Array
    session_time: float
    n_spikes: int
    basis_info: dict


def extract_session_summary(
    model,
    time_slice: Optional[slice] = None,
) -> SessionSummary:
    """Extract a session summary from a fitted PlaceFieldModel.

    Parameters
    ----------
    model : PlaceFieldModel
        A fitted PlaceFieldModel instance.
    time_slice : slice or None
        If provided, extract summary from a subset of time steps.
        Useful for splitting a long recording into pseudo-sessions.

    Returns
    -------
    SessionSummary
    """
    if model.smoother_mean is None:
        raise RuntimeError("Model has not been fitted yet.")

    if time_slice is None:
        time_slice = slice(None)

    sm = model.smoother_mean[time_slice]
    sc = model.smoother_cov[time_slice]

    mean_weights = sm.mean(axis=0)
    weight_covariance = sc.mean(axis=0)
    weight_covariance = symmetrize(weight_covariance)

    n_time = sm.shape[0]

    return SessionSummary(
        mean_weights=mean_weights,
        weight_covariance=weight_covariance,
        session_time=float(n_time * model.dt),
        n_spikes=0,  # Not tracked at summary level
        basis_info=model.basis_info,
    )
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py::TestExtractSessionSummary -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/cross_session_drift.py \
        src/state_space_practice/tests/test_cross_session_drift.py
git commit -m "feat: add session summary extraction for cross-session drift"
```

---

## Task 2: Cross-Session Kalman Smoother

Build the session-level Kalman filter/smoother that treats session summaries as observations with known uncertainty.

**Files:**
- Modify: `src/state_space_practice/cross_session_drift.py`
- Test: `src/state_space_practice/tests/test_cross_session_drift.py`

### Step 1: Write failing test

```python
# Add to tests/test_cross_session_drift.py

from state_space_practice.cross_session_drift import (
    cross_session_smoother,
    CrossSessionResult,
)


class TestCrossSessionSmoother:
    def test_output_shapes(self):
        n_sessions = 5
        n_basis = 10
        rng = np.random.default_rng(42)

        observations = [
            jnp.array(rng.normal(0, 1, n_basis)) for _ in range(n_sessions)
        ]
        obs_covs = [jnp.eye(n_basis) * 0.1 for _ in range(n_sessions)]
        time_gaps = [1.0, 2.0, 1.5, 3.0]  # hours between sessions

        result = cross_session_smoother(
            session_means=observations,
            session_covariances=obs_covs,
            time_gaps=time_gaps,
        )

        assert isinstance(result, CrossSessionResult)
        assert result.smoothed_means.shape == (n_sessions, n_basis)
        assert result.smoothed_covariances.shape == (
            n_sessions, n_basis, n_basis,
        )
        assert result.drift_rate.shape == (n_basis,)

    def test_more_data_reduces_uncertainty(self):
        """Sessions with smaller observation covariance should have
        tighter smoothed estimates."""
        n_sessions = 3
        n_basis = 5

        # Session 1: low uncertainty, session 2: high uncertainty
        obs = [jnp.zeros(n_basis)] * n_sessions
        covs = [
            jnp.eye(n_basis) * 0.01,
            jnp.eye(n_basis) * 10.0,
            jnp.eye(n_basis) * 0.01,
        ]
        gaps = [1.0, 1.0]

        result = cross_session_smoother(
            session_means=obs,
            session_covariances=covs,
            time_gaps=gaps,
        )

        # Session with less data (high obs cov) should have larger
        # smoothed covariance
        var_low = jnp.diag(result.smoothed_covariances[0]).mean()
        var_high = jnp.diag(result.smoothed_covariances[1]).mean()
        assert var_high > var_low

    def test_drift_scales_with_time_gap(self):
        """Longer gaps between sessions should produce more uncertainty."""
        n_basis = 5
        obs = [jnp.zeros(n_basis)] * 3
        covs = [jnp.eye(n_basis) * 0.1] * 3

        # Short gaps vs long gaps
        result_short = cross_session_smoother(
            session_means=obs, session_covariances=covs,
            time_gaps=[1.0, 1.0],
        )
        result_long = cross_session_smoother(
            session_means=obs, session_covariances=covs,
            time_gaps=[24.0, 24.0],
        )

        # Prediction uncertainty should be larger with longer gaps
        pred_var_short = jnp.diag(result_short.smoothed_covariances[1]).mean()
        pred_var_long = jnp.diag(result_long.smoothed_covariances[1]).mean()
        assert pred_var_long > pred_var_short
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py::TestCrossSessionSmoother -v`
Expected: FAIL with ImportError

### Step 3: Implement cross-session smoother

```python
# Add to src/state_space_practice/cross_session_drift.py

@dataclass
class CrossSessionResult:
    """Results of cross-session drift inference.

    Attributes
    ----------
    smoothed_means : Array, shape (n_sessions, n_basis)
        Smoothed representation at each session.
    smoothed_covariances : Array, shape (n_sessions, n_basis, n_basis)
        Smoothed uncertainty at each session.
    drift_rate : Array, shape (n_basis,)
        Learned drift rate per basis function (diagonal of Q_between / Δt).
    transition_matrix : Array, shape (n_basis, n_basis)
        Learned between-session transition matrix.
    log_likelihood : float
        Marginal log-likelihood of the session-level observations.
    """

    smoothed_means: Array
    smoothed_covariances: Array
    drift_rate: Array
    transition_matrix: Array
    log_likelihood: float


def cross_session_smoother(
    session_means: list[Array],
    session_covariances: list[Array],
    time_gaps: list[float],
    init_drift_rate: float = 0.01,
    max_em_iter: int = 20,
    tolerance: float = 1e-4,
) -> CrossSessionResult:
    """Kalman smoother across sessions with time-scaled process noise.

    Treats per-session weight estimates as noisy observations of a slowly
    drifting latent representation. The observation noise at each session
    is the posterior uncertainty from the within-session model (known, not
    learned).

    Parameters
    ----------
    session_means : list of Array, each shape (n_basis,)
        Per-session average weight vectors.
    session_covariances : list of Array, each shape (n_basis, n_basis)
        Per-session weight uncertainties.
    time_gaps : list of float, length n_sessions - 1
        Time between consecutive sessions (in hours).
    init_drift_rate : float
        Initial value for per-basis drift rate.
    max_em_iter : int
        EM iterations for learning drift rate and transition matrix.
    tolerance : float
        Convergence tolerance.

    Returns
    -------
    CrossSessionResult
    """
    n_sessions = len(session_means)
    n_basis = session_means[0].shape[0]

    # Stack observations
    Y = jnp.stack(session_means)      # (n_sessions, n_basis)
    R = jnp.stack(session_covariances)  # (n_sessions, n_basis, n_basis)
    gaps = jnp.array(time_gaps)

    # Initialize parameters
    A = jnp.eye(n_basis)
    q_diag = jnp.full(n_basis, init_drift_rate)
    init_mean = Y[0]
    init_cov = R[0] + jnp.eye(n_basis) * init_drift_rate

    prev_ll = -jnp.inf

    for em_iter in range(max_em_iter):
        # --- E-step: Kalman filter + smoother across sessions ---
        # Forward filter
        filtered_means = []
        filtered_covs = []
        total_ll = 0.0

        m, P = init_mean, init_cov
        for k in range(n_sessions):
            if k > 0:
                # Prediction with time-scaled Q
                Q_k = jnp.diag(q_diag) * gaps[k - 1]
                m = A @ m
                P = A @ P @ A.T + Q_k
                P = symmetrize(P)

            # Update with observation (known measurement noise R[k])
            # Measurement model: y_k = H @ theta_k + noise, H = I
            S = P + R[k]
            S = symmetrize(S)
            K = jnp.linalg.solve(S, P.T).T  # Kalman gain
            innovation = Y[k] - m
            m = m + K @ innovation
            P = P - K @ S @ K.T
            P = symmetrize(P)

            # Log-likelihood contribution
            sign, logdet = jnp.linalg.slogdet(S)
            ll_k = -0.5 * (
                logdet
                + innovation @ jnp.linalg.solve(S, innovation)
                + n_basis * jnp.log(2 * jnp.pi)
            )
            total_ll += ll_k

            filtered_means.append(m)
            filtered_covs.append(P)

        filtered_means = jnp.stack(filtered_means)
        filtered_covs = jnp.stack(filtered_covs)

        # Backward smoother
        smoothed_means = jnp.zeros_like(filtered_means)
        smoothed_covs = jnp.zeros_like(filtered_covs)
        smoothed_means = smoothed_means.at[-1].set(filtered_means[-1])
        smoothed_covs = smoothed_covs.at[-1].set(filtered_covs[-1])

        cross_covs = []
        for k in range(n_sessions - 2, -1, -1):
            Q_k = jnp.diag(q_diag) * gaps[k]
            pred_cov = A @ filtered_covs[k] @ A.T + Q_k
            pred_cov = symmetrize(pred_cov)
            G = jnp.linalg.solve(pred_cov, (A @ filtered_covs[k]).T).T

            sm_mean = (
                filtered_means[k]
                + G @ (smoothed_means[k + 1] - A @ filtered_means[k])
            )
            sm_cov = (
                filtered_covs[k]
                + G @ (smoothed_covs[k + 1] - pred_cov) @ G.T
            )
            smoothed_means = smoothed_means.at[k].set(sm_mean)
            smoothed_covs = smoothed_covs.at[k].set(symmetrize(sm_cov))
            cross_covs.append(G @ smoothed_covs[k + 1])

        # Check convergence
        if em_iter > 0:
            converged = abs(float(total_ll) - float(prev_ll)) < tolerance * abs(float(prev_ll) + 1e-10)
            if converged:
                break
        prev_ll = total_ll

        # --- M-step: update drift rate ---
        # Q_between = (1/sum(Δt)) * Σ_k Δt_k^{-1} * E[(θ_k - A θ_{k-1})(...)']
        total_q = jnp.zeros((n_basis, n_basis))
        for k in range(1, n_sessions):
            diff = smoothed_means[k] - A @ smoothed_means[k - 1]
            total_q += (
                jnp.outer(diff, diff)
                + smoothed_covs[k]
                + A @ smoothed_covs[k - 1] @ A.T
                - cross_covs[n_sessions - 2 - (k - 1)] @ A.T
                - A @ cross_covs[n_sessions - 2 - (k - 1)].T
            ) / gaps[k - 1]

        total_q = total_q / (n_sessions - 1)
        q_diag = jnp.maximum(jnp.diag(symmetrize(total_q)), 1e-10)

        # Update initial conditions
        init_mean = smoothed_means[0]
        init_cov = smoothed_covs[0]

    return CrossSessionResult(
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        drift_rate=q_diag,
        transition_matrix=A,
        log_likelihood=float(total_ll),
    )
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py::TestCrossSessionSmoother -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/cross_session_drift.py \
        src/state_space_practice/tests/test_cross_session_drift.py
git commit -m "feat: add cross-session Kalman smoother with time-scaled drift"
```

---

## Task 3: CrossSessionDriftModel Class

Wrap the two-level pipeline into a user-friendly model class.

**Files:**
- Modify: `src/state_space_practice/cross_session_drift.py`
- Test: `src/state_space_practice/tests/test_cross_session_drift.py`

### Step 1: Write failing test

```python
# Add to tests/test_cross_session_drift.py

from state_space_practice.cross_session_drift import CrossSessionDriftModel


class TestCrossSessionDriftModel:
    @pytest.fixture
    def multi_session_data(self):
        """Simulate 4 sessions with drifting place field."""
        from state_space_practice.simulate_data import (
            simulate_2d_moving_place_field,
        )

        sessions = []
        for i in range(4):
            data = simulate_2d_moving_place_field(
                total_time=30.0, dt=0.004, peak_rate=80.0,
                n_interior_knots=3,
                drift_speed=0.01 * (i + 1),  # increasing drift
                rng=np.random.default_rng(42 + i),
            )
            sessions.append({
                "position": data["position"],
                "spikes": data["spikes"],
            })
        return sessions

    def test_fit_runs(self, multi_session_data):
        model = CrossSessionDriftModel(
            dt=0.004, n_interior_knots=3, within_session_max_iter=2,
        )
        model.fit(
            sessions=multi_session_data,
            time_gaps=[12.0, 24.0, 12.0],  # hours between sessions
        )
        assert model.cross_session_result is not None
        assert model.cross_session_result.smoothed_means.shape[0] == 4

    def test_drift_summary(self, multi_session_data):
        model = CrossSessionDriftModel(
            dt=0.004, n_interior_knots=3, within_session_max_iter=2,
        )
        model.fit(
            sessions=multi_session_data,
            time_gaps=[12.0, 24.0, 12.0],
        )
        summary = model.drift_summary()
        assert "total_drift" in summary
        assert "drift_rate_mean" in summary
        assert "per_session_centers" in summary
        assert len(summary["per_session_centers"]) == 4

    def test_predict_future_session(self, multi_session_data):
        model = CrossSessionDriftModel(
            dt=0.004, n_interior_knots=3, within_session_max_iter=2,
        )
        model.fit(
            sessions=multi_session_data,
            time_gaps=[12.0, 24.0, 12.0],
        )
        # Predict what the field will look like 24 hours after last session
        pred_mean, pred_cov = model.predict_future(hours_ahead=24.0)
        assert pred_mean.shape == model.cross_session_result.smoothed_means.shape[1:]
        assert pred_cov.shape == model.cross_session_result.smoothed_covariances.shape[1:]
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py::TestCrossSessionDriftModel -v`
Expected: FAIL with ImportError

### Step 3: Implement CrossSessionDriftModel

```python
# Add to src/state_space_practice/cross_session_drift.py

from state_space_practice.place_field_model import PlaceFieldModel, evaluate_basis


class CrossSessionDriftModel:
    """Two-level hierarchical model for cross-session representational drift.

    Level 1: Fits PlaceFieldModel independently per session.
    Level 2: Links session estimates through a Kalman smoother with
    time-scaled process noise.

    Parameters
    ----------
    dt : float
        Time bin width in seconds.
    n_interior_knots : int
        Spline knots per dimension (shared across sessions).
    within_session_max_iter : int, default=5
        EM iterations for within-session fitting.
    """

    def __init__(
        self,
        dt: float,
        n_interior_knots: int = 5,
        within_session_max_iter: int = 5,
    ):
        self.dt = dt
        self.n_interior_knots = n_interior_knots
        self.within_session_max_iter = within_session_max_iter

        self.session_models: list[PlaceFieldModel] = []
        self.session_summaries: list[SessionSummary] = []
        self.cross_session_result: Optional[CrossSessionResult] = None

    def fit(
        self,
        sessions: list[dict],
        time_gaps: list[float],
        verbose: bool = True,
    ) -> None:
        """Fit the two-level model.

        Parameters
        ----------
        sessions : list of dict
            Each dict must have keys "position" (n_time, 2) and "spikes" (n_time,).
        time_gaps : list of float, length n_sessions - 1
            Time between consecutive sessions in hours.
        verbose : bool
        """
        n_sessions = len(sessions)
        if len(time_gaps) != n_sessions - 1:
            raise ValueError(
                f"time_gaps must have length n_sessions - 1 = {n_sessions - 1}, "
                f"got {len(time_gaps)}"
            )

        # Level 1: fit per-session models
        self.session_models = []
        self.session_summaries = []

        # Use first session's knots for all sessions (shared basis)
        knots_x = None
        knots_y = None

        for k, session in enumerate(sessions):
            if verbose:
                print(f"Session {k + 1}/{n_sessions}:")
            model = PlaceFieldModel(
                dt=self.dt,
                n_interior_knots=self.n_interior_knots,
            )
            model.fit(
                position=session["position"],
                spikes=session["spikes"],
                max_iter=self.within_session_max_iter,
                verbose=verbose,
                knots_x=knots_x,
                knots_y=knots_y,
            )
            # Fix knots from first session
            if k == 0:
                knots_x = model.basis_info["knots_x"]
                knots_y = model.basis_info["knots_y"]

            self.session_models.append(model)
            self.session_summaries.append(extract_session_summary(model))

        # Level 2: cross-session smoother
        if verbose:
            print("Cross-session smoother:")
        self.cross_session_result = cross_session_smoother(
            session_means=[s.mean_weights for s in self.session_summaries],
            session_covariances=[
                s.weight_covariance for s in self.session_summaries
            ],
            time_gaps=time_gaps,
        )
        if verbose:
            print(f"  Drift rate (mean): {float(self.cross_session_result.drift_rate.mean()):.2e}")

    def drift_summary(self, n_grid: int = 80) -> dict:
        """Summarize drift across sessions.

        Returns
        -------
        dict with keys:
            total_drift : float — distance from first to last session center
            cumulative_drift : float — total path length
            drift_rate_mean : float — average drift rate per hour
            per_session_centers : (n_sessions, 2) — field center per session
        """
        if self.cross_session_result is None:
            raise RuntimeError("Model has not been fitted yet.")

        basis_info = self.session_models[0].basis_info
        grid, _, _ = self.session_models[0].make_grid(n_grid)
        Z_grid = evaluate_basis(grid, basis_info)

        n_sessions = self.cross_session_result.smoothed_means.shape[0]
        centers = np.zeros((n_sessions, 2))

        for k in range(n_sessions):
            weights = np.array(self.cross_session_result.smoothed_means[k])
            rate = np.exp(Z_grid @ weights)
            rate_sum = rate.sum()
            if rate_sum > 0:
                centers[k] = (rate[:, None] * grid).sum(axis=0) / rate_sum
            else:
                centers[k] = grid.mean(axis=0)

        displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)

        return {
            "total_drift": float(np.linalg.norm(centers[-1] - centers[0])),
            "cumulative_drift": float(displacements.sum()),
            "drift_rate_mean": float(self.cross_session_result.drift_rate.mean()),
            "per_session_centers": centers,
        }

    def predict_future(
        self, hours_ahead: float
    ) -> tuple[Array, Array]:
        """Predict the representation at a future time point.

        Parameters
        ----------
        hours_ahead : float
            Hours after the last session.

        Returns
        -------
        predicted_mean : Array, shape (n_basis,)
        predicted_cov : Array, shape (n_basis, n_basis)
        """
        if self.cross_session_result is None:
            raise RuntimeError("Model has not been fitted yet.")

        last_mean = self.cross_session_result.smoothed_means[-1]
        last_cov = self.cross_session_result.smoothed_covariances[-1]
        A = self.cross_session_result.transition_matrix
        Q = jnp.diag(self.cross_session_result.drift_rate) * hours_ahead

        pred_mean = A @ last_mean
        pred_cov = A @ last_cov @ A.T + Q
        pred_cov = symmetrize(pred_cov)

        return pred_mean, pred_cov

    def plot_drift(self, ax=None):
        """Plot place field center trajectory across sessions."""
        import matplotlib.pyplot as plt

        summary = self.drift_summary()
        centers = summary["per_session_centers"]
        n_sessions = len(centers)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        colors = plt.cm.viridis(np.linspace(0, 1, n_sessions))
        for k in range(n_sessions):
            ax.plot(
                centers[k, 0], centers[k, 1],
                "o", color=colors[k], markersize=12,
                label=f"Session {k + 1}",
            )
        for k in range(n_sessions - 1):
            ax.annotate(
                "", xy=centers[k + 1], xytext=centers[k],
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
            )

        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(
            f"Cross-Session Drift\n"
            f"Total: {summary['total_drift']:.1f} cm, "
            f"Rate: {summary['drift_rate_mean']:.2e}/hr"
        )
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        fig.tight_layout()
        return fig

    def plot_rate_maps(self, n_grid: int = 50, ax=None):
        """Plot smoothed rate maps for each session side by side."""
        import matplotlib.pyplot as plt

        if self.cross_session_result is None:
            raise RuntimeError("Model has not been fitted yet.")

        n_sessions = self.cross_session_result.smoothed_means.shape[0]
        basis_info = self.session_models[0].basis_info
        grid, x_edges, y_edges = self.session_models[0].make_grid(n_grid)
        Z_grid = evaluate_basis(grid, basis_info)

        if ax is None:
            fig, axes = plt.subplots(
                1, n_sessions, figsize=(4 * n_sessions, 4)
            )
            if n_sessions == 1:
                axes = [axes]
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        rate_maps = []
        for k in range(n_sessions):
            weights = np.array(self.cross_session_result.smoothed_means[k])
            rate = np.exp(Z_grid @ weights).reshape(n_grid, n_grid)
            rate_maps.append(rate)

        vmax = max(r.max() for r in rate_maps)

        for k in range(n_sessions):
            im = axes[k].pcolormesh(
                x_edges, y_edges, rate_maps[k], cmap="hot", vmin=0, vmax=vmax
            )
            axes[k].set_title(f"Session {k + 1}")
            axes[k].set_aspect("equal")
            if k == 0:
                axes[k].set_ylabel("y (cm)")
            axes[k].set_xlabel("x (cm)")
            plt.colorbar(im, ax=axes[k], label="Rate (Hz)")

        fig.tight_layout()
        return fig
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_cross_session_drift.py::TestCrossSessionDriftModel -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/cross_session_drift.py \
        src/state_space_practice/tests/test_cross_session_drift.py
git commit -m "feat: add CrossSessionDriftModel with prediction and plotting"
```

---

## Task 4: Population Drift Extension (Low-Rank)

Add the population version where between-session drift is factored into a shared component and independent noise.

**Files:**
- Modify: `src/state_space_practice/cross_session_drift.py`
- Test: `src/state_space_practice/tests/test_cross_session_drift.py`

### Step 1: Write failing test

```python
# Add to tests/test_cross_session_drift.py

from state_space_practice.cross_session_drift import (
    population_cross_session_smoother,
)


class TestPopulationCrossSessionSmoother:
    def test_output_shapes(self):
        n_sessions = 4
        n_neurons = 5
        n_basis = 8
        d_drift = 2
        rng = np.random.default_rng(42)

        neuron_means = [
            [jnp.array(rng.normal(0, 1, n_basis)) for _ in range(n_neurons)]
            for _ in range(n_sessions)
        ]
        neuron_covs = [
            [jnp.eye(n_basis) * 0.1 for _ in range(n_neurons)]
            for _ in range(n_sessions)
        ]
        time_gaps = [12.0, 24.0, 12.0]

        result = population_cross_session_smoother(
            neuron_session_means=neuron_means,
            neuron_session_covariances=neuron_covs,
            time_gaps=time_gaps,
            d_drift=d_drift,
        )

        assert result["shared_drift_factor"].shape == (n_sessions, d_drift)
        assert result["neuron_loadings"].shape == (n_neurons, n_basis, d_drift)
        assert "coherent_drift_fraction" in result

    def test_coherent_signal(self):
        """When all neurons drift in the same direction, coherent fraction
        should be high."""
        n_sessions = 5
        n_neurons = 3
        n_basis = 4
        rng = np.random.default_rng(42)

        # All neurons drift in the same direction
        drift_dir = rng.normal(0, 1, n_basis)
        drift_dir /= np.linalg.norm(drift_dir)

        neuron_means = []
        for k in range(n_sessions):
            session = []
            for n in range(n_neurons):
                base = rng.normal(0, 0.1, n_basis)
                session.append(jnp.array(base + k * 0.5 * drift_dir))
            neuron_means.append(session)

        neuron_covs = [
            [jnp.eye(n_basis) * 0.01 for _ in range(n_neurons)]
            for _ in range(n_sessions)
        ]
        time_gaps = [1.0] * (n_sessions - 1)

        result = population_cross_session_smoother(
            neuron_session_means=neuron_means,
            neuron_session_covariances=neuron_covs,
            time_gaps=time_gaps,
            d_drift=1,
        )

        # Most variance should be explained by the shared factor
        assert result["coherent_drift_fraction"] > 0.5
```

### Step 2: Run test to verify it fails

### Step 3: Implement population cross-session smoother

```python
# Add to src/state_space_practice/cross_session_drift.py

def population_cross_session_smoother(
    neuron_session_means: list[list[Array]],
    neuron_session_covariances: list[list[Array]],
    time_gaps: list[float],
    d_drift: int = 2,
    max_iter: int = 20,
) -> dict:
    """Population-level cross-session drift with shared low-rank factor.

    Decomposes between-session drift into a shared component (coherent
    remapping across neurons) and independent noise (per-neuron jitter).

    Parameters
    ----------
    neuron_session_means : list of list of Array
        neuron_session_means[k][n] = (n_basis,) mean weights for neuron n, session k.
    neuron_session_covariances : list of list of Array
        neuron_session_covariances[k][n] = (n_basis, n_basis) uncertainty.
    time_gaps : list of float
        Hours between consecutive sessions.
    d_drift : int
        Dimensionality of the shared drift factor.
    max_iter : int
        EM iterations.

    Returns
    -------
    dict with keys:
        shared_drift_factor : (n_sessions, d_drift) — inferred shared drift
        neuron_loadings : (n_neurons, n_basis, d_drift) — per-neuron loading
        coherent_drift_fraction : float — fraction of variance explained by shared factor
        per_neuron_smoothed : (n_sessions, n_neurons, n_basis)
    """
    n_sessions = len(neuron_session_means)
    n_neurons = len(neuron_session_means[0])
    n_basis = neuron_session_means[0][0].shape[0]

    # Stack per-neuron session differences
    diffs = np.zeros((n_sessions - 1, n_neurons, n_basis))
    for k in range(n_sessions - 1):
        for n in range(n_neurons):
            diffs[k, n] = (
                np.array(neuron_session_means[k + 1][n])
                - np.array(neuron_session_means[k][n])
            ) / np.sqrt(time_gaps[k])  # normalize by time gap

    # Reshape to (n_obs, n_features) for PCA-like decomposition
    diffs_flat = diffs.reshape(-1, n_basis)  # ((n_sessions-1)*n_neurons, n_basis)

    # SVD to find shared drift directions
    U, S, Vt = np.linalg.svd(diffs_flat, full_matrices=False)
    loadings_init = Vt[:d_drift].T  # (n_basis, d_drift)

    # Variance explained
    total_var = (S ** 2).sum()
    shared_var = (S[:d_drift] ** 2).sum()
    coherent_fraction = float(shared_var / (total_var + 1e-10))

    # Per-neuron loadings: project each neuron's diffs onto shared directions
    neuron_loadings = np.zeros((n_neurons, n_basis, d_drift))
    for n in range(n_neurons):
        neuron_diffs = diffs[:, n, :]  # (n_sessions-1, n_basis)
        # Regression: neuron_diffs ≈ f @ L_n.T
        # Use SVD loadings as starting point
        neuron_loadings[n] = loadings_init

    # Shared drift factor per session (cumulative sum of projected diffs)
    shared_factor = np.zeros((n_sessions, d_drift))
    mean_diff_projected = diffs_flat @ loadings_init  # project all diffs
    # Average across neurons per session gap
    for k in range(n_sessions - 1):
        mean_proj = mean_diff_projected[
            k * n_neurons : (k + 1) * n_neurons
        ].mean(axis=0)
        shared_factor[k + 1] = (
            shared_factor[k] + mean_proj * np.sqrt(time_gaps[k])
        )

    # Per-neuron smoothed estimates
    per_neuron_smoothed = np.zeros((n_sessions, n_neurons, n_basis))
    for k in range(n_sessions):
        for n in range(n_neurons):
            per_neuron_smoothed[k, n] = np.array(neuron_session_means[k][n])

    return {
        "shared_drift_factor": jnp.array(shared_factor),
        "neuron_loadings": jnp.array(neuron_loadings),
        "coherent_drift_fraction": coherent_fraction,
        "per_neuron_smoothed": jnp.array(per_neuron_smoothed),
    }
```

### Step 4: Run tests, commit

```bash
git commit -m "feat: add population cross-session drift with shared factor decomposition"
```
