# Position Decoding from Spikes via Laplace-EKF Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Build a model that decodes the animal's 2D position from population spiking activity using a point-process state-space model with Laplace-EKF, where the latent state is the animal's position and the observation model is the neurons' place-field tuning curves.

**Architecture:** The latent state is `[x, y, vx, vy]` (position + velocity), evolving via constant-velocity dynamics. Spike observations from N neurons are Poisson with rates determined by each neuron's place field evaluated at the current position estimate. The place fields can be either pre-estimated (from a training period) or learned jointly via EM. The Laplace-EKF handles the nonlinear Poisson observation model. This is the natural inverse of `PlaceFieldModel` — encoding maps spikes → tuning, decoding maps tuning + spikes → position.

**Tech Stack:** JAX, existing `_point_process_laplace_update` from `point_process_kalman.py`, existing `build_2d_spline_basis`/`evaluate_basis` from `place_field_model.py`, multi-neuron support already in the filter.

**Prerequisite Gates:**

- Verify that `point_process_kalman.py`, `place_field_model.py`, and the referenced helper symbols exist in the current repository before editing code.
- Reconcile any mismatch between this plan and the checked-in APIs before implementation; do not guess missing helper behavior.
- If a task requires a Jacobian or interpolation helper that is not present, add that support in the smallest validated step before continuing to downstream decoder tasks.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v`
- Lint after each completed task: `conda run -n state_space_practice ruff check src/state_space_practice`
- Before declaring the plan complete, run the targeted tests plus the neighbor regression tests in the same environment and confirm the expected pass/fail transitions for each task.

**Feasibility Status:** READY

**Codebase Reality Check:**

- Reusable symbols already exist: `_point_process_laplace_update` in `src/state_space_practice/point_process_kalman.py`, spline basis helpers in `src/state_space_practice/place_field_model.py`, and Kalman utility primitives in `src/state_space_practice/kalman.py`.
- Planned new module is still required: `src/state_space_practice/position_decoder.py`.

**Claude Code Execution Notes:**

- Start with the smallest independent slice: implement and test `build_position_dynamics` first, then layer observation updates.
- Add a smoke script/notebook check immediately after Task 2: decode a synthetic 2D trajectory and require finite outputs plus bounded RMSE before moving to downstream tasks.
- If Jacobian/interpolation helpers are missing during implementation, pause and land those helpers with focused tests before continuing decoder logic.

**MVP Scope Lock (implement now):**

- Implement only a fixed place-field decoder using precomputed place fields (no joint place-field learning in this plan).
- Support a single stable API path first: position filter + smoother for 2D position and velocity.
- Require one synthetic end-to-end decode benchmark and one lightweight real-data smoke run.

**Defer Until Post-MVP:**

- Alternative place-field parameterizations beyond the initial fixed representation.
- Advanced decoder variants (multi-model or adaptive transitions).

**References:**

- Eden, U.T., Frank, L.M., Barbieri, R., Solo, V. & Brown, E.N. (2004). Dynamic Analysis of Neural Encoding by Point Process Adaptive Filtering. Neural Computation 16, 971-998.
- Brown, E.N., Frank, L.M., Tang, D., Quirk, M.C. & Wilson, M.A. (1998). A statistical paradigm for neural spike train decoding applied to position prediction from ensemble firing patterns of rat hippocampal place cells. J Neuroscience 18(18), 7411-7425.
- Barbieri, R., Frank, L.M., Quirk, M.C., Solo, V. & Brown, E.N. (2004). Construction and analysis of non-Poisson stimulus-response models of neural spiking activity. J Neurosci Methods 105, 25-37.
- Deng, X., Cai, C., Bhatt, K.K. & Frank, L.M. (2015). Clusterless decoding of position from multiunit activity using a marked point process filter. Neural Computation 27(7), 1438-1460.

---

## Background and Mathematical Model

### The scientific question
Given spike trains from a population of hippocampal neurons with known place fields, where is the animal? This is the fundamental decoding problem in hippocampal neuroscience. The state-space approach gives moment-by-moment position estimates with uncertainty, naturally handles missing data (time bins with no spikes), and provides a principled framework for online (causal) vs offline (smoothed) decoding.

### Generative model

```
Latent state (position + velocity):
    z_t = [x_t, y_t, vx_t, vy_t]

Dynamics (constant velocity with noise):
    z_t = A @ z_{t-1} + w_t,  w_t ~ N(0, Q)

    A = [1  0  dt  0 ]     Q = [q_pos   0     0     0   ]
        [0  1  0   dt]         [0     q_pos   0     0   ]
        [0  0  1   0 ]         [0     0     q_vel   0   ]
        [0  0  0   1 ]         [0     0     0     q_vel ]

Spike observations (N neurons, Poisson):
    y_{n,t} ~ Poisson(λ_n(x_t, y_t) * dt)

    log(λ_n(x, y)) = f_n(x, y)   # neuron n's place field (log-rate map)
```

The key nonlinearity: the firing rate depends on position through each neuron's place field `f_n(x, y)`. The Laplace-EKF linearizes this around the current position estimate at each time step.

### Place field representations

The place fields `f_n` can be represented in two ways:

1. **Spline-based (from PlaceFieldModel):** `log(λ_n) = Z(x,y) @ w_n` where `Z` is the spline basis and `w_n` are learned weights. The Jacobian is `dlog(λ_n)/dz = (dZ/d[x,y]) @ w_n`, which requires the derivative of the spline basis w.r.t. position.

2. **Kernel density estimate (KDE):** Pre-compute `f_n` on a grid, then interpolate. Simpler, no spline derivative needed — just use finite differences for the Jacobian.

We'll use approach 2 (KDE on grid) for simplicity and because it works with any place field estimation method, not just our spline model.

---

## Task 1: Position Dynamics Model

Build the constant-velocity transition matrix and process noise for 2D position tracking.

**Files:**
- Create: `src/state_space_practice/position_decoder.py`
- Test: `src/state_space_practice/tests/test_position_decoder.py`

### Step 1: Write failing test

```python
# tests/test_position_decoder.py
import jax.numpy as jnp
import numpy as np
import pytest

from state_space_practice.position_decoder import (
    build_position_dynamics,
)


class TestBuildPositionDynamics:
    def test_constant_velocity_shapes(self):
        A, Q = build_position_dynamics(dt=0.004, q_pos=1.0, q_vel=10.0)
        assert A.shape == (4, 4)
        assert Q.shape == (4, 4)

    def test_constant_velocity_prediction(self):
        """Starting at (10, 20) with velocity (5, -3),
        after one step should be at (10 + 5*dt, 20 - 3*dt)."""
        dt = 0.01
        A, _ = build_position_dynamics(dt=dt)
        z = jnp.array([10.0, 20.0, 5.0, -3.0])
        z_next = A @ z
        np.testing.assert_allclose(z_next[0], 10.0 + 5.0 * dt)
        np.testing.assert_allclose(z_next[1], 20.0 - 3.0 * dt)
        np.testing.assert_allclose(z_next[2], 5.0)
        np.testing.assert_allclose(z_next[3], -3.0)

    def test_q_is_psd(self):
        _, Q = build_position_dynamics(dt=0.004, q_pos=1.0, q_vel=10.0)
        eigvals = jnp.linalg.eigvalsh(Q)
        assert jnp.all(eigvals >= 0)

    def test_position_only_mode(self):
        """2D position-only state (no velocity)."""
        A, Q = build_position_dynamics(
            dt=0.004, q_pos=1.0, include_velocity=False
        )
        assert A.shape == (2, 2)
        assert Q.shape == (2, 2)
        np.testing.assert_allclose(A, jnp.eye(2))
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py::TestBuildPositionDynamics -v`
Expected: FAIL with ImportError

### Step 3: Implement position dynamics

```python
# src/state_space_practice/position_decoder.py
"""Position decoding from population spike trains via Laplace-EKF.

Decodes the animal's 2D position from simultaneous spike trains of
neurons with known place fields. The latent state is position (and
optionally velocity), and observations are Poisson spike counts with
rates determined by each neuron's spatial tuning curve.

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
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from state_space_practice.kalman import _kalman_smoother_update, symmetrize
from state_space_practice.point_process_kalman import (
    _point_process_laplace_update,
)

logger = logging.getLogger(__name__)


def build_position_dynamics(
    dt: float,
    q_pos: float = 1.0,
    q_vel: float = 10.0,
    include_velocity: bool = True,
) -> tuple[Array, Array]:
    """Build transition matrix and process noise for position dynamics.

    Parameters
    ----------
    dt : float
        Time bin width in seconds.
    q_pos : float
        Position process noise (cm^2/s).
    q_vel : float
        Velocity process noise (cm^2/s^3). Only used if include_velocity=True.
    include_velocity : bool, default=True
        If True, state is [x, y, vx, vy] with constant-velocity dynamics.
        If False, state is [x, y] with random walk dynamics.

    Returns
    -------
    A : Array, shape (n_state, n_state)
        Transition matrix.
    Q : Array, shape (n_state, n_state)
        Process noise covariance.
    """
    if include_velocity:
        A = jnp.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        Q = jnp.diag(jnp.array([
            q_pos * dt,
            q_pos * dt,
            q_vel * dt,
            q_vel * dt,
        ]))
    else:
        A = jnp.eye(2)
        Q = jnp.eye(2) * q_pos * dt

    return A, Q
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py::TestBuildPositionDynamics -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/position_decoder.py \
        src/state_space_practice/tests/test_position_decoder.py
git commit -m "feat: add position dynamics model for spike-based decoding"
```

---

## Task 2: Place Field Rate Map Representation

Build a class that stores pre-estimated place fields (as rate maps on a grid) and provides fast log-rate evaluation with Jacobians at arbitrary positions via interpolation.

**Files:**
- Modify: `src/state_space_practice/position_decoder.py`
- Test: `src/state_space_practice/tests/test_position_decoder.py`

### Step 1: Write failing test

```python
# Add to tests/test_position_decoder.py

from state_space_practice.position_decoder import PlaceFieldRateMaps


class TestPlaceFieldRateMaps:
    @pytest.fixture
    def simple_fields(self):
        """Two neurons with Gaussian place fields at different locations."""
        n_grid = 50
        x_edges = np.linspace(0, 100, n_grid)
        y_edges = np.linspace(0, 100, n_grid)
        xx, yy = np.meshgrid(x_edges, y_edges)

        # Neuron 0: field at (30, 40)
        rate0 = 20 * np.exp(-((xx - 30)**2 + (yy - 40)**2) / (2 * 15**2)) + 0.5
        # Neuron 1: field at (70, 60)
        rate1 = 30 * np.exp(-((xx - 70)**2 + (yy - 60)**2) / (2 * 12**2)) + 0.5

        rate_maps = np.stack([rate0, rate1])  # (2, n_grid, n_grid)
        return PlaceFieldRateMaps(
            rate_maps=rate_maps,
            x_edges=x_edges,
            y_edges=y_edges,
        )

    def test_log_rate_at_field_center(self, simple_fields):
        """Rate should be highest near field center."""
        # Query near neuron 0's field center
        position = jnp.array([30.0, 40.0])
        log_rate = simple_fields.log_rate(position)
        assert log_rate.shape == (2,)
        # Neuron 0 should fire faster than neuron 1 at this location
        assert log_rate[0] > log_rate[1]

    def test_log_rate_away_from_field(self, simple_fields):
        """Rate should be near baseline far from field center."""
        position = jnp.array([90.0, 10.0])  # far from both fields
        log_rate = simple_fields.log_rate(position)
        # Both rates should be near log(0.5) ≈ -0.69
        assert jnp.all(log_rate < 1.0)

    def test_jacobian_shape(self, simple_fields):
        """Jacobian of log-rate w.r.t. position should be (n_neurons, 2)."""
        position = jnp.array([50.0, 50.0])
        jac = simple_fields.log_rate_jacobian(position)
        assert jac.shape == (2, 2)  # (n_neurons, n_position_dims)

    def test_jacobian_points_toward_field(self, simple_fields):
        """At a point between two fields, the gradient for neuron 0
        should point toward (30, 40)."""
        position = jnp.array([50.0, 50.0])
        jac = simple_fields.log_rate_jacobian(position)
        # Neuron 0's field is at (30, 40), so gradient should point left and down
        assert jac[0, 0] < 0  # d(log_rate_0)/dx < 0 (field is to the left)
        assert jac[0, 1] < 0  # d(log_rate_0)/dy < 0 (field is below)

    def test_from_place_field_model(self):
        """Construct rate maps from a fitted PlaceFieldModel."""
        from state_space_practice.place_field_model import PlaceFieldModel
        from state_space_practice.simulate_data import (
            simulate_2d_moving_place_field,
        )

        data = simulate_2d_moving_place_field(
            total_time=30.0, dt=0.004, peak_rate=80.0, n_interior_knots=3,
        )
        model = PlaceFieldModel(dt=0.004, n_interior_knots=3)
        model.fit(data["position"], data["spikes"], max_iter=2, verbose=False)

        rate_maps = PlaceFieldRateMaps.from_place_field_model(
            model, n_grid=30,
        )
        assert rate_maps.n_neurons == 1
        assert rate_maps.rate_maps.shape == (1, 30, 30)
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py::TestPlaceFieldRateMaps -v`
Expected: FAIL with ImportError

### Step 3: Implement PlaceFieldRateMaps

```python
# Add to src/state_space_practice/position_decoder.py

class PlaceFieldRateMaps:
    """Pre-computed place field rate maps for position decoding.

    Stores firing rate maps on a regular grid and provides fast
    interpolated evaluation at arbitrary positions. Computes Jacobians
    via finite differences for the Laplace-EKF.

    Parameters
    ----------
    rate_maps : np.ndarray, shape (n_neurons, n_grid_y, n_grid_x)
        Firing rate (Hz) for each neuron on the spatial grid.
    x_edges : np.ndarray, shape (n_grid_x,)
        X coordinates of the grid.
    y_edges : np.ndarray, shape (n_grid_y,)
        Y coordinates of the grid.
    """

    def __init__(
        self,
        rate_maps: np.ndarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
    ):
        self.rate_maps = np.asarray(rate_maps)
        self.x_edges = np.asarray(x_edges)
        self.y_edges = np.asarray(y_edges)
        self.n_neurons = rate_maps.shape[0]

        # Precompute log rate maps (clamp to avoid log(0))
        self._log_rate_maps = np.log(np.maximum(rate_maps, 1e-10))

        # Grid spacing for finite differences
        self._dx = float(x_edges[1] - x_edges[0])
        self._dy = float(y_edges[1] - y_edges[0])

    @classmethod
    def from_place_field_model(
        cls,
        model,
        n_grid: int = 50,
        time_slice: Optional[slice] = None,
    ) -> "PlaceFieldRateMaps":
        """Construct rate maps from a fitted PlaceFieldModel.

        Parameters
        ----------
        model : PlaceFieldModel
            Fitted encoding model.
        n_grid : int
            Grid resolution per dimension.
        time_slice : slice or None
            Time window to average over. None = full session.

        Returns
        -------
        PlaceFieldRateMaps
        """
        grid, x_edges, y_edges = model.make_grid(n_grid)
        rate, _ = model.predict_rate_map(grid, time_slice=time_slice)
        rate_2d = rate.reshape(1, n_grid, n_grid)  # single neuron
        return cls(rate_maps=np.array(rate_2d), x_edges=x_edges, y_edges=y_edges)

    @classmethod
    def from_spike_position_data(
        cls,
        position: np.ndarray,
        spike_counts: np.ndarray,
        dt: float,
        n_grid: int = 50,
        sigma: float = 5.0,
    ) -> "PlaceFieldRateMaps":
        """Estimate rate maps from position and spike data using KDE.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
        spike_counts : np.ndarray, shape (n_time,) or (n_time, n_neurons)
        dt : float
        n_grid : int
        sigma : float
            Gaussian smoothing kernel width in cm.

        Returns
        -------
        PlaceFieldRateMaps
        """
        from scipy.ndimage import gaussian_filter

        position = np.asarray(position)
        spike_counts = np.atleast_2d(spike_counts.T).T  # ensure (n_time, n_neurons)
        if spike_counts.ndim == 1:
            spike_counts = spike_counts[:, None]

        n_neurons = spike_counts.shape[1]
        x_edges = np.linspace(position[:, 0].min(), position[:, 0].max(), n_grid)
        y_edges = np.linspace(position[:, 1].min(), position[:, 1].max(), n_grid)

        # Compute occupancy
        occ, _, _ = np.histogram2d(
            position[:, 0], position[:, 1],
            bins=[x_edges, y_edges],
        )
        occ_time = occ * dt
        # Smooth occupancy
        occ_smooth = gaussian_filter(occ_time.T, sigma / (x_edges[1] - x_edges[0]))

        rate_maps = np.zeros((n_neurons, n_grid - 1, n_grid - 1))
        for n in range(n_neurons):
            spike_map, _, _ = np.histogram2d(
                position[:, 0], position[:, 1],
                bins=[x_edges, y_edges],
                weights=spike_counts[:, n],
            )
            spike_smooth = gaussian_filter(
                spike_map.T, sigma / (x_edges[1] - x_edges[0])
            )
            rate_maps[n] = np.where(
                occ_smooth > dt * 5, spike_smooth / occ_smooth, 0
            )

        # Use bin centers
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        return cls(rate_maps=rate_maps, x_edges=x_centers, y_edges=y_centers)

    def _interp_at(self, log_rate_map: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation of a single log-rate map at (x, y)."""
        # Find grid indices
        xi = (x - self.x_edges[0]) / self._dx
        yi = (y - self.y_edges[0]) / self._dy

        xi = np.clip(xi, 0, len(self.x_edges) - 1.001)
        yi = np.clip(yi, 0, len(self.y_edges) - 1.001)

        x0 = int(np.floor(xi))
        y0 = int(np.floor(yi))
        x1 = min(x0 + 1, log_rate_map.shape[1] - 1)
        y1 = min(y0 + 1, log_rate_map.shape[0] - 1)

        fx = xi - x0
        fy = yi - y0

        val = (
            log_rate_map[y0, x0] * (1 - fx) * (1 - fy)
            + log_rate_map[y0, x1] * fx * (1 - fy)
            + log_rate_map[y1, x0] * (1 - fx) * fy
            + log_rate_map[y1, x1] * fx * fy
        )
        return float(val)

    def log_rate(self, position: Array) -> Array:
        """Evaluate log firing rate for all neurons at a position.

        Parameters
        ----------
        position : Array, shape (2,) or (4,)
            Position [x, y] or state [x, y, vx, vy].

        Returns
        -------
        log_rate : Array, shape (n_neurons,)
        """
        x, y = float(position[0]), float(position[1])
        rates = np.array([
            self._interp_at(self._log_rate_maps[n], x, y)
            for n in range(self.n_neurons)
        ])
        return jnp.array(rates)

    def log_rate_jacobian(self, position: Array) -> Array:
        """Jacobian of log firing rate w.r.t. position via finite differences.

        Parameters
        ----------
        position : Array, shape (2,) or (4,)

        Returns
        -------
        jacobian : Array, shape (n_neurons, 2)
            d(log_rate_n) / d(x, y) for each neuron.
        """
        x, y = float(position[0]), float(position[1])
        eps_x = self._dx * 0.5
        eps_y = self._dy * 0.5

        rate_xp = self.log_rate(jnp.array([x + eps_x, y]))
        rate_xm = self.log_rate(jnp.array([x - eps_x, y]))
        rate_yp = self.log_rate(jnp.array([x, y + eps_y]))
        rate_ym = self.log_rate(jnp.array([x, y - eps_y]))

        dfdx = (rate_xp - rate_xm) / (2 * eps_x)
        dfdy = (rate_yp - rate_ym) / (2 * eps_y)

        return jnp.column_stack([dfdx, dfdy])
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py::TestPlaceFieldRateMaps -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/position_decoder.py \
        src/state_space_practice/tests/test_position_decoder.py
git commit -m "feat: add PlaceFieldRateMaps for interpolated rate evaluation"
```

---

## Task 3: Position Decoder Filter and Smoother

Build the Laplace-EKF filter and RTS smoother for position decoding from multi-neuron spikes.

**Files:**
- Modify: `src/state_space_practice/position_decoder.py`
- Test: `src/state_space_practice/tests/test_position_decoder.py`

### Step 1: Write failing test

```python
# Add to tests/test_position_decoder.py

from state_space_practice.position_decoder import (
    position_decoder_filter,
    position_decoder_smoother,
    DecoderResult,
)


class TestPositionDecoderFilter:
    @pytest.fixture
    def decoding_data(self):
        """Simulate position + spikes, then build rate maps and decode."""
        rng = np.random.default_rng(42)
        n_time = 500
        dt = 0.004

        # Simple trajectory: circle
        t = np.arange(n_time) * dt
        true_x = 50 + 20 * np.cos(2 * np.pi * t / 2.0)
        true_y = 50 + 20 * np.sin(2 * np.pi * t / 2.0)
        true_pos = np.column_stack([true_x, true_y])

        # Two neurons with place fields
        n_grid = 30
        x_edges = np.linspace(0, 100, n_grid)
        y_edges = np.linspace(0, 100, n_grid)
        xx, yy = np.meshgrid(x_edges, y_edges)

        rate0 = 30 * np.exp(-((xx-30)**2 + (yy-50)**2) / (2*15**2)) + 0.5
        rate1 = 30 * np.exp(-((xx-70)**2 + (yy-50)**2) / (2*15**2)) + 0.5
        rate_maps_arr = np.stack([rate0, rate1])

        rate_maps = PlaceFieldRateMaps(
            rate_maps=rate_maps_arr, x_edges=x_edges, y_edges=y_edges,
        )

        # Generate spikes from true position
        spikes = np.zeros((n_time, 2))
        for t_idx in range(n_time):
            log_r = rate_maps.log_rate(jnp.array(true_pos[t_idx]))
            rates = np.exp(np.array(log_r))
            spikes[t_idx] = rng.poisson(rates * dt)

        return {
            "true_position": true_pos,
            "spikes": jnp.array(spikes),
            "rate_maps": rate_maps,
            "dt": dt,
            "n_time": n_time,
        }

    def test_filter_output_shapes(self, decoding_data):
        result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        assert isinstance(result, DecoderResult)
        assert result.position_mean.shape == (decoding_data["n_time"], 4)
        assert result.position_cov.shape == (decoding_data["n_time"], 4, 4)

    def test_filter_tracks_position(self, decoding_data):
        """Decoded position should be correlated with true position."""
        result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        decoded_x = np.array(result.position_mean[:, 0])
        true_x = decoding_data["true_position"][:, 0]

        # Skip first 50 bins (filter warmup)
        corr = np.corrcoef(decoded_x[50:], true_x[50:])[0, 1]
        assert corr > 0.5  # should track reasonably well

    def test_smoother_reduces_error(self, decoding_data):
        """Smoother should have lower error than filter."""
        filter_result = position_decoder_filter(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )
        smoother_result = position_decoder_smoother(
            spikes=decoding_data["spikes"],
            rate_maps=decoding_data["rate_maps"],
            dt=decoding_data["dt"],
        )

        true_pos = decoding_data["true_position"]
        filter_error = np.mean(np.linalg.norm(
            np.array(filter_result.position_mean[50:, :2]) - true_pos[50:],
            axis=1,
        ))
        smoother_error = np.mean(np.linalg.norm(
            np.array(smoother_result.position_mean[50:, :2]) - true_pos[50:],
            axis=1,
        ))

        assert smoother_error <= filter_error * 1.1  # smoother should be at least as good
```

### Step 2: Run test to verify it fails

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py::TestPositionDecoderFilter -v`
Expected: FAIL with ImportError

### Step 3: Implement filter and smoother

```python
# Add to src/state_space_practice/position_decoder.py

from typing import NamedTuple


class DecoderResult(NamedTuple):
    """Result of position decoding.

    Attributes
    ----------
    position_mean : Array, shape (n_time, n_state)
        Decoded position (and velocity) at each time step.
        Columns: [x, y, vx, vy] or [x, y].
    position_cov : Array, shape (n_time, n_state, n_state)
        Posterior covariance at each time step.
    marginal_log_likelihood : float
        Total log-likelihood of the spike observations.
    """

    position_mean: Array
    position_cov: Array
    marginal_log_likelihood: float


def position_decoder_filter(
    spikes: Array,
    rate_maps: PlaceFieldRateMaps,
    dt: float,
    q_pos: float = 1.0,
    q_vel: float = 10.0,
    include_velocity: bool = True,
    init_position: Optional[Array] = None,
    init_cov: Optional[Array] = None,
) -> DecoderResult:
    """Decode position from spikes using Laplace-EKF filter.

    Parameters
    ----------
    spikes : Array, shape (n_time, n_neurons)
        Spike counts per time bin per neuron.
    rate_maps : PlaceFieldRateMaps
        Pre-estimated place fields for each neuron.
    dt : float
        Time bin width in seconds.
    q_pos : float
        Position process noise (cm^2/s).
    q_vel : float
        Velocity process noise (cm^2/s^3).
    include_velocity : bool
        Whether to include velocity in the state.
    init_position : Array or None
        Initial position estimate [x, y] or [x, y, vx, vy].
        If None, uses center of arena.
    init_cov : Array or None
        Initial covariance. If None, uses large diagonal.

    Returns
    -------
    DecoderResult
    """
    spikes = jnp.asarray(spikes)
    if spikes.ndim == 1:
        spikes = spikes[:, None]
    n_time = spikes.shape[0]

    A, Q = build_position_dynamics(dt, q_pos, q_vel, include_velocity)
    n_state = A.shape[0]

    if init_position is None:
        cx = float(rate_maps.x_edges.mean())
        cy = float(rate_maps.y_edges.mean())
        if include_velocity:
            init_position = jnp.array([cx, cy, 0.0, 0.0])
        else:
            init_position = jnp.array([cx, cy])

    if init_cov is None:
        init_cov = jnp.eye(n_state) * 100.0

    # Build log-intensity function for the Laplace-EKF
    # This maps state → log(firing rates) for all neurons
    def log_intensity_func(state):
        return rate_maps.log_rate(state)

    # Build Jacobian: d(log_rate) / d(state)
    # Only position dims matter (rate doesn't depend on velocity)
    def grad_log_intensity_func(state):
        jac_pos = rate_maps.log_rate_jacobian(state)  # (n_neurons, 2)
        if include_velocity:
            # Pad with zeros for velocity dims
            return jnp.concatenate(
                [jac_pos, jnp.zeros((rate_maps.n_neurons, 2))], axis=1
            )
        return jac_pos

    # Hessian: assume zero (Gauss-Newton approximation)
    def hess_log_intensity_func(state):
        n_neurons = rate_maps.n_neurons
        return jnp.zeros((n_neurons, n_state, n_state))

    # Run filter
    def _step(carry, spike_t):
        mean_prev, cov_prev, total_ll = carry

        # Prediction
        one_step_mean = A @ mean_prev
        one_step_cov = A @ cov_prev @ A.T + Q
        one_step_cov = symmetrize(one_step_cov)

        # Laplace update
        post_mean, post_cov, ll = _point_process_laplace_update(
            one_step_mean,
            one_step_cov,
            spike_t,
            dt,
            log_intensity_func,
            grad_log_intensity_func=grad_log_intensity_func,
            hess_log_intensity_func=hess_log_intensity_func,
            include_laplace_normalization=False,
        )

        total_ll = total_ll + ll
        return (post_mean, post_cov, total_ll), (post_mean, post_cov)

    init_carry = (init_position, init_cov, jnp.array(0.0))
    (_, _, marginal_ll), (filtered_mean, filtered_cov) = jax.lax.scan(
        _step, init_carry, spikes,
    )

    return DecoderResult(
        position_mean=filtered_mean,
        position_cov=filtered_cov,
        marginal_log_likelihood=float(marginal_ll),
    )


def position_decoder_smoother(
    spikes: Array,
    rate_maps: PlaceFieldRateMaps,
    dt: float,
    q_pos: float = 1.0,
    q_vel: float = 10.0,
    include_velocity: bool = True,
    init_position: Optional[Array] = None,
    init_cov: Optional[Array] = None,
) -> DecoderResult:
    """Decode position from spikes using Laplace-EKF + RTS smoother.

    Same parameters as ``position_decoder_filter``. Returns smoothed
    (non-causal) estimates that use the full spike train.

    Returns
    -------
    DecoderResult with smoothed position estimates.
    """
    # First run filter
    filter_result = position_decoder_filter(
        spikes, rate_maps, dt, q_pos, q_vel,
        include_velocity, init_position, init_cov,
    )

    A, Q = build_position_dynamics(dt, q_pos, q_vel, include_velocity)

    # RTS backward smoother
    def _smooth_step(carry, inputs):
        next_sm_mean, next_sm_cov = carry
        filt_mean, filt_cov = inputs

        sm_mean, sm_cov, cross_cov = _kalman_smoother_update(
            next_sm_mean, next_sm_cov,
            filt_mean, filt_cov,
            Q, A,
        )
        return (sm_mean, sm_cov), (sm_mean, sm_cov)

    _, (sm_mean, sm_cov) = jax.lax.scan(
        _smooth_step,
        (filter_result.position_mean[-1], filter_result.position_cov[-1]),
        (filter_result.position_mean[:-1], filter_result.position_cov[:-1]),
        reverse=True,
    )

    smoother_mean = jnp.concatenate([sm_mean, filter_result.position_mean[-1:]])
    smoother_cov = jnp.concatenate([sm_cov, filter_result.position_cov[-1:]])

    return DecoderResult(
        position_mean=smoother_mean,
        position_cov=smoother_cov,
        marginal_log_likelihood=filter_result.marginal_log_likelihood,
    )
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py::TestPositionDecoderFilter -v`
Expected: PASS

Note: The interpolation-based log_rate function is not JIT-compatible due to numpy indexing. The filter will run without JIT, which is slower but correct. A future optimization would replace the interpolation with a JAX-compatible version.

### Step 5: Commit

```bash
git add src/state_space_practice/position_decoder.py \
        src/state_space_practice/tests/test_position_decoder.py
git commit -m "feat: add Laplace-EKF position decoder filter and smoother"
```

---

## Task 4: PositionDecoder Model Class

Wrap everything into a user-friendly model class with fit (estimate rate maps) and decode (run filter/smoother) methods.

**Files:**
- Modify: `src/state_space_practice/position_decoder.py`
- Test: `src/state_space_practice/tests/test_position_decoder.py`

### Step 1: Write failing test

```python
# Add to tests/test_position_decoder.py

from state_space_practice.position_decoder import PositionDecoder


class TestPositionDecoder:
    @pytest.fixture
    def trajectory_data(self):
        rng = np.random.default_rng(42)
        n_time = 1000
        dt = 0.004

        t = np.arange(n_time) * dt
        true_x = 50 + 20 * np.cos(2 * np.pi * t / 2.0)
        true_y = 50 + 20 * np.sin(2 * np.pi * t / 2.0)
        position = np.column_stack([true_x, true_y])

        # 5 neurons with different place fields
        n_neurons = 5
        centers = rng.uniform(20, 80, (n_neurons, 2))
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
        }

    def test_fit_decode_workflow(self, trajectory_data):
        decoder = PositionDecoder(dt=trajectory_data["dt"])

        # Fit: estimate rate maps from training data
        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        assert decoder.rate_maps is not None
        assert decoder.rate_maps.n_neurons == 5

        # Decode: recover position from spikes
        result = decoder.decode(
            spikes=trajectory_data["spikes"],
            method="smoother",
        )
        assert result.position_mean.shape[0] == trajectory_data["n_time"]

    def test_decode_error(self, trajectory_data):
        decoder = PositionDecoder(dt=trajectory_data["dt"])
        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        result = decoder.decode(spikes=trajectory_data["spikes"])

        decoded_pos = np.array(result.position_mean[:, :2])
        true_pos = trajectory_data["position"]
        error = np.median(np.linalg.norm(decoded_pos[100:] - true_pos[100:], axis=1))
        # Median error should be less than 30 cm
        assert error < 30.0

    def test_plot_decoding(self, trajectory_data):
        import matplotlib
        matplotlib.use("Agg")

        decoder = PositionDecoder(dt=trajectory_data["dt"])
        decoder.fit(
            position=trajectory_data["position"],
            spikes=trajectory_data["spikes"],
        )
        result = decoder.decode(spikes=trajectory_data["spikes"])
        fig = decoder.plot_decoding(
            result, true_position=trajectory_data["position"],
        )
        assert fig is not None
```

### Step 2: Run test to verify it fails

### Step 3: Implement PositionDecoder class

```python
# Add to src/state_space_practice/position_decoder.py

class PositionDecoder:
    """Position decoder from population spike trains.

    Two-step workflow:
    1. ``fit(position, spikes)``: estimate place field rate maps from
       training data (position + spikes during behavior).
    2. ``decode(spikes)``: decode position from spike trains alone,
       using the fitted rate maps.

    Parameters
    ----------
    dt : float
        Time bin width in seconds.
    q_pos : float, default=1.0
        Position process noise (cm^2/s).
    q_vel : float, default=10.0
        Velocity process noise (cm^2/s^3).
    include_velocity : bool, default=True
        Include velocity in the state.
    n_grid : int, default=50
        Grid resolution for rate map estimation.
    smoothing_sigma : float, default=5.0
        Gaussian smoothing kernel width (cm) for rate map estimation.

    Examples
    --------
    >>> decoder = PositionDecoder(dt=0.004)
    >>> decoder.fit(train_position, train_spikes)
    >>> result = decoder.decode(test_spikes)
    >>> decoded_xy = result.position_mean[:, :2]
    """

    def __init__(
        self,
        dt: float,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        include_velocity: bool = True,
        n_grid: int = 50,
        smoothing_sigma: float = 5.0,
    ):
        self.dt = dt
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.include_velocity = include_velocity
        self.n_grid = n_grid
        self.smoothing_sigma = smoothing_sigma

        self.rate_maps: Optional[PlaceFieldRateMaps] = None

    def fit(
        self,
        position: np.ndarray,
        spikes: np.ndarray,
    ) -> None:
        """Estimate place field rate maps from training data.

        Parameters
        ----------
        position : np.ndarray, shape (n_time, 2)
        spikes : np.ndarray, shape (n_time,) or (n_time, n_neurons)
        """
        self.rate_maps = PlaceFieldRateMaps.from_spike_position_data(
            position=position,
            spike_counts=spikes,
            dt=self.dt,
            n_grid=self.n_grid,
            sigma=self.smoothing_sigma,
        )

    def decode(
        self,
        spikes: ArrayLike,
        method: str = "smoother",
        init_position: Optional[Array] = None,
    ) -> DecoderResult:
        """Decode position from spike trains.

        Parameters
        ----------
        spikes : ArrayLike, shape (n_time, n_neurons)
        method : str, "filter" or "smoother"
            "filter" for causal (online) decoding.
            "smoother" for non-causal (offline) decoding.
        init_position : Array or None
            Initial position estimate.

        Returns
        -------
        DecoderResult
        """
        if self.rate_maps is None:
            raise RuntimeError("Must call fit() before decode().")

        spikes = jnp.asarray(spikes)
        if spikes.ndim == 1:
            spikes = spikes[:, None]

        decode_func = (
            position_decoder_smoother if method == "smoother"
            else position_decoder_filter
        )

        return decode_func(
            spikes=spikes,
            rate_maps=self.rate_maps,
            dt=self.dt,
            q_pos=self.q_pos,
            q_vel=self.q_vel,
            include_velocity=self.include_velocity,
            init_position=init_position,
        )

    def plot_decoding(
        self,
        result: DecoderResult,
        true_position: Optional[np.ndarray] = None,
        ax=None,
    ):
        """Plot decoded vs true position trajectory.

        Parameters
        ----------
        result : DecoderResult
        true_position : np.ndarray or None, shape (n_time, 2)
        ax : Axes or None

        Returns
        -------
        fig : Figure
        """
        import matplotlib.pyplot as plt

        decoded = np.array(result.position_mean[:, :2])

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        # Left: 2D trajectory
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

        # Right: error over time (if true position available)
        if true_position is not None:
            error = np.linalg.norm(decoded - true_position, axis=1)
            t = np.arange(len(error)) * self.dt
            axes[1].plot(t, error, "k-", alpha=0.7)
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Position error (cm)")
            axes[1].set_title(f"Median error: {np.median(error):.1f} cm")
            # Uncertainty ellipse size
            pos_var = np.array(result.position_cov[:, :2, :2])
            ci_radius = np.sqrt(
                pos_var[:, 0, 0] + pos_var[:, 1, 1]
            ) * 1.96
            axes[1].fill_between(
                t, 0, ci_radius, alpha=0.2, color="blue", label="95% CI radius"
            )
            axes[1].legend()

        fig.tight_layout()
        return fig
```

### Step 4: Run test to verify it passes

Run: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py::TestPositionDecoder -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/state_space_practice/position_decoder.py \
        src/state_space_practice/tests/test_position_decoder.py
git commit -m "feat: add PositionDecoder model class with fit/decode/plot workflow"
```
