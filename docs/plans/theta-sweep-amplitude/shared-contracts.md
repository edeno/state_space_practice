# Shared contracts

[← back to PLAN.md](PLAN.md)

Cross-phase types and signatures. Each appears once; phases link by anchor. Fields marked *(2a)* / *(2b)* are added by that phase and default to `None` before it ships. Everything lives in `src/state_space_practice/theta_sweep.py` unless stated. Units: positions and coefficients in cm, time in seconds, rates in Hz, phase in radians.

- [Design-matrix layout](#design-matrix-layout)
- [LinearRateMaps](#linearratemaps)
- [theta_sweep_smoother](#theta_sweep_smoother)
- [ThetaSweepResult](#thetasweepresult)
- [ThetaSweepModel](#thetasweepmodel)
- [Simulator output](#simulator-output)

## Design-matrix layout

`build_sweep_design(position, phase, direction, n_harmonics=1, phase_offset=None) -> np.ndarray` of shape `(n_time, 1 + K)`:

- column 0: `p_t`, the linear position (cm);
- columns `1..K`: the **phase basis** `b_t`, already multiplied by the travel direction `d_t in {-1, +1}`:
  - `phase_offset is None` (free offset): `K = 2 * n_harmonics`, columns interleaved as `d_t * [cos(phi_t), sin(phi_t), cos(2 phi_t), sin(2 phi_t), ..., cos(H phi_t), sin(H phi_t)]`, so the state is `x = (c_1, s_1, c_2, s_2, ..., c_H, s_H)`;
  - `phase_offset = psi` (fixed offset; only with `n_harmonics = 1`, else `ValueError`): `K = 1`, column `d_t * cos(phi_t - psi)`, state `x = (a,)` — a *signed* amplitude (negative means anti-phase to `psi`).

The represented position is `r_t(x) = Z_t[0] + Z_t[1:] @ x`. This layout is what every `log_conditional_intensity` callable in this plan expects as `design_t`; phase 2a **appends one column** `d_t` (index `1 + K`) for the cycle-offset state, so with `cycle_offset=True` the state is `(x, epsilon)` of dimension `K + 1` and `r_t = Z_t[0] + Z_t[1:] @ (x, epsilon)`. Do not reorder columns: the M-steps in phase 2b index the coefficient block as `[:K]` and the offset as `[-1]`.

## LinearRateMaps

```python
class LinearRateMaps:
    """Fixed 1-D rate maps on a uniform grid, evaluated by linear interpolation."""

    def __init__(
        self,
        rate_maps: ArrayLike,      # (n_neurons, n_grid), Hz, finite, >= 0
        grid: ArrayLike,           # (n_grid,), cm, strictly increasing, uniform spacing (rtol 1e-6), n_grid >= 2
        rate_floor_hz: float = 0.01,
    ) -> None: ...

    n_neurons: int
    grid: np.ndarray               # (n_grid,)
    rate_maps: np.ndarray          # (n_neurons, n_grid)
    grid_start: float              # grid[0]
    grid_step: float               # grid[1] - grid[0]
    log_rate_maps: Array           # (n_neurons, n_grid) = log(max(rate_maps, rate_floor_hz)), float64

    def log_rate(self, position: ArrayLike) -> Array:
        """log rate of every neuron at one scalar position -> (n_neurons,); constant beyond the grid ends."""

    @property
    def log_intensity(self) -> Callable[[Array, Array], Array]:
        """The cached ``(design_t, x) -> (n_neurons,)`` callable for the point-process filter.

        Built once (a ``functools.partial`` of ``sweep_log_intensity`` over this
        object's arrays); the filter jits on the callable's identity, so callers
        must reuse this object rather than rebuild it per call.
        """

    @classmethod
    def from_spike_position_data(
        cls,
        position: ArrayLike,       # (n_time,)
        spikes: ArrayLike,         # (n_time, n_neurons) counts
        dt: float,
        grid: ArrayLike,           # (n_grid,) bin centres
        smoothing_sigma: float = 5.0,   # cm
        min_occupancy: float = 0.5,     # seconds of (smoothed) occupancy below which a bin falls back to the neuron's mean rate
        bin_mask: ArrayLike | None = None,  # (n_time,) bool; True = this time bin contributes
        rate_floor_hz: float = 0.01,
    ) -> "LinearRateMaps": ...
```

Invariant (do not weaken): `log_rate(r)` and `log_intensity(design_t, x)` evaluate the **same** interpolant (`_interp_log_rate` in [designs.md](designs.md#log-rate-interpolation)); the simulator and the null log-likelihood use `log_rate`, the filter uses `log_intensity`, so the null and sweep likelihoods and the simulated spikes share one rate function.

`phase_restricted_bins(phase, phase_offset, half_width=np.pi / 8) -> np.ndarray[bool]` (module function): bins where `|wrap_to_pi(phase - phase_offset)|` is within `half_width` of `pi/2`, i.e. where `cos(phi - psi) ~ 0` and the represented position is closest to the animal — the mask to pass as `bin_mask` when refitting maps.

## theta_sweep_smoother

Functional entry point for one contiguous running segment (the analogue of `position_decoder_smoother`). Fast tests use it so they never call `.fit(`.

```python
def theta_sweep_smoother(
    spikes: ArrayLike,             # (n_time, n_neurons) counts
    design: ArrayLike,             # (n_time, 1 + K) from build_sweep_design (2a: 1 + K + 1 with cycle_offset)
    rate_maps: LinearRateMaps,
    dt: float,
    *,
    process_variance: float,       # q, cm^2/s; per-bin process covariance is q * dt * I_K
    init_std: float,               # x_0 ~ N(init_mean, init_std^2 I_K)
    init_mean: ArrayLike | None = None,   # default zeros
    max_newton_iter: int = 3,
    max_rate_hz: float = 500.0,    # -> max_log_count = log(max_rate_hz * dt)
    cycle_starts: ArrayLike | None = None,        # (2a) (n_time,) bool; None = no offset state
    cycle_offset_variance: float | None = None,   # (2a) sigma_c^2, required with cycle_starts
) -> tuple[Array, Array, Array, Array]:
    """Returns (smoother_mean (n_time, D), smoother_cov (n_time, D, D),
    smoother_cross_cov (n_time - 1, D, D), log_likelihood (scalar)),
    D = K (+ 1 with the offset state). Validates counts, dt and the PSD of the
    initial covariance host-side when inputs are concrete; skips when traced."""
```

Phase 1 implements it as a direct call to `stochastic_point_process_smoother(init_mean, init_cov, design, spikes, dt, I_K, q*dt*I_K, rate_maps.log_intensity, max_log_count=..., max_newton_iter=...)`. Phase 2a re-implements it over the time-varying scan ([designs.md](designs.md#time-varying-dynamics)) and keeps the signature and outputs; the equivalence test pins the constant-dynamics case to the library smoother.

## ThetaSweepResult

```python
@dataclass(frozen=True)
class ThetaSweepResult:
    coefficient_mean: np.ndarray        # (n_time, K); NaN outside running bouts
    coefficient_cov: np.ndarray         # (n_time, K, K); NaN outside
    amplitude: np.ndarray               # (n_time,) posterior median of the first-harmonic amplitude sqrt(c_1^2 + s_1^2);
                                        #   fixed-offset mode: the posterior mean of the signed a
    amplitude_interval: np.ndarray      # (n_time, 2) central 95% posterior interval of the same quantity
    phase_offset: np.ndarray            # (n_time,) circular posterior mean of atan2(s_1, c_1); fixed-offset mode: the given psi
    coefficient_correlation: np.ndarray # (n_time,) posterior corr(c_1, s_1); NaN in fixed-offset mode
    origin_in_credible_region: np.ndarray  # (n_time,) bool: m' P^{-1} m < chi2_K(0.95)
    represented_position: np.ndarray    # (n_time,) Z_t[0] + Z_t[1:] @ m_t (plug-in)
    running_mask: np.ndarray            # (n_time,) bool, union of the processed bouts
    bouts: list[tuple[int, int]]        # (start, end_exclusive) as returned by identify_behavioral_bouts
    log_likelihood: float               # sweep model, summed over bouts (Laplace marginal LL)
    null_log_likelihood: float          # a == 0 on the same bins
    cycle_index: np.ndarray | None = None        # (2a) (n_time,) int, -1 outside bouts
    cycle_offset_mean: np.ndarray | None = None  # (2a) (n_time,) smoothed epsilon_t; NaN outside
    cycle_offset_var: np.ndarray | None = None   # (2a) (n_time,)
    process_variance: float | None = None        # (2b) fitted q when fit_hyperparameters=True
    cycle_offset_variance: float | None = None   # (2b) fitted sigma_c^2
    em_log_likelihoods: list[float] | None = None  # (2b) accepted EM log-likelihoods

    @property
    def score(self) -> float:
        """log_likelihood - null_log_likelihood (nats). Positive favours sweeps."""
```

Invariant (do not weaken): every per-time array has the same length as the inputs and is `NaN` (or `False`, `-1`) exactly where `running_mask` is `False`.

## ThetaSweepModel

```python
class ThetaSweepModel:
    def __init__(
        self,
        rate_maps: LinearRateMaps,
        dt: float,
        *,
        n_harmonics: int = 1,
        phase_offset: float | None = None,
        process_variance: float = 0.5,      # cm^2/s
        init_std: float = 20.0,             # cm
        speed_threshold: float = 5.0,       # cm/s; running = |velocity| > threshold
        min_bout_duration: float = 1.0,     # s; shorter bouts are dropped
        max_newton_iter: int = 3,
        max_rate_hz: float = 500.0,
        n_posterior_samples: int = 1000,    # Monte-Carlo draws for the amplitude summaries
        seed: int = 0,                      # PRNG key for those draws
        cycle_offset: bool = False,                 # (2a)
        cycle_offset_variance: float = 25.0,        # (2a) sigma_c^2, cm^2 (fixed until 2b fits it)
        min_cycle_duration: float = 0.05,           # phase 1; s; shared cycle detection for surrogates and offsets
    ) -> None: ...

    result_: ThetaSweepResult   # NotFittedError on access before fit

    def fit(
        self,
        spikes: ArrayLike,          # (n_time, n_neurons)
        position: ArrayLike,        # (n_time,) linear position, cm
        phase: ArrayLike,           # (n_time,) theta phase, radians, any 2*pi range
        velocity: ArrayLike,        # (n_time,) signed velocity along the track, cm/s
        *,
        fit_hyperparameters: bool = False,   # (2b) EM over q (and sigma_c^2 with cycle_offset)
        max_iter: int = 50,                  # (2b)
        tolerance: float = 1e-4,             # (2b)
    ) -> ThetaSweepResult: ...

    def phase_shuffle_scores(
        self,
        spikes, position, phase, velocity,
        *,
        n_shuffles: int = 20,
        seed: int = 0,
    ) -> np.ndarray: ...            # (n_shuffles,) scores of refits on shuffled phase
```

Semantics (do not weaken): `fit` raises `ValueError` on shape mismatches, non-finite inputs, non-count spikes, `spikes.shape[1] != rate_maps.n_neurons`, or when no bout survives `min_bout_duration`; it never silently drops neurons. Bouts are found with `identify_behavioral_bouts(np.abs(velocity), speed_threshold, min_duration=round(min_bout_duration / dt))` and each is smoothed independently from the same prior. `fit` also stores `result_`. `phase_shuffle_scores` refits with the *same* hyperparameters (it never runs EM, even after 2b) and applies independent uniform rotations to the original cycles within each bout. The original phase determines cycle boundaries and, in phase 2a, offset resets; the surrogate phase is used only in the observation design. It raises `ValueError` if any retained bout has fewer than two detected cycles. This tests coherence across cycles under the fixed process variance; it does not guarantee amplitude collapse or preserve phase autocorrelation.

## Simulator output

`simulate_linear_track_theta_sweep(...)` (in `src/state_space_practice/simulate/simulate_theta_sweep.py`) returns a `dict`:

| key | shape / type | meaning |
| --- | --- | --- |
| `dt` | float | bin width (s) |
| `time` | `(n_time,)` | bin start times |
| `position` | `(n_time,)` | linear position, back-and-forth laps with pauses at both ends |
| `velocity` | `(n_time,)` | signed velocity (`+speed`, `-speed`, or 0 during pauses) |
| `phase` | `(n_time,)` | theta phase in `[-pi, pi)` |
| `direction` | `(n_time,)` | `sign(velocity)` |
| `spikes` | `(n_time, n_neurons)` | Poisson counts drawn from `rate_maps.log_rate(represented_position)` |
| `rate_maps` | `LinearRateMaps` | the true Gaussian fields on a `grid_step` grid |
| `true_amplitude` | `(n_time,)` | `a_t` (constant, step or drift schedule) |
| `true_coefficients` | `(n_time, 2)` | `a_t * (cos psi, sin psi)` |
| `true_phase_offset` | float | `psi` |
| `true_cycle_offset` | `(n_time,)` | per-cycle offset `epsilon_t` (used by 2a's tests); zeros when `cycle_offset_std = 0` |
| `cycle_index` | `(n_time,)` | theta-cycle index used to draw the offsets: `floor(theta_frequency * time + 0.5)`, which increments exactly where `phase` wraps from `+pi` to `-pi` (the same boundary `theta_cycle_index` detects) |
| `represented_position` | `(n_time,)` | `position + direction * (a_t cos(phase - psi) + epsilon_t)` |

Invariant: `spikes` are generated from `rate_maps.log_rate` at `represented_position`, so the model is exactly specified at the true parameters (this is what the calibration tests rely on).
