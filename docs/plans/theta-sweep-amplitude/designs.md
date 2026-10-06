# Designs — derivations and complete code

[← back to PLAN.md](PLAN.md) · [contracts](shared-contracts.md) · [overview](overview.md)

One section per component. Code is complete for the parts that are not obvious; names match [shared-contracts.md](shared-contracts.md). All library code lives in `src/state_space_practice/theta_sweep.py` unless stated; float64 is assumed (the test conftest enables it, and the module docstring repeats the import-order recipe from `README.md:41-58`).

- [Observation model](#observation-model)
- [Log-rate interpolation](#log-rate-interpolation)
- [Design matrix](#design-matrix)
- [Posterior summaries](#posterior-summaries)
- [Null model and score](#null-model-and-score)
- [Bout orchestration](#bout-orchestration)
- [Phase shuffle](#phase-shuffle)
- [Linear rate maps](#linear-rate-maps)
- [Simulator](#simulator)
- [Oracle generalisation](#oracle-generalisation)
- [Cycle boundaries](#cycle-boundaries) *(2a)*
- [Time-varying dynamics](#time-varying-dynamics) *(2a)*
- [Hyperparameter M-steps](#hyperparameter-m-steps) *(2b)*
- [Cycle-variance test](#cycle-variance-test) *(2b)*

## Observation model

Inputs per bin `t`: position `p_t`, phase `phi_t`, direction `d_t in {-1, +1}`, counts `y_t in N^{n_neurons}`, and fixed log-rate maps `l_n(r) = log lambda_n(r)` on a uniform grid `g_0 < ... < g_{G-1}` (step `Delta`).

**Phase basis and state.** With `H` harmonics the basis is `b_t = d_t [cos phi_t, sin phi_t, ..., cos(H phi_t), sin(H phi_t)] in R^K`, `K = 2H`, and the state is `x = (c_1, s_1, ..., c_H, s_H)`. The represented position is

```text
r_t(x) = p_t + b_t . x
```

For `H = 1`, `c cos phi + s sin phi = a cos(phi - psi)` with `a = sqrt(c^2 + s^2)`, `psi = atan2(s, c)`; the lead in the direction of travel is `d_t (r_t - p_t) = a cos(phi_t - psi)`, positive = ahead. With a known offset `psi` the basis is `b_t = d_t cos(phi_t - psi)` and the state is the signed scalar `a`.

**Likelihood.** `y_{n,t} ~ Poisson(mu_{n,t})`, `mu_{n,t} = exp(l_n(r_t(x))) dt` (clipped by `_safe_expected_count`, `point_process_kalman.py:589-612`, at `max_log_count = log(max_rate_hz dt)`).

**Jacobian.** By the chain rule, with `l_n'(r)` the slope of the interpolant,

```text
d log lambda_{n,t} / d x = l_n'(r_t) b_t ,       J_t = l'(r_t) (outer) b_t   in R^{n_neurons x K}
```

The filter obtains `J_t` by `jax.jacfwd` of the log-intensity callable (`point_process_kalman.py:1794-1798`); the analytic form is implemented too (`sweep_log_intensity_jacobian`) and a fast test asserts the two agree. Off the grid the interpolant is constant, so `l_n' = 0` and the bin carries no information about `x`.

**Score and Fisher information at `x`** (what `_point_process_laplace_update` forms at `point_process_kalman.py:1118-1139`):

```text
g_t(x) = sum_n (y_{n,t} - mu_{n,t}) l_n'(r_t) b_t = [sum_n (y_{n,t} - mu_{n,t}) l_n'(r_t)] b_t
I_t(x) = sum_n mu_{n,t} l_n'(r_t)^2 b_t b_t'     = w_t(x) b_t b_t' ,   w_t = sum_n mu_{n,t} l_n'(r_t)^2
```

`I_t` is **rank one** for every bin: one bin constrains `x` only along `b_t`. The `(c, s)` split is identified by pooling bins across phases within the memory of the random walk. Over a full cycle of `T_c` bins with roughly constant `w`, `sum_t b_t b_t' ~ (T_c / 2) I_2`, so the per-cycle information is `~ (T_c / 2) w I_2`. Order of magnitude for the simulator defaults (40 fields of width `w_f = 8 cm` spaced 3.5 cm, peak 20 Hz, `dt = 4 ms`): near a field `l_n'(r) ~ -(r - mu_n) / w_f^2` and `mu ~ 0.08 exp(-(r - mu_n)^2 / 2 w_f^2)`, so `w ~ (0.08 / 3.5) integral (delta^2 / w_f^4) exp(-delta^2 / 2 w_f^2) d delta ~ 0.08 sqrt(2 pi) / (3.5 w_f) ~ 7e-3` per bin, `~0.11` per 8 Hz cycle (31 bins), i.e. a posterior sd of `~3 cm` from one cycle and `~1 cm` from nine cycles (~1.1 s). A random walk with `q = 0.5 cm^2/s` moves `~0.7 cm` per second, so the default process variance lets the posterior pool about a second of data; this is the reasoning behind `process_variance = 0.5` and the "small process noise" decision.

**Process model (phase 1).** `x_t = x_{t-1} + w_t`, `w_t ~ N(0, q dt I_K)`, `x_0 ~ N(0, init_std^2 I_K)`; `A = I_K`, `Q = q dt I_K`.

**Why Fisher scoring, not Newton.** The log-rate is nonlinear in `x`, and the population log-likelihood in `r` is not log-concave (Gaussian fields). `_point_process_laplace_update` uses the expected Hessian `J' diag(mu) J`, PSD by construction (`point_process_kalman.py:955-976`), which is what makes the update stable for rate-map observations; the observed-Hessian correction would be indefinite here.

## Log-rate interpolation

The 1-D analogue of `_bilinear_log_rate` (`position_decoder.py:802-845`): clamped index, constant extrapolation, so `jax.jacfwd` gives the segment slope inside the grid and exactly zero outside.

```python
def _interp_log_rate(
    represented_position: Array,
    log_rate_maps: Array,
    grid_start: Array,
    grid_step: Array,
) -> Array:
    """Piecewise-linear log rate of every neuron at one track position.

    Parameters
    ----------
    represented_position : Array, shape ()
    log_rate_maps : Array, shape (n_neurons, n_grid)
    grid_start, grid_step : Array, shape ()

    Returns
    -------
    log_rate : Array, shape (n_neurons,)
        Constant beyond the grid ends (clamped index), so its derivative is
        zero there: a represented position off the track carries no
        information about the sweep coefficients.
    """
    n_grid = log_rate_maps.shape[1]
    u = (represented_position - grid_start) / grid_step
    u = jnp.clip(u, 0.0, n_grid - 1 - 1e-9)  # keeps i0 <= n_grid - 2
    i0 = jnp.floor(u).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, n_grid - 1)
    w = u - i0
    return log_rate_maps[:, i0] * (1.0 - w) + log_rate_maps[:, i1] * w


def sweep_log_intensity(
    design_t: Array,
    x: Array,
    *,
    log_rate_maps: Array,
    grid_start: Array,
    grid_step: Array,
) -> Array:
    """log lambda_n(p_t + b_t . x) for all neurons.

    ``design_t = [p_t, b_t]`` (shape ``(1 + K,)``), ``x`` shape ``(K,)``;
    returns ``(n_neurons,)``. This is the ``log_conditional_intensity``
    callable handed to the point-process filter.
    """
    represented_position = design_t[0] + design_t[1:] @ x
    return _interp_log_rate(represented_position, log_rate_maps, grid_start, grid_step)


def sweep_log_intensity_jacobian(
    design_t: Array,
    x: Array,
    *,
    log_rate_maps: Array,
    grid_start: Array,
    grid_step: Array,
) -> Array:
    """Analytic ``d log lambda / dx = l'(r_t) (outer) b_t``, shape (n_neurons, K).

    Equals ``jax.jacfwd(sweep_log_intensity, argnums=1)`` (asserted by a test);
    kept as the readable statement of the observation Jacobian.
    """
    n_grid = log_rate_maps.shape[1]
    r = design_t[0] + design_t[1:] @ x
    u = (r - grid_start) / grid_step
    inside = (u >= 0.0) & (u <= n_grid - 1 - 1e-9)
    u = jnp.clip(u, 0.0, n_grid - 1 - 1e-9)
    i0 = jnp.floor(u).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, n_grid - 1)
    slope = (log_rate_maps[:, i1] - log_rate_maps[:, i0]) / grid_step
    slope = jnp.where(inside, slope, 0.0)
    return slope[:, None] * design_t[None, 1:]
```

`LinearRateMaps.log_intensity` is `functools.partial(sweep_log_intensity, log_rate_maps=self.log_rate_maps, grid_start=jnp.asarray(self.grid_start), grid_step=jnp.asarray(self.grid_step))`, created once in `__init__`. The filter jits with the callable as a static argument (`point_process_kalman.py:1737-1744`); two `partial` objects over the same arrays are *not* equal, so rebuilding the partial per call would recompile every call. The arrays become compile-time constants of the trace, which is fine at `(n_neurons, n_grid) <~ (200, 500)`.

## Design matrix

```python
def build_sweep_design(
    position: ArrayLike,
    phase: ArrayLike,
    direction: ArrayLike,
    n_harmonics: int = 1,
    phase_offset: float | None = None,
) -> np.ndarray:
    """Design rows ``[p_t, b_t]`` for the sweep observation model.

    Parameters
    ----------
    position, phase, direction : array-like, shape (n_time,)
        Linear position (cm), theta phase (radians, any 2*pi range) and
        travel direction (-1, 0 or +1; 0 makes the basis zero).
    n_harmonics : int
        Number of harmonics H; the basis has 2H columns (cos, sin interleaved).
    phase_offset : float or None
        If given (only with ``n_harmonics == 1``) the basis is the single
        column ``d_t cos(phi_t - phase_offset)``.

    Returns
    -------
    design : ndarray, shape (n_time, 1 + K)
    """
    position = np.asarray(position, dtype=float)
    phase = np.asarray(phase, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if not (position.shape == phase.shape == direction.shape) or position.ndim != 1:
        raise ValueError("position, phase and direction must be 1-D arrays of the same length")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")
    if phase_offset is not None:
        if n_harmonics != 1:
            raise ValueError("phase_offset can only be fixed with n_harmonics=1")
        columns = [np.cos(phase - phase_offset)]
    else:
        columns = []
        for h in range(1, n_harmonics + 1):
            columns += [np.cos(h * phase), np.sin(h * phase)]
    basis = direction[:, None] * np.stack(columns, axis=1)
    return np.column_stack([position, basis])
```

## Posterior summaries

`x_t | y ~ N(m_t, P_t)` from the smoother. The amplitude `a = ||(c_1, s_1)||` is a nonlinear, non-negative function, so its posterior is summarised by Monte Carlo with common random numbers across bins (deterministic under the model's `seed`); for isotropic `P` it is a Rice distribution, which the unit test uses as the reference.

```python
def coefficient_posterior_summaries(
    mean: ArrayLike,
    cov: ArrayLike,
    *,
    n_samples: int,
    key: Array,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior median and central 95% interval of sqrt(c_1^2 + s_1^2) and the
    circular posterior mean of atan2(s_1, c_1).

    mean : (n_time, K) with K >= 2; cov : (n_time, K, K).
    Returns (median (n_time,), interval (n_time, 2), phase_offset (n_time,)).
    """
    mean = jnp.asarray(mean)
    cov = jnp.asarray(cov)
    n_time, n_coef = mean.shape
    z = jax.random.normal(key, (n_samples, n_coef))

    @jax.jit
    def _chunk(m, P):
        # psd_cholesky returns (U, False), with P = U.T @ U.
        upper = jax.vmap(lambda p: psd_cholesky(p)[0])(P)
        factor = jnp.swapaxes(upper, -1, -2)  # lower, scale-relative shift
        samples = m[:, None, :] + jnp.einsum("tij,sj->tsi", factor, z)  # (t, s, K)
        amplitude = jnp.hypot(samples[..., 0], samples[..., 1])
        offset = jnp.arctan2(samples[..., 1], samples[..., 0])
        median = jnp.median(amplitude, axis=1)
        lo, hi = jnp.percentile(amplitude, jnp.array([2.5, 97.5]), axis=1)
        circular_mean = jnp.angle(jnp.mean(jnp.exp(1j * offset), axis=1))
        return median, jnp.stack([lo, hi], axis=1), circular_mean

    parts = [
        _chunk(mean[s : s + chunk_size], cov[s : s + chunk_size])
        for s in range(0, n_time, chunk_size)
    ]
    return tuple(np.concatenate([np.asarray(p[i]) for p in parts]) for i in range(3))


def origin_in_credible_region(mean: ArrayLike, cov: ArrayLike, level: float = 0.95) -> np.ndarray:
    """True where the origin lies inside the ``level`` credible ellipsoid:
    ``m' P^{-1} m < chi2_K(level)``. Shape (n_time,)."""
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    solved = np.linalg.solve(cov, mean[..., None])[..., 0]
    return np.einsum("tk,tk->t", mean, solved) < chi2.ppf(level, df=mean.shape[1])
```

`coefficient_correlation = cov[:, 0, 1] / sqrt(cov[:, 0, 0] cov[:, 1, 1])`. In fixed-offset mode (`K = 1`) no Monte Carlo is needed: `amplitude = m[:, 0]`, `amplitude_interval = m[:, 0] -/+ 1.96 sqrt(P[:, 0, 0])`, `phase_offset = psi` everywhere, `coefficient_correlation = NaN`. `psd_cholesky` is `utils.py:114`; its shift is scale-relative so the factor exists for the smoother's PD covariances.

## Null model and score

The null (`a == 0`) is deterministic, so its log-likelihood is exact:

```text
LL_0 = sum_{t in bouts} sum_n log Poisson(y_{n,t}; exp(l_n(p_t)) dt)
```

```python
def null_log_likelihood(
    spikes: ArrayLike,
    position: ArrayLike,
    rate_maps: LinearRateMaps,
    dt: float,
    *,
    max_rate_hz: float,
) -> float:
    """Exact log-likelihood of the no-sweep model on the given bins, with the
    same rate ceiling the sweep model's updates use."""
    log_rate = jax.vmap(rate_maps.log_rate)(jnp.asarray(position))  # (n_time, n_neurons)
    expected = _safe_expected_count(
        log_rate, dt, max_log_count=float(np.log(max_rate_hz * dt))
    )
    return float(jnp.sum(jax.scipy.stats.poisson.logpmf(jnp.asarray(spikes), expected)))
```

`score = log_likelihood - null_log_likelihood`. The sweep model's log-likelihood is the Laplace *marginal* likelihood (`point_process_kalman.py:1192-1208` adds the prior and normaliser terms), which integrates over the random-walk prior; under a true null it is typically slightly below `LL_0` (an Occam penalty), so a positive score is evidence for sweeps only relative to the [phase-shuffle](#phase-shuffle) reference.

## Bout orchestration

`ThetaSweepModel.fit` (phase 1). Every bout is smoothed independently from the same prior; the result arrays are `NaN` outside bouts.

```python
def fit(self, spikes, position, phase, velocity):
    spikes, position, phase, velocity = self._validate_inputs(spikes, position, phase, velocity)
    bouts = identify_behavioral_bouts(
        np.abs(velocity), self.speed_threshold, min_duration=self._min_bout_bins
    )
    if not bouts:
        raise ValueError(
            f"no running bout of at least {self.min_bout_duration} s above "
            f"{self.speed_threshold} cm/s; lower speed_threshold or min_bout_duration"
        )
    fit_state = self._fit_bouts(spikes, position, phase, velocity, bouts)
    result = self._assemble_result(fit_state, bouts, n_time=spikes.shape[0])
    self._result = result
    return result


def _fit_bouts(self, spikes, position, phase, velocity, bouts, *, cycle_phase=None):
    """Smooth every bout; returns per-bout posteriors and the summed log-likelihoods."""
    # Phase 2a uses this original phase for offset resets, including surrogate fits.
    cycle_phase = phase if cycle_phase is None else cycle_phase
    direction = np.sign(velocity)
    posteriors, log_likelihood, null_log_lik = [], 0.0, 0.0
    for start, end in bouts:
        design = build_sweep_design(
            position[start:end], phase[start:end], direction[start:end],
            self.n_harmonics, self.phase_offset,
        )
        mean, cov, cross_cov, ll = theta_sweep_smoother(
            spikes[start:end], design, self.rate_maps, self.dt,
            process_variance=self.process_variance, init_std=self.init_std,
            max_newton_iter=self.max_newton_iter, max_rate_hz=self.max_rate_hz,
        )
        posteriors.append((np.asarray(mean), np.asarray(cov), np.asarray(cross_cov), design))
        log_likelihood += float(ll)
        null_log_lik += null_log_likelihood(
            spikes[start:end], position[start:end], self.rate_maps, self.dt,
            max_rate_hz=self.max_rate_hz,
        )
    return posteriors, log_likelihood, null_log_lik
```

`_validate_inputs` coerces to float64 NumPy arrays and raises `ValueError` for: `spikes` not `(n_time, n_neurons)` counts (`validate_count_array`, `utils.py:1016`), `spikes.shape[1] != rate_maps.n_neurons`, any of `position`/`phase`/`velocity` not `(n_time,)` or non-finite. Phase is passed through `wrap_to_pi` (`circular_stats.py:410`) so downstream code can assume `[-pi, pi]`. `_min_bout_bins = max(1, round(min_bout_duration / dt))`. `_assemble_result` scatters each bout's posterior into full-length `NaN`-initialised arrays, computes `represented_position = design[:, 0] + einsum("tk,tk->t", design[:, 1:], mean)`, then calls [the summaries](#posterior-summaries) on the concatenated running bins and scatters those back; `running_mask` is the union of bouts. Because `|velocity| > threshold > 0` throughout a bout, `d_t` is constant within a bout.

Each distinct bout length triggers one compile of the jitted filter (a shape change), as for the position decoder; document it in the class docstring.

## Phase shuffle

Use independent uniform phase rotations for the original theta cycles within each
bout. A single circular shift of constant-frequency phase only rotates the free
cosine/sine coefficients: the isotropic coefficient prior, amplitude and evidence
are invariant. It is therefore a regression check for rotational symmetry, not a
negative control.

The surrogate below breaks coherence of the spike–phase relation across cycles.
It preserves phase progression within a cycle, but introduces jumps at cycle
boundaries and does not preserve phase autocorrelation. Its reference distribution
depends on the fixed coefficient process variance: a model that can track arbitrary
cycle-to-cycle rotations need not lose amplitude. Report the score distribution;
do not require amplitude to collapse for every dataset or every process variance.

```python
def randomize_phase_within_cycles(
    phase: np.ndarray,
    bouts: list[tuple[int, int]],
    *,
    min_cycle_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = np.array(phase, dtype=float, copy=True)
    for start, end in bouts:
        cycle = theta_cycle_index(phase[start:end], min_cycle_bins)
        n_cycles = int(cycle[-1]) + 1
        if n_cycles < 2:
            raise ValueError("phase surrogates require at least two cycles per bout")
        rotation = rng.uniform(-np.pi, np.pi, n_cycles)
        angle = phase[start:end] + rotation[cycle]
        shuffled[start:end] = np.arctan2(np.sin(angle), np.cos(angle))
    return shuffled
```

`phase_shuffle_scores` re-runs `_fit_bouts` with the surrogate observation phase
and `cycle_phase=original_phase` (same bouts and hyperparameters, `result_`
untouched). It returns `log_likelihood_shuffle - null_log_likelihood` per draw;
the null term does not depend on phase. In phase 2a, offset resets must use
`cycle_phase`, never the discontinuous surrogate phase. `min_cycle_duration`
ships in phase 1 and sets `min_cycle_bins` for both the surrogate and later
offset resets. Use `(1 + count(shuffle_score >= score)) / (n_shuffles + 1)` as
the finite-sample tail fraction, and describe it as a surrogate comparison whose
null calibration is checked by simulation, rather than an exact permutation test.

## Linear rate maps

Occupancy-normalised, Gaussian-smoothed 1-D maps, the counterpart of `PlaceFieldRateMaps.from_spike_position_data` (`position_decoder.py:393-624`), plus a time-bin mask.

```python
@classmethod
def from_spike_position_data(cls, position, spikes, dt, grid, smoothing_sigma=5.0,
                             min_occupancy=0.5, bin_mask=None, rate_floor_hz=0.01):
    from scipy.ndimage import gaussian_filter1d

    position = np.asarray(position, dtype=float)
    spikes = np.asarray(spikes, dtype=float)
    grid = np.asarray(grid, dtype=float)
    # validation: position 1-D finite; spikes (n_time, n_neurons) counts
    # (validate_count_array(..., allow_empty=False)); dt positive
    # (validate_scalar); grid uniform (see __init__); bin_mask bool (n_time,)
    n_time, n_neurons = spikes.shape
    include = np.ones(n_time, dtype=bool) if bin_mask is None else np.asarray(bin_mask, dtype=bool)
    step = grid[1] - grid[0]
    edges = np.concatenate([[grid[0] - step / 2], (grid[:-1] + grid[1:]) / 2, [grid[-1] + step / 2]])
    pos = position[include]
    if np.any((pos < edges[0]) | (pos > edges[-1])):
        warnings.warn(
            f"{int(np.sum((pos < edges[0]) | (pos > edges[-1])))} position samples fall "
            "outside the rate-map grid and are ignored.", StateSpaceWarning, stacklevel=2,
        )
    occupancy = np.histogram(pos, bins=edges)[0] * dt  # seconds per bin
    if occupancy.sum() == 0.0:
        raise ValueError("no included time bin falls on the grid")
    spike_hist = np.stack(
        [np.histogram(pos, bins=edges, weights=spikes[include, n])[0] for n in range(n_neurons)]
    )
    sigma_bins = smoothing_sigma / step
    occupancy_s = gaussian_filter1d(occupancy, sigma_bins, mode="nearest")
    spikes_s = gaussian_filter1d(spike_hist, sigma_bins, axis=1, mode="nearest")
    mean_rate = spike_hist.sum(axis=1) / occupancy.sum()  # (n_neurons,)
    well_sampled = occupancy_s >= min_occupancy
    rates = np.where(
        well_sampled[None, :], spikes_s / np.maximum(occupancy_s, 1e-12), mean_rate[:, None]
    )
    return cls(rates, grid, rate_floor_hz=rate_floor_hz)
```

Rationale for the fallback: a bin the animal barely visited would otherwise get a near-zero or wild rate, and a spike there would dominate the Laplace update (the same failure the decoder's occupancy shrinkage guards against, `position_decoder.py:405-412`).

```python
def phase_restricted_bins(phase, phase_offset, half_width=np.pi / 8):
    """Bins where cos(phi - phase_offset) ~ 0, i.e. the represented position is
    closest to the animal; pass as ``bin_mask`` to refit rate maps free of the
    sweep-induced broadening."""
    relative = np.abs(wrap_to_pi(np.asarray(phase, dtype=float) - phase_offset))
    return np.abs(relative - np.pi / 2) < half_width
```

Documented recipe (one manual iteration): fit maps on all running bins -> fit the sweep model -> take the circular mean of `phase_offset` over running bins -> refit maps with `bin_mask = running & phase_restricted_bins(phase, psi_hat)` -> refit the sweep model. The amplitude should *increase* (broadening removed); the change is the size of the circularity bias.

## Simulator

`src/state_space_practice/simulate/simulate_theta_sweep.py`. Back-and-forth laps at constant speed with pauses at both ends (excluded by the speed threshold), constant theta frequency, and spikes drawn from the *model's own* interpolated rate maps at the represented position.

```python
def simulate_linear_track_theta_sweep(
    *,
    n_time: int = 15000,
    dt: float = 0.004,
    track_length: float = 150.0,
    speed: float = 20.0,
    pause_duration: float = 0.5,
    n_neurons: int = 40,
    field_width: float = 8.0,
    peak_rate: float = 20.0,
    baseline_rate: float = 0.5,
    grid_step: float = 1.0,
    theta_frequency: float = 8.0,
    amplitude: float | ArrayLike = 15.0,   # scalar or (n_time,) schedule
    phase_offset: float = 0.5,
    cycle_offset_std: float = 0.0,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    time = np.arange(n_time) * dt
    lap = track_length / speed
    period = 2.0 * (lap + pause_duration)
    tau = time % period
    position = np.zeros(n_time)
    velocity = np.zeros(n_time)
    outbound = tau < lap
    position[outbound] = speed * tau[outbound]
    velocity[outbound] = speed
    far_pause = (tau >= lap) & (tau < lap + pause_duration)
    position[far_pause] = track_length
    inbound = (tau >= lap + pause_duration) & (tau < 2 * lap + pause_duration)
    position[inbound] = track_length - speed * (tau[inbound] - lap - pause_duration)
    velocity[inbound] = -speed
    # tau >= 2 lap + pause: at position 0, velocity 0 (already zero)
    direction = np.sign(velocity)
    phase = wrap_to_pi(2.0 * np.pi * theta_frequency * time)
    amplitude = np.broadcast_to(np.asarray(amplitude, dtype=float), (n_time,)).copy()
    cycle_index = np.floor(theta_frequency * time + 0.5).astype(int)  # increments at the +pi -> -pi wrap
    n_cycles = int(cycle_index.max()) + 1
    per_cycle = rng.normal(0.0, cycle_offset_std, n_cycles) if cycle_offset_std > 0 else np.zeros(n_cycles)
    true_cycle_offset = per_cycle[cycle_index]
    represented = position + direction * (amplitude * np.cos(phase - phase_offset) + true_cycle_offset)
    grid = np.arange(0.0, track_length + grid_step / 2, grid_step)
    centers = np.linspace(field_width, track_length - field_width, n_neurons)
    rate_maps = baseline_rate + peak_rate * np.exp(-0.5 * ((grid[None, :] - centers[:, None]) / field_width) ** 2)
    maps = LinearRateMaps(rate_maps, grid)
    log_rate = np.asarray(jax.jit(jax.vmap(maps.log_rate))(jnp.asarray(represented)))
    spikes = rng.poisson(np.exp(log_rate) * dt).astype(float)
    return {
        "dt": dt, "time": time, "position": position, "velocity": velocity, "phase": phase,
        "direction": direction, "spikes": spikes, "rate_maps": maps,
        "true_amplitude": amplitude,
        "true_coefficients": np.stack([amplitude * np.cos(phase_offset), amplitude * np.sin(phase_offset)], axis=1),
        "true_phase_offset": phase_offset, "true_cycle_offset": true_cycle_offset,
        "cycle_index": cycle_index, "represented_position": represented,
    }
```

Amplitude schedules used by the tests: constant (`15.0`); step (`np.where(time < time[-1] / 2, 8.0, 20.0)`); drift (`np.linspace(5.0, 25.0, n_time)`); null (`0.0`).

## Oracle generalisation

`tests/test_oracle_point_process.py` computes the exact posterior by quadrature but hard-codes the affine log-rate in two places. Generalise without changing existing behaviour:

- `_Problem` (`:81-88`) gains three optional trailing fields: `log_rate: Callable | None = None` — the JAX `(design_t, x) -> (n_neurons,)` callable for the library — `grid_log_rate: Callable | None = None` — a NumPy `(points (n_points, d), design_t) -> (n_points, n_neurons)` callable for the grid — and `rate_maps: object | None = None` (the `LinearRateMaps` the sweep reference reads). `NamedTuple` allows trailing defaults, so existing constructors are unchanged.
- `_run_laplace` (`:124-146`): replace the literal `_affine_log_rate` at `:134` with `problem.log_rate or _affine_log_rate`.
- `_grid_posterior` (`:191-240`): replace `:210-211` with `log_rate = (problem.grid_log_rate or _affine_grid_log_rate)(points, design_t)` where

  ```python
  def _affine_grid_log_rate(points, design_t):
      return design_t[:, 0][None, :] + points @ design_t[:, 1:].T
  ```

  This reproduces the existing expression exactly, so every existing test's numbers are unchanged.
- `_log_evidence_terms` / `_laplace_filter_reference` (`:261-316`) stay affine-only; the sweep problem gets its own NumPy reference:

```python
def _sweep_log_evidence_terms(problem, design_t, y_t, x):
    """Poisson log-pmf, score and Fisher information of the sweep model at x
    (NumPy; interpolation by np.interp, slopes from the grid segments)."""
    maps = problem.rate_maps  # LinearRateMaps stashed on the problem by _sweep_problem
    r = design_t[0] + design_t[1:] @ x
    log_maps = np.asarray(maps.log_rate_maps)
    log_rate = np.array([np.interp(r, maps.grid, lm) for lm in log_maps])
    mu = np.exp(log_rate) * problem.dt
    u = np.clip((r - maps.grid_start) / maps.grid_step, 0.0, maps.grid.size - 1 - 1e-9)
    i0 = int(np.floor(u)); i1 = min(i0 + 1, maps.grid.size - 1)
    inside = maps.grid[0] <= r <= maps.grid[-1]
    slope = (log_maps[:, i1] - log_maps[:, i0]) / maps.grid_step if inside else np.zeros(len(log_maps))
    jac = slope[:, None] * design_t[None, 1:]
    logpmf = np.sum(y_t * np.log(mu) - mu - gammaln(y_t + 1.0))
    return logpmf, jac.T @ (y_t - mu), jac.T @ (mu[:, None] * jac)
```

  and a `_sweep_laplace_filter_reference` that is `_laplace_filter_reference` with `_log_evidence_terms` replaced by this (same recursion, `:279-316`). With `max_newton_iter=1` the library update is a single Fisher step from the prior mean, so it must match to roundoff; the converged update matches to `~1e-6` when no bin's line search rejects a step (the reference has no line search).

The sweep oracle problem (scalar amplitude, fixed offset `psi = 0`, so the latent is 1-D):

```python
def _sweep_problem(seed, n_time=6, dt=0.01, peak_rate=2000.0, amplitude=10.0, init_std=2.0, q=0.05):
    from state_space_practice.theta_sweep import LinearRateMaps, build_sweep_design

    rng = np.random.default_rng(seed)
    grid = np.arange(0.0, 80.0 + 0.5, 1.0)
    centers = np.arange(8.0, 80.0, 8.0)  # nine fields tiling the segment
    maps = LinearRateMaps(1.0 + peak_rate * np.exp(-0.5 * ((grid[None] - centers[:, None]) / 6.0) ** 2), grid)
    position = 30.0 + 2.0 * np.arange(n_time)
    phase = 2.0 * np.pi * np.arange(n_time) / n_time  # spread over one cycle
    design = build_sweep_design(position, phase, np.ones(n_time), n_harmonics=1, phase_offset=0.0)
    x = rng.normal(amplitude, init_std)
    spikes = []
    for t in range(n_time):
        x = x + rng.normal(0.0, np.sqrt(q))
        r = design[t, 0] + design[t, 1] * x
        spikes.append(rng.poisson(np.exp(np.asarray(maps.log_rate(r))) * dt))

    def grid_log_rate(points, design_t):  # points (n_points, 1)
        r = design_t[0] + points @ design_t[1:]
        return np.stack([np.interp(r, maps.grid, lm) for lm in np.asarray(maps.log_rate_maps)], axis=1)

    return _Problem(
        init_mean=np.array([amplitude]), init_cov=np.array([[init_std**2]]),
        a_diag=np.ones(1), q_diag=np.array([q]), design=design,
        spikes=np.asarray(spikes, dtype=float), dt=dt,
        log_rate=maps.log_intensity, grid_log_rate=grid_log_rate, rate_maps=maps,
    )
```

The exact grid over `a` is `np.linspace(lo, hi, 3001)` with `lo/hi` = Laplace filtered mean `-/+ 12 sd` (as `TestLaplaceIsAsymptoticallyExact` does at `:582-586`), and the transition kernel `_gauss_kernel(grid, a=1.0, q)` is the random walk. The `design` here is `(T, 2)`; `_grid_posterior` only iterates its leading axis, and the callable interprets the row.

## Cycle boundaries

*(Phase 1 helper, reused in phase 2a.)* Theta cycles start when unwrapped phase
crosses `pi + 2 pi k`, matching the simulator's `+pi -> -pi` boundary. Cycles
shorter than `min_cycle_bins` (phase jitter across the wrap) are merged into the
previous one. NumPy, at the model boundary. Adding any integer multiple of
`2 pi` to input samples must not change the detected boundaries.

```python
def theta_cycle_index(phase: ArrayLike, min_cycle_bins: int = 1) -> np.ndarray:
    """Zero-based theta-cycle index per bin of one contiguous segment.

    Parameters
    ----------
    phase : array-like, shape (n_time,)
        Theta phase in radians, any 2*pi range, advancing with time.
    min_cycle_bins : int
        A detected cycle start closer than this to the previous kept start is
        discarded (the bins join the previous cycle).

    Returns
    -------
    cycle_index : ndarray of int, shape (n_time,)
        Starts at 0 and increments by one at every kept cycle start.
    """
    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 1 or not np.all(np.isfinite(phase)):
        raise ValueError("phase must be a finite 1-D array")
    unwrapped = np.unwrap(phase)
    if phase.size > 1 and unwrapped[-1] <= unwrapped[0]:
        raise ValueError("theta phase must advance over the segment")
    turns = np.floor((unwrapped + np.pi) / (2.0 * np.pi))
    candidates = np.flatnonzero(np.diff(turns) > 0) + 1  # first bin of each new turn
    kept, last = [], 0
    for start in candidates:
        if start - last >= min_cycle_bins:
            kept.append(start)
            last = start
    index = np.zeros(phase.shape[0], dtype=int)
    index[kept] = 1
    return np.cumsum(index)


def cycle_starts_from_index(cycle_index: np.ndarray) -> np.ndarray:
    """True at the first bin of every cycle, including the segment's first bin."""
    starts = np.ones(cycle_index.shape[0], dtype=bool)
    starts[1:] = cycle_index[1:] != cycle_index[:-1]
    return starts
```

The model computes `min_cycle_bins = max(1, round(min_cycle_duration / dt))` (default 50 ms: a 12 Hz cycle is 83 ms). Cycle indices are per bout; the result's `cycle_index` is made unique across bouts by offsetting each bout by the running total and is `-1` outside bouts.

## Time-varying dynamics

*(Phase 2a.)* The offset state `epsilon` is appended to `x`: state `(x, epsilon) in R^{K+1}`, basis column `d_t` appended to the design ([layout](shared-contracts.md#design-matrix-layout)), so `r_t = p_t + b_t . x + d_t epsilon_t`. Per bin, with `s_t = 1` at cycle starts:

```text
A_t = blockdiag(I_K, 1 - s_t),     Q_t = blockdiag(q dt I_K, sigma_c^2 s_t)
```

Inside a cycle `epsilon_t = epsilon_{t-1}` (deterministic: `A = 1`, `Q = 0`); at a cycle start `epsilon_t ~ N(0, sigma_c^2)` independent of the past. `Q_t` is singular inside cycles, which is fine: the update factors the *predicted* covariance `A_t P A_t' + Q_t`, whose `epsilon` entry is `P_{epsilon epsilon} > 0` inside a cycle and `sigma_c^2 > 0` at a start. The first bin of a bout is a cycle start, so the prior's `epsilon` entry only needs to be positive definite (set to `sigma_c^2`).

```python
def sweep_transition_stacks(
    cycle_starts: np.ndarray | None,
    n_coefficients: int,
    process_variance: float,
    cycle_offset_variance: float | None,
    dt: float,
    n_time: int,
) -> tuple[Array, Array]:
    """Per-bin (A_t, Q_t), shapes (n_time, D, D); A_t/Q_t is the transition INTO bin t."""
    K = n_coefficients
    if cycle_starts is None:
        A = np.broadcast_to(np.eye(K), (n_time, K, K))
        Q = np.broadcast_to(process_variance * dt * np.eye(K), (n_time, K, K))
        return jnp.asarray(A), jnp.asarray(Q)
    D = K + 1
    starts = np.asarray(cycle_starts, dtype=float)
    A = np.tile(np.eye(D), (n_time, 1, 1))
    A[:, K, K] = 1.0 - starts
    Q = np.zeros((n_time, D, D))
    Q[:, np.arange(K), np.arange(K)] = process_variance * dt
    Q[:, K, K] = cycle_offset_variance * starts
    return jnp.asarray(A), jnp.asarray(Q)
```

Forward scan — `_stochastic_point_process_filter_impl` (`point_process_kalman.py:1745-1866`) with the transition and process covariance as scanned inputs, following `position_decoder._run_filter_scan` (`:969-1178`) for the direct `_point_process_laplace_update` call:

```python
@functools.partial(jax.jit, static_argnames=("log_intensity", "max_newton_iter"))
def _sweep_filter_scan(
    init_mean: Array, init_cov: Array, design: Array, spikes: Array,
    transition_stack: Array, process_cov_stack: Array,
    *, dt: float, max_log_count: float,
    log_intensity: Callable[[Array, Array], Array], max_newton_iter: int,
) -> tuple[Array, Array, Array]:
    """Laplace-EKF forward pass with per-bin dynamics. Returns
    (filtered_mean (T, D), filtered_cov (T, D, D), log_likelihood)."""
    grad_log_intensity = jax.jacfwd(log_intensity, argnums=1)

    def _step(carry, args):
        mean_prev, cov_prev, total_ll, n_failed_bins = carry
        design_t, spikes_t, A_t, Q_t = args
        one_step_mean = A_t @ mean_prev
        one_step_cov = symmetrize(A_t @ symmetrize(cov_prev) @ A_t.T + Q_t)
        post_mean, post_cov, ll, n_failed = _point_process_laplace_update(
            one_step_mean, one_step_cov, spikes_t, dt,
            lambda x: log_intensity(design_t, x),
            grad_log_intensity_func=lambda x: grad_log_intensity(design_t, x),
            include_laplace_normalization=True, max_log_count=max_log_count,
            max_newton_iter=max_newton_iter, return_line_search_failures=True,
        )
        carry = (post_mean, post_cov, total_ll + ll, n_failed_bins + (n_failed > 0).astype(jnp.int32))
        return carry, (post_mean, post_cov)

    init = (init_mean, init_cov, jnp.zeros((), init_mean.dtype), jnp.zeros((), jnp.int32))
    (_, _, log_likelihood, n_failed_bins), (filtered_mean, filtered_cov) = jax.lax.scan(
        _step, init, (design, spikes, transition_stack, process_cov_stack)
    )
    _warn_line_search_failures(n_failed_bins, spikes.shape[0], max_newton_iter, "theta_sweep_smoother")
    return filtered_mean, filtered_cov, log_likelihood
```

`theta_sweep_smoother` then builds the stacks, validates host-side when inputs are concrete (`contains_tracer`, `utils.py:910`; `validate_count_array`; `_validate_filter_numerics(init_cov, n_time, filter_name="theta_sweep_smoother")`, `utils.py:1187`), runs the scan and finishes with

```python
smoother_mean, smoother_cov, smoother_cross_cov = parallel_kalman_smoother(
    filtered_mean, filtered_cov, transition_stack[1:], process_cov_stack[1:]
)
```

because `parallel_kalman_smoother`'s `transition_matrix[t]` maps `t -> t+1` (`kalman.py:1090-1096`) while the stacks index the transition *into* bin `t`. With constant stacks this is algebraically the sequential RTS pass the library smoother runs (`point_process_kalman.py:2605-2641`), and the phase 2a equivalence test pins the two to `rtol 1e-7`.

## Hyperparameter M-steps

*(Phase 2b.)* Two scalars: the coefficient process variance `q` and the cycle-offset variance `sigma_c^2`. The initial prior `(0, init_std^2 I_K)` is not updated (it is shared by all bouts and is meant to be weakly informative).

**Complete-data log-likelihood terms** for one bout with bins `1..T` and prior state `x_0`:

```text
sum_{t=1}^{T} log N(u_t; u_{t-1}, q dt I_K)  +  sum_{t: s_t = 1} log N(epsilon_t; 0, sigma_c^2)
```

where `u` is the coefficient block; inside a cycle `epsilon_t = epsilon_{t-1}` deterministically and contributes nothing. Taking posterior expectations and maximising:

```text
q_new         = sum_b sum_{t=1}^{T_b} tr E[(u_t - u_{t-1})(u_t - u_{t-1})'] / (K dt sum_b T_b)
sigma_c^2_new = sum_b sum_{t: s_t=1} (m_{t,eps}^2 + P_{t,eps eps}) / (number of cycle starts)
```

The `q` update includes the `x_0 -> x_1` transition exactly, as `PlaceFieldModel._m_step` does (`place_field_model.py:1076-1098`): the smoothed `x_0` moments come from `smooth_initial_state_with_cross_cov` (`kalman.py:1320-1361`) applied to the coefficient block alone, which is exact because at the first bin `A_1 = blockdiag(I_K, 0)`, `Q_1 = blockdiag(q dt I_K, sigma_c^2)` and `P_0` are all block-diagonal, so the smoother gain `J_0` is block-diagonal with a zero `epsilon` block.

```python
def coefficient_process_variance_update(bout_posteriors, prior: InitialStatePrior, n_coefficients: int, dt: float) -> float:
    """Exact EM update of q from per-bout smoother outputs (x_0 -> x_1 included)."""
    K = n_coefficients
    numerator, n_transitions = 0.0, 0
    for mean, cov, cross_cov in bout_posteriors:  # (T, D), (T, D, D), (T-1, D, D)
        m0, P0, C01 = smooth_initial_state_with_cross_cov(prior, mean[0, :K], cov[0, :K, :K])
        means = jnp.concatenate([m0[None], mean[:, :K]], axis=0)  # (T + 1, K)
        sum_cov = cov[:, :K, :K].sum(axis=0)
        Q_b = process_cov_residual_form(
            means,
            sum_next_cov=sum_cov,
            sum_prev_cov=sum_cov - cov[-1, :K, :K] + P0,
            sum_cross_cov=cross_cov[:, :K, :K].sum(axis=0) + C01,
            transition_matrix=jnp.eye(K),
        )  # divides by T_b transitions
        T_b = mean.shape[0]
        numerator += float(jnp.trace(Q_b)) * T_b
        n_transitions += T_b
    return numerator / (K * dt * n_transitions)


def cycle_offset_variance_update(bout_posteriors, bout_cycle_starts) -> float:
    """Exact EM update of sigma_c^2: mean posterior second moment of epsilon at cycle starts."""
    numerator, count = 0.0, 0
    for (mean, cov, _), starts in zip(bout_posteriors, bout_cycle_starts):
        numerator += float(jnp.sum(mean[starts, -1] ** 2 + cov[starts, -1, -1]))
        count += int(np.sum(starts))
    return numerator / count
```

`prior = InitialStatePrior(jnp.zeros(K), init_std**2 * jnp.eye(K), jnp.eye(K), q_old * dt * jnp.eye(K))` — the parameters the E-step ran with. For `q`, the isotropic constraint's maximiser is the mean diagonal of the unconstrained update (`place_field_model.py:1120-1127`); `trace / K` is that mean.

**EM loop** (`fit(..., fit_hyperparameters=True)`), through `run_em` (`em_driver.py:58`) configured as `PointProcessModel.fit` does (`point_process_kalman.py:3219-3230`):

```python
def _e_step():
    posteriors, ll, null_ll = self._fit_bouts(spikes, position, phase, velocity, bouts)
    state["posteriors"], state["log_likelihood"], state["null_log_likelihood"] = posteriors, ll, null_ll
    return ll

def _m_step():
    prior = InitialStatePrior(jnp.zeros(K), self.init_std**2 * jnp.eye(K), jnp.eye(K), self.process_variance * self.dt * jnp.eye(K))
    self.process_variance = coefficient_process_variance_update([p[:3] for p in state["posteriors"]], prior, K, self.dt)
    if self.cycle_offset:
        self.cycle_offset_variance = max(
            cycle_offset_variance_update([p[:3] for p in state["posteriors"]], state["cycle_starts"]), 1e-8
        )

def _snapshot():
    return {"process_variance": self.process_variance, "cycle_offset_variance": self.cycle_offset_variance, **state}

def _restore(snapshot):
    self.process_variance = snapshot["process_variance"]; self.cycle_offset_variance = snapshot["cycle_offset_variance"]
    state.update({k: snapshot[k] for k in ("posteriors", "log_likelihood", "null_log_likelihood")})

em = run_em(_e_step, _m_step, _snapshot, _restore, max_iter=max_iter, tol=tolerance,
            on_first_nonfinite="clear", clear_state=lambda: state.clear(), logger=logger)
```

The `sigma_c^2` floor of `1e-8 cm^2` keeps the prior's `epsilon` entry positive definite (the input validator raises on a singular initial covariance); `q` needs no floor (`Q = 0` is PSD and the predicted covariance stays PD). The result carries `process_variance`, `cycle_offset_variance` and `em_log_likelihoods = em.log_likelihoods`.

## Cycle-variance test

*(Phase 2b.)* Null: the phase 1 model (no offset state, which *is* `sigma_c^2 = 0`) with `q` fitted by EM. Alternative: the offset model with `(q, sigma_c^2)` fitted by EM on the same bouts. Statistic `LR = 2 (LL_alt - LL_null)`. Because `sigma_c^2 = 0` is on the boundary of the parameter space, the asymptotic null distribution is `0.5 delta_0 + 0.5 chi^2_1` (Self & Liang 1987, case of one boundary parameter), so `p = 0.5 P(chi^2_1 > LR)` for `LR > 0` and `p = 1` otherwise.

```python
class CycleVarianceTest(NamedTuple):
    statistic: float          # max(2 (LL_alt - LL_null), 0)
    p_value: float            # boundary-mixture asymptotic p-value
    cycle_offset_variance: float


def cycle_variance_test(null_result: ThetaSweepResult, alternative_result: ThetaSweepResult) -> CycleVarianceTest:
    if null_result.cycle_offset_mean is not None:
        raise ValueError("null_result must come from a model without the cycle-offset state")
    if alternative_result.cycle_offset_variance is None:
        raise ValueError("alternative_result must come from fit(..., fit_hyperparameters=True) with cycle_offset=True")
    if not np.array_equal(null_result.running_mask, alternative_result.running_mask):
        raise ValueError("both results must be fit on the same running bins")
    statistic = 2.0 * (alternative_result.log_likelihood - null_result.log_likelihood)
    p_value = 1.0 if statistic <= 0.0 else 0.5 * float(chi2.sf(statistic, df=1))
    return CycleVarianceTest(max(statistic, 0.0), p_value, alternative_result.cycle_offset_variance)
```

The Laplace marginal likelihood is approximate, so the asymptotic reference is approximate. The smoke script adds a plug-in parametric bootstrap: draw `y* ~ Poisson(exp(l(p_t + b_t . m_t)) dt)` on the running bins from the null fit's smoothed coefficient path, refit both models, record `LR*`, repeat `B` times, `p_boot = mean(LR* >= LR)`. It stays in the script (see [overview open question 6](overview.md#open-questions)).
