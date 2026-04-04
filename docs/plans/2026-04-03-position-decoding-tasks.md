# Position Decoding — Task Breakdown

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.
>
> **Execution mode:** Finish one task completely before starting the next one. If any prerequisite gate or verification gate fails, stop and resolve that issue before continuing.

**Goal:** Implement the position decoder described in `docs/plans/2026-04-03-position-decoding.md`.

**Design doc:** `docs/plans/2026-04-03-position-decoding.md`

**Key files:**

- Create: `src/state_space_practice/position_decoder.py`
- Create: `src/state_space_practice/tests/test_position_decoder.py`
- Reference: `src/state_space_practice/point_process_kalman.py` (`_point_process_laplace_update`)
- Reference: `src/state_space_practice/kalman.py` (`_kalman_smoother_update`, `symmetrize`, `psd_solve`)
- Reference: `src/state_space_practice/place_field_model.py` (`PlaceFieldModel`, `evaluate_basis`)

**Prerequisite Gates:**

- Verify `_point_process_laplace_update` signature in `point_process_kalman.py` before implementation.
- Verify `_kalman_smoother_update` signature in `kalman.py`.
- Verify `PlaceFieldModel.make_grid()` and `PlaceFieldModel.predict_rate_map()` exist and work.

**Verification Gates:**

- Targeted tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_position_decoder.py -v`
- Neighbor regression tests: `conda run -n state_space_practice pytest src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v`
- Lint: `conda run -n state_space_practice ruff check src/state_space_practice`

**Critical design decisions:**

- The place field rate maps are stored on a grid and interpolated via bilinear interpolation. This is **not JIT-compatible** due to numpy indexing. The filter will run without JIT. This is an intentional tradeoff: generality (any rate map source) over speed.
- Jacobians are computed via finite differences on the interpolated rate maps, not analytically. This avoids depending on spline basis derivatives but introduces discretization error proportional to grid spacing.
- The Hessian of log-rate w.r.t. position is set to zero (Gauss-Newton approximation). This means the Fisher information matrix is used as the observation precision, which is standard for EKF with Poisson observations.
- The `from_spike_position_data` constructor uses `scipy.ndimage.gaussian_filter` for KDE smoothing. This is a soft dependency — if scipy is not available, the user must provide pre-computed rate maps.

**Known limitations (document, don't fix):**

- The Gaussian (Laplace) posterior approximation fails when the true posterior is multimodal (e.g., symmetric environments, very few neurons). This is inherent to the EKF approach. Particle filters would be needed for multimodal cases.
- Without JIT, decoding is ~10-100x slower than a JIT-compiled version. For long recordings, consider downsampling or using the `PositionDecoder` on segments.
- The `from_spike_position_data` KDE uses `np.histogram2d` which produces `(n_grid-1, n_grid-1)` bins, so the rate map grid has one fewer point per dimension than the input `n_grid`. This is handled by using bin centers.

**MVP Scope Lock:**

- Implement: `build_position_dynamics`, `PlaceFieldRateMaps`, `position_decoder_filter`, `position_decoder_smoother`, `PositionDecoder` class.
- Require: synthetic circular trajectory decode with correlation > 0.5 after warmup.
- Require: smoother error <= filter error.

**Defer:**

- JIT-compatible interpolation
- EM for learning q_pos/q_vel
- Particle filter variant

---

## Task 1: Position Dynamics

Implement `build_position_dynamics`. Follow the design doc Task 1 exactly.

### Tests to write:

- `test_constant_velocity_shapes`: A is (4,4), Q is (4,4)
- `test_constant_velocity_prediction`: position + velocity*dt
- `test_q_is_psd`: eigenvalues >= 0
- `test_position_only_mode`: A is (2,2) identity when include_velocity=False
- `test_dt_validation`: dt <= 0 should raise ValueError

### Verification checkpoint:

- [ ] All tests pass
- [ ] `ruff check` passes
- [ ] A @ [x, y, vx, vy] = [x + vx*dt, y + vy*dt, vx, vy]

### Commit:

```bash
git commit -m "Add position dynamics for spike-based decoder"
```

---

## Task 2: PlaceFieldRateMaps

Implement the `PlaceFieldRateMaps` class with `log_rate`, `log_rate_jacobian`, `from_place_field_model`, and `from_spike_position_data`.

### Tests to write:

- `test_log_rate_at_field_center`: highest rate near field center
- `test_log_rate_away_from_field`: rate near baseline far away
- `test_log_rate_clamps_at_boundary`: positions outside grid are clamped
- `test_jacobian_shape`: (n_neurons, 2)
- `test_jacobian_points_toward_field`: gradient direction is correct
- `test_jacobian_finite_difference_accuracy`: compare to numerical Jacobian from `log_rate`
- `test_from_place_field_model`: constructs from fitted PlaceFieldModel
- `test_from_spike_position_data`: constructs from raw position + spikes
- `test_from_spike_position_data_shape`: rate_maps shape is (n_neurons, n_grid-1, n_grid-1)

### Critical: Jacobian accuracy test

```python
def test_jacobian_matches_numerical(self, simple_fields):
    """Finite-difference Jacobian from log_rate_jacobian should match
    a second independent numerical derivative."""
    pos = jnp.array([45.0, 55.0])
    jac = simple_fields.log_rate_jacobian(pos)

    # Independent numerical derivative with different epsilon
    eps = 0.01
    for dim in range(2):
        pos_p = pos.at[dim].set(pos[dim] + eps)
        pos_m = pos.at[dim].set(pos[dim] - eps)
        numerical = (simple_fields.log_rate(pos_p) - simple_fields.log_rate(pos_m)) / (2 * eps)
        np.testing.assert_allclose(jac[:, dim], numerical, atol=0.1)
```

### Verification checkpoint:

- [ ] All tests pass
- [ ] Jacobian accuracy test passes (atol=0.1, given grid discretization)
- [ ] `from_place_field_model` produces rate maps with correct peak location
- [ ] `from_spike_position_data` handles multi-neuron input

### Commit:

```bash
git commit -m "Add PlaceFieldRateMaps with interpolation and Jacobians"
```

---

## Task 3: Position Decoder Filter and Smoother

Implement `position_decoder_filter` and `position_decoder_smoother`.

### Tests to write:

- `test_filter_output_shapes`: (n_time, 4) mean, (n_time, 4, 4) cov
- `test_filter_tracks_position`: correlation > 0.5 with true position (after warmup)
- `test_filter_covariance_is_psd`: all eigenvalues >= 0
- `test_filter_ll_is_finite`: marginal LL is finite
- `test_smoother_output_shapes`
- `test_smoother_reduces_error`: smoother error <= filter error * 1.1
- `test_smoother_last_matches_filter`: smoother[-1] == filter[-1]
- `test_smoother_variance_leq_filter`: average smoothed variance <= filtered

### Critical: Baseline parity test

```python
def test_stationary_animal_converges(self, decoding_data):
    """Animal at fixed position: decoded position should converge to
    true position with decreasing uncertainty."""
    # Create data where animal stays at (50, 50)
    # Generate spikes from known rate maps at (50, 50)
    # After warmup, decoded position should be within 10cm of (50, 50)
    # and posterior variance should be smaller than initial variance
```

### Verification checkpoint:

- [ ] All tests pass
- [ ] Synthetic circular trajectory: filter correlation > 0.5, smoother > 0.6
- [ ] No NaN/Inf in any output
- [ ] Smoother variance <= filter variance
- [ ] Stationary animal test passes (convergence)
- [ ] `ruff check` passes

### Commit:

```bash
git commit -m "Add Laplace-EKF position decoder filter and smoother"
```

---

## Task 4: PositionDecoder Model Class

Wrap into user-facing class with `fit`, `decode`, `plot_decoding`.

### Tests to write:

- `test_fit_decode_workflow`: fit → decode → check shapes
- `test_decode_error`: median error < 30cm on synthetic data
- `test_decode_requires_fit`: RuntimeError before fit
- `test_filter_vs_smoother_methods`: both "filter" and "smoother" work
- `test_invalid_method_raises`: method="invalid" raises ValueError
- `test_plot_decoding`: returns figure (smoke test)
- `test_plot_without_true_position`: still produces figure

### Additional methods to implement:

- `__repr__`: show dt, n_neurons, fitted status
- Input validation: spikes shape must match rate_maps.n_neurons

### Verification checkpoint:

- [ ] All tests pass
- [ ] Full end-to-end: simulate data → fit → decode → plot
- [ ] `ruff check` passes
- [ ] Neighbor regression tests pass:
  ```bash
  conda run -n state_space_practice pytest src/state_space_practice/tests/test_place_field_model.py src/state_space_practice/tests/test_point_process_kalman.py -v
  ```

### Commit:

```bash
git commit -m "Add PositionDecoder model class with fit/decode/plot"
```
