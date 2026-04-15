# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`state_space_practice` is a JAX-based library for state-space modeling, focusing on:

- Kalman filtering and smoothing
- Point-process (spike) observation models with Laplace-EKF
- Switching state-space models (SLDS)
- Coupled oscillator networks with directed influence models

## Development Environment

- **Conda environment**: `state_space_practice`
- **Python version**: 3.10-3.12
- **Activate**: `conda activate state_space_practice`

## Common Commands

**Important**: Always run pytest, mypy, and ruff via the conda environment to ensure correct dependencies.

```bash
# Run all tests
conda run -n state_space_practice pytest src/state_space_practice/tests/

# Run specific test file
conda run -n state_space_practice pytest src/state_space_practice/tests/test_kalman.py -v

# Run tests with coverage
conda run -n state_space_practice pytest --cov=src/state_space_practice --cov-report=term-missing

# Run linter
conda run -n state_space_practice ruff check src/

# Run type checker (if configured)
conda run -n state_space_practice mypy src/state_space_practice/
```

## Project Structure

```
src/state_space_practice/
├── kalman.py                  # Core Kalman filter/smoother, EM utilities
├── switching_kalman.py        # Switching (SLDS) Kalman filter/smoother
├── point_process_kalman.py    # Point-process observation model (Laplace-EKF)
├── switching_point_process.py # Switching point-process model (in development)
├── oscillator_utils.py        # Oscillator transition matrix construction
├── oscillator_models.py       # Coupled oscillator model classes
├── models.py                  # General state-space model classes
├── utils.py                   # Shared utilities
├── simulate_data.py           # Data simulation utilities
├── simulate/                  # Additional simulation modules
└── tests/                     # pytest test suite
```

## Code Patterns

### JAX Conventions

- Use `jax.numpy` (imported as `jnp`) instead of NumPy for array operations
- Functions should be JIT-compatible; use `jax.lax.scan` for loops
- Use `vmap` for batching operations across array dimensions
- Arrays use shape convention: `(n_latent, n_time)` or `(n_latent, n_latent, n_discrete_states)`
- use jax skill

### Type Annotations

- Use `ArrayLike` for function inputs (accepts numpy/jax arrays)
- Use `Array` (jax.Array) for return types
- Document shapes in docstrings: `mean: Array of shape (n_latent,)`

### Testing

- Tests use `pytest` with fixtures in `conftest.py`
- Use `hypothesis` for property-based testing where appropriate
- Test numerical properties (PSD covariances, probabilities sum to 1, etc.)
- Mark any test that runs EM, SGD, or full filter/smoother pipelines with `@pytest.mark.slow`. Fast suite (`-m "not slow"`) should finish in under a minute.
- **Prefer behavioral assertions over shape/type checks.** A test that only checks `.shape` or `isinstance` on a deterministic constructor will never catch a real bug. Instead test *meaning*: "smoother estimate is closer to truth than prior", "chosen option value increases", "LL improves over iterations".
- **Use fixtures for shared setup.** If 3+ tests construct the same parameters, extract a fixture. Don't duplicate 20 lines of boilerplate per test.
- **Statistical tests must actually test statistics.** Don't assert `0 < mean < 1` and call it a Poisson test. Use `assert_allclose` with a meaningful tolerance, or a proper goodness-of-fit test.
- **Every test must be able to fail.** If an assertion is vacuously true (e.g., `x <= x` when both sides return the same default), the test provides no signal. Add a guard assertion that the interesting condition was actually reached.

## Numerical precision

The Laplace-EKF filter (`stochastic_point_process_filter`,
`stochastic_point_process_smoother`, and all callers including
`PlaceFieldModel.fit` / `fit_sgd`) requires **float64** for long
sequences. In float32, accumulated roundoff in the covariance
propagation can drive the posterior covariance to lose PSD after a
few hundred to a few thousand time bins — the exact number depends on
the condition number of `init_cov`. The symptom is a silent NaN
somewhere in the forward pass.

**Always enable x64 before importing this library:**

```python
import jax
jax.config.update("jax_enable_x64", True)

# NOW import — the x64 flag must be set before any jax.numpy allocations
from state_space_practice import PlaceFieldModel
```

The filter validates `init_cov` at the top of every public entry
point. On a non-PSD prior it raises `ValueError` (the filter is
guaranteed to NaN). On f32 + long T + ill-conditioned `init_cov` it
emits a `UserWarning` pointing at the import-order recipe above.

Test suites always run with x64 enabled (see `jax.config.update` in
`tests/conftest.py`), so regression tests do not exercise the f32 NaN
path. Production users who omit the x64 flag will see the warning on
their first `fit` / `fit_sgd` call if their problem is at risk.
