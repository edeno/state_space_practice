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
- **Python version**: 3.8-3.10
- **Activate**: `conda activate state_space_practice`

## Common Commands

```bash
# Run all tests
pytest src/state_space_practice/tests/

# Run specific test file
pytest src/state_space_practice/tests/test_kalman.py -v

# Run tests with coverage
pytest --cov=src/state_space_practice --cov-report=term-missing

# Run linter
ruff check src/

# Run type checker (if configured)
mypy src/state_space_practice/
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
