"""Shared utilities for integration / recovery tests.

Centralises simulation helpers and assertion functions so that
model-specific test files stay DRY.

Note: callers must ensure ``jax.config.update("jax_enable_x64", True)``
is set before importing this module (see conftest.py).
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from state_space_practice.nonlinear_dynamics import apply_mlp, leapfrog_step


# ---------------------------------------------------------------------------
# State segmentation
# ---------------------------------------------------------------------------


def state_segmentation_accuracy(
    true_states: np.ndarray,
    smoother_discrete_state_prob: np.ndarray,
) -> float:
    """Best-permutation state segmentation accuracy.

    Parameters
    ----------
    true_states : shape (n_time,)
        Ground truth discrete state labels (integers).
    smoother_discrete_state_prob : shape (n_time, n_discrete_states)
        Posterior state probabilities from the model.

    Returns
    -------
    accuracy : float

    Notes
    -----
    Scales as O(n_states!).  Fine for 2–3 states; for larger numbers
    consider ``scipy.optimize.linear_sum_assignment`` instead.
    """
    inferred = np.array(jnp.argmax(smoother_discrete_state_prob, axis=1))
    true = np.array(true_states)
    n_states = smoother_discrete_state_prob.shape[1]

    best_acc = 0.0
    for perm in permutations(range(n_states)):
        remapped = np.array([perm[s] for s in inferred])
        acc = float(np.mean(remapped == true))
        best_acc = max(best_acc, acc)

    return best_acc


# ---------------------------------------------------------------------------
# Log-likelihood assertions
# ---------------------------------------------------------------------------


def assert_ll_improves(lls: Sequence[float], label: str = "") -> None:
    """Assert that the final LL is higher than the first."""
    assert len(lls) >= 2, f"Need at least 2 LL values, got {len(lls)}"
    prefix = f"[{label}] " if label else ""
    assert lls[-1] > lls[0], (
        f"{prefix}LL did not improve: first={lls[0]:.4f}, last={lls[-1]:.4f}"
    )


def assert_ll_monotonic(lls: Sequence[float], tol: float = 1e-3,
                         label: str = "") -> None:
    """Assert that LL is non-decreasing (within tolerance) at every step."""
    assert len(lls) >= 2, f"Need at least 2 LL values, got {len(lls)}"
    prefix = f"[{label}] " if label else ""
    for i in range(1, len(lls)):
        assert lls[i] >= lls[i - 1] - tol, (
            f"{prefix}LL decreased at step {i}: "
            f"{lls[i - 1]:.6f} -> {lls[i]:.6f}"
        )


# ---------------------------------------------------------------------------
# Smoother vs prior
# ---------------------------------------------------------------------------


def assert_smoother_beats_prior(
    smoother_estimate: ArrayLike,
    true_trajectory: ArrayLike,
    prior_estimate: ArrayLike,
) -> None:
    """Assert that smoother MSE is lower than prior MSE (scalar comparison)."""
    smoother_mse = float(jnp.mean((smoother_estimate - true_trajectory) ** 2))
    prior_mse = float(jnp.mean((prior_estimate - true_trajectory) ** 2))
    assert smoother_mse < prior_mse, (
        f"Smoother MSE ({smoother_mse:.4f}) should be less than "
        f"prior MSE ({prior_mse:.4f})"
    )


# ---------------------------------------------------------------------------
# Harmonic oscillator simulation helpers
# ---------------------------------------------------------------------------


def simulate_harmonic_oscillator(
    omega: float,
    n_time: int,
    dt: float,
    process_noise_std: float = 1e-3,
    x0: Array | None = None,
    key: Array | None = None,
    hidden_dims: list[int] | None = None,
) -> tuple[Array, dict]:
    """Simulate a pure harmonic oscillator via leapfrog integration.

    Returns zeroed MLP weights so the dynamics are purely Hamiltonian
    (no learned potential).

    Parameters
    ----------
    omega : Angular frequency.
    n_time : Number of time steps.
    dt : Time step.
    process_noise_std : Std-dev of additive Gaussian process noise.
    x0 : Initial state [q, p].  Defaults to [1, 0].
    key : Defaults to PRNGKey(42).
    hidden_dims : MLP hidden layer sizes. Defaults to [8].

    Returns
    -------
    x_true : shape (n_time, 2)
    mlp_params : Zeroed MLP parameters matching the architecture.
    """
    if x0 is None:
        x0 = jnp.array([1.0, 0.0])
    if key is None:
        key = jax.random.PRNGKey(42)
    if hidden_dims is None:
        hidden_dims = [8]

    # Build MLP param structure from a throwaway model, then zero everything.
    # Deferred import: avoids circular dependency at module level.
    from state_space_practice.hamiltonian_lfp import HamiltonianLFPModel

    _tmp = HamiltonianLFPModel(
        n_sources=2, n_oscillators=1, hidden_dims=hidden_dims, seed=0,
        sampling_freq=1.0 / dt,
    )
    mlp_params = jax.tree_util.tree_map(jnp.zeros_like, dict(_tmp.mlp_params))

    trans_params = {**mlp_params, "omega": omega}

    def sim_step(x, key_i):
        x_next = leapfrog_step(x, trans_params, apply_mlp, dt)
        x_next = x_next + jax.random.normal(key_i, x.shape) * process_noise_std
        return x_next, x_next

    keys = jax.random.split(key, n_time)
    _, x_true = jax.lax.scan(sim_step, x0, keys)

    return x_true, mlp_params


def simulate_lfp_observations(
    x_true: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    noise_std: float,
    key: Array | None = None,
) -> Array:
    """Generate noisy LFP observations from a latent trajectory.

    Parameters
    ----------
    x_true : shape (n_time, n_latent)
    C : shape (n_sources, n_latent)
    d : shape (n_sources,)
    noise_std : observation noise standard deviation
    key : Defaults to PRNGKey(99).

    Returns
    -------
    lfp : shape (n_time, n_sources)
    """
    if key is None:
        key = jax.random.PRNGKey(99)
    n_time = x_true.shape[0]
    n_sources = C.shape[0]
    noise = jax.random.normal(key, (n_time, n_sources)) * noise_std
    return x_true @ C.T + d + noise


def simulate_poisson_spikes(
    x_true: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    dt: float,
    key: Array | None = None,
) -> Array:
    """Generate Poisson spike counts from a latent trajectory.

    Parameters
    ----------
    x_true : shape (n_time, n_latent)
    C : shape (n_sources, n_latent)
    d : shape (n_sources,)
    dt : time bin width (seconds)
    key : Defaults to PRNGKey(77).

    Returns
    -------
    spikes : shape (n_time, n_sources)
    """
    if key is None:
        key = jax.random.PRNGKey(77)
    log_rates = x_true @ C.T + d
    rates = jnp.exp(jnp.clip(log_rates, -5, 3)) * dt
    return jax.random.poisson(key, rates)
