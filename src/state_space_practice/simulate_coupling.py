"""Simulate a Bernoulli-logistic spike-field coupling dataset with ground truth.

Generates a latent oscillator trajectory and Bernoulli spikes coupled to it, and
returns the ground-truth coupling so an estimator can be scored against it. See
:mod:`coupling_model` for the model definition. Requires float64 (the test suite
enables ``jax_enable_x64``).
"""

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

from state_space_practice.coupling_model import (
    CouplingModelParams,
    SimulatedCoupling,
    build_transition,
    validate_coupling_params,
)


def simulate_coupling(
    params: CouplingModelParams, n_time: int, seed: int
) -> SimulatedCoupling:
    """Simulate latent dynamics and coupled Bernoulli spikes.

    The latent starts from the oscillator stationary distribution and evolves as
    ``x_k = A x_{k-1} + w_k``; spikes are drawn ``y_{s,k} ~ Bernoulli(sigmoid(
    eta_{s,k}))`` with ``eta`` the model logit. The coupling mask is derived from
    the parameters (``True`` where either coupling component is nonzero, i.e.
    ``beta_real**2 + beta_imag**2 > 0``).

    Parameters
    ----------
    params : CouplingModelParams
        Fully specified model parameters. ``history_kernel`` must be ``None``.
    n_time : int
        Number of time bins to simulate.
    seed : int
        PRNG seed; the simulation is deterministic given ``(params, n_time, seed)``.

    Returns
    -------
    SimulatedCoupling
    """
    validate_coupling_params(params)
    if n_time <= 0:
        raise ValueError(f"n_time must be positive, got {n_time}")
    if params.history_kernel is not None:
        raise NotImplementedError(
            "history_kernel is not yet supported; pass history_kernel=None"
        )

    transition_matrix, process_covariance = build_transition(params)
    n_latent = transition_matrix.shape[0]

    init_key, noise_key, spike_key = jax.random.split(jax.random.PRNGKey(seed), 3)

    # Start from the oscillator stationary distribution so early bins are not a
    # warm-up transient: each component has variance var / (1 - decay**2).
    stationary_var = jnp.asarray(params.process_noise_var) / (
        1.0 - jnp.asarray(params.osc_decay) ** 2
    )
    x0 = jax.random.normal(init_key, (n_latent,)) * jnp.sqrt(
        jnp.repeat(stationary_var, 2)
    )

    # Q is diagonal, so its Cholesky factor is the elementwise square root.
    noise = jax.random.normal(noise_key, (n_time, n_latent)) * jnp.sqrt(
        jnp.diag(process_covariance)
    )

    def step(x, w):
        x_next = transition_matrix @ x + w
        return x_next, x_next

    _, latent = jax.lax.scan(step, x0, noise)  # (n_time, n_latent), x_1 .. x_T

    re = latent[:, 0::2]  # (T, J)
    im = latent[:, 1::2]  # (T, J)
    eta = params.baseline[None, :] + re @ params.beta_real.T + im @ params.beta_imag.T
    spikes = jax.random.bernoulli(spike_key, sigmoid(eta)).astype(jnp.float64)

    coupling_mask = (params.beta_real**2 + params.beta_imag**2) > 0

    return SimulatedCoupling(
        spikes=spikes,
        latent_true=latent,
        beta_real_true=params.beta_real,
        beta_imag_true=params.beta_imag,
        coupling_mask=coupling_mask,
        baseline_true=params.baseline,
        params=params,
        seed=seed,
    )
