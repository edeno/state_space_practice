"""Shared utilities for integration / recovery tests.

Centralises simulation helpers and assertion functions so that
model-specific test files stay DRY.

Note: callers must ensure ``jax.config.update("jax_enable_x64", True)``
is set before importing this module (see conftest.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

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


def assert_ll_monotonic(
    lls: Sequence[float], tol: float = 1e-3, label: str = ""
) -> None:
    """Assert that LL is non-decreasing (within tolerance) at every step."""
    assert len(lls) >= 2, f"Need at least 2 LL values, got {len(lls)}"
    prefix = f"[{label}] " if label else ""
    for i in range(1, len(lls)):
        assert lls[i] >= lls[i - 1] - tol, (
            f"{prefix}LL decreased at step {i}: {lls[i - 1]:.6f} -> {lls[i]:.6f}"
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
        n_sources=2,
        n_oscillators=1,
        hidden_dims=hidden_dims,
        seed=0,
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


# ---------------------------------------------------------------------------
# Switching oscillator models (COM / CNM / DIM) at known parameters
# ---------------------------------------------------------------------------

OSCILLATOR_KINDS = ("COM", "CNM", "DIM")
_OSC_FS = 100.0
_OSC_FREQS = (6.0, 11.0)
_OSC_DAMPING = (0.95, 0.9)
_OSC_MEASUREMENT_VARIANCE = 0.2
_OSC_SWITCHING_Z = ((0.97, 0.03), (0.05, 0.95))


def oscillator_model_at_truth(
    kind: str, n_discrete_states: int = 1, switching: bool = False, **kwargs
):
    """A two-oscillator COM / CNM / DIM model whose parameters *are* the truth.

    The model's own ``A, H, Q, R`` stacks (after ``_initialize_parameters``)
    define the generative process, so :func:`simulate_from_oscillator_model`
    samples exactly the model the E-step assumes.

    Parameters
    ----------
    kind : {"COM", "CNM", "DIM"}
    n_discrete_states : int
        1 or 2.
    switching : bool
        With two states, give them *distinct* state-dependent parameters (H for
        COM, Q for CNM, A for DIM).  Otherwise both states share the
        state-dependent parameter, so the model is a single regime written with
        two labels.
    **kwargs
        Forwarded to the model constructor (e.g. ``use_reparameterized_mstep``).
    """
    from state_space_practice.oscillator_models import (
        CommonOscillatorModel,
        CorrelatedNoiseModel,
        DirectedInfluenceModel,
    )

    S = n_discrete_states
    common = {
        "n_oscillators": 2,
        "n_discrete_states": S,
        "sampling_freq": _OSC_FS,
        "freqs": jnp.array(_OSC_FREQS),
        "damping_coef": jnp.array(_OSC_DAMPING),
        "measurement_variance": _OSC_MEASUREMENT_VARIANCE,
    }
    coupling = np.zeros((2, 2, S))
    phase = np.zeros((2, 2, S))
    if kind == "COM":
        model = CommonOscillatorModel(
            n_sources=3, process_variance=jnp.array([0.5, 0.3]), **common, **kwargs
        )
        model._initialize_parameters(jax.random.PRNGKey(0))
        rng = np.random.default_rng(1)
        H_first = rng.normal(size=(3, 4))
        H_second = rng.normal(size=(3, 4)) if switching else H_first
        model.measurement_matrix = jnp.stack([H_first, H_second][:S], axis=-1)
    elif kind == "CNM":
        coupling[0, 1, 0], phase[0, 1, 0] = 0.4, 0.5
        if S > 1 and not switching:
            coupling[0, 1, 1], phase[0, 1, 1] = 0.4, 0.5
        model = CorrelatedNoiseModel(
            process_variance=jnp.full((2, S), 0.5),
            phase_difference=jnp.asarray(phase),
            coupling_strength=jnp.asarray(coupling),
            **common,
            **kwargs,
        )
        model._initialize_parameters(jax.random.PRNGKey(0))
    elif kind == "DIM":
        coupling[1, 0, 0], phase[1, 0, 0] = 0.3, 0.4
        if S > 1:
            if switching:
                coupling[0, 1, 1], phase[0, 1, 1] = 0.3, -0.4
            else:
                coupling[1, 0, 1], phase[1, 0, 1] = 0.3, 0.4
        model = DirectedInfluenceModel(
            process_variance=jnp.array([0.5, 0.3]),
            phase_difference=jnp.asarray(phase),
            coupling_strength=jnp.asarray(coupling),
            **common,
            **kwargs,
        )
        model._initialize_parameters(jax.random.PRNGKey(0))
    else:
        raise ValueError(f"unknown oscillator model kind {kind!r}")

    n_latent = model.n_cont_states
    model.init_mean = jnp.zeros((n_latent, S))
    model.init_cov = jnp.stack([2.0 * jnp.eye(n_latent)] * S, axis=-1)
    if S == 2:
        model.discrete_transition_matrix = jnp.array(_OSC_SWITCHING_Z)
        model.init_discrete_state_prob = jnp.array([0.5, 0.5])
    return model


def simulate_from_oscillator_model(
    model, n_time: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``(x, y, s)`` from a switching LGSSM model (x_1 convention).

    ``s_1 ~ init_discrete_state_prob``, ``x_1 ~ N(init_mean_s1, init_cov_s1)``
    and ``y_t = H_{s_t} x_t + v_t`` for every ``t`` -- the convention of
    ``switching_kalman_filter``.  The number of random draws does not depend
    on the number of discrete states, so a one-state model and its two-label
    copy see identical ``(x, y)`` for the same generator state.

    Returns
    -------
    x : (n_time, n_latent), y : (n_time, n_obs), s : (n_time,)
    """
    A = np.asarray(model.continuous_transition_matrix)
    Q = np.asarray(model.process_cov)
    H = np.asarray(model.measurement_matrix)
    R = np.asarray(model.measurement_cov)
    Z = np.asarray(model.discrete_transition_matrix)
    pi0 = np.asarray(model.init_discrete_state_prob)
    m0 = np.asarray(model.init_mean)
    P0 = np.asarray(model.init_cov)
    n_states = A.shape[-1]
    n_latent, n_obs = A.shape[0], H.shape[0]

    s = rng.choice(n_states, p=pi0)
    x = rng.multivariate_normal(m0[:, s], P0[:, :, s])
    xs, ys, ss = [], [], []
    for t in range(n_time):
        if t > 0:
            s = rng.choice(n_states, p=Z[s])
            x = A[:, :, s] @ x + rng.multivariate_normal(np.zeros(n_latent), Q[:, :, s])
        y = H[:, :, s] @ x + rng.multivariate_normal(np.zeros(n_obs), R[:, :, s])
        xs.append(x)
        ys.append(y)
        ss.append(s)
    return np.array(xs), np.array(ys), np.array(ss)


def collapse_switching_posterior(
    discrete_prob: ArrayLike, state_cond_mean: ArrayLike, state_cond_cov: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match the state-conditional posteriors into one Gaussian per t.

    Parameters
    ----------
    discrete_prob : (n_time, n_states)
    state_cond_mean : (n_time, n_latent, n_states)
    state_cond_cov : (n_time, n_latent, n_latent, n_states)

    Returns
    -------
    mean : (n_time, n_latent), cov : (n_time, n_latent, n_latent)
    """
    p = np.asarray(discrete_prob)
    mu = np.asarray(state_cond_mean)
    P = np.asarray(state_cond_cov)
    mean = np.einsum("tj,tkj->tk", p, mu)
    dev = mu - mean[:, :, None]
    cov = np.einsum("tj,tklj->tkl", p, P) + np.einsum("tj,tkj,tlj->tkl", p, dev, dev)
    return mean, cov


def standardized_errors(
    truth: ArrayLike, mean: ArrayLike, cov: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Per-coordinate z-scores and per-time Mahalanobis distances.

    Returns
    -------
    z : (n_time, n_latent)
        ``(x - m) / sqrt(diag P)``; ~N(0, 1) under a calibrated posterior.
    mahalanobis : (n_time,)
        ``(x - m)^T P^{-1} (x - m)``; ~chi^2(n_latent) under calibration.
    """
    err = np.asarray(truth) - np.asarray(mean)
    cov = np.asarray(cov)
    z = err / np.sqrt(np.einsum("tkk->tk", cov))
    mahalanobis = np.einsum("tk,tkl,tl->t", err, np.linalg.inv(cov), err)
    return z, mahalanobis


# ---------------------------------------------------------------------------
# Expected complete-data objective of the switching LGSSM M-step
# ---------------------------------------------------------------------------
#
# Written out from the definitions, independently of the library's M-step, so
# tests can check that an installed M-step output is a stationary point (in
# the model's constrained parametrisation) and that it does not decrease the
# objective.  The posterior moments are the E-step's; with the GPB2 smoother
# the pair-conditional moments are available and every statistic below is the
# exact expectation under that (approximate) posterior.


def switching_posterior_stats(model) -> dict:
    """Posterior sufficient statistics of a fitted switching LGSSM model.

    Requires the GPB2 smoother outputs (``smoother_pair_cond_covs`` and
    ``smoother_next_pair_cond_means``).

    Returns
    -------
    dict with
        ``w`` (T, S)             P(S_t = j | y)
        ``xi`` (T-1, S, S)       P(S_t = i, S_{t+1} = j | y)
        ``mean`` (T, n, S), ``cov`` (T, n, n, S)  state-conditional moments
        ``gamma1`` (n, n, S)     sum_{t,i} xi_t(i,j) E[x_t x_t^T | i, j]
        ``beta`` (n, n, S)       sum_{t,i} xi_t(i,j) E[x_{t+1} x_t^T | i, j]
        ``gamma2`` (n, n, S)     sum_{t>=2} w_tj E[x_t x_t^T | j]
        ``n_trans`` (S,)         sum_{t>=2} w_tj
    """
    if model.smoother_pair_cond_covs is None:
        raise ValueError("switching_posterior_stats needs the GPB2 smoother outputs")
    w = np.asarray(model.smoother_discrete_state_prob)
    xi = np.asarray(model.smoother_joint_discrete_state_prob)
    mean = np.asarray(model.smoother_state_cond_mean)
    cov = np.asarray(model.smoother_state_cond_cov)
    pm = np.asarray(model.smoother_pair_cond_means)  # E[x_t | i, j]
    pc = np.asarray(model.smoother_pair_cond_covs)  # Cov[x_t | i, j]
    nm = np.asarray(model.smoother_next_pair_cond_means)  # E[x_{t+1} | i, j]
    cc = np.asarray(model.smoother_pair_cond_cross_cov)  # Cov[x_t, x_{t+1} | i, j]
    second = pc + np.einsum("taij,tbij->tabij", pm, pm)
    cross = np.swapaxes(cc, 1, 2) + np.einsum("taij,tbij->tabij", nm, pm)
    gamma1 = np.einsum("tij,tabij->abj", xi, second)
    beta = np.einsum("tij,tabij->abj", xi, cross)
    gamma2 = np.einsum("tj,tabj->abj", w[1:], cov[1:]) + np.einsum(
        "tj,taj,tbj->abj", w[1:], mean[1:], mean[1:]
    )
    return {
        "w": w,
        "xi": xi,
        "mean": mean,
        "cov": cov,
        "gamma1": gamma1,
        "beta": beta,
        "gamma2": gamma2,
        "n_trans": w[1:].sum(axis=0),
    }


def _logdet(m):
    return jnp.linalg.slogdet(m)[1]


def transition_objective(A, Q, stats: dict):
    """sum_j E[log N(x_{t+1}; A_j x_t, Q_j)] over transitions into state j (+const)."""
    total = 0.0
    for j in range(A.shape[-1]):
        Aj, Qj = A[..., j], Q[..., j]
        G1, B, G2 = (
            stats["gamma1"][..., j],
            stats["beta"][..., j],
            stats["gamma2"][..., j],
        )
        scatter = G2 - Aj @ B.T - B @ Aj.T + Aj @ G1 @ Aj.T
        total += -0.5 * (
            stats["n_trans"][j] * _logdet(Qj) + jnp.trace(jnp.linalg.solve(Qj, scatter))
        )
    return total


def observation_objective(H, R, obs, stats: dict):
    """sum_{t,j} w_tj E[log N(y_t; H_j x_t, R_j) | S_t = j] (+const)."""
    obs = jnp.asarray(obs)
    total = 0.0
    for j in range(H.shape[-1]):
        Hj, Rj = H[..., j], R[..., j]
        w = stats["w"][:, j]
        resid = obs - stats["mean"][:, :, j] @ Hj.T
        scatter = (
            jnp.einsum("t,ta,tb->ab", w, resid, resid)
            + Hj @ jnp.einsum("t,tab->ab", w, stats["cov"][..., j]) @ Hj.T
        )
        total += -0.5 * (
            w.sum() * _logdet(Rj) + jnp.trace(jnp.linalg.solve(Rj, scatter))
        )
    return total


def initial_state_objective(m0, P0, stats: dict):
    """sum_j w_1j E[log N(x_1; m0_j, P0_j) | S_1 = j] (x_1 convention, +const)."""
    total = 0.0
    for j in range(m0.shape[-1]):
        w1 = stats["w"][0, j]
        d = stats["mean"][0, :, j] - m0[:, j]
        scatter = stats["cov"][0, :, :, j] + jnp.outer(d, d)
        total += (
            -0.5
            * w1
            * (_logdet(P0[..., j]) + jnp.trace(jnp.linalg.solve(P0[..., j], scatter)))
        )
    return total


def discrete_objective(Z, pi0, stats: dict):
    """sum_t sum_ij xi_t(i,j) log Z_ij + sum_j w_1j log pi_j."""
    counts = stats["xi"].sum(axis=0)
    return jnp.sum(counts * jnp.log(Z)) + jnp.sum(stats["w"][0] * jnp.log(pi0))


def central_difference_gradient(f, theta: np.ndarray, step: float = 1e-5) -> np.ndarray:
    """Central finite-difference gradient of scalar ``f`` at flat ``theta``."""
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for k in range(theta.size):
        e = np.zeros_like(theta)
        e[k] = step
        grad[k] = (float(f(theta + e)) - float(f(theta - e))) / (2 * step)
    return grad


def symmetric_basis(n: int) -> list[np.ndarray]:
    """Basis of the symmetric n x n matrices (for directional derivatives)."""
    basis = []
    for a in range(n):
        for b in range(a, n):
            e = np.zeros((n, n))
            e[a, b] = e[b, a] = 1.0
            basis.append(e)
    return basis


def dim_projected_mstep_parameters(stats: dict, model) -> dict:
    """What the bare standard DIM M-step installs for these statistics.

    Unconstrained per-state ``A* = Beta Gamma1^{-1}``, projected onto the
    oscillator family (``project_transition_matrix_stack``), with the shared
    frequency/damping averaged across states -- i.e. the standard M-step
    *without* the generalized-EM safeguard.  Returns the public parameters in
    the joint optimizer's layout plus the rebuilt ``transition_matrix``.
    """
    from state_space_practice.oscillator_utils import (
        construct_stable_directed_influence_transition_stack,
        extract_dim_params_from_matrix_stack,
        project_transition_matrix_stack,
    )

    n_states = stats["gamma1"].shape[-1]
    a_star = jnp.stack(
        [
            jnp.linalg.solve(stats["gamma1"][..., j].T, stats["beta"][..., j].T).T
            for j in range(n_states)
        ],
        axis=-1,
    )
    projected = project_transition_matrix_stack(a_star, model.max_spectral_radius)
    params = extract_dim_params_from_matrix_stack(
        projected, model.sampling_freq, model.n_oscillators
    )
    params["transition_matrix"] = construct_stable_directed_influence_transition_stack(
        params["freq"],
        params["damping"],
        params["coupling_strength"],
        params["phase_diff"],
        model.sampling_freq,
        max_spectral_radius=model.max_spectral_radius,
    )
    return params


def set_dim_public_params(model, params: dict) -> None:
    """Install joint-optimizer-layout DIM parameters and rebuild ``A``."""
    model.freqs = jnp.asarray(params["freq"])
    model.damping_coef = jnp.asarray(params["damping"])
    model.coupling_strength = jnp.asarray(params["coupling_strength"])
    model.phase_difference = jnp.asarray(params["phase_diff"])
    model._rebuild_stable_transition_matrix()
