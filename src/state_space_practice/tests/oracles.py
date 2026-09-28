"""Brute-force reference implementations ("oracles") for Gaussian inference.

Everything here is plain NumPy (float64) and deliberately shares no code with
the library: posteriors come from *dense* Gaussian conditioning of the joint
distribution of all latent states and observations, and switching posteriors
come from enumerating every discrete path. They are exponentially or cubically
expensive and only meant for tiny problems (``T <= 8``, a handful of dims,
``n_discrete_states ** T`` paths), which is exactly what makes them trustworthy
references for the recursive filters and smoothers.

Model conventions
-----------------
Linear-Gaussian state-space model (the convention of
:func:`state_space_practice.kalman.kalman_filter`, which predicts before its
first update)::

    x_0 ~ N(m_0, P_0)
    x_t = A_t x_{t-1} + w_t,   w_t ~ N(0, Q_t),   t = 1..T
    y_t = H_t x_t + v_t,       v_t ~ N(0, R_t),   t = 1..T

With ``prior_on_first_state=True`` the prior is instead placed directly on
``x_1`` (the convention of
:func:`state_space_practice.switching_kalman.switching_kalman_filter`, which
applies only a measurement update at ``t = 1``); internally this is the same
model with ``A_1 = I`` and ``Q_1 = 0``.

Switching model (the convention of ``switching_kalman_filter``)::

    S_1 ~ pi,              S_t | S_{t-1} = i ~ Z[i, :]
    x_1 | S_1 = j ~ N(m_j, P_j)
    x_t = A_{S_t} x_{t-1} + w_t,   w_t ~ N(0, Q_{S_t}),   t = 2..T
    y_t = H_{S_t} x_t + v_t,       v_t ~ N(0, R_{S_t}),   t = 1..T
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


def _as_sequence(param: np.ndarray, n_time: int, ndim: int) -> np.ndarray:
    """Broadcast a constant parameter of rank ``ndim`` to a (n_time, ...) stack."""
    param = np.asarray(param, dtype=np.float64)
    if param.ndim == ndim:
        return np.broadcast_to(param, (n_time, *param.shape)).copy()
    if param.ndim == ndim + 1 and param.shape[0] == n_time:
        return param.copy()
    raise ValueError(f"expected rank {ndim} or ({n_time}, ...), got {param.shape}")


def gaussian_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """log N(x; mean, cov) via a Cholesky factorisation."""
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, x - mean)
    return float(
        -0.5 * (x.size * _LOG_2PI + 2.0 * np.sum(np.log(np.diag(chol))) + z @ z)
    )


def _condition(
    mean_x: np.ndarray,
    cov_x: np.ndarray,
    mean_y: np.ndarray,
    cov_y: np.ndarray,
    cov_xy: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Moments of x | y for jointly Gaussian (x, y)."""
    chol = np.linalg.cholesky(cov_y)
    gain_t = np.linalg.solve(chol.T, np.linalg.solve(chol, cov_xy.T))  # S^-1 C^T
    mean = mean_x + gain_t.T @ (y - mean_y)
    cov = cov_x - cov_xy @ gain_t
    return mean, 0.5 * (cov + cov.T)


@dataclass
class DenseGaussianPosterior:
    """Exact moments of a linear-Gaussian state-space model given ``y_{1:T}``.

    Time indices in the arrays are 0-based for ``x_1 .. x_T`` (entry ``t``
    is ``x_{t+1}``); ``x_0`` has its own fields.

    Attributes
    ----------
    filtered_mean : (T, n)
        E[x_t | y_{1:t}].
    filtered_cov : (T, n, n)
        Cov[x_t | y_{1:t}].
    smoothed_mean : (T, n)
        E[x_t | y_{1:T}].
    smoothed_cov : (T, n, n)
        Cov[x_t | y_{1:T}].
    smoothed_cross_cov : (T - 1, n, n)
        Cov[x_t, x_{t+1} | y_{1:T}] (``x_t`` first, the smoother's convention).
    init_smoothed_mean : (n,)
        E[x_0 | y_{1:T}] (equals ``smoothed_mean[0]`` when the prior is on x_1).
    init_smoothed_cov : (n, n)
        Cov[x_0 | y_{1:T}].
    init_cross_cov : (n, n)
        Cov[x_0, x_1 | y_{1:T}].
    joint_mean : ((T + 1) n,)
        E[x_{0:T} | y_{1:T}], blocks ordered x_0, x_1, ..., x_T.
    joint_cov : ((T + 1) n, (T + 1) n)
        Cov[x_{0:T} | y_{1:T}].
    filtered_log_likelihood : (T,)
        log p(y_{1:t}) for t = 1..T (cumulative).
    log_likelihood : float
        log p(y_{1:T}).
    """

    filtered_mean: np.ndarray
    filtered_cov: np.ndarray
    smoothed_mean: np.ndarray
    smoothed_cov: np.ndarray
    smoothed_cross_cov: np.ndarray
    init_smoothed_mean: np.ndarray
    init_smoothed_cov: np.ndarray
    init_cross_cov: np.ndarray
    joint_mean: np.ndarray
    joint_cov: np.ndarray
    filtered_log_likelihood: np.ndarray
    log_likelihood: float


def lgssm_joint_prior(
    init_mean: np.ndarray,
    init_cov: np.ndarray,
    transition_matrix: np.ndarray,
    process_cov: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_cov: np.ndarray,
    n_time: int,
    prior_on_first_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prior joint moments of (x_{0:T}, y_{1:T}), built explicitly.

    ``x - E[x] = G e`` with independent ``e = (x_0 - m_0, w_1, ..., w_T)``,
    so ``Cov[x] = G diag(P_0, Q_1, ..., Q_T) G^T``; then ``y = H x + v``.

    Parameters may be constant (``(n, n)`` etc.) or per-time stacks with a
    leading axis of length ``T`` (entry ``t - 1`` is used at time ``t``).

    Returns
    -------
    mean_x : ((T + 1) n,)
    cov_x : ((T + 1) n, (T + 1) n)
    mean_y : (T m,)
    cov_y : (T m, T m)
    cov_xy : ((T + 1) n, T m)
    """
    init_mean = np.asarray(init_mean, dtype=np.float64)
    init_cov = np.asarray(init_cov, dtype=np.float64)
    n = init_mean.shape[0]
    A = _as_sequence(transition_matrix, n_time, 2)
    Q = _as_sequence(process_cov, n_time, 2)
    H = _as_sequence(measurement_matrix, n_time, 2)
    R = _as_sequence(measurement_cov, n_time, 2)
    m = H.shape[1]
    if prior_on_first_state:
        A[0] = np.eye(n)
        Q[0] = np.zeros((n, n))

    n_x = (n_time + 1) * n
    # G[t-block, s-block] = A_t A_{t-1} ... A_{s+1}  (identity when s == t)
    G = np.zeros((n_x, n_x))
    for s in range(n_time + 1):
        block = np.eye(n)
        G[s * n : (s + 1) * n, s * n : (s + 1) * n] = block
        for t in range(s + 1, n_time + 1):
            block = A[t - 1] @ block
            G[t * n : (t + 1) * n, s * n : (s + 1) * n] = block
    noise_cov = np.zeros((n_x, n_x))
    noise_cov[:n, :n] = init_cov
    for t in range(1, n_time + 1):
        noise_cov[t * n : (t + 1) * n, t * n : (t + 1) * n] = Q[t - 1]
    mean_x = np.zeros(n_x)
    mean_x[:n] = init_mean
    for t in range(1, n_time + 1):
        mean_x[t * n : (t + 1) * n] = A[t - 1] @ mean_x[(t - 1) * n : t * n]
    cov_x = G @ noise_cov @ G.T
    cov_x = 0.5 * (cov_x + cov_x.T)

    H_big = np.zeros((n_time * m, n_x))
    R_big = np.zeros((n_time * m, n_time * m))
    for t in range(1, n_time + 1):
        H_big[(t - 1) * m : t * m, t * n : (t + 1) * n] = H[t - 1]
        R_big[(t - 1) * m : t * m, (t - 1) * m : t * m] = R[t - 1]
    mean_y = H_big @ mean_x
    cov_xy = cov_x @ H_big.T
    cov_y = H_big @ cov_xy + R_big
    return mean_x, cov_x, mean_y, 0.5 * (cov_y + cov_y.T), cov_xy


def lgssm_dense_posterior(
    init_mean: np.ndarray,
    init_cov: np.ndarray,
    obs: np.ndarray,
    transition_matrix: np.ndarray,
    process_cov: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_cov: np.ndarray,
    prior_on_first_state: bool = False,
) -> DenseGaussianPosterior:
    """Exact filtering/smoothing moments by dense Gaussian conditioning.

    See the module docstring for the model. ``obs`` has shape ``(T, m)``.
    Filtered moments condition on the prefix ``y_{1:t}`` of the same joint.
    """
    obs = np.asarray(obs, dtype=np.float64)
    n_time, m = obs.shape
    n = np.asarray(init_mean).shape[0]
    mean_x, cov_x, mean_y, cov_y, cov_xy = lgssm_joint_prior(
        init_mean,
        init_cov,
        transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
        n_time,
        prior_on_first_state=prior_on_first_state,
    )
    y = obs.reshape(-1)

    joint_mean, joint_cov = _condition(mean_x, cov_x, mean_y, cov_y, cov_xy, y)

    def block(t: int) -> slice:  # x_t, t = 0..T
        return slice(t * n, (t + 1) * n)

    smoothed_mean = np.stack([joint_mean[block(t)] for t in range(1, n_time + 1)])
    smoothed_cov = np.stack(
        [joint_cov[block(t), block(t)] for t in range(1, n_time + 1)]
    )
    smoothed_cross_cov = np.zeros((n_time - 1, n, n))
    for t in range(1, n_time):
        smoothed_cross_cov[t - 1] = joint_cov[block(t), block(t + 1)]

    filtered_mean = np.zeros((n_time, n))
    filtered_cov = np.zeros((n_time, n, n))
    filtered_ll = np.zeros(n_time)
    for t in range(1, n_time + 1):
        k = t * m
        sl = block(t)
        fm, fc = _condition(
            mean_x[sl],
            cov_x[sl, sl],
            mean_y[:k],
            cov_y[:k, :k],
            cov_xy[sl, :k],
            y[:k],
        )
        filtered_mean[t - 1] = fm
        filtered_cov[t - 1] = fc
        filtered_ll[t - 1] = gaussian_logpdf(y[:k], mean_y[:k], cov_y[:k, :k])

    return DenseGaussianPosterior(
        filtered_mean=filtered_mean,
        filtered_cov=filtered_cov,
        smoothed_mean=smoothed_mean,
        smoothed_cov=smoothed_cov,
        smoothed_cross_cov=smoothed_cross_cov,
        init_smoothed_mean=joint_mean[block(0)],
        init_smoothed_cov=joint_cov[block(0), block(0)],
        init_cross_cov=joint_cov[block(0), block(1)],
        joint_mean=joint_mean,
        joint_cov=joint_cov,
        filtered_log_likelihood=filtered_ll,
        log_likelihood=float(filtered_ll[-1]),
    )


def _expected_gaussian_log_density(
    second_moment_residual: np.ndarray, cov: np.ndarray, weight: float = 1.0
) -> float:
    """weight * E[log N(r; 0, cov)] given E[r r^T] = second_moment_residual."""
    dim = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    return float(
        -0.5
        * weight
        * (
            dim * _LOG_2PI
            + logdet
            + np.trace(np.linalg.solve(cov, second_moment_residual))
        )
    )


def lgssm_expected_complete_log_likelihood(
    joint_mean: np.ndarray,
    joint_cov: np.ndarray,
    obs: np.ndarray,
    init_mean: np.ndarray,
    init_cov: np.ndarray,
    transition_matrix: np.ndarray,
    process_cov: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_cov: np.ndarray,
    include_x0: bool = True,
) -> float:
    """EM auxiliary function Q(theta) = E[log p(x, y | theta)] for the LGSSM.

    The expectation is under the posterior N(joint_mean, joint_cov) of
    ``x_{0:T}`` (e.g. from :func:`lgssm_dense_posterior`), written out term by
    term with ``S_ab = Cov[x_a, x_b] + E[x_a] E[x_b]^T``::

        Q = E log N(x_0; m_0, P_0)
            + sum_{t=1}^T E log N(x_t; A x_{t-1}, Q)
            + sum_{t=1}^T E log N(y_t; H x_t, R)

    With ``include_x0=False`` the prior is placed on ``x_1`` instead and only
    the ``T - 1`` transitions within ``x_{1:T}`` are counted (the model that
    the legacy ``initial_state_prior=None`` M-step maximises).
    """
    obs = np.asarray(obs, dtype=np.float64)
    n_time = obs.shape[0]
    n = np.asarray(init_mean).shape[0]

    def blk(t: int) -> slice:
        return slice(t * n, (t + 1) * n)

    def second(a: int, b: int) -> np.ndarray:
        return joint_cov[blk(a), blk(b)] + np.outer(
            joint_mean[blk(a)], joint_mean[blk(b)]
        )

    A, Qc, H, R = (
        np.asarray(p, dtype=np.float64)
        for p in (transition_matrix, process_cov, measurement_matrix, measurement_cov)
    )
    first = 0 if include_x0 else 1
    diff0 = joint_mean[blk(first)] - init_mean
    total = _expected_gaussian_log_density(
        joint_cov[blk(first), blk(first)] + np.outer(diff0, diff0), init_cov
    )
    for t in range(first + 1, n_time + 1):
        resid = (
            second(t, t)
            - A @ second(t - 1, t)
            - second(t, t - 1) @ A.T
            + A @ second(t - 1, t - 1) @ A.T
        )
        total += _expected_gaussian_log_density(resid, Qc)
    for t in range(1, n_time + 1):
        r = obs[t - 1] - H @ joint_mean[blk(t)]
        resid = np.outer(r, r) + H @ joint_cov[blk(t), blk(t)] @ H.T
        total += _expected_gaussian_log_density(resid, R)
    return total


# ---------------------------------------------------------------------------
# Switching linear-Gaussian model: exact posterior by path enumeration
# ---------------------------------------------------------------------------


@dataclass
class SwitchingExactPosterior:
    """Exact posterior of a switching LGSSM by enumeration of discrete paths.

    Continuous quantities marginal over the discrete path are the exact
    mixture moments (moment matching of an exact Gaussian mixture, so the
    mean and covariance are exact; only higher moments are discarded).

    Attributes
    ----------
    log_likelihood : float
        log p(y_{1:T}).
    filtered_log_likelihood : (T,)
        log p(y_{1:t}).
    filtered_discrete_prob : (T, K)
        P(S_t = j | y_{1:t}).
    filtered_mean, filtered_cov : (T, n), (T, n, n)
        Moments of x_t | y_{1:t}.
    state_cond_filtered_mean, state_cond_filtered_cov : (T, n, K), (T, n, n, K)
        Moments of x_t | S_t = j, y_{1:t}.
    smoothed_discrete_prob : (T, K)
        P(S_t = j | y_{1:T}).
    smoothed_joint_discrete_prob : (T - 1, K, K)
        P(S_t = i, S_{t+1} = j | y_{1:T}).
    smoothed_mean, smoothed_cov : (T, n), (T, n, n)
        Moments of x_t | y_{1:T}.
    smoothed_cross_cov : (T - 1, n, n)
        Cov[x_t, x_{t+1} | y_{1:T}].
    state_cond_smoothed_mean, state_cond_smoothed_cov : (T, n, K), (T, n, n, K)
        Moments of x_t | S_t = j, y_{1:T}.
    pair_cond_smoothed_mean : (T - 1, n, K, K)
        E[x_t | S_t = i, S_{t+1} = j, y_{1:T}].
    pair_cond_smoothed_cov : (T - 1, n, n, K, K)
        Cov[x_t | S_t = i, S_{t+1} = j, y_{1:T}].
    next_pair_cond_smoothed_mean : (T - 1, n, K, K)
        E[x_{t+1} | S_t = i, S_{t+1} = j, y_{1:T}].
    pair_cond_smoothed_cross_cov : (T - 1, n, n, K, K)
        Cov[x_t, x_{t+1} | S_t = i, S_{t+1} = j, y_{1:T}].
    paths : (P, T) int
        Every discrete path.
    path_log_prior, path_log_likelihood : (P,)
        log p(s_{1:T}) and log p(y_{1:T} | s_{1:T}).
    path_posteriors : list[DenseGaussianPosterior]
        Per-path conditional Gaussian posterior.
    """

    log_likelihood: float
    filtered_log_likelihood: np.ndarray
    filtered_discrete_prob: np.ndarray
    filtered_mean: np.ndarray
    filtered_cov: np.ndarray
    state_cond_filtered_mean: np.ndarray
    state_cond_filtered_cov: np.ndarray
    smoothed_discrete_prob: np.ndarray
    smoothed_joint_discrete_prob: np.ndarray
    smoothed_mean: np.ndarray
    smoothed_cov: np.ndarray
    smoothed_cross_cov: np.ndarray
    state_cond_smoothed_mean: np.ndarray
    state_cond_smoothed_cov: np.ndarray
    pair_cond_smoothed_mean: np.ndarray
    pair_cond_smoothed_cov: np.ndarray
    next_pair_cond_smoothed_mean: np.ndarray
    pair_cond_smoothed_cross_cov: np.ndarray
    paths: np.ndarray
    path_log_prior: np.ndarray
    path_log_likelihood: np.ndarray
    path_posteriors: list

    @property
    def path_posterior_prob(self) -> np.ndarray:
        """P(s_{1:T} | y_{1:T}) for every path, shape (P,)."""
        log_w = self.path_log_prior + self.path_log_likelihood
        return np.exp(log_w - _logsumexp(log_w))


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return np.squeeze(out, axis=axis) if axis is not None else float(out.squeeze())


def _mixture_moments(
    weights: np.ndarray, means: np.ndarray, covs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of sum_k w_k N(means[k], covs[k]) (weights sum to 1)."""
    mean = np.einsum("k,kn->n", weights, means)
    diff = means - mean
    cov = np.einsum("k,kab->ab", weights, covs) + np.einsum(
        "k,ka,kb->ab", weights, diff, diff
    )
    return mean, 0.5 * (cov + cov.T)


def _mixture_cross_moments(
    weights: np.ndarray,
    means_a: np.ndarray,
    means_b: np.ndarray,
    cross: np.ndarray,
) -> np.ndarray:
    """Cov[a, b] of a mixture whose components have Cov[a, b] = cross[k]."""
    mean_a = np.einsum("k,kn->n", weights, means_a)
    mean_b = np.einsum("k,kn->n", weights, means_b)
    return (
        np.einsum("k,kab->ab", weights, cross)
        + np.einsum("k,ka,kb->ab", weights, means_a, means_b)
        - np.outer(mean_a, mean_b)
    )


def switching_lgssm_exact_posterior(
    init_state_cond_mean: np.ndarray,
    init_state_cond_cov: np.ndarray,
    init_discrete_state_prob: np.ndarray,
    obs: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    process_cov: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_cov: np.ndarray,
) -> SwitchingExactPosterior:
    """Exact switching-LGSSM posterior by enumerating all ``K ** T`` paths.

    Argument shapes follow ``switching_kalman_filter`` (discrete axis last):
    ``init_state_cond_mean (n, K)``, ``init_state_cond_cov (n, n, K)``,
    ``continuous_transition_matrix (n, n, K)``, ``process_cov (n, n, K)``,
    ``measurement_matrix (m, n, K)``, ``measurement_cov (m, m, K)``,
    ``discrete_transition_matrix (K, K)`` row-stochastic.

    Each path's conditional posterior is a linear-Gaussian model with
    time-varying parameters, solved exactly by :func:`lgssm_dense_posterior`
    with the prior on ``x_1``. Path weights are
    ``p(s) p(y | s) / p(y)``; filtered quantities weight path prefixes by
    ``p(s_{1:t}) p(y_{1:t} | s_{1:t})`` (summing a full path's prior over its
    suffixes gives the prefix prior, so full-path enumeration is exact).
    """
    m0 = np.asarray(init_state_cond_mean, dtype=np.float64)
    P0 = np.asarray(init_state_cond_cov, dtype=np.float64)
    pi = np.asarray(init_discrete_state_prob, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    Z = np.asarray(discrete_transition_matrix, dtype=np.float64)
    A = np.asarray(continuous_transition_matrix, dtype=np.float64)
    Q = np.asarray(process_cov, dtype=np.float64)
    H = np.asarray(measurement_matrix, dtype=np.float64)
    R = np.asarray(measurement_cov, dtype=np.float64)
    n_time = obs.shape[0]
    n, K = m0.shape

    paths = np.array(list(itertools.product(range(K), repeat=n_time)), dtype=int)
    n_paths = paths.shape[0]
    with np.errstate(divide="ignore"):
        log_pi = np.log(pi)
        log_Z = np.log(Z)
    log_prior = np.zeros(n_paths)
    log_lik = np.zeros(n_paths)
    prefix_log_prior = np.zeros((n_paths, n_time))
    prefix_log_lik = np.zeros((n_paths, n_time))
    posts = []
    for p, s in enumerate(paths):
        lp = log_pi[s[0]]
        prefix_log_prior[p, 0] = lp
        for t in range(1, n_time):
            lp = lp + log_Z[s[t - 1], s[t]]
            prefix_log_prior[p, t] = lp
        log_prior[p] = lp
        post = lgssm_dense_posterior(
            m0[:, s[0]],
            P0[:, :, s[0]],
            obs,
            np.stack([A[:, :, k] for k in s]),
            np.stack([Q[:, :, k] for k in s]),
            np.stack([H[:, :, k] for k in s]),
            np.stack([R[:, :, k] for k in s]),
            prior_on_first_state=True,
        )
        posts.append(post)
        log_lik[p] = post.log_likelihood
        prefix_log_lik[p] = post.filtered_log_likelihood

    log_joint = log_prior + log_lik
    log_evidence = _logsumexp(log_joint)
    w = np.exp(log_joint - log_evidence)

    sm_all = np.stack([p.smoothed_mean for p in posts])  # (P, T, n)
    sc_all = np.stack([p.smoothed_cov for p in posts])  # (P, T, n, n)
    scc_all = np.stack([p.smoothed_cross_cov for p in posts])  # (P, T-1, n, n)
    fm_all = np.stack([p.filtered_mean for p in posts])
    fc_all = np.stack([p.filtered_cov for p in posts])

    # --- filtered ---
    filtered_ll = np.zeros(n_time)
    filtered_prob = np.zeros((n_time, K))
    filtered_mean = np.zeros((n_time, n))
    filtered_cov = np.zeros((n_time, n, n))
    sc_f_mean = np.zeros((n_time, n, K))
    sc_f_cov = np.zeros((n_time, n, n, K))
    for t in range(n_time):
        # each distinct prefix appears K**(T-1-t) times with prior summing to
        # the prefix prior; the full-path weights do that sum for us.
        lj = log_prior + prefix_log_lik[:, t]
        norm = _logsumexp(lj)
        filtered_ll[t] = norm
        wt = np.exp(lj - norm)
        filtered_mean[t], filtered_cov[t] = _mixture_moments(
            wt, fm_all[:, t], fc_all[:, t]
        )
        for j in range(K):
            mask = paths[:, t] == j
            filtered_prob[t, j] = wt[mask].sum()
            if filtered_prob[t, j] > 0:
                mm, cc = _mixture_moments(
                    wt[mask] / wt[mask].sum(), fm_all[mask, t], fc_all[mask, t]
                )
                sc_f_mean[t, :, j], sc_f_cov[t, :, :, j] = mm, cc

    # --- smoothed ---
    smoothed_prob = np.zeros((n_time, K))
    smoothed_mean = np.zeros((n_time, n))
    smoothed_cov = np.zeros((n_time, n, n))
    sc_s_mean = np.zeros((n_time, n, K))
    sc_s_cov = np.zeros((n_time, n, n, K))
    for t in range(n_time):
        smoothed_mean[t], smoothed_cov[t] = _mixture_moments(
            w, sm_all[:, t], sc_all[:, t]
        )
        for j in range(K):
            mask = paths[:, t] == j
            smoothed_prob[t, j] = w[mask].sum()
            if smoothed_prob[t, j] > 0:
                mm, cc = _mixture_moments(
                    w[mask] / w[mask].sum(), sm_all[mask, t], sc_all[mask, t]
                )
                sc_s_mean[t, :, j], sc_s_cov[t, :, :, j] = mm, cc

    joint_prob = np.zeros((n_time - 1, K, K))
    cross_cov = np.zeros((n_time - 1, n, n))
    pair_mean = np.zeros((n_time - 1, n, K, K))
    pair_cov = np.zeros((n_time - 1, n, n, K, K))
    next_pair_mean = np.zeros((n_time - 1, n, K, K))
    pair_cross = np.zeros((n_time - 1, n, n, K, K))
    for t in range(n_time - 1):
        cross_cov[t] = _mixture_cross_moments(
            w, sm_all[:, t], sm_all[:, t + 1], scc_all[:, t]
        )
        for i in range(K):
            for j in range(K):
                mask = (paths[:, t] == i) & (paths[:, t + 1] == j)
                joint_prob[t, i, j] = w[mask].sum()
                if joint_prob[t, i, j] <= 0:
                    continue
                wm = w[mask] / w[mask].sum()
                pm, pc = _mixture_moments(wm, sm_all[mask, t], sc_all[mask, t])
                nm, _ = _mixture_moments(wm, sm_all[mask, t + 1], sc_all[mask, t + 1])
                pair_mean[t, :, i, j] = pm
                pair_cov[t, :, :, i, j] = pc
                next_pair_mean[t, :, i, j] = nm
                pair_cross[t, :, :, i, j] = _mixture_cross_moments(
                    wm, sm_all[mask, t], sm_all[mask, t + 1], scc_all[mask, t]
                )

    return SwitchingExactPosterior(
        log_likelihood=float(log_evidence),
        filtered_log_likelihood=filtered_ll,
        filtered_discrete_prob=filtered_prob,
        filtered_mean=filtered_mean,
        filtered_cov=filtered_cov,
        state_cond_filtered_mean=sc_f_mean,
        state_cond_filtered_cov=sc_f_cov,
        smoothed_discrete_prob=smoothed_prob,
        smoothed_joint_discrete_prob=joint_prob,
        smoothed_mean=smoothed_mean,
        smoothed_cov=smoothed_cov,
        smoothed_cross_cov=cross_cov,
        state_cond_smoothed_mean=sc_s_mean,
        state_cond_smoothed_cov=sc_s_cov,
        pair_cond_smoothed_mean=pair_mean,
        pair_cond_smoothed_cov=pair_cov,
        next_pair_cond_smoothed_mean=next_pair_mean,
        pair_cond_smoothed_cross_cov=pair_cross,
        paths=paths,
        path_log_prior=log_prior,
        path_log_likelihood=log_lik,
        path_posteriors=posts,
    )


def switching_expected_complete_log_likelihood(
    posterior: SwitchingExactPosterior,
    obs: np.ndarray,
    init_state_cond_mean: np.ndarray,
    init_state_cond_cov: np.ndarray,
    init_discrete_state_prob: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    process_cov: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_cov: np.ndarray,
) -> float:
    """Exact EM auxiliary function of the switching LGSSM, by path enumeration.

    ``Q(theta) = sum_s P(s | y) E[log p(s, x, y | theta) | s, y]`` with the
    posterior (path weights and per-path Gaussians) held fixed at
    ``posterior``. Each path's term is written out directly::

        log pi[s_1] + sum_{t>=2} log Z[s_{t-1}, s_t]
        + E log N(x_1; m_{s_1}, P_{s_1})
        + sum_{t>=2} E log N(x_t; A_{s_t} x_{t-1}, Q_{s_t})
        + sum_{t>=1} E log N(y_t; H_{s_t} x_t, R_{s_t})

    The per-path, per-time expected second moments are accumulated per
    discrete state first, so evaluating Q for a new theta is cheap.
    """
    stats = switching_path_sufficient_statistics(posterior, obs)
    return switching_q_from_statistics(
        stats,
        init_state_cond_mean,
        init_state_cond_cov,
        init_discrete_state_prob,
        discrete_transition_matrix,
        continuous_transition_matrix,
        process_cov,
        measurement_matrix,
        measurement_cov,
    )


def switching_path_sufficient_statistics(
    posterior: SwitchingExactPosterior, obs: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-state expected sufficient statistics accumulated over every path.

    Each statistic is ``sum_s P(s | y) sum_{t : s_t = j} E[... | s, y]``,
    computed from the per-path posteriors (not from marginal/pair moments).
    """
    obs = np.asarray(obs, dtype=np.float64)
    paths = posterior.paths
    w = posterior.path_posterior_prob
    n_time = obs.shape[0]
    n = posterior.smoothed_mean.shape[1]
    m = obs.shape[1]
    K = posterior.smoothed_discrete_prob.shape[1]

    init_w = np.zeros(K)
    init_m = np.zeros((n, K))
    init_S = np.zeros((n, n, K))
    trans_w = np.zeros(K)
    trans_xx = np.zeros((n, n, K))  # E[x_t x_t^T], t >= 2
    trans_xp = np.zeros((n, n, K))  # E[x_t x_{t-1}^T]
    trans_pp = np.zeros((n, n, K))  # E[x_{t-1} x_{t-1}^T]
    obs_w = np.zeros(K)
    obs_yy = np.zeros((m, m, K))
    obs_yx = np.zeros((m, n, K))
    obs_xx = np.zeros((n, n, K))
    counts = np.zeros((K, K))
    for p, s in enumerate(paths):
        post = posterior.path_posteriors[p]
        wp = w[p]
        mu = post.smoothed_mean
        second = post.smoothed_cov + np.einsum("ta,tb->tab", mu, mu)
        lag = post.smoothed_cross_cov + np.einsum("ta,tb->tab", mu[:-1], mu[1:])
        j = s[0]
        init_w[j] += wp
        init_m[:, j] += wp * mu[0]
        init_S[:, :, j] += wp * second[0]
        for t in range(n_time):
            j = s[t]
            obs_w[j] += wp
            obs_yy[:, :, j] += wp * np.outer(obs[t], obs[t])
            obs_yx[:, :, j] += wp * np.outer(obs[t], mu[t])
            obs_xx[:, :, j] += wp * second[t]
            if t >= 1:
                trans_w[j] += wp
                trans_xx[:, :, j] += wp * second[t]
                trans_xp[:, :, j] += wp * lag[t - 1].T
                trans_pp[:, :, j] += wp * second[t - 1]
                counts[s[t - 1], j] += wp
    return {
        "init_w": init_w,
        "init_m": init_m,
        "init_S": init_S,
        "trans_w": trans_w,
        "trans_xx": trans_xx,
        "trans_xp": trans_xp,
        "trans_pp": trans_pp,
        "obs_w": obs_w,
        "obs_yy": obs_yy,
        "obs_yx": obs_yx,
        "obs_xx": obs_xx,
        "counts": counts,
    }


def switching_q_from_statistics(
    stats: dict[str, np.ndarray],
    init_state_cond_mean: np.ndarray,
    init_state_cond_cov: np.ndarray,
    init_discrete_state_prob: np.ndarray,
    discrete_transition_matrix: np.ndarray,
    continuous_transition_matrix: np.ndarray,
    process_cov: np.ndarray,
    measurement_matrix: np.ndarray,
    measurement_cov: np.ndarray,
) -> float:
    """Evaluate the switching Q(theta) from :func:`switching_path_sufficient_statistics`."""
    K = stats["init_w"].shape[0]
    with np.errstate(divide="ignore"):
        total = float(np.sum(stats["init_w"] * np.log(init_discrete_state_prob)))
        total += float(
            np.sum(
                np.where(
                    stats["counts"] > 0,
                    stats["counts"] * np.log(discrete_transition_matrix),
                    0.0,
                )
            )
        )
    for j in range(K):
        mj = init_state_cond_mean[:, j]
        wj = stats["init_w"][j]
        if wj > 0:
            resid = (
                stats["init_S"][:, :, j]
                - np.outer(stats["init_m"][:, j], mj)
                - np.outer(mj, stats["init_m"][:, j])
                + wj * np.outer(mj, mj)
            )
            total += _expected_gaussian_log_density(
                resid / wj, init_state_cond_cov[:, :, j], weight=wj
            )
        Aj = continuous_transition_matrix[:, :, j]
        wj = stats["trans_w"][j]
        if wj > 0:
            xp = stats["trans_xp"][:, :, j]
            resid = (
                stats["trans_xx"][:, :, j]
                - Aj @ xp.T
                - xp @ Aj.T
                + Aj @ stats["trans_pp"][:, :, j] @ Aj.T
            )
            total += _expected_gaussian_log_density(
                resid / wj, process_cov[:, :, j], weight=wj
            )
        Hj = measurement_matrix[:, :, j]
        wj = stats["obs_w"][j]
        if wj > 0:
            yx = stats["obs_yx"][:, :, j]
            resid = (
                stats["obs_yy"][:, :, j]
                - Hj @ yx.T
                - yx @ Hj.T
                + Hj @ stats["obs_xx"][:, :, j] @ Hj.T
            )
            total += _expected_gaussian_log_density(
                resid / wj, measurement_cov[:, :, j], weight=wj
            )
    return total


# ---------------------------------------------------------------------------
# Random model generators (NumPy RNG; shared by the oracle test modules)
# ---------------------------------------------------------------------------


def random_stable_matrix(
    rng: np.random.Generator, n: int, max_radius: float = 0.95
) -> np.ndarray:
    """Random n x n matrix scaled to spectral radius in (0.3, max_radius)."""
    M = rng.normal(size=(n, n))
    radius = max(np.max(np.abs(np.linalg.eigvals(M))), 1e-12)
    return M * (rng.uniform(0.3, max_radius) / radius)


def random_spd_matrix(
    rng: np.random.Generator, n: int, scale: float = 1.0, min_eig: float = 0.1
) -> np.ndarray:
    """Random symmetric positive-definite matrix with eigenvalues in
    ``scale * [min_eig, min_eig + ~2]``."""
    M = rng.normal(size=(n, n))
    U, _ = np.linalg.qr(M)
    eigs = scale * (min_eig + rng.uniform(0.0, 2.0, size=n))
    S = (U * eigs) @ U.T
    return 0.5 * (S + S.T)
