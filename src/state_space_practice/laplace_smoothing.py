"""Joint Laplace inference for masked linear-log Poisson state-space models.

Newton directions use Gaussian information sites and the shared RTS smoother.
The first observation uses the initial prior directly. Zero process covariance
uses a reduced deterministic state; no zero covariance is inverted. Mode
derivatives use one exact Newton map at the converged mode, permitting adaptive
primal iteration without differentiating through a while loop.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array
from jax.ad_checkpoint import checkpoint_name
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import gammaln
from jax.typing import ArrayLike

from state_space_practice.kalman import (
    _parallel_information_filter,
    parallel_kalman_smoother,
    rts_backward_scan,
)
from state_space_practice.utils import symmetrize, typed_jit

# Checkpoint name of the adaptive iteration's outputs. A ``jax.checkpoint``
# policy that saves it lets the backward pass replay only the exact Newton map
# and final linear pass instead of the whole iteration.
LAPLACE_MODE_CHECKPOINT = "laplace_mode"

# Log-rate ceiling keeping exp finite in float64.
_MAX_LOG_RATE = 600.0


def _clipped_exp(log_rate: Array) -> Array:
    return jnp.exp(jnp.minimum(log_rate, _MAX_LOG_RATE))


class LinearSiteResult(NamedTuple):
    """Gaussian-site smoother moments: mean ``(n_time, rank)``, covariance
    ``(n_time, rank, rank)``, lag-one cross-covariance ``Cov(x_t, x_{t+1})``
    ``(n_time - 1, rank, rank)`` and scalar log normalizer."""

    mean: Array
    covariance: Array
    cross_covariance: Array
    log_normalizer: Array


class LaplaceTrajectoryResult(NamedTuple):
    """Joint mode, inverse-Hessian moments and evidence with diagnostics.

    ``mean`` is ``(n_time, rank)``, ``covariance`` ``(n_time, rank, rank)`` and
    ``cross_covariance`` ``Cov(x_t, x_{t+1})`` ``(n_time - 1, rank, rank)``; the
    remaining fields are scalars. ``relative_newton_step`` is
    ``max|proposed - path| / (1 + max|path|)`` at exit. ``converged`` is true when the final relative Newton step is within the
    solver tolerance and no line search was fully rejected; the evidence and
    its implicit derivatives are only valid when it holds.
    """

    mean: Array
    covariance: Array
    cross_covariance: Array
    log_evidence: Array
    relative_newton_step: Array
    n_iterations: Array
    n_rejected_steps: Array
    converged: Array


def resolve_parallel(parallel: bool | None) -> bool:
    """Whether to run the associative-scan linear pass.

    ``None`` selects it on accelerators (GPU/TPU), where it is much faster, and
    the sequential pass on CPU, where it is slower. ``True``/``False`` are kept.
    """
    if parallel is None:
        return jax.default_backend() != "cpu"
    return bool(parallel)


def _site_log_normalizer(v: Array, a: Array, w: Array, u: Array) -> Array:
    """``log ∫ N(x; m, P) exp(u z·x − ½ w (z·x)²) dx`` with ``v = zᵀPz``, ``a = z·m``."""
    denominator = 1 + w * v
    return (
        -0.5 * jnp.log1p(w * v)
        + 0.5 * (2 * u * a - w * a * a + v * u * u) / denominator
    )


def _linear_sites(
    mean0: Array,
    covariance0: Array,
    transition: Array,
    noise: Array,
    design: Array,
    weight: Array,
    information: Array,
) -> LinearSiteResult:
    """Gaussian smoothing with scalar information sites, including zero weight."""
    identity = jnp.eye(mean0.size, dtype=mean0.dtype)

    def update(
        mean: Array, covariance: Array, z: Array, w: Array, u: Array
    ) -> tuple[Array, Array, Array]:
        pz = covariance @ z
        v, a = z @ pz, z @ mean
        denominator = 1 + w * v
        posterior_mean = mean + pz * (u - w * a) / denominator
        # Joseph form without ever forming R=1/w; w=0 is an identity update.
        matrix = identity - (w / denominator) * jnp.outer(pz, z)
        posterior_covariance = symmetrize(
            matrix @ covariance @ matrix.T + w / denominator**2 * jnp.outer(pz, pz)
        )
        log_z = _site_log_normalizer(v, a, w, u)
        return posterior_mean, posterior_covariance, log_z

    first_mean, first_covariance, first_ll = update(
        mean0, covariance0, design[0], weight[0], information[0]
    )

    def step(
        carry: tuple[Array, Array, Array], inputs: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array]]:
        previous_mean, previous_covariance, ll = carry
        z, w, u = inputs
        predicted_covariance = symmetrize(
            transition @ previous_covariance @ transition.T + noise
        )
        mean, covariance, log_z = update(
            transition @ previous_mean, predicted_covariance, z, w, u
        )
        return (mean, covariance, ll + log_z), (mean, covariance)

    (_, _, ll), (remaining_mean, remaining_covariance) = jax.lax.scan(
        step,
        (first_mean, first_covariance, first_ll),
        (design[1:], weight[1:], information[1:]),
    )
    filtered_mean = jnp.concatenate((first_mean[None], remaining_mean))
    filtered_covariance = jnp.concatenate(
        (first_covariance[None], remaining_covariance)
    )
    mean, covariance, cross = rts_backward_scan(
        filtered_mean, filtered_covariance, transition, noise
    )
    return LinearSiteResult(mean, covariance, cross, ll)


def _linear_sites_parallel(
    mean0: Array,
    covariance0: Array,
    transition: Array,
    noise: Array,
    design: Array,
    weight: Array,
    information: Array,
) -> LinearSiteResult:
    """:func:`_linear_sites` by associative scans (O(log T) span, same outputs).

    Each rank-one site is ``G_t = w_t z_t z_tᵀ``, ``g_t = u_t z_t``; the first site
    conditions ``N(mean0, covariance0)`` directly, as in the sequential pass.
    """
    info_matrices = weight[:, None, None] * design[:, :, None] * design[:, None, :]
    filtered_mean, filtered_covariance = _parallel_information_filter(
        mean0,
        covariance0,
        transition,
        noise,
        info_matrices,
        information[:, None] * design,
    )
    predicted_mean = jnp.concatenate((mean0[None], filtered_mean[:-1] @ transition.T))
    predicted_covariance = jnp.concatenate(
        (
            covariance0[None],
            symmetrize(
                jnp.einsum(
                    "ij,tjk,lk->til", transition, filtered_covariance[:-1], transition
                )
                + noise
            ),
        )
    )
    variance = jnp.einsum("ti,tij,tj->t", design, predicted_covariance, design)
    log_z = _site_log_normalizer(
        variance, jnp.sum(design * predicted_mean, axis=1), weight, information
    )
    mean, covariance, cross = parallel_kalman_smoother(
        filtered_mean, filtered_covariance, transition, noise
    )
    return LinearSiteResult(mean, covariance, cross, jnp.sum(log_z))


def _transition_powers(transition: Array, n_time: int) -> Array:
    """``A^0 .. A^(n_time-1)``, shape ``(n_time, D, D)``, by associative scan."""
    identity = jnp.eye(transition.shape[0], dtype=transition.dtype)
    if n_time == 1:
        return identity[None]
    stacked = jnp.broadcast_to(transition, (n_time - 1, *transition.shape))
    powers = jax.lax.associative_scan(lambda earlier, later: later @ earlier, stacked)
    return jnp.concatenate((identity[None], powers))


def _iterate_mode(
    initial: Array,
    target: Callable[[Array], Array],
    merit: Callable[[Array], Array],
    max_iter: int,
    tolerance: float,
) -> tuple[Array, Array, Array, Array]:
    """Adaptive, gradient-stopped Armijo iteration.

    A fully rejected line search stops the iteration at the current point and
    is counted in the returned rejection count.
    """
    trial_steps = 0.5 ** jnp.arange(20, dtype=initial.dtype)

    def relative(path: Array, proposed: Array) -> Array:
        return jnp.max(jnp.abs(proposed - path)) / (1 + jnp.max(jnp.abs(path)))

    proposed = target(initial)

    def condition(carry: tuple[Array, Array, Array, Array]) -> Array:
        path, proposed, n, rejected = carry
        return (n < max_iter) & (relative(path, proposed) > tolerance) & (rejected == 0)

    def step(
        carry: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        path, proposed, n, rejected = carry
        direction = proposed - path
        current = merit(path)
        slope = jnp.sum(jax.grad(merit)(path) * direction)
        trials = (
            path[None] + trial_steps.reshape((-1,) + (1,) * path.ndim) * direction[None]
        )
        values = jax.vmap(merit)(trials)
        accepted = jnp.isfinite(values) & (
            values
            >= current + 1e-4 * trial_steps * slope - 1e-10 * (1 + jnp.abs(current))
        )
        any_accepted = jnp.any(accepted)
        index = jnp.argmax(accepted)
        next_path = jnp.where(any_accepted, trials[index], path)
        return (
            next_path,
            target(next_path),
            n + 1,
            rejected + (~any_accepted).astype(jnp.int32),
        )

    path, proposed, n, rejected = jax.lax.while_loop(
        condition,
        step,
        (initial, proposed, jnp.array(0), jnp.array(0)),
    )
    return path, relative(path, proposed), n, rejected


@partial(
    typed_jit,
    static_argnames=("dt", "max_iter", "tolerance", "static_noise", "parallel"),
)
def poisson_laplace_smoother(
    mean0: ArrayLike,
    covariance0: ArrayLike,
    transition: ArrayLike,
    noise: ArrayLike,
    design: ArrayLike,
    counts: ArrayLike,
    valid: ArrayLike,
    *,
    dt: float,
    max_iter: int = 50,
    tolerance: float = 1e-8,
    static_noise: bool | None = None,
    parallel: bool | None = None,
) -> LaplaceTrajectoryResult:
    """Infer the full trajectory mode and normalized joint Laplace evidence.

    The first observation is conditioned directly on ``N(mean0, covariance0)``
    with no preceding predict step. Positive
    definite noise or exactly zero noise is supported. Missing rows contribute
    zero information while transitions remain. Inputs must be validated by the
    model layer. Gradients are implicit mode derivatives and require convergence;
    callers must inspect ``converged``.
    At exactly zero noise, derivatives describe the reduced deterministic
    model with noise fixed to zero; they are not right derivatives in Q.
    Compare zero with positive-noise evidence explicitly when learning Q.
    Only first derivatives are exact: the one-Newton-map construction does not
    give correct second derivatives (e.g. ``jax.hessian`` of the evidence).

    Parameters
    ----------
    mean0 : ArrayLike, shape (rank,)
    covariance0 : ArrayLike, shape (rank, rank)
        Positive definite prior covariance of the first state.
    transition : ArrayLike, shape (rank, rank)
    noise : ArrayLike, shape (rank, rank)
        Process covariance; positive definite or exactly zero.
    design : ArrayLike, shape (n_time, rank)
        Log-rate design rows (the log rate is ``design[t] @ x_t + log(dt)``).
    counts : ArrayLike, shape (n_time,)
    valid : ArrayLike of bool, shape (n_time,)
        Observed rows; ``False`` rows contribute no likelihood.
    dt : float
        Bin width in seconds.
    max_iter : int, optional
        Newton iteration cap, by default 50.
    tolerance : float, optional
        Relative Newton step tolerance, by default 1e-8.
    static_noise : bool or None, optional
        Which solver to run. ``None`` (default) tests ``noise == 0`` with
        ``lax.cond``; under ``vmap`` with a batched noise that becomes a select
        that evaluates both solvers. ``True``/``False`` runs only the
        zero-noise/positive-definite solver; the caller guarantees the noise
        matches.
    parallel : bool or None, optional
        Run each linear pass (filter and RTS) and the prior powers by
        associative scans, with O(log n_time) sequential depth instead of
        O(n_time). ``None`` (default) selects them on GPU/TPU and the sequential
        pass on CPU (:func:`resolve_parallel`). Both converge to the same mode
        and evidence. On an A100 the parallel pass was 18-129x faster for one
        trajectory (n_time 750-70,930, rank 20) and 5-20x for 8-32 batched
        trajectories; on CPU it was 2-5x slower. With gradients it needs about
        3-4x the device memory of the sequential pass (12-17 GiB vs about
        4 GiB per trajectory at n_time = 70,930, rank 20), scaling with
        ``n_time * rank**2``. See Särkkä & García-Fernández (2021), IEEE TAC
        66(1), 299-306.

    Returns
    -------
    LaplaceTrajectoryResult
        Mode, inverse-Hessian moments, and the Laplace log evidence including
        the Poisson ``-log(y!)`` constants.
    """
    # Use one promoted floating dtype for loop carries and both cond branches.
    dtype = jnp.result_type(
        mean0, covariance0, transition, noise, design, counts, jnp.float32
    )
    mean0, covariance0, transition, noise, design, counts = (
        jnp.asarray(value, dtype=dtype)
        for value in (mean0, covariance0, transition, noise, design, counts)
    )
    valid = jnp.asarray(valid)
    n_time, rank = design.shape
    identity = jnp.eye(rank, dtype=dtype)
    is_static = jnp.all(noise == 0)
    use_parallel = resolve_parallel(parallel)
    linear_sites = _linear_sites_parallel if use_parallel else _linear_sites
    # vmap evaluates both cond branches. Keep the unused dynamic branch finite
    # for static neurons, including its reverse-mode derivatives.
    safe_noise = jnp.where(is_static, identity, noise)
    y = jnp.where(valid, counts, 0.0)

    # Each objective below is built twice: from live parameters for the exact
    # Newton map at the mode, and from gradient-stopped parameters for the
    # primal iteration. The derivative of that one exact map at its fixed
    # point supplies the mode derivative.

    def poisson_log_likelihood(z: Array, yy: Array, path: Array) -> Array:
        eta = jnp.sum(z * path, axis=1) + jnp.log(dt)
        return jnp.sum(
            jnp.where(valid, yy * eta - _clipped_exp(eta) - gammaln(yy + 1), 0.0)
        )

    def sites(z: Array, yy: Array, path: Array) -> tuple[Array, Array, Array]:
        eta = jnp.sum(z * path, axis=1)
        weight = jnp.where(valid, _clipped_exp(eta + jnp.log(dt)), 0.0)
        return eta, weight, yy - weight + weight * eta

    def dynamic(_: None) -> LaplaceTrajectoryResult:
        prior_chol, noise_chol = (
            jnp.linalg.cholesky(covariance0),
            jnp.linalg.cholesky(safe_noise),
        )

        def newton_target(
            m0: Array, p0: Array, a: Array, q: Array, z: Array, yy: Array
        ) -> Callable[[Array], Array]:
            def target(path: Array) -> Array:
                _, weight, information = sites(z, yy, path)
                return linear_sites(m0, p0, a, q, z, weight, information).mean

            return target

        def prior_step(previous: Array, _: None) -> tuple[Array, Array]:
            mean = transition @ previous
            return mean, mean

        if use_parallel:
            initial = _transition_powers(transition, n_time) @ mean0
        else:
            _last_mean, rest = jax.lax.scan(prior_step, mean0, None, length=n_time - 1)
            initial = jnp.concatenate((mean0[None], rest))

        pm, pp, pa, pq, pz, py, pchol, pqchol = jax.lax.stop_gradient(
            (
                mean0,
                covariance0,
                transition,
                safe_noise,
                design,
                y,
                prior_chol,
                noise_chol,
            )
        )

        def stopped_merit(path: Array) -> Array:
            first = solve_triangular(pchol, path[0] - pm, lower=True)
            rest = solve_triangular(pqchol, (path[1:] - path[:-1] @ pa.T).T, lower=True)
            return poisson_log_likelihood(pz, py, path) - 0.5 * (
                jnp.sum(first**2) + jnp.sum(rest**2)
            )

        stopped_path, residual, n, rejected = _iterate_mode(
            jax.lax.stop_gradient(initial),
            newton_target(pm, pp, pa, pq, pz, py),
            stopped_merit,
            max_iter,
            tolerance,
        )
        stopped_path, residual, n, rejected = (
            checkpoint_name(jax.lax.stop_gradient(value), LAPLACE_MODE_CHECKPOINT)
            for value in (stopped_path, residual, n, rejected)
        )
        exact_target = newton_target(
            mean0, covariance0, transition, safe_noise, design, y
        )(stopped_path)
        mode = stopped_path + (exact_target - jax.lax.stop_gradient(exact_target))
        eta, weight, information = sites(design, y, mode)
        gaussian = linear_sites(
            mean0, covariance0, transition, safe_noise, design, weight, information
        )
        correction = poisson_log_likelihood(design, y, mode) - jnp.sum(
            -0.5 * weight * eta**2 + information * eta
        )
        return LaplaceTrajectoryResult(
            mode,
            gaussian.covariance,
            gaussian.cross_covariance,
            gaussian.log_normalizer + correction,
            residual,
            n,
            rejected,
            (residual <= tolerance) & (rejected == 0),
        )

    def static(_: None) -> LaplaceTrajectoryResult:
        chol = jnp.linalg.cholesky(covariance0)

        def loading_step(previous: Array, _: None) -> tuple[Array, Array]:
            matrix = transition @ previous
            return matrix, matrix

        if use_parallel:
            loading = _transition_powers(transition, n_time)
        else:
            _last_loading, remaining = jax.lax.scan(
                loading_step, identity, None, length=n_time - 1
            )
            loading = jnp.concatenate((identity[None], remaining))
        prior_path = jnp.einsum("tij,j->ti", loading, mean0)
        white_loading = loading @ chol
        white_design = jnp.einsum("ti,tij->tj", design, white_loading)
        offset = jnp.sum(design * prior_path, axis=1) + jnp.log(dt)

        def whitened_newton(
            z: Array, off: Array, yy: Array
        ) -> tuple[
            Callable[[Array], Array], Callable[[Array], Array], Callable[[Array], Array]
        ]:
            def hessian(u: Array) -> Array:
                w = jnp.where(valid, _clipped_exp(off + z @ u), 0.0)
                return identity + z.T @ (w[:, None] * z)

            def target(u: Array) -> Array:
                w = jnp.where(valid, _clipped_exp(off + z @ u), 0.0)
                h = identity + z.T @ (w[:, None] * z)
                return cast(Array, u + jnp.linalg.solve(h, z.T @ (yy - w) - u))

            def merit(u: Array) -> Array:
                eta = off + z @ u
                return (
                    jnp.sum(
                        jnp.where(
                            valid, yy * eta - _clipped_exp(eta) - gammaln(yy + 1), 0.0
                        )
                    )
                    - 0.5 * u @ u
                )

            return hessian, target, merit

        hessian, target, merit = whitened_newton(white_design, offset, y)
        _stopped_hessian, stopped_target, stopped_merit = whitened_newton(
            *jax.lax.stop_gradient((white_design, offset, y))
        )

        u, residual, n, rejected = _iterate_mode(
            jnp.zeros(rank, dtype=dtype),
            stopped_target,
            stopped_merit,
            max_iter,
            tolerance,
        )
        u, residual, n, rejected = (
            checkpoint_name(jax.lax.stop_gradient(value), LAPLACE_MODE_CHECKPOINT)
            for value in (u, residual, n, rejected)
        )
        exact_target = target(u)
        mode_u = u + (exact_target - jax.lax.stop_gradient(exact_target))
        h = hessian(mode_u)
        inverse = cho_solve((jnp.linalg.cholesky(h), True), identity)
        mean = prior_path + jnp.einsum("tij,j->ti", white_loading, mode_u)
        covariance = jnp.einsum(
            "tij,jk,tlk->til", white_loading, inverse, white_loading
        )
        cross = jnp.einsum(
            "tij,jk,tlk->til", white_loading[:-1], inverse, white_loading[1:]
        )
        ll = merit(mode_u) - 0.5 * jnp.linalg.slogdet(h)[1]
        return LaplaceTrajectoryResult(
            mean,
            covariance,
            cross,
            ll,
            residual,
            n,
            rejected,
            (residual <= tolerance) & (rejected == 0),
        )

    if static_noise is None:
        return cast(
            LaplaceTrajectoryResult,
            jax.lax.cond(is_static, static, dynamic, operand=None),
        )
    return static(None) if static_noise else dynamic(None)
