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
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import gammaln

from state_space_practice.kalman import rts_backward_scan
from state_space_practice.utils import symmetrize, typed_jit


class LinearSiteResult(NamedTuple):
    mean: Array
    covariance: Array
    cross_covariance: Array
    log_normalizer: Array


class LaplaceTrajectoryResult(NamedTuple):
    """Joint mode, inverse-Hessian moments and evidence with diagnostics."""

    mean: Array
    covariance: Array
    cross_covariance: Array
    log_evidence: Array
    relative_newton_step: Array
    n_iterations: Array
    n_rejected_steps: Array


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
        log_z = (
            -0.5 * jnp.log1p(w * v)
            + 0.5 * (2 * u * a - w * a * a + v * u * u) / denominator
        )
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


def _iterate_mode(
    initial: Array,
    target: Callable[[Array], Array],
    merit: Callable[[Array], Array],
    max_iter: int,
    tolerance: float,
) -> tuple[Array, Array, Array, Array]:
    """Adaptive, gradient-stopped Armijo iteration; rejected steps keep the mode."""
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


@partial(typed_jit, static_argnames=("dt", "max_iter", "tolerance"))
def poisson_laplace_smoother(
    mean0: Array,
    covariance0: Array,
    transition: Array,
    noise: Array,
    design: Array,
    counts: Array,
    valid: Array,
    *,
    dt: float,
    max_iter: int = 50,
    tolerance: float = 1e-8,
) -> LaplaceTrajectoryResult:
    """Infer the full trajectory mode and normalized joint Laplace evidence.

    Parameters follow the graph model's direct-first-prior convention. Positive
    definite noise or exactly zero noise is supported. Missing rows contribute
    zero information while transitions remain. Inputs must be validated by the
    model layer. Gradients are implicit mode derivatives and require convergence;
    callers must inspect ``relative_newton_step``/``n_rejected_steps``.
    At exactly zero noise, derivatives describe the reduced deterministic
    model with noise fixed to zero; they are not right derivatives in Q.
    Compare zero with positive-noise evidence explicitly when learning Q.
    """
    n_time, rank = design.shape
    is_static = jnp.all(noise == 0)
    # vmap evaluates both cond branches. Keep the unused dynamic branch finite
    # for static neurons, including its reverse-mode derivatives.
    safe_noise = jnp.where(is_static, jnp.eye(rank, dtype=noise.dtype), noise)
    y = jnp.where(valid, counts, 0.0)

    def sites(path: Array) -> tuple[Array, Array, Array]:
        eta = jnp.sum(design * path, axis=1)
        rate = jnp.exp(jnp.minimum(eta + jnp.log(dt), 600.0))
        weight = jnp.where(valid, rate, 0.0)
        return eta, weight, y - weight + weight * eta

    def poisson_value(path: Array) -> Array:
        eta, weight, _ = sites(path)
        return jnp.sum(
            jnp.where(valid, y * (eta + jnp.log(dt)) - weight - gammaln(y + 1), 0.0)
        )

    def dynamic(_: None) -> LaplaceTrajectoryResult:
        prior_chol, noise_chol = (
            jnp.linalg.cholesky(covariance0),
            jnp.linalg.cholesky(safe_noise),
        )

        def target(path: Array) -> Array:
            _, weight, information = sites(path)
            return _linear_sites(
                mean0, covariance0, transition, safe_noise, design, weight, information
            ).mean

        def prior_step(previous: Array, _: None) -> tuple[Array, Array]:
            mean = transition @ previous
            return mean, mean

        _last_mean, rest = jax.lax.scan(prior_step, mean0, None, length=n_time - 1)
        initial = jnp.concatenate((mean0[None], rest))

        # All parameters of the primal solver are stopped. The derivative of
        # the exact Newton map at its fixed point supplies the mode derivative.
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

        def stopped_target(path: Array) -> Array:
            eta = jnp.sum(pz * path, axis=1)
            weight = jnp.where(
                valid, jnp.exp(jnp.minimum(eta + jnp.log(dt), 600.0)), 0.0
            )
            information = py - weight + weight * eta
            return _linear_sites(pm, pp, pa, pq, pz, weight, information).mean

        def stopped_merit(path: Array) -> Array:
            eta = jnp.sum(pz * path, axis=1) + jnp.log(dt)
            ll = jnp.sum(
                jnp.where(
                    valid,
                    py * eta - jnp.exp(jnp.minimum(eta, 600.0)) - gammaln(py + 1),
                    0.0,
                )
            )
            first = solve_triangular(pchol, path[0] - pm, lower=True)
            rest = solve_triangular(pqchol, (path[1:] - path[:-1] @ pa.T).T, lower=True)
            return ll - 0.5 * (jnp.sum(first**2) + jnp.sum(rest**2))

        stopped_path, residual, n, rejected = _iterate_mode(
            jax.lax.stop_gradient(initial),
            stopped_target,
            stopped_merit,
            max_iter,
            tolerance,
        )
        stopped_path = jax.lax.stop_gradient(stopped_path)
        exact_target = target(stopped_path)
        mode = stopped_path + (exact_target - jax.lax.stop_gradient(exact_target))
        eta, weight, information = sites(mode)
        gaussian = _linear_sites(
            mean0, covariance0, transition, safe_noise, design, weight, information
        )
        correction = poisson_value(mode) - jnp.sum(
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
        )

    def static(_: None) -> LaplaceTrajectoryResult:
        chol = jnp.linalg.cholesky(covariance0)

        def loading_step(previous: Array, _: None) -> tuple[Array, Array]:
            matrix = transition @ previous
            return matrix, matrix

        _last_loading, remaining = jax.lax.scan(
            loading_step, jnp.eye(rank), None, length=n_time - 1
        )
        loading = jnp.concatenate((jnp.eye(rank)[None], remaining))
        prior_path = jnp.einsum("tij,j->ti", loading, mean0)
        white_loading = loading @ chol
        white_design = jnp.einsum("ti,tij->tj", design, white_loading)
        offset = jnp.sum(design * prior_path, axis=1) + jnp.log(dt)

        def weights(u: Array) -> Array:
            return jnp.where(
                valid, jnp.exp(jnp.minimum(offset + white_design @ u, 600.0)), 0.0
            )

        def hessian(u: Array) -> Array:
            return jnp.eye(rank) + white_design.T @ (weights(u)[:, None] * white_design)

        def target(u: Array) -> Array:
            w = weights(u)
            gradient = white_design.T @ (y - w) - u
            return cast(Array, u + jnp.linalg.solve(hessian(u), gradient))

        def merit(u: Array) -> Array:
            eta = offset + white_design @ u
            return (
                jnp.sum(jnp.where(valid, y * eta - weights(u) - gammaln(y + 1), 0.0))
                - 0.5 * u @ u
            )

        pdesign, poffset, py = jax.lax.stop_gradient((white_design, offset, y))

        def stopped_target(u: Array) -> Array:
            w = jnp.where(
                valid, jnp.exp(jnp.minimum(poffset + pdesign @ u, 600.0)), 0.0
            )
            h = jnp.eye(rank) + pdesign.T @ (w[:, None] * pdesign)
            return cast(Array, u + jnp.linalg.solve(h, pdesign.T @ (py - w) - u))

        def stopped_merit(u: Array) -> Array:
            eta = poffset + pdesign @ u
            return (
                jnp.sum(
                    jnp.where(
                        valid,
                        py * eta - jnp.exp(jnp.minimum(eta, 600.0)) - gammaln(py + 1),
                        0.0,
                    )
                )
                - 0.5 * u @ u
            )

        u, residual, n, rejected = _iterate_mode(
            jnp.zeros(rank), stopped_target, stopped_merit, max_iter, tolerance
        )
        u = jax.lax.stop_gradient(u)
        exact_target = target(u)
        mode_u = u + (exact_target - jax.lax.stop_gradient(exact_target))
        h = hessian(mode_u)
        inverse = cho_solve((jnp.linalg.cholesky(h), True), jnp.eye(rank))
        mean = prior_path + jnp.einsum("tij,j->ti", white_loading, mode_u)
        covariance = jnp.einsum(
            "tij,jk,tlk->til", white_loading, inverse, white_loading
        )
        cross = jnp.einsum(
            "tij,jk,tlk->til", white_loading[:-1], inverse, white_loading[1:]
        )
        ll = merit(mode_u) - 0.5 * jnp.linalg.slogdet(h)[1]
        return LaplaceTrajectoryResult(
            mean, covariance, cross, ll, residual, n, rejected
        )

    return cast(
        LaplaceTrajectoryResult, jax.lax.cond(is_static, static, dynamic, operand=None)
    )
