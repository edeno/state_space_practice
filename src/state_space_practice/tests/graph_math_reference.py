"""Independent numerical references for graph place-field math validation.

No package inference, graph-basis, or covariance helpers are imported here.
The scalar Poisson reference discretizes Bayes' rule on a uniform grid. Direct
Gaussian convolution integrates random-walk transitions; no Gaussian posterior
approximation is made. Refinement and expanded-domain checks are required before
treating its moments or normalizer as an oracle.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln, logsumexp
from scipy.stats import norm


class GridPosterior(NamedTuple):
    grid: NDArray[np.float64]
    filtered_mean: NDArray[np.float64]
    filtered_variance: NDArray[np.float64]
    smoothed_mean: NDArray[np.float64]
    smoothed_variance: NDArray[np.float64]
    lag_covariance: NDArray[np.float64]
    log_evidence: float
    smoothed_mass: NDArray[np.float64]
    maximum_transition_mass_error: float
    maximum_boundary_mass: float


class GaussianPosterior(NamedTuple):
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    lag_covariance: NDArray[np.float64]
    joint_covariance: NDArray[np.float64]


def poisson_grid_posterior(
    initial_mean: float,
    initial_variance: float,
    process_variance: float,
    design: NDArray[np.float64],
    counts: NDArray[np.float64],
    valid: NDArray[np.bool_],
    dt: float,
    *,
    n_grid: int = 1001,
    radius_sd: float = 10.0,
    transition_radius_sd: float = 12.0,
) -> GridPosterior:
    """Scalar w_0 ~ N(m,P), w_t=w_(t-1)+N(0,Q), y_t~Pois(dt exp(z_t w_t)).

    Row zero uses P directly. Invalid rows integrate only the transition and
    contribute no observation likelihood. Q is variance per row, not per second.
    Bounds span radius_sd times the largest unconditional prior standard deviation.
    Transition tails are truncated at transition_radius_sd standard deviations.
    Insufficient resolution or significant probability loss raises instead of
    silently normalizing a bad discretization. Check transition-tail refinement
    as well as grid/domain refinement when assessing rare observations.
    """
    design = np.asarray(design, dtype=float)
    counts = np.asarray(counts, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    n_time = len(design)
    if not (initial_variance > 0 and process_variance >= 0 and dt > 0):
        raise ValueError("Positive prior variance/dt and nonnegative Q are required.")
    if n_time == 0 or counts.shape != design.shape or valid.shape != design.shape:
        raise ValueError("Design, counts and mask must be aligned nonempty vectors.")
    if n_grid < 101 or n_grid % 2 != 1:
        raise ValueError("Use an odd grid with at least 101 points.")
    radius = radius_sd * np.sqrt(initial_variance + (n_time - 1) * process_variance)
    grid = np.linspace(initial_mean - radius, initial_mean + radius, n_grid)
    dx = float(grid[1] - grid[0])
    prior = norm.pdf(grid, initial_mean, np.sqrt(initial_variance)) * dx
    prior /= prior.sum()
    kernel = None
    if process_variance > 0:
        offsets = np.arange(-(n_grid - 1), n_grid) * dx
        kernel = norm.pdf(offsets, 0, np.sqrt(process_variance)) * dx
        if abs(float(kernel.sum()) - 1) > 1e-9:
            raise ValueError("Transition kernel is under-resolved; increase n_grid.")

    def transition(values: NDArray[np.float64]) -> NDArray[np.float64]:
        if kernel is None:
            return values.copy()
        # Direct positive summation avoids FFT cancellation manufacturing tiny
        # probability floors in tails that rare observations can amplify.
        return np.asarray(
            gaussian_filter1d(
                values,
                np.sqrt(process_variance) / dx,
                mode="constant",
                cval=0.0,
                truncate=transition_radius_sd,
            ),
            dtype=float,
        )

    filtered = np.empty((n_time, n_grid))
    observation_weights = np.ones_like(filtered)
    log_evidence = 0.0
    mass_error, boundary_mass = 0.0, 0.0
    for t in range(n_time):
        if t > 0:
            prior = np.maximum(transition(filtered[t - 1]), 0.0)
            mass_error = max(mass_error, abs(float(prior.sum()) - 1))
            prior /= prior.sum()
        if valid[t]:
            log_mu = np.log(dt) + design[t] * grid
            log_likelihood = (
                counts[t] * log_mu - np.exp(log_mu) - gammaln(counts[t] + 1)
            )
            observation_weights[t] = np.exp(log_likelihood - log_likelihood.max())
            # Zero mass stays zero; never manufacture an arbitrary tail floor.
            log_prior = np.full(n_grid, -np.inf)
            np.log(prior, out=log_prior, where=prior > 0)
            normalizer = float(logsumexp(log_prior + log_likelihood))
            filtered[t] = np.exp(log_prior + log_likelihood - normalizer)
            log_evidence += normalizer
        else:
            filtered[t] = prior
        boundary_mass = max(
            boundary_mass, float(filtered[t, :4].sum() + filtered[t, -4:].sum())
        )

    backward = np.ones_like(filtered)
    for t in range(n_time - 2, -1, -1):
        message = np.maximum(
            transition(observation_weights[t + 1] * backward[t + 1]), 0.0
        )
        backward[t] = message / message.max()
    smoothed = filtered * backward
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    boundary_mass = max(
        boundary_mass,
        float(np.max(smoothed[:, :4].sum(axis=1) + smoothed[:, -4:].sum(axis=1))),
    )
    filtered_mean = filtered @ grid
    smoothed_mean = smoothed @ grid
    filtered_var = np.sum(
        filtered * (grid[None, :] - filtered_mean[:, None]) ** 2, axis=1
    )
    smoothed_var = np.sum(
        smoothed * (grid[None, :] - smoothed_mean[:, None]) ** 2, axis=1
    )
    lag_covariance = np.empty(n_time - 1)
    for t in range(n_time - 1):
        right = observation_weights[t + 1] * backward[t + 1]
        denominator = float(filtered[t] @ np.maximum(transition(right), 0.0))
        cross_moment = (
            float((filtered[t] * grid) @ transition(grid * right)) / denominator
        )
        lag_covariance[t] = cross_moment - smoothed_mean[t] * smoothed_mean[t + 1]
    if mass_error > 1e-8 or boundary_mass > 1e-8:
        raise ValueError("Posterior grid lost probability; expand/refine the domain.")
    return GridPosterior(
        grid,
        filtered_mean,
        filtered_var,
        smoothed_mean,
        smoothed_var,
        lag_covariance,
        log_evidence,
        smoothed,
        mass_error,
        boundary_mass,
    )


def interval_mass(
    posterior: GridPosterior, lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Probability assigned by the reference posterior to supplied intervals."""
    dx = posterior.grid[1] - posterior.grid[0]
    cdf = cumulative_trapezoid(
        posterior.smoothed_mass / dx, posterior.grid, axis=1, initial=0
    )
    cdf /= cdf[:, -1:]
    return np.array(
        [
            np.interp(hi, posterior.grid, row) - np.interp(lo, posterior.grid, row)
            for row, lo, hi in zip(cdf, lower, upper, strict=True)
        ]
    )


def gaussian_joint_posterior(
    initial_mean: NDArray[np.float64],
    initial_covariance: NDArray[np.float64],
    process_covariance: NDArray[np.float64],
    design: NDArray[np.float64],
    observations: NDArray[np.float64],
    observation_variance: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> GaussianPosterior:
    """Condition the full joint random-walk Gaussian using dense linear algebra.

    Sigma[t,s] = P0 + min(t,s)*Q constructs the prior without a filter/RTS
    recurrence. Each valid scalar observation is H_t w_t + N(0,R_t).
    """
    n_time, rank = design.shape
    covariance = np.kron(np.ones((n_time, n_time)), initial_covariance)
    covariance += np.kron(
        np.minimum.outer(np.arange(n_time), np.arange(n_time)), process_covariance
    )
    mean = np.tile(initial_mean, n_time)
    times = np.flatnonzero(valid)
    observation_matrix = np.zeros((len(times), n_time * rank))
    for row, t in enumerate(times):
        observation_matrix[row, t * rank : (t + 1) * rank] = design[t]
    cross = covariance @ observation_matrix.T
    obs_cov = observation_matrix @ cross + np.diag(observation_variance[valid])
    gain = np.linalg.solve(obs_cov, cross.T).T
    mean += gain @ (observations[valid] - observation_matrix @ mean)
    covariance -= gain @ cross.T
    blocks = [
        covariance[t * rank : (t + 1) * rank, t * rank : (t + 1) * rank]
        for t in range(n_time)
    ]
    lag = [
        covariance[t * rank : (t + 1) * rank, (t + 1) * rank : (t + 2) * rank]
        for t in range(n_time - 1)
    ]
    return GaussianPosterior(
        mean.reshape(n_time, rank),
        np.asarray(blocks),
        np.asarray(lag).reshape(n_time - 1, rank, rank),
        covariance,
    )


def gaussian_filtered_moments(
    initial_mean: NDArray[np.float64],
    initial_covariance: NDArray[np.float64],
    process_covariance: NDArray[np.float64],
    design: NDArray[np.float64],
    observations: NDArray[np.float64],
    observation_variance: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Filter references by conditioning each joint prefix, never Kalman recursion."""
    posteriors = [
        gaussian_joint_posterior(
            initial_mean,
            initial_covariance,
            process_covariance,
            design[:t],
            observations[:t],
            observation_variance[:t],
            valid[:t],
        )
        for t in range(1, len(design) + 1)
    ]
    return np.array([p.mean[-1] for p in posteriors]), np.array(
        [p.covariance[-1] for p in posteriors]
    )


def expected_gaussian_prior_nll(
    posterior: GaussianPosterior,
    initial_mean: NDArray[np.float64],
    initial_covariance: NDArray[np.float64],
    process_covariance: NDArray[np.float64],
) -> float:
    """Expected negative log prior from full dense moments (constants omitted).

    This reference evaluates Gaussian log densities via determinants/solves, not
    the production gamma/beta or diagonal-averaging M-step formulas.
    """
    n_time = len(posterior.mean)
    residual = posterior.mean[0] - initial_mean
    initial_second = posterior.covariance[0] + np.outer(residual, residual)
    value = np.linalg.slogdet(initial_covariance)[1] + np.trace(
        np.linalg.solve(initial_covariance, initial_second)
    )
    for t in range(1, n_time):
        delta = posterior.mean[t] - posterior.mean[t - 1]
        second = (
            posterior.covariance[t]
            + posterior.covariance[t - 1]
            - posterior.lag_covariance[t - 1]
            - posterior.lag_covariance[t - 1].T
            + np.outer(delta, delta)
        )
        value += np.linalg.slogdet(process_covariance)[1] + np.trace(
            np.linalg.solve(process_covariance, second)
        )
    return 0.5 * float(value)
