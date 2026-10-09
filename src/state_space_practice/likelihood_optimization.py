"""Deterministic profile and bounded-gradient optimization of model likelihoods.

These routines operate on caller-supplied objectives. They do not change the
inference approximation or declare a finite profile to be a global proof.
"""

import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize, minimize_scalar

from state_space_practice.exceptions import StateSpaceWarning
from state_space_practice.utils import validate_int, validate_scalar


class ScaleProfile(NamedTuple):
    """Minimum negative log likelihood, including the exact zero-scale candidate.

    Attributes
    ----------
    scale, loss : float
        Selected scale and its loss.
    scales, losses : NDArray, shape (n_evaluated,)
        Every evaluated scale (zero, the log grid, then refinements) and loss.
    at_upper_bound : bool
        Whether the selected scale is the upper grid bound.
    flat : bool
        Whether all finite losses agree within the objective tolerance.
    n_nonfinite : int
        Number of evaluations whose loss was non-finite and was skipped.
    """

    scale: float
    loss: float
    scales: NDArray[np.float64]
    losses: NDArray[np.float64]
    at_upper_bound: bool
    flat: bool
    n_nonfinite: int


class LikelihoodOptimum(NamedTuple):
    """Result with explicit projected-gradient and finite-iteration diagnostics.

    ``gradient_norm`` is the max-abs projected gradient. ``loss_history`` holds
    the starting loss and the loss after each accepted iteration (length
    ``n_iter + 1``); line-search trials are not included.
    """

    parameters: NDArray[np.float64]
    loss: float
    gradient_norm: float
    converged: bool
    n_iter: int
    loss_history: list[float]
    message: str


def warn_if_unconverged(result: LikelihoodOptimum, name: str) -> None:
    """Emit a ``StateSpaceWarning`` when ``result`` did not converge."""
    if not result.converged:
        warnings.warn(
            f"{name}: L-BFGS did not converge ({result.message}); max-abs "
            f"projected gradient {result.gradient_norm:.3g}. Increase max_iter "
            "or inspect optimizer_result_.",
            StateSpaceWarning,
            stacklevel=3,
        )


def profile_scale(
    objective: Callable[[float], float],
    *,
    bounds: tuple[float, float] = (1e-8, 10.0),
    n_grid: int = 49,
    objective_tolerance: float = 1e-6,
) -> ScaleProfile:
    """Search zero and a log grid, refining every strict sampled local minimum.

    ``objective`` receives the scale itself, including exactly zero. Bounds
    apply to the positive grid. Zero is preferred when its loss is within
    ``objective_tolerance`` of the best value. Non-finite points are unusable;
    an entirely non-finite search raises. Endpoint/flat diagnostics are retained.
    """
    lo = validate_scalar(bounds[0], "lower scale bound", positive=True)
    hi = validate_scalar(bounds[1], "upper scale bound", positive=True)
    if not lo < hi:
        raise ValueError("Scale bounds must be increasing.")
    n_grid = validate_int(n_grid, "n_grid", positive=True)
    if n_grid < 3:
        raise ValueError("n_grid must be at least three.")
    tolerance = validate_scalar(
        objective_tolerance, "objective_tolerance", nonnegative=True
    )
    log_grid = np.linspace(np.log(lo), np.log(hi), n_grid)
    points = [0.0, *np.exp(log_grid).tolist()]
    points[1], points[-1] = lo, hi
    values = [float(objective(q)) for q in points]
    raw_grid_values = np.asarray(values[1:])
    grid_values = np.where(np.isfinite(raw_grid_values), raw_grid_values, np.inf)
    for i in range(n_grid):
        left = grid_values[i - 1] if i else np.inf
        right = grid_values[i + 1] if i + 1 < n_grid else np.inf
        if (
            np.isfinite(grid_values[i])
            and grid_values[i] <= min(left, right)
            and (grid_values[i] < left or grid_values[i] < right)
        ):
            a, b = log_grid[max(i - 1, 0)], log_grid[min(i + 1, n_grid - 1)]

            def refinement_objective(
                x: float, sampled_loss: float = float(grid_values[i])
            ) -> float:
                loss = float(objective(float(np.exp(x))))
                if not np.isfinite(loss):
                    return float(np.pi / 2)
                # A monotone, finite objective keeps unusable trials worse than
                # the sampled minimum without inf arithmetic inside SciPy.
                return float(np.arctan(loss - sampled_loss))

            result = minimize_scalar(
                refinement_objective,
                bounds=(float(a), float(b)),
                method="bounded",
                options={"xatol": 1e-9},
            )
            points.append(float(np.exp(result.x)))
            values.append(float(objective(points[-1])))
    scales, losses = np.asarray(points), np.asarray(values)
    finite = np.isfinite(losses)
    if not finite.any():
        raise ValueError("No finite likelihood in the scale profile.")
    index = int(np.argmin(np.where(finite, losses, np.inf)))
    if np.isfinite(losses[0]) and losses[0] <= losses[index] + tolerance:
        index = 0
    return ScaleProfile(
        float(scales[index]),
        float(losses[index]),
        scales,
        losses,
        bool(scales[index] == hi),
        bool(np.ptp(losses[finite]) <= tolerance),
        int(np.sum(~finite)),
    )


def minimize_likelihood(
    value_and_gradient: Callable[
        [NDArray[np.float64]], tuple[float, NDArray[np.float64]]
    ],
    initial: NDArray[np.float64],
    *,
    bounds: list[tuple[float | None, float | None]] | None = None,
    max_iter: int = 500,
    gradient_tolerance: float = 1e-5,
) -> LikelihoodOptimum:
    """Minimize with L-BFGS-B and check the final projected gradient.

    Inputs are optimizer coordinates chosen by the caller (for example the
    package's positive/unconstrained parameter transforms). The result does not
    label a tiny relative likelihood change alone as convergence.
    """
    max_iter = validate_int(max_iter, "max_iter", positive=True)
    tolerance = validate_scalar(gradient_tolerance, "gradient_tolerance", positive=True)
    start = np.asarray(initial, dtype=float)
    if start.ndim != 1 or not start.size or not np.isfinite(start).all():
        raise ValueError(
            "Initial optimizer parameters must be a finite nonempty vector."
        )
    history: list[float] = []
    last: list[tuple[NDArray[np.float64], float, NDArray[np.float64]]] = []

    def evaluate(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        loss, gradient = value_and_gradient(x)
        last[:] = [(np.array(x), float(loss), np.asarray(gradient, dtype=float))]
        if not history:
            history.append(float(loss))
        return float(loss), np.asarray(gradient, dtype=float)

    def record(intermediate_result: OptimizeResult) -> None:
        history.append(float(intermediate_result.fun))

    result = minimize(
        evaluate,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=record,
        options={"maxiter": max_iter, "gtol": tolerance, "ftol": 1e-14, "maxls": 40},
    )
    x = np.asarray(result.x)
    if last and np.array_equal(last[0][0], x):
        # The optimizer usually stops at its latest evaluation; reuse it.
        _, loss, gradient = last[0]
    else:
        loss, gradient = value_and_gradient(x)
    projected = np.asarray(gradient).copy()
    if bounds is not None:
        lo, hi = (
            np.array([np.nan if b[k] is None else b[k] for b in bounds], dtype=float)
            for k in (0, 1)
        )
        projected[((x <= lo) & (projected > 0)) | ((x >= hi) & (projected < 0))] = 0
    norm = float(np.max(np.abs(projected)))
    finite = bool(np.isfinite(loss) and np.isfinite(gradient).all())
    return LikelihoodOptimum(
        np.asarray(result.x),
        float(loss),
        norm,
        bool(finite and norm <= tolerance),
        int(result.nit),
        history,
        str(result.message),
    )
