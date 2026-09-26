"""Shared EM driver: the E-step / convergence / rollback / M-step loop.

Every EM-fitted model in the library runs the same skeleton -- E-step,
non-finite check, convergence and log-likelihood-decrease checks, snapshot,
M-step, and a final E-step that re-syncs the posteriors with the last M-step
-- but the models had each grown their own copy of it, with slightly different
rollback policies.  :func:`run_em` is that skeleton written once; the policy
differences are explicit keyword options so each model's ``fit`` states what
it does instead of re-implementing the loop.

The driver never touches model attributes directly.  It talks to the model
through four callables (``e_step``, ``m_step``, ``snapshot``, ``restore``) so
it can be used by models that store their posteriors under different names.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from state_space_practice.utils import check_converged

_logger = logging.getLogger(__name__)

NonFiniteFirstPolicy = Literal["break", "raise", "clear"]


@dataclass
class EMResult:
    """Outcome of :func:`run_em`.

    Attributes
    ----------
    log_likelihoods : list[float]
        Marginal log-likelihood of every *accepted* E-step, in order.  A
        non-finite or rolled-back E-step is never left in the history.
    converged : bool
        True when the relative change fell below ``tol`` before ``max_iter``.
    reached_max_iter : bool
        True when the loop ran all ``max_iter`` iterations without stopping.
    """

    log_likelihoods: list[float] = field(default_factory=list)
    converged: bool = False
    reached_max_iter: bool = False


def run_em(
    e_step: Callable[[], float],
    m_step: Callable[[], None],
    snapshot: Callable[[], Any],
    restore: Callable[[Any], None],
    *,
    max_iter: int,
    tol: float,
    decrease_tol: float | None = None,
    on_first_nonfinite: NonFiniteFirstPolicy = "break",
    clear_state: Callable[[], None] | None = None,
    stop_on_decrease: bool = True,
    require_increase_to_converge: bool = False,
    refresh_after_restore: bool = False,
    track_best: bool = False,
    m_step_on_convergence: bool = False,
    logger: logging.Logger | None = None,
    on_iteration: Callable[[int, float, float], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> EMResult:
    """Run EM until convergence, a rejected step, or ``max_iter``.

    Each iteration calls ``e_step`` (which must install the posteriors on the
    model and return the marginal log-likelihood), checks the result, takes a
    ``snapshot`` of the accepted (parameters, posteriors) pair, then calls
    ``m_step``.  A later E-step that is non-finite or decreases the
    log-likelihood ``restore``\\ s that snapshot, so the model is always left
    with a parameter set and posteriors that belong together.

    Parameters
    ----------
    e_step, m_step, snapshot, restore : callable
        The model hooks.  ``restore`` receives whatever ``snapshot`` returned.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Relative log-likelihood change below which EM has converged (see
        :func:`state_space_practice.utils.check_converged`).
    decrease_tol : float or None
        Relative decrease that counts as a rejected step.  Defaults to ``tol``.
        Approximate E-steps (Laplace, GPB) can set a more lenient value so
        sub-tolerance decreases are treated as noise instead of divergence.
    on_first_nonfinite : {"break", "raise", "clear"}
        What to do when an E-step is non-finite before any step was accepted:
        stop quietly, raise ``ValueError``, or call ``clear_state`` (drop the
        NaN posteriors) and stop.  With an accepted step available the driver
        always rolls back to it and stops.
    clear_state : callable or None
        Used by the ``"clear"`` policy.
    stop_on_decrease : bool
        Roll back and stop on a decrease (default) or warn and continue.
    require_increase_to_converge : bool
        Only declare convergence on a non-decreasing step (used together
        with ``stop_on_decrease=False``).
    refresh_after_restore : bool
        Re-run ``e_step`` after every ``restore`` so posteriors stored by the
        E-step are recomputed under the restored parameters (for models whose
        snapshot holds parameters only).
    track_best : bool
        Remember the best accepted state and restore it at the end if the
        final log-likelihood is below it (for approximate EM that may drift).
    m_step_on_convergence : bool
        On convergence, run one more M-step plus a synchronising E-step,
        rolling back if that final E-step does not improve.
    logger : logging.Logger or None
        Logger for info messages; defaults to this module's logger.
    on_iteration : callable or None
        ``on_iteration(iteration, log_likelihood, change)`` progress hook,
        called after every E-step.  Defaults to a ``logger.info`` line.
    warn : callable or None
        Sink for warning messages; defaults to ``logger.warning``.  Models
        that also print progress pass a hook that does both.

    Returns
    -------
    EMResult
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}.")
    if on_first_nonfinite not in ("break", "raise", "clear"):
        raise ValueError(
            "on_first_nonfinite must be 'break', 'raise' or 'clear', got "
            f"{on_first_nonfinite!r}."
        )
    if (on_first_nonfinite == "clear") != (clear_state is not None):
        raise ValueError(
            "clear_state must be given exactly when on_first_nonfinite='clear' "
            f"(got on_first_nonfinite={on_first_nonfinite!r}, clear_state="
            f"{'None' if clear_state is None else 'a callable'})."
        )

    log = logger if logger is not None else _logger
    emit_warning = warn if warn is not None else log.warning
    decrease_tolerance = tol if decrease_tol is None else decrease_tol

    log_likelihoods: list[float] = []
    last_accepted: Any = None
    best_ll = -math.inf
    best_state: Any = None
    converged = False
    reached_max_iter = True

    def _restore_last_accepted() -> None:
        restore(last_accepted)
        if refresh_after_restore:
            refresh_ll = float(e_step())
            if not np.isfinite(refresh_ll):
                emit_warning(
                    f"Re-running the E-step after restoring the previous "
                    f"parameters gave a non-finite log-likelihood ({refresh_ll}); "
                    f"the stored posteriors are not usable."
                )

    def _report(iteration: int, ll: float) -> None:
        change = ll - log_likelihoods[-2] if iteration > 0 else math.nan
        if on_iteration is not None:
            on_iteration(iteration, ll, change)
        else:
            log.info(
                f"Iteration {iteration + 1}/{max_iter}\t"
                f"Log-Likelihood: {ll:.4f}\tChange: {change:.4f}"
            )

    for iteration in range(max_iter):
        current_ll = float(e_step())
        log_likelihoods.append(current_ll)
        _report(iteration, current_ll)

        if not np.isfinite(current_ll):
            bad_ll = log_likelihoods.pop()
            if last_accepted is None:
                if on_first_nonfinite == "raise":
                    raise ValueError(
                        f"Non-finite log-likelihood at iteration {iteration + 1}: "
                        f"{bad_ll}. This may indicate numerical instability."
                    )
                if clear_state is not None:
                    clear_state()
                emit_warning(
                    f"Non-finite log-likelihood ({bad_ll}) at iteration "
                    f"{iteration + 1} with no usable previous state; stopping EM."
                )
            else:
                _restore_last_accepted()
                emit_warning(
                    f"Non-finite log-likelihood ({bad_ll}) at iteration "
                    f"{iteration + 1}; rolling back to the previous E-step and "
                    f"stopping EM."
                )
            reached_max_iter = False
            break

        state = snapshot()
        if track_best and current_ll > best_ll:
            best_ll, best_state = current_ll, state

        if iteration > 0:
            previous_ll = log_likelihoods[-2]
            is_converged, _ = check_converged(current_ll, previous_ll, tol)
            _, is_increasing = check_converged(
                current_ll, previous_ll, decrease_tolerance
            )
            if not is_increasing:
                if stop_on_decrease:
                    bad_ll = log_likelihoods.pop()
                    _restore_last_accepted()
                    emit_warning(
                        f"LL decreased: {log_likelihoods[-1]:.4f} -> {bad_ll:.4f}; "
                        f"rolling back to the previous E-step and stopping EM."
                    )
                    reached_max_iter = False
                    break
                emit_warning(
                    f"Log-likelihood decreased at iteration {iteration + 1}: "
                    f"{previous_ll:.4f} -> {current_ll:.4f}"
                )
            if is_converged and (is_increasing or not require_increase_to_converge):
                if m_step_on_convergence:
                    last_accepted = state
                    m_step()
                    final_ll = float(e_step())
                    _, final_is_increasing = check_converged(
                        final_ll, log_likelihoods[-1], decrease_tolerance
                    )
                    if final_is_increasing and math.isfinite(final_ll):
                        log_likelihoods.append(final_ll)
                    else:
                        _restore_last_accepted()
                        emit_warning(
                            f"Post-convergence M-step gave log-likelihood "
                            f"{final_ll:.4f} (converged at "
                            f"{log_likelihoods[-1]:.4f}); rolling back to the "
                            f"converged E-step."
                        )
                log.info(f"Converged after {iteration + 1} iterations.")
                converged = True
                reached_max_iter = False
                break

        last_accepted = state
        m_step()

    if reached_max_iter:
        emit_warning("Reached maximum iterations without converging.")
        if log_likelihoods:
            # The last M-step ran without a following E-step: sync the stored
            # posteriors with the current parameters, rolling back if that
            # reveals the M-step made things worse.
            final_ll = float(e_step())
            if not np.isfinite(final_ll):
                if last_accepted is not None:
                    _restore_last_accepted()
                emit_warning(
                    "Final E-step produced non-finite log-likelihood; rolling "
                    "back to the previous E-step."
                )
            else:
                _, final_is_increasing = check_converged(
                    final_ll, log_likelihoods[-1], decrease_tolerance
                )
                if final_is_increasing:
                    log_likelihoods.append(final_ll)
                    if track_best and final_ll > best_ll:
                        best_ll, best_state = final_ll, snapshot()
                elif last_accepted is not None:
                    _restore_last_accepted()
                    emit_warning(
                        f"Final E-step decreased LL: {log_likelihoods[-1]:.4f} -> "
                        f"{final_ll:.4f}; rolling back to the previous E-step."
                    )
                else:
                    emit_warning(
                        "Final E-step decreased LL but no accepted state was "
                        "available for rollback."
                    )

    if (
        track_best
        and best_state is not None
        and log_likelihoods
        and log_likelihoods[-1] < best_ll
    ):
        log.info(
            f"Restoring best params from LL={best_ll:.4f} "
            f"(final was {log_likelihoods[-1]:.4f})"
        )
        restore(best_state)
        restored_ll = float(e_step())
        if not np.isfinite(restored_ll):
            emit_warning(
                f"Re-running the E-step under the restored best parameters gave "
                f"a non-finite log-likelihood ({restored_ll}); the stored "
                f"posteriors are not usable."
            )
        elif restored_ll != log_likelihoods[-1]:
            log_likelihoods.append(restored_ll)

    return EMResult(log_likelihoods, converged, reached_max_iter)
