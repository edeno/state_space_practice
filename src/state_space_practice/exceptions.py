"""Warning and exception classes raised by ``state_space_practice``.

Filter all library warnings at once with::

    import warnings
    from state_space_practice import StateSpaceWarning

    warnings.simplefilter("ignore", StateSpaceWarning)
"""

__all__ = ["NonFiniteLikelihoodError", "NotFittedError", "StateSpaceWarning"]


class StateSpaceWarning(UserWarning):
    """Base class for warnings emitted by ``state_space_practice``.

    Subclasses ``UserWarning`` so existing ``pytest.warns(UserWarning)`` and
    ``warnings.simplefilter(..., UserWarning)`` filters keep matching.
    """


class NotFittedError(RuntimeError):
    """Raised when a model method needs fitted or initialized state.

    Call ``fit`` / ``fit_sgd`` (or the model's initialization method) before
    using the method that raised.
    """


class NonFiniteLikelihoodError(ValueError):
    """Raised when a fit's starting parameters give a non-finite log-likelihood.

    Raised by ``fit_sgd`` (non-finite log-likelihood at the fitted parameters)
    and by the EM ``fit`` of the choice / belief models and the switching
    point-process models (non-finite first E-step). The model's fit outputs
    are cleared first, so it reads as unfitted. The oscillator, point-process,
    place-field and Smith learning models' EM ``fit`` instead clears its
    outputs, logs a warning and returns an empty history.

    A ``ValueError`` subclass, so ``except ValueError`` handlers keep
    matching; catch this class to tell a numerically failed fit apart from
    invalid input.
    """
