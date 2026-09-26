"""Warning and exception classes raised by ``state_space_practice``.

Filter all library warnings at once with::

    import warnings
    from state_space_practice import StateSpaceWarning

    warnings.simplefilter("ignore", StateSpaceWarning)
"""

__all__ = ["NotFittedError", "StateSpaceWarning"]


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
