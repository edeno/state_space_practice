"""Typed attributes for state that exists only after a model is fitted.

A model's fitted outputs (smoothed means, log likelihood, ...) do not exist
until ``fit`` runs. Declaring them ``X | None`` makes every reader narrow
before use; declaring them ``X`` without a value lets them raise a bare
``AttributeError``. :class:`FittedAttribute` gives them type ``X`` and makes
reading one before it is set raise :class:`UnfittedAttributeError`, a
:class:`~state_space_practice.exceptions.NotFittedError` that is also an
``AttributeError`` so ``hasattr`` / ``getattr(obj, name, default)`` keep
working.

The value lives in the instance ``__dict__`` under the attribute's own name,
so ``copy``, ``deepcopy``, pickling and ``vars(model)`` see plain attributes.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, overload

from state_space_practice.exceptions import NotFittedError

__all__ = ["FittedAttribute", "UnfittedAttributeError", "is_set"]

_T = TypeVar("_T")


class UnfittedAttributeError(NotFittedError, AttributeError):
    """Raised when a fitted attribute is read before the model sets it."""


class FittedAttribute(Generic[_T]):
    """Class-level declaration of an attribute that fitting sets.

    Declare on the class, e.g. ``smoother_mean: FittedAttribute[Array] =
    FittedAttribute()``. Reading ``model.smoother_mean`` returns an ``Array``
    or raises :class:`UnfittedAttributeError`; ``del model.smoother_mean``
    returns it to the unset state. Use :func:`is_set` to test without raising.
    """

    _name: str

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(
        self, obj: None, objtype: type | None = None
    ) -> FittedAttribute[_T]: ...

    @overload
    def __get__(self, obj: object, objtype: type | None = None) -> _T: ...

    def __get__(
        self, obj: object | None, objtype: type | None = None
    ) -> FittedAttribute[_T] | _T:
        if obj is None:
            return self
        try:
            value: _T = obj.__dict__[self._name]
        except KeyError:
            raise UnfittedAttributeError(
                f"{type(obj).__name__}.{self._name} is not available until the "
                "model is fitted. Call fit() or fit_sgd() first."
            ) from None
        return value

    def __set__(self, obj: object, value: _T) -> None:
        obj.__dict__[self._name] = value

    def __delete__(self, obj: object) -> None:
        obj.__dict__.pop(self._name, None)


def is_set(obj: Any, name: str) -> bool:
    """Whether the :class:`FittedAttribute` ``name`` has been set on ``obj``."""
    return name in vars(obj)
