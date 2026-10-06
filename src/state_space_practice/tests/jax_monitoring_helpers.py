"""Scoped compilation listeners across the supported JAX versions."""

import contextlib
from collections.abc import Callable, Iterator

import jax

_legacy_listeners: list[Callable[..., None]] = []
_legacy_dispatch_registered = False


def _dispatch_duration(event: str, duration: float, **kwargs: object) -> None:
    for listener in tuple(_legacy_listeners):
        listener(event, duration, **kwargs)


@contextlib.contextmanager
def listen_to_jax_durations(listener: Callable[..., None]) -> Iterator[None]:
    """Attach a listener for this context without clearing other listeners.

    JAX 0.6.2 cannot unregister individual listeners. Keep one dispatcher for
    the test session and remove each scoped callback from it on exit, so no
    counters accumulate and nested contexts keep their listeners intact.
    """
    global _legacy_dispatch_registered
    unregister = getattr(jax.monitoring, "unregister_event_duration_listener", None)
    if unregister is not None:
        jax.monitoring.register_event_duration_secs_listener(listener)
        try:
            yield
        finally:
            unregister(listener)
    else:
        if not _legacy_dispatch_registered:
            jax.monitoring.register_event_duration_secs_listener(_dispatch_duration)
            _legacy_dispatch_registered = True
        _legacy_listeners.append(listener)
        try:
            yield
        finally:
            _legacy_listeners.remove(listener)
