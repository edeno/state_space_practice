"""Scoped listeners must stop recording on exit and preserve nested listeners."""

import jax
import pytest

from state_space_practice.tests.jax_monitoring_helpers import listen_to_jax_durations


@pytest.mark.parametrize("legacy", [False, True])
def test_nested_duration_listeners_remain_scoped(monkeypatch, legacy):
    if legacy:
        monkeypatch.delattr(
            jax.monitoring, "unregister_event_duration_listener", raising=False
        )
    outer, inner = [], []
    event = "/state_space_practice/test/scoped_listener"

    def record_outer(name, duration, **kwargs):
        if name == event:
            outer.append(duration)

    def record_inner(name, duration, **kwargs):
        if name == event:
            inner.append(duration)

    with listen_to_jax_durations(record_outer):
        jax.monitoring.record_event_duration_secs(event, 1.0)
        with listen_to_jax_durations(record_inner):
            jax.monitoring.record_event_duration_secs(event, 2.0)
        jax.monitoring.record_event_duration_secs(event, 3.0)
    jax.monitoring.record_event_duration_secs(event, 4.0)

    assert outer == [1.0, 2.0, 3.0]
    assert inner == [2.0]
