"""profiler 内存字段解析单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

from chameleon.profile.counters import _profiler_memory_from_events


def test_profiler_memory_from_events_uses_device_fields() -> None:
    events = [
        SimpleNamespace(cpu_memory_usage=100, self_device_memory_usage=200),
        SimpleNamespace(cpu_memory_usage=50, device_memory_usage=300),
    ]
    cpu, device = _profiler_memory_from_events(events)
    assert cpu == 150
    assert device == 500
