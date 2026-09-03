"""Tests for source-age and fail-closed collision-node semantics."""

from types import SimpleNamespace

import pytest

from vr_rm75_teleop import collision_backend_node
from vr_rm75_teleop.collision_backend_node import (
    CollisionBackendNode,
    joint_state_input_age_s,
)


class RecordingPublisher:
    """Capture messages without constructing a ROS node."""

    def __init__(self):
        """Initialize an empty publication record."""
        self.messages = []

    def publish(self, message):
        """Record one would-be ROS publication."""
        self.messages.append(message)


class RecordingBackend:
    """Record source timestamps and return a minimal valid snapshot."""

    ready = True

    def __init__(self):
        """Initialize an empty measurement timestamp record."""
        self.measured = []

    def evaluate(self, _positions, measured_monotonic=None):
        """Return a complete synthetic RM75-only snapshot."""
        self.measured.append(measured_monotonic)
        return SimpleNamespace(
            distances_m=(0.2, 0.3, 0.4),
            reason="valid test snapshot",
        )


def make_node(input_age_s):
    """Build just the state needed to exercise the unbound callback."""
    now_ns = 2_000_000_000
    stamp_ns = now_ns - round(input_age_s * 1e9)
    stamp = SimpleNamespace(
        sec=stamp_ns // 1_000_000_000,
        nanosec=stamp_ns % 1_000_000_000,
    )
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp),
        name=["joint"],
        position=[0.0],
    )
    node = SimpleNamespace(
        state_timeout_s=0.10,
        _backend=RecordingBackend(),
        _configuration_joint_positions={},
        _last_state_source_monotonic=None,
        _last_input_state_age_s=None,
        _last_compute_time_s=None,
        _last_output_age_s=None,
        _last_reason="",
        _last_snapshot=object(),
        distance_publisher=RecordingPublisher(),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=now_ns)
        ),
    )
    node.diagnostics_times = []
    node.publish_diagnostics = node.diagnostics_times.append
    return node, message


def test_joint_state_age_comes_from_header_not_callback_arrival():
    """Measure backlog from the producer stamp and reject invalid stamps."""
    stamp = SimpleNamespace(sec=1, nanosec=950_000_000)
    assert joint_state_input_age_s(stamp, 2_000_000_000) == pytest.approx(
        0.05
    )
    with pytest.raises(ValueError, match="missing"):
        joint_state_input_age_s(SimpleNamespace(sec=0, nanosec=0), 10)
    with pytest.raises(ValueError, match="future"):
        joint_state_input_age_s(stamp, 1_900_000_000)


def test_stale_queued_input_fails_closed_before_compute(monkeypatch):
    """Do not spend FCL time or publish for an already stale source sample."""
    node, message = make_node(input_age_s=0.11)
    times = iter((10.0, 10.001))
    monkeypatch.setattr(
        collision_backend_node.time, "perf_counter", lambda: next(times)
    )

    CollisionBackendNode.state_callback(node, message)

    assert node._backend.measured == []
    assert node._last_snapshot is None
    assert node.distance_publisher.messages == []
    assert "stale before compute" in node._last_reason
    assert node._last_input_state_age_s == pytest.approx(0.11)


def test_result_that_becomes_stale_during_compute_is_not_published(
    monkeypatch,
):
    """Discard a result whose source exceeds the timeout during its solve."""
    node, message = make_node(input_age_s=0.03)
    times = iter((10.0, 10.08))
    monkeypatch.setattr(
        collision_backend_node.time, "perf_counter", lambda: next(times)
    )

    CollisionBackendNode.state_callback(node, message)

    assert node._backend.measured == pytest.approx([9.97])
    assert node._last_snapshot is None
    assert node.distance_publisher.messages == []
    assert node._last_compute_time_s == pytest.approx(0.08)
    assert node._last_output_age_s == pytest.approx(0.11)
    assert "stale before publish" in node._last_reason


def test_fresh_result_keeps_source_measurement_time(monkeypatch):
    """Publish a timely result while retaining its original source time."""
    node, message = make_node(input_age_s=0.03)
    times = iter((10.0, 10.02))
    monkeypatch.setattr(
        collision_backend_node.time, "perf_counter", lambda: next(times)
    )

    CollisionBackendNode.state_callback(node, message)

    assert node._backend.measured == pytest.approx([9.97])
    assert node._last_snapshot is not None
    assert len(node.distance_publisher.messages) == 1
    assert list(node.distance_publisher.messages[0].data) == pytest.approx(
        [0.2, 0.3, 0.4]
    )
    assert node._last_input_state_age_s == pytest.approx(0.03)
    assert node._last_compute_time_s == pytest.approx(0.02)
    assert node._last_output_age_s == pytest.approx(0.05)
