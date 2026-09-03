"""Offline tests for Quest reader source freshness."""

import math
import threading

import numpy as np
import pytest

import meta_quest_teleop.reader as reader_module
from meta_quest_teleop.reader import MetaQuestReader


def make_reader(last_sample):
    """Create only the state needed by freshness methods, without ADB."""
    reader = MetaQuestReader.__new__(MetaQuestReader)
    reader._lock = threading.Lock()
    reader._last_sample_monotonic = last_sample
    return reader


def make_stream_reader():
    """Create stream-processing state without connecting to ADB."""
    reader = make_reader(None)
    reader.tag = "wE9ryARX"
    reader.axis_mask = None
    reader.last_transforms = {}
    reader.last_buttons = {}
    reader._latest_transforms = {}
    reader._latest_buttons = {}
    reader._prev_button_states = {}
    event_names = (
        "button_b_pressed",
        "button_a_pressed",
        "button_x_pressed",
        "button_y_pressed",
        "button_rj_pressed",
        "button_lj_pressed",
    )
    reader._callbacks = {name: [] for name in event_names}
    reader._callbacks_locks = {
        name: threading.Lock() for name in event_names
    }
    reader.lines_received = 0
    reader.lines_committed = 0
    reader.backlog_lines_dropped = 0
    reader.last_batch_line_count = 0
    return reader


def sample_line(
    x,
    *,
    left_grip=0.0,
    right_grip=0.0,
    left_tracking=1,
    right_tracking=1,
):
    """Build one complete world-space APK logcat line."""
    left = np.eye(4)
    right = np.eye(4)
    left[0, 3] = x
    right[0, 3] = -x
    left_values = " ".join(str(value) for value in left.reshape(-1))
    right_values = " ".join(str(value) for value in right.reshape(-1))
    payload = (
        f"lg:{left_values}|lc:{left_tracking}|"
        f"rg:{right_values}|rc:{right_tracking}&"
        f"L,R,leftGrip {left_grip},rightGrip {right_grip}"
    )
    return f"09-03 I/wE9ryARX: {payload}\n".encode()


def test_missing_sample_is_stale():
    """Treat an uninitialized logcat stream as unavailable."""
    reader = make_reader(None)
    assert math.isinf(reader.get_data_age_s(10.0))
    assert not reader.data_is_fresh(0.2, 10.0)


def test_age_and_timeout_use_monotonic_receive_time():
    """Expire cached controller inputs after the configured deadline."""
    reader = make_reader(20.0)
    assert reader.get_data_age_s(20.1) == pytest.approx(0.1)
    assert reader.data_is_fresh(0.2, 20.2)
    assert not reader.data_is_fresh(0.2, 20.201)


def test_backwards_clock_jump_is_clamped():
    """Avoid a negative age if a supplied test clock moves backwards."""
    reader = make_reader(30.0)
    assert reader.get_data_age_s(29.0) == 0.0


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.nan, math.inf])
def test_invalid_timeout_is_rejected(timeout):
    """Require a finite positive freshness deadline."""
    reader = make_reader(1.0)
    with pytest.raises(ValueError):
        reader.data_is_fresh(timeout, 1.0)


def test_one_wakeup_commits_only_latest_complete_sample():
    """Collapse four complete pose lines into one latest-state commit."""
    reader = make_stream_reader()
    chunk = b"".join(sample_line(x) for x in (0.0, 0.1, 0.2, 0.3))

    trailing = reader._consume_logcat_bytes(b"", chunk)

    assert trailing == b""
    assert reader.get_hand_controller_transform_openxr("left")[0, 3] == 0.3
    assert reader.get_stream_diagnostics() == {
        "lines_received": 4,
        "lines_committed": 1,
        "backlog_lines_dropped": 3,
        "last_batch_line_count": 4,
    }


def test_pipe_drain_reads_until_no_bytes_are_immediately_available(monkeypatch):
    """Drain every currently queued chunk after the initial select wake-up."""
    chunks = iter((b"pose0\n", b"pose1\n", b"pose2\n"))
    readiness = iter((([7], [], []), ([7], [], []), ([], [], [])))
    monkeypatch.setattr(reader_module.os, "read", lambda _fd, _size: next(chunks))
    monkeypatch.setattr(
        reader_module.select,
        "select",
        lambda _read, _write, _error, _timeout: next(readiness),
    )

    drained, pipe_closed = MetaQuestReader._drain_logcat_pipe(7)

    assert drained == b"pose0\npose1\npose2\n"
    assert not pipe_closed


def test_latest_valid_sample_wins_when_final_line_is_malformed():
    """Ignore a bad final line instead of replacing the last valid state."""
    reader = make_stream_reader()
    malformed = b"09-03 I/wE9ryARX: bad&leftGrip not-a-number\n"

    reader._consume_logcat_bytes(b"", sample_line(0.4) + malformed)

    assert reader.get_hand_controller_transform_openxr("left")[0, 3] == 0.4
    assert reader.lines_committed == 1
    assert reader.backlog_lines_dropped == 0


def test_backlog_motion_is_not_replayed_to_consumer():
    """Expose the final 0.20 m state directly, without intermediate poses."""
    reader = make_stream_reader()
    committed_positions = []
    reader._handle_button_events = lambda _buttons: committed_positions.append(
        reader.get_hand_controller_transform_openxr("left")[0, 3]
    )
    chunk = b"".join(sample_line(x) for x in (0.02, 0.05, 0.10, 0.20))

    reader._consume_logcat_bytes(b"", chunk)

    assert committed_positions == [0.20]
    assert reader.get_hand_controller_transform_openxr("left")[0, 3] == 0.20


def test_partial_line_is_preserved_and_joined_across_reads():
    """Do not parse or corrupt a trailing line until its newline arrives."""
    reader = make_stream_reader()
    line = sample_line(0.125, left_grip=0.4)
    split_at = len(line) - 17

    pending = reader._consume_logcat_bytes(b"", line[:split_at])

    assert pending == line[:split_at]
    assert reader.lines_committed == 0
    assert reader._last_sample_monotonic is None

    pending = reader._consume_logcat_bytes(pending, line[split_at:])

    assert pending == b""
    assert reader.lines_committed == 1
    assert reader.get_hand_controller_transform_openxr("left")[0, 3] == 0.125
    assert reader.get_grip_value("left") == 0.4


def test_latest_grip_and_button_state_wins():
    """Treat Grip as current state and discard superseded backlog state."""
    reader = make_stream_reader()
    chunk = b"".join(
        (
            sample_line(0.0, left_grip=1.0, right_grip=1.0),
            sample_line(0.1, left_grip=0.0, right_grip=0.25),
        )
    )

    reader._consume_logcat_bytes(b"", chunk)

    assert reader.get_grip_value("left") == 0.0
    assert reader.get_grip_value("right") == 0.25
    assert reader.backlog_lines_dropped == 1


def test_tracking_valid_comes_from_latest_sample():
    """Do not retain HIGH tracking from an older line in the same backlog."""
    reader = make_stream_reader()
    chunk = b"".join(
        (
            sample_line(0.0, left_tracking=1, right_tracking=0),
            sample_line(0.1, left_tracking=0, right_tracking=1),
        )
    )

    reader._consume_logcat_bytes(b"", chunk)

    assert not reader.get_tracking_valid("left")
    assert reader.get_tracking_valid("right")


def test_freshness_timestamp_is_sampled_once_for_latest_commit(monkeypatch):
    """Refresh source age once per drained batch, never once per old frame."""
    reader = make_stream_reader()
    monotonic_calls = []

    def fake_monotonic():
        monotonic_calls.append(True)
        return 123.5

    monkeypatch.setattr(reader_module.time, "monotonic", fake_monotonic)
    reader._consume_logcat_bytes(
        b"",
        b"".join(sample_line(x) for x in (0.0, 0.1, 0.2, 0.3)),
    )

    assert len(monotonic_calls) == 1
    assert reader._last_sample_monotonic == 123.5
    assert reader.get_data_age_s(123.6) == pytest.approx(0.1)
