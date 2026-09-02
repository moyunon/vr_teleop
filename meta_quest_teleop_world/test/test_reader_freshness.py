"""Offline tests for Quest reader source freshness."""

import math
import threading

import pytest

from meta_quest_teleop.reader import MetaQuestReader


def make_reader(last_sample):
    """Create only the state needed by freshness methods, without ADB."""
    reader = MetaQuestReader.__new__(MetaQuestReader)
    reader._lock = threading.Lock()
    reader._last_sample_monotonic = last_sample
    return reader


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
