"""Tests for fresh, persistent, hysteretic following-error decisions."""

import numpy as np

from vr_rm75_teleop.following_error_monitor import (
    FollowingErrorMonitor,
    FollowingErrorState,
)


def monitor():
    """Return a two-axis monitor with short deterministic persistence."""
    return FollowingErrorMonitor(
        np.asarray([0.1, 0.1]),
        np.asarray([0.2, 0.2]),
        persistence_s=0.1,
        max_age_s=0.2,
        max_timestamp_skew_s=0.05,
    )


def evaluate(instance, error, now):
    """Evaluate equal-time fresh command/feedback vectors."""
    return instance.evaluate(error, now, [0.0, 0.0], now, now)


def test_warning_and_stop_require_persistence():
    """Ignore a single spike and commit only a sustained region."""
    instance = monitor()
    assert evaluate(instance, [0.15, 0.0], 1.0).state == (
        FollowingErrorState.NORMAL
    )
    assert evaluate(instance, [0.0, 0.0], 1.05).state == (
        FollowingErrorState.NORMAL
    )
    assert evaluate(instance, [0.25, 0.0], 2.0).state == (
        FollowingErrorState.NORMAL
    )
    result = evaluate(instance, [0.25, 0.0], 2.11)
    assert result.state == FollowingErrorState.STOP
    assert result.hold_required


def test_stop_hysteresis_prevents_chatter_then_clears_persistently():
    """Remain stopped above the lower clear band and debounce recovery."""
    instance = monitor()
    evaluate(instance, [0.25, 0.0], 1.0)
    assert evaluate(instance, [0.25, 0.0], 1.11).state == (
        FollowingErrorState.STOP
    )
    assert evaluate(instance, [0.17, 0.0], 1.20).state == (
        FollowingErrorState.STOP
    )
    assert evaluate(instance, [0.0, 0.0], 1.30).state == (
        FollowingErrorState.STOP
    )
    assert evaluate(instance, [0.0, 0.0], 1.41).state == (
        FollowingErrorState.NORMAL
    )


def test_continuous_joint_wrap_and_stale_or_skew_fail_closed():
    """Wrap only declared continuous joints and reject timestamp skew."""
    instance = FollowingErrorMonitor(
        [0.1],
        [0.2],
        continuous_joints=[True],
        max_age_s=0.2,
        max_timestamp_skew_s=0.05,
    )
    wrapped = instance.evaluate(
        [-np.pi + 0.01], 1.0, [np.pi - 0.01], 1.0, 1.0
    )
    assert abs(wrapped.error_rad[0]) < 0.03
    assert not instance.evaluate([0.0], 1.0, [0.0], 1.0, 1.3).ready
    assert not instance.evaluate([0.0], 2.0, [0.0], 1.9, 2.0).ready


def test_bounded_joint_does_not_wrap_across_limits():
    """Treat RM75 bounded joint representation as a direct signed error."""
    instance = FollowingErrorMonitor([0.1], [0.2], continuous_joints=[False])
    result = instance.evaluate([-3.13], 1.0, [3.13], 1.0, 1.0)
    assert abs(result.error_rad[0]) > 6.0


def test_reset_allows_release_and_reengage_after_stop():
    """Clear the monitor latch only as part of the external re-arm sequence."""
    instance = monitor()
    evaluate(instance, [0.25, 0.0], 1.0)
    assert evaluate(instance, [0.25, 0.0], 1.11).hold_required
    instance.reset()
    result = evaluate(instance, [0.0, 0.0], 1.12)
    assert result.state == FollowingErrorState.NORMAL
    assert not result.hold_required
