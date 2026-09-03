"""Tests for fresh, persistent, hysteretic following-error decisions."""

import numpy as np
import pytest

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


def test_pre_engagement_command_waits_without_holding():
    """Do not reuse a command timestamp from an earlier engagement epoch."""
    instance = monitor()
    result = instance.evaluate(
        [0.25, 0.0],
        90.0,
        [0.0, 0.0],
        100.0,
        100.0,
        engagement_start_monotonic=100.0,
    )

    assert result.state == FollowingErrorState.NORMAL
    assert result.ready
    assert not result.hold_required
    assert result.error_rad is None
    assert result.engagement_start_time == 100.0
    assert result.engagement_age_s == 0.0
    assert result.command_time == 90.0
    assert result.command_age_s == 10.0
    assert result.measurement_age_s == 0.0
    assert result.timestamp_skew_s == 10.0
    assert not result.command_is_current_engagement
    assert result.reason == "awaiting first post-engagement safe command"


def test_first_current_engagement_command_starts_comparison():
    """Start normal following protection on the first epoch-local command."""
    instance = monitor()
    result = instance.evaluate(
        [0.15, 0.0],
        100.01,
        [0.0, 0.0],
        100.01,
        100.02,
        engagement_start_monotonic=100.0,
    )

    assert result.ready
    assert not result.hold_required
    assert result.command_is_current_engagement
    assert result.max_abs_error_rad == 0.15
    assert result.command_time == 100.01
    assert result.command_age_s == pytest.approx(0.01)


def test_new_engagement_rejects_previous_epoch_command():
    """Return to awaiting after HOLD/READY starts a later engagement."""
    instance = monitor()
    first = instance.evaluate(
        [0.0, 0.0],
        100.01,
        [0.0, 0.0],
        100.01,
        100.02,
        engagement_start_monotonic=100.0,
    )
    second = instance.evaluate(
        [0.25, 0.0],
        100.01,
        [0.0, 0.0],
        110.0,
        110.0,
        engagement_start_monotonic=110.0,
    )

    assert first.command_is_current_engagement
    assert second.ready
    assert not second.hold_required
    assert not second.command_is_current_engagement
    assert second.reason == "awaiting first post-engagement safe command"


def test_current_engagement_command_still_fails_closed_when_stale():
    """Keep command freshness enforcement after epoch admission."""
    instance = monitor()
    result = instance.evaluate(
        [0.0, 0.0],
        100.01,
        [0.0, 0.0],
        100.30,
        100.30,
        engagement_start_monotonic=100.0,
    )

    assert result.command_is_current_engagement
    assert not result.ready
    assert result.hold_required
    assert result.reason == "following-error command or measurement is stale"


def test_epoch_wait_does_not_bypass_stale_measurement():
    """Fail closed on stale feedback even while awaiting the first command."""
    instance = monitor()
    result = instance.evaluate(
        [0.0, 0.0],
        90.0,
        [0.0, 0.0],
        99.0,
        100.0,
        engagement_start_monotonic=100.0,
    )

    assert not result.command_is_current_engagement
    assert not result.ready
    assert result.hold_required
    assert result.reason == "following-error command or measurement is stale"


def test_current_engagement_timestamp_skew_still_fails_closed():
    """Keep command/measurement alignment enforcement after engagement."""
    instance = monitor()
    result = instance.evaluate(
        [0.0, 0.0],
        100.10,
        [0.0, 0.0],
        100.04,
        100.10,
        engagement_start_monotonic=100.0,
    )

    assert result.command_is_current_engagement
    assert not result.ready
    assert result.hold_required
    assert result.timestamp_skew_s == pytest.approx(0.06)
    assert result.reason == "following-error timestamps are not aligned"
