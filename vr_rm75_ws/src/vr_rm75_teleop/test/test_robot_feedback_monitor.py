"""Tests for direct/fallback measured derivative quality."""

import numpy as np
import pytest

from vr_rm75_teleop.robot_feedback_monitor import RobotFeedbackMonitor


def test_direct_velocity_and_actual_dt_drive_filtered_acceleration():
    """Use UDP qdot and the observed 30 ms interval, not a nominal period."""
    monitor = RobotFeedbackMonitor(2, acceleration_filter_tau_s=0.0)
    first = monitor.update([0.0, 0.0], 1.0, [1.0, 2.0])
    second = monitor.update([0.03, 0.06], 1.03, [1.3, 2.6])

    assert first.velocity_source == "udp_direct"
    assert first.qddot is None
    assert second.dt_s == pytest.approx(0.03)
    assert second.qdot == pytest.approx([1.3, 2.6])
    assert second.qddot == pytest.approx([10.0, 20.0])


def test_fallback_is_explicit_and_uses_measured_period():
    """Label finite difference and compute it from the real sample interval."""
    monitor = RobotFeedbackMonitor(2)
    assert not monitor.update([0.0, 0.0], 1.0).valid
    result = monitor.update([0.2, -0.1], 1.1)
    assert result.valid
    assert result.velocity_source == "finite_difference"
    assert result.qdot == pytest.approx([2.0, -1.0])


def test_duplicate_stale_and_nan_samples_are_rejected():
    """Do not differentiate a duplicate, long gap, or non-finite packet."""
    monitor = RobotFeedbackMonitor(2, max_dt_s=0.2)
    monitor.update([0.0, 0.0], 1.0, [0.0, 0.0])
    assert not monitor.update([0.0, 0.0], 1.0, [0.0, 0.0]).valid
    assert not monitor.update([0.0, 0.0], 1.3, [0.0, 0.0]).valid
    with pytest.raises(ValueError, match="finite"):
        monitor.update([np.nan, 0.0], 1.4)
