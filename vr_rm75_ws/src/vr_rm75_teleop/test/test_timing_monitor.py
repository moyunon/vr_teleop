"""Tests for measured control period, percentiles, and deadline counters."""

import pytest

from vr_rm75_teleop.timing_monitor import TimingMonitor


def test_summary_reports_actual_frequency_jitter_and_percentiles():
    """Derive statistics from observations instead of configured frequency."""
    monitor = TimingMonitor(nominal_period_s=0.02, window_size=10)
    for timestamp in (1.0, 1.02, 1.04, 1.07):
        monitor.begin_cycle(timestamp)
    for duration in (0.001, 0.002, 0.003):
        monitor.record("ik_left", duration)
    summary = monitor.summary()

    assert summary["control"]["cycle_count"] == 4
    assert summary["control"]["deadline_miss_count"] >= 1
    assert summary["control"]["effective_frequency_hz"] == pytest.approx(
        1.0 / ((0.02 + 0.02 + 0.03) / 3.0)
    )
    assert summary["ik_left"]["max_s"] == pytest.approx(0.003)
    assert summary["ik_left"]["p95_s"] >= summary["ik_left"]["mean_s"]
