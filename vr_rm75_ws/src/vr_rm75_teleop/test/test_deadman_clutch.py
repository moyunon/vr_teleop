"""Tests for fail-closed dual-grip deadman behavior."""

import math

import pytest

from vr_rm75_teleop.deadman_clutch import DualGripDeadman


def release_to_ready(clutch, now=1.0):
    """Provide fresh released samples that clear the rearm latch."""
    clutch.update_grip("left", 0.0, now)
    decision = clutch.update_grip("right", 0.0, now)
    assert not decision.active
    assert not decision.rearm_required


def engage(clutch, now=1.1):
    """Apply a deliberate two-hand press."""
    clutch.update_grip("left", 0.8, now)
    decision = clutch.update_grip("right", 0.8, now)
    assert decision.active
    return decision


def test_requires_released_start_then_both_grips():
    """Do not engage from a held startup or from only one hand."""
    clutch = DualGripDeadman()
    assert not clutch.update_grip("left", 0.8, 1.0).active
    decision = clutch.update_grip("right", 0.8, 1.0)
    assert not decision.active
    assert decision.rearm_required

    release_to_ready(clutch, 1.1)
    assert not clutch.update_grip("left", 0.8, 1.2).active
    assert clutch.update_grip("right", 0.8, 1.2).active


def test_hysteresis_ignores_noise_between_thresholds():
    """Keep the prior per-hand state inside the 0.35/0.65 band."""
    clutch = DualGripDeadman()
    release_to_ready(clutch)
    engage(clutch)

    assert clutch.update_grip("left", 0.50, 1.11).active
    assert not clutch.update_grip("left", 0.34, 1.12).active
    assert not clutch.update_grip("left", 0.50, 1.13).active
    assert clutch.update_grip("left", 0.66, 1.14).active


def test_either_hand_release_drops_global_deadman():
    """Coordinate both arms by releasing globally when either grip opens."""
    clutch = DualGripDeadman()
    release_to_ready(clutch)
    engage(clutch)

    decision = clutch.update_grip("right", 0.2, 1.11)
    assert not decision.active
    assert decision.changed
    assert not decision.rearm_required


def test_timeout_requires_release_before_repress():
    """Never resume automatically from cached held values after a gap."""
    clutch = DualGripDeadman(input_timeout_s=0.20)
    release_to_ready(clutch)
    engage(clutch)

    decision = clutch.evaluate(1.31)
    assert not decision.active
    assert decision.rearm_required

    clutch.update_grip("left", 0.8, 1.32)
    decision = clutch.update_grip("right", 0.8, 1.32)
    assert not decision.active
    assert decision.rearm_required

    release_to_ready(clutch, 1.33)
    assert engage(clutch, 1.34).active


def test_source_stale_is_immediate_and_requires_rearm():
    """Fail closed from an explicit bridge source-freshness loss."""
    clutch = DualGripDeadman()
    release_to_ready(clutch)
    engage(clutch)

    decision = clutch.evaluate(1.11, source_fresh=False)
    assert not decision.active
    assert decision.rearm_required
    assert decision.reason == "Quest input source stale"


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1, "bad"])
def test_invalid_analog_values_fail_closed(value):
    """Reject non-finite, malformed, and out-of-range grip samples."""
    clutch = DualGripDeadman()
    release_to_ready(clutch)
    engage(clutch)

    decision = clutch.update_grip("left", value, 1.11)
    assert not decision.active
    assert decision.rearm_required


@pytest.mark.parametrize(
    "off_threshold,on_threshold,timeout",
    [
        (0.7, 0.6, 0.2),
        (-0.1, 0.6, 0.2),
        (0.3, 1.1, 0.2),
        (0.3, 0.6, 0.0),
    ],
)
def test_invalid_configuration_is_rejected(
    off_threshold,
    on_threshold,
    timeout,
):
    """Reject threshold or timeout settings that weaken determinism."""
    with pytest.raises(ValueError):
        DualGripDeadman(
            on_threshold=on_threshold,
            off_threshold=off_threshold,
            input_timeout_s=timeout,
        )
