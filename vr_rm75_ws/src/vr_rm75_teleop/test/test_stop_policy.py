"""Tests for edge-triggered dual-arm software-stop classification."""

import pytest

from vr_rm75_teleop.safety_supervisor import SafetyDecision, SafetyState
from vr_rm75_teleop.stop_policy import StopClass, stop_for_transition


def decision(previous, state, reason):
    """Build the minimal immutable Supervisor transition result."""
    return SafetyDecision(
        previous_state=previous,
        state=state,
        changed=previous != state,
        reason=reason,
        command_allowed=state == SafetyState.ENGAGED,
    )


def test_deadman_release_uses_controlled_stop():
    """Treat the intentional operator release as the gentler stop class."""
    request = stop_for_transition(
        decision(SafetyState.ENGAGED, SafetyState.HOLD, "deadman released"),
        12.0,
    )
    assert request.stop_class == StopClass.CONTROLLED_STOP
    assert request.requested_monotonic == pytest.approx(12.0)


@pytest.mark.parametrize(
    "state, reason",
    [
        (SafetyState.HOLD, "VR tracking lost or pose stale"),
        (SafetyState.HOLD, "collision STOP: left arm vs environment"),
        (SafetyState.HOLD, "safe command watchdog expired"),
        (SafetyState.FAULT, "robot or joint fault reported"),
    ],
)
def test_unexpected_engaged_exit_uses_safety_stop(state, reason):
    """Escalate every non-operator-loss edge out of ENGAGED."""
    request = stop_for_transition(
        decision(SafetyState.ENGAGED, state, reason),
        1.0,
    )
    assert request.stop_class == StopClass.SAFETY_STOP


def test_repeated_hold_or_non_engaged_transition_emits_no_stop():
    """Prevent periodic stop-command spam while a hold remains active."""
    assert stop_for_transition(
        decision(SafetyState.HOLD, SafetyState.HOLD, "still holding"),
        1.0,
    ) is None
    assert stop_for_transition(
        decision(SafetyState.READY, SafetyState.HOLD, "not engaged"),
        1.0,
    ) is None
