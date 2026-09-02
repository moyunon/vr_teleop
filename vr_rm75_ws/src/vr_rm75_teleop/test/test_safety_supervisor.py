"""Deterministic tests for the dual-arm safety state machine."""

import numpy as np
import pytest

from vr_rm75_teleop.joint_safety import make_teleop_soft_limits
from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.safety_supervisor import SafetyState, SafetySupervisor


Q = {
    "left": np.deg2rad([-40.0, -25.0, 15.0, -55.0, 10.0, -35.0, 80.0]),
    "right": np.deg2rad([20.0, 35.0, 25.0, 60.0, 15.0, 40.0, -120.0]),
}


def configured_soft_limits():
    """Return production-style side-specific teleoperation limits."""
    limits = {}
    for side in ("left", "right"):
        model = RM75Model(side)
        limits[side] = make_teleop_soft_limits(
            model.q_min,
            model.q_max,
            np.deg2rad(5.0),
            elbow_index=3,
            elbow_branch=-1 if side == "left" else 1,
            elbow_margin=np.deg2rad(15.0),
        )
    return limits


def configured_acceleration_limits(limit_deg_s2=30.0):
    """Return equal project-level qdd limits for both arms."""
    return {
        side: np.deg2rad(np.full(7, limit_deg_s2))
        for side in ("left", "right")
    }


def observe_healthy(
    supervisor,
    *,
    vr_ready=True,
    last_command=None,
    ik_failures=0,
):
    """Install a healthy observation for each arm."""
    supervisor.update_collision(
        ready=True,
        hold_required=False,
        speed_scale=1.0,
        reason="collision clear",
    )
    for side in ("left", "right"):
        supervisor.update_arm(
            side,
            q_measured=Q[side],
            q_candidate=Q[side],
            q_command=Q[side],
            joint_velocity=np.zeros(7),
            sigma_min=0.05,
            robot_initialized=True,
            robot_connected=True,
            robot_stale=False,
            robot_enabled=True,
            robot_fault=False,
            vr_tracking_valid=vr_ready,
            vr_stale=not vr_ready,
            last_command_monotonic=last_command,
            consecutive_ik_failures=ik_failures,
        )


def engage(supervisor, now=1.0):
    """Advance a healthy supervisor through INIT, READY, and ENGAGED."""
    observe_healthy(supervisor)
    assert supervisor.evaluate(False, now).state == SafetyState.READY
    decision = supervisor.evaluate(True, now)
    assert decision.state == SafetyState.ENGAGED
    assert decision.command_allowed


def test_init_ready_engaged_hold_ready_reengaged_sequence():
    """Require deadman release before a HOLD can engage again."""
    supervisor = SafetySupervisor(command_timeout_s=0.2)
    engage(supervisor, 1.0)

    decision = supervisor.evaluate(False, 1.01)
    assert decision.state == SafetyState.HOLD
    assert decision.reason == "deadman released"
    assert not decision.command_allowed

    assert supervisor.evaluate(True, 1.02).state == SafetyState.HOLD
    assert supervisor.evaluate(False, 1.03).state == SafetyState.READY
    assert supervisor.evaluate(True, 1.04).state == SafetyState.ENGAGED


@pytest.mark.parametrize(
    "condition, expected_reason",
    [
        ("vr_stale", "VR tracking lost or pose stale"),
        (
            "robot_stale",
            "robot communication stale, disconnected, or disabled",
        ),
        (
            "robot_disconnected",
            "robot communication stale, disconnected, or disabled",
        ),
        (
            "robot_disabled",
            "robot communication stale, disconnected, or disabled",
        ),
        ("ik_failures", "consecutive IK failure limit reached"),
    ],
)
def test_engaged_recoverable_failures_enter_hold(condition, expected_reason):
    """Convert every recoverable runtime safety loss into HOLD."""
    supervisor = SafetySupervisor(
        command_timeout_s=1.0,
        max_consecutive_ik_failures=3,
    )
    engage(supervisor, 2.0)

    values = {
        "q_measured": Q["left"],
        "q_candidate": Q["left"],
        "q_command": Q["left"],
        "sigma_min": 0.05,
        "robot_initialized": True,
        "robot_connected": condition != "robot_disconnected",
        "robot_stale": condition == "robot_stale",
        "robot_enabled": condition != "robot_disabled",
        "robot_fault": False,
        "vr_tracking_valid": True,
        "vr_stale": condition == "vr_stale",
        "last_command_monotonic": 2.0,
        "consecutive_ik_failures": 3 if condition == "ik_failures" else 0,
    }
    supervisor.update_arm("left", **values)

    decision = supervisor.evaluate(True, 2.01)
    assert decision.state == SafetyState.HOLD
    assert decision.reason == expected_reason


def test_command_watchdog_holds_when_either_arm_has_no_new_command():
    """Stop after the engagement grace period if a safe command is absent."""
    supervisor = SafetySupervisor(command_timeout_s=0.10)
    engage(supervisor, 3.0)

    assert supervisor.evaluate(True, 3.099).state == SafetyState.ENGAGED
    decision = supervisor.evaluate(True, 3.101)
    assert decision.state == SafetyState.HOLD
    assert decision.reason == "safe command watchdog expired"


def test_missing_collision_state_blocks_ready_transition():
    """Do not leave INIT before the global collision guard is available."""
    supervisor = SafetySupervisor()
    for side in ("left", "right"):
        supervisor.update_arm(
            side,
            q_measured=Q[side],
            q_candidate=Q[side],
            q_command=Q[side],
            joint_velocity=np.zeros(7),
            sigma_min=0.05,
            robot_initialized=True,
            robot_connected=True,
            robot_stale=False,
            robot_enabled=True,
            robot_fault=False,
            vr_tracking_valid=True,
            vr_stale=False,
        )

    decision = supervisor.evaluate(False, 3.11)

    assert decision.state == SafetyState.INIT
    assert not decision.command_allowed
    assert decision.reason == "collision distance unavailable"


def test_collision_warning_keeps_engaged_with_reduced_speed_scale():
    """Allow the rate limiters to consume a valid warning-region scale."""
    supervisor = SafetySupervisor(command_timeout_s=1.0)
    engage(supervisor, 3.12)
    supervisor.update_collision(
        ready=True,
        hold_required=False,
        speed_scale=0.4,
        reason="collision warning: inter_arm",
    )

    decision = supervisor.evaluate(True, 3.13)

    assert decision.state == SafetyState.ENGAGED
    assert decision.command_allowed
    assert supervisor.collision_speed_scale == pytest.approx(0.4)


def test_collision_stop_holds_both_arms_until_deadman_is_released():
    """Apply one global collision stop to the dual-arm command gate."""
    supervisor = SafetySupervisor(command_timeout_s=1.0)
    engage(supervisor, 3.14)
    supervisor.update_collision(
        ready=True,
        hold_required=True,
        speed_scale=0.0,
        reason="collision stop: robot_body",
    )

    decision = supervisor.evaluate(True, 3.15)

    assert decision.state == SafetyState.HOLD
    assert not decision.command_allowed
    assert decision.reason == "collision stop: robot_body"

    supervisor.update_collision(
        ready=True,
        hold_required=False,
        speed_scale=1.0,
        reason="collision clear",
    )
    assert supervisor.evaluate(True, 3.16).state == SafetyState.HOLD
    assert supervisor.evaluate(False, 3.17).state == SafetyState.READY


def test_collision_observation_rejects_invalid_scale():
    """Reject non-finite global collision speed scaling."""
    supervisor = SafetySupervisor()

    with pytest.raises(ValueError, match="collision speed_scale"):
        supervisor.update_collision(
            ready=True,
            hold_required=False,
            speed_scale=float("nan"),
            reason="invalid",
        )


def test_missing_required_actuator_state_blocks_ready_transition():
    """Keep motion-enabled startup in INIT until both sockets are ready."""
    supervisor = SafetySupervisor(require_actuator_safety=True)
    observe_healthy(supervisor)

    decision = supervisor.evaluate(False, 3.18)

    assert decision.state == SafetyState.INIT
    assert not decision.command_allowed
    assert decision.reason == "robot command interface unavailable"


def test_actuator_rejection_holds_and_requires_deadman_release():
    """Route a command jump rejection through the global HOLD state."""
    supervisor = SafetySupervisor(
        command_timeout_s=1.0,
        require_actuator_safety=True,
    )
    supervisor.update_actuator(
        ready=True,
        hold_required=False,
        fault=False,
        reason="command channels connected",
    )
    engage(supervisor, 3.19)
    supervisor.update_actuator(
        ready=True,
        hold_required=True,
        fault=False,
        reason="left command jump exceeds configured maximum",
    )

    decision = supervisor.evaluate(True, 3.20)

    assert decision.state == SafetyState.HOLD
    assert not decision.command_allowed
    assert decision.reason == "left command jump exceeds configured maximum"

    supervisor.update_actuator(
        ready=True,
        hold_required=False,
        fault=False,
        reason="command channels connected",
    )
    assert supervisor.evaluate(True, 3.21).state == SafetyState.HOLD
    assert supervisor.evaluate(False, 3.22).state == SafetyState.READY


def test_actuator_transport_failure_is_a_latched_fault():
    """Treat asymmetric dual-arm transport loss as a severe FAULT."""
    supervisor = SafetySupervisor(require_actuator_safety=True)
    supervisor.update_actuator(
        ready=False,
        hold_required=True,
        fault=True,
        reason="dual-arm command send failed; fault latched",
    )

    decision = supervisor.evaluate(False, 3.23)

    assert decision.state == SafetyState.FAULT
    assert not decision.command_allowed
    assert decision.reason == "dual-arm command send failed; fault latched"


def test_joint_velocity_above_scaled_model_limit_enters_hold():
    """Independently reject a finite command that bypasses the limiter."""
    supervisor = SafetySupervisor(
        command_timeout_s=1.0,
        joint_velocity_scale=0.10,
    )
    engage(supervisor, 3.2)
    _, _, qd_max = supervisor.get_joint_limits("left")
    excessive_velocity = 0.10 * np.asarray(qd_max)
    excessive_velocity[3] += 1e-4

    supervisor.update_arm(
        "left",
        q_measured=Q["left"],
        q_candidate=Q["left"],
        q_command=Q["left"],
        joint_velocity=excessive_velocity,
        sigma_min=0.05,
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        vr_tracking_valid=True,
        vr_stale=False,
    )

    decision = supervisor.evaluate(True, 3.21)
    assert decision.state == SafetyState.HOLD
    assert decision.reason == "teleoperation joint velocity limit exceeded"


def test_joint_velocity_exactly_at_scaled_limit_remains_engaged():
    """Accept the configured boundary without floating-point chatter."""
    supervisor = SafetySupervisor(
        command_timeout_s=1.0,
        joint_velocity_scale=0.10,
    )
    engage(supervisor, 3.3)
    _, _, qd_max = supervisor.get_joint_limits("right")
    boundary_velocity = 0.10 * np.asarray(qd_max)

    supervisor.update_arm(
        "right",
        q_measured=Q["right"],
        q_candidate=Q["right"],
        q_command=Q["right"],
        joint_velocity=boundary_velocity,
        sigma_min=0.05,
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        vr_tracking_valid=True,
        vr_stale=False,
    )

    assert supervisor.evaluate(True, 3.31).state == SafetyState.ENGAGED


def test_joint_acceleration_above_configured_limit_enters_hold():
    """Independently detect qdd that bypasses the online limiter."""
    supervisor = SafetySupervisor(
        command_timeout_s=1.0,
        joint_acceleration_limits=configured_acceleration_limits(),
    )
    engage(supervisor, 3.32)
    acceleration = configured_acceleration_limits()["left"]
    acceleration[4] += 1e-4

    supervisor.update_arm(
        "left",
        q_measured=Q["left"],
        q_candidate=Q["left"],
        q_command=Q["left"],
        joint_velocity=np.zeros(7),
        joint_acceleration=acceleration,
        sigma_min=0.05,
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        vr_tracking_valid=True,
        vr_stale=False,
    )

    decision = supervisor.evaluate(True, 3.33)
    assert decision.state == SafetyState.HOLD
    assert decision.reason == (
        "teleoperation joint acceleration limit exceeded"
    )


def test_joint_acceleration_at_limit_remains_engaged():
    """Accept the exact qdd boundary without floating-point chatter."""
    limits = configured_acceleration_limits()
    supervisor = SafetySupervisor(
        command_timeout_s=1.0,
        joint_acceleration_limits=limits,
    )
    engage(supervisor, 3.34)
    supervisor.update_arm(
        "right",
        q_measured=Q["right"],
        q_candidate=Q["right"],
        q_command=Q["right"],
        joint_velocity=np.zeros(7),
        joint_acceleration=limits["right"],
        sigma_min=0.05,
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        vr_tracking_valid=True,
        vr_stale=False,
    )

    assert supervisor.evaluate(True, 3.35).state == SafetyState.ENGAGED
    assert supervisor.get_joint_acceleration_limits("right") == pytest.approx(
        limits["right"]
    )


@pytest.mark.parametrize(
    "configured",
    [
        {"left": np.ones(7)},
        {"left": np.ones(7), "right": np.ones(6)},
        {"left": np.ones(7), "right": np.zeros(7)},
        {"left": np.ones(7), "right": np.full(7, np.nan)},
    ],
)
def test_invalid_joint_acceleration_limits_are_rejected(configured):
    """Reject incomplete, malformed, zero, and non-finite qdd limits."""
    with pytest.raises(ValueError, match="acceleration"):
        SafetySupervisor(joint_acceleration_limits=configured)


@pytest.mark.parametrize(
    "scale",
    [0.0, -0.1, 1.01, float("nan"), float("inf")],
)
def test_invalid_joint_velocity_scale_is_rejected(scale):
    """Prevent unsafe or nonsensical supervisor velocity configuration."""
    with pytest.raises(ValueError):
        SafetySupervisor(joint_velocity_scale=scale)


def test_startup_outside_soft_limits_remains_in_init():
    """Do not declare a robot ready when measured J4 is on the unsafe side."""
    supervisor = SafetySupervisor(
        joint_soft_limits=configured_soft_limits()
    )
    observe_healthy(supervisor)
    q_outside = Q["left"].copy()
    q_outside[3] = np.deg2rad(-10.0)
    supervisor.update_arm(
        "left",
        q_measured=q_outside,
        q_candidate=Q["left"],
        q_command=Q["left"],
        joint_velocity=np.zeros(7),
        sigma_min=0.05,
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
    )

    decision = supervisor.evaluate(False, 3.4)
    assert decision.state == SafetyState.INIT
    assert decision.reason == "left measured joints outside teleop soft limits"


def test_runtime_measured_soft_limit_violation_enters_hold():
    """Stop following if actual feedback leaves the commissioning region."""
    supervisor = SafetySupervisor(
        command_timeout_s=1.0,
        joint_soft_limits=configured_soft_limits(),
    )
    engage(supervisor, 3.5)
    q_outside = Q["right"].copy()
    q_outside[3] = np.deg2rad(10.0)
    supervisor.update_arm(
        "right",
        q_measured=q_outside,
        q_candidate=Q["right"],
        q_command=Q["right"],
        joint_velocity=np.zeros(7),
        sigma_min=0.05,
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        vr_tracking_valid=True,
        vr_stale=False,
    )

    decision = supervisor.evaluate(True, 3.51)
    assert decision.state == SafetyState.HOLD
    assert decision.reason == (
        "right measured joints outside teleop soft limits"
    )


def test_candidate_may_exceed_soft_limit_when_command_is_safely_bounded():
    """Allow the command-stage limiter to saturate an IK candidate."""
    supervisor = SafetySupervisor(
        command_timeout_s=1.0,
        joint_soft_limits=configured_soft_limits(),
    )
    engage(supervisor, 3.6)
    q_candidate = Q["left"].copy()
    q_candidate[3] = np.deg2rad(20.0)
    supervisor.update_arm(
        "left",
        q_measured=Q["left"],
        q_candidate=q_candidate,
        q_command=Q["left"],
        joint_velocity=np.zeros(7),
        sigma_min=0.05,
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        vr_tracking_valid=True,
        vr_stale=False,
    )

    assert supervisor.evaluate(True, 3.61).state == SafetyState.ENGAGED


def test_soft_limit_configuration_must_be_strictly_inside_hard_limits():
    """Reject a Supervisor boundary that merely duplicates hard limits."""
    limits = configured_soft_limits()
    model = RM75Model("left")
    limits["left"] = (model.q_min, model.q_max)
    with pytest.raises(ValueError, match="inside hard limits"):
        SafetySupervisor(joint_soft_limits=limits)


def test_robot_fault_is_latched_until_explicit_safe_reset():
    """Keep FAULT latched after the source clears until reset is requested."""
    supervisor = SafetySupervisor()
    observe_healthy(supervisor)
    assert supervisor.evaluate(False, 4.0).state == SafetyState.READY

    supervisor.update_arm(
        "left",
        q_measured=Q["left"],
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        robot_fault=True,
    )
    assert supervisor.evaluate(False, 4.1).state == SafetyState.FAULT

    observe_healthy(supervisor)
    assert supervisor.evaluate(False, 4.2).state == SafetyState.FAULT
    supervisor.request_fault_reset()
    assert supervisor.evaluate(False, 4.3).state == SafetyState.INIT
    assert supervisor.evaluate(False, 4.4).state == SafetyState.READY


def test_fault_reset_is_rejected_while_deadman_is_active():
    """Require a released operator control before leaving latched FAULT."""
    supervisor = SafetySupervisor()
    observe_healthy(supervisor)
    supervisor.update_arm(
        "left",
        q_measured=Q["left"],
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        robot_fault=True,
    )
    assert supervisor.evaluate(False, 4.5).state == SafetyState.FAULT

    observe_healthy(supervisor)
    supervisor.request_fault_reset()
    decision = supervisor.evaluate(True, 4.6)
    assert decision.state == SafetyState.FAULT
    assert decision.reason == "fault reset rejected while deadman active"


def test_reset_request_outside_fault_does_not_arm_a_future_reset():
    """Reject stale reset requests made before any FAULT is present."""
    supervisor = SafetySupervisor()
    observe_healthy(supervisor)
    assert supervisor.evaluate(False, 4.7).state == SafetyState.READY
    assert supervisor.request_fault_reset() is False

    supervisor.update_arm(
        "right",
        q_measured=Q["right"],
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        robot_fault=True,
    )
    assert supervisor.evaluate(False, 4.8).state == SafetyState.FAULT
    observe_healthy(supervisor)
    assert supervisor.evaluate(False, 4.9).state == SafetyState.FAULT


@pytest.mark.parametrize(
    "field, value",
    [
        ("q_measured", np.full(7, np.nan)),
        ("q_candidate", np.full(7, np.inf)),
        ("q_command", np.deg2rad([0, 0, 0, 136, 0, 0, 0])),
        ("joint_velocity", np.full(7, np.nan)),
        ("joint_acceleration", np.full(7, np.nan)),
        ("sigma_min", float("nan")),
    ],
)
def test_invalid_numeric_safety_data_enters_fault(field, value):
    """Treat non-finite or hard-limit data as a severe latched fault."""
    supervisor = SafetySupervisor()
    observe_healthy(supervisor)
    assert supervisor.evaluate(False, 5.0).state == SafetyState.READY

    values = {
        "q_measured": Q["right"],
        "q_candidate": Q["right"],
        "q_command": Q["right"],
        "joint_velocity": np.zeros(7),
        "sigma_min": 0.05,
        "robot_initialized": True,
        "robot_connected": True,
        "robot_stale": False,
        "robot_enabled": True,
    }
    values[field] = value
    observation = supervisor.update_arm("right", **values)

    assert not observation.numeric_valid
    assert supervisor.evaluate(False, 5.1).state == SafetyState.FAULT


def test_observation_retains_measured_candidate_command_and_sigma():
    """Expose the values that later joint and singularity guards consume."""
    supervisor = SafetySupervisor()
    observe_healthy(supervisor, last_command=6.0)

    observation = supervisor.get_observation("left")
    assert np.allclose(observation.q_measured, Q["left"])
    assert np.allclose(observation.q_candidate, Q["left"])
    assert np.allclose(observation.q_command, Q["left"])
    assert np.allclose(observation.joint_velocity, np.zeros(7))
    assert observation.sigma_min == 0.05
    assert observation.last_command_monotonic == 6.0

    q_min, q_max, qd_max = supervisor.get_joint_limits("left")
    assert len(q_min) == len(q_max) == len(qd_max) == 7


def test_upstream_numeric_invalidity_enters_fault():
    """Allow pose/Jacobian/result checks to fail closed at the boundary."""
    supervisor = SafetySupervisor()
    observe_healthy(supervisor)
    supervisor.update_arm(
        "left",
        q_measured=Q["left"],
        robot_initialized=True,
        robot_connected=True,
        robot_stale=False,
        robot_enabled=True,
        upstream_numeric_valid=False,
    )
    assert supervisor.evaluate(False, 6.1).state == SafetyState.FAULT


def test_transition_table_rejects_illegal_transition():
    """Reject state jumps that are absent from the explicit transition map."""
    supervisor = SafetySupervisor()
    with pytest.raises(RuntimeError, match="illegal safety transition"):
        supervisor._transition(SafetyState.HOLD, "test")


def test_every_declared_transition_is_executable():
    """Keep the declared legal-transition table internally consistent."""
    for source, targets in SafetySupervisor.ALLOWED_TRANSITIONS.items():
        for target in targets:
            supervisor = SafetySupervisor()
            supervisor.state = source
            supervisor._transition(target, "table test")
            assert supervisor.state == target
