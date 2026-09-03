"""ROS callback integration tests for dual-grip clutch re-anchoring."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray

import vr_rm75_teleop.quest_dual_ik_fusion as fusion_module
from vr_rm75_teleop.quest_dual_ik_fusion import QuestDualIKFusion
from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.safety_supervisor import SafetyState
from vr_rm75_teleop.stop_policy import StopClass
from vr_rm75_teleop.target_feasibility import (
    minimum_singular_value,
    singularity_speed_scale,
)


ROBOT_Q_DEG = {
    "left": [-40.0, -25.0, 15.0, -55.0, 10.0, -35.0, 80.0],
    "right": [20.0, 35.0, 25.0, 60.0, 15.0, 40.0, -120.0],
}

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class FakeEnabledCommandDispatcher:
    """Observe real-mode dispatch/stop lifecycle without any socket."""

    def __init__(self):
        self.connected = True
        self.faulted = False
        self.last_reason = "fake command channels connected"
        self.last_send_monotonic = None
        self.last_stop_result = None
        self.motion_armed = False
        self.dispatch_calls = []
        self.prime_calls = []
        self.stop_calls = []

    def dispatch(self, q_commands, q_measured, **kwargs):
        """Record a fully guarded dispatch without crossing a transport."""
        self.dispatch_calls.append((q_commands, q_measured, kwargs))
        self.last_send_monotonic = kwargs["now_monotonic"]
        self.motion_armed = True
        return SimpleNamespace(
            sent=True,
            reason="fake dual-arm send succeeded",
            ack_latency_s=None,
        )

    def prime_zero_motion(self, q_commands, q_measured, **kwargs):
        """Record PRIME separately while retaining normal send behavior."""
        self.prime_calls.append((q_commands, q_measured, kwargs))
        for side in ("left", "right"):
            assert np.array_equal(q_commands[side], q_measured[side])
        return self.dispatch(q_commands, q_measured, **kwargs)

    def request_stop(self, stop_class, reason, requested_monotonic):
        """Record the software-stop request and return two fake ACKs."""
        self.stop_calls.append((stop_class, reason, requested_monotonic))
        request = SimpleNamespace(
            stop_class=StopClass(stop_class),
            reason=str(reason),
            requested_monotonic=float(requested_monotonic),
        )
        arms = tuple(
            SimpleNamespace(
                side=side,
                attempted=True,
                acknowledged=True,
                ack_latency_s=0.001,
                error=None,
            )
            for side in ("left", "right")
        )
        result = SimpleNamespace(
            request=request,
            arms=arms,
            dry_run=False,
            all_acknowledged=True,
        )
        self.last_stop_result = result
        self.motion_armed = False
        return result

    def disarm(self):
        """Match the production dispatcher's epoch reset behavior."""
        self.last_send_monotonic = None
        self.motion_armed = False

    def close(self):
        """Close only the fake lifecycle; no transport exists."""
        self.connected = False
        self.disarm()

    def latch_transport_fault(self, reason):
        """Expose an unexpected fault as a failed testable fake channel."""
        self.faulted = True
        self.connected = False
        self.last_reason = str(reason)


@pytest.fixture
def fusion_node(tmp_path, monkeypatch):
    """Create the dry-run fusion node without spinning its timer."""
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    if not rclpy.ok():
        rclpy.init()
    node = QuestDualIKFusion()
    node.timer.cancel()
    node.robot_state_timeout_s = 2.0
    node.collision_protection_enabled = False
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def demo_collision_fusion_node(tmp_path, monkeypatch):
    """Create a dry-run fusion node with the explicit demo profile."""
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    if not rclpy.ok():
        rclpy.init(
            args=[
                "--ros-args",
                "--params-file",
                str(CONFIG_DIR / "demo_collision_profile.yaml"),
            ]
        )
    node = QuestDualIKFusion()
    node.timer.cancel()
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def robot_joint_state(node, side):
    """Build a valid stationary measured-state message."""
    return JointState(
        name=node.expected_robot_joint_names(side),
        position=np.deg2rad(ROBOT_Q_DEG[side]).tolist(),
    )


def pose(x, y, z):
    """Build a finite identity-orientation controller pose."""
    message = PoseStamped()
    message.pose.position.x = x
    message.pose.position.y = y
    message.pose.position.z = z
    message.pose.orientation.w = 1.0
    return message


def grip(node, side, value):
    """Deliver one analog grip callback directly."""
    node.grip_callback(side, Float32(data=value))


def collision_snapshot(node, values):
    """Deliver one atomic collision-distance report in the fixed order."""
    node.collision_distance_callback(
        Float64MultiArray(data=list(values))
    )


def enable_fake_command_mode(node):
    """Enable actuator integration paths while retaining zero network I/O."""
    dispatcher = FakeEnabledCommandDispatcher()
    node.enable_robot_motion = True
    node.robot_command_dispatcher = dispatcher
    node.safety_supervisor.require_actuator_safety = True
    node.robot_command_hold_required = False
    node.robot_command_transport_fault = False
    node.refresh_actuator_safety(time.perf_counter())
    return dispatcher


def command_allowed_decision():
    """Build the minimum immutable gate result used by direct dispatch tests."""
    return SimpleNamespace(
        command_allowed=True,
        state=SafetyState.ENGAGED,
        reason="all safety guards satisfied",
    )


def prepare_enabled_engagement_for_prime(node):
    """Reach anchored ENGAGED using only callbacks and a fake actuator."""
    initialize_ready_node(node)
    node.collision_protection_enabled = True
    collision_snapshot(node, [0.30, 0.30, 0.30])
    dispatcher = enable_fake_command_mode(node)
    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    assert node.safety_supervisor.state == SafetyState.ENGAGED
    node.left_pose_callback(pose(0.0, 0.0, 0.0))
    node.right_pose_callback(pose(0.0, 0.0, 0.0))
    assert all(state.anchored for state in node.arms.values())
    return dispatcher


def test_fusion_node_defaults_to_network_disconnected_dry_run(fusion_node):
    """Keep the integrated actuator boundary unreachable by default."""
    node = fusion_node

    assert node.enable_robot_motion is False
    assert node.safety_supervisor.require_actuator_safety is False
    assert node.robot_command_dispatcher.movej_response_mode == "send_only"
    assert node.robot_command_dispatcher.feedback_supervised
    assert node.safety_supervisor.require_following_safety
    assert set(node.following_monitors) == {"left", "right"}
    for command_client in node.robot_command_dispatcher.clients.values():
        assert command_client.movej_response_timeout_s == pytest.approx(0.05)
        assert command_client.stop_response_timeout_s == pytest.approx(0.01)
    assert not node.robot_command_dispatcher.connected
    assert all(
        not command_client.connected
        for command_client in node.robot_command_dispatcher.clients.values()
    )

    decision = node.update_safety_supervisor(time.perf_counter())
    assert node.dispatch_robot_commands(
        decision,
        time.perf_counter(),
    ) is None
    assert not node.robot_command_dispatcher.connected


def test_dispatch_awaits_both_first_epoch_commands_without_hold(
    fusion_node,
    monkeypatch,
):
    """Treat two missing bootstrap commands as bounded normal lifecycle."""
    node = fusion_node
    initialize_ready_node(node)
    dispatcher = enable_fake_command_mode(node)
    node.safety_supervisor.state = SafetyState.ENGAGED
    node.robot_command_gate_open_since = 100.0
    for state in node.arms.values():
        state.last_safe_command_time = None
    published_status = []
    monkeypatch.setattr(
        node,
        "robot_command_status_pub",
        SimpleNamespace(publish=published_status.append),
    )

    result = node.dispatch_robot_commands(
        command_allowed_decision(),
        100.01,
    )

    assert result is None
    assert not node.robot_command_hold_required
    assert not dispatcher.dispatch_calls
    assert not dispatcher.stop_calls
    assert node.last_robot_command_status["status"] == (
        "AWAITING_FIRST_SAFE_COMMAND"
    )
    assert node.last_robot_command_status[
        "current_engagement_command_ready"
    ] == {"left": False, "right": False}
    assert json.loads(published_status[-1].data) == (
        node.last_robot_command_status
    )


def test_dispatch_awaits_when_only_left_command_is_current(fusion_node):
    """Never split a dual-arm dispatch while right remains in bootstrap."""
    node = fusion_node
    initialize_ready_node(node)
    dispatcher = enable_fake_command_mode(node)
    node.safety_supervisor.state = SafetyState.ENGAGED
    node.robot_command_gate_open_since = 200.0
    node.arms["left"].last_safe_command_time = 200.001
    node.arms["right"].last_safe_command_time = None

    result = node.dispatch_robot_commands(
        command_allowed_decision(),
        200.01,
    )

    assert result is None
    assert not node.robot_command_hold_required
    assert not dispatcher.dispatch_calls
    assert not dispatcher.stop_calls
    assert node.last_robot_command_status[
        "current_engagement_command_ready"
    ] == {"left": True, "right": False}


def test_dispatch_rejects_previous_engagement_timestamps_without_hold(
    fusion_node,
):
    """Old dual commands remain awaiting rather than crossing a new epoch."""
    node = fusion_node
    initialize_ready_node(node)
    dispatcher = enable_fake_command_mode(node)
    node.safety_supervisor.state = SafetyState.ENGAGED
    node.robot_command_gate_open_since = 300.0
    for state in node.arms.values():
        state.last_safe_command_time = 290.0

    result = node.dispatch_robot_commands(
        command_allowed_decision(),
        300.01,
    )

    assert result is None
    assert not node.robot_command_hold_required
    assert not dispatcher.dispatch_calls
    assert not dispatcher.stop_calls
    assert "awaiting first current-engagement" in (
        node.last_robot_command_status["reason"]
    )
    assert node.last_robot_command_status[
        "current_engagement_command_ready"
    ] == {"left": False, "right": False}


def test_dispatch_first_dual_current_engagement_command_is_prime(fusion_node):
    """Make the first current-epoch dispatch the strict zero-motion PRIME."""
    node = fusion_node
    initialize_ready_node(node)
    dispatcher = enable_fake_command_mode(node)
    node.safety_supervisor.state = SafetyState.ENGAGED
    node.robot_command_gate_open_since = 400.0
    node.actuator_prime_prepared = True
    node.arms["left"].last_safe_command_time = 400.001
    node.arms["right"].last_safe_command_time = 400.002
    for state in node.arms.values():
        state.last_safe_command_dt_s = 0.02

    result = node.dispatch_robot_commands(
        command_allowed_decision(),
        400.01,
    )

    assert result.sent
    assert node.actuator_primed
    assert len(dispatcher.prime_calls) == 1
    assert len(dispatcher.dispatch_calls) == 1
    assert dispatcher.dispatch_calls[0][2][
        "generated_monotonic"
    ] == pytest.approx(400.001)
    assert node.last_robot_command_status["status"] == "PRIMED"
    assert "zero-motion actuator prime sent" in (
        node.last_robot_command_status["reason"]
    )
    assert not node.last_robot_command_status["actuator_prime_prepared"]
    assert node.last_robot_command_status["actuator_primed"]
    assert node.last_robot_command_status["command_dt_s"] == pytest.approx(
        0.02
    )
    assert node.last_robot_command_status[
        "current_engagement_command_ready"
    ] == {"left": True, "right": True}

    for state in node.arms.values():
        state.last_safe_command_time = 400.02
    normal_result = node.dispatch_robot_commands(
        command_allowed_decision(),
        400.03,
    )
    assert normal_result.sent
    assert len(dispatcher.prime_calls) == 1
    assert len(dispatcher.dispatch_calls) == 2
    assert node.last_robot_command_status["status"] == "SENT"


def test_current_commands_wait_in_actuator_priming_until_prepared(fusion_node):
    """Expose ACTUATOR_PRIMING without sending a non-PRIME first target."""
    node = fusion_node
    initialize_ready_node(node)
    dispatcher = enable_fake_command_mode(node)
    node.safety_supervisor.state = SafetyState.ENGAGED
    node.robot_command_gate_open_since = 450.0
    for state in node.arms.values():
        state.last_safe_command_time = 450.001

    result = node.dispatch_robot_commands(
        command_allowed_decision(),
        450.01,
    )

    assert result is None
    assert not dispatcher.dispatch_calls
    assert node.last_robot_command_status["status"] == "ACTUATOR_PRIMING"
    assert not node.last_robot_command_status["actuator_prime_prepared"]
    assert not node.last_robot_command_status["actuator_primed"]


def test_dual_arm_command_rejects_mismatched_canonical_dt(fusion_node):
    """Require both arm targets to originate in one control interval."""
    node = fusion_node
    initialize_ready_node(node)
    dispatcher = enable_fake_command_mode(node)
    engagement_start = time.perf_counter()
    node.safety_supervisor.state = SafetyState.ENGAGED
    node.deadman_active = True
    node.robot_command_gate_open_since = engagement_start
    node.actuator_primed = True
    for state in node.arms.values():
        state.last_safe_command_time = engagement_start + 0.001
    node.arms["left"].last_safe_command_dt_s = 0.020
    node.arms["right"].last_safe_command_dt_s = 0.019

    result = node.dispatch_robot_commands(
        command_allowed_decision(),
        engagement_start + 0.002,
    )

    assert result is None
    assert not dispatcher.dispatch_calls
    assert node.robot_command_hold_required
    assert "canonical command_dt_s" in node.robot_command_hold_reason


def test_bootstrap_timeout_still_holds_and_requests_safety_stop(fusion_node):
    """Bound bootstrap waiting with the unchanged actuator watchdog."""
    node = fusion_node
    initialize_ready_node(node)
    dispatcher = enable_fake_command_mode(node)
    dispatcher.last_send_monotonic = time.perf_counter() - 10.0
    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    engagement_start = node.robot_command_gate_open_since
    assert node.safety_supervisor.state == SafetyState.ENGAGED
    assert engagement_start is not None
    assert dispatcher.last_send_monotonic is None

    decision = node.update_safety_supervisor(
        engagement_start + node.command_timeout_s + 0.001
    )

    assert decision.state == SafetyState.HOLD
    assert node.robot_command_hold_required
    assert node.robot_command_hold_reason == (
        "real-robot command output watchdog expired"
    )
    assert len(dispatcher.stop_calls) == 1
    assert dispatcher.stop_calls[0][0] == StopClass.SAFETY_STOP
    assert not dispatcher.dispatch_calls


def test_post_first_dispatch_stale_watchdog_remains_active(fusion_node):
    """Continue guarding output cadence after bootstrap has completed."""
    node = fusion_node
    initialize_ready_node(node)
    dispatcher = enable_fake_command_mode(node)
    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    engagement_start = node.robot_command_gate_open_since
    first_command_time = engagement_start + 0.001
    node.actuator_prime_prepared = True
    for state in node.arms.values():
        state.last_safe_command_time = first_command_time
        state.last_safe_command_dt_s = 0.02

    result = node.dispatch_robot_commands(
        command_allowed_decision(),
        first_command_time + 0.001,
    )
    assert result.sent
    assert node.actuator_primed
    first_send_time = dispatcher.last_send_monotonic

    decision = node.update_safety_supervisor(
        first_send_time + node.command_timeout_s + 0.001
    )

    assert decision.state == SafetyState.HOLD
    assert node.robot_command_hold_reason == (
        "real-robot command output watchdog expired"
    )
    assert len(dispatcher.stop_calls) == 1
    assert dispatcher.stop_calls[0][0] == StopClass.SAFETY_STOP


def test_control_cycle_primes_exact_latest_measured_without_ik_progress(
    fusion_node,
    monkeypatch,
):
    """Freeze both fusion histories on the latest feedback for PRIME."""
    node = fusion_node
    dispatcher = prepare_enabled_engagement_for_prime(node)
    measured_at_prime = {}
    for index, (side, state) in enumerate(node.arms.items(), start=1):
        updated_q = state.q_measured.copy()
        updated_q[0] += index * 1e-6
        state.update_measured_q(updated_q, time.perf_counter())
        measured_at_prime[side] = updated_q.copy()

    update_arm_calls = []
    monkeypatch.setattr(
        node,
        "update_arm",
        lambda **kwargs: update_arm_calls.append(kwargs),
    )

    node.control_update()

    assert not update_arm_calls
    assert node.actuator_primed
    assert not node.actuator_prime_prepared
    assert len(dispatcher.prime_calls) == 1
    q_commands, q_measured, kwargs = dispatcher.prime_calls[0]
    assert kwargs["command_dt_s"] == pytest.approx(node.control_period_s)
    for side, state in node.arms.items():
        assert np.array_equal(q_commands[side], measured_at_prime[side])
        assert np.array_equal(q_measured[side], measured_at_prime[side])
        assert np.array_equal(state.q_safe, measured_at_prime[side])
        assert np.array_equal(state.q_command, measured_at_prime[side])
        assert np.array_equal(state.joint_velocity, np.zeros(7))
        assert np.array_equal(state.joint_acceleration, np.zeros(7))
        assert state.last_safe_command_dt_s == pytest.approx(
            node.control_period_s
        )
    assert node.last_robot_command_status["status"] == "PRIMED"


def test_hold_then_reengage_requires_a_new_actuator_prime(
    fusion_node,
    monkeypatch,
):
    """Clear PRIME and continuity on exit from every engagement epoch."""
    node = fusion_node
    dispatcher = prepare_enabled_engagement_for_prime(node)
    monkeypatch.setattr(node, "update_arm", lambda **_kwargs: None)
    node.control_update()
    assert node.actuator_primed
    assert len(dispatcher.prime_calls) == 1

    grip(node, "left", 0.0)
    assert node.safety_supervisor.state == SafetyState.HOLD
    assert not node.actuator_primed
    assert not node.actuator_prime_prepared
    grip(node, "right", 0.0)
    assert node.update_safety_supervisor(
        time.perf_counter()
    ).state == SafetyState.READY

    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    node.left_pose_callback(pose(0.2, 0.0, 0.0))
    node.right_pose_callback(pose(-0.2, 0.0, 0.0))
    assert node.update_safety_supervisor(
        time.perf_counter()
    ).state == SafetyState.ENGAGED
    node.left_pose_callback(pose(0.2, 0.0, 0.0))
    node.right_pose_callback(pose(-0.2, 0.0, 0.0))
    assert all(state.anchored for state in node.arms.values()), {
        side: {
            "anchored": state.anchored,
            "tracking": state.tracking_valid,
            "robot_ready": state.robot_state_ready(
                node.robot_state_timeout_s,
                time.perf_counter(),
            ),
            "soft": state.q_within_soft_limits(state.q_measured),
        }
        for side, state in node.arms.items()
    }
    collision_snapshot(node, [0.30, 0.30, 0.30])
    node.control_update()

    assert node.actuator_primed
    assert len(dispatcher.prime_calls) == 2


def test_explicit_demo_profile_resolves_category_thresholds(
    demo_collision_fusion_node,
):
    """Load demo YAML through ROS and expose its resolved safety decision."""
    node = demo_collision_fusion_node
    assert node.enable_robot_motion is False
    assert not node.robot_command_dispatcher.connected
    node.collision_monitor.update_snapshot(
        {
            "left_self": 0.053036446398235126,
            "right_self": 0.05301525627829068,
            "inter_arm": 0.13789999956812155,
        },
        received_monotonic=20.0,
    )

    decision = node.collision_monitor.evaluate(20.0)
    diagnostics = node.collision_monitor.category_diagnostics()
    assert decision.region.value == "warning"
    assert not decision.hold_required
    assert decision.limiting_source.value == "right_self"
    assert decision.speed_scale == pytest.approx(0.3530988, abs=1e-6)
    assert diagnostics["left_self"]["stop_distance_m"] == pytest.approx(
        0.045
    )
    assert diagnostics["right_self"]["warn_distance_m"] == pytest.approx(
        0.065
    )
    assert diagnostics["inter_arm"]["stop_distance_m"] == pytest.approx(
        0.05
    )
    assert diagnostics["inter_arm"]["warn_distance_m"] == pytest.approx(
        0.15
    )
    assert diagnostics["environment"]["status"] == (
        "DISABLED_BY_CONFIGURATION"
    )


def initialize_ready_node(node):
    """Supply read-only robot state and fresh controller poses."""
    for side in ("left", "right"):
        node.robot_connected_callback(side, Bool(data=True))
        node.robot_stale_callback(side, Bool(data=False))
        node.robot_enabled_callback(side, Bool(data=True))
        node.robot_fault_callback(side, Bool(data=False))
        node.robot_joint_state_callback(side, robot_joint_state(node, side))

    node.left_tracking_callback(Bool(data=True))
    node.right_tracking_callback(Bool(data=True))
    node.left_pose_callback(pose(0.0, 0.0, 0.0))
    node.right_pose_callback(pose(0.0, 0.0, 0.0))
    decision = node.update_safety_supervisor(time.perf_counter())
    assert decision.state == SafetyState.READY

    node.quest_input_fresh_callback(Bool(data=True))
    grip(node, "left", 0.0)
    grip(node, "right", 0.0)
    assert not node.deadman_clutch.rearm_required


def engage_and_anchor(node):
    """Enter ENGAGED and capture fresh coincident controller anchors."""
    initialize_ready_node(node)
    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    assert node.safety_supervisor.state == SafetyState.ENGAGED
    node.left_pose_callback(pose(0.0, 0.0, 0.0))
    node.right_pose_callback(pose(0.0, 0.0, 0.0))
    assert all(state.anchored for state in node.arms.values())


def engage_with_historical_commands(node):
    """Enter a new epoch while retaining commands from an earlier epoch."""
    initialize_ready_node(node)
    historical_time = time.perf_counter() - 10.0
    for state in node.arms.values():
        state.last_safe_command_time = historical_time
    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    assert node.safety_supervisor.state == SafetyState.ENGAGED
    assert node.robot_command_gate_open_since is not None
    return historical_time, node.robot_command_gate_open_since


def test_following_error_waits_for_each_arm_current_epoch_command(
    fusion_node,
    monkeypatch,
):
    """Track left/right post-engagement command admission independently."""
    node = fusion_node
    historical_time, engagement_start = engage_with_historical_commands(node)
    now = time.perf_counter()
    for state in node.arms.values():
        state.last_robot_state_rx_time = now
    node.arms["left"].last_safe_command_time = historical_time
    node.arms["right"].last_safe_command_time = engagement_start
    published = []
    monkeypatch.setattr(
        node,
        "following_error_pub",
        SimpleNamespace(publish=published.append),
    )

    node.refresh_following_safety(now)

    left = node.last_following_decisions["left"]
    right = node.last_following_decisions["right"]
    assert left.ready and not left.hold_required
    assert not left.command_is_current_engagement
    assert left.reason == "awaiting first post-engagement safe command"
    assert right.ready and not right.hold_required
    assert right.command_is_current_engagement
    assert right.error_rad is not None
    assert node.safety_supervisor.following_ready
    assert not node.safety_supervisor.following_hold_required
    assert node.safety_supervisor.following_reason.startswith("left awaiting")
    diagnostics = json.loads(published[-1].data)
    required_fields = {
        "engagement_start_time",
        "engagement_age",
        "command_time",
        "command_age_s",
        "measurement_age_s",
        "timestamp_skew_s",
        "command_is_current_engagement",
        "reason",
    }
    assert required_fields <= set(diagnostics["left"])
    assert diagnostics["left"]["engagement_start_time"] == pytest.approx(
        engagement_start
    )
    assert not diagnostics["left"]["command_is_current_engagement"]
    assert diagnostics["right"]["command_is_current_engagement"]

    node.arms["left"].last_safe_command_time = engagement_start
    node.arms["right"].last_safe_command_time = historical_time
    node.refresh_following_safety(now)

    assert node.last_following_decisions[
        "left"
    ].command_is_current_engagement
    assert not node.last_following_decisions[
        "right"
    ].command_is_current_engagement
    assert node.safety_supervisor.following_reason.startswith("right awaiting")


def test_reengagement_does_not_reuse_previous_epoch_commands(fusion_node):
    """Require new commands after ENGAGED to HOLD/READY to ENGAGED."""
    node = fusion_node
    _, first_engagement = engage_with_historical_commands(node)
    first_command_time = time.perf_counter()
    for state in node.arms.values():
        state.last_safe_command_time = first_command_time
        state.last_robot_state_rx_time = first_command_time
    node.refresh_following_safety(first_command_time)
    assert all(
        item.command_is_current_engagement
        for item in node.last_following_decisions.values()
    )

    grip(node, "left", 0.0)
    assert node.safety_supervisor.state == SafetyState.HOLD
    grip(node, "right", 0.0)
    decision = node.update_safety_supervisor(time.perf_counter())
    assert decision.state == SafetyState.READY

    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    node.left_pose_callback(pose(0.0, 0.0, 0.0))
    node.right_pose_callback(pose(0.0, 0.0, 0.0))
    decision = node.update_safety_supervisor(time.perf_counter())
    assert decision.state == SafetyState.ENGAGED
    second_engagement = node.robot_command_gate_open_since
    assert second_engagement > first_engagement
    assert first_command_time < second_engagement

    now = time.perf_counter()
    for state in node.arms.values():
        state.last_robot_state_rx_time = now
    decision = node.update_safety_supervisor(now)

    assert decision.state == SafetyState.ENGAGED
    assert all(
        item.reason == "awaiting first post-engagement safe command"
        for item in node.last_following_decisions.values()
    )
    assert not node.safety_supervisor.following_hold_required


def test_release_move_repress_captures_new_coincident_anchors(fusion_node):
    """Motion while released must never be replayed after re-engagement."""
    node = fusion_node
    initialize_ready_node(node)

    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    assert node.deadman_active
    assert node.safety_supervisor.state == SafetyState.ENGAGED

    first_left = pose(0.10, 0.20, 0.30)
    first_right = pose(-0.10, 0.15, 0.25)
    node.left_pose_callback(first_left)
    node.right_pose_callback(first_right)
    assert all(state.anchored for state in node.arms.values())

    grip(node, "left", 0.0)
    assert not node.deadman_active
    assert node.safety_supervisor.state == SafetyState.HOLD
    assert not any(state.anchored for state in node.arms.values())

    moved_left = pose(0.80, -0.40, 1.10)
    moved_right = pose(-0.70, 0.60, 0.90)
    node.left_pose_callback(moved_left)
    node.right_pose_callback(moved_right)
    decision = node.update_safety_supervisor(time.perf_counter())
    assert decision.state == SafetyState.READY

    grip(node, "left", 0.8)
    assert node.safety_supervisor.state == SafetyState.ENGAGED
    node.left_pose_callback(moved_left)
    node.right_pose_callback(moved_right)

    expected_positions = {
        "left": np.array([0.80, -0.40, 1.10]),
        "right": np.array([-0.70, 0.60, 0.90]),
    }
    for side, state in node.arms.items():
        assert state.anchored
        assert np.allclose(
            state.T_vr_anchor[:3, 3],
            expected_positions[side],
        )
        assert np.allclose(state.T_vr_latest, state.T_vr_anchor)
        assert np.allclose(state.T_ee_anchor, state.T_measured)
        assert np.allclose(
            np.linalg.inv(state.T_vr_anchor) @ state.T_vr_latest,
            np.eye(4),
        )


def test_source_gap_requires_release_before_reengagement(fusion_node):
    """Cached held grips cannot automatically recover from source loss."""
    node = fusion_node
    initialize_ready_node(node)
    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    assert node.safety_supervisor.state == SafetyState.ENGAGED

    node.quest_input_fresh_callback(Bool(data=False))
    assert not node.deadman_active
    assert node.deadman_clutch.rearm_required
    assert node.safety_supervisor.state == SafetyState.HOLD

    node.quest_input_fresh_callback(Bool(data=True))
    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    assert not node.deadman_active
    assert node.deadman_clutch.rearm_required

    grip(node, "left", 0.0)
    grip(node, "right", 0.0)
    assert not node.deadman_clutch.rearm_required
    decision = node.update_safety_supervisor(time.perf_counter())
    assert decision.state == SafetyState.READY

    grip(node, "left", 0.8)
    grip(node, "right", 0.8)
    assert node.deadman_active
    assert node.safety_supervisor.state == SafetyState.READY

    recovered_left = pose(0.45, -0.20, 0.70)
    recovered_right = pose(-0.35, 0.30, 0.65)
    node.left_pose_callback(recovered_left)
    node.right_pose_callback(recovered_right)
    decision = node.update_safety_supervisor(time.perf_counter())
    assert decision.state == SafetyState.ENGAGED

    node.left_pose_callback(recovered_left)
    node.right_pose_callback(recovered_right)
    assert all(state.anchored for state in node.arms.values())


def test_control_dt_uses_measured_period_without_late_jump_growth(fusion_node):
    """Use short actual cycles but cap delayed cycles at the nominal period."""
    node = fusion_node
    assert node.next_joint_limit_dt(10.0) == pytest.approx(0.02)
    assert node.next_joint_limit_dt(10.005) == pytest.approx(0.005)
    assert node.next_joint_limit_dt(10.100) == pytest.approx(0.02)
    assert node.next_joint_limit_dt(10.100) == 0.0


def test_collision_stop_holds_both_arms_and_invalidates_anchors(fusion_node):
    """Propagate any global collision stop through the dual-arm gate."""
    node = fusion_node
    node.collision_protection_enabled = True
    collision_snapshot(node, [0.30, 0.30, 0.30])
    engage_and_anchor(node)

    collision_snapshot(node, [0.30, 0.30, 0.04])

    assert node.safety_supervisor.state == SafetyState.HOLD
    assert not node.safety_supervisor.evaluate(
        True, time.perf_counter()
    ).command_allowed
    assert "inter_arm" in node.safety_supervisor.last_reason
    assert not any(state.anchored for state in node.arms.values())


def test_malformed_collision_snapshot_holds_both_arms(fusion_node):
    """Never reuse an old clear snapshot after a partial ROS report."""
    node = fusion_node
    node.collision_protection_enabled = True
    collision_snapshot(node, [0.30, 0.30, 0.30])
    engage_and_anchor(node)

    collision_snapshot(node, [0.30, 0.30])

    assert node.safety_supervisor.state == SafetyState.HOLD
    assert "exactly 3 values" in node.safety_supervisor.last_reason
    assert not any(state.anchored for state in node.arms.values())


def test_collision_warning_scales_cartesian_and_joint_rates(
    fusion_node,
    monkeypatch,
):
    """Apply one collision warning scale before both command limiters."""
    node = fusion_node
    node.collision_protection_enabled = True
    collision_snapshot(node, [0.30, 0.30, 0.10])
    engage_and_anchor(node)
    state = node.arms["left"]
    captured = {}
    real_limit_pose_step = fusion_module.limit_pose_step
    real_limit_joint_velocity = fusion_module.limit_joint_velocity

    def record_limit_pose_step(**kwargs):
        captured["translation_step"] = kwargs["max_translation_step"]
        captured["rotation_step"] = kwargs["max_rotation_step"]
        return real_limit_pose_step(**kwargs)

    def record_limit_joint_velocity(**kwargs):
        captured["qd_limit"] = np.asarray(kwargs["qd_limit"]).copy()
        return real_limit_joint_velocity(**kwargs)

    def projected_hold(**kwargs):
        q_safe = kwargs["q_safe"].copy()
        return {
            "success": True,
            "projected": False,
            "alpha": 1.0,
            "T_safe": forward_kinematics(q_safe, model=state.model),
            "q_safe": q_safe,
            "sigma_min": minimum_singular_value(q_safe, state.model),
            "raw_ik_success": True,
        }

    monkeypatch.setattr(
        fusion_module,
        "limit_pose_step",
        record_limit_pose_step,
    )
    monkeypatch.setattr(
        fusion_module,
        "limit_joint_velocity",
        record_limit_joint_velocity,
    )
    monkeypatch.setattr(
        fusion_module,
        "project_target_to_feasible",
        projected_hold,
    )

    node.left_pose_callback(pose(0.10, 0.0, 0.0))
    sigma_scale = singularity_speed_scale(
        minimum_singular_value(state.q_safe, state.model),
        sigma_stop=node.sigma_stop,
        sigma_warn=node.sigma_warn,
    )
    stamp = node.get_clock().now().to_msg()
    node.update_arm(state, stamp, joint_limit_dt_s=0.02)

    collision_scale = 0.5
    assert node.safety_supervisor.state == SafetyState.ENGAGED
    assert node.safety_supervisor.collision_speed_scale == pytest.approx(
        collision_scale
    )
    assert captured["translation_step"] == pytest.approx(
        node.max_cartesian_translation_rate
        * 0.02
        * sigma_scale
        * collision_scale
    )
    assert captured["rotation_step"] == pytest.approx(
        node.max_cartesian_rotation_rate
        * 0.02
        * sigma_scale
        * collision_scale
    )
    assert np.allclose(
        captured["qd_limit"],
        state.joint_velocity_limit * collision_scale,
    )


@pytest.mark.parametrize(
    "sigma_min,expected_scale,expected_region",
    [
        (0.030, 1.0, "safe"),
        (0.015, 0.5, "warning"),
        (0.010, 0.0, "stop"),
    ],
)
def test_online_cartesian_rate_uses_smooth_singularity_scale(
    fusion_node,
    monkeypatch,
    sigma_min,
    expected_scale,
    expected_region,
):
    """Apply one sigma scale to both Cartesian rate budgets before IK."""
    node = fusion_node
    engage_and_anchor(node)
    state = node.arms["left"]
    q_before = state.q_safe.copy()
    captured = {}
    real_limit_pose_step = fusion_module.limit_pose_step

    def record_limit_pose_step(**kwargs):
        captured["translation_step"] = kwargs["max_translation_step"]
        captured["rotation_step"] = kwargs["max_rotation_step"]
        return real_limit_pose_step(**kwargs)

    def projected_hold(**_kwargs):
        return {
            "success": True,
            "projected": False,
            "alpha": 1.0,
            "T_safe": state.T_safe.copy(),
            "q_safe": q_before.copy(),
            "sigma_min": sigma_min,
            "raw_ik_success": True,
        }

    monkeypatch.setattr(
        fusion_module,
        "limit_pose_step",
        record_limit_pose_step,
    )
    monkeypatch.setattr(
        fusion_module,
        "project_target_to_feasible",
        projected_hold,
    )
    monkeypatch.setattr(
        fusion_module,
        "minimum_singular_value",
        lambda _q, _model: sigma_min,
    )

    node.left_pose_callback(pose(0.10, 0.0, 0.0))
    stamp = node.get_clock().now().to_msg()
    node.update_arm(state, stamp, joint_limit_dt_s=0.02)

    assert captured["translation_step"] == pytest.approx(
        node.max_cartesian_translation_rate * 0.02 * expected_scale
    )
    assert captured["rotation_step"] == pytest.approx(
        node.max_cartesian_rotation_rate * 0.02 * expected_scale
    )
    assert state.last_singularity_speed_scale == pytest.approx(expected_scale)
    assert state.last_singularity_region == expected_region
    assert state.last_result["singularity_speed_scale"] == pytest.approx(
        expected_scale
    )


def test_online_candidate_is_rate_limited_then_fk_and_sigma_recomputed(
    fusion_node,
    monkeypatch,
):
    """Make q_safe/T_safe/sigma describe the limited command, not IK output."""
    node = fusion_node
    engage_and_anchor(node)
    state = node.arms["left"]
    q_before = state.q_safe.copy()
    q_candidate = q_before + np.deg2rad(np.full(7, 10.0))
    candidate_sigma = minimum_singular_value(q_candidate, state.model)

    def projected_candidate(**_kwargs):
        return {
            "success": True,
            "projected": False,
            "alpha": 1.0,
            "T_safe": forward_kinematics(q_candidate, model=state.model),
            "q_safe": q_candidate.copy(),
            "sigma_min": candidate_sigma,
            "raw_ik_success": True,
        }

    monkeypatch.setattr(
        fusion_module,
        "project_target_to_feasible",
        projected_candidate,
    )
    stamp = node.get_clock().now().to_msg()
    node.update_arm(state, stamp, joint_limit_dt_s=0.02)

    expected_velocity = state.joint_acceleration_limit * 0.02
    expected_delta = expected_velocity * 0.02
    expected_command = q_before + expected_delta
    expected_transform = forward_kinematics(
        expected_command,
        model=state.model,
    )
    expected_sigma = minimum_singular_value(expected_command, state.model)

    assert np.allclose(state.q_candidate, q_candidate)
    assert np.allclose(state.q_command, expected_command)
    assert np.allclose(state.q_safe, expected_command)
    assert np.allclose(state.joint_velocity, expected_velocity)
    assert np.allclose(
        state.joint_acceleration,
        state.joint_acceleration_limit,
    )
    assert np.allclose(state.T_safe, expected_transform)
    assert state.last_sigma_min == pytest.approx(expected_sigma)
    assert state.last_result["candidate_sigma_min"] == pytest.approx(
        candidate_sigma
    )
    assert state.last_result["sigma_min"] == pytest.approx(expected_sigma)
    assert state.last_result["joint_rate_limited"] is False
    assert state.last_result["joint_acceleration_limited"] is True
    assert state.last_result["joint_limit_dt_s"] == pytest.approx(0.02)
    assert state.last_safe_command_dt_s == pytest.approx(0.02)


def test_unsafe_limited_intermediate_configuration_holds_previous_safe_state(
    fusion_node,
    monkeypatch,
):
    """Reject a rate-limited step whose recomputed command sigma is unsafe."""
    node = fusion_node
    engage_and_anchor(node)
    state = node.arms["right"]
    q_before = state.q_safe.copy()
    T_before = state.T_safe.copy()
    q_candidate = q_before + np.deg2rad(np.full(7, 10.0))
    q_limited = (
        q_before
        + state.joint_acceleration_limit * 0.02 ** 2
    )

    def projected_candidate(**_kwargs):
        return {
            "success": True,
            "projected": False,
            "alpha": 1.0,
            "T_safe": forward_kinematics(q_candidate, model=state.model),
            "q_safe": q_candidate.copy(),
            "sigma_min": 0.03,
            "raw_ik_success": True,
        }

    def synthetic_sigma(q, _model):
        if np.allclose(q, q_limited):
            return 0.005
        if np.allclose(q, q_before):
            return 0.04
        raise AssertionError("unexpected sigma evaluation configuration")

    monkeypatch.setattr(
        fusion_module,
        "project_target_to_feasible",
        projected_candidate,
    )
    monkeypatch.setattr(
        fusion_module,
        "minimum_singular_value",
        synthetic_sigma,
    )
    stamp = node.get_clock().now().to_msg()
    node.update_arm(state, stamp, joint_limit_dt_s=0.02)

    assert np.allclose(state.q_candidate, q_candidate)
    assert np.array_equal(state.q_command, q_before)
    assert np.array_equal(state.q_safe, q_before)
    assert np.array_equal(state.T_safe, T_before)
    assert np.array_equal(state.joint_velocity, np.zeros(7))
    assert state.singularity_hold
    assert state.last_sigma_min == pytest.approx(0.04)
    assert state.last_result["limited_sigma_min"] == pytest.approx(0.005)
    assert state.last_result["sigma_min"] == pytest.approx(0.04)


def test_control_update_rate_limits_both_arms_with_one_cycle_period(
    fusion_node,
    monkeypatch,
):
    """Apply the same conservative timer-cycle dt to both arms."""
    node = fusion_node
    engage_and_anchor(node)
    q_before = {
        side: state.q_safe.copy()
        for side, state in node.arms.items()
    }

    def projected_candidate(**kwargs):
        model = kwargs["model"]
        q_candidate = kwargs["q_safe"] + np.deg2rad(np.full(7, 10.0))
        return {
            "success": True,
            "projected": False,
            "alpha": 1.0,
            "T_safe": forward_kinematics(q_candidate, model=model),
            "q_safe": q_candidate,
            "sigma_min": minimum_singular_value(q_candidate, model),
            "raw_ik_success": True,
        }

    monkeypatch.setattr(
        fusion_module,
        "project_target_to_feasible",
        projected_candidate,
    )
    node.control_update()

    for side, state in node.arms.items():
        expected_velocity = (
            state.joint_acceleration_limit
            * node.control_period_s
        )
        expected_delta = expected_velocity * node.control_period_s
        assert np.allclose(state.q_safe, q_before[side] + expected_delta)
        assert state.last_joint_limit_dt_s == pytest.approx(
            node.control_period_s
        )
        assert state.last_safe_command_dt_s == pytest.approx(
            node.control_period_s
        )
        assert np.all(
            np.abs(state.joint_velocity)
            <= state.joint_velocity_limit + 1e-12
        )
        assert np.all(
            np.abs(state.joint_acceleration)
            <= state.joint_acceleration_limit + 1e-12
        )
        assert state.last_joint_acceleration_limited
    assert node.safety_supervisor.state == SafetyState.ENGAGED


@pytest.mark.parametrize(
    "side,start_j4_deg,target_j4_deg,boundary_j4_deg",
    [
        ("left", -15.1, 20.0, -15.0),
        ("right", 15.1, -20.0, 15.0),
    ],
)
def test_online_soft_limit_preserves_each_elbow_branch(
    fusion_node,
    monkeypatch,
    side,
    start_j4_deg,
    target_j4_deg,
    boundary_j4_deg,
):
    """Saturate J4 before zero even when IK requests the opposite branch."""
    node = fusion_node
    engage_and_anchor(node)
    state = node.arms[side]
    q_current = state.q_safe.copy()
    q_current[3] = np.deg2rad(start_j4_deg)
    T_current = forward_kinematics(q_current, model=state.model)
    state.q_safe = q_current.copy()
    state.q_candidate = q_current.copy()
    state.q_command = q_current.copy()
    state.joint_velocity = np.zeros(7)
    state.T_safe = T_current.copy()
    state.T_ee_anchor = T_current.copy()

    q_candidate = q_current.copy()
    q_candidate[3] = np.deg2rad(target_j4_deg)

    def projected_candidate(**_kwargs):
        return {
            "success": True,
            "projected": False,
            "alpha": 1.0,
            "T_safe": forward_kinematics(q_candidate, model=state.model),
            "q_safe": q_candidate.copy(),
            "sigma_min": minimum_singular_value(q_candidate, state.model),
            "raw_ik_success": True,
        }

    monkeypatch.setattr(
        fusion_module,
        "project_target_to_feasible",
        projected_candidate,
    )
    stamp = node.get_clock().now().to_msg()
    node.update_arm(state, stamp, joint_limit_dt_s=0.02)

    assert np.rad2deg(state.q_candidate[3]) == pytest.approx(target_j4_deg)
    command_j4_deg = np.rad2deg(state.q_command[3])
    if side == "left":
        assert start_j4_deg < command_j4_deg <= boundary_j4_deg
    else:
        assert boundary_j4_deg <= command_j4_deg < start_j4_deg
    assert command_j4_deg == pytest.approx(
        np.rad2deg(state.q_safe[3])
    )
    assert state.last_joint_soft_limited
    assert state.last_result["joint_soft_limited"] is True
    assert np.allclose(
        state.T_safe,
        forward_kinematics(state.q_command, model=state.model),
    )
    assert node.safety_supervisor.state == SafetyState.ENGAGED
