"""ROS callback integration tests for dual-grip clutch re-anchoring."""

import json
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
from vr_rm75_teleop.target_feasibility import (
    minimum_singular_value,
    singularity_speed_scale,
)


ROBOT_Q_DEG = {
    "left": [-40.0, -25.0, 15.0, -55.0, 10.0, -35.0, 80.0],
    "right": [20.0, 35.0, 25.0, 60.0, 15.0, 40.0, -120.0],
}


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


def test_fusion_node_defaults_to_network_disconnected_dry_run(fusion_node):
    """Keep the integrated actuator boundary unreachable by default."""
    node = fusion_node

    assert node.enable_robot_motion is False
    assert node.safety_supervisor.require_actuator_safety is False
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
