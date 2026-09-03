"""Offline tests for measured-state startup and VR anchor alignment."""

import numpy as np
import pytest

from vr_rm75_teleop.arm_fusion_state import ArmFusionState
from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.target_feasibility import project_target_to_feasible
from vr_rm75_teleop.vr_pose_mapping import map_vr_pose_to_robot_target
from vr_rm75_teleop.joint_safety import make_teleop_soft_limits


Q_LEFT_A = np.deg2rad([-40.0, -25.0, 15.0, -55.0, 10.0, -35.0, 80.0])
Q_LEFT_B = np.deg2rad([-35.0, -20.0, 20.0, -45.0, 5.0, -30.0, 70.0])


def make_transform(position, rotvec=(0.0, 0.0, 0.0)):
    """Build a finite test transform without ROS messages."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = position
    return T


def make_ready(state, q, now):
    """Provide fresh connected feedback and initialize the safe state."""
    state.robot_connected = True
    state.robot_reported_stale = False
    state.update_measured_q(q, now)
    assert state.initialize_from_measured(0.25, now)


def configure_soft_limits(state, elbow_margin_deg=15.0):
    """Install the same side-specific limits used by the ROS node."""
    lower, upper = make_teleop_soft_limits(
        state.model.q_min,
        state.model.q_max,
        np.deg2rad(5.0),
        elbow_index=3,
        elbow_branch=-1 if state.side == "left" else 1,
        elbow_margin=np.deg2rad(elbow_margin_deg),
    )
    state.configure_teleop_soft_limits(
        lower,
        upper,
        -1 if state.side == "left" else 1,
    )


def test_real_arm_has_no_hardcoded_safe_state_before_feedback():
    """Do not create or publish a robot command before measured feedback."""
    state = ArmFusionState("left")

    assert state.q_safe is None
    assert state.T_safe is None
    assert state.robot_state_initialized is False
    assert state.capture_vr_anchor(np.eye(4), True, 0.25, 1.0) is False


def test_arbitrary_measured_q_initializes_safe_state_exactly():
    """Set q_safe and T_safe exactly from an arbitrary valid measurement."""
    state = ArmFusionState("left")
    make_ready(state, Q_LEFT_A, 10.0)

    assert state.initialized_from_robot is True
    assert np.array_equal(state.q_start, Q_LEFT_A)
    assert np.array_equal(state.q_safe, Q_LEFT_A)
    assert np.array_equal(state.q_candidate, Q_LEFT_A)
    assert np.array_equal(state.q_command, Q_LEFT_A)
    assert np.array_equal(state.q_preferred, Q_LEFT_A)
    assert state.last_safe_command_time is None
    assert state.last_safe_command_dt_s is None
    assert np.allclose(
        state.T_safe,
        forward_kinematics(Q_LEFT_A, model=state.model),
        atol=1e-12,
    )


def test_first_vr_pose_has_zero_target_jump():
    """Map the first VR pose back to the measured robot anchor exactly."""
    state = ArmFusionState("left")
    make_ready(state, Q_LEFT_A, 20.0)
    T_vr = make_transform([0.3, 1.1, -0.2], [0.2, -0.1, 0.05])

    assert state.capture_vr_anchor(T_vr, True, 0.25, 20.0)
    T_target = map_vr_pose_to_robot_target(
        T_vr_anchor=state.T_vr_anchor,
        T_vr_current=T_vr,
        T_ee_anchor=state.T_ee_anchor,
        side="left",
    )

    assert np.allclose(T_target, state.T_safe, atol=1e-12)
    assert np.array_equal(state.q_safe, Q_LEFT_A)

    result = project_target_to_feasible(
        T_safe=state.T_safe,
        T_raw=T_target,
        q_safe=state.q_safe,
        model=state.model,
        sigma_stop=0.01,
    )
    assert result["success"]
    assert np.array_equal(result["q_safe"], Q_LEFT_A)


def test_tracking_recovery_reanchors_at_new_measured_state():
    """Ignore motion during tracking loss and restart at current robot q."""
    state = ArmFusionState("left")
    make_ready(state, Q_LEFT_A, 30.0)
    T_vr_first = make_transform([0.0, 1.0, 0.0])
    assert state.capture_vr_anchor(T_vr_first, True, 0.25, 30.0)
    state.last_safe_command_dt_s = 0.02

    state.invalidate_anchor()
    state.update_measured_q(Q_LEFT_B, 30.1)
    T_vr_recovered = make_transform([0.7, 0.2, -0.5], [0.3, 0.1, -0.2])
    assert state.capture_vr_anchor(
        T_vr_recovered, True, 0.25, 30.1
    )

    T_target = map_vr_pose_to_robot_target(
        T_vr_anchor=state.T_vr_anchor,
        T_vr_current=T_vr_recovered,
        T_ee_anchor=state.T_ee_anchor,
        side="left",
    )
    expected = forward_kinematics(Q_LEFT_B, model=state.model)
    assert np.array_equal(state.q_safe, Q_LEFT_B)
    assert state.last_safe_command_dt_s is None
    assert np.allclose(state.T_safe, expected, atol=1e-12)
    assert np.allclose(T_target, expected, atol=1e-12)


def test_disconnected_reported_stale_and_local_timeout_block_alignment():
    """Require all three communication/freshness conditions."""
    state = ArmFusionState("right")
    state.update_measured_q(np.zeros(7), 40.0)

    assert state.robot_state_ready(0.25, 40.0) is False
    state.robot_connected = True
    assert state.robot_state_ready(0.25, 40.0) is False
    state.robot_reported_stale = False
    assert state.robot_state_ready(0.25, 40.24) is True
    assert state.robot_state_ready(0.25, 40.26) is False


@pytest.mark.parametrize(
    "bad_q",
    [
        np.zeros(6),
        np.full(7, np.nan),
        np.deg2rad([0.0, 0.0, 0.0, 136.0, 0.0, 0.0, 0.0]),
    ],
)
def test_invalid_measured_q_is_rejected(bad_q):
    """Reject wrong-size, non-finite, and hard-limit feedback."""
    state = ArmFusionState("left")
    with pytest.raises(ValueError):
        state.update_measured_q(bad_q, 50.0)
    assert state.q_measured is None
    assert state.q_safe is None


def test_rviz_fallback_requires_explicit_constructor_argument():
    """Preserve offline visualization only through an explicit fallback."""
    state = ArmFusionState("right", fallback_q=np.zeros(7))

    assert np.array_equal(state.q_safe, np.zeros(7))
    assert state.robot_state_initialized is True
    assert state.initialized_from_robot is False


def test_measured_state_outside_soft_limits_cannot_initialize_or_reanchor():
    """Observe an out-of-bounds robot without copying it into q_safe."""
    state = ArmFusionState("left")
    configure_soft_limits(state)
    q_outside = Q_LEFT_A.copy()
    q_outside[3] = np.deg2rad(-10.0)
    state.robot_connected = True
    state.robot_reported_stale = False
    state.update_measured_q(q_outside, 60.0)

    assert not state.initialize_from_measured(0.25, 60.0)
    assert not state.capture_vr_anchor(np.eye(4), True, 0.25, 60.0)
    assert state.q_safe is None
    assert state.q_command is None
    assert "soft limits" in state.last_robot_state_error


def test_reanchor_refuses_new_measured_state_outside_soft_limits():
    """Keep the previous safe command if feedback leaves the teleop region."""
    state = ArmFusionState("left")
    configure_soft_limits(state)
    make_ready(state, Q_LEFT_A, 70.0)
    q_safe_before = state.q_safe.copy()
    q_outside = Q_LEFT_A.copy()
    q_outside[3] = np.deg2rad(-10.0)
    state.update_measured_q(q_outside, 70.1)

    assert not state.synchronize_safe_to_measured(0.25, 70.1)
    assert np.array_equal(state.q_safe, q_safe_before)


def test_rviz_fallback_outside_configured_branch_is_rejected():
    """Catch an incompatible fallback pose during node configuration."""
    fallback = Q_LEFT_A.copy()
    fallback[3] = np.deg2rad(20.0)
    state = ArmFusionState("left", fallback_q=fallback)
    with pytest.raises(ValueError, match="existing safe state"):
        configure_soft_limits(state)
