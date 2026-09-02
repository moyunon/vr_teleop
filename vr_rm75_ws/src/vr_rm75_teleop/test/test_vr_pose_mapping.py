"""Automatic tests for Quest-to-RM75 orientation mapping."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vr_rm75_teleop.vr_pose_mapping import (
    C_VR_TO_LEFT_ARM,
    C_VR_TO_RIGHT_ARM,
    map_vr_pose_to_robot_target,
)


ANGLE = np.deg2rad(10.0)


def make_transform(rotation, position=(0.0, 0.0, 0.0)):
    """Build a homogeneous transform from a rotation matrix and position."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


ORIENTATION_CASES = [
    pytest.param("left", "+roll", [1, 0, 0], [0, 1, 0], id="left_+roll"),
    pytest.param("left", "-roll", [-1, 0, 0], [0, -1, 0], id="left_-roll"),
    pytest.param("left", "+pitch", [0, 1, 0], [0, 0, 1], id="left_+pitch"),
    pytest.param("left", "-pitch", [0, -1, 0], [0, 0, -1], id="left_-pitch"),
    pytest.param("left", "+yaw", [0, 0, 1], [1, 0, 0], id="left_+yaw"),
    pytest.param("left", "-yaw", [0, 0, -1], [-1, 0, 0], id="left_-yaw"),
    pytest.param("right", "+roll", [1, 0, 0], [0, 1, 0], id="right_+roll"),
    pytest.param("right", "-roll", [-1, 0, 0], [0, -1, 0], id="right_-roll"),
    pytest.param("right", "+pitch", [0, 1, 0], [0, 0, -1], id="right_+pitch"),
    pytest.param("right", "-pitch", [0, -1, 0], [0, 0, 1], id="right_-pitch"),
    pytest.param("right", "+yaw", [0, 0, 1], [-1, 0, 0], id="right_+yaw"),
    pytest.param("right", "-yaw", [0, 0, -1], [1, 0, 0], id="right_-yaw"),
]


@pytest.mark.parametrize(
    "side,label,vr_axis,expected_arm_axis",
    ORIENTATION_CASES,
)
def test_orientation_axis_and_sign_with_nonidentity_anchors(
    side,
    label,
    vr_axis,
    expected_arm_axis,
):
    """Check actual rotation-vector axes and signs for both controllers."""
    del label
    vr_anchor_rotation = Rotation.from_euler(
        "xyz", [0.31, -0.23, 0.47]
    ).as_matrix()
    vr_delta_rotation = Rotation.from_rotvec(
        ANGLE * np.asarray(vr_axis, dtype=float)
    ).as_matrix()
    vr_current_rotation = vr_delta_rotation @ vr_anchor_rotation
    ee_anchor_rotation = Rotation.from_euler(
        "xyz", [-0.29, 0.38, -0.17]
    ).as_matrix()

    target = map_vr_pose_to_robot_target(
        T_vr_anchor=make_transform(vr_anchor_rotation, [0.2, -0.4, 1.1]),
        T_vr_current=make_transform(vr_current_rotation, [0.2, -0.4, 1.1]),
        T_ee_anchor=make_transform(ee_anchor_rotation, [0.5, 0.1, 0.7]),
        side=side,
        orientation_scale=1.0,
    )

    actual_delta_rotation = target[:3, :3] @ ee_anchor_rotation.T
    actual_rotvec = Rotation.from_matrix(actual_delta_rotation).as_rotvec()
    expected_rotvec = ANGLE * np.asarray(expected_arm_axis, dtype=float)

    np.testing.assert_allclose(actual_rotvec, expected_rotvec, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "side,vr_to_arm",
    [
        pytest.param("left", C_VR_TO_LEFT_ARM, id="left"),
        pytest.param("right", C_VR_TO_RIGHT_ARM, id="right"),
    ],
)
def test_relative_rotation_order_and_coordinate_conjugation(side, vr_to_arm):
    """Verify R_current R_anchor.T and C R_delta C.T as matrices."""
    vr_anchor_rotation = Rotation.from_euler(
        "xyz", [0.41, -0.26, 0.19]
    ).as_matrix()
    requested_delta = Rotation.from_rotvec(
        np.array([0.13, -0.08, 0.17])
    ).as_matrix()
    vr_current_rotation = requested_delta @ vr_anchor_rotation
    ee_anchor_rotation = Rotation.from_euler(
        "xyz", [-0.22, 0.35, 0.28]
    ).as_matrix()

    target = map_vr_pose_to_robot_target(
        T_vr_anchor=make_transform(vr_anchor_rotation),
        T_vr_current=make_transform(vr_current_rotation),
        T_ee_anchor=make_transform(ee_anchor_rotation),
        side=side,
        orientation_scale=1.0,
    )

    expected_vr_delta = vr_current_rotation @ vr_anchor_rotation.T
    expected_arm_delta = vr_to_arm @ expected_vr_delta @ vr_to_arm.T
    actual_arm_delta = target[:3, :3] @ ee_anchor_rotation.T

    np.testing.assert_allclose(expected_vr_delta, requested_delta, atol=1e-12)
    np.testing.assert_allclose(actual_arm_delta, expected_arm_delta, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
def test_zero_relative_rotation_preserves_ee_anchor(side):
    """A coincident controller anchor must not create an orientation jump."""
    vr_anchor_rotation = Rotation.from_euler(
        "xyz", [0.37, 0.14, -0.32]
    ).as_matrix()
    ee_anchor_rotation = Rotation.from_euler(
        "xyz", [-0.18, 0.27, 0.44]
    ).as_matrix()
    vr_anchor = make_transform(vr_anchor_rotation)

    target = map_vr_pose_to_robot_target(
        T_vr_anchor=vr_anchor,
        T_vr_current=vr_anchor.copy(),
        T_ee_anchor=make_transform(ee_anchor_rotation),
        side=side,
        orientation_scale=1.0,
    )

    np.testing.assert_allclose(target[:3, :3], ee_anchor_rotation, atol=1e-12)
