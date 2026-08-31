import numpy as np

from scipy.spatial.transform import Rotation


# =============================================================
# VR frame -> robot control frame
#
# 根据当前实测：
#
# VR:
#   +x backward
#   +y right
#   +z up
#
# Robot control convention:
#   +x forward
#   +y left
#   +z up
#
# 因此等价于绕 z 轴旋转 180 deg。
# =============================================================

# =============================================================
# 1. VR frame -> RealBot body frame
#
# 实测 Quest:
#
#   +X = backward
#   +Y = right
#   +Z = up
#
# 我们定义 RealBot body control frame:
#
#   +X = forward
#   +Y = left
#   +Z = up
#
# 因此：
#
#   VR +X -> Body -X
#   VR +Y -> Body -Y
#   VR +Z -> Body +Z
# =============================================================

C_VR_TO_BODY = np.eye(
    3,
    dtype=float,
)


# =============================================================
# 2. RM75 arm base -> RealBot body
#
# 来自已经检查过的双臂 URDF：
#
# xb_link -> l_rm75_base_link
# xb_link -> r_rm75_base_link
#
# 矩阵含义：
#
#   p_body = R_BODY_FROM_ARM @ p_arm
# =============================================================

R_BODY_FROM_LEFT_ARM = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)

R_BODY_FROM_RIGHT_ARM = np.array(
    [
        [ 0.0, 1.0,  0.0],
        [ 0.0, 0.0, -1.0],
        [-1.0, 0.0,  0.0],
    ],
    dtype=float,
)


# =============================================================
# 3. VR frame -> each RM75 local base frame
#
# p_body = C_VR_TO_BODY @ p_vr
#
# p_arm =
#     R_BODY_FROM_ARM.T @ p_body
#
# therefore:
#
# C_VR_TO_ARM =
#     R_BODY_FROM_ARM.T @ C_VR_TO_BODY
# =============================================================

C_VR_TO_LEFT_ARM = (
    R_BODY_FROM_LEFT_ARM.T
    @ C_VR_TO_BODY
)

C_VR_TO_RIGHT_ARM = (
    R_BODY_FROM_RIGHT_ARM.T
    @ C_VR_TO_BODY
)


def get_vr_to_arm_rotation(
    side,
):

    side = str(
        side
    ).lower()

    if side == "left":
        return (
            C_VR_TO_LEFT_ARM.copy()
        )

    if side == "right":
        return (
            C_VR_TO_RIGHT_ARM.copy()
        )

    raise ValueError(
        "side must be 'left' or 'right', "
        f"got {side!r}"
    )


def position_quaternion_to_transform(
    position,
    quaternion_xyzw,
):
    """
    将：

        position = [x, y, z]
        quaternion = [qx, qy, qz, qw]

    转换成 4x4 SE(3) 齐次变换矩阵。

    ROS geometry_msgs/Quaternion
    与 scipy Rotation.from_quat()
    都采用 xyzw 顺序。
    """

    position = np.asarray(
        position,
        dtype=float,
    )

    quaternion_xyzw = np.asarray(
        quaternion_xyzw,
        dtype=float,
    )

    if position.shape != (3,):
        raise ValueError(
            "position must have shape (3,)"
        )

    if quaternion_xyzw.shape != (4,):
        raise ValueError(
            "quaternion must have shape (4,)"
        )

    R = (
        Rotation
        .from_quat(
            quaternion_xyzw
        )
        .as_matrix()
    )

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, :3] = R
    T[:3, 3] = position

    return T


def map_vr_pose_to_robot_target(
    T_vr_anchor,
    T_vr_current,
    T_ee_anchor,

    side,

    position_scale=1.0,
    orientation_scale=1.0,
):
    """
    将 Quest 手柄相对于起始时刻的位姿变化，
    映射为 RM75 末端目标位姿。

    这里使用增量式映射：

        VR absolute pose
              ↓
        VR relative motion
              ↓
        robot EE target

    而不是把 Quest 的绝对位置直接作为机械臂目标。
    """

    T_vr_anchor = np.asarray(
        T_vr_anchor,
        dtype=float,
    )

    T_vr_current = np.asarray(
        T_vr_current,
        dtype=float,
    )

    T_ee_anchor = np.asarray(
        T_ee_anchor,
        dtype=float,
    )

    C = get_vr_to_arm_rotation(
        side
    )

    # =========================================================
    # 1. Translation increment in VR frame
    # =========================================================

    p_vr_anchor = (
        T_vr_anchor[:3, 3]
    )

    p_vr_current = (
        T_vr_current[:3, 3]
    )

    delta_p_vr = (
        p_vr_current
        - p_vr_anchor
    )

    # =========================================================
    # 2. Express translation increment in robot frame
    # =========================================================

    delta_p_robot = (
        C
        @ delta_p_vr
    )

    p_ee_anchor = (
        T_ee_anchor[:3, 3]
    )

    p_target = (
        p_ee_anchor
        + position_scale
        * delta_p_robot
    )

    # =========================================================
    # 3. Spatial orientation increment in VR frame
    # =========================================================

    R_vr_anchor = (
        T_vr_anchor[:3, :3]
    )

    R_vr_current = (
        T_vr_current[:3, :3]
    )

    R_delta_vr = (
        R_vr_current
        @ R_vr_anchor.T
    )

    relative_rotvec_vr = (
        Rotation
        .from_matrix(
            R_delta_vr
        )
        .as_rotvec()
    )

    R_delta_vr = (
        Rotation
        .from_rotvec(
            orientation_scale
            * relative_rotvec_vr
        )
        .as_matrix()
    )

    # =========================================================
    # 4. Change coordinates:
    #
    #       R_robot = C R_vr C^T
    # =========================================================

    R_delta_robot = (
        C
        @ R_delta_vr
        @ C.T
    )

    # =========================================================
    # 5. Apply spatial increment to EE anchor
    # =========================================================

    R_ee_anchor = (
        T_ee_anchor[:3, :3]
    )

    R_target = (
        R_delta_robot
        @ R_ee_anchor
    )

    # =========================================================
    # 6. Compose target SE(3)
    # =========================================================

    T_target = np.eye(
        4,
        dtype=float,
    )

    T_target[:3, :3] = (
        R_target
    )

    T_target[:3, 3] = (
        p_target
    )

    return T_target