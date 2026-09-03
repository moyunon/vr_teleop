import numpy as np

from scipy.spatial.transform import Rotation


def limit_pose_step(
    T_current,
    T_desired,

    max_translation_step,
    max_rotation_step,
):
    """
    限制一次 SE(3) 位姿更新的最大变化量。

    Parameters
    ----------
    T_current:
        当前已经接受的 Pose。

    T_desired:
        VR 当前希望达到的 Pose。

    max_translation_step:
        单帧最大平移距离 [m]。

    max_rotation_step:
        单帧最大旋转角 [rad]。

    Returns
    -------
    {
        "T_limited",
        "translation_distance",
        "translation_step",
        "rotation_distance",
        "rotation_step",
        "translation_limited",
        "rotation_limited",
    }
    """

    T_current = np.asarray(
        T_current,
        dtype=float,
    )

    T_desired = np.asarray(
        T_desired,
        dtype=float,
    )

    # =========================================================
    # 1. Translation
    # =========================================================

    p_current = (
        T_current[:3, 3]
    )

    p_desired = (
        T_desired[:3, 3]
    )

    delta_p = (
        p_desired
        - p_current
    )

    translation_distance = (
        np.linalg.norm(
            delta_p
        )
    )

    if (
        translation_distance
        > max_translation_step
    ):

        delta_p_step = (
            delta_p
            * (
                max_translation_step
                /
                translation_distance
            )
        )

        translation_limited = True

    else:

        delta_p_step = (
            delta_p
        )

        translation_limited = False

    p_limited = (
        p_current
        + delta_p_step
    )

    translation_step = (
        np.linalg.norm(
            delta_p_step
        )
    )

    # =========================================================
    # 2. Rotation
    #
    # 使用 spatial relative rotation：
    #
    # R_desired
    # =
    # R_delta @ R_current
    #
    # 与当前 IK / feasibility 的旋转定义一致。
    # =========================================================

    R_current = (
        T_current[:3, :3]
    )

    R_desired = (
        T_desired[:3, :3]
    )

    R_relative = (
        R_desired
        @ R_current.T
    )

    rotvec = (
        Rotation
        .from_matrix(
            R_relative
        )
        .as_rotvec()
    )

    rotation_distance = (
        np.linalg.norm(
            rotvec
        )
    )

    if (
        rotation_distance
        > max_rotation_step
    ):

        rotvec_step = (
            rotvec
            * (
                max_rotation_step
                /
                rotation_distance
            )
        )

        rotation_limited = True

    else:

        rotvec_step = (
            rotvec
        )

        rotation_limited = False

    R_step = (
        Rotation
        .from_rotvec(
            rotvec_step
        )
        .as_matrix()
    )

    R_limited = (
        R_step
        @ R_current
    )

    rotation_step = (
        np.linalg.norm(
            rotvec_step
        )
    )

    # =========================================================
    # 3. Compose
    # =========================================================

    T_limited = np.eye(
        4,
        dtype=float,
    )

    T_limited[:3, :3] = (
        R_limited
    )

    T_limited[:3, 3] = (
        p_limited
    )

    return {
        "T_limited":
            T_limited,

        "translation_distance":
            float(
                translation_distance
            ),

        "translation_step":
            float(
                translation_step
            ),

        "rotation_distance":
            float(
                rotation_distance
            ),

        "rotation_step":
            float(
                rotation_step
            ),

        "translation_limited":
            translation_limited,

        "rotation_limited":
            rotation_limited,
    }