import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.rm75_jacobian import geometric_jacobian
from vr_rm75_teleop.rm75_nullspace import null_space_vector


np.set_printoptions(
    precision=8,
    suppress=True,
)


def pose_difference(
    T_a,
    T_b,
):
    """
    计算两个末端 Pose 的实际差异。

    返回：
        position_error_m
        orientation_error_rad
    """

    p_a = T_a[:3, 3]
    p_b = T_b[:3, 3]

    position_error = np.linalg.norm(
        p_b - p_a
    )

    R_a = T_a[:3, :3]
    R_b = T_b[:3, :3]

    R_error = (
        R_b
        @ R_a.T
    )

    rotvec = (
        Rotation
        .from_matrix(R_error)
        .as_rotvec()
    )

    orientation_error = np.linalg.norm(
        rotvec
    )

    return (
        position_error,
        orientation_error,
    )


def main():

    model = RM75Model(
        side="left",
    )

    # ============================================================
    # 使用前面验证过的普通非奇异构型
    # ============================================================

    q = np.deg2rad(
        [
             10.0,
            -20.0,
             30.0,
             40.0,
            -25.0,
             35.0,
             15.0,
        ]
    )

    print("")
    print("==============================")
    print("RM75-6FB NULL SPACE TEST")
    print("==============================")

    print("")
    print("q [deg]:")
    print(
        np.rad2deg(q)
    )

    # ============================================================
    # 1. Jacobian
    # ============================================================

    J = geometric_jacobian(
        q,
        model=model,
    )

    rank = np.linalg.matrix_rank(J)

    print("")
    print("Jacobian rank:")
    print(rank)

    # ============================================================
    # 2. Null-space vector
    # ============================================================

    n = null_space_vector(J)

    print("")
    print("Null-space vector n:")
    print(n)

    print("")
    print("||n||:")
    print(
        np.linalg.norm(n)
    )

    # ============================================================
    # 3. 最关键测试：
    #
    # J @ n 是否为 0
    # ============================================================

    task_velocity = J @ n

    print("")
    print("J @ n:")
    print(task_velocity)

    print("")
    print("||J @ n||:")
    print(
        np.linalg.norm(task_velocity)
    )

    # ============================================================
    # 4. 给机械臂一个很小的 null-space 关节变化
    # ============================================================

    epsilon = np.deg2rad(1.0)

    dq_null = (
        epsilon * n
    )

    print("")
    print(
        "Null-space joint perturbation "
        "[deg]:"
    )

    print(
        np.rad2deg(dq_null)
    )

    # 一阶 Jacobian 理论预测
    predicted_delta_x = (
        J @ dq_null
    )

    print("")
    print(
        "Predicted Cartesian delta "
        "(first order):"
    )

    print(
        predicted_delta_x
    )

    print("")
    print(
        "Predicted delta norm:"
    )

    print(
        np.linalg.norm(
            predicted_delta_x
        )
    )

    # ============================================================
    # 5. 使用真正 FK 检查有限关节变化后的 Pose
    # ============================================================

    T_before = forward_kinematics(
        q,
        model=model,
    )

    q_after = (
        q + dq_null
    )

    T_after = forward_kinematics(
        q_after,
        model=model,
    )

    (
        position_error,
        orientation_error,
    ) = pose_difference(
        T_before,
        T_after,
    )

    print("")
    print("------------------------------")
    print("FINITE FK CHECK")
    print("------------------------------")

    print("")
    print(
        "Actual position change [m]:"
    )
    print(position_error)

    print(
        "Actual position change [mm]:"
    )
    print(
        position_error * 1000.0
    )

    print(
        "Actual orientation change [deg]:"
    )
    print(
        np.rad2deg(
            orientation_error
        )
    )

    # ============================================================
    # Result
    # ============================================================

    null_error = np.linalg.norm(
        task_velocity
    )

    if null_error < 1e-10:

        print("")
        print(
            "NULL SPACE ALGEBRA TEST: PASS"
        )

    else:

        print("")
        print(
            "NULL SPACE ALGEBRA TEST: FAIL"
        )


if __name__ == "__main__":
    main()