import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.rm75_ik import solve_ik


np.set_printoptions(
    precision=6,
    suppress=True,
)


def rotation_error_deg(
    R_a,
    R_b,
):
    """
    返回两个 rotation matrix 之间的角度误差。
    """

    R_error = (
        R_a
        @ R_b.T
    )

    rotvec = (
        Rotation
        .from_matrix(R_error)
        .as_rotvec()
    )

    return np.rad2deg(
        np.linalg.norm(rotvec)
    )


def main():

    model = RM75Model(
        side="left",
    )

    # ============================================================
    # 我们假装不知道 q_true。
    #
    # 这里只利用 q_true 的 FK 来生成一个
    # “绝对保证机械臂可以到达”的目标 Pose。
    # ============================================================

    q_true = np.deg2rad(
        [
             15.0,
            -15.0,
             25.0,
             45.0,
            -20.0,
             30.0,
             20.0,
        ]
    )

    T_target = forward_kinematics(
        q_true,
        model=model,
    )

    # ============================================================
    # Seed
    #
    # 故意和 q_true 不一样。
    #
    # 但又不会相差特别大。
    #
    # 这正好模拟未来 VR 遥操：
    #
    # 当前帧 q(k)
    #     ↓
    # 下一帧目标 Pose
    #     ↓
    # 用 q(k) 当 seed
    # ============================================================

    q_seed = np.deg2rad(
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
    print("RM75-6FB DLS IK TEST")
    print("==============================")

    print("")
    print("q_true [deg]:")
    print(
        np.rad2deg(q_true)
    )

    print("")
    print("q_seed [deg]:")
    print(
        np.rad2deg(q_seed)
    )

    print("")
    print("Target T_07:")
    print(T_target)

    # ============================================================
    # Solve IK
    # ============================================================

    result = solve_ik(
        T_target=T_target,
        q_seed=q_seed,
        model=model,
        max_iterations=200,
        damping=0.02,
        step_gain=0.5,
        max_joint_step=np.deg2rad(5.0),
    )

    q_solution = result["q"]

    print("")
    print("------------------------------")
    print("IK RESULT")
    print("------------------------------")

    print("")
    print(
        "Success:",
        result["success"],
    )

    print(
        "Iterations:",
        result["iterations"],
    )

    print("")
    print(
        "q_solution [deg]:"
    )

    print(
        np.rad2deg(q_solution)
    )

    # ============================================================
    # FK check
    # ============================================================

    T_solution = forward_kinematics(
        q_solution,
        model=model,
    )

    print("")
    print("T_solution:")
    print(T_solution)

    # ============================================================
    # Final independent error
    # ============================================================

    position_error = np.linalg.norm(
        T_target[:3, 3]
        - T_solution[:3, 3]
    )

    orientation_error_deg = rotation_error_deg(
        T_target[:3, :3],
        T_solution[:3, :3],
    )

    print("")
    print(
        "Position error [m]:",
        position_error,
    )

    print(
        "Position error [mm]:",
        position_error * 1000.0,
    )

    print(
        "Orientation error [deg]:",
        orientation_error_deg,
    )

    # ============================================================
    # Important:
    #
    # 不要求 q_solution == q_true
    #
    # 因为 RM75 是 7DOF 冗余机械臂。
    # ============================================================

    if (
        result["success"]
        and position_error < 1e-4
        and orientation_error_deg < 0.1
    ):

        print("")
        print("IK TEST: PASS")

    else:

        print("")
        print("IK TEST: FAIL")


if __name__ == "__main__":
    main()