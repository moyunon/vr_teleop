import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.rm75_ik import solve_ik
from vr_rm75_teleop.rm75_nullspace import (
    joint_centering_cost,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)


def independent_pose_error(
    T_target,
    T_solution,
):

    position_error = np.linalg.norm(
        T_target[:3, 3]
        - T_solution[:3, 3]
    )

    R_error = (
        T_target[:3, :3]
        @ T_solution[:3, :3].T
    )

    orientation_error = np.linalg.norm(
        Rotation
        .from_matrix(R_error)
        .as_rotvec()
    )

    return (
        position_error,
        orientation_error,
    )


def print_result(
    name,
    result,
    T_target,
    model,
):

    q = result["q"]

    T_solution = forward_kinematics(
        q,
        model=model,
    )

    (
        p_error,
        r_error,
    ) = independent_pose_error(
        T_target,
        T_solution,
    )

    print("")
    print("==============================")
    print(name)
    print("==============================")

    print("")
    print(
        "Success:",
        result["success"],
    )

    print(
        "Secondary converged:",
        result["secondary_converged"],
    )

    print(
        "Iterations:",
        result["iterations"],
    )

    print("")
    print("q [deg]:")
    print(
        np.rad2deg(q)
    )

    print("")
    print(
        "Position error [mm]:",
        p_error * 1000.0,
    )

    print(
        "Orientation error [deg]:",
        np.rad2deg(r_error),
    )

    print(
        "Joint centering cost:",
        joint_centering_cost(
            q,
            model,
        ),
    )

    print(
        "Projected centering norm:",
        result["centering_norm"],
    )


def main():

    model = RM75Model(
        side="right",
    )

    # ============================================================
    # 同上一轮 IK 测试
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
    print("Target generated from q_true [deg]:")
    print(
        np.rad2deg(q_true)
    )

    print("")
    print("Seed [deg]:")
    print(
        np.rad2deg(q_seed)
    )

    print("")
    print(
        "Seed centering cost:",
        joint_centering_cost(
            q_seed,
            model,
        ),
    )

    # ============================================================
    # A: 普通 DLS
    # ============================================================

    baseline = solve_ik(
        T_target=T_target,
        q_seed=q_seed,
        model=model,

        max_iterations=300,

        damping=0.02,
        step_gain=0.5,

        joint_centering=False,
    )

    # ============================================================
    # B: DLS + Null-space joint centering
    # ============================================================

    centered = solve_ik(
        T_target=T_target,
        q_seed=q_seed,
        model=model,

        # Secondary task 需要更多迭代，
        # 此处是离线数学测试，不是实时控制周期。
        max_iterations=300,

        damping=0.02,
        step_gain=0.5,

        joint_centering=True,

        joint_centering_gain=1.0,

        max_null_step=np.deg2rad(
            0.5
        ),

        centering_tolerance=1e-5,
    )

    print_result(
        "A: BASELINE DLS",
        baseline,
        T_target,
        model,
    )

    print_result(
        "B: DLS + NULL-SPACE CENTERING",
        centered,
        T_target,
        model,
    )

    # ============================================================
    # Comparison
    # ============================================================

    baseline_cost = (
        baseline[
            "joint_centering_cost"
        ]
    )

    centered_cost = (
        centered[
            "joint_centering_cost"
        ]
    )

    print("")
    print("==============================")
    print("COMPARISON")
    print("==============================")

    print("")
    print(
        "Baseline cost:",
        baseline_cost,
    )

    print(
        "Centered cost:",
        centered_cost,
    )

    print(
        "Cost reduction:",
        baseline_cost
        - centered_cost,
    )

    if (
        baseline["success"]
        and
        centered["success"]
        and
        centered_cost
        < baseline_cost
    ):

        print("")
        print(
            "JOINT CENTERING TEST: PASS"
        )

    else:

        print("")
        print(
            "JOINT CENTERING TEST: FAIL"
        )


if __name__ == "__main__":
    main()