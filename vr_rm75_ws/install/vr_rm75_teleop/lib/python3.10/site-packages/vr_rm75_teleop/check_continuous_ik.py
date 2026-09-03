import time
import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import (
    RM75Model,
)

from vr_rm75_teleop.rm75_fk import (
    forward_kinematics,
)

from vr_rm75_teleop.rm75_ik import (
    solve_ik,
)

from vr_rm75_teleop.rm75_jacobian import (
    geometric_jacobian,
)

from vr_rm75_teleop.rm75_nullspace import (
    preferred_posture_cost,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)


def pose_error_independent(
    T_target,
    T_actual,
):
    """
    独立检查目标 Pose 与 FK 实际 Pose
    之间的误差。

    Returns
    -------
    position_error:
        meter

    orientation_error:
        rad
    """

    # =========================================================
    # Position
    # =========================================================

    position_error = np.linalg.norm(
        T_target[:3, 3]
        - T_actual[:3, 3]
    )

    # =========================================================
    # Orientation
    # =========================================================

    R_error = (
        T_target[:3, :3]
        @ T_actual[:3, :3].T
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


def generate_reachable_target(
    q_start,
    phase,
    model,
):
    """
    通过一条安全的小幅关节轨迹
    生成目标 Pose。

    因为：

        T_target = FK(q_reference)

    所以每一帧目标都严格可达。

    同时：

        phase = 0
        phase = 1

    时：

        q_reference = q_start

    因此整个轨迹首尾闭合。
    """

    angle = (
        2.0
        * np.pi
        * phase
    )

    delta_q_deg = np.array(
        [
            # J1
            5.0
            * np.sin(angle),

            # J2
            3.0
            * np.sin(
                2.0 * angle
            ),

            # J3
            4.0
            * np.sin(angle),

            # J4
            #
            # q_start = 40°
            # 变化范围约 ±5°
            #
            # 保持在 35° ~ 45°，
            # 避免 q4 -> 0 的伸直奇异构型。
            5.0
            * np.sin(
                2.0 * angle
            ),

            # J5
            4.0
            * np.sin(angle),

            # J6
            3.0
            * np.sin(
                2.0 * angle
            ),

            # J7
            5.0
            * np.sin(angle),
        ],
        dtype=float,
    )

    q_reference = (
        q_start
        + np.deg2rad(
            delta_q_deg
        )
    )

    T_target = (
        forward_kinematics(
            q_reference,
            model=model,
        )
    )

    return (
        T_target,
        q_reference,
    )


def main():

    model = RM75Model(
        side="left",
    )

    # ============================================================
    # Initial configuration
    # ============================================================

    q_start = np.deg2rad(
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

    # ============================================================
    # Preferred posture
    #
    # 当前数学验证阶段：
    # 使用已经验证过的安全非奇异构型。
    #
    # 未来实机双臂时要分别定义：
    #
    # left_q_preferred
    # right_q_preferred
    # ============================================================

    q_preferred = (
        q_start.copy()
    )

    T_start = (
        forward_kinematics(
            q_start,
            model=model,
        )
    )

    # ============================================================
    # Simulate Quest target rate
    #
    # 50 Hz
    # 6 seconds
    # ============================================================

    frequency = 50.0

    duration = 6.0

    num_frames = int(
        frequency
        * duration
    )

    # ============================================================
    # 上一帧 IK 结果作为下一帧 seed
    # ============================================================

    q_current = (
        q_start.copy()
    )

    # ============================================================
    # Statistics
    # ============================================================

    q_history = [
        q_current.copy()
    ]

    iteration_history = []

    solve_time_history = []

    position_error_history = []

    orientation_error_history = []

    posture_cost_history = []

    min_singular_value_history = []

    failed_frames = []

    # ============================================================
    # Header
    # ============================================================

    print("")

    print(
        "=============================="
    )

    print(
        "RM75-6FB CONTINUOUS IK TEST"
    )

    print(
        "=============================="
    )

    print("")

    print(
        f"Frames: {num_frames}"
    )

    print(
        f"Target rate: "
        f"{frequency} Hz"
    )

    print(
        f"Duration: "
        f"{duration} s"
    )

    print("")

    print(
        "Initial q [deg]:"
    )

    print(
        np.rad2deg(
            q_start
        )
    )

    print("")

    print(
        "Preferred q [deg]:"
    )

    print(
        np.rad2deg(
            q_preferred
        )
    )

    # ============================================================
    # Main sequence
    # ============================================================

    for frame in range(
        num_frames
    ):

        phase = (
            frame
            / (
                num_frames
                - 1
            )
        )

        (
            T_target,
            q_reference,
        ) = (
            generate_reachable_target(
                q_start,
                phase,
                model,
            )
        )

        # ========================================================
        # IK timing
        # ========================================================

        start_time = (
            time.perf_counter()
        )

        result = solve_ik(
            T_target=T_target,

            q_seed=q_current,

            model=model,

            max_iterations=20,

            position_tolerance=1e-4,

            orientation_tolerance=1e-3,

            damping=0.02,

            step_gain=0.7,

            max_joint_step=
                np.deg2rad(
                    2.0
                ),

            preferred_posture=
                q_preferred,

            preferred_posture_gain=
                1.0,

            max_null_step=
                np.deg2rad(
                    0.10
                ),
        )

        solve_time = (
            time.perf_counter()
            - start_time
        )

        solve_time_history.append(
            solve_time
        )

        iteration_history.append(
            result[
                "iterations"
            ]
        )

        # ========================================================
        # Failure handling
        #
        # IK 失败：
        # 保持上一帧关节角。
        #
        # 不把失败解作为下一帧 seed。
        # ========================================================

        if not result[
            "success"
        ]:

            failed_frames.append(
                frame
            )

            q_history.append(
                q_current.copy()
            )

            continue

        # ========================================================
        # Successful IK
        # ========================================================

        q_next = (
            result["q"]
        )

        # ========================================================
        # Independent FK verification
        # ========================================================

        T_actual = (
            forward_kinematics(
                q_next,
                model=model,
            )
        )

        (
            p_error,
            r_error,
        ) = (
            pose_error_independent(
                T_target,
                T_actual,
            )
        )

        position_error_history.append(
            p_error
        )

        orientation_error_history.append(
            r_error
        )

        # ========================================================
        # Singularity monitoring
        # ========================================================

        J = geometric_jacobian(
            q_next,
            model=model,
        )

        singular_values = (
            np.linalg.svd(
                J,
                compute_uv=False,
            )
        )

        min_singular_value_history.append(
            singular_values[-1]
        )

        # ========================================================
        # Preferred posture cost
        # ========================================================

        posture_cost = (
            preferred_posture_cost(
                q_next,
                q_preferred,
                model,
            )
        )

        posture_cost_history.append(
            posture_cost
        )

        # ========================================================
        # Advance
        # ========================================================

        q_current = (
            q_next
        )

        q_history.append(
            q_current.copy()
        )

    # ============================================================
    # Convert statistics
    # ============================================================

    q_history = np.asarray(
        q_history,
        dtype=float,
    )

    iteration_history = np.asarray(
        iteration_history,
        dtype=float,
    )

    solve_time_history = np.asarray(
        solve_time_history,
        dtype=float,
    )

    position_error_history = (
        np.asarray(
            position_error_history,
            dtype=float,
        )
    )

    orientation_error_history = (
        np.asarray(
            orientation_error_history,
            dtype=float,
        )
    )

    posture_cost_history = (
        np.asarray(
            posture_cost_history,
            dtype=float,
        )
    )

    min_singular_value_history = (
        np.asarray(
            min_singular_value_history,
            dtype=float,
        )
    )

    # ============================================================
    # Joint continuity
    # ============================================================

    if len(
        q_history
    ) >= 2:

        dq_frames = np.diff(
            q_history,
            axis=0,
        )

        max_joint_jump = np.max(
            np.abs(
                dq_frames
            )
        )

        max_joint_jump_deg = (
            np.rad2deg(
                max_joint_jump
            )
        )

    else:

        max_joint_jump_deg = (
            np.inf
        )

    # ============================================================
    # Results
    # ============================================================

    print("")

    print(
        "=============================="
    )

    print(
        "RESULT"
    )

    print(
        "=============================="
    )

    # ============================================================
    # Failure
    # ============================================================

    print("")

    print(
        "Failed frames:",
        len(
            failed_frames
        ),
    )

    if failed_frames:

        print(
            "Failed frame indices:",
            failed_frames,
        )

    # ============================================================
    # Iteration
    # ============================================================

    print("")

    print(
        "Average IK iterations:",
        np.mean(
            iteration_history
        ),
    )

    print(
        "Max IK iterations:",
        np.max(
            iteration_history
        ),
    )

    # ============================================================
    # Time
    # ============================================================

    print("")

    print(
        "Average solve time [ms]:",
        np.mean(
            solve_time_history
        )
        * 1000.0,
    )

    print(
        "Max solve time [ms]:",
        np.max(
            solve_time_history
        )
        * 1000.0,
    )

    # ============================================================
    # Joint continuity
    # ============================================================

    print("")

    print(
        "Max joint jump/frame [deg]:",
        max_joint_jump_deg,
    )

    # ============================================================
    # Cartesian error
    # ============================================================

    if (
        position_error_history.size
        > 0
    ):

        print("")

        print(
            "Max position error [mm]:",
            np.max(
                position_error_history
            )
            * 1000.0,
        )

        print(
            "Max orientation "
            "error [deg]:",
            np.rad2deg(
                np.max(
                    orientation_error_history
                )
            ),
        )

    # ============================================================
    # Singularity
    # ============================================================

    if (
        min_singular_value_history.size
        > 0
    ):

        print("")

        print(
            "Minimum singular value "
            "along trajectory:",
            np.min(
                min_singular_value_history
            ),
        )

    # ============================================================
    # Preferred posture
    # ============================================================

    if (
        posture_cost_history.size
        > 0
    ):

        initial_posture_cost = (
            preferred_posture_cost(
                q_start,
                q_preferred,
                model,
            )
        )

        print("")

        print(
            "Initial preferred "
            "posture cost:",
            initial_posture_cost,
        )

        print(
            "Maximum preferred "
            "posture cost:",
            np.max(
                posture_cost_history
            ),
        )

        print(
            "Final preferred "
            "posture cost:",
            posture_cost_history[-1],
        )

    # ============================================================
    # Final joint configuration
    # ============================================================

    print("")

    print(
        "Final q [deg]:"
    )

    print(
        np.rad2deg(
            q_current
        )
    )

    # ============================================================
    # Final TCP return
    #
    # 轨迹首尾闭合，
    # 最后一帧应该回到 T_start。
    # ============================================================

    T_final = (
        forward_kinematics(
            q_current,
            model=model,
        )
    )

    (
        final_p_error,
        final_r_error,
    ) = (
        pose_error_independent(
            T_start,
            T_final,
        )
    )

    print("")

    print(
        "Final TCP return "
        "error [mm]:",
        final_p_error
        * 1000.0,
    )

    print(
        "Final TCP return "
        "orientation error [deg]:",
        np.rad2deg(
            final_r_error
        ),
    )

    # ============================================================
    # Joint return difference
    # ============================================================

    print("")

    print(
        "q_final - q_start [deg]:"
    )

    print(
        np.rad2deg(
            q_current
            - q_start
        )
    )

    # 最后一帧：
    # q_reference == q_start

    print("")

    print(
        "q_final - q_reference "
        "[deg]:"
    )

    print(
        np.rad2deg(
            q_current
            - q_reference
        )
    )

    # ============================================================
    # PASS criteria
    # ============================================================

    all_success = (
        len(
            failed_frames
        )
        == 0
    )

    task_error_ok = (
        position_error_history.size
        > 0

        and

        np.max(
            position_error_history
        )
        < 1e-4

        and

        np.max(
            orientation_error_history
        )
        < 1e-3
    )

    continuity_ok = (
        max_joint_jump_deg
        < 2.0
    )

    # ============================================================
    # Final result
    # ============================================================

    if (
        all_success
        and
        task_error_ok
        and
        continuity_ok
    ):

        print("")

        print(
            "CONTINUOUS IK TEST: PASS"
        )

    else:

        print("")

        print(
            "CONTINUOUS IK TEST: FAIL"
        )


if __name__ == "__main__":
    main()