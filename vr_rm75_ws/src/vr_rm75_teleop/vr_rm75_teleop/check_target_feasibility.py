import time
import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import (
    RM75Model,
)

from vr_rm75_teleop.rm75_fk import (
    forward_kinematics,
)

from vr_rm75_teleop.rm75_nullspace import (
    preferred_posture_cost,
)

from vr_rm75_teleop.target_feasibility import (
    project_target_to_feasible,
    minimum_singular_value,
)

from vr_rm75_teleop.joint_safety import (
    limit_joint_velocity,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)


def generate_raw_target(
    T_start,
    phase,
):
    """
    重现之前造成 IK 失败的
    Cartesian 轨迹。

    注意：

    这条轨迹不是通过 FK 生成的，

    因此不能保证每一个目标 Pose
    都属于 RM75 工作空间。
    """

    angle = (
        2.0
        * np.pi
        * phase
    )

    T_target = (
        T_start.copy()
    )

    # =========================================================
    # Position
    # =========================================================

    delta_position = np.array(
        [
            0.030
            * np.sin(angle),

            0.020
            * np.sin(
                2.0 * angle
            ),

            0.015
            * (
                1.0
                - np.cos(angle)
            ),
        ]
    )

    T_target[:3, 3] = (
        T_start[:3, 3]
        + delta_position
    )

    # =========================================================
    # Orientation
    # =========================================================

    delta_rotvec = np.array(
        [
            np.deg2rad(3.0)
            * np.sin(angle),

            np.deg2rad(2.0)
            * np.sin(
                2.0 * angle
            ),

            np.deg2rad(4.0)
            * np.sin(angle),
        ]
    )

    R_delta = (
        Rotation
        .from_rotvec(
            delta_rotvec
        )
        .as_matrix()
    )

    T_target[:3, :3] = (
        R_delta
        @ T_start[:3, :3]
    )

    return T_target


def main():

    model = RM75Model(
        side="left",
    )

    # =========================================================
    # Initial safe configuration
    # =========================================================

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

    q_preferred = (
        q_start.copy()
    )

    T_start = (
        forward_kinematics(
            q_start,
            model=model,
        )
    )

    # =========================================================
    # Simulation
    # =========================================================

    frequency = 50.0
    duration = 6.0

    dt = (
        1.0
        / frequency
    )

    # 当前只用于仿真验证。
    # 最终实机值后续重新标定。

    teleop_qd_limit = (
        0.25
        * model.qd_max
    )

    num_frames = int(
        frequency
        * duration
    )

    # =========================================================
    # Safety thresholds
    #
    # 当前只是仿真阶段初始值。
    # =========================================================

    sigma_warn = 0.020

    sigma_stop = 0.010

    # =========================================================
    # Previous safe state
    # =========================================================

    T_safe = (
        T_start.copy()
    )

    q_safe = (
        q_start.copy()
    )

    # =========================================================
    # Statistics
    # =========================================================

    projected_frames = []

    warning_frames = []

    rate_limited_frames = []

    alpha_history = []

    sigma_history = []

    solve_time_history = []

    q_history = [
        q_safe.copy()
    ]

    raw_ik_failure_count = 0

    print("")

    print(
        "=================================="
    )

    print(
        "RM75 TARGET FEASIBILITY TEST"
    )

    print(
        "=================================="
    )

    print("")

    print(
        "sigma_warn:",
        sigma_warn,
    )

    print(
        "sigma_stop:",
        sigma_stop,
    )

    # =========================================================
    # IK settings
    # =========================================================

    ik_kwargs = {
        "max_iterations":
            20,

        "position_tolerance":
            1e-4,

        "orientation_tolerance":
            1e-3,

        "damping":
            0.02,

        "step_gain":
            0.7,

        "max_joint_step":
            np.deg2rad(
                2.0
            ),

        "preferred_posture":
            q_preferred,

        "preferred_posture_gain":
            1.0,

        "max_null_step":
            np.deg2rad(
                0.10
            ),
    }

    # =========================================================
    # Main loop
    # =========================================================

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

        T_raw = (
            generate_raw_target(
                T_start,
                phase,
            )
        )

        tic = (
            time.perf_counter()
        )

        projection = (
            project_target_to_feasible(
                T_safe=T_safe,

                T_raw=T_raw,

                q_safe=q_safe,

                model=model,

                sigma_stop=
                    sigma_stop,

                binary_iterations=
                    6,

                ik_kwargs=
                    ik_kwargs,
            )
        )

        elapsed = (
            time.perf_counter()
            - tic
        )

        solve_time_history.append(
            elapsed
        )

        # =====================================================
        # Projection 应始终返回安全状态
        # =====================================================

        if not projection[
            "success"
        ]:

            raise RuntimeError(
                f"Feasibility projection "
                f"failed at frame {frame}"
            )

        # =====================================================
        # Raw target 是否 IK 失败
        # =====================================================

        if not projection[
            "raw_ik_success"
        ]:

            raw_ik_failure_count += 1

        # =====================================================
        # 是否发生 Target Projection
        # =====================================================

        if projection[
            "projected"
        ]:

            projected_frames.append(
                frame
            )

        alpha_history.append(
            projection[
                "alpha"
            ]
        )

        sigma_history.append(
            projection[
                "sigma_min"
            ]
        )

        # =====================================================
        # Singularity warning
        # =====================================================

        if (
            projection[
                "sigma_min"
            ]
            < sigma_warn
        ):

            warning_frames.append(
                frame
            )

        # =====================================================
        # Accept safe state
        # =====================================================

        # =====================================================
        # Candidate joint target
        # =====================================================

        q_candidate = (
            projection[
                "q_safe"
            ]
        )

        # =====================================================
        # Joint velocity limiting
        # =====================================================

        (
            q_command,
            rate_limited,
        ) = limit_joint_velocity(
            q_current=q_safe,
            q_target=q_candidate,
            qd_limit=teleop_qd_limit,
            dt=dt,
        )

        if rate_limited:

            rate_limited_frames.append(
                frame
            )

        # =====================================================
        # Hard joint limit
        # =====================================================

        q_command = np.clip(
            q_command,
            model.q_min,
            model.q_max,
        )

        command_sigma = (
            minimum_singular_value(
                q_command,
                model,
            )
        )

        if (
            command_sigma
            < sigma_stop
        ):

            # 如果限速后的命令反而进入
            # 不安全奇异区域，
            # 本帧直接保持上一状态。

            q_command = (
                q_safe.copy()
            )

            T_command = (
                T_safe.copy()
            )

        else:

            T_command = (
                forward_kinematics(
                    q_command,
                    model=model,
                )
            )

        # 接受本帧真正执行状态

        q_safe = (
            q_command
        )

        T_safe = (
            T_command
        )

        # =====================================================
        # IMPORTANT:
        #
        # 真正执行的是 q_command，
        # 所以下一帧的 Cartesian safe state
        # 必须重新由 FK(q_command) 得到。
        # =====================================================

        q_safe = (
            q_command
        )

        T_safe = (
            forward_kinematics(
                q_safe,
                model=model,
            )
        )

        q_history.append(
            q_safe.copy()
        )

    # =========================================================
    # Statistics
    # =========================================================

    q_history = np.asarray(
        q_history
    )

    dq = np.diff(
        q_history,
        axis=0,
    )

    max_joint_jump_deg = (
        np.rad2deg(
            np.max(
                np.abs(dq)
            )
        )
    )

    alpha_history = np.asarray(
        alpha_history
    )

    sigma_history = np.asarray(
        sigma_history
    )

    solve_time_history = np.asarray(
        solve_time_history
    )

    # =========================================================
    # Results
    # =========================================================

    print("")

    print(
        "=================================="
    )

    print(
        "RESULT"
    )

    print(
        "=================================="
    )

    print("")

    print(
        "Frames:",
        num_frames,
    )

    print(
        "Projected frames:",
        len(
            projected_frames
        ),
    )

    print(
        "Raw IK failure count:",
        raw_ik_failure_count,
    )

    print(
        "Warning frames:",
        len(
            warning_frames
        ),
    )

    if projected_frames:

        print("")

        print(
            "First projected frame:",
            projected_frames[0],
        )

        print(
            "Last projected frame:",
            projected_frames[-1],
        )

    print("")

    print(
        "Minimum alpha:",
        np.min(
            alpha_history
        ),
    )

    print(
        "Minimum accepted sigma:",
        np.min(
            sigma_history
        ),
    )

    print("")

    print(
        "Average projection "
        "time [ms]:",
        np.mean(
            solve_time_history
        )
        * 1000.0,
    )

    print(
        "Max projection "
        "time [ms]:",
        np.max(
            solve_time_history
        )
        * 1000.0,
    )

    print("")

    print(
        "Max joint jump/frame [deg]:",
        max_joint_jump_deg,
    )

    print("")

    print(
        "Final preferred posture cost:",
        preferred_posture_cost(
            q_safe,
            q_preferred,
            model,
        ),
    )

    print("")

    print(
        "Rate-limited frames:",
        len(
            rate_limited_frames
        ),
    )

    print(
        "Teleop joint velocity "
        "limit [deg/s]:"
    )

    print(
        np.rad2deg(
            teleop_qd_limit
        )
    )

    # =========================================================
    # Final checks
    # =========================================================

    sigma_safe = (
        np.min(
            sigma_history
        )
        >= (
            sigma_stop
            - 1e-6
        )
    )

    continuity_ok = (
        max_joint_jump_deg
        < 2.0
    )

    if (
        sigma_safe
        and
        continuity_ok
    ):

        print("")

        print(
            "TARGET FEASIBILITY TEST: PASS"
        )

    else:

        print("")

        print(
            "TARGET FEASIBILITY TEST: FAIL"
        )


if __name__ == "__main__":
    main()