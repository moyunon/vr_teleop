import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.rm75_jacobian import geometric_jacobian

from vr_rm75_teleop.rm75_nullspace import (
    null_space_projector,
    preferred_posture_direction,
    preferred_posture_cost,
)


def pose_error(
    T_current,
    T_target,
):
    """
    计算当前末端位姿到目标末端位姿的 6D 误差。

    返回：
        [dx, dy, dz, rx, ry, rz]

    position error:
        meter

    rotation error:
        rotation vector, rad

    所有误差均表达在 robot base frame 中。
    """

    # =========================================================
    # Position error
    # =========================================================

    p_current = T_current[:3, 3]
    p_target = T_target[:3, 3]

    position_error = (
        p_target - p_current
    )

    # =========================================================
    # Orientation error
    # =========================================================

    R_current = T_current[:3, :3]
    R_target = T_target[:3, :3]

    # Spatial / base-frame orientation error
    #
    # geometric Jacobian 的角速度部分同样表达在 base frame，
    # 因此这里采用：
    #
    # R_error = R_target @ R_current.T

    R_error = (
        R_target
        @ R_current.T
    )

    rotation_error = (
        Rotation
        .from_matrix(R_error)
        .as_rotvec()
    )

    return np.concatenate(
        (
            position_error,
            rotation_error,
        )
    )


def damped_least_squares_step(
    J,
    error,
    damping=0.02,
):
    """
    Damped Least Squares:

        dq =
        J^T
        (J J^T + lambda^2 I)^-1
        error

    使用 np.linalg.solve()，
    避免显式求逆。
    """

    J = np.asarray(
        J,
        dtype=float,
    )

    error = np.asarray(
        error,
        dtype=float,
    )

    if J.ndim != 2:
        raise ValueError(
            f"J must be 2-D, "
            f"got shape {J.shape}"
        )

    if error.shape != (
        J.shape[0],
    ):
        raise ValueError(
            "error dimension must match "
            "Jacobian row dimension, "
            f"got error {error.shape}, "
            f"J {J.shape}"
        )

    identity = np.eye(
        J.shape[0],
        dtype=float,
    )

    A = (
        J @ J.T
        + (damping ** 2)
        * identity
    )

    y = np.linalg.solve(
        A,
        error,
    )

    dq = (
        J.T
        @ y
    )

    return dq


def solve_ik(
    T_target,
    q_seed,
    model=None,

    max_iterations=200,

    position_tolerance=1e-4,
    orientation_tolerance=1e-3,

    damping=0.02,
    step_gain=0.5,

    max_joint_step=np.deg2rad(
        5.0
    ),

    orientation_weight=1.0,

    # =========================================================
    # Secondary task:
    # Preferred Posture
    # =========================================================

    preferred_posture=None,

    preferred_posture_gain=1.0,

    max_null_step=np.deg2rad(
        0.10
    ),

    null_rcond=1e-6,
):
    """
    RM75-6FB Numerical IK Solver.

    Primary task:
        DLS Cartesian pose tracking.

    Secondary task:
        Preferred-posture motion projected
        into the exact Jacobian Null Space.

    实时控制原则：
        Primary task 一旦满足误差阈值，
        本帧立即返回。

        不为了继续优化 preferred posture
        而在目标已经到达之后继续运动。
    """

    if model is None:
        raise ValueError(
            "solve_ik() requires an explicit "
            "RM75Model(side='left' or 'right')."
        )

    # =========================================================
    # Input preparation
    # =========================================================

    T_target = np.asarray(
        T_target,
        dtype=float,
    )

    q = np.asarray(
        q_seed,
        dtype=float,
    ).copy()

    if T_target.shape != (
        4,
        4,
    ):
        raise ValueError(
            "T_target must have shape "
            f"(4, 4), got {T_target.shape}"
        )

    if q.shape != (
        model.DOF,
    ):
        raise ValueError(
            "q_seed must have shape "
            f"({model.DOF},), "
            f"got {q.shape}"
        )

    # =========================================================
    # Preferred posture validation
    # =========================================================

    if preferred_posture is not None:

        preferred_posture = np.asarray(
            preferred_posture,
            dtype=float,
        )

        if preferred_posture.shape != (
            model.DOF,
        ):
            raise ValueError(
                "preferred_posture must "
                "have shape "
                f"({model.DOF},), "
                f"got "
                f"{preferred_posture.shape}"
            )

        if (
            np.any(
                preferred_posture
                < model.q_min
            )
            or
            np.any(
                preferred_posture
                > model.q_max
            )
        ):
            raise ValueError(
                "preferred_posture is "
                "outside RM75 joint limits"
            )

    # =========================================================
    # Initial joint-limit clamp
    # =========================================================

    q = np.clip(
        q,
        model.q_min,
        model.q_max,
    )

    # =========================================================
    # IK iteration
    # =========================================================

    for iteration in range(
        max_iterations
    ):

        # =====================================================
        # 1. Forward Kinematics
        # =====================================================

        T_current = (
            forward_kinematics(
                q,
                model=model,
            )
        )

        # =====================================================
        # 2. Cartesian pose error
        # =====================================================

        error = pose_error(
            T_current,
            T_target,
        )

        position_error_norm = (
            np.linalg.norm(
                error[:3]
            )
        )

        orientation_error_norm = (
            np.linalg.norm(
                error[3:]
            )
        )

        # =====================================================
        # 3. Primary task convergence
        # =====================================================

        if (
            position_error_norm
            < position_tolerance
            and
            orientation_error_norm
            < orientation_tolerance
        ):

            posture_cost = None

            if (
                preferred_posture
                is not None
            ):

                posture_cost = (
                    preferred_posture_cost(
                        q,
                        preferred_posture,
                        model,
                    )
                )

            return {
                "success":
                    True,

                "q":
                    q,

                "iterations":
                    iteration,

                "position_error":
                    position_error_norm,

                "orientation_error":
                    orientation_error_norm,

                "preferred_posture_cost":
                    posture_cost,
            }

        # =====================================================
        # 4. Geometric Jacobian
        # =====================================================

        J = geometric_jacobian(
            q,
            model=model,
        )

        # =====================================================
        # 5. Primary task:
        #    Damped Least Squares
        # =====================================================

        weighted_error = (
            error.copy()
        )

        weighted_J = (
            J.copy()
        )

        weighted_error[3:] *= (
            orientation_weight
        )

        weighted_J[3:, :] *= (
            orientation_weight
        )

        dq_task = (
            damped_least_squares_step(
                weighted_J,
                weighted_error,
                damping=damping,
            )
        )

        dq_task *= (
            step_gain
        )

        # =====================================================
        # 6. Secondary task:
        #    Preferred Posture
        # =====================================================

        dq_null = np.zeros(
            model.DOF,
            dtype=float,
        )

        if (
            preferred_posture
            is not None
        ):

            posture_direction = (
                preferred_posture_direction(
                    q,
                    preferred_posture,
                    model,
                )
            )

            (
                N,
                _,
                _,
            ) = (
                null_space_projector(
                    J,
                    rcond=null_rcond,
                )
            )

            # 只保留 preferred posture
            # 中属于 Jacobian Null Space
            # 的运动分量。

            dq_null = (
                preferred_posture_gain
                * (
                    N
                    @ posture_direction
                )
            )

            # ===============================================
            # Null-space step limit
            # ===============================================

            null_norm = (
                np.linalg.norm(
                    dq_null
                )
            )

            if (
                null_norm
                > max_null_step
            ):

                dq_null *= (
                    max_null_step
                    / null_norm
                )

        # =====================================================
        # 7. Primary + Secondary
        # =====================================================

        dq = (
            dq_task
            + dq_null
        )

        # =====================================================
        # 8. IK internal iteration step limit
        #
        # 注意：
        # 这只是数值 IK 内部的单次迭代限制，
        # 不是未来真实机械臂的关节速度限制。
        # =====================================================

        max_abs_step = (
            np.max(
                np.abs(dq)
            )
        )

        if (
            max_abs_step
            > max_joint_step
        ):

            dq *= (
                max_joint_step
                / max_abs_step
            )

        # =====================================================
        # 9. Update
        # =====================================================

        q = (
            q + dq
        )

        # =====================================================
        # 10. Hard joint-limit clamp
        #
        # 当前阶段只作为数值保护。
        #
        # 后续实机前还需要：
        #   - joint-limit avoidance
        #   - joint-rate limiter
        # =====================================================

        q = np.clip(
            q,
            model.q_min,
            model.q_max,
        )

    # =========================================================
    # Maximum iterations reached
    # =========================================================

    T_current = (
        forward_kinematics(
            q,
            model=model,
        )
    )

    final_error = pose_error(
        T_current,
        T_target,
    )

    final_position_error = (
        np.linalg.norm(
            final_error[:3]
        )
    )

    final_orientation_error = (
        np.linalg.norm(
            final_error[3:]
        )
    )

    task_success = (
        final_position_error
        < position_tolerance
        and
        final_orientation_error
        < orientation_tolerance
    )

    posture_cost = None

    if (
        preferred_posture
        is not None
    ):

        posture_cost = (
            preferred_posture_cost(
                q,
                preferred_posture,
                model,
            )
        )

    return {
        "success":
            task_success,

        "q":
            q,

        "iterations":
            max_iterations,

        "position_error":
            final_position_error,

        "orientation_error":
            final_orientation_error,

        "preferred_posture_cost":
            posture_cost,
    }