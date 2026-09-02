"""Joint-space safety primitives shared by offline and ROS control paths."""

import math

import numpy as np


def make_teleop_soft_limits(
    hard_min,
    hard_max,
    joint_margin,
    elbow_index,
    elbow_branch,
    elbow_margin,
):
    """Build strict hard-limit margins and preserve one elbow branch.

    ``elbow_branch`` is -1 for a negative-angle branch and +1 for a
    positive-angle branch. The resulting elbow interval cannot reach zero.
    """
    try:
        hard_min = np.asarray(hard_min, dtype=float)
        hard_max = np.asarray(hard_max, dtype=float)
        joint_margin = np.asarray(joint_margin, dtype=float)
        elbow_index = int(elbow_index)
        elbow_branch = int(elbow_branch)
        elbow_margin = float(elbow_margin)
    except (TypeError, ValueError) as exc:
        raise ValueError("soft-limit inputs must be numeric") from exc

    if hard_min.shape != hard_max.shape or hard_min.ndim != 1:
        raise ValueError("hard_min and hard_max must be equal 1-D vectors")
    if not np.all(np.isfinite(hard_min)) or not np.all(np.isfinite(hard_max)):
        raise ValueError("hard limits must be finite")
    if np.any(hard_min >= hard_max):
        raise ValueError("every hard minimum must be below its maximum")

    if joint_margin.ndim == 0:
        joint_margin = np.full(hard_min.shape, float(joint_margin))
    if joint_margin.shape != hard_min.shape:
        raise ValueError("joint_margin must be scalar or match hard limits")
    if (
        not np.all(np.isfinite(joint_margin))
        or np.any(joint_margin <= 0.0)
    ):
        raise ValueError("joint_margin must be finite and strictly positive")
    if elbow_index < 0 or elbow_index >= hard_min.size:
        raise ValueError("elbow_index is outside the joint vector")
    if elbow_branch not in (-1, 1):
        raise ValueError("elbow_branch must be -1 or +1")
    if not math.isfinite(elbow_margin) or elbow_margin <= 0.0:
        raise ValueError("elbow_margin must be finite and positive")

    soft_min = hard_min + joint_margin
    soft_max = hard_max - joint_margin
    if elbow_branch < 0:
        soft_max[elbow_index] = min(
            soft_max[elbow_index],
            -elbow_margin,
        )
    else:
        soft_min[elbow_index] = max(
            soft_min[elbow_index],
            elbow_margin,
        )

    if np.any(soft_min >= soft_max):
        raise ValueError("configured soft-limit interval is empty")
    if np.any(soft_min <= hard_min) or np.any(soft_max >= hard_max):
        raise ValueError("soft limits must lie strictly inside hard limits")
    return soft_min, soft_max


def limit_joint_soft_position(
    q_current,
    q_target,
    soft_min,
    soft_max,
):
    """Saturate one rate-limited target without snapping invalid state."""
    try:
        q_current = np.asarray(q_current, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        soft_min = np.asarray(soft_min, dtype=float)
        soft_max = np.asarray(soft_max, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint soft-limit inputs must be numeric") from exc

    shape = q_current.shape
    if (
        q_current.ndim != 1
        or q_target.shape != shape
        or soft_min.shape != shape
        or soft_max.shape != shape
    ):
        raise ValueError("joint soft-limit inputs must be equal 1-D vectors")
    if not all(
        np.all(np.isfinite(item))
        for item in (q_current, q_target, soft_min, soft_max)
    ):
        raise ValueError("joint soft-limit inputs must be finite")
    if np.any(soft_min >= soft_max):
        raise ValueError("every soft minimum must be below its maximum")
    if np.any(q_current < soft_min) or np.any(q_current > soft_max):
        raise ValueError("q_current is outside teleoperation soft limits")

    q_command = np.clip(q_target, soft_min, soft_max)
    soft_limited = not np.array_equal(q_command, q_target)
    return q_command, soft_limited


def limit_joint_velocity(
    q_current,
    q_target,
    qd_limit,
    dt,
):
    """Limit one joint target using the actual control interval.

    Parameters
    ----------
    q_current:
        当前实际/命令关节角，rad。

    q_target:
        IK 给出的目标关节角，rad。

    qd_limit:
        每个关节允许的最大速度，rad/s。

    dt:
        控制周期，s。

    Returns
    -------
    q_command:
        本周期真正允许执行的关节角。

    rate_limited:
        是否发生了速度限制。
    """

    try:
        q_current = np.asarray(q_current, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        qd_limit = np.asarray(qd_limit, dtype=float)
        dt = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "joint velocity limiter inputs must be numeric"
        ) from exc

    if q_current.shape != q_target.shape:
        raise ValueError(
            "q_current and q_target "
            "must have the same shape"
        )

    if qd_limit.shape != q_current.shape:
        raise ValueError(
            "qd_limit must have the "
            "same shape as q"
        )

    if q_current.ndim != 1:
        raise ValueError("joint vectors must be one-dimensional")

    if (
        not np.all(np.isfinite(q_current))
        or not np.all(np.isfinite(q_target))
        or not np.all(np.isfinite(qd_limit))
    ):
        raise ValueError("joint velocity limiter inputs must be finite")

    if np.any(qd_limit < 0.0):
        raise ValueError("qd_limit must be non-negative")

    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            "dt must be finite and > 0"
        )

    delta_q = (
        q_target
        - q_current
    )

    max_delta_q = (
        qd_limit
        * dt
    )

    limited_delta_q = np.clip(
        delta_q,
        -max_delta_q,
        max_delta_q,
    )

    q_command = (
        q_current
        + limited_delta_q
    )

    rate_limited = not np.array_equal(
        delta_q,
        limited_delta_q,
    )

    return (
        q_command,
        rate_limited,
    )


def limit_joint_acceleration(
    q_current,
    q_target,
    qd_current,
    qdd_limit,
    dt,
):
    """
    Limit the change in commanded joint velocity over one cycle.

    This limiter shapes ordinary trajectory updates. Emergency safety gates
    remain free to stop transmitting immediately instead of attempting a
    gradual software deceleration.
    """
    try:
        q_current = np.asarray(q_current, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        qd_current = np.asarray(qd_current, dtype=float)
        qdd_limit = np.asarray(qdd_limit, dtype=float)
        dt = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "joint acceleration limiter inputs must be numeric"
        ) from exc

    shape = q_current.shape
    if (
        q_current.ndim != 1
        or q_target.shape != shape
        or qd_current.shape != shape
        or qdd_limit.shape != shape
    ):
        raise ValueError(
            "joint acceleration limiter inputs must be equal 1-D vectors"
        )
    if not all(
        np.all(np.isfinite(item))
        for item in (q_current, q_target, qd_current, qdd_limit)
    ):
        raise ValueError(
            "joint acceleration limiter inputs must be finite"
        )
    if np.any(qdd_limit < 0.0):
        raise ValueError("qdd_limit must be non-negative")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")

    position_error = q_target - q_current
    qd_desired = position_error / dt

    # Do not arrive at a fixed target with a velocity that cannot be reduced
    # smoothly afterwards. This conservative continuous stopping-distance
    # cap also prevents the normal limiter from oscillating across a target.
    for index, acceleration_limit in enumerate(qdd_limit):
        if acceleration_limit == 0.0:
            qd_desired[index] = 0.0
            continue
        distance = abs(position_error[index])
        stopping_speed = (
            -acceleration_limit * dt
            + math.sqrt(
                (acceleration_limit * dt) ** 2
                + 2.0 * acceleration_limit * distance
            )
        )
        qd_desired[index] = np.clip(
            qd_desired[index],
            -stopping_speed,
            stopping_speed,
        )
    max_delta_qd = qdd_limit * dt
    delta_qd = qd_desired - qd_current
    limited_delta_qd = np.clip(
        delta_qd,
        -max_delta_qd,
        max_delta_qd,
    )
    qd_command = qd_current + limited_delta_qd
    q_command = q_current + qd_command * dt
    acceleration_limited = not np.array_equal(
        delta_qd,
        limited_delta_qd,
    )
    return q_command, qd_command, acceleration_limited
