import numpy as np


def limit_joint_velocity(
    q_current,
    q_target,
    qd_limit,
    dt,
):
    """
    对一帧关节目标进行速度限制。

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

    q_current = np.asarray(
        q_current,
        dtype=float,
    )

    q_target = np.asarray(
        q_target,
        dtype=float,
    )

    qd_limit = np.asarray(
        qd_limit,
        dtype=float,
    )

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

    if dt <= 0.0:
        raise ValueError(
            "dt must be > 0"
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

    rate_limited = not np.allclose(
        delta_q,
        limited_delta_q,
        atol=1e-12,
        rtol=0.0,
    )

    return (
        q_command,
        rate_limited,
    )