import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_ik import solve_ik
from vr_rm75_teleop.rm75_jacobian import (
    geometric_jacobian,
)


def interpolate_pose(
    T_start,
    T_end,
    alpha,
):
    """
    在两个 SE(3) Pose 之间进行插值。

    alpha = 0:
        T_start

    alpha = 1:
        T_end

    Position:
        线性插值

    Orientation:
        使用相对 rotation vector 插值
        （等价于沿最短旋转路径插值）
    """

    T_start = np.asarray(
        T_start,
        dtype=float,
    )

    T_end = np.asarray(
        T_end,
        dtype=float,
    )

    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )

    # =========================================================
    # Position
    # =========================================================

    p_start = T_start[:3, 3]
    p_end = T_end[:3, 3]

    p = (
        p_start
        + alpha
        * (
            p_end
            - p_start
        )
    )

    # =========================================================
    # Orientation
    # =========================================================

    R_start = T_start[:3, :3]
    R_end = T_end[:3, :3]

    # Spatial relative rotation
    R_relative = (
        R_end
        @ R_start.T
    )

    relative_rotvec = (
        Rotation
        .from_matrix(
            R_relative
        )
        .as_rotvec()
    )

    R_delta = (
        Rotation
        .from_rotvec(
            alpha
            * relative_rotvec
        )
        .as_matrix()
    )

    R = (
        R_delta
        @ R_start
    )

    # =========================================================
    # Compose
    # =========================================================

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, :3] = R
    T[:3, 3] = p

    return T


def minimum_singular_value(
    q,
    model,
):
    """
    返回当前 RM75 Jacobian 的最小奇异值。
    """

    J = geometric_jacobian(
        q,
        model=model,
    )

    singular_values = np.linalg.svd(
        J,
        compute_uv=False,
    )

    return float(
        singular_values[-1]
    )


def try_target(
    T_target,
    q_seed,
    model,

    sigma_stop=0.010,

    ik_kwargs=None,
):
    """
    尝试求解一个 Cartesian Target。

    Target 被认为可以接受必须同时满足：

    1. IK success
    2. sigma_min >= sigma_stop
    """

    if ik_kwargs is None:
        ik_kwargs = {}

    result = solve_ik(
        T_target=T_target,
        q_seed=q_seed,
        model=model,
        **ik_kwargs,
    )

    # =========================================================
    # IK 本身失败
    # =========================================================

    if not result[
        "success"
    ]:

        return {
            "acceptable":
                False,

            "ik_success":
                False,

            "q":
                result["q"],

            "sigma_min":
                None,

            "ik_result":
                result,
        }

    # =========================================================
    # IK 成功后检查奇异性
    # =========================================================

    q = result["q"]

    sigma_min = (
        minimum_singular_value(
            q,
            model,
        )
    )

    acceptable = (
        sigma_min
        >= sigma_stop
    )

    return {
        "acceptable":
            acceptable,

        "ik_success":
            True,

        "q":
            q,

        "sigma_min":
            sigma_min,

        "ik_result":
            result,
    }


def project_target_to_feasible(
    T_safe,
    T_raw,
    q_safe,
    model,

    sigma_stop=0.010,

    binary_iterations=6,

    ik_kwargs=None,
):
    """
    将 Raw Cartesian Target 投影到
    当前机械臂的安全可达区域。

    ------------------------------------------------------------

    首先直接尝试 T_raw。

    如果：

        IK success
        &&
        sigma_min >= sigma_stop

    则直接接受：

        alpha = 1

    ------------------------------------------------------------

    如果 Raw Target 不可接受：

        T_safe -------- T_raw
           |             |
         alpha=0       alpha=1

    使用二分搜索寻找最大的安全 alpha。

    ------------------------------------------------------------

    Returns
    -------

    {
        success,
        projected,
        alpha,
        T_safe,
        q_safe,
        sigma_min,
        raw_ik_success,
    }
    """

    if ik_kwargs is None:
        ik_kwargs = {}

    # =========================================================
    # 1. 先直接尝试 Raw Target
    # =========================================================

    raw_test = try_target(
        T_target=T_raw,
        q_seed=q_safe,
        model=model,

        sigma_stop=sigma_stop,

        ik_kwargs=ik_kwargs,
    )

    if raw_test[
        "acceptable"
    ]:

        return {
            "success":
                True,

            "projected":
                False,

            "alpha":
                1.0,

            "T_safe":
                T_raw.copy(),

            "q_safe":
                raw_test["q"],

            "sigma_min":
                raw_test[
                    "sigma_min"
                ],

            "raw_ik_success":
                True,
        }

    # =========================================================
    # 2. Raw Target 不可接受
    #
    # 从上一帧安全 Target 开始二分。
    # =========================================================

    low = 0.0
    high = 1.0

    best_alpha = 0.0

    best_T = (
        T_safe.copy()
    )

    best_q = (
        q_safe.copy()
    )

    # 上一帧 q_safe 理论上已经通过安全检查。
    best_sigma = (
        minimum_singular_value(
            q_safe,
            model,
        )
    )

    # =========================================================
    # 3. Binary search
    # =========================================================

    for _ in range(
        binary_iterations
    ):

        mid = (
            low
            + high
        ) * 0.5

        T_candidate = (
            interpolate_pose(
                T_safe,
                T_raw,
                mid,
            )
        )

        candidate_test = (
            try_target(
                T_target=T_candidate,
                q_seed=best_q,
                model=model,

                sigma_stop=
                    sigma_stop,

                ik_kwargs=
                    ik_kwargs,
            )
        )

        # -----------------------------------------------------
        # Candidate 可以接受：
        # 尝试继续靠近 Raw Target
        # -----------------------------------------------------

        if candidate_test[
            "acceptable"
        ]:

            low = mid

            best_alpha = mid

            best_T = (
                T_candidate
            )

            best_q = (
                candidate_test["q"]
            )

            best_sigma = (
                candidate_test[
                    "sigma_min"
                ]
            )

        # -----------------------------------------------------
        # Candidate 不可接受：
        # 向安全端退回
        # -----------------------------------------------------

        else:

            high = mid

    # =========================================================
    # 4. 最终结果
    # =========================================================

    return {
        "success":
            True,

        "projected":
            True,

        "alpha":
            best_alpha,

        "T_safe":
            best_T,

        "q_safe":
            best_q,

        "sigma_min":
            best_sigma,

        "raw_ik_success":
            raw_test[
                "ik_success"
            ],
    }