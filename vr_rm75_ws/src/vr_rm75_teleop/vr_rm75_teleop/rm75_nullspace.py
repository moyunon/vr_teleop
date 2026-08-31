import numpy as np


def null_space_basis(
    J,
    rcond=1e-6,
):
    """
    使用 SVD 求 Jacobian 的 Null Space 基。

    Returns
    -------
    Z:
        shape = (n, nullity)

        每一列都是一个单位 Null Space 基向量。

    rank:
        Jacobian 数值秩。

    singular_values:
        Jacobian 奇异值。
    """

    J = np.asarray(
        J,
        dtype=float,
    )

    U, singular_values, Vt = np.linalg.svd(
        J,
        full_matrices=True,
    )

    if singular_values.size == 0:
        raise ValueError(
            "Jacobian has no singular values."
        )

    tolerance = (
        rcond
        * singular_values[0]
    )

    rank = int(
        np.sum(
            singular_values > tolerance
        )
    )

    # Vt:
    #
    # shape = n x n
    #
    # 第 rank 行之后对应 Null Space。
    #
    # 转置以后：
    #
    # Z shape = n x (n-rank)

    Z = Vt[rank:, :].T

    return (
        Z,
        rank,
        singular_values,
    )


def null_space_vector(
    J,
    rcond=1e-6,
):
    """
    RM75 普通非奇异构型下：

        J shape = 6 x 7
        rank = 6

    Null Space 是 1 维。

    返回单位基向量 n。
    """

    Z, rank, _ = null_space_basis(
        J,
        rcond=rcond,
    )

    if Z.shape[1] != 1:
        raise RuntimeError(
            "Expected 1D null space, "
            f"but got dimension {Z.shape[1]} "
            f"(rank={rank})."
        )

    return Z[:, 0]


def null_space_projector(
    J,
    rcond=1e-6,
):
    """
    构造严格的正交 Null Space projector：

        N = Z Z^T

    对任何向量 v：

        dq_null = N v

    都满足：

        J dq_null ~= 0
    """

    Z, rank, singular_values = (
        null_space_basis(
            J,
            rcond=rcond,
        )
    )

    if Z.shape[1] == 0:
        N = np.zeros(
            (
                J.shape[1],
                J.shape[1],
            )
        )
    else:
        N = Z @ Z.T

    return (
        N,
        rank,
        singular_values,
    )


def joint_centering_cost(
    q,
    model,
):
    """
    关节居中代价。

    每个关节首先按照自己的运动范围归一化：

        u_i =
        (q_i - q_mid_i)
        -----------------
          half_range_i

    关节正好位于中间：
        u_i = 0

    到达上下极限：
        |u_i| = 1
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    q_mid = (
        model.q_min
        + model.q_max
    ) / 2.0

    half_range = (
        model.q_max
        - model.q_min
    ) / 2.0

    normalized = (
        q - q_mid
    ) / half_range

    cost = (
        0.5
        * np.sum(
            normalized ** 2
        )
    )

    return float(cost)


def joint_centering_direction(
    q,
    model,
):
    """
    joint_centering_cost 的负梯度方向。

    它表示：

        如果暂时不考虑末端任务，
        关节应该朝哪个方向运动，
        才能更靠近关节范围中央。
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    q_mid = (
        model.q_min
        + model.q_max
    ) / 2.0

    half_range = (
        model.q_max
        - model.q_min
    ) / 2.0

    # cost =
    #
    # 1/2 * sum(
    #   ((q-q_mid)/half_range)^2
    # )
    #
    # negative gradient:

    direction = -(
        q - q_mid
    ) / (
        half_range ** 2
    )

    return direction

def preferred_posture_cost(
    q,
    q_preferred,
    model,
):
    """
    衡量当前关节构型距离 preferred posture 有多远。

    使用每个关节自己的运动范围进行归一化，
    避免 J7 ±360° 和 J6 ±128° 被同等对待。
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    q_preferred = np.asarray(
        q_preferred,
        dtype=float,
    )

    if q.shape != (model.DOF,):
        raise ValueError(
            f"q must have shape ({model.DOF},)"
        )

    if q_preferred.shape != (model.DOF,):
        raise ValueError(
            f"q_preferred must have shape ({model.DOF},)"
        )

    half_range = (
        model.q_max
        - model.q_min
    ) / 2.0

    normalized_error = (
        q - q_preferred
    ) / half_range

    cost = (
        0.5
        * np.sum(
            normalized_error ** 2
        )
    )

    return float(cost)


def preferred_posture_direction(
    q,
    q_preferred,
    model,
):
    """
    preferred_posture_cost 的负梯度。

    这是“不考虑末端任务时，希望关节运动的方向”。

    后续必须经过 Null Space projector，
    不能直接施加到机械臂。
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    q_preferred = np.asarray(
        q_preferred,
        dtype=float,
    )

    half_range = (
        model.q_max
        - model.q_min
    ) / 2.0

    direction = -(
        q - q_preferred
    ) / (
        half_range ** 2
    )

    return direction