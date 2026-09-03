import numpy as np

from vr_rm75_teleop.rm75_fk import forward_kinematics


def geometric_jacobian(q, model=None):
    """
    计算 RM75 的 6x7 几何 Jacobian。

    Jacobian 表达在机械臂 base frame 中。

    返回：
        J =
        [ Jv ]
        [ Jw ]

    其中：
        Jv: 3x7 线速度 Jacobian，长度采用模型的 m
        Jw: 3x7 角速度 Jacobian，角度采用 rad

    注意：直接对 [Jv; Jw] 做 SVD 会混合平移和旋转尺度，得到的
    sigma 只应与使用相同 m/rad 单位和相同堆叠定义的阈值比较。
    """

    if model is None:
        raise ValueError(
            "geometric_jacobian() requires an explicit "
            "RM75Model(side='left' or 'right')."
        )

    q = np.asarray(
        q,
        dtype=float,
    )

    if q.shape != (model.DOF,):
        raise ValueError(
            f"q must have shape ({model.DOF},), "
            f"but got {q.shape}"
        )

    # FK 给出 T_00 ~ T_07
    T_07, transforms = forward_kinematics(
        q,
        model=model,
        return_all=True,
    )

    # 末端在 base frame 下的位置
    p_e = T_07[:3, 3]

    J = np.zeros(
        (6, model.DOF),
        dtype=float,
    )

    for i in range(model.DOF):

        # ---------------------------------------------------------
        # Standard DH:
        #
        # 第 i+1 个关节绕 z_i 旋转
        #
        # 所以第 i 列使用 T_0i
        # ---------------------------------------------------------

        T_0i = transforms[i]

        # 当前关节轴原点
        p_i = T_0i[:3, 3]

        # 当前关节 z 轴
        z_i = T_0i[:3, 2]

        # 线速度部分
        J_v = np.cross(
            z_i,
            p_e - p_i,
        )

        # 角速度部分
        J_w = z_i

        J[:3, i] = J_v
        J[3:, i] = J_w

    return J
