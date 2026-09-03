import numpy as np

from vr_rm75_teleop.rm75_model import RM75Model

# 从第 i-1 个连杆坐标系到第 i 个连杆坐标系的 4×4 齐次变换矩阵
def sdh_transform(theta, d, a, alpha):
    """
    Standard Denavit-Hartenberg transformation.

    A_i =
        Rot_z(theta)
        Trans_z(d)
        Trans_x(a)
        Rot_x(alpha)
    """

    ct = np.cos(theta)
    st = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array(
        [
            [
                ct,
                -st * ca,
                st * sa,
                a * ct,
            ],
            [
                st,
                ct * ca,
                -ct * sa,
                a * st,
            ],
            [
                0.0,
                sa,
                ca,
                d,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=float,
    )


def forward_kinematics(q, model=None, return_all=False):
    """
    RM75 forward kinematics.

    Parameters
    ----------
    q:
        7 个机械臂关节角，单位 rad。

    model:
        RM75Model。
        如果不传入，则使用标准版 d7=0.144 m。

    return_all:
        False:
            只返回 T_07。

        True:
            同时返回 T_00 ~ T_07，
            后续计算 Jacobian 会直接使用这些矩阵。
    """

    if model is None:
        raise ValueError(
            "forward_kinematics() requires an explicit "
            "RM75Model(side='left' or 'right')."
        )

    q = np.asarray(q, dtype=float)

    if q.shape != (model.DOF,):
        raise ValueError(
            f"q must have shape ({model.DOF},), "
            f"but got {q.shape}"
        )

    # T_00
    T = np.eye(4)

    transforms = [T.copy()]

    for i in range(model.DOF):

        theta_i = (
            q[i]
            + model.theta_offset[i]
        )

        A_i = sdh_transform(
            theta=theta_i,
            d=model.d[i],
            a=model.a[i],
            alpha=model.alpha[i],
        )

        T = T @ A_i

        transforms.append(T.copy())

    if return_all:
        return T, transforms

    return T