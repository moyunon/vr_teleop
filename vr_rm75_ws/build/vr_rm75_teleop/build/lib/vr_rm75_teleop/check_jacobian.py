import numpy as np

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.rm75_jacobian import geometric_jacobian


np.set_printoptions(
    precision=6,
    suppress=True,
)


def vee(skew_matrix):
    """
    将反对称矩阵：

        [ 0  -wz  wy]
        [ wz  0  -wx]
        [-wy wx   0 ]

    转换为：

        [wx, wy, wz]
    """

    return np.array(
        [
            skew_matrix[2, 1],
            skew_matrix[0, 2],
            skew_matrix[1, 0],
        ]
    )


def numerical_jacobian(
    q,
    model,
    eps=1e-7,
):
    """
    使用中心有限差分，对 FK 数值求导。

    用它作为独立参考，
    验证 geometric_jacobian()。
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    J_num = np.zeros(
        (6, model.DOF)
    )

    # 当前姿态
    T_0 = forward_kinematics(
        q,
        model=model,
    )

    R_0 = T_0[:3, :3]

    for i in range(model.DOF):

        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[i] += eps
        q_minus[i] -= eps

        T_plus = forward_kinematics(
            q_plus,
            model=model,
        )

        T_minus = forward_kinematics(
            q_minus,
            model=model,
        )

        # ========================================================
        # Position numerical derivative
        # ========================================================

        p_plus = T_plus[:3, 3]
        p_minus = T_minus[:3, 3]

        dp_dq = (
            p_plus - p_minus
        ) / (2.0 * eps)

        J_num[:3, i] = dp_dq

        # ========================================================
        # Orientation numerical derivative
        # ========================================================

        R_plus = T_plus[:3, :3]
        R_minus = T_minus[:3, :3]

        # 数值计算 dR / dq_i
        R_dot = (
            R_plus - R_minus
        ) / (2.0 * eps)

        # 对 spatial angular velocity:
        #
        # omega_hat = R_dot * R^T
        #
        omega_hat = (
            R_dot @ R_0.T
        )

        omega = vee(
            omega_hat
        )

        J_num[3:, i] = omega

    return J_num


def main():

    model = RM75Model(
        side="left",
    )

    # 不在零位测试。
    # 零位附近机械臂可能处于奇异/特殊构型，
    # 不利于全面检查 Jacobian。

    q = np.deg2rad(
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
    print("==============================")
    print("RM75-6FB JACOBIAN TEST")
    print("==============================")

    print("")
    print("q [deg] =")
    print(
        np.rad2deg(q)
    )

    # ------------------------------------------------------------
    # Analytical geometric Jacobian
    # ------------------------------------------------------------

    J = geometric_jacobian(
        q,
        model=model,
    )

    print("")
    print("Analytical Jacobian:")
    print(J)

    # ------------------------------------------------------------
    # Numerical finite-difference Jacobian
    # ------------------------------------------------------------

    J_num = numerical_jacobian(
        q,
        model=model,
    )

    print("")
    print("Numerical Jacobian:")
    print(J_num)

    # ------------------------------------------------------------
    # Error
    # ------------------------------------------------------------

    error = J - J_num

    print("")
    print("Error:")
    print(error)

    translation_error = np.max(
        np.abs(error[:3, :])
    )

    rotation_error = np.max(
        np.abs(error[3:, :])
    )

    print("")
    print(
        "Max translation error:",
        translation_error,
    )

    print(
        "Max rotation error:",
        rotation_error,
    )

    # ------------------------------------------------------------
    # Rank
    # ------------------------------------------------------------

    rank = np.linalg.matrix_rank(
        J,
    )

    print("")
    print(
        "Jacobian rank:",
        rank,
    )

    # ------------------------------------------------------------
    # Singular values
    # ------------------------------------------------------------

    singular_values = np.linalg.svd(
        J,
        compute_uv=False,
    )

    print(
        "Singular values:",
        singular_values,
    )

    # ------------------------------------------------------------
    # Test
    # ------------------------------------------------------------

    tolerance = 1e-6

    if (
        translation_error < tolerance
        and rotation_error < tolerance
    ):
        print("")
        print(
            "JACOBIAN TEST: PASS"
        )
    else:
        print("")
        print(
            "JACOBIAN TEST: FAIL"
        )


if __name__ == "__main__":
    main()