import numpy as np


class RM75Model:
    """
    RealBot 上 RM75-6FB 的标准 D-H 数学模型。

    注意：
    q 始终表示机械臂控制器使用的关节角。

    DH 中真正参与计算的角度为：

        theta = q + theta_offset

    当前实机控制器 get_DH_data 验证结果：

    LEFT:
        joint_7 offset = +180 deg

    RIGHT:
        joint_7 offset = 0 deg
    """

    DOF = 7

    VALID_SIDES = (
        "left",
        "right",
    )

    def __init__(
        self,
        side,
    ):

        # =====================================================
        # Side
        # =====================================================

        side = str(
            side
        ).lower()

        if side not in self.VALID_SIDES:
            raise ValueError(
                "side must be 'left' or 'right', "
                f"got {side!r}"
            )

        self.side = side

        self.variant = (
            "RM75-6FB"
        )

        # 单位：
        #
        # length -> m
        # angle  -> rad

        # =====================================================
        # Standard DH:
        #
        #   a_i
        # =====================================================

        self.a = np.zeros(
            self.DOF,
            dtype=float,
        )

        # =====================================================
        # Standard DH:
        #
        #   alpha_i
        # =====================================================

        self.alpha = np.deg2rad(
            [
                -90.0,
                 90.0,
                -90.0,
                 90.0,
                -90.0,
                 90.0,
                  0.0,
            ]
        )

        # =====================================================
        # Standard DH:
        #
        #   d_i
        #
        # 来自实际控制器 get_DH_data。
        # =====================================================

        self.d = np.array(
            [
                0.240500,
                0.0,
                0.256000,
                0.0,
                0.210000,
                0.0,
                0.161199,
            ],
            dtype=float,
        )

        # =====================================================
        # Controller joint angle -> DH model angle
        #
        #   theta_i =
        #       q_i + theta_offset_i
        #
        # IMPORTANT:
        #
        # 这里的 q 仍然是控制器关节角。
        # 不要把 offset 预先加进 q。
        # =====================================================

        self.theta_offset = np.zeros(
            self.DOF,
            dtype=float,
        )

        if self.side == "left":

            self.theta_offset[6] = (
                np.pi
            )

        # =====================================================
        # Joint limits
        #
        # 注意：
        # 这些也是 controller-q 的范围，
        # 不是 theta = q + offset 的范围。
        # =====================================================

        self.q_min = np.deg2rad(
            [
                -178.0,
                -130.0,
                -178.0,
                -135.0,
                -178.0,
                -128.0,
                -360.0,
            ]
        )

        self.q_max = np.deg2rad(
            [
                178.0,
                130.0,
                178.0,
                135.0,
                178.0,
                128.0,
                360.0,
            ]
        )

        # =====================================================
        # Maximum joint speed
        # =====================================================

        self.qd_max = np.deg2rad(
            [
                180.0,
                180.0,
                225.0,
                225.0,
                225.0,
                225.0,
                225.0,
            ]
        )