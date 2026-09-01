#!/usr/bin/env python3

import time

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import (
    PoseStamped,
    TransformStamped,
)

from sensor_msgs.msg import JointState

from tf2_ros import TransformBroadcaster

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import (
    forward_kinematics,
)

from vr_rm75_teleop.vr_pose_mapping import (
    position_quaternion_to_transform,
    map_vr_pose_to_robot_target,
)

from vr_rm75_teleop.se3_rate_limiter import (
    limit_pose_step,
)

from vr_rm75_teleop.rm75_ik import (
    solve_ik,
)

from vr_rm75_teleop.target_feasibility import (
    project_target_to_feasible,
    minimum_singular_value,
)

from std_msgs.msg import Bool


class ArmFusionState:
    """
    保存单条机械臂在 VR 遥操过程中自己的状态。

    LEFT / RIGHT 使用完全相同的算法，
    只有 model、VR anchor、q_safe、T_safe 等状态独立。
    """

    def __init__(
        self,
        side,
        q_start,
    ):

        self.side = side

        if side == "left":
            self.prefix = "l"
        elif side == "right":
            self.prefix = "r"
        else:
            raise ValueError(
                "side must be 'left' or 'right'"
            )

        self.base_frame = (
            f"{self.prefix}_rm75_base_link"
        )

        # =====================================================
        # Robot model
        # =====================================================

        self.model = RM75Model(
            side=side,
        )

        # =====================================================
        # Robot configuration state
        # =====================================================

        self.q_start = np.asarray(
            q_start,
            dtype=float,
        ).copy()

        self.q_preferred = (
            self.q_start.copy()
        )

        self.q_safe = (
            self.q_start.copy()
        )

        self.T_ee_anchor = (
            forward_kinematics(
                self.q_start,
                model=self.model,
            )
        )

        self.T_safe = (
            self.T_ee_anchor.copy()
        )

        # =====================================================
        # VR state
        # =====================================================

        self.T_vr_latest = None
        self.T_vr_anchor = None

        self.anchored = False

        # Quest tracking state
        self.tracking_valid = False

        # tracking 恢复后必须重新建立 VR/robot anchor
        self.need_reanchor = True

        # 用于检测 VR Pose 数据流是否中断
        self.last_vr_rx_time = None
        self.pose_stale = False

        # =====================================================
        # Last diagnostics
        # =====================================================

        self.last_solve_ms = 0.0
        self.last_result = None
        self.last_limit_result = None


class QuestDualIKFusion(Node):

    def __init__(self):

        super().__init__(
            "quest_dual_ik_fusion"
        )

        # =====================================================
        # 1. Teleoperation parameters
        # =====================================================

        # =====================================================
        # Simulation mode
        #
        # True:
        #   用于当前纯 RViz / IK 验证
        #   VR Pose 1:1 映射
        #   不做 Cartesian rate limit
        #   不做 feasibility / singularity projection
        #
        # False:
        #   以后真实机械臂使用安全链
        # =====================================================

        self.unrestricted_simulation = False


        if self.unrestricted_simulation:

            self.position_scale = 1.0
            self.orientation_scale = 1.0

        else:

            self.position_scale = 1.0
            self.orientation_scale = 1.0

        # 50 Hz:
        #
        # 5 mm/frame
        # ~= 0.25 m/s
        self.max_translation_step = (
            0.005
        )

        # 2 deg/frame
        # ~= 100 deg/s
        self.max_rotation_step = (
            np.deg2rad(
                2.0
            )
        )

        # Quest pose timeout protection.
        # If no fresh VR pose arrives within 200 ms,
        # stop updating this arm and require re-anchor.
        self.vr_pose_timeout_s = 0.20

        # =====================================================
        # 2. Feasibility parameters
        # =====================================================

        self.sigma_stop = 0.010

        self.binary_iterations = 6

        # =====================================================
        # RealBot teleoperation ready pose
        #
        # 直接来自两台真实 RM75 控制器
        # get_current_arm_state 的关节角。
        #
        # 单位：
        # controller -> deg -> rad
        #
        # 注意：
        # LEFT J7 不要额外 +/- 180 deg。
        # RM75Model(side="left") 已经在内部处理
        # theta_offset[6] = pi。
        # =====================================================

        q_start_left = np.deg2rad(
            [
                -64.143,
                -33.259,
                -0.044,
                -80.671,
                8.438,
                -47.101,
                111.349,
            ]
        )

        q_start_right = np.deg2rad(
            [
                21.180,
                48.282,
                32.467,
                74.971,
                21.508,
                54.389,
            -158.273,
            ]
        )

        self.arms = {
            "left":
                ArmFusionState(
                    side="left",
                    q_start=q_start_left,
                ),

            "right":
                ArmFusionState(
                    side="right",
                    q_start=q_start_right,
                ),
        }

        # =====================================================
        # 4. Quest subscribers
        # =====================================================

        self.left_pose_sub = (
            self.create_subscription(
                PoseStamped,
                "/meta_quest/left_grip_pose",
                self.left_pose_callback,
                1,
            )
        )

        self.right_pose_sub = (
            self.create_subscription(
                PoseStamped,
                "/meta_quest/right_grip_pose",
                self.right_pose_callback,
                1,
            )
        )

        self.left_tracking_sub = (
            self.create_subscription(
                Bool,
                "/meta_quest/left_tracking_valid",
                self.left_tracking_callback,
                1,
            )
        )

        self.right_tracking_sub = (
            self.create_subscription(
                Bool,
                "/meta_quest/right_tracking_valid",
                self.right_tracking_callback,
                1,
            )
        )

        def left_tracking_callback(
            self,
            msg,
        ):
            self.update_tracking_state(
                side="left",
                valid=msg.data,
            )


        def right_tracking_callback(
            self,
            msg,
        ):
            self.update_tracking_state(
                side="right",
                valid=msg.data,
            )


        def update_tracking_state(
            self,
            side,
            valid,
        ):

            state = self.arms[side]

            valid = bool(valid)

            # 状态没变化就不用重复处理
            if valid == state.tracking_valid:
                return

            state.tracking_valid = valid

            if not valid:

                # 立即停止使用旧 VR pose
                state.T_vr_latest = None

                # 旧 anchor 作废
                state.anchored = False
                state.need_reanchor = True

                state.last_vr_rx_time = None

                self.get_logger().warning(
                    f"{side.upper()} Quest tracking LOST."
                )

            else:

                # 此时不立即 anchor：
                # 等下一帧有效 Pose 到达再建立。
                state.need_reanchor = True

                self.get_logger().info(
                    f"{side.upper()} Quest tracking RECOVERED. "
                    "Waiting for fresh pose to re-anchor."
                )

        # =====================================================
        # 5. Pose publishers
        # =====================================================

        self.pose_publishers = {}

        for side in (
            "left",
            "right",
        ):

            self.pose_publishers[side] = {
                "raw":
                    self.create_publisher(
                        PoseStamped,
                        f"/vr_rm75/{side}/raw_target",
                        10,
                    ),

                "command":
                    self.create_publisher(
                        PoseStamped,
                        f"/vr_rm75/{side}/command_target",
                        10,
                    ),

                "safe":
                    self.create_publisher(
                        PoseStamped,
                        f"/vr_rm75/{side}/safe_target",
                        10,
                    ),

                "actual":
                    self.create_publisher(
                        PoseStamped,
                        f"/vr_rm75/{side}/actual_tcp",
                        10,
                    ),
            }

        # =====================================================
        # 6. Combined dual-arm JointState
        #
        # 这一条消息同时携带 14 个 RM75 joint。
        #
        # 暂时仍然不用标准 /joint_states，
        # 避免与其他节点冲突。
        # =====================================================

        self.joint_pub = (
            self.create_publisher(
                JointState,
                "/vr_rm75/dual_joint_states",
                10,
            )
        )

        # =====================================================
        # 7. TF broadcaster
        # =====================================================

        self.tf_broadcaster = (
            TransformBroadcaster(
                self
            )
        )

        # =====================================================
        # 8. 50 Hz control loop
        # =====================================================

        self.timer = (
            self.create_timer(
                0.02,
                self.control_update,
            )
        )

        self.frame_counter = 0

        self.get_logger().info(
            "Quest dual-arm IK fusion started."
        )

        self.get_logger().info(
            "Simulation / visualization only. "
            "NO command is sent to the real robot."
        )

        self.get_logger().info(
            "Waiting for LEFT and RIGHT Quest poses..."
        )

    # =========================================================
    # Quest callbacks
    # =========================================================

    def left_tracking_callback(
        self,
        msg,
    ):
        self.update_tracking_state(
            side="left",
            valid=msg.data,
        )

    def right_tracking_callback(
        self,
        msg,
    ):
        self.update_tracking_state(
            side="right",
            valid=msg.data,
        )

    def update_tracking_state(
        self,
        side,
        valid,
    ):

        state = self.arms[
            side
        ]

        valid = bool(
            valid
        )

        # 状态没有发生变化，不重复处理。
        if valid == state.tracking_valid:
            return

        state.tracking_valid = valid

        if not valid:

            # 当前 VR pose 不再可信。
            state.T_vr_latest = None

            # 原来的 anchor 作废。
            state.anchored = False
            state.need_reanchor = True

            state.last_vr_rx_time = None

            self.get_logger().warning(
                f"{side.upper()} Quest tracking LOST."
            )

        else:

            # tracking 恢复以后，
            # 等下一帧有效 Pose 到达再重新 anchor。
            state.need_reanchor = True

            self.get_logger().info(
                f"{side.upper()} Quest tracking RECOVERED. "
                "Waiting for fresh pose to re-anchor."
            )

    def left_pose_callback(
        self,
        msg,
    ):

        self.update_vr_pose(
            side="left",
            msg=msg,
        )

    def right_pose_callback(
        self,
        msg,
    ):

        self.update_vr_pose(
            side="right",
            msg=msg,
        )

    def update_vr_pose(
        self,
        side,
        msg,
    ):

        state = self.arms[
            side
        ]

        # tracking 无效时，Pose 一律不接受
        if not state.tracking_valid:
            return

        p = msg.pose.position
        q = msg.pose.orientation

        try:

            T_vr = (
                position_quaternion_to_transform(
                    position=[
                        p.x,
                        p.y,
                        p.z,
                    ],

                    quaternion_xyzw=[
                        q.x,
                        q.y,
                        q.z,
                        q.w,
                    ],
                )
            )

        except ValueError as exc:

            self.get_logger().warning(
                f"{side.upper()} invalid Quest pose: "
                f"{exc}"
            )

            return

        state.T_vr_latest = T_vr

        state.last_vr_rx_time = (
            time.perf_counter()
        )

        state.pose_stale = False


        # 首次启动或 tracking 恢复后：
        # 同时重新建立 VR anchor 和 robot anchor。
        if (
            state.need_reanchor
            or
            not state.anchored
        ):

            # 当前 Quest 手柄作为新的 VR 零点
            state.T_vr_anchor = (
                T_vr.copy()
            )

            # 当前机械臂安全位姿作为新的 robot 零点
            state.T_ee_anchor = (
                state.T_safe.copy()
            )

            state.anchored = True
            state.need_reanchor = False

            self.get_logger().info(
                f"{side.upper()} VR/EE anchor captured."
            )

    # =========================================================
    # 50 Hz main loop
    # =========================================================

    def control_update(
        self,
    ):

        cycle_start = (
            time.perf_counter()
        )

        now = (
            self.get_clock()
            .now()
            .to_msg()
        )

        # =====================================================
        # LEFT / RIGHT 使用同一个函数处理。
        # =====================================================

        for side in (
            "left",
            "right",
        ):

            state = self.arms[
                side
            ]

            self.update_arm(
                state=state,
                stamp=now,
            )

        # =====================================================
        # 每周期发布一次完整双臂 q。
        #
        # 即使某一只手还没建立 VR anchor，
        # 该臂也会保持 q_start。
        # =====================================================

        self.publish_dual_joint_state(
            now
        )

        cycle_ms = (
            time.perf_counter()
            - cycle_start
        ) * 1000.0

        # =====================================================
        # Diagnostics
        # =====================================================

        self.frame_counter += 1

        if (
            self.frame_counter
            % 25
            == 0
        ):

            self.print_diagnostics(
                cycle_ms
            )

    # =========================================================
    # Process one arm
    # =========================================================

    def update_arm(
        self,
        state,
        stamp,
    ):

        if not state.tracking_valid:
            return

        if (
            not state.anchored
            or
            state.T_vr_latest is None
        ):
            return


        # VR data stream timeout protection
        if (
            state.last_vr_rx_time is None
            or
            (
                time.perf_counter()
                - state.last_vr_rx_time
            )
            > self.vr_pose_timeout_s
        ):

            if not state.pose_stale:

                state.pose_stale = True

                # 数据恢复后重新 anchor，
                # 防止断流期间手柄运动造成跳变。
                state.anchored = False
                state.need_reanchor = True

                self.get_logger().warning(
                    f"{state.side.upper()} Quest pose STALE."
                )

            return

        # =====================================================
        # 1. VR incremental pose -> Cartesian raw target
        # =====================================================

        T_raw = (
            map_vr_pose_to_robot_target(
                T_vr_anchor=
                    state.T_vr_anchor,

                T_vr_current=
                    state.T_vr_latest,

                T_ee_anchor=
                    state.T_ee_anchor,

                side=
                    state.side,

                position_scale=
                    self.position_scale,

                orientation_scale=
                    self.orientation_scale,
            )
        )

        # =====================================================
        # 2. Cartesian command
        # =====================================================

        if self.unrestricted_simulation:

            # -------------------------------------------------
            # Simulation:
            #
            # Quest raw target 直接送给 IK。
            #
            # 不做：
            # - translation rate limit
            # - rotation rate limit
            # -------------------------------------------------

            T_command = (
                T_raw.copy()
            )

            # 这里只构造 diagnostics，
            # 不参与任何限制。

            delta_p = (
                T_raw[:3, 3]
                -
                state.T_safe[:3, 3]
            )

            translation_distance = (
                np.linalg.norm(
                    delta_p
                )
            )

            R_relative = (
                T_raw[:3, :3]
                @ state.T_safe[:3, :3].T
            )

            rotation_distance = (
                np.linalg.norm(
                    Rotation
                    .from_matrix(
                        R_relative
                    )
                    .as_rotvec()
                )
            )

            limit_result = {
                "T_limited":
                    T_command,

                "translation_distance":
                    float(
                        translation_distance
                    ),

                "translation_step":
                    float(
                        translation_distance
                    ),

                "rotation_distance":
                    float(
                        rotation_distance
                    ),

                "rotation_step":
                    float(
                        rotation_distance
                    ),

                "translation_limited":
                    False,

                "rotation_limited":
                    False,
            }

        else:

            # -------------------------------------------------
            # Safe mode:
            # -------------------------------------------------

            limit_result = (
                limit_pose_step(
                    T_current=
                        state.T_safe,

                    T_desired=
                        T_raw,

                    max_translation_step=
                        self.max_translation_step,

                    max_rotation_step=
                        self.max_rotation_step,
                )
            )

            T_command = (
                limit_result[
                    "T_limited"
                ]
            )

        # =====================================================
        # 3. IK configuration for this arm
        # =====================================================

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
                state.q_preferred,

            "preferred_posture_gain":
                1.0,

            "max_null_step":
                np.deg2rad(
                    0.10
                ),
        }

        # =====================================================
        # 4. IK
        # =====================================================

        solve_start = (
            time.perf_counter()
        )

        if self.unrestricted_simulation:

            # -------------------------------------------------
            # UNRESTRICTED SIMULATION
            #
            # T_raw / T_command
            #       ↓
            # direct IK
            #
            # 完全绕过：
            #
            # sigma_stop
            # target feasibility
            # binary projection
            # -------------------------------------------------

            ik_result = (
                solve_ik(
                    T_target=
                        T_command,

                    q_seed=
                        state.q_safe,

                    model=
                        state.model,

                    **ik_kwargs,
                )
            )

            if ik_result[
                "success"
            ]:

                q_new = (
                    ik_result[
                        "q"
                    ].copy()
                )

                sigma_min = (
                    minimum_singular_value(
                        q_new,
                        state.model,
                    )
                )

                result = {
                    "success":
                        True,

                    "projected":
                        False,

                    "alpha":
                        1.0,

                    "T_safe":
                        T_command.copy(),

                    "q_safe":
                        q_new,

                    "sigma_min":
                        sigma_min,

                    "raw_ik_success":
                        True,
                }

            else:

                # IK 连数学解都没有得到，
                # 这时没有任何 q 可以发布。
                #
                # 保持上一帧仅仅是防止程序崩溃，
                # 不属于 feasibility 安全限制。

                result = {
                    "success":
                        False,

                    "projected":
                        False,

                    "alpha":
                        1.0,

                    "T_safe":
                        state.T_safe.copy(),

                    "q_safe":
                        state.q_safe.copy(),

                    "sigma_min":
                        minimum_singular_value(
                            state.q_safe,
                            state.model,
                        ),

                    "raw_ik_success":
                        False,
                }

        else:

            # -------------------------------------------------
            # SAFE MODE
            # -------------------------------------------------

            result = (
                project_target_to_feasible(
                    T_safe=
                        state.T_safe,

                    T_raw=
                        T_command,

                    q_safe=
                        state.q_safe,

                    model=
                        state.model,

                    sigma_stop=
                        self.sigma_stop,

                    binary_iterations=
                        self.binary_iterations,

                    ik_kwargs=
                        ik_kwargs,
                )
            )


        solve_ms = (
            time.perf_counter()
            - solve_start
        ) * 1000.0

        if not result[
            "success"
        ]:

            state.last_solve_ms = (
                solve_ms
            )

            state.last_result = (
                result
            )

            state.last_limit_result = (
                limit_result
            )

            return

        # =====================================================
        # 5. Advance safe state
        #
        # T_safe 与 q_safe 必须一起更新。
        # =====================================================

        state.T_safe = (
            result[
                "T_safe"
            ].copy()
        )

        state.q_safe = (
            result[
                "q_safe"
            ].copy()
        )

        # =====================================================
        # 6. Independent FK verification
        # =====================================================

        T_actual = (
            forward_kinematics(
                state.q_safe,
                model=state.model,
            )
        )

        # =====================================================
        # 7. Publish Pose topics
        # =====================================================

        pubs = (
            self.pose_publishers[
                state.side
            ]
        )

        self.publish_pose(
            T=T_raw,
            publisher=pubs["raw"],
            base_frame=state.base_frame,
            stamp=stamp,
        )

        self.publish_pose(
            T=T_command,
            publisher=pubs["command"],
            base_frame=state.base_frame,
            stamp=stamp,
        )

        self.publish_pose(
            T=state.T_safe,
            publisher=pubs["safe"],
            base_frame=state.base_frame,
            stamp=stamp,
        )

        self.publish_pose(
            T=T_actual,
            publisher=pubs["actual"],
            base_frame=state.base_frame,
            stamp=stamp,
        )

        # =====================================================
        # 8. Publish TF
        # =====================================================

        self.publish_tf(
            T=T_raw,
            parent_frame=state.base_frame,
            child_frame=(
                f"{state.side}_raw_target"
            ),
            stamp=stamp,
        )

        self.publish_tf(
            T=T_command,
            parent_frame=state.base_frame,
            child_frame=(
                f"{state.side}_command_target"
            ),
            stamp=stamp,
        )

        self.publish_tf(
            T=state.T_safe,
            parent_frame=state.base_frame,
            child_frame=(
                f"{state.side}_safe_target"
            ),
            stamp=stamp,
        )

        self.publish_tf(
            T=T_actual,
            parent_frame=state.base_frame,
            child_frame=(
                f"{state.side}_actual_tcp"
            ),
            stamp=stamp,
        )

        # =====================================================
        # 9. Save diagnostics
        # =====================================================

        state.last_solve_ms = (
            solve_ms
        )

        state.last_result = (
            result
        )

        state.last_limit_result = (
            limit_result
        )

    # =========================================================
    # Combined 14-joint publisher
    # =========================================================

    def publish_dual_joint_state(
        self,
        stamp,
    ):

        left = self.arms[
            "left"
        ]

        right = self.arms[
            "right"
        ]

        msg = JointState()

        msg.header.stamp = (
            stamp
        )

        msg.name = (
            [
                f"l_rm75_joint_{i}"
                for i in range(
                    1,
                    8,
                )
            ]
            +
            [
                f"r_rm75_joint_{i}"
                for i in range(
                    1,
                    8,
                )
            ]
        )

        msg.position = (
            left.q_safe.tolist()
            +
            right.q_safe.tolist()
        )

        self.joint_pub.publish(
            msg
        )

    # =========================================================
    # Pose publisher
    # =========================================================

    def publish_pose(
        self,
        T,
        publisher,
        base_frame,
        stamp,
    ):

        quat = (
            Rotation
            .from_matrix(
                T[:3, :3]
            )
            .as_quat()
        )

        msg = PoseStamped()

        msg.header.stamp = (
            stamp
        )

        msg.header.frame_id = (
            base_frame
        )

        msg.pose.position.x = float(
            T[0, 3]
        )

        msg.pose.position.y = float(
            T[1, 3]
        )

        msg.pose.position.z = float(
            T[2, 3]
        )

        msg.pose.orientation.x = float(
            quat[0]
        )

        msg.pose.orientation.y = float(
            quat[1]
        )

        msg.pose.orientation.z = float(
            quat[2]
        )

        msg.pose.orientation.w = float(
            quat[3]
        )

        publisher.publish(
            msg
        )

    # =========================================================
    # TF publisher
    # =========================================================

    def publish_tf(
        self,
        T,
        parent_frame,
        child_frame,
        stamp,
    ):

        quat = (
            Rotation
            .from_matrix(
                T[:3, :3]
            )
            .as_quat()
        )

        msg = TransformStamped()

        msg.header.stamp = (
            stamp
        )

        msg.header.frame_id = (
            parent_frame
        )

        msg.child_frame_id = (
            child_frame
        )

        msg.transform.translation.x = float(
            T[0, 3]
        )

        msg.transform.translation.y = float(
            T[1, 3]
        )

        msg.transform.translation.z = float(
            T[2, 3]
        )

        msg.transform.rotation.x = float(
            quat[0]
        )

        msg.transform.rotation.y = float(
            quat[1]
        )

        msg.transform.rotation.z = float(
            quat[2]
        )

        msg.transform.rotation.w = float(
            quat[3]
        )

        self.tf_broadcaster.sendTransform(
            msg
        )

    # =========================================================
    # Diagnostics
    # =========================================================

    def print_diagnostics(
        self,
        cycle_ms,
    ):

        pieces = []

        for side in (
            "left",
            "right",
        ):

            state = self.arms[
                side
            ]

            label = (
                "L"
                if side == "left"
                else "R"
            )

            if not state.anchored:

                pieces.append(
                    f"{label}:WAIT"
                )

                continue

            if (
                state.last_result is None
                or
                state.last_limit_result is None
            ):

                pieces.append(
                    f"{label}:INIT"
                )

                continue

            result = (
                state.last_result
            )

            limit_result = (
                state.last_limit_result
            )

            pieces.append(
                (
                    "{}:"
                    "d={:.1f}->{:.1f}mm "
                    "r={:.1f}->{:.1f}deg "
                    "lim=({},{}) "
                    "ikok={} "
                    "proj={} "
                    "a={:.3f} "
                    "s={:.4f} "
                    "ik={:.1f}ms"
                ).format(

                    label,

                    limit_result[
                        "translation_distance"
                    ] * 1000.0,

                    limit_result[
                        "translation_step"
                    ] * 1000.0,

                    np.rad2deg(
                        limit_result[
                            "rotation_distance"
                        ]
                    ),

                    np.rad2deg(
                        limit_result[
                            "rotation_step"
                        ]
                    ),

                    int(
                        limit_result[
                            "translation_limited"
                        ]
                    ),

                    int(
                        limit_result[
                            "rotation_limited"
                        ]
                    ),

                    int(
                        result[
                            "raw_ik_success"
                        ]
                    ),

                    int(
                        result[
                            "projected"
                        ]
                    ),

                    result[
                        "alpha"
                    ],

                    result[
                        "sigma_min"
                    ],

                    state.last_solve_ms,
                )
            )

        self.get_logger().info(
            " | ".join(
                pieces
            )
            +
            " | cycle={:.1f}ms".format(
                cycle_ms
            )
        )


def main(
    args=None,
):

    rclpy.init(
        args=args
    )

    node = (
        QuestDualIKFusion()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    node.destroy_node()

    if rclpy.ok():

        rclpy.shutdown()


if __name__ == "__main__":

    main()
