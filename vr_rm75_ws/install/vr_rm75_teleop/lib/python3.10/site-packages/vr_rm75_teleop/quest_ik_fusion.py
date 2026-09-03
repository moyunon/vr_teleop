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

from vr_rm75_teleop.target_feasibility import (
    project_target_to_feasible,
)

from vr_rm75_teleop.vr_pose_mapping import (
    position_quaternion_to_transform,
    map_vr_pose_to_robot_target,
    get_vr_to_arm_rotation,
)

from vr_rm75_teleop.se3_rate_limiter import (
    limit_pose_step,
)


class QuestRightIKFusion(Node):

    def __init__(self):

        super().__init__(
            "quest_right_ik_fusion"
        )

        # =====================================================
        # 1. Robot model
        # =====================================================

        self.side = "right"

        self.model = RM75Model(
            side=self.side,
        )

        # =====================================================
        # 2. Initial robot configuration
        #
        # 这是我们已经在 continuous IK 中验证过的
        # 安全初始构型。
        # =====================================================

        self.q_start = np.deg2rad(
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

        self.q_preferred = (
            self.q_start.copy()
        )

        self.q_safe = (
            self.q_start.copy()
        )

        # 起始机械臂末端 Pose
        self.T_ee_anchor = (
            forward_kinematics(
                self.q_start,
                model=self.model,
            )
        )

        # 当前最后一个安全 Cartesian target
        self.T_safe = (
            self.T_ee_anchor.copy()
        )

        # =====================================================
        # 3. VR state
        # =====================================================

        self.T_vr_latest = None

        self.T_vr_anchor = None

        self.anchored = False

        # 第一版先降低位置比例。
        #
        # 手移动 10 cm
        # → robot target 移动 5 cm
        #
        # 等第一轮验证通过再调到 1.0。
        self.position_scale = 0.5
        self.orientation_scale = 0.5

        # =====================================================
        # Cartesian command rate limit
        #
        # 50 Hz 下：
        #
        # 5 mm / frame
        # -> 理论最大 0.25 m/s
        #
        # 2 deg / frame
        # -> 理论最大 100 deg/s
        # =====================================================

        self.max_translation_step = (
            0.005
        )

        self.max_rotation_step = (
            np.deg2rad(
                2.0
            )
        )

        # =====================================================
        # 4. Feasibility parameters
        # =====================================================

        self.sigma_stop = 0.010

        self.binary_iterations = 6

        self.ik_kwargs = {
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
                self.q_preferred,

            "preferred_posture_gain":
                1.0,

            "max_null_step":
                np.deg2rad(
                    0.10
                ),
        }

        # =====================================================
        # 5. ROS subscriber
        #
        # depth = 1:
        # 我们只关心最新 Quest Pose，
        # 不希望 IK 忙的时候积压旧帧。
        # =====================================================

        self.pose_sub = (
            self.create_subscription(
                PoseStamped,
                "/meta_quest/right_grip_pose",
                self.right_pose_callback,
                1,
            )
        )

        # =====================================================
        # 6. Diagnostic publishers
        # =====================================================

        self.raw_pose_pub = (
            self.create_publisher(
                PoseStamped,
                "/vr_rm75/right/raw_target",
                10,
            )
        )

        self.safe_pose_pub = (
            self.create_publisher(
                PoseStamped,
                "/vr_rm75/right/safe_target",
                10,
            )
        )

        self.actual_pose_pub = (
            self.create_publisher(
                PoseStamped,
                "/vr_rm75/right/actual_tcp",
                10,
            )
        )

        # 暂时不用标准 /joint_states，
        # 避免以后和整机 JointState 冲突。
        self.joint_pub = (
            self.create_publisher(
                JointState,
                "/vr_rm75/right/joint_states",
                10,
            )
        )

        self.tf_broadcaster = (
            TransformBroadcaster(
                self
            )
        )

        self.command_pose_pub = (
            self.create_publisher(
                PoseStamped,
                "/vr_rm75/right/command_target",
                10,
            )
        )

        # =====================================================
        # 7. 50 Hz control timer
        # =====================================================

        self.timer = self.create_timer(
            0.02,
            self.control_update,
        )

        self.frame_counter = 0

        # RM75 local base frame
        self.base_frame = (
            "r_rm75_base_link"
        )

        self.get_logger().info(
            "Quest -> RIGHT RM75 IK fusion started"
        )

        self.get_logger().info(
            "Simulation / visualization only. "
            "No command is sent to the real robot."
        )

        self.get_logger().info(
            "Waiting for "
            "/meta_quest/right_grip_pose ..."
        )

        self.get_logger().info(
            "Initial q [deg] = "
            + np.array2string(
                np.rad2deg(
                    self.q_start
                ),
                precision=2,
            )
        )

    # =========================================================
    # Quest callback
    #
    # 只负责：
    #
    # PoseStamped
    #   ↓
    # 4x4 matrix
    #
    # 不在 callback 中做 IK。
    # =========================================================

    def right_pose_callback(
        self,
        msg,
    ):

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
                f"Invalid Quest pose: {exc}"
            )

            return

        self.T_vr_latest = (
            T_vr
        )

        # ---------------------------------------------
        # 第一帧自动建立 VR anchor
        # ---------------------------------------------

        if not self.anchored:

            self.T_vr_anchor = (
                T_vr.copy()
            )

            self.anchored = True

            self.get_logger().info(
                "VR anchor captured."
            )

    # =========================================================
    # Main control loop
    # =========================================================

    def control_update(
        self,
    ):

        if (
            not self.anchored
            or
            self.T_vr_latest is None
        ):
            return

        # =====================================================
        # 1. VR incremental mapping
        # =====================================================

        T_raw = (
            map_vr_pose_to_robot_target(
                T_vr_anchor=
                    self.T_vr_anchor,

                T_vr_current=
                    self.T_vr_latest,

                T_ee_anchor=
                    self.T_ee_anchor,

                side=
                    self.side,

                position_scale=
                    self.position_scale,

                orientation_scale=
                    self.orientation_scale,
            )
        )

        # =====================================================
        # 2. SE(3) rate limiter
        #
        # 注意：
        #
        # 从上一帧真正安全的 T_safe
        # 向当前 VR T_raw 前进一步。
        # =====================================================

        limit_result = (
            limit_pose_step(
                T_current=
                    self.T_safe,

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
        # Rate limiter diagnostics
        # =====================================================

        raw_distance_mm = (
            limit_result[
                "translation_distance"
            ]
            * 1000.0
        )

        command_step_mm = (
            limit_result[
                "translation_step"
            ]
            * 1000.0
        )

        raw_angle_deg = (
            np.rad2deg(
                limit_result[
                    "rotation_distance"
                ]
            )
        )

        command_angle_deg = (
            np.rad2deg(
                limit_result[
                    "rotation_step"
                ]
            )
        )

        # =====================================================
        # Live mapping diagnostics
        # =====================================================

        delta_p_vr = (
            self.T_vr_latest[:3, 3]
            -
            self.T_vr_anchor[:3, 3]
        )

        C = get_vr_to_arm_rotation(
            self.side
        )

        delta_p_arm = (
            self.position_scale
            * C
            @ delta_p_vr
        )

        # =====================================================
        # VR orientation diagnostics
        # =====================================================

        R_vr_anchor = (
            self.T_vr_anchor[:3, :3]
        )

        R_vr_current = (
            self.T_vr_latest[:3, :3]
        )

        R_delta_vr = (
            R_vr_current
            @ R_vr_anchor.T
        )

        rotvec_vr = (
            Rotation
            .from_matrix(
                R_delta_vr
            )
            .as_rotvec()
        )

        rotvec_arm = (
            C
            @ rotvec_vr
        )

        rotvec_vr_deg = (
            np.rad2deg(
                rotvec_vr
            )
        )

        rotvec_arm_deg = (
            np.rad2deg(
                rotvec_arm
            )
        )

        # =====================================================
        # 2. Target feasibility + IK
        #
        # 注意：
        # project_target_to_feasible 内部已经求 IK，
        # 返回 q_safe。
        #
        # 后面不要再调用一次 solve_ik。
        # =====================================================

        start_time = (
            time.perf_counter()
        )

        result = (
            project_target_to_feasible(
                T_safe=
                    self.T_safe,

                T_raw=
                    T_command,

                q_safe=
                    self.q_safe,

                model=
                    self.model,

                sigma_stop=
                    self.sigma_stop,

                binary_iterations=
                    self.binary_iterations,

                ik_kwargs=
                    self.ik_kwargs,
            )
        )

        solve_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        if not result["success"]:

            self.get_logger().warning(
                "Target feasibility failed."
            )

            return

        # =====================================================
        # 3. Advance safe state
        #
        # T_safe 和 q_safe 必须同时更新。
        # =====================================================

        self.T_safe = (
            result["T_safe"].copy()
        )

        self.q_safe = (
            result["q_safe"].copy()
        )

        # =====================================================
        # 4. Independent FK
        #
        # 用于验证：
        #
        # FK(q_safe) ≈ T_safe
        # =====================================================

        T_actual = (
            forward_kinematics(
                self.q_safe,
                model=self.model,
            )
        )

        # =====================================================
        # 5. Publish diagnostics
        # =====================================================

        now = (
            self.get_clock()
            .now()
            .to_msg()
        )

        self.publish_pose(
            T_raw,
            self.raw_pose_pub,
            now,
        )

        self.publish_pose(
            T_command,
            self.command_pose_pub,
            now,
        )

        self.publish_pose(
            self.T_safe,
            self.safe_pose_pub,
            now,
        )

        self.publish_pose(
            T_actual,
            self.actual_pose_pub,
            now,
        )

        self.publish_joint_state(
            now
        )

        self.publish_tf(
            T_raw,
            "right_raw_target",
            now,
        )

        self.publish_tf(
            T_command,
            "right_command_target",
            now,
        )

        self.publish_tf(
            self.T_safe,
            "right_safe_target",
            now,
        )

        self.publish_tf(
            T_actual,
            "right_actual_tcp",
            now,
        )

        # =====================================================
        # 6. Console diagnostics
        #
        # 每 25 帧输出一次，大约 0.5 秒。
        # =====================================================

        self.frame_counter += 1

        if (
            self.frame_counter
            % 25
            == 0
        ):

            self.get_logger().info(
                "VR rot=[{:+.1f},{:+.1f},{:+.1f}]deg  "
                "ARM rot=[{:+.1f},{:+.1f},{:+.1f}]deg  "
                "limited=({},{})  "
                "projected={}  "
                "sigma={:.5f}".format(

                    rotvec_vr_deg[0],
                    rotvec_vr_deg[1],
                    rotvec_vr_deg[2],

                    rotvec_arm_deg[0],
                    rotvec_arm_deg[1],
                    rotvec_arm_deg[2],

                    limit_result[
                        "translation_limited"
                    ],

                    limit_result[
                        "rotation_limited"
                    ],

                    result[
                        "projected"
                    ],

                    result[
                        "sigma_min"
                    ],
                )
            )

    # =========================================================
    # PoseStamped publisher
    # =========================================================

    def publish_pose(
        self,
        T,
        publisher,
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

        msg.header.stamp = stamp
        msg.header.frame_id = (
            self.base_frame
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
    # JointState publisher
    # =========================================================

    def publish_joint_state(
        self,
        stamp,
    ):

        msg = JointState()

        msg.header.stamp = stamp

        msg.name = [
            f"r_rm75_joint_{i}"
            for i in range(
                1,
                8,
            )
        ]

        msg.position = (
            self.q_safe.tolist()
        )

        self.joint_pub.publish(
            msg
        )

    # =========================================================
    # TF publisher
    # =========================================================

    def publish_tf(
        self,
        T,
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

        tf_msg = TransformStamped()

        tf_msg.header.stamp = stamp

        tf_msg.header.frame_id = (
            self.base_frame
        )

        tf_msg.child_frame_id = (
            child_frame
        )

        tf_msg.transform.translation.x = float(
            T[0, 3]
        )

        tf_msg.transform.translation.y = float(
            T[1, 3]
        )

        tf_msg.transform.translation.z = float(
            T[2, 3]
        )

        tf_msg.transform.rotation.x = float(
            quat[0]
        )

        tf_msg.transform.rotation.y = float(
            quat[1]
        )

        tf_msg.transform.rotation.z = float(
            quat[2]
        )

        tf_msg.transform.rotation.w = float(
            quat[3]
        )

        self.tf_broadcaster.sendTransform(
            tf_msg
        )


def main(
    args=None,
):

    rclpy.init(
        args=args
    )

    node = (
        QuestRightIKFusion()
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