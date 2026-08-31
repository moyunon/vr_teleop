#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

from scipy.spatial.transform import Rotation


class HandState:

    def __init__(self):

        # VR 当前位姿
        self.current_position = None
        self.current_rotation = None

        # 当前 Grip 模拟量
        self.grip = 0.0

        # clutch 是否已经激活
        self.active = False

        # 第一次收到 VR Pose 后建立虚拟目标
        self.initialized = False

        # 虚拟“机械臂末端目标”
        self.target_position = None
        self.target_rotation = None

        # 每一次 Grip 按下瞬间保存的锚点
        self.vr_anchor_position = None
        self.vr_anchor_rotation = None

        self.target_anchor_position = None
        self.target_anchor_rotation = None


class QuestClutchTarget(Node):

    def __init__(self):

        super().__init__("quest_clutch_target")

        self.left = HandState()
        self.right = HandState()

        # -----------------------------
        # clutch 阈值
        # -----------------------------
        self.grip_on_threshold = 0.65
        self.grip_off_threshold = 0.35

        # VR 位移 -> 虚拟末端位移比例
        # 目前先 1:1
        self.position_scale = 1.0

        # -----------------------------
        # VR Pose subscriptions
        # -----------------------------
        self.create_subscription(
            PoseStamped,
            "/meta_quest/left_grip_pose",
            lambda msg: self.pose_callback(msg, self.left),
            10,
        )

        self.create_subscription(
            PoseStamped,
            "/meta_quest/right_grip_pose",
            lambda msg: self.pose_callback(msg, self.right),
            10,
        )

        # -----------------------------
        # Grip subscriptions
        # -----------------------------
        self.create_subscription(
            Float32,
            "/meta_quest/left_grip",
            lambda msg: self.grip_callback(
                msg,
                self.left,
                "LEFT",
            ),
            10,
        )

        self.create_subscription(
            Float32,
            "/meta_quest/right_grip",
            lambda msg: self.grip_callback(
                msg,
                self.right,
                "RIGHT",
            ),
            10,
        )

        # -----------------------------
        # Target Pose publishers
        # -----------------------------
        self.left_target_pub = self.create_publisher(
            PoseStamped,
            "/teleop/left_target_pose",
            10,
        )

        self.right_target_pub = self.create_publisher(
            PoseStamped,
            "/teleop/right_target_pose",
            10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        # 50 Hz 更新虚拟目标
        self.timer = self.create_timer(
            0.02,
            self.update,
        )

        self.get_logger().info(
            "Quest clutch target node started"
        )

        self.get_logger().info(
            "Grip ON >= 0.65, OFF <= 0.35"
        )

    # ============================================================
    # VR Pose
    # ============================================================

    def pose_callback(self, msg, hand):

        position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=float,
        )

        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )

        if np.linalg.norm(quaternion) < 1e-6:
            return

        rotation = Rotation.from_quat(quaternion)

        hand.current_position = position
        hand.current_rotation = rotation

        # 第一次获得 VR 位姿：
        # 让虚拟目标先与手柄当前位置重合
        if not hand.initialized:

            hand.target_position = position.copy()
            hand.target_rotation = rotation

            hand.initialized = True

    # ============================================================
    # Grip
    # ============================================================

    def grip_callback(self, msg, hand, name):

        hand.grip = float(msg.data)

        if not hand.initialized:
            return

        # -----------------------------
        # Rising edge:
        # clutch 从 OFF -> ON
        # -----------------------------
        if (
            not hand.active
            and hand.grip >= self.grip_on_threshold
        ):

            hand.active = True

            # 记录此时 VR 的位姿
            hand.vr_anchor_position = (
                hand.current_position.copy()
            )

            hand.vr_anchor_rotation = (
                hand.current_rotation
            )

            # 同时记录当前虚拟目标
            hand.target_anchor_position = (
                hand.target_position.copy()
            )

            hand.target_anchor_rotation = (
                hand.target_rotation
            )

            self.get_logger().info(
                f"{name} clutch ON"
            )

        # -----------------------------
        # Falling edge:
        # clutch 从 ON -> OFF
        # -----------------------------
        elif (
            hand.active
            and hand.grip <= self.grip_off_threshold
        ):

            hand.active = False

            self.get_logger().info(
                f"{name} clutch OFF"
            )

    # ============================================================
    # Clutch mapping
    # ============================================================

    def update_hand_target(self, hand):

        if not hand.initialized:
            return

        if not hand.active:
            # clutch 未按下：
            # target 保持原位置
            return

        if hand.current_position is None:
            return

        # --------------------------------------------------------
        # 1. 平移增量
        # --------------------------------------------------------

        delta_position = (
            hand.current_position
            - hand.vr_anchor_position
        )

        hand.target_position = (
            hand.target_anchor_position
            + self.position_scale * delta_position
        )

        # --------------------------------------------------------
        # 2. 旋转增量
        # --------------------------------------------------------

        delta_rotation = (
            hand.current_rotation
            * hand.vr_anchor_rotation.inv()
        )

        hand.target_rotation = (
            delta_rotation
            * hand.target_anchor_rotation
        )

    # ============================================================
    # Publish target
    # ============================================================

    def publish_target(
        self,
        hand,
        publisher,
        child_frame,
    ):

        if not hand.initialized:
            return

        now = self.get_clock().now().to_msg()

        quaternion = (
            hand.target_rotation.as_quat()
        )

        # -----------------------------
        # PoseStamped
        # -----------------------------

        msg = PoseStamped()

        msg.header.stamp = now
        msg.header.frame_id = "map"

        msg.pose.position.x = float(
            hand.target_position[0]
        )
        msg.pose.position.y = float(
            hand.target_position[1]
        )
        msg.pose.position.z = float(
            hand.target_position[2]
        )

        msg.pose.orientation.x = float(
            quaternion[0]
        )
        msg.pose.orientation.y = float(
            quaternion[1]
        )
        msg.pose.orientation.z = float(
            quaternion[2]
        )
        msg.pose.orientation.w = float(
            quaternion[3]
        )

        publisher.publish(msg)

        # -----------------------------
        # TF
        # -----------------------------

        tf_msg = TransformStamped()

        tf_msg.header.stamp = now
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id = child_frame

        tf_msg.transform.translation.x = float(
            hand.target_position[0]
        )
        tf_msg.transform.translation.y = float(
            hand.target_position[1]
        )
        tf_msg.transform.translation.z = float(
            hand.target_position[2]
        )

        tf_msg.transform.rotation.x = float(
            quaternion[0]
        )
        tf_msg.transform.rotation.y = float(
            quaternion[1]
        )
        tf_msg.transform.rotation.z = float(
            quaternion[2]
        )
        tf_msg.transform.rotation.w = float(
            quaternion[3]
        )

        self.tf_broadcaster.sendTransform(tf_msg)

    # ============================================================
    # Main update
    # ============================================================

    def update(self):

        self.update_hand_target(self.left)
        self.update_hand_target(self.right)

        self.publish_target(
            self.left,
            self.left_target_pub,
            "left_virtual_target",
        )

        self.publish_target(
            self.right,
            self.right_target_pub,
            "right_virtual_target",
        )


def main(args=None):

    rclpy.init(args=args)

    node = QuestClutchTarget()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()