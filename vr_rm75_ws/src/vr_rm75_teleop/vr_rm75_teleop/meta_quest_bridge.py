#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from scipy.spatial.transform import Rotation

from meta_quest_teleop.reader import MetaQuestReader


class MetaQuestBridge(Node):

    def __init__(self):
        super().__init__("meta_quest_bridge")

        self.reader = MetaQuestReader()

        self.frame_id = "meta_world_ros"

        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.left_pose_pub = self.create_publisher(
            PoseStamped,
            "/meta_quest/left_grip_pose",
            self.qos,
        )

        self.right_pose_pub = self.create_publisher(
            PoseStamped,
            "/meta_quest/right_grip_pose",
            self.qos,
        )

        self.left_valid_pub = self.create_publisher(
            Bool,
            "/meta_quest/left_tracking_valid",
            self.qos,
        )

        self.right_valid_pub = self.create_publisher(
            Bool,
            "/meta_quest/right_tracking_valid",
            self.qos,
        )

        self.last_valid = {
            "left": None,
            "right": None,
        }

        self.timer = self.create_timer(
            1.0 / 50.0,
            self.timer_callback,
        )

        self.get_logger().info(
            "Meta Quest world-space bridge started"
        )

    def matrix_to_pose(
        self,
        T: np.ndarray,
    ) -> PoseStamped | None:

        if T is None:
            return None

        if T.shape != (4, 4):
            return None

        if not np.all(np.isfinite(T)):
            return None

        R = T[:3, :3]

        if abs(np.linalg.det(R) - 1.0) > 0.1:
            return None

        quat = Rotation.from_matrix(
            R
        ).as_quat()

        msg = PoseStamped()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.header.frame_id = self.frame_id

        msg.pose.position.x = float(T[0, 3])
        msg.pose.position.y = float(T[1, 3])
        msg.pose.position.z = float(T[2, 3])

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        return msg

    def publish_hand(
        self,
        hand: str,
        pose_pub,
        valid_pub,
    ):
        valid = self.reader.get_tracking_valid(
            hand
        )

        valid_msg = Bool()
        valid_msg.data = valid
        valid_pub.publish(valid_msg)

        if self.last_valid[hand] != valid:
            self.last_valid[hand] = valid

            if valid:
                self.get_logger().info(
                    f"{hand} tracking VALID"
                )
            else:
                self.get_logger().warning(
                    f"{hand} tracking INVALID"
                )

        # Important:
        # never publish a pose while tracking is invalid.
        if not valid:
            return

        T = (
            self.reader.get_hand_controller_transform_ros(
                hand,
                "grip",
            )
        )

        msg = self.matrix_to_pose(T)

        if msg is not None:
            pose_pub.publish(msg)

    def timer_callback(self):

        self.publish_hand(
            "left",
            self.left_pose_pub,
            self.left_valid_pub,
        )

        self.publish_hand(
            "right",
            self.right_pose_pub,
            self.right_valid_pub,
        )


def main(args=None):

    rclpy.init(args=args)

    node = MetaQuestBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.reader.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()