#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from scipy.spatial.transform import Rotation

from meta_quest_teleop.reader import MetaQuestReader

from std_msgs.msg import Float32


class QuestRVizPublisher(Node):

    def __init__(self):
        super().__init__("quest_rviz_publisher")

        # USB 连接 Quest
        self.reader = MetaQuestReader(ip_address=None)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Pose topics
        self.left_pose_pub = self.create_publisher(
            PoseStamped,
            "/meta_quest/left_grip_pose",
            10,
        )

        self.right_pose_pub = self.create_publisher(
            PoseStamped,
            "/meta_quest/right_grip_pose",
            10,
        )

        # Grip 模拟量发布器，范围 0.0 ~ 1.0
        self.left_grip_pub = self.create_publisher(
            Float32,
            "/meta_quest/left_grip",
            10,
        )

        self.right_grip_pub = self.create_publisher(
            Float32,
            "/meta_quest/right_grip",
            10,
        )

        # 50 Hz
        self.timer = self.create_timer(0.02, self.update)

        self.get_logger().info("Quest RViz publisher started")
        self.get_logger().info("USB connection")
        self.get_logger().info("Fixed frame: map")
        self.get_logger().info("Left frame : left_hand_grip")
        self.get_logger().info("Right frame: right_hand_grip")

    def publish_hand(self, hand, child_frame, publisher):

        # 直接使用仓库提供的 OpenXR -> ROS 坐标转换
        transform = self.reader.get_hand_controller_transform_ros(hand)

        if transform is None:
            return

        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]

        try:
            quat = Rotation.from_matrix(rotation_matrix).as_quat()
        except ValueError:
            return

        now = self.get_clock().now().to_msg()

        # --------------------
        # 发布 TF
        # --------------------
        tf_msg = TransformStamped()

        tf_msg.header.stamp = now
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id = child_frame

        tf_msg.transform.translation.x = float(position[0])
        tf_msg.transform.translation.y = float(position[1])
        tf_msg.transform.translation.z = float(position[2])

        tf_msg.transform.rotation.x = float(quat[0])
        tf_msg.transform.rotation.y = float(quat[1])
        tf_msg.transform.rotation.z = float(quat[2])
        tf_msg.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(tf_msg)

        # --------------------
        # 发布 PoseStamped
        # --------------------
        pose_msg = PoseStamped()

        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])

        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])

        publisher.publish(pose_msg)

    def update(self):

        self.publish_hand(
            "left",
            "left_hand_grip",
            self.left_pose_pub,
        )

        self.publish_hand(
            "right",
            "right_hand_grip",
            self.right_pose_pub,
        )

        left_grip_msg = Float32()
        left_grip_msg.data = float(
            self.reader.get_grip_value("left")
        )
        self.left_grip_pub.publish(left_grip_msg)

        right_grip_msg = Float32()
        right_grip_msg.data = float(
            self.reader.get_grip_value("right")
        )
        self.right_grip_pub.publish(right_grip_msg)


def main(args=None):

    rclpy.init(args=args)

    node = QuestRVizPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
