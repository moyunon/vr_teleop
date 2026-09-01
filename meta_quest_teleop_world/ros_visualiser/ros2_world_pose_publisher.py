#!/usr/bin/env python3

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool

from meta_quest_teleop.reader import MetaQuestReader


class MetaQuestWorldPosePublisher(Node):
    """Publish Meta Quest world-space controller poses to ROS2."""

    def __init__(self) -> None:
        super().__init__("meta_quest_world_pose_publisher")

        self.reader = MetaQuestReader(
            ip_address=None,
            port=5555,
        )

        # Do NOT call this frame "map".
        # It is the Meta Quest tracking world converted to ROS axis convention.
        self.frame_id = "meta_world_ros"

        self.transform_suffix = {
            "grip": "g",
            "model": "m",
            "pointer": "p",
        }

        self.pose_publishers = {}

        for hand in ("left", "right"):
            for transform_type in ("grip", "pointer", "model"):
                topic = f"/meta_quest/{hand}_{transform_type}_pose"

                self.pose_publishers[(hand, transform_type)] = (
                    self.create_publisher(
                        PoseStamped,
                        topic,
                        10,
                    )
                )

        self.tracking_publishers = {
            "left": self.create_publisher(
                Bool,
                "/meta_quest/left_tracking_valid",
                10,
            ),
            "right": self.create_publisher(
                Bool,
                "/meta_quest/right_tracking_valid",
                10,
            ),
        }

        # OpenXR -> ROS axis conversion.
        #
        # OpenXR:
        #   +X right
        #   +Y up
        #   +Z backward
        #
        # ROS:
        #   +X forward
        #   +Y left
        #   +Z up
        #
        # Quaternion [x, y, z, w]
        q = Rotation.from_quat(
            [0.5, -0.5, -0.5, 0.5]
        )

        self.T_ros_from_openxr = np.eye(4)
        self.T_ros_from_openxr[:3, :3] = q.as_matrix()

        self.last_tracking_state = {
            "left": None,
            "right": None,
        }

        self.timer = self.create_timer(
            1.0 / 50.0,
            self.publish_data,
        )

        self.get_logger().info(
            "Meta Quest world-space ROS2 bridge started"
        )
        self.get_logger().info(
            f"Frame: {self.frame_id}"
        )

    def openxr_to_ros(
        self,
        transform_openxr: np.ndarray,
    ) -> np.ndarray:
        return self.T_ros_from_openxr @ transform_openxr

    def matrix_to_pose(
        self,
        transform: np.ndarray,
    ) -> PoseStamped | None:

        if transform.shape != (4, 4):
            return None

        if not np.all(np.isfinite(transform)):
            return None

        rotation_matrix = transform[:3, :3]

        det = np.linalg.det(rotation_matrix)

        if abs(abs(det) - 1.0) > 0.1:
            return None

        try:
            quaternion = Rotation.from_matrix(
                rotation_matrix
            ).as_quat()
        except ValueError:
            return None

        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.position.x = float(transform[0, 3])
        msg.pose.position.y = float(transform[1, 3])
        msg.pose.position.z = float(transform[2, 3])

        msg.pose.orientation.x = float(quaternion[0])
        msg.pose.orientation.y = float(quaternion[1])
        msg.pose.orientation.z = float(quaternion[2])
        msg.pose.orientation.w = float(quaternion[3])

        return msg

    def publish_data(self) -> None:

        transforms, buttons = (
            self.reader.get_transformations_and_buttons()
        )

        if transforms is None or buttons is None:
            return

        for hand, side in (
            ("left", "l"),
            ("right", "r"),
        ):
            tracking_key = (
                "leftTrackingHigh"
                if hand == "left"
                else "rightTrackingHigh"
            )

            tracking_valid = bool(
                buttons.get(tracking_key, False)
            )

            tracking_msg = Bool()
            tracking_msg.data = tracking_valid

            self.tracking_publishers[hand].publish(
                tracking_msg
            )

            # Only print when state changes.
            if (
                self.last_tracking_state[hand]
                != tracking_valid
            ):
                self.last_tracking_state[hand] = (
                    tracking_valid
                )

                if tracking_valid:
                    self.get_logger().info(
                        f"{hand} tracking VALID"
                    )
                else:
                    self.get_logger().warning(
                        f"{hand} tracking INVALID - "
                        "pose publishing stopped"
                    )

            # Critical safety gate:
            # never publish bad controller pose.
            if not tracking_valid:
                continue

            for transform_type, suffix in (
                self.transform_suffix.items()
            ):
                key = f"{side}{suffix}"

                transform_openxr = transforms.get(key)

                if transform_openxr is None:
                    continue

                transform_ros = self.openxr_to_ros(
                    transform_openxr
                )

                pose_msg = self.matrix_to_pose(
                    transform_ros
                )

                if pose_msg is None:
                    continue

                self.pose_publishers[
                    (hand, transform_type)
                ].publish(pose_msg)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = MetaQuestWorldPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()