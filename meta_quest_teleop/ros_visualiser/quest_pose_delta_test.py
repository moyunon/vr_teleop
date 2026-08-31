#!/usr/bin/env python3

import threading
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from scipy.spatial.transform import Rotation


class QuestPoseDeltaTest(Node):

    def __init__(self):
        super().__init__("quest_pose_delta_test")

        self.lock = threading.Lock()

        self.latest = {
            "left": None,
            "right": None,
        }

        self.reference = {
            "left": None,
            "right": None,
        }

        self.create_subscription(
            PoseStamped,
            "/meta_quest/left_grip_pose",
            lambda msg: self.pose_callback(msg, "left"),
            10,
        )

        self.create_subscription(
            PoseStamped,
            "/meta_quest/right_grip_pose",
            lambda msg: self.pose_callback(msg, "right"),
            10,
        )

        thread = threading.Thread(
            target=self.keyboard_loop,
            daemon=True,
        )
        thread.start()

        print("")
        print("Quest pose delta test")
        print("---------------------")
        print("l : 将左手柄当前位置设为零点")
        print("r : 将右手柄当前位置设为零点")
        print("b : 同时将左右手柄设为零点")
        print("p : 打印当前位置相对于零点的位姿增量")
        print("Ctrl+C : 退出")
        print("")

    def pose_callback(self, msg, hand):

        p = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        q = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])

        if np.linalg.norm(q) < 1e-6:
            return

        with self.lock:
            self.latest[hand] = (p, q)

    def set_reference(self, hand):

        with self.lock:

            if self.latest[hand] is None:
                print(f"{hand}: 还没有收到 Pose 数据")
                return

            p, q = self.latest[hand]

            self.reference[hand] = (
                p.copy(),
                q.copy(),
            )

        print(f"{hand}: reference captured")

    def print_delta(self):

        with self.lock:

            for hand in ["left", "right"]:

                if self.latest[hand] is None:
                    continue

                if self.reference[hand] is None:
                    continue

                p, q = self.latest[hand]
                p0, q0 = self.reference[hand]

                # 平移增量，表达在 ROS map 坐标系中
                dp = p - p0

                R_current = Rotation.from_quat(q)
                R_reference = Rotation.from_quat(q0)

                # 世界/ROS坐标系下的相对旋转
                R_delta = R_current * R_reference.inv()

                rpy = R_delta.as_euler(
                    "xyz",
                    degrees=True,
                )

                print("")
                print(f"[{hand.upper()}]")
                print(
                    "Delta position [m]: "
                    f"x={dp[0]: .4f}, "
                    f"y={dp[1]: .4f}, "
                    f"z={dp[2]: .4f}"
                )

                print(
                    "Delta rotation [deg]: "
                    f"roll={rpy[0]: .2f}, "
                    f"pitch={rpy[1]: .2f}, "
                    f"yaw={rpy[2]: .2f}"
                )

    def keyboard_loop(self):

        while rclpy.ok():

            try:
                cmd = input("> ").strip().lower()
            except EOFError:
                return

            if cmd == "l":
                self.set_reference("left")

            elif cmd == "r":
                self.set_reference("right")

            elif cmd == "b":
                self.set_reference("left")
                self.set_reference("right")

            elif cmd == "p":
                self.print_delta()


def main(args=None):

    rclpy.init(args=args)

    node = QuestPoseDeltaTest()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
