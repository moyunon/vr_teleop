"""Offline ROS bridge tests for latest-state Quest input publication."""

import numpy as np
import pytest
import rclpy

import vr_rm75_teleop.meta_quest_bridge as bridge_module
from vr_rm75_teleop.meta_quest_bridge import MetaQuestBridge


class FakeReader:
    """In-memory reader replacement that never opens ADB or hardware."""

    def __init__(self):
        self.fresh = True
        self.transforms = {
            "left": self._pose(0.20),
            "right": self._pose(-0.30),
        }
        self.tracking = {"left": True, "right": True}
        self.grips = {"left": 0.25, "right": 0.75}

    @staticmethod
    def _pose(x):
        transform = np.eye(4)
        transform[0, 3] = x
        return transform

    def data_is_fresh(self, _timeout):
        return self.fresh

    def get_tracking_valid(self, hand):
        return self.tracking[hand]

    def get_hand_controller_transform_ros(self, hand, _pose_type):
        return self.transforms[hand].copy()

    def get_grip_value(self, hand):
        return self.grips[hand]

    def stop(self):
        pass


class RecordingPublisher:
    """Capture messages produced by a direct timer callback."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """Create the bridge with a fake reader and no running ROS timer."""
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setattr(bridge_module, "MetaQuestReader", FakeReader)
    if not rclpy.ok():
        rclpy.init()
    node = MetaQuestBridge()
    node.timer.cancel()
    node.left_pose_pub = RecordingPublisher()
    node.right_pose_pub = RecordingPublisher()
    node.left_valid_pub = RecordingPublisher()
    node.right_valid_pub = RecordingPublisher()
    node.left_grip_pub = RecordingPublisher()
    node.right_grip_pub = RecordingPublisher()
    node.input_fresh_pub = RecordingPublisher()
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_bridge_uses_keep_last_depth_one_and_current_reader_state(bridge):
    """Publish one coherent current snapshot with latest-state ROS QoS."""
    bridge.timer_callback()

    assert bridge.qos.depth == 1
    assert [message.data for message in bridge.input_fresh_pub.messages] == [True]
    assert [message.data for message in bridge.left_valid_pub.messages] == [True]
    assert [message.data for message in bridge.right_valid_pub.messages] == [True]
    assert [message.data for message in bridge.left_grip_pub.messages] == [0.25]
    assert [message.data for message in bridge.right_grip_pub.messages] == [0.75]
    assert bridge.left_pose_pub.messages[0].pose.position.x == 0.20
    assert bridge.right_pose_pub.messages[0].pose.position.x == -0.30


def test_stale_reader_state_never_publishes_pose_or_grip(bridge):
    """Continue fail-closed publication while the reader source is stale."""
    bridge.reader.fresh = False

    bridge.timer_callback()

    assert [message.data for message in bridge.input_fresh_pub.messages] == [False]
    assert [message.data for message in bridge.left_valid_pub.messages] == [False]
    assert [message.data for message in bridge.right_valid_pub.messages] == [False]
    assert bridge.left_pose_pub.messages == []
    assert bridge.right_pose_pub.messages == []
    assert bridge.left_grip_pub.messages == []
    assert bridge.right_grip_pub.messages == []
