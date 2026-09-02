"""Launch the read-only dual-RM75 actual-state publisher."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the state node with the packaged conservative configuration."""
    package_share = get_package_share_directory("vr_rm75_teleop")
    parameters = os.path.join(package_share, "config", "rm75_state.yaml")

    return LaunchDescription(
        [
            Node(
                package="vr_rm75_teleop",
                executable="rm75_state_node",
                name="rm75_state_node",
                output="screen",
                parameters=[parameters],
            )
        ]
    )
