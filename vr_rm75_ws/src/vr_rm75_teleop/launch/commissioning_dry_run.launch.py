"""Unified fail-closed dual-RM75 commissioning launch."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


RECORDER_TOPICS = [
    "/meta_quest/left_grip_pose",
    "/meta_quest/right_grip_pose",
    "/meta_quest/left_grip",
    "/meta_quest/right_grip",
    "/meta_quest/input_fresh",
    "/vr_rm75/deadman_active",
    "/rm75/left/actual_joint_states",
    "/rm75/right/actual_joint_states",
    "/rm75/actual_joint_states",
    "/vr_rm75/dual_joint_states",
    "/vr_rm75/measured_qdot_qddot",
    "/vr_rm75/following_error_diagnostics",
    "/vr_rm75/control_diagnostics",
    "/vr_rm75/collision/min_distances_m",
    "/vr_rm75/collision/backend_diagnostics",
    "/vr_rm75/collision/safety_diagnostics",
    "/vr_rm75/safety_state",
    "/vr_rm75/stop_event",
    "/vr_rm75/stop_acknowledged",
    "/vr_rm75/robot_command_status",
    "/vr_rm75/timing_diagnostics",
]


def _fusion_node(context, *, fusion_config):
    """Create fusion node, loading an optional explicit threshold profile."""
    collision_profile = LaunchConfiguration(
        "collision_threshold_profile"
    ).perform(context).strip()
    parameters = [fusion_config]
    if collision_profile:
        parameters.append(collision_profile)
    parameters.append(
        {
            "left_command_ip": LaunchConfiguration("left_ip"),
            "right_command_ip": LaunchConfiguration("right_ip"),
            "robot_command_port": LaunchConfiguration("tcp_port"),
            "enable_robot_motion": LaunchConfiguration(
                "enable_robot_motion"
            ),
        }
    )
    return [
        Node(
            package="vr_rm75_teleop",
            executable="quest_dual_ik_fusion",
            name="quest_dual_ik_fusion",
            output="screen",
            parameters=parameters,
        )
    ]


def generate_launch_description():
    """Start the read-only feedback, VR, collision, and dry-run chain."""
    share = get_package_share_directory("vr_rm75_teleop")
    state_config = os.path.join(share, "config", "rm75_state.yaml")
    bridge_config = os.path.join(share, "config", "meta_quest_bridge.yaml")
    fusion_config = os.path.join(share, "config", "safe_first_motion.yaml")
    geometry_default = os.path.join(share, "config", "collision_geometry.yaml")

    arguments = [
        DeclareLaunchArgument("left_ip", default_value="192.168.127.18"),
        DeclareLaunchArgument("right_ip", default_value="192.168.127.19"),
        DeclareLaunchArgument("tcp_port", default_value="8080"),
        DeclareLaunchArgument("udp_enabled", default_value="false"),
        DeclareLaunchArgument("left_udp_port", default_value="8089"),
        DeclareLaunchArgument("right_udp_port", default_value="8090"),
        DeclareLaunchArgument(
            "collision_config", default_value=geometry_default
        ),
        # Empty by default: demo thresholds are never selected implicitly.
        DeclareLaunchArgument(
            "collision_threshold_profile", default_value=""
        ),
        # Deliberately default-off. Only an on-site operator may override it.
        DeclareLaunchArgument("enable_robot_motion", default_value="false"),
        DeclareLaunchArgument("enable_bag_recording", default_value="false"),
        DeclareLaunchArgument(
            "bag_output", default_value="vr_rm75_commissioning_bag"
        ),
    ]

    nodes = [
        Node(
            package="vr_rm75_teleop",
            executable="meta_quest_bridge",
            name="meta_quest_bridge",
            output="screen",
            parameters=[bridge_config],
        ),
        Node(
            package="vr_rm75_teleop",
            executable="rm75_state_node",
            name="rm75_state_node",
            output="screen",
            parameters=[
                state_config,
                {
                    "left_ip": LaunchConfiguration("left_ip"),
                    "right_ip": LaunchConfiguration("right_ip"),
                    "tcp_port": LaunchConfiguration("tcp_port"),
                    "udp_enabled": LaunchConfiguration("udp_enabled"),
                    "left_udp_port": LaunchConfiguration("left_udp_port"),
                    "right_udp_port": LaunchConfiguration("right_udp_port"),
                },
            ],
        ),
        Node(
            package="vr_rm75_teleop",
            executable="collision_backend",
            name="collision_backend",
            output="screen",
            parameters=[
                {"geometry_config": LaunchConfiguration("collision_config")}
            ],
        ),
        OpaqueFunction(
            function=_fusion_node,
            kwargs={"fusion_config": fusion_config},
        ),
        ExecuteProcess(
            cmd=[
                "ros2",
                "bag",
                "record",
                "-o",
                LaunchConfiguration("bag_output"),
                *RECORDER_TOPICS,
            ],
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_bag_recording")),
        ),
    ]
    return LaunchDescription(arguments + nodes)
