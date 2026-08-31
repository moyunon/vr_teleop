import os

from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
)

from launch.conditions import (
    IfCondition,
)

from launch.substitutions import (
    LaunchConfiguration,
)

from launch_ros.actions import Node

from ament_index_python.packages import (
    get_package_share_directory,
)


def generate_launch_description():

    package_share = (
        get_package_share_directory(
            "lsrx_rm75_dual_description"
        )
    )

    urdf_path = os.path.join(
        package_share,
        "urdf",
        "LSRX_RM75_DUAL.urdf",
    )

    with open(
        urdf_path,
        "r",
        encoding="utf-8",
    ) as f:

        robot_description = (
            f.read()
        )

    use_rviz = LaunchConfiguration(
        "rviz"
    )

    # =========================================================
    # robot_state_publisher
    #
    # /joint_states
    #      ↓
    # full robot TF
    # =========================================================

    robot_state_publisher = Node(
        package=
            "robot_state_publisher",

        executable=
            "robot_state_publisher",

        name=
            "robot_state_publisher",

        output=
            "screen",

        parameters=[
            {
                "robot_description":
                    robot_description,
            }
        ],
    )

    # =========================================================
    # joint_state_publisher
    #
    # 它读取完整 URDF：
    #
    #   - 未受控关节 -> 默认位置
    #   - 左右 RM75 -> 从 dual_joint_states 更新
    #
    # 最终统一发布：
    #
    #   /joint_states
    # =========================================================

    joint_state_publisher = Node(
        package=
            "joint_state_publisher",

        executable=
            "joint_state_publisher",

        name=
            "joint_state_publisher",

        output=
            "screen",

        arguments=[
            urdf_path,
        ],

        parameters=[
            {
                "rate":
                    50,

                "source_list":
                    [
                        "/vr_rm75/dual_joint_states"
                    ],

                "publish_default_positions":
                    True,

                "use_mimic_tags":
                    True,
            }
        ],
    )

    # =========================================================
    # RViz
    # =========================================================

    rviz = Node(
        package=
            "rviz2",

        executable=
            "rviz2",

        name=
            "rviz2",

        output=
            "screen",

        condition=
            IfCondition(
                use_rviz
            ),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
            ),

            robot_state_publisher,
            joint_state_publisher,
            rviz,
        ]
    )