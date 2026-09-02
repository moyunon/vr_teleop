#!/usr/bin/env python3

import json
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

from vr_rm75_teleop.rm75_fk import (
    forward_kinematics,
)
from vr_rm75_teleop.arm_fusion_state import (
    ArmFusionState,
)
from vr_rm75_teleop.safety_supervisor import (
    SafetyState,
    SafetySupervisor,
)
from vr_rm75_teleop.deadman_clutch import (
    DualGripDeadman,
)
from vr_rm75_teleop.collision_safety import (
    CollisionSafetyMonitor,
    disabled_collision_decision,
)
from vr_rm75_teleop.rm75_command_interface import (
    DualArmCommandDispatcher,
    RM75CommandConnectionError,
    RM75CommandRejectedError,
    RM75ControllerCommandError,
    RM75LowFollowCommandClient,
)
from vr_rm75_teleop.stop_policy import (
    StopClass,
    stop_for_transition,
)
from vr_rm75_teleop.following_error_monitor import FollowingErrorMonitor
from vr_rm75_teleop.timing_monitor import TimingMonitor

from vr_rm75_teleop.vr_pose_mapping import (
    position_quaternion_to_transform,
    map_vr_pose_to_robot_target,
)

from vr_rm75_teleop.se3_rate_limiter import (
    limit_pose_step,
)

from vr_rm75_teleop.joint_safety import (
    limit_joint_acceleration,
    limit_joint_soft_position,
    limit_joint_velocity,
    make_teleop_soft_limits,
)

from vr_rm75_teleop.rm75_ik import (
    solve_ik,
)

from vr_rm75_teleop.target_feasibility import (
    project_target_to_feasible,
    minimum_singular_value,
    singularity_region,
    singularity_speed_scale,
    validate_singularity_thresholds,
)

from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from std_srvs.srv import Trigger


class QuestDualIKFusion(Node):

    def __init__(self):

        super().__init__(
            "quest_dual_ik_fusion"
        )

        # =====================================================
        # 1. Teleoperation parameters
        # =====================================================

        # =====================================================
        # Simulation mode
        #
        # True:
        #   用于当前纯 RViz / IK 验证
        #   VR Pose 1:1 映射
        #   不做 Cartesian rate limit
        #   不做 feasibility / singularity projection
        #   最终 joint velocity 与 command sigma 检查仍然保留
        #
        # False:
        #   以后真实机械臂使用安全链
        # =====================================================

        self.unrestricted_simulation = False

        # Real-robot startup is the safe default.  Setting this false is an
        # explicit RViz-only compatibility mode and never enables actuation.
        self.declare_parameter(
            "require_robot_state",
            True,
        )
        self.declare_parameter(
            "robot_state_timeout_s",
            0.25,
        )
        self.declare_parameter(
            "command_timeout_s",
            0.10,
        )
        self.declare_parameter(
            "enable_robot_motion",
            False,
        )
        self.declare_parameter(
            "left_command_ip",
            "192.168.127.18",
        )
        self.declare_parameter(
            "right_command_ip",
            "192.168.127.19",
        )
        self.declare_parameter(
            "robot_command_port",
            8080,
        )
        self.declare_parameter(
            "robot_command_transport_timeout_s",
            0.01,
        )
        self.declare_parameter(
            "max_robot_command_delta_deg",
            0.5,
        )
        self.declare_parameter(
            "collision_protection_enabled",
            True,
        )
        self.declare_parameter(
            "collision_enabled_categories",
            ["left_self", "right_self", "inter_arm"],
        )
        self.declare_parameter(
            "collision_distance_timeout_s",
            0.10,
        )
        self.declare_parameter(
            "collision_stop_distance_m",
            0.05,
        )
        self.declare_parameter(
            "collision_warn_distance_m",
            0.15,
        )
        self.declare_parameter(
            "max_consecutive_ik_failures",
            3,
        )
        self.declare_parameter(
            "control_frequency_hz",
            50.0,
        )
        self.declare_parameter(
            "joint_velocity_scale",
            0.10,
        )
        self.declare_parameter(
            "joint_acceleration_limit_deg_s2",
            [90.0] * 7,
        )
        self.declare_parameter(
            "max_cartesian_translation_rate_m_s",
            0.25,
        )
        self.declare_parameter(
            "max_cartesian_rotation_rate_rad_s",
            float(np.deg2rad(100.0)),
        )
        self.declare_parameter(
            "sigma_stop",
            0.010,
        )
        self.declare_parameter(
            "sigma_warn",
            0.020,
        )
        self.declare_parameter(
            "joint_soft_limit_margin_deg",
            5.0,
        )
        self.declare_parameter(
            "elbow_singularity_margin_deg",
            15.0,
        )
        self.declare_parameter(
            "deadman_grip_on_threshold",
            0.65,
        )
        self.declare_parameter(
            "deadman_grip_off_threshold",
            0.35,
        )
        self.declare_parameter(
            "deadman_input_timeout_s",
            0.20,
        )
        self.declare_parameter(
            "left_rviz_fallback_q_deg",
            [
                -64.143,
                -33.259,
                -0.044,
                -80.671,
                8.438,
                -47.101,
                111.349,
            ],
        )
        self.declare_parameter(
            "right_rviz_fallback_q_deg",
            [
                21.180,
                48.282,
                32.467,
                74.971,
                21.508,
                54.389,
                -158.273,
            ],
        )
        # PROVISIONAL commissioning values. Hardware latency calibration is
        # required before treating these as validated thresholds.
        self.declare_parameter("following_warning_deg", [2.0] * 7)
        self.declare_parameter("following_stop_deg", [5.0] * 7)
        self.declare_parameter("following_persistence_s", 0.10)
        self.declare_parameter("following_hysteresis_ratio", 0.8)
        self.declare_parameter("following_max_timestamp_skew_s", 0.10)

        self.require_robot_state = bool(
            self.get_parameter(
                "require_robot_state"
            ).value
        )
        self.robot_state_timeout_s = float(
            self.get_parameter(
                "robot_state_timeout_s"
            ).value
        )
        self.command_timeout_s = float(
            self.get_parameter(
                "command_timeout_s"
            ).value
        )
        self.enable_robot_motion = bool(
            self.get_parameter(
                "enable_robot_motion"
            ).value
        )
        self.command_hosts = {
            "left": str(
                self.get_parameter(
                    "left_command_ip"
                ).value
            ),
            "right": str(
                self.get_parameter(
                    "right_command_ip"
                ).value
            ),
        }
        self.robot_command_port = int(
            self.get_parameter(
                "robot_command_port"
            ).value
        )
        self.robot_command_transport_timeout_s = float(
            self.get_parameter(
                "robot_command_transport_timeout_s"
            ).value
        )
        self.max_robot_command_delta_rad = np.deg2rad(
            float(
                self.get_parameter(
                    "max_robot_command_delta_deg"
                ).value
            )
        )
        self.collision_protection_enabled = bool(
            self.get_parameter(
                "collision_protection_enabled"
            ).value
        )
        self.collision_enabled_categories = tuple(
            str(value)
            for value in self.get_parameter(
                "collision_enabled_categories"
            ).value
        )
        self.collision_distance_timeout_s = float(
            self.get_parameter(
                "collision_distance_timeout_s"
            ).value
        )
        self.collision_stop_distance_m = float(
            self.get_parameter(
                "collision_stop_distance_m"
            ).value
        )
        self.collision_warn_distance_m = float(
            self.get_parameter(
                "collision_warn_distance_m"
            ).value
        )
        self.max_consecutive_ik_failures = int(
            self.get_parameter(
                "max_consecutive_ik_failures"
            ).value
        )
        self.control_frequency_hz = float(
            self.get_parameter(
                "control_frequency_hz"
            ).value
        )
        self.joint_velocity_scale = float(
            self.get_parameter(
                "joint_velocity_scale"
            ).value
        )
        self.joint_acceleration_limit = np.deg2rad(
            np.asarray(
                self.get_parameter(
                    "joint_acceleration_limit_deg_s2"
                ).value,
                dtype=float,
            )
        )
        self.max_cartesian_translation_rate = float(
            self.get_parameter(
                "max_cartesian_translation_rate_m_s"
            ).value
        )
        self.max_cartesian_rotation_rate = float(
            self.get_parameter(
                "max_cartesian_rotation_rate_rad_s"
            ).value
        )
        self.sigma_stop = float(
            self.get_parameter(
                "sigma_stop"
            ).value
        )
        self.sigma_warn = float(
            self.get_parameter(
                "sigma_warn"
            ).value
        )
        self.joint_soft_limit_margin = np.deg2rad(
            float(
                self.get_parameter(
                    "joint_soft_limit_margin_deg"
                ).value
            )
        )
        self.elbow_singularity_margin = np.deg2rad(
            float(
                self.get_parameter(
                    "elbow_singularity_margin_deg"
                ).value
            )
        )
        self.following_warning_rad = np.deg2rad(
            np.asarray(
                self.get_parameter("following_warning_deg").value,
                dtype=float,
            )
        )
        self.following_stop_rad = np.deg2rad(
            np.asarray(
                self.get_parameter("following_stop_deg").value,
                dtype=float,
            )
        )
        self.following_persistence_s = float(
            self.get_parameter("following_persistence_s").value
        )
        self.following_hysteresis_ratio = float(
            self.get_parameter("following_hysteresis_ratio").value
        )
        self.following_max_timestamp_skew_s = float(
            self.get_parameter("following_max_timestamp_skew_s").value
        )
        if (
            not np.isfinite(self.control_frequency_hz)
            or self.control_frequency_hz <= 0.0
        ):
            raise ValueError(
                "control_frequency_hz must be finite and positive"
            )
        if (
            not np.isfinite(self.joint_velocity_scale)
            or not 0.0 < self.joint_velocity_scale <= 1.0
        ):
            raise ValueError(
                "joint_velocity_scale must be in the interval (0, 1]"
            )
        if (
            self.joint_acceleration_limit.shape != (7,)
            or not np.all(np.isfinite(self.joint_acceleration_limit))
            or np.any(self.joint_acceleration_limit <= 0.0)
        ):
            raise ValueError(
                "joint_acceleration_limit_deg_s2 must contain 7 finite "
                "positive values"
            )
        if (
            not np.isfinite(self.max_cartesian_translation_rate)
            or self.max_cartesian_translation_rate <= 0.0
        ):
            raise ValueError(
                "max_cartesian_translation_rate_m_s must be finite "
                "and positive"
            )
        if (
            not np.isfinite(self.max_cartesian_rotation_rate)
            or self.max_cartesian_rotation_rate <= 0.0
        ):
            raise ValueError(
                "max_cartesian_rotation_rate_rad_s must be finite "
                "and positive"
            )
        self.sigma_stop, self.sigma_warn = (
            validate_singularity_thresholds(
                self.sigma_stop,
                self.sigma_warn,
            )
        )
        self.control_period_s = 1.0 / self.control_frequency_hz
        if self.enable_robot_motion:
            if not self.require_robot_state:
                raise ValueError(
                    "enable_robot_motion requires require_robot_state=true"
                )
            if not self.collision_protection_enabled:
                raise ValueError(
                    "enable_robot_motion requires collision protection"
                )
            if self.joint_velocity_scale > 0.10:
                raise ValueError(
                    "real motion requires joint_velocity_scale <= 0.10"
                )
            if self.control_period_s > 0.020 + 1e-12:
                raise ValueError(
                    "real motion requires control_frequency_hz >= 50"
                )
        self.last_control_cycle_time = None
        self.deadman_grip_on_threshold = float(
            self.get_parameter(
                "deadman_grip_on_threshold"
            ).value
        )
        self.deadman_grip_off_threshold = float(
            self.get_parameter(
                "deadman_grip_off_threshold"
            ).value
        )
        self.deadman_input_timeout_s = float(
            self.get_parameter(
                "deadman_input_timeout_s"
            ).value
        )
        self.robot_system_ready = (
            not self.require_robot_state
        )

        if self.unrestricted_simulation:

            self.position_scale = 1.0
            self.orientation_scale = 1.0

        else:

            self.position_scale = 1.0
            self.orientation_scale = 1.0

        # Quest pose timeout protection.
        # If no fresh VR pose arrives within 200 ms,
        # stop updating this arm and require re-anchor.
        self.vr_pose_timeout_s = 0.20

        # =====================================================
        # 2. Feasibility parameters
        # =====================================================

        self.binary_iterations = 6

        # =====================================================
        # RViz-only fallback poses
        #
        # require_robot_state=True（默认）时完全忽略这些值；
        # q_safe/T_safe 只能由实时 q_measured 初始化。
        # =====================================================

        q_fallback_left = np.deg2rad(
            self.get_parameter(
                "left_rviz_fallback_q_deg"
            ).value
        )

        q_fallback_right = np.deg2rad(
            self.get_parameter(
                "right_rviz_fallback_q_deg"
            ).value
        )

        if self.require_robot_state:
            q_fallback_left = None
            q_fallback_right = None

        self.arms = {
            "left":
                ArmFusionState(
                    side="left",
                    fallback_q=q_fallback_left,
                ),

            "right":
                ArmFusionState(
                    side="right",
                    fallback_q=q_fallback_right,
                ),
        }

        for state in self.arms.values():
            state.joint_velocity_limit = (
                self.joint_velocity_scale
                * state.model.qd_max
            )
            state.joint_acceleration_limit = (
                self.joint_acceleration_limit.copy()
            )
            elbow_branch = -1 if state.side == "left" else 1
            (
                q_soft_min,
                q_soft_max,
            ) = make_teleop_soft_limits(
                hard_min=state.model.q_min,
                hard_max=state.model.q_max,
                joint_margin=self.joint_soft_limit_margin,
                elbow_index=3,
                elbow_branch=elbow_branch,
                elbow_margin=self.elbow_singularity_margin,
            )
            state.configure_teleop_soft_limits(
                q_soft_min,
                q_soft_max,
                elbow_branch,
            )

        self.following_monitors = {
            side: FollowingErrorMonitor(
                self.following_warning_rad,
                self.following_stop_rad,
                persistence_s=self.following_persistence_s,
                hysteresis_ratio=self.following_hysteresis_ratio,
                max_age_s=self.robot_state_timeout_s,
                max_timestamp_skew_s=(
                    self.following_max_timestamp_skew_s
                ),
                # RM75 joints in this model are bounded, not continuous.
                continuous_joints=[False] * 7,
            )
            for side in ("left", "right")
        }
        self.last_following_decisions = {}
        self.timing_monitor = TimingMonitor(
            nominal_period_s=self.control_period_s,
            window_size=500,
        )

        joint_soft_limits = {
            side: (state.q_soft_min, state.q_soft_max)
            for side, state in self.arms.items()
        }

        command_clients = {
            side: RM75LowFollowCommandClient(
                side=side,
                host=self.command_hosts[side],
                port=self.robot_command_port,
                timeout_s=self.robot_command_transport_timeout_s,
                enable_robot_motion=self.enable_robot_motion,
            )
            for side in ("left", "right")
        }
        self.robot_command_dispatcher = DualArmCommandDispatcher(
            command_clients,
            {
                side: state.joint_velocity_limit
                for side, state in self.arms.items()
            },
            enable_robot_motion=self.enable_robot_motion,
            joint_acceleration_limits={
                side: state.joint_acceleration_limit
                for side, state in self.arms.items()
            },
            max_command_delta_rad=self.max_robot_command_delta_rad,
            command_timeout_s=self.command_timeout_s,
            nominal_period_s=self.control_period_s,
            monotonic=time.perf_counter,
        )
        self.robot_command_hold_required = False
        self.robot_command_transport_fault = False
        self.robot_command_hold_reason = "command output healthy"
        self.robot_command_gate_open_since = None

        if self.enable_robot_motion:
            try:
                self.robot_command_dispatcher.connect()
            except RM75CommandConnectionError as exc:
                self.robot_command_transport_fault = True
                self.get_logger().error(
                    f"RM75 command interface connection failed: {exc}"
                )

        self.collision_monitor = CollisionSafetyMonitor(
            d_stop_m=self.collision_stop_distance_m,
            d_warn_m=self.collision_warn_distance_m,
            timeout_s=self.collision_distance_timeout_s,
            enabled_sources=self.collision_enabled_categories,
        )
        self.collision_sources = self.collision_monitor.enabled_sources
        if self.collision_protection_enabled:
            self.last_collision_decision = self.collision_monitor.evaluate()
        else:
            self.last_collision_decision = disabled_collision_decision()

        self.safety_supervisor = SafetySupervisor(
            command_timeout_s=self.command_timeout_s,
            max_consecutive_ik_failures=(
                self.max_consecutive_ik_failures
            ),
            joint_velocity_scale=self.joint_velocity_scale,
            joint_acceleration_limits={
                side: state.joint_acceleration_limit
                for side, state in self.arms.items()
            },
            joint_soft_limits=joint_soft_limits,
            require_collision_safety=self.collision_protection_enabled,
            require_actuator_safety=self.enable_robot_motion,
            require_following_safety=self.require_robot_state,
        )

        self.deadman_clutch = DualGripDeadman(
            on_threshold=self.deadman_grip_on_threshold,
            off_threshold=self.deadman_grip_off_threshold,
            input_timeout_s=self.deadman_input_timeout_s,
        )
        self.deadman_active = False
        self.quest_input_fresh_reported = False
        self.last_quest_input_status_time = None
        self.refresh_actuator_safety(time.perf_counter())

        # =====================================================
        # 4. Quest subscribers
        # =====================================================

        self.left_pose_sub = (
            self.create_subscription(
                PoseStamped,
                "/meta_quest/left_grip_pose",
                self.left_pose_callback,
                1,
            )
        )

        self.right_pose_sub = (
            self.create_subscription(
                PoseStamped,
                "/meta_quest/right_grip_pose",
                self.right_pose_callback,
                1,
            )
        )

        self.left_tracking_sub = (
            self.create_subscription(
                Bool,
                "/meta_quest/left_tracking_valid",
                self.left_tracking_callback,
                1,
            )
        )

        self.right_tracking_sub = (
            self.create_subscription(
                Bool,
                "/meta_quest/right_tracking_valid",
                self.right_tracking_callback,
                1,
            )
        )

        self.left_grip_sub = self.create_subscription(
            Float32,
            "/meta_quest/left_grip",
            lambda msg: self.grip_callback(
                "left",
                msg,
            ),
            5,
        )

        self.right_grip_sub = self.create_subscription(
            Float32,
            "/meta_quest/right_grip",
            lambda msg: self.grip_callback(
                "right",
                msg,
            ),
            5,
        )

        self.quest_input_fresh_sub = self.create_subscription(
            Bool,
            "/meta_quest/input_fresh",
            self.quest_input_fresh_callback,
            5,
        )

        # =====================================================
        # 4b. Read-only actual robot state subscribers
        # =====================================================

        self.robot_state_subscriptions = []

        for side in (
            "left",
            "right",
        ):

            self.robot_state_subscriptions.append(
                self.create_subscription(
                    JointState,
                    f"/rm75/{side}/actual_joint_states",
                    lambda msg, arm_side=side:
                        self.robot_joint_state_callback(
                            arm_side,
                            msg,
                        ),
                    10,
                )
            )

            self.robot_state_subscriptions.append(
                self.create_subscription(
                    Bool,
                    f"/rm75/{side}/connected",
                    lambda msg, arm_side=side:
                        self.robot_connected_callback(
                            arm_side,
                            msg,
                        ),
                    10,
                )
            )

            self.robot_state_subscriptions.append(
                self.create_subscription(
                    Bool,
                    f"/rm75/{side}/state_stale",
                    lambda msg, arm_side=side:
                        self.robot_stale_callback(
                            arm_side,
                            msg,
                        ),
                    10,
                )
            )

            self.robot_state_subscriptions.append(
                self.create_subscription(
                    Bool,
                    f"/rm75/{side}/joints_enabled",
                    lambda msg, arm_side=side:
                        self.robot_enabled_callback(
                            arm_side,
                            msg,
                        ),
                    10,
                )
            )

            self.robot_state_subscriptions.append(
                self.create_subscription(
                    Bool,
                    f"/rm75/{side}/fault",
                    lambda msg, arm_side=side:
                        self.robot_fault_callback(
                            arm_side,
                            msg,
                        ),
                    10,
                )
            )

        # One enabled-category collision snapshot is atomic and ordered by
        # self.collision_sources. A partial array invalidates the snapshot.
        self.collision_distance_sub = self.create_subscription(
            Float64MultiArray,
            "/vr_rm75/collision/min_distances_m",
            self.collision_distance_callback,
            10,
        )
        self.collision_diagnostics_sub = self.create_subscription(
            String,
            "/vr_rm75/collision/backend_diagnostics",
            self.collision_diagnostics_callback,
            10,
        )
        self.collision_ready_sub = self.create_subscription(
            Bool,
            "/vr_rm75/collision/backend_ready",
            self.collision_ready_callback,
            10,
        )

        # =====================================================
        # 5. Pose publishers
        # =====================================================

        self.pose_publishers = {}

        for side in (
            "left",
            "right",
        ):

            self.pose_publishers[side] = {
                "raw":
                    self.create_publisher(
                        PoseStamped,
                        f"/vr_rm75/{side}/raw_target",
                        10,
                    ),

                "command":
                    self.create_publisher(
                        PoseStamped,
                        f"/vr_rm75/{side}/command_target",
                        10,
                    ),

                "safe":
                    self.create_publisher(
                        PoseStamped,
                        f"/vr_rm75/{side}/safe_target",
                        10,
                    ),

                "actual":
                    self.create_publisher(
                        PoseStamped,
                        f"/vr_rm75/{side}/actual_tcp",
                        10,
                    ),
            }

        # =====================================================
        # 6. Combined dual-arm JointState
        #
        # 这一条消息同时携带 14 个 RM75 joint。
        #
        # 暂时仍然不用标准 /joint_states，
        # 避免与其他节点冲突。
        # =====================================================

        self.joint_pub = (
            self.create_publisher(
                JointState,
                "/vr_rm75/dual_joint_states",
                10,
            )
        )

        self.safety_state_pub = self.create_publisher(
            String,
            "/vr_rm75/safety_state",
            10,
        )

        self.command_allowed_pub = self.create_publisher(
            Bool,
            "/vr_rm75/command_allowed",
            10,
        )

        self.deadman_state_pub = self.create_publisher(
            Bool,
            "/vr_rm75/deadman_active",
            10,
        )

        self.collision_state_pub = self.create_publisher(
            String,
            "/vr_rm75/collision/state",
            10,
        )

        self.collision_speed_scale_pub = self.create_publisher(
            Float32,
            "/vr_rm75/collision/speed_scale",
            10,
        )

        self.robot_command_sent_pub = self.create_publisher(
            Bool,
            "/vr_rm75/robot_command_sent",
            10,
        )

        self.robot_command_status_pub = self.create_publisher(
            String,
            "/vr_rm75/robot_command_status",
            10,
        )

        self.stop_event_pub = self.create_publisher(
            String,
            "/vr_rm75/stop_event",
            10,
        )

        self.stop_ack_pub = self.create_publisher(
            Bool,
            "/vr_rm75/stop_acknowledged",
            10,
        )

        self.measured_kinematics_pub = self.create_publisher(
            Float64MultiArray,
            "/vr_rm75/measured_qdot_qddot",
            10,
        )

        self.following_error_pub = self.create_publisher(
            String,
            "/vr_rm75/following_error_diagnostics",
            10,
        )

        self.timing_diagnostics_pub = self.create_publisher(
            String,
            "/vr_rm75/timing_diagnostics",
            10,
        )

        self.control_diagnostics_pub = self.create_publisher(
            String,
            "/vr_rm75/control_diagnostics",
            10,
        )

        self.fault_reset_service = self.create_service(
            Trigger,
            "/vr_rm75/reset_safety_fault",
            self.reset_safety_fault_callback,
        )

        # =====================================================
        # 7. TF broadcaster
        # =====================================================

        self.tf_broadcaster = (
            TransformBroadcaster(
                self
            )
        )

        # =====================================================
        # 8. 50 Hz control loop
        # =====================================================

        self.timer = (
            self.create_timer(
                self.control_period_s,
                self.control_update,
            )
        )

        self.frame_counter = 0

        self.get_logger().info(
            "Quest dual-arm IK fusion started."
        )

        if self.enable_robot_motion:
            self.get_logger().warning(
                "REAL ROBOT MOTION GATE ENABLED. Low-follow movej_canfd "
                "remains blocked until every Safety Supervisor guard is "
                "ENGAGED."
            )
        else:
            self.get_logger().info(
                "Dry-run mode: enable_robot_motion=false; command sockets "
                "are not opened and NO robot command can be sent."
            )

        self.get_logger().info(
            "Waiting for LEFT and RIGHT Quest poses..."
        )

        if self.require_robot_state:

            self.get_logger().info(
                "Waiting for fresh LEFT and RIGHT RM75 actual state "
                "before initializing IK."
            )

        else:

            self.get_logger().warning(
                "RViz-only fallback enabled: real robot state is not "
                "required. Static validation prevents enabling real "
                "motion in this mode."
            )

        if self.collision_protection_enabled:
            source_order = ", ".join(
                source.value for source in self.collision_sources
            )
            self.get_logger().info(
                "Collision protection enabled; waiting for atomic "
                f"distance snapshots ordered as [{source_order}]."
            )
        else:
            self.get_logger().warning(
                "Collision protection explicitly disabled. This is only "
                "acceptable for offline/RViz commissioning."
            )

    # =========================================================
    # Read-only robot state callbacks
    # =========================================================

    @staticmethod
    def expected_robot_joint_names(
        side,
    ):

        prefix = (
            "l"
            if side == "left"
            else "r"
        )

        return [
            f"{prefix}_rm75_joint_{index}"
            for index in range(
                1,
                8,
            )
        ]

    def robot_joint_state_callback(
        self,
        side,
        msg,
    ):

        state = self.arms[
            side
        ]

        expected_names = (
            self.expected_robot_joint_names(
                side
            )
        )

        try:

            if len(msg.name) != len(msg.position):
                raise ValueError(
                    "JointState name/position lengths differ"
                )

            if len(set(msg.name)) != len(msg.name):
                raise ValueError(
                    "JointState contains duplicate names"
                )

            position_by_name = dict(
                zip(
                    msg.name,
                    msg.position,
                )
            )

            missing = [
                name
                for name in expected_names
                if name not in position_by_name
            ]

            if missing:
                raise ValueError(
                    "JointState missing joints: "
                    + ", ".join(
                        missing
                    )
                )

            q_measured = [
                position_by_name[
                    name
                ]
                for name in expected_names
            ]

            qdot_measured = None
            if msg.velocity:
                if len(msg.velocity) != len(msg.name):
                    raise ValueError(
                        "JointState name/velocity lengths differ"
                    )
                velocity_by_name = dict(zip(msg.name, msg.velocity))
                qdot_measured = [
                    velocity_by_name[name] for name in expected_names
                ]

            state.update_measured_q(
                q_measured,
                received_monotonic=
                    time.perf_counter(),
                qdot_measured=qdot_measured,
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            state.reject_measured_q(
                exc
            )

            self.get_logger().error(
                f"{side.upper()} invalid measured JointState: "
                f"{exc}"
            )

            self.robot_system_ready = (
                self.refresh_robot_system_readiness()
            )

            return

        self.robot_system_ready = (
            self.refresh_robot_system_readiness()
        )
        self.publish_measured_kinematics()

    def robot_connected_callback(
        self,
        side,
        msg,
    ):

        state = self.arms[
            side
        ]

        connected = bool(
            msg.data
        )

        if connected == state.robot_connected:
            return

        state.robot_connected = connected

        if not connected:

            state.invalidate_anchor()

            self.get_logger().warning(
                f"{side.upper()} RM75 state connection LOST."
            )

        self.robot_system_ready = (
            self.refresh_robot_system_readiness()
        )

    def robot_stale_callback(
        self,
        side,
        msg,
    ):

        state = self.arms[
            side
        ]

        stale = bool(
            msg.data
        )

        if stale == state.robot_reported_stale:
            return

        state.robot_reported_stale = stale

        if stale:

            state.invalidate_anchor()

            self.get_logger().warning(
                f"{side.upper()} RM75 state STALE."
            )

        self.robot_system_ready = (
            self.refresh_robot_system_readiness()
        )

    def robot_enabled_callback(
        self,
        side,
        msg,
    ):

        state = self.arms[
            side
        ]
        state.robot_joints_enabled = bool(
            msg.data
        )
        state.robot_enable_known = True

    def robot_fault_callback(
        self,
        side,
        msg,
    ):

        state = self.arms[
            side
        ]
        state.robot_fault = bool(
            msg.data
        )
        state.robot_fault_known = True

    def collision_distance_callback(
        self,
        msg,
    ):
        """Accept one atomic enabled-category minimum-distance snapshot."""
        values = list(msg.data)
        if len(values) != len(self.collision_sources):
            reason = (
                "collision distance array must contain exactly "
                f"{len(self.collision_sources)} values, got {len(values)}"
            )
            self.collision_monitor.reject_snapshot(reason)
            self.get_logger().error(reason)
        else:
            distances_m = {
                source: value
                for source, value in zip(self.collision_sources, values)
            }
            try:
                self.collision_monitor.update_snapshot(
                    distances_m,
                    received_monotonic=time.perf_counter(),
                )
            except (TypeError, ValueError) as exc:
                self.get_logger().error(
                    f"invalid collision distance snapshot: {exc}"
                )

        self.update_safety_supervisor(
            time.perf_counter()
        )

    def collision_diagnostics_callback(self, msg):
        """Import measured collision solve time into unified timing stats."""
        try:
            payload = json.loads(msg.data)
            solve_ms = payload.get("solve_ms")
            if solve_ms is not None:
                self.timing_monitor.record(
                    "collision_solve", float(solve_ms) / 1000.0
                )
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def collision_ready_callback(self, msg):
        """Immediately invalidate an old snapshot when its backend fails."""
        if bool(msg.data):
            return
        self.collision_monitor.reject_snapshot(
            "collision backend reported not ready"
        )
        self.update_safety_supervisor(time.perf_counter())

    def publish_measured_kinematics(self):
        """Publish 14 qdot then 14 qddot values only when both are valid."""
        states = [self.arms[side] for side in ("left", "right")]
        if any(
            state.qdot_measured is None or state.qddot_measured is None
            for state in states
        ):
            return
        qdot = np.concatenate([state.qdot_measured for state in states])
        qddot = np.concatenate([state.qddot_measured for state in states])
        self.measured_kinematics_pub.publish(
            Float64MultiArray(data=qdot.tolist() + qddot.tolist())
        )

    def refresh_following_safety(self, now_monotonic):
        """Evaluate fresh command/measurement pairs while motion is engaged."""
        if not self.require_robot_state:
            self.safety_supervisor.update_following(
                ready=True,
                hold_required=False,
                reason="following-error guard disabled in RViz-only mode",
            )
            return
        if self.safety_supervisor.state != SafetyState.ENGAGED:
            for monitor in self.following_monitors.values():
                monitor.reset()
            self.safety_supervisor.update_following(
                ready=True,
                hold_required=False,
                reason="following-error monitor armed; no motion engaged",
            )
            return

        decisions = {}
        for side, state in self.arms.items():
            if state.last_safe_command_time is None:
                self.safety_supervisor.update_following(
                    ready=True,
                    hold_required=False,
                    reason="awaiting first post-engagement safe command",
                )
                return
            decisions[side] = self.following_monitors[side].evaluate(
                state.q_command,
                state.last_safe_command_time,
                state.q_measured,
                state.last_robot_state_rx_time,
                now_monotonic,
            )
        self.last_following_decisions = decisions
        ready = all(item.ready for item in decisions.values())
        hold = any(item.hold_required for item in decisions.values())
        limiting_side = max(
            decisions,
            key=lambda side: (
                -1.0
                if decisions[side].max_abs_error_rad is None
                else decisions[side].max_abs_error_rad
            ),
        )
        detail = decisions[limiting_side]
        self.safety_supervisor.update_following(
            ready=ready,
            hold_required=hold,
            reason=f"{limiting_side} {detail.reason}",
        )
        self.following_error_pub.publish(
            String(
                data=json.dumps(
                    {
                        side: {
                            "state": item.state.value,
                            "ready": item.ready,
                            "hold_required": item.hold_required,
                            "max_abs_error_rad": item.max_abs_error_rad,
                            "command_age_s": item.command_age_s,
                            "measurement_age_s": item.measurement_age_s,
                            "timestamp_skew_s": item.timestamp_skew_s,
                            "reason": item.reason,
                        }
                        for side, item in decisions.items()
                    },
                    sort_keys=True,
                )
            )
        )

    def refresh_collision_safety(
        self,
        now_monotonic,
    ):
        """Refresh the collision watchdog and global dual-arm safety guard."""
        if self.collision_protection_enabled:
            decision = self.collision_monitor.evaluate(now_monotonic)
        else:
            decision = disabled_collision_decision()

        self.last_collision_decision = decision
        self.safety_supervisor.update_collision(
            ready=decision.ready,
            hold_required=decision.hold_required,
            speed_scale=decision.speed_scale,
            reason=decision.reason,
        )
        self.collision_state_pub.publish(
            String(
                data=f"{decision.region.value}: {decision.reason}"
            )
        )
        self.collision_speed_scale_pub.publish(
            Float32(data=decision.speed_scale)
        )
        return decision

    def refresh_actuator_safety(
        self,
        now_monotonic,
    ):
        """Refresh command-channel state and the output-side watchdog."""
        if not self.enable_robot_motion:
            self.safety_supervisor.update_actuator(
                ready=True,
                hold_required=False,
                fault=False,
                reason="robot motion explicitly disabled; dry-run only",
            )
            return

        if (
            self.robot_command_hold_required
            and not self.deadman_active
            and not self.robot_command_transport_fault
        ):
            self.robot_command_hold_required = False
            self.robot_command_hold_reason = (
                "command rejection cleared after deadman release"
            )
            self.robot_command_dispatcher.disarm()

        if self.safety_supervisor.state == SafetyState.ENGAGED:
            watchdog_reference = (
                self.robot_command_dispatcher.last_send_monotonic
            )
            if watchdog_reference is None:
                watchdog_reference = self.robot_command_gate_open_since
            if (
                watchdog_reference is not None
                and now_monotonic - watchdog_reference
                > self.command_timeout_s
            ):
                self.robot_command_hold_required = True
                self.robot_command_hold_reason = (
                    "real-robot command output watchdog expired"
                )

        fault = bool(
            self.robot_command_transport_fault
            or self.robot_command_dispatcher.faulted
        )
        ready = bool(
            self.robot_command_dispatcher.connected
            and not fault
        )
        if fault:
            reason = self.robot_command_dispatcher.last_reason
        elif self.robot_command_hold_required:
            reason = self.robot_command_hold_reason
        elif ready:
            reason = "dual-arm low-follow command channels connected"
        else:
            reason = "dual-arm command channels are not connected"

        self.safety_supervisor.update_actuator(
            ready=ready,
            hold_required=self.robot_command_hold_required,
            fault=fault,
            reason=reason,
        )

    def publish_robot_command_status(
        self,
        sent,
        reason,
    ):
        """Publish whether this cycle crossed the actuator boundary."""
        self.robot_command_sent_pub.publish(
            Bool(data=bool(sent))
        )
        self.robot_command_status_pub.publish(
            String(data=str(reason))
        )

    def handle_stop_transition(self, decision, now_monotonic):
        """Issue at most one dual-arm software stop per ENGAGED exit edge."""
        request = stop_for_transition(decision, now_monotonic)
        if request is None:
            return None

        previous_result = self.robot_command_dispatcher.last_stop_result
        if (
            previous_result is not None
            and previous_result.request.stop_class == request.stop_class
            and now_monotonic
            - previous_result.request.requested_monotonic
            <= self.command_timeout_s
        ):
            result = previous_result
        else:
            result = self.robot_command_dispatcher.request_stop(
                request.stop_class,
                request.reason,
                request.requested_monotonic,
            )

        event = {
            "stop_class": request.stop_class.value,
            "reason": request.reason,
            "dry_run": result.dry_run,
            "all_acknowledged": result.all_acknowledged,
            "arms": {
                arm.side: {
                    "attempted": arm.attempted,
                    "acknowledged": arm.acknowledged,
                    "ack_latency_s": arm.ack_latency_s,
                    "error": arm.error,
                }
                for arm in result.arms
            },
        }
        self.stop_event_pub.publish(
            String(data=json.dumps(event, sort_keys=True))
        )
        self.stop_ack_pub.publish(
            Bool(data=result.all_acknowledged)
        )

        if result.dry_run:
            self.get_logger().warning(
                f"DRY-RUN intended {request.stop_class.value}: "
                f"{request.reason}; no command socket was opened"
            )
        elif result.all_acknowledged:
            self.get_logger().warning(
                f"{request.stop_class.value} acknowledged by both arms: "
                f"{request.reason}"
            )
        else:
            self.get_logger().error(
                f"{request.stop_class.value} was not acknowledged by both "
                "arms; use the physical emergency stop and inspect the "
                "controller/network"
            )
            self.robot_command_transport_fault = True
            self.robot_command_dispatcher.latch_transport_fault(
                "dual-arm software stop acknowledgement incomplete"
            )
        return result

    def dispatch_robot_commands(
        self,
        safety_decision,
        now_monotonic,
    ):
        """Dispatch one fully guarded dual-arm target, or remain dry-run."""
        if not self.enable_robot_motion:
            reason = "enable_robot_motion=false; dry-run only"
            self.publish_robot_command_status(False, reason)
            return None

        if not safety_decision.command_allowed:
            self.robot_command_dispatcher.disarm()
            reason = (
                "Safety Supervisor command gate closed: "
                f"{safety_decision.reason}"
            )
            self.publish_robot_command_status(False, reason)
            return None

        q_commands = {
            side: state.q_safe
            for side, state in self.arms.items()
        }
        q_measured = {
            side: state.q_measured
            for side, state in self.arms.items()
        }
        generated_times = [
            state.last_safe_command_time
            for state in self.arms.values()
        ]
        if any(value is None for value in generated_times):
            self.robot_command_hold_required = True
            self.robot_command_hold_reason = (
                "safe dual-arm command timestamp unavailable"
            )
            self.update_safety_supervisor(now_monotonic)
            self.publish_robot_command_status(
                False,
                self.robot_command_hold_reason,
            )
            return None

        generated_monotonic = min(generated_times)
        try:
            result = self.robot_command_dispatcher.dispatch(
                q_commands,
                q_measured,
                generated_monotonic=generated_monotonic,
                safety_command_allowed=safety_decision.command_allowed,
                now_monotonic=now_monotonic,
            )
        except RM75CommandRejectedError as exc:
            self.robot_command_hold_required = True
            self.robot_command_hold_reason = str(exc)
            self.update_safety_supervisor(now_monotonic)
            self.publish_robot_command_status(False, exc)
            return None
        except (
            RM75CommandConnectionError,
            RM75ControllerCommandError,
        ) as exc:
            reason = f"real-robot command transport fault: {exc}"
            self.robot_command_transport_fault = True
            self.robot_command_dispatcher.latch_transport_fault(reason)
            self.update_safety_supervisor(now_monotonic)
            self.publish_robot_command_status(False, reason)
            return None

        self.publish_robot_command_status(result.sent, result.reason)
        return result

    def quest_input_source_is_fresh(
        self,
        now_monotonic,
    ):
        """Require a recent positive freshness report from the Quest bridge."""
        if (
            not self.quest_input_fresh_reported
            or self.last_quest_input_status_time is None
        ):
            return False
        age_s = max(
            0.0,
            now_monotonic - self.last_quest_input_status_time,
        )
        return age_s <= self.deadman_input_timeout_s

    def apply_deadman_decision(
        self,
        decision,
    ):
        """Propagate only deadman edges into the Safety Supervisor."""
        self.deadman_state_pub.publish(
            Bool(data=decision.active)
        )
        if decision.active == self.deadman_active:
            return
        if decision.active:
            self.get_logger().info(
                "Dual-grip deadman ENGAGED."
            )
        else:
            self.get_logger().warning(
                "Dual-grip deadman RELEASED: "
                f"{decision.reason}"
            )
        self.set_deadman_active(
            decision.active
        )

    def grip_callback(
        self,
        side,
        msg,
    ):
        """Consume one verified Quest analog grip sample."""
        now_monotonic = time.perf_counter()
        decision = self.deadman_clutch.update_grip(
            side,
            msg.data,
            now_monotonic=now_monotonic,
            source_fresh=self.quest_input_source_is_fresh(
                now_monotonic
            ),
        )
        self.apply_deadman_decision(
            decision
        )

    def quest_input_fresh_callback(
        self,
        msg,
    ):
        """Fail closed when the bridge reports an APK/logcat source gap."""
        now_monotonic = time.perf_counter()
        self.quest_input_fresh_reported = bool(
            msg.data
        )
        self.last_quest_input_status_time = now_monotonic
        decision = self.deadman_clutch.evaluate(
            now_monotonic,
            source_fresh=self.quest_input_fresh_reported,
        )
        self.apply_deadman_decision(
            decision
        )

    def refresh_deadman(
        self,
        now_monotonic,
    ):
        """Apply local topic and per-hand watchdogs every control cycle."""
        decision = self.deadman_clutch.evaluate(
            now_monotonic,
            source_fresh=self.quest_input_source_is_fresh(
                now_monotonic
            ),
        )
        self.apply_deadman_decision(
            decision
        )
        return decision

    def set_deadman_active(
        self,
        active,
    ):
        """Apply a verified physical deadman decision to safety state."""
        self.deadman_active = bool(
            active
        )
        self.update_safety_supervisor(
            time.perf_counter()
        )

    def reset_safety_fault_callback(
        self,
        _request,
        response,
    ):
        """Request reset of only the software latch, never robot faults."""
        for state in self.arms.values():
            if state.q_safe is not None:
                state.q_candidate = state.q_safe.copy()
                state.q_command = state.q_safe.copy()
                state.command_numeric_valid = True
        requested = self.safety_supervisor.request_fault_reset()
        if not requested:
            response.success = False
            response.message = "safety state is not FAULT"
            return response

        if (
            self.enable_robot_motion
            and not self.deadman_active
            and (
                self.robot_command_transport_fault
                or self.robot_command_dispatcher.faulted
            )
        ):
            self.robot_command_dispatcher.reset_fault()
            self.robot_command_transport_fault = False
            self.robot_command_hold_required = False
            try:
                self.robot_command_dispatcher.connect()
            except RM75CommandConnectionError as exc:
                reason = f"command reconnect failed: {exc}"
                self.robot_command_transport_fault = True
                self.robot_command_dispatcher.latch_transport_fault(
                    reason
                )

        decision = self.update_safety_supervisor(
            time.perf_counter()
        )
        response.success = decision.state != SafetyState.FAULT
        response.message = decision.reason
        return response

    def refresh_robot_readiness(
        self,
        state,
    ):

        if not self.require_robot_state:
            return True

        ready = state.robot_state_ready(
            self.robot_state_timeout_s,
            time.perf_counter(),
        )

        if (
            ready
            and
            not state.initialized_from_robot
        ):

            ready = state.initialize_from_measured(
                self.robot_state_timeout_s,
                time.perf_counter(),
            )

            if ready:

                self.get_logger().info(
                    f"{state.side.upper()} IK initialized from "
                    "fresh q_measured. Waiting for VR anchor."
                )

        if (
            state.robot_ready_previous
            and
            not ready
        ):

            state.invalidate_anchor()

            self.get_logger().warning(
                f"{state.side.upper()} robot state unavailable; "
                "IK target update is HOLD."
            )

        state.robot_ready_previous = ready

        return (
            ready
            and
            state.initialized_from_robot
        )

    def refresh_robot_system_readiness(
        self,
    ):

        if not self.require_robot_state:
            self.robot_system_ready = True
            return True

        readiness = [
            self.refresh_robot_readiness(
                state,
            )
            for state in self.arms.values()
        ]

        ready = all(
            readiness
        )

        if (
            self.robot_system_ready
            and
            not ready
        ):

            for state in self.arms.values():
                state.invalidate_anchor()

            self.get_logger().warning(
                "Dual-arm robot state unavailable; both VR anchors "
                "invalidated and IK target updates are HOLD."
            )

        self.robot_system_ready = ready

        return ready

    def vr_state_is_stale(
        self,
        state,
        now_monotonic,
    ):
        """Return local VR freshness without trusting message timestamps."""
        if state.last_vr_rx_time is None:
            return True
        age_s = max(
            0.0,
            now_monotonic - state.last_vr_rx_time,
        )
        return (
            state.pose_stale
            or age_s > self.vr_pose_timeout_s
        )

    def update_safety_supervisor(
        self,
        now_monotonic,
    ):
        """Refresh both arm observations and evaluate one explicit state."""
        supervisor_start = time.perf_counter()
        self.refresh_collision_safety(
            now_monotonic
        )
        self.refresh_actuator_safety(
            now_monotonic
        )
        self.refresh_following_safety(now_monotonic)

        for side, state in self.arms.items():
            if self.require_robot_state:
                robot_initialized = (
                    state.initialized_from_robot
                    and state.robot_fault_known
                    and state.robot_enable_known
                )
                robot_connected = state.robot_connected
                robot_stale = (
                    state.robot_reported_stale
                    or not state.robot_state_ready(
                        self.robot_state_timeout_s,
                        now_monotonic,
                    )
                )
                robot_enabled = state.robot_joints_enabled
                robot_fault = state.robot_fault
                q_measured = state.q_measured
                measured_numeric_valid = (
                    not state.initialized_from_robot
                    or state.robot_data_valid
                )
            else:
                robot_initialized = state.q_safe is not None
                robot_connected = True
                robot_stale = False
                robot_enabled = True
                robot_fault = False
                q_measured = state.q_safe
                measured_numeric_valid = True

            self.safety_supervisor.update_arm(
                side,
                q_measured=q_measured,
                q_candidate=state.q_candidate,
                q_command=state.q_command,
                joint_velocity=state.joint_velocity,
                joint_acceleration=state.joint_acceleration,
                sigma_min=state.last_sigma_min,
                robot_initialized=robot_initialized,
                robot_connected=robot_connected,
                robot_stale=robot_stale,
                robot_enabled=robot_enabled,
                robot_fault=robot_fault,
                vr_tracking_valid=state.tracking_valid,
                vr_stale=self.vr_state_is_stale(
                    state,
                    now_monotonic,
                ),
                last_command_monotonic=(
                    state.last_safe_command_time
                ),
                consecutive_ik_failures=(
                    state.consecutive_ik_failures
                ),
                upstream_numeric_valid=(
                    measured_numeric_valid
                    and state.vr_numeric_valid
                    and state.command_numeric_valid
                ),
            )

        decision = self.safety_supervisor.evaluate(
            deadman_active=self.deadman_active,
            now_monotonic=now_monotonic,
        )

        self.handle_stop_transition(decision, now_monotonic)

        if (
            decision.state == SafetyState.ENGAGED
            and decision.previous_state != SafetyState.ENGAGED
        ):
            self.robot_command_gate_open_since = now_monotonic
        elif decision.state != SafetyState.ENGAGED:
            self.robot_command_gate_open_since = None

        if decision.changed:
            message = (
                f"Safety {decision.previous_state.value} -> "
                f"{decision.state.value}: {decision.reason}"
            )
            if decision.state in (
                SafetyState.HOLD,
                SafetyState.FAULT,
            ):
                self.get_logger().warning(
                    message
                )
            else:
                self.get_logger().info(
                    message
                )

            if decision.state in (
                SafetyState.INIT,
                SafetyState.HOLD,
                SafetyState.FAULT,
            ):
                for state in self.arms.values():
                    state.invalidate_anchor()

        self.safety_state_pub.publish(
            String(data=decision.state.value)
        )
        self.command_allowed_pub.publish(
            Bool(data=decision.command_allowed)
        )
        self.timing_monitor.record(
            "supervisor_processing",
            time.perf_counter() - supervisor_start,
        )
        return decision

    # =========================================================
    # Quest callbacks
    # =========================================================

    def left_tracking_callback(
        self,
        msg,
    ):
        self.update_tracking_state(
            side="left",
            valid=msg.data,
        )

    def right_tracking_callback(
        self,
        msg,
    ):
        self.update_tracking_state(
            side="right",
            valid=msg.data,
        )

    def update_tracking_state(
        self,
        side,
        valid,
    ):

        state = self.arms[
            side
        ]

        valid = bool(
            valid
        )

        # 状态没有发生变化，不重复处理。
        if valid == state.tracking_valid:
            return

        state.tracking_valid = valid

        if not valid:

            # 当前 VR pose 不再可信，旧相对位移全部作废。
            state.invalidate_anchor()

            self.get_logger().warning(
                f"{side.upper()} Quest tracking LOST."
            )

        else:

            # tracking 恢复以后，
            # 等下一帧有效 Pose 到达再重新 anchor。
            state.need_reanchor = True

            self.get_logger().info(
                f"{side.upper()} Quest tracking RECOVERED. "
                "Waiting for fresh pose to re-anchor."
            )

    def left_pose_callback(
        self,
        msg,
    ):

        self.update_vr_pose(
            side="left",
            msg=msg,
        )

    def right_pose_callback(
        self,
        msg,
    ):

        self.update_vr_pose(
            side="right",
            msg=msg,
        )

    def update_vr_pose(
        self,
        side,
        msg,
    ):

        state = self.arms[
            side
        ]

        # tracking 无效时，Pose 一律不接受
        if not state.tracking_valid:
            return

        # 真实状态未初始化、通信断开或反馈 stale 时，
        # 不允许 capture anchor，更不允许 IK 更新目标。
        if (
            self.require_robot_state
            and
            not self.refresh_robot_system_readiness()
        ):
            return

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

            state.vr_numeric_valid = False

            self.get_logger().warning(
                f"{side.upper()} invalid Quest pose: "
                f"{exc}"
            )

            return

        state.vr_numeric_valid = True

        vr_rx_time = (
            time.perf_counter()
        )

        state.T_vr_latest = T_vr

        state.last_vr_rx_time = (
            vr_rx_time
        )

        state.pose_stale = False

        # READY/HOLD only observe freshness.  They never consume relative VR
        # displacement.  After ENGAGED, the next fresh sample becomes the new
        # coincident controller/robot anchor.
        if self.safety_supervisor.state != SafetyState.ENGAGED:
            return

        # 首次启动或 tracking 恢复后：
        # 同时重新建立 VR anchor 和 robot anchor。
        if (
            state.need_reanchor
            or
            not state.anchored
        ):

            captured = state.capture_vr_anchor(
                T_vr=T_vr,
                require_robot_state=
                    self.require_robot_state,
                robot_timeout_s=
                    self.robot_state_timeout_s,
                now_monotonic=
                    vr_rx_time,
            )

            if not captured:
                return

            state.last_vr_rx_time = (
                vr_rx_time
            )

            self.get_logger().info(
                f"{side.upper()} VR/EE anchor captured."
            )

    # =========================================================
    # 50 Hz main loop
    # =========================================================

    def next_joint_limit_dt(
        self,
        now_monotonic,
    ):
        """Return a conservative measured period for joint rate limiting."""
        now_monotonic = float(now_monotonic)
        if not np.isfinite(now_monotonic):
            raise ValueError("control clock must be finite")

        previous = self.last_control_cycle_time
        self.last_control_cycle_time = now_monotonic
        if previous is None:
            return self.control_period_s

        elapsed_s = now_monotonic - previous
        if not np.isfinite(elapsed_s) or elapsed_s <= 0.0:
            return 0.0

        # A late callback must not accumulate a larger one-frame joint jump.
        return min(elapsed_s, self.control_period_s)

    def control_update(
        self,
    ):

        cycle_start = (
            time.perf_counter()
        )
        self.timing_monitor.begin_cycle(cycle_start)

        joint_limit_dt_s = self.next_joint_limit_dt(
            cycle_start
        )

        now = (
            self.get_clock()
            .now()
            .to_msg()
        )

        self.robot_system_ready = (
            self.refresh_robot_system_readiness()
        )

        self.refresh_deadman(
            cycle_start
        )

        self.update_safety_supervisor(
            cycle_start
        )

        # =====================================================
        # LEFT / RIGHT 使用同一个函数处理。
        # =====================================================

        for side in (
            "left",
            "right",
        ):

            state = self.arms[
                side
            ]

            try:
                self.update_arm(
                    state=state,
                    stamp=now,
                    joint_limit_dt_s=joint_limit_dt_s,
                )
                self.timing_monitor.record(
                    f"ik_{side}", state.last_solve_ms / 1000.0
                )
            except (
                FloatingPointError,
                KeyError,
                TypeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                state.command_numeric_valid = False
                self.get_logger().error(
                    f"{side.upper()} rejected invalid numeric control "
                    f"result: {exc}"
                )
                self.update_safety_supervisor(
                    time.perf_counter()
                )

        # Re-evaluate after IK so malformed candidates and repeated failures
        # are visible before any actuator dispatch or dry-run publication.
        command_evaluation_time = time.perf_counter()
        safety_decision = self.update_safety_supervisor(
            command_evaluation_time
        )

        dispatch_start = time.perf_counter()
        dispatch_result = self.dispatch_robot_commands(
            safety_decision,
            time.perf_counter(),
        )
        self.timing_monitor.record(
            "command_send", time.perf_counter() - dispatch_start
        )
        if (
            dispatch_result is not None
            and dispatch_result.ack_latency_s is not None
        ):
            for side, latency in zip(
                ("left", "right"), dispatch_result.ack_latency_s
            ):
                self.timing_monitor.record(f"command_ack_{side}", latency)

        for side, state in self.arms.items():
            if state.last_vr_rx_time is not None:
                self.timing_monitor.record(
                    f"vr_input_age_{side}",
                    max(0.0, cycle_start - state.last_vr_rx_time),
                )
            if state.last_robot_state_rx_time is not None:
                self.timing_monitor.record(
                    f"robot_state_age_{side}",
                    max(0.0, cycle_start - state.last_robot_state_rx_time),
                )

        # =====================================================
        # 两臂都完成 measured-state 初始化后，
        # 每周期发布一次完整双臂安全 q。
        # =====================================================

        self.publish_dual_joint_state(
            now
        )

        cycle_ms = (
            time.perf_counter()
            - cycle_start
        ) * 1000.0
        self.timing_monitor.record("full_control_cycle", cycle_ms / 1000.0)

        # =====================================================
        # Diagnostics
        # =====================================================

        self.frame_counter += 1

        if (
            self.frame_counter
            % 25
            == 0
        ):

            self.print_diagnostics(
                cycle_ms
            )
            self.timing_diagnostics_pub.publish(
                String(
                    data=json.dumps(
                        self.timing_monitor.summary(), sort_keys=True
                    )
                )
            )
            self.publish_control_diagnostics()

    def publish_control_diagnostics(self):
        """Publish recordable IK, sigma, limiter, and feedback provenance."""
        payload = {
            "safety_state": self.safety_supervisor.state.value,
            "command_allowed": (
                self.safety_supervisor.state == SafetyState.ENGAGED
            ),
            "arms": {},
        }
        for side, state in self.arms.items():
            result = state.last_result or {}
            payload["arms"][side] = {
                "ik_success": result.get("success"),
                "ik_projected": result.get("projected"),
                "consecutive_ik_failures": state.consecutive_ik_failures,
                "sigma_min": state.last_sigma_min,
                "singularity_region": state.last_singularity_region,
                "singularity_speed_scale": (
                    state.last_singularity_speed_scale
                ),
                "ik_solve_ms": state.last_solve_ms,
                "joint_rate_limited": state.last_joint_rate_limited,
                "joint_acceleration_limited": (
                    state.last_joint_acceleration_limited
                ),
                "joint_soft_limited": state.last_joint_soft_limited,
                "measured_velocity_source": (
                    state.measured_velocity_source
                ),
                "measured_sample_period_s": (
                    state.measured_sample_period_s
                ),
                "measured_kinematics_valid": (
                    state.measured_kinematics_valid
                ),
            }
        self.control_diagnostics_pub.publish(
            String(data=json.dumps(payload, sort_keys=True))
        )

    # =========================================================
    # Process one arm
    # =========================================================

    def update_arm(
        self,
        state,
        stamp,
        joint_limit_dt_s,
    ):

        if self.safety_supervisor.state != SafetyState.ENGAGED:
            return

        if (
            self.require_robot_state
            and
            not self.robot_system_ready
        ):
            return

        if (
            state.q_safe is None
            or
            state.T_safe is None
        ):
            return

        if joint_limit_dt_s <= 0.0:
            return

        if not state.tracking_valid:
            return

        if (
            not state.anchored
            or
            state.T_vr_latest is None
        ):
            return

        # VR data stream timeout protection
        if (
            state.last_vr_rx_time is None
            or
            (
                time.perf_counter()
                - state.last_vr_rx_time
            )
            > self.vr_pose_timeout_s
        ):

            if not state.pose_stale:

                state.pose_stale = True

                # 数据恢复后重新 anchor，
                # 防止断流期间手柄运动造成跳变。
                state.invalidate_anchor()

                self.get_logger().warning(
                    f"{state.side.upper()} Quest pose STALE."
                )

            return

        # =====================================================
        # 1. VR incremental pose -> Cartesian raw target
        # =====================================================

        T_raw = (
            map_vr_pose_to_robot_target(
                T_vr_anchor=
                    state.T_vr_anchor,

                T_vr_current=
                    state.T_vr_latest,

                T_ee_anchor=
                    state.T_ee_anchor,

                side=
                    state.side,

                position_scale=
                    self.position_scale,

                orientation_scale=
                    self.orientation_scale,
            )
        )

        current_sigma_min = minimum_singular_value(
            state.q_safe,
            state.model,
        )
        singularity_rate_scale = singularity_speed_scale(
            current_sigma_min,
            sigma_stop=self.sigma_stop,
            sigma_warn=self.sigma_warn,
        )
        current_singularity_region = singularity_region(
            current_sigma_min,
            sigma_stop=self.sigma_stop,
            sigma_warn=self.sigma_warn,
        )
        state.last_current_sigma_min = float(current_sigma_min)
        state.last_singularity_speed_scale = float(
            singularity_rate_scale
        )
        state.last_singularity_region = current_singularity_region
        collision_rate_scale = float(
            self.safety_supervisor.collision_speed_scale
        )
        combined_rate_scale = (
            singularity_rate_scale
            * collision_rate_scale
        )

        # =====================================================
        # 2. Cartesian command
        # =====================================================

        if self.unrestricted_simulation:

            # -------------------------------------------------
            # Simulation:
            #
            # Quest raw target 直接送给 IK。
            #
            # 不做：
            # - translation rate limit
            # - rotation rate limit
            # -------------------------------------------------

            T_command = (
                T_raw.copy()
            )

            # 这里只构造 diagnostics，
            # 不参与任何限制。

            delta_p = (
                T_raw[:3, 3]
                -
                state.T_safe[:3, 3]
            )

            translation_distance = (
                np.linalg.norm(
                    delta_p
                )
            )

            R_relative = (
                T_raw[:3, :3]
                @ state.T_safe[:3, :3].T
            )

            rotation_distance = (
                np.linalg.norm(
                    Rotation
                    .from_matrix(
                        R_relative
                    )
                    .as_rotvec()
                )
            )

            limit_result = {
                "T_limited":
                    T_command,

                "translation_distance":
                    float(
                        translation_distance
                    ),

                "translation_step":
                    float(
                        translation_distance
                    ),

                "rotation_distance":
                    float(
                        rotation_distance
                    ),

                "rotation_step":
                    float(
                        rotation_distance
                    ),

                "translation_limited":
                    False,

                "rotation_limited":
                    False,
            }

        else:

            # -------------------------------------------------
            # Safe mode.  The nominal Cartesian rates are converted to this
            # cycle's step budget.  Smoothstep scaling is 1 in the safe
            # region, continuously decreases through warning, and is 0 at
            # or below sigma_stop.  Projection and final-command hold remain
            # independent hard barriers.
            # -------------------------------------------------

            max_translation_step = (
                self.max_cartesian_translation_rate
                * joint_limit_dt_s
                * combined_rate_scale
            )
            max_rotation_step = (
                self.max_cartesian_rotation_rate
                * joint_limit_dt_s
                * combined_rate_scale
            )

            limit_result = (
                limit_pose_step(
                    T_current=state.T_safe,
                    T_desired=T_raw,
                    max_translation_step=max_translation_step,
                    max_rotation_step=max_rotation_step,
                )
            )

            T_command = (
                limit_result[
                    "T_limited"
                ]
            )

        # =====================================================
        # 3. IK configuration for this arm
        # =====================================================

        ik_kwargs = {
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
                state.q_preferred,

            "preferred_posture_gain":
                1.0,

            "max_null_step":
                np.deg2rad(
                    0.10
                ),
        }

        # =====================================================
        # 4. IK
        # =====================================================

        solve_start = (
            time.perf_counter()
        )

        if self.unrestricted_simulation:

            # -------------------------------------------------
            # UNRESTRICTED SIMULATION
            #
            # T_raw / T_command
            #       ↓
            # direct IK
            #
            # 绕过 Cartesian target 级别的：
            #
            # target feasibility
            # binary projection
            #
            # 但最终 q_command 的限速、FK 和 sigma 检查不会绕过。
            # -------------------------------------------------

            ik_result = (
                solve_ik(
                    T_target=
                        T_command,

                    q_seed=
                        state.q_safe,

                    model=
                        state.model,

                    **ik_kwargs,
                )
            )

            if ik_result[
                "success"
            ]:

                q_new = (
                    ik_result[
                        "q"
                    ].copy()
                )

                sigma_min = (
                    minimum_singular_value(
                        q_new,
                        state.model,
                    )
                )

                result = {
                    "success":
                        True,

                    "projected":
                        False,

                    "alpha":
                        1.0,

                    "T_safe":
                        T_command.copy(),

                    "q_safe":
                        q_new,

                    "sigma_min":
                        sigma_min,

                    "raw_ik_success":
                        True,
                }

            else:

                # IK 连数学解都没有得到，
                # 这时没有任何 q 可以发布。
                #
                # 保持上一帧仅仅是防止程序崩溃，
                # 不属于 feasibility 安全限制。

                result = {
                    "success":
                        False,

                    "projected":
                        False,

                    "alpha":
                        1.0,

                    "T_safe":
                        state.T_safe.copy(),

                    "q_safe":
                        state.q_safe.copy(),

                    "sigma_min":
                        minimum_singular_value(
                            state.q_safe,
                            state.model,
                        ),

                    "raw_ik_success":
                        False,
                }

        else:

            # -------------------------------------------------
            # SAFE MODE
            # -------------------------------------------------

            result = (
                project_target_to_feasible(
                    T_safe=
                        state.T_safe,

                    T_raw=
                        T_command,

                    q_safe=
                        state.q_safe,

                    model=
                        state.model,

                    sigma_stop=
                        self.sigma_stop,

                    binary_iterations=
                        self.binary_iterations,

                    ik_kwargs=
                        ik_kwargs,
                )
            )

        result = dict(result)
        result["current_sigma_min"] = float(current_sigma_min)
        result["singularity_region"] = current_singularity_region
        result["singularity_speed_scale"] = float(
            singularity_rate_scale
        )
        result["collision_speed_scale"] = collision_rate_scale
        result["combined_speed_scale"] = combined_rate_scale

        solve_ms = (
            time.perf_counter()
            - solve_start
        ) * 1000.0

        if not result[
            "success"
        ]:

            try:
                state.q_candidate = np.asarray(
                    result[
                        "q_safe"
                    ],
                    dtype=float,
                ).copy()
                state.last_sigma_min = float(
                    result[
                        "sigma_min"
                    ]
                )
                failed_result_numeric_valid = (
                    state.q_candidate.shape == (state.model.DOF,)
                    and np.all(np.isfinite(state.q_candidate))
                    and np.isfinite(state.last_sigma_min)
                    and state.last_sigma_min >= 0.0
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                state.q_candidate = np.full(
                    state.model.DOF,
                    np.nan,
                )
                state.last_sigma_min = np.nan
                failed_result_numeric_valid = False

            state.command_numeric_valid = bool(
                failed_result_numeric_valid
            )
            state.consecutive_ik_failures += 1

            state.last_solve_ms = (
                solve_ms
            )

            state.last_result = (
                result
            )

            state.last_limit_result = (
                limit_result
            )

            self.update_safety_supervisor(
                time.perf_counter()
            )

            return

        # =====================================================
        # 5. Validate the projected IK candidate before rate limiting
        # =====================================================

        try:
            q_candidate = np.asarray(
                result[
                    "q_safe"
                ],
                dtype=float,
            )
            T_candidate = np.asarray(
                result[
                    "T_safe"
                ],
                dtype=float,
            )
            sigma_min = float(
                result[
                    "sigma_min"
                ]
            )
            candidate_numeric_valid = (
                q_candidate.shape == (state.model.DOF,)
                and T_candidate.shape == (4, 4)
                and np.all(np.isfinite(q_candidate))
                and np.all(np.isfinite(T_candidate))
                and np.all(q_candidate >= state.model.q_min)
                and np.all(q_candidate <= state.model.q_max)
                and np.isfinite(sigma_min)
                and sigma_min >= 0.0
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            q_candidate = np.full(
                state.model.DOF,
                np.nan,
            )
            T_candidate = state.T_safe.copy()
            sigma_min = np.nan
            candidate_numeric_valid = False

        state.q_candidate = q_candidate.copy()
        state.last_sigma_min = sigma_min
        state.command_numeric_valid = bool(
            candidate_numeric_valid
        )

        decision = self.update_safety_supervisor(
            time.perf_counter()
        )
        if not decision.command_allowed:
            return

        # =====================================================
        # 6. q_candidate -> soft target -> qddot/qdot -> q_command
        # =====================================================

        q_current = state.q_safe.copy()
        active_joint_velocity_limit = (
            state.joint_velocity_limit
            * collision_rate_scale
        )
        q_soft_target, joint_soft_limited = limit_joint_soft_position(
            q_current=q_current,
            q_target=q_candidate,
            soft_min=state.q_soft_min,
            soft_max=state.q_soft_max,
        )

        previous_joint_velocity = state.joint_velocity.copy()
        (
            q_command,
            _acceleration_limited_velocity,
            joint_acceleration_limited,
        ) = limit_joint_acceleration(
            q_current=q_current,
            q_target=q_soft_target,
            qd_current=previous_joint_velocity,
            qdd_limit=state.joint_acceleration_limit,
            dt=joint_limit_dt_s,
        )

        q_command, joint_rate_limited = limit_joint_velocity(
            q_current=q_current,
            q_target=q_command,
            qd_limit=active_joint_velocity_limit,
            dt=joint_limit_dt_s,
        )

        q_command, final_soft_limited = limit_joint_soft_position(
            q_current=q_current,
            q_target=q_command,
            soft_min=state.q_soft_min,
            soft_max=state.q_soft_max,
        )
        joint_soft_limited = bool(
            joint_soft_limited or final_soft_limited
        )

        limited_sigma_min = minimum_singular_value(
            q_command,
            state.model,
        )
        singularity_hold = limited_sigma_min <= self.sigma_stop
        if singularity_hold:
            q_command = q_current.copy()

        # The final Cartesian safe state and singular value must describe
        # the actual rate-limited joint command, never the IK candidate.
        T_joint_command = forward_kinematics(
            q_command,
            model=state.model,
        )
        command_sigma_min = minimum_singular_value(
            q_command,
            state.model,
        )
        joint_velocity = (
            q_command - q_current
        ) / joint_limit_dt_s
        joint_acceleration = (
            joint_velocity - previous_joint_velocity
        ) / joint_limit_dt_s

        command_numeric_valid = (
            q_command.shape == (state.model.DOF,)
            and T_joint_command.shape == (4, 4)
            and joint_velocity.shape == (state.model.DOF,)
            and joint_acceleration.shape == (state.model.DOF,)
            and np.all(np.isfinite(q_command))
            and np.all(np.isfinite(T_joint_command))
            and np.all(np.isfinite(joint_velocity))
            and np.all(np.isfinite(joint_acceleration))
            and np.all(q_command >= state.model.q_min)
            and np.all(q_command <= state.model.q_max)
            and np.all(q_command >= state.q_soft_min)
            and np.all(q_command <= state.q_soft_max)
            and np.isfinite(command_sigma_min)
            and command_sigma_min >= self.sigma_stop
        )

        state.last_candidate_sigma_min = sigma_min
        state.last_joint_rate_limited = bool(joint_rate_limited)
        state.last_joint_acceleration_limited = bool(
            joint_acceleration_limited
        )
        state.last_joint_soft_limited = bool(joint_soft_limited)
        state.last_joint_limit_dt_s = float(joint_limit_dt_s)
        state.singularity_hold = bool(singularity_hold)
        state.last_sigma_min = float(command_sigma_min)
        state.command_numeric_valid = bool(command_numeric_valid)

        if not command_numeric_valid:
            self.update_safety_supervisor(
                time.perf_counter()
            )
            return

        previous_q_command = state.q_command.copy()
        previous_joint_acceleration = state.joint_acceleration.copy()
        state.q_command = q_command.copy()
        state.joint_velocity = joint_velocity.copy()
        state.joint_acceleration = joint_acceleration.copy()

        decision = self.update_safety_supervisor(
            time.perf_counter()
        )
        if not decision.command_allowed:
            state.q_command = previous_q_command
            state.joint_velocity = previous_joint_velocity
            state.joint_acceleration = previous_joint_acceleration
            return

        state.T_safe = (
            T_joint_command.copy()
        )

        state.q_safe = (
            state.q_command.copy()
        )
        state.consecutive_ik_failures = 0
        state.last_safe_command_time = time.perf_counter()

        result = dict(result)
        result["candidate_sigma_min"] = sigma_min
        result["limited_sigma_min"] = limited_sigma_min
        result["sigma_min"] = command_sigma_min
        result["joint_rate_limited"] = bool(joint_rate_limited)
        result["joint_acceleration_limited"] = bool(
            joint_acceleration_limited
        )
        result["joint_acceleration"] = joint_acceleration.copy()
        result["joint_soft_limited"] = bool(joint_soft_limited)
        result["joint_limit_dt_s"] = float(joint_limit_dt_s)
        result["singularity_hold"] = bool(singularity_hold)

        # =====================================================
        # 7. Independent actual-pose reporting
        # =====================================================

        if (
            self.require_robot_state
            and
            state.T_measured is not None
        ):

            T_actual = (
                state.T_measured.copy()
            )

        else:

            T_actual = (
                forward_kinematics(
                    state.q_safe,
                    model=state.model,
                )
            )

        # =====================================================
        # 8. Publish Pose topics
        # =====================================================

        pubs = (
            self.pose_publishers[
                state.side
            ]
        )

        self.publish_pose(
            T=T_raw,
            publisher=pubs["raw"],
            base_frame=state.base_frame,
            stamp=stamp,
        )

        self.publish_pose(
            T=T_command,
            publisher=pubs["command"],
            base_frame=state.base_frame,
            stamp=stamp,
        )

        self.publish_pose(
            T=state.T_safe,
            publisher=pubs["safe"],
            base_frame=state.base_frame,
            stamp=stamp,
        )

        self.publish_pose(
            T=T_actual,
            publisher=pubs["actual"],
            base_frame=state.base_frame,
            stamp=stamp,
        )

        # =====================================================
        # 9. Publish TF
        # =====================================================

        self.publish_tf(
            T=T_raw,
            parent_frame=state.base_frame,
            child_frame=(
                f"{state.side}_raw_target"
            ),
            stamp=stamp,
        )

        self.publish_tf(
            T=T_command,
            parent_frame=state.base_frame,
            child_frame=(
                f"{state.side}_command_target"
            ),
            stamp=stamp,
        )

        self.publish_tf(
            T=state.T_safe,
            parent_frame=state.base_frame,
            child_frame=(
                f"{state.side}_safe_target"
            ),
            stamp=stamp,
        )

        self.publish_tf(
            T=T_actual,
            parent_frame=state.base_frame,
            child_frame=(
                f"{state.side}_actual_tcp"
            ),
            stamp=stamp,
        )

        # =====================================================
        # 10. Save diagnostics
        # =====================================================

        state.last_solve_ms = (
            solve_ms
        )

        state.last_result = (
            result
        )

        state.last_limit_result = (
            limit_result
        )

    # =========================================================
    # Combined 14-joint publisher
    # =========================================================

    def publish_dual_joint_state(
        self,
        stamp,
    ):

        left = self.arms[
            "left"
        ]

        right = self.arms[
            "right"
        ]

        if (
            left.q_command is None
            or
            right.q_command is None
        ):
            return

        msg = JointState()

        msg.header.stamp = (
            stamp
        )

        msg.name = (
            [
                f"l_rm75_joint_{i}"
                for i in range(
                    1,
                    8,
                )
            ]
            +
            [
                f"r_rm75_joint_{i}"
                for i in range(
                    1,
                    8,
                )
            ]
        )

        msg.position = (
            left.q_command.tolist()
            +
            right.q_command.tolist()
        )

        self.joint_pub.publish(
            msg
        )

    # =========================================================
    # Pose publisher
    # =========================================================

    def publish_pose(
        self,
        T,
        publisher,
        base_frame,
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

        msg.header.stamp = (
            stamp
        )

        msg.header.frame_id = (
            base_frame
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
    # TF publisher
    # =========================================================

    def publish_tf(
        self,
        T,
        parent_frame,
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

        msg = TransformStamped()

        msg.header.stamp = (
            stamp
        )

        msg.header.frame_id = (
            parent_frame
        )

        msg.child_frame_id = (
            child_frame
        )

        msg.transform.translation.x = float(
            T[0, 3]
        )

        msg.transform.translation.y = float(
            T[1, 3]
        )

        msg.transform.translation.z = float(
            T[2, 3]
        )

        msg.transform.rotation.x = float(
            quat[0]
        )

        msg.transform.rotation.y = float(
            quat[1]
        )

        msg.transform.rotation.z = float(
            quat[2]
        )

        msg.transform.rotation.w = float(
            quat[3]
        )

        self.tf_broadcaster.sendTransform(
            msg
        )

    # =========================================================
    # Diagnostics
    # =========================================================

    def print_diagnostics(
        self,
        cycle_ms,
    ):

        pieces = []

        for side in (
            "left",
            "right",
        ):

            state = self.arms[
                side
            ]

            label = (
                "L"
                if side == "left"
                else "R"
            )

            if (
                self.require_robot_state
                and
                not state.robot_state_ready(
                    self.robot_state_timeout_s,
                    time.perf_counter(),
                )
            ):

                pieces.append(
                    f"{label}:ROBOT_WAIT"
                )

                continue

            if not state.anchored:

                pieces.append(
                    f"{label}:WAIT"
                )

                continue

            if (
                state.last_result is None
                or
                state.last_limit_result is None
            ):

                pieces.append(
                    f"{label}:INIT"
                )

                continue

            result = (
                state.last_result
            )

            limit_result = (
                state.last_limit_result
            )

            pieces.append(
                (
                    "{}:"
                    "d={:.1f}->{:.1f}mm "
                    "r={:.1f}->{:.1f}deg "
                    "lim=({},{}) "
                    "ikok={} "
                    "proj={} "
                    "a={:.3f} "
                    "s={:.4f} "
                    "sg={}:x{:.2f} "
                    "jlim={} "
                    "slim={} "
                    "qd={:.1f}deg/s "
                    "shold={} "
                    "ik={:.1f}ms"
                ).format(

                    label,

                    limit_result[
                        "translation_distance"
                    ] * 1000.0,

                    limit_result[
                        "translation_step"
                    ] * 1000.0,

                    np.rad2deg(
                        limit_result[
                            "rotation_distance"
                        ]
                    ),

                    np.rad2deg(
                        limit_result[
                            "rotation_step"
                        ]
                    ),

                    int(
                        limit_result[
                            "translation_limited"
                        ]
                    ),

                    int(
                        limit_result[
                            "rotation_limited"
                        ]
                    ),

                    int(
                        result[
                            "raw_ik_success"
                        ]
                    ),

                    int(
                        result[
                            "projected"
                        ]
                    ),

                    result[
                        "alpha"
                    ],

                    result[
                        "sigma_min"
                    ],

                    state.last_singularity_region,

                    state.last_singularity_speed_scale,

                    int(
                        state.last_joint_rate_limited
                    ),

                    int(
                        state.last_joint_soft_limited
                    ),

                    np.rad2deg(
                        np.max(
                            np.abs(
                                state.joint_velocity
                            )
                        )
                    ),

                    int(
                        state.singularity_hold
                    ),

                    state.last_solve_ms,
                )
            )

        self.get_logger().info(
            " | ".join(
                pieces
            )
            +
            " | cycle={:.1f}ms".format(
                cycle_ms
            )
        )

    def destroy_node(self):
        """Best-effort controlled stop on an existing armed connection."""
        if self.robot_command_dispatcher.motion_armed:
            result = self.robot_command_dispatcher.request_stop(
                StopClass.CONTROLLED_STOP,
                "fusion node shutdown",
                time.perf_counter(),
            )
            if not result.dry_run and not result.all_acknowledged:
                self.get_logger().error(
                    "shutdown stop was not acknowledged by both arms; "
                    "physical emergency stop may be required"
                )
        self.robot_command_dispatcher.close()
        return super().destroy_node()


def main(
    args=None,
):

    rclpy.init(
        args=args
    )

    node = (
        QuestDualIKFusion()
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
