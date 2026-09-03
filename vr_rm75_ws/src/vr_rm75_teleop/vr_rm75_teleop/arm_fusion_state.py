"""Per-arm measured-state alignment for VR teleoperation."""

from __future__ import annotations

import math
import time

import numpy as np

from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.robot_feedback_monitor import RobotFeedbackMonitor


class ArmFusionState:
    """Hold one arm's feedback, safe command, and relative-pose anchors."""

    def __init__(self, side, fallback_q=None):
        """Create an uninitialized real-arm state or explicit RViz fallback."""
        self.side = str(side).lower()
        if self.side == "left":
            self.prefix = "l"
        elif self.side == "right":
            self.prefix = "r"
        else:
            raise ValueError("side must be 'left' or 'right'")

        self.base_frame = f"{self.prefix}_rm75_base_link"
        self.model = RM75Model(side=self.side)

        # Safe/command state remains absent until measured state is accepted.
        self.q_start = None
        self.q_preferred = None
        self.q_safe = None
        self.T_safe = None
        self.T_ee_anchor = None

        # Actual robot feedback is independent of the dry-run command state.
        self.q_measured = None
        self.T_measured = None
        self.last_robot_state_rx_time = None
        self.robot_connected = False
        self.robot_reported_stale = True
        self.robot_data_valid = False
        self.robot_state_initialized = False
        self.initialized_from_robot = False
        self.robot_fault = False
        self.robot_fault_known = False
        self.robot_joints_enabled = False
        self.robot_enable_known = False
        self.last_robot_state_error = None
        self.robot_ready_previous = False
        self.feedback_monitor = RobotFeedbackMonitor(self.model.DOF)
        self.qdot_measured = None
        self.qddot_measured = None
        self.measured_kinematics_valid = False
        self.measured_velocity_source = "unavailable"
        self.measured_sample_period_s = None
        self.measured_kinematics_reason = "no measured state"

        # VR relative-pose state.
        self.T_vr_latest = None
        self.T_vr_anchor = None
        self.anchored = False
        self.tracking_valid = False
        self.need_reanchor = True
        self.last_vr_rx_time = None
        self.pose_stale = False
        self.vr_numeric_valid = True

        # IK diagnostics.
        self.last_solve_ms = 0.0
        self.last_result = None
        self.last_limit_result = None
        self.q_candidate = None
        self.q_command = None
        self.joint_velocity = None
        self.joint_velocity_limit = None
        self.joint_acceleration = None
        self.joint_acceleration_limit = None
        self.q_soft_min = None
        self.q_soft_max = None
        self.elbow_branch = None
        self.last_joint_rate_limited = False
        self.last_joint_acceleration_limited = False
        self.last_joint_soft_limited = False
        self.last_joint_limit_dt_s = None
        self.last_candidate_sigma_min = None
        self.last_current_sigma_min = None
        self.last_singularity_region = None
        self.last_singularity_speed_scale = None
        self.singularity_hold = False
        self.last_sigma_min = None
        self.last_safe_command_time = None
        self.last_safe_command_dt_s = None
        self.consecutive_ik_failures = 0
        self.command_numeric_valid = True

        if fallback_q is not None:
            q = self.validate_q(fallback_q)
            self._set_initial_safe_state(q, from_robot=False)

    def validate_q(self, q):
        """Return a copied 7-vector after finite and hard-limit checks."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.model.DOF,):
            raise ValueError(f"q must have shape ({self.model.DOF},)")
        if not np.all(np.isfinite(q)):
            raise ValueError("q contains NaN or Infinity")
        if np.any(q < self.model.q_min) or np.any(q > self.model.q_max):
            raise ValueError("q is outside RM75 hard joint limits")
        return q.copy()

    def configure_teleop_soft_limits(
        self,
        lower,
        upper,
        elbow_branch,
    ):
        """Install strict command bounds before measured-state startup."""
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if (
            lower.shape != (self.model.DOF,)
            or upper.shape != (self.model.DOF,)
        ):
            raise ValueError("soft limits must be 7-vectors")
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
        ):
            raise ValueError("soft limits must be finite")
        if np.any(lower >= upper):
            raise ValueError("soft-limit intervals must be nonempty")
        if (
            np.any(lower <= self.model.q_min)
            or np.any(upper >= self.model.q_max)
        ):
            raise ValueError("soft limits must be strictly inside hard limits")
        elbow_branch = int(elbow_branch)
        if elbow_branch not in (-1, 1):
            raise ValueError("elbow_branch must be -1 or +1")
        self.q_soft_min = lower.copy()
        self.q_soft_max = upper.copy()
        self.elbow_branch = elbow_branch
        if (
            self.q_safe is not None
            and not self.q_within_soft_limits(self.q_safe)
        ):
            raise ValueError("existing safe state is outside soft limits")

    def q_within_soft_limits(self, q):
        """Return whether q is inside configured teleoperation bounds."""
        if self.q_soft_min is None or self.q_soft_max is None:
            return True
        q = np.asarray(q, dtype=float)
        return bool(
            q.shape == (self.model.DOF,)
            and np.all(np.isfinite(q))
            and np.all(q >= self.q_soft_min)
            and np.all(q <= self.q_soft_max)
        )

    def update_measured_q(
        self,
        q,
        received_monotonic=None,
        qdot_measured=None,
    ):
        """Store one valid measured sample without changing a live command."""
        q = self.validate_q(q)
        if received_monotonic is None:
            received_monotonic = time.monotonic()
        received_monotonic = float(received_monotonic)
        if not math.isfinite(received_monotonic):
            raise ValueError("received_monotonic must be finite")

        self.q_measured = q
        self.T_measured = forward_kinematics(q, model=self.model)
        self.last_robot_state_rx_time = received_monotonic
        self.robot_data_valid = True
        self.last_robot_state_error = None
        kinematics = self.feedback_monitor.update(
            q,
            received_monotonic,
            direct_qdot=qdot_measured,
        )
        self.qdot_measured = kinematics.qdot
        self.qddot_measured = kinematics.qddot
        self.measured_kinematics_valid = kinematics.valid
        self.measured_velocity_source = kinematics.velocity_source
        self.measured_sample_period_s = kinematics.dt_s
        self.measured_kinematics_reason = kinematics.reason

    def reject_measured_q(self, reason):
        """Make robot state unusable immediately after invalid feedback."""
        self.robot_data_valid = False
        self.last_robot_state_error = str(reason)
        self.invalidate_anchor()

    def robot_state_ready(self, timeout_s, now_monotonic=None):
        """Check transport flags, sample validity, and local receive age."""
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        now_monotonic = float(now_monotonic)

        if (
            not self.robot_connected
            or self.robot_reported_stale
            or not self.robot_data_valid
            or self.q_measured is None
            or self.last_robot_state_rx_time is None
        ):
            return False
        age_s = max(0.0, now_monotonic - self.last_robot_state_rx_time)
        return math.isfinite(age_s) and age_s <= timeout_s

    def initialize_from_measured(self, timeout_s, now_monotonic=None):
        """Initialize q_safe/T_safe exactly from fresh measured feedback."""
        if not self.robot_state_ready(timeout_s, now_monotonic):
            return False
        if not self.q_within_soft_limits(self.q_measured):
            self.last_robot_state_error = (
                "measured joints are outside teleoperation soft limits"
            )
            return False
        self._set_initial_safe_state(self.q_measured, from_robot=True)
        return True

    def synchronize_safe_to_measured(self, timeout_s, now_monotonic=None):
        """Reset the hold command to fresh actual state before re-anchoring."""
        if not self.robot_state_ready(timeout_s, now_monotonic):
            return False
        if not self.q_within_soft_limits(self.q_measured):
            self.last_robot_state_error = (
                "measured joints are outside teleoperation soft limits"
            )
            return False
        self.q_safe = self.q_measured.copy()
        self.T_safe = self.T_measured.copy()
        self.T_ee_anchor = self.T_safe.copy()
        self.q_candidate = self.q_safe.copy()
        self.q_command = self.q_safe.copy()
        self.joint_velocity = np.zeros(self.model.DOF)
        self.joint_acceleration = np.zeros(self.model.DOF)
        self.last_joint_rate_limited = False
        self.last_joint_acceleration_limited = False
        self.last_joint_soft_limited = False
        self.last_joint_limit_dt_s = None
        self.last_candidate_sigma_min = None
        self.last_current_sigma_min = None
        self.last_singularity_region = None
        self.last_singularity_speed_scale = None
        self.singularity_hold = False
        self.last_safe_command_time = None
        self.last_safe_command_dt_s = None
        self.consecutive_ik_failures = 0
        return True

    def capture_vr_anchor(
        self,
        T_vr,
        require_robot_state,
        robot_timeout_s,
        now_monotonic=None,
    ):
        """Capture coincident VR/robot anchors with no initial target jump."""
        T_vr = self._validate_transform(T_vr)
        if require_robot_state:
            if not self.robot_state_initialized:
                if not self.initialize_from_measured(
                    robot_timeout_s, now_monotonic
                ):
                    return False
            elif not self.synchronize_safe_to_measured(
                robot_timeout_s, now_monotonic
            ):
                return False
        elif self.q_safe is None or self.T_safe is None:
            return False

        self.T_vr_anchor = T_vr
        self.T_vr_latest = T_vr.copy()
        self.T_ee_anchor = self.T_safe.copy()
        self.anchored = True
        self.need_reanchor = False
        return True

    def invalidate_anchor(self):
        """Discard relative VR history so recovery cannot chase old motion."""
        self.T_vr_latest = None
        self.T_vr_anchor = None
        self.anchored = False
        self.need_reanchor = True
        self.last_vr_rx_time = None

    def _set_initial_safe_state(self, q, from_robot):
        q = self.validate_q(q)
        T = forward_kinematics(q, model=self.model)
        self.q_start = q.copy()
        self.q_preferred = q.copy()
        self.q_safe = q.copy()
        self.T_safe = T.copy()
        self.T_ee_anchor = T.copy()
        self.q_candidate = q.copy()
        self.q_command = q.copy()
        self.joint_velocity = np.zeros(self.model.DOF)
        self.joint_acceleration = np.zeros(self.model.DOF)
        self.last_joint_rate_limited = False
        self.last_joint_acceleration_limited = False
        self.last_joint_soft_limited = False
        self.last_joint_limit_dt_s = None
        self.last_candidate_sigma_min = None
        self.last_current_sigma_min = None
        self.last_singularity_region = None
        self.last_singularity_speed_scale = None
        self.singularity_hold = False
        self.last_safe_command_time = None
        self.last_safe_command_dt_s = None
        self.consecutive_ik_failures = 0
        self.robot_state_initialized = True
        self.initialized_from_robot = bool(from_robot)
        self.invalidate_anchor()

    @staticmethod
    def _validate_transform(T):
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError("T_vr must have shape (4, 4)")
        if not np.all(np.isfinite(T)):
            raise ValueError("T_vr contains NaN or Infinity")
        return T.copy()
