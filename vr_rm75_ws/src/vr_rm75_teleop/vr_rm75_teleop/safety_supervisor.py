"""Explicit dual-arm safety state machine for RM75 teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Dict, Optional, Tuple

import numpy as np

from vr_rm75_teleop.rm75_model import RM75Model


class SafetyState(str, Enum):
    """Operating states of the dual-arm safety boundary."""

    INIT = "INIT"
    READY = "READY"
    ENGAGED = "ENGAGED"
    HOLD = "HOLD"
    FAULT = "FAULT"


@dataclass(frozen=True)
class ArmSafetyObservation:
    """Latest safety-relevant values for one arm."""

    side: str
    q_measured: Optional[Tuple[float, ...]]
    q_candidate: Optional[Tuple[float, ...]]
    q_command: Optional[Tuple[float, ...]]
    joint_velocity: Optional[Tuple[float, ...]]
    joint_acceleration: Optional[Tuple[float, ...]]
    sigma_min: Optional[float]
    robot_initialized: bool
    robot_connected: bool
    robot_stale: bool
    robot_enabled: bool
    robot_fault: bool
    vr_tracking_valid: bool
    vr_stale: bool
    last_command_monotonic: Optional[float]
    consecutive_ik_failures: int
    numeric_valid: bool


@dataclass(frozen=True)
class SafetyDecision:
    """Result of one deterministic state-machine evaluation."""

    previous_state: SafetyState
    state: SafetyState
    changed: bool
    reason: str
    command_allowed: bool


class SafetySupervisor:
    """Maintain dual-arm observations and gate dry-run command generation."""

    SIDES = ("left", "right")

    ALLOWED_TRANSITIONS = {
        SafetyState.INIT: {
            SafetyState.READY,
            SafetyState.FAULT,
        },
        SafetyState.READY: {
            SafetyState.INIT,
            SafetyState.ENGAGED,
            SafetyState.HOLD,
            SafetyState.FAULT,
        },
        SafetyState.ENGAGED: {
            SafetyState.HOLD,
            SafetyState.FAULT,
        },
        SafetyState.HOLD: {
            SafetyState.INIT,
            SafetyState.READY,
            SafetyState.FAULT,
        },
        SafetyState.FAULT: {
            SafetyState.INIT,
        },
    }

    def __init__(
        self,
        command_timeout_s=0.10,
        max_consecutive_ik_failures=3,
        joint_velocity_scale=0.10,
        joint_acceleration_limits=None,
        joint_soft_limits=None,
        require_collision_safety=True,
        require_actuator_safety=False,
        require_following_safety=False,
        monotonic=time.monotonic,
    ):
        """Configure watchdog limits without enabling any command output."""
        command_timeout_s = float(command_timeout_s)
        if not math.isfinite(command_timeout_s) or command_timeout_s <= 0.0:
            raise ValueError("command_timeout_s must be finite and positive")
        if int(max_consecutive_ik_failures) < 1:
            raise ValueError("max_consecutive_ik_failures must be >= 1")
        joint_velocity_scale = float(joint_velocity_scale)
        if (
            not math.isfinite(joint_velocity_scale)
            or not 0.0 < joint_velocity_scale <= 1.0
        ):
            raise ValueError("joint_velocity_scale must be in (0, 1]")

        self.command_timeout_s = command_timeout_s
        self.max_consecutive_ik_failures = int(
            max_consecutive_ik_failures
        )
        self.joint_velocity_scale = joint_velocity_scale
        self.require_collision_safety = bool(require_collision_safety)
        self.require_actuator_safety = bool(require_actuator_safety)
        self.require_following_safety = bool(require_following_safety)
        self._monotonic = monotonic
        self.state = SafetyState.INIT
        self.last_reason = "waiting for dual-arm state"
        self._models = {
            side: RM75Model(side=side) for side in self.SIDES
        }
        self._joint_soft_limits = self._normalize_soft_limits(
            joint_soft_limits
        )
        self._joint_acceleration_limits = (
            self._normalize_joint_acceleration_limits(
                joint_acceleration_limits
            )
        )
        self._observations: Dict[str, ArmSafetyObservation] = {}
        self._fault_reset_requested = False
        self._engaged_since: Optional[float] = None
        self.collision_ready = not self.require_collision_safety
        self.collision_hold_required = False
        self.collision_speed_scale = (
            0.0 if self.require_collision_safety else 1.0
        )
        self.collision_reason = (
            "collision distance unavailable"
            if self.require_collision_safety
            else "collision protection explicitly disabled"
        )
        self.actuator_ready = not self.require_actuator_safety
        self.actuator_hold_required = False
        self.actuator_fault = False
        self.actuator_reason = (
            "robot command interface unavailable"
            if self.require_actuator_safety
            else "robot motion explicitly disabled"
        )
        self.following_ready = not self.require_following_safety
        self.following_hold_required = False
        self.following_reason = (
            "following-error observation unavailable"
            if self.require_following_safety
            else "following-error guard not required"
        )

    def update_arm(
        self,
        side,
        *,
        q_measured=None,
        q_candidate=None,
        q_command=None,
        joint_velocity=None,
        joint_acceleration=None,
        sigma_min=None,
        robot_initialized=False,
        robot_connected=False,
        robot_stale=True,
        robot_enabled=False,
        robot_fault=False,
        vr_tracking_valid=False,
        vr_stale=True,
        last_command_monotonic=None,
        consecutive_ik_failures=0,
        upstream_numeric_valid=True,
    ):
        """Replace one arm observation after validating all numeric fields."""
        side = str(side).lower()
        if side not in self.SIDES:
            raise ValueError("side must be 'left' or 'right'")
        model = self._models[side]

        measured, measured_valid = self._normalize_q(
            q_measured, model, required=bool(robot_initialized)
        )
        candidate, candidate_valid = self._normalize_q(
            q_candidate, model, required=False
        )
        command, command_valid = self._normalize_q(
            q_command, model, required=False
        )
        velocity, velocity_valid = self._normalize_finite_vector(
            joint_velocity, model.DOF
        )
        acceleration, acceleration_valid = self._normalize_finite_vector(
            joint_acceleration, model.DOF
        )

        sigma_valid = True
        if sigma_min is not None:
            try:
                sigma_min = float(sigma_min)
            except (TypeError, ValueError):
                sigma_valid = False
                sigma_min = None
            else:
                sigma_valid = math.isfinite(sigma_min) and sigma_min >= 0.0

        timestamp_valid = True
        if last_command_monotonic is not None:
            try:
                last_command_monotonic = float(last_command_monotonic)
            except (TypeError, ValueError):
                timestamp_valid = False
                last_command_monotonic = None
            else:
                timestamp_valid = math.isfinite(last_command_monotonic)

        try:
            consecutive_ik_failures = int(consecutive_ik_failures)
        except (TypeError, ValueError):
            consecutive_ik_failures = -1
        failure_count_valid = consecutive_ik_failures >= 0

        observation = ArmSafetyObservation(
            side=side,
            q_measured=measured,
            q_candidate=candidate,
            q_command=command,
            joint_velocity=velocity,
            joint_acceleration=acceleration,
            sigma_min=sigma_min,
            robot_initialized=bool(robot_initialized),
            robot_connected=bool(robot_connected),
            robot_stale=bool(robot_stale),
            robot_enabled=bool(robot_enabled),
            robot_fault=bool(robot_fault),
            vr_tracking_valid=bool(vr_tracking_valid),
            vr_stale=bool(vr_stale),
            last_command_monotonic=last_command_monotonic,
            consecutive_ik_failures=consecutive_ik_failures,
            numeric_valid=(
                measured_valid
                and candidate_valid
                and command_valid
                and velocity_valid
                and acceleration_valid
                and sigma_valid
                and timestamp_valid
                and failure_count_valid
                and bool(upstream_numeric_valid)
            ),
        )
        self._observations[side] = observation
        return observation

    def get_observation(self, side):
        """Return the immutable latest observation for one arm, if present."""
        return self._observations.get(str(side).lower())

    def get_joint_limits(self, side):
        """Return immutable hard position and velocity limits for one arm."""
        side = str(side).lower()
        if side not in self.SIDES:
            raise ValueError("side must be 'left' or 'right'")
        model = self._models[side]
        return (
            tuple(float(value) for value in model.q_min),
            tuple(float(value) for value in model.q_max),
            tuple(float(value) for value in model.qd_max),
        )

    def get_joint_soft_limits(self, side):
        """Return configured teleoperation limits, or None when disabled."""
        side = str(side).lower()
        if side not in self.SIDES:
            raise ValueError("side must be 'left' or 'right'")
        limits = self._joint_soft_limits.get(side)
        if limits is None:
            return None
        lower, upper = limits
        return tuple(lower), tuple(upper)

    def get_joint_acceleration_limits(self, side):
        """Return configured qdd limits, or None when not configured."""
        side = str(side).lower()
        if side not in self.SIDES:
            raise ValueError("side must be 'left' or 'right'")
        limits = self._joint_acceleration_limits.get(side)
        if limits is None:
            return None
        return tuple(float(value) for value in limits)

    def update_collision(
        self,
        *,
        ready,
        hold_required,
        speed_scale,
        reason,
    ):
        """Replace the global dual-arm collision guard observation."""
        speed_scale = float(speed_scale)
        if not math.isfinite(speed_scale) or not 0.0 <= speed_scale <= 1.0:
            raise ValueError("collision speed_scale must be in [0, 1]")

        ready = bool(ready)
        hold_required = bool(hold_required)
        if not ready:
            hold_required = True
            speed_scale = 0.0

        self.collision_ready = ready
        self.collision_hold_required = hold_required
        self.collision_speed_scale = speed_scale
        self.collision_reason = str(reason) or "collision safety hold"

    def update_actuator(
        self,
        *,
        ready,
        hold_required,
        fault,
        reason,
    ):
        """Replace the independent real-robot command guard observation."""
        ready = bool(ready)
        hold_required = bool(hold_required)
        fault = bool(fault)
        if not ready or fault:
            hold_required = True

        self.actuator_ready = ready
        self.actuator_hold_required = hold_required
        self.actuator_fault = fault
        self.actuator_reason = str(reason) or "robot command safety hold"

    def update_following(self, *, ready, hold_required, reason):
        """Replace the timestamp-aware closed-loop following-error guard."""
        self.following_ready = bool(ready)
        self.following_hold_required = bool(hold_required)
        if not self.following_ready:
            self.following_hold_required = True
        self.following_reason = str(reason) or "following-error safety hold"

    def request_fault_reset(self):
        """Request a latched FAULT reset on the next safe evaluation."""
        if self.state != SafetyState.FAULT:
            return False
        self._fault_reset_requested = True
        return True

    def evaluate(self, deadman_active=False, now_monotonic=None):
        """Evaluate all guards and perform at most one legal transition."""
        if now_monotonic is None:
            now_monotonic = self._monotonic()
        now_monotonic = float(now_monotonic)
        if not math.isfinite(now_monotonic):
            raise ValueError("now_monotonic must be finite")

        previous = self.state
        deadman_active = bool(deadman_active)

        fatal_reason = self._fatal_reason()
        robot_initialized = self._all(
            lambda item: item.robot_initialized
        )
        robot_ready = self._all(
            lambda item: (
                item.robot_initialized
                and item.robot_connected
                and not item.robot_stale
                and item.robot_enabled
            )
        )
        vr_ready = self._all(
            lambda item: item.vr_tracking_valid and not item.vr_stale
        )
        ik_failed = self._any(
            lambda item: (
                item.consecutive_ik_failures
                >= self.max_consecutive_ik_failures
            )
        )
        velocity_exceeded = self._any(
            self._joint_velocity_exceeded
        )
        acceleration_exceeded = self._any(
            self._joint_acceleration_exceeded
        )
        soft_limit_reason = self._soft_limit_reason()
        collision_hold_reason = self._collision_hold_reason()
        actuator_hold_reason = self._actuator_hold_reason()
        following_hold_reason = self._following_hold_reason()

        if self.state == SafetyState.FAULT:
            if (
                self._fault_reset_requested
                and fatal_reason is None
                and not deadman_active
            ):
                self._transition(
                    SafetyState.INIT,
                    "fault reset accepted; reinitialization required",
                )
            elif self._fault_reset_requested and deadman_active:
                self.last_reason = "fault reset rejected while deadman active"
            elif fatal_reason is not None:
                self.last_reason = fatal_reason
            self._fault_reset_requested = False

        elif fatal_reason is not None:
            self._transition(SafetyState.FAULT, fatal_reason)

        elif self.state == SafetyState.INIT:
            if soft_limit_reason is not None:
                self.last_reason = soft_limit_reason
            elif collision_hold_reason is not None:
                self.last_reason = collision_hold_reason
            elif actuator_hold_reason is not None:
                self.last_reason = actuator_hold_reason
            elif following_hold_reason is not None:
                self.last_reason = following_hold_reason
            elif robot_ready:
                self._transition(
                    SafetyState.READY,
                    "dual-arm state initialized and healthy",
                )

        elif self.state == SafetyState.READY:
            if not robot_initialized:
                self._transition(
                    SafetyState.INIT,
                    "robot initialization lost",
                )
            elif not robot_ready:
                self._transition(
                    SafetyState.HOLD,
                    "robot communication stale, disconnected, or disabled",
                )
            elif collision_hold_reason is not None:
                self._transition(
                    SafetyState.HOLD,
                    collision_hold_reason,
                )
            elif actuator_hold_reason is not None:
                self._transition(
                    SafetyState.HOLD,
                    actuator_hold_reason,
                )
            elif following_hold_reason is not None:
                self._transition(
                    SafetyState.HOLD,
                    following_hold_reason,
                )
            elif soft_limit_reason is not None:
                self._transition(
                    SafetyState.HOLD,
                    soft_limit_reason,
                )
            elif deadman_active and velocity_exceeded:
                self._transition(
                    SafetyState.HOLD,
                    "teleoperation joint velocity limit exceeded",
                )
            elif deadman_active and acceleration_exceeded:
                self._transition(
                    SafetyState.HOLD,
                    "teleoperation joint acceleration limit exceeded",
                )
            elif deadman_active and vr_ready:
                self._transition(
                    SafetyState.ENGAGED,
                    "deadman active with fresh dual-arm VR tracking",
                )
                self._engaged_since = now_monotonic

        elif self.state == SafetyState.ENGAGED:
            if not robot_ready:
                self._transition(
                    SafetyState.HOLD,
                    "robot communication stale, disconnected, or disabled",
                )
            elif not deadman_active:
                self._transition(SafetyState.HOLD, "deadman released")
            elif not vr_ready:
                self._transition(
                    SafetyState.HOLD,
                    "VR tracking lost or pose stale",
                )
            elif collision_hold_reason is not None:
                self._transition(
                    SafetyState.HOLD,
                    collision_hold_reason,
                )
            elif actuator_hold_reason is not None:
                self._transition(
                    SafetyState.HOLD,
                    actuator_hold_reason,
                )
            elif following_hold_reason is not None:
                self._transition(
                    SafetyState.HOLD,
                    following_hold_reason,
                )
            elif soft_limit_reason is not None:
                self._transition(
                    SafetyState.HOLD,
                    soft_limit_reason,
                )
            elif velocity_exceeded:
                self._transition(
                    SafetyState.HOLD,
                    "teleoperation joint velocity limit exceeded",
                )
            elif acceleration_exceeded:
                self._transition(
                    SafetyState.HOLD,
                    "teleoperation joint acceleration limit exceeded",
                )
            elif ik_failed:
                self._transition(
                    SafetyState.HOLD,
                    "consecutive IK failure limit reached",
                )
            elif self._command_watchdog_expired(now_monotonic):
                self._transition(
                    SafetyState.HOLD,
                    "safe command watchdog expired",
                )

        elif self.state == SafetyState.HOLD:
            if not robot_initialized:
                self._transition(
                    SafetyState.INIT,
                    "robot initialization lost",
                )
            elif collision_hold_reason is not None:
                self.last_reason = collision_hold_reason
            elif actuator_hold_reason is not None:
                self.last_reason = actuator_hold_reason
            elif following_hold_reason is not None:
                self.last_reason = following_hold_reason
            elif soft_limit_reason is not None:
                self.last_reason = soft_limit_reason
            elif robot_ready and not deadman_active:
                self._transition(
                    SafetyState.READY,
                    "hold conditions cleared; waiting for deadman",
                )

        if self.state != SafetyState.ENGAGED:
            self._engaged_since = None

        return SafetyDecision(
            previous_state=previous,
            state=self.state,
            changed=self.state != previous,
            reason=self.last_reason,
            command_allowed=self.state == SafetyState.ENGAGED,
        )

    def _fatal_reason(self):
        if self.require_actuator_safety and self.actuator_fault:
            return self.actuator_reason
        if len(self._observations) != len(self.SIDES):
            return None
        if self._any(lambda item: not item.numeric_valid):
            return "non-finite, malformed, or out-of-limit safety data"
        if self._any(lambda item: item.robot_fault):
            return "robot or joint fault reported"
        return None

    def _command_watchdog_expired(self, now_monotonic):
        if self._engaged_since is None:
            return False
        timestamps = [
            item.last_command_monotonic
            for item in self._observations.values()
        ]
        if len(timestamps) != len(self.SIDES) or any(
            value is None for value in timestamps
        ):
            reference = self._engaged_since
        else:
            reference = max(self._engaged_since, min(timestamps))
        return now_monotonic - reference > self.command_timeout_s

    def _joint_velocity_exceeded(self, observation):
        if observation.joint_velocity is None:
            return False
        velocity = np.asarray(observation.joint_velocity, dtype=float)
        limit = (
            self.joint_velocity_scale
            * self.collision_speed_scale
            * self._models[observation.side].qd_max
        )
        return bool(np.any(np.abs(velocity) > limit + 1e-12))

    def _joint_acceleration_exceeded(self, observation):
        limits = self._joint_acceleration_limits.get(observation.side)
        if limits is None or observation.joint_acceleration is None:
            return False
        acceleration = np.asarray(
            observation.joint_acceleration,
            dtype=float,
        )
        return bool(
            np.any(np.abs(acceleration) > limits + 1e-12)
        )

    def _soft_limit_reason(self):
        for side in self.SIDES:
            limits = self._joint_soft_limits.get(side)
            observation = self._observations.get(side)
            if limits is None or observation is None:
                continue
            lower, upper = limits
            for label, values in (
                ("measured", observation.q_measured),
                ("command", observation.q_command),
            ):
                if values is None:
                    continue
                q = np.asarray(values, dtype=float)
                if np.any(q < lower) or np.any(q > upper):
                    return f"{side} {label} joints outside teleop soft limits"
        return None

    def _collision_hold_reason(self):
        if not self.require_collision_safety:
            return None
        if not self.collision_ready or self.collision_hold_required:
            return self.collision_reason
        return None

    def _actuator_hold_reason(self):
        if not self.require_actuator_safety:
            return None
        if not self.actuator_ready or self.actuator_hold_required:
            return self.actuator_reason
        return None

    def _following_hold_reason(self):
        if not self.require_following_safety:
            return None
        if not self.following_ready or self.following_hold_required:
            return self.following_reason
        return None

    def _normalize_soft_limits(self, configured):
        if configured is None:
            return {}
        if set(configured) != set(self.SIDES):
            raise ValueError("joint_soft_limits must contain left and right")

        normalized = {}
        for side in self.SIDES:
            try:
                lower, upper = configured[side]
                lower = np.asarray(lower, dtype=float)
                upper = np.asarray(upper, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "joint soft limits must be numeric pairs"
                ) from exc
            model = self._models[side]
            if lower.shape != (model.DOF,) or upper.shape != (model.DOF,):
                raise ValueError("joint soft limits must contain 7-vectors")
            if (
                not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
            ):
                raise ValueError("joint soft limits must be finite")
            if np.any(lower >= upper):
                raise ValueError("joint soft-limit intervals must be nonempty")
            if np.any(lower <= model.q_min) or np.any(upper >= model.q_max):
                raise ValueError(
                    "joint soft limits must be inside hard limits"
                )
            normalized[side] = (lower.copy(), upper.copy())
        return normalized

    def _normalize_joint_acceleration_limits(self, configured):
        if configured is None:
            return {}
        if set(configured) != set(self.SIDES):
            raise ValueError(
                "joint_acceleration_limits must contain left and right"
            )

        normalized = {}
        for side in self.SIDES:
            try:
                limits = np.asarray(configured[side], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "joint acceleration limits must be numeric"
                ) from exc
            if limits.shape != (self._models[side].DOF,):
                raise ValueError(
                    "joint acceleration limits must contain 7-vectors"
                )
            if (
                not np.all(np.isfinite(limits))
                or np.any(limits <= 0.0)
            ):
                raise ValueError(
                    "joint acceleration limits must be finite and positive"
                )
            normalized[side] = limits.copy()
        return normalized

    def _all(self, predicate):
        if set(self._observations) != set(self.SIDES):
            return False
        return all(predicate(self._observations[side]) for side in self.SIDES)

    def _any(self, predicate):
        return any(predicate(item) for item in self._observations.values())

    def _transition(self, next_state, reason):
        next_state = SafetyState(next_state)
        if next_state == self.state:
            self.last_reason = str(reason)
            return
        if next_state not in self.ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(
                f"illegal safety transition {self.state.value} -> "
                f"{next_state.value}"
            )
        self.state = next_state
        self.last_reason = str(reason)

    @staticmethod
    def _normalize_q(q, model, required):
        if q is None:
            return None, not required
        try:
            array = np.asarray(q, dtype=float)
        except (TypeError, ValueError):
            return None, False
        valid = (
            array.shape == (model.DOF,)
            and np.all(np.isfinite(array))
            and np.all(array >= model.q_min)
            and np.all(array <= model.q_max)
        )
        if not valid:
            return None, False
        return tuple(float(value) for value in array), True

    @staticmethod
    def _normalize_finite_vector(values, size):
        if values is None:
            return None, True
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return None, False
        if array.shape != (size,) or not np.all(np.isfinite(array)):
            return None, False
        return tuple(float(value) for value in array), True
