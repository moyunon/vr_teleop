"""Timestamp-aware joint following-error monitor with persistence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class FollowingErrorState(str, Enum):
    """Commissioning regions for command-versus-feedback error."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    STOP = "STOP"


@dataclass(frozen=True)
class FollowingErrorDecision:
    """One following-error observation and persistent state."""

    state: FollowingErrorState
    ready: bool
    hold_required: bool
    error_rad: object
    max_abs_error_rad: object
    command_age_s: object
    measurement_age_s: object
    timestamp_skew_s: object
    reason: str


class FollowingErrorMonitor:
    """Reject stale/misaligned samples and debounce region transitions."""

    def __init__(
        self,
        warning_threshold_rad,
        stop_threshold_rad,
        persistence_s=0.10,
        hysteresis_ratio=0.8,
        max_age_s=0.10,
        max_timestamp_skew_s=0.10,
        continuous_joints=None,
    ):
        """Install provisional per-joint commissioning thresholds."""
        warning = np.asarray(warning_threshold_rad, dtype=float)
        stop = np.asarray(stop_threshold_rad, dtype=float)
        if warning.ndim != 1 or stop.shape != warning.shape:
            raise ValueError("following thresholds must be equal-size vectors")
        invalid_thresholds = any(
            (
                not np.all(np.isfinite(warning)),
                not np.all(np.isfinite(stop)),
                np.any(warning <= 0.0),
                np.any(stop <= warning),
            )
        )
        if invalid_thresholds:
            raise ValueError("require finite 0 < warning < stop thresholds")
        self.warning = warning
        self.stop = stop
        self.dof = len(warning)
        self.persistence_s = self._positive(persistence_s, "persistence_s")
        self.max_age_s = self._positive(max_age_s, "max_age_s")
        self.max_timestamp_skew_s = self._positive(
            max_timestamp_skew_s, "max_timestamp_skew_s"
        )
        self.hysteresis_ratio = float(hysteresis_ratio)
        if not 0.0 < self.hysteresis_ratio < 1.0:
            raise ValueError("hysteresis_ratio must be in (0, 1)")
        if continuous_joints is None:
            continuous_joints = np.zeros(self.dof, dtype=bool)
        self.continuous = np.asarray(continuous_joints, dtype=bool)
        if self.continuous.shape != (self.dof,):
            raise ValueError("continuous_joints shape mismatch")
        self.state = FollowingErrorState.NORMAL
        self._pending_state = None
        self._pending_since = None

    def reset(self):
        """Re-arm from NORMAL after the operator has released the deadman."""
        self.state = FollowingErrorState.NORMAL
        self._pending_state = None
        self._pending_since = None

    def evaluate(
        self,
        q_command,
        command_monotonic,
        q_measured,
        measurement_monotonic,
        now_monotonic,
    ):
        """Compare fresh samples whose timestamps are sufficiently close."""
        now = float(now_monotonic)
        command_time = float(command_monotonic)
        measurement_time = float(measurement_monotonic)
        timestamps = (now, command_time, measurement_time)
        if not all(math.isfinite(value) for value in timestamps):
            return self._unready("following-error timestamp is not finite")
        command_age = max(0.0, now - command_time)
        measurement_age = max(0.0, now - measurement_time)
        skew = abs(command_time - measurement_time)
        if command_age > self.max_age_s or measurement_age > self.max_age_s:
            return self._unready(
                "following-error command or measurement is stale",
                command_age,
                measurement_age,
                skew,
            )
        if skew > self.max_timestamp_skew_s:
            return self._unready(
                "following-error timestamps are not aligned",
                command_age,
                measurement_age,
                skew,
            )
        try:
            command = self._vector(q_command)
            measured = self._vector(q_measured)
        except (TypeError, ValueError) as exc:
            return self._unready(str(exc), command_age, measurement_age, skew)
        error = command - measured
        error[self.continuous] = (
            error[self.continuous] + math.pi
        ) % (2.0 * math.pi) - math.pi
        absolute = np.abs(error)
        target = self._target_state(absolute)
        if target == self.state:
            self._pending_state = None
            self._pending_since = None
        elif self._pending_state != target:
            self._pending_state = target
            self._pending_since = now
        elif now - self._pending_since >= self.persistence_s:
            self.state = target
            self._pending_state = None
            self._pending_since = None
        return FollowingErrorDecision(
            state=self.state,
            ready=True,
            hold_required=self.state == FollowingErrorState.STOP,
            error_rad=error.copy(),
            max_abs_error_rad=float(np.max(absolute)),
            command_age_s=command_age,
            measurement_age_s=measurement_age,
            timestamp_skew_s=skew,
            reason=(
                f"following error {self.state.value.lower()}: max "
                f"{np.max(absolute):.5f}rad"
            ),
        )

    def _target_state(self, absolute):
        warning_clear = self.warning * self.hysteresis_ratio
        stop_clear = self.stop * self.hysteresis_ratio
        if self.state == FollowingErrorState.STOP:
            if np.any(absolute >= stop_clear):
                return FollowingErrorState.STOP
            if np.any(absolute >= warning_clear):
                return FollowingErrorState.WARNING
            return FollowingErrorState.NORMAL
        if np.any(absolute >= self.stop):
            return FollowingErrorState.STOP
        if self.state == FollowingErrorState.WARNING:
            return (
                FollowingErrorState.WARNING
                if np.any(absolute >= warning_clear)
                else FollowingErrorState.NORMAL
            )
        return (
            FollowingErrorState.WARNING
            if np.any(absolute >= self.warning)
            else FollowingErrorState.NORMAL
        )

    def _unready(
        self, reason, command_age=None, measurement_age=None, skew=None
    ):
        self.state = FollowingErrorState.STOP
        self._pending_state = None
        self._pending_since = None
        return FollowingErrorDecision(
            state=self.state,
            ready=False,
            hold_required=True,
            error_rad=None,
            max_abs_error_rad=None,
            command_age_s=command_age,
            measurement_age_s=measurement_age,
            timestamp_skew_s=skew,
            reason=str(reason),
        )

    def _vector(self, values):
        result = np.asarray(values, dtype=float)
        if result.shape != (self.dof,) or not np.all(np.isfinite(result)):
            raise ValueError(
                f"joint vector must contain {self.dof} finite values"
            )
        return result

    @staticmethod
    def _positive(value, name):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value
