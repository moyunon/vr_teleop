"""Measured joint velocity/acceleration estimation using actual sample time."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MeasuredKinematics:
    """One validated feedback-derivative result and its provenance."""

    qdot: object
    qddot: object
    dt_s: object
    velocity_source: str
    valid: bool
    reason: str


class RobotFeedbackMonitor:
    """Prefer direct qdot and derive filtered qddot with measured dt."""

    def __init__(self, dof=7, acceleration_filter_tau_s=0.05, max_dt_s=0.25):
        """Configure a first-order acceleration low-pass filter."""
        self.dof = int(dof)
        self.filter_tau_s = float(acceleration_filter_tau_s)
        self.max_dt_s = float(max_dt_s)
        if self.dof < 1:
            raise ValueError("dof must be positive")
        if not math.isfinite(self.filter_tau_s) or self.filter_tau_s < 0.0:
            raise ValueError("filter tau must be finite and nonnegative")
        if not math.isfinite(self.max_dt_s) or self.max_dt_s <= 0.0:
            raise ValueError("max_dt_s must be finite and positive")
        self._previous_q = None
        self._previous_qdot = None
        self._previous_time = None
        self._filtered_qddot = None

    def update(self, q, measured_monotonic, direct_qdot=None):
        """Process one unique sample; never assume the nominal loop period."""
        q = self._vector(q, "q")
        measured_monotonic = float(measured_monotonic)
        if not math.isfinite(measured_monotonic):
            raise ValueError("measurement time must be finite")
        direct = None
        if direct_qdot is not None:
            direct = self._vector(direct_qdot, "direct_qdot")

        if self._previous_time is None:
            qdot = direct
            source = "udp_direct" if direct is not None else "unavailable"
            result = MeasuredKinematics(
                qdot=None if qdot is None else qdot.copy(),
                qddot=None,
                dt_s=None,
                velocity_source=source,
                valid=qdot is not None,
                reason=(
                    "direct velocity accepted; acceleration history pending"
                    if qdot is not None
                    else "velocity history pending"
                ),
            )
            self._store(q, qdot, measured_monotonic)
            return result

        dt_s = measured_monotonic - self._previous_time
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            return MeasuredKinematics(
                qdot=None,
                qddot=None,
                dt_s=dt_s,
                velocity_source="rejected",
                valid=False,
                reason="duplicate or non-advancing measurement time",
            )
        if dt_s > self.max_dt_s:
            self._previous_q = q.copy()
            self._previous_qdot = direct.copy() if direct is not None else None
            self._previous_time = measured_monotonic
            self._filtered_qddot = None
            return MeasuredKinematics(
                qdot=None if direct is None else direct.copy(),
                qddot=None,
                dt_s=dt_s,
                velocity_source=(
                    "udp_direct" if direct is not None else "rejected"
                ),
                valid=False,
                reason=f"measurement gap {dt_s:.3f}s exceeds limit",
            )

        if direct is not None:
            qdot = direct
            source = "udp_direct"
        else:
            qdot = (q - self._previous_q) / dt_s
            source = "finite_difference"

        qddot = None
        if self._previous_qdot is not None:
            raw_qddot = (qdot - self._previous_qdot) / dt_s
            alpha = (
                1.0
                if self.filter_tau_s == 0.0
                else dt_s / (self.filter_tau_s + dt_s)
            )
            if self._filtered_qddot is None:
                self._filtered_qddot = raw_qddot
            else:
                self._filtered_qddot = np.add(
                    self._filtered_qddot,
                    alpha * (raw_qddot - self._filtered_qddot),
                )
            qddot = self._filtered_qddot.copy()
        self._store(q, qdot, measured_monotonic)
        return MeasuredKinematics(
            qdot=qdot.copy(),
            qddot=qddot,
            dt_s=dt_s,
            velocity_source=source,
            valid=True,
            reason=(
                f"{source}; filtered acceleration available"
                if qddot is not None
                else f"{source}; acceleration history pending"
            ),
        )

    def _store(self, q, qdot, timestamp):
        self._previous_q = q.copy()
        self._previous_qdot = None if qdot is None else qdot.copy()
        self._previous_time = timestamp

    def _vector(self, value, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (self.dof,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {self.dof} finite values")
        return result
