"""Fail-closed dual-grip deadman logic for Quest teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DeadmanDecision:
    """Immutable aggregate result of one clutch evaluation."""

    active: bool
    changed: bool
    rearm_required: bool
    reason: str
    grip_values: Tuple[Optional[float], Optional[float]]


class DualGripDeadman:
    """Require two fresh analog grips with hysteresis and release-to-rearm."""

    SIDES = ("left", "right")

    def __init__(
        self,
        on_threshold=0.65,
        off_threshold=0.35,
        input_timeout_s=0.20,
        monotonic=time.monotonic,
    ):
        """Configure conservative thresholds without activating the clutch."""
        self.on_threshold = self._finite_float(
            on_threshold, "on_threshold"
        )
        self.off_threshold = self._finite_float(
            off_threshold, "off_threshold"
        )
        self.input_timeout_s = self._finite_float(
            input_timeout_s, "input_timeout_s"
        )
        if not 0.0 <= self.off_threshold < self.on_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= off < on <= 1"
            )
        if self.input_timeout_s <= 0.0:
            raise ValueError("input_timeout_s must be positive")

        self._monotonic = monotonic
        self._values: Dict[str, Optional[float]] = {
            side: None for side in self.SIDES
        }
        self._received: Dict[str, Optional[float]] = {
            side: None for side in self.SIDES
        }
        self._sample_valid = {side: False for side in self.SIDES}
        self._pressed = {side: False for side in self.SIDES}
        self.active = False
        self.rearm_required = True
        self.last_reason = "waiting for fresh released dual-grip input"

    def update_grip(
        self,
        side,
        value,
        now_monotonic=None,
        source_fresh=True,
    ):
        """Store one analog sample and evaluate the aggregate deadman."""
        side = str(side).lower()
        if side not in self.SIDES:
            raise ValueError("side must be 'left' or 'right'")
        now_monotonic = self._valid_time(now_monotonic)

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = math.nan

        valid = math.isfinite(value) and 0.0 <= value <= 1.0
        self._received[side] = now_monotonic
        self._sample_valid[side] = valid
        self._values[side] = value if valid else None

        if not valid:
            self._pressed[side] = False
            self._disarm("invalid grip sample")
        elif value >= self.on_threshold:
            self._pressed[side] = True
        elif value <= self.off_threshold:
            self._pressed[side] = False

        return self.evaluate(now_monotonic, source_fresh=source_fresh)

    def evaluate(
        self,
        now_monotonic=None,
        source_fresh=True,
    ):
        """Apply source and per-hand watchdogs, then compute dual engagement."""
        now_monotonic = self._valid_time(now_monotonic)
        previous = self.active

        if not bool(source_fresh):
            self._disarm("Quest input source stale")
        elif not all(self._sample_valid.values()):
            self._disarm("waiting for valid dual-grip samples")
        elif not self._all_samples_fresh(now_monotonic):
            self._disarm("dual-grip input timeout")
        elif self.rearm_required:
            if all(
                self._values[side] <= self.off_threshold
                for side in self.SIDES
            ):
                self.rearm_required = False
                self.last_reason = "dual grips released; ready for next press"
            else:
                self.last_reason = "release both grips before re-engaging"
            self.active = False
        else:
            self.active = all(self._pressed.values())
            if self.active:
                self.last_reason = "both grips continuously active"
            elif previous:
                self.last_reason = "one or both grips released"
            else:
                self.last_reason = "waiting for both grips"

        return DeadmanDecision(
            active=self.active,
            changed=self.active != previous,
            rearm_required=self.rearm_required,
            reason=self.last_reason,
            grip_values=(self._values["left"], self._values["right"]),
        )

    def invalidate(self, reason="Quest input invalid"):
        """Immediately release and require a fresh release/press sequence."""
        previous = self.active
        self._disarm(str(reason))
        return DeadmanDecision(
            active=False,
            changed=previous,
            rearm_required=True,
            reason=self.last_reason,
            grip_values=(self._values["left"], self._values["right"]),
        )

    def _all_samples_fresh(self, now_monotonic):
        return all(
            self._received[side] is not None
            and 0.0 <= now_monotonic - self._received[side]
            <= self.input_timeout_s
            for side in self.SIDES
        )

    def _disarm(self, reason):
        self.active = False
        self.rearm_required = True
        self.last_reason = str(reason)

    def _valid_time(self, value):
        if value is None:
            value = self._monotonic()
        return self._finite_float(value, "now_monotonic")

    @staticmethod
    def _finite_float(value, name):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
