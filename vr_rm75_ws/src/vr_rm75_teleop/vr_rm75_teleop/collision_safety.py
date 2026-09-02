"""
Backend-independent collision-distance safety boundary.

The geometry backend is deliberately outside this module.  It must evaluate
one complete dual-arm candidate and provide all entries in ``SOURCES`` as one
atomic snapshot.  This keeps mesh/FCL/MoveIt choices out of the IK solver and
gives the control layer one deterministic safe/warning/stop contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Mapping, Optional, Tuple


class CollisionRegion(str, Enum):
    """Collision-distance regions used by the command safety gate."""

    UNKNOWN = "unknown"
    SAFE = "safe"
    WARNING = "warning"
    STOP = "stop"
    DISABLED = "disabled"


class CollisionSource(str, Enum):
    """Required distance classes for one complete dual-arm check."""

    LEFT_SELF = "left_self"
    RIGHT_SELF = "right_self"
    INTER_ARM = "inter_arm"
    ENVIRONMENT = "environment"
    ROBOT_BODY = "robot_body"


SOURCES: Tuple[CollisionSource, ...] = tuple(CollisionSource)


@dataclass(frozen=True)
class CollisionSnapshot:
    """One atomic set of minimum signed distances, expressed in metres."""

    distances_m: Tuple[float, ...]
    received_monotonic: float


@dataclass(frozen=True)
class CollisionSafetyDecision:
    """Result consumed by the Safety Supervisor and rate limiters."""

    region: CollisionRegion
    ready: bool
    hold_required: bool
    speed_scale: float
    min_distance_m: Optional[float]
    limiting_source: Optional[CollisionSource]
    age_s: Optional[float]
    reason: str


def disabled_collision_decision():
    """Return the explicit RViz-only bypass state."""
    return CollisionSafetyDecision(
        region=CollisionRegion.DISABLED,
        ready=True,
        hold_required=False,
        speed_scale=1.0,
        min_distance_m=None,
        limiting_source=None,
        age_s=None,
        reason="collision protection explicitly disabled",
    )


class CollisionSafetyMonitor:
    """Validate atomic distance reports and apply a fail-closed watchdog."""

    def __init__(
        self,
        d_stop_m,
        d_warn_m,
        timeout_s,
        monotonic=time.monotonic,
    ):
        """Configure positive stop/warning thresholds and report timeout."""
        self.d_stop_m = self._positive_finite("d_stop_m", d_stop_m)
        self.d_warn_m = self._positive_finite("d_warn_m", d_warn_m)
        self.timeout_s = self._positive_finite("timeout_s", timeout_s)
        if self.d_stop_m >= self.d_warn_m:
            raise ValueError("d_stop_m must be strictly smaller than d_warn_m")
        self._monotonic = monotonic
        self._snapshot: Optional[CollisionSnapshot] = None
        self._invalid_reason: Optional[str] = None

    def update_snapshot(self, distances_m, received_monotonic=None):
        """Atomically accept all required minimum signed distances."""
        if received_monotonic is None:
            received_monotonic = self._monotonic()

        try:
            received_monotonic = float(received_monotonic)
            if not math.isfinite(received_monotonic):
                raise ValueError("received_monotonic must be finite")
            normalized = self._normalize_distances(distances_m)
        except (KeyError, TypeError, ValueError) as exc:
            self.reject_snapshot(exc)
            raise

        self._snapshot = CollisionSnapshot(
            distances_m=normalized,
            received_monotonic=received_monotonic,
        )
        self._invalid_reason = None
        return self._snapshot

    def reject_snapshot(self, reason):
        """Invalidate the complete report; an old safe report is not reused."""
        self._snapshot = None
        self._invalid_reason = str(reason)

    def evaluate(self, now_monotonic=None):
        """Return safe/warning/stop or fail closed on missing/stale input."""
        if now_monotonic is None:
            now_monotonic = self._monotonic()
        now_monotonic = float(now_monotonic)
        if not math.isfinite(now_monotonic):
            raise ValueError("now_monotonic must be finite")

        if self._snapshot is None:
            detail = self._invalid_reason or "no complete collision snapshot"
            return self._unknown_decision(
                f"collision distance unavailable: {detail}"
            )

        age_s = max(
            0.0,
            now_monotonic - self._snapshot.received_monotonic,
        )
        if age_s > self.timeout_s:
            return self._unknown_decision(
                f"collision distance stale ({age_s:.3f}s)",
                age_s=age_s,
            )

        min_index = min(
            range(len(SOURCES)),
            key=self._snapshot.distances_m.__getitem__,
        )
        min_distance_m = self._snapshot.distances_m[min_index]
        limiting_source = SOURCES[min_index]

        if min_distance_m <= self.d_stop_m:
            return CollisionSafetyDecision(
                region=CollisionRegion.STOP,
                ready=True,
                hold_required=True,
                speed_scale=0.0,
                min_distance_m=min_distance_m,
                limiting_source=limiting_source,
                age_s=age_s,
                reason=(
                    f"collision stop: {limiting_source.value} distance "
                    f"{min_distance_m:.4f}m <= {self.d_stop_m:.4f}m"
                ),
            )

        if min_distance_m <= self.d_warn_m:
            ratio = (
                (min_distance_m - self.d_stop_m)
                / (self.d_warn_m - self.d_stop_m)
            )
            speed_scale = ratio * ratio * (3.0 - 2.0 * ratio)
            return CollisionSafetyDecision(
                region=CollisionRegion.WARNING,
                ready=True,
                hold_required=False,
                speed_scale=float(speed_scale),
                min_distance_m=min_distance_m,
                limiting_source=limiting_source,
                age_s=age_s,
                reason=(
                    f"collision warning: {limiting_source.value} distance "
                    f"{min_distance_m:.4f}m"
                ),
            )

        return CollisionSafetyDecision(
            region=CollisionRegion.SAFE,
            ready=True,
            hold_required=False,
            speed_scale=1.0,
            min_distance_m=min_distance_m,
            limiting_source=limiting_source,
            age_s=age_s,
            reason=(
                f"collision clear: {limiting_source.value} distance "
                f"{min_distance_m:.4f}m"
            ),
        )

    @staticmethod
    def _positive_finite(name, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    @staticmethod
    def _normalize_distances(distances_m):
        if not isinstance(distances_m, Mapping):
            raise TypeError("distances_m must be a mapping")

        normalized = {}
        for key, value in distances_m.items():
            try:
                source = CollisionSource(key)
            except ValueError as exc:
                raise ValueError(f"unknown collision source {key!r}") from exc
            if source in normalized:
                raise ValueError(f"duplicate collision source {source.value}")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"{source.value} collision distance must be finite"
                )
            normalized[source] = value

        missing = [source.value for source in SOURCES if source not in normalized]
        if missing:
            raise ValueError(
                "collision snapshot missing sources: " + ", ".join(missing)
            )
        return tuple(normalized[source] for source in SOURCES)

    @staticmethod
    def _unknown_decision(reason, age_s=None):
        return CollisionSafetyDecision(
            region=CollisionRegion.UNKNOWN,
            ready=False,
            hold_required=True,
            speed_scale=0.0,
            min_distance_m=None,
            limiting_source=None,
            age_s=age_s,
            reason=str(reason),
        )
