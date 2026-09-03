"""
Backend-independent collision-distance safety boundary.

The geometry backend is deliberately outside this module.  It must evaluate
one complete dual-arm candidate and provide every *enabled* entry in
``SOURCES`` as one atomic snapshot.  This keeps mesh/FCL/MoveIt choices out of
the IK solver and gives the control layer one deterministic
safe/warning/stop contract while preserving all five categories for later
commissioning.
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


def normalize_collision_sources(sources):
    """Validate a configured subset and return canonical five-class order."""
    if sources is None:
        return SOURCES
    if isinstance(sources, (str, bytes)):
        raise TypeError("collision sources must be a sequence")

    normalized = []
    seen = set()
    for value in sources:
        try:
            source = CollisionSource(value)
        except ValueError as exc:
            raise ValueError(f"unknown collision source {value!r}") from exc
        if source in seen:
            raise ValueError(f"duplicate collision source {source.value}")
        seen.add(source)
        normalized.append(source)
    if not normalized:
        raise ValueError("at least one collision source must be enabled")
    return tuple(source for source in SOURCES if source in seen)


@dataclass(frozen=True)
class CollisionSnapshot:
    """One atomic set of minimum signed distances, expressed in metres."""

    distances_m: Tuple[float, ...]
    received_monotonic: float


@dataclass(frozen=True)
class CollisionThresholds:
    """Stop and warning clearances for one collision category."""

    stop_m: float
    warn_m: float


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
    stop_distance_m: Optional[float] = None
    warn_distance_m: Optional[float] = None


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
        enabled_sources=None,
        thresholds_by_source=None,
    ):
        """Configure default/per-category thresholds and report timeout."""
        self.d_stop_m = self._positive_finite("d_stop_m", d_stop_m)
        self.d_warn_m = self._positive_finite("d_warn_m", d_warn_m)
        self.timeout_s = self._positive_finite("timeout_s", timeout_s)
        if self.d_stop_m >= self.d_warn_m:
            raise ValueError("d_stop_m must be strictly smaller than d_warn_m")
        self.enabled_sources = normalize_collision_sources(enabled_sources)
        self.thresholds_by_source = self._normalize_thresholds(
            thresholds_by_source
        )
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

        evaluations = []
        for index, (source, distance_m) in enumerate(
            zip(self.enabled_sources, self._snapshot.distances_m)
        ):
            thresholds = self.thresholds_by_source[source]
            normalized_clearance = (
                (distance_m - thresholds.stop_m)
                / (thresholds.warn_m - thresholds.stop_m)
            )
            if distance_m <= thresholds.stop_m:
                region = CollisionRegion.STOP
                speed_scale = 0.0
            elif distance_m <= thresholds.warn_m:
                region = CollisionRegion.WARNING
                ratio = normalized_clearance
                speed_scale = ratio * ratio * (3.0 - 2.0 * ratio)
            else:
                region = CollisionRegion.SAFE
                speed_scale = 1.0
            evaluations.append(
                (
                    float(speed_scale),
                    float(normalized_clearance),
                    float(distance_m),
                    index,
                    source,
                    thresholds,
                    region,
                )
            )

        (
            speed_scale,
            _,
            min_distance_m,
            _,
            limiting_source,
            limiting_thresholds,
            limiting_region,
        ) = min(evaluations)

        if limiting_region == CollisionRegion.STOP:
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
                    f"{min_distance_m:.4f}m <= "
                    f"{limiting_thresholds.stop_m:.4f}m"
                ),
                stop_distance_m=limiting_thresholds.stop_m,
                warn_distance_m=limiting_thresholds.warn_m,
            )

        if limiting_region == CollisionRegion.WARNING:
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
                    f"{min_distance_m:.4f}m; thresholds stop="
                    f"{limiting_thresholds.stop_m:.4f}m warn="
                    f"{limiting_thresholds.warn_m:.4f}m"
                ),
                stop_distance_m=limiting_thresholds.stop_m,
                warn_distance_m=limiting_thresholds.warn_m,
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
            stop_distance_m=limiting_thresholds.stop_m,
            warn_distance_m=limiting_thresholds.warn_m,
        )

    def category_diagnostics(self):
        """Report resolved thresholds and latest state for all categories."""
        distances = {}
        if self._snapshot is not None:
            distances = dict(
                zip(self.enabled_sources, self._snapshot.distances_m)
            )

        diagnostics = {}
        for source in SOURCES:
            if source not in self.enabled_sources:
                diagnostics[source.value] = {
                    "status": "DISABLED_BY_CONFIGURATION",
                    "stop_distance_m": None,
                    "warn_distance_m": None,
                    "distance_m": None,
                    "region": CollisionRegion.DISABLED.value,
                }
                continue
            thresholds = self.thresholds_by_source[source]
            distance_m = distances.get(source)
            region = CollisionRegion.UNKNOWN
            if distance_m is not None:
                if distance_m <= thresholds.stop_m:
                    region = CollisionRegion.STOP
                elif distance_m <= thresholds.warn_m:
                    region = CollisionRegion.WARNING
                else:
                    region = CollisionRegion.SAFE
            diagnostics[source.value] = {
                "status": "ENABLED",
                "stop_distance_m": thresholds.stop_m,
                "warn_distance_m": thresholds.warn_m,
                "distance_m": distance_m,
                "region": region.value,
            }
        return diagnostics

    @staticmethod
    def _positive_finite(name, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def _normalize_thresholds(self, thresholds_by_source):
        overrides = {}
        if thresholds_by_source is not None:
            if not isinstance(thresholds_by_source, Mapping):
                raise TypeError("thresholds_by_source must be a mapping")
            for key, values in thresholds_by_source.items():
                try:
                    source = CollisionSource(key)
                except ValueError as exc:
                    raise ValueError(
                        f"unknown collision threshold source {key!r}"
                    ) from exc
                if isinstance(values, CollisionThresholds):
                    stop_m, warn_m = values.stop_m, values.warn_m
                else:
                    try:
                        stop_m, warn_m = values
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{source.value} thresholds require (stop, warn)"
                        ) from exc
                stop_m = self._positive_finite(
                    f"{source.value} stop distance", stop_m
                )
                warn_m = self._positive_finite(
                    f"{source.value} warn distance", warn_m
                )
                if stop_m >= warn_m:
                    raise ValueError(
                        f"{source.value} stop distance must be strictly "
                        "smaller than warn distance"
                    )
                overrides[source] = CollisionThresholds(stop_m, warn_m)

        return {
            source: overrides.get(
                source,
                CollisionThresholds(self.d_stop_m, self.d_warn_m),
            )
            for source in self.enabled_sources
        }

    def _normalize_distances(self, distances_m):
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
            if source not in self.enabled_sources:
                # Disabled categories are deliberately outside this safety
                # decision.  Do not require, validate, or substitute a value.
                continue
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"{source.value} collision distance must be finite"
                )
            normalized[source] = value

        missing = [
            source.value
            for source in self.enabled_sources
            if source not in normalized
        ]
        if missing:
            raise ValueError(
                "collision snapshot missing sources: " + ", ".join(missing)
            )
        return tuple(normalized[source] for source in self.enabled_sources)

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
