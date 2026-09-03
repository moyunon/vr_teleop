"""Dependency-free rolling latency, period, jitter, and deadline statistics."""

from __future__ import annotations

from collections import deque
import math

import numpy as np


class TimingMonitor:
    """Collect bounded-window stage durations and control-loop periods."""

    def __init__(self, nominal_period_s=0.02, window_size=500):
        """Configure the deadline and rolling sample count."""
        self.nominal_period_s = float(nominal_period_s)
        self.window_size = int(window_size)
        invalid_period = not math.isfinite(self.nominal_period_s)
        invalid_period = invalid_period or self.nominal_period_s <= 0
        if invalid_period:
            raise ValueError("nominal_period_s must be finite and positive")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._values = {}
        self._last_cycle_start = None
        self.deadline_miss_count = 0
        self.cycle_count = 0

    def record(self, name, duration_s):
        """Record one finite nonnegative stage duration."""
        value = float(duration_s)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("timing duration must be finite and nonnegative")
        self._values.setdefault(
            str(name), deque(maxlen=self.window_size)
        ).append(value)

    def begin_cycle(self, started_monotonic):
        """Record actual period, jitter, frequency, and deadline misses."""
        started = float(started_monotonic)
        if not math.isfinite(started):
            raise ValueError("cycle timestamp must be finite")
        if self._last_cycle_start is not None:
            period = started - self._last_cycle_start
            if period > 0.0 and math.isfinite(period):
                self.record("control_period", period)
                self.record(
                    "control_jitter",
                    abs(period - self.nominal_period_s),
                )
                if period > self.nominal_period_s:
                    self.deadline_miss_count += 1
        self._last_cycle_start = started
        self.cycle_count += 1

    def summary(self):
        """Return mean/max/p95/p99 plus measured loop-level counters."""
        result = {
            name: self._summarize(values)
            for name, values in sorted(self._values.items())
        }
        periods = self._values.get("control_period", ())
        result["control"] = {
            "cycle_count": self.cycle_count,
            "deadline_miss_count": self.deadline_miss_count,
            "effective_frequency_hz": (
                None if not periods else 1.0 / float(np.mean(periods))
            ),
            "nominal_period_s": self.nominal_period_s,
        }
        return result

    @staticmethod
    def _summarize(values):
        array = np.asarray(values, dtype=float)
        return {
            "count": int(len(array)),
            "mean_s": float(np.mean(array)),
            "max_s": float(np.max(array)),
            "p95_s": float(np.percentile(array, 95)),
            "p99_s": float(np.percentile(array, 99)),
        }
