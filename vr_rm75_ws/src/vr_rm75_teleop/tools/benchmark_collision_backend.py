#!/usr/bin/env python3
"""Benchmark the offline RM75-only collision backend at one static q."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import yaml

from vr_rm75_teleop.collision_backend import (
    FclDistanceEngine,
    FiveClassCollisionBackend,
    UrdfCollisionModel,
    enabled_sources_from_config,
    environment_geometry_from_config,
)
from vr_rm75_teleop.collision_safety import CollisionSafetyMonitor


class ExhaustiveDistanceEngine:
    """Expose only pair distance so the backend cannot apply AABB pruning."""

    def __init__(self, engine):
        """Wrap an initialized FCL engine without prepared-query methods."""
        self._engine = engine
        self.available = engine.available
        self.unavailable_reason = engine.unavailable_reason

    def distance(self, first, first_transform, second, second_transform):
        """Delegate one exact FCL pair query."""
        return self._engine.distance(
            first,
            first_transform,
            second,
            second_transform,
        )


def parse_arguments():
    """Parse explicit geometry and static joint-position inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--description-root", type=Path, required=True)
    parser.add_argument("--left-q-deg", type=float, nargs=7, required=True)
    parser.add_argument("--right-q-deg", type=float, nargs=7, required=True)
    parser.add_argument("--warm-runs", type=int, default=100)
    parser.add_argument("--verify-exhaustive", action="store_true")
    return parser.parse_args()


def percentile(values, quantile):
    """Return one percentile as a built-in float for JSON serialization."""
    return float(np.percentile(values, quantile))


def category_payload(snapshot):
    """Serialize each category's exact minimum and FCL witness points."""
    return {
        result.source.value: {
            "distance_m": result.distance_m,
            "closest_pair": result.closest_pair,
            "closest_points": result.closest_points,
        }
        for result in snapshot.category_results
    }


def main():
    """Run one cold construction/evaluation and repeated warm evaluations."""
    arguments = parse_arguments()
    if arguments.warm_runs < 100:
        raise ValueError("--warm-runs must be at least 100")
    with arguments.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    q_deg = arguments.left_q_deg + arguments.right_q_deg
    q_rad = np.deg2rad(q_deg)
    joints = {
        f"{side}_rm75_joint_{index}": float(q_rad[offset + index - 1])
        for side, offset in (("l", 0), ("r", 7))
        for index in range(1, 8)
    }

    cold_started = time.perf_counter()
    model = UrdfCollisionModel(
        arguments.urdf,
        package_roots={
            "lsrx_rm75_dual_description": arguments.description_root,
        },
    )
    environment = environment_geometry_from_config(
        config.get("environment", ()),
        base_directory=arguments.config.parent,
    )
    engine = FclDistanceEngine()
    backend = FiveClassCollisionBackend(
        model,
        engine,
        environment=environment,
        ignored_pairs=config.get("ignored_collision_pairs", ()),
        enabled_sources=enabled_sources_from_config(
            config.get("category_enabled")
        ),
        monitored_links=config.get("monitored_links"),
    )
    initialized = time.perf_counter()
    if not backend.ready:
        raise RuntimeError(backend.readiness_reason)
    cold_snapshot = backend.evaluate(joints, measured_monotonic=0.0)
    cold_finished = time.perf_counter()

    warm_ms = []
    snapshot = cold_snapshot
    for _ in range(arguments.warm_runs):
        started = time.perf_counter()
        snapshot = backend.evaluate(joints, measured_monotonic=0.0)
        warm_ms.append((time.perf_counter() - started) * 1000.0)

    monitor = CollisionSafetyMonitor(
        d_stop_m=0.05,
        d_warn_m=0.15,
        timeout_s=0.10,
        enabled_sources=backend.enabled_sources,
    )
    monitor.update_snapshot(
        dict(zip(backend.enabled_sources, snapshot.distances_m)),
        received_monotonic=0.0,
    )
    decision = asdict(monitor.evaluate(now_monotonic=0.0))
    decision["region"] = decision["region"].value
    if decision["limiting_source"] is not None:
        decision["limiting_source"] = decision["limiting_source"].value

    output = {
        "geometry_count_in_urdf": len(model.geometries),
        "prepared_collision_objects": len(engine._objects),
        "cold": {
            "initialization_and_preload_ms": (
                initialized - cold_started
            ) * 1000.0,
            "fcl_preload_ms": backend.preload_ms,
            "first_evaluation_ms": (
                cold_finished - initialized
            ) * 1000.0,
            "end_to_end_ms": (cold_finished - cold_started) * 1000.0,
        },
        "warm": {
            "runs": len(warm_ms),
            "mean_ms": float(np.mean(warm_ms)),
            "p50_ms": percentile(warm_ms, 50),
            "p95_ms": percentile(warm_ms, 95),
            "p99_ms": percentile(warm_ms, 99),
            "max_ms": max(warm_ms),
        },
        "categories": category_payload(snapshot),
        "global_closest": {
            "category": snapshot.closest_category.value,
            "pair": snapshot.closest_pair,
            "points": snapshot.closest_points,
        },
        "collision_safety_decision": decision,
    }
    if arguments.verify_exhaustive:
        exhaustive_backend = FiveClassCollisionBackend(
            model,
            ExhaustiveDistanceEngine(engine),
            environment=environment,
            ignored_pairs=config.get("ignored_collision_pairs", ()),
            enabled_sources=backend.enabled_sources,
            monitored_links=config.get("monitored_links"),
        )
        exhaustive = exhaustive_backend.evaluate(
            joints, measured_monotonic=0.0
        )
        errors = {
            source.value: abs(pruned - reference)
            for source, pruned, reference in zip(
                backend.enabled_sources,
                snapshot.distances_m,
                exhaustive.distances_m,
            )
        }
        if max(errors.values()) > 1e-9:
            raise RuntimeError(f"AABB/exhaustive mismatch: {errors}")
        output["exhaustive_validation"] = {
            "distance_absolute_error_m": errors,
            "categories": category_payload(exhaustive),
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
