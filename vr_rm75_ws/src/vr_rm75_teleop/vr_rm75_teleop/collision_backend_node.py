"""ROS 2 adapter for the fail-closed five-class FCL backend."""

from __future__ import annotations

import json
import os
import time

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String
import yaml

from vr_rm75_teleop.collision_backend import (
    CollisionBackendError,
    FclDistanceEngine,
    FiveClassCollisionBackend,
    UrdfCollisionModel,
    environment_geometry_from_config,
)
from vr_rm75_teleop.collision_safety import SOURCES


class CollisionBackendNode(Node):
    """Publish snapshots only while every quality gate passes."""

    def __init__(self):
        """Load declared geometry without inventing missing dimensions."""
        super().__init__("collision_backend")
        description_share = get_package_share_directory(
            "lsrx_rm75_dual_description"
        )
        teleop_share = get_package_share_directory("vr_rm75_teleop")
        self.declare_parameter(
            "urdf_path",
            os.path.join(description_share, "urdf", "LSRX_RM75_DUAL.urdf"),
        )
        self.declare_parameter(
            "geometry_config",
            os.path.join(teleop_share, "config", "collision_geometry.yaml"),
        )
        self.declare_parameter("state_timeout_s", 0.10)
        self.state_timeout_s = float(
            self.get_parameter("state_timeout_s").value
        )
        if self.state_timeout_s <= 0.0:
            raise ValueError("state_timeout_s must be positive")

        self.distance_publisher = self.create_publisher(
            Float64MultiArray,
            "/vr_rm75/collision/min_distances_m",
            10,
        )
        self.ready_publisher = self.create_publisher(
            Bool, "/vr_rm75/collision/backend_ready", 10
        )
        self.diagnostics_publisher = self.create_publisher(
            String, "/vr_rm75/collision/backend_diagnostics", 10
        )
        self._backend = None
        self._configuration_joint_positions = {}
        self._last_state_monotonic = None
        self._last_reason = "collision backend is initializing"
        self._last_snapshot = None

        try:
            config_path = str(self.get_parameter("geometry_config").value)
            with open(config_path, "r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream) or {}
            model = UrdfCollisionModel(
                str(self.get_parameter("urdf_path").value),
                package_roots={
                    "lsrx_rm75_dual_description": description_share,
                },
            )
            environment = environment_geometry_from_config(
                config.get("environment", ()),
                base_directory=os.path.dirname(config_path),
            )
            self._configuration_joint_positions = {
                str(name): float(value)
                for name, value in config.get(
                    "fixed_or_externally_verified_joint_positions", {}
                ).items()
            }
            self._backend = FiveClassCollisionBackend(
                model,
                FclDistanceEngine(),
                environment=environment,
                ignored_pairs=config.get("ignored_collision_pairs", ()),
                require_complete_geometry=True,
                require_environment=True,
            )
            self._last_reason = self._backend.readiness_reason
        except (CollisionBackendError, OSError, TypeError, ValueError) as exc:
            self._last_reason = (
                f"collision backend configuration invalid: {exc}"
            )

        self.create_subscription(
            JointState,
            "/rm75/actual_joint_states",
            self.state_callback,
            10,
        )
        self.create_timer(0.1, self.watchdog_callback)
        self.get_logger().warning(self._last_reason)

    def state_callback(self, message):
        """Compute from one coherent measured dual-arm JointState sample."""
        now = time.perf_counter()
        self._last_state_monotonic = now
        if self._backend is None or not self._backend.ready:
            self.publish_diagnostics(now)
            return
        names_aligned = len(message.name) == len(message.position)
        names_unique = len(set(message.name)) == len(message.name)
        if not names_aligned or not names_unique:
            self._last_reason = "invalid JointState name/position alignment"
            self.publish_diagnostics(now)
            return
        joint_positions = dict(self._configuration_joint_positions)
        joint_positions.update(
            {
                name: value
                for name, value in zip(message.name, message.position)
            }
        )
        try:
            snapshot = self._backend.evaluate(
                joint_positions, measured_monotonic=now
            )
        except (CollisionBackendError, TypeError, ValueError) as exc:
            self._last_snapshot = None
            self._last_reason = f"collision snapshot rejected: {exc}"
            self.publish_diagnostics(now)
            return
        self._last_snapshot = snapshot
        self._last_reason = snapshot.reason
        self.distance_publisher.publish(
            Float64MultiArray(data=list(snapshot.distances_m))
        )
        self.publish_diagnostics(now)

    def watchdog_callback(self):
        """Report stale/missing input; never republish an old safe snapshot."""
        now = time.perf_counter()
        state_stale = self._last_state_monotonic is None
        if self._last_state_monotonic is not None:
            state_stale = (
                now - self._last_state_monotonic > self.state_timeout_s
            )
        if state_stale:
            self._last_snapshot = None
            self._last_reason = "measured joint state missing or stale"
        self.publish_diagnostics(now)

    def publish_diagnostics(self, now):
        """Publish readiness, age, closest pair, and measured solve timing."""
        backend_ready = bool(self._backend is not None and self._backend.ready)
        snapshot = self._last_snapshot
        snapshot_fresh = False
        if snapshot is not None:
            snapshot_fresh = (
                now - snapshot.measured_monotonic <= self.state_timeout_s
            )
        sample_ready = bool(backend_ready and snapshot_fresh)
        self.ready_publisher.publish(Bool(data=sample_ready))
        payload = {
            "valid": bool(snapshot is not None and snapshot.valid),
            "ready": sample_ready,
            "geometry_backend_ready": backend_ready,
            "age_s": (
                None
                if snapshot is None
                else max(0.0, now - snapshot.measured_monotonic)
            ),
            "source_order": [source.value for source in SOURCES],
            "distances_m": (
                None if snapshot is None else list(snapshot.distances_m)
            ),
            "closest_category": (
                None
                if snapshot is None
                else snapshot.closest_category.value
            ),
            "closest_pair": (
                None if snapshot is None else snapshot.closest_pair
            ),
            "solve_ms": None if snapshot is None else snapshot.solve_ms,
            "mean_solve_ms": (
                None if snapshot is None else snapshot.mean_solve_ms
            ),
            "max_solve_ms": (
                None if snapshot is None else snapshot.max_solve_ms
            ),
            "reason": self._last_reason,
        }
        self.diagnostics_publisher.publish(
            String(data=json.dumps(payload, sort_keys=True))
        )


def main(args=None):
    """Run the collision backend ROS adapter."""
    rclpy.init(args=args)
    node = CollisionBackendNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
