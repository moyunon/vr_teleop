#!/usr/bin/env python3
"""ROS 2 publisher for validated, read-only dual-RM75 actual state."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from vr_rm75_teleop.rm75_hardware_interface import (
    GEN4_JOINT_SPEED_TO_RAD_S,
    RM75ArmState,
    RM75StateStatus,
    RM75StateWorker,
    RM75UDPStateReceiver,
)


class RM75StateNode(Node):
    """Read two controllers and publish actual state without actuation APIs."""

    def __init__(self) -> None:
        """Create publishers and start independent left/right state readers."""
        super().__init__("rm75_state_node")

        self.declare_parameter("left_ip", "192.168.127.18")
        self.declare_parameter("right_ip", "192.168.127.19")
        self.declare_parameter("tcp_port", 8080)
        self.declare_parameter("tcp_timeout_s", 0.25)
        self.declare_parameter("poll_period_s", 0.05)
        self.declare_parameter("diagnostics_period_s", 1.0)
        self.declare_parameter("reconnect_period_s", 1.0)
        self.declare_parameter("stale_timeout_s", 0.25)
        self.declare_parameter("publish_period_s", 0.02)

        # UDP support is passive and disabled until each controller has already
        # been configured to report to the matching host and unique port.
        self.declare_parameter("udp_enabled", False)
        self.declare_parameter("udp_bind_host", "0.0.0.0")
        self.declare_parameter("left_udp_port", 8089)
        self.declare_parameter("right_udp_port", 8090)
        self.declare_parameter(
            "udp_joint_speed_scale_rad_s",
            GEN4_JOINT_SPEED_TO_RAD_S,
        )

        self._stale_timeout_s = float(
            self.get_parameter("stale_timeout_s").value
        )
        tcp_port = int(self.get_parameter("tcp_port").value)
        tcp_timeout_s = float(self.get_parameter("tcp_timeout_s").value)
        poll_period_s = float(self.get_parameter("poll_period_s").value)
        diagnostics_period_s = float(
            self.get_parameter("diagnostics_period_s").value
        )
        reconnect_period_s = float(
            self.get_parameter("reconnect_period_s").value
        )

        self._workers: Dict[str, RM75StateWorker] = {}
        for side in ("left", "right"):
            host = str(self.get_parameter(f"{side}_ip").value)
            worker = RM75StateWorker(
                side=side,
                host=host,
                port=tcp_port,
                timeout_s=tcp_timeout_s,
                poll_period_s=poll_period_s,
                diagnostics_period_s=diagnostics_period_s,
                reconnect_period_s=reconnect_period_s,
                stale_timeout_s=self._stale_timeout_s,
            )
            self._workers[side] = worker
            worker.start()

        self._udp_receivers: Dict[str, RM75UDPStateReceiver] = {}
        if bool(self.get_parameter("udp_enabled").value):
            bind_host = str(self.get_parameter("udp_bind_host").value)
            for side in ("left", "right"):
                receiver = RM75UDPStateReceiver(
                    side=side,
                    bind_host=bind_host,
                    bind_port=int(
                        self.get_parameter(f"{side}_udp_port").value
                    ),
                    expected_source_ip=str(
                        self.get_parameter(f"{side}_ip").value
                    ),
                    timeout_s=tcp_timeout_s,
                    stale_timeout_s=self._stale_timeout_s,
                    joint_speed_scale_rad_s=float(
                        self.get_parameter(
                            "udp_joint_speed_scale_rad_s"
                        ).value
                    ),
                )
                self._udp_receivers[side] = receiver
                receiver.start()

        self._joint_publishers = {
            side: self.create_publisher(
                JointState, f"/rm75/{side}/actual_joint_states", 10
            )
            for side in ("left", "right")
        }
        self._connected_publishers = {
            side: self.create_publisher(Bool, f"/rm75/{side}/connected", 10)
            for side in ("left", "right")
        }
        self._stale_publishers = {
            side: self.create_publisher(Bool, f"/rm75/{side}/state_stale", 10)
            for side in ("left", "right")
        }
        self._enabled_publishers = {
            side: self.create_publisher(
                Bool, f"/rm75/{side}/joints_enabled", 10
            )
            for side in ("left", "right")
        }
        self._fault_publishers = {
            side: self.create_publisher(Bool, f"/rm75/{side}/fault", 10)
            for side in ("left", "right")
        }
        self._dual_joint_publisher = self.create_publisher(
            JointState, "/rm75/actual_joint_states", 10
        )
        self._diagnostics_publisher = self.create_publisher(
            DiagnosticArray, "/rm75/state_diagnostics", 10
        )

        self._last_published_sample: Dict[str, Optional[Tuple[str, float]]] = {
            "left": None,
            "right": None,
        }
        self._latest_states: Dict[str, Optional[RM75ArmState]] = {
            "left": None,
            "right": None,
        }
        self._timer = self.create_timer(
            float(self.get_parameter("publish_period_s").value), self._publish
        )

        self.get_logger().info(
            "Dual RM75 state interface started in READ-ONLY mode; "
            "no robot motion/configuration command exists in this node."
        )

    @staticmethod
    def _joint_names(side: str):
        prefix = "l" if side == "left" else "r"
        return [f"{prefix}_rm75_joint_{index}" for index in range(1, 8)]

    def _select_state(
        self,
        side: str,
        tcp_status: RM75StateStatus,
    ) -> Tuple[Optional[RM75ArmState], bool]:
        receiver = self._udp_receivers.get(side)
        if receiver is not None:
            udp_status = receiver.get_status()
            if udp_status.state is not None and not udp_status.stale:
                # Prefer the direct, high-rate controller velocity sample.
                # TCP remains the explicit fallback when UDP is unavailable.
                return udp_status.state, False
        if tcp_status.state is not None and not tcp_status.stale:
            return tcp_status.state, False
        return tcp_status.state, True

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        statuses = {}
        any_new_sample = False

        for side in ("left", "right"):
            tcp_status = self._workers[side].get_status()
            state, stale = self._select_state(side, tcp_status)
            statuses[side] = (tcp_status, state, stale)

            connected = tcp_status.connected
            udp_receiver = self._udp_receivers.get(side)
            if udp_receiver is not None:
                udp_status = udp_receiver.get_status()
                connected = connected or (
                    udp_status.connected and not udp_status.stale
                )

            self._connected_publishers[side].publish(Bool(data=connected))
            self._stale_publishers[side].publish(Bool(data=stale))
            self._enabled_publishers[side].publish(
                Bool(
                    data=(
                        state is not None
                        and not stale
                        and state.all_joints_enabled is True
                    )
                )
            )
            self._fault_publishers[side].publish(
                Bool(
                    data=(
                        state is not None and not stale and state.has_fault
                    )
                )
            )

            if state is None or stale:
                continue
            self._latest_states[side] = state
            sample_id = (state.source, state.measurement_seq)
            if sample_id == self._last_published_sample[side]:
                continue

            message = JointState()
            message.header.stamp = stamp
            message.name = self._joint_names(side)
            message.position = list(state.q_measured)
            if state.qd_measured is not None:
                message.velocity = list(state.qd_measured)
            self._joint_publishers[side].publish(message)
            self._last_published_sample[side] = sample_id
            any_new_sample = True

        left = self._latest_states["left"]
        right = self._latest_states["right"]
        if (
            any_new_sample
            and left is not None
            and right is not None
            and not statuses["left"][2]
            and not statuses["right"][2]
        ):
            message = JointState()
            message.header.stamp = stamp
            message.name = (
                self._joint_names("left") + self._joint_names("right")
            )
            message.position = list(left.q_measured) + list(right.q_measured)
            if left.qd_measured is not None and right.qd_measured is not None:
                message.velocity = (
                    list(left.qd_measured) + list(right.qd_measured)
                )
            self._dual_joint_publisher.publish(message)

        self._publish_diagnostics(stamp, statuses)

    def _publish_diagnostics(self, stamp, statuses) -> None:
        message = DiagnosticArray()
        message.header.stamp = stamp

        for side in ("left", "right"):
            tcp_status, state, stale = statuses[side]
            diagnostic = DiagnosticStatus()
            diagnostic.name = f"RM75 {side} read-only state"
            diagnostic.hardware_id = str(
                self.get_parameter(f"{side}_ip").value
            )

            if not tcp_status.connected and (state is None or stale):
                diagnostic.level = DiagnosticStatus.ERROR
                diagnostic.message = "state connection unavailable"
            elif stale:
                diagnostic.level = DiagnosticStatus.ERROR
                diagnostic.message = "state data stale"
            elif state is not None and state.has_fault:
                diagnostic.level = DiagnosticStatus.ERROR
                diagnostic.message = "robot fault reported"
            elif state is not None and state.all_joints_enabled is False:
                diagnostic.level = DiagnosticStatus.WARN
                diagnostic.message = "one or more joints disabled"
            else:
                diagnostic.level = DiagnosticStatus.OK
                diagnostic.message = "fresh read-only state"

            values = {
                "tcp_connected": str(tcp_status.connected).lower(),
                "stale": str(stale).lower(),
                "last_error": tcp_status.last_error or "",
                "source": state.source if state is not None else "none",
                "measurement_seq": str(
                    state.measurement_seq if state is not None else ""
                ),
                "measurement_period_s": (
                    f"{state.measurement_period_s:.6f}"
                    if state is not None
                    and state.measurement_period_s is not None
                    else "unknown"
                ),
                "effective_measurement_hz": (
                    f"{state.effective_measurement_hz:.3f}"
                    if state is not None
                    and state.effective_measurement_hz is not None
                    else "unknown"
                ),
                "velocity_source": (
                    state.velocity_source if state is not None else "none"
                ),
                "joint_velocity_valid": str(
                    state is not None and state.qd_measured is not None
                ).lower(),
                "query_latency_s": (
                    f"{state.query_latency_s:.6f}"
                    if state is not None and state.query_latency_s is not None
                    else "unknown"
                ),
                "joint_valid": str(state is not None and not stale).lower(),
                "age_s": (
                    f"{state.age_s():.6f}" if state is not None else "inf"
                ),
                "arm_error": str(state.arm_error if state is not None else ""),
                "controller_error": str(
                    state.controller_error
                    if state is not None and state.controller_error is not None
                    else "unknown"
                ),
                "all_joints_enabled": (
                    str(state.all_joints_enabled).lower()
                    if state is not None
                    else "unknown"
                ),
                "joint_errors": (
                    str(list(state.joint_errors))
                    if state is not None and state.joint_errors is not None
                    else "unknown"
                ),
                "brake_released": (
                    str(list(state.brake_released))
                    if state is not None and state.brake_released is not None
                    else "unknown"
                ),
                "arm_motion_state": (
                    state.arm_motion_state
                    if state is not None and state.arm_motion_state is not None
                    else "unknown"
                ),
            }
            diagnostic.values = [
                KeyValue(key=key, value=value) for key, value in values.items()
            ]
            message.status.append(diagnostic)

        self._diagnostics_publisher.publish(message)

    def destroy_node(self):
        """Stop all socket workers before destroying the ROS node."""
        for receiver in self._udp_receivers.values():
            receiver.stop()
        for worker in self._workers.values():
            worker.stop()
        return super().destroy_node()


def main(args=None) -> None:
    """Run the read-only dual-arm state publisher."""
    rclpy.init(args=args)
    node = RM75StateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
