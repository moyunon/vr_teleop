"""
Default-disabled RM75 low-follow joint command interface.

The selected wire command is ``movej_canfd`` with ``follow=false``.  RealMan
requires a cycle no longer than 10 ms for high-follow mode, while this control
loop is 20 ms.  High follow is therefore intentionally not configurable here.
Low-follow CANFD accepts a stream of already rate-limited joint targets and is
also the mode used by the existing RealMan ROS 2 driver for planned points.

Official protocol reference:
https://develop.realman-robotics.com/robot/json/motionConfig/
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
import time
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from vr_rm75_teleop.rm75_model import RM75Model


MOVEJ_CANFD = "movej_canfd"
SIDES = ("left", "right")
RAD_TO_MILLI_DEG = 180000.0 / math.pi
MILLI_DEG_TO_RAD = math.pi / 180000.0
MAX_RESPONSE_BYTES = 65536


class RM75CommandError(RuntimeError):
    """Base class for command-interface failures."""


class RM75MotionDisabledError(RM75CommandError):
    """Motion was attempted while the default-off gate was closed."""


class RM75CommandRejectedError(RM75CommandError):
    """A command failed a freshness, jump, velocity, or numeric check."""


class RM75CommandConnectionError(RM75CommandError):
    """A command transport could not connect or send a complete frame."""


class RM75ControllerCommandError(RM75CommandError):
    """The controller returned an error for a transmitted command."""


@dataclass(frozen=True)
class CommandDispatchResult:
    """Outcome of one atomic dual-arm dispatch attempt."""

    sent: bool
    reason: str
    generated_monotonic: Optional[float]
    sent_monotonic: Optional[float]


@dataclass(frozen=True)
class CommandControllerResponse:
    """Validated controller response to one low-follow point."""

    side: str
    q_reported: Tuple[float, ...]
    arm_error: int


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r}")


def decode_movej_canfd_response(data, side):
    """Decode one strict seven-axis ``joint_state`` response."""
    side = str(side).lower()
    model = RM75Model(side=side)
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RM75CommandConnectionError(
                "command response is not UTF-8"
            ) from exc
    try:
        response = json.loads(data, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RM75CommandConnectionError(
            f"invalid command JSON response: {exc}"
        ) from exc
    if not isinstance(response, Mapping):
        raise RM75CommandConnectionError(
            "command response root must be an object"
        )
    if response.get("state") != "joint_state":
        raise RM75CommandConnectionError(
            "unexpected movej_canfd response state"
        )

    joint = response.get("joint")
    if (
        not isinstance(joint, list)
        or len(joint) != model.DOF
        or any(isinstance(value, bool) for value in joint)
        or any(not isinstance(value, (int, float)) for value in joint)
    ):
        raise RM75CommandConnectionError(
            "command response joint must contain 7 numbers"
        )
    q_reported = np.asarray(joint, dtype=float) * MILLI_DEG_TO_RAD
    if not np.all(np.isfinite(q_reported)):
        raise RM75CommandConnectionError(
            "command response joint contains NaN or Infinity"
        )
    if (
        np.any(q_reported < model.q_min)
        or np.any(q_reported > model.q_max)
    ):
        raise RM75CommandConnectionError(
            "command response joint exceeds RM75 hard limits"
        )

    arm_error = response.get("arm_err")
    if (
        isinstance(arm_error, bool)
        or not isinstance(arm_error, (int, float))
        or not math.isfinite(float(arm_error))
        or not float(arm_error).is_integer()
    ):
        raise RM75CommandConnectionError(
            "command response arm_err must be an integer"
        )
    arm_error = int(arm_error)
    if arm_error != 0:
        raise RM75ControllerCommandError(
            f"{side} controller rejected movej_canfd: arm_err={arm_error}"
        )
    return CommandControllerResponse(
        side=side,
        q_reported=tuple(float(value) for value in q_reported),
        arm_error=arm_error,
    )


def encode_low_follow_movej_canfd(q_rad, side):
    """Encode one validated seven-axis target using integer 0.001 degrees."""
    side = str(side).lower()
    model = RM75Model(side=side)
    try:
        q_rad = np.asarray(q_rad, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RM75CommandRejectedError(
            "joint command must be numeric"
        ) from exc

    if q_rad.shape != (model.DOF,):
        raise RM75CommandRejectedError("joint command must contain 7 values")
    if not np.all(np.isfinite(q_rad)):
        raise RM75CommandRejectedError(
            "joint command contains NaN or Infinity"
        )
    if np.any(q_rad < model.q_min) or np.any(q_rad > model.q_max):
        raise RM75CommandRejectedError(
            "joint command exceeds RM75 hard limits"
        )

    joint_milli_degrees = [
        int(round(float(value) * RAD_TO_MILLI_DEG)) for value in q_rad
    ]
    message = {
        "command": MOVEJ_CANFD,
        "joint": joint_milli_degrees,
        "follow": False,
        "expand": 0,
        "trajectory_mode": 0,
        "radio": 0,
    }
    return json.dumps(
        message,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii") + b"\r\n"


class RM75LowFollowCommandClient:
    """Send-only TCP transport that cannot connect while motion is disabled."""

    def __init__(
        self,
        side,
        host,
        port=8080,
        timeout_s=0.01,
        enable_robot_motion=False,
        socket_factory: Optional[Callable[..., socket.socket]] = None,
    ):
        """Configure one endpoint without opening a socket or sending data."""
        self.side = str(side).lower()
        if self.side not in SIDES:
            raise ValueError("side must be 'left' or 'right'")
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be in [1, 65535]")
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")

        self.host = str(host)
        self.port = int(port)
        self.timeout_s = timeout_s
        self.enable_robot_motion = bool(enable_robot_motion)
        self._socket_factory = socket_factory or socket.create_connection
        self._socket: Optional[socket.socket] = None
        self._receive_buffer = bytearray()
        self.last_response_monotonic: Optional[float] = None

    @property
    def connected(self):
        """Return whether the command socket is currently open."""
        return self._socket is not None

    def connect(self):
        """Connect only after the explicit motion gate has been enabled."""
        if not self.enable_robot_motion:
            raise RM75MotionDisabledError("enable_robot_motion is false")
        self.close()
        try:
            sock = self._socket_factory((self.host, self.port), self.timeout_s)
            sock.settimeout(self.timeout_s)
        except (OSError, socket.timeout) as exc:
            raise RM75CommandConnectionError(
                "could not connect command socket "
                f"{self.host}:{self.port}: {exc}"
            ) from exc
        self._socket = sock
        self._receive_buffer.clear()
        self.last_response_monotonic = None

    def close(self):
        """Close the command socket without issuing any robot command."""
        sock, self._socket = self._socket, None
        self._receive_buffer.clear()
        self.last_response_monotonic = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def send_joint_target(self, q_rad):
        """Send one low-follow target without waiting for its response."""
        if not self.enable_robot_motion:
            raise RM75MotionDisabledError("enable_robot_motion is false")
        if self._socket is None:
            raise RM75CommandConnectionError("command socket is not connected")
        payload = encode_low_follow_movej_canfd(q_rad, self.side)
        try:
            self._socket.sendall(payload)
        except (OSError, socket.timeout) as exc:
            self.close()
            raise RM75CommandConnectionError(
                f"failed sending {self.side} {MOVEJ_CANFD}: {exc}"
            ) from exc
        return payload

    def receive_joint_result(self):
        """Wait for and validate one controller ``joint_state`` response."""
        if not self.enable_robot_motion:
            raise RM75MotionDisabledError("enable_robot_motion is false")
        if self._socket is None:
            raise RM75CommandConnectionError(
                "command socket is not connected"
            )

        while True:
            newline = self._receive_buffer.find(b"\n")
            if newline >= 0:
                frame = bytes(
                    self._receive_buffer[:newline]
                ).rstrip(b"\r")
                del self._receive_buffer[: newline + 1]
                if not frame:
                    continue
                response = decode_movej_canfd_response(frame, self.side)
                self.last_response_monotonic = time.monotonic()
                return response

            if len(self._receive_buffer) > MAX_RESPONSE_BYTES:
                self.close()
                raise RM75CommandConnectionError(
                    "command response exceeds size limit"
                )
            try:
                chunk = self._socket.recv(4096)
            except (OSError, socket.timeout) as exc:
                self.close()
                raise RM75CommandConnectionError(
                    f"timeout or failure receiving {self.side} "
                    f"{MOVEJ_CANFD} response: {exc}"
                ) from exc
            if not chunk:
                self.close()
                raise RM75CommandConnectionError(
                    f"{self.side} controller closed the command socket"
                )
            self._receive_buffer.extend(chunk)


class DualArmCommandDispatcher:
    """Validate both targets before either low-follow command is sent."""

    def __init__(
        self,
        clients,
        joint_velocity_limits,
        *,
        joint_acceleration_limits=None,
        enable_robot_motion=False,
        max_command_delta_rad=math.radians(0.5),
        command_timeout_s=0.10,
        nominal_period_s=0.02,
        monotonic=time.monotonic,
    ):
        """Configure a default-off latest-command dispatcher."""
        if set(clients) != set(SIDES):
            raise ValueError("clients must contain left and right")
        if set(joint_velocity_limits) != set(SIDES):
            raise ValueError(
                "joint_velocity_limits must contain left and right"
            )
        self.clients = dict(clients)
        self.enable_robot_motion = bool(enable_robot_motion)
        self.max_command_delta = self._normalize_positive_limit(
            max_command_delta_rad,
            "max_command_delta_rad",
        )
        self.command_timeout_s = self._positive_finite(
            command_timeout_s,
            "command_timeout_s",
        )
        self.nominal_period_s = self._positive_finite(
            nominal_period_s,
            "nominal_period_s",
        )
        self._velocity_limits = {
            side: self._normalize_positive_limit(
                joint_velocity_limits[side],
                f"{side} joint_velocity_limits",
            )
            for side in SIDES
        }
        if joint_acceleration_limits is None:
            if self.enable_robot_motion:
                raise ValueError(
                    "enabled motion requires joint_acceleration_limits"
                )
            self._acceleration_limits = {}
        else:
            if set(joint_acceleration_limits) != set(SIDES):
                raise ValueError(
                    "joint_acceleration_limits must contain left and right"
                )
            self._acceleration_limits = {
                side: self._normalize_positive_limit(
                    joint_acceleration_limits[side],
                    f"{side} joint_acceleration_limits",
                )
                for side in SIDES
            }
        self._monotonic = monotonic
        self._last_sent_q: Dict[str, np.ndarray] = {}
        self._last_sent_velocity: Dict[str, np.ndarray] = {}
        self._last_generated_monotonic: Optional[float] = None
        self.last_send_monotonic: Optional[float] = None
        self.last_reason = "enable_robot_motion is false"
        self.faulted = False

    @property
    def connected(self):
        """Require both independent command transports to be connected."""
        return all(self.clients[side].connected for side in SIDES)

    def connect(self):
        """Connect both transports, closing both after any partial failure."""
        if not self.enable_robot_motion:
            raise RM75MotionDisabledError("enable_robot_motion is false")
        try:
            for side in SIDES:
                self.clients[side].connect()
        except RM75CommandError:
            self.close()
            self.faulted = True
            self.last_reason = "dual-arm command connection failed"
            raise
        self.faulted = False
        self.last_reason = "dual-arm command sockets connected"

    def close(self):
        """Close both transports and discard command continuity history."""
        for client in self.clients.values():
            client.close()
        self.disarm()

    def disarm(self):
        """Discard the previous target so re-engagement resynchronizes."""
        self._last_sent_q.clear()
        self._last_sent_velocity.clear()
        self._last_generated_monotonic = None
        self.last_send_monotonic = None

    def reset_fault(self):
        """Clear the local latch; reconnect is still explicitly required."""
        self.close()
        self.faulted = False
        self.last_reason = "command fault reset; reconnect required"

    def latch_transport_fault(self, reason):
        """Close both transports and latch an externally detected failure."""
        self.close()
        self.faulted = True
        self.last_reason = str(reason) or "command transport fault latched"

    def dispatch(
        self,
        q_commands,
        q_measured,
        *,
        generated_monotonic,
        safety_command_allowed,
        now_monotonic=None,
    ):
        """Validate a fresh dual command, then send left and right once."""
        if not self.enable_robot_motion:
            self.last_reason = "enable_robot_motion is false; dry-run only"
            return CommandDispatchResult(False, self.last_reason, None, None)
        if self.faulted:
            raise RM75CommandConnectionError(
                "command interface fault is latched"
            )
        if not safety_command_allowed:
            self.disarm()
            self.last_reason = "Safety Supervisor command gate is closed"
            return CommandDispatchResult(False, self.last_reason, None, None)
        if not self.connected:
            raise RM75CommandConnectionError(
                "both command sockets must be connected before dispatch"
            )

        if now_monotonic is None:
            now_monotonic = self._monotonic()
        now_monotonic = self._finite_time(now_monotonic)
        if generated_monotonic is None:
            raise RM75CommandRejectedError(
                "safe command timestamp is unavailable"
            )
        generated_monotonic = self._finite_time(generated_monotonic)
        age_s = max(0.0, now_monotonic - generated_monotonic)
        if age_s > self.command_timeout_s:
            raise RM75CommandRejectedError(
                f"safe command is stale ({age_s:.3f}s)"
            )
        if (
            self._last_generated_monotonic is not None
            and generated_monotonic <= self._last_generated_monotonic
        ):
            self.last_reason = "safe command was already dispatched"
            return CommandDispatchResult(
                False,
                self.last_reason,
                generated_monotonic,
                None,
            )

        commands = self._normalize_dual_q(q_commands, "q_commands")
        measured = self._normalize_dual_q(q_measured, "q_measured")
        if self._last_generated_monotonic is None:
            dt_s = self.nominal_period_s
        else:
            dt_s = generated_monotonic - self._last_generated_monotonic
            if not math.isfinite(dt_s) or dt_s <= 0.0:
                raise RM75CommandRejectedError(
                    "safe command timestamp did not advance"
                )

        command_velocities = {}
        for side in SIDES:
            reference = self._last_sent_q.get(side, measured[side])
            delta = commands[side] - reference
            if np.any(np.abs(delta) > self.max_command_delta + 1e-12):
                raise RM75CommandRejectedError(
                    f"{side} command jump exceeds configured maximum"
                )
            qdot = delta / dt_s
            command_velocities[side] = qdot
            if np.any(
                np.abs(qdot) > self._velocity_limits[side] + 1e-12
            ):
                raise RM75CommandRejectedError(
                    f"{side} command velocity exceeds configured maximum"
                )
            previous_qdot = self._last_sent_velocity.get(
                side,
                np.zeros(7),
            )
            qddot = (qdot - previous_qdot) / dt_s
            if np.any(
                np.abs(qddot)
                > self._acceleration_limits[side] + 1e-12
            ):
                raise RM75CommandRejectedError(
                    f"{side} command acceleration exceeds configured maximum"
                )

        # Encoding both commands before the first socket write makes all
        # numeric/protocol rejection atomic across the dual-arm pair.
        for side in SIDES:
            encode_low_follow_movej_canfd(commands[side], side)

        try:
            for side in SIDES:
                self.clients[side].send_joint_target(commands[side])
            for side in SIDES:
                self.clients[side].receive_joint_result()
        except RM75CommandError:
            self.close()
            self.faulted = True
            self.last_reason = "dual-arm command send failed; fault latched"
            raise

        sent_monotonic = self._monotonic()
        self._last_sent_q = {
            side: commands[side].copy() for side in SIDES
        }
        self._last_sent_velocity = {
            side: command_velocities[side].copy() for side in SIDES
        }
        self._last_generated_monotonic = generated_monotonic
        self.last_send_monotonic = sent_monotonic
        self.last_reason = (
            "fresh dual-arm low-follow command accepted by controllers"
        )
        return CommandDispatchResult(
            True,
            self.last_reason,
            generated_monotonic,
            sent_monotonic,
        )

    @staticmethod
    def _positive_finite(value, name):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    @staticmethod
    def _normalize_positive_limit(values, name):
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if values.ndim == 0:
            values = np.full(7, float(values))
        if values.shape != (7,):
            raise ValueError(f"{name} must be scalar or contain 7 values")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
        return values.copy()

    @staticmethod
    def _finite_time(value):
        value = float(value)
        if not math.isfinite(value):
            raise RM75CommandRejectedError("command timestamp must be finite")
        return value

    @staticmethod
    def _normalize_dual_q(values, name):
        if not isinstance(values, Mapping) or set(values) != set(SIDES):
            raise RM75CommandRejectedError(
                f"{name} must contain left and right"
            )
        normalized = {}
        for side in SIDES:
            try:
                q = np.asarray(values[side], dtype=float)
            except (TypeError, ValueError) as exc:
                raise RM75CommandRejectedError(
                    f"{side} {name} must be numeric"
                ) from exc
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                raise RM75CommandRejectedError(
                    f"{side} {name} must be a finite 7-vector"
                )
            model = RM75Model(side)
            if np.any(q < model.q_min) or np.any(q > model.q_max):
                raise RM75CommandRejectedError(
                    f"{side} {name} exceeds RM75 hard limits"
                )
            normalized[side] = q.copy()
        return normalized
