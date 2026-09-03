"""Read-only RealMan RM75 state transport and protocol validation.

This module deliberately contains no motion command and no ROS dependency.  It
can therefore be tested without a controller and reused by a ROS node without
coupling socket lifetime to the control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import socket
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from vr_rm75_teleop.rm75_model import RM75Model


DOF = 7
MILLI_DEG_TO_RAD = math.pi / 180000.0
# Gen-4 documentation defines UDP ``joint_speed`` in 0.02 rpm units.  This
# scale is explicit so commissioning can detect a controller-generation
# mismatch instead of silently treating the raw integer as rad/s.
GEN4_JOINT_SPEED_TO_RAD_S = 0.02 * 2.0 * math.pi / 60.0
MAX_JSON_BYTES = 65536

GET_CURRENT_ARM_STATE = "get_current_arm_state"
GET_CONTROLLER_STATE = "get_controller_state"
GET_JOINT_ENABLE_STATE = "get_joint_en_state"
GET_JOINT_ERROR_FLAG = "get_joint_err_flag"
GET_REALTIME_PUSH = "get_realtime_push"

# This is intentionally a closed allowlist.  In particular it contains no
# move_*, set_*, clear_* or enable command.
READ_ONLY_COMMANDS = frozenset(
    {
        GET_CURRENT_ARM_STATE,
        GET_CONTROLLER_STATE,
        GET_JOINT_ENABLE_STATE,
        GET_JOINT_ERROR_FLAG,
        GET_REALTIME_PUSH,
    }
)


class RM75InterfaceError(RuntimeError):
    """Base class for a state-interface failure."""


class RM75ConnectionError(RM75InterfaceError):
    """The TCP/UDP transport is unavailable or disconnected."""


class RM75TimeoutError(RM75InterfaceError):
    """The controller did not return a complete response in time."""


class RM75ProtocolError(RM75InterfaceError):
    """A controller response is malformed, unexpected, or unsafe to use."""


@dataclass(frozen=True)
class RM75ArmState:
    """One validated seven-axis sample in controller joint convention."""

    side: str
    q_measured: Tuple[float, ...]
    arm_error: int
    controller_error: Optional[int]
    joint_enabled: Optional[Tuple[bool, ...]]
    joint_errors: Optional[Tuple[int, ...]]
    brake_released: Optional[Tuple[bool, ...]]
    arm_motion_state: Optional[str]
    received_monotonic: float
    source: str
    qd_measured: Optional[Tuple[float, ...]] = None
    velocity_source: str = "unavailable"
    measurement_seq: int = 0
    measurement_period_s: Optional[float] = None
    query_latency_s: Optional[float] = None

    def age_s(self, now_monotonic: Optional[float] = None) -> float:
        """Return local receive age, clamped against a backwards clock jump."""
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        return max(0.0, float(now_monotonic) - self.received_monotonic)

    def is_stale(
        self,
        stale_timeout_s: float,
        now_monotonic: Optional[float] = None,
    ) -> bool:
        """Return whether this sample is older than the configured deadline."""
        if not math.isfinite(stale_timeout_s) or stale_timeout_s <= 0.0:
            raise ValueError("stale_timeout_s must be finite and positive")
        return self.age_s(now_monotonic) > stale_timeout_s

    @property
    def has_fault(self) -> bool:
        """Return true for any controller/arm or per-joint error code."""
        return (
            self.arm_error != 0
            or self.controller_error not in (None, 0)
            or bool(
                self.joint_errors
                and any(code != 0 for code in self.joint_errors)
            )
        )

    @property
    def all_joints_enabled(self) -> Optional[bool]:
        """Return aggregate enable state, or None until it has been queried."""
        if self.joint_enabled is None:
            return None
        return all(self.joint_enabled)

    @property
    def effective_measurement_hz(self) -> Optional[float]:
        """Return receive-rate estimate without using the ROS publish rate."""
        period = self.measurement_period_s
        if period is None or not math.isfinite(period) or period <= 0.0:
            return None
        return 1.0 / period


@dataclass(frozen=True)
class RM75StateStatus:
    """Thread-safe transport status exposed by a background state reader."""

    connected: bool
    state: Optional[RM75ArmState]
    stale: bool
    last_error: Optional[str]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def decode_json_object(data: Any) -> Mapping[str, Any]:
    """Decode one strict JSON object and reject NaN/Infinity extensions."""
    if isinstance(data, Mapping):
        value = data
    else:
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RM75ProtocolError("response is not UTF-8") from exc
        if not isinstance(data, str):
            raise RM75ProtocolError("response must be a JSON object or bytes")
        try:
            value = json.loads(data, parse_constant=_reject_json_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RM75ProtocolError(f"invalid JSON response: {exc}") from exc

    if not isinstance(value, Mapping):
        raise RM75ProtocolError("response root must be a JSON object")
    return value


def _validate_side(side: str) -> str:
    side = str(side).lower()
    if side not in RM75Model.VALID_SIDES:
        raise ValueError("side must be 'left' or 'right'")
    return side


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RM75ProtocolError(f"{field} must contain numbers")
    value = float(value)
    if not math.isfinite(value):
        raise RM75ProtocolError(f"{field} contains NaN or Infinity")
    return value


def _integer(value: Any, field: str) -> int:
    number = _finite_number(value, field)
    if not number.is_integer():
        raise RM75ProtocolError(f"{field} must contain integers")
    return int(number)


def _array(value: Any, field: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != DOF
    ):
        raise RM75ProtocolError(f"{field} must contain exactly {DOF} values")
    return value


def _parse_joint_angles(
    value: Any,
    side: str,
    field: str,
) -> Tuple[float, ...]:
    raw = _array(value, field)
    q = tuple(_finite_number(item, field) * MILLI_DEG_TO_RAD for item in raw)

    model = RM75Model(side=side)
    for index, (angle, lower, upper) in enumerate(
        zip(q, model.q_min, model.q_max), start=1
    ):
        if angle < float(lower) - 1e-9 or angle > float(upper) + 1e-9:
            raise RM75ProtocolError(
                f"{field}[{index - 1}] is outside RM75 joint {index} limits"
            )
    return q


def _parse_bool_flags(value: Any, field: str) -> Tuple[bool, ...]:
    result = []
    for item in _array(value, field):
        integer = _integer(item, field)
        if integer not in (0, 1):
            raise RM75ProtocolError(f"{field} values must be 0 or 1")
        result.append(bool(integer))
    return tuple(result)


def _parse_int_flags(value: Any, field: str) -> Tuple[int, ...]:
    return tuple(_integer(item, field) for item in _array(value, field))


def _parse_optional_joint_speed(
    value: Any,
    field: str,
    scale_rad_s: float,
) -> Optional[Tuple[float, ...]]:
    """Parse optional controller velocity using an explicit protocol scale."""
    if value is None:
        return None
    scale_rad_s = float(scale_rad_s)
    if not math.isfinite(scale_rad_s) or scale_rad_s <= 0.0:
        raise ValueError("joint speed scale must be finite and positive")
    return tuple(
        _finite_number(item, field) * scale_rad_s
        for item in _array(value, field)
    )


def _parse_error_code(value: Any, field: str) -> int:
    """Accept scalar or documented single-element error-code arrays."""
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
    ):
        if len(value) != 1:
            raise RM75ProtocolError(
                f"{field} array must contain exactly one value"
            )
        value = value[0]
    return _integer(value, field)

def _parse_brake_released(value: Any) -> Tuple[bool, ...]:
    """Convert RM75 brake_state to whether each brake is released.

    Gen-4 JSON semantics:
    0 -> brake opened/released
    1 -> brake not opened
    """
    brake_state = _parse_bool_flags(value, "brake_state")
    return tuple(not engaged for engaged in brake_state)


def _matches_response(message: Mapping[str, Any], expected: str) -> bool:
    command = message.get("command")
    if command == expected:
        return True

    legacy_states = {
        GET_CURRENT_ARM_STATE: "current_arm_state",
        GET_CONTROLLER_STATE: "controller_state",
        GET_JOINT_ENABLE_STATE: "joint_en_state",
        GET_JOINT_ERROR_FLAG: "joint_err_flag",
    }
    state = message.get("state")
    if state == legacy_states.get(expected):
        return True

    # Some older current-state replies omit the echoed command.
    return (
        expected == GET_CURRENT_ARM_STATE
        and command is None
        and "arm_state" in message
    )


def parse_realtime_udp_state(
    data: Any,
    side: str,
    received_monotonic: Optional[float] = None,
    *,
    joint_speed_scale_rad_s: float = GEN4_JOINT_SPEED_TO_RAD_S,
    measurement_seq: int = 0,
    measurement_period_s: Optional[float] = None,
) -> RM75ArmState:
    """Validate a Gen-4 ``realtime_arm_joint_state`` UDP datagram."""
    side = _validate_side(side)
    message = decode_json_object(data)
    if message.get("state") != "realtime_arm_joint_state":
        raise RM75ProtocolError("unexpected UDP state message")

    joint_status = message.get("joint_status")
    if not isinstance(joint_status, Mapping):
        raise RM75ProtocolError("joint_status must be an object")

    q = _parse_joint_angles(
        joint_status.get("joint_position"), side, "joint_status.joint_position"
    )
    enabled = _parse_bool_flags(
        joint_status.get("joint_en_flag"), "joint_status.joint_en_flag"
    )
    errors = _parse_int_flags(
        joint_status.get("joint_err_code"), "joint_status.joint_err_code"
    )
    qd_measured = _parse_optional_joint_speed(
        joint_status.get("joint_speed"),
        "joint_status.joint_speed",
        joint_speed_scale_rad_s,
    )
    arm_error = _parse_error_code(message.get("err", 0), "err")
    motion_state = message.get("arm_current_status")
    if motion_state is not None and not isinstance(motion_state, str):
        raise RM75ProtocolError("arm_current_status must be a string")

    if received_monotonic is None:
        received_monotonic = time.monotonic()
    received_monotonic = _finite_number(
        received_monotonic, "received_monotonic"
    )
    measurement_seq = _integer(measurement_seq, "measurement_seq")
    if measurement_seq < 0:
        raise RM75ProtocolError("measurement_seq must be non-negative")
    if measurement_period_s is not None:
        measurement_period_s = _finite_number(
            measurement_period_s,
            "measurement_period_s",
        )
        if measurement_period_s <= 0.0:
            raise RM75ProtocolError(
                "measurement_period_s must be positive"
            )

    return RM75ArmState(
        side=side,
        q_measured=q,
        arm_error=arm_error,
        controller_error=None,
        joint_enabled=enabled,
        joint_errors=errors,
        brake_released=None,
        arm_motion_state=motion_state,
        received_monotonic=received_monotonic,
        source="udp",
        qd_measured=qd_measured,
        velocity_source=(
            "controller_udp_joint_speed"
            if qd_measured is not None
            else "unavailable"
        ),
        measurement_seq=measurement_seq,
        measurement_period_s=measurement_period_s,
    )


class RM75ReadOnlyClient:
    """Line-framed TCP client exposing only fixed RM75 state queries."""

    def __init__(
        self,
        host: str,
        port: int = 8080,
        timeout_s: float = 0.25,
        socket_factory: Optional[Callable[..., socket.socket]] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure one controller endpoint; connect only when requested."""
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")

        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._socket_factory = socket_factory or socket.create_connection
        self._monotonic = monotonic
        self._socket: Optional[socket.socket] = None
        self._receive_buffer = bytearray()
        self._joint_enabled: Optional[Tuple[bool, ...]] = None
        self._joint_errors: Optional[Tuple[int, ...]] = None
        self._brake_released: Optional[Tuple[bool, ...]] = None
        self._controller_error: Optional[int] = None

    @property
    def connected(self) -> bool:
        """Return whether a TCP socket is currently open."""
        return self._socket is not None

    def connect(self) -> None:
        """Open the state-query connection without sending any command."""
        self.close()
        try:
            sock = self._socket_factory((self.host, self.port), self.timeout_s)
            sock.settimeout(self.timeout_s)
        except (OSError, socket.timeout) as exc:
            raise RM75ConnectionError(
                f"could not connect to {self.host}:{self.port}: {exc}"
            ) from exc
        self._socket = sock
        self._receive_buffer.clear()

    def close(self) -> None:
        """Close the state socket; this never changes controller state."""
        sock, self._socket = self._socket, None
        self._receive_buffer.clear()
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def __enter__(self) -> "RM75ReadOnlyClient":
        """Connect for a context-managed read-only session."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close a context-managed read-only session."""
        self.close()

    def _send_query(self, command: str) -> None:
        if command not in READ_ONLY_COMMANDS:
            raise RM75ProtocolError(f"command {command!r} is not read-only")
        if self._socket is None:
            raise RM75ConnectionError("state socket is not connected")

        payload = json.dumps(
            {"command": command}, separators=(",", ":"), allow_nan=False
        ).encode("ascii") + b"\r\n"
        try:
            self._socket.sendall(payload)
        except socket.timeout as exc:
            raise RM75TimeoutError(f"timeout sending {command}") from exc
        except OSError as exc:
            raise RM75ConnectionError(
                f"failed sending {command}: {exc}"
            ) from exc

    def _receive_object(self) -> Mapping[str, Any]:
        if self._socket is None:
            raise RM75ConnectionError("state socket is not connected")

        while True:
            newline = self._receive_buffer.find(b"\n")
            if newline >= 0:
                frame = bytes(self._receive_buffer[:newline]).rstrip(b"\r")
                del self._receive_buffer[: newline + 1]
                if not frame:
                    continue
                return decode_json_object(frame)

            if len(self._receive_buffer) > MAX_JSON_BYTES:
                raise RM75ProtocolError("JSON response exceeds size limit")
            try:
                chunk = self._socket.recv(4096)
            except socket.timeout as exc:
                raise RM75TimeoutError(
                    "timeout waiting for state response"
                ) from exc
            except OSError as exc:
                raise RM75ConnectionError(
                    f"failed receiving state: {exc}"
                ) from exc
            if not chunk:
                raise RM75ConnectionError(
                    "controller closed the state connection"
                )
            self._receive_buffer.extend(chunk)

    def _query(self, command: str) -> Mapping[str, Any]:
        self._send_query(command)
        for _ in range(16):
            message = self._receive_object()
            if _matches_response(message, command):
                return message
        raise RM75ProtocolError(f"no matching response for {command}")

    def read_state(
        self,
        side: str,
        include_diagnostics: bool = True,
    ) -> RM75ArmState:
        """Read joint position and, periodically, enable/fault diagnostics."""
        side = _validate_side(side)
        query_started = float(self._monotonic())
        current = self._query(GET_CURRENT_ARM_STATE)
        arm_state = current.get("arm_state")
        if not isinstance(arm_state, Mapping):
            raise RM75ProtocolError("arm_state must be an object")

        q = _parse_joint_angles(
            arm_state.get("joint"), side, "arm_state.joint"
        )
        arm_error = _parse_error_code(
            arm_state.get("err", 0),
            "arm_state.err",
        )

        if include_diagnostics:
            enable = self._query(GET_JOINT_ENABLE_STATE)
            self._joint_enabled = _parse_bool_flags(
                enable.get("en_state"), "en_state"
            )

            errors = self._query(GET_JOINT_ERROR_FLAG)
            self._joint_errors = _parse_int_flags(
                errors.get("err_flag"), "err_flag"
            )
            self._brake_released = _parse_brake_released(
                errors.get("brake_state")
            )

            controller = self._query(GET_CONTROLLER_STATE)
            self._controller_error = _integer(
                controller.get("err_flag", 0), "controller.err_flag"
            )

        received_monotonic = float(self._monotonic())
        return RM75ArmState(
            side=side,
            q_measured=q,
            arm_error=arm_error,
            controller_error=self._controller_error,
            joint_enabled=self._joint_enabled,
            joint_errors=self._joint_errors,
            brake_released=self._brake_released,
            arm_motion_state=None,
            received_monotonic=received_monotonic,
            source="tcp",
            query_latency_s=max(0.0, received_monotonic - query_started),
        )

    def read_udp_configuration(self) -> Mapping[str, Any]:
        """Read (but never modify) the controller's UDP push configuration."""
        return self._query(GET_REALTIME_PUSH)


class RM75StateWorker:
    """Reconnect-capable background TCP reader for one RM75 controller."""

    def __init__(
        self,
        side: str,
        host: str,
        port: int = 8080,
        timeout_s: float = 0.25,
        poll_period_s: float = 0.05,
        diagnostics_period_s: float = 1.0,
        reconnect_period_s: float = 1.0,
        stale_timeout_s: float = 0.25,
        socket_factory: Optional[Callable[..., socket.socket]] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure one reconnecting TCP state-reader worker."""
        self.side = _validate_side(side)
        for value, name in (
            (poll_period_s, "poll_period_s"),
            (diagnostics_period_s, "diagnostics_period_s"),
            (reconnect_period_s, "reconnect_period_s"),
            (stale_timeout_s, "stale_timeout_s"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        self.poll_period_s = float(poll_period_s)
        self.diagnostics_period_s = float(diagnostics_period_s)
        self.reconnect_period_s = float(reconnect_period_s)
        self.stale_timeout_s = float(stale_timeout_s)
        self._monotonic = monotonic
        self._client = RM75ReadOnlyClient(
            host=host,
            port=port,
            timeout_s=timeout_s,
            socket_factory=socket_factory,
            monotonic=monotonic,
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._state: Optional[RM75ArmState] = None
        self._last_error: Optional[str] = None
        self._measurement_seq = 0
        self._last_measurement_receive: Optional[float] = None

    def start(self) -> None:
        """Start the background reader if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"rm75-{self.side}-state",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 2.0) -> None:
        """Stop the reader and close its TCP connection."""
        self._stop_event.set()
        self._client.close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=join_timeout_s)
        self._thread = None

    def get_status(self) -> RM75StateStatus:
        """Return an immutable snapshot of connection and data freshness."""
        with self._lock:
            connected = self._connected
            state = self._state
            last_error = self._last_error
        stale = state is None or state.is_stale(
            self.stale_timeout_s, self._monotonic()
        )
        return RM75StateStatus(connected, state, stale, last_error)

    def _set_transport(self, connected: bool, error: Optional[str]) -> None:
        with self._lock:
            self._connected = connected
            self._last_error = error

    def _set_state(self, state: RM75ArmState) -> None:
        measurement_period_s = None
        if self._last_measurement_receive is not None:
            elapsed = state.received_monotonic - self._last_measurement_receive
            if math.isfinite(elapsed) and elapsed > 0.0:
                measurement_period_s = elapsed
        self._measurement_seq += 1
        self._last_measurement_receive = state.received_monotonic
        state = replace(
            state,
            measurement_seq=self._measurement_seq,
            measurement_period_s=measurement_period_s,
        )
        with self._lock:
            self._state = state
            self._connected = True
            self._last_error = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._client.connect()
                self._set_transport(True, None)
                next_diagnostics = -math.inf

                while not self._stop_event.is_set():
                    cycle_start = self._monotonic()
                    include_diagnostics = cycle_start >= next_diagnostics
                    state = self._client.read_state(
                        self.side, include_diagnostics=include_diagnostics
                    )
                    self._set_state(state)
                    if include_diagnostics:
                        next_diagnostics = (
                            cycle_start + self.diagnostics_period_s
                        )

                    remaining = self.poll_period_s - (
                        self._monotonic() - cycle_start
                    )
                    if remaining > 0.0 and self._stop_event.wait(remaining):
                        break
            except RM75InterfaceError as exc:
                if not self._stop_event.is_set():
                    self._set_transport(False, str(exc))
            finally:
                self._client.close()

            if not self._stop_event.is_set():
                self._stop_event.wait(self.reconnect_period_s)

        self._set_transport(False, self._last_error)


class RM75UDPStateReceiver:
    """Passive listener for an already-configured Gen-4 UDP state stream.

    The receiver never transmits a datagram and never changes controller UDP
    configuration.  One receiver/port should be configured for each arm.
    """

    def __init__(
        self,
        side: str,
        bind_host: str,
        bind_port: int,
        expected_source_ip: str,
        timeout_s: float = 0.25,
        stale_timeout_s: float = 0.25,
        joint_speed_scale_rad_s: float = GEN4_JOINT_SPEED_TO_RAD_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure a passive receiver for one controller and UDP port."""
        self.side = _validate_side(side)
        if not 1 <= int(bind_port) <= 65535:
            raise ValueError("bind_port must be in [1, 65535]")
        if timeout_s <= 0.0 or stale_timeout_s <= 0.0:
            raise ValueError("UDP timeouts must be positive")
        self.bind_host = str(bind_host)
        self.bind_port = int(bind_port)
        self.expected_source_ip = str(expected_source_ip)
        self.timeout_s = float(timeout_s)
        self.stale_timeout_s = float(stale_timeout_s)
        self.joint_speed_scale_rad_s = float(joint_speed_scale_rad_s)
        if (
            not math.isfinite(self.joint_speed_scale_rad_s)
            or self.joint_speed_scale_rad_s <= 0.0
        ):
            raise ValueError(
                "joint_speed_scale_rad_s must be finite and positive"
            )
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        self._listening = False
        self._state: Optional[RM75ArmState] = None
        self._last_error: Optional[str] = None
        self._measurement_seq = 0
        self._last_measurement_receive: Optional[float] = None

    def start(self) -> None:
        """Start listening without sending or configuring any datagram."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"rm75-{self.side}-udp-state",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 2.0) -> None:
        """Stop listening and release the UDP port."""
        self._stop_event.set()
        sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
        self._thread = None

    def get_status(self) -> RM75StateStatus:
        """Return listener health and the latest validated datagram."""
        with self._lock:
            listening = self._listening
            state = self._state
            error = self._last_error
        stale = state is None or state.is_stale(
            self.stale_timeout_s, self._monotonic()
        )
        return RM75StateStatus(listening, state, stale, error)

    def accept_datagram(
        self,
        payload: bytes,
        address: Tuple[str, int],
        received_monotonic: Optional[float] = None,
    ) -> RM75ArmState:
        """Validate and atomically store one source-authenticated sample."""
        if not isinstance(address, tuple) or len(address) < 2:
            raise RM75ProtocolError("UDP source address is malformed")
        if self.expected_source_ip and address[0] != self.expected_source_ip:
            raise RM75ProtocolError(
                f"unexpected UDP source {address[0]!r}"
            )
        if len(payload) > MAX_JSON_BYTES:
            raise RM75ProtocolError("UDP state datagram exceeds size limit")
        if received_monotonic is None:
            received_monotonic = self._monotonic()
        received_monotonic = _finite_number(
            received_monotonic,
            "received_monotonic",
        )
        period_s = None
        if self._last_measurement_receive is not None:
            elapsed = received_monotonic - self._last_measurement_receive
            if elapsed <= 0.0:
                raise RM75ProtocolError(
                    "UDP measurement receive time did not advance"
                )
            period_s = elapsed
        next_sequence = self._measurement_seq + 1
        state = parse_realtime_udp_state(
            payload,
            self.side,
            received_monotonic,
            joint_speed_scale_rad_s=self.joint_speed_scale_rad_s,
            measurement_seq=next_sequence,
            measurement_period_s=period_s,
        )
        self._measurement_seq = next_sequence
        self._last_measurement_receive = received_monotonic
        with self._lock:
            self._state = state
            self._last_error = None
        return state

    def _run(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket = sock
            sock.settimeout(self.timeout_s)
            sock.bind((self.bind_host, self.bind_port))
            with self._lock:
                self._listening = True
                self._last_error = None

            while not self._stop_event.is_set():
                try:
                    payload, address = sock.recvfrom(MAX_JSON_BYTES + 1)
                except socket.timeout:
                    continue
                try:
                    self.accept_datagram(
                        payload,
                        address,
                        self._monotonic(),
                    )
                except RM75ProtocolError as exc:
                    with self._lock:
                        self._last_error = str(exc)
                    continue
        except OSError as exc:
            if not self._stop_event.is_set():
                with self._lock:
                    self._last_error = f"UDP listener failed: {exc}"
        finally:
            with self._lock:
                self._listening = False
            sock, self._socket = self._socket, None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
