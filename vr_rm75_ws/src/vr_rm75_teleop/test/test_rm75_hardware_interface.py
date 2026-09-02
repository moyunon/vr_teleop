"""Offline tests for the read-only RM75 state interface."""

import json
import socket
import threading
import time

import numpy as np
import pytest

from vr_rm75_teleop.rm75_hardware_interface import (
    GET_CONTROLLER_STATE,
    GET_CURRENT_ARM_STATE,
    GET_JOINT_ENABLE_STATE,
    GET_JOINT_ERROR_FLAG,
    READ_ONLY_COMMANDS,
    RM75ConnectionError,
    RM75ProtocolError,
    RM75ReadOnlyClient,
    RM75StateWorker,
    RM75TimeoutError,
    parse_realtime_udp_state,
)


JOINT_MILLI_DEGREES = [1000, -2000, 3000, 4000, -5000, 6000, -7000]


def response_for(command):
    """Return one valid controller response for a read-only query."""
    if command == GET_CURRENT_ARM_STATE:
        return {
            "command": command,
            "arm_state": {
                "joint": JOINT_MILLI_DEGREES[:],
                "pose": [0] * 6,
                "err": 0,
            },
        }
    if command == GET_CONTROLLER_STATE:
        return {"command": command, "err_flag": 0}
    if command == GET_JOINT_ENABLE_STATE:
        return {"command": command, "en_state": [1] * 7}
    if command == GET_JOINT_ERROR_FLAG:
        return {
            "command": command,
            "err_flag": [0] * 7,
            "brake_state": [1] * 7,
        }
    raise AssertionError(f"unexpected test command: {command}")


class ResponsiveSocket:
    """Small socket double that creates one response for each sent query."""

    def __init__(self, override=None):
        """Create a responsive socket, optionally overriding replies."""
        self.override = override
        self.pending = []
        self.sent = []
        self.timeout = None
        self.closed = False
        self.lock = threading.Lock()

    def settimeout(self, timeout):
        """Record the timeout requested by the client."""
        self.timeout = timeout

    def sendall(self, payload):
        """Record a command and enqueue its simulated response."""
        if self.closed:
            raise OSError("closed")
        command = json.loads(payload.decode("ascii"))["command"]
        with self.lock:
            self.sent.append(payload)
            item = (
                self.override(command)
                if self.override
                else response_for(command)
            )
            if isinstance(item, dict):
                item = json.dumps(item).encode("utf-8") + b"\r\n"
            self.pending.append(item)

    def recv(self, size):
        """Return the next simulated response or transport exception."""
        with self.lock:
            if not self.pending:
                raise socket.timeout("no response")
            item = self.pending.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        """Mark this test socket closed."""
        self.closed = True


def client_for(sock, monotonic=lambda: 10.0):
    """Build a client around a supplied socket double."""
    return RM75ReadOnlyClient(
        "192.0.2.1",
        timeout_s=0.01,
        socket_factory=lambda address, timeout: sock,
        monotonic=monotonic,
    )


def test_normal_tcp_state_is_scaled_validated_and_read_only():
    """Parse normal state while sending only correctly framed getters."""
    sock = ResponsiveSocket()
    client = client_for(sock)
    client.connect()

    state = client.read_state("left")

    expected_degrees = np.arange(1, 8) * [1, -1, 1, 1, -1, 1, -1]
    assert np.allclose(state.q_measured, np.deg2rad(expected_degrees))
    assert state.arm_error == 0
    assert state.controller_error == 0
    assert state.joint_enabled == (True,) * 7
    assert state.joint_errors == (0,) * 7
    assert state.brake_released == (True,) * 7
    assert state.received_monotonic == 10.0
    assert state.source == "tcp"
    assert state.all_joints_enabled is True
    assert state.has_fault is False

    commands = [json.loads(payload)["command"] for payload in sock.sent]
    assert commands == [
        GET_CURRENT_ARM_STATE,
        GET_JOINT_ENABLE_STATE,
        GET_JOINT_ERROR_FLAG,
        GET_CONTROLLER_STATE,
    ]
    assert all(command.startswith("get_") for command in READ_ONLY_COMMANDS)
    assert all(payload.endswith(b"\r\n") for payload in sock.sent)


def test_non_read_only_command_is_rejected_before_send():
    """Reject a motion command before it reaches the socket."""
    sock = ResponsiveSocket()
    client = client_for(sock)
    client.connect()

    with pytest.raises(RM75ProtocolError, match="not read-only"):
        client._send_query("movej")

    assert sock.sent == []


def test_timeout_is_reported():
    """Translate a socket receive timeout into interface status."""
    sock = ResponsiveSocket(override=lambda command: socket.timeout("test"))
    client = client_for(sock)
    client.connect()

    with pytest.raises(RM75TimeoutError, match="timeout"):
        client.read_state("right")


def test_malformed_json_is_rejected():
    """Reject malformed controller JSON instead of retaining it as state."""
    sock = ResponsiveSocket(override=lambda command: b"{not-json}\r\n")
    client = client_for(sock)
    client.connect()

    with pytest.raises(RM75ProtocolError, match="invalid JSON"):
        client.read_state("left")


@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), -float("inf")]
)
def test_nan_and_infinity_joint_values_are_rejected(bad_value):
    """Reject every non-finite joint representation."""
    def override(command):
        response = response_for(command)
        if command == GET_CURRENT_ARM_STATE:
            response["arm_state"]["joint"][3] = bad_value
        return response

    client = client_for(ResponsiveSocket(override=override))
    client.connect()

    error_pattern = "NaN or Infinity|invalid JSON"
    with pytest.raises(RM75ProtocolError, match=error_pattern):
        client.read_state("left")


def test_out_of_range_and_wrong_dof_joint_data_are_rejected():
    """Reject unsafe hard-limit values and non-RM75 vector lengths."""
    def too_large(command):
        response = response_for(command)
        if command == GET_CURRENT_ARM_STATE:
            response["arm_state"]["joint"][3] = 136000
        return response

    client = client_for(ResponsiveSocket(override=too_large))
    client.connect()
    with pytest.raises(RM75ProtocolError, match="outside RM75 joint 4 limits"):
        client.read_state("left")

    def six_axes(command):
        response = response_for(command)
        if command == GET_CURRENT_ARM_STATE:
            response["arm_state"]["joint"] = [0] * 6
        return response

    client = client_for(ResponsiveSocket(override=six_axes))
    client.connect()
    with pytest.raises(RM75ProtocolError, match="exactly 7"):
        client.read_state("right")


def test_disconnect_is_reported():
    """Treat an EOF response as a controller disconnect."""
    sock = ResponsiveSocket(override=lambda command: b"")
    client = client_for(sock)
    client.connect()

    with pytest.raises(RM75ConnectionError, match="closed"):
        client.read_state("left")


def test_stale_deadline_uses_receive_monotonic_time():
    """Determine freshness from receive time rather than publish time."""
    client = client_for(ResponsiveSocket(), monotonic=lambda: 12.5)
    client.connect()
    state = client.read_state("right")

    assert state.is_stale(0.25, 12.74) is False
    assert state.is_stale(0.25, 12.76) is True


def test_udp_realtime_state_parser():
    """Parse and scale a seven-axis Gen-4 realtime UDP sample."""
    payload = {
        "state": "realtime_arm_joint_state",
        "arm_current_status": "idle",
        "err": 0,
        "joint_status": {
            "joint_position": JOINT_MILLI_DEGREES,
            "joint_en_flag": [1] * 7,
            "joint_err_code": [0, 0, 0, 4, 0, 0, 0],
        },
    }

    state = parse_realtime_udp_state(payload, "right", 8.0)

    assert state.source == "udp"
    assert state.arm_motion_state == "idle"
    assert state.joint_errors[3] == 4
    assert state.has_fault is True
    assert np.allclose(state.q_measured, np.deg2rad([1, -2, 3, 4, -5, 6, -7]))


def test_worker_reconnects_after_disconnect_and_recovers_state():
    """Reconnect automatically after EOF and expose the recovered sample."""
    calls = []

    def factory(address, timeout):
        if not calls:
            sock = ResponsiveSocket(override=lambda command: b"")
        else:
            sock = ResponsiveSocket()
        calls.append(sock)
        return sock

    worker = RM75StateWorker(
        side="left",
        host="192.0.2.1",
        timeout_s=0.01,
        poll_period_s=0.01,
        diagnostics_period_s=0.02,
        reconnect_period_s=0.01,
        stale_timeout_s=0.1,
        socket_factory=factory,
    )
    worker.start()
    deadline = time.monotonic() + 1.0
    try:
        while time.monotonic() < deadline:
            status = worker.get_status()
            if (
                status.connected
                and status.state is not None
                and not status.stale
            ):
                break
            time.sleep(0.005)
        else:
            pytest.fail("worker did not reconnect and recover a valid state")

        assert len(calls) >= 2
        assert status.last_error is None
        assert status.state.side == "left"
    finally:
        worker.stop()
