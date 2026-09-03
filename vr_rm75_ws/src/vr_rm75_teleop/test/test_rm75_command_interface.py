"""Tests for the default-disabled RM75 low-follow command boundary."""

import json
import socket

import numpy as np
import pytest

from vr_rm75_teleop.rm75_command_interface import (
    DualArmCommandDispatcher,
    RM75CommandConnectionError,
    RM75CommandResponseTimeout,
    RM75CommandRejectedError,
    RM75ControllerCommandError,
    RM75LowFollowCommandClient,
    RM75MotionDisabledError,
    decode_stop_response,
    decode_movej_canfd_response,
    encode_stop_request,
    encode_low_follow_movej_canfd,
)
from vr_rm75_teleop.stop_policy import StopClass


Q_DEG = {
    "left": [-40.0, -25.0, 15.0, -55.0, 10.0, -35.0, 80.0],
    "right": [20.0, 35.0, 25.0, 60.0, 15.0, 40.0, -120.0],
}
Q = {side: np.deg2rad(values) for side, values in Q_DEG.items()}


class FakeSocket:
    """Record socket operations without touching the network."""

    def __init__(
        self,
        fail_send=False,
        response=None,
        *,
        auto_stop_ack=False,
    ):
        """Configure an optional synthetic send failure."""
        self.fail_send = fail_send
        self.auto_stop_ack = auto_stop_ack
        self.timeout = None
        self.timeout_history = []
        self.payloads = []
        self.events = []
        self.closed = False
        if response is None:
            response = {
                "state": "joint_state",
                "joint": [0] * 7,
                "arm_err": 0,
            }
        self.receive_chunks = [
            json.dumps(response).encode("ascii") + b"\r\n"
        ]

    def settimeout(self, timeout_s):
        """Record the requested timeout."""
        self.timeout = timeout_s
        self.timeout_history.append(timeout_s)

    def sendall(self, payload):
        """Record a full payload or raise the configured failure."""
        command = json.loads(payload)["command"]
        self.events.append(f"send:{command}")
        if self.fail_send:
            raise OSError("synthetic send failure")
        self.payloads.append(payload)
        if self.auto_stop_ack and command in (
            "set_arm_slow_stop",
            "set_arm_stop",
        ):
            stop_class = (
                StopClass.CONTROLLED_STOP
                if command == "set_arm_slow_stop"
                else StopClass.SAFETY_STOP
            )
            self.queue_response(stop_response(stop_class))

    def recv(self, _size):
        """Return one synthetic response or time out."""
        if not self.receive_chunks:
            self.events.append("recv:timeout")
            raise socket.timeout("synthetic response timeout")
        chunk = self.receive_chunks.pop(0)
        if isinstance(chunk, BaseException):
            self.events.append(f"recv:error:{type(chunk).__name__}")
            raise chunk
        if not chunk:
            self.events.append("recv:eof")
            return chunk
        response = json.loads(chunk)
        self.events.append(
            f"recv:{response.get('command', response.get('state'))}"
        )
        return chunk

    def close(self):
        """Record transport closure."""
        self.closed = True
        self.events.append("close")

    def queue_response(self, response):
        """Append one newline-framed synthetic controller response."""
        self.receive_chunks.append(
            json.dumps(response).encode("ascii") + b"\r\n"
        )


def stop_response(stop_class, acknowledged=True):
    """Return the documented response fields for one software stop."""
    if stop_class == StopClass.CONTROLLED_STOP:
        return {
            "command": "set_arm_slow_stop",
            "arm_slow_stop": acknowledged,
        }
    return {"command": "set_arm_stop", "arm_stop": acknowledged}


def client(
    side,
    fake_socket,
    enabled=True,
    *,
    movej_response_timeout_s=0.05,
    stop_response_timeout_s=0.01,
):
    """Build one client around a network-free socket factory."""
    return RM75LowFollowCommandClient(
        side,
        f"{side}.invalid",
        enable_robot_motion=enabled,
        movej_response_timeout_s=movej_response_timeout_s,
        stop_response_timeout_s=stop_response_timeout_s,
        socket_factory=lambda _address, _timeout: fake_socket,
    )


def connected_dispatcher(
    *,
    left_socket=None,
    right_socket=None,
    velocity_limit_deg_s=18.0,
    acceleration_limit_deg_s2=10000.0,
    max_delta_deg=0.5,
    movej_response_mode="send_only",
):
    """Build and connect an enabled dispatcher using fake transports."""
    sockets = {
        "left": left_socket or FakeSocket(),
        "right": right_socket or FakeSocket(),
    }
    clients = {
        side: client(side, sockets[side]) for side in ("left", "right")
    }
    dispatcher = DualArmCommandDispatcher(
        clients,
        {
            side: np.deg2rad(np.full(7, velocity_limit_deg_s))
            for side in ("left", "right")
        },
        joint_acceleration_limits={
            side: np.deg2rad(
                np.full(7, acceleration_limit_deg_s2)
            )
            for side in ("left", "right")
        },
        enable_robot_motion=True,
        max_command_delta_rad=np.deg2rad(max_delta_deg),
        command_timeout_s=0.1,
        nominal_period_s=0.02,
        movej_response_mode=movej_response_mode,
        monotonic=lambda: 1.01,
    )
    dispatcher.connect()
    return dispatcher, sockets


def test_encoder_uses_integer_millidegrees_and_forces_low_follow():
    """Make high-follow mode impossible at the current 20 ms period."""
    q_deg = [1.0, -2.0, 0.001, 4.25, -5.5, 6.0, -7.0]

    payload = encode_low_follow_movej_canfd(
        np.deg2rad(q_deg),
        "left",
    )
    message = json.loads(payload.decode("ascii"))

    assert payload.endswith(b"\r\n")
    assert message["command"] == "movej_canfd"
    assert message["joint"] == [1000, -2000, 1, 4250, -5500, 6000, -7000]
    assert message["follow"] is False
    assert message["trajectory_mode"] == 0


def test_controller_response_requires_zero_arm_error_and_valid_joints():
    """Accept only the documented successful seven-axis response."""
    response = decode_movej_canfd_response(
        json.dumps(
            {
                "state": "joint_state",
                "joint": [1000, 0, 0, 0, 0, 0, 0],
                "arm_err": 0,
            }
        ),
        "left",
    )

    assert response.arm_error == 0
    assert np.rad2deg(response.q_reported[0]) == pytest.approx(1.0)

    with pytest.raises(RM75ControllerCommandError, match="arm_err=7"):
        decode_movej_canfd_response(
            json.dumps(
                {
                    "state": "joint_state",
                    "joint": [0] * 7,
                    "arm_err": 7,
                }
            ),
            "right",
        )


@pytest.mark.parametrize(
    "q, expected",
    [
        (np.zeros(6), "7 values"),
        (np.full(7, np.nan), "NaN or Infinity"),
        (np.deg2rad([179.0, 0, 0, 0, 0, 0, 0]), "hard limits"),
    ],
)
def test_encoder_rejects_malformed_nonfinite_and_hard_limit_commands(
    q,
    expected,
):
    """Reject every target that cannot safely represent an RM75 joint set."""
    with pytest.raises(RM75CommandRejectedError, match=expected):
        encode_low_follow_movej_canfd(q, "right")


def test_disabled_client_cannot_open_a_socket_or_send():
    """Prove the default gate closes before the network factory is called."""
    factory_calls = []
    command_client = RM75LowFollowCommandClient(
        "left",
        "192.0.2.1",
        socket_factory=lambda *args: factory_calls.append(args),
    )

    with pytest.raises(RM75MotionDisabledError):
        command_client.connect()
    with pytest.raises(RM75MotionDisabledError):
        command_client.send_joint_target(Q["left"])

    assert factory_calls == []
    assert not command_client.connected


def test_movej_ack_timeout_preserves_socket_for_safety_stop():
    """Keep a timed-out channel usable until its stop ACK is collected."""
    fake_socket = FakeSocket(auto_stop_ack=True)
    fake_socket.receive_chunks.clear()
    command_client = client(
        "left",
        fake_socket,
        movej_response_timeout_s=0.05,
        stop_response_timeout_s=0.012,
    )
    command_client.connect()
    command_client.send_joint_target(Q["left"])

    with pytest.raises(
        RM75CommandResponseTimeout,
        match=r"timeout receiving left movej_canfd ACK.*0\.050",
    ):
        command_client.receive_joint_result()

    assert command_client.connected
    assert not fake_socket.closed
    command_client.send_stop_request(StopClass.SAFETY_STOP)
    assert command_client.receive_stop_result(StopClass.SAFETY_STOP)
    assert fake_socket.timeout_history[-1] == pytest.approx(0.012)


def test_disabled_dispatcher_returns_dry_run_before_validating_or_sending():
    """Keep malformed upstream data away from transports by default."""
    sockets = {side: FakeSocket() for side in ("left", "right")}
    clients = {
        side: client(side, sockets[side], enabled=False)
        for side in ("left", "right")
    }
    dispatcher = DualArmCommandDispatcher(
        clients,
        {side: np.ones(7) for side in ("left", "right")},
    )

    result = dispatcher.dispatch(
        {"bad": np.full(7, np.nan)},
        {},
        generated_monotonic=None,
        command_dt_s=0.02,
        safety_command_allowed=True,
    )

    assert not result.sent
    assert "dry-run" in result.reason
    assert all(
        not command_client.connected
        for command_client in clients.values()
    )
    assert all(fake_socket.payloads == [] for fake_socket in sockets.values())


@pytest.mark.parametrize(
    "stop_class, command, field",
    [
        (
            StopClass.CONTROLLED_STOP,
            "set_arm_slow_stop",
            "arm_slow_stop",
        ),
        (StopClass.SAFETY_STOP, "set_arm_stop", "arm_stop"),
    ],
)
def test_stop_codec_uses_documented_commands(stop_class, command, field):
    """Keep both software stop classes explicit and protocol-validated."""
    payload = encode_stop_request(stop_class)
    assert json.loads(payload) == {"command": command}
    assert decode_stop_response(
        json.dumps({"command": command, field: True}),
        "left",
        stop_class,
    )
    with pytest.raises(RM75ControllerCommandError):
        decode_stop_response(
            json.dumps({"command": command, field: False}),
            "left",
            stop_class,
        )


def test_dry_run_stop_records_intent_without_network_activity():
    """Report the intended stop while the default-off gate stays closed."""
    sockets = {side: FakeSocket() for side in ("left", "right")}
    clients = {
        side: client(side, sockets[side], enabled=False)
        for side in ("left", "right")
    }
    dispatcher = DualArmCommandDispatcher(
        clients,
        {side: np.ones(7) for side in ("left", "right")},
    )

    result = dispatcher.request_stop(
        StopClass.SAFETY_STOP,
        "synthetic collision event",
        1.0,
    )

    assert result.dry_run
    assert not result.all_acknowledged
    assert all(not arm.attempted for arm in result.arms)
    assert all(not client_.connected for client_ in clients.values())
    assert all(socket_.payloads == [] for socket_ in sockets.values())


@pytest.mark.parametrize(
    "stop_class",
    [StopClass.CONTROLLED_STOP, StopClass.SAFETY_STOP],
)
def test_dual_stop_sends_both_before_waiting_for_both_acks(stop_class):
    """Issue one request per arm and validate both acknowledgements."""
    dispatcher, sockets = connected_dispatcher()
    for socket_ in sockets.values():
        socket_.receive_chunks.clear()
        socket_.queue_response(stop_response(stop_class))

    result = dispatcher.request_stop(stop_class, "synthetic edge", 1.0)

    assert result.all_acknowledged
    assert not result.dry_run
    assert all(arm.attempted and arm.acknowledged for arm in result.arms)
    for socket_ in sockets.values():
        message = json.loads(socket_.payloads[-1])
        expected = (
            "set_arm_slow_stop"
            if stop_class == StopClass.CONTROLLED_STOP
            else "set_arm_stop"
        )
        assert message["command"] == expected


def test_malformed_stop_ack_is_reported_as_incomplete_dual_stop():
    """Fail closed when either controller returns a malformed stop ACK."""
    dispatcher, sockets = connected_dispatcher()
    for socket_ in sockets.values():
        socket_.receive_chunks.clear()
    sockets["left"].queue_response(stop_response(StopClass.SAFETY_STOP))
    sockets["right"].queue_response(
        {"command": "set_arm_stop", "arm_stop": "yes"}
    )

    result = dispatcher.request_stop(
        StopClass.SAFETY_STOP,
        "synthetic malformed acknowledgement",
        1.0,
    )

    assert not result.all_acknowledged
    assert result.arms[0].acknowledged
    assert not result.arms[1].acknowledged
    assert "must be boolean" in result.arms[1].error


def test_send_only_dual_command_sends_once_without_receiving():
    """Treat both successful writes as transport sends, never as ACKs."""
    dispatcher, sockets = connected_dispatcher()
    send_order = []
    for side in ("left", "right"):
        client_ = dispatcher.clients[side]
        send_target = client_.send_joint_target

        def record_send(q, *, _side=side, _send=send_target):
            send_order.append(_side)
            return _send(q)

        client_.send_joint_target = record_send

    result = dispatcher.dispatch(
        Q,
        Q,
        generated_monotonic=1.0,
        command_dt_s=0.02,
        safety_command_allowed=True,
        now_monotonic=1.001,
    )

    assert result.sent
    assert not result.response_expected
    assert result.ack_latency_s is None
    assert result.feedback_supervised
    assert "physical target attainment not implied" in result.reason
    assert send_order == ["left", "right"]
    assert all(
        not any(event.startswith("recv:") for event in fake_socket.events)
        for fake_socket in sockets.values()
    )
    assert all(
        len(fake_socket.payloads) == 1
        for fake_socket in sockets.values()
    )
    for side in ("left", "right"):
        message = json.loads(sockets[side].payloads[0].decode("ascii"))
        assert message["follow"] is False
        assert message["joint"] == [
            int(round(value * 1000)) for value in Q_DEG[side]
        ]

    repeated = dispatcher.dispatch(
        Q,
        Q,
        generated_monotonic=1.0,
        command_dt_s=0.02,
        safety_command_allowed=True,
        now_monotonic=1.002,
    )
    assert not repeated.sent
    assert "already dispatched" in repeated.reason
    assert all(
        len(fake_socket.payloads) == 1
        for fake_socket in sockets.values()
    )


def test_optional_joint_state_ack_mode_still_validates_responses():
    """Retain the old response path only for explicitly compatible firmware."""
    dispatcher, sockets = connected_dispatcher(
        movej_response_mode="joint_state_ack"
    )

    result = dispatcher.dispatch(
        Q,
        Q,
        generated_monotonic=1.0,
        command_dt_s=0.02,
        safety_command_allowed=True,
        now_monotonic=1.001,
    )

    assert result.sent
    assert result.response_expected
    assert result.ack_latency_s is not None
    assert result.feedback_supervised
    assert "response_expected=true" in result.reason
    assert all(
        "recv:joint_state" in socket_.events
        for socket_ in sockets.values()
    )


def test_closed_safety_gate_disarms_without_sending():
    """Require the independent Supervisor command gate for every dispatch."""
    dispatcher, sockets = connected_dispatcher()

    result = dispatcher.dispatch(
        Q,
        Q,
        generated_monotonic=1.0,
        command_dt_s=0.02,
        safety_command_allowed=False,
        now_monotonic=1.0,
    )

    assert not result.sent
    assert all(fake_socket.payloads == [] for fake_socket in sockets.values())


@pytest.mark.parametrize(
    "generated, now, expected",
    [
        (None, 1.0, "timestamp is unavailable"),
        (1.0, 1.101, "stale"),
        (float("nan"), 1.0, "timestamp must be finite"),
    ],
)
def test_missing_stale_or_nonfinite_command_timestamp_is_rejected(
    generated,
    now,
    expected,
):
    """Reject output that is missing or older than its watchdog budget."""
    dispatcher, sockets = connected_dispatcher()

    with pytest.raises(RM75CommandRejectedError, match=expected):
        dispatcher.dispatch(
            Q,
            Q,
            generated_monotonic=generated,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=now,
        )

    assert all(fake_socket.payloads == [] for fake_socket in sockets.values())


def test_dual_numeric_validation_finishes_before_either_write():
    """Reject a bad peer target before writing the otherwise valid arm."""
    dispatcher, sockets = connected_dispatcher()
    invalid = {side: values.copy() for side, values in Q.items()}
    invalid["right"][0] = np.nan

    with pytest.raises(RM75CommandRejectedError):
        dispatcher.dispatch(
            invalid,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    assert all(fake_socket.payloads == [] for fake_socket in sockets.values())


def test_abnormal_command_jump_is_rejected_instead_of_clipped():
    """Treat a discontinuous branch jump as HOLD-worthy rejection."""
    dispatcher, sockets = connected_dispatcher(
        velocity_limit_deg_s=100.0,
        max_delta_deg=0.5,
    )
    jumped = {side: values.copy() for side, values in Q.items()}
    jumped["left"][0] += np.deg2rad(0.6)

    with pytest.raises(RM75CommandRejectedError, match="jump"):
        dispatcher.dispatch(
            jumped,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    assert all(fake_socket.payloads == [] for fake_socket in sockets.values())


def test_command_velocity_is_checked_independently_of_jump_limit():
    """Enforce qdot even when a point passes the absolute-delta guard."""
    dispatcher, sockets = connected_dispatcher(
        velocity_limit_deg_s=1.0,
        max_delta_deg=1.0,
    )
    too_fast = {side: values.copy() for side, values in Q.items()}
    too_fast["right"][2] += np.deg2rad(0.2)

    with pytest.raises(RM75CommandRejectedError, match="velocity"):
        dispatcher.dispatch(
            too_fast,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    assert all(fake_socket.payloads == [] for fake_socket in sockets.values())


def test_command_acceleration_is_checked_at_final_send_boundary():
    """Reject qdd that bypasses the online trajectory limiter."""
    dispatcher, sockets = connected_dispatcher(
        velocity_limit_deg_s=100.0,
        acceleration_limit_deg_s2=100.0,
        max_delta_deg=1.0,
    )
    dispatcher.dispatch(
        Q,
        Q,
        generated_monotonic=1.0,
        command_dt_s=0.02,
        safety_command_allowed=True,
        now_monotonic=1.0,
    )
    accelerated = {side: values.copy() for side, values in Q.items()}
    accelerated["left"][1] += np.deg2rad(0.1)

    with pytest.raises(RM75CommandRejectedError, match="LEFT J2 qddot"):
        dispatcher.dispatch(
            accelerated,
            Q,
            generated_monotonic=1.02,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.02,
        )

    assert all(
        len(fake_socket.payloads) == 1
        for fake_socket in sockets.values()
    )


def test_zero_motion_prime_initializes_exact_zero_continuity():
    """Seed dispatcher history only by a fully validated measured target."""
    dispatcher, sockets = connected_dispatcher(
        velocity_limit_deg_s=18.0,
        acceleration_limit_deg_s2=30.0,
    )

    result = dispatcher.prime_zero_motion(
        Q,
        Q,
        generated_monotonic=1.0,
        command_dt_s=0.02,
        safety_command_allowed=True,
        now_monotonic=1.0,
    )

    assert result.sent
    assert dispatcher._last_generated_monotonic == pytest.approx(1.0)
    assert dispatcher.last_send_monotonic == pytest.approx(1.01)
    for side in ("left", "right"):
        assert np.array_equal(dispatcher._last_sent_q[side], Q[side])
        assert np.array_equal(
            dispatcher._last_sent_velocity[side],
            np.zeros(7),
        )
        assert len(sockets[side].payloads) == 1


def test_canonical_dt_passes_fusion_limit_despite_timestamp_spacing():
    """Use Fusion's 20 ms interval despite a 17 ms timestamp spacing."""
    dispatcher, _sockets = connected_dispatcher(
        velocity_limit_deg_s=18.0,
        acceleration_limit_deg_s2=30.0,
    )
    dispatcher.prime_zero_motion(
        Q,
        Q,
        generated_monotonic=1.0,
        command_dt_s=0.02,
        safety_command_allowed=True,
        now_monotonic=1.0,
    )
    measured_after_prime = {
        side: values + np.deg2rad(0.010)
        for side, values in Q.items()
    }
    next_command = {side: values.copy() for side, values in Q.items()}
    next_command["left"][0] += np.deg2rad(0.012)

    result = dispatcher.dispatch(
        next_command,
        measured_after_prime,
        generated_monotonic=1.017,
        command_dt_s=0.02,
        safety_command_allowed=True,
        now_monotonic=1.017,
    )

    assert result.sent
    assert result.command_dt_s == pytest.approx(0.02)
    assert np.rad2deg(dispatcher._last_sent_velocity["left"][0]) == (
        pytest.approx(0.6)
    )
    assert np.array_equal(
        dispatcher._last_sent_q["right"],
        Q["right"],
    )


def test_true_post_prime_acceleration_excess_has_detailed_diagnostics():
    """Do not weaken the independent 30 deg/s^2 actuator boundary."""
    dispatcher, sockets = connected_dispatcher(
        velocity_limit_deg_s=18.0,
        acceleration_limit_deg_s2=30.0,
    )
    dispatcher.prime_zero_motion(
        Q,
        Q,
        generated_monotonic=1.0,
        command_dt_s=0.02,
        safety_command_allowed=True,
        now_monotonic=1.0,
    )
    accelerated = {side: values.copy() for side, values in Q.items()}
    accelerated["left"][0] += np.deg2rad(0.013)

    with pytest.raises(RM75CommandRejectedError) as caught:
        dispatcher.dispatch(
            accelerated,
            Q,
            generated_monotonic=1.02,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.02,
        )

    message = str(caught.value)
    assert "LEFT J1" in message
    assert "qddot=" in message
    assert "> 30.000000 deg/s2" in message
    assert "command_dt_s=0.020000000" in message
    assert "delta_deg=" in message
    assert "qdot_deg_s=" in message
    assert "previous_qdot_deg_s=" in message
    assert "qddot_deg_s2=" in message
    assert "limit_deg_s2=30.000000" in message
    assert all(len(sock.payloads) == 1 for sock in sockets.values())


def test_second_and_third_commands_keep_canonical_velocity_history():
    """Carry qdot continuously across several canonical-dt commands."""
    dispatcher, _sockets = connected_dispatcher(
        velocity_limit_deg_s=18.0,
        acceleration_limit_deg_s2=30.0,
    )
    dispatcher.prime_zero_motion(
        Q,
        Q,
        generated_monotonic=1.0,
        command_dt_s=0.02,
        safety_command_allowed=True,
        now_monotonic=1.0,
    )
    cumulative_positions_deg = (0.012, 0.036, 0.060)
    expected_velocities_deg_s = (0.6, 1.2, 1.2)
    generated_times = (1.017, 1.035, 1.054)

    for position_deg, expected_qdot, generated in zip(
        cumulative_positions_deg,
        expected_velocities_deg_s,
        generated_times,
    ):
        command = {side: values.copy() for side, values in Q.items()}
        command["left"][0] += np.deg2rad(position_deg)
        result = dispatcher.dispatch(
            command,
            Q,
            generated_monotonic=generated,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=generated,
        )
        assert result.sent
        assert np.rad2deg(
            dispatcher._last_sent_velocity["left"][0]
        ) == pytest.approx(expected_qdot)


@pytest.mark.parametrize(
    "command_dt_s",
    [None, np.nan, np.inf, 0.0, -0.01, 0.020001],
)
def test_invalid_command_dt_is_rejected(command_dt_s):
    """Reject missing, non-finite, non-positive, or over-period dt."""
    dispatcher, sockets = connected_dispatcher(
        acceleration_limit_deg_s2=30.0,
    )

    with pytest.raises(RM75CommandRejectedError, match="command_dt_s"):
        dispatcher.dispatch(
            Q,
            Q,
            generated_monotonic=1.0,
            command_dt_s=command_dt_s,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    assert all(sock.payloads == [] for sock in sockets.values())


def test_prime_rejects_any_target_measurement_delta_before_send():
    """Make equality to fresh measured state an explicit PRIME invariant."""
    dispatcher, sockets = connected_dispatcher(
        acceleration_limit_deg_s2=30.0,
    )
    nonzero = {side: values.copy() for side, values in Q.items()}
    nonzero["right"][6] += 1e-12

    with pytest.raises(RM75CommandRejectedError, match="exactly equal"):
        dispatcher.prime_zero_motion(
            nonzero,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    assert all(sock.payloads == [] for sock in sockets.values())


def test_prime_send_failure_uses_existing_dual_safety_stop_path():
    """Retain fail-closed two-arm stop handling during PRIME transport."""
    left_socket = FakeSocket(auto_stop_ack=True)
    right_socket = FakeSocket(fail_send=True)
    dispatcher, _sockets = connected_dispatcher(
        left_socket=left_socket,
        right_socket=right_socket,
        acceleration_limit_deg_s2=30.0,
    )

    with pytest.raises(RM75CommandConnectionError, match="failed sending"):
        dispatcher.prime_zero_motion(
            Q,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    assert dispatcher.faulted
    assert left_socket.closed and right_socket.closed
    assert "send:set_arm_stop" in left_socket.events
    assert "send:set_arm_stop" in right_socket.events


def test_partial_transport_failure_closes_both_and_latches_fault():
    """Latch the unavoidable asymmetric network failure as a hard fault."""
    left_socket = FakeSocket(auto_stop_ack=True)
    right_socket = FakeSocket(fail_send=True)
    dispatcher, _sockets = connected_dispatcher(
        left_socket=left_socket,
        right_socket=right_socket,
    )

    with pytest.raises(RM75CommandConnectionError, match="failed sending"):
        dispatcher.dispatch(
            Q,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    assert dispatcher.faulted
    assert not dispatcher.connected
    assert left_socket.closed
    assert right_socket.closed
    assert len(left_socket.payloads) == 2
    assert json.loads(left_socket.payloads[-1])["command"] == "set_arm_stop"
    assert "send:set_arm_stop" in left_socket.events
    assert "send:set_arm_stop" in right_socket.events
    arms = {arm.side: arm for arm in dispatcher.last_stop_result.arms}
    assert arms["left"].acknowledged
    assert arms["right"].attempted
    assert not arms["right"].acknowledged


def test_controller_error_response_closes_both_and_latches_fault():
    """Reject a nonzero arm_err even when both TCP writes succeeded."""
    right_socket = FakeSocket(
        response={
            "state": "joint_state",
            "joint": [0] * 7,
            "arm_err": 23,
        }
    )
    dispatcher, sockets = connected_dispatcher(
        right_socket=right_socket,
        movej_response_mode="joint_state_ack",
    )

    with pytest.raises(RM75ControllerCommandError, match="arm_err=23"):
        dispatcher.dispatch(
            Q,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    assert all(fake_socket.closed for fake_socket in sockets.values())
    assert dispatcher.faulted


def test_movej_timeout_stops_both_then_closes_and_latches_fault():
    """Stop both retained channels before closing after a left ACK timeout."""
    left_socket = FakeSocket(auto_stop_ack=True)
    left_socket.receive_chunks.clear()
    right_socket = FakeSocket(auto_stop_ack=True)
    dispatcher, sockets = connected_dispatcher(
        left_socket=left_socket,
        right_socket=right_socket,
        movej_response_mode="joint_state_ack",
    )

    with pytest.raises(
        RM75CommandResponseTimeout,
        match="SAFETY_STOP acknowledged by both arms",
    ):
        dispatcher.dispatch(
            Q,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    assert all(fake_socket.closed for fake_socket in sockets.values())
    assert dispatcher.faulted
    assert dispatcher.last_stop_result.all_acknowledged
    for fake_socket in sockets.values():
        commands = [
            json.loads(payload)["command"]
            for payload in fake_socket.payloads
        ]
        assert commands == ["movej_canfd", "set_arm_stop"]
        assert (
            fake_socket.events.index("recv:set_arm_stop")
            < fake_socket.events.index("close")
        )

    with pytest.raises(RM75CommandConnectionError, match="latched"):
        dispatcher.dispatch(
            Q,
            Q,
            generated_monotonic=1.02,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.02,
        )


def test_stop_ack_timeout_after_movej_failure_still_latches_fault():
    """Treat a missing stop ACK as incomplete and require physical E-stop."""
    left_socket = FakeSocket()
    left_socket.receive_chunks.clear()
    right_socket = FakeSocket(auto_stop_ack=True)
    dispatcher, sockets = connected_dispatcher(
        left_socket=left_socket,
        right_socket=right_socket,
        movej_response_mode="joint_state_ack",
    )

    with pytest.raises(
        RM75CommandResponseTimeout,
        match="stop incomplete; physical E-stop required",
    ):
        dispatcher.dispatch(
            Q,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    arms = {arm.side: arm for arm in dispatcher.last_stop_result.arms}
    assert arms["left"].attempted
    assert not arms["left"].acknowledged
    assert "set_arm_stop ACK" in arms["left"].error
    assert arms["right"].acknowledged
    assert all(socket_.closed for socket_ in sockets.values())
    assert dispatcher.faulted


@pytest.mark.parametrize(
    "failure",
    [b"", ConnectionResetError("synthetic reset")],
    ids=["eof", "connection_reset"],
)
def test_broken_side_reports_physical_estop_and_peer_still_stops(failure):
    """Close an unavailable left channel but still stop and ACK the right."""
    left_socket = FakeSocket(auto_stop_ack=True)
    left_socket.receive_chunks = [failure]
    right_socket = FakeSocket(auto_stop_ack=True)
    dispatcher, sockets = connected_dispatcher(
        left_socket=left_socket,
        right_socket=right_socket,
        movej_response_mode="joint_state_ack",
    )

    with pytest.raises(
        RM75CommandConnectionError,
        match="stop incomplete; physical E-stop required",
    ):
        dispatcher.dispatch(
            Q,
            Q,
            generated_monotonic=1.0,
            command_dt_s=0.02,
            safety_command_allowed=True,
            now_monotonic=1.0,
        )

    arms = {arm.side: arm for arm in dispatcher.last_stop_result.arms}
    assert not arms["left"].attempted
    assert not arms["left"].acknowledged
    assert "physical E-stop required" in arms["left"].error
    assert arms["right"].attempted
    assert arms["right"].acknowledged
    left_commands = [
        json.loads(payload)["command"]
        for payload in left_socket.payloads
    ]
    right_commands = [
        json.loads(payload)["command"]
        for payload in right_socket.payloads
    ]
    assert left_commands == ["movej_canfd"]
    assert right_commands == [
        "movej_canfd",
        "set_arm_stop",
    ]
    assert all(fake_socket.closed for fake_socket in sockets.values())
    assert dispatcher.faulted


def test_connect_failure_closes_an_already_connected_peer():
    """Close the first arm when the second command channel cannot connect."""
    left_socket = FakeSocket()
    left_client = client("left", left_socket)

    def fail_connect(_address, _timeout):
        raise OSError("synthetic connect failure")

    right_client = RM75LowFollowCommandClient(
        "right",
        "right.invalid",
        enable_robot_motion=True,
        socket_factory=fail_connect,
    )
    dispatcher = DualArmCommandDispatcher(
        {"left": left_client, "right": right_client},
        {side: np.ones(7) for side in ("left", "right")},
        joint_acceleration_limits={
            side: np.ones(7) for side in ("left", "right")
        },
        enable_robot_motion=True,
    )

    with pytest.raises(RM75CommandConnectionError):
        dispatcher.connect()

    assert dispatcher.faulted
    assert left_socket.closed
    assert not dispatcher.connected
