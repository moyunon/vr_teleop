"""Selection tests for the read-only RM75 ROS state publisher."""

import json

from vr_rm75_teleop.rm75_hardware_interface import (
    RM75StateStatus,
    parse_realtime_udp_state,
)
from vr_rm75_teleop.rm75_state_node import RM75StateNode


def udp_state(received_monotonic=1.0):
    """Create one valid direct-velocity UDP measurement."""
    payload = {
        "state": "realtime_arm_joint_state",
        "arm_current_status": "idle",
        "err": 0,
        "joint_status": {
            "joint_position": [0] * 7,
            "joint_en_flag": [1] * 7,
            "joint_err_code": [0] * 7,
            "joint_speed": [0] * 7,
        },
    }
    return parse_realtime_udp_state(
        json.dumps(payload),
        "left",
        received_monotonic,
        measurement_seq=1,
    )


class StaticReceiver:
    """Expose a fixed receiver status to the source-selection method."""

    def __init__(self, status):
        """Store the immutable test status."""
        self.status = status

    def get_status(self):
        """Return the configured status."""
        return self.status


def test_fresh_udp_is_preferred_over_newer_tcp_for_direct_velocity():
    """Use UDP while valid instead of selecting by ROS publication time."""
    udp = udp_state(1.0)
    tcp = udp_state(2.0)
    tcp = type(tcp)(
        **{
            **tcp.__dict__,
            "source": "tcp",
            "qd_measured": None,
            "velocity_source": "unavailable",
        }
    )
    node = object.__new__(RM75StateNode)
    node._udp_receivers = {
        "left": StaticReceiver(RM75StateStatus(True, udp, False, None))
    }

    selected, stale = node._select_state(
        "left",
        RM75StateStatus(True, tcp, False, None),
    )

    assert selected is udp
    assert selected.velocity_source == "controller_udp_joint_speed"
    assert stale is False


def test_stale_udp_falls_back_to_fresh_tcp():
    """Retain read-only TCP state when realtime UDP becomes stale."""
    udp = udp_state(1.0)
    tcp = udp_state(2.0)
    node = object.__new__(RM75StateNode)
    node._udp_receivers = {
        "left": StaticReceiver(RM75StateStatus(True, udp, True, "stale"))
    }

    selected, stale = node._select_state(
        "left",
        RM75StateStatus(True, tcp, False, None),
    )

    assert selected is tcp
    assert stale is False
