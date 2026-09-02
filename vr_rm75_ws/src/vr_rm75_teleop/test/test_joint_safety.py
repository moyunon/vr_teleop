"""Tests for deterministic joint-space velocity limiting."""

import math

import numpy as np
import pytest

from vr_rm75_teleop.joint_safety import (
    limit_joint_acceleration,
    limit_joint_soft_position,
    limit_joint_velocity,
    make_teleop_soft_limits,
)
from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.target_feasibility import minimum_singular_value


READY_Q_DEG = {
    "left": [-64.143, -33.259, -0.044, -80.671, 8.438, -47.101, 111.349],
    "right": [21.180, 48.282, 32.467, 74.971, 21.508, 54.389, -158.273],
}


def soft_limits(side, joint_margin_deg=5.0, elbow_margin_deg=15.0):
    """Build the configured RM75 teleoperation interval for one side."""
    model = RM75Model(side)
    return make_teleop_soft_limits(
        model.q_min,
        model.q_max,
        np.deg2rad(joint_margin_deg),
        elbow_index=3,
        elbow_branch=-1 if side == "left" else 1,
        elbow_margin=np.deg2rad(elbow_margin_deg),
    )


def test_soft_limits_are_strictly_inside_every_hard_limit():
    """Keep a real angular margin from all controller hard stops."""
    for side in ("left", "right"):
        model = RM75Model(side)
        lower, upper = soft_limits(side)
        assert np.all(lower > model.q_min)
        assert np.all(upper < model.q_max)


def test_elbow_limits_preserve_opposite_ready_posture_branches():
    """Keep left J4 negative and right J4 positive by at least 15 degrees."""
    left_lower, left_upper = soft_limits("left")
    right_lower, right_upper = soft_limits("right")
    assert np.rad2deg(left_upper[3]) == pytest.approx(-15.0)
    assert np.rad2deg(right_lower[3]) == pytest.approx(15.0)

    for side, ready_deg in READY_Q_DEG.items():
        ready = np.deg2rad(ready_deg)
        lower, upper = soft_limits(side)
        assert np.all(ready >= lower)
        assert np.all(ready <= upper)


def test_elbow_margin_is_grounded_in_ready_posture_sigma_scan():
    """Document why the conservative first-pass elbow margin is 15 degrees."""
    for side, ready_deg in READY_Q_DEG.items():
        model = RM75Model(side)
        q_zero = np.deg2rad(ready_deg)
        q_zero[3] = 0.0
        assert minimum_singular_value(q_zero, model) < 1e-10

        q_margin = np.deg2rad(ready_deg)
        q_margin[3] = np.deg2rad(-15.0 if side == "left" else 15.0)
        assert minimum_singular_value(q_margin, model) > 0.020


def test_soft_limiter_saturates_without_crossing_elbow_zero():
    """Clamp targets at the configured side-specific J4 branch boundary."""
    for side, target_deg, expected_deg in (
        ("left", 30.0, -15.0),
        ("right", -30.0, 15.0),
    ):
        lower, upper = soft_limits(side)
        q_current = np.deg2rad(READY_Q_DEG[side])
        q_target = q_current.copy()
        q_target[3] = np.deg2rad(target_deg)
        q_command, limited = limit_joint_soft_position(
            q_current,
            q_target,
            lower,
            upper,
        )
        assert limited
        assert np.rad2deg(q_command[3]) == pytest.approx(expected_deg)


def test_soft_limiter_rejects_current_state_outside_interval():
    """Never snap a measured or previous command state onto a boundary."""
    lower, upper = soft_limits("left")
    q_current = np.deg2rad(READY_Q_DEG["left"])
    q_current[3] = np.deg2rad(-10.0)
    with pytest.raises(ValueError, match="q_current"):
        limit_joint_soft_position(
            q_current,
            q_current,
            lower,
            upper,
        )


@pytest.mark.parametrize(
    "joint_margin,elbow_branch,elbow_margin",
    [
        (0.0, -1, np.deg2rad(15.0)),
        (np.deg2rad(200.0), -1, np.deg2rad(15.0)),
        (np.deg2rad(5.0), 0, np.deg2rad(15.0)),
        (np.deg2rad(5.0), 1, 0.0),
    ],
)
def test_invalid_soft_limit_configuration_is_rejected(
    joint_margin,
    elbow_branch,
    elbow_margin,
):
    """Reject empty, hard-touching, or branchless soft intervals."""
    model = RM75Model("left")
    with pytest.raises(ValueError):
        make_teleop_soft_limits(
            model.q_min,
            model.q_max,
            joint_margin,
            elbow_index=3,
            elbow_branch=elbow_branch,
            elbow_margin=elbow_margin,
        )


def test_limits_each_joint_independently_at_requested_velocity():
    """Clip positive and negative deltas using qd_limit times dt."""
    q_current = np.zeros(4)
    q_target = np.array([1.0, -1.0, 0.05, -0.02])
    qd_limit = np.array([2.0, 1.0, 10.0, 10.0])

    q_command, limited = limit_joint_velocity(
        q_current,
        q_target,
        qd_limit,
        0.1,
    )

    assert limited
    assert np.allclose(q_command, [0.2, -0.1, 0.05, -0.02])
    assert np.all(np.abs((q_command - q_current) / 0.1) <= qd_limit)


def test_target_inside_rate_envelope_is_unchanged():
    """Return the target exactly when every delta is already admissible."""
    q_current = np.array([0.2, -0.3, 0.4])
    q_target = np.array([0.21, -0.32, 0.4])

    q_command, limited = limit_joint_velocity(
        q_current,
        q_target,
        np.ones(3),
        0.1,
    )

    assert not limited
    assert np.array_equal(q_command, q_target)


def test_zero_joint_limit_holds_only_that_joint():
    """Allow an explicitly frozen joint without weakening other limits."""
    q_command, limited = limit_joint_velocity(
        np.zeros(2),
        np.ones(2),
        np.array([0.0, 1.0]),
        0.2,
    )

    assert limited
    assert np.allclose(q_command, [0.0, 0.2])


def test_inputs_are_not_modified():
    """Keep caller-owned state immutable."""
    q_current = np.array([0.0, 0.0])
    q_target = np.array([1.0, -1.0])
    qd_limit = np.array([0.5, 0.5])
    originals = tuple(item.copy() for item in (q_current, q_target, qd_limit))

    limit_joint_velocity(q_current, q_target, qd_limit, 0.1)

    for value, original in zip((q_current, q_target, qd_limit), originals):
        assert np.array_equal(value, original)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("q_current", [0.0, math.nan]),
        ("q_target", [0.0, math.inf]),
        ("qd_limit", [1.0, -0.1]),
        ("qd_limit", [1.0, math.inf]),
        ("dt", 0.0),
        ("dt", -0.1),
        ("dt", math.nan),
        ("dt", math.inf),
    ],
)
def test_invalid_numeric_inputs_are_rejected(field, bad_value):
    """Fail closed on malformed limits, state, targets, and periods."""
    values = {
        "q_current": [0.0, 0.0],
        "q_target": [0.1, 0.1],
        "qd_limit": [1.0, 1.0],
        "dt": 0.1,
    }
    values[field] = bad_value
    with pytest.raises(ValueError):
        limit_joint_velocity(**values)


@pytest.mark.parametrize(
    "q_current,q_target,qd_limit",
    [
        ([0.0], [0.0, 0.0], [1.0]),
        ([0.0, 0.0], [0.0, 0.0], [1.0]),
        ([[0.0, 0.0]], [[0.1, 0.1]], [[1.0, 1.0]]),
    ],
)
def test_incompatible_shapes_are_rejected(q_current, q_target, qd_limit):
    """Require equal one-dimensional vectors."""
    with pytest.raises(ValueError):
        limit_joint_velocity(q_current, q_target, qd_limit, 0.1)


def test_acceleration_limiter_bounds_velocity_change_per_cycle():
    """Ramp commanded qdot by no more than qdd_limit times dt."""
    q_command, qd_command, limited = limit_joint_acceleration(
        q_current=np.zeros(3),
        q_target=np.array([1.0, -1.0, 0.001]),
        qd_current=np.array([0.0, 0.0, 0.02]),
        qdd_limit=np.array([2.0, 1.0, 3.0]),
        dt=0.1,
    )

    assert limited
    assert np.allclose(qd_command[:2], [0.2, -0.1])
    assert 0.0 < qd_command[2] < 0.01
    assert np.allclose(q_command[:2], [0.02, -0.01])
    assert 0.0 < q_command[2] < 0.001
    qdd = (qd_command - np.array([0.0, 0.0, 0.02])) / 0.1
    assert np.all(np.abs(qdd) <= np.array([2.0, 1.0, 3.0]))


def test_acceleration_limiter_preserves_already_admissible_target():
    """Return a stationary target exactly when no shaping is needed."""
    q_current = np.array([0.2, -0.4])
    q_target = q_current.copy()
    qd_current = np.zeros(2)

    q_command, qd_command, limited = limit_joint_acceleration(
        q_current,
        q_target,
        qd_current,
        np.array([1.0, 2.0]),
        0.1,
    )

    assert not limited
    assert np.array_equal(qd_command, np.zeros(2))
    assert np.array_equal(q_command, q_target)


def test_zero_acceleration_limit_preserves_previous_velocity():
    """Allow an intentionally frozen commanded velocity state."""
    q_command, qd_command, limited = limit_joint_acceleration(
        np.zeros(2),
        np.ones(2),
        np.array([0.1, -0.2]),
        np.zeros(2),
        0.1,
    )

    assert limited
    assert np.allclose(qd_command, [0.1, -0.2])
    assert np.allclose(q_command, [0.01, -0.02])


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("q_current", [0.0, math.nan]),
        ("q_target", [0.0, math.inf]),
        ("qd_current", [0.0, math.nan]),
        ("qdd_limit", [1.0, -0.1]),
        ("qdd_limit", [1.0, math.inf]),
        ("dt", 0.0),
        ("dt", math.nan),
    ],
)
def test_acceleration_limiter_rejects_invalid_inputs(field, bad_value):
    """Fail closed on malformed qdd state, limits, and periods."""
    values = {
        "q_current": [0.0, 0.0],
        "q_target": [0.1, 0.1],
        "qd_current": [0.0, 0.0],
        "qdd_limit": [1.0, 1.0],
        "dt": 0.1,
    }
    values[field] = bad_value
    with pytest.raises(ValueError):
        limit_joint_acceleration(**values)


def test_acceleration_limiter_rejects_incompatible_shapes():
    """Require every qdd limiter input to share one vector shape."""
    with pytest.raises(ValueError):
        limit_joint_acceleration(
            np.zeros(2),
            np.zeros(2),
            np.zeros(1),
            np.ones(2),
            0.1,
        )
