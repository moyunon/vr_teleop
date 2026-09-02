"""Tests for the backend-independent collision safety boundary."""

import math

import pytest

from vr_rm75_teleop.collision_safety import (
    CollisionRegion,
    CollisionSafetyMonitor,
    CollisionSource,
    SOURCES,
    disabled_collision_decision,
)


def distances(default=0.30, **overrides):
    """Build one complete atomic distance snapshot."""
    values = {source: default for source in SOURCES}
    values.update(
        {CollisionSource(name): value for name, value in overrides.items()}
    )
    return values


def monitor():
    """Use round thresholds to make boundary tests explicit."""
    return CollisionSafetyMonitor(
        d_stop_m=0.05,
        d_warn_m=0.15,
        timeout_s=0.10,
    )


def rm75_only_monitor():
    """Enable only the two arm self checks and the inter-arm check."""
    return CollisionSafetyMonitor(
        d_stop_m=0.05,
        d_warn_m=0.15,
        timeout_s=0.10,
        enabled_sources=("left_self", "right_self", "inter_arm"),
    )


def test_missing_snapshot_fails_closed():
    decision = monitor().evaluate(1.0)

    assert decision.region == CollisionRegion.UNKNOWN
    assert not decision.ready
    assert decision.hold_required
    assert decision.speed_scale == 0.0
    assert "unavailable" in decision.reason


def test_distance_above_warning_is_clear():
    checker = monitor()
    checker.update_snapshot(distances(inter_arm=0.151), 1.0)

    decision = checker.evaluate(1.05)

    assert decision.region == CollisionRegion.SAFE
    assert decision.ready
    assert not decision.hold_required
    assert decision.speed_scale == 1.0
    assert decision.limiting_source == CollisionSource.INTER_ARM
    assert decision.min_distance_m == pytest.approx(0.151)


@pytest.mark.parametrize(
    "distance_m,expected_scale",
    [
        (0.15, 1.0),
        (0.125, 0.84375),
        (0.10, 0.5),
        (0.075, 0.15625),
        (math.nextafter(0.05, math.inf), 0.0),
    ],
)
def test_warning_region_continuously_scales_speed(distance_m, expected_scale):
    checker = monitor()
    checker.update_snapshot(distances(environment=distance_m), 2.0)

    decision = checker.evaluate(2.01)

    assert decision.region == CollisionRegion.WARNING
    assert decision.ready
    assert not decision.hold_required
    assert decision.speed_scale == pytest.approx(expected_scale, abs=1e-12)
    assert decision.limiting_source == CollisionSource.ENVIRONMENT


@pytest.mark.parametrize("distance_m", [0.05, 0.0, -0.01])
def test_stop_boundary_and_penetration_require_hold(distance_m):
    checker = monitor()
    checker.update_snapshot(distances(robot_body=distance_m), 3.0)

    decision = checker.evaluate(3.01)

    assert decision.region == CollisionRegion.STOP
    assert decision.ready
    assert decision.hold_required
    assert decision.speed_scale == 0.0
    assert decision.limiting_source == CollisionSource.ROBOT_BODY


def test_each_required_collision_class_can_be_the_limiting_source():
    checker = monitor()

    for index, source in enumerate(SOURCES):
        checker.update_snapshot(distances(**{source.value: 0.12}), 4.0 + index)
        decision = checker.evaluate(4.0 + index)
        assert decision.limiting_source == source
        assert decision.region == CollisionRegion.WARNING


def test_stale_snapshot_fails_closed_at_strictly_greater_than_timeout():
    checker = monitor()
    checker.update_snapshot(distances(), 5.0)

    assert checker.evaluate(5.10).region == CollisionRegion.SAFE
    decision = checker.evaluate(5.100001)

    assert decision.region == CollisionRegion.UNKNOWN
    assert decision.hold_required
    assert decision.speed_scale == 0.0
    assert decision.age_s == pytest.approx(0.100001)
    assert "stale" in decision.reason


@pytest.mark.parametrize(
    "bad_snapshot",
    [
        {CollisionSource.INTER_ARM: 0.2},
        distances(left_self=float("nan")),
        distances(right_self=float("inf")),
        {**distances(), "unexpected": 0.2},
    ],
)
def test_malformed_snapshot_invalidates_previous_safe_data(bad_snapshot):
    checker = monitor()
    checker.update_snapshot(distances(), 6.0)
    assert checker.evaluate(6.01).region == CollisionRegion.SAFE

    with pytest.raises((TypeError, ValueError)):
        checker.update_snapshot(bad_snapshot, 6.02)

    decision = checker.evaluate(6.03)
    assert decision.region == CollisionRegion.UNKNOWN
    assert decision.hold_required


@pytest.mark.parametrize(
    "d_stop,d_warn,timeout",
    [
        (0.0, 0.15, 0.1),
        (0.05, 0.05, 0.1),
        (0.16, 0.15, 0.1),
        (0.05, float("nan"), 0.1),
        (0.05, 0.15, -0.1),
    ],
)
def test_invalid_threshold_configuration_is_rejected(d_stop, d_warn, timeout):
    with pytest.raises(ValueError):
        CollisionSafetyMonitor(d_stop, d_warn, timeout)


def test_disabled_decision_is_explicit_and_does_not_hold():
    decision = disabled_collision_decision()

    assert decision.region == CollisionRegion.DISABLED
    assert decision.ready
    assert not decision.hold_required
    assert decision.speed_scale == 1.0
    assert "disabled" in decision.reason


def test_rm75_only_consumer_requires_only_enabled_categories():
    """Absent disabled categories do not participate in validity or limiting."""
    checker = rm75_only_monitor()
    checker.update_snapshot(
        {
            "left_self": 0.30,
            "right_self": 0.20,
            "inter_arm": 0.10,
        },
        8.0,
    )

    decision = checker.evaluate(8.01)
    assert decision.region == CollisionRegion.WARNING
    assert decision.limiting_source == CollisionSource.INTER_ARM


def test_rm75_only_consumer_ignores_disabled_category_values():
    """A disabled category is neither required nor validity-checked."""
    checker = rm75_only_monitor()
    checker.update_snapshot(
        {
            "left_self": 0.30,
            "right_self": 0.30,
            "inter_arm": 0.30,
            "environment": "not-consumed",
        },
        9.0,
    )
    assert checker.evaluate(9.01).region == CollisionRegion.SAFE


def test_rm75_only_consumer_fails_closed_for_missing_enabled_category():
    """The narrowed scope remains atomic and fail-closed for all three checks."""
    checker = rm75_only_monitor()
    with pytest.raises(ValueError, match="inter_arm"):
        checker.update_snapshot(
            {"left_self": 0.30, "right_self": 0.30},
            10.0,
        )
    assert checker.evaluate(10.01).region == CollisionRegion.UNKNOWN
