"""Tests for URDF geometry readiness and coherent five-class distances."""

import numpy as np
import pytest

from vr_rm75_teleop.collision_backend import (
    CollisionBackendError,
    FiveClassCollisionBackend,
    PairDistance,
    UrdfCollisionModel,
    audit_urdf_collision,
    environment_geometry_from_config,
)
from vr_rm75_teleop.collision_safety import (
    CollisionRegion,
    CollisionSafetyMonitor,
    CollisionSource,
)


URDF = """<?xml version="1.0"?>
<robot name="test">
  <link name="body">
    <visual><geometry><box size="1 1 1"/></geometry></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision></link>
  <link name="l_rm75_base_link">
    <visual><geometry><sphere radius=".1"/></geometry></visual>
    <collision><geometry><sphere radius=".1"/></geometry></collision></link>
  <link name="l_rm75_link_1">
    <visual><geometry><sphere radius=".1"/></geometry></visual>
    <collision><geometry><sphere radius=".1"/></geometry></collision></link>
  <link name="l_rm75_link_2">
    <visual><geometry><sphere radius=".1"/></geometry></visual>
    <collision><geometry><sphere radius=".1"/></geometry></collision></link>
  <link name="r_rm75_base_link">
    <visual><geometry><sphere radius=".1"/></geometry></visual>
    <collision><geometry><sphere radius=".1"/></geometry></collision></link>
  <link name="r_rm75_link_1">
    <visual><geometry><sphere radius=".1"/></geometry></visual>
    <collision><geometry><sphere radius=".1"/></geometry></collision></link>
  <link name="r_rm75_link_2">
    <visual><geometry><sphere radius=".1"/></geometry></visual>
    <collision><geometry><sphere radius=".1"/></geometry></collision></link>
  <joint name="left_mount" type="fixed">
    <parent link="body"/><child link="l_rm75_base_link"/>
    <origin xyz="0 1 0" rpy="0 0 0"/></joint>
  <joint name="l_rm75_joint_1" type="revolute">
    <parent link="l_rm75_base_link"/><child link="l_rm75_link_1"/>
    <axis xyz="0 0 1"/></joint>
  <joint name="left_fixed" type="fixed">
    <parent link="l_rm75_link_1"/><child link="l_rm75_link_2"/>
    <origin xyz="1 0 0"/></joint>
  <joint name="right_mount" type="fixed">
    <parent link="body"/><child link="r_rm75_base_link"/>
    <origin xyz="0 -1 0" rpy="0 0 0"/></joint>
  <joint name="r_rm75_joint_1" type="revolute">
    <parent link="r_rm75_base_link"/><child link="r_rm75_link_1"/>
    <axis xyz="0 0 1"/></joint>
  <joint name="right_fixed" type="fixed">
    <parent link="r_rm75_link_1"/><child link="r_rm75_link_2"/>
    <origin xyz="1 0 0"/></joint>
</robot>
"""


class FakeDistanceEngine:
    """Return category-specific distances while recording evaluated pairs."""

    available = True
    unavailable_reason = None

    def __init__(self, override=None):
        """Configure an optional distance for one group pair."""
        self.override = override or {}
        self.calls = []

    def distance(self, first, first_transform, second, second_transform):
        """Record pair/poses and return a deterministic synthetic value."""
        groups = frozenset((first.group, second.group))
        self.calls.append(
            (first, first_transform.copy(), second, second_transform.copy())
        )
        if groups in self.override:
            value = self.override[groups]
        elif first.group == second.group == "left":
            value = 0.20
        elif first.group == second.group == "right":
            value = 0.30
        elif groups == frozenset(("left", "right")):
            value = 0.40
        elif "environment" in groups:
            value = 0.50
        else:
            value = 0.60
        return PairDistance(value, ((0.0, 0.0, 0.0), (value, 0.0, 0.0)))


def build_backend(tmp_path, engine=None, urdf=URDF):
    """Create one complete primitive-only model without external libraries."""
    path = tmp_path / "robot.urdf"
    path.write_text(urdf)
    model = UrdfCollisionModel(path)
    environment = environment_geometry_from_config(
        [
            {
                "name": "table",
                "type": "box",
                "dimensions": [2.0, 2.0, 0.1],
                "pose": {"xyz": [0.0, 0.0, -0.5]},
            }
        ]
    )
    return FiveClassCollisionBackend(
        model,
        engine or FakeDistanceEngine(),
        environment,
    )


def test_complete_snapshot_has_all_categories_and_mount_transforms(tmp_path):
    """Use the URDF tree transform and publish one coherent five-value set."""
    engine = FakeDistanceEngine()
    backend = build_backend(tmp_path, engine)
    snapshot = backend.evaluate(
        {"l_rm75_joint_1": 0.0, "r_rm75_joint_1": 0.0},
        measured_monotonic=7.0,
    )

    assert snapshot.ready and snapshot.valid
    assert snapshot.measured_monotonic == pytest.approx(7.0)
    assert snapshot.distances_m == pytest.approx(
        (0.2, 0.3, 0.4, 0.5, 0.6)
    )
    assert snapshot.closest_category == CollisionSource.LEFT_SELF
    inter_calls = [
        call for call in engine.calls
        if {call[0].group, call[2].group} == {"left", "right"}
    ]
    assert inter_calls
    assert any(call[1][1, 3] == pytest.approx(1.0) for call in inter_calls)
    assert any(call[3][1, 3] == pytest.approx(-1.0) for call in inter_calls)


def test_structurally_adjacent_links_are_excluded(tmp_path):
    """Never report normal parent-child contact as zero self distance."""
    engine = FakeDistanceEngine()
    backend = build_backend(tmp_path, engine)
    backend.evaluate({"l_rm75_joint_1": 0.0, "r_rm75_joint_1": 0.0})

    called_pairs = {
        frozenset((call[0].link, call[2].link)) for call in engine.calls
    }
    assert frozenset(("l_rm75_base_link", "l_rm75_link_1")) not in called_pairs
    assert frozenset(("l_rm75_link_1", "l_rm75_link_2")) not in called_pairs
    assert frozenset(("l_rm75_base_link", "l_rm75_link_2")) in called_pairs


@pytest.mark.parametrize(
    "distance, region",
    [
        (0.20, CollisionRegion.SAFE),
        (0.10, CollisionRegion.WARNING),
        (0.05, CollisionRegion.STOP),
        (-0.01, CollisionRegion.STOP),
    ],
)
def test_separation_warning_stop_and_overlap_feed_consumer(
    tmp_path, distance, region
):
    """Preserve signed FCL distance semantics through the existing monitor."""
    engine = FakeDistanceEngine({frozenset(("left", "right")): distance})
    snapshot = build_backend(tmp_path, engine).evaluate(
        {"l_rm75_joint_1": 0.0, "r_rm75_joint_1": 0.0},
        measured_monotonic=1.0,
    )
    monitor = CollisionSafetyMonitor(0.05, 0.15, 0.1)
    monitor.update_snapshot(
        dict(zip(tuple(CollisionSource), snapshot.distances_m)),
        received_monotonic=1.0,
    )
    assert monitor.evaluate(1.0).region == region


def test_missing_geometry_and_environment_are_not_ready(tmp_path):
    """Fail closed instead of inventing an envelope for a visible camera."""
    incomplete = URDF.replace(
        '<collision><geometry><box size="1 1 1"/></geometry></collision>',
        "",
        1,
    )
    path = tmp_path / "missing.urdf"
    path.write_text(incomplete)
    model = UrdfCollisionModel(path)
    assert audit_urdf_collision(path)[0].missing_collision
    backend = FiveClassCollisionBackend(model, FakeDistanceEngine())
    assert not backend.ready
    assert "no collision" in backend.readiness_reason
    assert "environment" in backend.readiness_reason
    with pytest.raises(CollisionBackendError):
        backend.evaluate({})


@pytest.mark.parametrize(
    "joints, expected",
    [
        ({"l_rm75_joint_1": 0.0}, "r_rm75_joint_1"),
        (
            {"l_rm75_joint_1": np.nan, "r_rm75_joint_1": 0.0},
            "finite",
        ),
    ],
)
def test_missing_transform_or_nan_invalidates_whole_snapshot(
    tmp_path, joints, expected
):
    """Never publish a partial set after joint/transform validation fails."""
    backend = build_backend(tmp_path)
    with pytest.raises(CollisionBackendError, match=expected):
        backend.evaluate(joints)
