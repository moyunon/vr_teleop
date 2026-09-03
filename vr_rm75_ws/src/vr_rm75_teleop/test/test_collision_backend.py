"""Tests for URDF geometry readiness and collision category distances."""

from pathlib import Path

import numpy as np
import pytest

from vr_rm75_teleop.collision_backend import (
    CollisionBackendError,
    FclDistanceEngine,
    FiveClassCollisionBackend,
    PairDistance,
    UrdfCollisionModel,
    audit_urdf_collision,
    collision_category_diagnostics,
    enabled_sources_from_config,
    environment_geometry_from_config,
)
from vr_rm75_teleop.collision_safety import (
    CollisionRegion,
    CollisionSafetyMonitor,
    CollisionSource,
)


RM75_ONLY_SOURCES = (
    CollisionSource.LEFT_SELF,
    CollisionSource.RIGHT_SELF,
    CollisionSource.INTER_ARM,
)
RM75_MONITORED_LINKS = {
    "left": ["l_rm75_base_link"]
    + [f"l_rm75_link_{index}" for index in range(1, 8)],
    "right": ["r_rm75_base_link"]
    + [f"r_rm75_link_{index}" for index in range(1, 8)],
}
RM75_CATEGORY_CONFIG = {
    "left_self": True,
    "right_self": True,
    "inter_arm": True,
    "environment": False,
    "robot_body": False,
}


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


def build_backend(tmp_path, engine=None, urdf=URDF, ignored_pairs=()):
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
        ignored_pairs=ignored_pairs,
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
    category_results = {
        result.source: result for result in snapshot.category_results
    }
    for source, distance in zip(tuple(CollisionSource), snapshot.distances_m):
        result = category_results[source]
        assert result.distance_m == pytest.approx(distance)
        assert len(result.closest_pair) == 2
        assert result.closest_points is not None
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


def test_configured_base_pair_ignore_excludes_only_base_base(tmp_path):
    """Keep both cross-arm base-to-moving-link directions monitored."""
    engine = FakeDistanceEngine()
    backend = build_backend(
        tmp_path,
        engine,
        ignored_pairs=(("l_rm75_base_link", "r_rm75_base_link"),),
    )
    backend.evaluate({"l_rm75_joint_1": 0.0, "r_rm75_joint_1": 0.0})

    called_pairs = {
        frozenset((call[0].link, call[2].link)) for call in engine.calls
    }
    assert frozenset(
        ("l_rm75_base_link", "r_rm75_base_link")
    ) not in called_pairs
    assert frozenset(
        ("l_rm75_base_link", "r_rm75_link_1")
    ) in called_pairs
    assert frozenset(
        ("r_rm75_base_link", "l_rm75_link_1")
    ) in called_pairs


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


def test_rm75_only_real_urdf_ignores_out_of_scope_camera_geometry():
    """Missing camera collisions and empty environment do not block arms."""
    source_root = Path(__file__).resolve().parents[2]
    description = source_root / "lsrx_rm75_dual_description"
    model = UrdfCollisionModel(
        description / "urdf" / "LSRX_RM75_DUAL.urdf",
        package_roots={"lsrx_rm75_dual_description": description},
    )
    assert model.missing_collision_links == (
        "tb_camera_link",
        "xb_camera_link_1",
        "xb_camera_link_2",
        "xb_camera_link_3",
        "base_camera_link_1",
        "base_camera_link_2",
    )

    backend = FiveClassCollisionBackend(
        model,
        FakeDistanceEngine(),
        environment=(),
        enabled_sources=enabled_sources_from_config(RM75_CATEGORY_CONFIG),
        monitored_links=RM75_MONITORED_LINKS,
    )
    assert backend.ready, backend.readiness_reason

    joints = {
        f"{side}_rm75_joint_{index}": 0.0
        for side in ("l", "r")
        for index in range(1, 8)
    }
    # None of the other movable URDF joints, including the right camera and
    # modeled tools, is supplied. Only the fourteen RM75 states are required.
    snapshot = backend.evaluate(joints)
    assert snapshot.sources == RM75_ONLY_SOURCES
    assert snapshot.distances_m == pytest.approx((0.2, 0.3, 0.4))


def test_disabled_categories_have_no_placeholder_distance():
    """Diagnostics state configuration explicitly, without synthetic inf."""
    enabled = enabled_sources_from_config(RM75_CATEGORY_CONFIG)
    diagnostics = collision_category_diagnostics(enabled)

    assert diagnostics["left_self"]["status"] == "ENABLED"
    for name in ("environment", "robot_body"):
        assert diagnostics[name]["status"] == "DISABLED_BY_CONFIGURATION"
        assert diagnostics[name]["distance_m"] is None
        assert diagnostics[name]["closest_pair"] is None
        assert diagnostics[name]["closest_points"] is None


def test_category_diagnostics_report_each_enabled_witness(tmp_path):
    """Expose per-category minima without replacing the global witness."""
    snapshot = build_backend(tmp_path).evaluate(
        {"l_rm75_joint_1": 0.0, "r_rm75_joint_1": 0.0}
    )
    diagnostics = collision_category_diagnostics(snapshot.sources, snapshot)

    for source, expected_distance in zip(
        tuple(CollisionSource), (0.2, 0.3, 0.4, 0.5, 0.6)
    ):
        category = diagnostics[source.value]
        assert category["status"] == "ENABLED"
        assert category["distance_m"] == pytest.approx(expected_distance)
        assert len(category["closest_pair"]) == 2
        assert category["closest_points"] is not None
    assert snapshot.closest_pair == diagnostics["left_self"]["closest_pair"]
    assert snapshot.closest_points == (
        diagnostics["left_self"]["closest_points"]
    )


def test_fcl_reuses_one_collision_object_per_geometry(tmp_path):
    """Preload once and update persistent CollisionObjects in steady state."""
    engine = FclDistanceEngine()
    if not engine.available:
        pytest.skip(engine.unavailable_reason)
    backend = build_backend(tmp_path, engine)
    assert backend.ready, backend.readiness_reason
    object_ids = {name: id(value) for name, value in engine._objects.items()}

    joints = {"l_rm75_joint_1": 0.0, "r_rm75_joint_1": 0.0}
    backend.evaluate(joints)
    backend.evaluate(joints)

    assert len(engine._objects) == len(backend._query_geometries)
    assert {name: id(value) for name, value in engine._objects.items()} == (
        object_ids
    )


def test_aabb_pruning_preserves_signed_minimum(monkeypatch):
    """Evaluate every overlapping AABB when searching penetration minima."""
    engine = FclDistanceEngine()
    specs = [SimpleSpec(name) for name in "abcdef"]
    for spec in specs[:4]:
        engine._world_bounds[spec.name] = (np.zeros(3), np.ones(3))
    engine._world_bounds["e"] = (np.full(3, 10.0), np.ones(3))
    engine._world_bounds["f"] = (np.full(3, 12.5), np.ones(3))
    values = {
        frozenset(("a", "b")): -0.1,
        frozenset(("c", "d")): -0.3,
        frozenset(("e", "f")): 0.5,
    }
    calls = []

    def prepared_distance(first, second):
        calls.append((first.name, second.name))
        return PairDistance(values[frozenset((first.name, second.name))], None)

    monkeypatch.setattr(engine, "_prepared_distance", prepared_distance)
    first, second, result = engine.minimum_distance(
        ((specs[0], specs[1]), (specs[2], specs[3]), (specs[4], specs[5]))
    )

    assert (first.name, second.name) == ("c", "d")
    assert result.distance_m == pytest.approx(-0.3)
    assert calls == [("a", "b"), ("c", "d")]


class SimpleSpec:
    """Minimal geometry identity used by the AABB unit test."""

    def __init__(self, name):
        """Store the lookup key used by the distance engine."""
        self.name = name
