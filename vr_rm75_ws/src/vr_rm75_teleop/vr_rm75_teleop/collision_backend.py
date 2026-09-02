"""URDF/FCL five-class collision-distance backend.

The module has no debug sphere fallback.  Missing required URDF geometry,
missing environment geometry, an unavailable transform, or an unavailable
FCL dependency keeps the backend not ready so the existing consumer fails
closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
from pathlib import Path
import struct
import time
from typing import Mapping, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from vr_rm75_teleop.collision_safety import CollisionSource, SOURCES


ARM_GROUPS = ("left", "right")


class CollisionBackendError(RuntimeError):
    """The backend cannot produce a complete trustworthy snapshot."""


@dataclass(frozen=True)
class GeometryAuditEntry:
    """Visual/collision presence for one URDF link."""

    link: str
    visual: str
    collision: str
    missing_collision: bool


@dataclass(frozen=True)
class GeometrySpec:
    """One collision geometry and its link-local or world transform."""

    name: str
    link: Optional[str]
    group: str
    kind: str
    dimensions: Tuple[float, ...]
    mesh_path: Optional[str]
    mesh_scale: Tuple[float, float, float]
    transform: np.ndarray


@dataclass(frozen=True)
class JointSpec:
    """One URDF tree edge."""

    name: str
    joint_type: str
    parent: str
    child: str
    axis: Tuple[float, float, float]
    origin: np.ndarray


@dataclass(frozen=True)
class PairDistance:
    """Distance result for one geometry pair."""

    distance_m: float
    closest_points: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]


@dataclass(frozen=True)
class CollisionBackendSnapshot:
    """One time-coherent five-class result with quality and timing."""

    measured_monotonic: float
    valid: bool
    ready: bool
    distances_m: Tuple[float, ...]
    closest_category: Optional[CollisionSource]
    closest_pair: Optional[Tuple[str, str]]
    closest_points: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]
    solve_ms: float
    mean_solve_ms: float
    max_solve_ms: float
    reason: str


def _numbers(text, count, default=None):
    if text is None:
        if default is None:
            raise CollisionBackendError("required numeric URDF field missing")
        values = default
    else:
        values = tuple(float(value) for value in text.split())
    if len(values) != count or not np.all(np.isfinite(values)):
        raise CollisionBackendError(f"expected {count} finite values")
    return tuple(float(value) for value in values)


def _rotation_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _transform(origin):
    result = np.eye(4)
    if origin is None:
        return result
    result[:3, :3] = _rotation_rpy(
        _numbers(origin.get("rpy"), 3, (0.0, 0.0, 0.0))
    )
    result[:3, 3] = _numbers(
        origin.get("xyz"), 3, (0.0, 0.0, 0.0)
    )
    return result


def _geometry_description(element):
    if element is None or len(element) != 1:
        return "missing"
    child = element[0]
    return f"{child.tag}:{dict(child.attrib)}"


def audit_urdf_collision(urdf_path):
    """Return a complete visual-versus-collision link audit."""
    root = ET.parse(str(urdf_path)).getroot()
    entries = []
    for link in root.findall("link"):
        visual = link.find("visual/geometry")
        collision = link.find("collision/geometry")
        entries.append(
            GeometryAuditEntry(
                link=link.get("name"),
                visual=_geometry_description(visual),
                collision=_geometry_description(collision),
                missing_collision=visual is not None and collision is None,
            )
        )
    return tuple(entries)


def _classify_link(link):
    if link.startswith(("l_rm75_", "ltool_")):
        return "left"
    if link.startswith(("r_rm75_", "rtool_")):
        return "right"
    return "body"


class UrdfCollisionModel:
    """Parse collision geometry and forward transforms from one URDF tree."""

    def __init__(self, urdf_path, package_roots=None):
        """Load the model without requiring ROS, trimesh, or FCL."""
        self.urdf_path = Path(urdf_path).resolve()
        self.package_roots = {
            str(key): Path(value).resolve()
            for key, value in (package_roots or {}).items()
        }
        root = ET.parse(str(self.urdf_path)).getroot()
        self.audit = audit_urdf_collision(self.urdf_path)
        self.missing_collision_links = tuple(
            entry.link for entry in self.audit if entry.missing_collision
        )
        self.geometries = self._parse_geometries(root)
        self.joints = self._parse_joints(root)
        children = {joint.child for joint in self.joints}
        links = {link.get("name") for link in root.findall("link")}
        roots = links - children
        if len(roots) != 1:
            raise CollisionBackendError(
                f"URDF must have exactly one root link, found {sorted(roots)}"
            )
        self.root_link = roots.pop()
        self._children = {}
        for joint in self.joints:
            self._children.setdefault(joint.parent, []).append(joint)
        self.structural_pairs = frozenset(
            frozenset((joint.parent, joint.child)) for joint in self.joints
        )

    def _resolve_mesh(self, filename):
        if filename.startswith("package://"):
            package, relative = filename[10:].split("/", 1)
            root = self.package_roots.get(package)
            if root is None:
                candidate = self.urdf_path.parents[1]
                if candidate.name == package:
                    root = candidate
            if root is None:
                raise CollisionBackendError(
                    f"package root unavailable for {filename}"
                )
            path = root / relative
        else:
            path = Path(filename)
            if not path.is_absolute():
                path = self.urdf_path.parent / path
        if not path.is_file():
            raise CollisionBackendError(f"collision mesh missing: {path}")
        return str(path.resolve())

    def _parse_geometries(self, root):
        geometries = []
        for link in root.findall("link"):
            link_name = link.get("name")
            for index, collision in enumerate(link.findall("collision")):
                geometry = collision.find("geometry")
                if geometry is None or len(geometry) != 1:
                    raise CollisionBackendError(
                        f"{link_name} collision geometry is malformed"
                    )
                shape = geometry[0]
                dimensions = ()
                mesh_path = None
                scale = (1.0, 1.0, 1.0)
                if shape.tag == "box":
                    dimensions = _numbers(shape.get("size"), 3)
                elif shape.tag == "sphere":
                    dimensions = (float(shape.get("radius")),)
                elif shape.tag == "cylinder":
                    dimensions = (
                        float(shape.get("radius")),
                        float(shape.get("length")),
                    )
                elif shape.tag == "mesh":
                    mesh_path = self._resolve_mesh(shape.get("filename"))
                    scale = _numbers(
                        shape.get("scale"), 3, (1.0, 1.0, 1.0)
                    )
                else:
                    raise CollisionBackendError(
                        f"unsupported URDF collision shape {shape.tag}"
                    )
                geometries.append(
                    GeometrySpec(
                        name=f"{link_name}:{index}",
                        link=link_name,
                        group=_classify_link(link_name),
                        kind=shape.tag,
                        dimensions=dimensions,
                        mesh_path=mesh_path,
                        mesh_scale=scale,
                        transform=_transform(collision.find("origin")),
                    )
                )
        return tuple(geometries)

    @staticmethod
    def _parse_joints(root):
        joints = []
        for element in root.findall("joint"):
            axis = element.find("axis")
            joints.append(
                JointSpec(
                    name=element.get("name"),
                    joint_type=element.get("type"),
                    parent=element.find("parent").get("link"),
                    child=element.find("child").get("link"),
                    axis=_numbers(
                        None if axis is None else axis.get("xyz"),
                        3,
                        (1.0, 0.0, 0.0),
                    ),
                    origin=_transform(element.find("origin")),
                )
            )
        return tuple(joints)

    def world_transforms(self, joint_positions):
        """Evaluate link poses using actual URDF fixed mount transforms."""
        values = {
            str(name): float(value) for name, value in joint_positions.items()
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise CollisionBackendError("joint positions must be finite")
        transforms = {self.root_link: np.eye(4)}
        queue = [self.root_link]
        while queue:
            parent = queue.pop(0)
            for joint in self._children.get(parent, ()):
                motion = np.eye(4)
                if joint.joint_type in ("revolute", "continuous"):
                    if joint.name not in values:
                        raise CollisionBackendError(
                            f"joint transform unavailable: {joint.name}"
                        )
                    axis = np.asarray(joint.axis, dtype=float)
                    norm = np.linalg.norm(axis)
                    if norm <= 0.0:
                        raise CollisionBackendError(
                            f"joint axis invalid: {joint.name}"
                        )
                    axis /= norm
                    angle = values[joint.name]
                    skew = np.asarray(
                        [
                            [0.0, -axis[2], axis[1]],
                            [axis[2], 0.0, -axis[0]],
                            [-axis[1], axis[0], 0.0],
                        ]
                    )
                    motion[:3, :3] = np.add(
                        np.add(np.eye(3), math.sin(angle) * skew),
                        (1.0 - math.cos(angle)) * (skew @ skew),
                    )
                elif joint.joint_type == "prismatic":
                    if joint.name not in values:
                        raise CollisionBackendError(
                            f"joint transform unavailable: {joint.name}"
                        )
                    motion[:3, 3] = (
                        np.asarray(joint.axis) * values[joint.name]
                    )
                elif joint.joint_type != "fixed":
                    raise CollisionBackendError(
                        f"unsupported joint type {joint.joint_type}"
                    )
                transforms[joint.child] = (
                    transforms[parent] @ joint.origin @ motion
                )
                queue.append(joint.child)
        return transforms


def environment_geometry_from_config(entries, base_directory=None):
    """Validate explicit world-frame box/sphere/cylinder/mesh configuration."""
    base = Path(base_directory or ".").resolve()
    result = []
    for index, entry in enumerate(entries or ()):
        if not isinstance(entry, Mapping):
            raise CollisionBackendError("environment entries must be mappings")
        kind = str(entry.get("type", ""))
        dimensions = tuple(
            float(value) for value in entry.get("dimensions", ())
        )
        expected = {"box": 3, "sphere": 1, "cylinder": 2, "mesh": 0}
        if kind not in expected or len(dimensions) != expected[kind]:
            raise CollisionBackendError(
                f"invalid dimensions for environment {kind!r}"
            )
        dimensions_invalid = any(
            not math.isfinite(value) or value <= 0.0
            for value in dimensions
        )
        if dimensions_invalid:
            raise CollisionBackendError(
                f"environment {kind!r} dimensions must be positive and finite"
            )
        mesh_path = entry.get("mesh")
        if kind == "mesh":
            if not mesh_path:
                raise CollisionBackendError("environment mesh path missing")
            path = Path(mesh_path)
            if not path.is_absolute():
                path = base / path
            if not path.is_file():
                raise CollisionBackendError(
                    f"environment mesh missing: {path}"
                )
            mesh_path = str(path.resolve())
        pose = entry.get("pose", {})
        origin = ET.Element(
            "origin",
            {
                "xyz": " ".join(str(v) for v in pose.get("xyz", (0, 0, 0))),
                "rpy": " ".join(str(v) for v in pose.get("rpy", (0, 0, 0))),
            },
        )
        scale = tuple(
            float(value) for value in entry.get("scale", (1, 1, 1))
        )
        if len(scale) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in scale
        ):
            raise CollisionBackendError(
                "environment mesh scale must contain 3 positive values"
            )
        result.append(
            GeometrySpec(
                name=str(entry.get("name", f"environment_{index}")),
                link=None,
                group="environment",
                kind=kind,
                dimensions=dimensions,
                mesh_path=mesh_path,
                mesh_scale=scale,
                transform=_transform(origin),
            )
        )
    return tuple(result)


def _load_stl(path, scale):
    """Read binary or ASCII STL into indexed triangle arrays."""
    data = Path(path).read_bytes()
    vertices = []
    triangles = []
    binary_count = (
        struct.unpack("<I", data[80:84])[0]
        if len(data) >= 84
        else 0
    )
    if len(data) >= 84 and 84 + 50 * binary_count == len(data):
        count = struct.unpack("<I", data[80:84])[0]
        for index in range(count):
            values = struct.unpack_from("<12fH", data, 84 + index * 50)
            triangle = []
            for offset in (3, 6, 9):
                triangle.append(len(vertices))
                vertices.append(values[offset:offset + 3])
            triangles.append(tuple(triangle))
    else:
        for line in data.decode("utf-8", errors="strict").splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append(tuple(float(value) for value in fields[1:]))
                if len(vertices) % 3 == 0:
                    triangles.append(
                        (
                            len(vertices) - 3,
                            len(vertices) - 2,
                            len(vertices) - 1,
                        )
                    )
    if not triangles:
        raise CollisionBackendError(f"STL has no triangles: {path}")
    return (
        np.asarray(vertices) * np.asarray(scale),
        np.asarray(triangles),
    )


class FclDistanceEngine:
    """Thin optional python-fcl adapter using full URDF collision geometry."""

    def __init__(self):
        """Record dependency availability without installing anything."""
        try:
            import fcl  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            self.fcl = None
            self.unavailable_reason = f"python-fcl unavailable: {exc}"
        else:
            self.fcl = fcl
            self.unavailable_reason = None
        self._models = {}

    @property
    def available(self):
        """Return whether python-fcl imported successfully."""
        return self.fcl is not None

    def _model(self, spec):
        if spec.name in self._models:
            return self._models[spec.name]
        fcl = self.fcl
        if spec.kind == "box":
            model = fcl.Box(*spec.dimensions)
        elif spec.kind == "sphere":
            model = fcl.Sphere(*spec.dimensions)
        elif spec.kind == "cylinder":
            model = fcl.Cylinder(*spec.dimensions)
        elif spec.kind == "mesh":
            vertices, triangles = _load_stl(
                spec.mesh_path, spec.mesh_scale
            )
            model = fcl.BVHModel()
            model.beginModel(len(vertices), len(triangles))
            model.addSubModel(vertices, triangles)
            model.endModel()
        else:
            raise CollisionBackendError(f"unsupported shape {spec.kind}")
        self._models[spec.name] = model
        return model

    def distance(self, first, first_transform, second, second_transform):
        """Return FCL minimum distance and optional nearest points."""
        if not self.available:
            raise CollisionBackendError(self.unavailable_reason)
        fcl = self.fcl
        first_object = fcl.CollisionObject(
            self._model(first),
            fcl.Transform(first_transform[:3, :3], first_transform[:3, 3]),
        )
        second_object = fcl.CollisionObject(
            self._model(second),
            fcl.Transform(second_transform[:3, :3], second_transform[:3, 3]),
        )
        request = fcl.DistanceRequest(enable_nearest_points=True)
        result = fcl.DistanceResult()
        distance = float(
            fcl.distance(first_object, second_object, request, result)
        )
        points = getattr(result, "nearest_points", None)
        closest = None
        if points is not None and len(points) == 2:
            closest = tuple(
                tuple(float(value) for value in point) for point in points
            )
        return PairDistance(distance, closest)


class FiveClassCollisionBackend:
    """Compute all five required minima from one joint-state snapshot."""

    def __init__(
        self,
        model,
        engine,
        environment=(),
        ignored_pairs=(),
        require_complete_geometry=True,
        require_environment=True,
    ):
        """Configure explicit readiness gates and allowed collision pairs."""
        self.model = model
        self.engine = engine
        self.environment = tuple(environment)
        self.ignored_pairs = set(model.structural_pairs)
        self.ignored_pairs.update(
            frozenset((str(first), str(second)))
            for first, second in ignored_pairs
        )
        reasons = []
        if require_complete_geometry and model.missing_collision_links:
            reasons.append(
                "links with visual but no collision: {}".format(
                    ", ".join(model.missing_collision_links)
                )
            )
        if require_environment and not self.environment:
            reasons.append("environment geometry is not configured")
        if not engine.available:
            reasons.append(engine.unavailable_reason)
        self.readiness_reason = "; ".join(reasons) or "collision backend ready"
        self.ready = not reasons
        self._solve_count = 0
        self._solve_total_ms = 0.0
        self._solve_max_ms = 0.0

    def _ignored(self, first, second):
        if first.link == second.link:
            return True
        return frozenset((first.link, second.link)) in self.ignored_pairs

    def evaluate(self, joint_positions, measured_monotonic=None):
        """Evaluate one joint sample or fail without a partial result."""
        if not self.ready:
            raise CollisionBackendError(self.readiness_reason)
        if measured_monotonic is None:
            measured_monotonic = time.monotonic()
        measured_monotonic = float(measured_monotonic)
        if not math.isfinite(measured_monotonic):
            raise CollisionBackendError("measurement timestamp is not finite")
        started = time.perf_counter()
        links = self.model.world_transforms(joint_positions)
        groups = {
            name: [g for g in self.model.geometries if g.group == name]
            for name in ("left", "right", "body")
        }

        def transform_for(geometry):
            if geometry.link is None:
                return geometry.transform
            if geometry.link not in links:
                raise CollisionBackendError(
                    f"link transform unavailable: {geometry.link}"
                )
            return links[geometry.link] @ geometry.transform

        pair_sets = {
            CollisionSource.LEFT_SELF: (
                pair for pair in combinations(groups["left"], 2)
                if not self._ignored(*pair)
            ),
            CollisionSource.RIGHT_SELF: (
                pair for pair in combinations(groups["right"], 2)
                if not self._ignored(*pair)
            ),
            CollisionSource.INTER_ARM: product(
                groups["left"], groups["right"]
            ),
            CollisionSource.ENVIRONMENT: product(
                groups["left"] + groups["right"] + groups["body"],
                self.environment,
            ),
            CollisionSource.ROBOT_BODY: (
                pair
                for pair in product(
                    groups["left"] + groups["right"], groups["body"]
                )
                if not self._ignored(*pair)
            ),
        }
        minima = {}
        detail = {}
        for source in SOURCES:
            found = False
            for first, second in pair_sets[source]:
                found = True
                pair_result = self.engine.distance(
                    first,
                    transform_for(first),
                    second,
                    transform_for(second),
                )
                distance = float(pair_result.distance_m)
                if not math.isfinite(distance):
                    raise CollisionBackendError(
                        f"non-finite {source.value} distance"
                    )
                if source not in minima or distance < minima[source]:
                    minima[source] = distance
                    detail[source] = (
                        (first.name, second.name),
                        pair_result.closest_points,
                    )
            if not found:
                raise CollisionBackendError(
                    f"no eligible pairs for {source.value}"
                )
        solve_ms = (time.perf_counter() - started) * 1000.0
        self._solve_count += 1
        self._solve_total_ms += solve_ms
        self._solve_max_ms = max(self._solve_max_ms, solve_ms)
        closest = min(SOURCES, key=minima.__getitem__)
        pair, points = detail[closest]
        return CollisionBackendSnapshot(
            measured_monotonic=measured_monotonic,
            valid=True,
            ready=True,
            distances_m=tuple(minima[source] for source in SOURCES),
            closest_category=closest,
            closest_pair=pair,
            closest_points=points,
            solve_ms=solve_ms,
            mean_solve_ms=self._solve_total_ms / self._solve_count,
            max_solve_ms=self._solve_max_ms,
            reason="complete five-class FCL snapshot",
        )
