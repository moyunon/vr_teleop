"""
URDF/FCL five-class collision-distance backend.

The module has no debug sphere fallback. Missing required monitored geometry,
missing geometry for an enabled category, an unavailable required transform,
or an unavailable FCL dependency keeps the backend not ready so the existing
consumer fails closed.
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

from vr_rm75_teleop.collision_safety import (
    CollisionSource,
    SOURCES,
    normalize_collision_sources,
)


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
class CategoryDistance:
    """Minimum distance and witness data for one enabled category."""

    source: CollisionSource
    distance_m: float
    closest_pair: Tuple[str, str]
    closest_points: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]


@dataclass(frozen=True)
class CollisionBackendSnapshot:
    """One time-coherent enabled-category result with quality and timing."""

    measured_monotonic: float
    valid: bool
    ready: bool
    sources: Tuple[CollisionSource, ...]
    distances_m: Tuple[float, ...]
    category_results: Tuple[CategoryDistance, ...]
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
        self.links = frozenset(links)
        self._children = {}
        self._parent_joint = {}
        for joint in self.joints:
            self._children.setdefault(joint.parent, []).append(joint)
            if joint.child in self._parent_joint:
                raise CollisionBackendError(
                    f"URDF link has multiple parents: {joint.child}"
                )
            self._parent_joint[joint.child] = joint
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

    @staticmethod
    def _joint_motion(joint, values):
        """Return one URDF joint motion, validating only that used joint."""
        motion = np.eye(4)
        if joint.joint_type in ("revolute", "continuous"):
            if joint.name not in values:
                raise CollisionBackendError(
                    f"joint transform unavailable: {joint.name}"
                )
            try:
                angle = float(values[joint.name])
            except (TypeError, ValueError) as exc:
                raise CollisionBackendError(
                    f"joint position invalid: {joint.name}"
                ) from exc
            if not math.isfinite(angle):
                raise CollisionBackendError(
                    f"joint position must be finite: {joint.name}"
                )
            axis = np.asarray(joint.axis, dtype=float)
            norm = np.linalg.norm(axis)
            if norm <= 0.0:
                raise CollisionBackendError(
                    f"joint axis invalid: {joint.name}"
                )
            axis /= norm
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
            try:
                displacement = float(values[joint.name])
            except (TypeError, ValueError) as exc:
                raise CollisionBackendError(
                    f"joint position invalid: {joint.name}"
                ) from exc
            if not math.isfinite(displacement):
                raise CollisionBackendError(
                    f"joint position must be finite: {joint.name}"
                )
            motion[:3, 3] = np.asarray(joint.axis) * displacement
        elif joint.joint_type != "fixed":
            raise CollisionBackendError(
                f"unsupported joint type {joint.joint_type}"
            )
        return motion

    def _lowest_common_ancestor(self, links):
        """Find a local fixed frame shared by all requested links."""
        chains = []
        for link in links:
            chain = [link]
            while chain[-1] != self.root_link:
                joint = self._parent_joint.get(chain[-1])
                if joint is None:
                    raise CollisionBackendError(
                        f"link is disconnected from URDF root: {link}"
                    )
                chain.append(joint.parent)
            chains.append(chain)
        common = set(chains[0]).intersection(
            *(set(chain) for chain in chains[1:])
        )
        return next(link for link in chains[0] if link in common)

    def world_transforms(self, joint_positions, required_links=None):
        """
        Evaluate all poses, or only paths needed by monitored links.

        A requested subset is expressed in its lowest common ancestor frame.
        Pairwise robot distances are invariant to that common transform, so
        unrelated mobile-base, body, camera, and tool joints are not required.
        """
        values = {str(name): value for name, value in joint_positions.items()}
        if required_links is not None:
            requested = tuple(
                dict.fromkeys(str(link) for link in required_links)
            )
            if not requested:
                return {}
            unknown = sorted(set(requested) - self.links)
            if unknown:
                raise CollisionBackendError(
                    "unknown required links: " + ", ".join(unknown)
                )
            local_root = self._lowest_common_ancestor(requested)
            transforms = {local_root: np.eye(4)}

            def evaluate_link(link):
                if link in transforms:
                    return transforms[link]
                joint = self._parent_joint[link]
                parent_transform = evaluate_link(joint.parent)
                transforms[link] = (
                    parent_transform
                    @ joint.origin
                    @ self._joint_motion(joint, values)
                )
                return transforms[link]

            for link in requested:
                evaluate_link(link)
            return transforms

        transforms = {self.root_link: np.eye(4)}
        queue = [self.root_link]
        while queue:
            parent = queue.pop(0)
            for joint in self._children.get(parent, ()):
                transforms[joint.child] = (
                    transforms[parent]
                    @ joint.origin
                    @ self._joint_motion(joint, values)
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
        self._objects = {}
        self._local_bounds = {}
        self._world_bounds = {}
        self._request = (
            None
            if self.fcl is None
            else self.fcl.DistanceRequest(enable_nearest_points=True)
        )

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
            half_extent = np.asarray(spec.dimensions, dtype=float) / 2.0
            local_center = np.zeros(3)
        elif spec.kind == "sphere":
            model = fcl.Sphere(*spec.dimensions)
            half_extent = np.full(3, spec.dimensions[0], dtype=float)
            local_center = np.zeros(3)
        elif spec.kind == "cylinder":
            model = fcl.Cylinder(*spec.dimensions)
            radius, length = spec.dimensions
            half_extent = np.asarray(
                (radius, radius, length / 2.0), dtype=float
            )
            local_center = np.zeros(3)
        elif spec.kind == "mesh":
            vertices, triangles = _load_stl(
                spec.mesh_path, spec.mesh_scale
            )
            model = fcl.BVHModel()
            model.beginModel(len(vertices), len(triangles))
            model.addSubModel(vertices, triangles)
            model.endModel()
            local_min = np.min(vertices, axis=0)
            local_max = np.max(vertices, axis=0)
            local_center = (local_min + local_max) / 2.0
            half_extent = (local_max - local_min) / 2.0
        else:
            raise CollisionBackendError(f"unsupported shape {spec.kind}")
        self._models[spec.name] = model
        self._local_bounds[spec.name] = (local_center, half_extent)
        return model

    def prepare(self, geometries):
        """Preload BVHs and create exactly one object per geometry."""
        if not self.available:
            raise CollisionBackendError(self.unavailable_reason)
        for spec in geometries:
            if spec.name not in self._objects:
                self._objects[spec.name] = self.fcl.CollisionObject(
                    self._model(spec)
                )

    def update_transforms(self, geometry_transforms):
        """Update each prepared object once and cache conservative AABBs."""
        geometry_transforms = tuple(geometry_transforms)
        self.prepare(spec for spec, _ in geometry_transforms)
        for spec, transform in geometry_transforms:
            transform = np.asarray(transform, dtype=float)
            self._objects[spec.name].setTransform(
                self.fcl.Transform(
                    transform[:3, :3],
                    transform[:3, 3],
                )
            )
            local_center, local_half_extent = self._local_bounds[spec.name]
            world_center = (
                transform[:3, :3] @ local_center + transform[:3, 3]
            )
            world_half_extent = (
                np.abs(transform[:3, :3]) @ local_half_extent
            )
            self._world_bounds[spec.name] = (
                world_center,
                world_half_extent,
            )

    def _prepared_distance(self, first, second):
        result = self.fcl.DistanceResult()
        distance = float(
            self.fcl.distance(
                self._objects[first.name],
                self._objects[second.name],
                self._request,
                result,
            )
        )
        points = getattr(result, "nearest_points", None)
        closest = None
        if (
            points is not None
            and len(points) == 2
            and all(point is not None for point in points)
        ):
            closest = tuple(
                tuple(float(value) for value in point) for point in points
            )
        return PairDistance(distance, closest)

    def _aabb_lower_bound(self, first, second):
        first_center, first_half = self._world_bounds[first.name]
        second_center, second_half = self._world_bounds[second.name]
        separation = np.maximum(
            np.abs(first_center - second_center)
            - first_half
            - second_half,
            0.0,
        )
        return float(np.linalg.norm(separation))

    def minimum_distance(self, pairs):
        """Return the exact pair minimum after conservative AABB pruning."""
        ranked_pairs = sorted(
            (
                (self._aabb_lower_bound(first, second), first, second)
                for first, second in pairs
            ),
            key=lambda item: item[0],
        )
        best = None
        best_pair = None
        for lower_bound, first, second in ranked_pairs:
            # An AABB separation is a lower bound for separated geometry.
            # When FCL reports penetration, every remaining zero-bound pair
            # must still be evaluated to preserve the exact signed minimum.
            if (
                best is not None
                and lower_bound > max(best.distance_m, 0.0) + 1e-12
            ):
                break
            result = self._prepared_distance(first, second)
            if best is None or result.distance_m < best.distance_m:
                best = result
                best_pair = (first, second)
        if best is None:
            raise CollisionBackendError("FCL query has no eligible pairs")
        return best_pair[0], best_pair[1], best

    def distance(self, first, first_transform, second, second_transform):
        """Return FCL minimum distance and optional nearest points."""
        if not self.available:
            raise CollisionBackendError(self.unavailable_reason)
        self.update_transforms(
            ((first, first_transform), (second, second_transform))
        )
        return self._prepared_distance(first, second)


def enabled_sources_from_config(category_enabled):
    """Parse an explicit five-category boolean configuration mapping."""
    if not isinstance(category_enabled, Mapping):
        raise TypeError("category_enabled must be a mapping")
    known = {source.value for source in SOURCES}
    unknown = sorted(set(category_enabled) - known)
    missing = sorted(known - set(category_enabled))
    if unknown:
        raise ValueError(
            "unknown collision categories: " + ", ".join(unknown)
        )
    if missing:
        raise ValueError(
            "category_enabled missing categories: " + ", ".join(missing)
        )
    for name, enabled in category_enabled.items():
        if not isinstance(enabled, bool):
            raise TypeError(f"category_enabled.{name} must be boolean")
    return normalize_collision_sources(
        source for source in SOURCES if category_enabled[source.value]
    )


def collision_category_diagnostics(enabled_sources, snapshot=None):
    """Describe every category without inventing disabled distances."""
    enabled_sources = normalize_collision_sources(enabled_sources)
    enabled = set(enabled_sources)
    results = {}
    if snapshot is not None:
        results = {
            result.source: result for result in snapshot.category_results
        }
    return {
        source.value: {
            "status": (
                "ENABLED"
                if source in enabled
                else "DISABLED_BY_CONFIGURATION"
            ),
            "distance_m": (
                results[source].distance_m
                if source in results
                else None
            ),
            "closest_pair": (
                results[source].closest_pair
                if source in results
                else None
            ),
            "closest_points": (
                results[source].closest_points
                if source in results
                else None
            ),
        }
        for source in SOURCES
    }


class FiveClassCollisionBackend:
    """Compute the configured subset while retaining all five categories."""

    def __init__(
        self,
        model,
        engine,
        environment=(),
        ignored_pairs=(),
        require_complete_geometry=True,
        require_environment=True,
        enabled_sources=None,
        monitored_links=None,
    ):
        """Configure explicit readiness gates and allowed collision pairs."""
        self.model = model
        self.engine = engine
        self.environment = tuple(environment)
        self.enabled_sources = normalize_collision_sources(enabled_sources)
        self.ignored_pairs = set(model.structural_pairs)
        self.ignored_pairs.update(
            frozenset((str(first), str(second)))
            for first, second in ignored_pairs
        )
        self.monitored_links = self._normalize_monitored_links(monitored_links)
        if self.monitored_links is None:
            self._groups = {
                name: tuple(
                    geometry
                    for geometry in model.geometries
                    if geometry.group == name
                )
                for name in ("left", "right", "body")
            }
        else:
            self._groups = {
                name: tuple(
                    geometry
                    for geometry in model.geometries
                    if geometry.link in self.monitored_links[name]
                )
                for name in ("left", "right", "body")
            }

        reasons = []
        if require_complete_geometry:
            if self.monitored_links is None:
                missing_geometry = model.missing_collision_links
            else:
                geometry_links = {
                    geometry.link for geometry in model.geometries
                }
                missing_geometry = tuple(
                    link
                    for group in ("left", "right", "body")
                    for link in self.monitored_links[group]
                    if link not in geometry_links
                )
            if missing_geometry:
                reasons.append(
                    "monitored links with no collision: "
                    + ", ".join(missing_geometry)
                )
        if (
            CollisionSource.ENVIRONMENT in self.enabled_sources
            and require_environment
            and not self.environment
        ):
            reasons.append("environment geometry is not configured")
        if not engine.available:
            reasons.append(engine.unavailable_reason)

        self._pairs = {
            source: self._candidate_pairs(source)
            for source in self.enabled_sources
        }
        for source, pairs in self._pairs.items():
            if not pairs and not (
                source == CollisionSource.ENVIRONMENT
                and require_environment
                and not self.environment
            ):
                reasons.append(f"no eligible pairs for {source.value}")

        required_links = {
            geometry.link
            for pairs in self._pairs.values()
            for pair in pairs
            for geometry in pair
            if geometry.link is not None
        }
        if CollisionSource.ENVIRONMENT in self.enabled_sources:
            # World/environment distance needs the monitored robot geometry
            # expressed relative to the URDF root, but unrelated branches are
            # still excluded from traversal.
            required_links.add(model.root_link)
        self._required_links = tuple(sorted(required_links))
        query_geometries = {
            geometry.name: geometry
            for pairs in self._pairs.values()
            for pair in pairs
            for geometry in pair
        }
        self._query_geometries = tuple(query_geometries.values())
        self._use_prepared_query = all(
            callable(getattr(engine, method, None))
            for method in (
                "prepare",
                "update_transforms",
                "minimum_distance",
            )
        )
        self.preload_ms = 0.0
        if engine.available and self._use_prepared_query:
            preload_started = time.perf_counter()
            try:
                engine.prepare(self._query_geometries)
            except (
                CollisionBackendError,
                OSError,
                TypeError,
                ValueError,
            ) as exc:
                reasons.append(f"FCL geometry preload failed: {exc}")
            self.preload_ms = (
                time.perf_counter() - preload_started
            ) * 1000.0
        self.readiness_reason = "; ".join(reasons) or "collision backend ready"
        self.ready = not reasons
        self._solve_count = 0
        self._solve_total_ms = 0.0
        self._solve_max_ms = 0.0

    def _normalize_monitored_links(self, monitored_links):
        if monitored_links is None:
            return None
        if not isinstance(monitored_links, Mapping):
            raise TypeError("monitored_links must be a mapping")
        valid_groups = {"left", "right", "body"}
        unknown_groups = sorted(set(monitored_links) - valid_groups)
        if unknown_groups:
            raise ValueError(
                "unknown monitored link groups: "
                + ", ".join(unknown_groups)
            )
        result = {}
        assigned = {}
        for group in ("left", "right", "body"):
            links = monitored_links.get(group, ())
            if isinstance(links, (str, bytes)):
                raise TypeError(f"monitored_links.{group} must be a sequence")
            normalized = tuple(dict.fromkeys(str(link) for link in links))
            for link in normalized:
                if link not in self.model.links:
                    raise ValueError(f"unknown monitored link: {link}")
                if link in assigned:
                    raise ValueError(
                        f"monitored link {link} appears in both "
                        f"{assigned[link]} and {group}"
                    )
                actual_group = _classify_link(link)
                if actual_group != group:
                    raise ValueError(
                        f"monitored link {link} belongs to {actual_group}, "
                        f"not {group}"
                    )
                assigned[link] = group
            result[group] = normalized
        return result

    def _ignored(self, first, second):
        if first.link == second.link:
            return True
        return frozenset((first.link, second.link)) in self.ignored_pairs

    def _candidate_pairs(self, source):
        left = self._groups["left"]
        right = self._groups["right"]
        body = self._groups["body"]
        if source == CollisionSource.LEFT_SELF:
            pairs = combinations(left, 2)
        elif source == CollisionSource.RIGHT_SELF:
            pairs = combinations(right, 2)
        elif source == CollisionSource.INTER_ARM:
            pairs = product(left, right)
        elif source == CollisionSource.ENVIRONMENT:
            pairs = product(left + right + body, self.environment)
        elif source == CollisionSource.ROBOT_BODY:
            pairs = product(left + right, body)
        else:  # pragma: no cover - the enum is exhaustive
            raise CollisionBackendError(f"unsupported source {source}")
        return tuple(
            pair for pair in pairs if not self._ignored(*pair)
        )

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
        links = self.model.world_transforms(
            joint_positions,
            required_links=self._required_links,
        )

        def transform_for(geometry):
            if geometry.link is None:
                return geometry.transform
            if geometry.link not in links:
                raise CollisionBackendError(
                    f"link transform unavailable: {geometry.link}"
                )
            return links[geometry.link] @ geometry.transform

        geometry_transforms = {
            geometry.name: (geometry, transform_for(geometry))
            for geometry in self._query_geometries
        }
        if self._use_prepared_query:
            self.engine.update_transforms(geometry_transforms.values())

        minima = {}
        detail = {}
        for source in self.enabled_sources:
            if self._use_prepared_query:
                first, second, pair_result = self.engine.minimum_distance(
                    self._pairs[source]
                )
                distance = float(pair_result.distance_m)
                if not math.isfinite(distance):
                    raise CollisionBackendError(
                        f"non-finite {source.value} distance"
                    )
                minima[source] = distance
                detail[source] = (
                    (first.name, second.name),
                    pair_result.closest_points,
                )
            else:
                for first, second in self._pairs[source]:
                    pair_result = self.engine.distance(
                        first,
                        geometry_transforms[first.name][1],
                        second,
                        geometry_transforms[second.name][1],
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
        solve_ms = (time.perf_counter() - started) * 1000.0
        self._solve_count += 1
        self._solve_total_ms += solve_ms
        self._solve_max_ms = max(self._solve_max_ms, solve_ms)
        closest = min(self.enabled_sources, key=minima.__getitem__)
        pair, points = detail[closest]
        category_results = tuple(
            CategoryDistance(
                source=source,
                distance_m=minima[source],
                closest_pair=detail[source][0],
                closest_points=detail[source][1],
            )
            for source in self.enabled_sources
        )
        return CollisionBackendSnapshot(
            measured_monotonic=measured_monotonic,
            valid=True,
            ready=True,
            sources=self.enabled_sources,
            distances_m=tuple(
                minima[source] for source in self.enabled_sources
            ),
            category_results=category_results,
            closest_category=closest,
            closest_pair=pair,
            closest_points=points,
            solve_ms=solve_ms,
            mean_solve_ms=self._solve_total_ms / self._solve_count,
            max_solve_ms=self._solve_max_ms,
            reason="complete enabled-category FCL snapshot",
        )
