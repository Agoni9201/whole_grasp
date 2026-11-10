"""Conservative per-link capsules fitted to the configured URDF collision STL.

Each triangle is assigned to one capsule which contains all three vertices.
Since a capsule is convex, this covers the entire triangle, not just sampled
surface points. No mesh simplification or extra runtime dependency is needed.
"""

import hashlib
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


_CACHE = {}
MESH_PADDING_M = 0.003


def read_stl_triangles(path):
    data = Path(path).read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    if count >= 0 and len(data) == 84 + 50 * count:
        dtype = np.dtype([
            ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ])
        triangles = np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"]
    else:
        vertices = re.findall(rb"\bvertex\s+([^\r\n]+)", data)
        triangles = np.asarray([
            [float(value) for value in vertex.split()] for vertex in vertices
        ], dtype=float).reshape(-1, 3, 3)
    triangles = np.asarray(triangles, dtype=float).copy()
    if not triangles.size or not np.isfinite(triangles).all():
        raise ValueError(f"Invalid collision STL: {path}")
    return triangles


def resolve_mesh(urdf_path, filename):
    path = Path(urdf_path).resolve()
    if filename.startswith("package://"):
        # SolidWorks' package name differs from this checkout's directory name.
        # Resolve the mesh subtree beside the URDF, rather than guessing by basename.
        relative = filename.split("/meshes/", 1)
        if len(relative) != 2:
            raise ValueError(f"Unsupported collision mesh URI: {filename}")
        result = path.parent.parent / "meshes" / relative[1]
    else:
        result = Path(filename.removeprefix("file://"))
        if not result.is_absolute():
            result = path.parent / result
    result = result.resolve()
    if not result.is_file() or result.suffix.lower() != ".stl":
        raise ValueError(f"Missing or unsupported collision mesh: {result}")
    return result


def _distances(points, a, b):
    axis = b - a
    length2 = float(axis @ axis)
    if length2 < 1e-18:
        return np.linalg.norm(points - a, axis=1)
    alpha = np.clip(((points - a) @ axis) / length2, 0.0, 1.0)
    return np.linalg.norm(points - a - alpha[:, None] * axis, axis=1)


def _fit_capsule(vertices):
    vertices = np.unique(np.asarray(vertices).reshape(-1, 3), axis=0)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    _, axes = np.linalg.eigh(np.cov(vertices, rowvar=False))
    best = None
    for axis in np.vstack([np.eye(3), axes.T]):
        offset = vertices - center
        projection = offset @ axis
        perpendicular2 = np.maximum(np.sum(offset * offset, axis=1) - projection**2, 0.0)
        radius = float(np.sqrt(perpendicular2.max())) + MESH_PADDING_M
        # Move the centerline endpoints inward only as far as ALL vertices allow.
        # Fit with padding already removed, leaving >=3 mm at every vertex.
        radial_allowance = np.sqrt(np.maximum(
            (radius - MESH_PADDING_M)**2 - perpendicular2, 0.0
        ))
        low = float(np.min(projection + radial_allowance))
        high = float(np.max(projection - radial_allowance))
        if low > high:
            low = high = 0.5 * (low + high)
        a, b = center + low * axis, center + high * axis
        volume = np.pi * radius**2 * max(high - low, 0.0) + 4 * np.pi * radius**3 / 3
        if best is None or volume < best[0]:
            best = (volume, a, b, radius, axis)
    return best


def fit_triangle_capsules(triangles):
    whole = _fit_capsule(triangles)
    groups = [np.asarray(triangles)]
    # Two pieces follow the taper of long links while keeping runtime cost bounded.
    centers = np.mean(triangles, axis=1)
    projection = centers @ whole[4]
    if np.ptp(projection) > 0.12:
        split = projection <= 0.5 * (projection.min() + projection.max())
        if np.any(split) and np.any(~split):
            proposed = [triangles[split], triangles[~split]]
            fitted = [_fit_capsule(group) for group in proposed]
            if sum(item[0] for item in fitted) < 0.90 * whole[0]:
                groups = proposed
    result = []
    for group in groups:
        _, a, b, radius, _ = _fit_capsule(group)
        clearance = radius - _distances(group.reshape(-1, 3), a, b)
        if float(clearance.min()) < MESH_PADDING_M - 1e-8:
            raise ValueError("URDF capsule does not fully cover its assigned triangles")
        result.append((a, b, radius))
    return result


def load_arm_geometry(args, chain):
    path = Path(args.ik_urdf).resolve()
    child_links = tuple(joint["child"] for joint in chain)
    key = (str(path), path.stat().st_mtime_ns, child_links)
    if key in _CACHE:
        return _CACHE[key]
    root = ET.parse(path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    models, report = {}, []
    for name in child_links:
        link = links[name]
        collisions = link.findall("collision")
        if not collisions:
            raise ValueError(f"URDF arm link {name} has no collision geometry")
        capsules, sources = [], []
        for collision in collisions:
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Expected a collision mesh for arm link {name}")
            mesh_path = resolve_mesh(path, mesh.get("filename", ""))
            triangles = read_stl_triangles(mesh_path)
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(f"Invalid collision mesh scale for {name}")
            origin = collision.find("origin")
            xyz = np.fromstring("0 0 0" if origin is None else origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring("0 0 0" if origin is None else origin.get("rpy", "0 0 0"), sep=" ")
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
                raise ValueError(f"Invalid collision origin for {name}")
            triangles = (triangles * scale) @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
            capsules.extend(fit_triangle_capsules(triangles))
            sources.append({"path": str(mesh_path), "sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(), "triangles": len(triangles)})
        models[name] = capsules
        report.append({"link": name, "radii_mm": [round(item[2] * 1000, 3) for item in capsules], "sources": sources})
    result = {"models": models, "report": report, "padding_m": MESH_PADDING_M, "urdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    _CACHE[key] = result
    return result


def arm_capsules(args, chain, link_frames):
    geometry = load_arm_geometry(args, chain)
    capsules = []
    for name, local_capsules in geometry["models"].items():
        transform = link_frames[name]
        for index, (a, b, radius) in enumerate(local_capsules):
            capsules.append((
                transform[:3, :3] @ a + transform[:3, 3],
                transform[:3, :3] @ b + transform[:3, 3],
                radius, f"urdf_{name}_part{index + 1}",
            ))
    return capsules
