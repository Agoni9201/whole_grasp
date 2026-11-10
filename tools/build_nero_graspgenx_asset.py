#!/usr/bin/env python3
"""Build a GraspGenX Nero gripper asset from the project's measured URDF meshes.

The released GraspGenX checkpoint uses sweep_volume_v2 conditioning. This
asset provides its open/half-open sweep boxes plus real open/closed geometry
for offline visualization and table-plane checks. The joint-to-width mapping
is derived from the URDF; validate it against measured jaw width before use
on hardware.
"""

from __future__ import annotations

import copy
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

ROOT = Path(__file__).resolve().parents[1]
GENX = ROOT / "third_party" / "GraspGenX"
LINKS = ("gripper_flange", "gripper_base", "gripper_link1", "gripper_link2")
JOINTS = ("gripper_base_joint", "gripper_joint1", "gripper_joint2")


def _scene_meshes(robot: yourdfpy.URDF, joints: dict[str, float]) -> dict[str, trimesh.Trimesh]:
    robot.update_cfg(joints)
    result = {}
    for name, mesh in robot.scene.geometry.items():
        transformed = mesh.copy()
        transformed.apply_transform(robot.scene.graph.get(name)[0])
        result[name] = transformed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-mm", type=int, choices=(85, 100), default=85)
    args = parser.parse_args()
    asset_name = f"nero_2f_{args.opening_mm}"
    opening = args.opening_mm / 1000.0
    OUTPUT = GENX / "assets" / "x_grippers" / asset_name
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Asset already exists: {OUTPUT}; inspect it before rebuilding")
    source = ET.parse(ROOT / "models" / "nero_with_gripper.urdf").getroot()
    robot_xml = ET.Element("robot", name=asset_name)
    ET.SubElement(robot_xml, "link", name="world")
    canonical_joint = ET.SubElement(robot_xml, "joint", name="world_to_gripper_flange", type="fixed")
    ET.SubElement(canonical_joint, "parent", link="world")
    ET.SubElement(canonical_joint, "child", link="gripper_flange")
    # Nero's jaw axis is +Y in the gripper base. GraspGenX expects +X.
    ET.SubElement(canonical_joint, "origin", xyz="0 0 0", rpy="0 0 -1.5707963267948966")

    OUTPUT.mkdir(parents=True)
    mesh_dir = OUTPUT / "meshes"
    mesh_dir.mkdir()
    for name in LINKS:
        element = copy.deepcopy(next(e for e in source.findall("link") if e.get("name") == name))
        for mesh in element.findall(".//mesh"):
            stem = Path(mesh.attrib["filename"]).stem
            filename = f"{stem}.stl"
            mesh.attrib["filename"] = f"meshes/{filename}"
            destination = mesh_dir / filename
            if not destination.exists():
                shutil.copy2(ROOT / "models" / "meshes" / filename, destination)
        robot_xml.append(element)
    for name in JOINTS:
        robot_xml.append(copy.deepcopy(next(e for e in source.findall("joint") if e.get("name") == name)))
    ET.indent(robot_xml, space="  ")
    ET.ElementTree(robot_xml).write(OUTPUT / "gripper.urdf", encoding="utf-8", xml_declaration=True)

    robot = yourdfpy.URDF.load(
        str(OUTPUT / "gripper.urdf"), build_scene_graph=True, load_meshes=True,
        build_collision_scene_graph=False, load_collision_meshes=False,
    )
    open_cfg = {"gripper_joint1": opening / 2, "gripper_joint2": -opening / 2}
    close_cfg = {"gripper_joint1": 0.0, "gripper_joint2": 0.0}
    open_parts = _scene_meshes(robot, open_cfg)
    close_parts = _scene_meshes(robot, close_cfg)
    left_name = "gripper_link1.stl"
    right_name = "gripper_link2.stl"
    open_gap = float(open_parts[left_name].bounds[0, 0] - open_parts[right_name].bounds[1, 0])
    closed_gap = float(close_parts[left_name].bounds[0, 0] - close_parts[right_name].bounds[1, 0])
    if abs(open_gap - opening) > 0.001 or abs(closed_gap) > 0.001:
        raise ValueError(f"URDF jaw gap unexpected: open={open_gap:.4f}, closed={closed_gap:.4f}")

    open_mesh = trimesh.util.concatenate(list(open_parts.values()))
    close_mesh = trimesh.util.concatenate(list(close_parts.values()))
    open_mesh.export(OUTPUT / "coll_mesh.obj")
    open_mesh.export(OUTPUT / "vis_mesh.obj")
    np.random.seed(0)
    open_points = trimesh.sample.sample_surface(open_mesh, 10500)[0]
    close_points = trimesh.sample.sample_surface(close_mesh, 10500)[0]
    (OUTPUT / "points.json").write_text(
        json.dumps({"open": open_points.tolist(), "close": close_points.tolist()}, separators=(",", ":")),
        encoding="utf-8",
    )

    # The contact sweep stays within the configured jaw gap and the distal
    # half of the 64.8–141.3 mm finger span in canonical +Z.
    fingertip_z = float(max(open_parts[left_name].bounds[1, 2], open_parts[right_name].bounds[1, 2]))
    sweep_z = 0.045
    sweep_center_z = fingertip_z - sweep_z / 2.0
    config = {
        "open": open_cfg,
        "close": close_cfg,
        "fingertip": [0.0, 0.0, fingertip_z],
        "sweep_volume": {
            "extents": [open_gap, 0.050, sweep_z],
            "offset": [0.0, 0.0, sweep_center_z],
            "extents2": [open_gap / 2.0, 0.050, sweep_z],
            "offset2": [0.0, 0.0, sweep_center_z],
        },
        "links": ["world", *LINKS],
        "standoff": [0.0, 0.020],
        "bbox": open_mesh.bounds.tolist(),
        "symmetric": True,
        "type": "parallel_2f",
    }
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "source_urdf": "models/nero_with_gripper.urdf",
        "source_meshes": [f"models/meshes/{name}.stl" for name in LINKS],
        "canonical_frame": "+Z approach, +X jaw opening; source flange rotated -90 degrees around Z",
        "jaw_gap_source": f"URDF geometry at joint1=+{opening/2:.4f} m, joint2=-{opening/2:.4f} m",
        "open_gap_m": open_gap,
        "closed_gap_m": closed_gap,
        "finger_tip_z_m": fingertip_z,
        "conditioning": "sweep_volume_v2; TSDF and pointnet-VAE assets not generated",
        "hardware_validation": "Offline geometry only; verify joint-width map and TCP against the real Nero gripper",
    }
    (OUTPUT / "nero_asset_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Built {OUTPUT}: open gap={open_gap*1000:.1f} mm, fingertip z={fingertip_z*1000:.1f} mm")


if __name__ == "__main__":
    main()
