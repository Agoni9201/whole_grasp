#!/usr/bin/env python3
"""Offline comparison of URDF finger projections with a saved self mask.

No device access. A mismatch detects inconsistency between CAD, hand-eye,
frame assumptions and the observed mask; it cannot identify which is wrong.
"""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from real_nero_grasp.arm_collision_geometry import read_stl_triangles


def joint_transform(xml, name, displacement=0.0):
    joint = next(e for e in xml.findall("joint") if e.get("name") == name)
    origin = joint.find("origin")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler(
        "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    ).as_matrix()
    result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    if displacement:
        if joint.get("type") != "prismatic":
            raise ValueError("Only prismatic finger displacements are supported")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        result[:3, 3] += result[:3, :3] @ axis * displacement
    return result


def project_triangles(triangles_camera, intrinsics, image_shape):
    triangles = np.asarray(triangles_camera, dtype=float)
    valid = np.all(triangles[:, :, 2] > .01, axis=1)
    triangles = triangles[valid]
    if not len(triangles):
        raise ValueError("No complete finger triangles in front of the camera")
    uv = triangles[:, :, :2] / triangles[:, :, 2:]
    uv[:, :, 0] = uv[:, :, 0] * intrinsics["fx"] + intrinsics["cx"]
    uv[:, :, 1] = uv[:, :, 1] * intrinsics["fy"] + intrinsics["cy"]
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e6:
        raise ValueError("Invalid or excessive projected finger coordinates")
    mask = np.zeros(image_shape, np.uint8)
    for triangle in uv:
        cv2.fillConvexPoly(mask, np.rint(triangle).astype(np.int32), 255)
    return mask, uv, int(np.count_nonzero(~valid))


def compare_table_planes(capture, T_flange_camera):
    """Compare the dominant nearby plane in archived, unregistered views."""
    import open3d as o3d
    folders = sorted((Path(capture) / "multiview_raw").glob("*/metadata.json"))
    initial = next((p.parent for p in folders
                    if json.loads(p.read_text())["view_label"] == "initial"), None)
    if initial is None:
        return {"available": False, "reason": "No archived initial view"}
    initial_meta = json.loads((initial / "metadata.json").read_text())
    initial_old_handeye = np.asarray(initial_meta["handeye"]["snapshot"]["matrix_4x4"])
    initial_transform = (np.asarray(initial_meta["T_base_camera"])
                         @ np.linalg.inv(initial_old_handeye) @ T_flange_camera)
    target = np.asarray(o3d.io.read_point_cloud(str(initial / "target_camera.ply")).points)
    target = target @ initial_transform[:3, :3].T + initial_transform[:3, 3]
    if not len(target):
        return {"available": False, "reason": "Empty initial target"}
    center_xy = np.median(target, axis=0)[:2]
    planes = []
    for metadata in folders:
        info = json.loads(metadata.read_text())
        old_handeye = np.asarray(info["handeye"]["snapshot"]["matrix_4x4"])
        transform = np.asarray(info["T_base_camera"]) @ np.linalg.inv(old_handeye) @ T_flange_camera
        points = np.asarray(o3d.io.read_point_cloud(str(metadata.parent / "scene_camera.ply")).points)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        keep = ((np.linalg.norm(points[:, :2] - center_xy, axis=1) < .25)
                & (points[:, 2] > -.08) & (points[:, 2] < .08))
        if np.count_nonzero(keep) < 100:
            continue
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[keep]))
        o3d.utility.random.seed(0)
        plane, inliers = cloud.segment_plane(distance_threshold=.0025, ransac_n=3, num_iterations=300)
        normal, d = np.array(plane[:3]), float(plane[3])
        if normal[2] < 0:
            normal, d = -normal, -d
        if normal[2] < .9:
            continue
        height = -(normal[:2] @ center_xy + d) / normal[2]
        planes.append(dict(view=info["view_label"], normal=normal.tolist(), d=d,
                           inliers=len(inliers), roi_points=int(np.count_nonzero(keep)),
                           height_at_target_mm=float(height * 1000)))
    heights = [p["height_at_target_mm"] for p in planes]
    return dict(available=len(planes) >= 2, evaluation_xy_m=center_xy.tolist(),
                roi_radius_m=.25, ransac_threshold_m=.0025, planes=planes,
                height_range_mm=max(heights) - min(heights) if heights else None,
                interpretation="Dominant plane consistency, assuming the same static tabletop. "
                               "Does not identify hand-eye versus depth or robot-model error.")


def audit(capture, calibration, opening_m, output):
    capture, calibration, output = map(Path, (capture, calibration, output))
    if not np.isfinite(opening_m) or not 0 < opening_m <= .100:
        raise ValueError("Expected an opening in (0, 100] mm")
    output.mkdir(parents=True, exist_ok=True)
    observed_image = cv2.imread(str(capture / "camera_self_mask.png"), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(capture / "color.png"))
    if observed_image is None or image is None or observed_image.shape != image.shape[:2]:
        raise ValueError("Saved RGB and camera_self_mask.png must exist and have matching sizes")
    observed = observed_image > 0
    intrinsics = json.loads((capture / "capture_meta.json").read_text())["intrinsics"]
    calib = json.loads(calibration.read_text())
    if calib.get("convention") != "flange_camera_xyzw":
        raise ValueError("This audit requires the declared flange_camera_xyzw convention")
    T_camera_flange = np.linalg.inv(np.asarray(calib["matrix_4x4"], float))
    source = ROOT / "models/nero_with_gripper.urdf"
    xml = ET.parse(source).getroot()
    predicted = np.zeros(observed.shape, np.uint8)
    report = dict(
        hardware_connected=False, capture=str(capture.resolve()),
        calibration=str(calibration.resolve()),
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        source_urdf=str(source), opening_m=opening_m, intrinsics=intrinsics,
        assumptions=["URDF link7 equals SDK feedback flange",
                     "URDF finger geometry and joint-to-opening map match the hardware",
                     "three-frame saved self mask corresponds to visible fingers"],
        interpretation="Projection consistency diagnostic only; no automatic TCP or hand-eye correction.",
        fingers=[],
    )
    for index, displacement in ((1, opening_m / 2), (2, -opening_m / 2)):
        transform = (T_camera_flange @ joint_transform(xml, "gripper_flange_joint")
                     @ joint_transform(xml, "gripper_base_joint")
                     @ joint_transform(xml, f"gripper_joint{index}", displacement))
        # This project keeps these source URDF meshes in models/meshes,
        # also used by build_nero_graspgenx_asset.py.
        mesh = ROOT / f"models/meshes/gripper_link{index}.stl"
        triangles = read_stl_triangles(mesh) @ transform[:3, :3].T + transform[:3, 3]
        part, uv, dropped = project_triangles(triangles, intrinsics, observed.shape)
        predicted |= part
        pixels = int(np.count_nonzero(part))
        report["fingers"].append(dict(
            finger=index, mesh=str(mesh),
            projected_bbox=np.round([uv.reshape(-1, 2).min(0), uv.reshape(-1, 2).max(0)], 1).tolist(),
            in_image_pixels=pixels, triangles_crossing_near_plane_dropped=dropped,
            observed_mask_coverage=(float(np.count_nonzero((part > 0) & observed) / pixels)
                                    if pixels else None),
        ))
    for mask, color in ((observed_image, (255, 0, 255)), (predicted, (255, 255, 0))):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 3)
    cv2.putText(image, "Cyan: CAD + hand-eye. Magenta: fixed mask from 3 RGB frames.",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
    cv2.imwrite(str(output / "cad_vs_observed_fingers.png"), cv2.resize(image, (1280, 720)))
    cv2.imwrite(str(output / "predicted_finger_mask.png"), predicted)
    report["table_plane_comparison"] = compare_table_planes(capture, np.asarray(calib["matrix_4x4"]))
    (output / "cad_projection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "config/2026-10-22_charuco_handeye.json")
    parser.add_argument("--opening-mm", type=float, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.capture_dir, args.calibration, args.opening_mm / 1000, args.output_dir)
    print(json.dumps(report["fingers"], indent=2))
    print(json.dumps(report["table_plane_comparison"], indent=2))
    print("Offline diagnostic only; no hardware connection or calibration change.")


if __name__ == "__main__":
    main()
