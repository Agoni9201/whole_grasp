"""Fail-closed GraspGenX proposals mapped into the existing Nero workflow."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .grasp_geometry import _axis_vector, _rotation_from_axis_constraints
from .paths import PROJECT_ROOT


def gripper_asset(args=None):
    opening = min(float(getattr(args, "gripper_open_width", .100)),
                  float(getattr(args, "gripper_hardware_max_width", .100)))
    for mm in (85, 100):
        if abs(opening - mm / 1000) < .0005:
            return f"nero_2f_{mm}", mm / 1000
    raise ValueError(f"[graspgenx] no validated asset for {opening*1000:.1f} mm opening")


def open_gripper_mesh(args=None):
    from .visualization import get_o3d
    name, _ = gripper_asset(args)
    path = PROJECT_ROOT / "third_party/GraspGenX/assets/x_grippers" / name / "coll_mesh.obj"
    mesh = get_o3d().io.read_triangle_mesh(str(path))
    if len(mesh.vertices) < 100 or len(mesh.triangles) < 100:
        raise RuntimeError(f"[graspgenx] Nero collision mesh missing or invalid: {path}")
    return mesh


def _joint_transform(origin):
    result = np.eye(4)
    result[:3, :3] = R.from_euler("xyz", [float(x) for x in origin.attrib["rpy"].split()]).as_matrix()
    result[:3, 3] = [float(x) for x in origin.attrib["xyz"].split()]
    return result


def canonical_to_tcp(args):
    """GenX asset root -> Nero link7 -> configured hardware TCP."""
    source = ET.parse(PROJECT_ROOT / "models/nero_with_gripper.urdf").getroot()
    name, _ = gripper_asset(args)
    asset = ET.parse(PROJECT_ROOT / "third_party/GraspGenX/assets/x_grippers" / name / "gripper.urdf").getroot()
    def joint(root, name):
        element = next(node for node in root.findall("joint") if node.attrib["name"] == name)
        return _joint_transform(element.find("origin"))
    T_canonical_flange = joint(asset, "world_to_gripper_flange")
    T_link7_flange = joint(source, "gripper_flange_joint")
    offset = np.asarray(args.tcp_offset, dtype=float).reshape(6)
    T_link7_tcp = np.eye(4)
    T_link7_tcp[:3, :3] = R.from_euler("xyz", offset[3:]).as_matrix()
    T_link7_tcp[:3, 3] = offset[:3]
    T_canonical_tcp = T_canonical_flange @ np.linalg.inv(T_link7_flange) @ T_link7_tcp
    if not np.isfinite(T_canonical_tcp).all():
        raise ValueError("nonfinite GenX to TCP transform")
    approach = T_canonical_tcp[:3, :3] @ _axis_vector(args.tool_approach_axis)
    width = T_canonical_tcp[:3, :3] @ _axis_vector(args.tcp_gripper_width_axis)
    # The custom GraspGenX asset assumes +Z approaches the object and +X spans its jaws.
    if np.dot(approach, [0, 0, 1]) < .995 or abs(np.dot(width, [1, 0, 0])) < .995:
        raise ValueError("configured Nero TCP axes do not match GraspGenX asset")
    return T_canonical_tcp


def _validate_pose(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("invalid GraspGenX pose")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("invalid GraspGenX homogeneous row")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=.01) or np.linalg.det(pose[:3, :3]) < .99:
        raise ValueError("invalid GraspGenX rotation")
    return pose


def grasp_mesh_table_margin(args):
    trial = getattr(args, "grasp_contact_check_clearance", None)
    return float(trial) if trial is not None else max(
        float(getattr(args, "graspgenx_table_margin", .005)),
        float(getattr(args, "motion_path_min_clearance", .005)),
    )


def open_gripper_vertices_tcp(args):
    """Shared CAD geometry for endpoint and sampled configuration checks."""
    vertices = getattr(args, "_open_gripper_vertices_tcp", None)
    if vertices is None:
        inverse = np.linalg.inv(canonical_to_tcp(args))
        vertices = np.asarray(open_gripper_mesh(args).vertices, dtype=float)
        vertices = vertices @ inverse[:3, :3].T + inverse[:3, 3]
        args._open_gripper_vertices_tcp = vertices
    return vertices


def selected_mesh_table_clearance(pre_tcp, grasp_tcp, args):
    """Recheck the actual selected and recentered open mesh before execution."""
    from .transforms import _make_T_from_pose
    pre = _make_T_from_pose(pre_tcp)
    grasp = _make_T_from_pose(grasp_tcp)
    if not np.allclose(pre[:3, :3], grasp[:3, :3], atol=1e-4):
        raise RuntimeError("[grasp][mesh-table] final approach rotates gripper; endpoint table check insufficient")
    vertices = open_gripper_vertices_tcp(args)
    minima = [float(np.min(vertices @ pose[2, :3]) + pose[2, 3]) for pose in (pre, grasp)]
    clearance = min(minima) - float(args.table_z)
    if not np.isfinite(clearance) or clearance < float(args.graspgenx_table_margin):
        raise RuntimeError(
            f"[grasp][mesh-table] open gripper CAD mesh vs table plane z={float(args.table_z):.4f}m: "
            f"modeled table clearance {clearance * 1000:.1f}mm "
            f"< required {float(args.graspgenx_table_margin) * 1000:.1f}mm"
        )
    return clearance


def generate_graspgenx_candidates(proposal_cloud, T_base_camera, args, out_dir, *, measured_target=None, runner=None):
    """Keep native-safe poses first, plus table-rejected poses for refinement only."""
    if args.topdown_grasp or args.fixed_rpy is not None or args.use_current_rpy or args.use_graspnet_depth_advance:
        raise ValueError("GraspGenX requires native orientation, no fixed RPY/topdown/depth override")
    auto_advance = args.graspgenx_contact_advance == "auto"
    if auto_advance and measured_target is None:
        raise ValueError("GraspGenX auto contact advance requires the measured target cloud")
    if (int(args.graspgenx_num_grasps) <= 0 or int(args.graspgenx_topk_per_pass) <= 0
            or not np.isfinite(float(args.graspgenx_timeout)) or float(args.graspgenx_timeout) <= 0
            or float(args.graspgenx_table_margin) < 0 or float(args.pregrasp_offset) < 0):
        raise ValueError("invalid GraspGenX candidate budget, timeout, table margin or pregrasp offset")
    asset_name, opening = gripper_asset(args)
    T_canonical_tcp = canonical_to_tcp(args)
    T_base_camera = _validate_pose(T_base_camera)
    folder = Path(out_dir) / "graspgenx"
    folder.mkdir(parents=True, exist_ok=True)
    python = Path(args.graspgenx_python).expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(f"GraspGenX worker Python missing: {python}")
    points = np.asarray(proposal_cloud.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError("GraspGenX requires at least 100 finite target points")
    np.save(folder / "proposal_camera.npy", points.astype(np.float32))
    output = folder / "candidates.json"
    output.unlink(missing_ok=True)
    request = {
        "gripper_name": asset_name,
        "opening_m": opening,
        "input_npy": str((folder / "proposal_camera.npy").resolve()),
        "output_json": str(output.resolve()),
        "T_base_camera": T_base_camera.tolist(),
        "table_z": float(args.table_z),
        "table_margin": float(args.graspgenx_table_margin)
            if getattr(args, "grasp_contact_check_clearance", None) is None
            else grasp_mesh_table_margin(args),
        "pregrasp_offset": float(args.pregrasp_offset),
        "num_grasps": int(args.graspgenx_num_grasps),
        "topk_per_pass": int(args.graspgenx_topk_per_pass),
        "seed": int(args.graspgenx_seed),
    }
    if not np.isfinite([request["table_z"], request["table_margin"], request["pregrasp_offset"]]).all():
        raise ValueError("invalid table/pregrasp settings")
    (folder / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    command = [str(python), str(PROJECT_ROOT / "tools/run_graspgenx_worker.py"), "--request", str((folder / "request.json").resolve())]
    try:
        with (folder / "worker.log").open("w") as log:
            (subprocess.run if runner is None else runner)(command, check=True, timeout=float(args.graspgenx_timeout), stdout=log, stderr=subprocess.STDOUT, env=env)
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(f"[graspgenx] worker failed; see {folder / 'worker.log'}: {exc}") from exc
    if not output.is_file():
        raise RuntimeError("[graspgenx] worker produced no candidates.json")
    data = json.loads(output.read_text())
    if data.get("source") != f"GraspGenX diffusion, {asset_name}":
        raise RuntimeError("[graspgenx] unexpected worker output")
    grasp_api = str(PROJECT_ROOT / "graspnet-baseline/graspnetAPI")
    if grasp_api not in sys.path:
        sys.path.insert(0, grasp_api)
    from graspnetAPI import Grasp
    native = [candidate for candidate in data["candidates"] if candidate["table_safe"]][:int(args.manual_topk)]
    deferred = []
    if getattr(args, "nero_grasp_refine", False):
        # Keep every rejected pose as repair material. Search still uses the
        # existing parent/route/time budgets; proximity to the table constraint
        # determines repair priority, not the network confidence alone.
        deferred = sorted((candidate for candidate in data["candidates"] if not candidate["table_safe"]),
                          key=lambda item: (-item["minimum_table_clearance_m"], -item["score"]))
    rows = []
    for candidate in native + deferred:
        T_camera_canonical = _validate_pose(candidate["pose_camera"])
        T_camera_tcp = T_camera_canonical @ T_canonical_tcp
        # Keep the legacy grasp-frame accessor for logging/preview, but carry
        # the exact mapped TCP pose into grasp_to_tcp_poses below.
        approach = T_camera_tcp[:3, :3] @ _axis_vector(args.tool_approach_axis)
        width = T_camera_tcp[:3, :3] @ _axis_vector(args.tcp_gripper_width_axis)
        virtual_R = _rotation_from_axis_constraints(args.graspnet_approach_axis, approach, args.graspnet_width_axis, width)
        score = float(candidate["score"])
        grasp = Grasp(score, opening, 0.020, 0.0, virtual_R, T_camera_tcp[:3, 3], -1)
        grasp.nero_tcp_pose_camera = T_camera_tcp
        center = np.median(points, axis=0)
        center_dist = float(np.linalg.norm(T_camera_tcp[:3, 3] - center))
        rows.append({
            "idx": len(rows), "grasp": grasp,
            "translation": T_camera_tcp[:3, 3].copy(),
            "generator": "graspgenx", "generator_index": int(candidate["index"]),
            "generator_score": score, "raw_score": score,
            "width_source": f"fixed_nero_opening_{round(opening*1000)}mm_not_model_prediction",
            "center_dist": center_dist, "center_score": float(np.clip(1 - center_dist / .12, 0, 1)),
            "priority": score, "target_direction_applicable": False,
            "genx_table_clearance_m": float(candidate["minimum_table_clearance_m"]),
            "genx_native_table_safe": bool(candidate["table_safe"]),
            "genx_refinement_only": not bool(candidate["table_safe"]),
            "genx_pose_camera": T_camera_canonical.tolist(),
            "genx_tcp_pose_camera": T_camera_tcp.tolist(),
            "genx_contact_advance_m": 0.0 if auto_advance else float(args.graspgenx_contact_advance),
            "genx_contact_advance_mode": "auto" if auto_advance else "fixed",
        })
    if not rows:
        raise RuntimeError("[graspgenx] no candidate passed full open-mesh table clearance")
    if auto_advance:
        from .grasp_candidates import annotate_grasp_candidate_physical_aperture
        from .grasp_geometry import grasp_to_tcp_poses
        from .transforms import _make_T_from_pose
        vertices = open_gripper_vertices_tcp(args)
        margin = grasp_mesh_table_margin(args)
        pending = [row for row in rows if not row["genx_refinement_only"]]
        for row in rows:
            row["genx_contact_refine_ok"] = False
        # Native orientation and width stay fixed. Probe real target contact
        # plus actual recentered CAD clearance, then keep the first feasible
        # depth on a bounded 1 mm grid. No IK or exception-driven search here.
        for millimeters in range(16):
            if not pending:
                break
            advance = millimeters / 1000.0
            for row in pending:
                row["grasp"].nero_tcp_contact_advance = advance
            probes = annotate_grasp_candidate_physical_aperture(
                pending, measured_target, T_base_camera, None, args, verbose=False)
            unresolved = []
            for original, probe in zip(pending, probes):
                variant = next((item for item in probe["physical_aperture_variants"] if item["ok"]), None)
                if variant is None:
                    unresolved.append(original)
                    continue
                pre, contact, _ = grasp_to_tcp_poses(
                    original["grasp"], T_base_camera, args, None,
                    width_sign=variant["width_sign"], width_shift=variant["tcp_width_shift"])
                poses = [_make_T_from_pose(pose) for pose in (pre, contact)]
                clearance = min(float(np.min(vertices @ pose[2, :3]) + pose[2, 3])
                                for pose in poses) - float(args.table_z)
                if clearance < margin:
                    unresolved.append(original)
                    continue
                original.update(genx_contact_refine_ok=True, genx_contact_advance_m=advance,
                                genx_refined_table_clearance_m=clearance)
                print(f"[graspgenx][contact-refine] candidate={original['idx']} advance={millimeters}mm "
                      f"contact={variant['left_contact_inset']*1000:.1f}/{variant['right_contact_inset']*1000:.1f}mm "
                      f"CAD/table={clearance*1000:.1f}mm", flush=True)
            pending = unresolved
            if not pending:
                break
        for row in pending:
            row["grasp"].nero_tcp_contact_advance = 0.0
            print(f"[graspgenx][contact-refine] candidate={row['idx']} no contact+table solution "
                  "on 0..15mm grid; native pose retained as rejected diagnostic", flush=True)
    print(f"[graspgenx] generated={len(data['candidates'])} native_table_safe={len(native)} "
          f"refinement_only={len(deferred)} retained={len(rows)}; "
          "table-rejected poses cannot enter IK until repaired contact and CAD checks pass", flush=True)
    return [row["grasp"] for row in rows], rows
