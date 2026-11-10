"""Capture evidence at the final open-gripper pose; never command motion."""

import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .planning_process import planning_health_check
from .transforms import _make_T_from_pose
from .types import PlanNotExecutable


def capture_grasp_contact_snapshot(arm, args, out_dir, planned_tcp, *, capture_rgbd_hook):
    if getattr(args, "rgb", None) is not None or getattr(args, "depth", None) is not None:
        raise ValueError("Contact check requires fresh live RGB-D, not offline image inputs")
    settings = copy.copy(args)
    capture_root = Path(out_dir) / "contact_check"
    # Numerical depth processing logs to save_dir before capture_rgbd creates
    # its timestamped image directory. Create the parent before any IPC call.
    capture_root.mkdir(parents=True, exist_ok=True)
    settings.save_dir = str(capture_root)
    planning_health_check(args, "contact check before capture")
    before = np.asarray(arm.get_flange_pose(), float).reshape(6)
    joints_before = np.asarray(arm.get_joint_angles(), float).reshape(7)
    started = time.monotonic()
    _, _, _, capture_dir = capture_rgbd_hook(settings)
    planning_health_check(args, "contact check after capture")
    after = np.asarray(arm.get_flange_pose(), float).reshape(6)
    joints_after = np.asarray(arm.get_joint_angles(), float).reshape(7)
    measured_tcp = np.asarray(arm.get_tcp_pose(), float).reshape(6)
    read_opening = getattr(arm, "print_gripper_status", None)
    gripper_status = read_opening() if callable(read_opening) else None
    measured_opening = getattr(gripper_status, "value", None)
    if measured_opening is not None:
        measured_opening = float(measured_opening)
        if not np.isfinite(measured_opening):
            measured_opening = None
    values = np.r_[before, after, joints_before, joints_after, measured_tcp]
    if not np.isfinite(values).all():
        raise PlanNotExecutable("Nonfinite robot feedback during contact check")
    delta = np.linalg.inv(_make_T_from_pose(before)) @ _make_T_from_pose(after)
    stationary = bool(
        np.linalg.norm(delta[:3, 3]) <= .001
        and Rotation.from_matrix(delta[:3, :3]).magnitude() <= np.deg2rad(.3)
        and np.max(np.abs(joints_after - joints_before)) <= np.deg2rad(.2)
    )
    if capture_dir is None:
        raise RuntimeError("Contact check capture did not provide an artifact directory")
    capture_dir = Path(capture_dir)
    for name in ("color.png", "depth_m.npy", "capture_meta.json"):
        if not (capture_dir / name).is_file():
            raise RuntimeError(f"Contact check artifact missing: {name}")
    report = dict(
        purpose="Final open-gripper contact-position diagnostic; no close or lift",
        camera_dir=str(capture_dir.resolve()), capture_started_monotonic=started,
        capture_finished_monotonic=time.monotonic(), stationary_during_capture=stationary,
        flange_pose_before=before.tolist(), flange_pose_after=after.tolist(),
        joints_before=joints_before.tolist(), joints_after=joints_after.tolist(),
        measured_tcp=measured_tcp.tolist(), planned_tcp=np.asarray(planned_tcp, float).tolist(),
        tcp_offset=np.asarray(args.tcp_offset, float).tolist(),
        commanded_opening_m=float(args.gripper_open_width),
        measured_opening_m=measured_opening,
        grasp_advance_m=float(getattr(args, "_selected_nero_execution_advance_m", args.grasp_advance)), calibration=str(args.calib_json),
        graspgenx_contact_advance_m=getattr(args, "_selected_genx_contact_advance_m", getattr(args, "graspgenx_contact_advance", 0.0)),
        contact_check_clearance_m=getattr(args, "grasp_contact_check_clearance", None),
        table_reference_report=getattr(args, "_grasp_table_reference_report", None),
        reference_selected_grasp=str(Path(out_dir).resolve() / "selected_grasp_pose.json"),
        reference_target_base=str(Path(out_dir).resolve() / "motion_safety_target_base.ply"),
        self_mask_applied=False, motion_commands_sent_by_snapshot=False,
    )
    report_path = capture_dir / "contact_pose.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    planning_health_check(args, "contact check after saving")
    print(f"[contact-check] RGB-D and final open-gripper pose saved: {report_path}")
    if not stationary:
        raise PlanNotExecutable("Robot moved during contact snapshot; image is diagnostic only")
    return str(report_path)
