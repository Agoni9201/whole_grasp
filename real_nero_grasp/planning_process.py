"""Persistent clean-interpreter numerical worker. No arm/camera objects cross IPC.

The worker can solve kinematics and rank NBV snapshots, never execute motion.
One serialized request is outstanding; failure invalidates the entire worker.
"""

import atexit
import contextlib
import io
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace


class PlanningProcessError(RuntimeError):
    pass


def _plain(value):
    """Whitelist data, not arbitrary picklable driver/thread/model objects."""
    import numpy as np
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        return value.copy()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(item) for key, item in value.items()}
    raise TypeError(f"Not numerical planning data: {type(value).__name__}")


def config_snapshot(args):
    keep_private = {"_active_target_label", "_active_fusion_target_shape_profile"}
    return {
        key: _plain(value) for key, value in vars(args).items()
        if not key.startswith("_") or key in keep_private
    }


class _FrozenArm:
    """Read-only numerical adapter; deliberately has no driver or motion API."""
    def __init__(self, state):
        self._state = state

    def get_joint_angles(self):
        return self._state["joints"].copy()

    def get_flange_pose(self):
        return self._state["flange"].copy()

    def get_tcp_pose(self):
        return self._state["tcp"].copy()


def _numerical_nbv_hooks(handeye):
    from . import active_view as av, active_view_scoring as scoring
    from . import pointcloud_fusion as fusion, ik_solver as ik, nbv_geometry as geometry
    hooks = av.ActiveViewHooks(
        transform_cloud=fusion.transform_cloud,
        points_from_cloud=fusion._points_from_cloud,
        make_cloud_from_points_colors=fusion._make_cloud_from_points_colors,
        handeye_transform=lambda *_: handeye.copy(),
        estimate_normals_toward_camera=scoring._estimate_normals_toward_camera,
        prepare_active_obstacle_cloud=geometry._prepare_active_obstacle_cloud,
        active_motion_scene_points=geometry._active_motion_scene_points,
        build_active_information_proxy=scoring._build_active_information_proxy,
        line_segment_obstacle_metrics=geometry._line_segment_obstacle_metrics,
        direction_context_penalty=scoring._direction_context_penalty,
        nearest_obstacle_distance=geometry._nearest_obstacle_distance,
        target_visibility_score=scoring._target_visibility_score,
        solve_ik_for_flange_pose=ik._solve_ik_for_flange_pose,
        score_active_candidate_information=scoring._score_active_candidate_information,
        build_scene_aware_nbv_records=geometry._build_scene_aware_nbv_records,
        active_spherical_camera_position=scoring._active_spherical_camera_position,
        parse_active_orbit_tilts=scoring._parse_active_orbit_tilts,
        normalize_dir_xy=scoring._normalize_dir_xy,
        get_ik_solver=ik._get_ik_solver,
        joint_limit_margin=ik._joint_limit_margin,
        look_at_camera_pose_flange_roll_candidates=av._look_at_camera_pose_flange_roll_candidates,
        spread_equivalent_roll_candidates=av._spread_equivalent_roll_candidates,
        rank_active_lookat_roll_solutions=lambda *a, **kw: av._rank_active_lookat_roll_solutions(
            *a, hooks=hooks, **kw),
    )
    return hooks


def _dispatch(operation, data, solvers):
    from kinematic_trac_ik import Kinematic
    import numpy as np
    if operation == "ping":
        import socket
        return {"pid": os.getpid(), "can_forbidden": socket.socket.__name__ == "NumericalSocket",
                "runtime_loaded": "real_nero_grasp.runtime" in sys.modules,
                "camera_loaded": "pyrealsense2" in sys.modules}
    if operation in ("ik_info", "ik", "fk"):
        key = tuple(data["solver"])
        if key not in solvers:
            path, base, tip, timeout, epsilon, kind = key
            solvers[key] = Kinematic(path, base, tip, timeout, epsilon, kind)
        solver = solvers[key]
        if operation == "ik_info":
            return {"num_joints": solver.num_joints, "limits": solver.get_joint_limits(),
                    "zero_pose": solver.get_zero_pose()}
        if operation == "fk":
            return solver.solve_fk(np.asarray(data["joints"], dtype=float))
        return solver.solve_ik(
            np.asarray(data["position"], dtype=float),
            np.asarray(data["wxyz"], dtype=float),
            None if data["seed"] is None else np.asarray(data["seed"], dtype=float),
            bool(data["previous"]),
        )
    if operation == "depth_median":
        from cameras.depth_fusion import median_valid_depth_frames
        return median_valid_depth_frames(
            data["frames"], min_valid_frames=data["min_valid_frames"],
            max_mad=data["max_mad"])
    if operation in {"supported_depth_median", "supported_depth_median_files"}:
        from cameras.depth_fusion import supported_depth_median
        file_mode = operation.endswith("_files")
        folder = Path(data["directory"]) if file_mode else None
        frames = np.load(folder / "processed.npy", mmap_mode="r") if file_mode else data["frames"]
        raw = np.load(folder / "raw.npy", mmap_mode="r") if file_mode else data["raw_frames"]
        result = supported_depth_median(
            frames, raw,
            min_valid_frames=data["min_valid_frames"], max_mad=data["max_mad"],
        )
        if file_mode:
            for key in ("depth_m", "sdk_depth_m", "raw_valid_count", "rejected_mask"):
                np.save(folder / (key + ".npy"), result[key])
            return {"report": result["report"]}
        return result
    if operation == "cloud_filter":
        from .pointcloud_fusion import filter_capture_cloud
        cloud = filter_capture_cloud(data["points"], data["colors"],
                                     data["voxel"], data["outlier"])
        return {"points": np.asarray(cloud.points).copy(),
                "colors": np.asarray(cloud.colors).copy()}
    if operation == "physical_table_reference":
        from .table_reference import build_table_reference
        try:
            report = build_table_reference(
                SimpleNamespace(**data["config"], _planning_worker=True),
                data["capture_dir"], data["T_base_camera"], data["output_dir"])
        except (ValueError, OSError, KeyError) as exc:
            # An unobserved/ambiguous table is a normal perception rejection,
            # not an IPC/control fault. Never continue with a guessed plane.
            return {"ok": False, "reason": f"physical table reference rejected: {exc}"}
        return {"ok": True, "report": report}
    if operation == "nero_grasp_geometry":
        from .nero_refinement import geometry_variants
        from .visualization import get_o3d
        args = SimpleNamespace(**data["config"], _planning_worker=True)
        o3d = get_o3d()
        target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["target_points"]))
        return [geometry_variants(row, target, np.asarray(data["T_base_camera"]), args)
                for row in data["rows"]]
    if operation == "post_grasp_joint_hold":
        from dataclasses import fields
        from .motion_planning import _motion_context
        from .post_grasp_rrt import plan_post_grasp_joint_hold
        from .types import MotionSafetyScene, PlanNotExecutable, UnsafeMotionTarget
        args = SimpleNamespace(**data["config"], _planning_worker=True)
        context = _motion_context(MotionSafetyScene(**data["scene"]), "carrying",
            attached_target_tcp=data["attached_target_tcp"],
            closed_gripper_width=data["closed_gripper_width"])
        try:
            stages = plan_post_grasp_joint_hold(data["q_start"], context, args, data["label"])
        except (PlanNotExecutable, UnsafeMotionTarget) as exc:
            return {"ok": False, "reason": str(exc)}
        return {"ok": True, "stages": [
            {field.name: getattr(stage, field.name) for field in fields(stage)} for stage in stages]}
    if operation == "environment_map":
        from dataclasses import fields
        from .pointcloud_fusion import fuse_dynamic_environment_clouds
        args = SimpleNamespace(**data["config"], _planning_worker=True)
        result = fuse_dynamic_environment_clouds(
            data["view_names"], data["clouds"], args,
            target_base=data["target"], map_id=data["map_id"],
            generation=data["generation"], target_center_base=data["target_center"],
        )
        # IPC carries only the canonical map's arrays/metadata, not an Open3D
        # object, SDK driver, callback, or control session.
        return _plain({field.name: getattr(result, field.name) for field in fields(result)})
    if operation == "nbv":
        import open3d as o3d
        from . import active_view
        args = SimpleNamespace(**data["config"], _planning_worker=True)
        # Same URDF/solver as the parent, but only a numerical local instance.
        def cloud(values):
            if values is None:
                return None
            item = o3d.geometry.PointCloud()
            item.points = o3d.utility.Vector3dVector(values["points"])
            if len(values["colors"]):
                item.colors = o3d.utility.Vector3dVector(values["colors"])
            return item
        hooks = _numerical_nbv_hooks(data["handeye"])
        result = active_view._build_active_view_candidates(
            _FrozenArm(data["state"]), args,
            cloud(data["target"]), cloud(data["scene"]), data["camera_transform"],
            obstacle_cloud=cloud(data["obstacles"]), hooks=hooks, **data["options"],
        )
        return result
    raise ValueError(f"Unknown numerical planning operation: {operation}")


def _worker_main(connection):
    # This is a fresh interpreter, not a forked Python runtime, with only plain
    # payloads. Forbid AF_CAN even if a future helper accidentally opens one.
    import socket
    original_socket = socket.socket
    class NumericalSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family == getattr(socket, "AF_CAN", None):
                raise RuntimeError("CAN connections are forbidden in the planning worker")
            super().__init__(family, *args, **kwargs)
    socket.socket = NumericalSocket
    # No CUDA model is constructed here. Bound native CPU pools as well as
    # avoiding the parent's GIL; this is not a hard real-time guarantee.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    # Use the project's geometry-only Open3D loader, not its optional ML stack.
    # Warm all NBV dependencies before the first reply (the initial IK request
    # is made by preflight, before the parent opens its CAN connection).
    from .visualization import get_o3d
    get_o3d()
    from . import active_view, nbv_geometry  # numerical modules only
    from threadpoolctl import threadpool_limits
    solvers = {}
    with threadpool_limits(limits=1):
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            if message is None:
                break
            request_id, operation, data = message
            started = time.monotonic()
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    value = _dispatch(operation, data, solvers)
                reply = {"id": request_id, "ok": True, "value": value}
            except Exception as exc:
                reply = {"id": request_id, "ok": False,
                         "error": f"{type(exc).__name__}: {exc}"}
            reply.update(pid=os.getpid(), elapsed_s=time.monotonic() - started,
                         stdout=output.getvalue())
            try:
                connection.send(reply)
            except (EOFError, BrokenPipeError):
                break
    connection.close()


class PlanningProcess:
    def __init__(self):
        self._connection, child = multiprocessing.Pipe()
        self._lock = threading.Lock()
        self._serial = 0
        self._closed = False
        self._io_thread = None
        environment = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[key] = "1"
        root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (root, environment.get("PYTHONPATH"))))
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "real_nero_grasp.planning_process",
                 "--worker-fd", str(child.fileno())],
                pass_fds=(child.fileno(),), close_fds=True, env=environment,
                stdin=subprocess.DEVNULL,
            )
        except BaseException:
            self._connection.close()
            raise
        finally:
            child.close()

    def request(self, operation, data, *, timeout_s=30.0, health_check=None, args=None):
        if not (0.0 < float(timeout_s) < float("inf")):
            raise ValueError("Planning request timeout must be finite and positive")
        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise PlanningProcessError("Numerical planning worker is unavailable")
            self._serial += 1
            request_id = self._serial
            started = time.monotonic()
            finished = threading.Event()
            exchange = {}
            def transfer():
                try:
                    self._connection.send((request_id, operation, _plain(data)))
                    exchange["reply"] = self._connection.recv()
                except Exception as exc:
                    exchange["error"] = exc
                finally:
                    finished.set()
            try:
                # Keep even a blocked pipe write bounded, not just the wait
                # for a reply. No IPC is performed in the MIT control thread.
                self._io_thread = threading.Thread(
                    target=transfer, name="numerical-planning-ipc", daemon=True)
                self._io_thread.start()
                while not finished.wait(0.02):
                    if health_check is not None:
                        health_check()
                    if self._process.poll() is not None:
                        raise PlanningProcessError("Numerical planning worker exited")
                    if time.monotonic() - started > timeout_s:
                        raise PlanningProcessError(f"Numerical planning {operation} timed out")
                if "error" in exchange:
                    raise PlanningProcessError(str(exchange["error"])) from exchange["error"]
                reply = exchange["reply"]
                if health_check is not None:
                    health_check()
                if reply["id"] != request_id or time.monotonic() - started > timeout_s:
                    raise PlanningProcessError("Late/mismatched numerical planning result")
                if not reply["ok"]:
                    raise PlanningProcessError(reply["error"])
                if reply["stdout"]:
                    print(reply["stdout"], end="")
                if args is not None and getattr(args, "save_dir", None):
                    record = {"timestamp": time.time(), "request_id": request_id,
                              "operation": operation, "pid": reply["pid"],
                              "worker_seconds": reply["elapsed_s"],
                              "round_trip_seconds": time.monotonic() - started}
                    path = Path(args.save_dir) / "planning_process.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                return reply["value"]
            except BaseException as exc:
                # Never consume a late reply as the answer to a later job.
                self.close()
                if isinstance(exc, (EOFError, OSError, KeyError, TypeError)):
                    raise PlanningProcessError(f"Numerical IPC failed: {exc}") from exc
                raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.close()
        finally:
            if self._process.poll() is None:
                self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
            if self._io_thread is not None:
                self._io_thread.join(timeout=1.0)


_CLIENT = None


def get_planning_process():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = PlanningProcess()
    return _CLIENT


def close_planning_process():
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.close()
    _CLIENT = None


atexit.register(close_planning_process)


def planning_health_check(args, label="numerical planning"):
    session = getattr(args, "_mit_session", None)
    if session is not None and session.owns_control:
        session._raise_if_terminal_fault(label)
        if getattr(session._arm, "_safety_stop_latched", False):
            from .types import ControlledMotionStopTriggered
            raise ControlledMotionStopTriggered("planning aborted after a latched MIT stop")


def checked_request(operation, data, args):
    try:
        return get_planning_process().request(
            operation, data, timeout_s=float(getattr(args, "planning_process_timeout_s", 60.0)),
            health_check=lambda: planning_health_check(args), args=args,
        )
    except PlanningProcessError as exc:
        session = getattr(args, "_mit_session", None)
        if session is not None and session.owns_control:
            from .types import ControlledMotionStopTriggered
            session._arm.terminate_motion()
            raise ControlledMotionStopTriggered(f"planning worker failed; stopped: {exc}") from exc
        raise


class IsolatedKinematic:
    def __init__(self, args):
        self._args = args
        self._key = (os.path.abspath(str(args.ik_urdf)), str(args.ik_base_link),
                     str(args.ik_tip_link), float(args.ik_timeout), float(args.ik_epsilon),
                     str(args.ik_solver_type))
        info = checked_request("ik_info", {"solver": self._key}, args)
        self.num_joints = info["num_joints"]
        self.lower_limits, self.upper_limits = info["limits"]
        self._zero_pose = info["zero_pose"]

    def get_joint_limits(self):
        return self.lower_limits.copy(), self.upper_limits.copy()

    def get_zero_pose(self):
        return self._zero_pose.copy()

    def solve_fk(self, joint_angles):
        return checked_request("fk", {"solver": self._key, "joints": joint_angles}, self._args)

    def solve_ik(self, target_position, target_wxyz, initial_guess=None, use_previous_solution=True):
        return checked_request("ik", {"solver": self._key, "position": target_position,
                                      "wxyz": target_wxyz, "seed": initial_guess,
                                      "previous": use_previous_solution}, self._args)


def isolated_nbv(arm, args, target, scene, camera_transform, *, obstacles=None, **options):
    import numpy as np
    from tools.real_nero_point_test import handeye_transform
    def cloud(item):
        return None if item is None else {"points": np.asarray(item.points).copy(),
                                          "colors": np.asarray(item.colors).copy()}
    planning_health_check(args)
    start = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7).copy()
    state = {"joints": start, "flange": np.asarray(arm.get_flange_pose()).copy(),
             "tcp": np.asarray(arm.get_tcp_pose()).copy()}
    calibration = handeye_transform(args.calib_json, args.handeye_convention)
    result = checked_request("nbv", {
        "config": config_snapshot(args), "state": state,
        "target": cloud(target), "scene": cloud(scene), "obstacles": cloud(obstacles),
        "camera_transform": camera_transform, "options": options,
        "handeye": calibration,
    }, args)
    planning_health_check(args)
    actual = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    tolerance = float(getattr(args, "active_center_joint_match_deg", 2.0))
    if not np.isfinite(actual).all() or np.rad2deg(np.max(np.abs(actual - start))) > tolerance:
        from .types import PlanNotExecutable
        raise PlanNotExecutable("NBV snapshot start changed during worker computation; discard result")
    if not np.array_equal(calibration, handeye_transform(args.calib_json, args.handeye_convention)):
        from .types import PlanNotExecutable
        raise PlanNotExecutable("Calibration changed while NBV was being computed; discard result")
    # The canonical live path/collision preflight remains mandatory downstream.
    return result


def isolated_depth_median(depth_frames, args):
    return checked_request("depth_median", {
        "frames": depth_frames,
        "min_valid_frames": getattr(args, "orbbec_depth_median_min_valid", 1),
        "max_mad": getattr(args, "orbbec_depth_median_max_mad", None),
    }, args)


def isolated_supported_depth_median(depth_frames, raw_frames, args):
    # Seven 1080p processed+raw frames exceed 100 MB. Avoid pickling that
    # payload under the parent's interpreter lock while CAN feedback is live.
    import numpy as np
    with tempfile.TemporaryDirectory(prefix="nero_raw_depth_") as directory:
        folder = Path(directory)
        for name, frames in (("processed", depth_frames), ("raw", raw_frames)):
            planning_health_check(args, "raw depth transfer")
            np.save(folder / (name + ".npy"), np.asarray(frames, dtype=np.float32))
        result = checked_request("supported_depth_median_files", {
            "directory": directory,
            "min_valid_frames": getattr(args, "orbbec_depth_median_min_valid", 1),
            "max_mad": getattr(args, "orbbec_depth_median_max_mad", None),
        }, args)
        for key in ("depth_m", "sdk_depth_m", "raw_valid_count", "rejected_mask"):
            result[key] = np.load(folder / (key + ".npy"), allow_pickle=False)
        planning_health_check(args, "raw depth result")
        return result


def isolated_environment_map(view_names, clouds, args, *, target_base=None,
                             map_id="dynamic_environment", generation=0,
                             target_center_base=None):
    from .pointcloud_fusion import _environment_points_base
    from .types import DynamicEnvironmentMap
    planning_health_check(args)
    result = checked_request("environment_map", {
        "view_names": tuple(view_names),
        "clouds": [_environment_points_base(cloud) for cloud in clouds],
        "config": config_snapshot(args),
        "target": None if target_base is None else _environment_points_base(target_base),
        "map_id": map_id, "generation": generation, "target_center": target_center_base,
    }, args)
    result = DynamicEnvironmentMap(**result)  # restore validated read-only arrays
    planning_health_check(args)
    return result


def isolated_capture_cloud(points, colors, voxel, outlier, args, *, get_o3d_hook=None):
    from .pointcloud_fusion import _make_cloud_from_points_colors
    result = checked_request("cloud_filter", {
        "points": points, "colors": colors, "voxel": float(voxel), "outlier": outlier,
    }, args)
    return _make_cloud_from_points_colors(
        result["points"], result["colors"], get_o3d_hook=get_o3d_hook)


if __name__ == "__main__":
    import argparse
    from multiprocessing.connection import Connection
    parser = argparse.ArgumentParser(description="Internal numerical-only planning worker")
    parser.add_argument("--worker-fd", type=int, required=True)
    _worker_main(Connection(parser.parse_args().worker_fd))
