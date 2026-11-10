"""Open3D window in a clean interpreter; no robot objects or CAN descriptors."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


def _pack_geometries(geometries):
    arrays, specifications = {}, []
    for index, geometry in enumerate(geometries):
        if hasattr(geometry, "triangles") and hasattr(geometry, "vertices"):
            kind, names = "mesh", ("vertices", "triangles", "vertex_colors", "vertex_normals")
        elif hasattr(geometry, "lines") and hasattr(geometry, "points"):
            kind, names = "lines", ("points", "lines", "colors")
        elif hasattr(geometry, "points"):
            kind, names = "cloud", ("points", "colors", "normals")
        else:
            raise TypeError(f"Unsupported preview geometry: {type(geometry).__name__}")
        fields = {}
        for name in names:
            values = np.asarray(getattr(geometry, name, []))
            if values.size:
                if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                    raise ValueError(f"Invalid preview geometry field: {name}")
                key = f"g{index}_{name}"
                arrays[key] = values
                fields[name] = key
        specifications.append({"kind": kind, "fields": fields})
    return arrays, specifications


def _load_geometries(specifications, array_file):
    from .visualization import get_o3d
    o3d = get_o3d()
    result = []
    with np.load(array_file, allow_pickle=False) as arrays:
        for item in specifications:
            constructors = {"mesh": o3d.geometry.TriangleMesh,
                            "lines": o3d.geometry.LineSet,
                            "cloud": o3d.geometry.PointCloud}
            geometry = constructors[item["kind"]]()
            for name, key in item["fields"].items():
                if name == "triangles":
                    vector = o3d.utility.Vector3iVector(arrays[key])
                elif name == "lines":
                    vector = o3d.utility.Vector2iVector(arrays[key])
                else:
                    vector = o3d.utility.Vector3dVector(arrays[key])
                setattr(geometry, name, vector)
            result.append(geometry)
    return result


def _draw(geometries, request):
    from .visualization import get_o3d
    viewer = get_o3d().visualization.Visualizer()
    created = False
    try:
        created = bool(viewer.create_window(window_name=request["window_name"],
                       width=request["width"], height=request["height"]))
        if not created:
            raise RuntimeError("Open3D could not create a visualization window")
        for geometry in geometries:
            viewer.add_geometry(geometry)
        options = viewer.get_render_option()
        if options is not None:
            options.point_size = request["point_size"]
        viewer.run()
    finally:
        if created:
            viewer.destroy_window()


def _stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=.5)


def show_preview(geometries, window_name, point_size, width, height, *,
                 health_check=None, popen_factory=None):
    """Wait with the parent GIL available and propagate any control fault."""
    if health_check is not None:
        health_check()
    arrays, specifications = _pack_geometries(geometries)
    with tempfile.TemporaryDirectory(prefix="grasp-preview-") as directory:
        directory = Path(directory)
        np.savez(directory / "geometry.npz", **arrays)
        request = dict(array_file=str(directory / "geometry.npz"),
                       geometries=specifications, window_name=str(window_name),
                       width=int(width), height=int(height),
                       point_size=float(np.clip(point_size, 1., 10.)))
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request))
        environment = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[key] = "1"
        root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (root, environment.get("PYTHONPATH"))))
        with (directory / "worker.log").open("w") as log:
            process = (popen_factory or subprocess.Popen)(
                [sys.executable, "-m", "real_nero_grasp.preview_process", "--request", str(request_path)],
                env=environment, close_fds=True, start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            print(f"[preview] isolated Open3D worker pid={process.pid}; no CAN ownership", flush=True)
            try:
                while process.poll() is None:
                    if health_check is not None:
                        health_check()
                    time.sleep(.02)
                if health_check is not None:
                    health_check()
                status_path = directory / "status.json"
                status = json.loads(status_path.read_text()) if status_path.is_file() else {}
                if process.returncode != 0 or not status.get("ok"):
                    detail = status.get("error") or (directory / "worker.log").read_text()[-1000:]
                    raise RuntimeError(f"Open3D preview worker failed: {detail}")
                return status
            finally:
                _stop_process(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    # This worker owns only numeric geometry and a window, never CAN/USB.
    import socket
    original_socket = socket.socket
    class PreviewSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *a, **kw):
            if family == getattr(socket, "AF_CAN", None):
                raise RuntimeError("CAN connections are forbidden in the preview worker")
            super().__init__(family, *a, **kw)
    socket.socket = PreviewSocket
    status = {"pid": os.getpid(), "can_forbidden": True}
    try:
        request = json.loads(args.request.read_text())
        geometries = _load_geometries(request["geometries"], request["array_file"])
        _draw(geometries, request)
        status["ok"] = True
        status["runtime_loaded"] = "real_nero_grasp.runtime" in sys.modules
        status["camera_loaded"] = "pyrealsense2" in sys.modules
    except Exception as exc:
        status.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    (args.request.parent / "status.json").write_text(json.dumps(status))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
