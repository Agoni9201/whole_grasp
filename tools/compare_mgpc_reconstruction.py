#!/usr/bin/env python3
"""Offline four-way target reconstruction viewer. No robot or GraspNet imports.

Stages: prepare saved captures; infer with official MGPC weights; render the
actual results. Prediction panels remain visibly unavailable until inference
has written their outputs. All clouds use metres in base_link.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_RUN = ROOT / "logs/2026-10-22_22-19-22_028479_p211282"
DEFAULT_OUTPUT = ROOT / "artifacts/mgpc_compare"


def transform(points, matrix):
    matrix = np.asarray(matrix)
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def fps(points, count=2048):
    """Deterministic farthest-point sampling; exact NumPy distance updates."""
    points = np.asarray(points, dtype=np.float64)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Point input must be nonempty and finite")
    indices = np.empty(min(count, len(points)), np.int64)
    distances = np.full(len(points), np.inf)
    chosen = 0
    for i in range(len(indices)):
        indices[i] = chosen
        distances = np.minimum(distances, np.sum((points-points[chosen])**2, axis=1))
        chosen = int(np.argmax(distances))
    if len(indices) < count:
        indices = np.resize(indices, count)
    return points[indices], indices


def normalize(points):
    """Match official in-the-wild samples: centroid zero, bbox diagonal one."""
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(diagonal) or diagonal < 1e-6:
        raise ValueError("Degenerate target extent")
    return ((points-centroid)/diagonal).astype(np.float32), dict(
        centroid_camera_m=centroid.tolist(), bbox_diagonal_m=diagonal,
        axes="native aligned RGB camera: +x right, +y down, +z forward",
        normalization="FPS 2048, subtract sample mean, divide by sample bbox diagonal")


def denormalize(points, normalization):
    return (np.asarray(points, np.float64)*normalization["bbox_diagonal_m"]
            + np.asarray(normalization["centroid_camera_m"]))


def save_cloud(path, points, colors=None):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, np.float64)))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(path)


def project_rgb(points_base, views, tolerance=.006):
    """Color registered geometry only where a masked depth observation supports it.

    Each view maps camera -> base -> registered target. Undo both transforms
    before projection. Unobserved surfaces stay grey, never inherit a nearby
    front-side label merely because it is the nearest measured point.
    """
    points_base = np.asarray(points_base)
    colors = np.full((len(points_base), 3), .55)
    source = np.full(len(points_base), -1, np.int16)
    residual = np.full(len(points_base), np.inf)
    for index, view in enumerate(views):
        registered_camera = np.asarray(view['target_transform']) @ np.asarray(view['T_base_camera'])
        camera = transform(points_base, np.linalg.inv(registered_camera))
        intr = view['intrinsics']
        valid = np.isfinite(camera).all(1) & (camera[:, 2] > 0)
        ids = np.flatnonzero(valid)
        u = np.rint(intr['fx']*camera[ids, 0]/camera[ids, 2]+intr['cx']).astype(int)
        v = np.rint(intr['fy']*camera[ids, 1]/camera[ids, 2]+intr['cy']).astype(int)
        height, width = view['depth'].shape
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        ids, u, v = ids[inside], u[inside], v[inside]
        depth = view['depth'][v, u]
        error = np.abs(camera[ids, 2]-depth)
        supported = (view['mask'][v, u] & np.isfinite(depth) & (depth > 0)
                     & (error <= tolerance) & (error < residual[ids]))
        ids, u, v, error = ids[supported], u[supported], v[supported], error[supported]
        colors[ids] = view['rgb'][v, u]/255.
        source[ids], residual[ids] = index, error
    return colors, source, residual


def target_rgb(args, manifest):
    """Rebuild display colors from captures without changing model outputs."""
    import open3d as o3d
    result, report = {}, {}
    if not manifest.get('source_run'):
        return result
    capture_root = next(Path(manifest['source_run']).glob('*/active_fusion_registration.json')).parent
    registration = json.loads((capture_root/'active_fusion_registration.json').read_text())
    corrections = {item['name']: item['target_transform'] for item in registration['views']}
    views = {}
    for path in (capture_root/'multiview_raw').glob('*/metadata.json'):
        metadata = json.loads(path.read_text())
        folder = path.parent
        bgr = cv2.imread(str(folder/'color.png'))
        mask = cv2.imread(str(folder/'selected_target_mask.png'), 0)
        if bgr is None or mask is None:
            raise FileNotFoundError(f'Missing RGB or target mask in {folder}')
        views[metadata['view_label']] = dict(rgb=bgr[:, :, ::-1], mask=mask > 0,
            depth=np.load(folder/'depth_m.npy'), intrinsics=metadata['intrinsics'],
            T_base_camera=metadata['T_base_camera'],
            target_transform=corrections.get(metadata['view_label'], np.eye(4)))
    for name, filename in (('single','active_center_target_base_registered.ply'),
                           ('multi','active_fused_target_camera0.ply')):
        directory = args.output_dir/name
        cloud = o3d.io.read_point_cloud(str(capture_root/filename))
        measured, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
        if name == 'multi':
            measured = transform(measured, views['initial']['T_base_camera'])
        observed = np.load(directory/'observed_base.npy')
        np.testing.assert_allclose(measured, observed, atol=1e-8)
        if colors.shape != observed.shape:
            raise ValueError(f'{filename} lacks per-point RGB')
        result[(name, False)] = colors
        np.save(directory/'observed_rgb.npy', colors)
        save_cloud(directory/'observed_rgb_base.ply', observed, colors)
        if not manifest['cases'][name].get('prediction_available'):
            continue
        points = np.load(directory/'prediction_base.npy')
        labels = ['center'] if name == 'single' else ['center','side1']
        rgb, source, residual = project_rgb(points, [views[label] for label in labels])
        result[(name, True)] = rgb
        np.savez_compressed(directory/'prediction_rgb.npz', rgb=rgb, source_view=source,
                            depth_residual_m=residual, view_labels=np.asarray(labels))
        save_cloud(directory/'prediction_rgb_base.ply', points, rgb)
        report[name] = dict(points=len(points), rgb_supported=int((source >= 0).sum()),
            unobserved_grey=int((source < 0).sum()), tolerance_m=.006,
            views={label:int((source == i).sum()) for i,label in enumerate(labels)})
    report['method'] = 'Measured PLY RGB; prediction projected into original masked RGB-D using inverse target registration and camera pose. Minimum depth residual wins. Unsupported points grey; no model texture prediction.'
    (args.output_dir/'color_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return result


def prepare(args):
    import open3d as o3d
    from real_nero_grasp.camera_self_mask import calibrated_self_mask
    capture_root = next(args.run_dir.glob("*/active_fusion_registration.json")).parent
    registration = json.loads((capture_root / "active_fusion_registration.json").read_text())
    transforms = {item["name"]: np.asarray(item["background_transform"])
                  for item in registration["views"]}
    views = {}
    for meta_path in (capture_root / "multiview_raw").glob("*/metadata.json"):
        meta = json.loads(meta_path.read_text())
        views[meta["view_label"]] = (meta_path.parent, meta)
    folder, center_meta = views["center"]
    T_center = np.asarray(center_meta["T_base_camera"])
    initial_T = np.asarray(views["initial"][1]["T_base_camera"])
    image = cv2.imread(str(folder / "color.png"))
    visible = cv2.imread(str(folder / "selected_target_mask.png"), 0) > 0
    ys, xs = np.where(visible)
    # Crop target with neutral background and square padding; do not include
    # another object as a semantic hint to the completion network.
    foreground = image.copy()
    foreground[~visible] = 71
    box = (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
    crop = foreground[box[1]:box[3], box[0]:box[2]]
    size = int(np.ceil(max(crop.shape[:2])/.85))
    square = np.full((size, size, 3), 71, np.uint8)
    y, x = (size-crop.shape[0])//2, (size-crop.shape[1])//2
    square[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
    cv2.imwrite(str(args.output_dir / "target_rgb.png"), cv2.resize(square, (224, 224)))
    cv2.imwrite(str(args.output_dir / "source_center_rgb.png"), image)
    single_base = np.asarray(o3d.io.read_point_cloud(str(capture_root / "active_center_target_base_registered.ply")).points)
    fused_camera0 = np.asarray(o3d.io.read_point_cloud(str(capture_root / "active_fused_target_camera0.ply")).points)
    multi_base = transform(fused_camera0, initial_T)
    center = np.median(single_base, axis=0)
    background = {}
    for name in ("center", "side1"):
        source, meta = views[name]
        arrays = np.load(source / "clouds.npz")
        rgb = cv2.imread(str(source / "color.png"))
        sensor_meta = json.loads((source / "capture_meta.json").read_text())
        mask, mask_report = calibrated_self_mask(rgb, args.self_mask, sensor_meta)
        camera_points = arrays["obstacle_camera_points"]
        intr = meta["intrinsics"]
        u = np.rint(intr["fx"]*camera_points[:, 0]/camera_points[:, 2]+intr["cx"]).astype(int)
        v = np.rint(intr["fy"]*camera_points[:, 1]/camera_points[:, 2]+intr["cy"]).astype(int)
        inside = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        keep = np.ones(len(camera_points), bool)
        keep[inside] = mask[v[inside], u[inside]] == 0
        points = transform(camera_points[keep], meta["T_base_camera"])
        if name in transforms:
            points = transform(points, transforms[name])
        colors = arrays["obstacle_camera_colors"][keep]
        roi = ((np.abs(points[:, 0]-center[0]) < .30)
               & (np.abs(points[:, 1]-center[1]) < .30)
               & (points[:, 2] > -.015) & (points[:, 2] < .40))
        background[name] = (points[roi], colors[roi])
    np.savez_compressed(args.output_dir / "background.npz",
        single=background["center"][0], single_rgb=background["center"][1],
        multi=np.vstack([background[n][0] for n in ("center", "side1")]),
        multi_rgb=np.vstack([background[n][1] for n in ("center", "side1")]))
    manifest = dict(source_run=str(args.run_dir.resolve()), category="bottle", units="metre",
        frame="base_link", T_base_camera=T_center.tolist(), target_center_base=center.tolist(),
        rgb_crop_xyxy=list(box), shared_rgb="target_rgb.png", cases={},
        note="MGPC predicts only the target. Background stays measured; no GraspNet or robot I/O. Single view is the same center view used by multiview registration. Background side view uses the saved background transform, not target ICP.")
    for name, points in (("single", single_base), ("multi", multi_base)):
        native_camera = transform(points, np.linalg.inv(T_center))
        sampled, indices = fps(native_camera)
        normalized, normalization = normalize(sampled)
        directory = args.output_dir / name
        directory.mkdir(exist_ok=True)
        np.save(directory / "input.npy", normalized)
        np.save(directory / "observed_base.npy", points)
        np.save(directory / "sample_indices.npy", indices)
        save_cloud(directory / "observed_base.ply", points)
        np.testing.assert_allclose(denormalize(normalized, normalization), sampled, atol=1e-7)
        manifest["cases"][name] = dict(normalization=normalization, observed_points=len(points),
            model_input_points=len(normalized), prediction_available=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared:", args.output_dir, flush=True)


def inference(args):
    import torch
    if not args.checkpoint.exists() and args.wait_for_checkpoint > 0:
        print("Waiting for the official checkpoint download to finish...",flush=True)
        deadline=time.monotonic()+args.wait_for_checkpoint
        while not args.checkpoint.exists() and time.monotonic()<deadline:
            time.sleep(3)
    if not args.checkpoint.exists(): raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not accessible; run this offline script with GPU access")
    import clip
    from transformers import AutoModel, Dinov2Config, Dinov2Model
    from torchvision import transforms as vision
    from PIL import Image
    from unittest.mock import patch
    repository = args.repository.resolve()
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "extensions/pointnet2_ops_lib"))
    # Same PointNet++ C++ API, already compiled against the installed Torch.
    # Use official MGPC Python wrappers; no approximate geometry operators.
    sys.modules["pointnet2_ops._ext"] = importlib.import_module("pointnet2._ext")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    # Author checkpoint stores argparse.Namespace metadata alongside tensors.
    # Whitelist that inert metadata class; retain weights_only deserialization.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module.").removeprefix("model."): value for key, value in state.items()}
    clip_prefix = "text_encoder.model.clip_model."
    clip_state = {key[len(clip_prefix):]: value for key, value in state.items() if key.startswith(clip_prefix)}
    if not clip_state: raise ValueError("Official checkpoint does not contain CLIP weights")
    # The official checkpoint contains frozen encoders. Instantiate the exact
    # architectures without downloading another copy of their initial weights.
    config = Dinov2Config()
    position_key = "rgb_encoder.model.embeddings.position_embeddings"
    if position_key not in state: raise ValueError("Checkpoint lacks DINOv2 encoder weights")
    patches = state[position_key].shape[1]-1
    config.image_size = int(round(np.sqrt(patches)))*config.patch_size
    def build_dino(*_args, **_kwargs): return Dinov2Model(config)
    def build_clip(*_args, **_kwargs): return clip.model.build_model(clip_state), None
    module = importlib.import_module("models.MGPC")
    with patch.object(AutoModel, "from_pretrained", side_effect=build_dino), patch.object(clip, "load", side_effect=build_clip):
        model = module.MGPC()
    model.load_state_dict(state, strict=True)
    del checkpoint, state, clip_state
    model = model.eval().to("cuda")
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    image_transform = vision.Compose([vision.Resize(224, interpolation=vision.InterpolationMode.BICUBIC),
        vision.ToTensor(), vision.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    image = image_transform(Image.open(args.output_dir / "target_rgb.png").convert("RGB")).unsqueeze(0).cuda()
    prompt = [f'This is a/an "{manifest["category"]}"']
    for name in ("single", "multi"):
        directory = args.output_dir / name
        points = torch.from_numpy(np.load(directory / "input.npy")).float().unsqueeze(0).cuda()
        started = time.monotonic()
        with torch.inference_mode():
            prediction, _, _ = model.predict(points, image, prompt)
        torch.cuda.synchronize()
        normalized = prediction[0].cpu().numpy()
        if normalized.shape != (8192, 3) or not np.isfinite(normalized).all():
            raise ValueError("Invalid MGPC output")
        camera = denormalize(normalized, manifest["cases"][name]["normalization"])
        base = transform(camera, manifest["T_base_camera"])
        np.save(directory / "prediction_normalized.npy", normalized)
        np.save(directory / "prediction_base.npy", base)
        save_cloud(directory / "prediction_base.ply", base, np.tile([1., .48, .15], (len(base), 1)))
        manifest["cases"][name].update(prediction_available=True, prediction_points=len(base),
                                       inference_seconds=time.monotonic()-started)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Predicted", name, manifest["cases"][name], flush=True)
    manifest["checkpoint"] = dict(path=str(args.checkpoint.resolve()), bytes=args.checkpoint.stat().st_size,
                                  strict_state_dict=True, pointnet_backend="official wrappers + installed pointnet2._ext CUDA binary")
    digest=hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):digest.update(block)
    manifest["checkpoint"]["sha256"]=digest.hexdigest()
    import subprocess
    manifest["checkpoint"]["official_repository_commit"]=subprocess.check_output(
        ["git","-C",str(repository),"rev-parse","HEAD"],text=True).strip()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def check_gpu(args):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    sys.path.insert(0,str(args.repository.resolve()/"extensions/pointnet2_ops_lib"))
    sys.modules["pointnet2_ops._ext"]=importlib.import_module("pointnet2._ext")
    from pointnet2_ops.pointnet2_utils import furthest_point_sample,gather_operation,grouping_operation
    rng=np.random.default_rng(42)
    xyz=rng.uniform(.1,1.,size=(32,3)).astype(np.float32)
    tensor=torch.from_numpy(xyz).unsqueeze(0).cuda()
    ids=furthest_point_sample(tensor,16)
    _,reference=fps(xyz,16)
    np.testing.assert_array_equal(ids.cpu().numpy()[0],reference)
    features=torch.arange(96,dtype=torch.float32,device="cuda").reshape(1,3,32)
    np.testing.assert_array_equal(gather_operation(features,ids).cpu(),features[:,:,ids[0].long()].cpu())
    groups=ids[:,:8].reshape(1,2,4).contiguous()
    expected=features[:,:,groups[0].long()]
    np.testing.assert_array_equal(grouping_operation(features,groups).cpu(),expected.cpu())
    info=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
              total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
              operators="official PointNet++ Python wrappers with existing compiled CUDA backend; FPS/gather/group exact checks passed")
    (args.output_dir/"gpu_check.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info,indent=2),flush=True)


def render(args):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from scipy.spatial import cKDTree
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    background = np.load(args.output_dir / "background.npz")
    rgb_targets = target_rgb(args, manifest)
    center = np.asarray(manifest["target_center_base"])
    titles = ["A · 单视角实测", "B · 多视角融合实测", "C · 单视角 → MGPC", "D · 多视角融合 → MGPC"]
    figure = make_subplots(rows=2, cols=2, specs=[[{"type":"scene"}]*2]*2,
                          subplot_titles=titles, horizontal_spacing=.025, vertical_spacing=.06)
    trace_roles, available, summaries, target_sets, target_colors = [], [], [], [], []
    for index, (name, predict) in enumerate((("single", False), ("multi", False), ("single", True), ("multi", True))):
        row, col = index//2+1, index%2+1
        observed = np.load(args.output_dir / name / "observed_base.npy")
        env, rgb = background[name], background[name+"_rgb"]
        color = ["rgb(%d,%d,%d)" % tuple(c) for c in np.clip(rgb*255, 0, 255).astype(int)]
        figure.add_trace(go.Scatter3d(x=env[:,0], y=env[:,1], z=env[:,2], mode="markers",
            marker=dict(size=1.3, color=color, opacity=.85), name="实测环境", showlegend=index==0,
            hovertemplate="环境实测<br>X=%{x:.4f} m<br>Y=%{y:.4f} m<br>Z=%{z:.4f} m<extra></extra>"), row=row, col=col)
        trace_roles.append("environment")
        present = bool(manifest["cases"][name].get("prediction_available")) if predict else True
        target = np.load(args.output_dir / name / "prediction_base.npy") if predict and present else observed
        if predict and not present: target = np.empty((0,3))
        if len(target): target_sets.append(target)
        semantic_color = "#ff8a32" if predict else "#08c9d6"
        rgb_color = rgb_targets.get((name, predict))
        display_color = (["rgb(%d,%d,%d)" % tuple(c) for c in np.clip(rgb_color*255,0,255).astype(int)]
                         if rgb_color is not None else semantic_color)
        target_colors.append(dict(index=len(trace_roles), rgb=display_color, source=semantic_color))
        if present:
            output_name = ("C_single_mgpc" if name=="single" else "D_multi_mgpc") if predict else ("A_single_measured" if name=="single" else "B_multi_measured")
            target_color = [1.,.54,.20] if predict else [0.,.78,.84]
            save_cloud(args.output_dir/(output_name+"_scene_base.ply"),np.vstack([env,target]),
                       np.vstack([rgb,np.tile(target_color,(len(target),1))]))
            if rgb_color is not None:
                save_cloud(args.output_dir/(output_name+"_rgb_scene_base.ply"),np.vstack([env,target]),
                           np.vstack([rgb,rgb_color]))
        label = "MGPC 预测目标（整组点）" if predict else "实测目标"
        figure.add_trace(go.Scatter3d(x=target[:,0], y=target[:,1], z=target[:,2], mode="markers",
            marker=dict(size=1.7, color=display_color, opacity=1.),
            name=label, showlegend=index in (0,2),
            hovertemplate=label+"<br>X=%{x:.4f} m<br>Y=%{y:.4f} m<br>Z=%{z:.4f} m<extra></extra>"), row=row, col=col)
        trace_roles.append("target")
        figure.add_trace(go.Scatter3d(x=observed[:,0] if predict else [], y=observed[:,1] if predict else [],
            z=observed[:,2] if predict else [], mode="markers", marker=dict(size=1.4, color="#08c9d6", opacity=.3),
            name="预测面板中的实测参考", showlegend=index==2, visible=False), row=row, col=col)
        trace_roles.append("reference")
        available.append(present)
        if predict and present:
            distances = cKDTree(target).query(observed)[0]*1000
            stats = dict(observed_to_prediction_nn_mm=np.percentile(distances, [50,95]).tolist(),
                         predicted_bbox_mm=(np.ptp(target,axis=0)*1000).tolist(),
                         observed_bbox_mm=(np.ptp(observed,axis=0)*1000).tolist())
            summaries.append(dict(case=titles[index], **stats))
        figure.update_scenes(dict(aspectmode="data", camera=dict(eye=dict(x=1.35,y=-1.65,z=1.15)),
            xaxis=dict(title="基座 X / m", range=[center[0]-.3,center[0]+.3]),
            yaxis=dict(title="基座 Y / m", range=[center[1]-.3,center[1]+.3]),
            zaxis=dict(title="基座 Z / m", range=[-.02,.40])), row=row,col=col)
        if not present:
            figure.add_annotation(text="尚未生成预测：等待官方模型权重 / 推理", x=.24 if col==1 else .76,
                                  y=.43 if row==1 else .01, xref="paper", yref="paper", showarrow=False,
                                  font=dict(color="#ff8a32",size=15))
    figure.update_layout(height=1050, margin=dict(l=0,r=0,b=0,t=45), paper_bgcolor="#101821",
        font=dict(color="#dce7f0"), legend=dict(orientation="h",y=1.04), uirevision="fixed_camera")
    for key in ("scene","scene2","scene3","scene4"):
        figure.layout[key].bgcolor="#101821"
    plot = figure.to_html(full_html=False, include_plotlyjs=True, div_id="comparison",
                          config=dict(scrollZoom=True,displaylogo=False))
    bounds = np.vstack(target_sets)
    target_midpoint = (bounds.min(0)+bounds.max(0))*.5
    target_radius = max(float(np.ptp(bounds,axis=0).max())*.65,.04)
    text = "四组结果已完成" if all(available) else "实测已准备；预测面板尚未完成"
    # Local HTML includes the plotting library and all points: no CDN/network.
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MGPC 离线重建对比</title>
<style>body{{background:#101821;color:#dce7f0;font:15px system-ui;margin:20px}}h1{{font-size:25px;margin-bottom:8px}}button{{padding:8px 15px;margin:8px 8px 0 0;background:#24374a;color:white;border:1px solid #53718a;border-radius:6px}}.note{{max-width:1100px;line-height:1.8;color:#b7cad9}}img{{width:100px;vertical-align:middle;margin-left:15px}}a{{color:#4fd8e0}}</style>
<h1>MGPC · 四种输入的离线重建对比</h1><div class="note">{text} · 同一中心视角 RGB 输入，坐标为真实米制 base_link。默认显示 RGB：实测目标保留原色；预测目标仅将深度一致的可见表面映射到保存的 RGB，未观测部分为灰色（不是模型预测的纹理）。来源配色模式：青色＝实测，橙色＝预测。MGPC 未预测桌面、方块或其他障碍。没有经过 GraspNet。</div>
<div><button onclick="colorMode(true)">RGB 颜色</button><button onclick="colorMode(false)">来源配色（青／橙）</button><button onclick="mode(false)">目标＋环境</button><button onclick="mode(true)">只看目标形状</button><button onclick="reference()">显示 / 隐藏实测参考</button><button onclick="resetView()">重置同步视角</button><label><input id="sync" type="checkbox" checked>同步四窗旋转</label><img src="target_rgb.png" title="共享目标 RGB 输入"></div>
<div class="note">拖动旋转，滚轮缩放。A/C 共用单视角环境，B/D 共用多视角环境；预测不会覆盖或删除实测背景。当前为 22:19 保存数据中的短瓶，23:17 的长瓶只有一个视角，不能用于同场景四组比较。</div>{plot}
<script>const p=document.getElementById('comparison');const roles={json.dumps(trace_roles)};let lock=false,ref=false;
const targetColors={json.dumps(target_colors)};
function colorMode(rgb){{targetColors.forEach(t=>Plotly.restyle(p,{{'marker.color':[rgb?t.rgb:t.source]}},[t.index]));}}
function mode(targetOnly){{roles.forEach((r,i)=>{{if(r==='environment')Plotly.restyle(p,{{visible:!targetOnly}},[i]);}});const c=targetOnly?{json.dumps(target_midpoint.tolist())}:{json.dumps(center.tolist())};const width=targetOnly?{target_radius}:.30;const patch={{}};['scene','scene2','scene3','scene4'].forEach(s=>{{patch[s+'.xaxis.range']=[c[0]-width,c[0]+width];patch[s+'.yaxis.range']=[c[1]-width,c[1]+width];patch[s+'.zaxis.range']=targetOnly?[c[2]-width,c[2]+width]:[-.02,.40];}});Plotly.relayout(p,patch);}}
function reference(){{ref=!ref;roles.forEach((r,i)=>{{if(r==='reference')Plotly.restyle(p,{{visible:ref}},[i]);}});}}
function resetView(){{const patch={{}};['scene','scene2','scene3','scene4'].forEach(s=>patch[s+'.camera']={{eye:{{x:1.35,y:-1.65,z:1.15}}}});Plotly.relayout(p,patch);}}
p.on('plotly_relayout',e=>{{if(lock||!document.getElementById('sync').checked)return;const k=Object.keys(e).find(k=>/^scene[234]?\\.camera$/.test(k));if(!k)return;lock=true;const patch={{}};['scene','scene2','scene3','scene4'].forEach(s=>{{if(s+'.camera'!==k)patch[s+'.camera']=e[k];}});Plotly.relayout(p,patch).then(()=>lock=false);}});
function fit(){{Plotly.relayout(p,{{height:Math.max(620,window.innerHeight-220)}});}}window.addEventListener('resize',fit);fit();</script></html>'''
    (args.output_dir / "comparison.html").write_text(page, encoding="utf-8")
    (args.output_dir / "geometry_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    render_snapshots(args, manifest, background, rgb_targets)
    print("Viewer:", args.output_dir / "comparison.html", flush=True)


def render_snapshots(args, manifest, background, rgb_targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = [("single",False,"A. Single measured"),("multi",False,"B. Multiview measured"),
             ("single",True,"C. Single + MGPC"),("multi",True,"D. Multiview + MGPC")]
    target_arrays = []
    for name, predicted, _ in cases:
        file = args.output_dir / name / ("prediction_base.npy" if predicted else "observed_base.npy")
        present = not predicted or manifest["cases"][name].get("prediction_available",False)
        target_arrays.append(np.load(file) if present else np.empty((0,3)))
    all_points = np.vstack([points for points in target_arrays if len(points)])
    midpoint = (all_points.min(0)+all_points.max(0))*.5
    radius = max(float(np.ptp(all_points,axis=0).max())*.65,.04)
    center = np.asarray(manifest["target_center_base"])
    for overview, filename in ((True,"environment_comparison.png"),(False,"target_comparison.png")):
        fig = plt.figure(figsize=(14,10),facecolor="#f5f7fa")
        for i,((name,predicted,title),points) in enumerate(zip(cases,target_arrays)):
            ax = fig.add_subplot(2,2,i+1,projection="3d",proj_type="ortho")
            if overview:
                env,colors = background[name],background[name+"_rgb"]
                stride = max(1,len(env)//14000)
                ax.scatter(*env[::stride].T,c=colors[::stride],s=.5,alpha=.8,rasterized=True)
                ax.set(xlim=(center[0]-.30,center[0]+.30),ylim=(center[1]-.30,center[1]+.30),zlim=(-.02,.40))
            else:
                ax.set(xlim=(midpoint[0]-radius,midpoint[0]+radius),
                       ylim=(midpoint[1]-radius,midpoint[1]+radius),zlim=(midpoint[2]-radius,midpoint[2]+radius))
            if len(points):
                colors = rgb_targets.get((name,predicted), "#ff8a32" if predicted else "#08b9c6")
                ax.scatter(*points.T,c=colors,s=1.4,alpha=1.,rasterized=True)
            else:
                ax.text2D(.15,.45,"Prediction not generated yet",transform=ax.transAxes,color="#ca6d24")
            ax.view_init(elev=27 if overview else 15,azim=-60)
            ax.set_box_aspect((1,1,.70) if overview else (1,1,1))
            ax.set_title(title,pad=4)
            ax.set_xlabel("Base X (m)",labelpad=2)
            ax.set_ylabel("Base Y (m)",labelpad=2)
            ax.set_zlabel("Base Z (m)",labelpad=2)
            ax.tick_params(labelsize=7)
        fig.suptitle("Offline reconstruction | observed RGB texture | unobserved prediction: grey | no GraspNet",fontsize=13)
        fig.subplots_adjust(left=.01,right=.98,bottom=.025,top=.93,wspace=.01,hspace=.02)
        fig.savefig(args.output_dir/filename,dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repository", type=Path, default=ROOT / "third_party/MGPC")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "third_party/MGPC/pretrained/ckpt_8192.pt")
    parser.add_argument("--self-mask", type=Path, default=ROOT / "config/2026-10-22_camera_self_mask.json")
    parser.add_argument("--wait-for-checkpoint",type=float,default=0)
    parser.add_argument("--stage", choices=("prepare","infer","render","gpu-check","all"),default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage=="gpu-check": check_gpu(args)
    if args.stage in ("prepare","all"): prepare(args)
    if args.stage in ("infer","all"): inference(args)
    if args.stage in ("render","all"): render(args)

if __name__ == "__main__": main()
