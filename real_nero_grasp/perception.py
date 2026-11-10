"""RGB-D capture, target selection, and perception cloud construction.

The high-level runtime injects its public/private helpers through keyword-only
hooks.  That keeps legacy monkey-patching behavior intact while this module
remains independently importable and does not depend on ``runtime``.
"""

import copy
import json
import os
import time

import cv2
import numpy as np

from .depth_processing import (
    _mask_center,
    _mask_iou,
    _mask_nested_overlap,
    _median_depth_for_mask,
    _resize_mask,
    depth_to_xyz_image,
    pointcloud2_xyz_at_from_image,
    pointcloud2_xyz_image,
    refine_xyz_inside_visible_mask,
)
from .paths import DEFAULT_YOLO_WEIGHTS, PROJECT_ROOT
from .types import TargetIdentityNotVerified
from .visualization import get_o3d


ROOT_DIR = str(PROJECT_ROOT)

__all__ = [
    "draw_instances",
    "_class_matches",
    "infer_yolo_instances",
    "mask_for_instance",
    "reidentify_instance_from_reference_mask",
    "click_point",
    "save_selected_target_visuals",
    "apply_target_prompt",
    "largest_visible_clear_instance",
    "select_target_mask",
    "mask_from_projected_target_center",
    "mask_from_clicked_target_point",
    "cloud_from_mask",
    "scene_cloud_from_xyz",
    "obstacle_cloud_from_non_target_mask",
    "crop_scene_cloud_near_target",
    "_deduplicate_strict_identity_candidates",
    "camera_self_occlusion_mask",
    "apply_camera_self_occlusion_mask",
    "camera_near_field_ghost_mask",
    "apply_camera_near_field_ghost_filter",
    "capture_rgbd_xyz_frame",
    "capture_target_cloud",
    "capture_target_cloud_with_retries",
    "recapture_original_target_after_clear",
    "_active_view_second_capture_args",
    "capture_active_projected_target_cloud",
]


def draw_instances(bgr, instances):
    out = bgr.copy()
    for idx, inst in enumerate(instances):
        mask = np.asarray(inst["mask"]) > 0
        color = (
            int((37 * (idx + 3)) % 255),
            int((97 * (idx + 5)) % 255),
            int((173 * (idx + 7)) % 255),
        )
        out[mask] = (
            0.55 * out[mask] + 0.45 * np.asarray(color, dtype=np.uint8)
        ).astype(np.uint8)
        x1, y1, x2, y2 = (
            np.asarray(inst["xyxy"], dtype=float).astype(int).tolist()
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{idx}:{inst['label']} {float(inst['score']):.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _class_matches(label, target_class):
    if target_class is None or str(target_class).strip() == "":
        return True
    label_norm = str(label).strip().lower().replace("_", " ")
    target_norm = str(target_class).strip().lower().replace("_", " ")
    return (
        label_norm == target_norm
        or target_norm in label_norm
        or label_norm in target_norm
    )


def infer_yolo_instances(
    bgr,
    args,
    output_vis_path=None,
    *,
    class_matches_hook=None,
    resize_mask_hook=None,
    mask_center_hook=None,
):
    class_matches_fn = _class_matches if class_matches_hook is None else class_matches_hook
    resize_mask_fn = _resize_mask if resize_mask_hook is None else resize_mask_hook
    mask_center_fn = _mask_center if mask_center_hook is None else mask_center_hook
    weights = args.yolo_weights or DEFAULT_YOLO_WEIGHTS
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}. Put yolov8s-seg.pt in {ROOT_DIR}, "
            "or pass --yolo-weights, or use --manual-click-only."
        )

    from ultralytics import YOLO

    model = YOLO(weights)
    if args.yolo_classes and hasattr(model, "set_classes"):
        classes = [
            x.strip() for x in str(args.yolo_classes).split(",") if x.strip()
        ]
        if classes:
            model.set_classes(classes)

    results = model.predict(
        bgr,
        conf=float(args.yolo_conf),
        imgsz=int(args.yolo_imgsz),
        verbose=False,
    )
    result = results[0]
    vis = result.plot() if hasattr(result, "plot") else bgr.copy()
    if output_vis_path is not None and vis is not None:
        cv2.imwrite(output_vis_path, vis)

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], vis

    masks = getattr(result, "masks", None)
    masks_np = (
        None
        if masks is None or masks.data is None
        else masks.data.detach().cpu().numpy()
    )
    names = getattr(result, "names", {}) or {}
    height, width = bgr.shape[:2]
    instances = []
    for i, box in enumerate(boxes):
        conf = float(box.conf.item())
        cls_id = int(box.cls.item()) if hasattr(box, "cls") else -1
        label = str(names.get(cls_id, cls_id))
        if conf < float(args.yolo_conf) or not class_matches_fn(
            label, args.target_class
        ):
            continue
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
        is_box_only = True
        if masks_np is not None and i < len(masks_np):
            mask = resize_mask_fn(masks_np[i], width, height)
            is_box_only = False
        else:
            x1, y1, x2, y2 = np.round(xyxy).astype(int).tolist()
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)
            mask = np.zeros((height, width), dtype=np.uint8)
            if x2 > x1 and y2 > y1:
                mask[y1 : y2 + 1, x1 : x2 + 1] = 1
        if int(np.count_nonzero(mask)) == 0:
            continue
        instances.append(
            {
                "label": label,
                "score": conf,
                "xyxy": xyxy,
                "mask": mask,
                "center": mask_center_fn(mask),
                "is_box_only": bool(is_box_only),
            }
        )
    instances.sort(
        key=lambda x: (
            float(x["score"]),
            int(np.count_nonzero(x["mask"])),
        ),
        reverse=True,
    )
    return instances, vis


def mask_for_instance(inst, xyz_img, args):
    mask = np.asarray(inst["mask"]) > 0
    if not bool(inst.get("is_box_only", False)):
        return mask
    valid = (
        mask
        & np.isfinite(xyz_img).all(axis=2)
        & (xyz_img[:, :, 2] > args.min_depth)
        & (xyz_img[:, :, 2] < args.max_depth)
    )
    zs = xyz_img[:, :, 2]
    center = inst.get("center", None)
    seed_z = None
    if center is not None:
        u, v = int(center[0]), int(center[1])
        if (
            0 <= v < xyz_img.shape[0]
            and 0 <= u < xyz_img.shape[1]
            and valid[v, u]
        ):
            seed_z = float(zs[v, u])
    if seed_z is None:
        zvals = zs[valid]
        if zvals.size == 0:
            return valid
        seed_z = float(np.percentile(zvals, 35.0))
    refined = valid & (np.abs(zs - seed_z) <= float(args.box_depth_band))
    if int(np.count_nonzero(refined)) < int(args.min_target_points):
        refined = valid & (zs <= seed_z + float(args.box_depth_band))
    return refined


def reidentify_instance_from_reference_mask(
    instances,
    xyz_img,
    reference_mask,
    reference_label,
    args,
    *,
    mask_for_instance_hook=None,
    class_matches_hook=None,
    mask_center_hook=None,
    mask_iou_hook=None,
):
    """Track the previously selected target after returning to the same camera view."""
    mask_for_instance_fn = (
        mask_for_instance
        if mask_for_instance_hook is None
        else mask_for_instance_hook
    )
    class_matches_fn = _class_matches if class_matches_hook is None else class_matches_hook
    mask_center_fn = _mask_center if mask_center_hook is None else mask_center_hook
    mask_iou_fn = _mask_iou if mask_iou_hook is None else mask_iou_hook
    reference = np.asarray(reference_mask) > 0
    reference_area = int(np.count_nonzero(reference))
    if reference.ndim != 2 or reference_area == 0:
        return None, None, "reference mask is empty"

    label = str(reference_label or "").strip()
    label_is_specific = label.lower() not in {"", "manual", "unknown", "target"}
    reference_center = mask_center_fn(reference)
    image_diag = float(np.hypot(reference.shape[0], reference.shape[1]))
    rows = []
    for inst in instances:
        if label_is_specific and not class_matches_fn(inst.get("label", ""), label):
            continue
        try:
            mask = np.asarray(mask_for_instance_fn(inst, xyz_img, args)) > 0
        except Exception:
            continue
        if mask.shape != reference.shape:
            continue
        intersection = int(np.count_nonzero(mask & reference))
        coverage = float(intersection / max(reference_area, 1))
        iou = mask_iou_fn(mask, reference)
        center = mask_center_fn(mask)
        if reference_center is None or center is None:
            center_distance_norm = float("inf")
        else:
            center_distance_norm = float(
                np.hypot(
                    float(center[0] - reference_center[0]),
                    float(center[1] - reference_center[1]),
                )
                / max(image_diag, 1.0)
            )
        rows.append(
            {
                "inst": inst,
                "mask": mask,
                "intersection": intersection,
                "coverage": coverage,
                "iou": iou,
                "center_distance_norm": center_distance_norm,
                "confidence": float(inst.get("score", 0.0)),
            }
        )

    if not rows:
        return None, None, f"no YOLO instance matches reference label {label!r}"
    rows.sort(
        key=lambda row: (
            float(row["coverage"]),
            float(row["iou"]),
            -float(row["center_distance_norm"]),
            float(row["confidence"]),
        ),
        reverse=True,
    )
    best = rows[0]
    min_coverage = float(getattr(args, "post_clear_reidentify_min_coverage", 0.30))
    if float(best["coverage"]) < min_coverage:
        return (
            None,
            None,
            f"best same-view reference coverage={best['coverage']:.3f} < {min_coverage:.3f}",
        )

    min_iou = float(getattr(args, "post_clear_reidentify_min_iou", 0.10))
    if float(best["iou"]) < min_iou:
        return (
            None,
            None,
            f"best same-view mask IoU={best['iou']:.3f} < {min_iou:.3f}",
        )

    ambiguity_margin = max(
        float(getattr(args, "post_clear_reidentify_ambiguity_margin", 0.15)),
        0.0,
    )
    if len(rows) > 1:
        second = rows[1]
        if (
            float(second["coverage"]) >= min_coverage
            and float(best["coverage"]) - float(second["coverage"])
            < ambiguity_margin
        ):
            return (
                None,
                None,
                "same-class reference match is ambiguous: "
                f"coverage={best['coverage']:.3f} and {second['coverage']:.3f}",
            )

    detail = (
        f"label={best['inst'].get('label', 'unknown')} "
        f"reference_coverage={best['coverage']:.3f} iou={best['iou']:.3f} "
        f"center_distance_norm={best['center_distance_norm']:.3f}"
    )
    return best["mask"], best["inst"], detail


def click_point(image, title):
    clicked = []

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[:] = [(int(x), int(y))]
            vis = image.copy()
            cv2.drawMarker(
                vis,
                (int(x), int(y)),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )
            cv2.imshow(title, vis)

    cv2.imshow(title, image)
    cv2.setMouseCallback(title, cb)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if clicked and key in (13, 10):
            cv2.destroyWindow(title)
            return clicked[0]
        if key in (27, ord("q")):
            cv2.destroyWindow(title)
            return None


def save_selected_target_visuals(bgr, mask, inst, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mask = np.asarray(mask).astype(bool)
    mask_u8 = mask.astype(np.uint8) * 255
    overlay = bgr.copy()
    overlay[mask] = (
        0.45 * overlay[mask]
        + 0.55 * np.asarray([0, 255, 0], dtype=np.uint8)
    ).astype(np.uint8)
    center = inst.get("center", None) if isinstance(inst, dict) else None
    if center is not None:
        cv2.drawMarker(
            overlay,
            (int(center[0]), int(center[1])),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            24,
            2,
        )
    label = str(inst.get("label", "unknown")) if isinstance(inst, dict) else "unknown"
    score = float(inst.get("score", 0.0)) if isinstance(inst, dict) else 0.0
    cv2.putText(
        overlay,
        f"selected: {label} {score:.2f}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    mask_path = os.path.join(out_dir, "selected_target_mask.png")
    overlay_path = os.path.join(out_dir, "selected_target_overlay.png")
    cv2.imwrite(mask_path, mask_u8)
    cv2.imwrite(overlay_path, overlay)
    print(f"[select] saved selected target mask={mask_path} overlay={overlay_path}")


def apply_target_prompt(args):
    if not args.target_prompt:
        return
    print("\n[target] Choose target selection mode:")
    print("[target]   type object name, e.g. apple")
    print("[target]   type CLICK to choose a YOLO mask with mouse")
    print("[target]   type POINT to click one target pixel and build a local cloud")
    raw = input("[target] object name / CLICK / POINT: ").strip()
    args.target_prompt = False
    if raw == "":
        has_existing_target = (
            args.target_class
            or args.target_index is not None
            or bool(args.manual_click_only)
            or bool(args.select_target)
        )
        if has_existing_target:
            print("[target] empty input; using existing target arguments")
        else:
            args.target_class = None
            args.target_index = None
            args.manual_click_only = False
            args.select_target = True
            print("[target] empty input; default mode = YOLO mouse selection")
        return
    mode = raw.lower()
    if mode in ("click", "mouse", "select"):
        args.target_class = None
        args.target_index = None
        args.manual_click_only = False
        args.select_target = True
        print("[target] mode = YOLO mouse selection")
        return
    if mode in ("point", "pixel", "manual"):
        args.target_class = None
        args.target_index = None
        args.manual_click_only = True
        args.select_target = False
        print("[target] mode = manual pixel point")
        return
    args.target_class = raw
    args.target_index = None
    args.manual_click_only = False
    args.select_target = False
    print(f"[target] mode = typed target class: {raw}")


def largest_visible_clear_instance(
    bgr,
    xyz_img,
    args,
    out_dir,
    instances=None,
    exclude_mask=None,
    *,
    infer_yolo_instances_hook=None,
    mask_for_instance_hook=None,
    mask_iou_hook=None,
    median_depth_for_mask_hook=None,
):
    """Choose the largest visible YOLO mask with enough valid depth points."""
    infer_fn = (
        infer_yolo_instances
        if infer_yolo_instances_hook is None
        else infer_yolo_instances_hook
    )
    mask_fn = mask_for_instance if mask_for_instance_hook is None else mask_for_instance_hook
    mask_iou_fn = _mask_iou if mask_iou_hook is None else mask_iou_hook
    median_depth_fn = (
        _median_depth_for_mask
        if median_depth_for_mask_hook is None
        else median_depth_for_mask_hook
    )
    if instances is None:
        all_args = copy.copy(args)
        all_args.target_class = None
        all_args.target_index = None
        all_args.select_target = False
        all_args.manual_click_only = False
        vis_path = os.path.join(out_dir, "largest_visible_fallback_instances.png")
        try:
            instances, _ = infer_fn(bgr, all_args, output_vis_path=vis_path)
        except Exception as exc:
            print(
                "[clear][largest][warn] all-class YOLO inference failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    exclude = None if exclude_mask is None else np.asarray(exclude_mask) > 0
    min_points = max(int(args.clear_min_points), int(args.min_target_points))
    candidates = []
    for source_idx, source_inst in enumerate(instances):
        raw_mask = np.asarray(source_inst.get("mask", 0)) > 0
        if raw_mask.ndim != 2 or int(np.count_nonzero(raw_mask)) == 0:
            continue
        if exclude is not None and mask_iou_fn(raw_mask, exclude) >= float(
            args.clear_skip_target_iou
        ):
            continue
        try:
            mask = np.asarray(mask_fn(source_inst, xyz_img, args)) > 0
        except Exception:
            mask = raw_mask
        valid = (
            mask
            & np.isfinite(xyz_img).all(axis=2)
            & (xyz_img[:, :, 2] > float(args.min_depth))
            & (xyz_img[:, :, 2] < float(args.max_depth))
        )
        valid_points = int(np.count_nonzero(valid))
        if valid_points < min_points:
            continue
        depth = median_depth_fn(xyz_img, valid, args)
        if not np.isfinite(depth):
            continue
        candidates.append(
            {
                "source_index": int(source_idx),
                "inst": dict(source_inst),
                "mask": valid.astype(np.uint8),
                "visible_area_px": int(np.count_nonzero(mask)),
                "valid_depth_points": valid_points,
                "depth": float(depth),
            }
        )
    if not candidates:
        print(
            "[clear][largest][warn] no visible YOLO instance has enough valid depth points"
        )
        return None
    candidates.sort(
        key=lambda item: (
            int(item["visible_area_px"]),
            int(item["valid_depth_points"]),
            float(item["inst"].get("score", 0.0)),
        ),
        reverse=True,
    )
    chosen = candidates[0]
    print(
        f"[clear][largest] selected label={chosen['inst'].get('label', 'unknown')} "
        f"visible_area={chosen['visible_area_px']} "
        f"valid_depth={chosen['valid_depth_points']} "
        f"score={float(chosen['inst'].get('score', 0.0)):.3f}"
    )
    return chosen

def _label_from_command(args, observed="manual"):
    requested = str(getattr(args, "target_class", None) or "").strip()
    observed = str(observed or "manual")
    if requested:
        return requested, requested
    return observed, None

def select_target_mask(
    bgr,
    xyz_img,
    args,
    out_dir,
    *,
    infer_yolo_instances_hook=None,
    reidentify_instance_hook=None,
    mask_for_instance_hook=None,
    save_selected_visuals_hook=None,
    draw_instances_hook=None,
    click_point_hook=None,
    largest_visible_instance_hook=None,
    pointcloud2_xyz_at_hook=None,
):
    infer_fn = (
        infer_yolo_instances
        if infer_yolo_instances_hook is None
        else infer_yolo_instances_hook
    )
    reidentify_fn = (
        reidentify_instance_from_reference_mask
        if reidentify_instance_hook is None
        else reidentify_instance_hook
    )
    mask_fn = mask_for_instance if mask_for_instance_hook is None else mask_for_instance_hook
    save_fn = (
        save_selected_target_visuals
        if save_selected_visuals_hook is None
        else save_selected_visuals_hook
    )
    draw_fn = draw_instances if draw_instances_hook is None else draw_instances_hook
    click_fn = click_point if click_point_hook is None else click_point_hook
    largest_fn = (
        largest_visible_clear_instance
        if largest_visible_instance_hook is None
        else largest_visible_instance_hook
    )
    xyz_at_fn = (
        pointcloud2_xyz_at_from_image
        if pointcloud2_xyz_at_hook is None
        else pointcloud2_xyz_at_hook
    )
    instances = []
    if not args.manual_click_only:
        vis_path = os.path.join(out_dir, "yolo_instances.png")
        os.makedirs(out_dir, exist_ok=True)
        instances, vis = infer_fn(bgr, args, output_vis_path=vis_path)
        print(f"[yolo] instances={len(instances)} overlay={vis_path}")

    reference_mask = getattr(args, "_post_clear_reference_mask", None)
    reference_label = getattr(args, "_post_clear_reference_label", None)
    reference_requested = reference_mask is not None
    if len(instances) > 0:
        if reference_requested:
            mask, tracked_inst, detail = reidentify_fn(
                instances,
                xyz_img,
                reference_mask,
                reference_label,
                args,
            )
            if mask is not None and tracked_inst is not None:
                print(
                    f"[select][reidentify] original target tracked automatically: {detail}"
                )
                save_fn(bgr, mask, tracked_inst, out_dir)
                return mask, tracked_inst
            print(
                f"[select][reidentify][warn] {detail}; "
                "automatic reassociation is forbidden"
            )
            raise TargetIdentityNotVerified(
                f"original target {str(reference_label or 'unknown')!r} "
                f"could not be re-identified: {detail}"
            )

        if args.target_index is not None and not reference_requested:
            idx = int(args.target_index)
            if not (0 <= idx < len(instances)):
                raise ValueError(
                    f"--target-index {idx} outside [0, {len(instances)-1}]"
                )
            mask = mask_fn(instances[idx], xyz_img, args)
            save_fn(bgr, mask, instances[idx], out_dir)
            return mask, instances[idx]

        if args.target_class and not args.select_target and not reference_requested:
            mask = mask_fn(instances[0], xyz_img, args)
            save_fn(bgr, mask, instances[0], out_dir)
            return mask, instances[0]

        overlay = draw_fn(bgr, instances)
        print(
            "[select][wait] In the OpenCV window, click the required target mask, "
            "then press Enter while that window has focus (Esc/q cancels)."
        )
        uv = click_fn(overlay, "Click target instance/mask, then Enter")
        if uv is not None:
            u, v = uv
            for inst in instances:
                mask = np.asarray(inst["mask"]) > 0
                if (
                    0 <= v < mask.shape[0]
                    and 0 <= u < mask.shape[1]
                    and bool(mask[v, u])
                ):
                    observed = str(inst.get("label", "unknown"))
                    label, requested = _label_from_command(args, observed)
                    inst = dict(inst)
                    inst["label"] = label
                    inst["observed_label"] = observed
                    if requested:
                        inst["requested_target_class"] = requested
                    print(
                        f"[select] click assigned command class {label!r} "
                        f"(yolo={observed!r})"
                    )
                    mask = mask_fn(inst, xyz_img, args)
                    save_fn(bgr, mask, inst, out_dir)
                    return mask, inst
            print(
                "[select] click did not hit a YOLO mask; fallback to local point cloud sphere."
            )

    if reference_requested:
        raise TargetIdentityNotVerified(
            f"original target {str(reference_label or 'unknown')!r} could not be "
            "re-identified because the fresh semantic observation contains no "
            "usable YOLO instance"
        )

    requested_class = str(args.target_class or "").strip()
    if (
        len(instances) == 0
        and requested_class
        and not reference_requested
        and bool(args.enable_clear_obstacle)
        and bool(args.clear_largest_visible_fallback)
    ):
        fallback = largest_fn(bgr, xyz_img, args, out_dir)
        if fallback is not None:
            inst = dict(fallback["inst"])
            inst["clear_fallback_largest_visible"] = True
            inst["clear_fallback_reason"] = "requested_target_not_detected"
            inst["requested_target_class"] = requested_class
            inst["visible_area_px"] = int(fallback["visible_area_px"])
            mask = np.asarray(fallback["mask"], dtype=np.uint8)
            print(
                f"[target][fallback] requested target {requested_class!r} was not detected; "
                f"use largest visible {inst.get('label', 'unknown')!r} as a clear-only target"
            )
            save_fn(bgr, mask, inst, out_dir)
            return mask, inst

    print(
        "[select][wait] In the OpenCV window, click one target point, then press "
        "Enter while that window has focus (Esc/q cancels)."
    )
    uv = click_fn(bgr, "Click target point for local cloud, then Enter")
    if uv is None:
        raise RuntimeError("target selection cancelled")
    u, v = uv
    seed = xyz_at_fn(xyz_img, u, v)
    d = np.linalg.norm(xyz_img - seed.reshape(1, 1, 3), axis=2)
    valid = np.isfinite(xyz_img).all(axis=2) & (xyz_img[:, :, 2] > 0.03)
    mask = valid & (d <= float(args.manual_radius))
    # inst = {
    #     "label": "manual",
    #     "score": 1.0,
    #     "center": (u, v),
    #     "mask": mask.astype(np.uint8),
    # }
    label, requested = _label_from_command(args, "manual")
    inst = {
        "label": label,
        "score": 1.0,
        "center": (u, v),
        "mask": mask.astype(np.uint8),
        "observed_label": "manual",
        "selection": "click_sphere",
    }
    if requested:
        inst["requested_target_class"] = requested
    print(
        f"[select] click sphere assigned command class {label!r} "
        f"radius={float(args.manual_radius):.3f}m"
    )
    save_fn(bgr, mask, inst, out_dir)
    return mask, inst


def mask_from_projected_target_center(
    bgr,
    xyz_img,
    T_base_camera,
    target_center_base,
    args,
    out_dir,
    label="active_projected",
    *,
    save_selected_visuals_hook=None,
):
    save_fn = (
        save_selected_target_visuals
        if save_selected_visuals_hook is None
        else save_selected_visuals_hook
    )
    T_camera_base = np.linalg.inv(
        np.asarray(T_base_camera, dtype=float).reshape(4, 4)
    )
    p_cam_h = T_camera_base @ np.r_[
        np.asarray(target_center_base, dtype=float).reshape(3), 1.0
    ]
    p_cam = p_cam_h[:3].astype(float)
    if not np.isfinite(p_cam).all() or p_cam[2] <= 1e-6:
        raise RuntimeError(
            "[active] projected target center is behind/invalid in camera frame: "
            f"{p_cam.tolist()}"
        )
    valid = (
        np.isfinite(xyz_img).all(axis=2)
        & (xyz_img[:, :, 2] > args.min_depth)
        & (xyz_img[:, :, 2] < args.max_depth)
    )
    d = np.linalg.norm(xyz_img - p_cam.reshape(1, 1, 3), axis=2)
    radius = float(args.active_project_radius)
    mask = valid & (d <= radius)
    if int(np.count_nonzero(mask)) < int(args.min_target_points):
        radius = max(radius * 1.5, float(args.manual_radius))
        mask = valid & (d <= radius)
    if int(np.count_nonzero(mask)) < int(args.min_target_points):
        raise RuntimeError(
            f"[active] projected target cloud too small: {int(np.count_nonzero(mask))} points "
            f"at p_cam={np.round(p_cam, 4).tolist()} radius={radius:.3f}m"
        )
    h, w = xyz_img.shape[:2]
    u = int(round((p_cam[0] / p_cam[2]) * (w * 0.5) + (w * 0.5)))
    v = int(round((p_cam[1] / p_cam[2]) * (h * 0.5) + (h * 0.5)))
    inst = {
        "label": label,
        "score": 1.0,
        "center": (u, v),
        "mask": mask.astype(np.uint8),
    }
    save_fn(bgr, mask, inst, out_dir)
    print(
        f"[active] projected target-center mask points={int(np.count_nonzero(mask))} "
        f"p_cam={np.round(p_cam, 4).tolist()} radius={radius:.3f}m"
    )
    return mask, inst


def mask_from_clicked_target_point(
    bgr,
    xyz_img,
    args,
    out_dir,
    label="active_clicked",
    *,
    click_point_hook=None,
    pointcloud2_xyz_at_hook=None,
    save_selected_visuals_hook=None,
):
    click_fn = click_point if click_point_hook is None else click_point_hook
    xyz_at_fn = (
        pointcloud2_xyz_at_from_image
        if pointcloud2_xyz_at_hook is None
        else pointcloud2_xyz_at_hook
    )
    save_fn = (
        save_selected_target_visuals
        if save_selected_visuals_hook is None
        else save_selected_visuals_hook
    )
    uv = click_fn(
        bgr, "Click active-view target point for local cloud, then Enter"
    )
    if uv is None:
        raise RuntimeError("active-view target click cancelled")
    u, v = uv
    seed = xyz_at_fn(xyz_img, u, v)
    d = np.linalg.norm(xyz_img - seed.reshape(1, 1, 3), axis=2)
    valid = (
        np.isfinite(xyz_img).all(axis=2)
        & (xyz_img[:, :, 2] > args.min_depth)
        & (xyz_img[:, :, 2] < args.max_depth)
    )
    mask = valid & (d <= float(args.manual_radius))
    if int(np.count_nonzero(mask)) < int(args.min_target_points):
        raise RuntimeError(
            f"clicked active-view target cloud too small: {int(np.count_nonzero(mask))} points"
        )
    inst = {
        "label": label,
        "score": 1.0,
        "center": (u, v),
        "mask": mask.astype(np.uint8),
    }
    save_fn(bgr, mask, inst, out_dir)
    print(
        f"[active] clicked target local mask points={int(np.count_nonzero(mask))} "
        f"pixel=({int(u)}, {int(v)}) radius={float(args.manual_radius):.3f}m"
    )
    return mask, inst


def _filter_capture_cloud(points, colors, voxel, outlier, args, get_o3d_hook):
    if (str(getattr(args, "trajectory_executor", "move_j")) == "mit"
            and not bool(getattr(args, "_planning_worker", False))):
        from .planning_process import isolated_capture_cloud
        return isolated_capture_cloud(points, colors, voxel, outlier, args,
                                      get_o3d_hook=get_o3d_hook)
    from .pointcloud_fusion import filter_capture_cloud
    return filter_capture_cloud(points, colors, voxel, outlier, get_o3d_hook=get_o3d_hook)


def cloud_from_mask(xyz_img, bgr, mask, args, *, get_o3d_hook=None):
    mask = np.asarray(mask).astype(bool)
    valid = (
        mask
        & np.isfinite(xyz_img).all(axis=2)
        & (xyz_img[:, :, 2] > args.min_depth)
        & (xyz_img[:, :, 2] < args.max_depth)
    )
    pts = xyz_img[valid].astype(np.float32)
    cols = bgr[valid][:, ::-1].astype(np.float32) / 255.0
    if pts.shape[0] < int(args.min_target_points):
        raise RuntimeError(f"target cloud too small: {pts.shape[0]} points")

    outlier = ((int(args.outlier_neighbors), float(args.outlier_std_ratio))
               if bool(getattr(args, "target_outlier_filter", True)) else None)
    return _filter_capture_cloud(pts, cols, args.voxel_size, outlier, args, get_o3d_hook)


def scene_cloud_from_xyz(xyz_img, bgr, args, *, get_o3d_hook=None):
    valid = (
        np.isfinite(xyz_img).all(axis=2)
        & (xyz_img[:, :, 2] > args.min_depth)
        & (xyz_img[:, :, 2] < args.max_depth)
    )
    pts = xyz_img[valid].astype(np.float32)
    cols = bgr[valid][:, ::-1].astype(np.float32) / 255.0
    pcd = _filter_capture_cloud(pts, cols, args.scene_voxel_size, None, args, get_o3d_hook)
    if len(pcd.points) > int(args.scene_cloud_cap):
        print(
            f"[cloud][warn] scene has {len(pcd.points)} occupied voxels > "
            f"scene_cloud_cap={int(args.scene_cloud_cap)}; retaining all points for "
            "fail-closed motion safety"
        )
    return pcd


def obstacle_cloud_from_non_target_mask(
    xyz_img,
    bgr,
    target_mask,
    args,
    *,
    get_o3d_hook=None,
):
    target_mask = np.asarray(target_mask).astype(bool)
    valid = (
        (~target_mask)
        & np.isfinite(xyz_img).all(axis=2)
        & (xyz_img[:, :, 2] > args.min_depth)
        & (xyz_img[:, :, 2] < args.max_depth)
    )
    pts = xyz_img[valid].astype(np.float32)
    cols = bgr[valid][:, ::-1].astype(np.float32) / 255.0
    pcd = _filter_capture_cloud(pts, cols, args.scene_voxel_size, None, args, get_o3d_hook)
    if len(pcd.points) > int(args.scene_cloud_cap):
        print(
            f"[cloud][warn] non-target scene has {len(pcd.points)} occupied voxels > "
            f"scene_cloud_cap={int(args.scene_cloud_cap)}; retaining all points for "
            "fail-closed motion safety"
        )
    return pcd


def crop_scene_cloud_near_target(
    scene_cloud,
    target_cloud,
    args,
    label="scene",
    *,
    get_o3d_hook=None,
):
    if not bool(getattr(args, "crop_scene_near_target", True)):
        return scene_cloud
    o3d = (get_o3d if get_o3d_hook is None else get_o3d_hook)()
    scene_pts = np.asarray(scene_cloud.points, dtype=np.float32).reshape(-1, 3)
    if scene_pts.shape[0] == 0:
        return scene_cloud
    target_pts = np.asarray(target_cloud.points, dtype=np.float32).reshape(-1, 3)
    if target_pts.shape[0] == 0:
        return scene_cloud
    center = np.median(target_pts, axis=0).astype(np.float32)
    radius_xy = float(args.scene_target_crop_radius)
    z_below = float(args.scene_target_crop_z_below)
    z_above = float(args.scene_target_crop_z_above)
    rel = scene_pts - center.reshape(1, 3)
    keep = (
        (np.linalg.norm(rel[:, :2], axis=1) <= radius_xy)
        & (rel[:, 2] >= -z_below)
        & (rel[:, 2] <= z_above)
    )
    kept = int(np.count_nonzero(keep))
    if kept < int(args.scene_target_crop_min_points):
        print(
            f"[cloud][warn] {label} target-near crop would keep only "
            f"{kept}/{scene_pts.shape[0]} points; keeping uncropped scene cloud."
        )
        return scene_cloud
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(scene_pts[keep].astype(float))
    if scene_cloud.has_colors():
        colors = np.asarray(scene_cloud.colors, dtype=np.float32).reshape(-1, 3)
        cropped.colors = o3d.utility.Vector3dVector(colors[keep].astype(float))
    if float(args.scene_voxel_size) > 0.0:
        cropped = cropped.voxel_down_sample(float(args.scene_voxel_size))
    print(
        f"[cloud] {label} target-near crop: kept={len(cropped.points)}/{scene_pts.shape[0]} "
        f"radius_xy={radius_xy:.3f}m z=[-{z_below:.3f}, +{z_above:.3f}]m "
        f"center={np.round(center, 4).tolist()}"
    )
    return cropped


def _deduplicate_strict_identity_candidates(
    scored,
    args,
    view_label="active",
    *,
    mask_iou_hook=None,
    mask_nested_overlap_hook=None,
):
    """Collapse proven duplicate 2D+3D detections before ambiguity checks."""
    mask_iou_fn = _mask_iou if mask_iou_hook is None else mask_iou_hook
    nested_overlap_fn = (
        _mask_nested_overlap
        if mask_nested_overlap_hook is None
        else mask_nested_overlap_hook
    )
    mask_iou_threshold = float(
        getattr(args, "active_identity_duplicate_mask_iou", 0.95)
    )
    center_threshold = float(
        getattr(args, "active_identity_duplicate_center_dist", 0.005)
    )
    nested_containment_threshold = float(
        getattr(args, "active_identity_duplicate_nested_min_containment", 0.95)
    )
    nested_max_area_ratio = float(
        getattr(args, "active_identity_duplicate_nested_max_area_ratio", 0.90)
    )
    nested_min_area_ratio = float(
        getattr(args, "active_identity_duplicate_nested_min_area_ratio", 0.50)
    )
    nested_center_threshold = float(
        getattr(args, "active_identity_duplicate_nested_center_dist", 0.015)
    )

    def _confidence(item):
        candidate = item[1]
        if not isinstance(candidate, dict):
            return 0.0
        return float(candidate.get("score", candidate.get("confidence", 0.0)))

    kept = []
    for item in sorted(scored, key=_confidence, reverse=True):
        duplicate_of = None
        for representative in kept:
            iou = mask_iou_fn(item[2], representative[2])
            containment, area_ratio = nested_overlap_fn(
                item[2], representative[2]
            )
            center_distance = float(
                np.linalg.norm(
                    np.asarray(item[3], dtype=float).reshape(3)
                    - np.asarray(representative[3], dtype=float).reshape(3)
                )
            )
            near_identical = (
                iou >= mask_iou_threshold and center_distance <= center_threshold
            )
            nested_duplicate = (
                containment >= nested_containment_threshold
                and area_ratio >= nested_min_area_ratio
                and area_ratio <= nested_max_area_ratio
                and center_distance <= nested_center_threshold
            )
            if near_identical or nested_duplicate:
                duplicate_of = (
                    representative,
                    iou,
                    containment,
                    area_ratio,
                    center_distance,
                    "near_identical" if near_identical else "nested",
                )
                break
        if duplicate_of is None:
            kept.append(item)
            continue
        (
            representative,
            iou,
            containment,
            area_ratio,
            center_distance,
            duplicate_kind,
        ) = duplicate_of
        print(
            f"[active][identity] {view_label} collapsed duplicate same-class detection: "
            f"kind={duplicate_kind} mask_iou={iou:.4f} "
            f"containment={containment:.4f} area_ratio={area_ratio:.4f} "
            f"center_sep={center_distance * 1000.0:.2f}mm "
            f"kept_conf={_confidence(representative):.4f} "
            f"dropped_conf={_confidence(item):.4f}"
        )
    return kept


def camera_self_occlusion_mask(image_shape, args):
    """Fixed camera-frame mask for the two gripper arms visible at image bottom."""
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    normalized_polygons = (
        (
            (0.255, 1.000),
            (0.350, 0.790),
            (0.375, 0.700),
            (0.425, 0.685),
            (0.465, 0.800),
            (0.515, 1.000),
        ),
        (
            (0.700, 1.000),
            (0.745, 0.830),
            (0.790, 0.700),
            (0.845, 0.690),
            (0.900, 0.785),
            (1.000, 0.830),
            (1.000, 1.000),
        ),
    )
    for polygon in normalized_polygons:
        pixels = np.array(
            [
                [
                    int(round(np.clip(x, 0.0, 1.0) * (width - 1))),
                    int(round(np.clip(y, 0.0, 1.0) * (height - 1))),
                ]
                for x, y in polygon
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pixels], 255)
    expand_px = max(int(getattr(args, "camera_self_mask_expand_px", 12)), 0)
    if expand_px > 0:
        kernel_size = 2 * expand_px + 1
        mask = cv2.dilate(
            mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )
    return mask


def apply_camera_self_occlusion_mask(
    xyz_img,
    bgr,
    args,
    out_dir,
    *,
    self_occlusion_mask_hook=None,
):
    if not bool(getattr(args, "camera_self_mask", True)):
        return np.asarray(xyz_img, dtype=np.float32)
    if getattr(args, "camera_self_mask_calibration", None):
        from .camera_self_mask import apply_calibrated_self_mask
        return apply_calibrated_self_mask(xyz_img, bgr, args, out_dir)
    mask_fn = (
        camera_self_occlusion_mask
        if self_occlusion_mask_hook is None
        else self_occlusion_mask_hook
    )
    xyz = np.asarray(xyz_img, dtype=np.float32).copy()
    mask = mask_fn(xyz.shape[:2], args)
    valid_before = np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0.0)
    removed = int(np.count_nonzero(valid_before & (mask > 0)))
    xyz[mask > 0] = np.nan

    # Fixed silhouette polygons can miss one or two quantised edge pixels even
    # after dilation.  Those pixels are especially harmful: they are very near
    # the camera, survive voxelisation as an apparently solid obstacle, and can
    # later make a finger collide with its own RGB-D return.  Remove only tiny
    # near-field components immediately adjacent to the already accepted self
    # mask.  Components are labelled over the whole remaining near-depth image,
    # so a real continuous close obstacle is retained rather than trimming the
    # small part of it that happens to touch the mask boundary.
    edge_rejected = np.zeros(mask.shape, dtype=np.uint8)
    edge_stats = {
        "enabled": bool(
            getattr(args, "camera_self_edge_island_filter", True)
        ),
        "components": 0,
        "removed_components": 0,
        "removed_pixels": 0,
        "removed": [],
    }
    if edge_stats["enabled"]:
        edge_px = max(
            int(getattr(args, "camera_self_edge_island_distance_px", 3)), 1
        )
        max_area = max(
            int(getattr(args, "camera_self_edge_island_max_area", 8)), 1
        )
        max_diameter = max(
            int(getattr(args, "camera_self_edge_island_max_diameter", 6)), 1
        )
        max_depth = max(
            float(getattr(args, "camera_self_edge_island_max_depth", 0.22)),
            0.0,
        )
        min_depth = max(float(getattr(args, "min_depth", 0.05)), 0.0)
        edge_band = (
            cv2.dilate(
                (mask > 0).astype(np.uint8),
                np.ones((2 * edge_px + 1, 2 * edge_px + 1), dtype=np.uint8),
                iterations=1,
            )
            > 0
        ) & (mask == 0)
        z = xyz[:, :, 2]
        near_valid = (
            np.isfinite(xyz).all(axis=2)
            & (z > min_depth)
            & (z <= max_depth)
        )
        component_count, labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(
                near_valid.astype(np.uint8), connectivity=8
            )
        )
        for component_id in range(1, int(component_count)):
            component = labels == component_id
            if not np.any(component & edge_band):
                continue
            edge_stats["components"] += 1
            area = int(component_stats[component_id, cv2.CC_STAT_AREA])
            box_x = int(component_stats[component_id, cv2.CC_STAT_LEFT])
            box_y = int(component_stats[component_id, cv2.CC_STAT_TOP])
            box_w = int(component_stats[component_id, cv2.CC_STAT_WIDTH])
            box_h = int(component_stats[component_id, cv2.CC_STAT_HEIGHT])
            if area > max_area or max(box_w, box_h) > max_diameter:
                continue
            values = z[component]
            row = {
                "bbox_xywh": [box_x, box_y, box_w, box_h],
                "area": area,
                "depth_median_m": float(np.median(values)),
            }
            edge_rejected[component] = 255
            xyz[component] = np.nan
            edge_stats["removed_components"] += 1
            edge_stats["removed_pixels"] += area
            edge_stats["removed"].append(row)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "camera_self_mask.png"), mask)
        overlay = np.asarray(bgr).copy()
        tint = np.zeros_like(overlay)
        tint[:, :, 2] = 255
        selected = mask > 0
        overlay[selected] = cv2.addWeighted(
            overlay[selected], 0.35, tint[selected], 0.65, 0.0
        )
        cv2.imwrite(
            os.path.join(out_dir, "camera_self_mask_overlay.png"), overlay
        )
        cv2.imwrite(
            os.path.join(out_dir, "camera_self_edge_island_mask.png"),
            edge_rejected,
        )
        edge_overlay = np.asarray(bgr).copy()
        edge_selected = edge_rejected > 0
        if np.any(edge_selected):
            tint = np.zeros_like(edge_overlay)
            tint[:, :, 1] = 255
            edge_overlay[edge_selected] = cv2.addWeighted(
                edge_overlay[edge_selected],
                0.30,
                tint[edge_selected],
                0.70,
                0.0,
            )
        cv2.imwrite(
            os.path.join(out_dir, "camera_self_edge_island_overlay.png"),
            edge_overlay,
        )
        with open(
            os.path.join(out_dir, "camera_self_edge_island_report.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(edge_stats, handle, indent=2)
    print(
        f"[camera][self_mask] removed_valid_depth={removed} "
        f"pixels={int(np.count_nonzero(mask))} expand_px="
        f"{int(getattr(args, 'camera_self_mask_expand_px', 12))}"
    )
    print(
        "[camera][self_edge_island] "
        f"enabled={edge_stats['enabled']} "
        f"removed_pixels={edge_stats['removed_pixels']} "
        f"removed_components={edge_stats['removed_components']}"
    )
    return xyz


def camera_near_field_ghost_mask(xyz_img, args, preserve_mask=None):
    """Find opt-in, profile-calibrated near-field depth plateaus."""
    xyz = np.asarray(xyz_img, dtype=np.float32)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"xyz image must be HxWx3, got shape {xyz.shape}")
    height, width = xyz.shape[:2]
    rejected = np.zeros((height, width), dtype=np.uint8)
    stats = {
        "enabled": bool(getattr(args, "camera_near_field_ghost_filter", False)),
        "seed_pixels": 0,
        "candidate_components": 0,
        "removed_components": 0,
        "removed_pixels": 0,
        "retained_seeded_components": 0,
        "removed": [],
        "retained": [],
    }
    if not stats["enabled"]:
        return rejected, stats

    expected_size = np.asarray(
        getattr(args, "camera_near_field_ghost_image_size", (1280, 720)),
        dtype=int,
    ).reshape(-1)
    if expected_size.size != 2 or np.any(expected_size <= 0):
        raise ValueError(
            "--camera-near-field-ghost-image-size requires positive WIDTH HEIGHT"
        )
    expected_width, expected_height = int(expected_size[0]), int(expected_size[1])
    stats["expected_image_size"] = [expected_width, expected_height]
    stats["actual_image_size"] = [int(width), int(height)]
    if (width, height) != (expected_width, expected_height):
        stats["profile_mismatch"] = True
        return rejected, stats

    roi = np.asarray(
        getattr(args, "camera_near_field_ghost_roi", (0.05, 0.28, 0.75, 1.00)),
        dtype=float,
    ).reshape(-1)
    if roi.size != 4 or not np.isfinite(roi).all():
        raise ValueError(
            "--camera-near-field-ghost-roi requires finite xmin xmax ymin ymax"
        )
    x_min, x_max, y_min, y_max = np.clip(roi, 0.0, 1.0)
    if not (x_min < x_max and y_min < y_max):
        raise ValueError(
            "--camera-near-field-ghost-roi must have xmin<xmax and ymin<ymax"
        )
    x0 = int(np.floor(x_min * width))
    x1 = int(np.ceil(x_max * width))
    y0 = int(np.floor(y_min * height))
    y1 = int(np.ceil(y_max * height))
    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[y0:y1, x0:x1] = True

    protected = np.zeros((height, width), dtype=bool)
    if preserve_mask is not None:
        protected = np.asarray(preserve_mask).astype(bool)
        if protected.shape != (height, width):
            raise ValueError(
                f"preserve mask shape {protected.shape} != xyz shape {(height, width)}"
            )
        dilate_px = max(
            int(getattr(args, "camera_near_field_preserve_dilate_px", 8)), 0
        )
        if dilate_px > 0:
            protected = (
                cv2.dilate(
                    protected.astype(np.uint8),
                    np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=np.uint8),
                    iterations=1,
                )
                > 0
            )

    z = xyz[:, :, 2]
    min_depth = max(float(getattr(args, "min_depth", 0.05)), 0.0)
    seed_depth_arg = float(
        getattr(args, "camera_near_field_ghost_seed_depth", 0.175)
    )
    grow_depth_arg = float(
        getattr(args, "camera_near_field_ghost_grow_depth", 0.200)
    )
    if not np.isfinite([min_depth, seed_depth_arg, grow_depth_arg]).all():
        raise ValueError("near-field ghost depth thresholds must be finite")
    seed_depth = max(seed_depth_arg, min_depth)
    grow_depth = max(grow_depth_arg, seed_depth)
    valid = np.isfinite(xyz).all(axis=2) & (z > min_depth)
    seed = valid & roi_mask & ~protected & (z <= seed_depth)
    growable = valid & ~protected & (z <= grow_depth)
    stats.update(
        {
            "roi_normalized": [
                float(x_min),
                float(x_max),
                float(y_min),
                float(y_max),
            ],
            "roi_pixels": [int(x0), int(x1), int(y0), int(y1)],
            "seed_depth_m": float(seed_depth),
            "grow_depth_m": float(grow_depth),
            "seed_pixels": int(np.count_nonzero(seed)),
        }
    )
    if not np.any(seed):
        return rejected, stats

    component_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        growable.astype(np.uint8), connectivity=8
    )
    min_seed_pixels = max(
        int(getattr(args, "camera_near_field_ghost_min_seed_pixels", 8)), 1
    )
    max_area = max(int(getattr(args, "camera_near_field_ghost_max_area", 5000)), 1)
    max_diameter = max(
        int(getattr(args, "camera_near_field_ghost_max_diameter", 120)), 1
    )
    for component_id in range(1, int(component_count)):
        component = labels == component_id
        seed_count = int(np.count_nonzero(component & seed))
        if seed_count < min_seed_pixels:
            continue
        area = int(component_stats[component_id, cv2.CC_STAT_AREA])
        box_x = int(component_stats[component_id, cv2.CC_STAT_LEFT])
        box_y = int(component_stats[component_id, cv2.CC_STAT_TOP])
        box_w = int(component_stats[component_id, cv2.CC_STAT_WIDTH])
        box_h = int(component_stats[component_id, cv2.CC_STAT_HEIGHT])
        values = z[component]
        row = {
            "bbox_xywh": [box_x, box_y, box_w, box_h],
            "area": area,
            "seed_pixels": seed_count,
            "depth_median_m": float(np.median(values)),
            "depth_p05_m": float(np.percentile(values, 5.0)),
            "depth_p95_m": float(np.percentile(values, 95.0)),
        }
        stats["candidate_components"] += 1
        if area > max_area or max(box_w, box_h) > max_diameter:
            stats["retained_seeded_components"] += 1
            stats["retained"].append(row)
            continue
        rejected[component] = 255
        stats["removed_components"] += 1
        stats["removed_pixels"] += area
        stats["removed"].append(row)
    return rejected, stats


def apply_camera_near_field_ghost_filter(
    xyz_img,
    bgr,
    args,
    out_dir=None,
    preserve_mask=None,
    *,
    ghost_mask_hook=None,
):
    """Remove only calibrated, compact near-field ghost components."""
    ghost_mask_fn = (
        camera_near_field_ghost_mask
        if ghost_mask_hook is None
        else ghost_mask_hook
    )
    xyz = np.asarray(xyz_img, dtype=np.float32).copy()
    rejected, stats = ghost_mask_fn(xyz, args, preserve_mask=preserve_mask)
    selected = rejected > 0
    xyz[selected] = np.nan
    print(
        "[camera][near_ghost] "
        f"enabled={stats.get('enabled', False)} "
        f"profile_mismatch={stats.get('profile_mismatch', False)} "
        f"seed_pixels={stats.get('seed_pixels', 0)} "
        f"removed_pixels={stats.get('removed_pixels', 0)} "
        f"removed_components={stats.get('removed_components', 0)} "
        f"retained_seeded_components={stats.get('retained_seeded_components', 0)}"
    )
    for row in stats.get("removed", [])[:5]:
        print(
            "[camera][near_ghost][removed] "
            f"bbox={row['bbox_xywh']} area={row['area']} "
            f"seed={row['seed_pixels']} "
            f"depth_med={row['depth_median_m']:.3f}m"
        )
    for row in stats.get("retained", [])[:5]:
        print(
            "[camera][near_ghost][retain] possible real close obstacle: "
            f"bbox={row['bbox_xywh']} area={row['area']} "
            f"seed={row['seed_pixels']}"
        )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "camera_near_field_ghost_mask.png"), rejected)
        overlay = np.asarray(bgr).copy()
        if overlay.shape[:2] == rejected.shape:
            tint = np.zeros_like(overlay)
            tint[:, :, 0] = 255
            tint[:, :, 2] = 255
            if np.any(selected):
                overlay[selected] = cv2.addWeighted(
                    overlay[selected], 0.30, tint[selected], 0.70, 0.0
                )
            cv2.imwrite(
                os.path.join(out_dir, "camera_near_field_ghost_overlay.png"), overlay
            )
        with open(
            os.path.join(out_dir, "camera_near_field_ghost_report.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(stats, handle, indent=2)
    return xyz


def _default_capture_rgbd(args):
    from tools.real_nero_point_test import capture_rgbd

    return capture_rgbd(args)


def _default_capture_ros_color_pointcloud_frame(args):
    from tools.real_nero_point_test import capture_ros_color_pointcloud_frame

    return capture_ros_color_pointcloud_frame(args)


def capture_rgbd_xyz_frame(
    args,
    *,
    capture_rgbd_hook=None,
    capture_ros_color_pointcloud_frame_hook=None,
    pointcloud2_xyz_image_hook=None,
    depth_to_xyz_image_hook=None,
    apply_camera_self_occlusion_mask_hook=None,
):
    capture_rgbd_fn = (
        _default_capture_rgbd if capture_rgbd_hook is None else capture_rgbd_hook
    )
    capture_points_fn = (
        _default_capture_ros_color_pointcloud_frame
        if capture_ros_color_pointcloud_frame_hook is None
        else capture_ros_color_pointcloud_frame_hook
    )
    pointcloud_image_fn = (
        pointcloud2_xyz_image
        if pointcloud2_xyz_image_hook is None
        else pointcloud2_xyz_image_hook
    )
    depth_xyz_fn = (
        depth_to_xyz_image
        if depth_to_xyz_image_hook is None
        else depth_to_xyz_image_hook
    )
    self_mask_fn = (
        apply_camera_self_occlusion_mask
        if apply_camera_self_occlusion_mask_hook is None
        else apply_camera_self_occlusion_mask_hook
    )
    if args.use_registered_points:
        bgr, cloud_msg, intr, out_dir = capture_points_fn(args)
        xyz_img = pointcloud_image_fn(cloud_msg)
    else:
        bgr, depth_m, intr, out_dir = capture_rgbd_fn(args)
        if depth_m.shape[:2] != bgr.shape[:2]:
            raise RuntimeError(
                f"color size {bgr.shape[1]}x{bgr.shape[0]} != depth size "
                f"{depth_m.shape[1]}x{depth_m.shape[0]}. Depth must be aligned "
                "to color. Configure SDK depth registration/alignment "
                "enabled."
            )
        xyz_img = depth_xyz_fn(depth_m, intr)
    if (xyz_img.shape[1], xyz_img.shape[0]) != (bgr.shape[1], bgr.shape[0]):
        raise RuntimeError(
            f"color size {bgr.shape[1]}x{bgr.shape[0]} != point cloud size "
            f"{xyz_img.shape[1]}x{xyz_img.shape[0]}"
        )
    xyz_img = self_mask_fn(xyz_img, bgr, args, out_dir)
    return bgr, xyz_img, intr, out_dir


def capture_target_cloud(
    args,
    *,
    apply_target_prompt_hook=None,
    capture_rgbd_xyz_frame_hook=None,
    select_target_mask_hook=None,
    apply_near_field_filter_hook=None,
    refine_xyz_inside_visible_mask_hook=None,
    cloud_from_mask_hook=None,
    scene_cloud_from_xyz_hook=None,
    obstacle_cloud_from_non_target_mask_hook=None,
):
    prompt_fn = apply_target_prompt if apply_target_prompt_hook is None else apply_target_prompt_hook
    capture_fn = (
        capture_rgbd_xyz_frame
        if capture_rgbd_xyz_frame_hook is None
        else capture_rgbd_xyz_frame_hook
    )
    select_fn = select_target_mask if select_target_mask_hook is None else select_target_mask_hook
    near_filter_fn = (
        apply_camera_near_field_ghost_filter
        if apply_near_field_filter_hook is None
        else apply_near_field_filter_hook
    )
    refine_fn = (
        refine_xyz_inside_visible_mask
        if refine_xyz_inside_visible_mask_hook is None
        else refine_xyz_inside_visible_mask_hook
    )
    target_cloud_fn = cloud_from_mask if cloud_from_mask_hook is None else cloud_from_mask_hook
    scene_cloud_fn = (
        scene_cloud_from_xyz
        if scene_cloud_from_xyz_hook is None
        else scene_cloud_from_xyz_hook
    )
    obstacle_cloud_fn = (
        obstacle_cloud_from_non_target_mask
        if obstacle_cloud_from_non_target_mask_hook is None
        else obstacle_cloud_from_non_target_mask_hook
    )
    prompt_fn(args)
    bgr, xyz_img, intr, out_dir = capture_fn(args)
    mask, inst = select_fn(bgr, xyz_img, args, out_dir)
    xyz_img = near_filter_fn(
        xyz_img,
        bgr,
        args,
        out_dir=out_dir,
        preserve_mask=mask,
    )
    xyz_img = refine_fn(
        xyz_img,
        mask,
        intr,
        args,
        out_dir=out_dir,
        label="target",
    )
    args._last_capture_bgr = bgr
    args._last_capture_xyz_img = xyz_img
    args._last_capture_intr = intr
    args._last_target_mask = np.asarray(mask).astype(np.uint8)
    args._last_target_inst = inst
    args._last_capture_out_dir = out_dir
    target_cloud = target_cloud_fn(xyz_img, bgr, mask, args)
    scene_cloud = scene_cloud_fn(xyz_img, bgr, args)
    args._last_obstacle_cloud = obstacle_cloud_fn(xyz_img, bgr, mask, args)
    print(
        f"[cloud] target_points={np.asarray(target_cloud.points).shape[0]} "
        f"scene_points={np.asarray(scene_cloud.points).shape[0]} "
        f"non_target_obstacle_points="
        f"{np.asarray(args._last_obstacle_cloud.points).shape[0]}"
    )
    return bgr, target_cloud, scene_cloud, out_dir, inst, intr


def capture_target_cloud_with_retries(
    args,
    *,
    capture_target_cloud_hook=None,
    sleep_hook=None,
):
    capture_fn = (
        capture_target_cloud
        if capture_target_cloud_hook is None
        else capture_target_cloud_hook
    )
    sleep_fn = time.sleep if sleep_hook is None else sleep_hook
    attempts = max(int(args.capture_retries), 1)
    for attempt in range(1, attempts + 1):
        try:
            return capture_fn(args)
        except TimeoutError as exc:
            print(
                f"[capture][warn] RGB-D capture timed out on attempt "
                f"{attempt}/{attempts}: {exc}"
            )
            if attempt < attempts:
                print(
                    f"[capture] retrying after {float(args.capture_retry_delay):.1f}s..."
                )
                sleep_fn(float(args.capture_retry_delay))
                continue
            print(
                "[capture][abort] No ROS RGB-D messages were received. "
                "Robot motion was not commanded."
            )
            print("[capture][check] Verify Orbbec is publishing:")
            print("  Check D435 SDK serial, USB3 connection and color profile.")
            print("  Check aligned depth profile and frame freshness.")
            print(
                "  Intrinsics must come from the active SDK color profile."
            )
            print(
                "[capture][check] If all are 0, restart Orbbec launch or "
                "source ROS before running this script."
            )
            return None
    return None


def recapture_original_target_after_clear(
    args,
    reference_mask,
    reference_inst,
    *,
    capture_target_cloud_with_retries_hook=None,
):
    """Recapture without silently switching to another same-class object."""
    capture_fn = (
        capture_target_cloud_with_retries
        if capture_target_cloud_with_retries_hook is None
        else capture_target_cloud_with_retries_hook
    )
    reference_inst = reference_inst if isinstance(reference_inst, dict) else {}
    reference_label = str(
        reference_inst.get("class_label", reference_inst.get("label", ""))
    ).strip()
    if bool(reference_inst.get("clear_fallback_largest_visible", False)):
        reference_label = str(
            reference_inst.get("requested_target_class", reference_label)
        ).strip()
        reference_mask = None
    if reference_label.lower() in {"", "manual", "unknown", "target"}:
        reference_mask = None

    selection_fields = (
        "target_prompt",
        "target_class",
        "target_index",
        "manual_click_only",
        "select_target",
    )
    saved = {name: getattr(args, name) for name in selection_fields}
    marker = object()
    saved_reference_mask = getattr(args, "_post_clear_reference_mask", marker)
    saved_reference_label = getattr(args, "_post_clear_reference_label", marker)
    try:
        args.target_prompt = False
        args.target_index = None
        args.manual_click_only = False
        args.select_target = False
        if reference_label.lower() not in {"", "manual", "unknown", "target"}:
            args.target_class = reference_label
        else:
            args.target_class = None
        if reference_mask is not None:
            args._post_clear_reference_mask = np.asarray(reference_mask).copy()
            args._post_clear_reference_label = reference_label
            print(
                f"[clear][reidentify] recapturing original {reference_label!r} from "
                "the restored initial camera view"
            )
        else:
            if hasattr(args, "_post_clear_reference_mask"):
                delattr(args, "_post_clear_reference_mask")
            if hasattr(args, "_post_clear_reference_label"):
                delattr(args, "_post_clear_reference_label")
        try:
            return capture_fn(args)
        except TargetIdentityNotVerified as exc:
            print(f"[clear][reidentify][abort] {exc}")
            return None
    finally:
        for name, value in saved.items():
            setattr(args, name, value)
        if saved_reference_mask is marker:
            if hasattr(args, "_post_clear_reference_mask"):
                delattr(args, "_post_clear_reference_mask")
        else:
            args._post_clear_reference_mask = saved_reference_mask
        if saved_reference_label is marker:
            if hasattr(args, "_post_clear_reference_label"):
                delattr(args, "_post_clear_reference_label")
        else:
            args._post_clear_reference_label = saved_reference_label


def _active_view_second_capture_args(args, first_inst):
    args2 = copy.copy(args)
    args2.target_prompt = False
    args2.target_index = None
    args2.select_target = False
    args2.yolo_conf = min(
        float(getattr(args, "yolo_conf", 0.25)),
        max(float(getattr(args, "active_yolo_conf", 0.05)), 0.0),
    )
    label = ""
    if isinstance(first_inst, dict):
        label = str(
            first_inst.get("class_label", first_inst.get("label", ""))
        ).strip()
    if label and label.lower() not in {"manual", "unknown", "target"}:
        args2.target_class = label
        args2.manual_click_only = False
    else:
        args2.target_class = None
        args2.manual_click_only = True
    return args2


def capture_active_projected_target_cloud(
    args,
    first_inst,
    T_base_camera,
    target_center_base,
    view_label,
    *,
    active_view_second_capture_args_hook=None,
    capture_rgbd_xyz_frame_hook=None,
    mask_from_projected_target_center_hook=None,
    mask_from_clicked_target_point_hook=None,
    infer_yolo_instances_hook=None,
    class_matches_hook=None,
    mask_for_instance_hook=None,
    deduplicate_identity_candidates_hook=None,
    save_selected_visuals_hook=None,
    apply_near_field_filter_hook=None,
    refine_xyz_inside_visible_mask_hook=None,
    cloud_from_mask_hook=None,
    scene_cloud_from_xyz_hook=None,
    obstacle_cloud_from_non_target_mask_hook=None,
):
    second_args_fn = (
        _active_view_second_capture_args
        if active_view_second_capture_args_hook is None
        else active_view_second_capture_args_hook
    )
    capture_fn = (
        capture_rgbd_xyz_frame
        if capture_rgbd_xyz_frame_hook is None
        else capture_rgbd_xyz_frame_hook
    )
    projected_fn = (
        mask_from_projected_target_center
        if mask_from_projected_target_center_hook is None
        else mask_from_projected_target_center_hook
    )
    clicked_fn = (
        mask_from_clicked_target_point
        if mask_from_clicked_target_point_hook is None
        else mask_from_clicked_target_point_hook
    )
    infer_fn = (
        infer_yolo_instances
        if infer_yolo_instances_hook is None
        else infer_yolo_instances_hook
    )
    class_matches_fn = _class_matches if class_matches_hook is None else class_matches_hook
    mask_fn = mask_for_instance if mask_for_instance_hook is None else mask_for_instance_hook
    dedup_fn = (
        _deduplicate_strict_identity_candidates
        if deduplicate_identity_candidates_hook is None
        else deduplicate_identity_candidates_hook
    )
    save_fn = (
        save_selected_target_visuals
        if save_selected_visuals_hook is None
        else save_selected_visuals_hook
    )
    near_filter_fn = (
        apply_camera_near_field_ghost_filter
        if apply_near_field_filter_hook is None
        else apply_near_field_filter_hook
    )
    refine_fn = (
        refine_xyz_inside_visible_mask
        if refine_xyz_inside_visible_mask_hook is None
        else refine_xyz_inside_visible_mask_hook
    )
    target_cloud_fn = cloud_from_mask if cloud_from_mask_hook is None else cloud_from_mask_hook
    scene_cloud_fn = (
        scene_cloud_from_xyz
        if scene_cloud_from_xyz_hook is None
        else scene_cloud_from_xyz_hook
    )
    obstacle_cloud_fn = (
        obstacle_cloud_from_non_target_mask
        if obstacle_cloud_from_non_target_mask_hook is None
        else obstacle_cloud_from_non_target_mask_hook
    )

    args2 = second_args_fn(args, first_inst)
    bgr, xyz_img, intr, out_dir = capture_fn(args2)
    args._last_capture_out_dir = out_dir
    mask = None
    inst = None
    target_class_label = ""
    if isinstance(first_inst, dict):
        target_class_label = str(
            first_inst.get("class_label", first_inst.get("label", ""))
        ).strip()

    mode = str(args.active_target_reselect).lower()
    if mode == "projected":
        try:
            mask, inst = projected_fn(
                bgr,
                xyz_img,
                T_base_camera,
                target_center_base,
                args,
                out_dir,
                label=f"{view_label}_projected",
            )
        except Exception as exc:
            print(
                f"[active][warn] {view_label} projected target-center selection "
                f"failed: {type(exc).__name__}: {exc}"
            )
            mask = None
            inst = None

    if mode == "click" and (mask is None or inst is None):
        try:
            mask, inst = clicked_fn(
                bgr,
                xyz_img,
                args,
                out_dir,
                label=f"{view_label}_clicked",
            )
        except Exception as exc:
            print(
                f"[active][warn] {view_label} click selection failed: "
                f"{type(exc).__name__}: {exc}"
            )
            mask = None
            inst = None

    if mode == "yolo" and (mask is None or inst is None):
        try:
            vis_path = os.path.join(out_dir, "yolo_instances.png")
            os.makedirs(out_dir, exist_ok=True)
            print(
                f"[yolo] {view_label} tracked-target confidence="
                f"{float(args2.yolo_conf):.3f}"
            )
            instances, vis = infer_fn(bgr, args2, output_vis_path=vis_path)
            print(
                f"[yolo] {view_label} instances={len(instances)} overlay={vis_path}"
            )
            if len(instances) > 0:
                chosen = None
                label = target_class_label
                if label and label.lower() not in {"manual", "unknown", "target"}:
                    matching = [
                        candidate
                        for candidate in instances
                        if class_matches_fn(candidate.get("label", ""), label)
                    ]
                    scored = []
                    for candidate in matching:
                        try:
                            candidate_mask = mask_fn(candidate, xyz_img, args2)
                            candidate_points = np.asarray(xyz_img, dtype=float)[
                                np.asarray(candidate_mask) > 0
                            ]
                            candidate_points = candidate_points[
                                np.isfinite(candidate_points).all(axis=1)
                                & (candidate_points[:, 2] > 0.03)
                            ]
                            if candidate_points.shape[0] < 3:
                                continue
                            center_camera = np.median(candidate_points, axis=0)
                            center_base = (
                                np.asarray(T_base_camera, dtype=float).reshape(4, 4)
                                @ np.r_[center_camera, 1.0]
                            )[:3]
                            distance = float(
                                np.linalg.norm(
                                    center_base
                                    - np.asarray(
                                        target_center_base, dtype=float
                                    ).reshape(3)
                                )
                            )
                            scored.append(
                                (distance, candidate, candidate_mask, center_base)
                            )
                        except Exception as candidate_exc:
                            print(
                                f"[active][warn] {view_label} cannot score same-class "
                                f"candidate: {type(candidate_exc).__name__}: "
                                f"{candidate_exc}"
                            )
                    scored = dedup_fn(scored, args, view_label=view_label)
                    scored.sort(key=lambda item: item[0])
                    if len(scored) > 0:
                        best_distance, chosen, mask, best_center = scored[0]
                        if bool(
                            getattr(args, "active_strict_target_identity", False)
                        ):
                            max_distance = float(args.active_project_radius)
                            ambiguity_margin = max(
                                float(
                                    getattr(
                                        args,
                                        "active_strict_identity_margin",
                                        0.015,
                                    )
                                ),
                                0.0,
                            )
                            if best_distance > max_distance:
                                raise RuntimeError(
                                    f"nearest {label!r} is "
                                    f"{best_distance * 1000.0:.1f}mm from the "
                                    "tracked clear target (limit "
                                    f"{max_distance * 1000.0:.1f}mm)"
                                )
                            if (
                                len(scored) > 1
                                and scored[1][0] - best_distance < ambiguity_margin
                            ):
                                raise RuntimeError(
                                    f"same-class {label!r} identity is ambiguous: "
                                    "nearest distances "
                                    f"{best_distance * 1000.0:.1f}mm and "
                                    f"{scored[1][0] * 1000.0:.1f}mm"
                                )
                        print(
                            f"[active][identity] {view_label} tracked {label!r} "
                            f"by 3D center: distance={best_distance * 1000.0:.1f}mm "
                            f"center={np.round(best_center, 4).tolist()}"
                        )
                    elif (
                        len(matching) > 0
                        and not bool(
                            getattr(args, "active_strict_target_identity", False)
                        )
                    ):
                        chosen = matching[0]
                if chosen is None:
                    raise RuntimeError(
                        "no YOLO instance matched the first-view target label"
                    )
                if mask is None:
                    mask = mask_fn(chosen, xyz_img, args2)
                inst = chosen
                save_fn(bgr, mask, inst, out_dir)
        except Exception as exc:
            print(
                f"[active][warn] {view_label} YOLO target reselection failed: "
                f"{type(exc).__name__}: {exc}"
            )
    if mask is None or inst is None:
        print(f"[active][warn] {view_label} target reselection found no usable target")
        if bool(getattr(args, "active_strict_target_identity", False)):
            raise RuntimeError(
                f"{view_label} could not re-identify required target class "
                f"{target_class_label!r}; strict clear-target tracking forbids "
                "projected/click fallback"
            )
        fallback = str(args.active_yolo_fallback).lower()
        if fallback == "abort":
            raise RuntimeError(
                f"{view_label} could not re-identify target class "
                f"{target_class_label!r}; refusing to fuse an unverified "
                "projected-radius cloud"
            )
        if fallback == "click":
            print("[active] fallback to mouse click target point in active-view image")
            try:
                mask, inst = clicked_fn(
                    bgr,
                    xyz_img,
                    args,
                    out_dir,
                    label=f"{view_label}_clicked",
                )
            except Exception as exc:
                print(
                    "[active][warn] active-view click fallback failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                mask = None
                inst = None
        if (mask is None or inst is None) and fallback == "projected":
            print("[active] fallback to projected target-center local cloud")
            mask, inst = projected_fn(
                bgr,
                xyz_img,
                T_base_camera,
                target_center_base,
                args,
                out_dir,
                label=f"{view_label}_projected",
            )
        if mask is None or inst is None:
            raise RuntimeError(
                f"{view_label} target fallback did not produce a verified mask"
            )
    xyz_img = near_filter_fn(
        xyz_img,
        bgr,
        args2,
        out_dir=out_dir,
        preserve_mask=mask,
    )
    xyz_img = refine_fn(
        xyz_img,
        mask,
        intr,
        args,
        out_dir=out_dir,
        label=view_label,
    )
    target_cloud = target_cloud_fn(xyz_img, bgr, mask, args)
    scene_cloud = scene_cloud_fn(xyz_img, bgr, args)
    obstacle_cloud = obstacle_cloud_fn(xyz_img, bgr, mask, args)
    print(
        f"[cloud] {view_label} "
        f"target_points={np.asarray(target_cloud.points).shape[0]} "
        f"scene_points={np.asarray(scene_cloud.points).shape[0]} "
        f"non_target_obstacle_points={np.asarray(obstacle_cloud.points).shape[0]}"
    )
    return bgr, target_cloud, scene_cloud, obstacle_cloud, out_dir, inst, intr
