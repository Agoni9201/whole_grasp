"""Amodal occlusion assessment and fail-closed clear-first policy.

Perception, active-view acquisition, and physical grasp execution are injected
explicitly by the runtime facade.  This keeps the policy independently
testable without importing the hardware-owning runtime module.
"""

import copy
import json
import os

import cv2
import numpy as np

from .types import GraspAttemptStatus


__all__ = [
    "configure_amodal_clear_model",
    "_empty_target_occlusion",
    "_target_occlusion_report",
    "infer_target_amodal_occlusion",
    "amodal_candidate_block_metrics",
    "clear_candidate_is_actionable",
    "_corridor_block_score_camera",
    "detect_clear_obstacle_candidates",
    "save_clear_obstacle_overlay",
    "maybe_clear_obstacles_before_active",
]


def _require_hook(name, hook):
    if hook is None:
        raise TypeError(
            f"{name} must be supplied by the perception/execution runtime"
        )
    return hook


def _depth_hook(name, hook):
    if hook is not None:
        return hook
    from . import depth_processing
    return getattr(depth_processing, name)

def configure_amodal_clear_model(args):
    weights = str(getattr(args, "amodal_weights", None) or "").strip()
    if not weights:
        return
    weights = os.path.abspath(os.path.expanduser(weights))
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Amodal weights not found: {weights}")
    args.amodal_weights = weights
    os.environ["YOLO26_AMODAL_WEIGHTS"] = weights
    os.environ["YOLO26_AMODAL_IMGSZ"] = str(
        int(getattr(args, "amodal_imgsz", None) or args.yolo_imgsz)
    )


def _empty_target_occlusion(reason, enabled=False):
    return {
        "enabled": bool(enabled),
        "valid": False,
        "reason": str(reason),
        "label": None,
        "match_score": 0.0,
        "visible_area_px": 0,
        "amodal_area_px": 0,
        "hidden_area_px": 0,
        "visible_ratio": None,
        "occlusion_ratio": None,
        "visible_inside_amodal_ratio": None,
        "hidden_mask": None,
        "amodal_mask": None,
    }


def _target_occlusion_report(evidence):
    keys = (
        "enabled",
        "valid",
        "reason",
        "label",
        "match_score",
        "visible_area_px",
        "amodal_area_px",
        "hidden_area_px",
        "visible_ratio",
        "occlusion_ratio",
        "visible_inside_amodal_ratio",
        "amodal_score",
    )
    return {key: evidence.get(key) for key in keys}


def infer_target_amodal_occlusion(
    bgr,
    target_mask,
    target_inst,
    args,
    out_dir,
    *,
    mask_bbox_hook=None,
):
    mask_bbox_fn = _depth_hook("_mask_bbox", mask_bbox_hook)
    if not str(getattr(args, "amodal_weights", None) or "").strip():
        return _empty_target_occlusion("amodal_weights_not_configured")

    label = str(target_inst.get("label", "")).strip()
    if not label or label.lower() in {"manual", "active_projected", "active_clicked"}:
        return _empty_target_occlusion("target_has_no_model_class", enabled=True)

    try:
        from cv_process_updated_yolo26seg import (
            draw_visible_amodal_occlusion_overlay,
            infer_instances_yolo26_amodal,
            match_visible_amodal_instances,
        )

        amodal_instances, _ = infer_instances_yolo26_amodal(
            bgr,
            target_class=label,
            output_vis_path=os.path.join(out_dir, "clear_amodal_instances.png"),
            conf_thresh=float(args.amodal_conf),
        )
        visible_inst = dict(target_inst)
        visible_inst["label"] = label
        visible_inst["mask"] = (np.asarray(target_mask) > 0).astype(np.uint8)
        if "xyxy" not in visible_inst:
            visible_inst["xyxy"] = mask_bbox_fn(visible_inst["mask"])
        visible_inst.setdefault("score", 1.0)
        matches = match_visible_amodal_instances(
            [visible_inst],
            amodal_instances,
            target_class=label,
        )
        overlay = draw_visible_amodal_occlusion_overlay(bgr, matches)
        overlay_path = os.path.join(out_dir, "clear_visible_amodal_occlusion.png")
        cv2.imwrite(overlay_path, overlay)
    except Exception as exc:
        print(f"[clear][amodal][warn] inference failed: {type(exc).__name__}: {exc}")
        return _empty_target_occlusion(
            f"amodal_inference_failed:{type(exc).__name__}",
            enabled=True,
        )

    valid_matches = [
        rec
        for rec in matches
        if bool(rec.get("occlusion_valid", False))
        and rec.get("visible_index") == 0
    ]
    if not valid_matches:
        print(
            f"[clear][amodal][warn] no valid visible/amodal match for {label}; "
            "using depth geometry only"
        )
        out = _empty_target_occlusion("visible_amodal_match_failed", enabled=True)
        out["label"] = label
        out["amodal_detections"] = int(len(amodal_instances))
        return out

    rec = max(valid_matches, key=lambda item: float(item.get("match_score", 0.0)))
    amodal_inst = rec["amodal_instance"]
    out = {
        "enabled": True,
        "valid": True,
        "reason": "matched",
        "label": label,
        "match_score": float(rec.get("match_score", 0.0)),
        "visible_area_px": int(rec.get("visible_area_px", 0)),
        "amodal_area_px": int(rec.get("amodal_area_px", 0)),
        "hidden_area_px": int(rec.get("hidden_area_px", 0)),
        "visible_ratio": float(rec.get("visible_ratio", 0.0)),
        "occlusion_ratio": float(rec.get("occlusion_ratio", 0.0)),
        "visible_inside_amodal_ratio": float(rec.get("visible_inside_amodal_ratio", 0.0)),
        "amodal_score": float(amodal_inst.get("score", 0.0)),
        "hidden_mask": np.asarray(rec["hidden_mask"], dtype=np.uint8),
        "amodal_mask": np.asarray(amodal_inst["mask"], dtype=np.uint8),
    }
    print(
        f"[clear][amodal] target={label} occ={out['occlusion_ratio']:.3f} "
        f"visible={out['visible_ratio']:.3f} match={out['match_score']:.3f} "
        f"hidden_px={out['hidden_area_px']}"
    )
    return out


def amodal_candidate_block_metrics(candidate_mask, target_occlusion, args):
    out = {
        "amodal_hidden_overlap_pixels": 0,
        "amodal_hidden_coverage_ratio": 0.0,
        "amodal_candidate_in_hidden_ratio": 0.0,
        "amodal_target_coverage_ratio": 0.0,
        "amodal_candidate_inside_target_ratio": 0.0,
        "amodal_block_score": 0.0,
        "amodal_actionable": False,
    }
    if not bool(target_occlusion.get("valid", False)):
        return out

    hidden = np.asarray(target_occlusion["hidden_mask"]) > 0
    amodal = np.asarray(target_occlusion["amodal_mask"]) > 0
    candidate = np.asarray(candidate_mask) > 0
    if candidate.shape != hidden.shape:
        candidate = cv2.resize(
            candidate.astype(np.uint8),
            (hidden.shape[1], hidden.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    candidate_area = int(np.count_nonzero(candidate))
    hidden_area = int(np.count_nonzero(hidden))
    amodal_area = int(np.count_nonzero(amodal))
    hidden_overlap = int(np.count_nonzero(candidate & hidden))
    target_overlap = int(np.count_nonzero(candidate & amodal))
    hidden_coverage = float(hidden_overlap / max(hidden_area, 1))
    candidate_in_hidden = float(hidden_overlap / max(candidate_area, 1))
    target_coverage = float(target_overlap / max(amodal_area, 1))
    candidate_inside_target = float(target_overlap / max(candidate_area, 1))

    min_overlap = max(float(args.clear_amodal_min_hidden_overlap), 1e-6)
    occ_threshold = max(float(args.clear_amodal_occlusion_threshold), 1e-6)
    overlap_strength = float(
        np.clip(max(hidden_coverage, candidate_in_hidden) / min_overlap, 0.0, 1.0)
    )
    occ_ratio = float(target_occlusion.get("occlusion_ratio", 0.0))
    occ_strength = float(np.clip(occ_ratio / occ_threshold, 0.0, 1.0))
    block_score = 0.0
    if hidden_overlap > 0:
        block_score = float(0.70 * overlap_strength + 0.30 * occ_strength)

    out.update({
        "amodal_hidden_overlap_pixels": hidden_overlap,
        "amodal_hidden_coverage_ratio": hidden_coverage,
        "amodal_candidate_in_hidden_ratio": candidate_in_hidden,
        "amodal_target_coverage_ratio": target_coverage,
        "amodal_candidate_inside_target_ratio": candidate_inside_target,
        "amodal_block_score": block_score,
        "amodal_actionable": bool(
            occ_ratio >= float(args.clear_amodal_occlusion_threshold)
            and (
                hidden_coverage >= float(args.clear_amodal_min_hidden_overlap)
                or candidate_in_hidden >= float(args.clear_amodal_min_hidden_overlap)
            )
        ),
    })
    return out


def clear_candidate_is_actionable(row, args):
    if bool(row.get("largest_visible_fallback", False)):
        return True
    if not bool(row.get("is_front", False)):
        return False
    threshold = float(args.clear_score_threshold)
    geometry_trigger = float(row.get("geometry_score", row.get("clear_score", 0.0))) >= threshold
    amodal_trigger = (
        bool(row.get("amodal_actionable", False))
        and float(row.get("clear_score", 0.0)) >= threshold
    )
    return bool(geometry_trigger or amodal_trigger)




def _corridor_block_score_camera(xyz_img, obj_mask, target_center_cam, radius, depth_margin):
    pts = xyz_img[np.asarray(obj_mask) > 0].astype(np.float32)
    pts = pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0.0)]
    if pts.shape[0] == 0:
        return 0.0, 0, float("inf")
    target = np.asarray(target_center_cam, dtype=np.float32).reshape(3)
    dist = float(np.linalg.norm(target))
    if dist < 1e-6:
        return 0.0, 0, float("inf")
    direction = target / dist
    axial = pts @ direction
    keep_axial = (axial > 0.02) & (axial < max(dist - float(depth_margin), 0.02))
    perp = np.linalg.norm(pts - axial[:, None] * direction.reshape(1, 3), axis=1)
    keep = keep_axial & (perp <= float(radius))
    count = int(np.count_nonzero(keep))
    if count == 0:
        return 0.0, 0, float("inf")
    ratio = float(count / max(pts.shape[0], 1))
    nearest = float(np.min(axial[keep]))
    score = float(np.clip(0.65 * min(ratio / 0.25, 1.0) + 0.35 * min(count / 250.0, 1.0), 0.0, 1.0))
    return score, count, nearest


def detect_clear_obstacle_candidates(
    bgr,
    xyz_img,
    target_mask,
    target_inst,
    args,
    out_dir,
    *,
    infer_yolo_instances_hook=None,
    infer_target_amodal_occlusion_hook=None,
    median_depth_for_mask_hook=None,
    mask_bbox_hook=None,
    expand_bbox_xyxy_hook=None,
    bbox_mask_hook=None,
    mask_iou_hook=None,
    mask_for_instance_hook=None,
    mask_center_hook=None,
    corridor_block_score_camera_hook=None,
    amodal_candidate_block_metrics_hook=None,
    clear_candidate_is_actionable_hook=None,
    largest_visible_clear_instance_hook=None,
    target_occlusion_report_hook=None,
    save_clear_obstacle_overlay_hook=None,
):
    infer_yolo_instances_fn = _require_hook(
        "infer_yolo_instances", infer_yolo_instances_hook
    )
    infer_amodal_fn = (
        infer_target_amodal_occlusion
        if infer_target_amodal_occlusion_hook is None
        else infer_target_amodal_occlusion_hook
    )
    median_depth_fn = _depth_hook(
        "_median_depth_for_mask", median_depth_for_mask_hook
    )
    mask_bbox_fn = _depth_hook("_mask_bbox", mask_bbox_hook)
    expand_bbox_fn = _depth_hook("_expand_bbox_xyxy", expand_bbox_xyxy_hook)
    bbox_mask_fn = _depth_hook("_bbox_mask", bbox_mask_hook)
    mask_iou_fn = _depth_hook("_mask_iou", mask_iou_hook)
    mask_for_instance_fn = _require_hook(
        "mask_for_instance", mask_for_instance_hook
    )
    mask_center_fn = _depth_hook("_mask_center", mask_center_hook)
    corridor_fn = (
        _corridor_block_score_camera
        if corridor_block_score_camera_hook is None
        else corridor_block_score_camera_hook
    )
    amodal_metrics_fn = (
        amodal_candidate_block_metrics
        if amodal_candidate_block_metrics_hook is None
        else amodal_candidate_block_metrics_hook
    )
    actionable_fn = (
        clear_candidate_is_actionable
        if clear_candidate_is_actionable_hook is None
        else clear_candidate_is_actionable_hook
    )
    largest_visible_fn = _require_hook(
        "largest_visible_clear_instance", largest_visible_clear_instance_hook
    )
    occlusion_report_fn = (
        _target_occlusion_report
        if target_occlusion_report_hook is None
        else target_occlusion_report_hook
    )
    save_overlay_fn = (
        save_clear_obstacle_overlay
        if save_clear_obstacle_overlay_hook is None
        else save_clear_obstacle_overlay_hook
    )

    h, w = bgr.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    all_args = copy.copy(args)
    all_args.target_class = None
    all_args.manual_click_only = False
    vis_path = os.path.join(out_dir, "clear_yolo_instances.png")
    try:
        instances, _ = infer_yolo_instances_fn(bgr, all_args, output_vis_path=vis_path)
    except Exception as exc:
        print(f"[clear][warn] YOLO instance detection failed: {type(exc).__name__}: {exc}")
        return []
    target_mask = np.asarray(target_mask) > 0
    target_occlusion = infer_amodal_fn(
        bgr,
        target_mask,
        target_inst,
        args,
        out_dir,
    )
    target_depth = median_depth_fn(xyz_img, target_mask, args)
    target_xyxy = np.asarray(target_inst.get("xyxy", mask_bbox_fn(target_mask)), dtype=np.float32).reshape(4)
    if not np.isfinite(target_xyxy).all() or np.allclose(target_xyxy, 0.0):
        target_xyxy = mask_bbox_fn(target_mask)
    target_region_xyxy = expand_bbox_fn(target_xyxy, w, h, args.clear_target_bbox_expand)
    target_region = bbox_mask_fn((h, w), target_region_xyxy)
    target_pts = xyz_img[target_mask & np.isfinite(xyz_img).all(axis=2)]
    if target_pts.shape[0] == 0:
        target_center_cam = np.array([0.0, 0.0, target_depth], dtype=np.float32)
    else:
        target_center_cam = np.median(target_pts, axis=0).astype(np.float32)
    tc = target_inst.get("center", None)
    if tc is None:
        ys, xs = np.nonzero(target_mask)
        tc = (float(np.median(xs)), float(np.median(ys))) if xs.size > 0 else (w * 0.5, h * 0.5)
    target_center_uv = np.asarray(tc, dtype=np.float32).reshape(2)

    rows = []
    for idx, inst in enumerate(instances):
        raw_mask = np.asarray(inst["mask"]) > 0
        iou = mask_iou_fn(raw_mask, target_mask)
        if iou >= float(args.clear_skip_target_iou):
            continue
        try:
            obj_mask = mask_for_instance_fn(inst, xyz_img, args)
        except Exception:
            obj_mask = raw_mask
        obj_mask = np.asarray(obj_mask) > 0
        if int(np.count_nonzero(obj_mask)) < int(args.clear_min_points):
            continue
        obj_depth = median_depth_fn(xyz_img, obj_mask, args)
        if not np.isfinite(obj_depth) or not np.isfinite(target_depth):
            continue
        depth_delta = float(target_depth - obj_depth)
        front_depth_score = float(np.clip(depth_delta / max(float(args.clear_front_depth_full_score), 1e-6), 0.0, 1.0))
        overlap_pixels = int(np.count_nonzero(obj_mask & target_region))
        overlap_ratio = float(overlap_pixels / max(int(np.count_nonzero(target_region)), 1))
        center = inst.get("center", None)
        if center is None:
            center = mask_center_fn(obj_mask)
        if center is None:
            center_block_score = 0.0
        else:
            center_uv = np.asarray(center, dtype=np.float32).reshape(2)
            bbox_w = max(float(target_region_xyxy[2] - target_region_xyxy[0]), 1.0)
            bbox_h = max(float(target_region_xyxy[3] - target_region_xyxy[1]), 1.0)
            norm_dist = float(np.linalg.norm((center_uv - target_center_uv) / np.array([bbox_w, bbox_h], dtype=np.float32)))
            center_block_score = float(np.clip(1.0 - norm_dist, 0.0, 1.0))
        corridor_score, corridor_count, corridor_nearest = corridor_fn(
            xyz_img,
            obj_mask,
            target_center_cam,
            args.clear_corridor_radius,
            args.clear_depth_margin,
        )
        geometry_score = (
            float(args.clear_overlap_weight) * overlap_ratio
            + float(args.clear_front_weight) * front_depth_score
            + float(args.clear_center_weight) * center_block_score
            + float(args.clear_corridor_weight) * corridor_score
        )
        amodal_metrics = amodal_metrics_fn(obj_mask, target_occlusion, args)
        if bool(target_occlusion.get("valid", False)):
            amodal_weight = float(np.clip(args.clear_amodal_weight, 0.0, 1.0))
            fused_score = (
                (1.0 - amodal_weight) * geometry_score
                + amodal_weight * float(amodal_metrics["amodal_block_score"])
            )
            clear_score = max(float(geometry_score), float(fused_score))
        else:
            fused_score = float(geometry_score)
            clear_score = float(geometry_score)
        is_front = depth_delta >= float(args.clear_depth_margin)
        row = {
            "idx": int(idx),
            "instance_index": int(idx),
            "label": str(inst.get("label", "unknown")),
            "score": float(inst.get("score", 0.0)),
            "clear_score": float(clear_score),
            "geometry_score": float(geometry_score),
            "fused_score": float(fused_score),
            "target_depth": float(target_depth),
            "object_depth": float(obj_depth),
            "depth_delta": float(depth_delta),
            "is_front": bool(is_front),
            "overlap_ratio": float(overlap_ratio),
            "overlap_pixels": int(overlap_pixels),
            "front_depth_score": float(front_depth_score),
            "center_block_score": float(center_block_score),
            "corridor_score": float(corridor_score),
            "corridor_count": int(corridor_count),
            "corridor_nearest": float(corridor_nearest),
            "iou_with_target": float(iou),
            "mask": obj_mask.astype(np.uint8),
            "inst": inst,
        }
        row.update(amodal_metrics)
        rows.append(row)

    # A high amodal-occlusion ratio is not, by itself, a reason to replace a
    # geometrically supported occluder with the largest object in the image.
    # The largest-visible policy is only a fallback for the case where no
    # normal candidate has actionable front/overlap/amodal evidence.  Without
    # this gate, a large but unrelated object can outrank the actual blocker
    # and send the clear workflow toward an unnecessarily crowded grasp.
    evidence_actionable = [
        row for row in rows if actionable_fn(row, args)
    ]
    high_occlusion_fallback = False
    if (
        bool(args.clear_largest_visible_fallback)
        and bool(target_occlusion.get("valid", False))
        and float(target_occlusion.get("occlusion_ratio", 0.0))
        >= float(args.clear_largest_fallback_occlusion)
        and not evidence_actionable
    ):
        largest = largest_visible_fn(
            bgr,
            xyz_img,
            args,
            out_dir,
            instances=instances,
            exclude_mask=target_mask,
        )
        if largest is not None:
            fallback_row = next(
                (row for row in rows if int(row.get("instance_index", -1)) == int(largest["source_index"])),
                None,
            )
            if fallback_row is not None:
                fallback_row["largest_visible_fallback"] = True
                fallback_row["fallback_reason"] = "target_high_occlusion"
                fallback_row["visible_area_px"] = int(largest["visible_area_px"])
                fallback_row["clear_score_before_fallback"] = float(fallback_row["clear_score"])
                fallback_row["clear_score"] = max(
                    float(fallback_row["clear_score"]),
                    float(args.clear_score_threshold),
                )
                high_occlusion_fallback = True
                print(
                    f"[clear][largest] target occlusion={float(target_occlusion['occlusion_ratio']):.3f} "
                    f">= {float(args.clear_largest_fallback_occlusion):.3f}; "
                    f"clear largest visible instance {fallback_row['label']!r} first"
                )
    elif (
        bool(args.clear_largest_visible_fallback)
        and bool(target_occlusion.get("valid", False))
        and float(target_occlusion.get("occlusion_ratio", 0.0))
        >= float(args.clear_largest_fallback_occlusion)
        and evidence_actionable
    ):
        print(
            "[clear][largest] high-occlusion fallback skipped because an "
            f"evidence-backed occluder is actionable: "
            f"{evidence_actionable[0]['label']!r} "
            f"score={float(evidence_actionable[0]['clear_score']):.3f}"
        )

    rows.sort(
        key=lambda r: (bool(r.get("largest_visible_fallback", False)), float(r["clear_score"])),
        reverse=True,
    )
    actionable = [row for row in rows if actionable_fn(row, args)]
    if high_occlusion_fallback and actionable:
        recommended_action = "clear_largest_visible"
    elif actionable:
        recommended_action = "clear_first"
    elif (
        bool(target_occlusion.get("valid", False))
        and float(target_occlusion.get("occlusion_ratio", 0.0))
        >= float(args.clear_amodal_occlusion_threshold)
    ):
        recommended_action = "active_view_no_actionable_occluder"
    else:
        recommended_action = "continue_target_grasp"
    report_rows = []
    for row in rows:
        serial = {k: v for k, v in row.items() if k not in {"mask", "inst"}}
        report_rows.append(serial)
    report = {
        "target_label": str(target_inst.get("label", "unknown")),
        "target_depth": float(target_depth),
        "target_xyxy": target_xyxy.astype(float).tolist(),
        "target_region_xyxy": target_region_xyxy.astype(float).tolist(),
        "target_occlusion": occlusion_report_fn(target_occlusion),
        "recommended_action": recommended_action,
        "recommended_candidate_idx": None if not actionable else int(actionable[0]["idx"]),
        "candidates": report_rows,
    }
    # Preserve the amodal decision for the caller.  In particular, high target
    # occlusion with no identifiable clear object must force another view; it
    # must not silently become "no clearing needed".
    args._last_clear_recommended_action = recommended_action
    with open(os.path.join(out_dir, "clear_obstacle_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    save_overlay_fn(
        bgr,
        target_mask,
        target_region,
        rows,
        out_dir,
        args,
        target_occlusion=target_occlusion,
        recommended_action=recommended_action,
    )
    if len(rows) == 0:
        print("[clear] no potential front occluder candidates")
    else:
        best = rows[0]
        print(
            f"[clear] best_candidate idx={best['idx']} label={best['label']} "
            f"clear_score={best['clear_score']:.3f} geometry={best['geometry_score']:.3f} "
            f"amodal={best['amodal_block_score']:.3f} front={best['is_front']} "
            f"depth_delta={best['depth_delta']:.3f}m overlap={best['overlap_ratio']:.3f} "
            f"corridor={best['corridor_score']:.3f}/{best['corridor_count']}"
        )
    print(f"[clear] recommended_action={recommended_action}")
    return rows


def save_clear_obstacle_overlay(
    bgr,
    target_mask,
    target_region,
    rows,
    out_dir,
    args,
    target_occlusion=None,
    recommended_action=None,
    *,
    mask_bbox_hook=None,
):
    mask_bbox_fn = _depth_hook("_mask_bbox", mask_bbox_hook)
    overlay = bgr.copy()
    target_mask = np.asarray(target_mask) > 0
    target_region = np.asarray(target_region) > 0
    overlay[target_region] = (0.82 * overlay[target_region] + 0.18 * np.asarray([255, 255, 0], dtype=np.uint8)).astype(np.uint8)
    overlay[target_mask] = (0.45 * overlay[target_mask] + 0.55 * np.asarray([0, 255, 0], dtype=np.uint8)).astype(np.uint8)
    if target_occlusion and bool(target_occlusion.get("valid", False)):
        amodal_mask = np.asarray(target_occlusion["amodal_mask"]) > 0
        hidden_mask = np.asarray(target_occlusion["hidden_mask"]) > 0
        overlay[hidden_mask] = (
            0.45 * overlay[hidden_mask]
            + 0.55 * np.asarray([0, 0, 255], dtype=np.uint8)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            amodal_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 220, 255), 2)
    for rank, row in enumerate(rows[:8]):
        mask = np.asarray(row["mask"]) > 0
        color = np.asarray([0, 0, 255] if rank == 0 else [255, 80, 0], dtype=np.uint8)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        inst = row.get("inst", {})
        xyxy = np.asarray(inst.get("xyxy", mask_bbox_fn(mask)), dtype=float).astype(int).tolist()
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(x) for x in color.tolist()), 2)
        cv2.putText(
            overlay,
            (
                f"{rank}:{row['label']} clear={row['clear_score']:.2f} "
                f"g={row.get('geometry_score', 0.0):.2f} "
                f"a={row.get('amodal_block_score', 0.0):.2f}"
            ),
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tuple(int(x) for x in color.tolist()),
            2,
            cv2.LINE_AA,
        )
    if target_occlusion and bool(target_occlusion.get("valid", False)):
        status = (
            f"target occ={float(target_occlusion['occlusion_ratio']) * 100.0:.1f}% "
            f"action={recommended_action or 'unknown'}"
        )
    else:
        status = f"target occ=n/a action={recommended_action or 'geometry_only'}"
    cv2.rectangle(overlay, (8, 8), (min(overlay.shape[1] - 8, 720), 42), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        status,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path = os.path.join(out_dir, "clear_obstacle_candidates.png")
    cv2.imwrite(path, overlay)
    print(f"[clear] saved obstacle candidates overlay={path}")

def maybe_clear_obstacles_before_active(
    arm,
    args,
    captured,
    *,
    detect_clear_obstacle_candidates_hook=None,
    clear_candidate_is_actionable_hook=None,
    refine_xyz_inside_visible_mask_hook=None,
    cloud_from_mask_hook=None,
    scene_cloud_from_xyz_hook=None,
    obstacle_cloud_from_non_target_mask_hook=None,
    save_selected_target_visuals_hook=None,
    T_base_camera_from_flange_hook=None,
    prepare_motion_safety_scene_hook=None,
    plan_and_maybe_execute_grasp_hook=None,
    record_workflow_motion_hook=None,
    run_optional_active_view_hook=None,
    recapture_original_target_after_clear_hook=None,
):
    if not bool(args.enable_clear_obstacle):
        return captured

    detect_candidates_fn = (
        detect_clear_obstacle_candidates
        if detect_clear_obstacle_candidates_hook is None
        else detect_clear_obstacle_candidates_hook
    )
    actionable_fn = (
        clear_candidate_is_actionable
        if clear_candidate_is_actionable_hook is None
        else clear_candidate_is_actionable_hook
    )
    refine_xyz_fn = _require_hook(
        "refine_xyz_inside_visible_mask", refine_xyz_inside_visible_mask_hook
    )
    cloud_from_mask_fn = _require_hook("cloud_from_mask", cloud_from_mask_hook)
    scene_cloud_from_xyz_fn = _require_hook(
        "scene_cloud_from_xyz", scene_cloud_from_xyz_hook
    )
    obstacle_cloud_fn = _require_hook(
        "obstacle_cloud_from_non_target_mask",
        obstacle_cloud_from_non_target_mask_hook,
    )
    save_target_visuals_fn = _require_hook(
        "save_selected_target_visuals", save_selected_target_visuals_hook
    )
    T_base_camera_fn = _require_hook(
        "T_base_camera_from_flange", T_base_camera_from_flange_hook
    )
    prepare_motion_scene_fn = prepare_motion_safety_scene_hook
    plan_grasp_fn = _require_hook(
        "plan_and_maybe_execute_grasp", plan_and_maybe_execute_grasp_hook
    )
    record_motion_fn = _require_hook(
        "record_workflow_motion", record_workflow_motion_hook
    )
    active_view_fn = _require_hook(
        "run_optional_active_view", run_optional_active_view_hook
    )
    recapture_fn = _require_hook(
        "recapture_original_target_after_clear",
        recapture_original_target_after_clear_hook,
    )

    args._force_active_multiview = False
    current = captured
    original_target_reference_mask = getattr(args, "_last_target_mask", None)
    if original_target_reference_mask is not None:
        original_target_reference_mask = np.asarray(
            original_target_reference_mask
        ).copy()
    original_target_reference_inst = (
        dict(captured[4]) if isinstance(captured[4], dict) else {}
    )
    clears_done = 0
    max_clear_rounds = int(max(args.clear_max_rounds, 1))
    empty_grasp_retries_used = 0
    max_empty_grasp_retries = max(
        int(getattr(args, "clear_empty_grasp_retries", 1)),
        0,
    )
    while True:
        bgr, target_cloud, scene_cloud, out_dir, inst, intr = current
        xyz_img = getattr(args, "_last_capture_xyz_img", None)
        target_mask = getattr(args, "_last_target_mask", None)
        if xyz_img is None or target_mask is None:
            print("[clear][abort] missing last RGB-D target mask; cannot prove that clearing is unnecessary")
            return None
        print(
            f"[clear] assessment after {clears_done} successful clear(s), "
            f"limit={max_clear_rounds}: visible/amodal + depth occlusion check"
        )
        missing_target_fallback = bool(inst.get("clear_fallback_largest_visible", False))
        if missing_target_fallback:
            area = int(inst.get("visible_area_px", np.count_nonzero(target_mask)))
            chosen = {
                "idx": -1,
                "label": str(inst.get("label", "unknown")),
                "score": float(inst.get("score", 0.0)),
                "clear_score": 1.0,
                "geometry_score": 0.0,
                "amodal_block_score": 0.0,
                "is_front": True,
                "largest_visible_fallback": True,
                "fallback_reason": str(inst.get("clear_fallback_reason", "requested_target_not_detected")),
                "visible_area_px": area,
                "mask": np.asarray(target_mask, dtype=np.uint8),
                "inst": inst,
            }
            rows = [chosen]
            actionable = rows
            print(
                f"[clear][largest] original target {inst.get('requested_target_class', 'unknown')!r} is missing; "
                f"clear selected largest visible {chosen['label']!r}"
            )
        else:
            args._last_clear_recommended_action = None
            rows = detect_candidates_fn(bgr, xyz_img, target_mask, inst, args, out_dir)
            actionable = [r for r in rows if actionable_fn(r, args)]
        if len(actionable) == 0:
            if (
                getattr(args, "_last_clear_recommended_action", None)
                == "active_view_no_actionable_occluder"
            ):
                args._force_active_multiview = True
                print(
                    "[clear] amodal target occlusion is high but no clear object can be "
                    "identified; force active multi-view before any target grasp"
                )
                return current
            print(
                f"[clear] no actionable obstacle exceeds threshold={float(args.clear_score_threshold):.3f}; "
                "continue to active view / target grasp"
            )
            return current
        if clears_done >= max_clear_rounds:
            chosen = actionable[0]
            print(
                f"[clear][abort] clear limit reached but obstacle {chosen.get('label', 'unknown')!r} "
                "is still actionable; target grasp is forbidden"
            )
            return None
        chosen = actionable[0]
        round_number = clears_done + 1
        retry_dir_suffix = (
            ""
            if empty_grasp_retries_used == 0
            else f"_retry_{empty_grasp_retries_used}"
        )
        retry_label_suffix = (
            ""
            if empty_grasp_retries_used == 0
            else f"_retry{empty_grasp_retries_used}"
        )
        print(
            f"[clear] CLEAR-FIRST recommended: idx={chosen['idx']} label={chosen['label']} "
            f"clear_score={chosen['clear_score']:.3f} geometry={chosen['geometry_score']:.3f} "
            f"amodal={chosen['amodal_block_score']:.3f}"
        )
        clear_inst = dict(chosen["inst"])
        clear_inst["label"] = f"clear_{chosen['label']}"
        clear_inst["class_label"] = str(chosen["label"])
        clear_inst["role"] = "clear_obstacle"
        clear_inst["score"] = float(chosen.get("clear_score", 0.0))
        clear_inst["mask"] = np.asarray(chosen["mask"], dtype=np.uint8)
        clear_inst["clear_metrics"] = {k: v for k, v in chosen.items() if k not in {"mask", "inst"}}
        clear_dir = os.path.join(
            out_dir,
            f"clear_round_{round_number}{retry_dir_suffix}",
        )
        os.makedirs(clear_dir, exist_ok=True)
        clear_xyz_img = refine_xyz_fn(
            xyz_img,
            chosen["mask"],
            intr,
            args,
            out_dir=clear_dir,
            label=f"clear_{round_number}",
        )
        clear_cloud = cloud_from_mask_fn(clear_xyz_img, bgr, chosen["mask"], args)
        clear_scene_cloud = scene_cloud_from_xyz_fn(clear_xyz_img, bgr, args)
        clear_obstacle_cloud = obstacle_cloud_fn(
            clear_xyz_img,
            bgr,
            chosen["mask"],
            args,
        )
        save_target_visuals_fn(bgr, chosen["mask"], clear_inst, clear_dir)
        T_base_camera = T_base_camera_fn(arm, args)
        clear_plan_args = copy.copy(args)
        # A validation-only pickup deliberately stops with the obstacle held
        # just above its support.  That is useful for final-target experiments
        # but it is not a completed clear action and must never advance the
        # clear state machine to recapture/re-identify the original target.
        clear_plan_args.grasp_validation_mode = False
        clear_plan_args.yes = bool(args.yes or args.yes_clear)
        clear_single_plan_kwargs = {}
        if (
            callable(prepare_motion_scene_fn)
            and bool(
                getattr(
                    clear_plan_args,
                    "motion_environment_map_enabled",
                    False,
                )
            )
        ):
            # Freeze one map generation for this capture/clear round.  Every
            # GraspNet candidate and both jaw orientations then consume the
            # exact same occupancy instead of rebuilding it independently.
            try:
                clear_single_scene = prepare_motion_scene_fn(
                    clear_cloud,
                    clear_scene_cloud,
                    T_base_camera,
                    clear_plan_args,
                    map_id=(
                        f"clear{round_number}{retry_label_suffix}_single_capture"
                    ),
                    motion_environment_cloud=clear_obstacle_cloud,
                    current_joints=np.asarray(
                        arm.get_joint_angles(), dtype=float
                    ).reshape(7),
                )
                if clear_single_scene.environment_map is not None:
                    clear_single_plan_kwargs["motion_environment_map"] = (
                        clear_single_scene.environment_map
                    )
            except Exception as exc:
                # Do not turn a map preparation failure into an uncaught clear
                # policy error.  The normal grasp planner immediately repeats
                # the same fail-closed scene construction and reports a
                # PLANNING_FAILED result through the existing fallback state
                # machine.
                print(
                    "[clear][motion][map][warn] shared single-view map "
                    f"unavailable; planner will rebuild/fail closed: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                generation_owner = getattr(
                    clear_plan_args,
                    "_motion_environment_generation_owner",
                    None,
                )
                if (
                    generation_owner is None
                    and "motion_environment_map" in clear_single_plan_kwargs
                ):
                    generation_owner = {
                        "value": int(
                            clear_single_plan_kwargs[
                                "motion_environment_map"
                            ].generation
                        )
                    }
                    clear_plan_args._motion_environment_generation_owner = (
                        generation_owner
                    )
                if generation_owner is not None:
                    args._motion_environment_generation_owner = (
                        generation_owner
                    )
                    args._motion_environment_generation = int(
                        generation_owner.get("value", 0)
                    )
        clear_result = plan_grasp_fn(
            arm,
            clear_plan_args,
            clear_cloud,
            clear_scene_cloud,
            T_base_camera,
            bgr,
            intr,
            clear_dir,
            clear_inst,
            plan_label=f"clear{round_number}{retry_label_suffix}_single",
            place_xyz_override=args.clear_place_xyz,
            confirmation_word="CLEAR",
            motion_environment_cloud=clear_obstacle_cloud,
            **clear_single_plan_kwargs,
        )
        record_motion_fn(args, clear_result.motion_sent)
        if (
            clear_result.status is GraspAttemptStatus.PLANNING_FAILED
            and not clear_result.motion_sent
        ):
            if not args.execute:
                print(
                    "[clear][abort] single-view clear planning failed and --execute is disabled, "
                    "so active multi-view motion cannot be acquired"
                )
                return None
            print(
                f"[clear] single-view planning failed for {clear_inst['label']}; "
                "force active multi-view for the same clear target"
            )
            clear_view_args = copy.copy(args)
            clear_view_args.active_view = True
            clear_view_args.active_multiview_gate = False
            clear_view_args.active_target_reselect = "yolo"
            clear_view_args.active_strict_target_identity = True
            clear_view_args.yes = bool(args.yes)
            clear_view_args._last_obstacle_cloud = clear_obstacle_cloud
            active_result = active_view_fn(
                arm,
                clear_view_args,
                clear_cloud,
                clear_scene_cloud,
                T_base_camera,
                clear_inst,
                clear_dir,
            )
            record_motion_fn(args, active_result.motion_sent)
            if not active_result.restore_ok:
                print(
                    "[clear][abort] failed to restore the expected center joint "
                    "configuration after active observation; clear execution is forbidden"
                )
                return None
            if active_result.used_multiview and active_result.kept_view_count >= 2:
                clear_active_label = "multiview"
            elif (
                active_result.reason == "center_view_only"
                and active_result.kept_view_count == 1
            ):
                print(
                    "[clear] no side view passed the safety/quality gates; replan the same "
                    "required clear target from the successfully captured center view"
                )
                clear_active_label = "center_fallback"
            else:
                print(
                    f"[clear][abort] active clear fallback unavailable: "
                    f"reason={active_result.reason} kept_views={active_result.kept_view_count}; "
                    "the obstacle was not cleared and the original target remains forbidden"
                )
                return None
            clear_multi_args = copy.copy(args)
            clear_multi_args.grasp_validation_mode = False
            clear_multi_args.yes = bool(args.yes or args.yes_clear)
            clear_multiview_plan_kwargs = {}
            active_environment_map = getattr(
                active_result, "motion_environment_map", None
            )
            if active_environment_map is not None:
                clear_multiview_plan_kwargs["motion_environment_map"] = (
                    active_environment_map
                )
            clear_result = plan_grasp_fn(
                arm,
                clear_multi_args,
                active_result.target_cloud,
                active_result.scene_cloud,
                active_result.T_base_camera,
                bgr,
                intr,
                clear_dir,
                clear_inst,
                plan_label=(
                    f"clear{round_number}{retry_label_suffix}_{clear_active_label}"
                ),
                place_xyz_override=args.clear_place_xyz,
                confirmation_word="CLEAR",
                motion_scene_cloud=(
                    active_result.motion_scene_cloud
                    if active_result.motion_scene_cloud is not None
                    else active_result.scene_cloud
                ),
                motion_environment_cloud=(
                    active_result.motion_environment_cloud
                    if active_result.motion_environment_cloud is not None
                    else clear_obstacle_cloud
                ),
                **clear_multiview_plan_kwargs,
            )
            record_motion_fn(args, clear_result.motion_sent)
        if (
            clear_result.status is GraspAttemptStatus.EXECUTION_FAILED
            and clear_result.empty_grasp_recovered_to_initial
        ):
            if empty_grasp_retries_used >= max_empty_grasp_retries:
                print(
                    "[clear][abort] confirmed empty grasp returned to the initial "
                    f"observation pose, but semantic retry limit "
                    f"{max_empty_grasp_retries} is exhausted; target grasp is forbidden"
                )
                return None
            reference_label = str(
                original_target_reference_inst.get(
                    "class_label",
                    original_target_reference_inst.get("label", ""),
                )
            ).strip()
            strict_reference_available = bool(
                original_target_reference_mask is not None
                and np.any(original_target_reference_mask)
                and reference_label.lower()
                not in {"", "manual", "unknown", "target"}
                and not bool(
                    original_target_reference_inst.get(
                        "clear_fallback_largest_visible",
                        False,
                    )
                )
            )
            if not strict_reference_available:
                print(
                    "[clear][abort] empty grasp recovered safely, but the first "
                    "selected target has no strict mask/label reference; refusing "
                    "an automatic retry that could switch targets"
                )
                return None
            empty_grasp_retries_used += 1
            print(
                f"[clear][retry] confirmed empty grasp and verified return to the "
                f"initial observation pose; recapture RGB-D and repeat semantic/"
                f"occlusion assessment for the original target {reference_label!r} "
                f"(retry {empty_grasp_retries_used}/{max_empty_grasp_retries})"
            )
            recaptured = recapture_fn(
                args,
                original_target_reference_mask,
                original_target_reference_inst,
            )
            if recaptured is None:
                print(
                    "[clear][abort] original target could not be re-identified "
                    "after the empty grasp; stop before any further motion"
                )
                return None
            current = recaptured
            continue
        if not clear_result.succeeded:
            print(
                f"[clear][abort] required obstacle was not cleared: "
                f"status={clear_result.status.value} detail={clear_result.detail}; "
                "target grasp is forbidden"
            )
            return None
        clears_done += 1
        empty_grasp_retries_used = 0
        print("[clear] clear-first executed; recapturing original target before active view / target grasp")
        recaptured = recapture_fn(
            args,
            original_target_reference_mask,
            original_target_reference_inst,
        )
        if recaptured is None:
            print("[clear][warn] recapture after clear failed; stop before target grasp")
            return None
        current = recaptured
