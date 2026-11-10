import gc
import logging
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO


logging.getLogger("ultralytics").setLevel(logging.WARNING)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO26_MODEL_DIR = os.path.expanduser(os.environ.get("YOLO26_MODEL_DIR", "~/Zgj"))
_YOLO26_SEG_MODEL = None
_YOLO26_AMODAL_MODEL = None
_LAST_SELECTED_INSTANCE = None


def _register_yolo26_compat_modules():
    """Register legacy YOLO26 layer names used by the saved checkpoints."""
    try:
        import ultralytics.nn.modules.head as head
        import ultralytics.nn.modules.block as block

        if not hasattr(head, "Segment26") and hasattr(head, "Segment"):
            head.Segment26 = head.Segment
        if not hasattr(block, "Proto26") and hasattr(block, "Proto"):
            block.Proto26 = block.Proto
    except Exception as exc:
        print(f"[YOLO26] warning: failed to register compatibility modules: {type(exc).__name__}: {exc}")


def _read_bgr_image(image_or_path):
    """
    image_or_path: str(path) or BGR np.ndarray
    return: uint8 BGR image
    """
    if isinstance(image_or_path, str):
        bgr_img = cv2.imread(image_or_path)
        if bgr_img is None:
            raise ValueError(f"Failed to read image from path: {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        bgr_img = image_or_path.copy()
    else:
        raise TypeError("image_or_path must be a file path or BGR numpy array")

    if bgr_img.dtype != np.uint8:
        bgr_img = np.clip(bgr_img, 0, 255).astype(np.uint8)
    return bgr_img


def _resolve_yolo26seg_weights():
    weights = os.environ.get("YOLO26_SEG_WEIGHTS", "").strip()
    if weights:
        return weights

    candidates = (
        os.path.join(YOLO26_MODEL_DIR, "visible_best_v2.pt"),
        os.path.join(ROOT_DIR, "visible_best_v2.pt"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "YOLO26 visible weights not found. Searched: {}".format(", ".join(candidates))
    )


def get_yolo26seg_model():
    global _YOLO26_SEG_MODEL
    if _YOLO26_SEG_MODEL is None:
        weights = _resolve_yolo26seg_weights()
        _register_yolo26_compat_modules()
        print(f"[YOLO26] loading weights: {weights}")
        _YOLO26_SEG_MODEL = YOLO(weights)
    return _YOLO26_SEG_MODEL


def _resolve_yolo26_amodal_weights():
    weights = os.environ.get("YOLO26_AMODAL_WEIGHTS", "").strip()
    if weights:
        return weights

    candidates = (
        os.path.join(YOLO26_MODEL_DIR, "amodal_best_v2.pt"),
        os.path.join(ROOT_DIR, "amodal_best_v2.pt"),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "YOLO26 amodal weights not found. Searched: {}".format(", ".join(candidates))
    )


def get_yolo26_amodal_model():
    global _YOLO26_AMODAL_MODEL
    if _YOLO26_AMODAL_MODEL is None:
        weights = _resolve_yolo26_amodal_weights()
        _register_yolo26_compat_modules()
        print(f"[YOLO26Amodal] loading weights: {weights}")
        _YOLO26_AMODAL_MODEL = YOLO(weights)
    return _YOLO26_AMODAL_MODEL


def get_last_selected_instance():
    return None if _LAST_SELECTED_INSTANCE is None else dict(_LAST_SELECTED_INSTANCE)


def release_vision_model_caches(*, collect_garbage=True):
    """Release cached YOLO26-seg model so GraspNet can reclaim GPU memory."""
    global _YOLO26_SEG_MODEL, _YOLO26_AMODAL_MODEL

    models = [_YOLO26_SEG_MODEL, _YOLO26_AMODAL_MODEL]
    _YOLO26_SEG_MODEL = None
    _YOLO26_AMODAL_MODEL = None
    for model in models:
        if model is None:
            continue
        try:
            if hasattr(model, "model") and hasattr(model.model, "to"):
                model.model.to("cpu")
        except Exception:
            pass
        del model

    # Full Python GC holds the interpreter lock; defer it while the same
    # process owns the high-rate MIT sender/receiver threads.
    if collect_garbage:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _mask_center(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def _class_matches(label, target_class):
    if target_class is None or str(target_class).strip() == "":
        return True
    label_norm = str(label).strip().lower().replace("_", " ")
    target_norm = str(target_class).strip().lower().replace("_", " ")
    return label_norm == target_norm or target_norm in label_norm or label_norm in target_norm


def _resize_mask(mask, width, height):
    mask = np.asarray(mask)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def _safe_name(text):
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out) or "unknown"


def _mask_overlay(image_bgr, instances, alpha=0.45):
    canvas = image_bgr.copy()
    for idx, inst in enumerate(instances):
        mask = (np.asarray(inst["mask"]) > 0)
        color = (
            int((37 * (idx + 3)) % 255),
            int((97 * (idx + 5)) % 255),
            int((173 * (idx + 7)) % 255),
        )
        if np.any(mask):
            color_img = np.zeros_like(canvas, dtype=np.uint8)
            color_img[:] = color
            canvas[mask] = (canvas[mask] * (1.0 - alpha) + color_img[mask] * alpha).astype(np.uint8)

        x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(inst["xyxy"], dtype=float).reshape(4,)]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        label = f"{idx}:{inst['label']} {float(inst['score']):.2f}"
        cv2.putText(canvas, label, (x0, max(18, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def _mask_bool(mask):
    return np.asarray(mask) > 0


def _mask_iou(mask_a, mask_b):
    a = _mask_bool(mask_a)
    b = _mask_bool(mask_b)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    return float(np.count_nonzero(a & b) / max(union, 1))


def _mask_xyxy(mask):
    m = _mask_bool(mask)
    ys, xs = np.where(m)
    if xs.size == 0 or ys.size == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _xyxy_center(xyxy):
    arr = np.asarray(xyxy, dtype=np.float32).reshape(4,)
    return np.array([(arr[0] + arr[2]) * 0.5, (arr[1] + arr[3]) * 0.5], dtype=np.float32)


def _same_instance_label(label_a, label_b):
    a = str(label_a or "").strip().lower().replace("_", " ")
    b = str(label_b or "").strip().lower().replace("_", " ")
    ac = a.replace(" ", "").replace("-", "")
    bc = b.replace(" ", "").replace("-", "")
    return bool(a and b and (a == b or a in b or b in a or ac == bc or ac in bc or bc in ac))


def compute_visible_amodal_occlusion_metrics(visible_mask, amodal_mask):
    """
    Compare a visible mask with an amodal mask.

    visible_ratio is the fraction of the amodal area that is currently visible.
    occlusion_ratio is 1 - visible_ratio, measured inside the amodal mask.
    """
    amodal = _mask_bool(amodal_mask)
    visible = _mask_bool(visible_mask)
    if visible.shape != amodal.shape:
        visible = cv2.resize(
            visible.astype(np.uint8),
            (amodal.shape[1], amodal.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    amodal_area = int(np.count_nonzero(amodal))
    visible_area = int(np.count_nonzero(visible))
    intersection = visible & amodal
    intersection_area = int(np.count_nonzero(intersection))
    hidden = amodal & ~visible
    hidden_area = int(np.count_nonzero(hidden))
    union_area = int(np.count_nonzero(visible | amodal))

    if amodal_area <= 0:
        visible_ratio = 0.0
        occlusion_ratio = 0.0
    else:
        visible_ratio = float(np.clip(intersection_area / max(amodal_area, 1), 0.0, 1.0))
        occlusion_ratio = float(np.clip(hidden_area / max(amodal_area, 1), 0.0, 1.0))

    visible_inside_amodal_ratio = 0.0
    if visible_area > 0:
        visible_inside_amodal_ratio = float(np.clip(intersection_area / max(visible_area, 1), 0.0, 1.0))

    iou = 0.0
    if union_area > 0:
        iou = float(intersection_area / max(union_area, 1))

    return {
        "visible_area_px": visible_area,
        "amodal_area_px": amodal_area,
        "intersection_area_px": intersection_area,
        "hidden_area_px": hidden_area,
        "visible_ratio": visible_ratio,
        "occlusion_ratio": occlusion_ratio,
        "visible_inside_amodal_ratio": visible_inside_amodal_ratio,
        "iou": iou,
        "hidden_mask": hidden.astype(np.uint8) * 255,
        "visible_in_amodal_mask": intersection.astype(np.uint8) * 255,
    }


def _visible_amodal_match_score(visible_inst, amodal_inst):
    metrics = compute_visible_amodal_occlusion_metrics(visible_inst["mask"], amodal_inst["mask"])
    h, w = np.asarray(amodal_inst["mask"]).shape[:2]
    diag = float(np.hypot(max(w, 1), max(h, 1)))
    visible_center = _xyxy_center(visible_inst.get("xyxy", _mask_xyxy(visible_inst["mask"])))
    amodal_center = _xyxy_center(amodal_inst.get("xyxy", _mask_xyxy(amodal_inst["mask"])))
    center_dist = float(np.linalg.norm(visible_center - amodal_center))
    center_score = float(np.clip(1.0 - center_dist / max(diag, 1.0), 0.0, 1.0))
    score = (
        1.40 * metrics["visible_inside_amodal_ratio"]
        + 0.90 * metrics["iou"]
        + 0.35 * metrics["visible_ratio"]
        + 0.15 * center_score
        + 0.05 * float(visible_inst.get("score", 0.0))
        + 0.05 * float(amodal_inst.get("score", 0.0))
    )
    return float(score), metrics


def _merge_visible_instance_fragments(visible_instances, visible_indices):
    indices = sorted({int(index) for index in visible_indices})
    if not indices:
        raise ValueError("visible_indices must not be empty")

    fragments = [visible_instances[index] for index in indices]
    reference_mask = _mask_bool(fragments[0]["mask"])
    merged_mask = np.zeros(reference_mask.shape, dtype=bool)
    for fragment in fragments:
        fragment_mask = _mask_bool(fragment["mask"])
        if fragment_mask.shape != merged_mask.shape:
            fragment_mask = cv2.resize(
                fragment_mask.astype(np.uint8),
                (merged_mask.shape[1], merged_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        merged_mask |= fragment_mask

    strongest = max(fragments, key=lambda inst: float(inst.get("score", 0.0)))
    merged = dict(strongest)
    merged["mask"] = merged_mask.astype(np.uint8) * 255
    merged["xyxy"] = _mask_xyxy(merged_mask)
    merged["center"] = _mask_center(merged_mask)
    merged["score"] = max(float(inst.get("score", 0.0)) for inst in fragments)
    merged["fragment_count"] = len(indices)
    merged["source_visible_indices"] = indices
    return merged


def match_visible_amodal_instances(visible_instances,
                                   amodal_instances,
                                   target_class=None,
                                   min_match_score=0.05,
                                   min_visible_inside_ratio=0.05,
                                   min_iou=0.02,
                                   merge_visible_fragments=True,
                                   fragment_min_inside_ratio=0.60,
                                   fragment_min_intersection_px=32,
                                   min_occlusion_containment_ratio=0.85):
    """
    Match visible detections to amodal detections and attach occlusion metrics.

    The returned records are centered on amodal instances, because occlusion rate
    needs a full-object mask as denominator. Unmatched visible detections are
    included with status="amodal_missing" so bad amodal misses are easy to spot.
    """
    visible_instances = list(visible_instances or [])
    amodal_instances = list(amodal_instances or [])

    if target_class is not None and str(target_class).strip():
        visible_instances = [inst for inst in visible_instances if _class_matches(inst.get("label", ""), target_class)]
        amodal_instances = [inst for inst in amodal_instances if _class_matches(inst.get("label", ""), target_class)]

    matches = []
    used_visible = set()
    sorted_amodal = sorted(
        enumerate(amodal_instances),
        key=lambda item: (float(item[1].get("score", 0.0)), int(np.count_nonzero(item[1].get("mask", 0)))),
        reverse=True,
    )

    for amodal_index, amodal_inst in sorted_amodal:
        best_visible_index = None
        best_score = -np.inf
        best_metrics = None
        for visible_index, visible_inst in enumerate(visible_instances):
            if visible_index in used_visible:
                continue
            if not _same_instance_label(visible_inst.get("label", ""), amodal_inst.get("label", "")):
                continue
            score, metrics = _visible_amodal_match_score(visible_inst, amodal_inst)
            if score > best_score:
                best_visible_index = visible_index
                best_score = score
                best_metrics = metrics

        visible_inst = None
        status = "visible_missing"
        overlap_ok = False
        if best_metrics is not None:
            overlap_ok = (
                float(best_metrics.get("visible_inside_amodal_ratio", 0.0)) >= float(min_visible_inside_ratio)
                or float(best_metrics.get("iou", 0.0)) >= float(min_iou)
            )
        if best_visible_index is not None and best_score >= float(min_match_score) and overlap_ok:
            visible_inst = visible_instances[best_visible_index]
            used_visible.add(best_visible_index)
            status = "matched"
        elif best_visible_index is not None:
            visible_inst = visible_instances[best_visible_index]
            used_visible.add(best_visible_index)
            status = "match_failed"
        else:
            empty_visible = np.zeros_like(np.asarray(amodal_inst["mask"], dtype=np.uint8))
            best_metrics = compute_visible_amodal_occlusion_metrics(empty_visible, amodal_inst["mask"])
            best_score = 0.0

        metrics = dict(best_metrics)
        record = {
            "status": status,
            "label": str(amodal_inst.get("label", "")),
            "occlusion_valid": bool(status == "matched"),
            "match_score": float(best_score),
            "visible_instance": visible_inst,
            "amodal_instance": amodal_inst,
            "visible_index": None if best_visible_index is None else int(best_visible_index),
            "visible_indices": [] if best_visible_index is None else [int(best_visible_index)],
            "amodal_index": int(amodal_index),
            **metrics,
        }
        matches.append(record)

    if bool(merge_visible_fragments):
        fragment_assignments = {}
        for visible_index, visible_inst in enumerate(visible_instances):
            if visible_index in used_visible:
                continue

            best_match_index = None
            best_rank = None
            for match_index, rec in enumerate(matches):
                if rec.get("status") != "matched":
                    continue
                amodal_inst = rec.get("amodal_instance")
                if amodal_inst is None or not _same_instance_label(
                    visible_inst.get("label", ""), amodal_inst.get("label", "")
                ):
                    continue

                score, metrics = _visible_amodal_match_score(visible_inst, amodal_inst)
                inside_ratio = float(metrics.get("visible_inside_amodal_ratio", 0.0))
                intersection_px = int(metrics.get("intersection_area_px", 0))
                if inside_ratio < float(fragment_min_inside_ratio):
                    continue
                if intersection_px < int(fragment_min_intersection_px):
                    continue

                rank = (
                    inside_ratio,
                    float(metrics.get("iou", 0.0)),
                    intersection_px,
                    float(score),
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_match_index = match_index

            if best_match_index is not None:
                fragment_assignments.setdefault(best_match_index, []).append(visible_index)
                used_visible.add(visible_index)

        for match_index, extra_indices in fragment_assignments.items():
            rec = matches[match_index]
            visible_indices = list(rec.get("visible_indices") or []) + list(extra_indices)
            merged_visible = _merge_visible_instance_fragments(visible_instances, visible_indices)
            merged_score, merged_metrics = _visible_amodal_match_score(
                merged_visible,
                rec["amodal_instance"],
            )
            rec.update(merged_metrics)
            rec["match_score"] = float(merged_score)
            rec["visible_instance"] = merged_visible
            rec["visible_indices"] = sorted({int(index) for index in visible_indices})

    for rec in matches:
        if rec.get("status") != "matched":
            continue
        containment = float(rec.get("visible_inside_amodal_ratio", 0.0))
        if containment < float(min_occlusion_containment_ratio):
            rec["status"] = "containment_failed"
            rec["occlusion_valid"] = False

    for visible_index, visible_inst in enumerate(visible_instances):
        if visible_index in used_visible:
            continue
        matches.append({
            "status": "amodal_missing",
            "label": str(visible_inst.get("label", "")),
            "occlusion_valid": False,
            "match_score": 0.0,
            "visible_instance": visible_inst,
            "amodal_instance": None,
            "visible_index": int(visible_index),
            "visible_indices": [int(visible_index)],
            "amodal_index": None,
            "visible_area_px": int(np.count_nonzero(visible_inst.get("mask", 0))),
            "amodal_area_px": 0,
            "intersection_area_px": 0,
            "hidden_area_px": 0,
            "visible_ratio": None,
            "occlusion_ratio": None,
            "visible_inside_amodal_ratio": None,
            "iou": None,
            "hidden_mask": np.zeros_like(np.asarray(visible_inst["mask"], dtype=np.uint8)),
            "visible_in_amodal_mask": np.asarray(visible_inst["mask"], dtype=np.uint8),
        })

    matches.sort(
        key=lambda rec: (
            rec["occlusion_ratio"] is None,
            -float(rec["amodal_area_px"]),
            -float(rec.get("match_score", 0.0)),
        )
    )
    return matches


def draw_visible_amodal_occlusion_overlay(image_bgr, matches, alpha_visible=0.35, alpha_hidden=0.58):
    canvas = image_bgr.copy()
    visible_color = np.array([30, 210, 70], dtype=np.float32)
    hidden_color = np.array([40, 40, 235], dtype=np.float32)
    amodal_color = (0, 220, 255)
    missing_color = (255, 180, 20)

    for idx, rec in enumerate(matches):
        visible_inst = rec.get("visible_instance")
        amodal_inst = rec.get("amodal_instance")
        if amodal_inst is not None:
            amodal_mask = _mask_bool(amodal_inst["mask"])
            hidden_mask = _mask_bool(rec.get("hidden_mask", np.zeros_like(amodal_inst["mask"])))
            visible_mask = _mask_bool(rec.get("visible_in_amodal_mask", np.zeros_like(amodal_inst["mask"])))
            if np.any(visible_mask):
                canvas[visible_mask] = (
                    canvas[visible_mask].astype(np.float32) * (1.0 - alpha_visible)
                    + visible_color * alpha_visible
                ).astype(np.uint8)
            if np.any(hidden_mask):
                canvas[hidden_mask] = (
                    canvas[hidden_mask].astype(np.float32) * (1.0 - alpha_hidden)
                    + hidden_color * alpha_hidden
                ).astype(np.uint8)

            contours, _ = cv2.findContours(
                amodal_mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, amodal_color, 2)
            x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(amodal_inst["xyxy"], dtype=float).reshape(4,)]
            color = amodal_color
        elif visible_inst is not None:
            visible_mask = _mask_bool(visible_inst["mask"])
            if np.any(visible_mask):
                canvas[visible_mask] = (
                    canvas[visible_mask].astype(np.float32) * (1.0 - alpha_visible)
                    + visible_color * alpha_visible
                ).astype(np.uint8)
            x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(visible_inst["xyxy"], dtype=float).reshape(4,)]
            color = missing_color
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        else:
            continue

        occ = rec.get("occlusion_ratio")
        vis_ratio = rec.get("visible_ratio")
        status = str(rec.get("status", ""))
        if occ is None:
            label = f"{idx}:{rec.get('label', '')} occ=n/a"
        elif not bool(rec.get("occlusion_valid", False)):
            label = f"{idx}:{rec.get('label', '')} {status}"
        else:
            label = f"{idx}:{rec.get('label', '')} occ={float(occ) * 100:.0f}% vis={float(vis_ratio) * 100:.0f}%"
        y = max(18, y0 - 7)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.rectangle(canvas, (x0, y - text_h - baseline - 3), (x0 + text_w + 6, y + baseline + 2), (20, 20, 20), -1)
        cv2.putText(canvas, label, (x0 + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    return canvas


def infer_visible_amodal_occlusion(image_or_path,
                                   target_class=None,
                                   visible_conf_thresh=0.25,
                                   amodal_conf_thresh=0.20,
                                   output_vis_path=None):
    bgr_img = _read_bgr_image(image_or_path)
    visible_instances, _ = infer_instances_yolo26seg(
        bgr_img,
        target_class=target_class,
        output_vis_path=None,
        conf_thresh=visible_conf_thresh,
    )
    amodal_instances, _ = infer_instances_yolo26_amodal(
        bgr_img,
        target_class=target_class,
        output_vis_path=None,
        conf_thresh=amodal_conf_thresh,
    )
    matches = match_visible_amodal_instances(
        visible_instances,
        amodal_instances,
        target_class=target_class,
    )
    vis_img = draw_visible_amodal_occlusion_overlay(bgr_img, matches)
    if output_vis_path is not None:
        cv2.imwrite(output_vis_path, vis_img)
    return matches, vis_img, visible_instances, amodal_instances


def _limit_instances_per_class(instances, max_per_class):
    if int(max_per_class) <= 0:
        return instances
    grouped = {}
    for inst in instances:
        grouped.setdefault(str(inst["label"]), []).append(inst)
    kept = []
    for label in sorted(grouped):
        items = sorted(
            grouped[label],
            key=lambda x: (float(x["score"]), int(np.count_nonzero(x["mask"]))),
            reverse=True,
        )
        kept.extend(items[: int(max_per_class)])
    kept.sort(
        key=lambda x: (float(x["score"]), int(np.count_nonzero(x["mask"]))),
        reverse=True,
    )
    return kept


def _debug_dir():
    path = os.environ.get("YOLO26_SEG_DEBUG_DIR", os.path.join(ROOT_DIR, "dbg_yolo26seg"))
    os.makedirs(path, exist_ok=True)
    return path


def _debug_prefix(output_mask=None, target_class=None, tag="detect"):
    base = os.path.splitext(os.path.basename(str(output_mask or tag)))[0]
    cls = "all" if target_class is None or str(target_class).strip() == "" else _safe_name(target_class)
    return f"{base}_{cls}"


def save_yolo26seg_debug(image_bgr, instances, prefix, selected_index=None):
    out_dir = _debug_dir()
    overlay = _mask_overlay(image_bgr, instances)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_overlay.png"), overlay)

    for idx, inst in enumerate(instances):
        mask_name = f"{prefix}_mask_{idx:02d}_{_safe_name(inst['label'])}.png"
        cv2.imwrite(os.path.join(out_dir, mask_name), np.asarray(inst["mask"], dtype=np.uint8))

    print(f"[YOLO26Debug] saved {len(instances)} instances to {out_dir}/{prefix}_*")


def infer_instances_yolo26seg(image_or_path,
                              target_class=None,
                              output_vis_path=None,
                              conf_thresh=0.25):
    """
    YOLO26 instance segmentation replacement for YOLOWorld+SAM.

    Returns:
        instances: [
            {
                "label": str,
                "score": float,
                "xyxy": np.ndarray(4,),
                "mask": np.ndarray(H,W) uint8,
                "center": tuple or None,
            },
            ...
        ]
        vis_img: visualization image
    """
    bgr_img = _read_bgr_image(image_or_path)
    height, width = bgr_img.shape[:2]

    model = get_yolo26seg_model()
    imgsz = int(os.environ.get("YOLO26_SEG_IMGSZ", "640"))
    env_conf = os.environ.get("YOLO26_SEG_CONF", "").strip()
    if env_conf:
        conf_thresh = float(env_conf)
    results = model.predict(
        bgr_img,
        conf=float(conf_thresh),
        imgsz=imgsz,
        retina_masks=True,
        verbose=False,
    )

    result = results[0]
    vis_img = result.plot() if hasattr(result, "plot") else bgr_img.copy()
    if output_vis_path is not None and vis_img is not None:
        cv2.imwrite(output_vis_path, vis_img)

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or masks.data is None or len(boxes) == 0:
        return [], vis_img

    masks_np = masks.data.detach().cpu().numpy()
    names = getattr(result, "names", {}) or {}
    instances = []

    for i, box in enumerate(boxes):
        conf = float(box.conf.item())
        cls_id = int(box.cls.item())
        label = str(names.get(cls_id, cls_id))
        if conf < float(conf_thresh) or not _class_matches(label, target_class):
            continue
        if i >= len(masks_np):
            continue

        mask = _resize_mask(masks_np[i], width, height)
        if int(np.count_nonzero(mask)) == 0:
            continue

        instances.append({
            "label": label,
            "score": conf,
            "xyxy": box.xyxy[0].detach().cpu().numpy().astype(np.float32),
            "mask": mask,
            "center": _mask_center(mask),
        })

    instances.sort(
        key=lambda x: (x["score"], np.count_nonzero(x["mask"])),
        reverse=True,
    )
    max_per_class = int(os.environ.get("YOLO26_SEG_MAX_PER_CLASS", "0"))
    instances = _limit_instances_per_class(instances, max_per_class)
    return instances, vis_img


def infer_instances_yolo26_amodal(image_or_path,
                                  target_class=None,
                                  output_vis_path=None,
                                  conf_thresh=0.20):
    """
    Amodal YOLO26 instance segmentation. This mirrors infer_instances_yolo26seg()
    but uses the amodal checkpoint and keeps its cache separate from visible YOLO.
    """
    bgr_img = _read_bgr_image(image_or_path)
    height, width = bgr_img.shape[:2]

    model = get_yolo26_amodal_model()
    imgsz = int(os.environ.get("YOLO26_AMODAL_IMGSZ", os.environ.get("YOLO26_SEG_IMGSZ", "640")))
    env_conf = os.environ.get("YOLO26_AMODAL_CONF", "").strip()
    if env_conf:
        conf_thresh = float(env_conf)
    results = model.predict(
        bgr_img,
        conf=float(conf_thresh),
        imgsz=imgsz,
        retina_masks=True,
        verbose=False,
    )

    result = results[0]
    vis_img = result.plot() if hasattr(result, "plot") else bgr_img.copy()
    if output_vis_path is not None and vis_img is not None:
        cv2.imwrite(output_vis_path, vis_img)

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or masks.data is None or len(boxes) == 0:
        return [], vis_img

    masks_np = masks.data.detach().cpu().numpy()
    names = getattr(result, "names", {}) or {}
    instances = []

    for i, box in enumerate(boxes):
        conf = float(box.conf.item())
        cls_id = int(box.cls.item())
        label = str(names.get(cls_id, cls_id))
        if conf < float(conf_thresh) or not _class_matches(label, target_class):
            continue
        if i >= len(masks_np):
            continue

        mask = _resize_mask(masks_np[i], width, height)
        if int(np.count_nonzero(mask)) == 0:
            continue

        instances.append({
            "label": label,
            "score": conf,
            "xyxy": box.xyxy[0].detach().cpu().numpy().astype(np.float32),
            "mask": mask,
            "center": _mask_center(mask),
        })

    instances.sort(
        key=lambda x: (x["score"], np.count_nonzero(x["mask"])),
        reverse=True,
    )
    max_per_class = int(os.environ.get("YOLO26_AMODAL_MAX_PER_CLASS", "0"))
    instances = _limit_instances_per_class(instances, max_per_class)
    return instances, vis_img


def _infer_with_optional_low_conf_retry(image_or_path,
                                        target_class=None,
                                        output_vis_path=None,
                                        conf_thresh=0.25,
                                        retry_if_empty=True):
    instances, vis_img = infer_instances_yolo26seg(
        image_or_path=image_or_path,
        target_class=target_class,
        output_vis_path=output_vis_path,
        conf_thresh=conf_thresh,
    )
    if instances or not retry_if_empty:
        return instances, vis_img

    fallback_conf = float(os.environ.get("YOLO26_SEG_FALLBACK_CONF", "0.05"))
    if fallback_conf >= float(conf_thresh):
        return instances, vis_img

    print(f"[YOLO26] no masks at conf={float(conf_thresh):.3f}; retry with conf={fallback_conf:.3f}")
    return infer_instances_yolo26seg(
        image_or_path=image_or_path,
        target_class=target_class,
        output_vis_path=output_vis_path,
        conf_thresh=fallback_conf,
    )


def detect_objects(image_or_path, target_class=None, conf_thresh=0.25):
    """
    Detect objects with YOLO26-seg.

    Returns the same box-oriented structure as cv_process_updated.detect_objects().
    """
    instances, vis_img = _infer_with_optional_low_conf_retry(
        image_or_path=image_or_path,
        target_class=target_class,
        output_vis_path=None,
        conf_thresh=conf_thresh,
    )

    valid_boxes = []
    for inst in instances:
        valid_boxes.append({
            "xyxy": np.asarray(inst["xyxy"], dtype=np.float32).tolist(),
            "conf": float(inst["score"]),
            "cls": str(inst["label"]),
        })

    return valid_boxes, vis_img


def _manual_click_select_instance(image_or_path, instances, vis_img=None, return_index=False):
    """Fallback target selection by clicking one YOLO26-seg instance mask."""
    bgr_img = _read_bgr_image(image_or_path)
    show_img = vis_img if vis_img is not None else bgr_img
    cv2.imshow("Select Object", show_img)

    point = []
    clicked = False

    def click_handler(event, x, y, flags, param):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at ({x}, {y})")
            point[:] = [x, y]
            clicked = True

    cv2.setMouseCallback("Select Object", click_handler)
    print("Waiting for user click on a YOLO26-seg instance...")

    while not clicked:
        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyAllWindows()
            raise ValueError("User cancelled selection")

    cv2.destroyAllWindows()

    if len(point) != 2:
        raise ValueError("No selection made")

    x, y = point
    ranked = []
    for idx, inst in enumerate(instances):
        mask = (np.asarray(inst["mask"]) > 0).astype(np.uint8)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            ranked.append((idx, inst))

    if not ranked:
        fallback = []
        for idx, inst in enumerate(instances):
            x0, y0, x1, y1 = [float(v) for v in np.asarray(inst["xyxy"], dtype=float).reshape(4,)]
            dx = max(x0 - float(x), 0.0, float(x) - x1)
            dy = max(y0 - float(y), 0.0, float(y) - y1)
            bbox_dist = float(np.hypot(dx, dy))
            center = inst.get("center", None)
            if center is None:
                cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            else:
                cx, cy = float(center[0]), float(center[1])
            center_dist = float(np.hypot(float(x) - cx, float(y) - cy))
            fallback.append((bbox_dist, center_dist, idx, inst))

        if not fallback:
            raise ValueError("No YOLO26-seg instances available for click selection")

        max_dist = float(os.environ.get("YOLO26_CLICK_FALLBACK_MAX_DIST", "120"))
        fallback.sort(key=lambda item: (item[0], item[1]))
        bbox_dist, center_dist, best_idx, best = fallback[0]
        if bbox_dist > max_dist:
            raise ValueError(
                f"Click did not hit any YOLO26-seg instance mask, nearest bbox is {bbox_dist:.1f}px away "
                f"(limit={max_dist:.1f}px)"
            )
        print(
            f"[TargetSelect] click missed masks; fallback to nearest instance "
            f"idx={best_idx} label={best['label']} score={float(best['score']):.2f} "
            f"bbox_dist={bbox_dist:.1f}px center_dist={center_dist:.1f}px"
        )
        if return_index:
            return best["center"], best["mask"], int(best_idx)
        return best["center"], best["mask"]

    ranked.sort(key=lambda item: (item[1]["score"], np.count_nonzero(item[1]["mask"])), reverse=True)
    best_idx, best = ranked[0]
    if return_index:
        return best["center"], best["mask"], int(best_idx)
    return best["center"], best["mask"]


def segment_image(image_or_path,
                  output_mask='mask1.png',
                  target_class=None,
                  interactive=True,
                  allow_manual_click=True,
                  conf_thresh=0.25,
                  force_manual_click=False):
    """
    Target instance segmentation.

    This keeps the old cv_process_updated.segment_image() signature, but uses
    YOLO26-seg masks directly instead of YOLOWorld boxes + SAM masks.
    """
    global _LAST_SELECTED_INSTANCE
    _LAST_SELECTED_INSTANCE = None

    if target_class is None and interactive:
        target_class = input("\n===============\nEnter class name: ").strip()
    elif target_class is not None:
        target_class = str(target_class).strip()

    force_manual_click = bool(force_manual_click)
    if target_class is not None and str(target_class).strip().lower() in {"click", "manual", "select", "__manual_click__"}:
        force_manual_click = True
        target_class = None

    instances, vis_img = _infer_with_optional_low_conf_retry(
        image_or_path=image_or_path,
        target_class=target_class,
        output_vis_path='detection_visualization.jpg',
        conf_thresh=conf_thresh,
    )

    selected_idx = None
    if force_manual_click:
        all_instances = instances
        all_vis = vis_img
        if not all_instances:
            all_instances, all_vis = _infer_with_optional_low_conf_retry(
                image_or_path=image_or_path,
                target_class=None,
                output_vis_path='detection_visualization.jpg',
                conf_thresh=conf_thresh,
            )
        if not all_instances:
            save_yolo26seg_debug(
                _read_bgr_image(image_or_path),
                [],
                prefix=_debug_prefix(output_mask=output_mask, target_class="empty", tag="segment"),
                selected_index=None,
            )
            raise ValueError("No YOLO26-seg instance masks available for click selection.")
        save_yolo26seg_debug(
            _read_bgr_image(image_or_path),
            all_instances,
            prefix=_debug_prefix(output_mask=output_mask, target_class="click_candidates", tag="segment"),
            selected_index=None,
        )
        print("Click one YOLO26-seg instance")
        center, mask, selected_idx = _manual_click_select_instance(
            image_or_path,
            all_instances,
            vis_img=all_vis,
            return_index=True,
        )
        instances = all_instances
        _LAST_SELECTED_INSTANCE = instances[int(selected_idx)]
        print(f"Selected {instances[int(selected_idx)]['label']} with confidence {instances[int(selected_idx)]['score']:.2f}")
    elif len(instances) > 0:
        selected_idx = 0
        best = instances[0]
        center, mask = best["center"], best["mask"]
        _LAST_SELECTED_INSTANCE = best
        print(f"Auto-selected {best['label']} with confidence {best['score']:.2f}")
    else:
        if not allow_manual_click:
            raise ValueError("No YOLO26-seg detections and manual click disabled.")

        all_instances, all_vis = _infer_with_optional_low_conf_retry(
            image_or_path=image_or_path,
            target_class=None,
            output_vis_path='detection_visualization.jpg',
            conf_thresh=conf_thresh,
        )
        if not all_instances:
            save_yolo26seg_debug(
                _read_bgr_image(image_or_path),
                [],
                prefix=_debug_prefix(output_mask=output_mask, target_class="empty", tag="segment"),
                selected_index=None,
            )
            raise ValueError("No YOLO26-seg instance masks available for click selection.")

        save_yolo26seg_debug(
            _read_bgr_image(image_or_path),
            all_instances,
            prefix=_debug_prefix(output_mask=output_mask, target_class="click_candidates", tag="segment"),
            selected_index=None,
        )
        print("No matching detection - click one YOLO26-seg instance")
        center, mask, selected_idx = _manual_click_select_instance(
            image_or_path,
            all_instances,
            vis_img=all_vis,
            return_index=True,
        )
        instances = all_instances
        _LAST_SELECTED_INSTANCE = instances[int(selected_idx)]
        print(f"Selected {instances[int(selected_idx)]['label']} with confidence {instances[int(selected_idx)]['score']:.2f}")

    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        save_yolo26seg_debug(
            _read_bgr_image(image_or_path),
            instances,
            prefix=_debug_prefix(output_mask=output_mask, target_class=target_class, tag="segment"),
            selected_index=selected_idx,
        )
    else:
        print("[WARNING] Could not generate mask")

    return mask


def detect_context_instances_center_view(image_or_path,
                                         target_mask,
                                         conf_thresh=0.25,
                                         context_classes=None,
                                         output_vis_path='context_detection_visualization.jpg',
                                         exclude_target_overlap=0.30):
    """
    Detect surrounding object instances with YOLO26-seg.

    Returns the same instance structure as cv_process_updated, so the obstacle
    pipeline can build its 3D scene context without other changes.
    """
    instances, _ = _infer_with_optional_low_conf_retry(
        image_or_path=image_or_path,
        target_class=None,
        output_vis_path=output_vis_path,
        conf_thresh=conf_thresh,
    )

    tgt = (target_mask > 0).astype(np.uint8)

    if context_classes is not None:
        context_classes = {str(c).lower().replace("_", " ") for c in context_classes}

    out = []
    for inst in instances:
        label = str(inst["label"]).lower().replace("_", " ")
        if context_classes is not None and label not in context_classes:
            continue

        m = (inst["mask"] > 0).astype(np.uint8)
        inter = np.count_nonzero((m > 0) & (tgt > 0))
        denom = max(np.count_nonzero(m), 1)
        overlap_ratio = inter / float(denom)

        if overlap_ratio > exclude_target_overlap:
            continue

        out.append(inst)

    save_yolo26seg_debug(
        _read_bgr_image(image_or_path),
        out,
        prefix=_debug_prefix(output_mask=output_vis_path, target_class="context", tag="context"),
        selected_index=None,
    )

    return out
