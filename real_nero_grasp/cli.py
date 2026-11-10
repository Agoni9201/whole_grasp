"""Command-line argument parser for the real Nero grasp workflow.

The parser is kept byte-for-byte equivalent to the legacy runtime function so
existing shell commands, defaults, option ordering, and help text stay stable.
"""

import argparse

import numpy as np

from real_nero_grasp.paths import DEFAULT_IK_URDF, DEFAULT_YOLO_WEIGHTS, PROJECT_ROOT
from tools.real_nero_point_test import DEFAULT_INITIAL_JOINT_DEG, DEFAULT_TCP_OFFSET


__all__ = ["parse_args"]


def parse_args():
    p = argparse.ArgumentParser(description="Single-view real Nero GraspNet runner.")
    p.add_argument("--calib-json", help="Current wrist camera to flange calibration JSON")
    p.add_argument("--handeye-convention", default="flange_camera_xyzw", choices=["flange_camera_xyzw", "camera_flange_xyzw", "flange_camera_wxyz", "camera_flange_wxyz"])
    p.add_argument("--orbbec-color-topic", default="/camera/color/image_raw")
    p.add_argument("--orbbec-depth-topic", default="/camera/depth/image_raw")
    p.add_argument("--orbbec-camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--orbbec-points-topic", default="/camera/depth_registered/points")
    p.add_argument("--use-registered-points", action="store_true", help="Use organized /camera/depth_registered/points instead of depth image back-projection")
    p.add_argument("--orbbec-depth-scale", type=float, default=None)
    p.add_argument("--orbbec-timeout", type=float, default=5.0)
    p.add_argument("--orbbec-warmup-frames", type=int, default=5)
    p.add_argument("--orbbec-depth-median-frames", type=int, default=7, help="Depth frames captured at each settled pose for a valid-pixel median")
    p.add_argument("--orbbec-depth-median-min-valid", type=int, default=5, help="Minimum valid observations required at a pixel in the depth median")
    p.add_argument("--orbbec-depth-median-max-mad", type=float, default=0.004, help="Reject a valid median-depth pixel when its temporal MAD exceeds this many metres; <=0 disables the High-Density flying-point filter")
    p.add_argument("--camera-self-mask", dest="camera_self_mask", action="store_true", help="Mask the calibrated fixed gripper silhouette from every aligned RGB-D frame")
    p.add_argument("--no-camera-self-mask", dest="camera_self_mask", action="store_false")
    p.set_defaults(camera_self_mask=False)
    p.add_argument("--camera-self-mask-calibration", default=None, help="Measured silhouette JSON with exact serial/profile/intrinsics and per-frame RGB validation; used with --camera-self-mask")
    p.add_argument("--camera-self-mask-expand-px", type=int, default=12, help="Pixel dilation around the fixed camera/gripper self-occlusion polygons")
    p.add_argument("--camera-self-edge-island-filter", dest="camera_self_edge_island_filter", action="store_true", help="Remove tiny near-depth components immediately beside the accepted gripper self mask")
    p.add_argument("--no-camera-self-edge-island-filter", dest="camera_self_edge_island_filter", action="store_false")
    p.set_defaults(camera_self_edge_island_filter=True)
    p.add_argument("--camera-self-edge-island-distance-px", type=int, default=3, help="Maximum image distance from the self-mask edge for tiny-island rejection")
    p.add_argument("--camera-self-edge-island-max-area", type=int, default=8, help="Maximum connected near-depth pixel area rejected beside the self mask")
    p.add_argument("--camera-self-edge-island-max-diameter", type=int, default=6, help="Maximum width or height of a rejected self-mask-edge island")
    p.add_argument("--camera-self-edge-island-max-depth", type=float, default=0.22, help="Maximum camera depth considered a possible tiny self-mask-edge return")
    p.add_argument("--camera-near-field-ghost-filter", dest="camera_near_field_ghost_filter", action="store_true", help="Opt in to a stream-profile-calibrated compact near-field ghost mask before building clouds")
    p.add_argument("--no-camera-near-field-ghost-filter", dest="camera_near_field_ghost_filter", action="store_false")
    p.set_defaults(camera_near_field_ghost_filter=False)
    p.add_argument("--camera-near-field-ghost-image-size", type=int, nargs=2, default=[1280, 720], metavar=("WIDTH", "HEIGHT"), help="Exact aligned RGB-D profile calibrated for the optional near-field ghost mask")
    p.add_argument("--camera-near-field-ghost-roi", type=float, nargs=4, default=[0.05, 0.28, 0.75, 1.00], metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="Normalized image ROI calibrated for the lower-left near-field ghost")
    p.add_argument("--camera-near-field-ghost-seed-depth", type=float, default=0.175, help="Maximum camera depth in meters that seeds a near-field ghost component")
    p.add_argument("--camera-near-field-ghost-grow-depth", type=float, default=0.200, help="Maximum camera depth in meters connected to a verified near-field ghost seed")
    p.add_argument("--camera-near-field-ghost-min-seed-pixels", type=int, default=8, help="Minimum very-near seed pixels required before a component can be removed")
    p.add_argument("--camera-near-field-ghost-max-area", type=int, default=5000, help="Retain larger near components fail-closed as possible real obstacles")
    p.add_argument("--camera-near-field-ghost-max-diameter", type=int, default=120, help="Retain longer/wider near components fail-closed as possible real obstacles")
    p.add_argument("--camera-near-field-preserve-dilate-px", type=int, default=8, help="Never remove the selected target mask or this many surrounding pixels")
    p.add_argument("--capture-retries", type=int, default=2, help="Retry initial RGB-D capture this many times before exiting cleanly")
    p.add_argument("--capture-retry-delay", type=float, default=1.0)
    p.add_argument("--save-dir", default="logs", help="Base directory for per-run experiment folders and the global results JSONL")
    p.add_argument("--rgb", default=None, help="Optional offline RGB image")
    p.add_argument("--depth", default=None, help="Optional offline depth image")
    p.add_argument("--depth-scale", type=float, default=None)
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)

    p.add_argument("--nero-interface", default="socketcan")
    p.add_argument("--nero-channel", default="can0")
    p.add_argument("--nero-bitrate", type=int, default=1000000)
    p.add_argument("--nero-firmware", default="V111", choices=["DEFAULT", "V111", "V112"])
    p.add_argument("--nero-enable-timeout", type=float, default=5.0)
    p.add_argument("--nero-speed-percent", type=int, default=5)
    p.add_argument("--tcp-offset", type=float, nargs=6, default=DEFAULT_TCP_OFFSET)
    p.add_argument("--initial-joints-deg", type=float, nargs=7, default=DEFAULT_INITIAL_JOINT_DEG)
    p.add_argument("--skip-initial-move", action="store_true")
    p.add_argument("--yes-initial", action="store_true")
    p.add_argument("--require-initial-reached", action="store_true", default=True)
    p.add_argument("--no-require-initial-reached", dest="require_initial_reached", action="store_false")

    p.add_argument("--target-class", default=None)
    p.add_argument("--target-index", type=int, default=None)
    p.add_argument("--select-target", action="store_true", help="Show YOLO detections and select the target by mouse instead of auto-picking the first target-class match")
    p.add_argument("--target-prompt", action="store_true", help="Prompt at runtime to type a target class, choose YOLO mouse selection, or click a target pixel")
    p.add_argument("--manual-click-only", action="store_true")
    p.add_argument("--manual-radius", type=float, default=0.045)
    p.add_argument("--yolo-weights", default=DEFAULT_YOLO_WEIGHTS, help="YOLO segmentation weights path; default is ./models/yolo26m-seg.pt")
    p.add_argument("--yolo-classes", default=None, help="Optional comma-separated class prompts for YOLO-World models; ignored by normal YOLO segmentation weights")
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--amodal-weights", default=None, help="Optional amodal YOLO26-seg weights used to estimate target occlusion during clear-obstacle decisions")
    p.add_argument("--amodal-imgsz", type=int, default=None, help="Amodal inference image size; defaults to --yolo-imgsz")
    p.add_argument("--amodal-conf", type=float, default=0.15)
    p.add_argument("--box-depth-band", type=float, default=0.045, help="Depth band in meters used to turn YOLO boxes into target clouds")
    p.add_argument("--min-depth", type=float, default=0.05)
    p.add_argument("--max-depth", type=float, default=1.50)
    p.add_argument("--min-target-points", type=int, default=250)
    p.add_argument("--depth-small-hole-fill", dest="depth_small_hole_fill", action="store_true", help="Fill only tiny enclosed depth dropouts inside the selected visible mask")
    p.add_argument("--no-depth-small-hole-fill", dest="depth_small_hole_fill", action="store_false")
    p.set_defaults(depth_small_hole_fill=True)
    p.add_argument("--depth-hole-max-area", type=int, default=36, help="Maximum enclosed invalid-depth component area eligible for filling")
    p.add_argument("--depth-hole-max-diameter", type=int, default=7, help="Maximum width or height in pixels eligible for filling")
    p.add_argument("--depth-hole-ring-radius", type=int, default=3, help="Neighbor radius used to estimate a tiny hole's depth")
    p.add_argument("--depth-hole-min-neighbors", type=int, default=8)
    p.add_argument("--depth-hole-max-span", type=float, default=0.015, help="Maximum surrounding 10-90%% depth span in meters for safe filling")
    p.add_argument("--voxel-size", type=float, default=0.0015)
    p.add_argument("--scene-voxel-size", type=float, default=0.004)
    p.add_argument("--scene-cloud-cap", type=int, default=60000)
    p.add_argument("--target-outlier-filter", dest="target_outlier_filter", action="store_true", help="Apply one statistical filter to each target view before fusion")
    p.add_argument("--no-target-outlier-filter", dest="target_outlier_filter", action="store_false")
    p.set_defaults(target_outlier_filter=True)
    p.add_argument("--fusion-target-outlier-filter", action="store_true", help="Opt in to a second target-cloud filter after fusion; disabled by default to retain detail")
    p.add_argument("--fusion-scene-outlier-filter", dest="fusion_scene_outlier_filter", action="store_true", help="Filter the fused scene/collision cloud once")
    p.add_argument("--no-fusion-scene-outlier-filter", dest="fusion_scene_outlier_filter", action="store_false")
    p.set_defaults(fusion_scene_outlier_filter=True)
    p.add_argument("--outlier-neighbors", type=int, default=20)
    p.add_argument("--outlier-std-ratio", type=float, default=2.0)
    p.add_argument("--crop-scene-near-target", action="store_true", default=True, help="Crop fused scene cloud to the target neighborhood before GraspNet")
    p.add_argument("--no-crop-scene-near-target", dest="crop_scene_near_target", action="store_false")
    p.add_argument("--scene-target-crop-radius", type=float, default=0.18, help="XY radius around target center kept in fused scene cloud")
    p.add_argument("--scene-target-crop-z-below", type=float, default=0.12, help="Scene crop lower z margin below target center")
    p.add_argument("--scene-target-crop-z-above", type=float, default=0.16, help="Scene crop upper z margin above target center")
    p.add_argument("--scene-target-crop-min-points", type=int, default=500, help="Do not apply target-near scene crop if fewer points would remain")
    p.add_argument("--preview-cloud", action="store_true", help="Show reconstructed scene/target point cloud in an Open3D window before GraspNet")
    p.add_argument("--preview-cloud-frame", action="store_true", help="Add a camera-frame coordinate axis to --preview-cloud")
    p.add_argument("--preview-point-size", type=float, default=1.5, help="Open3D point size in pixels for reconstructed, fused, and GraspNet previews")
    p.add_argument("--active-view", action="store_true", help="Optional equal-radius orbit side observation view before GraspNet")
    p.add_argument("--yes-active-view", action="store_true", help="Skip VIEW confirmation for --active-view")
    p.add_argument("--active-center-first", action="store_true", default=True, help="Move camera above target center before selecting side active views")
    p.add_argument("--no-active-center-first", dest="active_center_first", action="store_false", help="Use the old orbit active-view directly from the first observation pose")
    p.add_argument("--active-center-height", type=float, default=0.30, help="Camera height above target center for the centered active observation")
    p.add_argument("--active-center-roll-samples", type=int, default=72, help="Unique equivalent camera-roll samples generated over the full 360-degree centered look-at range")
    p.add_argument("--active-center-roll-plan-candidates", type=int, default=72, help="Maximum center rolls tested by guarded direct IK/path planning; the default covers the full 360-degree sample set")
    p.add_argument("--active-center-optical-relaxation", dest="active_center_optical_relaxation", action="store_true", help="If the exact downward center orientation has no IK path, keep the camera center fixed and try small optical-axis tilts before moving to a spherical fallback position")
    p.add_argument("--no-active-center-optical-relaxation", dest="active_center_optical_relaxation", action="store_false")
    p.set_defaults(active_center_optical_relaxation=True)
    p.add_argument("--active-center-optical-tilts", default="2,4,6", help="Small optical-axis tilt magnitudes in degrees tried at the unchanged overhead camera position")
    p.add_argument("--active-center-optical-azimuth-step-deg", type=float, default=45.0, help="Azimuth spacing for same-position optical-axis relaxation")
    p.add_argument("--active-center-optical-max-directions", type=int, default=24, help="Maximum same-position optical-axis directions checked before moving the camera position")
    p.add_argument("--active-center-optical-roll-plan-candidates", type=int, default=12, help="Full-circle-spread camera rolls checked per relaxed optical-axis direction")
    p.add_argument("--active-center-spherical-fallback", dest="active_center_spherical_fallback", action="store_true", help="When the exact overhead center view has no verified IK path, try equal-radius tilted look-at observation poses before abandoning active multi-view")
    p.add_argument("--no-active-center-spherical-fallback", dest="active_center_spherical_fallback", action="store_false")
    p.set_defaults(active_center_spherical_fallback=True)
    p.add_argument("--active-center-fallback-tilts", default="15,20,30,35", help="Polar offsets in degrees from the overhead center used only after the exact center view is unreachable")
    p.add_argument("--active-center-fallback-azimuth-step-deg", type=float, default=45.0, help="Azimuth spacing for the equal-radius fallback observation grid")
    p.add_argument("--active-center-fallback-max-positions", type=int, default=16, help="Maximum tilted spherical positions considered after exact center IK/path failure")
    p.add_argument("--active-center-fallback-roll-plan-candidates", type=int, default=12, help="Equivalent camera rolls checked per tilted fallback position")
    p.add_argument("--active-direct-min-joint-limit-margin-deg", type=float, default=1.5, help="Minimum endpoint joint-limit margin for guarded direct active-observation moves; MIT execution automatically raises this to buffer + tracking error + planning slack")
    p.add_argument("--active-direct-return-retreat-height", type=float, default=0.0, help="Diagnostic override for an extra vertical return retreat; default 0 keeps the same direct joint-space edge")
    p.add_argument("--active-direct-return-downward-tolerance", type=float, default=0.002, help="Maximum sampled TCP dip below the side-view start during its direct return")
    p.add_argument("--active-direct-return-runtime-downward-tolerance", type=float, default=0.010, help="Runtime-only total modeled TCP dip allowed below the side-view start during a direct return; planning keeps the stricter --active-direct-return-downward-tolerance floor and point-cloud collision checks remain active")
    p.add_argument("--active-direct-return-min-clearance", type=float, default=0.030, help="Stricter planned scene clearance required before the direct side-view return is allowed")
    p.add_argument("--active-return-execution-waypoint-deg", type=float, default=10.0, help="Maximum per-joint sweep in degrees for each side-view return micro-waypoint; <=0 restores one direct move_j command")
    p.add_argument("--active-fixed-target-center", action="store_true", default=True, help="Use the first-view target center for centered/side active-view planning")
    p.add_argument("--no-active-fixed-target-center", dest="active_fixed_target_center", action="store_false")
    p.add_argument("--active-project-radius", type=float, default=0.08, help="Fallback local-cloud radius around projected target center for active views")
    p.add_argument("--active-strict-identity-margin", type=float, default=0.015, help="Fail strict clear-target tracking when the two nearest same-class 3D instances differ by less than this distance")
    p.add_argument("--active-identity-duplicate-mask-iou", type=float, default=0.95, help="Collapse same-class active-view detections only above this mask IoU")
    p.add_argument("--active-identity-duplicate-center-dist", type=float, default=0.005, help="Also require 3D centers this close before collapsing duplicate detections")
    p.add_argument("--active-identity-duplicate-nested-min-containment", type=float, default=0.95, help="Collapse nested same-class duplicate masks only when this fraction of the smaller mask lies inside the larger mask")
    p.add_argument("--active-identity-duplicate-nested-min-area-ratio", type=float, default=0.50, help="Ignore tiny contained fragments when testing nested same-class duplicate masks")
    p.add_argument("--active-identity-duplicate-nested-max-area-ratio", type=float, default=0.90, help="Require the smaller nested duplicate mask to be at most this fraction of the larger mask, preserving equal-size same-class instances")
    p.add_argument("--active-identity-duplicate-nested-center-dist", type=float, default=0.015, help="Maximum 3D-center separation for nested same-class duplicate-mask collapse")
    p.add_argument("--active-target-reselect", default="yolo", choices=["projected", "yolo", "click"], help="How active views identify the same target after motion; yolo keeps the full segmented target mask")
    p.add_argument("--active-yolo-conf", type=float, default=0.05, help="YOLO confidence used only to re-identify the already selected target in active views")
    p.add_argument("--active-yolo-fallback", default="abort", choices=["abort", "click", "projected"], help="Fallback when YOLO misses the target in active views; abort keeps the original single-view target instead of fusing an unverified local sphere")
    p.add_argument("--active-orbit-max-angle-deg", type=float, default=45.0, help="Maximum equal-radius side-view orbit angle in degrees")
    p.add_argument("--active-side-view-count", type=int, default=2, help="Number of target-centered side views to capture after the centered view; use 2 to match the obstacle-version three-view pattern")
    p.add_argument("--save-active-raw-views", action="store_true", default=True,
                   help="Archive every captured view before registration/fusion, including rejected views")
    p.add_argument("--no-save-active-raw-views", dest="save_active_raw_views", action="store_false")
    p.add_argument("--active-anchor-min-disparity-deg", type=float, default=30.0, help="Reject an active candidate whose target-centered viewing direction differs from the current verified observation anchor by less than this angle")
    p.add_argument("--active-min-side-disparity-deg", type=float, default=45.0, help="Hard minimum angular separation when selecting multiple active side views")
    p.add_argument("--active-prefer-opposite-sides", action="store_true", default=True, help="Prefer left/right separated side views instead of two nearby high-score views")
    p.add_argument("--no-active-prefer-opposite-sides", dest="active_prefer_opposite_sides", action="store_false")
    p.add_argument("--active-preferred-side-separation-deg", type=float, default=90.0, help="Preferred separation between the first and second side views")
    p.add_argument("--active-nbv-mode", default="scene_aware", choices=["scene_aware", "fixed"], help="scene_aware scores orbit directions with obstacle-version NBV terms; fixed uses the old +/- orbit angles")
    p.add_argument("--active-nbv-candidate-num", type=int, default=16, help="Legacy fallback for horizontal direction count when no azimuth step is configured")
    p.add_argument("--active-nbv-azimuth-step-deg", type=float, default=22.5, help="Full-circle spherical NBV azimuth spacing; 22.5 degrees produces 16 directions")
    p.add_argument("--active-nbv-orbit-degs", default="20,30,35,45", help="Comma/space-separated polar tilts from +Z; values above --active-orbit-max-angle-deg are dropped instead of clamped")
    p.add_argument("--active-nbv-plan-candidate-num", type=int, default=32, help="Minimum number of information-ranked spherical positions sent through bounded roll/IK checking before early stopping")
    p.add_argument("--active-nbv-fast-plan-candidates", type=int, default=8, help="First-pass spherical position budget; the full NBV budget is used only when this fast pass finds no reachable view")
    p.add_argument("--active-nbv-fast-roll-candidates", type=int, default=2, help="Equivalent camera rolls checked per spherical position during the first-pass NBV search")
    p.add_argument("--active-nbv-fast-ik-seed-limit", type=int, default=2, help="IK seeds checked per camera roll during the first-pass NBV search")
    p.add_argument("--active-nbv-min-valid-candidates", type=int, default=8, help="Continue beyond the initial spherical shortlist until this many endpoint-valid viewpoints exist or the full grid is exhausted")
    p.add_argument("--active-nbv-shortlist-per-tilt", type=int, default=3, help="Front-load at least this many high-information azimuths from every configured tilt layer")
    p.add_argument("--active-side-roll-samples", type=int, default=16, help="Equivalent full-circle camera rolls generated for each shortlisted spherical viewpoint")
    p.add_argument("--active-side-roll-plan-candidates", type=int, default=8, help="Full-circle-spread roll candidates tested with IK for each shortlisted viewpoint")
    p.add_argument("--active-side-roll-ik-seed-limit", type=int, default=8, help="Fast-path TRAC-IK seed count per side-view roll; the prefix interleaves local and full-seven-joint seeds")
    p.add_argument("--active-side-roll-full-retry-candidates", type=int, default=3, help="If every fast roll solve fails, retry this many least-rotation rolls with the full legacy seed set")
    p.add_argument("--active-side-roll-keep-candidates", type=int, default=2, help="Keep this many endpoint-valid rolls per spherical viewpoint so the second roll can serve as a path-planning fallback")
    p.add_argument("--active-side-max-capture-attempts", type=int, default=3, help="Maximum outbound side-view motion/capture attempts used to obtain the requested accepted side views")
    p.add_argument("--active-nbv-surface-bin-count", type=int, default=512, help="Upper/side Fibonacci envelope directions used to estimate newly observable target surface")
    p.add_argument("--active-nbv-surface-min-normal-z", type=float, default=-0.20, help="Lowest envelope-normal z retained by the active-view surface proxy; the supported underside remains excluded")
    p.add_argument("--active-nbv-surface-coverage-sigma-deg", type=float, default=12.0, help="Angular kernel used to mark surface-envelope directions already covered")
    p.add_argument("--active-nbv-contact-sigma-deg", type=float, default=25.0, help="Angular width of PCA parallel-jaw contact-region proxies")
    p.add_argument("--active-nbv-grasp-width-margin", type=float, default=0.002, help="Margin subtracted from the configured gripper opening when accepting a PCA contact-axis proxy")
    p.add_argument("--active-nbv-surface-weight", type=float, default=0.35)
    p.add_argument("--active-nbv-grasp-contact-weight", type=float, default=0.30)
    p.add_argument("--active-nbv-sweep-weight", type=float, default=0.15)
    p.add_argument("--active-nbv-occlusion-weight", type=float, default=0.10)
    p.add_argument("--active-nbv-image-weight", type=float, default=0.10)
    p.add_argument("--active-nbv-redundancy-weight", type=float, default=0.10)
    p.add_argument("--active-nbv-motion-weight", type=float, default=0.10)
    p.add_argument("--active-nbv-score-weight", type=float, default=1.0, help="Optional global multiplier for the unified active-view information score")
    p.add_argument("--active-nbv-debug", action="store_true", help="Print detailed scene-aware NBV direction scores")
    p.add_argument("--active-multiview-gate", dest="active_multiview_gate", action="store_true", help="Use active multi-view only when non-target obstacle points are close to the segmented target cloud")
    p.add_argument("--no-active-multiview-gate", dest="active_multiview_gate", action="store_false", help="Always run active multi-view when --active-view is set")
    p.set_defaults(active_multiview_gate=True)
    p.add_argument("--active-multiview-obstacle-distance", type=float, default=0.055, help="Run active multi-view when non-target obstacle points are within this distance of the target cloud")
    p.add_argument("--active-multiview-min-obstacle-points", type=int, default=80, help="Minimum nearby non-target obstacle points required to trigger active multi-view")
    p.add_argument("--active-orbit-min-radius", type=float, default=0.30, help="Minimum camera-target radius for active side-view spherical orbit")
    p.add_argument("--active-orbit-lift", type=float, default=0.0, help="Optional upward camera-position lift for orbit active views in meters")
    p.add_argument("--active-orbit-orientation", default="limited_lookat", choices=["current", "limited_lookat", "lookat"], help="limited_lookat partially rotates toward target; current keeps flange RPY; lookat fully points camera at target")
    p.add_argument("--active-orbit-max-rpy-delta-deg", type=float, default=12.0, help="Maximum flange orientation change for --active-orbit-orientation limited_lookat")
    p.add_argument("--active-view-max-center-norm", type=float, default=0.75, help="Reject active views where target center is too far from camera optical axis")
    p.add_argument("--active-obstacle-aware", dest="active_obstacle_aware", action="store_true", help="Enable obstacle/occlusion-aware scoring and rejection for active-view candidates")
    p.add_argument("--no-active-obstacle-aware", dest="active_obstacle_aware", action="store_false", help="Disable obstacle/occlusion-aware active-view scoring for ablation")
    p.set_defaults(active_obstacle_aware=True)
    p.add_argument("--active-view-min-clearance", type=float, default=0.055, help="Minimum camera-to-obstacle clearance for active-view candidates")
    p.add_argument("--active-direction-collision-weight", type=float, default=0.08, help="Small extra penalty for active-view directions containing nearby scene obstacles; NBV/visibility remains primary")
    p.add_argument("--active-los-corridor-radius", type=float, default=0.045, help="Obstacle corridor radius from active camera to target center")
    p.add_argument("--active-traj-corridor-radius", type=float, default=0.055, help="Obstacle corridor radius along active camera motion path")
    p.add_argument("--active-max-los-penalty", type=float, default=0.92, help="Reject active views with line-of-sight obstacle penalty above this value")
    p.add_argument("--active-max-traj-penalty", type=float, default=0.92, help="Reject active views with trajectory obstacle penalty above this value")
    p.add_argument("--active-path-guard", dest="active_path_guard", action="store_true", help="Fail closed unless the sampled full-arm joint path clears the complete captured scene")
    p.add_argument("--no-active-path-guard", dest="active_path_guard", action="store_false", help="Diagnostic only: disable full-arm active-view path validation")
    p.set_defaults(active_path_guard=True)
    p.add_argument("--active-up-and-over", dest="active_up_and_over", action="store_true", help="Solve active center/side observations through sequential raise-orient-traverse-descend TRAC-IK waypoints")
    p.add_argument("--no-active-up-and-over", dest="active_up_and_over", action="store_false", help="Use one guarded direct IK-joint edge per active observation move; full-scene preflight and runtime collision monitoring remain enabled")
    p.set_defaults(active_up_and_over=False)
    p.add_argument("--active-confirm-each-view", dest="active_confirm_each_view", action="store_true", help="Require Enter/no confirmation before every verified side-view motion")
    p.add_argument("--no-active-confirm-each-view", dest="active_confirm_each_view", action="store_false")
    p.set_defaults(active_confirm_each_view=True)
    p.add_argument("--active-center-max-joint-delta-deg", type=float, default=45.0, help="Maximum single-joint sweep from the initial pose to active center")
    p.add_argument("--active-center-max-joint-l2-deg", type=float, default=90.0, help="Maximum L2 joint sweep from the initial pose to active center")
    p.add_argument("--active-side-max-joint-delta-deg", type=float, default=45.0, help="Maximum single-joint sweep from saved center to any side view")
    p.add_argument("--active-side-max-joint-l2-deg", type=float, default=75.0, help="Maximum L2 joint sweep from saved center to any side view")
    p.add_argument("--active-center-joint-match-deg", type=float, default=2.0, help="Require every side-view motion to start this close to the saved center joints")
    p.add_argument("--active-path-sample-joint-deg", type=float, default=2.0, help="Joint interpolation step used by the full-arm sweep check")
    p.add_argument("--active-path-scene-voxel", type=float, default=0.005, help="Voxel used only to bound full-scene path-check cost")
    p.add_argument("--active-path-obstacle-cap", type=int, default=50000, help="Maximum complete-scene points retained by the full-arm path check")
    p.add_argument("--active-path-link-radius", type=float, default=0.060, help="Conservative Nero arm-link capsule radius before extra clearance")
    p.add_argument("--active-path-tool-radius", type=float, default=0.090, help="Open-gripper capsule radius before extra clearance")
    p.add_argument("--active-path-camera-radius", type=float, default=0.055, help="Camera-body radius before extra active-path clearance; the body is modeled behind the optical centre")
    p.add_argument("--camera-collision-body-rear-extent", type=float, default=0.120, help="Camera-body depth behind the optical centre; it is conservatively clamped to at least twice the configured body radius")
    p.add_argument("--camera-collision-mount-radius", type=float, default=0.025, help="Radius of the thinner flange-to-camera support capsule")
    p.add_argument("--camera-collision-mount-attachment-depth", type=float, default=0.060, help="Support attachment depth behind the camera optical centre")
    p.add_argument("--active-path-min-clearance", type=float, default=0.015, help="Extra clearance required outside every active-path capsule")
    p.add_argument("--active-path-require-scene", dest="active_path_require_scene", action="store_true", help="Reject active motion if no complete scene cloud is available")
    p.add_argument("--no-active-path-require-scene", dest="active_path_require_scene", action="store_false")
    p.set_defaults(active_path_require_scene=True)
    p.add_argument("--active-path-require-camera-model", dest="active_path_require_camera_model", action="store_true", help="Reject active motion if the camera-mount capsule cannot be constructed")
    p.add_argument("--no-active-path-require-camera-model", dest="active_path_require_camera_model", action="store_false")
    p.set_defaults(active_path_require_camera_model=True)
    p.add_argument("--active-obstacle-cap", type=int, default=20000, help="Maximum obstacle points sampled for active-view obstacle scoring")
    p.add_argument("--active-opposite-bonus-weight", type=float, default=0.08, help="Small angular-separation bonus after the second view's conditional information gain is recomputed")
    p.add_argument("--active-target-cloud-exclusion-dist", type=float, default=0.012, help="Treat scene points within this distance of the segmented target cloud as target, not obstacles")
    p.add_argument("--active-target-exclusion-radius", type=float, default=0.075, help="Fallback center-radius target exclusion when no target cloud is available")
    p.add_argument("--active-obstacle-xy-radius", type=float, default=0.40, help="Use scene obstacle points within this XY radius around the target for active-view scoring")
    p.add_argument("--active-obstacle-min-height-margin", type=float, default=-0.005, help="Keep obstacle points above target_z_low plus this margin for active-view scoring")
    p.add_argument("--active-fusion-voxel", type=float, default=0.0025, help="Fine voxel size for confidence-weighted fused target cloud")
    p.add_argument("--active-target-fusion-mode", choices=["voxel", "surfel"], default="voxel", help="voxel=same-cell weighted mean; surfel=merge compatible surfaces across a small radius")
    p.add_argument("--active-surfel-merge-radius", type=float, default=0.008, help="Maximum metric radius for merging compatible surfel samples")
    p.add_argument("--active-surfel-normal-angle-deg", type=float, default=55.0, help="Maximum normal angle for merging surfels")
    p.add_argument("--active-surfel-normal-neighbors", type=int, default=12, help="Neighbors used to estimate surfel normals")
    p.add_argument("--active-round-table-registration", dest="active_round_table_registration", action="store_true", help="For round targets, align views using tabletop Z/roll/pitch instead of target ICP")
    p.add_argument("--no-active-round-table-registration", dest="active_round_table_registration", action="store_false")
    p.set_defaults(active_round_table_registration=True)
    p.add_argument("--active-round-static-xyyaw-registration", dest="active_round_static_xyyaw_registration", action="store_true")
    p.add_argument("--no-active-round-static-xyyaw-registration", dest="active_round_static_xyyaw_registration", action="store_false")
    p.set_defaults(active_round_static_xyyaw_registration=True)
    p.add_argument("--active-scene-table-registration", dest="active_scene_table_registration", action="store_true", help="For non-round targets, first align the shared table before local target ICP")
    p.add_argument("--no-active-scene-table-registration", dest="active_scene_table_registration", action="store_false")
    p.set_defaults(active_scene_table_registration=True)
    p.add_argument("--active-round-view-policy", dest="active_round_view_policy", action="store_true", help="Select side-view pairs by union coverage and angular separation for round targets")
    p.add_argument("--no-active-round-view-policy", dest="active_round_view_policy", action="store_false")
    p.set_defaults(active_round_view_policy=True)
    p.add_argument("--active-round-pair-min-separation-deg", type=float, default=70.0)
    p.add_argument("--active-round-pair-weight", type=float, default=0.35)
    p.add_argument("--active-elongated-bilateral-weight", type=float, default=0.25)
    p.add_argument("--active-capture-settle-seconds", type=float, default=0.0, help="Wait after an active-view move before sampling pose and RGB-D")
    p.add_argument("--active-table-reg-z-window", type=float, default=0.040)
    p.add_argument("--active-table-reg-inlier", type=float, default=0.003)
    p.add_argument("--active-table-reg-min-points", type=int, default=250)
    p.add_argument("--active-table-reg-max-tilt-deg", type=float, default=5.0)
    p.add_argument("--active-table-reg-max-height-shift", type=float, default=0.015)
    p.add_argument("--active-table-reg-max-residual-tilt-deg", type=float, default=0.5, help="Maximum remaining tabletop normal mismatch after registration")
    p.add_argument("--active-static-reg-table-band", type=float, default=0.008)
    p.add_argument("--active-static-reg-target-padding", type=float, default=0.035)
    p.add_argument("--active-static-reg-voxel", type=float, default=0.008)
    p.add_argument("--active-static-reg-min-pairs", type=int, default=120)
    p.add_argument("--active-static-reg-max-xy", type=float, default=0.005)
    p.add_argument("--active-static-reg-max-yaw-deg", type=float, default=1.0)
    p.add_argument("--active-static-reg-correspondence", type=float, default=0.025)
    p.add_argument("--active-round-fusion-max-median", type=float, default=0.015)
    p.add_argument("--active-round-fusion-max-p95", type=float, default=0.040)
    p.add_argument("--active-round-fusion-min-inlier-1cm", type=float, default=0.20)
    p.add_argument("--active-round-fusion-max-center-delta", type=float, default=0.040)
    p.add_argument("--target-fusion-single-view-radius", type=float, default=0.008, help="Keep a single-view target voxel only when this-radius target structure supports it")
    p.add_argument("--target-fusion-single-view-min-neighbors", type=int, default=2, help="Minimum nearby fused target voxels required to reject neither a single-view target surface nor its boundary")
    p.add_argument("--active-fusion-diagnostics", action="store_true", default=True, help="Save per-view transformed target clouds and print center offsets before fusion")
    p.add_argument("--no-active-fusion-diagnostics", dest="active_fusion_diagnostics", action="store_false")
    p.add_argument("--active-fusion-warn-delta", type=float, default=0.015, help="Warn when per-view target centers differ by more than this value in base frame")
    p.add_argument("--active-fusion-reject-bad-views", action="store_true", default=True, help="Exclude active views whose transformed target center differs too much from the first view")
    p.add_argument("--no-active-fusion-reject-bad-views", dest="active_fusion_reject_bad_views", action="store_false")
    p.add_argument("--active-fusion-reject-delta", type=float, default=0.015, help="Reject active-view target clouds whose base-frame median differs from first view by more than this value")
    p.add_argument("--active-fusion-overlap-override-delta", type=float, default=0.025, help="Deprecated compatibility option: center shift is diagnostic only when registered overlap quality passes")
    p.add_argument("--active-fusion-overlap-max-median", type=float, default=0.008, help="Maximum median residual on mutual overlap correspondences")
    p.add_argument("--active-fusion-overlap-max-p95", type=float, default=0.020, help="Maximum p95 residual on mutual overlap correspondences")
    p.add_argument("--active-fusion-overlap-min-inlier-1cm", type=float, default=0.80, help="Deprecated compatibility option; fusion acceptance now uses mutual overlap pairs and ratio")
    p.add_argument("--active-fusion-reject-axis-mismatch", action="store_true", help="Reject active-view target clouds whose dominant PCA axis differs too much from the first view")
    p.add_argument("--active-fusion-reject-axis-deg", type=float, default=35.0, help="PCA axis angle threshold for --active-fusion-reject-axis-mismatch")
    p.add_argument("--active-fusion-register", action="store_true", default=True, help="Register active-view target clouds before fusion using obstacle-version overlap logic")
    p.add_argument("--no-active-fusion-register", dest="active_fusion_register", action="store_false")
    p.add_argument("--active-fusion-translation-calib", action="store_true", default=True, help="Apply conservative target-cloud translation calibration before active-view fusion")
    p.add_argument("--no-active-fusion-translation-calib", dest="active_fusion_translation_calib", action="store_false")
    p.add_argument("--active-fusion-rigid-icp", action="store_true", help="Optionally apply conservative rigid ICP after translation calibration")
    p.add_argument("--active-reg-icp-estimation", choices=["point_to_point", "point_to_plane"], default="point_to_point", help="Conservative robust ICP estimator; point-to-point is safer for partially visible elongated food objects")
    p.add_argument("--active-reg-icp-robust-kernel", type=float, default=0.005, help="Huber loss scale in metres for conservative ICP; <=0 disables robust weighting")
    p.add_argument("--active-reg-apply-to-scene", action="store_true", default=True, help="Apply accepted view registration transform to scene cloud as well as target cloud")
    p.add_argument("--no-active-reg-apply-to-scene", dest="active_reg_apply_to_scene", action="store_false")
    p.add_argument("--active-reg-voxel", type=float, default=0.0025, help="Voxel size for active-view registration diagnostics")
    p.add_argument("--active-reg-nn-thr", type=float, default=0.045, help="Coarse NN gate for active-view target translation calibration")
    p.add_argument("--active-reg-max-shift", type=float, default=0.030, help="Maximum accepted target translation calibration shift in meters")
    p.add_argument("--active-reg-max-shift-xy", type=float, default=0.025, help="Maximum accepted XY translation calibration shift in meters; <=0 disables per-axis XY clipping")
    p.add_argument("--active-reg-max-shift-z", type=float, default=0.015, help="Maximum accepted Z translation calibration shift in meters; <=0 disables Z clipping")
    p.add_argument("--active-reg-trim-quantile", type=float, default=0.90, help="Keep this nearest-neighbor distance quantile before estimating translation; 1.0 disables trimming")
    p.add_argument("--active-reg-translation-iters", type=int, default=1, help="Small translation-only registration iterations; no rotation is estimated")
    p.add_argument("--active-reg-min-pairs", type=int, default=60, help="Minimum NN overlap pairs for active-view registration")
    p.add_argument("--active-reg-min-improve", type=float, default=0.04, help="Minimum median/p95 improvement ratio for accepting registration")
    p.add_argument("--active-reg-max-median", type=float, default=0.008, help="Reject registered view if target NN median remains above this")
    p.add_argument("--active-reg-max-p95", type=float, default=0.020, help="Reject registered view if target NN p95 remains above this")
    p.add_argument("--active-reg-min-inlier-1cm", type=float, default=0.70, help="Reject registered view if inlier@1cm remains below this")
    p.add_argument("--active-reg-icp-overlap-thr", type=float, default=0.018, help="NN gate used to select overlap subset for optional rigid ICP")
    p.add_argument("--active-reg-icp-min-overlap-ratio", type=float, default=0.08)
    p.add_argument("--active-reg-icp-voxel", type=float, default=0.0025, help="Downsampling inside enabled ICP; 0 keeps all registration points. Enable ICP with --active-fusion-rigid-icp")
    p.add_argument("--active-reg-icp-thr", type=float, default=0.018)
    p.add_argument("--active-reg-icp-thresholds", default="0.015,0.008", help="Comma-separated coarse-to-fine ICP correspondence thresholds")
    p.add_argument("--active-reg-icp-max-iter", type=int, default=25)
    p.add_argument("--active-reg-icp-max-trans", type=float, default=0.010)
    p.add_argument("--active-reg-icp-max-rot-deg", type=float, default=3.0)
    p.add_argument("--active-reg-icp-min-fit", type=float, default=0.10)
    p.add_argument("--active-restore-center", action="store_true", default=True, help="Return to first observation TCP pose after active-view capture")
    p.add_argument("--no-active-restore-center", dest="active_restore_center", action="store_false")
    p.add_argument("--preview-active-clouds", action="store_true", default=True, help="Show fused active-view point cloud before GraspNet")
    p.add_argument("--no-preview-active-clouds", dest="preview_active_clouds", action="store_false")

    p.add_argument("--enable-clear-obstacle", action="store_true", help="Before active view / target grasp, fuse visible/amodal masks with depth geometry to clear a likely front occluder")
    p.add_argument("--clear-max-rounds", type=int, default=1)
    p.add_argument(
        "--clear-empty-grasp-retries",
        type=int,
        default=1,
        help=(
            "Extra clear attempts after a confirmed empty grasp has safely returned "
            "to the first observation pose; each retry recaptures and strictly "
            "re-identifies the originally selected target"
        ),
    )
    p.add_argument("--clear-score-threshold", type=float, default=0.35)
    p.add_argument("--clear-depth-margin", type=float, default=0.02, help="Object must be this much closer than target depth to count as front occluder")
    p.add_argument("--clear-front-depth-full-score", type=float, default=0.08)
    p.add_argument("--clear-target-bbox-expand", type=float, default=35.0)
    p.add_argument("--clear-corridor-radius", type=float, default=0.045)
    p.add_argument("--clear-min-points", type=int, default=180)
    p.add_argument("--clear-skip-target-iou", type=float, default=0.35)
    p.add_argument("--clear-overlap-weight", type=float, default=0.35)
    p.add_argument("--clear-front-weight", type=float, default=0.30)
    p.add_argument("--clear-center-weight", type=float, default=0.15)
    p.add_argument("--clear-corridor-weight", type=float, default=0.20)
    p.add_argument("--clear-amodal-weight", type=float, default=0.40, help="Amodal evidence weight when fusing target hidden-area overlap with geometry")
    p.add_argument("--clear-amodal-occlusion-threshold", type=float, default=0.25, help="Target occlusion ratio above which amodal clear-first evidence becomes actionable")
    p.add_argument("--clear-amodal-min-hidden-overlap", type=float, default=0.05, help="Minimum candidate/target-hidden overlap ratio for amodal clear-first evidence")
    p.add_argument("--clear-largest-visible-fallback", action="store_true", help="If the requested target is missing or highly occluded, clear the largest visible valid-depth YOLO instance first")
    p.add_argument("--clear-largest-fallback-occlusion", type=float, default=0.70, help="Amodal occlusion ratio that activates --clear-largest-visible-fallback for a detected target")
    p.add_argument("--post-clear-reidentify-min-coverage", type=float, default=0.30, help="Minimum fraction of the pre-clear target mask covered by a same-view post-clear detection")
    p.add_argument("--post-clear-reidentify-min-iou", type=float, default=0.10, help="Minimum mask IoU for automatic same-view post-clear target reassociation")
    p.add_argument("--post-clear-reidentify-ambiguity-margin", type=float, default=0.15, help="Minimum reference-coverage lead over the second post-clear same-class detection")
    p.add_argument("--clear-place-xyz", type=float, nargs=3, default=[-0.35, -0.15, 0.20])
    p.add_argument("--yes-clear", action="store_true", help="Skip CLEAR confirmation for the clear-obstacle stage only")

    p.add_argument("--grasp-num-points", type=int, default=3000)
    p.add_argument("--grasp-generator", choices=["graspnet", "graspgenx"], default="graspnet", help="Candidate generator; both use the same measured-scene safety workflow")
    p.add_argument("--nero-grasp-refine", action="store_true", help="If native routes all fail, search Nero 100 mm pose/depth variants and qualify the requested hold route before selection")
    p.add_argument("--nero-refine-max-route-variants", type=int, default=8)
    p.add_argument("--nero-refine-timeout-s", type=float, default=60.)
    p.add_argument("--graspgenx-python", default=str(PROJECT_ROOT / "third_party/GraspGenX/.venv/bin/python"))
    p.add_argument("--graspgenx-timeout", type=float, default=180.0)
    p.add_argument("--graspgenx-seed", type=int, default=42)
    p.add_argument("--graspgenx-num-grasps", type=int, default=64)
    p.add_argument("--graspgenx-topk-per-pass", type=int, default=20)
    p.add_argument("--graspgenx-table-margin", type=float, default=0.005)
    p.add_argument("--graspgenx-contact-advance", type=lambda value: 'auto' if value == 'auto' else float(value), default=0.0,
                   help="Native GenX TCP adjustment in metres, or auto: choose the smallest "
                        "0..15 mm advance meeting measured contact and CAD/table constraints")
    p.add_argument("--mgpc", dest="mgpc", action="store_true", help="Use MGPC completion for GraspNet proposals only; keep measured occupancy/aperture/path geometry")
    p.add_argument("--no-mgpc", dest="mgpc", action="store_false")
    p.set_defaults(mgpc=False)
    p.add_argument("--mgpc-category", default=None, help="MGPC shape prompt independent of YOLO labels, e.g. cube")
    p.add_argument("--mgpc-python", default=str(PROJECT_ROOT / ".venv_mgpc/bin/python"), help="Dedicated offline MGPC worker interpreter")
    p.add_argument("--mgpc-checkpoint", default=str(PROJECT_ROOT / "third_party/MGPC/pretrained/ckpt_8192.pt"))
    p.add_argument("--mgpc-timeout", type=float, default=180.0)
    p.add_argument("--mgpc-seed", type=int, default=0)
    p.add_argument("--active-grasp-feedback", action="store_true", help="Opt in to measured-first completion and grasp-conditioned NBV; require one valid side and at most two attempts")
    p.add_argument("--feedback-max-attempts", type=int, choices=[1, 2], default=2)
    p.add_argument("--feedback-candidate-limit", type=int, default=12)
    p.add_argument("--feedback-probe-preflight-max-candidates", type=int, default=6,
                   help="Observation planning-only probes: cap expensive route checks; final grasp keeps its normal budget")
    p.add_argument("--feedback-probe-refine-timeout-s", type=float, default=15.,
                   help="Observation probes only: Nero refinement budget, checked between route variants")
    p.add_argument("--feedback-support-distance", type=float, default=.006)
    p.add_argument("--feedback-duplicate-distance", type=float, default=.003)
    p.add_argument("--feedback-prediction-ratio", type=float, default=.5)
    p.add_argument("--feedback-contact-support", type=float, default=.75)
    p.add_argument("--feedback-view-weight", type=float, default=.6)
    p.add_argument("--feedback-timeout", type=float, default=600., help="Total feedback wall-clock budget, checked between bounded model/planning calls")
    p.add_argument("--grasp-required-view-count", type=int, choices=[0, 1, 2, 3], default=0, help="For controlled comparisons, require this many retained target views from a verified active center; 0 keeps the existing fallback policy")
    p.add_argument("--grasp-angle-deg", type=float, default=85.0)
    p.add_argument(
        "--graspnet-collision-approach-dist",
        type=float,
        default=0.04,
        help=(
            "Approach-corridor length used only by the early GraspNet voxel "
            "collision filter. This does not change --pregrasp-offset or the "
            "later full robot trajectory collision checks"
        ),
    )
    p.add_argument("--manual-grasp-select", action="store_true")
    p.add_argument(
        "--manual-topk",
        type=int,
        default=24,
        help=(
            "Number of score-ranked GraspNet candidates retained for physical-"
            "aperture, endpoint-IK, and complete-workflow preflight"
        ),
    )
    p.add_argument("--select-grasp-candidate", action="store_true", help="Show/save top-k grasp candidates and choose one by clicking or typing an index")
    p.add_argument("--grasp-candidate-index", type=int, default=None, help="Select a grasp candidate index non-interactively")
    p.add_argument(
        "--recommended-grasp-finger-table-margin",
        type=float,
        default=0.003,
        help=(
            "Preferred minimum full open-finger surface margin above table_z "
            "when marking the recommended fully prevalidated candidate; this "
            "is a recommendation ranking rule, not a new hard planning gate"
        ),
    )
    p.add_argument(
        "--recommended-grasp-min-cross-axis-score",
        type=float,
        default=0.45,
        help=(
            "For elongated targets, require the jaw closing axis to cross "
            "rather than follow the target principal axis (1=perpendicular)"
        ),
    )
    p.add_argument(
        "--recommended-grasp-min-raw-score",
        type=float,
        default=0.10,
        help="Minimum GraspNet score for a candidate to be labelled RECOMMENDED",
    )
    p.add_argument(
        "--recommended-grasp-min-contact-points",
        type=int,
        default=80,
        help="Minimum target support in the closing slab for a RECOMMENDED label",
    )
    p.add_argument(
        "--recommended-grasp-contact-band",
        type=float,
        default=0.006,
        help=(
            "Width in metres of the target surface band used to measure "
            "left/right closing support. This changes ranking only."
        ),
    )
    p.add_argument(
        "--recommended-grasp-contact-side-points",
        type=int,
        default=25,
        help=(
            "Per-side contact points at which soft bilateral support reaches "
            "full point-count credit. This is not a hard rejection threshold."
        ),
    )
    p.add_argument("--print-grasp-candidates", action="store_true", help="Print full top-k GraspNet candidate list to terminal")
    p.add_argument("--visualize-grasps", action="store_true")
    p.add_argument("--allow-grasp-collision-fallback", action="store_true", help="Diagnostic only: restore pre-collision GraspNet candidates if every strict/relaxed candidate collides")
    p.add_argument("--allow-grasp-angle-fallback", action="store_true", help="Diagnostic only: relax the world-down approach-angle constraint when no candidate passes it")
    p.add_argument("--grasp-target-empty-thresh", type=float, default=0.01, help="Reject candidates whose closing volume contains too few segmented target points; <=0 disables this filter")
    p.add_argument("--grasp-target-empty-voxel-size", type=float, default=0.005, help="Voxel size used by the target-only empty-grasp check")
    p.add_argument(
        "--graspnet-main-component-filter",
        dest="graspnet_main_component_filter",
        action="store_true",
        help="When one connected target component dominates, send only that component to GraspNet",
    )
    p.add_argument(
        "--no-graspnet-main-component-filter",
        dest="graspnet_main_component_filter",
        action="store_false",
    )
    p.set_defaults(graspnet_main_component_filter=True)
    p.add_argument("--graspnet-component-eps", type=float, default=0.004, help="3D connectivity radius used only to clean the GraspNet target input")
    p.add_argument("--graspnet-component-min-points", type=int, default=8, help="DBSCAN core support used by the GraspNet target-component filter")
    p.add_argument("--graspnet-component-min-dominance", type=float, default=0.85, help="Apply largest-component-only GraspNet input when it contains at least this fraction of target points")
    p.add_argument("--graspnet-upper-focus", dest="graspnet_upper_focus", action="store_true", help="Crop table-adjacent bulk from the GraspNet-only target cloud")
    p.add_argument("--no-graspnet-upper-focus", dest="graspnet_upper_focus", action="store_false")
    p.set_defaults(graspnet_upper_focus=True)
    p.add_argument("--graspnet-lower-target-quantile", type=float, default=0.25)
    p.add_argument("--graspnet-upright-lower-target-quantile", type=float, default=0.15)
    p.add_argument("--graspnet-focus-elongation-threshold", type=float, default=1.5)
    p.add_argument("--graspnet-focus-min-retained-ratio", type=float, default=0.50)
    p.add_argument("--graspnet-focus-min-retained-points", type=int, default=1000)
    p.add_argument("--preview-graspnet-cloud", action="store_true", help="Show GraspNet candidates on the point cloud with approach and gripper width")
    p.add_argument("--no-preview-graspnet-cloud", dest="preview_graspnet_cloud", action="store_false")
    p.add_argument("--preview-final-grasp", dest="preview_final_grasp", action="store_true")
    p.add_argument("--no-preview-final-grasp", dest="preview_final_grasp", action="store_false")
    p.set_defaults(preview_final_grasp=True)
    p.add_argument(
        "--max-grasp-width",
        type=float,
        default=0.110,
        help=(
            "Maximum decoded GraspNet candidate width accepted as model "
            "tolerance; this does not change the physical gripper opening, "
            "and width-saturated candidates still require unambiguous physical-"
            "aperture validation"
        ),
    )
    p.add_argument("--graspnet-width-inflation", type=float, default=1.2, help="Width multiplier applied by GraspNet pred_decode; used only to recover the nominal predicted width for physical-aperture validation")
    p.add_argument("--graspnet-decode-max-width", type=float, default=0.110, help="GraspNet decoded-width saturation point; saturated candidates are rejected because their nominal width is ambiguous")
    p.add_argument(
        "--grasp-physical-aperture-margin",
        type=float,
        default=0.0005,
        help=(
            "Per-finger inward margin subtracted from the confirmed physical "
            "opening during target-fit validation; the default leaves an "
            "99mm effective aperture for a 100mm configured opening"
        ),
    )
    p.add_argument("--grasp-physical-aperture-quantile", type=float, default=0.01, help="Robust lower/upper target-cloud quantile used to validate fit between the physical fingers")
    p.add_argument("--grasp-physical-aperture-min-points", type=int, default=50, help="Minimum segmented target points required in a candidate closing slab")
    p.add_argument("--grasp-physical-recenter-max", type=float, default=0.020, help="Maximum automatic TCP correction along the final gripper width axis; corrected poses are rechecked by IK and full-scene path preflight")
    p.add_argument("--grasp-physical-max-finger-penetration", type=float, default=0.001, help="Absolute hard limit for any target point inside an open-finger capsule; complements the robust aperture quantile")
    p.add_argument(
        "--grasp-physical-min-contact-inset", type=float, default=0.0,
        help=("Require the median of each measured left/right contact band to sit "
              "this far behind the straight finger front endpoint, excluding capsule "
              "endcaps; 0 disables this depth check; safe launcher uses 10mm"),
    )
    p.add_argument("--max-candidate-pre-pos-delta", type=float, default=0.25, help="Do not recommend grasp candidates whose pregrasp TCP is farther than this from current TCP")
    p.add_argument("--max-candidate-pre-rpy-delta-deg", type=float, default=60.0, help="Fallback non-IK precheck: do not recommend candidates whose full RPY delta exceeds this angle")
    p.add_argument("--max-candidate-ik-joint-delta", type=float, default=np.deg2rad(100.0), help="IK endpoint precheck limit in radians; default matches the 100 degree complete-path limit used by the real safety wrapper")
    p.add_argument("--grasp-plan-start-joint-match-deg", type=float, default=2.0, help="Replan instead of executing if joints changed after full-sequence grasp IK precheck")
    p.add_argument("--grasp-motion-path-guard", dest="grasp_motion_path_guard", action="store_true", help="Plan every grasp/lift/transport/place/recovery edge with full-arm point-cloud collision checks")
    p.add_argument("--no-grasp-motion-path-guard", dest="grasp_motion_path_guard", action="store_false", help="Diagnostic only: restore the legacy endpoint-only grasp execution")
    p.set_defaults(grasp_motion_path_guard=True)
    p.add_argument("--motion-runtime-collision-guard", dest="motion_runtime_collision_guard", action="store_true", help="Recheck the frozen safety cloud from measured joints while each validated move_j is executing")
    p.add_argument("--no-motion-runtime-collision-guard", dest="motion_runtime_collision_guard", action="store_false")
    p.set_defaults(motion_runtime_collision_guard=True)
    p.add_argument(
        "--motion-runtime-torque-guard",
        dest="motion_runtime_torque_guard",
        action="store_true",
        help=(
            "Monitor Nero measured joint torque during every frozen motion "
            "stage; a sustained excursion stops normally and retracts along "
            "the verified incoming edge"
        ),
    )
    p.add_argument(
        "--no-motion-runtime-torque-guard",
        dest="motion_runtime_torque_guard",
        action="store_false",
    )
    p.set_defaults(motion_runtime_torque_guard=True)
    p.add_argument(
        "--motion-runtime-torque-delta",
        type=float,
        default=8.0,
        help=(
            "Per-joint N.m change from the stage-start torque baseline that "
            "counts as a possible external-contact event"
        ),
    )
    p.add_argument(
        "--motion-runtime-torque-confirm-samples",
        type=int,
        default=2,
        help=(
            "Consecutive torque samples above the excursion threshold "
            "required before stopping (prevents single-frame false alarms)"
        ),
    )
    p.add_argument(
        "--motion-runtime-torque-retract-speed-percent",
        type=float,
        default=2.0,
        help=(
            "Temporary Nero speed percent for the verified reverse retreat "
            "after a torque-contact stop; <=0 keeps the normal speed"
        ),
    )
    p.add_argument(
        "--motion-environment-map",
        dest="motion_environment_map_enabled",
        action="store_true",
        help=(
            "Build and consume a frozen table-level base_link occupancy map "
            "across captured views for grasp motion validation"
        ),
    )
    p.add_argument(
        "--no-motion-environment-map",
        dest="motion_environment_map_enabled",
        action="store_false",
        help="Use the legacy per-capture motion clouds instead of the table-level map",
    )
    p.set_defaults(motion_environment_map_enabled=True)
    p.add_argument(
        "--motion-environment-map-voxel",
        type=float,
        default=0.005,
        help="Voxel size in metres for persistent table-level occupancy",
    )
    p.add_argument(
        "--motion-environment-map-radius-xy",
        type=float,
        default=0.45,
        help=(
            "Default target-centred XY half-width in metres; place targets are "
            "also included with margin before explicit bounds are applied"
        ),
    )
    for axis in ("x", "y"):
        p.add_argument(
            f"--motion-environment-map-{axis}-min",
            type=float,
            default=None,
            help=f"Optional explicit minimum base_link {axis.upper()} bound in metres",
        )
        p.add_argument(
            f"--motion-environment-map-{axis}-max",
            type=float,
            default=None,
            help=f"Optional explicit maximum base_link {axis.upper()} bound in metres",
        )
    p.add_argument(
        "--motion-environment-map-z-below-table",
        type=float,
        default=0.02,
        help="Retain occupancy this far below table_z, in metres",
    )
    p.add_argument(
        "--motion-environment-map-z-above-table",
        type=float,
        default=0.60,
        help="Retain occupancy this far above table_z, in metres",
    )
    p.add_argument(
        "--motion-environment-map-min-view-support",
        type=int,
        default=2,
        help="Minimum distinct-view support required for hard persistent occupancy",
    )
    p.add_argument(
        "--motion-environment-map-single-view-local-radius",
        type=float,
        default=0.020,
        help=(
            "Local connectivity-search radius in metres for retaining coherent "
            "single-view obstacles near the target"
        ),
    )
    p.add_argument(
        "--motion-environment-map-single-view-neighbor-radius",
        type=float,
        default=0.009,
        help="Neighbor acceptance distance in metres for single-view occupancy",
    )
    p.add_argument(
        "--motion-environment-map-single-view-min-neighbors",
        type=int,
        default=2,
        help="Minimum same-view neighbors needed to retain local single-view occupancy",
    )
    p.add_argument(
        "--motion-environment-map-isolated-filter",
        dest="motion_environment_map_isolated_filter",
        action="store_true",
        help="Drop single-view environment voxels with no nearby same-view support before fusion/display/collision use",
    )
    p.add_argument(
        "--no-motion-environment-map-isolated-filter",
        dest="motion_environment_map_isolated_filter",
        action="store_false",
    )
    p.set_defaults(motion_environment_map_isolated_filter=True)
    p.add_argument(
        "--motion-collision-isolated-filter",
        dest="motion_collision_isolated_filter",
        action="store_true",
        help=(
            "Before collision checking, remove environment voxels without "
            "enough local 3D support; connected basket rims remain retained"
        ),
    )
    p.add_argument(
        "--no-motion-collision-isolated-filter",
        dest="motion_collision_isolated_filter",
        action="store_false",
    )
    p.set_defaults(motion_collision_isolated_filter=True)
    p.add_argument(
        "--motion-collision-isolated-radius",
        type=float,
        default=0.020,
        help="Neighbour radius in metres for collision-only isolated-voxel removal",
    )
    p.add_argument(
        "--motion-collision-isolated-min-support",
        type=int,
        default=8,
        help="Minimum voxel count, including itself, inside the local support radius",
    )
    p.add_argument("--arm-collision-model", choices=("urdf_capsules", "legacy_capsules"), default="urdf_capsules", help="Arm collision geometry shared by NBV, grasp planning and runtime guards; URDF mode encloses every collision-mesh triangle with 3 mm padding")
    p.add_argument("--motion-path-link-radius", type=float, default=0.060, help="Uniform arm radius used only by legacy_capsules")
    p.add_argument("--motion-path-tool-radius", type=float, default=0.035, help="Reduced capsule radius for the proximal gripper palm/body only; distal fingers are modeled separately")
    p.add_argument("--motion-path-open-finger-radius", type=float, default=0.007, help="Conservative capsule radius for each physical finger during open and post-close motion")
    p.add_argument("--motion-path-open-finger-back", type=float, default=0.065, help="Open-finger capsule extent behind the TCP along the tool approach axis")
    p.add_argument("--motion-path-open-finger-front", type=float, default=0.020, help="Open-finger capsule extent beyond the TCP along the tool approach axis")
    p.add_argument("--motion-closed-gripper-empty-width", type=float, default=0.000, help="Inner-face gap used only by the empty post-close collision model; zero overlaps both 7 mm-radius finger centerlines into one approximately 14 mm envelope")
    p.add_argument("--motion-closed-gripper-width-padding", type=float, default=0.010, help="Extra gap added only to the GraspNet width for post-grasp collision modeling; it does not change gripper commands or candidate width filtering")
    p.add_argument("--motion-closed-finger-contact-tolerance", type=float, default=0.010, help="Allowed environment penetration for each closed finger during support release, pickup lift, and carrying; palm/camera/arm/table guards remain strict")
    p.add_argument("--target-approach-finger-environment-guard", dest="motion_target_approach_finger_environment_guard", action="store_true", help="Check both open fingers against non-target environment points during the final pregrasp-to-grasp approach")
    p.add_argument("--no-target-approach-finger-environment-guard", dest="motion_target_approach_finger_environment_guard", action="store_false", help="Risk override: ignore open-finger/environment collisions only during final target_approach; arm, gripper body, camera, table, and all other phases remain guarded")
    p.set_defaults(motion_target_approach_finger_environment_guard=True)
    p.add_argument("--direct-pickup-contact-override", dest="direct_pickup_contact_override", action="store_true", help="Risk override: lift directly without a short support-release stage; ignore final-approach finger contact, initial pickup contact, and the legacy solid gripper-tool contact zone during carrying/transport while retaining body/arm/camera/table/hard-joint guards")
    p.add_argument("--no-direct-pickup-contact-override", dest="direct_pickup_contact_override", action="store_false")
    p.set_defaults(direct_pickup_contact_override=False)
    p.add_argument("--motion-path-camera-radius", type=float, default=0.055, help="Camera-body radius before extra grasp-motion clearance; shared optical-frame geometry is also used by active motion")
    p.add_argument("--motion-self-filter", dest="motion_self_filter", action="store_true", help="Remove URDF/link/palm/camera and two-finger self points from the frozen RGB-D safety cloud")
    p.add_argument("--no-motion-self-filter", dest="motion_self_filter", action="store_false")
    p.set_defaults(motion_self_filter=True)
    p.add_argument("--motion-self-filter-margin", type=float, default=0.005)
    p.add_argument("--motion-self-filter-link-radius", type=float, default=0.045)
    p.add_argument("--motion-self-filter-palm-radius", type=float, default=0.035)
    p.add_argument("--motion-self-filter-camera-radius", type=float, default=0.040)
    p.add_argument("--motion-self-filter-finger-radius", type=float, default=0.007)
    p.add_argument("--motion-self-filter-finger-back", type=float, default=0.065)
    p.add_argument("--motion-self-filter-finger-front", type=float, default=0.020)
    p.add_argument("--motion-target-tool-contact-length", type=float, default=0.045, help="Distal flange-to-TCP segment allowed to contact the selected target during final approach")
    p.add_argument("--motion-payload-robot-clearance", type=float, default=0.005, help="Clearance between an attached target and arm/camera primitives outside the distal grasp contact segment")
    p.add_argument("--motion-payload-table-percentile", type=float, default=10.0, help="Robust lower percentile used for attached-target table clearance so isolated low depth outliers do not define the payload floor")
    p.add_argument("--motion-payload-support-plane-clearance", type=float, default=0.0, help="Physical support-plane offset used only for attached-payload/table contact; independent of TCP/tool table clearance")
    p.add_argument("--motion-pickup-table-support-tolerance", type=float, default=0.005, help="Additional robust-bottom tolerance allowed only while initially lifting a target from its table support")
    p.add_argument("--motion-path-min-clearance", type=float, default=0.015, help="Planned clearance outside robot capsules for every grasp-workflow joint sweep")
    p.add_argument("--motion-runtime-stop-clearance", type=float, default=0.010, help="Software emergency-stop clearance used by measured-joint runtime monitoring")
    p.add_argument("--motion-runtime-open-finger-clearance-slack", type=float, default=0.001, help="Runtime-only tolerance subtracted from open-finger clearance during free motion; planning clearance and all other robot primitives remain unchanged")
    p.add_argument("--grasp-execution-waypoint-deg", type=float, default=20.0, help="Maximum per-joint sweep for each execution command on the frozen before-close route; <=0 restores one command per planned stage")
    p.add_argument("--carrying-execution-waypoint-deg", type=float, default=30.0, help="Maximum per-joint sweep for pickup/carrying execution commands")
    p.add_argument("--carrying-smooth-waypoint-deg", type=float, default=35.0)
    p.add_argument("--grasp-smooth-open-waypoint-deg", type=float, default=30.0)
    p.add_argument("--smooth-open-space-handoff", dest="smooth_open_space_handoff", action="store_true")
    p.add_argument("--no-smooth-open-space-handoff", dest="smooth_open_space_handoff", action="store_false")
    p.set_defaults(smooth_open_space_handoff=True)
    p.add_argument("--smooth-handoff-min-extra-clearance", type=float, default=0.020)
    p.add_argument("--smooth-handoff-previous-target-tol-deg", type=float, default=3.0)
    p.add_argument("--adaptive-execution-smoothing", dest="adaptive_execution_smoothing", action="store_true")
    p.add_argument("--no-adaptive-execution-smoothing", dest="adaptive_execution_smoothing", action="store_false")
    p.set_defaults(adaptive_execution_smoothing=True)
    p.add_argument("--adaptive-execution-min-extra-clearance", type=float, default=0.020)
    p.add_argument("--adaptive-execution-tight-extra-clearance", type=float, default=0.015)
    p.add_argument("--adaptive-execution-tight-waypoint-deg", type=float, default=12.0)
    p.add_argument("--place-release-retry-count", type=int, default=1)
    p.add_argument("--place-release-step-count", type=int, default=3)
    p.add_argument("--place-release-step-settle", type=float, default=0.25)
    p.add_argument("--motion-runtime-target-approach-finger-clearance-slack", type=float, default=0.002,
                   help="Runtime-only allowance for open fingers versus environment during final target approach; planning, arm, body and camera keep their thresholds")
    p.add_argument("--motion-runtime-finger-clearance-confirm-samples", type=int, default=2)
    p.add_argument("--motion-runtime-joint-path-residual-deg", type=float, default=8.0)
    p.add_argument("--motion-runtime-joint-path-confirm-samples", type=int, default=2)
    p.add_argument("--trajectory-executor", choices=["move_j", "mit"], default="move_j",
                   help="Joint-route executor: move_j (default) or mit (quintic-timed 200 Hz joint trajectory)")
    p.add_argument("--mit-kp", type=float, nargs=7, default=[30.0, 30.0, 30.0, 15.0, 10.0, 5.0, 5.0],
                   help="MIT joint-trajectory position gains (7)")
    p.add_argument("--mit-kd", type=float, nargs=7, default=[2.0, 2.0, 2.0, 1.0, 0.5, 0.3, 0.3],
                   help="MIT joint-trajectory velocity gains (7)")
    p.add_argument("--mit-max-joint-vel-deg-s", type=float, default=30.0)
    p.add_argument("--mit-max-joint-accel-deg-s2", type=float, default=120.0)
    p.add_argument("--mit-tracking-error-deg", type=float, default=2.0)
    p.add_argument("--mit-tracking-error-confirm-samples", type=int, default=3)
    p.add_argument("--mit-feedback-timeout-s", type=float, default=0.5)
    p.add_argument("--mit-max-feedback-age-s", type=float, default=0.1,
                   help="Hard per-axis CAN age budget in every MIT mode, including hold")
    p.add_argument("--mit-max-feedback-skew-s", type=float, default=0.05,
                   help="Maximum inter-axis frame skew within each CAN feedback stream")
    p.add_argument("--mit-max-control-gap-s", type=float, default=0.1,
                   help="MIT loop interval/work budget; only established stationary holds may use the bounded hold-recovery grace")
    p.add_argument("--mit-hold-recovery-timeout-s", type=float, default=0.5,
                   help="Fixed stationary hold only: bounded fresh-feedback recovery budget (0 disables, maximum 1s); moving trajectories and impedance remain strict")
    p.add_argument("--mit-torque-limit", type=float, default=14.0)
    p.add_argument("--mit-joint-limit-buffer-deg", type=float, default=5.0,
                   help="MIT-loop hard-joint-limit buffer; the current J7=75deg initial pose is compatible with the 5deg default")
    p.add_argument("--mit-joint-limit-planning-slack-deg", type=float, default=1.0,
                   help="Extra planned endpoint margin beyond MIT hard buffer + allowed tracking error")
    p.add_argument("--mit-runtime-guard-hz", type=float, default=20.0,
                   help="Measured-joint collision/torque guard frequency while MIT tracking is active")
    p.add_argument("--nbv-arrival-joint-error-deg", type=float, default=1.0)
    p.add_argument("--nbv-arrival-vel-deg-s", type=float, default=1.0)
    p.add_argument("--nbv-arrival-pose-error-mm", type=float, default=8.0)
    p.add_argument("--nbv-arrival-orientation-error-deg", type=float, default=3.0)
    p.add_argument("--nbv-arrival-stable-samples", type=int, default=5)
    p.add_argument("--nbv-arrival-timeout-s", type=float, default=5.0)
    p.add_argument("--mit-q-log", dest="mit_q_log", action="store_true",
                   help="Log planned/actual q(t) to mit_trajectory_<label>.jsonl")
    p.add_argument("--mit-q-log-dir", default=None)
    p.add_argument("--grasp-roll-candidates-deg", type=float, nargs="+", default=[0.0],
                    help="Sweep grasp roll offsets and select the lowest-cost fully collision-validated workflow; default [0.0] disables the sweep.")
    p.add_argument("--post-grasp-runtime-replan", dest="post_grasp_runtime_replan", action="store_true")
    p.add_argument("--no-post-grasp-runtime-replan", dest="post_grasp_runtime_replan", action="store_false")
    p.set_defaults(post_grasp_runtime_replan=False)
    p.add_argument("--post-grasp-runtime-replan-min-clearance", type=float, default=0.015)
    p.add_argument("--post-grasp-runtime-replan-timeout", type=float, default=1.5)
    p.add_argument("--post-grasp-runtime-replan-max-nodes", type=int, default=200)
    p.add_argument("--grasp-selective-ik-rescue", dest="grasp_selective_ik_rescue", action="store_true")
    p.add_argument("--no-grasp-selective-ik-rescue", dest="grasp_selective_ik_rescue", action="store_false")
    p.set_defaults(grasp_selective_ik_rescue=True)
    p.add_argument("--grasp-selective-ik-rescue-candidates", type=int, default=3)
    p.add_argument("--grasp-selective-ik-rescue-seed-limit", type=int, default=12)
    p.add_argument("--grasp-selective-ik-rescue-all-joint-seeds", type=int, default=8)
    p.add_argument("--grasp-selective-ik-rescue-topk", type=int, default=3)
    p.add_argument("--grasp-selective-ik-rescue-beam-width", type=int, default=3)
    p.add_argument("--active-nbv-selective-ik-rescue", dest="active_nbv_selective_ik_rescue", action="store_true")
    p.add_argument("--no-active-nbv-selective-ik-rescue", dest="active_nbv_selective_ik_rescue", action="store_false")
    p.set_defaults(active_nbv_selective_ik_rescue=True)
    p.add_argument("--active-nbv-selective-ik-rescue-candidates", type=int, default=4)
    p.add_argument("--active-nbv-selective-ik-rescue-all-joint-seeds", type=int, default=8)
    p.add_argument("--active-nbv-selective-ik-rescue-seed-limit", type=int, default=12)
    p.add_argument("--move-controller-aware-retry", dest="move_controller_aware_retry", action="store_true")
    p.add_argument("--no-move-controller-aware-retry", dest="move_controller_aware_retry", action="store_false")
    p.set_defaults(move_controller_aware_retry=True)
    p.add_argument("--move-unstarted-settle-timeout", type=float, default=2.5)
    p.add_argument("--move-unstarted-settle-samples", type=int, default=2)
    p.add_argument("--grasp-close-exact-retry", dest="grasp_close_exact_retry", action="store_true")
    p.add_argument("--no-grasp-close-exact-retry", dest="grasp_close_exact_retry", action="store_false")
    p.set_defaults(grasp_close_exact_retry=True)
    p.add_argument("--ordinary-stop-settle-timeout", type=float, default=3.0)
    p.add_argument("--ordinary-stop-trend-extension", type=float, default=2.0)
    p.add_argument(
        "--released-return-execution-waypoint-deg",
        type=float,
        default=20.0,
        help=(
            "Maximum per-joint step for the successful post-release return; "
            "large collision-checked home edges are split into guarded joint "
            "waypoints to avoid asynchronous wrist flips; <=0 disables"
        ),
    )
    p.add_argument("--motion-target-exclusion-dist", type=float, default=0.002, help="Tight NN radius for matching selected-target surface samples; keep adjacent uncertain points as obstacles")
    p.add_argument("--motion-target-non-tool-clearance", type=float, default=0.005, help="Target clearance required for arm/camera primitives while the gripper approaches")
    p.add_argument("--motion-target-gripper-body-clearance", type=float, default=0.002, help="Selected-target clearance used only for the gripper body during the final target approach; non-target environment clearance is unchanged")
    p.add_argument("--motion-attached-object-clearance", type=float, default=0.012, help="Clearance required between the TCP-attached target cloud and environment")
    p.add_argument("--motion-support-release-height", type=float, default=0.020, help="Initial near-vertical lift used to release a grasped object from its verified table/mat support")
    p.add_argument("--motion-support-strict-rise", type=float, default=0.018, help="Rise at which the initial support points become strict payload obstacles again")
    p.add_argument("--motion-support-plane-band", type=float, default=0.004, help="Half-thickness around calibrated table_z used to identify local support points")
    p.add_argument("--motion-support-target-bottom-tolerance", type=float, default=0.020, help="Maximum target-bottom distance from table_z that permits a support-release exception")
    p.add_argument("--motion-support-xy-margin", type=float, default=0.003, help="Maximum 3 mm expansion around the robust target XY convex hull for local support classification")
    p.add_argument("--motion-support-min-points", type=int, default=8, help="Minimum local planar points required before support-release can be used")
    p.add_argument("--motion-support-plane-max-mad", type=float, default=0.002, help="Maximum support-plane median absolute z deviation")
    p.add_argument("--motion-local-support-enable", dest="motion_local_support_enable", action="store_true", help="Infer a small horizontal support patch below a target resting above calibrated table_z, such as a basket bottom")
    p.add_argument("--no-motion-local-support-enable", dest="motion_local_support_enable", action="store_false")
    p.set_defaults(motion_local_support_enable=True)
    p.add_argument("--motion-local-support-max-gap", type=float, default=0.025, help="Maximum target-bottom to inferred local-support vertical gap")
    p.add_argument("--motion-local-support-contact-distance", type=float, default=0.008, help="Maximum 3D distance from inferred support points to the target cloud")
    p.add_argument("--motion-local-support-plane-max-mad", type=float, default=0.004, help="Maximum z MAD for an inferred basket-bottom/local support patch")
    p.add_argument("--motion-local-support-strict-rise", type=float, default=0.040, help="Near-vertical rise before inferred local support points become strict obstacles again")
    p.add_argument("--motion-support-max-xy-drift", type=float, default=0.005, help="Maximum TCP XY drift during the initial support-release stage")
    p.add_argument("--motion-runtime-support-xy-drift-slack", type=float, default=0.002, help="Runtime-only tracking slack added to the planned support-release XY drift limit; planning remains strict")
    p.add_argument("--motion-support-max-tilt-deg", type=float, default=3.0, help="Maximum TCP orientation change during initial support release")
    p.add_argument("--motion-runtime-support-tilt-slack-deg", type=float, default=2.0, help="Runtime-only angular tracking slack added to the planned support-release tilt limit")
    p.add_argument("--motion-support-downward-tolerance", type=float, default=0.001, help="Maximum permitted TCP downward deviation during support release")
    p.add_argument("--motion-runtime-support-drop-slack", type=float, default=0.005, help="Runtime-only settling/backlash allowance below the highest observed support-release rise; dropping below the grasp start remains strictly guarded")
    p.add_argument("--motion-support-target-contact-distance", type=float, default=0.008, help="Maximum distance used to identify selected-target segmentation leakage during closed-tool support release")
    p.add_argument("--motion-support-target-contact-xy-margin", type=float, default=0.005, help="XY dilation of the robust selected-target footprint used only for closed-tool contact classification")
    p.add_argument("--motion-support-target-contact-z-margin", type=float, default=0.006, help="Z dilation of the robust selected-target envelope used only for closed-tool contact classification")
    p.add_argument("--motion-path-sample-joint-deg", type=float, default=1.0)
    p.add_argument("--motion-path-scene-voxel", type=float, default=0.005)
    p.add_argument("--motion-path-obstacle-cap", type=int, default=50000)
    p.add_argument("--motion-path-fail-on-obstacle-cap", dest="motion_path_fail_on_obstacle_cap", action="store_true", help="Fail closed instead of subsampling occupied safety voxels above the configured cap")
    p.add_argument("--no-motion-path-fail-on-obstacle-cap", dest="motion_path_fail_on_obstacle_cap", action="store_false", help="Diagnostic only: deterministically subsample an oversized safety cloud")
    p.set_defaults(motion_path_fail_on_obstacle_cap=True)
    p.add_argument("--motion-path-max-joint-delta-deg", type=float, default=60.0)
    p.add_argument("--motion-path-require-scene", dest="motion_path_require_scene", action="store_true")
    p.add_argument("--no-motion-path-require-scene", dest="motion_path_require_scene", action="store_false")
    p.set_defaults(motion_path_require_scene=True)
    p.add_argument("--motion-path-require-camera-model", dest="motion_path_require_camera_model", action="store_true")
    p.add_argument("--no-motion-path-require-camera-model", dest="motion_path_require_camera_model", action="store_false")
    p.set_defaults(motion_path_require_camera_model=True)
    p.add_argument("--motion-transit-vertical-clearance", type=float, default=0.080, help="Preferred height above the destination/observed scene for up-and-over motion")
    p.add_argument("--motion-transit-max-tcp-z", type=float, default=0.50, help="Maximum TCP height searched by the up-and-over route planner")
    p.add_argument("--motion-transit-height-step", type=float, default=0.020, help="Height decrement when the highest up-and-over route is not reachable")
    p.add_argument("--motion-up-and-over", dest="motion_up_and_over", action="store_true", help="Enable scene-height raise/traverse/descend routes for grasp transport and recovery")
    p.add_argument("--no-motion-up-and-over", dest="motion_up_and_over", action="store_false", help="Use guarded direct joint paths for grasp transport and recovery; full-scene collision checks remain enabled")
    p.set_defaults(motion_up_and_over=False)
    p.add_argument("--grasp-stage-beam-search", dest="grasp_stage_beam_search", action="store_true", help="Jointly keep multiple collision-valid IK branches across pregrasp, grasp, lift, and transport instead of committing greedily at each stage")
    p.add_argument("--no-grasp-stage-beam-search", dest="grasp_stage_beam_search", action="store_false")
    p.set_defaults(grasp_stage_beam_search=False)
    p.add_argument("--grasp-stage-beam-width", "--grasp-beam-width", dest="grasp_stage_beam_width", type=int, default=6, help="Maximum complete partial IK paths retained after each grasp-motion stage")
    p.add_argument("--grasp-progressive-beam", dest="grasp_progressive_beam", action="store_true", help="Try a small complete-route beam first and expand to the configured full Top-K/beam only when the fast route fails")
    p.add_argument("--no-grasp-progressive-beam", dest="grasp_progressive_beam", action="store_false")
    p.set_defaults(grasp_progressive_beam=False)
    p.add_argument("--grasp-progressive-full-fallback", dest="grasp_progressive_full_fallback", action="store_true", help="Allow a failed fast grasp route to expand into the configured full Top-K/beam search")
    p.add_argument("--no-grasp-progressive-full-fallback", dest="grasp_progressive_full_fallback", action="store_false", help="Keep planning bounded; reject a failed fast route instead of starting full/global expansion")
    p.set_defaults(grasp_progressive_full_fallback=True)
    p.add_argument("--grasp-fast-stage-beam-width", type=int, default=2, help="Cross-stage beam width used by the first progressive complete-route attempt")
    p.add_argument("--grasp-fast-ik-solution-topk", type=int, default=2, help="Per-stage IK Top-K used by the first progressive complete-route attempt")
    p.add_argument("--grasp-fast-ik-seed-limit", type=int, default=5, help="Maximum TRAC-IK seeds checked per stage by the first progressive complete-route attempt")
    p.add_argument("--grasp-preflight-max-candidates", type=int, default=12, help="Maximum cheap-eligible candidates admitted to complete workflow preflight")
    p.add_argument("--grasp-target-valid-plans", type=int, default=3, help="Stop complete candidate preflight after this many frozen executable plans are found")
    p.add_argument("--grasp-micro-lift-height", type=float, default=0.008, help="Strict world-Z rise used by the cheap post-grasp lift-aware IK screen")
    p.add_argument("--grasp-micro-lift-ik-seed-limit", type=int, default=5, help="Maximum TRAC-IK seeds used by the cheap post-grasp micro-lift screen")
    p.add_argument("--grasp-coarse-collision-screen", dest="grasp_coarse_collision_screen", action="store_true", help="Run a 3-degree pregrasp path screen before the mandatory full-resolution collision validation")
    p.add_argument("--no-grasp-coarse-collision-screen", dest="grasp_coarse_collision_screen", action="store_false")
    p.set_defaults(grasp_coarse_collision_screen=False)
    p.add_argument("--motion-neutral-pregrasp-fallback", dest="motion_neutral_pregrasp_fallback", action="store_true", help="If every candidate and jaw direction fails direct beam planning, retry through one collision-checked neutral joint configuration")
    p.add_argument("--no-motion-neutral-pregrasp-fallback", dest="motion_neutral_pregrasp_fallback", action="store_false")
    p.set_defaults(motion_neutral_pregrasp_fallback=False)
    p.add_argument("--motion-neutral-transport-fallback", dest="motion_neutral_transport_fallback", action="store_true", help="When direct carrying/transport fails, retry only that transport through one collision-checked neutral joint configuration")
    p.add_argument("--no-motion-neutral-transport-fallback", dest="motion_neutral_transport_fallback", action="store_false")
    p.set_defaults(motion_neutral_transport_fallback=True)
    p.add_argument("--motion-neutral-joints-deg", type=float, nargs=7, default=None, help="Optional seven-joint neutral waypoint in degrees; defaults to --initial-joints-deg")
    p.add_argument("--motion-direct-release-retreat", type=float, default=0.020, help="Preferred local world-Z rise after release before a guarded direct return; bounded XY/tilt fallbacks may replace the exact endpoint")
    p.add_argument("--motion-release-retreat-xy-relax", type=float, default=0.010, help="Maximum cardinal XY shift used only when the exact post-release retreat endpoint has no safe IK route")
    p.add_argument("--motion-release-retreat-tilt-relax-deg", type=float, default=5.0, help="Maximum local-axis tilt used only when exact/XY post-release retreat endpoints fail")
    p.add_argument("--motion-release-retreat-max-attempts", type=int, default=9, help="Bounded number of post-release retreat endpoint variants including the exact endpoint")
    p.add_argument("--transport-place-relax", dest="transport_place_relax", action="store_true", help="After exact transport/place fails, try a bounded local destination region without replanning the grasp core")
    p.add_argument("--no-transport-place-relax", dest="transport_place_relax", action="store_false")
    p.set_defaults(transport_place_relax=True)
    p.add_argument("--transport-place-xy-relax", type=float, default=0.020, help="Maximum cardinal XY offset for bounded transport/place fallback endpoints")
    p.add_argument("--transport-place-z-relax", type=float, default=0.020, help="Upward place-height offset tried by the bounded transport fallback")
    p.add_argument("--transport-place-tilt-relax-deg", type=float, default=8.0, help="Small local-axis place orientation change tried after exact position fallbacks fail")
    p.add_argument("--transport-place-relax-max-attempts", type=int, default=9, help="Maximum bounded transport/place endpoint alternatives; no global route expansion is performed")
    p.add_argument("--post-grasp-free-route", dest="post_grasp_free_route", action="store_true", help="Opt in to lightweight RRT-Connect after the strict support release; legacy exact lift/transport remains the default")
    p.add_argument("--no-post-grasp-free-route", dest="post_grasp_free_route", action="store_false")
    p.set_defaults(post_grasp_free_route=False)
    p.add_argument("--post-grasp-hold-joints-deg", type=float, nargs=7, default=None,
                   help="After support release, carry to these exact seven joint angles and keep closed; requires a guarded no-place plan")
    p.add_argument("--no-post-grasp-hold", dest="post_grasp_hold_joints_deg",
                   action="store_const", const=None,
                   help="Finish at the original lift endpoint without the configured joint hold")
    p.add_argument("--post-grasp-total-rise", type=float, default=0.150, help="Required TCP height above the grasp point for the free post-grasp high region")
    p.add_argument("--post-grasp-high-region-max-xy", type=float, default=0.250, help="Maximum TCP XY drift admitted when selecting the free high joint region")
    p.add_argument("--post-grasp-high-region-samples", type=int, default=64, help="Low-cost FK-only joint samples used to find a high-region goal")
    p.add_argument("--post-grasp-high-goal-attempts", type=int, default=3, help="Maximum high-region goals attempted by RRT-Connect")
    p.add_argument("--post-grasp-rrt-timeout", type=float, default=3.0, help="Per-goal RRT-Connect timeout in seconds")
    p.add_argument("--post-grasp-rrt-max-nodes", type=int, default=400, help="Maximum nodes in one bounded RRT-Connect search")
    p.add_argument("--post-grasp-rrt-step-deg", type=float, default=8.0, help="Joint-space RRT extension length in degrees (L2 norm)")
    p.add_argument("--post-grasp-rrt-goal-bias", type=float, default=0.20, help="Probability of sampling the opposite RRT tree root")
    p.add_argument("--post-grasp-rrt-seed", type=int, default=0, help="Deterministic post-grasp RRT seed")
    p.add_argument("--post-grasp-rrt-coarse-sample-deg", type=float, default=4.0, help="Collision sampling used only during RRT exploration; accepted edges are rechecked at normal resolution")
    p.add_argument("--post-grasp-rrt-shortcut-attempts", type=int, default=80, help="Deterministic shortcut passes before final path validation")
    p.add_argument("--post-grasp-execution-waypoint-deg", type=float, default=20.0, help="Maximum per-joint delta between simplified post-grasp execution waypoints")
    p.add_argument("--post-grasp-place-ik-seed-limit", type=int, default=5, help="Bounded TRAC-IK seed budget for the final place endpoint")
    p.add_argument("--post-grasp-place-ik-topk", type=int, default=2, help="Maximum final-place IK solutions tried by RRT-Connect")
    p.add_argument("--motion-transition-max-segments", type=int, default=8, help="Maximum adaptive Slerp/Cartesian subdivisions for one up-and-over transition")
    p.add_argument("--motion-high-path-z-tolerance", type=float, default=0.020, help="Maximum FK TCP dip allowed during high-level orientation/horizontal joint sweeps")
    p.add_argument("--motion-stage-start-joint-match-deg", type=float, default=2.0)
    p.add_argument("--motion-min-joint-limit-margin-deg", type=float, default=0.0, help="Optional soft joint-limit margin for planned waypoints; MIT execution automatically raises this to buffer + tracking error + planning slack")
    p.add_argument("--allow-hard-grasp-candidate", action="store_true", help="Allow executing a selected candidate even if reachability precheck marks it as hard pose")
    p.add_argument("--allow-oversize-grasp", dest="allow_oversize_grasp", action="store_true", help="Diagnostic override: retain candidates wider than --max-grasp-width; physical opening remains hardware-capped")
    p.add_argument("--no-allow-oversize-grasp", dest="allow_oversize_grasp", action="store_false")
    p.set_defaults(allow_oversize_grasp=False)
    p.add_argument("--topdown-grasp", dest="topdown_grasp", action="store_true", help="Force the configured tool approach axis to point approximately downward")
    p.add_argument("--no-topdown-grasp", dest="topdown_grasp", action="store_false", help="Use GraspNet's generated grasp orientation without forcing a vertical approach axis")
    p.set_defaults(topdown_grasp=False)
    p.add_argument("--tool-approach-axis", default="x", choices=["x", "y", "z", "-x", "-y", "-z"])
    p.add_argument("--graspnet-approach-axis", default="x", choices=["x", "y", "z", "-x", "-y", "-z"], help="Local GraspNet axis used for pregrasp retreat and grasp advance when --no-topdown-grasp")
    p.add_argument("--graspnet-width-axis", default="y", choices=["x", "y", "z", "-x", "-y", "-z"], help="Local GraspNet axis drawn as gripper width/closing direction in candidate previews")
    p.add_argument("--tcp-gripper-width-axis", default="z", choices=["x", "y", "z", "-x", "-y", "-z"], help="Actual Nero TCP local axis along the gripper opening/closing width")
    p.add_argument("--approach-tilt-deg", type=float, default=5.0)
    p.add_argument("--use-current-rpy", action="store_true", help="Keep the current TCP RPY and only change TCP position")
    p.add_argument("--fixed-rpy", type=float, nargs=3, default=None, help="Override generated TCP RPY with fixed roll pitch yaw in radians")
    p.add_argument("--grasp-advance", type=float, default=0.030, help="Extra TCP advance along the generated approach before closing; default 30mm, in addition to optional GraspNet depth advance")
    p.add_argument(
        "--grasp-table-reference-z", type=float, default=None,
        help="Opt-in: register the center-aligned single/multiview target AND all captured obstacles to a "
             "confirmed physical tabletop height in base coordinates; not inferred from the table guard",
    )
    p.add_argument("--use-graspnet-depth-advance", action="store_true", help="Advance grasp pose along approach axis by GraspNet grasp.depth")
    p.add_argument("--use-intermediate-motion", dest="use_intermediate_motion", action="store_true", help="Diagnostic override: if an unvalidated Cartesian TCP target is not reached, try a raised intermediate TCP pose before retrying")
    p.add_argument("--no-use-intermediate-motion", dest="use_intermediate_motion", action="store_false", help="Do not insert a raised retry waypoint after a motion failure")
    p.set_defaults(use_intermediate_motion=False)
    p.add_argument("--intermediate-lift", type=float, default=0.08, help="Extra z lift in meters for automatically inserted intermediate TCP poses")
    p.add_argument("--motion-mode", default="ik_joint", choices=["cartesian", "ik_joint"], help="ik_joint solves flange IK with TRAC-IK and executes move_j; cartesian uses Nero move_p")
    p.add_argument("--ik-urdf", default=DEFAULT_IK_URDF)
    p.add_argument("--ik-base-link", default="base_link")
    p.add_argument("--ik-tip-link", default="link7")
    p.add_argument("--ik-timeout", type=float, default=0.02)
    p.add_argument("--planning-process-timeout-s", type=float, default=60.0,
                   help="MIT numerical worker request deadline, including the bounded NBV roll/IK search; failure never falls back inline")
    p.add_argument("--ik-epsilon", type=float, default=0.00001)
    p.add_argument("--ik-solver-type", default="Distance", choices=["Speed", "Distance", "Manip1", "Manip2"])
    p.add_argument("--ik-max-joint-delta", type=float, default=None, help="Reject IK solutions whose largest single-joint move exceeds this many radians")
    p.add_argument("--ik-joint7-weight", type=float, default=0.35, help="Extra score weight for J7 motion when choosing among IK solutions")
    p.add_argument("--ik-lift-joint7-weight", type=float, default=0.0, help="Lift-stage J7 motion weight; zero allows larger post-grasp J7 changes while hard joint limits remain enforced")
    p.add_argument("--ik-all-joint-seed-count", type=int, default=24, help="Deterministic low-discrepancy IK seeds distributed across all seven joint limits")
    p.add_argument("--ik-seed-limit-margin-deg", type=float, default=2.0, help="Keep global IK seeds this far inside every hard URDF joint limit")
    p.add_argument("--ik-solution-topk", type=int, default=6, help="Number of distinct score-ranked IK branches retained for complete-route search")
    p.add_argument("--ik-solution-dedup-deg", type=float, default=0.5, help="Treat IK branches within this maximum per-joint difference as duplicate solutions")
    p.add_argument("--ik-test-current", action="store_true", help="Read current Nero flange pose, solve IK/FK for it, print consistency diagnostics, and exit without motion")
    p.add_argument("--min-grasp-z", type=float, default=None)
    p.add_argument("--min-flange-z", type=float, default=0.08)
    p.add_argument("--table-guard-enable", action="store_true", help="Stop execution if current/target TCP goes below the configured table guard height")
    p.add_argument("--table-z", type=float, default=0.0, help="Table plane height in robot base frame, meters")
    p.add_argument("--table-tcp-clearance", type=float, default=0.03, help="Minimum TCP clearance above table_z before protection triggers")
    p.add_argument("--table-tool-bottom-offset", type=float, default=0.005, help="Extra TCP height margin for the lowest gripper/tool point below TCP")
    p.add_argument("--table-guard-stop-on-read-error", action="store_true", help="Trigger stop if TCP pose cannot be read during table guard polling")
    p.add_argument("--initial-table-path-sample-joint-deg", type=float, default=0.5, help="Joint interpolation step used to preflight TCP height before the initial move")
    p.add_argument("--initial-table-fk-live-z-tol", type=float, default=0.02, help="Maximum FK versus live TCP z mismatch allowed before the initial move")
    p.add_argument("--initial-path-max-joint-delta-deg", type=float, default=95.0, help="Largest single-joint sweep allowed for the initial move")
    p.add_argument("--pregrasp-offset", type=float, default=0.08)
    p.add_argument("--adaptive-pregrasp-offsets", dest="adaptive_pregrasp_offsets", action="store_true", help="Try several planning-only pregrasp distances before rejecting a grasp; execution still uses only the winning pregrasp pose")
    p.add_argument("--no-adaptive-pregrasp-offsets", dest="adaptive_pregrasp_offsets", action="store_false")
    p.set_defaults(adaptive_pregrasp_offsets=True)
    p.add_argument("--adaptive-pregrasp-step", type=float, default=0.020, help="Spacing between alternate pregrasp distances in meters")
    p.add_argument("--adaptive-pregrasp-attempts", type=int, default=4, help="Maximum number of pregrasp distances checked per grasp/jaw orientation")
    p.add_argument("--adaptive-pregrasp-min", type=float, default=0.040)
    p.add_argument("--adaptive-pregrasp-max", type=float, default=0.160)
    p.add_argument("--lift-height", type=float, default=0.12)
    p.add_argument("--adaptive-lift-heights", dest="adaptive_lift_heights", action="store_true", help="Treat --lift-height as a preferred maximum and try lower collision-checked lift endpoints when it is unreachable")
    p.add_argument("--no-adaptive-lift-heights", dest="adaptive_lift_heights", action="store_false")
    p.set_defaults(adaptive_lift_heights=True)
    p.add_argument("--adaptive-lift-step", type=float, default=0.020, help="Vertical spacing between descending lift-height fallback endpoints")
    p.add_argument("--adaptive-lift-min-height", type=float, default=0.030, help="Lowest grasp-relative lift height considered by adaptive lift planning; the complete collision guard remains active")
    p.add_argument("--adaptive-lift-attempts", type=int, default=10, help="Maximum number of lift heights including the requested height")
    p.add_argument("--grasp-lift-endpoint-precheck", dest="grasp_lift_endpoint_precheck", action="store_true", help="Run one cached full-seed lift endpoint IK check before expanding complete beam routes")
    p.add_argument("--no-grasp-lift-endpoint-precheck", dest="grasp_lift_endpoint_precheck", action="store_false")
    p.set_defaults(grasp_lift_endpoint_precheck=True)
    p.add_argument("--grasp-lift-min-joint-margin-deg", type=float, default=0.5, help="Soft joint-limit margin required only by the cached lift endpoint feasibility check; URDF hard limits remain mandatory")
    p.add_argument("--grasp-lift-feasible-region", dest="grasp_lift_feasible_region", action="store_true", help="After a verified support release, search descending lift heights plus bounded XY/orientation alternatives instead of requiring one exact full-lift TCP endpoint")
    p.add_argument("--no-grasp-lift-feasible-region", dest="grasp_lift_feasible_region", action="store_false")
    p.set_defaults(grasp_lift_feasible_region=False)
    p.add_argument("--grasp-lift-region-xy", type=float, default=0.030, help="Maximum cardinal XY endpoint shift used by lift feasible-region fallback")
    p.add_argument("--grasp-lift-region-max-attempts", type=int, default=24, help="Maximum lift endpoint variants considered per already-validated grasp branch")
    p.add_argument("--grasp-validation-mode", dest="grasp_validation_mode", action="store_true", help="If every full lift/transport option fails after a valid grasp core, allow execution through verified support release only and keep the gripper closed")
    p.add_argument("--no-grasp-validation-mode", dest="grasp_validation_mode", action="store_false")
    p.set_defaults(grasp_validation_mode=False)
    p.add_argument("--grasp-validation-first", dest="grasp_validation_first", action="store_true", help="Qualify and execute close plus verified support release before considering full lift/transport")
    p.add_argument("--no-grasp-validation-first", dest="grasp_validation_first", action="store_false")
    p.set_defaults(grasp_validation_first=False)
    p.add_argument("--grasp-validation-lift-height", type=float, default=0.070, help="Requested grasp-relative rise for validation-only fallback; the planner automatically raises it when the inferred support-release latch is higher")
    p.add_argument("--lift-orientation-relax", dest="lift_orientation_relax", action="store_true", help="After original-orientation planning fails, retry the lift endpoint with small orientation changes; no extra execution waypoint is added")
    p.add_argument("--no-lift-orientation-relax", dest="lift_orientation_relax", action="store_false")
    p.set_defaults(lift_orientation_relax=True)
    p.add_argument("--lift-orientation-relax-degrees", type=float, nargs="+", default=[3.0, 6.0, 10.0, 15.0, 20.0], help="Ordered absolute lift endpoint orientation fallback angles in degrees; grasp/close orientation is never changed")
    p.add_argument("--lift-orientation-relax-axis", choices=["tool_approach", "x", "y", "z", "-x", "-y", "-z"], default="tool_approach", help="TCP-local axis used for lift endpoint orientation relaxation")
    p.add_argument("--place-xyz", type=float, nargs=3, default=None)
    p.add_argument("--place-rpy-deg", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"), help="Optional fixed place TCP orientation in degrees; omitted preserves the grasp/lift orientation")

    p.add_argument("--disable-gripper", action="store_true")
    p.add_argument("--gripper-test", choices=["open", "close", "status"], default=None, help="Only test AGX gripper and exit")
    p.add_argument("--gripper-open-width", type=float, default=0.100)
    p.add_argument("--gripper-hardware-max-width", type=float, default=0.100, help="Configured hard upper bound for every physical gripper width command")
    p.add_argument("--gripper-hardware-max-force", type=float, default=0.50, help="Configured hard upper bound for every physical gripper force command")
    p.add_argument("--gripper-close-width", type=float, default=0.0)
    p.add_argument(
        "--grasp-contact-check-only", action="store_true",
        help="Move along the verified grasp route with open fingers, save fresh RGB-D/pose, "
             "then execute the verified empty recovery; do not close or lift",
    )
    p.add_argument("--grasp-contact-check-clearance", type=float, default=None,
                   help="Signed CAD/table trial clearance in metres (-0.005..0.005), only with "
                        "--grasp-contact-check-only; negative allowance applies to table surface only")
    p.add_argument("--gripper-force", type=float, default=0.50)
    p.add_argument("--gripper-settle", type=float, default=1.0)
    p.add_argument("--require-gripper-open", action="store_true", default=True)
    p.add_argument("--no-require-gripper-open", dest="require_gripper_open", action="store_false")
    p.add_argument("--gripper-open-tol", type=float, default=0.004)
    p.add_argument("--gripper-open-timeout", type=float, default=5.0)
    p.add_argument("--gripper-open-confirm-samples", type=int, default=2, help="Consecutive valid width-mode samples required before arm motion after opening")
    p.add_argument("--slow-close-gripper", dest="slow_close_gripper", action="store_true", help="Close gripper in width steps after reaching grasp pose")
    p.add_argument("--no-slow-close-gripper", dest="slow_close_gripper", action="store_false", help="Close gripper with one direct command instead of force-monitored slow close")
    p.set_defaults(slow_close_gripper=True)
    p.add_argument("--slow-close-step", type=float, default=0.005, help="Width decrement per slow-close step in meters")
    p.add_argument("--slow-close-interval", type=float, default=0.15, help="Delay between slow-close width commands in seconds")
    p.add_argument("--gripper-stop-force", type=float, default=0.40, help="During --slow-close-gripper, stop issuing narrower commands once abs(reported gripper force) reaches this value")
    p.add_argument("--gripper-contact-confirm-samples", type=int, default=2, help="Consecutive force-threshold samples required before contact is accepted")
    p.add_argument("--gripper-contact-confirm-interval", type=float, default=0.10, help="Seconds between force-contact confirmation samples")
    p.add_argument("--grasp-stability-samples", type=int, default=5, help="Gripper feedback samples recorded after lift while the object remains grasped")
    p.add_argument("--grasp-stability-interval", type=float, default=0.10, help="Seconds between post-lift grasp-retention samples")
    p.add_argument("--grasp-stability-max-width-drift", type=float, default=0.002, help="Maximum post-lift gripper-width drift counted as stable, in metres")
    p.add_argument("--require-gripper-contact", dest="require_gripper_contact", action="store_true", help="Require confirmed force contact before lifting after slow close")
    p.add_argument("--no-require-gripper-contact", dest="require_gripper_contact", action="store_false")
    p.set_defaults(require_gripper_contact=True)
    p.add_argument("--require-gripper-status", dest="require_gripper_status", action="store_true", help="Fail closed when gripper feedback is missing after close")
    p.add_argument("--no-require-gripper-status", dest="require_gripper_status", action="store_false")
    p.set_defaults(require_gripper_status=True)
    p.add_argument("--verbose-gripper-close", action="store_true", help="Print every slow gripper close command/status sample")
    p.add_argument("--grasp-close-pos-tol", type=float, default=0.003, help="Required flange position error before closing the gripper")
    p.add_argument("--grasp-close-rpy-tol", type=float, default=np.deg2rad(3.0), help="Required flange relative rotation angle before closing the gripper, in radians")
    p.add_argument("--grasp-close-wait-timeout", type=float, default=8.0, help="Extra seconds to wait for the tight grasp pose before closing")
    p.add_argument("--empty-grasp-width-threshold", type=float, default=0.001, help="For direct-close compatibility, treat width at/below this value as fully closed and empty")
    p.add_argument("--gripper-closed-width-tol", type=float, default=0.001, help="Measured-width tolerance above --gripper-close-width that counts as the physical fully closed limit")
    p.add_argument("--gripper-close-completion-timeout", type=float, default=2.0, help="After the 0 mm command, keep polling this long for physical closure or sustained force contact")
    p.add_argument("--gripper-close-completion-poll", type=float, default=0.10, help="Feedback polling interval while waiting for the final close command to complete")
    p.add_argument("--allow-empty-grasp-lift", dest="allow_empty_grasp_lift", action="store_true", help="Continue lift/place even if gripper appears fully closed with no object")
    p.add_argument("--no-allow-empty-grasp-lift", dest="allow_empty_grasp_lift", action="store_false")
    p.set_defaults(allow_empty_grasp_lift=False)
    p.add_argument("--empty-grasp-return-delay", type=float, default=1.0, help="Seconds to wait after an empty grasp is detected before opening gripper and returning")
    p.add_argument(
        "--empty-recovery-execution-waypoint-deg",
        type=float,
        default=15.0,
        help=(
            "Maximum per-joint step for empty-grasp recovery commands; split "
            "the collision-checked recovery edge into smaller commands so "
            "asynchronous joint motion stays close to the validated path; "
            "<=0 restores one command per planned stage"
        ),
    )
    p.add_argument("--skip-empty-grasp-return", dest="skip_empty_grasp_return", action="store_true", help="Do not return to initial pose after empty grasp; only open the gripper")
    p.add_argument("--no-skip-empty-grasp-return", dest="skip_empty_grasp_return", action="store_false")
    p.set_defaults(skip_empty_grasp_return=False)

    p.add_argument("--execute", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--no-motion-fail-prompt", action="store_true", help="Do not show STATUS/OPEN/INIT recovery prompt after a motion target is not reached")
    p.add_argument("--move-wait-timeout", type=float, default=20.0)
    p.add_argument("--move-poll-interval", type=float, default=0.10)
    p.add_argument("--move-pos-tol", type=float, default=0.015)
    p.add_argument("--move-rpy-tol", type=float, default=np.deg2rad(5.0))
    p.add_argument("--move-joint-tol", type=float, default=np.deg2rad(2.0))
    p.add_argument("--move-start-timeout", type=float, default=1.5, help="Seconds allowed for a joint command to show measurable motion")
    p.add_argument("--move-start-threshold-deg", type=float, default=0.20, help="Minimum joint displacement that proves a new move_j command started")
    p.add_argument("--move-precommand-settle-timeout", type=float, default=1.5, help="Maximum time to wait for the preceding waypoint to stop settling before sending the next move_j")
    p.add_argument("--move-precommand-settle-samples", type=int, default=3, help="Consecutive stable encoder samples required before a new move_j")
    p.add_argument("--move-precommand-settle-delta-deg", type=float, default=0.05, help="Maximum per-sample joint change counted as settled")
    p.add_argument("--move-precommand-settle-poll", type=float, default=0.08, help="Encoder polling interval used by the pre-command settle barrier")
    p.add_argument("--move-retry-unstarted", dest="move_retry_unstarted", action="store_true", help="Retry the exact same joint target once when it did not start and controller status is normal")
    p.add_argument("--no-move-retry-unstarted", dest="move_retry_unstarted", action="store_false")
    p.set_defaults(move_retry_unstarted=True)
    p.add_argument("--move-retry-delay", type=float, default=0.50, help="Delay before the one allowed exact-target retry")
    p.add_argument("--move-retry-stalled", dest="move_retry_stalled", action="store_true", help="Retry once when a command was acknowledged but remained at its measured start")
    p.add_argument("--no-move-retry-stalled", dest="move_retry_stalled", action="store_false")
    p.set_defaults(move_retry_stalled=True)
    p.add_argument("--move-late-retry-max-displacement-deg", type=float, default=0.50, help="Maximum motion from command start that still identifies a dropped command for the late exact-target retry")
    p.add_argument("--move-stall-grace", type=float, default=1.0, help="Seconds before detecting an idle command far from its target")
    p.add_argument("--move-stall-confirm-samples", type=int, default=5, help="Consecutive nearly stationary samples used to detect a dropped command")
    p.add_argument("--move-stall-delta-deg", type=float, default=0.02, help="Maximum per-sample joint change counted as stationary for dropped-command detection")
    p.add_argument("--move-stall-far-error-deg", type=float, default=5.0, help="Only declare an early stall while the joint target remains at least this far away")

    p.add_argument("--move-residual-retry", dest="move_residual_retry", action="store_true", help="Resend the same prevalidated joint target once after a stall within the residual window")
    p.add_argument("--no-move-residual-retry", dest="move_residual_retry", action="store_false")
    p.set_defaults(move_residual_retry=True)
    p.add_argument("--move-residual-retry-max-deg", type=float, default=12.0, help="Maximum joint-error L2 (deg) that still gets one residual resend")
    p.add_argument("--stop-confirm-timeout", type=float, default=2.0, help="Seconds allowed to confirm ordinary trajectory termination")

    p.add_argument("--move-status-max-age", type=float, default=1.0, help="Maximum controller-status age allowed before retrying an unstarted joint command")

    p.add_argument(
        "--phase", choices=["grasp", "servo", "all"], default="all",
        help="grasp=仅抓取; servo=仅伺服递送(调试); all=抓取+可选递送",
    )
    p.add_argument(
        "--enable-servo-delivery", action="store_true", default=False,
        help="仅当 --phase all 且抓取成功后：交接 MIT 后端并执行伺服递送",
    )
    from cameras.options import add_camera_arguments
    from real_nero_grasp.servo_cli import add_servo_arguments
    add_camera_arguments(p)
    add_servo_arguments(p)
    from cameras.options import apply_pipeline_defaults
    apply_pipeline_defaults(p)
    args = p.parse_args()
    from .active_grasp_feedback import validate_feedback_args
    try:
        validate_feedback_args(args)
    except ValueError as exc:
        p.error(str(exc))
    from handoff.fusion_cli import validate_fusion_args
    try:
        validate_fusion_args(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    if args.nero_grasp_refine:
        if (args.tool_approach_axis != 'x' or args.tcp_gripper_width_axis != 'z'
                or not np.isclose(min(args.gripper_open_width, args.gripper_hardware_max_width), .100)
                or args.topdown_grasp or args.fixed_rpy is not None or args.use_current_rpy
                or args.use_graspnet_depth_advance or args.min_grasp_z is not None
                or args.grasp_table_reference_z is None or args.grasp_contact_check_clearance is not None
                or args.motion_mode != 'ik_joint' or not args.grasp_motion_path_guard):
            p.error('--nero-grasp-refine requires native Nero x/z axes, 100 mm opening, confirmed table reference and guarded joint planning; no fixed-pose/depth/floor overrides or signed clearance trial')
        if (not 1 <= args.nero_refine_max_route_variants <= 8
                or not np.isfinite(args.nero_refine_timeout_s) or args.nero_refine_timeout_s <= 0):
            p.error('Nero refinement requires 1..8 route variants and a finite positive time budget')
    if args.grasp_contact_check_clearance is not None:
        if (not args.grasp_contact_check_only
                or not np.isfinite(args.grasp_contact_check_clearance)
                or not -0.005 <= args.grasp_contact_check_clearance <= 0.005):
            p.error('--grasp-contact-check-clearance requires --grasp-contact-check-only and -0.005..0.005 m')
        if args.grasp_contact_check_clearance < 0 and args.grasp_table_reference_z is None:
            p.error('negative contact-check clearance requires --grasp-table-reference-z')
    if args.graspgenx_contact_advance == 'auto':
        if args.grasp_generator != 'graspgenx':
            p.error('--graspgenx-contact-advance auto requires GraspGenX')
    elif (not np.isfinite(args.graspgenx_contact_advance)
            or not 0.0 <= args.graspgenx_contact_advance <= 0.015
            or (args.graspgenx_contact_advance != 0.0 and args.grasp_generator != 'graspgenx')):
        p.error('--graspgenx-contact-advance requires GraspGenX and 0..0.015 m or auto')
    if args.post_grasp_hold_joints_deg is not None:
        hold = np.asarray(args.post_grasp_hold_joints_deg, dtype=float)
        if hold.shape != (7,) or not np.isfinite(hold).all():
            p.error('--post-grasp-hold-joints-deg requires seven finite angles in degrees')
    if args.grasp_required_view_count and not args.active_view:
        p.error('--grasp-required-view-count requires --active-view')
    if args.phase != 'servo' and not args.calib_json:
        p.error('--calib-json is required for grasp/all')
    if (args.phase == 'servo' or args.enable_servo_delivery) and not args.handeye:
        p.error('--handeye is required for servo delivery')
    return args
