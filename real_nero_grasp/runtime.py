#!/usr/bin/env python3
"""Backward-compatible facade for the modular real Nero grasp workflow.

Focused sibling modules own perception, candidate validation, active-view
selection, collision checking, motion planning, frozen-plan execution,
clear-first policy, and top-level session orchestration.  This facade keeps the
legacy import surface and injects its current globals at each composite call so
existing monkeypatches and external helper imports retain their old behavior.
"""

import copy
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from real_nero_grasp.paths import (
    DEFAULT_IK_URDF,
    DEFAULT_YOLO_WEIGHTS,
    GRASPNET_IMPORT_DIRS,
    PROJECT_ROOT,
)

ROOT_DIR = str(PROJECT_ROOT)
for _graspnet_import_dir in GRASPNET_IMPORT_DIRS:
    if _graspnet_import_dir not in sys.path:
        sys.path.append(_graspnet_import_dir)

from real_nero_grasp import active_motion as _active_motion_impl
from real_nero_grasp import active_view as _active_view_impl
from real_nero_grasp import active_view_scoring as _active_view_scoring_impl
from real_nero_grasp import arm_execution as _arm_execution_impl
from real_nero_grasp import clear_policy as _clear_policy_impl
from real_nero_grasp import collision as _collision_impl
from real_nero_grasp import execution as _execution_impl
from real_nero_grasp import grasp_candidates as _grasp_candidates_impl
from real_nero_grasp import grasp_planner as _grasp_planner_impl
from real_nero_grasp import grasp_workflow as _grasp_workflow_impl
from real_nero_grasp import ik_solver as _ik_solver_impl
from real_nero_grasp import motion_planning as _motion_planning_impl
from real_nero_grasp import perception as _perception_impl
from real_nero_grasp import pointcloud_fusion as _pointcloud_fusion_impl
from real_nero_grasp import runtime_safety as _runtime_safety_impl
from real_nero_grasp import session_control as _session_control_impl
from real_nero_grasp import visualization as _visualization_impl
from real_nero_grasp import workflow as _workflow_impl
from real_nero_grasp.arm import RealNeroGraspArm
from real_nero_grasp.active_motion import _tcp_pose_for_flange_pose
from real_nero_grasp.active_view_scoring import (
    _active_balanced_contact_coverage,
    _active_cap_weights,
    _active_fibonacci_surface_bins,
    _active_spherical_azimuths,
    _active_spherical_camera_position,
    _build_active_information_proxy,
    _clamp01,
    _dir_from_angle_rad,
    _direction_context_penalty,
    _normalize_dir_xy,
    _parse_active_orbit_tilts,
    _pca_main_axes_xy,
    _predict_visibility_from_viewdir,
    _score_active_candidate_information,
    _smallest_angle_diff_deg,
    _target_visibility_score,
)
from real_nero_grasp.cli import parse_args
from real_nero_grasp.experiment_log import (
    finalize_experiment_log,
    initialize_experiment_log,
    mark_experiment_stage,
)
from real_nero_grasp.collision import (
    _URDF_CHAIN_CACHE,
    _active_motion_scene_points,
    _bounded_motion_points,
    _load_urdf_joint_chain,
    _urdf_origin_transform,
    _urdf_vector,
    assert_pickup_payload_table_start,
    assert_table_guard_target_tcp,
    table_guard_enabled,
    table_guard_min_tcp_z,
)
from real_nero_grasp.depth_processing import (
    _bbox_mask,
    _expand_bbox_xyxy,
    _mask_bbox,
    _mask_center,
    _mask_iou,
    _mask_nested_overlap,
    _median_depth_for_mask,
    _resize_mask,
    depth_to_xyz_image,
    fill_small_xyz_holes_in_mask,
    pointcloud2_xyz_at_from_image,
    pointcloud2_xyz_image,
    refine_xyz_inside_visible_mask,
)
from real_nero_grasp.pointcloud_fusion import (
    _apply_T_points,
    _cloud_stats,
    _mutual_overlap_stats,
    _nn_gate,
    _nn_stats,
    _points_from_cloud,
    transform_cloud,
)
from real_nero_grasp.grasp_geometry import (
    _approach_axis_local,
    _axis_index_sign,
    _axis_vector,
    _horizontal_unit,
    _rotation_from_axis_constraints,
    grasp_tcp_axis_diagnostics,
    grasp_to_tcp_poses,
    graspnet_rotation_to_tcp_rotation,
    topdown_rotation_from_hint,
)
from real_nero_grasp.ik_solver import (
    JOINT_LIMIT_NUMERICAL_TOL_RAD,
    _clip_numerical_joint_limit_overshoot,
    _get_ik_solver,
    _joint_limit_margin,
    _joint_seed_candidates,
    _pose_to_wxyz,
    _wrap_joint_delta,
)
from real_nero_grasp.motion_planning import (
    _motion_context,
    _motion_route_height_candidates,
)
from real_nero_grasp.transforms import (
    _as_pose_list,
    _make_T_from_pose,
    _point_segment_distances,
    _pose_rpy_error_norm,
    _transform_points,
    interpolate_pose,
)
from real_nero_grasp.types import (
    ActiveViewResult,
    ControlledMotionStopTriggered,
    DynamicEnvironmentMap,
    GraspAttemptResult,
    GraspAttemptStatus,
    GraspMotionPlan,
    GraspNotDetected,
    GraspSelectionDeclined,
    GripperCloseResult,
    InitialSupportRelease,
    InvalidGraspSelection,
    MotionCollisionContext,
    MotionNotReached,
    MotionSafetyScene,
    PlanNotExecutable,
    PlannedJointStage,
    SafetyStopTriggered,
    TargetIdentityNotVerified,
    TorqueCollisionStopTriggered,
    UnsafeMotionTarget,
)
from real_nero_grasp.visualization import (
    get_graspnet_helpers,
    get_o3d,
    release_vision_model_caches,
)

from tools.real_nero_point_test import (
    DEFAULT_INITIAL_JOINT_DEG,
    DEFAULT_TCP_OFFSET,
    capture_rgbd,
    capture_ros_color_pointcloud_frame,
    handeye_transform,
    make_T,
    pointcloud2_xyz_at,
    print_status,
    safe_stop,
    pose_error,
    wait_until_joints_reached as _base_wait_until_joints_reached,
    wait_until_pose_reached as _base_wait_until_pose_reached,
    wrap_angle_error,
)



def record_workflow_motion(args, motion_sent):
    return _workflow_impl.record_workflow_motion(args, motion_sent)


def with_workflow_motion(args, result):
    return _workflow_impl.with_workflow_motion(args, result)

def _servo_delivery_after_grasp(arm, args):
    """抓取成功后(arm 仍在 with 上下文内）：交接 MIT 后端并执行伺服递送。"""
    from handoff.grasp_to_servo_handoff import delivery_after_grasp
    return delivery_after_grasp(arm, args)


def _session_control_hooks():
    """Resolve legacy-patchable session dependencies at each facade call."""
    return _session_control_impl.SessionControlHooks(
        print_status=print_status,
        preflight_initial_table_path=_preflight_initial_table_path,
        wait_until_joints_reached=wait_until_joints_reached,
        wrap_angle_error=wrap_angle_error,
        safe_stop=safe_stop,
        terminate_motion=lambda arm, _reason: arm.terminate_motion(),
        confirm_motion=confirm_motion,
        wait_until_gripper_open=wait_until_gripper_open,
        move_initial=move_initial, # 人工确认→关节运动→到位等待
    )


def confirm_motion(action_description):
    return _session_control_impl.confirm_motion(action_description)


def draw_open3d_preview(geometries, window_name, args, width=1280, height=720):
    return _visualization_impl.draw_open3d_preview(
        geometries,
        window_name,
        args,
        width=width,
        height=height,
        get_o3d_hook=get_o3d,
    )







def set_tcp_offset(arm, args):
    return _session_control_impl.set_tcp_offset(
        arm,
        args,
        hooks=_session_control_hooks(),
    )


def move_initial(arm, args, require_confirm=True):
    return _session_control_impl.move_initial(
        arm,
        args,
        require_confirm=require_confirm,
        hooks=_session_control_hooks(),
    )


def return_initial_with_open_gripper(arm, args):
    return _session_control_impl.return_initial_with_open_gripper(
        arm,
        args,
        hooks=_session_control_hooks(),
    )


def recover_after_empty_grasp(arm, args):
    return _session_control_impl.recover_after_empty_grasp(
        arm,
        args,
        hooks=_session_control_hooks(),
    )


def verify_empty_grasp_recovery_at_initial_observation(arm, args, recovered):
    return _session_control_impl.verify_empty_grasp_recovery_at_initial_observation(
        arm,
        args,
        recovered,
        hooks=_session_control_hooks(),
    )


def wait_until_gripper_open(arm, args):
    return _session_control_impl.wait_until_gripper_open(arm, args)


def prepare_initial_pose_for_capture(arm, args):
    return _session_control_impl.prepare_initial_pose_for_capture(
        arm,
        args,
        hooks=_session_control_hooks(),
    )


def assert_grasp_detected(close_status, args):
    return _session_control_impl.assert_grasp_detected(close_status, args)


def check_table_guard_current(arm, args, label):
    return _arm_execution_impl.check_table_guard_current(
        arm,
        args,
        label,
        table_guard_enabled_hook=table_guard_enabled,
        table_guard_min_tcp_z_hook=table_guard_min_tcp_z,
        safe_stop_hook=safe_stop,
    )


def _runtime_safety_hooks():
    """Resolve legacy-patchable runtime protection dependencies per call."""
    return _runtime_safety_impl.RuntimeSafetyHooks(
        as_pose_list=_as_pose_list,
        tcp_transform_from_joints=_tcp_transform_from_joints,
        motion_configuration_metrics=_motion_configuration_metrics,
        persist_runtime_collision_diagnostic=_persist_runtime_collision_diagnostic,
        safe_stop=safe_stop,
    )


def _persist_runtime_collision_diagnostic(args, label, joints, context, metrics):
    return _runtime_safety_impl._persist_runtime_collision_diagnostic(
        args,
        label,
        joints,
        context,
        metrics,
        hooks=_runtime_safety_hooks(),
    )


def check_runtime_collision_current(arm, args, label, joints=None):
    return _runtime_safety_impl.check_runtime_collision_current(
        arm,
        args,
        label,
        joints=joints,
        hooks=_runtime_safety_hooks(),
    )


def wait_until_pose_reached(nero, target_pose, args, label):
    return _arm_execution_impl.wait_until_pose_reached(
        nero,
        target_pose,
        args,
        label,
        table_guard_enabled_hook=table_guard_enabled,
        base_wait_until_pose_reached_hook=_base_wait_until_pose_reached,
        check_table_guard_current_hook=check_table_guard_current,
        check_runtime_collision_current_hook=check_runtime_collision_current,
        pose_error_hook=pose_error,
    )


def _enum_numeric(value):
    return _arm_execution_impl._enum_numeric(value)


def _arm_motion_status_snapshot(nero, args=None):
    return _arm_execution_impl._arm_motion_status_snapshot(nero, args)


def _wait_for_joint_motion_start(nero, q_before, q_target, args, label):
    return _arm_execution_impl._wait_for_joint_motion_start(
        nero,
        q_before,
        q_target,
        args,
        label,
        wrap_angle_error_hook=wrap_angle_error,
        check_table_guard_current_hook=check_table_guard_current,
        check_runtime_collision_current_hook=check_runtime_collision_current,
    )


def _wait_for_joint_settle(nero, args, label):
    return _arm_execution_impl._wait_for_joint_settle(
        nero,
        args,
        label,
        wrap_angle_error_hook=wrap_angle_error,
        check_table_guard_current_hook=check_table_guard_current,
        check_runtime_collision_current_hook=check_runtime_collision_current,
    )


def _send_joint_target_with_start_retry(nero, q_target, args, label):
    return _arm_execution_impl._send_joint_target_with_start_retry(
        nero,
        q_target,
        args,
        label,
        wait_for_joint_motion_start_hook=_wait_for_joint_motion_start,
        wait_for_joint_settle_hook=_wait_for_joint_settle,
        arm_motion_status_snapshot_hook=_arm_motion_status_snapshot,
        validate_motion_joint_path_hook=validate_motion_joint_path,
    )


def wait_until_joints_reached(nero, target_joints, args, label):
    return _arm_execution_impl.wait_until_joints_reached(
        nero,
        target_joints,
        args,
        label,
        table_guard_enabled_hook=table_guard_enabled,
        base_wait_until_joints_reached_hook=_base_wait_until_joints_reached,
        check_table_guard_current_hook=check_table_guard_current,
        check_runtime_collision_current_hook=check_runtime_collision_current,
        wrap_angle_error_hook=wrap_angle_error,
    )




def draw_instances(bgr, instances):
    return _perception_impl.draw_instances(bgr, instances)


def _class_matches(label, target_class):
    return _perception_impl._class_matches(label, target_class)


def infer_yolo_instances(bgr, args, output_vis_path=None):
    return _perception_impl.infer_yolo_instances(
        bgr,
        args,
        output_vis_path=output_vis_path,
        class_matches_hook=_class_matches,
        resize_mask_hook=_resize_mask,
        mask_center_hook=_mask_center,
    )


def mask_for_instance(inst, xyz_img, args):
    return _perception_impl.mask_for_instance(inst, xyz_img, args)


def reidentify_instance_from_reference_mask(
    instances,
    xyz_img,
    reference_mask,
    reference_label,
    args,
):
    return _perception_impl.reidentify_instance_from_reference_mask(
        instances,
        xyz_img,
        reference_mask,
        reference_label,
        args,
        mask_for_instance_hook=mask_for_instance,
        class_matches_hook=_class_matches,
        mask_center_hook=_mask_center,
        mask_iou_hook=_mask_iou,
    )


def click_point(image, title):
    return _perception_impl.click_point(image, title)


def save_selected_target_visuals(bgr, mask, inst, out_dir):
    return _perception_impl.save_selected_target_visuals(
        bgr, mask, inst, out_dir
    )


def apply_target_prompt(args):
    return _perception_impl.apply_target_prompt(args)


def largest_visible_clear_instance(
    bgr,
    xyz_img,
    args,
    out_dir,
    instances=None,
    exclude_mask=None,
):
    return _perception_impl.largest_visible_clear_instance(
        bgr,
        xyz_img,
        args,
        out_dir,
        instances=instances,
        exclude_mask=exclude_mask,
        infer_yolo_instances_hook=infer_yolo_instances,
        mask_for_instance_hook=mask_for_instance,
        mask_iou_hook=_mask_iou,
        median_depth_for_mask_hook=_median_depth_for_mask,
    )


def select_target_mask(bgr, xyz_img, args, out_dir):
    return _perception_impl.select_target_mask(
        bgr,
        xyz_img,
        args,
        out_dir,
        infer_yolo_instances_hook=infer_yolo_instances,
        reidentify_instance_hook=reidentify_instance_from_reference_mask,
        mask_for_instance_hook=mask_for_instance,
        save_selected_visuals_hook=save_selected_target_visuals,
        draw_instances_hook=draw_instances,
        click_point_hook=click_point,
        largest_visible_instance_hook=largest_visible_clear_instance,
        pointcloud2_xyz_at_hook=pointcloud2_xyz_at_from_image,
    )


def mask_from_projected_target_center(
    bgr,
    xyz_img,
    T_base_camera,
    target_center_base,
    args,
    out_dir,
    label="active_projected",
):
    return _perception_impl.mask_from_projected_target_center(
        bgr,
        xyz_img,
        T_base_camera,
        target_center_base,
        args,
        out_dir,
        label=label,
        save_selected_visuals_hook=save_selected_target_visuals,
    )


def mask_from_clicked_target_point(
    bgr,
    xyz_img,
    args,
    out_dir,
    label="active_clicked",
):
    return _perception_impl.mask_from_clicked_target_point(
        bgr,
        xyz_img,
        args,
        out_dir,
        label=label,
        click_point_hook=click_point,
        pointcloud2_xyz_at_hook=pointcloud2_xyz_at_from_image,
        save_selected_visuals_hook=save_selected_target_visuals,
    )


def cloud_from_mask(xyz_img, bgr, mask, args):
    return _perception_impl.cloud_from_mask(
        xyz_img,
        bgr,
        mask,
        args,
        get_o3d_hook=get_o3d,
    )


def scene_cloud_from_xyz(xyz_img, bgr, args):
    return _perception_impl.scene_cloud_from_xyz(
        xyz_img,
        bgr,
        args,
        get_o3d_hook=get_o3d,
    )


def obstacle_cloud_from_non_target_mask(xyz_img, bgr, target_mask, args):
    return _perception_impl.obstacle_cloud_from_non_target_mask(
        xyz_img,
        bgr,
        target_mask,
        args,
        get_o3d_hook=get_o3d,
    )


def crop_scene_cloud_near_target(
    scene_cloud,
    target_cloud,
    args,
    label="scene",
):
    return _perception_impl.crop_scene_cloud_near_target(
        scene_cloud,
        target_cloud,
        args,
        label=label,
        get_o3d_hook=get_o3d,
    )


def _deduplicate_strict_identity_candidates(
    scored,
    args,
    view_label="active",
):
    return _perception_impl._deduplicate_strict_identity_candidates(
        scored,
        args,
        view_label=view_label,
        mask_iou_hook=_mask_iou,
        mask_nested_overlap_hook=_mask_nested_overlap,
    )


def configure_amodal_clear_model(args):
    return _clear_policy_impl.configure_amodal_clear_model(args)


def _empty_target_occlusion(reason, enabled=False):
    return _clear_policy_impl._empty_target_occlusion(
        reason,
        enabled=enabled,
    )


def _target_occlusion_report(evidence):
    return _clear_policy_impl._target_occlusion_report(evidence)


def infer_target_amodal_occlusion(
    bgr,
    target_mask,
    target_inst,
    args,
    out_dir,
):
    return _clear_policy_impl.infer_target_amodal_occlusion(
        bgr,
        target_mask,
        target_inst,
        args,
        out_dir,
        mask_bbox_hook=_mask_bbox,
    )


def amodal_candidate_block_metrics(candidate_mask, target_occlusion, args):
    return _clear_policy_impl.amodal_candidate_block_metrics(
        candidate_mask,
        target_occlusion,
        args,
    )


def clear_candidate_is_actionable(row, args):
    return _clear_policy_impl.clear_candidate_is_actionable(row, args)


def _corridor_block_score_camera(
    xyz_img,
    obj_mask,
    target_center_cam,
    radius,
    depth_margin,
):
    return _clear_policy_impl._corridor_block_score_camera(
        xyz_img,
        obj_mask,
        target_center_cam,
        radius,
        depth_margin,
    )


def detect_clear_obstacle_candidates(
    bgr,
    xyz_img,
    target_mask,
    target_inst,
    args,
    out_dir,
):
    return _clear_policy_impl.detect_clear_obstacle_candidates(
        bgr,
        xyz_img,
        target_mask,
        target_inst,
        args,
        out_dir,
        infer_yolo_instances_hook=infer_yolo_instances,
        infer_target_amodal_occlusion_hook=infer_target_amodal_occlusion,
        median_depth_for_mask_hook=_median_depth_for_mask,
        mask_bbox_hook=_mask_bbox,
        expand_bbox_xyxy_hook=_expand_bbox_xyxy,
        bbox_mask_hook=_bbox_mask,
        mask_iou_hook=_mask_iou,
        mask_for_instance_hook=mask_for_instance,
        mask_center_hook=_mask_center,
        corridor_block_score_camera_hook=_corridor_block_score_camera,
        amodal_candidate_block_metrics_hook=amodal_candidate_block_metrics,
        clear_candidate_is_actionable_hook=clear_candidate_is_actionable,
        largest_visible_clear_instance_hook=largest_visible_clear_instance,
        target_occlusion_report_hook=_target_occlusion_report,
        save_clear_obstacle_overlay_hook=save_clear_obstacle_overlay,
    )


def save_clear_obstacle_overlay(
    bgr,
    target_mask,
    target_region,
    rows,
    out_dir,
    args,
    target_occlusion=None,
    recommended_action=None,
):
    return _clear_policy_impl.save_clear_obstacle_overlay(
        bgr,
        target_mask,
        target_region,
        rows,
        out_dir,
        args,
        target_occlusion=target_occlusion,
        recommended_action=recommended_action,
        mask_bbox_hook=_mask_bbox,
    )


def _candidate_to_jsonable(row):
    return _grasp_candidates_impl._candidate_to_jsonable(
        row,
        grasp_camera_summary_hook=grasp_camera_summary,
    )


def grasp_camera_summary(grasp):
    return _grasp_candidates_impl.grasp_camera_summary(grasp)


def save_selected_grasp_pose(out_dir, grasp, T_base_camera, pre_tcp, grasp_tcp, lift_tcp, selected_row=None):
    return _grasp_candidates_impl.save_selected_grasp_pose(
        out_dir,
        grasp,
        T_base_camera,
        pre_tcp,
        grasp_tcp,
        lift_tcp,
        selected_row=selected_row,
        candidate_to_jsonable_hook=_candidate_to_jsonable,
        grasp_camera_summary_hook=grasp_camera_summary,
    )


def save_grasp_candidates_json(rows, out_dir):
    return _grasp_candidates_impl.save_grasp_candidates_json(
        rows,
        out_dir,
        candidate_to_jsonable_hook=_candidate_to_jsonable,
    )


def grasp_width_m(row):
    return _grasp_candidates_impl.grasp_width_m(
        row,
        grasp_camera_summary_hook=grasp_camera_summary,
    )


def annotate_grasp_candidate_physical_aperture(
    rows,
    target_cloud,
    T_base_camera,
    current_tcp_pose,
    args,
):
    return _grasp_candidates_impl.annotate_grasp_candidate_physical_aperture(
        rows,
        target_cloud,
        T_base_camera,
        current_tcp_pose,
        args,
        grasp_to_tcp_poses_hook=grasp_to_tcp_poses,
        make_T_from_pose_hook=_make_T_from_pose,
        axis_vector_hook=_axis_vector,
        motion_open_gripper_finger_capsules_hook=_motion_open_gripper_finger_capsules,
        point_segment_distances_hook=_point_segment_distances,
        grasp_width_m_hook=grasp_width_m,
    )


def grasp_candidate_width_shift(row, width_sign):
    return _grasp_candidates_impl.grasp_candidate_width_shift(row, width_sign)


def filter_grasp_candidates_by_width(rows, args):
    return _grasp_candidates_impl.filter_grasp_candidates_by_width(
        rows,
        args,
        grasp_width_m_hook=grasp_width_m,
    )


def filter_grasp_candidates_by_reachability(rows, strict=False):
    return _grasp_candidates_impl.filter_grasp_candidates_by_reachability(
        rows,
        strict=strict,
    )


def filter_grasp_candidates_by_workflow_preflight(rows, strict=False):
    return _grasp_candidates_impl.filter_grasp_candidates_by_workflow_preflight(
        rows,
        strict=strict,
    )


def filter_grasp_candidates_by_table_guard(rows, args=None):
    return _grasp_candidates_impl.filter_grasp_candidates_by_table_guard(
        rows,
        args,
        table_guard_enabled_hook=table_guard_enabled,
    )


def grasp_candidate_too_wide(row, args):
    return _grasp_candidates_impl.grasp_candidate_too_wide(
        row,
        args,
        grasp_width_m_hook=grasp_width_m,
    )



def _ordered_equivalent_jaw_width_signs(
    grasp,
    T_base_camera,
    current_tcp_pose,
    args,
):
    return _grasp_candidates_impl._ordered_equivalent_jaw_width_signs(
        grasp,
        T_base_camera,
        current_tcp_pose,
        args,
        graspnet_rotation_to_tcp_rotation_hook=graspnet_rotation_to_tcp_rotation,
    )


def _filter_equivalent_jaw_width_signs(
    ordered_signs,
    allowed_signs,
):
    return _grasp_candidates_impl._filter_equivalent_jaw_width_signs(
        ordered_signs,
        allowed_signs,
    )


def _candidate_preflight_width_signs(
    row,
    grasp,
    T_base_camera,
    current_tcp_pose,
    args,
):
    return _grasp_candidates_impl._candidate_preflight_width_signs(
        row,
        grasp,
        T_base_camera,
        current_tcp_pose,
        args,
        ordered_equivalent_jaw_width_signs_hook=_ordered_equivalent_jaw_width_signs,
        filter_equivalent_jaw_width_signs_hook=_filter_equivalent_jaw_width_signs,
    )


def _defer_candidate_endpoint_reachability(args):
    return _grasp_candidates_impl._defer_candidate_endpoint_reachability(args)


def annotate_grasp_candidate_reachability(rows, T_base_camera, current_tcp_pose, args, arm=None):
    return _grasp_candidates_impl.annotate_grasp_candidate_reachability(
        rows,
        T_base_camera,
        current_tcp_pose,
        args,
        arm=arm,
        ordered_equivalent_jaw_width_signs_hook=_ordered_equivalent_jaw_width_signs,
        filter_equivalent_jaw_width_signs_hook=_filter_equivalent_jaw_width_signs,
        grasp_candidate_width_shift_hook=grasp_candidate_width_shift,
        grasp_to_tcp_poses_hook=grasp_to_tcp_poses,
        assert_table_guard_target_tcp_hook=assert_table_guard_target_tcp,
        pose_rpy_error_norm_hook=_pose_rpy_error_norm,
        solve_ik_for_flange_pose_hook=_solve_ik_for_flange_pose,
        defer_candidate_endpoint_reachability_hook=_defer_candidate_endpoint_reachability,
    )


def execution_gripper_open_width(grasp, args):
    return _grasp_candidates_impl.execution_gripper_open_width(grasp, args)


def _project_camera_point(point_camera, intr, width, height):
    return _grasp_candidates_impl._project_camera_point(
        point_camera,
        intr,
        width,
        height,
    )


def _draw_grasp_pose_axes(canvas, center_camera, grasp, intr, image_width, image_height, args=None):
    return _grasp_candidates_impl._draw_grasp_pose_axes(
        canvas,
        center_camera,
        grasp,
        intr,
        image_width,
        image_height,
        args,
        axis_vector_hook=_axis_vector,
    )


def draw_grasp_candidate_overlay(bgr, rows, intr, out_dir, args=None):
    return _grasp_candidates_impl.draw_grasp_candidate_overlay(
        bgr,
        rows,
        intr,
        out_dir,
        args,
        grasp_candidate_too_wide_hook=grasp_candidate_too_wide,
        draw_grasp_pose_axes_hook=_draw_grasp_pose_axes,
        grasp_camera_summary_hook=grasp_camera_summary,
    )


def _make_grasp_lineset(grasp, color=(1.0, 0.0, 0.0), selected=False, args=None):
    o3d = get_o3d()
    R_cg = np.asarray(grasp.rotation_matrix, dtype=float).reshape(3, 3)
    t = np.asarray(grasp.translation, dtype=float).reshape(3)
    try:
        width = float(getattr(grasp, "width"))
    except Exception:
        width = 0.04
    if not np.isfinite(width) or width <= 0.0:
        width = 0.04
    width = min(width, 0.12)
    approach_len = 0.055 if not selected else 0.075
    jaw_len = 0.020 if not selected else 0.026
    approach_dir = R_cg @ _axis_vector(getattr(args, "graspnet_approach_axis", "x"))
    width_dir = R_cg @ _axis_vector(getattr(args, "graspnet_width_axis", "y"))
    left = t - width_dir * width * 0.5
    right = t + width_dir * width * 0.5
    approach = t + approach_dir * approach_len
    left_tip = left + approach_dir * jaw_len
    right_tip = right + approach_dir * jaw_len
    points = np.vstack([t, approach, left, right, left_tip, right_tip])
    lines = np.asarray([
        [0, 1],  # approach direction
        [2, 3],  # gripper width
        [2, 4],
        [3, 5],
    ], dtype=np.int32)
    colors = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 0.85, 1.0],
        color,
        color,
    ], dtype=np.float64)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def preview_graspnet_pointcloud(cloud, rows, args, out_dir, selected_idx=None):
    if not (args.preview_graspnet_cloud or args.active_view or args.select_grasp_candidate):
        return
    if cloud is None or rows is None or len(rows) == 0:
        return
    o3d = get_o3d()
    cloud_vis = copy.deepcopy(cloud)
    geoms = [cloud_vis]
    line_sets = []
    colors = [
        (0.0, 0.8, 0.0),
        (1.0, 0.6, 0.0),
        (0.0, 0.4, 1.0),
        (0.8, 0.0, 1.0),
        (0.0, 0.8, 0.8),
        (0.8, 0.8, 0.0),
        (0.2, 0.2, 1.0),
        (1.0, 0.2, 0.2),
    ]
    for display_i, row in enumerate(list(rows)[: max(int(args.manual_topk), 1)]):
        grasp = row.get("grasp", None)
        if grasp is None:
            continue
        idx = int(row.get("idx", display_i))
        selected = selected_idx is not None and idx == int(selected_idx)
        line_set = _make_grasp_lineset(grasp, colors[display_i % len(colors)], selected=selected, args=args)
        line_sets.append(line_set)
        geoms.append(line_set)
        center = np.asarray(grasp.translation, dtype=float).reshape(3)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0045 if not selected else 0.007)
        sphere.translate(center)
        sphere.paint_uniform_color([1.0, 1.0, 0.0] if selected else list(colors[display_i % len(colors)]))
        geoms.append(sphere)
    if args.preview_cloud_frame:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08))
    print("[grasp] showing GraspNet 3D candidates on point cloud.")
    print("[grasp] red line=approach direction; cyan line=gripper width.")
    save_cloud(os.path.join(out_dir, "graspnet_display_cloud.ply"), cloud_vis)
    if line_sets:
        merged_lines = o3d.geometry.LineSet()
        merged_points = []
        merged_edges = []
        merged_colors = []
        point_offset = 0
        for line_set in line_sets:
            pts = np.asarray(line_set.points)
            lines = np.asarray(line_set.lines, dtype=np.int32)
            cols = np.asarray(line_set.colors)
            merged_points.append(pts)
            merged_edges.append(lines + point_offset)
            merged_colors.append(cols)
            point_offset += len(pts)
        merged_lines.points = o3d.utility.Vector3dVector(np.vstack(merged_points))
        merged_lines.lines = o3d.utility.Vector2iVector(np.vstack(merged_edges))
        merged_lines.colors = o3d.utility.Vector3dVector(np.vstack(merged_colors))
        line_path = os.path.join(out_dir, "graspnet_candidate_grippers.ply")
        try:
            o3d.io.write_line_set(line_path, merged_lines)
            print(f"[grasp] saved GraspNet 3D gripper lines={line_path}")
        except Exception as exc:
            print(f"[grasp][warn] failed to save 3D gripper lines: {type(exc).__name__}: {exc}")
    try:
        draw_open3d_preview(
            geoms,
            "GraspNet candidates on point cloud",
            args,
            width=1280,
            height=720,
        )
    except (SafetyStopTriggered, ControlledMotionStopTriggered):
        raise
    except Exception as exc:
        print(f"[grasp][warn] GraspNet point-cloud preview unavailable: {type(exc).__name__}: {exc}")


def _make_execution_gripper_lineset(T_camera_tcp, opening, args):
    o3d = get_o3d()
    T_camera_tcp = np.asarray(T_camera_tcp, dtype=float).reshape(4, 4)
    rotation = T_camera_tcp[:3, :3]
    center = T_camera_tcp[:3, 3]

    width_axis = rotation @ _axis_vector(args.tcp_gripper_width_axis)
    width_axis /= np.linalg.norm(width_axis) + 1e-12

    # Use the collision model's actual centerlines.  Opening is the free gap,
    # so each centerline lies one finger radius outside half that gap.
    fingers = _motion_open_gripper_finger_capsules(
        T_camera_tcp, args, opening=opening
    )
    left_back, left_front, _, _ = fingers[0]
    right_back, right_front, _, _ = fingers[1]
    left = center + width_axis * np.dot(left_back - center, width_axis)
    right = center + width_axis * np.dot(right_back - center, width_axis)
    points = np.vstack(
        [
            center,
            left,
            right,
            left_back,
            left_front,
            right_back,
            right_front,
        ]
    )
    lines = np.asarray(
        # Draw the schematic palm on the proximal side.  A bridge at the TCP
        # looked like a solid bar passing through the target contact region.
        [[3, 5], [1, 3], [1, 4], [2, 5], [2, 6]],
        dtype=np.int32,
    )
    colors = np.tile(
        np.asarray([[1.0, 0.75, 0.0]], dtype=float),
        (lines.shape[0], 1),
    )
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(points)
    geometry.lines = o3d.utility.Vector2iVector(lines)
    geometry.colors = o3d.utility.Vector3dVector(colors)
    return geometry


def preview_final_grasp_candidate(
    cloud,
    grasp,
    selected_row,
    T_base_camera,
    pre_tcp,
    grasp_tcp,
    selected_width_sign,
    selected_width_shift,
    args,
    out_dir,
):
    if not bool(getattr(args, "preview_final_grasp", True)):
        return True
    if cloud is None or grasp is None:
        print("[grasp][final-preview][error] missing cloud or selected grasp")
        return False

    try:
        o3d = get_o3d()
        T_camera_base = np.linalg.inv(
            np.asarray(T_base_camera, dtype=float).reshape(4, 4)
        )
        T_camera_pre = T_camera_base @ _make_T_from_pose(pre_tcp)
        T_camera_grasp = T_camera_base @ _make_T_from_pose(grasp_tcp)
        opening = min(
            float(getattr(args, "gripper_open_width", 0.100)),
            float(getattr(args, "gripper_hardware_max_width", 0.100)),
        )

        final_gripper = _make_execution_gripper_lineset(
            T_camera_grasp,
            opening,
            args,
        )
        raw_grasp = _make_grasp_lineset(
            grasp,
            color=(0.0, 0.85, 0.0),
            selected=True,
            args=args,
        )
        genx_mesh = None
        if (selected_row is not None and selected_row.get("generator") == "graspgenx") or getattr(args, "nero_grasp_refine", False):
            from .graspgenx_integration import canonical_to_tcp, open_gripper_mesh
            genx_mesh = open_gripper_mesh(args)
            T_camera_asset = T_camera_grasp @ np.linalg.inv(canonical_to_tcp(args))
            genx_mesh.transform(T_camera_asset)
            genx_mesh.paint_uniform_color([0.08, 0.72, 0.85])
            mesh_name = ("final_graspgenx_open_gripper.obj" if selected_row is not None
                         and selected_row.get("generator") == "graspgenx" else "final_nero_open_gripper.obj")
            o3d.io.write_triangle_mesh(os.path.join(out_dir, mesh_name), genx_mesh)

        approach_path = o3d.geometry.LineSet()
        approach_path.points = o3d.utility.Vector3dVector(
            np.vstack([T_camera_pre[:3, 3], T_camera_grasp[:3, 3]])
        )
        approach_path.lines = o3d.utility.Vector2iVector(
            np.asarray([[0, 1]], dtype=np.int32)
        )
        approach_path.colors = o3d.utility.Vector3dVector(
            np.asarray([[0.8, 0.1, 1.0]], dtype=float)
        )

        tcp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.065
        )
        tcp_frame.transform(T_camera_grasp)

        pre_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        pre_sphere.translate(T_camera_pre[:3, 3])
        pre_sphere.paint_uniform_color([0.1, 0.4, 1.0])

        grasp_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        grasp_sphere.translate(T_camera_grasp[:3, 3])
        grasp_sphere.paint_uniform_color([1.0, 0.85, 0.0])

        selected_idx = (
            None
            if selected_row is None
            else int(selected_row.get("idx", 0))
        )
        metrics = {} if selected_row is None else selected_row
        print(
            f"[grasp][final-preview] candidate={selected_idx} "
            f"final_score={float(metrics.get('recommendation_score', 0.0)):.4f} "
            f"raw={float(metrics.get('raw_score', 0.0)):.4f} "
            f"bilateral={float(metrics.get('recommended_bilateral_score', 0.0)):.4f} "
            f"left/right="
            f"{metrics.get('recommended_left_contact_points', None)}/"
            f"{metrics.get('recommended_right_contact_points', None)} "
            f"route={float(metrics.get('recommended_route_score', 0.0)):.4f} "
            f"jaw_delta={float(metrics.get('jaw_selected_orientation_delta_deg', 0.0)):.1f}deg "
            f"width_sign={int(selected_width_sign):+d} "
            f"recenter={float(selected_width_shift) * 1000.0:.1f}mm"
        )
        print(
            "[grasp][final-preview] yellow=execution finger centerlines "
            "(bridge at rear; capsule radii not drawn), "
            f"green={str(metrics.get('generator', 'GraspNet'))} original candidate "
            "before execution advance/recentering; cyan line=candidate width, "
            "purple=pregrasp path, blue/yellow spheres=pregrasp/grasp TCP; "
            "cyan mesh=real Nero gripper used for planning."
        )
        axis_diagnostics = grasp_tcp_axis_diagnostics(
            grasp, T_base_camera, args, grasp_tcp,
            width_sign=selected_width_sign,
        )
        print(
            "[grasp][final-preview] base TCP xyz(m)="
            f"{np.round(np.asarray(grasp_tcp)[:3], 5).tolist()} "
            "rpy(deg)="
            f"{np.round(np.rad2deg(np.asarray(grasp_tcp)[3:]), 2).tolist()} "
            "approach(base)="
            f"{np.round(axis_diagnostics['tcp_approach_base'], 4).tolist()} "
            "GraspNet/TCP approach angle="
            f"{axis_diagnostics['approach_angle_deg']:.3f}deg"
        )
        print(
            "[grasp][final-preview] close the window to return to the "
            "terminal confirmation; closing does not execute automatically."
        )

        final_path = os.path.join(
            out_dir,
            "final_selected_execution_gripper.ply",
        )
        o3d.io.write_line_set(final_path, final_gripper)
        preview_metadata = {
            "frame": "camera0",
            "T_base_camera0": np.asarray(T_base_camera, dtype=float).tolist(),
            "T_camera0_tcp": T_camera_grasp.tolist(),
            "T_camera0_pregrasp": T_camera_pre.tolist(),
            "grasp_tcp_base": np.asarray(grasp_tcp, dtype=float).tolist(),
            "opening_m": opening,
            "finger_radius_m": max(float(getattr(args, "motion_path_open_finger_radius", 0.007)), 0.003),
            "graspnet_approach_base": axis_diagnostics["grasp_approach_base"].tolist(),
            "tcp_approach_base": axis_diagnostics["tcp_approach_base"].tolist(),
            "approach_angle_deg": axis_diagnostics["approach_angle_deg"],
            "generator": str(metrics.get("generator", "graspnet")),
        }
        with open(os.path.join(out_dir, "final_grasp_preview.json"), "w", encoding="utf-8") as stream:
            json.dump(preview_metadata, stream, indent=2)
        draw_open3d_preview(
            [
                copy.deepcopy(cloud),
                raw_grasp,
                final_gripper,
                approach_path,
                pre_sphere,
                grasp_sphere,
                tcp_frame,
            ] + ([] if genx_mesh is None else [genx_mesh]),
            f"FINAL grasp candidate {selected_idx} - close to review",
            args,
            width=1280,
            height=720,
        )
        return True
    except (SafetyStopTriggered, ControlledMotionStopTriggered):
        raise
    except Exception as exc:
        print(
            "[grasp][final-preview][error] "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def select_grasp_candidate(rows, bgr, intr, out_dir, args):
    return _grasp_candidates_impl.select_grasp_candidate(
        rows,
        bgr,
        intr,
        out_dir,
        args,
        save_grasp_candidates_json_hook=save_grasp_candidates_json,
        filter_grasp_candidates_by_width_hook=filter_grasp_candidates_by_width,
        filter_grasp_candidates_by_table_guard_hook=filter_grasp_candidates_by_table_guard,
        filter_grasp_candidates_by_reachability_hook=filter_grasp_candidates_by_reachability,
        filter_grasp_candidates_by_workflow_preflight_hook=filter_grasp_candidates_by_workflow_preflight,
        defer_candidate_endpoint_reachability_hook=_defer_candidate_endpoint_reachability,
        draw_grasp_candidate_overlay_hook=draw_grasp_candidate_overlay,
        grasp_camera_summary_hook=grasp_camera_summary,
        grasp_candidate_too_wide_hook=grasp_candidate_too_wide,
        grasp_width_m_hook=grasp_width_m,
    )


def choose_default_grasp_candidate(rows, gg, args):
    return _grasp_candidates_impl.choose_default_grasp_candidate(
        rows,
        gg,
        args,
        filter_grasp_candidates_by_width_hook=filter_grasp_candidates_by_width,
        filter_grasp_candidates_by_table_guard_hook=filter_grasp_candidates_by_table_guard,
        defer_candidate_endpoint_reachability_hook=_defer_candidate_endpoint_reachability,
        filter_grasp_candidates_by_reachability_hook=filter_grasp_candidates_by_reachability,
        filter_grasp_candidates_by_workflow_preflight_hook=filter_grasp_candidates_by_workflow_preflight,
    )


def camera_self_occlusion_mask(image_shape, args):
    return _perception_impl.camera_self_occlusion_mask(image_shape, args)


def apply_camera_self_occlusion_mask(xyz_img, bgr, args, out_dir):
    return _perception_impl.apply_camera_self_occlusion_mask(
        xyz_img,
        bgr,
        args,
        out_dir,
        self_occlusion_mask_hook=camera_self_occlusion_mask,
    )


def camera_near_field_ghost_mask(xyz_img, args, preserve_mask=None):
    return _perception_impl.camera_near_field_ghost_mask(
        xyz_img,
        args,
        preserve_mask=preserve_mask,
    )


def apply_camera_near_field_ghost_filter(
    xyz_img,
    bgr,
    args,
    out_dir=None,
    preserve_mask=None,
):
    return _perception_impl.apply_camera_near_field_ghost_filter(
        xyz_img,
        bgr,
        args,
        out_dir=out_dir,
        preserve_mask=preserve_mask,
        ghost_mask_hook=camera_near_field_ghost_mask,
    )


def capture_rgbd_xyz_frame(args):
    return _perception_impl.capture_rgbd_xyz_frame(
        args,
        capture_rgbd_hook=capture_rgbd,
        capture_ros_color_pointcloud_frame_hook=capture_ros_color_pointcloud_frame,
        pointcloud2_xyz_image_hook=pointcloud2_xyz_image,
        depth_to_xyz_image_hook=depth_to_xyz_image,
        apply_camera_self_occlusion_mask_hook=apply_camera_self_occlusion_mask,
    )


def capture_target_cloud(args):
    return _perception_impl.capture_target_cloud(
        args,
        apply_target_prompt_hook=apply_target_prompt,
        capture_rgbd_xyz_frame_hook=capture_rgbd_xyz_frame,
        select_target_mask_hook=select_target_mask,
        apply_near_field_filter_hook=apply_camera_near_field_ghost_filter,
        refine_xyz_inside_visible_mask_hook=refine_xyz_inside_visible_mask,
        cloud_from_mask_hook=cloud_from_mask,
        scene_cloud_from_xyz_hook=scene_cloud_from_xyz,
        obstacle_cloud_from_non_target_mask_hook=obstacle_cloud_from_non_target_mask,
    )


def capture_target_cloud_with_retries(args):
    return _perception_impl.capture_target_cloud_with_retries(
        args,
        capture_target_cloud_hook=capture_target_cloud,
        sleep_hook=time.sleep,
    )


def recapture_original_target_after_clear(
    args,
    reference_mask,
    reference_inst,
):
    return _perception_impl.recapture_original_target_after_clear(
        args,
        reference_mask,
        reference_inst,
        capture_target_cloud_with_retries_hook=capture_target_cloud_with_retries,
    )


def capture_active_projected_target_cloud(
    args,
    first_inst,
    T_base_camera,
    target_center_base,
    view_label,
):
    return _perception_impl.capture_active_projected_target_cloud(
        args,
        first_inst,
        T_base_camera,
        target_center_base,
        view_label,
        active_view_second_capture_args_hook=_active_view_second_capture_args,
        capture_rgbd_xyz_frame_hook=capture_rgbd_xyz_frame,
        mask_from_projected_target_center_hook=mask_from_projected_target_center,
        mask_from_clicked_target_point_hook=mask_from_clicked_target_point,
        infer_yolo_instances_hook=infer_yolo_instances,
        class_matches_hook=_class_matches,
        mask_for_instance_hook=mask_for_instance,
        deduplicate_identity_candidates_hook=_deduplicate_strict_identity_candidates,
        save_selected_visuals_hook=save_selected_target_visuals,
        apply_near_field_filter_hook=apply_camera_near_field_ghost_filter,
        refine_xyz_inside_visible_mask_hook=refine_xyz_inside_visible_mask,
        cloud_from_mask_hook=cloud_from_mask,
        scene_cloud_from_xyz_hook=scene_cloud_from_xyz,
        obstacle_cloud_from_non_target_mask_hook=obstacle_cloud_from_non_target_mask,
    )


def preview_reconstructed_cloud(target_cloud, scene_cloud, args):
    if not args.preview_cloud:
        return
    o3d = get_o3d()
    scene_vis = copy.deepcopy(scene_cloud)
    target_vis = copy.deepcopy(target_cloud)
    target_vis.paint_uniform_color([1.0, 0.05, 0.05])
    geoms = [scene_vis, target_vis]
    if args.preview_cloud_frame:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08))
    print("[cloud] showing this capture's point cloud. Close the Open3D window to continue perception and planning; the grasp generator has not run yet.")
    print("[cloud] target cloud is red; scene cloud keeps RGB colors.")
    try:
        draw_open3d_preview(
            geoms,
            "Reconstructed RGB-D cloud: red=selected target",
            args,
            width=1280,
            height=720,
        )
    except (SafetyStopTriggered, ControlledMotionStopTriggered):
        raise
    except Exception as exc:
        print(f"[cloud][warn] Open3D preview unavailable: {type(exc).__name__}: {exc}")


def merge_clouds(
    clouds,
    voxel_size=0.0015,
    filter_outliers=True,
    outlier_neighbors=20,
    outlier_std_ratio=2.0,
):
    return _pointcloud_fusion_impl.merge_clouds(
        clouds,
        voxel_size=voxel_size,
        filter_outliers=filter_outliers,
        outlier_neighbors=outlier_neighbors,
        outlier_std_ratio=outlier_std_ratio,
        get_o3d_hook=get_o3d,
    )


def fuse_target_clouds(
    clouds,
    voxel_size=0.0025,
    **kwargs,
):
    """Runtime facade for target-only confidence-weighted fusion."""
    return _pointcloud_fusion_impl.fuse_target_clouds(
        clouds,
        voxel_size=voxel_size,
        get_o3d_hook=get_o3d,
        **kwargs,
    )


def save_cloud(path, cloud):
    return _pointcloud_fusion_impl.save_cloud(
        path,
        cloud,
        get_o3d_hook=get_o3d,
    )


def _downsample_points(points, voxel):
    return _pointcloud_fusion_impl._downsample_points(
        points,
        voxel,
        get_o3d_hook=get_o3d,
    )


def _small_translation_registration(src_pts, tgt_pts, args):
    return _pointcloud_fusion_impl._small_translation_registration(
        src_pts,
        tgt_pts,
        args,
        nn_stats_hook=_nn_stats,
        apply_T_points_hook=_apply_T_points,
    )


def _rigid_icp_registration(src_pts, tgt_pts, args):
    return _pointcloud_fusion_impl._rigid_icp_registration(
        src_pts,
        tgt_pts,
        args,
        get_o3d_hook=get_o3d,
        nn_stats_hook=_nn_stats,
        nn_gate_hook=_nn_gate,
        apply_T_points_hook=_apply_T_points,
    )


def register_active_fusion_views(
    view_names,
    target_base_clouds,
    scene_base_clouds,
    T_base_camera0,
    out_dir,
    args,
):
    return _pointcloud_fusion_impl.register_active_fusion_views(
        view_names,
        target_base_clouds,
        scene_base_clouds,
        T_base_camera0,
        out_dir,
        args,
        points_from_cloud_hook=_points_from_cloud,
        downsample_points_hook=_downsample_points,
        nn_stats_hook=_nn_stats,
        small_translation_registration_hook=_small_translation_registration,
        apply_T_points_hook=_apply_T_points,
        rigid_icp_registration_hook=_rigid_icp_registration,
        transform_cloud_hook=transform_cloud,
        save_cloud_hook=save_cloud,
    )


def _active_side_capture_overlap_ok(
    reference_base_cloud,
    side_base_cloud,
    args,
    label,
    *,
    reference_scene_base_cloud=None,
    side_scene_base_cloud=None,
):
    return _pointcloud_fusion_impl._active_side_capture_overlap_ok(
        reference_base_cloud,
        side_base_cloud,
        args,
        label,
        reference_scene_base_cloud=reference_scene_base_cloud,
        side_scene_base_cloud=side_scene_base_cloud,
        points_from_cloud_hook=_points_from_cloud,
        downsample_points_hook=_downsample_points,
        small_translation_registration_hook=_small_translation_registration,
        apply_T_points_hook=_apply_T_points,
        nn_stats_hook=_nn_stats,
    )


def diagnose_active_fusion(
    view_names,
    target_base_clouds,
    T_base_camera0,
    out_dir,
    args,
):
    return _pointcloud_fusion_impl.diagnose_active_fusion(
        view_names,
        target_base_clouds,
        T_base_camera0,
        out_dir,
        args,
        cloud_stats_hook=_cloud_stats,
        points_from_cloud_hook=_points_from_cloud,
        downsample_points_hook=_downsample_points,
        nn_stats_hook=_nn_stats,
        mutual_overlap_stats_hook=_mutual_overlap_stats,
        save_cloud_hook=save_cloud,
        transform_cloud_hook=transform_cloud,
    )


def preview_fused_cloud(target_cloud, scene_cloud, args):
    return _pointcloud_fusion_impl.preview_fused_cloud(
        target_cloud,
        scene_cloud,
        args,
        get_o3d_hook=get_o3d,
        draw_open3d_preview_hook=draw_open3d_preview,
    )




def T_base_camera_from_flange(arm, args):
    flange_pose = arm.get_flange_pose()
    T_base_flange = _make_T_from_pose(flange_pose)
    T_flange_camera = handeye_transform(args.calib_json, args.handeye_convention)
    return T_base_flange @ T_flange_camera


def _make_cloud_from_points_colors(points, colors=None):
    return _pointcloud_fusion_impl._make_cloud_from_points_colors(
        points,
        colors=colors,
        get_o3d_hook=get_o3d,
    )


def fuse_dynamic_environment_clouds(
    view_names,
    environment_base_clouds,
    args,
    target_base=None,
    map_id="dynamic_environment",
    generation=0,
    target_center_base=None,
):
    """Compatibility facade for the canonical base-frame environment map."""
    return _pointcloud_fusion_impl.fuse_dynamic_environment_clouds(
        view_names,
        environment_base_clouds,
        args,
        target_base=target_base,
        map_id=map_id,
        generation=generation,
        target_center_base=target_center_base,
    )


def _active_view_hooks():
    return _active_view_impl.ActiveViewHooks(
        active_motion_scene_points=_active_motion_scene_points,
        active_spherical_camera_position=_active_spherical_camera_position,
        build_active_information_proxy=_build_active_information_proxy,
        build_scene_aware_nbv_records=_build_scene_aware_nbv_records,
        direction_context_penalty=_direction_context_penalty,
        estimate_normals_toward_camera=_estimate_normals_toward_camera,
        line_segment_obstacle_metrics=_line_segment_obstacle_metrics,
        nearest_obstacle_distance=_nearest_obstacle_distance,
        normalize_dir_xy=_normalize_dir_xy,
        parse_active_orbit_tilts=_parse_active_orbit_tilts,
        prepare_active_obstacle_cloud=_prepare_active_obstacle_cloud,
        score_active_candidate_information=_score_active_candidate_information,
        solve_ik_for_flange_pose=_solve_ik_for_flange_pose,
        target_visibility_score=_target_visibility_score,
        get_ik_solver=_get_ik_solver,
        joint_limit_margin=_joint_limit_margin,
        handeye_transform=handeye_transform,
        look_at_camera_pose_flange_roll_candidates=_look_at_camera_pose_flange_roll_candidates,
        rank_active_lookat_roll_solutions=_rank_active_lookat_roll_solutions,
        spread_equivalent_roll_candidates=_spread_equivalent_roll_candidates,
        active_center_joint_state_is_verified=_active_center_joint_state_is_verified,
        active_motion_capture_base=_active_motion_capture_base,
        active_side_capture_overlap_ok=_active_side_capture_overlap_ok,
        active_side_selection_order=_active_side_selection_order,
        active_view_second_capture_args=_active_view_second_capture_args,
        build_active_view_candidates=_build_active_view_candidates,
        capture_active_projected_target_cloud=capture_active_projected_target_cloud,
        capture_target_cloud=capture_target_cloud,
        confirm_motion=confirm_motion,
        crop_scene_cloud_near_target=crop_scene_cloud_near_target,
        diagnose_active_fusion=diagnose_active_fusion,
        execute_prevalidated_stages=execute_prevalidated_stages,
        format_active_view_candidate=_format_active_view_candidate,
        fuse_dynamic_environment_clouds=fuse_dynamic_environment_clouds,
        fuse_target_clouds=fuse_target_clouds,
        merge_clouds=merge_clouds,
        modeled_tcp_pose_for_joints=_modeled_tcp_pose_for_joints,
        move_flange_pose=move_flange_pose,
        move_tcp_pose=move_tcp_pose,
        plan_active_direct_route=plan_active_direct_route,
        plan_active_up_and_over_route=plan_active_up_and_over_route,
        points_from_cloud=_points_from_cloud,
        preview_fused_cloud=preview_fused_cloud,
        print_active_view_candidates=_print_active_view_candidates,
        register_active_fusion_views=register_active_fusion_views,
        save_cloud=save_cloud,
        should_use_active_multiview=_should_use_active_multiview,
        single_view_active_result=_single_view_active_result,
        smallest_angle_diff_deg=_smallest_angle_diff_deg,
        target_center_base_from_cloud=_target_center_base_from_cloud,
        target_center_overhead_flange_pose_candidates=_target_center_overhead_flange_pose_candidates,
        T_base_camera_from_flange=T_base_camera_from_flange,
        transform_cloud=transform_cloud,
        urdf_chain_geometry=_urdf_chain_geometry,
    )


def _candidate_pose_center_norm(T_base_camera, target_center_base):
    return _active_view_impl._candidate_pose_center_norm(
        T_base_camera,
        target_center_base,
    )


def _look_at_camera_pose(cam_pos_base, target_base):
    return _active_view_impl._look_at_camera_pose(cam_pos_base, target_base)


def _rotation_error_deg_between_poses(pose_a, pose_b):
    return _active_view_impl._rotation_error_deg_between_poses(pose_a, pose_b)


def _look_at_camera_pose_flange_roll_candidates(
    cam_pos_base,
    target_base,
    T_camera_flange,
    current_flange_pose,
    roll_samples=72,
):
    return _active_view_impl._look_at_camera_pose_flange_roll_candidates(
        cam_pos_base,
        target_base,
        T_camera_flange,
        current_flange_pose,
        roll_samples=roll_samples,
    )


def _look_at_camera_pose_minimize_flange_rotation(
    cam_pos_base,
    target_base,
    T_camera_flange,
    current_flange_pose,
    roll_samples=72,
):
    return _active_view_impl._look_at_camera_pose_minimize_flange_rotation(
        cam_pos_base,
        target_base,
        T_camera_flange,
        current_flange_pose,
        roll_samples=roll_samples,
    )


def _spread_equivalent_roll_candidates(candidates, limit):
    return _active_view_impl._spread_equivalent_roll_candidates(
        candidates,
        limit,
    )


def _rank_active_lookat_roll_solutions(
    arm,
    args,
    cam_pos,
    target_center,
    T_camera_flange,
    current_flange,
    current_joints,
    label,
    roll_plan_limit=None,
    quick_seed_limit=None,
    allow_full_retry=True,
):
    return _active_view_impl._rank_active_lookat_roll_solutions(
        arm,
        args,
        cam_pos,
        target_center,
        T_camera_flange,
        current_flange,
        current_joints,
        label,
        roll_plan_limit=roll_plan_limit,
        quick_seed_limit=quick_seed_limit,
        allow_full_retry=allow_full_retry,
        hooks=_active_view_hooks(),
    )


def _rot_z(angle_rad):
    return _active_view_impl._rot_z(angle_rad)


def _limit_flange_orientation(
    current_flange_pose,
    desired_flange_pose,
    max_delta_deg,
):
    return _active_view_impl._limit_flange_orientation(
        current_flange_pose,
        desired_flange_pose,
        max_delta_deg,
    )


def _estimate_normals_toward_camera(points_base, camera_pos_base):
    return _active_view_scoring_impl._estimate_normals_toward_camera(
        points_base,
        camera_pos_base,
        get_o3d_hook=get_o3d,
    )




from .nbv_geometry import (
    _prepare_active_obstacle_cloud, _build_scene_aware_nbv_records,
    _line_segment_obstacle_metrics, _nearest_obstacle_distance,
)


def _active_view_second_capture_args(args, first_inst):
    return _perception_impl._active_view_second_capture_args(args, first_inst)


def _target_center_base_from_cloud(target_cloud, T_base_camera):
    return _active_view_impl._target_center_base_from_cloud(
        target_cloud,
        T_base_camera,
    )


def _should_use_active_multiview(
    target_cloud,
    scene_cloud,
    T_base_camera,
    args,
    obstacle_cloud=None,
    target_center_override=None,
):
    return _active_view_impl._should_use_active_multiview(
        target_cloud,
        scene_cloud,
        T_base_camera,
        args,
        obstacle_cloud=obstacle_cloud,
        target_center_override=target_center_override,
        hooks=_active_view_hooks(),
    )


def _target_center_overhead_flange_pose_candidates(
    args,
    target_center_base,
    current_flange_pose=None,
):
    return _active_view_impl._target_center_overhead_flange_pose_candidates(
        args,
        target_center_base,
        current_flange_pose=current_flange_pose,
        hooks=_active_view_hooks(),
    )


def _target_center_overhead_flange_pose(
    args,
    target_center_base,
    current_flange_pose=None,
):
    return _active_view_impl._target_center_overhead_flange_pose(
        args,
        target_center_base,
        current_flange_pose=current_flange_pose,
        hooks=_active_view_hooks(),
    )


def _urdf_chain_geometry(joint_angles, args):
    return _collision_impl._urdf_chain_geometry(
        joint_angles,
        args,
        load_urdf_joint_chain_hook=_load_urdf_joint_chain,
    )


def _joint_path_tcp_z_profile(q_start, q_target, args):
    return _collision_impl._joint_path_tcp_z_profile(
        q_start,
        q_target,
        args,
        urdf_chain_geometry_hook=_urdf_chain_geometry,
    )




def _preflight_initial_table_path(arm, target_joints, args):
    """Reject an unsafe initial move before any motion command is transmitted."""
    if not table_guard_enabled(args):
        return None
    min_z = table_guard_min_tcp_z(args)
    try:
        current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
        current_tcp = np.asarray(arm.get_tcp_pose(), dtype=float).reshape(6)
    except Exception as exc:
        raise UnsafeMotionTarget(
            f"init table preflight could not read current robot state: "
            f"{type(exc).__name__}: {exc}; no motion command was sent"
        ) from exc
    live_z = float(current_tcp[2])
    if not np.isfinite(live_z):
        raise UnsafeMotionTarget(
            "init table preflight received a non-finite current TCP height; "
            "no motion command was sent"
        )
    if live_z < min_z:
        raise UnsafeMotionTarget(
            f"init starts below the table guard: current TCP z={live_z:.4f}m < "
            f"min_tcp_z={min_z:.4f}m; no motion command was sent. "
            "Use a manually supervised upward recovery before running the workflow."
        )

    metrics = _joint_path_tcp_z_profile(current_joints, target_joints, args)
    fk_live_tol = max(
        float(getattr(args, "initial_table_fk_live_z_tol", 0.020)),
        0.001,
    )
    fk_live_error = abs(float(metrics["start_z"]) - live_z)
    if fk_live_error > fk_live_tol:
        raise UnsafeMotionTarget(
            f"init table preflight FK/live TCP z mismatch={fk_live_error:.4f}m > "
            f"{fk_live_tol:.4f}m; no motion command was sent"
        )
    max_joint_delta = float(
        getattr(args, "initial_path_max_joint_delta_deg", 95.0)
    )
    if float(metrics["max_joint_delta_deg"]) > max_joint_delta:
        raise UnsafeMotionTarget(
            f"init path largest joint sweep={metrics['max_joint_delta_deg']:.2f}deg > "
            f"{max_joint_delta:.2f}deg; no motion command was sent"
        )
    if float(metrics["target_z"]) < min_z or float(metrics["min_z"]) < min_z:
        raise UnsafeMotionTarget(
            f"init sampled joint path reaches TCP z={metrics['min_z']:.4f}m "
            f"at sample {metrics['min_sample']}/{metrics['samples'] - 1}, below "
            f"min_tcp_z={min_z:.4f}m; no motion command was sent"
        )
    print(
        f"[init][table_preflight] current_z={live_z:.4f}m "
        f"modeled_start_z={metrics['start_z']:.4f}m "
        f"path_min_z={metrics['min_z']:.4f}m target_z={metrics['target_z']:.4f}m "
        f"samples={metrics['samples']} min_allowed={min_z:.4f}m"
    )
    return metrics


def _camera_collision_capsules(T_base_flange, args, body_radius):
    return _collision_impl._camera_collision_capsules(
        T_base_flange,
        args,
        body_radius,
        handeye_transform_hook=handeye_transform,
    )


def _active_robot_capsules(joint_angles, args):
    return _collision_impl._active_robot_capsules(
        joint_angles,
        args,
        urdf_chain_geometry_hook=_urdf_chain_geometry,
        camera_collision_capsules_hook=_camera_collision_capsules,
    )


def _tcp_transform_from_joints(joint_angles, args):
    return _collision_impl._tcp_transform_from_joints(
        joint_angles,
        args,
        urdf_chain_geometry_hook=_urdf_chain_geometry,
        make_T_from_pose_hook=_make_T_from_pose,
    )


def _motion_open_gripper_finger_capsules(
    T_base_tcp,
    args,
    *,
    opening=None,
    primitive_prefix="gripper_finger",
    clamp_to_hardware=True,
    overlap_centerlines=False,
):
    return _collision_impl._motion_open_gripper_finger_capsules(
        T_base_tcp,
        args,
        opening=opening,
        primitive_prefix=primitive_prefix,
        clamp_to_hardware=clamp_to_hardware,
        overlap_centerlines=overlap_centerlines,
        axis_vector_hook=_axis_vector,
    )


def _motion_robot_capsules(
    joint_angles,
    args,
    open_gripper=False,
    *,
    closed_gripper_width=None,
):
    return _collision_impl._motion_robot_capsules(
        joint_angles,
        args,
        open_gripper=open_gripper,
        closed_gripper_width=closed_gripper_width,
        urdf_chain_geometry_hook=_urdf_chain_geometry,
        make_T_from_pose_hook=_make_T_from_pose,
        open_gripper_finger_capsules_hook=_motion_open_gripper_finger_capsules,
        camera_collision_capsules_hook=_camera_collision_capsules,
    )


def _motion_self_filter_capsules(joint_angles, args):
    return _collision_impl._motion_self_filter_capsules(
        joint_angles,
        args,
        urdf_chain_geometry_hook=_urdf_chain_geometry,
        make_T_from_pose_hook=_make_T_from_pose,
        axis_vector_hook=_axis_vector,
        handeye_transform_hook=handeye_transform,
    )


def filter_motion_robot_self_points(points, joint_angles, args, label):
    return _collision_impl.filter_motion_robot_self_points(
        points,
        joint_angles,
        args,
        label,
        motion_self_filter_capsules_hook=_motion_self_filter_capsules,
        point_segment_distances_hook=_point_segment_distances,
    )




def _active_motion_capture_base(
    capture_cloud,
    T_base_camera,
    capture_joints,
    args,
    label,
):
    """Transform and self-filter one active-view motion-safety capture.

    A fused active-view safety map contains images taken with the arm in
    different configurations.  Robot pixels from a side view therefore must
    be removed with the *side-view* joints before the views are unioned.  A
    single self-filter at the later center/grasp configuration cannot remove
    those displaced robot pixels and would turn them into false obstacles.
    """
    capture_base = transform_cloud(capture_cloud, T_base_camera)
    if not bool(getattr(args, "motion_self_filter", True)):
        return capture_base
    filtered_points = filter_motion_robot_self_points(
        _points_from_cloud(capture_base),
        np.asarray(capture_joints, dtype=float).reshape(7),
        args,
        label,
    )
    # Motion occupancy does not consume RGB, so intentionally rebuild only
    # geometry instead of risking a stale point/color correspondence.
    return _make_cloud_from_points_colors(filtered_points)


def prepare_motion_safety_scene(
    target_cloud,
    motion_scene_cloud,
    T_base_camera,
    args,
    map_id="capture",
    motion_environment_cloud=None,
    current_joints=None,
    environment_map=None,
):
    return _collision_impl.prepare_motion_safety_scene(
        target_cloud,
        motion_scene_cloud,
        T_base_camera,
        args,
        map_id=map_id,
        motion_environment_cloud=motion_environment_cloud,
        current_joints=current_joints,
        environment_map=environment_map,
        fuse_dynamic_environment_clouds_hook=fuse_dynamic_environment_clouds,
        bounded_motion_points_hook=_bounded_motion_points,
        points_from_cloud_hook=_points_from_cloud,
        transform_cloud_hook=transform_cloud,
        filter_motion_robot_self_points_hook=filter_motion_robot_self_points,
    )





def _target_points_in_tcp(target_base, grasp_tcp, args):
    return _collision_impl._target_points_in_tcp(
        target_base,
        grasp_tcp,
        args,
        bounded_motion_points_hook=_bounded_motion_points,
        make_T_from_pose_hook=_make_T_from_pose,
        transform_points_hook=_transform_points,
    )


def _build_initial_support_release(scene, grasp_tcp, args):
    return _collision_impl._build_initial_support_release(
        scene,
        grasp_tcp,
        args,
        table_guard_enabled_hook=table_guard_enabled,
        bounded_motion_points_hook=_bounded_motion_points,
    )


def _motion_configuration_metrics(
    joint_angles,
    context,
    args,
    runtime=False,
    support_strict_latched=False,
    *,
    runtime_open_finger_start=False,
):
    return _collision_impl._motion_configuration_metrics(
        joint_angles,
        context,
        args,
        runtime=runtime,
        support_strict_latched=support_strict_latched,
        runtime_open_finger_start=runtime_open_finger_start,
        motion_robot_capsules_hook=_motion_robot_capsules,
        point_segment_distances_hook=_point_segment_distances,
        tcp_transform_from_joints_hook=_tcp_transform_from_joints,
        transform_points_hook=_transform_points,
        as_pose_list_hook=_as_pose_list,
        table_guard_enabled_hook=table_guard_enabled,
    )


def _motion_joint_path_metrics(q_start, q_target, context, args, *, allow_runtime_start=False):
    return _collision_impl._motion_joint_path_metrics(
        q_start,
        q_target,
        context,
        args,
        allow_runtime_start=allow_runtime_start,
        motion_configuration_metrics_hook=_motion_configuration_metrics,
    )


def validate_motion_joint_path(q_start, q_target, context, args, label, *, allow_runtime_start=False):
    return _collision_impl.validate_motion_joint_path(
        q_start,
        q_target,
        context,
        args,
        label,
        allow_runtime_start=allow_runtime_start,
        table_guard_enabled_hook=table_guard_enabled,
        table_guard_min_tcp_z_hook=table_guard_min_tcp_z,
        joint_path_tcp_z_profile_hook=_joint_path_tcp_z_profile,
        motion_joint_path_metrics_hook=_motion_joint_path_metrics,
    )





def _active_robot_path_metrics(q_start, q_target, scene_points_base, args):
    return _collision_impl._active_robot_path_metrics(
        q_start,
        q_target,
        scene_points_base,
        args,
        active_robot_capsules_hook=_active_robot_capsules,
        point_segment_distances_hook=_point_segment_distances,
    )


def _validate_active_joint_path(
    q_start,
    q_target,
    scene_points_base,
    args,
    label,
    max_joint_delta_deg,
    max_joint_l2_deg,
):
    return _collision_impl._validate_active_joint_path(
        q_start,
        q_target,
        scene_points_base,
        args,
        label,
        max_joint_delta_deg,
        max_joint_l2_deg,
        active_robot_path_metrics_hook=_active_robot_path_metrics,
    )




def _build_active_view_candidates(
    arm,
    args,
    target_cloud,
    scene_cloud,
    T_base_camera0,
    orbit_lateral_axis=None,
    orbit_geometry=None,
    target_center_override=None,
    obstacle_cloud=None,
):
    if (
        str(getattr(args, "trajectory_executor", "move_j")) == "mit"
        and not bool(getattr(args, "_planning_worker", False))
    ):
        from .planning_process import isolated_nbv
        return isolated_nbv(
            arm, args, target_cloud, scene_cloud, T_base_camera0,
            orbit_lateral_axis=orbit_lateral_axis, orbit_geometry=orbit_geometry,
            target_center_override=target_center_override, obstacles=obstacle_cloud,
        )
    return _active_view_impl._build_active_view_candidates(
        arm,
        args,
        target_cloud,
        scene_cloud,
        T_base_camera0,
        orbit_lateral_axis=orbit_lateral_axis,
        orbit_geometry=orbit_geometry,
        target_center_override=target_center_override,
        obstacle_cloud=obstacle_cloud,
        hooks=_active_view_hooks(),
    )


def _format_active_view_candidate(cand, prefix=""):
    return _active_view_impl._format_active_view_candidate(
        cand,
        prefix=prefix,
    )


def _print_active_view_candidates(candidates, args=None):
    return _active_view_impl._print_active_view_candidates(
        candidates,
        args=args,
    )


def _active_view_candidate_angle(cand):
    return _active_view_impl._active_view_candidate_angle(cand)


def _active_side_selection_order(valid, args):
    return _active_view_impl._active_side_selection_order(
        valid,
        args,
        hooks=_active_view_hooks(),
    )


def _single_view_active_result(
    target_cloud,
    scene_cloud,
    T_base_camera,
    reason,
    motion_sent=False,
    motion_environment_cloud=None,
    motion_environment_map=None,
    restore_ok=None,
):
    return _active_view_impl._single_view_active_result(
        target_cloud,
        scene_cloud,
        T_base_camera,
        reason,
        motion_sent=motion_sent,
        motion_environment_cloud=motion_environment_cloud,
        motion_environment_map=motion_environment_map,
        restore_ok=restore_ok,
    )


def _active_center_joint_state_is_verified(
    arm,
    expected_center_joints,
    args,
    context,
):
    return _active_view_impl._active_center_joint_state_is_verified(
        arm,
        expected_center_joints,
        args,
        context,
    )


def run_optional_active_view(
    arm,
    args,
    target_cloud,
    scene_cloud,
    T_base_camera0,
    first_inst,
    out_dir,
):
    return _active_view_impl.run_optional_active_view(
        arm,
        args,
        target_cloud,
        scene_cloud,
        T_base_camera0,
        first_inst,
        out_dir,
        hooks=_active_view_hooks(),
    )


def _grasp_workflow_hooks(args=None):
    return _grasp_workflow_impl.GraspWorkflowHooks(
        assert_table_guard_target_tcp=assert_table_guard_target_tcp,
        move_flange_pose=move_flange_pose,
        move_tcp_pose=move_tcp_pose,
        intermediate_tcp_pose=intermediate_tcp_pose,
        current_flange_pose_error=current_flange_pose_error,
        print_current_flange_pose_error=print_current_flange_pose_error,
        print_status=print_status,
        wait_for_joint_settle=_wait_for_joint_settle,
        arm_motion_status_snapshot=_arm_motion_status_snapshot,
        axis_vector=_axis_vector,
        candidate_preflight_width_signs=_candidate_preflight_width_signs,
        defer_candidate_endpoint_reachability=_defer_candidate_endpoint_reachability,
        recoverable_grasp_generation_failure=_recoverable_grasp_generation_failure,
        annotate_grasp_candidate_physical_aperture=annotate_grasp_candidate_physical_aperture,
        annotate_grasp_candidate_reachability=annotate_grasp_candidate_reachability,
        annotate_grasp_candidate_workflow_preflight=annotate_grasp_candidate_workflow_preflight,
        assert_grasp_detected=assert_grasp_detected,
        choose_default_grasp_candidate=choose_default_grasp_candidate,
        confirm_motion=confirm_motion,
        execute_prevalidated_stages=execute_prevalidated_stages,
        execution_gripper_open_width=execution_gripper_open_width,
        get_graspnet_helpers=get_graspnet_helpers,
        grasp_camera_summary=grasp_camera_summary,
        grasp_candidate_width_shift=grasp_candidate_width_shift,
        grasp_tcp_axis_diagnostics=grasp_tcp_axis_diagnostics,
        grasp_to_tcp_poses=grasp_to_tcp_poses,
        handle_motion_not_reached=handle_motion_not_reached,
        move_initial=move_initial,
        move_tcp_pose_with_intermediate=move_tcp_pose_with_intermediate,
        plan_up_and_over_grasp_sequence=plan_up_and_over_grasp_sequence,
        preview_graspnet_pointcloud=preview_graspnet_pointcloud,
        preview_final_grasp_candidate=preview_final_grasp_candidate,
        recover_after_empty_grasp=recover_after_empty_grasp,
        recover_after_torque_stop=recover_after_torque_stop,
        release_vision_model_caches=lambda: release_vision_model_caches(
            collect_garbage=not bool(getattr(getattr(args, "_mit_session", None), "owns_control", False))
        ),
        safe_stop=safe_stop,
        save_grasp_candidates_json=save_grasp_candidates_json,
        save_grasp_motion_plan=save_grasp_motion_plan,
        save_selected_grasp_pose=save_selected_grasp_pose,
        select_grasp_candidate=select_grasp_candidate,
        verify_empty_grasp_recovery_at_initial_observation=verify_empty_grasp_recovery_at_initial_observation,
        wait_until_grasp_close_pose_ready=wait_until_grasp_close_pose_ready,
        wait_until_gripper_open=wait_until_gripper_open,
        capture_grasp_contact_snapshot=capture_grasp_contact_snapshot,
        prepare_table_referenced_scene=prepare_table_referenced_scene,
    )


def capture_grasp_contact_snapshot(arm, args, out_dir, planned_tcp):
    from .contact_check import capture_grasp_contact_snapshot as capture_snapshot
    return capture_snapshot(arm, args, out_dir, planned_tcp, capture_rgbd_hook=capture_rgbd)


def prepare_table_referenced_scene(args, target, scene, T_base_camera, out_dir):
    from .table_reference import prepare_table_referenced_scene as prepare_reference
    return prepare_reference(args, target, scene, T_base_camera, out_dir)


def move_tcp_pose(
    arm,
    tcp_pose,
    args,
    label,
    planned_joint_target=None,
):
    return _grasp_workflow_impl.move_tcp_pose(
        arm,
        tcp_pose,
        args,
        label,
        planned_joint_target=planned_joint_target,
        hooks=_grasp_workflow_hooks(args),
    )


def intermediate_tcp_pose(current_tcp, target_tcp, args):
    return _grasp_workflow_impl.intermediate_tcp_pose(
        current_tcp,
        target_tcp,
        args,
    )


def move_tcp_pose_with_intermediate(
    arm,
    tcp_pose,
    args,
    label,
    planned_joint_target=None,
):
    return _grasp_workflow_impl.move_tcp_pose_with_intermediate(
        arm,
        tcp_pose,
        args,
        label,
        planned_joint_target=planned_joint_target,
        hooks=_grasp_workflow_hooks(),
    )


def current_flange_pose_error(arm, theoretical_flange_pose):
    return _grasp_workflow_impl.current_flange_pose_error(
        arm,
        theoretical_flange_pose,
    )


def print_current_flange_pose_error(
    arm,
    theoretical_flange_pose,
    label,
):
    return _grasp_workflow_impl.print_current_flange_pose_error(
        arm,
        theoretical_flange_pose,
        label,
        hooks=_grasp_workflow_hooks(),
    )


def wait_until_grasp_close_pose_ready(
    arm,
    theoretical_flange_pose,
    args,
):
    return _grasp_workflow_impl.wait_until_grasp_close_pose_ready(
        arm,
        theoretical_flange_pose,
        args,
        hooks=_grasp_workflow_hooks(),
    )


def _ik_runtime_hooks():
    """Resolve legacy-patchable IK dependencies at each facade call."""
    return {
        "get_solver": _get_ik_solver,
        "joint_seed_candidates": _joint_seed_candidates,
        "clip_limit": _clip_numerical_joint_limit_overshoot,
        "joint_limit_margin": _joint_limit_margin,
    }


def _ranked_ik_solution_records_for_flange_pose(
    arm,
    flange_pose,
    args,
    label,
    current_joints_override=None,
    quiet=False,
    seed_limit=None,
    top_k=None,
    dedup_tolerance_rad=None,
    early_accept_margin_rad=None,
):
    return _ik_solver_impl._ranked_ik_solution_records_for_flange_pose(
        arm,
        flange_pose,
        args,
        label,
        current_joints_override=current_joints_override,
        quiet=quiet,
        seed_limit=seed_limit,
        top_k=top_k,
        dedup_tolerance_rad=dedup_tolerance_rad,
        early_accept_margin_rad=early_accept_margin_rad,
        **_ik_runtime_hooks(),
    )


def _solve_ik_solutions_for_flange_pose(
    arm,
    flange_pose,
    args,
    label,
    current_joints_override=None,
    quiet=False,
    seed_limit=None,
    top_k=None,
    dedup_tolerance_rad=None,
):
    return _ik_solver_impl._solve_ik_solutions_for_flange_pose(
        arm,
        flange_pose,
        args,
        label,
        current_joints_override=current_joints_override,
        quiet=quiet,
        seed_limit=seed_limit,
        top_k=top_k,
        dedup_tolerance_rad=dedup_tolerance_rad,
        **_ik_runtime_hooks(),
    )


def _solve_ik_for_flange_pose(
    arm,
    flange_pose,
    args,
    label,
    current_joints_override=None,
    quiet=False,
    seed_limit=None,
):
    return _ik_solver_impl._solve_ik_for_flange_pose(
        arm,
        flange_pose,
        args,
        label,
        current_joints_override=current_joints_override,
        quiet=quiet,
        seed_limit=seed_limit,
        **_ik_runtime_hooks(),
    )




def move_flange_pose_ik_joint(arm, flange_pose, args, label, planned_joint_target=None):
    return _arm_execution_impl.move_flange_pose_ik_joint(
        arm,
        flange_pose,
        args,
        label,
        planned_joint_target=planned_joint_target,
        solve_ik_for_flange_pose_hook=_solve_ik_for_flange_pose,
        get_ik_solver_hook=_get_ik_solver,
        clip_numerical_joint_limit_overshoot_hook=(
            _clip_numerical_joint_limit_overshoot
        ),
        pose_error_hook=pose_error,
        send_joint_target_with_start_retry_hook=(
            _send_joint_target_with_start_retry
        ),
        wait_until_joints_reached_hook=wait_until_joints_reached,
        wait_until_pose_reached_hook=wait_until_pose_reached,
    )


def move_flange_pose(arm, flange_pose, args, label, planned_joint_target=None):
    return _arm_execution_impl.move_flange_pose(
        arm,
        flange_pose,
        args,
        label,
        planned_joint_target=planned_joint_target,
        move_flange_pose_ik_joint_hook=move_flange_pose_ik_joint,
        wait_until_pose_reached_hook=wait_until_pose_reached,
    )


def test_current_flange_ik(arm, args):
    flange_pose = np.asarray(arm.get_flange_pose(), dtype=float).reshape(6)
    current_joints = np.asarray(arm.get_joint_angles(), dtype=float).reshape(7)
    print("[ik-test] current_flange_pose =", np.round(flange_pose, 6).tolist())
    print("[ik-test] current_joints_deg  =", np.round(np.rad2deg(current_joints), 3).tolist())
    q = _solve_ik_for_flange_pose(arm, flange_pose, args, "ik_test_current")
    solver = _get_ik_solver(args)
    fk = np.asarray(solver.solve_fk(q), dtype=float).reshape(7)
    fk_pose = np.r_[
        fk[:3],
        R.from_quat([fk[4], fk[5], fk[6], fk[3]]).as_euler("xyz", degrees=False),
    ]
    pose_delta = fk_pose - flange_pose
    pose_delta[3:] = (pose_delta[3:] + np.pi) % (2.0 * np.pi) - np.pi
    print("[ik-test] ik_solution_deg   =", np.round(np.rad2deg(q), 3).tolist())
    print("[ik-test] fk_flange_pose    =", np.round(fk_pose, 6).tolist())
    print(
        "[ik-test] fk_vs_api_error: "
        f"xyz={np.round(pose_delta[:3], 6).tolist()}m "
        f"rpy={np.round(np.rad2deg(pose_delta[3:]), 3).tolist()}deg "
        f"norm_pos={np.linalg.norm(pose_delta[:3]):.4f}m "
        f"norm_rpy={np.rad2deg(np.linalg.norm(pose_delta[3:])):.2f}deg"
    )
    print("[ik-test] no motion command was sent")


def handle_motion_not_reached(arm, args, exc, context="grasp"):
    return _grasp_workflow_impl.handle_motion_not_reached(
        arm,
        args,
        exc,
        context=context,
        hooks=_grasp_workflow_hooks(),
    )


def _recoverable_grasp_generation_failure(exc):
    return _grasp_workflow_impl._recoverable_grasp_generation_failure(
        exc,
    )


def _check_motion_joint_margin(joints, args, label):
    return _motion_planning_impl._check_motion_joint_margin(
        joints,
        args,
        label,
        get_solver=_get_ik_solver,
        joint_limit_margin=_joint_limit_margin,
    )


def _plan_tcp_motion_stage(
    arm,
    tcp_pose,
    q_previous,
    context,
    args,
    label,
    fixed_joints=None,
):
    return _motion_planning_impl._plan_tcp_motion_stage(
        arm,
        tcp_pose,
        q_previous,
        context,
        args,
        label,
        fixed_joints=fixed_joints,
        assert_table_guard=assert_table_guard_target_tcp,
        solve_ik_solutions=_solve_ik_solutions_for_flange_pose,
        check_motion_joint_margin=_check_motion_joint_margin,
        validate_joint_path=validate_motion_joint_path,
        clip_limit=_clip_numerical_joint_limit_overshoot,
        get_solver=_get_ik_solver,
        urdf_chain_geometry=_urdf_chain_geometry,
        as_pose_list=_as_pose_list,
        make_T_from_pose=_make_T_from_pose,
        pose_error_fn=pose_error,
    )


def _plan_tcp_motion_stage_options(
    arm,
    tcp_pose,
    q_previous,
    context,
    args,
    label,
    fixed_joints=None,
):
    return _motion_planning_impl._plan_tcp_motion_stage_options(
        arm,
        tcp_pose,
        q_previous,
        context,
        args,
        label,
        fixed_joints=fixed_joints,
        assert_table_guard=assert_table_guard_target_tcp,
        solve_ik_solutions=_solve_ik_solutions_for_flange_pose,
        check_motion_joint_margin=_check_motion_joint_margin,
        validate_joint_path=validate_motion_joint_path,
        clip_limit=_clip_numerical_joint_limit_overshoot,
        get_solver=_get_ik_solver,
        urdf_chain_geometry=_urdf_chain_geometry,
        as_pose_list=_as_pose_list,
        make_T_from_pose=_make_T_from_pose,
        pose_error_fn=pose_error,
    )


def _plan_tcp_stage_sequence_beam(
    arm,
    stage_specs,
    q_start,
    args,
    label,
    beam_width=None,
    stage_option_cache=None,
):
    return _motion_planning_impl._plan_tcp_stage_sequence_beam(
        arm,
        stage_specs,
        q_start,
        args,
        label,
        beam_width=beam_width,
        stage_option_cache=stage_option_cache,
        plan_tcp_motion_stage_options=_plan_tcp_motion_stage_options,
    )


def _plan_up_and_over_route(
    arm,
    start_tcp,
    goal_tcp,
    q_start,
    scene,
    args,
    label,
    transit_phase,
    goal_phase=None,
    attached_target_tcp=None,
    placed_target_base=None,
    first_phase=None,
    goal_joints=None,
):
    return _motion_planning_impl._plan_up_and_over_route(
        arm,
        start_tcp,
        goal_tcp,
        q_start,
        scene,
        args,
        label,
        transit_phase,
        goal_phase=goal_phase,
        attached_target_tcp=attached_target_tcp,
        placed_target_base=placed_target_base,
        first_phase=first_phase,
        goal_joints=goal_joints,
        motion_route_height_candidates=_motion_route_height_candidates,
        motion_context=_motion_context,
        plan_tcp_motion_stage=_plan_tcp_motion_stage,
        interpolate_pose_fn=interpolate_pose,
        pose_rpy_error_norm=_pose_rpy_error_norm,
    )


def _plan_guarded_route(
    arm,
    start_tcp,
    goal_tcp,
    q_start,
    scene,
    args,
    label,
    transit_phase,
    goal_phase=None,
    attached_target_tcp=None,
    placed_target_base=None,
    first_phase=None,
    goal_joints=None,
):
    return _motion_planning_impl._plan_guarded_route(
        arm,
        start_tcp,
        goal_tcp,
        q_start,
        scene,
        args,
        label,
        transit_phase,
        goal_phase=goal_phase,
        attached_target_tcp=attached_target_tcp,
        placed_target_base=placed_target_base,
        first_phase=first_phase,
        goal_joints=goal_joints,
        plan_up_and_over_route=_plan_up_and_over_route,
        motion_context=_motion_context,
        plan_tcp_motion_stage=_plan_tcp_motion_stage,
        validate_joint_path=validate_motion_joint_path,
    )


def _modeled_tcp_pose_for_joints(joints, args):
    return _motion_planning_impl._modeled_tcp_pose_for_joints(
        joints,
        args,
        urdf_chain_geometry=_urdf_chain_geometry,
        as_pose_list=_as_pose_list,
        make_T_from_pose=_make_T_from_pose,
    )




def _prepare_active_route_safety_scene(
    q_start,
    scene_points_base,
    target_points_base,
    args,
    label,
    scene_capture_joints=None,
):
    return _active_motion_impl._prepare_active_route_safety_scene(
        q_start,
        scene_points_base,
        target_points_base,
        args,
        label,
        scene_capture_joints=scene_capture_joints,
        bounded_motion_points_hook=_bounded_motion_points,
        filter_motion_robot_self_points_hook=filter_motion_robot_self_points,
    )


def _subdivide_active_direct_stage(q_start, stage, args):
    return _active_motion_impl._subdivide_active_direct_stage(
        q_start,
        stage,
        args,
        urdf_chain_geometry_hook=_urdf_chain_geometry,
        tcp_pose_for_flange_pose_hook=_tcp_pose_for_flange_pose,
    )


def plan_active_direct_route(
    arm,
    goal_flange,
    q_start,
    scene_points_base,
    target_points_base,
    args,
    label,
    max_joint_delta_deg,
    max_joint_l2_deg,
    goal_joints=None,
    scene_capture_joints=None,
    initial_retreat_height=0.0,
    enforce_single_stage_return=False,
):
    return _active_motion_impl.plan_active_direct_route(
        arm,
        goal_flange,
        q_start,
        scene_points_base,
        target_points_base,
        args,
        label,
        max_joint_delta_deg,
        max_joint_l2_deg,
        goal_joints=goal_joints,
        scene_capture_joints=scene_capture_joints,
        initial_retreat_height=initial_retreat_height,
        enforce_single_stage_return=enforce_single_stage_return,
        prepare_active_route_safety_scene_hook=_prepare_active_route_safety_scene,
        tcp_pose_for_flange_pose_hook=_tcp_pose_for_flange_pose,
        modeled_tcp_pose_for_joints_hook=_modeled_tcp_pose_for_joints,
        motion_context_hook=_motion_context,
        plan_tcp_motion_stage_hook=_plan_tcp_motion_stage,
        subdivide_active_direct_stage_hook=_subdivide_active_direct_stage,
        validate_motion_joint_path_hook=validate_motion_joint_path,
        get_ik_solver_hook=_get_ik_solver,
        joint_limit_margin_hook=_joint_limit_margin,
    )


def plan_active_up_and_over_route(
    arm,
    start_tcp,
    goal_flange,
    q_start,
    scene_points_base,
    target_points_base,
    args,
    label,
    max_joint_delta_deg,
    max_joint_l2_deg,
    goal_joints=None,
    scene_capture_joints=None,
):
    return _active_motion_impl.plan_active_up_and_over_route(
        arm,
        start_tcp,
        goal_flange,
        q_start,
        scene_points_base,
        target_points_base,
        args,
        label,
        max_joint_delta_deg,
        max_joint_l2_deg,
        goal_joints=goal_joints,
        scene_capture_joints=scene_capture_joints,
        prepare_active_route_safety_scene_hook=_prepare_active_route_safety_scene,
        tcp_pose_for_flange_pose_hook=_tcp_pose_for_flange_pose,
        plan_up_and_over_route_hook=_plan_up_and_over_route,
        validate_active_joint_path_hook=_validate_active_joint_path,
    )




def plan_up_and_over_grasp_sequence(
    arm,
    pre_tcp,
    grasp_tcp,
    lift_tcp,
    place_tcp,
    target_cloud,
    motion_scene_cloud,
    T_base_camera,
    args,
    plan_label,
    motion_environment_cloud=None,
    motion_environment_map=None,
    pregrasp_route_mode="direct",
    transport_route_mode="direct",
    neutral_joints=None,
    planning_stage_cache=None,
    planning_metrics=None,
    validation_only=False,
):
    return _grasp_planner_impl.plan_up_and_over_grasp_sequence(
        arm,
        pre_tcp,
        grasp_tcp,
        lift_tcp,
        place_tcp,
        target_cloud,
        motion_scene_cloud,
        T_base_camera,
        args,
        plan_label,
        motion_environment_cloud=motion_environment_cloud,
        motion_environment_map=motion_environment_map,
        pregrasp_route_mode=pregrasp_route_mode,
        transport_route_mode=transport_route_mode,
        neutral_joints=neutral_joints,
        planning_stage_cache=planning_stage_cache,
        planning_metrics=planning_metrics,
        validation_only=bool(validation_only),
        prepare_motion_safety_scene_hook=prepare_motion_safety_scene,
        assert_pickup_payload_table_start_hook=assert_pickup_payload_table_start,
        plan_guarded_route_hook=_plan_guarded_route,
        motion_context_hook=_motion_context,
        plan_tcp_motion_stage_hook=_plan_tcp_motion_stage,
        plan_tcp_stage_sequence_beam_hook=_plan_tcp_stage_sequence_beam,
        target_points_in_tcp_hook=_target_points_in_tcp,
        build_initial_support_release_hook=_build_initial_support_release,
        validate_motion_joint_path_hook=validate_motion_joint_path,
        pose_rpy_error_norm_hook=_pose_rpy_error_norm,
        transform_points_hook=_transform_points,
        tcp_transform_from_joints_hook=_tcp_transform_from_joints,
        modeled_tcp_pose_for_joints_hook=_modeled_tcp_pose_for_joints,
    )


def annotate_grasp_candidate_workflow_preflight(
    rows,
    arm,
    args,
    target_cloud,
    motion_scene_cloud,
    T_base_camera,
    current_tcp_pose,
    plan_label,
    place_xyz=None,
    motion_environment_cloud=None,
    motion_environment_map=None,
):
    return _grasp_planner_impl.annotate_grasp_candidate_workflow_preflight(
        rows,
        arm,
        args,
        target_cloud,
        motion_scene_cloud,
        T_base_camera,
        current_tcp_pose,
        plan_label,
        place_xyz=place_xyz,
        motion_environment_cloud=motion_environment_cloud,
        motion_environment_map=motion_environment_map,
        filter_grasp_candidates_by_width_hook=filter_grasp_candidates_by_width,
        filter_grasp_candidates_by_table_guard_hook=filter_grasp_candidates_by_table_guard,
        defer_candidate_endpoint_reachability_hook=_defer_candidate_endpoint_reachability,
        filter_grasp_candidates_by_reachability_hook=filter_grasp_candidates_by_reachability,
        candidate_preflight_width_signs_hook=_candidate_preflight_width_signs,
        ordered_equivalent_jaw_width_signs_hook=_ordered_equivalent_jaw_width_signs,
        filter_equivalent_jaw_width_signs_hook=_filter_equivalent_jaw_width_signs,
        grasp_candidate_width_shift_hook=grasp_candidate_width_shift,
        grasp_to_tcp_poses_hook=grasp_to_tcp_poses,
        assert_table_guard_target_tcp_hook=assert_table_guard_target_tcp,
        plan_up_and_over_grasp_sequence_hook=plan_up_and_over_grasp_sequence,
    )




def _collision_context_for_planned_stage(plan, stage):
    return _execution_impl._collision_context_for_planned_stage(
        plan,
        stage,
        motion_context_hook=_motion_context,
    )


def execute_prevalidated_stage(
    arm,
    stage,
    expected_start_joints,
    plan,
    args,
):
    return _execution_impl.execute_prevalidated_stage(
        arm,
        stage,
        expected_start_joints,
        plan,
        args,
        collision_context_for_planned_stage_hook=(
            _collision_context_for_planned_stage
        ),
        validate_motion_joint_path_hook=validate_motion_joint_path,
        move_flange_pose_hook=move_flange_pose,
        safe_stop_hook=safe_stop,
    )


def execute_prevalidated_stages(
    arm,
    stages,
    expected_start_joints,
    plan,
    args,
):
    return _execution_impl.execute_prevalidated_stages(
        arm,
        stages,
        expected_start_joints,
        plan,
        args,
        execute_prevalidated_stage_hook=execute_prevalidated_stage,
        collision_context_for_planned_stage_hook=(
            _collision_context_for_planned_stage
        ),
        validate_motion_joint_path_hook=validate_motion_joint_path,
        check_runtime_collision_current_hook=check_runtime_collision_current,
        safe_stop_hook=safe_stop,
    )


def recover_after_torque_stop(arm, stop, args):
    return _execution_impl.recover_after_torque_stop(
        arm,
        stop,
        args,
        validate_motion_joint_path_hook=validate_motion_joint_path,
        wait_until_joints_reached_hook=wait_until_joints_reached,
    )


def save_grasp_motion_plan(out_dir, plan):
    return _execution_impl.save_grasp_motion_plan(
        out_dir,
        plan,
        save_cloud_hook=save_cloud,
        make_cloud_from_points_colors_hook=_make_cloud_from_points_colors,
    )


def plan_and_maybe_execute_grasp(
    arm,
    args,
    target_cloud,
    scene_cloud,
    T_base_camera,
    bgr,
    intr,
    out_dir,
    inst,
    plan_label="target",
    place_xyz_override=None,
    confirmation_word="GRASP",
    motion_scene_cloud=None,
    motion_environment_cloud=None,
    motion_environment_map=None,
    planning_only=False,
):
    return _grasp_workflow_impl.plan_and_maybe_execute_grasp(
        arm,
        args,
        target_cloud,
        scene_cloud,
        T_base_camera,
        bgr,
        intr,
        out_dir,
        inst,
        plan_label=plan_label,
        place_xyz_override=place_xyz_override,
        confirmation_word=confirmation_word,
        motion_scene_cloud=motion_scene_cloud,
        motion_environment_cloud=motion_environment_cloud,
        motion_environment_map=motion_environment_map,
        planning_only=planning_only,
        hooks=_grasp_workflow_hooks(args),
    )


def maybe_clear_obstacles_before_active(arm, args, captured):
    return _clear_policy_impl.maybe_clear_obstacles_before_active(
        arm,
        args,
        captured,
        detect_clear_obstacle_candidates_hook=detect_clear_obstacle_candidates,
        clear_candidate_is_actionable_hook=clear_candidate_is_actionable,
        refine_xyz_inside_visible_mask_hook=refine_xyz_inside_visible_mask,
        cloud_from_mask_hook=cloud_from_mask,
        scene_cloud_from_xyz_hook=scene_cloud_from_xyz,
        obstacle_cloud_from_non_target_mask_hook=obstacle_cloud_from_non_target_mask,
        save_selected_target_visuals_hook=save_selected_target_visuals,
        T_base_camera_from_flange_hook=T_base_camera_from_flange,
        prepare_motion_safety_scene_hook=prepare_motion_safety_scene,
        plan_and_maybe_execute_grasp_hook=plan_and_maybe_execute_grasp,
        record_workflow_motion_hook=record_workflow_motion,
        run_optional_active_view_hook=run_optional_active_view,
        recapture_original_target_after_clear_hook=recapture_original_target_after_clear,
    )


def _workflow_hooks(args=None):
    """Resolve all legacy-patchable session dependencies for one run."""
    return _workflow_impl.WorkflowHooks(
        arm_factory=RealNeroGraspArm,
        configure_amodal_clear_model=configure_amodal_clear_model, # 配置 amodal 相关模型的权重路径和输入图像尺寸
        test_current_flange_ik=test_current_flange_ik,# 检查当前法兰位姿的逆运动学求解
        prepare_initial_pose_for_capture=prepare_initial_pose_for_capture, # 准备采集前的初始位姿
        capture_target_cloud_with_retries=capture_target_cloud_with_retries,# 采集目标点云，支持重试
        preview_reconstructed_cloud=preview_reconstructed_cloud, # 预览重建点云
        T_base_camera_from_flange=T_base_camera_from_flange,# 根据法兰位姿计算相机与基座之间的变换
        maybe_clear_obstacles_before_active=maybe_clear_obstacles_before_active,# 在主动视图之前，可能清除障碍物
        run_optional_active_view=run_optional_active_view, # 按条件执行主动视角流程
        plan_and_maybe_execute_grasp=plan_and_maybe_execute_grasp,# 规划抓取，并按条件执行
        record_workflow_motion=record_workflow_motion,# 记录工作流运动
        with_workflow_motion=with_workflow_motion, # 为操作附加工作流运动相关处理
        after_grasp_lifted=(
            _servo_delivery_after_grasp # 抓取并抬起后的可选回调
            if getattr(args, "enable_servo_delivery", False)
            else None
        ),
    )


def run_single_grasp(args):
    from cameras.realsense import close_all
    try:
        _collision_impl.initialize_arm_collision_geometry(args)
        if str(getattr(args, "trajectory_executor", "move_j")) == "mit":
            # Also cover callers which use this API without main/preflight:
            # numerical-worker cold imports must precede any arm connection.
            _get_ik_solver(args)
        return _workflow_impl.run_single_grasp(args, hooks=_workflow_hooks(args))
    finally:
        close_all()


def main(argv=None):
    """Run the preserved CLI and return its historical process exit code."""
    from .camera_self_mask import CameraSelfMaskMismatch
    exit_code = 0
    args = None
    result = None
    terminal_status = None
    terminal_exception = None
    try:
        if argv is None:
            args = parse_args()
        else:
            original_argv = sys.argv
            try:
                sys.argv = [original_argv[0], *list(argv)]
                args = parse_args()
            finally:
                sys.argv = original_argv
        if getattr(args, "phase", "all") == "servo":
            from servo_delivery import run_servo_delivery
            raise SystemExit(run_servo_delivery(args, backend=None))
        effective_argv = (
            list(sys.argv)
            if argv is None
            else [sys.argv[0], *list(argv)]
        )
        initialize_experiment_log(args, effective_argv)
        mark_experiment_stage(args, "preflight")
        from cameras.preflight import grasp_preflight
        grasp_preflight(args)
        mark_experiment_stage(args, "workflow")
        result = run_single_grasp(args)
        if isinstance(result, GraspAttemptResult):
            print(
                f"[session] status={result.status.value} "
                f"motion_sent={result.motion_sent} detail={result.detail}"
            )
            if result.status not in {
                GraspAttemptStatus.SUCCESS, GraspAttemptStatus.PLANNED_ONLY,
                GraspAttemptStatus.CONTACT_CHECKED,
            }:
                exit_code = 2
    except CameraSelfMaskMismatch as exc:
        terminal_status = "capture_failed"
        terminal_exception = exc
        print(f"[capture][abort] {exc}")
        exit_code = 2
    except ControlledMotionStopTriggered as exc:
        terminal_status = "controlled_motion_stop"
        terminal_exception = exc
        print(f"[motion_stop] {exc}")
        print(
            "[motion_stop] Program exited after requesting ordinary trajectory "
            "termination; no electronic emergency stop was requested."
        )
        exit_code = 2
    except SafetyStopTriggered as exc:
        terminal_status = "safety_stop"
        terminal_exception = exc
        print(f"[protect][motion] {exc}")
        print("[protect][motion] Program exited after requesting electronic emergency stop.")
        exit_code = 3
    except KeyboardInterrupt as exc:
        terminal_status = "keyboard_interrupt"
        terminal_exception = exc
        # The arm adapter context has already confirmed either an idle arm or
        # ordinary trajectory termination before control reaches here.
        print("[interrupt] Program exited after Ctrl+C shutdown handling.")
        exit_code = 130
    except Exception as exc:
        terminal_status = "internal_error"
        terminal_exception = exc
        raise
    finally:
        from .planning_process import close_planning_process
        close_planning_process()
        _ik_solver_impl._IK_SOLVER = None
        _ik_solver_impl._IK_SOLVER_KEY = None
        if args is not None and hasattr(args, "_experiment_log_state"):
            finalize_experiment_log(
                args,
                result,
                terminal_status=terminal_status,
                exception=terminal_exception,
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
