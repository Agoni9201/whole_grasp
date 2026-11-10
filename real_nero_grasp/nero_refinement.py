"""Bounded Nero pose/depth refinement shared by GraspNet and native GraspGenX.

Geometry search can run in the CAN-forbidden numerical worker. Route checks use
the normal workflow and its solver process; this module never commands motion.
"""

import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_candidates import annotate_grasp_candidate_physical_aperture
from .grasp_geometry import grasp_to_tcp_poses, graspnet_rotation_to_tcp_rotation, _rotation_from_axis_constraints
from .graspgenx_integration import canonical_to_tcp, grasp_mesh_table_margin, open_gripper_vertices_tcp
from .transforms import _make_T_from_pose
from .types import PlanNotExecutable, UnsafeMotionTarget


def pack_grasp(grasp):
    result = {name: float(getattr(grasp, name)) for name in ('width', 'depth', 'height', 'score')}
    result.update(translation=np.asarray(grasp.translation).copy(),
                  rotation_matrix=np.asarray(grasp.rotation_matrix).copy())
    for name in ('nero_tcp_pose_camera', 'nero_tcp_contact_advance', 'nero_execution_advance_m'):
        if hasattr(grasp, name):
            result[name] = getattr(grasp, name)
    return result


def orientation_variants(base_tcp):
    approach, width = base_tcp[:, 0], base_tcp[:, 2]
    down = np.array([0., 0., -1.])
    horizontal = approach - down * np.dot(approach, down)
    tilt = float(np.rad2deg(np.arccos(np.clip(np.dot(approach, down), -1, 1))))
    families = [('native', base_tcp)]
    for cap, name in ((tilt, 'level_jaws'), (30., 'tilt30'), (15., 'tilt15'), (0., 'vertical')):
        if name != 'level_jaws' and tilt <= cap + 1e-6:
            continue
        direction = approach if name == 'level_jaws' else (
            np.cos(np.deg2rad(cap)) * down
            + np.sin(np.deg2rad(cap)) * horizontal / np.linalg.norm(horizontal))
        closing = np.cross([0., 0., 1.], direction)
        if np.linalg.norm(closing) < 1e-8:
            closing = np.array([width[0], width[1], 0.])
        closing /= np.linalg.norm(closing)
        if np.dot(closing, width) < 0:
            closing = -closing
        families.append((name, _rotation_from_axis_constraints('x', direction, 'z', closing)))
    families += [(name + '_flip', matrix @ np.diag([1., -1., -1.]))
                 for name, matrix in list(families)]
    seen = []
    for name, matrix in families:
        if any(np.allclose(matrix, other, atol=1e-7) for other in seen):
            continue
        seen.append(matrix)
        angle = float(np.rad2deg(Rotation.from_matrix(matrix @ base_tcp.T).magnitude()))
        yield name, matrix, angle


def geometry_variants(row, target, camera, args):
    """Prefer contact inside the fingers while retaining the first valid depth."""
    native = row['generator'] == 'graspgenx'
    grasp = SimpleNamespace(**row['grasp'])
    base_tcp = (camera[:3, :3] @ np.asarray(grasp.nero_tcp_pose_camera)[:3, :3]
                if native else graspnet_rotation_to_tcp_rotation(camera[:3, :3] @ grasp.rotation_matrix, args))
    mapping = graspnet_rotation_to_tcp_rotation(np.eye(3), args)
    vertices, margin = open_gripper_vertices_tcp(args), grasp_mesh_table_margin(args)
    proposals, fallbacks, samples = [], [], 0
    # Preserve GenX's validated 0..15 mm native offset envelope. GraspNet uses
    # the 0..30 mm total advance grid evaluated in the offline comparison.
    for family, rotation, angle in orientation_variants(base_tcp):
        altered = copy.deepcopy(grasp)
        altered.rotation_matrix = camera[:3, :3].T @ rotation @ mapping.T
        if native:
            altered.nero_tcp_pose_camera = np.asarray(grasp.nero_tcp_pose_camera).copy()
            altered.nero_tcp_pose_camera[:3, :3] = camera[:3, :3].T @ rotation
        probe_row = dict(idx=row['idx'], generator=row['generator'], grasp=altered)
        first_valid = None
        preferred = None
        for millimeters in range(16 if native else 31):
            advance = millimeters / 1000.
            altered.nero_execution_advance_m = advance
            if native:
                altered.nero_tcp_contact_advance = advance
            probe = annotate_grasp_candidate_physical_aperture(
                [probe_row], target, camera, None, args, verbose=False)[0]
            samples += 1
            if not probe['physical_aperture_ok']:
                continue
            variant = next(item for item in probe['physical_aperture_variants'] if item['ok'])
            pre, contact, _ = grasp_to_tcp_poses(altered, camera, args,
                width_sign=variant['width_sign'], width_shift=variant['tcp_width_shift'])
            clearance = min(float(np.min(vertices @ pose[2, :3]) + pose[2, 3])
                for pose in (_make_T_from_pose(pre), _make_T_from_pose(contact))) - args.table_z
            if clearance < margin:
                continue
            solution = dict(grasp=pack_grasp(altered), family=family,
                angle_change_deg=angle, advance_m=advance, cad_table_clearance_m=clearance)
            if first_valid is None:
                first_valid = solution
            if not native:
                break
            if advance - first_valid['advance_m'] > .005 + 1e-9:
                break
            # A just-passing 10 mm contact inset can leave the object at the
            # finger tips. Seek 5 mm more contact only when the CAD still has
            # 5 mm of clearance beyond the hard table margin. The first
            # feasible pose remains in the bounded route candidate pool.
            if (min(variant['left_contact_inset'], variant['right_contact_inset'])
                    >= args.grasp_physical_min_contact_inset + .005
                    and clearance >= margin + .005):
                preferred = solution
                break
        if first_valid is not None:
            proposals.append(preferred or first_valid)
            if preferred is not None and preferred['advance_m'] != first_valid['advance_m']:
                fallbacks.append(first_valid)
    proposals.extend(fallbacks)
    return dict(idx=row['idx'], proposals=proposals, samples=samples)


def qualify_hold(row, args, label):
    """Qualify requested carrying goal before exposing a row for selection."""
    if not row.get('workflow_preflight_ok', False):
        return False
    if args.post_grasp_hold_joints_deg is not None and not args.grasp_contact_check_only:
        from .motion_planning import _motion_context
        from .post_grasp_rrt import plan_post_grasp_joint_hold
        plan = row['_workflow_motion_plan']
        context = _motion_context(plan.safety_scene, 'carrying',
            attached_target_tcp=plan.attached_target_tcp, closed_gripper_width=plan.closed_gripper_width)
        try:
            stages = plan_post_grasp_joint_hold(plan.after_close[-1].joints, context, args, label + '_hold')
        except (PlanNotExecutable, UnsafeMotionTarget) as exc:
            row['workflow_preflight_ok'] = False
            row['workflow_preflight_reason'] = 'requested joint hold failed: ' + str(exc)
            return False
        row['_nero_joint_hold_stages'] = stages
    row['nero_full_route_ok'] = True
    return True


def refine_failed_candidates(rows, arm, args, target, scene, camera, current_tcp, label, out_dir,
                             *, preflight, aperture, reachability, preflight_options):
    """Retain valid native plans; repair failed poses only if no full plan exists."""
    from .planning_process import checked_request, config_snapshot, planning_health_check
    report = dict(generator=args.grasp_generator, native_full_routes=0, refined_full_routes=0,
                  input_candidates=len(rows),
                  refinement_only_candidates=sum(bool(row.get('genx_refinement_only', False)) for row in rows),
                  route_attempts=0, geometry_samples=0, attempts=[], stopped_by_budget=False)
    for row in rows:
        if qualify_hold(row, args, f'{label}_native{row["idx"]}'):
            report['native_full_routes'] += 1
    if report['native_full_routes']:
        print(f'[nero-refine] native full routes={report["native_full_routes"]}; reuse original candidates')
    else:
        started = time.monotonic()
        budgeted = rows[:max(int(args.grasp_preflight_max_candidates), 1)]
        payload = [dict(idx=int(row['idx']), generator=args.grasp_generator,
                        grasp=pack_grasp(row['grasp'])) for row in budgeted]
        groups = checked_request('nero_grasp_geometry', dict(config=config_snapshot(args), rows=payload,
            target_points=np.asarray(target.points).copy(), T_base_camera=np.asarray(camera)), args)
        report['geometry_samples'] = sum(group['samples'] for group in groups)
        target_valid = max(int(args.grasp_target_valid_plans), 1)
        for original, group in zip(budgeted, groups):
            for number, proposal in enumerate(group['proposals'][:args.nero_refine_max_route_variants]):
                if time.monotonic() - started >= args.nero_refine_timeout_s:
                    report['stopped_by_budget'] = True
                    break
                native_advance = (float(getattr(original['grasp'], 'nero_tcp_contact_advance', 0.))
                    if args.graspgenx_contact_advance == 'auto' else float(args.graspgenx_contact_advance))
                baseline_advance = native_advance if args.grasp_generator == 'graspgenx' else args.grasp_advance
                if proposal['family'] == 'native' and np.isclose(proposal['advance_m'], baseline_advance):
                    continue
                planning_health_check(args, label)
                altered = {key: value for key, value in original.items()
                           if not key.startswith('_') and not key.startswith('workflow_preflight_')}
                altered.update(grasp=SimpleNamespace(**proposal['grasp']), nero_refinement_applied=True,
                    nero_refinement_family=proposal['family'], nero_refinement_advance_m=proposal['advance_m'],
                    nero_refinement_angle_change_deg=proposal['angle_change_deg'],
                    nero_refinement_original_rotation_camera=np.asarray(original['grasp'].rotation_matrix).tolist())
                if original.get('target_direction_applicable', False):
                    from .grasp_geometry import _axis_vector
                    closing = altered['grasp'].rotation_matrix @ _axis_vector(args.graspnet_width_axis)
                    alignment = float(abs(np.dot(closing, original['target_main_axis_camera'])))
                    altered.update(width_long_axis_alignment=alignment, target_cross_axis_score=1. - alignment)
                if args.grasp_generator == 'graspgenx':
                    pose = altered['grasp'].nero_tcp_pose_camera
                    altered.update(genx_tcp_pose_camera=np.asarray(pose).tolist(),
                        genx_pose_camera=(pose @ np.linalg.inv(canonical_to_tcp(args))).tolist(),
                        genx_contact_advance_m=proposal['advance_m'],
                        genx_refinement_only=False, genx_contact_refine_ok=True,
                        genx_refined_table_clearance_m=proposal['cad_table_clearance_m'])
                local = copy.copy(args)
                local.grasp_preflight_max_candidates = local.grasp_target_valid_plans = 1
                probes = aperture([altered], target, camera, current_tcp, local)
                probes = reachability(probes, camera, current_tcp, local, arm=arm)
                probes = preflight(probes, arm, local, target, scene, camera, current_tcp,
                    f'{label}_refine{original["idx"]}_{number}', **preflight_options)
                accepted = qualify_hold(probes[0], local, f'{label}_refine{original["idx"]}_{number}')
                planning_health_check(args, label)
                report['route_attempts'] += 1
                report['attempts'].append(dict(idx=int(original['idx']), family=proposal['family'],
                    generator_index=original.get('generator_index'),
                    from_table_rejected_pool=bool(original.get('genx_refinement_only', False)),
                    advance_m=proposal['advance_m'], accepted=accepted,
                    reason=probes[0]['workflow_preflight_reason']))
                if accepted:
                    original.clear()
                    original.update(probes[0])
                    report['refined_full_routes'] += 1
                    print(f'[nero-refine] kept candidate={original["idx"]} family={proposal["family"]} '
                          f'advance={proposal["advance_m"]*1000:.1f}mm full route incl. hold passed')
                    break
            if report['stopped_by_budget'] or report['refined_full_routes'] >= target_valid:
                break
        report['seconds'] = time.monotonic() - started
    Path(out_dir, 'nero_refinement.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'[nero-refine] native={report["native_full_routes"]} refined={report["refined_full_routes"]} '
          f'route_attempts={report["route_attempts"]} budget_stop={report["stopped_by_budget"]}')
    return rows
