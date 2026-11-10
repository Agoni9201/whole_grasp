"""Bounded grasp-conditioned observation state, shared by live and saved replay.

This module never constructs hardware. The caller injects a planning-only hook;
actual observation moves remain in the existing guarded active-view workflow.
"""

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .perception_evidence import DepthEvidence, classify_prediction, contact_evidence, transform, visible_validation_gain


@dataclass
class FeedbackPlanningSnapshot:
    rows: list
    T_base_camera: np.ndarray
    target_cloud: object
    out_dir: str


def validate_feedback_args(args):
    if not getattr(args, 'active_grasp_feedback', False):
        return
    if not (args.active_view and args.active_center_first and args.active_restore_center
            and args.save_active_raw_views and args.mgpc and args.grasp_generator == 'graspgenx'
            and args.grasp_table_reference_z is not None and args.active_fusion_register
            and args.active_fusion_reject_bad_views):
        raise ValueError('--active-grasp-feedback requires active centered views, restore, raw archives, MGPC, GraspGenX and a confirmed table reference')
    if args.tool_approach_axis != 'x' or args.tcp_gripper_width_axis != 'z':
        raise ValueError('Feedback currently supports native Nero x/z TCP axes')
    if args.grasp_required_view_count not in (0, 2):
        raise ValueError('Feedback requires center plus at least one side; use --grasp-required-view-count 2')
    for name in ('feedback_support_distance', 'feedback_duplicate_distance', 'feedback_timeout',
                 'feedback_probe_refine_timeout_s'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if not 0 <= args.feedback_prediction_ratio <= 1 or not 0 < args.feedback_contact_support <= 1:
        raise ValueError('Invalid feedback prediction/support ratio')
    if not np.isfinite(args.feedback_view_weight) or args.feedback_view_weight < 0:
        raise ValueError('Feedback view weight must be finite and nonnegative')
    if not 1 <= args.feedback_candidate_limit <= 40:
        raise ValueError('Feedback candidate memory must contain 1..40 poses')
    if not 1 <= args.feedback_probe_preflight_max_candidates <= 40:
        raise ValueError('Feedback probe route checks must contain 1..40 poses')
    if args.feedback_max_attempts not in (1, 2):
        raise ValueError('Feedback permits at most two observation attempts')
    args.active_multiview_gate = False
    args.active_side_view_count = 2
    args.active_side_max_capture_attempts = args.feedback_max_attempts
    args.grasp_required_view_count = 2


def read_observations(directory, accepted_names=None):
    """Read only archived observations available in this round, not future data.

    Target registration moves both a target ray and its depth origin together;
    it is an observation-alignment proxy, distinct from the safety scene map.
    Initial selection geometry is excluded from target evidence.
    """
    directory = Path(directory)
    table_path = directory / 'table_reference/table_reference_report.json'
    correction = np.eye(4)
    if table_path.exists():
        table = json.loads(table_path.read_text())
        correction = np.asarray(next(v for v in table['views'] if v['view'] == 'center')['T_registration'])
    registration_path = directory / 'active_fusion_registration.json'
    registered = {}
    if registration_path.exists():
        registered = {v['name']: v for v in json.loads(registration_path.read_text())['views']}
    observations = []
    for path in sorted((directory / 'multiview_raw').glob('*/metadata.json')):
        meta = json.loads(path.read_text())
        name = meta['view_label']
        if name == 'initial' or (accepted_names is not None and name not in accepted_names):
            continue
        if meta.get('status', {}).get('capture_quality_keep') is False:
            continue
        reg = registered.get(name)
        if reg is not None and not reg['registration_keep']:
            continue
        import cv2
        mask = cv2.imread(str(path.parent / 'selected_target_mask.png'), 0)
        if mask is None or not (path.parent / 'depth_m.npy').exists():
            raise ValueError(f'Missing paired depth/mask for evidence: {path.parent}')
        pose = correction @ np.asarray(reg['target_transform'] if reg else np.eye(4)) @ np.asarray(meta['T_base_camera'])
        observations.append(DepthEvidence(name, pose, np.load(path.parent / 'depth_m.npy', allow_pickle=False),
                                          mask > 0, meta['intrinsics']))
    return observations, correction


class ActiveGraspFeedback:
    def __init__(self, args, probe_runner=None, clock=time.monotonic):
        self.args, self.probe_runner, self.clock = args, probe_runner, clock
        self.started = clock()
        self.attempts = self.valid_sides = self.generation_calls = 0
        self.version = 0
        self.memory = []
        self.next_id = 0
        self.prediction_base = np.empty((0, 3))
        self.measured_base = np.empty((0, 3))
        self.observations = []
        self.frame_correction = np.eye(4)
        self.accepted_names = ['center']
        self.targets = self.normals = np.empty((0, 3))
        self.weights = np.empty(0)
        self.events = []
        self.view_events = []
        self.unchanged_updates = []
        self.last_capture_revision = None
        self.ready = False
        self.last_directory = None

    @property
    def expired(self):
        return self.clock() - self.started >= self.args.feedback_timeout

    def geometry(self, target_cloud, T_base_camera, directory):
        self.measured_base = transform(np.asarray(target_cloud.points), T_base_camera)
        self.observations, self.frame_correction = read_observations(directory, self.accepted_names)
        self.valid_sides = sum(v.name.startswith('side') and np.count_nonzero(v.valid & v.target_interior) >= 10
                               for v in self.observations)

    def reuse_rows(self, T_base_camera):
        """Keep pose identity while discarding every old gate and path result."""
        rows = []
        inverse = np.linalg.inv(T_base_camera)
        for saved in self.memory:
            old = saved['row']
            grasp = copy.deepcopy(old['grasp'])
            relative = inverse @ saved['T_base_camera']
            grasp.translation = transform(np.asarray(grasp.translation).reshape(1, 3), relative)[0]
            grasp.rotation_matrix = relative[:3, :3] @ np.asarray(grasp.rotation_matrix)
            grasp.nero_tcp_pose_camera = relative @ np.asarray(grasp.nero_tcp_pose_camera)
            row = {key: copy.deepcopy(old[key]) for key in (
                'generator', 'raw_score', 'generator_score', 'generator_index', 'priority', 'width_source',
                'genx_refinement_only', 'genx_native_table_safe', 'genx_table_clearance_m',
                'nero_refinement_applied', 'genx_contact_advance_mode', 'genx_contact_advance_m',
                'feedback_memory_id', 'feedback_parent_id') if key in old}
            row.update(idx=len(rows), grasp=grasp, translation=grasp.translation.copy(),
                       genx_tcp_pose_camera=np.asarray(grasp.nero_tcp_pose_camera).tolist())
            rows.append(row)
        return rows

    def merge_rows(self, old, new, T_base_camera):
        result = list(old)
        for row in new:
            pose = np.asarray(row['grasp'].nero_tcp_pose_camera).copy()
            pose[:3, 3] += pose[:3, 0] * float(getattr(row['grasp'], 'nero_execution_advance_m',
                                                       getattr(row['grasp'], 'nero_tcp_contact_advance', 0.)))
            same = False
            for other in result:
                previous = np.asarray(other['grasp'].nero_tcp_pose_camera).copy()
                previous[:3, 3] += previous[:3, 0] * float(getattr(other['grasp'], 'nero_execution_advance_m',
                                                                 getattr(other['grasp'], 'nero_tcp_contact_advance', 0.)))
                angle = Rotation.from_matrix(pose[:3, :3] @ previous[:3, :3].T).magnitude()
                if np.linalg.norm(pose[:3, 3] - previous[:3, 3]) < .005 and angle < np.deg2rad(10):
                    same = True
                    break
            if not same:
                row = dict(row)
                row['feedback_memory_id'] = f'g{self.next_id}'
                self.next_id += 1
                result.append(row)
        # Preserve old hypotheses when the generator proposes new ones. Give
        # both sources a bounded chance before expensive route checks.
        # Interleave old and genuinely new hypotheses when memory is full.
        previous_ids = {id(row) for row in old}
        fresh = [row for row in result if id(row) not in previous_ids]
        previous = [row for row in result if id(row) in previous_ids]
        result = []
        for index in range(max(len(previous), len(fresh))):
            if index < len(previous):
                result.append(previous[index])
            if index < len(fresh):
                result.append(fresh[index])
        for index, row in enumerate(result):
            row['idx'] = index
        return result[:self.args.feedback_candidate_limit]

    def assess(self, rows, T_base_camera, directory):
        from .grasp_geometry import grasp_to_tcp_poses
        from .transforms import _make_T_from_pose
        self.ready = False
        states, support_views, conflict_views = classify_prediction(
            self.prediction_base, self.measured_base, self.observations, self.args.feedback_support_distance)
        targets, normals, weights = [], [], []
        records = []
        for row in rows:
            # Final TCP includes measured recentering and the same advance
            # that the production preflight evaluates.
            sign = float(row.get('tcp_width_sign', 1.))
            shift = float(next((v.get('tcp_width_shift', 0.) for v in row.get('physical_aperture_variants', [])
                                if v.get('ok') and float(v['width_sign']) == sign), 0.))
            _, tcp, _ = grasp_to_tcp_poses(row['grasp'], T_base_camera, self.args, None,
                                          width_sign=sign, width_shift=shift)
            tcp_matrix = _make_T_from_pose(tcp)
            evidence = contact_evidence(tcp_matrix, self.measured_base, self.prediction_base,
                self.observations, opening=min(self.args.gripper_open_width, self.args.gripper_hardware_max_width),
                tolerance=self.args.feedback_support_distance, required_support=self.args.feedback_contact_support,
                finger_back=self.args.motion_path_open_finger_back, finger_front=self.args.motion_path_open_finger_front,
                finger_radius=self.args.motion_path_open_finger_radius)
            row['feedback_contact_regions'] = evidence['regions']
            row['feedback_evidence_sufficient'] = evidence['sufficient']
            if 'feedback_memory_id' not in row:
                row['feedback_memory_id'] = f'g{self.next_id}'
                self.next_id += 1
            previous = next((item for item in self.memory
                             if item['row']['feedback_memory_id'] == row['feedback_memory_id']), None)
            if previous is not None and row.get('nero_refinement_applied', False):
                before = previous['T_base_camera'] @ np.asarray(previous['row']['grasp'].nero_tcp_pose_camera)
                after = np.asarray(T_base_camera) @ np.asarray(row['grasp'].nero_tcp_pose_camera)
                if (np.linalg.norm(before[:3, 3] - after[:3, 3]) > .005 or
                        Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).magnitude() > np.deg2rad(10)):
                    row['feedback_parent_id'] = row['feedback_memory_id']
                    row['feedback_memory_id'] = f'g{self.next_id}'
                    self.next_id += 1
            passes = bool(row.get('workflow_preflight_ok', False) and evidence['sufficient'])
            self.ready |= passes
            records.append(dict(memory_id=row['feedback_memory_id'], idx=int(row['idx']),
                                parent_memory_id=row.get('feedback_parent_id'), tcp_base=tcp_matrix.tolist(),
                                evidence_sufficient=evidence['sufficient'], contact_regions=evidence['regions'],
                                complete_route_pass=bool(row.get('workflow_preflight_ok', False)),
                                reason=str(row.get('workflow_preflight_reason', 'not_checked'))))
            # A pose known to hit the table is repair input, not an excuse to
            # move the camera. Unsupported but otherwise plausible poses stay.
            if not row.get('genx_refinement_only', False) and row.get('reachability_ok', True):
                targets.extend(evidence['target_points'])
                normals.extend(evidence['target_normals'])
                weights.extend(evidence['target_weights'])
        self.targets = np.asarray(targets).reshape(-1, 3)
        self.normals = np.asarray(normals).reshape(-1, 3)
        self.weights = np.asarray(weights)
        # Candidate count cannot multiply the value of one physical patch.
        if len(self.targets):
            _, keep = np.unique(np.floor(self.targets / .004).astype(int), axis=0, return_index=True)
            self.targets, self.normals, self.weights = self.targets[keep], self.normals[keep], self.weights[keep]
        self.ready = bool(self.ready and self.valid_sides >= 1 and not self.expired)
        self.memory = [dict(row=copy.deepcopy(r), T_base_camera=np.asarray(T_base_camera).copy())
                       for r in sorted(rows, key=lambda r: (not r.get('workflow_preflight_ok', False),
                                                          -float(r.get('raw_score', 0.))))[:self.args.feedback_candidate_limit]]
        event = dict(version=self.version, attempts=self.attempts, valid_sides=self.valid_sides,
                     ready=self.ready, generation_calls=self.generation_calls, candidates=records,
                     prediction_state_counts={name: int(np.sum(states == state)) for state, name in
                                              enumerate(('unknown', 'supported', 'conflict', 'inconsistent'))},
                     elapsed_s=self.clock()-self.started)
        self.events.append(event)
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / 'feedback_evaluation.json').write_text(json.dumps(event, indent=2, allow_nan=False)+'\n')
        np.savez_compressed(Path(directory)/'feedback_evidence.npz', measured_base=self.measured_base,
                            prediction_base=self.prediction_base, prediction_states=states,
                            support_view_count=support_views, conflict_view_count=conflict_views)
        self.last_directory = str(directory)

    def rank_views(self, candidates, scene_base):
        result = list(candidates)
        for candidate in result:
            pose = candidate.get('T_base_camera')
            gain = 0. if pose is None else visible_validation_gain(
                self.frame_correction @ np.asarray(pose), self.targets, self.normals, self.weights,
                transform(np.asarray(scene_base), self.frame_correction))
            candidate.setdefault('feedback_original_score', float(candidate.get('score', 0.)))
            candidate['feedback_validation_gain'] = gain
            candidate['feedback_selection_score'] = candidate['feedback_original_score'] + self.args.feedback_view_weight * gain
        return sorted(result, key=lambda c: c['feedback_selection_score'], reverse=True)

    def record_view_attempt(self, candidate):
        self.view_events.append(dict(attempt=self.attempts, observation_version=self.version,
            name=candidate.get('name'), T_base_camera=np.asarray(candidate['T_base_camera']).tolist(),
            validation_gain=float(candidate.get('feedback_validation_gain', 0.)),
            selection_score=float(candidate.get('feedback_selection_score', candidate.get('score', 0.))),
            target_points=len(self.targets)))

    def update(self, arm, target, scene, T_base_camera, bgr, intr, inst, archive_root, directory):
        if self.expired:
            self.ready = False
            return
        if self.probe_runner is None:
            raise RuntimeError('Feedback planning-only dependency missing')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        link = directory / 'multiview_raw'
        if not link.exists():
            link.symlink_to((Path(archive_root) / 'multiview_raw').resolve(), target_is_directory=True)
        self.version += 1
        # Recheck old hypotheses first. Only generate if no hypothesis remains
        # executable/supported. At most one new network pool per observation.
        for generate in ([True] if not self.memory else [False, True]):
            if self.expired:
                self.ready = False
                break
            args = copy.copy(self.args)
            args.execute = False
            args.active_side_view_count = 0  # Center-only probes may have no side archive yet.
            args._experiment_log_state = None
            args._active_feedback_session = self
            args._feedback_generate = generate
            # Probes select observation targets. Final execution goes through
            # the caller's original route and refinement budgets again.
            args.grasp_preflight_max_candidates = min(
                self.args.grasp_preflight_max_candidates,
                self.args.feedback_probe_preflight_max_candidates)
            args.grasp_target_valid_plans = 1
            args.nero_refine_timeout_s = min(
                self.args.nero_refine_timeout_s, self.args.feedback_probe_refine_timeout_s)
            args.preview_graspnet_cloud = args.preview_final_grasp = False
            if generate:
                self.generation_calls += 1
            snapshot = self.probe_runner(arm, args, target, scene, T_base_camera, bgr, intr,
                                         str(directory), inst, planning_only=True,
                                         plan_label=f'feedback_v{self.version}')
            if not isinstance(snapshot, FeedbackPlanningSnapshot):
                raise RuntimeError(f'Feedback planning failed: {getattr(snapshot, "detail", snapshot)}')
            self.assess(snapshot.rows, snapshot.T_base_camera, directory)
            if self.view_events:
                self.view_events[-1]['valid_sides_after_update'] = self.valid_sides
            if self.ready or self.expired or (not generate and any(
                    r.get('workflow_preflight_ok') and r.get('feedback_evidence_sufficient') for r in snapshot.rows)):
                break

    def write_summary(self, root):
        path = Path(root) / 'feedback/summary.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(attempts=self.attempts, valid_sides=self.valid_sides,
            ready=self.ready, expired=self.expired, generation_calls=self.generation_calls,
            events=self.events, view_events=self.view_events, minimum_effective_sides=1,
            unchanged_updates=self.unchanged_updates,
            maximum_attempts=self.args.feedback_max_attempts),
            indent=2, allow_nan=False)+'\n')
