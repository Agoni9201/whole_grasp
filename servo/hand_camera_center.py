"""Calibrated, whole-arm image centering with TCP-to-hand range control.

Only the three visual features (image x/y and hand range) are primary tasks.
Translation and rotation are solved together, rather than assigning centering
to the wrist or freezing TCP orientation. No device or camera I/O occurs here.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from servo.pbvs_pose_servo import PBVSPoseServo


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class HandCameraCenterServo(PBVSPoseServo):
    def __init__(self, params, model, tcp_camera, joint_bounds, *, gain=2., max_accel=.5):
        super().__init__(params=params, model=model)
        self.tcp_camera = np.asarray(tcp_camera, dtype=float).copy()
        self.joint_bounds = np.asarray(joint_bounds, dtype=float).copy()
        T = self.tcp_camera
        if (T.shape != (4, 4) or not np.isfinite(T).all()
                or not np.allclose(T[3], [0, 0, 0, 1])
                or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-6)):
            raise ValueError('hand camera centering requires a rigid TCP-to-camera transform')
        if (self.joint_bounds.shape != (7, 2) or not np.isfinite(self.joint_bounds).all()
                or np.any(self.joint_bounds[:, 1] - self.joint_bounds[:, 0] <= .04)):
            raise ValueError('invalid software joint bounds')
        self.gain = float(gain)
        self.max_accel = float(max_accel)
        if not (np.isfinite(self.gain) and self.gain > 0
                and np.isfinite(self.max_accel) and self.max_accel > 0):
            raise ValueError('centering gain and acceleration must be finite and positive')
        self.planning_bounds = self.joint_bounds + np.array([.02, -.02])
        self._camera_K = None
        self._image_center = None
        self._center_velocity = np.zeros(7)
        self._center_time = None

    def set_camera_intrinsics(self, K, width, height):
        K = np.asarray(K, dtype=float).reshape(3, 3)
        if not np.isfinite(K).all() or min(K[0, 0], K[1, 1], width, height) <= 0:
            raise ValueError('invalid camera centering intrinsics')
        self._camera_K = K.copy()
        self._image_center = (np.array([width, height], dtype=float) - 1.) / 2.

    def pause(self):
        self._center_velocity[:] = 0
        self._center_time = None

    def feature_system(self, q, hand):
        """Analytic image/range Jacobian, including the camera lever arm."""
        tcp, R = self._model.fk(q)
        R = np.asarray(R).copy()
        J = self._model.jacobian(q)
        lever = R @ self.tcp_camera[:3, 3]
        R_camera = R @ self.tcp_camera[:3, :3]
        point = R_camera.T @ (hand - tcp - lever)
        if not np.isfinite(point).all() or point[2] <= .03:
            raise ValueError('hand is behind or too close to the camera optical plane')
        camera_J = np.vstack((R_camera.T @ (J[:3] - skew(lever) @ J[3:]),
                              R_camera.T @ J[3:]))
        X, Y, Z = point
        projection = np.array([[1/Z, 0, -X/Z**2], [0, 1/Z, -Y/Z**2]])
        image_J = projection @ np.column_stack((-np.eye(3), skew(point))) @ camera_J
        K = self._camera_K
        normalized = point[:2] / Z
        desired = (self._image_center - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
        ray = np.asarray(tcp) - hand
        distance = float(np.linalg.norm(ray))
        if distance <= .03:
            raise ValueError('hand is too close to TCP for range control')
        direction = ray / distance
        range_J = direction @ J[:3]
        scale = max(.05, float(self.params.desired_distance))
        error = np.r_[(normalized - desired) * scale,
                      distance - self.params.desired_distance]
        A = np.vstack((image_J * scale, range_J))
        pixel = normalized * [K[0, 0], K[1, 1]] + K[:2, 2]
        # Suppress unnecessary spin around the observation ray, not pitch/yaw.
        roll_J = (point / np.linalg.norm(point)) @ camera_J[3:]
        return A, error, J, roll_J, pixel, distance, np.asarray(tcp), R, direction

    def _solve(self, q, A, error, J, roll_J, dt):
        """Small convex quadratic objective with box and TCP-speed constraints.

        Position/velocity limits dominate acceleration. When feedback or a
        changed Jacobian makes the old acceleration interval infeasible, hold;
        never return an unchecked solver iterate or rescale it after solving.
        """
        error = np.asarray(error, dtype=float)
        vmax = float(self.params.max_joint_vel)
        if min(vmax, self.params.max_linear_vel, self.params.max_angular_vel) <= 0:
            return np.zeros(7), 'motion_disabled', []
        lo, hi = self.planning_bounds.T
        lower = np.maximum(-vmax, np.minimum(0., lo - q))
        upper = np.minimum(vmax, np.maximum(0., hi - q))
        # A one-second velocity damper slows outward travel before the margin.
        step = self.max_accel * dt
        smooth_lower = np.maximum(lower, self._center_velocity - step)
        smooth_upper = np.minimum(upper, self._center_velocity + step)
        if np.any(smooth_lower > smooth_upper):
            return np.zeros(7), 'joint_braking_hold', []
        lower, upper = smooth_lower, smooth_upper
        center = np.mean(self.planning_bounds, axis=1)
        clearance = np.minimum(q - lo, hi - q)
        activation = np.clip((.25 - clearance) / .25, 0., 1.)
        bias = np.sign(center - q) * activation * min(.03, vmax)
        regularization = .006
        H = A.T @ A + .02**2 * np.outer(roll_J, roll_J) + regularization**2 * np.eye(7)
        b = A.T @ (-self.gain * error) + regularization**2 * bias
        # Scaling the objective helps SLSQP resolve low-speed corrections.
        objective_scale = 10.
        constraints = []
        for matrix, limit in ((J[:3], self.params.max_linear_vel),
                              (J[3:], self.params.max_angular_vel)):
            M = matrix / float(limit)
            constraints.append(dict(type='ineq',
                fun=lambda x, M=M: 1. - float((M @ x) @ (M @ x)),
                jac=lambda x, M=M: -2. * M.T @ (M @ x)))
        start = np.clip(self._center_velocity, lower, upper)
        result = minimize(lambda x: objective_scale * (.5 * x @ H @ x - b @ x),
                          start, jac=lambda x: objective_scale * (H @ x - b),
                          bounds=list(zip(lower, upper)), constraints=constraints,
                          method='SLSQP', options={'maxiter': 60, 'ftol': 1e-10})
        self._solver_message = str(getattr(result, 'message', 'unknown'))
        x = np.asarray(result.x)
        valid = (result.success and np.isfinite(x).all()
                 and np.all(x >= lower - 1e-8) and np.all(x <= upper + 1e-8)
                 and all(c['fun'](x) >= -1e-7 for c in constraints))
        if not valid:
            return np.zeros(7), 'solver_hold', []
        limited = (np.flatnonzero((np.abs(x - lower) < 1e-5)
                                 | (np.abs(x - upper) < 1e-5)) + 1).tolist()
        return x, 'ok', limited

    def update(self, q_now, target_position, target_normal=None, target_tangent=None,
               target_orientation=None, stamp=None):
        now = time.time() if stamp is None else float(stamp)
        # Retain the existing target jump, filtering, timeout and reset behavior.
        if target_position is not None:
            # Jump gating follows the TCP-hand offset, not a palm normal or a
            # viewing-ray offset that moves simply because the camera turns.
            ray = np.asarray(self._model.fk(q_now)[0]) - np.asarray(target_position)
            if np.linalg.norm(ray) > 1e-9:
                target_normal = ray / np.linalg.norm(ray)
        out = super().update(q_now, target_position, target_normal, target_tangent,
                             target_orientation, now)
        q = np.asarray(q_now, dtype=float)
        if out['target_lost'] or out['target_rejected']:
            self._center_velocity[:] = 0
            self._center_time = None
            return out
        dt = self.params.dt if self._center_time is None else min(
            self.params.dt, max(0., now - self._center_time))
        self._center_time = now
        detail = {'mode': 'whole_arm_camera_center'}
        try:
            if self._camera_K is None:
                raise ValueError('camera intrinsics unavailable')
            if np.any(q <= self.joint_bounds[:, 0]) or np.any(q >= self.joint_bounds[:, 1]):
                raise ValueError('software joint boundary reached')
            A, error, J, roll_J, pixel, distance, tcp, R, direction = self.feature_system(
                q, out['marker_center'])
            velocity, status, limited = self._solve(q, A, error, J, roll_J, dt)
            detail.update(pixel_error=pixel - self._image_center,
                          projected_palm_pixel=pixel, desired_pixel=self._image_center.copy(),
                          distance_m=distance, distance_error_mm=error[2]*1000,
                          joint_margin_deg=np.rad2deg(np.minimum(
                              q-self.joint_bounds[:, 0], self.joint_bounds[:, 1]-q)),
                          limited_joints=limited, solver_status=status)
            detail['solver_message'] = getattr(self, '_solver_message', '')
            if status not in ('ok', 'motion_disabled'):
                raise ValueError(status)
            self._center_velocity = velocity.copy()
            out.update(qdot=velocity, q_cmd=q + velocity * dt,
                       tcp_twist=J @ velocity, v_base=(J @ velocity)[:3],
                       error=error, target_tcp=out['marker_center'] + direction*self.params.desired_distance,
                       feature_jacobian=A.copy(),
                       target_rotation=R.copy(), orientation_error=np.zeros(3),
                       camera_center=detail, center_control=True)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._center_velocity[:] = 0
            out = self._hold(q, self._model.fk(q)[0], now)
            detail.update(solver_status=str(exc))
            out.update(camera_center=detail, center_control=True, control_hold=str(exc))
        return out

    def fixed_step(self, q, target, rotation, stamp):
        """Post-handover retreat is a fixed pose task, not a synthetic hand."""
        from scipy.spatial.transform import Rotation
        tcp, R = self._model.fk(q)
        R = R.copy()
        J = self._model.jacobian(q)
        error = np.asarray(tcp) - target
        angle = Rotation.from_matrix(R @ rotation.T).as_rotvec()
        dt = self.params.dt if self._center_time is None else min(
            self.params.dt, max(0., stamp - self._center_time))
        self._center_time = stamp
        dq, status, _ = self._solve(np.asarray(q), np.vstack((J[:3], .1*J[3:])),
                                   np.r_[error, .1*angle], J, np.zeros(7), dt)
        self._center_velocity = dq.copy()
        out = self._hold(np.asarray(q), tcp, stamp)
        out.update(qdot=dq, tcp_twist=J@dq, error=-error, target_tcp=target,
                   center_control=True)
        if status not in ('ok', 'motion_disabled'):
            out['control_hold'] = status
        return out
