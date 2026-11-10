import os

import numpy as np
from spatialmath import SE3

from .robot import Robot


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_T(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(*rpy)
    T[:3, 3] = np.array(xyz, dtype=float)
    return T


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return T


def _rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(R_current[:, 0], R_target[:, 0])
        + np.cross(R_current[:, 1], R_target[:, 1])
        + np.cross(R_current[:, 2], R_target[:, 2])
    )


class Nero7(Robot):
    """7-DOF Nero arm model built from the provided URDF joint chain."""

    DEFAULT_Q = np.array([0.0, 0.349066, 0.0, 1.151917, -1.493363, 0.139626, 0.0])

    def __init__(self) -> None:
        super().__init__()
        self._dof = 7
        self.q0 = Nero7.DEFAULT_Q.copy()
        self.joint_names = [f"joint{i}" for i in range(1, self._dof + 1)]
        self._q_lim_low = np.array([-2.70526, -1.74, -2.75, -1.01, -2.75, -0.73, -1.5707963])
        self._q_lim_up = np.array([2.70526, 1.74, 2.75, 2.14, 2.75, 0.95, 1.5707963])
        self._joint_origins = [
            _make_T((0, 0, 0.138), (0, 0, 0)),
            _make_T((0, 0, 0), (1.5707963, 3.1415926, 0)),
            _make_T((0, -0.31, 0), (-1.5707963, 0, 3.1415926926)),
            _make_T((0, 0, 0), (-1.5707963, 0, 3.1415926926)),
            _make_T((0, -0.27001, 0), (1.5707963, -1.5707963, 0)),
            _make_T((0, 0, 0), (1.5707963, -1.5707963, 0)),
            _make_T((0, -0.0235, 0), (1.5707963, 0, 0)),
        ]

    @property
    def q_lim_low(self):
        return self._q_lim_low.copy()

    @property
    def q_lim_up(self):
        return self._q_lim_up.copy()

    def set_base(self, base):
        if isinstance(base, SE3):
            self._base = base
        else:
            self._base = SE3.Trans(np.asarray(base, dtype=float).reshape(3,))

    def set_tool(self, tool: SE3):
        self._tool = tool

    def disable_base(self):
        self._base = SE3()

    def disable_tool(self):
        self._tool = SE3()

    def set_joint(self, q):
        q_new = np.array(self.q0, dtype=float)
        q_arr = np.array(q, dtype=float).reshape(-1)
        n = min(q_arr.size, self._dof)
        q_new[:n] = q_arr[:n]
        self.q0 = np.clip(q_new, self._q_lim_low, self._q_lim_up)

    def move_joint(self, q):
        self.set_joint(q)

    def set_robot_config(self, q):
        return None

    def _fkine_no_tool(self, q) -> np.ndarray:
        q = np.array(q, dtype=float).reshape(-1)
        T = self._base.A.copy()
        for origin, qi in zip(self._joint_origins, q):
            T = T @ origin @ _rotz(qi)
        return T

    def fkine(self, q) -> SE3:
        q_arr = np.array(self.q0, dtype=float)
        q_in = np.array(q, dtype=float).reshape(-1)
        q_arr[: min(q_in.size, self._dof)] = q_in[: self._dof]
        return SE3(self._fkine_no_tool(q_arr) @ self._tool.A)

    def get_cartesian(self):
        return self.fkine(self.q0)

    def _jacobian(self, q, eps=1e-5):
        T0 = self.fkine(q).A
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]
        J = np.zeros((6, self._dof))
        for i in range(self._dof):
            q_eps = np.array(q, dtype=float)
            q_eps[i] += eps
            Ti = self.fkine(q_eps).A
            J[:3, i] = (Ti[:3, 3] - p0) / eps
            J[3:, i] = _rotation_error(R0, Ti[:3, :3]) / eps
        return J

    def _ikine_from_seed(self, Tep: SE3, q_seed: np.ndarray) -> tuple:
        q = np.clip(np.array(q_seed, dtype=float).reshape(self._dof), self._q_lim_low, self._q_lim_up)
        T_target = Tep.A
        q_pref = 0.5 * (self._q_lim_low + self._q_lim_up)
        q7_pref = float(os.environ.get("NERO_IK_Q7_PREF", "-0.12"))
        q_pref[6] = np.clip(q7_pref, self._q_lim_low[6], self._q_lim_up[6])
        span = np.maximum(self._q_lim_up - self._q_lim_low, 1e-6)
        best_q = q.copy()
        best_score = np.inf
        for it in range(220):
            T_current = self.fkine(q).A
            pos_err = T_target[:3, 3] - T_current[:3, 3]
            rot_err = _rotation_error(T_current[:3, :3], T_target[:3, :3])
            err = np.r_[pos_err, rot_err]
            pos_norm = float(np.linalg.norm(pos_err))
            rot_norm = float(np.linalg.norm(rot_err))
            score = pos_norm + 0.08 * rot_norm
            if score < best_score:
                best_score = score
                best_q = q.copy()
            if pos_norm < 6e-4 and rot_norm < 1.2e-2:
                return np.clip(q, self._q_lim_low, self._q_lim_up), True, pos_norm, rot_norm
            J = self._jacobian(q)
            damping = 2e-3 + 3e-2 * min(1.0, score)
            A = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(A, err)
            # 7-DOF redundancy: weakly bias the nullspace toward joint centers.
            try:
                J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
                null_projector = np.eye(self._dof) - J_pinv @ J
                center_pull = -0.035 * ((q - q_pref) / span)
                dq = dq + null_projector @ center_pull
            except np.linalg.LinAlgError:
                pass
            step = 0.35 if it < 120 else 0.18
            max_step = 0.10
            step_norm = float(np.max(np.abs(dq)))
            if step_norm > max_step:
                dq = dq * (max_step / step_norm)
            q = np.clip(q + step * dq, self._q_lim_low, self._q_lim_up)
        T_best = self.fkine(best_q).A
        pos_norm = float(np.linalg.norm(T_target[:3, 3] - T_best[:3, 3]))
        rot_norm = float(np.linalg.norm(_rotation_error(T_best[:3, :3], T_target[:3, :3])))
        return np.clip(best_q, self._q_lim_low, self._q_lim_up), False, pos_norm, rot_norm

    def ikine(self, Tep: SE3) -> np.ndarray:
        q_cur = np.array(self.q0, dtype=float)
        q_mid = 0.5 * (self._q_lim_low + self._q_lim_up)
        q7_pref = float(os.environ.get("NERO_IK_Q7_PREF", "-0.12"))
        q7_pref = float(np.clip(q7_pref, self._q_lim_low[6], self._q_lim_up[6]))
        q_pref = q_mid.copy()
        q_pref[6] = q7_pref
        q_cur_pref = q_cur.copy()
        q_cur_pref[6] = q7_pref
        default_pref = self.DEFAULT_Q.copy()
        default_pref[6] = q7_pref
        seeds = [
            q_cur,
            q_cur_pref,
            default_pref,
            q_pref,
            np.array([0.0, 0.349066, 0.0, 1.151917, -1.493363, 0.139626, q7_pref]),
            q_cur + np.array([0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0]),
            q_cur + np.array([0.0, 0.0, -0.35, 0.0, 0.0, 0.0, 0.0]),
            q_cur + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45]),
            q_cur + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.45]),
        ]
        best = (None, np.inf, np.inf)
        for seed in seeds:
            q, ok, pos_err, rot_err = self._ikine_from_seed(Tep, seed)
            if ok:
                return q
            if pos_err + 0.08 * rot_err < best[1] + 0.08 * best[2]:
                best = (q, pos_err, rot_err)
        if best[0] is not None and best[1] < 0.003 and best[2] < 0.06:
            return best[0]
        return np.array([])

    def move_cartesian(self, T: SE3):
        q = self.ikine(T)
        assert len(q)  # inverse kinematics failure
        self.q0 = q[:]
