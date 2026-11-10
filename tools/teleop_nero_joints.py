import os
import threading
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(ROOT_DIR, "manipulator_grasp", "assets", "scenes", "scene.xml")

JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
GRIPPER_CTRL = 7
STEP_SMALL = np.deg2rad(2.0)
STEP_LARGE = np.deg2rad(10.0)
DEFAULT_Q = np.array([1.570796, 0.349066, 0.0, 1.151917, -1.493363, 0.139626, 0.0])


def _joint_qpos_addr(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    return int(model.jnt_qposadr[jid])


def _joint_range(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    return np.array(model.jnt_range[jid], dtype=float)


def _free_joint_qpos_slice(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    addr = int(model.jnt_qposadr[jid])
    return slice(addr, addr + 7)


def _attach_gripper_to_flange(model, data):
    flange_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "flange")
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "attach")
    if flange_id < 0 or eq_id < 0:
        return

    mujoco.mj_forward(model, data)
    data.qpos[_free_joint_qpos_slice(model, "2f85")] = np.r_[data.xpos[flange_id], data.xquat[flange_id]]
    model.equality("attach").data = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=float,
    )
    model.equality("attach").solimp = np.array([0.99, 0.99, 0.001, 0.5, 1.0], dtype=float)
    model.equality("attach").solref = np.array([0.0001, 1.0], dtype=float)
    data.eq_active[eq_id] = 1
    mujoco.mj_forward(model, data)


def _print_help():
    print(
        "\nNero7 terminal teleop\n"
        "Viewer window is display-only; type commands in this terminal.\n\n"
        "Commands:\n"
        "  1..7 or j N     select joint N\n"
        "  + / -           selected joint +/- 2 deg\n"
        "  ++ / --         selected joint +/- 10 deg\n"
        "  set N VALUE     set joint N to VALUE radians\n"
        "  deg N VALUE     set joint N to VALUE degrees\n"
        "  p               print current qpos and ctrl\n"
        "  s               save current qpos to nero_initial_q.npy\n"
        "  r               reset arm joints to zero\n"
        "  h               show this help\n"
        "  q               quit\n"
    )


def _input_worker(pending, pending_lock, running):
    while running["value"]:
        try:
            line = input("nero> ").strip()
        except EOFError:
            line = "q"
        with pending_lock:
            pending.append(line)
        if line.lower() in {"q", "quit", "exit"}:
            break


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    joint_addrs = [_joint_qpos_addr(model, name) for name in JOINT_NAMES]
    joint_ranges = np.array([_joint_range(model, name) for name in JOINT_NAMES])
    selected = {"idx": 0}
    pending = deque()
    pending_lock = threading.Lock()
    running = {"value": True}

    _attach_gripper_to_flange(model, data)
    for i, addr in enumerate(joint_addrs):
        data.qpos[addr] = DEFAULT_Q[i]
        data.ctrl[i] = DEFAULT_Q[i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    def read_qpos():
        return np.array([data.qpos[addr] for addr in joint_addrs], dtype=float)

    def read_ctrl():
        return np.array(data.ctrl[: len(JOINT_NAMES)], dtype=float)

    def apply_q(q):
        q = np.clip(np.asarray(q, dtype=float).reshape(len(JOINT_NAMES)), joint_ranges[:, 0], joint_ranges[:, 1])
        for i, addr in enumerate(joint_addrs):
            data.qpos[addr] = q[i]
            data.ctrl[i] = q[i]
        data.qvel[:] = 0.0
        if model.nu > GRIPPER_CTRL:
            data.ctrl[GRIPPER_CTRL] = 0.0
        mujoco.mj_forward(model, data)
        return q

    def print_q(prefix="current"):
        qpos = read_qpos()
        ctrl = read_ctrl()
        qpos_arr = ", ".join(f"{v:.6f}" for v in qpos)
        ctrl_arr = ", ".join(f"{v:.6f}" for v in ctrl)
        deg = ", ".join(f"{np.rad2deg(v):.1f}" for v in qpos)
        print(f"{prefix}_qpos_rad = np.array([{qpos_arr}])")
        print(f"{prefix}_qpos_deg = [{deg}]")
        print(f"{prefix}_ctrl_rad = np.array([{ctrl_arr}])")

    def update_joint(delta):
        q = read_qpos()
        i = selected["idx"]
        q[i] = float(np.clip(q[i] + delta, joint_ranges[i, 0], joint_ranges[i, 1]))
        apply_q(q)
        print(f"joint{i + 1}: {q[i]: .4f} rad ({np.rad2deg(q[i]): .1f} deg)")

    def set_joint(index, value_rad):
        q = read_qpos()
        i = int(index) - 1
        if not 0 <= i < len(JOINT_NAMES):
            print("joint index must be 1..7")
            return
        selected["idx"] = i
        q[i] = float(np.clip(value_rad, joint_ranges[i, 0], joint_ranges[i, 1]))
        apply_q(q)
        print(f"joint{i + 1}: {q[i]: .4f} rad ({np.rad2deg(q[i]): .1f} deg)")

    def process_command(line):
        if not line:
            return
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in {"q", "quit", "exit"}:
            running["value"] = False
        elif cmd in {"h", "help", "?"}:
            _print_help()
        elif cmd in {"+", "plus"}:
            update_joint(STEP_SMALL)
        elif cmd in {"-", "minus"}:
            update_joint(-STEP_SMALL)
        elif cmd in {"++"}:
            update_joint(STEP_LARGE)
        elif cmd in {"--"}:
            update_joint(-STEP_LARGE)
        elif cmd in {"p", "print"}:
            print_q("current")
        elif cmd in {"s", "save"}:
            out = os.path.join(ROOT_DIR, "nero_initial_q.npy")
            qpos = read_qpos()
            np.save(out, qpos)
            print_q("saved")
            print(f"saved to {out}")
        elif cmd in {"r", "reset"}:
            apply_q(np.zeros(len(JOINT_NAMES), dtype=float))
            print("reset arm joints to zero")
        elif cmd.isdigit() and 1 <= int(cmd) <= 7:
            selected["idx"] = int(cmd) - 1
            print(f"selected joint{selected['idx'] + 1}")
        elif cmd == "j" and len(parts) >= 2 and parts[1].isdigit():
            idx = int(parts[1])
            if 1 <= idx <= 7:
                selected["idx"] = idx - 1
                print(f"selected joint{idx}")
            else:
                print("joint index must be 1..7")
        elif cmd == "set" and len(parts) >= 3:
            set_joint(parts[1], float(parts[2]))
        elif cmd == "deg" and len(parts) >= 3:
            set_joint(parts[1], np.deg2rad(float(parts[2])))
        else:
            print(f"unknown command: {line}")
            print("type h for help")

    def process_pending():
        with pending_lock:
            commands = list(pending)
            pending.clear()
        for line in commands:
            try:
                process_command(line)
            except Exception as exc:
                print(f"command failed: {line!r}: {type(exc).__name__}: {exc}")

    _print_help()
    target_q = apply_q(read_qpos())

    input_thread = threading.Thread(target=_input_worker, args=(pending, pending_lock, running), daemon=True)
    input_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.8, 0.6, 1.1]
        viewer.cam.azimuth = 210
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.4
        while viewer.is_running() and running["value"]:
            process_pending()
            # Kinematic pose viewer: keep the selected arm pose fixed and avoid
            # gravity/constraint dynamics moving other joints while tuning q.
            target_q = apply_q(read_qpos())
            viewer.sync()
            time.sleep(0.02)

    running["value"] = False
    print_q("final")


if __name__ == "__main__":
    main()
