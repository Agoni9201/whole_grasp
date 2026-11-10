"""Optional URDF-mesh gripper collision.

Colleague code imports this module.  This tree uses capsule fingers instead,
so URDF mesh collision stays off unless a later mesh loader is added.
"""


def urdf_gripper_enabled(args=None):
    return bool(getattr(args, "urdf_gripper_collision", False)) if args is not None else False


def read_sdk_gripper_opening(arm, args=None):
    try:
        status = arm.get_gripper_status()
    except Exception:
        return None
    for key in ("value", "width", "opening"):
        if isinstance(status, dict) and key in status:
            try:
                return float(status[key])
            except (TypeError, ValueError):
                continue
        value = getattr(status, key, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def initialize_gripper_geometry(args=None):
    return None
