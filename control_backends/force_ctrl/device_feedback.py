"""Read-only descriptions of SDK feedback, distinct from host commands.

The Nero SDK getters used here return received messages; they do not request a
mode change, query over CAN, or send an enable/reset command.
"""
import math
import time


def _scalar(obj, name, convert):
    try:
        value = getattr(obj, name)
        return None if value is None else convert(value)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _stamp(message, wall_time):
    stamp = _scalar(message, 'timestamp', float)
    if stamp is None or not math.isfinite(stamp) or stamp <= 0:
        return {'timestamp': None, 'age_ms': None, 'fresh': False}
    age = wall_time - stamp
    return {'timestamp': stamp, 'age_ms': age * 1000,
            'fresh': 0 <= age <= .5}


def describe_device_feedback(status_feedback, driver_feedback, wall_time):
    status = getattr(status_feedback, 'msg', None)
    arm = _stamp(status_feedback, wall_time)
    for name in ('ctrl_mode', 'arm_status', 'mode_feedback', 'teach_status',
                 'motion_status', 'err_code'):
        arm[name] = _scalar(status, name, int)
    drivers = []
    for index, wrapped in enumerate(driver_feedback, 1):
        item = dict(joint=index, **_stamp(wrapped, wall_time))
        foc = getattr(getattr(wrapped, 'msg', None), 'foc_status', None)
        for name in ('driver_enable_status', 'driver_error_status', 'stall_status',
                     'collision_status', 'voltage_too_low', 'motor_overheating',
                     'driver_overcurrent', 'driver_overheating'):
            item[name] = _scalar(foc, name, bool)
        drivers.append(item)
    return {'sampled_wall_time': wall_time, 'status': arm, 'drivers': drivers}


def read_device_feedback(robot):
    errors = []

    def read(method, *args):
        try:
            return getattr(robot, method)(*args)
        except Exception as exc:
            errors.append(f'{method}{args}: {type(exc).__name__}: {exc}')
            return None

    status = read('get_arm_status')
    drivers = [read('get_driver_states', index) for index in range(1, 8)]
    result = describe_device_feedback(status, drivers, time.time())
    result['read_errors'] = errors
    return result
