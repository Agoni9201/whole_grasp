"""Shared SDK profile and YAML options; CLI overrides YAML defaults."""
import argparse
from pathlib import Path


def add_camera_arguments(p):
    p.add_argument('--config', default=None, help='Optional SDK pipeline YAML; CLI overrides YAML')
    p.add_argument('--camera-backend', choices=('realsense', 'orbbec'), default='realsense')
    p.add_argument('--camera-serial', default=None)
    p.add_argument('--camera-width', type=int, default=640)
    p.add_argument('--camera-height', type=int, default=480)
    p.add_argument('--camera-fps', type=int, default=30)
    p.add_argument('--camera-depth-width', type=int, default=None,
                   help='Native depth width; defaults to the color width when omitted')
    p.add_argument('--camera-depth-height', type=int, default=None,
                   help='Native depth height; defaults to the color height when omitted')
    p.add_argument('--camera-depth-fps', type=int, default=None,
                   help='Native depth frame rate; defaults to the color frame rate when omitted')
    p.add_argument('--camera-depth-filters', choices=('off', 'all'), default='off',
                   help='RealSense: SDK filter chain; Orbbec: noise removal only, without hole filling')
    p.add_argument('--camera-timeout', type=float, default=10.)
    p.add_argument('--camera-warmup', type=int, default=30)
    p.add_argument(
        '--camera-depth-preset',
        choices=('high_density', 'unchanged'),
        default='high_density',
        help='High Density depth preset; unchanged leaves the device setting untouched',
    )
    p.add_argument(
        '--camera-emitter-enabled',
        dest='camera_emitter_enabled',
        action='store_true',
        help='Enable the infrared emitter after applying the depth preset',
    )
    p.add_argument(
        '--camera-emitter-disabled',
        dest='camera_emitter_enabled',
        action='store_false',
        help='Disable the infrared emitter after applying the depth preset',
    )
    p.set_defaults(camera_emitter_enabled=None)
    p.add_argument(
        '--camera-laser-power',
        type=float,
        default=None,
        help='D435i laser power in the device-supported range (commonly 0..360)',
    )


def apply_pipeline_defaults(p, argv=None):
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    opts,_=probe.parse_known_args(argv)
    if not opts.config: return
    import yaml
    path=Path(opts.config).expanduser().resolve()
    data=yaml.safe_load(path.read_text()) or {}
    if data.get('global_camera',{}).get('enabled',False):
        p.error('Only one wrist camera is supported; disable global_camera')
    wrist=data.get('wrist_camera',{})
    if not wrist.get('enabled',True) or wrist.get('backend','realsense') not in ('realsense', 'orbbec'):
        p.error('Enable a realsense or orbbec wrist SDK backend')
    defaults={}
    grasp=data.get('grasp',{})
    if 'post_grasp_hold_joints_deg' in grasp:
        defaults['post_grasp_hold_joints_deg']=grasp['post_grasp_hold_joints_deg']
    for key in (
        'backend', 'serial', 'width', 'height', 'fps', 'timeout', 'warmup',
        'depth_width', 'depth_height', 'depth_fps', 'depth_filters',
        'depth_preset', 'emitter_enabled', 'laser_power',
    ):
        if key in wrist: defaults['camera_'+key]=wrist[key]
    mapping={'grasp_calib_json':'calib_json','grasp_convention':'handeye_convention',
             'self_mask':'camera_self_mask','self_mask_calibration':'camera_self_mask_calibration',
             'servo_calib_json':'handeye','servo_convention':'wrist_handeye_convention',
             'servo_mount_frame':'wrist_handeye_mount_frame'}
    for key,dest in mapping.items():
        if wrist.get(key) is not None:
            value=wrist[key]
            if key.endswith('calib_json') or key == 'self_mask_calibration':
                value=str((path.parent/Path(value).expanduser()).resolve())
            defaults[dest]=value
    # for key in ('target_source','max_linear_vel','max_joint_vel'):
        # if key in data.get('servo',{}): defaults[key]=data['servo'][key]
    servo=data.get('servo',{})
    for key in (
        'target_source','max_linear_vel','max_joint_vel',
        'hand_min_detection_confidence','hand_min_presence_confidence',
        'hand_min_tracking_confidence','hand_depth_radius',
        'hand_min_depth_landmarks','hand_min_depth','hand_max_depth',
        'hand_offset_base','hand_approach_mode',
        'target_filter_alpha','normal_filter_alpha','target_blend_alpha','twist_smooth_alpha',
        'follow_min_qdot','follow_kp_rise_time',
        'hand_camera_center','hand_center_gain','hand_center_max_joint_accel',
        'delivery_distance','gesture_handover','gesture_extra_close_mm',
        'gesture_open_seconds','gesture_close_seconds','gesture_near_tolerance',
        'gesture_stable_seconds','gesture_still_tolerance_mm',
        'gesture_ready_distance','hand_memory_approach',
    ):
        if key in servo:
            defaults[key]=servo[key]
    if servo.get('hand_model') is not None:
        model_path=Path(servo['hand_model']).expanduser()
        defaults['hand_model']=str(
            model_path if model_path.is_absolute()
            else (path.parent/model_path).resolve()
        )
    known={a.dest for a in p._actions}
    p.set_defaults(**{k:v for k,v in defaults.items() if k in known})
