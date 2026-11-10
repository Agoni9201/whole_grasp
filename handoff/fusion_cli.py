"""Shared opt-in handover arguments; validation does not touch hardware."""
from dataclasses import asdict
from pathlib import Path
import re

from .fusion_handover import FusionConfig, LinearPullModel
from .fusion_observer import validate_roi


def add_fusion_arguments(p):
    p.add_argument('--fusion-handover', choices=('shadow', 'release'), default=None,
                   help='External RGB + new seven-joint pull handover; shadow never opens')
    p.add_argument('--fusion-global-serial', default=None, help='Explicit D435/D435i/D455/D457 RGB observer serial')
    p.add_argument('--fusion-receiver-side', choices=('Left', 'Right'), default=None,
                   help='Receiver side as labeled in EXTERNAL unmirrored view; independent of wrist roles')
    p.add_argument('--fusion-roi', nargs=4, type=float, default=None,
                   metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'), help='Optional initial normalized ROI; reconfirm with R after hold')
    p.add_argument('--fusion-pull-model', default=None, help='Explicit trained linear action model JSON; absent=record only')
    p.add_argument('--fusion-collection-group', default=None,
                   help='Physical grasp identifier; reuse across restarts of the same grasp')
    p.add_argument('--fusion-collection-split', choices=('train', 'validation', 'test'), default=None,
                   help='Planned dataset partition; same physical grasp must stay in one partition')
    defaults = asdict(FusionConfig())
    options = {
        'ready_distance': 'ready-distance', 'ready_seconds': 'ready-seconds',
        'ready_image_radius_fraction': 'ready-image-radius-fraction',
        'baseline_seconds': 'baseline-seconds', 'baseline_max_std_nm': 'baseline-max-std-nm',
        'baseline_max_change_nm': 'baseline-max-change-nm', 'confirm_seconds': 'confirm-seconds',
        'window_seconds': 'window-seconds', 'open_seconds': 'open-seconds',
    }
    for field, name in options.items():
        p.add_argument('--fusion-'+name, dest='fusion_'+field, type=float, default=defaults[field])
    p.add_argument('--fusion-preview-grip-width-mm', type=float, default=None,
                   help='Dry-run only, simulated held opening in mm; never used on hardware')


def fusion_config(args):
    values = {key: getattr(args, 'fusion_'+key, default) for key, default in asdict(FusionConfig()).items()}
    values['maximum_width'] = min(values['maximum_width'],
        float(getattr(args, 'gripper_open_width', .085)),
        float(getattr(args, 'gripper_hardware_max_width', .085)))
    return FusionConfig(**values)


def validate_fusion_args(args):
    if not getattr(args, 'fusion_handover', None):
        return
    group = getattr(args, 'fusion_collection_group', None)
    split = getattr(args, 'fusion_collection_split', None)
    if bool(group) != bool(split):
        raise ValueError('collection group and split must be supplied together')
    if group is not None:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', group):
            raise ValueError('collection group must be a short alphanumeric identifier')
        if args.fusion_handover != 'shadow':
            raise ValueError('dataset collection identifiers require shadow mode')
    if getattr(args, 'phase', 'servo') != 'servo' and not getattr(args, 'enable_servo_delivery', False):
        raise ValueError('--fusion-handover requires --phase servo or --enable-servo-delivery')
    if getattr(args, 'gesture_handover', False) or getattr(args, 'servo_follow_only', False):
        raise ValueError('--fusion-handover cannot share gesture/follow-only gripper ownership')
    if getattr(args, 'target_source', None) != 'mediapipe' or not getattr(args, 'hand_camera_center', False):
        raise ValueError('--fusion-handover requires --target-source mediapipe --hand-camera-center')
    if not getattr(args, 'fusion_global_serial', None) or not getattr(args, 'fusion_receiver_side', None):
        raise ValueError('fusion requires --fusion-global-serial and --fusion-receiver-side')
    if (getattr(args, 'camera_backend', None) == 'realsense'
            and (not getattr(args, 'camera_serial', None)
                 or args.camera_serial == args.fusion_global_serial)):
        raise ValueError('two RealSense sources require distinct explicit serials')
    config = fusion_config(args)
    if config.ready_distance < float(getattr(args, 'delivery_distance', .15)):
        raise ValueError('fusion ready-distance must be >= delivery-distance')
    if args.fusion_roi is not None:
        validate_roi(args.fusion_roi)
    path = getattr(args, 'fusion_pull_model', None)
    if path:
        model = LinearPullModel.load(path)
        if abs(float(model.data.get('window_seconds', -1))-config.window_seconds) > 1e-9:
            raise ValueError('fusion model and runtime feature window must match')
        if (args.fusion_handover == 'release'
                and model.data.get('control_mode') != 'mit_cartesian_hold'):
            raise ValueError('fusion release requires a model collected in the same MIT holding mode')
    elif args.fusion_handover == 'release':
        raise ValueError('fusion release requires --fusion-pull-model; collect shadow data first')
    simulated = getattr(args, 'fusion_preview_grip_width_mm', None)
    if getattr(args, 'execute', False):
        if simulated is not None:
            raise ValueError('fusion simulated width is dry-run only')
    elif simulated is None or not 0 < simulated < 85:
        raise ValueError('dry-run fusion requires --fusion-preview-grip-width-mm in (0,85)')
    if not Path(args.hand_model).expanduser().is_file():
        raise ValueError('fusion requires a local hand landmarker model')
