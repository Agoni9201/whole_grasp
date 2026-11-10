import argparse
import math
from pathlib import Path

DEFAULT_HAND_MODEL = (
    Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
)


def parse_follow_kp(value):
    if isinstance(value, str):
        parts = value.split(",")
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = [value]
    try:
        gains = tuple(float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("follow-kp must contain one or seven finite nonnegative gains") from exc
    if len(gains) == 1:
        gains *= 7
    if len(gains) != 7 or not all(math.isfinite(gain) and gain >= 0.0 for gain in gains):
        raise argparse.ArgumentTypeError("follow-kp must contain one or seven finite nonnegative gains")
    return gains


def nonnegative_finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def filter_alpha(value):
    result = nonnegative_finite_float(value)
    if not 0.0 < result <= 1.0:
        raise argparse.ArgumentTypeError("filter alpha must be in (0, 1]")
    return result


def servo_parser():
    p = argparse.ArgumentParser(add_help=False)
    from handoff.fusion_cli import add_fusion_arguments
    add_fusion_arguments(p)
    p.add_argument("--execute", action="store_true", help="真实执行门槛（缺省 dry-run）")
    p.add_argument("--target-source", choices=["aruco", "mediapipe"], default="aruco")
    p.add_argument("--hand-model", default=str(DEFAULT_HAND_MODEL))
    # VIDEO-mode tracking is more stable with a little confidence hysteresis;
    # the 0.5 defaults caused repeated no_hand_2d gaps during live motion.
    p.add_argument("--hand-min-detection-confidence", type=float, default=0.35)
    p.add_argument("--hand-min-presence-confidence", type=float, default=0.35)
    p.add_argument("--hand-min-tracking-confidence", type=float, default=0.35)
    p.add_argument("--hand-depth-radius", type=int, default=4)
    p.add_argument("--hand-min-depth-landmarks", type=int, default=3)
    p.add_argument("--hand-min-depth", type=float, default=0.10)
    p.add_argument("--hand-max-depth", type=float, default=2.0)
    p.add_argument("--hand-offset-base", default="0,0,0")
    p.add_argument(
        "--hand-approach-mode",
        choices=["camera_ray", "palm_normal"],
        default="camera_ray",
        help="首轮使用 camera_ray；palm_normal 需另行验证姿态稳定性",
    )
    p.add_argument("--wrist-image-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--wrist-info-topic", default="/camera/camera/color/camera_info")
    p.add_argument("--handeye", default=None,
                   help="腕部相机手眼标定（伺服工程已验证的那套）")
    p.add_argument("--wrist-handeye-convention", default="link7_cam")
    p.add_argument("--wrist-handeye-mount-frame", default="feedback_link7")
    p.add_argument("--offset-base", default="0.02,-0.025,0")
    p.add_argument("--aruco-dict", default="ARUCO_ROS_ALL")
    p.add_argument("--marker-id", type=int, default=582)
    p.add_argument("--marker-size", type=float, default=0.0677)
    p.add_argument("--servo-hz", type=float, default=30.0)
    p.add_argument("--target-filter-alpha", type=filter_alpha, default=0.34,
                   help="PBVS position measurement weight; higher values reduce lag and pass more noise")
    p.add_argument("--normal-filter-alpha", type=filter_alpha, default=0.24,
                   help="PBVS approach-direction measurement weight; higher values reduce direction lag")
    p.add_argument("--target-blend-alpha", type=filter_alpha, default=0.82,
                   help="PBVS accepted position target blend, matching the original servo CLI")
    p.add_argument("--twist-smooth-alpha", type=filter_alpha, default=0.72,
                   help="PBVS velocity update weight, matching the original servo CLI")
    p.add_argument("--delivery-distance", type=float, default=0.15)
    p.add_argument("--max-linear-vel", type=float, default=0.03)
    p.add_argument("--max-angular-vel", type=float, default=0.6)
    p.add_argument("--max-joint-vel", type=nonnegative_finite_float, default=0.08)
    p.add_argument("--visual-orientation", action="store_true", default=False)
    p.add_argument('--hand-camera-center', action='store_true',
                   help='MediaPipe: coordinate all seven joints to center the palm and track TCP-hand distance')
    p.add_argument('--hand-center-gain', type=nonnegative_finite_float, default=2.,
                   help='Image/range feature convergence gain (1/s) for hand-camera-center')
    p.add_argument('--hand-center-max-joint-accel', type=nonnegative_finite_float, default=.5,
                   help='Joint reference acceleration bound (rad/s^2) for hand-camera-center')
    p.add_argument("--approach-error-m", type=float, default=0.03)
    p.add_argument("--approach-error-deg", type=float, default=5.0)
    p.add_argument("--handover-distance", type=float, default=0.10)
    p.add_argument("--stable-frames", type=int, default=10)
    p.add_argument("--handover-stable-frames", type=int, default=20)
    p.add_argument("--servo-follow-only", action="store_true",
                   help="Track at delivery-distance only; do not enter handover, open gripper, or retreat")
    p.add_argument('--gesture-handover', action='store_true',
                   help='Stable receiver permits gesture learning; follow again if receiver moves before release; first opening permanently holds arm')
    p.add_argument('--gesture-ready-distance', type=nonnegative_finite_float, default=.15,
                   help='Maximum measured TCP-palm distance to establish handover readiness (m); separate from follow distance')
    p.add_argument('--hand-memory-approach', action=argparse.BooleanOptionalAction, default=True,
                   help='Gesture mode only: bounded approach to last stable BASE position during short depth-only loss')
    p.add_argument('--gesture-stable-seconds', type=nonnegative_finite_float, default=3.,
                   help='Continuous receiver stillness time before latching handover readiness (seconds)')
    p.add_argument('--gesture-still-tolerance-mm', type=nonnegative_finite_float, default=10.,
                   help='Maximum BASE-frame palm position bounding-box diagonal during the stillness window (mm)')
    p.add_argument('--gesture-extra-close-mm', type=nonnegative_finite_float, default=1.)
    p.add_argument('--gesture-open-seconds', type=nonnegative_finite_float, default=5.)
    p.add_argument('--gesture-close-seconds', type=nonnegative_finite_float, default=2.)
    p.add_argument('--gesture-near-tolerance', type=nonnegative_finite_float, default=.01)
    p.add_argument('--gesture-preview-grip-width-mm', type=nonnegative_finite_float,
                   help='Dry-run only: explicitly supply simulated initial cup grip opening; never a hardware baseline')
    p.add_argument("--servo-preview", action="store_true",
                   help="Show current RGB/hand/depth and servo hold reason; Q/Esc closes via controlled shutdown")
    p.add_argument("--servo-diagnostics", action="store_true",
                   help="Record 1 Hz read-only servo motion/torque diagnostics under logs and print asynchronously")
    p.add_argument("--impedance-max-position-error", type=float, default=0.015)
    p.add_argument("--impedance-max-orientation-error-deg", type=float, default=5.0)
    p.add_argument("--impedance-target-linear-vel", type=float, default=0.03)
    p.add_argument("--impedance-target-angular-vel", type=float, default=0.5)
    p.add_argument("--cart-k-scale", type=float, default=1.0)
    p.add_argument("--cart-b-scale", type=float, default=1.0)
    p.add_argument("--follow-kp", type=parse_follow_kp, default=(2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0),
                   help="低增益固件关节跟随辅助（逗号分隔 7 个或单值；0 关闭）")
    p.add_argument("--follow-min-qdot", type=nonnegative_finite_float, default=0.02,
                   help="跟随辅助达到完整增益的参考关节速度（rad/s）")
    p.add_argument("--follow-kp-rise-time", type=nonnegative_finite_float, default=0.20,
                   help="跟随辅助从零升到设置上限的最短时间（s；0 关闭上升限速，释放不延迟）")
    p.add_argument("--follow-max-error-rad", type=nonnegative_finite_float, default=0.03,
                   help="关节跟随参考领先实际关节的角度上限（rad；独立于速度上限）")
    p.add_argument("--follow-release-error", type=nonnegative_finite_float, default=0.02,
                   help="实际关节被推到参考前方时释放辅助的误差（rad；0 关闭此释放条件）")
    p.add_argument("--dls-lambda", type=float, default=0.05)
    p.add_argument("--use-dls", action="store_true", default=False)
    p.add_argument("--vel-filter-tau", type=float, default=0.008)
    return p

def add_servo_arguments(parser):
    for action in servo_parser()._actions:
        if action.option_strings and not any(x in parser._option_string_actions for x in action.option_strings):
            parser._add_action(action)
