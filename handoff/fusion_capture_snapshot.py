"""Save the small source/config subset needed to reproduce collection offline."""
import hashlib
import json
from pathlib import Path
import shutil


def save_capture_snapshot(args, directory):
    root = Path(__file__).resolve().parents[1]
    destination = Path(directory)/'capture_snapshot'
    destination.mkdir(parents=True, exist_ok=False)
    paths = [root/name for name in (
        'scripts/collect_handover_shadow.sh', 'handoff/fusion_handover.py',
        'handoff/fusion_servo.py', 'handoff/fusion_observer.py', 'handoff/fusion_cli.py',
        'handoff/fusion_recording.py', 'handoff/fusion_capture_snapshot.py',
        'servo_delivery.py', 'targets/hand_selection.py',
        'targets/mediapipe_hand_landmarker.py', 'tools/train_handover_pull.py',
        'tools/check_handover_dataset.py', 'control_backends/force_ctrl/impedance_profile.py',
        'control_backends/force_ctrl/feedback_health.py')]
    for option in ('config', 'calib_json', 'handeye', 'camera_self_mask_calibration'):
        value = getattr(args, option, None)
        if value:
            paths.append(Path(value).expanduser().resolve())
    manifest = []
    for path in dict.fromkeys(paths):
        if not path.is_file():
            raise FileNotFoundError(f'collection snapshot source is missing: {path}')
        # Preserve project-relative names; external paths get an explicit bucket.
        relative = path.relative_to(root) if path.is_relative_to(root) else Path('external')/path.name
        target = destination/relative
        if target.exists():
            raise ValueError(f'collection snapshot filename collision: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest.append(dict(source=str(path), saved_as=str(relative),
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (destination/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return dict(directory='capture_snapshot', files=len(manifest))
