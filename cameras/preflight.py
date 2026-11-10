"""Read-only preflight performed before any SDK arm connection."""
from pathlib import Path
import numpy as np

# 自检
def grasp_preflight(args):
    from real_nero_grasp.mgpc_completion import validate_mgpc_configuration
    validate_mgpc_configuration(args)
    from real_nero_grasp.camera_self_mask import validate_self_mask_settings
    validate_self_mask_settings(args)
    if not getattr(args, 'execute', False):
        return
    from tools.real_nero_point_test import handeye_transform
    transform = handeye_transform(args.calib_json, args.handeye_convention)
    if not np.isfinite(transform).all():
        raise ValueError('Non-finite wrist calibration')
    for path in (args.yolo_weights, args.ik_urdf):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    root = Path(__file__).resolve().parents[1]
    checkpoint = root/'logs/log_rs/checkpoint-rs.tar'
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1000:
        raise FileNotFoundError(f'Missing real GraspNet checkpoint: {checkpoint}')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GraspNet requires CUDA. Fix PyTorch/driver compatibility before --execute.')
    from real_nero_grasp.ik_solver import _get_ik_solver
    _get_ik_solver(args)
    from real_nero_grasp.visualization import get_graspnet_helpers
    get_graspnet_helpers()
