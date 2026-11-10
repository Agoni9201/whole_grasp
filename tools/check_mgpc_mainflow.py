#!/usr/bin/env python3
"""Replay the online MGPC adapter on saved data, without importing hardware."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/mgpc_mainflow_smoke')
    opts=parser.parse_args()
    from real_nero_grasp.cli import parse_args
    from real_nero_grasp.mgpc_completion import complete_graspnet_target
    saved_argv=sys.argv
    try:
        sys.argv=['mgpc-smoke','--config',str(ROOT/'config/pipeline.yaml'),
                  '--phase','grasp','--mgpc','--mgpc-category','bottle']
        args=parse_args()
    finally:sys.argv=saved_argv
    args.execute=False
    output=opts.output_dir;output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'artifacts/mgpc_compare/manifest.json').read_text())
    source=next(Path(manifest['source_run']).glob('*/multiview_raw/center*/metadata.json')).parent
    archive=output/'multiview_raw';archive.mkdir(exist_ok=True)
    link=archive/'center_smoke'
    if not link.exists():link.symlink_to(source.resolve(),target_is_directory=True)
    T=np.asarray(manifest['T_base_camera'])
    base=np.load(ROOT/'artifacts/mgpc_compare/multi/observed_base.npy')
    camera=(base-T[:3,3])@T[:3,:3]
    target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(camera))
    completed=complete_graspnet_target(target,T,None,args,output)
    np.testing.assert_array_equal(target.points,camera)
    assert len(completed.points)==8192
    actual=np.load(output/'mgpc/prediction_base.npy')
    expected=np.load(ROOT/'artifacts/mgpc_compare/multi/prediction_base.npy')
    difference=float(np.max(np.abs(actual-expected)))
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=0)
    report=dict(worker_completed=True,points=len(completed.points),measured_unchanged=True,
        maximum_coordinate_difference_from_offline_m=difference,hardware_connected=False,
        test_category='bottle from saved dataset; live cube performance not tested')
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
