#!/usr/bin/env python3
"""Complete a saved request using MGPC; no arm/camera/runtime imports."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    print('[mgpc-worker] request process started',flush=True)
    import faulthandler
    faulthandler.dump_traceback_later(45,repeat=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    opts=parser.parse_args()
    request=json.loads(opts.request.read_text())
    directory=opts.request.parent
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    from tools.mgpc_inference import load_model,predict
    started=time.monotonic()
    model=load_model(Path(request['repository']),Path(request['checkpoint']))
    print('[mgpc-worker] inference',flush=True)
    result=predict(model,np.load(directory/'input_normalized.npy'),directory/'target_rgb.png',
                   request['category'],request['seed'])
    np.save(directory/'prediction_normalized.npy',result)
    (directory/'worker_result.json').write_text(json.dumps(dict(points=len(result),
        category=request['category'],seconds=time.monotonic()-started,strict_state_dict=True,
        geometry_only=True))+'\n')


if __name__=='__main__': main()
