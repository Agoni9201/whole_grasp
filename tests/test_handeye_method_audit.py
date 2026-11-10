"""Known transforms verify offline refinement; no hardware or dataset access."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.audit_handeye_methods import refine
from tools.calibrate_d435i_charuco import make_transform


def observations(count=18, outlier=False):
    rng=np.random.default_rng(240)
    x=make_transform(Rotation.from_euler('xyz',[.2,1.3,-.1]).as_matrix(),[.07,-.073,.002])
    y=make_transform(Rotation.from_euler('xyz',[.1,-.1,.3]).as_matrix(),[-.4,.02,.01])
    points=np.array([[a,b,0.] for a in np.linspace(0,.18,7) for b in np.linspace(0,.12,5)],np.float32)
    K=np.array([[700.,0,640.],[0,700.,360.],[0,0,1.]])
    samples=[]
    for i in range(count):
        c=make_transform(Rotation.from_euler('xyz',rng.uniform(-.4,.4,3)).as_matrix(),
                         rng.uniform([-.07,-.05,.4],[.03,.04,.75]))
        base=y@np.linalg.inv(c)@np.linalg.inv(x)
        observed=c.copy()
        if outlier and i==0:
            observed[:3,:3]=Rotation.from_euler('x',8.,degrees=True).as_matrix()@observed[:3,:3]
            observed[:3,3]+=[.04,-.025,.025]
        pixels=cv2.projectPoints(points,cv2.Rodrigues(observed[:3,:3])[0],observed[:3,3],K,np.zeros(5))[0]
        samples.append(dict(T_base_flange=base,T_camera_target=observed,K=K,D=np.zeros(5),
                            charuco_ids=list(range(len(points))),image_points=pixels.reshape(-1,2)))
    return x,points,samples


@pytest.mark.parametrize('model',['pose','pixels'])
def test_lm_from_perturbed_seed_recovers_known_calibration(model):
    x,points,samples=observations();seed=x.copy()
    seed[:3,:3]=Rotation.from_euler('xyz',[2.,-3.,1.],degrees=True).as_matrix()@x[:3,:3]
    seed[:3,3]+=[.01,-.008,.005]
    actual,detail=refine(samples,seed,points,model=model)
    assert detail['success']
    assert np.linalg.norm(actual[:3,3]-x[:3,3])<2e-5
    assert (Rotation.from_matrix(actual[:3,:3]).inv()*Rotation.from_matrix(x[:3,:3])).magnitude()<2e-5


def test_huber_improves_known_truth_when_one_pose_is_corrupted():
    x,points,samples=observations(count=24,outlier=True)
    linear,_=refine(samples,x,points,method='trf',loss='linear')
    robust,detail=refine(samples,x,points,method='trf',loss='huber')
    def error(estimate):
        return np.linalg.norm(estimate[:3,3]-x[:3,3])/.005 + (
            Rotation.from_matrix(estimate[:3,:3]).inv()*Rotation.from_matrix(x[:3,:3])).magnitude()/np.deg2rad(1.)
    assert detail['success'] and error(robust)<error(linear)*.65


def test_budget_exhaustion_is_reported_without_accepting_convergence():
    x,points,samples=observations();seed=x.copy();seed[:3,3]+=[.02,.01,0.]
    _,detail=refine(samples,seed,points,method='trf',max_nfev=1)
    assert detail['success'] is False and detail['status']==0
    assert detail['nfev']==1
