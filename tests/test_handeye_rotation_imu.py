import json
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools import verify_handeye_rotation_imu as app


def test_gyro_integration_recovers_known_angle_and_removes_fixed_bias():
    t=np.linspace(0.,3.,601)
    bias=np.array([.012,-.007,.004])
    velocity=np.array([0.,0.,np.deg2rad(20.)])
    samples=np.column_stack((t,np.tile(velocity+bias,(len(t),1))))
    actual=app.integrate_gyro(samples,.123,2.123,bias)
    expected=Rotation.from_rotvec(velocity*2.)
    assert (expected.inv()*actual).magnitude()<1e-10


def test_gyro_integration_compensates_linear_bias_change_between_still_endpoints():
    t=np.linspace(0.,3.,601)
    b0=np.array([.012,-.007,.004]);drift=np.array([.002,.003,-.001])
    velocity=np.array([0.,0.,np.deg2rad(20.)])
    samples=np.column_stack((t,velocity+b0+t[:,None]*drift))
    a,b=.123,2.123
    actual=app.integrate_gyro(samples,a,b,b0+a*drift,b0+b*drift)
    assert (Rotation.from_rotvec(velocity*(b-a)).inv()*actual).magnitude()<1e-10


@pytest.mark.parametrize('samples,start,end', [
    ([[0,0,0,0],[0,0,0,0]],0,.1),
    ([[0,0,0,0],[1,0,0,0]],-.1,.5),
    ([[0,0,0,0],[1,0,0,0]],.1,1.5),
])
def test_uncovered_or_nonincreasing_clock_is_not_reported_as_valid_rotation(samples,start,end):
    with pytest.raises(ValueError,match='timestamps'):
        app.integrate_gyro(samples,start,end,[0,0,0])


@pytest.mark.parametrize('startup_bias',[0.,.01])
def test_camera_and_feedback_trial_refreshes_bias_after_a_long_ready_wait(tmp_path,monkeypatch,startup_bias):
    from cameras.realsense import Frame
    board,*_=app.load_board_config('config/charuco_board.yaml')
    gray=board.generateImage((640,480),marginSize=85)
    image0=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    image1=cv2.warpAffine(image0,cv2.getRotationMatrix2D((320,240),20.,1.),(640,480),borderValue=(255,255,255))
    calls=dict(frame=-1,sensor_stopped=False,camera_closed=False,feedback_closed=False)
    class Sensor:
        def get_stream_profile_list(self): return NS(get_stream_profile_by_index=lambda _:object())
        def start(self,profile,callback): self.callback=callback
        def stop(self): calls['sensor_stopped']=True
        def emit(self,t,w):
            frame=NS(get_timestamp_us=lambda:int(round(t*1e6)),get_value=lambda:NS(x=0.,y=0.,z=w))
            frame.as_gyro_frame=lambda:frame
            self.callback(frame)
    sensor=Sensor()
    class Camera:
        serial='synthetic'
        pipeline=NS(get_device=lambda:NS(get_sensor=lambda _:sensor))
        def __enter__(self): return self
        def __exit__(self,*a): calls['camera_closed']=True
        def read(self):
            calls['frame']+=1;i=calls['frame']
            if i==0:
                for t in np.linspace(0.,2.,41): sensor.emit(t,startup_bias)
            elif i==1:
                for t in np.linspace(2.01,202.1,2002): sensor.emit(t,0.)
            elif i==2:
                for t in np.linspace(202.11,203.2,110): sensor.emit(t,np.deg2rad(20.))
            elif i==3:
                for t in np.linspace(203.21,205.3,210): sensor.emit(t,0.)
            return Frame(image0 if i<=1 else image1,None,NS(),
                np.array([[800.,0,320.],[0,800.,240.],[0,0,1.]]),np.zeros(5),i,
                2100. if i==0 else 202110. if i==1 else 203110.,100.+i,'synthetic')
    class ReadOnly:
        def __init__(self,*a): pass
        def __enter__(self): return self
        def __exit__(self,*a): calls['feedback_closed']=True
        def read(self):
            T=np.eye(4)
            T[:3,:3]=Rotation.from_euler('z',0. if calls['frame']<=1 else 20.,degrees=True).as_matrix()
            return np.zeros(6),T,1.
        def read_joints(self): return np.zeros(7),1.
        # No controller, enable, motion, stop, or gripper methods exist.
    monkeypatch.setattr(app,'parse_args',lambda _:NS(board='config/charuco_board.yaml',
        camera_backend='orbbec',camera_serial='synthetic',camera_width=640,camera_height=480,
        camera_fps=30,camera_timeout=2.,camera_warmup=0,channel='can0',firmware='V111',save_dir=tmp_path))
    import sys
    monkeypatch.setitem(sys.modules,'pyorbbecsdk',NS(OBSensorType=NS(GYRO_SENSOR='gyro')))
    monkeypatch.setattr(app,'create_camera',lambda **kw:Camera())
    monkeypatch.setattr(app,'NeroReadOnlyFeedback',ReadOnly)
    clock=NS(t=0.)
    def tick(): clock.t+=3.;return clock.t
    monkeypatch.setattr(app.time,'monotonic',tick)
    monkeypatch.setattr(app.cv2,'namedWindow',lambda *a:None)
    monkeypatch.setattr(app.cv2,'resizeWindow',lambda *a:None)
    monkeypatch.setattr(app.cv2,'imshow',lambda *a:None)
    monkeypatch.setattr(app.cv2,'destroyAllWindows',lambda:None)
    monkeypatch.setattr(app.cv2,'getWindowProperty',lambda *a:1.)
    monkeypatch.setattr(app.select,'select',lambda *a:([],[],[]))
    keys=iter([-1,ord('s'),ord('s'),-1,ord('q')]);monkeypatch.setattr(app.cv2,'waitKey',lambda _:next(keys))
    app.main()
    report=json.loads(next(tmp_path.glob('*/result.json')).read_text())
    assert report['robot_commands_sent'] is False
    assert report['gyro_ready_bias_rad_s']==pytest.approx([0.,0.,startup_bias])
    assert report['gyro_bias_rad_s']==pytest.approx([0.,0.,0.])
    assert report['angles']['robot_deg']==pytest.approx(20.,abs=1e-7)
    assert report['angles']['gyro_deg']==pytest.approx(20.,abs=1e-7)
    assert report['angles']['vision_deg']==pytest.approx(20.,abs=.3)
    assert calls['sensor_stopped'] and calls['camera_closed'] and calls['feedback_closed']
