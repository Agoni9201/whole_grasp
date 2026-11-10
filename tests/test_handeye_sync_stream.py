"""Clock and frame-order diagnostics with known traces; no USB/CAN."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tools.diagnose_handeye_sync import TimingCamera, fit_clock, summarize, record_joints


def trace():
    times=np.arange(120)/30.
    callbacks=[dict(sequence=i+100, sensor_us=int(round((50+t)*1e6)),
                    system_us=int(round((1_790_000_000+t)*1e6)), global_us=0,
                    callback_monotonic_s=200+t, callback_wall_s=1_790_000_000+t)
               for i,t in enumerate(times)]
    consumed=[dict(r,callback_to_read_ms=2.) for r in callbacks[::2]]
    joints=[dict(request_monotonic_s=200+i*.01,joint_feedback_wall_s=1_790_000_000+(i//2)*.02,
                 joint_feedback_age_ms=1.,q_rad=[0.]*7) for i in range(400)]
    return callbacks,consumed,joints


def test_clock_fit_recovers_rate_without_claiming_latency_or_motion_offset():
    sensor=np.linspace(50.,70.,601)
    host=1000.+(sensor-50)*1.000025+.12
    r=fit_clock(sensor*1e6,host)
    assert r['rate_difference_ppm']==pytest.approx(25.,abs=1e-5)
    assert r['residual_peak_to_peak_ms']<1e-6
    a,b,c=trace();s=summarize(a,b,c,30)
    assert s['camera_to_motion_time_offset_s'] is None
    assert s['old_dataset_repairable'] is None


def test_forward_skips_are_distinct_from_old_frames_and_polling_from_fresh_feedback():
    r=summarize(*trace(),30)
    assert r['consumed_sequence_gap_frames']==59
    assert r['consumed_sequence_nonincreasing']==0
    assert r['callback_sequence_nonincreasing']==0
    assert r['joint_unique_feedback']==200
    assert r['joint_unique_feedback_observed_hz']<r['joint_poll_hz']*.51


def test_timestamp_reversal_host_clock_jump_and_queue_delay_are_exposed():
    a,b,c=trace()
    a[20]['sensor_us']=a[19]['sensor_us']-1000
    a[50]['sequence']=a[49]['sequence']-1
    for row in a[60:]: row['callback_wall_s']+=.1
    b[5]['callback_to_read_ms']=150.
    r=summarize(a,b,c,30)
    assert r['device_timestamp_nonincreasing']==1
    assert r['callback_sequence_nonincreasing']==1
    assert r['callback_to_read_over_two_frame_periods']==1
    assert r['host_wall_minus_monotonic_step_max_ms']==pytest.approx(100.,abs=.001)


def test_sdk_callback_records_microsecond_times_and_keeps_latest_frame():
    c=TimingCamera(warmup=0,enable_depth=False)
    c.recording.set()
    for sequence in (5,6):
        color=NS(get_index=lambda:sequence,get_timestamp_us=lambda:sequence*33333,
                 get_system_timestamp_us=lambda:1_790_000_000_000_000+sequence*33333)
        c._on_frame(NS(get_color_frame=lambda:color))
    assert len(c.callback_rows)==2 and c._pending.maxlen==1
    assert c.callback_rows[-1]['sequence']==6
    assert c.callback_rows[-1]['global_us']==0
    assert c.by_sequence[5]['sensor_us']==5*33333


def test_joint_worker_requires_only_read_methods_and_preserves_feedback_samples():
    import threading
    stop=threading.Event();rows=[]
    class Arm:
        def read_joints(self): return np.zeros(7),123.
        def read(self):
            stop.set()
            return np.zeros(6),np.eye(4),123.
    record_joints(Arm(),stop,rows,100)
    assert len(rows)==1 and rows[0]['q_rad']==[0.]*7
    # This fake exposes no enable, motion, mode-change or gripper operations.


def test_recording_saves_paired_video_indices_and_closes_read_only_resources(tmp_path,monkeypatch):
    import json
    import time
    from tools import diagnose_handeye_sync as app
    closed=[];video=[]
    class Camera(TimingCamera):
        def __enter__(self):
            self.serial='synthetic';self.sequence=0
            self.pipeline=NS(get_device=lambda:NS(is_global_timestamp_supported=lambda:False))
            return self
        def __exit__(self,*args): closed.append('camera')
        def read(self):
            time.sleep(.004);self.sequence+=1;i=self.sequence
            color=NS(get_index=lambda:i,get_timestamp_us=lambda:i*33333,
                     get_system_timestamp_us=lambda:int(time.time()*1e6))
            self._on_frame(NS(get_color_frame=lambda:color))
            return NS(sequence=i,bgr=np.zeros((12,16,3),np.uint8))
    class Arm:
        def __init__(self,*args): self.arm=NS(get_arm_status=lambda:None)
        def __enter__(self): return self
        def __exit__(self,*args): closed.append('arm')
        def read_joints(self): return np.zeros(7),time.time()
        def read(self): return np.zeros(6),np.eye(4),time.time()
    args=app.parse_args(['--camera-backend','orbbec','--camera-width','16','--camera-height','12',
                         '--arm-state','enabled-hold','--duration','2','--no-preview','--save-dir',str(tmp_path)])
    args.duration=.045
    monkeypatch.setattr(app,'parse_args',lambda _:args)
    monkeypatch.setattr(app,'TimingCamera',Camera)
    monkeypatch.setattr(app,'NeroReadOnlyFeedback',Arm)
    monkeypatch.setattr(app.cv2,'VideoWriter',lambda *a:NS(isOpened=lambda:True,
                       write=lambda im:video.append(im.copy()),release=lambda:closed.append('video')))
    report=app.main()
    out=next(tmp_path.iterdir())
    frames=[json.loads(s) for s in (out/'frames.jsonl').read_text().splitlines()]
    assert len(frames)==len(video)>=2
    assert [s['video_frame_index'] for s in frames]==list(range(len(video)))
    assert report['status']=='completed' and report['robot_commands_sent'] is False
    assert report['operator_arm_state']=='enabled-hold'
    assert report['camera_to_motion_time_offset_s'] is None
    assert set(closed)=={'camera','arm','video'}
