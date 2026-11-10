"""A detected RGB hand with missing depth must remain available for diagnosis."""
import json
from types import SimpleNamespace as NS

import numpy as np

from scripts.test_realsense_hand import observation_status, save_snapshot


def test_missing_palm_depth_can_be_saved_with_matching_raw_and_effective_depth(tmp_path):
    frame = NS(bgr=np.zeros((20, 30, 3), dtype=np.uint8), depth_m=np.full((20, 30), .3),
               raw_depth_m=np.zeros((20, 30)), serial='test', K=np.eye(3), D=np.zeros(5), sequence=12)
    observation = NS(palm_pixel=(15, 10), palm_center_camera_m=None, palm_normal_camera=None,
                     palm_tangent_camera=None, handedness='Right', handedness_score=.95,
                     valid_depth_landmarks=0, palm_depth_m=float('nan'), palm_depth_mad_m=float('nan'))
    path = save_snapshot(tmp_path, frame, frame.bgr, observation, np.zeros((20, 30)))
    metadata = json.loads((path / 'hand_pose.json').read_text())
    assert metadata['palm_center_camera_m'] is None and metadata['valid_depth_landmarks'] == 0
    assert metadata['palm_depth_m'] is None and metadata['raw_current_support']
    np.testing.assert_array_equal(np.load(path / 'depth_m.npy'), 0.)
    np.testing.assert_array_equal(np.load(path / 'depth_filtered_m.npy'), .3)
    assert (path / 'depth_raw_m.npy').is_file()


def test_no_rgb_hand_snapshot_preserves_frame_without_inventing_landmarks(tmp_path):
    frame = NS(bgr=np.zeros((20, 30, 3), dtype=np.uint8), depth_m=np.full((20, 30), .3),
               raw_depth_m=np.full((20, 30), .3), serial='orbbec-test', backend='orbbec',
               K=np.eye(3), D=np.zeros(5), sequence=13, hardware_ms=123., received_monotonic=12.)
    path = save_snapshot(tmp_path, frame, frame.bgr, None, capture_settings={'depth_radius': 8})
    metadata = json.loads((path / 'hand_pose.json').read_text())
    assert metadata['reason'] == 'no_hand_2d' and not metadata['metric_position_usable']
    assert not metadata['hand_detected_2d'] and metadata['valid_depth_landmarks'] is None
    assert metadata['palm_pixel'] is None and metadata['landmarks_pixel'] is None
    assert metadata['camera_backend'] == 'orbbec' and metadata['hardware_ms'] == 123.
    assert metadata['capture_settings']['depth_radius'] == 8
    np.testing.assert_array_equal(np.load(path / 'depth_m.npy'), .3)
    assert (path / 'color.png').is_file() and (path / 'depth_raw_m.npy').is_file()


def test_palm_candidate_from_center_and_two_landmarks_is_not_servo_usable(tmp_path):
    # The detector may have three depth patches including the palm center,
    # while the servo still requires three actual landmark patches.
    observation = NS(palm_pixel=(15, 10), palm_center_camera_m=np.array([0., 0., .3]),
                     palm_normal_camera=None, palm_tangent_camera=None,
                     handedness='Right', handedness_score=.95, valid_depth_landmarks=2,
                     palm_depth_m=.3, palm_depth_mad_m=0.)
    assert observation_status(observation) == 'invalid_palm_depth'
    assert observation_status(observation, 2) == 'ok'
    frame = NS(bgr=np.zeros((20, 30, 3), dtype=np.uint8), depth_m=np.full((20, 30), .3),
               raw_depth_m=None, serial='test', K=np.eye(3), D=np.zeros(5), sequence=14)
    path = save_snapshot(tmp_path, frame, frame.bgr, observation)
    metadata = json.loads((path / 'hand_pose.json').read_text())
    assert metadata['hand_detected_2d'] and not metadata['metric_position_usable']
    assert metadata['reason'] == 'invalid_palm_depth' and metadata['required_depth_landmarks'] == 3
    assert metadata['palm_center_camera_m'] == [0., 0., .3]  # Preserve raw candidate as evidence.
