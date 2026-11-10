"""Check audit geometry against the actual production collision detector."""
import numpy as np
import pytest

from tools.compare_mgpc_graspnet import baseline_imports, detector_witness, input_indices, matched_ids, plain


@pytest.mark.parametrize('count', [1724, 2874, 5734, 6144])
def test_all_input_preserves_every_point_without_padding_or_reordering(count):
    np.random.seed(0)
    a=input_indices(count,3000,'all')
    np.random.seed(2)
    b=input_indices(count,3000,'all')
    np.testing.assert_array_equal(a,np.arange(count))
    np.testing.assert_array_equal(a,b)


def test_fixed_input_keeps_sparse_geometry_and_only_repeats_existing_points():
    np.random.seed(0)
    ids=input_indices(1724,3000)
    assert len(ids)==3000
    np.testing.assert_array_equal(ids[:1724],np.arange(1724))
    assert len(np.unique(ids))==1724
    assert ids.min()==0 and ids.max()==1723


def test_fixed_dense_input_matches_existing_replay_sampling():
    np.random.seed(2)
    expected=np.random.choice(6144,3000,replace=False)
    np.random.seed(2)
    np.testing.assert_array_equal(input_indices(6144,3000),expected)


def test_witness_points_match_official_collision_ious_and_are_not_closing_contacts():
    baseline_imports()
    from collision_detector import ModelFreeCollisionDetector
    from graspnetAPI import GraspGroup
    # Grasp x approach, y opening, z height. One point in each collider,
    # one legitimate contact in closing volume, one outside the gripper.
    points=np.array([[0,-.045,0],[0,.045,0],[-.035,0,0],[-.06,0,0],
                     [0,0,0],[.2,.2,.2]])
    detector=ModelFreeCollisionDetector(points,voxel_size=.001)
    raw=np.r_[1.,.08,.02,.03,np.eye(3).reshape(-1),[0,0,0],-1.][None]
    group=GraspGroup(raw)
    _,parts=detector.detect(group,approach_dist=.04,return_ious=True)
    witness=detector_witness(detector,group[0],.04)
    assert {k:len(v) for k,v in witness.items()} == dict(left_finger=1,right_finger=1,bottom=1,approach_corridor=1)
    volumes=[.02*.06*.01/.001**3]*2+[.02*.10*.01/.001**3,.02*.10*.04/.001**3]
    for value,volume in zip(parts[1:],volumes):
        assert value[0] == pytest.approx(1/(volume+1e-6))
    for points in witness.values(): assert not np.any(np.all(points==0,axis=1))


def test_nms_row_tracking_survives_float32_to_float64_conversion():
    raw=np.arange(51,dtype=np.float32).reshape(3,17)/7
    assert matched_ids(raw[[2,0]].astype(np.float64),raw) == [2,0]


def test_audit_serialization_is_strict_json_and_excludes_live_plans():
    data=dict(score=np.float32(.5),margin=np.inf,grasp=object(),_workflow_motion_plan=object(),
              nested=[np.bool_(True),np.array([np.nan,1.])])
    assert plain(data) == dict(score=.5,margin=None,nested=[True,[None,1.]])
    assert plain(dict(tcp_preview=dict(grasp=np.arange(6.)))) == dict(tcp_preview=dict(grasp=list(np.arange(6.))))
