import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.compare_mgpc_reconstruction import normalize, denormalize, transform, fps, render, project_rgb


def test_normalization_and_camera_roundtrip_preserve_metric_pose():
    rng = np.random.default_rng(8)
    points = rng.uniform([-.04, -.08, .30], [.05, .08, .38], size=(2048, 3))
    normalized, metadata = normalize(points)
    np.testing.assert_allclose(normalized.mean(0), 0, atol=1e-7)
    assert np.linalg.norm(np.ptp(normalized,axis=0)) == pytest.approx(1, abs=1e-7)
    recovered = denormalize(normalized, metadata)
    np.testing.assert_allclose(recovered, points, atol=1e-8)
    matrix = np.array([[0,-1,0,-.4], [1,0,0,.1], [0,0,1,.2], [0,0,0,1.]])
    np.testing.assert_allclose(transform(transform(recovered,matrix),np.linalg.inv(matrix)),points,atol=1e-8)


def test_fps_retains_real_points_and_handles_sparse_input():
    points = np.array([[0,0,1], [1,0,1], [0,1,1], [1,1,1]],float)
    result, ids = fps(points, 3)
    np.testing.assert_array_equal(result,points[ids])
    assert len(set(ids)) == 3
    expanded, ids = fps(points, 8)
    np.testing.assert_array_equal(expanded,points[ids])
    assert len(expanded) == 8
    with pytest.raises(ValueError,match="Degenerate"):
        normalize(np.ones((10,3)))


def test_rgb_projection_undoes_camera_and_registration_and_rejects_hidden_points():
    camera = np.array([[0,0,1], [.01,0,1], [0,.01,1], [0,0,1.02], [5,0,1], [0,0,-1.]])
    pose = np.array([[0,-1,0,-.4], [1,0,0,.1], [0,0,1,.2], [0,0,0,1.]])
    correction = np.eye(4)
    correction[:3,3] = [.008,-.003,.005]
    registered = transform(transform(camera,pose),correction)
    original = registered.copy()
    rgb = np.zeros((3,3,3),np.uint8)
    rgb[1,1], rgb[1,2] = [255,0,0], [0,0,255]
    mask = np.ones((3,3),bool)
    mask[2,1] = False
    view = dict(rgb=rgb, mask=mask, depth=np.ones((3,3)),
                intrinsics=dict(fx=100,fy=100,cx=1,cy=1),
                T_base_camera=pose,target_transform=correction)
    colors, source, residual = project_rgb(registered,[view])
    np.testing.assert_array_equal(colors[:2],[[1,0,0],[0,0,1]])
    np.testing.assert_allclose(colors[2:],.55)
    np.testing.assert_array_equal(source,[0,0,-1,-1,-1,-1])
    np.testing.assert_allclose(residual[:2],0,atol=1e-12)
    np.testing.assert_array_equal(registered,original)


def test_rgb_projection_prefers_the_view_with_smaller_depth_residual():
    common = dict(mask=np.ones((1,1),bool),intrinsics=dict(fx=1,fy=1,cx=0,cy=0),
                  T_base_camera=np.eye(4),target_transform=np.eye(4))
    front = dict(common,rgb=np.array([[[255,0,0]]]),depth=np.array([[1.004]]))
    side = dict(common,rgb=np.array([[[0,255,0]]]),depth=np.array([[1.001]]))
    colors,source,residual=project_rgb(np.array([[0,0,1.]]),[front,side])
    np.testing.assert_array_equal(colors,[[0,1,0]])
    assert source[0] == 1
    assert residual[0] == pytest.approx(.001)


def test_viewer_cannot_label_observations_as_available_mgpc_predictions(tmp_path):
    points = np.array([[-.4,0,.05],[-.38,0,.08]])
    for name in ('single','multi'):
        (tmp_path/name).mkdir()
        np.save(tmp_path/name/'observed_base.npy',points)
    np.savez(tmp_path/'background.npz',single=points,single_rgb=np.ones_like(points)*.5,
             multi=points,multi_rgb=np.ones_like(points)*.5)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(target_center_base=[-.39,0,.06],
        cases={name:dict(prediction_available=False) for name in ('single','multi')})))
    render(SimpleNamespace(output_dir=tmp_path))
    page=(tmp_path/'comparison.html').read_text()
    assert '预测面板尚未完成' in page
    assert json.loads((tmp_path/'geometry_summary.json').read_text()) == []
    assert not (tmp_path/'C_single_mgpc_scene_base.ply').exists()
    assert not (tmp_path/'D_multi_mgpc_scene_base.ply').exists()
