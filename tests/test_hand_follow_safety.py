"""Measured approach/stop regressions; synthetic feedback, no camera or CAN."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from models.nero_model import NeroPinocchioModel
from servo.execution_safety import ExecutionSafetyMonitor
from servo.hand_follow_safety import HandFollowSafetyMonitor


Q = np.array([.1, .2, -.1, 1.2, .05, -.1, .2])


def monitor():
    model = NS(tcp_reach_bound=lambda: (np.zeros(3), 1.),
               fk=lambda q: (np.asarray(q)[:3]-Q[:3], np.eye(3)))
    return HandFollowSafetyMonitor(model, position_leash=.015, joint_leash=.03)


def sample(q, hand_x=.4, *, image_error=0.):
    tcp = np.asarray(q)[:3]-Q[:3]
    hand = np.array([hand_x, 0., 0.])
    error = np.array([image_error, 0., hand_x-tcp[0]-.05])
    A = np.zeros((3, 7)); A[2, 0] = -1.
    return dict(center_control=True, target_lost=False, target_rejected=False,
        marker_center=hand, target_tcp=hand-[.05, 0, 0], error=error,
        feature_jacobian=A, camera_center=dict(pixel_error=[0., 0.],
            distance_error_mm=error[2]*1000, solver_status='ok'))


def evaluate(m, q, now, *, hand_x=.4, reference_tcp=None, reference_q=None):
    tcp = q[:3]-Q[:3]
    return m.evaluate(q, sample(q, hand_x), now,
        reference_tcp=tcp if reference_tcp is None else reference_tcp,
        reference_q=q if reference_q is None else reference_q)


def test_far_hand_continues_past_old_one_second_timeout_with_actual_progress():
    m, old = monitor(), ExecutionSafetyMonitor()
    for tick in range(91):
        t = tick/30.
        q = Q.copy(); q[0] += .02*t
        decision = evaluate(m, q, t)
        assert not decision.hold, decision
        legacy = old.evaluate(q, sample(q), t)
        if t >= 1.:
            assert legacy.status == 'large_error_hold'
    assert sample(q)['error'][2] > .2  # Still distant, but approaching normally.


def test_stall_stops_and_does_not_retry_without_a_meaningfully_new_hand_location():
    m = monitor()
    for tick in range(90):
        assert not evaluate(m, Q, tick/30.).hold
    assert evaluate(m, Q, 3.).status == 'hand_approach_stalled_hold'
    for t in [3.1, 5., 30., 300.]:
        assert evaluate(m, Q, t).hold
    # Loss and reacquisition of the SAME target must not restart the attempt.
    assert m.evaluate(Q, {'target_lost': True}, 301.).hold
    assert evaluate(m, Q, 302.).status == 'hand_approach_stalled_hold'
    assert evaluate(m, Q, 302.1, hand_x=.39).hold
    assert not evaluate(m, Q, 302.2, hand_x=.37).hold


def test_hand_motion_and_host_reference_motion_cannot_fake_robot_progress():
    m = monitor()
    for tick in range(91):
        # The hand approaches while actual robot feedback never moves. The
        # installed reference advances only within its existing 15mm leash.
        t = tick/30.
        decision = evaluate(m, Q, t, hand_x=.5-.01*t,
                            reference_tcp=[min(.015, t*.005), 0., 0.])
    assert decision.status == 'hand_approach_stalled_hold'


def test_motion_away_from_target_does_not_count_as_progress():
    m = monitor()
    for tick in range(91):
        q = Q.copy(); q[0] -= .005*tick/30.
        decision = evaluate(m, q, tick/30.)
    assert decision.status == 'hand_approach_stalled_hold'


def test_satisfied_task_can_hold_still_without_being_classed_as_stalled():
    m = monitor()
    for tick in range(120):
        assert not evaluate(m, Q, tick/30., hand_x=.055).hold


def test_small_centering_correction_uses_a_fraction_of_remaining_error_for_progress():
    m = monitor()
    for tick in range(91):
        q = Q.copy(); q[1] -= .00025*tick/30.
        out = sample(q, .05)
        out['error'] = np.array([.0015+q[1]-Q[1], 0., 0.])
        out['feature_jacobian'][:] = 0
        out['feature_jacobian'][0, 1] = 1.
        out['camera_center']['pixel_error'] = [out['error'][0]*20000, 0.]
        decision = m.evaluate(q, out, tick/30., reference_tcp=q[:3]-Q[:3], reference_q=q)
        assert not decision.hold, decision


def test_reference_lag_timeout_requires_continuous_fresh_samples():
    m = monitor()
    for t in [0., .1, .2, .3, .4, .5, 2., 2.1]:
        assert not evaluate(m, Q, t, reference_tcp=[.03, 0, 0]).hold


@pytest.mark.parametrize('bad_reference', ['tcp', 'joint'])
@pytest.mark.parametrize('hand_x', [.05, .1])
def test_installed_reference_tracking_error_stops_even_with_near_hand(bad_reference, hand_x):
    m = monitor()
    kwargs = ({'reference_tcp': [.03, 0., 0.]} if bad_reference == 'tcp'
              else {'reference_q': Q+[.05, 0, 0, 0, 0, 0, 0]})
    assert not evaluate(m, Q, 0., hand_x=hand_x, **kwargs).hold
    # Fresh samples, not a one-second camera dropout.
    for tick in range(1, 31):
        decision = evaluate(m, Q, tick/30., hand_x=hand_x, **kwargs)
    assert decision.status == 'reference_tracking_hold'


def test_outside_reach_bound_approaches_while_progressing_then_holds_after_stalling():
    m = monitor()
    # The hand is fixed beyond the model's 1m outer sphere. Actual feedback
    # approaches for six seconds, then stops without ever reaching the hand.
    for tick in range(300):
        t = tick/30.
        q = Q.copy(); q[0] += .02*min(t, 6.)
        out = sample(q, hand_x=1.2)
        decision = m.evaluate(q, out, t, reference_tcp=q[:3]-Q[:3], reference_q=q)
        assert out['camera_center']['execution_safety']['outside_reach_bound']
        if t < 9.:
            assert not decision.hold, decision
        else:
            assert decision.status == 'hand_approach_stalled_hold'
    assert not evaluate(m, q, 10., hand_x=1.17).hold


def test_nonfinite_hand_still_holds_without_a_reach_gate():
    assert evaluate(monitor(), Q, 0., hand_x=float('nan')).status == 'hand_geometry_hold'


@pytest.mark.parametrize('kind', ['target_lost', 'target_rejected', 'near_limit', 'hard_limit'])
def test_existing_target_and_joint_guards_have_priority(kind):
    m = monitor()
    q, out = Q.copy(), sample(Q)
    if kind in ('target_lost', 'target_rejected'):
        out[kind] = True
    else:
        q[0] = m.joint_limits[0, 1] + (.01 if kind == 'hard_limit' else -.01)
    assert m.evaluate(q, out, 0., reference_tcp=[0, 0, 0], reference_q=Q).hold


def test_missing_and_nonfinite_reference_or_feature_context_fails_closed():
    m = monitor()
    assert m.evaluate(Q, sample(Q), 0.).status == 'hand_geometry_hold'
    out = sample(Q); out['feature_jacobian'][0, 0] = float('nan')
    assert m.evaluate(Q, out, 0., reference_tcp=[0, 0, 0], reference_q=Q).hold


def test_fixed_pose_and_ordinary_pbvs_keep_the_original_large_error_check():
    m = monitor()
    out = dict(error=[.3, 0, 0], center_control=True)  # No image task, e.g. retreat.
    assert not m.evaluate(Q, out, 0.).hold
    assert m.evaluate(Q, out, 1.1).status == 'large_error_hold'


def test_reach_bound_contains_fk_samples_and_respects_tool_offset():
    random = np.random.default_rng(73)
    for offset in [None, [.1, .03, -.02]]:
        model = NeroPinocchioModel(tcp_offset=offset)
        center, radius = model.tcp_reach_bound()
        for q in random.uniform(model.JOINT_LIMITS[:, 0], model.JOINT_LIMITS[:, 1], (150, 7)):
            assert np.linalg.norm(model.fk(q)[0]-center) <= radius+1e-12
    model = NeroPinocchioModel()
    center, radius = model.tcp_reach_bound()
    np.testing.assert_allclose(center, [0, 0, .138])
    assert radius == pytest.approx(.7800808073266925)
