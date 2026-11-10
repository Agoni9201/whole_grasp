"""Counterfactual ranking must not change eligibility or candidate evidence."""
import copy

import pytest

from tools.rerank_grasp_weights import WEIGHTS, coefficients, eligible, rank_rows


def candidate(idx, network, other):
    return dict(idx=idx, raw_score=network, priority=.5,
                components={k: network if k == 'network' else other for k in WEIGHTS})


def test_reweight_changes_preference_without_changing_evidence():
    rows = [candidate(7, .1, .9), candidate(9, 1., .6)]
    original = copy.deepcopy(rows)
    assert rank_rows(rows, .1)[0]['candidate_id'] == 7
    assert rank_rows(rows, .4)[0]['candidate_id'] == 9
    assert rank_rows(rows, .7)[0]['candidate_id'] == 9
    assert rows == original
    for weight in (.1, .4, .7):
        ranking = rank_rows(rows, weight)
        assert {r['candidate_id'] for r in ranking} == {7, 9}
        assert sum(coefficients(weight).values()) == pytest.approx(1.)
        coefs = coefficients(weight)
        assert coefs['bilateral'] / coefs['table'] == pytest.approx(10.)


def test_increasing_network_weight_never_promotes_table_excluded_candidate():
    rows = [candidate(1, 0., .3), candidate(2, 1., 1.)]
    rows[1]['recommended_table_safe'] = False
    assert rank_rows(rows, 1.)[0]['candidate_id'] == 1


def test_missing_failed_or_unchecked_preflight_is_never_eligible():
    passed = dict(workflow_preflight_ok=True, physical_aperture_ok=True,
                  table_guard_ok=True, reachability_ok=True)
    assert eligible(passed)
    for key in passed:
        for value in (False, None):
            assert not eligible(dict(passed, **{key: value}))
        incomplete = passed.copy()
        del incomplete[key]
        assert not eligible(incomplete)
    assert rank_rows([], .4) == []


def test_exact_ties_keep_production_raw_score_then_priority_tiebreak():
    rows = [candidate(1, .5, .5), candidate(2, .5, .5)]
    rows[1]['raw_score'] = .8
    assert rank_rows(rows, .1)[0]['candidate_id'] == 2


@pytest.mark.parametrize('weight', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_experimental_weights_rejected(weight):
    with pytest.raises(ValueError):
        coefficients(weight)
