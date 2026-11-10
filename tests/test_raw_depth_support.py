"""SDK interpolation must not manufacture repeated physical observations."""

import numpy as np
import pytest

from cameras.depth_fusion import supported_depth_median


def test_filled_plateau_with_no_raw_returns_is_rejected():
    sdk = np.full((7, 3, 4), .305, np.float32)
    raw = np.zeros_like(sdk)
    result = supported_depth_median(sdk, raw, 5, .004)
    assert np.count_nonzero(result['depth_m']) == 0
    assert result['report']['rejected_insufficient_raw_pixels'] == 12
    assert result['report']['missing_depth_is_free_space'] is False
    np.testing.assert_array_equal(result['sdk_depth_m'], sdk[0])


def test_raw_window_count_is_used_instead_of_filtered_count():
    sdk = np.full((7, 1, 2), .5, np.float32)
    raw = sdk.copy()
    raw[4:, 0, 0] = 0  # Four real returns; seven identical filled values.
    raw[5:, 0, 1] = 0  # Five real returns; last frame invalid is not enough to reject.
    result = supported_depth_median(sdk, raw, 5, .004)
    np.testing.assert_array_equal(result['depth_m'], [[0, .5]])
    np.testing.assert_array_equal(result['raw_valid_count'], [[4, 5]])


def test_stable_sdk_history_cannot_hide_unstable_raw_depth():
    sdk = np.full((7, 1, 1), .5, np.float32)
    raw = np.array([.45, .55, .46, .54, .47, .53, .5], np.float32).reshape(7, 1, 1)
    result = supported_depth_median(sdk, raw, 5, .004)
    assert result['depth_m'][0, 0] == 0
    assert result['report']['rejected_raw_unstable_pixels'] == 1


def test_filter_smearing_between_real_surfaces_is_rejected():
    sdk = np.full((7, 1, 1), .4, np.float32)
    raw = np.full_like(sdk, .5)
    result = supported_depth_median(sdk, raw, 5, .004)
    assert result['depth_m'][0, 0] == 0
    assert result['report']['rejected_sdk_raw_difference_pixels'] == 1


def test_small_denoising_changes_are_preserved_without_modifying_inputs():
    sdk = np.full((7, 2, 2), .502, np.float32)
    raw = np.full_like(sdk, .5)
    original_sdk, original_raw = sdk.copy(), raw.copy()
    result = supported_depth_median(sdk, raw, 5, .004)
    np.testing.assert_array_equal(result['depth_m'], sdk[0])
    np.testing.assert_array_equal(raw, original_raw)
    np.testing.assert_array_equal(sdk, original_sdk)


def test_mismatched_raw_frames_fail_closed():
    with pytest.raises(ValueError, match='identical'):
        supported_depth_median(np.ones((7, 2, 2)), np.ones((6, 2, 2)), 5, .004)
