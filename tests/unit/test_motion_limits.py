import cv2
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import limosat.matcher as matcher_module
from limosat.image_processor import ImageProcessor
from limosat.keypoints import Keypoints
from limosat.matcher import Matcher
from tests.factories import MatcherStub, make_keypoints


def make_points():
    return Keypoints(
        {
            "trajectory_id": [0],
            "image_id": [1],
            "is_last": [1],
            "stopped": [False],
            "converged_to": [None],
            "time": [pd.Timestamp("2020-03-01 00:00:00")],
        },
        geometry=[Point(0, 0)],
        crs="EPSG:3413",
    )


@pytest.mark.unit
def test_matcher_motion_distance_limit_scales_with_time_gap():
    matcher = Matcher(max_speed_m_per_day=50000, use_model_estimation=False)
    previous_time = pd.Timestamp("2020-03-01 00:00:00")

    assert matcher.motion_distance_limit(previous_time, previous_time + pd.Timedelta(hours=12)) == pytest.approx(25000.0)
    assert matcher.motion_distance_limit(previous_time, previous_time + pd.Timedelta(days=3)) == pytest.approx(150000.0)


@pytest.mark.unit
def test_matcher_motion_distance_limit_clamps_backward_time_gap():
    matcher = Matcher(max_speed_m_per_day=50000, spatial_distance_max=100000, use_model_estimation=False)

    assert matcher.motion_distance_limit(
        pd.Timestamp("2020-03-13 00:00:00"),
        pd.Timestamp("2020-03-03 00:00:00"),
    ) == pytest.approx(0.0)


@pytest.mark.unit
def test_matcher_filter_respects_motion_distance_limit():
    matcher = Matcher(descriptor_distance_max=100, use_model_estimation=False)
    matches = [
        cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=10),
        cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=10),
    ]
    pos0 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    pos1 = np.array([[30.0, 0.0], [60.0, 0.0]], dtype=float)

    idx0, idx1, _ = matcher.filter(matches, pos0, pos1, max_distance_m=50.0)

    assert idx0.tolist() == [0]
    assert idx1.tolist() == [0]


@pytest.mark.unit
def test_homography_fit_uses_kilometres_and_returns_metre_residuals(
    monkeypatch,
):
    source = np.array(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
    ) + np.array([3_000_000.0, -1_000_000.0])
    target = source + np.array([3.0, -2.0])
    matches = [cv2.DMatch(i, i, 0, 10.0) for i in range(len(source))]
    captured = {}

    def fake_find_homography(source_scaled, target_scaled, method, threshold):
        captured["source"] = source_scaled
        captured["target"] = target_scaled
        captured["threshold"] = threshold
        return (
            np.array([[1.0, 0.0, 0.003], [0.0, 1.0, -0.002], [0.0, 0.0, 1.0]]),
            np.ones((len(source_scaled), 1), dtype=np.uint8),
        )

    monkeypatch.setattr(matcher_module.cv2, "findHomography", fake_find_homography)
    matcher = Matcher(
        descriptor_distance_max=100,
        model_threshold=15_000.0,
        min_homography_inliers=3,
    )

    idx0, idx1, residuals = matcher.filter(matches, source, target)

    np.testing.assert_allclose(captured["source"], source / 1_000.0)
    np.testing.assert_allclose(captured["target"], target / 1_000.0)
    assert captured["threshold"] == 15.0
    assert idx0.tolist() == [0, 1, 2, 3]
    assert idx1.tolist() == [0, 1, 2, 3]
    assert np.max(residuals) < 1e-9


@pytest.mark.unit
def test_matcher_filter_zero_motion_limit_keeps_only_exact_matches():
    matcher = Matcher(descriptor_distance_max=100, use_model_estimation=False)
    matches = [
        cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=10),
        cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=10),
    ]
    pos0 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    pos1 = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=float)

    idx0, idx1, _ = matcher.filter(matches, pos0, pos1, max_distance_m=0.0)

    assert idx0.tolist() == [0]
    assert idx1.tolist() == [0]


@pytest.mark.unit
def test_matcher_rejects_mixed_previous_times_within_group():
    matcher = Matcher(descriptor_distance_max=100, use_model_estimation=False, max_speed_m_per_day=50000)
    points_poly = make_keypoints(2, image_id=1, t0="2020-03-01 00:00:00", step_s=3600)
    points_grid = make_keypoints(2, image_id=2, t0="2020-03-02 00:00:00", step_s=0)
    matches = [
        cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=10),
        cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=10),
    ]

    matcher.match_with_crosscheck = lambda x0, x1: matches
    matcher.match_with_lowe_ratio = lambda matches_bf_initial, x0, x1, pos0, pos1: matches_bf_initial

    with pytest.raises(ValueError, match="previous image_id 1"):
        matcher.match_with_grid(points_poly, points_grid)


@pytest.mark.unit
def test_matcher_rejects_mixed_current_frame_times():
    matcher = Matcher(descriptor_distance_max=100, use_model_estimation=False, max_speed_m_per_day=50000)
    points_poly = make_keypoints(2, image_id=1, t0="2020-03-01 00:00:00", step_s=0)
    points_grid = make_keypoints(2, image_id=2, t0="2020-03-02 00:00:00", step_s=3600)
    matches = [
        cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=10),
        cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=10),
    ]

    matcher.match_with_crosscheck = lambda x0, x1: matches
    matcher.match_with_lowe_ratio = lambda matches_bf_initial, x0, x1, pos0, pos1: matches_bf_initial

    with pytest.raises(ValueError, match="Current frame points"):
        matcher.match_with_grid(points_poly, points_grid)


@pytest.mark.unit
def test_image_processor_max_speed_sets_all_motion_limits():
    matcher = MatcherStub()
    proc = ImageProcessor(
        points=make_points(),
        model=None,
        matcher=matcher,
        persist_updates=False,
        temporal_window=3,
        max_speed_m_per_day=50000,
    )

    assert proc.candidate_search_max_daily_drift_m == pytest.approx(50000.0)
    assert proc.max_valid_speed_m_per_day == pytest.approx(50000.0)
    assert proc.matcher.max_speed_m_per_day == pytest.approx(50000.0)
    assert proc._candidate_buffer_distance_m() == pytest.approx(150000.0)


@pytest.mark.unit
def test_image_processor_reuses_matcher_max_speed_when_processor_value_is_unset():
    matcher = MatcherStub()
    matcher.max_speed_m_per_day = 40000
    proc = ImageProcessor(
        points=make_points(),
        model=None,
        matcher=matcher,
        persist_updates=False,
        temporal_window=3,
    )

    assert proc.max_speed_m_per_day == pytest.approx(40000.0)
    assert proc.candidate_search_max_daily_drift_m == pytest.approx(40000.0)
    assert proc.max_valid_speed_m_per_day == pytest.approx(40000.0)
    assert proc.matcher.max_speed_m_per_day == pytest.approx(40000.0)


@pytest.mark.unit
def test_image_processor_rejects_conflicting_max_speed_values():
    matcher = MatcherStub()
    matcher.max_speed_m_per_day = 40000

    with pytest.raises(ValueError, match="must not be set to different values"):
        ImageProcessor(
            points=make_points(),
            model=None,
            matcher=matcher,
            persist_updates=False,
            temporal_window=3,
            max_speed_m_per_day=50000,
        )


@pytest.mark.unit
def test_image_processor_buffer_uses_spatial_cap_without_max_speed():
    matcher = MatcherStub()
    matcher.spatial_distance_max = 15000
    proc = ImageProcessor(
        points=make_points(),
        model=None,
        matcher=matcher,
        persist_updates=False,
        temporal_window=3,
        candidate_search_max_daily_drift_m=10000,
        max_valid_speed_m_per_day=50000,
    )

    assert proc._candidate_buffer_distance_m() == pytest.approx(15000.0)
