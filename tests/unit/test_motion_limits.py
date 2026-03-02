import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from tests.factories import MatcherStub, make_keypoints


def load_real_module(module_name):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    for name in list(sys.modules):
        if name == "limosat" or name.startswith("limosat."):
            del sys.modules[name]
    return importlib.import_module(module_name)


def make_points():
    Keypoints = load_real_module("limosat.keypoints").Keypoints

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
    Matcher = load_real_module("limosat.matcher").Matcher

    matcher = Matcher(max_speed_m_per_day=50000, use_model_estimation=False)
    previous_time = pd.Timestamp("2020-03-01 00:00:00")

    assert matcher.motion_distance_limit(previous_time, previous_time + pd.Timedelta(hours=12)) == pytest.approx(25000.0)
    assert matcher.motion_distance_limit(previous_time, previous_time + pd.Timedelta(days=3)) == pytest.approx(150000.0)


@pytest.mark.unit
def test_matcher_motion_distance_limit_clamps_backward_time_gap():
    Matcher = load_real_module("limosat.matcher").Matcher

    matcher = Matcher(max_speed_m_per_day=50000, spatial_distance_max=100000, use_model_estimation=False)

    assert matcher.motion_distance_limit(
        pd.Timestamp("2020-03-13 00:00:00"),
        pd.Timestamp("2020-03-03 00:00:00"),
    ) == pytest.approx(0.0)


@pytest.mark.unit
def test_matcher_filter_respects_motion_distance_limit():
    Matcher = load_real_module("limosat.matcher").Matcher

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
def test_matcher_filter_zero_motion_limit_keeps_only_exact_matches():
    Matcher = load_real_module("limosat.matcher").Matcher

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
    Matcher = load_real_module("limosat.matcher").Matcher

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
    Matcher = load_real_module("limosat.matcher").Matcher

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
    ImageProcessor = load_real_module("limosat.image_processor").ImageProcessor

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
    ImageProcessor = load_real_module("limosat.image_processor").ImageProcessor

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
    ImageProcessor = load_real_module("limosat.image_processor").ImageProcessor

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
    ImageProcessor = load_real_module("limosat.image_processor").ImageProcessor

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
