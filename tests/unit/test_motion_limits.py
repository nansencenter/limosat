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
def test_matcher_audit_records_candidate_fates_without_changing_result():
    Matcher = load_real_module("limosat.matcher").Matcher

    class Sink:
        def __init__(self):
            self.records = []

        def emit(self, stream, records):
            self.records.extend((stream, record) for record in records)

    sink = Sink()
    matcher = Matcher(
        descriptor_distance_max=100,
        use_model_estimation=False,
        audit_sink=sink,
    )
    matches = [
        cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=10),
        cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=110),
        cv2.DMatch(_queryIdx=2, _trainIdx=2, _distance=10),
    ]
    pos0 = np.zeros((3, 2), dtype=float)
    pos1 = np.array([[30.0, 0.0], [1.0, 0.0], [60.0, 0.0]], dtype=float)

    idx0, idx1, _ = matcher.filter(
        matches,
        pos0,
        pos1,
        max_distance_m=50.0,
        audit_context={
            "source_image_id": 1,
            "target_image_id": 2,
            "source_trajectory_ids": np.array([10, 11, 12]),
            "candidate_origins": {0: "crosscheck", 1: "crosscheck", 2: "lowe_ratio"},
        },
    )

    assert idx0.tolist() == [0]
    assert idx1.tolist() == [0]
    rows = [record for stream, record in sink.records if stream == "matcher_candidates"]
    assert [row["rejection_reason"] for row in rows] == [
        "accepted",
        "descriptor_distance",
        "motion_distance",
    ]
    assert rows[0]["candidate_id"] == "2:1:10:0:0"


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
def test_configured_affine_estimator_recovers_affine_inliers():
    Matcher = load_real_module("limosat.matcher").Matcher

    source = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 0.0],
            [0.0, 20.0],
            [20.0, 20.0],
            [30.0, 10.0],
        ]
    )
    affine = np.array([[1.02, 0.01], [-0.02, 0.98]])
    target = source @ affine.T + np.array([4.0, -3.0])
    matches = [cv2.DMatch(i, i, 0, 10.0) for i in range(len(source))]
    matcher = Matcher(
        descriptor_distance_max=100,
        model_threshold=0.5,
        min_homography_inliers=3,
        estimation_method="USAC_MAGSAC",
        model_estimator="configured_affine",
    )

    idx0, idx1, residuals = matcher.filter(matches, source, target)

    assert idx0.tolist() == list(range(len(source)))
    assert idx1.tolist() == list(range(len(source)))
    assert np.max(residuals) < 1e-6


@pytest.mark.unit
def test_homography_affine_union_retains_inliers_from_both_models(monkeypatch):
    matcher_module = load_real_module("limosat.matcher")
    Matcher = matcher_module.Matcher

    source = np.array(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [20.0, 0.0]]
    )
    target = source + np.array([3.0, -2.0])
    matches = [cv2.DMatch(i, i, 0, 10.0) for i in range(len(source))]
    monkeypatch.setattr(
        matcher_module.cv2,
        "findHomography",
        lambda *args, **kwargs: (
            np.eye(3),
            np.array([[1], [1], [1], [0], [0]], dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        matcher_module.cv2,
        "estimateAffine2D",
        lambda *args, **kwargs: (
            np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]]),
            np.array([[0], [0], [1], [1], [1]], dtype=np.uint8),
        ),
    )
    matcher = Matcher(
        descriptor_distance_max=100,
        min_homography_inliers=3,
        model_estimator="homography_affine_union",
    )

    idx0, idx1, residuals = matcher.filter(matches, source, target)

    assert idx0.tolist() == [0, 1, 2, 3, 4]
    assert idx1.tolist() == [0, 1, 2, 3, 4]
    assert residuals.shape == (5,)


@pytest.mark.unit
def test_configured_homography_scale_preserves_physical_threshold_and_residuals(
    monkeypatch,
):
    matcher_module = load_real_module("limosat.matcher")
    Matcher = matcher_module.Matcher

    source = np.array(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
    )
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
        model_coordinate_scale_m=1_000.0,
    )

    idx0, idx1, residuals = matcher.filter(matches, source, target)

    np.testing.assert_allclose(captured["source"], source / 1_000.0)
    np.testing.assert_allclose(captured["target"], target / 1_000.0)
    assert captured["threshold"] == 15.0
    assert idx0.tolist() == [0, 1, 2, 3]
    assert idx1.tolist() == [0, 1, 2, 3]
    assert np.max(residuals) < 1e-9


@pytest.mark.unit
@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_matcher_rejects_invalid_model_coordinate_scale(scale):
    Matcher = load_real_module("limosat.matcher").Matcher

    with pytest.raises(ValueError, match="model_coordinate_scale_m"):
        Matcher(model_coordinate_scale_m=scale)


@pytest.mark.unit
def test_local_physics_fallback_recovers_sources_after_global_mismatch_or_omission():
    matcher_module = load_real_module("limosat.matcher")
    Matcher = matcher_module.Matcher

    matcher = Matcher(
        norm=cv2.NORM_HAMMING2,
        descriptor_distance_max=120,
        use_model_estimation=False,
        candidate_selection="global_then_local_physics_fallback",
    )
    x0 = np.vstack(
        [np.zeros(32, dtype=np.uint8), np.full(32, 255, dtype=np.uint8)]
    )
    x1 = np.vstack(
        [
            np.zeros(32, dtype=np.uint8),
            np.r_[np.uint8(1), np.zeros(31, dtype=np.uint8)],
            np.r_[np.uint8(254), np.full(31, 255, dtype=np.uint8)],
        ]
    )
    pos0 = np.array([[0.0, 0.0], [100.0, 0.0]])
    pos1 = np.array([[1000.0, 0.0], [1.0, 0.0], [101.0, 0.0]])
    global_matches = [cv2.DMatch(0, 0, 0, 0.0)]

    combined, origins = matcher._add_local_physics_fallback(
        group_matches=global_matches,
        group_query_indices=np.array([0, 1]),
        x0=x0,
        x1=x1,
        pos0=pos0,
        pos1=pos1,
        target_tree=matcher_module.cKDTree(pos1),
        max_distance_m=5.0,
    )

    assert [(match.queryIdx, match.trainIdx) for match in combined] == [
        (0, 0),
        (0, 1),
        (1, 2),
    ]
    assert origins == {
        (0, 1): "local_physics_fallback",
        (1, 2): "local_physics_fallback",
    }
    idx0, idx1, _ = matcher.filter(
        combined, pos0, pos1, max_distance_m=5.0
    )
    assert idx0.tolist() == [0, 1]
    assert idx1.tolist() == [1, 2]


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
