"""Regression tests for land filtering during new trajectory seeding."""

import cv2
import geopandas as gpd
import numpy as np

from limosat.keypoint_detector import KeypointDetector


class _Image:
    angle = 0.0

    def __init__(self, image, mask):
        self._bands = {1: image, 2: mask}

    def __getitem__(self, band_id):
        return self._bands[band_id]

    def bands(self):
        return {1: {"name": "s0_HH"}, 2: {"name": "mask"}}


class _Detector:
    def detect(self, _window, _mask):
        return [
            cv2.KeyPoint(1.0, 1.0, 31, response=10.0),
            cv2.KeyPoint(4.0, 4.0, 31, response=5.0),
        ]


def test_land_candidate_is_rejected_before_window_winner_selection():
    image = np.ones((8, 8), dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[1, 1] = 2
    detector = KeypointDetector(_Detector())

    keypoints = detector.detect_new_keypoints(
        points=gpd.GeoDataFrame(geometry=[]),
        img=_Image(image, mask),
        octave=0,
        window_size=8,
        border_size=0,
        response_threshold=0,
        compute_descriptors=False,
    )

    assert len(keypoints) == 1
    assert keypoints[0][0].pt == (4.0, 4.0)


def test_no_keypoint_is_selected_when_all_candidates_are_on_land():
    image = np.ones((8, 8), dtype=np.uint8)
    mask = np.full((8, 8), 2, dtype=np.uint8)
    detector = KeypointDetector(_Detector())

    keypoints = detector.detect_new_keypoints(
        points=gpd.GeoDataFrame(geometry=[]),
        img=_Image(image, mask),
        octave=0,
        window_size=8,
        border_size=0,
        response_threshold=0,
        compute_descriptors=False,
    )

    assert keypoints == []
