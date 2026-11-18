import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point


@pytest.mark.unit
def test_convergence_minimal():
    from limosat.image_processor import ImageProcessor
    from limosat.keypoints import Keypoints

    points = Keypoints(
        {
            'trajectory_id': [10, 11],
            'image_id': [1, 1],
            'is_last': [1, 1],
            'stopped': [False, False],
            'converged_to': [None, None],
            'corr': [0.9, 0.8],
        },
        geometry=[Point(5, 5), Point(6, 5)],
        crs='EPSG:3413',
    )
    proc = ImageProcessor(points=points, model=None, matcher=None, persist_updates=False)

    # Two very close points in same image
    points_matched = points.copy()
    proc._handle_trajectory_convergence(points_matched)

    # One trajectory should be marked converged_to the winner and stopped
    assert points['converged_to'].isna().sum() < len(points)
    assert points['stopped'].any()
