import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

from tests.factories import ImageStub


@pytest.mark.unit
def test_interpolate_drift_minimal():
    from limosat.processing import interpolate_drift
    from limosat.keypoints import Keypoints

    img = ImageStub()
    # 5 trajectories in poly
    points_poly = gpd.GeoDataFrame(
        {
            'trajectory_id': [0, 1, 2, 3, 4],
        },
        geometry=[Point(10 + i * 2, 10) for i in range(5)],
        crs='EPSG:3413',
    )
    # matched subset 0,1
    points_fg1 = points_poly.iloc[:2].copy()
    points_fg1['image_id'] = 0
    # destination points for matched subset
    points_fg2 = points_fg1.copy()
    points_fg2['geometry'] = points_fg2['geometry'].translate(xoff=1)

    out = interpolate_drift(points_poly, points_fg1, points_fg2, img, 48, 4)
    # Should include original matched + at least one interpolated (from unmatched subset)
    assert set(out['trajectory_id']).issuperset(set(points_fg2['trajectory_id']))
    assert 'interpolated' in out.columns
    assert (out['geometry'].x.between(0, img._w - 1)).all()
    assert (out['geometry'].y.between(0, img._h - 1)).all()
