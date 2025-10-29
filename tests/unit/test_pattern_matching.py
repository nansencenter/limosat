import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

from tests.factories import ImageStub, make_templates


@pytest.mark.unit
def test_pattern_matching_shapes(monkeypatch):
    # monkeypatch cartopy transforms to pass-through handled by conftest
    from limosat.processing import pattern_matching

    img = ImageStub()
    n = 5
    points = gpd.GeoDataFrame(
        {
            'trajectory_id': np.arange(n, dtype=np.int64),
            'interpolated': np.zeros(n, dtype=int),
        },
        geometry=[Point(5 + i, 5 + i) for i in range(n)],
        crs='EPSG:3413',
    )
    points_fg1 = points.copy()
    points_fg1['angle'] = 0

    templates = make_templates(points['trajectory_id'].values, hs=2)

    xy, cr, corr = pattern_matching(points, img, templates, points_fg1, hs=2)

    assert len(xy) == len(points) == len(corr)
    assert cr.shape == (n, 2)

    # boundary case: point outside still clipped by ImageStub transform
    points.loc[0, 'geometry'] = Point(1000, 1000)
    xy, cr, corr = pattern_matching(points, img, templates, points_fg1, hs=2)
    assert cr.shape == (n, 2)
