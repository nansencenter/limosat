import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import xarray as xr
import importlib
import sys
from pathlib import Path
from shapely.geometry import Point

from tests.factories import ImageStub, make_templates


def load_real_module(module_name):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    for name in list(sys.modules):
        if name == "limosat" or name.startswith("limosat."):
            del sys.modules[name]
    return importlib.import_module(module_name)


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


@pytest.mark.unit
def test_pattern_matching_rejects_template_with_too_much_invalid_data():
    pattern_matching = load_real_module("limosat.processing").pattern_matching

    img = ImageStub(width=64, height=64)
    points = gpd.GeoDataFrame(
        {'trajectory_id': [1], 'interpolated': [0]},
        geometry=[Point(32, 32)],
        crs='EPSG:3413',
    )
    points_fg1 = points.copy()
    points_fg1['angle'] = 0
    template = np.zeros((1, 5, 5), dtype=np.uint8)
    template[0, :2, :] = np.arange(1, 11, dtype=np.uint8).reshape(2, 5)
    templates = xr.DataArray(
        template,
        dims=('trajectory_id', 'height', 'width'),
        coords={'trajectory_id': [1], 'height': range(5), 'width': range(5)},
    )

    _, _, corr = pattern_matching(
        points,
        img,
        templates,
        points_fg1,
        hs=2,
        min_valid_fraction=0.8,
    )

    assert corr.tolist() == [-1.0]


@pytest.mark.unit
def test_pattern_matching_rejects_invalid_destination_patches():
    pattern_matching = load_real_module("limosat.processing").pattern_matching

    class MaskedImageStub(ImageStub):
        def __init__(self):
            super().__init__(width=64, height=64)
            self._band = np.indices((64, 64)).sum(axis=0).astype(np.uint8) + 1
            self._mask = np.full((64, 64), 2, dtype=np.uint8)

        def __getitem__(self, band):
            return self._mask if band == 2 else self._band

    img = MaskedImageStub()
    points = gpd.GeoDataFrame(
        {'trajectory_id': [1], 'interpolated': [0]},
        geometry=[Point(32, 32)],
        crs='EPSG:3413',
    )
    points_fg1 = points.copy()
    points_fg1['angle'] = 0
    template = np.arange(1, 26, dtype=np.uint8).reshape(1, 5, 5)
    templates = xr.DataArray(
        template,
        dims=('trajectory_id', 'height', 'width'),
        coords={'trajectory_id': [1], 'height': range(5), 'width': range(5)},
    )

    _, _, corr = pattern_matching(
        points,
        img,
        templates,
        points_fg1,
        hs=2,
        min_valid_fraction=0.8,
    )

    assert corr.tolist() == [-1.0]
