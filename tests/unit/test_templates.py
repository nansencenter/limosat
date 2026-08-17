import importlib
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point


def load_real_templates():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    for name in list(sys.modules):
        if name == "limosat" or name.startswith("limosat."):
            del sys.modules[name]
    return importlib.import_module("limosat.templates").Templates


def fixture():
    rows, cols = np.meshgrid(
        np.arange(16, dtype=np.float32),
        np.arange(16, dtype=np.float32),
        indexing="ij",
    )
    image_data = 2.0 * cols + 3.0 * rows
    points = gpd.GeoDataFrame(
        {"trajectory_id": [1]},
        geometry=[Point(6.25, 7.5)],
        crs="EPSG:3413",
    )

    class Image:
        srs = points.crs

        def __getitem__(self, band):
            return image_data

        def transform_points(self, x, y, **kwargs):
            return np.asarray(x), np.asarray(y)

    return points, Image(), image_data


@pytest.mark.unit
def test_integer_template_sampling_preserves_legacy_slice():
    Templates = load_real_templates()
    points, image, image_data = fixture()

    patches, indices = Templates._extract_from_img(
        points, image, hs=1, sampling="integer"
    )

    assert indices.tolist() == [0]
    np.testing.assert_array_equal(patches[0], image_data[6:9, 5:8])


@pytest.mark.unit
def test_bilinear_template_sampling_uses_fractional_centre():
    Templates = load_real_templates()
    points, image, _ = fixture()

    patches, indices = Templates._extract_from_img(
        points, image, hs=1, sampling="bilinear"
    )

    assert indices.tolist() == [0]
    assert patches[0, 1, 1] == pytest.approx(2.0 * 6.25 + 3.0 * 7.5)


@pytest.mark.unit
def test_template_sampling_rejects_unknown_method():
    Templates = load_real_templates()
    points, image, _ = fixture()

    with pytest.raises(ValueError, match="sampling"):
        Templates._extract_from_img(points, image, hs=1, sampling="cubic")
