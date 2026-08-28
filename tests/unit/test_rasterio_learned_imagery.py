from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint

from limosat.learned_drift.imagery import (
    image_object,
    north_up_patch,
    projected_footprint,
    read_scene,
)


def _gcp_scene(path: Path) -> np.ndarray:
    rows, columns = np.mgrid[:101, :101]
    image = (rows + columns).astype(np.uint8)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[50, 50] = 2
    gcps = [
        GroundControlPoint(row=row, col=column, x=1_000 + 80 * column, y=2_000 - 80 * row)
        for row in (0, 50, 101)
        for column in (0, 50, 101)
    ]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=image.shape[1],
        height=image.shape[0],
        count=2,
        dtype="uint8",
        gcps=gcps,
        crs="EPSG:3413",
    ) as dataset:
        dataset.write(image, 1)
        dataset.write(mask, 2)
    return image


def test_gcp_tps_patch_and_footprint_use_float64_epsg3413(tmp_path: Path) -> None:
    path = tmp_path / "scene.tiff"
    image = _gcp_scene(path)

    scene = image_object(str(path), 3413)
    columns, rows = scene.analysis_to_pixels(
        np.asarray([5_000.0]), np.asarray([-2_000.0])
    )
    np.testing.assert_allclose(columns, [50.0], atol=1.0e-7)
    np.testing.assert_allclose(rows, [50.0], atol=1.0e-7)

    patch, valid = north_up_patch(str(path), (5_000.0, -2_000.0), 9, 80.0, 3413, 2)
    expected = image[46:55, 46:55].copy()
    expected[4, 4] = 0
    np.testing.assert_array_equal(patch, expected)
    assert valid.dtype == np.bool_
    assert valid.sum() == 80

    footprint = projected_footprint(str(path), 3413)
    np.testing.assert_allclose(footprint.bounds, (1_000, -6_080, 9_080, 2_000), atol=20.0)


def test_read_scene_scales_float_input_and_keeps_missing_mask(tmp_path: Path) -> None:
    path = tmp_path / "float.tiff"
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:3413",
        transform=rasterio.transform.from_origin(1_000, 2_000, 1, 1),
    ) as dataset:
        dataset.write(values, 1)
    image, mask = read_scene(str(path))
    assert image.dtype == np.uint8
    assert mask is None
    assert image.min() == 0
    assert image.max() == 255


def test_learned_imagery_import_does_not_load_nansat_or_osgeo() -> None:
    code = (
        "import sys; import limosat.learned_drift.imagery; "
        "assert 'nansat' not in sys.modules; assert 'osgeo' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
