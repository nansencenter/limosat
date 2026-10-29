from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from limosat.config import RunConfig, SurfaceConfig
from limosat.surface import (
    LandMask,
    SurfaceGate,
    seed_filter_factory,
    surface_validity_factory,
)


def _land_raster(tmp_path):
    inland = np.zeros((4, 4), dtype=np.float32)
    inland[0, 0] = 5.0  # north-west cell, well inland
    inland[0, 1] = 0.5  # coastal band
    path = tmp_path / "land.npz"
    np.savez(path, inland_km=inland, x0=0.0, y1=4000.0, res=1000.0)
    return path


def test_land_mask_lookup_and_tile_orientation(tmp_path):
    land = LandMask(_land_raster(tmp_path), 1.0)
    assert land.is_land([500.0, 1500.0, 500.0, -10.0], [3500.0, 3500.0, 2500.0, 3500.0]).tolist() == [
        True, False, False, False,
    ]
    tile = land.tile((2000.0, 2000.0), 4, 1000.0)
    # Row 0 is the north edge and column 0 the west edge.
    assert tile[0, 0] and tile.sum() == 1


def test_surface_validity_and_seed_filters(tmp_path):
    gate = SurfaceGate(LandMask(_land_raster(tmp_path), 1.0), None, 15.0)
    image = SimpleNamespace(time_utc=datetime(2020, 1, 1, tzinfo=timezone.utc))
    xy = np.array([[500.0, 3500.0], [2500.0, 2500.0], [3500.0, 500.0]])
    validity = surface_validity_factory(lambda _image: lambda p: np.array([True, True, False]), gate)
    assert validity(image)(xy).tolist() == [False, True, False]
    assert seed_filter_factory(gate)(image)(xy).tolist() == [False, True, True]
    base = object()
    assert surface_validity_factory(base, None) is base
    assert seed_filter_factory(None) is None


def test_point_open_water_keeps_missing_concentration(tmp_path):
    concentration = SimpleNamespace(sample=lambda x, y, day: np.array([5.0, 50.0, np.nan], np.float32))
    gate = SurfaceGate(None, concentration, 15.0)
    excluded = gate.excluded(np.zeros((3, 2)), datetime(2020, 1, 1))
    assert excluded.tolist() == [True, False, False]


def test_surface_config_off_preserves_identity():
    assert "surface" not in RunConfig("run", "catalogue", "database", "output").to_dict()
    with pytest.raises(ValueError):
        SurfaceConfig(point_open_water_threshold_percent=120.0)


def test_land_tile_matches_pointwise_lookup(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "land.npz"
    np.savez(path, inland_km=rng.uniform(0, 3, (20, 30)).astype(np.float32), x0=-5000.0, y1=7000.0, res=1000.0)
    land = LandMask(path, 1.0)
    center, pixels, size = (3210.0, -1870.0), 64, 400.0
    offsets = (np.arange(pixels) - (pixels - 1) / 2.0) * size
    x, y = np.meshgrid(center[0] + offsets, center[1] - offsets)
    assert np.array_equal(land.tile(center, pixels, size), land.is_land(x, y))


def test_asi_reads_netcdf_layout_in_file_crs(tmp_path):
    import rasterio
    import rasterio.shutil
    from rasterio.transform import from_origin

    from limosat.surface import AsiConcentration

    values = np.array([[10.0, np.nan], [90.0, 120.0]], dtype=np.float32)
    (tmp_path / "2020").mkdir()
    tif = tmp_path / "source.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=2, height=2, count=1, dtype="float32",
                       crs="EPSG:3413", transform=from_origin(0.0, 2000.0, 1000.0, 1000.0)) as out:
        out.write(values, 1)
    # The netCDF driver only supports CreateCopy.
    rasterio.shutil.copy(tif, tmp_path / "2020" / "asi-AMSR2-n3125-20200303-v5.4.nc", driver="netCDF")
    sic = AsiConcentration(((str(tmp_path), "n3125"),)).sample(
        [500.0, 1500.0, 500.0, 1500.0, 9000.0], [1500.0, 1500.0, 500.0, 500.0, 500.0], datetime(2020, 3, 3)
    )
    assert np.array_equal(sic, [10.0, np.nan, 90.0, np.nan, np.nan], equal_nan=True)
