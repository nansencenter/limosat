import json

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("pyarrow")
pytest.importorskip("pyproj")

from limosat import publish, publish_netcdf  # noqa: E402
from test_publish import PERIOD, build as build_parquet  # noqa: E402

pytestmark = pytest.mark.unit

METADATA = {"creators": [{"name": "Chua, Sean Minhui Tashi", "affiliation": "NERSC", "orcid": "0000-0002-8462-3170"}],
            "license": "CC-BY-4.0"}


def make(tmp_path):
    manifest, source, catalogs, parquet_dir = build_parquet(tmp_path)
    out = tmp_path / "nc"
    result = publish_netcdf.build(parquet_dir, out, METADATA)
    return manifest, parquet_dir, out, result


def test_netcdf_is_a_cf_contiguous_ragged_trajectory_file(tmp_path):
    manifest, parquet_dir, out, result = make(tmp_path)
    assert sorted(p.name for p in out.iterdir()) == sorted([f"limosat_{PERIOD}.nc", "manifest.json"])
    assert result["totals"] == manifest["totals"] and result["files"][0]["file"] == f"limosat_{PERIOD}.nc"
    d = netCDF4.Dataset(out / f"limosat_{PERIOD}.nc")
    try:
        assert d.featureType == "trajectory" and "CF-1.8" in d.Conventions
        assert d["trajectory_id"].cf_role == "trajectory_id"
        assert d["rowsize"].sample_dimension == "obs" and d["rowsize"][:].sum() == d.dimensions["obs"].size
        assert d["time"].standard_name == "time" and d["lat"].units == "degrees_north" and d["lon"].units == "degrees_east"
        assert d["x_m"].grid_mapping == "crs" and d["crs"].grid_mapping_name == "polar_stereographic"
        assert list(d["quality_flags"].flag_masks) == [1, 2, 4]
        assert d["quality_flags"].flag_meanings.split() == list(publish.FLAG_BITS)
        assert d.creator_name == "Chua, Sean Minhui Tashi" and d.creator_id == "https://orcid.org/0000-0002-8462-3170"
        assert d.license == "CC-BY-4.0" and d.release_id == "test-release" and d.period == PERIOD
        assert np.all(np.diff(d["time_start"][:]) >= 0)
        for name in ("x_m", "y_m", "dx_m", "dy_m", "speed_m_per_day", "correlation", "direct_match_share", "interpolated",
                     "quality_flags", "is_primary", "scene_index", "point_index", "lon", "lat", "time"):
            assert d[name].dimensions == ("obs",)
    finally:
        d.close()


def test_validate_passes_and_detects_a_changed_value(tmp_path):
    manifest, parquet_dir, out, result = make(tmp_path)
    assert publish_netcdf.validate(parquet_dir, out)[0]["status"] == "passed"
    path = out / f"limosat_{PERIOD}.nc"
    d = netCDF4.Dataset(path, "a")
    d["x_m"][0] = d["x_m"][0] + 1.0
    d.close()
    with pytest.raises(AssertionError):
        publish_netcdf.validate(parquet_dir, out)


def test_netcdf_values_round_trip_the_parquet_table(tmp_path):
    manifest, parquet_dir, out, result = make(tmp_path)
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_dir / f"limosat_{PERIOD}.parquet").to_pandas()
    d = netCDF4.Dataset(out / f"limosat_{PERIOD}.nc")
    d.set_auto_maskandscale(False)
    tid = np.repeat(d["trajectory_id"][:], d["rowsize"][:])
    names = list(d["scene_name"][:])
    nc = {"tid": tid, "pidx": d["point_index"][:], "x": d["x_m"][:], "scene": [names[i] for i in d["scene_index"][:]],
          "dx": d["dx_m"][:], "primary": d["is_primary"][:].astype(bool)}
    d.close()
    for tid_value, group in table.groupby("trajectory_id"):
        group = group.sort_values("point_index")
        sel = nc["tid"] == tid_value
        order = np.argsort(nc["pidx"][sel])
        assert np.array_equal(group.x_m.to_numpy(), nc["x"][sel][order])
        assert list(group.scene) == [s for s, keep in zip(nc["scene"], sel) if keep] or sorted(group.scene) == sorted(
            s for s, keep in zip(nc["scene"], sel) if keep)
        assert np.array_equal(group.is_primary.to_numpy(), nc["primary"][sel][order])
        assert np.isnan(nc["dx"][sel][order][0])  # the first observation has no displacement
    manifest_nc = json.loads((out / "manifest.json").read_text())
    assert manifest_nc["format"] == publish_netcdf.FORMAT and manifest_nc["release_id"] == "test-release"
