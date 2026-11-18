from datetime import datetime, timezone

import numpy as np
import xarray as xr
from pyproj import CRS

from experiments.osisaf405_drift import advect_with_osi405, osi405_path


def _write_constant_product(path, start, end, dx_km, status=30):
    shape = (1, 2, 2)
    start = start.replace(tzinfo=None)
    end = end.replace(tzinfo=None)
    dataset = xr.Dataset(
        data_vars={
            "Polar_Stereographic_Grid": xr.DataArray(
                0, attrs=CRS.from_epsg(3412).to_cf()
            ),
            "time_bnds": (("time", "nv"), np.array([[start, end]], dtype="datetime64[ns]")),
            "dX": (("time", "yc", "xc"), np.full(shape, dx_km)),
            "dY": (("time", "yc", "xc"), np.zeros(shape)),
            "uncert_dX_and_dY": (("time", "yc", "xc"), np.full(shape, 2.0)),
            "dt0": (("time", "yc", "xc"), np.zeros(shape)),
            "dt1": (("time", "yc", "xc"), np.zeros(shape)),
            "status_flag": (("time", "yc", "xc"), np.full(shape, status, dtype=np.int8)),
        },
        coords={
            "time": np.array([end], dtype="datetime64[ns]"),
            "yc": np.array([-100.0, 100.0]),
            "xc": np.array([-100.0, 100.0]),
        },
        attrs={"product_id": "OSI-405-d"},
    )
    dataset.to_netcdf(path)


def test_advect_scales_48_hour_displacement_as_velocity(tmp_path):
    product_end = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    product_start = datetime(2025, 12, 31, 12, tzinfo=timezone.utc)
    path = osi405_path(tmp_path, product_end)
    _write_constant_product(path, product_start, product_end, dx_km=48.0)

    result = advect_with_osi405(
        np.array([[0.0, 0.0]]),
        datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        product_end,
        tmp_path,
        analysis_epsg=3412,
    )

    np.testing.assert_allclose(result["displacement_m"], [[24_000.0, 0.0]], atol=0.01)
    np.testing.assert_allclose(result["uncertainty_m"], [1_000.0])
    assert result["available"].tolist() == [True]
    assert result["flags"].tolist() == ["30"]


def test_advect_marks_rejected_status_unavailable(tmp_path):
    product_end = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    product_start = datetime(2025, 12, 31, 12, tzinfo=timezone.utc)
    path = osi405_path(tmp_path, product_end)
    _write_constant_product(path, product_start, product_end, dx_km=48.0, status=2)

    result = advect_with_osi405(
        np.array([[0.0, 0.0]]),
        datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        product_end,
        tmp_path,
        analysis_epsg=3412,
    )

    assert result["available"].tolist() == [False]
    assert np.isnan(result["displacement_m"]).all()
