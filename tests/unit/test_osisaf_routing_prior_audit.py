from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from experiments.osisaf_routing_prior_audit import (
    OSI_FILENAME,
    advect_with_osi455,
    agreement_band,
    daily_segments,
)


UTC = timezone.utc


def test_daily_segments_cover_partial_noon_to_noon_products():
    start = datetime(2020, 3, 21, 18, tzinfo=UTC)
    end = datetime(2020, 3, 23, 6, tzinfo=UTC)

    segments = daily_segments(start, end)

    assert [segment.product_end.day for segment in segments] == [22, 23]
    assert [segment.fraction_of_day for segment in segments] == [0.75, 0.75]


def test_daily_segments_do_not_add_zero_length_product_at_noon():
    segments = daily_segments(
        datetime(2020, 3, 21, 12, tzinfo=UTC),
        datetime(2020, 3, 22, 12, tzinfo=UTC),
    )

    assert len(segments) == 1
    assert segments[0].product_end.day == 22
    assert segments[0].fraction_of_day == 1.0


def write_uniform_osi_file(path, end_day, dx_km, dy_km, flag=30):
    coords = np.array([-112.5, -37.5, 37.5, 112.5])
    shape = (1, 4, 4)
    dataset = xr.Dataset(
        {
            "dX": (("time", "yc", "xc"), np.full(shape, dx_km, np.float32)),
            "dY": (("time", "yc", "xc"), np.full(shape, dy_km, np.float32)),
            "uncert_dX_and_dY": (
                ("time", "yc", "xc"),
                np.full(shape, 2.0, np.float32),
            ),
            "status_flag": (
                ("time", "yc", "xc"),
                np.full(shape, flag, np.int8),
            ),
        },
        coords={"time": [0], "xc": coords, "yc": coords[::-1]},
        attrs={"product_id": "OSI-455"},
    )
    filename = OSI_FILENAME.format(date=end_day.strftime("%Y%m%d"))
    dataset.to_netcdf(path / filename)


def test_advect_scales_daily_vectors_to_exact_pair_interval(tmp_path):
    product_end = datetime(2020, 3, 22, 12, tzinfo=UTC)
    write_uniform_osi_file(tmp_path, product_end, dx_km=24.0, dy_km=-12.0)

    result = advect_with_osi455(
        np.array([[0.0, 0.0]]),
        datetime(2020, 3, 22, 0, tzinfo=UTC),
        datetime(2020, 3, 22, 12, tzinfo=UTC),
        tmp_path,
        analysis_epsg=6931,
    )

    assert result["available"].tolist() == [True]
    assert result["displacement_m"][0] == pytest.approx([12_000.0, -6_000.0])
    assert result["uncertainty_m"][0] == pytest.approx(1_000.0)
    assert result["flags"].tolist() == ["30"]


def test_advect_rejects_unusable_nearest_status(tmp_path):
    product_end = datetime(2020, 3, 22, 12, tzinfo=UTC)
    write_uniform_osi_file(tmp_path, product_end, dx_km=24.0, dy_km=-12.0, flag=2)

    result = advect_with_osi455(
        np.array([[0.0, 0.0]]),
        datetime(2020, 3, 21, 12, tzinfo=UTC),
        datetime(2020, 3, 22, 12, tzinfo=UTC),
        tmp_path,
        analysis_epsg=6931,
    )

    assert result["available"].tolist() == [False]
    assert np.isnan(result["displacement_m"][0]).all()


def test_agreement_bands_include_threshold_endpoints():
    assert agreement_band(5_000) == "agree_le05km"
    assert agreement_band(10_000) == "differ_05_10km"
    assert agreement_band(20_000) == "differ_10_20km"
    assert agreement_band(20_001) == "differ_gt20km"
