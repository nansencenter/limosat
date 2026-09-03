from dataclasses import replace
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from limosat import OpenWaterConfig
from limosat.pairs import PairProcessor
from limosat.tile_gates import (
    OpenWaterEvidence,
    SicField,
    SicFileIndex,
    load_sic_field,
    tile_open_water_evidence,
    valid_tile_overlap_gate,
)

from test_efficientloftr_fields import CountingMatcher, _config, _pair


def _write_sic(path: Path, value: float) -> None:
    values = np.full((40, 40), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=40,
        height=40,
        count=1,
        dtype="float32",
        crs="EPSG:3413",
        transform=from_origin(-20_000.0, 20_000.0, 1_000.0, 1_000.0),
    ) as dataset:
        dataset.write(values, 1)


def test_load_sic_field_preserves_masked_integer_nodata_as_nan(tmp_path):
    path = tmp_path / "ice_conc_nh_polstere-100_multi_202001011200.tif"
    nodata = -32767
    values = np.full((4, 5), 730, dtype=np.int16)
    values[1, 2] = nodata
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="int16",
        nodata=nodata,
        crs="EPSG:3413",
        transform=from_origin(-2_500.0, 2_000.0, 1_000.0, 1_000.0),
    ) as dataset:
        dataset.write(values, 1)
        dataset.scales = (0.1,)

    field = load_sic_field(path)

    assert field.values_percent.dtype == np.float64
    np.testing.assert_allclose(field.values_percent[0, 0], 73.0)
    assert np.isnan(field.values_percent[1, 2])


def test_open_water_requires_complete_below_threshold_evidence():
    field = SicField(
        np.zeros((5, 5), dtype=np.float64),
        from_origin(-2_500.0, 2_500.0, 1_000.0, 1_000.0),
        "EPSG:3413",
        Path("sic.tif"),
        "ice_conc_unfiltered",
    )
    complete = tile_open_water_evidence(field, (0.0, 0.0), 4_000.0, 3413)
    missing = tile_open_water_evidence(None, (0.0, 0.0), 4_000.0, 3413)
    incomplete_field = replace(
        field, values_percent=np.where(np.eye(5, dtype=bool), np.nan, 0.0)
    )
    incomplete = tile_open_water_evidence(
        incomplete_field, (0.0, 0.0), 4_000.0, 3413
    )

    assert complete == OpenWaterEvidence(True, 25, 0.0)
    assert not missing.confidently_open
    assert not incomplete.confidently_open


def test_open_water_skips_matcher_only_when_both_dates_are_open(tmp_path):
    source_sic = tmp_path / "ice_conc_nh_polstere-100_multi_202001011200.tif"
    target_sic = tmp_path / "ice_conc_nh_polstere-100_multi_202001021200.tif"
    _write_sic(source_sic, 0.0)
    _write_sic(target_sic, 0.0)
    config = replace(
        _config(tmp_path),
        open_water=OpenWaterConfig(enabled=True, sic_root=str(tmp_path)),
    )
    matcher = CountingMatcher()

    result = PairProcessor(
        config, matcher, SicFileIndex(tmp_path)
    ).process(_pair(tmp_path))

    assert matcher.calls == 0
    assert result.matcher_calls == 0
    assert result.diagnostics["skipped_open_water_both_dates"] > 0
    assert len(result.ancillary_inputs) == 2


def test_one_icy_date_keeps_tile_for_matching(tmp_path):
    source_sic = tmp_path / "ice_conc_nh_polstere-100_multi_202001011200.tif"
    target_sic = tmp_path / "ice_conc_nh_polstere-100_multi_202001021200.tif"
    _write_sic(source_sic, 0.0)
    _write_sic(target_sic, 50.0)
    config = replace(
        _config(tmp_path),
        open_water=OpenWaterConfig(enabled=True, sic_root=str(tmp_path)),
    )
    matcher = CountingMatcher()

    result = PairProcessor(config, matcher).process(_pair(tmp_path))

    assert matcher.calls > 0
    assert result.diagnostics["skipped_open_water_both_dates"] == 0


def test_physics_gate_requires_reachable_valid_support():
    source = np.zeros((10, 10), dtype=bool)
    target = np.zeros((10, 10), dtype=bool)
    source[4:6, 4:6] = True
    target[4:6, 4:6] = True

    gate = valid_tile_overlap_gate(
        source,
        target,
        (0.0, 0.0),
        (10_000.0, 0.0),
        100.0,
        1_000.0,
    )

    assert gate.skip
    assert gate.reason == "no_physics_reachable_valid_overlap"
