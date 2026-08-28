from datetime import datetime
from pathlib import Path

import numpy as np
import shapely
import torch

from experiments.run_efficientloftr_sequence import (
    PairSpec,
    acquisition_time_from_path,
    track_pair,
)
from limosat.learned_drift.config import EfficientLoFTRConfig
from limosat.learned_drift.features import TileRegion
from limosat.learned_drift.tile_gates import (
    SicField,
    SicFileIndex,
    tile_open_water_evidence,
    valid_tile_overlap_gate,
)


def synthetic_sic(value: float = 5.0) -> SicField:
    axis = np.arange(-100.0, 101.0, 10.0)
    values = np.full((len(axis), len(axis)), value, dtype=np.float32)
    return SicField(
        values,
        axis,
        axis,
        "EPSG:3413",
        Path("synthetic.nc"),
        "ice_conc_unfiltered",
    )


def test_sic_index_uses_latest_available_previous_day(tmp_path):
    first = tmp_path / "ice_conc_nh_polstere-100_multi_202001011200.nc"
    third = tmp_path / "ice_conc_nh_polstere-100_multi_202001031200.nc"
    first.touch()
    third.touch()
    index = SicFileIndex(tmp_path)

    assert index.resolve(datetime(2020, 1, 2), max_age_days=1) == first.resolve()
    assert index.resolve(datetime(2020, 1, 3), max_age_days=1) == third.resolve()
    assert index.resolve(datetime(2020, 1, 5), max_age_days=1) is None


def test_sentinel1_time_is_read_from_preprocessed_filename():
    value = acquisition_time_from_path(
        "/data/VAE_S1B_EW_GRDM_1SDH_20200115T194732_20200115T194832.tiff"
    )

    assert value == datetime(2020, 1, 15, 19, 47, 32)


def test_valid_overlap_gate_rejects_empty_endpoint_support():
    source = np.zeros((8, 8), dtype=bool)
    target = np.ones((8, 8), dtype=bool)

    result = valid_tile_overlap_gate(source, target, (0.0, 0.0), (0.0, 0.0), 1.0, 10.0)

    assert result.skip
    assert result.reason == "no_source_core_support"


def test_valid_overlap_gate_rejects_only_physics_unreachable_bounds():
    source = np.ones((8, 8), dtype=bool)
    target = np.ones((8, 8), dtype=bool)

    rejected = valid_tile_overlap_gate(
        source, target, (0.0, 0.0), (100.0, 0.0), 1.0, 50.0
    )
    retained = valid_tile_overlap_gate(
        source, target, (0.0, 0.0), (100.0, 0.0), 1.0, 100.0
    )

    assert rejected.skip
    assert rejected.reason == "no_physics_reachable_valid_overlap"
    assert not retained.skip


def test_open_water_requires_complete_low_sic_support():
    open_water = synthetic_sic()
    incomplete = synthetic_sic()
    incomplete.values_percent[10, 10] = np.nan
    ice = synthetic_sic()
    ice.values_percent[10, 10] = 80.0

    open_result = tile_open_water_evidence(
        open_water, (0.0, 0.0), 50.0, 3413, samples_per_axis=5
    )
    incomplete_result = tile_open_water_evidence(
        incomplete, (0.0, 0.0), 50.0, 3413, samples_per_axis=5
    )
    ice_result = tile_open_water_evidence(
        ice, (0.0, 0.0), 50.0, 3413, samples_per_axis=5
    )

    assert open_result.confidently_open
    assert not incomplete_result.confidently_open
    assert not ice_result.confidently_open


def test_track_pair_skips_open_water_before_calling_matcher(monkeypatch, tmp_path):
    config = EfficientLoFTRConfig(
        pixel_size_m=1.0,
        tile_size_px=16,
        tile_margin_px=2,
        endpoint_support_radius_px=1,
        transform_grid_spacing_px=4,
        grid_spacing_m=4.0,
        maximum_neighbour_distance_m=6.0,
        agreement_distance_m=1.0,
        maximum_triangle_edge_m=6.4,
        new_point_exclusion_radius_m=2.0,
    )
    domain = shapely.box(-6.0, -6.0, 6.0, 6.0)
    region = TileRegion(0, 0, 0, (0.0, 0.0), domain)

    monkeypatch.setattr(
        "experiments.run_efficientloftr_sequence.pair_domains",
        lambda *_args: (domain, domain),
    )
    monkeypatch.setattr(
        "experiments.run_efficientloftr_sequence.tile_layout",
        lambda *_args: (region,),
    )
    monkeypatch.setattr(
        "experiments.run_efficientloftr_sequence.north_up_patch",
        lambda *_args: (
            np.zeros((16, 16), dtype=np.uint8),
            np.ones((16, 16), dtype=bool),
        ),
    )

    def unexpected_matcher_call(*_args):
        raise AssertionError("matcher must not run for a confidently open-water tile")

    monkeypatch.setattr(
        "experiments.run_efficientloftr_sequence.run_optimized_matcher",
        unexpected_matcher_call,
    )
    spec = PairSpec(1, 2, "source.tif", "target.tif", 24.0, None)

    _field, summary = track_pair(
        spec,
        model=None,
        device=torch.device("cpu"),
        config=config,
        routing_mode="same_center",
        previous_field=None,
        previous_elapsed_days=None,
        initial_displacement_m=None,
        identity_sha256="test",
        output_dir=tmp_path,
        source_sic=synthetic_sic(),
        target_sic=synthetic_sic(),
    )

    assert summary["matched_source_tiles"] == 0
    assert summary["tile_skip_counts"] == {"open_water_both_dates": 1}
    assert summary["timing_seconds"]["matching"] == 0.0
