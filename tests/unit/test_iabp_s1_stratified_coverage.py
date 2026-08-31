from __future__ import annotations

import pandas as pd

from experiments.audit_iabp_s1_stratified_coverage import (
    cadence_band,
    bracket_track_quality,
    sic_regime,
    spatial_block_id,
)


def test_scientific_strata_have_explicit_boundaries():
    assert sic_regime(float("nan")) == "missing"
    assert sic_regime(0.1499) == "open_water_lt15"
    assert sic_regime(0.15) == "marginal_ice_15_80"
    assert sic_regime(0.7999) == "marginal_ice_15_80"
    assert sic_regime(0.80) == "pack_ice_ge80"
    assert cadence_band(5.9) == "under_6h"
    assert cadence_band(6.0) == "6_to_12h"
    assert cadence_band(24.0) == "24_to_48h"


def test_spatial_blocks_use_floor_for_negative_projected_coordinates():
    assert spatial_block_id(1.0, 1.0, 200_000.0) == "x+000_y+000"
    assert spatial_block_id(-1.0, -1.0, 200_000.0) == "x-001_y-001"
    assert spatial_block_id(-200_001.0, 400_001.0, 200_000.0) == "x-002_y+002"


def test_track_qc_uses_observed_bracket_not_exact_time_coincidence_gap():
    tracks = pd.DataFrame(
        {
            "buoy_id": ["ice", "ice", "jump", "jump"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01T00:00Z",
                    "2020-01-01T01:00Z",
                    "2020-01-01T00:00Z",
                    "2020-01-01T01:00Z",
                ],
                utc=True,
            ),
            "x": [0.0, 100.0, 0.0, 10_000.0],
            "y": [0.0, 0.0, 0.0, 0.0],
        }
    )
    exact = pd.DataFrame(
        {
            "buoy_id": ["ice", "jump"],
            "image_time": pd.to_datetime(
                ["2020-01-01T00:30Z", "2020-01-01T00:30Z"], utc=True
            ),
        }
    )
    qc = bracket_track_quality(
        exact,
        tracks,
        maximum_gap_hours=6.0,
        gross_speed_m_per_day=100_000.0,
    )
    assert qc.loc[0, "track_qc_pass"]
    assert not qc.loc[1, "track_qc_pass"]
    assert qc.loc[0, "track_qc_status"] == "pass"
    assert qc.loc[1, "track_qc_status"] == "gross_track_speed_exclude"
    assert qc.loc[0, "track_interpolation_fraction"] == 0.5
