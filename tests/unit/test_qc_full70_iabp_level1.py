import math

import pandas as pd

from experiments.qc_full70_iabp_level1 import (
    final_link_status,
    position_repeat_context,
)


def test_level1_can_resolve_catalog_track_context_hold():
    assert final_link_status(
        "hold_full_level1_track_context",
        True,
        True,
        1.0,
        20_000.0,
        15.0,
        "hold_insufficient_on_ice_evidence",
        6.0,
        100_000.0,
        500.0,
    ) == "ready_level1_validated"


def test_miz_requires_positive_platform_evidence():
    assert final_link_status(
        "hold_level1_on_ice_platform_evidence",
        True,
        True,
        1.0,
        20_000.0,
        15.0,
        "hold_insufficient_on_ice_evidence",
        6.0,
        100_000.0,
        500.0,
    ) == "hold_insufficient_on_ice_evidence"


def test_missing_or_invalid_level1_tracks_are_not_silently_accepted():
    assert final_link_status(
        "ready_current_catalog_qc",
        False,
        False,
        math.nan,
        math.nan,
        math.nan,
        "hold_missing_platform_metadata",
        6.0,
        100_000.0,
        500.0,
    ) == "hold_missing_level1_file"
    assert final_link_status(
        "ready_current_catalog_qc",
        True,
        True,
        1.0,
        20_000.0,
        750.0,
        "accept_on_ice_platform_evidence",
        6.0,
        100_000.0,
        500.0,
    ) == "reject_level1_catalog_position_disagreement"


def test_repeated_level1_positions_record_span_and_adjacent_jump_speed():
    track = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2020-01-01T00:00Z", "2020-01-01T01:00Z", "2020-01-01T02:00Z", "2020-01-01T03:00Z"]
            ),
            "Lat": [80.0, 80.1, 80.1, 80.2],
            "Lon": [10.0, 10.1, 10.1, 10.2],
            "x_3413": [0.0, 100.0, 100.0, 300.0],
            "y_3413": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = position_repeat_context(track, pd.Timestamp("2020-01-01T01:20Z"))

    assert result["level1_repeat_fix_count"] == 2
    assert result["level1_repeat_span_hours"] == 1.0
    assert result["level1_repeat_previous_speed_m_per_day"] == 2400.0
    assert result["level1_repeat_next_speed_m_per_day"] == 4800.0
