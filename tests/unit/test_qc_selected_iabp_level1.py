import pandas as pd

from experiments.qc_selected_iabp_level1 import (
    load_level1_track,
    platform_evidence_status,
)


def test_aoml_surface_velocity_drifter_is_not_accepted_as_ice_platform():
    assert (
        platform_evidence_status(
            "BD2GHI", "AOML", 4, 1.0, -10.0, 3, -1.8
        )
        == "reject_aoml_ocean_drifter"
    )


def test_cold_usiabp_platform_is_supported_as_on_ice():
    assert (
        platform_evidence_status(
            "SVP-B", "USIABP/AARI", 4, 1.0, -6.0, 3, -1.8
        )
        == "accept_on_ice_platform_evidence"
    )


def test_missing_platform_and_temperature_evidence_remains_on_hold():
    assert (
        platform_evidence_status("", "", 0, float("nan"), float("nan"), 3, -1.8)
        == "hold_missing_platform_metadata"
    )


def test_level1_loader_accepts_minute_second_header_variant(tmp_path):
    path = tmp_path / "track.csv"
    pd.DataFrame(
        {
            "BuoyID": ["123", "123"],
            "Year": [2020, 2020],
            "Month": [1, 1],
            "Day": [1, 1],
            "Hour": [0, 1],
            "Minute": [5, 5],
            "Second": [30, 30],
            "Lat": [80.0, 80.1],
            "Lon": [10.0, 10.1],
            "Ts": [-5.0, -5.0],
            "Ta": [-10.0, -10.0],
            "Th": [-4.0, -4.0],
        }
    ).to_csv(path, index=False)

    track = load_level1_track(path)

    assert track["buoy_id"].tolist() == ["123", "123"]
    assert track["time"].dt.minute.tolist() == [5, 5]
    assert track["time"].dt.second.tolist() == [30, 30]
