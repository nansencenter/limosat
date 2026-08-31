import pandas as pd

from experiments.build_osisaf_long_gap_references import (
    build_all,
    build_reference_rows,
)


def observations():
    rows = []
    for image_id, time, path, offset in (
        (1, "2020-01-01T00:00Z", "one.tif", 0.0),
        (2, "2020-01-02T00:00Z", "two.tif", 100.0),
        (3, "2020-01-04T00:00Z", "three.tif", 500.0),
    ):
        for buoy_id, extra in (("A", 0.0), ("B", 10.0)):
            rows.append(
                {
                    "buoy_id": buoy_id,
                    "image_id": image_id,
                    "image_time": pd.Timestamp(time),
                    "image_filepath": path,
                    "x": offset + extra,
                    "y": -offset + extra,
                    "experiment_split": "evaluation",
                    "sic_regime": "pack_ice_ge80",
                    "spatial_block": "x0_y0",
                }
            )
    return pd.DataFrame(rows)


def test_reference_uses_direct_endpoint_positions_and_elapsed_time():
    manifest, rows = build_reference_rows(observations(), 1, 3)

    assert manifest["elapsed_hours"] == 72.0
    assert manifest["buoys"] == 2
    assert rows["truth_dx_m"].tolist() == [500.0, 500.0]
    assert rows["truth_dy_m"].tolist() == [-500.0, -500.0]
    assert rows["experiment_split"].tolist() == ["evaluation", "evaluation"]


def test_build_all_applies_predeclared_duration_and_truth_rules(tmp_path):
    plan = {
        "sequences": [
            {"name": "long", "image_ids": [1, 2, 3]},
            {"name": "short", "image_ids": [1, 2]},
        ]
    }

    report = build_all(plan, observations(), tmp_path, minimum_hours=30.0)

    assert report["selected_pairs"] == 1
    assert report["selected"][0]["case_id"] == "full70_simulated_gap_1_3"
    assert report["excluded"] == [
        {"sequence": "short", "edge": [1, 2], "reason": "below_minimum_hours"}
    ]
