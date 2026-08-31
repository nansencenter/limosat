import pandas as pd
import pytest

from experiments.run_full70_efficientloftr_batch import (
    buoy_metrics,
    pair_reference_rows,
    validate_plan,
)


def transitions():
    return pd.DataFrame(
        {
            "buoy_id": ["A", "A"],
            "source_image_id": [1, 2],
            "target_image_id": [2, 3],
            "source_image_time": pd.to_datetime(
                ["2020-01-01T00:00Z", "2020-01-02T00:00Z"]
            ),
            "target_image_time": pd.to_datetime(
                ["2020-01-02T00:00Z", "2020-01-03T00:00Z"]
            ),
            "source_image_filepath": ["one.tif", "two.tif"],
            "target_image_filepath": ["two.tif", "three.tif"],
            "elapsed_hours": [24.0, 24.0],
            "truth_dx_m": [100.0, 120.0],
            "truth_dy_m": [-20.0, -10.0],
            "cadence_band": ["12_to_30h", "12_to_30h"],
            "experiment_split": ["evaluation", "evaluation"],
            "source_sic_regime": ["pack", "pack"],
            "target_sic_regime": ["pack", "pack"],
            "source_spatial_block": ["x0_y0", "x0_y0"],
            "target_spatial_block": ["x0_y0", "x0_y0"],
        }
    )


def test_plan_requires_real_edges_and_covers_every_image():
    result = validate_plan(
        {"sequences": [{"name": "chain", "image_ids": [1, 2, 3]}]},
        transitions(),
    )

    assert result == {
        "unique_images": 3,
        "sequence_paths": 1,
        "pair_runs": 2,
        "unique_pair_runs": 2,
    }
    with pytest.raises(ValueError, match="absent from transitions"):
        validate_plan(
            {"sequences": [{"name": "broken", "image_ids": [1, 3, 2]}]},
            transitions(),
        )


def test_reference_rows_use_source_buoy_position_and_declared_units():
    observations = pd.DataFrame(
        {
            "buoy_id": ["A"],
            "image_id": [1],
            "x": [1_000_000.0],
            "y": [-200_000.0],
        }
    )

    manifest, buoy = pair_reference_rows(1, 2, transitions(), observations)

    assert manifest["analysis_crs"] == "EPSG:3413"
    assert manifest["elapsed_hours"] == 24.0
    assert manifest["buoys"] == 1
    assert buoy.loc[0, "source_x"] == 1_000_000.0
    assert buoy.loc[0, "source_y"] == -200_000.0
    assert buoy.loc[0, "truth_dx_m"] == 100.0


def test_buoy_metrics_keep_unavailable_cases_in_accuracy_denominator():
    rows = pd.DataFrame(
        {
            "available": [True, True, False],
            "error_m": [100.0, 3_000.0, float("nan")],
        }
    )

    result = buoy_metrics(rows)

    assert result["expected"] == 3
    assert result["available"] == 2
    assert result["within_2km"] == 1
    assert result["within_2km_of_expected_fraction"] == pytest.approx(1 / 3)
    assert result["within_2km_of_available_fraction"] == 0.5
