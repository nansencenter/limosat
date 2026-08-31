from pathlib import Path

import pandas as pd

from experiments.build_full70_level1_tracking_fixtures import build_outputs


def frame(
    buoy_id: str,
    image_id: int,
    when: str,
    path: Path,
    ready: bool = True,
) -> dict:
    return {
        "buoy_id": buoy_id,
        "image_id": image_id,
        "image_time": when,
        "x": float(image_id * 100),
        "y": 0.0,
        "level1_x_3413": float(image_id * 100 + 5),
        "level1_y_3413": 2.0,
        "truth_ready_after_level1": ready,
        "level1_final_status": (
            "ready_level1_validated" if ready else "hold_128px_image_border"
        ),
        "standard_vae_output_path": str(path),
        "resolved_product_name": f"scene_{image_id}",
        "sic_regime": "pack_ice",
        "spatial_block": "x+000_y+000",
    }


def test_builder_uses_level1_truth_and_breaks_experiment_paths(tmp_path):
    paths = [tmp_path / f"scene_{number}.tiff" for number in range(1, 5)]
    for path in paths:
        path.touch()
    rows = [
        frame("10", 1, "2020-02-29T23:00:00Z", paths[0]),
        frame("10", 2, "2020-03-01T01:00:00Z", paths[1]),
        frame("10", 3, "2020-03-01T03:00:00Z", paths[2]),
        frame("10", 4, "2020-03-05T12:00:00Z", paths[3]),
    ]
    _, observations, transitions, products = build_outputs(
        pd.DataFrame(rows), maximum_gap_hours=72.0
    )

    assert observations.iloc[0]["x"] == 105.0
    assert observations.iloc[0]["catalog_x_3413"] == 100.0
    assert observations["continuous_trajectory_id"].nunique() == 2
    assert observations["experiment_trajectory_id"].nunique() == 3
    assert len(transitions) == 1
    assert transitions.iloc[0]["experiment_split"] == "development"
    assert products["summary"]["split_safe_transitions"] == 1


def test_builder_holds_missing_vae_and_preserves_audit_status(tmp_path):
    available = tmp_path / "available.tiff"
    available.touch()
    rows = [
        frame("20", 1, "2020-01-01T00:00:00Z", available),
        frame("20", 2, "2020-01-01T02:00:00Z", tmp_path / "missing.tiff"),
        frame("21", 3, "2020-01-01T00:00:00Z", available, ready=False),
    ]
    audit, observations, transitions, _ = build_outputs(
        pd.DataFrame(rows), maximum_gap_hours=72.0
    )

    assert len(observations) == 1
    assert transitions.empty
    assert audit.set_index("image_id").loc[2, "tracking_fixture_status"] == (
        "hold_standard_vae_scene_unavailable"
    )
    assert audit.set_index("image_id").loc[3, "tracking_fixture_status"] == (
        "hold_128px_image_border"
    )
