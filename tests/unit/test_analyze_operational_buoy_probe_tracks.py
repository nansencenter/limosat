import pandas as pd

from experiments.analyze_operational_buoy_probe_tracks import (
    attach_tracked_positions,
    metric_summary,
)


def test_missing_probe_tracks_remain_in_accuracy_denominator():
    expected = pd.DataFrame(
        {
            "probe_id": ["a|1", "b|1"],
            "target_image_id": [2, 2],
            "target_x": [100.0, 200.0],
            "target_y": [100.0, 200.0],
        }
    )
    linkage = pd.DataFrame(
        {"probe_id": ["a|1", "b|1"], "trajectory_id": [10, 11]}
    )
    tracks = pd.DataFrame(
        {
            "trajectory_id": [10],
            "image_id": [2],
            "tracked_x": [110.0],
            "tracked_y": [100.0],
            "interpolated": [0],
            "corr": [0.8],
        }
    )
    image_map = pd.DataFrame({"run_image_id": [2], "catalog_image_id": [2]})

    result = attach_tracked_positions(expected, linkage, tracks, image_map)
    summary = metric_summary(result, "fixture")

    assert result["tracked"].tolist() == [True, False]
    assert summary["tracked_fraction"] == 0.5
    assert summary["within_500m_fraction_all"] == 0.5
    assert summary["within_2km_fraction_tracked"] == 1.0
