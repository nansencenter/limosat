import pandas as pd

from experiments.compare_dense_field_buoy_runs import compare, summarize


def frame(nearest_distance, nearest_error, local10_error, local50_error):
    return pd.DataFrame(
        {
            "transition_id": ["a", "b", "c"],
            "buoy_id": [1, 1, 2],
            "source_image_id": [1, 2, 3],
            "target_image_id": [2, 3, 4],
            "elapsed_hours": [2.0, 24.0, 24.0],
            "cadence_band": ["short", "long", "long"],
            "month": ["2020-01", "2020-01", "2020-02"],
            "nearest_source_distance_m": nearest_distance,
            "nearest_endpoint_error_m": nearest_error,
            "local_average_10km_endpoint_error_m": local10_error,
            "local_average_50km_endpoint_error_m": local50_error,
        }
    )


def test_paired_summary_keeps_missing_predictions_and_counts_gains_and_losses():
    baseline = frame(
        [5_000.0, 5_000.0, 11_000.0],
        [100.0, 100.0, 100.0],
        [100.0, 100.0, None],
        [100.0, 100.0, 100.0],
    )
    candidate = frame(
        [5_000.0, 5_000.0, 5_000.0],
        [100.0, 3_000.0, 100.0],
        [100.0, 3_000.0, 100.0],
        [100.0, 3_000.0, 100.0],
    )

    paired = compare(baseline, candidate)
    summary = summarize(paired)
    nearest = summary.loc[
        (summary["stratum"] == "all") & (summary["method"] == "nearest_10km")
    ].iloc[0]

    assert nearest.expected == 3
    assert nearest.baseline_available == 2
    assert nearest.candidate_available == 3
    assert nearest.baseline_correct == 2
    assert nearest.candidate_correct == 2
    assert nearest.correct_gains == 1
    assert nearest.correct_losses == 1
