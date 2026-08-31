import pandas as pd

from experiments.analyze_buoy_probe_stage_fates import (
    pattern_threshold_diagnostic,
    summarize_fates,
    trace_one_step_fates,
)


def test_trace_records_interpolation_then_pattern_rejection():
    expected = pd.DataFrame(
        {
            "probe_id": ["a|1", "b|1"],
            "target_image_id": [2, 2],
            "target_x": [1000.0, 2000.0],
            "target_y": [1000.0, 2000.0],
        }
    )
    linkage = pd.DataFrame({"probe_id": ["a|1"], "trajectory_id": [10]})
    image_map = pd.DataFrame({"run_image_id": [2], "catalog_image_id": [2]})
    candidates = pd.DataFrame(
        {
            "trajectory_id": [10],
            "target_image_id": [2],
            "target_x": [101000.0],
            "target_y": [1000.0],
            "descriptor_pass": [True],
            "motion_pass": [False],
            "model_inlier": [False],
            "accepted": [False],
            "rejection_reason": ["motion_distance"],
        }
    )
    stages = pd.DataFrame(
        [
            {
                "stage": "post_interpolation",
                "trajectory_id": 10,
                "target_image_id": 2,
                "target_x": 1100.0,
                "target_y": 1000.0,
                "interpolated": 1,
                "accepted": True,
            },
            {
                "stage": "pattern_matching",
                "trajectory_id": 10,
                "target_image_id": 2,
                "pre_pattern_x": 1100.0,
                "pre_pattern_y": 1000.0,
                "corrected_x": 1400.0,
                "corrected_y": 1000.0,
                "pattern_available": True,
                "correlation": 0.296,
                "accepted": False,
            },
        ]
    )

    result = trace_one_step_fates(expected, linkage, image_map, candidates, stages)

    assert result.loc[0, "candidate_best_error_m"] == 100000.0
    assert result.loc[0, "post_interpolation_error_m"] == 100.0
    assert result.loc[0, "pattern_corrected_error_m"] == 400.0
    assert result.loc[0, "outcome_stage"] == "pattern_correlation"
    assert result.loc[1, "outcome_stage"] == "source_probe_unlinked"
    summary = summarize_fates(result)
    assert summary["expected_transitions"] == 2
    assert summary["final_accepted"] == 0
    calibration = pattern_threshold_diagnostic(result, thresholds=(0.29, 0.30))
    assert calibration["pattern_positions_retained"].tolist() == [1, 0]
    assert calibration.loc[0, "within_2km_fraction_all"] == 0.5


def test_trace_follows_replacement_after_convergence_pruning():
    expected = pd.DataFrame(
        {
            "probe_id": ["a|1"],
            "target_image_id": [2],
            "target_x": [1000.0],
            "target_y": [1000.0],
        }
    )
    linkage = pd.DataFrame({"probe_id": ["a|1"], "trajectory_id": [10]})
    image_map = pd.DataFrame({"run_image_id": [2], "catalog_image_id": [2]})
    candidates = pd.DataFrame(
        {
            "trajectory_id": [10],
            "target_image_id": [2],
            "target_x": [1100.0],
            "target_y": [1000.0],
            "descriptor_pass": [True],
            "motion_pass": [True],
            "model_inlier": [True],
            "accepted": [True],
            "rejection_reason": [None],
        }
    )
    stages = pd.DataFrame(
        [
            {
                "stage": "post_interpolation",
                "trajectory_id": 10,
                "target_image_id": 2,
                "target_x": 1100.0,
                "target_y": 1000.0,
                "interpolated": 0,
                "accepted": True,
            },
            {
                "stage": "convergence",
                "trajectory_id": 10,
                "target_image_id": 2,
                "converged_to": 20,
                "accepted": False,
            },
            {
                "stage": "final_acceptance",
                "trajectory_id": 20,
                "target_image_id": 2,
                "target_x": 1200.0,
                "target_y": 1000.0,
                "accepted": True,
            },
        ]
    )

    result = trace_one_step_fates(expected, linkage, image_map, candidates, stages)

    assert not result.loc[0, "final_accepted"]
    assert result.loc[0, "convergence_pruned"]
    assert result.loc[0, "replacement_trajectory_id"] == 20
    assert result.loc[0, "replacement_final_error_m"] == 200.0
    assert result.loc[0, "measurement_represented_final"]
    assert result.loc[0, "measurement_represented_error_m"] == 200.0
    summary = summarize_fates(result)
    assert summary["final_accepted"] == 0
    assert summary["replacement_final_accepted"] == 1
    assert summary["measurement_represented_within_2km"] == 1
