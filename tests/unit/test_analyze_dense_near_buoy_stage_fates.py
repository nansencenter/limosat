import pandas as pd

from experiments.analyze_dense_near_buoy_stage_fates import (
    trace_local_dense_points,
)


def test_trace_local_dense_point_uses_displacement_not_target_position():
    truth = pd.DataFrame(
        {
            "transition_id": ["a"],
            "buoy_id": [1],
            "source_image_id": [10],
            "target_image_id": [20],
            "source_x": [100.0],
            "source_y": [100.0],
            "target_x": [110.0],
            "target_y": [100.0],
        }
    )
    source_points = pd.DataFrame(
        {"trajectory_id": [5], "x": [105.0], "y": [100.0]}
    )
    candidates = pd.DataFrame(
        {
            "trajectory_id": [5],
            "target_image_id": [2],
            "target_x": [115.0],
            "target_y": [100.0],
            "descriptor_pass": [True],
            "motion_pass": [True],
            "model_inlier": [True],
            "accepted": [True],
        }
    )
    stages = pd.DataFrame(
        [
            {
                "stage": "final_acceptance",
                "trajectory_id": 5,
                "target_image_id": 2,
                "target_x": 115.0,
                "target_y": 100.0,
                "accepted": True,
            }
        ]
    )

    points, buoys = trace_local_dense_points(
        truth, source_points, candidates, stages, target_run_image_id=2, local_radius_m=10.0
    )

    assert points.loc[0, "candidate_best_displacement_error_m"] == 0.0
    assert points.loc[0, "final_displacement_error_m"] == 0.0
    assert buoys.loc[0, "has_final_vector_within_2km"]
