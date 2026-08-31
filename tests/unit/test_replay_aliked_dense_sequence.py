import numpy as np
import pandas as pd

from experiments.replay_aliked_dense_sequence import replay_sequence


def coherent_matches(source_x, displacement_x):
    offsets = np.array(
        [
            [-30, -30],
            [0, -30],
            [30, -30],
            [-30, 0],
            [0, 0],
            [30, 0],
            [-30, 30],
            [0, 30],
            [30, 30],
            [-15, -15],
            [15, -15],
            [0, 15],
        ],
        dtype=float,
    )
    source = offsets + np.array([source_x, 0.0])
    return pd.DataFrame(
        {
            "source_x": source[:, 0],
            "source_y": source[:, 1],
            "target_x": source[:, 0] + displacement_x,
            "target_y": source[:, 1],
            "lightglue_score": np.linspace(1.0, 0.5, len(source)),
            "physics_valid": True,
        }
    ).assign(dx_m=displacement_x, dy_m=0.0)


def buoy_transition(source_image_id, target_image_id, source_x, truth_dx):
    return pd.DataFrame(
        [
            {
                "continuous_trajectory_id": "path-1",
                "transition_id": f"path-1-{source_image_id}-{target_image_id}",
                "buoy_id": "buoy-1",
                "source_image_id": source_image_id,
                "target_image_id": target_image_id,
                "source_x": source_x,
                "source_y": 0.0,
                "truth_dx_m": truth_dx,
                "truth_dy_m": 0.0,
            }
        ]
    )


def test_replay_sequence_carries_previous_endpoint_into_next_pair():
    results, summary = replay_sequence(
        [coherent_matches(0.0, 11.0), coherent_matches(10.0, 20.0)],
        [buoy_transition(1, 2, 0.0, 10.0), buoy_transition(2, 3, 10.0, 20.0)],
    )

    propagated = results[results["mode"] == "propagated"].sort_values("step")
    assert propagated["tracking_source_error_m"].tolist() == [0.0, 1.0]
    assert propagated["endpoint_error_m"].tolist() == [1.0, 1.0]
    assert propagated["transition_id"].tolist() == ["path-1-1-2", "path-1-2-3"]
    assert summary["propagated"]["complete_paths"] == 1
    assert summary["propagated"]["final_median_error_m"] == 1.0
