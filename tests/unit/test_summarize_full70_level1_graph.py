import pandas as pd

from experiments.summarize_full70_level1_graph import (
    expected_transition_targets,
    paired_effects,
)


def test_expected_targets_and_paired_effects_keep_untracked_denominator():
    coincidences = pd.DataFrame(
        {
            "experiment_trajectory_id": ["path", "path", "path"],
            "image_time": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"], utc=True
            ),
            "image_id": [1, 2, 3],
            "month_exclusive_buoy": [True, True, True],
        }
    )
    records = pd.DataFrame(
        [
            {
                "config": "greedy_rolling",
                "trajectory_id": "path",
                "image_id": 2,
                "status": "ok",
                "observation_index": 1,
                "endpoint_error_m": 1000.0,
            },
            {
                "config": "beam_anchor",
                "trajectory_id": "path",
                "image_id": 2,
                "status": "ok",
                "observation_index": 1,
                "endpoint_error_m": 3000.0,
            },
            {
                "config": "beam_anchor",
                "trajectory_id": "path",
                "image_id": 3,
                "status": "ok",
                "observation_index": 2,
                "endpoint_error_m": 1000.0,
            },
        ]
    )

    assert len(expected_transition_targets(coincidences)) == 2
    effects = paired_effects(records, coincidences)
    beam = effects[
        effects["config"].eq("beam_anchor")
        & effects["evaluation_subset"].eq("all_temporal")
    ].iloc[0]
    assert beam.expected_transitions == 2
    assert beam.rescued_within_2km == 1
    assert beam.regressed_from_within_2km == 1
    assert beam.net_within_2km_change == 0
