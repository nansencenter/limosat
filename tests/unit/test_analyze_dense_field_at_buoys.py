import numpy as np
import pandas as pd

from experiments.analyze_dense_field_at_buoys import (
    estimate_local_average_within_radius,
    estimate_local_displacement,
    transition_truth,
)


def test_local_estimators_recover_uniform_translation():
    source = np.array(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
    )
    translation = np.array([3.0, -2.0])
    target = source + translation
    query = np.array([4.0, 4.0])

    result = estimate_local_displacement(source, target, query)

    np.testing.assert_allclose(
        [result["nearest_target_x"], result["nearest_target_y"]],
        query + translation,
    )
    np.testing.assert_allclose(
        [
            result["inverse_distance_target_x"],
            result["inverse_distance_target_y"],
        ],
        query + translation,
    )
    np.testing.assert_allclose(
        [result["triangle_target_x"], result["triangle_target_y"]],
        query + translation,
    )
    assert result["inverse_distance_neighbour_count"] == 4
    assert result["inverse_distance_vector_spread_m"] < 1e-12


def test_triangle_is_missing_outside_source_convex_hull():
    source = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    target = source + np.array([3.0, -2.0])

    result = estimate_local_displacement(source, target, np.array([20.0, 20.0]))

    assert np.isnan(result["triangle_target_x"])
    assert np.isnan(result["triangle_target_y"])
    assert np.isfinite(result["nearest_target_x"])


def test_local_average_uses_only_neighbours_inside_radius():
    source = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]])
    target = source + np.array([[3.0, -2.0], [3.0, -2.0], [50.0, 50.0]])

    result = estimate_local_average_within_radius(
        source,
        target,
        np.array([4.0, 0.0]),
        maximum_source_distance_m=20.0,
        maximum_neighbours=4,
    )

    assert result["neighbour_count"] == 2
    np.testing.assert_allclose(
        [result["target_x"], result["target_y"]], [7.0, -2.0]
    )


def test_transition_truth_preserves_authoritative_transition_coordinates():
    transitions = pd.DataFrame(
        {
            "transition_id": ["a"],
            "buoy_id": ["b"],
            "source_image_id": [101],
            "target_image_id": [102],
            "within_dataset_split": ["evaluation"],
            "source_x": [1000.0],
            "source_y": [2000.0],
            "target_x": [1010.0],
            "target_y": [1980.0],
            "truth_dx_m": [10.0],
            "truth_dy_m": [-20.0],
        }
    )
    observations = pd.DataFrame(
        {
            "buoy_id": ["b", "b"],
            "image_id": [101, 102],
            "within_dataset_split": ["evaluation", "evaluation"],
            "acquisition_pass_id": ["pass_a", "pass_b"],
        }
    )

    result = transition_truth(transitions, observations, "evaluation")

    assert result.loc[0, "source_x"] == 1000.0
    assert result.loc[0, "target_y"] == 1980.0
    assert not result.loc[0, "same_acquisition_pass"]
