import numpy as np
import pandas as pd

from experiments.compare_aliked_orb_northup import (
    attach_source_coordinates,
    interpolate_transform_grid,
    inverse_distance_local_proposal,
    retain_best_match_per_source,
    select_cases,
    tile_pixels,
    tile_origins,
    wrapped_angle_difference_degrees,
)
from experiments.replay_aliked_candidate_policies import (
    estimate_policy,
    recenter_matches,
    replay_propagated_paths,
    summarize_truth_reinitialized_paths,
    weighted_geometric_median,
)
from experiments.replay_aliked_pattern_variants import add_q4_direct_fallback


def test_select_cases_weights_image_pairs_once_and_keeps_rare_failures_separate():
    transitions = pd.DataFrame(
        {
            "transition_id": ["a", "b", "c", "d"],
            "within_dataset_split": ["development"] * 4,
            "elapsed_hours": [24.0] * 4,
            "truth_speed_km_per_day": [10.0] * 4,
            "source_image_id": [1, 1, 3, 5],
            "target_image_id": [2, 2, 4, 6],
        }
    )
    fates = pd.DataFrame(
        {
            "transition_id": ["a", "b", "c", "d"],
            "trajectory_id": [10, 11, 12, 13],
            "target_run_image_id": [20, 20, 40, 60],
            "outcome_stage": [
                "accepted",
                "no_descriptor_candidate",
                "accepted",
                "motion_gate",
            ],
            "candidate_count": [1, 0, 1, 1],
            "pattern_accepted": [True, False, True, False],
            "measurement_represented_error_m": [100.0, np.nan, 200.0, np.nan],
        }
    )

    selected = select_cases(transitions, fates, "fixed", 30.0, 30.0)

    representative = selected.loc[selected["representative_panel"]]
    assert len(representative) == 3
    assert not representative.duplicated(["source_image_id", "target_image_id"]).any()
    assert {"b", "d"}.issubset(
        set(selected.loc[selected["challenge_panel"], "transition_id"])
    )


def test_attach_source_coordinates_joins_by_buoy_and_source_image():
    cases = pd.DataFrame(
        {"transition_id": ["a"], "buoy_id": ["7"], "source_image_id": [3]}
    )
    observations = pd.DataFrame(
        {
            "buoy_id": ["7"],
            "image_id": [3],
            "x": [1000.0],
            "y": [-2000.0],
            "analysis_crs": ["EPSG:3413"],
        }
    )

    attached = attach_source_coordinates(cases, observations)

    assert attached.loc[0, "source_x"] == 1000.0
    assert attached.loc[0, "source_y"] == -2000.0


def test_inverse_distance_proposal_uses_local_physics_valid_vectors():
    source = np.array([[0.0, 0.0], [1000.0, 0.0], [11000.0, 0.0]])
    target = source + np.array([[100.0, 200.0], [300.0, 400.0], [900.0, 900.0]])

    proposal = inverse_distance_local_proposal(
        source,
        target,
        np.array([0.8, 0.7, 0.9]),
        np.array([0.0, 0.0]),
        elapsed_days=1.0,
        source_radius_m=10000.0,
        maximum_speed_m_per_day=30000.0,
        neighbours=4,
    )

    assert proposal["available"]
    assert proposal["local_match_count"] == 2
    assert proposal["proposal_dx_m"] < 101.0
    assert proposal["proposal_dy_m"] < 201.0


def test_tile_size_and_rotation_wrapping_are_explicit():
    assert tile_pixels(15000.0, 80.0) == 512
    assert tile_pixels(52500.0, 80.0) == 1344
    assert wrapped_angle_difference_degrees(350.0, 10.0) == -20.0


def test_overlapping_target_tiles_cover_edges_and_do_not_duplicate_sources():
    origins = tile_origins(1344, tile_pixels=512, overlap_pixels=64)
    assert origins == [0, 416, 832]

    keep = retain_best_match_per_source(
        np.array([3, 3, 7]), np.array([0.4, 0.9, 0.5])
    )
    assert keep.tolist() == [1, 2]


def test_transform_grid_interpolation_is_exact_for_affine_coordinates():
    rows, columns = np.meshgrid(
        np.linspace(0.0, 8.0, 3), np.linspace(0.0, 8.0, 3), indexing="ij"
    )
    coarse = 10.0 + 2.0 * columns - 3.0 * rows

    interpolated = interpolate_transform_grid(coarse, pixels=9)

    full_rows, full_columns = np.meshgrid(np.arange(9), np.arange(9), indexing="ij")
    expected = 10.0 + 2.0 * full_columns - 3.0 * full_rows
    np.testing.assert_allclose(interpolated, expected)


def test_weighted_geometric_median_resists_one_displacement_outlier():
    vectors = np.array([[100.0, 200.0], [110.0, 190.0], [90.0, 210.0], [9000.0, 0.0]])

    estimate = weighted_geometric_median(vectors, np.ones(4))

    assert np.linalg.norm(estimate - [100.0, 200.0]) < 20.0


def test_local_consensus_rejects_high_confidence_wrong_vector():
    matches = pd.DataFrame(
        {
            "physics_valid": [True] * 4,
            "source_distance_m": [200.0, 400.0, 600.0, 800.0],
            "dx_m": [100.0, 120.0, 80.0, 9000.0],
            "dy_m": [200.0, 180.0, 220.0, 0.0],
            "lightglue_score": [0.6, 0.7, 0.8, 0.99],
        }
    )

    confidence = estimate_policy(matches, "highest_confidence_within_2km")
    consensus = estimate_policy(matches, "consensus_within_2km")

    assert confidence["proposal_dx_m"] == 9000.0
    assert np.linalg.norm(
        [consensus["proposal_dx_m"] - 100.0, consensus["proposal_dy_m"] - 200.0]
    ) < 30.0
    assert consensus["selected_vectors"] == 3


def test_path_summary_keeps_missing_steps_in_the_denominator():
    results = pd.DataFrame(
        {
            "panel": ["representative"] * 4,
            "policy": ["p"] * 4,
            "continuous_trajectory_id": ["a", "a", "b", "b"],
            "source_image_time": [1, 2, 1, 2],
            "available": [True, True, True, False],
            "error_m": [10.0, 20.0, 10.0, np.nan],
        }
    )

    summary = summarize_truth_reinitialized_paths(results)[0]

    assert summary["paths_with_at_least_two_steps"] == 2
    assert summary["complete_paths"] == 1
    assert summary["all_steps_within_2km"] == 1


def test_recenter_matches_uses_estimated_source_state():
    matches = pd.DataFrame(
        {
            "source_x": [0.0, 5000.0],
            "source_y": [0.0, 0.0],
            "speed_m_per_day": [100.0, 100.0],
        }
    )

    recentered = recenter_matches(matches, np.array([5000.0, 0.0]), 2000.0)

    assert recentered["source_distance_m"].tolist() == [5000.0, 0.0]
    assert recentered["physics_valid"].tolist() == [False, True]


def test_propagated_replay_uses_previous_endpoint_as_next_source():
    transitions = pd.DataFrame(
        {
            "transition_id": ["a", "b"],
            "continuous_trajectory_id": ["path", "path"],
            "source_image_id": [1, 2],
            "target_image_id": [2, 3],
            "source_image_time": [1, 2],
            "representative_panel": [True, True],
            "source_x": [0.0, 100.0],
            "source_y": [0.0, 0.0],
            "truth_dx_m": [100.0, 100.0],
            "truth_dy_m": [0.0, 0.0],
        }
    )
    vectors = {
        transition_id: pd.DataFrame(
            {
                "source_x": [source_x],
                "source_y": [0.0],
                "target_x": [source_x + 110.0],
                "target_y": [0.0],
                "dx_m": [110.0],
                "dy_m": [0.0],
                "speed_m_per_day": [110.0],
                "lightglue_score": [0.9],
                "physics_valid": [True],
                "source_distance_m": [0.0],
            }
        )
        for transition_id, source_x in (("a", 0.0), ("b", 110.0))
    }

    results, summary = replay_propagated_paths(
        transitions, vectors, tight_radius_m=2000.0, consensus_radius_m=1000.0
    )

    consensus = results.loc[results["policy"].eq("consensus_within_2km")]
    assert consensus["source_state_error_m"].tolist() == [0.0, 10.0]
    assert consensus["error_m"].tolist() == [10.0, 20.0]
    consensus_summary = next(
        row for row in summary if row["policy"] == "consensus_within_2km"
    )
    assert consensus_summary["complete_paths"] == 1
    assert consensus_summary["median_complete_path_final_error_m"] == 20.0


def test_q4_fallback_keeps_direct_result_when_refinement_is_rejected():
    rows = pd.DataFrame(
        {
            "transition_id": ["a", "a", "b", "b"],
            "panel": ["representative"] * 4,
            "variant": [
                "direct_no_pattern",
                "quadratic_border4_bilinear_template",
                "direct_no_pattern",
                "quadratic_border4_bilinear_template",
            ],
            "accepted": [True, False, True, True],
            "error_m": [50.0, 500.0, 60.0, 20.0],
            "seconds": [0.0, 2.0, 0.0, 3.0],
        }
    )

    augmented = add_q4_direct_fallback(rows)
    fallback = augmented.loc[
        augmented["variant"].eq(
            "quadratic_border4_bilinear_with_direct_fallback"
        )
    ]

    assert fallback["accepted"].tolist() == [True, True]
    assert fallback["error_m"].tolist() == [50.0, 20.0]
    assert fallback["seconds"].tolist() == [2.0, 3.0]
