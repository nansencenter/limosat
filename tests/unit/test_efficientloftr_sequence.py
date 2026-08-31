import json

import numpy as np
import pandas as pd
import shapely

from experiments.run_efficientloftr_sequence import (
    PairSpec,
    field_from_csv,
    field_sha256,
    pair_identity,
    regions_near_points,
    routing_recovery_diagnostic,
    save_trajectory_products,
)
from experiments.run_efficientloftr_targeted_recovery import recovery_positions
from experiments.analyze_efficientloftr_targeted_recovery import evaluate_gate
from limosat.learned_drift import (
    DriftField,
    EfficientLoFTRConfig,
    coarse_phase_translation,
    preceding_field_shifts,
)
from limosat.learned_drift.features import TileRegion
from limosat.learned_drift.field import (
    estimate_field,
    estimate_queries,
    reject_folds,
    topology_summary,
)
from limosat.learned_drift.types import MotionMatches


def varying_field() -> DriftField:
    source = np.array(
        [(x * 4000.0, y * 4000.0) for y in range(5) for x in range(5)]
    )
    displacement = np.column_stack(
        (500.0 + 0.02 * source[:, 0], -200.0 + 0.01 * source[:, 1])
    )
    return DriftField(
        grid_row=np.repeat(np.arange(5), 5),
        grid_column=np.tile(np.arange(5), 5),
        source_xy_m=source,
        displacement_m=displacement,
        available=np.ones(len(source), dtype=bool),
        selected_matches=np.full(len(source), 8),
        candidate_matches=np.full(len(source), 12),
        support_radius_m=np.full(len(source), 5000.0),
        maximum_residual_m=np.full(len(source), 80.0),
    )


def empty_field() -> DriftField:
    field = varying_field()
    return DriftField(
        grid_row=field.grid_row,
        grid_column=field.grid_column,
        source_xy_m=field.source_xy_m,
        displacement_m=np.full_like(field.displacement_m, np.nan),
        available=np.zeros(len(field), dtype=bool),
        selected_matches=np.zeros(len(field)),
        candidate_matches=np.zeros(len(field)),
        support_radius_m=np.full(len(field), np.nan),
        maximum_residual_m=np.full(len(field), np.nan),
    )


def test_collinear_sparse_field_has_no_deformation_triangles():
    source = np.array([[0.0, 0.0], [4000.0, 0.0], [8000.0, 0.0]])
    field = DriftField(
        grid_row=np.zeros(3, dtype=int),
        grid_column=np.arange(3),
        source_xy_m=source,
        displacement_m=np.tile([100.0, -50.0], (3, 1)),
        available=np.ones(3, dtype=bool),
        selected_matches=np.full(3, 8),
        candidate_matches=np.full(3, 12),
        support_radius_m=np.full(3, 4000.0),
        maximum_residual_m=np.full(3, 100.0),
    )

    accepted, rejected = reject_folds(field, 4000.0)

    assert accepted.available.all()
    assert rejected.size == 0
    assert topology_summary(accepted, 4000.0) == {"triangles": 0}


def test_supplied_point_consensus_matches_regular_field_estimation():
    source = np.array(
        [[x, y] for y in (0.0, 1000.0, 2000.0) for x in (0.0, 1000.0, 2000.0)]
    )
    displacement = np.tile([500.0, -200.0], (len(source), 1))
    matches = MotionMatches(
        source_feature_id=np.arange(len(source)),
        source_tile_id=np.zeros(len(source), dtype=np.int32),
        target_tile_id=np.zeros(len(source), dtype=np.int32),
        source_xy_m=source,
        target_xy_m=source + displacement,
        score=np.ones(len(source), dtype=np.float32),
    )
    config = EfficientLoFTRConfig(
        grid_spacing_m=1000.0,
        neighbour_count=9,
        minimum_agreeing_matches=8,
        maximum_neighbour_distance_m=3000.0,
    )
    field = estimate_field(matches, shapely.box(-1.0, -1.0, 2001.0, 2001.0), config)

    estimates = estimate_queries(matches, field.source_xy_m, config)

    np.testing.assert_array_equal(estimates["available"], field.available)
    np.testing.assert_array_equal(
        estimates["selected_matches"], field.selected_matches
    )
    np.testing.assert_allclose(estimates["displacement_m"], field.displacement_m)


def test_local_routing_samples_velocity_in_current_image_coordinates():
    field = varying_field()
    selected = np.array([6, 7, 11, 12])
    current_centers = (
        field.source_xy_m[selected] + field.displacement_m[selected]
    )

    shifts, sources = preceding_field_shifts(
        current_centers,
        "sequential_local",
        field,
        previous_elapsed_days=1.0,
        current_elapsed_days=2.0,
        minimum_nodes=EfficientLoFTRConfig().minimum_agreeing_matches,
        grid_spacing_m=EfficientLoFTRConfig().grid_spacing_m,
    )

    np.testing.assert_allclose(shifts, field.displacement_m[selected] * 2.0)
    assert (sources == "preceding_local_velocity").all()


def test_local_routing_uses_global_velocity_only_outside_supported_field():
    field = varying_field()
    centers = np.array([[8500.0, 8500.0], [100_000.0, 100_000.0]])

    shifts, sources = preceding_field_shifts(
        centers,
        "sequential_local",
        field,
        previous_elapsed_days=1.0,
        current_elapsed_days=1.0,
        minimum_nodes=EfficientLoFTRConfig().minimum_agreeing_matches,
        grid_spacing_m=EfficientLoFTRConfig().grid_spacing_m,
    )

    assert sources.tolist() == [
        "preceding_local_velocity",
        "preceding_global_velocity",
    ]
    np.testing.assert_allclose(
        shifts[1], np.median(field.displacement_m, axis=0)
    )


def test_first_pair_has_explicit_same_center_fallback():
    shifts, sources = preceding_field_shifts(
        np.array([[0.0, 0.0], [4000.0, 0.0]]),
        "sequential_local",
        None,
        None,
        1.0,
        EfficientLoFTRConfig().minimum_agreeing_matches,
        EfficientLoFTRConfig().grid_spacing_m,
    )

    np.testing.assert_array_equal(shifts, np.zeros((2, 2)))
    assert (sources == "same_center_fallback").all()


def test_coarse_phase_translation_has_projected_y_sign(monkeypatch):
    generated = {}

    def synthetic_patch(path, _center, pixels, *_args):
        if not generated:
            rng = np.random.default_rng(42)
            source = rng.normal(size=(pixels, pixels)).astype(np.float32)
            generated["source"] = source
            generated["target"] = np.roll(source, shift=(5, 7), axis=(0, 1))
        return generated[path].copy(), np.ones((pixels, pixels), dtype=bool)

    monkeypatch.setattr(
        "limosat.learned_drift.routing.north_up_patch", synthetic_patch
    )
    result = coarse_phase_translation(
        "source",
        "target",
        shapely.box(0.0, 0.0, 96.0, 96.0),
        maximum_displacement_m=20.0,
        analysis_epsg=3413,
        transform_grid_spacing_px=32,
        preferred_pixel_size_m=1.0,
        maximum_pixels=256,
    )

    np.testing.assert_allclose(result.displacement_m, [7.0, -5.0], atol=0.1)
    assert result.response > 0.9


def test_routing_recovery_requires_large_residual_and_aligned_edge_pressure():
    config = EfficientLoFTRConfig()
    source = np.column_stack((np.arange(20) * 100.0, np.zeros(20)))
    target = source + np.array([-3000.0, 1500.0])
    target_px = np.column_stack(
        (np.full(20, 30.0), np.full(20, 250.0))
    )

    result = routing_recovery_diagnostic(
        source, target, target_px, np.zeros(2), config
    )

    assert result["triggered"]
    assert result["aligned_axes"] == ["x_negative"]
    assert result["usable_routing_slack_m"] == 1280.0
    assert result["median_residual_dx_m"] == -3000.0
    assert result["median_residual_dy_m"] == 1500.0


def test_routing_recovery_does_not_trigger_on_residual_without_edge_pressure():
    config = EfficientLoFTRConfig()
    source = np.column_stack((np.arange(20) * 100.0, np.zeros(20)))
    target = source + np.array([-3000.0, 0.0])
    target_px = np.full((20, 2), 255.5)

    result = routing_recovery_diagnostic(
        source, target, target_px, np.zeros(2), config
    )

    assert result["eligible"]
    assert not result["triggered"]


def test_larger_context_increases_recovery_slack_without_changing_core():
    baseline = EfficientLoFTRConfig(tile_size_px=512, tile_margin_px=32)
    larger = EfficientLoFTRConfig(tile_size_px=576, tile_margin_px=64)

    assert baseline.tile_core_size_m == larger.tile_core_size_m == 35_840.0
    source = np.column_stack((np.arange(20) * 100.0, np.zeros(20)))
    target = source + np.array([-3000.0, 0.0])
    target_px = np.column_stack(
        (np.full(20, 30.0), np.full(20, 250.0))
    )
    baseline_result = routing_recovery_diagnostic(
        source, target, target_px, np.zeros(2), baseline
    )
    larger_result = routing_recovery_diagnostic(
        source, target, target_px, np.zeros(2), larger
    )

    assert baseline_result["triggered"]
    assert not larger_result["triggered"]
    assert larger_result["usable_routing_slack_m"] == 3840.0


def test_field_hash_survives_csv_resume_round_trip(tmp_path):
    field = varying_field()
    displacement = field.displacement_m.copy()
    displacement[0] = np.nan
    field = DriftField(
        field.grid_row,
        field.grid_column,
        field.source_xy_m,
        displacement,
        np.arange(len(field)) != 0,
        field.selected_matches,
        field.candidate_matches,
        field.support_radius_m,
        np.where(np.arange(len(field)) == 0, np.nan, field.maximum_residual_m),
    )
    path = tmp_path / "field.csv"
    field.to_frame().to_csv(path, index=False)

    assert field_sha256(field_from_csv(path)) == field_sha256(field)


def test_persisted_field_hash_is_stable_at_rounding_boundary(tmp_path):
    field = varying_field()
    field.displacement_m[0, 0] = 500.00000050000006
    path = tmp_path / "field.csv"
    field.to_frame().to_csv(path, index=False)
    persisted = field_from_csv(path)

    assert field_sha256(field_from_csv(path)) == field_sha256(persisted)


def test_sequence_outputs_preserve_strict_and_add_new_point_graph(tmp_path):
    specs = [
        PairSpec(1, 2, "a.tif", "b.tif", 24.0, None),
        PairSpec(2, 3, "b.tif", "c.tif", 24.0, None),
    ]

    summary = save_trajectory_products(
        specs,
        [varying_field(), varying_field()],
        tmp_path,
        EfficientLoFTRConfig(),
    )

    assert (tmp_path / "trajectories_4km.csv").exists()
    assert (tmp_path / "trajectories_with_new_points_adjacent_graph.csv").exists()
    points_graph = summary["adjacent_observed_graph_with_new_points"]
    assert points_graph["trajectory_count"] >= points_graph["initial_trajectories"]
    assert points_graph["new_point_exclusion_radius_m"] == 2000.0


def test_sequence_with_no_supported_trajectories_is_valid_missing_data(tmp_path):
    specs = [
        PairSpec(1, 2, "a.tif", "b.tif", 24.0, None),
        PairSpec(2, 3, "b.tif", "c.tif", 24.0, None),
    ]

    summary = save_trajectory_products(
        specs,
        [empty_field(), empty_field()],
        tmp_path,
        EfficientLoFTRConfig(),
    )

    assert summary["seeded"] == 0
    assert summary["complete"] == 0
    assert summary["complete_fraction"] is None
    assert summary["active_by_image"] == [0, 0, 0]
    assert summary["active_fraction_by_image"] == [None, None, None]
    assert summary["adjacent_observed_graph_with_new_points"]["trajectory_count"] == 0
    assert summary["adjacent_observed_graph_with_new_points"][
        "active_fraction_by_image"
    ] == [None, None, None]
    assert summary["gap_aware_96h"]["complete"] == 0
    assert summary["gap_aware_96h"]["complete_fraction"] is None
    assert summary["gap_aware_96h"]["active_fraction_by_image"] == [None, None, None]
    json.dumps(summary, allow_nan=False)
    survival = pd.read_csv(tmp_path / "trajectory_survival.csv")
    assert survival[["sum", "count"]].to_numpy().tolist() == [[0, 0], [0, 0], [0, 0]]


def test_single_pair_with_no_supported_trajectories_is_valid(tmp_path):
    summary = save_trajectory_products(
        [PairSpec(1, 2, "a.tif", "b.tif", 24.0, None)],
        [empty_field()],
        tmp_path,
        EfficientLoFTRConfig(),
    )

    assert summary["seeded"] == 0
    assert summary["active_by_image"] == [0, 0]
    assert summary["active_fraction_by_image"] == [None, None]


def test_new_point_graph_can_start_after_an_empty_first_pair(tmp_path):
    specs = [
        PairSpec(1, 2, "a.tif", "b.tif", 24.0, None),
        PairSpec(2, 3, "b.tif", "c.tif", 24.0, None),
    ]

    summary = save_trajectory_products(
        specs,
        [empty_field(), varying_field()],
        tmp_path,
        EfficientLoFTRConfig(),
    )

    assert summary["seeded"] == 0
    graph = summary["adjacent_observed_graph_with_new_points"]
    assert graph["new_trajectories"] > 0
    assert graph["final_active"] > 0
    assert graph["active_fraction_by_image"][0] is None


def test_targeted_regions_keep_only_tile_cores_near_dormant_positions():
    regions = (
        TileRegion(0, 0, 0, (5.0, 5.0), shapely.box(0.0, 0.0, 10.0, 10.0)),
        TileRegion(1, 0, 1, (15.0, 5.0), shapely.box(10.0, 0.0, 20.0, 10.0)),
        TileRegion(2, 0, 2, (35.0, 5.0), shapely.box(30.0, 0.0, 40.0, 10.0)),
    )

    selected = regions_near_points(regions, np.array([[8.0, 5.0]]), 3.0)

    assert [region.tile_id for region in selected] == [0, 1]


def test_targeted_positions_change_pair_identity_without_changing_default_identity():
    spec = PairSpec(1, 3, "a.tif", "c.tif", 48.0, None)
    arguments = (
        spec,
        EfficientLoFTRConfig(),
        "sequential_local",
        "phase_correlation",
        (100.0, -50.0),
        "checkpoint",
        None,
        None,
    )

    default = pair_identity(*arguments)
    explicitly_default = pair_identity(
        *arguments,
        source_selection_xy_m=None,
        source_selection_buffer_m=None,
    )
    targeted = pair_identity(
        *arguments,
        source_selection_xy_m=np.array([[0.0, 0.0], [4000.0, 0.0]]),
        source_selection_buffer_m=6400.0,
    )

    assert default == explicitly_default
    assert targeted != default


def test_targeted_recovery_selects_positions_active_at_source_but_lost_by_target():
    trajectories = pd.DataFrame(
        {
            "trajectory_id": [0, 1, 0, 1, 0, 1],
            "image_index": [0, 0, 1, 1, 2, 2],
            "x_m": [0.0, 4000.0, np.nan, 4100.0, 200.0, np.nan],
            "y_m": [0.0, 0.0, np.nan, 0.0, 0.0, np.nan],
            "active": [True, True, False, True, True, False],
        }
    )

    selected = recovery_positions(
        trajectories, ["a", "b", "c"], "a", "c"
    )

    assert selected.trajectory_id.tolist() == [1]
    np.testing.assert_allclose(selected[["x_m", "y_m"]], [[4000.0, 0.0]])


def test_targeted_gate_uses_the_output_trajectory_graph_for_deformation():
    pair_reports = [
        {"targeted_topology": {"flipped_triangles": 0}}
    ]
    totals = {"matcher_call_reduction_fraction": 0.6}
    graph = {
        "adjacent_only_graph": {"complete": 100},
        "shortest_graph": {"complete": 190},
        "buoys_unsealed": {
            "shortest": {
                "available": 4,
                "correct_within_2km": 4,
                "median_error_m": 100.0,
                "p90_error_m": 200.0,
            }
        },
    }
    targeted_graph = {
        **graph,
        "shortest_graph": {"complete": 185},
    }
    graph_comparison = {
        "by_image": {
            "c": {
                "position_difference_m": {"median": 5.0, "p90": 20.0}
            }
        },
        "deformation_by_image": {"c": {"spearman_total_per_day": 0.9}},
    }
    gate = {
        "trajectory_gain_recovery_fraction_min": 0.9,
        "matcher_call_reduction_fraction_min": 0.5,
        "buoy_availability_loss_max": 0,
        "buoy_correct_within_2km_loss_max": 0,
        "buoy_median_error_increase_m_max": 1.0,
        "buoy_p90_error_increase_m_max": 1.0,
        "flipped_triangles_max": 0,
        "trajectory_position_median_difference_m_max": 80.0,
        "trajectory_position_p90_difference_m_max": 250.0,
        "trajectory_total_deformation_spearman_min": 0.8,
    }

    _metrics, checks = evaluate_gate(
        pair_reports,
        totals,
        targeted_graph,
        graph,
        graph_comparison,
        gate,
    )

    assert all(checks.values())
