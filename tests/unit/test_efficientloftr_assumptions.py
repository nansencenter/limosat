import numpy as np
import shapely

from limosat.learned_drift import (
    DriftField,
    EfficientLoFTRConfig,
    MotionMatches,
    estimate_field,
    regular_grid,
    sample_field,
    source_core_mask,
    speed_limit_mask,
    valid_endpoints,
    valid_support,
)


def motion_matches(vectors_m: np.ndarray, scores: np.ndarray) -> MotionMatches:
    count = len(vectors_m)
    source = np.column_stack(
        (np.linspace(-50.0, 50.0, count), np.zeros(count))
    )
    return MotionMatches(
        source_feature_id=np.arange(count, dtype=np.int64),
        source_tile_id=np.zeros(count, dtype=np.int32),
        target_tile_id=np.zeros(count, dtype=np.int32),
        source_xy_m=source,
        target_xy_m=source + vectors_m,
        score=np.asarray(scores, dtype=np.float32),
    )


def constant_field(spacing_m: float) -> DriftField:
    source = np.array(
        [(x * spacing_m, y * spacing_m) for y in range(5) for x in range(5)],
        dtype=float,
    )
    count = len(source)
    return DriftField(
        grid_row=np.repeat(np.arange(5), 5),
        grid_column=np.tile(np.arange(5), 5),
        source_xy_m=source,
        displacement_m=np.tile([100.0, 0.0], (count, 1)),
        available=np.ones(count, dtype=bool),
        selected_matches=np.full(count, 8, dtype=np.int32),
        candidate_matches=np.full(count, 12, dtype=np.int32),
        support_radius_m=np.full(count, 5_000.0),
        maximum_residual_m=np.full(count, 80.0),
    )
def test_efficientloftr_config_contains_only_relevant_matcher_settings():
    config = EfficientLoFTRConfig()

    assert config.model_name == "efficientloftr-official-opt"
    assert config.endpoint_support_radius_px == 16
    assert config.maximum_triangle_edge_m == 6_400.0
    assert not hasattr(config, "features_per_tile")
    assert not hasattr(config, "lightglue_layers")


def test_30_40_50_km_per_day_gates_have_explicit_metre_units():
    source = np.zeros((4, 2), dtype=float)
    target = np.array(
        [[29_000.0, 0.0], [30_000.0, 0.0], [40_000.0, 0.0], [49_000.0, 0.0]]
    )

    accepted_30 = speed_limit_mask(source, target, 24.0, 30_000.0)
    accepted_40 = speed_limit_mask(source, target, 24.0, 40_000.0)
    accepted_50 = speed_limit_mask(source, target, 24.0, 50_000.0)

    assert accepted_30.tolist() == [True, True, False, False]
    assert accepted_40.tolist() == [True, True, True, False]
    assert accepted_50.tolist() == [True, True, True, True]
    assert EfficientLoFTRConfig(
        maximum_speed_m_per_day=50_000.0
    ).maximum_displacement_m(12.0) == 25_000.0


def test_larger_speed_gate_does_not_replace_consensus_filtering():
    true_vectors = np.tile([20_000.0, 0.0], (8, 1))
    false_vectors = np.tile([45_000.0, 12_000.0], (4, 1))
    matches = motion_matches(
        np.vstack((true_vectors, false_vectors)),
        np.ones(12),
    )
    speed_valid = speed_limit_mask(
        matches.source_xy_m,
        matches.target_xy_m,
        elapsed_hours=24.0,
        maximum_speed_m_per_day=50_000.0,
    )

    field = estimate_field(
        matches.select(np.flatnonzero(speed_valid)),
        shapely.box(-1.0, -1.0, 1.0, 1.0),
        EfficientLoFTRConfig(),
    )

    assert speed_valid.all()
    assert field.available.tolist() == [True]
    np.testing.assert_allclose(field.displacement_m[0], [20_000.0, 0.0])
    assert field.selected_matches[0] == 8


def test_raw_matcher_scores_can_choose_a_smaller_wrong_cluster():
    true_vectors = np.tile([1_000.0, 0.0], (7, 1))
    false_vectors = np.tile([40_000.0, 0.0], (5, 1))
    matches = motion_matches(
        np.vstack((true_vectors, false_vectors)),
        np.r_[np.full(7, 0.1), np.full(5, 0.9)],
    )
    domain = shapely.box(-1.0, -1.0, 1.0, 1.0)
    common = dict(minimum_agreeing_matches=5)

    raw = estimate_field(
        matches,
        domain,
        EfficientLoFTRConfig(score_weighting="raw", **common),
    )
    uniform = estimate_field(
        matches,
        domain,
        EfficientLoFTRConfig(score_weighting="uniform", **common),
    )

    np.testing.assert_allclose(raw.displacement_m[0], [40_000.0, 0.0])
    np.testing.assert_allclose(uniform.displacement_m[0], [1_000.0, 0.0])
    assert raw.selected_matches[0] == 5
    assert uniform.selected_matches[0] == 7


def test_endpoint_erosion_and_rounding_are_explicit():
    valid = np.ones((9, 9), dtype=bool)
    eroded = valid_support(valid, radius_px=2)
    endpoints = np.array([[2.0, 2.0], [1.49, 4.0], [6.0, 6.0], [9.0, 4.0]])

    assert eroded.sum() == 25
    assert valid_endpoints(endpoints, eroded).tolist() == [True, False, True, False]
    np.testing.assert_array_equal(valid_support(valid, radius_px=0), valid)


def test_tile_core_margin_is_independent_of_endpoint_erosion():
    config = EfficientLoFTRConfig(
        tile_margin_px=32,
        endpoint_support_radius_px=8,
    )
    points = np.array(
        [[31.99, 200.0], [32.0, 32.0], [479.99, 479.99], [480.0, 200.0]]
    )

    assert config.tile_core_size_m == 35_840.0
    assert source_core_mask(
        points, config.tile_size_px, config.tile_margin_px
    ).tolist() == [False, True, True, False]


def test_five_kilometre_output_has_fewer_nodes_but_not_free_interpolation():
    domain = shapely.box(0.0, 0.0, 100_000.0, 100_000.0)
    nodes_4km = len(regular_grid(domain, 4_000.0)[2])
    nodes_5km = len(regular_grid(domain, 5_000.0)[2])
    field_5km = constant_field(5_000.0)
    query = np.array([[2_500.0, 2_500.0]])

    fixed_6_4km = sample_field(field_5km, query, maximum_triangle_edge_m=6_400.0)
    explicit_8km = sample_field(field_5km, query, maximum_triangle_edge_m=8_000.0)

    assert nodes_5km / nodes_4km == 0.64
    assert not fixed_6_4km.available[0]
    assert explicit_8km.available[0]
