from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from limosat.config import FieldConfig, RunConfig
from limosat.field import (
    estimate_points,
    estimate_queries,
    gap_matches,
    sample_field,
    sample_field_with_fallback,
)
from limosat.models import DisplacementField, MotionMatches

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
GAP_FILL = dict(
    gap_fill_rigid_edge_m=12_800.0, gap_fill_rigid_spread_m=300.0,
    gap_fill_point_neighbour_count=12, gap_fill_point_minimum_agreeing_matches=8,
    gap_fill_point_neighbour_distance_m=3_000.0, gap_fill_point_agreement_distance_m=500.0,
)


def _matches(seed=0, count=6_000, extent=40_000.0, shear_x=None):
    """Random matches moving 1 km east; west of ``shear_x`` they move 3 km north."""
    rng = np.random.default_rng(seed)
    source = rng.uniform(0.0, extent, size=(count, 2))
    motion = np.tile([1_000.0, 0.0], (count, 1))
    if shear_x is not None:
        motion[source[:, 0] < shear_x] = [0.0, 3_000.0]
    motion += rng.normal(0.0, 30.0, size=motion.shape)
    return MotionMatches(
        source, source + motion, rng.uniform(0.2, 1.0, count),
        np.zeros(count, dtype=int), np.zeros(count, dtype=int),
    )


def _field(matches, config, spacing=4_000.0, extent=40_000.0):
    axis = np.arange(0.0, extent + 1.0, spacing)
    xx, yy = np.meshgrid(axis, axis)
    query = np.column_stack((xx.ravel(), yy.ravel()))
    rows, columns = np.meshgrid(np.arange(len(axis)), np.arange(len(axis)), indexing="ij")
    estimates = estimate_queries(matches, query, config)
    return DisplacementField(
        pair_id="a__b", source_image_id="a", target_image_id="b",
        source_time_utc=START, target_time_utc=START + timedelta(days=1),
        grid_row=rows.ravel(), grid_column=columns.ravel(), source_xy_m=query, **estimates,
    )


def test_point_estimates_match_the_node_rule():
    config = FieldConfig()
    matches = _matches(seed=1, shear_x=20_000.0)
    query = np.random.default_rng(2).uniform(0.0, 40_000.0, size=(400, 2))
    looped = estimate_queries(matches, query, config)
    batched = estimate_points(
        matches, query, config.neighbour_count, config.minimum_agreeing_matches,
        config.maximum_neighbour_distance_m, config.agreement_distance_m,
    )
    np.testing.assert_array_equal(batched["available"], looped["available"])
    np.testing.assert_array_equal(batched["selected_matches"], looped["selected_matches"])
    both = looped["available"]
    np.testing.assert_allclose(batched["displacement_m"][both], looped["displacement_m"][both], atol=0.01)
    np.testing.assert_allclose(batched["maximum_residual_m"][both], looped["maximum_residual_m"][both], atol=0.01)


def test_rigid_triangles_bridge_holes_in_uniform_ice_only():
    config = FieldConfig(**GAP_FILL)
    uniform = _field(_matches(seed=3), config)
    holed = uniform.with_available(uniform.available & (np.hypot(*(uniform.source_xy_m - 20_000.0).T) > 2_000.0))
    query = np.array([[20_000.0, 20_000.0], [21_000.0, 19_000.0]])
    assert not sample_field(holed, query, config.maximum_triangle_edge_m).available.any()
    filled = sample_field(holed, query, config.maximum_triangle_edge_m,
                          rigid=(config.gap_fill_rigid_edge_m, config.gap_fill_rigid_spread_m))
    assert filled.available.all()
    np.testing.assert_allclose(filled.displacement_m, [[1_000.0, 0.0]] * 2, atol=50.0)

    # Normally usable triangles are never resampled as rigid ones.
    ordinary = np.array([[6_000.0, 6_000.0]])
    assert not sample_field(holed, ordinary, config.maximum_triangle_edge_m,
                            rigid=(12_800.0, 300.0)).available.any()


def test_gap_filling_keeps_existing_samples_and_fills_a_shear_zone():
    config = FieldConfig(**GAP_FILL)
    matches = _matches(seed=4, count=20_000, shear_x=20_000.0)
    field = _field(matches, config)
    query = np.column_stack((np.full(9, 19_400.0), np.linspace(8_000.0, 32_000.0, 9)))
    base = sample_field(field, query, config.maximum_triangle_edge_m)
    filled = sample_field_with_fallback(
        field, None, query, config.maximum_triangle_edge_m,
        field_config=config, gap_matches=gap_matches(field, matches, config),
    )
    np.testing.assert_array_equal(filled.displacement_m[base.available], base.displacement_m[base.available])
    assert filled.available.sum() > base.available.sum()
    new = filled.available & ~base.available
    # Points 600 m west of the shear move north with their side of it.
    np.testing.assert_allclose(filled.displacement_m[new], [[0.0, 3_000.0]] * new.sum(), atol=60.0)


def test_kept_gap_matches_give_the_same_point_estimates():
    config = FieldConfig(**GAP_FILL)
    matches = _matches(seed=5, count=20_000, shear_x=20_000.0)
    field = _field(matches, config)
    kept = gap_matches(field, matches, config)
    assert 0 < len(kept) < len(matches)
    query = np.random.default_rng(6).uniform(-2_000.0, 42_000.0, size=(3_000, 2))
    gaps = ~sample_field(field, query, config.maximum_triangle_edge_m).available
    arguments = (12, 8, 3_000.0, 500.0)
    full = estimate_points(matches, query[gaps], *arguments)
    subset = estimate_points(kept, query[gaps], *arguments)
    np.testing.assert_array_equal(subset["available"], full["available"])
    np.testing.assert_allclose(subset["displacement_m"][full["available"]],
                               full["displacement_m"][full["available"]], atol=1.0e-9)


def test_gap_filling_configuration_identity_and_requirements(tmp_path):
    base = RunConfig("run", "catalogue", "database", "output")
    assert not base.field.gap_filling
    filled = replace(base, field=FieldConfig(**GAP_FILL))
    assert filled.sha256 != base.sha256
    assert filled.to_dict()["field"]["gap_fill_rigid_edge_m"] == 12_800.0
    with pytest.raises(ValueError, match="retain_pair_matches"):
        replace(filled, retain_pair_matches=False)
    with pytest.raises(ValueError, match="exceed maximum_triangle_edge_m"):
        FieldConfig(gap_fill_rigid_edge_m=6_000.0)
