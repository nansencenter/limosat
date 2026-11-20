from dataclasses import replace

import numpy as np
import pytest

from limosat.config import FieldConfig, RunConfig
from limosat.field import (
    estimate_points,
    gap_matches,
    node_triangle_spread,
    sample_field,
    sample_field_with_fallback,
)
from test_gap_fill import GAP_FILL, _field, _matches

SHEAR = dict(GAP_FILL, shear_point_estimates=True)


def _truth(query, shear_x=18_000.0):
    return np.where((query[:, 0] < shear_x)[:, None], [0.0, 3_000.0], [1_000.0, 0.0])


def test_shear_point_estimates_follow_each_side_of_a_shear_line():
    config = FieldConfig(**SHEAR)
    # The shear line runs half way between the node columns at 16 and 20 km, so every
    # triangle between those columns spans it.
    matches = _matches(seed=7, count=40_000, shear_x=18_000.0)
    field = _field(matches, config)
    rng = np.random.default_rng(8)
    query = np.column_stack((rng.uniform(16_500.0, 19_500.0, 600), rng.uniform(4_000.0, 36_000.0, 600)))
    query = query[np.abs(query[:, 0] - 18_000.0) > 400.0]
    plain = sample_field_with_fallback(field, None, query, config.maximum_triangle_edge_m,
                                       field_config=FieldConfig(**GAP_FILL), gap_matches=gap_matches(field, matches, config))
    refined = sample_field_with_fallback(field, None, query, config.maximum_triangle_edge_m,
                                         field_config=config, gap_matches=gap_matches(field, matches, config))
    both = plain.available & refined.available
    plain_error = np.hypot(*(plain.displacement_m[both] - _truth(query[both])).T)
    refined_error = np.hypot(*(refined.displacement_m[both] - _truth(query[both])).T)
    assert np.median(plain_error) > 500.0          # interpolation blends the two sides
    assert np.quantile(refined_error, 0.95) < 100.0  # point estimates keep each side


def test_shear_point_estimates_leave_coherent_ice_interpolated():
    config = FieldConfig(**SHEAR)
    matches = _matches(seed=9, count=20_000, shear_x=20_000.0)
    field = _field(matches, config)
    query = np.random.default_rng(10).uniform(28_000.0, 38_000.0, size=(500, 2))
    kept = gap_matches(field, matches, config)
    plain = sample_field(field, query, config.maximum_triangle_edge_m)
    refined = sample_field_with_fallback(field, None, query, config.maximum_triangle_edge_m,
                                         field_config=config, gap_matches=kept)
    # The rule: a sample changes only where its nodes disagree more than its own matches.
    spread = node_triangle_spread(field, query)
    estimate = estimate_points(kept, query, 12, 8, 3_000.0, 500.0)
    switch = estimate["available"] & (spread > estimate["maximum_residual_m"])
    np.testing.assert_array_equal(refined.displacement_m[~switch], plain.displacement_m[~switch])
    np.testing.assert_allclose(refined.displacement_m[switch], estimate["displacement_m"][switch])
    # In uniform ice a switch only swaps one good estimate for another.
    assert np.abs(refined.displacement_m - [1_000.0, 0.0]).max() < 60.0


def test_shear_point_estimates_keep_every_match_without_tiles():
    config = FieldConfig(**SHEAR)
    matches = _matches(seed=11, count=5_000, shear_x=20_000.0)
    kept = gap_matches(_field(matches, config), matches, config)
    assert len(kept) == len(matches)
    np.testing.assert_array_equal(kept.displacement_m, matches.displacement_m)
    assert kept.source_tile.strides == (0,)


def test_shear_point_estimates_configuration():
    base = RunConfig("run", "catalogue", "database", "output", field=FieldConfig(**GAP_FILL))
    assert base.field.shear_point_estimates is False
    shear = replace(base, field=FieldConfig(**SHEAR))
    assert shear.sha256 != base.sha256
    assert shear.to_dict()["field"]["shear_point_estimates"] is True
    with pytest.raises(ValueError, match="retain_pair_matches"):
        replace(shear, retain_pair_matches=False)
    with pytest.raises(ValueError, match="gap_fill_point"):
        FieldConfig(shear_point_estimates=True)


def test_points_in_field_holes_use_their_own_matches_before_a_rigid_bridge():
    config = FieldConfig(**SHEAR)
    matches = _matches(seed=12, count=20_000, shear_x=18_000.0)
    field = _field(matches, config)
    hole = np.hypot(*(field.source_xy_m - [18_000.0, 20_000.0]).T) < 3_000.0
    holed = field.with_available(field.available & ~hole)
    query = np.array([[17_300.0, 20_000.0], [18_700.0, 20_500.0]])
    assert not sample_field(holed, query, config.maximum_triangle_edge_m).available.any()
    kept = gap_matches(holed, matches, config)
    sampled = sample_field_with_fallback(holed, None, query, config.maximum_triangle_edge_m,
                                         field_config=config, gap_matches=kept)
    own = estimate_points(kept, query, 12, 8, 3_000.0, 500.0)
    assert own["available"].all()
    np.testing.assert_allclose(sampled.displacement_m, own["displacement_m"])
    np.testing.assert_allclose(sampled.displacement_m, _truth(query), atol=100.0)
