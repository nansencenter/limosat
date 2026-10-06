from dataclasses import replace

import numpy as np
import pytest
from shapely import box

from limosat import MatcherConfig, MotionMatches
from limosat.imagery import ProjectedPatchCache, north_up_patch
from limosat.pairs import (
    PairProcessor,
    TileRegion,
    _PreparedTile,
    _deduplicate,
    _masked_tile,
    _merge_primary_first,
    tile_layout,
)
from test_coarse_routing import BatchTranslationMatcher
from test_efficientloftr_fields import TranslationMatcher, _config, _pair, _write_image

DIAGONAL = ((0.0, 0.0), (0.5, 0.5))


def _layered(config, **matcher):
    return replace(config, matcher=replace(config.matcher, **matcher))


def test_default_layout_options_keep_configuration_identity(tmp_path):
    config = _config(tmp_path)
    assert not config.matcher.layered_layout
    assert not {
        "tile_grid_offsets", "masked_rematch_target_valid_fraction", "layout_dedup_m",
        "layout_field_merge", "tile_sampling",
    } & config.to_dict()["matcher"].keys()
    layered = _layered(config, tile_grid_offsets=[[0, 0], [0.5, 0.5]])
    assert layered.matcher.tile_grid_offsets == DIAGONAL
    assert layered.matcher.layered_layout
    assert layered.to_dict()["matcher"]["tile_grid_offsets"] == DIAGONAL
    assert layered.sha256 != config.sha256


@pytest.mark.parametrize(
    "values",
    [
        {"tile_grid_offsets": ()},
        {"tile_grid_offsets": ((0.0, 0.0), (1.0, 0.0))},
        {"tile_grid_offsets": ((0.0, 0.0), (0.0, 0.0))},
        {"masked_rematch_target_valid_fraction": 1.5},
        {"layout_field_merge": "average"},
        {"tile_sampling": "mosaic"},
    ],
)
def test_invalid_layout_options_are_rejected(values):
    with pytest.raises(ValueError):
        MatcherConfig(**values)


@pytest.mark.parametrize("tile_size_px", [16, 24])
def test_offset_grid_adds_half_core_shifted_tiles_after_primary(tmp_path, tile_size_px):
    config = _layered(_config(tmp_path), tile_size_px=tile_size_px)
    domain = box(-700.0, -700.0, 700.0, 700.0)
    primary = tile_layout(domain, config)
    layered = tile_layout(domain, _layered(config, tile_grid_offsets=DIAGONAL))
    core = config.matcher.tile_core_size_m
    assert layered[: len(primary)] == primary
    secondary = [region for region in layered if region.grid == 1]
    assert secondary and len(primary) + len(secondary) == len(layered)
    assert [region.tile_id for region in layered] == list(range(len(layered)))
    for region in secondary:
        offset = (np.asarray(region.center_xy_m) - core / 2) % core
        np.testing.assert_allclose(offset, [core / 2, core / 2])


def _matches(source, displacement, score):
    source = np.asarray(source, dtype=np.float64)
    return MotionMatches(
        source, source + np.asarray(displacement, dtype=np.float64), np.asarray(score, float),
        np.arange(len(source), dtype=np.int32), np.arange(len(source), dtype=np.int32),
    )


def test_deduplication_keeps_best_colocated_match_and_reports_agreement():
    matches = _matches(
        [[10, 10], [20, 30], [500, 500], [90, 90]],
        [[100, 0], [130, 40], [0, 0], [1600, 0]],
        [0.5, 0.9, 0.7, 0.1],
    )
    kept, counts = _deduplicate(matches, 160.0)
    np.testing.assert_allclose(kept.source_xy_m, [[20, 30], [500, 500]])
    assert counts["layout_dedup_removed"] == 2
    assert counts["layout_colocated_bins"] == 1
    assert counts["layout_colocated_difference_gt_1km_share"] == 0.5


def test_masked_tile_shows_source_only_where_the_target_is_valid():
    target_valid = np.zeros((16, 16), dtype=bool)
    target_valid[:, :6] = True
    tile = _PreparedTile(
        TileRegion(0, 0, 0, (0.0, 0.0), box(0, 0, 1, 1)), np.zeros(2),
        np.full((16, 16), 200, np.uint8), np.full((16, 16), 50, np.uint8),
        np.ones((16, 16), bool), target_valid,
    )
    masked = _masked_tile(tile, support_radius_px=2)
    assert masked.masked and not tile.masked
    assert (masked.source[:, :8] == 200).all() and (masked.source[:, 8:] == 0).all()
    assert (masked.target[:, 6:] == 0).all() and (masked.target[:, :6] == 50).all()


def test_primary_first_merge_never_drops_primary_nodes(tmp_path):
    field = PairProcessor(_config(tmp_path), TranslationMatcher()).process(_pair(tmp_path)).field
    assert field.available.sum() > 4
    primary_available = field.available.copy()
    primary_available[np.flatnonzero(field.available)[:3]] = False
    primary = field.with_available(primary_available)
    combined_displacement = field.displacement_m.copy()
    combined_displacement[~primary_available] = [0.0, 5_000.0]  # crosses the row above: folds
    values = dict(field.__dict__)
    values["displacement_m"] = combined_displacement
    combined = type(field)(**values)
    merged, rejected = _merge_primary_first(primary, combined, 400.0)
    assert merged.available[primary_available].all()
    np.testing.assert_array_equal(
        merged.displacement_m[primary_available], field.displacement_m[primary_available]
    )
    assert len(rejected) and not primary_available[rejected].any()


def test_pair_cache_crops_equal_direct_patches(tmp_path):
    image = tmp_path / "image.tif"
    _write_image(image)
    cache = ProjectedPatchCache(transform_grid_spacing_px=2, block_px=8)
    for center in [(0.0, 0.0), (300.0, -200.0), (-1_100.0, 900.0)]:
        cropped = cache.patch(image, center, 8, 100.0)
        direct = north_up_patch(image, center, 8, 100.0, transform_grid_spacing_px=2)
        np.testing.assert_array_equal(cropped[1], direct[1])
        np.testing.assert_array_equal(cropped[0], direct[0])
    off_lattice = cache.patch(image, (50.0, 0.0), 8, 100.0)
    np.testing.assert_array_equal(
        off_lattice[0], north_up_patch(image, (50.0, 0.0), 8, 100.0, transform_grid_spacing_px=2)[0]
    )


def test_secondary_grid_shifts_come_from_primary_coarse_matches(tmp_path):
    config = _config(tmp_path)
    config = replace(
        config,
        matcher=replace(config.matcher, pixel_size_m=50.0),
        routing=replace(config.routing, coarse_matching=True, coarse_pixel_size_m=100,
                        coarse_support_radius_m=2000, coarse_minimum_matches=4),
    )
    core = config.matcher.tile_core_size_m
    regions = (
        TileRegion(0, 0, 0, (core / 2, core / 2), box(0, 0, core, core)),
        TileRegion(1, 0, 0, (core, core), box(core / 2, core / 2, 1.5 * core, 1.5 * core), 1),
    )
    prior = np.tile([25.0, -10.0], (2, 1))
    matcher = BatchTranslationMatcher()
    result = PairProcessor(config, matcher)._refine_tile_shifts(
        _pair(tmp_path), regions, prior, None, None)
    assert result.matcher_tiles == 1
    assert result.counts["coarse_secondary_refined"] == 1
    np.testing.assert_allclose(result.shifts, prior + [100, 0])


@pytest.mark.parametrize("sampling", ["per_tile", "pair_cache"])
def test_layered_pair_keeps_every_primary_node(tmp_path, sampling):
    config = _config(tmp_path)
    base = PairProcessor(config, TranslationMatcher()).process(_pair(tmp_path))
    layered_config = _layered(
        config, tile_grid_offsets=DIAGONAL, masked_rematch_target_valid_fraction=0.5,
        tile_sampling=sampling,
    )
    layered = PairProcessor(layered_config, TranslationMatcher()).process(_pair(tmp_path))
    assert layered.field.available[base.field.available].all()
    assert layered.diagnostics["layout_secondary_tiles"] > 0
    assert layered.diagnostics["layout_nodes"] >= layered.diagnostics["layout_primary_nodes"]
    assert layered.diagnostics["layout_dedup_removed"] > 0
    np.testing.assert_allclose(
        layered.field.displacement_m[layered.field.available] - [100.0, 0.0], 0.0, atol=1e-8
    )
