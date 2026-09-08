from dataclasses import replace

import numpy as np
import pytest
from shapely import box

from limosat import MotionMatches, RoutingConfig
from limosat.pairs import PairProcessor, TileRegion
from limosat.routing import coarse_match_shift
from test_efficientloftr_fields import EmptyMatcher, TranslationMatcher, _config, _pair


class BatchTranslationMatcher(TranslationMatcher):
    def __init__(self):
        self.batch_sizes = []

    def match_batch(self, sources, targets):
        self.batch_sizes.append(len(sources))
        return [self.match(source, target)
                for source, target in zip(sources, targets, strict=True)]


def _matches(source, displacement):
    source = np.asarray(source, dtype=np.float64)
    return MotionMatches(
        source, source + displacement, np.ones(len(source)),
        np.zeros(len(source), dtype=np.int32), np.zeros(len(source), dtype=np.int32),
    )


def test_local_median_excludes_distant_and_unphysical_matches():
    source = [[0, 0], [10, 0], [0, 10], [10, 10], [500, 0], [0, 0]]
    displacement = np.array([[40, -20], [40, -20], [40, -20], [70, 20],
                             [-80, 0], [1000, 0]])
    shift, reason = coarse_match_shift(_matches(source, displacement), (0, 0), 100, 4, 100)
    np.testing.assert_array_equal(shift, [40, -20])
    assert reason == 'refined'
    shift, reason = coarse_match_shift(_matches(source, displacement), (0, 0), 100, 5, 100)
    assert shift is None
    assert reason == 'insufficient_support'


@pytest.mark.parametrize('batch_size', [1, 3])
def test_coarse_pixel_translation_positions_fine_tiles_in_metres(tmp_path, batch_size):
    config = _config(tmp_path)
    config = replace(
        config,
        matcher=replace(config.matcher, tile_batch_size=batch_size),
        routing=replace(config.routing, coarse_matching=True, coarse_pixel_size_m=200,
                        coarse_support_radius_m=1000, coarse_minimum_matches=4),
    )
    pair = _pair(tmp_path)
    regions = tuple(TileRegion(i, 0, i, (float(i * 100), 0.0), box(-600, -600, 600, 600))
                    for i in range(4))
    matcher = BatchTranslationMatcher()
    processor = PairProcessor(config, matcher)
    prior = np.column_stack((np.arange(4) * 50.0, np.zeros(4)))
    original_prior = prior.copy()
    refined = processor._refine_tile_shifts(pair, regions, prior, None, None)
    # One coarse pixel is 200 metres; the following fine pixel is 100 metres.
    np.testing.assert_allclose(refined.shifts, prior + [200, 0])
    np.testing.assert_array_equal(prior, original_prior)
    assert matcher.batch_sizes == ([1, 1, 1, 1] if batch_size == 1 else [3, 1])
    assert refined.counts['coarse_tiles_refined'] == 4
    assert refined.matcher_tiles == 4
    result = processor.process(pair)
    assert len(result.matches) > 0
    np.testing.assert_allclose(result.matches.displacement_m,
                               np.tile([300, 0], (len(result.matches), 1)))
    # No 200 m coarse observations enter the scientific product.
    assert result.diagnostics['coarse_tiles_refined'] > 0


def test_missing_coarse_support_preserves_nonzero_prior(tmp_path):
    config = _config(tmp_path)
    processor = PairProcessor(config, EmptyMatcher())
    region = TileRegion(0, 0, 0, (0, 0), box(-600, -600, 600, 600))
    prior = np.array([[100.0, -50.0]])
    result = processor._refine_tile_shifts(_pair(tmp_path), (region,), prior, None, None)
    np.testing.assert_array_equal(result.shifts, prior)
    assert result.counts['coarse_fallback_insufficient_support'] == 1


@pytest.mark.parametrize('settings', [
    {'coarse_pixel_size_m': 0}, {'coarse_pixel_size_m': float('nan')},
    {'coarse_support_radius_m': float('inf')}, {'coarse_minimum_matches': 0},
    {'coarse_minimum_matches': 1.5}, {'coarse_matching': 'yes'},
])
def test_invalid_coarse_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        RoutingConfig(**settings)
