from __future__ import annotations

import math

import numpy as np
import pytest

from limosat import MatcherConfig, RoutingConfig, RunConfig
from limosat.efficientloftr import EfficientLoFTR
from limosat.pairs import PairProcessor

from test_efficientloftr_fields import StationaryMatcher, _config, _pair


class RecordingRunner:
    def __init__(self, torch) -> None:
        self.torch = torch
        self.shapes = []

    def __call__(self, image0, image1):
        self.shapes.append((tuple(image0.shape), tuple(image1.shape)))
        return {
            "m_bids": self.torch.tensor([0, 0, 1, 3]),
            "mkpts0_f": self.torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
            ),
            "mkpts1_f": self.torch.tensor(
                [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [8.0, 8.0]]
            ),
            "mconf": self.torch.tensor([21.0, 29.0, 25.0, 27.0]),
        }


class BatchStationaryMatcher:
    def __init__(self) -> None:
        self.batch_sizes = []

    def match_batch(self, sources, targets):
        self.batch_sizes.append(len(sources))
        matcher = StationaryMatcher()
        return [
            matcher.match(source, target)
            for source, target in zip(sources, targets, strict=True)
        ]


def test_match_batch_pads_to_static_shape_and_splits_real_tiles():
    torch = pytest.importorskip("torch")
    matcher = object.__new__(EfficientLoFTR)
    matcher.config = MatcherConfig(
        tile_size_px=2,
        tile_margin_px=0,
        endpoint_support_radius_px=0,
        transform_grid_spacing_px=1,
        tile_batch_size=4,
        prefix_cuda_graph=False,
    )
    matcher.device = torch.device("cpu")
    matcher._runner = RecordingRunner(torch)

    results = matcher.match_batch(
        (np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)),
        (np.ones((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)),
    )

    assert matcher._runner.shapes == [((4, 1, 2, 2), (4, 1, 2, 2))]
    assert len(results) == 2
    np.testing.assert_array_equal(results[0][0], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(results[0][2], [0.1, 0.9])
    np.testing.assert_array_equal(results[1][0], [[5.0, 6.0]])
    np.testing.assert_allclose(results[1][2], [0.5])


def test_pair_processor_batches_tiles_without_changing_products(tmp_path):
    config = _config(tmp_path)
    pair = _pair(tmp_path)
    reference = PairProcessor(config, StationaryMatcher()).process(pair)
    matcher = BatchStationaryMatcher()

    candidate = PairProcessor(config, matcher).process(pair)

    assert sum(matcher.batch_sizes) == candidate.diagnostics[
        "matcher_tile_evaluations"
    ]
    assert all(size <= config.matcher.tile_batch_size for size in matcher.batch_sizes)
    assert candidate.matcher_calls == len(matcher.batch_sizes)
    assert candidate.matcher_calls == math.ceil(
        candidate.diagnostics["matcher_tile_evaluations"]
        / config.matcher.tile_batch_size
    )
    np.testing.assert_array_equal(
        candidate.matches.source_xy_m, reference.matches.source_xy_m
    )
    np.testing.assert_array_equal(
        candidate.matches.target_xy_m, reference.matches.target_xy_m
    )
    np.testing.assert_array_equal(candidate.matches.score, reference.matches.score)
    np.testing.assert_array_equal(candidate.field.available, reference.field.available)
    np.testing.assert_array_equal(
        candidate.field.displacement_m, reference.field.displacement_m
    )


def test_pair_processor_batches_residual_recovery_tiles(tmp_path, monkeypatch):
    base = _config(tmp_path)
    config = RunConfig(
        **{
            **base.__dict__,
            "routing": RoutingConfig(
                initial="same_center", residual_edge_recovery=True
            ),
        }
    )
    monkeypatch.setattr(
        "limosat.pairs.residual_edge_correction",
        lambda *_args, **_kwargs: np.array([100.0, 0.0]),
    )
    matcher = BatchStationaryMatcher()

    result = PairProcessor(config, matcher).process(_pair(tmp_path))

    planned = result.diagnostics["planned_tiles"]
    assert result.diagnostics["matcher_tile_evaluations"] == 2 * planned
    assert result.matcher_calls == 2 * math.ceil(
        planned / config.matcher.tile_batch_size
    )
    assert sum(matcher.batch_sizes) == 2 * planned
