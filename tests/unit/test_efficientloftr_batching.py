from __future__ import annotations

import math
import threading

import numpy as np
import pytest
from shapely import box

from limosat import MatcherConfig, RoutingConfig, RunConfig
from limosat.efficientloftr import EfficientLoFTR
from limosat.pairs import PairProcessor, TileRegion

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


class PrefetchObservingMatcher(BatchStationaryMatcher):
    def __init__(self, second_batch_ready, sample_count, sample_lock) -> None:
        super().__init__()
        self.second_batch_ready = second_batch_ready
        self.sample_count = sample_count
        self.sample_lock = sample_lock
        self.sample_count_at_first_match = None
        self.thread_ids = []

    def match_batch(self, sources, targets):
        self.thread_ids.append(threading.get_ident())
        if not self.batch_sizes:
            assert self.second_batch_ready.wait(timeout=1.0)
            with self.sample_lock:
                self.sample_count_at_first_match = self.sample_count[0]
        return super().match_batch(sources, targets)


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


def test_pair_processor_prefetches_exactly_one_ordered_batch(tmp_path, monkeypatch):
    config = _config(tmp_path)
    pair = _pair(tmp_path)
    second_batch_ready = threading.Event()
    sample_lock = threading.Lock()
    sample_count = [0]
    sampled_centers = []
    sampling_thread_ids = []
    matcher = PrefetchObservingMatcher(
        second_batch_ready, sample_count, sample_lock
    )
    processor = PairProcessor(config, matcher)

    def sample_pair(_pair, center, _shift):
        with sample_lock:
            sampled_centers.append(center)
            sampling_thread_ids.append(threading.get_ident())
            sample_count[0] += 1
            if sample_count[0] == 8:
                second_batch_ready.set()
        image = np.zeros((16, 16), dtype=np.uint8)
        valid = np.ones_like(image, dtype=bool)
        return (image, image.copy(), valid, valid.copy()), None

    monkeypatch.setattr(processor, "_sample_pair", sample_pair)
    regions = tuple(
        TileRegion(index, 0, index, (float(index * 100), 0.0), box(0, 0, 1, 1))
        for index in range(12)
    )
    centers = np.asarray([item.center_xy_m for item in regions])
    main_thread = threading.get_ident()

    result = processor._process_hypothesis(  # noqa: SLF001
        pair,
        box(-1_000, -1_000, 2_000, 1_000),
        regions,
        centers,
        None,
        None,
        None,
        None,
        None,
    )

    assert matcher.batch_sizes == [4, 4, 4]
    assert matcher.sample_count_at_first_match == 8
    assert matcher.thread_ids == [main_thread] * 3
    assert all(thread_id != main_thread for thread_id in sampling_thread_ids)
    assert sampled_centers == [item.center_xy_m for item in regions]
    assert result.matcher_tiles == 12
