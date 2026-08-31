import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "experiments" / "buoy_descriptor_benchmark.py"
    spec = importlib.util.spec_from_file_location("buoy_descriptor_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_interpolate_xy_at_time_uses_consistent_timestamp_units():
    benchmark = load_benchmark_module()
    track = pd.DataFrame(
        {
            "timestamp": [
                "2026-06-01T00:00:00Z",
                "2026-06-01T01:00:00Z",
                "2026-06-01T02:00:00Z",
            ],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, -50.0, -100.0],
        }
    )

    x, y = benchmark.interpolate_xy_at_time(track, "2026-06-01T00:30:00Z")

    assert x == 50.0
    assert y == -25.0


def test_descriptor_distance_variants_have_expected_units():
    benchmark = load_benchmark_module()
    source = np.array([0b00000000], dtype=np.uint8)
    candidates = np.array([[0b00000000], [0b00000001], [0b00000101]], dtype=np.uint8)

    assert benchmark.descriptor_distances(source, candidates, "hamming").tolist() == [0.0, 1.0, 2.0]
    assert benchmark.descriptor_distances(source, candidates, "hamming2").tolist() == [0.0, 1.0, 2.0]


def test_physics_gate_scales_candidate_radius_without_changing_ranking():
    benchmark = load_benchmark_module()
    grid = benchmark.CandidateGrid(
        pixel_xy=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        map_xy=np.array([[0.0, 0.0], [1000.0, 0.0], [10000.0, 0.0]]),
    )
    distances = np.array([5.0, 2.0, 0.0])
    source = np.array([0.0, 0.0])
    target = np.array([1000.0, 0.0])

    global_result = benchmark.rank_and_retrieve(
        distances,
        grid,
        benchmark.select_gate(grid.map_xy, source, None),
        source,
        target,
        truth_distance=2.0,
    )
    gated_result = benchmark.rank_and_retrieve(
        distances,
        grid,
        benchmark.select_gate(grid.map_xy, source, 2000.0),
        source,
        target,
        truth_distance=2.0,
    )

    assert global_result["endpoint_error_m"] == 9000.0
    assert gated_result["endpoint_error_m"] == 0.0
    assert gated_result["candidate_count"] == 2


def test_interpolation_rejects_extrapolation():
    benchmark = load_benchmark_module()
    track = pd.DataFrame(
        {
            "timestamp": ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z"],
            "x": [0.0, 100.0],
            "y": [0.0, 0.0],
        }
    )

    with np.testing.assert_raises_regex(ValueError, "outside the buoy track"):
        benchmark.interpolate_xy_at_time(track, "2026-06-01T01:00:01Z")


def test_pair_builder_does_not_join_explicit_trajectory_segments():
    benchmark = load_benchmark_module()
    coincidences = pd.DataFrame(
        {
            "buoy_id": ["same", "same", "same", "same"],
            "experiment_trajectory_id": ["first", "first", "second", "second"],
            "image_time": pd.to_datetime(
                [
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T01:00:00Z",
                    "2020-03-01T00:00:00Z",
                    "2020-03-01T01:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    pairs = benchmark.build_pairs(coincidences)

    assert pairs["trajectory_id"].tolist() == ["first", "second"]
    assert len(pairs) == 2
